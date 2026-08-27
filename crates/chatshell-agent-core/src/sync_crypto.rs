//! Transport encryption for cloud-synced data (ADR 04).
//!
//! Encrypts only what leaves the device — the snapshot artifact and the blob
//! sidecar bytes. Local databases stay plaintext (platform FDE covers them).
//!
//! Key hierarchy: a random 256-bit content key (CK) per sync group,
//! independent of the provider-credential master key so the Keychain's
//! failure radius can never take chat readability down with it (ADR 04 §2).
//! Domain-separated subkeys derive from the CK:
//!
//! - `k_snapshot     = HKDF(CK, "chatshell/sync/snapshot-v1")` — chunked
//!   AEAD over the zstd-compressed snapshot.
//! - `k_blob_single  = HKDF(CK, "chatshell/sync/blob-single-v1")` —
//!   single-shot AEAD for blobs up to [`BLOB_STREAM_THRESHOLD`], with the
//!   per-blob nonce derived from the content digest.
//! - `k_blob_stream  = HKDF(CK, "chatshell/sync/blob-stream-v1")` —
//!   chunked AEAD for larger blobs. The single/stream split keeps the
//!   deterministic per-blob nonce space and the streaming counter space on
//!   different keys.
//!
//! Chunked AEAD is Tink-style counter chunking over the existing `aes-gcm`
//! dependency: the per-stream AES key is derived via HKDF from the parent key
//! plus a stream id (the blob digest, or the per-publish random header salt
//! for snapshots), each chunk's nonce is its counter, and every chunk's AAD
//! binds `(content id, key_version, chunk_index, final-chunk flag)` so
//! truncation, reorder, substitution, and cross-record splices all fail the
//! tag check.
//!
//! The artifact carries an age-style multi-slot plaintext header ahead of the
//! ciphertext payload so `plan_sync` can run the version compare without
//! touching the payload. Invariant: **no slot carries key material
//! recoverable from the artifact alone** — the Apple slot is a pure type
//! marker (the CK itself lives in a dedicated iCloud Keychain synchronizable
//! item), and every Argon2id slot wraps the CK under the group passphrase.
//!
//! Content ids are canonicalized once: for blobs, the ASCII bytes of the bare
//! lowercase hex digest (exactly the filename under `blobs/`); for the
//! snapshot payload, the ASCII constant `snapshot-v1`.

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use argon2::Argon2;
use hkdf::Hkdf;
use rand::RngCore;
use sha2::Sha256;

/// Length of a content key (and of every derived AEAD key).
pub const CK_LEN: usize = 32;
/// GCM nonce length.
const NONCE_LEN: usize = 12;
/// GCM tag length.
const TAG_LEN: usize = 16;

/// Magic of the encrypted snapshot artifact (`snapshot.db.enc`).
pub const ARTIFACT_MAGIC: &[u8; 8] = b"CHSNPENC";
/// Magic of an encrypted blob sidecar file.
pub const BLOB_MAGIC: &[u8; 8] = b"CHBLOBEC";
/// Wire format version of artifact/blob headers written by this module.
pub const FORMAT_VERSION: u16 = 1;

/// Canonical content id of the snapshot payload.
pub const SNAPSHOT_CONTENT_ID: &[u8] = b"snapshot-v1";

/// Blobs at or below this size use single-shot AEAD; larger blobs stream
/// through chunked AEAD under `k_blob_stream` (ADR 04 §5).
pub const BLOB_STREAM_THRESHOLD: usize = 4 * 1024 * 1024;

/// Plaintext bytes per chunk (ciphertext is the same length; the tag is
/// carried in the frame header). A 256 MB blob is 4096 chunks with 0.02%
/// tag overhead; the u32 counter ceilings at ~256 TiB — far past the cap.
pub const CHUNK_PLAINTEXT: usize = 64 * 1024;

/// Production Argon2id cost (~64 MiB, sub-second on current hardware per
/// ADR 04 §3). Tests use weaker parameters — the API takes them explicitly.
pub const DEFAULT_ARGON_PARAMS: ArgonParams = ArgonParams {
    m_cost: 64 * 1024,
    t_cost: 3,
    p_cost: 1,
};

/// Salt length for Argon2id slots and for the snapshot payload stream id.
const SALT_LEN: usize = 16;

/// Upper sanity bounds while parsing (defensive against corrupt headers).
const MAX_SYNC_VERSION_LEN: usize = 512;
const MAX_SLOTS: usize = 64;
const MAX_WRAPPED_LEN: usize = 4096;

/// Errors from transport encryption. Every failure is safe to surface: no
/// path silently falls back to treating data as plaintext.
#[derive(Debug, thiserror::Error)]
pub enum SyncCryptoError {
    #[error("artifact is corrupt or foreign: {0}")]
    Corrupt(String),
    #[error("unsupported transport format version {0}")]
    UnsupportedFormat(u16),
    #[error("content key version {0} is not held on this device")]
    UnknownKeyVersion(u32),
    #[error("decryption failed (wrong key or tampered data)")]
    Auth,
    #[error("content hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
    #[error("malformed content-key chain: {0}")]
    BadChain(String),
    #[error("wrong passphrase")]
    WrongPassphrase,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = SyncCryptoError> = std::result::Result<T, E>;

// ---------------------------------------------------------------------------
// Byte cursor
// ---------------------------------------------------------------------------

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Writer { buf: Vec::new() }
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }
    fn into_vec(self) -> Vec<u8> {
        self.buf
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    fn u8(&mut self) -> Result<u8> {
        self.take(1).map(|b| b[0])
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(SyncCryptoError::Corrupt("unexpected end of data".into()));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let b = self.take(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(b);
        Ok(out)
    }

    /// The unread tail of the buffer.
    fn rest(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }
}

// ---------------------------------------------------------------------------
// Subkeys and nonce derivation
// ---------------------------------------------------------------------------

/// HKDF-SHA256 expand with `ikm` as the PRK (extract was already done over
/// the CK when `ikm` is a derived subkey; for direct use the caller passes a
/// domain-separated label in `info`).
fn hkdf_expand(ikm: &[u8], info: &[u8], out: &mut [u8]) {
    let hk = Hkdf::<Sha256>::from_prk(ikm).expect("PRK length is 32 bytes");
    hk.expand(info, out).expect("output length is valid");
}

fn subkey(ck: &[u8; CK_LEN], label: &[u8]) -> [u8; CK_LEN] {
    let mut out = [0u8; CK_LEN];
    hkdf_expand(ck, label, &mut out);
    out
}

/// Domain-separated subkeys (ADR 04 §2).
pub fn derive_snapshot_key(ck: &[u8; CK_LEN]) -> [u8; CK_LEN] {
    subkey(ck, b"chatshell/sync/snapshot-v1")
}
pub fn derive_blob_single_key(ck: &[u8; CK_LEN]) -> [u8; CK_LEN] {
    subkey(ck, b"chatshell/sync/blob-single-v1")
}
pub fn derive_blob_stream_key(ck: &[u8; CK_LEN]) -> [u8; CK_LEN] {
    subkey(ck, b"chatshell/sync/blob-stream-v1")
}

/// Per-stream AES key: HKDF over the parent subkey with the stream id as
/// info. Distinct streams (different blobs, or different snapshot publishes
/// via the random header salt) get distinct AES keys, so the counter-derived
/// chunk nonces never repeat across streams under one parent key.
fn stream_key(parent: &[u8; CK_LEN], stream_id: &[u8]) -> [u8; CK_LEN] {
    let mut info = Vec::with_capacity(stream_id.len() + 32);
    info.extend_from_slice(b"chatshell/sync/stream-key/v1");
    info.extend_from_slice(stream_id);
    subkey(parent, &info)
}

/// Deterministic single-shot blob nonce: HKDF over `k_blob_single` with the
/// digest as info. Safe under GCM because the filename binds the plaintext —
/// a reused nonce always pairs with identical bytes (ADR 04 §5).
fn blob_nonce(k_blob_single: &[u8; CK_LEN], digest_ascii: &[u8]) -> [u8; NONCE_LEN] {
    let mut info = Vec::with_capacity(digest_ascii.len() + 32);
    info.extend_from_slice(b"chatshell/sync/blob-nonce/v1");
    info.extend_from_slice(digest_ascii);
    let mut out = [0u8; NONCE_LEN];
    hkdf_expand(k_blob_single, &info, &mut out);
    out
}

/// Tink-style counter nonce: eight zero bytes + big-endian chunk counter.
fn chunk_nonce(index: u32) -> [u8; NONCE_LEN] {
    let mut out = [0u8; NONCE_LEN];
    out[8..].copy_from_slice(&index.to_be_bytes());
    out
}

/// AAD binding every chunk: domain tag, length-prefixed content id,
/// key_version, chunk index, final-chunk flag. A truncation bug that drops
/// or reorders chunks fails the tag; a splice between records fails on the
/// content id.
fn chunk_aad(content_id: &[u8], key_version: u32, chunk_index: u32, final_chunk: bool) -> Vec<u8> {
    let mut w = Writer::new();
    w.bytes(b"chatshell/sync/chunk/v1");
    w.u16(content_id.len() as u16);
    w.bytes(content_id);
    w.u32(key_version);
    w.u32(chunk_index);
    w.u8(final_chunk as u8);
    w.into_vec()
}

fn seal(
    key: &[u8; CK_LEN],
    nonce: &[u8; NONCE_LEN],
    plaintext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    cipher
        .encrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| SyncCryptoError::Auth)
}

fn open(
    key: &[u8; CK_LEN],
    nonce: &[u8; NONCE_LEN],
    ciphertext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    cipher
        .decrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad,
            },
        )
        .map_err(|_| SyncCryptoError::Auth)
}

// ---------------------------------------------------------------------------
// Chunked AEAD
// ---------------------------------------------------------------------------

/// Chunked AEAD seal. Frame layout: `[ct_len u32][ciphertext][tag 16]`; the
/// last frame (and only it) carries `final = true` in its AAD. An exact
/// multiple of [`CHUNK_PLAINTEXT`] ends on a full final chunk; empty input
/// produces a single empty final chunk.
pub fn seal_chunks(
    parent: &[u8; CK_LEN],
    stream_id: &[u8],
    content_id: &[u8],
    key_version: u32,
    plaintext: &[u8],
) -> Result<Vec<u8>> {
    let key = stream_key(parent, stream_id);
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key));
    let mut out = Vec::with_capacity(plaintext.len() + plaintext.len() / CHUNK_PLAINTEXT * 24 + 32);
    let mut offset = 0usize;
    let mut index: u32 = 0;
    loop {
        let end = (offset + CHUNK_PLAINTEXT).min(plaintext.len());
        let chunk = &plaintext[offset..end];
        let final_chunk = end == plaintext.len();
        let sealed = cipher
            .encrypt(
                Nonce::from_slice(&chunk_nonce(index)),
                Payload {
                    msg: chunk,
                    aad: &chunk_aad(content_id, key_version, index, final_chunk),
                },
            )
            .map_err(|_| SyncCryptoError::Auth)?;
        let mut w = Writer::new();
        w.u32(sealed.len() as u32);
        out.extend_from_slice(&w.into_vec());
        out.extend_from_slice(&sealed);
        if final_chunk {
            break;
        }
        offset = end;
        index = index.checked_add(1).expect("chunk counter overflow");
    }
    Ok(out)
}

/// Chunked AEAD open — inverse of [`seal_chunks`]. Fails on truncation
/// (the surviving last frame then claims `final = false`), reordering
/// (chunk index in AAD), substitution (tag), and cross-record splices
/// (content id in AAD).
pub fn open_chunks(
    parent: &[u8; CK_LEN],
    stream_id: &[u8],
    content_id: &[u8],
    key_version: u32,
    ciphertext: &[u8],
) -> Result<Vec<u8>> {
    let key = stream_key(parent, stream_id);
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key));
    let mut out = Vec::with_capacity(ciphertext.len());
    let mut r = Reader::new(ciphertext);
    let mut index: u32 = 0;
    let mut saw_any = false;
    let mut last_plain: Option<Vec<u8>> = None;
    // Frames are decrypted one behind: only when the NEXT frame exists do we
    // know the current one was not final.
    while r.remaining() > 0 {
        let len = r.u32()? as usize;
        if !(TAG_LEN..=CHUNK_PLAINTEXT + TAG_LEN).contains(&len) {
            return Err(SyncCryptoError::Corrupt(
                "chunk length out of bounds".into(),
            ));
        }
        let sealed = r.take(len)?;
        let at_end = r.remaining() == 0;
        let plain = cipher
            .decrypt(
                Nonce::from_slice(&chunk_nonce(index)),
                Payload {
                    msg: sealed,
                    aad: &chunk_aad(content_id, key_version, index, at_end),
                },
            )
            .map_err(|_| SyncCryptoError::Auth)?;
        if let Some(prev) = last_plain.take() {
            out.extend_from_slice(&prev);
        }
        last_plain = Some(plain);
        saw_any = true;
        index = index.checked_add(1).expect("chunk counter overflow");
    }
    if !saw_any {
        return Err(SyncCryptoError::Corrupt("empty chunk stream".into()));
    }
    if let Some(prev) = last_plain.take() {
        out.extend_from_slice(&prev);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Content keys and the rotation chain
// ---------------------------------------------------------------------------

/// Argon2id cost parameters, carried inside every slot so future cost bumps
/// decode old artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArgonParams {
    /// Memory cost in KiB.
    pub m_cost: u32,
    /// Time cost (passes).
    pub t_cost: u32,
    /// Parallelism (lanes).
    pub p_cost: u32,
}

impl ArgonParams {
    /// Cheap parameters for tests only — never use in production paths.
    pub fn test() -> Self {
        ArgonParams {
            m_cost: 8,
            t_cost: 1,
            p_cost: 1,
        }
    }
}

/// One Argon2id-wrapped CK: `wrapped = nonce ‖ ct ‖ tag` of the 32-byte CK
/// under `AES-GCM(Argon2id(passphrase, salt, params))`, with a domain
/// constant as AAD.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgonSlot {
    pub salt: [u8; SALT_LEN],
    pub params: ArgonParams,
    pub wrapped: Vec<u8>,
}

impl ArgonSlot {
    fn encode_slot(&self, w: &mut Writer) {
        w.bytes(&self.salt);
        w.u32(self.params.m_cost);
        w.u32(self.params.t_cost);
        w.u32(self.params.p_cost);
        w.u16(self.wrapped.len() as u16);
        w.bytes(&self.wrapped);
    }

    fn decode_slot(r: &mut Reader) -> Result<Self> {
        let salt = r.array::<SALT_LEN>()?;
        let params = ArgonParams {
            m_cost: r.u32()?,
            t_cost: r.u32()?,
            p_cost: r.u32()?,
        };
        let len = r.u16()? as usize;
        if len == 0 || len > MAX_WRAPPED_LEN {
            return Err(SyncCryptoError::Corrupt(
                "slot payload length out of bounds".into(),
            ));
        }
        let wrapped = r.take(len)?.to_vec();
        Ok(ArgonSlot {
            salt,
            params,
            wrapped,
        })
    }
}

const CK_SLOT_AAD: &[u8] = b"chatshell/sync/ck-slot/v1";

fn argon2_derive(passphrase: &str, salt: &[u8], params: ArgonParams) -> Result<[u8; CK_LEN]> {
    let params = argon2::Params::new(params.m_cost, params.t_cost, params.p_cost, Some(CK_LEN))
        .map_err(|e| SyncCryptoError::Corrupt(format!("argon2 params: {e}")))?;
    let argon = Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
    let mut okm = [0u8; CK_LEN];
    argon
        .hash_password_into(passphrase.as_bytes(), salt, &mut okm)
        .map_err(|e| SyncCryptoError::Corrupt(format!("argon2: {e}")))?;
    Ok(okm)
}

/// Wrap a CK under the passphrase. A fresh random salt makes every wrap
/// unique; the params ride along in the slot.
pub fn wrap_ck(ck: &[u8; CK_LEN], passphrase: &str, params: ArgonParams) -> Result<ArgonSlot> {
    let mut salt = [0u8; SALT_LEN];
    rand::rngs::OsRng.fill_bytes(&mut salt);
    let kek = argon2_derive(passphrase, &salt, params)?;
    let mut nonce = [0u8; NONCE_LEN];
    rand::rngs::OsRng.fill_bytes(&mut nonce);
    let mut wrapped = nonce.to_vec();
    wrapped.extend(seal(&kek, &nonce, ck, CK_SLOT_AAD)?);
    Ok(ArgonSlot {
        salt,
        params,
        wrapped,
    })
}

/// Unwrap a CK from a slot. Any failure means wrong passphrase or tampering
/// — a deterministic, locally rate-limitable outcome (ADR 04 §3 rule 3).
pub fn unwrap_ck(slot: &ArgonSlot, passphrase: &str) -> Result<[u8; CK_LEN]> {
    let kek = argon2_derive(passphrase, &slot.salt, slot.params)?;
    if slot.wrapped.len() != NONCE_LEN + CK_LEN + TAG_LEN {
        return Err(SyncCryptoError::Corrupt(
            "slot payload has wrong length".into(),
        ));
    }
    let plain = open(
        &kek,
        &slot.wrapped[..NONCE_LEN].try_into().expect("nonce"),
        &slot.wrapped[NONCE_LEN..],
        CK_SLOT_AAD,
    )?;
    plain.try_into().map_err(|_| SyncCryptoError::Auth)
}

/// The CK chain held on this device. Versions are strictly ascending and
/// unique; a partial chain (only the current key, adopted from the Keychain
/// item) is legal — old blobs then need the passphrase ladder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentKeys {
    entries: Vec<(u32, [u8; CK_LEN])>,
}

impl ContentKeys {
    /// Mint a fresh group: one random CK at version 1.
    pub fn bootstrap() -> Self {
        ContentKeys {
            entries: vec![(1, random_ck())],
        }
    }

    /// Build from explicit material (Keychain adoption). `entries` must be
    /// non-empty with strictly ascending unique versions.
    pub fn from_entries(entries: Vec<(u32, [u8; CK_LEN])>) -> Result<Self> {
        if entries.is_empty() {
            return Err(SyncCryptoError::BadChain("empty key chain".into()));
        }
        for pair in entries.windows(2) {
            if pair[0].0 >= pair[1].0 {
                return Err(SyncCryptoError::BadChain(
                    "versions must be strictly ascending".into(),
                ));
            }
        }
        Ok(ContentKeys { entries })
    }

    /// Highest version — used for every new write.
    pub fn current_version(&self) -> u32 {
        self.entries.last().map(|(v, _)| *v).unwrap_or(0)
    }

    pub fn current_key(&self) -> &[u8; CK_LEN] {
        &self.entries.last().expect("chain is never empty").1
    }

    pub fn key_for_version(&self, version: u32) -> Option<&[u8; CK_LEN]> {
        self.entries
            .iter()
            .find(|(v, _)| *v == version)
            .map(|(_, k)| k)
    }

    /// Union in keys we lack (ascending, deduped). Used when unwrapping the
    /// full slot chain merges with a partial local chain.
    pub fn merge_from(&mut self, other: &ContentKeys) {
        let mut merged = self.entries.clone();
        for (v, k) in &other.entries {
            if !merged.iter().any(|(lv, _)| lv == v) {
                merged.push((*v, *k));
            }
        }
        merged.sort_by_key(|(v, _)| *v);
        if !merged.is_empty() {
            self.entries = merged;
        }
    }

    /// Rotate: mint the next CK (explicit user action only — suspected
    /// compromise, lost device; ADR 04 §5). Old keys stay for decrypt-only.
    pub fn rotate(&mut self) -> [u8; CK_LEN] {
        let next = self.current_version() + 1;
        let ck = random_ck();
        self.entries.push((next, ck));
        ck
    }

    // -- cache serialization --------------------------------------------

    const CHAIN_MAGIC: &[u8; 10] = b"CHCKCHAIN\x01";

    /// Serialize for the device-local cache file (kept beside the plaintext
    /// DB — explicitly permitted by ADR 04 §2; the cache protects only the
    /// cloud copy).
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.bytes(Self::CHAIN_MAGIC);
        w.u16(self.entries.len() as u16);
        for (v, k) in &self.entries {
            w.u32(*v);
            w.bytes(k);
        }
        w.into_vec()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let magic = r.take(Self::CHAIN_MAGIC.len())?;
        if magic != Self::CHAIN_MAGIC {
            return Err(SyncCryptoError::BadChain("bad cache magic".into()));
        }
        let count = r.u16()? as usize;
        if count == 0 || count > MAX_SLOTS {
            return Err(SyncCryptoError::BadChain("slot count out of bounds".into()));
        }
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let v = r.u32()?;
            let k = r.array::<CK_LEN>()?;
            entries.push((v, k));
        }
        Self::from_entries(entries)
    }
}

fn random_ck() -> [u8; CK_LEN] {
    let mut ck = [0u8; CK_LEN];
    rand::rngs::OsRng.fill_bytes(&mut ck);
    ck
}

// ---------------------------------------------------------------------------
// Apple-slot synchronizable item (ADR 04 §3, hardened by ADR 05)
// ---------------------------------------------------------------------------

/// Value format of the Apple-slot synchronizable item:
/// `v<key version>:<base64(32-byte CK)>`.
///
/// The version rides along so the adoption rung can reject a stale item
/// WITHOUT a decrypt attempt (ADR 05): after a rotation the artifact header
/// declares a higher key version than the frozen item carries, and rung 1
/// must fall through to the cached-chain rung instead of installing a
/// wrong-key engine (which would bounce already-unlocked devices through
/// the needs-unlock latch and delete their good cached chains).
///
/// Legacy bare-base64 values (pre-ADR-05) are rejected by
/// [`parse_synchronizable_item`] — the no-compat window applies; a device
/// holding one degrades to a single passphrase unlock, after which the
/// item is rewritten in this format (version-1 groups only).
pub fn encode_synchronizable_item(version: u32, key: &[u8; CK_LEN]) -> String {
    use base64::{Engine as _, engine::general_purpose};
    format!("v{}:{}", version, general_purpose::STANDARD.encode(key))
}

/// Parse a synchronizable-item value written by
/// [`encode_synchronizable_item`]. `None` for anything else: legacy
/// unversioned values, wrong lengths, malformed prefixes.
pub fn parse_synchronizable_item(value: &str) -> Option<(u32, [u8; CK_LEN])> {
    use base64::{Engine as _, engine::general_purpose};
    let (version_part, key_part) = value.split_once(':')?;
    let version = version_part.strip_prefix('v')?.parse().ok()?;
    let bytes = general_purpose::STANDARD.decode(key_part).ok()?;
    Some((version, bytes.try_into().ok()?))
}

// ---------------------------------------------------------------------------
// Artifact header (age-style multi-slot)
// ---------------------------------------------------------------------------

/// Parsed plaintext header of the encrypted snapshot artifact.
///
/// Slot ordering invariant: `slots[0]` wraps the current CK
/// (`key_version`), `slots[i]` wraps version `key_version - i`. Devices
/// that never learned the passphrase carry the slot bytes forward verbatim
/// from a previously seen artifact header — the slots are public ciphertext
/// and every publish must keep the fallback path alive for passphrase-only
/// devices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactHeader {
    /// `<counter>:<blake3_hex(device_id)>` — device ids are hashed out of
    /// the header so the provider sees no device census or activity
    /// timeline (ADR 04 §4).
    pub sync_version: String,
    /// Canonical schema version of the publishing database (diagnostic
    /// provenance, mirrors the in-DB `meta.schema_version` stamp).
    pub schema_version: u64,
    /// CK version the payload is sealed under.
    pub key_version: u32,
    /// Per-publish random stream id for the payload's chunk nonces.
    pub payload_salt: [u8; SALT_LEN],
    /// Whether the Apple marker slot is present (the CK itself lives in the
    /// iCloud Keychain synchronizable item, never in the artifact).
    pub icloud_slot: bool,
    /// Argon2id-wrapped CK chain, current first (see type docs).
    pub slots: Vec<ArgonSlot>,
}

impl ArtifactHeader {
    /// Serialize the header region (everything before the payload).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.bytes(ARTIFACT_MAGIC);
        w.u16(FORMAT_VERSION);
        w.u16(self.sync_version.len() as u16);
        w.bytes(self.sync_version.as_bytes());
        w.u64(self.schema_version);
        w.u32(self.key_version);
        w.bytes(&self.payload_salt);
        w.u8(self.icloud_slot as u8);
        w.u16(self.slots.len() as u16);
        for slot in &self.slots {
            w.bytes(&slot.salt);
            w.u32(slot.params.m_cost);
            w.u32(slot.params.t_cost);
            w.u32(slot.params.p_cost);
            w.u16(slot.wrapped.len() as u16);
            w.bytes(&slot.wrapped);
        }
        w.into_vec()
    }

    /// Parse a header from the front of `bytes`; returns it with the byte
    /// offset where the ciphertext payload starts.
    pub fn parse(bytes: &[u8]) -> Result<(ArtifactHeader, usize)> {
        let mut r = Reader::new(bytes);
        let magic = r.take(ARTIFACT_MAGIC.len())?;
        if magic != ARTIFACT_MAGIC {
            return Err(SyncCryptoError::Corrupt(format!(
                "unexpected magic {:02x?} — not an encrypted snapshot artifact",
                &magic[..4]
            )));
        }
        let format = r.u16()?;
        if format != FORMAT_VERSION {
            return Err(SyncCryptoError::UnsupportedFormat(format));
        }
        let sv_len = r.u16()? as usize;
        if sv_len == 0 || sv_len > MAX_SYNC_VERSION_LEN {
            return Err(SyncCryptoError::Corrupt(
                "sync_version length out of bounds".into(),
            ));
        }
        let sv_bytes = r.take(sv_len)?;
        let sync_version = std::str::from_utf8(sv_bytes)
            .map_err(|_| SyncCryptoError::Corrupt("sync_version is not UTF-8".into()))?
            .to_string();
        let schema_version = r.u64()?;
        let key_version = r.u32()?;
        let payload_salt = r.array::<SALT_LEN>()?;
        let icloud_slot = r.u8()? != 0;
        let slot_count = r.u16()? as usize;
        if slot_count == 0 || slot_count > MAX_SLOTS {
            return Err(SyncCryptoError::Corrupt("slot count out of bounds".into()));
        }
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            let salt = r.array::<SALT_LEN>()?;
            let m_cost = r.u32()?;
            let t_cost = r.u32()?;
            let p_cost = r.u32()?;
            let wrapped_len = r.u16()? as usize;
            if wrapped_len == 0 || wrapped_len > MAX_WRAPPED_LEN {
                return Err(SyncCryptoError::Corrupt(
                    "slot payload length out of bounds".into(),
                ));
            }
            let wrapped = r.take(wrapped_len)?.to_vec();
            slots.push(ArgonSlot {
                salt,
                params: ArgonParams {
                    m_cost,
                    t_cost,
                    p_cost,
                },
                wrapped,
            });
        }
        if key_version < slot_count as u32 {
            return Err(SyncCryptoError::Corrupt(
                "slot chain longer than the declared key version".into(),
            ));
        }
        Ok((
            ArtifactHeader {
                sync_version,
                schema_version,
                key_version,
                payload_salt,
                icloud_slot,
                slots,
            },
            r.pos,
        ))
    }
}

/// Read and parse the artifact header from a file without touching the
/// payload (the header-first `plan_sync` compare). Grows the read buffer
/// until the header parses or a sanity cap is hit.
pub fn read_artifact_header(path: &std::path::Path) -> Result<ArtifactHeader> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    loop {
        if buf.len() > 256 * 1024 {
            return Err(SyncCryptoError::Corrupt(
                "artifact header exceeds 256 KiB".into(),
            ));
        }
        match file.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) => return Err(SyncCryptoError::Io(e)),
        }
        if let Ok((header, _)) = ArtifactHeader::parse(&buf) {
            return Ok(header);
        }
    }
    let (header, _) = ArtifactHeader::parse(&buf)?;
    Ok(header)
}

// ---------------------------------------------------------------------------
// Snapshot payload + full artifact
// ---------------------------------------------------------------------------

/// Seal the zstd-compressed snapshot bytes as the artifact payload under the
/// chain's current key. The caller supplies the per-publish random salt that
/// also goes into the header.
pub fn encrypt_snapshot_payload(
    keys: &ContentKeys,
    salt: &[u8; SALT_LEN],
    compressed: &[u8],
) -> Result<Vec<u8>> {
    let parent = derive_snapshot_key(keys.current_key());
    seal_chunks(
        &parent,
        salt,
        SNAPSHOT_CONTENT_ID,
        keys.current_version(),
        compressed,
    )
}

/// Open the artifact payload. Uses the header's key version and salt, so a
/// device holding older keys (post-rotation) still decrypts.
pub fn decrypt_snapshot_payload(
    keys: &ContentKeys,
    header: &ArtifactHeader,
    payload: &[u8],
) -> Result<Vec<u8>> {
    let ck = keys
        .key_for_version(header.key_version)
        .ok_or(SyncCryptoError::UnknownKeyVersion(header.key_version))?;
    let parent = derive_snapshot_key(ck);
    open_chunks(
        &parent,
        &header.payload_salt,
        SNAPSHOT_CONTENT_ID,
        header.key_version,
        payload,
    )
}

/// Assemble the complete artifact bytes: header + payload.
pub fn build_artifact(header: &ArtifactHeader, payload: &[u8]) -> Vec<u8> {
    let mut out = header.to_bytes();
    out.extend_from_slice(payload);
    out
}

/// Split artifact bytes into `(header, payload_slice)`.
pub fn split_artifact(bytes: &[u8]) -> Result<(ArtifactHeader, &[u8])> {
    let (header, offset) = ArtifactHeader::parse(bytes)?;
    Ok((header, &bytes[offset..]))
}

// ---------------------------------------------------------------------------
// Blobs (ADR 02 Tier 2 carrier becomes ciphertext — ADR 04 §5)
// ---------------------------------------------------------------------------

/// Blob body kinds.
const BLOB_KIND_SINGLE: u8 = 1;
const BLOB_KIND_STREAM: u8 = 2;

/// Encrypt attachment bytes for the sidecar store.
///
/// **The pre-encrypt hash check is security-critical, not merely
/// consistency**: blob filenames are `blake3(plaintext)` and the nonce is
/// deterministic in that digest, so shipping different bytes under the same
/// name would be nonce reuse under the same key — catastrophic for GCM.
/// This function therefore asserts `blake3(plaintext) == digest_hex` before
/// encrypting, and the assertion is the single shared implementation point
/// for every upload path.
///
/// Deterministic by construction: concurrent uploads of the same attachment
/// from two devices produce byte-identical files (no iCloud conflict copies)
/// and uploads stay idempotent end-to-end.
pub fn encrypt_blob(keys: &ContentKeys, digest_hex: &str, plaintext: &[u8]) -> Result<Vec<u8>> {
    let actual = crate::crypto::blake3_hex(plaintext);
    if actual != digest_hex {
        return Err(SyncCryptoError::HashMismatch {
            expected: digest_hex.to_string(),
            actual,
        });
    }
    let version = keys.current_version();
    let ck = keys.current_key();
    let mut w = Writer::new();
    w.bytes(BLOB_MAGIC);
    w.u16(FORMAT_VERSION);
    if plaintext.len() <= BLOB_STREAM_THRESHOLD {
        let parent = derive_blob_single_key(ck);
        let nonce = blob_nonce(&parent, digest_hex.as_bytes());
        let sealed = seal(&parent, &nonce, plaintext, digest_hex.as_bytes())?;
        w.u8(BLOB_KIND_SINGLE);
        w.u32(version);
        w.bytes(&nonce);
        w.bytes(&sealed);
    } else {
        let parent = derive_blob_stream_key(ck);
        let chunks = seal_chunks(
            &parent,
            digest_hex.as_bytes(),
            digest_hex.as_bytes(),
            version,
            plaintext,
        )?;
        w.u8(BLOB_KIND_STREAM);
        w.u32(version);
        w.bytes(&chunks);
    }
    Ok(w.into_vec())
}

/// Decrypt a blob sidecar file. Enforces the magic (an unexpected magic
/// means corrupt or foreign: error, never a plaintext parse), resolves the
/// CK by the header's key version, and verifies `blake3(plaintext) ==
/// digest_hex` after decryption.
pub fn decrypt_blob(keys: &ContentKeys, digest_hex: &str, ciphertext: &[u8]) -> Result<Vec<u8>> {
    let mut r = Reader::new(ciphertext);
    let magic = r.take(BLOB_MAGIC.len())?;
    if magic != BLOB_MAGIC {
        return Err(SyncCryptoError::Corrupt(format!(
            "unexpected magic {:02x?} — not an encrypted blob",
            &magic[..4]
        )));
    }
    let format = r.u16()?;
    if format != FORMAT_VERSION {
        return Err(SyncCryptoError::UnsupportedFormat(format));
    }
    let kind = r.u8()?;
    let version = r.u32()?;
    let ck = keys
        .key_for_version(version)
        .ok_or(SyncCryptoError::UnknownKeyVersion(version))?
        .to_owned();
    let plain = match kind {
        BLOB_KIND_SINGLE => {
            let nonce = r.array::<NONCE_LEN>()?;
            let sealed = r.rest();
            let parent = derive_blob_single_key(&ck);
            open(&parent, &nonce, sealed, digest_hex.as_bytes())?
        }
        BLOB_KIND_STREAM => {
            let frames = r.rest();
            let parent = derive_blob_stream_key(&ck);
            open_chunks(
                &parent,
                digest_hex.as_bytes(),
                digest_hex.as_bytes(),
                version,
                frames,
            )?
        }
        other => {
            return Err(SyncCryptoError::Corrupt(format!(
                "unknown blob kind {other}"
            )));
        }
    };
    let actual = crate::crypto::blake3_hex(&plain);
    if actual != digest_hex {
        return Err(SyncCryptoError::HashMismatch {
            expected: digest_hex.to_string(),
            actual,
        });
    }
    Ok(plain)
}

// ---------------------------------------------------------------------------
// SyncCrypto: what the engine and the blob paths hold
// ---------------------------------------------------------------------------

/// Everything a device needs to publish and open transport artifacts:
/// the CK chain plus the Argon2id slot bytes to carry in headers.
///
/// Devices that adopted the CK from the Keychain item hold only the current
/// key but still carry the full slot chain forward from the last-seen
/// artifact header — those bytes are public ciphertext, and republishing
/// must keep the passphrase fallback alive for every peer.
#[derive(Debug, Clone)]
pub struct SyncCrypto {
    keys: ContentKeys,
    slots: Vec<ArgonSlot>,
    icloud_slot: bool,
}

impl SyncCrypto {
    /// First device of a sync group: mint CK v1, wrap it under the group
    /// passphrase. Bootstrap must set the passphrase even on Apple
    /// platforms (ADR 04 §3 rule 1) — otherwise the fallback does not exist
    /// when the Keychain slot fails.
    pub fn bootstrap(passphrase: &str) -> Result<Self> {
        Self::bootstrap_with_params(passphrase, DEFAULT_ARGON_PARAMS)
    }

    pub fn bootstrap_with_params(passphrase: &str, params: ArgonParams) -> Result<Self> {
        let keys = ContentKeys::bootstrap();
        let slot = wrap_ck(keys.current_key(), passphrase, params)?;
        Ok(SyncCrypto {
            keys,
            slots: vec![slot],
            icloud_slot: true,
        })
    }

    /// Adoption path: keys from the Keychain item / local cache (possibly
    /// partial — only the current version), slots carried from a previously
    /// seen artifact header. `slots` may be empty when no artifact exists
    /// yet (this device bootstraps a fresh group instead).
    pub fn from_keys(keys: ContentKeys, slots: Vec<ArgonSlot>, icloud_slot: bool) -> Result<Self> {
        if slots.len() > MAX_SLOTS {
            return Err(SyncCryptoError::BadChain("too many carried slots".into()));
        }
        Ok(SyncCrypto {
            keys,
            slots,
            icloud_slot,
        })
    }

    /// Passphrase path: unwrap every slot of a remote artifact header into
    /// the full chain, unioned with `known` (locally cached keys survive).
    pub fn from_passphrase(
        header: &ArtifactHeader,
        passphrase: &str,
        known: Option<&ContentKeys>,
        icloud_slot: bool,
    ) -> Result<Self> {
        let mut keys = known.cloned().unwrap_or_else(|| ContentKeys {
            entries: Vec::new(),
        });
        let mut unwrapped = Vec::with_capacity(header.slots.len());
        for (i, slot) in header.slots.iter().enumerate() {
            let version =
                header
                    .key_version
                    .checked_sub(i as u32)
                    .ok_or(SyncCryptoError::Corrupt(
                        "slot chain underflows key version".into(),
                    ))?;
            let ck = unwrap_ck(slot, passphrase)?;
            unwrapped.push((version, ck));
        }
        unwrapped.sort_by_key(|(v, _)| *v);
        let unwrapped = ContentKeys::from_entries(unwrapped)?;
        keys.merge_from(&unwrapped);
        if keys.entries.is_empty() {
            return Err(SyncCryptoError::BadChain("no keys after unwrap".into()));
        }
        Ok(SyncCrypto {
            keys,
            slots: header.slots.clone(),
            icloud_slot,
        })
    }

    /// Explicit user action (suspected compromise, lost device): mint the
    /// next CK and wrap it under the passphrase. Forward-only revocation —
    /// new snapshots and blobs are protected from a holder of the old CK;
    /// blobs last written before the rotation stay readable to them.
    pub fn rotate(&mut self, passphrase: &str) -> Result<()> {
        self.rotate_with_params(passphrase, DEFAULT_ARGON_PARAMS)
    }

    pub fn rotate_with_params(&mut self, passphrase: &str, params: ArgonParams) -> Result<()> {
        // Validate everything BEFORE mutating: on failure the object must
        // stay exactly as it was. Minting a key no slot can recover would
        // make a later publish forever un-recoverable via passphrase.
        if self.slots.len() + 1 > MAX_SLOTS {
            return Err(SyncCryptoError::BadChain(
                "rotation chain exceeded the slot budget".into(),
            ));
        }
        let next_version = self.keys.current_version() + 1;
        let ck = random_ck();
        let slot = wrap_ck(&ck, passphrase, params)?;
        self.keys.entries.push((next_version, ck));
        self.slots.insert(0, slot);
        Ok(())
    }

    pub fn keys(&self) -> &ContentKeys {
        &self.keys
    }

    /// Slots to write into the next published header: the tail of the
    /// carried chain matching this device's current key version, current
    /// first.
    pub fn slots_for_publish(&self) -> &[ArgonSlot] {
        let current = self.keys.current_version() as usize;
        if self.slots.len() <= current {
            &self.slots
        } else {
            &self.slots[self.slots.len() - current..]
        }
    }

    pub fn icloud_slot(&self) -> bool {
        self.icloud_slot
    }

    /// Refresh the carried slot chain from a remote header the engine just
    /// parsed (rotation happened elsewhere). Only a longer chain replaces
    /// ours — a shorter remote chain must not drop slots this device needs
    /// to publish. Called by the engine on every successful header read, so
    /// Keychain adopters pick up new wrapped slots without ever entering
    /// the passphrase.
    pub fn carry_slots_from(&mut self, header: &ArtifactHeader) {
        if header.slots.len() > self.slots.len() {
            self.slots = header.slots.clone();
        }
        self.icloud_slot = self.icloud_slot || header.icloud_slot;
    }

    /// Build the header for a publish. `sync_version` is the hashed form
    /// `<counter>:<blake3_hex(device_id)>`.
    pub fn build_header(
        &self,
        sync_version: &str,
        schema_version: u64,
    ) -> Result<(ArtifactHeader, [u8; SALT_LEN])> {
        let mut payload_salt = [0u8; SALT_LEN];
        rand::rngs::OsRng.fill_bytes(&mut payload_salt);
        let slots = self.slots_for_publish();
        if slots.len() != self.keys.current_version() as usize {
            return Err(SyncCryptoError::BadChain(format!(
                "slot chain ({} slots) does not cover key version {} — \
                 enter the sync passphrase to unwrap the full chain",
                slots.len(),
                self.keys.current_version()
            )));
        }
        Ok((
            ArtifactHeader {
                sync_version: sync_version.to_string(),
                schema_version,
                key_version: self.keys.current_version(),
                payload_salt,
                icloud_slot: self.icloud_slot,
                slots: slots.to_vec(),
            },
            payload_salt,
        ))
    }

    /// Fresh random salt for the payload stream id (used with
    /// [`encrypt_snapshot_payload`]; also embedded in the header).
    pub fn new_payload_salt() -> [u8; SALT_LEN] {
        let mut salt = [0u8; SALT_LEN];
        rand::rngs::OsRng.fill_bytes(&mut salt);
        salt
    }

    pub fn encrypt_blob(&self, digest_hex: &str, plaintext: &[u8]) -> Result<Vec<u8>> {
        encrypt_blob(&self.keys, digest_hex, plaintext)
    }

    pub fn decrypt_blob(&self, digest_hex: &str, ciphertext: &[u8]) -> Result<Vec<u8>> {
        decrypt_blob(&self.keys, digest_hex, ciphertext)
    }

    /// Value for the Apple-slot synchronizable item, or `None` once the
    /// group has rotated (`current_version >= 2`).
    ///
    /// **Freeze rule (ADR 05).** The item is the zero-ritual adoption
    /// channel — and it synchronizes to every device under the Apple ID,
    /// including a lost or compromised one. Writing a rotated CK into it
    /// (e.g. after a passphrase unlock re-learned the chain) would
    /// re-deliver the rotated key to exactly the devices the rotation was
    /// meant to shut out, defeating forward-only revocation. The item may
    /// therefore only ever carry the bootstrap-era key (v1); after any
    /// rotation, joining a group requires the passphrase once per device —
    /// the posture a user-initiated "suspected compromise / lost device"
    /// action should produce.
    pub fn synchronizable_item_value(&self) -> Option<String> {
        let version = self.keys.current_version();
        if version != 1 {
            return None;
        }
        Some(encode_synchronizable_item(version, self.keys.current_key()))
    }

    /// Serialize the FULL device-local state — key chain AND the wrapped
    /// slot chain — for the device-local cache file.
    ///
    /// Keys alone are not enough (real-device finding, 2026-08-27): the
    /// bootstrapping first device must publish the very first artifact
    /// BEFORE any artifact exists to carry slots from. A keys-only cache
    /// reconstructs a slot-less crypto state whose `build_header` refuses
    /// to publish — a liveness deadlock on the first publish, on every
    /// platform (desktop merely hides it behind the live bootstrap object
    /// until the process dies). The cache file sits beside the plaintext
    /// DB by the same ADR 04 §2 argument as the keys-only form it replaces.
    pub fn encode_state(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.bytes(Self::STATE_MAGIC);
        w.u16(self.keys.entries.len() as u16);
        for (v, k) in &self.keys.entries {
            w.u32(*v);
            w.bytes(k);
        }
        w.u16(self.slots.len() as u16);
        for slot in &self.slots {
            slot.encode_slot(&mut w);
        }
        w.u8(self.icloud_slot as u8);
        w.into_vec()
    }

    /// Parse a [`SyncCrypto::encode_state`] cache. Slot and key counts are
    /// bounds-checked like artifact headers; structural damage is an
    /// error, never a silent fallback.
    pub fn decode_state(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let magic = r.take(Self::STATE_MAGIC.len())?;
        if magic != Self::STATE_MAGIC {
            return Err(SyncCryptoError::BadChain("bad state-cache magic".into()));
        }
        let key_count = r.u16()? as usize;
        if key_count == 0 || key_count > MAX_SLOTS {
            return Err(SyncCryptoError::BadChain(
                "state-cache key count out of bounds".into(),
            ));
        }
        let mut entries = Vec::with_capacity(key_count);
        for _ in 0..key_count {
            let v = r.u32()?;
            let k = r.array::<CK_LEN>()?;
            entries.push((v, k));
        }
        let keys = ContentKeys::from_entries(entries)?;
        let slot_count = r.u16()? as usize;
        if slot_count > MAX_SLOTS {
            return Err(SyncCryptoError::BadChain(
                "state-cache slot count out of bounds".into(),
            ));
        }
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(ArgonSlot::decode_slot(&mut r)?);
        }
        let icloud_slot = r.u8()? != 0;
        if r.remaining() != 0 {
            return Err(SyncCryptoError::BadChain(
                "trailing bytes in state cache".into(),
            ));
        }
        Ok(SyncCrypto {
            keys,
            slots,
            icloud_slot,
        })
    }

    const STATE_MAGIC: &[u8; 11] = b"CHCRYPTOST\x01";
}

// ---------------------------------------------------------------------------
// Passphrase generation (diceware-style, ADR 04 §3)
// ---------------------------------------------------------------------------

/// 256-word list (8 bits/word). 12 words → 96 bits of entropy; Argon2id
/// multiplies offline attack cost on top. Generated rather than user-chosen
/// because passphrase quality bounds the cloud copy's security.
const PASSPHRASE_WORDS: [&str; 256] = [
    "acid",
    "acorn",
    "actor",
    "adapt",
    "agent",
    "album",
    "alert",
    "alien",
    "alloy",
    "amber",
    "anchor",
    "angle",
    "ankle",
    "apple",
    "april",
    "arcade",
    "arctic",
    "armor",
    "arrow",
    "aspen",
    "atlas",
    "aurora",
    "avoid",
    "axis",
    "bacon",
    "badge",
    "bagel",
    "baker",
    "balcony",
    "bamboo",
    "banjo",
    "barrel",
    "basin",
    "basket",
    "baton",
    "beacon",
    "beagle",
    "beaver",
    "bedrock",
    "beech",
    "beetle",
    "bell",
    "berg",
    "berry",
    "bison",
    "blaze",
    "blend",
    "blimp",
    "block",
    "bloom",
    "blossom",
    "blueprint",
    "boomerang",
    "boulder",
    "bracket",
    "bramble",
    "brass",
    "breeze",
    "bridge",
    "bronze",
    "brook",
    "bubble",
    "bucket",
    "buffalo",
    "bugle",
    "bunker",
    "burrow",
    "cabin",
    "cactus",
    "camera",
    "canal",
    "candle",
    "canoe",
    "canvas",
    "canyon",
    "capital",
    "caramel",
    "cargo",
    "carrot",
    "castle",
    "cedar",
    "cellar",
    "cement",
    "cereal",
    "chalk",
    "champion",
    "chapel",
    "charcoal",
    "cherry",
    "chimney",
    "chorus",
    "cinnamon",
    "circle",
    "citrus",
    "civil",
    "clarity",
    "claw",
    "clay",
    "cliff",
    "cloak",
    "clock",
    "clover",
    "cobalt",
    "cocoa",
    "colonel",
    "colt",
    "compass",
    "concert",
    "copper",
    "coral",
    "cosmos",
    "cotton",
    "county",
    "cover",
    "cowboy",
    "crayon",
    "creek",
    "cricket",
    "crossbow",
    "crown",
    "crystal",
    "cupcake",
    "curtain",
    "cycle",
    "dagger",
    "dahlia",
    "daisy",
    "dam",
    "dandelion",
    "daring",
    "dawn",
    "debut",
    "decade",
    "delta",
    "denim",
    "depot",
    "desert",
    "diamond",
    "diesel",
    "dinner",
    "dock",
    "dolphin",
    "domain",
    "donkey",
    "dough",
    "dove",
    "draft",
    "dragon",
    "draper",
    "dream",
    "drift",
    "driver",
    "drum",
    "dune",
    "duplex",
    "durable",
    "eagle",
    "earth",
    "easel",
    "east",
    "echo",
    "eclipse",
    "edge",
    "edible",
    "elegant",
    "elephant",
    "elevator",
    "ember",
    "emerald",
    "empire",
    "enamel",
    "engine",
    "enigma",
    "enroll",
    "envelope",
    "epoch",
    "equator",
    "erosion",
    "etching",
    "evolve",
    "excite",
    "exhale",
    "exile",
    "exit",
    "fable",
    "fabric",
    "factory",
    "falcon",
    "family",
    "famous",
    "fang",
    "farm",
    "fastball",
    "feather",
    "fellow",
    "fence",
    "ferry",
    "fiber",
    "fiction",
    "fiddle",
    "figure",
    "filter",
    "finch",
    "finger",
    "fjord",
    "flamingo",
    "flare",
    "flask",
    "fleet",
    "flint",
    "florist",
    "flower",
    "flute",
    "foam",
    "focus",
    "foliage",
    "forest",
    "format",
    "fossil",
    "fountain",
    "fox",
    "fragment",
    "freckle",
    "frigate",
    "frontier",
    "frost",
    "fruit",
    "fulcrum",
    "furniture",
    "fuzzy",
    "gadget",
    "galaxy",
    "gallery",
    "gamble",
    "garage",
    "garden",
    "garlic",
    "garnet",
    "gateway",
    "gazelle",
    "gear",
    "gemini",
    "general",
    "gentle",
    "geyser",
    "ginger",
    "giraffe",
    "glacier",
    "glide",
    "globe",
    "glory",
    "glove",
    "granite",
    "grape",
    "gravity",
    "green",
];

/// Number of words in a generated passphrase (256^12 ≈ 2^96).
pub const PASSPHRASE_WORDS_COUNT: usize = 12;

/// Generate a high-entropy diceware-style passphrase: 12 words from the
/// 256-word list, space-separated. Shown once with save guidance; never
/// stored by the app.
pub fn generate_passphrase() -> String {
    let mut indices = [0u8; PASSPHRASE_WORDS_COUNT];
    rand::rngs::OsRng.fill_bytes(&mut indices);
    indices
        .iter()
        .map(|i| PASSPHRASE_WORDS[*i as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weak_crypto() -> SyncCrypto {
        SyncCrypto::bootstrap_with_params("correct horse battery staple", ArgonParams::test())
            .unwrap()
    }

    fn digest_of(data: &[u8]) -> String {
        crate::crypto::blake3_hex(data)
    }

    #[test]
    fn chunk_roundtrip_various_sizes() {
        let parent = [7u8; 32];
        let stream_id = b"stream-identity";
        for size in [
            0usize,
            1,
            15,
            CHUNK_PLAINTEXT - 1,
            CHUNK_PLAINTEXT,
            CHUNK_PLAINTEXT + 1,
            2 * CHUNK_PLAINTEXT,
            2 * CHUNK_PLAINTEXT + 123,
            3 * CHUNK_PLAINTEXT + 1,
        ] {
            let data: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
            let sealed = seal_chunks(&parent, stream_id, b"content-a", 1, &data).unwrap();
            let opened = open_chunks(&parent, stream_id, b"content-a", 1, &sealed).unwrap();
            assert_eq!(opened, data, "size {size}");
        }
    }

    #[test]
    fn truncation_fails() {
        let parent = [7u8; 32];
        let data: Vec<u8> = (0..CHUNK_PLAINTEXT * 3 + 10).map(|i| i as u8).collect();
        let sealed = seal_chunks(&parent, b"id", b"content-a", 1, &data).unwrap();
        // Drop trailing bytes at every interesting offset: mid-tag, whole
        // final frame, mid-frame.
        for cut in [
            sealed.len() - 1,
            sealed.len() - 5,
            sealed.len() - 20,
            sealed.len() - 24,
            sealed.len() / 2,
        ] {
            let truncated = &sealed[..cut];
            assert!(
                open_chunks(&parent, b"id", b"content-a", 1, truncated).is_err(),
                "truncation at {cut} must fail"
            );
        }
    }

    #[test]
    fn truncation_dropping_final_frame_fails() {
        // Dropping exactly the last whole frame leaves an earlier frame as
        // the apparent final one — its AAD says final=false, so the tag
        // check must reject.
        let parent = [7u8; 32];
        let data = vec![9u8; CHUNK_PLAINTEXT * 2];
        let sealed = seal_chunks(&parent, b"id", b"content-a", 1, &data).unwrap();
        // Frame i: 4-byte len + 64 KiB ct + 16-byte tag = 4 + 65552 bytes.
        let frame_len = 4 + CHUNK_PLAINTEXT + TAG_LEN;
        let truncated = &sealed[..sealed.len() - frame_len];
        assert!(open_chunks(&parent, b"id", b"content-a", 1, truncated).is_err());
    }

    #[test]
    fn reorder_fails() {
        let parent = [7u8; 32];
        let data = vec![5u8; CHUNK_PLAINTEXT * 3];
        let sealed = seal_chunks(&parent, b"id", b"content-a", 1, &data).unwrap();
        let frame_len = 4 + CHUNK_PLAINTEXT + TAG_LEN;
        // Swap frames 0 and 1.
        let (f0, f1, rest) = (
            sealed[0..frame_len].to_vec(),
            sealed[frame_len..2 * frame_len].to_vec(),
            sealed[2 * frame_len..].to_vec(),
        );
        let reordered = [f1, f0, rest].concat();
        assert!(open_chunks(&parent, b"id", b"content-a", 1, &reordered).is_err());
    }

    #[test]
    fn substitution_fails() {
        let parent = [7u8; 32];
        let data = b"attack at dawn".to_vec();
        let sealed = seal_chunks(&parent, b"id", b"content-a", 1, &data).unwrap();
        let mut tampered = sealed.clone();
        tampered[6] ^= 0x01;
        assert!(open_chunks(&parent, b"id", b"content-a", 1, &tampered).is_err());
    }

    #[test]
    fn cross_record_splice_fails() {
        // Chunks spliced from one record's stream into another's must fail:
        // both the derived stream key (different stream ids) and the AAD
        // content id bind chunks to their record.
        let parent = [7u8; 32];
        let a = vec![1u8; CHUNK_PLAINTEXT * 2];
        let b = vec![2u8; CHUNK_PLAINTEXT * 2];
        let sealed_a = seal_chunks(&parent, b"id-a", b"digest-a", 1, &a).unwrap();
        let sealed_b = seal_chunks(&parent, b"id-b", b"digest-b", 1, &b).unwrap();
        let frame_len = 4 + CHUNK_PLAINTEXT + TAG_LEN;
        // Replace frame 1 of A with frame 1 of B.
        let mut spliced = sealed_a[..frame_len].to_vec();
        spliced.extend_from_slice(&sealed_b[frame_len..2 * frame_len]);
        assert!(open_chunks(&parent, b"id-a", b"digest-a", 1, &spliced).is_err());
        // Same stream id (same digest), different record content: the AAD
        // content id alone must catch it.
        let sealed_b_same_id = seal_chunks(&parent, b"id-a", b"digest-b", 1, &b).unwrap();
        let mut spliced2 = sealed_a[..frame_len].to_vec();
        spliced2.extend_from_slice(&sealed_b_same_id[frame_len..2 * frame_len]);
        assert!(open_chunks(&parent, b"id-a", b"digest-a", 1, &spliced2).is_err());
    }

    #[test]
    fn wrong_stream_id_or_content_fails() {
        let parent = [7u8; 32];
        let data = b"payload".to_vec();
        let sealed = seal_chunks(&parent, b"id-a", b"digest-a", 1, &data).unwrap();
        assert!(open_chunks(&parent, b"id-b", b"digest-a", 1, &sealed).is_err());
        assert!(open_chunks(&parent, b"id-a", b"digest-b", 1, &sealed).is_err());
        assert!(open_chunks(&parent, b"id-a", b"digest-a", 2, &sealed).is_err());
    }

    #[test]
    fn blob_roundtrip_and_determinism() {
        let crypto = weak_crypto();
        let data = b"attachment bytes".to_vec();
        let digest = digest_of(&data);
        let sealed1 = crypto.encrypt_blob(&digest, &data).unwrap();
        let sealed2 = crypto.encrypt_blob(&digest, &data).unwrap();
        assert_eq!(sealed1, sealed2, "encryption must be deterministic");
        let opened = crypto.decrypt_blob(&digest, &sealed1).unwrap();
        assert_eq!(opened, data);
    }

    #[test]
    fn blob_stream_roundtrip_above_threshold() {
        let crypto = weak_crypto();
        let data: Vec<u8> = (0..BLOB_STREAM_THRESHOLD + CHUNK_PLAINTEXT + 7)
            .map(|i| (i % 256) as u8)
            .collect();
        let digest = digest_of(&data);
        let sealed = crypto.encrypt_blob(&digest, &data).unwrap();
        assert!(sealed.len() > BLOB_STREAM_THRESHOLD);
        let opened = crypto.decrypt_blob(&digest, &sealed).unwrap();
        assert_eq!(opened, data);
    }

    #[test]
    fn blob_mismatched_name_bytes_rejected_pre_encrypt() {
        // Same name with different bytes = nonce reuse under the same key —
        // catastrophic for GCM. The pre-encrypt assertion must catch it.
        let crypto = weak_crypto();
        let data = b"real bytes".to_vec();
        let digest = digest_of(b"other bytes");
        match crypto.encrypt_blob(&digest, &data) {
            Err(SyncCryptoError::HashMismatch { .. }) => {}
            other => panic!("expected HashMismatch, got {other:?}"),
        }
    }

    #[test]
    fn blob_renamed_digest_fails_decrypt() {
        let crypto = weak_crypto();
        let data = b"bytes under digest a".to_vec();
        let digest_a = digest_of(&data);
        let sealed = crypto.encrypt_blob(&digest_a, &data).unwrap();
        let digest_b = digest_of(b"bytes under digest b");
        assert!(crypto.decrypt_blob(&digest_b, &sealed).is_err());
    }

    #[test]
    fn state_cache_roundtrip_and_first_publish() {
        // The 2026-08-27 liveness finding: a keys-only cache cannot publish
        // the FIRST artifact (no slots). The full-state cache must round-trip
        // the bootstrap slot so a fresh process can publish before any
        // artifact exists.
        let mut crypto = weak_crypto();
        crypto
            .rotate_with_params("correct horse battery staple", ArgonParams::test())
            .unwrap();
        let bytes = crypto.encode_state();
        let restored = SyncCrypto::decode_state(&bytes).unwrap();
        assert_eq!(restored.keys(), crypto.keys());
        assert_eq!(restored.slots_for_publish(), crypto.slots_for_publish());
        assert_eq!(restored.icloud_slot(), crypto.icloud_slot());
        // The restored state can build a header with NO artifact present.
        let (header, salt) = restored.build_header("1:aa", 1).unwrap();
        assert_eq!(header.slots.len(), 2);
        let payload = encrypt_snapshot_payload(restored.keys(), &salt, b"x").unwrap();
        let opened = decrypt_snapshot_payload(
            &SyncCrypto::decode_state(&bytes).unwrap().keys().clone(),
            &header,
            &payload,
        )
        .unwrap();
        assert_eq!(opened, b"x");
        // Structural damage is rejected.
        assert!(SyncCrypto::decode_state(b"garbage").is_err());
        assert!(SyncCrypto::decode_state(&bytes[..bytes.len() - 1]).is_err());
        // Keys-only legacy caches no longer decode (no-compat window).
        assert!(SyncCrypto::decode_state(&crypto.keys().encode()).is_err());
    }

    #[test]
    fn blob_tamper_and_truncate_fail() {
        let crypto = weak_crypto();
        let data = vec![42u8; 1024];
        let digest = digest_of(&data);
        let mut sealed = crypto.encrypt_blob(&digest, &data).unwrap();
        let mut tampered = sealed.clone();
        let cut = tampered.len() - 3;
        tampered.truncate(cut);
        assert!(crypto.decrypt_blob(&digest, &tampered).is_err());
        *sealed.last_mut().unwrap() ^= 1;
        assert!(crypto.decrypt_blob(&digest, &sealed).is_err());
    }

    #[test]
    fn rotation_keeps_old_keys_decrypt_only() {
        let mut crypto = weak_crypto();
        let old_data = b"blob written before rotation".to_vec();
        let old_digest = digest_of(&old_data);
        let old_sealed = crypto.encrypt_blob(&old_digest, &old_data).unwrap();
        crypto
            .rotate_with_params("correct horse battery staple", ArgonParams::test())
            .unwrap();
        assert_eq!(crypto.keys().current_version(), 2);
        // Old blob still decrypts (old key rides forward).
        assert_eq!(
            crypto.decrypt_blob(&old_digest, &old_sealed).unwrap(),
            old_data
        );
        // New writes use the new version.
        let new_data = b"blob written after rotation".to_vec();
        let new_digest = digest_of(&new_data);
        let new_sealed = crypto.encrypt_blob(&new_digest, &new_data).unwrap();
        // A holder of only the old key (fresh chain with just v1) cannot
        // decrypt new writes.
        let keys_v1 = ContentKeys::from_entries(vec![(
            1,
            crypto.keys().key_for_version(1).unwrap().to_owned(),
        )])
        .unwrap();
        let old_only = SyncCrypto::from_keys(keys_v1, vec![], false).unwrap();
        match old_only.decrypt_blob(&new_digest, &new_sealed) {
            Err(SyncCryptoError::UnknownKeyVersion(2)) => {}
            other => panic!("expected UnknownKeyVersion(2), got {other:?}"),
        }
    }

    #[test]
    fn slot_chain_carries_forward_without_passphrase() {
        // Bootstrap on device A, rotate once; device B adopts the CURRENT
        // key from the Keychain and carries A's slot bytes verbatim — it can
        // still publish a complete header.
        let mut a = weak_crypto();
        a.rotate_with_params("correct horse battery staple", ArgonParams::test())
            .unwrap();
        let (header, _salt) = a.build_header("3:cafe", 1).unwrap();
        assert_eq!(header.key_version, 2);
        assert_eq!(header.slots.len(), 2);

        let keys_b = ContentKeys::from_entries(vec![(2, *a.keys().current_key())]).unwrap();
        let b = SyncCrypto::from_keys(keys_b, header.slots.clone(), true).unwrap();
        let (header_b, _) = b.build_header("4:beef", 1).unwrap();
        assert_eq!(header_b.key_version, 2);
        assert_eq!(header_b.slots.len(), 2);
        // The carried slots unwrap under the passphrase to the full chain.
        let recovered =
            SyncCrypto::from_passphrase(&header_b, "correct horse battery staple", None, true)
                .unwrap();
        assert_eq!(recovered.keys(), a.keys());
    }

    #[test]
    fn partial_chain_cannot_publish_missing_slots() {
        let mut a = weak_crypto();
        a.rotate_with_params("correct horse battery staple", ArgonParams::test())
            .unwrap();
        // Device holds the full rotated chain but only the pre-rotation
        // slots (1 slot for version 2): publishing must fail loudly.
        let keys_b = ContentKeys::from_entries(vec![
            (1, *a.keys().key_for_version(1).unwrap()),
            (2, *a.keys().current_key()),
        ])
        .unwrap();
        let slots_v1 = a.slots_for_publish()[1..].to_vec();
        let b = SyncCrypto::from_keys(keys_b, slots_v1, true).unwrap();
        match b.build_header("5:f00d", 1) {
            Err(SyncCryptoError::BadChain(_)) => {}
            other => panic!("expected BadChain, got {other:?}"),
        }
    }

    #[test]
    fn synchronizable_item_roundtrip_and_legacy_rejection() {
        let key = [9u8; CK_LEN];
        let value = encode_synchronizable_item(3, &key);
        assert_eq!(parse_synchronizable_item(&value), Some((3, key)));
        // Legacy bare-base64 (pre-ADR-05) items never adopt — the version
        // is the anti-stale guard and must be mandatory.
        use base64::{Engine as _, engine::general_purpose};
        let legacy = general_purpose::STANDARD.encode(key);
        assert_eq!(parse_synchronizable_item(&legacy), None);
        // Malformed variants.
        assert_eq!(parse_synchronizable_item("v1:not-base64!!"), None);
        assert_eq!(parse_synchronizable_item("x1:AAAA"), None);
        assert_eq!(parse_synchronizable_item("v1"), None);
        assert_eq!(
            parse_synchronizable_item(&format!(
                "v1:{}",
                general_purpose::STANDARD.encode([1u8; 31])
            )),
            None,
            "wrong key length must not adopt"
        );
    }

    #[test]
    fn synchronizable_item_freezes_after_rotation() {
        // ADR 05 freeze rule: only the bootstrap-era (v1) key may ride the
        // zero-ritual item. Refreshing it after a rotation would re-deliver
        // the rotated CK to every device under the Apple ID — including a
        // lost one — defeating forward-only revocation.
        let mut crypto = weak_crypto();
        let v1_value = crypto.synchronizable_item_value().unwrap();
        assert_eq!(parse_synchronizable_item(&v1_value).unwrap().0, 1);
        crypto
            .rotate_with_params("correct horse battery staple", ArgonParams::test())
            .unwrap();
        assert!(
            crypto.synchronizable_item_value().is_none(),
            "a rotated group must never refresh the item"
        );
    }

    #[test]
    fn rotate_budget_error_leaves_object_unchanged() {
        // F3: the slot-budget rejection must fire BEFORE any mutation — a
        // minted key with no matching wrapped slot would be publishable but
        // never passphrase-recoverable.
        let mut crypto = weak_crypto();
        crypto.slots = (0..MAX_SLOTS)
            .map(|_| crypto.slots[0].clone())
            .collect::<Vec<_>>();
        let keys_before = crypto.keys().clone();
        let slots_before = crypto.slots.clone();
        match crypto.rotate_with_params("p", ArgonParams::test()) {
            Err(SyncCryptoError::BadChain(_)) => {}
            other => panic!("expected BadChain, got {other:?}"),
        }
        assert_eq!(crypto.keys(), &keys_before);
        assert_eq!(crypto.slots, slots_before);
        assert_eq!(crypto.keys().current_version(), 1);
    }

    #[test]
    fn artifact_roundtrip() {
        let mut crypto = weak_crypto();
        crypto
            .rotate_with_params("pass phrase two words", ArgonParams::test())
            .unwrap();
        let (header, salt) = crypto.build_header("7:abcd", 3).unwrap();
        let payload = encrypt_snapshot_payload(crypto.keys(), &salt, b"zstd-bytes-here").unwrap();
        let artifact = build_artifact(&header, &payload);
        let (parsed_header, parsed_payload) = split_artifact(&artifact).unwrap();
        assert_eq!(parsed_header, header);
        let opened =
            decrypt_snapshot_payload(crypto.keys(), &parsed_header, parsed_payload).unwrap();
        assert_eq!(opened, b"zstd-bytes-here");
    }

    #[test]
    fn artifact_wrong_passphrase_and_corruption() {
        let crypto = weak_crypto();
        let (header, salt) = crypto.build_header("1:aa", 1).unwrap();
        let payload = encrypt_snapshot_payload(crypto.keys(), &salt, b"data").unwrap();
        let artifact = build_artifact(&header, &payload);

        // Wrong passphrase fails on unwrap (AEAD tag failure).
        assert!(SyncCrypto::from_passphrase(&header, "wrong", None, true).is_err());

        // Garbage magic never parses.
        let mut foreign = artifact.clone();
        foreign[0] = b'X';
        assert!(split_artifact(&foreign).is_err());
        assert!(ArtifactHeader::parse(b"SQLite format 3\0 junk").is_err());

        // Header tamper on payload salt → tag failure at open.
        let (h, p) = split_artifact(&artifact).unwrap();
        let mut bad = h.clone();
        bad.payload_salt[0] ^= 1;
        assert!(decrypt_snapshot_payload(crypto.keys(), &bad, p).is_err());
    }

    #[test]
    fn header_parse_bounds() {
        let crypto = weak_crypto();
        let (header, _) = crypto.build_header("2:bb", 1).unwrap();
        let bytes = header.to_bytes();
        // Truncation at any offset must fail parsing.
        for cut in 0..bytes.len() {
            assert!(ArtifactHeader::parse(&bytes[..cut]).is_err(), "cut {cut}");
        }
        assert!(ArtifactHeader::parse(&bytes).is_ok());
    }

    #[test]
    fn chain_encode_decode_roundtrip() {
        let mut crypto = weak_crypto();
        crypto.rotate_with_params("p", ArgonParams::test()).unwrap();
        let encoded = crypto.keys().encode();
        let decoded = ContentKeys::decode(&encoded).unwrap();
        assert_eq!(decoded, *crypto.keys());
        // The cache file is intentionally AEAD-free material beside a
        // plaintext DB (ADR 04 §2); corruption surfaces as a wrong key at
        // first use, but structural damage must still be rejected.
        assert!(ContentKeys::decode(b"garbage").is_err());
        let truncated = &encoded[..encoded.len() - 1];
        assert!(ContentKeys::decode(truncated).is_err());
    }

    #[test]
    fn from_passphrase_recovers_full_chain() {
        let crypto = weak_crypto();
        let (header, _) = crypto.build_header("9:cc", 1).unwrap();
        let recovered =
            SyncCrypto::from_passphrase(&header, "correct horse battery staple", None, true)
                .unwrap();
        assert_eq!(recovered.keys(), crypto.keys());
    }
    #[test]
    fn passphrase_shape() {
        let mut seen = std::collections::HashSet::new();
        for w in PASSPHRASE_WORDS.iter() {
            assert!(w.len() >= 3 && w.len() <= 10, "word length: {w}");
            assert!(w.chars().all(|c| c.is_ascii_lowercase()), "word {w}");
            assert!(seen.insert(*w), "duplicate word {w}");
        }
        assert_eq!(seen.len(), 256);
        // Generator output shape.
        for _ in 0..8 {
            let p = generate_passphrase();
            let words: Vec<&str> = p.split(' ').collect();
            assert_eq!(words.len(), PASSPHRASE_WORDS_COUNT);
            for w in words {
                assert!(PASSPHRASE_WORDS.contains(&w));
            }
        }
        // Entropy sanity: two draws differ.
        assert_ne!(generate_passphrase(), generate_passphrase());
    }

    #[test]
    fn weak_test_params_are_not_default() {
        assert!(DEFAULT_ARGON_PARAMS.m_cost > ArgonParams::test().m_cost);
    }
}
