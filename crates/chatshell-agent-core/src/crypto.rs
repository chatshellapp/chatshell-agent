//! AES-256-GCM envelope encryption for provider API keys (ADR 01 §5.5).
//!
//! Single source of the ciphertext format shared by desktop and iOS:
//! `base64(12-byte nonce ‖ AES-256-GCM ciphertext‖tag)`. The master key is
//! 32 random bytes; each platform keeps it in its own keychain and never
//! persists it anywhere else. Keychain lookup/caching stays per-platform —
//! only the format lives here.

use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit, OsRng},
};
use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use rand::RngCore;

pub const MASTER_KEY_LEN: usize = 32;
const NONCE_LEN: usize = 12;

/// Random 32-byte master encryption key.
pub fn generate_master_key() -> [u8; MASTER_KEY_LEN] {
    let mut key = [0u8; MASTER_KEY_LEN];
    OsRng.fill_bytes(&mut key);
    key
}

/// Encrypt `plaintext` under `key`; returns `base64(nonce ‖ ciphertext‖tag)`.
pub fn encrypt(key: &[u8; MASTER_KEY_LEN], plaintext: &str) -> Result<String> {
    let cipher = Aes256Gcm::new(key.into());

    let mut nonce_bytes = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_bytes())
        .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

    let mut result = nonce_bytes.to_vec();
    result.extend_from_slice(&ciphertext);

    Ok(general_purpose::STANDARD.encode(&result))
}

/// Decrypt output of [`encrypt`].
pub fn decrypt(key: &[u8; MASTER_KEY_LEN], encrypted: &str) -> Result<String> {
    let cipher = Aes256Gcm::new(key.into());

    let data = general_purpose::STANDARD
        .decode(encrypted)
        .map_err(|e| anyhow::anyhow!("Base64 decode failed: {}", e))?;

    if data.len() < NONCE_LEN {
        return Err(anyhow::anyhow!("Invalid encrypted data"));
    }

    let (nonce_bytes, ciphertext) = data.split_at(NONCE_LEN);
    let nonce = Nonce::from_slice(nonce_bytes);

    let plaintext = cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))?;

    Ok(String::from_utf8(plaintext)?)
}

/// BLAKE3 hex digest of arbitrary bytes — byte-for-byte identical to
/// desktop's `storage::hash_bytes`, so content hashes in `files.content_hash`
/// and blob sidecar names (`<cloud>/blobs/<hash>`) match across platforms.
pub fn blake3_hex(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

/// How a device settles its master key, given the iCloud-synchronizable
/// transport item and its device-local copy. Pure policy shared by both
/// platforms (desktop consumes it directly, iOS over FFI) so the two apps
/// can never diverge on key authority:
///
/// - The synchronizable item is the ecosystem authority: any device that
///   can see it adopts it. Whoever creates a key first (desktop or a
///   standalone iOS device) therefore defines the one key; later devices
///   follow, and a lost local key (reinstall) is healed by the item.
/// - With no item visible (desktop dev builds under the macOS 26
///   entitlement blocker, offline first run), the local key stands and the
///   item write is retried later.
/// - Nothing anywhere: mint. iOS-only users mint on first key save; no
///   device ever needs the desktop to exist.
///
/// When the item replaces a *different* local key, the old key must be
/// demoted to a decrypt fallback (`master_encryption_key_previous`), not
/// discarded: rows this device encrypted earlier still open, and a repair
/// pass re-encrypts them under the adopted key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MasterKeyResolution {
    /// The existing local key stands (item equals local, or no item).
    KeepCurrent,
    /// The synchronizable item wins. `demote_current` is true when a
    /// different local key must be retained as the decrypt fallback.
    AdoptSync { demote_current: bool },
    /// No key exists anywhere: mint one and persist it locally (the item
    /// write is the caller's best effort).
    Mint,
}

/// Resolve the master key policy for one device.
pub fn resolve_master_key(sync_item: Option<&str>, local_key: Option<&str>) -> MasterKeyResolution {
    match (sync_item, local_key) {
        (Some(sync), Some(local)) => {
            if sync == local {
                MasterKeyResolution::KeepCurrent
            } else {
                MasterKeyResolution::AdoptSync {
                    demote_current: true,
                }
            }
        }
        // Fresh device or reinstalled one: the item heals the local copy.
        (Some(_), None) => MasterKeyResolution::AdoptSync {
            demote_current: false,
        },
        // Transport invisible (blocker/offline): local stands; retry the
        // item write when possible.
        (None, Some(_)) => MasterKeyResolution::KeepCurrent,
        (None, None) => MasterKeyResolution::Mint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let key = generate_master_key();
        let ct = encrypt(&key, "sk-test-123").unwrap();
        assert_eq!(decrypt(&key, &ct).unwrap(), "sk-test-123");
    }

    #[test]
    fn ciphertexts_are_randomized() {
        let key = generate_master_key();
        assert_ne!(
            encrypt(&key, "same").unwrap(),
            encrypt(&key, "same").unwrap()
        );
    }

    #[test]
    fn wrong_key_fails() {
        let ct = encrypt(&generate_master_key(), "secret").unwrap();
        assert!(decrypt(&generate_master_key(), &ct).is_err());
    }

    #[test]
    fn truncated_input_rejected() {
        let key = generate_master_key();
        assert!(decrypt(&key, "c2hvcnQ=").is_err());
    }
}

#[cfg(test)]
mod blake3_tests {
    #[test]
    fn blake3_hex_matches_known_digest() {
        assert_eq!(
            super::blake3_hex(b"hello"),
            "ea8f163db38682925e4491c5e58d4bb3506ef8c14eb78a86e908c5624a67200f"
        );
    }
}

#[cfg(test)]
mod resolve_tests {
    use super::*;

    #[test]
    fn identical_item_and_local_keeps_current() {
        assert_eq!(
            resolve_master_key(Some("k1"), Some("k1")),
            MasterKeyResolution::KeepCurrent
        );
    }

    #[test]
    fn different_item_adopts_and_demotes() {
        assert_eq!(
            resolve_master_key(Some("k2"), Some("k1")),
            MasterKeyResolution::AdoptSync {
                demote_current: true
            }
        );
    }

    #[test]
    fn item_without_local_adopts_without_demote() {
        // Fresh install, or reinstall after the local keychain was wiped:
        // the item restores the key, no fallback needed.
        assert_eq!(
            resolve_master_key(Some("k1"), None),
            MasterKeyResolution::AdoptSync {
                demote_current: false
            }
        );
    }

    #[test]
    fn local_without_item_keeps_current() {
        // Desktop under the macOS 26 blocker: item invisible, local stands.
        assert_eq!(
            resolve_master_key(None, Some("k1")),
            MasterKeyResolution::KeepCurrent
        );
    }

    #[test]
    fn nothing_anywhere_mints() {
        // iOS-only first run: minting must be possible without any desktop.
        assert_eq!(resolve_master_key(None, None), MasterKeyResolution::Mint);
    }
}
