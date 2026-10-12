//! Filesystem seam for the tool engines (ADR 03 Layer 1).
//!
//! The tool engines in `tools/` are pure decision logic plus IO routed
//! through this trait. `StdFs` is the std/`ignore`-backed implementation any
//! Rust host (desktop today, Android later) can use as-is; a platform with a
//! non-POSIX-visible filesystem implements `Fs` itself.
//!
//! The trait is intentionally synchronous: the engines already perform
//! blocking IO inside their async `call`s, and hosts that care wrap the call
//! in `spawn_blocking`. Streaming reads go through [`Fs::open`] so consumers
//! (grep) keep their bounded-memory behavior instead of buffering whole
//! files.

use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// File facts the engines need (metadata plus mtime for recency ordering).
#[derive(Debug, Clone, Copy)]
pub struct FileMeta {
    pub len: u64,
    pub is_dir: bool,
    pub is_file: bool,
    pub modified: Option<SystemTime>,
}

/// Directory-walk filters. Semantics follow the `ignore` crate (ripgrep):
/// `include` are gitignore-style override globs, `types` are ripgrep named
/// file types where the implementation supports them, `max_depth` bounds the
/// walk. Implementations that cannot honor a field must document the
/// deviation.
#[derive(Debug, Clone, Default)]
pub struct WalkFilters {
    pub include: Vec<String>,
    pub types: Vec<String>,
    pub max_depth: Option<usize>,
    /// Stop the walk after visiting this many files (worst-case bound).
    pub max_files: Option<usize>,
}

/// Filesystem seam for the LLM tool engines.
pub trait Fs: Send + Sync {
    /// Open for streaming reads (engines must not buffer whole large files).
    fn open(&self, path: &Path) -> io::Result<Box<dyn io::Read + Send>>;
    /// Read fully into memory (default: drain [`Fs::open`]).
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        let mut reader = self.open(path)?;
        reader.read_to_end(&mut buf)?;
        Ok(buf)
    }
    fn write(&self, path: &Path, contents: &[u8]) -> io::Result<()>;
    /// Mark a file executable (best-effort; default no-op for hosts whose
    /// filesystem has no exec bit). Never fails a write on unsupported hosts.
    fn set_executable(&self, _path: &Path) -> io::Result<()> {
        Ok(())
    }
    fn create_dir_all(&self, path: &Path) -> io::Result<()>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
    fn metadata(&self, path: &Path) -> io::Result<FileMeta>;
    /// Lexical existence check where metadata is unavailable; default
    /// derives from [`Fs::metadata`].
    fn exists(&self, path: &Path) -> bool {
        self.metadata(path).is_ok()
    }
    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf>;
    /// Walk `root` returning file paths (dirs excluded), gitignore-aware
    /// where the host supports it.
    fn walk(&self, root: &Path, filters: &WalkFilters) -> io::Result<Vec<PathBuf>>;
}

/// The default implementation over `std::fs` + the `ignore` walker.
#[derive(Debug, Clone, Copy, Default)]
pub struct StdFs;

impl Fs for StdFs {
    fn open(&self, path: &Path) -> io::Result<Box<dyn io::Read + Send>> {
        Ok(Box::new(std::fs::File::open(path)?))
    }

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        std::fs::read(path)
    }

    fn write(&self, path: &Path, contents: &[u8]) -> io::Result<()> {
        std::fs::write(path, contents)
    }

    fn set_executable(&self, path: &Path) -> io::Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(path)?.permissions();
            perms.set_mode(perms.mode() | 0o755);
            std::fs::set_permissions(path, perms)
        }
        #[cfg(not(unix))]
        {
            // No exec bit on Windows: executability follows the file
            // extension, so there is nothing to set.
            let _ = path;
            Ok(())
        }
    }

    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        std::fs::create_dir_all(path)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        std::fs::remove_file(path)
    }

    fn metadata(&self, path: &Path) -> io::Result<FileMeta> {
        let m = std::fs::metadata(path)?;
        Ok(FileMeta {
            len: m.len(),
            is_dir: m.is_dir(),
            is_file: m.is_file(),
            modified: m.modified().ok(),
        })
    }

    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        std::fs::canonicalize(path)
    }

    fn walk(&self, root: &Path, filters: &WalkFilters) -> io::Result<Vec<PathBuf>> {
        use ignore::WalkBuilder;
        use ignore::overrides::OverrideBuilder;
        use ignore::types::TypesBuilder;

        let mut builder = WalkBuilder::new(root);
        // Bare-default semantics, matching the original tool walkers: hidden
        // files skipped, gitignore/global-exclude respected.
        builder
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true);

        if let Some(depth) = filters.max_depth {
            builder.max_depth(Some(depth));
        }

        for glob in &filters.include {
            let mut overrides = OverrideBuilder::new(root);
            overrides
                .add(glob)
                .map_err(|e| io::Error::other(format!("invalid glob pattern {glob}: {e}")))?;
            builder.overrides(
                overrides
                    .build()
                    .map_err(|e| io::Error::other(format!("glob override build failed: {e}")))?,
            );
        }

        if !filters.types.is_empty() {
            let mut types = TypesBuilder::new();
            types.add_defaults();
            for ty in &filters.types {
                types.select(ty);
            }
            builder.types(
                types
                    .build()
                    .map_err(|e| io::Error::other(format!("type matcher build failed: {e}")))?,
            );
        }

        let mut files = Vec::new();
        for entry in builder.build().flatten() {
            if entry.file_type().is_some_and(|ft| ft.is_file()) {
                files.push(entry.into_path());
                if filters.max_files.is_some_and(|cap| files.len() >= cap) {
                    break;
                }
            }
        }
        Ok(files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn std_fs_roundtrip_and_walk() {
        let dir = std::env::temp_dir().join(format!("chatshell-stdfs-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let file = dir.join("src").join("a.rs");
        StdFs.write(&file, b"fn main() {}").unwrap();

        assert_eq!(StdFs.read(&file).unwrap(), b"fn main() {}");
        let meta = StdFs.metadata(&file).unwrap();
        assert!(meta.is_file && meta.len == 12 && meta.modified.is_some());

        let found = StdFs
            .walk(
                &dir,
                &WalkFilters {
                    include: vec!["*.rs".into()],
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(found, vec![file.clone()]);

        let _ = StdFs.remove_file(&file);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
