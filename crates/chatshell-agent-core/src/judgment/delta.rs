//! The per-execution jail delta a judgment allow authorizes (AR-14).
//!
//! An allow authorizes the action as the evidence described it. For
//! sandboxed bash that can mean more than the session profile permits:
//! network for detected egress, write access for targets outside the
//! writable roots, or for deleting an existing path outside the
//! workspace. The delta covers exactly one execution and is never a
//! grant (INV-A1).
//!
//! Roots are bounded: a delta never makes the filesystem root, the home
//! directory, or an ancestor of either writable — such an allow still
//! executes, and the jail denial surfaces as a `jail-denied` marker the
//! user can run anyway.

use super::JudgmentRequest;
use super::evidence::NetworkClass;
use crate::bash_security;
use crate::permission::GateContext;
use crate::tools::bash_sandbox::ExecutionDelta;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The delta an allow of `request` (a bash call with `args`) authorizes
/// under `ctx`. `None` when nothing beyond the session profile is needed,
/// or when bash is not jailed.
pub fn execution_delta(
    request: &JudgmentRequest,
    args: &Value,
    ctx: &GateContext,
) -> Option<ExecutionDelta> {
    if request.evidence.tool != "bash" {
        return None;
    }
    let egress = matches!(request.evidence.network, NetworkClass::Egress { .. });
    escape_delta(args, ctx, egress).delta
}

/// What approving a bash call's escapes authorizes (ADR 08 §1): the
/// per-execution delta, and the outside write targets no bounded delta can
/// open (they run jailed and surface a denial).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EscapePlan {
    pub delta: Option<ExecutionDelta>,
    pub unbounded: Vec<PathBuf>,
}

/// The escapes of bash call `args` under `ctx`, for a judge's allow or a
/// human's approval alike. `egress`: the call contacts the network (the
/// delta opens it unless the session flag already does).
pub fn escape_delta(args: &Value, ctx: &GateContext, egress: bool) -> EscapePlan {
    let Some(command) = args.get("command").and_then(Value::as_str) else {
        return EscapePlan::default();
    };
    if !ctx.sandbox_available {
        return EscapePlan::default();
    }
    let workdir = args.get("workdir").and_then(Value::as_str);
    let bash_cwd = ctx.bash_cwd(workdir);
    let cwd = bash_cwd.as_str();
    let root = ctx.analysis_root();
    let scratch: Vec<PathBuf> = ctx.scratch_dir.iter().cloned().collect();

    let network = egress && !ctx.network_allowed;

    let mut roots: Vec<PathBuf> = Vec::new();
    let mut unbounded: Vec<PathBuf> = Vec::new();
    for target in bash_security::outside_write_targets(command, root, cwd, &scratch) {
        match target
            .parent()
            .and_then(existing_dir)
            .filter(|r| boundable(r))
        {
            Some(dir) => push_root(&mut roots, dir),
            None => unbounded.push(target),
        }
    }
    if let Some(target) = bash_security::outside_deletion_target(command, root, cwd, &scratch) {
        let dir = if target.is_dir() && DIRECTORY_ROOT_DELETABLE {
            existing_dir(&target)
        } else {
            target.parent().and_then(existing_dir)
        };
        match dir.filter(|r| boundable(r)) {
            Some(dir) => push_root(&mut roots, dir),
            None => unbounded.push(target),
        }
    }
    let delta = ExecutionDelta {
        network,
        writable_roots: roots,
        unsandboxed: false,
    };
    EscapePlan {
        delta: (!delta.is_empty()).then_some(delta),
        unbounded,
    }
}

/// Whether a directory opened as a writable root can itself be removed.
/// Seatbelt's `subpath` rule covers the directory entry, so on macOS the
/// deleted directory is the root. bubblewrap bind-mounts each root onto
/// itself, and a mount point cannot be removed (`EBUSY`), so on Linux the
/// root is the directory's parent.
const DIRECTORY_ROOT_DELETABLE: bool = !cfg!(target_os = "linux");

fn existing_dir(path: &Path) -> Option<PathBuf> {
    let canonical = std::fs::canonicalize(path).ok()?;
    canonical.is_dir().then_some(canonical)
}

/// Whether `root` is narrow enough to open for one execution: never the
/// filesystem root, the home directory, or an ancestor of either.
fn boundable(root: &Path) -> bool {
    if root.parent().is_none() {
        return false;
    }
    let home = dirs::home_dir().and_then(|home| std::fs::canonicalize(home).ok());
    !home.as_deref().is_some_and(|home| home.starts_with(root))
}

/// Add a boundable `root` once.
fn push_root(roots: &mut Vec<PathBuf>, root: PathBuf) {
    if boundable(&root) && !roots.contains(&root) {
        roots.push(root);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judgment::{Evidence, JudgmentSource};
    use serde_json::json;

    fn request(args: &Value, ctx: &GateContext, source: JudgmentSource) -> JudgmentRequest {
        JudgmentRequest {
            subject: String::new(),
            source,
            forced: None,
            evidence: Evidence::collect("bash", args, ctx),
        }
    }

    fn fixture() -> (tempfile::TempDir, GateContext, PathBuf) {
        let dir = tempfile::tempdir_in(dirs::home_dir().unwrap()).unwrap();
        let base = std::fs::canonicalize(dir.path()).unwrap();
        let ws = base.join("ws");
        let outside = base.join("outside");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let ctx = GateContext {
            workspace: Some(ws),
            sandbox_available: true,
            network_allowed: false,
            scratch_dir: None,
            cwd_drifted: false,
        };
        (dir, ctx, outside)
    }

    #[test]
    fn egress_gets_network_for_one_execution() {
        let (_dir, ctx, _) = fixture();
        let args = json!({ "command": "curl https://example.com" });
        let delta =
            execution_delta(&request(&args, &ctx, JudgmentSource::Network), &args, &ctx).unwrap();
        assert!(delta.network);
        assert!(delta.writable_roots.is_empty());
        // Already enabled for the session: no delta needed.
        let session = GateContext {
            network_allowed: true,
            ..ctx.clone()
        };
        assert_eq!(
            execution_delta(
                &request(&args, &session, JudgmentSource::Network),
                &args,
                &session
            ),
            None
        );
    }

    #[test]
    fn out_of_root_writes_get_their_parent_directory() {
        let (_dir, ctx, outside) = fixture();
        let args = json!({ "command": format!("echo hi > {}/a.txt", outside.display()) });
        let delta = execution_delta(
            &request(&args, &ctx, JudgmentSource::OutOfWorkspaceWrite),
            &args,
            &ctx,
        )
        .unwrap();
        assert_eq!(delta.writable_roots, vec![outside]);
        assert!(!delta.network);
    }

    #[test]
    fn deletions_get_a_root_that_can_remove_the_directory() {
        let (_dir, ctx, outside) = fixture();
        let args = json!({ "command": format!("rm -rf {}", outside.display()) });
        let delta = execution_delta(
            &request(&args, &ctx, JudgmentSource::OutsideDeletion),
            &args,
            &ctx,
        )
        .unwrap();
        // Linux mounts the root onto itself, so only its parent lets the
        // directory entry go.
        let expected = if cfg!(target_os = "linux") {
            outside.parent().unwrap().to_path_buf()
        } else {
            outside
        };
        assert_eq!(delta.writable_roots, vec![expected]);
    }

    #[test]
    fn home_and_its_ancestors_are_never_opened() {
        let (_dir, ctx, _) = fixture();
        let home = dirs::home_dir().unwrap();
        let args = json!({ "command": format!("echo hi > {}/x.txt", home.display()) });
        assert_eq!(
            execution_delta(
                &request(&args, &ctx, JudgmentSource::OutOfWorkspaceWrite),
                &args,
                &ctx
            ),
            None
        );
    }

    #[test]
    fn no_delta_without_a_jail_or_for_other_tools() {
        let (_dir, ctx, outside) = fixture();
        let unjailed = GateContext {
            sandbox_available: false,
            ..ctx.clone()
        };
        let args = json!({ "command": format!("echo hi > {}/a.txt", outside.display()) });
        assert_eq!(
            execution_delta(
                &request(&args, &unjailed, JudgmentSource::UnsandboxedBash),
                &args,
                &unjailed
            ),
            None
        );
        let write = JudgmentRequest {
            subject: String::new(),
            source: JudgmentSource::OutOfWorkspaceWrite,
            forced: None,
            evidence: Evidence::collect("write", &json!({ "path": "/x" }), &ctx),
        };
        assert_eq!(
            execution_delta(&write, &json!({ "path": "/x" }), &ctx),
            None
        );
    }
}
