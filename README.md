# ChatShell Agent

Shared Rust core that powers ChatShell's LLM agent across desktop and iOS. Built on [rig-core](https://github.com/0xPlaygrounds/rig), it unifies provider routing, streaming, reasoning-content parsing, and client-side tool calling, and holds the shared tool engines, permission and sandbox policy, sync, and attachment handling, so both platforms stay behaviorally aligned.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

## Crates

| Crate | Description |
| --- | --- |
| [`chatshell-agent-core`](crates/chatshell-agent-core) | Provider-agnostic agent library: streaming generation, reasoning dedup, client-side tool calling, plus the shared tools, permissions, sandbox, sync, and attachment handling described below |
| [`chatshell-agent-ffi`](crates/chatshell-agent-ffi) | UniFFI bindings exposing the core to Swift/iOS as an XCFramework |

## Core modules

| Area | Modules |
| --- | --- |
| Generation | `agent`, `openai_compat`, `thinking`, `history`, `types` |
| Providers | `provider_registry`, `provider_catalog` |
| Tools | `tools` (tool engines over the `tool_fs` and `process_api` seams), `tool_definitions`, `edit_anchor`, `anchor_store`, `fuzzy_match`, `web_search`, `web_fetch`, `mcp` |
| Permissions | `permission` (approval gate decisions), `grant_keys`, `permission_store` (remembered approvals), `path_policy`, `net_policy`, `bash_security`, `bash_ast` |
| Sandbox | `sandbox` (macOS Seatbelt, Linux bubblewrap), `shell_env` |
| Auto-mode review | `judgment` (judge requests and ruleset) |
| File history | `file_history` (backups behind undoing a turn's file changes) |
| Attachments and media | `image_fit`, `tool_images`, `pdf_text`, `pdf_images`, `office`, `video` (optional ffmpeg) |
| Visual output | `visual_output` (system-prompt routing for charts, figures, diagrams, and math), `svg_figure` |
| Context | `compaction`, `snapcompact`, `context_pruning` |
| Sync and storage | `sync`, `sync_schema`, `sync_crypto`, `crypto` (API key encryption), `tokenizer` (CJK-aware FTS) |

## Provider support

| Provider | API surface |
| --- | --- |
| OpenAI | Responses API (default) or Chat Completions via `api_style = "chat_completions"` |
| Anthropic | Messages API |
| OpenRouter | OpenAI-compatible |
| Ollama | Local |
| Custom OpenAI-compatible | Chat Completions |

## Architecture

`run_generation(request, sender, cancellation)` streams [`AgentEvent`](crates/chatshell-agent-core/src/types.rs) values over a tokio `mpsc::Sender`:

- `TextDelta` / `ReasoningDelta` - streamed output; reasoning is deduped across providers
- `ToolCallStarted` / `ToolCallCompleted` - client-side tool calling (tools execute on the host, not inside the agent)
- `Image` - inline image data URL
- `Completed` - final content, reasoning, and token usage

`openai_compat` provides a reasoning-dedup streaming `CompletionModel` for OpenAI-compatible endpoints, keeping thinking-tag handling identical to Anthropic. `thinking` exposes `parse_thinking_content` for reasoning-tag extraction.

## Consumption

**Desktop (Rust)** - path dependency in `chatshell-desktop/src-tauri/Cargo.toml`:

```toml
chatshell-agent-core = { path = "../../chatshell-agent/crates/chatshell-agent-core" }
```

**iOS (Swift)** - build the XCFramework and bind via UniFFI:

```bash
./scripts/build-apple.sh
```

Produces `chatshell-ios/Vendor/ChatShellAgentFFI.xcframework` plus Swift bindings in `chatshell-ios/Sources/ChatShellAgentBridge/Generated/`. Requires iOS 17+ deployment target.

## Build

Requires Rust with iOS targets installed:

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
```

```bash
# Core library only
cargo build -p chatshell-agent-core

# FFI + Swift bindings + XCFramework (device + simulator + x86_64 sim)
./scripts/build-apple.sh
```

## License

[Apache License 2.0](LICENSE)

"ChatShell" and the ChatShell logo are trademarks of Seedvector LLC.
