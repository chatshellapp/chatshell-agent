# ChatShell Agent

Shared Rust core that powers ChatShell's LLM agent across desktop and iOS. Built on [rig-core](https://github.com/0xPlaygrounds/rig), it unifies provider routing, streaming, reasoning-content parsing, and client-side tool calling so both platforms stay behaviorally aligned.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

## Crates

| Crate | Description |
| --- | --- |
| [`chatshell-agent-core`](crates/chatshell-agent-core) | Provider-agnostic agent library: streaming generation, reasoning dedup, client-side tool calling, image attachments |
| [`chatshell-agent-ffi`](crates/chatshell-agent-ffi) | UniFFI bindings exposing the core to Swift/iOS as an XCFramework |

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

The workspace patches `rig-core` from a local path (`../chatshell-desktop/patches/rig-core`).

## License

[Apache License 2.0](LICENSE)

"ChatShell" and the ChatShell logo are trademarks of Seedvector LLC.
