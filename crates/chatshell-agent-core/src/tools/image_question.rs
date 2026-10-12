//! Vision-model question answering for image reads (host-injected seam).
//!
//! The core crate owns no LLM client that accepts image content, so hosts
//! inject an [`ImageQuestionResponder`] into [`super::ReadTool`]: `read`
//! with a `question` argument loads the image, hands it to the responder,
//! and returns the answer as text instead of inlining pixels. The answer
//! replaces the image in context (cheaper than raw image tokens) and works
//! even when the conversation's main model cannot accept images at all.

use futures::future::BoxFuture;

/// An image loaded by the read tool, ready for a one-shot vision call.
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// Raw image bytes (already size-capped by the read tool).
    pub bytes: Vec<u8>,
    /// MIME type (e.g. `image/png`).
    pub mime_type: String,
}

/// Host-provided one-shot vision-model call.
///
/// Implementations resolve a vision-capable model themselves and return the
/// answer text for the given question. Errors are surfaced to the caller as
/// a failed image question (the read tool adds recovery guidance).
pub trait ImageQuestionResponder: Send + Sync {
    /// Answer `question` about `image`.
    fn answer(
        &self,
        image: ImageInput,
        question: String,
    ) -> BoxFuture<'static, Result<String, String>>;
}
