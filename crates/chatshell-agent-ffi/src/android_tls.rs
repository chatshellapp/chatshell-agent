//! Android-specific TLS initialization for rustls-platform-verifier.
//!
//! reqwest's rustls backend uses rustls-platform-verifier for certificate
//! verification, which on Android needs the JVM class loader and application
//! context registered before the first TLS handshake. The Android app calls
//! `AgentFfi.androidInit(context)` right after `System.loadLibrary`.

use jni::objects::{JClass, JObject};
use jni::sys::jboolean;
use jni::{Env, EnvUnowned};

/// JNI entry point: `com.chatshell.app.ffi.AgentFfi.androidInit(context: Context): Boolean`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_chatshell_app_ffi_AgentFfi_androidInit<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    context: JObject<'local>,
) -> jboolean {
    let outcome = env.with_env(|env: &mut Env<'local>| -> jni::errors::Result<()> {
        rustls_platform_verifier::android::init_with_env(env, context)
    });
    match outcome.into_outcome() {
        jni::Outcome::Ok(()) => true,
        jni::Outcome::Err(error) => {
            eprintln!("rustls-platform-verifier init failed: {error}");
            false
        }
        jni::Outcome::Panic(_) => false,
    }
}
