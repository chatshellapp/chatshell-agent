#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IOS_DIR="${ROOT_DIR}/../chatshell-ios"
BUILD_DIR="${ROOT_DIR}/build/apple"
BINDINGS_DIR="${BUILD_DIR}/bindings"
HEADERS_DIR="${BUILD_DIR}/headers"
XCFRAMEWORK_PATH="${IOS_DIR}/Vendor/ChatShellAgentFFI.xcframework"
SWIFT_BINDINGS_DIR="${IOS_DIR}/Sources/ChatShellAgentBridge/Generated"
LIB_NAME="libchatshell_agent_ffi.a"

export IPHONEOS_DEPLOYMENT_TARGET=17.0
export CMAKE_OSX_DEPLOYMENT_TARGET=17.0

rm -rf "${BUILD_DIR}" "${XCFRAMEWORK_PATH}"
mkdir -p "${BINDINGS_DIR}" "${HEADERS_DIR}" "${SWIFT_BINDINGS_DIR}" "${IOS_DIR}/Vendor"

cd "${ROOT_DIR}"
cargo build -p chatshell-agent-ffi --release
cargo run -p chatshell-agent-ffi --bin uniffi-bindgen -- \
    generate \
    --library "${ROOT_DIR}/target/release/libchatshell_agent_ffi.dylib" \
    --language swift \
    --out-dir "${BINDINGS_DIR}"

cp "${BINDINGS_DIR}/chatshell_agent_ffiFFI.h" "${HEADERS_DIR}/"
cp "${BINDINGS_DIR}/chatshell_agent_ffiFFI.modulemap" "${HEADERS_DIR}/module.modulemap"
cp "${BINDINGS_DIR}/chatshell_agent_ffi.swift" "${SWIFT_BINDINGS_DIR}/"

cargo build -p chatshell-agent-ffi --release --target aarch64-apple-ios
cargo build -p chatshell-agent-ffi --release --target aarch64-apple-ios-sim
cargo build -p chatshell-agent-ffi --release --target x86_64-apple-ios

lipo -create \
    "${ROOT_DIR}/target/aarch64-apple-ios-sim/release/${LIB_NAME}" \
    "${ROOT_DIR}/target/x86_64-apple-ios/release/${LIB_NAME}" \
    -output "${BUILD_DIR}/${LIB_NAME}"

xcodebuild -create-xcframework \
    -library "${ROOT_DIR}/target/aarch64-apple-ios/release/${LIB_NAME}" \
    -headers "${HEADERS_DIR}" \
    -library "${BUILD_DIR}/${LIB_NAME}" \
    -headers "${HEADERS_DIR}" \
    -output "${XCFRAMEWORK_PATH}"

echo "Generated ${XCFRAMEWORK_PATH}"
