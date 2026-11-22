# Shopping Sessions: Rust implementation. Static musl binary in a scratch image.
# The scorer runs CMD with no arguments, as uid 1000, read-only root, no network.
FROM rust:1.100.0-bookworm@sha256:ade39a710ef85f3506cffcfec3e8a24ba090badef1dff2fb798f71d878833fea AS build
RUN apt-get update && apt-get install -y --no-install-recommends musl-tools \
 && rustup target add x86_64-unknown-linux-musl
WORKDIR /src
COPY impl/rs/Cargo.toml impl/rs/Cargo.lock ./
COPY impl/rs/src ./src
COPY impl/rs/pgo/merged.profdata /pgo/merged.profdata
# PGO: merged.profdata was recorded with the pinned toolchain above (LLVM 23.1.5) on this source tree.
# x86-64-v3 (AVX2): the scorer's machines are Cascade Lake or newer.
RUN CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3 -C profile-use=/pgo/merged.profdata -C llvm-args=-no-pgo-warn-mismatch" \
    cargo build --release --locked --target x86_64-unknown-linux-musl \
 && cp target/x86_64-unknown-linux-musl/release/shop /shop

FROM scratch
COPY --from=build /shop /app/shop
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch
CMD ["/app/shop"]
