# Shopping Sessions: Rust implementation. Static musl binary in a scratch image.
# The scorer runs CMD with no arguments, as uid 1000, read-only root, no network.
FROM rust:1.100.0-bookworm@sha256:ade39a710ef85f3506cffcfec3e8a24ba090badef1dff2fb798f71d878833fea AS build
RUN apt-get update && apt-get install -y --no-install-recommends musl-tools \
 && rustup target add x86_64-unknown-linux-musl
WORKDIR /src
COPY impl/rs/Cargo.toml impl/rs/Cargo.lock ./
COPY impl/rs/src ./src
COPY impl/launcher /launcher
# Two builds of the same source: x86-64-v3 (AVX2, runs anywhere the old image ran) and an AVX-512 one
# (256-bit vectors preferred, AVX-512 BW/VL/DQ mask and ternary ops). The launcher at /app/shop
# picks one at start-up from CPUID/XGETBV and execs it; any doubt selects the AVX2 build.
RUN CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3" \
    cargo build --release --locked --target x86_64-unknown-linux-musl \
 && cp target/x86_64-unknown-linux-musl/release/shop /shop-base
RUN CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+prefer-256-bit" \
    cargo build --release --locked --target x86_64-unknown-linux-musl --target-dir /target-wide \
 && cp /target-wide/x86_64-unknown-linux-musl/release/shop /shop-wide
RUN cd /launcher && cargo build --release --locked --target x86_64-unknown-linux-musl \
 && cp target/x86_64-unknown-linux-musl/release/launcher /shop

FROM scratch
COPY --from=build /shop /app/shop
COPY --from=build /shop-base /app/shop-base
COPY --from=build /shop-wide /app/shop-wide
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch
CMD ["/app/shop"]
