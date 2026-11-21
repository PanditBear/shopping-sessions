# Shopping Sessions: Rust implementation. Static musl binary in a scratch image.
# The scorer runs CMD with no arguments, as uid 1000, read-only root, no network.
FROM rust:1-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends musl-tools \
 && rustup target add x86_64-unknown-linux-musl
WORKDIR /src
COPY impl/rs/Cargo.toml impl/rs/Cargo.lock ./
COPY impl/rs/src ./src
# x86-64-v3 (AVX2): the scorer's machines are Cascade Lake or newer.
RUN CC_x86_64_unknown_linux_musl=musl-gcc RUSTFLAGS="-C target-cpu=x86-64-v3" \
    cargo build --release --locked --target x86_64-unknown-linux-musl \
 && cp target/x86_64-unknown-linux-musl/release/shop target/x86_64-unknown-linux-musl/release/probe /

FROM scratch
COPY --from=build /shop /app/shop
COPY --from=build /probe /app/probe
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch
CMD ["/app/probe"]
