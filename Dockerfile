# Shopping Sessions: Rust implementation. Static musl binary in a scratch image.
FROM rust:1-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends musl-tools \
 && rustup target add x86_64-unknown-linux-musl
WORKDIR /src
COPY impl/rs/Cargo.toml impl/rs/Cargo.loc[k] ./
COPY impl/rs/src ./src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    CC_x86_64_unknown_linux_musl=musl-gcc \
    cargo build --release --target x86_64-unknown-linux-musl \
 && cp target/x86_64-unknown-linux-musl/release/shop /shop

FROM scratch
COPY --from=build /shop /app/shop
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch
ENTRYPOINT ["/app/shop"]
