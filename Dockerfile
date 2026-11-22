# Shopping Sessions: Rust implementation. Static musl binary in a scratch image.
# The scorer runs CMD with no arguments, as uid 1000, read-only root, no network.
FROM rust:1.100.0-bookworm@sha256:ade39a710ef85f3506cffcfec3e8a24ba090badef1dff2fb798f71d878833fea AS build
RUN apt-get update && apt-get install -y --no-install-recommends musl-tools \
 && rustup target add x86_64-unknown-linux-musl && rustup component add llvm-tools
WORKDIR /src
COPY impl/rs/Cargo.toml impl/rs/Cargo.lock impl/rs/build.sh ./
COPY impl/rs/pgo ./pgo
COPY impl/rs/src ./src
# x86-64-v3 (AVX2): the scorer's machines are Cascade Lake or newer.
# build.sh: instrumented build, training runs on pgo/*.tar.gz, profile merge, final build with the profile.
RUN sh build.sh /shop

FROM scratch
COPY --from=build /shop /app/shop
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch
CMD ["/app/shop"]
