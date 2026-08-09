# LCM Dreamshaper V7 — multi-stage prod image for RK3588 (linux/arm64).
# Requires third_party/librknnrt.so in git (RKNN runtime 2.3.2).

FROM rust:1-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    clang \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src/
COPY third_party/librknnrt.so ./third_party/

ENV RKNNRT_LIB_DIR=/build/third_party

RUN cargo build --release --locked

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY third_party/librknnrt.so /usr/lib/librknnrt.so
RUN ldconfig

WORKDIR /opt/dreamshaper

COPY --from=builder /build/target/release/dreamshaper-cli /opt/dreamshaper/dreamshaper-cli
COPY scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh
RUN chmod +x ./scripts/docker-entrypoint.sh ./dreamshaper-cli

ENV LCM_MODELS_DIR=/models \
    PORT=8765 \
    HOST=0.0.0.0 \
    LD_LIBRARY_PATH=/usr/lib

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["./scripts/docker-entrypoint.sh"]
