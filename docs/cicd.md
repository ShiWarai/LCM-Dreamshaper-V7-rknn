# CI/CD

Workflows live in [`.github/workflows/`](../.github/workflows/). Prod images are built **only** for `linux/arm64` (RK3588).

## Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **Deploy** (`deploy.yml`) | Push to `main` / `dev`, manual run | clippy + cargo test in dev image |
| **Deploy → prerelease** | Push to `dev` with `[prerelease]` in commit message, or manual `publish_prerelease` flag | Publish `:prerelease` to GHCR |
| **Publish** (`publish.yml`) | Successful Deploy on `main` | Publish `:main` to GHCR |

## GHCR image

```
ghcr.io/shiwarai/lcm-dreamshaper-v7-rs
```

Tags: `:main`, `:prerelease`, `:<sha>`.

## Prod vs prerelease vs local build

| Method | Compose | Image |
|--------|---------|-------|
| **Local build** | `docker compose up -d --build` | `dreamshaper-api:latest` |
| **Prerelease** | `-f docker-compose.yml -f docker-compose.prerelease.yml` | `:prerelease` |
| **Prod** | `-f docker-compose.yml -f docker-compose.prod.yml` | `:main` |

### `:prerelease` — test candidate

```bash
git commit -m "feat: API update [prerelease]"
git push origin dev
```

On the staging host:

```bash
docker pull ghcr.io/shiwarai/lcm-dreamshaper-v7-rs:prerelease
docker compose -f docker-compose.yml -f docker-compose.prerelease.yml up -d
```

### `:main` — production

```bash
docker pull ghcr.io/shiwarai/lcm-dreamshaper-v7-rs:main
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Build cache

- **prerelease** — GHA cache scope `lcm-dreamshaper-prerelease`
- **main** — scope `lcm-dreamshaper-main`
- **test** (dev image) — scope `lcm-dreamshaper-dev`

## Local tests (same as CI)

```bash
docker compose -f docker-compose.dev.yml build dev
docker compose -f docker-compose.dev.yml run --rm -T dev \
  sh -c 'cargo clippy --all-targets -- -D warnings && cargo test --lib -- --nocapture'
```

## Telegram notifications

Repository secrets (Settings → Secrets and variables → Actions):

| Secret | Purpose |
|--------|---------|
| `TELEGRAM_TOKEN` | Bot token |
| `TELEGRAM_TO` | Chat ID |

Without secrets, notification steps do not fail (`continue-on-error: true`). Successful events are silent (`disable_notification`).

## Repository requirements

- **GitHub Actions** and **Packages** (GHCR) enabled.
- `third_party/librknnrt.so` must be in git (for prod builds in CI).
- RKNN models are **not** in the image — mounted via volume `LCM_MODELS_DIR:/models`.

## Self-hosted runner

Not required: prod builds run on `ubuntu-latest` via QEMU + Buildx (`platforms: linux/arm64`).
