# CI/CD

Workflows live in [`.github/workflows/`](../.github/workflows/). Prod images are built **only** for `linux/arm64` (RK3588).

## Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **Deploy** (`deploy.yml`) | Push to `master` / `dev`, manual run | clippy + cargo test in dev image |
| **Deploy → prerelease** | Push to `dev` with `[prerelease]` in commit message, or manual `publish_prerelease` flag | Publish `:prerelease` to GHCR |
| **Publish** (`publish.yml`) | Successful Deploy on `master` | Publish `:master` to GHCR |

## GHCR image

```
ghcr.io/shiwarai/lcm-dreamshaper-v7-rknn
```

Tags: `:master`, `:prerelease`, `:<sha>`.

## Prod vs prerelease vs local build

| Method | Compose | Image |
|--------|---------|-------|
| **Local build** | `docker compose up -d --build` | `dreamshaper-api:latest` |
| **Prerelease** | `-f docker-compose.yml -f docker-compose.prerelease.yml` | `:prerelease` |
| **Prod** | `-f docker-compose.yml -f docker-compose.prod.yml` | `:master` |

### `:prerelease` — test candidate

```bash
git commit -m "feat: API update [prerelease]"
git push origin dev
```

On the staging host:

```bash
docker pull ghcr.io/shiwarai/lcm-dreamshaper-v7-rknn:prerelease
docker compose -f docker-compose.yml -f docker-compose.prerelease.yml up -d
```

### `:master` — production

```bash
docker pull ghcr.io/shiwarai/lcm-dreamshaper-v7-rknn:master
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Build cache

- **prerelease** — GHA cache scope `lcm-dreamshaper-v7-rknn-prerelease`
- **master** — scope `lcm-dreamshaper-v7-rknn-master`
- **test** (dev image) — scope `lcm-dreamshaper-v7-rknn-dev`

## Local tests (same as CI)

Requires an **arm64** host (or arm64 Docker). `third_party/librknnrt.so` is aarch64 and will not link on x86_64.

```bash
docker compose -f docker-compose.dev.yml build dev
docker compose -f docker-compose.dev.yml run --rm -T --user 0:0 dev \
  sh -c 'cargo clippy --all-targets -- -D warnings && cargo test --lib -- --nocapture'
```

CI runs on GitHub-hosted **`ubuntu-24.04-arm`** (native aarch64).

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

## Runners

All compile/build jobs use GitHub-hosted **`ubuntu-24.04-arm`** (native aarch64, no QEMU):

| Job | Workflow |
|-----|----------|
| Tests + dev image | `deploy.yml` → `test` |
| `:prerelease` image | `deploy.yml` → `publish-prerelease` |
| `:master` image | `publish.yml` → `publish` |

Telegram notify jobs stay on `ubuntu-latest` (no Docker build).

Self-hosted runners are not required.
