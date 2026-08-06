#!/usr/bin/env bash
# API smoke tests: health, auth, optional image generation.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8765}"
API_KEY="${DREAMSHAPER_API_KEY:-}"
RUN_GENERATION="${RUN_GENERATION:-1}"
GENERATION_STEPS="${GENERATION_STEPS:-4}"
OUT_DIR="${OUT_DIR:-/tmp/dreamshaper-smoke}"

mkdir -p "${OUT_DIR}"

echo "== smoke: health (no auth required)"
health_json="$(curl -fsS "${BASE_URL}/health")"
echo "${health_json}"
[[ "$(echo "${health_json}" | jq -r '.status')" == "ok" ]]

if [[ -z "${API_KEY}" ]]; then
    echo "== smoke: DREAMSHAPER_API_KEY unset — skipping auth/generation checks"
    exit 0
fi

echo "== smoke: generation without auth (expect 401)"
code="$(curl -s -o "${OUT_DIR}/no-auth.json" -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"smoke test","steps":'"${GENERATION_STEPS}"'}')"
[[ "${code}" == "401" ]]
jq -e '.error.code == "invalid_api_key"' "${OUT_DIR}/no-auth.json" >/dev/null

echo "== smoke: generation with wrong key (expect 401)"
code="$(curl -s -o "${OUT_DIR}/bad-key.json" -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/images/generations" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer wrong-key" \
    -d '{"prompt":"smoke test","steps":'"${GENERATION_STEPS}"'}')"
[[ "${code}" == "401" ]]

if [[ "${RUN_GENERATION}" != "1" ]]; then
    echo "== smoke: auth checks OK (generation skipped)"
    exit 0
fi

echo "== smoke: generation with valid key (steps=${GENERATION_STEPS})"
png_out="${OUT_DIR}/smoke.png"
code="$(curl -s -o "${OUT_DIR}/gen.json" -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/images/generations" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d '{"prompt":"a red circle on white background","steps":'"${GENERATION_STEPS}"',"seed":42}')"
[[ "${code}" == "200" ]]
jq -e '.data[0].b64_json != null' "${OUT_DIR}/gen.json" >/dev/null
jq -r '.data[0].b64_json' "${OUT_DIR}/gen.json" | base64 -d > "${png_out}"
size="$(wc -c < "${png_out}")"
[[ "${size}" -gt 1000 ]]
echo "== smoke: saved ${png_out} (${size} bytes), seed=$(jq -r '.data[0].seed' "${OUT_DIR}/gen.json")"

echo "== smoke: all checks passed"
