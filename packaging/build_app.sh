#!/usr/bin/env bash
# Build, sign (ad-hoc) and zip `Socratic AI.app`.
#
#   packaging/build_app.sh            # full build with on-device transcription
#   SOCRATIC_BUNDLE_LOCAL=0 packaging/build_app.sh   # smaller bundle, API/none transcription only
#
# Needs: uv, Xcode Command Line Tools. Output: dist/Socratic AI.app and
# dist/SocraticAI-<version>-macos-arm64.zip
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="$(uv run --no-sync python -c 'import tomllib;print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')"
export SOCRATIC_VERSION="$VERSION"
export SOCRATIC_BUNDLE_LOCAL="${SOCRATIC_BUNDLE_LOCAL:-1}"
APP="dist/Socratic AI.app"
ZIP="dist/SocraticAI-${VERSION}-macos-arm64.zip"

echo "==> syncing deps (local transcription: ${SOCRATIC_BUNDLE_LOCAL})"
if [ "$SOCRATIC_BUNDLE_LOCAL" = "1" ]; then
  uv sync --extra app --extra local --extra dev
else
  uv sync --extra app --extra dev
fi

echo "==> pyinstaller"
rm -rf build "dist/Socratic AI" "$APP" "$ZIP"
uv run --no-sync pyinstaller --noconfirm --clean --log-level WARN packaging/socratic_ai.spec

if find "$APP" -name 'libtorch*' | grep -q .; then
  echo "torch leaked into the bundle; fix the excludes" >&2; exit 1
fi

echo "==> codesign (ad-hoc)"
codesign --force --deep --options runtime \
  --entitlements packaging/entitlements.plist -s - "$APP"
codesign --verify --deep --strict "$APP"

echo "==> smoke test"
SOCRATIC_CONFIG_DIR="$(mktemp -d)" SOCRATIC_DATA_DIR="$(mktemp -d)" \
  "$APP/Contents/MacOS/Socratic AI" --selftest

echo "==> zip"
ditto -c -k --keepParent "$APP" "$ZIP"
du -sh "$APP" "$ZIP"
