#!/usr/bin/env bash
# HandTex environment setup: install the package (+ dev tools) so tests,
# linters and the demo run in a fresh session. Idempotent and quiet.
set -euo pipefail

cd "$(dirname "$0")/../.."

if python3 -c "import fontTools, PIL, numpy, skimage, pytest, torch" >/dev/null 2>&1; then
  echo "HandTex deps already present."
  exit 0
fi

echo "Installing HandTex dependencies..."
python3 -m pip install -q -e ".[dev]" 2>&1 | tail -2 || {
  echo "editable install failed; installing core deps directly" >&2
  python3 -m pip install -q fonttools pillow numpy scikit-image pytest
}
echo "HandTex setup complete."
