#!/usr/bin/env bash
# Download a set of open-licensed (OFL) handwriting fonts from Google Fonts.
# These augment the OCR training set so the classifier generalizes to real
# handwriting instead of only printed type. Font binaries are NOT committed
# (see .gitignore); this script makes the set reproducible.
set -euo pipefail

DEST="${1:-assets/handwriting_fonts}"
mkdir -p "$DEST"
BASE="https://raw.githubusercontent.com/google/fonts/main/ofl"

# name:path-under-ofl
FONTS=(
  "PatrickHand:patrickhand/PatrickHand-Regular.ttf"
  "IndieFlower:indieflower/IndieFlower-Regular.ttf"
  "ShadowsIntoLight:shadowsintolight/ShadowsIntoLight-Regular.ttf"
  "ArchitectsDaughter:architectsdaughter/ArchitectsDaughter-Regular.ttf"
  "GloriaHallelujah:gloriahallelujah/GloriaHallelujah-Regular.ttf"
  "Kalam:kalam/Kalam-Regular.ttf"
  "PermanentMarker:permanentmarker/PermanentMarker-Regular.ttf"
  "ReenieBeanie:reeniebeanie/ReenieBeanie-Regular.ttf"
  "HomemadeApple:homemadeapple/HomemadeApple-Regular.ttf"
  "RockSalt:rocksalt/RockSalt-Regular.ttf"
  "ComingSoon:comingsoon/ComingSoon-Regular.ttf"
  "Schoolbell:schoolbell/Schoolbell-Regular.ttf"
  "CoveredByYourGrace:coveredbyyourgrace/CoveredByYourGrace-Regular.ttf"
  "Handlee:handlee/Handlee-Regular.ttf"
  "Neucha:neucha/Neucha.ttf"
  "GochiHand:gochihand/GochiHand-Regular.ttf"
  "JustAnotherHand:justanotherhand/JustAnotherHand-Regular.ttf"
  "Caveat:caveat/Caveat%5Bwght%5D.ttf"
)

ok=0; fail=0
for entry in "${FONTS[@]}"; do
  name="${entry%%:*}"; path="${entry#*:}"
  out="$DEST/${name}.ttf"
  if [ -f "$out" ]; then ok=$((ok+1)); continue; fi
  if curl -sfL -o "$out" "$BASE/$path"; then
    echo "  ok   $name"; ok=$((ok+1))
  else
    echo "  FAIL $name ($path)"; rm -f "$out"; fail=$((fail+1))
  fi
done
echo "downloaded $ok fonts to $DEST ($fail failed)"
