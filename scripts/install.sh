#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
if [[ -n "${HERMES_PROFILE:-}" ]]; then
  TARGET_DIR="$HERMES_HOME_DIR/profiles/${HERMES_PROFILE}/plugins/hermes-lcm"
else
  TARGET_DIR="$HERMES_HOME_DIR/plugins/hermes-lcm"
fi

mkdir -p "$(dirname "$TARGET_DIR")"

if [[ -L "$TARGET_DIR" ]]; then
  CURRENT_TARGET="$(readlink "$TARGET_DIR")"
  if [[ "$CURRENT_TARGET" != "$REPO_ROOT" ]]; then
    echo "Refusing to replace existing symlink: $TARGET_DIR -> $CURRENT_TARGET" >&2
    echo "Remove it manually or point it at this checkout before rerunning install.sh." >&2
    exit 1
  fi
elif [[ -e "$TARGET_DIR" ]]; then
  echo "Refusing to replace existing path: $TARGET_DIR" >&2
  echo "Move it aside or remove it manually before rerunning install.sh." >&2
  exit 1
else
  ln -s "$REPO_ROOT" "$TARGET_DIR"
fi

cat <<EOF
Installed hermes-lcm at:
  $TARGET_DIR

Activation requires both:

plugins:
  enabled:
    - hermes-lcm

context:
  engine: lcm

Verification:
  1. Restart Hermes.
  2. Run: hermes plugins
  3. Confirm the plugin list includes hermes-lcm and the selected context engine is lcm.
EOF
