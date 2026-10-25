#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
if [[ -n "${HERMES_PROFILE:-}" ]]; then
  TARGET_ROOT="$HERMES_HOME_DIR/profiles/${HERMES_PROFILE}"
else
  TARGET_ROOT="$HERMES_HOME_DIR"
fi

PLUGIN_TARGET="$TARGET_ROOT/plugins/hermes-lcm"
SKILL_SOURCE="$REPO_ROOT/skills/hermes-lcm"
SKILL_TARGET="$TARGET_ROOT/skills/hermes-lcm"

preflight_target() {
  local label="$1"
  local target="$2"
  local expected="$3"

  if [[ -L "$target" ]]; then
    local current_target
    current_target="$(readlink "$target")"
    if [[ "$current_target" != "$expected" ]]; then
      if [[ "$label" == "plugin" ]]; then
        echo "Refusing to replace existing symlink: $target -> $current_target" >&2
      else
        echo "Refusing to replace existing skill symlink: $target -> $current_target" >&2
      fi
      echo "Remove it manually or point it at this checkout before rerunning install.sh." >&2
      exit 1
    fi
  elif [[ -e "$target" ]]; then
    if [[ "$label" == "plugin" && -d "$target" ]]; then
      local physical_target
      physical_target="$(cd "$target" && pwd -P)"
      if [[ "$physical_target" == "$expected" ]]; then
        return
      fi
    fi
    if [[ "$label" == "plugin" ]]; then
      echo "Refusing to replace existing path: $target" >&2
    else
      echo "Refusing to replace existing skill path: $target" >&2
    fi
    echo "Move it aside or remove it manually before rerunning install.sh." >&2
    exit 1
  fi
}

preflight_target "plugin" "$PLUGIN_TARGET" "$REPO_ROOT"
preflight_target "skill" "$SKILL_TARGET" "$SKILL_SOURCE"

mkdir -p "$(dirname "$PLUGIN_TARGET")" "$(dirname "$SKILL_TARGET")"

if [[ ! -e "$PLUGIN_TARGET" && ! -L "$PLUGIN_TARGET" ]]; then
  ln -s "$REPO_ROOT" "$PLUGIN_TARGET"
fi
if [[ ! -e "$SKILL_TARGET" && ! -L "$SKILL_TARGET" ]]; then
  ln -s "$SKILL_SOURCE" "$SKILL_TARGET"
fi

cat <<EOF
Installed hermes-lcm at:
  $PLUGIN_TARGET

Discoverable skill:
  $SKILL_TARGET

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
  4. Confirm the available skills include hermes-lcm.
  5. Confirm the WAL generation guard is armed:

       cd "$PLUGIN_TARGET"
       python3 -c "import sys; sys.path.insert(0,'.'); import wal_generation_guard as g; print(g.supported())"

     Must print True. False means the guard is inert: it needs core's
     hermes_state_lockguard, and a failed import is silent by design, so
     lcm.db loses its protection against WAL-generation corruption with no
     error reported anywhere. See
     skills/hermes-lcm/references/wal-generation-corruption.md
EOF
