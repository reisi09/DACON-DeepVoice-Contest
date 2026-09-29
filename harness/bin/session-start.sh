#!/bin/bash
# Cloud sessions only: install the harness toolchain (Bun >= 1.3.14, omp, the jevmem CLI) unless the
# pinned versions are already there. The environment cache keeps them, so later sessions skip the installs.
# Install output goes to stderr: a SessionStart hook's stdout is added to the model's context.
set -euo pipefail
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

BUN_VERSION=1.4.2      # omp 18.x needs Bun >= 1.3.14; the image ships 1.3.11
OMP_VERSION=18.4.3
JEVMEM_VERSION=0.5.9   # >= 0.5.8: older CLIs sent PGPASSWORD= and "password": values unscrubbed

version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

if ! version_ge "$(bun --version 2>/dev/null || echo 0)" 1.3.14; then
  npm install -g --no-fund --no-audit --loglevel=error "bun@$BUN_VERSION" >&2
fi
if [ "$(omp --version 2>/dev/null)" != "omp/$OMP_VERSION" ]; then
  bun install -g "@oh-my-pi/pi-coding-agent@$OMP_VERSION" >&2
fi
if [ "$(jevmem --version 2>/dev/null)" != "$JEVMEM_VERSION" ]; then
  npm install -g --no-fund --no-audit --loglevel=error "jevmem@$JEVMEM_VERSION" >&2
fi
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export PATH="$HOME/.bun/bin:$PATH"' >> "$CLAUDE_ENV_FILE"
fi
