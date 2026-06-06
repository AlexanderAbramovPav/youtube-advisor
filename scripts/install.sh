#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "youtube-advisor installer"
echo "========================="

# ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo
  echo "✗ ffmpeg is required but not installed."
  echo "  macOS:  brew install ffmpeg"
  echo "  Linux:  apt install ffmpeg  (or yum/dnf/pacman equivalent)"
  exit 1
fi
echo "✓ ffmpeg found: $(ffmpeg -version 2>/dev/null | head -1 | sed 's/ffmpeg version //;s/ .*//')"

# Deno — recommended for solving YouTube's JS challenges via yt-dlp's
# `--remote-components ejs:github`. Without it, captions extraction will
# fail on many videos with HTTP 429 / "Sign in to confirm you're not a bot".
if ! command -v deno >/dev/null 2>&1; then
  echo
  echo "⚠  deno is not installed."
  echo "   Without it, YouTube may refuse to serve transcripts on some videos."
  echo "   Recommended:"
  echo "     macOS:  brew install deno"
  echo "     Linux:  curl -fsSL https://deno.land/install.sh | sh"
  echo "   Continuing without it; if you hit caption failures, install deno and re-run."
else
  echo "✓ deno found: $(deno --version 2>/dev/null | head -1)"
fi

# uv
if ! command -v uv >/dev/null 2>&1; then
  echo
  echo "Installing uv (Astral's Python package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Ensure uv is on PATH for the rest of this script
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "✓ uv found: $(uv --version)"

# venv
if [ ! -d ".venv" ]; then
  echo
  echo "Creating .venv..."
  uv venv
fi
echo "✓ .venv exists"

# install
echo
echo "Installing youtube-advisor (with dev + whisper extras)..."
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[dev,whisper]"
echo
echo "✓ Installed. Verify with:"
echo "    source .venv/bin/activate && youtube-advisor --help"

# Check ANTHROPIC_API_KEY (warn only, don't fail)
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo
  echo "⚠  ANTHROPIC_API_KEY is not set. Required for the LLM-drafted SKILL.md step."
  echo "   Get a key at https://console.anthropic.com → API Keys → Create Key,"
  echo "   then: echo 'export ANTHROPIC_API_KEY=sk-...' >> ~/.zshrc && source ~/.zshrc"
fi

echo
echo "Done. In Claude Code, invoke /youtube-advisor and paste a YouTube channel URL."
