"""Resolve cookies for yt-dlp / youtube-transcript-api calls.

Layered resolution (highest precedence first):
  1. Explicit kwargs (`from_browser=`, `cookies_file=`).
  2. Env var `YOUTUBE_ADVISOR_COOKIES_BROWSER`.
  3. Env var `YOUTUBE_ADVISOR_COOKIES_FILE`.
  4. Auto-discovery: `~/.config/youtube-advisor/cookies.txt`.
  5. Empty dict (no cookies).

The resolved dict carries either {"browser": str} or {"file": pathlib.Path}.
yt-dlp can consume either; youtube-transcript-api only consumes a file via a
custom requests.Session, so callers should branch accordingly.
"""
from __future__ import annotations
import os
import pathlib
import subprocess

DEFAULT_COOKIES_PATH = pathlib.Path.home() / ".config" / "youtube-advisor" / "cookies.txt"
VALID_BROWSERS = {
    "chrome", "firefox", "safari", "brave", "edge", "chromium", "opera", "vivaldi"
}


def _ytdlp_supports_remote_components() -> bool:
    """Probe `yt-dlp --help` once for the `--remote-components` flag.

    Older yt-dlp releases lack this flag, so we degrade gracefully when it
    isn't there. Any subprocess error (yt-dlp missing, timeout) → False.
    """
    try:
        out = subprocess.run(
            ["yt-dlp", "--help"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return "--remote-components" in (out.stdout or "") + (out.stderr or "")
    except Exception:
        return False


_HAS_REMOTE_COMPONENTS = _ytdlp_supports_remote_components()

# yt-dlp flags we always pass to reduce YT anti-bot block rates. 1-5s random
# pause between requests + exponential backoff on HTTP errors up to 30s.
#
# Additionally:
#   --extractor-args youtube:formats=missing_pot — tolerate missing PO tokens,
#     a common source of "Sign in to confirm you're not a bot" errors.
#   --remote-components ejs:github — auto-download yt-dlp's EJS JS-challenge
#     solver from GitHub; required since YouTube started serving JS challenges
#     on most extraction paths in 2025. Requires `deno` on PATH at runtime.
#     We only add this flag when the installed yt-dlp recognises it
#     (older releases lack it; passing it there would error out).
YTDLP_ANTIBOT_ARGS = [
    "--sleep-interval", "1",
    "--max-sleep-interval", "5",
    "--retries", "5",
    "--retry-sleep", "http:exp=1:30",
    "--extractor-args", "youtube:formats=missing_pot",
]
if _HAS_REMOTE_COMPONENTS:
    YTDLP_ANTIBOT_ARGS += ["--remote-components", "ejs:github"]


def resolve_cookies(
    *,
    from_browser: str | None = None,
    cookies_file: str | None = None,
) -> dict:
    """Resolve cookies source per layered precedence.

    Returns {"browser": <name>} | {"file": <Path>} | {}.

    Raises:
        ValueError: unknown browser.
        FileNotFoundError: explicit cookies file does not exist.
    """
    if from_browser:
        if from_browser not in VALID_BROWSERS:
            raise ValueError(
                f"Unknown browser '{from_browser}'. "
                f"Expected one of: {sorted(VALID_BROWSERS)}."
            )
        return {"browser": from_browser}
    if cookies_file:
        p = pathlib.Path(cookies_file).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Cookies file not found: {p}")
        return {"file": p}

    env_browser = os.getenv("YOUTUBE_ADVISOR_COOKIES_BROWSER")
    if env_browser:
        return resolve_cookies(from_browser=env_browser)
    env_file = os.getenv("YOUTUBE_ADVISOR_COOKIES_FILE")
    if env_file:
        return resolve_cookies(cookies_file=env_file)

    if DEFAULT_COOKIES_PATH.exists():
        return {"file": DEFAULT_COOKIES_PATH}

    return {}


def ytdlp_cookie_args(cookies: dict) -> list[str]:
    """Map resolve_cookies() output to yt-dlp CLI flags."""
    if "browser" in cookies:
        return ["--cookies-from-browser", cookies["browser"]]
    if "file" in cookies:
        return ["--cookies", str(cookies["file"])]
    return []


def ytdlp_args(cookies: dict | None) -> list[str]:
    """All extra yt-dlp args we pass on every invocation: cookies + anti-bot."""
    return ytdlp_cookie_args(cookies or {}) + YTDLP_ANTIBOT_ARGS
