# Third-party licenses

This skill's runtime dependencies and their licenses. None are vendored; all are installed via pip/uv into a local virtualenv.

| Package                                | License        | Project                                                                |
| -------------------------------------- | -------------- | ---------------------------------------------------------------------- |
| `yt-dlp`                               | Unlicense      | https://github.com/yt-dlp/yt-dlp                                       |
| `youtube-transcript-api`               | MIT            | https://github.com/jdepoix/youtube-transcript-api                      |
| `fastembed`                            | Apache-2.0     | https://github.com/qdrant/fastembed                                    |
| `rank-bm25`                            | Apache-2.0     | https://github.com/dorianbrown/rank_bm25                               |
| `numpy`                                | BSD-3-Clause   | https://numpy.org/                                                     |
| `pyyaml`                               | MIT            | https://pyyaml.org/                                                    |
| `jinja2`                               | BSD-3-Clause   | https://jinja.palletsprojects.com/                                     |
| `anthropic` (Python SDK)               | MIT            | https://github.com/anthropics/anthropic-sdk-python                     |
| `tqdm`                                 | MIT            | https://github.com/tqdm/tqdm                                           |
| `click`                                | BSD-3-Clause   | https://click.palletsprojects.com/                                     |
| `faster-whisper` (optional `whisper` extra) | MIT      | https://github.com/SYSTRAN/faster-whisper                              |
| `scriba` (optional shellout)           | MIT            | https://github.com/AlexanderAbramovPav/scriba                                     |
| `ffmpeg` (runtime binary)              | LGPL-2.1+      | https://ffmpeg.org/                                                    |

### Model licenses

| Model                              | License        | Note                                                                                                                              |
| ---------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `BAAI/bge-small-en-v1.5`           | MIT            | Default embedding model; downloaded by `fastembed` on first use.                                                                  |
| `BAAI/bge-m3` (optional)           | MIT            | Multilingual embedding; used when `--multilingual` is passed.                                                                     |
| Claude Haiku 4.5 (API-only)        | Anthropic Terms| Called via the Anthropic Python SDK; no model weights downloaded. Set `ANTHROPIC_API_KEY` in your environment.                    |
| `openai/whisper-large-v3` (via faster-whisper) | MIT | Local Whisper ASR fallback; ~3 GB model downloaded on first use only if videos lack captions.                                |
