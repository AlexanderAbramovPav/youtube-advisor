# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

This project follows [Semantic Versioning](https://semver.org/). Patch releases land for the latest minor only.

## Reporting a vulnerability

Please do **not** open a public issue for security vulnerabilities.

Email the maintainer or use [GitHub's private vulnerability reporting](https://github.com/AlexanderAbramovPav/youtube-advisor/security/advisories/new). Include:

- A description of the issue
- Steps to reproduce (minimal)
- Affected version(s)
- Your assessment of impact (data exposure, RCE, DoS, etc.)

Expect an acknowledgement within 7 days. Coordinated disclosure timeline is typically 30–90 days depending on severity, with a patch released before public disclosure.

## Threat model (brief)

youtube-advisor reads YouTube transcripts and writes them to local files. It does not:
- Open network ports
- Accept untrusted code as input
- Run with elevated privileges

The most likely real risks are:
- Malicious YouTube content embedded in transcripts (e.g., crafted Markdown that triggers the LLM downstream). Not currently sanitized.
- Cookie leakage if `--cookies` points at a sensitive file (the tool does not exfiltrate cookies; it only forwards them to `yt-dlp`).
