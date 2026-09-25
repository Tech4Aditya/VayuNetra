# Security Policy

## Project status

VayuNetra is a research prototype built for Smart India Hackathon 2026
(SIH26070). It is not a deployed production service, and does not process
live operational cyclone-warning data. Treat findings accordingly — this
is a student research project, not critical infrastructure.

## Supported versions

Only the `main` branch is maintained. There are no released versions with
backported fixes.

| Branch | Supported |
|---|---|
| `main` | ✅ |
| any other branch / fork | ❌ |

## Reporting a vulnerability

If you find a security issue (e.g. exposed credentials, an injection
vulnerability in the backend API, unsafe deserialization, path traversal
in file handling):

1. **Do not open a public GitHub issue for it.**
2. Report it privately by emailing the maintainer directly, or using
   GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
   feature on this repo (Security tab → "Report a vulnerability").
3. Include: what you found, steps to reproduce, and potential impact.

I'll acknowledge reports as soon as I can and aim to patch confirmed
issues promptly, but there's no formal SLA — this is maintained by a
single student developer, not a security team.

## Scope

In scope:
- The backend API (`backend/app.py` and related endpoints)
- Model-checkpoint loading and inference code
- Data-handling and file-path logic (upload/processing scripts)

Out of scope:
- The training data itself (public satellite/best-track datasets)
- Third-party dependencies (report those upstream instead)
- Denial-of-service via large file uploads on a local dev server — this
  isn't a hardened deployment

## Known limitations (disclosed, not vulnerabilities)

- No authentication is implemented on the API by default — do not expose
  this backend directly to the public internet without adding one.
- Model checkpoints are loaded with `torch.load(..., weights_only=False)`
  in places, which is unsafe for untrusted `.pt` files. Only load
  checkpoints you trained yourself or trust the source of.
