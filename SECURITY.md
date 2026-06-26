# Security Policy

## Supported versions

Security fixes target the current `main` branch and the latest GitHub release.
Older releases may receive guidance or a patch release when the issue is severe and
a safe backport is practical.

## Reporting a vulnerability

Please do not publish exploitable details in a public issue first.

Preferred reporting path:

1. Use GitHub's private vulnerability reporting / Security Advisory flow for this
   repository when available.
2. If private reporting is not available, open a minimal public issue that says you
   have a security report and asks the maintainer for a private contact path. Do
   not include secrets, tokens, database dumps, private transcripts, or exploit
   details in that public issue.

Useful report details:

- affected version or commit
- operating system and Python version
- configuration relevant to the issue
- minimal reproduction steps or proof of concept
- expected impact
- whether sensitive data, credentials, transcripts, or externalized payloads are involved

## What to avoid sending publicly

- API keys, bearer tokens, OAuth tokens, passwords, private keys, cookies, or session IDs
- private conversation transcripts or user data
- full `lcm.db` files unless explicitly requested through a private channel
- exploit steps that would let others reproduce harm before a fix exists

## Response expectations

The maintainer will triage reports as time permits. Confirmed vulnerabilities will
be fixed in the smallest safe change, with tests when practical. Release notes will
credit reporters when they want credit and when doing so does not increase risk.
