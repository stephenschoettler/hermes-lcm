# AccessContextV1 shared corpus
These JSON files are the cross-repository contract corpus for Hermes Agent and
Hermes-LCM. A producer or consumer can parse the `context` object with its own
JSON tooling; importing the `access_context` Python package is not required.

Each file has a `contract_revision`, one-line `description`, and `context`.
Negative, delegation, and revocation vectors also carry an `expected` object.
Timestamps are UTC ISO-8601 values. Fixture filenames are stable IDs for
conformance reports.
