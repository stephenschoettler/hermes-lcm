# Dependency assurance contract

Hermes-LCM uses a **host-owned dependency boundary**. It is installed as source
inside Hermes Agent rather than as an independently resolved Python package, so
Hermes Agent owns the environment, lock, upgrades, and any vulnerability scan of
the resolved environment. Hermes-LCM must not silently create a second resolver
or install packages to make an assurance tool pass.

The authoritative, versioned contract is
[`dependency-contract.json`](../dependency-contract.json). Contract version
`1.0.0` supports:

- Hermes Agent `>=0.16,<1`
- Python 3.11, 3.12, 3.13, and 3.14 (the CI matrix)
- the required `agent` host API
- explicitly listed compatibility-path and optional feature imports, with the
  imported API each host-resolved version must provide

The Hermes-LCM maintainers own contract updates. Increment the contract version
and review the boundary whenever a scanned runtime import, supported Python or
Hermes Agent version, or required imported API changes.

## Mechanical validation

Run:

```bash
python scripts/validate_dependency_contract.py --report-environment
```

The validator uses only the Python standard library. It:

1. validates the contract schema, ownership, version policies, and non-empty API
   requirements;
2. checks that the supported Python versions exactly match the CI matrix;
3. parses shipped plugin, operator-script, and benchmarking Python sources with
   `ast`, including literal `importlib.import_module(...)` and `__import__(...)`
   calls;
4. fails on undeclared external imports or declarations no longer observed; and
5. reports locally available modules and distribution versions literally.

The same command is a CI gate and a release-validation gate. The environment
report is **not a vulnerability scan**. It does not prove that an installed
version is vulnerability-free, and source checkouts may have no discoverable
distribution version.

## Resolver and scanner responsibility

For a release candidate, run the host repository's authoritative lock or SBOM
validation and whichever dependency scanner is actually available against that
resolved Hermes Agent environment. Record the exact host commit/lock identity,
scanner name/version, command, and output. If no scanner is available, record
that absence; do not report a clean CVE result.

Optional imports remain feature-scoped and fail closed or fall back as documented
in the contract. Adding an optional package solely to satisfy this validator is
not an acceptable fix.
