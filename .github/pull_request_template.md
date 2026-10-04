## What and why

## Checklist (review rules that no tool enforces — framework section 12)
- [ ] If this changes a FIXED item (public API, wire contract, error codes, ports, layers, retry matrix, id scheme): ADR added or superseded: <link>
- [ ] Snapshots regenerated deliberately (`python scripts/snapshots.py regenerate`), and the diff is intended
- [ ] Deprecations give ≥ 2 minor releases and ≥ 6 months of notice (G19)
- [ ] Each consumer depends on the narrowest port it uses (G21)
- [ ] Behaviour changes with an unchanged signature, error-message changes and retry/timeout default changes are in CHANGELOG.md
- [ ] Docs describe what the code does, not what it should do
