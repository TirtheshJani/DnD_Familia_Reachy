# Reachy Familiar

Brain service for a Reachy Mini Wireless acting as a D&D familiar. See `docs/superpowers/specs/2026-09-10-reachy-familiar-design.md`.

Run tests: `uv run pytest -q`

## Campaign notes

`data/notes/` is what the familiar remembers, and only curated fiction plus the table's homebrew rules may go in it. See `docs/notes-policy.md`.

## Rules attribution

This work includes material from the System Reference Document 5.1 and 5.2.1 ("SRD 5.1", "SRD 5.2.1") by Wizards of the Coast LLC, available at https://www.dndbeyond.com/srd. The SRDs are licensed under the Creative Commons Attribution 4.0 International License, available at https://creativecommons.org/licenses/by/4.0/legalcode.

The PDFs are not redistributed here. Download them into `data/rules/` and index them:

```bash
uv run familiar-brain index rules data/rules/SRD_CC_v5.1.pdf --edition 2014
uv run familiar-brain index rules data/rules/SRD_CC_v5.2.1.pdf --edition 2024
```
