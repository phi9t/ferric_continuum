# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT-MAP.md`** at the repo root if it exists. It points at the context docs relevant to each subsystem.
- **Context-local `CONTEXT.md` files** for the subsystem under work.
- **`docs/adr/`** for system-wide decisions.
- **Context-local `docs/adr/` directories** for decisions scoped to a subsystem.

If any of these files don't exist, proceed silently. Don't flag their absence or suggest creating them upfront. The `/domain-modeling` skill creates them lazily when terms or decisions actually get resolved.

## File structure

Ferric Continuum is a Bazel-built monorepo and is treated as a multi-context repo. The root map is the entrypoint, and each major subsystem owns its own vocabulary and local decisions.

```text
/
├── CONTEXT-MAP.md
├── docs/adr/
└── ferric_continuum/
    ├── cuda_gym/
    │   ├── CONTEXT.md
    │   └── docs/adr/
    ├── cuda_kernels/
    │   ├── CONTEXT.md
    │   └── docs/adr/
    ├── foundation/
    │   ├── CONTEXT.md
    │   └── docs/adr/
    ├── hello/
    │   ├── CONTEXT.md
    │   └── docs/adr/
    ├── optimizers/
    │   └── muon/
    │       ├── CONTEXT.md
    │       └── docs/adr/
    └── tnsr/
        ├── CONTEXT.md
        └── docs/adr/
```

The map may name additional contexts as the repo grows. Follow the map over this example when they differ.

## Use the glossary's vocabulary

When output names a domain concept in an issue title, refactor proposal, hypothesis, test name, or agent brief, use the term as defined in the relevant `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, either reconsider whether the repo already has a better term or note the gap for `/domain-modeling`.

## Flag ADR conflicts

If output contradicts an existing ADR, surface it explicitly instead of silently overriding it:

> Contradicts ADR-0007 (example decision), but worth reopening because...
