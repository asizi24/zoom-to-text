# Feature Spec: Key Terms Glossary

**Selected by:** autonomous PM cycle, 2026-05-07

## Research Basis

Trending in 2026 AI study tools (Mindgrasp, NoteGPT, Otter.ai, Notta): automatic extraction of domain-specific vocabulary with concise definitions is the top-requested student feature. Users need a quick-reference glossary to study from, not just a transcript or summary. This feature is absent from the current project despite the full extraction pipeline being in place.

## What We Are Building

After every processed lecture or meeting, the system will automatically extract a **Key Terms Glossary** — a structured list of important domain terms, each paired with a concise definition drawn directly from the recording's content. The glossary is produced by extending the existing LLM extraction call (Call 2 in the parallel synthesis‖extraction pipeline) with a new `key_terms` field: `[{term, definition, context?}]`. No additional LLM roundtrip is required. The results are stored in the `LessonResult` as an optional `key_terms: list[KeyTerm]` field (backward-compatible, defaults to `[]`), surfaced in the History tab under a new "Glossary" section (rendered with standard DOM APIs, no `innerHTML`), and included in the Obsidian Markdown export as a `## Key Terms` section. The feature respects the source language constraint (Hebrew recordings yield Hebrew terms and definitions). Implementation touches: `app/models.py`, `app/services/summarizer.py` (prompt + parser + merger), `app/services/exporters/markdown.py`, and `static/index.html`.
