# UI Redesign — Dark Glass (A2)
**Date:** 2026-04-16  
**Status:** Approved by user

## Summary
Full visual redesign of `static/index.html` to a "Dark Glass" aesthetic — deep dark-purple background with glass-morphism cards, gradient accents, and a premium SaaS feel. All existing JavaScript logic stays intact; only CSS and HTML structure change.

## Design Direction
- **Background:** `#090b14` + 3 large blurred color orbs (purple, pink, blue) via `position: fixed`
- **Card:** `rgba(255,255,255,.055)` + `backdrop-filter: blur(28px)` + subtle border + deep shadow
- **Accent color:** gradient `#7c3aed → #db2777` (purple → pink) for tabs, CTA, selected states
- **Typography:** Inter (Google Fonts), gradient text for main title (purple → pink → mint)
- **Mode cards:** glass panels with purple glow when selected
- **History button:** fixed bottom-left, frosted glass pill

## Scope
Single file change: `static/index.html`

All functional behavior preserved:
- URL tab / File upload tab
- 3 mode options (Gemini Direct, Whisper Local, OpenAI Whisper)
- Progress card
- Results card (summary, chapters, quiz, transcript toggle)
- History panel
- Export Markdown / Copy / Print
- Keyboard shortcuts (Enter, Escape)

## Out of Scope
- No backend changes
- No new features
- No JavaScript refactoring

## Deployment
1. Update `static/index.html`
2. `docker compose up -d --build api`
3. `git push origin local-monolith`
