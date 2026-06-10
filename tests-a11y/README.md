# A11y test harness

Two CI checks live here:

- **Playwright + @axe-core/playwright** — `tests-a11y/smoke.spec.ts`
  runs axe-core (WCAG 2.1 AA tags) against `/static/index.html` plus
  four keyboard/UX assertions (skip-link, RTL tab nav, theme persist,
  modal ESC restore). Desktop Chrome + Pixel 5 mobile profiles.

- **Lighthouse CI** — `lighthouserc.json` asserts:
  - accessibility ≥ 0.95 (hard error)
  - best-practices ≥ 0.90 (warn)
  - performance ≥ 0.80 (warn)

## Run locally

```bash
npm install
npx playwright install --with-deps chromium
npx playwright test tests-a11y/
npx lhci autorun
```

`playwright.config.ts` auto-spawns uvicorn on 127.0.0.1:8000 (or
reuses an existing server you have running). Lighthouse does the
same. The server tests in `tests/` are unaffected — this harness
only loads in CI when `static/`, `tests-a11y/`, or `app/` change.

## Files

| File | Purpose |
|---|---|
| `package.json` | Pins `@axe-core/playwright`, `@playwright/test`, `@lhci/cli` |
| `playwright.config.ts` | Runner + auto-spawn uvicorn; `he-IL` locale; chromium desktop + Pixel 5 |
| `tsconfig.json` | TypeScript config for `tests-a11y/` only |
| `lighthouserc.json` | LHCI thresholds + start command |
| `.github/workflows/a11y.yml` | Triggers on PR/push touching UI/server files |
