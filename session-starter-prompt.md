# Session Starter Prompt — Zoom to Text

> Copy-paste this at the start of a Claude Code session when you want to improve the project.

---

## PASTE THIS:

```
Load my project memory with /mem load, then read CLAUDE.md for full project context.

I want to improve the "Zoom to Text" project — a FastAPI service that transcribes Zoom recordings and generates summaries, chapters, and exams using Gemini 2.5 Flash.

Your job today:
1. Use /mem load to recall what we've worked on before
2. Read CLAUDE.md for architecture and priorities
3. Ask me which area I want to focus on (or suggest the highest-priority item from CLAUDE.md)
4. Use the right tool for the job:
   - Backend work → use /tdd and /debug from Superpowers
   - UI work → use /frontend-design
   - Before any commit → run /review
   - Security-sensitive code (cookies, subprocess, API keys) → pay attention to security-guidance warnings
   - Exploring complex tradeoffs → use /brainstorm or /pm from gstack
5. At the end, save progress with /mem save

Be proactive: if you spot an obvious bug or improvement while working, flag it. Don't wait for me to ask.

Active branch: local-monolith
Run locally: docker compose up -d → http://localhost:8000
```

---

## FOCUSED PROMPTS (for specific tasks)

### To fix a specific bug:
```
/mem load
Read CLAUDE.md. I have a bug: [DESCRIBE THE BUG].
Use /debug to work through it systematically — observe, hypothesize, test, then fix.
Run /review before we commit the fix.
```

### To improve the UI:
```
/mem load
Read CLAUDE.md. I want to improve the frontend at static/index.html.
Use /frontend-design to guide the redesign — aim for distinctive, non-generic UI.
Focus on: [SPECIFIC AREA e.g. progress bar / exam display / history tab]
```

### To add a new feature:
```
/mem load
Read CLAUDE.md. I want to add: [FEATURE].
Use /brainstorm first to explore approaches, then /tdd to implement with tests.
Check with /pm if this fits the current priorities.
Run /review before committing.
```

### To do a full code review session:
```
/mem load
Read CLAUDE.md. Do a thorough review of the codebase.
Use the Code Review plugin (/review) on the current branch vs main.
Also check for security issues in: zoom_downloader.py, config.py, routes.py.
Summarize findings by priority and suggest a fix order.
```

### To improve Gemini output quality:
```
/mem load
Read CLAUDE.md. I want to improve the quality of the AI output in app/services/summarizer.py.
The current issues are: exam questions too easy, summaries sometimes miss key points.
Use /brainstorm to redesign the prompts, then implement and test with a sample transcript.
```
