# פרומפט לקלוד קוד — שדרוג אסטרטגי של Zoom-to-Text

## הקשר ומשימה

אתה עובד על פרויקט `zoom-to-text` (branch: `local-monolith`), שירות FastAPI שממיר הקלטות Zoom לתמליל + סיכום + חידון. קראתי דוח אסטרטגי מעמיק המשווה את הפרויקט לפלטפורמות SaaS מסחריות (Otter.ai, Fireflies.ai, Fathom) וארכיטקטורות קוד פתוח (Meetily, Vexa). הדוח זיהה נקודות תורפה מבניות ו-5 וקטורי שדרוג אסטרטגיים. המטרה שלך היא ליישם את השדרוגים לפי סדר העדיפויות המפורט למטה.

**מגבלות קריטיות:**
- השרת הנוכחי קטן במשאבים — אל תוסיף תלויות כבדות שירוצו עליו בלי הסכמה מפורשת
- פיצ'רים "כבדים" (GPU, תלויות Rust, Pyannote) יש לכתוב כקוד אבל **לא להפעיל בפועל** בשרת הנוכחי — עתיד להתממש בשרת ביתי
- אסור לשבור את ה-API הקיים של `/api/*` — backward compatibility חובה
- לשמור על תאימות ל-3 מצבי התמלול הקיימים: `gemini_direct`, `whisper_local`, `whisper_api`

**תהליך עבודה חובה (מ-CLAUDE.md):**
1. התחל בפקודת `/mem load` לטעינת הקשר מהשיחות הקודמות
2. לפני כל פיצ'ר חדש — `/brainstorm` ואז `/tdd`
3. לפני כל commit — `/review`
4. קח ברצינות כל התראה של security plugin — במיוחד סביב `zoom_downloader.py` ו-`config.py`
5. סיים את הסשן ב-`/mem save`

---

## שלב 1 — שיפורים בשרת הנוכחי (עדיפות עליונה)

### Task 1.1: שדרוג Schema וחכמת הסינתוז של Gemini

**קובץ עיקרי:** `app/services/summarizer.py`, `app/models.py`

**מה לבנות:**
הרחב את ה-JSON schema שמוחזר מ-Gemini מעבר ל-(summary, chapters, exam) הקיים. הוסף:
- `action_items`: רשימה של `{owner, task, deadline, priority, source_quote}`
- `decisions`: רשימה של `{decision, context, stakeholders, source_quote}`
- `open_questions`: רשימה של `{question, raised_by, context}`
- `sentiment_analysis`: `{overall_tone, per_speaker_sentiment[], shifts_in_tone[]}`
- `objections_tracked`: רשימה של `{objection, raised_by, response_given, resolved}`

**דרישות:**
- כל שדה חדש יהיה optional ב-response model כדי לא לשבור frontend ישן
- Prompt engineering: כלול few-shot examples בעברית ובאנגלית
- הגדר `temperature=0.2` עבור חילוץ עובדתי (action items, decisions)
- שמור את ה-raw Gemini output ל-debug ב-SQLite תחת `result_json.raw_llm_response`

**קריטריוני הצלחה:**
- תמלול ישן פועל בלי שגיאות
- חידון נשאר כמו שהיה
- אם Gemini לא מחזיר שדה חדש — המערכת לא קורסת
- בדיקה על שתי הקלטות אמיתיות

### Task 1.2: Diarization טקסטואלי דרך Gemini (לא Pyannote)

**קובץ עיקרי:** `app/services/summarizer.py`

**רציונל:** Pyannote דורש GPU ולא מתאים לשרת הנוכחי. Gemini יכול להסיק דוברים סמנטית מהתמליל הגולמי של Whisper.

**מה לבנות:**
- שלב pre-processing חדש בפייפליין `WHISPER_LOCAL` ו-`WHISPER_API`: לפני הסיכום, שלח את התמליל ל-Gemini עם prompt ייעודי שמבקש:
  1. לזהות כמה דוברים יש
  2. לתייג כל paragraph/utterance לפי `Speaker A`, `Speaker B` וכו'
  3. אם יש רמזים סמנטיים לזהות (שמות, תפקידים) — להציע תיוג אנושי
- שמור את ה-annotated transcript כשדה נוסף: `result_json.diarized_transcript`
- הוסף flag ב-`config.py`: `ENABLE_DIARIZATION` (ברירת מחדל: True)

**לא בשלב זה:** Pyannote.audio, speaker embeddings, biometric separation — כל אלה נכנסים בשלב 3.

### Task 1.3: LLM Provider Abstraction

**קובץ חדש:** `app/services/llm_providers/` עם:
- `base.py` — interface `LLMProvider` עם `generate(prompt, schema) -> dict`
- `gemini.py` — עוטף את הקוד הקיים
- `openrouter.py` — חדש: wrapper ל-OpenRouter API (LLaMA 3 70B, Claude)
- `ollama.py` — חדש: wrapper ל-Ollama מקומי (לא מופעל בשרת, מוכן לעתיד)

**דרישות:**
- `summarizer.py` יקבל provider כ-dependency injection
- `config.py`: משתנה `LLM_PROVIDER` עם ערכים `gemini|openrouter|ollama`
- Retry logic משותף בבסיס
- כל provider יכבד את אותו JSON schema

**קריטריוני הצלחה:**
- החלפת `LLM_PROVIDER=openrouter` ב-`.env` מפעילה את OpenRouter בלי שינויי קוד אחרים
- Ollama driver נכתב במלואו אבל לא מופעל — כשיופעל בעתיד, יעבוד out-of-the-box

### Task 1.4: ייצוא Markdown ב-Obsidian Style

**קובץ:** `app/services/exporters/markdown.py` (חדש) + endpoint חדש `/api/tasks/{task_id}/export/obsidian`

**מה לבנות:**
- Frontmatter YAML בראש הקובץ:
  ```yaml
  ---
  date: 2026-04-24
  source: zoom
  duration: "01:15:23"
  participants: [אסף, דן]
  tags: [meeting, product, zoom-to-text]
  ---
  ```
- כותרות H2/H3 מובנות עם anchors
- Backlinks בסגנון `[[Meeting: YYYY-MM-DD - Topic]]`
- Section של action items כ-checkboxes `- [ ]`
- תגיות `#action/asaf` לפי owner
- החידון כ-collapsible details HTML

**אל תגע בממשק הקיים:** ה-Markdown הפשוט הנוכחי נשמר כ-default. ה-Obsidian format הוא endpoint נפרד.

### Task 1.5: שיפור Error Handling ב-processor.py

**קובץ:** `app/services/processor.py`, `app/state.py`

**מה לבנות:**
- מחלקה חדשה `ProcessingError` עם `stage`, `code`, `user_message`, `technical_details`
- כל `raise` בפייפליין מחזיר `ProcessingError` עם stage מדויק (download/preprocess/transcribe/diarize/summarize)
- שמור `error_details` מובנה ב-SQLite (לא רק string)
- ה-frontend בהיסטוריה יציג user_message ברור

**דוגמה לשגיאות שצריכות הודעה ברורה:**
- yt-dlp fails on private recording → "Chrome cookies expired, refresh from extension"
- Gemini 429 → "LLM rate limited, retrying in N seconds"
- ffmpeg chunk fails → "Audio file corrupted at chunk N"

---

## שלב 2 — פיצ'רים בקוד בלבד (לא פורסים כרגע)

### Task 2.1: Desktop Audio Loopback Capture (Vector 1 מהדוח)

**תיקייה חדשה:** `desktop/`

**מה לבנות (POC פייתון, לא Rust):**
- `desktop/capture.py` — משתמש ב-`sounddevice` לכידת System Loopback + Mic
- מיקס שני הערוצים לקובץ WAV
- Push אוטומטי ל-endpoint הקיים `/api/upload` של השרת
- CLI פשוט: `python -m desktop.capture --url http://server --duration auto`
- Stop detection: הקלטה נעצרת כשה-system audio שקט ליותר מ-2 דקות

**רציונל:** זה פותר את שבריריות הפלטפורמה. במקום להסתמך על cookies של Zoom, אתה לוכד audio מהמערכת ישירות — עובד על Zoom, Meet, Teams, Discord.

**מצב:** ניתן להריץ ידנית מהמחשב של אסף. השרת לא משתנה — משתמש ב-upload endpoint קיים.

### Task 2.2: Pyannote Diarization (לשרת ביתי עתידי)

**קובץ חדש:** `app/services/diarization/pyannote_provider.py`

**מה לבנות:**
- Class `PyannoteDiarizer` שמקבל audio path ומחזיר RTTM-style annotations
- תלות ב-`pyannote.audio` ב-`requirements-heavy.txt` (קובץ נפרד!)
- `config.py`: `DIARIZATION_PROVIDER=gemini|pyannote`
- כשהערך pyannote — השרת חייב GPU (זיהוי אוטומטי + error ברור אם אין)

**מצב:** הקוד קיים, `requirements-heavy.txt` לא מותקן בשרת הנוכחי. כשאסף יקים שרת ביתי — `pip install -r requirements-heavy.txt` + שינוי config.

### Task 2.3: WebSocket Streaming Endpoint (Vector 3)

**קובץ חדש:** `app/api/streaming.py`

**מה לבנות:**
- endpoint `/ws/transcribe` שמקבל audio chunks ב-WebSocket
- מעבד streaming דרך Gemini או whisper.cpp
- מחזיר transcript incremental עם speaker labels
- מצב MCP server: endpoints תואמי Model Context Protocol להזנת Claude/Cursor

**מצב:** נכתב במלואו, `ENABLE_STREAMING=false` ב-config. מופעל רק בשרת ביתי.

---

## שלב 3 — בדיקות ואימות

### Task 3.1: Test Suite

- Unit tests לכל LLM provider (mocked)
- Integration test: pipeline מלא עם Gemini על הקלטה קצרה (5 דקות)
- Test שהפורמט של action_items נכון
- Test ל-Obsidian export

### Task 3.2: הרצת `/review`

לפני כל commit ל-branch. תיקון כל finding בעדיפות high/medium.

### Task 3.3: עדכון `CLAUDE.md`

הוסף לסעיף Architecture את הרכיבים החדשים:
- LLM provider abstraction
- Diarization layer
- Export formats
- Desktop capture (לעתיד)

---

## סדר ביצוע מומלץ

1. Task 1.3 (LLM Provider abstraction) — תשתית שעליה יושבים השאר
2. Task 1.1 (Schema upgrade)
3. Task 1.2 (Gemini diarization)
4. Task 1.5 (Error handling) — בזמן שאתה נוגע בפייפליין
5. Task 1.4 (Obsidian export)
6. Task 2.2 (Pyannote — רק הקוד)
7. Task 2.1 (Desktop capture POC)
8. Task 2.3 (WebSocket streaming)
9. Task 3.* (Tests + review + docs)

---

## הנחיות פעולה לקלוד קוד

- לפני כל task — קרא את הקבצים הרלוונטיים המלאים, לא רק grep
- אחרי כל task — הרץ `docker compose up -d --build` ובדוק ש-`/health` עולה
- אל תמחק קוד ישן — שנה ל-optional path או deprecate מסודר
- כל תלות חדשה ב-`requirements.txt` — הסבר למה היא נחוצה ואם היא קלה
- שמור את הלוגים בעברית איפה שה-user-facing (הודעות שגיאה ב-UI)
- Commit אטומי לכל sub-task. הודעות commit בעברית/אנגלית לפי הקונבנציה הקיימת

**בסיום כל שלב, עצור ושאל לפני שאתה ממשיך לשלב הבא.** המשתמש רוצה ביקורת מעורבת בין השלבים.

## מקורות רקע

הדוח האסטרטגי שבסיסו ההמלצות: `השוואת פרויקט Zoom-to-Text למתחרים.docx` (בתיקיית uploads של הסשן הזה).
עיקר הממצאים הרלוונטיים:
- Otter/Fireflies/Fathom מבוססי בוט גלוי + ענן → פוגע בפרטיות
- Meetily — system loopback, no bot, 100% local
- Vexa — self-hosted bot infrastructure + MCP server
- פערי ליבה של הפרויקט: platform fragility (קשור ל-Zoom בלבד), diarization, last-mile synthesis
