/**
 * Client-side export helpers — ES module.
 *
 * Pure or near-pure functions for building/downloading lesson exports.
 * Callers pass state explicitly (no globals read here) so this module
 * is independently testable and tree-shakeable.
 *
 * Usage (dynamic import from non-module scripts):
 *   const { buildMarkdown } = await import('/static/js/export-utils.js');
 */

/**
 * Build a client-side Markdown representation of a lesson result.
 * @param {object} result  - LessonResult fields (summary, chapters, quiz, transcript…)
 * @param {string} [source] - optional source URL/label for the > מקור line
 * @returns {string}
 */
export function buildMarkdown(result, source = '') {
  const lines = [];
  const date = new Date().toLocaleDateString('he-IL');
  lines.push(`# סיכום שיעור — ${date}`);
  if (source) lines.push(`> מקור: ${source}`);
  lines.push('');

  if (result.summary) {
    lines.push('## 📝 סיכום');
    lines.push(result.summary);
    lines.push('');
  }

  if ((result.chapters || []).length) {
    lines.push('## 📚 פרקים ונושאים');
    result.chapters.forEach((ch, i) => {
      lines.push(`### ${i + 1}. ${ch.title}`);
      lines.push(ch.content);
      if (ch.key_points && ch.key_points.length) {
        ch.key_points.forEach(p => lines.push(`- ${p}`));
      }
      lines.push('');
    });
  }

  if ((result.quiz || []).length) {
    lines.push('## 🧠 מבחן');
    result.quiz.forEach((q, i) => {
      lines.push(`**שאלה ${i + 1}:** ${q.question}`);
      (q.options || []).forEach(o => lines.push(`- ${o}`));
      lines.push(`✅ תשובה נכונה: ${q.correct_answer}`);
      if (q.explanation) lines.push(`💡 ${q.explanation}`);
      lines.push('');
    });
  }

  if (result.transcript) {
    lines.push('## 📄 תמלול גולמי');
    lines.push(result.transcript);
  }

  return lines.join('\n');
}

/**
 * Trigger a client-side Markdown file download for a lesson result.
 * @param {object} result
 * @param {string} [source]
 */
export function downloadMarkdown(result, source = '') {
  const md = buildMarkdown(result, source);
  const date = new Date().toISOString().slice(0, 10);
  const blob = new Blob([md], { type: 'text/markdown; charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `lesson-summary-${date}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/**
 * Fetch an Obsidian-flavored Markdown export from the server and download it.
 * @param {string} taskId
 */
export async function downloadObsidian(taskId) {
  try {
    const resp = await fetch(`/api/tasks/${taskId}/export/obsidian`, {
      credentials: 'include',
    });
    if (!resp.ok) {
      alert('ייצוא Obsidian נכשל: ' + resp.status);
      return;
    }
    const md = await resp.text();
    const blob = new Blob([md], { type: 'text/markdown; charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `obsidian-${taskId.slice(0, 8)}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (err) {
    alert('שגיאה בייצוא Obsidian: ' + err.message);
  }
}

/**
 * Fetch a server-rendered PDF export and download it.
 * @param {string} taskId
 */
export async function downloadPdf(taskId) {
  try {
    const resp = await fetch(`/api/tasks/${taskId}/export/pdf`, {
      credentials: 'include',
    });
    if (!resp.ok) {
      const msg = resp.status === 503
        ? 'ייצוא PDF אינו זמין בסביבה זו (חסרות ספריות מערכת)'
        : 'ייצוא PDF נכשל: ' + resp.status;
      alert(msg);
      return;
    }
    const blob = await resp.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `lesson-${taskId.slice(0, 8)}.pdf`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (err) {
    alert('שגיאה בייצוא PDF: ' + err.message);
  }
}
