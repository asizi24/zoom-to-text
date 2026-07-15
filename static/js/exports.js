// Export builders: plain Markdown and Obsidian-flavored (YAML properties,
// callouts, foldable quiz).
import { store } from './store.js';

export function buildMarkdown(result) {
  const lines = [];
  const date = new Date().toLocaleDateString('he-IL');
  lines.push(`# סיכום שיעור — ${date}`);
  if (store.currentSource) lines.push(`> מקור: ${store.currentSource}`);
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

export function exportMarkdown() {
  if (!store.currentResult) return;
  const md = buildMarkdown(store.currentResult);
  const date = new Date().toISOString().slice(0, 10);
  downloadText(md, `lesson-summary-${date}.md`);
}

function sanitizeFilename(s) {
  return (s || '').replace(/[\\/:*?"<>|#^\[\]]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 80);
}

function downloadText(text, filename) {
  const blob = new Blob([text], { type: 'text/markdown; charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function buildObsidianMarkdown(result) {
  const dateISO = new Date().toISOString().slice(0, 10);
  const source  = (store.currentSource || '').replace(/^upload:/, '');
  const title   = sanitizeFilename(source.replace(/\.[a-z0-9]{2,5}$/i, '')) || `שיעור ${dateISO}`;
  const lines   = [];

  // YAML frontmatter → Obsidian properties
  lines.push('---');
  lines.push(`title: "${title.replace(/"/g, "'")}"`);
  lines.push(`date: ${dateISO}`);
  if (source) lines.push(`source: "${source.replace(/"/g, "'")}"`);
  lines.push('tags:');
  lines.push('  - שיעור');
  lines.push('  - zoom-to-text');
  lines.push('---');
  lines.push('');
  lines.push(`# ${title}`);
  lines.push('');

  if (result.summary) {
    lines.push('> [!abstract] סיכום השיעור');
    result.summary.split('\n').forEach(l => lines.push('> ' + l));
    lines.push('');
  }

  if ((result.chapters || []).length) {
    lines.push('## 🗂️ פרקים');
    lines.push('');
    result.chapters.forEach((ch, i) => {
      lines.push(`### ${i + 1}. ${ch.title || ''}`);
      if (ch.content) { lines.push(ch.content); lines.push(''); }
      (ch.key_points || []).forEach(p => lines.push(`- ${p}`));
      lines.push('');
    });
  }

  if ((result.quiz || []).length) {
    lines.push('## 🧠 מבחן לתרגול');
    lines.push('');
    result.quiz.forEach((q, i) => {
      // Foldable callout — the answer stays hidden until expanded in Obsidian
      lines.push(`> [!question]- שאלה ${i + 1}: ${q.question || ''}`);
      (q.options || []).forEach(o =>
        lines.push(`> - ${o === q.correct_answer ? '**' + o + '** ✅' : o}`));
      if (q.explanation) { lines.push('>'); lines.push(`> 💡 ${q.explanation}`); }
      lines.push('');
    });
  }

  if ((result.flashcards || []).length) {
    lines.push('## 🃏 כרטיסיות');
    lines.push('');
    result.flashcards.forEach(c => {
      lines.push(`**ש:** ${c.front || ''}`);
      lines.push(`**ת:** ${c.back || ''}`);
      lines.push('');
    });
  }

  if (result.transcript) {
    lines.push('## 📄 תמלול מלא');
    lines.push('');
    lines.push(result.transcript);
  }

  return { md: lines.join('\n'), filename: `${title} — ${dateISO}.md` };
}

export function exportObsidian() {
  if (!store.currentResult) return;
  const { md, filename } = buildObsidianMarkdown(store.currentResult);
  downloadText(md, filename);
}
