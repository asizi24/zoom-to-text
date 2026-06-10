import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

// We hit /static/index.html directly to bypass auth — the file is the
// product of the redesign and is served as a static asset.
const URL = '/static/index.html';

test('axe: no critical/serious violations on the SPA', async ({ page }) => {
  await page.goto(URL);
  // Wait for fonts + inline scripts to settle.
  await page.waitForLoadState('domcontentloaded');
  await page.waitForTimeout(400);

  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();

  const blocking = results.violations.filter(v =>
    ['critical', 'serious'].includes(v.impact ?? '')
  );
  if (blocking.length) {
    console.log('axe violations:', JSON.stringify(blocking, null, 2));
  }
  expect(blocking).toHaveLength(0);
});

test('skip-link is the first focusable and points to #main-content', async ({ page }) => {
  await page.goto(URL);
  await page.keyboard.press('Tab');
  const link = page.locator('a.skip-link');
  await expect(link).toBeFocused();
  await expect(link).toHaveAttribute('href', '#main-content');
});

test('tabs: ArrowLeft cycles forward in RTL', async ({ page }) => {
  await page.goto(URL);
  const url = page.locator('#tab-url-trigger');
  const file = page.locator('#tab-file-trigger');

  await url.focus();
  await expect(url).toHaveAttribute('aria-selected', 'true');

  await page.keyboard.press('ArrowLeft'); // forward in RTL
  await expect(file).toBeFocused();
  await expect(file).toHaveAttribute('aria-selected', 'true');
  await expect(url).toHaveAttribute('aria-selected', 'false');

  await page.keyboard.press('ArrowRight'); // backward in RTL
  await expect(url).toBeFocused();
  await expect(url).toHaveAttribute('aria-selected', 'true');
});

test('theme toggle persists across reload (no FOUC)', async ({ page }) => {
  await page.goto(URL);
  await page.click('#theme-toggle');
  const themeAfterClick = await page.evaluate(() =>
    document.documentElement.dataset.theme
  );
  expect(['light', 'dark']).toContain(themeAfterClick);
  const stored = await page.evaluate(() => localStorage.getItem('z2t-theme'));
  expect(stored).toBe(themeAfterClick);

  await page.reload();
  const themeAfterReload = await page.evaluate(() =>
    document.documentElement.dataset.theme
  );
  expect(themeAfterReload).toBe(themeAfterClick);
});

test('about modal: opens, traps ESC, restores focus', async ({ page }) => {
  await page.goto(URL);
  const trigger = page.locator('.about-toggle').first();
  await trigger.focus();
  await trigger.click();

  const modal = page.locator('#about-overlay, .about-overlay.open').first();
  await expect(modal).toBeVisible();

  await page.keyboard.press('Escape');
  await expect(trigger).toBeFocused();
});
