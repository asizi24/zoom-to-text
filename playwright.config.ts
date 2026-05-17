import { defineConfig, devices } from '@playwright/test';

// Spawns the same uvicorn the server tests already use.
// `webServer.reuseExistingServer` lets a dev keep `docker compose up`
// running and skip the spawn locally.
export default defineConfig({
  testDir: './tests-a11y',
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: 'http://127.0.0.1:8000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    locale: 'he-IL',
  },
  projects: [
    {
      name: 'chromium-desktop',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'chromium-mobile',
      use: { ...devices['Pixel 5'] },
    },
  ],
  webServer: {
    command: 'python -m uvicorn app.main:app --host 127.0.0.1 --port 8000',
    url: 'http://127.0.0.1:8000/health',
    timeout: 60_000,
    reuseExistingServer: !process.env.CI,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
