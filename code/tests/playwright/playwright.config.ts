import { defineConfig, devices } from '@playwright/test';

const bravePath = process.env.BRAVE_PATH || 'C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe';

export default defineConfig({
  globalSetup: require.resolve('./global-setup.ts'),
  outputDir: './test-output/results',
  reporter: [
    ['list'],
    ['html', { outputFolder: 'test-output/html-report', open: 'never' }]
  ],
  use: {
    baseURL: 'http://localhost:8000',
    headless: false,
    launchOptions: {
      slowMo: 500,
    },
    viewport: { width: 1280, height: 720 },
    screenshot: 'on',
    video: 'off',
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'brave',
      use: {
        ...devices['Desktop Chrome'],
        executablePath: bravePath,
      },
    },
  ],
});
