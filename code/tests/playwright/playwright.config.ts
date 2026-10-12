import { defineConfig } from '@playwright/test';

export default defineConfig({
  globalSetup: require.resolve('./global-setup.ts'),
  use: {
    baseURL: 'http://localhost:8000',
    headless: false,
    viewport: { width: 1280, height: 720 },
  },
  webServer: {
    command: 'powershell -Command "Start-Sleep -Seconds 60"',
    url: 'http://localhost:8000/api/health',
    timeout: 60000,
    reuseExistingOption: true,
  },
  projects: [
    {
      name: 'firefox',
      use: { ...require('@playwright/test').devices['Desktop Firefox'] },
    },
  ],
  outputDir: './results',
});
