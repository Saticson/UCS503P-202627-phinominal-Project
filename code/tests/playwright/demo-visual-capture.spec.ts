import { test } from '@playwright/test';
import { existsSync, mkdirSync } from 'fs';
import { join } from 'path';

test.describe('Sukio Demo — Visual Capture for Presentations & Reports', () => {

  const outputDir = join(__dirname, 'test-output', 'demo-artifacts');

  test.beforeAll(() => {
    if (!existsSync(outputDir)) {
      mkdirSync(outputDir, { recursive: true });
    }
  });

  test('capture presentation screenshots of all core control surfaces', async ({ page }) => {
    // 1. Dashboard View
    await page.goto('/dashboard');
    await page.waitForTimeout(600);
    await page.screenshot({ path: join(outputDir, '01_dashboard_live.png'), fullPage: true });

    // 2. Datasets Management View
    await page.goto('/datasets');
    await page.waitForTimeout(600);
    await page.screenshot({ path: join(outputDir, '02_datasets_live.png'), fullPage: true });

    // 3. AutoML Training Studio View
    await page.goto('/training');
    await page.waitForTimeout(600);
    await page.screenshot({ path: join(outputDir, '03_training_studio_live.png'), fullPage: true });

    // 4. Model Leaderboard View
    await page.goto('/models');
    await page.waitForTimeout(600);
    await page.screenshot({ path: join(outputDir, '04_models_leaderboard_live.png'), fullPage: true });

    // 5. Prediction Sandbox View
    await page.goto('/predict');
    await page.waitForTimeout(600);
    await page.screenshot({ path: join(outputDir, '05_prediction_sandbox_live.png'), fullPage: true });

    console.log(`[Demo] 5 presentation screenshots saved to: ${outputDir}`);
  });

});
