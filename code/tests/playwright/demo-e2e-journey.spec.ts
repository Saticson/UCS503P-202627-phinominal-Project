import { test, expect } from '@playwright/test';

test.describe('Sukio Demo — End-to-End User Journey', () => {

  test('complete workflow: Home -> Dashboard -> Datasets -> AutoML -> Leaderboard -> Predict', async ({ page }) => {
    // 1. Visit Landing Page
    console.log('[Demo] 1. Visiting Sukio Home...');
    await page.goto('/');
    await expect(page.locator('h1')).toContainText('Welcome to Sukio');
    await expect(page.locator('.navbar')).toBeVisible();

    // 2. Navigate to Dashboard
    console.log('[Demo] 2. Navigating to Platform Dashboard...');
    await page.click('a:text("Dashboard")');
    await expect(page).toHaveURL(/\/dashboard/);
    await expect(page.locator('h1')).toContainText('Platform Dashboard');
    await expect(page.locator('.metric-card')).toHaveCount(4);
    await expect(page.locator('#datasets')).toBeVisible();

    // 3. Navigate to Datasets & Diagnostics
    console.log('[Demo] 3. Opening Datasets Management...');
    await page.click('a:text("Datasets")');
    await expect(page).toHaveURL(/\/datasets/);
    await expect(page.locator('h1')).toContainText('Datasets');

    // Verify upload form and benchmark dataset buttons
    await expect(page.locator('input[type="file"]')).toBeVisible();
    await expect(page.locator('button:text("Iris")')).toBeVisible();

    // 4. Ingest Benchmark Dataset (Iris)
    console.log('[Demo] 4. Ingesting Iris benchmark dataset...');
    await page.click('button:text("Iris")');
    await page.waitForTimeout(800);

    // Verify dataset repository is populated
    await expect(page.locator('#datasets')).toBeVisible();

    // 5. Navigate to AutoML Training Studio
    console.log('[Demo] 5. Launching AutoML Training Studio...');
    await page.click('a:text("Training")');
    await expect(page).toHaveURL(/\/training/);
    await expect(page.locator('h1')).toContainText('AutoML Training Studio');

    // Verify configuration controls
    await expect(page.locator('select[name="task_type"]')).toBeVisible();
    await expect(page.locator('input[value="Random Forest"]')).toBeChecked();
    await expect(page.locator('input[value="XGBoost"]')).toBeChecked();

    // Configure experiment
    await page.fill('input[name="target_column"]', 'species');
    await page.fill('input[name="tuning_trials"]', '5');

    // Launch experiment
    console.log('[Demo] 6. Triggering AutoML Experiment...');
    await page.click('button:text("Launch AutoML Experiment")');
    await page.waitForTimeout(1000);

    // 6. Navigate to Model Leaderboard & Explainability
    console.log('[Demo] 7. Inspecting Model Leaderboard & SHAP Attributions...');
    await page.click('a:text("Models")');
    await expect(page).toHaveURL(/\/models/);
    await expect(page.locator('h1')).toContainText('Model Leaderboard');
    await expect(page.locator('table')).toBeVisible();

    // 7. Navigate to Live Inference Sandbox
    console.log('[Demo] 8. Testing Real-Time Inference Sandbox...');
    await page.click('a:text("Predict")');
    await expect(page).toHaveURL(/\/predict/);
    await expect(page.locator('h1')).toContainText('Inference');

    // Fill sample prediction features
    const sampleFeatures = JSON.stringify({
      sepal_length: 5.1,
      sepal_width: 3.5,
      petal_length: 1.4,
      petal_width: 0.2
    }, null, 2);

    await page.fill('textarea[name="features"]', sampleFeatures);
    await page.click('button:text("Run Real-Time Prediction")');
    await page.waitForTimeout(800);

    await expect(page.locator('#prediction-result')).toBeVisible();
    console.log('[Demo] End-to-End User Journey completed successfully!');
  });

});
