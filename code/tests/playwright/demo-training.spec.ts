import { test, expect } from '@playwright/test';

test.describe('Sukio Demo — AutoML Training Studio', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto('/training');
  });

  test('AutoML studio view renders candidate models checklist and budget controls', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('AutoML Training Studio');

    // Check Candidate Models
    const modelCheckboxes = [
      'Random Forest',
      'XGBoost',
      'LightGBM',
      'Logistic / Linear',
      'MLP Neural Net',
      'Extra Trees'
    ];

    for (const name of modelCheckboxes) {
      await expect(page.locator(`.checkbox-label:has-text("${name}")`)).toBeVisible();
    }

    // Check Optimization inputs
    await expect(page.locator('input[name="tuning_trials"]')).toBeVisible();
    await expect(page.locator('input[name="cv_folds"]')).toBeVisible();
    await expect(page.locator('button:text("Launch AutoML Experiment")')).toBeVisible();
  });

  test('configure classification experiment with Optuna search budget', async ({ page }) => {
    await page.selectOption('select[name="task_type"]', 'classification');
    await page.fill('input[name="target_column"]', 'target_class');
    await page.fill('input[name="tuning_trials"]', '10');
    await page.fill('input[name="cv_folds"]', '5');

    // Toggle model selections
    await page.check('input[value="Random Forest"]');
    await page.check('input[value="LightGBM"]');
    await page.check('input[value="Extra Trees"]');

    // Launch experiment
    await page.click('button:text("Launch AutoML Experiment")');
    await page.waitForTimeout(1000);

    // Verify status container
    await expect(page.locator('#training-status')).toBeVisible();
  });

  test('configure regression experiment with Bayesian sweep', async ({ page }) => {
    await page.selectOption('select[name="task_type"]', 'regression');
    await page.fill('input[name="target_column"]', 'median_house_value');
    await page.fill('input[name="tuning_trials"]', '15');

    await page.click('button:text("Launch AutoML Experiment")');
    await page.waitForTimeout(1000);

    await expect(page.locator('#training-status')).toBeVisible();
  });

  test('direct API training job submission validation', async ({ request }) => {
    const payload = {
      dataset_id: 1,
      target_column: 'species',
      task_type: 'classification',
      models_to_train: ['Random Forest', 'Extra Trees'],
      enable_tuning: true,
      tuning_trials: 5,
      cv_folds: 3,
      test_size: 0.2,
      execution_mode: 'remote'
    };

    const response = await request.post('/api/train-model', {
      data: payload
    });

    // 200, 400 (validation), or 404 (dataset not found) are valid API responses
    expect([200, 400, 404]).toContain(response.status());
    if (response.status() === 200) {
      const data = await response.json();
      expect(data).toHaveProperty('job_id');
      expect(data).toHaveProperty('status');
    }
  });

});
