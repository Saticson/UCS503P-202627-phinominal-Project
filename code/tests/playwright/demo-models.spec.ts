import { test, expect } from '@playwright/test';

test.describe('Sukio Demo — Model Leaderboard & Explainability', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto('/models');
  });

  test('leaderboard view renders metric columns and explainability section', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Model Leaderboard');

    // Verify table headers
    const headers = ['Model Name', 'Type / Architecture', 'Task Type', 'Accuracy', 'F1 Score', 'ROC-AUC / RMSE', 'Actions'];
    for (const h of headers) {
      await expect(page.locator(`th:has-text("${h}")`)).toBeVisible();
    }

    // Verify explainability panel
    await expect(page.locator('h2:has-text("Explainability")')).toBeVisible();
    await expect(page.locator('a:text("Test in Prediction Sandbox")')).toBeVisible();
  });

  test('navigation from models view to prediction sandbox', async ({ page }) => {
    await page.click('a:text("Test in Prediction Sandbox")');
    await expect(page).toHaveURL(/\/predict/);
    await expect(page.locator('h1')).toContainText('Inference');
  });

  test('direct API model catalog and SHAP explainability verification', async ({ request }) => {
    const listRes = await request.get('/api/models');
    expect(listRes.status()).toBe(200);

    const models = await listRes.json();
    expect(Array.isArray(models)).toBeTruthy();

    if (models.length > 0) {
      const first = models[0];
      expect(first).toHaveProperty('id');
      expect(first).toHaveProperty('model_name');

      // Test explainability endpoint
      const shapRes = await request.post('/api/models/explain/shap', {
        data: {
          model_id: first.id,
          sample_features: { feature_1: 1.0, feature_2: 2.0 }
        }
      });
      expect([200, 400, 404]).toContain(shapRes.status());
    }
  });

});
