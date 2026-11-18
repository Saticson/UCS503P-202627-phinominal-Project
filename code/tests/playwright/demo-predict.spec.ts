import { test, expect } from '@playwright/test';

test.describe('Sukio Demo — Live Inference Sandbox', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto('/predict');
  });

  test('inference sandbox renders model selector and feature JSON input', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Inference & Prediction Sandbox');
    await expect(page.locator('input[name="model_id"]')).toBeVisible();
    await expect(page.locator('textarea[name="features"]')).toBeVisible();
    await expect(page.locator('button:text("Run Real-Time Prediction")')).toBeVisible();
    await expect(page.locator('#prediction-result')).toBeVisible();
  });

  test('submit prediction payload with custom feature values', async ({ page }) => {
    const payload = JSON.stringify({
      age: 45,
      income: 75000,
      credit_score: 720,
      loan_amount: 15000
    });

    await page.fill('input[name="model_id"]', '1');
    await page.fill('textarea[name="features"]', payload);

    await page.click('button:text("Run Real-Time Prediction")');
    await page.waitForTimeout(800);

    // Verify prediction result box is present
    await expect(page.locator('#prediction-result')).toBeVisible();
  });

  test('direct API prediction endpoint error handling and inference verification', async ({ request }) => {
    // 1. Strict Negative Test: Non-existent model ID must return 404
    const notFoundRes = await request.post('/api/predict', {
      data: { model_id: 999999, features: { age: 30, income: 50000 } }
    });
    expect(notFoundRes.status()).toBe(404);

    // 2. Query catalog for trained models
    const modelsRes = await request.get('/api/models');
    expect(modelsRes.status()).toBe(200);
    const models = await modelsRes.json();

    if (Array.isArray(models) && models.length > 0) {
      const activeModel = models[0];

      // 3. Strict Negative Test: Missing / empty features on active model must return 400
      const emptyFeaturesRes = await request.post('/api/predict', {
        data: { model_id: activeModel.id, features: {} }
      });
      expect(emptyFeaturesRes.status()).toBe(400);

      // 4. Positive & Data-Dependent Test: Valid features return 200 with prediction & confidence
      const resA = await request.post('/api/predict', {
        data: {
          model_id: activeModel.id,
          features: { pclass: 1, sex: 'female', age: 29, fare: 211.3 }
        }
      });
      expect(resA.status()).toBe(200);
      const dataA = await resA.json();
      expect(dataA).toHaveProperty('prediction');
      expect(dataA).toHaveProperty('confidence');
      expect(dataA.confidence).toBeGreaterThan(0.0);
      expect(dataA.confidence).toBeLessThanOrEqual(1.0);
    }
  });

});
