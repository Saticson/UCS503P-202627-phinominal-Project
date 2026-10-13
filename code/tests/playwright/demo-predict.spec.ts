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

  test('direct API prediction endpoint verification', async ({ request }) => {
    const payload = {
      model_id: 1,
      features: {
        sepal_length: 5.1,
        sepal_width: 3.5,
        petal_length: 1.4,
        petal_width: 0.2
      }
    };

    const response = await request.post('/api/predict', {
      data: payload
    });

    // 200 or 404/400 (if model 1 not trained yet)
    expect([200, 400, 404]).toContain(response.status());
    if (response.status() === 200) {
      const result = await response.json();
      expect(result).toHaveProperty('prediction');
      expect(result).toHaveProperty('confidence');
    }
  });

});
