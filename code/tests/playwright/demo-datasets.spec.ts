import { test, expect } from '@playwright/test';

test.describe('Sukio Demo — Dataset Ingestion & Profiling', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto('/datasets');
  });

  test('dataset management view renders diagnostics controls and benchmarks', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Datasets & Diagnostics');
    await expect(page.locator('form[hx-post*="upload"]')).toBeVisible();
    await expect(page.locator('input[type="file"]')).toBeVisible();

    // Verify benchmark buttons
    const benchmarkButtons = ['Iris', 'Titanic', 'Diabetes', 'Housing'];
    for (const name of benchmarkButtons) {
      await expect(page.locator(`.example-btn:text("${name}")`)).toBeVisible();
    }
  });

  test('ingest benchmark datasets via HTMX', async ({ page }) => {
    // Click Iris benchmark
    await page.click('.example-btn:text("Iris")');
    await page.waitForTimeout(600);
    await expect(page.locator('#datasets')).toBeVisible();

    // Click Titanic benchmark
    await page.click('.example-btn:text("Titanic")');
    await page.waitForTimeout(600);
    await expect(page.locator('#datasets')).toBeVisible();
  });

  test('upload custom CSV dataset with target column', async ({ page }) => {
    const csvContent = 'feature_a,feature_b,feature_c,churn\n1.2,3.4,0.5,1\n2.1,1.1,0.9,0\n3.0,2.5,0.2,1\n4.2,3.1,0.8,0\n';
    
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'customer_churn.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent)
    });

    await page.fill('input[name="name"]', 'telecom_churn_q3');
    await page.fill('input[name="target_column"]', 'churn');

    await page.click('button:text("Upload Dataset")');
    await page.waitForTimeout(1000);

    await expect(page.locator('#datasets')).toBeVisible();
  });

  test('direct API verification: dataset listing and quality report endpoint', async ({ request }) => {
    const response = await request.get('/api/datasets');
    expect(response.status()).toBe(200);

    const datasets = await response.json();
    expect(Array.isArray(datasets)).toBeTruthy();

    if (datasets.length > 0) {
      const firstId = datasets[0].id;
      const reportRes = await request.get(`/api/datasets/${firstId}/quality-report`);
      expect([200, 404]).toContain(reportRes.status());
    }
  });

});
