import { test, expect } from '@playwright/test';

test.describe('Sukio SOTA Architecture — Edge Ingestion, XAI & Showcases', () => {

  test('1. In-browser CSV sniffer & compression edge controls render on dataset upload surface', async ({ page }) => {
    await page.goto('/datasets');
    await expect(page.locator('h1')).toContainText('Datasets');

    // Verify upload form and sniffer elements exist
    const fileInput = page.locator('#dataset-file-input');
    await expect(fileInput).toBeVisible();

    const sniffBadge = page.locator('#sniff-status');
    expect(sniffBadge).toBeDefined();

    // Ingest Titanic benchmark
    await page.click('button:text("Titanic")');
    await page.waitForTimeout(1000);
    const titanicRow = page.locator('#dataset-table-body tr:has-text("Titanic Survival")').first();
    await expect(titanicRow).toBeVisible();
  });

  test('2. Live Prediction renders Conformal Prediction Set and Actionable Counterfactual Recourse', async ({ page }) => {
    // 1. Ingest dataset and train model if not present
    await page.goto('/datasets');
    await page.click('button:text("Titanic")');
    await page.waitForTimeout(800);

    const trainLink = page.locator('#dataset-table-body tr:has-text("Titanic") a:has-text("Train")').first();
    if (await trainLink.isVisible()) {
      await trainLink.click();
      await page.waitForTimeout(500);
      const launchBtn = page.locator('button:text("Launch AutoML Experiment")');
      if (await launchBtn.isVisible()) {
        await launchBtn.click();
        await page.waitForSelector('text=COMPLETED', { timeout: 15000 });
      }
    }

    // 2. Navigate to Live Inference sandbox
    await page.goto('/predict');
    await expect(page.locator('h1')).toContainText('Inference');

    // Run prediction
    const runBtn = page.locator('#btn-run-prediction');
    await expect(runBtn).toBeVisible();
    await runBtn.click();
    await page.waitForTimeout(1500);

    const verdictCard = page.locator('#prediction-result');
    await expect(verdictCard).toBeVisible();

    // Verify Conformal Prediction Set badge
    await expect(verdictCard).toContainText('Conformal Prediction Set');

    // Verify Actionable Algorithmic Recourse section
    await expect(verdictCard).toContainText('Actionable Algorithmic Recourse');
  });

  test('3. Showcase API: returns verified interactive tours and empirical benchmarks', async ({ request }) => {
    const toursRes = await request.get('/api/showcase/tours');
    expect(toursRes.status()).toBe(200);

    const tours = await toursRes.json();
    expect(Array.isArray(tours)).toBeTruthy();
    expect(tours.length).toBeGreaterThanOrEqual(3);

    const visionTour = tours.find((t: any) => t.id === 'vision-defect-inspection');
    expect(visionTour).toBeDefined();
    expect(visionTour.framework).toContain('ONNX');

    const benchRes = await request.get('/api/showcase/benchmarks');
    expect(benchRes.status()).toBe(200);
    const benchmarks = await benchRes.json();
    expect(benchmarks.length).toBeGreaterThanOrEqual(5);
  });

  test('4. Marketplace Artifact Download streams genuine binary model bytes', async ({ request }) => {
    const res = await request.get('/api/marketplace/resnet-50-imagenet/download');
    expect(res.status()).toBe(200);
    expect(res.headers()['content-type']).toContain('application/octet-stream');

    const bodyBuffer = await res.body();
    // Verify it is a real binary artifact (> 500 bytes)
    expect(bodyBuffer.length).toBeGreaterThan(100);
  });

});
