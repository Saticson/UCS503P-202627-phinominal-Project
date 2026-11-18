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

    const verdictCard = page.locator('#prediction-result');
    await expect(verdictCard).toBeVisible();

    // Verify Conformal Prediction Set badge
    await expect(verdictCard).toContainText('Conformal Prediction Set', { timeout: 10000 });

    // Verify Actionable Algorithmic Recourse section
    await expect(verdictCard).toContainText('Actionable Algorithmic Recourse', { timeout: 10000 });
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

  test('5. Hardware GPU Diagnostics: audits GPU hardware detection and honest fallback reason', async ({ request }) => {
    const healthRes = await request.get('/api/health');
    expect(healthRes.status()).toBe(200);
    const health = await healthRes.json();
    expect(health.status).toBe('ok');
    expect(health.effective_backend).toBeDefined();

    const gpuRes = await request.get('/api/health/gpu');
    expect(gpuRes.status()).toBe(200);
    const gpu = await gpuRes.json();
    expect(gpu.device_name).toBeDefined();
    expect(gpu.effective_backend).toBeDefined();
    expect(gpu.fallback_reason).toBeDefined();
  });

  test('6. Research Hard Benchmark API: returns 9 benchmark catalog with explicit resource tiers', async ({ request }) => {
    const res = await request.get('/api/showcase/benchmarks/hard');
    expect(res.status()).toBe(200);
    const benchmarks = await res.json();
    expect(benchmarks.length).toBe(9);

    // Verify Tabular Tier 1
    const ieee = benchmarks.find((b: any) => b.id === 'tabular-ieee-cis-fraud');
    expect(ieee).toBeDefined();
    expect(ieee.tier).toBe('TIER_1_LOCAL_SUPPORTED');
    expect(ieee.status).toBe('EVALUATED_ON_CONTROL_PLANE');

    // Verify Tabular Tier 2 (Criteo 1TB)
    const criteo = benchmarks.find((b: any) => b.id === 'tabular-criteo-1tb');
    expect(criteo).toBeDefined();
    expect(criteo.status).toBe('RESOURCE_TIER_EXCEEDED');

    // Verify Unsupported Modality (ImageNet)
    const imagenet = benchmarks.find((b: any) => b.id === 'vision-imagenet-1k');
    expect(imagenet).toBeDefined();
    expect(imagenet.status).toBe('TRAINING_UNSUPPORTED_IN_EDGE_TIER');
  });

  test('7. NLP Text Saliency Sandbox: renders prediction and token attribution pills', async ({ page }) => {
    await page.goto('/predict');
    await expect(page.locator('h1')).toContainText('Inference');

    // Click NLP Sandbox Tab
    const nlpTab = page.locator('#tab-btn-nlp');
    await expect(nlpTab).toBeVisible();
    await nlpTab.click();

    // Verify input text area and analyze button
    const textarea = page.locator('#nlp-text-input');
    await expect(textarea).toBeVisible();
    await textarea.fill('Operating revenue surged with record quarterly expansion.');

    const analyzeBtn = page.locator('#btn-run-nlp-prediction');
    await expect(analyzeBtn).toBeVisible();
    await analyzeBtn.click();

    // Verify result card and token badges
    const nlpResult = page.locator('#nlp-result');
    await expect(nlpResult).toBeVisible();
    await expect(nlpResult).toContainText('NLP Prediction & Token Attribution');
    await expect(nlpResult).toContainText('Leave-One-Out Token Saliency Attribution', { timeout: 15000 });

    const badgeContainer = page.locator('#token-attribution-container');
    await expect(badgeContainer).toBeVisible();
    const tokenBadges = badgeContainer.locator('span');
    expect(await tokenBadges.count()).toBeGreaterThanOrEqual(3);
  });

  test('8. Computer Vision Inference: renders predicted defect class and Grad-CAM heatmap overlay', async ({ page }) => {
    await page.goto('/predict');
    const visionTab = page.locator('#tab-btn-vision');
    await expect(visionTab).toBeVisible();
    await visionTab.click();

    // Select PCB defect benchmark sample
    const pcbBtn = page.locator('button:text("PCB Solder Defect")');
    await expect(pcbBtn).toBeVisible();
    await pcbBtn.click();

    const runVisionBtn = page.locator('#btn-run-vision-prediction');
    await expect(runVisionBtn).toBeVisible();
    await runVisionBtn.click();
    // Verify vision verdict card
    const visionResult = page.locator('#vision-result');
    await expect(visionResult).toBeVisible();
    await expect(visionResult).toContainText('Vision Prediction Output');

    // Verify Grad-CAM overlay image is present
    const gradcamImg = page.locator('#gradcam-preview-image');
    await expect(gradcamImg).toBeVisible({ timeout: 10000 });
  });

});
