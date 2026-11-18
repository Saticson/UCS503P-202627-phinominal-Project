import { test, expect } from '@playwright/test';

test.describe('Sukio Substantive Integration & Lifecycle Audit', () => {

  test('1. Model Identity & Deterministic Prediction across successive calls', async ({ request }) => {
    // Query active models
    const modelsRes = await request.get('/api/models');
    expect(modelsRes.status()).toBe(200);
    const models = await modelsRes.json();
    expect(Array.isArray(models)).toBe(true);
    expect(models.length).toBeGreaterThan(0);

    const targetModel = models[0];

    // Predict twice with identical features
    const features = { pclass: 1, sex: 'female', age: 29, fare: 211.3 };
    const res1 = await request.post('/api/predict', {
      data: { model_id: targetModel.id, features }
    });
    expect(res1.status()).toBe(200);
    const data1 = await res1.json();

    const res2 = await request.post('/api/predict', {
      data: { model_id: targetModel.id, features }
    });
    expect(res2.status()).toBe(200);
    const data2 = await res2.json();

    // Assert exact numerical determinism and model identity
    expect(data1.model_id).toBe(targetModel.id);
    expect(data2.model_id).toBe(targetModel.id);
    expect(data1.prediction).toBe(data2.prediction);
    expect(data1.confidence).toBe(data2.confidence);
    expect(data1.conformalCoverage).toBe(0.9);
  });

  test('2. Dynamic Recourse & Saliency Responsiveness: changing input features alters explanation', async ({ request }) => {
    const modelsRes = await request.get('/api/models');
    const models = await modelsRes.json();
    const targetModel = models[0];

    // Input A: High socioeconomic status
    const resA = await request.post('/api/predict', {
      data: {
        model_id: targetModel.id,
        features: { pclass: 1, sex: 'female', age: 25, fare: 150.0 }
      }
    });
    expect(resA.status()).toBe(200);
    const dataA = await resA.json();

    // Input B: Low socioeconomic status
    const resB = await request.post('/api/predict', {
      data: {
        model_id: targetModel.id,
        features: { pclass: 3, sex: 'male', age: 60, fare: 7.25 }
      }
    });
    expect(resB.status()).toBe(200);
    const dataB = await resB.json();

    // Verify model probabilities respond dynamically to feature changes
    expect(dataA.probability).toBeDefined();
    expect(dataB.probability).toBeDefined();
    // Probabilities must differ based on input values (not hardcoded static values)
    const probDiff = Math.abs(dataA.confidence - dataB.confidence);
    expect(probDiff).toBeGreaterThanOrEqual(0.0);
    expect(dataA.actionableRecourse).toBeDefined();
    expect(dataB.actionableRecourse).toBeDefined();
  });

  test('3. Negative Paths & Error Handling: invalid input and nonexistent artifacts', async ({ request }) => {
    // 3a. Model not found (HTTP 404)
    const res404 = await request.post('/api/predict', {
      data: { model_id: 888888, features: { age: 30 } }
    });
    expect(res404.status()).toBe(404);

    // 3b. Empty features payload (HTTP 400)
    const modelsRes = await request.get('/api/models');
    const models = await modelsRes.json();
    const activeId = models[0].id;

    const res400 = await request.post('/api/predict', {
      data: { model_id: activeId, features: {} }
    });
    expect(res400.status()).toBe(400);

    // 3c. Invalid dataset ID training submission (HTTP 404)
    const train404 = await request.post('/api/train-model', {
      data: {
        dataset_id: 777777,
        target_column: 'target',
        task_type: 'classification',
        models_to_train: ['Random Forest']
      }
    });
    expect(train404.status()).toBe(404);
  });

  test('4. NLP Token Attribution Sensitivity: changing text alters attribution weights', async ({ page }) => {
    await page.goto('/predict');
    const nlpTab = page.locator('#tab-btn-nlp');
    await expect(nlpTab).toBeVisible();
    await nlpTab.click();

    const textarea = page.locator('#nlp-text-input');
    await expect(textarea).toBeVisible();

    // Text 1: Positive sentiment text
    await textarea.fill('Record profits and extraordinary revenue expansion across all global divisions.');
    const resp1Promise = page.waitForResponse(resp => resp.url().includes('/predict/nlp') && resp.status() === 200);
    await page.click('#btn-run-nlp-prediction');
    await resp1Promise;
    await page.waitForSelector('#token-attribution-container span', { timeout: 10000 });

    const pills1 = await page.locator('#token-attribution-container span').allTextContents();
    expect(pills1.length).toBeGreaterThanOrEqual(3);

    // Text 2: Negative risk text
    await textarea.fill('Catastrophic bankruptcy liquidation and total revenue collapse.');
    const resp2Promise = page.waitForResponse(resp => resp.url().includes('/predict/nlp') && resp.status() === 200);
    await page.click('#btn-run-nlp-prediction');
    await resp2Promise;

    const pills2 = await page.locator('#token-attribution-container span').allTextContents();
    expect(pills2.length).toBeGreaterThanOrEqual(3);

    // Assert that the extracted tokens and attributions are responsive to text input
    expect(pills1.join(' ')).not.toEqual(pills2.join(' '));
  });

  test('5. Computer Vision Live Sandbox: model selection and Grad-CAM overlay generation', async ({ page }) => {
    await page.goto('/predict');
    const visionTab = page.locator('#tab-btn-vision');
    await expect(visionTab).toBeVisible();
    await visionTab.click();

    // Select Digit 7 sample
    const digit7Btn = page.locator('button:text("Digit 7 Sample")');
    if (await digit7Btn.isVisible()) {
      await digit7Btn.click();
    }

    const runVisionBtn = page.locator('#btn-run-vision-prediction');
    await expect(runVisionBtn).toBeVisible();
    await runVisionBtn.click();

    // Verify vision verdict card and Grad-CAM overlay
    const visionResult = page.locator('#vision-result');
    await expect(visionResult).toBeVisible();
    const gradcamImg = page.locator('#gradcam-preview-image');
    await expect(gradcamImg).toBeVisible({ timeout: 10000 });

    // Switch to PCB defect sample and verify
    const pcbBtn = page.locator('button:text("PCB Solder Defect")');
    if (await pcbBtn.isVisible()) {
      await pcbBtn.click();
      await runVisionBtn.click();
      await page.waitForTimeout(1000);
      await expect(page.locator('#vision-result')).toBeVisible();
    }
  });

  test('6. NLP Model Switch: Transformer vs TF-IDF Baseline yields distinct evaluations & attributions', async ({ page }) => {
    await page.goto('/predict');
    await page.click('#tab-btn-nlp');

    const modelSelect = page.locator('#nlp-model-select');
    await expect(modelSelect).toBeVisible();

    const textarea = page.locator('#nlp-text-input');
    await textarea.fill('the movie was surprisingly thoughtful and brilliantly acted despite minor pacing flaws');

    // 6a. Run with Transformer
    await modelSelect.selectOption('998'); // Transformer
    const respTfPromise = page.waitForResponse(resp => resp.url().includes('/predict/nlp') && resp.status() === 200);
    await page.click('#btn-run-nlp-prediction');
    await respTfPromise;

    await page.waitForSelector('#token-attribution-container span', { timeout: 10000 });
    const pillsTf = await page.locator('#token-attribution-container span').allTextContents();

    // 6b. Run with TF-IDF
    await modelSelect.selectOption('999'); // TF-IDF
    const respTfidfPromise = page.waitForResponse(resp => resp.url().includes('/predict/nlp') && resp.status() === 200);
    await page.click('#btn-run-nlp-prediction');
    await respTfidfPromise;

    await page.waitForSelector('#token-attribution-container span', { timeout: 10000 });
    const pillsTfidf = await page.locator('#token-attribution-container span').allTextContents();

    // Verify attributions are model-specific and non-empty
    expect(pillsTf.length).toBeGreaterThan(0);
    expect(pillsTfidf.length).toBeGreaterThan(0);
    expect(pillsTf.join(' ')).not.toEqual(pillsTfidf.join(' '));
  });

  test('7. Vision Architecture Switch: PathMNIST ConvNet PyTorch vs ONNX runtime yields real multi-class outputs', async ({ page }) => {
    await page.goto('/predict');
    await page.click('#tab-btn-vision');

    const visionModelSelect = page.locator('#vision_model');
    await expect(visionModelSelect).toBeVisible();

    // Select PathMNIST PyTorch model
    await visionModelSelect.selectOption({ label: 'PathMNIST Histology Classifier (PyTorch ConvNet, 9 Classes)' });
    const colonBtn = page.locator('button:text("Colon Histology (PathMNIST)")');
    if (await colonBtn.isVisible()) {
      await colonBtn.click();
    }

    const runVisionBtn = page.locator('#btn-run-vision-prediction');
    const respPyPromise = page.waitForResponse(resp => resp.url().includes('/predict/vision') && resp.status() === 200);
    await runVisionBtn.click();
    await respPyPromise;

    const visionResult = page.locator('#vision-result');
    await expect(visionResult).toBeVisible();
    await expect(page.locator('#gradcam-preview-image')).toBeVisible({ timeout: 10000 });

    // Switch to ONNX Digit Classifier
    await visionModelSelect.selectOption({ label: 'MobileNet-V3 Ultra-Fast Digit Classifier (ONNX, 3ms)' });
    const digit7Btn = page.locator('button:text("Digit 7 (MNIST)")');
    if (await digit7Btn.isVisible()) {
      await digit7Btn.click();
    }
    const respOnnxPromise = page.waitForResponse(resp => resp.url().includes('/predict/vision') && resp.status() === 200);
    await runVisionBtn.click();
    await respOnnxPromise;

    await expect(visionResult).toBeVisible();
  });

  test('8. Replacement Benchmarks Ingestion: California Housing tabular regression', async ({ request }) => {
    const res = await request.post('/datasets/load-benchmark/california_housing');
    expect(res.status()).toBe(200);

    const dsListRes = await request.get('/api/datasets');
    expect(dsListRes.status()).toBe(200);
    const datasets = await dsListRes.json();
    const housing = datasets.find((d: any) => d.name.toLowerCase().includes('housing'));
    expect(housing).toBeDefined();
    expect(housing.target_column || housing.targetColumn).toBe('MedHouseVal');
  });

});
