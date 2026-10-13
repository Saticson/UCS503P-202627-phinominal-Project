import { test, expect } from '@playwright/test';

test.describe('Sukio — User Issue Verification & End-to-End Suite', () => {

  test('1. Titanic benchmark loads cleanly without JSON dump in name column', async ({ page }) => {
    await page.goto('/datasets');
    await expect(page.locator('h1')).toContainText('Datasets');

    // Click Titanic benchmark button
    console.log('[Test] Clicking Titanic benchmark button...');
    await page.click('button:text("Titanic")');
    await page.waitForTimeout(1000);

    // Locate the table row for Titanic
    const tableBody = page.locator('#dataset-table-body');
    await expect(tableBody).toBeVisible();

    const titanicRow = tableBody.locator('tr:has-text("Titanic Survival")').first();
    await expect(titanicRow).toBeVisible();

    // Verify Name cell does NOT contain raw JSON
    const rowText = await titanicRow.innerText();
    console.log('[Test] Titanic row text:', rowText);
    expect(rowText).not.toContain('"columns_info"');
    expect(rowText).not.toContain('"preview_data"');
    expect(rowText).not.toContain('{"id":');
    expect(rowText).toContain('Titanic Survival');
    expect(rowText).toContain('survived');
  });

  test('2. Manual CSV upload renders clean dataset name without raw JSON dump', async ({ page }) => {
    await page.goto('/datasets');

    const csvContent = 'pclass,survived,sex,age,fare\n1,1,female,29,150.0\n3,0,male,22,7.25\n1,0,male,45,55.0\n';
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'manual_passengers.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent)
    });

    await page.fill('input[name="name"]', 'manual_titanic_upload');
    await page.fill('input[name="target_column"]', 'survived');

    console.log('[Test] Submitting manual CSV upload...');
    await page.click('button:text("Upload Dataset")');
    await page.waitForTimeout(1000);

    // Verify table row
    const manualRow = page.locator('#dataset-table-body tr:has-text("manual_titanic_upload")').first();
    await expect(manualRow).toBeVisible();

    const rowText = await manualRow.innerText();
    console.log('[Test] Manual dataset row text:', rowText);
    expect(rowText).not.toContain('"columns_info"');
    expect(rowText).not.toContain('"preview_data"');
    expect(rowText).not.toContain('{"id":');
    expect(rowText).toContain('manual_titanic_upload');
    expect(rowText).toContain('survived');
  });

  test('3. Train button passes dataset to AutoML Studio and executes training to completion', async ({ page }) => {
    await page.goto('/datasets');

    // Ensure Titanic is present
    const titanicRow = page.locator('#dataset-table-body tr:has-text("Titanic Survival")').first();
    if (await titanicRow.count() === 0) {
      await page.click('button:text("Titanic")');
      await page.waitForTimeout(1000);
    }

    // Click "Train ->" link on the Titanic row
    console.log('[Test] Clicking Train -> on Titanic row...');
    await page.locator('#dataset-table-body tr:has-text("Titanic Survival") a:has-text("Train")').first().click();

    // Verify navigated to training studio with dataset selected
    await expect(page).toHaveURL(/\/training/);
    await expect(page.locator('h1')).toContainText('AutoML Training Studio');

    // Verify target column is set to survived
    const targetVal = await page.inputValue('#target_column');
    expect(targetVal).toBe('survived');

    // Launch experiment
    console.log('[Test] Launching AutoML Experiment...');
    await page.click('#btn-launch-training');

    // Verify training status card updates
    const statusCard = page.locator('#training-status');
    await expect(statusCard).toBeVisible();

    // Ensure no raw JSON was dumped into #training-status
    const statusText = await statusCard.innerText();
    expect(statusText).not.toContain('{"job_id"');
    expect(statusText).not.toContain('{"status"');

    // Wait for completion (simulation finishes quickly in background)
    console.log('[Test] Waiting for training completion...');
    await expect(statusCard.locator('.status-pill:has-text("COMPLETED")')).toBeVisible({ timeout: 15000 });
    await expect(statusCard.locator(':has-text("Champion Model Selected")').first()).toBeVisible();

    const championText = await statusCard.innerText();
    console.log('[Test] Training completed successfully. Champion summary:\n', championText);
  });

  test('4. Model Leaderboard displays champion model with clean metrics', async ({ page }) => {
    await page.goto('/models');
    await expect(page.locator('h1')).toContainText('Model Leaderboard');

    const tableBody = page.locator('#models-table-body');
    await expect(tableBody).toBeVisible();

    // Verify at least one model is present with Champion badge
    await expect(tableBody).not.toContainText('No trained models found');
    await expect(tableBody.locator('.badge:has-text("CHAMPION")').first()).toBeVisible();

    // Verify Test Inference button exists
    await expect(tableBody.locator('a:has-text("Test Inference")').first()).toBeVisible();
  });

  test('5. Prediction Sandbox runs real-time inference cleanly', async ({ page }) => {
    await page.goto('/predict');
    await expect(page.locator('h1')).toContainText('Inference & Prediction Sandbox');

    // Fill sample prediction features
    const sampleFeatures = JSON.stringify({
      pclass: 1,
      sex: 'female',
      age: 29,
      fare: 150.0
    });
    await page.fill('textarea[name="features"]', sampleFeatures);

    console.log('[Test] Triggering live prediction...');
    await page.click('#btn-run-prediction');
    await page.waitForTimeout(800);

    const resultBox = page.locator('#prediction-result');
    await expect(resultBox).toBeVisible();
    await expect(resultBox.locator('.badge:has-text("SUCCESS")')).toBeVisible();
    await expect(resultBox.locator('#prediction-value')).toBeVisible();

    const predVal = await resultBox.locator('#prediction-value').innerText();
    console.log('[Test] Prediction output value:', predVal);
    expect(predVal.trim().length).toBeGreaterThan(0);
  });

});
