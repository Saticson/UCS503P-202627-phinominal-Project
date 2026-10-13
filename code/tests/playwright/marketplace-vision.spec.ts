import { test, expect } from '@playwright/test';

test.describe('Sukio Marketplace & Computer Vision Integration Suite', () => {

  test('1. Marketplace Catalog: displays verified models and filters by modality', async ({ page }) => {
    await page.goto('/marketplace');
    await expect(page.locator('h1')).toContainText('Sukio Solution Marketplace');

    // Verify initial listings count
    const grid = page.locator('#listings-grid');
    await expect(grid).toBeVisible();
    const countAll = await grid.locator('.card-listing').count();
    console.log(`[Test 1] Total catalog listings: ${countAll}`);
    expect(countAll).toBeGreaterThanOrEqual(6);

    // Filter by Computer Vision
    console.log('[Test 1] Clicking Computer Vision filter...');
    await page.click('#filter-computer-vision');
    await expect(page).toHaveURL(/modality=COMPUTER_VISION/);
    
    // Ensure vision listings are visible
    await expect(page.locator('.card-listing:has-text("ResNet-50")').first()).toBeVisible();
    await expect(page.locator('.card-listing:has-text("MobileNet-V3")').first()).toBeVisible();
    await expect(page.locator('.card-listing:has-text("YOLOv8")').first()).toBeVisible();
    // Tabular models should not be visible in vision filter
    await expect(page.locator('.card-listing:has-text("XGBoost")')).toHaveCount(0);

    // Filter by Tabular
    console.log('[Test 1] Clicking Tabular filter...');
    await page.click('#filter-tabular');
    await expect(page).toHaveURL(/modality=TABULAR/);
    await expect(page.locator('.card-listing:has-text("XGBoost")').first()).toBeVisible();
    await expect(page.locator('.card-listing:has-text("LightGBM")').first()).toBeVisible();
    await expect(page.locator('.card-listing:has-text("ResNet-50")')).toHaveCount(0);
  });

  test('2. Listing Detail View: displays verified audit specs and model card', async ({ page }) => {
    await page.goto('/marketplace/resnet-50-imagenet');
    await expect(page.locator('h1')).toContainText('ResNet-50 Enterprise Visual Classifier');

    // Verify Sukio Verified badge
    await expect(page.locator('.badge-verified')).toContainText('Sukio Verified Asset');

    // Verify benchmark metrics
    await expect(page.locator('.detail-tile:has-text("92.4%")')).toBeVisible();
    await expect(page.locator('.detail-tile:has-text("14 ms")')).toBeVisible();
    await expect(page.locator('.detail-tile:has-text("PyTorch")')).toBeVisible();
    await expect(page.locator('.detail-tile:has-text("[3, 224, 224]")')).toBeVisible();
  });

  test('3. Vision Dataset: loads MNIST benchmark with VISION modality badge', async ({ page }) => {
    await page.goto('/datasets');
    await expect(page.locator('h1')).toContainText('Datasets & Diagnostics');

    console.log('[Test 3] Loading MNIST (Vision) benchmark...');
    await page.click('#btn-load-mnist');
    await page.waitForTimeout(1000);

    const tableBody = page.locator('#dataset-table-body');
    await expect(tableBody).toBeVisible();

    const mnistRow = tableBody.locator('tr:has-text("MNIST Digits")').first();
    await expect(mnistRow).toBeVisible();
    await expect(mnistRow.locator('.badge:has-text("VISION")')).toBeVisible();
    await expect(mnistRow).toContainText('label');
  });

  test('4. AutoML to Marketplace Bridge: "Find a Better Model" shortlist recommendation', async ({ page }) => {
    await page.goto('/training');
    await expect(page.locator('h1')).toContainText('AutoML Training Studio');

    // Launch experiment
    console.log('[Test 4] Launching AutoML sweep...');
    await page.click('#btn-launch-training');

    const statusCard = page.locator('#training-status');
    await expect(statusCard).toBeVisible();
    await expect(statusCard.locator('.status-pill:has-text("COMPLETED")')).toBeVisible({ timeout: 15000 });

    // Verify "Find a Better Model in Marketplace" bridge button exists
    const betterModelBtn = statusCard.locator('#btn-find-better-model');
    await expect(betterModelBtn).toBeVisible();

    console.log('[Test 4] Clicking "Find a Better Model in Marketplace ->"...');
    await betterModelBtn.click();

    // Verify navigated to marketplace with recommendation shortlist
    await expect(page).toHaveURL(/recommend=true/);
    const recBanner = page.locator('#recommendation-banner');
    await expect(recBanner).toBeVisible();
    await expect(recBanner).toContainText('Time-to-Shortlist (TTS): 3 Compatible Models Found');
  });

  test('5. Computer Vision Sandbox: executes live image classification with confidence and preview', async ({ page }) => {
    await page.goto('/predict');
    await expect(page.locator('h1')).toContainText('Inference & Prediction Sandbox');

    // Ensure on Computer Vision tab
    await page.click('#tab-btn-vision');

    // Run benchmark sample 1: Digit 7
    console.log('[Test 5] Testing Digit 7 vision inference...');
    await page.click('button:text("Digit 7 (MNIST)")');
    await page.click('#btn-run-vision-prediction');
    await page.waitForTimeout(800);

    const resultBox = page.locator('#vision-result');
    await expect(resultBox).toBeVisible();
    await expect(resultBox.locator('.badge:has-text("SUCCESS")')).toBeVisible();
    await expect(resultBox.locator('#vision-predicted-class')).toContainText('Digit 7');
    await expect(resultBox.locator('#vision-confidence')).toContainText('99.1%');
    await expect(resultBox.locator('#result-preview-image')).toBeVisible();

    // Run benchmark sample 2: PCB Defect
    console.log('[Test 5] Testing PCB Defect vision inference...');
    await page.click('button:text("PCB Solder Defect")');
    await page.click('#btn-run-vision-prediction');
    await page.waitForTimeout(800);

    await expect(resultBox.locator('#vision-predicted-class')).toContainText('Defective');
    await expect(resultBox.locator('#vision-confidence')).toContainText('94.6%');
  });

  test('6. Hire Funnel: publishes expert commission request and lists on board', async ({ page }) => {
    await page.goto('/hire');
    await expect(page.locator('h1')).toContainText('Expert Commission Board');

    // Fill new commission request
    console.log('[Test 6] Submitting expert commission request...');
    await page.fill('#client_name', 'Starlight Autonomous Systems');
    await page.fill('#email', 'engineering@starlight.io');
    await page.fill('#title', 'CubeSat Cloud Masking Vision Classifier');
    await page.selectOption('#modality', 'COMPUTER_VISION');
    await page.selectOption('#task_type', 'CLASSIFICATION');
    await page.fill('#budget', '$2,500');
    await page.fill('#description', 'Low-power 8-bit quantized ONNX model for real-time cloud masking on edge FPGA.');

    await page.click('#btn-submit-commission');
    await page.waitForTimeout(1000);

    // Verify added to Active Commission Board
    const commissionList = page.locator('#commission-list');
    await expect(commissionList.locator(':has-text("CubeSat Cloud Masking Vision Classifier")').first()).toBeVisible();
    await expect(commissionList.locator(':has-text("$2,500")').first()).toBeVisible();
  });

});
