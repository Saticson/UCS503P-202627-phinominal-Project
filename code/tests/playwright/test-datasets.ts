// test-datasets.ts
import { test, expect } from '@playwright/test';

test('datasets list loads via HTMX', async ({ page }) => {
  await page.goto('/datasets');
  await expect(page.locator('#datasets')).toBeVisible();
  await expect(page.locator('button:text(\"Upload Dataset\")')).toBeVisible();
});

test('upload dataset via HTMX', async ({ page }) => {
  await page.goto('/datasets');
  const fileInput = page.locator('input[type=\"file\"]');
  if (await fileInput.count() > 0) {
    await fileInput.setInputFiles({
      name: 'test.csv',
      buffer: Buffer.from('col1,col2\n1,2\n3,4\n'),
      mimeType: 'text/csv'
    });
  }
  await page.click('button:text(\"Upload Dataset\")');
  await expect(page.locator('#datasets')).toHaveText(/upload/i, { timeout: 5000 });
});
