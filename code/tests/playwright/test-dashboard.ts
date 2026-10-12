// test-dashboard.ts
import { test, expect } from '@playwright/test';

test('dashboard loads HTMX content', async ({ page }) => {
  await page.goto('/dashboard');
  await expect(page.locator('h2')).toContainText('Quick Stats');
  await page.waitForTimeout(1000); // allow HTMX load
  await expect(page.locator('#datasets')).toBeVisible();
});
