// test-home.ts
import { test, expect } from '@playwright/test';

test('home page renders with navigation', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('h1')).toContainText('Welcome to AxiomCloud');
  await expect(page.locator('nav a')).toContainText(['Home', 'Dashboard', 'Datasets', 'Training']);
});

test('dashboard link works', async ({ page }) => {
  await page.goto('/');
  await page.click('a:text(\"Dashboard\")');
  await expect(page).toHaveURL(/\/dashboard/);
  await expect(page.locator('h1')).toBeVisible();
});
