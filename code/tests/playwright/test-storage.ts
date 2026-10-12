// test-storage.ts
import { test, expect } from '@playwright/test';
import { existsSync } from 'fs';
import { join } from 'path';

test('storage directories exist', async ({ request }) => {
  const base = join(__dirname, '..', '..', 'storage');
  expect(existsSync(join(base, 'datasets'))).toBeTruthy();
  expect(existsSync(join(base, 'models'))).toBeTruthy();
  expect(existsSync(join(base, 'reports'))).toBeTruthy();
});
