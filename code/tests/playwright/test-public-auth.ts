// test-public-auth.ts
import { test, expect } from '@playwright/test';

test('public routes accessible without auth', async ({ request }) => {
  const health = await request.get('/api/health');
  expect(health.status()).toBe(200);
  
  const authUser = await request.get('/api/auth/user/test-uid');
  expect([200, 404]).toContain(authUser.status()); // 404 ok if user doesn't exist
});
