// test-cors.ts
import { test, expect } from '@playwright/test';

test('CORS headers present', async ({ request }) => {
  const response = await request.get('http://localhost:8000/api/health', {
    headers: { 'Origin': 'http://localhost:3000' }
  });
  const headers = response.headers();
  expect(headers['access-control-allow-origin']).toBeDefined();
  expect(headers['access-control-allow-origin']).toContain('localhost');
});
