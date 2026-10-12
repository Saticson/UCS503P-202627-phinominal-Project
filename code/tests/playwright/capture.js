import { chromium } from 'playwright';
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto('http://localhost:8000/');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'test-output/prototype.png' });
  await browser.close();
  console.log('Screenshot saved: test-output/prototype.png');
})();
