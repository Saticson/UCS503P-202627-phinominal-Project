import { chromium } from 'playwright';
(async () => {
  const bravePath = process.env.BRAVE_PATH || 'C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe';
  const browser = await chromium.launch({ headless: false, slowMo: 400, executablePath: bravePath });
  const page = await browser.newPage();
  await page.goto('http://localhost:8000/');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'test-output/prototype.png' });
  await browser.close();
  console.log('Screenshot saved: test-output/prototype.png');
})();
