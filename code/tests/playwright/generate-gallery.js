// generate-gallery.js
// Scans test-output for generated PNGs and GIFs/videos and produces a standalone gallery viewer.
const fs = require('fs');
const path = require('path');

const testOutputDir = path.join(__dirname, 'test-output');
const demoArtifactsDir = path.join(testOutputDir, 'demo-artifacts');
const resultsDir = path.join(testOutputDir, 'results');
const docsScreenshots = path.join(__dirname, '..', '..', 'docs', 'screenshots');
const docsGifs = path.join(__dirname, '..', '..', 'docs', 'gifs');

function findMediaFiles(dir, extensions = ['.png', '.gif', '.webm', '.mp4']) {
  if (!fs.existsSync(dir)) return [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  let files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files = files.concat(findMediaFiles(fullPath, extensions));
    } else {
      const ext = path.extname(entry.name).toLowerCase();
      if (extensions.includes(ext)) {
        files.push(fullPath);
      }
    }
  }
  return files;
}

const allArtifacts = [
  ...findMediaFiles(demoArtifactsDir),
  ...findMediaFiles(resultsDir),
  ...findMediaFiles(testOutputDir, ['.png', '.gif']),
];

// Deduplicate
const uniquePaths = Array.from(new Set(allArtifacts));

const itemsHtml = uniquePaths.map(filePath => {
  const relPath = path.relative(testOutputDir, filePath).replace(/\\/g, '/');
  const fileName = path.basename(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const isVideo = ext === '.webm' || ext === '.mp4';
  const isGif = ext === '.gif';

  const badge = isVideo ? '<span class="badge video">VIDEO / WEBM</span>'
    : isGif ? '<span class="badge gif">GIF RECORDING</span>'
    : '<span class="badge png">PNG SCREENSHOT</span>';

  const mediaTag = isVideo 
    ? `<video src="${relPath}" controls style="max-width:100%; border-radius:8px;"></video>`
    : `<a href="${relPath}" target="_blank"><img src="${relPath}" alt="${fileName}" loading="lazy" style="max-width:100%; border-radius:8px; border:1px solid #33333d;" /></a>`;

  return `
    <div class="gallery-card">
      <div class="card-header">
        <span class="file-title">${fileName}</span>
        ${badge}
      </div>
      <div class="media-container">
        ${mediaTag}
      </div>
      <div class="card-footer">
        <code>${relPath}</code>
      </div>
    </div>
  `;
}).join('\n');

const galleryHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Sukio — Visual Test Outputs &amp; Demo Gallery</title>
  <style>
    :root {
      --bg: #0e0e10;
      --surface: #19191d;
      --surface-variant: #25252b;
      --primary: #d0bcff;
      --accent: #7c3aed;
      --text: #e7e4ec;
      --muted: #acaab1;
      --border: #33333d;
      --success: #34d399;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 2rem;
      min-height: 100vh;
    }
    header {
      max-width: 1400px;
      margin: 0 auto 2rem auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.5rem;
    }
    h1 {
      font-size: 1.85rem;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .meta {
      font-size: 0.875rem;
      color: var(--muted);
      margin-top: 0.25rem;
    }
    .actions {
      display: flex;
      gap: 0.75rem;
    }
    .btn {
      background: var(--accent);
      color: #fff;
      text-decoration: none;
      padding: 0.5rem 1rem;
      border-radius: 6px;
      font-weight: 600;
      font-size: 0.85rem;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }
    .btn-secondary {
      background: var(--surface-variant);
      border: 1px solid var(--border);
      color: var(--text);
    }
    .gallery-grid {
      max-width: 1400px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 1.5rem;
    }
    .gallery-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      box-shadow: 0 4px 20px rgba(0,0,0,0.4);
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .gallery-card:hover {
      transform: translateY(-2px);
      border-color: var(--primary);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .file-title {
      font-weight: 600;
      font-size: 0.95rem;
      color: #fff;
      font-family: monospace;
    }
    .badge {
      font-size: 0.7rem;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-weight: 700;
      letter-spacing: 0.05em;
    }
    .badge.png { background: rgba(208, 188, 255, 0.15); color: var(--primary); border: 1px solid rgba(208, 188, 255, 0.3); }
    .badge.gif { background: rgba(52, 211, 153, 0.15); color: var(--success); border: 1px solid rgba(52, 211, 153, 0.3); }
    .badge.video { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .card-footer {
      font-size: 0.75rem;
      color: var(--muted);
      border-top: 1px solid var(--border);
      padding-top: 0.5rem;
      word-break: break-all;
    }
    .empty-state {
      max-width: 600px;
      margin: 4rem auto;
      text-align: center;
      background: var(--surface);
      padding: 3rem;
      border-radius: 12px;
      border: 1px solid var(--border);
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
        Sukio Visual Test Outputs
      </h1>
      <div class="meta">Automated Playwright test runs, full-page screenshots, and demo-level screen recordings. (Untracked in git)</div>
    </div>
    <div class="actions">
      <a href="html-report/index.html" class="btn" target="_blank">Open Full HTML Report &rarr;</a>
    </div>
  </header>

  <main>
    ${uniquePaths.length > 0 ? `<div class="gallery-grid">${itemsHtml}</div>` : `
      <div class="empty-state">
        <h2 style="margin-bottom:0.75rem;">No test media found yet</h2>
        <p style="color:var(--muted); font-size:0.9rem; margin-bottom:1.5rem;">Run the test suite or demo capture to generate PNG screenshots and video recordings:</p>
        <code style="background:var(--surface-variant); padding:0.5rem 1rem; border-radius:6px; font-size:0.85rem;">npm run test:demo</code>
      </div>
    `}
  </main>
</body>
</html>
`;

if (!fs.existsSync(testOutputDir)) {
  fs.mkdirSync(testOutputDir, { recursive: true });
}

const galleryPath = path.join(testOutputDir, 'gallery.html');
fs.writeFileSync(galleryPath, galleryHtml, 'utf8');
console.log(`[Gallery] Visual artifact gallery generated: ${galleryPath} (${uniquePaths.length} media items found)`);
