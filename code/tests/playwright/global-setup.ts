// global-setup.ts
import { execSync } from 'child_process';
import { existsSync, mkdirSync } from 'fs';
import { join } from 'path';

export default async () => {
  // Ensure storage and subdirectories exist
  const storageDir = join(__dirname, '..', '..', 'storage');
  if (!existsSync(storageDir)) {
    mkdirSync(storageDir, { recursive: true });
  }
  for (const sub of ['datasets', 'models', 'reports']) {
    const p = join(storageDir, sub);
    if (!existsSync(p)) {
      mkdirSync(p, { recursive: true });
    }
  }

  // Ensure test-output directory exists
  const testOutputDir = join(__dirname, 'test-output');
  if (!existsSync(testOutputDir)) {
    mkdirSync(testOutputDir, { recursive: true });
  }

  // Check if backend is already listening
  try {
    const probe = await fetch('http://localhost:8000/api/health').catch(() => null);
    if (probe && probe.ok) {
      console.log('Backend is already running on http://localhost:8000');
      return;
    }

    console.log('Attempting to start services via Docker compose...');
    try {
      execSync('docker compose up -d', { stdio: 'pipe', cwd: join(__dirname, '..', '..') });
    } catch (dockerErr) {
      console.warn('Docker daemon not available or docker compose skipped.');
    }

    // Wait for services to be ready
    console.log('Checking backend service availability...');
    let ready = false;
    for (let i = 0; i < 15; i++) {
      try {
        const response = await fetch('http://localhost:8000/api/health');
        if (response.ok) {
          ready = true;
          break;
        }
      } catch (e) {
        // wait
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    if (ready) {
      console.log('Backend services are ready');
    } else {
      console.log('Backend not reached yet; tests may run against standalone instances.');
    }
  } catch (error) {
    console.warn('Global setup finished with notice:', error);
  }
};
