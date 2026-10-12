// global-setup.ts
import { defineConfig, devices } from '@playwright/test';
import { execSync } from 'child_process';
import { existsSync, mkdirSync } from 'fs';
import { join } from 'path';

// This runs before all tests
export default async () => {
  // Ensure storage directory exists
  const storageDir = join(__dirname, '..', '..', 'storage');
  if (!existsSync(storageDir)) {
    mkdirSync(storageDir, { recursive: true });
  }

  // Start docker compose (assuming we are in the code directory)
  try {
    console.log('Starting Docker compose...');
    execSync('docker compose up -d', { stdio: 'inherit', cwd: join(__dirname, '..', '..') });
    
    // Wait for services to be ready
    console.log('Waiting for services to be ready...');
    let ready = false;
    for (let i = 0; i < 30; i++) {
      try {
        const response = await fetch('http://localhost:8000/api/health');
        if (response.ok) {
          ready = true;
          break;
        }
      } catch (e) {
        // not ready yet
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    if (!ready) {
      throw new Error('Backend did not become ready within 30 seconds');
    }
    
    console.log('Services are ready');
  } catch (error) {
    console.error('Failed to start services:', error);
    process.exit(1);
  }
};
