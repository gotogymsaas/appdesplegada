#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

function readText(filePath) {
  return fs.readFileSync(filePath, 'utf8');
}

function writeText(filePath, content) {
  fs.writeFileSync(filePath, content, 'utf8');
}

function getManifestPackage(manifestPath) {
  const xml = readText(manifestPath);
  const match = xml.match(/\bpackage\s*=\s*"([^"]+)"/);
  return match ? match[1] : null;
}

function ensureNamespaceInGradle(gradlePath, namespace) {
  const original = readText(gradlePath);

  // If already set, do nothing.
  if (/\bnamespace\b\s+['"][^'"]+['"]/.test(original)) {
    return { changed: false, reason: 'namespace already present' };
  }

  const lines = original.split(/\r?\n/);
  const androidOpenIndex = lines.findIndex((l) => /^\s*android\s*\{\s*$/.test(l));
  if (androidOpenIndex === -1) {
    return { changed: false, reason: 'android { block not found' };
  }

  const indentMatch = lines[androidOpenIndex].match(/^(\s*)/);
  const indent = (indentMatch ? indentMatch[1] : '') + '    ';

  lines.splice(androidOpenIndex + 1, 0, `${indent}namespace "${namespace}"`);
  const patched = lines.join('\n');
  writeText(gradlePath, patched);
  return { changed: true, reason: 'namespace inserted' };
}

function main() {
  const repoRoot = process.cwd();

  const targets = [
    {
      name: 'capacitor-google-fit',
      gradle: path.join(repoRoot, 'node_modules', 'capacitor-google-fit', 'android', 'build.gradle'),
      manifest: path.join(repoRoot, 'node_modules', 'capacitor-google-fit', 'android', 'src', 'main', 'AndroidManifest.xml'),
    },
  ];

  let anyChanged = false;

  for (const t of targets) {
    if (!fs.existsSync(t.gradle) || !fs.existsSync(t.manifest)) {
      // Dependency not installed in this environment.
      continue;
    }

    const pkg = getManifestPackage(t.manifest);
    if (!pkg) {
      console.warn(`[patch-capacitor-namespaces] ${t.name}: package not found in manifest; skipping`);
      continue;
    }

    const res = ensureNamespaceInGradle(t.gradle, pkg);
    if (res.changed) anyChanged = true;
    console.log(`[patch-capacitor-namespaces] ${t.name}: ${res.reason} (${pkg})`);
  }

  if (!anyChanged) {
    console.log('[patch-capacitor-namespaces] No changes needed');
  }
}

main();
