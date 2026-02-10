// Root service worker shim to keep scope at "/"
// It loads the real worker from /js/sw.js
try {
  importScripts('/js/sw.js');
} catch (e) {
  // Fail silently to avoid blocking page load
}
