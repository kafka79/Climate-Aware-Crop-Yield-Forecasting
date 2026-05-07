/**
 * Crop Yield Forecast — Service Worker
 *
 * [Marco · Apple]: "If a farmer is in a field with 2G connectivity, having this
 * cached as a lightweight app would be the 10/10 design victory."
 *
 * Strategy: Cache-First for all static assets (fonts, CSS, JS).
 *           Network-First for API / inference requests (fresh data preferred).
 *           Offline fallback page when network is completely unavailable.
 *
 * Cache tiers:
 *   STATIC_CACHE  — shell assets, fonts (long-lived, versioned by CACHE_VERSION)
 *   RESULT_CACHE  — last successful forecast responses (so field use shows
 *                   the most recent prediction even on 2G timeout)
 */

const CACHE_VERSION = "v1";
const STATIC_CACHE  = `crop-forecast-static-${CACHE_VERSION}`;
const RESULT_CACHE  = `crop-forecast-results-${CACHE_VERSION}`;

// Assets to pre-cache on install (app shell)
const PRECACHE_URLS = [
  "/",
  "/app/static/manifest.json",
  "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
];

// ── Install: pre-cache the app shell ─────────────────────────────────────────
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => {
      console.log("[SW] Pre-caching app shell");
      return cache.addAll(PRECACHE_URLS).catch((err) => {
        // Non-fatal: some resources may not be available at install time
        console.warn("[SW] Pre-cache partial failure:", err);
      });
    })
  );
  // Take control immediately — don't wait for old SW to expire
  self.skipWaiting();
});

// ── Activate: clean up stale caches from previous versions ───────────────────
self.addEventListener("activate", (event) => {
  const KEEP = new Set([STATIC_CACHE, RESULT_CACHE]);
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => !KEEP.has(key))
          .map((key) => {
            console.log("[SW] Deleting stale cache:", key);
            return caches.delete(key);
          })
      )
    )
  );
  self.clients.claim();
});

// ── Fetch: tiered strategy ────────────────────────────────────────────────────
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET and cross-origin requests we don't manage
  if (request.method !== "GET") return;

  // ── Static assets: Cache-First ──────────────────────────────────────────
  if (
    url.pathname.startsWith("/app/static/") ||
    url.hostname === "fonts.googleapis.com" ||
    url.hostname === "fonts.gstatic.com"
  ) {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached;
        return fetch(request).then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(STATIC_CACHE).then((cache) => cache.put(request, clone));
          }
          return response;
        });
      })
    );
    return;
  }

  // ── Streamlit prediction/inference calls: Network-First with result cache ─
  if (
    url.pathname.includes("/_stcore/stream") ||
    url.pathname.includes("/predict") ||
    url.searchParams.has("region")
  ) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(RESULT_CACHE).then((cache) => {
              cache.put(request, clone);
              // Keep result cache lean: max 20 entries
              cache.keys().then((keys) => {
                if (keys.length > 20) cache.delete(keys[0]);
              });
            });
          }
          return response;
        })
        .catch(() => {
          // Network unavailable — serve last cached result
          return caches.match(request).then((cached) => {
            if (cached) {
              console.warn("[SW] Offline: serving cached forecast result");
              return cached;
            }
            // Nothing cached — return a minimal offline indicator
            return new Response(
              JSON.stringify({
                offline: true,
                message:
                  "You are offline. Connect to the network to fetch a fresh forecast. " +
                  "Your last successful result is shown below.",
              }),
              {
                status: 503,
                headers: { "Content-Type": "application/json" },
              }
            );
          });
        })
    );
    return;
  }

  // ── Everything else: default browser behaviour ───────────────────────────
  // (no SW interception — keeps Streamlit's WebSocket comms unaffected)
});

// ── Background sync: retry failed prediction requests when back online ────────
self.addEventListener("sync", (event) => {
  if (event.tag === "retry-prediction") {
    console.log("[SW] Background sync: retrying prediction request");
    // Streamlit handles the re-render on reconnect — just log the event
  }
});
