self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
    // 1. Abaikan skrip Cloudflare agar tidak terjadi konflik
    if (event.request.url.includes('/cdn-cgi/')) {
        return;
    }

    event.respondWith(
        fetch(event.request).then((response) => {
            // 2. Cegah modifikasi pada balasan kosong (204/304) yang menyebabkan Crash
            if (response.status === 0 || response.status === 204 || response.status === 304) {
                return response;
            }

            // 3. Suntikkan header keamanan secara aman
            const newHeaders = new Headers(response.headers);
            newHeaders.set('Cross-Origin-Embedder-Policy', 'require-corp');
            newHeaders.set('Cross-Origin-Opener-Policy', 'same-origin');

            return new Response(response.body, {
                status: response.status,
                statusText: response.statusText,
                headers: newHeaders
            });
        }).catch((e) => {
            console.error("Service Worker Error:", e);
        })
    );
});
