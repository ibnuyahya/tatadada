import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v4.1
 * ./core/VideotronYouTubeEngine.js
 *
 * Outdoor Campus Monopole LED Videotron — Live YouTube / Playlist / 2-Layer Blurred Shorts CSS3D Engine:
 * 1. Zero-Extra-WebGL-Draw-Call (+0 Draw Calls) Hardware-Depth-Occluded CSS3D + WebGL Alpha Punch-Through:
 *    - Positions a 1280x720 px CSS3D DOM plane (0.01 m/px === 12.80 m x 7.20 m) directly behind the WebGL2 canvas,
 *      mathematically locked to Three.js PerspectiveCamera (including `camera.setViewOffset` drawer shifts!)
 *      with 0.000 px projective drift.
 *    - Foreground 3D geometry (campus trees, streetlamps, Gedung Rektorat UI, walking Sivitas NPCs, & 3D raindrops)
 *      naturally occludes the live YouTube video via the hardware WebGL depth buffer.
 * 2. Universal YouTube Input Parser:
 *    - Accepts single YouTube URLs (`watch?v=`, `youtu.be/`, `embed/`, `live/`), Playlist URLs (`playlist?list=`,
 *      `watch?v=...&list=...`, `embed/videoseries?list=...`), Shorts URLs (`/shorts/...`), raw Video/Playlist IDs,
 *      or full `<div><iframe ...></iframe></div>` HTML embed snippets.
 * 3. Automatic Vertical / YouTube Shorts Content Detection & 2-Layer Full-Width Blurred Display:
 *    - Detects vertical Shorts synchronously from `/shorts/` URLs or portrait `<iframe>` attributes, AND
 *      asynchronously for standard `watch?v=` / `youtu.be/` / playlist tracks whose content turns out to be a Short
 *      via YouTube oEmbed (`width < height`), `noembed.com`, and `oar2.jpg` portrait aspect-ratio probing.
 *    - Renders Shorts using a true 2-layer broadcast technique inside the 16:9 (1280x720) LED cabinet:
 *      - Layer 1 (Back Full-Width Blur): Muted synchronized 405x720 (9:16) YouTube iframe + instant portrait
 *        thumbnail pre-fill scaled 3.52x across the full 1280px width with `blur(20px) saturate(1.28) brightness(0.66)`
 *        (zero black pillarbox bars!).
 *      - Layer 2 (Front Sharp Center): Crisp 405x720 (9:16) foreground YouTube iframe centered with subtle optical
 *        edge vignette separation.
 */

const YT_ID_REGEX = /^[A-Za-z0-9_-]{11}$/;
const YT_LIST_ID_REGEX = /^(PL|UU|FL|LP|RD|OLAK5uy_)[A-Za-z0-9_-]{8,}$/;
const NON_EMBEDDABLE_LISTS = new Set(['WL', 'LL']);

export class VideotronYouTubeEngine {
  static SPEC = Object.freeze({
    cx: 28.4,
    cz: 78.2,
    yaw: -0.82928487, // Exact tangent to Lapangan Rotunda UI circle (-2.5, 106.5): atan2(-30.9, 28.3) = -47.51 deg
    screenCenterY: 9.48,
    screenW: 12.80,
    screenH: 7.20,
    screenMinY: 5.88,
    screenMaxY: 13.08,
    screenFwdOffset: 0.302,
    cabinetHalfW: 6.84,
    cabinetMinY: 5.36,
    cabinetMaxY: 13.86,
    domWidthPx: 1280,
    domHeightPx: 720,
    shortDomWidthPx: 405,
    shortDomHeightPx: 720,
    metersPerPixel: 0.01
  });

  static DEFAULT_VIDEO_ID = 'N5vSW33FwQQ';
  static DEFAULT_EMBED_URL = 'https://www.youtube.com/embed/N5vSW33FwQQ?rel=0';
  static DEFAULT_EMBED_HTML =
    '<div style="left: 0; width: 100%; height: 0; position: relative; padding-bottom: 56.25%;">' +
    '<iframe src="https://www.youtube.com/embed/N5vSW33FwQQ?rel=0" ' +
    'style="top: 0; left: 0; width: 100%; height: 100%; position: absolute; border: 0;" ' +
    'allowfullscreen scrolling="no" ' +
    'allow="accelerometer *; clipboard-write *; encrypted-media *; gyroscope *; picture-in-picture *; web-share *;" ' +
    'referrerpolicy="strict-origin"></iframe></div>';

  static _shortMetaCache = new Map();

  /**
   * Parses start timestamp strings such as "90", "1m30s", "2h3m4s" into integer seconds.
   */
  static _parseTimestampSeconds(raw) {
    if (!raw) return 0;
    const str = String(raw).trim().toLowerCase();
    if (/^\d+$/.test(str)) return Math.max(0, parseInt(str, 10) || 0);
    const hms = str.match(/^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?$/);
    if (!hms) return 0;
    const h = parseInt(hms[1] || '0', 10) || 0;
    const m = parseInt(hms[2] || '0', 10) || 0;
    const s = parseInt(hms[3] || '0', 10) || 0;
    return Math.max(0, h * 3600 + m * 60 + s);
  }

  /**
   * Universal parser for YouTube single links, playlists, Shorts, raw IDs, or pasted `<iframe>` HTML.
   */
  static parseYouTubeInput(rawInput) {
    const raw = typeof rawInput === 'string' ? rawInput.trim() : '';
    if (!raw) {
      return {
        valid: false,
        videoId: null,
        listId: null,
        isPlaylist: false,
        explicitShort: false,
        startSec: 0,
        normalizedInput: '',
        sourceLabel: '',
        error: 'Masukkan link YouTube (single / playlist / Shorts) atau kode embed <iframe>.'
      };
    }

    let candidate = raw;
    let explicitShort = false;

    // 1. Extract `src` and aspect hints if user pasted an HTML `<div><iframe ...></iframe></div>` snippet
    if (/<iframe\b/i.test(raw) || /src\s*=\s*["']/i.test(raw)) {
      const srcMatch = raw.match(/src\s*=\s*["']([^"']+)["']/i);
      if (srcMatch && srcMatch[1]) {
        candidate = srcMatch[1].replace(/&amp;/gi, '&').trim();
      }
      const padMatch = raw.match(/padding-bottom\s*:\s*([\d.]+)\s*%/i);
      if (padMatch && Number(padMatch[1]) > 100.0) {
        explicitShort = true;
      }
      const wMatch = raw.match(/\bwidth\s*=\s*["']?(\d+(?:\.\d+)?)\b/i);
      const hMatch = raw.match(/\bheight\s*=\s*["']?(\d+(?:\.\d+)?)\b/i);
      if (wMatch && hMatch) {
        const wNum = Number(wMatch[1]);
        const hNum = Number(hMatch[1]);
        if (wNum > 10 && hNum > wNum * 1.08) {
          explicitShort = true;
        }
      }
    }

    // 2. Direct bare 11-character video ID or bare playlist ID
    if (YT_ID_REGEX.test(candidate)) {
      return {
        valid: true,
        videoId: candidate,
        listId: null,
        isPlaylist: false,
        explicitShort: false,
        startSec: 0,
        normalizedInput: `https://www.youtube.com/embed/${candidate}?rel=0`,
        sourceLabel: `Video · ${candidate}`,
        error: null
      };
    }

    if (YT_LIST_ID_REGEX.test(candidate)) {
      return {
        valid: true,
        videoId: null,
        listId: candidate,
        isPlaylist: true,
        explicitShort: false,
        startSec: 0,
        normalizedInput: `https://www.youtube.com/embed/videoseries?list=${encodeURIComponent(candidate)}&rel=0`,
        sourceLabel: `Playlist · ${candidate.slice(0, 16)}`,
        error: null
      };
    }

    // 3. Ensure scheme so URL parser succeeds on inputs like "youtube.com/watch?v=..." or "//www.youtube.com/..."
    let urlStr = candidate;
    if (urlStr.startsWith('//')) {
      urlStr = `https:${urlStr}`;
    } else if (!/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(urlStr)) {
      urlStr = `https://${urlStr}`;
    }

    let parsedUrl = null;
    try {
      parsedUrl = new URL(urlStr);
    } catch (_err) {
      return {
        valid: false,
        videoId: null,
        listId: null,
        isPlaylist: false,
        explicitShort: false,
        startSec: 0,
        normalizedInput: raw,
        sourceLabel: '',
        error: 'Format URL YouTube tidak dikenali.'
      };
    }

    const host = parsedUrl.hostname.toLowerCase().replace(/^www\./, '').replace(/^m\./, '');
    const isYouTubeHost =
      host === 'youtube.com' ||
      host === 'youtu.be' ||
      host === 'youtube-nocookie.com' ||
      host === 'music.youtube.com' ||
      host.endsWith('.youtube.com');

    if (!isYouTubeHost) {
      return {
        valid: false,
        videoId: null,
        listId: null,
        isPlaylist: false,
        explicitShort: false,
        startSec: 0,
        normalizedInput: raw,
        sourceLabel: '',
        error: 'Harap gunakan domain YouTube resmi (youtube.com / youtu.be).'
      };
    }

    const pathParts = parsedUrl.pathname.split('/').filter(Boolean);
    let videoId = null;
    let listId = parsedUrl.searchParams.get('list') || null;
    if (listId) {
      listId = listId.trim();
      if (!/^[A-Za-z0-9_-]{4,}$/.test(listId)) listId = null;
    }

    if (host === 'youtu.be') {
      if (pathParts[0] && YT_ID_REGEX.test(pathParts[0])) {
        videoId = pathParts[0];
      }
    } else {
      const vParam = parsedUrl.searchParams.get('v');
      if (vParam && YT_ID_REGEX.test(vParam.trim())) {
        videoId = vParam.trim();
      }
      if (pathParts[0] === 'shorts' && pathParts[1]) {
        explicitShort = true;
        if (YT_ID_REGEX.test(pathParts[1])) {
          videoId = pathParts[1];
        }
      } else if (
        (pathParts[0] === 'embed' || pathParts[0] === 'live' || pathParts[0] === 'v' || pathParts[0] === 'e') &&
        pathParts[1]
      ) {
        if (pathParts[1] !== 'videoseries' && YT_ID_REGEX.test(pathParts[1])) {
          videoId = pathParts[1];
        }
      }
    }

    if (
      parsedUrl.searchParams.get('short') === '1' ||
      parsedUrl.searchParams.get('shorts') === '1' ||
      /\/shorts(\/|$)/i.test(parsedUrl.pathname)
    ) {
      explicitShort = true;
    }

    // Drop non-embeddable private pseudo-playlists (Watch Later / Liked) when a concrete videoId exists
    if (listId && NON_EMBEDDABLE_LISTS.has(listId.toUpperCase()) && videoId) {
      listId = null;
    }

    if (!videoId && !listId) {
      return {
        valid: false,
        videoId: null,
        listId: null,
        isPlaylist: false,
        explicitShort: false,
        startSec: 0,
        normalizedInput: raw,
        sourceLabel: '',
        error: 'ID video atau ID playlist YouTube tidak ditemukan pada link tersebut.'
      };
    }

    const startRaw = parsedUrl.searchParams.get('start') || parsedUrl.searchParams.get('t') || '';
    const startSec = VideotronYouTubeEngine._parseTimestampSeconds(startRaw);
    const isPlaylist = Boolean(listId);

    const normalizedInput =
      isPlaylist && !videoId
        ? `https://www.youtube.com/embed/videoseries?list=${encodeURIComponent(listId)}&rel=0`
        : isPlaylist && videoId
          ? `https://www.youtube.com/embed/${videoId}?list=${encodeURIComponent(listId)}&rel=0`
          : explicitShort
            ? `https://www.youtube.com/shorts/${videoId}`
            : `https://www.youtube.com/embed/${videoId}?rel=0`;

    const sourceLabel =
      isPlaylist && videoId
        ? `Playlist + Video · ${videoId}`
        : isPlaylist
          ? `Playlist · ${listId.slice(0, 16)}`
          : explicitShort
            ? `YouTube Shorts · ${videoId}`
            : `YouTube · ${videoId}`;

    return {
      valid: true,
      videoId,
      listId,
      isPlaylist,
      explicitShort,
      startSec,
      normalizedInput,
      sourceLabel,
      error: null
    };
  }

  /**
   * Asynchronously inspects whether a YouTube `videoId` is actually a vertical Short (height > width),
   * even when pasted as a regular `watch?v=`, `youtu.be/`, `embed/`, or playlist item!
   *
   * Note: Official `youtube.com/oembed` often returns default horizontal `200x113` embed dimensions even for
   * vertical Shorts when queried with `watch?v=`. Therefore, we ALWAYS probe YouTube's Original Aspect Ratio
   * (`OAR`) thumbnail endpoints (`oar2.jpg`, `oardefault.jpg`, `oar1.jpg` on `i.ytimg.com`) in parallel with
   * `oembed` metadata so any vertical Short (`naturalHeight > naturalWidth && naturalHeight >= 200`) is
   * reliably detected regardless of URL format.
   */
  static async probeShortMetadata(videoId, options = {}) {
    if (!videoId || !YT_ID_REGEX.test(videoId)) {
      return { isShort: false, title: '', author: '', portraitThumbUrl: '', method: 'invalid-id' };
    }

    if (!options.bypassCache && VideotronYouTubeEngine._shortMetaCache.has(videoId)) {
      return VideotronYouTubeEngine._shortMetaCache.get(videoId);
    }

    const fetchFn = options.fetchImpl || (typeof fetch === 'function' ? fetch.bind(globalThis) : null);
    const timeoutMs = Number(options.timeoutMs) || 3400;
    const shortsUrl = `https://www.youtube.com/shorts/${videoId}`;
    const watchUrl = `https://www.youtube.com/watch?v=${videoId}`;

    const fetchJsonWithTimeout = async (endpoint) => {
      if (!fetchFn) return null;
      const ctrl = typeof AbortController === 'function' ? new AbortController() : null;
      const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
      try {
        const res = await fetchFn(endpoint, ctrl ? { signal: ctrl.signal } : undefined);
        if (!res || !res.ok) return null;
        return await res.json();
      } catch (_e) {
        return null;
      } finally {
        if (timer) clearTimeout(timer);
      }
    };

    // Task A: Fetch video title, author, and any portrait oEmbed dimensions
    const oembedTask = (async () => {
      const ytShortOembed = await fetchJsonWithTimeout(
        `https://www.youtube.com/oembed?url=${encodeURIComponent(shortsUrl)}&format=json`
      );
      if (ytShortOembed && Number(ytShortOembed.width) > 0 && Number(ytShortOembed.height) > 0) {
        return {
          w: Number(ytShortOembed.width),
          h: Number(ytShortOembed.height),
          title: typeof ytShortOembed.title === 'string' ? ytShortOembed.title : '',
          author: typeof ytShortOembed.author_name === 'string' ? ytShortOembed.author_name : '',
          source: 'youtube-oembed'
        };
      }
      const ytWatchOembed = await fetchJsonWithTimeout(
        `https://www.youtube.com/oembed?url=${encodeURIComponent(watchUrl)}&format=json`
      );
      if (ytWatchOembed && Number(ytWatchOembed.width) > 0 && Number(ytWatchOembed.height) > 0) {
        return {
          w: Number(ytWatchOembed.width),
          h: Number(ytWatchOembed.height),
          title: typeof ytWatchOembed.title === 'string' ? ytWatchOembed.title : '',
          author: typeof ytWatchOembed.author_name === 'string' ? ytWatchOembed.author_name : '',
          source: 'youtube-oembed'
        };
      }
      const noEmbed = await fetchJsonWithTimeout(
        `https://noembed.com/embed?url=${encodeURIComponent(watchUrl)}`
      );
      if (noEmbed && Number(noEmbed.width) > 0 && Number(noEmbed.height) > 0) {
        return {
          w: Number(noEmbed.width),
          h: Number(noEmbed.height),
          title: typeof noEmbed.title === 'string' ? noEmbed.title : '',
          author: typeof noEmbed.author_name === 'string' ? noEmbed.author_name : '',
          source: 'noembed'
        };
      }
      return null;
    })();

    // Task B: CORS-free Original Aspect Ratio (`OAR`) Portrait Image Probe (`oar2.jpg`, `oardefault.jpg`, `oar1.jpg`)
    const oarProbeTask = (async () => {
      if (typeof Image !== 'function') return null;
      const variants = ['oar2.jpg', 'oardefault.jpg', 'oar1.jpg'];
      const probeSingleVariant = (variant) =>
        new Promise((resolve) => {
          let settled = false;
          const img = new Image();
          const url = `https://i.ytimg.com/vi/${videoId}/${variant}`;
          const finish = (val) => {
            if (settled) return;
            settled = true;
            resolve(val);
          };
          const tId = setTimeout(() => finish(null), 2600);
          img.onload = () => {
            clearTimeout(tId);
            const w = img.naturalWidth || 0;
            const h = img.naturalHeight || 0;
            // Missing thumbnails on i.ytimg.com return a 120x90 landscape placeholder; genuine Shorts return portrait (h > w, h >= 200)
            finish({ w, h, url, isPortrait: h > w && h >= 200 });
          };
          img.onerror = () => {
            clearTimeout(tId);
            finish(null);
          };
          img.referrerPolicy = 'no-referrer';
          img.src = url;
        });

      const probes = await Promise.all(variants.map(probeSingleVariant));
      const portraitHit = probes.find((p) => p && p.isPortrait);
      if (portraitHit) return portraitHit;
      return probes.find((p) => p && p.w > 0 && p.h > 0) || null;
    })();

    const [oembedRes, oarRes] = await Promise.all([oembedTask, oarProbeTask]);

    const oembedPortrait = Boolean(oembedRes && oembedRes.h > oembedRes.w);
    const oarPortrait = Boolean(oarRes && oarRes.isPortrait);

    if (oembedRes || oarRes) {
      const isShort = Boolean(oarPortrait || oembedPortrait);
      const result = {
        isShort,
        title: oembedRes ? oembedRes.title : '',
        author: oembedRes ? oembedRes.author : '',
        width: oarPortrait ? oarRes.w : oembedRes ? oembedRes.w : oarRes ? oarRes.w : 0,
        height: oarPortrait ? oarRes.h : oembedRes ? oembedRes.h : oarRes ? oarRes.h : 0,
        portraitThumbUrl: oarPortrait ? oarRes.url : '',
        method: oarPortrait
          ? 'ytimg-oar-portrait'
          : oembedRes
            ? oembedRes.source
            : 'ytimg-oar-landscape'
      };
      VideotronYouTubeEngine._shortMetaCache.set(videoId, result);
      return result;
    }

    return { isShort: false, title: '', author: '', portraitThumbUrl: '', method: 'fallback-16x9' };
  }

  constructor(options = {}) {
    this.spec = VideotronYouTubeEngine.SPEC;
    this.onStateChange = typeof options.onStateChange === 'function' ? options.onStateChange : null;

    // Precompute exact 3D World Transform of the 12.80m x 7.20m Active LED Screen Surface
    const yaw = this.spec.yaw;
    this._fwd = new THREE.Vector3(Math.sin(yaw), 0.0, Math.cos(yaw)).normalize();
    this._right = new THREE.Vector3(Math.cos(yaw), 0.0, -Math.sin(yaw)).normalize();
    this._screenWorldCenter = new THREE.Vector3(
      this.spec.cx + this._fwd.x * this.spec.screenFwdOffset,
      this.spec.screenCenterY,
      this.spec.cz + this._fwd.z * this.spec.screenFwdOffset
    );

    this._anchorObj = new THREE.Object3D();
    this._anchorObj.position.copy(this._screenWorldCenter);
    this._anchorObj.rotation.set(0.0, yaw, 0.0);
    this._anchorObj.updateMatrixWorld(true);
    this._objCssMatrixStr = VideotronYouTubeEngine._formatMatrix3d(this._anchorObj.matrixWorld.elements);

    this._frustum = new THREE.Frustum();
    this._projScreenMat = new THREE.Matrix4();
    this._boundSphere = new THREE.Sphere(this._screenWorldCenter.clone(), 7.8);
    this._toCamTmp = new THREE.Vector3();
    this._cornerTmp = new THREE.Vector3();

    // Playback & Format State
    this.rawInput = options.initialInput || VideotronYouTubeEngine.DEFAULT_EMBED_HTML;
    this.parsed = VideotronYouTubeEngine.parseYouTubeInput(this.rawInput);
    this.formatMode = 'AUTO'; // 'AUTO' | 'WIDE_16_9' | 'SHORT_2LAYER_BLUR'
    this.detectedShort = Boolean(this.parsed.explicitShort);
    this.activeLayout = this.detectedShort ? 'SHORT_2LAYER_BLUR' : 'WIDE_16_9';
    this.activeVideoId = this.parsed.videoId || VideotronYouTubeEngine.DEFAULT_VIDEO_ID;
    this.activeTitle = '';
    this.isPlaying = true;
    this.isMuted = true; // Starts muted so modern browsers allow 100% instant autoplay on load
    this.volume = 75;
    this.interactiveScreenClick = false;
    this.isVisibleInView = false;

    this._loadToken = 0;
    this._frontCurrentTime = 0;
    this._backCurrentTime = 0;
    this._lastSyncAtMs = 0;

    // DOM References
    this.rootEl = null;
    this.viewEl = null;
    this.cameraEl = null;
    this.objectEl = null;
    this.surfaceEl = null;
    this.backBlurLayerEl = null;
    this.backThumbEl = null;
    this.backIframeWrapEl = null;
    this.frontLayerEl = null;
    this.frontIframe = null;
    this.backIframe = null;

    this._lastPerspStr = '';
    this._lastViewOffsetStr = '';
    this._lastCamTransformStr = '';
    this._lastDisplayVisible = null;

    this._boundOnMessage = this._handleYouTubePostMessage.bind(this);
  }

  static _eps(v) {
    return Math.abs(v) < 1e-10 ? 0 : v;
  }

  static _formatMatrix3d(e) {
    const f = VideotronYouTubeEngine._eps;
    return (
      `matrix3d(` +
      `${f(e[0])},${f(e[1])},${f(e[2])},${f(e[3])},` +
      `${f(e[4])},${f(e[5])},${f(e[6])},${f(e[7])},` +
      `${f(e[8])},${f(e[9])},${f(e[10])},${f(e[11])},` +
      `${f(e[12])},${f(e[13])},${f(e[14])},${f(e[15])})`
    );
  }

  /**
   * Builds the YouTube embed URL preserving the user's `rel=0` behavior while enabling autoplay & JS API control.
   */
  _buildEmbedSrc(parsed, { isBackLayer = false, overrideVideoId = null } = {}) {
    const vid = overrideVideoId || parsed.videoId;
    const listId = parsed.listId;

    const basePath =
      listId && !vid
        ? 'https://www.youtube.com/embed/videoseries'
        : `https://www.youtube.com/embed/${encodeURIComponent(vid || VideotronYouTubeEngine.DEFAULT_VIDEO_ID)}`;

    const params = new URLSearchParams();
    params.set('rel', '0');
    params.set('playsinline', '1');
    params.set('autoplay', '1');
    params.set('mute', isBackLayer || this.isMuted ? '1' : '0');
    params.set('enablejsapi', '1');
    params.set('iv_load_policy', '3');
    params.set('modestbranding', '1');

    if (isBackLayer) {
      params.set('controls', '0');
      params.set('disablekb', '1');
      params.set('fs', '0');
    }

    if (listId && !(isBackLayer && vid)) {
      params.set('list', listId);
      params.set('loop', '1');
    }

    if (parsed.startSec > 0 && !overrideVideoId) {
      params.set('start', String(parsed.startSec));
    }

    if (
      typeof window !== 'undefined' &&
      window.location &&
      (window.location.protocol === 'http:' || window.location.protocol === 'https:') &&
      window.location.origin &&
      window.location.origin !== 'null'
    ) {
      params.set('origin', window.location.origin);
    }

    return `${basePath}?${params.toString()}`;
  }

  _createYouTubeIframeElement(src, isBackLayer = false) {
    const doc = typeof document !== 'undefined' ? document : null;
    if (!doc) return null;
    const iframe = doc.createElement('iframe');
    iframe.className = isBackLayer ? 'vt-yt-iframe vt-yt-iframe-back' : 'vt-yt-iframe vt-yt-iframe-front';
    iframe.src = src;
    iframe.setAttribute('allowfullscreen', '');
    iframe.setAttribute('scrolling', 'no');
    iframe.setAttribute(
      'allow',
      'accelerometer *; clipboard-write *; encrypted-media *; gyroscope *; picture-in-picture *; web-share *;'
    );
    iframe.setAttribute('referrerpolicy', 'strict-origin');
    iframe.setAttribute(
      'title',
      isBackLayer
        ? 'Videotron Kampus UI — Lapis Belakang Full-Width Blur'
        : 'Videotron Kampus UI — Pemutar Utama YouTube'
    );
    if (isBackLayer) {
      iframe.setAttribute('tabindex', '-1');
      iframe.setAttribute('aria-hidden', 'true');
    }

    iframe.addEventListener('load', () => {
      this._registerIframeListening(iframe, isBackLayer ? 2 : 1);
      if (!isBackLayer) {
        if (!this.isMuted) {
          this._sendPlayerCommand(iframe, 'unMute');
          this._sendPlayerCommand(iframe, 'setVolume', [this.volume]);
        } else {
          this._sendPlayerCommand(iframe, 'mute');
        }
        if (!this.isPlaying) {
          this._sendPlayerCommand(iframe, 'pauseVideo');
        }
      } else {
        this._sendPlayerCommand(iframe, 'mute');
        if (!this.isPlaying) {
          this._sendPlayerCommand(iframe, 'pauseVideo');
        } else if (this._frontCurrentTime > 0.4) {
          this._sendPlayerCommand(iframe, 'seekTo', [this._frontCurrentTime, true]);
        }
      }
    });

    return iframe;
  }

  _registerIframeListening(iframe, widgetId = 1) {
    if (!iframe || !iframe.contentWindow) return;
    try {
      iframe.contentWindow.postMessage(
        JSON.stringify({ event: 'listening', id: widgetId, channel: 'widget' }),
        '*'
      );
      iframe.contentWindow.postMessage(
        JSON.stringify({ event: 'command', func: 'addEventListener', args: ['onStateChange'] }),
        '*'
      );
    } catch (_e) {
      // Ignore cross-origin postMessage race before iframe finishes handshake
    }
  }

  _sendPlayerCommand(iframe, func, args = []) {
    if (!iframe || !iframe.contentWindow) return;
    try {
      iframe.contentWindow.postMessage(
        JSON.stringify({ event: 'command', func, args }),
        '*'
      );
    } catch (_e) {
      // Safe no-op
    }
  }

  /**
   * Mounts the CSS3D perspective stage behind the WebGL2 canvas inside `#sim-canvas-wrap`.
   */
  mount(containerEl) {
    if (!containerEl || typeof document === 'undefined') return this;

    let root = containerEl.querySelector('#videotron-css3d-root');
    if (!root) {
      root = document.createElement('div');
      root.id = 'videotron-css3d-root';
      containerEl.insertBefore(root, containerEl.firstChild);
    }

    root.innerHTML = '';

    const viewEl = document.createElement('div');
    viewEl.className = 'vt-css3d-view';

    const cameraEl = document.createElement('div');
    cameraEl.className = 'vt-css3d-camera';

    const objectEl = document.createElement('div');
    objectEl.className = 'vt-css3d-object';
    const s = this.spec.metersPerPixel;
    const halfW = this.spec.domWidthPx * 0.5;
    const halfH = this.spec.domHeightPx * 0.5;
    objectEl.style.width = `${this.spec.domWidthPx}px`;
    objectEl.style.height = `${this.spec.domHeightPx}px`;
    objectEl.style.transform =
      `${this._objCssMatrixStr} scale3d(${s}, ${-s}, ${s}) translate3d(${-halfW}px, ${-halfH}px, 0px)`;

    const surfaceEl = document.createElement('div');
    surfaceEl.className = 'vt-screen-surface';

    // Layer 1: Full-Width Blurred Back Layer (activated automatically for vertical Shorts!)
    const backBlurLayerEl = document.createElement('div');
    backBlurLayerEl.className = 'vt-back-blur-layer';

    const backThumbEl = document.createElement('div');
    backThumbEl.className = 'vt-back-thumb';

    const backIframeWrapEl = document.createElement('div');
    backIframeWrapEl.className = 'vt-back-iframe-wrap';

    const backVignetteEl = document.createElement('div');
    backVignetteEl.className = 'vt-back-vignette';

    backBlurLayerEl.appendChild(backThumbEl);
    backBlurLayerEl.appendChild(backIframeWrapEl);
    backBlurLayerEl.appendChild(backVignetteEl);

    // Layer 2: Foreground Primary Video Layer (1280x720 in 16:9 mode, or centered sharp 405x720 in 9:16 Short mode)
    const frontLayerEl = document.createElement('div');
    frontLayerEl.className = 'vt-front-layer';

    surfaceEl.appendChild(backBlurLayerEl);
    surfaceEl.appendChild(frontLayerEl);
    objectEl.appendChild(surfaceEl);
    cameraEl.appendChild(objectEl);
    viewEl.appendChild(cameraEl);
    root.appendChild(viewEl);

    this.rootEl = root;
    this.viewEl = viewEl;
    this.cameraEl = cameraEl;
    this.objectEl = objectEl;
    this.surfaceEl = surfaceEl;
    this.backBlurLayerEl = backBlurLayerEl;
    this.backThumbEl = backThumbEl;
    this.backIframeWrapEl = backIframeWrapEl;
    this.frontLayerEl = frontLayerEl;

    if (typeof window !== 'undefined') {
      window.removeEventListener('message', this._boundOnMessage);
      window.addEventListener('message', this._boundOnMessage);
    }

    this.setInteractiveScreenClick(this.interactiveScreenClick);
    this.loadSource(this.rawInput);
    return this;
  }

  dispose() {
    if (typeof window !== 'undefined') {
      window.removeEventListener('message', this._boundOnMessage);
    }
    if (this.rootEl && this.rootEl.parentNode) {
      this.rootEl.parentNode.removeChild(this.rootEl);
    }
  }

  _resolveTargetLayout() {
    if (this.formatMode === 'SHORT_2LAYER_BLUR') return 'SHORT_2LAYER_BLUR';
    if (this.formatMode === 'WIDE_16_9') return 'WIDE_16_9';
    return this.detectedShort ? 'SHORT_2LAYER_BLUR' : 'WIDE_16_9';
  }

  _updateBackThumbGraphic(videoId) {
    if (!this.backThumbEl) return;
    if (videoId && YT_ID_REGEX.test(videoId)) {
      const safeId = encodeURIComponent(videoId);
      const cached = VideotronYouTubeEngine._shortMetaCache.get(videoId);
      const primaryOar = cached && cached.portraitThumbUrl ? cached.portraitThumbUrl : `https://i.ytimg.com/vi/${safeId}/oar2.jpg`;
      this.backThumbEl.style.backgroundImage =
        `url("${primaryOar}"), url("https://i.ytimg.com/vi/${safeId}/oardefault.jpg"), url("https://i.ytimg.com/vi/${safeId}/hqdefault.jpg")`;
    } else {
      this.backThumbEl.style.backgroundImage = 'none';
    }
  }

  /**
   * Applies either `WIDE_16_9` or `SHORT_2LAYER_BLUR` layout in-place WITHOUT reloading the foreground iframe!
   */
  _syncActiveLayoutDOM(forceBackRebuild = false) {
    const nextLayout = this._resolveTargetLayout();
    const isShortLayout = nextLayout === 'SHORT_2LAYER_BLUR';
    this.activeLayout = nextLayout;

    if (!this.surfaceEl || !this.frontLayerEl || !this.backBlurLayerEl || !this.backIframeWrapEl) {
      this._emitState();
      return;
    }

    this.surfaceEl.classList.toggle('is-short-2layer', isShortLayout);
    this.frontLayerEl.classList.toggle('is-short', isShortLayout);
    this.backBlurLayerEl.classList.toggle('active', isShortLayout);

    if (isShortLayout) {
      this._updateBackThumbGraphic(this.activeVideoId);
      const desiredBackSrc = this._buildEmbedSrc(this.parsed, {
        isBackLayer: true,
        overrideVideoId: this.activeVideoId || this.parsed.videoId
      });
      if (forceBackRebuild || !this.backIframe || this.backIframe.dataset.baseVid !== (this.activeVideoId || 'list')) {
        this.backIframeWrapEl.innerHTML = '';
        this.backIframe = this._createYouTubeIframeElement(desiredBackSrc, true);
        if (this.backIframe) {
          this.backIframe.dataset.baseVid = this.activeVideoId || 'list';
          this.backIframeWrapEl.appendChild(this.backIframe);
        }
      }
    } else if (this.backIframe) {
      // Remove background blur iframe when in standard 16:9 mode so zero background CPU/GPU decoder runs!
      this.backIframeWrapEl.innerHTML = '';
      this.backIframe = null;
    }

    this._emitState();
  }

  /**
   * Loads any YouTube single link, playlist link, Shorts link, raw ID, or `<iframe>` HTML snippet.
   */
  async loadSource(rawInput) {
    const parsed = VideotronYouTubeEngine.parseYouTubeInput(rawInput);
    if (!parsed.valid) {
      this._emitState(parsed.error);
      return { ok: false, error: parsed.error, state: this.getState() };
    }

    const token = ++this._loadToken;
    this.rawInput = String(rawInput).trim();
    this.parsed = parsed;
    this.detectedShort = Boolean(parsed.explicitShort);
    this.activeVideoId = parsed.videoId || null;
    this.activeTitle = '';
    this._frontCurrentTime = parsed.startSec || 0;
    this._backCurrentTime = parsed.startSec || 0;
    this.isPlaying = true;

    if (this.frontLayerEl) {
      this.frontLayerEl.innerHTML = '';
      const frontSrc = this._buildEmbedSrc(parsed, { isBackLayer: false });
      this.frontIframe = this._createYouTubeIframeElement(frontSrc, false);
      if (this.frontIframe) {
        this.frontLayerEl.appendChild(this.frontIframe);
      }
    }

    this._syncActiveLayoutDOM(true);

    // Asynchronously probe whether a non-/shorts/ URL actually contains a vertical Short video (width < height)!
    if (parsed.videoId) {
      const meta = await VideotronYouTubeEngine.probeShortMetadata(parsed.videoId);
      if (token === this._loadToken) {
        if (meta.title) this.activeTitle = meta.title;
        const nextDetectedShort = Boolean(parsed.explicitShort || meta.isShort);
        if (nextDetectedShort !== this.detectedShort) {
          this.detectedShort = nextDetectedShort;
          this._syncActiveLayoutDOM(false);
        } else {
          this._emitState();
        }
      }
    }

    return { ok: true, error: null, state: this.getState() };
  }

  resetToDefault() {
    this.formatMode = 'AUTO';
    return this.loadSource(VideotronYouTubeEngine.DEFAULT_EMBED_HTML);
  }

  setFormatMode(mode) {
    if (mode !== 'AUTO' && mode !== 'WIDE_16_9' && mode !== 'SHORT_2LAYER_BLUR') return;
    this.formatMode = mode;
    this._syncActiveLayoutDOM(false);
  }

  togglePlayPause() {
    this.isPlaying = !this.isPlaying;
    const cmd = this.isPlaying ? 'playVideo' : 'pauseVideo';
    this._sendPlayerCommand(this.frontIframe, cmd);
    if (this.backIframe) {
      this._sendPlayerCommand(this.backIframe, cmd);
    }
    this._emitState();
    return this.isPlaying;
  }

  setMuted(muted) {
    this.isMuted = Boolean(muted);
    if (this.isMuted) {
      this._sendPlayerCommand(this.frontIframe, 'mute');
    } else {
      this._sendPlayerCommand(this.frontIframe, 'unMute');
      this._sendPlayerCommand(this.frontIframe, 'setVolume', [this.volume]);
    }
    if (this.backIframe) {
      this._sendPlayerCommand(this.backIframe, 'mute');
    }
    this._emitState();
    return this.isMuted;
  }

  toggleMute() {
    return this.setMuted(!this.isMuted);
  }

  setVolume(volPct) {
    this.volume = Math.max(0, Math.min(100, Math.round(Number(volPct) || 0)));
    if (this.volume === 0) {
      this.setMuted(true);
      return;
    }
    if (this.isMuted) {
      this.isMuted = false;
      this._sendPlayerCommand(this.frontIframe, 'unMute');
    }
    this._sendPlayerCommand(this.frontIframe, 'setVolume', [this.volume]);
    this._emitState();
  }

  nextTrack() {
    this._sendPlayerCommand(this.frontIframe, 'nextVideo');
    if (this.backIframe && this.parsed.isPlaylist && !this.activeVideoId) {
      this._sendPlayerCommand(this.backIframe, 'nextVideo');
    }
  }

  previousTrack() {
    this._sendPlayerCommand(this.frontIframe, 'previousVideo');
    if (this.backIframe && this.parsed.isPlaylist && !this.activeVideoId) {
      this._sendPlayerCommand(this.backIframe, 'previousVideo');
    }
  }

  setInteractiveScreenClick(enabled, webglCanvas = null) {
    this.interactiveScreenClick = Boolean(enabled);
    if (this.rootEl) {
      this.rootEl.classList.toggle('is-interactive', this.interactiveScreenClick);
    }
    const canvas =
      webglCanvas ||
      (this.rootEl && this.rootEl.parentElement ? this.rootEl.parentElement.querySelector('canvas') : null);
    if (canvas) {
      canvas.style.pointerEvents = this.interactiveScreenClick ? 'none' : 'auto';
    }
    this._emitState();
    return this.interactiveScreenClick;
  }

  _handleYouTubePostMessage(event) {
    if (!event || typeof event.data !== 'string') return;
    if (event.origin && !/youtube(-nocookie)?\.com$/i.test(event.origin.replace(/^https?:\/\//i, ''))) {
      return;
    }

    const fromFront = Boolean(this.frontIframe && event.source === this.frontIframe.contentWindow);
    const fromBack = Boolean(this.backIframe && event.source === this.backIframe.contentWindow);
    if (!fromFront && !fromBack) return;

    let payload = null;
    try {
      payload = JSON.parse(event.data);
    } catch (_e) {
      return;
    }
    if (!payload || typeof payload !== 'object') return;

    const info = payload.info;
    if (!info || typeof info !== 'object') return;

    if (fromBack) {
      if (typeof info.currentTime === 'number' && Number.isFinite(info.currentTime)) {
        this._backCurrentTime = info.currentTime;
      }
      return;
    }

    // --- Foreground Primary Player Telemetry ---
    if (typeof info.currentTime === 'number' && Number.isFinite(info.currentTime)) {
      this._frontCurrentTime = info.currentTime;
      if (this.backIframe && this.isPlaying) {
        const nowMs = Date.now();
        if (Math.abs(this._frontCurrentTime - this._backCurrentTime) > 0.65 && nowMs - this._lastSyncAtMs > 1400) {
          this._lastSyncAtMs = nowMs;
          this._sendPlayerCommand(this.backIframe, 'seekTo', [this._frontCurrentTime, true]);
        }
      }
    }

    if (typeof info.playerState === 'number') {
      if (info.playerState === 1) {
        this.isPlaying = true;
        if (this.backIframe) this._sendPlayerCommand(this.backIframe, 'playVideo');
      } else if (info.playerState === 2) {
        this.isPlaying = false;
        if (this.backIframe) this._sendPlayerCommand(this.backIframe, 'pauseVideo');
      } else if (info.playerState === 0 && !this.parsed.isPlaylist) {
        // Continuous 24/7 loop for single videos without showing YouTube's fake 1-item playlist chrome
        this._sendPlayerCommand(this.frontIframe, 'seekTo', [0, true]);
        this._sendPlayerCommand(this.frontIframe, 'playVideo');
        if (this.backIframe) {
          this._sendPlayerCommand(this.backIframe, 'seekTo', [0, true]);
          this._sendPlayerCommand(this.backIframe, 'playVideo');
        }
      }
    }

    // Detect dynamic track changes inside Playlists (and auto-probe whether the new playlist video is a Short!)
    if (info.videoData && typeof info.videoData === 'object') {
      const vid = info.videoData.video_id;
      const title = info.videoData.title;
      let changed = false;
      if (typeof title === 'string' && title.trim() && title.trim() !== this.activeTitle) {
        this.activeTitle = title.trim();
        changed = true;
      }
      if (typeof vid === 'string' && YT_ID_REGEX.test(vid) && vid !== this.activeVideoId) {
        this.activeVideoId = vid;
        changed = true;
        const token = this._loadToken;
        VideotronYouTubeEngine.probeShortMetadata(vid).then((meta) => {
          if (token !== this._loadToken || this.activeVideoId !== vid) return;
          if (meta.title && !this.activeTitle) this.activeTitle = meta.title;
          const isTrackShort = Boolean(this.parsed.explicitShort || meta.isShort);
          const layoutChanged = isTrackShort !== this.detectedShort;
          this.detectedShort = isTrackShort;
          this._syncActiveLayoutDOM(layoutChanged || this.activeLayout === 'SHORT_2LAYER_BLUR');
        });
      }
      if (changed) {
        this._emitState();
      }
    }
  }

  getState() {
    const layout = this.activeLayout;
    let badgeLabelId = 'Single 16:9';
    if (layout === 'SHORT_2LAYER_BLUR' && this.parsed.isPlaylist) {
      badgeLabelId = 'Playlist Short · 2-Lapis Blur';
    } else if (layout === 'SHORT_2LAYER_BLUR') {
      badgeLabelId = 'Short 9:16 · 2-Lapis Blur';
    } else if (this.parsed.isPlaylist) {
      badgeLabelId = 'Playlist 16:9';
    }

    return {
      rawInput: this.rawInput,
      normalizedUrl: this.parsed.normalizedInput,
      videoId: this.activeVideoId || this.parsed.videoId,
      listId: this.parsed.listId,
      isPlaylist: this.parsed.isPlaylist,
      detectedShort: this.detectedShort,
      formatMode: this.formatMode,
      activeLayout: layout,
      badgeLabelId,
      title: this.activeTitle || this.parsed.sourceLabel || 'Videotron Kampus UI',
      isPlaying: this.isPlaying,
      isMuted: this.isMuted,
      volume: this.volume,
      interactiveScreenClick: this.interactiveScreenClick,
      isVisibleInView: this.isVisibleInView
    };
  }

  _emitState(errorMessage = null) {
    if (this.onStateChange) {
      this.onStateChange(this.getState(), errorMessage);
    }
  }

  /**
   * Projects the 4 world-space active screen corners (TL, TR, BR, BL) into 2D viewport pixel coordinates.
   * Used for automated sub-pixel CSS3D <-> WebGL alignment audits.
   */
  projectCornersToScreen(camera, viewportW, viewportH) {
    const hw = this.spec.screenW * 0.5;
    const hh = this.spec.screenH * 0.5;
    const c = this._screenWorldCenter;
    const r = this._right;
    const localOffsets = [
      [-hw, +hh], // TL
      [+hw, +hh], // TR
      [+hw, -hh], // BR
      [-hw, -hh]  // BL
    ];
    return localOffsets.map(([lx, ly]) => {
      this._cornerTmp.set(c.x + r.x * lx, c.y + ly, c.z + r.z * lx);
      this._cornerTmp.project(camera);
      return {
        x: (this._cornerTmp.x * 0.5 + 0.5) * viewportW,
        y: (-this._cornerTmp.y * 0.5 + 0.5) * viewportH,
        ndcZ: this._cornerTmp.z
      };
    });
  }

  /**
   * Synchronizes the CSS3D Videotron transform & WebGL punch-through uniform every animation frame (0 GC!).
   */
  update(camera, viewportW, viewportH, sharedUniforms = null) {
    if (!camera) return;
    const w = Math.max(1, viewportW || 1);
    const h = Math.max(1, viewportH || 1);

    camera.updateMatrixWorld();

    // 1. Visibility & Optical Back-Face / Subterranean / Studio-Inspect Gating
    this._toCamTmp.subVectors(camera.position, this._screenWorldCenter);
    const facingDot = this._toCamTmp.dot(this._fwd);
    const inSubseaTunnel = camera.position.y < -2.2;
    const inspectMode =
      sharedUniforms && sharedUniforms.uZebraInspectMode
        ? Number(sharedUniforms.uZebraInspectMode.value) || 0.0
        : 0.0;

    let inFrustum = false;
    if (!inSubseaTunnel && facingDot > 0.05 && inspectMode < 0.5) {
      this._projScreenMat.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      this._frustum.setFromProjectionMatrix(this._projScreenMat);
      inFrustum = this._frustum.intersectsSphere(this._boundSphere);
    }

    this.isVisibleInView = inFrustum;

    if (sharedUniforms) {
      if (sharedUniforms.uVideotronPortalAlpha) {
        sharedUniforms.uVideotronPortalAlpha.value = inFrustum && this.rootEl ? 1.0 : 0.0;
      }
      if (sharedUniforms.uVideotronShortMode) {
        sharedUniforms.uVideotronShortMode.value = this.activeLayout === 'SHORT_2LAYER_BLUR' ? 1.0 : 0.0;
      }
    }

    if (!this.rootEl || !this.viewEl || !this.cameraEl) return;

    if (inFrustum !== this._lastDisplayVisible) {
      this._lastDisplayVisible = inFrustum;
      this.rootEl.style.visibility = inFrustum ? 'visible' : 'hidden';
    }
    if (!inFrustum) return;

    // 2. Exact CSS3D Perspective Focal Length & Non-Occluding Drawer ViewOffset Compensation
    const fovPx = 0.5 * camera.projectionMatrix.elements[5] * h;
    const perspStr = `${fovPx.toFixed(4)}px`;
    if (perspStr !== this._lastPerspStr) {
      this._lastPerspStr = perspStr;
      this.viewEl.style.perspective = perspStr;
    }

    let ox = 0;
    let oy = 0;
    if (camera.view && camera.view.enabled) {
      const scaleX = w / Math.max(1, camera.view.width || w);
      const scaleY = h / Math.max(1, camera.view.height || h);
      ox = (camera.view.offsetX ?? camera.view.x ?? 0) * scaleX;
      oy = (camera.view.offsetY ?? camera.view.y ?? 0) * scaleY;
    }
    const viewOffsetStr = `translate3d(${(-ox).toFixed(3)}px, ${(-oy).toFixed(3)}px, 0px)`;
    if (viewOffsetStr !== this._lastViewOffsetStr) {
      this._lastViewOffsetStr = viewOffsetStr;
      this.viewEl.style.transform = viewOffsetStr;
    }
    this.viewEl.style.width = `${w}px`;
    this.viewEl.style.height = `${h}px`;

    // 3. Exact Transform-Origin (0, 0) Camera Matrix Chain:
    //    translate3d(W/2, H/2, P) * scale3d(1, -1, 1) * camera.matrixWorldInverse
    const camMatCss = VideotronYouTubeEngine._formatMatrix3d(camera.matrixWorldInverse.elements);
    const camTransformStr =
      `translate3d(${(w * 0.5).toFixed(3)}px, ${(h * 0.5).toFixed(3)}px, ${fovPx.toFixed(4)}px) ` +
      `scale3d(1, -1, 1) ${camMatCss}`;
    if (camTransformStr !== this._lastCamTransformStr) {
      this._lastCamTransformStr = camTransformStr;
      this.cameraEl.style.transform = camTransformStr;
    }
  }
}
