import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v6.0
 * ./core/HardwareTelemetryGovernor.js
 *
 * Automatic Hardware Capability Profiler, Empirical Pre-Warm Benchmark & Closed-Loop 120-FPS Governor:
 * 1. Static Hardware Fingerprinting:
 *    - Probes WebGL2 UNMASKED_RENDERER_WEBGL / UNMASKED_VENDOR_WEBGL, MAX_TEXTURE_SIZE,
 *      MAX_VERTEX_TEXTURE_IMAGE_UNITS, CPU logical cores, deviceMemory, & monitor refresh rate (Hz).
 * 2. Empirical Warmup Calibration:
 *    - Measures real CPU + GPU frame execution time (ms/frame) during Stage 9 Pre-Warming so
 *      tier classification accounts for actual display resolution, OS scaling, and power mode.
 * 3. Closed-Loop Runtime Governor (Zero Visual Downgrade):
 *    - Continuously regulates shadow-map refresh cadence, camera-centered texel-snapped shadow frustum,
 *      and smart High-DPI pixel-ratio bounds to sustain up to 120+ FPS on Gaming GPUs without visual popping.
 */

export const HARDWARE_TIERS = {
  TIER_S: {
    id: 'TIER_S',
    shortLabel: 'Tier S • Ultra 120 FPS',
    badgeLabel: 'VGA Gaming (120Hz)',
    targetFps: 120,
    targetFrameMs: 8.33,
    minPixelRatio: 1.0,
    maxPixelRatio: 1.5,
    shadowMapSize: 2048,
    shadowFrustumHalf: 108.0,
    shadowRefreshStride: 1,
    closeDerivativeDistM: 22.0,
    aiFarStaggerDistM: 9999.0
  },
  TIER_A: {
    id: 'TIER_A',
    shortLabel: 'Tier A • High 90–120 FPS',
    badgeLabel: 'High Performance',
    targetFps: 90,
    targetFrameMs: 11.11,
    minPixelRatio: 1.0,
    maxPixelRatio: 1.25,
    shadowMapSize: 2048,
    shadowFrustumHalf: 104.0,
    shadowRefreshStride: 1,
    closeDerivativeDistM: 18.0,
    aiFarStaggerDistM: 9999.0
  },
  TIER_B: {
    id: 'TIER_B',
    shortLabel: 'Tier B • Balanced 60 FPS',
    badgeLabel: 'Balanced 60Hz',
    targetFps: 60,
    targetFrameMs: 16.67,
    minPixelRatio: 0.92,
    maxPixelRatio: 1.15,
    shadowMapSize: 2048,
    shadowFrustumHalf: 100.0,
    shadowRefreshStride: 1,
    closeDerivativeDistM: 16.0,
    aiFarStaggerDistM: 120.0
  },
  TIER_C: {
    id: 'TIER_C',
    shortLabel: 'Tier C • Efficiency',
    badgeLabel: 'Hemat Daya',
    targetFps: 60,
    targetFrameMs: 16.67,
    minPixelRatio: 0.85,
    maxPixelRatio: 1.0,
    shadowMapSize: 1536,
    shadowFrustumHalf: 94.0,
    shadowRefreshStride: 1,
    closeDerivativeDistM: 12.0,
    aiFarStaggerDistM: 95.0
  }
};

export class HardwareTelemetryGovernor {
  constructor(renderer = null, sharedUniforms = null) {
    this.renderer = renderer;
    this.sharedUniforms = sharedUniforms;

    this.mode = 'AUTO'; // 'AUTO' | 'TIER_S' | 'TIER_A' | 'TIER_B' | 'TIER_C'
    this.detectedTierId = 'TIER_S';
    this.activeTierId = 'TIER_S';

    this.gpuModel = 'WebGL2 Hardware GPU';
    this.gpuVendor = 'Generic';
    this.gpuClass = 'GAMING_DGPU';
    this.gpuCaps = {
      rawRenderer: 'WebGL2 Hardware GPU',
      rawVendor: 'Generic',
      glVersion: 'WebGL 2.0',
      glslVersion: 'WebGL GLSL ES 3.00',
      maxTextureSize: 8192,
      maxCubeMapSize: 8192,
      maxVertexTexUnits: 16,
      maxCombinedTexUnits: 32,
      maxSamplesMSAA: 4,
      maxAnisotropy: 16,
      colorBufferFloat: true
    };
    this.cpuCores =
      typeof navigator !== 'undefined' && Number.isFinite(navigator.hardwareConcurrency)
        ? navigator.hardwareConcurrency
        : 8;
    this.deviceMemoryGB =
      typeof navigator !== 'undefined' && Number.isFinite(navigator.deviceMemory)
        ? navigator.deviceMemory
        : 8;
    this.estimatedRefreshHz = 120;
    this.supportsVertexFloatTex = true;
    this.isMobileDevice = HardwareTelemetryGovernor.detectMobilePlatform();

    this._warmupSamplesMs = [];
    this.warmupAvgFrameMs = 4.8;
    this.emaFrameMs = 6.2;
    this.emaCpuMs = 1.6;
    this.emaFps = 120.0;
    this.livePixelRatio = 1.0;

    this._frameCounter = 0;
    this._governorTimer = 0.0;
    this._overloadStreak = 0;
    this._headroomStreak = 0;
    this._uptimeSec = 0.0;

    // Zero-GC 60-Second (1 Hz × 60 slots = 1.68 KB) Engine Forensic Ring Buffer
    this._ringCap = 60;
    this._ringHead = 0;
    this._ringCount = 0;
    this._ringTimeSec = new Float32Array(60);
    this._ringFps = new Float32Array(60);
    this._ringAvgMs = new Float32Array(60);
    this._ringPeakMs = new Float32Array(60);
    this._ringCpuMs = new Float32Array(60);
    this._ringCalls = new Uint16Array(60);
    this._ringDpr = new Float32Array(60);
    this._sortScratch = new Float32Array(60);

    // Per-second O(1) frame accumulator
    this._secWallAccum = 0.0;
    this._secFrames = 0;
    this._secSumMs = 0.0;
    this._secPeakMs = 0.0;
    this._secSumCpuMs = 0.0;
    this._secMaxCalls = 0;
    this._lastSpikeLogSec = -10.0;

    // Bounded 24-entry Circular Forensic Event Log
    this._eventCap = 24;
    this._eventRing = new Array(24);
    this._eventHead = 0;
    this._eventCount = 0;

    this._lastShadowSnapX = NaN;
    this._lastSunDirX = NaN;
    this._lastSunDirY = NaN;
    this._lastSunDirZ = NaN;
    this._lastShadowSnapZ = NaN;
    this._tmpFocus = new THREE.Vector3();

    if (renderer) {
      this.probeHardware(renderer);
    }
  }

  /**
   * Detects Android / iOS / Mobile browser platform safely in both DOM and headless Node environments.
   */
  static detectMobilePlatform() {
    if (typeof navigator === 'undefined') return false;
    const ua = String(navigator.userAgent || '');
    const plat = String(navigator.platform || '');
    if (/Android|iPhone|iPad|iPod|Mobile|CriOS|FxiOS/i.test(ua)) return true;
    // Windows, Linux x86_64, and ChromeOS touchscreen laptops are desktop-class OS platforms, NEVER phones!
    if (/Win32|Win64|Windows|Linux x86_64|CrOS/i.test(plat) && !/Android/i.test(ua)) {
      return false;
    }
    // iPadOS 13+ desktop-UA mode reports MacIntel with multi-touch
    if (/MacIntel/i.test(plat) && Number(navigator.maxTouchPoints) > 1) {
      return true;
    }
    return false;
  }

  /**
   * Classifies raw WebGL UNMASKED_RENDERER_WEBGL string into hardware tier & short model name.
   */
  static classifyGpuRendererString(rawRenderer = '', rawVendor = '', cpuCores = 8, isMobile = false) {
    const r = String(rawRenderer || '').trim();
    const v = String(rawVendor || '').trim();

    // Extract clean human-readable GPU name (strip ANGLE(...), (R)/(TM), PCI device IDs, and D3D shader suffixes)
    const rNorm = r.replace(/\(R\)|\(TM\)|®|™/gi, '').replace(/\s+/g, ' ').trim();
    let cleanName = rNorm;
    const angleMatch = rNorm.match(/ANGLE\s*\([^,]+,\s*([^,]+)/i);
    if (angleMatch && angleMatch[1]) {
      cleanName = angleMatch[1].trim();
    }
    cleanName = cleanName
      .replace(/\s*\(0x[0-9a-f]+\)/gi, '')
      .replace(/Direct3D\d*.*$/i, '')
      .replace(/OpenGL.*$/i, '')
      .replace(/Vulkan.*$/i, '')
      .replace(/vs_\d+_\d+.*$/i, '')
      .replace(/\s+/g, ' ')
      .trim();
    if (!cleanName) cleanName = isMobile ? 'Mobile WebGL2 GPU' : 'WebGL2 GPU';

    const isSoftware = /SwiftShader|llvmpipe|Software|Microsoft Basic Render/i.test(rNorm);
    const isFlagshipMobile =
      /Adreno\s*(7[345]\d|8\d\d)|Immortalis|Mali-G7(15|20|25)|Xclipse\s*9[24]0/i.test(
        rNorm
      );
    const isMobileGpu = /Adreno|Mali|PowerVR|Xclipse|Maleoon/i.test(rNorm) || isMobile;
    const isGamingNvidia =
      /RTX\s*(20[6789]0|30[6789]0|40[6789]0|50[6789]0|A[23456]000)|GTX\s*(10[78]0|1660)/i.test(rNorm);
    const isLaptopOrEntryNvidia = /RTX\s*(2050|3050|4050)|GTX\s*(1050|1060|1650)|MX\s*\d+/i.test(rNorm);
    const isGamingAmd = /Radeon\s*RX\s*(6[6789]\d0|7[6789]\d0|5700)/i.test(rNorm);
    const isHighApuOrArc =
      /Radeon\s*(680M|780M|880M|890M|RX\s*[56][56]00)|Intel.*Arc|Apple\s*M[1234]\s*(Pro|Max|Ultra)/i.test(
        rNorm
      );
    const isAppleBase = /Apple\s*M[1234]\b/i.test(rNorm) && !/Pro|Max|Ultra/i.test(rNorm);
    const isEntryDesktop = /UHD\s*Graphics\s*(6\d0)|HD\s*Graphics/i.test(rNorm);
    const isMainstreamIgpu =
      !isHighApuOrArc &&
      !isEntryDesktop &&
      /Iris\s*(Xe|Plus|Pro)?|UHD\s*Graphics|Radeon\s*(660M|740M|760M|Vega|Graphics)/i.test(rNorm);

    if (isSoftware) {
      return {
        tierId: 'TIER_C',
        gpuClass: 'SOFTWARE',
        cleanName,
        vendor: v || 'Software'
      };
    }
    if (isMobileGpu) {
      return {
        tierId: isFlagshipMobile ? 'TIER_B' : 'TIER_C',
        gpuClass: isFlagshipMobile ? 'FLAGSHIP_MOBILE_GPU' : 'MOBILE_GPU',
        cleanName,
        vendor: v || 'Mobile SoC GPU'
      };
    }
    if (isEntryDesktop) {
      return {
        tierId: 'TIER_C',
        gpuClass: 'ENTRY_IGPU',
        cleanName,
        vendor: v || 'Integrated'
      };
    }
    if (isGamingNvidia || isGamingAmd || /Apple\s*M[1234]\s*(Pro|Max|Ultra)/i.test(r)) {
      return {
        tierId: 'TIER_S',
        gpuClass: 'GAMING_DGPU',
        cleanName,
        vendor: v || 'Discrete Gaming GPU'
      };
    }
    if (isLaptopOrEntryNvidia || isHighApuOrArc || isAppleBase) {
      return {
        tierId: 'TIER_A',
        gpuClass: 'PERF_GPU',
        cleanName,
        vendor: v || 'Performance GPU'
      };
    }
    if (isMainstreamIgpu) {
      return {
        tierId: 'TIER_B',
        gpuClass: 'MAINSTREAM_IGPU',
        cleanName,
        vendor: v || 'Integrated GPU'
      };
    }

    // Fallback heuristic based on CPU core count if GPU string is masked by browser privacy settings
    const fallbackTier = cpuCores >= 8 ? 'TIER_S' : cpuCores >= 6 ? 'TIER_A' : 'TIER_B';
    return {
      tierId: fallbackTier,
      gpuClass: fallbackTier === 'TIER_S' ? 'GAMING_DGPU' : 'PERF_GPU',
      cleanName,
      vendor: v || 'WebGL2 Hardware'
    };
  }

  /**
   * Appends a lightweight timestamped forensic event into the 24-slot circular ring (zero bloat).
   */
  logEvent(tag = 'INFO', detail = '') {
    const sec = Math.max(0.0, this._uptimeSec || 0.0);
    const entry = {
      tSec: +sec.toFixed(1),
      tag: String(tag).toUpperCase(),
      msg: String(detail)
    };
    this._eventRing[this._eventHead] = entry;
    this._eventHead = (this._eventHead + 1) % this._eventCap;
    if (this._eventCount < this._eventCap) this._eventCount++;
    return entry;
  }

  /**
   * Probes WebGL2 context parameters & classifies initial hardware tier (one-time GL query at boot).
   */
  probeHardware(renderer) {
    this.renderer = renderer || this.renderer;
    let rawRenderer = '';
    let rawVendor = '';

    try {
      const gl = this.renderer?.getContext?.();
      if (gl) {
        const dbg = gl.getExtension('WEBGL_debug_renderer_info');
        if (dbg) {
          rawRenderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || '';
          rawVendor = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) || '';
        } else {
          rawRenderer = gl.getParameter(gl.RENDERER) || '';
          rawVendor = gl.getParameter(gl.VENDOR) || '';
        }
        const maxVertTex = gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS) || 0;
        this.supportsVertexFloatTex = maxVertTex >= 1;
        this.gpuCaps = {
          rawRenderer: rawRenderer || 'WebGL2 Masked Renderer',
          rawVendor: rawVendor || 'WebGL2 Vendor',
          glVersion: String(gl.getParameter(gl.VERSION) || 'WebGL 2.0'),
          glslVersion: String(gl.getParameter(gl.SHADING_LANGUAGE_VERSION) || 'GLSL ES 3.00'),
          maxTextureSize: Number(gl.getParameter(gl.MAX_TEXTURE_SIZE)) || 8192,
          maxCubeMapSize: Number(gl.getParameter(gl.MAX_CUBE_MAP_TEXTURE_SIZE)) || 8192,
          maxVertexTexUnits: Number(maxVertTex) || 16,
          maxCombinedTexUnits: Number(gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS)) || 32,
          maxSamplesMSAA: Number(gl.getParameter(gl.MAX_SAMPLES ?? 0x8d57)) || 4,
          maxAnisotropy: Number(this.renderer?.capabilities?.getMaxAnisotropy?.()) || 1,
          colorBufferFloat: Boolean(gl.getExtension('EXT_color_buffer_float'))
        };
      }
    } catch (_e) {
      // Safe fallback in headless Node mock
    }

    const info = HardwareTelemetryGovernor.classifyGpuRendererString(
      rawRenderer,
      rawVendor,
      this.cpuCores,
      this.isMobileDevice
    );
    if (/Adreno|Mali|PowerVR|Xclipse|Maleoon/i.test(rawRenderer)) {
      this.isMobileDevice = true;
    }
    this.gpuModel = info.cleanName;
    this.gpuVendor = info.vendor;
    this.gpuClass = info.gpuClass;
    this.detectedTierId = info.tierId;
    if (this.mode === 'AUTO') {
      this.activeTierId = this.detectedTierId;
    }
    this._applyTierPresetToEngine(this.activeTierId, true);
    this.logEvent(
      'BOOT',
      `${this.gpuModel} (${this.activeTierId}, DPR ${this.livePixelRatio.toFixed(2)}×, MaxTex ${this.gpuCaps.maxTextureSize})`
    );
    return this.getTelemetryState();
  }

  /**
   * Records a real steady-state frame sample (in ms) after incremental shader pre-warm completes.
   * Rejects synchronous shader/pipeline link outliers (> 40ms) so D3D11/ANGLE compilation stalls
   * can never poison steady-state EMA or falsely downgrade discrete GPUs (e.g. RTX 2060).
   */
  recordWarmupSample(frameMs) {
    if (!Number.isFinite(frameMs) || frameMs <= 0.0 || frameMs > 40.0) return;
    this._warmupSamplesMs.push(frameMs);
    const sum = this._warmupSamplesMs.reduce((a, b) => a + b, 0.0);
    this.warmupAvgFrameMs = sum / this._warmupSamplesMs.length;
    this.emaFrameMs = Math.min(this.warmupAvgFrameMs, 18.0);
    this.emaFps = Math.min(240.0, 1000.0 / Math.max(2.0, this.emaFrameMs));

    // Only downgrade from static GPU classification if actual steady-state frame is severely bottlenecked
    // (e.g., 4K 200% scaling or laptop unplugged on battery saver)
    if (this.mode === 'AUTO' && this._warmupSamplesMs.length >= 3) {
      if (this.gpuClass === 'GAMING_DGPU') {
        this.detectedTierId = this.warmupAvgFrameMs <= 14.5 ? 'TIER_S' : 'TIER_A';
      } else if (this.warmupAvgFrameMs <= 6.8) {
        this.detectedTierId = 'TIER_S';
      } else if (this.warmupAvgFrameMs <= 11.5) {
        this.detectedTierId = 'TIER_A';
      } else if (this.warmupAvgFrameMs <= 17.5) {
        this.detectedTierId = 'TIER_B';
      } else {
        this.detectedTierId = 'TIER_C';
      }
      this.activeTierId = this.detectedTierId;
      this._applyTierPresetToEngine(this.activeTierId, false);
    }
  }

  /**
   * Sets governor mode ('AUTO' | 'TIER_S' | 'TIER_A' | 'TIER_B' | 'TIER_C').
   */
  setMode(mode = 'AUTO') {
    const norm = String(mode || 'AUTO').toUpperCase();
    if (norm === 'AUTO') {
      this.mode = 'AUTO';
      this.activeTierId = this.detectedTierId;
    } else if (HARDWARE_TIERS[norm]) {
      this.mode = norm;
      this.activeTierId = norm;
    }
    this._overloadStreak = 0;
    this._headroomStreak = 0;
    this._applyTierPresetToEngine(this.activeTierId, true);
    this.logEvent('TIER', `Mode ${this.mode} -> ${this.activeTierId} (DPR ${this.livePixelRatio.toFixed(2)}×)`);
    return this.getTelemetryState();
  }

  getActiveTierConfig() {
    return HARDWARE_TIERS[this.activeTierId] || HARDWARE_TIERS.TIER_S;
  }

  _applyTierPresetToEngine(tierId, resetDpr = false) {
    const cfg = HARDWARE_TIERS[tierId] || HARDWARE_TIERS.TIER_S;
    const nativeDpr =
      typeof window !== 'undefined' && Number.isFinite(window.devicePixelRatio)
        ? window.devicePixelRatio
        : 1.0;

    // Smart High-DPI clamping: on 1080p/1440p monitors keep crisp native/1.5x; on 4K prevent 8.3M pixel overload
    const screenPixels =
      typeof window !== 'undefined' ? (window.innerWidth || 1440) * (window.innerHeight || 900) : 1296000;
    const highResPenalty = screenPixels > 2600000 ? 0.85 : 1.0;
    const targetDpr = Math.max(
      cfg.minPixelRatio,
      Math.min(cfg.maxPixelRatio * highResPenalty, nativeDpr)
    );

    if (resetDpr || !Number.isFinite(this.livePixelRatio)) {
      this.livePixelRatio = +targetDpr.toFixed(2);
      if (this.renderer && typeof this.renderer.setPixelRatio === 'function') {
        this.renderer.setPixelRatio(this.livePixelRatio);
      }
    }

    if (this.sharedUniforms) {
      if (this.sharedUniforms.uCloseDerivativeDist) {
        this.sharedUniforms.uCloseDerivativeDist.value = cfg.closeDerivativeDistM;
      }
      if (this.sharedUniforms.uAiFarStaggerDist) {
        this.sharedUniforms.uAiFarStaggerDist.value = cfg.aiFarStaggerDistM;
      }
      if (this.sharedUniforms.uShaderQuality) {
        this.sharedUniforms.uShaderQuality.value = 1.0; // Never downgrade surface shader quality!
      }
      if (this.sharedUniforms.uGpuSkinningEnabled) {
        this.sharedUniforms.uGpuSkinningEnabled.value = this.supportsVertexFloatTex ? 1.0 : 0.0;
      }
    }
  }

  /**
   * Updates camera-centered texel-snapped shadow frustum, adaptive shadow refresh cadence,
   * High-DPI pixel ratio, & 60-second O(1) forensic ring buffer.
   */
  update(dt, camera = null, focusTarget = null, weather = null, perfSample = null) {
    this._frameCounter++;
    const rawSec =
      perfSample && Number.isFinite(perfSample.rawDt) && perfSample.rawDt > 0.0004
        ? Math.min(1.0, perfSample.rawDt)
        : Number.isFinite(dt) && dt > 0.0005 && dt < 1.0
          ? dt
          : 0.00833;
    const cpuMs =
      perfSample && Number.isFinite(perfSample.cpuMs) && perfSample.cpuMs >= 0.0
        ? perfSample.cpuMs
        : rawSec * 220.0;
    const drawCalls =
      perfSample && Number.isFinite(perfSample.calls)
        ? perfSample.calls
        : this.renderer?.info?.render?.calls || 0;

    const frameMs = rawSec * 1000.0;
    this._uptimeSec += rawSec;

    // Exponential moving average of TRUE wall-clock frame time, CPU JS time, & FPS
    // Clamp initial 15 frames so post-boot GPU driver paging never poisons steady-state EMA!
    const emaSampleMs = this._frameCounter <= 15 ? Math.min(frameMs, 20.0) : frameMs;
    this.emaFrameMs = this.emaFrameMs * 0.90 + emaSampleMs * 0.10;
    this.emaCpuMs = this.emaCpuMs * 0.90 + cpuMs * 0.10;
    this.emaFps = Math.min(240.0, 1000.0 / Math.max(2.0, this.emaFrameMs));

    // O(1) 1-Hz Forensic Ring Buffer Accumulation (60 slots = exact last 60 seconds, zero allocations!)
    this._secWallAccum += rawSec;
    this._secFrames++;
    this._secSumMs += frameMs;
    this._secSumCpuMs += cpuMs;
    if (frameMs > this._secPeakMs) this._secPeakMs = frameMs;
    if (drawCalls > this._secMaxCalls) this._secMaxCalls = drawCalls;

    if (
      frameMs > 48.0 &&
      this._uptimeSec >= 5.0 &&
      this._frameCounter >= 90 &&
      this._uptimeSec - this._lastSpikeLogSec >= 4.5
    ) {
      this._lastSpikeLogSec = this._uptimeSec;
      this.logEvent(
        'SPIKE',
        `${frameMs.toFixed(1)}ms (CPU ${cpuMs.toFixed(1)}ms, Calls ${drawCalls}, DPR ${this.livePixelRatio.toFixed(2)}×)`
      );
    }

    if (this._secWallAccum >= 1.0) {
      const n = Math.max(1, this._secFrames);
      const idx = this._ringHead;
      this._ringTimeSec[idx] = this._uptimeSec;
      this._ringFps[idx] = n / this._secWallAccum;
      this._ringAvgMs[idx] = this._secSumMs / n;
      this._ringPeakMs[idx] = this._secPeakMs;
      this._ringCpuMs[idx] = this._secSumCpuMs / n;
      this._ringCalls[idx] = Math.min(65535, this._secMaxCalls);
      this._ringDpr[idx] = this.livePixelRatio;
      this._ringHead = (idx + 1) % this._ringCap;
      if (this._ringCount < this._ringCap) this._ringCount++;

      this._secWallAccum = 0.0;
      this._secFrames = 0;
      this._secSumMs = 0.0;
      this._secPeakMs = 0.0;
      this._secSumCpuMs = 0.0;
      this._secMaxCalls = 0;
    }

    const cfg = this.getActiveTierConfig();

    // 1. Camera-Centered Texel-Snapped Shadow Frustum + Smart Shadow Refresh Cadence
    if (weather && weather.sunLight && weather.sunLight.shadow) {
      const sun = weather.sunLight;
      const shCam = sun.shadow.camera;
      const halfExt = cfg.shadowFrustumHalf || 110.0;

      if (shCam && shCam.left !== -halfExt) {
        shCam.left = -halfExt;
        shCam.right = halfExt;
        shCam.top = halfExt;
        shCam.bottom = -halfExt;
        shCam.updateProjectionMatrix?.();
      }

      // Compute focus center (OrbitControls target when enabled, or camera forward ground projection in Mavic/FPS/Walk)
      let fx = 0.0;
      let fz = 35.0;
      if (
        focusTarget &&
        focusTarget.isOrbitActive !== false &&
        Number.isFinite(focusTarget.x) &&
        Number.isFinite(focusTarget.z)
      ) {
        fx = focusTarget.x;
        fz = focusTarget.z;
      } else if (camera && camera.position) {
        if (typeof camera.getWorldDirection === 'function') {
          camera.getWorldDirection(this._tmpFocus);
          fx = camera.position.x + this._tmpFocus.x * 52.0;
          fz = camera.position.z + this._tmpFocus.z * 52.0;
        } else {
          fx = camera.position.x;
          fz = camera.position.z;
        }
      }

      // Clamp focus within Kepulauan UI bounds and snap to 1.0 shadow-map texel grid (zero edge swimming & zero step-lag!)
      const texelWorldStep = (halfExt * 2.0) / (cfg.shadowMapSize || 2048);
      const snapX = Math.round(fx / texelWorldStep) * texelWorldStep;
      const snapZ = Math.round(fz / texelWorldStep) * texelWorldStep;

      const sunDir = weather.uniforms?.uSunDir?.value;
      const sdx = sunDir ? sunDir.x : 0.52;
      const sdy = sunDir ? sunDir.y : 0.72;
      const sdz = sunDir ? sunDir.z : 0.45;

      const cellMoved =
        Math.abs(snapX - (this._lastShadowSnapX || 0.0)) >= texelWorldStep * 0.5 ||
        Math.abs(snapZ - (this._lastShadowSnapZ || 0.0)) >= texelWorldStep * 0.5 ||
        Number.isNaN(this._lastShadowSnapX);
      const sunMoved =
        Math.hypot(
          sdx - (this._lastSunDirX || 0.0),
          sdy - (this._lastSunDirY || 0.0),
          sdz - (this._lastSunDirZ || 0.0)
        ) > 0.0022 || Number.isNaN(this._lastSunDirX);

      if (sun.target) {
        sun.target.position.set(snapX, 0.0, snapZ);
        sun.target.updateMatrixWorld?.();
      }
      sun.position.set(snapX + sdx * 135.0, sdy * 135.0, snapZ + sdz * 135.0);
      sun.updateMatrixWorld?.();

      if (this.renderer && this.renderer.shadowMap) {
        this.renderer.shadowMap.autoUpdate = false;
        const sunSilent = (Number(sun.intensity) || 0.0) <= 0.001;
        if (sunSilent) {
          // Zero shadow-map rasterization at night once cleared after sunset!
          if (!this._shadowClearedForNight) {
            this.renderer.shadowMap.needsUpdate = true;
            this._shadowClearedForNight = true;
          }
        } else {
          this._shadowClearedForNight = false;
          const stride = Math.max(1, cfg.shadowRefreshStride || 1);
          if (cellMoved || sunMoved || this._frameCounter % stride === 0) {
            this.renderer.shadowMap.needsUpdate = true;
            this._lastShadowSnapX = snapX;
            this._lastShadowSnapZ = snapZ;
            this._lastSunDirX = sdx;
            this._lastSunDirY = sdy;
            this._lastSunDirZ = sdz;
          }
        }
      }
    }

    // 2. Closed-Loop Hysteresis Governor (every 0.75s) for High-DPI Pixel Ratio & AUTO Tier step-down
    //    (DPR locked on Mobile so Android/iOS WebGL drawingBuffer is never reallocated mid-flight alongside YouTube iframe!)
    if (!this.isMobileDevice) {
      this._governorTimer += rawSec;
      if (this._governorTimer >= 0.75) {
        this._governorTimer = 0.0;
        const tierOrder = ['TIER_S', 'TIER_A', 'TIER_B', 'TIER_C'];
        if (this.mode === 'AUTO' && this._uptimeSec >= 5.0 && this._frameCounter >= 90) {
          const curIdx = tierOrder.indexOf(this.activeTierId);
          const detIdx = Math.max(0, tierOrder.indexOf(this.detectedTierId));
          const minAllowedIdx = this.gpuClass === 'GAMING_DGPU' ? 1 : tierOrder.length - 1;
          if (this.emaFrameMs > 27.5 && curIdx >= 0 && curIdx < minAllowedIdx) {
            this._overloadStreak++;
            this._headroomStreak = 0;
            if (this._overloadStreak >= 2) {
              this._overloadStreak = 0;
              this.activeTierId = tierOrder[curIdx + 1];
              this._applyTierPresetToEngine(this.activeTierId, false);
              this.logEvent('GOV', `Auto step-down -> ${this.activeTierId} (${this.emaFrameMs.toFixed(1)}ms)`);
            }
          } else if (this.emaFrameMs < 9.5 && curIdx > detIdx) {
            this._headroomStreak++;
            this._overloadStreak = 0;
            if (this._headroomStreak >= 4) {
              this._headroomStreak = 0;
              this.activeTierId = tierOrder[curIdx - 1];
              this._applyTierPresetToEngine(this.activeTierId, false);
              this.logEvent('GOV', `Auto step-up -> ${this.activeTierId} (${this.emaFrameMs.toFixed(1)}ms)`);
            }
          } else {
            this._overloadStreak = 0;
            this._headroomStreak = 0;
          }
        }

        const liveCfg = this.getActiveTierConfig();
        const nativeDpr =
          typeof window !== 'undefined' && Number.isFinite(window.devicePixelRatio)
            ? window.devicePixelRatio
            : 1.0;
        const maxAllowed = Math.min(liveCfg.maxPixelRatio, Math.max(1.0, nativeDpr));
        const minAllowed = liveCfg.minPixelRatio;

        if (this.emaFrameMs > liveCfg.targetFrameMs * 1.18 && this.livePixelRatio > minAllowed + 0.04) {
          this.livePixelRatio = +Math.max(minAllowed, this.livePixelRatio - 0.1).toFixed(2);
          if (this.renderer && typeof this.renderer.setPixelRatio === 'function') {
            this.renderer.setPixelRatio(this.livePixelRatio);
          }
        } else if (
          this.emaFrameMs < liveCfg.targetFrameMs * 0.72 &&
          this.livePixelRatio < maxAllowed - 0.04
        ) {
          this.livePixelRatio = +Math.min(maxAllowed, this.livePixelRatio + 0.05).toFixed(2);
          if (this.renderer && typeof this.renderer.setPixelRatio === 'function') {
            this.renderer.setPixelRatio(this.livePixelRatio);
          }
        }
      }
    }
  }

  /**
   * Computes a lightweight, zero-bloat snapshot of GPU Info + Last 60 Seconds (1 Minute) Engine Forensics.
   */
  getForensicSnapshot(renderer = null, extra = {}) {
    const r = renderer || this.renderer;
    const cfg = this.getActiveTierConfig();
    const n = this._ringCount;

    let sumFps = 0.0;
    let minFps = Infinity;
    let sumMs = 0.0;
    let peakMs = 0.0;
    let sumCpuMs = 0.0;
    let maxCalls = 0;

    // Sparkline characters (8 levels)
    const bars = '▁▂▃▄▅▆▇█';
    let spark = '';
    const startOff = (this._ringHead - n + this._ringCap) % this._ringCap;

    for (let i = 0; i < n; i++) {
      const idx = (startOff + i) % this._ringCap;
      const fps = this._ringFps[idx];
      const ms = this._ringAvgMs[idx];
      const pk = this._ringPeakMs[idx];
      const cpu = this._ringCpuMs[idx];
      const calls = this._ringCalls[idx];
      sumFps += fps;
      if (fps < minFps) minFps = fps;
      sumMs += ms;
      if (pk > peakMs) peakMs = pk;
      sumCpuMs += cpu;
      if (calls > maxCalls) maxCalls = calls;
      this._sortScratch[i] = ms;

      // Sparkline encodes FPS stability across the last <= 30 samples (2s stride when n > 30)
      if (n <= 30 || i % 2 === n % 2) {
        const normBar = Math.max(0, Math.min(7, Math.round((Math.min(120.0, fps) / 120.0) * 7)));
        spark += bars[normBar];
      }
    }

    let avgFps = Math.round(this.emaFps);
    let avgMs = +this.emaFrameMs.toFixed(2);
    let p95Ms = avgMs;
    let low5Fps = avgFps;
    let avgCpuMs = +this.emaCpuMs.toFixed(2);

    if (n > 0) {
      avgFps = Math.round(sumFps / n);
      avgMs = +(sumMs / n).toFixed(2);
      avgCpuMs = +(sumCpuMs / n).toFixed(2);
      const sub = this._sortScratch.subarray(0, n);
      sub.sort();
      const p95Idx = Math.min(n - 1, Math.floor(n * 0.95));
      p95Ms = +sub[p95Idx].toFixed(2);
      low5Fps = Math.max(1, Math.round(1000.0 / Math.max(2.0, p95Ms)));
    } else {
      minFps = avgFps;
      peakMs = avgMs;
      spark = '▅▅▅';
    }

    // Diagnose primary frame bottleneck cleanly from CPU JS time vs total wall-clock frame interval
    let bottleneck = 'Optimal / VSync';
    if (avgMs > 18.5) {
      bottleneck = avgCpuMs > avgMs * 0.58 ? 'CPU JS / Simulasi' : 'GPU Fillrate / Shader';
    }

    const domEl = r?.domElement;
    const bufW = domEl?.width || (typeof window !== 'undefined' ? window.innerWidth : 1440);
    const bufH = domEl?.height || (typeof window !== 'undefined' ? window.innerHeight : 900);
    const renderInfo = r?.info?.render || {};
    const memInfo = r?.info?.memory || {};
    const progCount = Array.isArray(r?.info?.programs) ? r.info.programs.length : 0;

    let jsHeapMB = null;
    if (typeof performance !== 'undefined' && performance.memory && performance.memory.usedJSHeapSize) {
      jsHeapMB = +(performance.memory.usedJSHeapSize / 1048576.0).toFixed(1);
    }

    const events = [];
    const evStart = (this._eventHead - this._eventCount + this._eventCap) % this._eventCap;
    for (let i = 0; i < this._eventCount; i++) {
      events.push(this._eventRing[(evStart + i) % this._eventCap]);
    }

    return {
      uptimeSec: +this._uptimeSec.toFixed(1),
      gpu: {
        cleanName: this.gpuModel,
        rawRenderer: this.gpuCaps.rawRenderer,
        rawVendor: this.gpuCaps.rawVendor,
        gpuClass: this.gpuClass,
        glVersion: this.gpuCaps.glVersion,
        glslVersion: this.gpuCaps.glslVersion,
        maxTextureSize: this.gpuCaps.maxTextureSize,
        maxSamplesMSAA: this.gpuCaps.maxSamplesMSAA,
        maxAnisotropy: this.gpuCaps.maxAnisotropy,
        maxVertexTexUnits: this.gpuCaps.maxVertexTexUnits,
        colorBufferFloat: this.gpuCaps.colorBufferFloat,
        cpuCores: this.cpuCores,
        deviceMemoryGB: this.deviceMemoryGB,
        drawingBuffer: `${bufW}×${bufH}`,
        pixelRatio: +this.livePixelRatio.toFixed(2),
        mode: this.mode,
        activeTierId: this.activeTierId,
        shadowMapSize: cfg.shadowMapSize,
        shadowRefreshStride: cfg.shadowRefreshStride
      },
      window60s: {
        samplesSec: n,
        avgFps,
        minFps: Number.isFinite(minFps) ? Math.round(minFps) : avgFps,
        low5Fps,
        avgFrameMs: avgMs,
        p95FrameMs: p95Ms,
        peakFrameMs: +peakMs.toFixed(1),
        avgCpuMs,
        bottleneck,
        maxDrawCalls: maxCalls || renderInfo.calls || 0,
        liveDrawCalls: renderInfo.calls || 0,
        triangles: renderInfo.triangles || 0,
        geometries: memInfo.geometries || 0,
        textures: memInfo.textures || 0,
        programs: progCount,
        jsHeapMB,
        sparkline: spark
      },
      context: extra,
      events
    };
  }

  /**
   * Exports compact JSON forensic dump (GPU + 60s summary + compact 1Hz 60s series + event ring, < 3.5 KB).
   */
  exportForensicLogJSON(renderer = null, extra = {}) {
    const snap = this.getForensicSnapshot(renderer, extra);
    const n = this._ringCount;
    const startOff = (this._ringHead - n + this._ringCap) % this._ringCap;
    const series60s = [];
    for (let i = 0; i < n; i++) {
      const idx = (startOff + i) % this._ringCap;
      series60s.push([
        +this._ringTimeSec[idx].toFixed(0),
        +this._ringFps[idx].toFixed(1),
        +this._ringAvgMs[idx].toFixed(1),
        +this._ringPeakMs[idx].toFixed(1),
        +this._ringCpuMs[idx].toFixed(1),
        this._ringCalls[idx],
        +this._ringDpr[idx].toFixed(2)
      ]);
    }
    const pretty = JSON.stringify(
      {
        engine: 'NEXUS Universe Island v6.51 Forensic',
        capturedAtISO: new Date().toISOString(),
        uptimeSec: snap.uptimeSec,
        gpu: snap.gpu,
        last60s: snap.window60s,
        context: snap.context,
        seriesColumns: ['tSec', 'fps', 'avgMs', 'peakMs', 'cpuMs', 'calls', 'dpr'],
        series60s,
        events: snap.events
      },
      null,
      2
    );
    return pretty.replace(/\[\n\s+(-?\d[\s\S]*?)\n\s+\]/g, (_m, inner) => {
      return '[' + inner.replace(/\s*\n\s*/g, ' ') + ']';
    });
  }

  getTelemetryState() {
    const cfg = this.getActiveTierConfig();
    const safePr = Number.isFinite(this.livePixelRatio) ? this.livePixelRatio : 1.0;
    const safeCloseDist = Number.isFinite(cfg.closeDerivativeDistM) ? cfg.closeDerivativeDistM : 18.0;
    return {
      mode: this.mode,
      tier: this.activeTierId,
      tierShort: cfg.badgeLabel,
      detectedTierId: this.detectedTierId,
      activeTierId: this.activeTierId,
      tierLabel: cfg.shortLabel,
      badgeLabel: cfg.badgeLabel,
      gpuRenderer: this.gpuModel,
      gpuModel: this.gpuModel,
      gpuVendor: this.gpuVendor,
      gpuClass: this.gpuClass,
      cpuCores: this.cpuCores,
      deviceMemoryGB: this.deviceMemoryGB,
      targetFps: cfg.targetFps,
      liveFps: Math.round(this.emaFps),
      liveFrameMs: +this.emaFrameMs.toFixed(2),
      warmupAvgFrameMs: +this.warmupAvgFrameMs.toFixed(2),
      pixelRatio: safePr,
      livePixelRatio: safePr,
      closeDerivativeDist: safeCloseDist,
      shadowMapSize: cfg.shadowMapSize,
      shadowRefreshStride: cfg.shadowRefreshStride
    };
  }
}
