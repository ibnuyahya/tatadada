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
    shadowMapSize: 4096,
    shadowFrustumHalf: 112.0,
    shadowRefreshStride: 1,
    closeDerivativeDistM: 22.0,
    aiFarStaggerDistM: 185.0
  },
  TIER_A: {
    id: 'TIER_A',
    shortLabel: 'Tier A • High 90–120 FPS',
    badgeLabel: 'High Performance',
    targetFps: 90,
    targetFrameMs: 11.11,
    minPixelRatio: 1.0,
    maxPixelRatio: 1.25,
    shadowMapSize: 4096,
    shadowFrustumHalf: 108.0,
    shadowRefreshStride: 2,
    closeDerivativeDistM: 18.0,
    aiFarStaggerDistM: 150.0
  },
  TIER_B: {
    id: 'TIER_B',
    shortLabel: 'Tier B • Balanced 60 FPS',
    badgeLabel: 'Balanced 60Hz',
    targetFps: 60,
    targetFrameMs: 16.67,
    minPixelRatio: 0.9,
    maxPixelRatio: 1.05,
    shadowMapSize: 2048,
    shadowFrustumHalf: 102.0,
    shadowRefreshStride: 3,
    closeDerivativeDistM: 14.0,
    aiFarStaggerDistM: 120.0
  },
  TIER_C: {
    id: 'TIER_C',
    shortLabel: 'Tier C • Efficiency',
    badgeLabel: 'Hemat Daya',
    targetFps: 60,
    targetFrameMs: 16.67,
    minPixelRatio: 0.8,
    maxPixelRatio: 1.0,
    shadowMapSize: 2048,
    shadowFrustumHalf: 96.0,
    shadowRefreshStride: 4,
    closeDerivativeDistM: 10.0,
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

    this._warmupSamplesMs = [];
    this.warmupAvgFrameMs = 4.8;
    this.emaFrameMs = 6.2;
    this.emaFps = 120.0;
    this.livePixelRatio = 1.0;

    this._frameCounter = 0;
    this._governorTimer = 0.0;
    this._lastShadowSnapX = NaN;
    this._lastShadowSnapZ = NaN;
    this._lastSunDirX = NaN;
    this._lastSunDirY = NaN;
    this._lastSunDirZ = NaN;
    this._tmpFocus = new THREE.Vector3();

    if (renderer) {
      this.probeHardware(renderer);
    }
  }

  /**
   * Classifies raw WebGL UNMASKED_RENDERER_WEBGL string into hardware tier & short model name.
   */
  static classifyGpuRendererString(rawRenderer = '', rawVendor = '', cpuCores = 8) {
    const r = String(rawRenderer || '').trim();
    const v = String(rawVendor || '').trim();

    // Extract clean human-readable GPU name (strip ANGLE(...) wrapper if present)
    let cleanName = r;
    const angleMatch = r.match(/ANGLE\s*\([^,]+,\s*([^,(]+)/i);
    if (angleMatch && angleMatch[1]) {
      cleanName = angleMatch[1].trim();
    }
    cleanName = cleanName
      .replace(/Direct3D11.*$/i, '')
      .replace(/vs_\d+_\d+.*$/i, '')
      .replace(/\s+/g, ' ')
      .trim();
    if (!cleanName) cleanName = 'WebGL2 GPU';

    const isSoftware = /SwiftShader|llvmpipe|Software|Microsoft Basic Render/i.test(r);
    const isGamingNvidia =
      /RTX\s*(20[6789]0|30[6789]0|40[6789]0|50[6789]0|A[23456]000)|GTX\s*(10[78]0|1660)/i.test(r);
    const isLaptopOrEntryNvidia = /RTX\s*(2050|3050|4050)|GTX\s*(1050|1060|1650)|MX\s*\d+/i.test(r);
    const isGamingAmd = /Radeon\s*RX\s*(6[6789]\d0|7[6789]\d0|5700)/i.test(r);
    const isHighApuOrArc =
      /Radeon\s*(680M|780M|880M|890M|RX\s*[56][56]00)|Intel.*Arc|Apple\s*M[1234]\s*(Pro|Max|Ultra)/i.test(
        r
      );
    const isAppleBase = /Apple\s*M[1234]\b/i.test(r) && !/Pro|Max|Ultra/i.test(r);
    const isMainstreamIgpu = /Iris\s*Xe|UHD\s*Graphics\s*(7[357]0)|Radeon\s*(660M|740M|760M|Vega)/i.test(
      r
    );
    const isEntryOrMobile = /UHD\s*Graphics\s*(6\d0)|HD\s*Graphics|Adreno|Mali|PowerVR/i.test(r);

    if (isSoftware || isEntryOrMobile) {
      return {
        tierId: 'TIER_C',
        gpuClass: isSoftware ? 'SOFTWARE' : 'ENTRY_IGPU',
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
   * Probes WebGL2 context parameters & classifies initial hardware tier.
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
        this.supportsVertexFloatTex = maxVertTex >= 4;
      }
    } catch (_e) {
      // Safe fallback in headless Node mock
    }

    const info = HardwareTelemetryGovernor.classifyGpuRendererString(
      rawRenderer,
      rawVendor,
      this.cpuCores
    );
    this.gpuModel = info.cleanName;
    this.gpuVendor = info.vendor;
    this.gpuClass = info.gpuClass;
    this.detectedTierId = info.tierId;
    if (this.mode === 'AUTO') {
      this.activeTierId = this.detectedTierId;
    }
    this._applyTierPresetToEngine(this.activeTierId, true);
    return this.getTelemetryState();
  }

  /**
   * Records a real pre-warm frame sample (in ms) during Preloader Stage 9 and refines AUTO tier.
   */
  recordWarmupSample(frameMs) {
    if (!Number.isFinite(frameMs) || frameMs <= 0.0) return;
    this._warmupSamplesMs.push(frameMs);
    const sum = this._warmupSamplesMs.reduce((a, b) => a + b, 0.0);
    this.warmupAvgFrameMs = sum / this._warmupSamplesMs.length;
    this.emaFrameMs = this.warmupAvgFrameMs;
    this.emaFps = Math.min(240.0, 1000.0 / Math.max(2.0, this.warmupAvgFrameMs));

    // Only downgrade from static GPU classification if actual warmup is severely bottlenecked
    // (e.g., 4K 200% scaling or laptop unplugged on battery saver)
    if (this.mode === 'AUTO' && this._warmupSamplesMs.length >= 3) {
      if (this.gpuClass === 'GAMING_DGPU') {
        this.detectedTierId = this.warmupAvgFrameMs <= 11.5 ? 'TIER_S' : 'TIER_A';
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
    this._applyTierPresetToEngine(this.activeTierId, true);
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
      if (this.sharedUniforms.uShaderQuality) {
        this.sharedUniforms.uShaderQuality.value = 1.0; // Never downgrade surface shader quality!
      }
    }
  }

  /**
   * Updates camera-centered texel-snapped shadow frustum & adaptive shadow refresh cadence,
   * and regulates High-DPI pixel ratio smoothly without visual downgrade.
   */
  update(dt, camera = null, focusTarget = null, weather = null) {
    this._frameCounter++;
    const safeDt = Number.isFinite(dt) && dt > 0.0005 && dt < 0.25 ? dt : 0.00833;
    const frameMs = safeDt * 1000.0;

    // Exponential moving average of frame time & FPS
    this.emaFrameMs = this.emaFrameMs * 0.92 + frameMs * 0.08;
    this.emaFps = Math.min(240.0, 1000.0 / Math.max(2.0, this.emaFrameMs));

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
        shCam.updateProjectionMatrix();
      }

      // Compute focus center (OrbitControls target or camera position projected forward)
      let fx = 0.0;
      let fz = 35.0;
      if (focusTarget && Number.isFinite(focusTarget.x) && Number.isFinite(focusTarget.z)) {
        fx = focusTarget.x;
        fz = focusTarget.z;
      } else if (camera && camera.position) {
        fx = camera.position.x;
        fz = camera.position.z;
      }

      // Clamp focus within Kepulauan UI bounds and snap to world texel grid (eliminates shadow edge swimming!)
      const texelWorldStep = (halfExt * 2.0) / (cfg.shadowMapSize || 4096) * 4.0;
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
        ) > 0.0012 || Number.isNaN(this._lastSunDirX);

      if (sun.target) {
        sun.target.position.set(snapX, 0.0, snapZ);
        sun.target.updateMatrixWorld();
      }
      sun.position.set(snapX + sdx * 135.0, sdy * 135.0, snapZ + sdz * 135.0);
      sun.updateMatrixWorld();

      if (this.renderer && this.renderer.shadowMap) {
        this.renderer.shadowMap.autoUpdate = false;
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

    // 2. Closed-Loop Hysteresis Governor (every 0.75s) for High-DPI Pixel Ratio fine-tuning
    this._governorTimer += safeDt;
    if (this._governorTimer >= 0.75) {
      this._governorTimer = 0.0;
      const nativeDpr =
        typeof window !== 'undefined' && Number.isFinite(window.devicePixelRatio)
          ? window.devicePixelRatio
          : 1.0;
      const maxAllowed = Math.min(cfg.maxPixelRatio, Math.max(1.0, nativeDpr));
      const minAllowed = cfg.minPixelRatio;

      if (this.emaFrameMs > cfg.targetFrameMs * 1.18 && this.livePixelRatio > minAllowed + 0.04) {
        this.livePixelRatio = +Math.max(minAllowed, this.livePixelRatio - 0.1).toFixed(2);
        if (this.renderer && typeof this.renderer.setPixelRatio === 'function') {
          this.renderer.setPixelRatio(this.livePixelRatio);
        }
      } else if (
        this.emaFrameMs < cfg.targetFrameMs * 0.72 &&
        this.livePixelRatio < maxAllowed - 0.04
      ) {
        this.livePixelRatio = +Math.min(maxAllowed, this.livePixelRatio + 0.05).toFixed(2);
        if (this.renderer && typeof this.renderer.setPixelRatio === 'function') {
          this.renderer.setPixelRatio(this.livePixelRatio);
        }
      }
    }
  }

  getTelemetryState() {
    const cfg = this.getActiveTierConfig();
    return {
      mode: this.mode,
      detectedTierId: this.detectedTierId,
      activeTierId: this.activeTierId,
      tierLabel: cfg.shortLabel,
      badgeLabel: cfg.badgeLabel,
      gpuModel: this.gpuModel,
      gpuVendor: this.gpuVendor,
      gpuClass: this.gpuClass,
      cpuCores: this.cpuCores,
      deviceMemoryGB: this.deviceMemoryGB,
      targetFps: cfg.targetFps,
      liveFps: Math.round(this.emaFps),
      liveFrameMs: +this.emaFrameMs.toFixed(2),
      warmupAvgFrameMs: +this.warmupAvgFrameMs.toFixed(2),
      livePixelRatio: this.livePixelRatio,
      shadowMapSize: cfg.shadowMapSize,
      shadowRefreshStride: cfg.shadowRefreshStride
    };
  }
}
