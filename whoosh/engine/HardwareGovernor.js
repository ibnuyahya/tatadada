import * as THREE from 'three';

/**
 * ============================================================================
 * HARDWARE TELEMETRY GOVERNOR & TEXEL-SNAPPED SHADOW STABILIZER
 * - Profiles display refresh rate (60Hz / 120Hz / 144Hz+), CPU threads & mobile
 * - Supports AUTO closed-loop scaling or manual ULTRA / HIGH / BALANCED / ECO
 * - Keeps WebGPU `ShadowDepthTexture` & `castShadow` state 100% immutable after
 *   boot (avoiding Three.js r170 ShadowNode bind-group invalidation freezes)
 *   while snapping directional light frustum in world-texel increments!
 * ============================================================================
 */
export const QUALITY_TIERS = {
  ULTRA: {
    id: 'ULTRA',
    label: 'ULTRA (PBR Clearcoat + Soft Shadows)',
    maxDpr: 1.65,
    shadows: true,
    shadowMapSize: 2048,
    shadowRadius: 88,
    chunksAhead: 15,
    chunksBehind: 3,
    stationNpcCount: 56,
    npcCullDist: 110
  },
  HIGH: {
    id: 'HIGH',
    label: 'HIGH (PBR + Soft Shadows)',
    maxDpr: 1.35,
    shadows: true,
    shadowMapSize: 1536,
    shadowRadius: 76,
    chunksAhead: 13,
    chunksBehind: 3,
    stationNpcCount: 44,
    npcCullDist: 90
  },
  BALANCED: {
    id: 'BALANCED',
    label: 'BALANCED (60+ FPS Stabil)',
    maxDpr: 1.15,
    shadows: true,
    shadowMapSize: 1536,
    shadowRadius: 64,
    chunksAhead: 12,
    chunksBehind: 2,
    stationNpcCount: 32,
    npcCullDist: 72
  },
  ECO: {
    id: 'ECO',
    label: 'ECO (Hemat Resource / Laptop)',
    maxDpr: 1.0,
    shadows: true,
    shadowMapSize: 1024,
    shadowRadius: 52,
    chunksAhead: 10,
    chunksBehind: 2,
    stationNpcCount: 22,
    npcCullDist: 55
  }
};

const TIER_ORDER = ['ULTRA', 'HIGH', 'BALANCED', 'ECO'];

const _lightSpacePos = new THREE.Vector3();
const _lightSpaceTarget = new THREE.Vector3();
const _lightMatrix = new THREE.Matrix4();
const _lightInvMatrix = new THREE.Matrix4();
const _snappedFocus = new THREE.Vector3();
const _worldUp = new THREE.Vector3(0, 1, 0);
const _fallbackUp = new THREE.Vector3(0, 0, 1);
const _lightDir = new THREE.Vector3();

export class HardwareTelemetryGovernor {
  constructor(onTierChanged) {
    this.onTierChanged = onTierChanged;
    this.mode = 'AUTO'; // 'AUTO' | 'ULTRA' | 'HIGH' | 'BALANCED' | 'ECO'

    const isMobile = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || '');
    const cores = navigator.hardwareConcurrency || 4;
    const memGb = navigator.deviceMemory || 8;

    this.shadowsEnabledAtBoot = !(isMobile && cores <= 4);

    if (isMobile || cores <= 4 || memGb <= 4) {
      this.activeTierKey = 'BALANCED';
    } else if (cores >= 8 && memGb >= 8) {
      this.activeTierKey = 'HIGH';
    } else {
      this.activeTierKey = 'BALANCED';
    }

    this.bootShadowMapSize = this.activeTier.shadowMapSize;
    this.detectedHz = 60;
    this.emaFrameMs = 16.6;
    this.evalTimer = 0;
    this.cooldown = 6.0; // Grace period after boot & pipeline warmup
  }

  get activeTier() {
    const base = QUALITY_TIERS[this.activeTierKey] || QUALITY_TIERS.HIGH;
    return {
      ...base,
      shadows: this.shadowsEnabledAtBoot,
      shadowMapSize: this.bootShadowMapSize
    };
  }

  setMode(modeKey) {
    if (modeKey === 'AUTO') {
      this.mode = 'AUTO';
      return;
    }
    if (QUALITY_TIERS[modeKey]) {
      this.mode = modeKey;
      if (this.activeTierKey !== modeKey) {
        this.activeTierKey = modeKey;
        if (this.onTierChanged) this.onTierChanged(this.activeTier, this.mode);
      }
    }
  }

  sampleFrame(dtSeconds) {
    const ms = Math.min(80, Math.max(2, dtSeconds * 1000));
    this.emaFrameMs = this.emaFrameMs * 0.94 + ms * 0.06;

    if (this.emaFrameMs < 9.5 && this.detectedHz < 120) {
      this.detectedHz = 120;
    }

    if (this.cooldown > 0) {
      this.cooldown -= dtSeconds;
      return;
    }

    if (this.mode !== 'AUTO') return;

    this.evalTimer += dtSeconds;
    if (this.evalTimer < 4.0) return;
    this.evalTimer = 0;

    const idx = TIER_ORDER.indexOf(this.activeTierKey);
    // Step down only if sustained FPS drops below ~35 FPS (frameMs > 28.5ms)
    if (this.emaFrameMs > 28.5 && idx < TIER_ORDER.length - 1) {
      this.activeTierKey = TIER_ORDER[idx + 1];
      this.cooldown = 5.0;
      if (this.onTierChanged) this.onTierChanged(this.activeTier, this.mode);
    } else if (this.emaFrameMs < 11.2 && idx > 0) {
      // Step up when ample GPU headroom exists (< 11.2ms average frame time)
      this.activeTierKey = TIER_ORDER[idx - 1];
      this.cooldown = 7.0;
      if (this.onTierChanged) this.onTierChanged(this.activeTier, this.mode);
    }
  }

  /**
   * Updates directional sun shadow camera around `focusWorldPos` snapped to
   * shadow-map world texels so shadows never crawl or shimmer while moving,
   * without mutating `autoUpdate` or `castShadow` flags on WebGPU!
   */
  updateDirectionalShadow(sunLight, focusWorldPos, sunOffset) {
    const tier = this.activeTier;
    if (!this.shadowsEnabledAtBoot) {
      sunLight.target.position.copy(focusWorldPos);
      sunLight.target.updateMatrixWorld();
      sunLight.position.copy(focusWorldPos).add(sunOffset);
      return;
    }

    const r = tier.shadowRadius;
    const mapSize = this.bootShadowMapSize;
    const worldUnitsPerTexel = (r * 2.0) / mapSize;

    _lightDir.copy(sunOffset).normalize();
    const upRef = Math.abs(_lightDir.y) > 0.95 ? _fallbackUp : _worldUp;
    _lightMatrix.lookAt(_lightDir, _snappedFocus.set(0, 0, 0), upRef);
    _lightInvMatrix.copy(_lightMatrix).invert();

    _lightSpacePos.copy(focusWorldPos).applyMatrix4(_lightInvMatrix);
    _lightSpacePos.x = Math.round(_lightSpacePos.x / worldUnitsPerTexel) * worldUnitsPerTexel;
    _lightSpacePos.y = Math.round(_lightSpacePos.y / worldUnitsPerTexel) * worldUnitsPerTexel;
    _lightSpacePos.z = Math.round(_lightSpacePos.z / worldUnitsPerTexel) * worldUnitsPerTexel;

    _lightSpaceTarget.copy(_lightSpacePos).applyMatrix4(_lightMatrix);

    sunLight.target.position.copy(_lightSpaceTarget);
    sunLight.target.updateMatrixWorld();
    sunLight.position.copy(_lightSpaceTarget).add(sunOffset);

    const cam = sunLight.shadow.camera;
    if (cam.left !== -r || cam.right !== r) {
      cam.left = -r;
      cam.right = r;
      cam.top = r;
      cam.bottom = -r;
      cam.near = 4.0;
      cam.far = 260.0;
      cam.updateProjectionMatrix();
    }
  }
}
