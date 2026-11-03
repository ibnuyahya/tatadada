import * as THREE from 'three';
import { PhysicsKernel } from './PhysicsKernel.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v4.0 — Hard 3D Solid-Body Non-Penetration Camera Solver)
 * ./core/CameraDirector.js
 *
 * Implements Hukum Emas 15 & Bagian 7 UI Ergonomics:
 * - Horizon-Locked Nadir 90° View (eps = 0.45m micro-azimuthal shift preventing 90° world roll)
 * - Non-Occluding Portrait & Desktop ViewOffset (applyNonOccludingViewportOffset)
 * - Strict 3D Solid-Body Non-Penetration Solver (resolveSoftSpatialCollision -> PhysicsKernel.resolveHardCameraCollision3D)
 * - Frame-rate-independent Exponential Damping 6-DOF Drone & Sivitas Pedestrian Camera Director
 */

export function setHorizonLockedNadirView(
  camera,
  controls,
  targetX = 0.0,
  targetZ = 0.0,
  heightY = 142.0,
  topVecX = 0.0,
  topVecZ = -1.0
) {
  camera.up.set(0, 1, 0);
  camera.rotation.z = 0;
  const len = Math.hypot(topVecX, topVecZ) || 1.0;
  const eps = 0.45;
  if (controls && controls.target) {
    controls.target.set(targetX, 0, targetZ);
  }
  camera.position.set(
    targetX - (topVecX / len) * eps,
    heightY,
    targetZ - (topVecZ / len) * eps
  );
  camera.lookAt(targetX, 0, targetZ);
  camera.updateProjectionMatrix();
  if (controls && typeof controls.update === 'function') {
    controls.update();
  }
}

export function applyNonOccludingViewportOffset(
  camera,
  viewportW,
  viewportH,
  isPanelExpanded,
  panelRect = { width: 280, height: 0 },
  baseVerticalFovOverride = null
) {
  const safeW = Math.max(1, viewportW);
  const safeH = Math.max(1, viewportH);
  const isPortrait = safeH > safeW || safeW < 768;
  camera.aspect = safeW / safeH;

  const baseVerticalFov = Number.isFinite(baseVerticalFovOverride) ? baseVerticalFovOverride : 48;
  const maxPortraitFov = baseVerticalFov > 60 ? 96 : 78;
  if (camera.aspect < 1.0) {
    const rad = (baseVerticalFov * Math.PI) / 180;
    const hFov = 2 * Math.atan(Math.tan(rad / 2) / Math.max(0.52, camera.aspect));
    camera.fov = Math.min(maxPortraitFov, (hFov * 180) / Math.PI);
  } else {
    camera.fov = baseVerticalFov;
  }

  if (!isPanelExpanded) {
    camera.clearViewOffset();
    camera.updateProjectionMatrix();
    return;
  }

  if (isPortrait) {
    // Portrait / Android Bottom-Sheet (max-height <= 34vh): shift optical center upward into free canvas
    const sheetH = Math.min(panelRect.height || safeH * 0.32, safeH * 0.34);
    const offsetY = Math.round(sheetH * 0.5);
    camera.setViewOffset(safeW, safeH, 0, offsetY, safeW, safeH);
  } else {
    // Desktop / Landscape Left/Right Side-Drawer (max-width <= 280px): shift optical center horizontally
    const drawerW = Math.min(panelRect.width || 270, 280);
    // Positive x offset shifts rendered scene left when drawer is on the right, negative when on the left.
    const sideSign = panelRect.side === 'left' ? -1 : 1;
    const offsetX = Math.round(drawerW * 0.5) * sideSign;
    camera.setViewOffset(safeW, safeH, offsetX, 0, safeW, safeH);
  }
  camera.updateProjectionMatrix();
}

export function resolveSoftSpatialCollision(
  pos,
  vel,
  colliders = [],
  safetyRadius = 0.68,
  minFloorY = 1.65,
  dt = 0.016,
  prevPos = null,
  physicsContext = null,
  isWalk = false
) {
  void dt;
  const ctx = physicsContext || null;
  const staticColliders =
    ctx && Array.isArray(ctx.staticColliders) && ctx.staticColliders.length > 0
      ? ctx.staticColliders
      : colliders;
  const dynamicColliders =
    ctx && typeof ctx.getDynamicColliders === 'function'
      ? ctx.getDynamicColliders()
      : ctx && Array.isArray(ctx.dynamicColliders)
        ? ctx.dynamicColliders
        : [];

  const hit = PhysicsKernel.resolveHardCameraCollision3D(pos, prevPos, vel, {
    staticColliders,
    dynamicColliders,
    radius: safetyRadius,
    bottomOffset: isWalk ? 1.68 : safetyRadius,
    topOffset: isWalk ? 0.24 : safetyRadius,
    isWalk,
    groundHeightFn: ctx?.groundHeightFn || null,
    walkableHeightFn: ctx?.walkableHeightFn || null,
    coastDistFn: ctx?.coastDistFn || null,
    waterSurfaceFn: ctx?.waterSurfaceFn || null,
    oceanWaterY: Number.isFinite(ctx?.oceanWaterY) ? ctx.oceanWaterY : -1.45,
    underpassConfig: ctx?.underpassConfig || null,
    bridgeSoffitFn: ctx?.bridgeSoffitFn || null,
    bridgeDeckFn: ctx?.bridgeDeckFn || null,
    bridgeCenterX: Number.isFinite(ctx?.bridgeCenterX) ? ctx.bridgeCenterX : -28.0,
    bridgeHalfW: Number.isFinite(ctx?.bridgeHalfW) ? ctx.bridgeHalfW : 2.6,
    ferryPierDeckFn: ctx?.ferryPierDeckFn || null
  });

  // Fallback floor clamp when no full world heightfield context was injected (e.g. standalone unit tests)
  if (!ctx && Number.isFinite(minFloorY) && pos.y < minFloorY) {
    pos.y = minFloorY;
    if (vel && vel.y < 0.0) vel.y = 0.0;
    return true;
  }

  return hit;
}

/**
 * Multi-Mode Camera Director (Google Earth Fluid Parabolic Flights, Free Orbit WASD, 6-DOF Drone, Pedestrian Walk, & 6 Living FPS Views)
 */
export class CameraDirector {
  static FPS_MODES = new Set([
    'FPS_JOGGER',
    'FPS_BUS_DRIVER',
    'FPS_GOLF_DRIVER',
    'FPS_CHAT_NPC',
    'FPS_CAT',
    'FPS_PILOT'
  ]);

  constructor(camera, controls, colliders = []) {
    this.camera = camera;
    this.controls = controls;
    this.colliders = colliders;
    this.mode = 'ORBIT'; // 'ORBIT' | 'DRONE_6DOF' | 'WALK_SIVITAS' | 'AUTO_CINEMATIC' | 'FPS_*'
    this._baseNear = Math.max(0.12, Number(camera?.near) || 0.18);

    this.velocity = new THREE.Vector3();
    this._orbitPanVel = new THREE.Vector3();
    this.yaw = 0.0;
    this.pitch = -0.15;
    this.roll = 0.0;
    this.cinematicAngle = 1.38;

    // 6 Natural Living Campus First-Person (FPS) State
    this.fpsAnchorProvider = null;
    this.sharedUniforms = null;
    this.fpsSubjectOffset = 0;
    this.fpsYawOffset = 0.0;
    this.fpsPitchOffset = 0.0;
    this.lastFPSStatus = null;
    this._fpsSmoothPos = new THREE.Vector3();
    this._fpsSmoothFwd = new THREE.Vector3(0, 0, -1);
    this._fpsSmoothUp = new THREE.Vector3(0, 1, 0);
    this._fpsPoseInitialized = false;
    this._fpsTmpF1 = new THREE.Vector3();
    this._fpsTmpR1 = new THREE.Vector3();
    this._fpsTmpF2 = new THREE.Vector3();
    this._fpsTmpU2 = new THREE.Vector3();

    // Google Earth C^2 Parabolic Flight State
    this.transitionActive = false;
    this.flightElapsed = 0.0;
    this.flightDuration = 2.0;
    this.flightArcLiftY = 0.0;
    this.flightStartPos = new THREE.Vector3();
    this.flightStartTarget = new THREE.Vector3();
    this.goalPos = new THREE.Vector3();
    this.goalTarget = new THREE.Vector3();
    this._flightNadirSpec = null;
    this._flightLookTmp = new THREE.Vector3();

    this.keyState = new Set();
    this._forward = new THREE.Vector3();
    this._right = new THREE.Vector3();
    this._up = new THREE.Vector3(0, 1, 0);
    this._droneManualEngaged = false;
    this._mavicGimbalYaw = 0.0;
    this._mavicGimbalPitch = 0.0;

    // DJI Avata 2 FPV Action Cinewhoop State (155° Ultra-Wide Dynamics, Sport/Acro+, Easy-Acro, APAS 360°, & 5-Act Archipelago Freestyle)
    this.avataFlightProfile = 'SPORT'; // 'SPORT' (Auto-level ±52° bank) | 'ACRO' (Full Acro ±76° / 360° roll & vertical dive)
    this.avataAutoFreestyle = true;
    this._avataSplineT = 0.014;
    this._avataLapCount = 0;
    this._avataRejoinBlend = 0.0;
    this._avataRejoinPos = new THREE.Vector3();
    this._avataEuler = new THREE.Euler(0, 0, 0, 'YXZ');
    this._avataTmpP0 = new THREE.Vector3();
    this._avataTmpP1 = new THREE.Vector3();
    this._avataTmpP2 = new THREE.Vector3();
    this._avataVelPrev = new THREE.Vector3();
    this._avataMeta = { speedMs: 23.0, corridorWeight: 0.0, corridorType: 0 };
    this._avataMetaDummy = { speedMs: 23.0, corridorWeight: 0.0, corridorType: 0 };
    this._avataLiveLaneOffset = 0.0;
    this._avataApasTimer = 0.0;
    this._avataRepelScratch = { nx: 0, ny: 0, nz: 0, pen: 0 };
    this._avataStick = { moveX: 0, moveY: 0, lookX: 0, lookY: 0, climbY: 0, boost: false };
    this._avataYawRateAccum = 0.0;
    this._avataYawRateSmooth = 0.0;
    this._avataGForce = 1.0;
    this._avataThrottle01 = 0.35;
    this._avataBaseVerticalFov = 74.0;
    this._avataCurrentBaseFov = 48.0;
    this._avataAutoRollCooldown = 5.5;
    this._avataStuntIndex = 0;
    this._avataStunt = { active: false, type: 'ROLL_RIGHT', elapsed: 0.0, duration: 0.68 };

    // Hard 3D solid-world physics context & previous-frame position for continuous sweep anti-tunneling
    this.physicsContext = null;
    this._prevCamPos = new THREE.Vector3();
    this._hasPrevCamPos = false;
  }

  /**
   * 5-Act Full-Archipelago Action FPV Choreographed Nodes across Pulau U & Pulau I:
   * Format: [x, y, z, targetSpeedMs, corridorType]
   * corridorType:
   *   0 = Open-Air Dynamic Freestyle (multi-harmonic S-carves + context-gated Easy-Acro + 3D APAS canopy clearance)
   *   1 = Lorong Jembatan Teksas UI Precision Lock (threads INSIDE Pilon Lingga -> 148m covered truss/sunroof -> Pilon Yoni!)
   *   2 = Underpass Bawah Laut Selatan U–I Precision Lock (dives -8.2m beneath Selat UI seabed + live Bus Kuning avoidance!)
   *   3 = Kolong Jembatan Teksas Lagoon Skim (perpendicular water-skim underneath the +7.8m parabolic arch soffit!)
   *   4 = Open-Air Portal Pre-Alignment (zero lateral S-carve for laser-straight portal entry/exit + full 3D APAS clearance!)
   */
  static AVATA_ARCHIPELAGO_NODES = [
    // ACT 1: 50m Gedung Rektorat Tajug Spire Dive, YouTube Videotron Whip, & Rotunda Dancing Fountain Carve
    [-32.0, 58.0, -34.0, 22.8, 0], // 0: High NW climb above Gedung Rektorat UI
    [-27.5, 51.5, 4.0, 25.6, 0],   // 1: Cresting beside 50m Nusantara Tajug Makara spire
    [-14.5, 22.5, 46.0, 26.8, 0],  // 2: 96 km/h steep proximity dive over Courtyard & Cloister
    [6.5, 11.8, 74.0, 24.5, 0],    // 3: High-speed whip down tree-free South Plaza facing YouTube Videotron UI
    [21.5, 5.6, 106.5, 20.6, 0],   // 4: Banked high-G carve around Lapangan Rotunda & Dancing Fountain
    [-10.5, 4.8, 125.0, 19.6, 4],  // 5: Exiting Rotunda ring toward Jembatan Teksas approach
    // ACT 2: Threading Straight INSIDE Lorong Jembatan Teksas UI (West Ramp -> Pilon Lingga -> Covered Arch -> Pilon Yoni!)
    [-28.0, 3.4, 138.5, 19.0, 4],  // 6: Collinear open-air pre-alignment at x = -28.0
    [-28.0, 2.35, 152.0, 19.8, 1], // 7: Entering West Approach Ramp above central stair rail
    [-28.0, 6.28, 172.0, 21.8, 1], // 8: Shooting through Pilon Lingga Portal into covered bridge corridor
    [-28.0, 9.55, 205.0, 23.8, 1], // 9: Ascending inside crimson Warren trusses & blue wave sunroof
    [-28.0, 11.22, 245.65, 24.8, 1],// 10: Mid-span crown inside Lorong Jembatan Teksas UI (+11.22m)
    [-28.0, 9.55, 286.0, 24.2, 1], // 11: Descending East arch inside covered corridor
    [-28.0, 6.28, 319.4, 22.6, 1], // 12: Blasting out through Pilon Yoni Curvilinear Portal
    [-28.0, 3.65, 336.0, 21.5, 1], // 13: Clearing East FIB Approach Ramp
    [-28.0, 6.8, 349.0, 22.5, 4],  // 14: Collinear open-air exit over East FIB Overlook Plaza
    // ACT 3: Kompleks DPPD–ILRC–STP Waterfront Sweep & Subsea Dive INTO Underpass Bawah Laut Selatan U–I!
    [-76.0, 24.0, 342.0, 24.2, 0], // 15: Banked climb SW over East Teluk U shoreline
    [-148.0, 31.0, 312.0, 25.0, 0],// 16: Panoramic sweep toward Kompleks Inovasi DPPD–ILRC–STP
    [-208.0, 35.5, 274.0, 24.8, 0],// 17: Soaring cleanly above DPPD perimeter trees & 6-Story ILRC / 8-Story STP-UI
    [-292.0, 43.5, 236.0, 24.2, 0],// 18: Cresting cleanly above Southern Highland Rainforest canopy (max tree 37m)
    [-352.0, 21.5, 226.0, 22.0, 4],// 19: Banking onto tree-free Southern Arterial Road axis (x = -356.0)
    [-356.0, 4.8, 256.0, 20.5, 4], // 20: Collinear tree-free road alignment for West Underpass Ramp
    [-356.0, 2.85, 274.0, 21.8, 2],// 21: Diving into West Open-Cut Ramp (Pulau U)
    [-356.0, -2.5, 302.0, 23.8, 2],// 22: Descending sunken retaining-wall trench
    [-356.0, -6.12, 330.0, 25.6, 2],// 23: Entering -8.20m Subsea Tunnel Bore beneath West Portal Lintel!
    [-356.0, -6.12, 415.0, 26.8, 2],// 24: 96 km/h Subsea Run beneath Selat UI (with live Bus Kuning dodge!)
    [-356.0, -6.12, 525.0, 26.8, 2],// 25: Passing subsea marine-acrylic caustic windows
    [-356.0, -6.12, 620.0, 25.4, 2],// 26: Approaching East Subsea Portal
    [-356.0, -1.8, 648.0, 23.6, 2],// 27: Rocketing up East Open-Cut Ramp onto Pulau I
    [-356.0, 3.1, 672.0, 22.8, 2], // 28: Emerging into Pulau I daylight!
    [-356.0, 7.2, 688.0, 23.5, 4], // 29: Collinear open-air exit climb on SW Pulau I
    // ACT 4: Pulau I — ICAO Runway 36/18 Low-Pass, Gunung Api Crater Rim, & Air Terjun Niagara Mini Dive!
    [-326.0, 9.2, 714.0, 25.8, 0], // 30: Banking onto Landasan Pacu Pulau I (RWY 36/18)
    [-228.0, 8.6, 716.0, 27.0, 0], // 31: 97 km/h full-throttle runway low pass past Apron & ATC Tower
    [-136.0, 23.5, 720.0, 25.2, 0],// 32: Pulling up cleanly above volcanic foothill pines
    [-72.0, 49.5, 722.0, 22.8, 0], // 33: Ascending southern flank of Gunung Api Pulau I
    [-28.0, 68.5, 702.0, 21.5, 0], // 34: Cresting 58.5m Gunung Api Crater Rim over volcanic caldera!
    [-34.0, 52.0, 666.0, 24.5, 4], // 35: Nose-down tip over SW crater lip onto tree-free waterfall sightline
    [-55.0, 26.5, 640.0, 26.8, 4], // 36: Vertical proximity dive down center of Air Terjun Niagara Mini!
    [-58.0, 11.8, 608.0, 25.6, 0], // 37: Pulling out above waterfall plunge pool & Dermaga I
    // ACT 5: Northern Selat UI Ferry Skim, Northern Cape, Kolong Jembatan Teksas Water-Skim, & Tajug Roll Climb!
    [-24.0, 11.5, 536.0, 26.0, 0], // 38: Blasting across Selat UI past Electric Catamaran Ferry
    [68.0, 14.0, 456.0, 25.5, 0],  // 39: Sweeping around NE Cape of Pulau U over open ocean
    [104.0, 10.5, 366.0, 25.2, 0], // 40: Rounding Northern Tip of Right Arm into mouth of Teluk U
    [54.0, 4.8, 252.0, 25.8, 3],   // 41: Dropping low onto turquoise Teluk U lagoon facing Jembatan Teksas arch!
    [-28.0, 1.65, 245.65, 26.6, 3],// 42: Shooting PERPENDICULARLY UNDERNEATH Jembatan Teksas's +7.8m arch!
    [-86.0, 5.2, 226.0, 25.0, 0],  // 43: Skimming inner Teluk U lagoon past Flybridge Mini Yacht
    [-118.0, 14.5, 136.0, 23.8, 0],// 44: Banking NW past VIP Waterfront Helipad & eVTOL Heli Taxi
    [-86.0, 39.5, 32.0, 23.2, 0]   // 45: High clear punch-out Easy-Acro climb back toward 50m Tajug spire!
  ];

  static AVATA_FREESTYLE_WAYPOINTS = CameraDirector.AVATA_ARCHIPELAGO_NODES.map((n) => [n[0], n[1], n[2]]);

  isAvataFPVMode(mode = this.mode) {
    return mode === 'DRONE_AVATA2_FPV';
  }

  _classifyAvataZoneLabel(x, y, z) {
    if (z >= 270.0 && z <= 672.0 && Math.abs(x - -356.0) <= 6.2 && y < 3.5) {
      return z >= 326.0 && z <= 624.0 ? 'UNDERPASS BAWAH LAUT' : 'RAMP UNDERPASS LAUT';
    }
    if (z >= 149.0 && z <= 342.0 && Math.abs(x - -28.0) <= 3.6) {
      const bD = this.physicsContext?.bridgeDeckFn ? this.physicsContext.bridgeDeckFn(-28.0, z) : null;
      if (bD !== null && y >= bD - 0.2 && y <= bD + 3.4) {
        return 'LORONG JEMBATAN TEKSAS';
      }
      if (bD !== null && y < bD - 0.5) {
        return 'KOLONG JEMBATAN TEKSAS';
      }
    }
    if (z > 590.0) {
      if (Math.hypot(x - -32.0, z - 698.0) < 58.0 && y > 32.0) return 'KAWAH GUNUNG API I';
      if (Math.hypot(x - -44.0, z - 643.0) < 42.0) return 'AIR TERJUN NIAGARA MINI';
      if (Math.abs(z - 708.0) < 28.0 && x >= -380.0 && x <= -130.0) return 'LANDASAN PACU PULAU I';
      return 'PULAU I · BANDARA UI';
    }
    if (z > 465.0 && z <= 590.0) return 'SELAT UI & DERMAGA';
    if (Math.hypot(x - -255.0, z - 329.0) < 88.0) return 'KOMPLEKS DPPD · STP UI';
    if (Math.hypot(x - -104.0, z - 106.5) < 42.0) return 'HELIPAD VIP PULAU U';
    if (Math.hypot(x - -2.5, z - 106.5) < 42.0) return 'AIR MANCUR ROTUNDA UI';
    if (Math.hypot(x - 28.4, z - 78.2) < 30.0) return 'VIDEOTRON YOUTUBE UI';
    if (Math.hypot(x, z) < 62.0 && y > 28.0) return 'PUNCAK TAJUG REKTORAT';
    if (z >= 145.0 && z <= 335.0 && x > -135.0) return 'TELUK U · KAMPUS DEPOK';
    return 'PULAU U · REKTORAT UI';
  }

  setAvataStickInput(partial = {}) {
    if (!partial || typeof partial !== 'object') return;
    if (Number.isFinite(partial.moveX)) this._avataStick.moveX = Math.max(-1, Math.min(1, partial.moveX));
    if (Number.isFinite(partial.moveY)) this._avataStick.moveY = Math.max(-1, Math.min(1, partial.moveY));
    if (Number.isFinite(partial.lookX)) this._avataStick.lookX = Math.max(-1, Math.min(1, partial.lookX));
    if (Number.isFinite(partial.lookY)) this._avataStick.lookY = Math.max(-1, Math.min(1, partial.lookY));
    if (Number.isFinite(partial.climbY)) this._avataStick.climbY = Math.max(-1, Math.min(1, partial.climbY));
    if (typeof partial.boost === 'boolean') this._avataStick.boost = partial.boost;
  }

  setAvataFlightProfile(profile) {
    this.avataFlightProfile = profile === 'ACRO' ? 'ACRO' : 'SPORT';
    return this.avataFlightProfile;
  }

  setAvataAutoFreestyle(enabled) {
    const next = Boolean(enabled);
    if (next && !this.avataAutoFreestyle) {
      // Find nearest spline parameter across 128 samples so rejoining Auto Freestyle blends with zero teleport
      let bestT = this._avataSplineT;
      let bestDistSq = Infinity;
      const samples = 128;
      for (let i = 0; i < samples; i++) {
        const candT = i / samples;
        this._evalAvataCatmullPoint(candT, this._avataTmpP0);
        const dSq = this._avataTmpP0.distanceToSquared(this.camera.position);
        if (dSq < bestDistSq) {
          bestDistSq = dSq;
          bestT = candT;
        }
      }
      this._avataSplineT = bestT;
      this._avataRejoinPos.copy(this.camera.position);
      this._avataRejoinBlend = 1.0;
    }
    this.avataAutoFreestyle = next;
    return this.avataAutoFreestyle;
  }

  triggerAvataStunt(type = 'ROLL_RIGHT') {
    if (this.mode !== 'DRONE_AVATA2_FPV') return false;
    if (this._avataStunt.active) return false;
    // Block 360° flips/rolls inside enclosed tunnels or covered bridge corridors for physical safety!
    const pos = this.camera?.position;
    if (pos) {
      const inUnderpass =
        pos.z >= 270.0 && pos.z <= 672.0 && Math.abs(pos.x - -356.0) <= 5.2 && pos.y < 2.5;
      const inBridge =
        pos.z >= 149.0 && pos.z <= 342.0 && Math.abs(pos.x - -28.0) <= 3.2 && pos.y < 13.2;
      if (inUnderpass || inBridge) return false;
    }
    const valid =
      type === 'ROLL_LEFT' || type === 'ROLL_RIGHT' || type === 'SPLIT_S' || type === 'FLIP_BACK'
        ? type
        : 'ROLL_RIGHT';
    this._avataStunt.active = true;
    this._avataStunt.type = valid;
    this._avataStunt.elapsed = 0.0;
    this._avataStunt.duration = valid === 'SPLIT_S' ? 0.92 : 0.66;
    if (!this.avataAutoFreestyle) {
      // Real cinewhoop upward collective pop before barrel-roll / flip so altitude stays safe
      this.velocity.y = Math.max(this.velocity.y + 4.2, 3.2);
    }
    return true;
  }

  getAvataTelemetry() {
    const active = this.mode === 'DRONE_AVATA2_FPV';
    const pos = this.camera ? this.camera.position : this._avataTmpP0.set(0, 0, 0);
    const speedMs = this.velocity.length();
    const ctx = this.physicsContext;
    const gFn = ctx?.groundHeightFn || null;
    const oceanY = Number.isFinite(ctx?.oceanWaterY) ? ctx.oceanWaterY : -1.45;
    let surfaceY = Math.max(oceanY, gFn ? gFn(pos.x, pos.z) : 0.0);

    // True local surface AGL inside Underpass Bawah Laut (-8.20m) or Lorong Jembatan Teksas UI
    const uCfg = ctx?.underpassConfig;
    if (
      uCfg &&
      pos.z >= uCfg.zWestStart &&
      pos.z <= uCfg.zEastEnd &&
      Math.abs(pos.x - uCfg.ux) <= uCfg.boreHalfW + 0.5 &&
      pos.y < 2.2
    ) {
      const rY = uCfg.sampleRampY ? uCfg.sampleRampY(uCfg.ux, pos.z) : null;
      if (rY !== null) surfaceY = rY;
    } else if (ctx?.bridgeDeckFn && Math.abs(pos.x - -28.0) <= 2.65) {
      const bY = ctx.bridgeDeckFn(-28.0, pos.z);
      if (bY !== null && pos.y >= bY - 0.2) surfaceY = bY;
    }

    const agl = Math.max(0.0, pos.y - surfaceY);
    const homeDist = Math.hypot(pos.x - -2.5, pos.z - 141.5);
    return {
      active,
      profile: this.avataFlightProfile,
      autoFreestyle: this.avataAutoFreestyle,
      speedMs: +speedMs.toFixed(1),
      speedKmh: Math.round(speedMs * 3.6),
      altitudeAGL: +agl.toFixed(1),
      altitudeASL: +(pos.y + 1.45).toFixed(1),
      homeDistM: Math.round(homeDist),
      pitchDeg: Math.round((this.pitch * 180) / Math.PI),
      rollDeg: Math.round((this.roll * 180) / Math.PI),
      yawDeg: ((Math.round(((-this.yaw * 180) / Math.PI) % 360) + 360) % 360),
      gForce: +this._avataGForce.toFixed(1),
      throttlePct: Math.round(Math.max(0, Math.min(1, this._avataThrottle01)) * 100),
      fovDeg: Math.round(this.camera?.fov || 74),
      turnRate: Math.abs(this._avataYawRateSmooth),
      stuntActive: this._avataStunt.active,
      apasActive: this._avataApasTimer > 0.0,
      zoneLabel: this._classifyAvataZoneLabel(pos.x, pos.y, pos.z),
      lapCount: this._avataLapCount
    };
  }

  isFPSMode(mode = this.mode) {
    return CameraDirector.FPS_MODES.has(mode);
  }

  setFPSAnchorProvider(fn) {
    this.fpsAnchorProvider = typeof fn === 'function' ? fn : null;
  }

  setSharedUniforms(uniforms) {
    this.sharedUniforms = uniforms || null;
  }

  cycleFPSSubject(delta = 1) {
    const step = Number.isFinite(delta) ? Math.trunc(delta) : 1;
    this.fpsSubjectOffset = Math.max(0, (this.fpsSubjectOffset + step + 240) % 240);
    this.fpsYawOffset = 0.0;
    this.fpsPitchOffset = 0.0;
    this._fpsPoseInitialized = false;
  }

  resetFPSHeadLook() {
    this.fpsYawOffset = 0.0;
    this.fpsPitchOffset = 0.0;
  }

  /**
   * Injects full solid-world physics context (staticColliders, getDynamicColliders, underpassConfig,
   * groundHeightFn, walkableHeightFn, coastDistFn, waterSurfaceFn, bridgeDeckFn, bridgeSoffitFn, ferryPierDeckFn).
   */
  setWorldPhysicsContext(ctx) {
    this.physicsContext = ctx || null;
    if (ctx && Array.isArray(ctx.staticColliders)) {
      this.colliders = ctx.staticColliders;
    }
    if (this.camera && this.camera.position) {
      this._prevCamPos.copy(this.camera.position);
      this._hasPrevCamPos = true;
    }
  }

  syncPreviousCameraPosition() {
    if (this.camera && this.camera.position) {
      this._prevCamPos.copy(this.camera.position);
      this._hasPrevCamPos = true;
    }
  }

  get isTransitioning() {
    return this.transitionActive;
  }

  bindKeyboard(targetWindow = typeof window !== 'undefined' ? window : null) {
    if (!targetWindow) return;
    const isEditableTarget = (el) => {
      if (!el || typeof el !== 'object') return false;
      const tag = typeof el.tagName === 'string' ? el.tagName.toUpperCase() : '';
      return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || Boolean(el.isContentEditable);
    };
    targetWindow.addEventListener('keydown', (e) => {
      if (isEditableTarget(e && e.target)) {
        this.keyState.clear();
        return;
      }
      if (this.isFPSMode()) {
        if (e.code === 'BracketRight') {
          this.cycleFPSSubject(1);
        } else if (e.code === 'BracketLeft') {
          this.cycleFPSSubject(-1);
        } else if (e.code === 'KeyR') {
          this.resetFPSHeadLook();
        }
      } else if (this.mode === 'DRONE_AVATA2_FPV' && !e.repeat) {
        if (e.code === 'KeyF') {
          this.setAvataAutoFreestyle(!this.avataAutoFreestyle);
        } else if (e.code === 'KeyM') {
          this.setAvataFlightProfile(this.avataFlightProfile === 'SPORT' ? 'ACRO' : 'SPORT');
        } else if (e.code === 'Digit1') {
          this.triggerAvataStunt('ROLL_LEFT');
        } else if (e.code === 'Digit2') {
          this.triggerAvataStunt('ROLL_RIGHT');
        } else if (e.code === 'Digit3') {
          this.triggerAvataStunt('SPLIT_S');
        }
      }
      this.keyState.add(e.code);
    });
    targetWindow.addEventListener('keyup', (e) => {
      this.keyState.delete(e.code);
    });
  }

  /**
   * Immediately yields control back to the user at the exact in-flight pose (zero jump/snap)
   * whenever the user drags, scrolls the wheel, or presses movement keys mid-flight.
   */
  cancelTransition() {
    if (!this.transitionActive) return;
    this.transitionActive = false;
    this._flightNadirSpec = null;
    if (this.mode === 'DRONE_6DOF' || this.mode === 'DRONE_AVATA2_FPV' || this.mode === 'WALK_SIVITAS') {
      if (this.controls) this.controls.enabled = false;
      this.camera.getWorldDirection(this._flightLookTmp);
      this.yaw = Math.atan2(-this._flightLookTmp.x, -this._flightLookTmp.z);
      this.pitch = Math.asin(Math.max(-0.95, Math.min(0.95, this._flightLookTmp.y)));
      if (this.mode !== 'DRONE_AVATA2_FPV') this.roll = 0.0;
    } else if (this.controls) {
      this.controls.enabled = true;
      this.controls.update();
    }
  }

  _captureCurrentLookTarget(outVec) {
    if (this.controls && this.controls.enabled && this.controls.target) {
      return outVec.copy(this.controls.target);
    }
    this.camera.getWorldDirection(this._flightLookTmp);
    return outVec.copy(this.camera.position).addScaledVector(this._flightLookTmp, 36.0);
  }

  _beginGoogleEarthFlight(goalPos, goalTarget, nadirSpec = null) {
    this._captureCurrentLookTarget(this.flightStartTarget);
    this.flightStartPos.copy(this.camera.position);
    this.goalPos.copy(goalPos);
    this.goalTarget.copy(goalTarget);
    this._flightNadirSpec = nadirSpec;

    if (this.controls && this.controls.target) {
      this.controls.target.copy(this.flightStartTarget);
    }

    const posDist = this.flightStartPos.distanceTo(this.goalPos);
    const horizTargetDist = Math.hypot(
      this.goalTarget.x - this.flightStartTarget.x,
      this.goalTarget.z - this.flightStartTarget.z
    );
    const effectiveDist = Math.max(posDist, horizTargetDist);

    if (effectiveDist < 0.12) {
      this.camera.position.copy(this.goalPos);
      if (nadirSpec) {
        setHorizonLockedNadirView(
          this.camera,
          this.controls,
          nadirSpec.targetX,
          nadirSpec.targetZ,
          nadirSpec.heightY,
          nadirSpec.topVecX,
          nadirSpec.topVecZ
        );
      } else if (this.controls) {
        this.controls.target.copy(this.goalTarget);
        this.controls.update();
      }
      this.transitionActive = false;
      this.syncPreviousCameraPosition();
      this._updateAltitudeAdaptiveNearPlane();
      return;
    }

    // Distance-adaptive Google Earth flight time: ~1.48s nearby campus -> ~3.05s cross-archipelago
    this.flightDuration = Math.min(
      3.15,
      Math.max(1.45, 1.32 + Math.log1p(effectiveDist * 0.022) * 0.58)
    );

    // Parabolic mid-flight zoom-out altitude arc: gracefully clears 50m Tajug, forest canopies, & Selat UI!
    // Suppress upward parabolic lift when flying between two subterranean underpass viewpoints (y < -2.2m).
    if (this.flightStartPos.y < -2.2 && this.goalPos.y < -2.2) {
      this.flightArcLiftY = 0.0;
    } else {
      const maxEndpointY = Math.max(this.flightStartPos.y, this.goalPos.y);
      const meanEndpointY = 0.5 * (this.flightStartPos.y + this.goalPos.y);
      const desiredPeakY = Math.max(
        maxEndpointY + Math.min(18.0, horizTargetDist * 0.10),
        Math.min(
          435.0,
          Math.max(this.flightStartTarget.y, this.goalTarget.y) +
            horizTargetDist * 0.42 +
            posDist * 0.12
        )
      );
      this.flightArcLiftY = Math.max(0.0, desiredPeakY - meanEndpointY);
    }
    this.flightElapsed = 0.0;
    this.transitionActive = true;
  }

  _updateAltitudeAdaptiveNearPlane() {
    if (!this.camera || typeof this.camera.updateProjectionMatrix !== 'function') return;
    // Dynamically raise near clip at aerial altitudes (y > 35m) to boost 24-bit depth resolution up to 15.5x!
    const camY = Math.max(0.0, Number(this.camera.position.y) || 0.0);
    const desiredNear = Math.min(2.8, Math.max(this._baseNear, this._baseNear + Math.max(0.0, camY - 35.0) * 0.0036));
    if (Math.abs(this.camera.near - desiredNear) > 0.025) {
      this.camera.near = desiredNear;
      this.camera.updateProjectionMatrix();
    }
  }

  _evalMavicIslandOrbit(angle, floorHeightFn, outPos, outTarget) {
    // Mavic 3 Pro Island-Circling Aerial Trajectory around Pulau U & Teluk U
    // Tours Gedung Rektorat UI, Lapangan Rotunda, Air Mancur Dancing, Videotron, Danau Kenanga,
    // Jembatan Teksas UI, Kompleks DPPD-ILRC-STP, Helipad VIP, & Hutan Selatan
    const cx = -68.0;
    const cz = 148.0;
    const rx = 162.0 + Math.cos(angle * 2.0) * 14.0;
    const rz = 182.0 + Math.sin(angle * 2.0) * 14.0;
    const px = cx + Math.sin(angle) * rx;
    const pz = cz + Math.cos(angle) * rz;
    const gFn = floorHeightFn || this.physicsContext?.groundHeightFn || null;
    const groundY = gFn ? Math.max(0.0, gFn(px, pz)) : 0.0;
    const py = Math.max(55.5, groundY + 38.0 + Math.sin(angle * 1.4) * 9.0);
    outPos.set(px, py, pz);

    const tx = -52.0 + Math.sin(angle + 0.42) * 66.0;
    const tz = 116.0 + Math.cos(angle + 0.42) * 86.0;
    const ty = 14.5 + 4.5 * Math.cos(angle);
    outTarget.set(tx, ty, tz);
  }

  _applyBaseVerticalFov(baseVFov = 48.0) {
    if (!this.camera || typeof this.camera.updateProjectionMatrix !== 'function') return;
    const aspect = Number(this.camera.aspect) || 1.6;
    const maxPortraitFov = baseVFov > 60 ? 96 : 78;
    let nextFov = baseVFov;
    if (aspect < 1.0) {
      const rad = (baseVFov * Math.PI) / 180;
      const hFov = 2 * Math.atan(Math.tan(rad / 2) / Math.max(0.52, aspect));
      nextFov = Math.min(maxPortraitFov, (hFov * 180) / Math.PI);
    }
    if (Math.abs(this.camera.fov - nextFov) > 0.04) {
      this.camera.fov = nextFov;
      this.camera.updateProjectionMatrix();
    }
  }

  _evalAvataCatmullPoint(normT, outVec, outMeta = null) {
    const nodes = CameraDirector.AVATA_ARCHIPELAGO_NODES;
    const n = nodes.length;
    const wrapped = ((normT % 1.0) + 1.0) % 1.0;
    const scaled = wrapped * n;
    const idx = Math.floor(scaled) % n;
    const u = scaled - Math.floor(scaled);
    const p0 = nodes[(idx - 1 + n) % n];
    const p1 = nodes[idx];
    const p2 = nodes[(idx + 1) % n];
    const p3 = nodes[(idx + 2) % n];
    const u2 = u * u;
    const u3 = u2 * u;
    const c0 = -0.5 * u3 + u2 - 0.5 * u;
    const c1 = 1.5 * u3 - 2.5 * u2 + 1.0;
    const c2 = -1.5 * u3 + 2.0 * u2 + 0.5 * u;
    const c3 = 0.5 * u3 - 0.5 * u2;

    let px = c0 * p0[0] + c1 * p1[0] + c2 * p2[0] + c3 * p3[0];
    let py = c0 * p0[1] + c1 * p1[1] + c2 * p2[1] + c3 * p3[1];
    let pz = c0 * p0[2] + c1 * p1[2] + c2 * p2[2] + c3 * p3[2];

    const sSmooth = u * u * (3.0 - 2.0 * u);
    const speedMs = p1[3] + (p2[3] - p1[3]) * sSmooth;
    const c1Lock = p1[4] > 0 ? 1.0 : 0.0;
    const c2Lock = p2[4] > 0 ? 1.0 : 0.0;
    const corridorWeight = c1Lock + (c2Lock - c1Lock) * sSmooth;
    const domType = u < 0.5 ? p1[4] : p2[4];

    const wNodeBridge = (p1[4] === 1 ? 1.0 - sSmooth : 0.0) + (p2[4] === 1 ? sSmooth : 0.0);
    const wNodeUnder = (p1[4] === 2 ? 1.0 - sSmooth : 0.0) + (p2[4] === 2 ? sSmooth : 0.0);
    const wArchSkim = (p1[4] === 3 ? 1.0 - sSmooth : 0.0) + (p2[4] === 3 ? sSmooth : 0.0);

    const zBridgeIn = PhysicsKernel.smootherstep(145.5, 152.0, pz) * (1.0 - PhysicsKernel.smootherstep(336.0, 344.5, pz));
    const wBridge = wNodeBridge * zBridgeIn;

    const zUnderIn = PhysicsKernel.smootherstep(264.0, 274.0, pz) * (1.0 - PhysicsKernel.smootherstep(672.0, 681.0, pz));
    const wUnderpass = wNodeUnder * zUnderIn;

    const ctx = this.physicsContext;
    if (corridorWeight < 0.98) {
      // Non-monotonous multi-lap dynamic freestyle S-carve harmonics strictly in open airspace!
      // seamFade = sin(pi * wrapped) guarantees 0.000m jump across the lap boundary when _avataLapCount increments!
      const seamFade = Math.sin(Math.PI * wrapped);
      const openGain = (1.0 - corridorWeight) * (1.0 - corridorWeight) * (0.25 + 0.75 * seamFade);
      const tx = p2[0] - p1[0];
      const tz = p2[2] - p1[2];
      const tHoriz = Math.hypot(tx, tz) || 1.0;
      const rightX = -tz / tHoriz;
      const rightZ = tx / tHoriz;
      const lapSeed = this._avataLapCount * 2.399963; // Golden-angle lap variation!
      const baseWave = 1.65 * Math.sin(wrapped * Math.PI * 16.0);
      const lapWave =
        seamFade *
        (1.65 * Math.sin(wrapped * Math.PI * 16.0 + lapSeed) +
          1.25 * Math.cos(wrapped * Math.PI * 26.0 - lapSeed * 1.618));
      const sCarve = openGain * (baseWave + lapWave);
      const vWave = openGain * seamFade * 1.05 * Math.sin(wrapped * Math.PI * 12.0 + lapSeed * 1.3);
      px += rightX * sCarve;
      pz += rightZ * sCarve;
      py += vWave;
    }

    const oceanY = Number.isFinite(ctx?.oceanWaterY) ? ctx.oceanWaterY : -1.45;
    const strictFactor = Math.max(wBridge, wUnderpass);
    const openFactor = Math.max(0.0, 1.0 - Math.max(strictFactor, wArchSkim));

    if (wBridge > 1e-4) {
      // C1-Continuous Analytical Lock inside Lorong Jembatan Teksas UI (+2.12m above parabolic deck)
      const zClB = Math.max(149.65, Math.min(341.65, pz));
      const targetBx = -28.0 + this._avataLiveLaneOffset * wBridge;
      px = px * (1.0 - wBridge) + targetBx * wBridge;
      const bD = ctx?.bridgeDeckFn ? ctx.bridgeDeckFn(-28.0, zClB) : null;
      if (bD !== null) {
        const roofLock = PhysicsKernel.smootherstep(168.0, 173.5, zClB) * (1.0 - PhysicsKernel.smootherstep(317.8, 323.0, zClB));
        const targetCorrY = (bD + 2.12) * roofLock + Math.max(py, bD + 2.10) * (1.0 - roofLock);
        py = py * (1.0 - wBridge) + targetCorrY * wBridge;
      }
    } else if (wUnderpass > 1e-4) {
      // C1-Continuous Analytical Lock inside Underpass Bawah Laut Selatan U–I (-6.12m beneath Selat UI seabed)
      const zClU = Math.max(272.0, Math.min(670.0, pz));
      const targetUx = -356.0 + this._avataLiveLaneOffset * wUnderpass;
      px = px * (1.0 - wUnderpass) + targetUx * wUnderpass;
      const uCfg = ctx?.underpassConfig;
      const rY = uCfg?.sampleRampY ? uCfg.sampleRampY(-356.0, zClU) : null;
      if (rY !== null) {
        const targetSubY = rY + 2.08;
        py = py * (1.0 - wUnderpass) + targetSubY * wUnderpass;
      }
    }

    if (wArchSkim > 1e-3) {
      // Smooth C1 water-skim window UNDERNEATH Jembatan Teksas parabolic arch over Teluk U!
      const bSof = ctx?.bridgeSoffitFn ? ctx.bridgeSoffitFn(pz) : null;
      const maxArchY = bSof !== null ? bSof - 1.65 : 5.85;
      const clampedArchY = Math.max(oceanY + 2.45, Math.min(maxArchY, py));
      py += (clampedArchY - py) * wArchSkim;
    }

    if (openFactor > 1e-3 && ctx?.groundHeightFn) {
      // Smooth terrain/ocean floor inside spline evaluator so P0/P1/P2 tangents & arc-lengths match 100%!
      const spGround = ctx.groundHeightFn(px, pz);
      const minSplineY = Math.max(oceanY + 2.15, spGround + 2.65);
      if (py < minSplineY) {
        py += (minSplineY - py) * openFactor;
      }
    }

    if (outMeta) {
      outMeta.speedMs = speedMs;
      outMeta.corridorWeight = corridorWeight;
      outMeta.corridorType = domType;
      outMeta.wBridge = wBridge;
      outMeta.wUnderpass = wUnderpass;
      outMeta.wArchSkim = wArchSkim;
    }
    return outVec.set(px, py, pz);
  }

  /**
   * Predictive Omnidirectional Binocular-Fisheye APAS (Advanced Pilot Assistance System) for ALL Drones
   * (DJI Avata 2 FPV & Mavic 3 Pro 6-DOF). Smoothly deflects velocity around static/dynamic 3D obstacles
   * and funnels cleanly into clear bridge/underpass portals before hard MTV separation even triggers!
   */
  _applyDroneAPASAvoidance(pos, vel, dt, bodyRadius, ctx, commandedSpeed = 0.0, preferUpVault = true) {
    if (!ctx || !vel) return false;
    const speed = vel.length();
    const refSpeed = Math.max(speed, commandedSpeed);
    if (refSpeed < 0.25) return false;

    let dirX = 0.0;
    let dirY = 0.0;
    let dirZ = -1.0;
    if (speed > 0.25) {
      const invSpd = 1.0 / speed;
      dirX = vel.x * invSpd;
      dirY = vel.y * invSpd;
      dirZ = vel.z * invSpd;
    } else {
      const cosP = Math.cos(this.pitch);
      dirX = -Math.sin(this.yaw) * cosP;
      dirY = Math.sin(this.pitch);
      dirZ = -Math.cos(this.yaw) * cosP;
    }

    let deflected = false;

    // 1. Smart Portal Center-Line Assist when aiming manually into Jembatan Teksas or Underpass Bawah Laut
    const insideEnclosedRoof =
      (pos.z >= 169.0 &&
        pos.z <= 322.0 &&
        Math.abs(pos.x - -28.0) < 2.9 &&
        pos.y > 4.8 &&
        pos.y < 12.6) ||
      (pos.z >= 324.0 &&
        pos.z <= 626.0 &&
        Math.abs(pos.x - -356.0) < 5.0 &&
        pos.y < -3.2);

    if (
      pos.z >= 144.0 &&
      pos.z <= 346.0 &&
      Math.abs(pos.x - -28.0) < 3.6 &&
      Math.abs(dirZ) > 0.65
    ) {
      const bD = ctx.bridgeDeckFn ? ctx.bridgeDeckFn(-28.0, Math.max(149.65, Math.min(341.65, pos.z))) : null;
      if (bD !== null && pos.y >= bD - 0.1 && pos.y <= bD + 3.35) {
        const errX = -28.0 - pos.x;
        const targetY = bD + 1.95;
        const errY = targetY - pos.y;
        vel.x += errX * 6.5 * dt;
        if (pos.y > bD + 2.35 || pos.y < bD + 0.95) {
          vel.y += errY * 6.0 * dt;
        }
      }
    } else if (
      pos.z >= 266.0 &&
      pos.z <= 676.0 &&
      Math.abs(pos.x - -356.0) < 5.2 &&
      pos.y < 3.2 &&
      Math.abs(dirZ) > 0.65
    ) {
      const uCfg = ctx.underpassConfig;
      const rY = uCfg?.sampleRampY ? uCfg.sampleRampY(-356.0, Math.max(272.0, Math.min(670.0, pos.z))) : null;
      if (rY !== null && pos.y <= rY + 4.2) {
        const errX = -356.0 - pos.x;
        if (Math.abs(errX) > 2.6) vel.x += errX * 5.0 * dt;
        const safeY = rY + 2.05;
        if (pos.y > rY + 3.25 || pos.y < rY + 1.0) {
          vel.y += (safeY - pos.y) * 5.8 * dt;
        }
      }
    }

    // 2. Three-Horizon Predictive Sphere Probes (Contact Halo 0.0s + Near 0.16s + Mid 0.32s)
    const rep = this._avataRepelScratch;
    let nx = 0.0;
    let ny = 0.0;
    let nz = 0.0;
    let urgency = 0.0;

    const penContact = PhysicsKernel.probeDroneAPASRepulsion(
      pos.x + dirX * 0.28,
      pos.y + dirY * 0.28,
      pos.z + dirZ * 0.28,
      pos.x,
      pos.y,
      pos.z,
      bodyRadius + 0.42,
      ctx,
      rep
    );
    if (penContact > 0.0) {
      nx = rep.nx;
      ny = rep.ny;
      nz = rep.nz;
      urgency = Math.min(1.0, 0.72 + penContact * 0.95);
    }

    if (urgency < 0.75) {
      const dNear = Math.min(4.8, 0.55 + refSpeed * 0.16);
      const penNear = PhysicsKernel.probeDroneAPASRepulsion(
        pos.x + dirX * dNear,
        pos.y + dirY * dNear,
        pos.z + dirZ * dNear,
        pos.x,
        pos.y,
        pos.z,
        bodyRadius + 0.48,
        ctx,
        rep
      );
      if (penNear > 0.0) {
        nx = rep.nx;
        ny = rep.ny;
        nz = rep.nz;
        urgency = Math.max(urgency, Math.min(1.0, 0.48 + penNear * 0.85));
      }
    }

    if (urgency < 0.35 && refSpeed > 4.0) {
      const dFar = Math.min(9.6, 1.35 + refSpeed * 0.32);
      const penFar = PhysicsKernel.probeDroneAPASRepulsion(
        pos.x + dirX * dFar,
        pos.y + dirY * dFar,
        pos.z + dirZ * dFar,
        pos.x,
        pos.y,
        pos.z,
        bodyRadius + 0.72,
        ctx,
        rep
      );
      if (penFar > 0.0) {
        nx = rep.nx;
        ny = rep.ny;
        nz = rep.nz;
        urgency = Math.min(0.78, 0.25 + penFar * 0.45);
      }
    }

    if (urgency > 0.0) {
      const dDotN = dirX * nx + dirY * ny + dirZ * nz;
      const vDotN = vel.x * nx + vel.y * ny + vel.z * nz;
      if (vDotN < 0.0 || dDotN < -0.05) {
        // Cancel inward closing velocity smoothly & steer tangentially around/over obstacle without EVER stalling!
        if (vDotN < 0.0) {
          const dampIn = Math.min(1.0, urgency * 15.0 * dt);
          vel.x -= vDotN * nx * dampIn;
          vel.y -= vDotN * ny * dampIn;
          vel.z -= vDotN * nz * dampIn;
        }

        // Tangential bypass vector derived from intended travel direction so even a dead-center hit banks/vaults past!
        let tx = dirX - dDotN * nx;
        let ty = dirY - dDotN * ny;
        let tz = dirZ - dDotN * nz;
        const hTang = Math.hypot(tx, tz);
        if (hTang < 0.42 && Math.abs(ny) < 0.88) {
          // Dead-on horizontal approach onto trunk, column, crown, or wall: inject smooth lateral bank bypass
          const latScale = 0.82 - hTang;
          tx += -nz * latScale;
          tz += nx * latScale;
        }
        if (preferUpVault && !insideEnclosedRoof && ny > -0.15) {
          ty += ny > 0.12 ? 0.68 : 0.34;
        }
        let tLen = Math.hypot(tx, ty, tz);
        if (tLen < 0.25) {
          tx = -nz;
          ty = !insideEnclosedRoof && Math.abs(ny) < 0.85 ? 0.55 : 0.0;
          tz = nx;
          tLen = Math.hypot(tx, ty, tz) || 1.0;
        }
        const bypassAccel = urgency * Math.max(18.5, refSpeed * 1.05);
        vel.x += (tx / tLen + nx * 0.42) * bypassAccel * dt;
        vel.y += (ty / tLen + ny * 0.42) * bypassAccel * dt;
        vel.z += (tz / tLen + nz * 0.42) * bypassAccel * dt;
        deflected = true;
      }
    }

    return deflected;
  }

  setMode(newMode, options = {}) {
    const prevMode = this.mode;
    const wasFPS = this.isFPSMode(prevMode);
    const nextIsFPS = this.isFPSMode(newMode);

    // Capture current world view direction BEFORE resetting camera.up so leaving FPS/Drone/Walk preserves look aim
    if (
      (wasFPS ||
        prevMode === 'DRONE_6DOF' ||
        prevMode === 'DRONE_AVATA2_FPV' ||
        prevMode === 'WALK_SIVITAS') &&
      !nextIsFPS
    ) {
      this.camera.getWorldDirection(this._flightLookTmp);
    }

    if (prevMode === 'DRONE_AVATA2_FPV' && newMode !== 'DRONE_AVATA2_FPV') {
      this._avataStick.moveX = 0;
      this._avataStick.moveY = 0;
      this._avataStick.lookX = 0;
      this._avataStick.lookY = 0;
      this._avataStick.climbY = 0;
      this._avataStick.boost = false;
      this._avataStunt.active = false;
      this._avataCurrentBaseFov = 48.0;
      this._applyBaseVerticalFov(48.0);
    }

    if (wasFPS && !nextIsFPS) {
      if (this.sharedUniforms && this.sharedUniforms.uCockpitGlassCutRadius) {
        this.sharedUniforms.uCockpitGlassCutRadius.value = 0.0;
      }
      this.lastFPSStatus = null;
      this._fpsPoseInitialized = false;
    } else if (prevMode !== newMode) {
      this.lastFPSStatus = null;
    }

    this.mode = newMode;
    this.transitionActive = false;
    this._flightNadirSpec = null;
    this.camera.up.set(0, 1, 0);
    this._orbitPanVel.set(0, 0, 0);

    if (nextIsFPS) {
      if (this.controls) this.controls.enabled = false;
      this.velocity.set(0, 0, 0);
      this.fpsYawOffset = 0.0;
      this.fpsPitchOffset = 0.0;
      if (options.subjectOffset !== undefined) {
        this.fpsSubjectOffset = Math.max(0, Math.floor(Number(options.subjectOffset) || 0));
      } else if (prevMode !== newMode) {
        this.fpsSubjectOffset = 0;
      }
      this._fpsPoseInitialized = false;
      if (this.camera && typeof this.camera.updateProjectionMatrix === 'function') {
        this.camera.near = 0.08;
        this.camera.updateProjectionMatrix();
      }
      return;
    }

    if (newMode === 'ORBIT' || newMode === 'AUTO_CINEMATIC') {
      this.roll = 0;
      if (newMode === 'AUTO_CINEMATIC' && this.camera.position.y < 38.0) {
        this._evalMavicIslandOrbit(this.cinematicAngle, null, this.camera.position, this.goalTarget);
        this.syncPreviousCameraPosition();
      }
      if (this.controls) {
        if (newMode === 'AUTO_CINEMATIC') {
          this.controls.target.copy(this.goalTarget);
        } else if (
          wasFPS ||
          prevMode === 'DRONE_6DOF' ||
          prevMode === 'DRONE_AVATA2_FPV' ||
          prevMode === 'WALK_SIVITAS'
        ) {
          if (wasFPS && this.physicsContext && this.physicsContext.walkableHeightFn) {
            const minStandY =
              this.physicsContext.walkableHeightFn(this.camera.position.x, this.camera.position.z) + 1.65;
            if (this.camera.position.y < minStandY && this.camera.position.y > -2.0) {
              this.camera.position.y = minStandY;
            }
          }
          this.controls.target.copy(this.camera.position).addScaledVector(this._flightLookTmp, 28.0);
        }
        this.controls.enabled = true;
      }
    } else if (newMode === 'DRONE_6DOF') {
      if (this.controls) this.controls.enabled = false;
      if (options.startIslandTour === true) {
        this._evalMavicIslandOrbit(this.cinematicAngle, null, this.camera.position, this.goalTarget);
        const lookDx = this.goalTarget.x - this.camera.position.x;
        const lookDz = this.goalTarget.z - this.camera.position.z;
        this.yaw = Math.atan2(-lookDx, -lookDz);
        this.pitch = Math.atan2(this.goalTarget.y - this.camera.position.y, Math.hypot(lookDx, lookDz));
        this.roll = 0.035;
        const euler = new THREE.Euler(this.pitch, this.yaw, this.roll, 'YXZ');
        this.camera.quaternion.setFromEuler(euler);
        this.syncPreviousCameraPosition();
      } else {
        const dir = new THREE.Vector3();
        this.camera.getWorldDirection(dir);
        this.yaw = Math.atan2(-dir.x, -dir.z);
        this.pitch = Math.asin(Math.max(-0.95, Math.min(0.95, dir.y)));
      }
      this.velocity.set(0, 0, 0);
      this._droneManualEngaged = false;
      this._mavicGimbalYaw = 0.0;
      this._mavicGimbalPitch = 0.0;
    } else if (newMode === 'DRONE_AVATA2_FPV') {
      if (this.controls) this.controls.enabled = false;
      this._avataYawRateAccum = 0.0;
      this._avataYawRateSmooth = 0.0;
      this._avataGForce = 1.0;
      this._avataStunt.active = false;
      this._avataCurrentBaseFov = 74.0;
      this._applyBaseVerticalFov(74.0);
      if (options.profile === 'ACRO' || options.profile === 'SPORT') {
        this.avataFlightProfile = options.profile;
      }
      if (options.freeFlight === true) {
        this.avataAutoFreestyle = false;
        this._avataRejoinBlend = 0.0;
        const dir = this._flightLookTmp;
        if (dir.lengthSq() < 0.2) this.camera.getWorldDirection(dir);
        this.yaw = Math.atan2(-dir.x, -dir.z);
        this.pitch = Math.asin(Math.max(-0.95, Math.min(0.95, dir.y)));
        this.roll = 0.0;
        if (this.camera.position.y > 160.0) {
          this.camera.position.set(-18.0, 26.0, 82.0);
          this.yaw = Math.atan2(18.0, 82.0);
          this.pitch = -0.18;
        }
        this.velocity.set(0, 0, 0);
        this._avataEuler.set(this.pitch, this.yaw, this.roll, 'YXZ');
        this.camera.quaternion.setFromEuler(this._avataEuler);
        this.syncPreviousCameraPosition();
      } else {
        this.avataAutoFreestyle = true;
        this._avataSplineT = Number.isFinite(options.startT) ? options.startT : 0.014;
        this._avataLiveLaneOffset = 0.0;
        this._avataRejoinBlend = 0.0;
        this._evalAvataCatmullPoint(this._avataSplineT, this.camera.position, this._avataMeta);
        this.goalPos.copy(this.camera.position);
        this._evalAvataCatmullPoint(this._avataSplineT + 0.0012, this._avataTmpP1);
        const dx = this._avataTmpP1.x - this.camera.position.x;
        const dy = this._avataTmpP1.y - this.camera.position.y;
        const dz = this._avataTmpP1.z - this.camera.position.z;
        const hLen = Math.hypot(dx, dz) || 1.0;
        this.yaw = Math.atan2(-dx, -dz);
        this.pitch = Math.atan2(dy, hLen);
        this.roll = -0.28;
        const initDirLen = Math.hypot(dx, dy, dz) || 1.0;
        const initSpd = this._avataMeta.speedMs || 23.0;
        this.velocity.set((dx / initDirLen) * initSpd, (dy / initDirLen) * initSpd, (dz / initDirLen) * initSpd);
        this._avataVelPrev.copy(this.velocity);
        this._avataEuler.set(this.pitch, this.yaw, this.roll, 'YXZ');
        this.camera.quaternion.setFromEuler(this._avataEuler);
        this.camera.up.set(0, 1, 0).applyQuaternion(this.camera.quaternion);
        this.syncPreviousCameraPosition();
      }
      if (this.camera && typeof this.camera.updateProjectionMatrix === 'function') {
        this.camera.near = 0.1;
        this.camera.updateProjectionMatrix();
      }
      return;
    } else if (newMode === 'WALK_SIVITAS') {
      if (this.controls) this.controls.enabled = false;
      const walkPos = new THREE.Vector3(-2.5, 1.72, 74.5);
      const walkLook = new THREE.Vector3(-2.5, 6.8, 38.5);
      this.yaw = 0.0;
      this.pitch = 0.14;
      this.roll = 0.0;
      this.velocity.set(0, 0, 0);

      if (options.immediate === true || this.camera.position.distanceTo(walkPos) < 3.5) {
        this.camera.position.copy(walkPos);
        this.syncPreviousCameraPosition();
      } else {
        // Smooth Google Earth swoop down to Sivitas pedestrian eye-level instead of an instant teleport!
        this._beginGoogleEarthFlight(walkPos, walkLook, null);
      }
    }
    this._updateAltitudeAdaptiveNearPlane();
  }

  applyOrthoPreset(presetName, presets, options = {}) {
    if (!presets || !presets[presetName]) return;
    this.setMode('ORBIT');
    const p = presets[presetName];

    if (presetName === 'nadir90') {
      const targetX = p.targetX ?? 0.0;
      const targetZ = p.targetZ ?? 0.0;
      const heightY = p.heightY ?? 142.0;
      const topVecX = p.topVecX ?? 0.0;
      const topVecZ = p.topVecZ ?? -1.0;

      if (options.immediate === true) {
        this.transitionActive = false;
        setHorizonLockedNadirView(this.camera, this.controls, targetX, targetZ, heightY, topVecX, topVecZ);
        this.syncPreviousCameraPosition();
        this._updateAltitudeAdaptiveNearPlane();
        return;
      }

      const len = Math.hypot(topVecX, topVecZ) || 1.0;
      const eps = 0.45;
      const nadirPos = new THREE.Vector3(
        targetX - (topVecX / len) * eps,
        heightY,
        targetZ - (topVecZ / len) * eps
      );
      const nadirTarget = new THREE.Vector3(targetX, 0.0, targetZ);
      this._beginGoogleEarthFlight(nadirPos, nadirTarget, {
        targetX,
        targetZ,
        heightY,
        topVecX,
        topVecZ
      });
      return;
    }

    const nextPos = new THREE.Vector3(p.pos.x, p.pos.y, p.pos.z);
    const nextTarget = new THREE.Vector3(p.target.x, p.target.y, p.target.z);
    if (options.immediate === true) {
      this.camera.position.copy(nextPos);
      if (this.controls) {
        this.controls.target.copy(nextTarget);
        this.controls.update();
      }
      this.transitionActive = false;
      this.syncPreviousCameraPosition();
      this._updateAltitudeAdaptiveNearPlane();
      return;
    }

    this._beginGoogleEarthFlight(nextPos, nextTarget, null);
  }

  lookDelta(dx, dy, sensitivity = 0.0024) {
    if (this.transitionActive && (Math.abs(dx) > 1.5 || Math.abs(dy) > 1.5)) {
      this.cancelTransition();
    }
    if (this.isFPSMode()) {
      this.fpsYawOffset = Math.max(-2.05, Math.min(2.05, this.fpsYawOffset - dx * sensitivity));
      this.fpsPitchOffset = Math.max(-1.12, Math.min(1.12, this.fpsPitchOffset - dy * sensitivity));
      return;
    }
    if (this.mode === 'DRONE_AVATA2_FPV') {
      if (this.avataAutoFreestyle && Math.hypot(dx, dy) > 2.5) {
        this.avataAutoFreestyle = false;
        this._avataRejoinBlend = 0.0;
      }
      const dYaw = -dx * sensitivity * 1.12;
      this.yaw += dYaw;
      this._avataYawRateAccum += dYaw;
      const maxDownPitch = this.avataFlightProfile === 'ACRO' ? -1.48 : -1.32;
      const maxUpPitch = this.avataFlightProfile === 'ACRO' ? 1.38 : 1.18;
      this.pitch = Math.max(maxDownPitch, Math.min(maxUpPitch, this.pitch - dy * sensitivity * 1.08));
      return;
    }
    if (this.mode !== 'DRONE_6DOF' && this.mode !== 'WALK_SIVITAS') return;
    if (this.mode === 'DRONE_6DOF' && !this._droneManualEngaged) {
      this._mavicGimbalYaw = Math.max(-2.6, Math.min(2.6, this._mavicGimbalYaw - dx * sensitivity));
      this._mavicGimbalPitch = Math.max(-0.95, Math.min(0.85, this._mavicGimbalPitch - dy * sensitivity));
      return;
    }
    this.yaw -= dx * sensitivity;
    this.pitch = Math.max(-1.35, Math.min(1.35, this.pitch - dy * sensitivity));
  }

  _stepGoogleEarthFlight(clampedDt) {
    this.flightElapsed += clampedDt;
    const s = Math.min(1.0, this.flightElapsed / Math.max(0.1, this.flightDuration));
    // C^2-continuous quintic smootherstep (zero velocity & zero acceleration jerk at both takeoff and landing!)
    const u = s * s * s * (s * (s * 6.0 - 15.0) + 10.0);
    const arcSin = Math.sin(Math.PI * u);
    const arcY = this.flightArcLiftY * arcSin * arcSin;

    this.camera.position.lerpVectors(this.flightStartPos, this.goalPos, u);
    this.camera.position.y += arcY;
    this.camera.up.set(0, 1, 0);

    this._flightLookTmp.lerpVectors(this.flightStartTarget, this.goalTarget, u);
    if (this.controls && this.mode === 'ORBIT') {
      this.controls.target.copy(this._flightLookTmp);
      this.controls.update();
    } else {
      this.camera.lookAt(this._flightLookTmp);
    }

    if (s >= 1.0) {
      this.camera.position.copy(this.goalPos);
      if (this._flightNadirSpec) {
        setHorizonLockedNadirView(
          this.camera,
          this.controls,
          this._flightNadirSpec.targetX,
          this._flightNadirSpec.targetZ,
          this._flightNadirSpec.heightY,
          this._flightNadirSpec.topVecX,
          this._flightNadirSpec.topVecZ
        );
      } else if (this.controls && this.mode === 'ORBIT') {
        this.controls.target.copy(this.goalTarget);
        this.controls.update();
      } else if (this.mode === 'WALK_SIVITAS') {
        const euler = new THREE.Euler(this.pitch, this.yaw, 0.0, 'YXZ');
        this.camera.quaternion.setFromEuler(euler);
      }
      this._flightNadirSpec = null;
      this.transitionActive = false;
    }
  }

  update(dt, floorHeightFn = null) {
    const clampedDt = Math.min(0.05, Math.max(0.001, dt));
    const dampAlpha = 1.0 - Math.exp(-7.5 * clampedDt); // Hukum Emas 15: frame-rate independent damping
    if (!this._hasPrevCamPos) {
      this._prevCamPos.copy(this.camera.position);
      this._hasPrevCamPos = true;
    }

    const ctx = this.physicsContext;
    if (ctx && floorHeightFn && !ctx.walkableHeightFn) {
      ctx.walkableHeightFn = floorHeightFn;
    }

    // -------------------------------------------------------------------------
    // 6 NATURAL LIVING CAMPUS FIRST-PERSON (FPS) PERSPECTIVES
    // ('FPS_JOGGER' | 'FPS_BUS_DRIVER' | 'FPS_GOLF_DRIVER' | 'FPS_CHAT_NPC' | 'FPS_CAT' | 'FPS_PILOT')
    // -------------------------------------------------------------------------
    if (this.isFPSMode()) {
      const anchor = this.fpsAnchorProvider
        ? this.fpsAnchorProvider(this.mode, this.fpsSubjectOffset)
        : null;
      if (anchor && anchor.eyePos && anchor.forward && anchor.up && anchor.right) {
        if (this.sharedUniforms && this.sharedUniforms.uCockpitGlassCutRadius) {
          this.sharedUniforms.uCockpitGlassCutRadius.value = anchor.isVehicleCockpit ? 2.45 : 0.0;
        }
        if (this.camera && Math.abs(this.camera.near - 0.08) > 0.005) {
          this.camera.near = 0.08;
          if (typeof this.camera.updateProjectionMatrix === 'function') {
            this.camera.updateProjectionMatrix();
          }
        }

        // Apply interactive head-look yaw (around anchor.up) & pitch (around rotated right)
        const cY = Math.cos(this.fpsYawOffset);
        const sY = Math.sin(this.fpsYawOffset);
        const F = anchor.forward;
        const R = anchor.right;
        const U = anchor.up;

        this._fpsTmpF1.set(F.x * cY - R.x * sY, F.y * cY - R.y * sY, F.z * cY - R.z * sY).normalize();
        this._fpsTmpR1.set(R.x * cY + F.x * sY, R.y * cY + F.y * sY, R.z * cY + F.z * sY).normalize();

        const cP = Math.cos(this.fpsPitchOffset);
        const sP = Math.sin(this.fpsPitchOffset);
        this._fpsTmpF2
          .set(
            this._fpsTmpF1.x * cP + U.x * sP,
            this._fpsTmpF1.y * cP + U.y * sP,
            this._fpsTmpF1.z * cP + U.z * sP
          )
          .normalize();
        this._fpsTmpU2
          .set(
            U.x * cP - this._fpsTmpF1.x * sP,
            U.y * cP - this._fpsTmpF1.y * sP,
            U.z * cP - this._fpsTmpF1.z * sP
          )
          .normalize();

        if (!this._fpsPoseInitialized || anchor.eyePos.distanceTo(this._fpsSmoothPos) > 12.0) {
          this._fpsSmoothPos.copy(anchor.eyePos);
          this._fpsSmoothFwd.copy(this._fpsTmpF2);
          this._fpsSmoothUp.copy(this._fpsTmpU2);
          this._fpsPoseInitialized = true;
        } else if (anchor.isVehicleCockpit) {
          // Lock cockpit seat position 1:1 so high-speed Learjet/Cessna/Bus never lags behind cockpit dashboard
          this._fpsSmoothPos.copy(anchor.eyePos);
          const aRot = 1.0 - Math.exp(-22.0 * clampedDt);
          this._fpsSmoothFwd.lerp(this._fpsTmpF2, aRot).normalize();
          this._fpsSmoothUp.lerp(this._fpsTmpU2, aRot).normalize();
        } else {
          // Organic physiological head stabilization for Joggers, Conversing Sivitas, & Prowling Cats
          const aPos = 1.0 - Math.exp(-24.0 * clampedDt);
          const aRot = 1.0 - Math.exp(-15.0 * clampedDt);
          this._fpsSmoothPos.lerp(anchor.eyePos, aPos);
          this._fpsSmoothFwd.lerp(this._fpsTmpF2, aRot).normalize();
          this._fpsSmoothUp.lerp(this._fpsTmpU2, aRot).normalize();
        }

        this.camera.position.copy(this._fpsSmoothPos);
        this.camera.up.copy(this._fpsSmoothUp);
        this.camera.lookAt(
          this._fpsSmoothPos.x + this._fpsSmoothFwd.x * 10.0,
          this._fpsSmoothPos.y + this._fpsSmoothFwd.y * 10.0,
          this._fpsSmoothPos.z + this._fpsSmoothFwd.z * 10.0
        );
        this.lastFPSStatus = anchor;
      }
      this._prevCamPos.copy(this.camera.position);
      return;
    }

    if (this.mode === 'ORBIT') {
      this.camera.up.set(0, 1, 0);

      // Free Orbit WASD / Arrow / Q-E Movement (cancels active parabolic flight immediately on user key press!)
      let orbF = 0;
      let orbR = 0;
      let orbU = 0;
      if (this.keyState.has('KeyW') || this.keyState.has('ArrowUp')) orbF += 1;
      if (this.keyState.has('KeyS') || this.keyState.has('ArrowDown')) orbF -= 1;
      if (this.keyState.has('KeyD') || this.keyState.has('ArrowRight')) orbR += 1;
      if (this.keyState.has('KeyA') || this.keyState.has('ArrowLeft')) orbR -= 1;
      if (this.keyState.has('KeyE')) orbU += 1;
      if (this.keyState.has('KeyQ')) orbU -= 1;

      const hasOrbitKeys = orbF !== 0 || orbR !== 0 || orbU !== 0;
      if (hasOrbitKeys && this.transitionActive) {
        this.cancelTransition();
      }

      if (this.transitionActive) {
        this._stepGoogleEarthFlight(clampedDt);
        resolveSoftSpatialCollision(
          this.camera.position,
          null,
          this.colliders,
          0.56,
          0.56,
          clampedDt,
          null,
          ctx,
          false
        );
      } else {
        // Integrate smooth damped Free-Orbit WASD pan velocity scaled to current orbit distance
        let fx = 0.0;
        let fz = -1.0;
        if (this.controls && this.controls.target) {
          fx = this.controls.target.x - this.camera.position.x;
          fz = this.controls.target.z - this.camera.position.z;
        }
        let fHoriz = Math.hypot(fx, fz);
        if (fHoriz < 0.75) {
          this.camera.getWorldDirection(this._flightLookTmp);
          fx = this._flightLookTmp.x;
          fz = this._flightLookTmp.z;
          fHoriz = Math.hypot(fx, fz);
          if (fHoriz < 1e-4) {
            fx = 0.0;
            fz = -1.0;
            fHoriz = 1.0;
          }
        }
        fx /= fHoriz;
        fz /= fHoriz;
        const rx = -fz;
        const rz = fx;

        const pivotDist =
          this.controls && this.controls.target
            ? this.camera.position.distanceTo(this.controls.target)
            : 36.0;
        const shiftBoost =
          this.keyState.has('ShiftLeft') || this.keyState.has('ShiftRight') ? 2.15 : 1.0;
        const maxOrbitSpeed = Math.min(145.0, Math.max(14.0, pivotDist * 0.58)) * shiftBoost;
        const orbitAccel = maxOrbitSpeed * 5.4;

        const inLen = Math.hypot(orbF, orbR);
        const nF = inLen > 1.0 ? orbF / inLen : orbF;
        const nR = inLen > 1.0 ? orbR / inLen : orbR;

        this._orbitPanVel.x += (fx * nF + rx * nR) * orbitAccel * clampedDt;
        this._orbitPanVel.z += (fz * nF + rz * nR) * orbitAccel * clampedDt;
        this._orbitPanVel.y += orbU * orbitAccel * 0.65 * clampedDt;

        const orbitDrag = Math.exp(-6.2 * clampedDt);
        this._orbitPanVel.multiplyScalar(orbitDrag);
        if (this._orbitPanVel.length() > maxOrbitSpeed) {
          this._orbitPanVel.setLength(maxOrbitSpeed);
        }
        if (!hasOrbitKeys && this._orbitPanVel.lengthSq() < 1e-4) {
          this._orbitPanVel.set(0, 0, 0);
        }

        const wasOrbitPanning = this._orbitPanVel.lengthSq() > 0.0;
        if (wasOrbitPanning) {
          const dx = this._orbitPanVel.x * clampedDt;
          const dy = this._orbitPanVel.y * clampedDt;
          const dz = this._orbitPanVel.z * clampedDt;
          this.camera.position.x += dx;
          this.camera.position.y += dy;
          this.camera.position.z += dz;
          if (this.controls && this.controls.target) {
            this.controls.target.x += dx;
            this.controls.target.y += dy;
            this.controls.target.z += dz;
          }
        }

        if (this.controls) {
          // Prevent OrbitControls pivot target from being dragged deep underground/underwater (outside Underpass trench)
          if (this.controls.target && ctx) {
            const tx = this.controls.target.x;
            const tz = this.controls.target.z;
            const uCfg = ctx.underpassConfig;
            const targetInUnderpass =
              uCfg &&
              tz >= uCfg.zWestStart - 4.0 &&
              tz <= uCfg.zEastEnd + 4.0 &&
              Math.abs(tx - uCfg.ux) <= uCfg.boreHalfW + 3.0;
            if (!targetInUnderpass) {
              const tGround = ctx.groundHeightFn ? ctx.groundHeightFn(tx, tz) : 0.0;
              const minTargetY = Math.max(
                Number.isFinite(ctx.oceanWaterY) ? ctx.oceanWaterY : -1.45,
                tGround - 0.1
              );
              if (this.controls.target.y < minTargetY) {
                this.controls.target.y = minTargetY;
              }
            }
          }
          this.controls.update();
        }
        const rawOrbitGroundY = floorHeightFn
          ? floorHeightFn(this.camera.position.x, this.camera.position.z)
          : 0.0;
        const orbitFloorY =
          rawOrbitGroundY < -0.5 ? rawOrbitGroundY + 0.56 : Math.max(0.56, rawOrbitGroundY + 0.56);
        const preColX = this.camera.position.x;
        const preColZ = this.camera.position.z;
        resolveSoftSpatialCollision(
          this.camera.position,
          this._orbitPanVel,
          this.colliders,
          0.56,
          orbitFloorY,
          clampedDt,
          this._prevCamPos,
          ctx,
          false
        );
        if (wasOrbitPanning && this.controls && this.controls.target) {
          this.controls.target.x += this.camera.position.x - preColX;
          this.controls.target.z += this.camera.position.z - preColZ;
        }
      }
      this._prevCamPos.copy(this.camera.position);
      this._updateAltitudeAdaptiveNearPlane();
      return;
    }

    if (this.mode === 'AUTO_CINEMATIC') {
      this.camera.up.set(0, 1, 0);
      this.cinematicAngle += clampedDt * 0.115;
      this._evalMavicIslandOrbit(this.cinematicAngle, floorHeightFn, this.goalPos, this.goalTarget);
      this.camera.position.lerp(this.goalPos, dampAlpha);
      if (this.controls) {
        this.controls.target.lerp(this.goalTarget, dampAlpha);
        this.controls.update();
      }
      resolveSoftSpatialCollision(
        this.camera.position,
        null,
        this.colliders,
        1.15,
        2.5,
        clampedDt,
        this._prevCamPos,
        ctx,
        false
      );
      this._prevCamPos.copy(this.camera.position);
      this._updateAltitudeAdaptiveNearPlane();
      return;
    }

    if (this.mode === 'DRONE_AVATA2_FPV') {
      this._updateAvata2FPV(clampedDt, floorHeightFn, ctx);
      return;
    }

    // DRONE_6DOF or WALK_SIVITAS
    if (this.transitionActive) {
      const anyKey =
        this.keyState.has('KeyW') ||
        this.keyState.has('KeyS') ||
        this.keyState.has('KeyA') ||
        this.keyState.has('KeyD') ||
        this.keyState.has('ArrowUp') ||
        this.keyState.has('ArrowDown') ||
        this.keyState.has('ArrowLeft') ||
        this.keyState.has('ArrowRight');
      if (anyKey) {
        this.cancelTransition();
      } else {
        this._stepGoogleEarthFlight(clampedDt);
        resolveSoftSpatialCollision(
          this.camera.position,
          null,
          this.colliders,
          0.68,
          0.68,
          clampedDt,
          null,
          ctx,
          false
        );
        this._prevCamPos.copy(this.camera.position);
        this._updateAltitudeAdaptiveNearPlane();
        return;
      }
    }
    const isWalk = this.mode === 'WALK_SIVITAS';
    const maxSpeed = isWalk ? 6.2 : 28.0;
    const accel = isWalk ? 26.0 : 65.0;

    this._forward.set(-Math.sin(this.yaw), isWalk ? 0 : Math.sin(this.pitch), -Math.cos(this.yaw)).normalize();
    this._right.set(Math.cos(this.yaw), 0, -Math.sin(this.yaw)).normalize();

    let moveF = 0;
    let moveR = 0;
    let moveU = 0;
    if (this.keyState.has('KeyW') || this.keyState.has('ArrowUp')) moveF += 1;
    if (this.keyState.has('KeyS') || this.keyState.has('ArrowDown')) moveF -= 1;
    if (this.keyState.has('KeyD') || this.keyState.has('ArrowRight')) moveR += 1;
    if (this.keyState.has('KeyA') || this.keyState.has('ArrowLeft')) moveR -= 1;
    if (!isWalk) {
      if (this.keyState.has('KeyE') || this.keyState.has('Space')) moveU += 1;
      if (this.keyState.has('KeyQ') || this.keyState.has('ShiftLeft')) moveU -= 1;
    }

    // Mavic Drone Island-Circling Tour in DRONE_6DOF before manual drone keys are engaged
    const hasManualInput = moveF !== 0 || moveR !== 0 || moveU !== 0;
    if (hasManualInput) {
      this._droneManualEngaged = true;
    }
    if (!isWalk && !hasManualInput && !this._droneManualEngaged && this.camera.position.y > 2.0 && this.velocity.lengthSq() < 0.04) {
      this.cinematicAngle += clampedDt * 0.115;
      this._evalMavicIslandOrbit(this.cinematicAngle, floorHeightFn, this.goalPos, this.goalTarget);
      this.camera.position.x += (this.goalPos.x - this.camera.position.x) * dampAlpha * 0.55;
      this.camera.position.z += (this.goalPos.z - this.camera.position.z) * dampAlpha * 0.55;
      this.camera.position.y += (this.goalPos.y - this.camera.position.y) * dampAlpha * 0.55;
      const lookDx = this.goalTarget.x - this.camera.position.x;
      const lookDz = this.goalTarget.z - this.camera.position.z;
      this.yaw = Math.atan2(-lookDx, -lookDz) + this._mavicGimbalYaw;
      const basePitch = Math.atan2(this.goalTarget.y - this.camera.position.y, Math.hypot(lookDx, lookDz));
      this.pitch = Math.max(-1.35, Math.min(1.15, basePitch + this._mavicGimbalPitch));
      this.roll += (0.036 * Math.cos(this.cinematicAngle) - this.roll) * dampAlpha; // Mavic centripetal roll banking
    } else {
      this.velocity.addScaledVector(this._forward, moveF * accel * clampedDt);
      this.velocity.addScaledVector(this._right, moveR * accel * clampedDt);
      this.velocity.y += moveU * accel * clampedDt;

      const drag = Math.exp(-4.8 * clampedDt);
      this.velocity.multiplyScalar(drag);
      if (this.velocity.length() > maxSpeed) {
        this.velocity.setLength(maxSpeed);
      }

      if (!isWalk) {
        const cmdMavicSpd = Math.hypot(moveF, moveR, moveU) > 0.04 ? maxSpeed : 0.0;
        this._applyDroneAPASAvoidance(this.camera.position, this.velocity, clampedDt, 0.68, ctx, cmdMavicSpd, true);
      }

      this.camera.position.addScaledVector(this.velocity, clampedDt);

      // Centripetal roll banking on lateral drone maneuvers (Hukum Emas 15)
      const targetRoll = isWalk ? 0 : -moveR * 0.085;
      this.roll += (targetRoll - this.roll) * dampAlpha;
    }

    const groundY = floorHeightFn ? floorHeightFn(this.camera.position.x, this.camera.position.z) : 0.0;
    const minCamY = isWalk
      ? groundY + 1.68
      : groundY < -0.5
        ? groundY + 0.68
        : Math.max(0.68, groundY + 0.68);

    if (isWalk) {
      this.camera.position.y += (minCamY - this.camera.position.y) * (1.0 - Math.exp(-14.0 * clampedDt));
      resolveSoftSpatialCollision(
        this.camera.position,
        this.velocity,
        this.colliders,
        0.36,
        minCamY,
        clampedDt,
        this._prevCamPos,
        ctx,
        true
      );
      const postWalkY = (floorHeightFn ? floorHeightFn(this.camera.position.x, this.camera.position.z) : 0.0) + 1.68;
      if (this.camera.position.y < postWalkY) {
        this.camera.position.y = postWalkY;
      }
    } else {
      resolveSoftSpatialCollision(
        this.camera.position,
        this.velocity,
        this.colliders,
        0.68,
        minCamY,
        clampedDt,
        this._prevCamPos,
        ctx,
        false
      );
    }

    this._prevCamPos.copy(this.camera.position);
    this._updateAltitudeAdaptiveNearPlane();
    const euler = new THREE.Euler(this.pitch, this.yaw, this.roll, 'YXZ');
    this.camera.quaternion.setFromEuler(euler);
  }

  _updateAvata2FPV(clampedDt, floorHeightFn, ctx) {
    const isAcro = this.avataFlightProfile === 'ACRO';
    this._avataVelPrev.copy(this.velocity);

    let keyF = 0;
    let keyR = 0;
    let keyU = 0;
    let rollCmd = 0;
    if (this.keyState.has('KeyW') || this.keyState.has('ArrowUp')) keyF += 1;
    if (this.keyState.has('KeyS') || this.keyState.has('ArrowDown')) keyF -= 1;
    if (this.keyState.has('KeyD') || this.keyState.has('ArrowRight')) keyR += 1;
    if (this.keyState.has('KeyA') || this.keyState.has('ArrowLeft')) keyR -= 1;
    if (this.keyState.has('Space')) keyU += 1;
    if (this.keyState.has('KeyC') || this.keyState.has('ControlLeft') || this.keyState.has('ControlRight')) keyU -= 1;
    if (this.keyState.has('KeyQ')) rollCmd += 1;
    if (this.keyState.has('KeyE')) rollCmd -= 1;

    const surge = Math.max(-1, Math.min(1, keyF + this._avataStick.moveY));
    const strafe = Math.max(-1, Math.min(1, keyR + this._avataStick.moveX));
    const climb = Math.max(-1, Math.min(1, keyU + this._avataStick.climbY));
    const turbo =
      this.keyState.has('ShiftLeft') || this.keyState.has('ShiftRight') || Boolean(this._avataStick.boost);

    const hasManualInput =
      Math.abs(surge) > 0.04 ||
      Math.abs(strafe) > 0.04 ||
      Math.abs(climb) > 0.04 ||
      Math.abs(rollCmd) > 0.04 ||
      Math.abs(this._avataStick.lookX) > 0.05 ||
      Math.abs(this._avataStick.lookY) > 0.05 ||
      turbo;

    if (hasManualInput && this.avataAutoFreestyle) {
      // Instant seamless transition from Autonomous Proximity Showreel -> 100% Free-Flight anywhere!
      this.avataAutoFreestyle = false;
      this._avataRejoinBlend = 0.0;
    }

    const oceanY = Number.isFinite(ctx?.oceanWaterY) ? ctx.oceanWaterY : -1.45;
    this._avataApasTimer = Math.max(0.0, this._avataApasTimer - clampedDt);

    if (this.avataAutoFreestyle) {
      this._avataYawRateAccum = 0.0;
      const phasePrev = ((this._avataSplineT % 1.0) + 1.0) % 1.0;

      // Live Dynamic Obstacle & Traffic Lane-Shifting inside Subsea Underpass & Lorong Jembatan Teksas!
      let desiredLane = 0.0;
      let underpassOverBusLift = 0.0;
      const camX = this.camera.position.x;
      const camY = this.camera.position.y;
      const camZ = this.camera.position.z;
      const dynList =
        ctx && typeof ctx.getDynamicColliders === 'function'
          ? ctx.getDynamicColliders()
          : ctx && Array.isArray(ctx.dynamicColliders)
            ? ctx.dynamicColliders
            : null;

      if (camZ >= 250.0 && camZ <= 688.0 && camX < -305.0) {
        // Inside or entering Underpass Bawah Laut Selatan U–I: score candidate lanes against OBB vehicles (Bus Kuning UI)
        // while ignoring outer-sidewalk pedestrians so a sidewalk NPC NEVER steers Avata into an active bus lane!
        desiredLane = 1.05 * Math.sin(phasePrev * Math.PI * 28.0);
        if (dynList) {
          let blockedWest = false;
          let blockedEast = false;
          let nearestVehicleZ = 78.0;
          let singleDodgeLane = 0.0;
          for (let i = 0; i < dynList.length; i++) {
            const dc = dynList[i];
            if (!dc || dc.enabled === false) continue;
            // Strict 3D co-elevation check so surface aircraft/vehicles on Pulau I NEVER trigger underground subsea lane shifts!
            if (dc.minY !== undefined && (dc.minY > camY + 2.8 || dc.maxY < camY - 2.8)) continue;
            const isObb = (dc.kind || dc.type) === 'OBB';
            // Only carriageway vehicles (OBB or |x + 356| < 3.1m), never outer sidewalk pedestrians at ±3.85m
            const dcX = isObb ? dc.cx : Number.isFinite(dc.x) ? dc.x : 0.5 * (dc.minX + dc.maxX);
            if (!isObb && Math.abs(dcX - -356.0) > 3.05) continue;
            const dcZ = isObb ? dc.cz : Number.isFinite(dc.z) ? dc.z : 0.5 * (dc.minZ + dc.maxZ);
            const dzRel = dcZ - camZ;
            const dzAbs = Math.abs(dzRel);
            if (Math.abs(dcX - -356.0) < 3.6 && dcZ >= 260.0 && dcZ <= 680.0 && dzRel >= -9.5 && dzRel <= 48.0) {
              if (dcX < -356.0) blockedWest = true;
              else blockedEast = true;
              if (dzAbs < nearestVehicleZ) {
                nearestVehicleZ = dzAbs;
                singleDodgeLane = dcX >= -356.0 ? -2.35 : 2.35;
              }
            }
          }
          if (blockedWest && !blockedEast) {
            desiredLane = 2.35;
            this._avataApasTimer = 0.35;
          } else if (blockedEast && !blockedWest) {
            desiredLane = -2.35;
            this._avataApasTimer = 0.35;
          } else if (blockedWest && blockedEast) {
            desiredLane = singleDodgeLane;
            if (camZ < 321.0 || camZ > 629.0) {
              underpassOverBusLift = 2.35;
            }
            this._avataApasTimer = 0.35;
          }
        }
      } else if (camZ >= 142.0 && camZ <= 348.0 && Math.abs(camX - -28.0) < 9.0 && camY > 4.8) {
        // Inside or entering Lorong Jembatan Teksas UI: center-locked with live pedestrian lateral courtesy shift
        desiredLane = 0.0;
        if (dynList) {
          let nearestNpcZ = 24.0;
          for (let i = 0; i < dynList.length; i++) {
            const dc = dynList[i];
            if (!dc || dc.enabled === false) continue;
            if (dc.minY !== undefined && (dc.minY > camY + 2.5 || dc.maxY < camY - 2.8)) continue;
            const isObb = (dc.kind || dc.type) === 'OBB';
            const dcX = isObb ? dc.cx : Number.isFinite(dc.x) ? dc.x : 0.5 * (dc.minX + dc.maxX);
            const dcZ = isObb ? dc.cz : Number.isFinite(dc.z) ? dc.z : 0.5 * (dc.minZ + dc.maxZ);
            const dzRel = dcZ - camZ;
            const dzAbs = Math.abs(dzRel);
            if (Math.abs(dcX - -28.0) < 2.45 && dcZ >= 149.0 && dcZ <= 342.0 && dzRel >= -6.0 && dzRel <= 22.0 && dzAbs < nearestNpcZ) {
              nearestNpcZ = dzAbs;
              desiredLane = dcX >= -28.0 ? -1.15 : 1.15;
              this._avataApasTimer = 0.28;
            }
          }
        }
      }
      this._avataLiveLaneOffset += (desiredLane - this._avataLiveLaneOffset) * (1.0 - Math.exp(-12.5 * clampedDt));

      const eps = 0.0010;
      this._evalAvataCatmullPoint(this._avataSplineT, this._avataTmpP1, this._avataMeta);
      this._evalAvataCatmullPoint(this._avataSplineT - eps, this._avataTmpP0);
      this._evalAvataCatmullPoint(this._avataSplineT + eps, this._avataTmpP2);
      const dSplineLen = this._avataTmpP2.distanceTo(this._avataTmpP0) / (2.0 * eps);

      // Closed-Loop Spline Leash against lifted goalPos: prevents target from ever running ahead if drone detours!
      const lagDist =
        this._avataRejoinBlend > 0.002 ? 0.0 : this.camera.position.distanceTo(this.goalPos);
      const leashScale = lagDist > 7.5 ? Math.max(0.25, 1.0 - (lagDist - 7.5) / 14.0) : 1.0;

      const targetSpeed = this._avataMeta.speedMs;
      const effTargetSpeed = targetSpeed * leashScale;
      const dtSpline = (effTargetSpeed * clampedDt) / Math.max(120.0, dSplineLen);
      const nextSplineT = this._avataSplineT + dtSpline;
      if (nextSplineT >= 1.0) {
        this._avataLapCount += Math.floor(nextSplineT);
      }
      this._avataSplineT = nextSplineT % 1.0;
      const phase = ((this._avataSplineT % 1.0) + 1.0) % 1.0;

      this._evalAvataCatmullPoint(this._avataSplineT, this.goalPos, this._avataMeta);
      this._evalAvataCatmullPoint(this._avataSplineT - eps, this._avataTmpP0);
      this._evalAvataCatmullPoint(this._avataSplineT + eps, this._avataTmpP2);

      const corrType = this._avataMeta.corridorType;
      const corrWeight = this._avataMeta.corridorWeight;
      const wBridge = this._avataMeta.wBridge || 0.0;
      const wUnderpass = this._avataMeta.wUnderpass || 0.0;
      const wArchSkim = this._avataMeta.wArchSkim || 0.0;
      const strictFactor = Math.max(wBridge, wUnderpass);
      const openFactor = Math.max(0.0, 1.0 - Math.max(strictFactor, wArchSkim));
      if (underpassOverBusLift > 0.0 && wUnderpass > 0.2) {
        this.goalPos.y += underpassOverBusLift * wUnderpass;
      }

      if (openFactor > 1e-3 && ctx && strictFactor < 0.35) {
        // 3-pass 3D canopy/roof repulsion lift across open-air segments!
        const rep = this._avataRepelScratch;
        for (let pass = 0; pass < 3; pass++) {
          const penGoal = PhysicsKernel.probeDroneAPASRepulsion(
            this.goalPos.x,
            this.goalPos.y,
            this.goalPos.z,
            this.camera.position.x,
            this.camera.position.y,
            this.camera.position.z,
            1.95,
            ctx,
            rep
          );
          if (penGoal <= 0.0) break;
          this.goalPos.x += rep.nx * (penGoal + 0.25) * openFactor;
          this.goalPos.y += Math.max(0.38, rep.ny) * (penGoal + 0.35) * openFactor;
          this.goalPos.z += rep.nz * (penGoal + 0.25) * openFactor;
          this._avataApasTimer = 0.25;
        }
      }

      const tx = this._avataTmpP2.x - this._avataTmpP0.x;
      const ty = this._avataTmpP2.y - this._avataTmpP0.y;
      const tz = this._avataTmpP2.z - this._avataTmpP0.z;
      const tLen = Math.hypot(tx, ty, tz) || 1.0;
      const tanVx = (tx / tLen) * effTargetSpeed;
      const tanVy = (ty / tLen) * effTargetSpeed;
      const tanVz = (tz / tLen) * effTargetSpeed;

      if (this._avataRejoinBlend > 0.002) {
        this._avataRejoinBlend *= Math.exp(-2.8 * clampedDt);
        this.camera.position.lerpVectors(this.goalPos, this._avataRejoinPos, this._avataRejoinBlend);
        this.velocity.set(tanVx, tanVy, tanVz);
      } else {
        this._avataRejoinBlend = 0.0;
        // Unified C1 Feedforward + Proportional Pursuit Velocity across ALL 46 nodes (zero regime-switch hiccups!):
        const errX = this.goalPos.x - this.camera.position.x;
        const errY = this.goalPos.y - this.camera.position.y;
        const errZ = this.goalPos.z - this.camera.position.z;
        const gainK = 9.0 + strictFactor * 11.5;
        this.velocity.set(tanVx + errX * gainK, tanVy + errY * gainK, tanVz + errZ * gainK);
        const maxPursuitSpd = targetSpeed * 1.48;
        if (this.velocity.length() > maxPursuitSpd) {
          this.velocity.setLength(maxPursuitSpd);
        }
        if (
          strictFactor < 0.45 &&
          wArchSkim < 0.35 &&
          this._applyDroneAPASAvoidance(
            this.camera.position,
            this.velocity,
            clampedDt,
            0.52,
            ctx,
            targetSpeed,
            true
          )
        ) {
          this._avataApasTimer = 0.28;
        }
        this.camera.position.addScaledVector(this.velocity, clampedDt);
      }

      // Second derivative curvature vector for physically accurate centripetal FPV bank angle
      const scaleA = Math.pow(targetSpeed / Math.max(120.0, dSplineLen), 2) / (eps * eps);
      const ax = (this._avataTmpP2.x - 2.0 * this.goalPos.x + this._avataTmpP0.x) * scaleA;
      const ay = (this._avataTmpP2.y - 2.0 * this.goalPos.y + this._avataTmpP0.y) * scaleA;
      const az = (this._avataTmpP2.z - 2.0 * this.goalPos.z + this._avataTmpP0.z) * scaleA;

      const targetYaw = Math.atan2(-tx, -tz);
      let yawDiff = targetYaw - this.yaw;
      while (yawDiff > Math.PI) yawDiff -= Math.PI * 2.0;
      while (yawDiff < -Math.PI) yawDiff += Math.PI * 2.0;
      const rotAlpha = 1.0 - Math.exp((corrWeight > 0.25 ? -16.0 : -11.5) * clampedDt);
      this.yaw += yawDiff * rotAlpha;
      this._avataYawRateSmooth = (yawDiff * rotAlpha) / clampedDt;

      const horizLen = Math.hypot(tx, tz) || 1.0;
      const targetPitch = Math.max(-1.25, Math.min(1.15, Math.atan2(ty, horizLen)));
      this.pitch += (targetPitch - this.pitch) * rotAlpha;

      const rx = Math.cos(this.yaw);
      const rz = -Math.sin(this.yaw);
      const aDotRight = ax * rx + az * rz;
      const maxAutoBank = corrType === 1 ? 0.24 : corrType === 2 ? 0.42 : 0.92;
      const targetBank = Math.max(
        -maxAutoBank,
        Math.min(maxAutoBank, -Math.atan2(aDotRight * 1.38, 9.81 + Math.max(-5.5, ay)))
      );
      this.roll += (targetBank - this.roll) * (1.0 - Math.exp(-9.0 * clampedDt));
      const speedBoost = Math.max(0.0, (targetSpeed - 20.5) * 0.045);
      this._avataThrottle01 = Math.max(0.28, Math.min(0.98, 0.56 + (ty / tLen) * 0.34 + speedBoost));

      // Context-gated Easy-Acro 360° Stunts strictly in high open airspace (Tajug climb & Volcano/Niagara dive!)
      this._avataAutoRollCooldown = Math.max(0.0, this._avataAutoRollCooldown - clampedDt);
      const inHighOpenWindow =
        corrWeight < 0.02 &&
        ((phase > 0.945 && phase < 0.982) || (phase > 0.595 && phase < 0.630));
      if (inHighOpenWindow && this._avataAutoRollCooldown <= 0.0 && !this._avataStunt.active) {
        const stunts = ['ROLL_RIGHT', 'ROLL_LEFT', 'SPLIT_S', 'ROLL_RIGHT'];
        const chosenStunt = stunts[this._avataStuntIndex % stunts.length];
        if (this.triggerAvataStunt(chosenStunt)) {
          this._avataStuntIndex++;
          this._avataAutoRollCooldown = 12.5;
        }
      }
    } else {
      // 100% UNRESTRICTED FREE-FLIGHT FPV ACTION QUAD DYNAMICS + BINOCULAR APAS 360° (Fly Anywhere Across Pulau U & Pulau I!)
      const stickYawDelta = -this._avataStick.lookX * 2.45 * clampedDt;
      const stickPitchDelta = this._avataStick.lookY * 1.75 * clampedDt;
      this.yaw += stickYawDelta;
      this._avataYawRateAccum += stickYawDelta;
      const maxDownPitch = isAcro ? -1.48 : -1.32;
      const maxUpPitch = isAcro ? 1.38 : 1.18;
      this.pitch = Math.max(maxDownPitch, Math.min(maxUpPitch, this.pitch + stickPitchDelta));

      const instYawRate = this._avataYawRateAccum / clampedDt;
      this._avataYawRateAccum = 0.0;
      const yawSmoothAlpha = 1.0 - Math.exp(-11.0 * clampedDt);
      this._avataYawRateSmooth += (instYawRate - this._avataYawRateSmooth) * yawSmoothAlpha;

      const cosP = Math.cos(this.pitch);
      this._forward.set(-Math.sin(this.yaw) * cosP, Math.sin(this.pitch), -Math.cos(this.yaw) * cosP).normalize();
      this._right.set(Math.cos(this.yaw), 0.0, -Math.sin(this.yaw)).normalize();

      const maxSpeed = turbo ? 27.0 : 19.5; // Exact DJI Avata 2 Manual spec: 27 m/s = 97.2 km/h!
      const surgeAccel = (turbo ? 48.0 : 34.0) * (surge < 0 ? 0.68 : 1.0);
      const strafeAccel = turbo ? 36.0 : 27.0;
      const climbAccel = turbo ? 34.0 : 25.0;

      this.velocity.addScaledVector(this._forward, surge * surgeAccel * clampedDt);
      this.velocity.addScaledVector(this._right, strafe * strafeAccel * clampedDt);
      this.velocity.y += climb * climbAccel * clampedDt;

      // Authentic FPV aerodynamic drag + responsive air-brake when sticks center (preserves cornering drift!)
      const thrustActive = Math.abs(surge) > 0.04 || Math.abs(strafe) > 0.04 || Math.abs(climb) > 0.04;
      const linDrag = Math.exp(-(thrustActive ? 1.65 : 3.65) * clampedDt);
      this.velocity.multiplyScalar(linDrag);
      const curSpd = this.velocity.length();
      if (curSpd > maxSpeed) {
        this.velocity.setLength(maxSpeed);
      } else if (!thrustActive && curSpd < 0.04) {
        this.velocity.set(0, 0, 0);
      }

      const manualCmdSpd = thrustActive ? maxSpeed : 0.0;
      if (this._applyDroneAPASAvoidance(this.camera.position, this.velocity, clampedDt, 0.42, ctx, manualCmdSpd, true)) {
        this._avataApasTimer = 0.32;
      }

      this.camera.position.addScaledVector(this.velocity, clampedDt);

      // Coordinated Yaw-Roll Banking + Lateral Strafe Bank + Direct Q/E Acro Roll
      const horizSpd = Math.hypot(this.velocity.x, this.velocity.z);
      const yawBank = Math.atan2(this._avataYawRateSmooth * Math.max(5.5, horizSpd) * 0.92, 9.81);
      const strafeBank = -strafe * (isAcro ? 0.84 : 0.62);

      if (isAcro && Math.abs(rollCmd) > 0.05) {
        this.roll += rollCmd * 5.25 * clampedDt;
        if (this.roll > Math.PI) this.roll -= Math.PI * 2.0;
        if (this.roll < -Math.PI) this.roll += Math.PI * 2.0;
      } else {
        const maxBank = isAcro ? 1.28 : 0.92; // ±52.7° Sport, ±73.3° Acro
        const directBank = rollCmd * 0.92;
        const targetRoll = Math.max(-maxBank, Math.min(maxBank, yawBank + strafeBank + directBank));
        // Wrap shortest angular distance back to horizon when coming out of an Acro roll
        while (this.roll - targetRoll > Math.PI) this.roll -= Math.PI * 2.0;
        while (this.roll - targetRoll < -Math.PI) this.roll += Math.PI * 2.0;
        this.roll += (targetRoll - this.roll) * (1.0 - Math.exp(-9.5 * clampedDt));
      }

      const cmdMag = Math.min(1.0, Math.hypot(surge, strafe, climb));
      this._avataThrottle01 = Math.max(
        0.18,
        Math.min(1.0, 0.22 + cmdMag * (turbo ? 0.78 : 0.58) + (curSpd / 27.0) * 0.2)
      );
    }

    // Hard 3D Solid-Body Non-Penetration (0.42m Ducted Cinewhoop Prop-Guard Sphere)
    const walkY = floorHeightFn ? floorHeightFn(this.camera.position.x, this.camera.position.z) : 0.0;
    const minAvataY =
      walkY < -0.5 ? walkY + 0.48 : Math.max(oceanY + 0.48, walkY + 0.48);
    resolveSoftSpatialCollision(
      this.camera.position,
      this.velocity,
      this.colliders,
      0.42,
      minAvataY,
      clampedDt,
      this._prevCamPos,
      ctx,
      false
    );

    // Easy-Acro 360° Stunt Overlay (Barrel Roll Left/Right, Backflip, Split-S Dive)
    let stuntRollAdd = 0.0;
    let stuntPitchAdd = 0.0;
    if (this._avataStunt.active) {
      this._avataStunt.elapsed += clampedDt;
      const s = Math.min(1.0, this._avataStunt.elapsed / Math.max(0.2, this._avataStunt.duration));
      const u = s * s * s * (s * (s * 6.0 - 15.0) + 10.0);
      if (this._avataStunt.type === 'ROLL_LEFT') {
        stuntRollAdd = u * Math.PI * 2.0;
      } else if (this._avataStunt.type === 'ROLL_RIGHT') {
        stuntRollAdd = -u * Math.PI * 2.0;
      } else if (this._avataStunt.type === 'FLIP_BACK') {
        stuntPitchAdd = u * Math.PI * 2.0;
      } else if (this._avataStunt.type === 'SPLIT_S') {
        const halfRoll = Math.min(1.0, s * 2.1);
        const uRoll = halfRoll * halfRoll * (3.0 - 2.0 * halfRoll);
        stuntRollAdd = Math.PI * Math.sin(Math.PI * u);
        stuntPitchAdd = -0.68 * Math.sin(Math.PI * uRoll);
      }
      if (s >= 1.0) {
        this._avataStunt.active = false;
      }
    }

    // Instantaneous G-Force from 3D velocity vector derivative + gravity vector
    const dvx = (this.velocity.x - this._avataVelPrev.x) / clampedDt;
    const dvy = (this.velocity.y - this._avataVelPrev.y) / clampedDt + 9.81;
    const dvz = (this.velocity.z - this._avataVelPrev.z) / clampedDt;
    const rawG = Math.max(0.2, Math.min(6.2, Math.hypot(dvx, dvy, dvz) / 9.81));
    this._avataGForce += (rawG - this._avataGForce) * (1.0 - Math.exp(-8.0 * clampedDt));

    // Speed-Coupled 155° Super-Wide Action FOV (74° hover -> 88° at 97 km/h!) & Razor-Close 0.10m Near Plane
    const speedNorm = Math.min(1.0, this.velocity.length() / 27.0);
    const desiredBaseFov = 74.0 + speedNorm * 14.0;
    this._avataCurrentBaseFov += (desiredBaseFov - this._avataCurrentBaseFov) * (1.0 - Math.exp(-7.5 * clampedDt));
    this._applyBaseVerticalFov(this._avataCurrentBaseFov);

    if (this.camera && typeof this.camera.updateProjectionMatrix === 'function') {
      const camY = Math.max(0.0, Number(this.camera.position.y) || 0.0);
      const desiredNear = camY < 32.0 ? 0.1 : Math.min(1.6, 0.1 + (camY - 32.0) * 0.003);
      if (Math.abs(this.camera.near - desiredNear) > 0.02) {
        this.camera.near = desiredNear;
        this.camera.updateProjectionMatrix();
      }
    }

    this._prevCamPos.copy(this.camera.position);
    this._avataEuler.set(this.pitch + stuntPitchAdd, this.yaw, this.roll + stuntRollAdd, 'YXZ');
    this.camera.quaternion.setFromEuler(this._avataEuler);
    this.camera.up.set(0, 1, 0).applyQuaternion(this.camera.quaternion);
  }
}
