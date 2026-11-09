import * as THREE from 'three';
import { PhysicsKernel } from './PhysicsKernel.js';

export { PhysicsKernel };

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
  isWalk = false,
  enforceDroneNfz = false
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
    enforceDroneNfz: Boolean(enforceDroneNfz || ctx?.enforceDroneNfz),
    groundHeightFn: ctx?.groundHeightFn || null,
    walkableHeightFn: ctx?.walkableHeightFn || null,
    coastDistFn: ctx?.coastDistFn || null,
    waterSurfaceFn: ctx?.waterSurfaceFn || null,
    oceanWaterY: Number.isFinite(ctx?.oceanWaterY) ? ctx.oceanWaterY : -1.45,
    oceanWaveSurfaceFn: ctx?.oceanWaveSurfaceFn || null,
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
    this._mavicGimbalIdle = 0.0;
    this._mavicCinemaFov = 43.0;
    this._exploreStick = { moveX: 0, moveY: 0, climbY: 0, boost: false };
    this._avataZoneJumpFlip = 0;
    this._mavicSeedPhase = ((Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0) % 1000 * 0.00628318;
    this.cinematicAngle = 1.88 + (this._mavicSeedPhase % 0.16);

    // DJI Avata 2 Pro-Pilot FPV State (Cinema 51.5°–56.5° Optics, Non-Repeating Generative Graph, Whisker APAS 360°)
    this.avataFlightProfile = 'SPORT'; // 'SPORT' (Auto-level ±50° bank) | 'ACRO' (Full Acro ±74° / 360° roll & dive)
    this.avataAutoFreestyle = true;
    this._avataRngState = ((Date.now() ^ Math.floor(Math.random() * 0xffffffff) ^ 0x9e3779b9) >>> 0) || 1;
    this._avataStreamNodes = [];
    this._avataStreamBaseIdx = 0;
    this._avataStreamCursor = 1.0;
    this._avataSplineT = 0.0;
    this._avataLapCount = 0;
    this._avataCurrentGate = 'G_ROTUNDA_N';
    this._avataManeuverHistory = [];
    this._avataRecentTabu = [];
    this._avataUnvisitedSigZones = new Set(['TEKSAS_IN', 'UNDERPASS', 'PULAU_I', 'KOLONG']);
    this._avataActiveManeuverId = 'INIT';
    this._avataRejoinBlend = 0.0;
    this._avataRejoinPos = new THREE.Vector3();
    this._avataEuler = new THREE.Euler(0, 0, 0, 'YXZ');
    this._avataTmpP0 = new THREE.Vector3();
    this._avataTmpP1 = new THREE.Vector3();
    this._avataTmpP2 = new THREE.Vector3();
    this._avataVelPrev = new THREE.Vector3();
    this._avataMeta = {
      speedMs: 11.8,
      corridorWeight: 0.0,
      corridorType: 0,
      lookX: 0,
      lookY: 28,
      lookZ: 0,
      lookBlend: 0,
      allowStunt: 0
    };
    this._avataLiveLaneOffset = 0.0;
    this._avataApasTimer = 0.0;
    this._avataRepelScratch = { nx: 0, ny: 0, nz: 0, pen: 0 };
    this._avataWhiskerScratch = { nx: 0, ny: 0, nz: 0, pen: 0 };
    this._avataStick = { moveX: 0, moveY: 0, lookX: 0, lookY: 0, climbY: 0, boost: false };
    this._avataFreelookYaw = 0.0;
    this._avataFreelookPitch = 0.0;
    this._avataFreelookIdle = 5.0;
    this._avataYawRateAccum = 0.0;
    this._avataYawRateSmooth = 0.0;
    this._avataGForce = 1.0;
    this._avataThrottle01 = 0.35;
    this._avataBaseVerticalFov = 51.5;
    this._avataCurrentBaseFov = 48.0;
    this._avataAutoRollCooldown = 8.5;
    this._avataStuntIndex = 0;
    this._avataStunt = { active: false, type: 'ROLL_RIGHT', elapsed: 0.0, duration: 0.72 };

    // Hard 3D solid-world physics context & previous-frame position for continuous sweep anti-tunneling
    this.physicsContext = null;
    this._prevCamPos = new THREE.Vector3();
    this._hasPrevCamPos = false;
  }

  /**
   * Non-Memorized Directed Pro-Pilot Maneuver Graph across Pulau U & Pulau I (10 Directional Gates, 21 Maneuvers).
   * Node format: [x, y, z, speedMs, corridorType, lookX, lookY, lookZ, lookBlend, allowStunt]
   * corridorType:
   *   0 = Open-Air Pro-Pilot Line (seeded racing-line perturbation + 3D APAS pre-screening + heroic upward landmark gaze)
   *   1 = Lorong Jembatan Teksas UI Precision Lock (bidirectional Northbound & Southbound inside covered truss!)
   *   2 = Underpass Bawah Laut Selatan U–I Precision Lock (bidirectional -8.2m subsea bore + live Bus Kuning dodge!)
   *   3 = Kolong Jembatan Teksas Lagoon Skim (bidirectional Eastbound & Westbound underneath +7.8m arch soffit!)
   *   4 = Precision Portal / Architectural Canyon Pre-Alignment (zero random lateral drift)
   */
  static AVATA_PRO_MANEUVER_GRAPH = [
    // --- FROM G_ROTUNDA_N ([-16, 8.5, 118] flowing North) ---
    {
      id: 'M01_LORONG_TEKSAS_NORTH',
      from: 'G_ROTUNDA_N',
      to: 'G_TEKSAS_N',
      sigZone: 'TEKSAS_IN',
      nodes: [
        [-28.0, 4.4, 138.5, 11.6, 4, -28.0, 18.5, 171.65, 0.28, 0],
        [-28.0, 3.85, 152.0, 12.0, 1, -28.0, 14.0, 171.65, 0.0, 0],
        [-28.0, 6.28, 172.0, 12.6, 1, -28.0, 11.2, 245.65, 0.0, 0],
        [-28.0, 9.55, 205.0, 13.2, 1, -28.0, 11.2, 245.65, 0.0, 0],
        [-28.0, 11.22, 245.65, 13.5, 1, -28.0, 10.5, 319.65, 0.0, 0],
        [-28.0, 9.55, 286.0, 13.2, 1, -28.0, 12.5, 319.65, 0.0, 0],
        [-28.0, 6.28, 319.4, 12.5, 1, -28.0, 8.0, 350.0, 0.0, 0],
        [-28.0, 3.65, 336.0, 12.0, 1, -32.0, 44.0, 698.0, 0.0, 0],
        [-28.0, 7.4, 350.5, 12.2, 4, -32.0, 46.0, 698.0, 0.25, 0]
      ]
    },
    {
      id: 'M02_KOLONG_TEKSAS_EASTBOUND',
      from: 'G_ROTUNDA_N',
      to: 'G_TELUK_EAST',
      sigZone: 'KOLONG',
      nodes: [
        [-54.0, 8.2, 166.0, 11.8, 0, -28.0, 21.0, 171.65, 0.34, 0],
        [-68.0, 2.35, 244.2, 12.4, 3, -28.0, 9.5, 245.65, 0.0, 0],
        [-28.0, 1.72, 245.65, 13.0, 3, 24.0, 4.5, 247.0, 0.0, 0],
        [18.0, 2.45, 246.8, 12.8, 3, 68.0, 9.0, 266.0, 0.0, 0],
        [68.0, 11.2, 286.0, 12.4, 0, -32.0, 46.0, 698.0, 0.28, 0],
        [92.0, 11.2, 272.0, 12.0, 0, -28.0, 16.0, 245.65, 0.24, 0],
        [62.0, 7.8, 254.0, 11.8, 0, -28.0, 14.0, 245.65, 0.24, 0]
      ]
    },
    {
      id: 'M03_VIP_HELIPAD_WEST_BAY',
      from: 'G_ROTUNDA_N',
      to: 'G_TELUK_WEST',
      sigZone: null,
      nodes: [
        [-48.0, 9.4, 148.0, 11.4, 0, 0.0, 33.0, 0.0, 0.36, 0],
        [-88.0, 10.2, 188.0, 11.8, 0, -255.0, 26.5, 329.0, 0.34, 0],
        [-96.0, 9.5, 228.0, 11.8, 0, -255.0, 26.5, 329.0, 0.32, 0]
      ]
    },

    // --- FROM G_TEKSAS_N ([-28, 7.4, 350.5] flowing North) ---
    {
      id: 'M04_DPPD_CANYON_GAP_LINE',
      from: 'G_TEKSAS_N',
      to: 'G_UNDERPASS_U_N',
      sigZone: null,
      nodes: [
        [-102.0, 15.2, 358.0, 12.2, 0, -255.0, 27.0, 332.0, 0.42, 0],
        [-188.0, 18.2, 344.0, 11.6, 0, -255.0, 27.5, 332.0, 0.48, 0],
        [-226.0, 20.2, 336.5, 11.6, 4, -255.0, 27.5, 332.0, 0.40, 0],
        [-255.0, 20.8, 331.0, 11.6, 4, -272.0, 26.5, 339.0, 0.28, 0],
        [-271.8, 21.2, 339.2, 11.8, 4, -262.0, 28.0, 352.0, 0.22, 0],
        [-296.0, 22.8, 342.0, 12.0, 0, -255.0, 26.5, 332.0, 0.24, 0],
        [-344.0, 22.5, 292.0, 12.2, 0, -255.0, 26.0, 329.0, 0.22, 0],
        [-384.0, 16.8, 254.0, 12.0, 0, -356.0, 6.0, 268.0, 0.0, 0],
        [-382.0, 12.8, 226.0, 12.0, 0, -356.0, 6.0, 268.0, 0.0, 0],
        [-362.0, 8.2, 242.0, 12.0, 4, -356.0, 4.0, 280.0, 0.0, 0],
        [-356.0, 4.8, 262.0, 12.4, 4, -356.0, -2.0, 310.0, 0.0, 0]
      ]
    },
    {
      id: 'M05_DPPD_WATERFRONT_BELVEDERE',
      from: 'G_TEKSAS_N',
      to: 'G_UNDERPASS_U_N',
      sigZone: null,
      nodes: [
        [-84.0, 16.5, 338.0, 12.2, 0, -255.0, 26.5, 329.0, 0.36, 0],
        [-162.0, 18.8, 294.0, 11.6, 0, -255.0, 27.5, 335.0, 0.46, 0],
        [-222.0, 22.5, 258.0, 11.4, 0, -255.0, 28.0, 338.0, 0.44, 0],
        [-300.0, 42.5, 234.0, 12.4, 0, -255.0, 26.0, 329.0, 0.20, 1],
        [-348.0, 29.5, 196.0, 12.4, 0, -356.0, 8.0, 262.0, 0.0, 0],
        [-386.0, 20.5, 210.0, 12.4, 4, -356.0, 6.0, 268.0, 0.0, 0],
        [-364.0, 9.6, 238.0, 12.2, 4, -356.0, 4.5, 274.0, 0.0, 0],
        [-356.0, 4.8, 262.0, 12.4, 4, -356.0, -2.0, 310.0, 0.0, 0]
      ]
    },
    {
      id: 'M06_DERMAGA_FERRY_STRAIT_DIRECT',
      from: 'G_TEKSAS_N',
      to: 'G_PULAU_I_RUNWAY_E',
      sigZone: null,
      nodes: [
        [-46.0, 10.5, 426.0, 14.0, 0, -32.0, 48.0, 698.0, 0.36, 0],
        [-134.0, 9.8, 536.0, 15.0, 0, -32.0, 49.0, 698.0, 0.38, 0],
        [-252.0, 10.2, 636.0, 13.8, 0, -32.0, 48.0, 698.0, 0.36, 0],
        [-332.0, 8.8, 686.0, 11.6, 4, -32.0, 49.0, 698.0, 0.42, 0]
      ]
    },

    // --- FROM G_TELUK_WEST ([-96, 9.5, 228] over Inner Teluk U) ---
    {
      id: 'M07_WEST_BAY_TO_DPPD_CANYON',
      from: 'G_TELUK_WEST',
      to: 'G_UNDERPASS_U_N',
      sigZone: null,
      nodes: [
        [-152.0, 13.8, 296.0, 11.8, 0, -255.0, 27.0, 332.0, 0.44, 0],
        [-198.0, 18.2, 340.0, 11.6, 0, -255.0, 27.5, 332.0, 0.46, 0],
        [-226.0, 20.2, 336.5, 11.6, 4, -255.0, 27.5, 332.0, 0.38, 0],
        [-255.0, 20.8, 331.0, 11.6, 4, -272.0, 26.5, 339.0, 0.28, 0],
        [-271.8, 21.2, 339.2, 11.8, 4, -262.0, 28.0, 352.0, 0.22, 0],
        [-296.0, 22.8, 342.0, 12.0, 0, -255.0, 26.5, 332.0, 0.24, 0],
        [-344.0, 22.5, 292.0, 12.2, 0, -255.0, 26.0, 329.0, 0.22, 0],
        [-384.0, 16.8, 254.0, 12.0, 0, -356.0, 6.0, 268.0, 0.0, 0],
        [-382.0, 12.8, 226.0, 12.0, 0, -356.0, 6.0, 268.0, 0.0, 0],
        [-362.0, 8.2, 242.0, 12.0, 4, -356.0, 4.0, 280.0, 0.0, 0],
        [-356.0, 4.8, 262.0, 12.4, 4, -356.0, -2.0, 310.0, 0.0, 0]
      ]
    },
    {
      id: 'M08_WEST_BAY_HIGHLAND_SWEEP',
      from: 'G_TELUK_WEST',
      to: 'G_UNDERPASS_U_N',
      sigZone: null,
      nodes: [
        [-165.0, 17.5, 268.0, 11.4, 0, -255.0, 27.5, 334.0, 0.44, 0],
        [-228.0, 23.0, 252.0, 11.2, 0, -255.0, 28.0, 336.0, 0.42, 0],
        [-300.0, 42.5, 232.0, 12.4, 0, -255.0, 26.0, 329.0, 0.20, 1],
        [-348.0, 29.5, 196.0, 12.4, 0, -356.0, 8.0, 262.0, 0.0, 0],
        [-386.0, 20.5, 210.0, 12.4, 4, -356.0, 6.0, 268.0, 0.0, 0],
        [-364.0, 9.6, 238.0, 12.2, 4, -356.0, 4.5, 274.0, 0.0, 0],
        [-356.0, 4.8, 262.0, 12.4, 4, -356.0, -2.0, 310.0, 0.0, 0]
      ]
    },
    {
      id: 'M09_WEST_BAY_TO_REKTORAT_AWE',
      from: 'G_TELUK_WEST',
      to: 'G_ROTUNDA_S',
      sigZone: null,
      nodes: [
        [-74.0, 13.2, 182.0, 11.6, 0, 0.0, 33.0, 0.0, 0.40, 0],
        [-38.0, 12.4, 154.0, 11.2, 0, 0.0, 33.0, 0.0, 0.52, 0],
        [-18.0, 11.2, 126.0, 10.8, 0, 0.0, 33.0, 0.0, 0.62, 0]
      ]
    },
    {
      id: 'M09B_WEST_BAY_KOLONG_TEKSAS_EASTBOUND',
      from: 'G_TELUK_WEST',
      to: 'G_TELUK_EAST',
      sigZone: 'KOLONG',
      nodes: [
        [-68.0, 2.45, 244.2, 12.4, 3, -28.0, 9.5, 245.65, 0.0, 0],
        [-28.0, 1.72, 245.65, 13.0, 3, 24.0, 4.5, 247.0, 0.0, 0],
        [18.0, 2.45, 246.8, 12.6, 3, 62.0, 7.8, 254.0, 0.0, 0],
        [62.0, 7.8, 254.0, 11.8, 0, -28.0, 14.0, 245.65, 0.24, 0]
      ]
    },

    // --- FROM G_UNDERPASS_U_N ([-356, 4.8, 256] aligned North into Subsea Underpass) ---
    {
      id: 'M10_SUBSEA_UNDERPASS_NORTHBOUND',
      from: 'G_UNDERPASS_U_N',
      to: 'G_PULAU_I_RUNWAY_E',
      sigZone: 'UNDERPASS',
      nodes: [
        [-356.0, 2.85, 274.0, 13.2, 2, -356.0, -4.0, 330.0, 0.0, 0],
        [-356.0, -2.5, 302.0, 14.4, 2, -356.0, -6.12, 360.0, 0.0, 0],
        [-356.0, -6.12, 330.0, 15.2, 2, -356.0, -6.12, 420.0, 0.0, 0],
        [-356.0, -6.12, 420.0, 15.6, 2, -356.0, -6.12, 525.0, 0.0, 0],
        [-356.0, -6.12, 525.0, 15.6, 2, -356.0, -6.12, 620.0, 0.0, 0],
        [-356.0, -6.12, 620.0, 14.8, 2, -356.0, 0.0, 660.0, 0.0, 0],
        [-356.0, -1.8, 648.0, 13.6, 2, -356.0, 4.0, 680.0, 0.0, 0],
        [-356.0, 3.1, 672.0, 12.8, 2, -32.0, 48.0, 698.0, 0.0, 0],
        [-352.0, 7.4, 682.0, 12.2, 4, -32.0, 48.0, 698.0, 0.30, 0],
        [-332.0, 8.8, 686.0, 11.4, 4, -32.0, 49.0, 698.0, 0.42, 0]
      ]
    },

    // --- FROM G_PULAU_I_RUNWAY_E ([-332, 8.8, 686] South Apron Corridor facing East toward 60m Gunung Api, outside Runway NFZ!) ---
    {
      id: 'M11_GUNUNG_API_CRATER_AND_NIAGARA_PLUNGE',
      from: 'G_PULAU_I_RUNWAY_E',
      to: 'G_PULAU_I_NORTH',
      sigZone: 'PULAU_I',
      nodes: [
        [-232.0, 10.4, 686.0, 11.2, 4, -32.0, 50.0, 698.0, 0.44, 0],
        [-124.0, 23.5, 692.0, 10.6, 0, -32.0, 53.0, 698.0, 0.42, 0],
        [-74.0, 47.5, 720.0, 10.0, 0, -32.0, 58.5, 698.0, 0.36, 0],
        [-28.0, 65.5, 701.0, 9.8, 0, -32.0, 46.0, 692.0, 0.26, 1],
        [-35.0, 50.5, 666.0, 11.6, 4, -48.0, 14.0, 636.0, 0.0, 0],
        [-53.0, 25.0, 640.0, 13.2, 4, -56.0, 8.0, 608.0, 0.0, 0],
        [-58.0, 10.8, 604.0, 12.6, 0, 0.0, 32.0, 0.0, 0.24, 0]
      ]
    },
    {
      id: 'M12_NIAGARA_WATERFALL_ASCENT_TO_CRATER',
      from: 'G_PULAU_I_RUNWAY_E',
      to: 'G_PULAU_I_NORTH',
      sigZone: 'PULAU_I',
      nodes: [
        [-228.0, 10.2, 686.0, 11.6, 4, -32.0, 50.0, 698.0, 0.44, 0],
        [-128.0, 12.5, 594.0, 11.6, 0, -38.0, 46.0, 682.0, 0.46, 0],
        [-56.0, 11.8, 608.0, 11.2, 4, -44.0, 36.0, 660.0, 0.38, 0],
        [-48.0, 28.5, 644.0, 11.0, 4, -34.0, 56.0, 686.0, 0.28, 0],
        [-32.0, 54.0, 676.0, 10.6, 4, -22.0, 60.0, 702.0, 0.24, 0],
        [-8.0, 65.5, 702.0, 10.4, 0, -32.0, 46.0, 698.0, 0.26, 1],
        [12.0, 44.0, 664.0, 12.0, 0, -48.0, 12.0, 608.0, 0.20, 0],
        [-22.0, 22.0, 626.0, 12.4, 0, -58.0, 10.8, 604.0, 0.20, 0],
        [-58.0, 10.8, 604.0, 12.6, 0, 0.0, 32.0, 0.0, 0.24, 0]
      ]
    },

    // --- FROM G_PULAU_I_NORTH ([-58, 10.8, 604] off Waterfall / Dermaga I) ---
    {
      id: 'M13_SELAT_DIRECT_TO_KOLONG_WEST',
      from: 'G_PULAU_I_NORTH',
      to: 'G_TELUK_EAST',
      sigZone: null,
      nodes: [
        [-16.0, 10.2, 522.0, 14.6, 0, 0.0, 32.0, 0.0, 0.28, 0],
        [48.0, 11.5, 422.0, 14.6, 0, 0.0, 32.0, 0.0, 0.30, 0],
        [88.0, 10.2, 326.0, 13.2, 0, -28.0, 18.0, 245.65, 0.28, 0],
        [62.0, 7.8, 254.0, 11.8, 0, -28.0, 12.0, 245.65, 0.24, 0]
      ]
    },
    {
      id: 'M14_SELAT_FERRY_TO_TEKSAS_SOUTHBOUND',
      from: 'G_PULAU_I_NORTH',
      to: 'G_TEKSAS_S_ENTRY',
      sigZone: null,
      nodes: [
        [-44.0, 8.4, 528.0, 14.2, 0, -28.0, 15.0, 319.65, 0.30, 0],
        [-26.0, 9.8, 436.0, 13.4, 0, -28.0, 15.0, 319.65, 0.32, 0],
        [-28.0, 7.2, 351.0, 12.0, 4, -28.0, 13.0, 319.65, 0.22, 0]
      ]
    },
    {
      id: 'M15_REVERSE_SUBSEA_SOUTHBOUND',
      from: 'G_PULAU_I_NORTH',
      to: 'G_UNDERPASS_U_S_EXIT',
      sigZone: 'UNDERPASS',
      nodes: [
        [-165.0, 14.5, 636.0, 13.8, 0, -233.5, 14.0, 665.5, 0.28, 0],
        [-296.0, 16.5, 686.0, 13.0, 4, -356.0, 4.0, 672.0, 0.20, 0],
        [-348.0, 10.8, 686.0, 12.4, 4, -356.0, 4.0, 672.0, 0.0, 0],
        [-356.0, 6.8, 680.0, 12.2, 4, -356.0, 2.0, 660.0, 0.0, 0],
        [-356.0, 3.1, 672.0, 13.0, 2, -356.0, -3.0, 635.0, 0.0, 0],
        [-356.0, -1.8, 648.0, 14.2, 2, -356.0, -6.12, 600.0, 0.0, 0],
        [-356.0, -6.12, 620.0, 15.2, 2, -356.0, -6.12, 525.0, 0.0, 0],
        [-356.0, -6.12, 525.0, 15.6, 2, -356.0, -6.12, 420.0, 0.0, 0],
        [-356.0, -6.12, 420.0, 15.6, 2, -356.0, -6.12, 330.0, 0.0, 0],
        [-356.0, -6.12, 330.0, 14.8, 2, -356.0, -1.0, 290.0, 0.0, 0],
        [-356.0, -2.5, 302.0, 13.6, 2, -356.0, 3.0, 270.0, 0.0, 0],
        [-356.0, 2.85, 274.0, 12.6, 2, -356.0, 7.0, 248.0, 0.0, 0],
        [-362.0, 10.8, 254.0, 12.4, 4, -380.0, 16.0, 224.0, 0.0, 0],
        [-384.0, 19.5, 228.0, 12.6, 0, -255.0, 26.0, 329.0, 0.15, 0],
        [-356.0, 24.5, 202.0, 12.6, 0, -255.0, 26.0, 329.0, 0.25, 0]
      ]
    },

    // --- FROM G_TELUK_EAST ([62, 7.8, 254] flowing West into Teluk U Mouth) ---
    {
      id: 'M16_KOLONG_TEKSAS_WESTBOUND',
      from: 'G_TELUK_EAST',
      to: 'G_TELUK_WEST',
      sigZone: 'KOLONG',
      nodes: [
        [18.0, 2.35, 246.8, 12.6, 3, -28.0, 9.5, 245.65, 0.0, 0],
        [-28.0, 1.68, 245.65, 13.2, 3, -68.0, 4.5, 244.0, 0.0, 0],
        [-68.0, 2.45, 244.2, 12.6, 3, -96.0, 9.5, 228.0, 0.0, 0],
        [-96.0, 9.5, 228.0, 11.8, 0, -255.0, 26.5, 329.0, 0.32, 0]
      ]
    },
    {
      id: 'M17_PILON_LINGGA_VAULT_TO_REKTORAT',
      from: 'G_TELUK_EAST',
      to: 'G_ROTUNDA_S',
      sigZone: null,
      nodes: [
        [24.0, 15.5, 202.0, 11.8, 0, -28.0, 22.0, 171.65, 0.36, 0],
        [-2.0, 14.2, 162.0, 11.2, 0, 0.0, 33.0, 0.0, 0.48, 0],
        [-18.0, 11.2, 126.0, 10.8, 0, 0.0, 33.0, 0.0, 0.62, 0]
      ]
    },

    // --- FROM G_TEKSAS_S_ENTRY ([-28, 7.2, 351] aligned South into Lorong Jembatan Teksas!) ---
    {
      id: 'M18_LORONG_TEKSAS_SOUTHBOUND',
      from: 'G_TEKSAS_S_ENTRY',
      to: 'G_ROTUNDA_S',
      sigZone: 'TEKSAS_IN',
      nodes: [
        [-28.0, 3.65, 336.0, 11.8, 1, -28.0, 10.0, 319.65, 0.0, 0],
        [-28.0, 6.28, 319.4, 12.4, 1, -28.0, 11.2, 245.65, 0.0, 0],
        [-28.0, 9.55, 286.0, 13.0, 1, -28.0, 11.2, 245.65, 0.0, 0],
        [-28.0, 11.22, 245.65, 13.4, 1, -28.0, 9.5, 172.0, 0.0, 0],
        [-28.0, 9.55, 205.0, 13.0, 1, 0.0, 33.0, 0.0, 0.0, 0],
        [-28.0, 6.28, 172.0, 12.4, 1, 0.0, 33.0, 0.0, 0.0, 0],
        [-28.0, 3.85, 152.0, 11.8, 1, 0.0, 33.0, 0.0, 0.0, 0],
        [-25.0, 6.8, 138.0, 11.2, 4, 0.0, 33.0, 0.0, 0.45, 0],
        [-18.0, 11.2, 126.0, 10.8, 0, 0.0, 33.0, 0.0, 0.62, 0]
      ]
    },

    // --- FROM G_UNDERPASS_U_S_EXIT ([-356, 6.8, 248] emerging Southbound onto Pulau U) ---
    {
      id: 'M19_SOUTH_COAST_RETURN_TO_REKTORAT',
      from: 'G_UNDERPASS_U_S_EXIT',
      to: 'G_ROTUNDA_S',
      sigZone: null,
      nodes: [
        [-292.0, 24.5, 198.0, 13.2, 0, 0.0, 33.0, 0.0, 0.26, 0],
        [-168.0, 22.5, 172.0, 13.2, 0, 0.0, 33.0, 0.0, 0.36, 0],
        [-62.0, 14.8, 150.0, 11.8, 0, 0.0, 33.0, 0.0, 0.50, 0],
        [-18.0, 11.2, 126.0, 10.8, 0, 0.0, 33.0, 0.0, 0.62, 0]
      ]
    },
    {
      id: 'M20_SOUTH_EXIT_TO_WEST_BAY',
      from: 'G_UNDERPASS_U_S_EXIT',
      to: 'G_TELUK_WEST',
      sigZone: null,
      nodes: [
        [-308.0, 41.5, 224.0, 12.2, 0, -255.0, 27.0, 332.0, 0.34, 1],
        [-212.0, 22.5, 252.0, 12.2, 0, -255.0, 27.0, 332.0, 0.38, 0],
        [-96.0, 9.5, 228.0, 11.8, 0, -28.0, 18.0, 245.65, 0.28, 0]
      ]
    },

    // --- FROM G_ROTUNDA_S ([-18, 11.2, 126] facing South straight at 50m Gedung Rektorat UI!) ---
    {
      id: 'M21_REKTORAT_LOW_PLAZA_AWE_CLIMB',
      from: 'G_ROTUNDA_S',
      to: 'G_ROTUNDA_N',
      sigZone: null,
      nodes: [
        [-44.0, 11.8, 86.0, 10.8, 0, 0.0, 30.0, 0.0, 0.72, 0],
        [-68.0, 16.5, 26.0, 11.4, 0, 0.0, 34.0, 0.0, 0.64, 0],
        [-54.0, 22.0, -48.0, 11.8, 0, 0.0, 35.0, 0.0, 0.56, 0],
        [16.0, 24.0, -72.0, 12.0, 0, 0.0, 35.0, 0.0, 0.54, 0],
        [68.0, 19.5, -16.0, 12.0, 0, 0.0, 33.0, 0.0, 0.52, 0],
        [66.0, 19.5, 86.0, 11.8, 0, 0.0, 30.0, 0.0, 0.42, 0],
        [-16.0, 8.5, 118.0, 11.6, 0, -28.0, 18.0, 171.65, 0.28, 0]
      ]
    },
    {
      id: 'M22_REKTORAT_CROWN_SPLIT_S_DIVE',
      from: 'G_ROTUNDA_S',
      to: 'G_ROTUNDA_N',
      sigZone: null,
      nodes: [
        [34.0, 18.8, 104.0, 11.2, 0, 0.0, 30.0, 0.0, 0.72, 0],
        [68.0, 22.5, 42.0, 11.6, 0, 0.0, 36.0, 0.0, 0.62, 0],
        [62.0, 33.5, -42.0, 11.6, 0, 0.0, 38.0, 0.0, 0.52, 1],
        [-10.0, 34.0, -74.0, 12.0, 0, 0.0, 35.0, 0.0, 0.52, 0],
        [-66.0, 23.0, -18.0, 12.2, 0, 0.0, 33.0, 0.0, 0.48, 0],
        [-64.0, 20.5, 76.0, 11.8, 0, 0.0, 30.0, 0.0, 0.38, 0],
        [-16.0, 8.5, 118.0, 11.6, 0, -28.0, 18.0, 171.65, 0.28, 0]
      ]
    },
    {
      id: 'M23_VIDEOTRON_EAST_PROMENADE_TO_KOLONG',
      from: 'G_ROTUNDA_S',
      to: 'G_TELUK_EAST',
      sigZone: null,
      nodes: [
        [36.0, 17.8, 106.0, 11.4, 0, 0.0, 29.0, 0.0, 0.68, 0],
        [76.0, 16.5, 108.0, 11.8, 0, 0.0, 31.0, 0.0, 0.48, 0],
        [94.0, 13.5, 158.0, 12.4, 0, -28.0, 18.0, 245.65, 0.34, 0],
        [88.0, 10.5, 222.0, 12.2, 0, -28.0, 16.0, 245.65, 0.30, 0],
        [62.0, 7.8, 254.0, 11.8, 0, -28.0, 12.0, 245.65, 0.24, 0]
      ]
    }
  ];

  static AVATA_ARCHIPELAGO_NODES = CameraDirector.AVATA_PRO_MANEUVER_GRAPH.flatMap((m) => m.nodes);
  static AVATA_FREESTYLE_WAYPOINTS = CameraDirector.AVATA_ARCHIPELAGO_NODES.map((n) => [n[0], n[1], n[2]]);

  getActiveBaseVerticalFov(mode = this.mode) {
    if (mode === 'DRONE_AVATA2_FPV') {
      return Number.isFinite(this._avataCurrentBaseFov) ? this._avataCurrentBaseFov : 51.5;
    }
    if (mode === 'DRONE_6DOF' || mode === 'AUTO_CINEMATIC') {
      return Number.isFinite(this._mavicCinemaFov) ? this._mavicCinemaFov : 43.0;
    }
    return 48.0;
  }

  adjustMavicCinemaZoom(deltaDeg = 0.0) {
    const next = Math.max(28.0, Math.min(54.0, (Number(this._mavicCinemaFov) || 43.0) + (Number(deltaDeg) || 0.0)));
    this._mavicCinemaFov = next;
    if (this.mode === 'DRONE_6DOF' || this.mode === 'AUTO_CINEMATIC') {
      this._applyBaseVerticalFov(next);
    }
    return next;
  }

  resetMavicGimbalAndZoom() {
    this._mavicGimbalYaw = 0.0;
    this._mavicGimbalPitch = 0.0;
    this._mavicGimbalIdle = 0.0;
    this._mavicCinemaFov = 43.0;
    if (this.mode === 'DRONE_6DOF' || this.mode === 'AUTO_CINEMATIC') {
      this._applyBaseVerticalFov(43.0);
    }
  }

  resumeMavicIslandTour() {
    if (this.mode !== 'DRONE_6DOF') {
      this.setMode('DRONE_6DOF', { startIslandTour: true });
      return;
    }
    let bestAng = this.cinematicAngle;
    let bestD2 = Infinity;
    const steps = 72;
    for (let i = 0; i < steps; i++) {
      const a = (Math.PI * 2.0 * i) / steps;
      this._evalMavicIslandOrbit(a, null, this._avataTmpP0, this._avataTmpP1);
      const d2 = this._avataTmpP0.distanceToSquared(this.camera.position);
      if (d2 < bestD2) {
        bestD2 = d2;
        bestAng = a;
      }
    }
    this.cinematicAngle = bestAng;
    this._droneManualEngaged = false;
    this._exploreStick.moveX = 0;
    this._exploreStick.moveY = 0;
    this._exploreStick.climbY = 0;
    this._exploreStick.boost = false;
    this.velocity.set(0, 0, 0);
    this._mavicGimbalYaw = 0.0;
    this._mavicGimbalPitch = 0.0;
    this._mavicGimbalIdle = 0.0;
  }

  setExploreStickInput(partial = {}) {
    if (!partial || typeof partial !== 'object') return;
    if (Number.isFinite(partial.moveX)) this._exploreStick.moveX = Math.max(-1, Math.min(1, partial.moveX));
    if (Number.isFinite(partial.moveY)) this._exploreStick.moveY = Math.max(-1, Math.min(1, partial.moveY));
    if (Number.isFinite(partial.climbY)) this._exploreStick.climbY = Math.max(-1, Math.min(1, partial.climbY));
    if (typeof partial.boost === 'boolean') this._exploreStick.boost = partial.boost;
  }

  jumpAvataToZone(zoneKey = 'RANDOM') {
    if (!this._avataZoneJumpCounts) this._avataZoneJumpCounts = {};
    const idx = this._avataZoneJumpCounts[zoneKey] || 0;
    this._avataZoneJumpCounts[zoneKey] = idx + 1;
    const zoneMap = {
      REKTORAT: [
        { startManeuver: 'M21_REKTORAT_LOW_PLAZA_AWE_CLIMB', cursorOffset: 1.15 },
        { startManeuver: 'M22_REKTORAT_CROWN_SPLIT_S_DIVE', cursorOffset: 1.15 }
      ],
      LORONG_TEKSAS: [
        { startManeuver: 'M01_LORONG_TEKSAS_NORTH', cursorOffset: 2.85 },
        { startManeuver: 'M18_LORONG_TEKSAS_SOUTHBOUND', cursorOffset: 1.85 }
      ],
      KOLONG_TEKSAS: [
        { startManeuver: 'M16_KOLONG_TEKSAS_WESTBOUND', cursorOffset: 1.15 },
        { startManeuver: 'M09B_WEST_BAY_KOLONG_TEKSAS_EASTBOUND', cursorOffset: 1.15 },
        { startManeuver: 'M02_KOLONG_TEKSAS_EASTBOUND', cursorOffset: 1.85 }
      ],
      DPPD_CANYON: [
        { startManeuver: 'M04_DPPD_CANYON_GAP_LINE', cursorOffset: 2.15 },
        { startManeuver: 'M07_WEST_BAY_TO_DPPD_CANYON', cursorOffset: 2.15 }
      ],
      UNDERPASS: [
        { startManeuver: 'M10_SUBSEA_UNDERPASS_NORTHBOUND', cursorOffset: 2.85 },
        { startManeuver: 'M15_REVERSE_SUBSEA_SOUTHBOUND', cursorOffset: 5.85 }
      ],
      GUNUNG_NIAGARA: [
        { startManeuver: 'M12_NIAGARA_WATERFALL_ASCENT_TO_CRATER', cursorOffset: 2.05 },
        { startManeuver: 'M11_GUNUNG_API_CRATER_AND_NIAGARA_PLUNGE', cursorOffset: 2.35 }
      ]
    };
    const variants = zoneMap[zoneKey];
    if (!variants) {
      this.setMode('DRONE_AVATA2_FPV', {});
      return this._avataActiveManeuverId;
    }
    const chosen = variants[idx % variants.length];
    this.setMode('DRONE_AVATA2_FPV', chosen);
    return this._avataActiveManeuverId;
  }

  _nextAvataRand01() {
    let t = (this._avataRngState += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  _minHopsFromGateToSigZone(startGate, targetSigZone) {
    const graph = CameraDirector.AVATA_PRO_MANEUVER_GRAPH;
    const queue = [{ gate: startGate, hops: 0 }];
    const seen = new Set([startGate]);
    while (queue.length > 0) {
      const cur = queue.shift();
      if (cur.hops > 7) break;
      for (let i = 0; i < graph.length; i++) {
        const m = graph[i];
        if (m.from !== cur.gate) continue;
        if (m.sigZone === targetSigZone) return cur.hops + 1;
        if (!seen.has(m.to)) {
          seen.add(m.to);
          queue.push({ gate: m.to, hops: cur.hops + 1 });
        }
      }
    }
    return 8;
  }

  _prescreenAndPerturbProNode(rawNode, prevNode, isGateEndpoint) {
    let x = rawNode[0];
    let y = rawNode[1];
    let z = rawNode[2];
    let spd = rawNode[3];
    const cType = rawNode[4] || 0;
    const lx = rawNode[5] ?? 0.0;
    const ly = rawNode[6] ?? 26.0;
    const lz = rawNode[7] ?? 0.0;
    const lBlend = rawNode[8] ?? 0.0;
    const allowStunt = rawNode[9] ?? 0;

    // Apply bounded pro-pilot racing-line variation strictly on non-endpoint open-air nodes (cType === 0)
    if (cType === 0 && !isGateEndpoint) {
      const rLat = (this._nextAvataRand01() - 0.5) * 5.2;
      const rAlt = (this._nextAvataRand01() - 0.42) * 3.6;
      const rSpd = (this._nextAvataRand01() - 0.5) * 1.1;
      if (prevNode) {
        const dx = x - prevNode[0];
        const dz = z - prevNode[2];
        const hL = Math.hypot(dx, dz) || 1.0;
        x += (-dz / hL) * rLat;
        z += (dx / hL) * rLat;
      } else {
        x += rLat;
      }
      y += rAlt;
      spd = Math.max(9.0, Math.min(15.6, spd + rSpd));
    }

    // Pre-screen open-air node AND chord midpoint against ALL 3D trees, buildings, bridges, & terrain!
    const ctx = this.physicsContext;
    if (ctx && cType === 0) {
      const oceanY = Number.isFinite(ctx.oceanWaterY) ? ctx.oceanWaterY : -1.45;
      const gY = ctx.groundHeightFn ? ctx.groundHeightFn(x, z) : 0.0;
      y = Math.max(y, Math.max(oceanY + 2.8, gY + 3.4));
      const rep = this._avataRepelScratch;
      for (let pass = 0; pass < 4; pass++) {
        const pen = PhysicsKernel.probeDroneAPASRepulsion(x, y, z, x, y + 4.0, z, 3.35, ctx, rep);
        if (pen <= 0.0) break;
        x += rep.nx * (pen + 0.45);
        y += Math.max(0.48, rep.ny) * (pen + 0.55);
        z += rep.nz * (pen + 0.45);
      }
      if (prevNode) {
        for (const frac of [0.35, 0.55, 0.75]) {
          const mx = prevNode[0] + (x - prevNode[0]) * frac;
          const my = prevNode[1] + (y - prevNode[1]) * frac;
          const mz = prevNode[2] + (z - prevNode[2]) * frac;
          const penMid = PhysicsKernel.probeDroneAPASRepulsion(mx, my, mz, mx, my + 4.0, mz, 3.05, ctx, rep);
          if (penMid > 0.0) {
            x += rep.nx * (penMid + 0.65);
            y += Math.max(0.58, rep.ny) * (penMid + 0.90);
            z += rep.nz * (penMid + 0.65);
          }
        }
      }
    }

    return [x, y, z, spd, cType, lx, ly, lz, lBlend, allowStunt];
  }

  _appendNextProPilotManeuver(forcedManeuverId = null) {
    const graph = CameraDirector.AVATA_PRO_MANEUVER_GRAPH;
    let chosen = null;
    if (forcedManeuverId) {
      chosen = graph.find((m) => m.id === forcedManeuverId) || null;
    }
    if (!chosen) {
      const candidates = graph.filter((m) => m.from === this._avataCurrentGate);
      const weights = [];
      let sumW = 0.0;
      for (let i = 0; i < candidates.length; i++) {
        const m = candidates[i];
        let w = 1.0;
        // Tabu recency penalty so the track NEVER repeats a memorized loop
        const recIdx = this._avataRecentTabu.lastIndexOf(m.id);
        if (recIdx !== -1) {
          const age = this._avataRecentTabu.length - recIdx;
          w *= age <= 2 ? 0.05 : age <= 5 ? 0.24 : 0.55;
        }
        // Novelty hunger toward unvisited signature zones while discouraging re-running already-visited tunnels
        if (this._avataUnvisitedSigZones.size > 0) {
          if (m.sigZone && this._avataUnvisitedSigZones.has(m.sigZone)) {
            w *= 32.0;
          } else if (m.sigZone && !this._avataUnvisitedSigZones.has(m.sigZone)) {
            w *= 0.01;
          } else {
            let bestHops = 9;
            for (const zTag of this._avataUnvisitedSigZones) {
              const h = this._minHopsFromGateToSigZone(m.to, zTag);
              if (h < bestHops) bestHops = h;
            }
            const lenBias = 6.0 / Math.max(4.0, m.nodes.length);
            w *= (bestHops <= 1 ? 7.5 : bestHops === 2 ? 2.2 : 0.2) * lenBias;
          }
        }
        weights.push(w);
        sumW += w;
      }
      let pick = this._nextAvataRand01() * Math.max(1e-6, sumW);
      chosen = candidates[0] || graph[0];
      for (let i = 0; i < candidates.length; i++) {
        pick -= weights[i];
        if (pick <= 0.0) {
          chosen = candidates[i];
          break;
        }
      }
    }

    this._avataActiveManeuverId = chosen.id;
    this._avataCurrentGate = chosen.to;
    this._avataManeuverHistory.push(chosen.id);
    this._avataRecentTabu.push(chosen.id);
    if (this._avataRecentTabu.length > 9) this._avataRecentTabu.shift();

    if (chosen.sigZone) {
      this._avataUnvisitedSigZones.delete(chosen.sigZone);
      if (this._avataUnvisitedSigZones.size === 0) {
        this._avataLapCount++;
        this._avataUnvisitedSigZones = new Set(['TEKSAS_IN', 'UNDERPASS', 'PULAU_I', 'KOLONG']);
      }
    }

    for (let i = 0; i < chosen.nodes.length; i++) {
      const prev =
        this._avataStreamNodes.length > 0
          ? this._avataStreamNodes[this._avataStreamNodes.length - 1]
          : null;
      const isGateEnd = i === chosen.nodes.length - 1;
      const built = this._prescreenAndPerturbProNode(chosen.nodes[i], prev, isGateEnd);
      built.maneuverId = chosen.id;
      this._avataStreamNodes.push(built);
    }
  }

  _ensureAvataStreamAhead() {
    while (this._avataStreamNodes.length - (this._avataStreamCursor - this._avataStreamBaseIdx) < 22) {
      this._appendNextProPilotManeuver();
    }
    const localIdx = Math.floor(this._avataStreamCursor) - this._avataStreamBaseIdx;
    if (localIdx > 28) {
      const drop = localIdx - 8;
      this._avataStreamNodes.splice(0, drop);
      this._avataStreamBaseIdx += drop;
    }
  }

  _initAvataProPilotStream(options = {}) {
    if (Number.isFinite(options.seed)) {
      this._avataRngState = (Math.trunc(options.seed) >>> 0) || 1;
    } else {
      this._avataRngState = ((Date.now() ^ Math.floor(Math.random() * 0xffffffff) ^ (this._avataRngState * 1664525)) >>> 0) || 1;
    }
    this._avataStreamNodes = [];
    this._avataStreamBaseIdx = 0;
    this._avataLapCount = 0;
    this._avataManeuverHistory = [];
    this._avataRecentTabu = [];
    this._avataUnvisitedSigZones = new Set(['TEKSAS_IN', 'UNDERPASS', 'PULAU_I', 'KOLONG']);

    let startManeuver = options.startManeuver || null;
    let cursorOffset = Number.isFinite(options.cursorOffset) ? options.cursorOffset : 1.05;
    if (!startManeuver && Number.isFinite(options.startT) && options.startT > 0.03) {
      const tNorm = ((options.startT % 1.0) + 1.0) % 1.0;
      if (tNorm >= 0.14 && tNorm < 0.31) {
        startManeuver = 'M01_LORONG_TEKSAS_NORTH';
        cursorOffset = 3.2;
      } else if (tNorm >= 0.31 && tNorm < 0.46) {
        startManeuver = 'M04_DPPD_CANYON_GAP_LINE';
        cursorOffset = 2.2;
      } else if (tNorm >= 0.46 && tNorm < 0.64) {
        startManeuver = 'M10_SUBSEA_UNDERPASS_NORTHBOUND';
        cursorOffset = 4.2;
      } else if (tNorm >= 0.64 && tNorm < 0.84) {
        startManeuver = 'M11_GUNUNG_API_CRATER_AND_NIAGARA_PLUNGE';
        cursorOffset = 2.0;
      } else if (tNorm >= 0.84) {
        startManeuver = 'M16_KOLONG_TEKSAS_WESTBOUND';
        cursorOffset = 1.65;
      }
    }

    if (startManeuver) {
      const mSpec = CameraDirector.AVATA_PRO_MANEUVER_GRAPH.find((m) => m.id === startManeuver);
      if (mSpec) {
        this._avataCurrentGate = mSpec.from;
        // Prepend anchor node before startManeuver so Catmull-Rom p0 is well-defined
        const firstN = mSpec.nodes[0];
        const secondN = mSpec.nodes[1] || firstN;
        this._avataStreamNodes.push([
          firstN[0] - (secondN[0] - firstN[0]),
          firstN[1] - (secondN[1] - firstN[1]) * 0.5,
          firstN[2] - (secondN[2] - firstN[2]),
          firstN[3],
          firstN[4],
          firstN[5] ?? 0,
          firstN[6] ?? 26,
          firstN[7] ?? 0,
          firstN[8] ?? 0,
          0
        ]);
        this._appendNextProPilotManeuver(startManeuver);
        this._avataStreamCursor = cursorOffset;
        this._ensureAvataStreamAhead();
        this._avataSplineT = (this._avataStreamCursor / 42.0) % 1.0;
        return;
      }
    }

    // Randomize opening heroic Rektorat reveal line so NO TWO SESSIONS start identically!
    const openingPool = [
      'M21_REKTORAT_LOW_PLAZA_AWE_CLIMB',
      'M22_REKTORAT_CROWN_SPLIT_S_DIVE',
      'M23_VIDEOTRON_EAST_PROMENADE_TO_KOLONG'
    ];
    const openId = openingPool[Math.floor(this._nextAvataRand01() * openingPool.length)];
    this._avataCurrentGate = 'G_ROTUNDA_S';
    this._avataStreamNodes.push([-14.0, 12.8, 154.0, 11.0, 0, 0.0, 31.0, 0.0, 0.65, 0]);
    this._avataStreamNodes.push([-18.0, 11.2, 126.0, 10.8, 0, 0.0, 31.0, 0.0, 0.72, 0]);
    this._appendNextProPilotManeuver(openId);
    // Start right into the monumental 50m Rektorat upward reveal, with random sub-node phase!
    this._avataStreamCursor = 1.15 + this._nextAvataRand01() * 0.85;
    this._ensureAvataStreamAhead();
    this._avataSplineT = (this._avataStreamCursor / 42.0) % 1.0;
  }

  getAvataManeuverHistory() {
    return this._avataManeuverHistory.slice();
  }

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
        return 'LORONG JEMBATAN SURYA';
      }
      if (bD !== null && y < bD - 0.5) {
        return 'KOLONG JEMBATAN SURYA';
      }
    }
    if (z > 590.0) {
      if (Math.hypot(x - -32.0, z - 698.0) < 58.0 && y > 32.0) return 'KAWAH GUNUNG API I';
      if (Math.hypot(x - -44.0, z - 643.0) < 42.0) return 'AIR TERJUN NIAGARA MINI';
      if (Math.abs(z - 708.0) < 28.0 && x >= -380.0 && x <= -130.0) return 'LANDASAN PACU PULAU I';
      return 'PULAU I · BANDARA UNIVERSE';
    }
    if (z > 465.0 && z <= 590.0) return 'SELAT UNIVERSE & DERMAGA';
    if (Math.hypot(x - -255.0, z - 329.0) < 88.0) return 'KOMPLEKS NEXUS · TECHNO PARK';
    if (Math.hypot(x - -104.0, z - 106.5) < 42.0) return 'HELIPAD VIP PULAU U';
    if (Math.hypot(x - -2.5, z - 106.5) < 42.0) return 'AIR MANCUR ROTUNDA';
    if (Math.hypot(x - 28.4, z - 78.2) < 30.0) return 'VIDEOTRON UNIVERSE 3D';
    if (Math.hypot(x, z) < 68.0 && y > 24.0) return 'PUNCAK MENARA UTAMA';
    if (z >= 145.0 && z <= 335.0 && x > -135.0) return 'TELUK U · UNIVERSE ISLAND';
    return 'PULAU U · MENARA UTAMA';
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
      if (this._avataStreamNodes.length < 6) {
        this._initAvataProPilotStream();
      } else {
        this._ensureAvataStreamAhead();
      }
      // Find nearest safe point on active rolling stream so rejoining Pro-Pilot blends smoothly
      let bestCursor = this._avataStreamCursor;
      let bestDistSq = Infinity;
      const minC = this._avataStreamBaseIdx + 1.05;
      const maxC = this._avataStreamBaseIdx + this._avataStreamNodes.length - 3.05;
      const steps = 64;
      for (let i = 0; i <= steps; i++) {
        const candC = minC + ((maxC - minC) * i) / steps;
        this._evalAvataStreamPoint(candC, this._avataTmpP0);
        const dSq = this._avataTmpP0.distanceToSquared(this.camera.position);
        if (dSq < bestDistSq) {
          bestDistSq = dSq;
          bestCursor = candC;
        }
      }
      this._avataStreamCursor = bestCursor;
      this._avataRejoinPos.copy(this.camera.position);
      this._avataRejoinBlend = 1.0;
      this._avataFreelookYaw = 0.0;
      this._avataFreelookPitch = 0.0;
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
        pos.z >= 149.0 && pos.z <= 342.0 && Math.abs(pos.x - -28.0) <= 3.5 && pos.y < 13.5;
      if (inUnderpass || inBridge) return false;
    }
    const valid =
      type === 'ROLL_LEFT' || type === 'ROLL_RIGHT' || type === 'SPLIT_S' || type === 'FLIP_BACK'
        ? type
        : 'ROLL_RIGHT';
    this._avataStunt.active = true;
    this._avataStunt.type = valid;
    this._avataStunt.elapsed = 0.0;
    this._avataStunt.duration = valid === 'SPLIT_S' ? 0.96 : 0.72;
    if (!this.avataAutoFreestyle) {
      this.velocity.y = Math.max(this.velocity.y + 3.6, 2.8);
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
      maneuverId: this._avataActiveManeuverId,
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
      fovDeg: Math.round(this.camera?.fov || 52),
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

  setFPSSubjectOffset(offset = 0) {
    this.fpsSubjectOffset = Math.max(0, Math.floor(Number(offset) || 0)) % 240;
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
    // Quasi-Periodic Cinema Aerial Trajectory (Irrational Golden/Sqrt2 Harmonics -> Never Traces Identical Circle!)
    // Frames 50m Gedung Rektorat, Jembatan Teksas, 8-Story STP-UI / 6-Story ILRC, & 60m Gunung Api from heroic angles!
    const pA = angle * 0.6180339887 + this._mavicSeedPhase;
    const pB = angle * 1.4142135623 - this._mavicSeedPhase * 0.73;
    const cx = -32.0 + Math.sin(pA) * 10.0;
    const cz = 124.0 + Math.cos(pA * 0.85) * 10.0;
    const rx = 162.0 + Math.cos(pB) * 12.0 + Math.sin(angle * 2.0) * 5.0;
    const rz = 238.0 + Math.sin(pB) * 12.0 + Math.cos(angle * 2.0) * 6.0;
    let px = cx + Math.sin(angle) * rx;
    let pz = cz + Math.cos(angle) * rz;

    const ctx = this.physicsContext;
    const gFn = ctx?.groundHeightFn || floorHeightFn || null;
    const rawGround = gFn
      ? 0.2 *
        (gFn(px, pz) +
          gFn(px + 16.0, pz) +
          gFn(px - 16.0, pz) +
          gFn(px, pz + 16.0) +
          gFn(px, pz - 16.0))
      : 0.0;
    const oceanY = Number.isFinite(ctx?.oceanWaterY) ? ctx.oceanWaterY : -1.45;
    const baseSurf = Math.max(oceanY, rawGround);

    // Heroic low-to-mid cinema elevation (22.5m..32m over water/plaza so 50m towers & 60m volcano tower overhead,
    // smoothly lifting above 27m hill canopies over elevated forest terrain!)
    const landBoost = PhysicsKernel.smootherstep(-0.8, 4.5, rawGround) * 8.5;
    let py = Math.max(22.5, baseSurf + 20.5 + landBoost + Math.sin(pB * 0.9) * 3.0 + Math.cos(pA) * 2.0);

    if (ctx) {
      const rep = this._avataRepelScratch;
      for (let pass = 0; pass < 4; pass++) {
        const pen = PhysicsKernel.probeDroneAPASRepulsion(px, py, pz, px, py + 6.0, pz, 4.8, ctx, rep);
        if (pen <= 0.0) break;
        px += rep.nx * pen * 1.15;
        py += Math.max(0.55, rep.ny) * pen * 1.25;
        pz += rep.nz * pen * 1.15;
      }
    }
    outPos.set(px, py, pz);

    // Smooth inverse-distance landmark gaze blending with near-field fade so horizontal distance stays >= 65m
    const dRekt = Math.hypot(px - 0.0, pz - 4.0);
    const dTeks = Math.hypot(px - -28.0, pz - 236.0);
    const dDppd = Math.hypot(px - -255.0, pz - 332.0);
    const dVolc = Math.hypot(px - -32.0, pz - 698.0);

    const wRekt = (4.25 * PhysicsKernel.smootherstep(32.0, 78.0, dRekt)) / (1.0 + Math.pow(dRekt / 185.0, 2.1));
    const wTeks = (1.15 * PhysicsKernel.smootherstep(42.0, 92.0, dTeks)) / (1.0 + Math.pow(dTeks / 155.0, 2.1));
    const wDppd = (1.95 * PhysicsKernel.smootherstep(42.0, 92.0, dDppd)) / (1.0 + Math.pow(dDppd / 170.0, 2.1));
    const wVolc =
      ((pz > 240.0 ? 1.35 : 0.25) * PhysicsKernel.smootherstep(48.0, 105.0, dVolc)) /
      (1.0 + Math.pow(dVolc / 340.0, 2.0));
    const wArch = 0.45;
    const wSum = wRekt + wTeks + wDppd + wVolc + wArch || 1.0;

    const tx = (wRekt * 0.0 + wTeks * -28.0 + wDppd * -255.0 + wVolc * -32.0 + wArch * -75.0) / wSum;
    const ty = (wRekt * 24.5 + wTeks * 14.5 + wDppd * 21.5 + wVolc * 38.0 + wArch * 14.0) / wSum;
    const tz = (wRekt * 4.0 + wTeks * 236.0 + wDppd * 332.0 + wVolc * 698.0 + wArch * 175.0) / wSum;
    outTarget.set(tx, ty, tz);
  }

  _applyBaseVerticalFov(baseVFov = 48.0) {
    if (!this.camera || typeof this.camera.updateProjectionMatrix !== 'function') return;
    const aspect = Number(this.camera.aspect) || 1.6;
    const maxPortraitFov = 68;
    let nextFov = baseVFov;
    if (aspect < 1.0) {
      const rad = (baseVFov * Math.PI) / 180;
      const hFov = 2 * Math.atan(Math.tan(rad / 2) / Math.max(0.56, aspect));
      nextFov = Math.min(maxPortraitFov, (hFov * 180) / Math.PI);
    }
    if (Math.abs(this.camera.fov - nextFov) > 0.04) {
      this.camera.fov = nextFov;
      this.camera.updateProjectionMatrix();
    }
  }

  _evalAvataStreamPoint(streamCursor, outVec, outMeta = null) {
    if (this._avataStreamNodes.length < 6) {
      this._initAvataProPilotStream();
    }
    const nodes = this._avataStreamNodes;
    const nLen = nodes.length;
    const rel = Math.max(1.0001, Math.min(nLen - 2.0001, streamCursor - this._avataStreamBaseIdx));
    const idx = Math.floor(rel);
    const u = rel - idx;

    const p0 = nodes[Math.max(0, idx - 1)];
    const p1 = nodes[idx];
    const p2 = nodes[Math.min(nLen - 1, idx + 1)];
    const p3 = nodes[Math.min(nLen - 1, idx + 2)];

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

    const lookX = (p1[5] ?? 0) + ((p2[5] ?? 0) - (p1[5] ?? 0)) * sSmooth;
    const lookY = (p1[6] ?? 26) + ((p2[6] ?? 26) - (p1[6] ?? 26)) * sSmooth;
    const lookZ = (p1[7] ?? 0) + ((p2[7] ?? 0) - (p1[7] ?? 0)) * sSmooth;
    const lookBlend = ((p1[8] ?? 0) + ((p2[8] ?? 0) - (p1[8] ?? 0)) * sSmooth) * (1.0 - corridorWeight);
    const allowStunt = (p1[9] || 0) > 0 || (p2[9] || 0) > 0 ? 1 : 0;

    const wNodeBridge = (p1[4] === 1 ? 1.0 - sSmooth : 0.0) + (p2[4] === 1 ? sSmooth : 0.0);
    const wNodeUnder = (p1[4] === 2 ? 1.0 - sSmooth : 0.0) + (p2[4] === 2 ? sSmooth : 0.0);
    const wArchSkim = (p1[4] === 3 ? 1.0 - sSmooth : 0.0) + (p2[4] === 3 ? sSmooth : 0.0);

    const zBridgeIn = PhysicsKernel.smootherstep(145.5, 152.0, pz) * (1.0 - PhysicsKernel.smootherstep(336.0, 344.5, pz));
    const wBridge = wNodeBridge * zBridgeIn;

    const zUnderIn = PhysicsKernel.smootherstep(264.0, 274.0, pz) * (1.0 - PhysicsKernel.smootherstep(672.0, 681.0, pz));
    const wUnderpass = wNodeUnder * zUnderIn;

    const ctx = this.physicsContext;
    if (corridorWeight < 0.98) {
      // Irrational non-repeating organic pilot flow harmonics using C^1-continuous analytical Catmull-Rom tangent (zero node-crossing snap!)
      const openGain = (1.0 - corridorWeight) * (1.0 - corridorWeight);
      const dc0 = -1.5 * u2 + 2.0 * u - 0.5;
      const dc1 = 4.5 * u2 - 5.0 * u;
      const dc2 = -4.5 * u2 + 4.0 * u + 0.5;
      const dc3 = 1.5 * u2 - u;
      const tx = dc0 * p0[0] + dc1 * p1[0] + dc2 * p2[0] + dc3 * p3[0];
      const tz = dc0 * p0[2] + dc1 * p1[2] + dc2 * p2[2] + dc3 * p3[2];
      const tHoriz = Math.hypot(tx, tz) || 1.0;
      const rightX = -tz / tHoriz;
      const rightZ = tx / tHoriz;
      const sCarve =
        openGain *
        (1.15 * Math.sin(streamCursor * 1.178097) + 0.80 * Math.cos(streamCursor * 0.728318 + 1.1));
      const vWave = openGain * 0.65 * Math.sin(streamCursor * 0.883572 + 0.5);
      px += rightX * sCarve;
      pz += rightZ * sCarve;
      py += vWave;
    }

    const oceanY = Number.isFinite(ctx?.oceanWaterY) ? ctx.oceanWaterY : -1.45;
    const strictFactor = Math.max(wBridge, wUnderpass);
    const openFactor = Math.max(0.0, 1.0 - Math.max(strictFactor, wArchSkim));

    if (wBridge > 1e-4) {
      // Bidirectional C1-Continuous Lock inside Lorong Jembatan Teksas UI (+2.12m above parabolic deck)
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
      // Bidirectional C1-Continuous Lock inside Underpass Bawah Laut Selatan U–I (-6.12m beneath Selat UI)
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
      const maxArchY = bSof !== null ? bSof - 1.75 : 5.75;
      const clampedArchY = Math.max(oceanY + 2.55, Math.min(maxArchY, py));
      py += (clampedArchY - py) * wArchSkim;
    }

    const underpassCutMask =
      (1.0 - PhysicsKernel.smootherstep(8.0, 16.0, Math.abs(px - -356.0))) *
      PhysicsKernel.smootherstep(262.0, 274.0, pz) *
      (1.0 - PhysicsKernel.smootherstep(668.0, 680.0, pz));
    const effOpenSpline = openFactor * (1.0 - underpassCutMask);
    if (effOpenSpline > 1e-3 && ctx?.groundHeightFn) {
      const spGround = ctx.groundHeightFn(px, pz);
      const minSplineY = Math.max(oceanY + 2.35, spGround + 2.95);
      if (py < minSplineY) {
        py += (minSplineY - py) * effOpenSpline;
      }
    }

    if (outMeta) {
      outMeta.speedMs = speedMs;
      outMeta.corridorWeight = corridorWeight;
      outMeta.corridorType = domType;
      outMeta.wBridge = wBridge;
      outMeta.wUnderpass = wUnderpass;
      outMeta.wArchSkim = wArchSkim;
      outMeta.lookX = lookX;
      outMeta.lookY = lookY;
      outMeta.lookZ = lookZ;
      outMeta.lookBlend = lookBlend;
      outMeta.allowStunt = allowStunt;
      outMeta.maneuverId = p1.maneuverId || this._avataActiveManeuverId;
    }
    return outVec.set(px, py, pz);
  }

  _evalAvataCatmullPoint(normT, outVec, outMeta = null) {
    if (this._avataStreamNodes.length < 6) {
      this._initAvataProPilotStream();
    }
    const minC = this._avataStreamBaseIdx + 1.05;
    const maxC = Math.max(minC + 1.0, this._avataStreamBaseIdx + this._avataStreamNodes.length - 3.05);
    const wrapped = ((normT % 1.0) + 1.0) % 1.0;
    return this._evalAvataStreamPoint(minC + wrapped * (maxC - minC), outVec, outMeta);
  }

  /**
   * Whisker-Guided 360° Binocular-Fisheye APAS (Advanced Pilot Assistance System) for ALL Drones
   * (DJI Avata 2 FPV & Mavic 3 Pro 6-DOF). Probes Center + Left-28° + Right-28° + Up-24° whiskers
   * to steer cleanly into open air around outer leaf lobes, spires, and bridge trusses with ZERO stall!
   */
  _applyDroneAPASAvoidance(pos, vel, dt, bodyRadius, ctx, commandedSpeed = 0.0, preferUpVault = true) {
    if (!ctx || !vel) return false;
    const speed = vel.length();
    const refSpeed = Math.max(speed, commandedSpeed);
    if (refSpeed < 0.20) return false;

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

    // 2. Three-Horizon Predictive Sphere Probes (Contact Halo + Near + Far Lookahead)
    const rep = this._avataRepelScratch;
    let nx = 0.0;
    let ny = 0.0;
    let nz = 0.0;
    let urgency = 0.0;
    let hardCloseContact = false;

    const penContact = PhysicsKernel.probeDroneAPASRepulsion(
      pos.x + dirX * 0.34,
      pos.y + dirY * 0.34,
      pos.z + dirZ * 0.34,
      pos.x,
      pos.y,
      pos.z,
      bodyRadius + 0.58,
      ctx,
      rep
    );
    if (penContact > 0.0) {
      nx = rep.nx;
      ny = rep.ny;
      nz = rep.nz;
      urgency = Math.min(1.0, 0.80 + penContact * 0.95);
      hardCloseContact = true;
    }

    const dNear = Math.min(5.6, 0.75 + refSpeed * 0.22);
    if (urgency < 0.82) {
      const penNear = PhysicsKernel.probeDroneAPASRepulsion(
        pos.x + dirX * dNear,
        pos.y + dirY * dNear,
        pos.z + dirZ * dNear,
        pos.x,
        pos.y,
        pos.z,
        bodyRadius + 0.68,
        ctx,
        rep
      );
      if (penNear > 0.0) {
        nx = rep.nx;
        ny = rep.ny;
        nz = rep.nz;
        urgency = Math.max(urgency, Math.min(1.0, 0.55 + penNear * 0.85));
        if (penNear > 0.14) hardCloseContact = true;
      }
    }

    if (urgency < 0.42 && refSpeed > 3.2) {
      const dFar = Math.min(10.5, 1.55 + refSpeed * 0.40);
      const penFar = PhysicsKernel.probeDroneAPASRepulsion(
        pos.x + dirX * dFar,
        pos.y + dirY * dFar,
        pos.z + dirZ * dFar,
        pos.x,
        pos.y,
        pos.z,
        bodyRadius + 0.92,
        ctx,
        rep
      );
      if (penFar > 0.0) {
        nx = rep.nx;
        ny = rep.ny;
        nz = rep.nz;
        urgency = Math.min(0.82, 0.30 + penFar * 0.48);
      }
    }

    if (urgency > 0.0) {
      const dDotN = dirX * nx + dirY * ny + dirZ * nz;
      const vDotN = vel.x * nx + vel.y * ny + vel.z * nz;
      if (vDotN < 0.0 || dDotN < -0.04) {
        // Project out inward closing velocity immediately when close so manual WASD can NEVER press into leaves!
        let removedInwardSpd = 0.0;
        if (vDotN < 0.0) {
          const dampIn = hardCloseContact ? 1.0 : Math.min(1.0, urgency * 20.0 * dt);
          removedInwardSpd = -vDotN * dampIn;
          vel.x -= vDotN * nx * dampIn;
          vel.y -= vDotN * ny * dampIn;
          vel.z -= vDotN * nz * dampIn;
        }

        // 3D Whisker Fan: test Left-28°, Right-28°, and Up-24° lookahead rays to pick the clearest open-air bypass!
        let tx = dirX - dDotN * nx;
        let ty = dirY - dDotN * ny;
        let tz = dirZ - dDotN * nz;

        if (!insideEnclosedRoof) {
          const hDir = Math.hypot(dirX, dirZ) || 1.0;
          const rgtX = -dirZ / hDir;
          const rgtZ = dirX / hDir;
          const wDist = Math.min(6.2, 1.4 + refSpeed * 0.28);
          const wRad = bodyRadius + 0.62;
          const wScratch = this._avataWhiskerScratch;

          // Left whisker (-rgt * 0.47 + dir * 0.88), Right whisker (+rgt * 0.47 + dir * 0.88), Up whisker (+Y * 0.44 + dir * 0.90)
          const lX = dirX * 0.88 - rgtX * 0.47;
          const lZ = dirZ * 0.88 - rgtZ * 0.47;
          const rX = dirX * 0.88 + rgtX * 0.47;
          const rZ = dirZ * 0.88 + rgtZ * 0.47;

          const penL = PhysicsKernel.probeDroneAPASRepulsion(
            pos.x + lX * wDist,
            pos.y + dirY * wDist,
            pos.z + lZ * wDist,
            pos.x,
            pos.y,
            pos.z,
            wRad,
            ctx,
            wScratch
          );
          const penR = PhysicsKernel.probeDroneAPASRepulsion(
            pos.x + rX * wDist,
            pos.y + dirY * wDist,
            pos.z + rZ * wDist,
            pos.x,
            pos.y,
            pos.z,
            wRad,
            ctx,
            wScratch
          );
          const penU = preferUpVault
            ? PhysicsKernel.probeDroneAPASRepulsion(
                pos.x + dirX * 0.88 * wDist,
                pos.y + (dirY * 0.88 + 0.48) * wDist,
                pos.z + dirZ * 0.88 * wDist,
                pos.x,
                pos.y,
                pos.z,
                wRad,
                ctx,
                wScratch
              )
            : 99.0;

          if (preferUpVault && penU <= Math.min(penL, penR) + 0.05 && ny > -0.22) {
            ty += 0.78;
          } else if (penL <= penR) {
            tx -= rgtX * 0.72;
            tz -= rgtZ * 0.72;
          } else {
            tx += rgtX * 0.72;
            tz += rgtZ * 0.72;
          }
        }

        const hTang = Math.hypot(tx, tz);
        if (hTang < 0.42 && Math.abs(ny) < 0.88) {
          const latScale = 0.84 - hTang;
          tx += -nz * latScale;
          tz += nx * latScale;
        }
        if (preferUpVault && !insideEnclosedRoof && ny > -0.15) {
          ty += ny > 0.12 ? 0.62 : 0.30;
        }
        let tLen = Math.hypot(tx, ty, tz);
        if (tLen < 0.25) {
          tx = -nz;
          ty = !insideEnclosedRoof && Math.abs(ny) < 0.85 ? 0.55 : 0.0;
          tz = nx;
          tLen = Math.hypot(tx, ty, tz) || 1.0;
        }
        const bypassAccel = urgency * Math.max(19.5, refSpeed * 1.25);
        const redirectSpd = !insideEnclosedRoof ? removedInwardSpd * 0.78 : 0.0;
        vel.x += (tx / tLen) * redirectSpd + (tx / tLen + nx * 0.48) * bypassAccel * dt;
        vel.y += (ty / tLen) * redirectSpd + (ty / tLen + ny * 0.48) * bypassAccel * dt;
        vel.z += (tz / tLen) * redirectSpd + (tz / tLen + nz * 0.48) * bypassAccel * dt;
        const maxBypassSpd = Math.max(8.5, refSpeed * 1.18);
        const postSpd = Math.hypot(vel.x, vel.y, vel.z);
        if (postSpd > maxBypassSpd) {
          const sc = maxBypassSpd / postSpd;
          vel.x *= sc;
          vel.y *= sc;
          vel.z *= sc;
        }
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

    if (prevMode !== newMode) {
      this._exploreStick.moveX = 0;
      this._exploreStick.moveY = 0;
      this._exploreStick.climbY = 0;
      this._exploreStick.boost = false;
    }

    if (prevMode === 'DRONE_AVATA2_FPV' && newMode !== 'DRONE_AVATA2_FPV') {
      this._avataStick.moveX = 0;
      this._avataStick.moveY = 0;
      this._avataStick.lookX = 0;
      this._avataStick.lookY = 0;
      this._avataStick.climbY = 0;
      this._avataStick.boost = false;
      this._avataStunt.active = false;
      this._avataFreelookYaw = 0.0;
      this._avataFreelookPitch = 0.0;
      this._avataCurrentBaseFov = 48.0;
    }

    if (wasFPS && !nextIsFPS) {
      if (this.sharedUniforms && this.sharedUniforms.uCockpitGlassCutRadius) {
        this.sharedUniforms.uCockpitGlassCutRadius.value = 0.0;
      }
      this.lastFPSStatus = null;
      this.lastFPSSubjectKind = null;
      this._fpsPoseInitialized = false;
    } else if (prevMode !== newMode) {
      this.lastFPSStatus = null;
      this.lastFPSSubjectKind = null;
    }

    this.mode = newMode;
    this.transitionActive = false;
    this._flightNadirSpec = null;
    this.camera.up.set(0, 1, 0);
    this._orbitPanVel.set(0, 0, 0);
    this._applyBaseVerticalFov(this.getActiveBaseVerticalFov(newMode));

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
      if (newMode === 'AUTO_CINEMATIC' && this.camera.position.y < 16.0) {
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
      if (Number.isFinite(options.seed)) {
        this._mavicSeedPhase = ((Math.trunc(options.seed) >>> 0) % 1000) * 0.00628318;
      }
      if (options.startIslandTour === true) {
        this._evalMavicIslandOrbit(this.cinematicAngle, null, this.camera.position, this.goalTarget);
        const lookDx = this.goalTarget.x - this.camera.position.x;
        const lookDz = this.goalTarget.z - this.camera.position.z;
        this.yaw = Math.atan2(-lookDx, -lookDz);
        const rawP = Math.atan2(this.goalTarget.y - this.camera.position.y, Math.hypot(lookDx, lookDz) || 1.0);
        this.pitch = Math.max(-0.32, Math.min(0.14, rawP));
        this.roll = 0.022;
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
      this._mavicGimbalIdle = 0.0;
      if (options.startIslandTour === true) {
        this._mavicCinemaFov = 43.0;
        this._applyBaseVerticalFov(43.0);
      }
    } else if (newMode === 'DRONE_AVATA2_FPV') {
      if (this.controls) this.controls.enabled = false;
      this._avataYawRateAccum = 0.0;
      this._avataYawRateSmooth = 0.0;
      this._avataGForce = 1.0;
      this._avataStunt.active = false;
      this._avataFreelookYaw = 0.0;
      this._avataFreelookPitch = 0.0;
      this._avataCurrentBaseFov = 51.5;
      this._applyBaseVerticalFov(51.5);
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
          this.camera.position.set(-12.0, 14.0, 86.0);
          this.yaw = Math.atan2(-12.0, 86.0);
          this.pitch = 0.14;
        }
        this.velocity.set(0, 0, 0);
        this._avataEuler.set(this.pitch, this.yaw, this.roll, 'YXZ');
        this.camera.quaternion.setFromEuler(this._avataEuler);
        this.syncPreviousCameraPosition();
      } else {
        this.avataAutoFreestyle = true;
        this._avataLiveLaneOffset = 0.0;
        this._avataRejoinBlend = 0.0;
        this._initAvataProPilotStream(options);
        this._evalAvataStreamPoint(this._avataStreamCursor, this.camera.position, this._avataMeta);
        this.goalPos.copy(this.camera.position);
        this._evalAvataStreamPoint(this._avataStreamCursor + 0.04, this._avataTmpP1);
        const dx = this._avataTmpP1.x - this.camera.position.x;
        const dy = this._avataTmpP1.y - this.camera.position.y;
        const dz = this._avataTmpP1.z - this.camera.position.z;
        const hLen = Math.hypot(dx, dz) || 1.0;
        this.yaw = Math.atan2(-dx, -dz);
        this.pitch = Math.atan2(dy, hLen);
        this.roll = -0.18;
        const initDirLen = Math.hypot(dx, dy, dz) || 1.0;
        const initSpd = this._avataMeta.speedMs || 11.5;
        this.velocity.set((dx / initDirLen) * initSpd, (dy / initDirLen) * initSpd, (dz / initDirLen) * initSpd);
        this._avataVelPrev.copy(this.velocity);
        this._avataEuler.set(this.pitch, this.yaw, this.roll, 'YXZ');
        this.camera.quaternion.setFromEuler(this._avataEuler);
        this.camera.up.set(0, 1, 0).applyQuaternion(this.camera.quaternion);
        this.syncPreviousCameraPosition();
      }
      if (this.camera && typeof this.camera.updateProjectionMatrix === 'function') {
        this.camera.near = 0.14;
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

  lookDelta(dx, dy, sensitivity = 0.0024, pointerLocked = false) {
    if (this.transitionActive && (Math.abs(dx) > 1.5 || Math.abs(dy) > 1.5)) {
      this.cancelTransition();
    }
    if (this.isFPSMode()) {
      this.fpsYawOffset = Math.max(-2.05, Math.min(2.05, this.fpsYawOffset - dx * sensitivity));
      this.fpsPitchOffset = Math.max(-1.12, Math.min(1.12, this.fpsPitchOffset - dy * sensitivity));
      return;
    }
    if (this.mode === 'DRONE_AVATA2_FPV') {
      if (this.avataAutoFreestyle && !pointerLocked) {
        // Mouse drag in Auto Pro-Pilot pans gimbal/head-look WITHOUT aborting auto-flight into an obstacle!
        this._avataFreelookYaw = Math.max(-1.45, Math.min(1.45, this._avataFreelookYaw - dx * sensitivity));
        this._avataFreelookPitch = Math.max(-0.78, Math.min(0.85, this._avataFreelookPitch - dy * sensitivity));
        this._avataFreelookIdle = 0.0;
        return;
      }
      if (this.avataAutoFreestyle && pointerLocked && Math.hypot(dx, dy) > 3.5) {
        this.avataAutoFreestyle = false;
        this._avataRejoinBlend = 0.0;
      }
      const dYaw = -dx * sensitivity * 1.05;
      this.yaw += dYaw;
      this._avataYawRateAccum += dYaw;
      const maxDownPitch = this.avataFlightProfile === 'ACRO' ? -1.48 : -1.28;
      const maxUpPitch = this.avataFlightProfile === 'ACRO' ? 1.38 : 1.18;
      this.pitch = Math.max(maxDownPitch, Math.min(maxUpPitch, this.pitch - dy * sensitivity * 1.02));
      return;
    }
    if (this.mode !== 'DRONE_6DOF' && this.mode !== 'WALK_SIVITAS') return;
    if (this.mode === 'DRONE_6DOF' && !this._droneManualEngaged) {
      this._mavicGimbalYaw = Math.max(-2.6, Math.min(2.6, this._mavicGimbalYaw - dx * sensitivity));
      this._mavicGimbalPitch = Math.max(-0.95, Math.min(0.85, this._mavicGimbalPitch - dy * sensitivity));
      this._mavicGimbalIdle = 0.0;
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
        this.lastFPSSubjectKind = anchor.subjectKind || null;
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
      this._applyBaseVerticalFov(this.getActiveBaseVerticalFov('AUTO_CINEMATIC'));
      this.cinematicAngle += clampedDt * 0.042;
      this._evalMavicIslandOrbit(this.cinematicAngle, floorHeightFn, this.goalPos, this.goalTarget);
      const stepDx = (this.goalPos.x - this.camera.position.x) * dampAlpha;
      const stepDy = (this.goalPos.y - this.camera.position.y) * dampAlpha;
      const stepDz = (this.goalPos.z - this.camera.position.z) * dampAlpha;
      const stepLen = Math.hypot(stepDx, stepDy, stepDz);
      const maxAutoStep = 13.8 * clampedDt;
      const stepScale = stepLen > maxAutoStep ? maxAutoStep / stepLen : 1.0;
      this.camera.position.x += stepDx * stepScale;
      this.camera.position.y += stepDy * stepScale;
      this.camera.position.z += stepDz * stepScale;
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
    this._applyBaseVerticalFov(this.getActiveBaseVerticalFov(this.mode));
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
          0.72,
          0.72,
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
    const mavicBoost =
      !isWalk &&
      (this.keyState.has('ShiftLeft') || this.keyState.has('ShiftRight') || Boolean(this._exploreStick.boost));
    const maxSpeed = isWalk ? (this._exploreStick.boost ? 8.8 : 6.2) : mavicBoost ? 17.5 : 11.5;
    const accel = isWalk ? 32.0 : mavicBoost ? 88.0 : 58.0;

    this._forward.set(-Math.sin(this.yaw), isWalk ? 0 : Math.sin(this.pitch), -Math.cos(this.yaw)).normalize();
    this._right.set(Math.cos(this.yaw), 0, -Math.sin(this.yaw)).normalize();

    let keyF = 0;
    let keyR = 0;
    let keyU = 0;
    if (this.keyState.has('KeyW') || this.keyState.has('ArrowUp')) keyF += 1;
    if (this.keyState.has('KeyS') || this.keyState.has('ArrowDown')) keyF -= 1;
    if (this.keyState.has('KeyD') || this.keyState.has('ArrowRight')) keyR += 1;
    if (this.keyState.has('KeyA') || this.keyState.has('ArrowLeft')) keyR -= 1;
    if (!isWalk) {
      if (this.keyState.has('KeyE') || this.keyState.has('Space')) keyU += 1;
      if (
        this.keyState.has('KeyQ') ||
        this.keyState.has('KeyC') ||
        this.keyState.has('ControlLeft') ||
        this.keyState.has('ControlRight')
      ) {
        keyU -= 1;
      }
    }

    const moveF = Math.max(-1, Math.min(1, keyF + this._exploreStick.moveY));
    const moveR = Math.max(-1, Math.min(1, keyR + this._exploreStick.moveX));
    const moveU = isWalk ? 0 : Math.max(-1, Math.min(1, keyU + this._exploreStick.climbY));

    // Mavic 3 Pro Cinema Island-Circling Tour in DRONE_6DOF before manual drone keys/sticks are engaged
    const hasManualInput = Math.abs(moveF) > 0.04 || Math.abs(moveR) > 0.04 || Math.abs(moveU) > 0.04;
    if (hasManualInput) {
      this._droneManualEngaged = true;
    }
    if (!isWalk && !hasManualInput && !this._droneManualEngaged && this.camera.position.y > 2.0) {
      this._mavicGimbalIdle += clampedDt;
      if (this._mavicGimbalIdle > 4.2) {
        const gDecay = Math.exp(-1.4 * clampedDt);
        this._mavicGimbalYaw *= gDecay;
        this._mavicGimbalPitch *= gDecay;
      }
      this.velocity.set(0, 0, 0);
      this.cinematicAngle += clampedDt * 0.042;
      this._evalMavicIslandOrbit(this.cinematicAngle, floorHeightFn, this.goalPos, this.goalTarget);
      const mavAlpha = 1.0 - Math.exp(-4.6 * clampedDt);
      const mDx = (this.goalPos.x - this.camera.position.x) * mavAlpha;
      const mDy = (this.goalPos.y - this.camera.position.y) * mavAlpha;
      const mDz = (this.goalPos.z - this.camera.position.z) * mavAlpha;
      const mLen = Math.hypot(mDx, mDy, mDz);
      const maxMavStep = 13.8 * clampedDt;
      const mScale = mLen > maxMavStep ? maxMavStep / mLen : 1.0;
      this.camera.position.x += mDx * mScale;
      this.camera.position.y += mDy * mScale;
      this.camera.position.z += mDz * mScale;
      const lookDx = this.goalTarget.x - this.camera.position.x;
      const lookDz = this.goalTarget.z - this.camera.position.z;
      const targetMavYaw = Math.atan2(-lookDx, -lookDz) + this._mavicGimbalYaw;
      let dyMav = targetMavYaw - this.yaw;
      while (dyMav > Math.PI) dyMav -= Math.PI * 2.0;
      while (dyMav < -Math.PI) dyMav += Math.PI * 2.0;
      this.yaw += dyMav * dampAlpha;
      const basePitch = Math.max(
        -0.32,
        Math.min(0.14, Math.atan2(this.goalTarget.y - this.camera.position.y, Math.hypot(lookDx, lookDz) || 1.0))
      );
      const targetMavPitch = Math.max(-1.35, Math.min(1.15, basePitch + this._mavicGimbalPitch));
      this.pitch += (targetMavPitch - this.pitch) * dampAlpha;
      this.roll += (0.028 * Math.cos(this.cinematicAngle) - this.roll) * dampAlpha;
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
        this._applyDroneAPASAvoidance(this.camera.position, this.velocity, clampedDt, 0.72, ctx, cmdMavicSpd, true);
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
        ? groundY + 0.72
        : Math.max(0.72, groundY + 0.72);

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
        0.72,
        minCamY,
        clampedDt,
        this._prevCamPos,
        ctx,
        false,
        true
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
      // Explicit flight stick / WASD input seamlessly hands over from Pro-Pilot AI -> 100% Manual Free-Flight!
      this.avataAutoFreestyle = false;
      this._avataRejoinBlend = 0.0;
      this._avataFreelookYaw = 0.0;
      this._avataFreelookPitch = 0.0;
    }

    const oceanY = Number.isFinite(ctx?.oceanWaterY) ? ctx.oceanWaterY : -1.45;
    this._avataApasTimer = Math.max(0.0, this._avataApasTimer - clampedDt);

    if (this.avataAutoFreestyle) {
      this._avataYawRateAccum = 0.0;
      this._avataFreelookIdle += clampedDt;
      if (this._avataFreelookIdle > 1.8) {
        const flDecay = Math.exp(-2.6 * clampedDt);
        this._avataFreelookYaw *= flDecay;
        this._avataFreelookPitch *= flDecay;
      }

      this._ensureAvataStreamAhead();
      const epsC = 0.035;
      this._evalAvataStreamPoint(this._avataStreamCursor, this._avataTmpP1, this._avataMeta);
      this._evalAvataStreamPoint(this._avataStreamCursor - epsC, this._avataTmpP0);
      this._evalAvataStreamPoint(this._avataStreamCursor + epsC, this._avataTmpP2);
      const travelSignZ = this._avataTmpP2.z - this._avataTmpP0.z >= 0.0 ? 1.0 : -1.0;

      // Bidirectional Dynamic Obstacle & Traffic Lane-Shifting inside Subsea Underpass & Lorong Jembatan Teksas!
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

      if (camZ >= 250.0 && camZ <= 688.0 && camX < -305.0 && camY < 4.5) {
        // Bidirectional Subsea Underpass Bus Kuning UI Dodge (works Northbound & Southbound!)
        desiredLane = 0.92 * Math.sin(this._avataStreamCursor * 0.85);
        if (dynList) {
          let blockedWest = false;
          let blockedEast = false;
          let nearestVehicleZ = 78.0;
          let singleDodgeLane = 0.0;
          for (let i = 0; i < dynList.length; i++) {
            const dc = dynList[i];
            if (!dc || dc.enabled === false) continue;
            if (dc.minY !== undefined && (dc.minY > camY + 2.8 || dc.maxY < camY - 2.8)) continue;
            const isObb = (dc.kind || dc.type) === 'OBB';
            const dcX = isObb ? dc.cx : Number.isFinite(dc.x) ? dc.x : 0.5 * (dc.minX + dc.maxX);
            if (!isObb && Math.abs(dcX - -356.0) > 3.05) continue;
            const dcZ = isObb ? dc.cz : Number.isFinite(dc.z) ? dc.z : 0.5 * (dc.minZ + dc.maxZ);
            const dzAhead = (dcZ - camZ) * travelSignZ;
            const dzAbs = Math.abs(dcZ - camZ);
            if (
              Math.abs(dcX - -356.0) < 3.6 &&
              dcZ >= 260.0 &&
              dcZ <= 680.0 &&
              dzAhead >= -9.5 &&
              dzAhead <= 48.0
            ) {
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
            // Both lanes occupied simultaneously: thread cleanly above the -5.22m bus roof inside the -4.05m crown!
            if (camZ >= 321.0 && camZ <= 629.0) {
              desiredLane = 0.0;
              underpassOverBusLift = 1.42;
            } else {
              desiredLane = singleDodgeLane;
              underpassOverBusLift = 2.35;
            }
            this._avataApasTimer = 0.35;
          }
        }
      } else if (camZ >= 142.0 && camZ <= 348.0 && Math.abs(camX - -28.0) < 9.0 && camY > 1.8) {
        // Bidirectional Lorong Jembatan Teksas UI: center-locked with live pedestrian lateral courtesy shift
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
            const dzAhead = (dcZ - camZ) * travelSignZ;
            const dzAbs = Math.abs(dcZ - camZ);
            if (
              Math.abs(dcX - -28.0) < 2.45 &&
              dcZ >= 149.0 &&
              dcZ <= 342.0 &&
              dzAhead >= -6.0 &&
              dzAhead <= 22.0 &&
              dzAbs < nearestNpcZ
            ) {
              nearestNpcZ = dzAbs;
              desiredLane = dcX >= -28.0 ? -1.12 : 1.12;
              this._avataApasTimer = 0.28;
            }
          }
        }
      }
      this._avataLiveLaneOffset += (desiredLane - this._avataLiveLaneOffset) * (1.0 - Math.exp(-12.5 * clampedDt));

      const dArcPerUnit = Math.max(1.2, this._avataTmpP2.distanceTo(this._avataTmpP0) / (2.0 * epsC));
      const lagDist =
        this._avataRejoinBlend > 0.002 ? 0.0 : this.camera.position.distanceTo(this.goalPos);
      const leashScale = lagDist > 7.0 ? Math.max(0.25, 1.0 - (lagDist - 7.0) / 13.0) : 1.0;

      const targetSpeed = this._avataMeta.speedMs;
      const effTargetSpeed = targetSpeed * leashScale;
      this._avataStreamCursor += (effTargetSpeed * clampedDt) / dArcPerUnit;
      this._ensureAvataStreamAhead();
      this._avataSplineT = (this._avataStreamCursor / 42.0) % 1.0;

      this._evalAvataStreamPoint(this._avataStreamCursor, this.goalPos, this._avataMeta);
      this._evalAvataStreamPoint(this._avataStreamCursor - epsC, this._avataTmpP0);
      this._evalAvataStreamPoint(this._avataStreamCursor + epsC, this._avataTmpP2);
      if (this._avataMeta.maneuverId) {
        this._avataActiveManeuverId = this._avataMeta.maneuverId;
      }

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

      const underpassCutGoalMask =
        (1.0 - PhysicsKernel.smootherstep(8.0, 16.0, Math.abs(this.goalPos.x - -356.0))) *
        PhysicsKernel.smootherstep(262.0, 274.0, this.goalPos.z) *
        (1.0 - PhysicsKernel.smootherstep(668.0, 680.0, this.goalPos.z));
      const effOpenGoal = openFactor * (1.0 - underpassCutGoalMask);
      if (effOpenGoal > 1e-3 && ctx && strictFactor < 0.35) {
        // 3-pass 3D outer-leaf/roof repulsion lift across open-air segments!
        const rep = this._avataRepelScratch;
        for (let pass = 0; pass < 3; pass++) {
          const penGoal = PhysicsKernel.probeDroneAPASRepulsion(
            this.goalPos.x,
            this.goalPos.y,
            this.goalPos.z,
            this.camera.position.x,
            this.camera.position.y,
            this.camera.position.z,
            2.65,
            ctx,
            rep
          );
          if (penGoal <= 0.0) break;
          this.goalPos.x += rep.nx * penGoal * 1.15 * effOpenGoal;
          this.goalPos.y += Math.max(0.45, rep.ny) * penGoal * 1.25 * effOpenGoal;
          this.goalPos.z += rep.nz * penGoal * 1.15 * effOpenGoal;
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
        const errX = this.goalPos.x - this.camera.position.x;
        const errY = this.goalPos.y - this.camera.position.y;
        const errZ = this.goalPos.z - this.camera.position.z;
        const gainK = 8.8 + strictFactor * 11.5;
        this.velocity.set(tanVx + errX * gainK, tanVy + errY * gainK, tanVz + errZ * gainK);
        const maxPursuitSpd = targetSpeed * 1.38;
        if (this.velocity.length() > maxPursuitSpd) {
          this.velocity.setLength(maxPursuitSpd);
        }
        if (
          strictFactor < 0.45 &&
          wArchSkim < 0.35 &&
          underpassCutGoalMask < 0.35 &&
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
      const scaleA = Math.pow(targetSpeed / dArcPerUnit, 2) / (epsC * epsC);
      const ax = (this._avataTmpP2.x - 2.0 * this.goalPos.x + this._avataTmpP0.x) * scaleA;
      const ay = (this._avataTmpP2.y - 2.0 * this.goalPos.y + this._avataTmpP0.y) * scaleA;
      const az = (this._avataTmpP2.z - 2.0 * this.goalPos.z + this._avataTmpP0.z) * scaleA;

      const tanYaw = Math.atan2(-tx, -tz);
      const horizLen = Math.hypot(tx, tz) || 1.0;
      const tanPitch = Math.max(-1.20, Math.min(1.10, Math.atan2(ty, horizLen)));

      // Heroic Monumental Scale Gaze: blend flight tangent with upward gaze toward 50m Rektorat / STP-UI / 60m Volcano!
      let targetYaw = tanYaw;
      let targetPitch = tanPitch;
      const lookBlend = this._avataMeta.lookBlend || 0.0;
      if (lookBlend > 0.01 && corrWeight < 0.15) {
        const lx = this._avataMeta.lookX - this.camera.position.x;
        const ly = this._avataMeta.lookY - this.camera.position.y;
        const lz = this._avataMeta.lookZ - this.camera.position.z;
        const lHoriz = Math.hypot(lx, lz);
        if (lHoriz > 6.0) {
          const lmYaw = Math.atan2(-lx, -lz);
          let dyLm = lmYaw - tanYaw;
          while (dyLm > Math.PI) dyLm -= Math.PI * 2.0;
          while (dyLm < -Math.PI) dyLm += Math.PI * 2.0;
          dyLm = Math.max(-0.88, Math.min(0.88, dyLm));
          const lmPitch = Math.atan2(ly, lHoriz);
          const dpLm = Math.max(-0.26, Math.min(0.48, lmPitch - tanPitch));
          const bLm = Math.min(0.84, lookBlend);
          targetYaw = tanYaw + dyLm * bLm;
          targetPitch = Math.max(-1.20, Math.min(1.10, tanPitch + dpLm * bLm));
        }
      }

      let yawDiff = targetYaw - this.yaw;
      while (yawDiff > Math.PI) yawDiff -= Math.PI * 2.0;
      while (yawDiff < -Math.PI) yawDiff += Math.PI * 2.0;
      const rotAlpha = 1.0 - Math.exp((corrWeight > 0.25 ? -15.5 : -9.5) * clampedDt);
      this.yaw += yawDiff * rotAlpha;
      this._avataYawRateSmooth = (yawDiff * rotAlpha) / clampedDt;
      this.pitch += (targetPitch - this.pitch) * rotAlpha;

      const rx = Math.cos(this.yaw);
      const rz = -Math.sin(this.yaw);
      const aDotRight = ax * rx + az * rz;
      const maxAutoBank = corrType === 1 ? 0.20 : corrType === 2 ? 0.34 : 0.72;
      const targetBank = Math.max(
        -maxAutoBank,
        Math.min(maxAutoBank, -Math.atan2(aDotRight * 1.25, 9.81 + Math.max(-5.5, ay)))
      );
      this.roll += (targetBank - this.roll) * (1.0 - Math.exp(-8.5 * clampedDt));
      const speedBoost = Math.max(0.0, (targetSpeed - 11.5) * 0.065);
      this._avataThrottle01 = Math.max(0.26, Math.min(0.95, 0.48 + (ty / tLen) * 0.32 + speedBoost));

      // Context-gated Pro-Pilot Acro Stunts strictly in designated high-clearance open airspace!
      this._avataAutoRollCooldown = Math.max(0.0, this._avataAutoRollCooldown - clampedDt);
      if (
        this._avataMeta.allowStunt > 0 &&
        corrWeight < 0.02 &&
        this.camera.position.y > 16.0 &&
        this._avataAutoRollCooldown <= 0.0 &&
        !this._avataStunt.active
      ) {
        const stunts = ['ROLL_RIGHT', 'ROLL_LEFT', 'SPLIT_S'];
        const chosenStunt = stunts[Math.floor(this._nextAvataRand01() * stunts.length)];
        if (this.triggerAvataStunt(chosenStunt)) {
          this._avataStuntIndex++;
          this._avataAutoRollCooldown = 14.5 + this._nextAvataRand01() * 8.0;
        }
      }
    } else {
      // 100% UNRESTRICTED FREE-FLIGHT FPV CINEWHOOP DYNAMICS + WHISKER APAS 360° (Controlled Pro Speeds!)
      const stickYawDelta = -this._avataStick.lookX * 2.25 * clampedDt;
      const stickPitchDelta = this._avataStick.lookY * 1.65 * clampedDt;
      this.yaw += stickYawDelta;
      this._avataYawRateAccum += stickYawDelta;
      const maxDownPitch = isAcro ? -1.48 : -1.28;
      const maxUpPitch = isAcro ? 1.38 : 1.18;
      this.pitch = Math.max(maxDownPitch, Math.min(maxUpPitch, this.pitch + stickPitchDelta));

      const instYawRate = this._avataYawRateAccum / clampedDt;
      this._avataYawRateAccum = 0.0;
      const yawSmoothAlpha = 1.0 - Math.exp(-11.0 * clampedDt);
      this._avataYawRateSmooth += (instYawRate - this._avataYawRateSmooth) * yawSmoothAlpha;

      const cosP = Math.cos(this.pitch);
      this._forward.set(-Math.sin(this.yaw) * cosP, Math.sin(this.pitch), -Math.cos(this.yaw) * cosP).normalize();
      this._right.set(Math.cos(this.yaw), 0.0, -Math.sin(this.yaw)).normalize();

      // Realistic Pro Cinewhoop Speeds: 10.8 m/s (39 km/h) Normal Cruise, 16.8 m/s (60 km/h) Shift/Boost!
      const maxSpeed = turbo ? 16.8 : 10.8;
      const surgeAccel = (turbo ? 34.0 : 22.0) * (surge < 0 ? 0.72 : 1.0);
      const strafeAccel = turbo ? 26.0 : 18.0;
      const climbAccel = turbo ? 24.0 : 17.0;

      this.velocity.addScaledVector(this._forward, surge * surgeAccel * clampedDt);
      this.velocity.addScaledVector(this._right, strafe * strafeAccel * clampedDt);
      this.velocity.y += climb * climbAccel * clampedDt;

      // Authentic FPV aerodynamic drag + responsive air-brake when sticks center
      const thrustActive = Math.abs(surge) > 0.04 || Math.abs(strafe) > 0.04 || Math.abs(climb) > 0.04;
      const linDrag = Math.exp(-(thrustActive ? 1.75 : 3.85) * clampedDt);
      this.velocity.multiplyScalar(linDrag);
      const curSpd = this.velocity.length();
      if (curSpd > maxSpeed) {
        this.velocity.setLength(maxSpeed);
      } else if (!thrustActive && curSpd < 0.04) {
        this.velocity.set(0, 0, 0);
      }

      const manualCmdSpd = thrustActive ? maxSpeed : 0.0;
      if (this._applyDroneAPASAvoidance(this.camera.position, this.velocity, clampedDt, 0.50, ctx, manualCmdSpd, true)) {
        this._avataApasTimer = 0.32;
      }

      this.camera.position.addScaledVector(this.velocity, clampedDt);

      // Coordinated Yaw-Roll Banking + Lateral Strafe Bank + Direct Q/E Acro Roll
      const horizSpd = Math.hypot(this.velocity.x, this.velocity.z);
      const yawBank = Math.atan2(this._avataYawRateSmooth * Math.max(4.5, horizSpd) * 0.88, 9.81);
      const strafeBank = -strafe * (isAcro ? 0.78 : 0.56);

      if (isAcro && Math.abs(rollCmd) > 0.05) {
        this.roll += rollCmd * 5.15 * clampedDt;
        if (this.roll > Math.PI) this.roll -= Math.PI * 2.0;
        if (this.roll < -Math.PI) this.roll += Math.PI * 2.0;
      } else {
        const maxBank = isAcro ? 1.24 : 0.84;
        const directBank = rollCmd * 0.84;
        const targetRoll = Math.max(-maxBank, Math.min(maxBank, yawBank + strafeBank + directBank));
        while (this.roll - targetRoll > Math.PI) this.roll -= Math.PI * 2.0;
        while (this.roll - targetRoll < -Math.PI) this.roll += Math.PI * 2.0;
        this.roll += (targetRoll - this.roll) * (1.0 - Math.exp(-9.5 * clampedDt));
      }

      const cmdMag = Math.min(1.0, Math.hypot(surge, strafe, climb));
      this._avataThrottle01 = Math.max(
        0.18,
        Math.min(1.0, 0.22 + cmdMag * (turbo ? 0.76 : 0.56) + (curSpd / 16.8) * 0.22)
      );
    }

    // Hard 3D Solid-Body Non-Penetration (0.50m exterior guard / 0.42m inside enclosed hollow bridge & underpass)
    const walkY = floorHeightFn ? floorHeightFn(this.camera.position.x, this.camera.position.z) : 0.0;
    const minAvataY =
      walkY < -0.5 ? walkY + 0.50 : Math.max(oceanY + 0.50, walkY + 0.50);
    const inHollowCorridor =
      (Math.abs(this.camera.position.x - -28.0) < 2.62 &&
        this.camera.position.z >= 148.5 &&
        this.camera.position.z <= 342.5) ||
      (Math.abs(this.camera.position.x - -356.0) < 4.85 &&
        this.camera.position.z >= 268.0 &&
        this.camera.position.z <= 674.0 &&
        this.camera.position.y < 3.0);
    const avataColRadius = inHollowCorridor ? 0.42 : 0.50;
    resolveSoftSpatialCollision(
      this.camera.position,
      this.velocity,
      this.colliders,
      avataColRadius,
      minAvataY,
      clampedDt,
      this._prevCamPos,
      ctx,
      false,
      true
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

    // Monumental Cinema FPV Lens (51.5° -> 56.5°): preserves towering architectural & volcanic scale without fisheye miniaturization!
    const speedNorm = Math.min(1.0, this.velocity.length() / 16.8);
    const desiredBaseFov = 51.5 + speedNorm * 5.0;
    this._avataCurrentBaseFov += (desiredBaseFov - this._avataCurrentBaseFov) * (1.0 - Math.exp(-7.5 * clampedDt));
    this._applyBaseVerticalFov(this._avataCurrentBaseFov);

    if (this.camera && typeof this.camera.updateProjectionMatrix === 'function') {
      const camY = Math.max(0.0, Number(this.camera.position.y) || 0.0);
      const desiredNear = camY < 32.0 ? 0.14 : Math.min(1.6, 0.14 + (camY - 32.0) * 0.003);
      if (Math.abs(this.camera.near - desiredNear) > 0.02) {
        this.camera.near = desiredNear;
        this.camera.updateProjectionMatrix();
      }
    }

    this._prevCamPos.copy(this.camera.position);
    const freeYaw = this.avataAutoFreestyle ? this._avataFreelookYaw : 0.0;
    const freePitch = this.avataAutoFreestyle ? this._avataFreelookPitch : 0.0;
    this._avataEuler.set(this.pitch + freePitch + stuntPitchAdd, this.yaw + freeYaw, this.roll + stuntRollAdd, 'YXZ');
    this.camera.quaternion.setFromEuler(this._avataEuler);
    this.camera.up.set(0, 1, 0).applyQuaternion(this.camera.quaternion);
  }
}
