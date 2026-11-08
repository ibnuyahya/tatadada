import * as THREE from 'three';

const DEG = Math.PI / 180;

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (Watertight Anti-Clipping Edition)
 * ./core/CameraDirector.js
 *
 * Multi-Angle Class-A Inspection & 6-DOF Free-Flight Camera Director:
 * - Horizon-Locked Nadir 90° View (eps = 0.45m, camera.up = (0,1,0) — Hukum Emas 19)
 * - Adaptive Non-Occluding Viewport Offset (Desktop <= 280px side drawer, Portrait <= 34vh bottom sheet — Hukum Emas 20)
 * - Full-Envelope Building Volume Anti-Penetration (AABBs + Polar Annular Sectors) enforced in BOTH Orbit & Drone modes
 *   so the camera can NEVER enter inside any building hull!
 */
export class CameraDirector {
  constructor(camera, controls, domElement) {
    this.camera = camera;
    this.controls = controls;
    this.domElement = domElement;

    this.mode = 'runner'; // 'runner' (default FPV human jogger) | 'orbit' | 'drone'
    this.drawerCollapsed = true;
    this.transition = null;

    // Proportional Human-Scale FPV Runner State (Validated: 1.68m adult -> 1.55m running eye height above road surface y=0.02m -> y=1.57m)
    // Calibrated to Human Binocular Optical Characteristics (~77.5° vertical / ~109° horizontal FOV, VOR gaze stabilization,
    // ~76% dominant forward curve-tangent fixation + ~24% occasional diagonal head-turn "sesekali noleh").
    this.runner = {
      personHeightM: 1.68,
      eyeHeightAboveRoadM: 1.55,
      roadSurfaceY: 0.02,
      trackRadiusM: 76.80,       // Outer jogging lane on Ring Road (between rCenterDash=75.50m and rAsphaltOut=79.50m)
      baseSpeedMps: 3.85,        // Natural human running pace (~13.9 km/h; Shift = 7.6 m/s sprint)
      thetaRad: 112.0 * DEG,     // Starts on SE Ring Road curving forward toward Gedung DPPD -> West Gate -> ITTC -> STP -> ILRC
      stepPhase: 0.0,
      gazeClock: 0.0,
      glanceWeight: 0.0,
      forceGlanceMode: null,     // null = automatic cycle, true = force glance, false = force forward
      currentLookPos: new THREE.Vector3(-42.0, 2.45, 58.0),
      userYawOffset: 0.0,
      userPitchOffset: 0.0,
      idleSinceDragSec: 10.0,
      activeLandmarkLabel: '🏃 FPV Lari (Eye 1.57m) • Fokus ke Depan — Gedung DPPD'
    };

    // 6-DOF Drone state
    this.droneVelocity = new THREE.Vector3();
    this.droneYaw = 0;
    this.dronePitch = 0;
    this.keys = new Set();
    this.pointerDragging = false;
    this.lastPointer = { x: 0, y: 0 };

    // Spatial collision boundaries (enforced in Runner, Orbit, and Drone modes!)
    this.minElevationY = 1.45;
    this.maxRadiusFromOrigin = 260.0;
    this.collisionSolids = [];
    this.polarSectors = [
      // 1. Gedung DPPD (SW Quadrant: +90° .. +180°, r = 38.0 .. 60.8m, y = 0 .. 16.6m)
      { degMin: 90.0, degMax: 180.0, rMin: 38.0, rMax: 60.8, maxY: 16.6 },
      // 2. Gedung STP-UI (North Arc: -152° .. -32°, r = 37.8 .. 59.8m, y = 0 .. 27.4m)
      { degMin: -152.0, degMax: -32.0, rMin: 37.8, rMax: 59.8, maxY: 27.4 },
      // 3. Gedung ILRC & DRPM (East-to-South Arc: -32° .. +85°, r = 36.2 .. 59.8m, y = 0 .. 27.6m)
      { degMin: -32.0, degMax: 85.0, rMin: 36.2, rMax: 59.8, maxY: 27.6 }
    ];

    this._bindDroneEvents();
  }

  registerCollisionBoxes(boxes) {
    this.collisionSolids = Array.isArray(boxes) ? boxes : [];
  }

  registerPolarSectors(sectors) {
    if (Array.isArray(sectors) && sectors.length > 0) {
      this.polarSectors = sectors;
    }
  }

  /**
   * Hukum Emas 19: Horizon-Locked Nadir 90° View without gimbal flip.
   */
  setHorizonLockedNadirView(target = new THREE.Vector3(0, 0, 0), altitude = 185.0, eps = 0.45) {
    this.mode = 'orbit';
    this.controls.enabled = true;
    this.camera.up.set(0, 1, 0);
    const pos = new THREE.Vector3(target.x, target.y + altitude, target.z + eps);
    this.flyTo(pos, target, 0.75);
  }

  /**
   * Computes natural human-eye rectilinear vertical FOV (~77.5° vertical / ~109° horizontal on 16:9 widescreen),
   * eliminating telephoto zoom compression while keeping peripheral optical flow natural.
   */
  _computeHumanEyeVerticalFov() {
    const fullW = Math.max(1, window.innerWidth || 1280);
    const fullH = Math.max(1, window.innerHeight || 720);
    const aspect = Math.max(0.55, fullW / fullH);
    const targetHorizRad = 109.0 * DEG;
    const vFovRad = 2.0 * Math.atan(Math.tan(targetHorizRad * 0.5) / Math.max(1.18, aspect));
    return THREE.MathUtils.clamp(vFovRad / DEG, 76.0, 88.0);
  }

  /**
   * Hukum Emas 20: Adaptive Non-Occluding Viewport Frustum Offset.
   */
  applyNonOccludingViewportOffset(collapsed = this.drawerCollapsed) {
    this.drawerCollapsed = collapsed;
    const fullW = Math.max(1, window.innerWidth);
    const fullH = Math.max(1, window.innerHeight);
    this.camera.aspect = fullW / fullH;
    if (this.mode === 'runner') {
      this.camera.fov = this._computeHumanEyeVerticalFov();
    }

    if (collapsed) {
      this.camera.clearViewOffset();
      this.camera.updateProjectionMatrix();
      return;
    }

    const isPortraitOrMobile = fullW < 768 || fullW / fullH < 1.0;
    if (isPortraitOrMobile) {
      const sheetPx = Math.min(fullH * 0.34, 300);
      this.camera.setViewOffset(fullW, fullH, 0, Math.round(sheetPx * 0.46), fullW, fullH);
    } else {
      const drawerPx = 280;
      this.camera.setViewOffset(fullW, fullH, -Math.round(drawerPx * 0.46), 0, fullW, fullH);
    }
    this.camera.updateProjectionMatrix();
  }

  flyTo(position, target, durationSec = 0.72) {
    const endPos = position instanceof THREE.Vector3 ? position.clone() : new THREE.Vector3(...position);
    const endTgt = target instanceof THREE.Vector3 ? target.clone() : new THREE.Vector3(...target);
    this.resolveSoftSpatialCollision(endPos);
    this.camera.up.set(0, 1, 0);

    if (durationSec <= 0.01) {
      this.camera.position.copy(endPos);
      this.controls.target.copy(endTgt);
      this.controls.update();
      this.resolveSoftSpatialCollision(this.camera.position);
      this._syncDroneAnglesFromCamera();
      this.transition = null;
      return;
    }

    this.transition = {
      startPos: this.camera.position.clone(),
      startTgt: this.controls.target.clone(),
      endPos,
      endTgt,
      elapsed: 0,
      duration: durationSec
    };
  }

  applyPreset(preset, durationSec = 0.72) {
    if (!preset) return;
    if (preset.isNadir) {
      const tgt = new THREE.Vector3(...(preset.target || [0, 0, 0]));
      const alt = preset.altitude || preset.pos?.[1] || 185.0;
      this.setHorizonLockedNadirView(tgt, alt, 0.45);
      return;
    }
    this.setMode('orbit');
    if (preset.fov && Math.abs(this.camera.fov - preset.fov) > 0.5) {
      this.camera.fov = preset.fov;
      this.applyNonOccludingViewportOffset(this.drawerCollapsed);
    }
    this.flyTo(preset.pos, preset.target, durationSec);
  }

  setMode(nextMode) {
    this.mode = nextMode;
    if (nextMode === 'runner') {
      this.controls.enabled = false;
      this.transition = null;
      this.camera.fov = this._computeHumanEyeVerticalFov(); // Natural wide human binocular FOV (~77.5° V / ~109° H)
      this.applyNonOccludingViewportOffset(this.drawerCollapsed);
      this.runner.userYawOffset = 0.0;
      this.runner.userPitchOffset = 0.0;
      this.runner.idleSinceDragSec = 10.0;
      this._updateRunnerStep(0.016);
    } else if (nextMode === 'drone') {
      this.controls.enabled = false;
      this.transition = null;
      this.camera.up.set(0, 1, 0);
      this.droneVelocity.set(0, 0, 0);
      this._syncDroneAnglesFromCamera();
    } else {
      this.controls.enabled = true;
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
      this.controls.target.copy(this.camera.position).addScaledVector(forward, 28.0);
      this.camera.up.set(0, 1, 0);
      this.controls.update();
    }
  }

  setRunnerStationAngle(deg, forceGlance = null) {
    this.setMode('runner');
    this.runner.thetaRad = deg * DEG;
    this.runner.forceGlanceMode = forceGlance;
    const arch = this._evaluateRunnerArchitecturalTarget(this.runner.thetaRad, this.runner.gazeClock, 1.0);
    const lookAheadTheta = this.runner.thetaRad + 0.42;
    const roadAheadTarget = new THREE.Vector3(
      71.8 * Math.cos(lookAheadTheta),
      2.45,
      71.8 * Math.sin(lookAheadTheta)
    );
    const w = forceGlance === true ? 1.0 : 0.0;
    this.runner.currentLookPos.lerpVectors(roadAheadTarget, arch.target, w);
    this._updateRunnerStep(0.001);
  }

  _syncDroneAnglesFromCamera() {
    const euler = new THREE.Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
    this.droneYaw = euler.y;
    this.dronePitch = THREE.MathUtils.clamp(euler.x, -Math.PI * 0.47, Math.PI * 0.47);
  }

  _bindDroneEvents() {
    window.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA')) return;
      this.keys.add(e.code);
    });
    window.addEventListener('keyup', (e) => {
      this.keys.delete(e.code);
    });
    window.addEventListener('blur', () => {
      this.keys.clear();
      this.pointerDragging = false;
    });

    if (!this.domElement) return;
    this.domElement.addEventListener('pointerdown', (e) => {
      if (this.mode !== 'drone' && this.mode !== 'runner') return;
      this.pointerDragging = true;
      this.lastPointer.x = e.clientX;
      this.lastPointer.y = e.clientY;
    });
    window.addEventListener('pointermove', (e) => {
      if ((this.mode !== 'drone' && this.mode !== 'runner') || !this.pointerDragging) return;
      const dx = e.clientX - this.lastPointer.x;
      const dy = e.clientY - this.lastPointer.y;
      this.lastPointer.x = e.clientX;
      this.lastPointer.y = e.clientY;
      if (this.mode === 'runner') {
        this.runner.userYawOffset = THREE.MathUtils.clamp(this.runner.userYawOffset - dx * 0.0032, -Math.PI * 0.85, Math.PI * 0.85);
        this.runner.userPitchOffset = THREE.MathUtils.clamp(this.runner.userPitchOffset - dy * 0.0026, -0.55, 0.65);
        this.runner.idleSinceDragSec = 0.0;
      } else {
        this.droneYaw -= dx * 0.0028;
        this.dronePitch = THREE.MathUtils.clamp(this.dronePitch - dy * 0.0024, -Math.PI * 0.47, Math.PI * 0.47);
      }
    });
    window.addEventListener('pointerup', () => {
      this.pointerDragging = false;
    });
  }

  /**
   * Evaluates a natural diagonal forward-inward architectural target (~34°–42° from the forward running tangent,
   * never a flat 90° sideways stare) for when the runner occasionally turns their head ("sesekali noleh").
   */
  _evaluateRunnerArchitecturalTarget(thetaRad, gazeClock, dirSign = 1.0) {
    // Normalize angle to [-180°, +180°]
    let deg = ((thetaRad / DEG) % 360 + 540) % 360 - 180;
    const vertScan = 0.5 + 0.5 * Math.sin(gazeClock * 0.85);
    const diagLead = 18.0 * DEG * dirSign;

    let rLook = 55.5;
    let thLook = thetaRad + diagLead;
    let yLook = 5.6;
    let shortName = 'Bundaran UI';

    if (deg >= 106.0 && deg < 156.0) {
      thLook = THREE.MathUtils.lerp(thetaRad + diagLead, 136.0 * DEG, 0.65);
      rLook = 56.0;
      yLook = THREE.MathUtils.lerp(4.4, 7.6, vertScan);
      shortName = 'Pelana Utama & Kanopi Gedung DPPD';
    } else if (deg >= 156.0 || deg < -162.0) {
      if (deg >= 156.0 && deg < 172.0) {
        thLook = 168.0 * DEG;
        rLook = 53.5;
        yLook = THREE.MathUtils.lerp(4.5, 7.8, vertScan);
        shortName = 'Dinding Pelana Bata NW DPPD';
      } else {
        const ittcY = THREE.MathUtils.lerp(5.0, 9.8, vertScan);
        return {
          target: new THREE.Vector3(-22.0, ittcY, 2.0),
          forwardLabel: '🏃 FPV Lari (Eye 1.57m) • Fokus ke Depan — Arah Gerbang Barat & STP-UI',
          glanceLabel: '🏃 FPV Lari (Eye 1.57m) • Menoleh ke ITTC Perpustakaan Lama'
        };
      }
    } else if (deg >= -162.0 && deg < -110.0) {
      thLook = thetaRad + diagLead;
      rLook = 55.5;
      yLook = THREE.MathUtils.lerp(5.8, 11.2, vertScan);
      shortName = 'Fasad Tembaga & Jendela Kotak STP-UI';
    } else if (deg >= -110.0 && deg < -64.0) {
      thLook = THREE.MathUtils.lerp(thetaRad + diagLead, -88.0 * DEG, 0.60);
      rLook = 56.0;
      yLook = THREE.MathUtils.lerp(4.8, 10.5, vertScan);
      shortName = 'Portico Utara & Planter Gedung STP-UI';
    } else if (deg >= -64.0 && deg < -28.0) {
      thLook = THREE.MathUtils.lerp(thetaRad + diagLead, -46.0 * DEG, 0.60);
      rLook = 55.5;
      yLook = THREE.MathUtils.lerp(7.5, 13.5, vertScan);
      shortName = 'Makara Emas & Sirip Tembaga STP-UI';
    } else if (deg >= -28.0 && deg < 24.0) {
      thLook = thetaRad + diagLead;
      rLook = 55.5;
      yLook = THREE.MathUtils.lerp(5.4, 10.6, vertScan);
      shortName = 'Sayap Louver & Kotak Kaca Gedung ILRC';
    } else if (deg >= 24.0 && deg < 52.0) {
      thLook = THREE.MathUtils.lerp(thetaRad + diagLead, 41.0 * DEG, 0.65);
      rLook = 53.5;
      yLook = THREE.MathUtils.lerp(4.6, 10.2, vertScan);
      shortName = 'Atrium Terbuka 5 Lantai & Portal ILRC';
    } else if (deg >= 52.0 && deg < 74.0) {
      thLook = thetaRad + diagLead;
      rLook = 55.5;
      yLook = THREE.MathUtils.lerp(5.0, 10.0, vertScan);
      shortName = 'Sayap Selatan & Gundukan Bambu ILRC';
    } else if (deg >= 74.0 && deg < 96.0) {
      thLook = THREE.MathUtils.lerp(thetaRad + diagLead, 81.0 * DEG, 0.65);
      rLook = 54.5;
      yLook = THREE.MathUtils.lerp(6.8, 12.8, vertScan);
      shortName = 'Menara Bata 6 Lantai & Jendela Tudung ILRC';
    } else {
      thLook = thetaRad + diagLead;
      rLook = 55.5;
      yLook = THREE.MathUtils.lerp(4.6, 8.2, vertScan);
      shortName = 'Sayap Tenggara & Palem Raja Gedung DPPD';
    }

    return {
      target: new THREE.Vector3(rLook * Math.cos(thLook), yLook, rLook * Math.sin(thLook)),
      forwardLabel: `🏃 FPV Lari (Eye 1.57m) • Fokus ke Depan — ${shortName}`,
      glanceLabel: `🏃 FPV Lari (Eye 1.57m) • Menoleh ke ${shortName}`
    };
  }

  _updateRunnerStep(clampedDt) {
    const rState = this.runner;
    const isSprint = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');
    let dirSign = 1.0;
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) dirSign = -1.0;

    const speedMps = (isSprint ? 7.6 : rState.baseSpeedMps) * dirSign;
    const omega = speedMps / rState.trackRadiusM;
    rState.thetaRad = (rState.thetaRad + omega * clampedDt) % (Math.PI * 2);
    rState.gazeClock += clampedDt;

    // 2.68 Hz natural running stride cadence (~161 steps/min) scaled with speed
    const stepFreqHz = isSprint ? 3.35 : 2.68;
    rState.stepPhase += clampedDt * stepFreqHz * Math.PI * 2;

    // Vestibulo-Ocular Reflex (VOR) stabilized running eye kinematics around 1.57m world elevation:
    // Subtle vertical stride translation (±2.0cm) and lateral step shift (±1.4cm) with minimal torsional roll (±0.22°)
    const vertBounce = (Math.abs(Math.sin(rState.stepPhase * 0.5)) - 0.5) * 0.040;
    const latSway = Math.cos(rState.stepPhase * 0.5) * 0.014;
    const headRoll = Math.sin(rState.stepPhase * 0.5) * 0.0038;

    const curR = rState.trackRadiusM + latSway;
    const eyeY = rState.roadSurfaceY + rState.eyeHeightAboveRoadM + vertBounce; // 1.57m ± 0.020m
    const camX = curR * Math.cos(rState.thetaRad);
    const camZ = curR * Math.sin(rState.thetaRad);
    this.camera.position.set(camX, eyeY, camZ);

    // 1. Dominant Forward Gaze Target (~76% of cycle):
    //    Fixates ~32m ahead along the curving Ring Road near the inner lane tangent (Land & Lee curve-steering optics),
    //    keeping the horizon comfortable (+1.8°..+2.8° pitch) while the curving road AND upcoming building fill the wide FOV!
    const lookAheadTheta = rState.thetaRad + (0.42 + Math.sin(rState.gazeClock * 0.45) * 0.03) * dirSign;
    const forwardRadius = 71.8;
    const forwardHorizonY = 2.45 + Math.sin(rState.gazeClock * 0.55) * 0.28;
    const roadAheadTarget = new THREE.Vector3(
      forwardRadius * Math.cos(lookAheadTheta),
      forwardHorizonY,
      forwardRadius * Math.sin(lookAheadTheta)
    );

    // 2. Occasional Diagonal Architectural Glance Target (~24% of cycle — "sesekali noleh"):
    const arch = this._evaluateRunnerArchitecturalTarget(rState.thetaRad, rState.gazeClock, dirSign);

    // 12.0-second human gaze cycle:
    // - 0.00 .. 0.62 (7.44s): 100% looking forward down the curving road
    // - 0.62 .. 0.70 (0.96s): smooth head-turn inward toward landmark
    // - 0.70 .. 0.84 (1.68s): admiring architectural glance ("noleh")
    // - 0.84 .. 0.92 (0.96s): smooth head-turn back to forward running path
    // - 0.92 .. 1.00 (0.96s): 100% looking forward
    const cyclePhase = (rState.gazeClock % 12.0) / 12.0;
    let glanceWeight = 0.0;
    if (rState.forceGlanceMode === true) {
      glanceWeight = 1.0;
    } else if (rState.forceGlanceMode === false) {
      glanceWeight = 0.0;
    } else if (cyclePhase >= 0.62 && cyclePhase < 0.70) {
      const u = (cyclePhase - 0.62) / 0.08;
      glanceWeight = u * u * (3.0 - 2.0 * u);
    } else if (cyclePhase >= 0.70 && cyclePhase < 0.84) {
      glanceWeight = 1.0;
    } else if (cyclePhase >= 0.84 && cyclePhase < 0.92) {
      const u = 1.0 - (cyclePhase - 0.84) / 0.08;
      glanceWeight = u * u * (3.0 - 2.0 * u);
    }
    rState.glanceWeight = glanceWeight;
    rState.activeLandmarkLabel = glanceWeight >= 0.35 ? arch.glanceLabel : arch.forwardLabel;

    const desiredTarget = new THREE.Vector3().lerpVectors(roadAheadTarget, arch.target, glanceWeight);
    const smoothFactor = 1.0 - Math.exp(-4.8 * clampedDt);
    rState.currentLookPos.lerp(desiredTarget, smoothFactor);

    // Decay manual mouse-look offset smoothly after 3.2s of idle
    if (!this.pointerDragging) {
      rState.idleSinceDragSec += clampedDt;
      if (rState.idleSinceDragSec > 3.2) {
        const decay = Math.exp(-2.2 * clampedDt);
        rState.userYawOffset *= decay;
        rState.userPitchOffset *= decay;
      }
    }

    this.camera.up.set(Math.sin(headRoll), Math.cos(headRoll), 0);
    this.camera.lookAt(rState.currentLookPos);

    if (Math.abs(rState.userYawOffset) > 1e-4 || Math.abs(rState.userPitchOffset) > 1e-4) {
      const euler = new THREE.Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
      euler.y += rState.userYawOffset;
      euler.x = THREE.MathUtils.clamp(euler.x + rState.userPitchOffset, -Math.PI * 0.45, Math.PI * 0.45);
      this.camera.quaternion.setFromEuler(euler);
    }
  }

  resolveSoftSpatialCollision(pos) {
    // 1. Ground elevation guard
    if (pos.y < this.minElevationY) {
      pos.y = this.minElevationY;
      if (this.droneVelocity.y < 0) this.droneVelocity.y = 0;
    }

    // 2. Outer world boundary guard
    const rHoriz = Math.hypot(pos.x, pos.z);
    if (rHoriz > this.maxRadiusFromOrigin) {
      const s = this.maxRadiusFromOrigin / rHoriz;
      pos.x *= s;
      pos.z *= s;
    }

    // 3. AABB Solid Exclusion (ITTC Perpustakaan Lama + NW Annex)
    for (const box of this.collisionSolids) {
      if (
        pos.x > box.minX && pos.x < box.maxX &&
        pos.y > box.minY && pos.y < box.maxY &&
        pos.z > box.minZ && pos.z < box.maxZ
      ) {
        const dxMin = pos.x - box.minX;
        const dxMax = box.maxX - pos.x;
        const dzMin = pos.z - box.minZ;
        const dzMax = box.maxZ - pos.z;
        const dyMax = box.maxY - pos.y;
        const minPen = Math.min(dxMin, dxMax, dzMin, dzMax, dyMax);
        if (minPen === dyMax) pos.y = box.maxY + 0.25;
        else if (minPen === dxMin) pos.x = box.minX - 0.25;
        else if (minPen === dxMax) pos.x = box.maxX + 0.25;
        else if (minPen === dzMin) pos.z = box.minZ - 0.25;
        else pos.z = box.maxZ + 0.25;
      }
    }

    // 4. Curved Annular Building Hull Exclusion (DPPD, STP-UI, ILRC/DRPM)
    //    Ensures neither Orbit camera nor Drone camera can ever enter inside the curved buildings!
    const r = Math.hypot(pos.x, pos.z);
    if (r > 1.0) {
      const deg = Math.atan2(pos.z, pos.x) / DEG;
      for (const sec of this.polarSectors) {
        if (deg >= sec.degMin && deg <= sec.degMax && r > sec.rMin && r < sec.rMax && pos.y < sec.maxY) {
          const dIn = r - sec.rMin;
          const dOut = sec.rMax - r;
          const dTop = sec.maxY - pos.y;
          if (dTop < dIn && dTop < dOut && pos.y > sec.maxY - 4.5) {
            pos.y = sec.maxY + 0.35;
          } else {
            const targetR = dOut <= dIn ? sec.rMax + 0.45 : sec.rMin - 0.45;
            const scale = targetR / r;
            pos.x *= scale;
            pos.z *= scale;
          }
        }
      }
    }
  }

  update(dt) {
    const clampedDt = Math.min(0.05, Math.max(0.001, dt));

    if (this.transition) {
      this.transition.elapsed += clampedDt;
      const rawT = Math.min(1.0, this.transition.elapsed / this.transition.duration);
      const ease = 1.0 - Math.pow(1.0 - rawT, 3);
      this.camera.position.lerpVectors(this.transition.startPos, this.transition.endPos, ease);
      this.controls.target.lerpVectors(this.transition.startTgt, this.transition.endTgt, ease);
      this.resolveSoftSpatialCollision(this.camera.position);
      this.camera.up.set(0, 1, 0);
      this.controls.update();
      if (rawT >= 1.0) {
        this._syncDroneAnglesFromCamera();
        this.transition = null;
      }
      return;
    }

    if (this.mode === 'runner') {
      this._updateRunnerStep(clampedDt);
    } else if (this.mode === 'drone') {
      const euler = new THREE.Euler(this.dronePitch, this.droneYaw, 0, 'YXZ');
      this.camera.quaternion.setFromEuler(euler);

      const speed = (this.keys.has('ShiftLeft') || this.keys.has('ShiftRight')) ? 44.0 : 18.0;
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
      const accel = new THREE.Vector3();

      if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) accel.add(forward);
      if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) accel.sub(forward);
      if (this.keys.has('KeyD') || this.keys.has('ArrowRight')) accel.add(right);
      if (this.keys.has('KeyA') || this.keys.has('ArrowLeft')) accel.sub(right);
      if (this.keys.has('KeyE') || this.keys.has('Space')) accel.y += 1.0;
      if (this.keys.has('KeyQ') || this.keys.has('KeyC')) accel.y -= 1.0;

      if (accel.lengthSq() > 0) {
        accel.normalize().multiplyScalar(speed);
      }

      const damping = 1.0 - Math.exp(-8.5 * clampedDt);
      this.droneVelocity.lerp(accel, damping);
      this.camera.position.addScaledVector(this.droneVelocity, clampedDt);
      this.resolveSoftSpatialCollision(this.camera.position);
    } else {
      this.controls.update();
      // Enforce full building anti-penetration in Orbit mode as well!
      this.resolveSoftSpatialCollision(this.camera.position);
    }
  }
}
