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
  panelRect = { width: 280, height: 0 }
) {
  const safeW = Math.max(1, viewportW);
  const safeH = Math.max(1, viewportH);
  const isPortrait = safeH > safeW || safeW < 768;
  camera.aspect = safeW / safeH;

  const baseVerticalFov = 48;
  if (camera.aspect < 1.0) {
    const rad = (baseVerticalFov * Math.PI) / 180;
    const hFov = 2 * Math.atan(Math.tan(rad / 2) / Math.max(0.52, camera.aspect));
    camera.fov = Math.min(78, (hFov * 180) / Math.PI);
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
 * Multi-Mode Camera Director (Google Earth Fluid Parabolic Flights, Orbit, 6-DOF Drone, & Pedestrian Walk)
 */
export class CameraDirector {
  constructor(camera, controls, colliders = []) {
    this.camera = camera;
    this.controls = controls;
    this.colliders = colliders;
    this.mode = 'ORBIT'; // 'ORBIT' | 'DRONE_6DOF' | 'WALK_SIVITAS' | 'AUTO_CINEMATIC'
    this._baseNear = Math.max(0.12, Number(camera?.near) || 0.18);

    this.velocity = new THREE.Vector3();
    this.yaw = 0.0;
    this.pitch = -0.15;
    this.roll = 0.0;
    this.cinematicAngle = 0.55;

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

    // Hard 3D solid-world physics context & previous-frame position for continuous sweep anti-tunneling
    this.physicsContext = null;
    this._prevCamPos = new THREE.Vector3();
    this._hasPrevCamPos = false;
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
    if (this.mode === 'DRONE_6DOF' || this.mode === 'WALK_SIVITAS') {
      if (this.controls) this.controls.enabled = false;
      this.camera.getWorldDirection(this._flightLookTmp);
      this.yaw = Math.atan2(-this._flightLookTmp.x, -this._flightLookTmp.z);
      this.pitch = Math.asin(Math.max(-0.95, Math.min(0.95, this._flightLookTmp.y)));
      this.roll = 0.0;
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

  setMode(newMode, options = {}) {
    const prevMode = this.mode;
    this.mode = newMode;
    this.transitionActive = false;
    this._flightNadirSpec = null;
    this.camera.up.set(0, 1, 0);

    if (newMode === 'ORBIT' || newMode === 'AUTO_CINEMATIC') {
      this.roll = 0;
      if (this.controls) {
        if (prevMode === 'DRONE_6DOF' || prevMode === 'WALK_SIVITAS') {
          this.camera.getWorldDirection(this._flightLookTmp);
          this.controls.target.copy(this.camera.position).addScaledVector(this._flightLookTmp, 36.0);
        }
        this.controls.enabled = true;
      }
    } else if (newMode === 'DRONE_6DOF') {
      if (this.controls) this.controls.enabled = false;
      const dir = new THREE.Vector3();
      this.camera.getWorldDirection(dir);
      this.yaw = Math.atan2(-dir.x, -dir.z);
      this.pitch = Math.asin(Math.max(-0.95, Math.min(0.95, dir.y)));
      this.velocity.set(0, 0, 0);
      this._droneManualEngaged = false;
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
    if (this.mode !== 'DRONE_6DOF' && this.mode !== 'WALK_SIVITAS') return;
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

    if (this.mode === 'ORBIT') {
      this.camera.up.set(0, 1, 0);
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
        resolveSoftSpatialCollision(
          this.camera.position,
          null,
          this.colliders,
          0.56,
          orbitFloorY,
          clampedDt,
          this._prevCamPos,
          ctx,
          false
        );
      }
      this._prevCamPos.copy(this.camera.position);
      this._updateAltitudeAdaptiveNearPlane();
      return;
    }

    if (this.mode === 'AUTO_CINEMATIC') {
      this.camera.up.set(0, 1, 0);
      this.cinematicAngle += clampedDt * 0.14;
      const radius = 88.0 + Math.sin(this.cinematicAngle * 0.7) * 14.0;
      const height = 34.0 + Math.sin(this.cinematicAngle * 0.45) * 16.0;
      this.goalPos.set(
        Math.sin(this.cinematicAngle) * radius,
        height,
        6.0 + Math.cos(this.cinematicAngle) * radius
      );
      this.camera.position.lerp(this.goalPos, dampAlpha);
      if (this.controls) {
        this.controls.target.lerp(this.goalTarget.set(0, 18.5, 4.0), dampAlpha);
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

    // Auto gentle orbit cruise in DRONE_6DOF only near Gedung Rektorat before manual drone keys are engaged
    const hasManualInput = moveF !== 0 || moveR !== 0 || moveU !== 0;
    if (hasManualInput) {
      this._droneManualEngaged = true;
    }
    const nearRektorat =
      Math.hypot(this.camera.position.x, this.camera.position.z - 6.0) < 155.0 &&
      this.camera.position.y > 2.0;
    if (!isWalk && !hasManualInput && !this._droneManualEngaged && nearRektorat && this.velocity.lengthSq() < 0.04) {
      this.cinematicAngle += clampedDt * 0.15;
      const targetOrbitX = Math.sin(this.cinematicAngle) * 86.0;
      const targetOrbitZ = 6.0 + Math.cos(this.cinematicAngle) * 86.0;
      const targetOrbitY = 36.0 + Math.sin(this.cinematicAngle * 1.3) * 11.0;
      this.camera.position.x += (targetOrbitX - this.camera.position.x) * dampAlpha * 0.45;
      this.camera.position.z += (targetOrbitZ - this.camera.position.z) * dampAlpha * 0.45;
      this.camera.position.y += (targetOrbitY - this.camera.position.y) * dampAlpha * 0.45;
      const lookDx = -this.camera.position.x;
      const lookDz = 4.0 - this.camera.position.z;
      this.yaw = Math.atan2(-lookDx, -lookDz);
      this.pitch = Math.atan2(18.5 - this.camera.position.y, Math.hypot(lookDx, lookDz));
      this.roll += (0.045 - this.roll) * dampAlpha; // Subtle centripetal roll banking
    } else {
      this.velocity.addScaledVector(this._forward, moveF * accel * clampedDt);
      this.velocity.addScaledVector(this._right, moveR * accel * clampedDt);
      this.velocity.y += moveU * accel * clampedDt;

      const drag = Math.exp(-4.8 * clampedDt);
      this.velocity.multiplyScalar(drag);
      if (this.velocity.length() > maxSpeed) {
        this.velocity.setLength(maxSpeed);
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
}
