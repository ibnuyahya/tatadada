import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./core/CameraDirector.js
 *
 * Multi-Angle Class-A Inspection & 6-DOF Free-Flight Camera Director:
 * - Horizon-Locked Nadir 90° View (eps = 0.45m, camera.up = (0,1,0) — Hukum Emas 19)
 * - Adaptive Non-Occluding Viewport Offset (Desktop <= 280px side drawer, Portrait <= 34vh bottom sheet — Hukum Emas 20)
 * - Smooth Exponential Camera Transitions + 6-DOF Drone Flight with Soft Anti-Clipping Spatial Collision
 * - Geodetically Calibrated UI Depok Bundaran Presets (Tri-Angle + Sector & Street View Nodes)
 */
export class CameraDirector {
  constructor(camera, controls, domElement) {
    this.camera = camera;
    this.controls = controls;
    this.domElement = domElement;

    this.mode = 'orbit'; // 'orbit' | 'drone'
    this.drawerCollapsed = true;
    this.transition = null;

    // 6-DOF Drone state
    this.droneVelocity = new THREE.Vector3();
    this.droneYaw = 0;
    this.dronePitch = 0;
    this.keys = new Set();
    this.pointerDragging = false;
    this.lastPointer = { x: 0, y: 0 };

    // Soft collision constraints (ground + central island building mass envelopes)
    this.minElevationY = 1.45;
    this.maxRadiusFromOrigin = 260.0;
    this.collisionSolids = [];

    this._bindDroneEvents();
  }

  registerCollisionBoxes(boxes) {
    this.collisionSolids = Array.isArray(boxes) ? boxes : [];
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
   * Hukum Emas 20: Adaptive Non-Occluding Viewport Frustum Offset.
   */
  applyNonOccludingViewportOffset(collapsed = this.drawerCollapsed) {
    this.drawerCollapsed = collapsed;
    const fullW = Math.max(1, window.innerWidth);
    const fullH = Math.max(1, window.innerHeight);
    this.camera.aspect = fullW / fullH;

    if (collapsed) {
      this.camera.clearViewOffset();
      this.camera.updateProjectionMatrix();
      return;
    }

    const isPortraitOrMobile = fullW < 768 || fullW / fullH < 1.0;
    if (isPortraitOrMobile) {
      // Bottom sheet <= 34vh -> shift optical center upward so 3D object stays clear in top 66vh
      const sheetPx = Math.min(fullH * 0.34, 300);
      this.camera.setViewOffset(fullW, fullH, 0, Math.round(sheetPx * 0.46), fullW, fullH);
    } else {
      // Left drawer <= 280px -> shift optical center rightward so 3D object stays centered in remaining space
      const drawerPx = 280;
      this.camera.setViewOffset(fullW, fullH, -Math.round(drawerPx * 0.46), 0, fullW, fullH);
    }
    this.camera.updateProjectionMatrix();
  }

  flyTo(position, target, durationSec = 0.72) {
    const endPos = position instanceof THREE.Vector3 ? position.clone() : new THREE.Vector3(...position);
    const endTgt = target instanceof THREE.Vector3 ? target.clone() : new THREE.Vector3(...target);
    this.camera.up.set(0, 1, 0);

    if (durationSec <= 0.01) {
      this.camera.position.copy(endPos);
      this.controls.target.copy(endTgt);
      this.controls.update();
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
    if (this.mode === nextMode) return;
    this.mode = nextMode;
    if (nextMode === 'drone') {
      this.controls.enabled = false;
      this.transition = null;
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
      if (this.mode !== 'drone') return;
      this.pointerDragging = true;
      this.lastPointer.x = e.clientX;
      this.lastPointer.y = e.clientY;
    });
    window.addEventListener('pointermove', (e) => {
      if (this.mode !== 'drone' || !this.pointerDragging) return;
      const dx = e.clientX - this.lastPointer.x;
      const dy = e.clientY - this.lastPointer.y;
      this.lastPointer.x = e.clientX;
      this.lastPointer.y = e.clientY;
      this.droneYaw -= dx * 0.0028;
      this.dronePitch = THREE.MathUtils.clamp(this.dronePitch - dy * 0.0024, -Math.PI * 0.47, Math.PI * 0.47);
    });
    window.addEventListener('pointerup', () => {
      this.pointerDragging = false;
    });
  }

  resolveSoftSpatialCollision(pos) {
    if (pos.y < this.minElevationY) {
      pos.y = THREE.MathUtils.lerp(pos.y, this.minElevationY, 0.65);
      if (this.droneVelocity.y < 0) this.droneVelocity.y = 0;
    }
    const rHoriz = Math.hypot(pos.x, pos.z);
    if (rHoriz > this.maxRadiusFromOrigin) {
      const s = this.maxRadiusFromOrigin / rHoriz;
      pos.x *= s;
      pos.z *= s;
    }
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
        if (minPen === dyMax) pos.y = box.maxY + 0.05;
        else if (minPen === dxMin) pos.x = box.minX - 0.05;
        else if (minPen === dxMax) pos.x = box.maxX + 0.05;
        else if (minPen === dzMin) pos.z = box.minZ - 0.05;
        else pos.z = box.maxZ + 0.05;
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
      this.camera.up.set(0, 1, 0);
      this.controls.update();
      if (rawT >= 1.0) {
        this._syncDroneAnglesFromCamera();
        this.transition = null;
      }
      return;
    }

    if (this.mode === 'drone') {
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
      if (this.camera.position.y < 1.15) {
        this.camera.position.y = 1.15;
      }
    }
  }
}
