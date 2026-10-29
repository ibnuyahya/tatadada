import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./world-gis/FloatingOriginManager.js
 *
 * Relative-To-Center (RTC) Double-Precision Spatial Anchor Manager for Kampus UI Depok.
 * Eliminates Float32 GPU vertex jitter & Z-fighting across planetary/GIS ECEF coordinates
 * by computing p_local = P_world - C_center strictly in IEEE-754 Float64 before writing
 * local coordinates into Float32 THREE.Object3D transforms.
 */
export class FloatingOriginManager {
  static DEFAULT_RTC_CENTER = Object.freeze({
    x: -1834520.0,
    y: 6041210.0,
    z: -703850.0
  });

  static DEFAULT_GEO_ANCHOR = Object.freeze({
    lat: -6.3658,
    lon: 106.8286,
    altMeters: 72.5
  });

  /**
   * @param {Object} [centerOrContext] - Either { x, y, z } world center or kampus-ui-context JSON object
   */
  constructor(centerOrContext = FloatingOriginManager.DEFAULT_RTC_CENTER) {
    this.rtcCenter64 = new Float64Array(3);
    this._scratchDelta64 = new Float64Array(3);

    const srcCenter =
      centerOrContext && centerOrContext.rtcCenterWorld
        ? centerOrContext.rtcCenterWorld
        : centerOrContext || FloatingOriginManager.DEFAULT_RTC_CENTER;

    this.lat =
      centerOrContext && Number.isFinite(centerOrContext.lat)
        ? Number(centerOrContext.lat)
        : FloatingOriginManager.DEFAULT_GEO_ANCHOR.lat;
    this.lon =
      centerOrContext && Number.isFinite(centerOrContext.lon)
        ? Number(centerOrContext.lon)
        : FloatingOriginManager.DEFAULT_GEO_ANCHOR.lon;

    this.setCenter(srcCenter);
  }

  /**
   * Updates the active RTC world center in IEEE-754 Float64 precision.
   * @param {{x: number, y: number, z: number}} worldCenter
   */
  setCenter(worldCenter) {
    this.rtcCenter64[0] = Number(worldCenter?.x ?? FloatingOriginManager.DEFAULT_RTC_CENTER.x);
    this.rtcCenter64[1] = Number(worldCenter?.y ?? FloatingOriginManager.DEFAULT_RTC_CENTER.y);
    this.rtcCenter64[2] = Number(worldCenter?.z ?? FloatingOriginManager.DEFAULT_RTC_CENTER.z);
    return this;
  }

  /**
   * Returns a plain Float64 snapshot of the RTC world anchor.
   */
  getCenterWorld() {
    return {
      x: this.rtcCenter64[0],
      y: this.rtcCenter64[1],
      z: this.rtcCenter64[2]
    };
  }

  /**
   * Computes p_local = P_world - C_center strictly in Float64 before writing to Float32 targetVec.
   * @param {{x: number, y: number, z: number}} worldPos - Global ECEF/world coordinate in meters
   * @param {THREE.Vector3} [targetVec] - Destination Three.js Vector3 (e.g. Object3D.position)
   * @returns {THREE.Vector3}
   */
  toLocal(worldPos, targetVec = new THREE.Vector3()) {
    this._scratchDelta64[0] = Number(worldPos.x) - this.rtcCenter64[0];
    this._scratchDelta64[1] = Number(worldPos.y) - this.rtcCenter64[1];
    this._scratchDelta64[2] = Number(worldPos.z) - this.rtcCenter64[2];

    targetVec.set(
      this._scratchDelta64[0],
      this._scratchDelta64[1],
      this._scratchDelta64[2]
    );
    return targetVec;
  }

  /**
   * Static helper for direct Float64 RTC subtraction without instantiating a manager.
   */
  static toLocal(
    worldPos,
    rtcCenter = FloatingOriginManager.DEFAULT_RTC_CENTER,
    targetVec = new THREE.Vector3()
  ) {
    const dx = Number(worldPos.x) - Number(rtcCenter.x);
    const dy = Number(worldPos.y) - Number(rtcCenter.y);
    const dz = Number(worldPos.z) - Number(rtcCenter.z);
    return targetVec.set(dx, dy, dz);
  }

  /**
   * Converts a local scene position back to global Float64 world coordinates.
   * @param {{x: number, y: number, z: number}} localPos
   * @returns {{x: number, y: number, z: number}}
   */
  toWorld(localPos) {
    return {
      x: this.rtcCenter64[0] + Number(localPos.x),
      y: this.rtcCenter64[1] + Number(localPos.y),
      z: this.rtcCenter64[2] + Number(localPos.z)
    };
  }

  /**
   * Anchors a THREE.Group / THREE.Object3D cleanly into the local RTC coordinate frame.
   * Automatically handles either a local offset { x, y, z } or an ECEF world coordinate (> 100 km).
   *
   * @param {THREE.Object3D} group
   * @param {{x: number, y: number, z: number, isWorld?: boolean}|null} [localOffset]
   * @returns {THREE.Object3D}
   */
  anchorGroup(group, localOffset = null) {
    if (!group) return group;

    if (!localOffset) {
      group.position.set(0.0, 0.0, 0.0);
    } else {
      const ox = Number(localOffset.x) || 0.0;
      const oy = Number(localOffset.y) || 0.0;
      const oz = Number(localOffset.z) || 0.0;
      const isGlobalECEF =
        localOffset.isWorld === true ||
        Math.abs(ox) > 100000.0 ||
        Math.abs(oy) > 100000.0 ||
        Math.abs(oz) > 100000.0;

      if (isGlobalECEF) {
        this.toLocal(localOffset, group.position);
      } else {
        group.position.set(ox, oy, oz);
      }
    }

    group.userData = group.userData || {};
    group.userData.rtcAnchored = true;
    group.userData.rtcCenterWorld = this.getCenterWorld();
    group.updateMatrixWorld(true);
    return group;
  }

  /**
   * Converts small geographic (lat/lon degree) offsets around Kampus UI Depok into local meters
   * (+X East, +Y Up, +Z South/Rotunda, -Z North/Danau Kenanga).
   */
  geoToLocal(latDeg, lonDeg, elevationY = 0.0, targetVec = new THREE.Vector3()) {
    const metersPerDegLat = 110574.0;
    const metersPerDegLon = 111320.0 * Math.cos((this.lat * Math.PI) / 180.0);
    const eastX = (Number(lonDeg) - this.lon) * metersPerDegLon;
    // Note: In our architectural frame, +Z faces South (Rotunda) and -Z faces North (Danau Kenanga)
    const southZ = -(Number(latDeg) - this.lat) * metersPerDegLat;
    return targetVec.set(eastX, Number(elevationY) || 0.0, southZ);
  }

  /**
   * Loads ./data-routes/kampus-ui-context.json via strict relative path in browser environments,
   * falling back cleanly to canonical defaults in headless verification.
   */
  static async loadCampusContext(relativeUrl = './data-routes/kampus-ui-context.json') {
    if (typeof fetch === 'function') {
      try {
        const res = await fetch(relativeUrl);
        if (res.ok) {
          return await res.json();
        }
      } catch (_err) {
        // Fallback to built-in RTC context in offline/headless mode
      }
    }
    return {
      lat: FloatingOriginManager.DEFAULT_GEO_ANCHOR.lat,
      lon: FloatingOriginManager.DEFAULT_GEO_ANCHOR.lon,
      rtcCenterWorld: { ...FloatingOriginManager.DEFAULT_RTC_CENTER }
    };
  }
}
