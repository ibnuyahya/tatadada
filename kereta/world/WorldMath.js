import * as THREE from 'three';
import { WorldMathCore } from './WorldMathCore.js';

/**
 * ============================================================================
 * THREE.JS WORLD MATHEMATICS ADAPTER (FLOATING-ORIGIN AWARE + ZERO-GC POOLS)
 * Wraps `WorldMathCore` (64-bit Float64 spline & terrain solver) into Three.js
 * `Vector3` structures, supporting both global/chunk-local `originS` offsets
 * and reusable frame objects for per-frame kinematics.
 * ============================================================================
 */
const _rawFrame = {
  px: 0, py: 0, pz: 0,
  tx: 0, ty: 0, tz: -1,
  rx: 1, ry: 0, rz: 0,
  ux: 0, uy: 1, uz: 0,
  curvature: 0,
  cantAngle: 0,
  gradePermille: 0,
  speedLimit: 350,
  inTunnel: false
};

export function createTrackFrameStruct() {
  return {
    position: new THREE.Vector3(),
    tangent: new THREE.Vector3(0, 0, -1),
    right: new THREE.Vector3(1, 0, 0),
    up: new THREE.Vector3(0, 1, 0),
    curvature: 0,
    cantAngle: 0,
    gradePermille: 0,
    speedLimit: 350,
    inTunnel: false,
    worldS: 0
  };
}

export class EndlessWorldMathematics {
  static originS = 0.0;

  static setOriginS(originS) {
    this.originS = originS;
  }

  static getBiomeAtDistance(s) {
    return WorldMathCore.getBiomeAtDistance(s);
  }

  static getStationFlattenMask(s) {
    return WorldMathCore.getStationFlattenMask(s);
  }

  static isNearStation(s, radius = 95) {
    return WorldMathCore.isNearStation(s, radius);
  }

  static isTunnelZone(s) {
    return WorldMathCore.isTunnelZone(s);
  }

  static getViaductFactor(s) {
    return WorldMathCore.getViaductFactor(s);
  }

  static isViaductZone(s) {
    return WorldMathCore.isViaductZone(s);
  }

  static getRiverWaterLevel(s) {
    return WorldMathCore.getRiverWaterLevel(s);
  }

  static getHighwayElevation(s, trackCenterY) {
    return WorldMathCore.getHighwayElevation(s, trackCenterY);
  }

  static getTrackPosition(s, lateralOffset = 0, customOriginS = this.originS, targetVec = new THREE.Vector3()) {
    WorldMathCore.getTrackPositionRaw(s, lateralOffset, customOriginS, _rawFrame);
    return targetVec.set(_rawFrame.x, _rawFrame.y, _rawFrame.z);
  }

  /**
   * Computes track frame at 64-bit `s` with local Z relative to `customOriginS`
   * (defaults to `EndlessWorldMathematics.originS`).
   * Pass `outFrame` (from `createTrackFrameStruct()`) in hot loops for zero GC!
   */
  static getTrackFrame(s, lateralOffset = 0, customOriginS = this.originS, outFrame = null) {
    WorldMathCore.getTrackFrameRaw(s, lateralOffset, customOriginS, _rawFrame);
    const frame = outFrame || createTrackFrameStruct();
    frame.position.set(_rawFrame.px, _rawFrame.py, _rawFrame.pz);
    frame.tangent.set(_rawFrame.tx, _rawFrame.ty, _rawFrame.tz);
    frame.right.set(_rawFrame.rx, _rawFrame.ry, _rawFrame.rz);
    frame.up.set(_rawFrame.ux, _rawFrame.uy, _rawFrame.uz);
    frame.curvature = _rawFrame.curvature;
    frame.cantAngle = _rawFrame.cantAngle;
    frame.gradePermille = _rawFrame.gradePermille;
    frame.speedLimit = _rawFrame.speedLimit;
    frame.inTunnel = _rawFrame.inTunnel;
    frame.worldS = s;
    return frame;
  }

  /**
   * Evaluates terrain height using true 64-bit `worldS` so terrain never shifts
   * when `originS` rebases.
   */
  static getTerrainHeightAtS(x, worldS, centerFrame = null) {
    if (centerFrame) {
      return WorldMathCore.getTerrainHeightAtS(x, worldS, centerFrame.position.x, centerFrame.position.y);
    }
    return WorldMathCore.getTerrainHeightAtS(x, worldS);
  }
}
