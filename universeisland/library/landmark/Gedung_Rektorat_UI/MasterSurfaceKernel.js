import * as THREE from 'three';
import { MasterSurfaceBuilder } from '../../../core/MasterSurfaceBuilder.js';
import {
  MASTER_PROFILE_STATIONS,
  CORNER_TOWER_CROWN_SPEC
} from './AssetMetaAndEnvelope.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.1 DJI Mavic Photogrammetric Calibration)
 * ./library/landmark/Gedung_Rektorat_UI/MasterSurfaceKernel.js
 *
 * TAHAP 2 & TAHAP 3 (Gate 2):
 * Kerangka Tulang Punggung Aksial & Blok Permukaan Induk Rapat Air (Watertight SSoST).
 *
 * Mengimplementasikan Hukum Emas 1, 2, 3, & 17:
 * - Satu fungsi penampang dihedral D4 krusiform kontinu yang menyatukan Fondasi (-18m),
 *   Kolonnade Putih Lantai Dasar, 6 Tingkat Teritisan Pagoda Genteng Merah + Ceruk Kaca Berlian Tengah,
 *   Leher Mahkota Putih Makara UI + Jendela Clerestory, Atap Puncak Limas Tajug Curam (~48.7°),
 *   Tudung Puncak Putih, & Penangkal Petir (50.00m).
 * - Seluruh subdomain di Tahap 4 & 5 memotong langsung dari fungsi ini (0.0000m Gap).
 */

const _pPlusU = new THREE.Vector3();
const _pMinusU = new THREE.Vector3();
const _pPlusV = new THREE.Vector3();
const _pMinusV = new THREE.Vector3();
const _tU = new THREE.Vector3();
const _tV = new THREE.Vector3();

/**
 * Interpolates the axial profile station parameters at normalized vertical coordinate v in [0, 1].
 */
export function sampleAxialProfile(v) {
  const clampedV = Math.max(0.0, Math.min(1.0, v));
  const stations = MASTER_PROFILE_STATIONS;
  const maxIdx = stations.length - 1;
  const scaled = clampedV * maxIdx;
  const idx0 = Math.min(maxIdx - 1, Math.floor(scaled));
  const idx1 = idx0 + 1;
  const rawT = scaled - idx0;

  const s0 = stations[idx0];
  const s1 = stations[idx1];

  const exp = s0.curveExp || 1.0;
  const shapeT = exp !== 1.0 ? 1.0 - Math.pow(1.0 - rawT, exp) : rawT;

  return {
    y: s0.y + (s1.y - s0.y) * rawT,
    coreHalf: s0.coreHalf + (s1.coreHalf - s0.coreHalf) * shapeT,
    coreChamfer: s0.coreChamfer + (s1.coreChamfer - s0.coreChamfer) * shapeT,
    towerActive: s0.towerActive + (s1.towerActive - s0.towerActive) * shapeT,
    towerCenter: s0.towerCenter + (s1.towerCenter - s0.towerCenter) * shapeT,
    towerHalf: s0.towerHalf + (s1.towerHalf - s0.towerHalf) * shapeT,
    towerChamfer: s0.towerChamfer + (s1.towerChamfer - s0.towerChamfer) * shapeT
  };
}

/**
 * Evaluates the 8 control vertices Q0..Q7 of Quadrant 0 (+Z front centerline -> +X east centerline)
 * blending seamlessly between square-pyramid morphology (alpha=0) and
 * the Mavic-calibrated 4-Corner-Wing + Central Recessed Glass Atrium morphology (alpha=1).
 */
function computeQuadrantControlPoints(prof, outPts) {
  const bc = Math.max(0.0005, prof.coreHalf);
  const rc = Math.min(bc * 0.45, Math.max(0.0, prof.coreChamfer));
  const alpha = Math.max(0.0, Math.min(1.0, prof.towerActive));

  // Pure chamfered square control points (alpha = 0: Crown Drum & Steep Top Tajug Pyramid!)
  // Notice: Q1 aligns at x = min(2.3/10.65 * span0, 0.42 * span0) so Crown Clerestory vs Makara Panel slices cleanly!
  const span0 = Math.max(0.0002, bc - rc);
  const q0x_0 = 0.0,          q0z_0 = bc;
  const q1x_0 = 0.40 * span0, q1z_0 = bc;
  const q2x_0 = 0.72 * span0, q2z_0 = bc;
  const q3x_0 = span0,        q3z_0 = bc;
  const q4x_0 = bc,           q4z_0 = span0;
  const q5x_0 = bc,           q5z_0 = 0.72 * span0;
  const q6x_0 = bc,           q6z_0 = 0.40 * span0;
  const q7x_0 = bc,           q7z_0 = 0.0;

  if (alpha <= 1e-5) {
    outPts[0][0] = q0x_0; outPts[0][1] = q0z_0;
    outPts[1][0] = q1x_0; outPts[1][1] = q1z_0;
    outPts[2][0] = q2x_0; outPts[2][1] = q2z_0;
    outPts[3][0] = q3x_0; outPts[3][1] = q3z_0;
    outPts[4][0] = q4x_0; outPts[4][1] = q4z_0;
    outPts[5][0] = q5x_0; outPts[5][1] = q5z_0;
    outPts[6][0] = q6x_0; outPts[6][1] = q6z_0;
    outPts[7][0] = q7x_0; outPts[7][1] = q7z_0;
    return;
  }

  // Cruciform 4-Corner-Wing + Central Recessed Glass Atrium control points (alpha = 1)
  const xIn = prof.towerCenter - prof.towerHalf;
  const xOut = prof.towerCenter + prof.towerHalf;
  const rt = Math.min(prof.towerHalf * 0.45, prof.towerChamfer);

  const q0x_1 = 0.0,       q0z_1 = bc;
  const q1x_1 = xIn,       q1z_1 = bc;
  const q2x_1 = xIn,       q2z_1 = xOut;
  const q3x_1 = xOut - rt, q3z_1 = xOut;
  const q4x_1 = xOut,      q4z_1 = xOut - rt;
  const q5x_1 = xOut,      q5z_1 = xIn;
  const q6x_1 = bc,        q6z_1 = xIn;
  const q7x_1 = bc,        q7z_1 = 0.0;

  const inv = 1.0 - alpha;
  outPts[0][0] = inv * q0x_0 + alpha * q0x_1; outPts[0][1] = inv * q0z_0 + alpha * q0z_1;
  outPts[1][0] = inv * q1x_0 + alpha * q1x_1; outPts[1][1] = inv * q1z_0 + alpha * q1z_1;
  outPts[2][0] = inv * q2x_0 + alpha * q2x_1; outPts[2][1] = inv * q2z_0 + alpha * q2z_1;
  outPts[3][0] = inv * q3x_0 + alpha * q3x_1; outPts[3][1] = inv * q3z_0 + alpha * q3z_1;
  outPts[4][0] = inv * q4x_0 + alpha * q4x_1; outPts[4][1] = inv * q4z_0 + alpha * q4z_1;
  outPts[5][0] = inv * q5x_0 + alpha * q5x_1; outPts[5][1] = inv * q5z_0 + alpha * q5z_1;
  outPts[6][0] = inv * q6x_0 + alpha * q6x_1; outPts[6][1] = inv * q6z_0 + alpha * q6z_1;
  outPts[7][0] = inv * q7x_0 + alpha * q7x_1; outPts[7][1] = inv * q7z_0 + alpha * q7z_1;
}

const _quadPts = [
  [0, 0], [0, 0], [0, 0], [0, 0],
  [0, 0], [0, 0], [0, 0], [0, 0]
];

/**
 * Evaluates raw 3D position on the D4-symmetric perimeter at (u, v) with C1 Quadratic Bezier fillets
 * so Studio Zebra reflection lines flow cleanly with 0 self-intersections (Gate 2).
 */
function evalPositionOnly(u, v, outVec) {
  const prof = sampleAxialProfile(v);
  computeQuadrantControlPoints(prof, _quadPts);

  let uw = u % 1.0;
  if (uw < 0) uw += 1.0;

  const qFloat = uw * 4.0;
  const qIdx = Math.min(3, Math.floor(qFloat));
  const localQ = qFloat - qIdx;

  const segFloat = localQ * 7.0;
  const segIdx = Math.min(6, Math.floor(segFloat));
  const segT = segFloat - segIdx;

  const pA = _quadPts[segIdx];
  const pB = _quadPts[segIdx + 1];

  let qx = pA[0] + (pB[0] - pA[0]) * segT;
  let qz = pA[1] + (pB[1] - pA[1]) * segT;

  const lenCur = Math.hypot(pB[0] - pA[0], pB[1] - pA[1]);
  if (lenCur > 1e-5) {
    const eCurX = (pB[0] - pA[0]) / lenCur;
    const eCurZ = (pB[1] - pA[1]) / lenCur;

    if (segIdx > 0) {
      const pPrev = _quadPts[segIdx - 1];
      const lenPrev = Math.hypot(pA[0] - pPrev[0], pA[1] - pPrev[1]);
      if (lenPrev > 1e-5) {
        const dFillet = Math.min(0.11, 0.22 * Math.min(lenPrev, lenCur));
        const cR = dFillet / lenCur;
        if (segT < cR) {
          const eInX = (pA[0] - pPrev[0]) / lenPrev;
          const eInZ = (pA[1] - pPrev[1]) / lenPrev;
          const oneMinusEta = 1.0 - segT / cR;
          const w = 0.25 * dFillet * oneMinusEta * oneMinusEta;
          qx += w * (eCurX - eInX);
          qz += w * (eCurZ - eInZ);
        }
      }
    }

    if (segIdx < 6) {
      const pNext = _quadPts[segIdx + 2];
      const lenNext = Math.hypot(pNext[0] - pB[0], pNext[1] - pB[1]);
      if (lenNext > 1e-5) {
        const dFillet = Math.min(0.11, 0.22 * Math.min(lenCur, lenNext));
        const cL = dFillet / lenCur;
        if (segT > 1.0 - cL) {
          const eOutX = (pNext[0] - pB[0]) / lenNext;
          const eOutZ = (pNext[1] - pB[1]) / lenNext;
          const xi = (segT - (1.0 - cL)) / cL;
          const w = 0.25 * dFillet * xi * xi;
          qx += w * (eOutX - eCurX);
          qz += w * (eOutZ - eCurZ);
        }
      }
    }
  }

  let wx, wz;
  if (qIdx === 0) {
    wx = qx;
    wz = qz;
  } else if (qIdx === 1) {
    wx = qz;
    wz = -qx;
  } else if (qIdx === 2) {
    wx = -qx;
    wz = -qz;
  } else {
    wx = -qz;
    wz = qx;
  }

  outVec.set(wx, prof.y, wz);
  return outVec;
}

export class MasterSurfaceKernel {
  /**
   * Primary Single Source of Spatial Truth (SSoST) Master Surface Evaluator:
   * Computes exact 3D position P(u, v) and outward unit normal N(u, v) = normalize(T_u x T_v).
   */
  static evaluateMasterSurface(
    u,
    v,
    targetPos = new THREE.Vector3(),
    targetNorm = new THREE.Vector3()
  ) {
    evalPositionOnly(u, v, targetPos);

    if (!targetNorm) return { position: targetPos, normal: null };

    if (v <= 0.002) {
      targetNorm.set(0, -1, 0);
      return { position: targetPos, normal: targetNorm };
    }
    if (v >= 0.998) {
      targetNorm.set(0, 1, 0);
      return { position: targetPos, normal: targetNorm };
    }

    const epsU = 0.00045;
    const epsV = 0.00035;

    evalPositionOnly(u + epsU, v, _pPlusU);
    evalPositionOnly(u - epsU, v, _pMinusU);
    _tU.subVectors(_pPlusU, _pMinusU);
    const lenTu = _tU.length();
    if (lenTu > 1e-12) _tU.multiplyScalar(1.0 / lenTu);

    const vLo = Math.max(0.0, v - epsV);
    const vHi = Math.min(1.0, v + epsV);
    evalPositionOnly(u, vHi, _pPlusV);
    evalPositionOnly(u, vLo, _pMinusV);
    _tV.subVectors(_pPlusV, _pMinusV);
    const lenTv = _tV.length();
    if (lenTv > 1e-12) _tV.multiplyScalar(1.0 / lenTv);

    targetNorm.crossVectors(_tU, _tV);
    if (targetNorm.lengthSq() < 1e-10) {
      const hLen = Math.hypot(_tU.x, _tU.z);
      if (hLen > 1e-8) {
        targetNorm.set(-_tU.z / hLen, 0.0, _tU.x / hLen);
      } else {
        targetNorm.set(0, 1, 0);
      }
    } else {
      targetNorm.normalize();
    }

    return { position: targetPos, normal: targetNorm };
  }

  /**
   * Exact 3D Signed Distance Field (Kernel 2 Hybrid SSoST) for spatial & manifold audits.
   */
  static evaluateMasterSDF(p) {
    const y = p.y;
    if (y < -18.0) return -18.0 - y;
    if (y > 50.0) return y - 50.0;

    const stations = MASTER_PROFILE_STATIONS;
    let bestIdx = 1;
    for (let i = 1; i < stations.length; i++) {
      if (stations[i].y >= y) {
        bestIdx = i;
        break;
      }
    }
    const s0 = stations[Math.max(0, bestIdx - 1)];
    const s1 = stations[bestIdx];
    const dy = Math.max(1e-5, s1.y - s0.y);
    const t = Math.max(0.0, Math.min(1.0, (y - s0.y) / dy));
    const vApprox = s0.v + (s1.v - s0.v) * t;
    const prof = sampleAxialProfile(vApprox);

    const dCore = MasterSurfaceBuilder.sdChamferBox2D(
      p.x,
      p.z,
      prof.coreHalf,
      prof.coreHalf,
      prof.coreChamfer
    );

    if (prof.towerActive <= 0.01) {
      return dCore;
    }

    const dTowers = MasterSurfaceBuilder.sdChamferBox2D(
      Math.abs(p.x) - prof.towerCenter,
      Math.abs(p.z) - prof.towerCenter,
      prof.towerHalf,
      prof.towerHalf,
      prof.towerChamfer
    );

    const dCombined = MasterSurfaceBuilder.opSmoothUnion(dCore, dTowers, 0.16);
    return (1.0 - prof.towerActive) * dCore + prof.towerActive * dCombined;
  }

  /**
   * Parametric SSoST Evaluator for the 4 Big Corner Pyramidal/Limas Roofs (Tier 7 — Mavic Photos #1, #2, #3, #5)
   * Centered at (±10.30, ±10.30), rising from y = 25.15m (eaveHalf = 8.25m) to y = 30.85m (ridgeCapHalf = 1.35m).
   */
  static createCornerTowerCrownEvaluator(quadrant) {
    const cx = quadrant.sx * CORNER_TOWER_CROWN_SPEC.centerXY;
    const cz = quadrant.sz * CORNER_TOWER_CROWN_SPEC.centerXY;

    return MasterSurfaceKernel.createHippedRoofEvaluator({
      cx,
      cz,
      halfX: CORNER_TOWER_CROWN_SPEC.eaveHalf,
      halfZ: CORNER_TOWER_CROWN_SPEC.eaveHalf,
      ridgeHalfX: CORNER_TOWER_CROWN_SPEC.ridgeCapHalf,
      ridgeHalfZ: CORNER_TOWER_CROWN_SPEC.ridgeCapHalf,
      eaveY: CORNER_TOWER_CROWN_SPEC.eaveBaseY,
      fasciaY: CORNER_TOWER_CROWN_SPEC.eaveFasciaY,
      ridgeY: CORNER_TOWER_CROWN_SPEC.roofTopY - 0.12,
      capTopY: CORNER_TOWER_CROWN_SPEC.roofTopY,
      curveExp: 1.07
    });
  }

  /**
   * Universal Watertight Parametric Hipped / Pyramidal Nusantara Roof Evaluator:
   * - v in [0.00 .. 0.25]: Closed horizontal underside soffit
   * - v in [0.25 .. 0.50]: Crisp eave fascia lip
   * - v in [0.50 .. 0.75]: Sloped terracotta hipped/pyramidal roof faces
   * - v in [0.75 .. 1.00]: Crisp white/terracotta ridge cap closure
   */
  static createHippedRoofEvaluator({
    cx = 0,
    cz = 0,
    halfX = 8.0,
    halfZ = 8.0,
    ridgeHalfX = 0.25,
    ridgeHalfZ = 0.25,
    eaveY = 5.6,
    fasciaY = 5.8,
    ridgeY = 10.5,
    capTopY = 10.65,
    rotY = 0.0,
    curveExp = 1.06
  }) {
    const stations = [
      { y: eaveY, hx: 0.001, hz: 0.001 },
      { y: eaveY, hx: halfX, hz: halfZ },
      { y: fasciaY, hx: halfX, hz: halfZ },
      { y: ridgeY, hx: ridgeHalfX, hz: ridgeHalfZ },
      { y: capTopY, hx: 0.001, hz: 0.001 }
    ];
    const cosR = Math.cos(rotY);
    const sinR = Math.sin(rotY);

    const evalLocal = (u, v, out) => {
      const clampedV = Math.max(0, Math.min(1, v));
      const maxI = stations.length - 1;
      const sc = clampedV * maxI;
      const i0 = Math.min(maxI - 1, Math.floor(sc));
      const t = sc - i0;
      const a = stations[i0];
      const b = stations[i0 + 1];
      const st = i0 === 2 ? 1.0 - Math.pow(1.0 - t, curveExp) : t;

      const y = a.y + (b.y - a.y) * t;
      const hx = Math.max(0.001, a.hx + (b.hx - a.hx) * st);
      const hz = Math.max(0.001, a.hz + (b.hz - a.hz) * st);

      let uw = u % 1.0;
      if (uw < 0) uw += 1.0;
      const qF = uw * 4.0;
      const qI = Math.min(3, Math.floor(qF));
      const lq = qF - qI;

      let lx, lz;
      if (qI === 0) {
        // +Z -> +X
        if (lq < 0.5) { lx = (lq / 0.5) * hx; lz = hz; }
        else { lx = hx; lz = (1.0 - (lq - 0.5) / 0.5) * hz; }
      } else if (qI === 1) {
        // +X -> -Z
        if (lq < 0.5) { lx = hx; lz = -(lq / 0.5) * hz; }
        else { lx = (1.0 - (lq - 0.5) / 0.5) * hx; lz = -hz; }
      } else if (qI === 2) {
        // -Z -> -X
        if (lq < 0.5) { lx = -(lq / 0.5) * hx; lz = -hz; }
        else { lx = -hx; lz = -(1.0 - (lq - 0.5) / 0.5) * hz; }
      } else {
        // -X -> +Z
        if (lq < 0.5) { lx = -hx; lz = (lq / 0.5) * hz; }
        else { lx = -(1.0 - (lq - 0.5) / 0.5) * hx; lz = hz; }
      }

      const rx = lx * cosR + lz * sinR;
      const rz = -lx * sinR + lz * cosR;
      out.set(cx + rx, y, cz + rz);
    };

    return (u, v, targetPos = new THREE.Vector3(), targetNorm = new THREE.Vector3()) => {
      evalLocal(u, v, targetPos);
      if (v <= 0.002) {
        targetNorm.set(0, -1, 0);
        return;
      }
      if (v >= 0.998) {
        targetNorm.set(0, 1, 0);
        return;
      }
      const pU1 = new THREE.Vector3(), pU0 = new THREE.Vector3();
      const pV1 = new THREE.Vector3(), pV0 = new THREE.Vector3();
      evalLocal(u + 0.001, v, pU1);
      evalLocal(u - 0.001, v, pU0);
      evalLocal(u, Math.min(1, v + 0.001), pV1);
      evalLocal(u, Math.max(0, v - 0.001), pV0);
      targetNorm.crossVectors(pU1.sub(pU0), pV1.sub(pV0)).normalize();
      if (targetNorm.lengthSq() < 0.1) targetNorm.set(0, 1, 0);
    };
  }

  /**
   * Parametric Octagonal Roof + Glass Pyramid Skylight Evaluator for the 45° SE Lakeside Pendopo
   * (Mavic Photos #1, #3, & #5: octagonal terracotta roof crowned by translucent glass pyramid skylight!).
   * - v in [0.00 .. 0.25]: Underside soffit
   * - v in [0.25 .. 0.50]: White eave fascia
   * - v in [0.50 .. 0.75]: Octagonal terracotta roof slope (r = outerR -> skylightR)
   * - v in [0.75 .. 1.00]: Translucent glass pyramid skylight cap (r = skylightR -> 0)
   */
  static createOctagonalSkylitRoofEvaluator({
    cx = 29.5,
    cz = 29.5,
    outerR = 8.2,
    skylightR = 2.25,
    eaveY = 3.75,
    fasciaY = 3.94,
    skylightBaseY = 6.85,
    skylightApexY = 8.15
  } = {}) {
    const stations = [
      { y: eaveY, r: 0.001 },
      { y: eaveY, r: outerR },
      { y: fasciaY, r: outerR },
      { y: skylightBaseY, r: skylightR },
      { y: skylightApexY, r: 0.001 }
    ];

    const evalLocal = (u, v, out) => {
      const clampedV = Math.max(0, Math.min(1, v));
      const maxI = stations.length - 1;
      const sc = clampedV * maxI;
      const i0 = Math.min(maxI - 1, Math.floor(sc));
      const t = sc - i0;
      const a = stations[i0];
      const b = stations[i0 + 1];
      const y = a.y + (b.y - a.y) * t;
      const r = Math.max(0.001, a.r + (b.r - a.r) * t);

      let uw = u % 1.0;
      if (uw < 0) uw += 1.0;
      const sides = 8;
      const segF = uw * sides;
      const sI = Math.min(sides - 1, Math.floor(segF));
      const st = segF - sI;
      const th0 = (sI / sides) * Math.PI * 2.0 + Math.PI / 8.0;
      const th1 = ((sI + 1) / sides) * Math.PI * 2.0 + Math.PI / 8.0;

      // Counter-clockwise in Three.js (+Z -> +X -> -Z -> -X): x = sin(th)*r, z = cos(th)*r
      const x0 = Math.sin(th0) * r;
      const z0 = Math.cos(th0) * r;
      const x1 = Math.sin(th1) * r;
      const z1 = Math.cos(th1) * r;

      out.set(cx + x0 + (x1 - x0) * st, y, cz + z0 + (z1 - z0) * st);
    };

    return (u, v, targetPos = new THREE.Vector3(), targetNorm = new THREE.Vector3()) => {
      evalLocal(u, v, targetPos);
      if (v <= 0.002) {
        targetNorm.set(0, -1, 0);
        return;
      }
      if (v >= 0.998) {
        targetNorm.set(0, 1, 0);
        return;
      }
      const pU1 = new THREE.Vector3(), pU0 = new THREE.Vector3();
      const pV1 = new THREE.Vector3(), pV0 = new THREE.Vector3();
      evalLocal(u + 0.001, v, pU1);
      evalLocal(u - 0.001, v, pU0);
      evalLocal(u, Math.min(1, v + 0.001), pV1);
      evalLocal(u, Math.max(0, v - 0.001), pV0);
      targetNorm.crossVectors(pU1.sub(pU0), pV1.sub(pV0)).normalize();
      if (targetNorm.lengthSq() < 0.1) targetNorm.set(0, 1, 0);
    };
  }
}
