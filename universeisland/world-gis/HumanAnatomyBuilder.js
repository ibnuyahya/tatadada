import * as THREE from 'three';
import { HUMAN_MAT } from './HumanShaderMaterial.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.4
 * ./world-gis/HumanAnatomyBuilder.js
 *
 * Seamless Watertight 22-Joint Dual-Bone Linear-Blend Skinned (LBS) Human Anatomy & Sartorial Builder:
 * - ZERO wooden-doll capsule seams: Continuous torso, open-front 3D layered UI Yellow Jacket ("Jakun"),
 *   continuous elbow-blended arms, 5-finger sculpted hands (thumb + 4 distinct fingers) + wristwatch,
 *   continuous knee-blended trousers OR flowing A-line pleated long skirt, and articulated metatarsal
 *   toe-roll campus sneakers!
 * - True 3D Facial Sculpture: Jawline, chin, upper/lower lip relief, cheekbones, recessed orbital sockets,
 *   real 3D spherical eyeballs (sclera, dark-amber iris, pupil, wet corneal catchlight), upper/lower
 *   3D eyelids, arched eyebrows, multi-station nose with nostril wings, 3D ears, optional 3D eyeglasses,
 *   and 4 hairstyles including chest-draped LBS Hijab shawl that turns cleanly without neck tearing!
 * - Dynamic 3D 8-Rib Curved Rain Umbrella (J_PROP_UMBRELLA) + Academic Props (Daypack, Open/Closed Book,
 *   Live-Viewfinder Smartphone, & UI Student Lanyard + KTM ID Badge).
 */

export const J_PELVIS = 0;
export const J_WAIST = 1;
export const J_CHEST = 2;
export const J_NECK = 3;
export const J_HEAD = 4;
export const J_SHOULDER_L = 5;
export const J_ELBOW_L = 6;
export const J_WRIST_L = 7;
export const J_SHOULDER_R = 8;
export const J_ELBOW_R = 9;
export const J_WRIST_R = 10;
export const J_HIP_L = 11;
export const J_KNEE_L = 12;
export const J_ANKLE_L = 13;
export const J_TOE_L = 14;
export const J_HIP_R = 15;
export const J_KNEE_R = 16;
export const J_ANKLE_R = 17;
export const J_TOE_R = 18;
export const J_SKIRT_HEM = 19;
export const J_PROP_MAIN = 20;
export const J_PROP_UMBRELLA = 21;
export const JOINTS_PER_NPC = 22;

// Exact Standing Bind-Pose Joint Pivots in Character Space (hScale = 1.0, facing +Z, soles flush at y = 0.000m)
export const BIND_PIVOTS = [
  [0.0, 0.890, 0.0],       // 0: J_PELVIS
  [0.0, 0.995, 0.0],       // 1: J_WAIST      (+0.105 from PELVIS)
  [0.0, 1.140, 0.0],       // 2: J_CHEST      (+0.145 from WAIST)
  [0.0, 1.435, 0.006],     // 3: J_NECK       (+0.295, +0.006 from CHEST)
  [0.0, 1.513, 0.008],     // 4: J_HEAD       (+0.078, +0.002 from NECK)
  [0.184, 1.386, -0.006],  // 5: J_SHOULDER_L (+0.184, +0.246, -0.006 from CHEST)
  [0.184, 1.128, -0.006],  // 6: J_ELBOW_L    (0, -0.258, 0 from SHOULDER_L)
  [0.184, 0.892, -0.006],  // 7: J_WRIST_L    (0, -0.236, 0 from ELBOW_L)
  [-0.184, 1.386, -0.006], // 8: J_SHOULDER_R (-0.184, +0.246, -0.006 from CHEST)
  [-0.184, 1.128, -0.006], // 9: J_ELBOW_R    (0, -0.258, 0 from SHOULDER_R)
  [-0.184, 0.892, -0.006], // 10: J_WRIST_R   (0, -0.236, 0 from ELBOW_R)
  [0.090, 0.860, 0.0],     // 11: J_HIP_L     (+0.090, -0.030, 0 from PELVIS)
  [0.090, 0.460, 0.0],     // 12: J_KNEE_L    (0, -0.400, 0 from HIP_L)
  [0.090, 0.065, 0.0],     // 13: J_ANKLE_L   (0, -0.395, 0 from KNEE_L)
  [0.090, 0.022, 0.095],   // 14: J_TOE_L     (0, -0.043, +0.095 from ANKLE_L)
  [-0.090, 0.860, 0.0],    // 15: J_HIP_R     (-0.090, -0.030, 0 from PELVIS)
  [-0.090, 0.460, 0.0],    // 16: J_KNEE_R    (0, -0.400, 0 from HIP_R)
  [-0.090, 0.065, 0.0],    // 17: J_ANKLE_R   (0, -0.395, 0 from KNEE_R)
  [-0.090, 0.022, 0.095],  // 18: J_TOE_R     (0, -0.043, +0.095 from ANKLE_R)
  [0.0, 0.310, 0.0],       // 19: J_SKIRT_HEM (0, -0.580, 0 from PELVIS)
  [0.0, 0.0, 0.0],         // 20: J_PROP_MAIN (prop-local origin)
  [0.0, 0.0, 0.0]          // 21: J_PROP_UMBRELLA (umbrella-local origin)
];

// Linear-safe photometrically calibrated Indonesian & UI Depok palette
export const HUMAN_PALETTE = {
  JAKUN_YELLOW: [0.90, 0.67, 0.10],
  JAKUN_LINING: [0.74, 0.53, 0.08],
  MAKARA_GOLD: [0.88, 0.68, 0.16],
  LANYARD_BLUE: [0.11, 0.27, 0.56],
  INNER_WHITE: [0.87, 0.86, 0.83],
  INNER_SKY: [0.64, 0.75, 0.86],
  INNER_NAVY: [0.14, 0.19, 0.29],
  BATIK_SOGAN: [0.43, 0.24, 0.14],
  BATIK_INDIGO: [0.16, 0.24, 0.38],
  BATIK_MAROON: [0.45, 0.15, 0.17],
  FORMAL_SHIRT: [0.72, 0.80, 0.88],
  CHINO_KHAKI: [0.46, 0.39, 0.30],
  TROUSER_NAVY: [0.11, 0.15, 0.22],
  TROUSER_CHARCOAL: [0.13, 0.14, 0.16],
  DENIM_BLUE: [0.17, 0.26, 0.39],
  SKIRT_NAVY: [0.11, 0.14, 0.20],
  SKIRT_CHARCOAL: [0.14, 0.15, 0.17],
  SKIRT_ESPRESSO: [0.20, 0.15, 0.13],
  SKIN_TONES: [
    [0.77, 0.55, 0.40], // Warm Kuning Langsat
    [0.71, 0.49, 0.35], // Natural Sawo Matang Terang
    [0.65, 0.44, 0.31], // Classic Sawo Matang
    [0.74, 0.52, 0.38], // Warm Olive Tan
    [0.60, 0.40, 0.28]  // Rich Tropical Tan
  ],
  SCLERA_WHITE: [0.91, 0.89, 0.86],
  IRIS_BROWN: [0.22, 0.12, 0.07],
  PUPIL_BLACK: [0.03, 0.03, 0.04],
  BROW_LASH: [0.09, 0.07, 0.06],
  HAIR_JET: [0.06, 0.055, 0.055],
  HAIR_ESPRESSO: [0.10, 0.075, 0.065],
  HIJAB_CREAM: [0.80, 0.71, 0.58],
  HIJAB_MAROON: [0.46, 0.15, 0.18],
  HIJAB_NAVY: [0.13, 0.20, 0.32],
  HIJAB_SAGE: [0.38, 0.46, 0.37],
  HIJAB_DUSTY_ROSE: [0.62, 0.42, 0.43],
  HIJAB_MOCHA: [0.44, 0.33, 0.26],
  CIPUT_DARK: [0.16, 0.16, 0.18],
  METAL_DARK: [0.18, 0.19, 0.22],
  METAL_SILVER: [0.72, 0.74, 0.77],
  SHOE_NAVY: [0.11, 0.14, 0.20],
  SHOE_BLACK: [0.09, 0.09, 0.11],
  SHOE_WHITE_TRIM: [0.82, 0.81, 0.78],
  SHOE_GUM_SOLE: [0.58, 0.46, 0.33],
  BAG_CHARCOAL: [0.14, 0.16, 0.20],
  BAG_OLIVE: [0.20, 0.24, 0.19],
  BOOK_CRIMSON: [0.54, 0.19, 0.16],
  BOOK_NAVY: [0.14, 0.22, 0.38],
  PHONE_BODY: [0.11, 0.13, 0.16],
  PHONE_SCREEN: [0.55, 0.76, 0.92],
  UMBRELLA_NAVY: [0.11, 0.18, 0.32],
  UMBRELLA_MAROON: [0.44, 0.13, 0.17],
  UMBRELLA_SLATE: [0.18, 0.21, 0.25]
};

function smooth01(edge0, edge1, x) {
  const t = Math.max(0.0, Math.min(1.0, (x - edge0) / (edge1 - edge0 || 1e-6)));
  return t * t * (3.0 - 2.0 * t);
}

function lerp3(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/**
 * Accumulates watertight sub-meshes with per-vertex dual-bone LBS weights `(j0, j1, w0)`,
 * garment-locked `uv`, and shader material-class metadata `aHumanMeta`.
 */
class SkinnedMeshPartBuilder {
  constructor() {
    this.pos = [];
    this.norm = [];
    this.col = [];
    this.uv = [];
    this.meta = [];
    this.j0 = [];
    this.j1 = [];
    this.w0 = [];
    this.idx = [];
  }

  /**
   * Finalizes normals on a sub-part and appends it cleanly into the character accumulator.
   */
  appendSubPart(subPos, subCol, subUv, subMeta, subJ0, subJ1, subW0, subIdx) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(subPos), 3));
    geo.setIndex(subIdx);
    geo.computeVertexNormals();

    const nArr = geo.attributes.normal.array;
    const vCount = subPos.length / 3;
    const base = this.pos.length / 3;

    for (let i = 0; i < vCount; i++) {
      const i3 = i * 3;
      const i2 = i * 2;
      this.pos.push(subPos[i3], subPos[i3 + 1], subPos[i3 + 2]);

      let nx = nArr[i3];
      let ny = nArr[i3 + 1];
      let nz = nArr[i3 + 2];
      const len = Math.hypot(nx, ny, nz);
      if (len > 1e-8) {
        nx /= len;
        ny /= len;
        nz /= len;
      } else {
        nx = 0.0;
        ny = 1.0;
        nz = 0.0;
      }
      this.norm.push(nx, ny, nz);
      this.col.push(subCol[i3], subCol[i3 + 1], subCol[i3 + 2]);
      this.uv.push(subUv[i2], subUv[i2 + 1]);
      this.meta.push(subMeta[i3], subMeta[i3 + 1], subMeta[i3 + 2]);
      this.j0.push(subJ0[i]);
      this.j1.push(subJ1[i]);
      this.w0.push(subW0[i]);
    }

    for (let i = 0; i < subIdx.length; i++) {
      this.idx.push(base + subIdx[i]);
    }
  }

  /**
   * Lofts a closed watertight tube along arbitrary 3D cross-section rings (along Y, Z, or 3D curve!).
   * Supports radial relief `modFn(th)`, Z/X displacement `offsetFn(th)`, per-angle `colorFn(th)`,
   * per-angle `matFn(th)`, `aoFn(th)`, and continuous 2-bone LBS `weightFn(st, tRing, th, px, py, pz)`.
   */
  addWatertightLoft(stations, radialSegs, opts = {}) {
    const {
      defaultMat = HUMAN_MAT.SKIN,
      defaultJoint = J_PELVIS,
      weightFn = null,
      axis = 'Y', // 'Y' (vertical rings) | 'Z' (longitudinal rings, e.g. shoe/nose) | 'CURVE' (3D Frenet tube)
      capStart = true,
      capEnd = true,
      uvScaleU = 1.0,
      uvScaleV = 1.0
    } = opts;

    const subPos = [];
    const subCol = [];
    const subUv = [];
    const subMeta = [];
    const subJ0 = [];
    const subJ1 = [];
    const subW0 = [];
    const subIdx = [];

    const pushV = (px, py, pz, col, u, v, matClass, ao, jA, jB, wA) => {
      const id = subPos.length / 3;
      subPos.push(px, py, pz);
      subCol.push(col[0], col[1], col[2]);
      subUv.push(u, v);
      subMeta.push(matClass, ao, 0.5);
      subJ0.push(jA);
      subJ1.push(jB);
      subW0.push(wA);
      return id;
    };

    const nSt = stations.length;
    const ringGrid = [];

    for (let is = 0; is < nSt; is++) {
      const st = stations[is];
      const tRing = nSt > 1 ? is / (nSt - 1) : 0.0;
      const cx = st.cx ?? 0.0;
      const cy = st.cy ?? st.y ?? 0.0;
      const cz = st.cz ?? st.z ?? 0.0;

      // Local cross-section basis vectors (rightVec, fwdVec)
      let rightX = 1.0, rightY = 0.0, rightZ = 0.0;
      let fwdX = 0.0, fwdY = 0.0, fwdZ = 1.0;

      if (axis === 'Z') {
        // Cross-section lies in XY plane along Z axis (+X right, +Y up)
        rightX = 1.0; rightY = 0.0; rightZ = 0.0;
        fwdX = 0.0; fwdY = 1.0; fwdZ = 0.0;
      } else if (axis === 'CURVE') {
        const prev = stations[Math.max(0, is - 1)];
        const next = stations[Math.min(nSt - 1, is + 1)];
        let tx = (next.cx ?? 0) - (prev.cx ?? 0);
        let ty = (next.cy ?? next.y ?? 0) - (prev.cy ?? prev.y ?? 0);
        let tz = (next.cz ?? next.z ?? 0) - (prev.cz ?? prev.z ?? 0);
        const tLen = Math.hypot(tx, ty, tz) || 1.0;
        tx /= tLen; ty /= tLen; tz /= tLen;

        // Reference up/side vector
        const refX = Math.abs(tx) < 0.82 ? 1.0 : 0.0;
        const refY = Math.abs(tx) < 0.82 ? 0.0 : 1.0;
        const refZ = 0.0;
        // fwd = normalize(tangent x ref)
        fwdX = ty * refZ - tz * refY;
        fwdY = tz * refX - tx * refZ;
        fwdZ = tx * refY - ty * refX;
        const fLen = Math.hypot(fwdX, fwdY, fwdZ) || 1.0;
        fwdX /= fLen; fwdY /= fLen; fwdZ /= fLen;
        // right = fwd x tangent
        rightX = fwdY * tz - fwdZ * ty;
        rightY = fwdZ * tx - fwdX * tz;
        rightZ = fwdX * ty - fwdY * tx;
      }

      const row = [];
      for (let k = 0; k < radialSegs; k++) {
        const uNorm = k / radialSegs;
        const th = uNorm * Math.PI * 2.0;
        const sTh = Math.sin(th);
        const cTh = Math.cos(th);
        const m = typeof st.modFn === 'function' ? st.modFn(th) : 1.0;
        const rx = (st.rx ?? st.r ?? 0.02) * m;
        const rz = (st.rz ?? st.ry ?? st.r ?? 0.02) * m;

        let px = cx + (rightX * sTh * rx + fwdX * cTh * rz);
        let py = cy + (rightY * sTh * rx + fwdY * cTh * rz);
        let pz = cz + (rightZ * sTh * rx + fwdZ * cTh * rz);

        if (typeof st.offsetFn === 'function') {
          const off = st.offsetFn(th);
          px += off[0];
          py += off[1];
          pz += off[2];
        }

        const col = typeof st.colorFn === 'function' ? st.colorFn(th) : st.color;
        const matClass =
          typeof st.matFn === 'function' ? st.matFn(th) : (st.mat ?? defaultMat);
        const ao = typeof st.aoFn === 'function' ? st.aoFn(th) : (st.ao ?? 1.0);
        const wSpec = weightFn
          ? weightFn(st, tRing, th, px, py, pz)
          : st.weights || [defaultJoint, defaultJoint, 1.0];

        row.push(
          pushV(
            px,
            py,
            pz,
            col,
            uNorm * uvScaleU,
            tRing * uvScaleV,
            matClass,
            ao,
            wSpec[0],
            wSpec[1],
            wSpec[2]
          )
        );
      }
      ringGrid.push(row);
    }

    // Determine natural winding order from ring 0 -> ring N-1 progression
    let flipWinding = false;
    if (axis === 'Y') {
      const y0 = stations[0].cy ?? stations[0].y ?? 0;
      const y1 = stations[nSt - 1].cy ?? stations[nSt - 1].y ?? 0;
      flipWinding = y1 < y0;
    } else if (axis === 'Z') {
      const z0 = stations[0].cz ?? stations[0].z ?? 0;
      const z1 = stations[nSt - 1].cz ?? stations[nSt - 1].z ?? 0;
      flipWinding = z1 > z0;
    }

    for (let is = 0; is < nSt - 1; is++) {
      const rA = ringGrid[is];
      const rB = ringGrid[is + 1];
      for (let k = 0; k < radialSegs; k++) {
        const kNext = (k + 1) % radialSegs;
        const a = rA[k];
        const b = rA[kNext];
        const c = rB[kNext];
        const d = rB[k];
        if (!flipWinding) {
          subIdx.push(a, b, c, a, c, d);
        } else {
          subIdx.push(a, d, c, a, c, b);
        }
      }
    }

    const addPoleCap = (stIdx, isEnd) => {
      const st = stations[stIdx];
      const cx = st.cx ?? 0.0;
      const cy = st.cy ?? st.y ?? 0.0;
      const cz = st.cz ?? st.z ?? 0.0;
      const row = ringGrid[stIdx];

      // Compute average ring center + tiny outward pole dome
      let avgX = 0, avgY = 0, avgZ = 0;
      for (let k = 0; k < radialSegs; k++) {
        const vi = row[k] * 3;
        avgX += subPos[vi];
        avgY += subPos[vi + 1];
        avgZ += subPos[vi + 2];
      }
      avgX /= radialSegs;
      avgY /= radialSegs;
      avgZ /= radialSegs;

      const prevIdx = isEnd ? Math.max(0, stIdx - 1) : Math.min(nSt - 1, stIdx + 1);
      const stAdj = stations[prevIdx];
      let dx = cx - (stAdj.cx ?? 0.0);
      let dy = cy - (stAdj.cy ?? stAdj.y ?? 0.0);
      let dz = cz - (stAdj.cz ?? stAdj.z ?? 0.0);
      const dLen = Math.hypot(dx, dy, dz) || 1.0;
      const dome = st.poleDome ?? 0.005;

      const px = avgX + (dx / dLen) * dome;
      const py = avgY + (dy / dLen) * dome;
      const pz = avgZ + (dz / dLen) * dome;

      const col = typeof st.colorFn === 'function' ? st.colorFn(0.0) : st.color;
      const matClass = typeof st.matFn === 'function' ? st.matFn(0.0) : (st.mat ?? defaultMat);
      const ao = st.ao ?? 1.0;
      const wSpec = weightFn
        ? weightFn(st, isEnd ? 1.0 : 0.0, 0.0, px, py, pz)
        : st.weights || [defaultJoint, defaultJoint, 1.0];

      const pCenter = pushV(
        px,
        py,
        pz,
        col,
        0.5,
        isEnd ? uvScaleV : 0.0,
        matClass,
        ao,
        wSpec[0],
        wSpec[1],
        wSpec[2]
      );

      for (let k = 0; k < radialSegs; k++) {
        const kNext = (k + 1) % radialSegs;
        const outwardEnd = !flipWinding ? isEnd : !isEnd;
        if (outwardEnd) {
          subIdx.push(pCenter, row[k], row[kNext]);
        } else {
          subIdx.push(pCenter, row[kNext], row[k]);
        }
      }
    };

    if (capStart) addPoleCap(0, false);
    if (capEnd) addPoleCap(nSt - 1, true);

    this.appendSubPart(subPos, subCol, subUv, subMeta, subJ0, subJ1, subW0, subIdx);
  }

  /**
   * Builds a true 3D double-walled watertight open-arc garment shell (used for the Open-Front
   * Layered UI Yellow Jacket "Jakun" with notched lapels, and Draped Pleated Hijab Hood/Shawl)
   * so both outer surface, inner lining, top/bottom hems, and front edges have real 3D thickness!
   */
  addThickOpenGarmentShell(stations, arcSegs, opts = {}) {
    const {
      thickness = 0.007,
      outerMat = HUMAN_MAT.JAKUN_TWILL,
      innerMat = HUMAN_MAT.JAKUN_TWILL,
      innerColorScale = 0.80,
      weightFn = null,
      defaultJoint = J_CHEST,
      uvScaleU = 1.0,
      uvScaleV = 1.0
    } = opts;

    const subPos = [];
    const subCol = [];
    const subUv = [];
    const subMeta = [];
    const subJ0 = [];
    const subJ1 = [];
    const subW0 = [];
    const subIdx = [];

    const pushV = (px, py, pz, col, u, v, mat, ao, jA, jB, wA) => {
      const id = subPos.length / 3;
      subPos.push(px, py, pz);
      subCol.push(col[0], col[1], col[2]);
      subUv.push(u, v);
      subMeta.push(mat, ao, 0.5);
      subJ0.push(jA);
      subJ1.push(jB);
      subW0.push(wA);
      return id;
    };

    const nSt = stations.length;
    const outerGrid = [];
    const innerGrid = [];

    for (let is = 0; is < nSt; is++) {
      const st = stations[is];
      const tV = nSt > 1 ? is / (nSt - 1) : 0.0;
      const thStart = st.thStart ?? 0.28;
      const thEnd = st.thEnd ?? Math.PI * 2.0 - 0.28;
      const cx = st.cx ?? 0.0;
      const cz = st.cz ?? 0.0;
      const oRow = [];
      const iRow = [];

      for (let ia = 0; ia <= arcSegs; ia++) {
        const tU = ia / arcSegs;
        const th = thStart + (thEnd - thStart) * tU;
        const sTh = Math.sin(th);
        const cTh = Math.cos(th);
        const m = typeof st.modFn === 'function' ? st.modFn(th, tU) : 1.0;

        const rxOut = st.rx * m;
        const rzOut = st.rz * m;
        const pxOut = cx + sTh * rxOut;
        const pzOut = cz + cTh * rzOut;

        const rxIn = Math.max(0.015, rxOut - thickness);
        const rzIn = Math.max(0.015, rzOut - thickness);
        const pxIn = cx + sTh * rxIn;
        const pzIn = cz + cTh * rzIn;

        const colOut = typeof st.colorFn === 'function' ? st.colorFn(th, tU) : st.color;
        const colIn = st.innerColor || [
          colOut[0] * innerColorScale,
          colOut[1] * innerColorScale,
          colOut[2] * innerColorScale
        ];
        const matOut = typeof st.matFn === 'function' ? st.matFn(th, tU) : (st.mat ?? outerMat);
        const aoOut = typeof st.aoFn === 'function' ? st.aoFn(th, tU) : (st.ao ?? 1.0);
        const wSpec = weightFn
          ? weightFn(st, tV, th, pxOut, st.y, pzOut)
          : [defaultJoint, defaultJoint, 1.0];

        oRow.push(
          pushV(
            pxOut,
            st.y,
            pzOut,
            colOut,
            tU * uvScaleU,
            tV * uvScaleV,
            matOut,
            aoOut,
            wSpec[0],
            wSpec[1],
            wSpec[2]
          )
        );
        iRow.push(
          pushV(
            pxIn,
            st.y,
            pzIn,
            colIn,
            tU * uvScaleU,
            tV * uvScaleV,
            innerMat,
            aoOut * 0.78,
            wSpec[0],
            wSpec[1],
            wSpec[2]
          )
        );
      }
      outerGrid.push(oRow);
      innerGrid.push(iRow);
    }

    const yAsc = stations[nSt - 1].y >= stations[0].y;

    // Outer & Inner shells
    for (let is = 0; is < nSt - 1; is++) {
      for (let ia = 0; ia < arcSegs; ia++) {
        const oa = outerGrid[is][ia];
        const ob = outerGrid[is][ia + 1];
        const oc = outerGrid[is + 1][ia + 1];
        const od = outerGrid[is + 1][ia];

        const ia0 = innerGrid[is][ia];
        const ib0 = innerGrid[is][ia + 1];
        const ic0 = innerGrid[is + 1][ia + 1];
        const id0 = innerGrid[is + 1][ia];

        if (yAsc) {
          subIdx.push(oa, ob, oc, oa, oc, od);
          subIdx.push(ia0, id0, ic0, ia0, ic0, ib0);
        } else {
          subIdx.push(oa, od, oc, oa, oc, ob);
          subIdx.push(ia0, ib0, ic0, ia0, ic0, id0);
        }
      }
    }

    // Left & Right front placket/edge thickness walls (ia = 0 and ia = arcSegs)
    for (let is = 0; is < nSt - 1; is++) {
      const oL0 = outerGrid[is][0];
      const oL1 = outerGrid[is + 1][0];
      const iL0 = innerGrid[is][0];
      const iL1 = innerGrid[is + 1][0];

      const oR0 = outerGrid[is][arcSegs];
      const oR1 = outerGrid[is + 1][arcSegs];
      const iR0 = innerGrid[is][arcSegs];
      const iR1 = innerGrid[is + 1][arcSegs];

      if (yAsc) {
        subIdx.push(iL0, oL0, oL1, iL0, oL1, iL1);
        subIdx.push(oR0, iR0, iR1, oR0, iR1, oR1);
      } else {
        subIdx.push(oL0, iL0, iL1, oL0, iL1, oL1);
        subIdx.push(iR0, oR0, oR1, iR0, oR1, iR1);
      }
    }

    // Bottom & Top hem thickness rims (is = 0 and is = nSt - 1)
    for (let ia = 0; ia < arcSegs; ia++) {
      const ob0 = outerGrid[0][ia];
      const ob1 = outerGrid[0][ia + 1];
      const ib0 = innerGrid[0][ia];
      const ib1 = innerGrid[0][ia + 1];

      const ot0 = outerGrid[nSt - 1][ia];
      const ot1 = outerGrid[nSt - 1][ia + 1];
      const it0 = innerGrid[nSt - 1][ia];
      const it1 = innerGrid[nSt - 1][ia + 1];

      if (yAsc) {
        subIdx.push(ib0, ib1, ob1, ib0, ob1, ob0);
        subIdx.push(ot0, ot1, it1, ot0, it1, it0);
      } else {
        subIdx.push(ob0, ob1, ib1, ob0, ib1, ib0);
        subIdx.push(it0, it1, ot1, it0, ot1, ot0);
      }
    }

    this.appendSubPart(subPos, subCol, subUv, subMeta, subJ0, subJ1, subW0, subIdx);
  }
}

export class HumanAnatomyBuilder {
  /**
   * Builds a complete, seamless 22-joint Linear-Blend Skinned (LBS) Sivitas Akademika human
   * BufferGeometry bundle (~8,600 sculpted triangles) customized to `profile`.
   */
  static buildSkinnedCharacter(profile) {
    const builder = new SkinnedMeshPartBuilder();

    HumanAnatomyBuilder._addContinuousTorsoAndGarments(builder, profile);
    HumanAnatomyBuilder._addSeamlessArmsAnd5FingerHands(builder, profile);
    HumanAnatomyBuilder._addLowerBodyAndToeRollShoes(builder, profile);
    HumanAnatomyBuilder._addSculptedHeadFaceEyesAndHair(builder, profile);
    HumanAnatomyBuilder._addMainAcademicProp(builder, profile);
    HumanAnatomyBuilder._addDynamicRainUmbrella(builder, profile);

    return {
      pos: new Float32Array(builder.pos),
      norm: new Float32Array(builder.norm),
      col: new Float32Array(builder.col),
      uv: new Float32Array(builder.uv),
      meta: new Float32Array(builder.meta),
      j0: new Uint8Array(builder.j0),
      j1: new Uint8Array(builder.j1),
      w0: new Float32Array(builder.w0),
      idx: new Uint32Array(builder.idx)
    };
  }

  /**
   * Continuous Spine LBS Weight Resolver across Pelvis (0.89m) -> Waist (0.995m) -> Chest (1.14m..1.435m)
   */
  static _torsoSpineWeights(py) {
    if (py < 0.945) {
      const w = smooth01(0.875, 0.945, py);
      return [J_WAIST, J_PELVIS, w];
    }
    if (py < 1.135) {
      const w = smooth01(1.035, 1.135, py);
      return [J_CHEST, J_WAIST, w];
    }
    return [J_CHEST, J_CHEST, 1.0];
  }

  /**
   * 1. CONTINUOUS TORSO, INNER SHIRT / BATIK SHIRT, TAILORED BELT, LANYARD KTM,
   *    AND DOUBLE-WALLED 3D OPEN-FRONT UI ALMA MATER YELLOW JACKET ("JAKUN") + 3D MAKARA CREST!
   */
  static _addContinuousTorsoAndGarments(builder, profile) {
    const {
      garmentStyle, // 'JAKUN_OPEN' | 'BATIK_LECTURER' | 'FORMAL_SHIRT'
      torsoMain,
      innerShirtColor,
      lowerColor,
      skinColor,
      hasLanyard,
      buildScaleX = 1.0
    } = profile;

    const isJakun = garmentStyle === 'JAKUN_OPEN';
    const isBatik = garmentStyle === 'BATIK_LECTURER';
    const baseTorsoColor = isJakun ? innerShirtColor : torsoMain;
    const baseTorsoMat = isBatik
      ? HUMAN_MAT.BATIK_MOTIF
      : isJakun
        ? HUMAN_MAT.FABRIC_LOWER
        : HUMAN_MAT.HIJAB_SILK;

    const beltLeather = HUMAN_PALETTE.SHOE_BLACK;
    const buckleMetal = HUMAN_PALETTE.METAL_SILVER;

    // Continuous 15-station inner torso from hip/pelvis (0.825m) to shirt collar & neck base (1.432m)
    const torsoStations = [
      { y: 0.825, rx: 0.134 * buildScaleX, rz: 0.096, color: lowerColor, mat: HUMAN_MAT.FABRIC_LOWER, ao: 0.84 },
      { y: 0.865, rx: 0.141 * buildScaleX, rz: 0.101, color: lowerColor, mat: HUMAN_MAT.FABRIC_LOWER, ao: 0.92 },
      // Tailored belt + silver front belt buckle!
      {
        y: 0.902,
        rx: 0.139 * buildScaleX,
        rz: 0.099,
        colorFn: (th) => {
          const absTh = Math.abs(th > Math.PI ? th - Math.PI * 2.0 : th);
          return absTh < 0.13 ? buckleMetal : beltLeather;
        },
        matFn: (th) => {
          const absTh = Math.abs(th > Math.PI ? th - Math.PI * 2.0 : th);
          return absTh < 0.13 ? HUMAN_MAT.METAL_ACCENT : HUMAN_MAT.SHOE_BAG;
        },
        ao: 0.90
      },
      { y: 0.926, rx: 0.136 * buildScaleX, rz: 0.097, color: baseTorsoColor, mat: baseTorsoMat, ao: 0.88 },
      {
        y: 0.985,
        rx: 0.132 * buildScaleX,
        rz: 0.094,
        modFn: (th) => 1.0 + 0.015 * Math.sin(5.0 * th),
        color: baseTorsoColor,
        mat: baseTorsoMat,
        ao: 0.95
      },
      {
        y: 1.060,
        rx: 0.137 * buildScaleX,
        rz: 0.098,
        modFn: (th) => 1.0 + 0.014 * Math.sin(5.0 * th + 0.4),
        color: baseTorsoColor,
        mat: baseTorsoMat,
        ao: 0.98
      },
      {
        y: 1.145,
        rx: 0.148 * buildScaleX,
        rz: 0.105,
        cz: 0.003,
        colorFn: (th) => {
          if (!isJakun) {
            const absTh = Math.abs(th > Math.PI ? th - Math.PI * 2.0 : th);
            if (absTh < 0.045) return lerp3(baseTorsoColor, HUMAN_PALETTE.INNER_WHITE, 0.35);
          }
          return baseTorsoColor;
        },
        mat: baseTorsoMat,
        ao: 1.0
      },
      {
        y: 1.225,
        rx: 0.158 * buildScaleX,
        rz: 0.110,
        cz: 0.005,
        modFn: (th) => {
          // Subtle anatomical pectoral / chest fullness on +Z
          const c = Math.cos(th);
          return c > 0.2 ? 1.0 + 0.028 * c : 1.0;
        },
        color: baseTorsoColor,
        mat: baseTorsoMat,
        ao: 1.0
      },
      { y: 1.305, rx: 0.168 * buildScaleX, rz: 0.107, cz: 0.003, color: baseTorsoColor, mat: baseTorsoMat, ao: 1.0 },
      { y: 1.375, rx: 0.172 * buildScaleX, rz: 0.099, cz: -0.002, color: baseTorsoColor, mat: baseTorsoMat, ao: 0.98 },
      { y: 1.408, rx: 0.126 * buildScaleX, rz: 0.082, cz: -0.001, color: baseTorsoColor, mat: baseTorsoMat, ao: 0.94 },
      {
        y: 1.430,
        rx: 0.068,
        rz: 0.066,
        cz: 0.003,
        colorFn: (th) => {
          const absTh = Math.abs(th > Math.PI ? th - Math.PI * 2.0 : th);
          return absTh < 0.38 ? skinColor : baseTorsoColor;
        },
        matFn: (th) => {
          const absTh = Math.abs(th > Math.PI ? th - Math.PI * 2.0 : th);
          return absTh < 0.38 ? HUMAN_MAT.SKIN : baseTorsoMat;
        },
        ao: 0.88
      }
    ];

    builder.addWatertightLoft(torsoStations, 26, {
      defaultMat: baseTorsoMat,
      uvScaleU: 2.0,
      uvScaleV: 1.6,
      weightFn: (_st, _t, _th, _px, py) => HumanAnatomyBuilder._torsoSpineWeights(py)
    });

    // Optional UI Student Lanyard Strap + White/Blue KTM ID Card hanging on chest!
    if (hasLanyard) {
      const ktmStations = [
        { y: 1.155, cz: 0.111, rx: 0.025, rz: 0.003, color: HUMAN_PALETTE.INNER_WHITE, mat: HUMAN_MAT.SHOE_BAG },
        { y: 1.182, cz: 0.112, rx: 0.026, rz: 0.003, color: HUMAN_PALETTE.INNER_WHITE, mat: HUMAN_MAT.SHOE_BAG },
        { y: 1.212, cz: 0.112, rx: 0.026, rz: 0.003, color: HUMAN_PALETTE.LANYARD_BLUE, mat: HUMAN_MAT.CREST_GOLD },
        { y: 1.224, cz: 0.111, rx: 0.008, rz: 0.003, color: HUMAN_PALETTE.METAL_SILVER, mat: HUMAN_MAT.METAL_ACCENT }
      ];
      builder.addWatertightLoft(ktmStations, 10, {
        defaultJoint: J_CHEST,
        uvScaleU: 1.0,
        uvScaleV: 1.0
      });
    }

    // 2. TRUE 3D DOUBLE-WALLED OPEN-FRONT UI ALMA MATER YELLOW JACKET ("JAKUN") + 3D MAKARA CREST!
    if (isJakun) {
      const jakunCol = torsoMain;
      const jakunDark = HUMAN_PALETTE.JAKUN_LINING;

      const jakunShellStations = [
        {
          y: 0.882,
          rx: 0.149 * buildScaleX,
          rz: 0.109,
          thStart: 0.24,
          thEnd: Math.PI * 2.0 - 0.24,
          modFn: (th) => 1.0 + 0.016 * Math.cos(8.0 * th),
          color: jakunDark,
          ao: 0.88
        },
        {
          y: 0.955,
          rx: 0.146 * buildScaleX,
          rz: 0.107,
          thStart: 0.25,
          thEnd: Math.PI * 2.0 - 0.25,
          modFn: (th) => {
            // Side welt pocket ridges at th ~ +-0.85
            const sTh = th > Math.PI ? th - Math.PI * 2.0 : th;
            const pocket = Math.abs(Math.abs(sTh) - 0.82) < 0.18 ? 1.028 : 1.0;
            return pocket * (1.0 + 0.015 * Math.sin(6.0 * th));
          },
          color: jakunCol,
          ao: 0.96
        },
        {
          y: 1.055,
          rx: 0.150 * buildScaleX,
          rz: 0.111,
          cz: 0.002,
          thStart: 0.27,
          thEnd: Math.PI * 2.0 - 0.27,
          modFn: (th, tU) => (tU < 0.06 || tU > 0.94 ? 1.032 : 1.0), // Raised front zipper/button placket!
          color: jakunCol,
          ao: 0.99
        },
        {
          y: 1.165,
          rx: 0.161 * buildScaleX,
          rz: 0.118,
          cz: 0.004,
          thStart: 0.30,
          thEnd: Math.PI * 2.0 - 0.30,
          modFn: (_th, tU) => (tU < 0.08 || tU > 0.92 ? 1.045 : 1.0), // Turned-out lower lapel fold!
          color: jakunCol,
          ao: 1.0
        },
        {
          y: 1.255,
          rx: 0.172 * buildScaleX,
          rz: 0.121,
          cz: 0.005,
          thStart: 0.34,
          thEnd: Math.PI * 2.0 - 0.34,
          modFn: (_th, tU) => (tU < 0.09 || tU > 0.91 ? 1.055 : 1.0), // Crisp 3D notched blazer lapel!
          colorFn: (th) => {
            // Left-chest breast pocket seam under Makara UI emblem (+X side: th in [0.48..0.82])
            if (th > 0.48 && th < 0.82) return lerp3(jakunCol, jakunDark, 0.22);
            return jakunCol;
          },
          ao: 1.0
        },
        {
          y: 1.345,
          rx: 0.181 * buildScaleX,
          rz: 0.114,
          cz: 0.001,
          thStart: 0.39,
          thEnd: Math.PI * 2.0 - 0.39,
          modFn: (_th, tU) => (tU < 0.08 || tU > 0.92 ? 1.042 : 1.0),
          color: jakunCol,
          ao: 1.0
        },
        {
          y: 1.402,
          rx: 0.136 * buildScaleX,
          rz: 0.092,
          cz: -0.002,
          thStart: 0.44,
          thEnd: Math.PI * 2.0 - 0.44,
          color: jakunCol,
          ao: 0.96
        },
        // Stand-up / Folded UI Jakun Collar around neck!
        {
          y: 1.448,
          rx: 0.078,
          rz: 0.076,
          cz: 0.002,
          thStart: 0.52,
          thEnd: Math.PI * 2.0 - 0.52,
          color: jakunDark,
          ao: 0.92
        }
      ];

      builder.addThickOpenGarmentShell(jakunShellStations, 24, {
        thickness: 0.008,
        outerMat: HUMAN_MAT.JAKUN_TWILL,
        innerMat: HUMAN_MAT.JAKUN_TWILL,
        innerColorScale: 0.76,
        uvScaleU: 2.2,
        uvScaleV: 1.5,
        weightFn: (_st, _t, _th, _px, py) => HumanAnatomyBuilder._torsoSpineWeights(py)
      });

      // Physical 3D Raised Embroidered Makara UI Shield Badge on Left Breast (+X = +0.086, y = 1.248, +Z = +0.108)!
      const crestStations = [
        { cx: 0.086 * buildScaleX, cy: 1.248, cz: 0.104, rx: 0.022, ry: 0.026, color: jakunDark, mat: HUMAN_MAT.CREST_GOLD },
        { cx: 0.086 * buildScaleX, cy: 1.248, cz: 0.111, rx: 0.020, ry: 0.024, color: HUMAN_PALETTE.MAKARA_GOLD, mat: HUMAN_MAT.CREST_GOLD },
        { cx: 0.086 * buildScaleX, cy: 1.248, cz: 0.114, rx: 0.012, ry: 0.015, color: HUMAN_PALETTE.MAKARA_GOLD, mat: HUMAN_MAT.CREST_GOLD }
      ];
      builder.addWatertightLoft(crestStations, 12, {
        axis: 'Z',
        defaultJoint: J_CHEST,
        defaultMat: HUMAN_MAT.CREST_GOLD
      });
    }
  }

  /**
   * 2. CONTINUOUS SEAMLESS LBS ARMS (Shoulder -> Elbow Crease -> Rolled Cuff -> Forearm)
   *    AND TRUE 5-FINGER SCULPTED HANDS (Palm + Opposable Thumb + 4 Individual Fingers + Left Wristwatch)!
   */
  static _addSeamlessArmsAnd5FingerHands(builder, profile) {
    const { garmentStyle, torsoMain, skinColor } = profile;
    const isJakun = garmentStyle === 'JAKUN_OPEN';
    const isBatik = garmentStyle === 'BATIK_LECTURER';
    const sleeveMat = isJakun
      ? HUMAN_MAT.JAKUN_TWILL
      : isBatik
        ? HUMAN_MAT.BATIK_MOTIF
        : HUMAN_MAT.HIJAB_SILK;
    const cuffShade = lerp3(torsoMain, [0.05, 0.05, 0.05], 0.18);
    const skinShade = lerp3(skinColor, [0.18, 0.08, 0.05], 0.12);

    for (const side of [1, -1]) {
      const shX = side * 0.184;
      const shZ = -0.006;
      const jSh = side > 0 ? J_SHOULDER_L : J_SHOULDER_R;
      const jEl = side > 0 ? J_ELBOW_L : J_ELBOW_R;
      const jWr = side > 0 ? J_WRIST_L : J_WRIST_R;

      // Continuous 13-station arm from Deltoid Cap (1.416m) through Elbow (1.128m) to Wrist (0.896m)
      const armStations = [
        { cx: shX - side * 0.016, y: 1.416, cz: shZ, rx: 0.052, rz: 0.054, color: torsoMain, mat: sleeveMat, ao: 0.90 },
        { cx: shX, y: 1.382, cz: shZ, rx: 0.056, rz: 0.056, color: torsoMain, mat: sleeveMat, ao: 0.98 },
        { cx: shX, y: 1.305, cz: shZ, rx: 0.051, rz: 0.052, color: torsoMain, mat: sleeveMat, ao: 1.0 },
        {
          cx: shX,
          y: 1.215,
          cz: shZ,
          rx: 0.047,
          rz: 0.048,
          modFn: (th) => 1.0 + 0.022 * Math.sin(4.0 * th),
          color: torsoMain,
          mat: sleeveMat,
          ao: 0.98
        },
        // Elbow joint zone (y = 1.128m — seamless 50/50 LBS blend!)
        {
          cx: shX,
          y: 1.155,
          cz: shZ,
          rx: 0.045,
          rz: 0.046,
          modFn: (th) => 1.0 + 0.028 * Math.cos(3.0 * th),
          color: torsoMain,
          mat: sleeveMat,
          ao: 0.94
        },
        {
          cx: shX,
          y: 1.128,
          cz: shZ - 0.002,
          rx: 0.044,
          rz: 0.046,
          modFn: (th) => 1.0 + 0.025 * Math.cos(5.0 * th),
          color: torsoMain,
          mat: sleeveMat,
          ao: 0.92
        },
        { cx: shX, y: 1.095, cz: shZ, rx: 0.045, rz: 0.046, color: torsoMain, mat: sleeveMat, ao: 0.96 },
        { cx: shX, y: 1.020, cz: shZ + 0.001, rx: 0.042, rz: 0.043, color: torsoMain, mat: sleeveMat, ao: 1.0 },
        // Rolled / buttoned 3D jacket or shirt sleeve cuff!
        { cx: shX, y: 0.958, cz: shZ + 0.002, rx: 0.043, rz: 0.044, color: cuffShade, mat: sleeveMat, ao: 0.94 },
        { cx: shX, y: 0.940, cz: shZ + 0.002, rx: 0.041, rz: 0.042, color: HUMAN_PALETTE.INNER_WHITE, mat: HUMAN_MAT.FABRIC_LOWER, ao: 0.90 },
        // Exposed anatomical wrist skin emerging cleanly from cuff
        { cx: shX, y: 0.934, cz: shZ + 0.002, rx: 0.030, rz: 0.032, color: skinShade, mat: HUMAN_MAT.SKIN, ao: 0.84 },
        { cx: shX, y: 0.896, cz: shZ + 0.003, rx: 0.027, rz: 0.030, color: skinColor, mat: HUMAN_MAT.SKIN, ao: 0.96 }
      ];

      builder.addWatertightLoft(armStations, 20, {
        uvScaleU: 1.2,
        uvScaleV: 1.8,
        weightFn: (_st, _t, _th, _px, py) => {
          if (py > 1.388) {
            // Smooth armpit/deltoid shoulder-to-chest blend!
            const wSh = 1.0 - 0.45 * smooth01(1.388, 1.418, py);
            return [jSh, J_CHEST, wSh];
          }
          if (py > 1.080) {
            // Smooth upper-arm to forearm elbow LBS transition (1.175 -> 1.080)
            const wEl = 1.0 - smooth01(1.080, 1.175, py);
            return [jEl, jSh, wEl];
          }
          if (py < 0.916) {
            const wWr = 1.0 - smooth01(0.894, 0.916, py);
            return [jWr, jEl, wWr];
          }
          return [jEl, jEl, 1.0];
        }
      });

      // Left-Wrist Metallic Watch (side > 0)
      if (side > 0) {
        builder.addWatertightLoft(
          [
            { cx: shX, y: 0.916, cz: shZ + 0.003, rx: 0.031, rz: 0.034, color: HUMAN_PALETTE.METAL_DARK, mat: HUMAN_MAT.METAL_ACCENT },
            {
              cx: shX,
              y: 0.906,
              cz: shZ + 0.003,
              rx: 0.032,
              rz: 0.035,
              colorFn: (th) => (side * Math.sin(th) > 0.65 ? HUMAN_PALETTE.METAL_SILVER : HUMAN_PALETTE.METAL_DARK),
              mat: HUMAN_MAT.METAL_ACCENT
            },
            { cx: shX, y: 0.899, cz: shZ + 0.003, rx: 0.031, rz: 0.034, color: HUMAN_PALETTE.METAL_DARK, mat: HUMAN_MAT.METAL_ACCENT }
          ],
          14,
          { defaultJoint: jWr, defaultMat: HUMAN_MAT.METAL_ACCENT }
        );
      }

      // === TRUE 5-FINGER SCULPTED ANATOMICAL HAND (Palm + Opposable Thumb + 4 Distinct Fingers!) ===
      // Hand orientation in relaxed standing pose: palm faces inward (-side * X), thumb points forward (+Z)!
      const palmZ = shZ + 0.004;
      const palmStations = [
        { cx: shX, y: 0.898, cz: palmZ, rx: 0.024, rz: 0.029, color: skinColor, ao: 0.95 },
        { cx: shX - side * 0.002, y: 0.874, cz: palmZ + 0.004, rx: 0.019, rz: 0.038, color: skinColor, ao: 0.98 },
        {
          cx: shX - side * 0.003,
          y: 0.844,
          cz: palmZ + 0.006,
          rx: 0.017,
          rz: 0.041,
          modFn: (th) => 1.0 + 0.035 * Math.cos(8.0 * th), // Metacarpal knuckle contour
          color: skinColor,
          ao: 0.96
        },
        { cx: shX - side * 0.004, y: 0.830, cz: palmZ + 0.005, rx: 0.014, rz: 0.039, color: skinShade, ao: 0.90 }
      ];
      builder.addWatertightLoft(palmStations, 16, {
        defaultJoint: jWr,
        defaultMat: HUMAN_MAT.SKIN
      });

      // Opposable 3-Station Curved Thumb (+Z anterior side of palm)
      const thumbCurve = [
        { cx: shX - side * 0.005, cy: 0.874, cz: palmZ + 0.026, r: 0.0115, color: skinColor, ao: 0.95 },
        { cx: shX - side * 0.011, cy: 0.852, cz: palmZ + 0.042, r: 0.0098, color: skinColor, ao: 0.98 },
        { cx: shX - side * 0.014, cy: 0.831, cz: palmZ + 0.049, r: 0.0084, color: skinColor, ao: 1.0 },
        { cx: shX - side * 0.013, cy: 0.816, cz: palmZ + 0.051, r: 0.0068, color: lerp3(skinColor, HUMAN_PALETTE.INNER_WHITE, 0.14), ao: 1.0 }
      ];
      builder.addWatertightLoft(thumbCurve, 9, {
        axis: 'CURVE',
        defaultJoint: jWr,
        defaultMat: HUMAN_MAT.SKIN
      });

      // 4 Distinct Anatomically Proportioned Fingers (Index, Middle, Ring, Little Finger!)
      const fingerSpecs = [
        { zOff: 0.028, len: 0.068, r0: 0.0086, curlIn: 0.014 },  // Index finger
        { zOff: 0.010, len: 0.075, r0: 0.0090, curlIn: 0.017 },  // Middle finger (longest)
        { zOff: -0.008, len: 0.070, r0: 0.0084, curlIn: 0.019 }, // Ring finger
        { zOff: -0.025, len: 0.056, r0: 0.0074, curlIn: 0.021 }  // Little finger (pinky)
      ];

      for (const fg of fingerSpecs) {
        const fBaseX = shX - side * 0.003;
        const fBaseY = 0.833;
        const fBaseZ = palmZ + 0.005 + fg.zOff;
        const tipCol = lerp3(skinColor, HUMAN_PALETTE.INNER_WHITE, 0.12);

        builder.addWatertightLoft(
          [
            { cx: fBaseX, cy: fBaseY, cz: fBaseZ, r: fg.r0, color: skinShade, ao: 0.88 },
            {
              cx: fBaseX - side * (fg.curlIn * 0.30),
              cy: fBaseY - fg.len * 0.36,
              cz: fBaseZ + 0.003,
              r: fg.r0 * 0.91,
              color: skinColor,
              ao: 0.95
            },
            {
              cx: fBaseX - side * (fg.curlIn * 0.72),
              cy: fBaseY - fg.len * 0.71,
              cz: fBaseZ + 0.005,
              r: fg.r0 * 0.81,
              color: skinColor,
              ao: 0.98
            },
            {
              cx: fBaseX - side * fg.curlIn,
              cy: fBaseY - fg.len,
              cz: fBaseZ + 0.006,
              r: fg.r0 * 0.66,
              color: tipCol,
              ao: 1.0
            }
          ],
          8,
          {
            axis: 'CURVE',
            defaultJoint: jWr,
            defaultMat: HUMAN_MAT.SKIN
          }
        );
      }
    }
  }

  /**
   * 3. CONTINUOUS SEAMLESS LBS LOWER BODY (Trousers OR Flowing A-Line Pleated Long Skirt)
   *    AND ARTICULATED METATARSAL TOE-ROLL SNEAKERS / LOAFERS (J_ANKLE -> J_TOE)!
   */
  static _addLowerBodyAndToeRollShoes(builder, profile) {
    const { lowerColor, isLongSkirt, shoeColor, soleColor, buildScaleX = 1.0 } = profile;
    const lowerDark = lerp3(lowerColor, [0.02, 0.02, 0.03], 0.20);

    if (isLongSkirt) {
      // === TRUE CONTINUOUS FLOWING A-LINE PLEATED LONG SKIRT (ZERO SPLIT TROUSER TUBES!) ===
      const skirtStations = [];
      const nSkirtRings = 11;
      for (let i = 0; i < nSkirtRings; i++) {
        const t = i / (nSkirtRings - 1); // 0 at waist (0.875m) -> 1 at ankle-grazing hem (0.145m)
        const py = 0.875 - t * (0.875 - 0.145);
        const rx = (0.140 + Math.pow(t, 1.08) * 0.125) * buildScaleX;
        const rz = 0.102 + Math.pow(t, 1.12) * 0.118;
        const pleatAmp = 0.004 + t * 0.036;

        skirtStations.push({
          y: py,
          cz: -0.004 * t,
          rx,
          rz,
          modFn: (th) => 1.0 + pleatAmp * Math.cos(14.0 * th) + pleatAmp * 0.25 * Math.sin(7.0 * th),
          aoFn: (th) => 0.88 + 0.12 * (0.5 + 0.5 * Math.cos(14.0 * th)),
          color: t > 0.94 ? lowerDark : lowerColor,
          mat: HUMAN_MAT.FABRIC_LOWER
        });
      }

      builder.addWatertightLoft(skirtStations, 36, {
        defaultMat: HUMAN_MAT.FABRIC_LOWER,
        uvScaleU: 2.5,
        uvScaleV: 2.0,
        weightFn: (_st, _t, _th, px, py) => {
          if (py > 0.80) {
            return [J_PELVIS, J_PELVIS, 1.0];
          }
          // Upper-mid skirt blends laterally between Left/Right Hip and dynamic Skirt-Hem bone!
          const sideT = smooth01(-0.06, 0.06, px); // 0 = Right (-X), 1 = Left (+X)
          const legHip = sideT >= 0.5 ? J_HIP_L : J_HIP_R;
          const legLatStrength = Math.abs(sideT - 0.5) * 2.0; // 1 on outer sides, 0 along center seam
          if (py > 0.45) {
            const wHip = (0.35 + 0.55 * legLatStrength) * smooth01(0.45, 0.80, py);
            return [legHip, J_SKIRT_HEM, wHip];
          }
          const legKnee = sideT >= 0.5 ? J_KNEE_L : J_KNEE_R;
          const wKnee = 0.26 * legLatStrength * smooth01(0.14, 0.45, py);
          return [J_SKIRT_HEM, legKnee, 1.0 - wKnee];
        }
      });

      // Modest lower calves & socks beneath long skirt hem (y = 0.22m -> 0.065m)
      for (const side of [1, -1]) {
        const lx = side * 0.090;
        const jKn = side > 0 ? J_KNEE_L : J_KNEE_R;
        const jAn = side > 0 ? J_ANKLE_L : J_ANKLE_R;
        builder.addWatertightLoft(
          [
            { cx: lx, y: 0.225, rx: 0.045, rz: 0.046, color: lowerDark, mat: HUMAN_MAT.FABRIC_LOWER },
            { cx: lx, y: 0.130, rx: 0.039, rz: 0.040, color: HUMAN_PALETTE.INNER_WHITE, mat: HUMAN_MAT.FABRIC_LOWER },
            { cx: lx, y: 0.068, rx: 0.035, rz: 0.037, color: HUMAN_PALETTE.INNER_WHITE, mat: HUMAN_MAT.FABRIC_LOWER }
          ],
          14,
          {
            weightFn: (_st, _t, _th, _px, py) => [jKn, jAn, smooth01(0.068, 0.115, py)]
          }
        );
      }
    } else {
      // === CONTINUOUS SEAMLESS LBS TAILORED TROUSERS / DENIM CHINOS (Hip -> Patella Knee -> Calf -> Cuff!) ===
      for (const side of [1, -1]) {
        const lx = side * 0.090;
        const jHip = side > 0 ? J_HIP_L : J_HIP_R;
        const jKnee = side > 0 ? J_KNEE_L : J_KNEE_R;
        const jAnk = side > 0 ? J_ANKLE_L : J_ANKLE_R;

        const legStations = [
          { cx: lx * 0.92, y: 0.875, rx: 0.080 * buildScaleX, rz: 0.082, color: lowerColor, ao: 0.88 },
          { cx: lx, y: 0.805, rx: 0.076 * buildScaleX, rz: 0.078, color: lowerColor, ao: 0.96 },
          {
            cx: lx,
            y: 0.680,
            rx: 0.069 * buildScaleX,
            rz: 0.072,
            modFn: (th) => 1.0 + 0.018 * Math.cos(2.0 * th), // Pressed trouser front crease!
            color: lowerColor,
            ao: 1.0
          },
          { cx: lx, y: 0.555, rx: 0.061, rz: 0.063, color: lowerColor, ao: 0.99 },
          // Patella & Popliteal Knee LBS Zone (y = 0.460m!)
          {
            cx: lx,
            y: 0.490,
            cz: 0.002,
            rx: 0.057,
            rz: 0.059,
            modFn: (th) => 1.0 + 0.022 * Math.sin(4.0 * th),
            color: lowerColor,
            ao: 0.96
          },
          {
            cx: lx,
            y: 0.460,
            cz: 0.003,
            rx: 0.055,
            rz: 0.058,
            modFn: (th) => 1.0 + 0.025 * Math.cos(3.0 * th),
            color: lowerColor,
            ao: 0.94
          },
          { cx: lx, y: 0.420, cz: -0.002, rx: 0.056, rz: 0.058, color: lowerColor, ao: 0.97 },
          // Gastrocnemius calf fullness
          { cx: lx, y: 0.335, cz: -0.005, rx: 0.058, rz: 0.061, color: lowerColor, ao: 1.0 },
          { cx: lx, y: 0.220, cz: -0.002, rx: 0.049, rz: 0.051, color: lowerColor, ao: 0.99 },
          { cx: lx, y: 0.122, rx: 0.045, rz: 0.047, color: lowerColor, ao: 0.96 },
          // Rolled 3D trouser cuff hem!
          { cx: lx, y: 0.098, rx: 0.047, rz: 0.049, color: lowerDark, ao: 0.90 },
          { cx: lx, y: 0.088, rx: 0.037, rz: 0.039, color: HUMAN_PALETTE.INNER_WHITE, ao: 0.86 },
          { cx: lx, y: 0.066, rx: 0.035, rz: 0.037, color: HUMAN_PALETTE.INNER_WHITE, ao: 0.92 }
        ];

        builder.addWatertightLoft(legStations, 20, {
          defaultMat: HUMAN_MAT.FABRIC_LOWER,
          uvScaleU: 1.4,
          uvScaleV: 2.2,
          weightFn: (_st, _t, _th, _px, py) => {
            if (py > 0.835) {
              const wHip = 1.0 - 0.55 * smooth01(0.835, 0.878, py);
              return [jHip, J_PELVIS, wHip];
            }
            if (py > 0.405) {
              const wKnee = 1.0 - smooth01(0.405, 0.515, py);
              return [jKnee, jHip, wKnee];
            }
            if (py < 0.082) {
              const wAnk = 1.0 - smooth01(0.064, 0.082, py);
              return [jAnk, jKnee, wAnk];
            }
            return [jKnee, jKnee, 1.0];
          }
        });
      }
    }

    // === ARTICULATED HEEL-TO-TOE CAMPUS SNEAKERS / LOAFERS (Longitudinal Z-Loft with J_ANKLE <-> J_TOE Flex!) ===
    // Sole rests EXACTLY at y = 0.000m across heel (z = -0.045) through ball (z = +0.095) to toe (z = +0.166)!
    for (const side of [1, -1]) {
      const lx = side * 0.090;
      const jAnk = side > 0 ? J_ANKLE_L : J_ANKLE_R;
      const jToe = side > 0 ? J_TOE_L : J_TOE_R;

      const makeShoeSection = (z, halfW, heightTop, yArchLift = 0.0, isLaceZone = false, isToeCap = false) => {
        const cy = 0.5 * (yArchLift + heightTop);
        const ry = Math.max(0.010, 0.5 * (heightTop - yArchLift));
        return {
          cx: lx,
          cy,
          cz: z,
          rx: halfW,
          ry,
          // Flatten outsole bottom strictly to y >= yArchLift (0.000m at heel & forefoot!)
          offsetFn: (th) => {
            const c = Math.cos(th); // +1 top (+Y), -1 bottom (-Y)
            if (c < -0.35) {
              const rawY = cy + c * ry;
              const clampedY = Math.max(yArchLift, rawY * 0.35 + yArchLift * 0.65);
              return [0.0, clampedY - rawY, 0.0];
            }
            return [0.0, 0.0, 0.0];
          },
          modFn: (th) => {
            if (isLaceZone && Math.cos(th) > 0.55) {
              return 1.045; // Raised tongue & criss-cross shoelace ridge
            }
            return 1.0;
          },
          colorFn: (th) => {
            const c = Math.cos(th);
            // Rubber midsole & outsole band along lower rim!
            if (c < -0.22) return soleColor;
            if (isLaceZone && c > 0.68) return HUMAN_PALETTE.SHOE_WHITE_TRIM; // White laces & tongue!
            if (isToeCap && c > -0.22 && c < 0.25) return soleColor; // Classic sneaker toe bumper
            return shoeColor;
          },
          aoFn: (th) => (Math.cos(th) < -0.55 ? 0.72 : 0.98),
          mat: HUMAN_MAT.SHOE_BAG
        };
      };

      const shoeStations = [
        makeShoeSection(-0.046, 0.033, 0.066, 0.002),
        makeShoeSection(-0.025, 0.041, 0.074, 0.000), // Heel strike pad (y = 0.000m!)
        makeShoeSection(0.005, 0.043, 0.076, 0.000),  // Ankle collar
        makeShoeSection(0.038, 0.041, 0.064, 0.004, true), // Midfoot arch & laces
        makeShoeSection(0.068, 0.045, 0.054, 0.001, true), // Metatarsal flex start
        makeShoeSection(0.098, 0.048, 0.045, 0.000, false), // Ball of foot (J_TOE pivot z = 0.095, y = 0.000m!)
        makeShoeSection(0.132, 0.045, 0.038, 0.001, false, true),
        makeShoeSection(0.156, 0.036, 0.031, 0.004, false, true),
        makeShoeSection(0.170, 0.022, 0.023, 0.008, false, true) // Slightly upturned ergonomic toe spring!
      ];

      builder.addWatertightLoft(shoeStations, 18, {
        axis: 'Z',
        defaultMat: HUMAN_MAT.SHOE_BAG,
        uvScaleU: 1.0,
        uvScaleV: 1.0,
        weightFn: (_st, _t, _th, _px, _py, pz) => {
          const wToe = smooth01(0.062, 0.112, pz);
          return [jToe, jAnk, wToe];
        }
      });
    }
  }

  /**
   * 4. HYPER-SCULPTED ANATOMICAL NECK, CRANIUM, JAWLINE, LIPS, CHEEKBONES, RECESSED EYE SOCKETS,
   *    REAL 3D SPHERICAL EYEBALLS (Sclera + Iris + Pupil + Wet Cornea), 3D EYELIDS & ARCHED BROWS,
   *    MULTI-STATION NOSE WITH NOSTRILS, 3D EARS, EYEGLASSES, AND 4 HAIRSTYLES / LBS HIJAB SHAWL!
   */
  static _addSculptedHeadFaceEyesAndHair(builder, profile) {
    const {
      skinColor,
      hairStyle, // 'HIJAB' | 'HAIR_PARTED' | 'HAIR_WAVY' | 'HAIR_PONYTAIL'
      headwearColor,
      hasGlasses,
      isFemale
    } = profile;

    const isHijab = hairStyle === 'HIJAB';
    const skinShadow = lerp3(skinColor, [0.22, 0.09, 0.06], 0.16);
    const cheekWarm = lerp3(skinColor, [0.85, 0.32, 0.26], isFemale ? 0.11 : 0.06);
    const lipUpper = lerp3(skinColor, [0.62, 0.21, 0.20], isFemale ? 0.42 : 0.26);
    const lipLower = lerp3(skinColor, [0.72, 0.27, 0.25], isFemale ? 0.38 : 0.22);

    // 4A. Seamless LBS Anatomical Neck (y = 1.420 -> 1.530, blended J_CHEST -> J_NECK -> J_HEAD!)
    builder.addWatertightLoft(
      [
        { y: 1.420, cz: 0.004, rx: 0.055, rz: 0.054, color: skinShadow, ao: 0.82 },
        { y: 1.455, cz: 0.006, rx: 0.048, rz: 0.049, color: skinColor, ao: 0.90 },
        { y: 1.492, cz: 0.008, rx: 0.046, rz: 0.048, color: skinShadow, ao: 0.85 },
        { y: 1.526, cz: 0.008, rx: 0.048, rz: 0.052, color: skinShadow, ao: 0.80 }
      ],
      18,
      {
        defaultMat: HUMAN_MAT.SKIN,
        weightFn: (_st, _t, _th, _px, py) => {
          if (py < 1.445) return [J_NECK, J_CHEST, smooth01(1.420, 1.445, py)];
          return [J_HEAD, J_NECK, smooth01(1.488, 1.524, py)];
        }
      }
    );

    // 4B. 17-Station × 32-Seg Sculpted Cranium & True 3D Facial Contour (J_HEAD pivot at y = 1.513, z = 0.008)
    const jawScale = isFemale ? 0.94 : 1.0;
    const faceHelper = (th) => {
      const s = th > Math.PI ? th - Math.PI * 2.0 : th; // [-PI, PI], 0 is +Z front center!
      return { s, a: Math.abs(s) };
    };

    const craniumStations = [
      // Under-chin & Mentum (chin prominence at +Z)
      { y: 1.514, cz: 0.022, rx: 0.032 * jawScale, rz: 0.042, color: skinShadow, ao: 0.84 },
      {
        y: 1.528,
        cz: 0.022,
        rx: 0.048 * jawScale,
        rz: 0.062,
        modFn: (th) => {
          const { a } = faceHelper(th);
          return a < 0.32 ? 1.055 : 1.0; // Defined 3D chin!
        },
        color: skinColor,
        ao: 0.94
      },
      // Lower Lip Vermilion Relief (y = 1.547)
      {
        y: 1.547,
        cz: 0.018,
        rx: 0.060 * jawScale,
        rz: 0.074,
        modFn: (th) => {
          const { a } = faceHelper(th);
          return a < 0.24 ? 1.048 * Math.cos((a / 0.24) * 1.1) + 0.01 : 1.0;
        },
        colorFn: (th) => (faceHelper(th).a < 0.23 ? lipLower : skinColor),
        ao: 0.97
      },
      // Mouth Parting Seam Crease (y = 1.555)
      {
        y: 1.555,
        cz: 0.017,
        rx: 0.064 * jawScale,
        rz: 0.076,
        modFn: (th) => (faceHelper(th).a < 0.26 ? 0.985 : 1.0),
        colorFn: (th) => (faceHelper(th).a < 0.25 ? lerp3(lipUpper, [0.1, 0.03, 0.03], 0.35) : skinColor),
        aoFn: (th) => (faceHelper(th).a < 0.26 ? 0.78 : 0.98)
      },
      // Upper Lip Cupid's Bow Relief (y = 1.563)
      {
        y: 1.563,
        cz: 0.017,
        rx: 0.067 * jawScale,
        rz: 0.078,
        modFn: (th) => {
          const { a } = faceHelper(th);
          return a < 0.24 ? 1.046 : 1.0;
        },
        colorFn: (th) => (faceHelper(th).a < 0.23 ? lipUpper : skinColor),
        ao: 0.98
      },
      // Sub-nasal Philtrum & Lower Cheek (y = 1.577)
      { y: 1.577, cz: 0.015, rx: 0.073 * jawScale, rz: 0.083, color: skinColor, ao: 0.96 },
      // Malar Cheekbones & Zygomatic Contour (y = 1.598)
      {
        y: 1.598,
        cz: 0.012,
        rx: 0.079,
        rz: 0.088,
        modFn: (th) => {
          const { a } = faceHelper(th);
          return a > 0.32 && a < 0.85 ? 1.036 : 1.0; // Natural cheekbone fullness!
        },
        colorFn: (th) => {
          const { a } = faceHelper(th);
          return a > 0.28 && a < 0.75 ? cheekWarm : skinColor;
        },
        ao: 1.0
      },
      // Infraorbital Rim (below eyes, y = 1.616)
      { y: 1.616, cz: 0.010, rx: 0.081, rz: 0.089, color: skinColor, ao: 0.97 },
      // RECESSED ORBITAL EYE SOCKETS (y = 1.631 — recessed so real 3D eyeballs sit cleanly inside!)
      {
        y: 1.631,
        cz: 0.009,
        rx: 0.081,
        rz: 0.089,
        modFn: (th) => {
          const { a } = faceHelper(th);
          // Concave orbital socket hollow at th ~ +-0.37 rad (x ~ +-0.031m)
          const dEye = Math.abs(a - 0.37);
          if (dEye < 0.20) {
            return 1.0 - 0.085 * Math.cos((dEye / 0.20) * Math.PI * 0.5);
          }
          return 1.0;
        },
        aoFn: (th) => (Math.abs(faceHelper(th).a - 0.37) < 0.20 ? 0.76 : 0.98),
        color: skinColor
      },
      // Supraorbital Brow Ridge (y = 1.648 — overhangs eye sockets for real facial shadow!)
      {
        y: 1.648,
        cz: 0.010,
        rx: 0.082,
        rz: 0.092,
        modFn: (th) => {
          const { a } = faceHelper(th);
          return a < 0.66 ? 1.034 : 1.0;
        },
        color: skinColor,
        ao: 1.0
      },
      // Forehead & Temples (y = 1.672 -> 1.734)
      { y: 1.672, cz: 0.007, rx: 0.081, rz: 0.092, color: skinColor, ao: 1.0 },
      { y: 1.696, cz: 0.003, rx: 0.077, rz: 0.089, color: skinColor, ao: 1.0 },
      { y: 1.718, cz: -0.002, rx: 0.066, rz: 0.078, color: skinColor, ao: 1.0 },
      { y: 1.733, cz: -0.006, rx: 0.044, rz: 0.054, color: skinColor, ao: 1.0 }
    ];

    builder.addWatertightLoft(craniumStations, 32, {
      defaultJoint: J_HEAD,
      defaultMat: HUMAN_MAT.SKIN,
      uvScaleU: 1.0,
      uvScaleV: 1.0
    });

    // 4C. Sculpted Multi-Station 3D Nose (Bridge, Dorsum, Tip, Columella, & Flared Nostril Wings!)
    const noseTipCol = lerp3(cheekWarm, HUMAN_PALETTE.INNER_WHITE, 0.05);
    builder.addWatertightLoft(
      [
        { y: 1.580, cz: 0.094, rx: 0.016, rz: 0.012, modFn: (th) => 1.0 + 0.14 * Math.abs(Math.sin(th)), color: skinShadow, ao: 0.76 }, // Flared nostrils + nostril shadow!
        { y: 1.590, cz: 0.105, rx: 0.015, rz: 0.018, color: noseTipCol, ao: 0.98 },
        { y: 1.601, cz: 0.111, rx: 0.010, rz: 0.017, color: noseTipCol, ao: 1.0 }, // Pronasale nose tip!
        { y: 1.618, cz: 0.103, rx: 0.0085, rz: 0.014, color: skinColor, ao: 1.0 },
        { y: 1.635, cz: 0.096, rx: 0.0080, rz: 0.011, color: skinColor, ao: 0.96 }, // Nasion bridge
        { y: 1.647, cz: 0.094, rx: 0.0095, rz: 0.009, color: skinColor, ao: 0.96 }
      ],
      14,
      { defaultJoint: J_HEAD, defaultMat: HUMAN_MAT.SKIN }
    );

    // 4D. PAIR OF REAL 3D SPHERICAL EYEBALLS + 3D UPPER/LOWER EYELIDS + SCULPTED ARCHED EYEBROWS!
    for (const sx of [1, -1]) {
      const eyeX = sx * 0.0305;
      const eyeY = 1.631;
      const eyeZ = 0.0835;

      // Real 3D Eyeball Sphere oriented along +Z (z = front cornea/pupil -> back sclera!)
      builder.addWatertightLoft(
        [
          { cx: eyeX, cy: eyeY, cz: eyeZ - 0.006, r: 0.0075, color: HUMAN_PALETTE.SCLERA_WHITE, mat: HUMAN_MAT.EYE_CORNEA },
          { cx: eyeX, cy: eyeY, cz: eyeZ + 0.001, r: 0.0115, color: HUMAN_PALETTE.SCLERA_WHITE, mat: HUMAN_MAT.EYE_CORNEA },
          { cx: eyeX, cy: eyeY, cz: eyeZ + 0.0072, r: 0.0092, color: HUMAN_PALETTE.SCLERA_WHITE, mat: HUMAN_MAT.EYE_CORNEA },
          // Dark-amber Indonesian iris ring!
          { cx: eyeX, cy: eyeY, cz: eyeZ + 0.0104, r: 0.0058, color: HUMAN_PALETTE.IRIS_BROWN, mat: HUMAN_MAT.EYE_CORNEA },
          // Jet-black pupil + corneal dome bulge!
          { cx: eyeX, cy: eyeY, cz: eyeZ + 0.0118, r: 0.0029, color: HUMAN_PALETTE.PUPIL_BLACK, mat: HUMAN_MAT.EYE_CORNEA, poleDome: 0.0008 }
        ],
        14,
        { axis: 'Z', defaultJoint: J_HEAD, defaultMat: HUMAN_MAT.EYE_CORNEA }
      );

      // 3D Curved Upper Eyelid & Lash Line (frames top of eyeball for natural human expression!)
      builder.addWatertightLoft(
        [
          { cx: eyeX - sx * 0.012, cy: eyeY - 0.001, cz: eyeZ + 0.006, r: 0.0024, color: skinShadow },
          { cx: eyeX - sx * 0.005, cy: eyeY + 0.0058, cz: eyeZ + 0.0115, r: 0.0032, color: HUMAN_PALETTE.BROW_LASH },
          { cx: eyeX + sx * 0.005, cy: eyeY + 0.0056, cz: eyeZ + 0.0112, r: 0.0032, color: HUMAN_PALETTE.BROW_LASH },
          { cx: eyeX + sx * 0.013, cy: eyeY - 0.0005, cz: eyeZ + 0.005, r: 0.0024, color: skinShadow }
        ],
        8,
        { axis: 'CURVE', defaultJoint: J_HEAD, defaultMat: HUMAN_MAT.SKIN }
      );

      // 3D Sculpted Arched Eyebrow
      const browThick = isFemale ? 0.0028 : 0.0038;
      builder.addWatertightLoft(
        [
          { cx: eyeX - sx * 0.014, cy: eyeY + 0.015, cz: eyeZ + 0.011, r: browThick * 0.85, color: HUMAN_PALETTE.BROW_LASH },
          { cx: eyeX - sx * 0.004, cy: eyeY + 0.0185, cz: eyeZ + 0.013, r: browThick, color: HUMAN_PALETTE.BROW_LASH },
          { cx: eyeX + sx * 0.007, cy: eyeY + 0.0180, cz: eyeZ + 0.011, r: browThick * 0.92, color: HUMAN_PALETTE.BROW_LASH },
          { cx: eyeX + sx * 0.016, cy: eyeY + 0.0135, cz: eyeZ + 0.006, r: browThick * 0.62, color: HUMAN_PALETTE.BROW_LASH }
        ],
        7,
        { axis: 'CURVE', defaultJoint: J_HEAD, defaultMat: HUMAN_MAT.HAIR }
      );
    }

    // 4E. Optional 3D Sculpted Eyeglasses (`hasGlasses` on Lecturers & Select Scholars!)
    if (hasGlasses) {
      const frameCol = HUMAN_PALETTE.METAL_DARK;
      for (const sx of [1, -1]) {
        const gx = sx * 0.0315;
        const gy = 1.631;
        const gz = 0.101;
        const rimPts = [];
        const rimN = 12;
        for (let i = 0; i <= rimN; i++) {
          const a = (i / rimN) * Math.PI * 2.0;
          rimPts.push({
            cx: gx + Math.cos(a) * 0.0185,
            cy: gy + Math.sin(a) * 0.0145,
            cz: gz - Math.abs(Math.cos(a)) * 0.002,
            r: 0.0018,
            color: frameCol
          });
        }
        builder.addWatertightLoft(rimPts, 6, {
          axis: 'CURVE',
          capStart: false,
          capEnd: false,
          defaultJoint: J_HEAD,
          defaultMat: HUMAN_MAT.METAL_ACCENT
        });

        // Side temple arm back to ear
        builder.addWatertightLoft(
          [
            { cx: sx * 0.050, cy: gy + 0.003, cz: gz - 0.003, r: 0.0017, color: frameCol },
            { cx: sx * 0.079, cy: gy + 0.002, cz: 0.042, r: 0.0017, color: frameCol },
            { cx: sx * 0.084, cy: gy - 0.003, cz: 0.002, r: 0.0016, color: frameCol }
          ],
          6,
          { axis: 'CURVE', defaultJoint: J_HEAD, defaultMat: HUMAN_MAT.METAL_ACCENT }
        );
      }
      // Nose bridge bar
      builder.addWatertightLoft(
        [
          { cx: -0.013, cy: 1.634, cz: 0.102, r: 0.0018, color: frameCol },
          { cx: 0.0, cy: 1.636, cz: 0.105, r: 0.0018, color: frameCol },
          { cx: 0.013, cy: 1.634, cz: 0.102, r: 0.0018, color: frameCol }
        ],
        6,
        { axis: 'CURVE', defaultJoint: J_HEAD, defaultMat: HUMAN_MAT.METAL_ACCENT }
      );
    }

    // 4F. Sculpted 3D Left & Right Ears (when uncovered by Hijab)
    if (!isHijab) {
      for (const sx of [1, -1]) {
        builder.addWatertightLoft(
          [
            { cx: sx * 0.080, y: 1.592, cz: 0.001, rx: 0.006, rz: 0.010, color: skinShadow, ao: 0.86 },
            { cx: sx * 0.086, y: 1.610, cz: -0.002, rx: 0.010, rz: 0.016, color: cheekWarm, ao: 0.94 },
            {
              cx: sx * 0.088,
              y: 1.628,
              cz: -0.004,
              rx: 0.011,
              rz: 0.018,
              modFn: (th) => 1.0 + 0.10 * Math.cos(2.0 * th), // Outer helix rim & concha hollow
              color: skinColor,
              ao: 0.96
            },
            { cx: sx * 0.083, y: 1.644, cz: -0.001, rx: 0.007, rz: 0.013, color: cheekWarm, ao: 0.92 }
          ],
          12,
          { defaultJoint: J_HEAD, defaultMat: HUMAN_MAT.SKIN }
        );
      }
    }

    // 4G. HEADWEAR: DOUBLE-WALLED LBS HIJAB SHAWL (J_HEAD -> J_NECK -> J_CHEST!) OR 3D LAYERED HAIRSTYLES!
    if (isHijab) {
      // Inner forehead 'ciput' underscarf band peeking subtly above brow
      builder.addThickOpenGarmentShell(
        [
          { y: 1.682, cz: 0.004, rx: 0.082, rz: 0.094, thStart: -0.58, thEnd: 0.58, color: HUMAN_PALETTE.CIPUT_DARK },
          { y: 1.704, cz: 0.002, rx: 0.079, rz: 0.091, thStart: -0.62, thEnd: 0.62, color: HUMAN_PALETTE.CIPUT_DARK }
        ],
        12,
        {
          thickness: 0.004,
          outerMat: HUMAN_MAT.HIJAB_SILK,
          innerMat: HUMAN_MAT.HIJAB_SILK,
          defaultJoint: J_HEAD
        }
      );

      // Double-walled Voal Hijab Hood & Chest-Draped Shawl with clean +Z oval face opening!
      // Lower shawl drapes onto shoulders/chest and is LBS-weighted to J_CHEST so head turns NEVER tear or clip!
      const hijabDark = lerp3(headwearColor, [0.04, 0.03, 0.04], 0.16);
      const hijabShell = [
        // Chest & shoulder shawl drape (360-deg closed wrap via overlapping wrap arc!)
        {
          y: 1.335,
          cz: 0.010,
          rx: 0.192,
          rz: 0.136,
          thStart: 0.04,
          thEnd: Math.PI * 2.0 - 0.04,
          modFn: (th) => 1.0 + 0.045 * Math.sin(7.0 * th + 0.4),
          color: hijabDark,
          ao: 0.92
        },
        {
          y: 1.395,
          cz: 0.005,
          rx: 0.172,
          rz: 0.124,
          thStart: 0.05,
          thEnd: Math.PI * 2.0 - 0.05,
          modFn: (th) => 1.0 + 0.038 * Math.sin(7.0 * th + 0.2),
          color: headwearColor,
          ao: 0.96
        },
        {
          y: 1.460,
          cz: 0.006,
          rx: 0.118,
          rz: 0.112,
          thStart: 0.08,
          thEnd: Math.PI * 2.0 - 0.08,
          modFn: (th) => 1.0 + 0.025 * Math.sin(8.0 * th),
          color: headwearColor,
          ao: 0.95
        },
        // Under-chin border (opens cleanly around +Z face above y = 1.522!)
        {
          y: 1.515,
          cz: 0.006,
          rx: 0.098,
          rz: 0.104,
          thStart: 0.22,
          thEnd: Math.PI * 2.0 - 0.22,
          modFn: (th) => 1.0 + 0.016 * Math.sin(9.0 * th),
          color: headwearColor,
          ao: 0.96
        },
        {
          y: 1.568,
          cz: 0.004,
          rx: 0.094,
          rz: 0.103,
          thStart: 0.66,
          thEnd: Math.PI * 2.0 - 0.66,
          modFn: (th) => 1.0 + 0.012 * Math.sin(9.0 * th),
          color: headwearColor,
          ao: 1.0
        },
        {
          y: 1.628,
          cz: 0.003,
          rx: 0.094,
          rz: 0.104,
          thStart: 0.70,
          thEnd: Math.PI * 2.0 - 0.70,
          color: headwearColor,
          ao: 1.0
        },
        {
          y: 1.686,
          cz: 0.003,
          rx: 0.091,
          rz: 0.103,
          thStart: 0.56,
          thEnd: Math.PI * 2.0 - 0.56,
          color: headwearColor,
          ao: 1.0
        },
        {
          y: 1.724,
          cz: -0.001,
          rx: 0.076,
          rz: 0.090,
          thStart: 0.18,
          thEnd: Math.PI * 2.0 - 0.18,
          color: headwearColor,
          ao: 1.0
        },
        {
          y: 1.746,
          cz: -0.006,
          rx: 0.042,
          rz: 0.052,
          thStart: 0.06,
          thEnd: Math.PI * 2.0 - 0.06,
          color: headwearColor,
          ao: 1.0
        }
      ];

      builder.addThickOpenGarmentShell(hijabShell, 26, {
        thickness: 0.0065,
        outerMat: HUMAN_MAT.HIJAB_SILK,
        innerMat: HUMAN_MAT.HIJAB_SILK,
        innerColorScale: 0.78,
        uvScaleU: 2.0,
        uvScaleV: 2.0,
        weightFn: (_st, _t, _th, _px, py) => {
          if (py < 1.445) {
            return [J_NECK, J_CHEST, smooth01(1.385, 1.455, py) * 0.45];
          }
          if (py < 1.535) {
            return [J_HEAD, J_NECK, smooth01(1.455, 1.535, py)];
          }
          return [J_HEAD, J_HEAD, 1.0];
        }
      });

      // Hijab top crown dome cap + front chest shawl overlap panel (so 0 gap on +Z chest!)
      builder.addWatertightLoft(
        [
          { y: 1.718, cz: -0.002, rx: 0.074, rz: 0.088, color: headwearColor },
          { y: 1.748, cz: -0.006, rx: 0.044, rz: 0.054, color: headwearColor }
        ],
        18,
        { defaultJoint: J_HEAD, defaultMat: HUMAN_MAT.HIJAB_SILK }
      );

      // Front asymmetrical draped shawl bib covering front neck & upper chest cleanly
      builder.addThickOpenGarmentShell(
        [
          { y: 1.320, cz: 0.016, rx: 0.168, rz: 0.118, thStart: -0.52, thEnd: 0.62, modFn: (th) => 1.0 + 0.03 * Math.cos(6.0 * th), color: headwearColor },
          { y: 1.415, cz: 0.012, rx: 0.132, rz: 0.104, thStart: -0.48, thEnd: 0.54, color: headwearColor },
          { y: 1.498, cz: 0.010, rx: 0.092, rz: 0.096, thStart: -0.42, thEnd: 0.42, color: hijabDark }
        ],
        14,
        {
          thickness: 0.006,
          outerMat: HUMAN_MAT.HIJAB_SILK,
          innerMat: HUMAN_MAT.HIJAB_SILK,
          weightFn: (_st, _t, _th, _px, py) => {
            if (py < 1.435) return [J_CHEST, J_NECK, 1.0 - smooth01(1.37, 1.44, py) * 0.35];
            return [J_HEAD, J_NECK, smooth01(1.46, 1.51, py) * 0.65];
          }
        }
      );
    } else {
      // 3D Sculpted Hairstyles ('HAIR_PARTED', 'HAIR_WAVY', 'HAIR_PONYTAIL') leaving +Z face & ears clean!
      const isWavy = hairStyle === 'HAIR_WAVY';
      const isPonytail = hairStyle === 'HAIR_PONYTAIL';
      const lockFreq = isWavy ? 13.0 : 9.0;
      const lockAmp = isWavy ? 0.034 : 0.020;

      // Back & Side Hair Shell (open on +Z face below forehead hairline!)
      const hairShell = [
        {
          y: isPonytail ? 1.545 : 1.568,
          cz: -0.022,
          rx: isPonytail ? 0.078 : 0.070,
          rz: 0.074,
          thStart: 1.05,
          thEnd: Math.PI * 2.0 - 1.05,
          modFn: (th) => 1.0 + lockAmp * Math.sin(lockFreq * th),
          color: headwearColor
        },
        {
          y: 1.615,
          cz: -0.014,
          rx: 0.084,
          rz: 0.086,
          thStart: 0.88,
          thEnd: Math.PI * 2.0 - 0.88,
          modFn: (th) => 1.0 + lockAmp * Math.sin(lockFreq * th + 0.5),
          color: headwearColor
        },
        {
          y: 1.668,
          cz: -0.004,
          rx: 0.087,
          rz: 0.094,
          thStart: 0.64,
          thEnd: Math.PI * 2.0 - 0.64,
          modFn: (th) => 1.0 + lockAmp * Math.cos(lockFreq * th),
          color: headwearColor
        },
        // Forehead hairline & side-parted / wavy fringe!
        {
          y: 1.702,
          cz: 0.003,
          rx: 0.085,
          rz: 0.098,
          thStart: 0.14,
          thEnd: Math.PI * 2.0 - 0.24, // Natural side-parted asymmetry!
          modFn: (th) => 1.0 + lockAmp * 1.15 * Math.sin(lockFreq * th),
          color: headwearColor
        }
      ];

      builder.addThickOpenGarmentShell(hairShell, 22, {
        thickness: 0.008,
        outerMat: HUMAN_MAT.HAIR,
        innerMat: HUMAN_MAT.HAIR,
        defaultJoint: J_HEAD
      });

      // Full 360-deg Textured Upper Hair Crown Volume
      builder.addWatertightLoft(
        [
          {
            y: 1.694,
            cz: 0.001,
            rx: 0.084,
            rz: 0.096,
            modFn: (th) => 1.0 + lockAmp * Math.sin(lockFreq * th),
            color: headwearColor
          },
          {
            y: 1.724,
            cz: -0.002,
            rx: 0.074,
            rz: 0.088,
            modFn: (th) => 1.0 + lockAmp * 0.9 * Math.cos(lockFreq * th),
            color: headwearColor
          },
          {
            y: 1.746,
            cz: -0.005,
            rx: 0.048,
            rz: 0.060,
            modFn: (th) => 1.0 + lockAmp * 0.6 * Math.sin(lockFreq * th),
            color: headwearColor,
            poleDome: 0.008
          }
        ],
        24,
        { defaultJoint: J_HEAD, defaultMat: HUMAN_MAT.HAIR }
      );

      // Uncovered Female 3D Articulated Curved Ponytail + Scrunchie Band!
      if (isPonytail) {
        builder.addWatertightLoft(
          [
            { cx: 0.0, cy: 1.618, cz: -0.090, r: 0.020, color: HUMAN_PALETTE.MAKARA_GOLD, mat: HUMAN_MAT.HIJAB_SILK }, // Hair tie scrunchie
            { cx: 0.0, cy: 1.606, cz: -0.106, r: 0.026, color: headwearColor },
            { cx: 0.0, cy: 1.568, cz: -0.122, r: 0.030, modFn: (th) => 1.0 + 0.05 * Math.sin(7.0 * th), color: headwearColor },
            { cx: 0.0, cy: 1.515, cz: -0.124, r: 0.024, modFn: (th) => 1.0 + 0.05 * Math.sin(7.0 * th), color: headwearColor },
            { cx: 0.0, cy: 1.468, cz: -0.114, r: 0.012, color: headwearColor }
          ],
          14,
          {
            axis: 'CURVE',
            defaultMat: HUMAN_MAT.HAIR,
            weightFn: (_st, t) => [J_HEAD, J_CHEST, 1.0 - t * 0.45] // Tip trails subtly with torso!
          }
        );
      }
    }
  }

  /**
   * 5. MAIN ACADEMIC PROP (Prop-Local Origin -> J_PROP_MAIN = 20):
   *    - 'BACKPACK': Sculpted Campus Daypack + Zippered Pocket + Side Bottle + Twin Padded Straps
   *    - 'OPEN_BOOK': Two-Page Spread Hardcover Textbook (for seated bench scholars & seminar speaker!)
   *    - 'BOOK': Closed Hardcover Textbook / Academic Binder
   *    - 'PHONE': Live-Viewfinder Smartphone w/ Triple-Camera Island & Glowing Screen (HUMAN_MAT.PHONE_GLOW)
   */
  static _addMainAcademicProp(builder, profile) {
    const { propType, bagColor = HUMAN_PALETTE.BAG_CHARCOAL, bookColor = HUMAN_PALETTE.BOOK_CRIMSON } = profile;

    if (propType === 'PHONE') {
      builder.addWatertightLoft(
        [
          { y: -0.074, rx: 0.036, rz: 0.0055, color: HUMAN_PALETTE.PHONE_BODY, mat: HUMAN_MAT.METAL_ACCENT },
          {
            y: -0.058,
            rx: 0.038,
            rz: 0.006,
            colorFn: (th) => (Math.cos(th) < -0.25 ? HUMAN_PALETTE.PHONE_SCREEN : HUMAN_PALETTE.PHONE_BODY),
            matFn: (th) => (Math.cos(th) < -0.25 ? HUMAN_MAT.PHONE_GLOW : HUMAN_MAT.METAL_ACCENT)
          },
          {
            y: 0.058,
            rx: 0.038,
            rz: 0.006,
            colorFn: (th) => (Math.cos(th) < -0.25 ? HUMAN_PALETTE.PHONE_SCREEN : HUMAN_PALETTE.PHONE_BODY),
            matFn: (th) => (Math.cos(th) < -0.25 ? HUMAN_MAT.PHONE_GLOW : HUMAN_MAT.METAL_ACCENT)
          },
          { y: 0.074, rx: 0.036, rz: 0.0055, color: HUMAN_PALETTE.PHONE_BODY, mat: HUMAN_MAT.METAL_ACCENT }
        ],
        18,
        { defaultJoint: J_PROP_MAIN, uvScaleU: 1.0, uvScaleV: 1.0 }
      );
      // Rear Triple-Camera Bump (+Z)
      builder.addWatertightLoft(
        [
          { cx: 0.015, y: 0.030, cz: 0.007, rx: 0.012, rz: 0.004, color: HUMAN_PALETTE.BROW_LASH },
          { cx: 0.015, y: 0.058, cz: 0.007, rx: 0.012, rz: 0.004, color: HUMAN_PALETTE.BROW_LASH }
        ],
        10,
        { defaultJoint: J_PROP_MAIN, defaultMat: HUMAN_MAT.METAL_ACCENT }
      );
    } else if (propType === 'OPEN_BOOK') {
      // Spread Open Hardcover Textbook with Left & Right Arched White Pages!
      for (const sx of [-1, 1]) {
        builder.addWatertightLoft(
          [
            { cx: sx * 0.006, cy: 0.0, cz: -0.095, rx: 0.007, ry: 0.008, color: bookColor },
            { cx: sx * 0.068, cy: 0.016, cz: -0.095, rx: 0.062, ry: 0.009, color: HUMAN_PALETTE.INNER_WHITE },
            { cx: sx * 0.068, cy: 0.016, cz: 0.095, rx: 0.062, ry: 0.009, color: HUMAN_PALETTE.INNER_WHITE },
            { cx: sx * 0.006, cy: 0.0, cz: 0.095, rx: 0.007, ry: 0.008, color: bookColor }
          ],
          12,
          { axis: 'Z', defaultJoint: J_PROP_MAIN, defaultMat: HUMAN_MAT.SHOE_BAG }
        );
      }
    } else if (propType === 'BOOK') {
      builder.addWatertightLoft(
        [
          { y: -0.115, rx: 0.023, rz: 0.088, color: bookColor },
          { y: -0.025, rx: 0.024, rz: 0.091, color: bookColor },
          { y: 0.015, rx: 0.024, rz: 0.091, color: HUMAN_PALETTE.MAKARA_GOLD, mat: HUMAN_MAT.CREST_GOLD },
          { y: 0.070, rx: 0.024, rz: 0.091, color: bookColor },
          {
            y: 0.114,
            rx: 0.022,
            rz: 0.087,
            colorFn: (th) => (Math.cos(th) > -0.35 ? HUMAN_PALETTE.INNER_WHITE : bookColor)
          }
        ],
        16,
        { defaultJoint: J_PROP_MAIN, defaultMat: HUMAN_MAT.SHOE_BAG }
      );
    } else {
      // Sculpted Ergonomic Campus Daypack + Front Pocket + Water Bottle + Twin 3D Padded Shoulder Straps
      const bagDark = lerp3(bagColor, [0.02, 0.02, 0.03], 0.25);
      builder.addWatertightLoft(
        [
          { y: -0.155, rx: 0.116, rz: 0.066, color: bagDark },
          { y: -0.075, rx: 0.128, rz: 0.078, color: bagColor },
          { y: 0.020, rx: 0.126, rz: 0.078, color: bagColor },
          { y: 0.105, rx: 0.110, rz: 0.066, color: bagColor },
          { y: 0.156, rx: 0.082, rz: 0.046, color: bagDark }
        ],
        18,
        { defaultJoint: J_PROP_MAIN, defaultMat: HUMAN_MAT.SHOE_BAG }
      );
      builder.addWatertightLoft(
        [
          { cz: -0.056, y: -0.135, rx: 0.092, rz: 0.036, color: bagDark },
          { cz: -0.058, y: -0.055, rx: 0.096, rz: 0.042, color: bagDark },
          { cz: -0.056, y: 0.002, rx: 0.090, rz: 0.036, color: HUMAN_PALETTE.METAL_SILVER, mat: HUMAN_MAT.METAL_ACCENT },
          { cz: -0.050, y: 0.016, rx: 0.080, rz: 0.028, color: bagDark }
        ],
        14,
        { defaultJoint: J_PROP_MAIN, defaultMat: HUMAN_MAT.SHOE_BAG }
      );
      // Side Water Tumbler in Left Mesh Pocket!
      builder.addWatertightLoft(
        [
          { cx: 0.128, y: -0.130, cz: -0.010, rx: 0.026, rz: 0.026, color: bagDark },
          { cx: 0.130, y: -0.045, cz: -0.010, rx: 0.025, rz: 0.025, color: HUMAN_PALETTE.LANYARD_BLUE, mat: HUMAN_MAT.METAL_ACCENT },
          { cx: 0.130, y: 0.005, cz: -0.010, rx: 0.022, rz: 0.022, color: HUMAN_PALETTE.METAL_DARK, mat: HUMAN_MAT.METAL_ACCENT }
        ],
        10,
        { defaultJoint: J_PROP_MAIN, defaultMat: HUMAN_MAT.SHOE_BAG }
      );
      // Twin Padded 3D Shoulder Straps
      for (const sx of [-0.090, 0.090]) {
        builder.addWatertightLoft(
          [
            { cx: sx, cy: -0.105, cz: 0.068, r: 0.019, color: bagDark },
            { cx: sx * 1.12, cy: 0.030, cz: 0.135, r: 0.021, color: bagDark },
            { cx: sx * 1.04, cy: 0.150, cz: 0.145, r: 0.022, color: bagDark },
            { cx: sx * 0.90, cy: 0.175, cz: 0.062, r: 0.019, color: bagDark }
          ],
          8,
          { axis: 'CURVE', defaultJoint: J_PROP_MAIN, defaultMat: HUMAN_MAT.SHOE_BAG }
        );
      }
    }
  }

  /**
   * 6. DYNAMIC 3D 8-RIB SCALLOPED CAMPUS RAIN UMBRELLA (J_PROP_UMBRELLA = 21):
   *    Origin (0,0,0) is at the hand grip; chrome shaft rises to y = +0.62m; 8-rib domed waterproof
   *    canopy (radius = 0.54m) crowns y = 0.44m..0.64m with top ferrule spike at y = 0.68m!
   */
  static _addDynamicRainUmbrella(builder, profile) {
    const { umbrellaColor = HUMAN_PALETTE.UMBRELLA_NAVY, hasUmbrella = true } = profile;
    if (!hasUmbrella) return;

    const canopyTrim = HUMAN_PALETTE.JAKUN_YELLOW;

    // Ergonomic J-Hook Handle + Chrome Shaft + Top Ferrule
    builder.addWatertightLoft(
      [
        { cx: 0.028, cy: -0.045, cz: 0.0, r: 0.010, color: HUMAN_PALETTE.SHOE_BLACK, mat: HUMAN_MAT.SHOE_BAG },
        { cx: 0.016, cy: -0.068, cz: 0.0, r: 0.010, color: HUMAN_PALETTE.SHOE_BLACK, mat: HUMAN_MAT.SHOE_BAG },
        { cx: 0.0, cy: -0.052, cz: 0.0, r: 0.011, color: HUMAN_PALETTE.SHOE_BLACK, mat: HUMAN_MAT.SHOE_BAG },
        { cx: 0.0, cy: 0.035, cz: 0.0, r: 0.011, color: HUMAN_PALETTE.SHOE_BLACK, mat: HUMAN_MAT.SHOE_BAG },
        { cx: 0.0, cy: 0.040, cz: 0.0, r: 0.0055, color: HUMAN_PALETTE.METAL_SILVER, mat: HUMAN_MAT.METAL_ACCENT },
        { cx: 0.0, cy: 0.635, cz: 0.0, r: 0.0055, color: HUMAN_PALETTE.METAL_SILVER, mat: HUMAN_MAT.METAL_ACCENT },
        { cx: 0.0, cy: 0.685, cz: 0.0, r: 0.0030, color: HUMAN_PALETTE.METAL_SILVER, mat: HUMAN_MAT.METAL_ACCENT }
      ],
      8,
      { axis: 'CURVE', defaultJoint: J_PROP_UMBRELLA }
    );

    // Double-Walled 8-Rib Scalloped Waterproof Umbrella Dome Canopy (R = 0.54m!)
    const ribMod = (th) => 1.0 + 0.032 * Math.abs(Math.cos(4.0 * th)); // 8 taut structural umbrella ribs!
    const canopyRings = [
      {
        y: 0.455,
        rx: 0.545,
        rz: 0.545,
        thStart: 0.001,
        thEnd: Math.PI * 2.0 - 0.001,
        modFn: ribMod,
        color: canopyTrim
      },
      {
        y: 0.515,
        rx: 0.485,
        rz: 0.485,
        thStart: 0.001,
        thEnd: Math.PI * 2.0 - 0.001,
        modFn: ribMod,
        color: umbrellaColor
      },
      {
        y: 0.575,
        rx: 0.365,
        rz: 0.365,
        thStart: 0.001,
        thEnd: Math.PI * 2.0 - 0.001,
        modFn: ribMod,
        color: umbrellaColor
      },
      {
        y: 0.622,
        rx: 0.195,
        rz: 0.195,
        thStart: 0.001,
        thEnd: Math.PI * 2.0 - 0.001,
        modFn: ribMod,
        color: umbrellaColor
      },
      {
        y: 0.642,
        rx: 0.032,
        rz: 0.032,
        thStart: 0.001,
        thEnd: Math.PI * 2.0 - 0.001,
        color: HUMAN_PALETTE.METAL_DARK
      }
    ];

    builder.addThickOpenGarmentShell(canopyRings, 24, {
      thickness: 0.005,
      outerMat: HUMAN_MAT.UMBRELLA_CANOPY,
      innerMat: HUMAN_MAT.UMBRELLA_CANOPY,
      innerColorScale: 0.68,
      defaultJoint: J_PROP_UMBRELLA
    });
  }
}
