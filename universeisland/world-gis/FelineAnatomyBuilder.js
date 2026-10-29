import * as THREE from 'three';
import { HUMAN_MAT } from './HumanShaderMaterial.js';

/**
 * NEXUS Omni-Scale Procedural Engine v3.3 (Kucing Kepulauan UI Edition)
 * ./world-gis/FelineAnatomyBuilder.js
 *
 * Anatomically & Proportionally Realistic 18-Joint Dual-Bone Linear-Blend Skinned (LBS)
 * Domestic Cat (Felis catus) Procedural Generator:
 * - True Zoological Metric Scale (Adult Withers Height ~0.252m, Nose-to-Rump Length ~0.472m,
 *   Kittens ~0.16m tall, Lithe Juveniles ~0.22m, Chonky Tomcats ~0.28m).
 * - Seamless Watertight Head-Neck-Torso Loft (Sculpted Cranium, Cheek Jowls, Forehead Stop,
 *   Tapered Muzzle, Mystacial Whisker Pads, Split Upper Lip & Wet Rhinarium Nose Leather).
 * - Cup-Shaped Triangular Pinnae Ears with Recessed Translucent Rose Concha (Independent Swivel),
 *   Forward-Slanted Almond Feline Eyes (Diurnal Vertical-Slit Pupils + Night Tapetum Lucidum +
 *   Odd-Eye Heterochromia), 6 Fine 3D Whiskers, & Optional Campus Breakaway Collar + Bell.
 * - True Digitigrade Forelimbs & Hindlimbs (Muscular Haunch -> Stifle -> Raised Calcaneus Hock ->
 *   Metatarsus -> Sculpted 4-Toe Rounded Paws) + 4 Authentic Tail Morphologies (Full Sleek,
 *   Angora Fluffy Plume, Nusantara Short Bobtail, & Nusantara Kinked Tail).
 * - 12 Procedural Genetic Coat Patterns rendered in 0 Extra Draw Calls inside the unified LBS mesh!
 */

export const CJ_PELVIS = 0;
export const CJ_SPINE_MID = 1;
export const CJ_CHEST = 2;
export const CJ_NECK = 3;
export const CJ_HEAD = 4;
export const CJ_EAR_L = 5;
export const CJ_EAR_R = 6;
export const CJ_TAIL_BASE = 7;
export const CJ_TAIL_MID = 8;
export const CJ_TAIL_TIP = 9;
export const CJ_FRONT_SHOULDER_L = 10;
export const CJ_FRONT_PAW_L = 11;
export const CJ_FRONT_SHOULDER_R = 12;
export const CJ_FRONT_PAW_R = 13;
export const CJ_HIND_HIP_L = 14;
export const CJ_HIND_PAW_L = 15;
export const CJ_HIND_HIP_R = 16;
export const CJ_HIND_PAW_R = 17;
export const CAT_JOINTS_PER_CAT = 18;

// Exact Standing Bind-Pose Joint Pivots for Standard Adult Cat (overallScale = 1.0, facing +Z, paws at y = 0.000m)
export const CAT_BIND_PIVOTS = [
  [0.0, 0.208, -0.142],    // 0: CJ_PELVIS (Root Sacrum / Hip Pivot)
  [0.0, 0.216, -0.022],    // 1: CJ_SPINE_MID (Flexible Lumbar Spine)
  [0.0, 0.208, 0.098],     // 2: CJ_CHEST (Thoracic Ribcage & Scapula)
  [0.0, 0.238, 0.156],     // 3: CJ_NECK (Cervical Neck)
  [0.0, 0.268, 0.192],     // 4: CJ_HEAD (Cranium Pivot)
  [0.027, 0.298, 0.190],   // 5: CJ_EAR_L (Left Pinna Base)
  [-0.027, 0.298, 0.190],  // 6: CJ_EAR_R (Right Pinna Base)
  [0.0, 0.224, -0.204],    // 7: CJ_TAIL_BASE (Caudal Root)
  [0.0, 0.228, -0.296],    // 8: CJ_TAIL_MID (Mid-Tail S-Curve)
  [0.0, 0.234, -0.388],    // 9: CJ_TAIL_TIP (Expressive Tail Tip)
  [0.041, 0.196, 0.112],   // 10: CJ_FRONT_SHOULDER_L
  [0.039, 0.092, 0.114],   // 11: CJ_FRONT_PAW_L (Elbow/Forearm/Carpus)
  [-0.041, 0.196, 0.112],  // 12: CJ_FRONT_SHOULDER_R
  [-0.039, 0.092, 0.114],  // 13: CJ_FRONT_PAW_R
  [0.045, 0.200, -0.146],  // 14: CJ_HIND_HIP_L
  [0.043, 0.086, -0.174],  // 15: CJ_HIND_PAW_L (Raised Digitigrade Hock!)
  [-0.045, 0.200, -0.146], // 16: CJ_HIND_HIP_R
  [-0.043, 0.086, -0.174]  // 17: CJ_HIND_PAW_R (Raised Digitigrade Hock!)
];

export const CAT_COAT_SPECS = [
  {
    coatId: 1,
    code: 'OYEN_MACKEREL',
    labelId: 'Kucing Oyen Belang Harimau',
    baseFur: [0.82, 0.43, 0.13],
    noseCol: [0.74, 0.41, 0.36],
    earInnerCol: [0.78, 0.52, 0.48],
    eyePool: [[0.76, 0.55, 0.12], [0.46, 0.68, 0.20], [0.82, 0.44, 0.10]]
  },
  {
    coatId: 2,
    code: 'OYEN_BICOLOR_WHITE',
    labelId: 'Kucing Oyen-Putih Dasi & Kaos Kaki',
    baseFur: [0.84, 0.46, 0.15],
    noseCol: [0.84, 0.52, 0.50],
    earInnerCol: [0.84, 0.58, 0.56],
    eyePool: [[0.78, 0.56, 0.14], [0.42, 0.70, 0.24]]
  },
  {
    coatId: 3,
    code: 'MUJAIR_BROWN_TABBY',
    labelId: 'Kucing Mujair Belang Kampung',
    baseFur: [0.47, 0.39, 0.30],
    noseCol: [0.58, 0.30, 0.28],
    earInnerCol: [0.64, 0.46, 0.42],
    eyePool: [[0.42, 0.72, 0.22], [0.68, 0.64, 0.16], [0.74, 0.54, 0.14]]
  },
  {
    coatId: 4,
    code: 'SILVER_BLOTCHED_TABBY',
    labelId: 'Kucing Tabby Perak Marmer',
    baseFur: [0.60, 0.61, 0.63],
    noseCol: [0.56, 0.32, 0.32],
    earInnerCol: [0.68, 0.52, 0.52],
    eyePool: [[0.32, 0.74, 0.30], [0.74, 0.58, 0.14]]
  },
  {
    coatId: 5,
    code: 'CALICO_BELANG_TIGA',
    labelId: 'Kucing Calico Belang Tiga',
    baseFur: [0.92, 0.91, 0.88],
    noseCol: [0.84, 0.53, 0.52],
    earInnerCol: [0.85, 0.58, 0.56],
    eyePool: [[0.76, 0.56, 0.14], [0.38, 0.72, 0.26]]
  },
  {
    coatId: 6,
    code: 'TUXEDO_HITAM_PUTIH',
    labelId: 'Kucing Tuxedo Hitam Dasi Putih',
    baseFur: [0.055, 0.055, 0.062],
    noseCol: [0.14, 0.13, 0.15],
    earInnerCol: [0.48, 0.34, 0.36],
    eyePool: [[0.44, 0.76, 0.22], [0.80, 0.60, 0.12]]
  },
  {
    coatId: 7,
    code: 'TORTOISESHELL_TORTIE',
    labelId: 'Kucing Tortie Hitam-Oyen',
    baseFur: [0.16, 0.11, 0.08],
    noseCol: [0.24, 0.16, 0.16],
    earInnerCol: [0.52, 0.36, 0.34],
    eyePool: [[0.82, 0.54, 0.10], [0.46, 0.72, 0.20]]
  },
  {
    coatId: 8,
    code: 'SIAMESE_SEAL_POINT',
    labelId: 'Kucing Siam Colorpoint Mata Biru',
    baseFur: [0.85, 0.78, 0.67],
    noseCol: [0.12, 0.09, 0.09],
    earInnerCol: [0.32, 0.22, 0.22],
    eyePool: [[0.18, 0.52, 0.92], [0.24, 0.60, 0.95]]
  },
  {
    coatId: 9,
    code: 'SOLID_BLACK_BOMBAY',
    labelId: 'Kucing Hitam Legam Mata Emas',
    baseFur: [0.048, 0.046, 0.054],
    noseCol: [0.09, 0.09, 0.10],
    earInnerCol: [0.28, 0.20, 0.22],
    eyePool: [[0.86, 0.60, 0.10], [0.76, 0.68, 0.14]]
  },
  {
    coatId: 10,
    code: 'SOLID_WHITE_ANGORA',
    labelId: 'Kucing Putih Angora (Odd-Eye / Biru)',
    baseFur: [0.93, 0.92, 0.90],
    noseCol: [0.88, 0.58, 0.58],
    earInnerCol: [0.88, 0.62, 0.62],
    eyePool: [[0.22, 0.58, 0.94], [0.36, 0.76, 0.30]]
  },
  {
    coatId: 11,
    code: 'RUSSIAN_BLUE_KORAT',
    labelId: 'Kucing Abu-Abu Perak Mata Zamrud',
    baseFur: [0.31, 0.35, 0.41],
    noseCol: [0.22, 0.23, 0.27],
    earInnerCol: [0.54, 0.44, 0.46],
    eyePool: [[0.24, 0.78, 0.36], [0.38, 0.74, 0.24]]
  },
  {
    coatId: 12,
    code: 'COW_CAT_HARLEQUIN',
    labelId: 'Kucing Putih Belang Sapi',
    baseFur: [0.92, 0.91, 0.89],
    noseCol: [0.84, 0.54, 0.54],
    earInnerCol: [0.82, 0.56, 0.56],
    eyePool: [[0.76, 0.60, 0.14], [0.42, 0.74, 0.22]]
  }
];

const COLLAR_COLORS = [
  [0.78, 0.16, 0.18], // Crimson Red
  [0.14, 0.38, 0.76], // UI Cobalt Blue
  [0.88, 0.66, 0.10], // UI Jakun Gold
  [0.16, 0.58, 0.48], // Tosca Emerald
  [0.68, 0.26, 0.52]  // Berry Rose
];

function fract(v) {
  return v - Math.floor(v);
}

function hash01(seed) {
  return fract(Math.sin(seed * 127.1 + 311.7) * 43758.5453123);
}

function smooth01(a, b, x) {
  const t = Math.max(0.0, Math.min(1.0, (x - a) / (b - a || 1e-6)));
  return t * t * (3.0 - 2.0 * t);
}

function lerp3(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

class FelineMeshAccumulator {
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

  appendPart(subPos, subCol, subUv, subMeta, subJ0, subJ1, subW0, subIdx) {
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
   * Lofts a watertight anatomical 3D tube along an arbitrary curve of elliptical/sculpted stations.
   * Cross-section angle `th = 0` is Dorsal (+upNormal), `th = +PI/2` is Left (+rightVec), `th = PI` is Ventral (-upNormal).
   */
  addFelineTubeLoft(stations, radialSegs, opts = {}) {
    const {
      defaultMat = HUMAN_MAT.CAT_FUR,
      metaZ = 1.15,
      defaultJoint = CJ_PELVIS,
      weightFn = null,
      capStart = true,
      capEnd = true
    } = opts;

    const subPos = [];
    const subCol = [];
    const subUv = [];
    const subMeta = [];
    const subJ0 = [];
    const subJ1 = [];
    const subW0 = [];
    const subIdx = [];

    const pushV = (px, py, pz, col, u, v, matClass, ao, mz, jA, jB, wA) => {
      const id = subPos.length / 3;
      subPos.push(px, py, pz);
      subCol.push(col[0], col[1], col[2]);
      subUv.push(u, v);
      subMeta.push(matClass, ao, mz);
      subJ0.push(jA);
      subJ1.push(jB);
      subW0.push(wA);
      return id;
    };

    const nSt = stations.length;
    const ringGrid = [];
    const tangents = [];

    for (let is = 0; is < nSt; is++) {
      const st = stations[is];
      const pPrev = stations[Math.max(0, is - 1)];
      const pNext = stations[Math.min(nSt - 1, is + 1)];

      let tx = pNext.cx - pPrev.cx;
      let ty = pNext.cy - pPrev.cy;
      let tz = pNext.cz - pPrev.cz;
      const tLen = Math.hypot(tx, ty, tz) || 1.0;
      tx /= tLen;
      ty /= tLen;
      tz /= tLen;
      tangents.push([tx, ty, tz]);

      // Reference world-up vector (fallback to +Z if vertical limb)
      let refUx = 0.0;
      let refUy = 1.0;
      let refUz = 0.0;
      if (Math.abs(ty) > 0.94) {
        refUx = 0.0;
        refUy = 0.0;
        refUz = 1.0;
      }

      // rightVec = normalize(refUp x tangent) -> points +X when tangent points +Z
      let rxV = refUy * tz - refUz * ty;
      let ryV = refUz * tx - refUx * tz;
      let rzV = refUx * ty - refUy * tx;
      const rLen = Math.hypot(rxV, ryV, rzV) || 1.0;
      rxV /= rLen;
      ryV /= rLen;
      rzV /= rLen;

      // upVec = normalize(tangent x rightVec) -> points +Y when tangent points +Z
      let uxV = ty * rzV - tz * ryV;
      let uyV = tz * rxV - tx * rzV;
      let uzV = tx * ryV - ty * rxV;
      const uLen = Math.hypot(uxV, uyV, uzV) || 1.0;
      uxV /= uLen;
      uyV /= uLen;
      uzV /= uLen;

      const ring = [];
      const uBody = st.uBody ?? is / Math.max(1, nSt - 1);

      for (let ir = 0; ir < radialSegs; ir++) {
        const th = (ir / radialSegs) * Math.PI * 2.0; // 0 = dorsal, PI/2 = left, PI = ventral, 3PI/2 = right
        const sinT = Math.sin(th);
        const cosT = Math.cos(th);

        let radX = Math.max(0.0022, st.rx);
        let radY = Math.max(0.0022, st.ry);
        if (st.modFn) {
          const m = st.modFn(th);
          radX *= m.sx ?? 1.0;
          radY *= m.sy ?? 1.0;
        }

        let offR = 0.0;
        let offU = 0.0;
        let offT = 0.0;
        if (st.offsetFn) {
          const o = st.offsetFn(th);
          offR = o.dr ?? 0.0;
          offU = o.du ?? 0.0;
          offT = o.dt ?? 0.0;
        }

        const localR = sinT * radX + offR;
        const localU = cosT * radY + offU;

        const px = st.cx + rxV * localR + uxV * localU + tx * offT;
        const py = st.cy + ryV * localR + uyV * localU + ty * offT;
        const pz = st.cz + rzV * localR + uzV * localU + tz * offT;

        // Circumferential coordinate vCirc in [-1, +1]: 0 at dorsal (th=0), +-1 at ventral (th=PI)
        const vCirc = th <= Math.PI ? th / Math.PI : (th - Math.PI * 2.0) / Math.PI;
        const col = st.colorFn ? st.colorFn(th) : st.color;
        const matClass = st.matFn ? st.matFn(th) : st.mat ?? defaultMat;
        // Subtle natural ventral & crevice ambient occlusion
        const ao = st.aoFn ? st.aoFn(th) : 0.94 - 0.16 * Math.max(0.0, -cosT);

        let jA = st.j0 ?? defaultJoint;
        let jB = st.j1 ?? jA;
        let wA = st.w0 ?? 1.0;
        if (weightFn) {
          const wSpec = weightFn(st, is / Math.max(1, nSt - 1), th, px, py, pz);
          jA = wSpec[0];
          jB = wSpec[1];
          wA = wSpec[2];
        }

        ring.push(
          pushV(
            px,
            py,
            pz,
            col,
            st.uvOverrideU !== undefined ? st.uvOverrideU(th) : uBody,
            st.uvOverrideV !== undefined ? st.uvOverrideV(th) : vCirc,
            matClass,
            ao,
            metaZ,
            jA,
            jB,
            wA
          )
        );
      }
      ringGrid.push(ring);
    }

    for (let is = 0; is < nSt - 1; is++) {
      const rA = ringGrid[is];
      const rB = ringGrid[is + 1];
      for (let ir = 0; ir < radialSegs; ir++) {
        const irNext = (ir + 1) % radialSegs;
        const a0 = rA[ir];
        const a1 = rA[irNext];
        const b0 = rB[ir];
        const b1 = rB[irNext];
        // Outward normal: (b1 - a0) x (a1 - a0) = (+t) x (+r) = +u
        subIdx.push(a0, b1, a1, a0, b0, b1);
      }
    }

    if (capStart) {
      const st0 = stations[0];
      const t0 = tangents[0];
      const nudge0 = Math.abs(st0.capNudgeZ ?? 0.0018);
      const w0Spec = weightFn
        ? weightFn(st0, 0.0, 0.0, st0.cx, st0.cy, st0.cz)
        : [st0.j0 ?? defaultJoint, st0.j1 ?? st0.j0 ?? defaultJoint, st0.w0 ?? 1.0];
      const cStart = pushV(
        st0.cx - t0[0] * nudge0,
        st0.cy - t0[1] * nudge0,
        st0.cz - t0[2] * nudge0,
        st0.capColor || (st0.colorFn ? st0.colorFn(0.0) : st0.color),
        st0.uBody ?? 0.0,
        0.0,
        st0.capMat ?? st0.mat ?? defaultMat,
        0.92,
        metaZ,
        w0Spec[0],
        w0Spec[1],
        w0Spec[2]
      );
      const r0 = ringGrid[0];
      for (let ir = 0; ir < radialSegs; ir++) {
        const irNext = (ir + 1) % radialSegs;
        // Outward normal: (+u) x (+r) = -t
        subIdx.push(cStart, r0[ir], r0[irNext]);
      }
    }

    if (capEnd) {
      const stN = stations[nSt - 1];
      const tN = tangents[nSt - 1];
      const nudgeN = Math.abs(stN.capNudgeZ ?? 0.0018);
      const wNSpec = weightFn
        ? weightFn(stN, 1.0, 0.0, stN.cx, stN.cy, stN.cz)
        : [stN.j0 ?? defaultJoint, stN.j1 ?? stN.j0 ?? defaultJoint, stN.w0 ?? 1.0];
      const cEnd = pushV(
        stN.cx + tN[0] * nudgeN,
        stN.cy + tN[1] * nudgeN,
        stN.cz + tN[2] * nudgeN,
        stN.capColor || (stN.colorFn ? stN.colorFn(0.0) : stN.color),
        stN.uBody ?? 1.0,
        0.0,
        stN.capMat ?? stN.mat ?? defaultMat,
        0.95,
        metaZ,
        wNSpec[0],
        wNSpec[1],
        wNSpec[2]
      );
      const rN = ringGrid[nSt - 1];
      for (let ir = 0; ir < radialSegs; ir++) {
        const irNext = (ir + 1) % radialSegs;
        // Outward normal: (+r) x (+u) = +t
        subIdx.push(cEnd, rN[irNext], rN[ir]);
      }
    }

    this.appendPart(subPos, subCol, subUv, subMeta, subJ0, subJ1, subW0, subIdx);
  }
}

export class FelineAnatomyBuilder {
  /**
   * Deterministically generates a biologically realistic, highly varied feline phenotype
   * covering 12 coat genetics patterns, 5 proportional age/body morphs (Kitten -> Chonky Tomcat),
   * 4 Indonesian & pedigree tail morphologies, eye colors (including heterochromia), and campus collars!
   */
  static generateCatPhenotype(seedIdx, zoneTag = 'CAMPUS') {
    const h1 = hash01(seedIdx * 13.7 + 1.9);
    const h2 = hash01(seedIdx * 29.3 + 5.1);
    const h3 = hash01(seedIdx * 47.9 + 9.7);
    const h4 = hash01(seedIdx * 71.1 + 14.3);
    const h5 = hash01(seedIdx * 97.7 + 21.8);

    // Ensure all 12 coat patterns appear across the archipelago with iconic Oyen, Mujair, Calico & Tuxedo well represented
    const coatSpec = CAT_COAT_SPECS[(seedIdx * 5 + Math.floor(h1 * 3)) % CAT_COAT_SPECS.length];
    const coatJitter = (h2 - 0.5) * 0.08;
    const baseFurColor = [
      Math.max(0.03, Math.min(0.96, coatSpec.baseFur[0] * (1.0 + coatJitter))),
      Math.max(0.03, Math.min(0.96, coatSpec.baseFur[1] * (1.0 + coatJitter))),
      Math.max(0.03, Math.min(0.96, coatSpec.baseFur[2] * (1.0 + coatJitter)))
    ];

    // Proportional Age & Body Build Class (Realistic Felis catus Metric Proportions!)
    let ageBuildClass = 'ADULT_STANDARD';
    let overallScale = 0.98 + h3 * 0.07; // ~25.0–26.5 cm shoulder height
    let girthScale = 0.96 + h4 * 0.10;
    let jowlScale = 1.0;
    let headRatio = 1.0;
    let earScale = 1.0;
    let legSlimScale = 1.0;
    let isLonghair = coatSpec.coatId === 10 || (coatSpec.coatId === 4 && h4 > 0.55) || h5 > 0.87;

    if (seedIdx % 8 === 3) {
      // ~12.5% Proportional Kittens (3–4 months old: ~15.5–17.2 cm shoulder height, slightly larger head & ears!)
      ageBuildClass = 'KITTEN';
      overallScale = 0.61 + h3 * 0.07;
      girthScale = 0.94;
      headRatio = 1.13;
      earScale = 1.12;
      legSlimScale = 0.95;
      isLonghair = coatSpec.coatId === 10;
    } else if (h3 < 0.26) {
      // Lithe Juvenile / Slender Female Kampung Cat (~21.5–23.5 cm shoulder height)
      ageBuildClass = 'LITHE_JUVENILE';
      overallScale = 0.86 + h4 * 0.07;
      girthScale = 0.88 + h2 * 0.06;
      legSlimScale = 0.94;
      earScale = 1.04;
    } else if (h3 > 0.82 && zoneTag === 'CAMPUS') {
      // Big Cheeky Chonky Campus Tomcat ("Oyen / Tuxedo Gembul Rektorat": ~27–28.5 cm shoulder height!)
      ageBuildClass = 'CHONKY_TOMCAT';
      overallScale = 1.06 + h4 * 0.07;
      girthScale = 1.18 + h2 * 0.08;
      jowlScale = 1.15;
    } else if (isLonghair) {
      ageBuildClass = 'FLUFFY_LONGHAIR';
      overallScale = 0.96 + h3 * 0.08;
      girthScale = 1.08;
    }

    // Authentic Tail Morphology: Sleek Full, Fluffy Angora Plume, Nusantara Short Bobtail, or Nusantara Kinked Tail!
    let tailMorph = 'FULL_SLEEK';
    if (isLonghair) {
      tailMorph = 'FLUFFY_PLUME';
    } else if (h4 < 0.16) {
      tailMorph = 'NUSANTARA_BOBTAIL';
    } else if (h4 < 0.27) {
      tailMorph = 'NUSANTARA_KINKED';
    }

    // Eye Color & Odd-Eye Heterochromia (Common on White Angora cats!)
    const primaryEye = coatSpec.eyePool[seedIdx % coatSpec.eyePool.length];
    let leftEyeColor = primaryEye;
    let rightEyeColor = primaryEye;
    if (coatSpec.coatId === 10 && h5 > 0.45) {
      leftEyeColor = [0.20, 0.56, 0.94];  // Sapphire Blue
      rightEyeColor = [0.80, 0.58, 0.12]; // Warm Amber Gold
    }

    // Friendly Campus Breakaway Collar + Bell on ~26% of Rektorat/Rotunda Campus Adults
    const hasCollar = zoneTag === 'CAMPUS' && ageBuildClass !== 'KITTEN' && h5 < 0.34;
    const collarColor = COLLAR_COLORS[seedIdx % COLLAR_COLORS.length];

    // Shader pattern metadata: integer part = coatId (1..12), fractional part = unique organic pattern seed
    const metaZ = coatSpec.coatId + Math.min(0.89, Math.max(0.05, h1 * 0.84 + 0.05));

    return {
      coatId: coatSpec.coatId,
      coatCode: coatSpec.code,
      labelId: coatSpec.labelId,
      ageBuildClass,
      tailMorph,
      isLonghair,
      overallScale: Number(overallScale.toFixed(3)),
      girthScale: Number(girthScale.toFixed(3)),
      jowlScale: Number(jowlScale.toFixed(3)),
      headRatio: Number(headRatio.toFixed(3)),
      earScale: Number(earScale.toFixed(3)),
      legSlimScale: Number(legSlimScale.toFixed(3)),
      baseFurColor,
      noseColor: coatSpec.noseCol,
      earInnerColor: coatSpec.earInnerCol,
      leftEyeColor,
      rightEyeColor,
      hasCollar,
      collarColor,
      metaZ: Number(metaZ.toFixed(4))
    };
  }

  /**
   * Builds a complete watertight 18-joint Dual-Bone LBS domestic cat mesh bundle
   * `{ pos, norm, col, uv, meta, j0, j1, w0, idx }` in unit adult space (scaled live by `overallScale` on CJ_PELVIS).
   */
  static buildSkinnedCat(phenotype) {
    const acc = new FelineMeshAccumulator();
    const fur = phenotype.baseFurColor;
    const noseCol = phenotype.noseColor;
    const earInCol = phenotype.earInnerColor;
    const gS = phenotype.girthScale || 1.0;
    const hR = phenotype.headRatio || 1.0;
    const jS = phenotype.jowlScale || 1.0;
    const eS = (phenotype.earScale || 1.0) * hR;
    const lS = phenotype.legSlimScale || 1.0;
    const ruff = phenotype.isLonghair ? 1.22 : 1.0;
    const metaZ = phenotype.metaZ;

    // =========================================================================
    // 1. UNIFIED SEAMLESS RUMP -> TORSO -> CHEST -> NECK -> CRANIUM -> MUZZLE LOFT
    //    Runs caudally-to-rostrally from z = -0.214m (rump) to z = +0.258m (wet rhinarium nose tip!)
    //    uBody maps 0.72 (rump) -> 0.30 (chest) -> 0.20 (neck) -> 0.00 (nose tip)
    // =========================================================================
    const bodyWeightFn = (_st, _t, _th, _px, _py, pz) => {
      if (pz < -0.082) {
        const w = smooth01(-0.115, -0.050, pz);
        return [CJ_PELVIS, CJ_SPINE_MID, 1.0 - w];
      }
      if (pz < 0.042) {
        const w = smooth01(0.010, 0.072, pz);
        return [CJ_SPINE_MID, CJ_CHEST, 1.0 - w];
      }
      if (pz < 0.132) {
        const w = smooth01(0.105, 0.155, pz);
        return [CJ_CHEST, CJ_NECK, 1.0 - w];
      }
      if (pz < 0.184) {
        const w = smooth01(0.160, 0.192, pz);
        return [CJ_NECK, CJ_HEAD, 1.0 - w];
      }
      return [CJ_HEAD, CJ_HEAD, 1.0];
    };

    const bodyStations = [
      // Rump & Muscular Croup
      { cx: 0.0, cy: 0.210, cz: -0.214, rx: 0.032 * gS, ry: 0.034 * gS, uBody: 0.71, color: fur, capNudgeZ: 0.003 },
      { cx: 0.0, cy: 0.212, cz: -0.182, rx: 0.052 * gS, ry: 0.056 * gS, uBody: 0.67, color: fur },
      // Pelvis / Haunch Crest (Withers-matching croup height ~0.268m)
      {
        cx: 0.0,
        cy: 0.210,
        cz: -0.132,
        rx: 0.057 * gS,
        ry: 0.062 * gS,
        uBody: 0.61,
        color: fur,
        modFn: (th) => ({ sx: 1.0 + 0.08 * Math.sin(th) * Math.sin(th), sy: 1.0 })
      },
      // Flank & Lumbar Waist (with optional primordial belly pouch on chonky tomcats)
      {
        cx: 0.0,
        cy: 0.206,
        cz: -0.058,
        rx: 0.054 * gS,
        ry: 0.061 * gS,
        uBody: 0.52,
        color: fur,
        offsetFn: (th) => ({ du: gS > 1.1 ? -0.009 * Math.max(0.0, -Math.cos(th)) : 0.0 })
      },
      // Mid Ribcage
      { cx: 0.0, cy: 0.202, cz: 0.022, rx: 0.058 * gS, ry: 0.066 * gS, uBody: 0.43, color: fur },
      // Deep Thoracic Chest & Scapula Withers (Top reaches 0.204 + 0.062 = 0.266m!)
      {
        cx: 0.0,
        cy: 0.204,
        cz: 0.095,
        rx: 0.056 * gS,
        ry: 0.063 * gS,
        uBody: 0.34,
        color: fur,
        modFn: (th) => ({ sx: 0.96, sy: 1.0 + 0.06 * Math.max(0.0, Math.cos(th)) })
      },
      // Anterior Chest / Shoulder Pectoral Taper (+ Fluffy Ruff on Angora/Longhair)
      { cx: 0.0, cy: 0.218, cz: 0.140, rx: 0.044 * gS * ruff, ry: 0.048 * gS * ruff, uBody: 0.27, color: fur },
      // Graceful Cervical Neck
      { cx: 0.0, cy: 0.244, cz: 0.168, rx: 0.036 * hR * ruff, ry: 0.038 * hR * ruff, uBody: 0.21, color: fur },
      // Occiput / Back of Feline Skull
      { cx: 0.0, cy: 0.272, cz: 0.186, rx: 0.043 * hR * jS, ry: 0.040 * hR, uBody: 0.17, color: fur },
      // Wide Zygomatic Cheekbones / Tomcat Jowls & Domed Forehead Crown
      {
        cx: 0.0,
        cy: 0.275,
        cz: 0.205,
        rx: 0.047 * hR * jS,
        ry: 0.042 * hR,
        uBody: 0.13,
        color: fur,
        modFn: (th) => ({
          sx: 1.0 + 0.10 * Math.pow(Math.abs(Math.sin(th)), 1.6) * (th > 1.2 && th < 5.08 ? 1.0 : 0.4),
          sy: 1.0
        })
      },
      // Orbital Brow Stop (Slight recessed eye-socket planes at th = +-0.75 rad)
      {
        cx: 0.0,
        cy: 0.272,
        cz: 0.226,
        rx: 0.039 * hR,
        ry: 0.035 * hR,
        uBody: 0.09,
        color: fur,
        modFn: (th) => {
          const eyeDent =
            Math.exp(-Math.pow(th - 0.72, 2.0) * 7.5) +
            Math.exp(-Math.pow(th - (Math.PI * 2.0 - 0.72), 2.0) * 7.5);
          return { sx: 1.0 - 0.10 * eyeDent, sy: 1.0 - 0.08 * eyeDent };
        }
      },
      // Nasal Bridge & Puffy Mystacial Whisker Pads!
      {
        cx: 0.0,
        cy: 0.264,
        cz: 0.243,
        rx: 0.025 * hR,
        ry: 0.022 * hR,
        uBody: 0.05,
        color: fur,
        modFn: (th) => {
          const whiskerPuff = Math.pow(Math.sin(th), 2.0) * smooth01(-0.2, 0.5, -Math.cos(th));
          return { sx: 1.0 + 0.22 * whiskerPuff, sy: 1.0 - 0.06 * Math.max(0.0, -Math.cos(th)) };
        }
      },
      // Anterior Muzzle & Chin Tip
      {
        cx: 0.0,
        cy: 0.261,
        cz: 0.254,
        rx: 0.015 * hR,
        ry: 0.014 * hR,
        uBody: 0.02,
        color: fur,
        colorFn: (th) => (Math.cos(th) > -0.15 ? noseCol : fur),
        matFn: (th) => (Math.cos(th) > -0.15 ? HUMAN_MAT.CAT_NOSE_EAR : HUMAN_MAT.CAT_FUR)
      },
      // Wet Triangular Rhinarium Nose Leather Tip!
      {
        cx: 0.0,
        cy: 0.263,
        cz: 0.259,
        rx: 0.0075 * hR,
        ry: 0.0065 * hR,
        uBody: 0.005,
        color: noseCol,
        mat: HUMAN_MAT.CAT_NOSE_EAR,
        capMat: HUMAN_MAT.CAT_NOSE_EAR,
        capColor: noseCol,
        capNudgeZ: 0.0015
      }
    ];

    acc.addFelineTubeLoft(bodyStations, 14, {
      defaultMat: HUMAN_MAT.CAT_FUR,
      metaZ,
      weightFn: bodyWeightFn,
      capStart: true,
      capEnd: true
    });

    // =========================================================================
    // 2. CUP-SHAPED TRIANGULAR FELINE PINNAE EARS (CJ_EAR_L & CJ_EAR_R)
    //    Outer furred convex pinna + recessed rose-pink translucent inner concha!
    // =========================================================================
    for (const side of [1, -1]) {
      const earJoint = side > 0 ? CJ_EAR_L : CJ_EAR_R;
      const bx = side * 0.026 * hR;
      const by = 0.296;
      const bz = 0.194;
      const earH = 0.048 * eS;
      const tiltOut = side * 0.24;

      const earStations = [];
      const nEar = 5;
      for (let k = 0; k < nEar; k++) {
        const t = k / (nEar - 1);
        const taper = 1.0 - Math.pow(t, 1.35) * 0.86;
        const cx = bx + tiltOut * earH * t;
        const cy = by + earH * t;
        const cz = bz + 0.005 * t;
        const wBlend = smooth01(0.0, 0.35, t);

        earStations.push({
          cx,
          cy,
          cz,
          rx: 0.017 * eS * taper,
          ry: 0.010 * eS * taper,
          uBody: 0.17 + t * 0.04,
          j0: earJoint,
          j1: CJ_HEAD,
          w0: wBlend,
          color: fur,
          // Concave cup on anterior-facing side (th around 0 in vertical tube where refUp = +Z!)
          // Note: for vertical ear tube (ty > 0.94), refUp = +Z, so cos(th) > 0 faces +Z (front of ear)!
          colorFn: (th) => (Math.cos(th) > 0.28 && t > 0.12 && t < 0.88 ? earInCol : fur),
          matFn: (th) =>
            Math.cos(th) > 0.28 && t > 0.12 && t < 0.88 ? HUMAN_MAT.CAT_NOSE_EAR : HUMAN_MAT.CAT_FUR,
          offsetFn: (th) => ({
            du: Math.cos(th) > 0.2 ? -0.0035 * eS * taper * Math.cos(th) : 0.0
          }),
          capNudgeZ: 0.001
        });
      }

      acc.addFelineTubeLoft(earStations, 9, {
        defaultMat: HUMAN_MAT.CAT_FUR,
        metaZ,
        defaultJoint: earJoint,
        capStart: false,
        capEnd: true
      });
    }

    // =========================================================================
    // 3. FORWARD-SLANTED ALMOND FELINE EYES (Diurnal Slit Pupil + Tapetum Lucidum!)
    // =========================================================================
    for (const side of [1, -1]) {
      const eyeCol = side > 0 ? phenotype.leftEyeColor : phenotype.rightEyeColor;
      const ex = side * 0.0195 * hR;
      const ey = 0.275;
      const ez = 0.233;
      const radEye = 0.0082 * hR;
      const slant = side * 0.22; // True 13-deg feline almond canthal tilt!

      const subPos = [];
      const subCol = [];
      const subUv = [];
      const subMeta = [];
      const subJ0 = [];
      const subJ1 = [];
      const subW0 = [];
      const subIdx = [];

      // Center corneal apex
      subPos.push(ex + side * 0.0015, ey, ez + 0.0046 * hR);
      subCol.push(eyeCol[0], eyeCol[1], eyeCol[2]);
      subUv.push(0.5, 0.5);
      subMeta.push(HUMAN_MAT.CAT_EYE_TAPETUM, 1.0, metaZ);
      subJ0.push(CJ_HEAD);
      subJ1.push(CJ_HEAD);
      subW0.push(1.0);

      const nSeg = 12;
      for (let ring = 1; ring <= 2; ring++) {
        const rf = ring * 0.5;
        const bulge = Math.cos(rf * Math.PI * 0.46) * 0.0046 * hR;
        for (let k = 0; k < nSeg; k++) {
          const a = (k / nSeg) * Math.PI * 2.0;
          const lx = Math.cos(a) * radEye * rf * 1.08;
          const ly = Math.sin(a) * radEye * rf * 0.86;
          const rx = lx * Math.cos(slant) - ly * Math.sin(slant);
          const ry = lx * Math.sin(slant) + ly * Math.cos(slant);

          subPos.push(ex + rx, ey + ry, ez + bulge - Math.abs(lx) * 0.14);
          subCol.push(eyeCol[0], eyeCol[1], eyeCol[2]);
          subUv.push(0.5 + Math.cos(a) * 0.48 * rf, 0.5 + Math.sin(a) * 0.48 * rf);
          subMeta.push(HUMAN_MAT.CAT_EYE_TAPETUM, 1.0, metaZ);
          subJ0.push(CJ_HEAD);
          subJ1.push(CJ_HEAD);
          subW0.push(1.0);
        }
      }

      for (let k = 0; k < nSeg; k++) {
        const kNext = (k + 1) % nSeg;
        subIdx.push(0, 1 + k, 1 + kNext);
        const r1a = 1 + k;
        const r1b = 1 + kNext;
        const r2a = 1 + nSeg + k;
        const r2b = 1 + nSeg + kNext;
        subIdx.push(r1a, r2a, r2b, r1a, r2b, r1b);
      }

      acc.appendPart(subPos, subCol, subUv, subMeta, subJ0, subJ1, subW0, subIdx);
    }

    // =========================================================================
    // 4. 6 FINE 3D MYSTACIAL WHISKERS ("Kumis Kucing") + OPTIONAL CAMPUS COLLAR & BELL
    // =========================================================================
    const whiskerCol = [0.92, 0.91, 0.88];
    for (const side of [1, -1]) {
      for (let w = 0; w < 3; w++) {
        const wyOff = (w - 1) * 0.0032;
        const droop = -0.004 - w * 0.0025;
        const sweepBack = -0.010 - w * 0.003;
        const wStations = [
          {
            cx: side * 0.017 * hR,
            cy: 0.261 + wyOff,
            cz: 0.246,
            rx: 0.0008,
            ry: 0.0008,
            uBody: 0.04,
            color: whiskerCol,
            j0: CJ_HEAD,
            j1: CJ_HEAD,
            w0: 1.0
          },
          {
            cx: side * 0.042 * hR,
            cy: 0.260 + wyOff + droop * 0.4,
            cz: 0.246 + sweepBack * 0.45,
            rx: 0.0006,
            ry: 0.0006,
            uBody: 0.04,
            color: whiskerCol,
            j0: CJ_HEAD,
            j1: CJ_HEAD,
            w0: 1.0
          },
          {
            cx: side * 0.066 * hR,
            cy: 0.258 + wyOff + droop,
            cz: 0.246 + sweepBack,
            rx: 0.0004,
            ry: 0.0004,
            uBody: 0.04,
            color: whiskerCol,
            j0: CJ_HEAD,
            j1: CJ_HEAD,
            w0: 1.0
          }
        ];
        acc.addFelineTubeLoft(wStations, 4, {
          defaultMat: HUMAN_MAT.SHOE_BAG,
          metaZ: 0.0,
          defaultJoint: CJ_HEAD,
          capStart: false,
          capEnd: false
        });
      }
    }

    if (phenotype.hasCollar) {
      const cCol = phenotype.collarColor;
      const collarStations = [
        { cx: 0.0, cy: 0.236, cz: 0.156, rx: 0.039 * hR * ruff, ry: 0.041 * hR * ruff, uBody: 0.5, color: cCol },
        { cx: 0.0, cy: 0.241, cz: 0.164, rx: 0.038 * hR * ruff, ry: 0.040 * hR * ruff, uBody: 0.5, color: cCol }
      ];
      acc.addFelineTubeLoft(collarStations, 12, {
        defaultMat: HUMAN_MAT.SHOE_BAG,
        metaZ: 0.0,
        defaultJoint: CJ_NECK,
        capStart: false,
        capEnd: false
      });
      // Tiny golden cat bell ("lonceng kucing") under throat!
      const bellCol = [0.90, 0.72, 0.18];
      const bellY = 0.240 - 0.043 * hR * ruff;
      acc.addFelineTubeLoft(
        [
          { cx: 0.0, cy: bellY, cz: 0.158, rx: 0.0055, ry: 0.0055, uBody: 0.5, color: bellCol },
          { cx: 0.0, cy: bellY, cz: 0.166, rx: 0.0055, ry: 0.0055, uBody: 0.5, color: bellCol }
        ],
        7,
        {
          defaultMat: HUMAN_MAT.METAL_ACCENT,
          metaZ: 0.0,
          defaultJoint: CJ_NECK,
          capStart: true,
          capEnd: true
        }
      );
    }

    // =========================================================================
    // 5. FOUR DIGITIGRADE SCULPTED FELINE LIMBS (uBody in [1.05 .. 1.45] for Socks/Bands!)
    // =========================================================================
    // 5A. Left & Right Forelimbs (Scapula/Humerus -> Elbow -> Slender Forearm -> Carpus -> Rounded 4-Toe Paw)
    for (const side of [1, -1]) {
      const jUpper = side > 0 ? CJ_FRONT_SHOULDER_L : CJ_FRONT_SHOULDER_R;
      const jLower = side > 0 ? CJ_FRONT_PAW_L : CJ_FRONT_PAW_R;
      const lx = side * 0.040 * gS;

      const foreStations = [
        { cx: lx, cy: 0.204, cz: 0.110, rx: 0.023 * gS, ry: 0.029 * gS, uBody: 1.05, color: fur, j0: jUpper, j1: CJ_CHEST, w0: 0.72 },
        { cx: lx, cy: 0.146, cz: 0.104, rx: 0.017 * lS, ry: 0.021 * lS, uBody: 1.15, color: fur, j0: jUpper, j1: jLower, w0: 0.82 },
        // Elbow pivot zone (y = 0.094m)
        { cx: lx * 0.96, cy: 0.094, cz: 0.112, rx: 0.014 * lS, ry: 0.016 * lS, uBody: 1.25, color: fur, j0: jLower, j1: jUpper, w0: 0.68 },
        // Carpal wrist
        { cx: lx * 0.94, cy: 0.036, cz: 0.118, rx: 0.0125 * lS, ry: 0.0135 * lS, uBody: 1.35, color: fur, j0: jLower, j1: jLower, w0: 1.0 },
        // Rounded Feline Forepaw with subtle 4-toe digital modulation!
        {
          cx: lx * 0.94,
          cy: 0.013,
          cz: 0.129,
          rx: 0.0165 * lS,
          ry: 0.019 * lS,
          uBody: 1.42,
          color: fur,
          j0: jLower,
          j1: jLower,
          w0: 1.0,
          modFn: (th) => ({
            sx: 1.0,
            sy: Math.cos(th) > 0.2 ? 1.0 + 0.06 * Math.cos(th * 4.0) : 1.0
          })
        },
        // Flush Paw Pad Sole at y = 0.001m
        {
          cx: lx * 0.94,
          cy: 0.001,
          cz: 0.130,
          rx: 0.0135 * lS,
          ry: 0.0155 * lS,
          uBody: 1.45,
          color: noseCol,
          mat: HUMAN_MAT.CAT_NOSE_EAR,
          capMat: HUMAN_MAT.CAT_NOSE_EAR,
          capColor: noseCol,
          j0: jLower,
          j1: jLower,
          w0: 1.0,
          capNudgeZ: 0.0005
        }
      ];

      acc.addFelineTubeLoft(foreStations, 8, {
        defaultMat: HUMAN_MAT.CAT_FUR,
        metaZ,
        defaultJoint: jUpper,
        capStart: true,
        capEnd: true
      });
    }

    // 5B. Left & Right Digitigrade Hindlimbs (Muscular Haunch -> Forward Stifle -> Raised Hock -> Metatarsus -> Hindpaw!)
    for (const side of [1, -1]) {
      const jHip = side > 0 ? CJ_HIND_HIP_L : CJ_HIND_HIP_R;
      const jHock = side > 0 ? CJ_HIND_PAW_L : CJ_HIND_PAW_R;
      const hx = side * 0.044 * gS;

      const hindStations = [
        // Muscular Feline Haunch / Upper Thigh
        { cx: hx, cy: 0.202, cz: -0.144, rx: 0.026 * gS, ry: 0.040 * gS, uBody: 1.05, color: fur, j0: jHip, j1: CJ_PELVIS, w0: 0.72 },
        // Forward-Angled Stifle (Knee)
        { cx: hx * 0.98, cy: 0.138, cz: -0.126, rx: 0.020 * gS, ry: 0.028 * gS, uBody: 1.16, color: fur, j0: jHip, j1: jHock, w0: 0.80 },
        // Backward-Angled Raised Digitigrade Calcaneus Hock! (y = 0.086m, z = -0.174m — unmistakable real cat silhouette!)
        { cx: hx * 0.95, cy: 0.086, cz: -0.174, rx: 0.0135 * lS, ry: 0.0175 * lS, uBody: 1.27, color: fur, j0: jHock, j1: jHip, w0: 0.74 },
        // Slender Metatarsus descending forward toward toes
        { cx: hx * 0.94, cy: 0.034, cz: -0.160, rx: 0.012 * lS, ry: 0.014 * lS, uBody: 1.36, color: fur, j0: jHock, j1: jHock, w0: 1.0 },
        // Rounded Feline Hindpaw
        {
          cx: hx * 0.94,
          cy: 0.013,
          cz: -0.148,
          rx: 0.016 * lS,
          ry: 0.0195 * lS,
          uBody: 1.42,
          color: fur,
          j0: jHock,
          j1: jHock,
          w0: 1.0,
          modFn: (th) => ({
            sx: 1.0,
            sy: Math.cos(th) > 0.2 ? 1.0 + 0.06 * Math.cos(th * 4.0) : 1.0
          })
        },
        // Flush Hindpaw Pad Sole at y = 0.001m
        {
          cx: hx * 0.94,
          cy: 0.001,
          cz: -0.147,
          rx: 0.013 * lS,
          ry: 0.0155 * lS,
          uBody: 1.45,
          color: noseCol,
          mat: HUMAN_MAT.CAT_NOSE_EAR,
          capMat: HUMAN_MAT.CAT_NOSE_EAR,
          capColor: noseCol,
          j0: jHock,
          j1: jHock,
          w0: 1.0,
          capNudgeZ: 0.0005
        }
      ];

      acc.addFelineTubeLoft(hindStations, 8, {
        defaultMat: HUMAN_MAT.CAT_FUR,
        metaZ,
        defaultJoint: jHip,
        capStart: true,
        capEnd: true
      });
    }

    // =========================================================================
    // 6. 3-JOINT EXPRESSIVE FELINE TAIL (uBody in [0.72 .. 1.00])
    //    Supports FULL_SLEEK, FLUFFY_PLUME, NUSANTARA_BOBTAIL, & NUSANTARA_KINKED!
    // =========================================================================
    const tailLen =
      phenotype.tailMorph === 'NUSANTARA_BOBTAIL'
        ? 0.105
        : phenotype.tailMorph === 'NUSANTARA_KINKED'
          ? 0.165
          : phenotype.tailMorph === 'FLUFFY_PLUME'
            ? 0.285
            : 0.275;

    const plumeMult = phenotype.tailMorph === 'FLUFFY_PLUME' ? 1.62 : 1.0;
    const isBob = phenotype.tailMorph === 'NUSANTARA_BOBTAIL';
    const isKink = phenotype.tailMorph === 'NUSANTARA_KINKED';

    const tailStations = [];
    const nTail = 8;
    for (let k = 0; k < nTail; k++) {
      const t = k / (nTail - 1);
      const cz = -0.202 - t * tailLen;
      // Nusantara kinked tail hooks gently to the side & up near the tip!
      const kinkX = isKink && t > 0.62 ? Math.pow((t - 0.62) / 0.38, 1.5) * 0.022 : 0.0;
      const cy = 0.222 + t * 0.014 + (isKink && t > 0.65 ? (t - 0.65) * 0.028 : 0.0);

      // Plume swells in mid-tail before tapering; bobtail ends in a cute rounded pom-pom!
      const plumeBell = phenotype.tailMorph === 'FLUFFY_PLUME' ? 1.0 + 0.38 * Math.sin(t * Math.PI) : 1.0;
      const rad = isBob
        ? (0.0175 - 0.005 * t + 0.004 * Math.sin(t * Math.PI)) * gS
        : (0.0175 * (1.0 - 0.56 * t)) * plumeMult * plumeBell;

      let jA = CJ_TAIL_BASE;
      let jB = CJ_TAIL_MID;
      let wA = 1.0;
      if (t < 0.12) {
        jA = CJ_TAIL_BASE;
        jB = CJ_PELVIS;
        wA = 0.65 + 0.35 * (t / 0.12);
      } else if (t < 0.52) {
        const uSeg = smooth01(0.24, 0.50, t);
        jA = CJ_TAIL_BASE;
        jB = CJ_TAIL_MID;
        wA = 1.0 - uSeg;
      } else {
        const uSeg = smooth01(0.56, 0.82, t);
        jA = CJ_TAIL_MID;
        jB = CJ_TAIL_TIP;
        wA = 1.0 - uSeg;
      }

      tailStations.push({
        cx: kinkX,
        cy,
        cz,
        rx: Math.max(0.0045, rad),
        ry: Math.max(0.0045, rad),
        uBody: 0.72 + t * 0.28,
        color: fur,
        j0: jA,
        j1: jB,
        w0: wA,
        capNudgeZ: -0.002
      });
    }

    acc.addFelineTubeLoft(tailStations, 9, {
      defaultMat: HUMAN_MAT.CAT_FUR,
      metaZ,
      defaultJoint: CJ_TAIL_BASE,
      capStart: false,
      capEnd: true
    });

    return {
      pos: acc.pos,
      norm: acc.norm,
      col: acc.col,
      uv: acc.uv,
      meta: acc.meta,
      j0: acc.j0,
      j1: acc.j1,
      w0: acc.w0,
      idx: acc.idx
    };
  }
}
