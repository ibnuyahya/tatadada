import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./core/MasterSurfaceBuilder.js
 *
 * SSoST Subdomain Slicer, Normal-Offset Applique Generator, & Analytical SDF Combinators
 * Implements Hukum Emas 1, 2, 3, & 17 (Zero-Gap Subdomain Continuity ||P_A - P_B|| = 0).
 */
export class MasterSurfaceBuilder {
  /**
   * Extracts a watertight subdomain mesh directly from a Single Source of Spatial Truth
   * evaluator function evaluateFn(u, v, targetPos, targetNorm).
   *
   * Guarantees 0.0000m seam gap between adjacent subdomains sharing boundary parameters
   * and enforces outward triangle winding order (N_out = T_u x T_v).
   *
   * @param {Function} evaluateFn - (u, v, targetPos, targetNorm) => void
   * @param {number} uMin - Start u in [0, 1]
   * @param {number} uMax - End u in [0, 1]
   * @param {number} vMin - Start v in [0, 1]
   * @param {number} vMax - End v in [0, 1]
   * @param {number} uSteps - Longitudinal/azimuthal subdivisions
   * @param {number} vSteps - Vertical/axial subdivisions
   * @param {Function|null} normalOffsetFn - Optional (u, v) => scalar displacement along N
   * @param {Object} options - Optional UV & winding controls
   * @returns {THREE.BufferGeometry}
   */
  static sliceSubdomain(
    evaluateFn,
    uMin,
    uMax,
    vMin,
    vMax,
    uSteps = 32,
    vSteps = 32,
    normalOffsetFn = null,
    options = {}
  ) {
    const {
      uvMode = 'parametric', // 'parametric' | 'worldHorizontal' | 'facadeNormalized'
      uvScaleU = 1.0,
      uvScaleV = 1.0,
      yMinNorm = 0.0,
      totalHeightNorm = 46.5,
      flipWinding = false
    } = options;

    const vertCount = (uSteps + 1) * (vSteps + 1);
    const positions = new Float32Array(vertCount * 3);
    const normals = new Float32Array(vertCount * 3);
    const uvs = new Float32Array(vertCount * 2);
    const indices = [];

    const p = new THREE.Vector3();
    const n = new THREE.Vector3();
    let ptr = 0;
    let uvPtr = 0;

    for (let iu = 0; iu <= uSteps; iu++) {
      const tU = iu / uSteps;
      const u = uMin + (uMax - uMin) * tU;

      for (let iv = 0; iv <= vSteps; iv++) {
        const tV = iv / vSteps;
        const v = vMin + (vMax - vMin) * tV;

        evaluateFn(u, v, p, n);
        if (n.lengthSq() < 1e-10) {
          n.set(0, 1, 0);
        } else {
          n.normalize();
        }

        if (normalOffsetFn) {
          const d = normalOffsetFn(u, v);
          if (d !== 0) {
            p.addScaledVector(n, d);
          }
        }

        positions[ptr] = p.x;
        positions[ptr + 1] = p.y;
        positions[ptr + 2] = p.z;

        normals[ptr] = n.x;
        normals[ptr + 1] = n.y;
        normals[ptr + 2] = n.z;
        ptr += 3;

        if (uvMode === 'worldHorizontal') {
          // Hukum Emas 10: Planar horizontal UV mapping preventing vertical streak stretching
          uvs[uvPtr++] = p.x * uvScaleU;
          uvs[uvPtr++] = p.z * uvScaleV;
        } else if (uvMode === 'facadeNormalized') {
          // Hukum Emas 3: Normalized vertical hNorm & non-mirrored horizontal UV
          const hNorm = Math.max(0.0, Math.min(1.0, (p.y - yMinNorm) / Math.max(0.001, totalHeightNorm)));
          uvs[uvPtr++] = u * uvScaleU;
          uvs[uvPtr++] = hNorm * uvScaleV;
        } else {
          uvs[uvPtr++] = u * uvScaleU;
          uvs[uvPtr++] = v * uvScaleV;
        }
      }
    }

    // Verify natural winding orientation at subdomain midpoint so triangle normals always match outward N
    const p00 = new THREE.Vector3();
    const p10 = new THREE.Vector3();
    const p01 = new THREE.Vector3();
    const nMid = new THREE.Vector3();
    const uMid = 0.5 * (uMin + uMax);
    const vMid = 0.5 * (vMin + vMax);
    const du = Math.max(1e-4, (uMax - uMin) / Math.max(4, uSteps));
    const dv = Math.max(1e-4, (vMax - vMin) / Math.max(4, vSteps));

    evaluateFn(uMid, vMid, p00, nMid);
    evaluateFn(Math.min(1.0, uMid + du), vMid, p10, new THREE.Vector3());
    evaluateFn(uMid, Math.min(1.0, vMid + dv), p01, new THREE.Vector3());

    const tU = p10.sub(p00);
    const tV = p01.clone().sub(p00);
    const crossUV = new THREE.Vector3().crossVectors(tU, tV);
    const shouldReverse = flipWinding ? crossUV.dot(nMid) >= 0 : crossUV.dot(nMid) < 0;

    const stride = vSteps + 1;
    for (let iu = 0; iu < uSteps; iu++) {
      for (let iv = 0; iv < vSteps; iv++) {
        const a = iu * stride + iv;
        const b = (iu + 1) * stride + iv;
        const c = (iu + 1) * stride + (iv + 1);
        const d = iu * stride + (iv + 1);

        if (shouldReverse) {
          indices.push(a, d, c, a, c, b);
        } else {
          indices.push(a, b, c, a, c, d);
        }
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    return geo;
  }

  /**
   * Builds a watertight Normal-Offset Applique shell (outer face + return flange walls)
   * directly on top of the Master Surface so secondary details never float or sink (Hukum Emas 17).
   */
  static buildNormalOffsetAppliqueBand(
    evaluateFn,
    uMin,
    uMax,
    vMin,
    vMax,
    uSteps,
    vSteps,
    offsetThickness,
    options = {}
  ) {
    const { closeU = false, closeVBottom = true, closeVTop = true, uvScaleU = 1.0, uvScaleV = 1.0 } = options;

    const positions = [];
    const normals = [];
    const uvs = [];
    const indices = [];

    const pBase = new THREE.Vector3();
    const pExt = new THREE.Vector3();
    const nBase = new THREE.Vector3();

    const pushVert = (pos, norm, u, v) => {
      const idx = positions.length / 3;
      positions.push(pos.x, pos.y, pos.z);
      normals.push(norm.x, norm.y, norm.z);
      uvs.push(u * uvScaleU, v * uvScaleV);
      return idx;
    };

    // 1. Outer offset surface
    const outerGrid = [];
    const baseGrid = [];
    for (let iu = 0; iu <= uSteps; iu++) {
      const u = uMin + (uMax - uMin) * (iu / uSteps);
      const outerCol = [];
      const baseCol = [];
      for (let iv = 0; iv <= vSteps; iv++) {
        const v = vMin + (vMax - vMin) * (iv / vSteps);
        evaluateFn(u, v, pBase, nBase);
        nBase.normalize();
        const d = typeof offsetThickness === 'function' ? offsetThickness(u, v) : offsetThickness;
        pExt.copy(pBase).addScaledVector(nBase, d);
        outerCol.push(pushVert(pExt, nBase, u, v));
        baseCol.push({ pos: pBase.clone(), ext: pExt.clone(), norm: nBase.clone(), u, v });
      }
      outerGrid.push(outerCol);
      baseGrid.push(baseCol);
    }

    // Check outward winding against base normal
    const midU = Math.floor(uSteps / 2);
    const midV = Math.floor(vSteps / 2);
    const c00 = baseGrid[midU][midV].ext;
    const c10 = baseGrid[Math.min(uSteps, midU + 1)][midV].ext;
    const c01 = baseGrid[midU][Math.min(vSteps, midV + 1)].ext;
    const triCross = new THREE.Vector3().subVectors(c10, c00).cross(new THREE.Vector3().subVectors(c01, c00));
    const rev = triCross.dot(baseGrid[midU][midV].norm) < 0;

    const pushQuad = (a, b, c, d, reverseFlag) => {
      if (reverseFlag) {
        indices.push(a, d, c, a, c, b);
      } else {
        indices.push(a, b, c, a, c, d);
      }
    };

    for (let iu = 0; iu < uSteps; iu++) {
      for (let iv = 0; iv < vSteps; iv++) {
        pushQuad(
          outerGrid[iu][iv],
          outerGrid[iu + 1][iv],
          outerGrid[iu + 1][iv + 1],
          outerGrid[iu][iv + 1],
          rev
        );
      }
    }

    // 2. Return flange walls connecting applique edge cleanly back to P_master(u, v)
    if (closeVBottom) {
      const nDown = new THREE.Vector3(0, -1, 0);
      for (let iu = 0; iu < uSteps; iu++) {
        const b0 = baseGrid[iu][0];
        const b1 = baseGrid[iu + 1][0];
        const i0 = pushVert(b0.pos, nDown, b0.u, b0.v);
        const i1 = pushVert(b1.pos, nDown, b1.u, b1.v);
        const i2 = pushVert(b1.ext, nDown, b1.u, b1.v);
        const i3 = pushVert(b0.ext, nDown, b0.u, b0.v);
        pushQuad(i0, i1, i2, i3, rev);
      }
    }

    if (closeVTop) {
      const nUp = new THREE.Vector3(0, 1, 0);
      for (let iu = 0; iu < uSteps; iu++) {
        const b0 = baseGrid[iu][vSteps];
        const b1 = baseGrid[iu + 1][vSteps];
        const i0 = pushVert(b0.ext, nUp, b0.u, b0.v);
        const i1 = pushVert(b1.ext, nUp, b1.u, b1.v);
        const i2 = pushVert(b1.pos, nUp, b1.u, b1.v);
        const i3 = pushVert(b0.pos, nUp, b0.u, b0.v);
        pushQuad(i0, i1, i2, i3, rev);
      }
    }

    if (closeU) {
      const nSideStart = new THREE.Vector3();
      const nSideEnd = new THREE.Vector3();
      for (let iv = 0; iv < vSteps; iv++) {
        const s0 = baseGrid[0][iv];
        const s1 = baseGrid[0][iv + 1];
        nSideStart.set(-s0.norm.z, 0, s0.norm.x).normalize();
        const a0 = pushVert(s0.pos, nSideStart, s0.u, s0.v);
        const a1 = pushVert(s0.ext, nSideStart, s0.u, s0.v);
        const a2 = pushVert(s1.ext, nSideStart, s1.u, s1.v);
        const a3 = pushVert(s1.pos, nSideStart, s1.u, s1.v);
        pushQuad(a0, a1, a2, a3, rev);

        const e0 = baseGrid[uSteps][iv];
        const e1 = baseGrid[uSteps][iv + 1];
        nSideEnd.set(e0.norm.z, 0, -e0.norm.x).normalize();
        const z0 = pushVert(e0.ext, nSideEnd, e0.u, e0.v);
        const z1 = pushVert(e0.pos, nSideEnd, e0.u, e0.v);
        const z2 = pushVert(e1.pos, nSideEnd, e1.u, e1.v);
        const z3 = pushVert(e1.ext, nSideEnd, e1.u, e1.v);
        pushQuad(z0, z1, z2, z3, rev);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    geo.setIndex(indices);
    return geo;
  }

  /** SDF Smooth-Union (Kernel 2 & Kernel 3) */
  static opSmoothUnion(d1, d2, k = 0.1) {
    if (k <= 1e-6) return Math.min(d1, d2);
    const h = Math.max(k - Math.abs(d1 - d2), 0.0) / k;
    return Math.min(d1, d2) - h * h * k * 0.25;
  }

  /** SDF Smooth-Subtraction */
  static opSmoothSubtraction(d1, d2, k = 0.1) {
    if (k <= 1e-6) return Math.max(-d1, d2);
    const h = Math.max(k - Math.abs(-d1 - d2), 0.0) / k;
    return Math.max(-d1, d2) + h * h * k * 0.25;
  }

  /** 2D Chamfered / Octagonal Box Exact-Style SDF */
  static sdChamferBox2D(x, z, halfW, halfL, chamfer = 0.0) {
    const ax = Math.abs(x);
    const az = Math.abs(z);
    const dx = ax - halfW;
    const dz = az - halfL;
    const rectDist = Math.hypot(Math.max(dx, 0.0), Math.max(dz, 0.0)) + Math.min(Math.max(dx, dz), 0.0);
    if (chamfer <= 1e-5) return rectDist;
    // Octagonal corner cut plane at 45 degrees
    const cutPlane = (ax + az - (halfW + halfL - chamfer * 1.41421356)) * 0.70710678;
    return Math.max(rectDist, cutPlane);
  }
}
