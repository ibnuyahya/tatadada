import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./core/MasterSurfaceKernels.js
 *
 * Layer 1 Mathematical SSoST Kernels:
 * - Kernel 1: Parametric Surface Slicing & Polar-Geodetic Annular Lofting (0.00m seam, outward winding guarantee)
 * - Kernel 2: Analytical Roof Valley / Courtyard Hip-Limasan Slicing & CSG Smooth SDF Operators
 * - Surface Applique Normal-Offset Helper (Zero Z-Fighting on flat & curved facades)
 */
export class NexusSSoSTKernel {
  /**
   * Smooth CSG union & subtraction primitives (Kernel 2 SDF helpers)
   */
  static opSmoothUnion(d1, d2, k = 0.15) {
    const kk = Math.max(1e-5, k);
    const h = THREE.MathUtils.clamp(0.5 + (0.5 * (d2 - d1)) / kk, 0.0, 1.0);
    return THREE.MathUtils.lerp(d2, d1, h) - kk * h * (1.0 - h);
  }

  static opSmoothSubtraction(d1, d2, k = 0.15) {
    const kk = Math.max(1e-5, k);
    const h = THREE.MathUtils.clamp(0.5 - (0.5 * (d2 + d1)) / kk, 0.0, 1.0);
    return THREE.MathUtils.lerp(d2, -d1, h) + kk * h * (1.0 - h);
  }

  /**
   * Sanitizes a Float32Array in-place so zero NaN/Infinity values can ever reach WebGL buffers.
   */
  static sanitizeFloatArray(arr, fallback = 0.0) {
    for (let i = 0, len = arr.length; i < len; i++) {
      if (!Number.isFinite(arr[i])) arr[i] = fallback;
    }
    return arr;
  }

  /**
   * Slices a continuous subdomain [uMin..uMax] x [vMin..vMax] from an analytical Master Surface S(u,v)->(p,n).
   * Guarantees outward counter-clockwise triangle winding matching analytical normal n(u,v) and finite buffers.
   */
  static sliceSubdomainFromMaster(
    evaluateFn,
    uMin = 0.0,
    uMax = 1.0,
    vMin = 0.0,
    vMax = 1.0,
    uSegs = 24,
    vSegs = 8,
    uvScaleU = 1.0,
    uvScaleV = 1.0
  ) {
    const us = Math.max(1, Math.floor(uSegs));
    const vs = Math.max(1, Math.floor(vSegs));
    const vertCount = (us + 1) * (vs + 1);
    const positions = new Float32Array(vertCount * 3);
    const normals = new Float32Array(vertCount * 3);
    const uvs = new Float32Array(vertCount * 2);
    const indices = [];

    const p = new THREE.Vector3();
    const n = new THREE.Vector3();
    let pPtr = 0;
    let uvPtr = 0;

    for (let iv = 0; iv <= vs; iv++) {
      const tv = iv / vs;
      const v = vMin + (vMax - vMin) * tv;
      for (let iu = 0; iu <= us; iu++) {
        const tu = iu / us;
        const u = uMin + (uMax - uMin) * tu;
        evaluateFn(u, v, p, n);

        const px = Number.isFinite(p.x) ? p.x : 0.0;
        const py = Number.isFinite(p.y) ? p.y : 0.0;
        const pz = Number.isFinite(p.z) ? p.z : 0.0;

        let nx = Number.isFinite(n.x) ? n.x : 0.0;
        let ny = Number.isFinite(n.y) ? n.y : 1.0;
        let nz = Number.isFinite(n.z) ? n.z : 0.0;
        const nLen = Math.hypot(nx, ny, nz);
        if (nLen > 1e-6) {
          nx /= nLen;
          ny /= nLen;
          nz /= nLen;
        } else {
          nx = 0.0;
          ny = 1.0;
          nz = 0.0;
        }

        positions[pPtr] = px;
        positions[pPtr + 1] = py;
        positions[pPtr + 2] = pz;

        normals[pPtr] = nx;
        normals[pPtr + 1] = ny;
        normals[pPtr + 2] = nz;

        uvs[uvPtr] = u * uvScaleU;
        uvs[uvPtr + 1] = v * uvScaleV;

        pPtr += 3;
        uvPtr += 2;
      }
    }

    const stride = us + 1;
    const vA = new THREE.Vector3();
    const vB = new THREE.Vector3();
    const vC = new THREE.Vector3();
    const e1 = new THREE.Vector3();
    const e2 = new THREE.Vector3();
    const faceNorm = new THREE.Vector3();
    const refNorm = new THREE.Vector3();

    for (let iv = 0; iv < vs; iv++) {
      for (let iu = 0; iu < us; iu++) {
        const a = iv * stride + iu;
        const b = a + 1;
        const c = (iv + 1) * stride + iu;
        const d = c + 1;

        vA.fromArray(positions, a * 3);
        vB.fromArray(positions, b * 3);
        vC.fromArray(positions, c * 3);
        refNorm.fromArray(normals, a * 3);

        e1.subVectors(vB, vA);
        e2.subVectors(vC, vA);
        faceNorm.crossVectors(e1, e2);

        if (faceNorm.dot(refNorm) >= 0.0) {
          indices.push(a, b, c, b, d, c);
        } else {
          indices.push(a, c, b, b, c, d);
        }
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    return geo;
  }

  /**
   * Builds a watertight polar-cylindrical sector band [th0..th1] x [rIn..rOut] x [yBot..yTop]
   * with metric world-space UVs so brick, plaster, louvers, and concrete bands never stretch.
   */
  static addCurvedPolarBand(parent, mat, th0, th1, rIn, rOut, yBot, yTop, segs = 32, opts = {}) {
    const {
      top = true,
      bottom = yBot > 0.2,
      outer = true,
      inner = true,
      startCap = true,
      endCap = true,
      uvMetersPerTile = 4.0,
      castShadow = true,
      receiveShadow = true
    } = opts;

    const spanRad = th1 - th0;
    const midR = 0.5 * (rIn + rOut);
    const arcMeters = Math.max(0.1, Math.abs(spanRad) * midR);
    const heightMeters = Math.max(0.05, Math.abs(yTop - yBot));
    const radialMeters = Math.max(0.05, Math.abs(rOut - rIn));
    const uvU = arcMeters / uvMetersPerTile;
    const uvH = heightMeters / uvMetersPerTile;
    const uvR = radialMeters / uvMetersPerTile;

    const addFace = (geo) => {
      const m = new THREE.Mesh(geo, mat);
      m.castShadow = castShadow;
      m.receiveShadow = receiveShadow;
      parent.add(m);
      return m;
    };

    if (outer) {
      addFace(
        NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const th = th0 + u * spanRad;
            const c = Math.cos(th);
            const s = Math.sin(th);
            p.set(rOut * c, yBot + v * (yTop - yBot), rOut * s);
            n.set(c, 0, s);
          },
          0, 1, 0, 1, segs, 1, uvU, uvH
        )
      );
    }

    if (inner) {
      addFace(
        NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const th = th0 + u * spanRad;
            const c = Math.cos(th);
            const s = Math.sin(th);
            p.set(rIn * c, yBot + v * (yTop - yBot), rIn * s);
            n.set(-c, 0, -s);
          },
          0, 1, 0, 1, segs, 1, uvU, uvH
        )
      );
    }

    if (top) {
      addFace(
        NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const th = th0 + u * spanRad;
            const r = rIn + v * (rOut - rIn);
            p.set(r * Math.cos(th), yTop, r * Math.sin(th));
            n.set(0, 1, 0);
          },
          0, 1, 0, 1, segs, 1, uvU, uvR
        )
      );
    }

    if (bottom) {
      addFace(
        NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const th = th0 + u * spanRad;
            const r = rIn + v * (rOut - rIn);
            p.set(r * Math.cos(th), yBot, r * Math.sin(th));
            n.set(0, -1, 0);
          },
          0, 1, 0, 1, segs, 1, uvU, uvR
        )
      );
    }

    const signSpan = spanRad >= 0 ? 1 : -1;
    if (startCap) {
      const c0 = Math.cos(th0);
      const s0 = Math.sin(th0);
      addFace(
        NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const r = rIn + u * (rOut - rIn);
            const y = yBot + v * (yTop - yBot);
            p.set(r * c0, y, r * s0);
            n.set(s0 * signSpan, 0, -c0 * signSpan);
          },
          0, 1, 0, 1, 1, 1, uvR, uvH
        )
      );
    }

    if (endCap) {
      const c1 = Math.cos(th1);
      const s1 = Math.sin(th1);
      addFace(
        NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const r = rIn + u * (rOut - rIn);
            const y = yBot + v * (yTop - yBot);
            p.set(r * c1, y, r * s1);
            n.set(-s1 * signSpan, 0, c1 * signSpan);
          },
          0, 1, 0, 1, 1, 1, uvR, uvH
        )
      );
    }
  }

  /**
   * Builds a sloped polar-conical annular surface (for sloped grass berms, sloped stone retaining walls,
   * ditch side walls, and conical curved soffits) with exact analytical surface normals.
   */
  static createPolarSlopedRibbon(th0, th1, r0, y0, r1, y1, segs = 48, uvMetersPerTile = 3.5, flipNormal = false) {
    const spanRad = th1 - th0;
    const midR = 0.5 * (r0 + r1);
    const arcLen = Math.max(0.1, Math.abs(spanRad) * midR);
    const profLen = Math.max(0.05, Math.hypot(r1 - r0, y1 - y0));
    const dr = r1 - r0;
    const dy = y1 - y0;
    const signFlip = flipNormal ? -1.0 : 1.0;

    return NexusSSoSTKernel.sliceSubdomainFromMaster(
      (u, v, p, n) => {
        const th = th0 + u * spanRad;
        const r = r0 + v * dr;
        const y = y0 + v * dy;
        const c = Math.cos(th);
        const s = Math.sin(th);
        p.set(r * c, y, r * s);
        // Tangent along theta: (-r*s, 0, r*c); tangent along v: (dr*c, dy, dr*s)
        // Outward/upward normal proportional to (-dy*c, dr, -dy*s) or (+dy*c, -dr, +dy*s)
        let nx = -dy * c * signFlip;
        let ny = dr * signFlip;
        let nz = -dy * s * signFlip;
        if (ny < 0 && !flipNormal && Math.abs(dr) > 1e-4) {
          nx = -nx;
          ny = -ny;
          nz = -nz;
        }
        n.set(nx, ny, nz).normalize();
      },
      0,
      1,
      0,
      1,
      segs,
      2,
      arcLen / uvMetersPerTile,
      profLen / uvMetersPerTile
    );
  }

  /**
   * Creates a 3D cylindrical/hexagonal ceramic ridge-cap ("nok genteng keramik") tube along a polyline of 3D points.
   */
  static createRidgeCapStrip(points, radius = 0.14, radialSegs = 6) {
    if (!points || points.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.02);
    const tubularSegs = Math.max(2, (points.length - 1) * 4);
    const geo = new THREE.TubeGeometry(curve, tubularSegs, radius, radialSegs, false);
    return geo;
  }

  /**
   * Pushes a True-3D Thickened & Stepped-Course Terracotta Roof Quad (or Triangle when p2==p3)
   * into (pos, norm, uvs) buffers.
   * - Generates 3D overlapping horizontal tile courses with physical step lips (stepHeight = 0.028m)
   * - Generates 3D solid slab thickness along the eave edge, ridge edge, and side verges (slabThickness = 0.11m)
   * - Generates 3D corrugated ceramic eave drip-tile nosings ("profil gelombang genteng cucuran 3D") along the eave edge!
   */
  static pushThickSteppedRoofQuad(pos, norm, uvs, inP0, inP1, inP2, inP3, opts = {}) {
    const tileM = opts.uvTileMeters || 1.8;
    const courses = opts.courses || 9;
    const stepH = opts.stepHeight ?? 0.028;
    const slabThick = opts.slabThickness ?? 0.10;
    const addEaveNosings = opts.eaveNosings !== false;

    let p0 = inP0;
    let p1 = inP1;
    let p2 = inP2;
    let p3 = inP3;

    const v0 = new THREE.Vector3(...p0);
    const v1 = new THREE.Vector3(...p1);
    const v2 = new THREE.Vector3(...p2);
    const v3 = new THREE.Vector3(...p3);

    // Ensure upward normal (+Y) while keeping (p0, p1) at the lower eave and (p3, p2) at the upper ridge
    const e1 = new THREE.Vector3().subVectors(v1, v0);
    const e2 = new THREE.Vector3().subVectors(v3, v0);
    if (e2.lengthSq() < 1e-6) e2.subVectors(v2, v0);
    const fn = new THREE.Vector3().crossVectors(e1, e2).normalize();
    if (fn.y < 0) {
      p0 = inP1;
      p1 = inP0;
      p2 = inP3;
      p3 = inP2;
      fn.negate();
    }

    const downDir = new THREE.Vector3(
      0.5 * (p0[0] + p1[0] - p2[0] - p3[0]),
      0.5 * (p0[1] + p1[1] - p2[1] - p3[1]),
      0.5 * (p0[2] + p1[2] - p2[2] - p3[2])
    ).normalize();

    const emitTri = (a, b, c, nVec) => {
      for (const pt of [a, b, c]) {
        pos.push(pt[0], pt[1], pt[2]);
        norm.push(nVec.x, nVec.y, nVec.z);
        uvs.push(pt[0] / tileM, (pt[2] + pt[1]) / tileM);
      }
    };
    const emitQuad = (a, b, c, d, nVec) => {
      emitTri(a, b, c, nVec);
      emitTri(a, c, d, nVec);
    };

    const lerp3 = (a, b, t) => [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t
    ];
    const offset3 = (a, n, dist) => [
      a[0] + n.x * dist,
      a[1] + n.y * dist,
      a[2] + n.z * dist
    ];

    // 1. Stepped 3D overlapping tile courses from eave (t=0) to ridge (t=1)
    for (let k = 0; k < courses; k++) {
      const t0 = k / courses;
      const t1 = (k + 1) / courses;
      const lBotBase = lerp3(p0, p3, t0);
      const rBotBase = lerp3(p1, p2, t0);
      const lTopBase = lerp3(p0, p3, t1);
      const rTopBase = lerp3(p1, p2, t1);

      const lBotUp = offset3(lBotBase, fn, stepH);
      const rBotUp = offset3(rBotBase, fn, stepH);
      const lTopUp = offset3(lTopBase, fn, 0.005);
      const rTopUp = offset3(rTopBase, fn, 0.005);

      // Upper weathered ceramic tile slope of course k
      emitQuad(lBotUp, rBotUp, rTopUp, lTopUp, fn);

      // 3D vertical front overlap lip of course k (gives visible depth to every tile row!)
      const lBotLow = offset3(lBotBase, fn, k === 0 ? -slabThick : 0.003);
      const rBotLow = offset3(rBotBase, fn, k === 0 ? -slabThick : 0.003);
      emitQuad(lBotLow, rBotLow, rBotUp, lBotUp, downDir);
    }

    // 2. Solid 3D Slab Side Verges & Underside Back-Face Thickness
    const b0 = offset3(p0, fn, -slabThick);
    const b1 = offset3(p1, fn, -slabThick);
    const b2 = offset3(p2, fn, -slabThick);
    const b3 = offset3(p3, fn, -slabThick);
    const t0 = offset3(p0, fn, stepH);
    const t1 = offset3(p1, fn, stepH);
    const t2 = offset3(p2, fn, 0.005);
    const t3 = offset3(p3, fn, 0.005);

    const botNorm = fn.clone().negate();
    emitQuad(b0, b3, b2, b1, botNorm);

    const sideLNorm = new THREE.Vector3().crossVectors(fn, new THREE.Vector3().subVectors(new THREE.Vector3(...p3), new THREE.Vector3(...p0))).normalize();
    const sideRNorm = new THREE.Vector3().crossVectors(new THREE.Vector3().subVectors(new THREE.Vector3(...p2), new THREE.Vector3(...p1)), fn).normalize();
    emitQuad(b3, b0, t0, t3, sideLNorm);
    emitQuad(b1, b2, t2, t1, sideRNorm);

    // 3. 3D Corrugated Ceramic Eave Drip-Tile Nosings ("genteng list cucuran 3D") along the eave edge (p0 -> p1)
    const eaveLen = Math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
    if (addEaveNosings && eaveLen > 0.6) {
      const tilePitch = 0.34;
      const nTiles = Math.min(64, Math.max(2, Math.floor(eaveLen / tilePitch)));
      const eaveTang = new THREE.Vector3(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]).normalize();
      for (let i = 0; i < nTiles; i++) {
        const uA = (i + 0.14) / nTiles;
        const uB = (i + 0.86) / nTiles;
        const eA = lerp3(p0, p1, uA);
        const eB = lerp3(p0, p1, uB);
        const eBackA = lerp3(eA, lerp3(p3, p2, 0.5 * (uA + uB)), 0.055);
        const eBackB = lerp3(eB, lerp3(p3, p2, 0.5 * (uA + uB)), 0.055);

        const crownH = stepH + 0.034;
        const fA = offset3(eA, fn, crownH);
        const fB = offset3(eB, fn, crownH);
        const fBotA = offset3(eA, fn, -0.02);
        const fBotB = offset3(eB, fn, -0.02);
        const bkA = offset3(eBackA, fn, stepH * 0.6);
        const bkB = offset3(eBackB, fn, stepH * 0.6);

        // Front lip, top arch, and side cheeks of each 3D ceramic eave tile
        emitQuad(fBotA, fBotB, fB, fA, downDir);
        emitQuad(fA, fB, bkB, bkA, fn);
        emitTri(fBotA, fA, bkA, eaveTang.clone().negate());
        emitTri(fBotB, bkB, fB, eaveTang);
      }
    }
  }

  /**
   * Builds a watertight 4-wing Courtyard Hip/Limasan Roof (for ITTC Perpustakaan Lama 42m x 42m w/ 16m courtyard).
   * Outer eave square: [-halfOut, +halfOut], Inner courtyard eave square: [-halfIn, +halfIn],
   * Continuous square ridge loop at halfRidge = 0.5*(halfOut + halfIn) at y = ridgeY, eaves at y = eaveY.
   * Includes 3D stepped tile courses, 3D slab thickness, 3D corrugated eave tile nosings, and flat underside soffit.
   */
  static createSquareCourtyardLimasanRoof(halfOut, halfIn, eaveY, ridgeY, uvTileMeters = 2.0) {
    const halfRidge = 0.5 * (halfOut + halfIn);
    const pos = [];
    const norm = [];
    const uvs = [];

    const pushQuad = (p0, p1, p2, p3) => {
      NexusSSoSTKernel.pushThickSteppedRoofQuad(pos, norm, uvs, p0, p1, p2, p3, {
        uvTileMeters,
        courses: 10,
        stepHeight: 0.030,
        slabThickness: 0.12,
        eaveNosings: true
      });
    };

    // 4 cardinal directions: South (+Z), East (+X), North (-Z), West (-X)
    // Outer corners (CCW from top view):
    const oSW = [-halfOut, eaveY, halfOut];
    const oSE = [halfOut, eaveY, halfOut];
    const oNE = [halfOut, eaveY, -halfOut];
    const oNW = [-halfOut, eaveY, -halfOut];

    const rSW = [-halfRidge, ridgeY, halfRidge];
    const rSE = [halfRidge, ridgeY, halfRidge];
    const rNE = [halfRidge, ridgeY, -halfRidge];
    const rNW = [-halfRidge, ridgeY, -halfRidge];

    const iSW = [-halfIn, eaveY, halfIn];
    const iSE = [halfIn, eaveY, halfIn];
    const iNE = [halfIn, eaveY, -halfIn];
    const iNW = [-halfIn, eaveY, -halfIn];

    // 4 Outer trapezoidal hip slopes
    pushQuad(oSW, oSE, rSE, rSW); // South outer slope
    pushQuad(oSE, oNE, rNE, rSE); // East outer slope
    pushQuad(oNE, oNW, rNW, rNE); // North outer slope
    pushQuad(oNW, oSW, rSW, rNW); // West outer slope

    // 4 Inner trapezoidal courtyard valley slopes
    pushQuad(iSE, iSW, rSW, rSE); // South inner courtyard slope
    pushQuad(iNE, iSE, rSE, rNE); // East inner courtyard slope
    pushQuad(iNW, iNE, rNE, rNW); // North inner courtyard slope
    pushQuad(iSW, iNW, rNW, rSW); // West inner courtyard slope

    const roofGeo = new THREE.BufferGeometry();
    roofGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    roofGeo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
    roofGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    roofGeo.computeBoundingBox();
    roofGeo.computeBoundingSphere();

    // Flat underside eave soffit ring [halfIn..halfOut] at eaveY
    const sPos = [];
    const sNorm = [];
    const sUv = [];
    const addSoffitTrap = (a, b, c, d) => {
      const tris = [a, c, b, a, d, c];
      for (const pt of tris) {
        sPos.push(pt[0], eaveY - 0.12, pt[2]);
        sNorm.push(0, -1, 0);
        sUv.push(pt[0] / 3.0, pt[2] / 3.0);
      }
    };
    addSoffitTrap(oSW, oSE, iSE, iSW);
    addSoffitTrap(oSE, oNE, iNE, iSE);
    addSoffitTrap(oNE, oNW, iNW, iNE);
    addSoffitTrap(oNW, oSW, iSW, iNW);

    const soffitGeo = new THREE.BufferGeometry();
    soffitGeo.setAttribute('position', new THREE.Float32BufferAttribute(sPos, 3));
    soffitGeo.setAttribute('normal', new THREE.Float32BufferAttribute(sNorm, 3));
    soffitGeo.setAttribute('uv', new THREE.Float32BufferAttribute(sUv, 2));
    soffitGeo.computeBoundingBox();
    soffitGeo.computeBoundingSphere();

    const toV3 = (arr) => new THREE.Vector3(arr[0], arr[1] + 0.055, arr[2]);
    const ridgePointsList = [
      // Main top ridge loop (4 segments)
      [toV3(rSW), toV3(rSE)],
      [toV3(rSE), toV3(rNE)],
      [toV3(rNE), toV3(rNW)],
      [toV3(rNW), toV3(rSW)],
      // 4 outer diagonal hip ridges
      [toV3(oSW), toV3(rSW)],
      [toV3(oSE), toV3(rSE)],
      [toV3(oNE), toV3(rNE)],
      [toV3(oNW), toV3(rNW)],
      // 4 inner diagonal valley caps
      [toV3(iSW), toV3(rSW)],
      [toV3(iSE), toV3(rSE)],
      [toV3(iNE), toV3(rNE)],
      [toV3(iNW), toV3(rNW)]
    ];

    return { roofGeo, soffitGeo, ridgePointsList };
  }

  /**
   * Positions a flat plaque/signage mesh cleanly offset by deltaMeters along a curved polar cylinder's outward normal
   * so signage never Z-fights or sinks into curved walls.
   */
  static placeOnPolarCylinder(mesh, radius, thetaRad, y, deltaMeters = 0.025, faceOutward = true) {
    const r = radius + (faceOutward ? deltaMeters : -deltaMeters);
    const c = Math.cos(thetaRad);
    const s = Math.sin(thetaRad);
    mesh.position.set(r * c, y, r * s);
    // Outward normal is (c, 0, s); Three.js PlaneGeometry faces +Z by default, so rotation.y = atan2(c, s)
    const yaw = Math.atan2(c, s) + (faceOutward ? 0 : Math.PI);
    mesh.rotation.set(0, yaw, 0);
    return mesh;
  }
}
