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
   * Builds a watertight 4-wing Courtyard Hip/Limasan Roof (for ITTC Perpustakaan Lama 34m x 34m w/ 14m courtyard).
   * Outer eave square: [-halfOut, +halfOut], Inner courtyard eave square: [-halfIn, +halfIn],
   * Continuous square ridge loop at halfRidge = 0.5*(halfOut + halfIn) at y = ridgeY, eaves at y = eaveY.
   * Includes flat underside soffit and returns { roofGeo, soffitGeo, ridgePointsList }.
   */
  static createSquareCourtyardLimasanRoof(halfOut, halfIn, eaveY, ridgeY, uvTileMeters = 2.0) {
    const halfRidge = 0.5 * (halfOut + halfIn);
    const pos = [];
    const norm = [];
    const uvs = [];
    const vA = new THREE.Vector3();
    const vB = new THREE.Vector3();
    const vC = new THREE.Vector3();
    const e1 = new THREE.Vector3();
    const e2 = new THREE.Vector3();
    const fn = new THREE.Vector3();

    const pushQuad = (p0, p1, p2, p3) => {
      vA.set(p0[0], p0[1], p0[2]);
      vB.set(p1[0], p1[1], p1[2]);
      vC.set(p2[0], p2[1], p2[2]);
      e1.subVectors(vB, vA);
      e2.subVectors(vC, vA);
      fn.crossVectors(e1, e2).normalize();
      if (fn.y < 0) {
        // Flip winding so roof normal points upward (+Y)
        return pushQuad(p0, p3, p2, p1);
      }
      const uLenBottom = Math.hypot(p1[0] - p0[0], p1[2] - p0[2]) / uvTileMeters;
      const uLenTop = Math.hypot(p2[0] - p3[0], p2[2] - p3[2]) / uvTileMeters;
      const slopeLen = Math.hypot(
        0.5 * (p2[0] + p3[0] - p0[0] - p1[0]),
        p2[1] - p0[1],
        0.5 * (p2[2] + p3[2] - p0[2] - p1[2])
      ) / uvTileMeters;
      const uTrim = 0.5 * (uLenBottom - uLenTop);

      const triVerts = [
        [p0, [0, 0]],
        [p1, [uLenBottom, 0]],
        [p2, [uLenBottom - uTrim, slopeLen]],
        [p0, [0, 0]],
        [p2, [uLenBottom - uTrim, slopeLen]],
        [p3, [uTrim, slopeLen]]
      ];
      for (const [pt, uv] of triVerts) {
        pos.push(pt[0], pt[1], pt[2]);
        norm.push(fn.x, fn.y, fn.z);
        uvs.push(uv[0], uv[1]);
      }
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
        sPos.push(pt[0], eaveY - 0.02, pt[2]);
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

    const toV3 = (arr) => new THREE.Vector3(arr[0], arr[1] + 0.04, arr[2]);
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
