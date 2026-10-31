import * as THREE from 'three';
import { TerrainDEMLoader } from './TerrainDEMLoader.js';

/**
 * NEXUS Omni-Scale Procedural Engine v5.0 — Kepulauan UI Edition
 * ./world-gis/DPPDComplexBuilder.js
 *
 * High-Detail Procedural Digital Twin of Kompleks Kantor DPPD (eks-KSDP),
 * Gedung ITTC / Perpustakaan Lama UI, Gedung Science Techno Park (STP-UI 8 Lantai),
 * & Gedung Laboratorium ILRC / DRPM (4 Lantai + Menara Bata Terakota) on Pulau U.
 *
 * Extracted & Perfected from Studio Kelas-A Ground-Truth SSoST:
 * - Scaled (0.58x) and oriented on Pulau U's +4.85m graded terrace (-255.0, +329.0)
 *   so Gedung DPPD's front heritage gable faces Gedung Rektorat UI (0, 0) across Teluk U.
 * - Full night-lit architectural office glazing (UV sentinel v = -720.0) & warm photocell
 *   portico/streetlamp luminaires (UV sentinel v = -890.0).
 * - Batched directly into SplineCorridorBuilder's { horizontalGeo, verticalGeo } pair
 *   for +0 extra WebGL Draw Calls, with exact world-space solid-body colliders.
 */
export class DPPDComplexBuilder {
  static _sliceSubdomain(evaluateFn, uMin, uMax, vMin, vMax, uSegs = 24, vSegs = 12) {
    const vertCount = (uSegs + 1) * (vSegs + 1);
    const positions = new Float32Array(vertCount * 3);
    const normals = new Float32Array(vertCount * 3);
    const uvs = new Float32Array(vertCount * 2);
    const indices = [];
    const p = new THREE.Vector3();
    const n = new THREE.Vector3();
    let vIdx = 0;
    let uvIdx = 0;

    for (let iv = 0; iv <= vSegs; iv++) {
      const v = vMin + (vMax - vMin) * (iv / vSegs);
      for (let iu = 0; iu <= uSegs; iu++) {
        const u = uMin + (uMax - uMin) * (iu / uSegs);
        evaluateFn(u, v, p, n);
        positions[vIdx] = p.x;
        positions[vIdx + 1] = p.y;
        positions[vIdx + 2] = p.z;
        normals[vIdx] = n.x;
        normals[vIdx + 1] = n.y;
        normals[vIdx + 2] = n.z;
        uvs[uvIdx] = u;
        uvs[uvIdx + 1] = v;
        vIdx += 3;
        uvIdx += 2;
      }
    }

    const stride = uSegs + 1;
    const vA = new THREE.Vector3();
    const vB = new THREE.Vector3();
    const vC = new THREE.Vector3();
    const e1 = new THREE.Vector3();
    const e2 = new THREE.Vector3();
    const faceNorm = new THREE.Vector3();
    const refNorm = new THREE.Vector3();

    for (let iv = 0; iv < vSegs; iv++) {
      for (let iu = 0; iu < uSegs; iu++) {
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

        if (faceNorm.dot(refNorm) >= 0) {
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
    return geo;
  }

  static evaluateWingRoofLocalSurface(u, v, targetPos, targetNorm) {
    const alphaMin = (-52.0 * Math.PI) / 180.0;
    const alphaMax = (46.0 * Math.PI) / 180.0;
    const alphaSpan = alphaMax - alphaMin;

    const sectorCount = 6.0;
    const sIdx = Math.floor(Math.min(sectorCount - 0.0001, u * sectorCount));
    const sFrac = u * sectorCount - sIdx;
    const a0 = alphaMin + (sIdx / sectorCount) * alphaSpan;
    const a1 = alphaMin + ((sIdx + 1) / sectorCount) * alphaSpan;

    const rInner = 26.2;
    const rRidge = 35.8;
    const rOuter = 45.9;
    const r = rInner + v * (rOuter - rInner);

    const x0 = r * Math.sin(a0);
    const z0 = r * Math.cos(a0);
    const x1 = r * Math.sin(a1);
    const z1 = r * Math.cos(a1);
    const x = x0 + (x1 - x0) * sFrac;
    const z = z0 + (z1 - z0) * sFrac;

    const distFromRidge = Math.abs(r - rRidge) / ((rOuter - rInner) * 0.5);
    const endHipTaper = Math.min(1.0, Math.min(u, 1.0 - u) * 6.8);
    const eaveY = 6.25;
    const ridgeH = 5.35 * endHipTaper;
    const y = eaveY + Math.max(0.0, (1.0 - distFromRidge) * ridgeH);

    targetPos.set(x, y, z);
    const alphaMid = (a0 + a1) * 0.5;
    const radialSign = r >= rRidge ? 0.38 : -0.38;
    targetNorm.set(Math.sin(alphaMid) * radialSign, 0.92, Math.cos(alphaMid) * radialSign).normalize();
  }

  static evaluateBermLocalSurface(alphaStartDeg, alphaEndDeg, u, v, targetPos, targetNorm) {
    const a0 = (alphaStartDeg * Math.PI) / 180.0;
    const a1 = (alphaEndDeg * Math.PI) / 180.0;
    const alpha = a0 + u * (a1 - a0);

    const rTop = 44.15;
    const rBot = 50.85;
    const r = rTop + v * (rBot - rTop);

    const smoothV = v * v * (3.0 - 2.0 * v);
    const y = 3.68 + (0.32 - 3.68) * smoothV;

    targetPos.set(r * Math.sin(alpha), y, r * Math.cos(alpha));
    targetNorm.set(Math.sin(alpha) * 0.48, 0.88, Math.cos(alpha) * 0.48).normalize();
  }

  /**
   * Builds a watertight extruded 3D prism along +Z (z in [0, depth]) from a 2D convex/concave polygon
   * (compatible with both real Three.js WebGL2 and headless verification shims).
   */
  static _buildPolygonPrismGeo(pts2D, depth) {
    const n = pts2D.length;
    // Signed area to ensure CCW winding on +Z front face
    let area2 = 0.0;
    for (let i = 0; i < n; i++) {
      const p0 = pts2D[i];
      const p1 = pts2D[(i + 1) % n];
      area2 += p0[0] * p1[1] - p1[0] * p0[1];
    }
    const ordered = area2 >= 0 ? pts2D : [...pts2D].reverse();

    const pos = [];
    const norm = [];
    const uvs = [];
    const idx = [];
    let vCount = 0;

    // Front face (z = depth, normal +Z) & Back face (z = 0, normal -Z) via fan from centroid
    let cx = 0.0;
    let cy = 0.0;
    for (let i = 0; i < n; i++) {
      cx += ordered[i][0];
      cy += ordered[i][1];
    }
    cx /= n;
    cy /= n;

    // Front (+Z)
    const fCenter = vCount++;
    pos.push(cx, cy, depth);
    norm.push(0, 0, 1);
    uvs.push(0.5, 0.5);
    for (let i = 0; i < n; i++) {
      pos.push(ordered[i][0], ordered[i][1], depth);
      norm.push(0, 0, 1);
      uvs.push(0.5, 0.5);
      vCount++;
    }
    for (let i = 0; i < n; i++) {
      idx.push(fCenter, fCenter + 1 + i, fCenter + 1 + ((i + 1) % n));
    }

    // Back (-Z)
    const bCenter = vCount++;
    pos.push(cx, cy, 0.0);
    norm.push(0, 0, -1);
    uvs.push(0.5, 0.5);
    for (let i = 0; i < n; i++) {
      pos.push(ordered[i][0], ordered[i][1], 0.0);
      norm.push(0, 0, -1);
      uvs.push(0.5, 0.5);
      vCount++;
    }
    for (let i = 0; i < n; i++) {
      idx.push(bCenter, bCenter + 1 + ((i + 1) % n), bCenter + 1 + i);
    }

    // Side quads around perimeter
    for (let i = 0; i < n; i++) {
      const p0 = ordered[i];
      const p1 = ordered[(i + 1) % n];
      const ex = p1[0] - p0[0];
      const ey = p1[1] - p0[1];
      const el = Math.hypot(ex, ey) || 1.0;
      const nx = ey / el;
      const ny = -ex / el;

      const base = vCount;
      pos.push(
        p0[0], p0[1], 0.0,
        p1[0], p1[1], 0.0,
        p1[0], p1[1], depth,
        p0[0], p0[1], depth
      );
      for (let k = 0; k < 4; k++) {
        norm.push(nx, ny, 0.0);
        uvs.push(k === 1 || k === 2 ? 1.0 : 0.0, k >= 2 ? 1.0 : 0.0);
      }
      idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
      vCount += 4;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(norm), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    geo.setIndex(idx);
    return geo;
  }

  /**
   * Orients a Z-aligned mesh (length along local Z) to span from pA to pB with optional Y lift.
   */
  static _orientBeamAlongZ(mesh, pA, pB, liftY = 0.0) {
    mesh.position.set(
      (pA.x + pB.x) * 0.5,
      (pA.y + pB.y) * 0.5 + liftY,
      (pA.z + pB.z) * 0.5
    );
    const dx = pB.x - pA.x;
    const dy = pB.y - pA.y;
    const dz = pB.z - pA.z;
    const len = Math.hypot(dx, dy, dz) || 1.0;
    const yaw = Math.atan2(dx, dz);
    const pitch = -Math.asin(Math.max(-1.0, Math.min(1.0, dy / len)));
    mesh.rotation.set(pitch, yaw, 0.0, 'YXZ');
  }

  static _createPalette() {
    const mk = (hex, kind = 'H', uvSentinel = null) => {
      const c = new THREE.Color(hex);
      const m = new THREE.MeshBasicMaterial({ color: c });
      m.userData.batchKind = kind;
      m.userData.rgb = [c.r, c.g, c.b];
      m.userData.uvSentinel = uvSentinel;
      return m;
    };

    return {
      terracottaRoof: mk(0x8c3322),
      terracottaRidge: mk(0x692215),
      creamWall: mk(0xefe9d8),
      concreteTrim: mk(0x8b867e),
      darkWoodFascia: mk(0x3b2318),
      grassBerm: mk(0x4f7836),
      LushShrub: mk(0x356221),
      stoneRetaining: mk(0x54585d),
      terracottaFloor: mk(0xa64432),
      windowGlass: mk(0x132030, 'H', [0.5, -720.0]),
      warmLuminaire: mk(0xfff1c4, 'H', [0.5, -890.0]),
      whiteMullion: mk(0xf4f6f9),
      disabilityBlue: mk(0x2563eb),
      brassSign: mk(0xd8b137),
      asphaltRoad: mk(0x2d3138),
      curbStriped: mk(0xf5f7fa, 'Z'),
      roadYellowLine: mk(0xf5b81d),
      ilrcConcrete: mk(0xd6d3cd),
      ilrcBrickWing: mk(0x8e5241),
      solarPanel: mk(0x162756),
      yellowMetal: mk(0xd99b07),
      concreteDriveway: mk(0xd8d4cc),
      pavingBlock: mk(0x5c5752),
      safetyOrange: mk(0xea580c),
      salmonGable: mk(0xd07c68),
      mulchSoil: mk(0x6e503b),
      carWhite: mk(0xf1f5f9),
      carSilver: mk(0xcbd5e1),
      carDarkGrey: mk(0x333842),
      carNavy: mk(0x1e3a5f),
      hydrantRed: mk(0xdc2626),
      purpleSign: mk(0x453263),
      copperACP: mk(0xc68b59)
    };
  }

  static _buildParkedCar(mats, bodyMat, x, y, z, rotY) {
    const g = new THREE.Group();
    g.position.set(x, y, z);
    g.rotation.y = rotY;

    // Lower sculpted body & bumpers
    const lowerBody = new THREE.Mesh(new THREE.BoxGeometry(1.82, 0.56, 4.35), bodyMat);
    lowerBody.position.set(0, 0.50, 0);
    g.add(lowerBody);

    const frontBumper = new THREE.Mesh(new THREE.BoxGeometry(1.84, 0.28, 0.24), mats.asphaltRoad);
    frontBumper.position.set(0, 0.34, 2.15);
    g.add(frontBumper);

    const rearBumper = new THREE.Mesh(new THREE.BoxGeometry(1.84, 0.30, 0.24), mats.asphaltRoad);
    rearBumper.position.set(0, 0.35, -2.15);
    g.add(rearBumper);

    // Greenhouse cabin + night-reflective glass
    const cabinRoof = new THREE.Mesh(new THREE.BoxGeometry(1.52, 0.10, 2.05), bodyMat);
    cabinRoof.position.set(0, 1.32, -0.18);
    g.add(cabinRoof);

    const cabinGlass = new THREE.Mesh(new THREE.BoxGeometry(1.56, 0.52, 2.18), mats.windowGlass);
    cabinGlass.position.set(0, 1.02, -0.16);
    g.add(cabinGlass);

    // Pillars & side mirrors
    for (const sx of [-0.77, 0.77]) {
      const bPillar = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.54, 0.14), mats.asphaltRoad);
      bPillar.position.set(sx, 1.02, -0.12);
      g.add(bPillar);

      const mirror = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.11, 0.12), bodyMat);
      mirror.position.set(sx + Math.sign(sx) * 0.11, 0.86, 0.72);
      g.add(mirror);
    }

    // Headlamps, grille, & taillamps
    const grille = new THREE.Mesh(new THREE.BoxGeometry(1.06, 0.22, 0.08), mats.asphaltRoad);
    grille.position.set(0, 0.56, 2.16);
    g.add(grille);

    for (const sx of [-0.62, 0.62]) {
      const headLamp = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.14, 0.08), mats.whiteMullion);
      headLamp.position.set(sx, 0.63, 2.16);
      g.add(headLamp);

      const tailLamp = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.14, 0.08), mats.hydrantRed);
      tailLamp.position.set(sx, 0.66, -2.16);
      g.add(tailLamp);
    }

    // 4 tires + alloy rims
    const tireGeo = new THREE.CylinderGeometry(0.32, 0.32, 0.22, 14);
    tireGeo.rotateZ(Math.PI * 0.5);
    const rimGeo = new THREE.CylinderGeometry(0.19, 0.19, 0.23, 10);
    rimGeo.rotateZ(Math.PI * 0.5);
    for (const sx of [-0.84, 0.84]) {
      for (const sz of [-1.32, 1.34]) {
        const tire = new THREE.Mesh(tireGeo, mats.asphaltRoad);
        tire.position.set(sx, 0.32, sz);
        g.add(tire);

        const rim = new THREE.Mesh(rimGeo, mats.carSilver);
        rim.position.set(sx, 0.32, sz);
        g.add(rim);
      }
    }
    return g;
  }

  static _buildParkedMotorbike(mats, bodyMat, x, y, z, rotY) {
    const g = new THREE.Group();
    g.position.set(x, y, z);
    g.rotation.y = rotY;

    const wheelGeo = new THREE.CylinderGeometry(0.26, 0.26, 0.10, 12);
    wheelGeo.rotateZ(Math.PI * 0.5);
    for (const sz of [-0.62, 0.62]) {
      const w = new THREE.Mesh(wheelGeo, mats.asphaltRoad);
      w.position.set(0, 0.26, sz);
      g.add(w);
    }
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.42, 1.15), bodyMat);
    body.position.set(0, 0.56, -0.05);
    g.add(body);

    const seat = new THREE.Mesh(new THREE.BoxGeometry(0.30, 0.12, 0.68), mats.asphaltRoad);
    seat.position.set(0, 0.80, -0.22);
    g.add(seat);

    const fairing = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.55, 0.28), bodyMat);
    fairing.position.set(0, 0.78, 0.48);
    fairing.rotation.x = -0.24;
    g.add(fairing);

    const handleBar = new THREE.Mesh(new THREE.BoxGeometry(0.68, 0.05, 0.05), mats.concreteTrim);
    handleBar.position.set(0, 1.04, 0.42);
    g.add(handleBar);
    return g;
  }

  /**
   * Builds and bakes the entire DPPD-ITTC-STP-ILRC architectural complex into
   * { horizontalGeo, verticalGeo } ready for SplineCorridorBuilder.buildRotundaAndPromenades().
   */
  static buildForCorridor() {
    const mats = DPPDComplexBuilder._createPalette();
    const root = new THREE.Group();
    root.position.set(
      TerrainDEMLoader.DPPD_CX,
      TerrainDEMLoader.DPPD_PLATEAU_Y,
      TerrainDEMLoader.DPPD_CZ
    );
    root.rotation.y = TerrainDEMLoader.DPPD_Y_ROT;
    root.scale.set(
      TerrainDEMLoader.DPPD_SCALE,
      TerrainDEMLoader.DPPD_SCALE,
      TerrainDEMLoader.DPPD_SCALE
    );

    const dppdGroup = new THREE.Group();
    const contextGroup = new THREE.Group();
    root.add(dppdGroup, contextGroup);

    const entranceGroup = new THREE.Group();
    entranceGroup.rotation.y = -0.25 * Math.PI;
    dppdGroup.add(entranceGroup);

    // 1. Watertight Campus Terrace Plinth, Courtyard Pad, & 360° Ring Road (r <= 65.15 unscaled = 37.79m world)
    const campusLawnPad = new THREE.Mesh(
      new THREE.CylinderGeometry(53.05, 53.25, 0.90, 72),
      mats.grassBerm
    );
    campusLawnPad.position.y = -0.43;
    dppdGroup.add(campusLawnPad);

    const outerFoundationSkirt = new THREE.Mesh(
      new THREE.CylinderGeometry(65.12, 66.20, 3.60, 72),
      mats.stoneRetaining
    );
    outerFoundationSkirt.position.y = -1.78;
    dppdGroup.add(outerFoundationSkirt);

    const courtyardPad = new THREE.Mesh(
      new THREE.CylinderGeometry(31.0, 31.0, 0.14, 48),
      mats.pavingBlock
    );
    courtyardPad.position.y = 0.01;
    dppdGroup.add(courtyardPad);

    const ringRoadGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = u * Math.PI * 2.0;
        const r = 53.2 + v * (64.8 - 53.2);
        p.set(r * Math.cos(th), 0.08, r * Math.sin(th));
        n.set(0, 1, 0);
      },
      0.0, 1.0, 0.0, 1.0, 80, 2
    );
    dppdGroup.add(new THREE.Mesh(ringRoadGeo, mats.asphaltRoad));

    // Smooth curb-cut factor [0..1] dropping outer ring-road curb flush across South & East Pulau U Spur Mouths!
    const evalOuterMouthCurbFactor = (rLocal, th) => {
      const wp = DPPDComplexBuilder.localToWorld(rLocal * Math.cos(th), 0.0, rLocal * Math.sin(th), 0.0);
      const dSouth = Math.hypot(wp.x - -291.7, wp.z - 336.7);
      const dEast = Math.hypot(wp.x - -218.5, wp.z - 338.3);
      const dMin = Math.min(dSouth, dEast);
      const t = Math.max(0.0, Math.min(1.0, (dMin - 4.4) / 2.2));
      return t * t * (3.0 - 2.0 * t);
    };

    // Inner & Outer Striped Zebra Curbs (with top & vertical faces, UV u scaled for zebra striping!)
    const buildRingZebraCurb = (rIn, rOut, yTop, cutSpurMouths = false) => {
      const segs = 128;
      const circum = Math.PI * (rIn + rOut) * TerrainDEMLoader.DPPD_SCALE;
      const zebraRepeats = Math.max(12, Math.round(circum * 0.26));
      const evalYTop = (th) => {
        if (!cutSpurMouths) return yTop;
        const cf = evalOuterMouthCurbFactor(0.5 * (rIn + rOut), th);
        return 0.082 + (yTop - 0.082) * cf;
      };

      const topGeo = DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const th = u * Math.PI * 2.0;
          const r = rIn + v * (rOut - rIn);
          p.set(r * Math.cos(th), evalYTop(th), r * Math.sin(th));
          n.set(0, 1, 0);
        },
        0.0, 1.0, 0.0, 1.0, segs, 1
      );
      const innerFaceGeo = DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const th = u * Math.PI * 2.0;
          const yt = evalYTop(th);
          p.set(rIn * Math.cos(th), 0.08 * (1.0 - v) + v * yt, rIn * Math.sin(th));
          n.set(-Math.cos(th), 0, -Math.sin(th));
        },
        0.0, 1.0, 0.0, 1.0, segs, 1
      );
      const outerFaceGeo = DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const th = u * Math.PI * 2.0;
          const yt = evalYTop(th);
          p.set(rOut * Math.cos(th), -0.35 + v * (yt + 0.35), rOut * Math.sin(th));
          n.set(Math.cos(th), 0, Math.sin(th));
        },
        0.0, 1.0, 0.0, 1.0, segs, 1
      );

      for (const g of [topGeo, innerFaceGeo, outerFaceGeo]) {
        const uv = g.attributes.uv;
        for (let i = 0; i < uv.count; i++) {
          uv.array[i * uv.itemSize] = uv.getX(i) * zebraRepeats;
        }
        const m = new THREE.Mesh(g, mats.curbStriped);
        m.userData.keepCurbUV = true;
        dppdGroup.add(m);
      }
    };

    buildRingZebraCurb(52.86, 53.24, 0.21, false);
    buildRingZebraCurb(64.78, 65.16, 0.21, true);

    const yellowLineInnerGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = u * Math.PI * 2.0;
        const r = 53.45 + v * 0.18;
        p.set(r * Math.cos(th), 0.098, r * Math.sin(th));
        n.set(0, 1, 0);
      },
      0.0, 1.0, 0.0, 1.0, 96, 1
    );
    dppdGroup.add(new THREE.Mesh(yellowLineInnerGeo, mats.roadYellowLine));

    const yellowLineOuterGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = u * Math.PI * 2.0;
        const r = 64.38 + v * 0.18;
        p.set(r * Math.cos(th), 0.098, r * Math.sin(th));
        n.set(0, 1, 0);
      },
      0.0, 1.0, 0.0, 1.0, 96, 1
    );
    dppdGroup.add(new THREE.Mesh(yellowLineOuterGeo, mats.roadYellowLine));

    for (let d = 0; d < 48; d++) {
      const th = (d / 48) * Math.PI * 2.0;
      const dash = new THREE.Mesh(new THREE.BoxGeometry(0.20, 0.04, 2.4), mats.whiteMullion);
      dash.position.set(59.0 * Math.cos(th), 0.098, 59.0 * Math.sin(th));
      dash.rotation.y = -th;
      dppdGroup.add(dash);
    }

    // Zebra Crosswalks across the DPPD Ring Road in front of Gedung DPPD & STP-UI North Portico
    for (const crossDeg of [45.0, -79.5, 19.5]) {
      const thC = (crossDeg * Math.PI) / 180.0;
      for (let st = -4; st <= 4; st++) {
        const rSt = 59.0 + st * 1.12;
        const stripe = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.042, 3.2), mats.whiteMullion);
        stripe.position.set(rSt * Math.cos(thC), 0.099, rSt * Math.sin(thC));
        stripe.rotation.y = -thC;
        dppdGroup.add(stripe);
      }
    }

    // 2. TAHAP 3: SSoST BUSUR SAYAP GEDUNG DPPD + GARIS BUMBUNGAN JURING & PENANGKAL PETIR
    const wingRoofGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => DPPDComplexBuilder.evaluateWingRoofLocalSurface(u, v, p, n),
      0.0, 1.0, 0.0, 1.0, 60, 20
    );
    entranceGroup.add(new THREE.Mesh(wingRoofGeo, mats.terracottaRoof));

    const alphaMin = (-52.0 * Math.PI) / 180.0;
    const alphaMax = (46.0 * Math.PI) / 180.0;
    const sectorCount = 6;
    const pStart = new THREE.Vector3();
    const pEnd = new THREE.Vector3();
    const nDummy = new THREE.Vector3();

    // Main curved roof spine ridge cap along v = 0.49
    for (let s = 0; s < sectorCount; s++) {
      const u0 = s / sectorCount;
      const u1 = (s + 1) / sectorCount;
      DPPDComplexBuilder.evaluateWingRoofLocalSurface(u0, 0.49, pStart, nDummy);
      DPPDComplexBuilder.evaluateWingRoofLocalSurface(u1, 0.49, pEnd, nDummy);
      const spineLen = pStart.distanceTo(pEnd);
      const spineCap = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.24, spineLen + 0.12), mats.terracottaRidge);
      DPPDComplexBuilder._orientBeamAlongZ(spineCap, pStart, pEnd, 0.08);
      entranceGroup.add(spineCap);
    }

    for (let s = 0; s <= sectorCount; s++) {
      const uSeam = s / sectorCount;
      DPPDComplexBuilder.evaluateWingRoofLocalSurface(uSeam, 0.49, pStart, nDummy);
      DPPDComplexBuilder.evaluateWingRoofLocalSurface(uSeam, 1.0, pEnd, nDummy);
      const ridgeLen = pStart.distanceTo(pEnd);
      const hipRidge = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.18, ridgeLen), mats.darkWoodFascia);
      DPPDComplexBuilder._orientBeamAlongZ(hipRidge, pStart, pEnd, 0.06);
      entranceGroup.add(hipRidge);

      if (s > 0 && s < sectorCount) {
        const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.035, 1.45, 6), mats.whiteMullion);
        rod.position.set(pStart.x, pStart.y + 0.72, pStart.z);
        entranceGroup.add(rod);
      }
    }

    for (let s = 0; s < sectorCount; s++) {
      const u0 = s / sectorCount;
      const u1 = (s + 1) / sectorCount;
      DPPDComplexBuilder.evaluateWingRoofLocalSurface(u0, 1.0, pStart, nDummy);
      DPPDComplexBuilder.evaluateWingRoofLocalSurface(u1, 1.0, pEnd, nDummy);
      const eaveLen = pStart.distanceTo(pEnd);
      const eaveBoard = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.28, eaveLen + 0.15), mats.darkWoodFascia);
      DPPDComplexBuilder._orientBeamAlongZ(eaveBoard, pStart, pEnd, 0.0);
      entranceGroup.add(eaveBoard);
    }

    const evalPolarCylinderBand = (rVal, yMin, yMax, aStart = alphaMin, aEnd = alphaMax) =>
      DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const a = aStart + u * (aEnd - aStart);
          p.set(rVal * Math.sin(a), yMin + v * (yMax - yMin), rVal * Math.cos(a));
          n.set(Math.sin(a), 0, Math.cos(a));
        },
        0.0, 1.0, 0.0, 1.0, 48, 4
      );

    entranceGroup.add(
      new THREE.Mesh(
        DPPDComplexBuilder._sliceSubdomain(
          (u, v, p, n) => {
            const a = alphaMin + u * (alphaMax - alphaMin);
            p.set(27.5 * Math.sin(a), v * 6.25, 27.5 * Math.cos(a));
            n.set(-Math.sin(a), 0, -Math.cos(a));
          },
          0.0, 1.0, 0.0, 1.0, 48, 4
        ),
        mats.creamWall
      )
    );
    entranceGroup.add(new THREE.Mesh(evalPolarCylinderBand(43.35, 0.0, 4.10), mats.creamWall));

    const wingSoffitGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const a = alphaMin + u * (alphaMax - alphaMin);
        const r = 26.2 + v * (45.9 - 26.2);
        p.set(r * Math.sin(a), 6.24, r * Math.cos(a));
        n.set(0, -1, 0);
      },
      0.0, 1.0, 0.0, 1.0, 48, 4
    );
    entranceGroup.add(new THREE.Mesh(wingSoffitGeo, mats.creamWall));
    entranceGroup.add(new THREE.Mesh(evalPolarCylinderBand(43.45, 4.05, 6.25), mats.windowGlass));

    const sillTopGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const a = alphaMin + u * (alphaMax - alphaMin);
        const r = 43.3 + v * 1.15;
        p.set(r * Math.sin(a), 4.06, r * Math.cos(a));
        n.set(0, 1, 0);
      },
      0.0, 1.0, 0.0, 1.0, 48, 2
    );
    entranceGroup.add(new THREE.Mesh(sillTopGeo, mats.creamWall));
    entranceGroup.add(new THREE.Mesh(evalPolarCylinderBand(44.42, 3.62, 4.06), mats.creamWall));

    const mullionGeo = new THREE.BoxGeometry(0.24, 2.22, 0.32);
    const acBoxGeo = new THREE.BoxGeometry(0.78, 0.54, 0.36);
    const acFanGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.38, 10);
    acFanGeo.rotateX(Math.PI * 0.5);

    const wingCols = 36;
    for (let i = 0; i <= wingCols; i++) {
      const a = alphaMin + (i / wingCols) * (alphaMax - alphaMin);
      if (Math.abs(a) < (11.8 * Math.PI) / 180.0) continue;
      const mCol = new THREE.Mesh(mullionGeo, mats.whiteMullion);
      mCol.position.set(44.18 * Math.sin(a), 5.16, 44.18 * Math.cos(a));
      mCol.rotation.y = a;
      entranceGroup.add(mCol);

      if (i < wingCols && i % 2 === 1) {
        const aMid = a + (0.5 / wingCols) * (alphaMax - alphaMin);
        const acUnit = new THREE.Mesh(acBoxGeo, mats.whiteMullion);
        acUnit.position.set(44.05 * Math.sin(aMid), 4.35, 44.05 * Math.cos(aMid));
        acUnit.rotation.y = aMid;
        entranceGroup.add(acUnit);

        const acFan = new THREE.Mesh(acFanGeo, mats.asphaltRoad);
        acFan.position.set(44.06 * Math.sin(aMid), 4.35, 44.06 * Math.cos(aMid));
        acFan.rotation.y = aMid;
        entranceGroup.add(acFan);
      }
    }

    // 3. TAHAP 4: FASAD DEPAN GEDUNG DPPD (MENGHADAP REKTORAT UI!)
    const gHalfW = 10.6;
    const gBaseY = 6.22;
    const gPeakY = 15.45;
    const zGableBack = 35.8;
    const zGableWall = 44.5;
    const zGableFront = 45.85;

    const tympanumGeo = DPPDComplexBuilder._buildPolygonPrismGeo(
      [
        [-gHalfW + 0.35, gBaseY],
        [gHalfW - 0.35, gBaseY],
        [0.0, gPeakY - 0.32]
      ],
      8.6
    );
    const tympanumMesh = new THREE.Mesh(tympanumGeo, mats.creamWall);
    tympanumMesh.position.set(0, 0, zGableWall - 8.6);
    entranceGroup.add(tympanumMesh);

    // Golden UI Makara Emblem on DPPD Central Gable Tympanum facing Rektorat UI!
    const dppdMakaraBack = new THREE.Mesh(new THREE.CylinderGeometry(1.12, 1.12, 0.14, 18), mats.darkWoodFascia);
    dppdMakaraBack.rotation.x = Math.PI * 0.5;
    dppdMakaraBack.position.set(0, 9.85, zGableWall + 0.08);
    entranceGroup.add(dppdMakaraBack);

    const dppdMakaraGold = new THREE.Mesh(new THREE.CylinderGeometry(0.92, 0.92, 0.18, 16), mats.brassSign);
    dppdMakaraGold.rotation.x = Math.PI * 0.5;
    dppdMakaraGold.position.set(0, 9.85, zGableWall + 0.12);
    entranceGroup.add(dppdMakaraGold);

    const roofSlopeLen = Math.hypot(gHalfW + 0.45, gPeakY - gBaseY);
    const roofSlopeAngle = Math.atan2(gPeakY - gBaseY, gHalfW);
    const gableDepth = zGableFront - zGableBack;
    const gableMidZ = (zGableFront + zGableBack) * 0.5;
    const midSlopeX = gHalfW * 0.5;
    const midSlopeY = (gBaseY + gPeakY) * 0.5;

    const leftGableRoof = new THREE.Mesh(new THREE.BoxGeometry(roofSlopeLen, 0.28, gableDepth), mats.terracottaRoof);
    leftGableRoof.position.set(-midSlopeX, midSlopeY + 0.12, gableMidZ);
    leftGableRoof.rotation.z = roofSlopeAngle;
    entranceGroup.add(leftGableRoof);

    const rightGableRoof = new THREE.Mesh(new THREE.BoxGeometry(roofSlopeLen, 0.28, gableDepth), mats.terracottaRoof);
    rightGableRoof.position.set(midSlopeX, midSlopeY + 0.12, gableMidZ);
    rightGableRoof.rotation.z = -roofSlopeAngle;
    entranceGroup.add(rightGableRoof);

    const gableRidgeCap = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.32, gableDepth + 0.18), mats.terracottaRidge);
    gableRidgeCap.position.set(0, gPeakY + 0.14, gableMidZ);
    entranceGroup.add(gableRidgeCap);

    const fasciaGeo = new THREE.BoxGeometry(roofSlopeLen + 0.2, 0.36, 0.22);
    const leftFascia = new THREE.Mesh(fasciaGeo, mats.darkWoodFascia);
    leftFascia.position.set(-midSlopeX, midSlopeY - 0.04, zGableFront);
    leftFascia.rotation.z = roofSlopeAngle;
    entranceGroup.add(leftFascia);

    const rightFascia = new THREE.Mesh(fasciaGeo, mats.darkWoodFascia);
    rightFascia.position.set(midSlopeX, midSlopeY - 0.04, zGableFront);
    rightFascia.rotation.z = -roofSlopeAngle;
    entranceGroup.add(rightFascia);

    const peakLightningRod = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.04, 1.6, 8), mats.whiteMullion);
    peakLightningRod.position.set(0, gPeakY + 0.75, zGableFront - 0.4);
    entranceGroup.add(peakLightningRod);

    const apexLamp = new THREE.Mesh(new THREE.SphereGeometry(0.24, 12, 8), mats.warmLuminaire);
    apexLamp.position.set(0, gPeakY - 0.85, zGableWall + 0.42);
    entranceGroup.add(apexLamp);

    const centerWinGlass = new THREE.Mesh(new THREE.BoxGeometry(19.0, 2.12, 0.35), mats.windowGlass);
    centerWinGlass.position.set(0, 5.18, 44.32);
    entranceGroup.add(centerWinGlass);

    const centerWinSill = new THREE.Mesh(new THREE.BoxGeometry(20.6, 0.38, 0.85), mats.creamWall);
    centerWinSill.position.set(0, 4.02, 44.45);
    entranceGroup.add(centerWinSill);

    for (let w = 0; w <= 10; w++) {
      const xPos = -9.5 + w * (19.0 / 10);
      const mBar = new THREE.Mesh(new THREE.BoxGeometry(0.22, 2.18, 0.48), mats.whiteMullion);
      mBar.position.set(xPos, 5.18, 44.38);
      entranceGroup.add(mBar);
    }

    const miniDish = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.12, 0.14, 12), mats.whiteMullion);
    miniDish.position.set(-6.15, 4.48, 44.62);
    miniDish.rotation.x = Math.PI * 0.42;
    miniDish.rotation.z = -0.25;
    entranceGroup.add(miniDish);

    const gableRightAC = new THREE.Mesh(new THREE.BoxGeometry(0.82, 0.56, 0.38), mats.whiteMullion);
    gableRightAC.position.set(7.25, 4.38, 44.52);
    entranceGroup.add(gableRightAC);

    // Front Grass Berms, Terraces, & Wing End Gables
    const leftBermGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => DPPDComplexBuilder.evaluateBermLocalSurface(-50.0, -7.8, u, v, p, n),
      0.0, 1.0, 0.0, 1.0, 32, 10
    );
    entranceGroup.add(new THREE.Mesh(leftBermGeo, mats.grassBerm));

    entranceGroup.add(
      new THREE.Mesh(
        evalPolarCylinderBand(48.85, 0.45, 1.18, (-46.0 * Math.PI) / 180.0, (-12.5 * Math.PI) / 180.0),
        mats.stoneRetaining
      )
    );

    const leftShrub = new THREE.Mesh(new THREE.SphereGeometry(0.62, 8, 6), mats.LushShrub);
    leftShrub.position.set(-10.2, 0.85, 48.2);
    entranceGroup.add(leftShrub);

    entranceGroup.add(
      new THREE.Mesh(
        evalPolarCylinderBand(44.12, 4.05, 6.22, (-51.5 * Math.PI) / 180.0, (-41.5 * Math.PI) / 180.0),
        mats.creamWall
      )
    );

    for (let aw = 0; aw < 3; aw++) {
      const aWin = ((-49.2 + aw * 2.8) * Math.PI) / 180.0;
      const awningPane = new THREE.Mesh(new THREE.BoxGeometry(1.45, 0.68, 0.28), mats.windowGlass);
      awningPane.position.set(44.28 * Math.sin(aWin), 5.65, 44.28 * Math.cos(aWin));
      awningPane.rotation.y = aWin;
      awningPane.rotation.x = -0.22;
      entranceGroup.add(awningPane);
    }

    for (let sb = 0; sb < 9; sb++) {
      const aBlk = ((-48.0 + sb * 4.5) * Math.PI) / 180.0;
      const sillBlock = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.46, 0.65), mats.creamWall);
      sillBlock.position.set(44.55 * Math.sin(aBlk), 3.82, 44.55 * Math.cos(aBlk));
      sillBlock.rotation.y = aBlk;
      entranceGroup.add(sillBlock);
    }

    const farLeftWallGroup = new THREE.Group();
    farLeftWallGroup.rotation.y = (-51.2 * Math.PI) / 180.0;
    const farLeftExtrude = DPPDComplexBuilder._buildPolygonPrismGeo(
      [
        [44.0, 0.0],
        [51.2, 0.0],
        [51.2, 0.72],
        [44.0, 3.85]
      ],
      0.46
    );
    farLeftExtrude.rotateY(-Math.PI * 0.5);
    farLeftWallGroup.add(new THREE.Mesh(farLeftExtrude, mats.stoneRetaining));

    const nwEndWallExtrude = DPPDComplexBuilder._buildPolygonPrismGeo(
      [
        [27.5, 0.0],
        [44.2, 0.0],
        [44.2, 6.25],
        [35.8, 10.65],
        [27.5, 6.25]
      ],
      0.42
    );
    nwEndWallExtrude.rotateY(-Math.PI * 0.5);
    const nwEndWallMesh = new THREE.Mesh(nwEndWallExtrude, mats.ilrcBrickWing);
    nwEndWallMesh.position.x = -0.35;
    farLeftWallGroup.add(nwEndWallMesh);

    const nwBeamHoriz = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.34, 16.8), mats.whiteMullion);
    nwBeamHoriz.position.set(-0.38, 6.18, 35.85);
    farLeftWallGroup.add(nwBeamHoriz);

    const nwBeamLow = new THREE.Mesh(new THREE.BoxGeometry(0.50, 0.28, 16.8), mats.whiteMullion);
    nwBeamLow.position.set(-0.38, 3.65, 35.85);
    farLeftWallGroup.add(nwBeamLow);

    const nwYellowPillar = new THREE.Mesh(new THREE.BoxGeometry(0.65, 3.65, 0.85), mats.brassSign);
    nwYellowPillar.position.set(-0.42, 1.82, 41.2);
    farLeftWallGroup.add(nwYellowPillar);

    const nwSideDoorRecess = new THREE.Mesh(new THREE.BoxGeometry(0.65, 3.55, 2.85), mats.creamWall);
    nwSideDoorRecess.position.set(-0.32, 1.78, 42.85);
    farLeftWallGroup.add(nwSideDoorRecess);

    const nwSideDoubleDoor = new THREE.Mesh(new THREE.BoxGeometry(0.72, 2.35, 1.55), mats.darkWoodFascia);
    nwSideDoubleDoor.position.set(-0.35, 1.22, 42.85);
    farLeftWallGroup.add(nwSideDoubleDoor);

    const sideDoorWalkway = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.18, 7.5), mats.concreteDriveway);
    sideDoorWalkway.position.set(-1.15, 0.12, 45.8);
    farLeftWallGroup.add(sideDoorWalkway);

    for (let hg = 0; hg < 9; hg++) {
      const hBush = new THREE.Mesh(new THREE.SphereGeometry(0.68, 8, 6), mats.LushShrub);
      hBush.scale.set(0.95, 1.15, 1.25);
      hBush.position.set(-2.15, 0.58, 38.5 + hg * 1.45);
      farLeftWallGroup.add(hBush);
    }
    entranceGroup.add(farLeftWallGroup);

    const blackMonolithSign = new THREE.Mesh(new THREE.BoxGeometry(0.95, 2.35, 0.28), mats.asphaltRoad);
    blackMonolithSign.position.set(-36.5, 1.25, 51.8);
    blackMonolithSign.rotation.y = -0.45;
    entranceGroup.add(blackMonolithSign);

    const quoteMonolith = new THREE.Mesh(new THREE.BoxGeometry(1.35, 2.45, 0.32), mats.asphaltRoad);
    quoteMonolith.position.set(-39.6, 1.35, 48.2);
    quoteMonolith.rotation.y = 0.65;
    entranceGroup.add(quoteMonolith);

    const quoteGoldAccent = new THREE.Mesh(new THREE.BoxGeometry(1.38, 0.28, 0.34), mats.brassSign);
    quoteGoldAccent.position.set(-39.6, 0.35, 48.2);
    quoteGoldAccent.rotation.y = 0.65;
    entranceGroup.add(quoteGoldAccent);

    // Garden Lamps with Warm Photocell Bulbs
    for (const deg of [-17.5, -28.5, -39.5, 21.5, 33.8, 41.2]) {
      const aLamp = (deg * Math.PI) / 180.0;
      const rLamp = 50.15;
      const lx = rLamp * Math.sin(aLamp);
      const lz = rLamp * Math.cos(aLamp);
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 2.35, 8), mats.asphaltRoad);
      post.position.set(lx, 1.32, lz);
      entranceGroup.add(post);

      const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.11, 8, 6), mats.warmLuminaire);
      bulb.position.set(lx, 2.42, lz);
      entranceGroup.add(bulb);

      const cap = new THREE.Mesh(new THREE.ConeGeometry(0.18, 0.14, 8), mats.asphaltRoad);
      cap.position.set(lx, 2.54, lz);
      entranceGroup.add(cap);
    }

    for (const brDeg of [-13.5, -26.0, -38.5]) {
      const aBr = (brDeg * Math.PI) / 180.0;
      for (let s = -1; s <= 1; s += 2) {
        const slab = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.09, 1.15), mats.concreteDriveway);
        const aOff = aBr + s * 0.0045;
        slab.position.set(51.3 * Math.sin(aOff), 0.26, 51.3 * Math.cos(aOff));
        slab.rotation.y = aBr;
        entranceGroup.add(slab);
      }
    }

    const signAngle = (-11.2 * Math.PI) / 180.0;
    const signPost = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 2.55, 8), mats.concreteTrim);
    signPost.position.set(52.35 * Math.sin(signAngle), 1.35, 52.35 * Math.cos(signAngle));
    entranceGroup.add(signPost);

    const blueSignDisk = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.05, 16), mats.disabilityBlue);
    blueSignDisk.rotation.z = Math.PI * 0.5;
    blueSignDisk.position.set(52.35 * Math.sin(signAngle), 2.35, 52.35 * Math.cos(signAngle));
    blueSignDisk.rotation.y = signAngle;
    entranceGroup.add(blueSignDisk);

    const stpArrowSign = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.65, 0.14), mats.disabilityBlue);
    stpArrowSign.position.set(51.5 * Math.sin((-54.5 * Math.PI) / 180.0), 0.95, 51.5 * Math.cos((-54.5 * Math.PI) / 180.0));
    stpArrowSign.rotation.y = -0.85;
    entranceGroup.add(stpArrowSign);

    const stpArrowStripe = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.16, 0.16), mats.whiteMullion);
    stpArrowStripe.position.copy(stpArrowSign.position);
    stpArrowStripe.rotation.y = -0.85;
    entranceGroup.add(stpArrowStripe);

    const uiGateBoard = new THREE.Mesh(new THREE.BoxGeometry(0.98, 1.25, 0.12), mats.solarPanel);
    uiGateBoard.position.set(51.2 * Math.sin((-55.2 * Math.PI) / 180.0), 2.15, 51.2 * Math.cos((-55.2 * Math.PI) / 180.0));
    uiGateBoard.rotation.y = -0.85;
    entranceGroup.add(uiGateBoard);

    const uiGateBoardFrame = new THREE.Mesh(new THREE.BoxGeometry(1.06, 1.32, 0.10), mats.whiteMullion);
    uiGateBoardFrame.position.copy(uiGateBoard.position);
    uiGateBoardFrame.rotation.y = -0.85;
    entranceGroup.add(uiGateBoardFrame);

    const islandMonolithPos = new THREE.Vector3(
      52.2 * Math.sin((-59.0 * Math.PI) / 180.0),
      1.35,
      52.2 * Math.cos((-59.0 * Math.PI) / 180.0)
    );
    const islandMonolith = new THREE.Mesh(new THREE.BoxGeometry(0.72, 1.75, 0.24), mats.asphaltRoad);
    islandMonolith.position.copy(islandMonolithPos);
    islandMonolith.rotation.y = -0.78;
    entranceGroup.add(islandMonolith);

    const islandMonolithBase = new THREE.Mesh(new THREE.BoxGeometry(0.92, 0.38, 0.44), mats.concreteDriveway);
    islandMonolithBase.position.set(islandMonolithPos.x, 0.22, islandMonolithPos.z);
    islandMonolithBase.rotation.y = -0.78;
    entranceGroup.add(islandMonolithBase);

    const yellowCircleBadge = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.26, 14), mats.roadYellowLine);
    yellowCircleBadge.rotation.x = Math.PI * 0.5;
    yellowCircleBadge.position.set(islandMonolithPos.x, 1.22, islandMonolithPos.z);
    yellowCircleBadge.rotation.y = -0.78;
    entranceGroup.add(yellowCircleBadge);

    const umbulPole = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 4.2, 8), mats.whiteMullion);
    umbulPole.position.set(-38.6, 2.1, 44.0);
    entranceGroup.add(umbulPole);

    for (let uSeg = 0; uSeg < 6; uSeg++) {
      const uCloth = new THREE.Mesh(
        new THREE.BoxGeometry(0.42, 0.52, 0.08),
        uSeg % 2 === 0 ? mats.terracottaRoof : mats.whiteMullion
      );
      uCloth.position.set(-38.45, 1.15 + uSeg * 0.50, 44.0);
      uCloth.rotation.z = 0.08;
      entranceGroup.add(uCloth);
    }

    // Ring Road Streetlamps (PJU) tucked cleanly inside r <= 64.85 with warm night LED heads!
    const addRingPJU = (parent, rPju, angRad) => {
      const px = rPju * Math.cos(angRad);
      const pz = rPju * Math.sin(angRad);
      const pjuPole = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.13, 9.0, 8), mats.concreteTrim);
      pjuPole.position.set(px, 4.5, pz);
      parent.add(pjuPole);

      const armR = rPju - 1.05;
      const pjuArm = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.09, 0.09), mats.concreteTrim);
      pjuArm.position.set(armR * Math.cos(angRad), 8.85, armR * Math.sin(angRad));
      pjuArm.rotation.y = -angRad;
      pjuArm.rotation.z = 0.18;
      parent.add(pjuArm);

      const headR = rPju - 2.05;
      const head = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.10, 0.28), mats.asphaltRoad);
      head.position.set(headR * Math.cos(angRad), 9.04, headR * Math.sin(angRad));
      head.rotation.y = -angRad;
      parent.add(head);

      const lens = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.05, 0.22), mats.warmLuminaire);
      lens.position.set(headR * Math.cos(angRad), 8.98, headR * Math.sin(angRad));
      lens.rotation.y = -angRad;
      parent.add(lens);
    };

    for (let pj = 0; pj < 10; pj++) {
      addRingPJU(dppdGroup, 64.75, (pj / 10) * Math.PI * 2.0 + 0.18);
    }

    // Right Berm & Retaining Wall
    const rightBermGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => DPPDComplexBuilder.evaluateBermLocalSurface(7.8, 46.0, u, v, p, n),
      0.0, 1.0, 0.0, 1.0, 32, 10
    );
    entranceGroup.add(new THREE.Mesh(rightBermGeo, mats.grassBerm));

    entranceGroup.add(
      new THREE.Mesh(
        evalPolarCylinderBand(48.85, 0.45, 1.12, (14.0 * Math.PI) / 180.0, (43.5 * Math.PI) / 180.0),
        mats.stoneRetaining
      )
    );

    entranceGroup.add(
      new THREE.Mesh(
        evalPolarCylinderBand(44.12, 4.05, 6.22, (38.5 * Math.PI) / 180.0, (46.0 * Math.PI) / 180.0),
        mats.creamWall
      )
    );

    for (let sw = 0; sw < 3; sw++) {
      const aSw = ((39.8 + sw * 2.4) * Math.PI) / 180.0;
      const highWin = new THREE.Mesh(new THREE.BoxGeometry(1.38, 0.65, 0.28), mats.windowGlass);
      highWin.position.set(44.28 * Math.sin(aSw), 5.62, 44.28 * Math.cos(aSw));
      highWin.rotation.y = aSw;
      entranceGroup.add(highWin);
    }

    const createSlopedWingWall = (xPos) => {
      const extrude = DPPDComplexBuilder._buildPolygonPrismGeo(
        [
          [44.0, 0.0],
          [50.8, 0.0],
          [50.8, 0.65],
          [44.0, 3.85]
        ],
        0.48
      );
      extrude.rotateY(-Math.PI * 0.5);
      const mesh = new THREE.Mesh(extrude, mats.stoneRetaining);
      mesh.position.set(xPos, 0.0, 0.0);
      entranceGroup.add(mesh);
    };
    createSlopedWingWall(-6.45);
    createSlopedWingWall(6.93);

    // Front Porch Canopy, Warm Downlight, Lobby Door, Disability Ramps, & Flagpole with Merah-Putih Flag
    const canopyRoof = new THREE.Mesh(new THREE.BoxGeometry(13.2, 0.42, 3.4), mats.terracottaRoof);
    canopyRoof.position.set(0, 3.42, 45.6);
    canopyRoof.rotation.x = 0.08;
    entranceGroup.add(canopyRoof);

    const canopyFascia = new THREE.Mesh(new THREE.BoxGeometry(13.35, 0.20, 3.52), mats.darkWoodFascia);
    canopyFascia.position.set(0, 3.24, 45.6);
    entranceGroup.add(canopyFascia);

    const canopyLightStrip = new THREE.Mesh(new THREE.BoxGeometry(8.4, 0.05, 0.42), mats.warmLuminaire);
    canopyLightStrip.position.set(0, 3.12, 45.4);
    entranceGroup.add(canopyLightStrip);

    const groundWall = new THREE.Mesh(new THREE.BoxGeometry(13.2, 3.5, 0.5), mats.creamWall);
    groundWall.position.set(0, 1.75, 44.0);
    entranceGroup.add(groundWall);

    const lobbyDoor = new THREE.Mesh(new THREE.BoxGeometry(3.6, 2.55, 0.62), mats.windowGlass);
    lobbyDoor.position.set(0.0, 1.42, 44.05);
    entranceGroup.add(lobbyDoor);

    const narrowSlotL = new THREE.Mesh(new THREE.BoxGeometry(0.36, 1.65, 0.58), mats.windowGlass);
    narrowSlotL.position.set(-4.1, 1.95, 44.05);
    entranceGroup.add(narrowSlotL);

    const narrowSlotR = new THREE.Mesh(new THREE.BoxGeometry(0.36, 1.65, 0.58), mats.windowGlass);
    narrowSlotR.position.set(4.5, 1.95, 44.05);
    entranceGroup.add(narrowSlotR);

    const wallSign = new THREE.Mesh(new THREE.BoxGeometry(1.15, 0.42, 0.12), mats.brassSign);
    wallSign.position.set(2.55, 2.32, 44.28);
    entranceGroup.add(wallSign);

    for (const colX of [-1.85, 1.85]) {
      const porchCol = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 3.1, 10), mats.whiteMullion);
      porchCol.position.set(colX, 1.72, 45.8);
      entranceGroup.add(porchCol);
    }

    const plazaFloor = new THREE.Mesh(new THREE.BoxGeometry(13.0, 0.36, 3.8), mats.terracottaFloor);
    plazaFloor.position.set(0, 0.22, 45.9);
    entranceGroup.add(plazaFloor);

    for (let s = 0; s < 5; s++) {
      const stepL = new THREE.Mesh(new THREE.BoxGeometry(4.0, 0.10, 0.68 + s * 0.45), mats.terracottaFloor);
      stepL.position.set(-4.2, 0.36 - s * 0.06, 48.0 + s * 0.36);
      entranceGroup.add(stepL);

      const stepR = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.10, 0.68 + s * 0.45), mats.terracottaFloor);
      stepR.position.set(4.7, 0.36 - s * 0.06, 48.0 + s * 0.36);
      entranceGroup.add(stepR);
    }

    const greyRibbedRamp = new THREE.Mesh(new THREE.BoxGeometry(1.18, 0.16, 3.35), mats.stoneRetaining);
    greyRibbedRamp.position.set(-1.5, 0.21, 48.65);
    greyRibbedRamp.rotation.x = 0.085;
    entranceGroup.add(greyRibbedRamp);

    for (const rx of [-2.15, -0.85, 2.65]) {
      const rampCurb = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.48, 3.38), mats.terracottaRoof);
      rampCurb.position.set(rx, 0.34, 48.65);
      entranceGroup.add(rampCurb);
    }

    const centerRedRamp = new THREE.Mesh(new THREE.BoxGeometry(3.3, 0.16, 3.35), mats.terracottaFloor);
    centerRedRamp.position.set(0.9, 0.21, 48.65);
    centerRedRamp.rotation.x = 0.085;
    entranceGroup.add(centerRedRamp);

    const pavingForecourt = new THREE.Mesh(new THREE.BoxGeometry(6.2, 0.08, 2.2), mats.pavingBlock);
    pavingForecourt.position.set(1.2, 0.06, 50.5);
    entranceGroup.add(pavingForecourt);

    const concreteDriveway = new THREE.Mesh(new THREE.BoxGeometry(5.6, 0.12, 2.45), mats.concreteDriveway);
    concreteDriveway.position.set(1.15, 0.085, 52.25);
    entranceGroup.add(concreteDriveway);

    // Ceremonial Flagpole + Merah-Putih Indonesian Flag
    const flagDrumBase = new THREE.Mesh(new THREE.CylinderGeometry(0.46, 0.50, 0.28, 16), mats.concreteDriveway);
    flagDrumBase.position.set(-1.55, 0.18, 51.25);
    entranceGroup.add(flagDrumBase);

    const mainFlagpole = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.09, 11.4, 12), mats.whiteMullion);
    mainFlagpole.position.set(-1.55, 5.85, 51.25);
    entranceGroup.add(mainFlagpole);

    const flagFinial = new THREE.Mesh(new THREE.SphereGeometry(0.11, 10, 8), mats.brassSign);
    flagFinial.position.set(-1.55, 11.60, 51.25);
    entranceGroup.add(flagFinial);

    const flagRed = new THREE.Mesh(new THREE.BoxGeometry(1.85, 0.62, 0.05), mats.hydrantRed);
    flagRed.position.set(-0.58, 11.12, 51.28);
    flagRed.rotation.y = 0.12;
    entranceGroup.add(flagRed);

    const flagWhite = new THREE.Mesh(new THREE.BoxGeometry(1.85, 0.62, 0.05), mats.whiteMullion);
    flagWhite.position.set(-0.58, 10.50, 51.28);
    flagWhite.rotation.y = 0.12;
    entranceGroup.add(flagWhite);

    const blueParking = new THREE.Mesh(new THREE.BoxGeometry(4.4, 0.08, 3.1), mats.disabilityBlue);
    blueParking.position.set(-4.15, 0.075, 50.35);
    entranceGroup.add(blueParking);

    const whiteBorder = new THREE.Mesh(new THREE.BoxGeometry(4.6, 0.06, 3.3), mats.whiteMullion);
    whiteBorder.position.set(-4.15, 0.06, 50.35);
    entranceGroup.add(whiteBorder);

    const totemSilverBody = new THREE.Mesh(new THREE.BoxGeometry(0.96, 1.82, 0.18), mats.concreteDriveway);
    totemSilverBody.position.set(-4.85, 1.06, 51.15);
    totemSilverBody.rotation.y = 0.08;
    entranceGroup.add(totemSilverBody);

    const totemGoldTop = new THREE.Mesh(new THREE.BoxGeometry(1.06, 0.24, 0.22), mats.brassSign);
    totemGoldTop.position.set(-4.88, 1.92, 51.15);
    totemGoldTop.rotation.y = 0.08;
    entranceGroup.add(totemGoldTop);

    const infoFrame = new THREE.Mesh(new THREE.BoxGeometry(1.42, 1.06, 0.14), mats.whiteMullion);
    infoFrame.position.set(5.65, 1.48, 50.65);
    infoFrame.rotation.y = -0.18;
    entranceGroup.add(infoFrame);

    const infoMapFace = new THREE.Mesh(new THREE.BoxGeometry(1.26, 0.88, 0.16), mats.brassSign);
    infoMapFace.position.set(5.65, 1.48, 50.65);
    infoMapFace.rotation.y = -0.18;
    entranceGroup.add(infoMapFace);

    const infoPost = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.95, 10), mats.whiteMullion);
    infoPost.position.set(5.65, 0.52, 50.65);
    entranceGroup.add(infoPost);

    const ditchWallInner = new THREE.Mesh(evalPolarCylinderBand(50.85, 0.0, 0.36, alphaMin, alphaMax), mats.stoneRetaining);
    const ditchWallOuter = new THREE.Mesh(evalPolarCylinderBand(51.65, 0.0, 0.36, alphaMin, alphaMax), mats.stoneRetaining);
    entranceGroup.add(ditchWallInner, ditchWallOuter);

    // 3 Royal Palms in front of Gedung DPPD Right Wing
    const palmAnglesDeg = [19.2, 31.5, 38.8];
    const palmHeights = [14.5, 16.2, 13.8];
    palmAnglesDeg.forEach((deg, idx) => {
      const aPalm = (deg * Math.PI) / 180.0;
      const rPalm = idx === 0 ? 49.8 : 50.2;
      const hPalm = palmHeights[idx];
      const px = rPalm * Math.sin(aPalm);
      const pz = rPalm * Math.cos(aPalm);

      const palmTrunk = new THREE.Mesh(new THREE.CylinderGeometry(0.20, 0.32, hPalm, 10), mats.concreteTrim);
      palmTrunk.position.set(px, hPalm * 0.5, pz);
      entranceGroup.add(palmTrunk);

      for (let f = 0; f < 8; f++) {
        const frond = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.08, 3.05), mats.LushShrub);
        const fa = (f / 8) * Math.PI * 2.0;
        frond.position.set(px + Math.sin(fa) * 1.42, hPalm - 0.25, pz + Math.cos(fa) * 1.42);
        frond.rotation.y = fa;
        frond.rotation.x = 0.38;
        entranceGroup.add(frond);
      }
    });

    // NW & SE Terracotta Wing End Gables
    const sideGableGeo = DPPDComplexBuilder._buildPolygonPrismGeo(
      [
        [26.5, 6.22],
        [45.5, 6.22],
        [35.8, 11.45]
      ],
      0.48
    );
    sideGableGeo.rotateY(-Math.PI * 0.5);

    const nwGableGroup = new THREE.Group();
    nwGableGroup.rotation.y = (-52.0 * Math.PI) / 180.0;
    entranceGroup.add(nwGableGroup);

    const sideWallMain = new THREE.Mesh(new THREE.BoxGeometry(0.45, 6.25, 17.0), mats.salmonGable);
    sideWallMain.position.set(0, 3.12, 36.0);
    nwGableGroup.add(sideWallMain);

    const sideGableMesh = new THREE.Mesh(sideGableGeo, mats.salmonGable);
    sideGableMesh.position.set(0.24, 0, 0);
    nwGableGroup.add(sideGableMesh);

    const whiteBeamMid = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.36, 17.4), mats.whiteMullion);
    whiteBeamMid.position.set(0, 3.65, 36.0);
    nwGableGroup.add(whiteBeamMid);

    const whiteBeamTop = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.38, 18.8), mats.whiteMullion);
    whiteBeamTop.position.set(0, 6.20, 36.0);
    nwGableGroup.add(whiteBeamTop);

    for (const zPil of [28.2, 35.8, 43.5]) {
      const yellowPilaster = new THREE.Mesh(new THREE.BoxGeometry(0.62, 3.65, 0.55), mats.brassSign);
      yellowPilaster.position.set(0, 1.82, zPil);
      nwGableGroup.add(yellowPilaster);
    }

    const sideEntryDoor = new THREE.Mesh(new THREE.BoxGeometry(0.65, 2.35, 1.85), mats.windowGlass);
    sideEntryDoor.position.set(-0.15, 1.25, 42.2);
    nwGableGroup.add(sideEntryDoor);

    const sideWalkway = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.18, 9.5), mats.concreteDriveway);
    sideWalkway.position.set(-1.3, 0.09, 44.8);
    nwGableGroup.add(sideWalkway);

    const seGableGroup = new THREE.Group();
    seGableGroup.rotation.y = (46.0 * Math.PI) / 180.0;
    entranceGroup.add(seGableGroup);

    const seSideWallMain = new THREE.Mesh(new THREE.BoxGeometry(0.45, 6.25, 17.0), mats.salmonGable);
    seSideWallMain.position.set(0, 3.12, 36.0);
    seGableGroup.add(seSideWallMain);

    const seSideGableMesh = new THREE.Mesh(sideGableGeo, mats.salmonGable);
    seSideGableMesh.position.set(0.24, 0, 0);
    seGableGroup.add(seSideGableMesh);

    const seWhiteBeamMid = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.36, 17.4), mats.whiteMullion);
    seWhiteBeamMid.position.set(0, 3.65, 36.0);
    seGableGroup.add(seWhiteBeamMid);

    const seWhiteBeamTop = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.38, 18.8), mats.whiteMullion);
    seWhiteBeamTop.position.set(0, 6.20, 36.0);
    seGableGroup.add(seWhiteBeamTop);

    const seBermEndExtrude = DPPDComplexBuilder._buildPolygonPrismGeo(
      [
        [44.0, 0.0],
        [50.85, 0.0],
        [50.85, 0.68],
        [44.0, 3.75]
      ],
      0.46
    );
    seBermEndExtrude.rotateY(-Math.PI * 0.5);
    const seBermEndMesh = new THREE.Mesh(seBermEndExtrude, mats.stoneRetaining);
    seBermEndMesh.position.set(0.22, 0, 0);
    seGableGroup.add(seBermEndMesh);

    // 4. TAHAP 5: WEST DRIVEWAY, PARKING LOTS, PARKED CAMPUS SUVS/SEDANS, & HYDRANT NETWORK
    const westDriveway = new THREE.Mesh(new THREE.BoxGeometry(6.8, 0.09, 30.0), mats.asphaltRoad);
    westDriveway.position.set(-34.5, 0.085, -3.5);
    westDriveway.rotation.y = Math.PI * 0.5;
    contextGroup.add(westDriveway);

    const stpParkingLot = new THREE.Mesh(new THREE.BoxGeometry(18.0, 0.08, 22.0), mats.pavingBlock);
    stpParkingLot.position.set(-36.0, 0.075, -16.5);
    contextGroup.add(stpParkingLot);

    const westParkingAsphalt = new THREE.Mesh(new THREE.BoxGeometry(24.0, 0.09, 26.0), mats.asphaltRoad);
    westParkingAsphalt.position.set(-32.0, 0.08, -20.5);
    contextGroup.add(westParkingAsphalt);

    // White parking stall lines + realistic parked campus vehicles
    for (let st = 0; st < 6; st++) {
      const zSlot = -28.0 + st * 3.1;
      const line = new THREE.Mesh(new THREE.BoxGeometry(4.8, 0.03, 0.14), mats.whiteMullion);
      line.position.set(-31.5, 0.13, zSlot);
      contextGroup.add(line);
    }
    contextGroup.add(DPPDComplexBuilder._buildParkedCar(mats, mats.carWhite, -31.6, 0.12, -26.4, Math.PI * 0.5));
    contextGroup.add(DPPDComplexBuilder._buildParkedCar(mats, mats.carDarkGrey, -31.5, 0.12, -20.2, Math.PI * 0.5));
    contextGroup.add(DPPDComplexBuilder._buildParkedCar(mats, mats.carSilver, -31.6, 0.12, -14.0, Math.PI * 0.5));
    contextGroup.add(DPPDComplexBuilder._buildParkedCar(mats, mats.carNavy, -38.8, 0.12, -18.5, -Math.PI * 0.5));

    for (let mb = 0; mb < 4; mb++) {
      const mMat = mb % 2 === 0 ? mats.hydrantRed : mats.carNavy;
      contextGroup.add(DPPDComplexBuilder._buildParkedMotorbike(mats, mMat, 43.2 + mb * 0.95, 0.10, 29.0 + mb * 0.65, -0.58));
    }

    const mulchIsland1 = new THREE.Mesh(new THREE.CylinderGeometry(6.8, 7.0, 0.22, 24), mats.mulchSoil);
    mulchIsland1.scale.set(1.0, 1.0, 0.82);
    mulchIsland1.position.set(-45.2, 0.12, -11.5);
    contextGroup.add(mulchIsland1);

    const masukSignBlue = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.85, 1.15), mats.disabilityBlue);
    masukSignBlue.position.set(-47.8, 0.85, -6.5);
    masukSignBlue.rotation.y = 0.25;
    contextGroup.add(masukSignBlue);

    const uiPurpleBoard = new THREE.Mesh(new THREE.BoxGeometry(0.14, 1.35, 1.15), mats.purpleSign);
    uiPurpleBoard.position.set(-47.2, 1.85, -6.4);
    uiPurpleBoard.rotation.y = 0.25;
    contextGroup.add(uiPurpleBoard);

    for (const zLeg of [-6.85, -6.05]) {
      const signLeg = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 2.45, 8), mats.stoneRetaining);
      signLeg.position.set(-47.5, 1.22, zLeg);
      contextGroup.add(signLeg);
    }

    const islandTrees = [
      { x: -48.5, z: -13.2, h: 8.5, r: 2.6 },
      { x: -44.2, z: -9.5, h: 7.2, r: 2.8 },
      { x: -47.2, z: -16.8, h: 7.8, r: 2.2 },
      { x: -45.5, z: -21.5, h: 8.2, r: 2.4 },
      { x: -43.5, z: -26.5, h: 8.8, r: 2.5 },
      { x: -41.2, z: -31.0, h: 9.2, r: 2.6 }
    ];
    islandTrees.forEach((tr) => {
      const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.24, tr.h, 8), mats.mulchSoil);
      trunk.position.set(tr.x, tr.h * 0.5, tr.z);
      contextGroup.add(trunk);

      const tier1 = new THREE.Mesh(new THREE.CylinderGeometry(tr.r * 0.9, tr.r * 1.15, 0.85, 9), mats.LushShrub);
      tier1.position.set(tr.x, tr.h * 0.68, tr.z);
      contextGroup.add(tier1);

      const tier2 = new THREE.Mesh(new THREE.CylinderGeometry(tr.r * 0.55, tr.r * 0.85, 0.85, 9), mats.grassBerm);
      tier2.position.set(tr.x, tr.h * 0.92, tr.z);
      contextGroup.add(tier2);
    });

    const stpDriveway = new THREE.Mesh(new THREE.BoxGeometry(20.0, 0.095, 7.2), mats.asphaltRoad);
    stpDriveway.position.set(-34.5, 0.082, -36.5);
    stpDriveway.rotation.y = -0.32;
    contextGroup.add(stpDriveway);

    // Fire Hydrant Ring Pipe along North Ditch
    const pipeSegs = 22;
    const pipeThStart = (-132.0 * Math.PI) / 180.0;
    const pipeThEnd = (-62.0 * Math.PI) / 180.0;
    const rPipe = 51.35;
    for (let ps = 0; ps < pipeSegs; ps++) {
      const t0 = pipeThStart + (ps / pipeSegs) * (pipeThEnd - pipeThStart);
      const t1 = pipeThStart + ((ps + 1) / pipeSegs) * (pipeThEnd - pipeThStart);
      const x0 = rPipe * Math.cos(t0);
      const z0 = rPipe * Math.sin(t0);
      const x1 = rPipe * Math.cos(t1);
      const z1 = rPipe * Math.sin(t1);
      const segLen = Math.hypot(x1 - x0, z1 - z0);

      const pipeGeo = new THREE.CylinderGeometry(0.065, 0.065, segLen + 0.05, 8);
      pipeGeo.rotateX(Math.PI * 0.5);
      const pipeMesh = new THREE.Mesh(pipeGeo, mats.hydrantRed);
      pStart.set(x0, 0.42, z0);
      pEnd.set(x1, 0.42, z1);
      DPPDComplexBuilder._orientBeamAlongZ(pipeMesh, pStart, pEnd, 0.0);
      contextGroup.add(pipeMesh);
    }

    // 5. GEDUNG INTI TENGAH: ITTC / PERPUSTAKAAN LAMA UI (4-STORY COURTYARD BUILDING)
    const libGroup = new THREE.Group();
    contextGroup.add(libGroup);

    const libWings = [
      { x: 0, z: -12.0, w: 34.0, d: 10.0 },
      { x: 0, z: 12.0, w: 34.0, d: 10.0 },
      { x: -12.0, z: 0, w: 10.0, d: 14.0 },
      { x: 12.0, z: 0, w: 10.0, d: 14.0 }
    ];
    libWings.forEach((lw) => {
      const wMesh = new THREE.Mesh(new THREE.BoxGeometry(lw.w, 13.5, lw.d), mats.creamWall);
      wMesh.position.set(lw.x, 6.75, lw.z);
      libGroup.add(wMesh);

      const gMesh = new THREE.Mesh(new THREE.BoxGeometry(lw.w + 0.18, 9.8, lw.d + 0.18), mats.windowGlass);
      gMesh.position.set(lw.x, 8.2, lw.z);
      libGroup.add(gMesh);
    });

    for (let floor = 1; floor <= 4; floor++) {
      for (const sx of [-1, 1]) {
        const spandrelX = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.52, 34.4), mats.whiteMullion);
        spandrelX.position.set(sx * 17.05, floor * 3.25, 0);
        libGroup.add(spandrelX);

        const spandrelZ = new THREE.Mesh(new THREE.BoxGeometry(34.4, 0.52, 0.55), mats.whiteMullion);
        spandrelZ.position.set(0, floor * 3.25, sx * 17.05);
        libGroup.add(spandrelZ);
      }
    }
    for (let col = -5; col <= 5; col++) {
      for (const sx of [-1, 1]) {
        const vPillarX = new THREE.Mesh(new THREE.BoxGeometry(0.58, 13.6, 0.42), mats.whiteMullion);
        vPillarX.position.set(sx * 17.08, 6.8, col * 3.1);
        libGroup.add(vPillarX);

        const vPillarZ = new THREE.Mesh(new THREE.BoxGeometry(0.42, 13.6, 0.58), mats.whiteMullion);
        vPillarZ.position.set(col * 3.1, 6.8, sx * 17.08);
        libGroup.add(vPillarZ);
      }
    }

    const libRoofSlope = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = u * Math.PI * 2.0;
        const c = Math.cos(th);
        const s = Math.sin(th);
        const maxAbs = Math.max(Math.abs(c), Math.abs(s));
        const rSquare = (6.0 + v * 12.2) / maxAbs;
        const y = 13.5 + Math.sin(v * Math.PI) * 4.4;
        p.set(rSquare * c, y, rSquare * s);
        n.set(c * 0.25, 0.96, s * 0.25).normalize();
      },
      0.0, 1.0, 0.0, 1.0, 32, 6
    );
    libGroup.add(new THREE.Mesh(libRoofSlope, mats.terracottaRoof));

    // 6. GEDUNG SCIENCE TECHNO PARK (STP-UI 8-LANTAI) & GEDUNG LABORATORIUM ILRC / DRPM (4-LANTAI + MENARA BATA)
    const ilrcStart = (-129.0 * Math.PI) / 180.0;
    const stpArcSpan = (91.0 * Math.PI) / 180.0;
    const ilrcArcSpan = (179.0 * Math.PI) / 180.0;
    const ilrcHeight = 25.4;

    const evalILRCHull = (rVal, yMin, yMax, uSegs = 44, vSegs = 6, spanRad = stpArcSpan, startRad = ilrcStart, inward = false) =>
      DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const th = startRad + u * spanRad;
          p.set(rVal * Math.cos(th), yMin + v * (yMax - yMin), rVal * Math.sin(th));
          const sgn = inward ? -1.0 : 1.0;
          n.set(sgn * Math.cos(th), 0, sgn * Math.sin(th));
        },
        0.0, 1.0, 0.0, 1.0, uSegs, vSegs
      );

    // STP-UI 8-Story Outer & Inner Courtyard Hulls + Roof Deck Slab + Night-Lit Glazing
    contextGroup.add(new THREE.Mesh(evalILRCHull(43.2, 0.0, ilrcHeight, 44, 6, stpArcSpan, ilrcStart, false), mats.ilrcConcrete));
    contextGroup.add(new THREE.Mesh(evalILRCHull(30.8, 0.0, ilrcHeight, 44, 6, stpArcSpan, ilrcStart, true), mats.ilrcConcrete));
    contextGroup.add(new THREE.Mesh(evalILRCHull(43.32, 4.2, 23.6, 44, 4, stpArcSpan, ilrcStart, false), mats.windowGlass));
    contextGroup.add(new THREE.Mesh(evalILRCHull(30.68, 4.2, 23.2, 36, 4, stpArcSpan, ilrcStart, true), mats.windowGlass));

    const stpRoofSlabGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = ilrcStart + u * stpArcSpan;
        const r = 30.8 + v * (43.2 - 30.8);
        p.set(r * Math.cos(th), ilrcHeight, r * Math.sin(th));
        n.set(0, 1, 0);
      },
      0.0, 1.0, 0.0, 1.0, 44, 2
    );
    contextGroup.add(new THREE.Mesh(stpRoofSlabGeo, mats.ilrcConcrete));

    // ILRC / DRPM 4-Story East Laboratory Wing (-31.5° .. +50°)
    const eastIlrcStart = (-31.5 * Math.PI) / 180.0;
    const eastIlrcSpan = (81.5 * Math.PI) / 180.0;
    const eastIlrcH = 15.2;
    contextGroup.add(new THREE.Mesh(evalILRCHull(42.6, 4.0, eastIlrcH, 36, 4, eastIlrcSpan, eastIlrcStart, false), mats.ilrcConcrete));
    contextGroup.add(new THREE.Mesh(evalILRCHull(39.8, 0.0, 4.0, 36, 2, eastIlrcSpan, eastIlrcStart, false), mats.whiteMullion));
    contextGroup.add(new THREE.Mesh(evalILRCHull(30.8, 0.0, eastIlrcH, 36, 4, eastIlrcSpan, eastIlrcStart, true), mats.ilrcConcrete));
    contextGroup.add(new THREE.Mesh(evalILRCHull(42.72, 4.2, eastIlrcH - 0.5, 36, 4, eastIlrcSpan, eastIlrcStart, false), mats.windowGlass));

    const eastIlrcRoofSlab = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = eastIlrcStart + u * eastIlrcSpan;
        const r = 30.8 + v * (42.6 - 30.8);
        p.set(r * Math.cos(th), eastIlrcH, r * Math.sin(th));
        n.set(0, 1, 0);
      },
      0.0, 1.0, 0.0, 1.0, 36, 2
    );
    contextGroup.add(new THREE.Mesh(eastIlrcRoofSlab, mats.ilrcConcrete));

    const eastIlrcNorthEndWall = new THREE.Mesh(new THREE.BoxGeometry(11.8, eastIlrcH - 4.0, 0.65), mats.stoneRetaining);
    eastIlrcNorthEndWall.position.set(36.7 * Math.cos(eastIlrcStart), 9.6, 36.7 * Math.sin(eastIlrcStart));
    eastIlrcNorthEndWall.rotation.y = -eastIlrcStart;
    contextGroup.add(eastIlrcNorthEndWall);

    const eastIlrcCornerPillar = new THREE.Mesh(new THREE.BoxGeometry(0.95, 4.2, 0.95), mats.whiteMullion);
    eastIlrcCornerPillar.position.set(42.1 * Math.cos(eastIlrcStart), 2.1, 42.1 * Math.sin(eastIlrcStart));
    eastIlrcCornerPillar.rotation.y = -eastIlrcStart;
    contextGroup.add(eastIlrcCornerPillar);

    const gapBridgeTh = (-34.8 * Math.PI) / 180.0;
    const gapConnectorWall = new THREE.Mesh(new THREE.BoxGeometry(2.4, 11.5, 4.8), mats.whiteMullion);
    gapConnectorWall.position.set(35.5 * Math.cos(gapBridgeTh), 6.2, 35.5 * Math.sin(gapBridgeTh));
    gapConnectorWall.rotation.y = -gapBridgeTh;
    contextGroup.add(gapConnectorWall);

    for (const yLedgeE of [4.2, 7.8, 11.4, 15.2]) {
      const eLedgeGeo = DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const th = eastIlrcStart + u * eastIlrcSpan;
          const r = 42.5 + v * 1.35;
          p.set(r * Math.cos(th), yLedgeE, r * Math.sin(th));
          n.set(0, 1, 0);
        },
        0.0, 1.0, 0.0, 1.0, 32, 2
      );
      contextGroup.add(new THREE.Mesh(eLedgeGeo, mats.whiteMullion));
    }

    for (let elv = 0; elv < 28; elv++) {
      const thLv = eastIlrcStart + (0.04 + elv * 0.032) * eastIlrcSpan;
      const lvBar = new THREE.Mesh(new THREE.BoxGeometry(0.16, 10.6, 0.22), mats.concreteTrim);
      lvBar.position.set(43.45 * Math.cos(thLv), 9.6, 43.45 * Math.sin(thLv));
      lvBar.rotation.y = -thLv;
      contextGroup.add(lvBar);
    }

    // STP-UI Main North Drop-Off Portico, 3D "STP-UI" Letters, Warm Soffit Lighting, & Directory Totem
    const northPorticoTh = (-79.5 * Math.PI) / 180.0;
    const porticoGroup = new THREE.Group();
    porticoGroup.position.set(45.2 * Math.cos(northPorticoTh), 0.0, 45.2 * Math.sin(northPorticoTh));
    porticoGroup.rotation.y = -northPorticoTh;
    contextGroup.add(porticoGroup);

    const northPorticoCanopy = new THREE.Mesh(new THREE.BoxGeometry(6.6, 0.72, 12.6), mats.ilrcConcrete);
    northPorticoCanopy.position.set(0.7, 4.18, 0.0);
    porticoGroup.add(northPorticoCanopy);

    const woodCeiling = new THREE.Mesh(new THREE.BoxGeometry(5.6, 0.12, 11.4), mats.copperACP);
    woodCeiling.position.set(0.5, 3.80, 0.0);
    porticoGroup.add(woodCeiling);

    for (const zDown of [-3.6, 0.0, 3.6]) {
      const porticoLight = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.05, 0.45), mats.warmLuminaire);
      porticoLight.position.set(0.8, 3.72, zDown);
      porticoGroup.add(porticoLight);
    }

    for (let ch = 0; ch < 24; ch++) {
      const chZ = -5.1 + ch * 0.44;
      if (ch === 7 || ch === 14 || ch === 17) continue;
      const letterMesh = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.28, 0.28), mats.asphaltRoad);
      letterMesh.position.set(4.02, 4.18, chZ);
      porticoGroup.add(letterMesh);
    }

    const vColLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.26, 4.15, 12), mats.ilrcConcrete);
    vColLeft.position.set(2.8, 2.05, -4.4);
    vColLeft.rotation.x = -0.22;
    porticoGroup.add(vColLeft);

    for (const zCyl of [-2.8, 4.4]) {
      const mainCyl = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.36, 4.10, 14), mats.ilrcConcrete);
      mainCyl.position.set(2.6, 2.05, zCyl);
      porticoGroup.add(mainCyl);
    }

    const blueLobbyBanner = new THREE.Mesh(new THREE.BoxGeometry(0.45, 3.65, 1.65), mats.disabilityBlue);
    blueLobbyBanner.position.set(-1.35, 1.85, -3.8);
    porticoGroup.add(blueLobbyBanner);

    const porticoLobbyGlass = new THREE.Mesh(new THREE.BoxGeometry(0.85, 3.85, 10.6), mats.windowGlass);
    porticoLobbyGlass.position.set(-1.6, 1.95, 0.0);
    porticoGroup.add(porticoLobbyGlass);

    for (const zPiloti of [-8.8, -14.2, -19.6]) {
      const pilotiCol = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 4.15, 14), mats.ilrcConcrete);
      pilotiCol.position.set(-0.65, 2.08, zPiloti);
      porticoGroup.add(pilotiCol);
    }
    for (let bp = 0; bp < 5; bp++) {
      const bFin = new THREE.Mesh(new THREE.BoxGeometry(0.32, 3.4, 0.55), mats.disabilityBlue);
      bFin.position.set(-1.35, 1.75, -6.5 - bp * 1.45);
      porticoGroup.add(bFin);
    }

    const wedgePlanter = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.92, 9.2), mats.whiteMullion);
    wedgePlanter.position.set(5.4, 0.44, -2.4);
    wedgePlanter.rotation.z = -0.18;
    porticoGroup.add(wedgePlanter);

    const wedgeFoliage = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.62, 8.4), mats.LushShrub);
    wedgeFoliage.position.set(5.45, 0.85, -2.4);
    wedgeFoliage.rotation.z = -0.18;
    porticoGroup.add(wedgeFoliage);

    const stpUiOffsetsZ = [-5.4, -3.9, -2.4, -1.1, 0.2, 1.6];
    stpUiOffsetsZ.forEach((zOff, idx) => {
      const isDash = idx === 3;
      const letter3D = new THREE.Mesh(
        new THREE.BoxGeometry(isDash ? 0.38 : 1.25, 0.24, isDash ? 0.75 : 0.92),
        mats.copperACP
      );
      letter3D.position.set(5.55, 1.18, zOff);
      letter3D.rotation.z = -0.18;
      porticoGroup.add(letter3D);
    });

    const whiteTiledApron = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.16, 18.5), mats.concreteDriveway);
    whiteTiledApron.position.set(7.45, 0.12, -2.2);
    porticoGroup.add(whiteTiledApron);

    const stpTotemSign = new THREE.Mesh(new THREE.BoxGeometry(1.08, 2.65, 0.28), mats.asphaltRoad);
    stpTotemSign.position.set(6.4, 1.75, -9.2);
    stpTotemSign.rotation.y = -0.25;
    porticoGroup.add(stpTotemSign);

    const stpTotemYellowBand = new THREE.Mesh(new THREE.BoxGeometry(1.12, 0.42, 0.32), mats.roadYellowLine);
    stpTotemYellowBand.position.set(6.4, 0.72, -9.2);
    stpTotemYellowBand.rotation.y = -0.25;
    porticoGroup.add(stpTotemYellowBand);

    // STP-UI L2 & L3 Copper-Bronze ACP Geometric Facade + L4..L8 5 Curved Concrete Sunshade Ledges
    for (const yFrame of [4.2, 7.6, 11.0]) {
      const frameLedgeGeo = DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const th = ilrcStart + u * stpArcSpan;
          const r = 43.2 + v * 1.15;
          p.set(r * Math.cos(th), yFrame, r * Math.sin(th));
          n.set(0, 1, 0);
        },
        0.0, 1.0, 0.0, 1.0, 36, 2
      );
      contextGroup.add(new THREE.Mesh(frameLedgeGeo, mats.concreteTrim));
    }

    const copperBays = 22;
    for (let cb = 1; cb < copperBays; cb++) {
      const thBay = ilrcStart + (cb / copperBays) * (0.94 * stpArcSpan);
      const rScreen = 44.25;
      const bx = rScreen * Math.cos(thBay);
      const bz = rScreen * Math.sin(thBay);
      const tiltSign = cb % 2 === 0 ? 1.0 : -1.0;

      const panelL2 = new THREE.Mesh(new THREE.BoxGeometry(0.14, 3.38, 1.35), mats.copperACP);
      panelL2.position.set(bx, 5.90, bz);
      panelL2.rotation.y = -thBay;
      panelL2.rotation.x = tiltSign * 0.14;
      contextGroup.add(panelL2);

      const panelL3 = new THREE.Mesh(new THREE.BoxGeometry(0.14, 3.38, 1.35), mats.copperACP);
      panelL3.position.set(bx, 9.30, bz);
      panelL3.rotation.y = -thBay;
      panelL3.rotation.x = -tiltSign * 0.14;
      contextGroup.add(panelL3);
    }

    for (let fl = 0; fl < 5; fl++) {
      const yFl = 11.0 + fl * 2.85;
      const sunLedgeGeo = DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const th = ilrcStart + u * stpArcSpan;
          const r = 43.1 + v * 1.35;
          p.set(r * Math.cos(th), yFl, r * Math.sin(th));
          n.set(0, 1, 0);
        },
        0.0, 1.0, 0.0, 1.0, 44, 2
      );
      contextGroup.add(new THREE.Mesh(sunLedgeGeo, mats.ilrcConcrete));
      contextGroup.add(new THREE.Mesh(evalILRCHull(44.45, yFl - 0.22, yFl + 0.12, 44, 2, stpArcSpan), mats.whiteMullion));
    }

    for (let wb = 0; wb < 26; wb++) {
      const thW = ilrcStart + (0.04 + wb * 0.034) * stpArcSpan;
      const mFl4 = new THREE.Mesh(new THREE.BoxGeometry(0.18, 2.65, 0.14), mats.whiteMullion);
      mFl4.position.set(43.38 * Math.cos(thW), 12.42, 43.38 * Math.sin(thW));
      mFl4.rotation.y = -thW;
      contextGroup.add(mFl4);

      for (let fl = 1; fl <= 4; fl++) {
        const yWin = 11.0 + fl * 2.85 + 1.35;
        if ((wb + fl) % 3 !== 0) {
          const wallPier = new THREE.Mesh(new THREE.BoxGeometry(0.22, 2.55, 1.85), mats.ilrcConcrete);
          wallPier.position.set(43.36 * Math.cos(thW), yWin, 43.36 * Math.sin(thW));
          wallPier.rotation.y = -thW;
          contextGroup.add(wallPier);
        } else {
          const winFrame = new THREE.Mesh(new THREE.BoxGeometry(0.26, 1.75, 0.92), mats.whiteMullion);
          winFrame.position.set(43.38 * Math.cos(thW), yWin, 43.38 * Math.sin(thW));
          winFrame.rotation.y = -thW;
          contextGroup.add(winFrame);

          const winPane = new THREE.Mesh(new THREE.BoxGeometry(0.30, 1.55, 0.72), mats.windowGlass);
          winPane.position.set(43.40 * Math.cos(thW), yWin, 43.40 * Math.sin(thW));
          winPane.rotation.y = -thW;
          contextGroup.add(winPane);
        }
      }
    }

    // STP-UI NE Copper Fin Wall, Golden UI Makara Emblem, 2-Line 3D "SCIENCE TECHNO PARK / UNIVERSITAS INDONESIA" & Solar Array
    const neWingEndTh = (-38.0 * Math.PI) / 180.0;
    const neFinWall = new THREE.Mesh(new THREE.BoxGeometry(2.8, 25.6, 1.35), mats.copperACP);
    neFinWall.position.set(44.1 * Math.cos(neWingEndTh), 12.8, 44.1 * Math.sin(neWingEndTh));
    neFinWall.rotation.y = -neWingEndTh;
    contextGroup.add(neFinWall);

    const seStairWall = new THREE.Mesh(new THREE.BoxGeometry(12.2, 24.8, 0.85), mats.whiteMullion);
    seStairWall.position.set(37.0 * Math.cos(neWingEndTh + 0.018), 12.4, 37.0 * Math.sin(neWingEndTh + 0.018));
    seStairWall.rotation.y = -neWingEndTh;
    contextGroup.add(seStairWall);

    const makaraTh = (-40.8 * Math.PI) / 180.0;
    const makaraGroup = new THREE.Group();
    makaraGroup.position.set(44.58 * Math.cos(makaraTh), 23.15, 44.58 * Math.sin(makaraTh));
    makaraGroup.rotation.y = -makaraTh;
    const makaraBacking = new THREE.Mesh(new THREE.CylinderGeometry(1.18, 1.18, 0.16, 18), mats.asphaltRoad);
    makaraBacking.rotation.z = Math.PI * 0.5;
    makaraGroup.add(makaraBacking);
    const makaraGoldCrest = new THREE.Mesh(new THREE.CylinderGeometry(0.96, 0.96, 0.22, 16), mats.brassSign);
    makaraGoldCrest.rotation.z = Math.PI * 0.5;
    makaraGroup.add(makaraGoldCrest);
    contextGroup.add(makaraGroup);

    for (let chTop = 0; chTop < 19; chTop++) {
      if (chTop === 7 || chTop === 14) continue;
      const thCh = (-43.2 - chTop * 0.88) * (Math.PI / 180.0);
      const topLetter = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.68, 0.46), mats.whiteMullion);
      topLetter.position.set(44.58 * Math.cos(thCh), 23.62, 44.58 * Math.sin(thCh));
      topLetter.rotation.y = -thCh;
      contextGroup.add(topLetter);
    }
    for (let chBot = 0; chBot < 21; chBot++) {
      if (chBot === 11) continue;
      const thCh = (-43.0 - chBot * 0.82) * (Math.PI / 180.0);
      const botLetter = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.62, 0.42), mats.whiteMullion);
      botLetter.position.set(44.58 * Math.cos(thCh), 22.65, 44.58 * Math.sin(thCh));
      botLetter.rotation.y = -thCh;
      contextGroup.add(botLetter);
    }

    const roofCrownGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = ilrcStart + u * stpArcSpan;
        const r = 42.6 + v * 2.45;
        p.set(r * Math.cos(th), ilrcHeight + 0.15, r * Math.sin(th));
        n.set(0, 1, 0);
      },
      0.0, 1.0, 0.0, 1.0, 44, 2
    );
    contextGroup.add(new THREE.Mesh(roofCrownGeo, mats.copperACP));
    contextGroup.add(new THREE.Mesh(evalILRCHull(45.02, ilrcHeight - 0.55, ilrcHeight + 0.35, 44, 2, stpArcSpan, ilrcStart), mats.copperACP));

    for (let sp = 0; sp < 14; sp++) {
      const th = ilrcStart + (0.10 + sp * 0.055) * stpArcSpan;
      const panel = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.24, 6.4), mats.solarPanel);
      panel.position.set(37.2 * Math.cos(th), ilrcHeight + 0.35, 37.2 * Math.sin(th));
      panel.rotation.y = -th + Math.PI * 0.5;
      panel.rotation.z = 0.16;
      contextGroup.add(panel);
    }

    // NW Glass Stair Tower & Louver Fins
    const nwStairGlass = new THREE.Mesh(new THREE.BoxGeometry(4.4, 24.8, 13.6), mats.windowGlass);
    nwStairGlass.position.set(37.2 * Math.cos(ilrcStart), 12.5, 37.2 * Math.sin(ilrcStart));
    nwStairGlass.rotation.y = -ilrcStart;
    contextGroup.add(nwStairGlass);

    const louverTowerGroup = new THREE.Group();
    louverTowerGroup.position.set(37.2 * Math.cos(ilrcStart), 0.0, 37.2 * Math.sin(ilrcStart));
    louverTowerGroup.rotation.y = -ilrcStart;
    for (let lv = -6; lv <= 6; lv++) {
      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.18, 25.2, 0.45), mats.whiteMullion);
      fin.position.set(-2.35, 12.6, lv * 1.02);
      louverTowerGroup.add(fin);
    }
    contextGroup.add(louverTowerGroup);

    // East Grove Trees, Breezeway Canopy, Cantilevered L4 Glass Box, & SE 25.2m Terracotta Brick Tower
    const eastGroveCoords = [
      { x: 48.2, z: -15.8, h: 8.2, r: 3.4, redFlush: true },
      { x: 49.0, z: -11.5, h: 8.6, r: 3.6, redFlush: true },
      { x: 49.5, z: -7.2, h: 8.0, r: 3.3, redFlush: true },
      { x: 49.2, z: -2.8, h: 7.8, r: 3.2, redFlush: false },
      { x: 48.8, z: 1.8, h: 8.2, r: 3.5, redFlush: true },
      { x: 48.2, z: 6.4, h: 8.4, r: 3.6, redFlush: false },
      { x: 47.2, z: 11.2, h: 8.0, r: 3.4, redFlush: false },
      { x: 45.8, z: 15.8, h: 7.8, r: 3.2, redFlush: false }
    ];
    eastGroveCoords.forEach((eg) => {
      const gTrunk = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.28, eg.h * 0.55, 8), mats.darkWoodFascia);
      gTrunk.position.set(eg.x, eg.h * 0.27, eg.z);
      contextGroup.add(gTrunk);

      const gCrown = new THREE.Mesh(new THREE.SphereGeometry(eg.r, 8, 6), mats.LushShrub);
      gCrown.position.set(eg.x, eg.h * 0.68, eg.z);
      contextGroup.add(gCrown);

      if (eg.redFlush) {
        const flushCrown = new THREE.Mesh(new THREE.SphereGeometry(eg.r * 0.62, 8, 6), mats.terracottaFloor);
        flushCrown.position.set(eg.x + 1.35, eg.h * 0.90, eg.z - 0.45);
        flushCrown.scale.set(1.15, 0.65, 1.10);
        contextGroup.add(flushCrown);
      }
    });

    const seBreezewayTh = (19.5 * Math.PI) / 180.0;
    const seBreezewayCanopy = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.36, 5.8), mats.whiteMullion);
    seBreezewayCanopy.position.set(41.8 * Math.cos(seBreezewayTh), 3.65, 41.8 * Math.sin(seBreezewayTh));
    seBreezewayCanopy.rotation.y = -seBreezewayTh;
    contextGroup.add(seBreezewayCanopy);

    for (const degCol of [15.5, 23.5, 29.5]) {
      const thCol = (degCol * Math.PI) / 180.0;
      const sePiloti = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 4.2, 12), mats.whiteMullion);
      sePiloti.position.set(41.5 * Math.cos(thCol), 2.1, 41.5 * Math.sin(thCol));
      contextGroup.add(sePiloti);
    }

    const cantTh = (22.5 * Math.PI) / 180.0;
    const cantGroup = new THREE.Group();
    cantGroup.position.set(43.8 * Math.cos(cantTh), 11.4, 43.8 * Math.sin(cantTh));
    cantGroup.rotation.y = -cantTh;
    const cantSlab = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.38, 2.8), mats.whiteMullion);
    cantSlab.position.set(0.6, 0.0, 0.0);
    cantGroup.add(cantSlab);
    const cantGlassBox = new THREE.Mesh(new THREE.BoxGeometry(2.2, 3.8, 2.4), mats.windowGlass);
    cantGlassBox.position.set(0.5, 2.05, 0.0);
    cantGroup.add(cantGlassBox);
    const cantTopCap = new THREE.Mesh(new THREE.BoxGeometry(2.5, 0.28, 2.7), mats.whiteMullion);
    cantTopCap.position.set(0.55, 4.05, 0.0);
    cantGroup.add(cantTopCap);
    contextGroup.add(cantGroup);

    // DRPM South-East Driveway, Open Iron Gates, Bamboo-Staked Mulch Mound, & SE 25.2m Terracotta Brick Tower
    const n16Driveway = new THREE.Mesh(new THREE.BoxGeometry(14.5, 0.10, 6.2), mats.asphaltRoad);
    n16Driveway.position.set(39.2, 0.088, 31.2);
    n16Driveway.rotation.y = -0.62;
    contextGroup.add(n16Driveway);

    const bambooMoundGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = (48.5 * Math.PI) / 180.0 + u * ((47.5 * Math.PI) / 180.0);
        const r = 45.8 + v * (52.0 - 45.8);
        const taperEnd = Math.min(1.0, (1.0 - u) * 4.5);
        const y = Math.sin(v * Math.PI) * 2.15 * taperEnd + 0.15;
        p.set(r * Math.cos(th), y, r * Math.sin(th));
        n.set(Math.cos(th) * 0.35, 0.93, Math.sin(th) * 0.35).normalize();
      },
      0.0, 1.0, 0.0, 1.0, 36, 6
    );
    contextGroup.add(new THREE.Mesh(bambooMoundGeo, mats.mulchSoil));

    for (let bm = 0; bm < 56; bm++) {
      const thBm = (49.0 + bm * 0.82) * (Math.PI / 180.0);
      const stakeH = bm > 44 ? 0.65 * (1.0 - (bm - 44) / 14.0) : 0.65;
      const stake = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.05, stakeH, 6), mats.copperACP);
      stake.position.set(51.85 * Math.cos(thBm), 0.12 + stakeH * 0.5, 51.85 * Math.sin(thBm));
      contextGroup.add(stake);
    }

    for (const [txM, tzM, rM] of [
      [24.5, 44.2, 3.6],
      [18.5, 47.5, 3.7],
      [12.8, 49.8, 3.4],
      [7.2, 50.8, 3.5],
      [1.8, 51.0, 3.3]
    ]) {
      const mTreeTrunk = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.34, 4.8, 8), mats.darkWoodFascia);
      mTreeTrunk.position.set(txM, 2.6, tzM);
      contextGroup.add(mTreeTrunk);
      const mTreeCrown = new THREE.Mesh(new THREE.SphereGeometry(rM, 8, 6), mats.LushShrub);
      mTreeCrown.position.set(txM, 6.2, tzM);
      contextGroup.add(mTreeCrown);
    }

    const swWingGroup = new THREE.Group();
    swWingGroup.position.set(14.2, 0.0, 41.8);
    swWingGroup.rotation.y = -0.32;
    contextGroup.add(swWingGroup);

    for (const xPil of [-3.6, 0.0, 3.6]) {
      const swPiloti = new THREE.Mesh(new THREE.BoxGeometry(0.68, 3.8, 0.68), mats.whiteMullion);
      swPiloti.position.set(xPil, 1.9, 0.0);
      swWingGroup.add(swPiloti);
    }
    const swGroundWall = new THREE.Mesh(new THREE.BoxGeometry(8.6, 3.6, 0.45), mats.creamWall);
    swGroundWall.position.set(0.0, 1.8, -0.8);
    swWingGroup.add(swGroundWall);
    const swCantileverFloor = new THREE.Mesh(new THREE.BoxGeometry(9.4, 0.42, 2.6), mats.whiteMullion);
    swCantileverFloor.position.set(0.0, 3.85, 0.2);
    swWingGroup.add(swCantileverFloor);
    const swCornerGlass = new THREE.Mesh(new THREE.BoxGeometry(9.2, 2.85, 2.4), mats.windowGlass);
    swCornerGlass.position.set(0.0, 5.48, 0.2);
    swWingGroup.add(swCornerGlass);
    const swUpperLouverBlock = new THREE.Mesh(new THREE.BoxGeometry(9.4, 8.8, 2.5), mats.ilrcConcrete);
    swUpperLouverBlock.position.set(0.0, 11.3, 0.2);
    swWingGroup.add(swUpperLouverBlock);

    const ilrcEnd = ilrcStart + ilrcArcSpan;
    const endTowerSE = new THREE.Mesh(new THREE.BoxGeometry(8.5, 25.2, 16.5), mats.ilrcBrickWing);
    endTowerSE.position.set(38.2 * Math.cos(ilrcEnd), 12.6, 38.2 * Math.sin(ilrcEnd));
    endTowerSE.rotation.y = -ilrcEnd;
    contextGroup.add(endTowerSE);

    const seTowerCrown = new THREE.Mesh(new THREE.BoxGeometry(9.2, 3.4, 18.2), mats.ilrcBrickWing);
    seTowerCrown.position.set(39.2 * Math.cos(ilrcEnd), 23.6, 39.2 * Math.sin(ilrcEnd));
    seTowerCrown.rotation.y = -ilrcEnd;
    seTowerCrown.rotation.x = 0.18;
    contextGroup.add(seTowerCrown);

    for (const yHood of [11.5, 15.2, 18.9]) {
      const hoodBox = new THREE.Mesh(new THREE.BoxGeometry(1.65, 1.55, 0.95), mats.stoneRetaining);
      hoodBox.position.set(38.2 * Math.cos(ilrcEnd) - 2.2, yHood, 38.2 * Math.sin(ilrcEnd) + 5.8);
      hoodBox.rotation.y = -ilrcEnd;
      contextGroup.add(hoodBox);

      const hoodPane = new THREE.Mesh(new THREE.BoxGeometry(1.35, 1.22, 1.02), mats.windowGlass);
      hoodPane.position.copy(hoodBox.position);
      hoodPane.rotation.y = -ilrcEnd;
      contextGroup.add(hoodPane);

      const hoodCanopy = new THREE.Mesh(new THREE.BoxGeometry(1.85, 0.16, 1.25), mats.whiteMullion);
      hoodCanopy.position.set(38.2 * Math.cos(ilrcEnd) - 2.3, yHood + 0.85, 38.2 * Math.sin(ilrcEnd) + 6.0);
      hoodCanopy.rotation.y = -ilrcEnd;
      hoodCanopy.rotation.x = 0.28;
      contextGroup.add(hoodCanopy);

      const balcFrame = new THREE.Mesh(new THREE.BoxGeometry(4.2, 2.8, 0.65), mats.whiteMullion);
      balcFrame.position.set(42.2 * Math.cos(ilrcEnd - 0.04), yHood, 42.2 * Math.sin(ilrcEnd - 0.04));
      balcFrame.rotation.y = -ilrcEnd;
      contextGroup.add(balcFrame);
    }

    // 7. BAKE ENTIRE HIERARCHY INTO { horizontalGeo, verticalGeo } FOR +0 EXTRA DRAW CALLS!
    root.updateMatrixWorld(true);

    const hPos = [];
    const hNorm = [];
    const hUv = [];
    const hCol = [];
    const hIdx = [];
    let hCount = 0;

    const vPos = [];
    const vNorm = [];
    const vUv = [];
    const vCol = [];
    const vIdx = [];
    let vCount = 0;

    const tmpV = new THREE.Vector3();
    const tmpN = new THREE.Vector3();

    root.traverse((obj) => {
      if (!obj.isMesh || !obj.geometry || !obj.material) return;
      const mat = obj.material;
      const isZebra = mat.userData.batchKind === 'Z';
      const rgb = mat.userData.rgb || [0.85, 0.85, 0.85];
      const uvSentinel = mat.userData.uvSentinel;
      const keepCurbUV = Boolean(obj.userData.keepCurbUV);

      const g = obj.geometry;
      const posAttr = g.attributes.position;
      if (!posAttr) return;
      if (!g.attributes.normal) g.computeVertexNormals();
      const normAttr = g.attributes.normal;
      const uvAttr = g.attributes.uv;
      const indexAttr = g.index;

      const base = isZebra ? vCount : hCount;
      const tPos = isZebra ? vPos : hPos;
      const tNorm = isZebra ? vNorm : hNorm;
      const tUv = isZebra ? vUv : hUv;
      const tCol = isZebra ? vCol : hCol;
      const tIdx = isZebra ? vIdx : hIdx;

      for (let i = 0; i < posAttr.count; i++) {
        tmpV.set(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i)).applyMatrix4(obj.matrixWorld);
        tmpN.set(normAttr.getX(i), normAttr.getY(i), normAttr.getZ(i)).transformDirection(obj.matrixWorld);

        tPos.push(tmpV.x, tmpV.y, tmpV.z);
        tNorm.push(tmpN.x, tmpN.y, tmpN.z);
        tCol.push(rgb[0], rgb[1], rgb[2]);

        if (uvSentinel) {
          tUv.push(uvSentinel[0], uvSentinel[1]);
        } else if (isZebra && keepCurbUV && uvAttr) {
          tUv.push(uvAttr.getX(i), uvAttr.getY(i));
        } else if (isZebra) {
          tUv.push(0.055, 0.5);
        } else if (uvAttr) {
          tUv.push(uvAttr.getX(i), Math.max(0.0, uvAttr.getY(i)));
        } else {
          tUv.push(0.5, 0.5);
        }
      }

      if (indexAttr) {
        for (let i = 0; i < indexAttr.count; i++) {
          tIdx.push(base + indexAttr.getX(i));
        }
      } else {
        for (let i = 0; i < posAttr.count; i++) {
          tIdx.push(base + i);
        }
      }

      if (isZebra) {
        vCount += posAttr.count;
      } else {
        hCount += posAttr.count;
      }
    });

    const horizontalGeo = new THREE.BufferGeometry();
    horizontalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(hPos), 3));
    horizontalGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(hNorm), 3));
    horizontalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(hUv), 2));
    horizontalGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(hCol), 3));
    horizontalGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(hIdx), 1));

    const verticalGeo = new THREE.BufferGeometry();
    verticalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vPos), 3));
    verticalGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(vNorm), 3));
    verticalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(vUv), 2));
    verticalGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(vCol), 3));
    verticalGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(vIdx), 1));

    return { horizontalGeo, verticalGeo };
  }

  /**
   * Transforms local complex coordinates (x, y, z) to Pulau U world coordinates.
   */
  static localToWorld(lx, ly, lz, extraRotY = 0.0) {
    const s = TerrainDEMLoader.DPPD_SCALE;
    const totalRot = TerrainDEMLoader.DPPD_Y_ROT + extraRotY;
    const c = Math.cos(totalRot);
    const sn = Math.sin(totalRot);
    return {
      x: TerrainDEMLoader.DPPD_CX + s * (lx * c + lz * sn),
      y: TerrainDEMLoader.DPPD_PLATEAU_Y + s * ly,
      z: TerrainDEMLoader.DPPD_CZ + s * (-lx * sn + lz * c)
    };
  }

  /**
   * Returns exact world-space solid-body colliders for Kompleks DPPD, Perpustakaan Lama (ITTC),
   * STP-UI 8-Lantai, ILRC/DRPM 4-Lantai, Menara Bata Terakota, & Parked Campus Vehicles.
   */
  static getSolidColliders() {
    const cols = [];
    const s = TerrainDEMLoader.DPPD_SCALE;
    const baseY = TerrainDEMLoader.DPPD_PLATEAU_Y;
    const rotRoot = TerrainDEMLoader.DPPD_Y_ROT;
    const rotEntrance = -0.25 * Math.PI;

    // 1. Central ITTC / Perpustakaan Lama UI 4-Story Courtyard Building (Centered at local 0, 0)
    cols.push({
      id: 'DPPD_ITTC_PerpustakaanLama',
      type: 'OBB',
      cx: TerrainDEMLoader.DPPD_CX,
      cz: TerrainDEMLoader.DPPD_CZ,
      halfW: 17.4 * s,
      halfL: 17.4 * s,
      yaw: rotRoot,
      minY: baseY,
      maxY: baseY + 17.8 * s
    });

    // 2. Gedung DPPD Curved Heritage Wing (7 overlapping cylindrical/OBB arc sectors + Front Porch & Gable)
    const alphaMin = (-52.0 * Math.PI) / 180.0;
    const alphaMax = (46.0 * Math.PI) / 180.0;
    for (let k = 0; k < 7; k++) {
      const a = alphaMin + ((k + 0.5) / 7) * (alphaMax - alphaMin);
      const rMid = 36.2;
      const wp = DPPDComplexBuilder.localToWorld(rMid * Math.sin(a), 0, rMid * Math.cos(a), rotEntrance);
      cols.push({
        id: `DPPD_HeritageWingSector_${k}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 9.2 * s,
        minY: baseY,
        maxY: baseY + 11.6 * s
      });
    }
    const porchW = DPPDComplexBuilder.localToWorld(0, 0, 44.5, rotEntrance);
    cols.push({
      id: 'DPPD_FrontCentralGableAndPorch',
      type: 'CYL',
      x: porchW.x,
      z: porchW.z,
      r: 7.4 * s,
      minY: baseY,
      maxY: baseY + 15.6 * s
    });

    // 3. STP-UI 8-Story Curved Tower (-129° .. -38°) + North Drop-Off Portico
    const stpStart = (-129.0 * Math.PI) / 180.0;
    const stpSpan = (91.0 * Math.PI) / 180.0;
    for (let k = 0; k < 8; k++) {
      const th = stpStart + ((k + 0.5) / 8) * stpSpan;
      const wp = DPPDComplexBuilder.localToWorld(37.4 * Math.cos(th), 0, 37.4 * Math.sin(th), 0.0);
      cols.push({
        id: `DPPD_STPUI_TowerSector_${k}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 7.2 * s,
        minY: baseY,
        maxY: baseY + 25.6 * s
      });
    }
    const porticoTh = (-79.5 * Math.PI) / 180.0;
    const porticoW = DPPDComplexBuilder.localToWorld(46.8 * Math.cos(porticoTh), 0, 46.8 * Math.sin(porticoTh), 0.0);
    cols.push({
      id: 'DPPD_STPUI_NorthPortico',
      type: 'CYL',
      x: porticoW.x,
      z: porticoW.z,
      r: 5.4 * s,
      minY: baseY,
      maxY: baseY + 4.8 * s
    });

    // 4. ILRC / DRPM 4-Story East Laboratory Wing (-31.5° .. +50°) + SE 25.2m Terracotta Brick Tower
    const eastStart = (-31.5 * Math.PI) / 180.0;
    const eastSpan = (81.5 * Math.PI) / 180.0;
    for (let k = 0; k < 7; k++) {
      const th = eastStart + ((k + 0.5) / 7) * eastSpan;
      const wp = DPPDComplexBuilder.localToWorld(37.0 * Math.cos(th), 0, 37.0 * Math.sin(th), 0.0);
      cols.push({
        id: `DPPD_ILRC_EastLabSector_${k}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 6.8 * s,
        minY: baseY,
        maxY: baseY + 15.4 * s
      });
    }
    const ilrcEnd = (-129.0 + 179.0) * (Math.PI / 180.0);
    const seTowerW = DPPDComplexBuilder.localToWorld(38.5 * Math.cos(ilrcEnd), 0, 38.5 * Math.sin(ilrcEnd), 0.0);
    cols.push({
      id: 'DPPD_ILRC_SEBrickTower',
      type: 'CYL',
      x: seTowerW.x,
      z: seTowerW.z,
      r: 7.6 * s,
      minY: baseY,
      maxY: baseY + 25.4 * s
    });

    // 5. Parked Campus Vehicles in West Parking Bay
    const parkedLocal = [
      [-31.6, -26.4],
      [-31.5, -20.2],
      [-31.6, -14.0],
      [-38.8, -18.5]
    ];
    parkedLocal.forEach(([px, pz], idx) => {
      const wp = DPPDComplexBuilder.localToWorld(px, 0, pz, 0.0);
      cols.push({
        id: `DPPD_ParkedCar_${idx}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 2.35 * s,
        minY: baseY,
        maxY: baseY + 1.55 * s
      });
    });

    return cols;
  }
}
