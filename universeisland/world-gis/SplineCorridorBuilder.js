import * as THREE from 'three';
import { NexusBatchOptimizer } from '../core/BatchRenderer.js';
import {
  SURFACE_PROFILE,
  createZebraCurbTexture,
  createSharedSurfaceUniforms,
  injectOmniScaleSurfaceShader
} from '../core/ProceduralTextureFactory.js';
import { TerrainDEMLoader } from './TerrainDEMLoader.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.3 — 10x Ultra-High Level of Detail Edition)
 * ./world-gis/SplineCorridorBuilder.js
 *
 * Builds the 10x High-Detail Campus Road, Botanical Rotunda, Street Furniture, & Covered Walkway Network:
 * 1. Lapangan Rotunda UI Circular Boulevard (288 segments, 4-strip cambered asphalt, 3D beveled black-white
 *    zebra curbs, and multi-tier sculpted 3D botanical topiary + flowering Ixora centerpiece).
 * 2. West Campus Boulevard, West Courtyard Parking Lane, East Lakeside Brick Turnaround, & Danau Kenanga Promenade.
 * 3. Long South-East Terracotta-Roofed Covered Campus Walkway (120 segments, 3D corrugated tile courses,
 *    cylindrical ridge roll, 60 exposed ceiling rafter pairs, & 62 sculpted plinth/capital columns).
 * 4. 14 Teak-and-Concrete Campus Park Benches & 24 Sculpted Pedestrian Bollards.
 *
 * Batched via NexusBatchOptimizer into STRICTLY 2 DRAW CALLS TOTAL!
 */
export class SplineCorridorBuilder {
  static SKIRT_BOTTOM_Y = -2.5;
  static ROTUNDA_CX = -2.5;
  static ROTUNDA_CZ = 106.5;

  static _createHighPrecisionSpline(points, closed = false) {
    const curve = new THREE.CatmullRomCurve3(points, closed);
    curve.arcLengthDivisions = 900;
    return curve;
  }

  static _sampleCurveFrame(curve, u, targetPos, targetTan, targetSide) {
    const uSafe = !curve.closed && u >= 1.0 ? 1.0 - 1e-6 : u;
    curve.getPointAt(uSafe, targetPos);

    const du = 0.0012;
    const pPrev = new THREE.Vector3();
    const pNext = new THREE.Vector3();
    if (curve.closed) {
      const u0 = (u - du + 1.0) % 1.0;
      const u1 = (u + du) % 1.0;
      curve.getPointAt(u0, pPrev);
      curve.getPointAt(u1, pNext);
    } else {
      const u0 = Math.max(0.0, Math.min(1.0 - 1e-6, uSafe - du));
      const u1 = Math.max(0.0, Math.min(1.0 - 1e-6, uSafe + du));
      curve.getPointAt(u0, pPrev);
      curve.getPointAt(u1, pNext);
    }

    targetTan.subVectors(pNext, pPrev);
    targetTan.y = 0.0;
    if (targetTan.lengthSq() < 1e-10) {
      targetTan.set(0.0, 0.0, 1.0);
    } else {
      targetTan.normalize();
    }

    targetSide.set(targetTan.z, 0.0, -targetTan.x).normalize();
  }

  /**
   * Lofts a spline corridor into two high-detail separated vertex-colored geometries:
   * - horizontalGeo: 4-strip cambered asphalt / brick / stone deck with gutter shading (Material 1)
   * - verticalGeo: 3D chamfer-beveled curb profile + risers & -2.5m sub-terrain skirts (Material 2)
   */
  static _loftSeparatedCorridor(curve, options = {}) {
    const {
      segments = 120,
      halfWidth = 4.0,
      curbWidth = 0.45,
      curbRise = 0.14,
      skirtBottomY = SplineCorridorBuilder.SKIRT_BOTTOM_Y,
      deckColor = [0.22, 0.23, 0.25],
      curbTint = [0.96, 0.95, 0.93],
      zebraCurbs = false,
      zebraFreq = 0.125
    } = options;

    const dPos = [];
    const dNorm = [];
    const dUv = [];
    const dCol = [];
    const dIdx = [];

    const cPos = [];
    const cNorm = [];
    const cUv = [];
    const cCol = [];
    const cIdx = [];

    const pushD = (x, y, z, nx, ny, nz, col = deckColor) => {
      const idx = dPos.length / 3;
      dPos.push(x, y, z);
      dNorm.push(nx, ny, nz);
      dUv.push(x * 0.2, z * 0.2);
      dCol.push(col[0], col[1], col[2]);
      return idx;
    };

    const pushC = (x, y, z, nx, ny, nz, u, v, col = curbTint) => {
      const idx = cPos.length / 3;
      cPos.push(x, y, z);
      cNorm.push(nx, ny, nz);
      cUv.push(u, v);
      cCol.push(col[0], col[1], col[2]);
      return idx;
    };

    const frames = [];
    const pTmp = new THREE.Vector3();
    const tTmp = new THREE.Vector3();
    const sTmp = new THREE.Vector3();
    let cumulativeArc = 0.0;

    for (let i = 0; i <= segments; i++) {
      const u = i / segments;
      if (curve.closed && i === segments && frames.length > 0) {
        const f0 = frames[0];
        frames.push({
          pos: f0.pos.clone(),
          tan: f0.tan.clone(),
          side: f0.side.clone(),
          arc: cumulativeArc + frames[frames.length - 1].pos.distanceTo(f0.pos)
        });
      } else {
        SplineCorridorBuilder._sampleCurveFrame(curve, u, pTmp, tTmp, sTmp);
        if (i > 0) {
          cumulativeArc += frames[i - 1].pos.distanceTo(pTmp);
        }
        frames.push({
          pos: pTmp.clone(),
          tan: tTmp.clone(),
          side: sTmp.clone(),
          arc: cumulativeArc
        });
      }
    }

    let activeZebraFreq = zebraFreq;
    if (zebraCurbs && curve.closed && cumulativeArc > 1.0) {
      const wholeCycles = Math.max(4, Math.round(cumulativeArc * zebraFreq));
      activeZebraFreq = wholeCycles / cumulativeArc;
    }

    const curbU = (arc) => (zebraCurbs ? arc * activeZebraFreq : 0.055);

    // 1. 4-Strip Cambered Horizontal Road/Walkway Deck (subtle 1.6cm drainage crown + gutter AO!)
    const lateralFracs = [-1.0, -0.55, 0.0, 0.55, 1.0];
    const deckRows = [];
    for (let i = 0; i <= segments; i++) {
      const { pos, side } = frames[i];
      const row = [];
      for (let k = 0; k < lateralFracs.length; k++) {
        const f = lateralFracs[k];
        const camber = (1.0 - f * f) * 0.016;
        const gutterShade = 1.0 - 0.14 * Math.pow(Math.abs(f), 3.0);
        const slopeX = (-2.0 * f * 0.016) / Math.max(0.5, halfWidth);
        const nx = side.x * slopeX;
        const nz = side.z * slopeX;
        const invN = 1.0 / Math.hypot(nx, 1.0, nz);
        row.push(
          pushD(
            pos.x + side.x * halfWidth * f,
            pos.y + camber,
            pos.z + side.z * halfWidth * f,
            nx * invN,
            invN,
            nz * invN,
            [deckColor[0] * gutterShade, deckColor[1] * gutterShade, deckColor[2] * gutterShade]
          )
        );
      }
      deckRows.push(row);
    }

    for (let i = 0; i < segments; i++) {
      for (let k = 0; k < lateralFracs.length - 1; k++) {
        const a = deckRows[i][k];
        const b = deckRows[i][k + 1];
        const c = deckRows[i + 1][k + 1];
        const d = deckRows[i + 1][k];
        dIdx.push(a, d, c, a, c, b);
      }
    }

    // 2. 3D Chamfer-Beveled Curb Caps (Outer Chamfer + Crown + Inner Road Bevel)
    const bevelW = Math.min(0.055, curbWidth * 0.18);
    const bevelDrop = Math.min(0.035, curbRise * 0.28);
    const outerSpan = halfWidth + curbWidth;

    const buildBeveledCurbTop = (sideSign) => {
      // Ordered from left-to-right along +side so winding (a, d, c, a, c, b) always has +Y normal!
      const offsets =
        sideSign < 0
          ? [
              { lat: -outerSpan, dy: curbRise - bevelDrop, nSide: -0.55, ny: 0.835, v: 0.1 },
              { lat: -outerSpan + bevelW, dy: curbRise, nSide: 0.0, ny: 1.0, v: 0.22 },
              { lat: -halfWidth - bevelW, dy: curbRise, nSide: 0.0, ny: 1.0, v: 0.78 },
              { lat: -halfWidth, dy: curbRise - bevelDrop, nSide: 0.55, ny: 0.835, v: 0.9 }
            ]
          : [
              { lat: halfWidth, dy: curbRise - bevelDrop, nSide: -0.55, ny: 0.835, v: 0.1 },
              { lat: halfWidth + bevelW, dy: curbRise, nSide: 0.0, ny: 1.0, v: 0.22 },
              { lat: outerSpan - bevelW, dy: curbRise, nSide: 0.0, ny: 1.0, v: 0.78 },
              { lat: outerSpan, dy: curbRise - bevelDrop, nSide: 0.55, ny: 0.835, v: 0.9 }
            ];

      const rows = [];
      for (let i = 0; i <= segments; i++) {
        const { pos, side, arc } = frames[i];
        const uC = curbU(arc);
        const r = [];
        for (let k = 0; k < offsets.length; k++) {
          const o = offsets[k];
          r.push(
            pushC(
              pos.x + side.x * o.lat,
              pos.y + o.dy,
              pos.z + side.z * o.lat,
              side.x * o.nSide,
              o.ny,
              side.z * o.nSide,
              uC,
              o.v
            )
          );
        }
        rows.push(r);
      }

      for (let i = 0; i < segments; i++) {
        for (let k = 0; k < offsets.length - 1; k++) {
          const a = rows[i][k];
          const b = rows[i][k + 1];
          const c = rows[i + 1][k + 1];
          const d = rows[i + 1][k];
          cIdx.push(a, d, c, a, c, b);
        }
      }
    };

    buildBeveledCurbTop(-1);
    buildBeveledCurbTop(1);

    // 3. Vertical Curb Risers & -2.5m Sub-Terrain Skirts (Hukum Emas 5 & 10)
    const addVerticalRibbon = (lateralOffset, yTopRel, useAbsBottom, bottomVal, normalSign, isSkirt) => {
      const topRow = [];
      const botRow = [];
      const col = isSkirt ? [0.42, 0.41, 0.39] : curbTint;
      for (let i = 0; i <= segments; i++) {
        const { pos, side, arc } = frames[i];
        const px = pos.x + side.x * lateralOffset;
        const pz = pos.z + side.z * lateralOffset;
        const yTop = pos.y + yTopRel;
        const yBot = useAbsBottom ? bottomVal : pos.y + bottomVal;

        const nx = side.x * normalSign;
        const nz = side.z * normalSign;
        const uC = isSkirt ? 0.055 : curbU(arc);

        topRow.push(pushC(px, yTop, pz, nx, 0.0, nz, uC, 0.15, col));
        botRow.push(pushC(px, yBot, pz, nx, 0.0, nz, uC, 0.92, col));
      }

      for (let i = 0; i < segments; i++) {
        const t0 = topRow[i];
        const t1 = topRow[i + 1];
        const b0 = botRow[i];
        const b1 = botRow[i + 1];
        if (normalSign < 0) {
          cIdx.push(t0, b0, b1, t0, b1, t1);
        } else {
          cIdx.push(t0, t1, b1, t0, b1, b0);
        }
      }
    };

    const riserTop = curbRise - bevelDrop;
    addVerticalRibbon(-outerSpan, riserTop, true, skirtBottomY, -1.0, true);
    addVerticalRibbon(-halfWidth, riserTop, false, 0.0, 1.0, false);
    addVerticalRibbon(halfWidth, riserTop, false, 0.0, -1.0, false);
    addVerticalRibbon(outerSpan, riserTop, true, skirtBottomY, 1.0, true);

    if (!curve.closed) {
      const addEndCap = (frame, isEnd) => {
        const { pos, tan, side } = frame;
        const nx = isEnd ? tan.x : -tan.x;
        const nz = isEnd ? tan.z : -tan.z;
        const yTop = pos.y + curbRise;
        const yBot = skirtBottomY;

        const xL = pos.x - side.x * outerSpan;
        const zL = pos.z - side.z * outerSpan;
        const xR = pos.x + side.x * outerSpan;
        const zR = pos.z + side.z * outerSpan;

        const iTL = pushC(xL, yTop, zL, nx, 0.0, nz, 0.055, 0.2);
        const iTR = pushC(xR, yTop, zR, nx, 0.0, nz, 0.055, 0.2);
        const iBR = pushC(xR, yBot, zR, nx, 0.0, nz, 0.055, 0.9);
        const iBL = pushC(xL, yBot, zL, nx, 0.0, nz, 0.055, 0.9);

        if (isEnd) {
          cIdx.push(iTL, iBL, iBR, iTL, iBR, iTR);
        } else {
          cIdx.push(iTL, iTR, iBR, iTL, iBR, iBL);
        }
      };

      addEndCap(frames[0], false);
      addEndCap(frames[segments], true);
    }

    const horizontalGeo = new THREE.BufferGeometry();
    horizontalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(dPos), 3));
    horizontalGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(dNorm), 3));
    horizontalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(dUv), 2));
    horizontalGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(dCol), 3));
    horizontalGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(dIdx), 1));

    const verticalGeo = new THREE.BufferGeometry();
    verticalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cPos), 3));
    verticalGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(cNorm), 3));
    verticalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(cUv), 2));
    verticalGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(cCol), 3));
    verticalGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(cIdx), 1));

    return { horizontalGeo, verticalGeo };
  }

  /**
   * Builds the 10x Multi-Tier Sculpted Botanical Topiary Shrub Rings, Flowering Ixora Garland,
   * Dark Topsoil Bed, & Molded Circular Stone Planter in Lapangan Rotunda UI (-2.5, +106.5m).
   */
  static _buildRotundaBotanicalCenterpiece(
    centerX = SplineCorridorBuilder.ROTUNDA_CX,
    centerZ = SplineCorridorBuilder.ROTUNDA_CZ
  ) {
    const dPos = [];
    const dNorm = [];
    const dUv = [];
    const dCol = [];
    const dIdx = [];

    const cPos = [];
    const cNorm = [];
    const cUv = [];
    const cCol = [];
    const cIdx = [];

    const pushD = (x, y, z, nx, ny, nz, col) => {
      const idx = dPos.length / 3;
      dPos.push(x, y, z);
      dNorm.push(nx, ny, nz);
      dUv.push(x * 0.25, z * 0.25);
      dCol.push(col[0], col[1], col[2]);
      return idx;
    };

    const pushC = (x, y, z, nx, ny, nz, u, v, col = [0.94, 0.93, 0.9]) => {
      const idx = cPos.length / 3;
      cPos.push(x, y, z);
      cNorm.push(nx, ny, nz);
      cUv.push(u, v);
      cCol.push(col[0], col[1], col[2]);
      return idx;
    };

    // 1. Molded Circular Stone Planter Curb (96 radial segs: outer skirt + chamfer coping + inner wall)
    const segs = 96;
    const curbProfile = [
      { r: 5.52, y: SplineCorridorBuilder.SKIRT_BOTTOM_Y, nr: 1, ny: 0 },
      { r: 5.52, y: 0.22, nr: 1, ny: 0 },
      { r: 5.46, y: 0.28, nr: 0.35, ny: 0.93 },
      { r: 5.16, y: 0.28, nr: -0.35, ny: 0.93 },
      { r: 5.12, y: 0.14, nr: -1, ny: 0 }
    ];
    for (let s = 0; s < curbProfile.length - 1; s++) {
      const p0 = curbProfile[s];
      const p1 = curbProfile[s + 1];
      const rA = [];
      const rB = [];
      for (let i = 0; i <= segs; i++) {
        const th = (i / segs) * Math.PI * 2.0;
        const c = Math.cos(th);
        const sn = Math.sin(th);
        rA.push(pushC(centerX + c * p0.r, p0.y, centerZ + sn * p0.r, c * p0.nr, p0.ny, sn * p0.nr, 0.055, 0.3));
        rB.push(pushC(centerX + c * p1.r, p1.y, centerZ + sn * p1.r, c * p1.nr, p1.ny, sn * p1.nr, 0.055, 0.7));
      }
      for (let i = 0; i < segs; i++) {
        cIdx.push(rA[i], rB[i], rB[i + 1], rA[i], rB[i + 1], rA[i + 1]);
      }
    }

    // 2. Dark Humus Topsoil Bed inside Planter (r = 5.14m, y = 0.14m)
    const soilCenter = pushD(centerX, 0.15, centerZ, 0, 1, 0, [0.16, 0.12, 0.09]);
    const soilRing = [];
    for (let i = 0; i <= segs; i++) {
      const th = (i / segs) * Math.PI * 2.0;
      soilRing.push(
        pushD(centerX + Math.cos(th) * 5.14, 0.14, centerZ + Math.sin(th) * 5.14, 0, 1, 0, [0.15, 0.11, 0.08])
      );
    }
    for (let i = 0; i < segs; i++) {
      dIdx.push(soilCenter, soilRing[i + 1], soilRing[i]);
    }

    // Helper: Organic Leaf-Cluster Harmonic Topiary Torus Ring
    const addBotanicalTorus = (majorR, rxTube, ryTube, yBase, uSegs, vSegs, colorFn) => {
      const rows = [];
      for (let i = 0; i <= uSegs; i++) {
        const th = (i / uSegs) * Math.PI * 2.0;
        const cTh = Math.cos(th);
        const sTh = Math.sin(th);
        const row = [];
        for (let j = 0; j <= vSegs; j++) {
          const ph = (j / vSegs) * Math.PI * 2.0;
          const cPh = Math.cos(ph);
          const sPh = Math.sin(ph);
          const leafRipple =
            1.0 +
            0.075 * Math.sin(18.0 * th + 4.0 * ph) +
            0.045 * Math.cos(29.0 * th - 5.0 * ph);
          const rad = majorR + rxTube * leafRipple * cPh;
          const py = yBase + ryTube * leafRipple * sPh;
          const nx = cTh * cPh;
          const ny = sPh;
          const nz = sTh * cPh;
          const sun = 0.5 + 0.5 * Math.max(0.0, ny);
          row.push(pushD(centerX + cTh * rad, py, centerZ + sTh * rad, nx, ny, nz, colorFn(sun, th, ph)));
        }
        rows.push(row);
      }
      for (let i = 0; i < uSegs; i++) {
        for (let j = 0; j < vSegs; j++) {
          const a = rows[i][j];
          const b = rows[i + 1][j];
          const c = rows[i + 1][j + 1];
          const d = rows[i][j + 1];
          dIdx.push(a, d, c, a, c, b);
        }
      }
    };

    // 3. Outer Manicured Green Topiary Torus Ring (R = 4.22m)
    addBotanicalTorus(4.22, 0.78, 0.44, 0.30, 96, 26, (sun, th) => {
      const v = 0.02 * Math.sin(12.0 * th);
      return [0.09 + 0.09 * sun + v, 0.25 + 0.22 * sun + v, 0.08 + 0.07 * sun];
    });

    // 4. Middle Tropical Flowering Ixora / Bougainvillea Shrub Garland (R = 2.86m)
    addBotanicalTorus(2.86, 0.48, 0.34, 0.28, 84, 20, (sun, th, ph) => {
      const blossom = Math.max(0.0, Math.sin(16.0 * th) * Math.sin(3.0 * ph)) * Math.max(0.0, sun - 0.35);
      return [
        0.11 + 0.11 * sun + 0.55 * blossom,
        0.24 + 0.19 * sun + 0.08 * blossom,
        0.09 + 0.06 * sun + 0.05 * blossom
      ];
    });

    // 5. Sculpted Multi-Ring Central Botanical Dome Mound (r = 2.05m, apex y = 0.78m)
    const domeU = 72;
    const domeV = 18;
    const domeRows = [];
    for (let iv = 0; iv <= domeV; iv++) {
      const tv = iv / domeV;
      const phi = tv * Math.PI * 0.5;
      const sPhi = Math.sin(phi);
      const cPhi = Math.cos(phi);
      const row = [];
      for (let iu = 0; iu <= domeU; iu++) {
        const th = (iu / domeU) * Math.PI * 2.0;
        const lobe = 1.0 + 0.055 * Math.sin(11.0 * th + 3.0 * phi) * sPhi;
        const r = 2.02 * sPhi * lobe;
        const py = 0.14 + 0.64 * cPhi * lobe;
        const nx = Math.cos(th) * sPhi;
        const ny = Math.max(0.2, cPhi);
        const nz = Math.sin(th) * sPhi;
        const invL = 1.0 / Math.hypot(nx, ny, nz);
        const sun = 0.45 + 0.55 * cPhi;
        row.push(
          pushD(
            centerX + Math.cos(th) * r,
            py,
            centerZ + Math.sin(th) * r,
            nx * invL,
            ny * invL,
            nz * invL,
            [0.10 + 0.11 * sun, 0.27 + 0.23 * sun, 0.08 + 0.07 * sun]
          )
        );
      }
      domeRows.push(row);
    }
    for (let iv = 0; iv < domeV; iv++) {
      for (let iu = 0; iu < domeU; iu++) {
        const a = domeRows[iv][iu];
        const b = domeRows[iv][iu + 1];
        const c = domeRows[iv + 1][iu + 1];
        const d = domeRows[iv + 1][iu];
        dIdx.push(a, d, b, b, d, c);
      }
    }

    const horizontalGeo = new THREE.BufferGeometry();
    horizontalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(dPos), 3));
    horizontalGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(dNorm), 3));
    horizontalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(dUv), 2));
    horizontalGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(dCol), 3));
    horizontalGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(dIdx), 1));

    const verticalGeo = new THREE.BufferGeometry();
    verticalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cPos), 3));
    verticalGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(cNorm), 3));
    verticalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(cUv), 2));
    verticalGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(cCol), 3));
    verticalGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(cIdx), 1));

    return { horizontalGeo, verticalGeo };
  }

  /**
   * Lofts the 10x High-Detail South-East Terracotta-Roofed Covered Campus Walkway (120 segments,
   * subdivided 3D corrugated tile courses, 3D cylindrical ridge roll, 60 exposed ceiling rafter pairs,
   * longitudinal eave beams, & 62 sculpted plinth/capital columns) using the SAME 2 materials!
   */
  static _buildCoveredCampusWalkwayRoofAndColumns(curve, segments = 120) {
    const rPos = [];
    const rNorm = [];
    const rUv = [];
    const rCol = [];
    const rIdx = [];

    const colPos = [];
    const colNorm = [];
    const colUv = [];
    const colCol = [];
    const colIdx = [];

    const tileColor = [0.69, 0.30, 0.23];
    const ridgeColor = [0.76, 0.35, 0.27];
    const whiteCol = [0.95, 0.93, 0.89];

    const pushR = (x, y, z, nx, ny, nz, col = tileColor) => {
      const idx = rPos.length / 3;
      rPos.push(x, y, z);
      rNorm.push(nx, ny, nz);
      rUv.push(x * 0.25, z * 0.25);
      rCol.push(col[0], col[1], col[2]);
      return idx;
    };

    const frames = [];
    const pTmp = new THREE.Vector3();
    const tTmp = new THREE.Vector3();
    const sTmp = new THREE.Vector3();
    for (let i = 0; i <= segments; i++) {
      SplineCorridorBuilder._sampleCurveFrame(curve, i / segments, pTmp, tTmp, sTmp);
      frames.push({ pos: pTmp.clone(), tan: tTmp.clone(), side: sTmp.clone() });
    }

    const hw = 2.15;
    const eaveY = 3.35;
    const ridgeY = 4.78;
    const dy = ridgeY - eaveY;
    const nLen = Math.hypot(hw, dy);
    const slopeCourses = 8;

    // 1. Subdivided 3D Corrugated Terracotta Roof Slopes (Left & Right)
    for (const sideSign of [1, -1]) {
      const grid = [];
      for (let i = 0; i <= segments; i++) {
        const f = frames[i];
        const u = i / segments;
        const rib = Math.cos(2.0 * Math.PI * 190.0 * u);
        const nSx = (sideSign * f.side.x * dy) / nLen;
        const nSy = hw / nLen;
        const nSz = (sideSign * f.side.z * dy) / nLen;

        const row = [];
        for (let k = 0; k <= slopeCourses; k++) {
          const ts = k / slopeCourses; // 0 ridge -> 1 eave
          const envS = Math.sin(Math.PI * ts);
          const courseStep = (0.5 - ((ts * 9.0) % 1.0)) * 0.7;
          const disp = 0.016 * envS * (0.68 * rib + 0.32 * courseStep);
          const shade = 0.94 + 0.10 * (0.5 + 0.5 * rib) - 0.04 * ts;

          const px = f.pos.x + f.side.x * (sideSign * hw * ts) + nSx * disp;
          const py = ridgeY - dy * ts + nSy * disp;
          const pz = f.pos.z + f.side.z * (sideSign * hw * ts) + nSz * disp;

          row.push(
            pushR(px, py, pz, nSx, nSy, nSz, [
              tileColor[0] * shade,
              tileColor[1] * shade,
              tileColor[2] * shade
            ])
          );
        }
        grid.push(row);
      }

      for (let i = 0; i < segments; i++) {
        for (let k = 0; k < slopeCourses; k++) {
          const a = grid[i][k];
          const b = grid[i + 1][k];
          const c = grid[i + 1][k + 1];
          const d = grid[i][k + 1];
          if (sideSign > 0) {
            rIdx.push(a, b, c, a, c, d);
          } else {
            rIdx.push(a, d, c, a, c, b);
          }
        }
      }
    }

    // 2. 3D Cylindrical Half-Round Ridge Roll ('Bubungan Genteng') along Spine
    const ridgeSegs = 8;
    const ridgeGrid = [];
    for (let i = 0; i <= segments; i++) {
      const f = frames[i];
      const collar = 1.0 + 0.14 * Math.pow(0.5 + 0.5 * Math.cos((i / segments) * Math.PI * 2.0 * 140.0), 4.0);
      const rad = 0.105 * collar;
      const row = [];
      for (let jr = 0; jr <= ridgeSegs; jr++) {
        const ang = -Math.PI * 0.55 + (jr / ridgeSegs) * Math.PI * 1.1;
        const sA = Math.sin(ang);
        const cA = Math.cos(ang);
        row.push(
          pushR(
            f.pos.x + f.side.x * sA * rad,
            ridgeY - 0.02 + cA * rad,
            f.pos.z + f.side.z * sA * rad,
            f.side.x * sA,
            cA,
            f.side.z * sA,
            ridgeColor
          )
        );
      }
      ridgeGrid.push(row);
    }
    for (let i = 0; i < segments; i++) {
      for (let jr = 0; jr < ridgeSegs; jr++) {
        const a = ridgeGrid[i][jr];
        const b = ridgeGrid[i + 1][jr];
        const c = ridgeGrid[i + 1][jr + 1];
        const d = ridgeGrid[i][jr + 1];
        rIdx.push(a, b, c, a, c, d);
      }
    }

    // Helper: Append oriented 3D box into Material 2 (colPos/colNorm/colIdx)
    const appendOrientedConcreteBox = (cx, cy, cz, wSide, hY, dTan, sideVec, tanVec, pitchSide = 0.0, shade = 1.0) => {
      const box = new THREE.BoxGeometry(wSide, hY, dTan);
      if (pitchSide !== 0.0) {
        box.rotateZ(pitchSide);
      }
      const p = box.attributes.position.array;
      const n = box.attributes.normal.array;
      const base = colPos.length / 3;
      const col = [whiteCol[0] * shade, whiteCol[1] * shade, whiteCol[2] * shade];

      for (let k = 0; k < box.attributes.position.count; k++) {
        const lx = p[k * 3];
        const ly = p[k * 3 + 1];
        const lz = p[k * 3 + 2];
        const lnx = n[k * 3];
        const lny = n[k * 3 + 1];
        const lnz = n[k * 3 + 2];

        colPos.push(
          cx + sideVec.x * lx + tanVec.x * lz,
          cy + ly,
          cz + sideVec.z * lx + tanVec.z * lz
        );
        colNorm.push(
          sideVec.x * lnx + tanVec.x * lnz,
          lny,
          sideVec.z * lnx + tanVec.z * lnz
        );
        colUv.push(0.055, 0.5);
        colCol.push(col[0], col[1], col[2]);
      }
      const idx = box.index.array;
      for (let k = 0; k < idx.length; k++) {
        colIdx.push(base + idx[k]);
      }
    };

    // 3. Sculpted Plinth/Capital Columns + Exposed 3D Ceiling Rafters & Eave Beams
    const pitch = Math.atan2(dy, hw);
    const slopeLen = Math.hypot(hw, dy);

    for (let i = 0; i <= segments; i += 2) {
      const f = frames[i];
      // Exposed transverse ceiling rafter pair ('Kasau') every 2 segments (~1.55m spacing)
      for (const sgn of [-1, 1]) {
        appendOrientedConcreteBox(
          f.pos.x + f.side.x * (sgn * hw * 0.5),
          eaveY + dy * 0.5 - 0.06,
          f.pos.z + f.side.z * (sgn * hw * 0.5),
          slopeLen,
          0.085,
          0.075,
          f.side,
          f.tan,
          -sgn * pitch,
          0.95
        );
      }

      // Twin sculpted columns every 4 segments (~3.1m bay spacing)
      if (i % 4 === 0) {
        for (const sgn of [-1.55, 1.55]) {
          const px = f.pos.x + f.side.x * sgn;
          const pz = f.pos.z + f.side.z * sgn;
          // Buried plinth base
          appendOrientedConcreteBox(px, -0.15, pz, 0.48, 0.62, 0.48, f.side, f.tan, 0.0, 0.94);
          // Plinth collar
          appendOrientedConcreteBox(px, 0.22, pz, 0.41, 0.14, 0.41, f.side, f.tan, 0.0, 0.97);
          // Main column shaft
          appendOrientedConcreteBox(px, 1.78, pz, 0.34, 3.0, 0.34, f.side, f.tan, 0.0, 1.0);
          // Stepped Nusantara capital
          appendOrientedConcreteBox(px, 3.33, pz, 0.46, 0.16, 0.46, f.side, f.tan, 0.0, 0.98);
        }
      }
    }

    // Longitudinal eave tie-beams ('Blandar') & central ridge beam ('Balok Nok')
    for (let i = 0; i < segments; i += 2) {
      const fA = frames[i];
      const fB = frames[Math.min(segments, i + 2)];
      const segLen = fA.pos.distanceTo(fB.pos) + 0.04;
      const midX = 0.5 * (fA.pos.x + fB.pos.x);
      const midZ = 0.5 * (fA.pos.z + fB.pos.z);

      appendOrientedConcreteBox(midX, ridgeY - 0.11, midZ, 0.13, 0.16, segLen, fA.side, fA.tan, 0.0, 0.94);
      for (const sgn of [-1.55, 1.55]) {
        appendOrientedConcreteBox(
          midX + fA.side.x * sgn,
          eaveY + 0.04,
          midZ + fA.side.z * sgn,
          0.22,
          0.20,
          segLen,
          fA.side,
          fA.tan,
          0.0,
          0.99
        );
      }
    }

    const horizontalGeo = new THREE.BufferGeometry();
    horizontalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(rPos), 3));
    horizontalGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(rNorm), 3));
    horizontalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(rUv), 2));
    horizontalGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(rCol), 3));
    horizontalGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(rIdx), 1));

    const verticalGeo = new THREE.BufferGeometry();
    verticalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(colPos), 3));
    verticalGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(colNorm), 3));
    verticalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(colUv), 2));
    verticalGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colCol), 3));
    verticalGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(colIdx), 1));

    return { horizontalGeo, verticalGeo };
  }

  /**
   * Builds 14 Sculpted Teak-and-Concrete Campus Park Benches & 24 Chamfered Stone Bollards
   * along the Danau Kenanga Promenade & Lapangan Rotunda UI outer pedestrian apron (0 extra Draw Calls!).
   */
  static _buildCampusPromenadeBenchesAndBollards() {
    const hPos = [];
    const hNorm = [];
    const hUv = [];
    const hCol = [];
    const hIdx = [];

    const vPos = [];
    const vNorm = [];
    const vUv = [];
    const vCol = [];
    const vIdx = [];

    const appendPart = (targetPos, targetNorm, targetUv, targetCol, targetIdx, geo, col, uvFixed = null) => {
      if (!geo.attributes.normal) geo.computeVertexNormals();
      const p = geo.attributes.position.array;
      const n = geo.attributes.normal.array;
      const base = targetPos.length / 3;
      for (let i = 0; i < geo.attributes.position.count; i++) {
        const px = p[i * 3];
        const py = p[i * 3 + 1];
        const pz = p[i * 3 + 2];
        targetPos.push(px, py, pz);
        targetNorm.push(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]);
        if (uvFixed) {
          targetUv.push(uvFixed[0], uvFixed[1]);
        } else {
          targetUv.push(px * 0.4, pz * 0.4);
        }
        targetCol.push(col[0], col[1], col[2]);
      }
      const idx = geo.index.array;
      for (let i = 0; i < idx.length; i++) {
        targetIdx.push(base + idx[i]);
      }
    };

    const addBench = (bx, yFloor, bz, yaw) => {
      const rotM = new THREE.Matrix4().makeRotationY(yaw);
      const xform = (g) => {
        g.applyMatrix4(rotM);
        g.translate(bx, yFloor, bz);
        return g;
      };

      // Sculpted architectural concrete bench legs & armrests (Material 2)
      for (const sx of [-0.72, 0.72]) {
        const leg = new THREE.BoxGeometry(0.11, 0.72, 0.56);
        leg.translate(sx, 0.22, 0.0);
        appendPart(vPos, vNorm, vUv, vCol, vIdx, xform(leg), [0.91, 0.89, 0.85], [0.055, 0.5]);

        const arm = new THREE.BoxGeometry(0.09, 0.06, 0.58);
        arm.translate(sx, 0.61, 0.02);
        appendPart(vPos, vNorm, vUv, vCol, vIdx, xform(arm), [0.94, 0.92, 0.88], [0.055, 0.5]);
      }

      // 4 Contoured Tropical Teak Wood Seat Slats & 3 Backrest Slats (Material 1)
      const teakSeat = [0.46, 0.27, 0.14];
      const teakBack = [0.50, 0.30, 0.16];
      for (let s = 0; s < 4; s++) {
        const slat = new THREE.BoxGeometry(1.66, 0.042, 0.095);
        slat.translate(0.0, 0.46, -0.15 + s * 0.115);
        appendPart(hPos, hNorm, hUv, hCol, hIdx, xform(slat), teakSeat);
      }
      for (let b = 0; b < 3; b++) {
        const back = new THREE.BoxGeometry(1.66, 0.088, 0.038);
        back.rotateX(-0.18);
        back.translate(0.0, 0.59 + b * 0.11, -0.22 - b * 0.02);
        appendPart(hPos, hNorm, hUv, hCol, hIdx, xform(back), teakBack);
      }
    };

    // 7 Lakeside Promenade Benches facing +X toward Danau Kenanga (outside NPC lane & Annex Hall!)
    for (const lz of [-48.0, -35.0, -24.0, 17.5, 27.5, 37.5, 47.5]) {
      const px = TerrainDEMLoader.getShorelinePromenadeX(lz) - 2.62;
      addBench(px, 0.04, lz, Math.PI * 0.5);
    }

    // 7 Rotunda Outer Pedestrian Apron Benches facing inward toward Lapangan Rotunda UI (-2.5, 106.5)
    for (let i = 0; i < 7; i++) {
      const th = 0.42 + (i / 6) * (Math.PI - 0.84); // Southern semicircle around Rotunda lawn
      const rBench = 35.4;
      const bx = -2.5 + Math.cos(th) * rBench;
      const bz = 106.5 + Math.sin(th) * rBench;
      const yawInward = Math.atan2(-2.5 - bx, 106.5 - bz);
      addBench(bx, 0.03, bz, yawInward);
    }

    // 24 Sculpted Octagonal Concrete Pedestrian Bollards with Brass/Gold Cap Accent
    const bollardCoords = [];
    for (let i = 0; i < 16; i++) {
      const th = (i / 16) * Math.PI * 2.0 + 0.12;
      bollardCoords.push({ x: -2.5 + Math.cos(th) * 34.65, z: 106.5 + Math.sin(th) * 34.65 });
    }
    for (const sx of [-10.2, -8.2, 3.2, 5.2]) {
      bollardCoords.push({ x: sx, z: 71.85 });
    }
    for (const lz of [-18.0, -6.0, 6.0, 14.0]) {
      bollardCoords.push({ x: TerrainDEMLoader.getShorelinePromenadeX(lz) - 0.35, z: lz });
    }

    for (const bc of bollardCoords) {
      const body = new THREE.CylinderGeometry(0.105, 0.125, 0.95, 12, 1);
      body.translate(bc.x, 0.28, bc.z);
      appendPart(vPos, vNorm, vUv, vCol, vIdx, body, [0.92, 0.91, 0.88], [0.055, 0.5]);

      const cap = new THREE.SphereGeometry(0.105, 12, 8);
      cap.scale(1.0, 0.55, 1.0);
      cap.translate(bc.x, 0.76, bc.z);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, cap, [0.76, 0.62, 0.28]);
    }

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
   * Builds the Straight ICAO Island Runway / Airstrip ("Landasan Pacu Pulau I — RWY 36/18")
   * along the southern/central lowland corridor of Pulau I at the foot of Gunung Api Pulau I:
   * - 234m x 22m Grooved Asphalt Main Runway Deck (x in [-372, -138], z = 708, y = 0.58m)
   * - 2 x 10m Blast Pads with 3D Aviation-Yellow Chevrons (x in [-382, -128] = 254m total strip)
   * - 3D White ICAO Piano-Key Thresholds, Vector Numerals "36" & "18", Aiming Points, TDZ & Centerline
   * - Twin Perpendicular Stub Taxiways (TWY A & TWY B) + 60m x 24m West Aircraft Parking Apron
   * - Elevated Runway Edge, Emerald Threshold, Red End, 4-Unit PAPI Array, & 6.8m Airfield Windsock Mast
   * Merged via NexusBatchOptimizer into the existing 2 corridor draw calls (0 extra Draw Calls!).
   */
  static _buildPulauIAirstripRunway() {
    const hPos = [];
    const hNorm = [];
    const hUv = [];
    const hCol = [];
    const hIdx = [];

    const vPos = [];
    const vNorm = [];
    const vUv = [];
    const vCol = [];
    const vIdx = [];

    const appendPart = (targetPos, targetNorm, targetUv, targetCol, targetIdx, geo, col, uvFixed = null) => {
      if (!geo.attributes.normal) geo.computeVertexNormals();
      const p = geo.attributes.position.array;
      const n = geo.attributes.normal.array;
      const uArr = geo.attributes.uv ? geo.attributes.uv.array : null;
      const base = targetPos.length / 3;
      for (let i = 0; i < geo.attributes.position.count; i++) {
        const px = p[i * 3];
        const py = p[i * 3 + 1];
        const pz = p[i * 3 + 2];
        targetPos.push(px, py, pz);
        targetNorm.push(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]);
        if (uvFixed) {
          targetUv.push(uvFixed[0], uvFixed[1]);
        } else if (uArr) {
          targetUv.push(uArr[i * 2], uArr[i * 2 + 1]);
        } else {
          targetUv.push(px * 0.25, pz * 0.25);
        }
        targetCol.push(col[0], col[1], col[2]);
      }
      if (geo.index) {
        const idx = geo.index.array;
        for (let i = 0; i < idx.length; i++) {
          targetIdx.push(base + idx[i]);
        }
      } else {
        for (let i = 0; i < geo.attributes.position.count; i++) {
          targetIdx.push(base + i);
        }
      }
    };

    const rCz = TerrainDEMLoader.RUNWAY_CZ; // 708.0
    const xMin = TerrainDEMLoader.RUNWAY_X_MIN; // -372.0
    const xMax = TerrainDEMLoader.RUNWAY_X_MAX; // -138.0
    const halfW = TerrainDEMLoader.RUNWAY_HALF_W; // 11.0
    const deckY = TerrainDEMLoader.RUNWAY_DECK_Y; // 0.58
    const blastLen = 10.0;
    const totalXMin = xMin - blastLen; // -382.0
    const totalXMax = xMax + blastLen; // -128.0
    const totalLen = totalXMax - totalXMin; // 254.0
    const midX = 0.5 * (totalXMin + totalXMax); // -255.0
    const whiteUV = [0.055, 0.5];

    const asphaltMainCol = [0.165, 0.175, 0.195];
    const asphaltShoulderCol = [0.22, 0.23, 0.25];
    const concretePlinthCol = [0.89, 0.87, 0.83];
    const whiteMarkCol = [0.95, 0.96, 0.98];
    const yellowMarkCol = [0.96, 0.77, 0.12];

    // 1. Watertight Reinforced Concrete Sub-Grade Foundation Plinth & Drainage Edge Curbs (y = -2.35m .. +0.58m)
    const plinthH = deckY + 2.35;
    const plinthGeo = new THREE.BoxGeometry(totalLen + 1.4, plinthH, (halfW + 1.75) * 2.0);
    plinthGeo.translate(midX, deckY - plinthH * 0.5 - 0.025, rCz);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, plinthGeo, concretePlinthCol, whiteUV);

    // Paved Asphalt Stabilized Shoulders (24.8m overall width) & Grooved Main ICAO Tarmac Deck (22.0m width)
    const shoulderGeo = new THREE.BoxGeometry(totalLen + 0.6, 0.045, (halfW + 1.4) * 2.0);
    shoulderGeo.translate(midX, deckY - 0.028, rCz);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, shoulderGeo, asphaltShoulderCol);

    const deckGeo = new THREE.BoxGeometry(totalLen, 0.05, halfW * 2.0);
    deckGeo.translate(midX, deckY - 0.025, rCz);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, deckGeo, asphaltMainCol);

    for (const sz of [-(halfW + 1.56), halfW + 1.56]) {
      const curb = new THREE.BoxGeometry(totalLen + 1.2, 0.09, 0.36);
      curb.translate(midX, deckY + 0.01, rCz + sz);
      appendPart(vPos, vNorm, vUv, vCol, vIdx, curb, [0.93, 0.92, 0.88], whiteUV);
    }

    // 2. Twin Stub Taxiways (TWY A at x = -270, TWY B at x = -238) & 60m x 24m West Aircraft Parking Apron
    const apronXMin = -284.0;
    const apronXMax = -224.0;
    const apronMidX = 0.5 * (apronXMin + apronXMax); // -254.0
    const apronZMin = 660.0;
    const apronZMax = 684.0;
    const apronMidZ = 0.5 * (apronZMin + apronZMax); // 672.0
    const apronLenX = apronXMax - apronXMin; // 60.0
    const apronWidZ = apronZMax - apronZMin; // 24.0

    const apronPlinth = new THREE.BoxGeometry(apronLenX + 1.2, plinthH, apronWidZ + 1.2);
    apronPlinth.translate(apronMidX, deckY - plinthH * 0.5 - 0.028, apronMidZ);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, apronPlinth, concretePlinthCol, whiteUV);

    const apronDeck = new THREE.BoxGeometry(apronLenX, 0.05, apronWidZ);
    apronDeck.translate(apronMidX, deckY - 0.027, apronMidZ);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, apronDeck, [0.19, 0.20, 0.22]);

    const twyXs = [-270.0, -238.0];
    const twyZ0 = apronZMax - 0.4; // 683.6
    const twyZ1 = rCz - halfW + 0.4; // 697.4
    const twyLenZ = twyZ1 - twyZ0;
    const twyMidZ = 0.5 * (twyZ0 + twyZ1);
    for (const tx of twyXs) {
      const twyPlinth = new THREE.BoxGeometry(12.2, plinthH, twyLenZ + 0.6);
      twyPlinth.translate(tx, deckY - plinthH * 0.5 - 0.028, twyMidZ);
      appendPart(vPos, vNorm, vUv, vCol, vIdx, twyPlinth, concretePlinthCol, whiteUV);

      const twyDeck = new THREE.BoxGeometry(11.0, 0.05, twyLenZ + 0.8);
      twyDeck.translate(tx, deckY - 0.026, twyMidZ);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, twyDeck, [0.18, 0.19, 0.21]);
    }

    // 3. Precision 3D Inlaid ICAO Annex 14 Runway, Blast Pad, Taxiway & Apron Markings (zero Z-fighting!)
    const markY = deckY + 0.012;

    // 3A. Continuous White Runway Side Stripes & Transverse Threshold Bars
    const rwyInnerLen = xMax - xMin - 1.6;
    for (const sz of [-(halfW - 0.65), halfW - 0.65]) {
      const sideStripe = new THREE.BoxGeometry(rwyInnerLen, 0.016, 0.45);
      sideStripe.translate(midX, markY, rCz + sz);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, sideStripe, whiteMarkCol);
    }
    for (const txEnd of [xMin + 0.9, xMax - 0.9]) {
      const threshBar = new THREE.BoxGeometry(0.55, 0.016, (halfW - 0.5) * 2.0);
      threshBar.translate(txEnd, markY, rCz);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, threshBar, whiteMarkCol);
    }

    // 3B. 6 White Threshold "Piano-Key" Stripes at Each End (RWY 36 South & RWY 18 North)
    const pianoOffsetsZ = [-7.8, -5.0, -2.2, 2.2, 5.0, 7.8];
    for (const endCfg of [
      { cx: xMin + 6.4 },
      { cx: xMax - 6.4 }
    ]) {
      for (const pzOff of pianoOffsetsZ) {
        const keyBar = new THREE.BoxGeometry(9.0, 0.018, 1.35);
        keyBar.translate(endCfg.cx, markY + 0.001, rCz + pzOff);
        appendPart(hPos, hNorm, hUv, hCol, hIdx, keyBar, whiteMarkCol);
      }
    }

    // 3C. 3D Vector 7-Segment ICAO Runway Designation Numerals: "36" (South, facing +X) & "18" (North, facing -X)
    const addRunwayDigit = (ch, cx, cz, dirX, rightZ) => {
      const H = 6.6;
      const W = 3.1;
      const st = 0.58;
      const halfH = H * 0.5;
      const halfWDigit = W * 0.5;

      const putSeg = (uCenter, vCenter, uLen, vLen) => {
        const seg = new THREE.BoxGeometry(uLen, 0.02, vLen);
        seg.translate(cx + dirX * uCenter, markY + 0.002, cz + rightZ * vCenter);
        appendPart(hPos, hNorm, hUv, hCol, hIdx, seg, whiteMarkCol);
      };

      if (ch === '1') {
        putSeg(0.0, 0.0, H, st * 1.08);
        putSeg(halfH - st * 0.5, -0.45, st, 0.95);
        return;
      }

      const segMap = {
        '3': ['a', 'b', 'c', 'd', 'g'],
        '6': ['a', 'c', 'd', 'e', 'f', 'g'],
        '8': ['a', 'b', 'c', 'd', 'e', 'f', 'g']
      }[ch] || [];

      const vertSegLen = halfH + st * 0.35;
      const vertCenter = halfH * 0.5;
      for (const s of segMap) {
        if (s === 'a') putSeg(halfH - st * 0.5, 0.0, st, W);
        else if (s === 'g') putSeg(0.0, 0.0, st, W * 0.94);
        else if (s === 'd') putSeg(-(halfH - st * 0.5), 0.0, st, W);
        else if (s === 'b') putSeg(vertCenter, halfWDigit - st * 0.5, vertSegLen, st);
        else if (s === 'c') putSeg(-vertCenter, halfWDigit - st * 0.5, vertSegLen, st);
        else if (s === 'f') putSeg(vertCenter, -(halfWDigit - st * 0.5), vertSegLen, st);
        else if (s === 'e') putSeg(-vertCenter, -(halfWDigit - st * 0.5), vertSegLen, st);
      }
    };

    // South Threshold: "36" (read looking +X North toward Gunung Api Pulau I!)
    addRunwayDigit('3', xMin + 19.5, rCz - 2.35, 1.0, 1.0);
    addRunwayDigit('6', xMin + 19.5, rCz + 2.35, 1.0, 1.0);

    // North Threshold: "18" (read looking -X South from volcano flank!)
    addRunwayDigit('1', xMax - 19.5, rCz - 2.05 * -1.0, -1.0, -1.0);
    addRunwayDigit('8', xMax - 19.5, rCz + 2.05 * -1.0, -1.0, -1.0);

    // 3D. Touchdown Zone (TDZ) Twin Bars, Aiming Point Blocks, & Centerline Dashes
    for (const tdzX of [xMin + 32.0, xMax - 32.0]) {
      for (const sgn of [-1, 1]) {
        for (const zOff of [4.3, 6.2]) {
          const tdz = new THREE.BoxGeometry(8.2, 0.018, 0.95);
          tdz.translate(tdzX, markY + 0.001, rCz + sgn * zOff);
          appendPart(hPos, hNorm, hUv, hCol, hIdx, tdz, whiteMarkCol);
        }
      }
    }
    for (const aimX of [xMin + 49.0, xMax - 49.0]) {
      for (const sgn of [-1, 1]) {
        const aimBlock = new THREE.BoxGeometry(16.0, 0.018, 2.55);
        aimBlock.translate(aimX, markY + 0.001, rCz + sgn * 5.35);
        appendPart(hPos, hNorm, hUv, hCol, hIdx, aimBlock, whiteMarkCol);
      }
    }
    for (let dx = xMin + 29.0; dx <= xMax - 29.0; dx += 15.6) {
      const dash = new THREE.BoxGeometry(9.2, 0.018, 0.52);
      dash.translate(dx, markY + 0.001, rCz);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, dash, whiteMarkCol);
    }

    // 3E. Aviation-Yellow Blast Pad Overrun Chevrons (3 V-Chevrons on Each 10m Blast Pad)
    const addBlastChevron = (tipX, dirOutX) => {
      const armLen = 9.4;
      const ang = 0.68; // ~39 degrees
      for (const sideZ of [-1, 1]) {
        const arm = new THREE.BoxGeometry(armLen, 0.018, 0.65);
        arm.rotateY(-dirOutX * sideZ * ang);
        arm.translate(
          tipX - dirOutX * Math.cos(ang) * armLen * 0.46,
          markY + 0.001,
          rCz + sideZ * Math.sin(ang) * armLen * 0.46
        );
        appendPart(hPos, hNorm, hUv, hCol, hIdx, arm, yellowMarkCol);
      }
    };
    for (let k = 0; k < 3; k++) {
      addBlastChevron(xMin - 1.4 - k * 3.0, -1.0);
      addBlastChevron(xMax + 1.4 + k * 3.0, 1.0);
    }

    // 3F. Yellow Taxiway Centerlines, Holding-Position Bars, & West Apron Aircraft T-Stands
    const apronTaxilane = new THREE.BoxGeometry(apronLenX - 10.0, 0.016, 0.28);
    apronTaxilane.translate(apronMidX, markY, 677.5);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, apronTaxilane, yellowMarkCol);

    for (const tx of twyXs) {
      const twyLine = new THREE.BoxGeometry(0.28, 0.016, rCz - 677.5);
      twyLine.translate(tx, markY, 0.5 * (677.5 + rCz));
      appendPart(hPos, hNorm, hUv, hCol, hIdx, twyLine, yellowMarkCol);

      // ICAO Pattern-A Runway Holding-Position Double Bar at z = 693.8m
      for (const dzHold of [-0.28, 0.28]) {
        const holdBar = new THREE.BoxGeometry(9.4, 0.018, 0.22);
        holdBar.translate(tx, markY + 0.001, 693.8 + dzHold);
        appendPart(hPos, hNorm, hUv, hCol, hIdx, holdBar, yellowMarkCol);
      }

      // Aircraft Nosewheel Parking T-Stand on West Apron (z = 665.5 .. 677.5)
      const standStem = new THREE.BoxGeometry(0.26, 0.018, 11.2);
      standStem.translate(tx, markY + 0.001, 671.8);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, standStem, yellowMarkCol);

      const standCross = new THREE.BoxGeometry(5.2, 0.018, 0.30);
      standCross.translate(tx, markY + 0.001, 666.2);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, standCross, yellowMarkCol);
    }

    // 4. Elevated ICAO Night-Glowing Runway Edge, Threshold, End & 4-Unit PAPI Approach Lights
    //    (Lenses sit at y = 0.85m..0.94m > 0.72m so E2 lights them up crisply at NIGHT while staying 0.0 during DAY!)
    const addEdgeFixture = (lx, lz, lensCol) => {
      const stem = new THREE.CylinderGeometry(0.045, 0.075, 0.26, 8, 1);
      stem.translate(lx, deckY + 0.12, lz);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, stem, [0.85, 0.72, 0.14]);

      const lens = new THREE.SphereGeometry(0.115, 10, 6);
      lens.translate(lx, deckY + 0.28, lz); // y = 0.86m
      appendPart(hPos, hNorm, hUv, hCol, hIdx, lens, lensCol);
    };

    const warmEdgeLens = [0.95, 0.85, 0.50];
    const greenThreshLens = [0.06, 0.94, 0.44];
    const redEndLens = [0.95, 0.12, 0.09];

    for (let i = 0; i < 15; i++) {
      const lx = xMin + 3.0 + (i / 14) * (xMax - xMin - 6.0);
      for (const sz of [-(halfW + 0.95), halfW + 0.95]) {
        // Leave taxiway mouths clear on the West (-Z) edge
        if (sz < 0.0 && (Math.abs(lx - twyXs[0]) < 6.4 || Math.abs(lx - twyXs[1]) < 6.4)) continue;
        addEdgeFixture(lx, rCz + sz, warmEdgeLens);
      }
    }

    for (const endX of [xMin - 0.35, xMax + 0.35]) {
      for (const zOff of [-8.4, -4.2, 4.2, 8.4]) {
        addEdgeFixture(endX, rCz + zOff, greenThreshLens);
      }
      for (const zOff of [-10.2, 10.2]) {
        addEdgeFixture(endX, rCz + zOff, redEndLens);
      }
    }

    // 4-Unit PAPI (2 White Inner + 2 Red Outer Glide-Slope Projector Boxes) beside RWY 36 Touchdown Zone
    const papiX = xMin + 39.0; // -333.0
    for (let p = 0; p < 4; p++) {
      const pz = rCz + halfW + 4.4 + p * 2.2;
      const leg = new THREE.CylinderGeometry(0.05, 0.06, 0.34, 8, 1);
      leg.translate(papiX, deckY + 0.10, pz);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, leg, [0.32, 0.34, 0.38]);

      const housing = new THREE.BoxGeometry(0.68, 0.24, 0.52);
      housing.rotateZ(-0.06);
      housing.translate(papiX + 0.12, deckY + 0.32, pz);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, housing, [0.88, 0.42, 0.10]);

      const papiLens = new THREE.SphereGeometry(0.11, 10, 6);
      papiLens.translate(papiX - 0.22, deckY + 0.33, pz); // y = 0.91m
      appendPart(hPos, hNorm, hUv, hCol, hIdx, papiLens, p < 2 ? warmEdgeLens : redEndLens);
    }

    // 5. 6.8m Striped Airfield Windsock Mast, Billowing Orange/White Windsock Cone & Red Obstruction Beacon
    //    Positioned north of the West Apron at (x = -214.0, z = 686.5)
    const wsX = -214.0;
    const wsZ = 686.5;
    const wsPed = new THREE.CylinderGeometry(0.46, 0.58, 0.72, 8, 1);
    wsPed.translate(wsX, 0.36, wsZ);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, wsPed, [0.92, 0.90, 0.86], whiteUV);

    for (let b = 0; b < 6; b++) {
      const y0 = 0.64 + b * 0.98;
      const rBot = 0.115 - b * 0.011;
      const rTop = 0.115 - (b + 1) * 0.011;
      const band = new THREE.CylinderGeometry(rTop, rBot, 0.98, 12, 1);
      band.translate(wsX, y0 + 0.49, wsZ);
      appendPart(
        hPos,
        hNorm,
        hUv,
        hCol,
        hIdx,
        band,
        b % 2 === 1 ? [0.84, 0.18, 0.12] : [0.94, 0.95, 0.96]
      );
    }

    const wsBeacon = new THREE.SphereGeometry(0.15, 12, 8);
    wsBeacon.translate(wsX, 6.64, wsZ);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, wsBeacon, [0.96, 0.10, 0.08]);

    const windAz = 0.48;
    const boom = new THREE.CylinderGeometry(0.03, 0.03, 0.75, 8, 1);
    boom.rotateZ(Math.PI * 0.5);
    boom.translate(0.38, 0.0, 0.0);
    boom.rotateY(-windAz);
    boom.translate(wsX, 6.30, wsZ);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, boom, [0.35, 0.38, 0.42]);

    const throatRing = new THREE.TorusGeometry(0.32, 0.026, 8, 16);
    throatRing.rotateY(Math.PI * 0.5 - windAz);
    throatRing.translate(wsX + Math.cos(windAz) * 0.72, 6.26, wsZ + Math.sin(windAz) * 0.72);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, throatRing, [0.32, 0.35, 0.39]);

    for (let s = 0; s < 5; s++) {
      const segLen = 0.46;
      const r0 = 0.32 - s * 0.038;
      const r1 = 0.32 - (s + 1) * 0.038;
      const droop0 = 0.04 * s * s * 0.32;
      const droop1 = 0.04 * (s + 1) * (s + 1) * 0.32;
      const pitchDown = Math.atan2(droop1 - droop0, segLen);
      const distMid = 0.74 + (s + 0.5) * segLen;

      const coneSeg = new THREE.CylinderGeometry(r1, r0, segLen + 0.02, 16, 1, false);
      coneSeg.rotateZ(Math.PI * 0.5 - pitchDown);
      coneSeg.rotateY(-windAz);
      coneSeg.translate(
        wsX + Math.cos(windAz) * distMid,
        6.26 - 0.5 * (droop0 + droop1),
        wsZ + Math.sin(windAz) * distMid
      );
      appendPart(
        hPos,
        hNorm,
        hUv,
        hCol,
        hIdx,
        coneSeg,
        s % 2 === 0 ? [0.96, 0.36, 0.08] : [0.95, 0.96, 0.97]
      );
    }

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
   * Builds the Class-A ICAO VIP Waterfront Helipad on the South-West Island Headland (-104.0, 106.5),
   * complete with Raised Octagonal Concrete Podium, Cantilevered Perimeter Safety-Net Outriggers,
   * Zebra-Curbed East Vehicular/Pedestrian Ramp, 3D Inlaid ICAO "H" + TDPM + Yellow FATO Markings,
   * 16 Night-Glowing TLOF Perimeter Aviation Light Pods, 6.8m Banded Windsock Mast & 5-Ring Cone,
   * and a Nusantara Limas VIP Waiting Pavilion with Teak Benches (Strictly 0 extra Draw Calls!).
   */
  static _buildIslandVIPHelipad() {
    const hPos = [];
    const hNorm = [];
    const hUv = [];
    const hCol = [];
    const hIdx = [];

    const vPos = [];
    const vNorm = [];
    const vUv = [];
    const vCol = [];
    const vIdx = [];

    const appendPart = (targetPos, targetNorm, targetUv, targetCol, targetIdx, geo, col, uvFixed = null) => {
      const nonIndexed = geo.index ? geo : geo;
      if (!nonIndexed.attributes.normal) nonIndexed.computeVertexNormals();
      const p = nonIndexed.attributes.position.array;
      const n = nonIndexed.attributes.normal.array;
      const uArr = nonIndexed.attributes.uv ? nonIndexed.attributes.uv.array : null;
      const base = targetPos.length / 3;
      for (let i = 0; i < nonIndexed.attributes.position.count; i++) {
        const px = p[i * 3];
        const py = p[i * 3 + 1];
        const pz = p[i * 3 + 2];
        targetPos.push(px, py, pz);
        targetNorm.push(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]);
        if (uvFixed) {
          targetUv.push(uvFixed[0], uvFixed[1]);
        } else if (uArr) {
          targetUv.push(uArr[i * 2], uArr[i * 2 + 1]);
        } else {
          targetUv.push(px * 0.25, pz * 0.25);
        }
        targetCol.push(col[0], col[1], col[2]);
      }
      if (nonIndexed.index) {
        const idx = nonIndexed.index.array;
        for (let i = 0; i < idx.length; i++) {
          targetIdx.push(base + idx[i]);
        }
      } else {
        for (let i = 0; i < nonIndexed.attributes.position.count; i++) {
          targetIdx.push(base + i);
        }
      }
    };

    const hCx = TerrainDEMLoader.HELIPAD_CX; // -104.0
    const hCz = TerrainDEMLoader.HELIPAD_CZ; //  106.5
    const deckY = TerrainDEMLoader.HELIPAD_DECK_Y; // 0.48
    const whiteUV = [0.055, 0.5];

    // 1. Octagonal Reinforced-Concrete Helipad Structural Podium (y = -2.52m .. +0.48m)
    const podiumH = deckY + 2.52;
    const podiumGeo = new THREE.CylinderGeometry(15.2, 15.95, podiumH, 8, 1);
    podiumGeo.rotateY(Math.PI / 8);
    podiumGeo.translate(hCx, deckY - podiumH * 0.5, hCz);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, podiumGeo, [0.92, 0.90, 0.86], whiteUV);

    // Dark Anti-Skid Aviation Tarmac Octagonal FATO Deck (R = 14.65m, y = +0.485m)
    const tarmacGeo = new THREE.CylinderGeometry(14.65, 14.72, 0.04, 8, 1);
    tarmacGeo.rotateY(Math.PI / 8);
    tarmacGeo.translate(hCx, deckY - 0.01, hCz);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, tarmacGeo, [0.16, 0.17, 0.19]);

    // 2. Perimeter Concrete Edge Curbs & 7 Cantilevered ICAO Steel Safety-Net Outrigger Frames
    //    (Face k = 0 faces +X East toward Campus Access Ramp -> left open as the entry gateway!)
    const apothemPodium = 15.2 * Math.cos(Math.PI / 8); // 14.043 m
    const edgeLenPodium = 2.0 * 15.2 * Math.sin(Math.PI / 8); // 11.634 m
    for (let k = 0; k < 8; k++) {
      const faceAngle = (k * Math.PI) / 4.0; // k=0 -> +X (East), k=2 -> +Z (South), k=4 -> -X (West Sea)
      const cosF = Math.cos(faceAngle);
      const sinF = Math.sin(faceAngle);
      const rotFace = new THREE.Matrix4().makeRotationY(-faceAngle);

      if (k !== 0) {
        // Raised concrete perimeter safety coping curb
        const curb = new THREE.BoxGeometry(0.46, 0.11, edgeLenPodium * 0.96);
        curb.applyMatrix4(rotFace);
        curb.translate(hCx + cosF * (apothemPodium - 0.22), deckY + 0.045, hCz + sinF * (apothemPodium - 0.22));
        appendPart(vPos, vNorm, vUv, vCol, vIdx, curb, [0.94, 0.93, 0.89], whiteUV);

        // Cantilevered ICAO perimeter safety-net outer tubular rail + 3 angled steel outrigger struts + mesh deck
        const outerRailR = apothemPodium + 1.48;
        const rail = new THREE.BoxGeometry(0.14, 0.10, edgeLenPodium * 1.04);
        rail.applyMatrix4(rotFace);
        rail.translate(hCx + cosF * outerRailR, deckY + 0.12, hCz + sinF * outerRailR);
        appendPart(hPos, hNorm, hUv, hCol, hIdx, rail, [0.32, 0.35, 0.39]);

        const netMesh = new THREE.BoxGeometry(1.42, 0.028, edgeLenPodium * 0.98);
        netMesh.rotateZ(0.08);
        netMesh.applyMatrix4(rotFace);
        netMesh.translate(hCx + cosF * (apothemPodium + 0.74), deckY + 0.05, hCz + sinF * (apothemPodium + 0.74));
        appendPart(hPos, hNorm, hUv, hCol, hIdx, netMesh, [0.24, 0.27, 0.30]);

        for (const lat of [-0.38, 0.0, 0.38]) {
          const strut = new THREE.BoxGeometry(1.65, 0.08, 0.12);
          strut.rotateZ(0.14);
          strut.translate(0.0, 0.0, lat * edgeLenPodium);
          strut.applyMatrix4(rotFace);
          strut.translate(hCx + cosF * (apothemPodium + 0.70), deckY - 0.04, hCz + sinF * (apothemPodium + 0.70));
          appendPart(hPos, hNorm, hUv, hCol, hIdx, strut, [0.36, 0.39, 0.43]);
        }
      }
    }

    // 3. East Vehicular & VIP Pedestrian Access Ramp + Turnaround Plaza (x = -90.1 .. -81.0, z = 106.5)
    //    Connects RotundaWestCampusBoulevard (y = 0.032m) smoothly up to deckY (+0.48m)!
    const rampX0 = hCx + apothemPodium - 0.15; // -90.11 (top at deckY)
    const rampX1 = -81.2; // bottom meeting boulevard at y = 0.032
    const rampLen = rampX1 - rampX0; // 8.91 m
    const rampMidX = 0.5 * (rampX0 + rampX1);
    const rampRise = deckY - 0.032;
    const rampSlopeAngle = -Math.atan2(rampRise, rampLen);
    const rampHypot = Math.hypot(rampLen, rampRise);

    const rampSlab = new THREE.BoxGeometry(rampHypot + 0.3, 0.32, 8.8);
    rampSlab.rotateZ(rampSlopeAngle);
    rampSlab.translate(rampMidX, 0.5 * (deckY + 0.032) - 0.15, hCz);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, rampSlab, [0.20, 0.21, 0.23]);

    // Ramp sub-grade foundation block & flanking zebra-striped concrete safety cheek-curbs
    const rampBase = new THREE.BoxGeometry(rampLen + 0.4, 2.2, 9.8);
    rampBase.translate(rampMidX, -0.95, hCz);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, rampBase, [0.88, 0.86, 0.82], whiteUV);

    for (const sz of [-4.65, 4.65]) {
      const rampCurb = new THREE.BoxGeometry(rampHypot + 0.2, 0.26, 0.52);
      // Map U along ramp length for crisp black-and-white zebra safety curb stripes!
      const pArr = rampCurb.attributes.position.array;
      const uArr = rampCurb.attributes.uv.array;
      for (let vi = 0; vi < rampCurb.attributes.position.count; vi++) {
        uArr[vi * 2] = (pArr[vi * 3] + rampHypot * 0.5) * 0.45;
        uArr[vi * 2 + 1] = 0.5;
      }
      rampCurb.rotateZ(rampSlopeAngle);
      rampCurb.translate(rampMidX, 0.5 * (deckY + 0.032) + 0.05, hCz + sz);
      appendPart(vPos, vNorm, vUv, vCol, vIdx, rampCurb, [0.97, 0.96, 0.94]);
    }

    // 4. Precision 3D Inlaid ICAO Annex 14 Helipad Markings (y = deckY + 0.016m .. +0.022m, zero Z-fighting!)
    const markY = deckY + 0.014;
    const yellowCol = [0.96, 0.76, 0.12];
    const whiteMarkCol = [0.95, 0.96, 0.98];

    // 4A. Octagonal Aviation-Yellow FATO Aiming Ring (apothem = 11.4m, width = 0.44m)
    const fatoApothem = 11.4;
    const fatoSideLen = 2.0 * fatoApothem * Math.tan(Math.PI / 8) + 0.42;
    for (let k = 0; k < 8; k++) {
      const ang = (k * Math.PI) / 4.0;
      const bar = new THREE.BoxGeometry(0.44, 0.016, fatoSideLen);
      bar.rotateY(-ang);
      bar.translate(hCx + Math.cos(ang) * fatoApothem, markY, hCz + Math.sin(ang) * fatoApothem);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, bar, yellowCol);
    }

    // 4B. 64-Segment Smooth Circular White Touchdown & Positioning Marking (TDPM) Ring (R = 7.45m .. 8.00m)
    const tdpmSegs = 64;
    const rIn = 7.42;
    const rOut = 7.98;
    const segArcLen = ((2.0 * Math.PI) / tdpmSegs) * rOut * 1.04;
    for (let i = 0; i < tdpmSegs; i++) {
      const th = ((i + 0.5) / tdpmSegs) * Math.PI * 2.0;
      const rMid = 0.5 * (rIn + rOut);
      const seg = new THREE.BoxGeometry(rOut - rIn, 0.018, segArcLen);
      seg.rotateY(-th);
      seg.translate(hCx + Math.cos(th) * rMid, markY + 0.002, hCz + Math.sin(th) * rMid);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, seg, whiteMarkCol);
    }

    // 4C. Monumental White ICAO "H" Identification Marking (6.4m x 4.6m, stroke 0.80m)
    //     Oriented along the East-West sea-to-campus approach axis!
    for (const sz of [-1.9, 1.9]) {
      const hLeg = new THREE.BoxGeometry(6.4, 0.022, 0.80);
      hLeg.translate(hCx, markY + 0.004, hCz + sz);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, hLeg, whiteMarkCol);
    }
    const hCross = new THREE.BoxGeometry(0.84, 0.022, 3.04);
    hCross.translate(hCx, markY + 0.004, hCz);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, hCross, whiteMarkCol);

    // 4D. UI Gold Approach Guidance Chevrons at West Over-Water & East Campus Thresholds
    for (const dirX of [-1, 1]) {
      for (const sideZ of [-1, 1]) {
        const chev = new THREE.BoxGeometry(1.55, 0.018, 0.30);
        chev.rotateY(dirX * sideZ * 0.58);
        chev.translate(hCx + dirX * 9.75, markY + 0.002, hCz + sideZ * 0.58);
        appendPart(hPos, hNorm, hUv, hCol, hIdx, chev, yellowCol);
      }
    }

    // 5. 16 Sculpted Flush Inset TLOF Perimeter Aviation Light Pods (Night-Glowing HDR Emerald Green!)
    const lightPodR = 13.05;
    for (let i = 0; i < 16; i++) {
      const th = (i / 16) * Math.PI * 2.0 + Math.PI / 16;
      const lx = hCx + Math.cos(th) * lightPodR;
      const lz = hCz + Math.sin(th) * lightPodR;

      const bezel = new THREE.CylinderGeometry(0.24, 0.28, 0.05, 12, 1);
      bezel.translate(lx, deckY + 0.025, lz);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, bezel, [0.28, 0.30, 0.34]);

      const lens = new THREE.SphereGeometry(0.145, 10, 6);
      lens.scale(1.0, 0.55, 1.0);
      lens.translate(lx, deckY + 0.055, lz);
      // Vivid emerald chrominance triggers Linear-HDR nighttime TLOF illumination (0.0 during DAY!)
      appendPart(hPos, hNorm, hUv, hCol, hIdx, lens, [0.06, 0.94, 0.44]);
    }

    // 6. 6.8m Banded Aviation Windsock Mast & 5-Segment Orange/White Windsock Cone at (-87.2, 92.8)
    const wsX = -87.2;
    const wsZ = 92.8;
    const wsPed = new THREE.CylinderGeometry(0.45, 0.56, 0.65, 8, 1);
    wsPed.translate(wsX, 0.22, wsZ);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, wsPed, [0.92, 0.90, 0.86], whiteUV);

    for (let b = 0; b < 6; b++) {
      const y0 = 0.52 + b * 0.98;
      const rBot = 0.115 - b * 0.011;
      const rTop = 0.115 - (b + 1) * 0.011;
      const band = new THREE.CylinderGeometry(rTop, rBot, 0.98, 12, 1);
      band.translate(wsX, y0 + 0.49, wsZ);
      const isRed = b % 2 === 1;
      appendPart(
        hPos,
        hNorm,
        hUv,
        hCol,
        hIdx,
        band,
        isRed ? [0.84, 0.18, 0.12] : [0.94, 0.95, 0.96]
      );
    }

    // Top Flashing Red Aviation Obstruction Beacon Dome (y = 6.52m > 5.2m night beacon threshold!)
    const wsBeacon = new THREE.SphereGeometry(0.15, 12, 8);
    wsBeacon.translate(wsX, 6.52, wsZ);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, wsBeacon, [0.96, 0.10, 0.08]);

    // Swivel Boom & 5-Band Tapered Billowing International-Orange / White Windsock Cone
    const windAz = 0.48; // Prevailing tropical Depok breeze vector
    const boom = new THREE.CylinderGeometry(0.03, 0.03, 0.75, 8, 1);
    boom.rotateZ(Math.PI * 0.5);
    boom.translate(0.38, 0.0, 0.0);
    boom.rotateY(-windAz);
    boom.translate(wsX, 6.18, wsZ);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, boom, [0.35, 0.38, 0.42]);

    const throatRing = new THREE.TorusGeometry(0.32, 0.026, 8, 16);
    throatRing.rotateY(Math.PI * 0.5 - windAz);
    throatRing.translate(wsX + Math.cos(windAz) * 0.72, 6.14, wsZ + Math.sin(windAz) * 0.72);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, throatRing, [0.32, 0.35, 0.39]);

    for (let s = 0; s < 5; s++) {
      const segLen = 0.46;
      const r0 = 0.32 - s * 0.038;
      const r1 = 0.32 - (s + 1) * 0.038;
      const droop0 = 0.04 * s * s * 0.32;
      const droop1 = 0.04 * (s + 1) * (s + 1) * 0.32;
      const pitchDown = Math.atan2(droop1 - droop0, segLen);
      const distMid = 0.74 + (s + 0.5) * segLen;

      const coneSeg = new THREE.CylinderGeometry(r1, r0, segLen + 0.02, 16, 1, false);
      coneSeg.rotateZ(Math.PI * 0.5 - pitchDown);
      coneSeg.rotateY(-windAz);
      coneSeg.translate(
        wsX + Math.cos(windAz) * distMid,
        6.14 - 0.5 * (droop0 + droop1),
        wsZ + Math.sin(windAz) * distMid
      );
      const isOrange = s % 2 === 0;
      appendPart(
        hPos,
        hNorm,
        hUv,
        hCol,
        hIdx,
        coneSeg,
        isOrange ? [0.96, 0.36, 0.08] : [0.95, 0.96, 0.97]
      );
    }

    // 7. Nusantara VIP Helipad Transit Pavilion (4-Column Pyramidal Javanese Terracotta Limas Shelter + 2 Teak Benches)
    //    Positioned at (-84.8, 118.2) overlooking the Helipad Entry Plaza & South-West Island Coast!
    const pavX = -84.8;
    const pavZ = 118.2;

    const pavPlinth = new THREE.BoxGeometry(5.4, 2.24, 5.4);
    pavPlinth.translate(pavX, -0.88, pavZ);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, pavPlinth, [0.89, 0.87, 0.83], whiteUV);

    const pavFloor = new THREE.BoxGeometry(5.1, 0.08, 5.1);
    pavFloor.translate(pavX, 0.26, pavZ);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, pavFloor, [0.54, 0.52, 0.48]);

    for (const cxOff of [-2.05, 2.05]) {
      for (const czOff of [-2.05, 2.05]) {
        const colBase = new THREE.BoxGeometry(0.46, 0.32, 0.46);
        colBase.translate(pavX + cxOff, 0.44, pavZ + czOff);
        appendPart(vPos, vNorm, vUv, vCol, vIdx, colBase, [0.90, 0.88, 0.84], whiteUV);

        const colShaft = new THREE.CylinderGeometry(0.15, 0.17, 2.55, 14, 1);
        colShaft.translate(pavX + cxOff, 1.85, pavZ + czOff);
        appendPart(vPos, vNorm, vUv, vCol, vIdx, colShaft, [0.96, 0.95, 0.92], whiteUV);
      }
    }

    // 2-Tier Nusantara Terracotta Limas Roof over VIP Helipad Pavilion
    const eaveBeam = new THREE.BoxGeometry(5.0, 0.20, 5.0);
    eaveBeam.translate(pavX, 3.16, pavZ);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, eaveBeam, [0.95, 0.94, 0.90], whiteUV);

    const roofLower = new THREE.CylinderGeometry(1.75, 4.35, 0.92, 4, 1);
    roofLower.rotateY(Math.PI * 0.25);
    roofLower.translate(pavX, 3.68, pavZ);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, roofLower, [0.76, 0.36, 0.28]);

    const roofUpper = new THREE.ConeGeometry(2.15, 1.05, 4, 1);
    roofUpper.rotateY(Math.PI * 0.25);
    roofUpper.translate(pavX, 4.55, pavZ);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, roofUpper, [0.79, 0.38, 0.29]);

    const finial = new THREE.CylinderGeometry(0.04, 0.11, 0.55, 8, 1);
    finial.translate(pavX, 5.25, pavZ);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, finial, [0.96, 0.95, 0.92], whiteUV);

    // 2 Teak Waiting Benches inside the VIP Pavilion facing the Helipad (-X)
    for (const bzOff of [-1.15, 1.15]) {
      for (const legZ of [-0.62, 0.62]) {
        const leg = new THREE.BoxGeometry(0.48, 0.42, 0.10);
        leg.translate(pavX + 0.35, 0.49, pavZ + bzOff + legZ);
        appendPart(vPos, vNorm, vUv, vCol, vIdx, leg, [0.92, 0.90, 0.86], whiteUV);
      }
      const seat = new THREE.BoxGeometry(0.50, 0.055, 1.52);
      seat.translate(pavX + 0.35, 0.72, pavZ + bzOff);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, seat, [0.48, 0.28, 0.15]);

      const back = new THREE.BoxGeometry(0.055, 0.32, 1.52);
      back.translate(pavX + 0.58, 0.92, pavZ + bzOff);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, back, [0.51, 0.30, 0.16]);
    }

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

  static _makeCirclePoints(cx, y, cz, radius, count = 48) {
    const pts = [];
    for (let i = 0; i < count; i++) {
      const th = (i / count) * Math.PI * 2.0;
      pts.push(new THREE.Vector3(cx + Math.cos(th) * radius, y, cz + Math.sin(th) * radius));
    }
    return pts;
  }

  /**
   * Builds the complete Mavic-calibrated Lapangan Rotunda UI, Zebra Curbs, Botanical Centerpiece,
   * Campus Park Benches & Bollards, West Parking, East Lakeside Promenade, & Long SE Covered Walkway
   * in STRICTLY 2 DRAW CALLS!
   */
  static buildRotundaAndPromenades(sharedSurfaceUniforms = null) {
    const uniforms = sharedSurfaceUniforms || createSharedSurfaceUniforms();
    const corridorGroup = new THREE.Group();
    corridorGroup.name = 'KampusUIRotundaAndPromenades';

    // Material 1: Vertex-Colored Campus Asphalt, Brick, Botanical Topiary, Teak Benches, & SE Terracotta Roof
    const matDeckAndRoof = new THREE.MeshStandardMaterial({
      name: 'CampusAsphaltBrickAndWalkwayRoofMat',
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.84,
      metalness: 0.03,
      envMapIntensity: 0.11
    });
    injectOmniScaleSurfaceShader(matDeckAndRoof, {
      ySegmentStart: 120.0,
      segmentCount: 0.0,
      surfaceProfile: SURFACE_PROFILE.ASPHALT_TEAK_BARK,
      bumpScale: 0.65,
      sharedUniforms: uniforms
    });

    // Material 2: Black-and-White Zebra-Striped Concrete Curbs, Bollards, & Crisp Off-White Walkway Columns
    const matZebraCurbAndConcrete = new THREE.MeshStandardMaterial({
      name: 'RotundaZebraCurbAndConcreteMat',
      map: createZebraCurbTexture(),
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.82,
      metalness: 0.03,
      envMapIntensity: 0.11
    });
    injectOmniScaleSurfaceShader(matZebraCurbAndConcrete, {
      ySegmentStart: 120.0,
      segmentCount: 0.0,
      surfaceProfile: SURFACE_PROFILE.PRECAST_CONCRETE,
      bumpScale: 0.65,
      sharedUniforms: uniforms
    });

    const addPair = (namePrefix, pair) => {
      const hMesh = new THREE.Mesh(pair.horizontalGeo, matDeckAndRoof);
      hMesh.name = `${namePrefix}_DeckOrRoof`;
      hMesh.castShadow = true;
      hMesh.receiveShadow = true;

      const vMesh = new THREE.Mesh(pair.verticalGeo, matZebraCurbAndConcrete);
      vMesh.name = `${namePrefix}_ZebraCurbOrCol`;
      vMesh.castShadow = true;
      vMesh.receiveShadow = true;

      corridorGroup.add(hMesh, vMesh);
    };

    const rCx = SplineCorridorBuilder.ROTUNDA_CX;
    const rCz = SplineCorridorBuilder.ROTUNDA_CZ;

    // 1. Lapangan Rotunda UI Circular Asphalt Boulevard (288 segments!)
    const rotundaSpline = SplineCorridorBuilder._createHighPrecisionSpline(
      SplineCorridorBuilder._makeCirclePoints(rCx, 0.035, rCz, 28.8, 64),
      true
    );
    addPair(
      'LapanganRotundaUI_CircularRoad',
      SplineCorridorBuilder._loftSeparatedCorridor(rotundaSpline, {
        segments: 288,
        halfWidth: 4.9,
        curbWidth: 0.55,
        curbRise: 0.15,
        deckColor: [0.21, 0.22, 0.24],
        zebraCurbs: true,
        zebraFreq: 0.135
      })
    );

    // 2. Multi-Tier Botanical Topiary Shrub Rings & Flowering Garland at Center of Lapangan Rotunda UI
    addPair(
      'LapanganRotundaUI_BotanicalCenter',
      SplineCorridorBuilder._buildRotundaBotanicalCenterpiece(rCx, rCz)
    );

    // 3. West Campus Helipad Boulevard Branching from Lapangan Rotunda UI directly into the VIP Helipad Ramp!
    const westRoadSpline = SplineCorridorBuilder._createHighPrecisionSpline(
      [
        new THREE.Vector3(rCx - 33.2, 0.032, rCz),
        new THREE.Vector3(-54.0, 0.032, rCz),
        new THREE.Vector3(-68.5, 0.032, rCz),
        new THREE.Vector3(-81.6, 0.032, rCz)
      ],
      false
    );
    addPair(
      'RotundaWestCampusBoulevard',
      SplineCorridorBuilder._loftSeparatedCorridor(westRoadSpline, {
        segments: 96,
        halfWidth: 4.5,
        curbWidth: 0.48,
        curbRise: 0.14,
        deckColor: [0.21, 0.22, 0.24],
        zebraCurbs: true,
        zebraFreq: 0.13
      })
    );

    // 4. West Courtyard Parking Lot & Access Lane (x = -62.5m)
    const westParkingSpline = SplineCorridorBuilder._createHighPrecisionSpline(
      [
        new THREE.Vector3(-62.5, 0.03, 102.0),
        new THREE.Vector3(-62.5, 0.03, 56.0),
        new THREE.Vector3(-62.5, 0.03, 10.0),
        new THREE.Vector3(-62.5, 0.03, -34.0)
      ],
      false
    );
    addPair(
      'WestCourtyardParkingLane',
      SplineCorridorBuilder._loftSeparatedCorridor(westParkingSpline, {
        segments: 96,
        halfWidth: 5.8,
        curbWidth: 0.42,
        curbRise: 0.12,
        deckColor: [0.24, 0.25, 0.27],
        zebraCurbs: false
      })
    );

    // 5. East Lakeside Warm Brick Turnaround Courtyard
    const eastBrickSpline = SplineCorridorBuilder._createHighPrecisionSpline(
      [
        new THREE.Vector3(18.0, 0.034, -39.0),
        new THREE.Vector3(31.0, 0.034, -40.5),
        new THREE.Vector3(44.5, 0.034, -36.5),
        new THREE.Vector3(48.5, 0.034, -25.5)
      ],
      false
    );
    addPair(
      'EastLakesideBrickTurnaround',
      SplineCorridorBuilder._loftSeparatedCorridor(eastBrickSpline, {
        segments: 84,
        halfWidth: 4.6,
        curbWidth: 0.38,
        curbRise: 0.11,
        deckColor: [0.54, 0.31, 0.26],
        zebraCurbs: false
      })
    );

    // 6. Danau Kenanga East (+X) Lakeside Promenade
    const lakePts = [];
    for (let z = -66.0; z <= 54.0; z += 10.0) {
      const px = TerrainDEMLoader.getShorelinePromenadeX(z) - 1.45;
      lakePts.push(new THREE.Vector3(px, 0.038, z));
    }
    const eastLakesideSpline = SplineCorridorBuilder._createHighPrecisionSpline(lakePts, false);
    addPair(
      'DanauKenangaEastPromenade',
      SplineCorridorBuilder._loftSeparatedCorridor(eastLakesideSpline, {
        segments: 140,
        halfWidth: 1.35,
        curbWidth: 0.32,
        curbRise: 0.12,
        deckColor: [0.52, 0.50, 0.46],
        zebraCurbs: false
      })
    );

    // 7. Long South-East Terracotta-Roofed Covered Campus Walkway (120 segments!)
    const seWalkwaySpline = SplineCorridorBuilder._createHighPrecisionSpline(
      [
        new THREE.Vector3(34.2, 0.04, 35.5),
        new THREE.Vector3(37.4, 0.04, 58.0),
        new THREE.Vector3(40.6, 0.04, 92.0),
        new THREE.Vector3(43.8, 0.04, 128.0)
      ],
      false
    );
    addPair(
      'SELongCampusWalkway_Deck',
      SplineCorridorBuilder._loftSeparatedCorridor(seWalkwaySpline, {
        segments: 120,
        halfWidth: 1.75,
        curbWidth: 0.28,
        curbRise: 0.11,
        deckColor: [0.48, 0.47, 0.44],
        zebraCurbs: false
      })
    );
    addPair(
      'SELongCampusWalkway_TerracottaRoof',
      SplineCorridorBuilder._buildCoveredCampusWalkwayRoofAndColumns(seWalkwaySpline, 120)
    );

    // 8. Sculpted Teak-and-Concrete Campus Park Benches & Pedestrian Bollards
    addPair(
      'CampusPromenadeFurniture',
      SplineCorridorBuilder._buildCampusPromenadeBenchesAndBollards()
    );

    // 9. Class-A ICAO VIP Waterfront Helipad, Windsock Mast, & Nusantara Transit Pavilion (Pulau U)
    addPair(
      'PulauRektoratUI_VIPHelipad',
      SplineCorridorBuilder._buildIslandVIPHelipad()
    );

    // 10. Class-A ICAO Straight Island Runway RWY 36/18, Blast Pads, Taxiways, West Apron, Lights & Windsock (Pulau I)
    addPair(
      'PulauI_ICAORunwayAndApron',
      SplineCorridorBuilder._buildPulauIAirstripRunway()
    );

    // Hukum Emas 12: Batch all decks/roofs/topiary/teak/helipad/runway into Draw Call #1 & all curbs/columns/bollards into Draw Call #2!
    NexusBatchOptimizer.optimizeHierarchy(corridorGroup);
    corridorGroup.updateMatrixWorld(true);
    return corridorGroup;
  }
}
