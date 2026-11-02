import * as THREE from 'three';
import { NexusBatchOptimizer } from '../core/BatchRenderer.js';
import { PhysicsKernel } from '../core/PhysicsKernel.js';
import {
  SURFACE_PROFILE,
  createZebraCurbTexture,
  createSharedSurfaceUniforms,
  injectOmniScaleSurfaceShader
} from '../core/ProceduralTextureFactory.js';
import { TerrainDEMLoader } from './TerrainDEMLoader.js';
import { AutogenBiomeSpawner } from './AutogenBiomeSpawner.js';
import { DPPDComplexBuilder } from './DPPDComplexBuilder.js';

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
  static VIDEOTRON_SPEC = Object.freeze({
    cx: 28.4,
    cz: 78.2,
    yaw: -0.82928487, // Exact tangent to Lapangan Rotunda UI circle (-2.5, 106.5): atan2(-30.9, 28.3) = -47.51 deg
    screenCenterY: 9.48,
    screenW: 12.80,
    screenH: 7.20,
    screenFwdOffset: 0.302,
    cabinetHalfW: 6.84,
    cabinetMinY: 5.36,
    cabinetMaxY: 13.86
  });

  static FOUNTAIN_SPEC = Object.freeze({
    cx: -2.5,
    cz: 106.5,
    waterY: 0.46,
    poolInnerR: 4.34,
    outerPlanterR: 5.54,
    innerNozzleR: 1.35,
    outerNozzleR: 2.85
  });

  static CALISTHENICS_SPEC = Object.freeze({
    cx: -62.0,
    cz: 136.5,
    ovalRx: 15.5,
    ovalRz: 6.8,
    rotundaTrackR: 37.15
  });

  static MUSTER_POINTS = Object.freeze([
    { id: 'RotundaRektorat', x: -31.0, z: 76.5, yaw: 0.45 },
    { id: 'KompleksDPPD_STP', x: -296.5, z: 326.5, yaw: 1.42 },
    { id: 'DermagaUtaraU', x: -68.0, z: 444.0, yaw: 1.57 },
    { id: 'BandaraPulauI', x: -222.5, z: 665.5, yaw: -1.57 }
  ]);

  static KIOSK_SPECS = Object.freeze([
    { id: 'KiosTamanRotunda', x: -39.5, z: 134.5, yaw: 0.52 },
    { id: 'KiosPlazaTeksas', x: -16.5, z: 345.0, yaw: -1.45 },
    { id: 'KiosKompleksDPPD', x: -213.0, z: 326.0, yaw: -1.35 },
    { id: 'KiosDermagaPulauI', x: -116.5, z: 619.5, yaw: 0.95 }
  ]);

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
      zebraFreq = 0.125,
      curbCutZones = []
    } = options;

    const evalCurbCutFactor = (qx, qz) => {
      let fac = 1.0;
      for (let zIdx = 0; zIdx < curbCutZones.length; zIdx++) {
        const cz = curbCutZones[zIdx];
        const d = Math.hypot(qx - cz.x, qz - cz.z);
        const fade = cz.fade !== undefined ? cz.fade : 1.4;
        if (d <= cz.r) return 0.0;
        if (d < cz.r + fade) {
          const t = (d - cz.r) / fade;
          const s = t * t * (3.0 - 2.0 * t);
          if (s < fac) fac = s;
        }
      }
      return fac;
    };

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
        const midLat = sideSign * (halfWidth + curbWidth * 0.5);
        const cFac = evalCurbCutFactor(pos.x + side.x * midLat, pos.z + side.z * midLat);
        const r = [];
        for (let k = 0; k < offsets.length; k++) {
          const o = offsets[k];
          r.push(
            pushC(
              pos.x + side.x * o.lat,
              pos.y + 0.002 + o.dy * cFac,
              pos.z + side.z * o.lat,
              side.x * o.nSide * cFac,
              1.0 - (1.0 - o.ny) * cFac,
              side.z * o.nSide * cFac,
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
      const sideSign = lateralOffset < 0 ? -1.0 : 1.0;
      for (let i = 0; i <= segments; i++) {
        const { pos, side, arc } = frames[i];
        const px = pos.x + side.x * lateralOffset;
        const pz = pos.z + side.z * lateralOffset;
        const midLat = sideSign * (halfWidth + curbWidth * 0.5);
        const cFac = evalCurbCutFactor(pos.x + side.x * midLat, pos.z + side.z * midLat);
        const yTop = pos.y + 0.002 + yTopRel * cFac;
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
      const addCapQuad = (pos, side, nx, nz, latA, latB, yTopVal, yBot, isEnd) => {
        const xL = pos.x + side.x * latA;
        const zL = pos.z + side.z * latA;
        const xR = pos.x + side.x * latB;
        const zR = pos.z + side.z * latB;
        const iTL = pushC(xL, yTopVal, zL, nx, 0.0, nz, 0.055, 0.2);
        const iTR = pushC(xR, yTopVal, zR, nx, 0.0, nz, 0.055, 0.2);
        const iBR = pushC(xR, yBot, zR, nx, 0.0, nz, 0.055, 0.9);
        const iBL = pushC(xL, yBot, zL, nx, 0.0, nz, 0.055, 0.9);
        if (isEnd) {
          cIdx.push(iTL, iBL, iBR, iTL, iBR, iTR);
        } else {
          cIdx.push(iTL, iTR, iBR, iTL, iBR, iBL);
        }
      };

      const addEndCap = (frame, isEnd) => {
        const { pos, tan, side } = frame;
        const nx = isEnd ? tan.x : -tan.x;
        const nz = isEnd ? tan.z : -tan.z;
        const yBot = skirtBottomY;
        const cFacL = evalCurbCutFactor(
          pos.x - side.x * (halfWidth + curbWidth * 0.5),
          pos.z - side.z * (halfWidth + curbWidth * 0.5)
        );
        const cFacR = evalCurbCutFactor(
          pos.x + side.x * (halfWidth + curbWidth * 0.5),
          pos.z + side.z * (halfWidth + curbWidth * 0.5)
        );
        // Never cap above asphalt across the central carriageway (-halfWidth .. +halfWidth)!
        addCapQuad(pos, side, nx, nz, -outerSpan, -halfWidth, pos.y + 0.002 + curbRise * cFacL, yBot, isEnd);
        addCapQuad(pos, side, nx, nz, -halfWidth, halfWidth, pos.y + 0.002, yBot, isEnd);
        addCapQuad(pos, side, nx, nz, halfWidth, outerSpan, pos.y + 0.002 + curbRise * cFacR, yBot, isEnd);
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

    // 3. Outer Tropical Flowering Ixora & Manicured Topiary Garland Framing the Fountain Basin (R = 4.86m)
    addBotanicalTorus(4.86, 0.42, 0.29, 0.26, 96, 20, (sun, th, ph) => {
      const blossom = Math.max(0.0, Math.sin(18.0 * th) * Math.sin(3.0 * ph)) * Math.max(0.0, sun - 0.32);
      const v = 0.02 * Math.sin(12.0 * th);
      return [
        0.09 + 0.10 * sun + 0.56 * blossom + v,
        0.25 + 0.21 * sun + 0.08 * blossom + v,
        0.08 + 0.06 * sun + 0.05 * blossom
      ];
    });

    // 4. Architectural Tiered Dark-Andesite & Granite Dancing Fountain Reflecting Basin (R <= 4.62m, water y = 0.46m)
    const basinProfile = [
      { r: 4.62, y: 0.14, nr: 1, ny: 0, col: [0.36, 0.37, 0.40] },
      { r: 4.62, y: 0.52, nr: 1, ny: 0, col: [0.42, 0.43, 0.46] },
      { r: 4.56, y: 0.57, nr: 0.35, ny: 0.93, col: [0.76, 0.75, 0.72] },
      { r: 4.36, y: 0.57, nr: -0.35, ny: 0.93, col: [0.76, 0.75, 0.72] },
      { r: 4.34, y: 0.52, nr: -1, ny: 0, col: [0.22, 0.24, 0.27] },
      { r: 4.34, y: 0.19, nr: -1, ny: 0, col: [0.14, 0.16, 0.19] }
    ];
    for (let s = 0; s < basinProfile.length - 1; s++) {
      const p0 = basinProfile[s];
      const p1 = basinProfile[s + 1];
      const rA = [];
      const rB = [];
      for (let i = 0; i <= segs; i++) {
        const th = (i / segs) * Math.PI * 2.0;
        const c = Math.cos(th);
        const sn = Math.sin(th);
        rA.push(pushC(centerX + c * p0.r, p0.y, centerZ + sn * p0.r, c * p0.nr, p0.ny, sn * p0.nr, 0.055, 0.35, p0.col));
        rB.push(pushC(centerX + c * p1.r, p1.y, centerZ + sn * p1.r, c * p1.nr, p1.ny, sn * p1.nr, 0.055, 0.65, p1.col));
      }
      for (let i = 0; i < segs; i++) {
        cIdx.push(rA[i], rB[i], rB[i + 1], rA[i], rB[i + 1], rA[i + 1]);
      }
    }

    // Helper: Solid circular disk / cylinder cap + side in dPos (with optional UV sentinel for submerged RGBW LEDs!)
    const pushDWithUV = (x, y, z, nx, ny, nz, col, u, v) => {
      const idx = dPos.length / 3;
      dPos.push(x, y, z);
      dNorm.push(nx, ny, nz);
      dUv.push(u, v);
      dCol.push(col[0], col[1], col[2]);
      return idx;
    };
    const addCylinderD = (cx0, cz0, rTop, rBot, y0, y1, rSegs, col, topUV = null) => {
      const ringB = [];
      const ringT = [];
      for (let i = 0; i <= rSegs; i++) {
        const th = (i / rSegs) * Math.PI * 2.0;
        const c = Math.cos(th);
        const sn = Math.sin(th);
        ringB.push(pushD(cx0 + c * rBot, y0, cz0 + sn * rBot, c, 0, sn, col));
        ringT.push(pushD(cx0 + c * rTop, y1, cz0 + sn * rTop, c, 0, sn, col));
      }
      for (let i = 0; i < rSegs; i++) {
        dIdx.push(ringB[i], ringT[i + 1], ringT[i], ringB[i], ringB[i + 1], ringT[i + 1]);
      }
      const uT = topUV ? topUV[0] : cx0 * 0.25;
      const vT = topUV ? topUV[1] : cz0 * 0.25;
      const cTop = pushDWithUV(cx0, y1, cz0, 0, 1, 0, col, uT, vT);
      const capRing = [];
      for (let i = 0; i <= rSegs; i++) {
        const th = (i / rSegs) * Math.PI * 2.0;
        const px = cx0 + Math.cos(th) * rTop;
        const pz = cz0 + Math.sin(th) * rTop;
        capRing.push(
          pushDWithUV(px, y1, pz, 0, 1, 0, col, topUV ? topUV[0] : px * 0.25, topUV ? topUV[1] : pz * 0.25)
        );
      }
      for (let i = 0; i < rSegs; i++) {
        dIdx.push(cTop, capRing[i + 1], capRing[i]);
      }
    };

    // Dark basalt mosaic basin floor (y = 0.20m) & calm reflecting pool water surface (y = 0.46m, R = 4.33m)
    addCylinderD(centerX, centerZ, 4.33, 4.33, 0.19, 0.21, 64, [0.10, 0.13, 0.17]);
    addCylinderD(centerX, centerZ, 4.335, 4.335, 0.44, 0.46, 72, [0.05, 0.15, 0.23]);

    // Central 2-Tier Sculpted Andesite & Brass Fountain Manifold Pedestal
    addCylinderD(centerX, centerZ, 1.56, 1.66, 0.21, 0.44, 36, [0.18, 0.20, 0.23]);
    addCylinderD(centerX, centerZ, 0.56, 0.64, 0.44, 0.51, 24, [0.24, 0.26, 0.30]);
    addCylinderD(centerX, centerZ, 0.22, 0.26, 0.51, 0.55, 16, [0.78, 0.61, 0.18]);
    // Central Brass Spire Nozzle
    addCylinderD(centerX, centerZ, 0.055, 0.075, 0.55, 0.63, 12, [0.85, 0.68, 0.22]);

    // Stainless Submerged Distribution Ring Pipes (R = 1.35m & R = 2.85m)
    for (const rRing of [1.35, 2.85]) {
      const pipeSegs = rRing < 2.0 ? 24 : 40;
      for (let i = 0; i < pipeSegs; i++) {
        const th0 = (i / pipeSegs) * Math.PI * 2.0;
        const th1 = ((i + 1) / pipeSegs) * Math.PI * 2.0;
        const mx = centerX + Math.cos(0.5 * (th0 + th1)) * rRing;
        const mz = centerZ + Math.sin(0.5 * (th0 + th1)) * rRing;
        addCylinderD(mx, mz, 0.035, 0.035, 0.36, 0.41, 6, [0.42, 0.45, 0.49]);
      }
    }

    // 8 Inner Crown Brass Gimbal Nozzles + 8 Submerged RGBW Show LED Pods (UV sentinel v = -920.0!)
    const fSpec = SplineCorridorBuilder.FOUNTAIN_SPEC;
    for (let k = 0; k < 8; k++) {
      const th = (k / 8) * Math.PI * 2.0;
      const nx = centerX + Math.cos(th) * fSpec.innerNozzleR;
      const nz = centerZ + Math.sin(th) * fSpec.innerNozzleR;
      addCylinderD(nx, nz, 0.042, 0.058, 0.43, 0.52, 8, [0.84, 0.67, 0.20]);
      const lx = centerX + Math.cos(th + 0.20) * (fSpec.innerNozzleR + 0.16);
      const lz = centerZ + Math.sin(th + 0.20) * (fSpec.innerNozzleR + 0.16);
      addCylinderD(lx, lz, 0.085, 0.095, 0.41, 0.458, 10, [0.16, 0.18, 0.22], [0.5, -920.0]);
    }

    // 16 Outer Dancing Ring Brass Gimbal Nozzles + 16 Submerged RGBW Show LED Pods (UV sentinel v = -920.0!)
    for (let k = 0; k < 16; k++) {
      const th = (k / 16) * Math.PI * 2.0;
      const nx = centerX + Math.cos(th) * fSpec.outerNozzleR;
      const nz = centerZ + Math.sin(th) * fSpec.outerNozzleR;
      addCylinderD(nx, nz, 0.038, 0.054, 0.39, 0.495, 8, [0.84, 0.67, 0.20]);
      const lx = centerX + Math.cos(th + 0.12) * (fSpec.outerNozzleR + 0.18);
      const lz = centerZ + Math.sin(th + 0.12) * (fSpec.outerNozzleR + 0.18);
      addCylinderD(lx, lz, 0.088, 0.098, 0.39, 0.458, 10, [0.16, 0.18, 0.22], [0.5, -920.0]);
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

  /**
   * Builds the High-Accuracy Architectural Digital Twin of "Jembatan Teksas UI" (Teknik–Sastra, Krakatau Steel BTKC 2007)
   * spanning across Teluk U between the Left Arm (FT / Rotunda Shore) and Right Arm (FIB / Sastra Shore) of Pulau U:
   * - 148.0m clear-span parabolic steel arch (0 piers in Teluk U water) assembled from 7 bolted BTKC box-girder segments
   *   with 9.25m clear mid-span air-draft headroom above water (WATER_LEVEL_Y = -1.45m -> lowest soffit +7.80m) for Mini Yachts!
   * - Krakatau Steel crimson-red edge fascia girders, twin parabolic overhead through-arch truss chords, vertical hangers,
   *   diagonal Warren web struts, overhead portal sway braces, and 5.20m pedestrian/bicycle deck with UI-yellow handrails.
   * - Asymmetric Twin UI Golden-Yellow Pylons:
   *   1. West Portal (z = 171.65m): Pilon Lingga (25.0m tall -> y = +25.25m) shaped like a soaring curved ship's sail (layar kapal)
   *      with sail batten ribs, 3 wind-relief oculi rings, masthead red night beacon, & 10 tensioned steel fan stay cables.
   *   2. East Portal (z = 319.65m): Pilon Yoni (15.0m tall -> y = +15.34m) shaped like a sculpted 3D curvilinear curved hip
   *      (melengkung menyerupai pinggul — slender waist, flared rounded hips, united rounded crown) & 8 fan stay cables.
   * - 30 Translucent-Look Cerulean-Blue Wave-Ripple Sunroof Canopy Units (30 unit atap sunroof riak gelombang biru)
   *   sheltering pedestrians from rain via RainCollisionBake + recessed 3000K photocell LED walkway lighting.
   * - Inclusive Approaches on Both Shores: Central 22-step ceremonial staircases (tangga) + twin flanking accessible
   *   bicycle/wheelchair ramps (rampa), concrete shore abutments, West Forecourt Plaza, & East FIB Overlook Plaza + teak benches.
   *
   * Batched directly into KampusUIRotundaAndPromenades -> +0 Extra Draw Calls!
   */
  static _buildJembatanTeksasUI() {
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

    const whiteUV = [0.055, 0.5];

    const appendPart = (targetPos, targetNorm, targetUv, targetCol, targetIdx, geo, col, uvFixed = null) => {
      if (!geo.attributes.normal) geo.computeVertexNormals();
      const p = geo.attributes.position.array;
      const n = geo.attributes.normal.array;
      const uArr = geo.attributes.uv ? geo.attributes.uv.array : null;
      const base = targetPos.length / 3;
      const count = geo.attributes.position.count;
      for (let i = 0; i < count; i++) {
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
        for (let i = 0; i < count; i++) {
          targetIdx.push(base + i);
        }
      }
      geo.dispose();
    };

    const pushVert = (tPos, tNorm, tUv, tCol, x, y, z, nx, ny, nz, col, uvFixed = null) => {
      const idx = tPos.length / 3;
      tPos.push(x, y, z);
      tNorm.push(nx, ny, nz);
      if (uvFixed) {
        tUv.push(uvFixed[0], uvFixed[1]);
      } else {
        tUv.push(x * 0.22, z * 0.22);
      }
      tCol.push(col[0], col[1], col[2]);
      return idx;
    };

    // Zero-allocation CCW outward analytical tube builder for steel struts, arch chords, cables, & curved pylon ribs
    const pushTubeH = (x0, y0, z0, x1, y1, z1, r0, r1, segs, col, capEnds = false) => {
      const dx = x1 - x0;
      const dy = y1 - y0;
      const dz = z1 - z0;
      const len = Math.hypot(dx, dy, dz);
      if (len < 1e-4) return;
      const tx = dx / len;
      const ty = dy / len;
      const tz = dz / len;
      let rx = 0.0;
      let ry = 1.0;
      let rz = 0.0;
      if (Math.abs(ty) >= 0.92) {
        rx = 1.0;
        ry = 0.0;
        rz = 0.0;
      }
      let ux = ry * tz - rz * ty;
      let uy = rz * tx - rx * tz;
      let uz = rx * ty - ry * tx;
      const uLen = Math.hypot(ux, uy, uz) || 1.0;
      ux /= uLen;
      uy /= uLen;
      uz /= uLen;
      const vx = ty * uz - tz * uy;
      const vy = tz * ux - tx * uz;
      const vz = tx * uy - ty * ux;

      const ring0 = [];
      const ring1 = [];
      for (let k = 0; k <= segs; k++) {
        const th = (k / segs) * Math.PI * 2.0;
        const c = Math.cos(th);
        const s = Math.sin(th);
        const nx = ux * c + vx * s;
        const ny = uy * c + vy * s;
        const nz = uz * c + vz * s;
        ring0.push(pushVert(hPos, hNorm, hUv, hCol, x0 + nx * r0, y0 + ny * r0, z0 + nz * r0, nx, ny, nz, col));
        ring1.push(pushVert(hPos, hNorm, hUv, hCol, x1 + nx * r1, y1 + ny * r1, z1 + nz * r1, nx, ny, nz, col));
      }
      for (let k = 0; k < segs; k++) {
        const b0 = ring0[k];
        const b1 = ring0[k + 1];
        const t1 = ring1[k + 1];
        const t0 = ring1[k];
        hIdx.push(b0, b1, t1, b0, t1, t0);
      }
      if (capEnds) {
        const c0 = pushVert(hPos, hNorm, hUv, hCol, x0, y0, z0, -tx, -ty, -tz, col);
        const c1 = pushVert(hPos, hNorm, hUv, hCol, x1, y1, z1, tx, ty, tz, col);
        for (let k = 0; k < segs; k++) {
          const th0 = (k / segs) * Math.PI * 2.0;
          const th1 = ((k + 1) / segs) * Math.PI * 2.0;
          const p0a = pushVert(
            hPos,
            hNorm,
            hUv,
            hCol,
            x0 + (ux * Math.cos(th0) + vx * Math.sin(th0)) * r0,
            y0 + (uy * Math.cos(th0) + vy * Math.sin(th0)) * r0,
            z0 + (uz * Math.cos(th0) + vz * Math.sin(th0)) * r0,
            -tx,
            -ty,
            -tz,
            col
          );
          const p0b = pushVert(
            hPos,
            hNorm,
            hUv,
            hCol,
            x0 + (ux * Math.cos(th1) + vx * Math.sin(th1)) * r0,
            y0 + (uy * Math.cos(th1) + vy * Math.sin(th1)) * r0,
            z0 + (uz * Math.cos(th1) + vz * Math.sin(th1)) * r0,
            -tx,
            -ty,
            -tz,
            col
          );
          hIdx.push(c0, p0b, p0a);

          const p1a = pushVert(
            hPos,
            hNorm,
            hUv,
            hCol,
            x1 + (ux * Math.cos(th0) + vx * Math.sin(th0)) * r1,
            y1 + (uy * Math.cos(th0) + vy * Math.sin(th0)) * r1,
            z1 + (uz * Math.cos(th0) + vz * Math.sin(th0)) * r1,
            tx,
            ty,
            tz,
            col
          );
          const p1b = pushVert(
            hPos,
            hNorm,
            hUv,
            hCol,
            x1 + (ux * Math.cos(th1) + vx * Math.sin(th1)) * r1,
            y1 + (uy * Math.cos(th1) + vy * Math.sin(th1)) * r1,
            z1 + (uz * Math.cos(th1) + vz * Math.sin(th1)) * r1,
            tx,
            ty,
            tz,
            col
          );
          hIdx.push(c1, p1a, p1b);
        }
      }
    };

    // Double-sided / thickened quad helper with exact analytic face normal
    const pushQuadH = (ax, ay, az, bx0, by0, bz0, cx, cy, cz, dx0, dy0, dz0, col) => {
      const ux = bx0 - ax;
      const uy = by0 - ay;
      const uz = bz0 - az;
      const vx = dx0 - ax;
      const vy = dy0 - ay;
      const vz = dz0 - az;
      let nx = uy * vz - uz * vy;
      let ny = uz * vx - ux * vz;
      let nz = ux * vy - uy * vx;
      const nl = Math.hypot(nx, ny, nz) || 1.0;
      nx /= nl;
      ny /= nl;
      nz /= nl;
      const i0 = pushVert(hPos, hNorm, hUv, hCol, ax, ay, az, nx, ny, nz, col);
      const i1 = pushVert(hPos, hNorm, hUv, hCol, bx0, by0, bz0, nx, ny, nz, col);
      const i2 = pushVert(hPos, hNorm, hUv, hCol, cx, cy, cz, nx, ny, nz, col);
      const i3 = pushVert(hPos, hNorm, hUv, hCol, dx0, dy0, dz0, nx, ny, nz, col);
      hIdx.push(i0, i1, i2, i0, i2, i3);
    };

    const bx = TerrainDEMLoader.TEKSAS_BRIDGE_X; // -28.0
    const zW = TerrainDEMLoader.TEKSAS_WEST_PORTAL_Z; // 171.65
    const zE = TerrainDEMLoader.TEKSAS_EAST_PORTAL_Z; // 319.65
    const zC = TerrainDEMLoader.TEKSAS_SPAN_CENTER_Z; // 245.65
    const halfSpan = TerrainDEMLoader.TEKSAS_HALF_SPAN_Z; // 74.0
    const spanLen = zE - zW; // 148.0
    const deckHalfW = TerrainDEMLoader.TEKSAS_DECK_HALF_W; // 2.60
    const portalDeckY = TerrainDEMLoader.TEKSAS_PORTAL_DECK_Y; // 4.15
    const archRiseY = TerrainDEMLoader.TEKSAS_ARCH_RISE_Y; // 4.95
    const girderDepth = TerrainDEMLoader.TEKSAS_GIRDER_DEPTH_M; // 1.30
    const zRampW0 = TerrainDEMLoader.TEKSAS_WEST_RAMP_START_Z; // 149.65
    const zRampE1 = TerrainDEMLoader.TEKSAS_EAST_RAMP_END_Z; // 341.65
    const eastPlazaY = TerrainDEMLoader.TEKSAS_EAST_PLAZA_Y; // 0.34

    // Authentic Jembatan Teksas UI Palette (PT Krakatau Steel BTKC + UI Identity)
    const krakatauRed = [0.68, 0.11, 0.09];
    const krakatauRedDark = [0.54, 0.08, 0.07];
    const btkcCharcoal = [0.15, 0.16, 0.18];
    const btkcFlangeSteel = [0.25, 0.27, 0.30];
    const uiGoldenYellow = [0.92, 0.68, 0.07];
    const uiYellowDeep = [0.80, 0.56, 0.05];
    const stayCableSteel = [0.62, 0.65, 0.70];
    const deckPaverGrey = [0.35, 0.36, 0.39];
    const bikeLaneBrick = [0.47, 0.32, 0.25];
    const sunroofBlue = [0.10, 0.46, 0.88];
    const sunroofBlueUnderside = [0.15, 0.52, 0.92];
    const warmLedStrip = [0.96, 0.90, 0.74];
    const concreteWarm = [0.91, 0.89, 0.85];
    const concreteStep = [0.86, 0.84, 0.80];

    const evalArchDeckY = (z) => {
      const u = (z - zC) / halfSpan;
      return portalDeckY + archRiseY * Math.max(0.0, 1.0 - u * u);
    };
    const evalArchSlope = (z) => {
      const u = (z - zC) / halfSpan;
      return (-2.0 * archRiseY * u) / halfSpan;
    };
    const evalArchChordY = (z) => {
      const u = (z - zC) / halfSpan;
      return evalArchDeckY(z) + 1.18 + 2.58 * Math.max(0.0, 1.0 - u * u);
    };

    // =========================================================================
    // 1. REINFORCED-CONCRETE SHORE ABUTMENTS, PYLON PEDESTALS, & FORECOURT PLAZAS
    // =========================================================================
    // West Shore Abutment Pier (z = 171.65, firmly inland of Left-Arm stone turap at z ~ 173.76)
    const westAbutH = portalDeckY - girderDepth + 3.45;
    const westAbut = new THREE.BoxGeometry(6.8, westAbutH, 3.6);
    westAbut.translate(bx, portalDeckY - girderDepth - westAbutH * 0.5 + 0.06, zW - 0.35);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, westAbut, concreteWarm, whiteUV);

    // East Shore Abutment Pier (z = 319.65, firmly inland of Right-Arm stone turap at z ~ 317.73)
    const eastAbutH = portalDeckY - girderDepth + 3.45;
    const eastAbut = new THREE.BoxGeometry(6.8, eastAbutH, 3.6);
    eastAbut.translate(bx, portalDeckY - girderDepth - eastAbutH * 0.5 + 0.06, zE + 0.35);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, eastAbut, concreteWarm, whiteUV);

    // Pilon Lingga & Pilon Yoni Flared Concrete Anchor Pedestals (straddling walkway wide clear at x = bx +/- 3.85m)
    for (const sx of [-1, 1]) {
      const pedW = new THREE.CylinderGeometry(0.95, 1.25, 3.85, 12, 1);
      pedW.translate(bx + sx * 3.85, 0.35 - 3.85 * 0.5, zW);
      appendPart(vPos, vNorm, vUv, vCol, vIdx, pedW, concreteWarm, whiteUV);

      const capW = new THREE.CylinderGeometry(1.02, 1.02, 0.18, 12, 1);
      capW.translate(bx + sx * 3.85, 0.35, zW);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, capW, btkcFlangeSteel);

      const pedE = new THREE.CylinderGeometry(0.88, 1.18, 3.95, 12, 1);
      pedE.translate(bx + sx * 3.85, 0.44 - 3.95 * 0.5, zE);
      appendPart(vPos, vNorm, vUv, vCol, vIdx, pedE, concreteWarm, whiteUV);

      const capE = new THREE.CylinderGeometry(0.95, 0.95, 0.18, 12, 1);
      capE.translate(bx + sx * 3.85, 0.44, zE);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, capE, btkcFlangeSteel);
    }

    // West Forecourt Entry Plaza (FT / Rotunda Side: x in [-34.6, -21.4], z in [144.2, 150.4])
    const westPlazaBase = new THREE.BoxGeometry(13.2, 2.54, 6.4);
    westPlazaBase.translate(bx, 0.04 - 1.27, 147.3);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, westPlazaBase, concreteWarm, whiteUV);

    const westPlazaTop = new THREE.BoxGeometry(12.6, 0.05, 5.9);
    westPlazaTop.translate(bx, 0.045, 147.3);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, westPlazaTop, [0.44, 0.43, 0.41]);

    // East FIB Scenic Overlook Plaza (Right Arm Side: x in [-35.2, -20.8], z in [341.0, 349.8], y = +0.36m)
    const eastPlazaBase = new THREE.BoxGeometry(14.4, 2.76, 8.8);
    eastPlazaBase.translate(bx, eastPlazaY + 0.02 - 1.38, 345.4);
    appendPart(vPos, vNorm, vUv, vCol, vIdx, eastPlazaBase, concreteWarm, whiteUV);

    const eastPlazaTop = new THREE.BoxGeometry(13.7, 0.055, 8.2);
    eastPlazaTop.translate(bx, eastPlazaY + 0.03, 345.4);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, eastPlazaTop, [0.46, 0.45, 0.42]);

    // 2 Sculpted Teak-and-Concrete Overlook Park Benches on East FIB Plaza facing Teluk U & Jembatan Teksas UI
    for (const sx of [-4.2, 4.2]) {
      for (const lz of [-0.65, 0.65]) {
        const leg = new THREE.BoxGeometry(0.12, 0.44, 0.50);
        leg.translate(bx + sx + lz, eastPlazaY + 0.25, 347.8);
        appendPart(vPos, vNorm, vUv, vCol, vIdx, leg, concreteWarm, whiteUV);
      }
      const seat = new THREE.BoxGeometry(1.62, 0.06, 0.52);
      seat.translate(bx + sx, eastPlazaY + 0.49, 347.76);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, seat, [0.49, 0.29, 0.15]);

      const back = new THREE.BoxGeometry(1.62, 0.34, 0.06);
      back.translate(bx + sx, eastPlazaY + 0.70, 348.0);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, back, [0.52, 0.31, 0.16]);
    }

    // =========================================================================
    // 2. INCLUSIVE DUAL-MODE SHORE APPROACHES (CENTRAL STAIRCASE + TWIN ACCESSIBLE RAMPS)
    // =========================================================================
    const smoother01 = (t) => {
      const c = Math.max(0.0, Math.min(1.0, t));
      return c * c * c * (c * (c * 6.0 - 15.0) + 10.0);
    };

    const evalApproachFrac = TerrainDEMLoader.evalTeksasApproachFrac;

    const buildInclusiveShoreApproach = (zOuter, zPortal, yOuter) => {
      const steps = 24;
      const isWest = zOuter < zPortal;
      // Supporting concrete spine wall & twin ramp cheek walls (watertight down to y = -2.6m)
      for (let i = 0; i < steps; i++) {
        const t0 = i / steps;
        const t1 = (i + 1) / steps;
        const tm = 0.5 * (t0 + t1);
        const za = zOuter + (zPortal - zOuter) * t0;
        const zb = zOuter + (zPortal - zOuter) * t1;
        const zm = 0.5 * (za + zb);
        const dzSeg = Math.abs(zb - za);

        const yStepTop = yOuter + evalApproachFrac(tm) * (portalDeckY - yOuter);
        const subH = yStepTop + 2.6;

        // Full-width stepped concrete substructure + stair tread
        const subBox = new THREE.BoxGeometry(5.52, subH, dzSeg + 0.04);
        subBox.translate(bx, yStepTop - subH * 0.5 - 0.03, zm);
        appendPart(vPos, vNorm, vUv, vCol, vIdx, subBox, concreteWarm, whiteUV);

        // Central ceremonial staircase tread (x in [bx - 1.18, bx + 1.18], uniform rise matching walkable grade!)
        const tread = new THREE.BoxGeometry(2.36, 0.06, dzSeg + 0.03);
        tread.translate(bx, yStepTop, zm);
        appendPart(hPos, hNorm, hUv, hCol, hIdx, tread, i % 2 === 0 ? deckPaverGrey : [0.41, 0.42, 0.45]);

        // Smooth linear-grade flanking accessible wheelchair/bicycle ramp decks (0.000m match with sampleTeksasBridgeDeckY!)
        const yR0 = yOuter + evalApproachFrac(t0) * (portalDeckY - yOuter);
        const yR1 = yOuter + evalApproachFrac(t1) * (portalDeckY - yOuter);
        const dyRamp = isWest ? yR1 - yR0 : yR0 - yR1;
        const dzSigned = Math.abs(zb - za);
        const hyp = Math.hypot(dzSigned, dyRamp);
        const pitchX = -Math.atan2(dyRamp, dzSigned);

        for (const sx of [-1, 1]) {
          const rampSlab = new THREE.BoxGeometry(1.38, 0.07, hyp + 0.03);
          rampSlab.rotateX(pitchX);
          rampSlab.translate(bx + sx * 1.94, 0.5 * (yR0 + yR1) - 0.032, zm);
          appendPart(hPos, hNorm, hUv, hCol, hIdx, rampSlab, sx > 0 ? bikeLaneBrick : deckPaverGrey);
        }

        // Approach handrails & posts every 2 steps
        if (i % 2 === 0 || i === steps - 1) {
          const yPostBase = yOuter + evalApproachFrac(tm) * (portalDeckY - yOuter);
          for (const rxOff of [-2.66, -1.22, 1.22, 2.66]) {
            pushTubeH(
              bx + rxOff,
              yPostBase - 0.05,
              zm,
              bx + rxOff,
              yPostBase + 1.06,
              zm,
              0.032,
              0.032,
              8,
              krakatauRed
            );
          }
        }
      }

      // Continuous smooth tubular UI-Yellow top handrails & mid-rails along approach stairs & ramps
      const railSegs = 12;
      for (let r = 0; r < railSegs; r++) {
        const t0 = r / railSegs;
        const t1 = (r + 1) / railSegs;
        const za = zOuter + (zPortal - zOuter) * t0;
        const zb = zOuter + (zPortal - zOuter) * t1;
        const ya = yOuter + evalApproachFrac(t0) * (portalDeckY - yOuter);
        const yb = yOuter + evalApproachFrac(t1) * (portalDeckY - yOuter);
        for (const rxOff of [-2.66, -1.22, 1.22, 2.66]) {
          pushTubeH(bx + rxOff, ya + 1.06, za, bx + rxOff, yb + 1.06, zb, 0.038, 0.038, 8, uiGoldenYellow);
          pushTubeH(bx + rxOff, ya + 0.56, za, bx + rxOff, yb + 0.56, zb, 0.024, 0.024, 6, stayCableSteel);
        }
      }
    };

    buildInclusiveShoreApproach(zRampW0, zW, 0.04);
    buildInclusiveShoreApproach(zRampE1, zE, eastPlazaY);

    // =========================================================================
    // 3. 7-BOX-SEGMENT PARABOLIC BTKC STEEL BOX GIRDER, WALKWAY DECK, & SPLICE JOINTS
    //    (84 longitudinal stations = 12 per each of the 7 Krakatau Steel box segments!)
    // =========================================================================
    const archSegs = 84;
    const topGirderHalfW = 2.08;
    const botGirderHalfW = 1.46;

    // Lofted Parabolic Trapezoidal BTKC Weathering-Steel Box Girder (soffit strictly at y_deck - 1.30m!)
    for (let i = 0; i < archSegs; i++) {
      const z0 = zW + (i / archSegs) * spanLen;
      const z1 = zW + ((i + 1) / archSegs) * spanLen;
      const yD0 = evalArchDeckY(z0);
      const yD1 = evalArchDeckY(z1);
      const yTop0 = yD0 - 0.07;
      const yTop1 = yD1 - 0.07;
      const yBot0 = yD0 - girderDepth;
      const yBot1 = yD1 - girderDepth;

      // Bottom soffit plate (facing -Y downward toward Mini Yacht!)
      pushQuadH(
        bx - botGirderHalfW,
        yBot0,
        z0,
        bx - botGirderHalfW,
        yBot1,
        z1,
        bx + botGirderHalfW,
        yBot1,
        z1,
        bx + botGirderHalfW,
        yBot0,
        z0,
        btkcCharcoal
      );
      // North (+X) inclined box-girder web
      pushQuadH(
        bx + botGirderHalfW,
        yBot0,
        z0,
        bx + botGirderHalfW,
        yBot1,
        z1,
        bx + topGirderHalfW,
        yTop1,
        z1,
        bx + topGirderHalfW,
        yTop0,
        z0,
        btkcCharcoal
      );
      // South (-X) inclined box-girder web
      pushQuadH(
        bx - topGirderHalfW,
        yTop0,
        z0,
        bx - topGirderHalfW,
        yTop1,
        z1,
        bx - botGirderHalfW,
        yBot1,
        z1,
        bx - botGirderHalfW,
        yBot0,
        z0,
        btkcCharcoal
      );
      // Under-deck cantilever soffit wings (facing -Y downward, connecting top of box girder out to crimson fascia girders)
      pushQuadH(
        bx + topGirderHalfW,
        yTop0,
        z0,
        bx + deckHalfW + 0.10,
        yTop0,
        z0,
        bx + deckHalfW + 0.10,
        yTop1,
        z1,
        bx + topGirderHalfW,
        yTop1,
        z1,
        btkcFlangeSteel
      );
      pushQuadH(
        bx - deckHalfW - 0.10,
        yTop0,
        z0,
        bx - topGirderHalfW,
        yTop0,
        z0,
        bx - topGirderHalfW,
        yTop1,
        z1,
        bx - deckHalfW - 0.10,
        yTop1,
        z1,
        btkcFlangeSteel
      );

      // Parabolic Cambered Pedestrian & Bicycle Deck Surface (5.20m clear width, +Y upward normals!)
      const strips = [
        { xA: -deckHalfW, xB: -2.36, col: uiYellowDeep },
        { xA: -2.36, xB: 0.72, col: deckPaverGrey },
        { xA: 0.72, xB: 0.84, col: [0.88, 0.88, 0.90] },
        { xA: 0.84, xB: 2.36, col: bikeLaneBrick },
        { xA: 2.36, xB: deckHalfW, col: uiYellowDeep }
      ];
      for (let s = 0; s < strips.length; s++) {
        const st = strips[s];
        const cA = 0.015 * (1.0 - Math.pow(st.xA / deckHalfW, 2.0));
        const cB = 0.015 * (1.0 - Math.pow(st.xB / deckHalfW, 2.0));
        pushQuadH(
          bx + st.xA,
          yD0 + cA,
          z0,
          bx + st.xA,
          yD1 + cA,
          z1,
          bx + st.xB,
          yD1 + cB,
          z1,
          bx + st.xB,
          yD0 + cB,
          z0,
          st.col
        );
      }

      // Twin Krakatau Steel Crimson-Red Outer Edge Fascia I-Girders (x = bx +/- 2.68m)
      const fIn = deckHalfW + 0.02;
      const fOut = deckHalfW + 0.16;
      const fTop0 = yD0 + 0.16;
      const fTop1 = yD1 + 0.16;
      const fBot0 = yD0 - 0.36;
      const fBot1 = yD1 - 0.36;
      for (const sx of [-1, 1]) {
        const xOuter = bx + sx * fOut;
        const xInner = bx + sx * fIn;
        if (sx > 0) {
          pushQuadH(xOuter, fBot0, z0, xOuter, fBot1, z1, xOuter, fTop1, z1, xOuter, fTop0, z0, krakatauRed);
          pushQuadH(xInner, fTop0, z0, xOuter, fTop0, z0, xOuter, fTop1, z1, xInner, fTop1, z1, krakatauRed);
          pushQuadH(xInner, fBot0, z0, xInner, fTop0, z0, xInner, fTop1, z1, xInner, fBot1, z1, krakatauRedDark);
          pushQuadH(xOuter, fBot0, z0, xInner, fBot0, z0, xInner, fBot1, z1, xOuter, fBot1, z1, krakatauRedDark);
        } else {
          pushQuadH(xOuter, fTop0, z0, xOuter, fTop1, z1, xOuter, fBot1, z1, xOuter, fBot0, z0, krakatauRed);
          pushQuadH(xOuter, fTop0, z0, xInner, fTop0, z0, xInner, fTop1, z1, xOuter, fTop1, z1, krakatauRed);
          pushQuadH(xInner, fTop0, z0, xInner, fBot0, z0, xInner, fBot1, z1, xInner, fTop1, z1, krakatauRedDark);
          pushQuadH(xInner, fBot0, z0, xOuter, fBot0, z0, xOuter, fBot1, z1, xInner, fBot1, z1, krakatauRedDark);
        }
      }
    }

    // 6 Prominent Bolted BTKC Segment-Splice Flange Collars (strictly below deck yDJ - 0.10m so 5.20m walkway stays 100% smooth!)
    for (let segJoint = 1; segJoint <= 6; segJoint++) {
      const zJ = zW + (segJoint / 7) * spanLen;
      const yDJ = evalArchDeckY(zJ);
      const slopeJ = evalArchSlope(zJ);
      const pitchJ = -Math.atan2(slopeJ, 1.0);

      const collarH = girderDepth - 0.10;
      const collar = new THREE.BoxGeometry(4.36, collarH, 0.34);
      collar.rotateX(pitchJ);
      collar.translate(bx, yDJ - 0.12 - collarH * 0.5, zJ);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, collar, btkcFlangeSteel);

      const underCrossBeam = new THREE.BoxGeometry(5.36, 0.32, 0.24);
      underCrossBeam.rotateX(pitchJ);
      underCrossBeam.translate(bx, yDJ - 0.27, zJ);
      appendPart(hPos, hNorm, hUv, hCol, hIdx, underCrossBeam, krakatauRedDark);

      for (const sx of [-1, 1]) {
        const edgeGusset = new THREE.BoxGeometry(0.18, 0.54, 0.26);
        edgeGusset.rotateX(pitchJ);
        edgeGusset.translate(bx + sx * (deckHalfW + 0.09), yDJ - 0.10, zJ);
        appendPart(hPos, hNorm, hUv, hCol, hIdx, edgeGusset, krakatauRed);
      }
    }

    // 28 Transverse Under-Deck Cantilever Outrigger Brackets
    for (let b = 0; b <= 28; b++) {
      const zB = zW + (b / 28) * spanLen;
      const yDB = evalArchDeckY(zB);
      for (const sx of [-1, 1]) {
        pushTubeH(
          bx + sx * (botGirderHalfW + 0.15),
          yDB - 0.72,
          zB,
          bx + sx * (deckHalfW + 0.08),
          yDB - 0.22,
          zB,
          0.075,
          0.065,
          6,
          btkcFlangeSteel
        );
      }
    }

    // =========================================================================
    // 4. KRAKATAU STEEL CRIMSON-RED PARABOLIC THROUGH-ARCH TRUSS, WARREN WEB STRUTS,
    //    OVERHEAD SWAY BRACES, & PEDESTRIAN SAFETY BALUSTRADES
    // =========================================================================
    const trussBays = 28;
    const xTruss = deckHalfW + 0.12; // 2.72m
    const xRail = deckHalfW - 0.06; // 2.54m

    for (let k = 0; k < trussBays; k++) {
      const z0 = zW + (k / trussBays) * spanLen;
      const z1 = zW + ((k + 1) / trussBays) * spanLen;
      const zm = 0.5 * (z0 + z1);
      const yD0 = evalArchDeckY(z0);
      const yD1 = evalArchDeckY(z1);
      const yDm = evalArchDeckY(zm);
      const yC0 = evalArchChordY(z0);
      const yC1 = evalArchChordY(z1);

      for (const sx of [-1, 1]) {
        const xt = bx + sx * xTruss;
        const xr = bx + sx * xRail;

        // Upper Parabolic Crimson-Red Arch Truss Chord
        pushTubeH(xt, yC0, z0, xt, yC1, z1, 0.145, 0.145, 10, krakatauRed);

        // Vertical Steel Truss Hanger Post at z0
        pushTubeH(xt, yD0 + 0.08, z0, xt, yC0, z0, 0.085, 0.085, 8, krakatauRed);

        // Alternating Warren Diagonal Web Strut
        if (k % 2 === 0) {
          pushTubeH(xt, yD0 + 0.12, z0, xt, yC1 - 0.04, z1, 0.068, 0.068, 8, krakatauRedDark);
        } else {
          pushTubeH(xt, yC0 - 0.04, z0, xt, yD1 + 0.12, z1, 0.068, 0.068, 8, krakatauRedDark);
        }

        // Pedestrian & Cyclist Safety Balustrade: UI Golden-Yellow Ergonomic Top Handrail + 2 Mid Tension Bars + Mid Post
        pushTubeH(xr, yD0 + 1.10, z0, xr, yD1 + 1.10, z1, 0.042, 0.042, 8, uiGoldenYellow);
        pushTubeH(xr, yD0 + 0.72, z0, xr, yD1 + 0.72, z1, 0.022, 0.022, 6, stayCableSteel);
        pushTubeH(xr, yD0 + 0.36, z0, xr, yD1 + 0.36, z1, 0.022, 0.022, 6, stayCableSteel);
        pushTubeH(xr, yDm, zm, xr, yDm + 1.10, zm, 0.030, 0.030, 6, krakatauRed);

        // Recessed Warm-White 3000K Photocell LED Handrail Under-Glow Strip (off during clear day, glows at dusk/night/storm!)
        pushTubeH(xr - sx * 0.04, yD0 + 1.05, z0 + 0.25, xr - sx * 0.04, yD1 + 1.05, z1 - 0.25, 0.018, 0.018, 6, warmLedStrip);
      }

      // Overhead Transverse Portal Sway Braces across central high-arch bays (headroom >= 3.1m above deck!)
      if (k >= 6 && k <= 22 && k % 4 === 2) {
        pushTubeH(bx - xTruss, yC0, z0, bx + xTruss, yC0, z0, 0.095, 0.095, 8, krakatauRed);
        pushTubeH(bx - xTruss, yC0 - 0.32, z0, bx - 1.25, yC0, z0, 0.055, 0.055, 6, krakatauRedDark);
        pushTubeH(bx + xTruss, yC0 - 0.32, z0, bx + 1.25, yC0, z0, 0.055, 0.055, 6, krakatauRedDark);
      }
    }
    // Final East vertical truss posts at zE
    for (const sx of [-1, 1]) {
      const xt = bx + sx * xTruss;
      pushTubeH(xt, evalArchDeckY(zE) + 0.08, zE, xt, evalArchChordY(zE), zE, 0.085, 0.085, 8, krakatauRed);
    }

    // =========================================================================
    // 5. 30 TRANSLUCENT-LOOK CERULEAN-BLUE WAVE-RIPPLE SUNROOF CANOPY UNITS
    //    ("30 unit atap menyerupai sunroof yang melengkung seperti riak gelombang berwarna biru")
    // =========================================================================
    const canopyCount = 30;
    const zCanopyStart = zW + 3.4;
    const zCanopySpan = spanLen - 6.8; // 141.2m -> 4.707m pitch per module
    const moduleLenZ = 4.18;
    const moduleHalfW = 2.48;
    const shellThick = 0.052;
    const gridU = 6;
    const gridV = 6;

    for (let m = 0; m < canopyCount; m++) {
      const zm = zCanopyStart + ((m + 0.5) / canopyCount) * zCanopySpan;
      const z0 = zm - moduleLenZ * 0.5;
      const phaseOffset = (m % 2) * 0.55;

      // Supporting dark-steel arched purlin ribs & central photocell LED luminaire under each wave module
      const yBaseMid = evalArchDeckY(zm) + 3.04;
      const yChordMid = Math.max(evalArchChordY(zm) - 0.15, evalArchDeckY(zm) + 2.55);
      for (const sx of [-1, 1]) {
        pushTubeH(
          bx + sx * xTruss,
          yChordMid,
          zm,
          bx + sx * (moduleHalfW - 0.08),
          yBaseMid,
          zm,
          0.045,
          0.040,
          6,
          btkcFlangeSteel
        );
      }
      pushTubeH(bx - moduleHalfW + 0.1, yBaseMid, zm, bx, yBaseMid + 0.34, zm, 0.036, 0.036, 6, btkcFlangeSteel);
      pushTubeH(bx + moduleHalfW - 0.1, yBaseMid, zm, bx, yBaseMid + 0.34, zm, 0.036, 0.036, 6, btkcFlangeSteel);
      if (m % 2 === 0) {
        pushTubeH(bx, yBaseMid + 0.27, zm - 0.75, bx, yBaseMid + 0.27, zm + 0.75, 0.028, 0.028, 6, warmLedStrip);
      }

      // Evaluate double-curved sinusoidal water-ripple wave elevation y(u, v) for sunroof shell #m
      const evalWaveY = (u01, v01) => {
        const zCur = z0 + v01 * moduleLenZ;
        const xRel = (u01 * 2.0 - 1.0) * moduleHalfW;
        const archCrown = 0.34 * (1.0 - Math.pow(xRel / moduleHalfW, 2.0));
        const rippleWave =
          0.16 * Math.sin(v01 * Math.PI * 2.0 + phaseOffset) +
          0.055 * Math.cos(u01 * Math.PI * 3.0);
        return evalArchDeckY(zCur) + 3.02 + archCrown + rippleWave;
      };

      for (let iu = 0; iu < gridU; iu++) {
        const u0 = iu / gridU;
        const u1 = (iu + 1) / gridU;
        const xA = bx + (u0 * 2.0 - 1.0) * moduleHalfW;
        const xB = bx + (u1 * 2.0 - 1.0) * moduleHalfW;
        for (let iv = 0; iv < gridV; iv++) {
          const v0 = iv / gridV;
          const v1 = (iv + 1) / gridV;
          const zA = z0 + v0 * moduleLenZ;
          const zB = z0 + v1 * moduleLenZ;

          const y00 = evalWaveY(u0, v0);
          const y10 = evalWaveY(u1, v0);
          const y11 = evalWaveY(u1, v1);
          const y01 = evalWaveY(u0, v1);

          // Top sky-facing cerulean-blue polycarbonate wave surface (+Y normal)
          pushQuadH(xA, y00, zA, xA, y01, zB, xB, y11, zB, xB, y10, zA, sunroofBlue);
          // Underside pedestrian-facing translucent blue wave soffit (-Y normal)
          pushQuadH(
            xA,
            y00 - shellThick,
            zA,
            xB,
            y10 - shellThick,
            zA,
            xB,
            y11 - shellThick,
            zB,
            xA,
            y01 - shellThick,
            zB,
            sunroofBlueUnderside
          );
        }
      }
    }

    // =========================================================================
    // 6. WEST PORTAL — "PILON LINGGA" (25.0m UI GOLDEN-YELLOW CURVED SHIP'S SAIL / LAYAR KAPAL)
    //    Wide-bowed portal arch (5.50m clear entrance headroom!) -> Masthead apex y = +25.35m
    // =========================================================================
    const linggaMergeY = 11.15;
    const linggaTopY = 25.35;

    // 6A. Sculpted Wide-Bowed Twin Straddling Portal Legs framing the entire 5.20m walkway & blue canopy entrance!
    const legSteps = 14;
    const evalLinggaLegHalfX = (s) => 3.85 * Math.cos(Math.pow(s, 1.85) * Math.PI * 0.5);
    for (const sx of [-1, 1]) {
      for (let i = 0; i < legSteps; i++) {
        const s0 = i / legSteps;
        const s1 = (i + 1) / legSteps;
        const x0 = bx + sx * evalLinggaLegHalfX(s0);
        const x1 = bx + sx * evalLinggaLegHalfX(s1);
        const y0 = 0.35 + s0 * (linggaMergeY - 0.35);
        const y1 = 0.35 + s1 * (linggaMergeY - 0.35);
        const zL0 = zW - 0.12 * s0;
        const zL1 = zW - 0.12 * s1;
        const r0 = 0.46 - 0.11 * s0;
        const r1 = 0.46 - 0.11 * s1;
        pushTubeH(x0, y0, zL0, x1, y1, zL1, r0, r1, 10, uiGoldenYellow, i === legSteps - 1);
      }
    }
    // Portal cross-tie beam under deck & high overhead arch collar at y = 9.65m (5.50m clear headroom above deck!)
    pushTubeH(bx - 3.55, portalDeckY - 0.55, zW, bx + 3.55, portalDeckY - 0.55, zW, 0.28, 0.28, 8, uiYellowDeep);
    const sCollar = (9.65 - 0.35) / (linggaMergeY - 0.35);
    const collarHalfX = evalLinggaLegHalfX(sCollar) + 0.35;
    pushTubeH(bx - collarHalfX, 9.65, zW - 0.10, bx + collarHalfX, 9.65, zW - 0.10, 0.20, 0.20, 8, uiYellowDeep);

    // 6B. Towering 3D Curved Ship's Sail Mast & Billowing Golden-Yellow Steel Sail Fin (y = 10.65m .. 25.35m)
    const sailSegs = 22;
    const evalLinggaMast = (s) => {
      const y = 10.65 + s * (linggaTopY - 10.65);
      // Graceful forward nautical curve toward Teluk U (+Z) peaking mid-mast then sweeping to apex
      const zLead = zW + 2.25 * Math.sin(s * 1.32);
      const chord = 3.95 * Math.sin(Math.PI * Math.pow(Math.max(0.001, s), 0.68)) * (1.0 - 0.28 * s) + 0.28 * (1.0 - s);
      const zTrail = zLead - chord;
      const camberX = 0.34 * Math.sin(Math.PI * s);
      const halfThick = 0.26 * (1.0 - 0.62 * s) + 0.06;
      return { y, zLead, zTrail, chord, camberX, halfThick };
    };

    for (let i = 0; i < sailSegs; i++) {
      const s0 = i / sailSegs;
      const s1 = (i + 1) / sailSegs;
      const m0 = evalLinggaMast(s0);
      const m1 = evalLinggaMast(s1);

      // Leading curved tubular steel sail mast (+Z edge facing Teluk U)
      pushTubeH(
        bx,
        m0.y,
        m0.zLead,
        bx,
        m1.y,
        m1.zLead,
        0.36 * (1.0 - 0.58 * s0),
        0.36 * (1.0 - 0.58 * s1),
        12,
        uiGoldenYellow,
        i === sailSegs - 1
      );
      // Trailing curved leech edge tube (-Z edge facing Rotunda/FT)
      pushTubeH(
        bx,
        m0.y,
        m0.zTrail,
        bx,
        m1.y,
        m1.zTrail,
        0.15 * (1.0 - 0.45 * s0),
        0.15 * (1.0 - 0.45 * s1),
        10,
        uiYellowDeep,
        i === sailSegs - 1
      );

      // Sculpted 3D Cambered Billowing Steel Sail Web (4 chord strips across Z on both +X & -X faces!)
      const chordSteps = 4;
      for (let c = 0; c < chordSteps; c++) {
        const uA = c / chordSteps;
        const uB = (c + 1) / chordSteps;
        const profileA0 = Math.sin(Math.PI * uA);
        const profileB0 = Math.sin(Math.PI * uB);

        const zA0 = m0.zLead + (m0.zTrail - m0.zLead) * uA;
        const zB0 = m0.zLead + (m0.zTrail - m0.zLead) * uB;
        const zA1 = m1.zLead + (m1.zTrail - m1.zLead) * uA;
        const zB1 = m1.zLead + (m1.zTrail - m1.zLead) * uB;

        const xPosA0 = bx + m0.camberX * profileA0 + m0.halfThick * (0.35 + 0.65 * profileA0);
        const xPosB0 = bx + m0.camberX * profileB0 + m0.halfThick * (0.35 + 0.65 * profileB0);
        const xPosA1 = bx + m1.camberX * profileA0 + m1.halfThick * (0.35 + 0.65 * profileA0);
        const xPosB1 = bx + m1.camberX * profileB0 + m1.halfThick * (0.35 + 0.65 * profileB0);

        const xNegA0 = bx + m0.camberX * profileA0 - m0.halfThick * (0.35 + 0.65 * profileA0);
        const xNegB0 = bx + m0.camberX * profileB0 - m0.halfThick * (0.35 + 0.65 * profileB0);
        const xNegA1 = bx + m1.camberX * profileA0 - m1.halfThick * (0.35 + 0.65 * profileA0);
        const xNegB1 = bx + m1.camberX * profileB0 - m1.halfThick * (0.35 + 0.65 * profileB0);

        // North (+X) billowing sail face
        pushQuadH(xPosA0, m0.y, zA0, xPosB0, m0.y, zB0, xPosB1, m1.y, zB1, xPosA1, m1.y, zA1, uiGoldenYellow);
        // South (-X) billowing sail face
        pushQuadH(xNegA0, m0.y, zA0, xNegA1, m1.y, zA1, xNegB1, m1.y, zB1, xNegB0, m0.y, zB0, uiGoldenYellow);
      }

      // Horizontal Nautical Sail Batten Ribs every 3 stations
      if (i > 1 && i < sailSegs - 1 && i % 3 === 0) {
        for (const sx of [-1, 1]) {
          const xRib = bx + m0.camberX * 0.6 + sx * (m0.halfThick + 0.045);
          pushTubeH(xRib, m0.y, m0.zLead - 0.1, xRib, m0.y, m0.zTrail + 0.1, 0.055, 0.045, 6, uiYellowDeep);
        }
      }
    }

    // 3 Sculpted Circular Wind-Relief Sail Oculi Rings + Flashing Red Aviation/Marine Masthead Beacon (y = 25.52m)
    for (const sOculus of [0.30, 0.50, 0.68]) {
      const mO = evalLinggaMast(sOculus);
      const rOc = 0.46 * (1.0 - 0.35 * sOculus);
      const oculus = new THREE.TorusGeometry(rOc, 0.085, 8, 18);
      oculus.rotateY(Math.PI * 0.5);
      oculus.scale(1.8, 1.0, 1.0);
      oculus.translate(bx + mO.camberX * 0.85, mO.y, 0.5 * (mO.zLead + mO.zTrail));
      appendPart(hPos, hNorm, hUv, hCol, hIdx, oculus, btkcFlangeSteel);
    }

    const topMast = evalLinggaMast(1.0);
    const linggaSpire = new THREE.ConeGeometry(0.16, 0.85, 10, 1);
    linggaSpire.translate(bx, topMast.y + 0.40, topMast.zLead);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, linggaSpire, uiGoldenYellow);

    const linggaBeacon = new THREE.SphereGeometry(0.17, 12, 8);
    linggaBeacon.translate(bx, topMast.y + 0.90, topMast.zLead);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, linggaBeacon, [0.96, 0.08, 0.06]);

    // 10 Tensioned Steel Fan Stay Cables from Pilon Lingga (6 forestays to Teluk U arch + 4 backstays to West anchorages)
    const linggaStayHeights = [0.46, 0.64, 0.82];
    for (let c = 0; c < linggaStayHeights.length; c++) {
      const mS = evalLinggaMast(linggaStayHeights[c]);
      const zFore = zW + 13.5 + c * 14.5; // 185.15, 199.65, 214.15
      const yFore = evalArchDeckY(zFore) + 0.18;
      for (const sx of [-1, 1]) {
        pushTubeH(bx, mS.y, mS.zLead - 0.12, bx + sx * xTruss, yFore, zFore, 0.036, 0.036, 6, stayCableSteel);
      }
      if (c >= 1) {
        const zBack = 152.0 + (c - 1) * 5.2;
        const yBack = TerrainDEMLoader.sampleTeksasBridgeDeckY(bx, zBack) || 0.25;
        for (const sx of [-1, 1]) {
          pushTubeH(bx, mS.y, mS.zTrail + 0.12, bx + sx * 2.75, yBack + 0.15, zBack, 0.038, 0.038, 6, stayCableSteel);
        }
      }
    }

    // =========================================================================
    // 7. EAST PORTAL — "PILON YONI" (15.0m UI GOLDEN-YELLOW CURVILINEAR CURVED HIP ARCH)
    //    ("melengkung menyerupai pinggul": slender waist -> flared rounded hip curves -> united crown at y = +15.38m)
    // =========================================================================
    const yoniBaseY = 0.44;
    const yoniCrownY = 15.38; // ~15.0m above East pedestal!
    const yoniSegs = 28;

    // Smooth analytic half-arch curve (s in [0, 1] from pedestal base -> waist -> flared hip -> top crown apex)
    const evalYoniHalfCurve = (s, sx) => {
      const y = yoniBaseY + (yoniCrownY - yoniBaseY) * Math.sin(s * Math.PI * 0.5);
      // Base = 3.85m -> clear waist (~3.35m at s = 0.24, keeping 5.2m deck & railings 100% unobstructed!) ->
      // wide rounded hip flare (~4.82m at s = 0.58, y ~ 12.2m) -> smooth rounded crown apex (0.0m at s = 1.0)!
      const waistDip = -0.35 * Math.exp(-Math.pow((s - 0.23) / 0.12, 2.0));
      const hipFlare = 1.88 * Math.exp(-Math.pow((s - 0.57) / 0.21, 2.0));
      const baseTaper = 3.85 * Math.cos(s * Math.PI * 0.5);
      const halfX = Math.max(0.0, (baseTaper + waistDip + hipFlare) * (1.0 - Math.pow(s, 6.0)));
      // Subtle graceful longitudinal hip lean (+0.68m curvature in Z)
      const zLean = zE + 0.68 * Math.sin(s * Math.PI);
      const radius = 0.42 - 0.10 * s + 0.06 * Math.sin(s * Math.PI);
      return { x: bx + sx * halfX, y, z: zLean, radius };
    };

    for (const sx of [-1, 1]) {
      for (let i = 0; i < yoniSegs; i++) {
        const s0 = i / yoniSegs;
        const s1 = (i + 1) / yoniSegs;
        const p0 = evalYoniHalfCurve(s0, sx);
        const p1 = evalYoniHalfCurve(s1, sx);

        // Twin coupled curvilinear steel hip-arch ribs (West-facing & East-facing flanges tied into a sculpted box-arch!)
        pushTubeH(p0.x, p0.y, p0.z - 0.34, p1.x, p1.y, p1.z - 0.34, p0.radius * 0.72, p1.radius * 0.72, 10, uiGoldenYellow, true);
        pushTubeH(p0.x, p0.y, p0.z + 0.34, p1.x, p1.y, p1.z + 0.34, p0.radius * 0.72, p1.radius * 0.72, 10, uiGoldenYellow, true);

        // Connecting golden-yellow web struts between the twin curvilinear ribs
        if (i % 2 === 0) {
          pushTubeH(p0.x, p0.y, p0.z - 0.34, p0.x, p0.y, p0.z + 0.34, 0.16, 0.16, 8, uiYellowDeep);
        }
      }
    }

    // Portal under-deck cross-beam & upper crown medallion ring on Pilon Yoni
    pushTubeH(bx - 3.15, portalDeckY - 0.55, zE, bx + 3.15, portalDeckY - 0.55, zE, 0.26, 0.26, 8, uiYellowDeep);
    const yoniCrownRing = new THREE.TorusGeometry(0.62, 0.13, 10, 20);
    yoniCrownRing.rotateY(Math.PI * 0.5);
    yoniCrownRing.translate(bx, yoniCrownY - 1.15, zE + 0.18);
    appendPart(hPos, hNorm, hUv, hCol, hIdx, yoniCrownRing, uiYellowDeep);

    // 8 Tensioned Steel Fan Stay Cables from Pilon Yoni's Upper Hip Crown (4 forestays to Teluk U arch + 4 backstays)
    for (let c = 0; c < 2; c++) {
      const sAttach = 0.68 + c * 0.15;
      for (const sx of [-1, 1]) {
        const pA = evalYoniHalfCurve(sAttach, sx);
        const zFore = zE - (13.0 + c * 15.0); // 306.65, 291.65
        const yFore = evalArchDeckY(zFore) + 0.18;
        pushTubeH(pA.x, pA.y, pA.z - 0.30, bx + sx * xTruss, yFore, zFore, 0.036, 0.036, 6, stayCableSteel);

        const zBack = 334.0 + c * 5.2;
        const yBack = TerrainDEMLoader.sampleTeksasBridgeDeckY(bx, zBack) || eastPlazaY;
        pushTubeH(pA.x, pA.y, pA.z + 0.30, bx + sx * 2.75, yBack + 0.15, zBack, 0.036, 0.036, 6, stayCableSteel);
      }
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
   * Shared Dual-Material Geometry Accumulator (Material 1: Asphalt/Steel/Terracotta/Teak/Acrylic,
   * Material 2: Precast/Zebra Concrete). Guarantees 100% outward unit normals & CCW front-face winding!
   */
  static _createDualBufferAccumulator() {
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

    const whiteUV = [0.055, 0.5];

    const appendPart = (tPos, tNorm, tUv, tCol, tIdx, geo, col, uvFixed = null) => {
      if (!geo.attributes.normal) geo.computeVertexNormals();
      const p = geo.attributes.position.array;
      const n = geo.attributes.normal.array;
      const uArr = geo.attributes.uv ? geo.attributes.uv.array : null;
      const base = tPos.length / 3;
      for (let i = 0; i < geo.attributes.position.count; i++) {
        const px = p[i * 3];
        const py = p[i * 3 + 1];
        const pz = p[i * 3 + 2];
        let nx = n[i * 3];
        let ny = n[i * 3 + 1];
        let nz = n[i * 3 + 2];
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
        tPos.push(px, py, pz);
        tNorm.push(nx, ny, nz);
        if (uvFixed) {
          tUv.push(uvFixed[0], uvFixed[1]);
        } else if (uArr) {
          tUv.push(uArr[i * 2], uArr[i * 2 + 1]);
        } else {
          tUv.push(px * 0.25, pz * 0.25);
        }
        tCol.push(col[0], col[1], col[2]);
      }
      if (geo.index) {
        const idx = geo.index.array;
        for (let i = 0; i < idx.length; i++) tIdx.push(base + idx[i]);
      } else {
        for (let i = 0; i < geo.attributes.position.count; i++) tIdx.push(base + i);
      }
    };

    const pushOrientedQuad = (
      tPos,
      tNorm,
      tUv,
      tCol,
      tIdx,
      ax,
      ay,
      az,
      bx,
      by,
      bz,
      cx,
      cy,
      cz,
      dx,
      dy,
      dz,
      col,
      aimNx = null,
      aimNy = null,
      aimNz = null,
      uvA = null,
      uvB = null,
      uvC = null,
      uvD = null
    ) => {
      const e1x = bx - ax;
      const e1y = by - ay;
      const e1z = bz - az;
      const e2x = dx - ax;
      const e2y = dy - ay;
      const e2z = dz - az;
      let nx = e1y * e2z - e1z * e2y;
      let ny = e1z * e2x - e1x * e2z;
      let nz = e1x * e2y - e1y * e2x;
      let len = Math.hypot(nx, ny, nz);
      if (len < 1e-8) return;
      nx /= len;
      ny /= len;
      nz /= len;

      let flip = false;
      if (aimNx !== null) {
        if (nx * aimNx + ny * aimNy + nz * aimNz < 0.0) {
          flip = true;
          nx = -nx;
          ny = -ny;
          nz = -nz;
        }
      }

      const base = tPos.length / 3;
      tPos.push(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz);
      for (let k = 0; k < 4; k++) {
        tNorm.push(nx, ny, nz);
        tCol.push(col[0], col[1], col[2]);
      }
      if (uvA && uvB && uvC && uvD) {
        tUv.push(uvA[0], uvA[1], uvB[0], uvB[1], uvC[0], uvC[1], uvD[0], uvD[1]);
      } else {
        tUv.push(ax * 0.2, az * 0.2, bx * 0.2, bz * 0.2, cx * 0.2, cz * 0.2, dx * 0.2, dz * 0.2);
      }

      if (!flip) {
        tIdx.push(base, base + 1, base + 2, base, base + 2, base + 3);
      } else {
        tIdx.push(base, base + 3, base + 2, base, base + 2, base + 1);
      }
    };

    const pushTube = (isH, x0, y0, z0, x1, y1, z1, r0, r1, segs, col, capped = false, uvFixed = null) => {
      const dx = x1 - x0;
      const dy = y1 - y0;
      const dz = z1 - z0;
      const len = Math.hypot(dx, dy, dz);
      if (len < 1e-4) return;
      const tx = dx / len;
      const ty = dy / len;
      const tz = dz / len;
      let rx = 0.0;
      let ry = 1.0;
      let rz = 0.0;
      if (Math.abs(ty) >= 0.92) {
        rx = 1.0;
        ry = 0.0;
        rz = 0.0;
      }
      let ux = ry * tz - rz * ty;
      let uy = rz * tx - rx * tz;
      let uz = rx * ty - ry * tx;
      const uLen = Math.hypot(ux, uy, uz) || 1.0;
      ux /= uLen;
      uy /= uLen;
      uz /= uLen;
      const vx = ty * uz - tz * uy;
      const vy = tz * ux - tx * uz;
      const vz = tx * uy - ty * ux;

      const tPos = isH ? hPos : vPos;
      const tNorm = isH ? hNorm : vNorm;
      const tUv = isH ? hUv : vUv;
      const tCol = isH ? hCol : vCol;
      const tIdx = isH ? hIdx : vIdx;
      const fixedUV = isH ? uvFixed : uvFixed || whiteUV;

      const pushTVert = (px, py, pz, nx, ny, nz) => {
        const idx = tPos.length / 3;
        const nl = Math.hypot(nx, ny, nz) || 1.0;
        tPos.push(px, py, pz);
        tNorm.push(nx / nl, ny / nl, nz / nl);
        if (fixedUV) tUv.push(fixedUV[0], fixedUV[1]);
        else tUv.push(px * 0.2, pz * 0.2);
        tCol.push(col[0], col[1], col[2]);
        return idx;
      };

      const taperSlope = (r0 - r1) / len;
      const ring0 = [];
      const ring1 = [];
      for (let k = 0; k <= segs; k++) {
        const th = (k / segs) * Math.PI * 2.0;
        const c = Math.cos(th);
        const s = Math.sin(th);
        const rnx = ux * c + vx * s;
        const rny = uy * c + vy * s;
        const rnz = uz * c + vz * s;
        const nx = rnx + tx * taperSlope;
        const ny = rny + ty * taperSlope;
        const nz = rnz + tz * taperSlope;
        ring0.push(pushTVert(x0 + rnx * r0, y0 + rny * r0, z0 + rnz * r0, nx, ny, nz));
        ring1.push(pushTVert(x1 + rnx * r1, y1 + rny * r1, z1 + rnz * r1, nx, ny, nz));
      }
      for (let k = 0; k < segs; k++) {
        const b0 = ring0[k];
        const b1 = ring0[k + 1];
        const t1 = ring1[k + 1];
        const t0 = ring1[k];
        // Outward CCW winding: (b1 - b0) x (t1 - b0) = dP x (len * t) = +len * r * dth * r_n!
        tIdx.push(b0, b1, t1, b0, t1, t0);
      }
      if (capped) {
        const c0 = pushTVert(x0, y0, z0, -tx, -ty, -tz);
        const c1 = pushTVert(x1, y1, z1, tx, ty, tz);
        for (let k = 0; k < segs; k++) {
          const th0 = (k / segs) * Math.PI * 2.0;
          const th1 = ((k + 1) / segs) * Math.PI * 2.0;
          const p0a = pushTVert(
            x0 + (ux * Math.cos(th0) + vx * Math.sin(th0)) * r0,
            y0 + (uy * Math.cos(th0) + vy * Math.sin(th0)) * r0,
            z0 + (uz * Math.cos(th0) + vz * Math.sin(th0)) * r0,
            -tx,
            -ty,
            -tz
          );
          const p0b = pushTVert(
            x0 + (ux * Math.cos(th1) + vx * Math.sin(th1)) * r0,
            y0 + (uy * Math.cos(th1) + vy * Math.sin(th1)) * r0,
            z0 + (uz * Math.cos(th1) + vz * Math.sin(th1)) * r0,
            -tx,
            -ty,
            -tz
          );
          // At end 0 (outward normal -t): (p0a - c0) x (p0b - c0) = u x v * dth = +t * dth -> reverse to (c0, p0b, p0a)!
          tIdx.push(c0, p0b, p0a);

          const p1a = pushTVert(
            x1 + (ux * Math.cos(th0) + vx * Math.sin(th0)) * r1,
            y1 + (uy * Math.cos(th0) + vy * Math.sin(th0)) * r1,
            z1 + (uz * Math.cos(th0) + vz * Math.sin(th0)) * r1,
            tx,
            ty,
            tz
          );
          const p1b = pushTVert(
            x1 + (ux * Math.cos(th1) + vx * Math.sin(th1)) * r1,
            y1 + (uy * Math.cos(th1) + vy * Math.sin(th1)) * r1,
            z1 + (uz * Math.cos(th1) + vz * Math.sin(th1)) * r1,
            tx,
            ty,
            tz
          );
          tIdx.push(c1, p1a, p1b);
        }
      }
    };

    return {
      whiteUV,
      appendH: (geo, col, uvFixed = null) => appendPart(hPos, hNorm, hUv, hCol, hIdx, geo, col, uvFixed),
      appendV: (geo, col, uvFixed = whiteUV) => appendPart(vPos, vNorm, vUv, vCol, vIdx, geo, col, uvFixed),
      quadH: (ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, col, nx = null, ny = null, nz = null) =>
        pushOrientedQuad(hPos, hNorm, hUv, hCol, hIdx, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, col, nx, ny, nz),
      quadV: (
        ax,
        ay,
        az,
        bx,
        by,
        bz,
        cx,
        cy,
        cz,
        dx,
        dy,
        dz,
        col,
        nx = null,
        ny = null,
        nz = null,
        uvA = whiteUV,
        uvB = whiteUV,
        uvC = whiteUV,
        uvD = whiteUV
      ) =>
        pushOrientedQuad(
          vPos,
          vNorm,
          vUv,
          vCol,
          vIdx,
          ax,
          ay,
          az,
          bx,
          by,
          bz,
          cx,
          cy,
          cz,
          dx,
          dy,
          dz,
          col,
          nx,
          ny,
          nz,
          uvA,
          uvB,
          uvC,
          uvD
        ),
      tubeH: (x0, y0, z0, x1, y1, z1, r0, r1, segs, col, capped = false) =>
        pushTube(true, x0, y0, z0, x1, y1, z1, r0, r1, segs, col, capped),
      tubeV: (x0, y0, z0, x1, y1, z1, r0, r1, segs, col, capped = false) =>
        pushTube(false, x0, y0, z0, x1, y1, z1, r0, r1, segs, col, capped, whiteUV),
      finalize: () => {
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
    };
  }

  /**
   * Builds the Integrated 3D Slope-Normal Island Road Network ("Jaringan Jalan Terpadu Pulau U & Transisi Pulau I"):
   * - 4-strip cambered asphalt carriageways conforming smoothly to cut-and-fill hill grades
   * - 3D inlaid UI Golden-Yellow dashed centerlines & continuous crisp white shoulder lines
   * - 3D chamfered concrete & zebra safety curbs with -3.4m watertight sub-terrain skirts
   * - Galvanized steel W-beam hillside safety guardrails on elevated mountain cuts/embankments
   * - Flared asphalt T-junction aprons & Jembatan Teksas UI East FIB Pedestrian Drop-Off Connector
   * Merged via NexusBatchOptimizer into the existing 2 corridor draw calls (+0 Draw Calls!).
   */
  static _buildPulauUAndPulauIRoadNetwork() {
    const acc = SplineCorridorBuilder._createDualBufferAccumulator();
    const { corridors } = TerrainDEMLoader.getRoadNetwork3DCorridors();

    const asphaltCrown = [0.20, 0.21, 0.23];
    const asphaltGutter = [0.16, 0.17, 0.19];
    const uiYellowLine = [0.96, 0.77, 0.12];
    const whiteEdgeLine = [0.94, 0.95, 0.97];
    const concreteCurb = [0.92, 0.91, 0.87];
    const concreteSkirt = [0.76, 0.74, 0.70];
    const guardRailSteel = [0.66, 0.69, 0.73];
    const guardPostSteel = [0.46, 0.49, 0.53];

    // Helper: smooth [0..1] curb-height factor that drops curbs flush (0.0m) across T-junction throats,
    // ferry terminal forecourts, campus core entrance, aircraft apron throat, & underpass ramp thresholds!
    const evalCurbRiseFactor = (selfCorrIdx, qx, qz) => {
      let factor = 1.0;
      for (let c = 0; c < corridors.length; c++) {
        if (c === selfCorrIdx) continue;
        const oc = corridors[c];
        if (qx < oc.minX || qx > oc.maxX || qz < oc.minZ || qz > oc.maxZ) continue;
        const pts = oc.points;
        let minD2 = Infinity;
        for (let k = 0; k < pts.length - 1; k++) {
          const a = pts[k];
          const b = pts[k + 1];
          const dx = b.x - a.x;
          const dz = b.z - a.z;
          const len2 = dx * dx + dz * dz || 1e-8;
          const t = Math.max(0.0, Math.min(1.0, ((qx - a.x) * dx + (qz - a.z) * dz) / len2));
          const px = a.x + dx * t;
          const pz = a.z + dz * t;
          const d2 = (qx - px) * (qx - px) + (qz - pz) * (qz - pz);
          if (d2 < minD2) minD2 = d2;
        }
        const d = Math.sqrt(minD2);
        const fOther = PhysicsKernel.smootherstep(oc.halfWidth + 0.35, oc.halfWidth + 1.85, d);
        if (fOther < factor) factor = fOther;
      }

      // West Campus Courtyard entrance (-66.5, 75.0)
      if (qx > -68.5 && qz > 67.0 && qz < 83.0) {
        factor = Math.min(factor, PhysicsKernel.smootherstep(-66.8, -68.4, qx));
      }
      // Southern Underpass West & East Ramp throats (-356.0, 272.0 & 670.0)
      if (Math.abs(qx - TerrainDEMLoader.UNDERPASS_X) < 5.2) {
        if (qz > 269.8 && qz < 273.5) {
          factor = Math.min(factor, PhysicsKernel.smootherstep(271.8, 270.0, qz));
        } else if (qz > 668.5 && qz < 672.2) {
          factor = Math.min(factor, PhysicsKernel.smootherstep(670.2, 672.0, qz));
        }
      }
      // Dermaga Utara Pulau U & Pulau I Forecourt Plazas (flush pier-side pedestrian & shuttle turnaround access)
      if (qz > 449.5 && qz < 460.0 && Math.abs(qx - TerrainDEMLoader.DERMAGA_U_X) < 11.5) {
        factor = Math.min(factor, PhysicsKernel.smootherstep(9.0, 11.5, Math.abs(qx - TerrainDEMLoader.DERMAGA_U_X)));
      }
      if (qz > 612.0 && qz < 626.0 && Math.abs(qx - TerrainDEMLoader.DERMAGA_I_X) < 13.5) {
        factor = Math.min(factor, PhysicsKernel.smootherstep(11.0, 13.5, Math.abs(qx - TerrainDEMLoader.DERMAGA_I_X)));
      }
      // Seamless flush merge across interior median of Dermaga Pulau I bus turnaround cul-de-sac
      if (qz > 628.2 && qz < 633.8 && qx > -119.5 && qx < -85.0) {
        const fx = Math.max(
          PhysicsKernel.smootherstep(-87.0, -85.0, qx),
          PhysicsKernel.smootherstep(-116.5, -119.5, qx)
        );
        const fz = Math.max(
          PhysicsKernel.smootherstep(629.4, 628.2, qz),
          PhysicsKernel.smootherstep(632.6, 633.8, qz)
        );
        factor = Math.min(factor, Math.max(fx, fz));
      }
      // Pulau I West Aircraft Apron connection (-254.0, 660.2)
      if (qz > 658.2 && qz < 661.5 && Math.abs(qx + 254.0) < 5.5) {
        factor = Math.min(factor, PhysicsKernel.smootherstep(659.8, 658.2, qz));
      }
      // Jembatan Teksas UI East FIB Plaza drop-off (-28.0, 353.5)
      if (qz > 348.5 && qz < 354.2 && Math.abs(qx - TerrainDEMLoader.TEKSAS_BRIDGE_X) < 4.2) {
        factor = Math.min(factor, PhysicsKernel.smootherstep(2.6, 4.2, Math.abs(qx - TerrainDEMLoader.TEKSAS_BRIDGE_X)));
      }
      // Kompleks DPPD–STP–ILRC 360° Ring Road & Outer Sidewalk Promenade South & East Spur Throats
      const rDppd = Math.hypot(qx - TerrainDEMLoader.DPPD_CX, qz - TerrainDEMLoader.DPPD_CZ);
      if (rDppd < TerrainDEMLoader.DPPD_RING_OUTER_R + 4.0) {
        factor = Math.min(
          factor,
          PhysicsKernel.smootherstep(
            TerrainDEMLoader.DPPD_RING_OUTER_R + 1.85,
            TerrainDEMLoader.DPPD_RING_OUTER_R + 4.0,
            rDppd
          )
        );
      }
      return factor;
    };

    for (let cIdx = 0; cIdx < corridors.length; cIdx++) {
      const corr = corridors[cIdx];
      // Loft directly along the exact SSoST 3.0m-spaced 3D graded vertices (0.000m spline-vs-DEM discrepancy!)
      const pts = corr.points;
      const segs = pts.length - 1;
      const hw = corr.halfWidth;
      const cw = 0.46;
      const curbRise = 0.13;

      const frames = [];
      let arc = 0.0;

      for (let i = 0; i <= segs; i++) {
        const pos = pts[i];
        const pPrev = pts[Math.max(0, i - 1)];
        const pNext = pts[Math.min(segs, i + 1)];

        const tan = new THREE.Vector3().subVectors(pNext, pPrev);
        if (tan.lengthSq() < 1e-8) tan.set(0, 0, 1);
        else tan.normalize();

        const side = new THREE.Vector3(tan.z, 0.0, -tan.x);
        if (side.lengthSq() < 1e-8) side.set(1, 0, 0);
        else side.normalize();

        const up = new THREE.Vector3().crossVectors(tan, side).normalize();
        if (up.y < 0.0) up.negate();

        if (i > 0) arc += frames[i - 1].pos.distanceTo(pos);
        const curbL = evalCurbRiseFactor(cIdx, pos.x - side.x * hw, pos.z - side.z * hw);
        const curbR = evalCurbRiseFactor(cIdx, pos.x + side.x * hw, pos.z + side.z * hw);
        frames.push({ pos, tan, side, up, arc, curbL, curbR });
      }

      const latOffsets = [-hw, -hw * 0.5, 0.0, hw * 0.5, hw];
      const latLifts = [0.020, 0.035, 0.042, 0.035, 0.020];
      const latCols = [asphaltGutter, asphaltCrown, asphaltCrown, asphaltCrown, asphaltGutter];

      for (let i = 0; i < segs; i++) {
        const f0 = frames[i];
        const f1 = frames[i + 1];

        // 1. 4-Strip Cambered 3D Slope-Normal Asphalt Carriageway
        for (let s = 0; s < 4; s++) {
          const oA = latOffsets[s];
          const oB = latOffsets[s + 1];
          const yA = latLifts[s];
          const yB = latLifts[s + 1];
          const col = 0.5 * (latCols[s][0] + latCols[s + 1][0]) > 0.185 ? asphaltCrown : asphaltGutter;

          acc.quadH(
            f0.pos.x + f0.side.x * oA,
            f0.pos.y + yA,
            f0.pos.z + f0.side.z * oA,
            f1.pos.x + f1.side.x * oA,
            f1.pos.y + yA,
            f1.pos.z + f1.side.z * oA,
            f1.pos.x + f1.side.x * oB,
            f1.pos.y + yB,
            f1.pos.z + f1.side.z * oB,
            f0.pos.x + f0.side.x * oB,
            f0.pos.y + yB,
            f0.pos.z + f0.side.z * oB,
            col,
            f0.up.x,
            f0.up.y,
            f0.up.z
          );
        }

        // 2. 3D Inlaid Highway Markings (+0.013m above asphalt crown/shoulders, zero Z-fighting!)
        const midArc = 0.5 * (f0.arc + f1.arc);
        if (midArc % 7.2 < 3.6 && i > 0 && i < segs - 1) {
          const clW = 0.085;
          acc.quadH(
            f0.pos.x - f0.side.x * clW,
            f0.pos.y + 0.056,
            f0.pos.z - f0.side.z * clW,
            f1.pos.x - f1.side.x * clW,
            f1.pos.y + 0.056,
            f1.pos.z - f1.side.z * clW,
            f1.pos.x + f1.side.x * clW,
            f1.pos.y + 0.056,
            f1.pos.z + f1.side.z * clW,
            f0.pos.x + f0.side.x * clW,
            f0.pos.y + 0.056,
            f0.pos.z + f0.side.z * clW,
            uiYellowLine,
            f0.up.x,
            f0.up.y,
            f0.up.z
          );
        }

        for (const sgn of [-1.0, 1.0]) {
          const cFac0 = sgn < 0.0 ? f0.curbL : f0.curbR;
          const cFac1 = sgn < 0.0 ? f1.curbL : f1.curbR;
          if (cFac0 > 0.35 && cFac1 > 0.35) {
            const eIn = sgn * (hw - 0.36);
            const eOut = sgn * (hw - 0.20);
            acc.quadH(
              f0.pos.x + f0.side.x * eIn,
              f0.pos.y + 0.036,
              f0.pos.z + f0.side.z * eIn,
              f1.pos.x + f1.side.x * eIn,
              f1.pos.y + 0.036,
              f1.pos.z + f1.side.z * eIn,
              f1.pos.x + f1.side.x * eOut,
              f1.pos.y + 0.034,
              f1.pos.z + f1.side.z * eOut,
              f0.pos.x + f0.side.x * eOut,
              f0.pos.y + 0.034,
              f0.pos.z + f0.side.z * eOut,
              whiteEdgeLine,
              f0.up.x,
              f0.up.y,
              f0.up.z
            );
          }
        }

        // 3. Left & Right Chamfered Concrete Curbs (flush-dropped across intersections!) + -3.4m Watertight Skirts
        const useZebra = Math.abs(f0.tan.x * f1.tan.z - f0.tan.z * f1.tan.x) > 0.028 || i < 3 || i > segs - 4;
        const uv0 = useZebra ? [f0.arc * 0.14, 0.5] : acc.whiteUV;
        const uv1 = useZebra ? [f1.arc * 0.14, 0.5] : acc.whiteUV;

        for (const sgn of [-1.0, 1.0]) {
          const cFac0 = sgn < 0.0 ? f0.curbL : f0.curbR;
          const cFac1 = sgn < 0.0 ? f1.curbL : f1.curbR;
          const rIn = sgn * hw;
          const rTop = sgn * (hw + 0.08);
          const rOut = sgn * (hw + cw);
          const yG0 = f0.pos.y + 0.020;
          const yG1 = f1.pos.y + 0.020;
          const yT0 = yG0 + curbRise * cFac0;
          const yT1 = yG1 + curbRise * cFac1;
          const yOuterTop0 = yG0 + (curbRise - 0.015) * cFac0 - 0.025 * (1.0 - cFac0);
          const yOuterTop1 = yG1 + (curbRise - 0.015) * cFac1 - 0.025 * (1.0 - cFac1);
          const yS0 = f0.pos.y - 3.40;
          const yS1 = f1.pos.y - 3.40;

          if (cFac0 > 0.35 && cFac1 > 0.35) {
            // Inner chamfered curb riser facing road centerline (-sgn * side)
            acc.quadV(
              f0.pos.x + f0.side.x * rIn,
              yG0,
              f0.pos.z + f0.side.z * rIn,
              f1.pos.x + f1.side.x * rIn,
              yG1,
              f1.pos.z + f1.side.z * rIn,
              f1.pos.x + f1.side.x * rTop,
              yT1,
              f1.pos.z + f1.side.z * rTop,
              f0.pos.x + f0.side.x * rTop,
              yT0,
              f0.pos.z + f0.side.z * rTop,
              concreteCurb,
              -sgn * f0.side.x * 0.8 + f0.up.x * 0.6,
              0.6,
              -sgn * f0.side.z * 0.8 + f0.up.z * 0.6,
              uv0,
              uv1,
              uv1,
              uv0
            );

            // Curb top coping (+up)
            acc.quadV(
              f0.pos.x + f0.side.x * rTop,
              yT0,
              f0.pos.z + f0.side.z * rTop,
              f1.pos.x + f1.side.x * rTop,
              yT1,
              f1.pos.z + f1.side.z * rTop,
              f1.pos.x + f1.side.x * rOut,
              yOuterTop1,
              f1.pos.z + f1.side.z * rOut,
              f0.pos.x + f0.side.x * rOut,
              yOuterTop0,
              f0.pos.z + f0.side.z * rOut,
              concreteCurb,
              f0.up.x,
              f0.up.y,
              f0.up.z,
              uv0,
              uv1,
              uv1,
              uv0
            );
          }

          // Outer watertight concrete embankment skirt (+sgn * side)
          acc.quadV(
            f0.pos.x + f0.side.x * rOut,
            yOuterTop0,
            f0.pos.z + f0.side.z * rOut,
            f1.pos.x + f1.side.x * rOut,
            yOuterTop1,
            f1.pos.z + f1.side.z * rOut,
            f1.pos.x + f1.side.x * rOut,
            yS1,
            f1.pos.z + f1.side.z * rOut,
            f0.pos.x + f0.side.x * rOut,
            yS0,
            f0.pos.z + f0.side.z * rOut,
            concreteSkirt,
            sgn * f0.side.x,
            0.0,
            sgn * f0.side.z
          );

          // 4. Hillside Galvanized Steel W-Beam Safety Guardrail wherever road traverses elevated hills (y > 4.2m)
          if (f0.pos.y > 4.2 && f1.pos.y > 4.2 && cFac0 > 0.8 && cFac1 > 0.8 && i > 2 && i < segs - 3) {
            const gOff = sgn * (hw + cw - 0.14);
            const gx0 = f0.pos.x + f0.side.x * gOff;
            const gz0 = f0.pos.z + f0.side.z * gOff;
            const gx1 = f1.pos.x + f1.side.x * gOff;
            const gz1 = f1.pos.z + f1.side.z * gOff;
            acc.tubeH(gx0, yT0 + 0.56, gz0, gx1, yT1 + 0.56, gz1, 0.055, 0.055, 6, guardRailSteel);
            if (i % 2 === 0) {
              acc.tubeH(gx0, yT0 - 0.05, gz0, gx0, yT0 + 0.64, gz0, 0.045, 0.045, 6, guardPostSteel, true);
            }
          }
        }
      }
    }

    // 5. Flush Asphalt T-Junction Throat Fillets, Underpass Threshold Seals, & Jembatan Teksas East Drop-Off Link
    const addJunctionPad = (cx, yTop, cz, sx, sz, yaw = 0.0, col = asphaltCrown) => {
      const deck = new THREE.BoxGeometry(sx, 0.046, sz);
      if (yaw !== 0.0) deck.rotateY(yaw);
      deck.translate(cx, yTop - 0.021, cz);
      acc.appendH(deck, col);

      const sub = new THREE.BoxGeometry(Math.max(0.4, sx - 0.30), 2.4, Math.max(0.4, sz - 0.30));
      if (yaw !== 0.0) sub.rotateY(yaw);
      sub.translate(cx, yTop - 1.24, cz);
      acc.appendV(sub, concreteSkirt);
    };

    // West Campus Courtyard Road <-> Pulau U West-South Arterial T-Junction (-65.6, 75.0)
    addJunctionPad(-65.6, 0.038, 75.0, 4.4, 7.6);
    // South Arterial <-> Right-Arm Boulevard T-Junction (-352.6, 263.5 — stays strictly inside outer -X curb!)
    const rgSouthJunc = TerrainDEMLoader.evalRoadNetworkCorridorGrade(-356.0, 263.0);
    addJunctionPad(-352.6, (rgSouthJunc ? rgSouthJunc.targetY : 0.92) + 0.039, 263.5, 6.8, 8.2, -0.18);
    // Right-Arm Boulevard <-> Mid-Peninsula Saddle Connector T-Junction (-175.0, 439.8 — stays strictly inside +Z curb!)
    const rgSaddle = TerrainDEMLoader.evalRoadNetworkCorridorGrade(-175.0, 443.0);
    addJunctionPad(-175.0, (rgSaddle ? rgSaddle.targetY : 1.65) + 0.039, 439.8, 8.4, 6.4, 0.14);
    // Pulau I West Transit Road <-> West Aircraft Apron Spur T-Junction (-254.0, 649.0 — stays strictly inside -Z curb!)
    const rgApronJunc = TerrainDEMLoader.evalRoadNetworkCorridorGrade(-254.0, 646.0);
    addJunctionPad(-254.0, (rgApronJunc ? rgApronJunc.targetY : 0.48) + 0.039, 649.0, 8.2, 6.0);
    // Watertight Camber-to-Ramp Threshold Seals at West (z = 272.0m) & East (z = 670.0m) Subsea Underpass Ramps
    addJunctionPad(
      TerrainDEMLoader.UNDERPASS_X,
      TerrainDEMLoader.UNDERPASS_WEST_RAMP_TOP_Y + 0.039,
      TerrainDEMLoader.UNDERPASS_WEST_RAMP_START_Z,
      7.42,
      1.35
    );
    addJunctionPad(
      TerrainDEMLoader.UNDERPASS_X,
      TerrainDEMLoader.UNDERPASS_EAST_RAMP_TOP_Y + 0.039,
      TerrainDEMLoader.UNDERPASS_EAST_RAMP_END_Z,
      7.22,
      1.35
    );
    // Jembatan Teksas UI East FIB Plaza Pedestrian Drop-Off Link (-28.0, z = 348.8..352.0, between the two x = +-4.2 benches!)
    addJunctionPad(
      TerrainDEMLoader.TEKSAS_BRIDGE_X,
      TerrainDEMLoader.TEKSAS_EAST_PLAZA_Y + 0.038,
      350.6,
      5.2,
      4.4,
      0.0,
      [0.46, 0.45, 0.42]
    );
    // Kompleks DPPD–STP–ILRC South & East Spur T-Junctions + Ring-Road Mouth Transition Pads
    const rgDppdSouth = TerrainDEMLoader.evalRoadNetworkCorridorGrade(-302.5, 338.8);
    addJunctionPad(-301.2, (rgDppdSouth ? rgDppdSouth.targetY : 4.85) + 0.039, 338.5, 6.6, 7.6, -0.19);
    addJunctionPad(-291.0, TerrainDEMLoader.DPPD_PLATEAU_Y + 0.044, 336.6, 5.4, 7.2, -0.21);

    const rgDppdEast = TerrainDEMLoader.evalRoadNetworkCorridorGrade(-156.0, 355.0);
    addJunctionPad(-157.8, (rgDppdEast ? rgDppdEast.targetY : 1.72) + 0.039, 354.5, 6.8, 7.6, 0.25);
    addJunctionPad(-219.4, TerrainDEMLoader.DPPD_PLATEAU_Y + 0.044, 338.1, 5.4, 7.2, 0.26);

    return acc.finalize();
  }

  /**
   * Builds the Southern Undersea Underpass ("Underpass Bawah Laut Selatan antara Pulau U dan Pulau I")
   * along x = -356.0m beneath southern Selat UI:
   * - West Open-Cut Retained Ramp on SE Pulau U (z = 272 -> 326m, descending +0.92m -> -8.20m)
   * - East Open-Cut Retained Ramp on SW Pulau I (z = 624 -> 670m, ascending -8.20m -> +0.56m)
   * - 3.10m-wide Ground-Rim Sealing Parapet Collars & Buttressed Retaining Walls (zero grass overhang!)
   * - West & East Nusantara-UI Golden-Yellow Portal Arches + Twin Tajug Subsea Ventilation Towers
   * - 298.0m Enclosed Subsea Immersed-Tube Tunnel Bore (z = 326 -> 624m, floor y = -8.20m, clear height 4.15m):
   *   7.20m dual-lane asphalt carriageway, 2 x 0.95m raised pedestrian walkways, 26 steel portal ribs,
   *   16 panoramic subsea marine-acrylic observation windows (glowing with Selat UI ocean caustics!),
   *   continuous 3800K LED ceiling luminaires, & 6 twin overhead jet-fan boosters
   * - Submerged seabed concrete/rock-armour outer carapace (crown y = -3.72m, 2.27m clear navigable depth!)
   *   + 4 floating marine tunnel marker buoys on Selat UI!
   * Merged via NexusBatchOptimizer into the existing 2 corridor draw calls (+0 Draw Calls!).
   */
  static _buildUnderpassBawahLautUI() {
    const acc = SplineCorridorBuilder._createDualBufferAccumulator();

    const ux = TerrainDEMLoader.UNDERPASS_X; // -356.0
    const zW0 = TerrainDEMLoader.UNDERPASS_WEST_RAMP_START_Z; // 272.0
    const zWP = TerrainDEMLoader.UNDERPASS_WEST_PORTAL_Z; // 326.0
    const zEP = TerrainDEMLoader.UNDERPASS_EAST_PORTAL_Z; // 624.0
    const zE1 = TerrainDEMLoader.UNDERPASS_EAST_RAMP_END_Z; // 670.0
    const floorY = TerrainDEMLoader.UNDERPASS_TUNNEL_FLOOR_Y; // -8.20
    const clearH = TerrainDEMLoader.UNDERPASS_CLEAR_HEIGHT_M; // 4.15 (crown = -4.05)
    const roadHW = TerrainDEMLoader.UNDERPASS_CARRIAGEWAY_HALF_W; // 3.60
    const boreHW = TerrainDEMLoader.UNDERPASS_BORE_HALF_W; // 4.55

    const asphaltTunnel = [0.17, 0.18, 0.20];
    const uiYellow = [0.96, 0.77, 0.12];
    const uiGoldPortal = [0.95, 0.71, 0.06];
    const uiGoldDeep = [0.80, 0.56, 0.05];
    const whiteLine = [0.95, 0.96, 0.98];
    const concreteWall = [0.88, 0.87, 0.84];
    const concreteVault = [0.82, 0.82, 0.80];
    const concreteWalk = [0.76, 0.75, 0.72];
    const steelRibCol = [0.22, 0.25, 0.29];
    const subseaAcrylicCol = [0.08, 0.46, 0.72]; // Triggers E2C animated Selat UI underwater ocean caustics!
    const ledLuminaireCol = [0.99, 0.95, 0.84]; // Triggers E2C 24/7 3800K LED ceiling luminaires!
    const terracottaRoof = [0.68, 0.27, 0.16];
    const handrailSteel = [0.72, 0.75, 0.79];
    const seabedArmourRock = [0.20, 0.23, 0.24];
    const seabedConcreteTube = [0.46, 0.49, 0.50];

    const sampleSideRimNaturalY = (sx, pz) => {
      // Sample right at the outer face of the 0.70m retaining parapet (+-4.90m..5.30m) where the axis-snapped DEM meets the wall flush!
      const hInner = TerrainDEMLoader.sampleGroundHeight(ux + sx * 4.90, pz);
      const hOuter = TerrainDEMLoader.sampleGroundHeight(ux + sx * 5.30, pz);
      return Math.max(0.14, hInner, hOuter);
    };

    const evalParapetCrownY = (sx, pz, py, zTopEnd) => {
      const wY = py + 0.18;
      const rimY = sampleSideRimNaturalY(sx, pz);
      const u = Math.max(0.0, Math.min(1.0, Math.abs(pz - zTopEnd) / 10.0));
      const entT = u * u * u * (u * (u * 6.0 - 15.0) + 10.0);
      const targetEntY = Math.max(wY + 0.36, rimY + 0.22);
      const targetDeepY = Math.max(wY + 0.68, rimY + 0.32);
      return targetEntY + (targetDeepY - targetEntY) * entT;
    };

    // =========================================================================
    // 1. WEST (Pulau U) & EAST (Pulau I) OPEN-CUT RETAINED RAMPS
    // =========================================================================
    const buildOpenCutRamp = (zStart, zEnd, rampSegs) => {
      const zTopEnd = zStart === zW0 ? zW0 : zE1;
      const topOutSignZ = zStart === zW0 ? -1.0 : 1.0;
      const yTopEnd = TerrainDEMLoader.sampleUnderpassDeckY(ux, zTopEnd);

      // Solid concrete front entrance cheek-wall & walkway step end-caps at surface transition (z = 272m / z = 670m)
      for (const sx of [-1.0, 1.0]) {
        const xCurb = ux + sx * roadHW;
        const xWall = ux + sx * boreHW;
        const xParapetOut = ux + sx * 5.25;
        const wYTop = yTopEnd + 0.18;
        const rimYTop = sampleSideRimNaturalY(sx, zTopEnd);
        const parYTop = evalParapetCrownY(sx, zTopEnd, yTopEnd, zTopEnd);
        const footYTop = Math.min(yTopEnd - 1.4, rimYTop - 1.8);

        acc.quadV(
          xWall,
          footYTop,
          zTopEnd,
          xParapetOut,
          footYTop,
          zTopEnd,
          xParapetOut,
          parYTop,
          zTopEnd,
          xWall,
          parYTop,
          zTopEnd,
          concreteWall,
          0,
          0,
          topOutSignZ
        );
        acc.quadV(
          xCurb,
          yTopEnd - 0.35,
          zTopEnd,
          xWall,
          yTopEnd - 0.35,
          zTopEnd,
          xWall,
          wYTop,
          zTopEnd,
          xCurb,
          wYTop,
          zTopEnd,
          concreteWalk,
          0,
          0,
          topOutSignZ
        );
      }

      for (let i = 0; i < rampSegs; i++) {
        const za = zStart + ((zEnd - zStart) * i) / rampSegs;
        const zb = zStart + ((zEnd - zStart) * (i + 1)) / rampSegs;
        const ya = TerrainDEMLoader.sampleUnderpassDeckY(ux, za);
        const yb = TerrainDEMLoader.sampleUnderpassDeckY(ux, zb);

        // A. Dual-lane 7.20m asphalt ramp roadway + sub-slab concrete bed
        acc.quadH(
          ux - roadHW,
          ya,
          za,
          ux - roadHW,
          yb,
          zb,
          ux + roadHW,
          yb,
          zb,
          ux + roadHW,
          ya,
          za,
          asphaltTunnel,
          0,
          1,
          0
        );

        // Dashed UI-yellow centerline + continuous white lane-edge stripes
        if (i % 2 === 0) {
          acc.quadH(
            ux - 0.09,
            ya + 0.015,
            za,
            ux - 0.09,
            yb + 0.015,
            zb,
            ux + 0.09,
            yb + 0.015,
            zb,
            ux + 0.09,
            ya + 0.015,
            za,
            uiYellow,
            0,
            1,
            0
          );
        }
        for (const sx of [-1.0, 1.0]) {
          const e0 = sx * (roadHW - 0.28);
          const e1 = sx * (roadHW - 0.12);
          acc.quadH(
            ux + e0,
            ya + 0.014,
            za,
            ux + e0,
            yb + 0.014,
            zb,
            ux + e1,
            yb + 0.014,
            zb,
            ux + e1,
            ya + 0.014,
            za,
            whiteLine,
            0,
            1,
            0
          );
        }

        // B. Raised 0.95m Pedestrian Walkways, Zebra Safety Curbs, Buttressed Retaining Walls & Per-Side Rim Sealing Collar
        const uvZa = [za * 0.15, 0.5];
        const uvZb = [zb * 0.15, 0.5];
        for (const sx of [-1.0, 1.0]) {
          const rimYa = sampleSideRimNaturalY(sx, za);
          const rimYb = sampleSideRimNaturalY(sx, zb);
          const xCurb = ux + sx * roadHW;
          const xWall = ux + sx * boreHW; // +-4.55m
          const xParapetOut = ux + sx * 5.25; // Sleek 0.70m concrete retaining parapet embedding +-4.88m DEM cutout

          const wYa = ya + 0.18;
          const wYb = yb + 0.18;
          const parYa = evalParapetCrownY(sx, za, ya, zTopEnd);
          const parYb = evalParapetCrownY(sx, zb, yb, zTopEnd);
          const outerFootYa = Math.min(ya - 1.60, rimYa - 2.40);
          const outerFootYb = Math.min(yb - 1.60, rimYb - 2.40);

          // Zebra safety curb riser facing carriageway (-sx, 0, 0)
          acc.quadV(
            xCurb,
            ya,
            za,
            xCurb,
            yb,
            zb,
            xCurb,
            wYb,
            zb,
            xCurb,
            wYa,
            za,
            [0.96, 0.95, 0.92],
            -sx,
            0,
            0,
            uvZa,
            uvZb,
            uvZb,
            uvZa
          );

          // Pedestrian walkway deck (+Y)
          acc.quadV(xCurb, wYa, za, xCurb, wYb, zb, xWall, wYb, zb, xWall, wYa, za, concreteWalk, 0, 1, 0);

          // Pedestrian stainless handrail along walkway edge
          acc.tubeH(
            xCurb + sx * 0.12,
            wYa + 0.92,
            za,
            xCurb + sx * 0.12,
            wYb + 0.92,
            zb,
            0.032,
            0.032,
            6,
            handrailSteel
          );
          if (i % 3 === 0) {
            acc.tubeH(
              xCurb + sx * 0.12,
              wYa,
              za,
              xCurb + sx * 0.12,
              wYa + 0.92,
              za,
              0.026,
              0.026,
              6,
              handrailSteel
            );
          }

          // Vertical inner reinforced-concrete retaining wall rising from walkway up to parapet crown (-sx, 0, 0)
          acc.quadV(
            xWall,
            ya - 0.80,
            za,
            xWall,
            yb - 0.80,
            zb,
            xWall,
            parYb,
            zb,
            xWall,
            parYa,
            za,
            concreteWall,
            -sx,
            0,
            0
          );

          // Parapet top coping (+Y) & deep outer vertical retaining face (+sx, 0, 0) seated cleanly in hillside
          acc.quadV(
            xWall,
            parYa,
            za,
            xWall,
            parYb,
            zb,
            xParapetOut,
            parYb,
            zb,
            xParapetOut,
            parYa,
            za,
            [0.94, 0.93, 0.90],
            0,
            1,
            0
          );
          acc.quadV(
            xParapetOut,
            outerFootYa,
            za,
            xParapetOut,
            outerFootYb,
            zb,
            xParapetOut,
            parYb,
            zb,
            xParapetOut,
            parYa,
            za,
            concreteWall,
            sx,
            0,
            0
          );
          // Tubular steel top safety rail along ground-level parapet
          acc.tubeH(
            0.5 * (xWall + xParapetOut),
            parYa + 0.34,
            za,
            0.5 * (xWall + xParapetOut),
            parYb + 0.34,
            zb,
            0.036,
            0.036,
            6,
            handrailSteel
          );

          // Sculpted vertical concrete pilaster buttress ribs every 3 ramp segments
          if (i > 1 && i < rampSegs - 1 && i % 3 === 0) {
            const pilH = parYa - wYa;
            if (pilH > 0.8) {
              const pil = new THREE.BoxGeometry(0.24, pilH, 0.44);
              pil.translate(xWall - sx * 0.10, wYa + pilH * 0.5, za);
              acc.appendV(pil, [0.92, 0.91, 0.88]);
            }
          }
        }
      }
    };

    buildOpenCutRamp(zW0, zWP, 20);
    buildOpenCutRamp(zEP, zE1, 18);

    // =========================================================================
    // 2. WEST (z = 326m) & EAST (z = 624m) NUSANTARA-UI TUNNEL PORTALS & VENTILATION TOWERS
    // =========================================================================
    const buildPortalGatewayAndVentTowers = (zPortal, outSignZ) => {
      // outSignZ = -1 for West Portal (faces -Z into West Ramp), +1 for East Portal (faces +Z into East Ramp)
      const zCenter = zPortal - outSignZ * 3.55; // Spans 7.3m back behind the portal plane (completely sealing DEM cutout!)
      const behindGroundY = Math.max(
        0.55,
        TerrainDEMLoader.sampleGroundHeight(ux, zPortal - outSignZ * 6.8)
      );
      const zTopRef = zPortal === zWP ? zW0 : zE1;
      const rampEndParY = Math.max(
        evalParapetCrownY(-1.0, zPortal, floorY, zTopRef),
        evalParapetCrownY(1.0, zPortal, floorY, zTopRef)
      );
      const deckTopY = Math.max(behindGroundY + 0.42, rampEndParY);
      const archCrownY = floorY + clearH; // -4.05m

      // Upper portal headwall concrete lintel beam spanning above tunnel arch crown (y = -4.05m .. deckTopY)
      const lintelH = deckTopY - archCrownY + 0.15;
      const lintel = new THREE.BoxGeometry(15.1, lintelH, 7.3);
      lintel.translate(ux, archCrownY + lintelH * 0.5 - 0.05, zCenter);
      acc.appendV(lintel, concreteWall);

      // Left & Right full-height portal abutment piers flanking the 9.10m tunnel mouth (y = -9.4m .. deckTopY)
      const pierH = deckTopY - (floorY - 1.2);
      for (const sx of [-1.0, 1.0]) {
        const pier = new THREE.BoxGeometry(3.0, pierH, 7.3);
        pier.translate(ux + sx * (boreHW + 1.50), floorY - 1.2 + pierH * 0.5, zCenter);
        acc.appendV(pier, concreteWall);
      }

      // Monumental UI Golden-Yellow Sculpted Nusantara Portal Arch Collar & Tri-Tier Makara Crest
      const zFace = zPortal + outSignZ * 0.22;
      for (const sx of [-1.0, 1.0]) {
        const jambH = clearH + 1.85;
        const jamb = new THREE.BoxGeometry(0.76, jambH, 0.68);
        jamb.translate(ux + sx * (boreHW + 0.34), floorY + jambH * 0.5, zFace);
        acc.appendH(jamb, uiGoldPortal);
      }
      const fasciaBeam = new THREE.BoxGeometry(boreHW * 2.0 + 2.1, 1.25, 0.74);
      fasciaBeam.translate(ux, archCrownY + 1.15, zFace);
      acc.appendH(fasciaBeam, uiGoldPortal);

      // Stepped Nusantara Tri-Tier Crown Crest + Warm Photocell Uplight Bar above Portal Mouth
      for (let t = 0; t < 3; t++) {
        const tier = new THREE.BoxGeometry(7.6 - t * 2.1, 0.38, 0.88 - t * 0.12);
        tier.translate(ux, deckTopY + 0.22 + t * 0.38, zPortal - outSignZ * 0.65);
        acc.appendH(tier, t % 2 === 0 ? uiGoldPortal : uiGoldDeep);
      }
      const portalLightBar = new THREE.BoxGeometry(8.4, 0.18, 0.26);
      portalLightBar.translate(ux, archCrownY + 0.42, zFace + outSignZ * 0.24);
      acc.appendH(portalLightBar, [0.98, 0.92, 0.76]);

      // Twin Subsea Tunnel Ventilation & Control Towers ("Menara Ventilasi Bawah Laut UI") on Portal Terrace
      for (const sx of [-1.0, 1.0]) {
        const tx = ux + sx * 5.65;
        const tz = zPortal - outSignZ * 2.55;
        const towerH = 4.35;

        const shaft = new THREE.BoxGeometry(2.55, towerH, 2.55);
        shaft.translate(tx, deckTopY + towerH * 0.5, tz);
        acc.appendV(shaft, [0.93, 0.92, 0.88]);

        // Dark acoustic ventilation louver grilles on 4 faces
        const louver = new THREE.BoxGeometry(2.64, 1.55, 1.82);
        louver.translate(tx, deckTopY + towerH - 1.15, tz);
        acc.appendH(louver, steelRibCol);
        const louver2 = new THREE.BoxGeometry(1.82, 1.55, 2.64);
        louver2.translate(tx, deckTopY + towerH - 1.15, tz);
        acc.appendH(louver2, steelRibCol);

        // UI Golden-Yellow cornice collar + Pyramidal Javanese Terracotta Tajug Roof & Bronze Finial
        const cornice = new THREE.BoxGeometry(2.92, 0.22, 2.92);
        cornice.translate(tx, deckTopY + towerH + 0.10, tz);
        acc.appendH(cornice, uiGoldPortal);

        const roof = new THREE.ConeGeometry(2.38, 1.55, 4, 1);
        roof.rotateY(Math.PI * 0.25);
        roof.translate(tx, deckTopY + towerH + 0.96, tz);
        acc.appendH(roof, terracottaRoof);

        const finial = new THREE.ConeGeometry(0.16, 0.62, 8, 1);
        finial.translate(tx, deckTopY + towerH + 1.98, tz);
        acc.appendH(finial, uiGoldPortal);
      }
    };

    buildPortalGatewayAndVentTowers(zWP, -1.0);
    buildPortalGatewayAndVentTowers(zEP, 1.0);

    // =========================================================================
    // 3. 298.0m ENCLOSED IMMERSED-TUBE SUBSEA TUNNEL BORE (z = 326 -> 624m, y = -8.20m)
    // =========================================================================
    const boreLen = zEP - zWP; // 298.0m
    const boreSegs = 60; // ~4.97m per structural bay

    // 7-point Arched Vault Cross-Section Profile (from South springline x = -4.55 -> Crown -> North springline x = +4.55)
    const vaultNodes = [];
    const vaultSteps = 8;
    for (let k = 0; k <= vaultSteps; k++) {
      const th = Math.PI * (1.0 - k / vaultSteps); // PI -> 0
      const vx = ux - Math.cos(th) * boreHW; // ux - 4.55 -> ux + 4.55
      const vy = floorY + 2.20 + Math.sin(th) * (clearH - 2.20); // -6.00 -> -4.05 -> -6.00
      vaultNodes.push({ x: vx, y: vy, inNx: Math.cos(th), inNy: -Math.sin(th) });
    }

    for (let i = 0; i < boreSegs; i++) {
      const za = zWP + (boreLen * i) / boreSegs;
      const zb = zWP + (boreLen * (i + 1)) / boreSegs;
      const zMid = 0.5 * (za + zb);

      // 8 Panoramic Subsea Observation Window Bays under Selat UI (z in [366..584], every 3rd bay!)
      const isSubseaWindowBay = zMid > 364.0 && zMid < 588.0 && i % 6 >= 2 && i % 6 <= 3;

      // A. Dual-Lane 7.20m Subsea Asphalt Roadway + UI-Yellow Dashed Centerline + White Edge Stripes
      acc.quadH(
        ux - roadHW,
        floorY,
        za,
        ux - roadHW,
        floorY,
        zb,
        ux + roadHW,
        floorY,
        zb,
        ux + roadHW,
        floorY,
        za,
        asphaltTunnel,
        0,
        1,
        0
      );

      if (i % 2 === 0) {
        acc.quadH(
          ux - 0.09,
          floorY + 0.015,
          za,
          ux - 0.09,
          floorY + 0.015,
          zb,
          ux + 0.09,
          floorY + 0.015,
          zb,
          ux + 0.09,
          floorY + 0.015,
          za,
          uiYellow,
          0,
          1,
          0
        );
      }
      for (const sx of [-1.0, 1.0]) {
        const e0 = sx * (roadHW - 0.28);
        const e1 = sx * (roadHW - 0.12);
        acc.quadH(
          ux + e0,
          floorY + 0.014,
          za,
          ux + e0,
          floorY + 0.014,
          zb,
          ux + e1,
          floorY + 0.014,
          zb,
          ux + e1,
          floorY + 0.014,
          za,
          whiteLine,
          0,
          1,
          0
        );
      }

      // B. Twin Raised 0.95m Walkways, Handrails, & Lower Tunnel Side Walls (with Subsea Acrylic Windows!)
      const wY = floorY + 0.18;
      const springY = floorY + 2.20; // -6.00m
      const uvA = [za * 0.15, 0.5];
      const uvB = [zb * 0.15, 0.5];

      for (const sx of [-1.0, 1.0]) {
        const xCurb = ux + sx * roadHW;
        const xWall = ux + sx * boreHW;

        // Zebra safety curb & raised pedestrian walkway
        acc.quadV(
          xCurb,
          floorY,
          za,
          xCurb,
          floorY,
          zb,
          xCurb,
          wY,
          zb,
          xCurb,
          wY,
          za,
          [0.96, 0.95, 0.92],
          -sx,
          0,
          0,
          uvA,
          uvB,
          uvB,
          uvA
        );
        acc.quadV(xCurb, wY, za, xCurb, wY, zb, xWall, wY, zb, xWall, wY, za, concreteWalk, 0, 1, 0);

        // Stainless pedestrian safety rail
        acc.tubeH(xCurb + sx * 0.12, wY + 0.92, za, xCurb + sx * 0.12, wY + 0.92, zb, 0.030, 0.030, 6, handrailSteel);
        if (i % 2 === 0) {
          acc.tubeH(xCurb + sx * 0.12, wY, za, xCurb + sx * 0.12, wY + 0.92, za, 0.025, 0.025, 6, handrailSteel);
        }

        if (isSubseaWindowBay) {
          // Lower concrete dado wall (wY .. wY + 0.55m)
          acc.quadV(xWall, wY, za, xWall, wY, zb, xWall, wY + 0.55, zb, xWall, wY + 0.55, za, concreteWall, -sx, 0, 0);
          // Panoramic Subsea Marine-Acrylic Observation Window (y = -7.47m .. -5.98m) glowing with Selat UI ocean caustics!
          acc.quadH(
            xWall - sx * 0.03,
            wY + 0.55,
            za + 0.22,
            xWall - sx * 0.03,
            wY + 0.55,
            zb - 0.22,
            xWall - sx * 0.03,
            springY + 0.02,
            zb - 0.22,
            xWall - sx * 0.03,
            springY + 0.02,
            za + 0.22,
            subseaAcrylicCol,
            -sx,
            0,
            0
          );
          // Heavy bolted dark-steel subsea window mullions & sill/header frames
          const sill = new THREE.BoxGeometry(0.18, 0.12, zb - za);
          sill.translate(xWall - sx * 0.06, wY + 0.55, zMid);
          acc.appendH(sill, steelRibCol);
          const hdr = new THREE.BoxGeometry(0.18, 0.14, zb - za);
          hdr.translate(xWall - sx * 0.06, springY, zMid);
          acc.appendH(hdr, steelRibCol);
        } else {
          // Solid precast acoustic concrete tunnel wall panel (with UI-yellow longitudinal accent stripe!)
          acc.quadV(xWall, wY, za, xWall, wY, zb, xWall, springY, zb, xWall, springY, za, concreteWall, -sx, 0, 0);
          acc.quadH(
            xWall - sx * 0.018,
            wY + 1.05,
            za,
            xWall - sx * 0.018,
            wY + 1.05,
            zb,
            xWall - sx * 0.018,
            wY + 1.23,
            zb,
            xWall - sx * 0.018,
            wY + 1.23,
            za,
            uiYellow,
            -sx,
            0,
            0
          );
        }
      }

      // C. 8-Facet Arched Concrete Tunnel Ceiling Vault (inward-facing normals!)
      for (let k = 0; k < vaultSteps; k++) {
        const n0 = vaultNodes[k];
        const n1 = vaultNodes[k + 1];
        acc.quadV(
          n0.x,
          n0.y,
          za,
          n0.x,
          n0.y,
          zb,
          n1.x,
          n1.y,
          zb,
          n1.x,
          n1.y,
          za,
          concreteVault,
          0.5 * (n0.inNx + n1.inNx),
          0.5 * (n0.inNy + n1.inNy),
          0.0
        );
      }

      // D. Continuous Twin 3800K LED Ceiling Luminaire Strips (x = ux +- 1.65m, y = -4.22m)
      for (const sx of [-1.0, 1.0]) {
        const lx = ux + sx * 1.65;
        acc.quadH(
          lx - 0.16,
          floorY + clearH - 0.17,
          za + 0.18,
          lx - 0.16,
          floorY + clearH - 0.17,
          zb - 0.18,
          lx + 0.16,
          floorY + clearH - 0.17,
          zb - 0.18,
          lx + 0.16,
          floorY + clearH - 0.17,
          za + 0.18,
          ledLuminaireCol,
          0,
          -1,
          0
        );
      }

      // E. Structural Dark-Steel Portal Arch Ribs every 2 bays + 6 Overhead Twin Jet-Fan Ventilation Boosters
      if (i > 0 && i % 2 === 0) {
        for (const sx of [-1.0, 1.0]) {
          const post = new THREE.BoxGeometry(0.22, 2.05, 0.26);
          post.translate(ux + sx * (boreHW - 0.09), wY + 1.025, za);
          acc.appendH(post, steelRibCol);
        }
        for (let k = 0; k < vaultSteps; k++) {
          const n0 = vaultNodes[k];
          const n1 = vaultNodes[k + 1];
          acc.tubeH(
            n0.x - n0.inNx * 0.02,
            n0.y + n0.inNy * 0.08,
            za,
            n1.x - n1.inNx * 0.02,
            n1.y + n1.inNy * 0.08,
            za,
            0.12,
            0.12,
            6,
            steelRibCol
          );
        }
      }

      if (i > 4 && i < boreSegs - 4 && i % 10 === 5) {
        for (const sx of [-0.75, 0.75]) {
          const fan = new THREE.CylinderGeometry(0.32, 0.32, 2.25, 10, 1);
          fan.rotateX(Math.PI * 0.5);
          fan.translate(ux + sx, floorY + clearH - 0.42, zMid);
          acc.appendH(fan, steelRibCol);
        }
      }
    }

    // =========================================================================
    // 4. SUBMERGED SEABED OUTER IMMERSED-TUBE SHELL (y = -3.72m) & 4 SURFACE MARKER BUOYS
    //    Visible beneath clear Selat UI ocean water while leaving 2.27m navigable draft above!
    // =========================================================================
    const subZ0 = 357.0;
    const subZ1 = 602.5;
    const subLen = subZ1 - subZ0;
    const subMidZ = 0.5 * (subZ0 + subZ1);
    const outerCrownY = TerrainDEMLoader.UNDERPASS_OUTER_CROWN_Y; // -3.72m

    // Dorsal concrete immersed-tube carapace & chamfered side haunches (+Y / +-X normals facing upward into ocean!)
    acc.quadV(ux - 3.6, outerCrownY, subZ0, ux - 3.6, outerCrownY, subZ1, ux + 3.6, outerCrownY, subZ1, ux + 3.6, outerCrownY, subZ0, seabedConcreteTube, 0, 1, 0);
    for (const sx of [-1.0, 1.0]) {
      acc.quadV(
        ux + sx * 3.6,
        outerCrownY,
        subZ0,
        ux + sx * 3.6,
        outerCrownY,
        subZ1,
        ux + sx * 6.2,
        -5.45,
        subZ1,
        ux + sx * 6.2,
        -5.45,
        subZ0,
        seabedConcreteTube,
        sx * 0.55,
        0.83,
        0
      );
      acc.quadH(
        ux + sx * 6.2,
        -5.45,
        subZ0,
        ux + sx * 6.2,
        -5.45,
        subZ1,
        ux + sx * 9.8,
        -9.25,
        subZ1,
        ux + sx * 9.8,
        -9.25,
        subZ0,
        seabedArmourRock,
        sx * 0.72,
        0.69,
        0
      );
    }
    // 14 Transverse Precast Ballast Collar Rings across the submerged tube
    for (let r = 0; r < 14; r++) {
      const rz = subZ0 + ((r + 0.5) / 14) * subLen;
      const ring = new THREE.BoxGeometry(8.2, 0.26, 1.35);
      ring.translate(ux, outerCrownY + 0.08, rz);
      acc.appendH(ring, uiGoldDeep);
    }
    // 4 Floating Marine Subsea-Tunnel Corridor Marker Buoys on Selat UI surface (y = -1.45m)
    const waterY = TerrainDEMLoader.WATER_LEVEL_Y;
    for (const bz of [396.0, 562.0]) {
      for (const sx of [-10.5, 10.5]) {
        const bx = ux + sx;
        const floatDrum = new THREE.CylinderGeometry(0.62, 0.74, 0.78, 12, 1);
        floatDrum.translate(bx, waterY + 0.16, bz);
        acc.appendH(floatDrum, [0.96, 0.52, 0.08]);

        const mast = new THREE.CylinderGeometry(0.06, 0.09, 1.35, 8, 1);
        mast.translate(bx, waterY + 1.12, bz);
        acc.appendH(mast, uiGoldPortal);

        const topMark = new THREE.SphereGeometry(0.18, 10, 8);
        topMark.translate(bx, waterY + 1.88, bz);
        acc.appendH(topMark, sx > 0 ? [0.08, 0.94, 0.42] : [0.96, 0.10, 0.08]);
      }
    }

    return acc.finalize();
  }

  /**
   * Builds the Twin Northern Ferry Crossing Terminals ("Dermaga Penyeberangan Utara antara Pulau U dan Pulau I")
   * facing each other across northern Selat UI:
   * - Dermaga Pulau U (x = -48.0m, protruding +Z eastward from z = 446.5m forecourt out to z = 491.5m T-head berth)
   * - Dermaga Pulau I (x = -98.0m, protruding -Z westward from z = 624.5m forecourt out to z = 582.0m T-head berth)
   * Complete with Granite Forecourts, Twin-Tier Javanese Terracotta Tajug Passenger Terminal Pavilions,
   * 18-Pile Marine Concrete Trestle Piers, 28m T-Head Catamaran Berths, Lower Boarding Stages & Gangways,
   * Vulcanized Rubber D-Fenders, UI-Yellow Twin-Horn Mooring Bollards, SOLAS Lifebuoys, & Red/Green Harbour Beacons!
   * Merged via NexusBatchOptimizer into the existing 2 corridor draw calls (+0 Draw Calls!).
   */
  static _buildDermagaPenyeberanganUtaraUI() {
    const acc = SplineCorridorBuilder._createDualBufferAccumulator();

    const forecourtY = TerrainDEMLoader.DERMAGA_FORECOURT_Y; // 0.48
    const pierDeckY = TerrainDEMLoader.DERMAGA_DECK_Y; // 0.72
    const trestleHW = TerrainDEMLoader.DERMAGA_TRESTLE_HALF_W; // 4.20
    const tHeadHW = TerrainDEMLoader.DERMAGA_THEAD_HALF_LEN_X; // 14.0
    const tHeadDepth = TerrainDEMLoader.DERMAGA_THEAD_DEPTH_Z; // 7.60
    const waterY = TerrainDEMLoader.WATER_LEVEL_Y; // -1.45

    const graniteDeck = [0.50, 0.49, 0.46];
    const pierConcrete = [0.87, 0.86, 0.82];
    const pierDeckGrey = [0.36, 0.38, 0.40];
    const teakBoardwalk = [0.48, 0.29, 0.15];
    const terracottaRoof = [0.69, 0.27, 0.16];
    const uiGold = [0.95, 0.72, 0.08];
    const marinePileConcrete = [0.40, 0.42, 0.41];
    const fenderBlack = [0.10, 0.11, 0.13];
    const handrailSteel = [0.74, 0.77, 0.81];
    const lifebuoyOrange = [0.96, 0.36, 0.08];

    const buildSingleFerryTerminal = (cx, zForeInland, zShore, zHead, dirZ) => {
      // dirZ = +1 for Dermaga Pulau U (+Z into Selat UI), -1 for Dermaga Pulau I (-Z into Selat UI)

      // 1. Coastal Granite Forecourt Plaza & Drop-Off Apron (28m x 16.5m, watertight down to y = -2.4m)
      const zForeEdge = zShore - dirZ * 2.5;
      const foreLenZ = Math.abs(zForeEdge - zForeInland);
      const foreMidZ = 0.5 * (zForeEdge + zForeInland);
      const foreTopY = forecourtY + 0.035;

      const forePlinth = new THREE.BoxGeometry(28.4, foreTopY + 2.45, foreLenZ + 0.4);
      forePlinth.translate(cx, foreTopY - (foreTopY + 2.45) * 0.5 - 0.02, foreMidZ);
      acc.appendV(forePlinth, pierConcrete);

      const foreSlab = new THREE.BoxGeometry(27.8, 0.055, foreLenZ);
      foreSlab.translate(cx, foreTopY - 0.02, foreMidZ);
      acc.appendH(foreSlab, graniteDeck);

      // Zebra safety drop-off curb along road side & 4 teak waiting benches
      const dropCurb = new THREE.BoxGeometry(27.8, 0.12, 0.42);
      const pArr = dropCurb.attributes.position.array;
      const uArr = dropCurb.attributes.uv.array;
      for (let vi = 0; vi < dropCurb.attributes.position.count; vi++) {
        uArr[vi * 2] = (pArr[vi * 3] + 14.0) * 0.38;
        uArr[vi * 2 + 1] = 0.5;
      }
      dropCurb.translate(cx, foreTopY + 0.04, zForeInland + dirZ * 0.25);
      acc.appendV(dropCurb, [0.96, 0.95, 0.92], null);

      for (const bxOff of [-8.8, -5.2, 5.2, 8.8]) {
        const bz = foreMidZ + dirZ * 1.4;
        for (const lx of [-0.65, 0.65]) {
          const leg = new THREE.BoxGeometry(0.12, 0.44, 0.50);
          leg.translate(cx + bxOff + lx, foreTopY + 0.22, bz);
          acc.appendV(leg, pierConcrete);
        }
        const seat = new THREE.BoxGeometry(1.58, 0.055, 0.50);
        seat.translate(cx + bxOff, foreTopY + 0.46, bz);
        acc.appendH(seat, teakBoardwalk);
        const back = new THREE.BoxGeometry(1.58, 0.32, 0.055);
        back.translate(cx + bxOff, foreTopY + 0.66, bz - dirZ * 0.22);
        acc.appendH(back, teakBoardwalk);
      }

      // 2. Nusantara Twin-Tier Terracotta Tajug Ferry Terminal Pavilion ("Terminal Penyeberangan UI", 12.6m x 8.4m)
      const pavZ = foreMidZ + dirZ * 1.2;
      const eaveY1 = foreTopY + 3.05;
      const colCoords = [
        [-5.4, -3.4],
        [-1.8, -3.4],
        [1.8, -3.4],
        [5.4, -3.4],
        [-5.4, 3.4],
        [-1.8, 3.4],
        [1.8, 3.4],
        [5.4, 3.4]
      ];
      for (const [ox, oz] of colCoords) {
        const plinth = new THREE.BoxGeometry(0.56, 0.42, 0.56);
        plinth.translate(cx + ox, foreTopY + 0.21, pavZ + oz);
        acc.appendV(plinth, [0.93, 0.92, 0.88]);

        const col = new THREE.CylinderGeometry(0.18, 0.21, 2.55, 8, 1);
        col.translate(cx + ox, foreTopY + 1.68, pavZ + oz);
        acc.appendV(col, [0.95, 0.94, 0.90]);
      }

      // Ring beam, UI Golden-Yellow fascia portal sign, & warm 3000K photocell ceiling luminaire strip
      const ringBeam = new THREE.BoxGeometry(11.6, 0.32, 7.5);
      ringBeam.translate(cx, eaveY1 - 0.06, pavZ);
      acc.appendV(ringBeam, [0.94, 0.93, 0.89]);

      const goldSign = new THREE.BoxGeometry(6.8, 0.44, 7.82);
      goldSign.translate(cx, eaveY1 + 0.04, pavZ);
      acc.appendH(goldSign, uiGold);

      const ceilLight = new THREE.BoxGeometry(8.6, 0.10, 0.42);
      ceilLight.translate(cx, eaveY1 - 0.22, pavZ);
      acc.appendH(ceilLight, [0.98, 0.92, 0.76]);

      // Lower & Upper Javanese Terracotta Hip/Tajug Roof Tiers + UI Gold Finial Spire
      const lowerRoof = new THREE.CylinderGeometry(3.4, 8.6, 1.35, 4, 1);
      lowerRoof.rotateY(Math.PI * 0.25);
      lowerRoof.scale(1.12, 1.0, 0.76);
      lowerRoof.translate(cx, eaveY1 + 0.72, pavZ);
      acc.appendH(lowerRoof, terracottaRoof);

      const lanternNeck = new THREE.BoxGeometry(4.4, 0.45, 3.2);
      lanternNeck.translate(cx, eaveY1 + 1.48, pavZ);
      acc.appendH(lanternNeck, uiGold);

      const upperRoof = new THREE.ConeGeometry(4.1, 1.38, 4, 1);
      upperRoof.rotateY(Math.PI * 0.25);
      upperRoof.scale(1.10, 1.0, 0.78);
      upperRoof.translate(cx, eaveY1 + 2.28, pavZ);
      acc.appendH(upperRoof, terracottaRoof);

      const spire = new THREE.ConeGeometry(0.18, 0.85, 8, 1);
      spire.translate(cx, eaveY1 + 3.32, pavZ);
      acc.appendH(spire, uiGold);

      // 3. Accessible Shore-to-Pier Bridge Ramp spanning cleanly OVER the intact 360° Stone Turap (foreTopY -> pierDeckY)
      const zRamp0 = zShore - dirZ * 2.5;
      const zRamp1 = zShore + dirZ * 2.0;
      const rampSteps = 8;
      for (let i = 0; i < rampSteps; i++) {
        const t0 = i / rampSteps;
        const t1 = (i + 1) / rampSteps;
        const za = zRamp0 + (zRamp1 - zRamp0) * t0;
        const zb = zRamp0 + (zRamp1 - zRamp0) * t1;
        const ya = foreTopY + (pierDeckY - foreTopY) * t0;
        const yb = foreTopY + (pierDeckY - foreTopY) * t1;
        const zLo = Math.min(za, zb);
        const zHi = Math.max(za, zb);
        const yLo = za < zb ? ya : yb;
        const yHi = za < zb ? yb : ya;

        acc.quadH(cx - 5.4, yLo, zLo, cx - 5.4, yHi, zHi, cx + 5.4, yHi, zHi, cx + 5.4, yLo, zLo, pierDeckGrey, 0, 1, 0);
        acc.quadH(
          cx - 2.1,
          yLo + 0.015,
          zLo,
          cx - 2.1,
          yHi + 0.015,
          zHi,
          cx + 2.1,
          yHi + 0.015,
          zHi,
          cx + 2.1,
          yLo + 0.015,
          zLo,
          teakBoardwalk,
          0,
          1,
          0
        );
      }
      const abutment = new THREE.BoxGeometry(11.4, pierDeckY + 1.8, 4.6);
      abutment.translate(cx, pierDeckY - (pierDeckY + 1.8) * 0.5 - 0.05, 0.5 * (zRamp0 + zRamp1));
      acc.appendV(abutment, pierConcrete);

      // 4. Pile-Supported Marine Concrete Trestle Pier (width 8.4m) + 28.0m x 7.6m T-Head Catamaran Berth (y = +0.72m)
      const zTrestle0 = zRamp1;
      const zTrestle1 = zHead - dirZ * tHeadDepth;
      const trestleLen = Math.abs(zTrestle1 - zTrestle0);
      const trestleMidZ = 0.5 * (zTrestle0 + zTrestle1);

      const trestleSlab = new THREE.BoxGeometry(trestleHW * 2.0, 0.42, trestleLen + 0.2);
      trestleSlab.translate(cx, pierDeckY - 0.21, trestleMidZ);
      acc.appendV(trestleSlab, pierConcrete);

      const trestleTop = new THREE.BoxGeometry(trestleHW * 2.0 - 0.24, 0.04, trestleLen);
      trestleTop.translate(cx, pierDeckY - 0.01, trestleMidZ);
      acc.appendH(trestleTop, pierDeckGrey);

      const trestleTeak = new THREE.BoxGeometry(3.4, 0.05, trestleLen + 0.2);
      trestleTeak.translate(cx, pierDeckY + 0.005, trestleMidZ);
      acc.appendH(trestleTeak, teakBoardwalk);

      // 28.0m x 7.6m T-Head Ferry Berth Deck
      const tHeadMidZ = zHead - dirZ * (tHeadDepth * 0.5);
      const tHeadSlab = new THREE.BoxGeometry(tHeadHW * 2.0, 0.48, tHeadDepth);
      tHeadSlab.translate(cx, pierDeckY - 0.24, tHeadMidZ);
      acc.appendV(tHeadSlab, pierConcrete);

      const tHeadTop = new THREE.BoxGeometry(tHeadHW * 2.0 - 0.28, 0.045, tHeadDepth - 0.24);
      tHeadTop.translate(cx, pierDeckY - 0.01, tHeadMidZ);
      acc.appendH(tHeadTop, pierDeckGrey);

      // UI Golden-Yellow Marine Safety Edge Band around T-Head Berth Perimeter
      const safetyBand = new THREE.BoxGeometry(tHeadHW * 2.0 - 0.36, 0.055, 0.36);
      safetyBand.translate(cx, pierDeckY + 0.005, zHead - dirZ * 0.32);
      acc.appendH(safetyBand, uiGold);

      // 5. 18 Driven Cylindrical Marine Concrete-Steel Piles (y = -6.50m .. pierDeckY - 0.35m) + Pile-Cap Crossbeams
      const pileRowsTrestle = [0.22, 0.55, 0.88];
      for (const f of pileRowsTrestle) {
        const pz = zTrestle0 + (zTrestle1 - zTrestle0) * f;
        const capBeam = new THREE.BoxGeometry(trestleHW * 2.0 + 0.35, 0.38, 0.62);
        capBeam.translate(cx, pierDeckY - 0.56, pz);
        acc.appendV(capBeam, pierConcrete);

        for (const sx of [-3.3, 3.3]) {
          const pileH = pierDeckY - 0.50 - -6.50;
          const pile = new THREE.CylinderGeometry(0.32, 0.35, pileH, 10, 1);
          pile.translate(cx + sx, -6.50 + pileH * 0.5, pz);
          acc.appendH(pile, marinePileConcrete);
        }
      }

      for (const fZ of [-0.28, 0.28]) {
        const pz = tHeadMidZ + dirZ * (tHeadDepth * fZ);
        const capBeam = new THREE.BoxGeometry(tHeadHW * 2.0 + 0.35, 0.42, 0.68);
        capBeam.translate(cx, pierDeckY - 0.62, pz);
        acc.appendV(capBeam, pierConcrete);

        for (const sx of [-12.2, -7.4, -2.6, 2.6, 7.4, 12.2]) {
          const pileH = pierDeckY - 0.55 - -6.60;
          const pile = new THREE.CylinderGeometry(0.34, 0.37, pileH, 10, 1);
          pile.translate(cx + sx, -6.60 + pileH * 0.5, pz);
          acc.appendH(pile, marinePileConcrete);
        }
      }

      // 6. Lower Passenger Boarding Landing Stage (y = -0.22m, matching Catamaran Ferry deck!) & Hinged Aluminum Gangway
      const stageZ = zHead + dirZ * 1.45;
      const stageY = -0.22;
      const stageBox = new THREE.BoxGeometry(6.8, 0.34, 2.75);
      stageBox.translate(cx, stageY - 0.17, stageZ);
      acc.appendH(stageBox, pierDeckGrey);

      const stageTeak = new THREE.BoxGeometry(6.2, 0.05, 2.35);
      stageTeak.translate(cx, stageY + 0.01, stageZ);
      acc.appendH(stageTeak, teakBoardwalk);

      for (const sx of [-2.8, 2.8]) {
        const sPile = new THREE.CylinderGeometry(0.24, 0.26, 6.2, 8, 1);
        sPile.translate(cx + sx, -3.45, stageZ + dirZ * 0.6);
        acc.appendH(sPile, marinePileConcrete);
      }

      // Sloped aluminum boarding gangway brow connecting T-Head Deck (+0.72m) down to Boarding Stage (-0.22m)
      const gwZ0 = zHead - dirZ * 0.35;
      const gwZ1 = stageZ - dirZ * 0.15;
      acc.quadH(
        cx - 1.35,
        Math.min(pierDeckY, stageY) + (gwZ0 < gwZ1 ? pierDeckY - Math.min(pierDeckY, stageY) : stageY - Math.min(pierDeckY, stageY)),
        Math.min(gwZ0, gwZ1),
        cx - 1.35,
        gwZ0 < gwZ1 ? stageY : pierDeckY,
        Math.max(gwZ0, gwZ1),
        cx + 1.35,
        gwZ0 < gwZ1 ? stageY : pierDeckY,
        Math.max(gwZ0, gwZ1),
        cx + 1.35,
        gwZ0 < gwZ1 ? pierDeckY : stageY,
        Math.min(gwZ0, gwZ1),
        handrailSteel,
        0,
        1,
        0
      );
      for (const sx of [-1.35, 1.35]) {
        acc.tubeH(cx + sx, pierDeckY + 0.88, gwZ0, cx + sx, stageY + 0.88, gwZ1, 0.032, 0.032, 6, uiGold);
      }

      // 7. Marine Perimeter Stainless Safety Railings (leaving central 3.4m gangway gate open!)
      for (const sx of [-1.0, 1.0]) {
        const rx = cx + sx * (trestleHW - 0.18);
        acc.tubeH(rx, pierDeckY + 0.96, zRamp0, rx, pierDeckY + 0.96, zTrestle1, 0.035, 0.035, 6, handrailSteel);
        acc.tubeH(rx, pierDeckY + 0.50, zRamp0, rx, pierDeckY + 0.50, zTrestle1, 0.025, 0.025, 6, handrailSteel);
        for (let p = 0; p <= 5; p++) {
          const pz = zRamp0 + ((zTrestle1 - zRamp0) * p) / 5;
          acc.tubeH(rx, pierDeckY, pz, rx, pierDeckY + 0.96, pz, 0.030, 0.030, 6, handrailSteel);
        }

        // T-head outer wing railings (leaving |x - cx| < 1.8m open for boarding gangway!)
        const zOuterRail = zHead - dirZ * 0.22;
        acc.tubeH(
          cx + sx * 1.85,
          pierDeckY + 0.96,
          zOuterRail,
          cx + sx * (tHeadHW - 0.22),
          pierDeckY + 0.96,
          zOuterRail,
          0.035,
          0.035,
          6,
          handrailSteel
        );
        acc.tubeH(
          cx + sx * (tHeadHW - 0.22),
          pierDeckY + 0.96,
          zTrestle1,
          cx + sx * (tHeadHW - 0.22),
          pierDeckY + 0.96,
          zOuterRail,
          0.035,
          0.035,
          6,
          handrailSteel
        );
      }

      // 8. Black Vulcanized-Rubber Marine D-Fenders, UI-Yellow Twin-Horn Mooring Bollards, SOLAS Lifebuoys, & Harbour Beacons
      const zBerthFace = zHead + dirZ * 0.14;
      for (const fx of [-11.8, -8.4, -5.0, 5.0, 8.4, 11.8]) {
        const fender = new THREE.CylinderGeometry(0.26, 0.26, 1.55, 10, 1);
        fender.translate(cx + fx, pierDeckY - 0.82, zBerthFace);
        acc.appendH(fender, fenderBlack);

        // UI Golden-Yellow Cast-Iron Twin-Horn Marine Mooring Bollard on T-head edge
        const bBase = new THREE.BoxGeometry(0.54, 0.10, 0.36);
        bBase.translate(cx + fx, pierDeckY + 0.05, zHead - dirZ * 0.55);
        acc.appendH(bBase, fenderBlack);
        for (const hx of [-0.14, 0.14]) {
          const horn = new THREE.CylinderGeometry(0.085, 0.10, 0.36, 8, 1);
          horn.translate(cx + fx + hx, pierDeckY + 0.26, zHead - dirZ * 0.55);
          acc.appendH(horn, uiGold);
        }
      }

      // 2 International-Orange SOLAS Lifebuoy Rings mounted on T-Head Safety Posts
      for (const lx of [-4.2, 4.2]) {
        const post = new THREE.CylinderGeometry(0.045, 0.05, 1.35, 8, 1);
        post.translate(cx + lx, pierDeckY + 0.68, tHeadMidZ);
        acc.appendV(post, [0.95, 0.94, 0.90]);

        const ring = new THREE.TorusGeometry(0.32, 0.085, 8, 16);
        ring.translate(cx + lx, pierDeckY + 1.05, tHeadMidZ + dirZ * 0.09);
        acc.appendH(ring, lifebuoyOrange);
      }

      // Twin Driven Mooring Dolphin Pile Clusters & Night-Glowing Port-Red / Starboard-Green Harbour Entrance Beacons!
      for (const sx of [-1.0, 1.0]) {
        const dx = cx + sx * (tHeadHW + 2.35);
        const dz = zHead + dirZ * 0.45;
        const dolphH = 3.15 - -6.50;
        const mainDolph = new THREE.CylinderGeometry(0.30, 0.36, dolphH, 10, 1);
        mainDolph.translate(dx, -6.50 + dolphH * 0.5, dz);
        acc.appendH(mainDolph, uiGold);

        const cap = new THREE.CylinderGeometry(0.42, 0.38, 0.24, 10, 1);
        cap.translate(dx, 3.18, dz);
        acc.appendV(cap, [0.94, 0.93, 0.90]);

        const beacon = new THREE.SphereGeometry(0.20, 12, 8);
        beacon.translate(dx, 3.44, dz);
        // Starboard Green on +X north side, Flashing Port Red on -X south side (y = 3.44m > 2.4m night maritime threshold!)
        acc.appendH(beacon, sx > 0 ? [0.06, 0.94, 0.44] : [0.96, 0.09, 0.07]);

        // Tire/rubber fender collar at waterline
        const collar = new THREE.CylinderGeometry(0.46, 0.46, 0.65, 10, 1);
        collar.translate(dx, waterY + 0.25, dz);
        acc.appendH(collar, fenderBlack);
      }
    };

    // Build Dermaga Penyeberangan Utara Pulau U (+Z eastward into Selat UI)
    buildSingleFerryTerminal(
      TerrainDEMLoader.DERMAGA_U_X,
      446.5,
      TerrainDEMLoader.DERMAGA_U_SHORE_Z,
      TerrainDEMLoader.DERMAGA_U_HEAD_Z,
      1.0
    );

    // Build Dermaga Penyeberangan Utara Pulau I (-Z westward into Selat UI)
    buildSingleFerryTerminal(
      TerrainDEMLoader.DERMAGA_I_X,
      624.5,
      TerrainDEMLoader.DERMAGA_I_SHORE_Z,
      TerrainDEMLoader.DERMAGA_I_HEAD_Z,
      -1.0
    );

    return acc.finalize();
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
   * Builds the High-Detail Outdoor Campus Monopole Widescreen LED Videotron ("Videotron Kampus UI")
   * on the South-East Rotunda Lawn (28.4, 78.2, yaw = -0.26 rad facing Lapangan Rotunda UI):
   * - Sculpted 16-sided Precast Concrete Pedestal + UI Gold Collar + Bolted Steel Flange + 12 Anchor Bolts + 8 Gusset Ribs
   * - Tapered Structural Steel Monopole Shaft + Rear IP65 Novastar/Power Control Box + Rear OSHA Caged Service Ladder
   * - Cantilever Space-Frame Steel K-Truss + 2-Tier Galvanized Rear Maintenance Catwalks with Safety Handrails
   * - 13.68m x 8.10m Die-Cast Outdoor LED Cabinet Housing, 32 Rear Service Doors with Cooling Louvers, UI Precast Side Pilasters,
   *   8-Pod Bottom Line-Array Speaker Bar, Angled Tropical Anti-Glare Rain/Sun Visor Hood, UI Makara Crown Header,
   *   Twin Night Obstruction Red Beacons, & Active 12.80m x 7.20m (16:9) SMD P8 Front LED Screen Surface (+0 Draw Calls!).
   */
  static _buildKampusUIVideotron() {
    const acc = SplineCorridorBuilder._createDualBufferAccumulator();
    const spec = SplineCorridorBuilder.VIDEOTRON_SPEC;
    const { cx, cz, yaw } = spec;
    const cY = Math.cos(yaw);
    const sY = Math.sin(yaw);
    const gy = Math.max(0.0, TerrainDEMLoader.sampleGroundHeight(cx, cz));

    // Local frame -> World coordinates: +lx = screen right, +ly = world Y, +lz = screen outward normal (+fwd)
    const toW = (lx, ly, lz) => [cx + lx * cY + lz * sY, ly, cz - lx * sY + lz * cY];
    const placeGeo = (geo, lx, ly, lz, rx = 0, ry = 0, rz = 0) => {
      if (rx !== 0) geo.rotateX(rx);
      if (ry !== 0) geo.rotateY(ry);
      if (rz !== 0) geo.rotateZ(rz);
      geo.rotateY(yaw);
      const [wx, wy, wz] = toW(lx, ly, lz);
      geo.translate(wx, wy, wz);
      return geo;
    };
    const tubeLocalH = (x0, y0, z0, x1, y1, z1, r0, r1, segs, col, capped = false) => {
      const a = toW(x0, y0, z0);
      const b = toW(x1, y1, z1);
      acc.tubeH(a[0], a[1], a[2], b[0], b[1], b[2], r0, r1, segs, col, capped);
    };

    const precastCream = [0.94, 0.92, 0.88];
    const uiGold = [0.90, 0.70, 0.16];
    const darkSteel = [0.15, 0.17, 0.20];
    const charcoalCabinet = [0.10, 0.12, 0.15];
    const matteBlackBezel = [0.055, 0.062, 0.075];
    const galvSteel = [0.58, 0.61, 0.65];
    const ctrlBoxSlate = [0.27, 0.30, 0.34];
    const speakerGrille = [0.075, 0.082, 0.096];
    const ledDoorGrey = [0.18, 0.20, 0.24];
    const beaconRed = [0.96, 0.08, 0.06];
    const statusGreen = [0.08, 0.92, 0.38];

    // 1. 16-Sided Precast Concrete Foundation Plinth, Sub-Terrain Skirt, & UI Gold Collar
    const plinthTopY = gy + 0.74;
    const plinthH = plinthTopY - SplineCorridorBuilder.SKIRT_BOTTOM_Y;
    acc.appendV(
      placeGeo(new THREE.CylinderGeometry(1.26, 1.34, plinthH, 16, 1), 0, SplineCorridorBuilder.SKIRT_BOTTOM_Y + plinthH * 0.5, 0),
      precastCream
    );
    acc.appendV(
      placeGeo(new THREE.CylinderGeometry(1.10, 1.26, 0.14, 16, 1), 0, plinthTopY + 0.07, 0),
      precastCream
    );
    acc.appendH(
      placeGeo(new THREE.CylinderGeometry(1.285, 1.285, 0.10, 16, 1), 0, gy + 0.54, 0),
      uiGold
    );

    // 2. Heavy Bolted Steel Baseplate Flange, 12 Hex Anchor Bolts, 8 Radial Gusset Ribs, & Tapered Monopole Shaft
    const flangeY = plinthTopY + 0.18;
    acc.appendH(
      placeGeo(new THREE.CylinderGeometry(1.02, 1.04, 0.08, 24, 1), 0, flangeY, 0),
      darkSteel
    );
    for (let i = 0; i < 12; i++) {
      const ang = (i / 12) * Math.PI * 2.0;
      const bx = Math.cos(ang) * 0.89;
      const bz = Math.sin(ang) * 0.89;
      acc.appendH(
        placeGeo(new THREE.CylinderGeometry(0.038, 0.038, 0.075, 6, 1), bx, flangeY + 0.065, bz),
        galvSteel
      );
    }
    for (let i = 0; i < 8; i++) {
      const ang = (i / 8) * Math.PI * 2.0;
      const gx = Math.cos(ang) * 0.74;
      const gz = Math.sin(ang) * 0.74;
      const gusset = new THREE.BoxGeometry(0.042, 0.70, 0.42);
      gusset.rotateY(-ang + Math.PI * 0.5);
      acc.appendH(placeGeo(gusset, gx, flangeY + 0.37, gz), darkSteel);
    }

    const poleBotY = flangeY + 0.02;
    const poleTopY = 5.44;
    const poleH = poleTopY - poleBotY;
    acc.appendH(
      placeGeo(new THREE.CylinderGeometry(0.52, 0.57, poleH, 24, 1), 0, poleBotY + poleH * 0.5, 0),
      darkSteel
    );
    // Mid-height welded splice flange ring & top structural saddle headplate
    acc.appendH(
      placeGeo(new THREE.CylinderGeometry(0.60, 0.60, 0.09, 24, 1), 0, 3.22, 0),
      galvSteel
    );
    acc.appendH(
      placeGeo(new THREE.BoxGeometry(1.62, 0.18, 1.18), 0, 5.42, -0.08),
      darkSteel
    );
    // Armored power & fiber-optic conduit riser pipe along rear monopole face
    tubeLocalH(-0.32, poleBotY, -0.55, -0.32, 5.38, -0.52, 0.062, 0.062, 8, galvSteel, true);

    // 3. Rear IP65 Weatherproof Novastar / Power Distribution Cabinet & OSHA Caged Maintenance Ladder
    acc.appendH(
      placeGeo(new THREE.BoxGeometry(0.74, 0.94, 0.38), -0.26, 1.92, -0.73),
      ctrlBoxSlate
    );
    acc.appendH(
      placeGeo(new THREE.BoxGeometry(0.80, 0.05, 0.44), -0.26, 2.41, -0.74, -0.10, 0, 0),
      darkSteel
    );
    for (let lv = 0; lv < 4; lv++) {
      acc.appendH(
        placeGeo(new THREE.BoxGeometry(0.52, 0.025, 0.02), -0.26, 1.66 + lv * 0.08, -0.925),
        matteBlackBezel
      );
    }
    // Live Night-Glowing Emerald Status Indicator LED on Novastar Controller Door
    acc.appendH(
      placeGeo(new THREE.SphereGeometry(0.026, 8, 6), -0.04, 2.22, -0.925),
      statusGreen
    );

    // Rear Vertical Service Ladder (lx = +0.34, lz = -0.66) + Safety Cage Hoops (y = 2.25..5.30m)
    const ladX0 = 0.16;
    const ladX1 = 0.52;
    const ladZ = -0.64;
    tubeLocalH(ladX0, flangeY + 0.04, ladZ, ladX0, 5.40, ladZ, 0.024, 0.024, 6, galvSteel, true);
    tubeLocalH(ladX1, flangeY + 0.04, ladZ, ladX1, 5.40, ladZ, 0.024, 0.024, 6, galvSteel, true);
    for (let ry = flangeY + 0.28; ry <= 5.28; ry += 0.31) {
      tubeLocalH(ladX0, ry, ladZ, ladX1, ry, ladZ, 0.016, 0.016, 6, galvSteel, false);
    }
    for (const standY of [1.55, 2.85, 4.15, 5.20]) {
      tubeLocalH(ladX0, standY, -0.50, ladX0, standY, ladZ, 0.020, 0.020, 6, darkSteel, false);
      tubeLocalH(ladX1, standY, -0.50, ladX1, standY, ladZ, 0.020, 0.020, 6, darkSteel, false);
    }
    const cageHoopYs = [2.30, 3.05, 3.80, 4.55, 5.26];
    for (const hy of cageHoopYs) {
      const hoop = new THREE.TorusGeometry(0.35, 0.017, 6, 12, Math.PI * 1.32);
      hoop.rotateX(Math.PI * 0.5);
      hoop.rotateY(-Math.PI * 0.16);
      acc.appendH(placeGeo(hoop, 0.34, hy, -0.82), galvSteel);
    }
    for (const ang of [-2.25, -1.57, -0.88]) {
      const sx = 0.34 + Math.cos(ang) * 0.35;
      const sz = -0.82 + Math.sin(ang) * 0.35;
      tubeLocalH(sx, 2.30, sz, sx, 5.26, sz, 0.014, 0.014, 6, galvSteel, false);
    }

    // 4. Cantilever Space-Frame Steel K-Truss & 2-Tier Rear Maintenance Catwalks with Handrails
    acc.appendH(placeGeo(new THREE.BoxGeometry(13.28, 0.34, 0.32), 0, 5.54, -0.38), darkSteel);
    acc.appendH(placeGeo(new THREE.BoxGeometry(13.20, 0.20, 0.24), 0, 9.36, -0.36), darkSteel);
    acc.appendH(placeGeo(new THREE.BoxGeometry(13.28, 0.28, 0.28), 0, 12.94, -0.38), darkSteel);

    const trussColsX = [-6.20, -3.72, -1.24, 1.24, 3.72, 6.20];
    for (const tx of trussColsX) {
      acc.appendH(placeGeo(new THREE.BoxGeometry(0.18, 7.56, 0.22), tx, 9.24, -0.36), darkSteel);
    }
    // Primary Cantilever Diagonal Support Struts from Monopole Collar out to Spreader Girder
    for (const sgn of [-1, 1]) {
      tubeLocalH(sgn * 0.42, 3.75, -0.14, sgn * 3.72, 5.46, -0.38, 0.095, 0.088, 10, darkSteel, true);
      tubeLocalH(sgn * 0.46, 4.35, -0.16, sgn * 6.20, 5.46, -0.38, 0.082, 0.076, 10, darkSteel, true);
    }
    // Rear Space-Frame K-Brace Diagonal Struts across Truss Bays
    for (let b = 0; b < trussColsX.length - 1; b++) {
      const xA = trussColsX[b];
      const xB = trussColsX[b + 1];
      const xMid = 0.5 * (xA + xB);
      tubeLocalH(xA, 5.62, -0.38, xMid, 9.34, -0.38, 0.048, 0.048, 6, darkSteel, false);
      tubeLocalH(xB, 5.62, -0.38, xMid, 9.34, -0.38, 0.048, 0.048, 6, darkSteel, false);
      tubeLocalH(xA, 12.88, -0.38, xMid, 9.38, -0.38, 0.048, 0.048, 6, darkSteel, false);
      tubeLocalH(xB, 12.88, -0.38, xMid, 9.38, -0.38, 0.048, 0.048, 6, darkSteel, false);
    }

    // 2-Tier Rear Galvanized Service Catwalks (y = 5.42m & y = 9.38m) + 0.98m Safety Handrails
    for (const cwY of [5.42, 9.38]) {
      acc.appendH(placeGeo(new THREE.BoxGeometry(13.06, 0.055, 0.52), 0, cwY, -0.70), galvSteel);
      acc.appendH(placeGeo(new THREE.BoxGeometry(13.06, 0.11, 0.03), 0, cwY + 0.06, -0.95), darkSteel);
      tubeLocalH(-6.52, cwY + 0.98, -0.95, 6.52, cwY + 0.98, -0.95, 0.022, 0.022, 6, galvSteel, true);
      tubeLocalH(-6.52, cwY + 0.50, -0.95, 6.52, cwY + 0.50, -0.95, 0.018, 0.018, 6, galvSteel, true);
      for (let px = -6.50; px <= 6.51; px += 2.166) {
        tubeLocalH(px, cwY, -0.95, px, cwY + 0.98, -0.95, 0.022, 0.022, 6, galvSteel, true);
      }
      for (const endSgn of [-1, 1]) {
        const ex = endSgn * 6.51;
        tubeLocalH(ex, cwY + 0.98, -0.44, ex, cwY + 0.98, -0.95, 0.020, 0.020, 6, galvSteel, false);
        tubeLocalH(ex, cwY + 0.50, -0.44, ex, cwY + 0.50, -0.95, 0.018, 0.018, 6, galvSteel, false);
      }
    }

    // 5. Modular Outdoor Die-Cast LED Cabinet Housing, 32 Rear Service Doors, UI Side Pilasters & Bezel
    // Backing enclosure behind active LED plane (zLocal = -0.24 .. +0.26m)
    acc.appendH(placeGeo(new THREE.BoxGeometry(13.36, 7.96, 0.50), 0, 9.38, 0.01), charcoalCabinet);

    // 4 Front Recessed Matte-Black Anti-Reflective Bezel Frame Bars around exact 12.80m x 7.20m (y = 5.88..13.08m) LED Matrix
    acc.appendH(placeGeo(new THREE.BoxGeometry(13.36, 0.28, 0.11), 0, 13.22, 0.305), matteBlackBezel);
    acc.appendH(placeGeo(new THREE.BoxGeometry(13.36, 0.10, 0.11), 0, 5.83, 0.305), matteBlackBezel);
    for (const sgn of [-1, 1]) {
      acc.appendH(placeGeo(new THREE.BoxGeometry(0.28, 7.20, 0.11), sgn * 6.54, 9.48, 0.305), matteBlackBezel);
      // Architectural UI Precast Concrete Side Cheek Pilasters + Vertical UI Gold Pinstripe Accent
      acc.appendV(placeGeo(new THREE.BoxGeometry(0.18, 8.04, 0.58), sgn * 6.76, 9.38, 0.04), precastCream);
      acc.appendH(placeGeo(new THREE.BoxGeometry(0.045, 7.88, 0.04), sgn * 6.76, 9.38, 0.345), uiGold);
    }

    // 8 x 4 = 32 Individual Rear Die-Cast LED Service Cabinet Doors with Cooling Louvers & Latches (zLocal = -0.255m)
    for (let r = 0; r < 4; r++) {
      const cyDoor = 6.38 + r * 1.84;
      for (let c = 0; c < 8; c++) {
        const cxDoor = -5.60 + c * 1.60;
        acc.appendH(placeGeo(new THREE.BoxGeometry(1.48, 1.70, 0.036), cxDoor, cyDoor, -0.255), ledDoorGrey);
        acc.appendH(placeGeo(new THREE.BoxGeometry(1.16, 0.12, 0.018), cxDoor, cyDoor + 0.52, -0.28), matteBlackBezel);
        acc.appendH(placeGeo(new THREE.BoxGeometry(1.16, 0.12, 0.018), cxDoor, cyDoor - 0.52, -0.28), matteBlackBezel);
        acc.appendH(placeGeo(new THREE.BoxGeometry(0.06, 0.22, 0.025), cxDoor + 0.58, cyDoor, -0.282), galvSteel);
      }
    }

    // 6. Bottom Integrated 8-Pod Outdoor Line-Array Speaker Bar (y = 5.38..5.80m, zLocal = +0.33m)
    acc.appendH(placeGeo(new THREE.BoxGeometry(13.36, 0.42, 0.16), 0, 5.59, 0.31), charcoalCabinet);
    for (let sp = 0; sp < 8; sp++) {
      const sxPod = -5.46 + sp * 1.56;
      acc.appendH(
        placeGeo(new THREE.BoxGeometry(1.38, 0.30, 0.05), sxPod, 5.60, 0.40, 0.08, 0, 0),
        speakerGrille
      );
    }
    acc.appendH(placeGeo(new THREE.BoxGeometry(3.20, 0.055, 0.03), 0, 5.40, 0.395), uiGold);

    // 7. Top Tropical Anti-Glare Rain & Sun Visor Hood (projecting +0.48m forward at y = 13.12..13.36m)
    acc.appendH(
      placeGeo(new THREE.BoxGeometry(13.74, 0.075, 0.52), 0, 13.25, 0.52, 0.22, 0, 0),
      charcoalCabinet
    );
    acc.appendH(placeGeo(new THREE.BoxGeometry(13.74, 0.045, 0.04), 0, 13.18, 0.77), uiGold);
    for (const sgn of [-1, 1]) {
      // Side sun-blinder wings flanking the LED display
      acc.appendH(placeGeo(new THREE.BoxGeometry(0.055, 7.42, 0.26), sgn * 6.66, 9.52, 0.46), charcoalCabinet);
    }

    // 8. Top UI Nusantara Header Crown & Sculpted 3D Golden Makara Diamond Crest + Twin Red Night Beacons
    acc.appendH(placeGeo(new THREE.BoxGeometry(8.60, 0.48, 0.26), 0, 13.60, 0.22), charcoalCabinet);
    acc.appendH(placeGeo(new THREE.BoxGeometry(8.76, 0.06, 0.30), 0, 13.86, 0.23), uiGold);
    const diamondCrest = new THREE.BoxGeometry(0.32, 0.32, 0.07);
    diamondCrest.rotateZ(Math.PI * 0.25);
    acc.appendH(placeGeo(diamondCrest, 0, 13.61, 0.37), uiGold);
    for (const sgn of [-1, 1]) {
      acc.appendH(placeGeo(new THREE.BoxGeometry(2.85, 0.075, 0.04), sgn * 2.15, 13.61, 0.365), uiGold);
      // Twin Red Night Aviation / Obstruction Beacons (y = 13.94m > 5.2m night pulsing threshold!)
      tubeLocalH(sgn * 6.56, 13.40, 0.08, sgn * 6.56, 13.84, 0.08, 0.032, 0.028, 8, galvSteel, true);
      acc.appendH(placeGeo(new THREE.SphereGeometry(0.11, 10, 8), sgn * 6.56, 13.93, 0.08), beaconRed);
    }

    // 9. Active 12.80m x 7.20m (16:9) Front SMD P8 LED Screen Quad (Tagged with UV Sentinel v = -1050.0!)
    const screenPlane = placeGeo(
      new THREE.PlaneGeometry(spec.screenW, spec.screenH, 1, 1),
      0,
      spec.screenCenterY,
      spec.screenFwdOffset
    );
    acc.appendH(screenPlane, [0.06, 0.07, 0.09], [0.5, -1050.0]);

    return acc.finalize();
  }

  /**
   * Builds Full Multi-Lane Campus Road Markings ("Marka Lajur Jalan Kampus UI") & Flush Junction Aprons:
   * - Flush asphalt intersection flare pads eliminating any visual seam at Rotunda/West Boulevard/West Parking/Teksas throats
   * - Lapangan Rotunda UI 2-Lane Concentric Markings (dashed white lane divider at R = 28.80m, inner/outer white edge lines,
   *   8 clockwise directional circulatory lane arrows, & 3 radial pedestrian zebra crosswalks)
   * - RotundaWestCampusBoulevard 2-lane UI-Yellow dashed centerline, edge lines, directional arrows, & zebra crosswalk
   * - WestCourtyardParkingLane UI-Yellow dashed centerline, stop bar, & angled white parking bay markings
   * Batched into corridorGroup for +0 extra Draw Calls!
   */
  static _buildCampusRoadLaneMarkings() {
    const acc = SplineCorridorBuilder._createDualBufferAccumulator();
    const rCx = SplineCorridorBuilder.ROTUNDA_CX;
    const rCz = SplineCorridorBuilder.ROTUNDA_CZ;

    const asphaltApron = [0.20, 0.21, 0.23];
    const paverApron = [0.46, 0.45, 0.42];
    const whiteLine = [0.95, 0.96, 0.98];
    const uiYellowLine = [0.96, 0.77, 0.12];
    const skirtCol = [0.42, 0.41, 0.39];

    const addFlatPad = (cx, yTop, cz, sx, sz, yaw = 0.0, col = asphaltApron) => {
      const top = new THREE.BoxGeometry(sx, 0.024, sz);
      if (yaw !== 0.0) top.rotateY(yaw);
      top.translate(cx, yTop - 0.012, cz);
      acc.appendH(top, col);

      const sub = new THREE.BoxGeometry(Math.max(0.3, sx - 0.2), 1.8, Math.max(0.3, sz - 0.2));
      if (yaw !== 0.0) sub.rotateY(yaw);
      sub.translate(cx, yTop - 0.92, cz);
      acc.appendV(sub, skirtCol);
    };

    // 1. Flush Asphalt / Stone Junction Flare Pads at Campus Core Intersections
    addFlatPad(-36.2, 0.044, rCz, 3.6, 9.2, 0.0, asphaltApron);
    addFlatPad(-62.5, 0.042, 101.6, 11.6, 3.2, 0.0, asphaltApron);
    addFlatPad(-20.4, 0.044, 135.0, 4.9, 3.2, -0.57, paverApron);

    // 2. Lapangan Rotunda UI 2-Lane Concentric Dashed Divider (R = 28.80m) & Inner/Outer White Edge Stripes
    const dashCount = 48;
    for (let i = 0; i < dashCount; i++) {
      const th0 = (i / dashCount) * Math.PI * 2.0;
      const th1 = ((i + 0.52) / dashCount) * Math.PI * 2.0;
      const subSteps = 3;
      for (let s = 0; s < subSteps; s++) {
        const a0 = th0 + ((th1 - th0) * s) / subSteps;
        const a1 = th0 + ((th1 - th0) * (s + 1)) / subSteps;
        const rIn = 28.71;
        const rOut = 28.89;
        acc.quadH(
          rCx + Math.cos(a0) * rIn,
          0.056,
          rCz + Math.sin(a0) * rIn,
          rCx + Math.cos(a1) * rIn,
          0.056,
          rCz + Math.sin(a1) * rIn,
          rCx + Math.cos(a1) * rOut,
          0.056,
          rCz + Math.sin(a1) * rOut,
          rCx + Math.cos(a0) * rOut,
          0.056,
          rCz + Math.sin(a0) * rOut,
          whiteLine,
          0,
          1,
          0
        );
      }
    }

    const edgeSegs = 160;
    for (let i = 0; i < edgeSegs; i++) {
      const a0 = (i / edgeSegs) * Math.PI * 2.0;
      const a1 = ((i + 1) / edgeSegs) * Math.PI * 2.0;
      const aMid = 0.5 * (a0 + a1);
      const pxOut = rCx + Math.cos(aMid) * 33.4;
      const pzOut = rCz + Math.sin(aMid) * 33.4;

      // Continuous inner edge line (R = 24.16 .. 24.30m)
      acc.quadH(
        rCx + Math.cos(a0) * 24.16,
        0.044,
        rCz + Math.sin(a0) * 24.16,
        rCx + Math.cos(a1) * 24.16,
        0.044,
        rCz + Math.sin(a1) * 24.16,
        rCx + Math.cos(a1) * 24.30,
        0.044,
        rCz + Math.sin(a1) * 24.30,
        rCx + Math.cos(a0) * 24.30,
        0.044,
        rCz + Math.sin(a0) * 24.30,
        whiteLine,
        0,
        1,
        0
      );

      // Outer edge line (R = 33.28 .. 33.42m), interrupted cleanly across West Boulevard & Teksas mouths
      const inWestMouth = Math.hypot(pxOut - -36.2, pzOut - rCz) < 5.2;
      const inTeksasMouth = Math.hypot(pxOut - -20.4, pzOut - 135.0) < 3.0;
      if (!inWestMouth && !inTeksasMouth) {
        acc.quadH(
          rCx + Math.cos(a0) * 33.28,
          0.044,
          rCz + Math.sin(a0) * 33.28,
          rCx + Math.cos(a1) * 33.28,
          0.044,
          rCz + Math.sin(a1) * 33.28,
          rCx + Math.cos(a1) * 33.42,
          0.044,
          rCz + Math.sin(a1) * 33.42,
          rCx + Math.cos(a0) * 33.42,
          0.044,
          rCz + Math.sin(a0) * 33.42,
          whiteLine,
          0,
          1,
          0
        );
      }
    }

    // Helper: Directional Lane Arrow at (ax, ay, az) pointing along heading yawRad
    const addLaneArrow = (ax, ay, az, yawRad) => {
      const shaft = new THREE.BoxGeometry(0.24, 0.014, 1.85);
      shaft.rotateY(yawRad);
      shaft.translate(ax - Math.sin(yawRad) * 0.45, ay, az - Math.cos(yawRad) * 0.45);
      acc.appendH(shaft, whiteLine);

      const head = new THREE.ConeGeometry(0.46, 0.95, 3, 1);
      head.rotateX(Math.PI * 0.5);
      head.scale(1.0, 0.04, 1.0);
      head.rotateY(yawRad);
      head.translate(ax + Math.sin(yawRad) * 0.72, ay, az + Math.cos(yawRad) * 0.72);
      acc.appendH(head, whiteLine);
    };

    // 8 Clockwise Circulatory Arrows on Lapangan Rotunda UI (Inner Lane R = 26.35m & Outer Lane R = 31.25m)
    for (let q = 0; q < 4; q++) {
      const th = q * Math.PI * 0.5 + 0.38;
      // Clockwise tangent direction in XZ: (-sin(th), +cos(th)) -> yaw = atan2(-sin(th), cos(th)) = -th
      const cwYaw = Math.atan2(-Math.sin(th), Math.cos(th));
      for (const rLane of [26.35, 31.25]) {
        addLaneArrow(rCx + Math.cos(th) * rLane, 0.056, rCz + Math.sin(th) * rLane, cwYaw);
      }
    }

    // 3 Radial Zebra Pedestrian Crosswalks on Lapangan Rotunda UI (North Rektorat Gate, East SE Walkway, South Teksas)
    for (const thCross of [-Math.PI * 0.5, 0.16, 2.02]) {
      const cT = Math.cos(thCross);
      const sT = Math.sin(thCross);
      const yawCross = Math.atan2(-sT, cT);
      for (let st = 0; st < 8; st++) {
        const rStripe = 24.75 + st * 1.15;
        const bar = new THREE.BoxGeometry(0.54, 0.015, 2.85);
        bar.rotateY(yawCross);
        bar.translate(rCx + cT * rStripe, 0.056, rCz + sT * rStripe);
        acc.appendH(bar, whiteLine);
      }
    }

    // 3. RotundaWestCampusBoulevard (x = -37.5 .. -81.0, z = 106.5): 2-Lane UI-Yellow Dashed Centerline, Edge Lines, Crosswalk & Arrows
    for (let x = -39.5; x >= -79.5; x -= 5.2) {
      const dash = new THREE.BoxGeometry(2.7, 0.014, 0.18);
      dash.translate(x - 1.35, 0.054, rCz);
      acc.appendH(dash, uiYellowLine);
    }
    for (const sz of [-1, 1]) {
      const zEdge = rCz + sz * 4.18;
      for (let x = -37.2; x >= -80.4; x -= 2.2) {
        const xm = x - 1.1;
        if (sz < 0 && xm < -56.2 && xm > -68.8) continue; // Open across WestCourtyardParkingLane mouth!
        const eSeg = new THREE.BoxGeometry(2.22, 0.012, 0.15);
        eSeg.translate(xm, 0.044, zEdge);
        acc.appendH(eSeg, whiteLine);
      }
    }
    // Pedestrian Zebra Crosswalk & Stop Bars at West Boulevard / Rotunda Throat (x = -38.4)
    for (let s = -3; s <= 3; s++) {
      const stripe = new THREE.BoxGeometry(2.6, 0.015, 0.54);
      stripe.translate(-38.4, 0.055, rCz + s * 1.12);
      acc.appendH(stripe, whiteLine);
    }
    addLaneArrow(-48.5, 0.054, rCz - 2.15, -Math.PI * 0.5);
    addLaneArrow(-48.5, 0.054, rCz + 2.15, Math.PI * 0.5);
    addLaneArrow(-74.0, 0.054, rCz - 2.15, -Math.PI * 0.5);
    addLaneArrow(-74.0, 0.054, rCz + 2.15, Math.PI * 0.5);

    // 4. WestCourtyardParkingLane (x = -62.5, z = -33.0 .. 100.5): Dashed Centerline, Stop Bar, Crosswalk, & Parking Bay Stalls
    for (let z = 96.5; z >= -30.0; z -= 5.6) {
      const dash = new THREE.BoxGeometry(0.18, 0.014, 2.8);
      dash.translate(-62.5, 0.052, z - 1.4);
      acc.appendH(dash, uiYellowLine);
    }
    for (let s = -4; s <= 4; s++) {
      const stripe = new THREE.BoxGeometry(0.54, 0.015, 2.5);
      stripe.translate(-62.5 + s * 1.15, 0.053, 99.8);
      acc.appendH(stripe, whiteLine);
    }
    const stopBar = new THREE.BoxGeometry(5.2, 0.015, 0.36);
    stopBar.translate(-59.8, 0.053, 97.6);
    acc.appendH(stopBar, whiteLine);

    // White perpendicular parking bay stall dividers along East side of WestCourtyardParkingLane (x = -59.2 .. -56.9)
    for (let zSlot = -28.0; zSlot <= 64.0; zSlot += 3.2) {
      const stall = new THREE.BoxGeometry(2.35, 0.012, 0.13);
      stall.translate(-58.0, 0.044, zSlot);
      acc.appendH(stall, whiteLine);
    }

    return acc.finalize();
  }

  /**
   * Builds the Dedicated EPDM Campus Jogging Tracks & Outdoor Street-Workout / Calisthenics Park ("Sudut Calisthenics UI"):
   * 1. Lintasan Joging EPDM Taman Rotunda (332° terracotta-red & royal-blue 2-lane cushioned running track at R = 37.15m)
   * 2. Lintasan Joging EPDM Taman Teluk U (100% closed 2-lane oval running track wrapping around (-62.0, 136.5))
   * 3. Sudut Calisthenics UI inside the Teluk U oval infield (-62.0, 136.5): multi-bay pull-up & monkey-bar rig,
   *    Swedish stall ladder, parallel dip bars, push-up bars, 2 inclined sit-up benches, 3 plyo boxes, & illuminated guide board.
   * Batched into corridorGroup for +0 extra Draw Calls!
   */
  static _buildCampusJoggingTrackAndCalisthenics() {
    const acc = SplineCorridorBuilder._createDualBufferAccumulator();
    const rCx = SplineCorridorBuilder.ROTUNDA_CX;
    const rCz = SplineCorridorBuilder.ROTUNDA_CZ;
    const spec = SplineCorridorBuilder.CALISTHENICS_SPEC;

    const epdmRed = [0.71, 0.23, 0.16];
    const epdmBlue = [0.12, 0.35, 0.72];
    const epdmCharcoal = [0.16, 0.18, 0.21];
    const whiteLine = [0.95, 0.96, 0.98];
    const uiGold = [0.93, 0.70, 0.09];
    const anthraciteSteel = [0.14, 0.16, 0.19];
    const stainlessBar = [0.76, 0.79, 0.83];
    const teakWood = [0.54, 0.33, 0.18];
    const concreteBase = [0.86, 0.85, 0.82];
    const warmLed = [0.99, 0.94, 0.78];

    // Helper: Lofted 2-Lane Cushioned EPDM Running Track along an array of 2D centerline points
    const buildEPDMTrack = (pts2D, closed, halfW = 1.05) => {
      const count = pts2D.length;
      const segs = closed ? count : count - 1;
      const borderW = 0.14;
      const frames = [];
      let arc = 0.0;

      for (let i = 0; i <= segs; i++) {
        const idx = i % count;
        const p = pts2D[idx];
        const pPrev = pts2D[closed ? (idx - 1 + count) % count : Math.max(0, idx - 1)];
        const pNext = pts2D[closed ? (idx + 1) % count : Math.min(count - 1, idx + 1)];
        let tx = pNext[0] - pPrev[0];
        let tz = pNext[1] - pPrev[1];
        const tl = Math.hypot(tx, tz) || 1.0;
        tx /= tl;
        tz /= tl;
        const sx = tz;
        const sz = -tx;
        const gy = Math.max(0.0, TerrainDEMLoader.sampleGroundHeight(p[0], p[1]));
        if (i > 0) {
          const prev = frames[i - 1];
          arc += Math.hypot(p[0] - prev.x, p[1] - prev.z);
        }
        frames.push({ x: p[0], y: gy + 0.026, z: p[1], sx, sz, arc });
      }

      for (let i = 0; i < segs; i++) {
        const f0 = frames[i];
        const f1 = frames[i + 1];

        // Skip raised EPDM across paved RotundaEastTeksasConnector walkway crossing (-22.6, 137.8) so walkway stays flush
        const xm = 0.5 * (f0.x + f1.x);
        const zm = 0.5 * (f0.z + f1.z);
        const onTeksasPromenade = Math.hypot(xm - -22.5, zm - 137.5) < 2.65;
        const yLift = onTeksasPromenade ? 0.022 : 0.0;

        // Main terracotta-red EPDM running deck
        acc.quadH(
          f0.x - f0.sx * halfW,
          f0.y + yLift,
          f0.z - f0.sz * halfW,
          f1.x - f1.sx * halfW,
          f1.y + yLift,
          f1.z - f1.sz * halfW,
          f1.x + f1.sx * halfW,
          f1.y + yLift,
          f1.z + f1.sz * halfW,
          f0.x + f0.sx * halfW,
          f0.y + yLift,
          f0.z + f0.sz * halfW,
          epdmRed,
          0,
          1,
          0
        );

        // Royal-blue EPDM outer & inner edge trim borders + concrete sub-skirt
        for (const sgn of [-1, 1]) {
          const rA = sgn * halfW;
          const rB = sgn * (halfW + borderW);
          acc.quadH(
            f0.x + f0.sx * rA,
            f0.y + yLift - 0.002,
            f0.z + f0.sz * rA,
            f1.x + f1.sx * rA,
            f1.y + yLift - 0.002,
            f1.z + f1.sz * rA,
            f1.x + f1.sx * rB,
            f1.y + yLift - 0.004,
            f1.z + f1.sz * rB,
            f0.x + f0.sx * rB,
            f0.y + yLift - 0.004,
            f0.z + f0.sz * rB,
            epdmBlue,
            0,
            1,
            0
          );
          acc.quadV(
            f0.x + f0.sx * rB,
            f0.y + yLift - 0.004,
            f0.z + f0.sz * rB,
            f1.x + f1.sx * rB,
            f1.y + yLift - 0.004,
            f1.z + f1.sz * rB,
            f1.x + f1.sx * rB,
            f1.y - 0.35,
            f1.z + f1.sz * rB,
            f0.x + f0.sx * rB,
            f0.y - 0.35,
            f0.z + f0.sz * rB,
            concreteBase,
            sgn * f0.sx,
            0,
            sgn * f0.sz
          );
        }

        // Crisp white 2-lane center divider stripe + 25m distance tick bars
        acc.quadH(
          f0.x - f0.sx * 0.035,
          f0.y + yLift + 0.004,
          f0.z - f0.sz * 0.035,
          f1.x - f1.sx * 0.035,
          f1.y + yLift + 0.004,
          f1.z - f1.sz * 0.035,
          f1.x + f1.sx * 0.035,
          f1.y + yLift + 0.004,
          f1.z + f1.sz * 0.035,
          f0.x + f0.sx * 0.035,
          f0.y + yLift + 0.004,
          f0.z + f0.sz * 0.035,
          whiteLine,
          0,
          1,
          0
        );

        if (Math.floor(f0.arc / 25.0) !== Math.floor(f1.arc / 25.0)) {
          acc.quadH(
            f0.x - f0.sx * (halfW - 0.12),
            f0.y + yLift + 0.005,
            f0.z - f0.sz * (halfW - 0.12),
            f1.x - f1.sx * (halfW - 0.12),
            f1.y + yLift + 0.005,
            f1.z - f1.sz * (halfW - 0.12),
            f1.x + f1.sx * (halfW - 0.12),
            f1.y + yLift + 0.005,
            f1.z + f1.sz * (halfW - 0.12),
            f0.x + f0.sx * (halfW - 0.12),
            f0.y + yLift + 0.005,
            f0.z + f0.sz * (halfW - 0.12),
            whiteLine,
            0,
            1,
            0
          );
        }
      }

      if (!closed && frames.length >= 2) {
        for (const fEnd of [frames[0], frames[frames.length - 1]]) {
          const cap = new THREE.CylinderGeometry(halfW + borderW, halfW + borderW, 0.36, 18, 1);
          cap.translate(fEnd.x, fEnd.y - 0.18, fEnd.z);
          acc.appendH(cap, epdmBlue);
          const pad = new THREE.CylinderGeometry(halfW, halfW, 0.02, 18, 1);
          pad.translate(fEnd.x, fEnd.y - 0.006, fEnd.z);
          acc.appendH(pad, epdmRed);
        }
      }
    };

    // 1. Lintasan Joging EPDM Taman Rotunda UI (R = 37.15m, -75° .. +255° framing Lapangan Rotunda UI!)
    const rotundaTrackPts = [];
    const rotTrackSegs = 132;
    const thStart = (-75.0 * Math.PI) / 180.0;
    const thEnd = (255.0 * Math.PI) / 180.0;
    for (let i = 0; i <= rotTrackSegs; i++) {
      const th = thStart + ((thEnd - thStart) * i) / rotTrackSegs;
      rotundaTrackPts.push([
        rCx + Math.cos(th) * spec.rotundaTrackR,
        rCz + Math.sin(th) * spec.rotundaTrackR
      ]);
    }
    buildEPDMTrack(rotundaTrackPts, false, 1.02);

    // 2. Lintasan Joging EPDM Taman Teluk U (100% Closed 2-Lane Athletic Oval wrapping Sudut Calisthenics!)
    const ovalPts = [];
    const ovalSegs = 72;
    for (let i = 0; i < ovalSegs; i++) {
      const th = (i / ovalSegs) * Math.PI * 2.0;
      ovalPts.push([
        spec.cx + Math.cos(th) * spec.ovalRx,
        spec.cz + Math.sin(th) * spec.ovalRz
      ]);
    }
    buildEPDMTrack(ovalPts, true, 1.08);

    // 3. Sudut Calisthenics UI (Outdoor Street-Workout & Fitness Court inside the Teluk U Oval Infield!)
    const cx = spec.cx;
    const cz = spec.cz;
    const gy = Math.max(0.0, TerrainDEMLoader.sampleGroundHeight(cx, cz));
    const padY = gy + 0.036;

    // Shock-absorbing Charcoal & Royal-Blue EPDM Fitness Court Pad (15.6m x 6.4m) + UI-Gold Border
    const courtSub = new THREE.BoxGeometry(16.0, 0.48, 6.8);
    courtSub.translate(cx, padY - 0.24, cz);
    acc.appendV(courtSub, concreteBase);

    const courtBorder = new THREE.BoxGeometry(15.8, 0.03, 6.6);
    courtBorder.translate(cx, padY - 0.01, cz);
    acc.appendH(courtBorder, uiGold);

    const courtMain = new THREE.BoxGeometry(15.3, 0.034, 6.1);
    courtMain.translate(cx, padY, cz);
    acc.appendH(courtMain, epdmCharcoal);

    for (const zoneX of [-4.4, 0.8, 5.2]) {
      const zoneMat = new THREE.BoxGeometry(3.8, 0.038, 4.6);
      zoneMat.translate(cx + zoneX, padY + 0.003, cz);
      acc.appendH(zoneMat, epdmBlue);
    }

    // 3A. Main Multi-Bay Pull-Up, Monkey-Bar & Swedish Stall-Ladder Rig (centered at cx - 4.4, cz)
    const rigX = cx - 4.4;
    const postCoords = [
      [-1.75, -1.10, 2.48],
      [0.0, -1.10, 2.48],
      [1.75, -1.10, 2.48],
      [-1.75, 1.10, 2.48],
      [0.0, 1.10, 2.48],
      [1.75, 1.10, 2.48],
      [-2.75, -1.10, 2.15],
      [-2.75, 0.0, 2.35]
    ];
    for (const [ox, oz, h] of postCoords) {
      const flange = new THREE.CylinderGeometry(0.11, 0.12, 0.05, 8, 1);
      flange.translate(rigX + ox, padY + 0.04, cz + oz);
      acc.appendH(flange, anthraciteSteel);
      acc.tubeH(rigX + ox, padY + 0.04, cz + oz, rigX + ox, padY + h, cz + oz, 0.054, 0.054, 8, uiGold, true);
    }
    // Longitudinal & transverse overhead pull-up bars + 11-rung horizontal monkey-bar ladder
    for (const oz of [-1.10, 1.10]) {
      acc.tubeH(rigX - 1.75, padY + 2.34, cz + oz, rigX + 1.75, padY + 2.34, cz + oz, 0.032, 0.032, 8, anthraciteSteel, false);
    }
    for (const ox of [-1.75, 0.0, 1.75]) {
      acc.tubeH(rigX + ox, padY + 2.24, cz - 1.10, rigX + ox, padY + 2.24, cz + 1.10, 0.028, 0.028, 8, stainlessBar, false);
    }
    for (let r = 0; r <= 10; r++) {
      const rx = rigX - 1.55 + r * 0.31;
      acc.tubeH(rx, padY + 2.34, cz - 1.10, rx, padY + 2.34, cz + 1.10, 0.022, 0.022, 6, stainlessBar, false);
    }
    // Side Freestyle Pull-Up Bar & Vertical Swedish Stall Ladder (8 rungs)
    acc.tubeH(rigX - 2.75, padY + 2.04, cz - 1.10, rigX - 1.75, padY + 2.04, cz - 1.10, 0.028, 0.028, 8, stainlessBar, false);
    for (let sr = 0; sr < 8; sr++) {
      const ry = padY + 0.38 + sr * 0.25;
      acc.tubeH(rigX - 2.75, ry, cz - 1.10, rigX - 2.75, ry, cz, 0.020, 0.020, 6, stainlessBar, false);
    }
    // Human-flag vertical grip pole & low/mid push-up bars
    acc.tubeH(rigX + 1.75, padY + 0.32, cz + 1.10, rigX + 1.75, padY + 1.45, cz + 1.85, 0.024, 0.024, 6, stainlessBar, true);
    for (const [px0, pz0, px1, pz1, ph] of [
      [rigX - 0.8, cz + 1.85, rigX + 0.6, cz + 1.85, 0.32],
      [rigX + 0.6, cz + 1.85, rigX + 1.75, cz + 1.85, 0.52]
    ]) {
      acc.tubeH(px0, padY + 0.02, pz0, px0, padY + ph + 0.04, pz0, 0.036, 0.036, 6, uiGold, true);
      acc.tubeH(px1, padY + 0.02, pz1, px1, padY + ph + 0.04, pz1, 0.036, 0.036, 6, uiGold, true);
      acc.tubeH(px0, padY + ph, pz0, px1, padY + ph, pz1, 0.024, 0.024, 6, stainlessBar, false);
    }

    // 3B. Parallel Dip Bars & Twin Inclined Sit-Up / Core Benches (centered at cx + 0.8, cz)
    const dipX = cx + 0.1;
    for (const dz of [-0.34, 0.34]) {
      for (const dx of [-0.95, 0.95]) {
        acc.tubeH(dipX + dx, padY + 0.02, cz - 1.15 + dz, dipX + dx, padY + 1.22, cz - 1.15 + dz, 0.038, 0.038, 8, uiGold, true);
      }
      acc.tubeH(dipX - 1.08, padY + 1.22, cz - 1.15 + dz, dipX + 1.08, padY + 1.22, cz - 1.15 + dz, 0.028, 0.028, 8, stainlessBar, true);
    }

    for (const bxOff of [-0.55, 0.85]) {
      const abX = cx + 0.6 + bxOff;
      const abZ = cz + 1.15;
      const board = new THREE.BoxGeometry(0.52, 0.06, 1.68);
      board.rotateX(0.24);
      board.translate(abX, padY + 0.42, abZ);
      acc.appendH(board, teakWood);
      acc.tubeH(abX, padY + 0.02, abZ - 0.68, abX, padY + 0.62, abZ - 0.68, 0.036, 0.036, 6, anthraciteSteel, true);
      acc.tubeH(abX, padY + 0.02, abZ + 0.68, abX, padY + 0.24, abZ + 0.68, 0.036, 0.036, 6, anthraciteSteel, true);
      acc.tubeH(abX - 0.30, padY + 0.66, abZ - 0.72, abX + 0.30, padY + 0.66, abZ - 0.72, 0.032, 0.032, 6, epdmCharcoal, true);
    }

    // 3C. 3 Stepped Plyometric Jump Boxes & Illuminated UI Calisthenics Workout Guide Board (cx + 5.2, cz)
    const plyoSpecs = [
      { ox: 4.3, oz: -0.9, h: 0.40, col: epdmBlue },
      { ox: 5.3, oz: -0.2, h: 0.56, col: uiGold },
      { ox: 4.5, oz: 0.9, h: 0.72, col: epdmRed }
    ];
    for (const pl of plyoSpecs) {
      const box = new THREE.BoxGeometry(0.76, pl.h, 0.76);
      box.translate(cx + pl.ox, padY + pl.h * 0.5 + 0.015, cz + pl.oz);
      acc.appendH(box, anthraciteSteel);
      const pad = new THREE.BoxGeometry(0.72, 0.035, 0.72);
      pad.translate(cx + pl.ox, padY + pl.h + 0.02, cz + pl.oz);
      acc.appendH(pad, pl.col);
    }

    const signX = cx + 6.55;
    const signZ = cz - 1.75;
    for (const sz of [-0.52, 0.52]) {
      acc.tubeH(signX, padY + 0.02, signZ + sz, signX, padY + 2.05, signZ + sz, 0.038, 0.038, 6, anthraciteSteel, true);
    }
    const boardFrame = new THREE.BoxGeometry(0.10, 1.05, 1.22);
    boardFrame.translate(signX, padY + 1.45, signZ);
    acc.appendH(boardFrame, anthraciteSteel);
    const boardHeader = new THREE.BoxGeometry(0.12, 0.20, 1.18);
    boardHeader.translate(signX, padY + 1.86, signZ);
    acc.appendH(boardHeader, uiGold);
    const boardLight = new THREE.BoxGeometry(0.14, 0.05, 1.08);
    boardLight.translate(signX, padY + 2.01, signZ);
    acc.appendH(boardLight, warmLed, [0.5, -890.0]);

    return acc.finalize();
  }

  /**
   * Builds the Compact Air Traffic Control Tower ("ATC Kecil Bandara Pulau I") at (-233.5, 665.5, y = 0.58m):
   * - Paved operational plinth with UI-Gold & Zebra apron curb
   * - Octagonal Precast-Concrete & UI-Gold ribbed shaft (y = 0.58 .. 8.35m) + exterior steel stair & 360° catwalk balcony
   * - Cantilevered 360° outward-tilted anti-glare glass ATC Control Cab (y = 8.40 .. 11.35m, night-lit v = -720.0)
   *   with interior radar/radio controller consoles & warm cab ceiling ring (v = -890.0)
   * - Overhanging roof deck with primary rotating-look radar bar, anemometer, VHF/ADS-B antennas, & red beacon (y = 13.45m)
   * Batched into corridorGroup for +0 extra Draw Calls!
   */
  static _buildPulauICompactATCTower() {
    const acc = SplineCorridorBuilder._createDualBufferAccumulator();
    const cx = TerrainDEMLoader.ATC_I_CX;
    const cz = TerrainDEMLoader.ATC_I_CZ;
    const baseY = Math.max(TerrainDEMLoader.ATC_I_BASE_Y, TerrainDEMLoader.sampleGroundHeight(cx, cz) + 0.04);

    const precastCream = [0.92, 0.90, 0.86];
    const uiGold = [0.93, 0.70, 0.09];
    const darkSteel = [0.15, 0.17, 0.21];
    const galvSteel = [0.64, 0.67, 0.72];
    const cabGlass = [0.09, 0.18, 0.28];
    const warmCeiling = [0.99, 0.93, 0.76];
    const radarWhite = [0.95, 0.96, 0.98];
    const beaconRed = [0.96, 0.08, 0.06];

    // 1. Operational Concrete Plinth & Sub-Terrain Skirt
    const plinthH = baseY + 0.22 - SplineCorridorBuilder.SKIRT_BOTTOM_Y;
    const plinth = new THREE.CylinderGeometry(3.85, 4.10, plinthH, 16, 1);
    plinth.translate(cx, SplineCorridorBuilder.SKIRT_BOTTOM_Y + plinthH * 0.5, cz);
    acc.appendV(plinth, precastCream);

    const collar = new THREE.CylinderGeometry(3.92, 3.92, 0.10, 16, 1);
    collar.translate(cx, baseY + 0.18, cz);
    acc.appendH(collar, uiGold);

    // 2. Octagonal Tower Shaft (y = baseY + 0.22 .. baseY + 7.80m) + 4 Vertical UI-Gold Accent Ribs + Slit Windows
    const shaftBotY = baseY + 0.22;
    const shaftTopY = baseY + 7.80;
    const shaftH = shaftTopY - shaftBotY;
    const shaft = new THREE.CylinderGeometry(1.88, 2.16, shaftH, 8, 1);
    shaft.translate(cx, shaftBotY + shaftH * 0.5, cz);
    acc.appendV(shaft, precastCream);

    for (let k = 0; k < 4; k++) {
      const ang = k * Math.PI * 0.5 + Math.PI * 0.25;
      const rx = cx + Math.cos(ang) * 2.02;
      const rz = cz + Math.sin(ang) * 2.02;
      const rib = new THREE.BoxGeometry(0.24, shaftH, 0.24);
      rib.rotateY(-ang);
      rib.translate(rx, shaftBotY + shaftH * 0.5, rz);
      acc.appendH(rib, uiGold);
    }

    // Ground-floor steel entry door & 2-level shaft equipment room windows (night-lit v = -720.0!)
    const door = new THREE.BoxGeometry(1.05, 2.15, 0.22);
    door.translate(cx, shaftBotY + 1.08, cz - 2.08);
    acc.appendH(door, darkSteel);
    const doorCanopy = new THREE.BoxGeometry(1.48, 0.10, 0.72);
    doorCanopy.translate(cx, shaftBotY + 2.24, cz - 2.32);
    acc.appendH(doorCanopy, uiGold);
    const doorLight = new THREE.BoxGeometry(0.68, 0.05, 0.18);
    doorLight.translate(cx, shaftBotY + 2.17, cz - 2.28);
    acc.appendH(doorLight, warmCeiling, [0.5, -890.0]);

    for (const wy of [shaftBotY + 3.4, shaftBotY + 5.7]) {
      for (const sgn of [-1, 1]) {
        const slitZ = new THREE.BoxGeometry(0.72, 1.35, 0.16);
        slitZ.translate(cx, wy, cz + sgn * 2.02);
        acc.appendH(slitZ, cabGlass, [0.5, -720.0]);

        const slitX = new THREE.BoxGeometry(0.16, 1.35, 0.72);
        slitX.translate(cx + sgn * 2.02, wy, cz);
        acc.appendH(slitX, cabGlass, [0.5, -720.0]);
      }
    }

    // 3. Cantilevered 360° Catwalk Observation Deck (y = shaftTopY) & Safety Handrails
    const catwalk = new THREE.CylinderGeometry(3.22, 2.35, 0.34, 8, 1);
    catwalk.translate(cx, shaftTopY + 0.17, cz);
    acc.appendV(catwalk, precastCream);

    const catwalkTrim = new THREE.CylinderGeometry(3.26, 3.26, 0.12, 8, 1);
    catwalkTrim.translate(cx, shaftTopY + 0.30, cz);
    acc.appendH(catwalkTrim, darkSteel);

    const deckY = shaftTopY + 0.34;
    for (let k = 0; k < 8; k++) {
      const a0 = (k / 8) * Math.PI * 2.0 + Math.PI / 8;
      const a1 = ((k + 1) / 8) * Math.PI * 2.0 + Math.PI / 8;
      const rRail = 3.12;
      const x0 = cx + Math.cos(a0) * rRail;
      const z0 = cz + Math.sin(a0) * rRail;
      const x1 = cx + Math.cos(a1) * rRail;
      const z1 = cz + Math.sin(a1) * rRail;
      acc.tubeH(x0, deckY, z0, x0, deckY + 1.02, z0, 0.028, 0.028, 6, galvSteel, true);
      acc.tubeH(x0, deckY + 1.02, z0, x1, deckY + 1.02, z1, 0.030, 0.030, 6, galvSteel, false);
      acc.tubeH(x0, deckY + 0.52, z0, x1, deckY + 0.52, z1, 0.022, 0.022, 6, galvSteel, false);
    }

    // 4. 360° Outward-Tilted Anti-Glare Octagonal ATC Control Cab (y = deckY .. deckY + 2.75m)
    const cabBotY = deckY;
    const cabTopY = deckY + 2.75;
    const cabWall = new THREE.CylinderGeometry(2.58, 2.42, 0.82, 8, 1);
    cabWall.translate(cx, cabBotY + 0.41, cz);
    acc.appendH(cabWall, darkSteel);

    // Outward-tilted panoramic ATC glazing (rBot = 2.52m -> rTop = 3.02m, UV sentinel v = -720.0!)
    const cabGlazing = new THREE.CylinderGeometry(3.00, 2.52, 1.92, 8, 1);
    cabGlazing.translate(cx, cabBotY + 0.80 + 0.96, cz);
    acc.appendH(cabGlazing, cabGlass, [0.5, -720.0]);

    // 8 Inclined Steel Corner Mullions around the ATC Control Cab
    for (let k = 0; k < 8; k++) {
      const ang = (k / 8) * Math.PI * 2.0 + Math.PI / 8;
      acc.tubeH(
        cx + Math.cos(ang) * 2.54,
        cabBotY + 0.78,
        cz + Math.sin(ang) * 2.54,
        cx + Math.cos(ang) * 3.04,
        cabTopY,
        cz + Math.sin(ang) * 3.04,
        0.055,
        0.055,
        6,
        darkSteel,
        false
      );
    }

    // Interior ATC Controller Console Ring & Warm Ceiling Halo (v = -890.0!)
    const consoleRing = new THREE.CylinderGeometry(2.15, 2.05, 0.92, 8, 1);
    consoleRing.translate(cx, cabBotY + 0.46, cz);
    acc.appendH(consoleRing, [0.22, 0.25, 0.30]);

    const cabCeilingGlow = new THREE.CylinderGeometry(2.65, 2.65, 0.08, 8, 1);
    cabCeilingGlow.translate(cx, cabTopY - 0.06, cz);
    acc.appendH(cabCeilingGlow, warmCeiling, [0.5, -890.0]);

    // 5. Overhanging Insulated Cab Roof, UI-Gold Fascia Band, Primary Radar Bar, Anemometer, VHF Antennas & Red Beacon
    const roofVisor = new THREE.CylinderGeometry(3.35, 3.55, 0.28, 8, 1);
    roofVisor.translate(cx, cabTopY + 0.14, cz);
    acc.appendH(roofVisor, darkSteel);

    const roofGoldBand = new THREE.CylinderGeometry(3.57, 3.57, 0.10, 8, 1);
    roofGoldBand.translate(cx, cabTopY + 0.16, cz);
    acc.appendH(roofGoldBand, uiGold);

    const roofCap = new THREE.ConeGeometry(3.38, 0.48, 8, 1);
    roofCap.translate(cx, cabTopY + 0.52, cz);
    acc.appendV(roofCap, precastCream);

    const roofApexY = cabTopY + 0.76;
    const ped = new THREE.CylinderGeometry(0.28, 0.34, 0.55, 10, 1);
    ped.translate(cx, roofApexY + 0.24, cz);
    acc.appendH(ped, darkSteel);

    const radarScanner = new THREE.BoxGeometry(2.35, 0.24, 0.34);
    radarScanner.rotateY(0.42);
    radarScanner.translate(cx, roofApexY + 0.62, cz);
    acc.appendH(radarScanner, radarWhite);

    // Central Mast + Flashing Red Aviation Obstruction Beacon (y > 5.2m night pulsing!) + 3 VHF/ADS-B Whip Antennas
    acc.tubeH(cx, roofApexY + 0.50, cz, cx, roofApexY + 1.95, cz, 0.045, 0.032, 8, galvSteel, true);
    const beacon = new THREE.SphereGeometry(0.16, 10, 8);
    beacon.translate(cx, roofApexY + 2.08, cz);
    acc.appendH(beacon, beaconRed);

    for (let a = 0; a < 3; a++) {
      const ang = (a / 3) * Math.PI * 2.0 + 0.3;
      const ax = cx + Math.cos(ang) * 2.15;
      const az = cz + Math.sin(ang) * 2.15;
      acc.tubeH(ax, cabTopY + 0.28, az, ax, cabTopY + 2.05, az, 0.026, 0.014, 6, galvSteel, true);
    }

    return acc.finalize();
  }

  /**
   * Builds the 4 K3 Emergency Assembly Points ("Titik Kumpul Keadaan Darurat") & 4 Modern UI Campus Coffee/Snack Kiosks ("Titik Kios"):
   * - 4 Titik Kumpul: 3.2m x 3.2m safety-green pad with white border, 4 inward white chevron arrows, & 2.45m double-sided K3 signpost
   * - 4 Titik Kios: Nusantara timber-slat & UI-Yellow modern campus kiosks with warm LED soffits (v = -890.0), night-lit glazing (v = -720.0),
   *   teak service counter, espresso machine, pastry display, menu board, & 2 shaded outdoor café umbrella tables per kiosk!
   * Batched into corridorGroup for +0 extra Draw Calls!
   */
  static _buildEmergencyMusterAndCampusKiosks() {
    const acc = SplineCorridorBuilder._createDualBufferAccumulator();

    const safetyGreen = [0.06, 0.55, 0.24];
    const crispWhite = [0.96, 0.97, 0.98];
    const galvSteel = [0.66, 0.69, 0.74];
    const darkSteel = [0.14, 0.16, 0.19];
    const uiGold = [0.93, 0.70, 0.09];
    const teakWood = [0.56, 0.34, 0.17];
    const teakDark = [0.38, 0.22, 0.11];
    const granitePlinth = [0.76, 0.75, 0.72];
    const kioskGlass = [0.10, 0.19, 0.29];
    const warmLuminaire = [0.99, 0.93, 0.75];
    const umbrellaCream = [0.95, 0.92, 0.84];

    // 1. 4 Titik Kumpul Keadaan Darurat (Emergency Assembly Points)
    for (const mp of SplineCorridorBuilder.MUSTER_POINTS) {
      const gy = Math.max(0.0, TerrainDEMLoader.sampleGroundHeight(mp.x, mp.z));
      const py = gy + 0.032;
      const cY = Math.cos(mp.yaw);
      const sY = Math.sin(mp.yaw);
      const toW = (lx, ly, lz) => [mp.x + lx * cY + lz * sY, ly, mp.z - lx * sY + lz * cY];
      const place = (geo, lx, ly, lz, rx = 0, ry = 0, rz = 0) => {
        if (rx !== 0) geo.rotateX(rx);
        if (ry !== 0) geo.rotateY(ry);
        if (rz !== 0) geo.rotateZ(rz);
        geo.rotateY(mp.yaw);
        const [wx, wy, wz] = toW(lx, ly, lz);
        geo.translate(wx, wy, wz);
        return geo;
      };

      acc.appendV(place(new THREE.BoxGeometry(3.4, 0.46, 3.4), 0, py - 0.24, 0), granitePlinth);
      acc.appendH(place(new THREE.BoxGeometry(3.24, 0.024, 3.24), 0, py - 0.004, 0), crispWhite);
      acc.appendH(place(new THREE.BoxGeometry(2.88, 0.028, 2.88), 0, py, 0), safetyGreen);

      // Central white muster square + 4 inward diagonal K3 assembly arrows on pad
      acc.appendH(place(new THREE.BoxGeometry(0.76, 0.032, 0.76), 0, py + 0.003, 0), crispWhite);
      for (let k = 0; k < 4; k++) {
        const ang = k * Math.PI * 0.5 + Math.PI * 0.25;
        const dist = 0.94;
        const lx = Math.sin(ang) * dist;
        const lz = Math.cos(ang) * dist;
        acc.appendH(place(new THREE.BoxGeometry(0.18, 0.032, 0.62), lx, py + 0.003, lz, 0, ang, 0), crispWhite);
        const head = new THREE.ConeGeometry(0.26, 0.34, 3, 1);
        head.rotateX(-Math.PI * 0.5);
        head.scale(1.0, 0.08, 1.0);
        acc.appendH(place(head, lx * 0.62, py + 0.016, lz * 0.62, 0, ang, 0), crispWhite);
      }

      // 2.45m Galvanized K3 Signpost + Double-Sided Green/White "TITIK KUMPUL" Signboard + Solar Night Lamp
      const [pxPost, , pzPost] = toW(0, 0, -1.32);
      acc.tubeH(pxPost, py, pzPost, pxPost, py + 2.45, pzPost, 0.045, 0.045, 8, galvSteel, true);
      acc.appendH(place(new THREE.BoxGeometry(0.88, 0.88, 0.07), 0, py + 2.02, -1.32), crispWhite);
      acc.appendH(place(new THREE.BoxGeometry(0.78, 0.78, 0.085), 0, py + 2.02, -1.32), safetyGreen);
      for (const sx of [-0.22, 0.22]) {
        for (const sy of [-0.22, 0.22]) {
          acc.appendH(place(new THREE.BoxGeometry(0.16, 0.06, 0.095), sx, py + 2.02 + sy, -1.32, 0, 0, sx * sy > 0 ? -0.78 : 0.78), crispWhite);
        }
      }
      acc.appendH(place(new THREE.BoxGeometry(0.32, 0.06, 0.16), 0, py + 2.50, -1.32), warmLuminaire, [0.5, -890.0]);
    }

    // 2. 4 Modern UI Campus Coffee & Snack Kiosks ("Titik Kios Kampus UI") + Shaded Outdoor Café Tables
    for (const ks of SplineCorridorBuilder.KIOSK_SPECS) {
      const gy = Math.max(0.0, TerrainDEMLoader.sampleGroundHeight(ks.x, ks.z));
      const deckY = gy + 0.12;
      const cY = Math.cos(ks.yaw);
      const sY = Math.sin(ks.yaw);
      const toW = (lx, ly, lz) => [ks.x + lx * cY + lz * sY, ly, ks.z - lx * sY + lz * cY];
      const place = (geo, lx, ly, lz, rx = 0, ry = 0, rz = 0) => {
        if (rx !== 0) geo.rotateX(rx);
        if (ry !== 0) geo.rotateY(ry);
        if (rz !== 0) geo.rotateZ(rz);
        geo.rotateY(ks.yaw);
        const [wx, wy, wz] = toW(lx, ly, lz);
        geo.translate(wx, wy, wz);
        return geo;
      };

      // Granite Terrace Plinth (5.8m x 4.2m, watertight sub-skirt)
      acc.appendV(place(new THREE.BoxGeometry(5.8, 0.92, 4.2), 0, deckY - 0.46, 0.25), granitePlinth);
      acc.appendH(place(new THREE.BoxGeometry(5.6, 0.04, 4.0), 0, deckY, 0.25), [0.46, 0.45, 0.43]);

      // Kiosk Cabin Core (3.6m wide x 2.3m deep x 2.65m tall, centered at lz = -0.45)
      acc.appendH(place(new THREE.BoxGeometry(3.52, 2.52, 2.18), 0, deckY + 1.28, -0.45), darkSteel);
      // Warm Teak Vertical Timber-Slat Cladding on Rear & Lower Front Counter Apron
      for (let sl = -7; sl <= 7; sl++) {
        acc.appendH(place(new THREE.BoxGeometry(0.12, 2.42, 0.06), sl * 0.22, deckY + 1.24, -1.56), teakWood);
        acc.appendH(place(new THREE.BoxGeometry(0.12, 0.96, 0.06), sl * 0.22, deckY + 0.50, 0.66), teakWood);
      }
      for (let sl = -4; sl <= 4; sl++) {
        for (const sgn of [-1, 1]) {
          acc.appendH(place(new THREE.BoxGeometry(0.06, 2.42, 0.12), sgn * 1.78, deckY + 1.24, -0.45 + sl * 0.22), teakWood);
        }
      }

      // Open Front Serving Window + Night-Lit Interior Glazing (v = -720.0), Teak Bar Counter, Espresso Machine & Pastry Case
      acc.appendH(place(new THREE.BoxGeometry(3.18, 1.32, 0.14), 0, deckY + 1.72, 0.58), kioskGlass, [0.5, -720.0]);
      acc.appendH(place(new THREE.BoxGeometry(3.48, 0.10, 0.56), 0, deckY + 1.02, 0.78), teakDark);
      // Espresso Machine + Grinder + Acrylic Pastry Showcase
      acc.appendH(place(new THREE.BoxGeometry(0.68, 0.46, 0.42), -0.88, deckY + 1.30, 0.72), galvSteel);
      acc.appendH(place(new THREE.BoxGeometry(0.84, 0.44, 0.38), 0.78, deckY + 1.29, 0.74), kioskGlass, [0.5, -720.0]);
      // Illuminated Overhead Menu Board (v = -890.0)
      acc.appendH(place(new THREE.BoxGeometry(1.95, 0.34, 0.08), 0, deckY + 2.22, 0.64), warmLuminaire, [0.5, -890.0]);

      // Overhanging Pitched Tropical Canopy Roof + UI-Golden-Yellow Header Fascia + Warm Photocell Downlight Strip (v = -890.0)
      acc.appendH(place(new THREE.BoxGeometry(4.24, 0.16, 3.45), 0, deckY + 2.68, -0.10, 0.07, 0, 0), darkSteel);
      acc.appendH(place(new THREE.BoxGeometry(4.28, 0.26, 0.12), 0, deckY + 2.74, 1.62), uiGold);
      acc.appendH(place(new THREE.BoxGeometry(3.35, 0.05, 0.24), 0, deckY + 2.57, 1.18), warmLuminaire, [0.5, -890.0]);

      // 2 Shaded Outdoor Café Umbrella Tables & Stools flanking the Kiosk Forecourt (lx = -2.15 & +2.15, lz = +1.15)
      for (const txOff of [-2.15, 2.15]) {
        const [wxT, , wzT] = toW(txOff, 0, 1.12);
        acc.tubeH(wxT, deckY + 0.02, wzT, wxT, deckY + 2.28, wzT, 0.035, 0.032, 8, darkSteel, true);
        acc.appendH(place(new THREE.CylinderGeometry(0.56, 0.56, 0.05, 14, 1), txOff, deckY + 0.76, 1.12), teakWood);
        const parasol = new THREE.ConeGeometry(1.18, 0.44, 8, 1);
        acc.appendH(
          place(parasol, txOff, deckY + 2.38, 1.12),
          txOff < 0 ? uiGold : umbrellaCream
        );
        for (const stAng of [0.4, 2.4, 4.4]) {
          const sxS = txOff + Math.cos(stAng) * 0.74;
          const szS = 1.12 + Math.sin(stAng) * 0.74;
          const [wxS, , wzS] = toW(sxS, 0, szS);
          acc.tubeH(wxS, deckY + 0.02, wzS, wxS, deckY + 0.45, wzS, 0.028, 0.034, 6, darkSteel, false);
          acc.appendH(place(new THREE.CylinderGeometry(0.18, 0.18, 0.05, 10, 1), sxS, deckY + 0.47, szS), teakDark);
        }
      }
    }

    return acc.finalize();
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

    // 1. Lapangan Rotunda UI Circular Asphalt Boulevard (288 segments, with smooth curb-cuts across junction throats!)
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
        zebraFreq: 0.135,
        curbCutZones: [
          { x: rCx - 33.95, z: rCz, r: 5.05, fade: 1.5 },
          { x: -20.6, z: 135.2, r: 2.75, fade: 1.2 },
          { x: rCx, z: rCz - 33.95, r: 2.4, fade: 1.0 }
        ]
      })
    );

    // 2. Architectural Dancing Fountain Reflecting Basin & Flowering Topiary Garland at Center of Lapangan Rotunda UI
    addPair(
      'LapanganRotundaUI_BotanicalCenter',
      SplineCorridorBuilder._buildRotundaBotanicalCenterpiece(rCx, rCz)
    );

    // 3. West Campus Helipad Boulevard Branching Flush from Lapangan Rotunda UI Outer Edge into VIP Helipad Ramp!
    const westRoadSpline = SplineCorridorBuilder._createHighPrecisionSpline(
      [
        new THREE.Vector3(rCx - 33.95, 0.032, rCz),
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
        zebraFreq: 0.13,
        curbCutZones: [
          { x: rCx - 33.95, z: rCz, r: 1.8, fade: 1.2 },
          { x: -62.5, z: 101.6, r: 6.25, fade: 1.5 },
          { x: -81.6, z: rCz, r: 2.2, fade: 1.2 }
        ]
      })
    );

    // 4. West Courtyard Parking Lot & Access Lane (x = -62.5m, trimmed flush to West Boulevard south curb!)
    const westParkingSpline = SplineCorridorBuilder._createHighPrecisionSpline(
      [
        new THREE.Vector3(-62.5, 0.03, 101.45),
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
        zebraCurbs: false,
        curbCutZones: [
          { x: -62.5, z: 101.45, r: 1.8, fade: 1.2 },
          { x: -68.0, z: 75.0, r: 4.5, fade: 1.5 }
        ]
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

    // 11. Lapangan Rotunda UI <-> Jembatan Teksas UI West Forecourt Promenade Connector (trimmed flush to Rotunda outer curb!)
    const teksasConnectorSpline = SplineCorridorBuilder._createHighPrecisionSpline(
      [
        new THREE.Vector3(-20.65, 0.035, 135.30),
        new THREE.Vector3(-24.6, 0.036, 140.8),
        new THREE.Vector3(TerrainDEMLoader.TEKSAS_BRIDGE_X, 0.038, 146.8)
      ],
      false
    );
    addPair(
      'RotundaEastTeksasConnector',
      SplineCorridorBuilder._loftSeparatedCorridor(teksasConnectorSpline, {
        segments: 36,
        halfWidth: 2.35,
        curbWidth: 0.34,
        curbRise: 0.11,
        deckColor: [0.46, 0.45, 0.42],
        zebraCurbs: false,
        curbCutZones: [
          { x: -20.65, z: 135.30, r: 1.8, fade: 1.1 },
          { x: TerrainDEMLoader.TEKSAS_BRIDGE_X, z: 146.8, r: 2.2, fade: 1.1 }
        ]
      })
    );

    // 12. High-Accuracy Architectural Digital Twin of "Jembatan Teksas UI" (7-Segment BTKC Arch, Pilon Lingga 25m,
    //     Pilon Yoni 15m, 30 Blue Wave Sunroof Canopies, Inclusive Ramps+Stairs, & 9.25m Mini-Yacht Clearance!)
    addPair(
      'PulauU_JembatanTeksasUI',
      SplineCorridorBuilder._buildJembatanTeksasUI()
    );

    // 13. Integrated Pulau U & Pulau I Multi-Loop Asphalt Road Network (3D Terrain-Graded Cambered Carriageways,
    //     UI-Yellow Center Dashed Lines, White Edge Lines, W-Beam Hillside Guardrails, & Junction Plazas!)
    addPair(
      'KepulauanUI_RoadNetwork',
      SplineCorridorBuilder._buildPulauUAndPulauIRoadNetwork()
    );

    // 14. Southern Undersea Underpass Tunnel U <-> I (Open-Cut Retained Ramps, Twin Nusantara Portals & Tajug Ventilation
    //     Towers, 298m Immersed-Tube Subsea Bore at y = -8.20m, 16 Panoramic Ocean-Caustic Windows, & 24/7 LED Luminaires!)
    addPair(
      'KepulauanUI_UnderpassBawahLaut',
      SplineCorridorBuilder._buildUnderpassBawahLautUI()
    );

    // 15. Northern Ferry Crossing Terminals U <-> I ("Dermaga Penyeberangan Utara Pulau U & Pulau I": Twin-Tier Tajug
    //     Passenger Pavilions, Pile-Supported Trestles & T-Head Berths, Boarding Stages, Fenders, Bollards & Beacons!)
    addPair(
      'KepulauanUI_DermagaPenyeberanganUtara',
      SplineCorridorBuilder._buildDermagaPenyeberanganUtaraUI()
    );

    // 16. Outdoor Campus Monopole Widescreen LED Videotron ("Videotron Kampus UI" — 12.80m x 7.20m SMD P8 Display,
    //     Octagonal Precast Plinth, Monopole + Rear Novastar Box + Caged Ladder, Space-Frame K-Truss & 2-Tier Catwalks!)
    addPair(
      'KampusUI_MonopoleVideotron',
      SplineCorridorBuilder._buildKampusUIVideotron()
    );

    // 17. Full Multi-Lane Campus Road Markings (2-Lane Rotunda Dividers, Crosswalks, Arrows, & Flush Junction Aprons)
    addPair(
      'KampusUI_RoadLaneMarkings',
      SplineCorridorBuilder._buildCampusRoadLaneMarkings()
    );

    // 18. Kompleks Kantor DPPD, Perpustakaan Lama (ITTC), Science Techno Park (STP-UI 8-Lantai), & ILRC/DRPM on Pulau U
    //     (Facing Gedung Rektorat UI across Teluk U — +0 extra Draw Calls!)
    addPair(
      'PulauU_DPPD_ILRC_STP_Complex',
      DPPDComplexBuilder.buildForCorridor()
    );

    // 19. Dedicated EPDM Campus Jogging Tracks (Rotunda & Teluk U Oval) + Sudut Calisthenics Street-Workout Park
    addPair(
      'KampusUI_JoggingTrackAndCalisthenics',
      SplineCorridorBuilder._buildCampusJoggingTrackAndCalisthenics()
    );

    // 20. Compact Air Traffic Control Tower ("ATC Kecil Bandara Pulau I") overlooking RWY 36/18 & West Apron
    addPair(
      'PulauI_CompactATCTower',
      SplineCorridorBuilder._buildPulauICompactATCTower()
    );

    // 21. 4 K3 Emergency Assembly Points ("Titik Kumpul Keadaan Darurat") & 4 Modern UI Campus Coffee/Snack Kiosks ("Titik Kios")
    addPair(
      'KampusUI_MusterAndKiosks',
      SplineCorridorBuilder._buildEmergencyMusterAndCampusKiosks()
    );

    // Hukum Emas 12: Batch all decks/roofs/topiary/teak/helipad/runway/bridge/roads/underpass/piers/videotron/DPPD/ATC/kiosks into Draw Call #1 & all curbs/columns/bollards into Draw Call #2!
    NexusBatchOptimizer.optimizeHierarchy(corridorGroup);
    corridorGroup.updateMatrixWorld(true);
    return corridorGroup;
  }

  /**
   * Single Source of Truth (SSoST) World Solid-Body Collider Registry (v4.0 Strict Non-Penetration Edition).
   * Constructs pre-indexed, zero-GC 3D solid volumes (BOX, CYL, OBB, ISLAND_PERIMETER) for:
   * - Gedung Rektorat UI (Main Pagoda Tower, Tajug Spire, 44 Colonnade Pillars, 8 Planters, Enclosed Annex Wings,
   *   and true open-air columns + walls + overhead roofs for Selasar Agung, Flag Cloister, Gatehouse, Pendopo & North Pavilion!)
   * - Lapangan Rotunda Botanical Centerpiece, 62 SE Covered Walkway Columns + 12 Roof Bays, 14 Park Benches & 24 Bollards
   * - VIP Helipad Pavilion (4 Columns, 2 Benches, Limas Roof) + Helipad & Pulau I Airfield Windsock Masts
   * - Jembatan Teksas UI Shore Abutments, Pilon Lingga (25m Sail), Pilon Yoni (15m Hip Arch) & 2 East Overlook Benches
   * - Underpass Bawah Laut Selatan U–I Two-Regime Topological Config + Portal Headwalls, Wing Piers & 4 Tajug Vent Towers
   * - Dermaga Penyeberangan Utara Pulau U & Pulau I (16 Pavilion Columns, 8 Benches, 2 Tajug Roofs, T-Head Marine Berths & Dolphins)
   * - All ~198 Spawned Campus/Forest Tree Trunks & All 64 Fluted Nusantara Streetlamps!
   */
  static buildWorldSolidColliders(biomeGroup = null) {
    const staticColliders = [];
    const marineColliders = [];

    const pushStatic = (col) => {
      staticColliders.push(PhysicsKernel.prepareSolidCollider(col));
    };
    const pushMarine = (col) => {
      marineColliders.push(PhysicsKernel.prepareSolidCollider(col));
    };

    // 0. Pedestrian Shoreline & Bridge Railing Containment Rule (active in WALK_SIVITAS mode)
    staticColliders.push({
      type: 'ISLAND_PERIMETER',
      coastDistFn: TerrainDEMLoader.getIslandCoastDistance,
      margin: 1.15,
      bridgeDeckFn: TerrainDEMLoader.sampleTeksasBridgeDeckY,
      bridgeCenterX: TerrainDEMLoader.TEKSAS_BRIDGE_X,
      bridgeWalkHalfW: 2.25
    });

    // =========================================================================
    // 1. GEDUNG REKTORAT UI — EXACT ARCHITECTURAL SOLID VOLUMES & OPEN COLONNADES
    // =========================================================================
    // 1A. Main Cruciform Pagoda Tower Lobby Core, 6-Tier Pagoda Upper Eaves, & Steep Tajug Spire
    pushStatic({ id: 'Rektorat_TowerCore', type: 'BOX', minX: -14.6, maxX: 14.6, minY: 0.0, maxY: 33.2, minZ: -14.6, maxZ: 14.6 });
    pushStatic({ id: 'Rektorat_TowerPagodaUpper', type: 'BOX', minX: -18.6, maxX: 18.6, minY: 5.55, maxY: 30.9, minZ: -18.6, maxZ: 18.6 });
    pushStatic({ id: 'Rektorat_TajugSpire', type: 'CYL', x: 0.0, z: 0.0, r: 13.5, minY: 32.5, maxY: 50.2 });

    // 44 Ground-Floor Sculpted Colonnade Pillars & 8 Recessed Base Planter Troughs around Main Tower
    const quadrants = [
      { id: 'NE', sx: 1, sz: -1 },
      { id: 'NW', sx: -1, sz: -1 },
      { id: 'SW', sx: -1, sz: 1 },
      { id: 'SE', sx: 1, sz: 1 }
    ];
    const wingPillarCoords = [2.85, 5.33, 7.81, 10.29, 12.77, 15.25];
    const wingOutCoord = 15.42;
    for (const q of quadrants) {
      for (let i = 0; i < wingPillarCoords.length; i++) {
        const c = wingPillarCoords[i];
        const pxZ = q.sx * c;
        const pzZ = q.sz * wingOutCoord;
        pushStatic({
          id: `Rektorat_TowerColZ_${q.id}_${i}`,
          type: 'BOX',
          minX: pxZ - 0.36,
          maxX: pxZ + 0.36,
          minY: 0.0,
          maxY: 5.68,
          minZ: pzZ - 0.38,
          maxZ: pzZ + 0.38
        });
        if (i < wingPillarCoords.length - 1) {
          const pxX = q.sx * wingOutCoord;
          const pzX = q.sz * c;
          pushStatic({
            id: `Rektorat_TowerColX_${q.id}_${i}`,
            type: 'BOX',
            minX: pxX - 0.38,
            maxX: pxX + 0.38,
            minY: 0.0,
            maxY: 5.68,
            minZ: pzX - 0.36,
            maxZ: pzX + 0.36
          });
        }
      }
      const plZCx = q.sx * 9.05;
      const plZCz = q.sz * 15.15;
      pushStatic({
        id: `Rektorat_PlanterZ_${q.id}`,
        type: 'BOX',
        minX: plZCx - 6.12,
        maxX: plZCx + 6.12,
        minY: 0.0,
        maxY: 1.02,
        minZ: plZCz - 0.45,
        maxZ: plZCz + 0.45
      });
      const plXCx = q.sx * 15.15;
      const plXCz = q.sz * 9.05;
      pushStatic({
        id: `Rektorat_PlanterX_${q.id}`,
        type: 'BOX',
        minX: plXCx - 0.45,
        maxX: plXCx + 0.45,
        minY: 0.0,
        maxY: 1.02,
        minZ: plXCz - 6.12,
        maxZ: plXCz + 6.12
      });
    }

    // 1B. Enclosed West & East Lakeside Courtyard Annex Buildings + Overhead Tower Bridges
    pushStatic({ id: 'WestAnnex_NorthFlatBlock', type: 'BOX', minX: -43.5, maxX: -21.0, minY: 0.0, maxY: 6.35, minZ: -18.8, maxZ: -3.6 });
    pushStatic({ id: 'WestAnnex_SouthFlatBlock', type: 'BOX', minX: -43.5, maxX: -21.0, minY: 0.0, maxY: 6.35, minZ: 2.6, maxZ: 16.2 });
    pushStatic({ id: 'WestAnnex_PyramidalHallBody', type: 'BOX', minX: -52.8, maxX: -34.7, minY: 0.0, maxY: 5.85, minZ: -12.3, maxZ: 9.7 });
    pushStatic({ id: 'WestAnnex_PyramidalHallRoof', type: 'BOX', minX: -54.0, maxX: -33.5, minY: 5.80, maxY: 12.30, minZ: -13.5, maxZ: 10.9 });
    pushStatic({ id: 'WestAnnex_TowerBridgeRoof', type: 'BOX', minX: -21.6, maxX: -15.2, minY: 4.35, maxY: 6.15, minZ: -3.1, maxZ: 2.1 });

    pushStatic({ id: 'EastAnnex_NorthFlatBlock', type: 'BOX', minX: 21.0, maxX: 43.5, minY: 0.0, maxY: 6.35, minZ: -21.8, maxZ: -6.6 });
    pushStatic({ id: 'EastAnnex_SouthFlatBlock', type: 'BOX', minX: 21.0, maxX: 43.5, minY: 0.0, maxY: 6.35, minZ: -0.4, maxZ: 13.2 });
    pushStatic({ id: 'EastAnnex_LakesideHallBody', type: 'BOX', minX: 34.7, maxX: 52.8, minY: 0.0, maxY: 5.85, minZ: -15.3, maxZ: 6.7 });
    pushStatic({ id: 'EastAnnex_LakesideHallRoof', type: 'BOX', minX: 33.5, maxX: 54.0, minY: 5.80, maxY: 12.30, minZ: -16.5, maxZ: 7.9 });
    pushStatic({ id: 'EastAnnex_TowerBridgeRoof', type: 'BOX', minX: 15.2, maxX: 21.6, minY: 4.35, maxY: 6.15, minZ: -6.1, maxZ: -0.9 });

    // 1C. Selasar Agung (Open-Air Ceremonial Breezeway: 12 Columns + High Gable Roof)
    for (const pz of [17.5, 21.5, 25.5, 29.5, 33.5, 37.5]) {
      for (const sx of [-1, 1]) {
        const px = -2.5 + sx * 3.35;
        pushStatic({
          id: `SelasarAgung_Col_${pz}_${sx}`,
          type: 'BOX',
          minX: px - 0.29,
          maxX: px + 0.29,
          minY: 0.0,
          maxY: 4.90,
          minZ: pz - 0.29,
          maxZ: pz + 0.29
        });
      }
    }
    pushStatic({ id: 'SelasarAgung_Roof', type: 'BOX', minX: -6.7, maxX: 1.7, minY: 4.82, maxY: 9.40, minZ: 15.5, maxZ: 39.5 });

    // 1D. 4-Sided Rectangular Flag Cloister (2 Front Perimeter Walls + 26 Columns + 4 Gallery Roofs)
    pushStatic({ id: 'Cloister_FrontWallL', type: 'BOX', minX: -20.33, maxX: -8.67, minY: 0.0, maxY: 2.75, minZ: 62.31, maxZ: 62.89 });
    pushStatic({ id: 'Cloister_FrontWallR', type: 'BOX', minX: 3.67, maxX: 15.33, minY: 0.0, maxY: 2.75, minZ: 62.31, maxZ: 62.89 });
    for (const px of [-15.8, -11.4, -7.0, -2.5, 2.0, 6.4, 10.8]) {
      for (const pz of [43.3, 59.1]) {
        pushStatic({
          id: `Cloister_NSCol_${px}_${pz}`,
          type: 'BOX',
          minX: px - 0.23,
          maxX: px + 0.23,
          minY: 0.0,
          maxY: 3.60,
          minZ: pz - 0.23,
          maxZ: pz + 0.23
        });
      }
    }
    for (const pz of [46.8, 51.2, 55.6]) {
      for (const px of [-20.0, -15.8, 10.8, 15.0]) {
        pushStatic({
          id: `Cloister_EWCol_${px}_${pz}`,
          type: 'BOX',
          minX: px - 0.23,
          maxX: px + 0.23,
          minY: 0.0,
          maxY: 3.60,
          minZ: pz - 0.23,
          maxZ: pz + 0.23
        });
      }
    }
    pushStatic({ id: 'Cloister_NorthRoof', type: 'BOX', minX: -20.5, maxX: 15.5, minY: 3.52, maxY: 5.58, minZ: 39.0, maxZ: 43.8 });
    pushStatic({ id: 'Cloister_SouthRoof', type: 'BOX', minX: -20.5, maxX: 15.5, minY: 3.52, maxY: 5.58, minZ: 58.7, maxZ: 63.5 });
    pushStatic({ id: 'Cloister_WestRoof', type: 'BOX', minX: -20.5, maxX: -15.7, minY: 3.52, maxY: 5.58, minZ: 39.0, maxZ: 63.5 });
    pushStatic({ id: 'Cloister_EastRoof', type: 'BOX', minX: 10.7, maxX: 15.5, minY: 3.52, maxY: 5.58, minZ: 39.0, maxZ: 63.5 });

    // 1E. Front Rotunda Gatehouse Pavilion (9 Columns + Hipped Roof)
    const gateCols = [
      [-7.4, 60.8], [-2.5, 60.8], [2.4, 60.8],
      [-7.4, 65.5],               [2.4, 65.5],
      [-7.4, 70.2], [-4.4, 70.2], [-0.6, 70.2], [2.4, 70.2]
    ];
    for (let i = 0; i < gateCols.length; i++) {
      const [gx, gz] = gateCols[i];
      pushStatic({
        id: `FrontGatehouse_Col_${i}`,
        type: 'BOX',
        minX: gx - 0.28,
        maxX: gx + 0.28,
        minY: 0.0,
        maxY: 3.82,
        minZ: gz - 0.28,
        maxZ: gz + 0.28
      });
    }
    pushStatic({ id: 'FrontGatehouse_Roof', type: 'BOX', minX: -9.0, maxX: 4.0, minY: 3.72, maxY: 7.50, minZ: 59.0, maxZ: 72.0 });

    // 1F. 45° SE Diagonal Breezeway, Octagonal Skylit Pendopo, & North Standalone Pavilion
    for (let s = 0; s < 4; s++) {
      const dist = -6.5 + s * 4.3;
      for (const side of [-1.75, 1.75]) {
        const cos45 = Math.SQRT1_2;
        const wx = 19.2 + side * cos45 + dist * cos45;
        const wz = 19.2 - side * cos45 + dist * cos45;
        pushStatic({ id: `SE45Breezeway_Col_${s}_${side}`, type: 'CYL', x: wx, z: wz, r: 0.27, minY: 0.0, maxY: 3.68 });
      }
    }
    pushStatic({
      id: 'SE45Breezeway_Roof',
      type: 'OBB',
      cx: 19.2,
      cz: 19.2,
      halfW: 2.35,
      halfL: 8.75,
      yaw: Math.PI * 0.25,
      minY: 3.62,
      maxY: 5.58
    });

    for (let k = 0; k < 8; k++) {
      const ang = (k + 0.5) * (Math.PI * 0.25);
      pushStatic({
        id: `OctSkylitPendopo_Col_${k}`,
        type: 'CYL',
        x: 29.5 + 6.55 * Math.cos(ang),
        z: 29.5 + 6.55 * Math.sin(ang),
        r: 0.31,
        minY: 0.0,
        maxY: 3.86
      });
    }
    pushStatic({ id: 'OctSkylitPendopo_Roof', type: 'CYL', x: 29.5, z: 29.5, r: 8.25, minY: 3.72, maxY: 8.20 });

    for (const lx of [-6.8, 0.0, 6.8]) {
      for (const lz of [-6.8, 0.0, 6.8]) {
        if (lx === 0.0 && lz === 0.0) continue;
        pushStatic({
          id: `NorthPavilion_Col_${lx}_${lz}`,
          type: 'BOX',
          minX: 5.0 + lx - 0.25,
          maxX: 5.0 + lx + 0.25,
          minY: 0.0,
          maxY: 3.75,
          minZ: -43.5 + lz - 0.25,
          maxZ: -43.5 + lz + 0.25
        });
      }
    }
    pushStatic({ id: 'NorthPavilion_Roof', type: 'BOX', minX: -3.5, maxX: 13.5, minY: 3.72, maxY: 8.60, minZ: -52.0, maxZ: -35.0 });

    // =========================================================================
    // 2. ROTUNDA PLANTER, SE COVERED WALKWAY, 14 PARK BENCHES, & 24 BOLLARDS
    // =========================================================================
    pushStatic({
      id: 'RotundaBotanicalCenterpiece',
      type: 'CYL',
      x: SplineCorridorBuilder.ROTUNDA_CX,
      z: SplineCorridorBuilder.ROTUNDA_CZ,
      r: 5.54,
      minY: -1.0,
      maxY: 0.90
    });

    // Outdoor Campus Monopole Widescreen LED Videotron Solid-Body Colliders (Pedestal, Monopole/ControlBox, & Upper Cabinet/Catwalk)
    {
      const vt = SplineCorridorBuilder.VIDEOTRON_SPEC;
      const vtFwdX = Math.sin(vt.yaw);
      const vtFwdZ = Math.cos(vt.yaw);
      pushStatic({
        id: 'Videotron_Pedestal',
        type: 'CYL',
        x: vt.cx,
        z: vt.cz,
        r: 1.36,
        minY: -1.0,
        maxY: 1.02
      });
      pushStatic({
        id: 'Videotron_MonopoleAndControlBox',
        type: 'OBB',
        cx: vt.cx - vtFwdX * 0.22,
        cz: vt.cz - vtFwdZ * 0.22,
        halfW: 0.78,
        halfL: 0.90,
        yaw: vt.yaw,
        minY: 0.92,
        maxY: 5.38
      });
      pushStatic({
        id: 'Videotron_CabinetAndCatwalk',
        type: 'OBB',
        cx: vt.cx - vtFwdX * 0.10,
        cz: vt.cz - vtFwdZ * 0.10,
        halfW: 6.90,
        halfL: 0.92,
        yaw: vt.yaw,
        minY: 5.34,
        maxY: 14.02
      });
    }

    const seWalkwaySpline = SplineCorridorBuilder._createHighPrecisionSpline(
      [
        new THREE.Vector3(34.2, 0.04, 35.5),
        new THREE.Vector3(37.4, 0.04, 58.0),
        new THREE.Vector3(40.6, 0.04, 92.0),
        new THREE.Vector3(43.8, 0.04, 128.0)
      ],
      false
    );
    const pTmp = new THREE.Vector3();
    const tTmp = new THREE.Vector3();
    const sTmp = new THREE.Vector3();
    const pTmpB = new THREE.Vector3();
    for (let i = 0; i <= 120; i += 4) {
      SplineCorridorBuilder._sampleCurveFrame(seWalkwaySpline, i / 120, pTmp, tTmp, sTmp);
      for (const sgn of [-1.55, 1.55]) {
        pushStatic({
          id: `SEWalkwayCol_${i}_${sgn > 0 ? 'R' : 'L'}`,
          type: 'CYL',
          x: pTmp.x + sTmp.x * sgn,
          z: pTmp.z + sTmp.z * sgn,
          r: 0.24,
          minY: -0.3,
          maxY: 3.45
        });
      }
    }
    for (let i = 0; i < 120; i += 10) {
      SplineCorridorBuilder._sampleCurveFrame(seWalkwaySpline, i / 120, pTmp, tTmp, sTmp);
      SplineCorridorBuilder._sampleCurveFrame(seWalkwaySpline, (i + 10) / 120, pTmpB, tTmp, sTmp);
      const dx = pTmpB.x - pTmp.x;
      const dz = pTmpB.z - pTmp.z;
      const segLen = Math.hypot(dx, dz);
      pushStatic({
        id: `SEWalkwayRoof_${i}`,
        type: 'OBB',
        cx: 0.5 * (pTmp.x + pTmpB.x),
        cz: 0.5 * (pTmp.z + pTmpB.z),
        halfW: 2.18,
        halfL: segLen * 0.5 + 0.15,
        yaw: Math.atan2(dx, dz),
        minY: 3.30,
        maxY: 4.82
      });
    }

    // 7 Lakeside Promenade Benches & 7 Rotunda Outer Apron Benches
    const lakesideBenchZs = [-48.0, -35.0, -24.0, 17.5, 27.5, 37.5, 47.5];
    for (let i = 0; i < lakesideBenchZs.length; i++) {
      const lz = lakesideBenchZs[i];
      const px = TerrainDEMLoader.getShorelinePromenadeX(lz) - 2.62;
      pushStatic({
        id: `LakesideBench_${lz}`,
        type: 'OBB',
        cx: px,
        cz: lz,
        halfW: 0.84,
        halfL: 0.30,
        yaw: Math.PI * 0.5,
        minY: 0.0,
        maxY: 0.88,
        isBench: true
      });
    }
    for (let i = 0; i < 7; i++) {
      const th = 0.42 + (i / 6) * (Math.PI - 0.84);
      const bx = -2.5 + Math.cos(th) * 35.4;
      const bz = 106.5 + Math.sin(th) * 35.4;
      const yawInward = Math.atan2(-2.5 - bx, 106.5 - bz);
      pushStatic({
        id: `RotundaBench_${i}`,
        type: 'OBB',
        cx: bx,
        cz: bz,
        halfW: 0.84,
        halfL: 0.30,
        yaw: yawInward,
        minY: 0.0,
        maxY: 0.88,
        isBench: true
      });
    }

    // 24 Stone Pedestrian Bollards
    for (let i = 0; i < 16; i++) {
      const th = (i / 16) * Math.PI * 2.0 + 0.12;
      pushStatic({
        id: `RotundaBollard_${i}`,
        type: 'CYL',
        x: -2.5 + Math.cos(th) * 34.65,
        z: 106.5 + Math.sin(th) * 34.65,
        r: 0.13,
        minY: 0.0,
        maxY: 0.84
      });
    }
    for (const sx of [-10.2, -8.2, 3.2, 5.2]) {
      pushStatic({ id: `GateBollard_${sx}`, type: 'CYL', x: sx, z: 71.85, r: 0.13, minY: 0.0, maxY: 0.84 });
    }
    for (const lz of [-18.0, -6.0, 6.0, 14.0]) {
      pushStatic({
        id: `LakeBollard_${lz}`,
        type: 'CYL',
        x: TerrainDEMLoader.getShorelinePromenadeX(lz) - 0.35,
        z: lz,
        r: 0.13,
        minY: 0.0,
        maxY: 0.84
      });
    }

    // =========================================================================
    // 3. VIP HELIPAD PAVILION, WINDSOCK MASTS, & JEMBATAN TEKSAS UI SOLIDS
    // =========================================================================
    for (const cxOff of [-2.05, 2.05]) {
      for (const czOff of [-2.05, 2.05]) {
        pushStatic({
          id: `HelipadPavCol_${cxOff}_${czOff}`,
          type: 'CYL',
          x: -84.8 + cxOff,
          z: 118.2 + czOff,
          r: 0.24,
          minY: 0.0,
          maxY: 3.16
        });
      }
    }
    for (const bzOff of [-1.15, 1.15]) {
      pushStatic({
        id: `HelipadPavBench_${bzOff}`,
        type: 'BOX',
        minX: -84.72,
        maxX: -84.18,
        minY: 0.25,
        maxY: 1.08,
        minZ: 118.2 + bzOff - 0.77,
        maxZ: 118.2 + bzOff + 0.77,
        isBench: true
      });
    }
    pushStatic({ id: 'HelipadPavRoof', type: 'BOX', minX: -87.9, maxX: -81.7, minY: 3.10, maxY: 5.55, minZ: 115.1, maxZ: 121.3 });
    pushStatic({ id: 'HelipadWindsock', type: 'CYL', x: -87.2, z: 92.8, r: 0.56, minY: 0.0, maxY: 6.7 });
    pushStatic({ id: 'PulauIWindsock', type: 'CYL', x: -214.0, z: 686.5, r: 0.58, minY: 0.0, maxY: 6.8 });

    // Jembatan Teksas UI Shore Abutments, Pilon Lingga, Pilon Yoni, & East Overlook Benches
    const bx = TerrainDEMLoader.TEKSAS_BRIDGE_X; // -28.0
    const zW = TerrainDEMLoader.TEKSAS_WEST_PORTAL_Z; // 171.65
    const zE = TerrainDEMLoader.TEKSAS_EAST_PORTAL_Z; // 319.65
    pushStatic({ id: 'Teksas_WestAbutment', type: 'BOX', minX: bx - 3.4, maxX: bx + 3.4, minY: -3.5, maxY: 2.95, minZ: zW - 2.15, maxZ: zW + 1.45 });
    pushStatic({ id: 'Teksas_EastAbutment', type: 'BOX', minX: bx - 3.4, maxX: bx + 3.4, minY: -3.5, maxY: 2.95, minZ: zE - 1.45, maxZ: zE + 2.15 });
    for (const sx of [-1, 1]) {
      pushStatic({ id: `Teksas_LinggaLegLo_${sx}`, type: 'CYL', x: bx + sx * 3.82, z: zW, r: 0.55, minY: -2.0, maxY: 5.8 });
      pushStatic({ id: `Teksas_LinggaLegHi_${sx}`, type: 'CYL', x: bx + sx * 3.12, z: zW - 0.06, r: 0.44, minY: 5.8, maxY: 9.2 });
      pushStatic({ id: `Teksas_YoniLegLo_${sx}`, type: 'CYL', x: bx + sx * 3.82, z: zE, r: 0.52, minY: -2.0, maxY: 6.2 });
      pushStatic({ id: `Teksas_YoniHipHi_${sx}`, type: 'CYL', x: bx + sx * 4.35, z: zE + 0.45, r: 0.48, minY: 6.2, maxY: 12.8 });
    }
    pushStatic({ id: 'Teksas_LinggaSailMast', type: 'BOX', minX: bx - 0.72, maxX: bx + 1.05, minY: 10.65, maxY: 26.25, minZ: zW - 3.2, maxZ: zW + 2.4 });
    pushStatic({ id: 'Teksas_YoniCrownArch', type: 'BOX', minX: bx - 3.0, maxX: bx + 3.0, minY: 12.8, maxY: 15.55, minZ: zE - 0.55, maxZ: zE + 1.15 });
    for (const sx of [-4.2, 4.2]) {
      pushStatic({
        id: `Teksas_EastOverlookBench_${sx}`,
        type: 'BOX',
        minX: bx + sx - 0.82,
        maxX: bx + sx + 0.82,
        minY: 0.30,
        maxY: 1.24,
        minZ: 347.48,
        maxZ: 348.08,
        isBench: true
      });
    }

    // =========================================================================
    // 4. UNDERPASS BAWAH LAUT SELATAN U–I PORTALS, VENT TOWERS & TOPOLOGICAL CONFIG
    // =========================================================================
    const ux = TerrainDEMLoader.UNDERPASS_X; // -356.0
    const zWP = TerrainDEMLoader.UNDERPASS_WEST_PORTAL_Z; // 326.0
    const zEP = TerrainDEMLoader.UNDERPASS_EAST_PORTAL_Z; // 624.0
    const boreHW = TerrainDEMLoader.UNDERPASS_BORE_HALF_W; // 4.55
    const floorY = TerrainDEMLoader.UNDERPASS_TUNNEL_FLOOR_Y; // -8.20
    const clearH = TerrainDEMLoader.UNDERPASS_CLEAR_HEIGHT_M; // 4.15 -> crown -4.05

    for (const [tag, zPortal, outSignZ] of [
      ['West', zWP, -1.0],
      ['East', zEP, 1.0]
    ]) {
      const zCenter = zPortal - outSignZ * 3.55;
      const behindGroundY = Math.max(0.55, TerrainDEMLoader.sampleGroundHeight(ux, zPortal - outSignZ * 6.8));
      const deckTopY = behindGroundY + 0.75;
      // Portal headwall concrete lintel above tunnel arch crown (-4.05m .. deckTopY + 1.4m)
      pushStatic({
        id: `Underpass_${tag}PortalLintel`,
        type: 'BOX',
        minX: ux - 7.55,
        maxX: ux + 7.55,
        minY: floorY + clearH,
        maxY: deckTopY + 1.35,
        minZ: zCenter - 3.65,
        maxZ: zCenter + 3.65
      });
      // Left & Right full-height portal abutment wing piers
      for (const sx of [-1.0, 1.0]) {
        const px = ux + sx * (boreHW + 1.50);
        pushStatic({
          id: `Underpass_${tag}PortalPier_${sx}`,
          type: 'BOX',
          minX: px - 1.50,
          maxX: px + 1.50,
          minY: floorY - 1.2,
          maxY: deckTopY + 0.4,
          minZ: zCenter - 3.65,
          maxZ: zCenter + 3.65
        });
        // Twin Tajug Subsea Ventilation Towers on Portal Terrace
        const tx = ux + sx * 5.65;
        const tz = zPortal - outSignZ * 2.55;
        pushStatic({
          id: `Underpass_${tag}VentTower_${sx}`,
          type: 'BOX',
          minX: tx - 1.75,
          maxX: tx + 1.75,
          minY: deckTopY - 0.5,
          maxY: deckTopY + 6.95,
          minZ: tz - 1.75,
          maxZ: tz + 1.75
        });
      }
    }

    const underpassConfig = {
      ux,
      boreHalfW: boreHW,
      roadHalfW: TerrainDEMLoader.UNDERPASS_CARRIAGEWAY_HALF_W,
      zWestStart: TerrainDEMLoader.UNDERPASS_WEST_RAMP_START_Z,
      zWestPortal: zWP,
      zEastPortal: zEP,
      zEastEnd: TerrainDEMLoader.UNDERPASS_EAST_RAMP_END_Z,
      floorY,
      clearH,
      sampleRampY: TerrainDEMLoader.sampleUnderpassDeckY,
      sampleGroundY: TerrainDEMLoader.sampleGroundHeight
    };

    // =========================================================================
    // 5. DERMAGA PENYEBERANGAN UTARA PULAU U & PULAU I (PAVILIONS & MARINE BERTHS)
    // =========================================================================
    const addFerryTerminalColliders = (tag, cx, zForeInland, zShore, zHead, dirZ) => {
      const zForeEdge = zShore - dirZ * 2.5;
      const foreMidZ = 0.5 * (zForeEdge + zForeInland);
      const foreTopY = TerrainDEMLoader.DERMAGA_FORECOURT_Y + 0.035;
      const pierDeckY = TerrainDEMLoader.DERMAGA_DECK_Y;
      const tHeadHW = TerrainDEMLoader.DERMAGA_THEAD_HALF_LEN_X; // 14.0
      const tHeadDepth = TerrainDEMLoader.DERMAGA_THEAD_DEPTH_Z; // 7.6

      // 4 Forecourt Waiting Benches
      const bz = foreMidZ + dirZ * 1.4;
      for (const bxOff of [-8.8, -5.2, 5.2, 8.8]) {
        pushStatic({
          id: `Dermaga_${tag}_Bench_${bxOff}`,
          type: 'BOX',
          minX: cx + bxOff - 0.80,
          maxX: cx + bxOff + 0.80,
          minY: foreTopY,
          maxY: foreTopY + 0.86,
          minZ: bz - 0.28,
          maxZ: bz + 0.28,
          isBench: true
        });
      }

      // 8 Twin-Tier Tajug Terminal Pavilion Columns + Overhead Ring Beam & Roof
      const pavZ = foreMidZ + dirZ * 1.2;
      const eaveY1 = foreTopY + 3.05;
      for (const ox of [-5.4, -1.8, 1.8, 5.4]) {
        for (const oz of [-3.4, 3.4]) {
          pushStatic({
            id: `Dermaga_${tag}_PavCol_${ox}_${oz}`,
            type: 'CYL',
            x: cx + ox,
            z: pavZ + oz,
            r: 0.30,
            minY: foreTopY - 0.2,
            maxY: eaveY1
          });
        }
      }
      pushStatic({
        id: `Dermaga_${tag}_PavRingBeam`,
        type: 'BOX',
        minX: cx - 5.8,
        maxX: cx + 5.8,
        minY: eaveY1 - 0.22,
        maxY: eaveY1 + 0.28,
        minZ: pavZ - 3.92,
        maxZ: pavZ + 3.92
      });
      pushStatic({
        id: `Dermaga_${tag}_PavRoof`,
        type: 'BOX',
        minX: cx - 6.82,
        maxX: cx + 6.82,
        minY: eaveY1 + 0.04,
        maxY: eaveY1 + 3.75,
        minZ: pavZ - 4.62,
        maxZ: pavZ + 4.62
      });

      // Marine Pier Structure Colliders (T-Head Berth Deck, Lower Boarding Stage, & Off-Deck Mooring Dolphins)
      const zTHead0 = Math.min(zHead, zHead - dirZ * tHeadDepth);
      const zTHead1 = Math.max(zHead, zHead - dirZ * tHeadDepth);
      const tHeadCol = {
        id: `Dermaga_${tag}_THeadDeck`,
        type: 'BOX',
        minX: cx - tHeadHW,
        maxX: cx + tHeadHW,
        minY: -6.6,
        maxY: pierDeckY + 0.05,
        minZ: zTHead0 - 0.32,
        maxZ: zTHead1 + 0.32
      };
      pushMarine(tHeadCol);
      pushStatic({ ...tHeadCol, skipWalk: true });

      const stageZ = zHead + dirZ * 1.45;
      const stageCol = {
        id: `Dermaga_${tag}_BoardingStage`,
        type: 'BOX',
        minX: cx - 3.42,
        maxX: cx + 3.42,
        minY: -6.5,
        maxY: -0.18,
        minZ: stageZ - 1.40,
        maxZ: stageZ + 1.40
      };
      pushMarine(stageCol);
      pushStatic({ ...stageCol, skipWalk: true });

      for (const sx of [-1.0, 1.0]) {
        const dolph = {
          id: `Dermaga_${tag}_Dolphin_${sx}`,
          type: 'CYL',
          x: cx + sx * (tHeadHW + 2.35),
          z: zHead + dirZ * 0.45,
          r: 0.48,
          minY: -6.5,
          maxY: 3.65
        };
        pushStatic(dolph);
        pushMarine({ ...dolph });
      }
    };

    addFerryTerminalColliders(
      'U',
      TerrainDEMLoader.DERMAGA_U_X,
      446.5,
      TerrainDEMLoader.DERMAGA_U_SHORE_Z,
      TerrainDEMLoader.DERMAGA_U_HEAD_Z,
      1.0
    );
    addFerryTerminalColliders(
      'I',
      TerrainDEMLoader.DERMAGA_I_X,
      624.5,
      TerrainDEMLoader.DERMAGA_I_SHORE_Z,
      TerrainDEMLoader.DERMAGA_I_HEAD_Z,
      -1.0
    );

    // =========================================================================
    // 6. ALL ~198 SPAWNED TREE TRUNKS & ALL 64 FLUTED CAMPUS STREETLAMPS
    // =========================================================================
    const spawnedTrees =
      biomeGroup && biomeGroup.userData && Array.isArray(biomeGroup.userData.spawnedTrees)
        ? biomeGroup.userData.spawnedTrees
        : AutogenBiomeSpawner.DEFAULT_TREE_SPAWNS;
    for (let i = 0; i < spawnedTrees.length; i++) {
      const t = spawnedTrees[i];
      const gy = TerrainDEMLoader.sampleGroundHeight(t.x, t.z);
      const sx = t.scale || 1.0;
      const trunkR = (t.species === 'PALEM_RAJA' ? 0.34 : 0.42) * sx;
      pushStatic({
        id: `TreeTrunk_${i}_${t.species}`,
        type: 'CYL',
        x: t.x,
        z: t.z,
        r: trunkR,
        minY: gy - 1.0,
        maxY: gy + 5.8 * sx
      });
    }

    const lampCoords =
      biomeGroup && biomeGroup.userData && Array.isArray(biomeGroup.userData.lampCoords)
        ? biomeGroup.userData.lampCoords
        : AutogenBiomeSpawner.getLampCoords();
    for (let i = 0; i < lampCoords.length; i++) {
      const lc = lampCoords[i];
      const pierY = TerrainDEMLoader.sampleFerryPierDeckY(lc.x, lc.z);
      const ly = pierY !== null ? pierY : Math.max(0.0, TerrainDEMLoader.sampleGroundHeight(lc.x, lc.z));
      pushStatic({
        id: `StreetLamp_${i}`,
        type: 'CYL',
        x: lc.x,
        z: lc.z,
        r: 0.19,
        minY: ly - 0.3,
        maxY: ly + 4.02
      });
    }

    // =========================================================================
    // 7. KOMPLEKS DPPD–ILRC–STP, SUDUT CALISTHENICS, ATC KECIL, TITIK KUMPUL & KIOSKS
    // =========================================================================
    for (const col of DPPDComplexBuilder.getSolidColliders()) {
      pushStatic(col);
    }

    {
      const cs = SplineCorridorBuilder.CALISTHENICS_SPEC;
      const cgy = Math.max(0.0, TerrainDEMLoader.sampleGroundHeight(cs.cx, cs.cz));
      pushStatic({
        id: 'Calisthenics_MainRig',
        type: 'BOX',
        minX: cs.cx - 7.3,
        maxX: cs.cx - 2.5,
        minY: cgy,
        maxY: cgy + 2.55,
        minZ: cs.cz - 1.25,
        maxZ: cs.cz + 1.95
      });
      pushStatic({
        id: 'Calisthenics_DipBars',
        type: 'BOX',
        minX: cs.cx - 1.05,
        maxX: cs.cx + 1.25,
        minY: cgy,
        maxY: cgy + 1.32,
        minZ: cs.cz - 1.60,
        maxZ: cs.cz - 0.70
      });
      pushStatic({
        id: 'Calisthenics_AbBenches',
        type: 'BOX',
        minX: cs.cx - 0.30,
        maxX: cs.cx + 1.80,
        minY: cgy,
        maxY: cgy + 0.85,
        minZ: cs.cz + 0.25,
        maxZ: cs.cz + 2.05
      });
      pushStatic({
        id: 'Calisthenics_PlyoAndSign',
        type: 'BOX',
        minX: cs.cx + 3.85,
        maxX: cs.cx + 6.75,
        minY: cgy,
        maxY: cgy + 2.15,
        minZ: cs.cz - 2.35,
        maxZ: cs.cz + 1.35
      });
    }

    pushStatic({
      id: 'PulauI_CompactATCTower',
      type: 'CYL',
      x: TerrainDEMLoader.ATC_I_CX,
      z: TerrainDEMLoader.ATC_I_CZ,
      r: 2.95,
      minY: -1.0,
      maxY: 14.2
    });

    for (const mp of SplineCorridorBuilder.MUSTER_POINTS) {
      const mgy = Math.max(0.0, TerrainDEMLoader.sampleGroundHeight(mp.x, mp.z));
      const cY = Math.cos(mp.yaw);
      const sY = Math.sin(mp.yaw);
      pushStatic({
        id: `MusterSign_${mp.id}`,
        type: 'CYL',
        x: mp.x - 1.32 * sY,
        z: mp.z - 1.32 * cY,
        r: 0.28,
        minY: mgy,
        maxY: mgy + 2.65
      });
    }

    for (const ks of SplineCorridorBuilder.KIOSK_SPECS) {
      const kgy = Math.max(0.0, TerrainDEMLoader.sampleGroundHeight(ks.x, ks.z));
      const cY = Math.cos(ks.yaw);
      const sY = Math.sin(ks.yaw);
      pushStatic({
        id: `CampusKiosk_${ks.id}`,
        type: 'OBB',
        cx: ks.x - 0.35 * sY,
        cz: ks.z - 0.35 * cY,
        halfW: 1.92,
        halfL: 1.42,
        yaw: ks.yaw,
        minY: kgy,
        maxY: kgy + 3.05
      });
      for (const txOff of [-2.15, 2.15]) {
        pushStatic({
          id: `CampusKioskTable_${ks.id}_${txOff > 0 ? 'R' : 'L'}`,
          type: 'CYL',
          x: ks.x + txOff * cY + 1.12 * sY,
          z: ks.z - txOff * sY + 1.12 * cY,
          r: 0.92,
          minY: kgy,
          maxY: kgy + 2.55
        });
      }
    }

    return { staticColliders, marineColliders, underpassConfig };
  }
}
