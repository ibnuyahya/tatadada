import { GAME_PREMISE_CARTRIDGE } from '../data/PremiseCartridge.js';
import { WorldMathCore } from '../world/WorldMathCore.js';

/**
 * ============================================================================
 * PURE TYPED-ARRAY CHUNK GEOMETRY SYNTHESIZER (SHARED BY WORKER & MAIN THREAD)
 * Generates all chunk-local (`zLocal = -(s - chunkStartS)` in [0..-100m])
 * typed arrays + precomputed vertex normals off the main UI thread!
 * ============================================================================
 */
function srgbToLinear(c) {
  return c <= 0.04045 ? c * 0.0773993808 : Math.pow(c * 0.9478672986 + 0.0521327014, 2.4);
}

function hexToRgb(hex) {
  return {
    r: srgbToLinear(((hex >> 16) & 255) / 255),
    g: srgbToLinear(((hex >> 8) & 255) / 255),
    b: srgbToLinear((hex & 255) / 255)
  };
}

function lerpRgb(out, a, b, t) {
  out.r = a.r + (b.r - a.r) * t;
  out.g = a.g + (b.g - a.g) * t;
  out.b = a.b + (b.b - a.b) * t;
  return out;
}

function clamp(v, min, max) {
  return v < min ? min : (v > max ? max : v);
}

function mulberry32(seed) {
  let a = (seed | 0) + 0x6D2B79F5;
  return function () {
    a = Math.imul(a ^ (a >>> 15), a | 1);
    a ^= a + Math.imul(a ^ (a >>> 7), a | 61);
    return ((a ^ (a >>> 14)) >>> 0) / 4294967296;
  };
}

function computeIndexedNormals(positions, indices) {
  const normals = new Float32Array(positions.length);
  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i] * 3;
    const i1 = indices[i + 1] * 3;
    const i2 = indices[i + 2] * 3;

    const ax = positions[i1] - positions[i0];
    const ay = positions[i1 + 1] - positions[i0 + 1];
    const az = positions[i1 + 2] - positions[i0 + 2];

    const bx = positions[i2] - positions[i0];
    const by = positions[i2 + 1] - positions[i0 + 1];
    const bz = positions[i2 + 2] - positions[i0 + 2];

    const nx = ay * bz - az * by;
    const ny = az * bx - ax * bz;
    const nz = ax * by - ay * bx;

    normals[i0] += nx; normals[i0 + 1] += ny; normals[i0 + 2] += nz;
    normals[i1] += nx; normals[i1 + 1] += ny; normals[i1 + 2] += nz;
    normals[i2] += nx; normals[i2 + 1] += ny; normals[i2 + 2] += nz;
  }

  for (let i = 0; i < normals.length; i += 3) {
    const l = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1.0;
    normals[i] /= l;
    normals[i + 1] /= l;
    normals[i + 2] /= l;
  }
  return normals;
}

function computeNonIndexedNormals(positions) {
  const normals = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 9) {
    const ax = positions[i + 3] - positions[i];
    const ay = positions[i + 4] - positions[i + 1];
    const az = positions[i + 5] - positions[i + 2];

    const bx = positions[i + 6] - positions[i];
    const by = positions[i + 7] - positions[i + 1];
    const bz = positions[i + 8] - positions[i + 2];

    let nx = ay * bz - az * by;
    let ny = az * bx - ax * bz;
    let nz = ax * by - ay * bx;
    const l = Math.hypot(nx, ny, nz) || 1.0;
    nx /= l; ny /= l; nz /= l;

    normals[i] = nx; normals[i + 1] = ny; normals[i + 2] = nz;
    normals[i + 3] = nx; normals[i + 4] = ny; normals[i + 5] = nz;
    normals[i + 6] = nx; normals[i + 7] = ny; normals[i + 8] = nz;
  }
  return normals;
}

export class ChunkGeometryCore {
  static buildChunkBuffers(chunkIdx) {
    const chunkLen = GAME_PREMISE_CARTRIDGE.railway.chunkLength;
    const sStart = chunkIdx * chunkLen;
    const sEnd = sStart + chunkLen;
    const { biome } = WorldMathCore.getBiomeAtDistance(sStart + 50);
    const rng = mulberry32(chunkIdx * 73856093 + 19349663);

    const steps = 40;
    const ds = chunkLen / steps; // 2.5m high-precision spline chord resolution
    const secondTrackOff = GAME_PREMISE_CARTRIDGE.railway.doubleTrackOffset;
    const gaugeHalf = GAME_PREMISE_CARTRIDGE.railway.standardGauge * 0.5;

    const ballastVerts = [];
    const ballastUVs = [];
    const ballastCols = [];
    const ballastIdx = [];

    const railVerts = [];
    const railCols = [];
    const railIdx = [];

    const wireVerts = [];
    const wireIdx = [];

    const terrainVerts = [];
    const terrainUVs = [];
    const terrainCols = [];
    const terrainIdx = [];

    const waterVerts = [];
    const waterUVs = [];
    const waterCols = [];
    const waterIdx = [];

    const lateralOffsets = [
      -460, -370, -290, -220, -165, -120, -86, -60, -40, -26, -18, -15, -11.5, -7, -3.5, 0,
      3.5, 7, 11.5, 14.5, 19.5, 24.5, 27.5, 40, 60, 86, 120, 165, 220, 290, 370, 460
    ];
    const tGridCols = lateralOffsets.length - 1;

    const baseCol = hexToRgb(biome.groundColor);
    const cliffCol = hexToRgb(biome.cliffColor);
    const lushCol = {
      r: clamp(baseCol.r * 0.88, 0, 1),
      g: clamp(baseCol.g * 1.06, 0, 1),
      b: clamp(baseCol.b * 0.84, 0, 1)
    };
    const paddyCol = hexToRgb(0x4d7c0f);
    const paddyWaterCol = hexToRgb(0xbae6fd);
    const snowCol = hexToRgb(0xf8fafc);
    const shoulderCol = hexToRgb(0x475569);
    const sandCol = hexToRgb(0xfbbf24);
    const cobbleBankCol = hexToRgb(0x64748b);
    const wetRockCol = hexToRgb(0x1e293b);
    const asphaltCol = hexToRgb(0x1e293b);

    const railHeadCol = hexToRgb(0xf8fafc);
    const railWebCol = hexToRgb(0x3f2e21);
    const railFootCol = hexToRgb(0x292524);
    const ballastTintOpen = hexToRgb(0xffffff);
    const ballastTintSlab = hexToRgb(0xcbd5e1);
    const prevWireBase = [null, null];

    const f1 = {};
    const fMid = {};
    const colTmp = { r: 0, g: 0, b: 0 };

    const wireTrackOffsets = [0, secondTrackOff];
    const cxInt = GAME_PREMISE_CARTRIDGE.railway.crossingInterval;
    const stInt = GAME_PREMISE_CARTRIDGE.railway.stationInterval;

    // Structural placements discovered along chunk to instantiate on main thread
    const tunnelSpans = [];
    const tunnelPortals = [];
    const viaductSpans = [];
    const bridgeAbutments = [];
    const catenarySpans = [];
    let chunkMaxViaduct = 0.0;

    // Helper to extrude secondary 3D steel rails (guard/check rails & HSR crossover turnouts) into the 1-draw-call rail buffer
    const pushExtrudedRailStrip = (pts, headHalfW = 0.028, topY = 0.102, botY = 0.005) => {
      if (pts.length < 2) return;
      const baseIdx = railVerts.length / 3;
      const footHalfW = headHalfW * 1.45;
      for (let p = 0; p < pts.length; p++) {
        const pt = pts[p];
        const { x, y, z, rx, ry, rz, ux, uy, uz, bright } = pt;
        const hCol = bright ? railHeadCol : railWebCol;
        railVerts.push(
          x - rx * footHalfW + ux * botY, y - ry * footHalfW + uy * botY, z - rz * footHalfW + uz * botY,
          x - rx * headHalfW + ux * topY, y - ry * headHalfW + uy * topY, z - rz * headHalfW + uz * topY,
          x + rx * headHalfW + ux * topY, y + ry * headHalfW + uy * topY, z + rz * headHalfW + uz * topY,
          x + rx * footHalfW + ux * botY, y + ry * footHalfW + uy * botY, z + rz * footHalfW + uz * botY
        );
        railCols.push(
          railFootCol.r, railFootCol.g, railFootCol.b,
          hCol.r, hCol.g, hCol.b,
          hCol.r, hCol.g, hCol.b,
          railFootCol.r, railFootCol.g, railFootCol.b
        );
        if (p < pts.length - 1) {
          const b0 = baseIdx + p * 4;
          const b1 = b0 + 4;
          railIdx.push(
            b0, b0 + 1, b1, b0 + 1, b1 + 1, b1,
            b0 + 1, b0 + 2, b1 + 1, b0 + 2, b1 + 2, b1 + 1,
            b0 + 2, b0 + 3, b1 + 2, b0 + 3, b1 + 3, b1 + 2
          );
        }
      }
    };

    for (let i = 0; i <= steps; i++) {
      const s = sStart + i * ds;
      WorldMathCore.getTrackFrameRaw(s, 0, sStart, f1);
      WorldMathCore.getTrackFrameRaw(s, secondTrackOff * 0.5, sStart, fMid);
      const nextInTunnel = i < steps ? WorldMathCore.isTunnelZone(s + ds) : f1.inTunnel;
      const isPortalTransition = (i < steps) && (f1.inTunnel !== nextInTunnel);

      const vFac = WorldMathCore.getViaductFactor(s);
      const nextVFac = i < steps ? WorldMathCore.getViaductFactor(s + ds) : vFac;
      if (vFac > chunkMaxViaduct) chunkMaxViaduct = vFac;

      const pmx = fMid.px, pmy = fMid.py, pmz = fMid.pz;
      const rx = fMid.rx, ry = fMid.ry, rz = fMid.rz;
      const ux = fMid.ux, uy = fMid.uy, uz = fMid.uz;

      // 1. 6-Vertex Engineered Double-Track Formation / Ballast Prism (switches cleanly to concrete sub-deck on bridges/tunnels)
      const isSlabZone = f1.inTunnel || vFac > 0.06 || WorldMathCore.isNearStation(s, 88);
      const bTint = isSlabZone ? ballastTintSlab : ballastTintOpen;
      const bedBotY = isSlabZone ? -0.22 : -0.56;
      const outerW = isSlabZone ? 4.85 : 5.85;
      ballastVerts.push(
        pmx + rx * -outerW + ux * bedBotY, pmy + ry * -outerW + uy * bedBotY, pmz + rz * -outerW + uz * bedBotY,
        pmx + rx * -4.25 + ux * -0.05, pmy + ry * -4.25 + uy * -0.05, pmz + rz * -4.25 + uz * -0.05,
        pmx + rx * -0.55 + ux * -0.06, pmy + ry * -0.55 + uy * -0.06, pmz + rz * -0.55 + uz * -0.06,
        pmx + rx * 0.55 + ux * -0.06, pmy + ry * 0.55 + uy * -0.06, pmz + rz * 0.55 + uz * -0.06,
        pmx + rx * 4.25 + ux * -0.05, pmy + ry * 4.25 + uy * -0.05, pmz + rz * 4.25 + uz * -0.05,
        pmx + rx * outerW + ux * bedBotY, pmy + ry * outerW + uy * bedBotY, pmz + rz * outerW + uz * bedBotY
      );
      const vCoord = s * 0.25;
      ballastUVs.push(0, vCoord, 0.14, vCoord, 0.46, vCoord, 0.54, vCoord, 0.86, vCoord, 1, vCoord);
      for (let k = 0; k < 6; k++) {
        const cShade = (k === 2 || k === 3) ? 0.82 : 1.0;
        ballastCols.push(bTint.r * cShade, bTint.g * cShade, bTint.b * cShade);
      }

      if (i < steps) {
        const r0 = i * 6;
        const r1 = (i + 1) * 6;
        for (let q = 0; q < 5; q++) {
          ballastIdx.push(r0 + q, r0 + q + 1, r1 + q, r0 + q + 1, r1 + q + 1, r1 + q);
        }
      }

      // 2. 4 Continuous 8-Vertex UIC-60 Steel I-Beam Rails (Foot + Recessed Oxidized Web + Mirror-Polished Running Head at y=0.105m)
      const cosC = Math.cos(f1.cantAngle);
      const sinC = Math.sin(f1.cantAngle);
      const crx = rx * cosC + ux * sinC, cry = ry * cosC + uy * sinC, crz = rz * cosC + uz * sinC;
      const cux = -rx * sinC + ux * cosC, cuy = -ry * sinC + uy * cosC, cuz = -rz * sinC + uz * cosC;

      for (let rNumber = 0; rNumber < 4; rNumber++) {
        const trackCenterOff = rNumber < 2 ? 0 : secondTrackOff;
        const railLocalX = (rNumber % 2 === 0) ? -gaugeHalf : gaugeHalf;
        const tcx = f1.px + rx * trackCenterOff;
        const tcy = f1.py + ry * trackCenterOff;
        const tcz = f1.pz + rz * trackCenterOff;

        const rcx = tcx + crx * railLocalX;
        const rcy = tcy + cry * railLocalX;
        const rcz = tcz + crz * railLocalX;

        railVerts.push(
          rcx + crx * -0.075 + cux * -0.012, rcy + cry * -0.075 + cuy * -0.012, rcz + crz * -0.075 + cuz * -0.012,
          rcx + crx * -0.017 + cux * 0.008,  rcy + cry * -0.017 + cuy * 0.008,  rcz + crz * -0.017 + cuz * 0.008,
          rcx + crx * -0.014 + cux * 0.074,  rcy + cry * -0.014 + cuy * 0.074,  rcz + crz * -0.014 + cuz * 0.074,
          rcx + crx * -0.036 + cux * 0.105,  rcy + cry * -0.036 + cuy * 0.105,  rcz + crz * -0.036 + cuz * 0.105,
          rcx + crx *  0.036 + cux * 0.105,  rcy + cry *  0.036 + cuy * 0.105,  rcz + crz *  0.036 + cuz * 0.105,
          rcx + crx *  0.014 + cux * 0.074,  rcy + cry *  0.014 + cuy * 0.074,  rcz + crz *  0.014 + cuz * 0.074,
          rcx + crx *  0.017 + cux * 0.008,  rcy + cry *  0.017 + cuy * 0.008,  rcz + crz *  0.017 + cuz * 0.008,
          rcx + crx *  0.075 + cux * -0.012, rcy + cry *  0.075 + cuy * -0.012, rcz + crz *  0.075 + cuz * -0.012
        );
        railCols.push(
          railFootCol.r, railFootCol.g, railFootCol.b,
          railWebCol.r,  railWebCol.g,  railWebCol.b,
          railWebCol.r,  railWebCol.g,  railWebCol.b,
          railHeadCol.r, railHeadCol.g, railHeadCol.b,
          railHeadCol.r, railHeadCol.g, railHeadCol.b,
          railWebCol.r,  railWebCol.g,  railWebCol.b,
          railWebCol.r,  railWebCol.g,  railWebCol.b,
          railFootCol.r, railFootCol.g, railFootCol.b
        );

        if (i < steps) {
          const b0 = (i * 32) + (rNumber * 8);
          const b1 = ((i + 1) * 32) + (rNumber * 8);
          for (let q = 0; q < 7; q++) {
            railIdx.push(b0 + q, b0 + q + 1, b1 + q, b0 + q + 1, b1 + q + 1, b1 + q);
          }
        }
      }

      // 3. Full 3D Overhead Catenary Wire System with Authentic ±0.20m Zig-Zag Pantograph Stagger & 5m Droppers
      const spanPhase = (((s % 50) + 50) % 50) / 50;
      const staggerCycle = (((s % 100) + 100) % 100) / 50; // 0..2 triangular wave synced to 50m masts
      const zigZagStagger = (staggerCycle <= 1.0 ? (staggerCycle * 2.0 - 1.0) : (3.0 - staggerCycle * 2.0)) * 0.20;
      const sagCurve = 4.0 * spanPhase * (1.0 - spanPhase);
      const contactH = 5.32;
      const messengerH = 6.28 - sagCurve * 0.48;

      for (let wNumber = 0; wNumber < 2; wNumber++) {
        const staggerSign = wNumber === 0 ? 1 : -1;
        const tOff = wireTrackOffsets[wNumber] + zigZagStagger * staggerSign;
        const ccx = f1.px + rx * tOff + ux * contactH;
        const ccy = f1.py + ry * tOff + uy * contactH;
        const ccz = f1.pz + rz * tOff + uz * contactH;

        const mcx = f1.px + rx * tOff + ux * messengerH;
        const mcy = f1.py + ry * tOff + uy * messengerH;
        const mcz = f1.pz + rz * tOff + uz * messengerH;

        const curBase = wireVerts.length / 3;
        wireVerts.push(
          ccx + ux * -0.024, ccy + uy * -0.024, ccz + uz * -0.024,
          ccx + ux * 0.024 + rx * 0.018, ccy + uy * 0.024 + ry * 0.018, ccz + uz * 0.024 + rz * 0.018,
          mcx + ux * -0.022, mcy + uy * -0.022, mcz + uz * -0.022,
          mcx + ux * 0.022 + rx * 0.018, mcy + uy * 0.022 + ry * 0.018, mcz + uz * 0.022 + rz * 0.018
        );

        if (prevWireBase[wNumber] !== null) {
          const wb0 = prevWireBase[wNumber];
          const wb1 = curBase;
          wireIdx.push(wb0, wb0 + 1, wb1, wb0 + 1, wb1 + 1, wb1);
          wireIdx.push(wb0 + 2, wb0 + 3, wb1 + 2, wb0 + 3, wb1 + 3, wb1 + 2);
        }
        prevWireBase[wNumber] = curBase;

        if (i % 2 === 0 && i < steps) {
          const dBase = wireVerts.length / 3;
          wireVerts.push(
            ccx + rx * -0.012, ccy + ry * -0.012, ccz + rz * -0.012,
            ccx + rx * 0.012, ccy + ry * 0.012, ccz + rz * 0.012,
            mcx + rx * -0.012, mcy + ry * -0.012, mcz + rz * -0.012,
            mcx + rx * 0.012, mcy + ry * 0.012, mcz + rz * 0.012
          );
          wireIdx.push(dBase, dBase + 1, dBase + 2, dBase + 1, dBase + 3, dBase + 2);
        }
      }

      // 4. Non-Uniform 32-Column Terrain Mesh Strip (with Twin Railway + Highway Tunnel Portal Cutouts & Riparian Riverbanks)
      const worldZ = -s;
      const riverWaterY = WorldMathCore.getRiverWaterLevel(s);

      for (let c = 0; c <= tGridCols; c++) {
        const lx = lateralOffsets[c];
        const wx = pmx + lx;
        const localZ = pmz;
        const wy = WorldMathCore.getTerrainHeightAtS(wx, s, pmx, pmy);
        terrainVerts.push(wx, wy, localZ);
        terrainUVs.push(wx * 0.045, worldZ * 0.045);

        const relH = wy - pmy;
        const absLat = Math.abs(lx);
        const vFacAt = WorldMathCore.getViaductFactorAt(wx, s, pmx);
        colTmp.r = baseCol.r; colTmp.g = baseCol.g; colTmp.b = baseCol.b;

        const wyEast = WorldMathCore.getTerrainHeightAtS(wx + 3.5, s, pmx, pmy);
        const slopeX = (wy - wyEast) / 3.5;
        const sunShade = clamp(1.0 + slopeX * 0.28, 0.68, 1.26);

        const noiseVar = Math.sin(wx * 0.03 + worldZ * 0.03) * 0.5 + 0.5;
        lerpRgb(colTmp, colTmp, lushCol, noiseVar * 0.65);

        if (biome.id === "PRIANGAN" && relH > 1.2 && relH < 16.5) {
          const band = Math.abs(Math.sin(wy * 1.9));
          lerpRgb(colTmp, colTmp, paddyCol, band * 0.75);
          if (band < 0.22) lerpRgb(colTmp, colTmp, paddyWaterCol, 0.32);
        }

        const steep = clamp((Math.abs(relH) - 6.0) / 15.0, 0, 1);
        lerpRgb(colTmp, colTmp, cliffCol, steep * 0.84);

        if (relH > 22.0) {
          const snowT = clamp((relH - 22.0) / 14.0, 0, 1);
          lerpRgb(colTmp, colTmp, snowCol, snowT * (biome.id === "ALPINE" ? 0.96 : 0.38));
        }

        // Riparian river cobble banks, wet shoreline rock & coastal sand
        if (vFacAt > 0.05 && wy < riverWaterY + 3.2) {
          const bankT = clamp(1.0 - (wy - riverWaterY) / 3.2, 0.0, 1.0);
          lerpRgb(colTmp, colTmp, cobbleBankCol, bankT * 0.85);
          if (wy < riverWaterY + 0.65) {
            lerpRgb(colTmp, colTmp, wetRockCol, 0.72);
          }
        } else if (wy < biome.waterLevel + 2.2) {
          lerpRgb(colTmp, colTmp, sandCol, 0.58);
        }

        if (!f1.inTunnel && vFacAt < 0.04) {
          if (absLat < 14.2) {
            lerpRgb(colTmp, colTmp, shoulderCol, 0.85);
          } else if (lx >= 14.4 && lx <= 24.6) {
            colTmp.r = asphaltCol.r; colTmp.g = asphaltCol.g; colTmp.b = asphaltCol.b;
          }
        }

        const valleyAO = relH < -2.0 ? clamp(1.0 + relH * 0.025, 0.62, 1.0) : 1.0;
        const mult = sunShade * valleyAO;
        terrainCols.push(colTmp.r * mult, colTmp.g * mult, colTmp.b * mult);

        if (i < steps && c < tGridCols) {
          // Cut open both the 23m twin-track HSR tunnel mouth [-11.5..+11.5] AND the 10m parallel highway tunnel mouth [14.5..24.5]!
          const skipRailMouth = isPortalTransition && lateralOffsets[c] >= -11.6 && lateralOffsets[c + 1] <= 11.6;
          const skipHwyMouth = isPortalTransition && lateralOffsets[c] >= 14.4 && lateralOffsets[c + 1] <= 24.6;
          if (!skipRailMouth && !skipHwyMouth) {
            const tr0 = i * (tGridCols + 1) + c;
            const tr1 = (i + 1) * (tGridCols + 1) + c;
            terrainIdx.push(tr0, tr0 + 1, tr1, tr0 + 1, tr1 + 1, tr1);
          }
        }
      }

      if (isPortalTransition) {
        const portalS = s + ds * 0.5;
        tunnelPortals.push({ s: portalS, isExit: f1.inTunnel });
        tunnelSpans.push(f1.inTunnel ? s : (s + ds));
      }
      if (f1.inTunnel && i % 4 === 0 && i < steps) {
        tunnelSpans.push(s);
      }

      // Bridge Abutments at exact gorge transitions (eliminating floating tracks at bridge approaches!)
      const wasBridge = vFac > 0.06;
      const willBeBridge = nextVFac > 0.06;
      if (i < steps && wasBridge !== willBeBridge) {
        bridgeAbutments.push({ s: s + ds * 0.5, isExit: wasBridge });
      }

      // Continuous Bridge Spans: Approach Box-Girders (0.06 < vFac <= 0.50) + Main River Steel Truss (vFac > 0.50)
      if (vFac > 0.06 && i % 4 === 0 && i < steps) {
        const isMainTruss = vFac > 0.50;
        const groundAtTrack = WorldMathCore.getTerrainHeightAtS(pmx, s, pmx, pmy);
        viaductSpans.push({
          s,
          mode: isMainTruss ? 'TRUSS' : 'BOX_GIRDER',
          hasPier: i % 8 === 0,
          hasCatenaryMast: !isMainTruss && (i % 16 === 0),
          inWater: groundAtTrack < riverWaterY - 0.35,
          riverLocalY: riverWaterY - pmy
        });
      }

      if (
        !f1.inTunnel &&
        vFac <= 0.06 &&
        !WorldMathCore.isNearStation(s, 84) &&
        i % 20 === 0 &&
        i < steps
      ) {
        catenarySpans.push(s);
      }
    }

    // 4B. Inner Steel Check / Guard Rails (`Rel Paksa`) across Bridges & Level Crossings
    {
      const nearCx = Math.round((sStart + 50) / cxInt) * cxInt;
      const nearSt = Math.round(nearCx / stInt) * stInt;
      const hasActiveCx = nearCx > 300 && Math.abs(nearCx - nearSt) > 350 && nearCx >= sStart - 12 && nearCx <= sEnd + 12;

      if (chunkMaxViaduct > 0.06 || hasActiveCx) {
        for (let tIdx = 0; tIdx < 2; tIdx++) {
          const trackOff = tIdx * secondTrackOff;
          for (let side of [-1, 1]) {
            let runPts = [];
            for (let i = 0; i <= steps; i++) {
              const s = sStart + i * ds;
              const vFac = WorldMathCore.getViaductFactor(s);
              const onCx = hasActiveCx && Math.abs(s - nearCx) <= 8.5;
              if (vFac > 0.06 || onCx) {
                WorldMathCore.getTrackFrameRaw(s, trackOff, sStart, f1);
                const cosC = Math.cos(f1.cantAngle), sinC = Math.sin(f1.cantAngle);
                const crx = f1.rx * cosC + f1.ux * sinC, cry = f1.ry * cosC + f1.uy * sinC, crz = f1.rz * cosC + f1.uz * sinC;
                const cux = -f1.rx * sinC + f1.ux * cosC, cuy = -f1.ry * sinC + f1.uy * cosC, cuz = -f1.rz * sinC + f1.uz * cosC;
                // Flare guard rail ends inward smoothly at bridge entries/exits
                let guardOff = gaugeHalf - 0.19;
                if (!onCx && vFac < 0.13) {
                  const flareT = clamp((0.13 - vFac) / 0.07, 0.0, 1.0);
                  guardOff -= flareT * 0.22;
                }
                const gx = f1.px + crx * (side * guardOff);
                const gy = f1.py + cry * (side * guardOff);
                const gz = f1.pz + crz * (side * guardOff);
                runPts.push({ x: gx, y: gy, z: gz, rx: crx, ry: cry, rz: crz, ux: cux, uy: cuy, uz: cuz, bright: false });
              } else if (runPts.length > 0) {
                pushExtrudedRailStrip(runPts, 0.024, 0.104, 0.010);
                runPts = [];
              }
            }
            if (runPts.length > 0) {
              pushExtrudedRailStrip(runPts, 0.024, 0.104, 0.010);
            }
          }
        }
      }
    }

    // 4C. Double-Track High-Speed Crossover Switches (`Wesel Silang HSR 1:18`) at Station Approach & Departure Throats
    const switchActuators = [];
    {
      const nearStIdx = Math.round((sStart + 50) / stInt);
      const stCenter = nearStIdx * stInt;
      // Departure crossover [stCenter + 102 .. +154] (including right ahead of start s=50!) & Arrival crossover [stCenter - 248 .. -196]
      const candidateSwitches = [
        { s0: stCenter + 102, len: 52, dir: 1 },
        ...(nearStIdx >= 1 ? [{ s0: stCenter - 248, len: 52, dir: -1 }] : [])
      ];

      for (const sw of candidateSwitches) {
        const swEnd = sw.s0 + sw.len;
        if (swEnd < sStart || sw.s0 > sEnd) continue;
        const clipS0 = Math.max(sStart, sw.s0);
        const clipS1 = Math.min(sEnd, swEnd);
        const segs = Math.max(2, Math.ceil((clipS1 - clipS0) / 2.0));
        const leftRailPts = [];
        const rightRailPts = [];

        for (let k = 0; k <= segs; k++) {
          const s = clipS0 + (k / segs) * (clipS1 - clipS0);
          const u = clamp((s - sw.s0) / sw.len, 0.0, 1.0);
          // Smooth S-curve turnout geometry (quintic C2 continuous transition from Track 1 -> Track 2)
          const smoothS = u * u * u * (u * (u * 6 - 15) + 10);
          const latCenter = sw.dir > 0 ? (smoothS * secondTrackOff) : ((1.0 - smoothS) * secondTrackOff);
          const dLatDs = (30 * u * u * (u - 1) * (u - 1) * secondTrackOff / sw.len) * (sw.dir > 0 ? 1 : -1);
          const yaw = Math.atan(dLatDs);

          WorldMathCore.getTrackFrameRaw(s, latCenter, sStart, f1);
          const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
          const srx = f1.rx * cosY + f1.tx * sinY;
          const sry = f1.ry * cosY + f1.ty * sinY;
          const srz = f1.rz * cosY + f1.tz * sinY;

          leftRailPts.push({
            x: f1.px - srx * gaugeHalf, y: f1.py - sry * gaugeHalf, z: f1.pz - srz * gaugeHalf,
            rx: srx, ry: sry, rz: srz, ux: f1.ux, uy: f1.uy, uz: f1.uz, bright: true
          });
          rightRailPts.push({
            x: f1.px + srx * gaugeHalf, y: f1.py + sry * gaugeHalf, z: f1.pz + srz * gaugeHalf,
            rx: srx, ry: sry, rz: srz, ux: f1.ux, uy: f1.uy, uz: f1.uz, bright: true
          });
        }
        pushExtrudedRailStrip(leftRailPts, 0.032, 0.104, -0.008);
        pushExtrudedRailStrip(rightRailPts, 0.032, 0.104, -0.008);

        // Opposite mainline check rails beside the common crossing frogs + electric point machine locations
        if (sw.s0 + 3.0 >= sStart && sw.s0 + 3.0 < sEnd) {
          switchActuators.push({ s: sw.s0 + 3.0, lat: sw.dir > 0 ? -1.45 : secondTrackOff + 1.45, longBearersS: sw.s0 });
        }
        if (swEnd - 3.0 >= sStart && swEnd - 3.0 < sEnd) {
          switchActuators.push({ s: swEnd - 3.0, lat: sw.dir > 0 ? secondTrackOff + 1.45 : -1.45, longBearersS: sw.s0 });
        }
      }
    }

    // 4D. Chunk-Local 3D River & Coastal Water Surface Mesh (Zero Shoreline Gaps + Depth-Based Rapids Foam!)
    const hasRiverNearby =
      chunkMaxViaduct > 0.02 ||
      WorldMathCore.getViaductFactor(sStart - 32) > 0.02 ||
      WorldMathCore.getViaductFactor(sEnd + 32) > 0.02;
    const buildChunkWater = hasRiverNearby || biome.id === 'COASTAL';

    if (buildChunkWater) {
      const wCols = 24;
      const wRows = 20;
      const deepRiverCol = hexToRgb(biome.id === 'COASTAL' ? 0x0369a1 : 0x0e7490);
      const shallowRiverCol = hexToRgb(0x38bdf8);
      const foamWhiteCol = hexToRgb(0xf0f9ff);
      const wColTmp = { r: 0, g: 0, b: 0 };

      for (let r = 0; r <= wRows; r++) {
        const s = sStart + (r / wRows) * chunkLen;
        WorldMathCore.getTrackFrameRaw(s, secondTrackOff * 0.5, sStart, fMid);
        const wLevel = WorldMathCore.getRiverWaterLevel(s);
        const worldZ = -s;

        for (let c = 0; c <= wCols; c++) {
          const uFrac = (c / wCols) * 2.0 - 1.0; // -1 .. +1
          // Concentrate resolution near the railway + highway bridge piers while spanning [-450..+450m]
          const lx = Math.sign(uFrac) * Math.pow(Math.abs(uFrac), 1.55) * 450.0;
          const wx = fMid.px + lx;
          const groundY = WorldMathCore.getTerrainHeightAtS(wx, s, fMid.px, fMid.py);
          const depth = wLevel - groundY;
          const waveY = Math.sin(wx * 0.14 + worldZ * 0.11) * 0.06;

          waterVerts.push(wx, wLevel + waveY, fMid.pz);
          waterUVs.push(wx * 0.055, worldZ * 0.055);

          const shallowT = clamp(1.0 - depth / 2.6, 0.0, 1.0);
          lerpRgb(wColTmp, deepRiverCol, shallowRiverCol, shallowT * 0.72);

          // Natural shoreline surf + whitewater rapids around railway & highway bridge piers!
          const nearRailPier = Math.abs(lx) < 5.6 && depth > 0.3;
          const nearHwyPier = Math.abs(lx - 19.5) < 5.2 && depth > 0.3;
          const rapidsWave = Math.sin(wx * 0.35 + worldZ * 0.42) * 0.5 + 0.5;
          let foam = depth > -0.4 && depth < 0.55 ? (1.0 - Math.abs(depth - 0.1) / 0.55) * 0.65 : 0.0;
          if (nearRailPier || nearHwyPier) foam = Math.max(foam, 0.55 + rapidsWave * 0.40);
          lerpRgb(wColTmp, wColTmp, foamWhiteCol, clamp(foam, 0.0, 0.92));

          waterCols.push(wColTmp.r, wColTmp.g, wColTmp.b);

          if (r < wRows && c < wCols) {
            const i0 = r * (wCols + 1) + c;
            const i1 = (r + 1) * (wCols + 1) + c;
            waterIdx.push(i0, i0 + 1, i1, i0 + 1, i1 + 1, i1);
          }
        }
      }
    }

    // 5. Batched Scenery, Multi-Species Trees, Landmarks, SUTET & Villages (1 Draw Call!)
    const batchVerts = [];
    const batchCols = [];
    const metropolisBuildings = [];
    const windTurbines = [];

    const pushTri = (x1, y1, z1, x2, y2, z2, x3, y3, z3, c1, c2, c3) => {
      batchVerts.push(x1, y1, z1, x2, y2, z2, x3, y3, z3);
      batchCols.push(c1.r, c1.g, c1.b, c2.r, c2.g, c2.b, c3.r, c3.g, c3.b);
    };

    const pushBox = (cx, cy, cz, sx, sy, sz, colTop, colSide) => {
      const hx = sx * 0.5, hy = sy * 0.5, hz = sz * 0.5;
      const x0 = cx - hx, x1 = cx + hx;
      const y0 = cy - hy, y1 = cy + hy;
      const z0 = cz - hz, z1 = cz + hz;
      pushTri(x0, y0, z1, x1, y0, z1, x1, y1, z1, colSide, colSide, colTop);
      pushTri(x0, y0, z1, x1, y1, z1, x0, y1, z1, colSide, colTop, colTop);
      pushTri(x1, y0, z0, x0, y0, z0, x0, y1, z0, colSide, colSide, colTop);
      pushTri(x1, y0, z0, x0, y1, z0, x1, y1, z0, colSide, colTop, colTop);
      pushTri(x0, y0, z0, x0, y0, z1, x0, y1, z1, colSide, colSide, colTop);
      pushTri(x0, y0, z0, x0, y1, z1, x0, y1, z0, colSide, colTop, colTop);
      pushTri(x1, y0, z1, x1, y0, z0, x1, y1, z0, colSide, colSide, colTop);
      pushTri(x1, y0, z1, x1, y1, z0, x1, y1, z1, colSide, colTop, colTop);
      pushTri(x0, y1, z1, x1, y1, z1, x1, y1, z0, colTop, colTop, colTop);
      pushTri(x0, y1, z1, x1, y1, z0, x0, y1, z0, colTop, colTop, colTop);
    };

    // Orthonormal-basis oriented 3D box (aligns sleepers, slabs, bridges & road markings to exact curve tangent, grade & cant!)
    const pushBasisBox = (
      cx, cy, cz,
      axX, axY, axZ,
      upX, upY, upZ,
      bzX, bzY, bzZ,
      sx, sy, sz,
      colTop, colSide,
      withBottom = false
    ) => {
      const hx = sx * 0.5, hy = sy * 0.5, hz = sz * 0.5;
      const rX = axX * hx, rY = axY * hx, rZ = axZ * hx;
      const uX = upX * hy, uY = upY * hy, uZ = upZ * hy;
      const fX = bzX * hz, fY = bzY * hz, fZ = bzZ * hz;

      const x000 = cx - rX - uX - fX, y000 = cy - rY - uY - fY, z000 = cz - rZ - uZ - fZ;
      const x100 = cx + rX - uX - fX, y100 = cy + rY - uY - fY, z100 = cz + rZ - uZ - fZ;
      const x010 = cx - rX + uX - fX, y010 = cy - rY + uY - fY, z010 = cz - rZ + uZ - fZ;
      const x110 = cx + rX + uX - fX, y110 = cy + rY + uY - fY, z110 = cz + rZ + uZ - fZ;
      const x001 = cx - rX - uX + fX, y001 = cy - rY - uY + fY, z001 = cz - rZ - uZ + fZ;
      const x101 = cx + rX - uX + fX, y101 = cy + rY - uY + fY, z101 = cz + rZ - uZ + fZ;
      const x011 = cx - rX + uX + fX, y011 = cy - rY + uY + fY, z011 = cz - rZ + uZ + fZ;
      const x111 = cx + rX + uX + fX, y111 = cy + rY + uY + fY, z111 = cz + rZ + uZ + fZ;

      pushTri(x001, y001, z001, x101, y101, z101, x111, y111, z111, colSide, colSide, colTop);
      pushTri(x001, y001, z001, x111, y111, z111, x011, y011, z011, colSide, colTop, colTop);
      pushTri(x100, y100, z100, x000, y000, z000, x010, y010, z010, colSide, colSide, colTop);
      pushTri(x100, y100, z100, x010, y010, z010, x110, y110, z110, colSide, colTop, colTop);
      pushTri(x000, y000, z000, x001, y001, z001, x011, y011, z011, colSide, colSide, colTop);
      pushTri(x000, y000, z000, x011, y011, z011, x010, y010, z010, colSide, colTop, colTop);
      pushTri(x101, y101, z101, x100, y100, z100, x110, y110, z110, colSide, colSide, colTop);
      pushTri(x101, y101, z101, x110, y110, z110, x111, y111, z111, colSide, colTop, colTop);
      pushTri(x011, y011, z011, x111, y111, z111, x110, y110, z110, colTop, colTop, colTop);
      pushTri(x011, y011, z011, x110, y110, z110, x010, y010, z010, colTop, colTop, colTop);
      if (withBottom) {
        pushTri(x000, y000, z000, x100, y100, z100, x101, y101, z101, colSide, colSide, colSide);
        pushTri(x000, y000, z000, x101, y101, z101, x001, y001, z001, colSide, colSide, colSide);
      }
    };

    // 5A. High-Detail 3D Prestressed Concrete Sleepers + Pandrol Clips, CRTS-III Slab Track, Switch Machines & Cable Troughs
    {
      const sleeperTop = hexToRgb(0x94a3b8);
      const sleeperSide = hexToRgb(0x475569);
      const slabTop = hexToRgb(0xcbd5e1);
      const slabSide = hexToRgb(0x64748b);
      const clipDark = hexToRgb(0x1e293b);
      const troughTop = hexToRgb(0x94a3b8);
      const troughSide = hexToRgb(0x334155);
      const ditchWater = hexToRgb(0x0f172a);
      const swYellow = hexToRgb(0xfacc15);

      const nearStIdx = Math.round((sStart + 50) / stInt);
      const stCenter = nearStIdx * stInt;
      const nearCx = Math.round((sStart + 50) / cxInt) * cxInt;
      const hasCxInChunk = nearCx > 300 && Math.abs(nearCx - stCenter) > 350 && nearCx >= sStart - 10 && nearCx <= sEnd + 10;

      const sleeperStep = 1.25;
      const rowCount = Math.floor(chunkLen / sleeperStep);
      for (let r = 0; r < rowCount; r++) {
        const s = sStart + (r + 0.5) * sleeperStep;
        if (hasCxInChunk && Math.abs(s - nearCx) < 3.9) continue; // Crossing rubber pads cover rails here

        WorldMathCore.getTrackFrameRaw(s, 0, sStart, f1);
        const vFac = WorldMathCore.getViaductFactor(s);
        const isSlabTrack = f1.inTunnel || vFac > 0.06 || WorldMathCore.isNearStation(s, 88);
        const inDepSwitch = (s >= stCenter + 100 && s <= stCenter + 156);
        const inArrSwitch = (nearStIdx >= 1 && s >= stCenter - 250 && s <= stCenter - 194);
        const inCrossover = inDepSwitch || inArrSwitch;

        const cosC = Math.cos(f1.cantAngle), sinC = Math.sin(f1.cantAngle);
        const crx = f1.rx * cosC + f1.ux * sinC, cry = f1.ry * cosC + f1.uy * sinC, crz = f1.rz * cosC + f1.uz * sinC;
        const cux = -f1.rx * sinC + f1.ux * cosC, cuy = -f1.ry * sinC + f1.uy * cosC, cuz = -f1.rz * sinC + f1.uz * cosC;
        const bzx = -f1.tx, bzy = -f1.ty, bzz = -f1.tz;

        if (inCrossover) {
          // Full-width 6.95m prestressed concrete turnout bearer ties spanning Track 1 + Track 2!
          const midX = f1.px + f1.rx * (secondTrackOff * 0.5) + cux * -0.024;
          const midY = f1.py + f1.ry * (secondTrackOff * 0.5) + cuy * -0.024;
          const midZ = f1.pz + f1.rz * (secondTrackOff * 0.5) + cuz * -0.024;
          pushBasisBox(midX, midY, midZ, crx, cry, crz, cux, cuy, cuz, bzx, bzy, bzz, 6.95, 0.068, 0.25, sleeperTop, sleeperSide);
        } else if (isSlabTrack) {
          // CRTS-III Precast Concrete Ballastless Track Slabs (4.86m modular slabs with 14cm expansion joints)
          if (r % 4 === 0) {
            const sSlab = s + 1.875;
            WorldMathCore.getTrackFrameRaw(sSlab, 0, sStart, fMid);
            const scos = Math.cos(fMid.cantAngle), ssin = Math.sin(fMid.cantAngle);
            const srx = fMid.rx * scos + fMid.ux * ssin, sry = fMid.ry * scos + fMid.uy * ssin, srz = fMid.rz * scos + fMid.uz * ssin;
            const sux = -fMid.rx * ssin + fMid.ux * scos, suy = -fMid.ry * ssin + fMid.uy * scos, suz = -fMid.rz * ssin + fMid.uz * scos;
            const sbzX = -fMid.tx, sbzY = -fMid.ty, sbzZ = -fMid.tz;
            for (let tIdx = 0; tIdx < 2; tIdx++) {
              const tOff = tIdx * secondTrackOff;
              const cx = fMid.px + fMid.rx * tOff + sux * -0.024;
              const cy = fMid.py + fMid.ry * tOff + suy * -0.024;
              const cz = fMid.pz + fMid.rz * tOff + suz * -0.024;
              pushBasisBox(cx, cy, cz, srx, sry, srz, sux, suy, suz, sbzX, sbzY, sbzZ, 2.56, 0.072, 4.86, slabTop, slabSide);
            }
          }
        } else {
          // Open Corridor: 3D Prestressed Monoblock Concrete Sleepers under both tracks
          for (let tIdx = 0; tIdx < 2; tIdx++) {
            const tOff = tIdx * secondTrackOff;
            const cx = f1.px + f1.rx * tOff + cux * -0.026;
            const cy = f1.py + f1.ry * tOff + cuy * -0.026;
            const cz = f1.pz + f1.rz * tOff + cuz * -0.026;
            pushBasisBox(cx, cy, cz, crx, cry, crz, cux, cuy, cuz, bzx, bzy, bzz, 2.52, 0.068, 0.24, sleeperTop, sleeperSide);
          }
        }

        // Cast-iron Pandrol / Vossloh resilient rail fastener baseplates under all 4 running rails
        for (let rNum = 0; rNum < 4; rNum++) {
          const tOff = rNum < 2 ? 0 : secondTrackOff;
          const rLoc = (rNum % 2 === 0) ? -gaugeHalf : gaugeHalf;
          const px = f1.px + f1.rx * tOff + crx * rLoc + cux * 0.002;
          const py = f1.py + f1.ry * tOff + cry * rLoc + cuy * 0.002;
          const pz = f1.pz + f1.rz * tOff + crz * rLoc + cuz * 0.002;
          pushBasisBox(px, py, pz, crx, cry, crz, cux, cuy, cuz, bzx, bzy, bzz, 0.28, 0.022, 0.20, clipDark, clipDark);
        }
      }

      // Electric Switch Point Machines (`Motor Penggerak Wesel`) & stretcher rods at HSR crossover throats
      for (const act of switchActuators) {
        WorldMathCore.getTrackFrameRaw(act.s, act.lat, sStart, f1);
        const bzx = -f1.tx, bzy = -f1.ty, bzz = -f1.tz;
        pushBasisBox(
          f1.px + f1.ux * 0.08, f1.py + f1.uy * 0.08, f1.pz + f1.uz * 0.08,
          f1.rx, f1.ry, f1.rz, f1.ux, f1.uy, f1.uz, bzx, bzy, bzz,
          0.62, 0.28, 0.95, swYellow, troughSide
        );
        const rodDir = act.lat < 0 ? 0.55 : -0.55;
        pushBasisBox(
          f1.px + f1.rx * rodDir + f1.ux * 0.02, f1.py + f1.ry * rodDir + f1.uy * 0.02, f1.pz + f1.rz * rodDir + f1.uz * 0.02,
          f1.rx, f1.ry, f1.rz, f1.ux, f1.uy, f1.uz, bzx, bzy, bzz,
          0.95, 0.05, 0.12, troughSide, clipDark
        );
      }

      // Trackside Precast Concrete Signal Cable Troughs & Lined Drainage U-Ditches (every 10m along open corridor)
      for (let cs = sStart + 5; cs < sEnd; cs += 10) {
        if (hasCxInChunk && Math.abs(cs - nearCx) < 9.0) continue;
        const vFac = WorldMathCore.getViaductFactor(cs);
        if (WorldMathCore.isTunnelZone(cs) || vFac > 0.06 || WorldMathCore.isNearStation(cs, 90)) continue;
        WorldMathCore.getTrackFrameRaw(cs, secondTrackOff * 0.5, sStart, fMid);
        const bzx = -fMid.tx, bzy = -fMid.ty, bzz = -fMid.tz;
        for (const side of [-1, 1]) {
          // Precast concrete CTCS-3 cable trough at ±4.55m from corridor center
          const tx = fMid.px + fMid.rx * (side * 4.55) + fMid.ux * -0.36;
          const ty = fMid.py + fMid.ry * (side * 4.55) + fMid.uy * -0.36;
          const tz = fMid.pz + fMid.rz * (side * 4.55) + fMid.uz * -0.36;
          pushBasisBox(tx, ty, tz, fMid.rx, fMid.ry, fMid.rz, fMid.ux, fMid.uy, fMid.uz, bzx, bzy, bzz, 0.42, 0.34, 9.96, troughTop, troughSide);

          // Concrete lined side drainage U-ditch at ±6.25m from corridor center
          const dx = fMid.px + fMid.rx * (side * 6.25) + fMid.ux * -0.50;
          const dy = fMid.py + fMid.ry * (side * 6.25) + fMid.uy * -0.50;
          const dz = fMid.pz + fMid.rz * (side * 6.25) + fMid.uz * -0.50;
          pushBasisBox(dx, dy, dz, fMid.rx, fMid.ry, fMid.rz, fMid.ux, fMid.uy, fMid.uz, bzx, bzy, bzz, 0.68, 0.16, 9.98, ditchWater, troughSide);
        }
      }
    }

    if (biome.id === "METROPOLIS") {
      for (let i = 0; i < 10; i++) {
        const s = sStart + (i + 0.5) * (chunkLen / 10);
        if (WorldMathCore.isViaductZone(s)) continue;
        WorldMathCore.getTrackFrameRaw(s, 2.2, sStart, fMid);
        const side = i % 2 === 0 ? 1 : -1;
        const off = (28 + rng() * 75) * side;
        const wx = fMid.px + off;
        const wz = fMid.pz;
        const wy = WorldMathCore.getTerrainHeightAtS(wx, s, fMid.px, fMid.py);
        if (wy < biome.waterLevel + 0.5) continue;

        const bw = 11 + rng() * 14;
        const bh = 26 + rng() * 62;
        const bd = 11 + rng() * 14;
        metropolisBuildings.push({ x: wx, y: wy + bh * 0.5 - 2.0, z: wz, bw, bh, bd });
      }
    }

    const trunkDark = hexToRgb(0x3e2723);
    const trunkMid = hexToRgb(0x5d4037);
    const leafBase = hexToRgb(biome.id === "ALPINE" ? 0x64748b : (biome.id === "CANYON" ? 0x78350f : 0x0f3d1e));
    const leafMid = hexToRgb(biome.id === "ALPINE" ? 0xcbd5e1 : (biome.id === "CANYON" ? 0xb45309 : 0x15803d));
    const leafSun = hexToRgb(biome.id === "ALPINE" ? 0xf8fafc : (biome.id === "CANYON" ? 0xd97706 : 0x4ade80));
    const rockDark = hexToRgb(0x334155);
    const rockLight = hexToRgb(0x64748b);

    const pushTaperedCylinder = (cx, yBot, yTop, cz, rBot, rTop, colB, colT, segs = 8) => {
      for (let sIdx = 0; sIdx < segs; sIdx++) {
        const a0 = (sIdx / segs) * Math.PI * 2;
        const a1 = ((sIdx + 1) / segs) * Math.PI * 2;
        const x0b = cx + Math.cos(a0) * rBot, z0b = cz + Math.sin(a0) * rBot;
        const x1b = cx + Math.cos(a1) * rBot, z1b = cz + Math.sin(a1) * rBot;
        const x0t = cx + Math.cos(a0) * rTop, z0t = cz + Math.sin(a0) * rTop;
        const x1t = cx + Math.cos(a1) * rTop, z1t = cz + Math.sin(a1) * rTop;
        pushTri(x0b, yBot, z0b, x1b, yBot, z1b, x1t, yTop, z1t, colB, colB, colT);
        pushTri(x0b, yBot, z0b, x1t, yTop, z1t, x0t, yTop, z0t, colB, colT, colT);
      }
    };

    const pushFoliageDome = (cx, cy, cz, rx, ry, colBot, colMid, colTop) => {
      const segs = 9;
      const rings = [
        { yA: -0.65 * ry, rA: 0.55 * rx, yB: -0.10 * ry, rB: 0.98 * rx, cA: colBot, cB: colMid },
        { yA: -0.10 * ry, rA: 0.98 * rx, yB: 0.52 * ry, rB: 0.82 * rx, cA: colMid, cB: colTop },
        { yA: 0.52 * ry, rA: 0.82 * rx, yB: 0.95 * ry, rB: 0.0, cA: colTop, cB: colTop }
      ];
      for (const rg of rings) {
        for (let sIdx = 0; sIdx < segs; sIdx++) {
          const a0 = (sIdx / segs) * Math.PI * 2;
          const a1 = ((sIdx + 1) / segs) * Math.PI * 2;
          const x00 = cx + Math.cos(a0) * rg.rA, z00 = cz + Math.sin(a0) * rg.rA;
          const x01 = cx + Math.cos(a1) * rg.rA, z01 = cz + Math.sin(a1) * rg.rA;
          if (rg.rB <= 0.001) {
            pushTri(x00, cy + rg.yA, z00, x01, cy + rg.yA, z01, cx, cy + rg.yB, cz, rg.cA, rg.cA, rg.cB);
          } else {
            const x10 = cx + Math.cos(a0) * rg.rB, z10 = cz + Math.sin(a0) * rg.rB;
            const x11 = cx + Math.cos(a1) * rg.rB, z11 = cz + Math.sin(a1) * rg.rB;
            pushTri(x00, cy + rg.yA, z00, x01, cy + rg.yA, z01, x11, cy + rg.yB, z11, rg.cA, rg.cA, rg.cB);
            pushTri(x00, cy + rg.yA, z00, x11, cy + rg.yB, z11, x10, cy + rg.yB, z10, rg.cA, rg.cB, rg.cB);
          }
        }
      }
    };

    if (biome.id !== "METROPOLIS") {
      const count = Math.floor(38 * biome.treeDensity);
      for (let i = 0; i < count; i++) {
        const s = sStart + rng() * chunkLen;
        if (WorldMathCore.getViaductFactor(s) > 0.12) continue;

        WorldMathCore.getTrackFrameRaw(s, 2.2, sStart, fMid);
        const side = rng() > 0.5 ? 1 : -1;
        const minOff = side > 0 ? 28 : 22;
        const off = (minOff + rng() * 145) * side;

        const wx = fMid.px + off;
        const wz = fMid.pz;
        const wy = WorldMathCore.getTerrainHeightAtS(wx, s, fMid.px, fMid.py);
        if (wy < biome.waterLevel + 1.5) continue;

        const sc = 0.85 + rng() * 1.35;
        const baseY = wy - 0.35;
        const tint = 0.90 + rng() * 0.20;
        const lBase = { r: clamp(leafBase.r * tint, 0, 1), g: clamp(leafBase.g * tint, 0, 1), b: clamp(leafBase.b * tint, 0, 1) };
        const lMid = { r: clamp(leafMid.r * tint, 0, 1), g: clamp(leafMid.g * tint, 0, 1), b: clamp(leafMid.b * tint, 0, 1) };
        const lSun = { r: clamp(leafSun.r * tint, 0, 1), g: clamp(leafSun.g * tint, 0, 1), b: clamp(leafSun.b * tint, 0, 1) };

        const species = (biome.id === "ALPINE") ? 2 : (i % 3);
        if (species === 0) {
          pushTaperedCylinder(wx, baseY, baseY + 3.4 * sc, wz, 0.34 * sc, 0.18 * sc, trunkDark, trunkMid);
          pushFoliageDome(wx, baseY + 4.8 * sc, wz, 2.35 * sc, 2.0 * sc, lBase, lMid, lSun);
          pushFoliageDome(wx - 1.15 * sc, baseY + 3.9 * sc, wz + 0.65 * sc, 1.55 * sc, 1.35 * sc, lBase, lMid, lSun);
          pushFoliageDome(wx + 1.10 * sc, baseY + 4.1 * sc, wz - 0.70 * sc, 1.50 * sc, 1.35 * sc, lBase, lMid, lSun);
          pushFoliageDome(wx + 0.25 * sc, baseY + 5.9 * sc, wz + 0.35 * sc, 1.45 * sc, 1.25 * sc, lMid, lSun, lSun);
        } else if (species === 1) {
          const topY = baseY + 5.8 * sc;
          pushTaperedCylinder(wx, baseY, baseY + 2.9 * sc, wz, 0.26 * sc, 0.20 * sc, trunkDark, trunkMid, 7);
          pushTaperedCylinder(wx, baseY + 2.9 * sc, topY, wz, 0.20 * sc, 0.14 * sc, trunkMid, trunkDark, 7);
          const frondCount = 7;
          for (let f = 0; f < frondCount; f++) {
            const ang = (f / frondCount) * Math.PI * 2 + (i * 0.5);
            const ca = Math.cos(ang), sa = Math.sin(ang);
            const px = -sa * 0.42 * sc, pz = ca * 0.42 * sc;
            const midX = wx + ca * 1.45 * sc, midY = topY + 0.55 * sc, midZ = wz + sa * 1.45 * sc;
            const tipX = wx + ca * 2.75 * sc, tipY = topY - 0.65 * sc, tipZ = wz + sa * 2.75 * sc;
            pushTri(wx - px * 0.4, topY, wz - pz * 0.4, wx + px * 0.4, topY, wz + pz * 0.4, midX + px, midY, midZ + pz, lBase, lMid, lSun);
            pushTri(wx - px * 0.4, topY, wz - pz * 0.4, midX + px, midY, midZ + pz, midX - px, midY, midZ - pz, lBase, lSun, lSun);
            pushTri(midX - px, midY, midZ - pz, midX + px, midY, midZ + pz, tipX, tipY, tipZ, lSun, lSun, lMid);
          }
        } else {
          pushTaperedCylinder(wx, baseY, baseY + 2.6 * sc, wz, 0.28 * sc, 0.15 * sc, trunkDark, trunkMid);
          pushFoliageDome(wx, baseY + 3.2 * sc, wz, 1.95 * sc, 1.45 * sc, lBase, lMid, lMid);
          pushFoliageDome(wx, baseY + 4.7 * sc, wz, 1.50 * sc, 1.35 * sc, lBase, lMid, lSun);
          pushFoliageDome(wx, baseY + 6.0 * sc, wz, 1.02 * sc, 1.25 * sc, lMid, lSun, lSun);
        }

        if (i % 3 === 0) {
          const rx = wx + (rng() - 0.5) * 6.0;
          const rDeltaZ = (rng() - 0.5) * 6.0;
          const rz = wz + rDeltaZ;
          const ry = WorldMathCore.getTerrainHeightAtS(rx, s - rDeltaZ, fMid.px, fMid.py);
          const rScale = 0.7 + rng() * 1.4;
          pushBox(rx, ry + 0.35 * rScale, rz, 1.5 * rScale, 1.1 * rScale, 1.4 * rScale, rockLight, rockDark);
        }
      }
    }

    // Trackside KM posts & relay cabinets
    const kmPostWhite = hexToRgb(0xf8fafc);
    const kmPostBlack = hexToRgb(0x0f172a);
    const relayGrey = hexToRgb(0x94a3b8);
    const sKm = sStart + 25;
    WorldMathCore.getTrackFrameRaw(sKm, 0, sStart, f1);
    if (
      !f1.inTunnel &&
      WorldMathCore.getViaductFactor(sKm) < 0.01 &&
      WorldMathCore.getViaductFactor(sKm + 15) < 0.01 &&
      !WorldMathCore.isNearStation(sKm, 85)
    ) {
      const kx = f1.px - 3.65;
      const kz = f1.pz;
      const ky = f1.py - 0.52;
      pushBox(kx, ky + 0.55, kz, 0.22, 1.65, 0.14, kmPostWhite, kmPostWhite);
      pushBox(kx, ky + 1.22, kz, 0.24, 0.32, 0.16, kmPostBlack, kmPostWhite);

      const rx = f1.px + 8.0;
      const rz = kz - 12;
      const ry = WorldMathCore.getTerrainHeightAtS(rx, sKm + 12, f1.px + 2.2, f1.py);
      if (ry > biome.waterLevel + 1.5) {
        pushBox(rx, ry + 0.65, rz, 0.9, 1.85, 0.6, relayGrey, rockDark);
      }
    }

    // 5B. Continuous Parallel 2-Lane Toll Highway (Open Embankment + Concrete River Bridge + Mountain Road Tunnel!)
    {
      const stripeYellow = hexToRgb(0xfacc15);
      const guardrailCol = hexToRgb(0xcbd5e1);
      const bridgeConcrete = hexToRgb(0x94a3b8);
      const bridgeGirder = hexToRgb(0x64748b);
      const roadAsphaltTop = hexToRgb(0x1e293b);
      const signGreen = hexToRgb(0x047857);
      const lampGlow = hexToRgb(0xfef08a);

      for (let hs = sStart + 5; hs < sEnd; hs += 10) {
        WorldMathCore.getTrackFrameRaw(hs, 2.2, sStart, fMid);
        const vFac = WorldMathCore.getViaductFactor(hs);
        const nearCx = Math.round(hs / cxInt) * cxInt;
        const nearSt = Math.round(nearCx / stInt) * stInt;
        const atActiveCx = nearCx > 300 && Math.abs(nearCx - nearSt) > 350 && Math.abs(hs - nearCx) < 9.0 && !fMid.inTunnel && vFac <= 0.04;

        const rx = fMid.rx, ry = fMid.ry, rz = fMid.rz;
        const ux = fMid.ux, uy = fMid.uy, uz = fMid.uz;
        const bzx = -fMid.tx, bzy = -fMid.ty, bzz = -fMid.tz;

        const hx = fMid.px + rx * 19.5;
        const hz = fMid.pz + rz * 19.5;
        const hy = WorldMathCore.getHighwayElevation(hs, fMid.py);

        if (fMid.inTunnel) {
          // Parallel 2-Lane Mountain Highway Tunnel Bore (Asphalt floor + concrete curbs + vaulted arch + ceiling LEDs)
          pushBasisBox(hx, hy - 0.12, hz, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 9.6, 0.26, 10.15, roadAsphaltTop, bridgeGirder, true);
          pushBasisBox(hx, hy + 0.03, hz, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.20, 0.04, 4.2, kmPostWhite, kmPostWhite);
          pushBasisBox(hx - rx * 4.55, hy + 0.22, hz - rz * 4.55, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.55, 0.42, 10.15, bridgeConcrete, bridgeGirder);
          pushBasisBox(hx + rx * 4.55, hy + 0.22, hz + rz * 4.55, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.55, 0.42, 10.15, bridgeConcrete, bridgeGirder);
          // Side concrete liner walls & vaulted roof slab
          pushBasisBox(hx - rx * 4.95, hy + 3.2, hz - rz * 4.95, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.65, 6.4, 10.15, bridgeGirder, rockDark);
          pushBasisBox(hx + rx * 4.95, hy + 3.2, hz + rz * 4.95, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.65, 6.4, 10.15, bridgeGirder, rockDark);
          pushBasisBox(hx, hy + 6.35, hz, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 10.55, 0.65, 10.15, rockDark, bridgeGirder, true);
          pushBasisBox(hx - rx * 2.1, hy + 5.98, hz - rz * 2.1, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.28, 0.12, 3.2, lampGlow, lampGlow, true);
          pushBasisBox(hx + rx * 2.1, hy + 5.98, hz + rz * 2.1, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.28, 0.12, 3.2, lampGlow, lampGlow, true);
        } else if (vFac > 0.04) {
          // Parallel Concrete Box-Girder Highway Bridge across River Gorges!
          pushBasisBox(hx, hy - 0.12, hz, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 9.6, 0.26, 10.12, roadAsphaltTop, bridgeConcrete, true);
          pushBasisBox(hx, hy - 0.92, hz, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 7.6, 1.38, 10.12, bridgeGirder, bridgeGirder, true);
          // Center dashed line + white lane edge stripes
          pushBasisBox(hx, hy + 0.03, hz, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.20, 0.04, 4.2, kmPostWhite, kmPostWhite);
          pushBasisBox(hx - rx * 4.05, hy + 0.03, hz - rz * 4.05, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.14, 0.04, 9.95, kmPostWhite, kmPostWhite);
          pushBasisBox(hx + rx * 4.05, hy + 0.03, hz + rz * 4.05, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.14, 0.04, 9.95, stripeYellow, stripeYellow);
          // Concrete New Jersey bridge parapets + steel bridge top rails
          pushBasisBox(hx - rx * 4.58, hy + 0.46, hz - rz * 4.58, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.42, 0.92, 10.12, bridgeConcrete, bridgeGirder);
          pushBasisBox(hx + rx * 4.58, hy + 0.46, hz + rz * 4.58, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.42, 0.92, 10.12, bridgeConcrete, bridgeGirder);
          pushBasisBox(hx - rx * 4.58, hy + 1.12, hz - rz * 4.58, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.12, 0.10, 10.12, guardrailCol, bridgeGirder);
          pushBasisBox(hx + rx * 4.58, hy + 1.12, hz + rz * 4.58, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.12, 0.10, 10.12, guardrailCol, bridgeGirder);

          // Twin Reinforced Concrete River Piers every 20m + Hydrodynamic Waterline Cutwaters!
          if ((((hs - 5) % 20) + 20) % 20 === 0) {
            const groundY = WorldMathCore.getTerrainHeightAtS(hx, hs, fMid.px, fMid.py);
            const pierBotY = Math.min(groundY - 1.8, -13.5);
            const capY = hy - 1.92;
            const pierH = Math.max(2.0, capY - pierBotY);
            const pierMidY = pierBotY + pierH * 0.5;
            pushBasisBox(hx, capY, hz, rx, ry, rz, 0, 1, 0, bzx, 0, bzz, 8.4, 0.85, 2.2, bridgeConcrete, bridgeGirder, true);
            for (const pOff of [-2.65, 2.65]) {
              pushTaperedCylinder(hx + rx * pOff, pierBotY, capY, hz + rz * pOff, 0.88, 0.72, rockDark, bridgeConcrete, 10);
            }
            const riverY = WorldMathCore.getRiverWaterLevel(hs);
            if (groundY < riverY - 0.35) {
              pushBasisBox(hx, riverY - 0.25, hz, rx, 0, rz, 0, 1, 0, bzx, 0, bzz, 8.8, 2.6, 3.4, bridgeConcrete, wetRockCol);
            }
          }
        } else if (!atActiveCx) {
          // Open Embankment Highway Markings & W-Beam Guardrails (oriented tangentially to curves!)
          pushBasisBox(hx, hy + 0.04, hz, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.20, 0.05, 4.2, kmPostWhite, kmPostWhite);
          pushBasisBox(hx - rx * 4.05, hy + 0.03, hz - rz * 4.05, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.14, 0.04, 9.8, kmPostWhite, kmPostWhite);
          pushBasisBox(hx + rx * 4.05, hy + 0.03, hz + rz * 4.05, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.14, 0.04, 9.8, stripeYellow, stripeYellow);
          pushBasisBox(hx - rx * 4.38, hy + 0.38, hz - rz * 4.38, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.26, 0.68, 9.85, guardrailCol, rockDark);
          pushBasisBox(hx + rx * 4.38, hy + 0.38, hz + rz * 4.38, rx, ry, rz, ux, uy, uz, bzx, bzy, bzz, 0.24, 0.64, 9.85, guardrailCol, rockDark);
        }

        // Highway Streetlights every 40m (both on open embankment & across highway river bridges!)
        if (!fMid.inTunnel && !atActiveCx && ((((hs - 5) % 40) + 40) % 40) === 0) {
          for (const side of [-1, 1]) {
            const lxPos = hx + rx * (side * 4.45);
            const lzPos = hz + rz * (side * 4.45);
            pushBasisBox(lxPos, hy + 3.6, lzPos, rx, 0, rz, 0, 1, 0, bzx, 0, bzz, 0.16, 7.2, 0.16, guardrailCol, rockDark);
            const armX = hx + rx * (side * 3.25);
            const armZ = hz + rz * (side * 3.25);
            pushBasisBox(armX, hy + 7.1, armZ, rx, 0, rz, 0, 1, 0, bzx, 0, bzz, 2.5, 0.14, 0.20, guardrailCol, lampGlow, true);
          }
        }
      }

      // Concrete Highway Tunnel Portal Headwalls at Mountain Tunnel Transitions
      for (const tp of tunnelPortals) {
        WorldMathCore.getTrackFrameRaw(tp.s, 2.2, sStart, fMid);
        const rx = fMid.rx, ry = fMid.ry, rz = fMid.rz;
        const bzx = -fMid.tx, bzy = -fMid.ty, bzz = -fMid.tz;
        const hx = fMid.px + rx * 19.5;
        const hz = fMid.pz + rz * 19.5;
        const hy = WorldMathCore.getHighwayElevation(tp.s, fMid.py);
        pushBasisBox(hx - rx * 5.8, hy + 3.6, hz - rz * 5.8, rx, 0, rz, 0, 1, 0, bzx, 0, bzz, 2.4, 7.8, 2.2, bridgeConcrete, bridgeGirder);
        pushBasisBox(hx + rx * 5.8, hy + 3.6, hz + rz * 5.8, rx, 0, rz, 0, 1, 0, bzx, 0, bzz, 2.4, 7.8, 2.2, bridgeConcrete, bridgeGirder);
        pushBasisBox(hx, hy + 7.1, hz, rx, 0, rz, 0, 1, 0, bzx, 0, bzz, 14.2, 1.85, 2.4, kmPostWhite, bridgeConcrete, true);
        pushBasisBox(hx, hy + 6.45, hz, rx, 0, rz, 0, 1, 0, bzx, 0, bzz, 13.8, 0.42, 2.55, signGreen, signGreen, true);
      }

      // Overhead Toll Road Green Directional Sign Gantry (Open Ground)
      if (
        Math.abs(chunkIdx) % 3 === 0 &&
        !WorldMathCore.isNearStation(sStart + 50, 120) &&
        !WorldMathCore.isTunnelZone(sStart + 50) &&
        WorldMathCore.getViaductFactor(sStart + 50) <= 0.04
      ) {
        WorldMathCore.getTrackFrameRaw(sStart + 50, 2.2, sStart, fMid);
        const rx = fMid.rx, rz = fMid.rz;
        const bzx = -fMid.tx, bzz = -fMid.tz;
        const gx = fMid.px + rx * 19.5;
        const gz = fMid.pz + rz * 19.5;
        const gy = WorldMathCore.getHighwayElevation(sStart + 50, fMid.py);
        pushBasisBox(gx - rx * 4.65, gy + 3.4, gz - rz * 4.65, rx, 0, rz, 0, 1, 0, bzx, 0, bzz, 0.28, 6.8, 0.28, guardrailCol, rockDark);
        pushBasisBox(gx + rx * 4.65, gy + 3.4, gz + rz * 4.65, rx, 0, rz, 0, 1, 0, bzx, 0, bzz, 0.28, 6.8, 0.28, guardrailCol, rockDark);
        pushBasisBox(gx, gy + 6.5, gz, rx, 0, rz, 0, 1, 0, bzx, 0, bzz, 9.6, 0.35, 0.35, guardrailCol, rockDark, true);
        pushBasisBox(gx - rx * 2.0, gy + 6.5, gz - rz * 2.0, rx, 0, rz, 0, 1, 0, bzx, 0, bzz, 3.5, 1.5, 0.18, kmPostWhite, signGreen, true);
        pushBasisBox(gx + rx * 2.0, gy + 6.5, gz + rz * 2.0, rx, 0, rz, 0, 1, 0, bzx, 0, bzz, 3.5, 1.5, 0.18, kmPostWhite, signGreen, true);
      }
    }

    // 5C. Riparian Riverbank Boulders (`Batu Kali`) along Meandering River Gorges
    if (hasRiverNearby) {
      for (let b = 0; b < 18; b++) {
        const bs = sStart + rng() * chunkLen;
        WorldMathCore.getTrackFrameRaw(bs, 2.2, sStart, fMid);
        const lx = (rng() > 0.5 ? 1 : -1) * (8.0 + rng() * 140.0);
        if (Math.abs(lx) < 6.5 || (lx > 13.0 && lx < 26.0)) continue;
        const bx = fMid.px + lx;
        const bz = fMid.pz;
        const by = WorldMathCore.getTerrainHeightAtS(bx, bs, fMid.px, fMid.py);
        const rWater = WorldMathCore.getRiverWaterLevel(bs);
        if (by > rWater - 0.45 && by < rWater + 2.4 && WorldMathCore.getViaductFactorAt(bx, bs, fMid.px) > 0.15) {
          const rSc = 0.75 + rng() * 1.65;
          pushBox(bx, by + 0.28 * rSc, bz, 1.7 * rSc, 0.95 * rSc, 1.5 * rSc, rockLight, wetRockCol);
        }
      }
    }

    // SUTET 500kV High-Voltage Transmission Towers
    if (Math.abs(chunkIdx) % 2 === 1 && !WorldMathCore.isNearStation(sStart + 50, 110)) {
      const sSutet = sStart + 40;
      WorldMathCore.getTrackFrameRaw(sSutet, 2.2, sStart, fMid);
      const sx = fMid.px + 74.0;
      const sz = fMid.pz;
      const sy = WorldMathCore.getTerrainHeightAtS(sx, sSutet, fMid.px, fMid.py);
      if (sy > biome.waterLevel + 4) {
        const sutetSteel = hexToRgb(0x94a3b8);
        for (let tier = 0; tier < 6; tier++) {
          const w = 3.8 - tier * 0.45;
          pushBox(sx, sy + 2.5 + tier * 5.0, sz, w, 5.0, w, sutetSteel, rockDark);
        }
        [16.0, 22.0, 28.0].forEach((cyOff, cIdx) => {
          const armSpan = 11.5 - cIdx * 1.8;
          pushBox(sx, sy + cyOff, sz, armSpan, 0.55, 0.65, sutetSteel, rockDark);
          pushBox(sx - armSpan * 0.48, sy + cyOff - 0.9, sz, 0.22, 1.6, 0.22, kmPostWhite, sutetSteel);
          pushBox(sx + armSpan * 0.48, sy + cyOff - 0.9, sz, 0.22, 1.6, 0.22, kmPostWhite, sutetSteel);
        });
      }
    }

    // Multi-Zone Thematic Landmarks
    const zoneType = Math.abs(chunkIdx) % 6;
    const sMid = sStart + 50;
    WorldMathCore.getTrackFrameRaw(sMid, 2.2, sStart, fMid);
    const isOpenGround = !fMid.inTunnel && WorldMathCore.getViaductFactor(sMid) < 0.05 && !WorldMathCore.isNearStation(sMid, 110);

    if (isOpenGround) {
      if (zoneType === 1) {
        const archCol = hexToRgb(0xe11d48);
        const deckCol = hexToRgb(0x64748b);
        const cableCol = hexToRgb(0xf8fafc);
        const ox = fMid.px, oz = fMid.pz, oy = fMid.py;
        pushBox(ox - 16.5, oy + 11.5, oz - 3.9, 2.2, 24.0, 2.2, cableCol, deckCol);
        pushBox(ox - 16.5, oy + 11.5, oz + 3.9, 2.2, 24.0, 2.2, cableCol, deckCol);
        pushBox(ox - 16.5, oy + 19.5, oz, 2.2, 1.6, 9.8, archCol, archCol);
        pushBox(ox + 29.5, oy + 11.5, oz - 3.9, 2.2, 24.0, 2.2, cableCol, deckCol);
        pushBox(ox + 29.5, oy + 11.5, oz + 3.9, 2.2, 24.0, 2.2, cableCol, deckCol);
        pushBox(ox + 29.5, oy + 19.5, oz, 2.2, 1.6, 9.8, archCol, archCol);

        pushBox(ox + 6.5, oy + 7.6, oz, 48.0, 0.95, 8.6, rockDark, deckCol);
        pushBox(ox + 6.5, oy + 8.35, oz - 4.1, 48.0, 0.55, 0.3, archCol, archCol);
        pushBox(ox + 6.5, oy + 8.35, oz + 4.1, 48.0, 0.55, 0.3, archCol, archCol);
        // Sloped approach viaduct ramps + piers anchoring the overpass into both hillsides (no floating road ends!)
        for (let step = 1; step <= 4; step++) {
          const dropY = oy + 7.6 - step * 1.85;
          const lxRamp = ox - 17.5 - (step - 0.5) * 8.0;
          const rxRamp = ox + 30.5 + (step - 0.5) * 8.0;
          pushBox(lxRamp, dropY, oz, 8.4, 0.85, 8.2, rockDark, deckCol);
          pushBox(rxRamp, dropY, oz, 8.4, 0.85, 8.2, rockDark, deckCol);
          pushBox(lxRamp, dropY - 4.5, oz, 1.4, 9.0, 4.6, deckCol, rockDark);
          pushBox(rxRamp, dropY - 4.5, oz, 1.4, 9.0, 4.6, deckCol, rockDark);
        }
        for (let aStep = -2; aStep <= 2; aStep++) {
          const ax = ox + 6.5 + aStep * 7.5;
          const ah = 5.4 - Math.abs(aStep) * 1.1;
          pushBox(ax, oy + 8.2 + ah, oz - 4.1, 7.8, 0.45, 0.45, archCol, archCol);
          pushBox(ax, oy + 8.2 + ah, oz + 4.1, 7.8, 0.45, 0.45, archCol, archCol);
          // Vertical steel hanger cables connecting red arch ribs to bridge deck!
          pushBox(ax, oy + 8.0 + ah * 0.5, oz - 4.1, 0.12, ah, 0.12, cableCol, cableCol);
          pushBox(ax, oy + 8.0 + ah * 0.5, oz + 4.1, 0.12, ah, 0.12, cableCol, cableCol);
        }
      } else if (zoneType === 2) {
        const cRed = hexToRgb(0xdc2626);
        const cBlue = hexToRgb(0x0284c7);
        const cGreen = hexToRgb(0x16a34a);
        const cAmber = hexToRgb(0xd97706);
        const cCols = [cRed, cBlue, cGreen, cAmber];
        const yardX = fMid.px - 28.0;
        const yardZ = fMid.pz;
        const yardY = WorldMathCore.getTerrainHeightAtS(yardX, sMid, fMid.px, fMid.py);
        pushBox(yardX, yardY - 0.3, yardZ, 24.0, 1.4, 65.0, rockLight, rockDark);
        for (let row = 0; row < 3; row++) {
          for (let slot = -2; slot <= 2; slot++) {
            const cx = yardX - 6 + row * 5.2;
            const cz = yardZ + slot * 11.5;
            const c1 = cCols[Math.abs(row + slot) % cCols.length];
            const c1s = { r: c1.r * 0.8, g: c1.g * 0.8, b: c1.b * 0.8 };
            const c2 = cCols[Math.abs(row * 2 + slot + 1) % cCols.length];
            const c2s = { r: c2.r * 0.8, g: c2.g * 0.8, b: c2.b * 0.8 };
            pushBox(cx, yardY + 1.3, cz, 2.5, 2.6, 9.8, c1, c1s);
            if ((row + slot) % 2 === 0) {
              pushBox(cx, yardY + 3.9, cz, 2.5, 2.6, 9.8, c2, c2s);
            }
          }
        }
        pushBox(yardX - 9.5, yardY + 6.0, yardZ, 0.8, 12.0, 0.8, cAmber, rockDark);
        pushBox(yardX + 7.5, yardY + 6.0, yardZ, 0.8, 12.0, 0.8, cAmber, rockDark);
        pushBox(yardX - 1.0, yardY + 11.8, yardZ, 19.5, 1.1, 1.2, cAmber, cRed);
      } else if (zoneType === 3) {
        const solarBlue = hexToRgb(0x1e3a8a);
        const solarFrame = hexToRgb(0x94a3b8);
        [-1, 1].forEach((side, sIdx) => {
          const tx = fMid.px + side * (54 + sIdx * 12);
          const zDelta = (sIdx === 0 ? -15 : 18);
          const tz = fMid.pz + zDelta;
          const ty = WorldMathCore.getTerrainHeightAtS(tx, sMid - zDelta, fMid.px, fMid.py);
          if (ty > biome.waterLevel + 4) {
            windTurbines.push({ x: tx, y: ty, z: tz });
          }
        });
        for (let sr = -2; sr <= 2; sr++) {
          const sx = fMid.px - 24;
          const zDelta = sr * 9.0;
          const sz = fMid.pz + zDelta;
          const sy = WorldMathCore.getTerrainHeightAtS(sx, sMid - zDelta, fMid.px, fMid.py);
          pushBox(sx, sy + 0.9, sz, 6.5, 0.25, 3.6, solarBlue, solarFrame);
        }
      } else if (zoneType === 4) {
        const saungRoof = hexToRgb(0x92400e);
        const saungWood = hexToRgb(0xd97706);
        const teaBushCol = hexToRgb(0x15803d);
        for (let tr = -2; tr <= 2; tr++) {
          const tx = fMid.px - 26 - Math.abs(tr) * 5;
          const zDelta = tr * 14.0;
          const tz = fMid.pz + zDelta;
          const ty = WorldMathCore.getTerrainHeightAtS(tx, sMid - zDelta, fMid.px, fMid.py);
          if (ty > biome.waterLevel + 2) {
            pushBox(tx, ty + 1.1, tz, 3.4, 1.8, 3.4, saungWood, trunkDark);
            pushBox(tx, ty + 2.3, tz, 4.4, 0.7, 4.4, saungRoof, saungWood);
            pushBox(tx + 5.5, ty + 0.5, tz, 4.2, 0.9, 9.5, teaBushCol, leafBase);
          }
        }
      } else if (zoneType === 5) {
        const stadWhite = hexToRgb(0xf8fafc);
        const stadBlue = hexToRgb(0x0284c7);
        const sx = fMid.px + 56.0;
        const sz = fMid.pz;
        const sy = WorldMathCore.getTerrainHeightAtS(sx, sMid, fMid.px, fMid.py);
        if (sy > biome.waterLevel + 3) {
          pushBox(sx, sy + 4.5, sz, 28.0, 9.0, 36.0, stadWhite, stadBlue);
          pushBox(sx, sy + 10.2, sz, 31.0, 2.4, 39.0, stadBlue, stadWhite);
        }
      }
    }

    // Scenic Rural Houses, Ruko, Masjid Agung & BTS Towers
    if (biome.id !== "METROPOLIS") {
      const wallLight = hexToRgb(0xf8fafc);
      const wallShade = hexToRgb(0xcbd5e1);
      const roofTerracotta = hexToRgb(0xc2410c);
      const roofDark = hexToRgb(0x7c2d12);
      const winGlow = hexToRgb(0x38bdf8);
      const doorWood = hexToRgb(0x78350f);
      const domeEmerald = hexToRgb(0x10b981);
      const goldTrim = hexToRgb(0xfacc15);
      const btsRed = hexToRgb(0xef4444);

      for (let h = 0; h < 6; h++) {
        const hs = sStart + 12 + (h * 15) % Math.max(20, chunkLen - 20);
        if (WorldMathCore.getViaductFactor(hs) > 0.08 || WorldMathCore.isNearStation(hs, 95)) continue;
        WorldMathCore.getTrackFrameRaw(hs, 2.2, sStart, fMid);
        const hSide = h % 2 === 0 ? -1 : 1;
        const hOff = hSide * (32 + (h * 13) % 48);
        const hx = fMid.px + hOff;
        const hz = fMid.pz;
        const hy = WorldMathCore.getTerrainHeightAtS(hx, hs, fMid.px, fMid.py);
        if (hy < biome.waterLevel + 2.0) continue;

        const faceDir = hSide < 0 ? 1 : -1;
        if (h === 1) {
          pushBox(hx, hy + 2.2, hz, 8.2, 4.4, 8.2, wallLight, wallShade);
          pushBox(hx, hy + 4.6, hz, 8.8, 0.45, 8.8, goldTrim, wallShade);
          pushBox(hx, hy + 5.6, hz, 5.2, 1.6, 5.2, domeEmerald, domeEmerald);
          pushBox(hx, hy + 6.8, hz, 3.2, 1.2, 3.2, domeEmerald, goldTrim);
          pushBox(hx, hy + 7.9, hz, 0.35, 1.4, 0.35, goldTrim, goldTrim);
          pushBox(hx + faceDir * 5.2, hy + 5.5, hz - 4.2, 1.6, 11.0, 1.6, wallLight, wallShade);
          pushBox(hx + faceDir * 5.2, hy + 11.6, hz - 4.2, 1.2, 1.6, 1.2, domeEmerald, goldTrim);
        } else if (h === 3) {
          pushBox(hx, hy + 4.8, hz, 5.8, 9.6, 7.4, wallLight, wallShade);
          pushBox(hx + faceDir * 3.1, hy + 2.8, hz, 0.85, 0.35, 7.0, btsRed, roofDark);
          pushBox(hx + faceDir * 2.95, hy + 4.8, hz, 0.18, 1.2, 5.6, winGlow, winGlow);
          pushBox(hx + faceDir * 2.95, hy + 7.6, hz, 0.18, 1.2, 5.6, winGlow, winGlow);
          pushBox(hx - faceDir * 1.2, hy + 10.4, hz - 1.5, 1.3, 1.6, 1.3, roofTerracotta, roofDark);
        } else {
          pushBox(hx, hy + 1.8, hz, 5.8, 3.6, 6.6, wallLight, wallShade);
          pushBox(hx, hy + 3.9, hz, 6.8, 0.9, 7.6, roofTerracotta, roofDark);
          pushBox(hx, hy + 4.6, hz, 4.4, 0.7, 5.4, roofTerracotta, roofDark);
          pushBox(hx + faceDir * 3.4, hy + 2.6, hz, 1.4, 0.25, 4.2, roofDark, roofTerracotta);
          pushBox(hx + faceDir * 2.92, hy + 1.2, hz, 0.15, 2.1, 1.1, doorWood, doorWood);
          pushBox(hx + faceDir * 2.92, hy + 1.9, hz - 1.8, 0.15, 1.1, 1.3, winGlow, winGlow);
          pushBox(hx + faceDir * 2.92, hy + 1.9, hz + 1.8, 0.15, 1.1, 1.3, winGlow, winGlow);
        }
      }

      if (Math.abs(chunkIdx) % 2 === 0) {
        const bs = sStart + 72;
        if (WorldMathCore.getViaductFactor(bs) < 0.05 && !WorldMathCore.isNearStation(bs, 110)) {
          WorldMathCore.getTrackFrameRaw(bs, 2.2, sStart, fMid);
          const bx = fMid.px - 62;
          const bz = fMid.pz;
          const by = WorldMathCore.getTerrainHeightAtS(bx, bs, fMid.px, fMid.py);
          if (by > biome.waterLevel + 4) {
            for (let sec = 0; sec < 5; sec++) {
              const w = 2.6 - sec * 0.38;
              const c = sec % 2 === 0 ? btsRed : wallLight;
              pushBox(bx, by + 2.5 + sec * 5.0, bz, w, 5.0, w, c, c);
            }
            pushBox(bx, by + 25.5, bz, 3.2, 0.45, 3.2, wallLight, wallShade);
          }
        }
      }
    }

    // Pack into typed arrays & precompute smooth normals (Uint32Array ensures 4-byte WebGPU buffer alignment!)
    const bPos = new Float32Array(ballastVerts);
    const bUV = new Float32Array(ballastUVs);
    const bCol = new Float32Array(ballastCols);
    const bIdx = new Uint32Array(ballastIdx);
    const bNrm = computeIndexedNormals(bPos, bIdx);

    const rPos = new Float32Array(railVerts);
    const rCol = new Float32Array(railCols);
    const rIdx = new Uint32Array(railIdx);
    const rNrm = computeIndexedNormals(rPos, rIdx);

    const wPos = new Float32Array(wireVerts);
    const wIdx = new Uint32Array(wireIdx);

    const tPos = new Float32Array(terrainVerts);
    const tUV = new Float32Array(terrainUVs);
    const tCol = new Float32Array(terrainCols);
    const tIdx = new Uint32Array(terrainIdx);
    const tNrm = computeIndexedNormals(tPos, tIdx);

    const watPos = new Float32Array(waterVerts);
    const watUV = new Float32Array(waterUVs);
    const watCol = new Float32Array(waterCols);
    const watIdx = new Uint32Array(waterIdx);
    const watNrm = watPos.length > 0 ? computeIndexedNormals(watPos, watIdx) : new Float32Array(0);

    const sPos = new Float32Array(batchVerts);
    const sCol = new Float32Array(batchCols);
    const sNrm = sPos.length > 0 ? computeNonIndexedNormals(sPos) : new Float32Array(0);

    return {
      chunkIdx,
      sStart,
      sEnd,
      biomeId: biome.id,
      biomeGroundColor: biome.groundColor,
      ballast: { pos: bPos, uv: bUV, col: bCol, idx: bIdx, nrm: bNrm },
      rails: { pos: rPos, col: rCol, idx: rIdx, nrm: rNrm },
      wires: { pos: wPos, idx: wIdx },
      terrain: { pos: tPos, uv: tUV, col: tCol, idx: tIdx, nrm: tNrm },
      water: { pos: watPos, uv: watUV, col: watCol, idx: watIdx, nrm: watNrm },
      scenery: { pos: sPos, col: sCol, nrm: sNrm },
      tunnelSpans,
      tunnelPortals,
      viaductSpans,
      bridgeAbutments,
      catenarySpans,
      metropolisBuildings,
      windTurbines
    };
  }
}
