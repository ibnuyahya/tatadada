import { GAME_PREMISE_CARTRIDGE } from '../data/PremiseCartridge.js';

/**
 * ============================================================================
 * PURE FLOAT64 ANALYTICAL SPLINE, TUNNEL, VIADUCT & TERRAIN CORE
 * Zero DOM / Zero Three.js dependencies so Web Workers can execute at full
 * native V8 JIT speed with zero allocations and 64-bit precision.
 * ============================================================================
 */
function clamp(v, min, max) {
  return v < min ? min : (v > max ? max : v);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

const _xyTmp0 = { x: 0, y: 0 };
const _xyTmp1 = { x: 0, y: 0 };
const _xyTmp2 = { x: 0, y: 0 };

export class WorldMathCore {
  static getBiomeAtDistance(s) {
    const span = 4400; // 4.4 km per biome
    const biomes = GAME_PREMISE_CARTRIDGE.biomes;
    const idx = Math.floor(Math.max(0, s) / span) % biomes.length;
    return { biome: biomes[idx], index: idx };
  }

  static getStationFlattenMask(s) {
    const interval = GAME_PREMISE_CARTRIDGE.railway.stationInterval;
    const stIdx = Math.max(0, Math.round(s / interval));
    const stCenter = stIdx * interval;
    const d = Math.abs(s - stCenter);
    if (d < 255) return 0.0;
    if (d < 495) {
      const t = (d - 255) / 240;
      return t * t * (3 - 2 * t);
    }
    return 1.0;
  }

  static isNearStation(s, radius = 95) {
    const interval = GAME_PREMISE_CARTRIDGE.railway.stationInterval;
    const stIdx = Math.max(0, Math.round(s / interval));
    // Extended 276m HSR platform spans z_st in [-86m, +190m] => s in [stCenter - 190m, stCenter + 86m] (midpoint stCenter - 52)
    const platMidS = stIdx * interval - 52;
    return Math.abs(s - platMidS) < (radius + 62);
  }

  static _evalCenterlineXY(s, out) {
    const stMask = this.getStationFlattenMask(s);
    if (stMask <= 0.00001) {
      out.x = 0.0;
      out.y = 3.2;
      return out;
    }
    // Realistic inter-station HSR sweeping S-curves (R ~ 3,800m - 6,800m) & Priangan ruling grades (±15‰)
    const xRaw = Math.sin(s * 0.0022) * 28.0 + Math.sin(s * 0.0041 + 1.1) * 8.5;
    const yClimb = (0.5 - 0.5 * Math.cos(s * 0.0023)) * 9.4 + Math.sin(s * 0.0049 + 0.6) * 1.25;
    out.x = stMask * xRaw;
    out.y = 3.2 + stMask * yClimb;
    return out;
  }

  /**
   * Evaluates 64-bit world X, Y and local Z (relative to `originS`).
   * Writes into `out` object `{ x, y, z }` to avoid heap allocations.
   */
  static getTrackPositionRaw(s, lateralOffset = 0, originS = 0, out = { x: 0, y: 0, z: 0 }) {
    this._evalCenterlineXY(s, _xyTmp1);
    out.x = _xyTmp1.x + lateralOffset;
    out.y = _xyTmp1.y;
    out.z = -(s - originS);
    return out;
  }

  static isTunnelZone(s) {
    if (this.getStationFlattenMask(s) < 0.98 || s < 550) return false;
    const wave = Math.sin(s * 0.0042 - 0.8);
    return wave > 0.75;
  }

  static getViaductFactor(s) {
    const stMask = this.getStationFlattenMask(s);
    if (stMask < 0.5 || s < 320) return 0.0;
    // Mutually exclude viaduct gorges from mountain tunnel zones & portal approaches!
    const tunWave = Math.sin(s * 0.0042 - 0.8);
    if (tunWave >= 0.64) return 0.0;
    const tunClear = tunWave > 0.44 ? clamp((0.64 - tunWave) / 0.20, 0.0, 1.0) : 1.0;

    const wave = Math.sin(s * 0.0058 + 1.4);
    if (wave <= 0.42) return 0.0;
    const t = clamp((wave - 0.42) / 0.26, 0.0, 1.0);
    return t * t * (3 - 2 * t) * stMask * tunClear;
  }

  static isViaductZone(s) {
    return this.getViaductFactor(s) > 0.50;
  }

  static getRiverWaterLevel(s) {
    const { biome } = this.getBiomeAtDistance(s);
    if (biome.id === 'COASTAL') return biome.waterLevel;
    return -7.8;
  }

  /**
   * Evaluates river/gorge factor at world `(x, s)`. Keeps the railway + parallel highway
   * bridge corridor (`|x - tcx| <= 32m`) aligned squarely with `s`, while meandering
   * naturally across the left/right valley so rivers wind organically instead of straight trenches.
   */
  static getViaductFactorAt(x, s, tcx = 0) {
    const absCorridor = Math.abs(x - tcx);
    if (absCorridor <= 32.0) return this.getViaductFactor(s);
    const meanderBlend = clamp((absCorridor - 32.0) / 36.0, 0.0, 1.0);
    const sShift = (Math.sin(x * 0.011) * 18.0 + Math.cos(x * 0.0047) * 9.0) * meanderBlend;
    return this.getViaductFactor(s + sShift);
  }

  /**
   * Continuous road-deck elevation for the parallel 2-lane highway (`x = fMid.px + 19.5`).
   * Carries road vehicles level across parallel concrete river bridges and through
   * parallel mountain highway tunnels without plunging into gorges or climbing cliffs.
   */
  static getHighwayElevation(s, trackCenterY) {
    let tcy = trackCenterY;
    if (tcy === undefined) {
      this._evalCenterlineXY(s, _xyTmp1);
      tcy = _xyTmp1.y;
    }
    return tcy - 0.54;
  }

  /**
   * Computes full Frenet-like railway frame at 64-bit distance `s`,
   * outputting local Z coordinates relative to `originS` (`chunkStartS` or `floatingOrigin.originS`).
   */
  static getTrackFrameRaw(s, lateralOffset = 0, originS = 0, out = null) {
    const ds = 1.0;

    this._evalCenterlineXY(s - ds, _xyTmp0);
    const x0 = _xyTmp0.x, y0 = _xyTmp0.y;
    const z0 = -((s - ds) - originS);

    this._evalCenterlineXY(s, _xyTmp1);
    const x1 = _xyTmp1.x, y1 = _xyTmp1.y;
    const z1 = -(s - originS);

    this._evalCenterlineXY(s + ds, _xyTmp2);
    const x2 = _xyTmp2.x, y2 = _xyTmp2.y;
    const z2 = -((s + ds) - originS);

    // Tangent = normalize(p2 - p0)
    let tx = x2 - x0;
    let ty = y2 - y0;
    let tz = z2 - z0;
    const tLen = Math.hypot(tx, ty, tz) || 1.0;
    tx /= tLen; ty /= tLen; tz /= tLen;

    // Right = normalize(tangent x worldUp(0,1,0)) = (-tz, 0, tx)
    let rx = -tz;
    let ry = 0.0;
    let rz = tx;
    const rLen = Math.hypot(rx, rz) || 1.0;
    rx /= rLen; rz /= rLen;

    // Up = normalize(right x tangent)
    let ux = ry * tz - rz * ty;
    let uy = rz * tx - rx * tz;
    let uz = rx * ty - ry * tx;
    const uLen = Math.hypot(ux, uy, uz) || 1.0;
    ux /= uLen; uy /= uLen; uz /= uLen;

    const px = x1 + rx * lateralOffset;
    const py = y1 + ry * lateralOffset;
    const pz = z1 + rz * lateralOffset;

    // Curvature in XZ plane (+curvature = curving right toward +X while traveling -Z)
    let v1x = x1 - x0, v1z = z1 - z0;
    const v1L = Math.hypot(v1x, v1z) || 1.0;
    v1x /= v1L; v1z /= v1L;

    let v2x = x2 - x1, v2z = z2 - z1;
    const v2L = Math.hypot(v2x, v2z) || 1.0;
    v2x /= v2L; v2z /= v2L;

    const curvature = (v1x * v2z - v1z * v2x) / ds;

    // UIC inward superelevation (negative Z-roll leans roof into +X right-hand curve; eased to 0 across level crossings)
    const cxInt = GAME_PREMISE_CARTRIDGE.railway.crossingInterval;
    const dCx = Math.abs(s - Math.round(s / cxInt) * cxInt);
    const cxCantMask = dCx < 25 ? 0.0 : (dCx < 65 ? (dCx - 25) / 40 : 1.0);
    const cantAngle = clamp(-curvature * 240.0 * cxCantMask, -0.068, 0.068);
    const gradePermille = ((y2 - y0) / (2 * ds)) * 1000;

    const { biome } = this.getBiomeAtDistance(s);
    const speedLimit = biome.speedLimit;
    const inTunnel = this.isTunnelZone(s);

    if (!out) {
      return {
        px, py, pz,
        tx, ty, tz,
        rx, ry, rz,
        ux, uy, uz,
        curvature,
        cantAngle,
        gradePermille,
        speedLimit,
        inTunnel
      };
    }
    out.px = px; out.py = py; out.pz = pz;
    out.tx = tx; out.ty = ty; out.tz = tz;
    out.rx = rx; out.ry = ry; out.rz = rz;
    out.ux = ux; out.uy = uy; out.uz = uz;
    out.curvature = curvature;
    out.cantAngle = cantAngle;
    out.gradePermille = gradePermille;
    out.speedLimit = speedLimit;
    out.inTunnel = inTunnel;
    return out;
  }

  /**
   * Computes terrain height in 64-bit world space (`worldS = -worldZ`).
   * Note: Always pass true 64-bit longitudinal `s` (NOT rebased local Z) so terrain
   * noise phases never shift during Floating-Origin rebasing!
   */
  static getTerrainHeightAtS(x, s, trackCenterX, trackCenterY) {
    const z = -s; // True 64-bit world Z for continuous multi-octave noise
    let tcx = trackCenterX;
    let tcy = trackCenterY;
    if (tcx === undefined || tcy === undefined) {
      this._evalCenterlineXY(s, _xyTmp1);
      tcx = _xyTmp1.x + 2.2;
      tcy = _xyTmp1.y;
    }

    const signedLat = x - tcx;
    const distFromCorridor = Math.abs(signedLat);
    const { biome } = this.getBiomeAtDistance(s);
    const vFactor = this.getViaductFactorAt(x, s, tcx);
    const inTunnel = this.isTunnelZone(s);

    const flatRightLimit = 26.5; // Covers double-track corridor + full 2-lane parallel toll road [15..25m]
    const inFlatCorridor = (signedLat >= -16.5 && signedLat <= flatRightLimit);
    const trackBedY = tcy - 0.54;
    if (!inTunnel && inFlatCorridor && vFactor < 0.01) {
      return trackBedY;
    }

    const n1 = Math.sin(x * 0.0042 + z * 0.0026) * Math.cos(x * 0.0032 - z * 0.0038);
    const ridge = 1.0 - Math.abs(Math.sin(x * 0.0055 - z * 0.0040));
    const mediumHills = (Math.cos(x * 0.012 - z * 0.008) * 0.5 + 0.5) * 12.5 + Math.sin(x * 0.024 + z * 0.019) * 1.8;
    const microDetail = Math.cos(x * 0.055 + z * 0.048) * 0.35;

    let hillHeight =
      tcy - 0.5 +
      Math.abs(n1) * 22.0 +
      ridge * ridge * 24.0 +
      mediumHills +
      microDetail;

    if (biome.id === "PRIANGAN" && distFromCorridor > 52.0 && hillHeight > tcy + 4.0 && hillHeight < tcy + 22.0) {
      const stepSize = 2.0;
      const terraced = Math.floor(hillHeight / stepSize) * stepSize + 0.55 * (hillHeight % stepSize);
      const terraceBlend = clamp((distFromCorridor - 52.0) / 28.0, 0, 0.55);
      hillHeight = lerp(hillHeight, terraced, terraceBlend);
    }

    if (biome.id === "COASTAL" && signedLat > 48.0) {
      const oceanT = clamp((signedLat - 48.0) / 55.0, 0, 1);
      hillHeight = lerp(hillHeight, biome.waterLevel - 4.2, oceanT * oceanT);
    }

    // Approach valley ridge & full mountain overburden directly above bored HSR + Highway twin tunnels
    const tunWave = (this.getStationFlattenMask(s) >= 0.98 && s >= 550) ? Math.sin(s * 0.0042 - 0.8) : -1.0;
    const edgeStart = signedLat > 0 ? flatRightLimit : 16.5;
    const distPastBench = Math.max(0.0, distFromCorridor - edgeStart);

    if (tunWave > 0.58 && distPastBench > 0.0) {
      const flankT = clamp((tunWave - 0.58) / 0.17, 0, 1);
      const latBlend = clamp(distPastBench / 28.0, 0, 1);
      hillHeight += 18.0 * flankT * latBlend;
    }

    let finalY = hillHeight;
    if (!inTunnel) {
      if (inFlatCorridor) {
        finalY = trackBedY;
      } else if (distPastBench < 81.5) {
        const t = distPastBench / 81.5;
        const smooth = t * t * t * (t * (t * 6 - 15) + 10);
        finalY = lerp(trackBedY, hillHeight, smooth);
      }
    } else {
      // Directly over both the twin-track HSR tunnel [tcx] and parallel highway tunnel [tcx + 19.5],
      // guarantee solid mountain rock overburden sits cleanly above both tunnel crowns!
      const tunDepth = clamp((tunWave - 0.75) / 0.25, 0.0, 1.0);
      const centerSpan = signedLat > 10.0 ? Math.abs(signedLat - 19.5) : distFromCorridor;
      const crownRidge = tcy + 10.4 + Math.sin(tunDepth * Math.PI * 0.5) * 18.5 + Math.cos(x * 0.05) * 0.9;
      if (centerSpan < 48.0) {
        const ridgeBlend = 1.0 - clamp(centerSpan / 48.0, 0.0, 1.0);
        const minRidgeY = lerp(tcy + 6.5, crownRidge, ridgeBlend * ridgeBlend * (3 - 2 * ridgeBlend));
        finalY = Math.max(hillHeight, minRidgeY);
      }
    }

    if (vFactor > 0.001 && !inTunnel) {
      // Natural river cross-section: deeper central thalweg (-12.2m) + cobble/gravel riparian shelves
      const channelProfile = vFactor > 0.58 ? clamp((vFactor - 0.58) / 0.42, 0.0, 1.0) : 0.0;
      const gorgeFloor = -9.4 - channelProfile * 2.8 + Math.sin(x * 0.06 + z * 0.04) * 0.65;
      finalY = lerp(finalY, gorgeFloor, vFactor);
    }

    return finalY;
  }
}
