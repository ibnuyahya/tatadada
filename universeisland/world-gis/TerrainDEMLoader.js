import * as THREE from 'three';
import { NexusBatchOptimizer } from '../core/BatchRenderer.js';
import { PhysicsKernel } from '../core/PhysicsKernel.js';
import {
  SURFACE_PROFILE,
  createSharedSurfaceUniforms,
  injectOmniScaleSurfaceShader
} from '../core/ProceduralTextureFactory.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v3.0 — Kepulauan UI: Pulau "U" & Pulau "I" Edition)
 * ./world-gis/TerrainDEMLoader.js
 *
 * Dual-Island "U" & "I" Archipelago SSoST, Dual High-Density DEMs with -18.0m Watertight Skirts,
 * Dual Closed 360° Coursed River-Stone Revetments (Turap Batu Keliling Pulau U & Pulau I)
 * with Radial Masonry Buttress Piers & Waterside Bollards, Navigable Bridge Strait (Selat UI),
 * Inner Lagoon of Pulau U (Teluk U), and 3.3 km Surrounding Ocean Horizon
 * (Strictly 2 Draw Calls Total: 1 Batched Archipelago Land+Skirts+Turap, 1 Ocean Surface!).
 */
export class TerrainDEMLoader {
  // Campus Head Anchor on Left Arm of Pulau "U"
  static ISLAND_CX = -18.0;
  static ISLAND_CZ = 34.0;
  static GRID_HALF_SPAN = 166.0;
  static TERRAIN_HALF_SIZE = 166.0;

  // Archipelago Center & 3.3 km Ocean Horizon Radius
  static ARCHIPELAGO_CX = -155.0;
  static ARCHIPELAGO_CZ = 330.0;
  static WATER_OUTER_RADIUS = 1650.0;

  static SUB_TERRAIN_SKIRT_Y = -18.0;
  static WATER_LEVEL_Y = -1.45;
  static LAKE_BED_Y = -3.85;

  // Pulau "U" Horseshoe SSoST Parameters (Upright when +X = Up, +Z = Right)
  static U_ARC_X = -145.0;
  static U_LEFT_Z = 22.0;
  static U_RIGHT_Z = 392.0;
  static U_ARC_Z = 207.0; // 0.5 * (U_LEFT_Z + U_RIGHT_Z)
  static U_ARC_R = 185.0; // 0.5 * (U_RIGHT_Z - U_LEFT_Z)
  static U_TOP_LEFT_X = -62.0;
  static U_TOP_RIGHT_X = 10.0; // Top tip reaches +10 + 74 = +84.0m (matches Left Arm top!)
  static U_ARM_HALF_W = 74.0;
  static U_BLEND_K = 28.0;

  // Pulau "I" Volcanic & Airstrip Island SSoST Parameters (Across 140m Navigable Bridge Strait Selat UI)
  static I_CENTER_X = -160.0;
  static I_CENTER_Z = 694.0;
  static I_HALF_LEN_X = 244.0; // Spans x in [-404.0m, +84.0m] = 488.0m tall (matches Pulau U!)
  static I_HALF_WID_Z = 88.0;  // Spans z in [+606.0m, +782.0m] = 176.0m wide (Selat UI stays 140m wide!)
  static I_CORNER_R = 56.0;

  // Active Stratovolcano ("Gunung Api Pulau I") SSoST Parameters (Northern Section of Pulau I)
  static VOLCANO_CX = -32.0;
  static VOLCANO_CZ = 698.0;
  static VOLCANO_PEAK_RIM_Y = 58.5;
  static VOLCANO_CRATER_FLOOR_Y = 44.5;
  static VOLCANO_RIM_R = 19.5;
  static VOLCANO_BASE_R = 98.0;

  // Pulau "I" Mini-Niagara Horseshoe Waterfall SSoST ("Air Terjun Niagara Mini" on West-SW Volcano Flank)
  static NIAGARA_CX = -44.0;
  static NIAGARA_CZ = 643.0;
  static NIAGARA_LIP_R = 15.6;
  static NIAGARA_HALF_ANGLE = 1.18; // ~135° concave horseshoe brink (~36.8m curved crest!)
  static NIAGARA_UPPER_WATER_Y = 18.80;
  static NIAGARA_POOL_WATER_Y = 1.80;

  // Straight ICAO Island Runway / Airstrip ("Landasan Pacu Pulau I — RWY 36/18") SSoST Parameters
  static RUNWAY_CZ = 708.0;
  static RUNWAY_X_MIN = -372.0;
  static RUNWAY_X_MAX = -138.0;
  static RUNWAY_HALF_W = 11.0;
  static RUNWAY_DECK_Y = 0.58;

  // VIP Waterfront Helipad Peninsula Coordinates (Inner Lagoon Headland on Left Arm of Pulau U)
  static HELIPAD_CX = -104.0;
  static HELIPAD_CZ = 106.5;
  static HELIPAD_DECK_Y = 0.48;
  static HELIPAD_OUTER_R = 16.3;

  // Jembatan Teksas UI ("Teknik - Sastra" Krakatau Steel BTKC Arch Bridge across Teluk U) SSoST Parameters:
  // Connects Left Arm (FT / Rotunda East Shore, Pilon Lingga 25m) <-> Right Arm (FIB / Sastra Shore, Pilon Yoni 15m)
  // with >= 9.25m clear mid-span air-draft headroom above WATER_LEVEL_Y (-1.45m) for Flybridge Mini Yachts!
  static TEKSAS_BRIDGE_X = -28.0;
  static TEKSAS_WEST_PORTAL_Z = 171.65;
  static TEKSAS_EAST_PORTAL_Z = 319.65;
  static TEKSAS_SPAN_CENTER_Z = 245.65;
  static TEKSAS_HALF_SPAN_Z = 74.0;
  static TEKSAS_DECK_HALF_W = 2.60;
  static TEKSAS_PORTAL_DECK_Y = 4.15;
  static TEKSAS_ARCH_RISE_Y = 4.95; // Mid-span deck peak = +9.10m (+10.55m above water)
  static TEKSAS_GIRDER_DEPTH_M = 1.30; // Mid-span lowest soffit = +7.80m (+9.25m net clearance above water!)
  static TEKSAS_WEST_RAMP_START_Z = 149.65;
  static TEKSAS_EAST_RAMP_END_Z = 341.65;
  static TEKSAS_EAST_PLAZA_Y = 0.34;

  /**
   * Evaluates exact walkable deck/ramp elevation Y on Jembatan Teksas UI at (x, z), or null if outside bridge footprint.
   */
  static sampleTeksasBridgeDeckY(x, z) {
    const dx = Math.abs(x - TerrainDEMLoader.TEKSAS_BRIDGE_X);
    if (dx > TerrainDEMLoader.TEKSAS_DECK_HALF_W + 0.25) return null;

    // 1. Main 7-Box-Segment Parabolic Clear-Span Arch across Teluk U (z in [171.65, 319.65])
    if (
      z >= TerrainDEMLoader.TEKSAS_WEST_PORTAL_Z &&
      z <= TerrainDEMLoader.TEKSAS_EAST_PORTAL_Z
    ) {
      if (dx > TerrainDEMLoader.TEKSAS_DECK_HALF_W + 0.05) return null;
      const u =
        (z - TerrainDEMLoader.TEKSAS_SPAN_CENTER_Z) / TerrainDEMLoader.TEKSAS_HALF_SPAN_Z;
      return (
        TerrainDEMLoader.TEKSAS_PORTAL_DECK_Y +
        TerrainDEMLoader.TEKSAS_ARCH_RISE_Y * Math.max(0.0, 1.0 - u * u)
      );
    }

    // 2. West Approach Ramp & Ceremonial Staircase (FT / Rotunda Side: z in [149.65, 171.65))
    if (
      z >= TerrainDEMLoader.TEKSAS_WEST_RAMP_START_Z &&
      z < TerrainDEMLoader.TEKSAS_WEST_PORTAL_Z
    ) {
      const t =
        (z - TerrainDEMLoader.TEKSAS_WEST_RAMP_START_Z) /
        (TerrainDEMLoader.TEKSAS_WEST_PORTAL_Z - TerrainDEMLoader.TEKSAS_WEST_RAMP_START_Z);
      const s = PhysicsKernel.smootherstep(0.0, 1.0, t);
      return 0.04 + s * (TerrainDEMLoader.TEKSAS_PORTAL_DECK_Y - 0.04);
    }

    // 3. East Approach Ramp & Ceremonial Staircase (FIB / Right-Arm Side: z in (319.65, 341.65])
    if (
      z > TerrainDEMLoader.TEKSAS_EAST_PORTAL_Z &&
      z <= TerrainDEMLoader.TEKSAS_EAST_RAMP_END_Z
    ) {
      const t =
        (TerrainDEMLoader.TEKSAS_EAST_RAMP_END_Z - z) /
        (TerrainDEMLoader.TEKSAS_EAST_RAMP_END_Z - TerrainDEMLoader.TEKSAS_EAST_PORTAL_Z);
      const s = PhysicsKernel.smootherstep(0.0, 1.0, t);
      return (
        TerrainDEMLoader.TEKSAS_EAST_PLAZA_Y +
        s * (TerrainDEMLoader.TEKSAS_PORTAL_DECK_Y - TerrainDEMLoader.TEKSAS_EAST_PLAZA_Y)
      );
    }

    return null;
  }

  /**
   * Evaluates lowest under-bridge steel box-girder soffit elevation Y(z) across Teluk U (for Mini-Yacht air-draft verification).
   */
  static sampleTeksasBridgeSoffitY(z) {
    if (
      z < TerrainDEMLoader.TEKSAS_WEST_PORTAL_Z ||
      z > TerrainDEMLoader.TEKSAS_EAST_PORTAL_Z
    ) {
      return null;
    }
    const u =
      (z - TerrainDEMLoader.TEKSAS_SPAN_CENTER_Z) / TerrainDEMLoader.TEKSAS_HALF_SPAN_Z;
    const deckY =
      TerrainDEMLoader.TEKSAS_PORTAL_DECK_Y +
      TerrainDEMLoader.TEKSAS_ARCH_RISE_Y * Math.max(0.0, 1.0 - u * u);
    return deckY - TerrainDEMLoader.TEKSAS_GIRDER_DEPTH_M;
  }

  // Pulau "U" Freshwater Hydrology SSoST:
  // Southern Highland Forest Spring ("Mata Air") -> Meandering Northbound River -> Small Lake near Rektorat
  static SPRING_U_CX = -286.0;
  static SPRING_U_CZ = 48.0;
  static SPRING_U_R = 9.4;
  static SPRING_U_WATER_Y = 11.20;
  static RIVER_U_X_START = -286.0;
  static RIVER_U_X_DROP_START = -275.5;
  static RIVER_U_X_END = -95.0;
  static LAKE_U_CX = -82.5;
  static LAKE_U_CZ = -22.0;
  static LAKE_U_RX = 11.8;
  static LAKE_U_RZ = 21.5;
  static LAKE_U_WATER_Y = -0.22;

  /**
   * Compact-support C1 quadratic smooth-minimum (identical in JS & GLSL).
   * Zero perturbation whenever |a - b| >= k!
   */
  static _smin(a, b, k) {
    const h = Math.max(0.0, k - Math.abs(a - b)) / k;
    return Math.min(a, b) - h * h * k * 0.25;
  }

  /**
   * SSoST Polar Radius R(theta) for the Existing Kampus Rektorat Head (Upper Left Arm of Pulau U)
   * around (ISLAND_CX = -18.0, ISLAND_CZ = 34.0).
   */
  static getIslandRadiusAtAngle(theta) {
    const eastCove = -46.0 * Math.pow(0.5 + 0.5 * Math.cos(theta + 0.25), 4.4);
    const neHeadland = 23.5 * Math.pow(0.5 + 0.5 * Math.cos(theta + 1.04), 8.5);
    const southRotundaLobe = 15.5 * Math.pow(0.5 + 0.5 * Math.cos(theta - 1.12), 2.3);
    const swHelipadPeninsula = 20.5 * Math.pow(0.5 + 0.5 * Math.cos(theta - 2.44), 6.5);
    const organicMeander =
      2.15 * Math.sin(3.0 * theta + 0.40) +
      1.20 * Math.cos(5.0 * theta - 0.70) +
      0.55 * Math.sin(8.0 * theta + 1.10);
    return 121.0 + eastCove + neHeadland + southRotundaLobe + swHelipadPeninsula + organicMeander;
  }

  /**
   * Evaluates perpendicular signed coast distance dU(x, z) for Pulau "U" (Rektorat + Teluk U + Right Arm):
   * - dU < 0.0 : Inland on Pulau U
   * - dU = 0.0 : Exact 360° Stone Turap Coping Perimeter of Pulau U
   * - dU > 0.0 : Outward into surrounding ocean or inner lagoon Teluk U
   */
  static getIslandUCoastDistance(x, z) {
    // 1. Campus Head on Upper Left Arm of Pulau U (true perpendicular Euclidean correction)
    const dx = x - TerrainDEMLoader.ISLAND_CX;
    const dz = z - TerrainDEMLoader.ISLAND_CZ;
    const r = Math.hypot(dx, dz) + 1e-5;
    const theta = Math.atan2(dz, dx);
    const r0 = TerrainDEMLoader.getIslandRadiusAtAngle(theta);
    const epsTh = 0.008;
    const dRdTh =
      (TerrainDEMLoader.getIslandRadiusAtAngle(theta + epsTh) -
        TerrainDEMLoader.getIslandRadiusAtAngle(theta - epsTh)) /
      (2.0 * epsTh);
    const slopeRatio = dRdTh / Math.max(20.0, r0);
    const dCampusHead = (r - r0) / Math.sqrt(1.0 + slopeRatio * slopeRatio);

    // 2. Horseshoe "U" Arch Spine (Left Leg -> Semicircular U-Bottom -> Right Peninsula Arm)
    let distUCenter;
    if (x <= TerrainDEMLoader.U_ARC_X) {
      distUCenter = Math.abs(
        Math.hypot(x - TerrainDEMLoader.U_ARC_X, z - TerrainDEMLoader.U_ARC_Z) -
          TerrainDEMLoader.U_ARC_R
      );
    } else {
      const dLegL = Math.hypot(
        Math.max(0.0, x - TerrainDEMLoader.U_TOP_LEFT_X),
        z - TerrainDEMLoader.U_LEFT_Z
      );
      const dLegR = Math.hypot(
        Math.max(0.0, x - TerrainDEMLoader.U_TOP_RIGHT_X),
        z - TerrainDEMLoader.U_RIGHT_Z
      );
      distUCenter = Math.min(dLegL, dLegR);
    }

    const meanderU =
      2.0 * Math.sin(x * 0.035 + z * 0.025) + 1.1 * Math.cos(z * 0.048 - x * 0.031);
    const dUHorseshoe = distUCenter - TerrainDEMLoader.U_ARM_HALF_W - meanderU;

    return TerrainDEMLoader._smin(dCampusHead, dUHorseshoe, TerrainDEMLoader.U_BLEND_K);
  }

  /**
   * Evaluates exact 2D Euclidean signed coast distance dI(x, z) for Pulau "I" (Empty Green Island):
   * - dI < 0.0 : Inland on Pulau I
   * - dI = 0.0 : Exact 360° Stone Turap Coping Perimeter of Pulau I
   * - dI > 0.0 : Outward into Selat UI / open ocean
   */
  static getIslandICoastDistance(x, z) {
    const rC = TerrainDEMLoader.I_CORNER_R;
    const qx = Math.abs(x - TerrainDEMLoader.I_CENTER_X) - (TerrainDEMLoader.I_HALF_LEN_X - rC);
    const qz = Math.abs(z - TerrainDEMLoader.I_CENTER_Z) - (TerrainDEMLoader.I_HALF_WID_Z - rC);
    const outsideDist = Math.hypot(Math.max(qx, 0.0), Math.max(qz, 0.0));
    const insideDist = Math.min(Math.max(qx, qz), 0.0);
    const meanderI =
      1.9 * Math.sin(x * 0.039 - z * 0.027) + 1.05 * Math.cos(x * 0.061 + z * 0.033);
    return outsideDist + insideDist - rC - meanderI;
  }

  /**
   * Combined Archipelago Signed Coast Distance dCoast(x, z) = min(dPulauU, dPulauI):
   * - dCoast < 0.0 : Dry land on either Pulau U or Pulau I
   * - dCoast = 0.0 : Exact 360° Stone Turap Promenade Coping Line
   * - dCoast > 0.0 : Surrounding Ocean, Teluk U Lagoon, or Selat UI Bridge Strait (waterline at +3.52m)
   */
  static getIslandCoastDistance(x, z) {
    return Math.min(
      TerrainDEMLoader.getIslandUCoastDistance(x, z),
      TerrainDEMLoader.getIslandICoastDistance(x, z)
    );
  }

  /**
   * Analytical East (+X) shoreline X-coordinate of Danau Kenanga on the Campus Head of Pulau U.
   */
  static getShorelinePromenadeX(z) {
    const dz = z - TerrainDEMLoader.ISLAND_CZ;
    let x = 58.0;
    for (let i = 0; i < 5; i++) {
      const th = Math.atan2(dz, x - TerrainDEMLoader.ISLAND_CX);
      const R = TerrainDEMLoader.getIslandRadiusAtAngle(th);
      const dxSq = Math.max(4.0, R * R - dz * dz);
      x = TerrainDEMLoader.ISLAND_CX + Math.sqrt(dxSq);
    }
    return x;
  }

  /**
   * Pulau "U" Northbound Meandering River Centerline Z(x):
   * Flows from Mata Air in Southern Highland (-286.0, +48.0) -> Northward (+X) through 4 full S-bends
   * -> Mouth of Danau Kecil Rektorat (-95.0, -18.0).
   */
  static evalPulauURiverCenterZ(x) {
    const u = Math.max(
      0.0,
      Math.min(
        1.0,
        (x - TerrainDEMLoader.RIVER_U_X_START) /
          (TerrainDEMLoader.RIVER_U_X_END - TerrainDEMLoader.RIVER_U_X_START)
      )
    );
    const env = Math.pow(Math.max(0.0, Math.sin(Math.PI * u)), 0.85);
    const meander =
      29.0 * Math.sin(4.0 * Math.PI * u) + 5.2 * Math.sin(8.0 * Math.PI * u - 0.6);
    return 48.0 - 66.0 * u + env * meander;
  }

  static evalPulauURiverSlopeZ(x) {
    const eps = 0.35;
    return (
      (TerrainDEMLoader.evalPulauURiverCenterZ(x + eps) -
        TerrainDEMLoader.evalPulauURiverCenterZ(x - eps)) /
      (2.0 * eps)
    );
  }

  static evalPulauURiverCurvatureZ(x) {
    const eps = 0.85;
    return (
      (TerrainDEMLoader.evalPulauURiverCenterZ(x + eps) -
        2.0 * TerrainDEMLoader.evalPulauURiverCenterZ(x) +
        TerrainDEMLoader.evalPulauURiverCenterZ(x - eps)) /
      (eps * eps)
    );
  }

  /**
   * Monotonic downhill freshwater elevation Y_water(x) along Pulau U's Spring-to-Lake system:
   * - Level +11.20m across Mata Air basin (x <= -275.5m)
   * - Smooth strictly downhill descent from +11.20m -> -0.22m across x in (-275.5m, -95.0m)
   * - Level -0.22m (C^1 tangent-continuous!) across Danau Kecil Rektorat (x >= -95.0m)
   */
  static evalPulauURiverWaterY(x) {
    const uDrop = Math.max(
      0.0,
      Math.min(
        1.0,
        (x - TerrainDEMLoader.RIVER_U_X_DROP_START) /
          (TerrainDEMLoader.RIVER_U_X_END - TerrainDEMLoader.RIVER_U_X_DROP_START)
      )
    );
    return (
      TerrainDEMLoader.LAKE_U_WATER_Y +
      (TerrainDEMLoader.SPRING_U_WATER_Y - TerrainDEMLoader.LAKE_U_WATER_Y) *
        Math.pow(1.0 - uDrop, 1.72)
    );
  }

  static evalPulauURiverHalfWidth(x) {
    const u = Math.max(
      0.0,
      Math.min(
        1.0,
        (x - TerrainDEMLoader.RIVER_U_X_START) /
          (TerrainDEMLoader.RIVER_U_X_END - TerrainDEMLoader.RIVER_U_X_START)
      )
    );
    // Natural pool-and-riffle width breathing (narrower ~3.8m rocky riffle chutes, wider ~6.6m calm meander pools)
    const poolRiffle =
      0.66 * Math.sin(x * 0.088 + 0.7) +
      0.40 * Math.cos(x * 0.165 - 1.1) +
      0.22 * Math.sin(x * 0.31 + 2.0);
    return Math.max(3.65, 4.45 + 1.65 * u + poolRiffle);
  }

  static evalPulauUSpringRadiusAtAngle(theta) {
    return (
      TerrainDEMLoader.SPRING_U_R +
      0.62 * Math.sin(3.0 * theta + 0.4) +
      0.38 * Math.cos(5.0 * theta - 0.7) +
      0.22 * Math.sin(8.0 * theta + 1.2)
    );
  }

  static evalPulauULakeRadiusMod(theta) {
    return (
      1.0 +
      0.11 * Math.sin(2.0 * theta + 0.5) -
      0.085 * Math.cos(3.0 * theta - 0.4) +
      0.045 * Math.sin(5.0 * theta) +
      0.025 * Math.cos(8.0 * theta + 0.9)
    );
  }

  /**
   * Evaluates Pulau "U" Unified Natural Fluvial Hydrology Field (Mata Air + Meandering River + Danau Kecil Rektorat).
   * Features curvature-driven asymmetric Beta-thalweg scour pools, shelving inner meander gravel point-bars,
   * steeper outer eroded cut-banks, and organic bank micro-scalloping.
   * Returns null outside influence band (w >= 3.65 or x >= -68.0m Campus Core boundary).
   */
  static evalPulauUHydrology(x, z) {
    if (x < -328.0 || x >= -68.0 || z < -96.0 || z > 116.0) return null;

    // 1. Artesian Forest Spring ("Mata Air Hutan Selatan") at (-286.0, +48.0)
    const dxS = x - TerrainDEMLoader.SPRING_U_CX;
    const dzS = z - TerrainDEMLoader.SPRING_U_CZ;
    const thS = Math.atan2(dzS, dxS);
    const rSpring = TerrainDEMLoader.evalPulauUSpringRadiusAtAngle(thS);
    const wSpring = Math.hypot(dxS, dzS) / rSpring;

    // 2. Meandering Northbound River (true 2D perpendicular projection onto closest curve station s)
    let s = Math.max(
      TerrainDEMLoader.RIVER_U_X_START,
      Math.min(TerrainDEMLoader.RIVER_U_X_END, x)
    );
    for (let k = 0; k < 2; k++) {
      const zs = TerrainDEMLoader.evalPulauURiverCenterZ(s);
      const m = TerrainDEMLoader.evalPulauURiverSlopeZ(s);
      const ds = (x - s + (z - zs) * m) / (1.0 + m * m);
      s = Math.max(
        TerrainDEMLoader.RIVER_U_X_START,
        Math.min(TerrainDEMLoader.RIVER_U_X_END, s + ds * 0.88)
      );
    }
    const zsFinal = TerrainDEMLoader.evalPulauURiverCenterZ(s);
    const mFinal = TerrainDEMLoader.evalPulauURiverSlopeZ(s);
    const invT = 1.0 / Math.hypot(1.0, mFinal);
    const cxVec = -mFinal * invT;
    const czVec = invT;
    const latDist = (x - s) * cxVec + (z - zsFinal) * czVec;
    const sideSign = latDist >= 0.0 ? 1.0 : -1.0;

    // Asymmetric left/right organic shoreline micro-scalloping (breaks up ruler-straight bank lines!)
    const bankScallop =
      0.25 * Math.sin(s * 0.43 + sideSign * 1.75) +
      0.16 * Math.cos(s * 0.94 - sideSign * 0.85) +
      0.09 * Math.sin(s * 1.88 + z * 0.62);
    const halfW = TerrainDEMLoader.evalPulauURiverHalfWidth(s);
    const effHalfW = Math.max(3.40, halfW + bankScallop);
    const signedCross = latDist / effHalfW;
    const wRiver = Math.hypot(x - s, z - zsFinal) / effHalfW;

    // 3. Scenic Small Lake beside Gedung Rektorat UI ("Danau Kecil Rektorat") at (-82.5, -22.0)
    const dxL = x - TerrainDEMLoader.LAKE_U_CX;
    const dzL = z - TerrainDEMLoader.LAKE_U_CZ;
    const thL = Math.atan2(dzL, dxL);
    const modL = TerrainDEMLoader.evalPulauULakeRadiusMod(thL);
    const wLake = Math.hypot(
      dxL / (TerrainDEMLoader.LAKE_U_RX * modL),
      dzL / (TerrainDEMLoader.LAKE_U_RZ * modL)
    );

    const wSR = TerrainDEMLoader._smin(wSpring, wRiver, 0.28);
    const w = TerrainDEMLoader._smin(wSR, wLake, 0.30);
    if (w >= 3.65) return null;

    const waterY = TerrainDEMLoader.evalPulauURiverWaterY(s);
    const springWeight = 1.0 - PhysicsKernel.smootherstep(0.75, 1.75, wSpring);
    const lakeWeight = 1.0 - PhysicsKernel.smootherstep(0.75, 1.75, wLake);
    const riverDominance = (1.0 - springWeight) * (1.0 - lakeWeight);

    // Meander curvature kappa(s) -> shifts deep thalweg toward outer cut-bank & forms shallow inner gravel point-bar!
    const curv = TerrainDEMLoader.evalPulauURiverCurvatureZ(s);
    const bendBias = Math.max(-0.80, Math.min(0.80, Math.tanh(-curv * 34.0))) * riverDominance;
    const clampedCross = Math.max(-1.35, Math.min(1.35, signedCross));
    const bendSide = bendBias * clampedCross; // > 0 on outer cut-bank, < 0 on inner gravel/sand point-bar

    // Asymmetric Beta-distribution channel bathymetry B(u): zero depth right at unified shoreline contour w = 1.03!
    const effSignedCross =
      w < wRiver && wRiver > 1e-4 ? (signedCross / wRiver) * w : signedCross;
    const u01 = Math.max(0.0, Math.min(1.0, 0.5 + 0.5 * (effSignedCross / 1.03)));
    const pL = 1.12 + 0.60 * bendBias;
    const pR = 1.12 - 0.60 * bendBias;
    const uPeak = pL / (pL + pR);
    const peakNorm = Math.pow(uPeak, pL) * Math.pow(1.0 - uPeak, pR) || 1.0;
    const riverBetaProfile =
      w < 1.03 ? (Math.pow(u01, pL) * Math.pow(1.0 - u01, pR)) / peakNorm : 0.0;

    const wBasinNorm = Math.min(1.0, w / 1.03);
    const basinProfile = Math.pow(Math.max(0.0, 1.0 - wBasinNorm * wBasinNorm), 0.68);
    const channelProfile =
      riverBetaProfile * riverDominance + basinProfile * (1.0 - riverDominance);

    const maxDepth =
      1.28 +
      0.36 * springWeight +
      0.48 * lakeWeight +
      0.18 * Math.abs(bendBias);
    const pebble =
      (0.055 * Math.sin(x * 0.82 + z * 0.71) * Math.cos(z * 0.91 - x * 0.58) +
        0.032 * Math.sin(x * 1.92 - z * 1.68)) *
      channelProfile;
    const bedY = waterY - maxDepth * channelProfile + pebble;

    return {
      w,
      wSpring,
      wRiver,
      wLake,
      stationX: s,
      effHalfW,
      signedCross,
      bendBias,
      bendSide,
      waterY,
      bedY
    };
  }

  /**
   * Returns exact water surface elevation Y at (x, z) if over Ocean, Lagoon, Strait,
   * Pulau U Spring/River/Small-Lake, or Pulau I Mini-Niagara Upper-River/Plunge-Pool/Outlet; else null.
   */
  static sampleWaterSurfaceY(x, z) {
    const dCoast = TerrainDEMLoader.getIslandCoastDistance(x, z);
    if (dCoast > 3.50) {
      return TerrainDEMLoader.WATER_LEVEL_Y;
    }
    if (z < 536.0) {
      const hydro = TerrainDEMLoader.evalPulauUHydrology(x, z);
      if (hydro && hydro.w <= 1.03) {
        return hydro.waterY;
      }
      return null;
    }
    // Pulau I Mini-Niagara Horseshoe Falls water zones
    const dxW = x - TerrainDEMLoader.NIAGARA_CX;
    const dzW = z - TerrainDEMLoader.NIAGARA_CZ;
    const rW = Math.hypot(dxW, dzW);
    const alpha = Math.atan2(dxW, dzW);
    const absAlpha = Math.abs(alpha);
    if (rW >= 15.3 && rW <= 29.8 && absAlpha <= 1.15) {
      for (const spAlpha of [-0.62, 0.54]) {
        const sx = TerrainDEMLoader.NIAGARA_CX + Math.sin(spAlpha) * 15.9;
        const sz = TerrainDEMLoader.NIAGARA_CZ + Math.cos(spAlpha) * 15.9;
        if (Math.hypot(x - sx, z - sz) < 2.3) return null;
      }
      return TerrainDEMLoader.NIAGARA_UPPER_WATER_Y;
    }
    if (rW <= 14.8) {
      return TerrainDEMLoader.NIAGARA_POOL_WATER_Y;
    }
    if (z >= 610.5 && z <= 630.5 && Math.abs(dxW) <= 3.7) {
      const coastCollar = PhysicsKernel.smootherstep(-1.2, -10.5, dCoast);
      const uOut = Math.max(0.0, Math.min(1.0, (630.0 - z) / 19.5));
      return (TerrainDEMLoader.NIAGARA_POOL_WATER_Y * (1.0 - uOut) + 0.16 * uOut) * coastCollar;
    }
    return null;
  }

  /**
   * True whenever (x, z) lies inside or within marginMeters of any freshwater spring, river, small lake,
   * or Mini-Niagara horseshoe waterfall gorge — keeps spawned tree trunks cleanly on dry banks & canyon rims!
   */
  static isFreshwaterExclusion(x, z, marginMeters = 2.2) {
    if (z < 536.0) {
      const hydro = TerrainDEMLoader.evalPulauUHydrology(x, z);
      if (!hydro) return false;
      const approxScale =
        hydro.wLake < hydro.wRiver
          ? 14.5
          : hydro.effHalfW || TerrainDEMLoader.evalPulauURiverHalfWidth(hydro.stationX);
      return hydro.w < 1.06 + marginMeters / Math.max(4.2, approxScale);
    }
    const dxW = x - TerrainDEMLoader.NIAGARA_CX;
    const dzW = z - TerrainDEMLoader.NIAGARA_CZ;
    const rW = Math.hypot(dxW, dzW);
    const absAlpha = Math.abs(Math.atan2(dxW, dzW));
    if (rW < 16.5 + marginMeters) return true;
    if (rW < 32.0 + marginMeters && absAlpha < 1.24 + marginMeters * 0.04) return true;
    if (z > 607.0 && z < 632.0 && Math.abs(dxW) < 5.2 + marginMeters) return true;
    return false;
  }

  /**
   * Evaluates continuous ground / plaza / helipad headland / green park meadow / seabed elevation y(x, z)
   * across the entire Kepulauan UI (Pulau U & Pulau I).
   */
  static sampleGroundHeight(x, z) {
    const dU = TerrainDEMLoader.getIslandUCoastDistance(x, z);
    const dI = TerrainDEMLoader.getIslandICoastDistance(x, z);
    const dCoast = Math.min(dU, dI);

    // 1. Coastal Embankment & Submerged Seabed / Lagoon / Strait Basin
    if (dCoast > 0.0) {
      if (dCoast <= 3.6) {
        const tUpper = PhysicsKernel.smootherstep(0.0, 3.6, dCoast);
        return TerrainDEMLoader.WATER_LEVEL_Y * tUpper;
      }
      const tLower = PhysicsKernel.smootherstep(3.6, 12.0, dCoast);
      const bedMacro =
        0.14 *
        Math.sin(x * 0.05) *
        Math.cos(z * 0.05) *
        PhysicsKernel.smootherstep(12.0, 26.0, dCoast);
      const bedMicro =
        0.045 *
        Math.sin(x * 0.24 + z * 0.19) *
        Math.cos(z * 0.31 - x * 0.14) *
        PhysicsKernel.smootherstep(6.0, 16.0, dCoast);
      return (
        TerrainDEMLoader.WATER_LEVEL_Y +
        (TerrainDEMLoader.LAKE_BED_Y - TerrainDEMLoader.WATER_LEVEL_Y) * tLower +
        bedMacro +
        bedMicro
      );
    }

    // Coastal promenade leveling collar: smoothly brings all inland terrain to 0.0m datum
    // within 10.5m of the 360° stone turap coping wall (ready for future bridge abutments!)
    const coastCollar = PhysicsKernel.smootherstep(-1.2, -10.5, dCoast);

    // 2. Pulau "I": Active Stratovolcano ("Gunung Api") + Mini-Niagara Waterfall + Flat ICAO Runway Plateau
    if (dI < dU) {
      // 2A. Engineered flat airfield embankment (y = 0.44m under y = 0.58m runway/apron deck)
      const rwyXMask =
        PhysicsKernel.smootherstep(-394.0, -384.0, x) *
        PhysicsKernel.smootherstep(-114.0, -126.0, x);
      const rwyZMask =
        PhysicsKernel.smootherstep(648.0, 658.0, z) *
        PhysicsKernel.smootherstep(736.0, 724.0, z);
      const airfieldPlatMask = rwyXMask * rwyZMask;
      const airfieldPadY = 0.44 * coastCollar * airfieldPlatMask;

      // 2B. Gunung Gede Bogor Tropical Montane Stratovolcano Morphology at (-32, 698):
      //     Sweeping verdant montane ridges & Cibodas-style stream ravines, a gentle Suryakencana
      //     sub-alpine upper saddle contour, and weathered grey-andesite Kawah Ratu crater bowl (ZERO molten lava!)
      const dxV = x - TerrainDEMLoader.VOLCANO_CX;
      const dzV = z - TerrainDEMLoader.VOLCANO_CZ;
      const rV = Math.hypot(dxV, dzV);
      const thV = Math.atan2(dzV, dxV);
      const rimWobble =
        1.25 * Math.sin(3.0 * thV + 0.6) +
        0.80 * Math.cos(5.0 * thV - 0.9) +
        0.42 * Math.sin(7.0 * thV + 1.4);
      const rRim = TerrainDEMLoader.VOLCANO_RIM_R + rimWobble;
      const rimHeight =
        TerrainDEMLoader.VOLCANO_PEAK_RIM_Y +
        2.15 * Math.cos(2.0 * thV - 0.35) +
        1.05 * Math.sin(3.0 * thV + 1.15);

      // Smoothly clear southern toe before North Runway Blast Pad (x = -128m)
      const southApronFade = PhysicsKernel.smootherstep(-126.0, -100.0, x);

      let hVolcano = 0.0;
      if (rV < rRim) {
        // Sunken ~14m-deep Kawah Ratu Andesite Crater Bowl & Quiet Hydrothermal Mineral Floor (y ~ 44.6m)
        const uIn = rV / Math.max(1.0, rRim);
        const wallT = PhysicsKernel.smootherstep(0.50, 1.0, uIn);
        const craterWallCrag =
          0.72 * Math.sin(13.0 * thV + uIn * 5.2) * wallT * (1.0 - wallT) * 4.0;
        const mineralTalus =
          0.38 * (1.0 - PhysicsKernel.smootherstep(0.0, 0.48, uIn)) +
          0.16 * Math.sin(x * 0.48) * Math.cos(z * 0.48) * (1.0 - wallT);
        hVolcano =
          (TerrainDEMLoader.VOLCANO_CRATER_FLOOR_Y +
            mineralTalus +
            (rimHeight - TerrainDEMLoader.VOLCANO_CRATER_FLOOR_Y) * Math.pow(wallT, 1.38) +
            craterWallCrag) *
          southApronFade *
          coastCollar;
      } else if (rV < TerrainDEMLoader.VOLCANO_BASE_R) {
        const s = (rV - rRim) / (TerrainDEMLoader.VOLCANO_BASE_R - rRim);
        // Broader, lush West-Javanese Gunung Gede shoulder profile + subtle Suryakencana SE sub-alpine bench
        const coneProfile = Math.pow(Math.max(0.0, 1.0 - s), 1.54);
        const suryakencanaAz = Math.max(0.0, Math.cos(thV - 0.72));
        const suryakencanaBench =
          3.4 *
          Math.pow(suryakencanaAz, 3.2) *
          Math.exp(-Math.pow((s - 0.34) / 0.14, 2.0));
        const gullyEnv = Math.sin(Math.min(1.0, s / 0.90) * Math.PI) * PhysicsKernel.smootherstep(0.0, 0.08, s);
        const montaneRidges =
          2.65 * Math.cos(8.0 * thV + 0.55 * Math.sin(2.0 * thV) + s * 1.35) +
          1.25 * Math.sin(15.0 * thV - s * 2.1);
        const forestCanopyUndul = 0.58 * Math.sin(x * 0.12 + z * 0.10) * Math.cos(z * 0.14 - x * 0.11);
        hVolcano =
          Math.max(
            0.0,
            rimHeight * coneProfile +
              suryakencanaBench +
              (montaneRidges + forestCanopyUndul) * gullyEnv * coneProfile
          ) *
          southApronFade *
          (1.0 - airfieldPlatMask * 0.96) *
          coastCollar;
      }

      // Gentle coastal meadow & West Selat UI bridge-landing terrace outside runway & volcano cone
      const meadowWave =
        0.52 +
        0.28 * Math.sin(x * 0.032 + 0.7) * Math.cos((z - 694.0) * 0.044) +
        0.14 * Math.cos((x + z) * 0.024 - 0.4);
      const bridgeReserveGate = PhysicsKernel.smootherstep(614.0, 646.0, z);
      const foothillMeadow =
        (0.35 + 1.15 * bridgeReserveGate) *
        coastCollar *
        (1.0 - airfieldPlatMask) *
        Math.max(0.0, Math.min(1.0, meadowWave));

      let yI = Math.max(
        airfieldPadY,
        hVolcano + foothillMeadow * (1.0 - Math.min(1.0, hVolcano / 7.5))
      );

      // 2C. Mini-Niagara Horseshoe Waterfall Escarpment ("Air Terjun Niagara Mini") on West-SW Volcano Flank
      if (x > -104.0 && x < 16.0 && z > 606.0 && z < 688.0) {
        const dxW = x - TerrainDEMLoader.NIAGARA_CX;
        const dzW = z - TerrainDEMLoader.NIAGARA_CZ;
        const rW = Math.hypot(dxW, dzW) + 1e-5;
        const alpha = Math.atan2(dxW, dzW);
        const absAlpha = Math.abs(alpha);

        // North & South upper basalt canyon rim walls (guarantees upper river y = 18.80m is 100% rock-contained)
        const upperRimGuard =
          PhysicsKernel.smootherstep(1.04, 1.20, absAlpha) *
          (1.0 - PhysicsKernel.smootherstep(1.30, 1.62, absAlpha)) *
          PhysicsKernel.smootherstep(13.8, 15.8, rW) *
          (1.0 - PhysicsKernel.smootherstep(27.0, 32.5, rW)) *
          coastCollar;
        if (upperRimGuard > 0.001) {
          yI = yI * (1.0 - upperRimGuard) + Math.max(yI, 20.75 * coastCollar) * upperRimGuard;
        }

        // Upper basalt feeding river shelf (bed ~17.65m under y_water = 18.80m, emerging from volcanic spring gorge)
        const upperAngMask = 1.0 - PhysicsKernel.smootherstep(1.02, 1.28, absAlpha);
        const upperRadMask =
          PhysicsKernel.smootherstep(14.8, 16.1, rW) *
          (1.0 - PhysicsKernel.smootherstep(28.5, 33.8, rW));
        const upperMask = upperAngMask * upperRadMask;
        if (upperMask > 0.001) {
          const upperBedY =
            (17.65 +
              0.38 * PhysicsKernel.smootherstep(16.0, 29.5, rW) +
              0.10 * Math.sin(x * 0.58) * Math.cos(z * 0.62)) *
            coastCollar;
          yI = yI * (1.0 - upperMask) + Math.min(yI, upperBedY) * upperMask;
        }

        // 2 Natural Dark-Basalt Crest-Splitter Islets ("Goat Island" boulders) splitting the 36.8m horseshoe brink
        for (const spAlpha of [-0.62, 0.54]) {
          const sx = TerrainDEMLoader.NIAGARA_CX + Math.sin(spAlpha) * 15.9;
          const sz = TerrainDEMLoader.NIAGARA_CZ + Math.cos(spAlpha) * 15.9;
          const dSp = Math.hypot(x - sx, z - sz);
          if (dSp < 3.4) {
            const rockH =
              (20.25 - 0.34 * dSp + 0.16 * Math.sin(x * 1.8 - z * 1.6)) * coastCollar;
            const spT = 1.0 - PhysicsKernel.smootherstep(1.05, 3.3, dSp);
            yI = yI * (1.0 - spT) + Math.max(yI, rockH) * spT;
          }
        }

        // Lower amphitheater side-berms (guarantees plunge pool y = 1.80m is enclosed >= +3.8m on North & South!)
        const poolRimGuard =
          PhysicsKernel.smootherstep(1.18, 1.38, absAlpha) *
          (1.0 - PhysicsKernel.smootherstep(2.48, 2.86, absAlpha)) *
          PhysicsKernel.smootherstep(13.2, 15.4, rW) *
          (1.0 - PhysicsKernel.smootherstep(18.5, 24.5, rW)) *
          coastCollar;
        if (poolRimGuard > 0.001) {
          yI = yI * (1.0 - poolRimGuard) + Math.max(yI, 4.15 * coastCollar) * poolRimGuard;
        }

        // Sheer 17m Curved Columnar-Basalt Horseshoe Cliff & Sunken Turquoise Plunge Pool (bed ~0.18m, water 1.80m)
        const lipFlute =
          TerrainDEMLoader.NIAGARA_LIP_R +
          0.28 * Math.sin(alpha * 9.0) +
          0.16 * Math.cos(alpha * 15.0);
        const effLipR = absAlpha <= 1.24 ? lipFlute : 14.5;
        if (rW < effLipR + 0.6) {
          const poolCarveT = 1.0 - PhysicsKernel.smootherstep(effLipR - 2.0, effLipR + 0.3, rW);
          const poolBedY =
            (0.18 + 0.10 * Math.sin(x * 0.45) * Math.cos(z * 0.45)) * coastCollar;
          yI = yI * (1.0 - poolCarveT) + poolBedY * poolCarveT;
        }

        // Western Tailrace Outlet Gorge draining Plunge Pool toward Selat UI (z: 630 -> 609)
        if (z >= 607.5 && z <= 632.0 && Math.abs(dxW) < 12.5) {
          const uOut = Math.max(0.0, Math.min(1.0, (630.0 - z) / 19.5));
          const outFadeZ =
            PhysicsKernel.smootherstep(607.5, 610.5, z) *
            (1.0 - PhysicsKernel.smootherstep(629.5, 632.0, z));
          const outWaterY =
            (TerrainDEMLoader.NIAGARA_POOL_WATER_Y * (1.0 - uOut) + 0.16 * uOut) * coastCollar;
          const outHalfW = 3.85;
          const wOut = Math.abs(dxW) / outHalfW;

          if (wOut < 3.1) {
            const leveeT =
              PhysicsKernel.smootherstep(0.95, 1.22, wOut) *
              (1.0 - PhysicsKernel.smootherstep(1.65, 3.05, wOut)) *
              outFadeZ;
            const minLeveeY = (outWaterY + 0.95) * coastCollar;
            yI = yI * (1.0 - leveeT) + Math.max(yI, minLeveeY) * leveeT;

            if (wOut < 1.15) {
              const chCarveT =
                (1.0 - PhysicsKernel.smootherstep(0.32, 1.12, wOut)) * outFadeZ;
              const outBedY = (outWaterY - 0.86) * coastCollar;
              yI = yI * (1.0 - chCarveT) + Math.min(yI, outBedY) * chCarveT;
            }
          }
        }
      }

      return yI;
    }

    // 3. Pulau "U" — Gedung Rektorat Complex, Courtyards, Rotunda, & VIP Helipad Corridor: strict 0.0m landai datum
    const distHeli = Math.hypot(x - TerrainDEMLoader.HELIPAD_CX, z - TerrainDEMLoader.HELIPAD_CZ);
    const inCampusCore = x >= -68.0 && x <= 78.0 && z >= -58.0 && z <= 156.0;
    const inHelipadCorridor = z >= 96.0 && z <= 118.0 && x >= -112.0 && x <= 10.0;
    if (inCampusCore || distHeli <= 24.0 || inHelipadCorridor) {
      return 0.0;
    }

    // 4A. Gentle Campus Urban Forest Mounds around West/North Rectorate Tree Belt (preserving all campus trees!)
    const westMask = PhysicsKernel.smootherstep(-68.0, -86.0, x);
    const northMask = PhysicsKernel.smootherstep(-58.0, -76.0, z);
    const southUBendMask = PhysicsKernel.smootherstep(-118.0, -148.0, x);
    const rightArmMask = PhysicsKernel.smootherstep(160.0, 206.0, z);
    const forestMask = Math.max(westMask, northMask, southUBendMask, rightArmMask);

    const distRotunda = Math.hypot(x + 2.5, z - 106.5);
    const rotundaClearance = PhysicsKernel.smootherstep(37.0, 50.0, distRotunda);
    const helipadClearance =
      PhysicsKernel.smootherstep(24.0, 34.0, distHeli) *
      (x < -66.0 && z < 160.0
        ? Math.min(
            1.0,
            PhysicsKernel.smootherstep(96.0, 84.0, z) + PhysicsKernel.smootherstep(118.0, 130.0, z)
          )
        : 1.0);

    const harmonicA = Math.sin(x * 0.041 + 0.35) * Math.cos(z * 0.034 - 0.25);
    const harmonicB = Math.cos((x - z) * 0.027 + 0.8);
    const microRootRelief =
      0.07 * Math.sin(x * 0.18 + 0.5) * Math.cos(z * 0.20 - 0.3) +
      0.03 * Math.cos((x + z) * 0.36);
    const normalizedWave = Math.max(
      0.0,
      Math.min(1.0, 0.54 + 0.28 * harmonicA + 0.14 * harmonicB + microRootRelief)
    );

    const baseParkMoundY =
      1.26 *
      forestMask *
      rotundaClearance *
      Math.min(1.0, helipadClearance) *
      coastCollar *
      normalizedWave;

    // 4B. Pulau "U" Scenic Rolling Tropical Hills ("Perbukitan Hijau Pulau U") along Southern Highland & Right Arm
    //     while keeping the entire Rectorate / Rotunda / Helipad campus head gently flat ("landai")!
    const leftArmHillGate = PhysicsKernel.smootherstep(-114.0, -168.0, x);
    const rightArmHillGate = PhysicsKernel.smootherstep(178.0, 242.0, z);
    const hillRegionMask = Math.max(leftArmHillGate, rightArmHillGate);

    let uHillsY = 0.0;
    if (hillRegionMask > 0.0001) {
      const inlandHillEnv = PhysicsKernel.smootherstep(-7.5, -46.0, dU);
      // Natural coastal bridge-landing saddle on Pulau U's eastern shore facing Selat UI & Pulau I
      const distBridgeSaddleU = Math.hypot((x + 215.0) * 0.82, z - 454.0);
      const bridgeSaddleU = 0.22 + 0.78 * PhysicsKernel.smootherstep(20.0, 58.0, distBridgeSaddleU);

      const gBell = (cx, cz, rx, rz, amp) => {
        const ux = (x - cx) / rx;
        const uz = (z - cz) / rz;
        const d2 = ux * ux + uz * uz;
        return d2 > 9.0 ? 0.0 : amp * Math.exp(-d2);
      };

      const hillsChain =
        gBell(-222.0, 26.0, 56.0, 46.0, 12.6) +   // SW Left-Arm Hill
        gBell(-302.0, 52.0, 52.0, 48.0, 11.8) +   // Southern Highland Forest Spring Amphitheater Ridge
        gBell(-322.0, 122.0, 58.0, 54.0, 15.2) +  // SW U-Bend Hill
        gBell(-336.0, 236.0, 54.0, 64.0, 17.8) +  // Grand South Teluk U Overlook Peak (~19m)
        gBell(-268.0, 386.0, 54.0, 46.0, 14.4) +  // SE Right-Arm Hill
        gBell(-68.0, 392.0, 74.0, 46.0, 16.4) +   // NE Right-Arm Peninsula Hill overlooking Teluk U & Selat UI
        gBell(18.0, 392.0, 44.0, 42.0, 10.6);     // North Right-Arm Headland Knoll

      const rollingRidgeUndulation =
        3.5 +
        2.3 * Math.sin(x * 0.026 + 0.4) * Math.cos(z * 0.024 - 0.3) +
        1.4 * Math.cos((x + z) * 0.037 + 0.9) +
        0.65 * Math.sin(x * 0.085 - z * 0.075);

      uHillsY =
        Math.max(0.0, rollingRidgeUndulation + hillsChain) *
        hillRegionMask *
        inlandHillEnv *
        bridgeSaddleU *
        rotundaClearance *
        Math.min(1.0, helipadClearance) *
        coastCollar;
    }

    let uRawY = baseParkMoundY + uHillsY;

    // 4B2. Jembatan Teksas UI Eastern FIB Landing Terrace (Right Arm of Pulau U: x ~ -28, z in 319..345)
    //      Smoothly grades a flat +0.34m coastal plaza for Pilon Yoni, stairs, accessible ramps & benches
    //      via a broad elliptical amphitheater meadow transition while leaving Right-Arm hill peaks (z ~ 392m) 100.0% untouched!
    if (z > 314.0 && z < 384.0 && Math.abs(x - TerrainDEMLoader.TEKSAS_BRIDGE_X) < 37.0) {
      const dxPad = Math.max(0.0, Math.abs(x - TerrainDEMLoader.TEKSAS_BRIDGE_X) - 8.2) / 28.0;
      const dzPad =
        z < 332.0
          ? Math.max(0.0, 319.0 - z) / 4.8
          : Math.max(0.0, z - 345.0) / 38.0;
      const rPad = Math.hypot(dxPad, dzPad);
      if (rPad < 1.0) {
        const teksasEastPadMask = 1.0 - PhysicsKernel.smootherstep(0.0, 1.0, rPad);
        const targetPadY = TerrainDEMLoader.TEKSAS_EAST_PLAZA_Y * coastCollar;
        uRawY = uRawY * (1.0 - teksasEastPadMask) + targetPadY * teksasEastPadMask;
      }
    }

    // 4C. Pulau "U" Natural Fluvial Valley, Shelving Point-Bar Beaches, & Eroded Cut-Banks
    //     (Eliminates flat artificial drainage-canal sidewalks; shoreline meets waterY continuously at w = 1.03!)
    const hydro = TerrainDEMLoader.evalPulauUHydrology(x, z);
    if (hydro) {
      if (hydro.w <= 1.03) {
        return hydro.bedY * coastCollar;
      }
      const campusCoreTaper = PhysicsKernel.smootherstep(-68.0, -70.8, x);
      const shoreY = hydro.waterY * coastCollar;
      const uBank = Math.max(0.0, Math.min(1.0, (hydro.w - 1.03) / (3.65 - 1.03)));
      // Inner bend (bendSide < 0): gentle shelving gravel/sand point-bar; Outer bend (bendSide > 0): steeper mossy cut-bank
      const s0 = Math.max(0.36, Math.min(1.62, 0.92 + 0.62 * hydro.bendSide));
      const vCurve =
        s0 * uBank * (1.0 - uBank) * (1.0 - uBank) +
        uBank * uBank * (3.0 - 2.0 * uBank);

      const hillSpan = Math.max(0.0, uRawY - shoreY);
      const valleyMicro =
        Math.min(0.42, hillSpan * 0.08) *
        (0.65 * Math.sin(x * 0.29 + z * 0.33) * Math.cos(z * 0.24 - x * 0.21) +
          0.35 * Math.sin(x * 0.71 - z * 0.64)) *
        Math.sin(Math.PI * uBank);

      const targetRimY = (hydro.waterY + 0.22 + 0.34 * campusCoreTaper) * coastCollar;
      const neededLift = Math.max(0.0, targetRimY - (shoreY + (uRawY - shoreY) * 0.42));
      const bermArch = neededLift * Math.sin(Math.PI * Math.pow(uBank, 0.62));

      return shoreY + (uRawY - shoreY) * vCurve + bermArch + valleyMicro;
    }

    return uRawY;
  }

  /**
   * Computes smooth central-difference surface normal for sampleGroundHeight(x, z).
   */
  static sampleGroundNormal(x, z, target = new THREE.Vector3()) {
    const eps = 0.28;
    const hL = TerrainDEMLoader.sampleGroundHeight(x - eps, z);
    const hR = TerrainDEMLoader.sampleGroundHeight(x + eps, z);
    const hD = TerrainDEMLoader.sampleGroundHeight(x, z - eps);
    const hU = TerrainDEMLoader.sampleGroundHeight(x, z + eps);
    return target.set(hL - hR, 2.0 * eps, hD - hU).normalize();
  }

  /**
   * Builds a Rectangular High-Density Island DEM Patch + 4-Sided -18.0m Watertight Submerged Skirt (Hukum Emas 5).
   * Outer borders lie strictly in submerged seabed (y = -3.85m) so patches blend seamlessly beneath the ocean!
   */
  static _buildIslandPatchWithSkirtGeometry(xMin, xMax, zMin, zMax, segsX, segsZ, coordFocus = null) {
    const spanX = xMax - xMin;
    const spanZ = zMax - zMin;

    // Strict C-infinity monotonic 1D coordinate concentration warp (g'(u) >= 0.46 > 0 everywhere)
    // Concentrates DEM vertex density 2.1x across the Volcano Crater Rim & Magma Bowl!
    const makeMonotonicAxisFn = (minVal, spanVal, focusVal, sigmaNorm) => {
      if (focusVal === null || focusVal === undefined) {
        return (u) => minVal + u * spanVal;
      }
      const uc = (focusVal - minVal) / spanVal;
      const k = 0.54;
      const raw = (u) => {
        const d = (u - uc) / sigmaNorm;
        return u - k * (u - uc) * Math.exp(-d * d);
      };
      const f0 = raw(0.0);
      const fSpan = raw(1.0) - f0 || 1.0;
      return (u) => minVal + ((raw(u) - f0) / fSpan) * spanVal;
    };

    const mapX = makeMonotonicAxisFn(
      xMin,
      spanX,
      coordFocus ? coordFocus.focusX : null,
      coordFocus?.sigmaX || 0.13
    );
    const mapZ = makeMonotonicAxisFn(
      zMin,
      spanZ,
      coordFocus ? coordFocus.focusZ : null,
      coordFocus?.sigmaZ || 0.20
    );

    const gridVerts = (segsX + 1) * (segsZ + 1);
    const skirtVerts = 2 * 2 * (segsX + 1) + 2 * 2 * (segsZ + 1);
    const totalVerts = gridVerts + skirtVerts;
    const totalTris = segsX * segsZ * 2 + 2 * segsX * 2 + 2 * segsZ * 2;

    const positions = new Float32Array(totalVerts * 3);
    const normals = new Float32Array(totalVerts * 3);
    const uvs = new Float32Array(totalVerts * 2);
    const colors = new Float32Array(totalVerts * 3);
    const indices = new Uint32Array(totalTris * 3);

    const nTmp = new THREE.Vector3();
    let vPtr = 0;
    let iPtr = 0;

    const writeVert = (px, py, pz, nx, ny, nz, u, v, r, g, b) => {
      const idx = vPtr++;
      const o3 = idx * 3;
      const o2 = idx * 2;
      positions[o3] = px;
      positions[o3 + 1] = py;
      positions[o3 + 2] = pz;
      normals[o3] = nx;
      normals[o3 + 1] = ny;
      normals[o3 + 2] = nz;
      uvs[o2] = u;
      uvs[o2 + 1] = v;
      colors[o3] = r;
      colors[o3 + 1] = g;
      colors[o3 + 2] = b;
      return idx;
    };

    const stride = segsX + 1;
    for (let iz = 0; iz <= segsZ; iz++) {
      const z = mapZ(iz / segsZ);
      for (let ix = 0; ix <= segsX; ix++) {
        const x = mapX(ix / segsX);
        const y = TerrainDEMLoader.sampleGroundHeight(x, z);
        TerrainDEMLoader.sampleGroundNormal(x, z, nTmp);

        let r = 0.145;
        let g = 0.345;
        let b = 0.078;

        const dCoastVert = z < 536.0 ? TerrainDEMLoader.getIslandUCoastDistance(x, z) : TerrainDEMLoader.getIslandICoastDistance(x, z);

        if (dCoastVert > -0.35 && y < -1.52) {
          const depthT = 1.0 - PhysicsKernel.smootherstep(-3.75, -1.52, y);
          const pebble = 0.02 * Math.sin(x * 0.85) * Math.cos(z * 0.92);
          r = 0.22 * (1.0 - depthT) + 0.09 * depthT + pebble;
          g = 0.26 * (1.0 - depthT) + 0.16 * depthT + pebble;
          b = 0.21 * (1.0 - depthT) + 0.15 * depthT + pebble;
        } else if (dCoastVert > -0.35 && y < -0.06) {
          const bankT = 1.0 - PhysicsKernel.smootherstep(-1.52, -0.06, y);
          r = 0.15 * (1.0 - bankT) + 0.31 * bankT;
          g = 0.33 * (1.0 - bankT) + 0.29 * bankT;
          b = 0.09 * (1.0 - bankT) + 0.24 * bankT;
        } else {
          const dappleMacro = 0.5 + 0.5 * Math.sin(x * 0.09 + 1.1) * Math.cos(z * 0.08 - 0.7);
          const turfMicro = 0.5 + 0.5 * Math.sin(x * 0.42 - z * 0.37) * Math.cos(x * 0.29 + z * 0.51);
          const moundTint = Math.max(0.0, Math.min(1.0, y / 1.8));
          r = 0.105 + 0.042 * dappleMacro + 0.015 * turfMicro + 0.024 * moundTint;
          g = 0.285 + 0.075 * dappleMacro + 0.026 * turfMicro + 0.035 * moundTint;
          b = 0.056 + 0.022 * dappleMacro + 0.010 * turfMicro + 0.014 * moundTint;

          if (z < 536.0 && y > 2.2) {
            // Pulau U Rolling Tropical Hills: sunlit golden-emerald hill crests & rich forest soil slope variation
            const hillElevT = PhysicsKernel.smootherstep(2.2, 16.5, y);
            const steepnessT = 1.0 - PhysicsKernel.smootherstep(0.70, 0.93, nTmp.y);
            const humusPatch = 0.5 + 0.5 * Math.sin(x * 0.21 + z * 0.17) * Math.cos(z * 0.26 - x * 0.15);
            r = r * (1.0 - hillElevT * 0.20) + (0.145 + 0.055 * dappleMacro) * (hillElevT * 0.20) + steepnessT * (0.038 + 0.028 * humusPatch);
            g = g * (1.0 - hillElevT * 0.18) + (0.345 + 0.055 * dappleMacro) * (hillElevT * 0.18) - steepnessT * (0.042 - 0.018 * humusPatch);
            b = b * (1.0 - hillElevT * 0.14) + 0.075 * (hillElevT * 0.14) - steepnessT * 0.008;
          } else if (z >= 536.0) {
            // Pulau I — Gunung Gede Bogor Tropical Montane Ecological & Geological Zonations:
            // (Lush Cibodas Montane Cloud-Forest -> Suryakencana Sub-Alpine Meadow Bench -> Grey-Andesite Kawah Ratu Rim & Mineral Crater!)
            const dxV = x - TerrainDEMLoader.VOLCANO_CX;
            const dzV = z - TerrainDEMLoader.VOLCANO_CZ;
            const rV = Math.hypot(dxV, dzV);
            const thV = Math.atan2(dzV, dxV);

            // 1. Lush Emerald West-Javanese Montane Rainforest & Mossy Ravine Shading climbing high up the slopes (y: 3m .. 48m)
            const montaneForestT =
              PhysicsKernel.smootherstep(2.5, 11.0, y) *
              (1.0 - PhysicsKernel.smootherstep(43.5, 52.5, y));
            if (montaneForestT > 0.005) {
              const ridgeCrest = 0.5 + 0.5 * Math.cos(8.0 * thV + 0.55 * Math.sin(2.0 * thV));
              const canopyTex = 0.5 + 0.5 * Math.sin(x * 0.19 - z * 0.16) * Math.cos(z * 0.23 + x * 0.14);
              // Deep cool mossy ravine greens in valleys + warm sunlit sub-montane canopy greens on ridges
              const mR = 0.088 + 0.048 * ridgeCrest + 0.024 * canopyTex;
              const mG = 0.245 + 0.088 * ridgeCrest + 0.036 * dappleMacro;
              const mB = 0.072 + 0.022 * canopyTex;
              r = r * (1.0 - montaneForestT * 0.86) + mR * (montaneForestT * 0.86);
              g = g * (1.0 - montaneForestT * 0.86) + mG * (montaneForestT * 0.86);
              b = b * (1.0 - montaneForestT * 0.86) + mB * (montaneForestT * 0.86);
            }

            // 2. Alun-Alun Suryakencana Sub-Alpine Golden-Olive Edelweiss Meadow Shoulder (SE/E upper flank, y ~ 30..46m)
            const suryaAz = Math.max(0.0, Math.cos(thV - 0.72));
            const suryaMeadowT =
              Math.pow(suryaAz, 2.8) *
              PhysicsKernel.smootherstep(28.0, 35.0, y) *
              (1.0 - PhysicsKernel.smootherstep(43.0, 49.0, y)) *
              PhysicsKernel.smootherstep(0.68, 0.90, nTmp.y);
            if (suryaMeadowT > 0.01) {
              const edelweiss = 0.5 + 0.5 * Math.sin(x * 0.42 + z * 0.37);
              const sR = 0.205 + 0.055 * edelweiss;
              const sG = 0.295 + 0.050 * edelweiss;
              const sB = 0.125 + 0.035 * edelweiss;
              r = r * (1.0 - suryaMeadowT * 0.75) + sR * (suryaMeadowT * 0.75);
              g = g * (1.0 - suryaMeadowT * 0.75) + sG * (suryaMeadowT * 0.75);
              b = b * (1.0 - suryaMeadowT * 0.75) + sB * (suryaMeadowT * 0.75);
            }

            // 3. Upper Sub-Alpine Treeline -> Weathered Grey-Andesite Summit Crown ("Puncak Gede & Kawah Ratu", y > 44m)
            const treeLineJitter = 2.4 * Math.sin(thV * 7.0 + 0.4) + 1.4 * Math.cos(thV * 13.0);
            const andesiteT = Math.max(
              PhysicsKernel.smootherstep(43.5, 52.5, y + treeLineJitter),
              PhysicsKernel.smootherstep(24.0, 38.0, y) * (1.0 - PhysicsKernel.smootherstep(0.55, 0.76, nTmp.y)) * 0.65
            );
            if (andesiteT > 0.005) {
              const strata = 0.5 + 0.5 * Math.cos(8.0 * thV + y * 0.35);
              const andR = 0.255 + 0.065 * strata + 0.025 * turfMicro;
              const andG = 0.258 + 0.062 * strata + 0.025 * turfMicro;
              const andB = 0.246 + 0.055 * strata + 0.022 * turfMicro;
              r = r * (1.0 - andesiteT) + andR * andesiteT;
              g = g * (1.0 - andesiteT) + andG * andesiteT;
              b = b * (1.0 - andesiteT) + andB * andesiteT;
            }

            // 4. Inner Sunken Kawah Ratu Crater Walls & Quiet Pale Hydrothermal Mineral-Grey Floor (ZERO Red Magma!)
            if (rV < 20.5 && y > 42.0) {
              const innerWallT = 1.0 - PhysicsKernel.smootherstep(14.0, 20.5, rV);
              const mineralFloorT = 1.0 - PhysicsKernel.smootherstep(9.5, 14.5, rV);
              const sulfurHint = 0.5 + 0.5 * Math.sin(x * 0.52 - z * 0.47);
              // Pale bleached silica-grey crater walls
              r = r * (1.0 - innerWallT * 0.62) + (0.335 + 0.040 * sulfurHint) * (innerWallT * 0.62);
              g = g * (1.0 - innerWallT * 0.62) + (0.332 + 0.042 * sulfurHint) * (innerWallT * 0.62);
              b = b * (1.0 - innerWallT * 0.62) + (0.305 + 0.025 * sulfurHint) * (innerWallT * 0.62);
              if (mineralFloorT > 0.01) {
                // Quiet damp geothermal mineral-grey / subtle celadon-silica crater floor
                const fR = 0.275 + 0.048 * sulfurHint;
                const fG = 0.286 + 0.050 * sulfurHint;
                const fB = 0.268 + 0.032 * turfMicro;
                r = r * (1.0 - mineralFloorT) + fR * mineralFloorT;
                g = g * (1.0 - mineralFloorT) + fG * mineralFloorT;
                b = b * (1.0 - mineralFloorT) + fB * mineralFloorT;
              }
            }

            // Mini-Niagara Wet Columnar-Basalt Horseshoe Cliff, Crest-Splitter Boulders, & Plunge Gorge
            const dxN = x - TerrainDEMLoader.NIAGARA_CX;
            const dzN = z - TerrainDEMLoader.NIAGARA_CZ;
            const rN = Math.hypot(dxN, dzN);
            if (rN < 34.5 || (z > 607.5 && z < 632.0 && Math.abs(dxN) < 8.5)) {
              const absAlphaN = Math.abs(Math.atan2(dxN, dzN));
              const cliffWallT =
                (1.0 - PhysicsKernel.smootherstep(0.45, 0.86, nTmp.y)) *
                (1.0 - PhysicsKernel.smootherstep(20.0, 32.0, rN));
              const gorgeBedT =
                rN < 15.8
                  ? 1.0 - PhysicsKernel.smootherstep(13.2, 15.8, rN)
                  : absAlphaN < 1.24 && rN < 31.5
                    ? 1.0 - PhysicsKernel.smootherstep(1.04, 1.26, absAlphaN)
                    : 0.0;
              const outletBedT =
                z > 607.5 && z < 632.0
                  ? 1.0 - PhysicsKernel.smootherstep(3.2, 5.6, Math.abs(dxN))
                  : 0.0;
              const wetBasaltT = Math.min(1.0, Math.max(cliffWallT, gorgeBedT * 0.88, outletBedT * 0.82));
              if (wetBasaltT > 0.01) {
                const mossLedge = cliffWallT * (0.25 + 0.25 * dappleMacro) * nTmp.y;
                const colR = 0.075 + 0.025 * turfMicro + mossLedge * 0.05;
                const colG = 0.088 + 0.028 * turfMicro + mossLedge * 0.18;
                const colB = 0.098 + 0.024 * turfMicro + mossLedge * 0.05;
                r = r * (1.0 - wetBasaltT) + colR * wetBasaltT;
                g = g * (1.0 - wetBasaltT) + colG * wetBasaltT;
                b = b * (1.0 - wetBasaltT) + colB * wetBasaltT;
              }
            }
          }

          // Pulau U Southern Rainforest Canopy Humus + Natural Riparian Point-Bars, Cut-Banks, & Alluvial Riverbed
          if (z < 536.0) {
            if (x < -135.0 && z < 315.0 && y > 1.2) {
              const forestFloorT =
                PhysicsKernel.smootherstep(-135.0, -192.0, x) *
                (1.0 - PhysicsKernel.smootherstep(255.0, 315.0, z)) *
                0.56;
              const leafLitter = 0.5 + 0.5 * Math.sin(x * 0.34 - z * 0.28) * Math.cos(x * 0.19 + z * 0.41);
              const fR = 0.115 + 0.055 * leafLitter + 0.025 * turfMicro;
              const fG = 0.265 + 0.085 * dappleMacro - 0.025 * leafLitter;
              const fB = 0.085 + 0.025 * turfMicro;
              r = r * (1.0 - forestFloorT) + fR * forestFloorT;
              g = g * (1.0 - forestFloorT) + fG * forestFloorT;
              b = b * (1.0 - forestFloorT) + fB * forestFloorT;
            }
            const hydroCol = TerrainDEMLoader.evalPulauUHydrology(x, z);
            if (hydroCol && hydroCol.w < 2.95) {
              const innerBarAffinity = Math.max(0.0, Math.min(1.0, -hydroCol.bendSide * 1.45));
              const outerCutAffinity = Math.max(0.0, Math.min(1.0, hydroCol.bendSide * 1.45));
              const cobVar = 0.5 + 0.5 * Math.sin(x * 1.28 + z * 0.94) * Math.cos(z * 1.36 - x * 0.87);

              // Lush damp riparian fern/moss valley corridor (1.05 < w < 2.95)
              const riparianValleyT =
                PhysicsKernel.smootherstep(0.95, 1.22, hydroCol.w) *
                (1.0 - PhysicsKernel.smootherstep(1.85, 2.95, hydroCol.w)) *
                0.62;
              if (riparianValleyT > 0.01) {
                const ripR = 0.115 + 0.045 * turfMicro;
                const ripG = 0.275 + 0.075 * dappleMacro;
                const ripB = 0.090 + 0.020 * turfMicro;
                r = r * (1.0 - riparianValleyT) + ripR * riparianValleyT;
                g = g * (1.0 - riparianValleyT) + ripG * riparianValleyT;
                b = b * (1.0 - riparianValleyT) + ripB * riparianValleyT;
              }

              // Dry warm alluvial sand & river-pebble point-bar beach on inner meander bends (1.03 < w < 1.56)
              const pointBarT =
                innerBarAffinity *
                PhysicsKernel.smootherstep(0.98, 1.12, hydroCol.w) *
                (1.0 - PhysicsKernel.smootherstep(1.26, 1.62, hydroCol.w)) *
                (0.72 + 0.28 * cobVar);
              if (pointBarT > 0.01) {
                const pbR = 0.295 + 0.055 * cobVar;
                const pbG = 0.262 + 0.048 * cobVar;
                const pbB = 0.195 + 0.036 * cobVar;
                r = r * (1.0 - pointBarT) + pbR * pointBarT;
                g = g * (1.0 - pointBarT) + pbG * pointBarT;
                b = b * (1.0 - pointBarT) + pbB * pointBarT;
              }

              // Eroded dark tropical loam & bryophyte moss on steeper outer meander cut-banks (1.02 < w < 1.55)
              const cutBankT =
                outerCutAffinity *
                PhysicsKernel.smootherstep(0.98, 1.10, hydroCol.w) *
                (1.0 - PhysicsKernel.smootherstep(1.25, 1.58, hydroCol.w)) *
                0.78;
              if (cutBankT > 0.01) {
                const mossStreak = 0.5 + 0.5 * Math.sin(x * 0.85 - z * 0.75);
                const cbR = 0.155 * (1.0 - mossStreak * 0.4) + 0.095 * (mossStreak * 0.4);
                const cbG = 0.135 * (1.0 - mossStreak * 0.5) + 0.225 * (mossStreak * 0.5);
                const cbB = 0.098;
                r = r * (1.0 - cutBankT) + cbR * cutBankT;
                g = g * (1.0 - cutBankT) + cbG * cutBankT;
                b = b * (1.0 - cutBankT) + cbB * cutBankT;
              }

              // Submerged cobble/sand riverbed & darkened wet capillary shoreline fringe (w < 1.15)
              // Matches Regime 1 effDepth -> 0 wet pebble/sand albedo so the water edge vanishes seamlessly!
              const wetBedAndShoreT = 1.0 - PhysicsKernel.smootherstep(0.94, 1.16, hydroCol.w);
              if (wetBedAndShoreT > 0.01) {
                const sandMix = innerBarAffinity * PhysicsKernel.smootherstep(0.35, 0.98, hydroCol.w);
                const wetR = (0.138 + 0.042 * cobVar) * (1.0 - sandMix) + (0.205 + 0.035 * cobVar) * sandMix;
                const wetG = (0.146 + 0.040 * cobVar) * (1.0 - sandMix) + (0.182 + 0.030 * cobVar) * sandMix;
                const wetB = (0.112 + 0.028 * cobVar) * (1.0 - sandMix) + (0.134 + 0.024 * cobVar) * sandMix;
                r = r * (1.0 - wetBedAndShoreT) + wetR * wetBedAndShoreT;
                g = g * (1.0 - wetBedAndShoreT) + wetG * wetBedAndShoreT;
                b = b * (1.0 - wetBedAndShoreT) + wetB * wetBedAndShoreT;
              }
            }
          }
        }

        writeVert(x, y, z, nTmp.x, nTmp.y, nTmp.z, x * 0.08, z * 0.08, r, g, b);
      }
    }

    for (let iz = 0; iz < segsZ; iz++) {
      for (let ix = 0; ix < segsX; ix++) {
        const a = iz * stride + ix;
        const b = a + 1;
        const c = a + stride + 1;
        const d = a + stride;
        indices[iPtr++] = a;
        indices[iPtr++] = d;
        indices[iPtr++] = b;
        indices[iPtr++] = b;
        indices[iPtr++] = d;
        indices[iPtr++] = c;
      }
    }

    // Watertight 4-sided Submerged Seabed Skirt down to y = -18.0m (Hukum Emas 5)
    const skirtY = TerrainDEMLoader.SUB_TERRAIN_SKIRT_Y;
    const addSkirtSide = (edgeSegs, sampleCoordFn, nx, ny, nz) => {
      const topRow = new Uint32Array(edgeSegs + 1);
      const botRow = new Uint32Array(edgeSegs + 1);
      for (let i = 0; i <= edgeSegs; i++) {
        const t = i / edgeSegs;
        const { x, z, arcU } = sampleCoordFn(t);
        const yTop = TerrainDEMLoader.sampleGroundHeight(x, z);
        topRow[i] = writeVert(x, yTop, z, nx, ny, nz, arcU * 0.06, 1.0, 0.16, 0.17, 0.15);
        botRow[i] = writeVert(x, skirtY, z, nx, ny, nz, arcU * 0.06, 0.0, 0.10, 0.11, 0.10);
      }
      for (let i = 0; i < edgeSegs; i++) {
        const t0 = topRow[i];
        const t1 = topRow[i + 1];
        const b0 = botRow[i];
        const b1 = botRow[i + 1];
        indices[iPtr++] = t0;
        indices[iPtr++] = b0;
        indices[iPtr++] = b1;
        indices[iPtr++] = t0;
        indices[iPtr++] = b1;
        indices[iPtr++] = t1;
      }
    };

    addSkirtSide(segsX, (t) => ({ x: mapX(t), z: zMax, arcU: t * spanX }), 0, 0, 1);
    addSkirtSide(segsZ, (t) => ({ x: xMax, z: mapZ(1.0 - t), arcU: t * spanZ }), 1, 0, 0);
    addSkirtSide(segsX, (t) => ({ x: mapX(1.0 - t), z: zMin, arcU: t * spanX }), 0, 0, -1);
    addSkirtSide(segsZ, (t) => ({ x: xMin, z: mapZ(t), arcU: t * spanZ }), -1, 0, 0);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    return geo;
  }

  /**
   * Projects a closed 2D seed polyline onto the exact SDF zero-contour sdfFn(x, z) === 0
   * via damped Newton-Raphson steps, resamples to exact uniform arc-length, and orients
   * the loop so outward normals x tangent yield right-handed outward/upward faces.
   */
  static _buildUniformSDFCoastRing(seedPts, sdfFn, targetSegs) {
    const eps = 0.25;
    const evalGrad = (px, pz) => {
      const gx = (sdfFn(px + eps, pz) - sdfFn(px - eps, pz)) / (2.0 * eps);
      const gz = (sdfFn(px, pz + eps) - sdfFn(px, pz - eps)) / (2.0 * eps);
      const len = Math.hypot(gx, gz) || 1.0;
      return { nx: gx / len, nz: gz / len, len };
    };

    const projectPt = (pt, iters = 6) => {
      let px = pt.x;
      let pz = pt.z;
      for (let k = 0; k < iters; k++) {
        const d = sdfFn(px, pz);
        if (Math.abs(d) < 1e-6) break;
        const g = evalGrad(px, pz);
        const step = Math.max(-5.0, Math.min(5.0, d / Math.max(0.25, g.len)));
        px -= g.nx * step;
        pz -= g.nz * step;
      }
      return { x: px, z: pz };
    };

    // 1. Project initial dense seed vertices onto exact sdf(x, z) = 0
    const projected = seedPts.map((p) => projectPt(p, 7));

    // 2. Measure closed cumulative perimeter arc-length
    const nSeed = projected.length;
    const cumLen = new Float64Array(nSeed + 1);
    cumLen[0] = 0.0;
    for (let i = 0; i < nSeed; i++) {
      const a = projected[i];
      const b = projected[(i + 1) % nSeed];
      cumLen[i + 1] = cumLen[i] + Math.hypot(b.x - a.x, b.z - a.z);
    }
    const totalPerim = cumLen[nSeed];

    // 3. Uniform arc-length resampling + final 4-step Newton-Raphson snap onto sdf = 0
    const uniformPts = [];
    let segCursor = 0;
    for (let i = 0; i < targetSegs; i++) {
      const targetS = (i / targetSegs) * totalPerim;
      while (segCursor < nSeed - 1 && cumLen[segCursor + 1] < targetS) {
        segCursor++;
      }
      const s0 = cumLen[segCursor];
      const s1 = cumLen[segCursor + 1];
      const frac = s1 > s0 ? (targetS - s0) / (s1 - s0) : 0.0;
      const pA = projected[segCursor];
      const pB = projected[(segCursor + 1) % nSeed];
      uniformPts.push(
        projectPt(
          {
            x: pA.x + (pB.x - pA.x) * frac,
            z: pA.z + (pB.z - pA.z) * frac
          },
          4
        )
      );
    }

    // 4. Ensure winding satisfies (onx * tz - onz * tx) > 0 so face normals point strictly outward/upward!
    let orientScore = 0.0;
    for (let i = 0; i < targetSegs; i++) {
      const prev = uniformPts[(i - 1 + targetSegs) % targetSegs];
      const curr = uniformPts[i];
      const next = uniformPts[(i + 1) % targetSegs];
      const g = evalGrad(curr.x, curr.z);
      const tx = next.x - prev.x;
      const tz = next.z - prev.z;
      orientScore += g.nx * tz - g.nz * tx;
    }
    if (orientScore < 0.0) {
      uniformPts.reverse();
    }

    // 5. Build closed ringStations [0 .. targetSegs] with exact closure at index === targetSegs
    const ringStations = [];
    let cumArc = 0.0;
    for (let i = 0; i <= targetSegs; i++) {
      const idx = i % targetSegs;
      const curr = uniformPts[idx];
      const g = evalGrad(curr.x, curr.z);
      if (i > 0) {
        const prev = ringStations[i - 1];
        cumArc += Math.hypot(curr.x - prev.x0, curr.z - prev.z0);
      }
      ringStations.push({
        phase: (i / targetSegs) * Math.PI * 2.0,
        x0: curr.x,
        z0: curr.z,
        onx: g.nx,
        onz: g.nz,
        arc: cumArc
      });
    }
    return ringStations;
  }

  /**
   * Generates ordered closed seed loops for Pulau "U" (horseshoe + Campus Head) and Pulau "I" (rounded stadium).
   */
  static _generatePulauUAndPulauIRings() {
    // --- A. Ordered Closed Seed Loop for Pulau "U" ---
    const seedU = [];
    const cx = TerrainDEMLoader.ISLAND_CX;
    const cz = TerrainDEMLoader.ISLAND_CZ;

    // 1. Campus Head arc from Inner SW Helipad junction (theta = +2.56) -> Rotunda -> East Danau Kenanga -> North -> Outer West junction (theta = -2.54)
    const headSteps = 260;
    const thStart = 2.56;
    const thEnd = -2.54;
    for (let i = 0; i <= headSteps; i++) {
      const th = thStart + (thEnd - thStart) * (i / headSteps);
      const R = TerrainDEMLoader.getIslandRadiusAtAngle(th);
      seedU.push({ x: cx + R * Math.cos(th), z: cz + R * Math.sin(th) });
    }

    // 2. Outer Bottom Semicircle of Pulau "U" around (U_ARC_X = -145, U_ARC_Z = 207), radius = 185 + 74 = 259m
    const rOuterArc = TerrainDEMLoader.U_ARC_R + TerrainDEMLoader.U_ARM_HALF_W;
    const outerArcSteps = 180;
    for (let i = 1; i <= outerArcSteps; i++) {
      // Sweep from Left (-Z side: z = 207 - 259 = -52) through Bottom (-X apex: x = -404) to Right (+Z side: z = +466)
      const phi = (i / outerArcSteps) * Math.PI; // 0 -> PI
      seedU.push({
        x: TerrainDEMLoader.U_ARC_X - Math.sin(phi) * rOuterArc,
        z: TerrainDEMLoader.U_ARC_Z - Math.cos(phi) * rOuterArc
      });
    }

    // 3. Outer Right Wall of Pulau "U" going up (+X) from x = -145 to x = +10 at z = +466
    const rightWallSteps = 60;
    for (let i = 1; i <= rightWallSteps; i++) {
      const t = i / rightWallSteps;
      seedU.push({
        x: TerrainDEMLoader.U_ARC_X + (TerrainDEMLoader.U_TOP_RIGHT_X - TerrainDEMLoader.U_ARC_X) * t,
        z: TerrainDEMLoader.U_RIGHT_Z + TerrainDEMLoader.U_ARM_HALF_W
      });
    }

    // 4. Rounded Top Headland of Right Arm of Pulau "U" around (+10, +392), radius = 74m
    const capSteps = 56;
    for (let i = 1; i <= capSteps; i++) {
      const phi = (i / capSteps) * Math.PI; // 0 (+Z) -> PI/2 (+X top tip) -> PI (-Z inner shore)
      seedU.push({
        x: TerrainDEMLoader.U_TOP_RIGHT_X + Math.sin(phi) * TerrainDEMLoader.U_ARM_HALF_W,
        z: TerrainDEMLoader.U_RIGHT_Z + Math.cos(phi) * TerrainDEMLoader.U_ARM_HALF_W
      });
    }

    // 5. Inner Right Wall of Teluk "U" going down (-X) from x = +10 to x = -145 at z = +318
    for (let i = 1; i <= rightWallSteps; i++) {
      const t = i / rightWallSteps;
      seedU.push({
        x: TerrainDEMLoader.U_TOP_RIGHT_X + (TerrainDEMLoader.U_ARC_X - TerrainDEMLoader.U_TOP_RIGHT_X) * t,
        z: TerrainDEMLoader.U_RIGHT_Z - TerrainDEMLoader.U_ARM_HALF_W
      });
    }

    // 6. Inner Lagoon Bowl Semicircle of Teluk "U" around (-145, +207), radius = 185 - 74 = 111m
    const rInnerArc = TerrainDEMLoader.U_ARC_R - TerrainDEMLoader.U_ARM_HALF_W;
    const innerArcSteps = 110;
    for (let i = 1; i < innerArcSteps; i++) {
      const phi = (i / innerArcSteps) * Math.PI; // 0 (z = +318) -> PI/2 (x = -256) -> PI (z = +96)
      seedU.push({
        x: TerrainDEMLoader.U_ARC_X - Math.sin(phi) * rInnerArc,
        z: TerrainDEMLoader.U_ARC_Z + Math.cos(phi) * rInnerArc
      });
    }

    const ringU = TerrainDEMLoader._buildUniformSDFCoastRing(
      seedU,
      TerrainDEMLoader.getIslandUCoastDistance,
      720
    );

    // --- B. Ordered Closed Seed Loop for Pulau "I" ---
    const seedI = [];
    const stepsI = 320;
    for (let i = 0; i < stepsI; i++) {
      const ang = (i / stepsI) * Math.PI * 2.0;
      // Superellipse initial guess closely hugging the 488m x 148m rounded-box shoreline of Pulau I
      const cA = Math.cos(ang);
      const sA = Math.sin(ang);
      const powC = Math.sign(cA) * Math.pow(Math.abs(cA), 0.55);
      const powS = Math.sign(sA) * Math.pow(Math.abs(sA), 0.55);
      seedI.push({
        x: TerrainDEMLoader.I_CENTER_X + powC * TerrainDEMLoader.I_HALF_LEN_X,
        z: TerrainDEMLoader.I_CENTER_Z + powS * TerrainDEMLoader.I_HALF_WID_Z
      });
    }

    const ringI = TerrainDEMLoader._buildUniformSDFCoastRing(
      seedI,
      TerrainDEMLoader.getIslandICoastDistance,
      420
    );

    return { ringU, ringI };
  }

  /**
   * Builds the 100% Closed 360° Coursed River-Stone Revetment Walls (Turap Batu Kali)
   * around BOTH Pulau "U" (720 segments + 84 buttress piers & bollards) AND Pulau "I" (420 segments + 48 piers & bollards)!
   */
  static _buildLakesideStoneTurapGeometry() {
    const positions = [];
    const normals = [];
    const uvs = [];
    const colors = [];
    const indices = [];

    const pushVert = (px, py, pz, nx, ny, nz, u, v, r, g, b) => {
      const idx = positions.length / 3;
      positions.push(px, py, pz);
      normals.push(nx, ny, nz);
      uvs.push(u, v);
      colors.push(r, g, b);
      return idx;
    };

    const profile = [
      { dr: -0.35, y: -2.35, r: 0.30, g: 0.29, b: 0.26, masonryRelief: 0.0 },
      { dr: -0.35, y: 0.10, r: 0.56, g: 0.54, b: 0.50, masonryRelief: 0.0 },
      { dr: -0.25, y: 0.19, r: 0.65, g: 0.63, b: 0.59, masonryRelief: 0.0 },
      { dr: 0.36, y: 0.19, r: 0.68, g: 0.66, b: 0.61, masonryRelief: 0.0 },
      { dr: 0.44, y: 0.12, r: 0.62, g: 0.60, b: 0.56, masonryRelief: 0.0 },
      { dr: 0.44, y: 0.02, r: 0.48, g: 0.46, b: 0.42, masonryRelief: 0.015 },
      { dr: 0.72, y: -0.24, r: 0.49, g: 0.47, b: 0.43, masonryRelief: 0.032 },
      { dr: 1.15, y: -0.30, r: 0.52, g: 0.50, b: 0.46, masonryRelief: 0.025 },
      { dr: 1.85, y: -0.68, r: 0.45, g: 0.43, b: 0.40, masonryRelief: 0.042 },
      { dr: 2.32, y: -0.76, r: 0.47, g: 0.45, b: 0.41, masonryRelief: 0.035 },
      { dr: 3.38, y: -1.36, r: 0.37, g: 0.36, b: 0.33, masonryRelief: 0.048 },
      { dr: 4.15, y: -1.85, r: 0.28, g: 0.29, b: 0.26, masonryRelief: 0.04 },
      { dr: 5.20, y: -2.60, r: 0.22, g: 0.23, b: 0.21, masonryRelief: 0.0 }
    ];

    const appendOrientedTurapBox = (px, py, pz, sx, sy, sz, yawY, cr, cg, cb) => {
      const box = new THREE.BoxGeometry(sx, sy, sz);
      box.rotateY(yawY);
      box.translate(px, py, pz);
      const p = box.attributes.position.array;
      const n = box.attributes.normal.array;
      const u = box.attributes.uv.array;
      const base = positions.length / 3;
      for (let i = 0; i < box.attributes.position.count; i++) {
        positions.push(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
        normals.push(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]);
        uvs.push(u[i * 2], u[i * 2 + 1]);
        colors.push(cr, cg, cb);
      }
      const idx = box.index.array;
      for (let i = 0; i < idx.length; i++) {
        indices.push(base + idx[i]);
      }
    };

    const extrudeClosedTurapRing = (ringStations, pierCount) => {
      const angSegs = ringStations.length - 1;
      const nVec = new THREE.Vector3();

      for (let s = 0; s < profile.length - 1; s++) {
        const p0 = profile[s];
        const p1 = profile[s + 1];
        const dDr = p1.dr - p0.dr;
        const dDy = p1.y - p0.y;
        const nHoriz = -dDy;
        const nVert = dDr;

        const rowA = [];
        const rowB = [];

        for (let ia = 0; ia <= angSegs; ia++) {
          const st = ringStations[ia];
          const iaMod = ia % angSegs;
          nVec.set(st.onx * nHoriz, nVert, st.onz * nHoriz).normalize();

          const blockWave0 = Math.sin(st.phase * 180.0 + s * 1.7) * Math.cos(st.phase * 96.0 - s * 0.8);
          const blockWave1 =
            Math.sin(st.phase * 180.0 + (s + 1) * 1.7) * Math.cos(st.phase * 96.0 - (s + 1) * 0.8);
          const stoneVar = (((iaMod * 17 + s * 31) % 13) - 6) * 0.011 + blockWave0 * 0.018;

          const offA = p0.dr + p0.masonryRelief * blockWave0;
          const offB = p1.dr + p1.masonryRelief * blockWave1;

          rowA.push(
            pushVert(
              st.x0 + offA * st.onx,
              p0.y,
              st.z0 + offA * st.onz,
              nVec.x,
              nVec.y,
              nVec.z,
              st.arc * 0.18,
              s * 0.28,
              Math.max(0.05, p0.r + stoneVar),
              Math.max(0.05, p0.g + stoneVar),
              Math.max(0.05, p0.b + stoneVar)
            )
          );
          rowB.push(
            pushVert(
              st.x0 + offB * st.onx,
              p1.y,
              st.z0 + offB * st.onz,
              nVec.x,
              nVec.y,
              nVec.z,
              st.arc * 0.18,
              (s + 1) * 0.28,
              Math.max(0.05, p1.r + stoneVar),
              Math.max(0.05, p1.g + stoneVar),
              Math.max(0.05, p1.b + stoneVar)
            )
          );
        }

        for (let ia = 0; ia < angSegs; ia++) {
          const a = rowA[ia];
          const b = rowA[ia + 1];
          const c = rowB[ia + 1];
          const d = rowB[ia];
          indices.push(a, b, c, a, c, d);
        }
      }

      for (let ip = 0; ip < pierCount; ip++) {
        const stIdx = Math.round((ip / pierCount) * angSegs) % angSegs;
        const st = ringStations[stIdx];
        const yawOut = Math.atan2(-st.onz, st.onx);
        appendOrientedTurapBox(
          st.x0 + 1.15 * st.onx,
          -0.72,
          st.z0 + 1.15 * st.onz,
          2.35,
          1.86,
          0.78,
          yawOut,
          0.50,
          0.48,
          0.44
        );
        appendOrientedTurapBox(
          st.x0 + 0.08 * st.onx,
          0.28,
          st.z0 + 0.08 * st.onz,
          0.86,
          0.26,
          0.86,
          yawOut,
          0.72,
          0.70,
          0.65
        );
      }
    };

    const { ringU, ringI } = TerrainDEMLoader._generatePulauUAndPulauIRings();
    extrudeClosedTurapRing(ringU, 84);
    extrudeClosedTurapRing(ringI, 48);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
    return geo;
  }

  /**
   * Generates a deterministic curated catalog of ~94 sculpted 3D alluvial river boulders & cobble clusters
   * along Pulau U's Mata Air rim, riffle rapids, inner gravel point-bars, mossy outer cut-bank toes,
   * and Danau Kecil Rektorat inlet. Shared between 3D terrain rock batching (+0 Draw Calls) and
   * hydrodynamic bow-wave / Kelvin-wake computation on the river water surface!
   */
  static getPulauURiverBoulders() {
    if (TerrainDEMLoader._cachedPulauURiverBoulders) {
      return TerrainDEMLoader._cachedPulauURiverBoulders;
    }

    const boulders = [];
    const hash01 = (seed) => {
      const s = Math.sin(seed * 127.1 + 311.7) * 43758.5453123;
      return s - Math.floor(s);
    };

    const pushRiverBoulder = (x, z, rx, ry, rz, yaw, seed, inStream = false, targetCrownAboveWater = 0.0) => {
      const hydro = TerrainDEMLoader.evalPulauUHydrology(x, z);
      const groundY = TerrainDEMLoader.sampleGroundHeight(x, z);
      const waterY = hydro ? hydro.waterY : groundY - 0.5;
      let cy = groundY + ry * 0.42;
      if (inStream && hydro) {
        // Seat boulder firmly into riverbed while letting its mossy rounded crown poke above the stream
        const crownY = waterY + targetCrownAboveWater;
        const neededRy = Math.max(ry, (crownY - groundY) * 0.62);
        ry = Math.min(1.05, neededRy);
        cy = crownY - ry * 0.88;
      }
      boulders.push({
        x,
        cy,
        z,
        rx,
        ry,
        rz,
        yaw,
        seed,
        waterY,
        inStream,
        wakeRadius: Math.max(rx, rz)
      });
    };

    // 1. 22 Emergent In-Stream Riffle & Meander-Shoal Boulders (create real 3D bow waves & downstream wakes!)
    const riffleStations = [
      -276.5, -268.0, -259.5, -251.0, -242.5, -233.0, -224.5, -216.0, -207.5, -198.5, -189.0,
      -180.5, -171.5, -162.5, -153.5, -144.0, -135.5, -126.5, -118.0, -110.5, -103.5, -97.5
    ];
    for (let i = 0; i < riffleStations.length; i++) {
      const sx = riffleStations[i];
      const sz = TerrainDEMLoader.evalPulauURiverCenterZ(sx);
      const m = TerrainDEMLoader.evalPulauURiverSlopeZ(sx);
      const invT = 1.0 / Math.hypot(1.0, m);
      const tx = invT;
      const tz = m * invT;
      const cxVec = -tz;
      const czVec = tx;
      const curv = TerrainDEMLoader.evalPulauURiverCurvatureZ(sx);
      const bendBias = Math.tanh(-curv * 34.0);
      const hw = TerrainDEMLoader.evalPulauURiverHalfWidth(sx);

      // Prefer shallower inner-shoal / mid-riffle positions (|crossU| in 0.26 .. 0.74)
      const side = hash01(i * 7.3 + 1.1) < 0.62 ? (bendBias >= 0 ? -1.0 : 1.0) : (bendBias >= 0 ? 1.0 : -1.0);
      const crossMag = 0.28 + 0.44 * hash01(i * 11.7 + 2.3);
      const crossU = side * crossMag;
      const bx = sx + cxVec * crossU * hw;
      const bz = sz + czVec * crossU * hw;
      const rx = 0.46 + 0.42 * hash01(i * 17.1 + 3.7);
      const rz = 0.38 + 0.34 * hash01(i * 23.9 + 5.1);
      const ry = 0.30 + 0.22 * hash01(i * 29.3 + 6.9);
      const yaw = Math.atan2(tz, tx) + (hash01(i * 31.7) - 0.5) * 0.7;
      const crownAbove = 0.11 + 0.24 * hash01(i * 41.3 + 0.9);
      pushRiverBoulder(bx, bz, rx, ry, rz, yaw, i * 13.7 + 1.9, true, crownAbove);
    }

    // 2. 54 Shoreline, Gravel Point-Bar, & Cut-Bank Toe Boulder/Cobble Clusters (break up bank edges organically!)
    for (let i = 0; i < 36; i++) {
      const u = (i + 0.45) / 36.0;
      const sx =
        TerrainDEMLoader.RIVER_U_X_START +
        2.5 +
        u * (TerrainDEMLoader.RIVER_U_X_END - TerrainDEMLoader.RIVER_U_X_START - 3.5);
      const sz = TerrainDEMLoader.evalPulauURiverCenterZ(sx);
      const m = TerrainDEMLoader.evalPulauURiverSlopeZ(sx);
      const invT = 1.0 / Math.hypot(1.0, m);
      const cxVec = -m * invT;
      const czVec = invT;
      const sideSign = i % 2 === 0 ? 1.0 : -1.0;
      const bankScallop =
        0.25 * Math.sin(sx * 0.43 + sideSign * 1.75) +
        0.16 * Math.cos(sx * 0.94 - sideSign * 0.85);
      const effHw = Math.max(3.40, TerrainDEMLoader.evalPulauURiverHalfWidth(sx) + bankScallop);
      const crossU = sideSign * (0.93 + 0.26 * hash01(i * 19.3 + 4.2));
      const bx = sx + cxVec * crossU * effHw;
      const bz = sz + czVec * crossU * effHw;
      const rx = 0.44 + 0.52 * hash01(i * 27.1 + 8.4);
      const rz = 0.38 + 0.44 * hash01(i * 33.7 + 2.6);
      const ry = 0.24 + 0.28 * hash01(i * 39.1 + 7.5);
      const yaw = hash01(i * 47.9) * Math.PI * 2.0;
      pushRiverBoulder(bx, bz, rx, ry, rz, yaw, 100.0 + i * 9.3, Math.abs(crossU) < 1.02, 0.16);

      // Companion smaller river cobble beside every second shoreline boulder for natural clustering
      if (i % 2 === 0) {
        const offAng = hash01(i * 53.1 + 1.4) * Math.PI * 2.0;
        const offD = rx * 0.85 + 0.28;
        pushRiverBoulder(
          bx + Math.cos(offAng) * offD,
          bz + Math.sin(offAng) * offD,
          rx * 0.56,
          ry * 0.58,
          rz * 0.54,
          yaw + 1.1,
          200.0 + i * 7.7,
          false,
          0.0
        );
      }
    }

    // 3. 18 Mossy Artesian Spring ("Mata Air") Rim & Danau Kecil Rektorat Inlet Boulders
    for (let i = 0; i < 10; i++) {
      const th = (i / 10) * Math.PI * 2.0 + 0.32;
      if (Math.abs(th) < 0.36) continue; // Keep spring-to-river east outlet open
      const rSpr = TerrainDEMLoader.evalPulauUSpringRadiusAtAngle(th) * (0.95 + 0.14 * hash01(i * 15.7));
      const bx = TerrainDEMLoader.SPRING_U_CX + Math.cos(th) * rSpr;
      const bz = TerrainDEMLoader.SPRING_U_CZ + Math.sin(th) * rSpr;
      pushRiverBoulder(
        bx,
        bz,
        0.58 + 0.38 * hash01(i * 21.3),
        0.34 + 0.24 * hash01(i * 25.9),
        0.50 + 0.34 * hash01(i * 31.1),
        th + 0.6,
        300.0 + i * 11.1,
        false,
        0.0
      );
    }
    for (let i = 0; i < 9; i++) {
      const th = Math.PI * (0.62 + (i / 8) * 0.86); // Western & southern forested shore of Danau Kecil Rektorat
      const mod = TerrainDEMLoader.evalPulauULakeRadiusMod(th) * (0.97 + 0.08 * hash01(i * 18.9));
      const bx = TerrainDEMLoader.LAKE_U_CX + Math.cos(th) * TerrainDEMLoader.LAKE_U_RX * mod;
      const bz = TerrainDEMLoader.LAKE_U_CZ + Math.sin(th) * TerrainDEMLoader.LAKE_U_RZ * mod;
      if (bx > -71.5) continue;
      pushRiverBoulder(
        bx,
        bz,
        0.50 + 0.36 * hash01(i * 22.7),
        0.28 + 0.20 * hash01(i * 28.3),
        0.44 + 0.32 * hash01(i * 35.3),
        th,
        400.0 + i * 13.3,
        false,
        0.0
      );
    }

    TerrainDEMLoader._cachedPulauURiverBoulders = boulders;
    return boulders;
  }

  /**
   * Builds watertight 3D sculpted mossy river-worn andesite boulders & cobbles merged directly
   * into `KampusUITerrainAndDanauKenanga` static land batch (+0 Draw Calls, 0 degenerate normals!).
   */
  static _buildFreshwaterRiverBouldersGeometry() {
    const boulders = TerrainDEMLoader.getPulauURiverBoulders();
    const uSegs = 14;
    const vRings = 7;

    const positions = [];
    const uvs = [];
    const colors = [];
    const indices = [];

    const evalSculptedLocalPoint = (phi, th, b) => {
      const sPhi = Math.sin(phi);
      const cPhi = Math.cos(phi);
      const ux = sPhi * Math.cos(th);
      const uy = cPhi;
      const uz = sPhi * Math.sin(th);

      // Organic river-worn rock faceting + flattened riverbed base
      const facet1 = 0.14 * Math.sin(2.0 * th + b.seed) * sPhi + 0.10 * Math.cos(3.0 * th - b.seed * 0.7) * sPhi;
      const facet2 = 0.07 * Math.sin(4.0 * phi + 2.0 * th + b.seed * 1.3);
      const flatBottom = uy < 0.0 ? 0.68 : 1.0;
      const radMod = 1.0 + facet1 + facet2;

      const lx = ux * b.rx * radMod;
      const ly = uy * b.ry * radMod * flatBottom;
      const lz = uz * b.rz * radMod;

      const cY = Math.cos(b.yaw);
      const sY = Math.sin(b.yaw);
      const wx = b.x + lx * cY - lz * sY;
      const wy = b.cy + ly;
      const wz = b.z + lx * sY + lz * cY;
      return { wx, wy, wz, uy };
    };

    const pushBoulderVert = (pt, b) => {
      const idx = positions.length / 3;
      positions.push(pt.wx, pt.wy, pt.wz);
      uvs.push(pt.wx * 0.25, pt.wz * 0.25);

      // 3-zone river stone coloration: dark wet waterline collar, mottled river andesite, & velvet bryophyte moss crown
      const aboveWater = pt.wy - b.waterY;
      const wetT = 1.0 - PhysicsKernel.smootherstep(-0.04, 0.13, aboveWater);
      const forestShade = PhysicsKernel.smootherstep(-105.0, -185.0, pt.wx);
      const mottle = 0.5 + 0.5 * Math.sin(pt.wx * 3.1 + b.seed) * Math.cos(pt.wz * 2.9 - b.seed);
      const mossT =
        PhysicsKernel.smootherstep(0.32, 0.76, pt.uy) *
        PhysicsKernel.smootherstep(0.02, 0.14, aboveWater) *
        (0.38 + 0.54 * forestShade) *
        (0.65 + 0.35 * mottle);

      let cr = 0.225 + 0.045 * mottle;
      let cg = 0.212 + 0.040 * mottle;
      let cb = 0.184 + 0.032 * mottle;

      // Upper velvet bryophyte moss cap
      cr = cr * (1.0 - mossT) + (0.115 + 0.03 * mottle) * mossT;
      cg = cg * (1.0 - mossT) + (0.265 + 0.05 * mottle) * mossT;
      cb = cb * (1.0 - mossT) + 0.088 * mossT;

      // Darkened wet stone collar & submerged periphyton sheen
      cr = cr * (1.0 - wetT) + 0.092 * wetT;
      cg = cg * (1.0 - wetT) + 0.098 * wetT;
      cb = cb * (1.0 - wetT) + 0.082 * wetT;

      colors.push(cr, cg, cb);
      return idx;
    };

    for (let ib = 0; ib < boulders.length; ib++) {
      const b = boulders[ib];
      const poleN = pushBoulderVert(evalSculptedLocalPoint(0.0, 0.0, b), b);
      const rings = [];
      for (let ir = 1; ir <= vRings; ir++) {
        const phi = (ir / (vRings + 1)) * Math.PI;
        const row = [];
        for (let iu = 0; iu < uSegs; iu++) {
          const th = (iu / uSegs) * Math.PI * 2.0;
          row.push(pushBoulderVert(evalSculptedLocalPoint(phi, th, b), b));
        }
        row.push(row[0]); // Closed ring seam without duplicate vertex
        rings.push(row);
      }
      const poleS = pushBoulderVert(evalSculptedLocalPoint(Math.PI, 0.0, b), b);

      // Top pole fan (+Y outward winding)
      const firstRow = rings[0];
      for (let iu = 0; iu < uSegs; iu++) {
        indices.push(poleN, firstRow[iu + 1], firstRow[iu]);
      }
      // Intermediate quad bands (strictly outward winding: +theta x +phi)
      for (let ir = 0; ir < vRings - 1; ir++) {
        const r0 = rings[ir];
        const r1 = rings[ir + 1];
        for (let iu = 0; iu < uSegs; iu++) {
          const p00 = r0[iu];
          const p01 = r0[iu + 1];
          const p10 = r1[iu];
          const p11 = r1[iu + 1];
          indices.push(p00, p01, p10, p01, p11, p10);
        }
      }
      // Bottom pole fan (-Y outward winding)
      const lastRow = rings[vRings - 1];
      for (let iu = 0; iu < uSegs; iu++) {
        indices.push(poleS, lastRow[iu], lastRow[iu + 1]);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    geo.setAttribute(
      'normal',
      new THREE.BufferAttribute(new Float32Array(positions.length), 3)
    );
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
    geo.computeVertexNormals();
    return geo;
  }

  /**
   * Builds the Seamless 4-Zone Unified Ocean, Freshwater Hydrology, & Mini-Niagara Cataract Geometry
   * (Strictly 1 Single Draw Call via `aHydroMeta` + `aRiverDetail` regime dispatch!):
   * - Zone 1 & 2: 3.3 km Archipelago Ocean, Teluk U Lagoon, & Selat UI Strait (`aHydroMeta.x = 0.0`)
   * - Zone 3: Pulau U Artesian Forest Spring ("Mata Air") -> 193m 4-Bend Meandering River -> Danau Kecil Rektorat (`aHydroMeta.x = 1.0`)
   * - Zone 4: Pulau I Mini-Niagara Upper Basalt Stream, 3D Parabolic Concave Horseshoe Waterfall Curtain (`aHydroMeta.x = 2.0`),
   *   Turquoise Plunge Pool, & Western Tailrace Outlet (`aHydroMeta.x = 1.0`)!
   */
  static _buildDanauKenangaWaterGeometry() {
    const yWater = TerrainDEMLoader.WATER_LEVEL_Y;
    const xMin = -465.0;
    const xMax = 145.0;
    const zMin = -155.0;
    const zMax = 845.0;
    const nx = 216;
    const nz = 344;
    const stepX = (xMax - xMin) / nx;
    const stepZ = (zMax - zMin) / nz;

    const positions = [];
    const normals = [];
    const uvs = [];
    const hydroMeta = [];
    const riverDetail = [];
    const indices = [];

    const emergentBoulders = TerrainDEMLoader.getPulauURiverBoulders().filter((b) => b.inStream);

    const evalBoulderWakeAt = (px, pz, tx, tz) => {
      let wakeSum = 0.0;
      const cxV = -tz;
      const czV = tx;
      for (let i = 0; i < emergentBoulders.length; i++) {
        const b = emergentBoulders[i];
        const dx = px - b.x;
        const dz = pz - b.z;
        if (Math.abs(dx) > 4.8 || Math.abs(dz) > 4.8) continue;
        const br = Math.max(0.38, b.wakeRadius);
        const dAlong = dx * tx + dz * tz; // > 0 downstream, < 0 upstream
        const dCross = dx * cxV + dz * czV;

        // Upstream hydraulic bow-wave cushion & stone-collar wrap
        if (dAlong > -1.7 * br && dAlong < 0.85 * br) {
          const uA = (dAlong + 0.25 * br) / (0.85 * br);
          const uC = dCross / (1.05 * br);
          wakeSum += 1.05 * Math.exp(-(uA * uA + uC * uC));
        }
        // Downstream spreading Kelvin V-wake & whitewater micro-bubble trail
        if (dAlong >= -0.2 * br && dAlong < 5.2 * br) {
          const sDown = Math.max(0.0, dAlong / br);
          const halfSpread = br * (0.72 + 0.30 * sDown);
          const vArm = Math.exp(-Math.pow((Math.abs(dCross) - halfSpread * 0.68) / (0.42 * br), 2.0));
          const core = Math.exp(-Math.pow(dCross / halfSpread, 2.0));
          wakeSum += (0.58 * core + 0.52 * vArm) * Math.exp(-sDown * 0.42);
        }
      }
      return Math.min(1.65, wakeSum);
    };

    const pushCustomWaterVert = (
      px,
      py,
      pz,
      nxV,
      nyV,
      nzV,
      u,
      v,
      h0,
      h1,
      h2,
      h3,
      d0 = 2.5,
      d1 = 0.0,
      d2 = 0.0,
      d3 = 0.0
    ) => {
      const idx = positions.length / 3;
      positions.push(px, py, pz);
      normals.push(nxV, nyV, nzV);
      uvs.push(u, v);
      hydroMeta.push(h0, h1, h2, h3);
      riverDetail.push(d0, d1, d2, d3);
      return idx;
    };

    const pushWaterVert = (x, z) =>
      pushCustomWaterVert(x, yWater, z, 0, 1, 0, x * 0.04, z * 0.04, 0.0, 0.0, 0.0, 0.0);

    // Zone 1: Inner Archipelago Surf, Lagoon (Teluk U), & Bridge Strait (Selat UI) Grid
    const stride = nx + 1;
    const coastCache = new Float32Array((nx + 1) * (nz + 1));
    for (let iz = 0; iz <= nz; iz++) {
      const z = zMin + iz * stepZ;
      for (let ix = 0; ix <= nx; ix++) {
        const x = xMin + ix * stepX;
        pushWaterVert(x, z);
        coastCache[iz * stride + ix] = TerrainDEMLoader.getIslandCoastDistance(x, z);
      }
    }

    for (let iz = 0; iz < nz; iz++) {
      for (let ix = 0; ix < nx; ix++) {
        const a = iz * stride + ix;
        const b = a + 1;
        const c = a + stride + 1;
        const d = a + stride;
        // Cull quads whose 4 corners are all safely inland behind the stone turap (dCoast < 0.85m; waterline is +3.52m)
        const maxD = Math.max(coastCache[a], coastCache[b], coastCache[c], coastCache[d]);
        if (maxD < 0.85) continue;
        // Right-handed +Y upward winding: (ad = +Z) x (ab = +X) = (0, +1, 0)
        indices.push(a, d, b, b, d, c);
      }
    }

    // Zone 2: Seamless Crack-Free Outer Ocean Skirt from Inner Grid Perimeter -> R = 1,650m Circle!
    const perimIndices = [];
    for (let ix = 0; ix < nx; ix++) perimIndices.push(nz * stride + ix);
    for (let iz = nz; iz > 0; iz--) perimIndices.push(iz * stride + nx);
    for (let ix = nx; ix > 0; ix--) perimIndices.push(ix);
    for (let iz = 0; iz < nz; iz++) perimIndices.push(iz * stride);

    const nPerim = perimIndices.length;
    const radRings = 28;
    const acx = TerrainDEMLoader.ARCHIPELAGO_CX;
    const acz = TerrainDEMLoader.ARCHIPELAGO_CZ;
    const rOuter = TerrainDEMLoader.WATER_OUTER_RADIUS;

    let prevRing = perimIndices.slice();
    prevRing.push(perimIndices[0]); // Close seam

    for (let ir = 1; ir <= radRings; ir++) {
      const u = ir / radRings;
      const t = Math.pow(u, 2.15);
      const currRing = [];

      for (let ip = 0; ip < nPerim; ip++) {
        const baseIdx = perimIndices[ip];
        const bx = positions[baseIdx * 3];
        const bz = positions[baseIdx * 3 + 2];
        const dx = bx - acx;
        const dz = bz - acz;
        const dLen = Math.hypot(dx, dz) || 1.0;
        const ox = acx + (dx / dLen) * rOuter;
        const oz = acz + (dz / dLen) * rOuter;

        const wx = bx + (ox - bx) * t;
        const wz = bz + (oz - bz) * t;
        currRing.push(pushWaterVert(wx, wz));
      }
      currRing.push(currRing[0]); // Exact seam closure without duplicate vertex!

      for (let ip = 0; ip < nPerim; ip++) {
        const p0 = prevRing[ip];
        const p1 = prevRing[ip + 1];
        const c0 = currRing[ip];
        const c1 = currRing[ip + 1];
        indices.push(p0, c0, p1, p1, c0, c1);
      }
      prevRing = currRing;
    }

    // Helper: Polar Organic Freshwater Disk (+Y Upward Winding: Tangent(+th) x Radial(+r) = +Y!)
    const appendOrganicPolarPool = (
      cx,
      cz,
      waterY,
      rings,
      angSegs,
      radiusEvalFn,
      flowBase,
      subMode,
      clampMinDepth = -1.0,
      defaultFlowDirX = 1.0,
      defaultFlowDirZ = 0.0
    ) => {
      let prevPoolRing = null;
      for (let ir = 0; ir <= rings; ir++) {
        const rNorm = ir / rings;
        const row = [];
        for (let ia = 0; ia <= angSegs; ia++) {
          const th = (ia / angSegs) * Math.PI * 2.0;
          const { rx, rz } = radiusEvalFn(th);
          const px = cx + Math.cos(th) * rx * rNorm;
          const pz = cz + Math.sin(th) * rz * rNorm;
          const rawSignedDepth = waterY - TerrainDEMLoader.sampleGroundHeight(px, pz);
          const signedDepth =
            clampMinDepth > 0.0 ? Math.max(clampMinDepth, rawSignedDepth) : rawSignedDepth;
          const spd = subMode > 4.0 ? 0.95 : subMode > 2.8 ? 0.52 * (1.0 - 0.35 * rNorm) : 0.24;
          row.push(
            pushCustomWaterVert(
              px,
              waterY,
              pz,
              0.0,
              1.0,
              0.0,
              defaultFlowDirX,
              defaultFlowDirZ,
              1.0,
              rNorm,
              flowBase + rNorm * 8.5,
              subMode,
              signedDepth,
              spd,
              0.0,
              0.0
            )
          );
        }
        if (prevPoolRing) {
          for (let ia = 0; ia < angSegs; ia++) {
            const p00 = prevPoolRing[ia];
            const p01 = prevPoolRing[ia + 1];
            const p10 = row[ia];
            const p11 = row[ia + 1];
            indices.push(p00, p01, p10, p01, p11, p10);
          }
        }
        prevPoolRing = row;
      }
    };

    // Zone 3A: Pulau "U" Artesian Forest Spring ("Mata Air Hutan Selatan") at (-286.0, +48.0), y = +11.20m
    appendOrganicPolarPool(
      TerrainDEMLoader.SPRING_U_CX,
      TerrainDEMLoader.SPRING_U_CZ,
      TerrainDEMLoader.SPRING_U_WATER_Y,
      15,
      56,
      (th) => {
        const r = TerrainDEMLoader.evalPulauUSpringRadiusAtAngle(th) * 1.07;
        return { rx: r, rz: r };
      },
      0.0,
      3.2,
      -1.0,
      0.96,
      -0.28
    );

    // Zone 3B: Pulau "U" 4-Bend Meandering Northbound River (-285.0 -> -93.5), strictly monotonic downhill!
    // Dense 224 x 22 ribbon extending across crossU in [-1.08, +1.08] so signedDepth zero-crossing
    // meets the shelving 3D gravel point-bars & mossy cut-banks organically at sub-centimeter accuracy!
    const riverSegs = 224;
    const riverCrossSegs = 22;
    const rx0 = TerrainDEMLoader.RIVER_U_X_START + 1.0;
    const rx1 = TerrainDEMLoader.RIVER_U_X_END + 1.5;
    let prevRiverRow = null;
    let cumRiverArc = 0.0;
    let prevRx = rx0;
    let prevRz = TerrainDEMLoader.evalPulauURiverCenterZ(rx0);

    for (let is = 0; is <= riverSegs; is++) {
      const sx = rx0 + (rx1 - rx0) * (is / riverSegs);
      const sz = TerrainDEMLoader.evalPulauURiverCenterZ(sx);
      if (is > 0) {
        cumRiverArc += Math.hypot(sx - prevRx, sz - prevRz);
      }
      prevRx = sx;
      prevRz = sz;

      const m = TerrainDEMLoader.evalPulauURiverSlopeZ(sx);
      const invT = 1.0 / Math.hypot(1.0, m);
      const tx = invT;
      const tz = m * invT;
      const cxVec = -tz;
      const czVec = tx;

      const wy = TerrainDEMLoader.evalPulauURiverWaterY(sx);
      const wyAhead = TerrainDEMLoader.evalPulauURiverWaterY(sx + 0.5);
      const wyBehind = TerrainDEMLoader.evalPulauURiverWaterY(sx - 0.5);
      const dropSlope = Math.max(0.0, wyBehind - wyAhead);
      const nLen = Math.hypot(dropSlope * tx, 1.0, dropSlope * tz);
      const rnx = (dropSlope * tx) / nLen;
      const rny = 1.0 / nLen;
      const rnz = (dropSlope * tz) / nLen;

      const baseHalfW = TerrainDEMLoader.evalPulauURiverHalfWidth(sx);
      // Continuity + slope velocity: narrower riffle chutes & steeper drops flow faster than wide calm pools!
      const widthAccel = Math.max(0.78, Math.min(1.36, 5.1 / Math.max(3.6, baseHalfW)));
      const meanFlowSpd = (0.72 + Math.min(1.15, dropSlope * 10.5)) * widthAccel;
      const curv = TerrainDEMLoader.evalPulauURiverCurvatureZ(sx);
      const bendBias = Math.max(-0.80, Math.min(0.80, Math.tanh(-curv * 34.0)));

      const row = [];
      for (let ic = 0; ic <= riverCrossSegs; ic++) {
        const crossU = -1.08 + (2.16 * ic) / riverCrossSegs;
        const sideSign = crossU >= 0.0 ? 1.0 : -1.0;
        const bankScallop =
          0.25 * Math.sin(sx * 0.43 + sideSign * 1.75) +
          0.16 * Math.cos(sx * 0.94 - sideSign * 0.85);
        const effHw = Math.max(3.40, baseHalfW + bankScallop);

        const vx = sx + cxVec * crossU * effHw;
        const vz = sz + czVec * crossU * effHw;
        const signedDepth = wy - TerrainDEMLoader.sampleGroundHeight(vx, vz);

        // Parabolic Poiseuille + meander-skewed thalweg velocity profile (fast in deep thalweg, slow near banks)
        const thalwegShift = crossU - 0.34 * bendBias;
        const poiseuille = Math.max(0.22, 1.22 - 0.72 * Math.min(1.3, thalwegShift * thalwegShift));
        const localFlowSpd = meanFlowSpd * poiseuille;
        const boulderWake = evalBoulderWakeAt(vx, vz, tx, tz);
        const bendSide = bendBias * Math.max(-1.2, Math.min(1.2, crossU));

        row.push(
          pushCustomWaterVert(
            vx,
            wy,
            vz,
            rnx,
            rny,
            rnz,
            tx,
            tz,
            1.0,
            Math.max(-1.0, Math.min(1.0, crossU)),
            cumRiverArc,
            meanFlowSpd,
            signedDepth,
            localFlowSpd,
            boulderWake,
            bendSide
          )
        );
      }
      if (prevRiverRow) {
        for (let ic = 0; ic < riverCrossSegs; ic++) {
          const a = prevRiverRow[ic];
          const b = prevRiverRow[ic + 1];
          const d = row[ic];
          const c = row[ic + 1];
          // (b - a = Cross) x (d - a = Forward) = +Y!
          indices.push(a, b, d, b, c, d);
        }
      }
      prevRiverRow = row;
    }

    // Zone 3C: Pulau "U" Small Scenic Lake near Gedung Rektorat UI ("Danau Kecil Rektorat", y = -0.22m)
    appendOrganicPolarPool(
      TerrainDEMLoader.LAKE_U_CX,
      TerrainDEMLoader.LAKE_U_CZ,
      TerrainDEMLoader.LAKE_U_WATER_Y,
      16,
      68,
      (th) => {
        const mod = TerrainDEMLoader.evalPulauULakeRadiusMod(th) * 1.055;
        return {
          rx: TerrainDEMLoader.LAKE_U_RX * mod,
          rz: TerrainDEMLoader.LAKE_U_RZ * mod
        };
      },
      cumRiverArc,
      0.20,
      -1.0,
      0.92,
      -0.39
    );

    // Zone 4A: Pulau "I" Mini-Niagara Upper Basalt River Shelf (rW: 29.8m -> 15.52m, y = +18.80m -> +18.66m brink)
    const upRSteps = 14;
    const upASteps = 48;
    const halfAng = TerrainDEMLoader.NIAGARA_HALF_ANGLE;
    let prevUpRow = null;
    for (let ir = 0; ir <= upRSteps; ir++) {
      const uR = ir / upRSteps;
      const rW = 29.8 + (15.52 - 29.8) * uR;
      const brinkDrawdown = 0.14 * Math.pow(PhysicsKernel.smootherstep(0.68, 1.0, uR), 1.6);
      const wy = TerrainDEMLoader.NIAGARA_UPPER_WATER_Y - brinkDrawdown;
      const flowArc = 30.0 - rW;
      const spdUp = 1.45 + 0.55 * uR;
      const row = [];
      for (let ia = 0; ia <= upASteps; ia++) {
        const alphaNorm = -1.0 + (2.0 * ia) / upASteps;
        const alpha = alphaNorm * (halfAng * 0.985);
        const sA = Math.sin(alpha);
        const cA = Math.cos(alpha);
        const px = TerrainDEMLoader.NIAGARA_CX + sA * rW;
        const pz = TerrainDEMLoader.NIAGARA_CZ + cA * rW;
        const depthUp = Math.max(0.22, wy - TerrainDEMLoader.sampleGroundHeight(px, pz));
        row.push(
          pushCustomWaterVert(
            px,
            wy,
            pz,
            0.0,
            1.0,
            0.0,
            -sA,
            -cA,
            1.0,
            alphaNorm,
            flowArc,
            spdUp,
            depthUp,
            spdUp,
            0.25 * uR,
            0.0
          )
        );
      }
      if (prevUpRow) {
        for (let ia = 0; ia < upASteps; ia++) {
          const a = prevUpRow[ia];
          const b = prevUpRow[ia + 1];
          const d = row[ia];
          const c = row[ia + 1];
          // (b - a = +Alpha/+X) x (d - a = -R/-Z) = +Y!
          indices.push(a, b, d, b, c, d);
        }
      }
      prevUpRow = row;
    }

    // Zone 4B: Pulau "I" Mini-Niagara 3D Parabolic Concave Horseshoe Falling Water Curtain (~36.8m wide x 17.0m tall!)
    const fallASteps = 72;
    const fallTSteps = 26;
    let prevFallRow = null;
    for (let it = 0; it <= fallTSteps; it++) {
      const tDrop = it / fallTSteps;
      const yCur =
        18.66 - 17.06 * (0.14 * tDrop + 0.86 * tDrop * tDrop);
      const dyDt = -17.06 * (0.14 + 1.72 * tDrop);
      const drDt = -3.12 * 0.74 * Math.pow(Math.max(0.015, tDrop), -0.26);
      const nInMag = Math.abs(dyDt);
      const nUpMag = Math.abs(drDt);
      const nInv = 1.0 / Math.hypot(nInMag, nUpMag);

      const row = [];
      for (let ia = 0; ia <= fallASteps; ia++) {
        const alphaNorm = -1.0 + (2.0 * ia) / fallASteps;
        const alpha = alphaNorm * (halfAng * 0.985);
        const lipFlute =
          (0.24 * Math.sin(alpha * 9.0) + 0.14 * Math.cos(alpha * 15.0)) * (1.0 - 0.65 * tDrop);
        const rCur = 15.52 - 3.12 * Math.pow(tDrop, 0.74) + lipFlute;
        const sA = Math.sin(alpha);
        const cA = Math.cos(alpha);
        const px = TerrainDEMLoader.NIAGARA_CX + sA * rCur;
        const pz = TerrainDEMLoader.NIAGARA_CZ + cA * rCur;
        const nxV = -sA * nInMag * nInv;
        const nyV = nUpMag * nInv;
        const nzV = -cA * nInMag * nInv;
        const arcMeters = (alpha + halfAng) * TerrainDEMLoader.NIAGARA_LIP_R;

        row.push(
          pushCustomWaterVert(
            px,
            yCur,
            pz,
            nxV,
            nyV,
            nzV,
            alphaNorm,
            tDrop,
            2.0,
            alphaNorm,
            arcMeters,
            tDrop
          )
        );
      }
      if (prevFallRow) {
        for (let ia = 0; ia < fallASteps; ia++) {
          const a = prevFallRow[ia];
          const b = prevFallRow[ia + 1];
          const d = row[ia];
          const c = row[ia + 1];
          indices.push(a, b, d, b, c, d);
        }
      }
      prevFallRow = row;
    }

    // Zone 4C: Pulau "I" Mini-Niagara Turquoise Horseshoe Plunge Pool (y = +1.80m) + Western Tailrace Outlet
    appendOrganicPolarPool(
      TerrainDEMLoader.NIAGARA_CX,
      TerrainDEMLoader.NIAGARA_CZ,
      TerrainDEMLoader.NIAGARA_POOL_WATER_Y,
      11,
      48,
      () => ({ rx: 14.95, rz: 14.95 }),
      0.0,
      4.5,
      0.24,
      0.0,
      -1.0
    );

    const outSegs = 26;
    const outCrossSegs = 8;
    let prevOutRow = null;
    for (let iz = 0; iz <= outSegs; iz++) {
      const uOut = iz / outSegs;
      const pz = 630.2 + (610.6 - 630.2) * uOut;
      const dCoastOut = TerrainDEMLoader.getIslandICoastDistance(TerrainDEMLoader.NIAGARA_CX, pz);
      const collarOut = PhysicsKernel.smootherstep(-1.2, -10.5, dCoastOut);
      const wy =
        (TerrainDEMLoader.NIAGARA_POOL_WATER_Y * (1.0 - uOut) + 0.16 * uOut) * collarOut;
      const row = [];
      for (let ic = 0; ic <= outCrossSegs; ic++) {
        const crossU = -1.0 + (2.0 * ic) / outCrossSegs;
        const px = TerrainDEMLoader.NIAGARA_CX + crossU * 3.80;
        const depthOut = Math.max(0.16, wy - TerrainDEMLoader.sampleGroundHeight(px, pz));
        row.push(
          pushCustomWaterVert(
            px,
            wy,
            pz,
            0.0,
            0.996,
            -0.088,
            0.0,
            -1.0,
            1.0,
            crossU,
            uOut * 19.6,
            1.65,
            depthOut,
            1.65 * (1.15 - 0.55 * crossU * crossU),
            0.18,
            0.0
          )
        );
      }
      if (prevOutRow) {
        for (let ic = 0; ic < outCrossSegs; ic++) {
          const a = prevOutRow[ic];
          const b = prevOutRow[ic + 1];
          const d = row[ic];
          const c = row[ic + 1];
          // (b - a = +X) x (d - a = -Z) = +Y!
          indices.push(a, b, d, b, c, d);
        }
      }
      prevOutRow = row;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    geo.setAttribute('aHydroMeta', new THREE.BufferAttribute(new Float32Array(hydroMeta), 4));
    geo.setAttribute('aRiverDetail', new THREE.BufferAttribute(new Float32Array(riverDetail), 4));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    return geo;
  }

  /**
   * Constructs the complete Kepulauan UI ("Pulau U" & "Pulau I") Terrain, Dual -18.0m Submerged Skirts,
   * Dual Closed 360° Perimeter Stone Turap Walls, Sculpted 3D Mossy River Boulders,
   * Freshwater Spring-River-Lake, Mini-Niagara Waterfall, & 3.3 km Surrounding Ocean Expanse.
   * Optimized to strictly 2 Draw Calls (1 Batched Archipelago Land+Skirts+Turap+RiverBoulders, 1 Unified Ocean+Freshwater+Cataract).
   */
  static buildCampusTerrain(waterMaterial = null, sharedSurfaceUniforms = null) {
    const uniforms = sharedSurfaceUniforms || createSharedSurfaceUniforms();
    const group = new THREE.Group();
    group.name = 'KampusUITerrainAndDanauKenanga';

    const terrainAndTurapMat = new THREE.MeshStandardMaterial({
      name: 'CampusTerrainAndTurapMat',
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.90,
      metalness: 0.02,
      envMapIntensity: 0.10
    });
    injectOmniScaleSurfaceShader(terrainAndTurapMat, {
      ySegmentStart: 120.0,
      segmentCount: 0.0,
      surfaceProfile: SURFACE_PROFILE.TERRAIN_AND_TURAP,
      bumpScale: 0.78,
      sharedUniforms: uniforms
    });

    // 1A. Pulau "U" High-Density DEM Patch + -18.0m Watertight Skirt (2.1x vertex concentration across
    //     the 4-Bend Meandering River valley so high-altitude aerial views never clip river bends!)
    const pulauUMesh = new THREE.Mesh(
      TerrainDEMLoader._buildIslandPatchWithSkirtGeometry(-436.0, 114.0, -122.0, 536.0, 348, 396, {
        focusX: -192.0,
        focusZ: 14.0,
        sigmaX: 0.24,
        sigmaZ: 0.15
      }),
      terrainAndTurapMat
    );
    pulauUMesh.name = 'PulauU_TerrainWithSkirt';
    pulauUMesh.receiveShadow = true;

    // 1B. Pulau "I" High-Density Volcano + Mini-Niagara + Runway DEM Patch + -18.0m Watertight Skirt
    const pulauIMesh = new THREE.Mesh(
      TerrainDEMLoader._buildIslandPatchWithSkirtGeometry(-436.0, 114.0, 536.0, 818.0, 336, 224, {
        focusX: -38.0,
        focusZ: 672.0,
        sigmaX: 0.15,
        sigmaZ: 0.25
      }),
      terrainAndTurapMat
    );
    pulauIMesh.name = 'PulauI_TerrainWithSkirt';
    pulauIMesh.receiveShadow = true;

    // 1C. Closed 360° Stone Turap Seawalls around BOTH Pulau "U" & Pulau "I"
    const turapMesh = new THREE.Mesh(
      TerrainDEMLoader._buildLakesideStoneTurapGeometry(),
      terrainAndTurapMat
    );
    turapMesh.name = 'KepulauanUI_Dual360StoneTurap';
    turapMesh.castShadow = true;
    turapMesh.receiveShadow = true;

    // 1D. Sculpted 3D Mossy River Boulders, Riffle Stones, & Point-Bar Cobbles (+0 Draw Calls after batch merge!)
    const riverBouldersMesh = new THREE.Mesh(
      TerrainDEMLoader._buildFreshwaterRiverBouldersGeometry(),
      terrainAndTurapMat
    );
    riverBouldersMesh.name = 'PulauU_MossyRiverBoulders';
    riverBouldersMesh.castShadow = true;
    riverBouldersMesh.receiveShadow = true;

    group.add(pulauUMesh, pulauIMesh, turapMesh, riverBouldersMesh);
    // Merges pulauUMesh + pulauIMesh + turapMesh + riverBouldersMesh into STRICTLY 1 SINGLE DRAW CALL!
    NexusBatchOptimizer.optimizeHierarchy(group);

    const activeWaterMat =
      waterMaterial ||
      new THREE.MeshStandardMaterial({
        name: 'DanauKenangaFallbackWaterMat',
        color: 0x195256,
        roughness: 0.12,
        metalness: 0.15,
        envMapIntensity: 0.14
      });

    const waterMesh = new THREE.Mesh(
      TerrainDEMLoader._buildDanauKenangaWaterGeometry(),
      activeWaterMat
    );
    waterMesh.name = 'DanauKenangaWaterSurface';
    waterMesh.receiveShadow = true;
    group.add(waterMesh);

    group.updateMatrixWorld(true);
    return {
      group,
      groundHeightFn: TerrainDEMLoader.sampleGroundHeight
    };
  }
}
