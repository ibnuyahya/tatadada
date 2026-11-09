import * as THREE from 'three';
import {
  SURFACE_PROFILE,
  createSharedSurfaceUniforms,
  injectOmniScaleSurfaceShader
} from '../core/ProceduralTextureFactory.js';
import { TerrainDEMLoader } from './TerrainDEMLoader.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.3 — 10x Ultra-High Level of Detail Edition)
 * ./world-gis/AutogenBiomeSpawner.js
 *
 * Spawns 10x High-Detail Kampus UI Depok Multi-Species Organic Vegetation, Courtyard Planter Shrubs, & Garden Lamps:
 * - Multi-branch fluted organic tree trunks (26x28 main trunk + 5 surface root spurs + 4 primary boughs +
 *   8 secondary forked sub-branches + -3.6m subterranean root skirt).
 * - High-order 6-harmonic botanical leaf-cluster foliage clouds (36x24 = 1,728 tris/cloud, 0 cones!)
 *   across 56 campus trees + 48 manicured lush green shrubs inside the 8 Gedung Rektorat UI base planters.
 * - Fluted 28-segment Nusantara pedestrian garden post-lamps with 4-strut 3D lantern cages & Tajug finials.
 *
 * Entire biome (~970,000 triangles) rendered in STRICTLY 3 INSTANCED DRAW CALLS!
 */
export class AutogenBiomeSpawner {
  static ROOT_SKIRT_DEPTH_Y = -3.6;

  static DEFAULT_TREE_SPAWNS = [
    // 6 Tall Layered Tropical Cemara / Pine Trees flanking the Flag Cloister (Mavic Photo #1!)
    { species: 'CEMARA', x: -23.8, z: 65.5, scale: 1.18, yaw: 0.35 },
    { species: 'CEMARA', x: -27.6, z: 59.5, scale: 1.08, yaw: 1.42 },
    { species: 'CEMARA', x: 18.8, z: 64.8, scale: 1.20, yaw: 2.15 },
    { species: 'CEMARA', x: 22.8, z: 58.8, scale: 1.10, yaw: 3.24 },
    { species: 'CEMARA', x: -24.5, z: 48.0, scale: 1.05, yaw: 4.12 },
    { species: 'CEMARA', x: 19.5, z: 48.5, scale: 1.06, yaw: 5.08 },

    // 18 Pohon Trembesi / Ki Hujan (Umbrella-Dome Giant Shade Trees — UI Urban Forest)
    { species: 'TREMBESI', x: -34.0, z: 52.0, scale: 1.16, yaw: 0.42 },
    { species: 'TREMBESI', x: -46.5, z: 48.0, scale: 1.20, yaw: 1.71 },
    { species: 'TREMBESI', x: -45.0, z: 68.0, scale: 1.14, yaw: 2.63 },
    { species: 'TREMBESI', x: 26.5, z: 52.0, scale: 1.15, yaw: 3.52 },
    { species: 'TREMBESI', x: 48.5, z: 48.0, scale: 1.18, yaw: 4.21 },
    { species: 'TREMBESI', x: 54.0, z: 72.0, scale: 1.22, yaw: 5.14 },
    { species: 'TREMBESI', x: 58.0, z: 98.0, scale: 1.16, yaw: 0.93 },
    { species: 'TREMBESI', x: 52.0, z: 122.0, scale: 1.14, yaw: 2.12 },
    { species: 'TREMBESI', x: -78.0, z: 22.0, scale: 1.22, yaw: 1.24 },
    { species: 'TREMBESI', x: -71.5, z: -1.5, scale: 1.24, yaw: 2.88 },
    { species: 'TREMBESI', x: -78.0, z: 58.0, scale: 1.18, yaw: 3.81 },
    { species: 'TREMBESI', x: -84.0, z: 86.0, scale: 1.16, yaw: 4.72 },
    { species: 'TREMBESI', x: -42.0, z: -52.0, scale: 1.20, yaw: 5.49 },
    { species: 'TREMBESI', x: -18.0, z: -64.0, scale: 1.22, yaw: 0.62 },
    { species: 'TREMBESI', x: 14.0, z: -64.0, scale: 1.18, yaw: 1.91 },
    { species: 'TREMBESI', x: 36.0, z: -58.0, scale: 1.15, yaw: 3.34 },
    { species: 'TREMBESI', x: -76.0, z: -48.0, scale: 1.20, yaw: 2.05 },
    { species: 'TREMBESI', x: -52.0, z: 126.0, scale: 1.16, yaw: 4.15 },

    // 20 Pohon Mahoni (Dense Billowing Tropical Hardwood Trees)
    { species: 'MAHONI', x: -31.5, z: 64.0, scale: 1.08, yaw: 0.81 },
    { species: 'MAHONI', x: -50.0, z: 58.0, scale: 1.12, yaw: 2.24 },
    { species: 'MAHONI', x: -42.0, z: 82.0, scale: 1.05, yaw: 3.41 },
    { species: 'MAHONI', x: 27.5, z: 65.0, scale: 1.10, yaw: 4.62 },
    { species: 'MAHONI', x: 47.0, z: 62.0, scale: 1.08, yaw: 5.83 },
    { species: 'MAHONI', x: 50.0, z: 86.0, scale: 1.14, yaw: 1.14 },
    { species: 'MAHONI', x: 66.0, z: 84.0, scale: 1.10, yaw: 2.73 },
    { species: 'MAHONI', x: 64.0, z: 112.0, scale: 1.12, yaw: 4.02 },
    { species: 'MAHONI', x: 46.5, z: 22.0, scale: 1.05, yaw: 5.19 },
    { species: 'MAHONI', x: 48.0, z: 34.0, scale: 1.04, yaw: 0.34 },
    { species: 'MAHONI', x: -76.0, z: 40.0, scale: 1.11, yaw: 1.52 },
    { species: 'MAHONI', x: -75.0, z: 6.5, scale: 1.12, yaw: 2.84 },
    { species: 'MAHONI', x: -72.0, z: -43.5, scale: 1.12, yaw: 4.11 },
    { species: 'MAHONI', x: -96.0, z: 18.0, scale: 1.16, yaw: 5.32 },
    { species: 'MAHONI', x: -99.2, z: -22.0, scale: 1.16, yaw: 0.51 },
    { species: 'MAHONI', x: -94.0, z: 64.0, scale: 1.14, yaw: 1.84 },
    { species: 'MAHONI', x: -60.0, z: -56.0, scale: 1.10, yaw: 3.12 },
    { species: 'MAHONI', x: -28.0, z: -58.0, scale: 1.08, yaw: 4.41 },
    { species: 'MAHONI', x: -4.0, z: -62.0, scale: 1.12, yaw: 5.71 },
    { species: 'MAHONI', x: 26.0, z: -64.0, scale: 1.09, yaw: 0.92 },

    // 12 Palem Raja (Curving Royal Palms lining Lapangan Rotunda UI outer perimeter & Lakeside)
    { species: 'PALEM_RAJA', x: -39.5, z: 93.0, scale: 1.08, yaw: 1.65 },
    { species: 'PALEM_RAJA', x: -37.9, z: 123.8, scale: 1.10, yaw: 0.95 },
    { species: 'PALEM_RAJA', x: 33.2, z: 90.2, scale: 1.08, yaw: 4.55 },
    { species: 'PALEM_RAJA', x: 36.75, z: 106.5, scale: 1.10, yaw: 4.82 },
    { species: 'PALEM_RAJA', x: 32.6, z: 125.6, scale: 1.07, yaw: 5.25 },
    { species: 'PALEM_RAJA', x: 51.0, z: 18.5, scale: 1.08, yaw: 1.52 },
    { species: 'PALEM_RAJA', x: 54.5, z: 29.0, scale: 1.10, yaw: 1.62 },
    { species: 'PALEM_RAJA', x: 58.5, z: 40.5, scale: 1.09, yaw: 1.58 },
    { species: 'PALEM_RAJA', x: 49.5, z: -32.0, scale: 1.06, yaw: 1.48 },
    { species: 'PALEM_RAJA', x: 47.0, z: -46.0, scale: 1.08, yaw: 1.42 },
    { species: 'PALEM_RAJA', x: -25.8, z: 73.6, scale: 1.06, yaw: 2.35 },
    { species: 'PALEM_RAJA', x: 18.2, z: 72.5, scale: 1.06, yaw: 3.92 }
  ];

  /**
   * Single Source of Truth (SSoST) 64 Fluted Nusantara Campus Streetlamp Coordinates
   * across Pulau U & Pulau I (shared by InstancedMesh placement & Solid-Body Physics Colliders).
   */
  static getLampCoords() {
    const lampCoords = [];
    // Group A (12): Lapangan Rotunda UI Ring (R = 34.9m around (-2.5, 106.5))
    for (let i = 0; i < 12; i++) {
      const th = (i / 12) * Math.PI * 2.0 + Math.PI / 12;
      lampCoords.push({ x: -2.5 + Math.cos(th) * 34.9, z: 106.5 + Math.sin(th) * 34.9 });
    }
    // Group B (4): South Rektorat Approach Plaza
    lampCoords.push(
      { x: -11.5, z: 68.5 },
      { x: 6.5, z: 68.5 },
      { x: -21.2, z: 51.5 },
      { x: 16.2, z: 51.5 }
    );
    // Group E (8): East Lakeside Promenade (uniformly spaced z = -52 .. +48)
    for (let k = 0; k < 8; k++) {
      const lz = -52.0 + k * (100.0 / 7.0);
      lampCoords.push({ x: TerrainDEMLoader.getShorelinePromenadeX(lz) - 0.55, z: lz });
    }
    // Group C (6): West Helipad Boulevard & Helipad Entry Plaza
    lampCoords.push(
      { x: -50.0, z: 100.8 },
      { x: -50.0, z: 112.2 },
      { x: -72.0, z: 100.8 },
      { x: -72.0, z: 112.2 },
      { x: -83.2, z: 99.6 },
      { x: -83.2, z: 113.4 }
    );
    // Group D (6): West Courtyard / Electric Golf-Cart Shuttle Boulevard (x = -55.4, z = -22..88)
    for (let k = 0; k < 6; k++) {
      lampCoords.push({ x: -55.4, z: -22.0 + k * 22.0 });
    }
    // Group F (4): Pulau I West Aircraft Apron Perimeter (z = 658.4, x = -276.5..-231.5)
    for (let k = 0; k < 4; k++) {
      lampCoords.push({ x: -276.5 + k * 15.0, z: 658.4 });
    }
    // Group G (4): Jembatan Teksas UI West Pilon Lingga Forecourt & East Pilon Yoni FIB Overlook Plaza
    lampCoords.push(
      { x: TerrainDEMLoader.TEKSAS_BRIDGE_X + 5.8, z: 148.5 },
      { x: TerrainDEMLoader.TEKSAS_BRIDGE_X - 5.8, z: 148.5 },
      { x: TerrainDEMLoader.TEKSAS_BRIDGE_X + 5.8, z: 343.5 },
      { x: TerrainDEMLoader.TEKSAS_BRIDGE_X - 5.8, z: 343.5 }
    );
    // Group H1 (6): Pulau U West-Arm Arterial Boulevard (-Z coastal shoulder, 4.9m outside road centerline)
    for (let k = 0; k < 6; k++) {
      lampCoords.push({ x: -96.0 - k * 46.0, z: 71.3 + 2.36 * k + 0.24 * k * k });
    }
    // Group H2 (4): Pulau U Southern Arterial Boulevard (+X inner foothill shoulder, >= 5.0m outside road centerline)
    for (let k = 0; k < 4; k++) {
      lampCoords.push({ x: -355.2 - 9.6 * k + 3.0 * k * k, z: 128.0 + k * 40.0 });
    }
    // Group H3 (4): Underpass Bawah Laut Selatan U-I West & East Portal Gateways
    lampCoords.push(
      { x: -348.1, z: 324.5 },
      { x: -363.9, z: 324.5 },
      { x: -348.1, z: 625.5 },
      { x: -363.9, z: 625.5 }
    );
    // Group H4 (6): Dermaga Penyeberangan Utara Pulau U & Pulau I + Right-Arm Saddle Shoulder & North Headland Overlook
    lampCoords.push(
      { x: -41.2, z: 462.5 },
      { x: -54.8, z: 462.5 },
      { x: -91.2, z: 611.0 },
      { x: -104.8, z: 611.0 },
      { x: -164.5, z: 438.2 },
      { x: 37.2, z: 396.0 }
    );
    return lampCoords;
  }

  static isAllowedTreeSpawn(x, z, groundHeightFn = TerrainDEMLoader.sampleGroundHeight) {
    if (x > -55.8 && x < 55.2 && z > -37.2 && z < 41.2) return false;
    if (x > -21.8 && x < 16.8 && z >= 38.0 && z <= 73.5) return false;
    if (x > -6.0 && x < 51.0 && z > -54.5 && z <= -34.0) return false;
    if (Math.hypot(x - 29.5, z - 29.5) < 10.6) return false;
    // Outdoor Campus Monopole LED Videotron Plinth & Sightline Clearance Zone (28.4, 78.2)
    if (Math.hypot(x - 28.4, z - 78.2) < 9.5) return false;
    if (z >= 33.0 && z <= 130.0) {
      const walkwayX = 34.2 + ((z - 35.5) / 92.5) * 9.6;
      if (Math.abs(x - walkwayX) < 4.6) return false;
    }
    if (x > -69.5 && x < -55.5 && z > -36.5 && z < 104.0) return false;
    if (Math.hypot(x + 2.5, z - 106.5) < 35.8) return false;
    if (x > -89.0 && x < -32.0 && z > 99.5 && z < 113.5) return false;

    // Obstacle-free ICAO FATO safety zone around VIP Waterfront Helipad & Transit Pavilion
    if (Math.hypot(x - TerrainDEMLoader.HELIPAD_CX, z - TerrainDEMLoader.HELIPAD_CZ) < 23.5) {
      return false;
    }
    if (Math.hypot(x + 84.8, z - 118.2) < 6.5 || Math.hypot(x + 87.2, z - 92.8) < 4.5) {
      return false;
    }

    // Jembatan Teksas UI West Connector, Forecourt, Span, & East FIB Overlook Plaza sightline corridor
    if (Math.abs(x - TerrainDEMLoader.TEKSAS_BRIDGE_X) < 13.5 && z > 129.0 && z < 354.0) {
      return false;
    }

    // Teluk U EPDM Jogging Track Oval + Sudut Calisthenics (-62.0, 136.5), Campus Kiosks, & Muster Pads
    if (Math.hypot((x + 62.0) / 19.5, (z - 136.5) / 10.5) < 1.0) return false;
    if (
      Math.hypot(x + 39.5, z - 134.5) < 6.0 ||
      Math.hypot(x + 16.5, z - 345.0) < 6.0 ||
      Math.hypot(x + 213.0, z - 326.0) < 6.0 ||
      Math.hypot(x + 116.5, z - 619.5) < 6.0 ||
      Math.hypot(x + 31.0, z - 76.5) < 4.5 ||
      Math.hypot(x + 296.5, z - 326.5) < 4.5 ||
      Math.hypot(x + 68.0, z - 444.0) < 4.5 ||
      Math.hypot(x + 222.5, z - 665.5) < 4.5
    ) {
      return false;
    }

    // Integrated Pulau U / Pulau I Road Network, Underpass Open-Cut Ramps/Portals, & Northern Ferry Terminals
    if (TerrainDEMLoader.isTransitInfrastructureExclusion(x, z, 5.2)) {
      return false;
    }

    // True 360-degree closed island shoreline safety margin (keeps all trunks >= 3.5m inside stone turap!)
    if (TerrainDEMLoader.getIslandCoastDistance(x, z) > -3.5) return false;

    // Freshwater hydrology exclusion: keeps all trunks cleanly on dry banks outside Mata Air,
    // Meandering River, Danau Kecil Rektorat, and Pulau I Mini-Niagara Horseshoe Gorge!
    if (TerrainDEMLoader.isFreshwaterExclusion(x, z, 2.2)) return false;

    const gy = groundHeightFn(x, z);
    if (gy < -0.15) return false;

    // Pulau I Aviation, Volcanic Cone, & Mini-Niagara Sightline Rules:
    if (z >= 536.0) {
      // 1. ICAO Runway RWY 36/18, Blast Pad, Apron, & Taxiway obstacle-free aviation zone
      if (x < -122.5) return false;
      // 2. Upper Gunung Api volcanic ash/magma crater cone stays barren basalt & tephra
      if (Math.hypot(x - TerrainDEMLoader.VOLCANO_CX, z - TerrainDEMLoader.VOLCANO_CZ) < 32.5 || gy > 34.8) {
        return false;
      }
      // 3. Open front western amphitheater sightline to Mini-Niagara Horseshoe Waterfall
      if (z < 646.0 && Math.abs(x - TerrainDEMLoader.NIAGARA_CX) < 16.0) {
        return false;
      }
    }

    return true;
  }

  /**
   * Deterministic procedural generator for:
   * 1. Pulau U Dense Southern Highland Rainforest ("Hutan Lebat Dataran Tinggi Selatan") cradling
   *    Mata Air Hutan Selatan + Riparian Gallery along the 4-bend Meandering River to Danau Kecil Rektorat
   *    + Lush Forest-Floor & Spring-Rim Understory Shrubs/Ferns (0 extra Draw Calls!).
   * 2. Pulau I Sparse Volcanic Montane Forest ("Hutan Gunung Tidak Begitu Lebat") framing
   *    Mini-Niagara Horseshoe Waterfall & scattered across Gunung Api mid/lower slopes.
   */
  static _generateArchipelagoForestSpawns(groundHeightFn, baseTrees) {
    const fract = (v) => v - Math.floor(v);
    const hash01 = (seed) => fract(Math.sin(seed * 127.1 + 311.7) * 43758.5453123);

    const placed = baseTrees.map((t) => ({ x: t.x, z: t.z }));
    const extraTrees = [];
    const forestShrubs = [];

    const hasMinSpacing = (x, z, minDist) => {
      const minSq = minDist * minDist;
      for (let i = 0; i < placed.length; i++) {
        const dx = x - placed[i].x;
        const dz = z - placed[i].z;
        if (dx * dx + dz * dz < minSq) return false;
      }
      return true;
    };

    const tryAddTree = (species, x, z, scale, yaw, minDist) => {
      if (!AutogenBiomeSpawner.isAllowedTreeSpawn(x, z, groundHeightFn)) return false;
      if (!hasMinSpacing(x, z, minDist)) return false;
      const rx = Number(x.toFixed(2));
      const rz = Number(z.toFixed(2));
      placed.push({ x: rx, z: rz });
      extraTrees.push({
        species,
        x: rx,
        z: rz,
        scale: Number(scale.toFixed(3)),
        yaw: Number(yaw.toFixed(3))
      });
      return true;
    };

    // --- A1. MATA AIR HUTAN SELATAN CATHEDRAL CANOPY SANCTUARY (Pulau U) ---
    const spCx = TerrainDEMLoader.SPRING_U_CX;
    const spCz = TerrainDEMLoader.SPRING_U_CZ;
    for (let ring = 0; ring < 2; ring++) {
      const count = ring === 0 ? 10 : 12;
      const baseR = ring === 0 ? 15.2 : 22.8;
      for (let k = 0; k < count; k++) {
        // Span azimuths while leaving narrow northern river outlet (|th| < 0.34 around +X) open
        const u = (k + 0.5) / count;
        const th = 0.36 + u * (Math.PI * 2.0 - 0.72);
        const r = baseR + (hash01(10 + ring * 31 + k) - 0.5) * 2.6;
        const tx = spCx + Math.cos(th) * r;
        const tz = spCz + Math.sin(th) * r;
        const h1 = hash01(50 + ring * 19 + k);
        const species = h1 < 0.52 ? 'TREMBESI' : h1 < 0.85 ? 'MAHONI' : 'CEMARA';
        const scale = (species === 'TREMBESI' ? 1.24 : 1.16) + hash01(90 + ring * 23 + k) * 0.16;
        const yaw = hash01(130 + ring * 29 + k) * Math.PI * 2.0;
        tryAddTree(species, tx, tz, scale, yaw, 9.2);
      }
    }

    // --- A2. RIPARIAN GALLERY ALONG 4-BEND MEANDERING NORTHBOUND RIVER & DANAU KECIL REKTORAT ---
    let stepIdx = 0;
    for (let rx = -268.0; rx <= -102.0; rx += 7.2) {
      const cz = TerrainDEMLoader.evalPulauURiverCenterZ(rx);
      const hw = TerrainDEMLoader.evalPulauURiverHalfWidth(rx);
      const primarySide = stepIdx % 2 === 0 ? 1.0 : -1.0;
      const sides = stepIdx % 3 === 0 ? [primarySide, -primarySide] : [primarySide];
      for (let sIdx = 0; sIdx < sides.length; sIdx++) {
        const side = sides[sIdx];
        const seed = 210 + stepIdx * 7 + sIdx * 3;
        const bankOffset = hw + 5.1 + hash01(seed) * 3.4;
        const tx = rx + (hash01(seed + 1) - 0.5) * 2.6;
        const tz = cz + side * bankOffset;
        const hs = hash01(seed + 2);
        const species = hs < 0.44 ? 'TREMBESI' : hs < 0.82 ? 'MAHONI' : 'CEMARA';
        const scale = 1.06 + hash01(seed + 3) * 0.22;
        const yaw = hash01(seed + 4) * Math.PI * 2.0;
        tryAddTree(species, tx, tz, scale, yaw, 9.8);
      }
      stepIdx++;
    }

    // Scenic Lakeside Framing around Danau Kecil Dekat Rektorat (-82.5, -22.0)
    const lakeTrees = [
      { species: 'TREMBESI', x: -99.5, z: -33.5, scale: 1.18, yaw: 0.74 },
      { species: 'MAHONI', x: -98.8, z: -8.5, scale: 1.12, yaw: 2.18 },
      { species: 'PALEM_RAJA', x: -91.5, z: -46.2, scale: 1.10, yaw: 1.45 },
      { species: 'PALEM_RAJA', x: -83.5, z: 4.2, scale: 1.08, yaw: 3.62 },
      { species: 'MAHONI', x: -73.2, z: 3.4, scale: 1.06, yaw: 4.85 },
      { species: 'TREMBESI', x: -103.5, z: -21.0, scale: 1.22, yaw: 1.92 },
      { species: 'CEMARA', x: -93.0, z: 2.8, scale: 1.12, yaw: 5.24 }
    ];
    for (let i = 0; i < lakeTrees.length; i++) {
      const lt = lakeTrees[i];
      tryAddTree(lt.species, lt.x, lt.z, lt.scale, lt.yaw, 8.2);
    }

    // --- A3. DENSE SOUTHERN HIGHLAND TROPICAL RAINFOREST (Pulau U Southern Hills) ---
    // Low-discrepancy R2 quasi-random lattice over x in [-382, -146], z in [-38, +318] where gy >= 2.2m
    let pulauUExtraCount = extraTrees.length;
    for (let i = 0; i < 680 && pulauUExtraCount < 128; i++) {
      const u = fract(0.5 + i * 0.7548776662466927);
      const v = fract(0.5 + i * 0.5698402909980532);
      const tx = -382.0 + u * 236.0;
      const tz = -38.0 + v * 356.0;
      const gy = groundHeightFn(tx, tz);
      // Concentrate closed-canopy rainforest on the elevated Southern Highland hills & ridges
      if (gy < 2.2) continue;

      const hs = hash01(600 + i * 5);
      const species = hs < 0.46 ? 'TREMBESI' : hs < 0.85 ? 'MAHONI' : 'CEMARA';
      const scale =
        (species === 'TREMBESI' ? 1.08 : species === 'MAHONI' ? 1.04 : 1.06) +
        hash01(601 + i * 5) * 0.25;
      const yaw = hash01(602 + i * 5) * Math.PI * 2.0;
      if (tryAddTree(species, tx, tz, scale, yaw, 10.6)) {
        pulauUExtraCount++;
      }
    }

    // --- A3b. KOMPLEKS DPPD – ILRC – STP-UI GREEN CAMPUS PERIMETER TREE BELT (Pulau U Right-Arm Terrace) ---
    // Frames the 2/6/8-story campus cluster with authentic UI shade trees while keeping the Teluk U -> Rektorat vista open!
    const dppdCx = TerrainDEMLoader.DPPD_CX;
    const dppdCz = TerrainDEMLoader.DPPD_CZ;
    const toRekTh = Math.atan2(0.0 - dppdCz, 0.0 - dppdCx);
    for (let ring = 0; ring < 2; ring++) {
      const count = ring === 0 ? 18 : 22;
      const baseR = ring === 0 ? 46.8 : 57.5;
      for (let k = 0; k < count; k++) {
        const th = -Math.PI + ((k + 0.5) / count) * Math.PI * 2.0;
        // Leave clear 62-degree waterfront Belvedere vista straight toward Gedung Rektorat UI across Teluk U
        const dAng = Math.abs(Math.atan2(Math.sin(th - toRekTh), Math.cos(th - toRekTh)));
        if (dAng < 0.54) continue;
        const r = baseR + (hash01(1900 + ring * 29 + k) - 0.5) * 3.4;
        const tx = dppdCx + Math.cos(th) * r;
        const tz = dppdCz + Math.sin(th) * r;
        const hSp = hash01(1950 + ring * 31 + k);
        const species = hSp < 0.56 ? 'TREMBESI' : 'MAHONI';
        const scale = 1.08 + hash01(2010 + ring * 37 + k) * 0.18;
        const yaw = hash01(2070 + ring * 41 + k) * Math.PI * 2.0;
        tryAddTree(species, tx, tz, scale, yaw, 9.6);
      }
    }

    // --- A4. TROPICAL FOREST-FLOOR & MATA AIR UNDERSTORY SHRUBS/FERNS (0 Extra Draw Calls!) ---
    for (let k = 0; k < 18; k++) {
      const u = (k + 0.5) / 18;
      const th = 0.30 + u * (Math.PI * 2.0 - 0.60);
      const r = 11.8 + hash01(1300 + k) * 3.8;
      const sx = spCx + Math.cos(th) * r;
      const sz = spCz + Math.sin(th) * r;
      if (
        !TerrainDEMLoader.isFreshwaterExclusion(sx, sz, 1.05) &&
        TerrainDEMLoader.getIslandCoastDistance(sx, sz) < -3.5
      ) {
        forestShrubs.push({
          x: sx,
          z: sz,
          yaw: hash01(1350 + k) * Math.PI * 2.0,
          sx: 1.55 + hash01(1400 + k) * 0.65,
          sy: 0.72 + hash01(1450 + k) * 0.30,
          sz: 1.55 + hash01(1500 + k) * 0.65
        });
      }
    }
    for (let i = 0; i < 160 && forestShrubs.length < 48; i++) {
      const u = fract(0.25 + i * 0.7548776662466927);
      const v = fract(0.75 + i * 0.5698402909980532);
      const sx = -362.0 + u * 212.0;
      const sz = -24.0 + v * 310.0;
      if (
        groundHeightFn(sx, sz) >= 2.6 &&
        !TerrainDEMLoader.isFreshwaterExclusion(sx, sz, 1.25) &&
        !TerrainDEMLoader.isTransitInfrastructureExclusion(sx, sz, 3.8) &&
        TerrainDEMLoader.getIslandCoastDistance(sx, sz) < -4.5
      ) {
        forestShrubs.push({
          x: sx,
          z: sz,
          yaw: hash01(1600 + i) * Math.PI * 2.0,
          sx: 1.50 + hash01(1650 + i) * 0.75,
          sy: 0.68 + hash01(1700 + i) * 0.32,
          sz: 1.50 + hash01(1750 + i) * 0.75
        });
      }
    }
    for (let k = 0; k < 8; k++) {
      const th = 0.75 + (k / 7) * 3.85;
      const sx = TerrainDEMLoader.LAKE_U_CX - Math.cos(th) * 14.8;
      const sz = TerrainDEMLoader.LAKE_U_CZ + Math.sin(th) * 24.4;
      if (
        sx < -70.0 &&
        !TerrainDEMLoader.isFreshwaterExclusion(sx, sz, 1.15) &&
        TerrainDEMLoader.getIslandCoastDistance(sx, sz) < -3.8
      ) {
        forestShrubs.push({
          x: sx,
          z: sz,
          yaw: hash01(1850 + k) * Math.PI * 2.0,
          sx: 1.38 + hash01(1870 + k) * 0.48,
          sy: 0.64 + hash01(1890 + k) * 0.25,
          sz: 1.38 + hash01(1910 + k) * 0.48
        });
      }
    }

    // --- B. PULAU I SPARSE VOLCANIC MONTANE FOREST ("Hutan Gunung Tidak Begitu Lebat") ---
    // B1. Mini-Niagara Horseshoe Canyon Rim Framing (North & South basalt cliffs flanking waterfall)
    for (const side of [-1.0, 1.0]) {
      for (let k = 0; k < 5; k++) {
        const alpha = side * (1.36 + k * 0.16);
        const rW = 19.5 + (k % 3) * 4.2 + hash01(2000 + side * 11 + k) * 1.6;
        const tx = TerrainDEMLoader.NIAGARA_CX + Math.sin(alpha) * rW;
        const tz = TerrainDEMLoader.NIAGARA_CZ + Math.cos(alpha) * rW;
        const hs = hash01(2050 + side * 13 + k);
        const species = hs < 0.60 ? 'CEMARA' : hs < 0.86 ? 'MAHONI' : 'TREMBESI';
        const scale = 0.96 + hash01(2100 + side * 17 + k) * 0.20;
        const yaw = hash01(2150 + side * 19 + k) * Math.PI * 2.0;
        tryAddTree(species, tx, tz, scale, yaw, 11.8);
      }
    }

    // B2. Scattered Montane Cloud-Forest Slopes of Gunung Gede Pulau I (Open >= 14.8m montane spacing climbing to y <= 41m!)
    let pulauICount = extraTrees.filter((t) => t.z >= 536.0).length;
    for (let i = 0; i < 460 && pulauICount < 42; i++) {
      const u = fract(0.38 + i * 0.7548776662466927);
      const v = fract(0.62 + i * 0.5698402909980532);
      const tx = -121.0 + u * 198.0; // [-121m .. +77m] northern volcanic half of Pulau I
      const tz = 612.0 + v * 164.0;  // [612m .. 776m] across Gunung Gede Pulau I flanks
      const gy = groundHeightFn(tx, tz);
      if (gy < 1.4 || gy > 41.0) continue;

      const hs = hash01(2300 + i * 3);
      // Higher elevations favor montane Cemara Gunung pines; lower foothills mix Mahoni & Trembesi
      const species = gy > 22.0 ? (hs < 0.80 ? 'CEMARA' : 'MAHONI') : hs < 0.54 ? 'CEMARA' : hs < 0.84 ? 'MAHONI' : 'TREMBESI';
      const scale = (gy > 28.0 ? 0.84 : 0.94) + hash01(2301 + i * 3) * 0.22;
      const yaw = hash01(2302 + i * 3) * Math.PI * 2.0;
      if (tryAddTree(species, tx, tz, scale, yaw, 14.8)) {
        pulauICount++;
      }
    }

    return { extraTrees, forestShrubs };
  }

  /**
   * Builds a 10x High-Detail organic lofted branching tree trunk geometry (~4,090 tris/trunk):
   * - 26x28 7-flute tropical buttress main trunk + -3.60m subterranean root skirt
   * - 5 exposed surface buttress root spurs
   * - 4 diverging primary boughs + 8 secondary forked sub-branches!
   */
  static _buildOrganicBranchingTrunkGeometry() {
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

    const addLoftedLimb = (radialSegs, axialSegs, sampleSectionFn, capTop = true) => {
      const rings = [];
      for (let ia = 0; ia <= axialSegs; ia++) {
        const t = ia / axialSegs;
        const sec = sampleSectionFn(t);
        const secNext = sampleSectionFn(Math.min(1.0, t + 0.02));
        let tx = secNext.cx - sec.cx;
        let ty = secNext.cy - sec.cy;
        let tz = secNext.cz - sec.cz;
        if (tx * tx + ty * ty + tz * tz < 1e-8) {
          ty = 1.0;
        }
        const tangent = new THREE.Vector3(tx, ty, tz).normalize();

        const ref = Math.abs(tangent.x) < 0.8 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1);
        const binorm = new THREE.Vector3().crossVectors(tangent, ref).normalize();
        const normAxis = new THREE.Vector3().crossVectors(binorm, tangent).normalize();

        const ring = [];
        const fluteFreq = sec.fluteCount || 7.0;
        for (let ir = 0; ir <= radialSegs; ir++) {
          const u = ir / radialSegs;
          const th = u * Math.PI * 2.0;
          const c = Math.cos(th);
          const s = Math.sin(th);

          // Multi-harmonic tropical buttress & fissured bark contour
          const flute =
            1.0 +
            sec.fluteAmp * Math.cos(fluteFreq * th + sec.twist) +
            sec.fluteAmp * 0.28 * Math.sin((fluteFreq * 2.0 + 1.0) * th - sec.twist * 1.4);
          const rad = sec.radius * flute;

          const dirX = binorm.x * c + normAxis.x * s;
          const dirY = binorm.y * c + normAxis.y * s;
          const dirZ = binorm.z * c + normAxis.z * s;

          const px = sec.cx + dirX * rad;
          const py = sec.cy + dirY * rad;
          const pz = sec.cz + dirZ * rad;

          const nOut = new THREE.Vector3(dirX, dirY + 0.12 * sec.taperSlope, dirZ).normalize();
          const barkGrain =
            0.91 + 0.06 * Math.sin(11.0 * th + t * 12.0) + 0.03 * Math.cos(19.0 * th - t * 7.0);

          ring.push(
            pushVert(
              px,
              py,
              pz,
              nOut.x,
              nOut.y,
              nOut.z,
              u * 2.0,
              t * 3.0,
              sec.colR * barkGrain,
              sec.colG * barkGrain,
              sec.colB * barkGrain
            )
          );
        }
        rings.push(ring);
      }

      for (let ia = 0; ia < axialSegs; ia++) {
        for (let ir = 0; ir < radialSegs; ir++) {
          const a = rings[ia][ir];
          const b = rings[ia][ir + 1];
          const c = rings[ia + 1][ir + 1];
          const d = rings[ia + 1][ir];
          indices.push(a, d, c, a, c, b);
        }
      }

      if (capTop) {
        const topSec = sampleSectionFn(1.0);
        const topCenter = pushVert(
          topSec.cx,
          topSec.cy + 0.03,
          topSec.cz,
          0.0,
          1.0,
          0.0,
          0.5,
          1.0,
          topSec.colR,
          topSec.colG,
          topSec.colB
        );
        const lastRing = rings[axialSegs];
        for (let ir = 0; ir < radialSegs; ir++) {
          indices.push(topCenter, lastRing[ir + 1], lastRing[ir]);
        }
      }
    };

    // 1. Main Organic Fluted Trunk + -3.60m Subterranean Root Skirt (18 radial x 18 axial = 666 tris)
    addLoftedLimb(18, 18, (t) => {
      const y = AutogenBiomeSpawner.ROOT_SKIRT_DEPTH_Y + t * (4.15 - AutogenBiomeSpawner.ROOT_SKIRT_DEPTH_Y);
      const isRoot = y < 0.55;
      const rootFlare = y < 0.7 ? Math.pow(Math.max(0.0, (0.7 - y) / 4.3), 1.35) * 0.44 : 0.0;
      const trunkTaper = y >= 0.0 ? Math.max(0.23, 0.46 - (y / 4.15) * 0.2) : 0.46;
      const radius = trunkTaper + rootFlare;

      const aboveT = Math.max(0.0, y / 4.15);
      const cx = Math.sin(aboveT * Math.PI * 0.9) * 0.16;
      const cz = Math.cos(aboveT * Math.PI * 0.7) * 0.11 - 0.11;

      const mossMix = Math.max(0.0, Math.min(1.0, (y + 1.2) / 3.2));
      return {
        cx,
        cy: y,
        cz,
        radius,
        fluteCount: 7.0,
        fluteAmp: isRoot ? 0.19 : 0.075,
        twist: t * 0.9,
        taperSlope: 0.15,
        colR: 0.105 + 0.085 * mossMix,
        colG: 0.084 + 0.068 * mossMix,
        colB: 0.058 + 0.046 * mossMix
      };
    });

    // 2. Five Sculpted Surface Buttress Root Spurs anchoring trunk into soil (5 x 6 x 5)
    for (let r = 0; r < 5; r++) {
      const az = (r / 5) * Math.PI * 2.0 + 0.22;
      const dx = Math.cos(az);
      const dz = Math.sin(az);
      addLoftedLimb(
        6,
        5,
        (t) => {
          // Lofts upward from buried root tip (-1.85m, out 0.95m) into trunk collar (+0.48m, out 0.28m)
          const invT = 1.0 - t;
          const outDist = 0.28 + Math.pow(invT, 1.25) * 0.68;
          const cy = -1.85 + t * 2.33;
          return {
            cx: dx * outDist,
            cy,
            cz: dz * outDist,
            radius: 0.10 + 0.08 * Math.sin(t * Math.PI),
            fluteCount: 3.0,
            fluteAmp: 0.12,
            twist: r * 0.7,
            taperSlope: 0.2,
            colR: 0.115 + 0.045 * t,
            colG: 0.092 + 0.036 * t,
            colB: 0.064 + 0.025 * t
          };
        },
        false
      );
    }

    // 3. Four Diverging Primary Boughs + Eight Secondary Forked Sub-Branches
    for (let b = 0; b < 4; b++) {
      const az = (b / 4) * Math.PI * 2.0 + 0.35;
      const dirX = Math.cos(az);
      const dirZ = Math.sin(az);
      const spread = 1.38 + (b % 2) * 0.18;
      const topY = 5.58 + (b % 3) * 0.12;

      const samplePrimary = (t) => {
        const curveOut = Math.pow(t, 1.28) * spread;
        return {
          cx: 0.08 + dirX * curveOut,
          cy: 3.28 + t * (topY - 3.28),
          cz: 0.04 + dirZ * curveOut
        };
      };

      addLoftedLimb(10, 8, (t) => {
        const p = samplePrimary(t);
        return {
          cx: p.cx,
          cy: p.cy,
          cz: p.cz,
          radius: 0.23 * (1.0 - t * 0.56),
          fluteCount: 5.0,
          fluteAmp: 0.05,
          twist: b * 1.1 + t * 0.6,
          taperSlope: 0.22,
          colR: 0.175,
          colG: 0.138,
          colB: 0.096
        };
      });

      // Two Secondary Forked Sub-Branches sprouting off each Primary Bough!
      for (const forkSign of [-1, 1]) {
        const tFork = forkSign < 0 ? 0.52 : 0.68;
        const pAnchor = samplePrimary(tFork);
        const subAz = az + forkSign * 0.62;
        const sDx = Math.cos(subAz);
        const sDz = Math.sin(subAz);

        addLoftedLimb(6, 5, (t) => {
          const out = Math.pow(t, 1.15) * 0.72;
          return {
            cx: pAnchor.cx + sDx * out,
            cy: pAnchor.cy + t * 0.92,
            cz: pAnchor.cz + sDz * out,
            radius: 0.115 * (1.0 - t * 0.58),
            fluteCount: 4.0,
            fluteAmp: 0.03,
            twist: t * 0.4,
            taperSlope: 0.25,
            colR: 0.188,
            colG: 0.148,
            colB: 0.104
          };
        });
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
    geo.computeVertexNormals();
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    return geo;
  }

  /**
   * Builds a Multi-Lobed Phyllotaxis Botanical Leaf-Cluster Foliage Cloud Geometry (~1,864 tris/cloud, +0 Draw Calls!):
   * Combines a shaded interior umbrella core + 6 overlapping peripheral bough leaf-tuft pads + 2 sunlit apical shoot puffs
   * with deep inter-lobe pocket AO and young-leaf chlorophyll tip highlights.
   */
  static _buildOrganicFoliageCloudGeometry() {
    const positions = [];
    const normals = [];
    const uvs = [];
    const colors = [];
    const indices = [];

    const appendLeafSubTuft = (cx, cy, cz, rx, ry, rz, phase, aoMult, shootBoost, uSegs, vSegs) => {
      const baseIdx = positions.length / 3;
      for (let iv = 0; iv <= vSegs; iv++) {
        const v = iv / vSegs;
        const phi = v * Math.PI;
        const sinPhi = Math.sin(phi);
        const cosPhi = Math.cos(phi);

        for (let iu = 0; iu <= uSegs; iu++) {
          const u = iu / uSegs;
          const th = u * Math.PI * 2.0 + phase;

          const nx = -Math.cos(th) * sinPhi;
          const ny = cosPhi;
          const nz = Math.sin(th) * sinPhi;

          // Deep scalloped tropical leaflet cluster harmonics
          const lobeA = 0.145 * Math.sin(3.0 * th + 0.5) * sinPhi * sinPhi;
          const lobeB = 0.095 * Math.cos(5.0 * th - 2.0 * phi);
          const lobeC = 0.072 * Math.sin(4.0 * phi + 2.0 * th);
          const tuftD = 0.052 * Math.sin(9.0 * th + 5.0 * phi) * sinPhi;
          const tuftE = 0.034 * Math.cos(13.0 * th - 7.0 * phi) * sinPhi;

          const bottomFlatten = ny < 0.0 ? 0.74 : 1.0;
          const r = 1.0 + lobeA + lobeB + lobeC + tuftD + tuftE;

          const px = cx + nx * r * rx;
          const py = cy + ny * r * ry * bottomFlatten;
          const pz = cz + nz * r * rz;

          // Outward radial + local sub-tuft normal blend for crisp leaf-clump self-shading
          const outLen = Math.hypot(px, py * 1.15, pz) || 1.0;
          const bNx = nx * 0.68 + (px / outLen) * 0.32;
          const bNy = ny * 0.68 + ((py * 1.15) / outLen) * 0.32;
          const bNz = nz * 0.68 + (pz / outLen) * 0.32;
          const nInv = 1.0 / (Math.hypot(bNx, bNy, bNz) || 1.0);

          // Deep interior bough pocket AO + sunlit young-shoot leaf tips
          const radialExp = Math.min(1.0, Math.hypot(px, Math.max(0.0, py) * 1.1, pz) / 1.06);
          const sunGrad = Math.max(0.0, Math.min(1.0, (ny + 0.78) / 1.78));
          const pocketAO = (0.56 + 0.44 * radialExp) * aoMult;
          const clumpVar = 0.5 + 0.5 * Math.sin(3.0 * th + phi * 2.0 + phase);
          const shootTip = Math.max(0.0, tuftD + tuftE) * 6.8 * sunGrad * shootBoost;

          const colR = (0.034 + 0.104 * sunGrad + 0.022 * clumpVar) * pocketAO + 0.042 * shootTip;
          const colG = (0.118 + 0.246 * sunGrad + 0.038 * clumpVar) * pocketAO + 0.074 * shootTip;
          const colB = (0.024 + 0.052 * sunGrad + 0.012 * clumpVar) * pocketAO + 0.015 * shootTip;

          positions.push(px, py, pz);
          normals.push(bNx * nInv, bNy * nInv, bNz * nInv);
          uvs.push(u, 1.0 - v);
          colors.push(colR, colG, colB);
        }
      }

      const stride = uSegs + 1;
      for (let iv = 0; iv < vSegs; iv++) {
        for (let iu = 0; iu < uSegs; iu++) {
          const a = baseIdx + iv * stride + iu;
          const b = a + 1;
          const c = a + stride + 1;
          const d = a + stride;
          indices.push(a, d, b, b, d, c);
        }
      }
    };

    // 1. Shaded Interior Umbrella Core
    appendLeafSubTuft(0.0, -0.02, 0.0, 0.76, 0.76, 0.76, 0.0, 0.78, 0.65, 14, 9);

    // 2. Six Overlapping Phyllotaxis Peripheral Bough Leaf-Cluster Pads
    for (let k = 0; k < 6; k++) {
      const az = k * 2.3999632 + 0.25;
      const dist = 0.42 + (k % 2) * 0.07;
      const cy = -0.08 + (k % 3) * 0.10;
      const rx = 0.55 + (k % 2) * 0.04;
      const ry = 0.54 + ((k + 1) % 2) * 0.05;
      const rz = 0.53 + (k % 3) * 0.03;
      const ao = 0.90 + (k % 3) * 0.06;
      appendLeafSubTuft(
        Math.cos(az) * dist,
        cy,
        Math.sin(az) * dist,
        rx,
        ry,
        rz,
        az * 0.7,
        ao,
        1.05,
        9,
        6
      );
    }

    // 3. Two Sunlit Apical Shoot Crownlets
    for (let k = 0; k < 2; k++) {
      const az = k * Math.PI + 0.85;
      appendLeafSubTuft(
        Math.cos(az) * 0.24,
        0.34 + k * 0.05,
        Math.sin(az) * 0.24,
        0.48,
        0.46,
        0.48,
        az,
        1.06,
        1.30,
        8,
        5
      );
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
    geo.computeVertexNormals();
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    return geo;
  }

  /**
   * Builds a 10x High-Detail UI Depok Campus Pedestrian Garden Lamp (~1,360 tris/lamp):
   * - 28-segment fluted pedestal & tapered steel shaft with torus moldings
   * - Warm amber-ivory lantern glass diffuser + 4 physical 3D corner bronze lantern cage struts
   * - 2-tier Tajug-inspired tropical rain-hood & spire finial!
   */
  static _buildCampusGardenLampGeometry() {
    const radialSegs = 18;
    const stations = [
      { y: -1.25, r: 0.15, flute: 0.0, uvV: 0.00, col: [0.14, 0.15, 0.16] },
      { y: 0.0, r: 0.18, flute: 0.0, uvV: 0.06, col: [0.17, 0.18, 0.20] },
      { y: 0.08, r: 0.19, flute: 0.0, uvV: 0.10, col: [0.20, 0.21, 0.23] },
      { y: 0.14, r: 0.16, flute: 0.012, uvV: 0.14, col: [0.18, 0.19, 0.21] },
      { y: 0.44, r: 0.145, flute: 0.012, uvV: 0.22, col: [0.18, 0.19, 0.21] },
      { y: 0.49, r: 0.165, flute: 0.0, uvV: 0.26, col: [0.22, 0.23, 0.25] }, // Molded pedestal torus ring
      { y: 0.54, r: 0.085, flute: 0.006, uvV: 0.30, col: [0.16, 0.17, 0.19] },
      { y: 1.85, r: 0.068, flute: 0.005, uvV: 0.44, col: [0.16, 0.17, 0.19] },
      { y: 1.90, r: 0.088, flute: 0.0, uvV: 0.48, col: [0.22, 0.23, 0.25] }, // Mid-shaft decorative collar
      { y: 1.95, r: 0.065, flute: 0.004, uvV: 0.52, col: [0.16, 0.17, 0.19] },
      { y: 3.18, r: 0.055, flute: 0.003, uvV: 0.60, col: [0.16, 0.17, 0.19] },
      { y: 3.24, r: 0.11, flute: 0.0, uvV: 0.63, col: [0.20, 0.21, 0.23] },
      { y: 3.30, r: 0.165, flute: 0.0, uvV: 0.66, col: [0.21, 0.22, 0.24] }, // Lantern lower bronze tray
      // Photocell-controlled Frosted Borosilicate Lantern Glass Chamber (uvV > 1.50; neutral glass by day, 2900K HDR by night!)
      { y: 3.31, r: 0.134, flute: 0.0, uvV: 1.70, col: [0.75, 0.80, 0.85] },
      { y: 3.51, r: 0.149, flute: 0.0, uvV: 2.00, col: [0.80, 0.85, 0.90] }, // LED / warm mantle waist core
      { y: 3.70, r: 0.162, flute: 0.0, uvV: 1.70, col: [0.76, 0.81, 0.86] },
      // 2-Tier Tajug Rain-Hood (uvV = 1.35 on lower eave underside for warm 2900K downward bounce!)
      { y: 3.71, r: 0.27, flute: 0.0, uvV: 1.35, col: [0.20, 0.21, 0.23] },
      { y: 3.73, r: 0.272, flute: 0.0, uvV: 0.82, col: [0.15, 0.16, 0.18] },
      { y: 3.81, r: 0.14, flute: 0.0, uvV: 0.86, col: [0.17, 0.18, 0.20] },
      { y: 3.84, r: 0.18, flute: 0.0, uvV: 0.90, col: [0.15, 0.16, 0.18] },
      { y: 3.94, r: 0.036, flute: 0.0, uvV: 0.94, col: [0.22, 0.21, 0.18] }, // Photocell mounting collar
      // Top NEMA Twist-Lock Ambient-Lux Photoelectric Light Sensor Cap ("Sensor Cahaya")
      { y: 3.95, r: 0.034, flute: 0.0, uvV: 0.96, col: [0.10, 0.19, 0.32] },
      { y: 4.00, r: 0.032, flute: 0.0, uvV: 0.98, col: [0.12, 0.22, 0.36] },
      { y: 4.02, r: 0.008, flute: 0.0, uvV: 1.00, col: [0.18, 0.28, 0.42] }
    ];

    const positions = [];
    const normals = [];
    const uvs = [];
    const colors = [];
    const indices = [];

    const rings = [];
    for (let is = 0; is < stations.length; is++) {
      const st = stations[is];
      const ring = [];
      for (let ir = 0; ir <= radialSegs; ir++) {
        const u = ir / radialSegs;
        const th = u * Math.PI * 2.0;
        const c = Math.cos(th);
        const s = Math.sin(th);
        const r = st.r + st.flute * Math.cos(8.0 * th);
        const idx = positions.length / 3;
        positions.push(c * r, st.y, s * r);
        normals.push(c, 0.08, s);
        uvs.push(u, st.uvV);
        colors.push(st.col[0], st.col[1], st.col[2]);
        ring.push(idx);
      }
      rings.push(ring);
    }

    for (let is = 0; is < stations.length - 1; is++) {
      for (let ir = 0; ir < radialSegs; ir++) {
        const a = rings[is][ir];
        const b = rings[is][ir + 1];
        const c = rings[is + 1][ir + 1];
        const d = rings[is + 1][ir];
        indices.push(a, d, c, a, c, b);
      }
    }

    // 4 Physical 3D Corner Bronze Lantern Cage Struts framing the warm lantern glass
    for (let k = 0; k < 4; k++) {
      const ang = (k + 0.5) * (Math.PI * 0.5);
      const strut = new THREE.BoxGeometry(0.018, 0.42, 0.018);
      strut.rotateZ(-0.06);
      strut.rotateY(ang);
      strut.translate(Math.cos(ang) * 0.162, 3.51, Math.sin(ang) * 0.162);

      const p = strut.attributes.position.array;
      const n = strut.attributes.normal.array;
      const base = positions.length / 3;
      for (let i = 0; i < strut.attributes.position.count; i++) {
        positions.push(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
        normals.push(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]);
        uvs.push(0.5, 0.85);
        colors.push(0.15, 0.16, 0.18);
      }
      const idx = strut.index.array;
      for (let i = 0; i < idx.length; i++) {
        indices.push(base + idx[i]);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
    geo.computeVertexNormals();
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    return geo;
  }

  /**
   * Builds the entire Kampus UI Depok multi-species biome + courtyard planter shrubs + garden lamps
   * in strictly 3 Draw Calls.
   */
  static buildCampusBiome(
    groundHeightFn = TerrainDEMLoader.sampleGroundHeight,
    sharedSurfaceUniforms = null
  ) {
    const uniforms = sharedSurfaceUniforms || createSharedSurfaceUniforms();
    const biomeGroup = new THREE.Group();
    biomeGroup.name = 'KampusUIAutogenBiome';

    const baseValidTrees = AutogenBiomeSpawner.DEFAULT_TREE_SPAWNS.filter((t) =>
      AutogenBiomeSpawner.isAllowedTreeSpawn(t.x, t.z, groundHeightFn)
    );
    const { extraTrees, forestShrubs } = AutogenBiomeSpawner._generateArchipelagoForestSpawns(
      groundHeightFn,
      baseValidTrees
    );
    const validTrees = baseValidTrees.concat(extraTrees);

    const trunkMatrices = [];
    const trunkColors = [];
    const canopyMatrices = [];
    const canopyColors = [];

    const pos = new THREE.Vector3();
    const euler = new THREE.Euler();
    const quat = new THREE.Quaternion();
    const scl = new THREE.Vector3();
    const mat = new THREE.Matrix4();

    const pushTransform = (targetList, px, py, pz, rx, ry, rz, sx, sy, sz, colorList = null, cr = 1.0, cg = 1.0, cb = 1.0) => {
      pos.set(px, py, pz);
      euler.set(rx, ry, rz);
      quat.setFromEuler(euler);
      scl.set(sx, sy, sz);
      const m = new THREE.Matrix4().compose(pos, quat, scl);
      targetList.push(m);
      if (colorList) {
        colorList.push(cr, cg, cb);
      }
    };

    for (let i = 0; i < validTrees.length; i++) {
      const tree = validTrees[i];
      const gy = groundHeightFn(tree.x, tree.z);
      const s = tree.scale || 1.0;
      const yaw = tree.yaw || 0.0;
      const bioVar = Math.sin(tree.x * 0.37 + tree.z * 0.53) * 0.5;

      if (tree.species === 'TREMBESI') {
        const tb = 1.0 + bioVar * 0.10;
        pushTransform(
          trunkMatrices,
          tree.x,
          gy,
          tree.z,
          0.02 * Math.sin(yaw),
          yaw,
          0.02 * Math.cos(yaw),
          1.32 * s,
          1.06 * s,
          1.32 * s,
          trunkColors,
          1.12 * tb,
          1.14 * tb,
          1.22 * tb
        );

        pushTransform(
          canopyMatrices,
          tree.x,
          gy + 6.95 * s,
          tree.z,
          0.0,
          yaw,
          0.0,
          4.55 * s,
          1.85 * s,
          4.55 * s,
          canopyColors,
          1.10 + bioVar * 0.09,
          1.06 + bioVar * 0.07,
          0.90 + bioVar * 0.05
        );

        for (let k = 0; k < 6; k++) {
          const az = yaw + (k / 6) * Math.PI * 2.0 + (k % 2) * 0.14;
          const dist = (3.65 + (k % 3) * 0.35) * s;
          const hOff = (5.55 + (k % 2) * 0.55) * s;
          const rxPad = (3.05 + (k % 3) * 0.25) * s;
          const ryPad = (1.32 + (k % 2) * 0.18) * s;
          const subLit = 0.80 + (k % 2) * 0.12 + bioVar * 0.06;
          pushTransform(
            canopyMatrices,
            tree.x + Math.cos(az) * dist,
            gy + hOff,
            tree.z + Math.sin(az) * dist,
            0.05 * Math.sin(az),
            az * 0.7,
            0.05 * Math.cos(az),
            rxPad,
            ryPad,
            rxPad * 0.92,
            canopyColors,
            0.94 * subLit,
            0.98 * subLit,
            0.90 * subLit
          );
        }
      } else if (tree.species === 'MAHONI') {
        const mb = 1.0 + bioVar * 0.10;
        pushTransform(
          trunkMatrices,
          tree.x,
          gy,
          tree.z,
          0.025 * Math.cos(yaw),
          yaw,
          -0.02 * Math.sin(yaw),
          0.94 * s,
          1.26 * s,
          0.94 * s,
          trunkColors,
          1.28 * mb,
          0.92 * mb,
          0.78 * mb
        );

        for (let k = 0; k < 3; k++) {
          const az = yaw + (k / 3) * Math.PI * 2.0;
          pushTransform(
            canopyMatrices,
            tree.x + Math.cos(az) * 1.65 * s,
            gy + (6.15 + k * 0.25) * s,
            tree.z + Math.sin(az) * 1.65 * s,
            0.0,
            az,
            0.0,
            2.65 * s,
            2.15 * s,
            2.45 * s,
            canopyColors,
            0.76 + bioVar * 0.06,
            0.80 + bioVar * 0.06,
            0.76
          );
        }
        for (let k = 0; k < 2; k++) {
          const az = yaw + 0.9 + k * Math.PI;
          pushTransform(
            canopyMatrices,
            tree.x + Math.cos(az) * 1.05 * s,
            gy + (7.95 + k * 0.22) * s,
            tree.z + Math.sin(az) * 1.05 * s,
            0.0,
            az,
            0.0,
            2.35 * s,
            2.05 * s,
            2.25 * s,
            canopyColors,
            0.96 + bioVar * 0.08,
            0.92 + bioVar * 0.06,
            0.82
          );
        }
        pushTransform(
          canopyMatrices,
          tree.x,
          gy + 9.45 * s,
          tree.z,
          0.0,
          yaw + 0.5,
          0.0,
          2.05 * s,
          1.95 * s,
          2.05 * s,
          canopyColors,
          1.20 + bioVar * 0.10,
          0.96 + bioVar * 0.06,
          0.78
        );
      } else if (tree.species === 'CEMARA') {
        const cb = 1.0 + bioVar * 0.08;
        pushTransform(
          trunkMatrices,
          tree.x,
          gy,
          tree.z,
          0.0,
          yaw,
          0.0,
          0.52 * s,
          1.72 * s,
          0.52 * s,
          trunkColors,
          0.82 * cb,
          0.84 * cb,
          0.96 * cb
        );
        for (let tier = 0; tier < 6; tier++) {
          const frac = tier / 5.0;
          const tierY = gy + (3.1 + tier * 1.95) * s;
          const tierR = 2.65 * (1.0 - frac * 0.72) * s;
          const tierH = 1.35 * (1.0 - frac * 0.35) * s;
          const tierAO = 0.72 + 0.32 * frac;
          pushTransform(
            canopyMatrices,
            tree.x,
            tierY,
            tree.z,
            0.0,
            yaw + tier * 0.75,
            0.0,
            tierR,
            tierH,
            tierR,
            canopyColors,
            0.84 * tierAO,
            0.92 * tierAO,
            1.14 * tierAO
          );
        }
      } else {
        const leanAngle = 0.065;
        const leanX = Math.sin(yaw) * leanAngle;
        const leanZ = -Math.cos(yaw) * leanAngle;
        const trunkSy = 1.48 * s;
        const pb = 1.0 + bioVar * 0.08;

        pushTransform(
          trunkMatrices,
          tree.x,
          gy,
          tree.z,
          leanX,
          yaw,
          leanZ,
          0.35 * s,
          trunkSy,
          0.35 * s,
          trunkColors,
          3.08 * pb,
          3.84 * pb,
          5.18 * pb
        );

        const palmTopH = 5.75 * trunkSy;
        euler.set(leanX, yaw, leanZ);
        quat.setFromEuler(euler);
        pos.set(tree.x, gy, tree.z);
        scl.set(1.0, 1.0, 1.0);
        mat.compose(pos, quat, scl);
        const topVecX = mat.elements[4] * palmTopH + mat.elements[12];
        const topVecY = mat.elements[5] * palmTopH + mat.elements[13];
        const topVecZ = mat.elements[6] * palmTopH + mat.elements[14];

        // Smooth emerald-green Roystonea regia crownshaft ("pelepah hijau mulus") + apical spear heart
        pushTransform(
          canopyMatrices,
          topVecX,
          topVecY - 0.18 * s,
          topVecZ,
          leanX,
          yaw,
          leanZ,
          0.24 * s,
          0.92 * s,
          0.24 * s,
          canopyColors,
          0.82,
          1.16,
          0.72
        );
        pushTransform(
          canopyMatrices,
          topVecX,
          topVecY + 0.52 * s,
          topVecZ,
          leanX,
          yaw,
          leanZ,
          0.16 * s,
          0.62 * s,
          0.16 * s,
          canopyColors,
          0.96,
          1.22,
          0.78
        );

        // 10 Parabolic-Arching 3-Segment Pinnate Royal Palm Fronds (2 phyllotaxis whorls, +0 Draw Calls!)
        for (let f = 0; f < 10; f++) {
          const isUpperWhorl = f < 5;
          const whorlIdx = f % 5;
          const frondYaw = yaw + (whorlIdx / 5) * Math.PI * 2.0 + (isUpperWhorl ? 0.31 : 0.0);
          const sinY = Math.sin(frondYaw);
          const cosY = Math.cos(frondYaw);
          const baseY = topVecY + (isUpperWhorl ? 0.34 : 0.14) * s;
          const frondAge = isUpperWhorl ? 1.08 : 0.94;
          const segSpecs = isUpperWhorl
            ? [
                { r: 0.72, dy: 0.34, pitch: -0.34, sx: 0.34, sy: 0.14, sz: 0.95 },
                { r: 1.68, dy: 0.38, pitch: 0.22, sx: 0.54, sy: 0.17, sz: 1.24 },
                { r: 2.62, dy: -0.06, pitch: 0.74, sx: 0.42, sy: 0.14, sz: 1.08 }
              ]
            : [
                { r: 0.78, dy: 0.14, pitch: -0.08, sx: 0.38, sy: 0.15, sz: 1.02 },
                { r: 1.82, dy: -0.08, pitch: 0.46, sx: 0.58, sy: 0.18, sz: 1.32 },
                { r: 2.78, dy: -0.68, pitch: 0.98, sx: 0.44, sy: 0.14, sz: 1.12 }
              ];
          for (let sg = 0; sg < 3; sg++) {
            const sp = segSpecs[sg];
            const tipTint = 1.0 - sg * 0.04;
            pushTransform(
              canopyMatrices,
              topVecX + sinY * sp.r * s,
              baseY + sp.dy * s,
              topVecZ + cosY * sp.r * s,
              sp.pitch,
              frondYaw,
              0.0,
              sp.sx * s,
              sp.sy * s,
              sp.sz * s,
              canopyColors,
              1.08 * frondAge * tipTint,
              1.06 * frondAge,
              0.78 * frondAge * tipTint
            );
          }
        }
      }
    }

    // 48 Lush Green Manicured Ornamental Hedge Shrubs inside the 8 Gedung Rektorat UI Base Planters
    // (0 extra Draw Calls — shares KampusUITreeCanopiesInstanced!)
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        for (let b = 0; b < 6; b++) {
          const offset = 3.95 + b * 2.04;
          const ph = b * 0.85 + sx * 0.4 + sz * 0.7;
          const hv = 0.96 + (b % 3) * 0.05;
          // Z-facing planter trough (x = sx * offset, z = sz * 15.15)
          pushTransform(
            canopyMatrices,
            sx * offset,
            1.08,
            sz * 15.15,
            0.0,
            ph,
            0.0,
            0.96,
            0.34,
            0.36,
            canopyColors,
            1.08 * hv,
            0.98 * hv,
            0.82 * hv
          );
          // X-facing planter trough (x = sx * 15.15, z = sz * offset)
          pushTransform(
            canopyMatrices,
            sx * 15.15,
            1.08,
            sz * offset,
            0.0,
            ph + 1.1,
            0.0,
            0.36,
            0.34,
            0.96,
            canopyColors,
            1.04 * hv,
            1.00 * hv,
            0.84 * hv
          );
        }
      }
    }

    // Lush Rainforest-Floor, Mata Air Rim, & Danau Kecil Rektorat Understory Shrubs/Ferns
    // (0 extra Draw Calls — shares KampusUITreeCanopiesInstanced!)
    for (let i = 0; i < forestShrubs.length; i++) {
      const sh = forestShrubs[i];
      const sgy = groundHeightFn(sh.x, sh.z);
      const fv = 0.88 + 0.22 * (0.5 + 0.5 * Math.sin(sh.x * 0.41 - sh.z * 0.33));
      pushTransform(
        canopyMatrices,
        sh.x,
        sgy + 0.42 * sh.sy,
        sh.z,
        0.0,
        sh.yaw,
        0.0,
        sh.sx,
        sh.sy,
        sh.sz,
        canopyColors,
        0.84 * fv,
        0.95 * fv,
        0.88 * fv
      );
    }

    // Draw Call #1: All Multi-Branch Organic Tree Trunks & Ringed Royal Palms
    const trunkMat = new THREE.MeshStandardMaterial({
      name: 'CampusOrganicBarkMat',
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.90,
      metalness: 0.02,
      envMapIntensity: 0.20
    });
    injectOmniScaleSurfaceShader(trunkMat, {
      ySegmentStart: 120.0,
      segmentCount: 0.0,
      enableWindSway: true,
      surfaceProfile: SURFACE_PROFILE.ASPHALT_TEAK_BARK,
      bumpScale: 0.74,
      sharedUniforms: uniforms
    });

    const trunkGeo = AutogenBiomeSpawner._buildOrganicBranchingTrunkGeometry();
    const trunkInstanced = new THREE.InstancedMesh(trunkGeo, trunkMat, trunkMatrices.length);
    trunkInstanced.name = 'KampusUITreeTrunksInstanced';
    trunkInstanced.castShadow = true;
    trunkInstanced.receiveShadow = true;
    const tmpCol = new THREE.Color();
    for (let i = 0; i < trunkMatrices.length; i++) {
      trunkInstanced.setMatrixAt(i, trunkMatrices[i]);
      tmpCol.setRGB(trunkColors[i * 3], trunkColors[i * 3 + 1], trunkColors[i * 3 + 2], THREE.LinearSRGBColorSpace);
      trunkInstanced.setColorAt(i, tmpCol);
    }
    if (trunkInstanced.instanceMatrix) {
      trunkInstanced.instanceMatrix.needsUpdate = true;
    }
    if (trunkInstanced.instanceColor) {
      trunkInstanced.instanceColor.needsUpdate = true;
    }

    // Draw Call #2: All Multi-Species Organic Foliage Canopy Clouds + Courtyard Planter Shrubs
    const canopyMat = new THREE.MeshStandardMaterial({
      name: 'CampusOrganicCanopyMat',
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.82,
      metalness: 0.02,
      envMapIntensity: 0.24
    });
    injectOmniScaleSurfaceShader(canopyMat, {
      ySegmentStart: 120.0,
      segmentCount: 0.0,
      enableWindSway: true,
      surfaceProfile: SURFACE_PROFILE.FOLIAGE_CANOPY,
      bumpScale: 0.92,
      sharedUniforms: uniforms
    });

    const canopyGeo = AutogenBiomeSpawner._buildOrganicFoliageCloudGeometry();
    const canopyInstanced = new THREE.InstancedMesh(canopyGeo, canopyMat, canopyMatrices.length);
    canopyInstanced.name = 'KampusUITreeCanopiesInstanced';
    canopyInstanced.castShadow = true;
    canopyInstanced.receiveShadow = true;
    for (let i = 0; i < canopyMatrices.length; i++) {
      canopyInstanced.setMatrixAt(i, canopyMatrices[i]);
      tmpCol.setRGB(canopyColors[i * 3], canopyColors[i * 3 + 1], canopyColors[i * 3 + 2], THREE.LinearSRGBColorSpace);
      canopyInstanced.setColorAt(i, tmpCol);
    }
    if (canopyInstanced.instanceMatrix) {
      canopyInstanced.instanceMatrix.needsUpdate = true;
    }
    if (canopyInstanced.instanceColor) {
      canopyInstanced.instanceColor.needsUpdate = true;
    }

    // Draw Call #3: 64 Photocell-Equipped Fluted Nusantara Campus Streetlamps (+0 Draw Calls!)
    const lampCoords = AutogenBiomeSpawner.getLampCoords();

    const lampMat = new THREE.MeshStandardMaterial({
      name: 'CampusGardenLampMat',
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.50,
      metalness: 0.38,
      envMapIntensity: 0.36
    });
    injectOmniScaleSurfaceShader(lampMat, {
      ySegmentStart: 120.0,
      segmentCount: 0.0,
      surfaceProfile: SURFACE_PROFILE.MAKARA_STONE_BRONZE,
      bumpScale: 0.85,
      sharedUniforms: uniforms
    });

    const lampGeo = AutogenBiomeSpawner._buildCampusGardenLampGeometry();
    const lampInstanced = new THREE.InstancedMesh(lampGeo, lampMat, lampCoords.length);
    lampInstanced.name = 'KampusUIGardenLampsInstanced';
    lampInstanced.castShadow = true;
    lampInstanced.receiveShadow = true;

    const unitQuat = new THREE.Quaternion();
    const unitScale = new THREE.Vector3(1.0, 1.0, 1.0);
    for (let i = 0; i < lampCoords.length; i++) {
      const lc = lampCoords[i];
      const pierY = TerrainDEMLoader.sampleFerryPierDeckY(lc.x, lc.z);
      const ly = pierY !== null ? pierY : Math.max(0.0, groundHeightFn(lc.x, lc.z));
      pos.set(lc.x, ly, lc.z);
      mat.compose(pos, unitQuat, unitScale);
      lampInstanced.setMatrixAt(i, mat);
    }
    if (lampInstanced.instanceMatrix) {
      lampInstanced.instanceMatrix.needsUpdate = true;
    }

    biomeGroup.add(trunkInstanced, canopyInstanced, lampInstanced);
    biomeGroup.userData.treeCount = validTrees.length;
    biomeGroup.userData.lampCount = lampCoords.length;
    biomeGroup.userData.forestShrubCount = forestShrubs.length;
    biomeGroup.userData.pulauUSouthForestCount = validTrees.filter(
      (t) => t.z < 536.0 && t.x < -110.0
    ).length;
    biomeGroup.userData.pulauIForestCount = validTrees.filter((t) => t.z >= 536.0).length;
    biomeGroup.userData.spawnedTrees = validTrees;
    biomeGroup.userData.forestShrubs = forestShrubs;
    biomeGroup.userData.lampCoords = lampCoords;
    biomeGroup.updateMatrixWorld(true);
    AutogenBiomeSpawner._initFrustumCompactionState(biomeGroup, [
      trunkInstanced,
      canopyInstanced,
      lampInstanced
    ]);
    return biomeGroup;
  }

  /**
   * Precomputes immutable source instance matrices/colors + exact local-to-group bounding spheres
   * (with +1.35m wind-sway & soft-shadow safety margin) for zero-GC per-pass frustum compaction.
   * Build-time counts & attributes remain 100% intact for 4-Gate verification & shader pre-warming.
   */
  static _initFrustumCompactionState(biomeGroup, instancedList) {
    const entries = [];
    for (let m = 0; m < instancedList.length; m++) {
      const inst = instancedList[m];
      if (!inst || !inst.instanceMatrix || !inst.geometry) continue;
      const fullCount = inst.count | 0;
      if (fullCount <= 0) continue;

      let bx = 0.0, by = 0.0, bz = 0.0, br = 4.0;
      const posArr = inst.geometry.attributes?.position?.array;
      if (posArr && posArr.length >= 3) {
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        for (let k = 0, kLen = posArr.length; k < kLen; k += 3) {
          const px = posArr[k], py = posArr[k + 1], pz = posArr[k + 2];
          if (px < minX) minX = px;
          if (px > maxX) maxX = px;
          if (py < minY) minY = py;
          if (py > maxY) maxY = py;
          if (pz < minZ) minZ = pz;
          if (pz > maxZ) maxZ = pz;
        }
        bx = 0.5 * (minX + maxX);
        by = 0.5 * (minY + maxY);
        bz = 0.5 * (minZ + maxZ);
        let maxR2 = 0.0;
        for (let k = 0, kLen = posArr.length; k < kLen; k += 3) {
          const dx = posArr[k] - bx;
          const dy = posArr[k + 1] - by;
          const dz = posArr[k + 2] - bz;
          const r2 = dx * dx + dy * dy + dz * dz;
          if (r2 > maxR2) maxR2 = r2;
        }
        br = Math.sqrt(maxR2);
      } else {
        if (!inst.geometry.boundingSphere && typeof inst.geometry.computeBoundingSphere === 'function') {
          inst.geometry.computeBoundingSphere();
        }
        const bs = inst.geometry.boundingSphere;
        bx = bs?.center?.x || 0.0;
        by = bs?.center?.y || 0.0;
        bz = bs?.center?.z || 0.0;
        br = bs?.radius || 4.0;
      }

      const rawMatArr = inst.instanceMatrix.array || inst.instanceMatrix;
      if (!rawMatArr || rawMatArr.length < fullCount * 16) continue;
      const srcMat = new Float32Array(rawMatArr);
      const rawColArr = inst.instanceColor ? (inst.instanceColor.array || inst.instanceColor) : null;
      const hasColor = Boolean(rawColArr && rawColArr.length >= fullCount * 3);
      const srcCol = hasColor ? new Float32Array(rawColArr) : null;
      const spheres = new Float32Array(fullCount * 4);

      for (let i = 0; i < fullCount; i++) {
        const o = i * 16;
        const m0 = srcMat[o], m1 = srcMat[o + 1], m2 = srcMat[o + 2];
        const m4 = srcMat[o + 4], m5 = srcMat[o + 5], m6 = srcMat[o + 6];
        const m8 = srcMat[o + 8], m9 = srcMat[o + 9], m10 = srcMat[o + 10];
        const m12 = srcMat[o + 12], m13 = srcMat[o + 13], m14 = srcMat[o + 14];

        const cx = m0 * bx + m4 * by + m8 * bz + m12;
        const cy = m1 * bx + m5 * by + m9 * bz + m13;
        const cz = m2 * bx + m6 * by + m10 * bz + m14;

        const sx2 = m0 * m0 + m1 * m1 + m2 * m2;
        const sy2 = m4 * m4 + m5 * m5 + m6 * m6;
        const sz2 = m8 * m8 + m9 * m9 + m10 * m10;
        const maxScale = Math.sqrt(Math.max(sx2, sy2, sz2));

        const so = i * 4;
        spheres[so] = cx;
        spheres[so + 1] = cy;
        spheres[so + 2] = cz;
        spheres[so + 3] = br * maxScale + 1.35;
      }

      inst.frustumCulled = false;
      inst.userData.fullInstanceCount = fullCount;
      inst.userData.srcInstanceMatrices = srcMat;
      if (srcCol) inst.userData.srcInstanceColors = srcCol;

      entries.push({
        mesh: inst,
        fullCount,
        srcMat,
        srcCol,
        spheres
      });
    }

    biomeGroup.userData._frustumCompaction = {
      entries,
      mainProjView: new THREE.Matrix4(),
      shadowProjView: new THREE.Matrix4(),
      planesA: new Float32Array(24),
      planesB: new Float32Array(24)
    };
  }

  static _extractFrustumPlanes(projViewMat, outPlanes24) {
    const me = projViewMat.elements;
    const m0 = me[0], m1 = me[1], m2 = me[2], m3 = me[3];
    const m4 = me[4], m5 = me[5], m6 = me[6], m7 = me[7];
    const m8 = me[8], m9 = me[9], m10 = me[10], m11 = me[11];
    const m12 = me[12], m13 = me[13], m14 = me[14], m15 = me[15];

    // Right, Left, Bottom, Top, Far, Near (Gribb-Hartmann inward OpenGL clip planes)
    outPlanes24[0] = m3 - m0; outPlanes24[1] = m7 - m4; outPlanes24[2] = m11 - m8; outPlanes24[3] = m15 - m12;
    outPlanes24[4] = m3 + m0; outPlanes24[5] = m7 + m4; outPlanes24[6] = m11 + m8; outPlanes24[7] = m15 + m12;
    outPlanes24[8] = m3 + m1; outPlanes24[9] = m7 + m5; outPlanes24[10] = m11 + m9; outPlanes24[11] = m15 + m13;
    outPlanes24[12] = m3 - m1; outPlanes24[13] = m7 - m5; outPlanes24[14] = m11 - m9; outPlanes24[15] = m15 - m13;
    outPlanes24[16] = m3 - m2; outPlanes24[17] = m7 - m6; outPlanes24[18] = m11 - m10; outPlanes24[19] = m15 - m14;
    outPlanes24[20] = m3 + m2; outPlanes24[21] = m7 + m6; outPlanes24[22] = m11 + m10; outPlanes24[23] = m15 + m14;

    for (let p = 0; p < 24; p += 4) {
      const nx = outPlanes24[p];
      const ny = outPlanes24[p + 1];
      const nz = outPlanes24[p + 2];
      const invLen = 1.0 / (Math.sqrt(nx * nx + ny * ny + nz * nz) || 1.0);
      outPlanes24[p] = nx * invLen;
      outPlanes24[p + 1] = ny * invLen;
      outPlanes24[p + 2] = nz * invLen;
      outPlanes24[p + 3] *= invLen;
    }
  }

  /**
   * Compacts all 3 biome InstancedMeshes for the Planar Water Reflection pass:
   * keeps only instances intersecting the mirrored reflection frustum AND extending above waterY - 0.25m.
   */
  static cullForPlanarReflection(biomeGroup, reflUnobliquedProjViewMat, waterY = -1.45) {
    const fc = biomeGroup?.userData?._frustumCompaction;
    if (!fc || !reflUnobliquedProjViewMat) return;

    const planes = fc.planesA;
    AutogenBiomeSpawner._extractFrustumPlanes(reflUnobliquedProjViewMat, planes);
    const p0 = planes[0], p1 = planes[1], p2 = planes[2], p3 = planes[3];
    const p4 = planes[4], p5 = planes[5], p6 = planes[6], p7 = planes[7];
    const p8 = planes[8], p9 = planes[9], p10 = planes[10], p11 = planes[11];
    const p12 = planes[12], p13 = planes[13], p14 = planes[14], p15 = planes[15];
    const p16 = planes[16], p17 = planes[17], p18 = planes[18], p19 = planes[19];
    const p20 = planes[20], p21 = planes[21], p22 = planes[22], p23 = planes[23];
    const minReflY = waterY - 0.25;

    const entries = fc.entries;
    for (let eIdx = 0; eIdx < entries.length; eIdx++) {
      const entry = entries[eIdx];
      const inst = entry.mesh;
      const mw = inst.matrixWorld.elements;
      const ox = mw[12], oy = mw[13], oz = mw[14];
      const spheres = entry.spheres;
      const srcMat = entry.srcMat;
      const srcCol = entry.srcCol;
      const dstMat = inst.instanceMatrix.array || inst.instanceMatrix;
      const dstCol = srcCol && inst.instanceColor ? (inst.instanceColor.array || inst.instanceColor) : null;
      const fullCount = entry.fullCount;

      let vis = 0;
      for (let i = 0; i < fullCount; i++) {
        const so = i * 4;
        const r = spheres[so + 3];
        const cy = spheres[so + 1] + oy;
        if (cy + r < minReflY) continue;
        const cx = spheres[so] + ox;
        const cz = spheres[so + 2] + oz;
        const nr = -r;
        if (
          p0 * cx + p1 * cy + p2 * cz + p3 < nr ||
          p4 * cx + p5 * cy + p6 * cz + p7 < nr ||
          p8 * cx + p9 * cy + p10 * cz + p11 < nr ||
          p12 * cx + p13 * cy + p14 * cz + p15 < nr ||
          p16 * cx + p17 * cy + p18 * cz + p19 < nr ||
          p20 * cx + p21 * cy + p22 * cz + p23 < nr
        ) {
          continue;
        }
        const s16 = i * 16;
        const d16 = vis * 16;
        dstMat[d16] = srcMat[s16];
        dstMat[d16 + 1] = srcMat[s16 + 1];
        dstMat[d16 + 2] = srcMat[s16 + 2];
        dstMat[d16 + 3] = srcMat[s16 + 3];
        dstMat[d16 + 4] = srcMat[s16 + 4];
        dstMat[d16 + 5] = srcMat[s16 + 5];
        dstMat[d16 + 6] = srcMat[s16 + 6];
        dstMat[d16 + 7] = srcMat[s16 + 7];
        dstMat[d16 + 8] = srcMat[s16 + 8];
        dstMat[d16 + 9] = srcMat[s16 + 9];
        dstMat[d16 + 10] = srcMat[s16 + 10];
        dstMat[d16 + 11] = srcMat[s16 + 11];
        dstMat[d16 + 12] = srcMat[s16 + 12];
        dstMat[d16 + 13] = srcMat[s16 + 13];
        dstMat[d16 + 14] = srcMat[s16 + 14];
        dstMat[d16 + 15] = srcMat[s16 + 15];
        if (dstCol) {
          const s3 = i * 3;
          const d3 = vis * 3;
          dstCol[d3] = srcCol[s3];
          dstCol[d3 + 1] = srcCol[s3 + 1];
          dstCol[d3 + 2] = srcCol[s3 + 2];
        }
        vis++;
      }

      inst.count = vis;
      inst.visible = vis > 0;
      if (vis > 0) {
        const matAttr = inst.instanceMatrix;
        if (typeof matAttr.clearUpdateRanges === 'function') {
          matAttr.clearUpdateRanges();
          matAttr.addUpdateRange(0, vis * 16);
        }
        matAttr.needsUpdate = true;
        if (dstCol && inst.instanceColor) {
          const colAttr = inst.instanceColor;
          if (typeof colAttr.clearUpdateRanges === 'function') {
            colAttr.clearUpdateRanges();
            colAttr.addUpdateRange(0, vis * 3);
          }
          colAttr.needsUpdate = true;
        }
      }
    }
  }

  /**
   * Compacts all 3 biome InstancedMeshes for the Directional Shadow + Main Camera render pass:
   * keeps every instance inside (MainCameraFrustum U ActiveSunShadowFrustum) so off-screen trees
   * casting shadows into the view are 100.000% preserved with ZERO shadow pop or lag!
   */
  static cullForMainAndShadow(biomeGroup, mainCamera, sunLight = null, shadowActiveThisFrame = false) {
    const fc = biomeGroup?.userData?._frustumCompaction;
    if (!fc || !mainCamera) return;

    fc.mainProjView.multiplyMatrices(mainCamera.projectionMatrix, mainCamera.matrixWorldInverse);
    const planesA = fc.planesA;
    AutogenBiomeSpawner._extractFrustumPlanes(fc.mainProjView, planesA);
    const a0 = planesA[0], a1 = planesA[1], a2 = planesA[2], a3 = planesA[3];
    const a4 = planesA[4], a5 = planesA[5], a6 = planesA[6], a7 = planesA[7];
    const a8 = planesA[8], a9 = planesA[9], a10 = planesA[10], a11 = planesA[11];
    const a12 = planesA[12], a13 = planesA[13], a14 = planesA[14], a15 = planesA[15];
    const a16 = planesA[16], a17 = planesA[17], a18 = planesA[18], a19 = planesA[19];
    const a20 = planesA[20], a21 = planesA[21], a22 = planesA[22], a23 = planesA[23];

    let hasShadowFrustum = false;
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    let b4 = 0, b5 = 0, b6 = 0, b7 = 0;
    let b8 = 0, b9 = 0, b10 = 0, b11 = 0;
    let b12 = 0, b13 = 0, b14 = 0, b15 = 0;
    let b16 = 0, b17 = 0, b18 = 0, b19 = 0;
    let b20 = 0, b21 = 0, b22 = 0, b23 = 0;

    if (
      shadowActiveThisFrame &&
      sunLight &&
      sunLight.castShadow &&
      sunLight.shadow &&
      sunLight.shadow.camera &&
      (Number(sunLight.intensity) || 0.0) > 0.001
    ) {
      if (typeof sunLight.shadow.updateMatrices === 'function') {
        sunLight.shadow.updateMatrices(sunLight);
      }
      const shCam = sunLight.shadow.camera;
      fc.shadowProjView.multiplyMatrices(shCam.projectionMatrix, shCam.matrixWorldInverse);
      const planesB = fc.planesB;
      AutogenBiomeSpawner._extractFrustumPlanes(fc.shadowProjView, planesB);
      b0 = planesB[0]; b1 = planesB[1]; b2 = planesB[2]; b3 = planesB[3];
      b4 = planesB[4]; b5 = planesB[5]; b6 = planesB[6]; b7 = planesB[7];
      b8 = planesB[8]; b9 = planesB[9]; b10 = planesB[10]; b11 = planesB[11];
      b12 = planesB[12]; b13 = planesB[13]; b14 = planesB[14]; b15 = planesB[15];
      b16 = planesB[16]; b17 = planesB[17]; b18 = planesB[18]; b19 = planesB[19];
      b20 = planesB[20]; b21 = planesB[21]; b22 = planesB[22]; b23 = planesB[23];
      hasShadowFrustum = true;
    }

    const entries = fc.entries;
    for (let eIdx = 0; eIdx < entries.length; eIdx++) {
      const entry = entries[eIdx];
      const inst = entry.mesh;
      const mw = inst.matrixWorld.elements;
      const ox = mw[12], oy = mw[13], oz = mw[14];
      const spheres = entry.spheres;
      const srcMat = entry.srcMat;
      const srcCol = entry.srcCol;
      const dstMat = inst.instanceMatrix.array || inst.instanceMatrix;
      const dstCol = srcCol && inst.instanceColor ? (inst.instanceColor.array || inst.instanceColor) : null;
      const fullCount = entry.fullCount;

      let vis = 0;
      for (let i = 0; i < fullCount; i++) {
        const so = i * 4;
        const cx = spheres[so] + ox;
        const cy = spheres[so + 1] + oy;
        const cz = spheres[so + 2] + oz;
        const nr = -spheres[so + 3];

        const inMain = !(
          a0 * cx + a1 * cy + a2 * cz + a3 < nr ||
          a4 * cx + a5 * cy + a6 * cz + a7 < nr ||
          a8 * cx + a9 * cy + a10 * cz + a11 < nr ||
          a12 * cx + a13 * cy + a14 * cz + a15 < nr ||
          a16 * cx + a17 * cy + a18 * cz + a19 < nr ||
          a20 * cx + a21 * cy + a22 * cz + a23 < nr
        );
        if (!inMain) {
          if (
            !hasShadowFrustum ||
            b0 * cx + b1 * cy + b2 * cz + b3 < nr ||
            b4 * cx + b5 * cy + b6 * cz + b7 < nr ||
            b8 * cx + b9 * cy + b10 * cz + b11 < nr ||
            b12 * cx + b13 * cy + b14 * cz + b15 < nr ||
            b16 * cx + b17 * cy + b18 * cz + b19 < nr ||
            b20 * cx + b21 * cy + b22 * cz + b23 < nr
          ) {
            continue;
          }
        }

        const s16 = i * 16;
        const d16 = vis * 16;
        dstMat[d16] = srcMat[s16];
        dstMat[d16 + 1] = srcMat[s16 + 1];
        dstMat[d16 + 2] = srcMat[s16 + 2];
        dstMat[d16 + 3] = srcMat[s16 + 3];
        dstMat[d16 + 4] = srcMat[s16 + 4];
        dstMat[d16 + 5] = srcMat[s16 + 5];
        dstMat[d16 + 6] = srcMat[s16 + 6];
        dstMat[d16 + 7] = srcMat[s16 + 7];
        dstMat[d16 + 8] = srcMat[s16 + 8];
        dstMat[d16 + 9] = srcMat[s16 + 9];
        dstMat[d16 + 10] = srcMat[s16 + 10];
        dstMat[d16 + 11] = srcMat[s16 + 11];
        dstMat[d16 + 12] = srcMat[s16 + 12];
        dstMat[d16 + 13] = srcMat[s16 + 13];
        dstMat[d16 + 14] = srcMat[s16 + 14];
        dstMat[d16 + 15] = srcMat[s16 + 15];
        if (dstCol) {
          const s3 = i * 3;
          const d3 = vis * 3;
          dstCol[d3] = srcCol[s3];
          dstCol[d3 + 1] = srcCol[s3 + 1];
          dstCol[d3 + 2] = srcCol[s3 + 2];
        }
        vis++;
      }

      inst.count = vis;
      inst.visible = vis > 0;
      if (vis > 0) {
        const matAttr = inst.instanceMatrix;
        if (typeof matAttr.clearUpdateRanges === 'function') {
          matAttr.clearUpdateRanges();
          matAttr.addUpdateRange(0, vis * 16);
        }
        matAttr.needsUpdate = true;
        if (dstCol && inst.instanceColor) {
          const colAttr = inst.instanceColor;
          if (typeof colAttr.clearUpdateRanges === 'function') {
            colAttr.clearUpdateRanges();
            colAttr.addUpdateRange(0, vis * 3);
          }
          colAttr.needsUpdate = true;
        }
      }
    }
  }
}
