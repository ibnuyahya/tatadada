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
    { species: 'PALEM_RAJA', x: -36.0, z: 123.0, scale: 1.10, yaw: 0.95 },
    { species: 'PALEM_RAJA', x: 31.5, z: 88.0, scale: 1.08, yaw: 4.55 },
    { species: 'PALEM_RAJA', x: 34.5, z: 106.5, scale: 1.10, yaw: 4.82 },
    { species: 'PALEM_RAJA', x: 31.0, z: 124.5, scale: 1.07, yaw: 5.25 },
    { species: 'PALEM_RAJA', x: 51.0, z: 18.5, scale: 1.08, yaw: 1.52 },
    { species: 'PALEM_RAJA', x: 54.5, z: 29.0, scale: 1.10, yaw: 1.62 },
    { species: 'PALEM_RAJA', x: 58.5, z: 40.5, scale: 1.09, yaw: 1.58 },
    { species: 'PALEM_RAJA', x: 49.5, z: -32.0, scale: 1.06, yaw: 1.48 },
    { species: 'PALEM_RAJA', x: 47.0, z: -46.0, scale: 1.08, yaw: 1.42 },
    { species: 'PALEM_RAJA', x: -24.0, z: 75.5, scale: 1.06, yaw: 2.35 },
    { species: 'PALEM_RAJA', x: 19.0, z: 75.5, scale: 1.06, yaw: 3.92 }
  ];

  static isAllowedTreeSpawn(x, z, groundHeightFn = TerrainDEMLoader.sampleGroundHeight) {
    if (x > -55.8 && x < 55.2 && z > -37.2 && z < 41.2) return false;
    if (x > -21.8 && x < 16.8 && z >= 38.0 && z <= 73.5) return false;
    if (x > -6.0 && x < 51.0 && z > -54.5 && z <= -34.0) return false;
    if (Math.hypot(x - 29.5, z - 29.5) < 10.6) return false;
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
    for (let i = 0; i < 520 && pulauUExtraCount < 128; i++) {
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
    for (let i = 0; i < 120 && forestShrubs.length < 48; i++) {
      const u = fract(0.25 + i * 0.7548776662466927);
      const v = fract(0.75 + i * 0.5698402909980532);
      const sx = -362.0 + u * 212.0;
      const sz = -24.0 + v * 310.0;
      if (
        groundHeightFn(sx, sz) >= 2.6 &&
        !TerrainDEMLoader.isFreshwaterExclusion(sx, sz, 1.25) &&
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

    // 1. Main Organic Fluted Trunk + -3.60m Subterranean Root Skirt (26 radial x 28 axial = 1,456 tris)
    addLoftedLimb(26, 28, (t) => {
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

    // 2. Five Sculpted Surface Buttress Root Spurs anchoring trunk into soil (5 x 8 x 6)
    for (let r = 0; r < 5; r++) {
      const az = (r / 5) * Math.PI * 2.0 + 0.22;
      const dx = Math.cos(az);
      const dz = Math.sin(az);
      addLoftedLimb(
        8,
        6,
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

      addLoftedLimb(14, 11, (t) => {
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

        addLoftedLimb(8, 7, (t) => {
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
   * Builds a 10x High-Detail 6-harmonic botanical leaf-cluster foliage cloud geometry
   * (38 x 26 = 1,976 tris/cloud, ZERO cones!) with natural tropical AO & sunlit leaf-tip shading.
   */
  static _buildOrganicFoliageCloudGeometry() {
    const uSegs = 38;
    const vSegs = 26;
    const positions = [];
    const normals = [];
    const uvs = [];
    const colors = [];
    const indices = [];

    for (let iv = 0; iv <= vSegs; iv++) {
      const v = iv / vSegs;
      const phi = v * Math.PI;
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);

      for (let iu = 0; iu <= uSegs; iu++) {
        const u = iu / uSegs;
        const th = u * Math.PI * 2.0;

        const nx = -Math.cos(th) * sinPhi;
        const ny = cosPhi;
        const nz = Math.sin(th) * sinPhi;

        // 6-Harmonic Multi-Scale Organic Leaf-Cluster Relief (macro bough lobes + micro leaf tufts!)
        const lobeA = 0.135 * Math.sin(3.0 * th + 0.5) * sinPhi * sinPhi;
        const lobeB = 0.085 * Math.cos(5.0 * th - 2.0 * phi);
        const lobeC = 0.065 * Math.sin(4.0 * phi + 2.0 * th);
        const tuftD = 0.042 * Math.sin(9.0 * th + 5.0 * phi) * sinPhi;
        const tuftE = 0.028 * Math.cos(14.0 * th - 7.0 * phi) * sinPhi;
        const tuftF = 0.018 * Math.sin(8.0 * phi - 6.0 * th);

        const bottomFlatten = ny < 0.0 ? 0.78 : 1.0;
        const r = 1.0 + lobeA + lobeB + lobeC + tuftD + tuftE + tuftF;

        const px = nx * r;
        const py = ny * r * bottomFlatten;
        const pz = nz * r;

        // Botanical AO, inner bough shadow, & sunlit young-shoot leaf tips
        const sunGrad = Math.max(0.0, Math.min(1.0, (ny + 0.85) / 1.85));
        const clumpVar = 0.5 + 0.5 * Math.sin(3.0 * th + phi * 2.0);
        const shootTip = Math.max(0.0, tuftD + tuftE) * 6.5 * sunGrad;

        const colR = 0.038 + 0.102 * sunGrad + 0.022 * clumpVar + 0.038 * shootTip;
        const colG = 0.128 + 0.238 * sunGrad + 0.035 * clumpVar + 0.068 * shootTip;
        const colB = 0.026 + 0.054 * sunGrad + 0.012 * clumpVar + 0.014 * shootTip;

        positions.push(px, py, pz);
        normals.push(nx, ny, nz);
        uvs.push(u, 1.0 - v);
        colors.push(colR, colG, colB);
      }
    }

    const stride = uSegs + 1;
    for (let iv = 0; iv < vSegs; iv++) {
      for (let iu = 0; iu < uSegs; iu++) {
        const a = iv * stride + iu;
        const b = a + 1;
        const c = a + stride + 1;
        const d = a + stride;
        indices.push(a, d, b, b, d, c);
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
   * Builds a 10x High-Detail UI Depok Campus Pedestrian Garden Lamp (~1,360 tris/lamp):
   * - 28-segment fluted pedestal & tapered steel shaft with torus moldings
   * - Warm amber-ivory lantern glass diffuser + 4 physical 3D corner bronze lantern cage struts
   * - 2-tier Tajug-inspired tropical rain-hood & spire finial!
   */
  static _buildCampusGardenLampGeometry() {
    const radialSegs = 28;
    const stations = [
      { y: -1.25, r: 0.15, flute: 0.0, col: [0.14, 0.15, 0.16] },
      { y: 0.0, r: 0.18, flute: 0.0, col: [0.17, 0.18, 0.20] },
      { y: 0.08, r: 0.19, flute: 0.0, col: [0.20, 0.21, 0.23] },
      { y: 0.14, r: 0.16, flute: 0.012, col: [0.18, 0.19, 0.21] },
      { y: 0.44, r: 0.145, flute: 0.012, col: [0.18, 0.19, 0.21] },
      { y: 0.49, r: 0.165, flute: 0.0, col: [0.22, 0.23, 0.25] }, // Molded pedestal torus ring
      { y: 0.54, r: 0.085, flute: 0.006, col: [0.16, 0.17, 0.19] },
      { y: 1.85, r: 0.068, flute: 0.005, col: [0.16, 0.17, 0.19] },
      { y: 1.90, r: 0.088, flute: 0.0, col: [0.22, 0.23, 0.25] }, // Mid-shaft decorative collar
      { y: 1.95, r: 0.065, flute: 0.004, col: [0.16, 0.17, 0.19] },
      { y: 3.18, r: 0.055, flute: 0.003, col: [0.16, 0.17, 0.19] },
      { y: 3.24, r: 0.11, flute: 0.0, col: [0.20, 0.21, 0.23] },
      { y: 3.29, r: 0.165, flute: 0.0, col: [0.21, 0.22, 0.24] }, // Lantern lower tray
      { y: 3.33, r: 0.135, flute: 0.0, col: [0.98, 0.87, 0.58] }, // Warm amber-ivory lantern glass bottom
      { y: 3.51, r: 0.148, flute: 0.0, col: [1.00, 0.90, 0.64] }, // Lantern glass waist
      { y: 3.69, r: 0.162, flute: 0.0, col: [1.00, 0.92, 0.68] }, // Lantern glass top
      { y: 3.72, r: 0.27, flute: 0.0, col: [0.15, 0.16, 0.18] },  // Lower Tajug rain-hood eave
      { y: 3.81, r: 0.14, flute: 0.0, col: [0.17, 0.18, 0.20] },  // Upper Tajug tier neck
      { y: 3.84, r: 0.18, flute: 0.0, col: [0.15, 0.16, 0.18] },  // Upper Tajug eave
      { y: 3.96, r: 0.015, flute: 0.0, col: [0.24, 0.22, 0.18] }  // Tajug finial spire
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
        uvs.push(u, is / (stations.length - 1));
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
    const canopyMatrices = [];

    const pos = new THREE.Vector3();
    const euler = new THREE.Euler();
    const quat = new THREE.Quaternion();
    const scl = new THREE.Vector3();
    const mat = new THREE.Matrix4();

    const pushTransform = (targetList, px, py, pz, rx, ry, rz, sx, sy, sz) => {
      pos.set(px, py, pz);
      euler.set(rx, ry, rz);
      quat.setFromEuler(euler);
      scl.set(sx, sy, sz);
      const m = new THREE.Matrix4().compose(pos, quat, scl);
      targetList.push(m);
    };

    for (let i = 0; i < validTrees.length; i++) {
      const tree = validTrees[i];
      const gy = groundHeightFn(tree.x, tree.z);
      const s = tree.scale || 1.0;
      const yaw = tree.yaw || 0.0;

      if (tree.species === 'TREMBESI') {
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
          1.32 * s
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
          4.55 * s
        );

        for (let k = 0; k < 6; k++) {
          const az = yaw + (k / 6) * Math.PI * 2.0 + (k % 2) * 0.14;
          const dist = (3.65 + (k % 3) * 0.35) * s;
          const hOff = (5.55 + (k % 2) * 0.55) * s;
          const rxPad = (3.05 + (k % 3) * 0.25) * s;
          const ryPad = (1.32 + (k % 2) * 0.18) * s;
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
            rxPad * 0.92
          );
        }
      } else if (tree.species === 'MAHONI') {
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
          0.94 * s
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
            2.45 * s
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
            2.25 * s
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
          2.05 * s
        );
      } else if (tree.species === 'CEMARA') {
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
          0.52 * s
        );
        for (let tier = 0; tier < 6; tier++) {
          const frac = tier / 5.0;
          const tierY = gy + (3.1 + tier * 1.95) * s;
          const tierR = 2.65 * (1.0 - frac * 0.72) * s;
          const tierH = 1.35 * (1.0 - frac * 0.35) * s;
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
            tierR
          );
        }
      } else {
        const leanAngle = 0.065;
        const leanX = Math.sin(yaw) * leanAngle;
        const leanZ = -Math.cos(yaw) * leanAngle;
        const trunkSy = 1.48 * s;

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
          0.35 * s
        );

        const palmTopH = 5.75 * trunkSy;
        const topVec = new THREE.Vector3(0.0, palmTopH, 0.0);
        euler.set(leanX, yaw, leanZ);
        quat.setFromEuler(euler);
        mat.compose(new THREE.Vector3(tree.x, gy, tree.z), quat, new THREE.Vector3(1, 1, 1));
        topVec.applyMatrix4(mat);

        pushTransform(
          canopyMatrices,
          topVec.x,
          topVec.y + 0.15 * s,
          topVec.z,
          leanX,
          yaw,
          leanZ,
          0.52 * s,
          1.15 * s,
          0.52 * s
        );

        for (let f = 0; f < 6; f++) {
          const frondYaw = yaw + (f / 6) * Math.PI * 2.0;
          const droopPitch = 0.42 + (f % 2) * 0.16;
          const radialOut = 1.65 * s;
          const fx = topVec.x + Math.sin(frondYaw) * radialOut;
          const fz = topVec.z + Math.cos(frondYaw) * radialOut;
          const fy = topVec.y + (0.28 - (f % 2) * 0.18) * s;

          pushTransform(
            canopyMatrices,
            fx,
            fy,
            fz,
            droopPitch,
            frondYaw,
            0.0,
            0.62 * s,
            0.26 * s,
            2.35 * s
          );
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
            0.36
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
            0.96
          );
        }
      }
    }

    // Lush Rainforest-Floor, Mata Air Rim, & Danau Kecil Rektorat Understory Shrubs/Ferns
    // (0 extra Draw Calls — shares KampusUITreeCanopiesInstanced!)
    for (let i = 0; i < forestShrubs.length; i++) {
      const sh = forestShrubs[i];
      const sgy = groundHeightFn(sh.x, sh.z);
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
        sh.sz
      );
    }

    // Draw Call #1: All Multi-Branch Organic Tree Trunks
    const trunkMat = new THREE.MeshStandardMaterial({
      name: 'CampusOrganicBarkMat',
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.92,
      metalness: 0.02,
      envMapIntensity: 0.09
    });
    injectOmniScaleSurfaceShader(trunkMat, {
      ySegmentStart: 120.0,
      segmentCount: 0.0,
      enableWindSway: true,
      surfaceProfile: SURFACE_PROFILE.ASPHALT_TEAK_BARK,
      bumpScale: 0.68,
      sharedUniforms: uniforms
    });

    const trunkGeo = AutogenBiomeSpawner._buildOrganicBranchingTrunkGeometry();
    const trunkInstanced = new THREE.InstancedMesh(trunkGeo, trunkMat, trunkMatrices.length);
    trunkInstanced.name = 'KampusUITreeTrunksInstanced';
    trunkInstanced.castShadow = true;
    trunkInstanced.receiveShadow = true;
    for (let i = 0; i < trunkMatrices.length; i++) {
      trunkInstanced.setMatrixAt(i, trunkMatrices[i]);
    }
    if (trunkInstanced.instanceMatrix) {
      trunkInstanced.instanceMatrix.needsUpdate = true;
    }

    // Draw Call #2: All Multi-Species Organic Foliage Canopy Clouds + Courtyard Planter Shrubs
    const canopyMat = new THREE.MeshStandardMaterial({
      name: 'CampusOrganicCanopyMat',
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.85,
      metalness: 0.02,
      envMapIntensity: 0.11
    });
    injectOmniScaleSurfaceShader(canopyMat, {
      ySegmentStart: 120.0,
      segmentCount: 0.0,
      enableWindSway: true,
      surfaceProfile: SURFACE_PROFILE.FOLIAGE_CANOPY,
      bumpScale: 0.30,
      sharedUniforms: uniforms
    });

    const canopyGeo = AutogenBiomeSpawner._buildOrganicFoliageCloudGeometry();
    const canopyInstanced = new THREE.InstancedMesh(canopyGeo, canopyMat, canopyMatrices.length);
    canopyInstanced.name = 'KampusUITreeCanopiesInstanced';
    canopyInstanced.castShadow = true;
    canopyInstanced.receiveShadow = true;
    for (let i = 0; i < canopyMatrices.length; i++) {
      canopyInstanced.setMatrixAt(i, canopyMatrices[i]);
    }
    if (canopyInstanced.instanceMatrix) {
      canopyInstanced.instanceMatrix.needsUpdate = true;
    }

    // Draw Call #3: 24 Fluted Nusantara Pedestrian Garden Post-Lamps with 4-Strut Lantern Cages
    const lampCoords = [];
    for (let i = 0; i < 12; i++) {
      const th = (i / 12) * Math.PI * 2.0 + Math.PI / 12;
      lampCoords.push({ x: -2.5 + Math.cos(th) * 34.9, z: 106.5 + Math.sin(th) * 34.9 });
    }
    lampCoords.push(
      { x: -11.5, z: 68.5 },
      { x: 6.5, z: 68.5 },
      { x: -21.2, z: 51.5 },
      { x: 16.2, z: 51.5 }
    );
    for (const lz of [-52.0, -37.0, -22.0, -8.0, 6.0, 20.0, 34.0, 48.0]) {
      lampCoords.push({ x: TerrainDEMLoader.getShorelinePromenadeX(lz) - 0.55, z: lz });
    }
    // 6 Nusantara Garden Post-Lamps lining the West Helipad Boulevard & Helipad Entry Plaza
    lampCoords.push(
      { x: -50.0, z: 100.8 },
      { x: -50.0, z: 112.2 },
      { x: -72.0, z: 100.8 },
      { x: -72.0, z: 112.2 },
      { x: -83.2, z: 99.6 },
      { x: -83.2, z: 113.4 }
    );

    const lampMat = new THREE.MeshStandardMaterial({
      name: 'CampusGardenLampMat',
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.55,
      metalness: 0.35,
      envMapIntensity: 0.14
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
      const ly = Math.max(0.0, groundHeightFn(lc.x, lc.z));
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
    biomeGroup.updateMatrixWorld(true);
    return biomeGroup;
  }
}
