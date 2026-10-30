/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.1 DJI Mavic Photogrammetric Calibration)
 * ./library/landmark/Gedung_Rektorat_UI/AssetMetaAndEnvelope.js
 *
 * TAHAP 1 (Gate 1): Kalibrasi Amplop Ortografis Fotogrametri 5 Foto DJI Mavic,
 * Volume Kolisi Potensial Lunak, Preset Kamera Tri-Sudut Mavic, & Stasiun Profil SSoST.
 *
 * Referensi Arsitektural & Fotogrametri:
 * Gedung Pusat Administrasi Universitas Indonesia (Rektorat UI — Kampus Depok)
 * Arsitek: Prof. Ir. Gunawan Tjahjono, M.Arch., Ph.D. (1984–1987)
 */

export const GEDUNG_REKTORAT_UI_META = {
  id: 'GEDUNG_REKTORAT_UI_DEPOK',
  name: 'Gedung Pusat Administrasi Universitas Indonesia (Rektorat UI Depok — Mavic Digital Twin)',
  category: 'LANDMARK',
  kernelType: 'HYBRID_SSOST',
  scaleUnit: 'METER',
  realDimensionsMeters: {
    width: 108.0, // [-54.0m West Annex Eave .. +54.0m East Lakeside Annex Eave]
    height: 50.0, // [0.0m Ground Plaza .. +50.0m Steep Tajug Spire Finial]
    length: 124.0 // [-52.0m North Pavilion Eave .. +72.0m Front Rotunda Gatehouse Eave]
  },
  foundationSkirtDepth: 18.0,

  // Soft Potential Field Collision Avoidance Volumes (Hukum Emas 15)
  obstacleColliders: [
    // 0. Base Courtyard Plinth (skipped for pedestrian walk mode)
    { type: 'BOX', minX: -38.0, maxX: 38.0, minY: -2.0, maxY: 0.25, minZ: -35.0, maxZ: 39.0 },
    // 1. Main Cruciform 6-Eave Pagoda Tower (36.8 x 36.8 m)
    { type: 'BOX', minX: -18.6, maxX: 18.6, minY: 0.0, maxY: 33.2, minZ: -18.6, maxZ: 18.6 },
    // 2. Monumental Steep Top Pyramidal Tajug Spire
    { type: 'CYL', x: 0.0, z: 0.0, r: 13.5, minY: 32.5, maxY: 50.2 },
    // 3. Front High-Gabled Breezeway (Selasar Agung)
    { type: 'BOX', minX: -6.8, maxX: 1.8, minY: 0.0, maxY: 9.6, minZ: 15.6, maxZ: 39.4 },
    // 4. Front Cloister Galleries & Rotunda Gatehouse (leaving inner flag court walkable/viewable from above)
    { type: 'BOX', minX: -20.5, maxX: 15.5, minY: 0.0, maxY: 5.9, minZ: 58.2, maxZ: 63.6 },
    { type: 'BOX', minX: -9.0, maxX: 4.0, minY: 0.0, maxY: 7.6, minZ: 59.0, maxZ: 72.0 },
    // 5. West & East (Lakeside) Courtyard Annex Wings
    { type: 'BOX', minX: -54.0, maxX: -20.5, minY: 0.0, maxY: 12.4, minZ: -19.5, maxZ: 16.5 },
    { type: 'BOX', minX: 20.5, maxX: 54.0, minY: 0.0, maxY: 12.4, minZ: -22.5, maxZ: 13.5 },
    // 6. 45-Deg SE Octagonal Skylit Pendopo & North Pavilion
    { type: 'CYL', x: 29.5, z: 29.5, r: 8.5, minY: 0.0, maxY: 8.4 },
    { type: 'BOX', minX: -3.5, maxX: 13.5, minY: 0.0, maxY: 8.8, minZ: -52.0, maxZ: -35.0 }
  ],

  // Preset Kamera Validasi Ortogonal Tri-Sudut (Terkalibrasi Langsung ke 5 Foto DJI Mavic!)
  orthoValidationPresets: {
    // Matches DJI Mavic Photo #1 (Hovering over Lapangan Rotunda UI facing Front Cloister & Tower)
    elevationFront: {
      pos: { x: -2.5, y: 27.5, z: 98.0 },
      target: { x: -1.2, y: 18.0, z: 4.0 }
    },
    // Matches DJI Mavic Photo #2 (High-Oblique 45° showing 4 Corner Wings, Makara Crown, & ◇◇◇ Plaza)
    oblique45: {
      pos: { x: -64.0, y: 54.0, z: 66.0 },
      target: { x: 0.0, y: 18.0, z: -2.0 }
    },
    // Matches DJI Mavic Photos #3 & #5 (Nadir 90° Horizon-Locked Orthophoto)
    nadir90: {
      targetX: 2.0,
      targetZ: 10.0,
      heightY: 178.0,
      topVecX: 0.0,
      topVecZ: -1.0
    },
    // Full Upright Aerial Overview of Kepulauan UI: Pulau "U" (Hills + Flat Rectorat) + Selat UI + Pulau "I" (Volcano + Runway)
    archipelagoUI: {
      pos: { x: -785.0, y: 760.0, z: 342.0 },
      target: { x: -155.0, y: 6.0, z: 342.0 }
    },
    // Full Upright Horseshoe Overview of Pulau "U" (Flat Rektorat Campus Head + Rolling Green Peninsula Hills & Teluk U)
    island360: {
      pos: { x: -585.0, y: 465.0, z: 185.0 },
      target: { x: -148.0, y: 6.0, z: 185.0 }
    },
    // Aerial Overview of Pulau "I": Straight ICAO Runway RWY 36/18, West Apron, & Active Stratovolcano
    pulauI: {
      pos: { x: -535.0, y: 215.0, z: 688.0 },
      target: { x: -135.0, y: 22.0, z: 700.0 }
    },
    // Final-Approach Aviation View Down Pulau I Straight Runway RWY 36 Toward Gunung Api Pulau I
    runwayPulauI: {
      pos: { x: -418.0, y: 21.0, z: 708.0 },
      target: { x: -155.0, y: 18.0, z: 704.0 }
    },
    // Summit & Crater Rim Overlook of Gunung Gede Bogor (Kawah Ratu, Alun-Alun Suryakencana, & Thin Fumarole Steam Wisp)
    volcanoCrater: {
      pos: { x: -124.0, y: 98.0, z: 632.0 },
      target: { x: -32.0, y: 47.0, z: 698.0 }
    },
    // Close Architectural & Aviation View of the South-West Waterfront VIP Helipad & Windsock
    helipadVIP: {
      pos: { x: -132.0, y: 17.5, z: 132.0 },
      target: { x: -96.0, y: 2.4, z: 102.0 }
    },
    // Overlook of Pulau U Dense Southern Highland Rainforest, Mata Air, 4-Bend Meandering River, & Danau Kecil Rektorat
    hutanSungaiU: {
      pos: { x: -348.0, y: 86.0, z: -38.0 },
      target: { x: -182.0, y: 5.5, z: 14.0 }
    },
    // Front-Oblique Amphitheater View of Pulau I Horseshoe Mini-Niagara Waterfall, Rising Mist, & Sparse Montane Forest
    niagaraPulauI: {
      pos: { x: -46.0, y: 26.5, z: 598.0 },
      target: { x: -43.0, y: 12.5, z: 654.0 }
    },
    // Close-Up Ground-Level Vignette of Campus Cats ("Kucing UI": Oyen, Tuxedo, Calico Loaf & Kitten) in front of Rektorat
    kucingUI: {
      pos: { x: -8.6, y: 0.92, z: 26.6 },
      target: { x: -6.3, y: 0.34, z: 23.4 }
    },
    // Street-Level Campus View of the Patrolling 6-Seater Electric Campus Golf Cart ("Mobil Golf Listrik UI")
    mobilGolfUI: {
      pos: { x: -12.5, y: 3.15, z: 83.5 },
      target: { x: -2.5, y: 1.08, z: 76.3 }
    },
    // Scenic Coastal & Horizon Vista Overlooking Distant Nusantara Fishing Boats, Small Ships, & Bortle 1–2 Sky Dome
    perahuLautUI: {
      pos: { x: 54.0, y: 24.0, z: 18.0 },
      target: { x: 460.0, y: 4.0, z: 380.0 }
    },
    // Hero Architectural & Maritime Clearance Vista of Jembatan Teksas UI (Pilon Lingga 25m, 7-Segment BTKC Arch,
    // 30 Blue Wave Sunroofs, Pilon Yoni 15m, & 44-ft Flybridge Mini Yacht cruising underneath across Teluk U!)
    jembatanTeksasUI: {
      pos: { x: -188.0, y: 29.5, z: 232.0 },
      target: { x: -26.0, y: 9.2, z: 245.65 }
    },
    // Southern Subsea Underpass West Open-Cut Ramp, UI Golden Nusantara Portal Arch, Twin Tajug Ventilation Towers, & Bus Kuning UI!
    underpassPortalSelatan: {
      pos: { x: -357.2, y: 7.6, z: 256.0 },
      target: { x: -355.6, y: -4.6, z: 338.0 }
    },
    // Inside the -8.20m Subsea Immersed-Tube Underpass Bore beneath Selat UI (3800K LED Luminaires & Panoramic Ocean-Caustic Windows!)
    underpassBawahLautInterior: {
      pos: { x: -353.35, y: -6.22, z: 434.0 },
      target: { x: -355.85, y: -6.42, z: 504.0 }
    },
    // Northern Ferry Crossing Terminals ("Dermaga Penyeberangan Utara Pulau U & Pulau I") & Shuttling Electric Catamaran Ferry!
    dermagaPenyeberanganUtara: {
      pos: { x: -10.0, y: 23.5, z: 452.0 },
      target: { x: -74.0, y: 1.4, z: 545.0 }
    },
    // Elevated Scenic Overlook of Pulau U Integrated Multi-Loop Hillside Road Network, Underpass Selatan, & Dermaga Utara!
    jaringanJalanPulauU: {
      pos: { x: -565.0, y: 470.0, z: 348.0 },
      target: { x: -175.0, y: 0.0, z: 348.0 }
    }
  }
};

/**
 * Kuadran 4 Sayap Sudut Menara Utama ("4 Corner Pagoda Wings")
 */
export const CORNER_QUADRANTS = [
  { id: 'NE', sx: 1, sz: -1 },
  { id: 'NW', sx: -1, sz: -1 },
  { id: 'SW', sx: -1, sz: 1 },
  { id: 'SE', sx: 1, sz: 1 }
];

/**
 * Spesifikasi 4 Atap Limas Piramidal Besar di Puncak Ke-4 Sayap Sudut (Tier 7 — Mavic Photos #1, #2, #3, #5)
 * Berpusat di (±10.30, ±10.30) sehingga bentang teritisan luar mencapai ±18.55m dan tepi dalam ±2.05m
 * (menyisakan celah atrium kaca gelap tengah [-2.05, +2.05] serta membingkai Dinding Putih Makara UI!).
 */
export const CORNER_TOWER_CROWN_SPEC = {
  centerXY: 10.3,
  eaveBaseY: 25.15,
  eaveFasciaY: 25.35,
  eaveHalf: 8.25,     // 10.30 + 8.25 = 18.55m outer overhang, 10.30 - 8.25 = 2.05m inner atrium edge
  ridgeCapHalf: 1.35, // Crisp flat horizontal ridge cap at y = 30.85m (Mavic Photo #1 & #2)
  roofTopY: 30.85
};

/**
 * Membangun tabel stasiun aksial SSoST Menara Utama [v in 0..1] secara deterministik.
 * Kunci Geometri Mavic:
 * - Di sepanjang Lantai 2..7 (6 tingkat teritisan pagoda), `coreHalf = 13.10m` dan `xIn = 2.30m` DIKUNCI KONSTAN
 *   sehingga ceruk vertikal kaca gelap tengah (lebar 4.60m, mundur 2.75m) naik lurus menerus tanpa terputus!
 * - Sementara `xOut` pada ke-4 sayap sudut melebar 6 kali ke `17.92m` membentuk 6 sabuk teritisan genteng merah!
 */
function buildMasterProfileStations() {
  const raw = [];
  const pushWithXOut = (y, coreHalf, coreChamfer, towerActive, xIn, xOut, towerChamfer, zone, curveExp = 1.0) => {
    const towerCenter = 0.5 * (xIn + xOut);
    const towerHalf = Math.max(0.1, 0.5 * (xOut - xIn));
    raw.push({
      y,
      coreHalf,
      coreChamfer,
      towerActive,
      towerCenter,
      towerHalf,
      towerChamfer,
      zone,
      curveExp
    });
  };

  // --- ZONA 0: SUB-TERRAIN SKIRT (-18.00m s.d. 0.00m) & PLINTH MENARA ---
  pushWithXOut(-18.0, 0.001, 0.0, 0.0, 2.3, 15.85, 0.22, 'BOTTOM_CAP');
  pushWithXOut(-18.0, 17.2, 0.35, 0.0, 2.3, 15.85, 0.22, 'FOUNDATION_SKIRT');
  pushWithXOut(0.0, 17.2, 0.35, 0.0, 2.3, 15.85, 0.22, 'TOWER_BASE_STEP1');
  pushWithXOut(0.36, 16.4, 0.3, 0.0, 2.3, 15.85, 0.22, 'TOWER_BASE_DECK');

  // --- ZONA 1: LANTAI DASAR KOLONNADE PUTIH TINGGI (2-Story Lobby Base, y = 0.36m .. 5.65m) ---
  // Recessed dark lobby glazing behind tall white ground-floor columns (placed at ±15.55m in Stage 5)
  pushWithXOut(0.42, 13.1, 0.25, 1.0, 2.3, 14.35, 0.22, 'LOBBY_GLASS');
  pushWithXOut(5.55, 13.1, 0.25, 1.0, 2.3, 14.35, 0.22, 'LOBBY_HEAD');
  pushWithXOut(5.65, 13.1, 0.25, 1.0, 2.3, 15.85, 0.22, 'EAVE_1_SOFFIT');

  // --- ZONA 2: 6 TINGKAT TERITISAN PAGODA GENTENG MERAH & PITA JENDELA 6-MODUL (y = 5.65m .. 25.15m) ---
  // Exactly 6 flared terracotta eave skirts + 6 white spandrel & dark window bands below the 4 Corner Pyramidal Roofs!
  for (let tier = 0; tier < 6; tier++) {
    const tNum = tier + 1;
    const y0 = 5.65 + tier * 3.25;
    const yDrip = y0 + 0.09;       // Underside soffit flares out from 15.85m to 17.92m
    const yFasciaTop = y0 + 0.21;  // Crisp eave fascia lip at 17.92m
    const ySlopeTop = y0 + 1.44;   // Sloped terracotta-red tiled roof returns from 17.92m -> 15.85m
    const ySpandrelTop = y0 + 2.04;// Crisp off-white concrete spandrel band at 15.85m
    const yWinBot = y0 + 2.11;     // Recessed 6-bay dark ribbon window sill at 15.64m
    const yWinTop = y0 + 3.17;     // Ribbon window head tucked right under next eave soffit
    const yNext = y0 + 3.25;

    pushWithXOut(yDrip, 13.1, 0.25, 1.0, 2.3, 17.92, 0.20, `EAVE_${tNum}_FASCIA`);
    pushWithXOut(yFasciaTop, 13.1, 0.25, 1.0, 2.3, 17.92, 0.20, `EAVE_${tNum}_SLOPE`, 1.08);
    pushWithXOut(ySlopeTop, 13.1, 0.25, 1.0, 2.3, 15.85, 0.22, `FLOOR_${tNum + 1}_SPANDREL`);
    pushWithXOut(ySpandrelTop, 13.1, 0.25, 1.0, 2.3, 15.85, 0.22, `FLOOR_${tNum + 1}_WIN_RECESS`);
    pushWithXOut(yWinBot, 13.1, 0.25, 1.0, 2.3, 15.64, 0.22, `FLOOR_${tNum + 1}_GLASS`);
    pushWithXOut(yWinTop, 13.1, 0.25, 1.0, 2.3, 15.64, 0.22, `FLOOR_${tNum + 1}_WIN_HEAD`);

    if (tier < 5) {
      pushWithXOut(yNext, 13.1, 0.25, 1.0, 2.3, 15.85, 0.22, `EAVE_${tNum + 1}_SOFFIT`);
    } else {
      // Transition at y = 25.15m into the Central White Makara Crown Drum (surrounded by the 4 Corner Pyramidal Roofs!)
      pushWithXOut(25.15, 13.1, 0.25, 1.0, 2.3, 15.85, 0.22, 'CROWN_DRUM_STEP');
    }
  }

  // --- ZONA 3: LEHER MAHKOTA PUTIH, PANEL MAKARA UI, & JENDELA CLERESTORY (y = 25.15m .. 32.85m) ---
  pushWithXOut(25.32, 10.65, 0.22, 0.0, 2.3, 10.65, 0.22, 'CROWN_LOWER_WALL');
  pushWithXOut(30.85, 10.65, 0.22, 0.0, 2.3, 10.65, 0.22, 'CROWN_CLERESTORY_BAND');
  pushWithXOut(32.78, 10.65, 0.22, 0.0, 2.3, 10.65, 0.22, 'TAJUG_MAIN_SOFFIT');

  // --- ZONA 4: ATAP PUNCAK LIMAS TAJUG PIRAMIDAL TUNGGAL CURAM (y = 32.85m .. 47.75m, ~48.7°!) ---
  pushWithXOut(32.88, 12.95, 0.08, 0.0, 2.3, 10.65, 0.08, 'TAJUG_MAIN_FASCIA');
  pushWithXOut(33.10, 12.95, 0.08, 0.0, 2.3, 10.65, 0.08, 'TAJUG_MAIN_SLOPE', 1.06);
  pushWithXOut(46.95, 0.84, 0.03, 0.0, 2.3, 10.65, 0.03, 'MUSTAKA_WHITE_CAP');

  // --- ZONA 5: TUDUNG PUNCAK PUTIH (WHITE APEX CAP) & TIANG PENANGKAL PETIR (y = 46.95m .. 50.00m) ---
  pushWithXOut(47.75, 0.09, 0.02, 0.0, 2.3, 10.65, 0.02, 'LIGHTNING_SPIRE');
  pushWithXOut(50.0, 0.001, 0.0, 0.0, 2.3, 10.65, 0.0, 'APEX_CLOSURE');

  const lastIdx = raw.length - 1;
  return raw.map((st, idx) => ({
    ...st,
    index: idx,
    v: idx / lastIdx
  }));
}

export const MASTER_PROFILE_STATIONS = buildMasterProfileStations();

export function getStationSegmentByZone(zoneName) {
  const idx = MASTER_PROFILE_STATIONS.findIndex((s) => s.zone === zoneName);
  if (idx < 0 || idx >= MASTER_PROFILE_STATIONS.length - 1) return null;
  return {
    vMin: MASTER_PROFILE_STATIONS[idx].v,
    vMax: MASTER_PROFILE_STATIONS[idx + 1].v,
    s0: MASTER_PROFILE_STATIONS[idx],
    s1: MASTER_PROFILE_STATIONS[idx + 1]
  };
}
