/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./world-gis/MasterPlanConfig.js
 *
 * Single Source of Spatial Truth (MasterPlanSSoST) for Kompleks Bundaran Gedung DPPD,
 * ITTC Perpustakaan Lama, STP-UI, & ILRC/DRPM Universitas Indonesia, Depok.
 *
 * Geodetically calibrated against:
 * - Satellite Nadir 90° (ref_174.jpg, ref_206.jpg, ref_207.jpg)
 * - Street View GPS Stations around Jl. Prof. Dr. Nugroho Notosusanto (151.0m road centerline diameter)
 *
 * World Coordinate Convention (1.0 unit = 1.0 meter):
 *   +X = East (0°), +Z = South (+90°), -X = West (±180°), -Z = North (-90°), +Y = Up
 */
const DEG = Math.PI / 180;

export const UI_DPPD_MASTER_PLAN = {
  id: 'UI_DEPOK_BUNDARAN_DPPD_STP_ILRC_V2',
  name: 'Kompleks Bundaran Gedung DPPD (eks-KSDP), ITTC, STP-UI & ILRC Universitas Indonesia',
  gpsOrigin: {
    lat: -6.365429,
    lon: 106.826801,
    elevationMdpl: 73.0
  },
  envelope: {
    width: 176.0,
    length: 176.0,
    height: 27.2,
    foundationSkirtDepth: 18.0
  },
  ringRoad: {
    rDitchInner: 68.55,
    rDitchOuter: 69.45,
    ditchFloorY: -0.68,
    rInnerVerge: 71.15,
    rInnerCurb: 71.50,
    rAsphaltIn: 71.50,
    rCenterDash: 75.50,
    rAsphaltOut: 79.50,
    rOuterCurb: 79.85,
    rPlinthRim: 88.00,
    trafficFlow: 'counter-clockwise'
  },
  sectors: {
    // 1. ITTC Perpustakaan Lama — Cardinal N-S / E-W Square at Center
    ittc: {
      centerX: -3.0,
      centerZ: 1.5,
      outerHalf: 21.0,      // 42.0m x 42.0m footprint
      courtyardHalf: 8.0,   // 16.0m x 16.0m courtyard
      skylightHalf: 4.8,    // 9.6m x 9.6m nested skylight roof cap (ref_174)
      floors: 4,
      eaveY: 14.20,
      ridgeY: 18.80,
      annexNW: { x: -22.5, z: -21.0, w: 13.0, d: 7.5, h: 4.2 }
    },

    // 2. Gedung DPPD (eks-KSDP) — SW Quadrant (+92° South to +178° West, Main Axis +135.5° SW)
    dppd: {
      thStart: 92.0 * DEG,
      thEnd: 178.0 * DEG,
      thAxis: 135.5 * DEG,
      rInner: 39.5,
      rOuter: 57.5,
      rLoggiaWall: 54.8,
      rBermToe: 66.8,
      l1TopY: 3.72,
      eaveY: 6.45,
      wingRidgeY: 11.85,
      gableHalfWidth: 11.2,
      gablePeakY: 16.25
    },

    // 3. Gedung STP-UI (Science Techno Park) — North Arc (-150° WNW across -90° North to -34° ENE)
    stp: {
      thStart: -150.0 * DEG,
      thEnd: -34.0 * DEG,
      thPortico: -88.0 * DEG,
      rInner: 39.0,
      rOuter: 57.8,
      floors: 8,
      l1TopY: 4.30,
      l3TopY: 11.10,
      roofSlabY: 25.40,
      crownTopY: 26.80
    },

    // 4. Gedung ILRC & DRPM UI — East-to-SSE Arc (-31° ENE across 0° East & +41° Atrium to +83° SSE Brick Tower)
    ilrc: {
      thStart: -31.0 * DEG,
      thEastEnd: 36.0 * DEG,
      thAtriumStart: 36.0 * DEG,
      thAtriumEnd: 46.0 * DEG,
      thSouthEnd: 75.0 * DEG,
      thTowerEnd: 83.0 * DEG,
      rInnerEast: 37.5,
      rInnerSouth: 41.5,
      rOuter: 58.0,
      corniceTopY: 21.20,
      towerTopY: 25.20,
      penthouseTopY: 27.20
    }
  }
};

/**
 * Helper to create polar camera preset at radius rCam, angleDeg, height yCam looking at (rTgt, tgtDeg, yTgt).
 */
function polarPreset(label, sector, rCam, camDeg, yCam, rTgt, tgtDeg, yTgt, fov = 56) {
  const cRad = camDeg * DEG;
  const tRad = tgtDeg * DEG;
  return {
    label,
    sector,
    fov,
    pos: [+(rCam * Math.cos(cRad)).toFixed(2), yCam, +(rCam * Math.sin(cRad)).toFixed(2)],
    target: [+(rTgt * Math.cos(tRad)).toFixed(2), yTgt, +(rTgt * Math.sin(tRad)).toFixed(2)]
  };
}

export const BUNDARAN_CAMERA_PRESETS = {
  // Mandatory Tri-Angle Validation Presets (Gate 4)
  elevasi_horisontal: polarPreset('Elevasi 0° — Fasad Utama DPPD', 'tri', 82.0, 135.5, 3.6, 54.0, 135.5, 6.8, 54),
  oblique_45: {
    label: 'Oblique 45° — Ikhtisar Bundaran',
    sector: 'tri',
    fov: 48,
    pos: [-112.0, 96.0, 112.0],
    target: [0.0, 6.0, 0.0]
  },
  nadir_zenith_90: {
    label: 'Nadir 90° — Satelit Orthografik',
    sector: 'tri',
    isNadir: true,
    altitude: 198.0,
    pos: [0.0, 198.0, 0.45],
    target: [0.0, 0.0, 0.0]
  },

  // Sector Overviews
  overview_dppd: polarPreset('Gedung DPPD (Barat Daya)', 'dppd', 84.0, 135.5, 6.5, 52.0, 135.5, 7.2, 54),
  overview_ittc: {
    label: 'ITTC Perpustakaan Lama (Tengah)',
    sector: 'ittc',
    fov: 52,
    pos: [-66.0, 26.0, 8.0],
    target: [-3.0, 8.5, 1.5]
  },
  overview_stp: polarPreset('Gedung STP-UI (Utara)', 'stp', 88.0, -86.0, 11.0, 52.0, -88.0, 13.5, 58),
  overview_ilrc: polarPreset('Gedung ILRC & Atrium (Tenggara)', 'ilrc', 86.0, 41.0, 9.0, 52.0, 41.0, 10.5, 56),

  // Calibrated Street View Node Stations (Clockwise / Counter-Clockwise Ring Road Tour)
  sv_dppd_front: polarPreset('SV-01 • Depan Pelana Utama DPPD (ref_084)', 'dppd', 76.5, 135.5, 2.3, 55.0, 135.5, 7.2, 62),
  sv_dppd_totem: polarPreset('SV-02 • Kanopi L1, Ramp & Totem Huawei (ref_184)', 'dppd', 73.5, 142.0, 2.2, 57.0, 136.5, 4.5, 58),
  sv_dppd_se_wing: polarPreset('SV-03 • Sayap Kanan SE & 3 Palem Raja (ref_078)', 'dppd', 76.5, 112.0, 2.3, 55.0, 118.0, 6.0, 60),
  sv_dppd_nw_gable: polarPreset('SV-04 • Dinding Pelana Bata NW & Akses Barat (ref_144)', 'dppd', 76.5, 174.0, 2.4, 54.0, 168.0, 6.2, 58),
  sv_west_ittc: {
    label: 'SV-05 • Pintu Masuk Barat → ITTC Perpustakaan Lama (ref_203)',
    sector: 'ittc',
    fov: 62,
    pos: [-76.8, 2.3, -2.8],
    target: [-24.0, 6.8, -0.5]
  },
  sv_stp_sw_core: polarPreset('SV-06 • Sudut SW Tembaga & Tangga STP-UI (ref_148)', 'stp', 77.5, -146.0, 2.6, 54.0, -135.0, 13.0, 64),
  sv_stp_portico: polarPreset('SV-07 • Portico Utara & Planter STP-UI (ref_164)', 'stp', 78.5, -90.5, 2.2, 56.0, -89.0, 7.8, 68),
  sv_stp_ne_makara: polarPreset('SV-08 • Puncak Makara Emas & Sirip NE STP-UI (ref_111)', 'stp', 77.5, -40.0, 2.6, 54.0, -52.0, 15.0, 68),
  sv_ilrc_east_box: polarPreset('SV-09 • Sayap Louver Timur & Kotak Kaca ILRC (ref_009)', 'ilrc', 77.0, 6.0, 2.4, 55.0, 8.0, 9.8, 60),
  sv_ilrc_drpm_gate: polarPreset('SV-10 • Gerbang Parkir Pimpinan & DRPM (ref_012)', 'ilrc', 77.0, 28.0, 2.3, 55.0, 30.0, 7.5, 60),
  sv_ilrc_atrium: polarPreset('SV-11 • Atrium Terbuka 5 Lantai & Portal ILRC (ref_029)', 'ilrc', 76.5, 41.0, 2.4, 52.0, 41.0, 10.5, 62),
  sv_ilrc_mound: polarPreset('SV-12 • Sayap Selatan & Gundukan Bambu ILRC (ref_033)', 'ilrc', 76.5, 60.0, 2.3, 55.0, 58.0, 8.5, 60),
  sv_ilrc_brick_tower: polarPreset('SV-13 • Menara Bata 6 Lantai & 3 Jendela Tudung (ref_076)', 'ilrc', 76.8, 86.0, 2.4, 54.0, 79.0, 12.5, 60)
};
