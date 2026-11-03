/**
 * ============================================================================
 * LAYER 1: DECLARATIVE PREMISE CARTRIDGE (DATA MANIFEST)
 * NEXUS-3D: KCIC Whoosh 350 km/h High-Speed Rail Simulator (v6.0 Next-Gen Engine)
 * Pure ES Module — Shared by Main Thread & Background Chunk Builder Worker
 * ============================================================================
 */
export const GAME_PREMISE_CARTRIDGE = {
  meta: {
    title: "NEXUS-3D: KCIC Whoosh 350 km/h High-Speed Rail Simulator",
    version: "6.0-WebGPU-Hybrid-Edition",
    metricScale: 1.0 // 1 unit = 1.0 meter
  },
  railway: {
    standardGauge: 1.435,
    doubleTrackOffset: 4.4,     // Distance between Mainline 1 and Mainline 2 centerlines
    jointInterval: 25.0,        // Continuous welded HSR slab/ballast rail cadence
    stationInterval: 3600,      // High-Speed Station every 3.6 km
    signalInterval: 450,        // ETCS/CTCS-3 Block Marker every 450 m
    crossingInterval: 1800,     // Overpass / Crossing interval
    chunkLength: 100,           // 100m streaming spline chunk
    chunksAhead: 15,            // 1.5 km forward visibility (worker-streamed + instanced)
    chunksBehind: 3             // 300 m rear visibility
  },
  biomes: [
    {
      id: "PRIANGAN",
      name: "Koridor HSR Jakarta–Bandung",
      skyTop: 0x0b2447,
      skyHorizon: 0x7dd3fc,
      fogColor: 0x93c5fd,
      groundColor: 0x2d5a27,
      cliffColor: 0x475569,
      waterLevel: -28.0,
      treeDensity: 1.0,
      speedLimit: 350
    },
    {
      id: "ALPINE",
      name: "Puncak Salju (Glacial HSR Pass)",
      skyTop: 0x09152b,
      skyHorizon: 0xcbd5e1,
      fogColor: 0xdbeafe,
      groundColor: 0xe2e8f0,
      cliffColor: 0x334155,
      waterLevel: -28.0,
      treeDensity: 0.75,
      speedLimit: 350
    },
    {
      id: "COASTAL",
      name: "Pesisir Samudra (Ocean Viaduct)",
      skyTop: 0x0369a1,
      skyHorizon: 0xbae6fd,
      fogColor: 0xbae6fd,
      groundColor: 0xd97706,
      cliffColor: 0x57534e,
      waterLevel: -2.4,
      treeDensity: 0.45,
      speedLimit: 350
    },
    {
      id: "CANYON",
      name: "Ngarai Terowongan (Redstone HSR)",
      skyTop: 0x431407,
      skyHorizon: 0xfdba74,
      fogColor: 0xfed7aa,
      groundColor: 0xb45309,
      cliffColor: 0x7c2d12,
      waterLevel: -28.0,
      treeDensity: 0.25,
      speedLimit: 350
    },
    {
      id: "METROPOLIS",
      name: "Distrik Halim Metropolis (HSR)",
      skyTop: 0x060a14,
      skyHorizon: 0x1e293b,
      fogColor: 0x1e293b,
      groundColor: 0x1e293b,
      cliffColor: 0x0f172a,
      waterLevel: -28.0,
      treeDensity: 0.3,
      speedLimit: 350
    }
  ],
  stations: [
    "STASIUN HALIM (KCIC)",
    "STASIUN KARAWANG (KCIC)",
    "STASIUN PADALARANG (KCIC)",
    "STASIUN TEGALLUAR SUMMARECON",
    "STASIUN CIREBON HSR",
    "STASIUN PEKALONGAN HSR",
    "STASIUN SEMARANG TAWANG HSR",
    "STASIUN SOLO BALAPAN HSR",
    "STASIUN MADIUN RAYA HSR",
    "STASIUN KERTOSONO HSR",
    "STASIUN MOJOKERTO HSR",
    "STASIUN SURABAYA PASAR TURI HSR"
  ],
  liveries: {
    WHOOSH: {
      name: "KCIC WHOOSH CR400AF",
      consistType: "EMU",
      tractionType: "ELECTRIC",
      maxSpeedKmh: 350,
      maxTractionKn: 520,
      powerMw: 10.4,
      motoredAxleRatio: 0.50,
      locoMassTons: 74,
      carMassTons: 38,
      bodyColor: 0xf1f5f9,
      stripe1: 0xdc2626,
      stripe2: 0x0f172a,
      noseType: "BULLET"
    },
    WHOOSH_CIT: {
      name: "KCIC CIT DOCTOR YELLOW",
      consistType: "EMU",
      tractionType: "ELECTRIC",
      maxSpeedKmh: 350,
      maxTractionKn: 520,
      powerMw: 10.4,
      motoredAxleRatio: 0.50,
      locoMassTons: 74,
      carMassTons: 38,
      bodyColor: 0xfacc15,
      stripe1: 0x1d4ed8,
      stripe2: 0x0f172a,
      noseType: "BULLET"
    },
    NUSANTARA: {
      name: "KA ARGO PARAHYANGAN (CC206)",
      consistType: "KAI_PASSENGER",
      tractionType: "DIESEL",
      maxSpeedKmh: 150,
      maxTractionKn: 290,
      powerMw: 2.25,
      motoredAxleRatio: 1.0,
      locoMassTons: 90,
      carMassTons: 42,
      bodyColor: 0xf8fafc,
      stripe1: 0x1d4ed8,
      stripe2: 0xf97316,
      noseType: "HOOD"
    },
    KAI_STAINLESS: {
      name: "KA GAJAYANA STAINLESS NEW GEN",
      consistType: "KAI_STAINLESS",
      tractionType: "DIESEL",
      maxSpeedKmh: 150,
      maxTractionKn: 295,
      powerMw: 2.35,
      motoredAxleRatio: 1.0,
      locoMassTons: 90,
      carMassTons: 44,
      bodyColor: 0xcbd5e1,
      stripe1: 0x0284c7,
      stripe2: 0xf59e0b,
      noseType: "HOOD"
    },
    FREIGHT: {
      name: "KA KARGO PETI KEMAS (KALOG)",
      consistType: "FREIGHT_CONTAINER",
      tractionType: "DIESEL",
      maxSpeedKmh: 120,
      maxTractionKn: 390,
      powerMw: 3.60,
      motoredAxleRatio: 1.0,
      locoMassTons: 126,
      carMassTons: 64,
      bodyColor: 0xf8fafc,
      stripe1: 0x1d4ed8,
      stripe2: 0xf97316,
      noseType: "HOOD"
    },
    TANKER: {
      name: "KA KETEL BBM PERTAMINA",
      consistType: "FREIGHT_TANKER",
      tractionType: "DIESEL",
      maxSpeedKmh: 110,
      maxTractionKn: 390,
      powerMw: 3.60,
      motoredAxleRatio: 1.0,
      locoMassTons: 126,
      carMassTons: 68,
      bodyColor: 0xf8fafc,
      stripe1: 0x16a34a,
      stripe2: 0xdc2626,
      noseType: "HOOD"
    },
    COAL_FREIGHT: {
      name: "KA BABARANJANG BATUBARA",
      consistType: "FREIGHT_COAL",
      tractionType: "DIESEL",
      maxSpeedKmh: 110,
      maxTractionKn: 560,
      powerMw: 4.50,
      motoredAxleRatio: 1.0,
      locoMassTons: 180,
      carMassTons: 74,
      bodyColor: 0xdc2626,
      stripe1: 0x0f172a,
      stripe2: 0xfacc15,
      noseType: "HOOD"
    },
    KRL_COMMUTER: {
      name: "KRL COMMUTER LINE",
      consistType: "EMU",
      tractionType: "ELECTRIC",
      maxSpeedKmh: 130,
      maxTractionKn: 340,
      powerMw: 3.20,
      motoredAxleRatio: 0.50,
      locoMassTons: 68,
      carMassTons: 36,
      bodyColor: 0xf1f5f9,
      stripe1: 0xdc2626,
      stripe2: 0xfacc15,
      noseType: "BULLET"
    },
    GLACIER: {
      name: "ALPINE PANORAMA EXPRESS",
      consistType: "EMU",
      tractionType: "ELECTRIC",
      maxSpeedKmh: 200,
      maxTractionKn: 380,
      powerMw: 5.60,
      motoredAxleRatio: 0.55,
      locoMassTons: 84,
      carMassTons: 40,
      bodyColor: 0xdc2626,
      stripe1: 0xf8fafc,
      stripe2: 0x1e293b,
      noseType: "BULLET"
    }
  }
};
