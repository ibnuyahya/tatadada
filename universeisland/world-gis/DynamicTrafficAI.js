import * as THREE from 'three';
import { createSharedSurfaceUniforms } from '../core/ProceduralTextureFactory.js';
import { TerrainDEMLoader } from './TerrainDEMLoader.js';
import { createPhotorealHumanMaterial } from './HumanShaderMaterial.js';
import {
  HumanAnatomyBuilder,
  HUMAN_PALETTE,
  BIND_PIVOTS,
  JOINTS_PER_NPC,
  J_PELVIS,
  J_WAIST,
  J_CHEST,
  J_NECK,
  J_HEAD,
  J_SHOULDER_L,
  J_ELBOW_L,
  J_WRIST_L,
  J_SHOULDER_R,
  J_ELBOW_R,
  J_WRIST_R,
  J_HIP_L,
  J_KNEE_L,
  J_ANKLE_L,
  J_TOE_L,
  J_HIP_R,
  J_KNEE_R,
  J_ANKLE_R,
  J_TOE_R,
  J_SKIRT_HEM,
  J_PROP_MAIN,
  J_PROP_UMBRELLA
} from './HumanAnatomyBuilder.js';
import {
  FelineAnatomyBuilder,
  CAT_BIND_PIVOTS,
  CAT_JOINTS_PER_CAT,
  CJ_PELVIS,
  CJ_SPINE_MID,
  CJ_CHEST,
  CJ_NECK,
  CJ_HEAD,
  CJ_EAR_L,
  CJ_EAR_R,
  CJ_TAIL_BASE,
  CJ_TAIL_MID,
  CJ_TAIL_TIP,
  CJ_FRONT_SHOULDER_L,
  CJ_FRONT_PAW_L,
  CJ_FRONT_SHOULDER_R,
  CJ_FRONT_PAW_R,
  CJ_HIND_HIP_L,
  CJ_HIND_PAW_L,
  CJ_HIND_HIP_R,
  CJ_HIND_PAW_R
} from './FelineAnatomyBuilder.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.4
 * ./world-gis/DynamicTrafficAI.js
 *
 * 100x Hyper-Realistic Sivitas Akademika Universitas Indonesia — Seamless 22-Joint Dual-Bone LBS,
 * Photorealistic Skin SSS / Eye Catchlight / Batik Kawung Shader, Clinical 60/40 Asymmetric Gait,
 * Zero-Slide Foot-Ground Locking, Seated Park-Bench Posture, Social Walking Pairs, Helbing-Molnar
 * Personal-Space Avoidance, Player Eye-Contact Nod, & Live Rain/Umbrella Roof-Aware Adaptation!
 *
 * - 24 Distinct Sivitas Akademika Characters (24 x 22 = 528 Live Articulated Joints)
 * - ~200,000+ Sculpted Anatomical Triangles rendered in STRICTLY 1 SINGLE DRAW CALL!
 */

// 8-Keyframe Clinical 60% Stance / 40% Swing Human Gait Cycle:
// u in [0, 1): 0.00 Heel-Strike -> 0.12 Loading Shock-Absorption -> 0.30 Mid-Stance ->
//              0.48 Heel-Off (Metatarsal Toe-Roll!) -> 0.62 Toe-Off -> 0.78 Mid-Swing -> 0.92 Reach
const CLINICAL_GAIT_KEYS = [
  { u: 0.00, hipRx: -0.43, kneeRx: 0.07, footPitch: -0.22, toeBend: -0.05 }, // Initial Contact (Heel Strike!)
  { u: 0.12, hipRx: -0.27, kneeRx: 0.25, footPitch: 0.00, toeBend: 0.00 },   // Loading Response (Flat Foot + Knee Flex)
  { u: 0.30, hipRx: 0.03, kneeRx: 0.08, footPitch: 0.00, toeBend: 0.00 },    // Mid-Stance (Tall CoM Peak)
  { u: 0.48, hipRx: 0.34, kneeRx: 0.13, footPitch: 0.42, toeBend: -0.42 },   // Terminal Stance (Heel-Off, Toes Planted!)
  { u: 0.62, hipRx: 0.24, kneeRx: 0.80, footPitch: 0.30, toeBend: -0.14 },   // Pre-Swing (Toe-Off Propulsion)
  { u: 0.76, hipRx: -0.16, kneeRx: 1.04, footPitch: -0.08, toeBend: -0.06 }, // Initial/Mid-Swing (60-deg Knee Clearance!)
  { u: 0.90, hipRx: -0.44, kneeRx: 0.30, footPitch: -0.20, toeBend: -0.06 }, // Terminal Swing (Pendulum Shin Reach)
  { u: 1.00, hipRx: -0.43, kneeRx: 0.07, footPitch: -0.22, toeBend: -0.05 }  // Wrap to Heel-Strike
];

function wrapAngle(a) {
  let r = a;
  while (r > Math.PI) r -= Math.PI * 2.0;
  while (r < -Math.PI) r += Math.PI * 2.0;
  return r;
}

function dampAngle(curr, target, lambda, dt) {
  const diff = wrapAngle(target - curr);
  return wrapAngle(curr + diff * (1.0 - Math.exp(-lambda * dt)));
}

function dampScalar(curr, target, lambda, dt) {
  return curr + (target - curr) * (1.0 - Math.exp(-lambda * dt));
}

function smooth01(a, b, x) {
  const t = Math.max(0.0, Math.min(1.0, (x - a) / (b - a || 1e-6)));
  return t * t * (3.0 - 2.0 * t);
}

/**
 * Evaluates smooth cosine-Hermite interpolated clinical human leg angles at normalized gait cycle u in [0, 1).
 */
function sampleClinicalLegGait(uRaw, strideScale = 1.0) {
  const u = ((uRaw % 1.0) + 1.0) % 1.0;
  let kIdx = 0;
  for (let i = 0; i < CLINICAL_GAIT_KEYS.length - 1; i++) {
    if (u >= CLINICAL_GAIT_KEYS[i].u && u <= CLINICAL_GAIT_KEYS[i + 1].u) {
      kIdx = i;
      break;
    }
  }
  const kA = CLINICAL_GAIT_KEYS[kIdx];
  const kB = CLINICAL_GAIT_KEYS[kIdx + 1];
  const span = Math.max(1e-5, kB.u - kA.u);
  const lin = (u - kA.u) / span;
  const s = 0.5 - 0.5 * Math.cos(lin * Math.PI);

  const hipRx = (kA.hipRx + (kB.hipRx - kA.hipRx) * s) * strideScale;
  const kneeRx =
    0.05 + (kA.kneeRx - 0.05 + (kB.kneeRx - kA.kneeRx) * s) * Math.max(0.25, strideScale);
  const footPitch = (kA.footPitch + (kB.footPitch - kA.footPitch) * s) * strideScale;
  const toeRx = (kA.toeBend + (kB.toeBend - kA.toeBend) * s) * strideScale;

  return { hipRx, kneeRx, footPitch, toeRx };
}

export class DynamicTrafficAI {
  /**
   * @param {Function|Object} [optionsOrGroundFn]
   * @param {Object} [sharedSurfaceUniforms]
   */
  constructor(optionsOrGroundFn = {}, sharedSurfaceUniforms = null) {
    if (typeof optionsOrGroundFn === 'function') {
      this.groundHeightFn = optionsOrGroundFn;
      this.sharedUniforms = sharedSurfaceUniforms || createSharedSurfaceUniforms();
    } else {
      const opts = optionsOrGroundFn || {};
      this.groundHeightFn = opts.groundHeightFn || TerrainDEMLoader.sampleGroundHeight;
      this.sharedUniforms =
        opts.sharedSurfaceUniforms || sharedSurfaceUniforms || createSharedSurfaceUniforms();
    }

    this.group = new THREE.Group();
    this.group.name = 'KampusUISivitasAkademikaAI';

    this.npcs = [];
    this.cats = [];
    this.mesh = null;
    this.elapsedTime = 0.0;

    // Flat typed buffers for zero-GC single-draw-call 22-joint Dual-Bone Linear-Blend Skinning (LBS)
    this._restPos = null;
    this._restNorm = null;
    this._vertJ0 = null;
    this._vertJ1 = null;
    this._vertW0 = null;
    this._livePos = null;
    this._liveNorm = null;

    this._jointWorldMatrices = null; // Float32Array(N * 22 * 16)
    this._skinMatrices = null;       // Float32Array(N * 22 * 16): M_world * M_bind^-1
    this._skinNormalMatrices = null; // Float32Array(N * 22 * 9)

    // Reusable scratch math objects (0 heap allocations per frame)
    this._tmpPos = new THREE.Vector3();
    this._tmpEuler = new THREE.Euler();
    this._tmpQuat = new THREE.Quaternion();
    this._tmpScale = new THREE.Vector3(1, 1, 1);
    this._tmpLocalMat = new THREE.Matrix4();
    this._tmpParentMat = new THREE.Matrix4();
    this._tmpWorldMat = new THREE.Matrix4();
  }

  /**
   * Exact campus walkable surface elevation resolver (supports ◇◇◇ Courtyard, Flag Cloister,
   * Selasar Agung podium, Rotunda curbs, SE Covered Walkway, and DEM terrain).
   */
  _samplePedestrianGroundY(x, z) {
    if (x >= -19.5 && x <= 18.5 && z >= -19.5 && z <= 18.5) {
      return 0.37; // Rektorat Tower Colonnade plinth
    }
    if (x >= -36.5 && x <= 36.5 && z >= -35.5 && z <= 39.2) {
      return 0.135; // ◇◇◇ Diamond Courtyard Plaza & Selasar Agung
    }
    if (x >= -20.5 && x <= 15.5 && z > 39.2 && z <= 64.2) {
      return 0.142; // Flag Cloister Galleries & Inner Plaza
    }
    if (x >= -12.0 && x <= 7.0 && z > 64.2 && z <= 72.2) {
      return 0.130; // Front Gate Pendopo
    }
    // SE Covered Campus Walkway deck
    if (z >= 35.0 && z <= 128.5) {
      const tW = (z - 35.5) / (128.0 - 35.5);
      const wx = 34.2 + tW * (43.8 - 34.2);
      if (Math.abs(x - wx) < 1.95) return 0.055;
    }
    return Math.max(-0.15, this.groundHeightFn(x, z) + 0.032);
  }

  /**
   * Checks whether world coordinate (x, z) is sheltered beneath a real 3D architectural roof
   * (Selasar Agung, Flag Cloister Covered Perimeter Galleries, Tower Colonnade, 45-deg Breezeway,
   * Octagonal Skylit Pendopo, or the 120-seg SE Terracotta Covered Campus Walkway).
   */
  static _isShelteredByRoof(x, z) {
    // 1. Main Tower Colonnade & Annex Wings
    if (Math.abs(x) <= 18.4 && Math.abs(z) <= 18.4) return true;
    // 2. Selasar Agung Gable Roof
    if (x >= -5.9 && x <= 0.7 && z >= 15.5 && z <= 39.4) return true;
    // 3. Flag Cloister 4 Covered Perimeter Galleries (excluding open central 15-flag lawn!)
    if (x >= -20.6 && x <= 15.6 && z >= 38.8 && z <= 63.8) {
      const inOpenFlagLawn = x > -16.4 && x < 11.4 && z > 42.6 && z < 59.8;
      if (!inOpenFlagLawn) return true;
    }
    // 4. Front Gate Pendopo Roof
    if (x >= -9.5 && x <= 4.5 && z >= 60.0 && z <= 72.0) return true;
    // 5. 45-deg SE Breezeway & Octagonal Skylit Pendopo (29.5, 29.5)
    if (Math.hypot(x - 29.5, z - 29.5) <= 6.8) return true;
    if (x >= 15.0 && x <= 35.0 && z >= 15.0 && z <= 36.0 && Math.abs(x - z) < 3.4) return true;
    // 6. Long South-East Covered Campus Walkway ((34.2, 35.5) -> (43.8, 128.0))
    if (z >= 35.0 && z <= 128.5) {
      const tW = (z - 35.5) / (128.0 - 35.5);
      const wx = 34.2 + tW * (43.8 - 34.2);
      if (Math.abs(x - wx) <= 2.15) return true;
    }
    return false;
  }

  /**
   * Initializes the diverse 24-person Sivitas Akademika Universitas Indonesia roster across
   * 7 rich living campus behaviors (walking & chatting pairs, multi-waypoint navigators,
   * anatomically seated teak-bench scholars, lakeside contemplators, 4-person outdoor seminar,
   * and landmark/selfie photographers).
   */
  _initNPCRoster() {
    // Exact Rotunda outer bench helper matching SplineCorridorBuilder._buildCampusPromenadeBenchesAndBollards
    const getRotundaBench = (benchIdx, lateralOffset = 0.0) => {
      const th = 0.42 + (benchIdx / 6) * (Math.PI - 0.84);
      const rBench = 35.4;
      const bx = -2.5 + Math.cos(th) * rBench;
      const bz = 106.5 + Math.sin(th) * rBench;
      const yaw = Math.atan2(-2.5 - bx, 106.5 - bz);
      // Forward (+Z in bench local) is (sin(yaw), cos(yaw)); Left (+X) is (cos(yaw), -sin(yaw))
      const seatFwd = 0.04;
      return {
        x: bx + Math.cos(yaw) * lateralOffset + Math.sin(yaw) * seatFwd,
        z: bz - Math.sin(yaw) * lateralOffset + Math.cos(yaw) * seatFwd,
        seatY: 0.03 + 0.482,
        yaw
      };
    };

    const getLakesideBench = (lz, lateralOffset = 0.0) => {
      const bx = TerrainDEMLoader.getShorelinePromenadeX(lz) - 2.62;
      const yaw = Math.PI * 0.5; // Facing +X toward Danau Kenanga
      return {
        x: bx + 0.04,
        z: lz - lateralOffset,
        seatY: 0.04 + 0.482,
        yaw
      };
    };

    const bLakeNorth = getLakesideBench(-35.0, 0.10);
    const bLakeSouth = getLakesideBench(27.5, -0.12);
    const bRotundaPairL = getRotundaBench(2, -0.34);
    const bRotundaPairR = getRotundaBench(2, 0.34);

    const rosterSpecs = [
      // === A. 3 SOCIAL WALKING & CHATTING PAIRS (6 NPCs walking side-by-side in lively conversation!) ===
      // Pair 1: Walking & conversing around Lapangan Rotunda UI circular promenade
      {
        behavior: 'PAIR_ROTUNDA',
        pairId: 1,
        pairSide: -1,
        archetype: 'MAHASISWA_JAKUN',
        hairStyle: 'HAIR_WAVY',
        radius: 23.3,
        angle0: 0.35,
        dir: 1.0,
        baseSpeed: 1.16,
        heightScale: 1.01,
        buildScaleX: 1.02,
        propType: 'BACKPACK',
        hasUmbrella: true,
        hasLanyard: true
      },
      {
        behavior: 'PAIR_ROTUNDA',
        pairId: 1,
        pairSide: 1,
        archetype: 'MAHASISWI_HIJAB_JAKUN',
        hijabCol: HUMAN_PALETTE.HIJAB_CREAM,
        radius: 24.15,
        angle0: 0.35,
        dir: 1.0,
        baseSpeed: 1.16,
        heightScale: 0.93,
        buildScaleX: 0.96,
        propType: 'BOOK',
        hasUmbrella: true,
        hasLanyard: true
      },
      // Pair 2: Walking & chatting along the 120-segment Covered SE Terracotta Campus Walkway!
      {
        behavior: 'PAIR_SE_WALKWAY',
        pairId: 2,
        pairSide: -1,
        archetype: 'MAHASISWI_PONYTAIL_JAKUN',
        phase0: 0.18,
        baseSpeed: 1.14,
        heightScale: 0.95,
        buildScaleX: 0.95,
        propType: 'BOOK',
        hasUmbrella: false,
        hasLanyard: true
      },
      {
        behavior: 'PAIR_SE_WALKWAY',
        pairId: 2,
        pairSide: 1,
        archetype: 'MAHASISWI_HIJAB_JAKUN',
        hijabCol: HUMAN_PALETTE.HIJAB_DUSTY_ROSE,
        phase0: 0.18,
        baseSpeed: 1.14,
        heightScale: 0.92,
        buildScaleX: 0.96,
        propType: 'BACKPACK',
        hasUmbrella: true,
        hasLanyard: true
      },
      // Pair 3: Lecturer & Graduate Student strolling across Rotunda North-West Arc
      {
        behavior: 'PAIR_ROTUNDA',
        pairId: 3,
        pairSide: -1,
        archetype: 'DOSEN_BATIK_INDIGO',
        hairStyle: 'HAIR_PARTED',
        hasGlasses: true,
        radius: 33.1,
        angle0: 2.55,
        dir: -1.0,
        baseSpeed: 1.08,
        heightScale: 1.02,
        buildScaleX: 1.04,
        propType: 'BOOK',
        hasUmbrella: true
      },
      {
        behavior: 'PAIR_ROTUNDA',
        pairId: 3,
        pairSide: 1,
        archetype: 'MAHASISWA_JAKUN',
        hairStyle: 'HAIR_PARTED',
        hasGlasses: true,
        radius: 33.95,
        angle0: 2.55,
        dir: -1.0,
        baseSpeed: 1.08,
        heightScale: 0.99,
        buildScaleX: 0.99,
        propType: 'BACKPACK',
        hasUmbrella: false,
        hasLanyard: true
      },

      // === B. 5 AUTONOMOUS MULTI-WAYPOINT CAMPUS NAVIGATORS (Smooth Bezier Turns + Social Avoidance!) ===
      // Loop 1: Rotunda Gate (z=73) -> Covered Flag Cloister West Gallery -> Selasar Agung -> Lobby -> East Gallery
      {
        behavior: 'WAYPOINT_LOOP',
        waypoints: [
          [-3.2, 73.5],
          [-18.2, 61.5],
          [-18.2, 41.0],
          [-3.6, 37.5],
          [-3.4, 18.6],
          [13.4, 41.0],
          [13.4, 61.5],
          [-1.5, 73.5]
        ],
        wpIdx: 0,
        baseSpeed: 1.22,
        archetype: 'MAHASISWA_JAKUN',
        hairStyle: 'HAIR_PARTED',
        heightScale: 1.02,
        propType: 'BACKPACK',
        hasUmbrella: true,
        hasLanyard: true
      },
      {
        behavior: 'WAYPOINT_LOOP',
        waypoints: [
          [-3.2, 73.5],
          [-18.2, 61.5],
          [-18.2, 41.0],
          [-3.6, 37.5],
          [-3.4, 18.6],
          [13.4, 41.0],
          [13.4, 61.5],
          [-1.5, 73.5]
        ],
        wpIdx: 4,
        baseSpeed: 1.15,
        archetype: 'MAHASISWI_HIJAB_JAKUN',
        hijabCol: HUMAN_PALETTE.HIJAB_MAROON,
        heightScale: 0.94,
        propType: 'BOOK',
        hasUmbrella: true,
        hasLanyard: true
      },
      // Loop 2: ◇◇◇ Diamond Courtyard -> 45-deg SE Breezeway -> Octagonal Skylit Pendopo -> East Courtyard
      {
        behavior: 'WAYPOINT_LOOP',
        waypoints: [
          [-1.2, 34.0],
          [16.5, 22.0],
          [28.5, 28.5],
          [34.2, 37.0],
          [29.5, 29.5],
          [18.5, -12.0],
          [4.0, -28.0],
          [-16.0, -18.0],
          [-16.0, 18.0]
        ],
        wpIdx: 1,
        baseSpeed: 1.20,
        archetype: 'DOSEN_FORMAL',
        hairStyle: 'HAIR_PARTED',
        hasGlasses: true,
        heightScale: 1.03,
        buildScaleX: 1.03,
        propType: 'BOOK',
        hasUmbrella: true
      },
      {
        behavior: 'WAYPOINT_LOOP',
        waypoints: [
          [-1.2, 34.0],
          [16.5, 22.0],
          [28.5, 28.5],
          [34.2, 37.0],
          [29.5, 29.5],
          [18.5, -12.0],
          [4.0, -28.0],
          [-16.0, -18.0],
          [-16.0, 18.0]
        ],
        wpIdx: 6,
        baseSpeed: 1.26,
        archetype: 'MAHASISWA_JAKUN',
        hairStyle: 'HAIR_WAVY',
        heightScale: 0.98,
        propType: 'BACKPACK',
        hasUmbrella: false, // Will hurry & shield head with hand during rain!
        hasLanyard: true
      },
      // Loop 3: Ceremonial Flag Cloister & Selasar Agung Direct Axis Stroller
      {
        behavior: 'WAYPOINT_LOOP',
        waypoints: [
          [-2.2, 68.5],
          [-2.2, 38.5],
          [-2.2, 18.4],
          [-0.8, 38.5],
          [-0.8, 68.5]
        ],
        wpIdx: 1,
        baseSpeed: 1.10,
        archetype: 'MAHASISWI_HIJAB_JAKUN',
        hijabCol: HUMAN_PALETTE.HIJAB_NAVY,
        hasGlasses: true,
        heightScale: 0.93,
        propType: 'BACKPACK',
        hasUmbrella: true,
        hasLanyard: true
      },

      // === C. 4 ANATOMICALLY SEATED PARK-BENCH SCHOLARS (Sitting Flush on Exact 3D Teak Benches!) ===
      // Bench 1: Lakeside North Bench (-35m) — Student reading a Two-Page Spread Open Hardcover Textbook!
      {
        behavior: 'BENCH_SEATED',
        seatSpec: bLakeNorth,
        seatActivity: 'READ_OPEN_BOOK',
        archetype: 'MAHASISWI_HIJAB_JAKUN',
        hijabCol: HUMAN_PALETTE.HIJAB_SAGE,
        hasGlasses: true,
        heightScale: 0.94,
        propType: 'OPEN_BOOK',
        hasUmbrella: true,
        hasLanyard: true
      },
      // Bench 2: Lakeside South Bench (+27.5m) — Student checking Smartphone & admiring Danau Kenanga
      {
        behavior: 'BENCH_SEATED',
        seatSpec: bLakeSouth,
        seatActivity: 'SCROLL_PHONE',
        archetype: 'MAHASISWA_JAKUN',
        hairStyle: 'HAIR_WAVY',
        heightScale: 1.0,
        propType: 'PHONE',
        hasUmbrella: true,
        hasLanyard: true
      },
      // Bench 3 & 4: Two UI Friends Seated Side-by-Side on Rotunda Teak Bench Chatting & Pointing at Rektorat!
      {
        behavior: 'BENCH_SEATED',
        seatSpec: bRotundaPairL,
        seatActivity: 'BENCH_CHAT_LEFT',
        archetype: 'MAHASISWA_JAKUN',
        hairStyle: 'HAIR_PARTED',
        heightScale: 1.01,
        propType: 'BOOK',
        hasUmbrella: true,
        hasLanyard: true
      },
      {
        behavior: 'BENCH_SEATED',
        seatSpec: bRotundaPairR,
        seatActivity: 'BENCH_CHAT_RIGHT',
        archetype: 'MAHASISWI_PONYTAIL_JAKUN',
        heightScale: 0.94,
        propType: 'OPEN_BOOK',
        hasUmbrella: true,
        hasLanyard: true
      },

      // === D. 2 LAKESIDE PROMENADE STROLLERS & WATERFRONT CONTEMPLATORS (+X Danau Kenanga) ===
      {
        behavior: 'LAKESIDE_STROLL',
        centerZ: -16.0,
        spanZ: 15.0,
        phase0: 0.3,
        baseSpeed: 1.06,
        archetype: 'DOSEN_BATIK_SOGAN',
        hairStyle: 'HAIR_PARTED',
        hasGlasses: true,
        heightScale: 1.01,
        buildScaleX: 1.04,
        propType: 'BOOK',
        hasUmbrella: true
      },
      {
        behavior: 'LAKESIDE_STROLL',
        centerZ: 34.0,
        spanZ: 14.0,
        phase0: 2.4,
        baseSpeed: 1.12,
        archetype: 'MAHASISWI_HIJAB_JAKUN',
        hijabCol: HUMAN_PALETTE.HIJAB_MOCHA,
        heightScale: 0.93,
        propType: 'BACKPACK',
        hasUmbrella: false,
        hasLanyard: true
      },

      // === E. 4-PERSON OUTDOOR ACADEMIC SEMINAR CIRCLE ON ◇◇◇ DIAMOND COURTYARD PLAZA ===
      {
        behavior: 'SEMINAR_CIRCLE',
        clusterX: -19.5,
        clusterZ: 25.5,
        memberIdx: 0,
        memberCount: 4,
        archetype: 'DOSEN_BATIK_MAROON',
        hairStyle: 'HAIR_PARTED',
        hasGlasses: true,
        heightScale: 1.02,
        buildScaleX: 1.03,
        propType: 'OPEN_BOOK',
        hasUmbrella: true
      },
      {
        behavior: 'SEMINAR_CIRCLE',
        clusterX: -19.5,
        clusterZ: 25.5,
        memberIdx: 1,
        memberCount: 4,
        archetype: 'MAHASISWA_JAKUN',
        hairStyle: 'HAIR_WAVY',
        heightScale: 1.0,
        propType: 'BACKPACK',
        hasUmbrella: true,
        hasLanyard: true
      },
      {
        behavior: 'SEMINAR_CIRCLE',
        clusterX: -19.5,
        clusterZ: 25.5,
        memberIdx: 2,
        memberCount: 4,
        archetype: 'MAHASISWI_HIJAB_JAKUN',
        hijabCol: HUMAN_PALETTE.HIJAB_CREAM,
        heightScale: 0.92,
        propType: 'BOOK',
        hasUmbrella: true,
        hasLanyard: true
      },
      {
        behavior: 'SEMINAR_CIRCLE',
        clusterX: -19.5,
        clusterZ: 25.5,
        memberIdx: 3,
        memberCount: 4,
        archetype: 'MAHASISWI_PONYTAIL_JAKUN',
        hasGlasses: true,
        heightScale: 0.95,
        propType: 'OPEN_BOOK',
        hasUmbrella: true,
        hasLanyard: true
      },

      // === F. 3 CAMPUS LANDMARK & SELFIE SMARTPHONE PHOTOGRAPHERS (Glowing Viewfinder Screens!) ===
      {
        behavior: 'PHOTO_LANDMARK',
        spotX: -7.8,
        spotZ: 74.4,
        aimX: 0.0,
        aimZ: 0.0,
        phase0: 0.14,
        archetype: 'MAHASISWA_JAKUN',
        hairStyle: 'HAIR_WAVY',
        heightScale: 1.01,
        propType: 'PHONE',
        hasUmbrella: false,
        hasLanyard: true
      },
      {
        behavior: 'PHOTO_LANDMARK',
        spotX: 3.6,
        spotZ: 74.8,
        aimX: -1.5,
        aimZ: 12.0,
        phase0: 0.54,
        archetype: 'MAHASISWI_HIJAB_JAKUN',
        hijabCol: HUMAN_PALETTE.HIJAB_MAROON,
        heightScale: 0.94,
        propType: 'PHONE',
        hasUmbrella: true,
        hasLanyard: true
      },
      {
        behavior: 'PHOTO_LANDMARK',
        spotX: 19.8,
        spotZ: 24.8,
        aimX: 0.0,
        aimZ: 0.0,
        phase0: 0.82,
        archetype: 'MAHASISWI_PONYTAIL_JAKUN',
        heightScale: 0.95,
        propType: 'PHONE',
        hasUmbrella: true,
        hasLanyard: true
      }
    ];

    const umbrellaPalette = [
      HUMAN_PALETTE.UMBRELLA_NAVY,
      HUMAN_PALETTE.UMBRELLA_MAROON,
      HUMAN_PALETTE.UMBRELLA_SLATE
    ];
    const innerPalette = [
      HUMAN_PALETTE.INNER_WHITE,
      HUMAN_PALETTE.INNER_SKY,
      HUMAN_PALETTE.INNER_NAVY
    ];

    this.npcs = rosterSpecs.map((cfg, idx) => {
      const skinColor = HUMAN_PALETTE.SKIN_TONES[idx % HUMAN_PALETTE.SKIN_TONES.length];
      const umbrellaColor = umbrellaPalette[idx % umbrellaPalette.length];
      const innerShirtColor = innerPalette[idx % innerPalette.length];
      const bagColor = idx % 3 === 0 ? HUMAN_PALETTE.BAG_OLIVE : HUMAN_PALETTE.BAG_CHARCOAL;
      const bookColor = idx % 2 === 0 ? HUMAN_PALETTE.BOOK_CRIMSON : HUMAN_PALETTE.BOOK_NAVY;
      const shoeColor = idx % 2 === 0 ? HUMAN_PALETTE.SHOE_NAVY : HUMAN_PALETTE.SHOE_BLACK;
      const soleColor = idx % 4 === 0 ? HUMAN_PALETTE.SHOE_GUM_SOLE : HUMAN_PALETTE.SHOE_WHITE_TRIM;

      let profile;
      if (cfg.archetype === 'MAHASISWI_HIJAB_JAKUN') {
        const skirts = [
          HUMAN_PALETTE.SKIRT_NAVY,
          HUMAN_PALETTE.SKIRT_CHARCOAL,
          HUMAN_PALETTE.SKIRT_ESPRESSO
        ];
        profile = {
          garmentStyle: 'JAKUN_OPEN',
          torsoMain: HUMAN_PALETTE.JAKUN_YELLOW,
          innerShirtColor: HUMAN_PALETTE.INNER_WHITE,
          lowerColor: skirts[idx % skirts.length],
          skinColor,
          isFemale: true,
          hairStyle: 'HIJAB',
          isLongSkirt: true,
          headwearColor: cfg.hijabCol || HUMAN_PALETTE.HIJAB_CREAM,
          hasGlasses: Boolean(cfg.hasGlasses),
          hasLanyard: Boolean(cfg.hasLanyard),
          hasUmbrella: cfg.hasUmbrella !== false,
          umbrellaColor,
          propType: cfg.propType,
          bagColor,
          bookColor,
          shoeColor: HUMAN_PALETTE.SHOE_BLACK,
          soleColor: HUMAN_PALETTE.SHOE_WHITE_TRIM,
          buildScaleX: cfg.buildScaleX || 0.96
        };
      } else if (cfg.archetype === 'MAHASISWI_PONYTAIL_JAKUN') {
        profile = {
          garmentStyle: 'JAKUN_OPEN',
          torsoMain: HUMAN_PALETTE.JAKUN_YELLOW,
          innerShirtColor,
          lowerColor: idx % 2 === 0 ? HUMAN_PALETTE.DENIM_BLUE : HUMAN_PALETTE.TROUSER_NAVY,
          skinColor,
          isFemale: true,
          hairStyle: 'HAIR_PONYTAIL',
          isLongSkirt: idx % 3 === 0,
          headwearColor: HUMAN_PALETTE.HAIR_ESPRESSO,
          hasGlasses: Boolean(cfg.hasGlasses),
          hasLanyard: Boolean(cfg.hasLanyard),
          hasUmbrella: cfg.hasUmbrella !== false,
          umbrellaColor,
          propType: cfg.propType,
          bagColor,
          bookColor,
          shoeColor,
          soleColor: HUMAN_PALETTE.SHOE_WHITE_TRIM,
          buildScaleX: cfg.buildScaleX || 0.95
        };
      } else if (cfg.archetype.startsWith('DOSEN_BATIK')) {
        const batikCol =
          cfg.archetype === 'DOSEN_BATIK_INDIGO'
            ? HUMAN_PALETTE.BATIK_INDIGO
            : cfg.archetype === 'DOSEN_BATIK_MAROON'
              ? HUMAN_PALETTE.BATIK_MAROON
              : HUMAN_PALETTE.BATIK_SOGAN;
        profile = {
          garmentStyle: 'BATIK_LECTURER',
          torsoMain: batikCol,
          innerShirtColor: HUMAN_PALETTE.INNER_WHITE,
          lowerColor: HUMAN_PALETTE.TROUSER_CHARCOAL,
          skinColor,
          isFemale: false,
          hairStyle: cfg.hairStyle || 'HAIR_PARTED',
          isLongSkirt: false,
          headwearColor: HUMAN_PALETTE.HAIR_JET,
          hasGlasses: Boolean(cfg.hasGlasses),
          hasLanyard: false,
          hasUmbrella: cfg.hasUmbrella !== false,
          umbrellaColor: HUMAN_PALETTE.UMBRELLA_SLATE,
          propType: cfg.propType,
          bagColor,
          bookColor,
          shoeColor: HUMAN_PALETTE.SHOE_BLACK,
          soleColor: HUMAN_PALETTE.SHOE_BLACK,
          buildScaleX: cfg.buildScaleX || 1.03
        };
      } else if (cfg.archetype === 'DOSEN_FORMAL') {
        profile = {
          garmentStyle: 'FORMAL_SHIRT',
          torsoMain: HUMAN_PALETTE.FORMAL_SHIRT,
          innerShirtColor: HUMAN_PALETTE.INNER_WHITE,
          lowerColor: HUMAN_PALETTE.TROUSER_CHARCOAL,
          skinColor,
          isFemale: false,
          hairStyle: 'HAIR_PARTED',
          isLongSkirt: false,
          headwearColor: HUMAN_PALETTE.HAIR_JET,
          hasGlasses: Boolean(cfg.hasGlasses),
          hasLanyard: false,
          hasUmbrella: cfg.hasUmbrella !== false,
          umbrellaColor: HUMAN_PALETTE.UMBRELLA_NAVY,
          propType: cfg.propType,
          bagColor,
          bookColor,
          shoeColor: HUMAN_PALETTE.SHOE_BLACK,
          soleColor: HUMAN_PALETTE.SHOE_BLACK,
          buildScaleX: cfg.buildScaleX || 1.03
        };
      } else {
        // MAHASISWA_JAKUN
        const pants = [
          HUMAN_PALETTE.TROUSER_NAVY,
          HUMAN_PALETTE.DENIM_BLUE,
          HUMAN_PALETTE.CHINO_KHAKI,
          HUMAN_PALETTE.TROUSER_CHARCOAL
        ];
        profile = {
          garmentStyle: 'JAKUN_OPEN',
          torsoMain: HUMAN_PALETTE.JAKUN_YELLOW,
          innerShirtColor,
          lowerColor: pants[idx % pants.length],
          skinColor,
          isFemale: false,
          hairStyle: cfg.hairStyle || 'HAIR_WAVY',
          isLongSkirt: false,
          headwearColor: idx % 2 === 0 ? HUMAN_PALETTE.HAIR_JET : HUMAN_PALETTE.HAIR_ESPRESSO,
          hasGlasses: Boolean(cfg.hasGlasses),
          hasLanyard: Boolean(cfg.hasLanyard),
          hasUmbrella: cfg.hasUmbrella !== false,
          umbrellaColor,
          propType: cfg.propType,
          bagColor,
          bookColor,
          shoeColor,
          soleColor,
          buildScaleX: cfg.buildScaleX || 1.0
        };
      }

      // Initialize live continuous kinematic state
      let initX = 0.0;
      let initZ = 70.0;
      let initYaw = 0.0;
      if (cfg.behavior === 'WAYPOINT_LOOP') {
        const wp = cfg.waypoints[cfg.wpIdx % cfg.waypoints.length];
        const wpNext = cfg.waypoints[(cfg.wpIdx + 1) % cfg.waypoints.length];
        initX = wp[0];
        initZ = wp[1];
        initYaw = Math.atan2(wpNext[0] - initX, wpNext[1] - initZ);
      } else if (cfg.behavior === 'BENCH_SEATED') {
        initX = cfg.seatSpec.x;
        initZ = cfg.seatSpec.z;
        initYaw = cfg.seatSpec.yaw;
      }

      return {
        index: idx,
        ...cfg,
        profile,
        // Live continuous biomechanical state (zero teleportation, zero foot-slide!)
        worldX: initX,
        worldY: 0.0,
        worldZ: initZ,
        yaw: initYaw,
        turnRate: 0.0,
        liveSpeed: 0.0,
        gaitCycleU: (idx * 0.27) % 1.0,
        wpCursor: cfg.wpIdx || 0,
        pauseTimer: 0.0,
        umbrellaBlend: 0.0,
        rainHurryBlend: 0.0,
        playerGlanceBlend: 0.0,
        skirtSwingRx: 0.0,
        skirtSwingRz: 0.0,
        initializedPos: cfg.behavior === 'WAYPOINT_LOOP' || cfg.behavior === 'BENCH_SEATED'
      };
    });
  }

  /**
   * Walkable surface elevation resolver for cats across both Pulau U (plazas, lawns, forest hills,
   * riverbanks) and Pulau I (apron edge, western meadow, Niagara rim, & volcano foothills).
   */
  _sampleCatSurfaceY(x, z) {
    if (z < 536.0) {
      if (x >= -19.5 && x <= 18.5 && z >= -19.5 && z <= 18.5) return 0.37;
      if (x >= -36.5 && x <= 36.5 && z >= -35.5 && z <= 39.2) return 0.135;
      if (x >= -20.5 && x <= 15.5 && z > 39.2 && z <= 64.2) return 0.142;
      if (x >= -12.0 && x <= 7.0 && z > 64.2 && z <= 72.2) return 0.130;
      if (z >= 35.0 && z <= 128.5) {
        const tW = (z - 35.5) / (128.0 - 35.5);
        const wx = 34.2 + tW * (43.8 - 34.2);
        if (Math.abs(x - wx) < 1.95) return 0.055;
      }
    } else if (x >= -284.6 && x <= -223.4 && z >= 659.5 && z <= 697.5) {
      return TerrainDEMLoader.RUNWAY_DECK_Y;
    }
    return Math.max(-0.12, this.groundHeightFn(x, z) + 0.012);
  }

  /**
   * Ensures every cat spawn & patrol waypoint lies strictly on dry island land, clear of ocean turap,
   * freshwater springs/rivers/lakes/waterfall gorge, Helipad FATO, and Pulau I active Runway RWY 36/18.
   */
  _ensureSafeCatCoord(rawX, rawZ) {
    let x = rawX;
    let z = rawZ;

    // Keep Pulau I cats clear of active Runway RWY 36/18 strip (z = 708 +- 15m)
    if (z >= 536.0 && x >= -384.0 && x <= -124.0 && Math.abs(z - TerrainDEMLoader.RUNWAY_CZ) < 15.5) {
      z = 688.0;
    }

    // Keep Pulau U cats outside VIP Helipad FATO safety circle
    const dHeli = Math.hypot(x - TerrainDEMLoader.HELIPAD_CX, z - TerrainDEMLoader.HELIPAD_CZ);
    if (dHeli < 19.5) {
      const ang = Math.atan2(z - TerrainDEMLoader.HELIPAD_CZ, x - TerrainDEMLoader.HELIPAD_CX);
      x = TerrainDEMLoader.HELIPAD_CX + Math.cos(ang) * 20.5;
      z = TerrainDEMLoader.HELIPAD_CZ + Math.sin(ang) * 20.5;
    }

    // Nudge out of any freshwater body & inland from coastal stone turap
    for (let iter = 0; iter < 16; iter++) {
      const inFw = TerrainDEMLoader.isFreshwaterExclusion(x, z, 1.45);
      const dCoast = TerrainDEMLoader.getIslandCoastDistance(x, z);
      const gy = this.groundHeightFn(x, z);
      if (!inFw && dCoast <= -3.2 && gy >= -0.10) break;

      if (dCoast > -3.2) {
        const anchorX = z >= 536.0 ? TerrainDEMLoader.I_CENTER_X : -120.0;
        const anchorZ = z >= 536.0 ? TerrainDEMLoader.I_CENTER_Z : 95.0;
        const dx = anchorX - x;
        const dz = anchorZ - z;
        const len = Math.hypot(dx, dz) || 1.0;
        x += (dx / len) * 2.5;
        z += (dz / len) * 2.5;
      } else if (inFw) {
        if (z < 536.0) {
          const cz =
            x > -100.0
              ? TerrainDEMLoader.LAKE_U_CZ
              : x < -275.0
                ? TerrainDEMLoader.SPRING_U_CZ
                : TerrainDEMLoader.evalPulauURiverCenterZ(x);
          z += (z >= cz ? 1.0 : -1.0) * 2.2;
        } else {
          const dxW = x - TerrainDEMLoader.NIAGARA_CX;
          const dzW = z - TerrainDEMLoader.NIAGARA_CZ;
          const lenW = Math.hypot(dxW, dzW) || 1.0;
          x += (dxW / lenW) * 2.4;
          z += (dzW / lenW) * 2.4;
        }
      } else {
        x += 1.5;
      }
    }

    return [Number(x.toFixed(2)), Number(z.toFixed(2))];
  }

  /**
   * Initializes 48 super-realistic, proportionally accurate, genetically varied cats across
   * BOTH Pulau U (36 cats: Rektorat, Rotunda benches, Danau Kenanga, Danau Kecil, Helipad Pavilion,
   * Mata Air, Meandering River, & Southern Rainforest Hills) and Pulau I (12 cats: West Apron,
   * Coastal Meadow, Mini-Niagara Canyon Rim, & Gunung Api Sparse Mountain Forest)!
   */
  _initFelineRoster() {
    const getFreeRotundaBench = (benchIdx, lateralOffset = 0.0) => {
      const th = 0.42 + (benchIdx / 6) * (Math.PI - 0.84);
      const rBench = 35.4;
      const bx = -2.5 + Math.cos(th) * rBench;
      const bz = 106.5 + Math.sin(th) * rBench;
      const yaw = Math.atan2(-2.5 - bx, 106.5 - bz);
      return {
        x: bx + Math.cos(yaw) * lateralOffset,
        z: bz - Math.sin(yaw) * lateralOffset,
        perchY: 0.512,
        yaw
      };
    };

    const getFreeLakesideBench = (lz, lateralOffset = 0.0) => {
      const bx = TerrainDEMLoader.getShorelinePromenadeX(lz) - 2.62;
      return {
        x: bx + 0.02,
        z: lz - lateralOffset,
        perchY: 0.522,
        yaw: Math.PI * 0.5
      };
    };

    const benchRot0 = getFreeRotundaBench(0, 0.14);
    const benchRot4 = getFreeRotundaBench(4, -0.12);
    const benchLakeMid = getFreeLakesideBench(-8.0, 0.06);
    const promX14 = TerrainDEMLoader.getShorelinePromenadeX(14.0) - 3.25;
    const promXm22 = TerrainDEMLoader.getShorelinePromenadeX(-22.0) - 3.35;
    const promXm12 = TerrainDEMLoader.getShorelinePromenadeX(-12.0) - 3.35;

    const rZ225 = TerrainDEMLoader.evalPulauURiverCenterZ(-225.0);
    const rZ180 = TerrainDEMLoader.evalPulauURiverCenterZ(-180.0);
    const rZ194 = TerrainDEMLoader.evalPulauURiverCenterZ(-194.0);
    const rZ135 = TerrainDEMLoader.evalPulauURiverCenterZ(-135.0);

    const catPlacements = [
      // --- PULAU U: GEDUNG REKTORAT UI, SELASAR AGUNG, CLOISTER & SE WALKWAY (11 Cats) ---
      { zone: 'CAMPUS', behavior: 'PROWL_LOOP', coatPref: 1, buildPref: 'CHONKY_TOMCAT', speed: 0.58, wps: [[-6.8, 22.4], [-11.5, 29.8], [-5.2, 34.5], [-4.2, 25.0]] },
      { zone: 'CAMPUS', behavior: 'SIT_ALERT', coatPref: 6, x: -7.5, z: 23.8, yaw: 0.65 },
      { zone: 'CAMPUS', behavior: 'LOAF_LOUNGE', coatPref: 5, x: -5.5, z: 23.5, yaw: -0.45 },
      { zone: 'CAMPUS', behavior: 'PLAY_STALK', coatPref: 2, buildPref: 'KITTEN', x: -6.3, z: 25.2, yaw: 2.35 },
      { zone: 'CAMPUS', behavior: 'GROOM_PAW', coatPref: 10, x: -17.8, z: 49.5, yaw: 1.42 },
      { zone: 'CAMPUS', behavior: 'SIT_ALERT', coatPref: 8, x: 13.6, z: 51.0, yaw: -1.52 },
      { zone: 'CAMPUS', behavior: 'PROWL_LOOP', coatPref: 3, speed: 0.64, wps: [[-2.5, 65.8], [3.0, 73.2], [-6.2, 74.8]] },
      { zone: 'CAMPUS', behavior: 'PLAY_STALK', coatPref: 1, buildPref: 'KITTEN', x: -1.6, z: 66.8, yaw: 0.40 },
      { zone: 'CAMPUS', behavior: 'LOAF_LOUNGE', coatPref: 7, x: 25.4, z: 31.6, yaw: 2.15 },
      { zone: 'CAMPUS', behavior: 'PROWL_LOOP', coatPref: 4, speed: 0.66, wps: [[35.5, 46.5], [38.2, 72.0], [36.4, 58.0]] },
      { zone: 'CAMPUS', behavior: 'SIT_ALERT', coatPref: 9, x: -26.5, z: -18.0, yaw: 0.85 },

      // --- PULAU U: PLAZA ROTUNDA BENCH PERCHES & DANAU KENANGA PROMENADE (8 Cats) ---
      { zone: 'CAMPUS', behavior: 'LOAF_LOUNGE', coatPref: 2, x: benchRot0.x, z: benchRot0.z, perchY: benchRot0.perchY, yaw: benchRot0.yaw },
      { zone: 'CAMPUS', behavior: 'SIT_ALERT', coatPref: 11, x: benchRot4.x, z: benchRot4.z, perchY: benchRot4.perchY, yaw: benchRot4.yaw },
      { zone: 'CAMPUS', behavior: 'LOAF_LOUNGE', coatPref: 5, x: benchLakeMid.x, z: benchLakeMid.z, perchY: benchLakeMid.perchY, yaw: benchLakeMid.yaw },
      { zone: 'CAMPUS', behavior: 'SIT_ALERT', coatPref: 12, x: promX14, z: 14.0, yaw: Math.PI * 0.5 },
      { zone: 'CAMPUS', behavior: 'PROWL_LOOP', coatPref: 3, speed: 0.62, wps: [[promXm22, -22.0], [promXm12, -12.0]] },
      { zone: 'CAMPUS', behavior: 'GROOM_PAW', coatPref: 1, x: -16.5, z: 96.0, yaw: 0.52 },
      { zone: 'CAMPUS', behavior: 'PROWL_LOOP', coatPref: 6, speed: 0.68, wps: [[12.5, 98.5], [19.0, 109.5], [8.5, 117.5]] },
      { zone: 'CAMPUS', behavior: 'PLAY_STALK', coatPref: 5, buildPref: 'KITTEN', x: -4.5, z: 128.0, yaw: 3.10 },

      // --- PULAU U: DANAU KECIL DEKAT REKTORAT & WEST PARKLAND (6 Cats) ---
      { zone: 'ISLAND_U', behavior: 'SIT_ALERT', coatPref: 3, x: -68.4, z: -18.5, yaw: -Math.PI * 0.5 },
      { zone: 'ISLAND_U', behavior: 'PLAY_STALK', coatPref: 2, x: -68.8, z: -26.5, yaw: -1.35 },
      { zone: 'ISLAND_U', behavior: 'LOAF_LOUNGE', coatPref: 5, x: -82.5, z: 3.2, yaw: 3.05 },
      { zone: 'ISLAND_U', behavior: 'PROWL_LOOP', coatPref: 4, speed: 0.64, wps: [[-97.8, -19.5], [-96.5, -31.0], [-99.2, -10.5]] },
      { zone: 'ISLAND_U', behavior: 'GROOM_PAW', coatPref: 7, x: -83.8, z: -46.4, yaw: 0.15 },
      { zone: 'ISLAND_U', behavior: 'PROWL_LOOP', coatPref: 10, speed: 0.60, wps: [[-62.0, -8.0], [-58.2, 3.5], [-66.0, 1.8]] },

      // --- PULAU U: HELIPAD PAVILION, MATA AIR, MEANDERING RIVER & SOUTHERN HIGHLAND FOREST (11 Cats) ---
      { zone: 'ISLAND_U', behavior: 'SIT_ALERT', coatPref: 6, x: -80.5, z: 116.5, yaw: -1.15 },
      { zone: 'ISLAND_U', behavior: 'PROWL_LOOP', coatPref: 1, speed: 0.65, wps: [[-75.0, 103.5], [-61.5, 104.2]] },
      { zone: 'ISLAND_U', behavior: 'LOAF_LOUNGE', coatPref: 11, x: -84.0, z: 88.5, yaw: 1.85 },
      { zone: 'ISLAND_U', behavior: 'SIT_ALERT', coatPref: 9, x: -286.0, z: 34.2, yaw: 0.05 },
      { zone: 'ISLAND_U', behavior: 'PROWL_LOOP', coatPref: 3, speed: 0.66, wps: [[-272.5, 57.5], [-261.0, 61.0], [-278.0, 63.0]] },
      { zone: 'ISLAND_U', behavior: 'GROOM_PAW', coatPref: 7, x: -299.0, z: 51.5, yaw: 1.48 },
      { zone: 'ISLAND_U', behavior: 'SIT_ALERT', coatPref: 1, x: -225.0, z: rZ225 + 7.8, yaw: 3.12 },
      { zone: 'ISLAND_U', behavior: 'PROWL_LOOP', coatPref: 5, speed: 0.63, wps: [[-180.0, rZ180 - 8.0], [-194.0, rZ194 - 8.2]] },
      { zone: 'ISLAND_U', behavior: 'PLAY_STALK', coatPref: 12, x: -135.0, z: rZ135 + 8.2, yaw: -2.85 },
      { zone: 'ISLAND_U', behavior: 'SIT_ALERT', coatPref: 3, x: -328.0, z: 195.0, yaw: 0.78 },
      { zone: 'ISLAND_U', behavior: 'PROWL_LOOP', coatPref: 2, speed: 0.64, wps: [[-85.0, 385.0], [-102.0, 378.0]] },

      // --- PULAU I: WEST APRON EDGE, COASTAL MEADOW, MINI-NIAGARA RIM & SPARSE VOLCANO FOREST (12 Cats) ---
      { zone: 'ISLAND_I', behavior: 'SIT_ALERT', coatPref: 1, x: -252.0, z: 661.2, yaw: -Math.PI * 0.5 },
      { zone: 'ISLAND_I', behavior: 'LOAF_LOUNGE', coatPref: 6, x: -236.0, z: 662.0, yaw: 1.25 },
      { zone: 'ISLAND_I', behavior: 'PROWL_LOOP', coatPref: 3, speed: 0.66, wps: [[-268.0, 651.0], [-246.0, 645.5]] },
      { zone: 'ISLAND_I', behavior: 'GROOM_PAW', coatPref: 5, x: -195.0, z: 648.0, yaw: -1.40 },
      { zone: 'ISLAND_I', behavior: 'PLAY_STALK', coatPref: 12, buildPref: 'KITTEN', x: -315.0, z: 655.0, yaw: 0.92 },
      { zone: 'ISLAND_I', behavior: 'SIT_ALERT', coatPref: 3, x: -63.0, z: 636.0, yaw: 1.05 },
      { zone: 'ISLAND_I', behavior: 'SIT_ALERT', coatPref: 11, x: -25.0, z: 636.5, yaw: 2.15 },
      { zone: 'ISLAND_I', behavior: 'PROWL_LOOP', coatPref: 2, speed: 0.64, wps: [[-58.5, 623.5], [-66.5, 631.5]] },
      { zone: 'ISLAND_I', behavior: 'LOAF_LOUNGE', coatPref: 8, x: -27.0, z: 622.0, yaw: -0.75 },
      { zone: 'ISLAND_I', behavior: 'PROWL_LOOP', coatPref: 4, speed: 0.65, wps: [[-98.0, 676.0], [-109.0, 688.0]] },
      { zone: 'ISLAND_I', behavior: 'SIT_ALERT', coatPref: 9, x: -78.0, z: 742.0, yaw: 1.62 },
      { zone: 'ISLAND_I', behavior: 'PROWL_LOOP', coatPref: 7, speed: 0.62, wps: [[18.0, 672.0], [28.0, 686.0]] }
    ];

    this.cats = catPlacements.map((spec, idx) => {
      const phenotype = FelineAnatomyBuilder.generateCatPhenotype(idx, spec.zone);
      // Apply curated coat & age overrides for signature campus vignettes while preserving full 12-coat diversity
      if (spec.coatPref) {
        const cSpec = FelineAnatomyBuilder.generateCatPhenotype(spec.coatPref * 12 + idx, spec.zone);
        // Find exact requested coatId
        for (let s = 0; s < 36; s++) {
          const cand = FelineAnatomyBuilder.generateCatPhenotype(s + idx * 7, spec.zone);
          if (cand.coatId === spec.coatPref) {
            Object.assign(phenotype, cand);
            break;
          }
        }
        void cSpec;
      }
      if (spec.buildPref === 'KITTEN') {
        phenotype.ageBuildClass = 'KITTEN';
        phenotype.overallScale = Number((0.61 + (idx % 3) * 0.025).toFixed(3));
        phenotype.headRatio = 1.13;
        phenotype.earScale = 1.12;
        phenotype.girthScale = 0.94;
        phenotype.hasCollar = false;
      } else if (spec.buildPref === 'CHONKY_TOMCAT') {
        phenotype.ageBuildClass = 'CHONKY_TOMCAT';
        phenotype.overallScale = 1.11;
        phenotype.girthScale = 1.22;
        phenotype.jowlScale = 1.16;
      }

      let safeWps = null;
      let initX = spec.x ?? 0.0;
      let initZ = spec.z ?? 0.0;
      let initYaw = spec.yaw ?? 0.0;

      if (spec.behavior === 'PROWL_LOOP' && Array.isArray(spec.wps)) {
        safeWps = spec.wps.map((pt) => this._ensureSafeCatCoord(pt[0], pt[1]));
        initX = safeWps[0][0];
        initZ = safeWps[0][1];
        const nextPt = safeWps[1 % safeWps.length];
        initYaw = Math.atan2(nextPt[0] - initX, nextPt[1] - initZ);
      } else if (spec.perchY === undefined) {
        const safePt = this._ensureSafeCatCoord(initX, initZ);
        initX = safePt[0];
        initZ = safePt[1];
      }

      return {
        index: idx,
        zone: spec.zone,
        behavior: spec.behavior,
        phenotype,
        perchY: spec.perchY,
        waypoints: safeWps,
        wpCursor: 1,
        baseSpeed: spec.speed || 0.62,
        homeX: initX,
        homeZ: initZ,
        worldX: initX,
        worldY: spec.perchY !== undefined ? spec.perchY : this._sampleCatSurfaceY(initX, initZ),
        worldZ: initZ,
        yaw: initYaw,
        slopePitch: 0.0,
        slopeRoll: 0.0,
        liveSpeed: 0.0,
        gaitU: (idx * 0.31) % 1.0,
        pauseTimer: (idx % 4) * 0.65,
        phaseSeed: idx * 2.37,
        playerLookBlend: 0.0
      };
    });
  }

  /**
   * Builds the unified single-draw-call Dual-Bone LBS Mesh for all 24 Sivitas Akademika NPCs
   * AND all 48 Multi-Island Proportional Articulated Cats (Strictly 1 Draw Call Total!).
   * @returns {THREE.Group}
   */
  build() {
    this._initNPCRoster();
    this._initFelineRoster();

    const charBundles = this.npcs.map((npc) =>
      HumanAnatomyBuilder.buildSkinnedCharacter(npc.profile)
    );
    const catBundles = this.cats.map((cat) =>
      FelineAnatomyBuilder.buildSkinnedCat(cat.phenotype)
    );

    let totalVerts = 0;
    let totalIndices = 0;
    for (let n = 0; n < charBundles.length; n++) {
      totalVerts += charBundles[n].pos.length / 3;
      totalIndices += charBundles[n].idx.length;
    }
    for (let c = 0; c < catBundles.length; c++) {
      totalVerts += catBundles[c].pos.length / 3;
      totalIndices += catBundles[c].idx.length;
    }

    this._restPos = new Float32Array(totalVerts * 3);
    this._restNorm = new Float32Array(totalVerts * 3);
    this._vertJ0 = new Uint16Array(totalVerts);
    this._vertJ1 = new Uint16Array(totalVerts);
    this._vertW0 = new Float32Array(totalVerts);
    this._livePos = new Float32Array(totalVerts * 3);
    this._liveNorm = new Float32Array(totalVerts * 3);

    const colors = new Float32Array(totalVerts * 3);
    const uvs = new Float32Array(totalVerts * 2);
    const humanMeta = new Float32Array(totalVerts * 3);
    const indices = new Uint32Array(totalIndices);

    let vCursor = 0;
    let iCursor = 0;

    for (let n = 0; n < charBundles.length; n++) {
      const b = charBundles[n];
      const vCount = b.pos.length / 3;
      const jointBase = n * JOINTS_PER_NPC;

      this._restPos.set(b.pos, vCursor * 3);
      this._restNorm.set(b.norm, vCursor * 3);
      colors.set(b.col, vCursor * 3);
      uvs.set(b.uv, vCursor * 2);
      humanMeta.set(b.meta, vCursor * 3);
      this._vertW0.set(b.w0, vCursor);

      for (let k = 0; k < vCount; k++) {
        this._vertJ0[vCursor + k] = jointBase + b.j0[k];
        this._vertJ1[vCursor + k] = jointBase + b.j1[k];
      }

      for (let k = 0; k < b.idx.length; k++) {
        indices[iCursor++] = vCursor + b.idx[k];
      }
      vCursor += vCount;
    }

    const catGlobalJointStart = this.npcs.length * JOINTS_PER_NPC;
    for (let c = 0; c < catBundles.length; c++) {
      const b = catBundles[c];
      const vCount = b.pos.length / 3;
      const jointBase = catGlobalJointStart + c * CAT_JOINTS_PER_CAT;

      this._restPos.set(b.pos, vCursor * 3);
      this._restNorm.set(b.norm, vCursor * 3);
      colors.set(b.col, vCursor * 3);
      uvs.set(b.uv, vCursor * 2);
      humanMeta.set(b.meta, vCursor * 3);
      this._vertW0.set(b.w0, vCursor);

      for (let k = 0; k < vCount; k++) {
        this._vertJ0[vCursor + k] = jointBase + b.j0[k];
        this._vertJ1[vCursor + k] = jointBase + b.j1[k];
      }

      for (let k = 0; k < b.idx.length; k++) {
        indices[iCursor++] = vCursor + b.idx[k];
      }
      vCursor += vCount;
    }

    const totalJoints =
      this.npcs.length * JOINTS_PER_NPC + this.cats.length * CAT_JOINTS_PER_CAT;
    this._jointWorldMatrices = new Float32Array(totalJoints * 16);
    this._skinMatrices = new Float32Array(totalJoints * 16);
    this._skinNormalMatrices = new Float32Array(totalJoints * 9);

    const unifiedGeo = new THREE.BufferGeometry();
    unifiedGeo.setAttribute('position', new THREE.BufferAttribute(this._livePos, 3));
    unifiedGeo.setAttribute('normal', new THREE.BufferAttribute(this._liveNorm, 3));
    unifiedGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    unifiedGeo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    unifiedGeo.setAttribute('aHumanMeta', new THREE.BufferAttribute(humanMeta, 3));
    unifiedGeo.setIndex(new THREE.BufferAttribute(indices, 1));

    const npcMaterial = createPhotorealHumanMaterial(this.sharedUniforms);

    this.mesh = new THREE.Mesh(unifiedGeo, npcMaterial);
    this.mesh.name = 'SivitasAkademikaUIUnified18JointMesh';
    this.mesh.userData.dynamicKinematic = true;
    this.mesh.userData.skipRainBake = true;
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.mesh.frustumCulled = false;

    while (this.group.children.length > 0) {
      this.group.remove(this.group.children[0]);
    }
    this.group.add(this.mesh);
    this.group.userData.dynamicKinematic = true;
    this.group.userData.skipRainBake = true;
    this.group.userData.npcCount = this.npcs.length;
    this.group.userData.jointsPerNpc = JOINTS_PER_NPC;
    this.group.userData.catCount = this.cats.length;
    this.group.userData.jointsPerCat = CAT_JOINTS_PER_CAT;
    this.group.userData.catsOnPulauU = this.cats.filter((c) => c.worldZ < 536.0).length;
    this.group.userData.catsOnPulauI = this.cats.filter((c) => c.worldZ >= 536.0).length;
    this.group.userData.uniqueCatCoatCount = new Set(this.cats.map((c) => c.phenotype.coatId)).size;
    this.group.userData.uniqueCatAgeBuildCount = new Set(
      this.cats.map((c) => c.phenotype.ageBuildClass)
    ).size;

    // Populate initial articulated poses immediately so t = 0 buffers are 100% valid & verified
    this.update(0.016, 0.0);
    this.group.updateMatrixWorld(true);
    return this.group;
  }

  /**
   * Composes a child joint's 4x4 live world matrix into `this._jointWorldMatrices` AND immediately
   * computes its exact Bind-Relative Skin Matrix `S[j] = M_world[j] * Translation(-bindPivot[j])`!
   */
  _writeJointWorldMatrix(
    npcIdx,
    jointId,
    parentJointId,
    lx,
    ly,
    lz,
    rx,
    ry,
    rz,
    sx = 1.0,
    sy = 1.0,
    sz = 1.0
  ) {
    this._tmpPos.set(lx, ly, lz);
    this._tmpEuler.set(rx, ry, rz);
    this._tmpQuat.setFromEuler(this._tmpEuler);
    this._tmpScale.set(sx, sy, sz);
    this._tmpLocalMat.compose(this._tmpPos, this._tmpQuat, this._tmpScale);

    const globalJ = npcIdx * JOINTS_PER_NPC + jointId;
    const outOffset = globalJ * 16;
    const wMats = this._jointWorldMatrices;

    if (parentJointId < 0) {
      const e = this._tmpLocalMat.elements;
      for (let i = 0; i < 16; i++) wMats[outOffset + i] = e[i];
    } else {
      const parentOffset = (npcIdx * JOINTS_PER_NPC + parentJointId) * 16;
      for (let i = 0; i < 16; i++) {
        this._tmpParentMat.elements[i] = wMats[parentOffset + i];
      }
      this._tmpWorldMat.multiplyMatrices(this._tmpParentMat, this._tmpLocalMat);
      const e = this._tmpWorldMat.elements;
      for (let i = 0; i < 16; i++) wMats[outOffset + i] = e[i];
    }

    // Compute Skin Matrix S[j] = M_world[j] * Translation(-bx, -by, -bz)
    const bp = BIND_PIVOTS[jointId];
    const bx = bp[0];
    const by = bp[1];
    const bz = bp[2];
    const sMats = this._skinMatrices;
    for (let i = 0; i < 12; i++) {
      sMats[outOffset + i] = wMats[outOffset + i];
    }
    sMats[outOffset + 12] =
      wMats[outOffset + 12] -
      (wMats[outOffset] * bx + wMats[outOffset + 4] * by + wMats[outOffset + 8] * bz);
    sMats[outOffset + 13] =
      wMats[outOffset + 13] -
      (wMats[outOffset + 1] * bx + wMats[outOffset + 5] * by + wMats[outOffset + 9] * bz);
    sMats[outOffset + 14] =
      wMats[outOffset + 14] -
      (wMats[outOffset + 2] * bx + wMats[outOffset + 6] * by + wMats[outOffset + 10] * bz);
    sMats[outOffset + 15] = 1.0;
  }

  /**
   * Step 1 of Frame Update: Advances autonomous navigation, social-pair synchronization,
   * Helbing-Molnar personal-space collision avoidance, and live rain/shelter behavioral state!
   */
  _advanceCrowdNavigation(dt, t, cameraPos = null) {
    const rainInt = this.sharedUniforms?.uRainIntensity?.value ?? 0.0;

    for (let i = 0; i < this.npcs.length; i++) {
      const npc = this.npcs[i];
      const prevX = npc.worldX;
      const prevZ = npc.worldZ;
      const prevYaw = npc.yaw;

      // Determine rain exposure vs architectural roof shelter!
      const isSheltered = DynamicTrafficAI._isShelteredByRoof(npc.worldX, npc.worldZ);
      const exposedRain = !isSheltered && rainInt > 0.10 ? Math.min(1.0, (rainInt - 0.08) * 1.45) : 0.0;

      const wantUmbrella = npc.profile.hasUmbrella ? exposedRain : 0.0;
      const wantHurry = !npc.profile.hasUmbrella && npc.behavior !== 'BENCH_SEATED' ? exposedRain : 0.0;

      npc.umbrellaBlend = dampScalar(npc.umbrellaBlend, wantUmbrella, 2.6, dt);
      npc.rainHurryBlend = dampScalar(npc.rainHurryBlend, wantHurry, 2.8, dt);

      const speedMult = 1.0 + 0.44 * npc.rainHurryBlend;

      if (npc.behavior === 'PAIR_ROTUNDA') {
        const speed = npc.baseSpeed * speedMult;
        const omega = (npc.dir * speed) / 23.7;
        const theta = npc.angle0 + omega * t;
        const tx = -2.5 + Math.cos(theta) * npc.radius;
        const tz = 106.5 + Math.sin(theta) * npc.radius;
        const targetYaw = Math.atan2(-npc.dir * Math.sin(theta), npc.dir * Math.cos(theta));

        if (!npc.initializedPos) {
          npc.worldX = tx;
          npc.worldZ = tz;
          npc.yaw = targetYaw;
          npc.initializedPos = true;
        } else {
          npc.worldX = tx;
          npc.worldZ = tz;
          npc.yaw = dampAngle(npc.yaw, targetYaw, 8.0, dt);
        }
      } else if (npc.behavior === 'PAIR_SE_WALKWAY') {
        // Walk & converse smoothly up and down the Covered SE Terracotta Campus Walkway ((34.2, 35.5) <-> (43.2, 121.0))
        const cycle = 56.0;
        const u = (((t / cycle + npc.phase0) % 1.0) + 1.0) % 1.0;
        let sPos = 0.0;
        let dirSign = 1.0;
        if (u < 0.46) {
          sPos = u / 0.46;
          dirSign = 1.0;
        } else if (u < 0.50) {
          sPos = 1.0;
          dirSign = 0.0;
        } else if (u < 0.96) {
          sPos = 1.0 - (u - 0.50) / 0.46;
          dirSign = -1.0;
        } else {
          sPos = 0.0;
          dirSign = 0.0;
        }
        const cx = 34.2 + sPos * (43.2 - 34.2);
        const cz = 37.0 + sPos * (120.0 - 37.0);
        const lat = npc.pairSide * 0.46;
        const tx = cx + lat;
        const tz = cz;
        const walkAz = Math.atan2(43.2 - 34.2, 120.0 - 37.0);
        const targetYaw = dirSign >= 0 ? (dirSign === 0 && u < 0.5 ? walkAz + Math.PI * 0.5 : walkAz) : walkAz + Math.PI;

        if (!npc.initializedPos) {
          npc.worldX = tx;
          npc.worldZ = tz;
          npc.yaw = targetYaw;
          npc.initializedPos = true;
        } else {
          npc.worldX = dampScalar(npc.worldX, tx, 10.0, dt);
          npc.worldZ = dampScalar(npc.worldZ, tz, 10.0, dt);
          npc.yaw = dampAngle(npc.yaw, targetYaw, 5.5, dt);
        }
      } else if (npc.behavior === 'WAYPOINT_LOOP') {
        if (npc.pauseTimer > 0.0) {
          npc.pauseTimer = Math.max(0.0, npc.pauseTimer - dt);
        } else {
          const wps = npc.waypoints;
          const targetWp = wps[npc.wpCursor % wps.length];
          let dx = targetWp[0] - npc.worldX;
          let dz = targetWp[1] - npc.worldZ;
          const dist = Math.hypot(dx, dz);

          if (dist < 1.15) {
            npc.wpCursor = (npc.wpCursor + 1) % wps.length;
            // Brief natural contemplative pause at lobby colonnade or pendopo!
            if (targetWp[1] < 19.5 && Math.abs(targetWp[0]) < 5.0 && npc.rainHurryBlend < 0.2) {
              npc.pauseTimer = 1.8;
            }
          } else {
            let dirX = dx / dist;
            let dirZ = dz / dist;

            // Helbing-Molnar Social-Force Personal-Space Avoidance against other NPCs & Player Camera!
            let avoidX = 0.0;
            let avoidZ = 0.0;
            for (let j = 0; j < this.npcs.length; j++) {
              if (i === j) continue;
              const other = this.npcs[j];
              const ox = npc.worldX - other.worldX;
              const oz = npc.worldZ - other.worldZ;
              const dSq = ox * ox + oz * oz;
              if (dSq < 2.25 && dSq > 1e-4) {
                const d = Math.sqrt(dSq);
                const push = ((1.5 - d) / 1.5) * 0.85;
                avoidX += (ox / d) * push;
                avoidZ += (oz / d) * push;
              }
            }
            if (cameraPos && Number.isFinite(cameraPos.x) && cameraPos.y < 4.5) {
              const cx = npc.worldX - cameraPos.x;
              const cz = npc.worldZ - cameraPos.z;
              const cSq = cx * cx + cz * cz;
              if (cSq < 3.24 && cSq > 1e-4) {
                const cd = Math.sqrt(cSq);
                const push = ((1.8 - cd) / 1.8) * 1.15;
                avoidX += (cx / cd) * push;
                avoidZ += (cz / cd) * push;
              }
            }

            dirX += avoidX;
            dirZ += avoidZ;
            const dNorm = Math.hypot(dirX, dirZ) || 1.0;
            dirX /= dNorm;
            dirZ /= dNorm;

            const desiredYaw = Math.atan2(dirX, dirZ);
            npc.yaw = dampAngle(npc.yaw, desiredYaw, 5.2, dt);

            // Slow down slightly while making sharp corner turns (natural centripetal human pacing!)
            const turnSlow = 1.0 - 0.35 * Math.min(1.0, Math.abs(wrapAngle(desiredYaw - npc.yaw)));
            const stepSpeed = npc.baseSpeed * speedMult * turnSlow;
            npc.worldX += Math.sin(npc.yaw) * stepSpeed * dt;
            npc.worldZ += Math.cos(npc.yaw) * stepSpeed * dt;
          }
        }
      } else if (npc.behavior === 'LAKESIDE_STROLL') {
        const period = 24.0;
        const localT = (t + npc.phase0 * 3.7) % period;
        let tz = npc.centerZ;
        let targetYaw = Math.PI * 0.5;
        if (localT < 7.5) {
          tz = npc.centerZ - npc.spanZ * 0.5 + (localT / 7.5) * npc.spanZ;
          targetYaw = 0.0;
        } else if (localT < 12.0) {
          tz = npc.centerZ + npc.spanZ * 0.5;
          targetYaw = Math.PI * 0.5; // Admiring Danau Kenanga water (+X)!
        } else if (localT < 19.5) {
          tz = npc.centerZ + npc.spanZ * 0.5 - ((localT - 12.0) / 7.5) * npc.spanZ;
          targetYaw = Math.PI;
        } else {
          tz = npc.centerZ - npc.spanZ * 0.5;
          targetYaw = Math.PI * 0.5;
        }
        const tx = TerrainDEMLoader.getShorelinePromenadeX(tz) - 1.38;
        if (!npc.initializedPos) {
          npc.worldX = tx;
          npc.worldZ = tz;
          npc.yaw = targetYaw;
          npc.initializedPos = true;
        } else {
          npc.worldX = dampScalar(npc.worldX, tx, 8.0, dt);
          npc.worldZ = dampScalar(npc.worldZ, tz, 8.0, dt);
          npc.yaw = dampAngle(npc.yaw, targetYaw, 4.8, dt);
        }
      } else if (npc.behavior === 'SEMINAR_CIRCLE') {
        const rCluster = 1.24;
        const az = (npc.memberIdx / npc.memberCount) * Math.PI * 2.0 + 0.38;
        npc.worldX = npc.clusterX + Math.cos(az) * rCluster;
        npc.worldZ = npc.clusterZ + Math.sin(az) * rCluster;
        npc.yaw = Math.atan2(npc.clusterX - npc.worldX, npc.clusterZ - npc.worldZ);
        npc.initializedPos = true;
      } else if (npc.behavior === 'PHOTO_LANDMARK') {
        npc.worldX = npc.spotX;
        npc.worldZ = npc.spotZ;
        npc.yaw = Math.atan2(npc.aimX - npc.worldX, npc.aimZ - npc.worldZ);
        npc.initializedPos = true;
      } else if (npc.behavior === 'BENCH_SEATED') {
        npc.worldX = npc.seatSpec.x;
        npc.worldZ = npc.seatSpec.z;
        npc.yaw = npc.seatSpec.yaw;
        npc.initializedPos = true;
      }

      npc.worldY =
        npc.behavior === 'BENCH_SEATED'
          ? npc.seatSpec.seatY
          : this._samplePedestrianGroundY(npc.worldX, npc.worldZ);

      // Measure TRUE instantaneous world displacement -> ZERO FOOT SLIDING / MOONWALKING!
      const disp = dt > 1e-5 ? Math.hypot(npc.worldX - prevX, npc.worldZ - prevZ) : 0.0;
      const instSpeed = dt > 1e-5 && disp < 3.0 ? disp / dt : 0.0;
      npc.liveSpeed = dampScalar(npc.liveSpeed, instSpeed, 12.0, dt);
      npc.turnRate =
        dt > 1e-5 ? dampScalar(npc.turnRate, wrapAngle(npc.yaw - prevYaw) / dt, 8.0, dt) : 0.0;

      // Exact Stride-Length Phase Integration (1 full L+R gait cycle = ~1.44m * hScale)
      const hScale = npc.heightScale || 1.0;
      const fullCycleMeters = 1.42 * hScale * (0.82 + 0.18 * Math.min(1.6, npc.liveSpeed / 1.2));
      if (disp > 1e-5 && disp < 3.0) {
        npc.gaitCycleU = (npc.gaitCycleU + disp / fullCycleMeters) % 1.0;
      }

      // Player Camera Proximity Eye-Contact & Friendly Campus Recognition!
      let wantPlayerGlance = 0.0;
      if (cameraPos && Number.isFinite(cameraPos.x)) {
        const toCamX = cameraPos.x - npc.worldX;
        const toCamZ = cameraPos.z - npc.worldZ;
        const camDist = Math.hypot(toCamX, toCamZ);
        if (camDist > 0.7 && camDist < 4.6 && Math.abs(cameraPos.y - (npc.worldY + 1.55)) < 2.2) {
          const fwdDot = (Math.sin(npc.yaw) * toCamX + Math.cos(npc.yaw) * toCamZ) / camDist;
          if (fwdDot > -0.10) {
            wantPlayerGlance = 1.0 - smooth01(2.2, 4.6, camDist);
          }
        }
      }
      npc.playerGlanceBlend = dampScalar(npc.playerGlanceBlend, wantPlayerGlance, 4.0, dt);
    }
  }

  /**
   * Step 2 of Frame Update: Solves full 22-joint human biomechanics for a single Sivitas Akademika NPC:
   * - Clinical 60/40 Asymmetric Gait + Metatarsal Toe-Roll + Exact Lowest-Foot Ground Contact
   * - Contrapposto Asymmetric Idle Weight-Shifting (no stiff symmetrical statues!)
   * - Seated 90-deg Park Bench Posture (reading open book, scrolling phone, or chatting with bench friend)
   * - Double-Pendulum Phase-Lagged Arm/Forearm/Wrist Swing + Vestibular Head Stabilization
   * - Expressive Social Pair Conversation, Seminar Teaching Gestures, & Rain Umbrella Wind-Tilting!
   */
  _solveNPCBiomechanics(npc, t, dt, cameraPos = null) {
    const idx = npc.index;
    const hScale = npc.heightScale || 1.0;
    const windVec = this.sharedUniforms?.uWindVec?.value;
    const windX = windVec ? windVec.x : 0.65;
    const windZ = windVec ? windVec.z : 0.35;

    // Breathing rhythm (accelerates slightly when walking briskly in rain!)
    const breathFreq = 1.55 + 0.85 * Math.min(1.5, npc.liveSpeed) + 0.65 * npc.rainHurryBlend;
    const breathe = Math.sin(t * breathFreq + idx * 1.13);

    // =========================================================================
    // CASE 1: ANATOMICALLY SEATED PARK-BENCH POSTURE ('BENCH_SEATED')
    // =========================================================================
    if (npc.behavior === 'BENCH_SEATED') {
      const pelvisY = npc.worldY + 0.056 * hScale; // Seated flush on teak bench slats!
      const act = npc.seatActivity;

      let spinePitch = 0.06;
      let spineYaw = 0.0;
      let headPitch = 0.04;
      let headYaw = 0.0;
      let shL = { rx: -0.35, ry: 0.0, rz: 0.08, elRx: -0.95, wrRx: -0.08 };
      let shR = { rx: -0.35, ry: 0.0, rz: -0.08, elRx: -0.95, wrRx: -0.08 };

      if (act === 'READ_OPEN_BOOK') {
        // Seated scholar holding open two-page textbook & scanning lines left-to-right!
        spinePitch = 0.13;
        const readScan = Math.sin(t * 0.95 + idx);
        const lookUpLake = smooth01(0.82, 0.96, Math.sin(t * 0.23 + idx)); // Occasionally gazes up at Danau Kenanga!
        headPitch = (0.30 + 0.03 * Math.sin(t * 1.8)) * (1.0 - lookUpLake) - 0.04 * lookUpLake;
        headYaw = 0.11 * readScan * (1.0 - lookUpLake) + 0.18 * lookUpLake;
        shL = { rx: -0.58, ry: -0.18, rz: 0.14, elRx: -1.12, wrRx: -0.22 };
        shR = { rx: -0.56, ry: 0.18, rz: -0.14, elRx: -1.10 + 0.06 * readScan, wrRx: -0.20 };
      } else if (act === 'SCROLL_PHONE') {
        spinePitch = 0.11;
        const scrollWave = Math.sin(t * 2.4 + idx);
        const glanceAround = smooth01(0.78, 0.95, Math.sin(t * 0.29 + 1.4));
        headPitch = 0.32 * (1.0 - glanceAround);
        headYaw = 0.24 * Math.sin(t * 0.5) * glanceAround;
        shR = { rx: -0.54, ry: 0.20, rz: -0.15, elRx: -1.24 + 0.05 * scrollWave, wrRx: -0.18 };
        shL = { rx: -0.18, ry: -0.08, rz: 0.09, elRx: -0.82, wrRx: 0.05 };
      } else {
        // Two UI friends seated side-by-side on Rotunda bench chatting & pointing at Rektorat Tajug!
        const isLeftFriend = act === 'BENCH_CHAT_LEFT';
        const turnSign = isLeftFriend ? -1.0 : 1.0; // Turn toward each other!
        const speakerTurn = Math.sin(t * 0.65) > 0.0 ? isLeftFriend : !isLeftFriend;
        const talkW = Math.sin(t * 4.6 + idx);
        spinePitch = 0.04;
        spineYaw = turnSign * 0.16;
        headYaw = turnSign * 0.32 + 0.06 * Math.sin(t * 1.3 + idx);
        headPitch = speakerTurn ? -0.05 + 0.05 * talkW : 0.07 * Math.max(0.0, Math.sin(t * 3.8));
        if (speakerTurn && isLeftFriend) {
          // Pointing/gesturing toward Gedung Rektorat UI spire!
          shR = { rx: -0.78 - 0.12 * talkW, ry: 0.10, rz: -0.18, elRx: -0.72 + 0.15 * talkW, wrRx: -0.10 };
          shL = { rx: -0.32, ry: -0.10, rz: 0.10, elRx: -0.92, wrRx: 0.0 };
        } else {
          shL = { rx: -0.46, ry: -0.15, rz: 0.12, elRx: -1.05, wrRx: -0.12 };
          shR = { rx: -0.42 + (speakerTurn ? 0.12 * talkW : 0.0), ry: 0.12, rz: -0.12, elRx: -1.02, wrRx: -0.10 };
        }
      }

      // Override right arm if holding open rain umbrella while seated!
      if (npc.umbrellaBlend > 0.03) {
        const uB = npc.umbrellaBlend;
        shR.rx = shR.rx * (1.0 - uB) + -0.82 * uB;
        shR.rz = shR.rz * (1.0 - uB) + 0.14 * uB;
        shR.elRx = shR.elRx * (1.0 - uB) + -1.02 * uB;
      }

      this._writeJointWorldMatrix(
        idx,
        J_PELVIS,
        -1,
        npc.worldX,
        pelvisY,
        npc.worldZ,
        -0.06,
        npc.yaw,
        0.0,
        hScale,
        hScale,
        hScale
      );
      this._writeJointWorldMatrix(idx, J_WAIST, J_PELVIS, 0.0, 0.105, 0.0, spinePitch * 0.6, spineYaw * 0.5, 0.0);
      this._writeJointWorldMatrix(
        idx,
        J_CHEST,
        J_WAIST,
        0.0,
        0.145,
        0.0,
        spinePitch * 0.4 + 0.014 * breathe,
        spineYaw * 0.5,
        0.0
      );
      this._writeJointWorldMatrix(idx, J_NECK, J_CHEST, 0.0, 0.295, 0.006, headPitch * 0.35, headYaw * 0.35, 0.0);
      this._writeJointWorldMatrix(idx, J_HEAD, J_NECK, 0.0, 0.078, 0.002, headPitch * 0.65, headYaw * 0.65, 0.0);

      // Seated 90-deg Thigh & Knee Posture with subtle living leg relaxation shift
      const legShift = 0.04 * Math.sin(t * 0.42 + idx * 1.9);
      this._writeJointWorldMatrix(idx, J_HIP_L, J_PELVIS, 0.090, -0.030, 0.0, -1.46, 0.05, 0.0);
      this._writeJointWorldMatrix(idx, J_KNEE_L, J_HIP_L, 0.0, -0.400, 0.0, 1.50 + legShift, 0.0, 0.0);
      this._writeJointWorldMatrix(idx, J_ANKLE_L, J_KNEE_L, 0.0, -0.395, 0.0, -0.04 - legShift * 0.5, 0.0, 0.0);
      this._writeJointWorldMatrix(idx, J_TOE_L, J_ANKLE_L, 0.0, -0.043, 0.095, 0.0, 0.0, 0.0);

      this._writeJointWorldMatrix(idx, J_HIP_R, J_PELVIS, -0.090, -0.030, 0.0, -1.46, -0.05, 0.0);
      this._writeJointWorldMatrix(idx, J_KNEE_R, J_HIP_R, 0.0, -0.400, 0.0, 1.56 - legShift, 0.0, 0.0);
      this._writeJointWorldMatrix(idx, J_ANKLE_R, J_KNEE_R, 0.0, -0.395, 0.0, -0.08 + legShift * 0.5, 0.0, 0.0);
      this._writeJointWorldMatrix(idx, J_TOE_R, J_ANKLE_R, 0.0, -0.043, 0.095, 0.0, 0.0, 0.0);

      // Seated Skirt Drape Bone
      this._writeJointWorldMatrix(idx, J_SKIRT_HEM, J_PELVIS, 0.0, -0.44, 0.24, -0.32, 0.0, 0.0);

      // Seated Arms
      this._writeJointWorldMatrix(idx, J_SHOULDER_L, J_CHEST, 0.184, 0.246, -0.006, shL.rx, shL.ry, shL.rz);
      this._writeJointWorldMatrix(idx, J_ELBOW_L, J_SHOULDER_L, 0.0, -0.258, 0.0, shL.elRx, 0.0, 0.0);
      this._writeJointWorldMatrix(idx, J_WRIST_L, J_ELBOW_L, 0.0, -0.236, 0.0, shL.wrRx, 0.0, 0.0);

      this._writeJointWorldMatrix(idx, J_SHOULDER_R, J_CHEST, -0.184, 0.246, -0.006, shR.rx, shR.ry, shR.rz);
      this._writeJointWorldMatrix(idx, J_ELBOW_R, J_SHOULDER_R, 0.0, -0.258, 0.0, shR.elRx, 0.0, 0.0);
      this._writeJointWorldMatrix(idx, J_WRIST_R, J_ELBOW_R, 0.0, -0.236, 0.0, shR.wrRx, 0.0, 0.0);

      this._attachPropsAndUmbrella(npc, windX, windZ);
      return;
    }

    // =========================================================================
    // CASE 2: STANDING CONTRAPPOSTO & CLINICAL 60/40 WALKING / RAIN-HURRY GAIT
    // =========================================================================
    const walkBlend = smooth01(0.06, 0.45, npc.liveSpeed);
    const strideScale = Math.min(1.38, 0.72 + 0.28 * (npc.liveSpeed / 1.15));
    const uL = npc.gaitCycleU;
    const uR = (npc.gaitCycleU + 0.5) % 1.0;

    const gaitL = sampleClinicalLegGait(uL, strideScale);
    const gaitR = sampleClinicalLegGait(uR, strideScale);

    // Contrapposto Asymmetric Idle Stance when standing (switches stance leg smoothly every ~10s!)
    const idleBlend = 1.0 - walkBlend;
    const stanceSwitch = Math.tanh(2.4 * Math.sin(t * 0.31 + idx * 1.7)); // [-1, +1] smooth dwell!
    const idleHipShiftX = idleBlend * stanceSwitch * 0.024;
    const idlePelvisRoll = -idleBlend * stanceSwitch * 0.045;
    const idleFreeL = idleBlend * Math.max(0.0, -stanceSwitch);
    const idleFreeR = idleBlend * Math.max(0.0, stanceSwitch);

    const hipRxL = walkBlend * gaitL.hipRx - idleFreeL * 0.11;
    const kneeRxL = walkBlend * gaitL.kneeRx + idleBlend * 0.05 + idleFreeL * 0.22;
    const footPitchL = walkBlend * gaitL.footPitch + idleFreeL * 0.14;
    const toeRxL = walkBlend * gaitL.toeRx - idleFreeL * 0.14;

    const hipRxR = walkBlend * gaitR.hipRx - idleFreeR * 0.11;
    const kneeRxR = walkBlend * gaitR.kneeRx + idleBlend * 0.05 + idleFreeR * 0.22;
    const footPitchR = walkBlend * gaitR.footPitch + idleFreeR * 0.14;
    const toeRxR = walkBlend * gaitR.toeRx - idleFreeR * 0.14;

    // Forward-lean during walking + brisk rain-hurry lean + centripetal turn banking!
    const pelvisPitch = walkBlend * (0.025 + 0.08 * npc.rainHurryBlend);
    const ankleRxL = footPitchL - (pelvisPitch + hipRxL + kneeRxL);
    const ankleRxR = footPitchR - (pelvisPitch + hipRxR + kneeRxR);

    // === EXACT FORWARD-KINEMATIC LOWEST-FOOT GROUND LOCK (0.0mm Floating, 0.0mm Sinking!) ===
    const evalLowestFootY = (hipRx, kneeRx, footPitch, toeRx) => {
      const a1 = pelvisPitch + hipRx;
      const a2 = a1 + kneeRx;
      const ankleY = -0.030 - 0.400 * Math.cos(a1) - 0.395 * Math.cos(a2);
      // Heel bottom relative to ankle (-0.065m down, -0.035m back)
      const heelY = ankleY - 0.065 * Math.cos(footPitch) + 0.035 * Math.sin(footPitch);
      // Ball of foot relative to ankle (-0.065m down, +0.095m fwd)
      const ballY = ankleY - 0.065 * Math.cos(footPitch) - 0.095 * Math.sin(footPitch);
      // Toe tip relative to ball (+0.070m fwd along footPitch + toeRx)
      const toeY = ballY - 0.070 * Math.sin(footPitch + toeRx) + 0.004;
      return Math.min(heelY, ballY, toeY);
    };

    const minFootLocalY = Math.min(
      evalLowestFootY(hipRxL, kneeRxL, footPitchL, toeRxL),
      evalLowestFootY(hipRxR, kneeRxR, footPitchR, toeRxR)
    );
    const pelvisY = npc.worldY - minFootLocalY * hScale;

    // 3D Pelvis Lateral Weight-Shift (Trendelenburg stance shift) & Transverse Rotation
    const gaitSin = Math.sin(uL * Math.PI * 2.0);
    const walkSwayX = -walkBlend * gaitSin * 0.021; // Shifts over planted stance leg!
    const totalSwayX = walkSwayX + idleHipShiftX;
    const pelvisRollZ = walkBlend * gaitSin * 0.042 + idlePelvisRoll - npc.turnRate * 0.035 * walkBlend;
    const pelvisYaw = npc.yaw - walkBlend * gaitSin * 0.058;

    // Contralateral Spine & Thoracic Counter-Rotation with Viscoelastic Phase Lag!
    const laggedSin = Math.sin(uL * Math.PI * 2.0 - 0.24);
    let waistPitch = walkBlend * (0.032 + 0.09 * npc.rainHurryBlend);
    let waistYaw = walkBlend * laggedSin * 0.068 + npc.turnRate * 0.06;
    const waistRollZ = -pelvisRollZ * 0.62;

    let chestPitch = 0.016 * breathe + 0.05 * npc.rainHurryBlend;
    let chestYaw = walkBlend * laggedSin * 0.062 + npc.turnRate * 0.08;
    const chestRollZ = -pelvisRollZ * 0.32;

    // Vestibular Head Stabilization + Behavior-Specific Gaze & Arm Gestures!
    let targetHeadYaw = npc.turnRate * 0.22; // Head leads corner turns!
    let targetHeadPitch = -waistPitch * 0.45;
    let customLeftArm = null;
    let customRightArm = null;

    if (npc.behavior === 'PAIR_ROTUNDA' || npc.behavior === 'PAIR_SE_WALKWAY') {
      // Conversing while walking side-by-side: periodic head turns toward companion + speaking hand gesture!
      const chatCycle = Math.sin(t * 0.55 + npc.pairId * 2.1);
      const isLookingAtFriend = smooth01(0.15, 0.65, Math.abs(chatCycle));
      const lookTowardPartner = -npc.pairSide * 0.38 * isLookingAtFriend;
      const isSpeaking = (chatCycle > 0.0 && npc.pairSide < 0) || (chatCycle <= 0.0 && npc.pairSide > 0);
      const talkOsc = Math.sin(t * 4.8 + idx);

      targetHeadYaw += lookTowardPartner;
      targetHeadPitch += isSpeaking ? 0.04 * talkOsc : 0.06 * Math.max(0.0, Math.sin(t * 3.6)) * isLookingAtFriend;

      if (isSpeaking && isLookingAtFriend > 0.35 && npc.umbrellaBlend < 0.2 && npc.rainHurryBlend < 0.2) {
        if (npc.pairSide < 0) {
          customRightArm = {
            rx: -0.48 - 0.14 * talkOsc,
            ry: 0.14,
            rz: -0.12,
            elRx: -0.96 + 0.18 * Math.cos(t * 4.1),
            wrRx: -0.12 * talkOsc
          };
        } else {
          customLeftArm = {
            rx: -0.46 - 0.14 * talkOsc,
            ry: -0.14,
            rz: 0.12,
            elRx: -0.94 + 0.18 * Math.cos(t * 4.1),
            wrRx: -0.12 * talkOsc
          };
        }
      }
    } else if (npc.behavior === 'LAKESIDE_STROLL' && walkBlend < 0.25) {
      // Pausing at Danau Kenanga waterfront: serene gaze across water + occasional wristwatch glance
      const checkWatch = smooth01(0.84, 0.96, Math.sin(t * 0.28 + idx));
      targetHeadYaw = 0.24 * Math.sin(t * 0.55 + idx) * (1.0 - checkWatch) + 0.22 * checkWatch;
      targetHeadPitch = 0.06 * (1.0 - checkWatch) + 0.32 * checkWatch;
      if (checkWatch > 0.05) {
        customLeftArm = {
          rx: -0.58 * checkWatch,
          ry: -0.35 * checkWatch,
          rz: 0.10,
          elRx: -0.25 - 1.05 * checkWatch,
          wrRx: -0.20 * checkWatch
        };
      }
    } else if (npc.behavior === 'SEMINAR_CIRCLE') {
      const activeSpeaker = Math.floor(t / 3.8) % npc.memberCount;
      const talkA = Math.sin(t * 5.0 + idx);
      const talkB = Math.cos(t * 3.6 + idx);
      if (activeSpeaker === npc.memberIdx) {
        targetHeadYaw = 0.24 * talkB;
        targetHeadPitch = -0.03 + 0.05 * talkA;
        if (npc.umbrellaBlend < 0.2) {
          customRightArm = {
            rx: -0.56 - 0.16 * talkA,
            ry: 0.14 * talkB,
            rz: -0.14,
            elRx: -1.08 - 0.20 * talkB,
            wrRx: -0.16 * talkA
          };
        }
        customLeftArm = {
          rx: -0.44 + 0.08 * talkB,
          ry: -0.12,
          rz: 0.10,
          elRx: -1.02,
          wrRx: -0.10
        };
      } else {
        // Attentive listener looking toward current speaker & nodding!
        const speakerAz = (activeSpeaker / npc.memberCount) * Math.PI * 2.0 + 0.38;
        const sx = npc.clusterX + Math.cos(speakerAz) * 1.24;
        const sz = npc.clusterZ + Math.sin(speakerAz) * 1.24;
        const yawToSpeaker = wrapAngle(Math.atan2(sx - npc.worldX, sz - npc.worldZ) - npc.yaw);
        targetHeadYaw = Math.max(-0.55, Math.min(0.55, yawToSpeaker * 0.75));
        targetHeadPitch = 0.08 * Math.max(0.0, Math.sin(t * 3.9 + npc.memberIdx * 1.8));
        if (npc.propType === 'OPEN_BOOK' || npc.propType === 'BOOK') {
          customLeftArm = { rx: -0.42, ry: -0.14, rz: 0.09, elRx: -1.08, wrRx: -0.12 };
        }
      }
    } else if (npc.behavior === 'PHOTO_LANDMARK') {
      const photoCycle = ((t * 0.16 + npc.phase0) % 1.0 + 1.0) % 1.0;
      const raiseAmt =
        smooth01(0.04, 0.22, photoCycle) * (1.0 - 0.62 * smooth01(0.72, 0.88, photoCycle));
      targetHeadPitch = -0.21 * raiseAmt + 0.18 * (1.0 - raiseAmt);
      targetHeadYaw = 0.05 * Math.sin(t * 0.8 + idx);
      customRightArm = {
        rx: -0.34 - 0.86 * raiseAmt,
        ry: 0.18 * raiseAmt,
        rz: 0.14 * raiseAmt,
        elRx: -0.82 - 0.26 * (1.0 - raiseAmt),
        wrRx: -0.18 * raiseAmt
      };
      if (npc.umbrellaBlend < 0.2) {
        customLeftArm = {
          rx: -0.24 - 0.64 * raiseAmt,
          ry: -0.20 * raiseAmt,
          rz: -0.12 * raiseAmt,
          elRx: -0.92,
          wrRx: 0.0
        };
      }
    } else {
      // Natural pedestrian sightseeing micro-glances toward Gedung Rektorat UI or Danau Kenanga
      targetHeadYaw += 0.15 * Math.sin(t * 0.52 + idx * 2.3);
    }

    // Blend in Player Eye-Contact & Friendly Campus Nod when user walks close!
    if (npc.playerGlanceBlend > 0.01 && cameraPos) {
      const pB = npc.playerGlanceBlend;
      const dx = cameraPos.x - npc.worldX;
      const dz = cameraPos.z - npc.worldZ;
      const dy = cameraPos.y - (pelvisY + 0.68 * hScale);
      const horizD = Math.hypot(dx, dz) || 1.0;
      const relYaw = Math.max(-0.68, Math.min(0.68, wrapAngle(Math.atan2(dx, dz) - npc.yaw)));
      const relPitch = Math.max(-0.32, Math.min(0.32, -Math.atan2(dy, horizD)));
      const friendlyNod = 0.07 * Math.sin(pB * Math.PI);
      targetHeadYaw = targetHeadYaw * (1.0 - pB) + relYaw * pB;
      targetHeadPitch = targetHeadPitch * (1.0 - pB) + (relPitch + friendlyNod) * pB;
    }

    // === DOUBLE-PENDULUM PHASE-LAGGED ARM, FOREARM & WRIST SWING ===
    // Upper arm swings opposite to same-side leg; forearm lags by 0.34 rad; wrist lags by 0.52 rad!
    const armPhaseL = uL * Math.PI * 2.0;
    const shSinL = Math.sin(armPhaseL); // Opposite to hipRxL (-sin) -> positive swings backward, negative forward!
    const elSinL = Math.sin(armPhaseL - 0.34);
    const wrSinL = Math.sin(armPhaseL - 0.52);

    const shSinR = -shSinL;
    const elSinR = Math.sin(armPhaseL + Math.PI - 0.34);
    const wrSinR = Math.sin(armPhaseL + Math.PI - 0.52);

    const armAmp = 0.36 * strideScale * walkBlend;

    let armL = customLeftArm || {
      rx: -shSinL * armAmp - 0.03,
      ry: -0.09 * Math.max(0.0, shSinL) * walkBlend, // Subtle natural cross-body medial arc!
      rz: 0.065 + 0.012 * breathe,
      elRx: -0.18 - walkBlend * (0.14 + 0.24 * Math.max(0.0, elSinL) * strideScale),
      wrRx: -0.05 - 0.08 * wrSinL * walkBlend
    };

    // If holding a closed BOOK in left hand while walking, carry it naturally crooked against hip/waist!
    if (!customLeftArm && (npc.propType === 'BOOK' || npc.propType === 'OPEN_BOOK')) {
      armL = {
        rx: -0.16 - 0.10 * shSinL * walkBlend,
        ry: -0.14,
        rz: 0.08,
        elRx: -0.82 - 0.08 * elSinL * walkBlend,
        wrRx: -0.08
      };
    }

    let armR = customRightArm || {
      rx: -shSinR * armAmp - 0.03,
      ry: 0.09 * Math.max(0.0, shSinR) * walkBlend,
      rz: -0.065 - 0.012 * breathe,
      elRx: -0.18 - walkBlend * (0.14 + 0.24 * Math.max(0.0, elSinR) * strideScale),
      wrRx: -0.05 - 0.08 * wrSinR * walkBlend
    };

    // Weather Adaptation A: Holding Open 3D Rain Umbrella Overhead!
    if (npc.umbrellaBlend > 0.02) {
      const uB = npc.umbrellaBlend;
      const useLeftForUmbrella = npc.propType === 'PHONE';
      if (useLeftForUmbrella) {
        armL = {
          rx: armL.rx * (1.0 - uB) + -0.80 * uB,
          ry: armL.ry * (1.0 - uB) + -0.22 * uB,
          rz: armL.rz * (1.0 - uB) + -0.06 * uB,
          elRx: armL.elRx * (1.0 - uB) + -1.04 * uB,
          wrRx: armL.wrRx * (1.0 - uB) + 0.14 * uB
        };
      } else {
        armR = {
          rx: armR.rx * (1.0 - uB) + -0.80 * uB,
          ry: armR.ry * (1.0 - uB) + 0.22 * uB,
          rz: armR.rz * (1.0 - uB) + 0.06 * uB,
          elRx: armR.elRx * (1.0 - uB) + -1.04 * uB,
          wrRx: armR.wrRx * (1.0 - uB) + 0.14 * uB
        };
      }
    }

    // Weather Adaptation B: Un-umbrellaed Pedestrian Shielding Forehead/Head while Hurrying in Rain!
    if (npc.rainHurryBlend > 0.03) {
      const hB = npc.rainHurryBlend;
      armL = {
        rx: armL.rx * (1.0 - hB) + -1.32 * hB,
        ry: armL.ry * (1.0 - hB) + -0.36 * hB,
        rz: armL.rz * (1.0 - hB) + 0.18 * hB,
        elRx: armL.elRx * (1.0 - hB) + -1.48 * hB,
        wrRx: armL.wrRx * (1.0 - hB) + -0.25 * hB
      };
    }

    // === WRITE ALL 22 HIERARCHICAL WORLD & BIND-RELATIVE SKIN MATRICES ===
    this._writeJointWorldMatrix(
      idx,
      J_PELVIS,
      -1,
      npc.worldX + Math.cos(npc.yaw) * totalSwayX,
      pelvisY,
      npc.worldZ - Math.sin(npc.yaw) * totalSwayX,
      pelvisPitch,
      pelvisYaw,
      pelvisRollZ,
      hScale,
      hScale,
      hScale
    );

    this._writeJointWorldMatrix(idx, J_WAIST, J_PELVIS, 0.0, 0.105, 0.0, waistPitch, waistYaw, waistRollZ);
    this._writeJointWorldMatrix(idx, J_CHEST, J_WAIST, 0.0, 0.145, 0.0, chestPitch, chestYaw, chestRollZ);

    // Cervical Spine (38%) + Atlanto-Occipital Head (62%) Split + Vestibular Roll Cancellation!
    const vestRollCancel = -(pelvisRollZ + waistRollZ + chestRollZ) * 0.72;
    this._writeJointWorldMatrix(
      idx,
      J_NECK,
      J_CHEST,
      0.0,
      0.295,
      0.006,
      targetHeadPitch * 0.38,
      targetHeadYaw * 0.38,
      vestRollCancel * 0.45
    );
    this._writeJointWorldMatrix(
      idx,
      J_HEAD,
      J_NECK,
      0.0,
      0.078,
      0.002,
      targetHeadPitch * 0.62,
      targetHeadYaw * 0.62,
      vestRollCancel * 0.55
    );

    // Left Leg + Metatarsal Toe-Roll
    this._writeJointWorldMatrix(idx, J_HIP_L, J_PELVIS, 0.090, -0.030, 0.0, hipRxL, 0.0, -pelvisRollZ * 0.35);
    this._writeJointWorldMatrix(idx, J_KNEE_L, J_HIP_L, 0.0, -0.400, 0.0, kneeRxL, 0.0, 0.0);
    this._writeJointWorldMatrix(idx, J_ANKLE_L, J_KNEE_L, 0.0, -0.395, 0.0, ankleRxL, 0.0, 0.0);
    this._writeJointWorldMatrix(idx, J_TOE_L, J_ANKLE_L, 0.0, -0.043, 0.095, toeRxL, 0.0, 0.0);

    // Right Leg + Metatarsal Toe-Roll
    this._writeJointWorldMatrix(idx, J_HIP_R, J_PELVIS, -0.090, -0.030, 0.0, hipRxR, 0.0, -pelvisRollZ * 0.35);
    this._writeJointWorldMatrix(idx, J_KNEE_R, J_HIP_R, 0.0, -0.400, 0.0, kneeRxR, 0.0, 0.0);
    this._writeJointWorldMatrix(idx, J_ANKLE_R, J_KNEE_R, 0.0, -0.395, 0.0, ankleRxR, 0.0, 0.0);
    this._writeJointWorldMatrix(idx, J_TOE_R, J_ANKLE_R, 0.0, -0.043, 0.095, toeRxR, 0.0, 0.0);

    // Dynamic Long-Skirt Hem Follow-Through & Wind Drape Bone (J_SKIRT_HEM = 19)
    const targetSkirtRx = 0.5 * (hipRxL + hipRxR) + 0.06 * walkBlend;
    const localWindLat = Math.cos(npc.yaw) * windX - Math.sin(npc.yaw) * windZ;
    const targetSkirtRz = -pelvisRollZ * 0.5 + localWindLat * 0.045;
    npc.skirtSwingRx = dampScalar(npc.skirtSwingRx, targetSkirtRx, 7.5, dt);
    npc.skirtSwingRz = dampScalar(npc.skirtSwingRz, targetSkirtRz, 7.5, dt);
    this._writeJointWorldMatrix(
      idx,
      J_SKIRT_HEM,
      J_PELVIS,
      0.0,
      -0.580,
      -0.02 * walkBlend,
      npc.skirtSwingRx,
      0.0,
      npc.skirtSwingRz
    );

    // Left & Right Arms + 5-Finger Hands
    this._writeJointWorldMatrix(idx, J_SHOULDER_L, J_CHEST, 0.184, 0.246, -0.006, armL.rx, armL.ry, armL.rz);
    this._writeJointWorldMatrix(idx, J_ELBOW_L, J_SHOULDER_L, 0.0, -0.258, 0.0, armL.elRx, 0.0, 0.0);
    this._writeJointWorldMatrix(idx, J_WRIST_L, J_ELBOW_L, 0.0, -0.236, 0.0, armL.wrRx, 0.0, 0.0);

    this._writeJointWorldMatrix(idx, J_SHOULDER_R, J_CHEST, -0.184, 0.246, -0.006, armR.rx, armR.ry, armR.rz);
    this._writeJointWorldMatrix(idx, J_ELBOW_R, J_SHOULDER_R, 0.0, -0.258, 0.0, armR.elRx, 0.0, 0.0);
    this._writeJointWorldMatrix(idx, J_WRIST_R, J_ELBOW_R, 0.0, -0.236, 0.0, armR.wrRx, 0.0, 0.0);

    this._attachPropsAndUmbrella(npc, windX, windZ, walkBlend, gaitSin);
  }

  /**
   * Attaches Main Academic Prop (`J_PROP_MAIN = 20`) and Dynamic Wind-Tilted Rain Umbrella (`J_PROP_UMBRELLA = 21`).
   */
  _attachPropsAndUmbrella(npc, windX, windZ, walkBlend = 0.0, gaitSin = 0.0) {
    const idx = npc.index;

    // 1. Main Academic Prop
    if (npc.propType === 'PHONE') {
      this._writeJointWorldMatrix(idx, J_PROP_MAIN, J_WRIST_R, 0.024, -0.055, 0.052, 0.22, 0.0, 0.0);
    } else if (npc.propType === 'OPEN_BOOK') {
      this._writeJointWorldMatrix(idx, J_PROP_MAIN, J_WRIST_L, -0.085, -0.045, 0.075, 0.35, -0.15, -0.22);
    } else if (npc.propType === 'BOOK') {
      this._writeJointWorldMatrix(idx, J_PROP_MAIN, J_WRIST_L, -0.020, -0.055, 0.045, 0.14, 0.0, 0.10);
    } else {
      // Backpack anchored on back (-Z) of CHEST with subtle secondary gait inertia bounce!
      const bagBounce = Math.abs(gaitSin) * 0.006 * walkBlend;
      this._writeJointWorldMatrix(
        idx,
        J_PROP_MAIN,
        J_CHEST,
        0.0,
        0.132 + bagBounce,
        -0.142,
        -0.04 + 0.02 * bagBounce,
        0.0,
        0.0
      );
    }

    // 2. Dynamic 3D 8-Rib Rain Umbrella (`J_PROP_UMBRELLA = 21`)
    // Stowed at non-zero 0.003 micro-scale inside opaque torso when dry (guarantees valid unit normals!)
    if (!npc.profile.hasUmbrella || npc.umbrellaBlend < 0.015) {
      this._writeJointWorldMatrix(
        idx,
        J_PROP_UMBRELLA,
        J_CHEST,
        0.0,
        0.12,
        0.0,
        0.0,
        0.0,
        0.0,
        0.003,
        0.003,
        0.003
      );
    } else {
      const uEase = smooth01(0.0, 1.0, npc.umbrellaBlend);
      const handJoint = npc.propType === 'PHONE' ? J_WRIST_L : J_WRIST_R;
      const handMatOff = (idx * JOINTS_PER_NPC + handJoint) * 16;
      const wMats = this._jointWorldMatrices;
      const hx = wMats[handMatOff + 12];
      const hy = wMats[handMatOff + 13];
      const hz = wMats[handMatOff + 14];

      // Anchor handle in world space right at the gripping hand, oriented upright (+Y) and tilted into the wind!
      const hScale = npc.heightScale || 1.0;
      const canopyScale = (0.15 + 0.85 * uEase) * hScale;
      const lateralScale = (0.08 + 0.92 * uEase) * hScale;
      const tiltPitch = Math.max(-0.24, Math.min(0.24, -windZ * 0.16)) * uEase;
      const tiltRoll = Math.max(-0.24, Math.min(0.24, windX * 0.16)) * uEase;
      const handSideOff = npc.propType === 'PHONE' ? -0.02 : 0.02;

      this._writeJointWorldMatrix(
        idx,
        J_PROP_UMBRELLA,
        -1,
        hx + Math.cos(npc.yaw) * handSideOff,
        hy - 0.04 * hScale,
        hz - Math.sin(npc.yaw) * handSideOff,
        tiltPitch,
        npc.yaw,
        tiltRoll,
        lateralScale,
        canopyScale,
        lateralScale
      );
    }
  }

  /**
   * Composes a feline joint's 4x4 live world matrix (in YXZ heading-first Euler order) into
   * `this._jointWorldMatrices` and writes its Bind-Relative Skin Matrix into `this._skinMatrices`.
   */
  _writeCatJointWorldMatrix(
    catIdx,
    jointId,
    parentJointId,
    lx,
    ly,
    lz,
    rx,
    ry,
    rz,
    sx = 1.0,
    sy = 1.0,
    sz = 1.0
  ) {
    this._tmpPos.set(lx, ly, lz);
    this._tmpEuler.set(rx, ry, rz, 'YXZ');
    this._tmpQuat.setFromEuler(this._tmpEuler);
    this._tmpScale.set(sx, sy, sz);
    this._tmpLocalMat.compose(this._tmpPos, this._tmpQuat, this._tmpScale);

    const catGlobalBase = this.npcs.length * JOINTS_PER_NPC + catIdx * CAT_JOINTS_PER_CAT;
    const globalJ = catGlobalBase + jointId;
    const outOffset = globalJ * 16;
    const wMats = this._jointWorldMatrices;

    if (parentJointId < 0) {
      const e = this._tmpLocalMat.elements;
      for (let i = 0; i < 16; i++) wMats[outOffset + i] = e[i];
    } else {
      const parentOffset = (catGlobalBase + parentJointId) * 16;
      for (let i = 0; i < 16; i++) {
        this._tmpParentMat.elements[i] = wMats[parentOffset + i];
      }
      this._tmpWorldMat.multiplyMatrices(this._tmpParentMat, this._tmpLocalMat);
      const e = this._tmpWorldMat.elements;
      for (let i = 0; i < 16; i++) wMats[outOffset + i] = e[i];
    }

    const bp = CAT_BIND_PIVOTS[jointId];
    const bx = bp[0];
    const by = bp[1];
    const bz = bp[2];
    const sMats = this._skinMatrices;
    for (let i = 0; i < 12; i++) {
      sMats[outOffset + i] = wMats[outOffset + i];
    }
    sMats[outOffset + 12] =
      wMats[outOffset + 12] -
      (wMats[outOffset] * bx + wMats[outOffset + 4] * by + wMats[outOffset + 8] * bz);
    sMats[outOffset + 13] =
      wMats[outOffset + 13] -
      (wMats[outOffset + 1] * bx + wMats[outOffset + 5] * by + wMats[outOffset + 9] * bz);
    sMats[outOffset + 14] =
      wMats[outOffset + 14] -
      (wMats[outOffset + 2] * bx + wMats[outOffset + 6] * by + wMats[outOffset + 10] * bz);
    sMats[outOffset + 15] = 1.0;
  }

  /**
   * Advances autonomous feline patrol loops, playful kitten/stalk bursts, terrain-slope alignment,
   * rain huddle reaction, and curious player/human gaze tracking across Pulau U & Pulau I.
   */
  _advanceFelineAI(dt, t, cameraPos = null) {
    const rainInt = this.sharedUniforms?.uRainIntensity?.value ?? 0.0;

    for (let i = 0; i < this.cats.length; i++) {
      const cat = this.cats[i];
      const prevX = cat.worldX;
      const prevZ = cat.worldZ;
      const s = cat.phenotype.overallScale || 1.0;

      if (cat.behavior === 'PROWL_LOOP' && Array.isArray(cat.waypoints)) {
        // Exposed cats pause & crouch during heavy rain unless sheltered
        const sheltered = cat.worldZ < 536.0 && DynamicTrafficAI._isShelteredByRoof(cat.worldX, cat.worldZ);
        if (rainInt > 0.45 && !sheltered) {
          cat.pauseTimer = Math.max(cat.pauseTimer, 0.5);
        }

        if (cat.pauseTimer > 0.0) {
          cat.pauseTimer = Math.max(0.0, cat.pauseTimer - dt);
        } else {
          const wp = cat.waypoints[cat.wpCursor % cat.waypoints.length];
          const dx = wp[0] - cat.worldX;
          const dz = wp[1] - cat.worldZ;
          const dist = Math.hypot(dx, dz);

          if (dist < 0.32) {
            cat.wpCursor = (cat.wpCursor + 1) % cat.waypoints.length;
            cat.pauseTimer = 2.4 + (cat.index % 4) * 0.85;
          } else {
            let dirX = dx / dist;
            let dirZ = dz / dist;

            // Gentle avoidance around humans & player feet
            if (cameraPos && Number.isFinite(cameraPos.x) && cameraPos.y < 3.5) {
              const cx = cat.worldX - cameraPos.x;
              const cz = cat.worldZ - cameraPos.z;
              const cSq = cx * cx + cz * cz;
              if (cSq < 1.44 && cSq > 1e-4) {
                const cd = Math.sqrt(cSq);
                dirX += (cx / cd) * (1.2 - cd);
                dirZ += (cz / cd) * (1.2 - cd);
              }
            }
            const dLen = Math.hypot(dirX, dirZ) || 1.0;
            dirX /= dLen;
            dirZ /= dLen;

            const desiredYaw = Math.atan2(dirX, dirZ);
            cat.yaw = dampAngle(cat.yaw, desiredYaw, 6.5, dt);
            const turnSlow = 1.0 - 0.35 * Math.min(1.0, Math.abs(wrapAngle(desiredYaw - cat.yaw)));
            const stepSpeed = cat.baseSpeed * (0.82 + 0.18 * s) * turnSlow;
            const nx = cat.worldX + Math.sin(cat.yaw) * stepSpeed * dt;
            const nz = cat.worldZ + Math.cos(cat.yaw) * stepSpeed * dt;

            if (
              !TerrainDEMLoader.isFreshwaterExclusion(nx, nz, 1.1) &&
              TerrainDEMLoader.getIslandCoastDistance(nx, nz) < -2.6
            ) {
              cat.worldX = nx;
              cat.worldZ = nz;
            } else {
              cat.wpCursor = (cat.wpCursor + 1) % cat.waypoints.length;
            }
          }
        }
      } else if (cat.behavior === 'PLAY_STALK') {
        // Playful circular scamper burst alternating with low hunter crouch-wiggle!
        const cyc = (t * 0.42 + cat.phaseSeed) % 6.0;
        if (cyc > 3.6 && rainInt < 0.35) {
          const ang = (t * 1.35 + cat.phaseSeed) % (Math.PI * 2.0);
          const rad = 0.65 * s;
          const tx = cat.homeX + Math.cos(ang) * rad;
          const tz = cat.homeZ + Math.sin(ang) * rad;
          if (!TerrainDEMLoader.isFreshwaterExclusion(tx, tz, 1.1)) {
            const targetYaw = Math.atan2(tx - cat.worldX, tz - cat.worldZ);
            if (Math.hypot(tx - cat.worldX, tz - cat.worldZ) > 0.01) {
              cat.yaw = dampAngle(cat.yaw, targetYaw, 8.0, dt);
            }
            cat.worldX = dampScalar(cat.worldX, tx, 4.5, dt);
            cat.worldZ = dampScalar(cat.worldZ, tz, 4.5, dt);
          }
        }
      }

      if (cat.perchY !== undefined) {
        cat.worldY = cat.perchY;
        cat.slopePitch = 0.0;
        cat.slopeRoll = 0.0;
      } else {
        cat.worldY = this._sampleCatSurfaceY(cat.worldX, cat.worldZ);
        // Align feline spine & all 4 paws smoothly to hillside / riverbank terrain slope!
        const probe = 0.20 * s;
        const sinY = Math.sin(cat.yaw);
        const cosY = Math.cos(cat.yaw);
        const yFront = this._sampleCatSurfaceY(cat.worldX + sinY * probe, cat.worldZ + cosY * probe);
        const yBack = this._sampleCatSurfaceY(cat.worldX - sinY * probe, cat.worldZ - cosY * probe);
        const yLeft = this._sampleCatSurfaceY(cat.worldX + cosY * probe, cat.worldZ - sinY * probe);
        const yRight = this._sampleCatSurfaceY(cat.worldX - cosY * probe, cat.worldZ + sinY * probe);
        // Note: rx < 0 pitches +Z nose UPWARD when yFront > yBack!
        const targetPitch = -Math.atan2(yFront - yBack, probe * 2.0);
        const targetRoll = Math.atan2(yRight - yLeft, probe * 2.0);
        cat.slopePitch = dampScalar(cat.slopePitch, Math.max(-0.45, Math.min(0.45, targetPitch)), 10.0, dt);
        cat.slopeRoll = dampScalar(cat.slopeRoll, Math.max(-0.35, Math.min(0.35, targetRoll)), 10.0, dt);
      }

      const disp = dt > 1e-5 ? Math.hypot(cat.worldX - prevX, cat.worldZ - prevZ) : 0.0;
      const instSpeed = dt > 1e-5 && disp < 2.0 ? disp / dt : 0.0;
      cat.liveSpeed = dampScalar(cat.liveSpeed, instSpeed, 14.0, dt);

      // Proportional feline stride integration (1 full 4-paw cycle = ~0.38m * overallScale — zero paw slide!)
      const strideMeters = Math.max(0.18, 0.38 * s);
      if (disp > 1e-5 && disp < 2.0) {
        cat.gaitU = (cat.gaitU + disp / strideMeters) % 1.0;
      }

      // Curious feline head & ear tracking when player approaches within 7.5m!
      let wantLook = 0.0;
      if (cameraPos && Number.isFinite(cameraPos.x)) {
        const dCam = Math.hypot(cameraPos.x - cat.worldX, cameraPos.z - cat.worldZ);
        if (dCam < 7.5 && Math.abs((cameraPos.y || 1.7) - cat.worldY) < 5.0) {
          const relAz = wrapAngle(
            Math.atan2(cameraPos.x - cat.worldX, cameraPos.z - cat.worldZ) - cat.yaw
          );
          if (Math.abs(relAz) < 1.85) {
            wantLook = 1.0 - smooth01(2.5, 7.5, dCam);
          }
        }
      }
      cat.playerLookBlend = dampScalar(cat.playerLookBlend, wantLook, 4.2, dt);
    }
  }

  /**
   * Solves full 18-joint Dual-Bone LBS feline biomechanics across all 5 living postures:
   * PROWL_LOOP (4-beat quadruped walk + alert sniff pause), SIT_ALERT, LOAF_LOUNGE, GROOM_PAW, & PLAY_STALK.
   */
  _solveCatBiomechanics(cat, t, cameraPos = null) {
    const idx = cat.index;
    const s = cat.phenotype.overallScale || 1.0;
    const bp = CAT_BIND_PIVOTS;
    const ph = cat.phaseSeed;
    const rainInt = this.sharedUniforms?.uRainIntensity?.value ?? 0.0;

    const dPivot = (childId, parentId) => [
      bp[childId][0] - bp[parentId][0],
      bp[childId][1] - bp[parentId][1],
      bp[childId][2] - bp[parentId][2]
    ];

    // Curious gaze angle toward player or ambient environment scanning
    let gazeYaw = Math.sin(t * 0.85 + ph) * 0.24 + Math.sin(t * 1.9 + ph * 1.7) * 0.08;
    let gazePitch = Math.sin(t * 1.15 + ph * 0.6) * 0.07;
    if (cat.playerLookBlend > 0.01 && cameraPos && Number.isFinite(cameraPos.x)) {
      const toCamAz = wrapAngle(
        Math.atan2(cameraPos.x - cat.worldX, cameraPos.z - cat.worldZ) - cat.yaw
      );
      const clampedAz = Math.max(-1.05, Math.min(1.05, toCamAz));
      const lookUp = -Math.max(-0.25, Math.min(0.48, ((cameraPos.y || 1.6) - (cat.worldY + 0.25 * s)) * 0.14));
      gazeYaw = gazeYaw * (1.0 - cat.playerLookBlend) + clampedAz * cat.playerLookBlend;
      gazePitch = gazePitch * (1.0 - cat.playerLookBlend) + lookUp * cat.playerLookBlend;
    }

    // Independent feline ear radar twitches!
    const twitchPulseL = Math.pow(Math.max(0.0, Math.sin(t * 2.7 + ph * 3.1)), 12.0);
    const twitchPulseR = Math.pow(Math.max(0.0, Math.sin(t * 3.1 + ph * 1.9 + 1.4)), 14.0);
    let earLRy = twitchPulseL * 0.42 + gazeYaw * 0.22;
    let earRRy = -twitchPulseR * 0.44 + gazeYaw * 0.22;
    let earAirplaneRz = rainInt > 0.35 ? 0.32 : 0.0; // Airplane ears in rain!

    // Default neutral joint rotations
    let pelvisY = cat.worldY + bp[CJ_PELVIS][1] * s;
    let pelvisRx = cat.slopePitch;
    let pelvisRz = cat.slopeRoll;
    let spineRx = 0.0, spineRy = 0.0;
    let chestRx = 0.0, chestRy = 0.0, chestBreath = 1.0 + 0.018 * Math.sin(t * 2.6 + ph);
    let neckRx = 0.0, neckRy = gazeYaw * 0.42;
    let headRx = gazePitch, headRy = gazeYaw * 0.58, headRz = Math.sin(t * 0.9 + ph) * 0.06;

    let tailBaseRx = 0.75, tailBaseRy = 0.0;
    let tailMidRx = -0.22, tailMidRy = 0.0;
    let tailTipRx = -0.32, tailTipRy = 0.0;

    let fShLRx = 0.0, fShLRz = 0.0, fPawLRx = 0.0;
    let fShRRx = 0.0, fShRRz = 0.0, fPawRRx = 0.0;
    let hHipLRx = 0.0, hPawLRx = 0.0;
    let hHipRRx = 0.0, hPawRRx = 0.0;

    const walkBlend = smooth01(0.03, 0.22, cat.liveSpeed);

    if ((cat.behavior === 'PROWL_LOOP' || cat.behavior === 'PLAY_STALK') && walkBlend > 0.02) {
      // --- 4-BEAT QUADRUPED FELINE WALK / TROT BIOMECHANICS ---
      const sampleCatLimb = (phaseOffset, isFore) => {
        const u = ((cat.gaitU + phaseOffset) % 1.0 + 1.0) % 1.0;
        // u in [0, 0.62]: Stance (paw sweeps front -> back: rx goes -0.34 -> +0.34)
        // u in [0.62, 1.0]: Swing (paw lifts & reaches forward: rx goes +0.34 -> -0.34 with carpal/hock flex!)
        if (u < 0.62) {
          const st = u / 0.62;
          const upperRx = (-0.34 + 0.68 * st) * walkBlend;
          const lowerRx = (isFore ? 0.06 : -0.06) * Math.sin(st * Math.PI) * walkBlend;
          return [upperRx, lowerRx];
        }
        const sw = (u - 0.62) / 0.38;
        const cosSw = 0.5 - 0.5 * Math.cos(sw * Math.PI);
        const upperRx = (0.34 - 0.68 * cosSw) * walkBlend;
        const flex = Math.sin(sw * Math.PI) * (isFore ? 0.52 : -0.46) * walkBlend;
        return [upperRx, flex];
      };

      // True lateral-sequence feline 4-beat gait: LH (0.00) -> LF (0.26) -> RH (0.50) -> RF (0.76)
      [hHipLRx, hPawLRx] = sampleCatLimb(0.0, false);
      [fShLRx, fPawLRx] = sampleCatLimb(0.26, true);
      [hHipRRx, hPawRRx] = sampleCatLimb(0.50, false);
      [fShRRx, fPawRRx] = sampleCatLimb(0.76, true);

      const bob = Math.sin(cat.gaitU * Math.PI * 4.0) * 0.006 * s * walkBlend;
      pelvisY += bob;
      pelvisRz += Math.sin(cat.gaitU * Math.PI * 2.0) * 0.045 * walkBlend;
      spineRy = Math.sin(cat.gaitU * Math.PI * 2.0) * 0.05 * walkBlend;
      chestRy = -spineRy * 0.8;

      // Happy upright S-curved "question-mark" greeting tail swaying with stride!
      tailBaseRx = 0.98;
      tailMidRx = -0.24;
      tailTipRx = -0.38;
      tailBaseRy = Math.sin(t * 3.2 + ph) * 0.18;
      tailMidRy = Math.sin(t * 3.2 + ph - 0.6) * 0.22;
      tailTipRy = Math.sin(t * 3.2 + ph - 1.2) * 0.25;
    } else if (cat.behavior === 'PLAY_STALK') {
      // --- LOW HUNTER CROUCH & EXCITED POUNCE-WIGGLE! ---
      const wiggle = Math.sin(t * 14.0 + ph) * smooth01(1.4, 2.6, (t * 0.42 + ph) % 6.0);
      pelvisY = cat.worldY + 0.135 * s;
      pelvisRx = cat.slopePitch + 0.08;
      pelvisRz = cat.slopeRoll + wiggle * 0.08;
      spineRx = 0.06;
      chestRx = -0.10;
      neckRx = -0.14;
      headRx = gazePitch - 0.05;
      fShLRx = 0.38; fPawLRx = -0.74;
      fShRRx = 0.38; fPawRRx = -0.74;
      hHipLRx = -0.42; hPawLRx = 0.78;
      hHipRRx = -0.42; hPawRRx = 0.78;
      tailBaseRx = 0.12;
      tailMidRy = Math.sin(t * 6.5 + ph) * 0.38;
      tailTipRy = Math.sin(t * 6.5 + ph - 0.7) * 0.48;
      earAirplaneRz = 0.16;
    } else if (cat.behavior === 'LOAF_LOUNGE') {
      // --- COZY INDONESIAN CAMPUS "KUCING ROTI" (TUCKED CAT-LOAF / SUNBATHING SPHINX)! ---
      pelvisY = cat.worldY + 0.082 * s;
      pelvisRx = cat.slopePitch - 0.03;
      spineRx = 0.05;
      chestRx = -0.04;
      chestBreath = 1.0 + 0.026 * Math.sin(t * 2.1 + ph);
      // Drowsy slow-blinking head bob that perks up when player comes close!
      const drowsyNod = (1.0 - cat.playerLookBlend) * (0.10 + 0.05 * Math.sin(t * 0.75 + ph));
      neckRx = 0.04;
      headRx = gazePitch + drowsyNod;
      // Folded cat-loaf paws tucked neatly underneath chest & haunches
      fShLRx = 0.64; fPawLRx = -1.38;
      fShRRx = 0.64; fPawRRx = -1.38;
      hHipLRx = -0.68; hPawLRx = 1.28;
      hHipRRx = -0.68; hPawRRx = 1.28;
      // Tail curled gracefully around the right/left flank
      const curlSide = cat.index % 2 === 0 ? 1.0 : -1.0;
      tailBaseRx = -0.14;
      tailBaseRy = curlSide * 0.62;
      tailMidRy = curlSide * 0.74;
      tailTipRy = curlSide * (0.58 + 0.14 * Math.sin(t * 1.6 + ph));
    } else if (cat.behavior === 'SIT_ALERT' || cat.behavior === 'GROOM_PAW' || walkBlend <= 0.02) {
      // --- REGAL UPRIGHT SEATED CAT (+ PAW/FACE-WASHING GROOM CYCLE!) ---
      pelvisY = cat.worldY + 0.092 * s;
      pelvisRx = cat.slopePitch - 0.58; // Tilts chest upward while rump sits firmly on ground/bench!
      spineRx = 0.10;
      chestRx = 0.14;
      neckRx = 0.18;
      headRx = gazePitch + 0.16;

      // Hind haunches folded cleanly forward onto ground; front legs planted straight down!
      hHipLRx = 0.52; hPawLRx = -0.46;
      hHipRRx = 0.52; hPawRRx = -0.46;
      fShLRx = 0.34; fPawLRx = -0.04;
      fShRRx = 0.34; fPawRRx = -0.04;

      // Tail resting near ground with expressive rhythmic tip flicks
      tailBaseRx = -0.44;
      tailMidRx = 0.36;
      tailTipRx = 0.12;
      tailMidRy = Math.sin(t * 2.1 + ph) * 0.26;
      tailTipRy = Math.sin(t * 2.1 + ph - 0.7) * 0.38;

      if (cat.behavior === 'GROOM_PAW') {
        const groomBout = smooth01(0.15, 0.45, 0.5 + 0.5 * Math.sin(t * 0.55 + ph));
        if (groomBout > 0.02) {
          const lick = Math.sin(t * 10.5 + ph);
          // Lift right forepaw to cheek/muzzle while dipping & licking with head!
          fShRRx = 0.34 * (1.0 - groomBout) + (-0.68 + 0.08 * lick) * groomBout;
          fShRRz = 0.30 * groomBout;
          fPawRRx = -0.58 * groomBout;
          headRx = headRx * (1.0 - groomBout) + (0.42 + 0.13 * lick) * groomBout;
          headRy = headRy * (1.0 - groomBout) + -0.22 * groomBout;
          headRz = 0.18 * groomBout;
        }
      }
    }

    // Write hierarchical 18-joint feline skeleton (root CJ_PELVIS carries uniform biological scale `s`!)
    this._writeCatJointWorldMatrix(
      idx,
      CJ_PELVIS,
      -1,
      cat.worldX,
      pelvisY,
      cat.worldZ,
      pelvisRx,
      cat.yaw,
      pelvisRz,
      s,
      s,
      s
    );

    const dSpine = dPivot(CJ_SPINE_MID, CJ_PELVIS);
    this._writeCatJointWorldMatrix(idx, CJ_SPINE_MID, CJ_PELVIS, dSpine[0], dSpine[1], dSpine[2], spineRx, spineRy, 0.0);

    const dChest = dPivot(CJ_CHEST, CJ_SPINE_MID);
    this._writeCatJointWorldMatrix(
      idx,
      CJ_CHEST,
      CJ_SPINE_MID,
      dChest[0],
      dChest[1],
      dChest[2],
      chestRx,
      chestRy,
      0.0,
      chestBreath,
      chestBreath,
      1.0
    );

    const dNeck = dPivot(CJ_NECK, CJ_CHEST);
    this._writeCatJointWorldMatrix(idx, CJ_NECK, CJ_CHEST, dNeck[0], dNeck[1], dNeck[2], neckRx, neckRy, 0.0);

    const dHead = dPivot(CJ_HEAD, CJ_NECK);
    this._writeCatJointWorldMatrix(idx, CJ_HEAD, CJ_NECK, dHead[0], dHead[1], dHead[2], headRx, headRy, headRz);

    const dEarL = dPivot(CJ_EAR_L, CJ_HEAD);
    this._writeCatJointWorldMatrix(idx, CJ_EAR_L, CJ_HEAD, dEarL[0], dEarL[1], dEarL[2], 0.0, earLRy, -earAirplaneRz);

    const dEarR = dPivot(CJ_EAR_R, CJ_HEAD);
    this._writeCatJointWorldMatrix(idx, CJ_EAR_R, CJ_HEAD, dEarR[0], dEarR[1], dEarR[2], 0.0, earRRy, earAirplaneRz);

    const dTail0 = dPivot(CJ_TAIL_BASE, CJ_PELVIS);
    this._writeCatJointWorldMatrix(idx, CJ_TAIL_BASE, CJ_PELVIS, dTail0[0], dTail0[1], dTail0[2], tailBaseRx, tailBaseRy, 0.0);

    const dTail1 = dPivot(CJ_TAIL_MID, CJ_TAIL_BASE);
    this._writeCatJointWorldMatrix(idx, CJ_TAIL_MID, CJ_TAIL_BASE, dTail1[0], dTail1[1], dTail1[2], tailMidRx, tailMidRy, 0.0);

    const dTail2 = dPivot(CJ_TAIL_TIP, CJ_TAIL_MID);
    this._writeCatJointWorldMatrix(idx, CJ_TAIL_TIP, CJ_TAIL_MID, dTail2[0], dTail2[1], dTail2[2], tailTipRx, tailTipRy, 0.0);

    const dFShL = dPivot(CJ_FRONT_SHOULDER_L, CJ_CHEST);
    this._writeCatJointWorldMatrix(idx, CJ_FRONT_SHOULDER_L, CJ_CHEST, dFShL[0], dFShL[1], dFShL[2], fShLRx, 0.0, fShLRz);

    const dFPawL = dPivot(CJ_FRONT_PAW_L, CJ_FRONT_SHOULDER_L);
    this._writeCatJointWorldMatrix(idx, CJ_FRONT_PAW_L, CJ_FRONT_SHOULDER_L, dFPawL[0], dFPawL[1], dFPawL[2], fPawLRx, 0.0, 0.0);

    const dFShR = dPivot(CJ_FRONT_SHOULDER_R, CJ_CHEST);
    this._writeCatJointWorldMatrix(idx, CJ_FRONT_SHOULDER_R, CJ_CHEST, dFShR[0], dFShR[1], dFShR[2], fShRRx, 0.0, fShRRz);

    const dFPawR = dPivot(CJ_FRONT_PAW_R, CJ_FRONT_SHOULDER_R);
    this._writeCatJointWorldMatrix(idx, CJ_FRONT_PAW_R, CJ_FRONT_SHOULDER_R, dFPawR[0], dFPawR[1], dFPawR[2], fPawRRx, 0.0, 0.0);

    const dHHipL = dPivot(CJ_HIND_HIP_L, CJ_PELVIS);
    this._writeCatJointWorldMatrix(idx, CJ_HIND_HIP_L, CJ_PELVIS, dHHipL[0], dHHipL[1], dHHipL[2], hHipLRx, 0.0, 0.0);

    const dHPawL = dPivot(CJ_HIND_PAW_L, CJ_HIND_HIP_L);
    this._writeCatJointWorldMatrix(idx, CJ_HIND_PAW_L, CJ_HIND_HIP_L, dHPawL[0], dHPawL[1], dHPawL[2], hPawLRx, 0.0, 0.0);

    const dHHipR = dPivot(CJ_HIND_HIP_R, CJ_PELVIS);
    this._writeCatJointWorldMatrix(idx, CJ_HIND_HIP_R, CJ_PELVIS, dHHipR[0], dHHipR[1], dHHipR[2], hHipRRx, 0.0, 0.0);

    const dHPawR = dPivot(CJ_HIND_PAW_R, CJ_HIND_HIP_R);
    this._writeCatJointWorldMatrix(idx, CJ_HIND_PAW_R, CJ_HIND_HIP_R, dHPawR[0], dHPawR[1], dHPawR[2], hPawRRx, 0.0, 0.0);
  }

  /**
   * Advances all 24 Sivitas Akademika NPCs & executes fast Dual-Bone Linear-Blend Skinning (LBS)
   * across all 528 articulated joints into 1 single unified Mesh (< 0.7ms CPU, 0 GC, 1 Draw Call!).
   *
   * @param {number} dt - Frame delta time in seconds
   * @param {number} [elapsedTime] - Optional absolute elapsed time in seconds
   * @param {THREE.Vector3|null} [cameraPos] - Optional live camera position for eye-contact & avoidance
   */
  update(dt = 0.016, elapsedTime = undefined, cameraPos = null) {
    if (!this.mesh || !this._restPos) return;

    const safeDt = Number.isFinite(dt) ? Math.max(0.0005, Math.min(0.1, dt)) : 0.016;
    if (Number.isFinite(elapsedTime)) {
      this.elapsedTime = elapsedTime;
    } else {
      this.elapsedTime += safeDt;
    }

    const t = this.elapsedTime;
    const npcCount = this.npcs.length;
    const catCount = this.cats.length;

    // 1. Advance Crowd & Feline Navigation, Social Avoidance, & Weather State
    this._advanceCrowdNavigation(safeDt, t, cameraPos);
    this._advanceFelineAI(safeDt, t, cameraPos);

    // 2. Solve Full 22-Joint Biomechanics for All 24 Humans + 18-Joint Biomechanics for All 48 Cats
    for (let n = 0; n < npcCount; n++) {
      this._solveNPCBiomechanics(this.npcs[n], t, safeDt, cameraPos);
    }
    for (let c = 0; c < catCount; c++) {
      this._solveCatBiomechanics(this.cats[c], t, cameraPos);
    }

    // 3. Pre-normalize 3x3 Normal Matrices for all (24 x 22 + 48 x 18 = 1,392) Live Skin Matrices
    const sMats = this._skinMatrices;
    const nMats = this._skinNormalMatrices;
    const totalJoints = npcCount * JOINTS_PER_NPC + catCount * CAT_JOINTS_PER_CAT;

    for (let j = 0; j < totalJoints; j++) {
      const mOff = j * 16;
      const nOff = j * 9;
      const c0x = sMats[mOff];
      const c0y = sMats[mOff + 1];
      const c0z = sMats[mOff + 2];
      const c1x = sMats[mOff + 4];
      const c1y = sMats[mOff + 5];
      const c1z = sMats[mOff + 6];
      const c2x = sMats[mOff + 8];
      const c2y = sMats[mOff + 9];
      const c2z = sMats[mOff + 10];

      const inv0 = 1.0 / (Math.sqrt(c0x * c0x + c0y * c0y + c0z * c0z) || 1.0);
      const inv1 = 1.0 / (Math.sqrt(c1x * c1x + c1y * c1y + c1z * c1z) || 1.0);
      const inv2 = 1.0 / (Math.sqrt(c2x * c2x + c2y * c2y + c2z * c2z) || 1.0);

      nMats[nOff] = c0x * inv0;
      nMats[nOff + 1] = c0y * inv0;
      nMats[nOff + 2] = c0z * inv0;
      nMats[nOff + 3] = c1x * inv1;
      nMats[nOff + 4] = c1y * inv1;
      nMats[nOff + 5] = c1z * inv1;
      nMats[nOff + 6] = c2x * inv2;
      nMats[nOff + 7] = c2y * inv2;
      nMats[nOff + 8] = c2z * inv2;
    }

    // 4. Fast Register-Cached Dual-Bone Linear-Blend Skinning Loop (0 GC, 1 Draw Call!)
    const restPos = this._restPos;
    const restNorm = this._restNorm;
    const livePos = this._livePos;
    const liveNorm = this._liveNorm;
    const vertJ0 = this._vertJ0;
    const vertJ1 = this._vertJ1;
    const vertW0 = this._vertW0;
    const vCount = vertJ0.length;

    let cachedJ0 = -1;
    let s00 = 0, s01 = 0, s02 = 0, s04 = 0, s05 = 0, s06 = 0, s08 = 0, s09 = 0, s10 = 0, s12 = 0, s13 = 0, s14 = 0;
    let n00 = 0, n01 = 0, n02 = 0, n03 = 0, n04 = 0, n05 = 0, n06 = 0, n07 = 0, n08 = 0;

    for (let i = 0; i < vCount; i++) {
      const j0 = vertJ0[i];
      if (j0 !== cachedJ0) {
        cachedJ0 = j0;
        const m0 = j0 * 16;
        const n0 = j0 * 9;
        s00 = sMats[m0];     s01 = sMats[m0 + 1]; s02 = sMats[m0 + 2];
        s04 = sMats[m0 + 4]; s05 = sMats[m0 + 5]; s06 = sMats[m0 + 6];
        s08 = sMats[m0 + 8]; s09 = sMats[m0 + 9]; s10 = sMats[m0 + 10];
        s12 = sMats[m0 + 12]; s13 = sMats[m0 + 13]; s14 = sMats[m0 + 14];

        n00 = nMats[n0];     n01 = nMats[n0 + 1]; n02 = nMats[n0 + 2];
        n03 = nMats[n0 + 3]; n04 = nMats[n0 + 4]; n05 = nMats[n0 + 5];
        n06 = nMats[n0 + 6]; n07 = nMats[n0 + 7]; n08 = nMats[n0 + 8];
      }

      const w0 = vertW0[i];
      const i3 = i * 3;
      const lx = restPos[i3];
      const ly = restPos[i3 + 1];
      const lz = restPos[i3 + 2];
      const nx = restNorm[i3];
      const ny = restNorm[i3 + 1];
      const nz = restNorm[i3 + 2];

      const px0 = s00 * lx + s04 * ly + s08 * lz + s12;
      const py0 = s01 * lx + s05 * ly + s09 * lz + s13;
      const pz0 = s02 * lx + s06 * ly + s10 * lz + s14;

      const rnx0 = n00 * nx + n03 * ny + n06 * nz;
      const rny0 = n01 * nx + n04 * ny + n07 * nz;
      const rnz0 = n02 * nx + n05 * ny + n08 * nz;

      if (w0 > 0.999 || j0 === vertJ1[i]) {
        livePos[i3] = px0;
        livePos[i3 + 1] = py0;
        livePos[i3 + 2] = pz0;
        liveNorm[i3] = rnx0;
        liveNorm[i3 + 1] = rny0;
        liveNorm[i3 + 2] = rnz0;
      } else {
        const j1 = vertJ1[i];
        const w1 = 1.0 - w0;
        const m1 = j1 * 16;
        const n1 = j1 * 9;

        const px1 = sMats[m1] * lx + sMats[m1 + 4] * ly + sMats[m1 + 8] * lz + sMats[m1 + 12];
        const py1 = sMats[m1 + 1] * lx + sMats[m1 + 5] * ly + sMats[m1 + 9] * lz + sMats[m1 + 13];
        const pz1 = sMats[m1 + 2] * lx + sMats[m1 + 6] * ly + sMats[m1 + 10] * lz + sMats[m1 + 14];

        livePos[i3] = w0 * px0 + w1 * px1;
        livePos[i3 + 1] = w0 * py0 + w1 * py1;
        livePos[i3 + 2] = w0 * pz0 + w1 * pz1;

        const rnx = w0 * rnx0 + w1 * (nMats[n1] * nx + nMats[n1 + 3] * ny + nMats[n1 + 6] * nz);
        const rny = w0 * rny0 + w1 * (nMats[n1 + 1] * nx + nMats[n1 + 4] * ny + nMats[n1 + 7] * nz);
        const rnz = w0 * rnz0 + w1 * (nMats[n1 + 2] * nx + nMats[n1 + 5] * ny + nMats[n1 + 8] * nz);

        const invN = 1.0 / (Math.sqrt(rnx * rnx + rny * rny + rnz * rnz) || 1.0);
        liveNorm[i3] = rnx * invN;
        liveNorm[i3 + 1] = rny * invN;
        liveNorm[i3 + 2] = rnz * invN;
      }
    }

    const geo = this.mesh.geometry;
    geo.attributes.position.needsUpdate = true;
    geo.attributes.normal.needsUpdate = true;
    if (!geo.boundingSphere) {
      const center = new THREE.Vector3(0, 2, 35);
      geo.boundingSphere = THREE.Sphere ? new THREE.Sphere(center, 150) : { center, radius: 150 };
    }
  }
}
