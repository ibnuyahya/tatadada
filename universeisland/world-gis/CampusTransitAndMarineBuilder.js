import * as THREE from 'three';
import { HUMAN_MAT } from './HumanShaderMaterial.js';

/**
 * NEXUS Omni-Scale Procedural Engine v3.7
 * ./world-gis/CampusTransitAndMarineBuilder.js
 *
 * Pure-Procedural Builders for:
 * 1. Super-Accurate 6-Seater Electric Campus Patrol & Shuttle Golf Cart ("Mobil Golf Listrik UI"):
 *    - 6 Articulated Joints (Chassis, Tilted Steering Wheel, 4 Independent Ackermann/Rolling Wheels)
 *    - Authentic Indonesian Right-Hand-Drive (Setir Kanan) EV Cockpit, OLED Dash, Pedals, Split Windshield,
 *      Side/Rear Mirrors, 3 Contoured Tufted Seat Rows (4 Forward + 2 Rear-Facing Flip Shuttle Seats),
 *      Rear Footrest & Stainless Grab Bar, Dual A-Arm Suspension, 6-Spoke Alloy Wheels + Treaded Tires,
 *      UI Alma Mater Yellow & Pearl-White Livery + Makara Crest, and Photocell LED Headlights & Taillights!
 * 2. Distant Nusantara Outrigger Fishing Boats ("Perahu Nelayan Cadik", 9.5–11.5 m) with Linggi Stemposts,
 *    Kajang Cabin, Bamboo Cadik Outriggers, Caping Fisherman Silhouette, & Warm 2500K Fishing Lanterns.
 * 3. Distant Small Coastal Ships ("Kapal Kecil Pesisir", 22–26 m) with Flared Steel Bow, 2-Tier Wheelhouse,
 *    Radar Mast, & COLREGs Port/Starboard/Masthead Navigation Lights.
 *
 * Merged directly into DynamicTrafficAI's Unified LBS BufferGeometry -> +0 Extra Draw Calls!
 */

export const CART_JOINTS_PER_VEHICLE = 6;
export const CART_J_CHASSIS = 0;
export const CART_J_STEER_WHEEL = 1;
export const CART_J_WHEEL_FL = 2;
export const CART_J_WHEEL_FR = 3;
export const CART_J_WHEEL_RL = 4;
export const CART_J_WHEEL_RR = 5;

export const CART_WHEEL_RADIUS = 0.235;
export const CART_WHEELBASE = 2.14;
export const CART_TRACK_HALF_W = 0.53;

// Bind-pose 3D pivots in local vehicle frame (+Z Forward, +Y Up, +X Left, -X Right / Driver Side)
export const CART_BIND_PIVOTS = [
  [0.0, 0.0, 0.0],                         // 0: CART_J_CHASSIS
  [-0.27, 1.045, 0.745],                   // 1: CART_J_STEER_WHEEL (Right-Hand Drive hub)
  [CART_TRACK_HALF_W, CART_WHEEL_RADIUS, 1.18],   // 2: CART_J_WHEEL_FL (+X Left, +Z Front)
  [-CART_TRACK_HALF_W, CART_WHEEL_RADIUS, 1.18],  // 3: CART_J_WHEEL_FR (-X Right, +Z Front)
  [CART_TRACK_HALF_W, CART_WHEEL_RADIUS, -0.96],  // 4: CART_J_WHEEL_RL (+X Left, -Z Rear)
  [-CART_TRACK_HALF_W, CART_WHEEL_RADIUS, -0.96]  // 5: CART_J_WHEEL_RR (-X Right, -Z Rear)
];

// Exact 6-Seat Local Anchors on the Golf Cart (y = 0.740m seat cushion top, floorboard y = 0.310m)
export const CART_SEAT_SLOTS = [
  { id: 0, role: 'DRIVER', lx: -0.27, ly: 0.74, lz: 0.34, yawOffset: 0.0, sideExit: -1.0 },
  { id: 1, role: 'FRONT_LEFT', lx: 0.27, ly: 0.74, lz: 0.34, yawOffset: 0.0, sideExit: 1.0 },
  { id: 2, role: 'MID_LEFT', lx: 0.27, ly: 0.74, lz: -0.48, yawOffset: 0.0, sideExit: 1.0 },
  { id: 3, role: 'MID_RIGHT', lx: -0.27, ly: 0.74, lz: -0.48, yawOffset: 0.0, sideExit: -1.0 },
  { id: 4, role: 'REAR_LEFT', lx: 0.27, ly: 0.74, lz: -1.16, yawOffset: Math.PI, sideExit: 1.0 },
  { id: 5, role: 'REAR_RIGHT', lx: -0.27, ly: 0.74, lz: -1.16, yawOffset: Math.PI, sideExit: -1.0 }
];

// 7-Joint Articulated UI Inter-Island Electric Bus ("Bus Kuning Listrik Lintas Pulau UI" — 7.8m x 2.38m x 2.96m)
export const BUS_JOINTS_PER_VEHICLE = 7;
export const BUS_J_CHASSIS = 0;
export const BUS_J_WHEEL_FL = 1;
export const BUS_J_WHEEL_FR = 2;
export const BUS_J_WHEEL_RL = 3;
export const BUS_J_WHEEL_RR = 4;
export const BUS_J_STEER_WHEEL = 5;
export const BUS_J_DOOR = 6;

export const BUS_WHEEL_RADIUS = 0.42;
export const BUS_WHEELBASE = 4.20;
export const BUS_TRACK_HALF_W = 1.04;

export const BUS_BIND_PIVOTS = [
  [0.0, 0.0, 0.0],                               // 0: BUS_J_CHASSIS
  [BUS_TRACK_HALF_W, BUS_WHEEL_RADIUS, 2.10],    // 1: BUS_J_WHEEL_FL (+X Left, +Z Front)
  [-BUS_TRACK_HALF_W, BUS_WHEEL_RADIUS, 2.10],   // 2: BUS_J_WHEEL_FR (-X Right, +Z Front)
  [BUS_TRACK_HALF_W, BUS_WHEEL_RADIUS, -2.10],   // 3: BUS_J_WHEEL_RL (+X Left, -Z Rear Dual)
  [-BUS_TRACK_HALF_W, BUS_WHEEL_RADIUS, -2.10],  // 4: BUS_J_WHEEL_RR (-X Right, -Z Rear Dual)
  [-0.68, 1.48, 3.36],                           // 5: BUS_J_STEER_WHEEL (RHD Driver Steering Hub)
  [1.155, 1.54, 3.40]                            // 6: BUS_J_DOOR (Curbside +X Folding/Swinging Door Hinge)
];

// 2-Joint Northern Selat UI Electric Catamaran Shuttle Ferry (Hull + Spinning X-Band Open-Array Radar)
export const FERRY_JOINTS_PER_VESSEL = 2;
export const FERRY_J_HULL = 0;
export const FERRY_J_RADAR = 1;
export const FERRY_BIND_PIVOTS = [
  [0.0, 0.0, 0.0],        // 0: Twin-Hull Catamaran & Saloon
  [0.0, 4.92, 1.85]       // 1: Spinning Marine X-Band Open-Array Radar Scanner
];

// 7-Joint Modern 2-Passenger Electric Helicopter Taxi ("Taxi Helicopter Listrik UI" — 2-Seat Multi-Rotor eVTOL)
export const EVTOL_JOINTS_PER_VEHICLE = 7;
export const EVTOL_J_FUSELAGE = 0;
export const EVTOL_ROTOR_RING_R = 2.02;
export const EVTOL_ROTOR_HUB_Y = 2.34;
export const EVTOL_BIND_PIVOTS = [
  [0.0, 0.0, 0.0],
  [Math.cos(0.0) * EVTOL_ROTOR_RING_R, EVTOL_ROTOR_HUB_Y, Math.sin(0.0) * EVTOL_ROTOR_RING_R],
  [Math.cos(Math.PI / 3) * EVTOL_ROTOR_RING_R, EVTOL_ROTOR_HUB_Y, Math.sin(Math.PI / 3) * EVTOL_ROTOR_RING_R],
  [Math.cos((2 * Math.PI) / 3) * EVTOL_ROTOR_RING_R, EVTOL_ROTOR_HUB_Y, Math.sin((2 * Math.PI) / 3) * EVTOL_ROTOR_RING_R],
  [Math.cos(Math.PI) * EVTOL_ROTOR_RING_R, EVTOL_ROTOR_HUB_Y, Math.sin(Math.PI) * EVTOL_ROTOR_RING_R],
  [Math.cos((4 * Math.PI) / 3) * EVTOL_ROTOR_RING_R, EVTOL_ROTOR_HUB_Y, Math.sin((4 * Math.PI) / 3) * EVTOL_ROTOR_RING_R],
  [Math.cos((5 * Math.PI) / 3) * EVTOL_ROTOR_RING_R, EVTOL_ROTOR_HUB_Y, Math.sin((5 * Math.PI) / 3) * EVTOL_ROTOR_RING_R]
];

// 2-Joint Cessna 172 Skyhawk Campus Shuttle Aircraft ("PK-UIA")
export const CESSNA_JOINTS_PER_AIRCRAFT = 2;
export const CESSNA_J_AIRFRAME = 0;
export const CESSNA_J_PROP = 1;
export const CESSNA_BIND_PIVOTS = [
  [0.0, 0.0, 0.0],       // 0: Airframe (+Z Nose, +Y Up, +X Left Wing)
  [0.0, 1.18, 2.58]      // 1: Spinning McCauley Propeller & Spinner Hub
];

// 6-Joint Citation Learjet Executive Campus Shuttle Jet ("PK-UIB") with Retractable Tricycle Landing Gear!
export const LEARJET_JOINTS_PER_AIRCRAFT = 6;
export const LEARJET_J_AIRFRAME = 0;
export const LEARJET_J_FAN_L = 1;
export const LEARJET_J_FAN_R = 2;
export const LEARJET_J_GEAR_NOSE = 3;
export const LEARJET_J_GEAR_MAIN_L = 4;
export const LEARJET_J_GEAR_MAIN_R = 5;
export const LEARJET_BIND_PIVOTS = [
  [0.0, 0.0, 0.0],        // 0: Airframe (+Z Radome Nose, +Y Up, +X Left Wing)
  [1.46, 1.94, -2.48],    // 1: Port Turbofan Spinning Intake Fan Stage
  [-1.46, 1.94, -2.48],   // 2: Starboard Turbofan Spinning Intake Fan Stage
  [0.0, 0.84, 4.42],      // 3: Retractable Nose Oleo Gear Pivot (folds forward/up into nose well)
  [1.82, 0.88, -0.78],    // 4: Retractable Port Trailing-Link Main Gear Pivot (folds inward into belly bay)
  [-1.82, 0.88, -0.78]    // 5: Retractable Starboard Trailing-Link Main Gear Pivot (folds inward into belly bay)
];

/**
 * Smooths vertex normals across shared 3D vertex positions where adjacent face normals meet
 * within the crease threshold (`cosThreshold = cos(50 deg) ≈ 0.64`), keeping sharp chines,
 * transoms, wingtip endcaps, and airfoil trailing edges crisp while eliminating flat-faceted
 * station bands across curved aircraft fuselages, radomes, wings, and marine hulls.
 * Strictly preserves unit-length normals (`1.000`) and zero NaNs.
 */
function computeCreaseSmoothNormals(posArr, normArr, cosThreshold = 0.64) {
  const vCount = (posArr.length / 3) | 0;
  if (vCount < 3) return;
  const origNorm = new Float32Array(normArr);
  const triWeights = new Float32Array((vCount / 3) | 0);
  for (let t = 0, p = 0; p + 8 < posArr.length; t++, p += 9) {
    const e1x = posArr[p + 3] - posArr[p];
    const e1y = posArr[p + 4] - posArr[p + 1];
    const e1z = posArr[p + 5] - posArr[p + 2];
    const e2x = posArr[p + 6] - posArr[p];
    const e2y = posArr[p + 7] - posArr[p + 1];
    const e2z = posArr[p + 8] - posArr[p + 2];
    const cx = e1y * e2z - e1z * e2y;
    const cy = e1z * e2x - e1x * e2z;
    const cz = e1x * e2y - e1y * e2x;
    triWeights[t] = Math.max(1e-5, Math.hypot(cx, cy, cz));
  }

  const buckets = new Map();
  for (let i = 0; i < vCount; i++) {
    const p = i * 3;
    const key = `${Math.round(posArr[p] * 10000)},${Math.round(posArr[p + 1] * 10000)},${Math.round(posArr[p + 2] * 10000)}`;
    let list = buckets.get(key);
    if (!list) {
      list = [];
      buckets.set(key, list);
    }
    list.push(i);
  }

  for (const list of buckets.values()) {
    const lenList = list.length;
    if (lenList < 2) continue;
    for (let a = 0; a < lenList; a++) {
      const i = list[a];
      const pi = i * 3;
      const nix = origNorm[pi];
      const niy = origNorm[pi + 1];
      const niz = origNorm[pi + 2];
      let sx = 0.0;
      let sy = 0.0;
      let sz = 0.0;
      for (let b = 0; b < lenList; b++) {
        const j = list[b];
        const pj = j * 3;
        const njx = origNorm[pj];
        const njy = origNorm[pj + 1];
        const njz = origNorm[pj + 2];
        if (nix * njx + niy * njy + niz * njz >= cosThreshold) {
          const w = triWeights[(j / 3) | 0];
          sx += njx * w;
          sy += njy * w;
          sz += njz * w;
        }
      }
      const sLen = Math.hypot(sx, sy, sz);
      if (sLen > 1e-6) {
        normArr[pi] = sx / sLen;
        normArr[pi + 1] = sy / sLen;
        normArr[pi + 2] = sz / sLen;
      }
    }
  }
}

/**
 * Builds a watertight, outward-oriented multi-station lofted marine/fuselage hull BufferGeometry
 * from longitudinal stations ordered from Stern (-Z) to Bow (+Z). Zero pyramid cones & zero NaN normals!
 * Each station: `{ z, keelY, bilgeW, bilgeY, chineW, chineY, sheerW, sheerY, deckCrownY }`.
 */
function buildLoftedHullStations(stations, centerX = 0.0, includeDeck = true) {
  const pos = [];
  const norm = [];
  const uv = [];

  const pushTriOutward = (p0, p1, p2, u0, u1, u2, outHintX, outHintY, outHintZ) => {
    const e1x = p1[0] - p0[0];
    const e1y = p1[1] - p0[1];
    const e1z = p1[2] - p0[2];
    const e2x = p2[0] - p0[0];
    const e2y = p2[1] - p0[1];
    const e2z = p2[2] - p0[2];
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const len = Math.hypot(nx, ny, nz);
    if (len < 1e-7) return;
    nx /= len;
    ny /= len;
    nz /= len;
    let a = p0;
    let b = p1;
    let c = p2;
    let ua = u0;
    let ub = u1;
    let uc = u2;
    if (nx * outHintX + ny * outHintY + nz * outHintZ < 0.0) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
      b = p2;
      c = p1;
      ub = u2;
      uc = u1;
    }
    pos.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
    norm.push(nx, ny, nz, nx, ny, nz, nx, ny, nz);
    uv.push(ua[0], ua[1], ub[0], ub[1], uc[0], uc[1]);
  };

  const normStations = stations.map((st) => {
    const chineW = st.chineW ?? st.chineX ?? 0.0;
    const chineY = st.chineY ?? st.keelY;
    const sheerW = st.sheerW ?? st.deckX ?? chineW;
    const sheerY = st.sheerY ?? st.deckY ?? chineY;
    const bilgeW = st.bilgeW ?? chineW * 0.58;
    const bilgeY = st.bilgeY ?? (st.keelY + (chineY - st.keelY) * 0.34);
    const crownY = st.deckCrownY !== undefined ? st.deckCrownY : sheerY + Math.min(0.025, sheerW * 0.035);
    return { z: st.z, keelY: st.keelY, bilgeW, bilgeY, chineW, chineY, sheerW, sheerY, crownY };
  });

  const ringOf = (st) => {
    const pts = [
      [centerX, st.keelY, st.z],                         // 0: Keel
      [centerX - st.bilgeW, st.bilgeY, st.z],            // 1: Stbd Bilge
      [centerX - st.chineW, st.chineY, st.z],            // 2: Stbd Chine
      [centerX - st.sheerW, st.sheerY, st.z]             // 3: Stbd Sheer
    ];
    if (includeDeck) {
      pts.push([centerX, st.crownY, st.z]);              // 4: Deck Center Crown
    }
    pts.push(
      [centerX + st.sheerW, st.sheerY, st.z],            // Port Sheer
      [centerX + st.chineW, st.chineY, st.z],            // Port Chine
      [centerX + st.bilgeW, st.bilgeY, st.z]             // Port Bilge
    );
    return pts;
  };

  const rings = normStations.map(ringOf);
  const nS = rings.length;
  const nP = rings[0].length;

  for (let s = 0; s < nS - 1; s++) {
    const r0 = rings[s];
    const r1 = rings[s + 1];
    const u0 = s / Math.max(1, nS - 1);
    const u1 = (s + 1) / Math.max(1, nS - 1);
    const midY =
      0.25 * (normStations[s].keelY + normStations[s].sheerY + normStations[s + 1].keelY + normStations[s + 1].sheerY);

    for (let k = 0; k < nP; k++) {
      const kNext = (k + 1) % nP;
      if (!includeDeck && k === 3) continue; // Leave open hull top if deck is built separately
      const p00 = r0[k];
      const p10 = r1[k];
      const p11 = r1[kNext];
      const p01 = r0[kNext];
      const v0 = k / nP;
      const v1 = (k + 1) / nP;
      const cx = 0.25 * (p00[0] + p10[0] + p11[0] + p01[0]) - centerX;
      const cy = 0.25 * (p00[1] + p10[1] + p11[1] + p01[1]) - midY;
      const dz = 0.5 * (normStations[s + 1].z - normStations[s].z);
      const flareZ = Math.abs(cx) < 0.04 && cy > 0.0 ? 0.0 : (s > nS * 0.65 ? 0.25 : s < nS * 0.25 ? -0.20 : 0.0);
      pushTriOutward(p00, p10, p11, [u0, v0], [u1, v0], [u1, v1], cx, cy, flareZ * Math.sign(dz || 1));
      pushTriOutward(p00, p11, p01, [u0, v0], [u1, v1], [u0, v1], cx, cy, flareZ * Math.sign(dz || 1));
    }
  }

  // Stern (-Z) and Bow (+Z) watertight transom/stem caps
  for (const [endIdx, outZ] of [[0, -1.0], [nS - 1, 1.0]]) {
    const r = rings[endIdx];
    const st = normStations[endIdx];
    if (st.sheerW > 0.02 || st.chineW > 0.02) {
      const cPt = [centerX, 0.5 * (st.keelY + st.sheerY), st.z];
      for (let k = 0; k < nP; k++) {
        const kNext = (k + 1) % nP;
        if (!includeDeck && k === 3) continue;
        pushTriOutward(cPt, r[k], r[kNext], [0.5, 0.5], [0.0, 0.0], [1.0, 0.0], 0.0, 0.0, outZ);
      }
    }
  }

  computeCreaseSmoothNormals(pos, norm, 0.64);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  return geo;
}

/**
 * Builds a true cambered NACA / Supercritical Airfoil Wing or Tail Surface BufferGeometry!
 * Leading edge faces +Z, trailing edge faces -Z, span runs from root `(rootX, rootY, rootZ)` to tip `(tipX, tipY, tipZ)`.
 */
function buildAirfoilWingPanel(opts = {}) {
  const side = opts.side ?? 1.0;
  const span = opts.span ?? 5.0;
  const rootX = opts.rootX ?? 0.0;
  const rootY = opts.rootY ?? 0.0;
  const rootZ = opts.rootZ ?? 0.0;
  const tipX = opts.tipX !== undefined ? opts.tipX : rootX + side * span;
  const tipY = opts.tipY !== undefined ? opts.tipY : rootY + (opts.dihedralY ?? 0.0);
  const tipZ = opts.tipZ !== undefined ? opts.tipZ : rootZ + (opts.sweepZ ?? 0.0);
  const rootChord = opts.rootChord ?? 1.5;
  const tipChord = opts.tipChord ?? rootChord;
  const rootThickRatio = opts.rootThickRatio ?? opts.thicknessRatio ?? 0.12;
  const tipThickRatio = opts.tipThickRatio ?? opts.thicknessRatio ?? rootThickRatio * 0.92;
  const camberRatio = opts.camberRatio ?? opts.camber ?? 0.0;
  const washoutRad = opts.washoutRad ?? 0.0;
  const spanSteps = opts.spanSteps ?? 5;
  // 14-point closed NACA 4-digit airfoil contour around chord x_c in [0 (LE) .. 1 (TE)]
  const xSamples = [1.0, 0.82, 0.60, 0.38, 0.20, 0.08, 0.02, 0.0, 0.02, 0.08, 0.20, 0.38, 0.60, 0.82];
  const isUpper = [true, true, true, true, true, true, true, true, false, false, false, false, false, false];

  const evalSection = (uSpan) => {
    const sx = rootX + (tipX - rootX) * uSpan;
    const sy = rootY + (tipY - rootY) * uSpan;
    const sz = rootZ + (tipZ - rootZ) * uSpan;
    const chord = rootChord + (tipChord - rootChord) * uSpan;
    const tRatio = rootThickRatio + (tipThickRatio - rootThickRatio) * uSpan;
    const twist = washoutRad * uSpan;
    const cT = Math.cos(twist);
    const sT = Math.sin(twist);

    const pts = [];
    for (let i = 0; i < xSamples.length; i++) {
      const xc = xSamples[i];
      // Standard NACA thickness distribution
      const yt =
        5.0 *
        tRatio *
        (0.2969 * Math.sqrt(xc) - 0.1260 * xc - 0.3516 * xc * xc + 0.2843 * xc * xc * xc - 0.1015 * xc * xc * xc * xc);
      // Parabolic NACA camber line peaking at 40% chord
      const yc =
        xc < 0.4
          ? (camberRatio / 0.16) * (0.8 * xc - xc * xc)
          : (camberRatio / 0.36) * (0.2 + 0.8 * xc - xc * xc);
      const localY = ((i === 7 ? yc : isUpper[i] ? yc + yt : yc - yt * 0.78)) * chord;
      const localZ = (0.35 - xc) * chord; // Quarter-chord aerodynamic center near 0
      const rotY = localY * cT - localZ * sT;
      const rotZ = localY * sT + localZ * cT;
      pts.push([sx, sy + rotY, sz + rotZ]);
    }
    return { pts, center: [sx, sy, sz] };
  };

  const secs = [];
  for (let s = 0; s <= spanSteps; s++) {
    secs.push(evalSection(s / spanSteps));
  }
  if (tipX < rootX) {
    secs.reverse();
  }

  const pos = [];
  const norm = [];
  const uv = [];
  const pushTri = (p0, p1, p2, u0, u1, u2, hx, hy, hz) => {
    const e1x = p1[0] - p0[0];
    const e1y = p1[1] - p0[1];
    const e1z = p1[2] - p0[2];
    const e2x = p2[0] - p0[0];
    const e2y = p2[1] - p0[1];
    const e2z = p2[2] - p0[2];
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const len = Math.hypot(nx, ny, nz);
    if (len < 1e-7) return;
    nx /= len;
    ny /= len;
    nz /= len;
    let a = p0;
    let b = p1;
    let c = p2;
    if (nx * hx + ny * hy + nz * hz < 0.0) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
      b = p2;
      c = p1;
    }
    pos.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
    norm.push(nx, ny, nz, nx, ny, nz, nx, ny, nz);
    uv.push(u0[0], u0[1], u1[0], u1[1], u2[0], u2[1]);
  };

  const nK = secs[0].pts.length;
  for (let s = 0; s < secs.length - 1; s++) {
    const r0 = secs[s];
    const r1 = secs[s + 1];
    const u0 = s / (secs.length - 1);
    const u1 = (s + 1) / (secs.length - 1);
    for (let k = 0; k < nK; k++) {
      const kNext = (k + 1) % nK;
      const p00 = r0.pts[k];
      const p10 = r1.pts[k];
      const p11 = r1.pts[kNext];
      const p01 = r0.pts[kNext];
      const my = 0.25 * (p00[1] + p10[1] + p11[1] + p01[1]) - 0.5 * (r0.center[1] + r1.center[1]);
      const mz = 0.25 * (p00[2] + p10[2] + p11[2] + p01[2]) - 0.5 * (r0.center[2] + r1.center[2]);
      pushTri(p00, p10, p11, [u0, k / nK], [u1, k / nK], [u1, (k + 1) / nK], 0.0, my, mz);
      pushTri(p00, p11, p01, [u0, k / nK], [u1, (k + 1) / nK], [u0, (k + 1) / nK], 0.0, my, mz);
    }
  }
  for (const [eIdx, outX] of [[0, -1.0], [secs.length - 1, 1.0]]) {
    const sec = secs[eIdx];
    for (let k = 0; k < nK; k++) {
      pushTri(sec.center, sec.pts[k], sec.pts[(k + 1) % nK], [0.5, 0.5], [0, 0], [1, 0], outX, 0.0, 0.0);
    }
  }

  computeCreaseSmoothNormals(pos, norm, 0.64);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  return geo;
}

class RigidJointBundleAccumulator {
  constructor() {
    this.pos = [];
    this.norm = [];
    this.col = [];
    this.uv = [];
    this.meta = [];
    this.j0 = [];
    this.j1 = [];
    this.w0 = [];
    this.idx = [];
  }

  addGeo(geo, color, matClass, jointId = 0, ao = 1.0, metaZ = 0.0, fixedUV = null) {
    if (!geo.attributes.normal) {
      geo.computeVertexNormals();
    }
    const p = geo.attributes.position.array;
    const n = geo.attributes.normal.array;
    const u = geo.attributes.uv ? geo.attributes.uv.array : null;
    const vCount = geo.attributes.position.count;
    const base = this.pos.length / 3;

    for (let i = 0; i < vCount; i++) {
      this.pos.push(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
      this.norm.push(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]);
      this.col.push(color[0], color[1], color[2]);
      if (fixedUV) {
        this.uv.push(fixedUV[0], fixedUV[1]);
      } else if (u) {
        this.uv.push(u[i * 2], u[i * 2 + 1]);
      } else {
        this.uv.push(0.5, 0.5);
      }
      this.meta.push(matClass, ao, metaZ);
      this.j0.push(jointId);
      this.j1.push(jointId);
      this.w0.push(1.0);
    }

    if (geo.index) {
      const id = geo.index.array;
      for (let i = 0; i < id.length; i++) {
        this.idx.push(base + id[i]);
      }
    } else {
      for (let i = 0; i < vCount; i++) {
        this.idx.push(base + i);
      }
    }
    geo.dispose();
  }

  toBundle() {
    return {
      pos: new Float32Array(this.pos),
      norm: new Float32Array(this.norm),
      col: new Float32Array(this.col),
      uv: new Float32Array(this.uv),
      meta: new Float32Array(this.meta),
      j0: new Uint8Array(this.j0),
      j1: new Uint8Array(this.j1),
      w0: new Float32Array(this.w0),
      idx: new Uint32Array(this.idx)
    };
  }
}

export class CampusTransitAndMarineBuilder {
  /**
   * Builds a 1:1 Super-Accurate 6-Seater UI Electric Campus Golf Cart ("Mobil Golf Listrik UI")
   * rigged across 6 articulated joints (`CART_J_CHASSIS`, `CART_J_STEER_WHEEL`, `CART_J_WHEEL_FL..RR`).
   */
  static buildElectricGolfCart() {
    const acc = new RigidJointBundleAccumulator();

    const uiYellow = [0.92, 0.69, 0.09];
    const pearlWhite = [0.93, 0.94, 0.96];
    const chassisBlack = [0.11, 0.12, 0.14];
    const rubberDark = [0.085, 0.088, 0.095];
    const alloySilver = [0.76, 0.79, 0.83];
    const chromeSteel = [0.84, 0.86, 0.90];
    const seatCream = [0.83, 0.77, 0.65];
    const seatPiping = [0.22, 0.20, 0.18];
    const dashCharcoal = [0.15, 0.16, 0.19];
    const glassTint = [0.58, 0.72, 0.82];
    const makaraGold = [0.90, 0.70, 0.16];
    const solarNavy = [0.09, 0.16, 0.28];

    // =========================================================================
    // 1. TUBULAR STEEL CHASSIS FRAME, BATTERY BAY, STEERING RACK, & SUSPENSION
    // =========================================================================
    for (const sx of [-0.36, 0.36]) {
      const rail = new THREE.BoxGeometry(0.065, 0.075, 2.95);
      rail.translate(sx, 0.245, -0.05);
      acc.addGeo(rail, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.72);
    }
    for (const cz of [1.18, 0.34, -0.48, -0.96]) {
      const cross = new THREE.BoxGeometry(0.76, 0.045, 0.055);
      cross.translate(0.0, 0.235, cz);
      acc.addGeo(cross, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.70);
    }

    // Anti-slip ribbed rubber passenger floorboard deck (top surface strictly at y = 0.310m)
    const floorDeck = new THREE.BoxGeometry(1.16, 0.045, 2.42);
    floorDeck.translate(0.0, 0.2875, -0.06);
    acc.addGeo(floorDeck, rubberDark, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.82);

    // Subtle rubber floor ribs for close-up inspection realism
    for (let rz = -1.10; rz <= 0.96; rz += 0.18) {
      const rib = new THREE.BoxGeometry(1.08, 0.008, 0.04);
      rib.translate(0.0, 0.312, rz);
      acc.addGeo(rib, [0.13, 0.135, 0.145], HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.88);
    }

    // Side rocker sills in UI Yellow + anodized aluminum scuff trim along floorboard edges
    for (const sx of [-0.59, 0.59]) {
      const sill = new THREE.BoxGeometry(0.045, 0.09, 2.40);
      sill.translate(sx, 0.285, -0.06);
      acc.addGeo(sill, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.94);

      const scuffTrim = new THREE.BoxGeometry(0.022, 0.016, 2.38);
      scuffTrim.translate(sx * 0.985, 0.332, -0.06);
      acc.addGeo(scuffTrim, alloySilver, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.96);
    }

    // Front Steering Rack-and-Pinion Housing + Tie-Rod Linkages + Rear EV PMSM Motor Differential Axle
    const steerRack = new THREE.CylinderGeometry(0.022, 0.022, 0.62, 10, 1);
    steerRack.rotateZ(Math.PI * 0.5);
    steerRack.translate(0.0, 0.24, 1.12);
    acc.addGeo(steerRack, alloySilver, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.78);

    for (const sx of [-1, 1]) {
      const tieRod = new THREE.CylinderGeometry(0.010, 0.010, 0.24, 8, 1);
      tieRod.rotateZ(Math.PI * 0.5);
      tieRod.translate(sx * 0.39, 0.24, 1.14);
      acc.addGeo(tieRod, chromeSteel, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.80);
    }

    const rearAxle = new THREE.CylinderGeometry(0.032, 0.032, 1.02, 12, 1);
    rearAxle.rotateZ(Math.PI * 0.5);
    rearAxle.translate(0.0, CART_WHEEL_RADIUS, -0.96);
    acc.addGeo(rearAxle, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.70);

    const diffGearbox = new THREE.SphereGeometry(0.095, 14, 10);
    diffGearbox.scale(1.15, 0.95, 1.1);
    diffGearbox.translate(0.06, CART_WHEEL_RADIUS, -0.96);
    acc.addGeo(diffGearbox, [0.20, 0.22, 0.25], HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.72);

    const evMotorCan = new THREE.CylinderGeometry(0.078, 0.078, 0.22, 12, 1);
    evMotorCan.rotateZ(Math.PI * 0.5);
    evMotorCan.translate(-0.14, CART_WHEEL_RADIUS + 0.03, -0.84);
    acc.addGeo(evMotorCan, [0.26, 0.29, 0.34], HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.76);

    for (const sx of [-1, 1]) {
      // Front upper & lower A-arm wishbones + steering knuckles
      for (const ay of [0.20, 0.27]) {
        const arm = new THREE.CylinderGeometry(0.014, 0.014, 0.32, 8, 1);
        arm.rotateZ(Math.PI * 0.5);
        arm.translate(sx * 0.37, ay, 1.18);
        acc.addGeo(arm, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.76);
      }
      // Front & Rear coil-over shock struts with helical spring collars
      for (const wz of [1.18, -0.96]) {
        const strut = new THREE.CylinderGeometry(0.024, 0.024, 0.24, 10, 1);
        strut.rotateZ(sx * 0.22);
        strut.translate(sx * 0.43, 0.34, wz);
        acc.addGeo(strut, uiYellow, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.85);
      }
    }

    // =========================================================================
    // 2. SCULPTED UI YELLOW & PEARL-WHITE BODYWORK, COWL, FENDERS & EV PORT
    // =========================================================================
    const frontBumper = new THREE.BoxGeometry(1.22, 0.11, 0.14);
    frontBumper.translate(0.0, 0.27, 1.69);
    acc.addGeo(frontBumper, rubberDark, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.90);

    for (const sx of [-0.59, 0.59]) {
      const bCap = new THREE.CylinderGeometry(0.055, 0.055, 0.11, 10, 1);
      bCap.translate(sx, 0.27, 1.67);
      acc.addGeo(bCap, rubberDark, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.90);
    }

    // Aerodynamic sloped front nose cowl (UI Alma Mater Yellow!)
    const noseLower = new THREE.BoxGeometry(1.16, 0.30, 0.66);
    noseLower.translate(0.0, 0.48, 1.32);
    acc.addGeo(noseLower, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.98);

    const hoodSlope = new THREE.BoxGeometry(1.12, 0.14, 0.68);
    hoodSlope.rotateX(0.30);
    hoodSlope.translate(0.0, 0.74, 1.25);
    acc.addGeo(hoodSlope, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 1.0);

    // Pearl-white center hood accent stripe + Golden UI Makara Emblem Shield on front nose!
    const hoodStripe = new THREE.BoxGeometry(0.38, 0.025, 0.62);
    hoodStripe.rotateX(0.30);
    hoodStripe.translate(0.0, 0.805, 1.26);
    acc.addGeo(hoodStripe, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 1.0);

    const frontCrest = new THREE.CylinderGeometry(0.075, 0.075, 0.022, 16, 1);
    frontCrest.rotateX(Math.PI * 0.5 + 0.30);
    frontCrest.translate(0.0, 0.77, 1.54);
    acc.addGeo(frontCrest, makaraGold, HUMAN_MAT.CREST_GOLD, CART_J_CHASSIS, 1.0, 0.0, [0.5, 0.5]);

    // Slatted front cooling intake grille + chrome surround
    const frontGrille = new THREE.BoxGeometry(0.56, 0.13, 0.03);
    frontGrille.translate(0.0, 0.44, 1.645);
    acc.addGeo(frontGrille, chassisBlack, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.76);
    for (const gy of [0.405, 0.44, 0.475]) {
      const slat = new THREE.BoxGeometry(0.52, 0.010, 0.036);
      slat.translate(0.0, gy, 1.648);
      acc.addGeo(slat, chromeSteel, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.94);
    }

    // Flared Front & Rear Wheel Arch Mudguards
    for (const sx of [-1, 1]) {
      for (const wz of [1.18, -0.96]) {
        const arch = new THREE.TorusGeometry(0.285, 0.042, 8, 16, Math.PI);
        arch.rotateY(Math.PI * 0.5);
        arch.scale(1.0, 0.95, 2.2);
        arch.translate(sx * 0.54, 0.25, wz);
        acc.addGeo(arch, chassisBlack, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.86);
      }
    }

    // Under-Seat Lithium EV Battery Body Tubs (Row 1 + Row 2/3) + CCS2 Charging Port Flap
    const tubFront = new THREE.BoxGeometry(1.10, 0.38, 0.56);
    tubFront.translate(0.0, 0.50, 0.32);
    acc.addGeo(tubFront, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.94);

    const chargePortFlap = new THREE.BoxGeometry(0.018, 0.11, 0.11);
    chargePortFlap.translate(-0.556, 0.52, 0.44);
    acc.addGeo(chargePortFlap, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.95);
    const chargeLed = new THREE.BoxGeometry(0.022, 0.018, 0.06);
    chargeLed.translate(-0.558, 0.56, 0.44);
    acc.addGeo(chargeLed, [0.16, 0.84, 0.42], HUMAN_MAT.VEHICLE_HEADLIGHT, CART_J_CHASSIS, 1.0, 1.0);

    const tubRear = new THREE.BoxGeometry(1.14, 0.39, 1.18);
    tubRear.translate(0.0, 0.505, -0.82);
    acc.addGeo(tubRear, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.94);

    for (const sx of [-1, 1]) {
      const sidePanel = new THREE.BoxGeometry(0.025, 0.18, 1.12);
      sidePanel.translate(sx * 0.575, 0.53, -0.82);
      acc.addGeo(sidePanel, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.98);

      const sideCrest = new THREE.CylinderGeometry(0.065, 0.065, 0.018, 16, 1);
      sideCrest.rotateZ(Math.PI * 0.5);
      sideCrest.translate(sx * 0.588, 0.54, -0.78);
      acc.addGeo(sideCrest, makaraGold, HUMAN_MAT.CREST_GOLD, CART_J_CHASSIS, 1.0, 0.0, [0.5, 0.5]);
    }

    // Rear Fold-Out Diamond-Plate Footrest Deck (for Rear-Facing Row 3!) & Stainless U-Shaped Safety Grab Bar
    const rearFootrest = new THREE.BoxGeometry(1.14, 0.055, 0.36);
    rearFootrest.translate(0.0, 0.26, -1.58);
    acc.addGeo(rearFootrest, [0.24, 0.26, 0.29], HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.88);

    for (const sx of [-0.48, 0.48]) {
      const grabV = new THREE.CylinderGeometry(0.018, 0.018, 0.56, 10, 1);
      grabV.translate(sx, 0.55, -1.72);
      acc.addGeo(grabV, chromeSteel, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.96);
    }
    const grabTop = new THREE.CylinderGeometry(0.018, 0.018, 0.98, 10, 1);
    grabTop.rotateZ(Math.PI * 0.5);
    grabTop.translate(0.0, 0.83, -1.72);
    acc.addGeo(grabTop, chromeSteel, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.98);

    // =========================================================================
    // 3. PROJECTOR DUAL-LENS LED HEADLIGHTS, AMBER CORNERS & RUBY TAILLIGHTS
    // =========================================================================
    for (const sx of [-0.39, 0.39]) {
      const bezel = new THREE.BoxGeometry(0.24, 0.105, 0.042);
      bezel.translate(sx, 0.58, 1.64);
      acc.addGeo(bezel, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.90);

      for (const lx of [-0.042, 0.038]) {
        const proj = new THREE.CylinderGeometry(0.028, 0.028, 0.028, 12, 1);
        proj.rotateX(Math.PI * 0.5);
        proj.translate(sx + lx, 0.58, 1.655);
        acc.addGeo(proj, [0.90, 0.94, 0.98], HUMAN_MAT.VEHICLE_HEADLIGHT, CART_J_CHASSIS, 1.0, 0.0);
      }

      const turnAmber = new THREE.BoxGeometry(0.042, 0.078, 0.032);
      turnAmber.translate(sx + Math.sign(sx) * 0.098, 0.58, 1.648);
      acc.addGeo(turnAmber, [0.94, 0.58, 0.08], HUMAN_MAT.VEHICLE_HEADLIGHT, CART_J_CHASSIS, 1.0, 0.0);
    }

    for (const sx of [-0.46, 0.46]) {
      const tailBezel = new THREE.BoxGeometry(0.20, 0.095, 0.030);
      tailBezel.translate(sx, 0.56, -1.408);
      acc.addGeo(tailBezel, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.90);

      const tailHousing = new THREE.BoxGeometry(0.14, 0.075, 0.038);
      tailHousing.translate(sx - Math.sign(sx) * 0.02, 0.56, -1.412);
      acc.addGeo(tailHousing, [0.56, 0.07, 0.07], HUMAN_MAT.VEHICLE_TAILLIGHT, CART_J_CHASSIS, 1.0, 2.0);

      const revLens = new THREE.BoxGeometry(0.038, 0.075, 0.038);
      revLens.translate(sx + Math.sign(sx) * 0.072, 0.56, -1.412);
      acc.addGeo(revLens, [0.88, 0.90, 0.94], HUMAN_MAT.VEHICLE_GLASS, CART_J_CHASSIS, 0.98);
    }

    // =========================================================================
    // 4. RIGHT-HAND-DRIVE EV COCKPIT, DASHBOARD, OLED SCREEN, PEDALS & STEERING
    // =========================================================================
    const dashBody = new THREE.BoxGeometry(1.10, 0.24, 0.26);
    dashBody.translate(0.0, 0.82, 0.95);
    acc.addGeo(dashBody, dashCharcoal, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.88);

    // Digital EV Speedometer & Battery SOC Display Screen (metaZ = 2.1 -> procedural EV speedometer + SOC arc!)
    const oledBezel = new THREE.BoxGeometry(0.21, 0.108, 0.018);
    oledBezel.rotateX(-0.38);
    oledBezel.translate(-0.27, 0.95, 0.882);
    acc.addGeo(oledBezel, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.95);

    const oledScreen = new THREE.BoxGeometry(0.19, 0.09, 0.022);
    oledScreen.rotateX(-0.38);
    oledScreen.translate(-0.27, 0.95, 0.88);
    acc.addGeo(oledScreen, [0.12, 0.38, 0.46], HUMAN_MAT.VEHICLE_HEADLIGHT, CART_J_CHASSIS, 1.0, 2.1);

    // Center Dashboard Console: Recessed Twin Cupholders + UI Tumbler + F/N/R Drive Rocker + Hazard + Key Switch
    const cupConsole = new THREE.BoxGeometry(0.26, 0.06, 0.17);
    cupConsole.translate(0.01, 0.93, 0.87);
    acc.addGeo(cupConsole, [0.18, 0.19, 0.22], HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.92);

    for (const cxOff of [-0.055, 0.065]) {
      const cupRim = new THREE.TorusGeometry(0.038, 0.006, 6, 14);
      cupRim.rotateX(Math.PI * 0.5);
      cupRim.translate(0.01 + cxOff, 0.962, 0.87);
      acc.addGeo(cupRim, chromeSteel, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.95);
    }
    const tumbler = new THREE.CylinderGeometry(0.032, 0.027, 0.125, 10, 1);
    tumbler.translate(0.075, 0.995, 0.87);
    acc.addGeo(tumbler, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.98);

    const fnrSwitch = new THREE.BoxGeometry(0.032, 0.038, 0.022);
    fnrSwitch.translate(-0.145, 0.94, 0.84);
    acc.addGeo(fnrSwitch, [0.86, 0.20, 0.16], HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.96);

    const keyBarrel = new THREE.CylinderGeometry(0.014, 0.014, 0.022, 8, 1);
    keyBarrel.rotateX(Math.PI * 0.5);
    keyBarrel.translate(-0.145, 0.885, 0.822);
    acc.addGeo(keyBarrel, chromeSteel, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.96);

    // Turn-Signal & Headlight Stalk on Steering Column
    const signalStalk = new THREE.CylinderGeometry(0.007, 0.008, 0.13, 6, 1);
    signalStalk.rotateZ(Math.PI * 0.5 - 0.18);
    signalStalk.translate(-0.34, 0.985, 0.78);
    acc.addGeo(signalStalk, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.92);

    // Front & Rear Campus EV License Plates
    for (const [pz, ry] of [[1.762, 0.0], [-1.762, Math.PI]]) {
      const plate = new THREE.BoxGeometry(0.28, 0.085, 0.012);
      plate.rotateY(ry);
      plate.translate(0.0, 0.30, pz);
      acc.addGeo(plate, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.96);
      const evBlueStrip = new THREE.BoxGeometry(0.282, 0.020, 0.014);
      evBlueStrip.rotateY(ry);
      evBlueStrip.translate(0.0, 0.268, pz);
      acc.addGeo(evBlueStrip, [0.10, 0.38, 0.82], HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.98);
    }

    // Windshield Pantograph Wiper Arm (parked low on left passenger corner outside RHD driver FOV) + Overhead Grab Handles
    const wiperArm = new THREE.BoxGeometry(0.014, 0.34, 0.012);
    wiperArm.rotateZ(0.28);
    wiperArm.rotateX(-0.20);
    wiperArm.translate(0.28, 1.54, 0.942);
    acc.addGeo(wiperArm, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.94);

    for (const sx of [-0.46, 0.46]) {
      for (const gz of [0.35, -0.48]) {
        const roofHandle = new THREE.TorusGeometry(0.055, 0.009, 6, 12, Math.PI);
        roofHandle.rotateY(Math.PI * 0.5);
        roofHandle.rotateX(Math.PI);
        roofHandle.translate(sx, 1.83, gz);
        acc.addGeo(roofHandle, chassisBlack, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.92);
      }
    }

    // Front Left Co-Rider Grab Bar on Dashboard
    const passGrab = new THREE.CylinderGeometry(0.015, 0.015, 0.30, 10, 1);
    passGrab.rotateZ(Math.PI * 0.5);
    passGrab.translate(0.28, 0.92, 0.83);
    acc.addGeo(passGrab, chromeSteel, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.95);

    // Accelerator & Brake Pedals on Driver Side (-X = -0.27m)
    for (const px of [-0.21, -0.33]) {
      const pedal = new THREE.BoxGeometry(0.065, 0.13, 0.02);
      pedal.rotateX(-0.55);
      pedal.translate(px, 0.36, 0.82);
      acc.addGeo(pedal, chassisBlack, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.80);
    }

    // Angled Steering Column Shaft (fixed to chassis, pointing up to [-0.27, 1.045, 0.745])
    const steerTilt = 0.66;
    const colLen = 0.44;
    const steerCol = new THREE.CylinderGeometry(0.026, 0.032, colLen, 12, 1);
    steerCol.rotateX(steerTilt);
    steerCol.translate(
      -0.27,
      1.045 - Math.cos(steerTilt) * colLen * 0.5,
      0.745 + Math.sin(steerTilt) * colLen * 0.5
    );
    acc.addGeo(steerCol, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.88);

    // ARTICULATED JOINT #1: 3-Spoke Sport Steering Wheel (`CART_J_STEER_WHEEL` at [-0.27, 1.045, 0.745])
    const wheelRim = new THREE.TorusGeometry(0.165, 0.019, 10, 24);
    wheelRim.rotateX(Math.PI * 0.5 - steerTilt);
    wheelRim.translate(-0.27, 1.045, 0.745);
    acc.addGeo(wheelRim, rubberDark, HUMAN_MAT.RUBBER_TIRE, CART_J_STEER_WHEEL, 0.96);

    const wheelHub = new THREE.CylinderGeometry(0.048, 0.052, 0.045, 14, 1);
    wheelHub.rotateX(steerTilt);
    wheelHub.translate(-0.27, 1.045, 0.745);
    acc.addGeo(wheelHub, dashCharcoal, HUMAN_MAT.METAL_ACCENT, CART_J_STEER_WHEEL, 0.96);

    const hubCap = new THREE.CylinderGeometry(0.028, 0.028, 0.050, 12, 1);
    hubCap.rotateX(steerTilt);
    hubCap.translate(-0.27, 1.047, 0.743);
    acc.addGeo(hubCap, makaraGold, HUMAN_MAT.CREST_GOLD, CART_J_STEER_WHEEL, 1.0, 0.0, [0.5, 0.5]);

    for (let sp = 0; sp < 3; sp++) {
      const spAng = (sp / 3) * Math.PI * 2.0 + Math.PI * 0.5;
      const spoke = new THREE.BoxGeometry(0.15, 0.016, 0.024);
      spoke.translate(0.08, 0.0, 0.0);
      spoke.rotateZ(spAng);
      spoke.rotateX(Math.PI * 0.5 - steerTilt);
      spoke.translate(-0.27, 1.045, 0.745);
      acc.addGeo(spoke, alloySilver, HUMAN_MAT.METAL_ACCENT, CART_J_STEER_WHEEL, 0.94);
    }

    // =========================================================================
    // 5. THREE CONTOURED TUFTED BENCH SEAT ROWS (6 SEATS TOTAL, TOP Y = 0.740m!)
    // =========================================================================
    const addBenchSeatRow = (cz, facingSign) => {
      // Main plush vinyl seat cushion (top strictly at y = 0.740m)
      const cushion = new THREE.BoxGeometry(1.10, 0.105, 0.46);
      cushion.translate(0.0, 0.6875, cz);
      acc.addGeo(cushion, seatCream, HUMAN_MAT.SHOE_BAG, CART_J_CHASSIS, 0.96);

      // Center seam + thigh bolster ribs (nested at y <= 0.740m so NPC thigh placement never clips)
      for (const bx of [-0.52, 0.0, 0.52]) {
        const seam = new THREE.BoxGeometry(0.024, 0.106, 0.465);
        seam.translate(bx, 0.687, cz);
        acc.addGeo(seam, seatPiping, HUMAN_MAT.SHOE_BAG, CART_J_CHASSIS, 0.86);
      }

      // Curved hip restraint armrests with padded caps on left & right sides
      for (const sx of [-0.555, 0.555]) {
        const armLoop = new THREE.BoxGeometry(0.032, 0.16, 0.36);
        armLoop.translate(sx, 0.80, cz - facingSign * 0.03);
        acc.addGeo(armLoop, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.90);

        const armPad = new THREE.BoxGeometry(0.044, 0.026, 0.28);
        armPad.translate(sx, 0.885, cz - facingSign * 0.03);
        acc.addGeo(armPad, seatPiping, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.94);
      }
    };

    // Row 1 (Front Forward-Facing, z = +0.34m) + Backrest at z = +0.10m
    addBenchSeatRow(0.34, 1.0);
    const back1 = new THREE.BoxGeometry(1.06, 0.30, 0.095);
    back1.rotateX(-0.12);
    back1.translate(0.0, 1.01, 0.10);
    acc.addGeo(back1, seatCream, HUMAN_MAT.SHOE_BAG, CART_J_CHASSIS, 0.97);
    for (const sx of [-0.36, 0.36]) {
      const post1 = new THREE.CylinderGeometry(0.016, 0.016, 0.42, 8, 1);
      post1.translate(sx, 0.86, 0.08);
      acc.addGeo(post1, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.88);
    }

    // Row 2 (Mid Forward-Facing, z = -0.48m) & Row 3 (Rear-Facing Shuttle Row, z = -1.16m)
    addBenchSeatRow(-0.48, 1.0);
    addBenchSeatRow(-1.16, -1.0);

    // Shared Dual-Sided Contoured Backrest Wall between Row 2 and Row 3 (z = -0.82m)
    const back23Core = new THREE.BoxGeometry(1.06, 0.34, 0.05);
    back23Core.translate(0.0, 1.02, -0.82);
    acc.addGeo(back23Core, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.90);

    const back2Pad = new THREE.BoxGeometry(1.05, 0.31, 0.075);
    back2Pad.translate(0.0, 1.02, -0.77);
    acc.addGeo(back2Pad, seatCream, HUMAN_MAT.SHOE_BAG, CART_J_CHASSIS, 0.97);

    const back3Pad = new THREE.BoxGeometry(1.05, 0.31, 0.075);
    back3Pad.translate(0.0, 1.02, -0.87);
    acc.addGeo(back3Pad, seatCream, HUMAN_MAT.SHOE_BAG, CART_J_CHASSIS, 0.97);

    // =========================================================================
    // 6. SPLIT POLYCARBONATE WINDSHIELD, MIRRORS, & SOLAR-ASSISTED CANOPY ROOF
    // =========================================================================
    const aPillarTilt = -0.20;
    for (const sx of [-0.535, 0.535]) {
      // Front A-Pillar Struts
      const pA = new THREE.CylinderGeometry(0.019, 0.019, 1.02, 10, 1);
      pA.rotateX(aPillarTilt);
      pA.translate(sx, 1.35, 0.96);
      acc.addGeo(pA, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.94);

      // Side-View Mirrors with bright reflective chrome mirror faces
      const mArm = new THREE.BoxGeometry(0.11, 0.02, 0.02);
      mArm.translate(sx + Math.sign(sx) * 0.06, 1.32, 0.97);
      acc.addGeo(mArm, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.92);

      const mHead = new THREE.BoxGeometry(0.04, 0.14, 0.09);
      mHead.rotateY(Math.sign(sx) * 0.25);
      mHead.translate(sx + Math.sign(sx) * 0.12, 1.33, 0.96);
      acc.addGeo(mHead, chassisBlack, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.95);

      const mGlass = new THREE.BoxGeometry(0.008, 0.12, 0.076);
      mGlass.rotateY(Math.sign(sx) * 0.25);
      mGlass.translate(sx + Math.sign(sx) * 0.118, 1.33, 0.942);
      acc.addGeo(mGlass, chromeSteel, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 1.0);

      // Mid B-Pillar & Rear C-Pillar Brushed Aluminum Canopy Struts
      for (const pz of [-0.28, -1.42]) {
        const strut = new THREE.CylinderGeometry(0.019, 0.019, 1.14, 10, 1);
        strut.translate(sx, 1.28, pz);
        acc.addGeo(strut, chromeSteel, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.95);
      }
    }

    // Lower & Upper Split Fold-Down Polycarbonate Windshield Panes
    const wsLower = new THREE.BoxGeometry(1.04, 0.36, 0.016);
    wsLower.rotateX(aPillarTilt);
    wsLower.translate(0.0, 1.13, 1.005);
    acc.addGeo(wsLower, glassTint, HUMAN_MAT.VEHICLE_GLASS, CART_J_CHASSIS, 1.0);

    const wsHinge = new THREE.BoxGeometry(1.07, 0.024, 0.026);
    wsHinge.translate(0.0, 1.31, 0.97);
    acc.addGeo(wsHinge, chassisBlack, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.92);

    const wsUpper = new THREE.BoxGeometry(1.04, 0.44, 0.016);
    wsUpper.rotateX(aPillarTilt);
    wsUpper.translate(0.0, 1.54, 0.925);
    acc.addGeo(wsUpper, glassTint, HUMAN_MAT.VEHICLE_GLASS, CART_J_CHASSIS, 1.0);

    // Center Panoramic Rear-View Mirror + Reflective Face under front roof header
    const centerMirror = new THREE.BoxGeometry(0.42, 0.075, 0.036);
    centerMirror.translate(0.0, 1.77, 0.83);
    acc.addGeo(centerMirror, chassisBlack, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.90);
    const centerMirrorFace = new THREE.BoxGeometry(0.39, 0.060, 0.008);
    centerMirrorFace.translate(0.0, 1.77, 0.810);
    acc.addGeo(centerMirrorFace, chromeSteel, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 1.0);

    // Extended 6-Seater Pearl-White Canopy Roof + UI Yellow Valance + Rooftop Solar Trickle-Charge Array!
    const roofCore = new THREE.BoxGeometry(1.22, 0.055, 2.74);
    roofCore.translate(0.0, 1.865, -0.26);
    acc.addGeo(roofCore, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 1.0);

    const roofCrown = new THREE.BoxGeometry(1.06, 0.035, 2.58);
    roofCrown.translate(0.0, 1.895, -0.26);
    acc.addGeo(roofCrown, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 1.0);

    for (const sz of [0.36, -0.88]) {
      const solarCell = new THREE.BoxGeometry(0.92, 0.012, 1.08);
      solarCell.translate(0.0, 1.916, sz);
      acc.addGeo(solarCell, solarNavy, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 1.0);
    }

    const roofValance = new THREE.BoxGeometry(1.25, 0.032, 2.77);
    roofValance.translate(0.0, 1.842, -0.26);
    acc.addGeo(roofValance, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.98);

    // =========================================================================
    // 7. FOUR ARTICULATED 10-INCH ALLOY WHEELS, BRAKE ROTORS, CALIPERS & LUGS
    // =========================================================================
    const wheelSpecs = [
      { joint: CART_J_WHEEL_FL, pivot: CART_BIND_PIVOTS[CART_J_WHEEL_FL], side: 1 },
      { joint: CART_J_WHEEL_FR, pivot: CART_BIND_PIVOTS[CART_J_WHEEL_FR], side: -1 },
      { joint: CART_J_WHEEL_RL, pivot: CART_BIND_PIVOTS[CART_J_WHEEL_RL], side: 1 },
      { joint: CART_J_WHEEL_RR, pivot: CART_BIND_PIVOTS[CART_J_WHEEL_RR], side: -1 }
    ];

    for (const ws of wheelSpecs) {
      const [wx, wy, wz] = ws.pivot;
      const jId = ws.joint;

      // Stationary Brake Caliper on chassis inboard of wheel rim
      const caliper = new THREE.BoxGeometry(0.045, 0.075, 0.055);
      caliper.translate(wx - ws.side * 0.042, wy + 0.062, wz - 0.052);
      acc.addGeo(caliper, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.92);

      // Ventilated Stainless Brake Rotor Disc spinning with wheel
      const brakeDisc = new THREE.CylinderGeometry(0.112, 0.112, 0.014, 16, 1);
      brakeDisc.rotateZ(Math.PI * 0.5);
      brakeDisc.translate(wx - ws.side * 0.038, wy, wz);
      acc.addGeo(brakeDisc, chromeSteel, HUMAN_MAT.METAL_ACCENT, jId, 0.88);

      // Main vulcanized rubber tire carcass (outer radius = 0.235m, width = 0.185m)
      const tireMain = new THREE.CylinderGeometry(CART_WHEEL_RADIUS - 0.008, CART_WHEEL_RADIUS - 0.008, 0.185, 24, 1);
      tireMain.rotateZ(Math.PI * 0.5);
      tireMain.translate(wx, wy, wz);
      acc.addGeo(tireMain, rubberDark, HUMAN_MAT.RUBBER_TIRE, jId, 0.92);

      // 12 Physical Raised Shoulder Tread Blocks around circumference
      for (let tb = 0; tb < 12; tb++) {
        const ang = (tb / 12) * Math.PI * 2.0;
        const lug = new THREE.BoxGeometry(0.192, 0.022, 0.044);
        lug.translate(0.0, CART_WHEEL_RADIUS - 0.009, 0.0);
        lug.rotateX(ang);
        lug.translate(wx, wy, wz);
        acc.addGeo(lug, [0.12, 0.125, 0.135], HUMAN_MAT.RUBBER_TIRE, jId, 0.88);
      }

      // Machined 10-inch Alloy Rim Barrel & Outer Lip
      const rimBarrel = new THREE.CylinderGeometry(0.145, 0.145, 0.172, 20, 1);
      rimBarrel.rotateZ(Math.PI * 0.5);
      rimBarrel.translate(wx + ws.side * 0.008, wy, wz);
      acc.addGeo(rimBarrel, [0.22, 0.24, 0.27], HUMAN_MAT.METAL_ACCENT, jId, 0.85);

      const rimLip = new THREE.TorusGeometry(0.142, 0.014, 8, 20);
      rimLip.rotateY(Math.PI * 0.5);
      rimLip.translate(wx + ws.side * 0.092, wy, wz);
      acc.addGeo(rimLip, alloySilver, HUMAN_MAT.METAL_ACCENT, jId, 0.98);

      // 6 Sculpted Silver Alloy Spokes + 5 Chrome Lug Nuts + UI Gold Center Hubcap
      for (let sp = 0; sp < 6; sp++) {
        const spAng = (sp / 6) * Math.PI * 2.0;
        const spoke = new THREE.BoxGeometry(0.018, 0.132, 0.028);
        spoke.translate(0.0, 0.068, 0.0);
        spoke.rotateX(spAng);
        spoke.translate(wx + ws.side * 0.090, wy, wz);
        acc.addGeo(spoke, alloySilver, HUMAN_MAT.METAL_ACCENT, jId, 0.98);
      }
      for (let ln = 0; ln < 5; ln++) {
        const lAng = (ln / 5) * Math.PI * 2.0;
        const nut = new THREE.CylinderGeometry(0.008, 0.008, 0.018, 6, 1);
        nut.rotateZ(Math.PI * 0.5);
        nut.translate(wx + ws.side * 0.098, wy + Math.cos(lAng) * 0.056, wz + Math.sin(lAng) * 0.056);
        acc.addGeo(nut, chromeSteel, HUMAN_MAT.METAL_ACCENT, jId, 0.98);
      }

      const centerCap = new THREE.CylinderGeometry(0.040, 0.044, 0.032, 12, 1);
      centerCap.rotateZ(Math.PI * 0.5);
      centerCap.translate(wx + ws.side * 0.094, wy, wz);
      acc.addGeo(centerCap, makaraGold, HUMAN_MAT.CREST_GOLD, jId, 1.0);
    }

    return acc.toBundle();
  }

  /**
   * Builds a Traditional Nusantara Outrigger Fishing Boat ("Perahu Nelayan Cadik Nusantara", ~10.8m long):
   * - True 9-station lofted double-ended wooden Jukung hull + carved upswept Linggi bow & stern stemposts
   * - Bamboo cadik outriggers with wave-slicing ski tips, S-curved katir cross-booms, ijuk rope lashings
   * - Kajang woven cabin shelter, foredeck fishing nets & baskets, stern Ketinting long-tail engine, & 2500K lanterns!
   */
  static buildDistantFishingBoat(variant = 0) {
    const acc = new RigidJointBundleAccumulator();

    const hullPalettes = [
      { hull: [0.12, 0.34, 0.68], trim: [0.92, 0.72, 0.14], cabin: [0.58, 0.42, 0.24] },
      { hull: [0.10, 0.52, 0.48], trim: [0.88, 0.30, 0.18], cabin: [0.62, 0.46, 0.26] },
      { hull: [0.72, 0.22, 0.16], trim: [0.93, 0.94, 0.95], cabin: [0.54, 0.39, 0.22] },
      { hull: [0.90, 0.92, 0.94], trim: [0.14, 0.38, 0.72], cabin: [0.60, 0.44, 0.25] }
    ];
    const pal = hullPalettes[variant % hullPalettes.length];
    const woodTeak = [0.46, 0.31, 0.18];
    const bambooGold = [0.72, 0.60, 0.32];
    const ijukBlack = [0.10, 0.10, 0.11];
    const netTurquoise = [0.14, 0.56, 0.58];
    const lanternGlass = [0.96, 0.82, 0.46];

    // 1. 9-Station Lofted Double-Ended Nusantara Jukung Wooden Hull (+Z bow, -Z stern, waterline y = 0.0m)
    const jukungStations = [
      { z: -5.25, keelY: 0.28, chineX: 0.05, chineY: 0.56, deckX: 0.09, deckY: 1.02 },
      { z: -4.20, keelY: -0.10, chineX: 0.36, chineY: 0.22, deckX: 0.50, deckY: 0.88 },
      { z: -2.60, keelY: -0.28, chineX: 0.62, chineY: 0.14, deckX: 0.78, deckY: 0.82 },
      { z: -0.80, keelY: -0.34, chineX: 0.72, chineY: 0.12, deckX: 0.88, deckY: 0.80 },
      { z: 0.80, keelY: -0.34, chineX: 0.72, chineY: 0.12, deckX: 0.88, deckY: 0.80 },
      { z: 2.60, keelY: -0.28, chineX: 0.60, chineY: 0.15, deckX: 0.76, deckY: 0.84 },
      { z: 4.20, keelY: -0.08, chineX: 0.34, chineY: 0.24, deckX: 0.48, deckY: 0.92 },
      { z: 5.30, keelY: 0.32, chineX: 0.05, chineY: 0.60, deckX: 0.09, deckY: 1.08 }
    ];
    acc.addGeo(buildLoftedHullStations(jukungStations, 0.0, false), pal.hull, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.96);

    // Recessed planked wooden interior cockpit sole + colorful Nusantara sheer gunwale wale strake
    const cockpitSole = new THREE.BoxGeometry(1.38, 0.06, 6.90);
    cockpitSole.translate(0.0, 0.58, 0.0);
    acc.addGeo(cockpitSole, woodTeak, HUMAN_MAT.SHOE_BAG, 0, 0.92, 2.0);

    for (const sx of [-1, 1]) {
      const sheerWale = new THREE.BoxGeometry(0.08, 0.14, 7.60);
      sheerWale.translate(sx * 0.82, 0.78, 0.0);
      acc.addGeo(sheerWale, pal.trim, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);
    }

    // Carved upswept Nusantara Linggi bow & stern stemposts + decorative crown finials
    for (const sz of [-1, 1]) {
      const linggi = new THREE.BoxGeometry(0.15, 1.55, 0.28);
      linggi.rotateX(-sz * 0.26);
      linggi.translate(0.0, 1.24, sz * 5.18);
      acc.addGeo(linggi, pal.trim, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

      const finial = new THREE.SphereGeometry(0.12, 10, 8);
      finial.scale(0.85, 1.35, 1.0);
      finial.translate(0.0, 2.02, sz * 5.38);
      acc.addGeo(finial, pal.hull, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);
    }

    // 2. Pitched Kajang Fisherman Cabin Shelter with Bamboo Roof Rafters
    const cabinBase = new THREE.BoxGeometry(1.32, 0.78, 2.55);
    cabinBase.translate(0.0, 1.02, -0.35);
    acc.addGeo(cabinBase, pal.cabin, HUMAN_MAT.SHOE_BAG, 0, 0.92);

    const cabinRoof = new THREE.CylinderGeometry(0.94, 0.94, 2.95, 3, 1);
    cabinRoof.rotateZ(Math.PI);
    cabinRoof.rotateX(Math.PI * 0.5);
    cabinRoof.scale(0.95, 0.48, 1.0);
    cabinRoof.translate(0.0, 1.56, -0.35);
    acc.addGeo(cabinRoof, woodTeak, HUMAN_MAT.SHOE_BAG, 0, 0.96);

    const ridgePole = new THREE.CylinderGeometry(0.045, 0.045, 3.12, 8, 1);
    ridgePole.rotateX(Math.PI * 0.5);
    ridgePole.translate(0.0, 1.80, -0.35);
    acc.addGeo(ridgePole, bambooGold, HUMAN_MAT.SHOE_BAG, 0, 0.98);

    // Foredeck Coiled Turquoise Gillnet, Orange Buoys, & Woven Fish Baskets
    const netPile = new THREE.SphereGeometry(0.42, 10, 8);
    netPile.scale(1.25, 0.55, 1.55);
    netPile.translate(0.0, 0.76, 2.15);
    acc.addGeo(netPile, netTurquoise, HUMAN_MAT.SHOE_BAG, 0, 0.90);

    for (const [bx, bz] of [[-0.36, 1.55], [0.38, 1.68]]) {
      const basket = new THREE.CylinderGeometry(0.19, 0.15, 0.30, 10, 1);
      basket.translate(bx, 0.75, bz);
      acc.addGeo(basket, bambooGold, HUMAN_MAT.SHOE_BAG, 0, 0.92);
    }
    for (const fx of [-0.32, 0.0, 0.32]) {
      const buoy = new THREE.SphereGeometry(0.085, 8, 6);
      buoy.translate(fx, 0.92, 2.35);
      acc.addGeo(buoy, [0.92, 0.38, 0.10], HUMAN_MAT.VEHICLE_GLOSS, 0, 0.96);
    }

    // 3. Twin Curved Katir Cross-Booms, Black Ijuk Rope Lashings, & Bamboo Cadik Outriggers with Ski Tips
    for (const bz of [-1.75, 1.55]) {
      const katirCenter = new THREE.CylinderGeometry(0.058, 0.058, 4.40, 8, 1);
      katirCenter.rotateZ(Math.PI * 0.5);
      katirCenter.translate(0.0, 0.92, bz);
      acc.addGeo(katirCenter, woodTeak, HUMAN_MAT.SHOE_BAG, 0, 0.92);

      for (const sx of [-1, 1]) {
        const katirDrop = new THREE.CylinderGeometry(0.048, 0.055, 1.28, 8, 1);
        katirDrop.rotateZ(sx * 1.02);
        katirDrop.translate(sx * 2.66, 0.56, bz);
        acc.addGeo(katirDrop, woodTeak, HUMAN_MAT.SHOE_BAG, 0, 0.90);

        const lashing = new THREE.BoxGeometry(0.16, 0.16, 0.16);
        lashing.translate(sx * 3.15, 0.18, bz);
        acc.addGeo(lashing, ijukBlack, HUMAN_MAT.RUBBER_TIRE, 0, 0.85);
      }
    }

    for (const sx of [-3.15, 3.15]) {
      const cadik = new THREE.CylinderGeometry(0.135, 0.135, 7.40, 10, 1);
      cadik.rotateX(Math.PI * 0.5);
      cadik.translate(sx, 0.06, -0.05);
      acc.addGeo(cadik, bambooGold, HUMAN_MAT.SHOE_BAG, 0, 0.95);

      // Upturned wave-slicing bamboo bow & stern ski tips
      for (const sz of [-1, 1]) {
        const skiTip = new THREE.ConeGeometry(0.135, 0.95, 10, 1);
        skiTip.rotateX(sz * (Math.PI * 0.5 - 0.30));
        skiTip.translate(sx, 0.20, -0.05 + sz * 4.05);
        acc.addGeo(skiTip, bambooGold, HUMAN_MAT.SHOE_BAG, 0, 0.97);
      }
    }

    // 4. Stern Mesin Ketinting Long-Tail Engine + Angled Propeller Shaft + Caping Fisherman
    const ketintingBlock = new THREE.BoxGeometry(0.32, 0.28, 0.36);
    ketintingBlock.translate(0.14, 0.98, -3.55);
    acc.addGeo(ketintingBlock, [0.76, 0.18, 0.14], HUMAN_MAT.METAL_ACCENT, 0, 0.94);

    const longTailShaft = new THREE.CylinderGeometry(0.022, 0.022, 2.35, 8, 1);
    longTailShaft.rotateX(1.18);
    longTailShaft.translate(0.14, 0.48, -4.58);
    acc.addGeo(longTailShaft, [0.78, 0.80, 0.84], HUMAN_MAT.METAL_ACCENT, 0, 0.92);

    const fisherBody = new THREE.CylinderGeometry(0.19, 0.23, 0.74, 8, 1);
    fisherBody.translate(-0.08, 1.06, -2.85);
    acc.addGeo(fisherBody, [0.22, 0.28, 0.36], HUMAN_MAT.SHOE_BAG, 0, 0.92);

    const caping = new THREE.ConeGeometry(0.42, 0.24, 12, 1);
    caping.translate(-0.08, 1.58, -2.85);
    acc.addGeo(caping, bambooGold, HUMAN_MAT.SHOE_BAG, 0, 0.98);

    // 5. Wooden Mast, Bamboo Boom, & Warm 2500K Fisherman Attractant Lanterns (metaZ = 0.0)
    const mast = new THREE.CylinderGeometry(0.055, 0.08, 3.6, 8, 1);
    mast.translate(0.0, 2.45, 1.25);
    acc.addGeo(mast, woodTeak, HUMAN_MAT.SHOE_BAG, 0, 0.94);

    const mastLantern = new THREE.SphereGeometry(0.22, 10, 8);
    mastLantern.translate(0.0, 4.32, 1.25);
    acc.addGeo(mastLantern, lanternGlass, HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 0.0);

    const bowLantern = new THREE.SphereGeometry(0.20, 10, 8);
    bowLantern.translate(0.0, 2.24, 5.42);
    acc.addGeo(bowLantern, lanternGlass, HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 0.0);

    return acc.toBundle();
  }

  /**
   * Builds a Detailed Coastal Freighter / Patrol Coaster ("Kapal Kecil Pesisir / Coaster", ~24.0m long):
   * - 9-station lofted steel displacement hull + submerged bulbous bow + red anti-fouling boot-top
   * - Raised forecastle bulwark, hawsepipes & stowed anchors, corrugated cargo hatch & A-frame cargo derrick crane
   * - Aft 2-deck wheelhouse with bridge wings, SOLAS liferaft canisters, funnel exhausts, lattice radar mast & COLREGs!
   */
  static buildDistantSmallShip(variant = 0) {
    const acc = new RigidJointBundleAccumulator();

    const hullColors = [
      [0.11, 0.22, 0.38],
      [0.14, 0.34, 0.36],
      [0.20, 0.24, 0.30],
      [0.48, 0.16, 0.15]
    ];
    const hullCol = hullColors[variant % hullColors.length];
    const bootRed = [0.58, 0.14, 0.12];
    const superWhite = [0.92, 0.94, 0.96];
    const deckGreen = [0.19, 0.33, 0.28];
    const hatchRust = [0.46, 0.24, 0.18];
    const craneYellow = [0.88, 0.64, 0.12];
    const darkSteel = [0.15, 0.17, 0.20];
    const funnelAccent = variant % 2 === 0 ? [0.88, 0.62, 0.10] : [0.16, 0.42, 0.74];

    // 1. 9-Station Lofted Steel Coaster Hull + Red Anti-Fouling Lower Boot-Top + Bulbous Bow
    const bootStations = [
      { z: -11.6, keelY: -0.35, chineX: 1.55, chineY: 0.05, deckX: 2.15, deckY: 0.56 },
      { z: -7.5, keelY: -0.95, chineX: 2.25, chineY: -0.25, deckX: 2.48, deckY: 0.56 },
      { z: 0.0, keelY: -1.05, chineX: 2.38, chineY: -0.28, deckX: 2.54, deckY: 0.56 },
      { z: 6.5, keelY: -0.98, chineX: 2.10, chineY: -0.22, deckX: 2.36, deckY: 0.56 },
      { z: 10.2, keelY: -0.72, chineX: 1.05, chineY: -0.10, deckX: 1.32, deckY: 0.58 },
      { z: 12.1, keelY: -0.30, chineX: 0.18, chineY: 0.12, deckX: 0.24, deckY: 0.62 }
    ];
    acc.addGeo(buildLoftedHullStations(bootStations, 0.0, false), bootRed, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.90);

    const bulbousBow = new THREE.SphereGeometry(0.62, 12, 10);
    bulbousBow.scale(0.82, 0.76, 2.35);
    bulbousBow.translate(0.0, -0.32, 11.85);
    acc.addGeo(bulbousBow, bootRed, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.88);

    const topsideStations = [
      { z: -11.8, keelY: 0.52, chineX: 2.15, chineY: 1.15, deckX: 2.35, deckY: 2.18 },
      { z: -7.5, keelY: 0.52, chineX: 2.48, chineY: 1.25, deckX: 2.58, deckY: 2.18 },
      { z: 0.0, keelY: 0.52, chineX: 2.54, chineY: 1.28, deckX: 2.62, deckY: 2.22 },
      { z: 6.5, keelY: 0.52, chineX: 2.36, chineY: 1.42, deckX: 2.56, deckY: 2.48 },
      { z: 10.2, keelY: 0.54, chineX: 1.32, chineY: 1.68, deckX: 1.86, deckY: 2.86 },
      { z: 12.4, keelY: 0.58, chineX: 0.20, chineY: 1.92, deckX: 0.32, deckY: 3.12 }
    ];
    acc.addGeo(buildLoftedHullStations(topsideStations, 0.0, true), hullCol, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.97);

    // Non-skid green steel weather deck + Forecastle anchor windlass + Stowed bower anchors + Mooring bollards
    const weatherDeck = new THREE.BoxGeometry(4.85, 0.08, 18.6);
    weatherDeck.translate(0.0, 2.24, -1.2);
    acc.addGeo(weatherDeck, deckGreen, HUMAN_MAT.RUBBER_TIRE, 0, 0.94);

    const windlass = new THREE.BoxGeometry(1.35, 0.48, 0.72);
    windlass.translate(0.0, 2.88, 9.4);
    acc.addGeo(windlass, darkSteel, HUMAN_MAT.METAL_ACCENT, 0, 0.92);

    for (const sx of [-1, 1]) {
      const anchorPocket = new THREE.BoxGeometry(0.18, 0.58, 0.64);
      anchorPocket.rotateY(sx * 0.34);
      anchorPocket.translate(sx * 1.48, 2.18, 10.35);
      acc.addGeo(anchorPocket, darkSteel, HUMAN_MAT.METAL_ACCENT, 0, 0.88);

      for (const bz of [-10.4, -2.0, 7.8]) {
        const bitt = new THREE.CylinderGeometry(0.11, 0.12, 0.38, 8, 1);
        bitt.translate(sx * 2.22, 2.42, bz);
        acc.addGeo(bitt, darkSteel, HUMAN_MAT.METAL_ACCENT, 0, 0.92);
      }
    }

    // 2. Forward Corrugated Steel Cargo Hatch Coaming + MacGregor Covers + A-Frame Cargo Derrick Boom
    const hatchCoaming = new THREE.BoxGeometry(3.45, 0.62, 6.90);
    hatchCoaming.translate(0.0, 2.56, 3.35);
    acc.addGeo(hatchCoaming, hatchRust, HUMAN_MAT.METAL_ACCENT, 0, 0.94);

    for (let hz = 0.35; hz <= 6.35; hz += 1.20) {
      const hatchPanel = new THREE.BoxGeometry(3.56, 0.16, 1.08);
      hatchPanel.translate(0.0, 2.94, hz);
      acc.addGeo(hatchPanel, [0.54, 0.28, 0.20], HUMAN_MAT.METAL_ACCENT, 0, 0.97);
    }

    for (const sx of [-0.95, 0.95]) {
      const kingpost = new THREE.CylinderGeometry(0.12, 0.16, 3.85, 8, 1);
      kingpost.rotateZ(-sx * 0.16);
      kingpost.translate(sx, 4.12, 7.25);
      acc.addGeo(kingpost, craneYellow, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.96);
    }
    const kingCross = new THREE.BoxGeometry(1.42, 0.20, 0.24);
    kingCross.translate(0.0, 5.95, 7.25);
    acc.addGeo(kingCross, craneYellow, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    const derrickBoom = new THREE.CylinderGeometry(0.085, 0.11, 6.20, 8, 1);
    derrickBoom.rotateX(-1.12);
    derrickBoom.translate(0.0, 4.18, 4.45);
    acc.addGeo(derrickBoom, craneYellow, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    const toppingStay = new THREE.CylinderGeometry(0.022, 0.022, 5.40, 6, 1);
    toppingStay.rotateX(-1.38);
    toppingStay.translate(0.0, 5.48, 4.65);
    acc.addGeo(toppingStay, darkSteel, HUMAN_MAT.METAL_ACCENT, 0, 0.92);

    // 3. Aft 2-Deck White Superstructure, Bridge Wings, SOLAS Liferaft Canisters, & Companionway Stairs
    const lowerDeckhouse = new THREE.BoxGeometry(4.10, 1.72, 7.80);
    lowerDeckhouse.translate(0.0, 3.12, -5.20);
    acc.addGeo(lowerDeckhouse, superWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    for (const pz of [-3.2, -5.2, -7.2]) {
      const portStrip = new THREE.BoxGeometry(4.16, 0.28, 0.62);
      portStrip.translate(0.0, 3.24, pz);
      acc.addGeo(portStrip, [0.22, 0.30, 0.38], HUMAN_MAT.BOAT_LANTERN, 0, 0.98, 3.0);
    }

    const bridgeHouse = new THREE.BoxGeometry(3.75, 1.58, 3.95);
    bridgeHouse.translate(0.0, 4.74, -3.55);
    acc.addGeo(bridgeHouse, superWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    // Overhanging Port & Starboard Open Bridge Wings + White Cylindrical SOLAS Liferaft Canisters
    const bridgeWingDeck = new THREE.BoxGeometry(5.10, 0.14, 1.65);
    bridgeWingDeck.translate(0.0, 4.02, -2.85);
    acc.addGeo(bridgeWingDeck, deckGreen, HUMAN_MAT.RUBBER_TIRE, 0, 0.96);

    const bridgeWingBulwark = new THREE.BoxGeometry(5.16, 0.68, 1.70);
    bridgeWingBulwark.translate(0.0, 4.40, -2.85);
    acc.addGeo(bridgeWingBulwark, superWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    for (const sx of [-2.28, 2.28]) {
      const liferaft = new THREE.CylinderGeometry(0.28, 0.28, 0.96, 12, 1);
      liferaft.rotateX(Math.PI * 0.5);
      liferaft.translate(sx, 3.92, -4.45);
      acc.addGeo(liferaft, superWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);
    }

    // Wrap-around Forward-Raked Bridge Windows (glowing softly warm-white metaZ = 3.0 at night)
    const bridgeGlassFront = new THREE.BoxGeometry(3.62, 0.58, 0.18);
    bridgeGlassFront.rotateX(-0.14);
    bridgeGlassFront.translate(0.0, 4.96, -1.56);
    acc.addGeo(bridgeGlassFront, [0.22, 0.30, 0.38], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const bridgeGlassSides = new THREE.BoxGeometry(3.82, 0.54, 3.10);
    bridgeGlassSides.translate(0.0, 4.96, -3.45);
    acc.addGeo(bridgeGlassSides, [0.22, 0.30, 0.38], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const bridgeRoof = new THREE.BoxGeometry(4.15, 0.18, 4.45);
    bridgeRoof.translate(0.0, 5.58, -3.50);
    acc.addGeo(bridgeRoof, superWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    // 4. Tapered Funnel Stack with Black Soot Cowl & Exhaust Pipes + Lattice Radar Mast + COLREGs
    const funnel = new THREE.CylinderGeometry(0.54, 0.68, 2.15, 12, 1);
    funnel.rotateX(-0.16);
    funnel.translate(0.0, 4.92, -7.25);
    acc.addGeo(funnel, funnelAccent, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.96);

    const funnelCowl = new THREE.CylinderGeometry(0.55, 0.55, 0.38, 12, 1);
    funnelCowl.rotateX(-0.16);
    funnelCowl.translate(0.0, 5.98, -7.42);
    acc.addGeo(funnelCowl, darkSteel, HUMAN_MAT.METAL_ACCENT, 0, 0.92);

    for (const ex of [-0.18, 0.18]) {
      const exhPipe = new THREE.CylinderGeometry(0.11, 0.11, 0.45, 8, 1);
      exhPipe.rotateX(-0.16);
      exhPipe.translate(ex, 6.26, -7.46);
      acc.addGeo(exhPipe, darkSteel, HUMAN_MAT.METAL_ACCENT, 0, 0.90);
    }

    const radarMast = new THREE.CylinderGeometry(0.09, 0.16, 3.10, 8, 1);
    radarMast.rotateX(-0.08);
    radarMast.translate(0.0, 7.12, -3.35);
    acc.addGeo(radarMast, superWhite, HUMAN_MAT.METAL_ACCENT, 0, 0.96);

    const yardarm = new THREE.CylinderGeometry(0.04, 0.04, 2.60, 8, 1);
    yardarm.rotateZ(Math.PI * 0.5);
    yardarm.translate(0.0, 7.95, -3.42);
    acc.addGeo(yardarm, superWhite, HUMAN_MAT.METAL_ACCENT, 0, 0.98);

    const radarScanner = new THREE.BoxGeometry(1.65, 0.15, 0.24);
    radarScanner.translate(0.0, 7.55, -3.20);
    acc.addGeo(radarScanner, superWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    // COLREGs Marine Navigation Lights (Masthead White = 3.0, Starboard (-X) Green = 1.0, Port (+X) Red = 2.0)
    const mastLight = new THREE.SphereGeometry(0.26, 10, 8);
    mastLight.translate(0.0, 8.72, -3.48);
    acc.addGeo(mastLight, [0.96, 0.94, 0.85], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const stbdGreen = new THREE.SphereGeometry(0.22, 10, 8);
    stbdGreen.translate(-2.52, 4.78, -2.15);
    acc.addGeo(stbdGreen, [0.20, 0.85, 0.35], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 1.0);

    const portRed = new THREE.SphereGeometry(0.22, 10, 8);
    portRed.translate(2.52, 4.78, -2.15);
    acc.addGeo(portRed, [0.88, 0.20, 0.16], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 2.0);

    return acc.toBundle();
  }

  /**
   * Builds a Sleek 44-ft (13.4m) Pearl-White & Royal-Navy Flybridge Mini Yacht ("Mini Yacht Teluk U"):
   * - Exact air draft = 5.05m above waterline (world top = +3.60m at WATER_LEVEL_Y = -1.45m),
   *   cruising smoothly underneath Jembatan Teksas UI's +7.80m soffit (+4.20m spare headroom!).
   * - True lofted deep-V planing hull, underwater twin brass propellers & rudders, caulked Sikaflex teak decks,
   *   folding swim ladder, bow windlass + Bruce anchor, sliding saloon glass door, flybridge helm wheel & chartplotter MFD.
   */
  static buildFlybridgeMiniYacht() {
    const acc = new RigidJointBundleAccumulator();

    const yachtWhite = [0.95, 0.96, 0.98];
    const royalNavy = [0.09, 0.20, 0.40];
    const goldStripe = [0.90, 0.68, 0.14];
    const teakDeck = [0.54, 0.36, 0.20];
    const cushionCream = [0.89, 0.85, 0.76];
    const stainless = [0.84, 0.87, 0.91];
    const brassBronze = [0.76, 0.58, 0.22];
    const tintedGlass = [0.16, 0.24, 0.32];
    const darkCarbon = [0.12, 0.13, 0.16];

    // 1. Lofted Deep-V Planing Lower Hull (Royal-Navy) + Upper Flared Clipper Freeboard (Pearl-White)
    const lowerChineStations = [
      { z: -5.30, keelY: -0.58, chineX: 1.66, chineY: -0.14, deckX: 1.82, deckY: 0.50 },
      { z: -2.40, keelY: -0.68, chineX: 1.80, chineY: -0.18, deckX: 1.94, deckY: 0.50 },
      { z: 0.80, keelY: -0.72, chineX: 1.84, chineY: -0.16, deckX: 1.98, deckY: 0.50 },
      { z: 3.40, keelY: -0.62, chineX: 1.44, chineY: -0.06, deckX: 1.62, deckY: 0.52 },
      { z: 5.20, keelY: -0.34, chineX: 0.68, chineY: 0.12, deckX: 0.86, deckY: 0.58 },
      { z: 6.45, keelY: 0.08, chineX: 0.08, chineY: 0.35, deckX: 0.14, deckY: 0.68 }
    ];
    acc.addGeo(buildLoftedHullStations(lowerChineStations, 0.0, false), royalNavy, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.92);

    const upperFreeboardStations = [
      { z: -5.35, keelY: 0.48, chineX: 1.82, chineY: 0.98, deckX: 1.90, deckY: 1.48 },
      { z: -2.40, keelY: 0.48, chineX: 1.94, chineY: 1.02, deckX: 2.00, deckY: 1.48 },
      { z: 0.80, keelY: 0.48, chineX: 1.98, chineY: 1.04, deckX: 2.02, deckY: 1.50 },
      { z: 3.40, keelY: 0.50, chineX: 1.62, chineY: 1.08, deckX: 1.78, deckY: 1.54 },
      { z: 5.25, keelY: 0.56, chineX: 0.86, chineY: 1.15, deckX: 1.12, deckY: 1.60 },
      { z: 6.72, keelY: 0.66, chineX: 0.12, chineY: 1.24, deckX: 0.22, deckY: 1.66 }
    ];
    acc.addGeo(buildLoftedHullStations(upperFreeboardStations, 0.0, true), yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    // Golden cove stripe & hullside dark-tinted cabin porthole windows
    for (const sx of [-1, 1]) {
      const goldCove = new THREE.BoxGeometry(0.04, 0.065, 8.80);
      goldCove.translate(sx * 1.92, 0.52, -0.60);
      acc.addGeo(goldCove, goldStripe, HUMAN_MAT.CREST_GOLD, 0, 0.98);
    }
    const hullWindowStrip = new THREE.BoxGeometry(4.02, 0.22, 4.50);
    hullWindowStrip.translate(0.0, 1.00, 0.35);
    acc.addGeo(hullWindowStrip, tintedGlass, HUMAN_MAT.BOAT_LANTERN, 0, 0.98, 3.0);

    // Underwater Twin Brass Shafts, 4-Blade Propellers, Spade Rudders & Stern Trim Tabs
    for (const sx of [-0.88, 0.88]) {
      const pShaft = new THREE.CylinderGeometry(0.032, 0.032, 1.55, 8, 1);
      pShaft.rotateX(Math.PI * 0.5 + 0.18);
      pShaft.translate(sx, -0.58, -4.15);
      acc.addGeo(pShaft, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.82);

      const propHub = new THREE.ConeGeometry(0.065, 0.22, 10, 1);
      propHub.rotateX(-Math.PI * 0.5);
      propHub.translate(sx, -0.72, -4.85);
      acc.addGeo(propHub, brassBronze, HUMAN_MAT.METAL_ACCENT, 0, 0.88);

      for (let pb = 0; pb < 4; pb++) {
        const blade = new THREE.BoxGeometry(0.025, 0.22, 0.095);
        blade.rotateY(0.45);
        blade.translate(0.0, 0.11, 0.0);
        blade.rotateZ((pb / 4) * Math.PI * 2.0);
        blade.translate(sx, -0.72, -4.78);
        acc.addGeo(blade, brassBronze, HUMAN_MAT.METAL_ACCENT, 0, 0.88);
      }

      const rudder = new THREE.BoxGeometry(0.045, 0.42, 0.30);
      rudder.translate(sx, -0.64, -5.18);
      acc.addGeo(rudder, brassBronze, HUMAN_MAT.METAL_ACCENT, 0, 0.86);

      const trimTab = new THREE.BoxGeometry(0.48, 0.025, 0.26);
      trimTab.rotateX(0.12);
      trimTab.translate(sx * 1.32, -0.22, -5.42);
      acc.addGeo(trimTab, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.88);
    }

    // 2. Caulked Sikaflex Teak Swim Platform (metaZ = 2.0), Swim Ladder, Transom Steps, & Sunpads
    const swimTrim = new THREE.BoxGeometry(3.72, 0.14, 1.52);
    swimTrim.translate(0.0, 0.18, -5.88);
    acc.addGeo(swimTrim, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.95);

    const swimPlatform = new THREE.BoxGeometry(3.62, 0.08, 1.44);
    swimPlatform.translate(0.0, 0.27, -5.86);
    acc.addGeo(swimPlatform, teakDeck, HUMAN_MAT.SHOE_BAG, 0, 0.96, 2.0);

    // Folding Stainless Telescopic Swim Ladder on Starboard Swim Platform
    for (const lx of [-1.42, -1.14]) {
      const rail = new THREE.CylinderGeometry(0.016, 0.016, 0.56, 8, 1);
      rail.rotateX(0.32);
      rail.translate(lx, 0.12, -6.58);
      acc.addGeo(rail, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.96);
    }
    for (const ry of [0.22, 0.04, -0.12]) {
      const rung = new THREE.BoxGeometry(0.28, 0.018, 0.035);
      rung.translate(-1.28, ry, -6.56 - (0.22 - ry) * 0.30);
      acc.addGeo(rung, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.95);
    }

    // Caulked Teak Main Promenade & Foredeck Planking (metaZ = 2.0)
    const mainTeakDeck = new THREE.BoxGeometry(3.74, 0.05, 8.95);
    mainTeakDeck.translate(0.0, 1.515, -0.62);
    acc.addGeo(mainTeakDeck, teakDeck, HUMAN_MAT.SHOE_BAG, 0, 0.97, 2.0);

    const foreTeakStations = [
      { z: 3.40, keelY: 1.51, chineX: 1.52, chineY: 1.54, deckX: 1.66, deckY: 1.56 },
      { z: 5.20, keelY: 1.56, chineX: 0.82, chineY: 1.59, deckX: 0.98, deckY: 1.62 },
      { z: 6.45, keelY: 1.62, chineX: 0.12, chineY: 1.64, deckX: 0.18, deckY: 1.66 }
    ];
    acc.addGeo(buildLoftedHullStations(foreTeakStations, 0.0, true), teakDeck, HUMAN_MAT.SHOE_BAG, 0, 0.98, 2.0);

    // Bow Electric Anchor Windlass, Bow Roller & Polished Stainless Bruce Anchor
    const bowRoller = new THREE.BoxGeometry(0.20, 0.10, 0.68);
    bowRoller.translate(0.0, 1.69, 6.58);
    acc.addGeo(bowRoller, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.98);

    const bruceAnchor = new THREE.ConeGeometry(0.18, 0.42, 6, 1);
    bruceAnchor.rotateX(-Math.PI * 0.65);
    bruceAnchor.translate(0.0, 1.52, 6.84);
    acc.addGeo(bruceAnchor, stainless, HUMAN_MAT.METAL_ACCENT, 0, 1.0);

    const yachtWindlass = new THREE.CylinderGeometry(0.14, 0.16, 0.18, 12, 1);
    yachtWindlass.translate(0.0, 1.70, 5.88);
    acc.addGeo(yachtWindlass, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.98);

    // Twin Foredeck Luxury Sunpad Mattresses & Headrest Bolsters + Aft Cockpit U-Lounge & Teak Table
    for (const sx of [-0.52, 0.52]) {
      const sunpad = new THREE.BoxGeometry(0.84, 0.13, 2.02);
      sunpad.rotateX(0.04);
      sunpad.translate(sx, 1.62, 3.65);
      acc.addGeo(sunpad, cushionCream, HUMAN_MAT.SHOE_BAG, 0, 0.98);

      const bolster = new THREE.CylinderGeometry(0.08, 0.08, 0.80, 10, 1);
      bolster.rotateZ(Math.PI * 0.5);
      bolster.translate(sx, 1.71, 2.74);
      acc.addGeo(bolster, cushionCream, HUMAN_MAT.SHOE_BAG, 0, 0.99);
    }
    const aftBench = new THREE.BoxGeometry(2.85, 0.42, 0.68);
    aftBench.translate(0.0, 1.72, -4.65);
    acc.addGeo(aftBench, cushionCream, HUMAN_MAT.SHOE_BAG, 0, 0.96);

    const aftTable = new THREE.BoxGeometry(1.45, 0.055, 0.68);
    aftTable.translate(0.0, 1.98, -3.95);
    acc.addGeo(aftTable, teakDeck, HUMAN_MAT.SHOE_BAG, 0, 0.98, 2.0);
    const tablePedestal = new THREE.CylinderGeometry(0.045, 0.06, 0.44, 10, 1);
    tablePedestal.translate(0.0, 1.74, -3.95);
    acc.addGeo(tablePedestal, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.94);

    // Stainless Steel Bow Pulpit Safety Guardrails + Cylindrical Fender Baskets
    for (const sx of [-1, 1]) {
      for (let pz = 1.6; pz <= 5.6; pz += 1.0) {
        const taper = Math.max(0.22, 1.0 - Math.max(0.0, pz - 3.4) * 0.34);
        const stanchion = new THREE.CylinderGeometry(0.018, 0.018, 0.58, 6, 1);
        stanchion.translate(sx * 1.76 * taper, 1.84, pz);
        acc.addGeo(stanchion, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.98);
      }
      const bowRail = new THREE.CylinderGeometry(0.020, 0.020, 4.4, 8, 1);
      bowRail.rotateX(Math.PI * 0.5);
      bowRail.rotateY(-sx * 0.25);
      bowRail.translate(sx * 1.22, 2.12, 3.65);
      acc.addGeo(bowRail, stainless, HUMAN_MAT.METAL_ACCENT, 0, 1.0);

      const fender = new THREE.CylinderGeometry(0.11, 0.11, 0.52, 10, 1);
      fender.translate(sx * 1.48, 1.84, 2.55);
      acc.addGeo(fender, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);
    }

    // 3. Panoramic Main Saloon Cabin Superstructure + Stainless Aft Sliding Glass Doors
    const saloonCore = new THREE.BoxGeometry(3.05, 1.22, 5.25);
    saloonCore.translate(0.0, 2.13, -0.85);
    acc.addGeo(saloonCore, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    const aftSlidingGlass = new THREE.BoxGeometry(2.42, 1.06, 0.06);
    aftSlidingGlass.translate(0.0, 2.08, -3.48);
    acc.addGeo(aftSlidingGlass, tintedGlass, HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const windshield = new THREE.BoxGeometry(2.88, 0.88, 1.35);
    windshield.rotateX(0.56);
    windshield.translate(0.0, 2.14, 1.95);
    acc.addGeo(windshield, tintedGlass, HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const sideSaloonGlass = new THREE.BoxGeometry(3.12, 0.62, 4.35);
    sideSaloonGlass.translate(0.0, 2.24, -0.68);
    acc.addGeo(sideSaloonGlass, tintedGlass, HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const flySole = new THREE.BoxGeometry(3.22, 0.16, 6.15);
    flySole.translate(0.0, 2.80, -1.35);
    acc.addGeo(flySole, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    for (const sx of [-1.42, 1.42]) {
      const aftPillar = new THREE.CylinderGeometry(0.055, 0.065, 1.28, 8, 1);
      aftPillar.rotateX(-0.14);
      aftPillar.translate(sx, 2.14, -4.18);
      acc.addGeo(aftPillar, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.96);
    }

    // 4. Open Upper Flybridge Helm Console, Stainless Destroyer Wheel, Twin Throttles & Chartplotter MFD (metaZ = 2.2)
    const flyCoaming = new THREE.BoxGeometry(2.76, 0.52, 3.95);
    flyCoaming.translate(0.0, 3.12, -1.05);
    acc.addGeo(flyCoaming, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    const helmPod = new THREE.BoxGeometry(1.12, 0.34, 0.54);
    helmPod.translate(-0.46, 3.42, 0.54);
    acc.addGeo(helmPod, darkCarbon, HUMAN_MAT.METAL_ACCENT, 0, 0.96);

    const chartplotter = new THREE.BoxGeometry(0.42, 0.22, 0.03);
    chartplotter.rotateX(-0.46);
    chartplotter.translate(-0.46, 3.61, 0.48);
    acc.addGeo(chartplotter, [0.12, 0.48, 0.54], HUMAN_MAT.VEHICLE_HEADLIGHT, 0, 1.0, 2.2);

    const helmWheel = new THREE.TorusGeometry(0.18, 0.016, 8, 18);
    helmWheel.rotateX(0.55);
    helmWheel.translate(-0.46, 3.52, 0.24);
    acc.addGeo(helmWheel, stainless, HUMAN_MAT.METAL_ACCENT, 0, 1.0);

    for (const tx of [-0.14, -0.08]) {
      const throttle = new THREE.CylinderGeometry(0.010, 0.012, 0.16, 6, 1);
      throttle.rotateX(0.32);
      throttle.translate(tx, 3.64, 0.44);
      acc.addGeo(throttle, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.98);
    }

    const flyWindscreen = new THREE.BoxGeometry(2.58, 0.34, 0.55);
    flyWindscreen.rotateX(0.52);
    flyWindscreen.translate(0.0, 3.48, 0.92);
    acc.addGeo(flyWindscreen, [0.36, 0.56, 0.72], HUMAN_MAT.VEHICLE_GLASS, 0, 1.0);

    for (const sx of [-0.55, 0.55]) {
      const helmSeat = new THREE.BoxGeometry(0.54, 0.56, 0.52);
      helmSeat.translate(sx, 3.36, -0.22);
      acc.addGeo(helmSeat, cushionCream, HUMAN_MAT.SHOE_BAG, 0, 0.98);
    }
    const flyLounge = new THREE.BoxGeometry(2.25, 0.34, 1.25);
    flyLounge.translate(0.0, 3.18, -2.05);
    acc.addGeo(flyLounge, cushionCream, HUMAN_MAT.SHOE_BAG, 0, 0.98);

    // 5. Swept FRP Radar Arch, Garmin Marine Radome, GPS Pod, & COLREGs Navigation Lights
    //    (CRITICAL INVARIANT: Highest vertex strictly at y = 5.05m above waterline!)
    for (const sx of [-1.26, 1.26]) {
      const archWing = new THREE.BoxGeometry(0.18, 1.28, 0.58);
      archWing.rotateX(-0.26);
      archWing.rotateZ(sx > 0 ? 0.12 : -0.12);
      archWing.translate(sx, 3.82, -1.75);
      acc.addGeo(archWing, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);
    }
    const radarCrossbar = new THREE.BoxGeometry(2.52, 0.16, 0.62);
    radarCrossbar.translate(0.0, 4.44, -1.92);
    acc.addGeo(radarCrossbar, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    const radome = new THREE.CylinderGeometry(0.44, 0.46, 0.28, 16, 1);
    radome.translate(0.0, 4.68, -1.88);
    acc.addGeo(radome, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    for (const gx of [-0.82, 0.82]) {
      const gpsMushroom = new THREE.SphereGeometry(0.09, 10, 8);
      gpsMushroom.scale(1.0, 0.65, 1.0);
      gpsMushroom.translate(gx, 4.58, -1.92);
      acc.addGeo(gpsMushroom, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);
    }

    const lightMast = new THREE.CylinderGeometry(0.03, 0.04, 0.24, 8, 1);
    lightMast.translate(0.0, 4.90, -1.88);
    acc.addGeo(lightMast, stainless, HUMAN_MAT.METAL_ACCENT, 0, 1.0);

    // Top All-Round White Anchor/Masthead Light strictly peaking at y = 4.96 + 0.09 = 5.05m!
    const mastheadLight = new THREE.SphereGeometry(0.09, 10, 8);
    mastheadLight.translate(0.0, 4.96, -1.88);
    acc.addGeo(mastheadLight, [0.98, 0.96, 0.88], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const stbdGreen = new THREE.SphereGeometry(0.085, 8, 6);
    stbdGreen.translate(-1.42, 3.24, 0.35);
    acc.addGeo(stbdGreen, [0.18, 0.88, 0.36], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 1.0);

    const portRed = new THREE.SphereGeometry(0.085, 8, 6);
    portRed.translate(1.42, 3.24, 0.35);
    acc.addGeo(portRed, [0.90, 0.18, 0.14], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 2.0);

    return acc.toBundle();
  }

  /**
   * Builds the Iconic 7.8m UI Alma Mater Yellow Electric Inter-Island Campus Bus ("Bus Kuning Listrik Lintas Pulau UI"):
   * - Overall Envelope: 7.80m Length x 2.38m Body Width x 2.96m Roof Peak (clears the 4.15m Underpass Bawah Laut bore!)
   * - 6 Articulated Joints (`BUS_J_CHASSIS`, `BUS_J_STEER_WHEEL`, + 4 Heavy-Duty Wheels `BUS_J_WHEEL_FL..RR`)
   * - Dual Rear Tires (`Ban Belakang Ganda`), Ventilated Front Brake Rotors & Calipers, Rubber Mudflaps,
   *   Curved Twin-Arm Horn Transit Mirrors (`Spion Tanduk`) + Curb Spotter Mirror, Recessed Stepwell,
   *   Front & Curbside Amber LED Dot-Matrix Route Signs (`metaZ = 2.0`), 4-Fan EV Battery/HVAC Roof Pod,
   *   and Full RHD Transit Cockpit (`metaZ = 2.1` EV Cluster, `metaZ = 2.2` CCTV/GPS Screen, DNR Keypad, Door Buttons).
   */
  static buildInterIslandElectricBus() {
    const acc = new RigidJointBundleAccumulator();

    const bikunYellow = [0.94, 0.70, 0.08];
    const pearlWhite = [0.94, 0.95, 0.97];
    const royalNavy = [0.09, 0.19, 0.38];
    const chassisDark = [0.11, 0.12, 0.14];
    const rubberTire = [0.085, 0.088, 0.095];
    const alloySilver = [0.78, 0.81, 0.85];
    const chromeSteel = [0.85, 0.88, 0.92];
    const makaraGold = [0.91, 0.70, 0.15];
    const glassTint = [0.22, 0.32, 0.42];
    const seatBlue = [0.16, 0.32, 0.56];
    const hvOrange = [0.92, 0.38, 0.06];

    // 1. Underfloor EV Chassis Frame, Air-Ride Bellows, Portal Axles, & Navy Rocker Skirt
    const underChassis = new THREE.BoxGeometry(2.12, 0.24, 7.36);
    underChassis.translate(0.0, 0.44, 0.0);
    acc.addGeo(underChassis, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.74);

    for (const wz of [2.10, -2.10]) {
      const axle = new THREE.CylinderGeometry(0.062, 0.062, 1.96, 12, 1);
      axle.rotateZ(Math.PI * 0.5);
      axle.translate(0.0, BUS_WHEEL_RADIUS, wz);
      acc.addGeo(axle, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.72);

      for (const sx of [-0.72, 0.72]) {
        const airBag = new THREE.CylinderGeometry(0.13, 0.13, 0.22, 10, 1);
        airBag.translate(sx, 0.56, wz);
        acc.addGeo(airBag, rubberTire, HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.76);
      }
    }

    const lowerSkirt = new THREE.BoxGeometry(2.34, 0.34, 7.58);
    lowerSkirt.translate(0.0, 0.56, 0.0);
    acc.addGeo(lowerSkirt, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.92);

    // Side battery-bay service latch lines & amber LED side marker lights
    for (const sx of [-1.0, 1.0]) {
      for (const mz of [-3.10, -0.95, 0.95, 3.15]) {
        const marker = new THREE.BoxGeometry(0.025, 0.038, 0.095);
        marker.translate(sx * 1.178, 0.58, mz);
        acc.addGeo(marker, [0.96, 0.60, 0.08], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 0.0);
      }
      // Front & rear rubber mudflaps behind wheel wells
      for (const fz of [1.56, -2.66]) {
        const mudflap = new THREE.BoxGeometry(0.32, 0.26, 0.024);
        mudflap.translate(sx * 1.00, 0.28, fz);
        acc.addGeo(mudflap, rubberTire, HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.84);
      }
    }

    // Front & Rear sculpted energy-absorbing bumpers with corner wraps
    for (const sz of [-1.0, 1.0]) {
      const bumper = new THREE.BoxGeometry(2.38, 0.25, 0.22);
      bumper.translate(0.0, 0.48, sz * 3.81);
      acc.addGeo(bumper, chassisDark, HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.90);
    }

    // 2. Main Iconic Bikun UI Alma Mater Yellow Hollow Perimeter Bodywork + Pearl-White Waistband
    for (const sx of [-1.0, 1.0]) {
      // Leave curbside (+X) front boarding doorway open for recessed stepwell at z = [2.28..3.42]!
      if (sx < 0) {
        const sideWallR = new THREE.BoxGeometry(0.11, 0.84, 7.66);
        sideWallR.translate(-1.135, 1.14, 0.0);
        acc.addGeo(sideWallR, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);
      } else {
        const sideWallLAft = new THREE.BoxGeometry(0.11, 0.84, 6.08);
        sideWallLAft.translate(1.135, 1.14, -0.78);
        acc.addGeo(sideWallLAft, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);

        const sideWallLFwd = new THREE.BoxGeometry(0.11, 0.84, 0.40);
        sideWallLFwd.translate(1.135, 1.14, 3.63);
        acc.addGeo(sideWallLFwd, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);
      }

      const sideWaist = new THREE.BoxGeometry(0.12, 0.11, sx < 0 ? 7.68 : 6.10);
      sideWaist.translate(sx * 1.14, 1.52, sx < 0 ? 0.0 : -0.78);
      acc.addGeo(sideWaist, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 1.0);
    }

    const rearWall = new THREE.BoxGeometry(2.38, 0.84, 0.14);
    rearWall.translate(0.0, 1.14, -3.77);
    acc.addGeo(rearWall, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);

    // Rear EV Inverter Louvered Access Hatch
    const rearHatch = new THREE.BoxGeometry(1.18, 0.46, 0.03);
    rearHatch.translate(0.0, 1.04, -3.845);
    acc.addGeo(rearHatch, [0.84, 0.62, 0.07], HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.94);
    for (let ly = 0.88; ly <= 1.18; ly += 0.075) {
      const louver = new THREE.BoxGeometry(0.96, 0.018, 0.042);
      louver.translate(0.0, ly, -3.85);
      acc.addGeo(louver, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.86);
    }

    const rearWaist = new THREE.BoxGeometry(2.38, 0.11, 0.15);
    rearWaist.translate(0.0, 1.52, -3.775);
    acc.addGeo(rearWaist, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 1.0);

    // Sculpted Front Transit Mask + Black Gloss Grille Accent Panel
    const frontNoseApron = new THREE.BoxGeometry(2.36, 0.76, 0.16);
    frontNoseApron.translate(0.0, 1.10, 3.76);
    acc.addGeo(frontNoseApron, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);

    const frontBlackSmile = new THREE.BoxGeometry(1.42, 0.24, 0.04);
    frontBlackSmile.translate(0.0, 0.86, 3.835);
    acc.addGeo(frontBlackSmile, chassisDark, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.95);

    const frontNoseWaist = new THREE.BoxGeometry(2.38, 0.10, 0.17);
    frontNoseWaist.translate(0.0, 1.46, 3.765);
    acc.addGeo(frontNoseWaist, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 1.0);

    // Flared Wheel Arch Fenders
    for (const sx of [-1.0, 1.0]) {
      for (const wz of [2.10, -2.10]) {
        const arch = new THREE.TorusGeometry(0.47, 0.065, 8, 16, Math.PI);
        arch.rotateY(Math.PI * 0.5);
        arch.translate(sx * 1.16, 0.43, wz);
        acc.addGeo(arch, chassisDark, HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.88);
      }
    }

    // Golden UI Makara Emblem Shields on Front Nose & Both Flanks!
    const noseCrest = new THREE.CylinderGeometry(0.14, 0.14, 0.03, 16, 1);
    noseCrest.rotateX(Math.PI * 0.5);
    noseCrest.translate(0.0, 1.18, 3.85);
    acc.addGeo(noseCrest, makaraGold, HUMAN_MAT.CREST_GOLD, BUS_J_CHASSIS, 1.0, 0.0, [0.5, 0.5]);

    for (const sx of [-1.0, 1.0]) {
      const sideCrest = new THREE.CylinderGeometry(0.16, 0.16, 0.025, 16, 1);
      sideCrest.rotateZ(Math.PI * 0.5);
      sideCrest.translate(sx * 1.195, 1.16, 0.15);
      acc.addGeo(sideCrest, makaraGold, HUMAN_MAT.CREST_GOLD, BUS_J_CHASSIS, 1.0, 0.0, [0.5, 0.5]);
    }

    // 3. Interior Cabin Floor, Recessed Stepwell with Yellow Safety Nosings, & Contoured Transit Seats
    const interiorCore = new THREE.BoxGeometry(2.14, 0.12, 7.10);
    interiorCore.translate(0.0, 0.78, 0.0);
    acc.addGeo(interiorCore, [0.22, 0.24, 0.27], HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.82);

    // Curbside (+X) Two-Tier Boarding Stepwell + High-Visibility Yellow Safety Nosings
    for (const [stX, stY] of [[1.04, 0.46], [0.88, 0.64]]) {
      const stepTread = new THREE.BoxGeometry(0.26, 0.04, 1.06);
      stepTread.translate(stX, stY, 2.86);
      acc.addGeo(stepTread, chassisDark, HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.86);

      const nosing = new THREE.BoxGeometry(0.035, 0.045, 1.06);
      nosing.translate(stX + 0.12, stY + 0.004, 2.86);
      acc.addGeo(nosing, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);
    }

    for (let rz = -2.85; rz <= 1.65; rz += 0.90) {
      for (const sx of [-0.68, 0.68]) {
        const seat = new THREE.BoxGeometry(0.74, 0.54, 0.46);
        seat.translate(sx, 1.14, rz);
        acc.addGeo(seat, seatBlue, HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.90);

        const headrest = new THREE.BoxGeometry(0.56, 0.22, 0.10);
        headrest.translate(sx, 1.49, rz - 0.16);
        acc.addGeo(headrest, [0.12, 0.24, 0.44], HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.92);

        const seatGrab = new THREE.TorusGeometry(0.065, 0.012, 6, 10, Math.PI);
        seatGrab.translate(sx - Math.sign(sx) * 0.28, 1.60, rz - 0.16);
        acc.addGeo(seatGrab, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.96);
      }
    }

    // Interior UI-Yellow Vertical Stanchions, Overhead Handrails, & Hanging Commuter Straps (z <= 1.8m, clear of cockpit!)
    for (const sz of [-2.40, -0.60, 1.20]) {
      for (const sx of [-0.30, 0.30]) {
        const stanchion = new THREE.CylinderGeometry(0.018, 0.018, 1.92, 8, 1);
        stanchion.translate(sx, 1.80, sz);
        acc.addGeo(stanchion, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.95);
      }
    }
    for (const sx of [-0.36, 0.36]) {
      const overheadRail = new THREE.CylinderGeometry(0.016, 0.016, 5.40, 8, 1);
      overheadRail.rotateX(Math.PI * 0.5);
      overheadRail.translate(sx, 2.58, -0.35);
      acc.addGeo(overheadRail, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.96);

      for (const hz of [-2.0, -1.0, 0.0, 1.0, 1.8]) {
        const strap = new THREE.BoxGeometry(0.018, 0.16, 0.028);
        strap.translate(sx, 2.48, hz);
        acc.addGeo(strap, chassisDark, HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.92);
        const ring = new THREE.TorusGeometry(0.042, 0.009, 6, 10);
        ring.rotateY(Math.PI * 0.5);
        ring.translate(sx, 2.37, hz);
        acc.addGeo(ring, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.96);
      }
    }

    // =========================================================================
    // 3B. SUPER-DETAILED RIGHT-HAND-DRIVE (RHD, -X) BUS KUNING DRIVER COCKPIT
    //     (FPS_BUS_DRIVER eye at [-0.68, 1.83, 3.08] kept 100% unobstructed!)
    // =========================================================================
    const drvPedestal = new THREE.CylinderGeometry(0.14, 0.17, 0.32, 10, 1);
    drvPedestal.translate(-0.68, 0.98, 2.94);
    acc.addGeo(drvPedestal, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.86);

    const drvCushion = new THREE.BoxGeometry(0.54, 0.12, 0.50);
    drvCushion.translate(-0.68, 1.18, 2.96);
    acc.addGeo(drvCushion, [0.15, 0.18, 0.24], HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.94);

    const drvBack = new THREE.BoxGeometry(0.50, 0.62, 0.11);
    drvBack.rotateX(-0.10);
    drvBack.translate(-0.68, 1.54, 2.73);
    acc.addGeo(drvBack, [0.15, 0.18, 0.24], HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.95);

    const drvPartition = new THREE.BoxGeometry(0.86, 1.05, 0.04);
    drvPartition.translate(-0.66, 1.42, 2.60);
    acc.addGeo(drvPartition, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.90);

    const drvPartGlass = new THREE.BoxGeometry(0.80, 0.72, 0.025);
    drvPartGlass.translate(-0.66, 2.24, 2.60);
    acc.addGeo(drvPartGlass, glassTint, HUMAN_MAT.VEHICLE_GLASS, BUS_J_CHASSIS, 0.98);

    // Uniformed Bikun Driver Torso, Shoulders, Arms, & Head (at z = 2.90m, safely behind FPS eye z = 3.08m!)
    const drvTorso = new THREE.CylinderGeometry(0.17, 0.15, 0.48, 10, 1);
    drvTorso.scale(1.15, 1.0, 0.78);
    drvTorso.translate(-0.68, 1.46, 2.91);
    acc.addGeo(drvTorso, bikunYellow, HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.96);

    const drvHead = new THREE.SphereGeometry(0.105, 12, 10);
    drvHead.scale(0.92, 1.08, 0.98);
    drvHead.translate(-0.68, 1.83, 2.90);
    acc.addGeo(drvHead, [0.76, 0.56, 0.42], HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.98);

    const drvCap = new THREE.CylinderGeometry(0.106, 0.110, 0.065, 12, 1);
    drvCap.translate(-0.68, 1.90, 2.90);
    acc.addGeo(drvCap, royalNavy, HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.98);

    for (const armS of [-1, 1]) {
      const drvArm = new THREE.CylinderGeometry(0.044, 0.038, 0.42, 8, 1);
      drvArm.rotateX(1.22);
      drvArm.rotateY(-armS * 0.22);
      drvArm.translate(-0.68 + armS * 0.21, 1.50, 3.13);
      acc.addGeo(drvArm, bikunYellow, HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.95);
    }

    // Wrap-Around Ergonomic Transit Dashboard, EV Speed/SOC Cluster (metaZ = 2.1), 4-Cam CCTV/GPS Monitor (metaZ = 2.2)
    const busDashMain = new THREE.BoxGeometry(2.16, 0.48, 0.42);
    busDashMain.translate(0.0, 1.18, 3.58);
    acc.addGeo(busDashMain, [0.14, 0.15, 0.18], HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.90);

    const busDashHood = new THREE.BoxGeometry(0.62, 0.16, 0.28);
    busDashHood.translate(-0.68, 1.45, 3.58);
    acc.addGeo(busDashHood, chassisDark, HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.94);

    const busClusterScreen = new THREE.BoxGeometry(0.38, 0.13, 0.022);
    busClusterScreen.rotateX(-0.28);
    busClusterScreen.translate(-0.68, 1.44, 3.47);
    acc.addGeo(busClusterScreen, [0.10, 0.42, 0.56], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 2.1);

    const busCctvMonitor = new THREE.BoxGeometry(0.25, 0.16, 0.026);
    busCctvMonitor.rotateY(0.28);
    busCctvMonitor.rotateX(-0.18);
    busCctvMonitor.translate(-0.22, 1.50, 3.48);
    acc.addGeo(busCctvMonitor, [0.14, 0.52, 0.44], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 2.2);

    // Right-Side Driver Control Pod: D-N-R Pushbuttons, Green/Red Door Open/Close Buttons, & Yellow Parking Brake Lever
    const sidePod = new THREE.BoxGeometry(0.18, 0.14, 0.46);
    sidePod.translate(-0.98, 1.34, 3.30);
    acc.addGeo(sidePod, [0.18, 0.20, 0.23], HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.92);

    const btnColors = [[0.18, 0.82, 0.36], [0.90, 0.68, 0.12], [0.88, 0.20, 0.16]];
    for (let bi = 0; bi < 3; bi++) {
      const btn = new THREE.BoxGeometry(0.032, 0.018, 0.032);
      btn.translate(-0.96, 1.418, 3.42 - bi * 0.052);
      acc.addGeo(btn, btnColors[bi], HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);
    }
    const parkValve = new THREE.CylinderGeometry(0.018, 0.014, 0.085, 8, 1);
    parkValve.rotateX(-0.24);
    parkValve.translate(-0.96, 1.44, 3.16);
    acc.addGeo(parkValve, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);

    // Driver Accelerator & Air-Brake Floor Pedals
    for (const px of [-0.60, -0.76]) {
      const bPedal = new THREE.BoxGeometry(0.075, 0.16, 0.022);
      bPedal.rotateX(-0.52);
      bPedal.translate(px, 0.91, 3.34);
      acc.addGeo(bPedal, chassisDark, HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.86);
    }

    // Curbside Front Boarding TapCash / UI SmartCard Fare Validator Terminal
    const tapPole = new THREE.CylinderGeometry(0.022, 0.022, 1.18, 8, 1);
    tapPole.translate(0.76, 1.32, 3.16);
    acc.addGeo(tapPole, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.95);

    const tapBox = new THREE.BoxGeometry(0.14, 0.21, 0.08);
    tapBox.rotateX(-0.35);
    tapBox.translate(0.76, 1.54, 3.16);
    acc.addGeo(tapBox, [0.12, 0.65, 0.85], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 2.1);

    // Angled Bus Steering Column + Stalks + ARTICULATED JOINT #5: 4-Spoke Transit Steering Wheel (`BUS_J_STEER_WHEEL`)
    const busSteerTilt = 0.44;
    const busCol = new THREE.CylinderGeometry(0.036, 0.044, 0.52, 10, 1);
    busCol.rotateX(busSteerTilt);
    busCol.translate(-0.68, 1.25, 3.47);
    acc.addGeo(busCol, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.90);

    for (const stSide of [-1, 1]) {
      const stalk = new THREE.CylinderGeometry(0.008, 0.009, 0.16, 6, 1);
      stalk.rotateZ(Math.PI * 0.5 - stSide * 0.16);
      stalk.translate(-0.68 + stSide * 0.11, 1.40, 3.40);
      acc.addGeo(stalk, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.92);
    }

    const busWheelRim = new THREE.TorusGeometry(0.225, 0.022, 10, 24);
    busWheelRim.rotateX(Math.PI * 0.5 - busSteerTilt);
    busWheelRim.translate(-0.68, 1.48, 3.36);
    acc.addGeo(busWheelRim, rubberTire, HUMAN_MAT.RUBBER_TIRE, BUS_J_STEER_WHEEL, 0.97);

    const busWheelHub = new THREE.CylinderGeometry(0.062, 0.068, 0.052, 14, 1);
    busWheelHub.rotateX(busSteerTilt);
    busWheelHub.translate(-0.68, 1.48, 3.36);
    acc.addGeo(busWheelHub, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_STEER_WHEEL, 0.98);

    const busHubCrest = new THREE.CylinderGeometry(0.036, 0.036, 0.056, 12, 1);
    busHubCrest.rotateX(busSteerTilt);
    busHubCrest.translate(-0.68, 1.483, 3.358);
    acc.addGeo(busHubCrest, makaraGold, HUMAN_MAT.CREST_GOLD, BUS_J_STEER_WHEEL, 1.0, 0.0, [0.5, 0.5]);

    for (let sp = 0; sp < 4; sp++) {
      const spAng = (sp / 4) * Math.PI * 2.0 + Math.PI * 0.25;
      const spoke = new THREE.BoxGeometry(0.20, 0.018, 0.028);
      spoke.translate(0.10, 0.0, 0.0);
      spoke.rotateZ(spAng);
      spoke.rotateX(Math.PI * 0.5 - busSteerTilt);
      spoke.translate(-0.68, 1.48, 3.36);
      acc.addGeo(spoke, alloySilver, HUMAN_MAT.METAL_ACCENT, BUS_J_STEER_WHEEL, 0.95);
    }

    // Twin Pantograph Front Windshield Wipers (parked low along sill at y = 1.51m!), Roof Hatches, & EV Blue-Strip License Plates
    for (const wxOff of [-0.48, 0.48]) {
      const wiper = new THREE.BoxGeometry(0.022, 0.58, 0.018);
      wiper.rotateZ(Math.PI * 0.5 - 0.08);
      wiper.rotateX(-0.11);
      wiper.translate(wxOff, 1.51, 3.84);
      acc.addGeo(wiper, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.94);
    }
    for (const hz of [-2.65, 2.25]) {
      const hatch = new THREE.BoxGeometry(0.68, 0.045, 0.68);
      hatch.translate(0.0, 2.89, hz);
      acc.addGeo(hatch, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);
    }
    for (const [pz, ry] of [[3.93, 0.0], [-3.93, Math.PI]]) {
      const plate = new THREE.BoxGeometry(0.42, 0.12, 0.014);
      plate.rotateY(ry);
      plate.translate(0.0, 0.49, pz);
      acc.addGeo(plate, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.96);
      const blueEv = new THREE.BoxGeometry(0.422, 0.026, 0.016);
      blueEv.rotateY(ry);
      blueEv.translate(0.0, 0.443, pz);
      acc.addGeo(blueEv, [0.10, 0.40, 0.84], HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);
    }

    // Curbside (+X) Articulated Pneumatic Glazed Boarding Door (`BUS_J_DOOR`) + Side LED Route Matrix Sign (`metaZ = 2.0`)
    const curbDoorHeader = new THREE.BoxGeometry(0.06, 0.14, 1.18);
    curbDoorHeader.translate(1.165, 2.48, 2.86);
    acc.addGeo(curbDoorHeader, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.94);
    for (const dz of [-0.27, 0.27]) {
      const doorFrame = new THREE.BoxGeometry(0.048, 1.78, 0.50);
      doorFrame.translate(1.16, 1.54, 2.86 + dz);
      acc.addGeo(doorFrame, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_DOOR, 0.94);

      const doorGlass = new THREE.BoxGeometry(0.056, 1.56, 0.40);
      doorGlass.translate(1.165, 1.58, 2.86 + dz);
      acc.addGeo(doorGlass, glassTint, HUMAN_MAT.VEHICLE_GLASS, BUS_J_DOOR, 0.98);

      const doorBar = new THREE.CylinderGeometry(0.012, 0.012, 0.44, 6, 1);
      doorBar.rotateX(Math.PI * 0.5);
      doorBar.translate(1.125, 1.46, 2.86 + dz);
      acc.addGeo(doorBar, makaraGold, HUMAN_MAT.CREST_GOLD, BUS_J_DOOR, 0.98);
    }

    const sideLedSign = new THREE.BoxGeometry(0.028, 0.15, 0.98);
    sideLedSign.translate(1.182, 2.44, 1.55);
    acc.addGeo(sideLedSign, [0.96, 0.66, 0.12], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 2.0);

    // 4. Panoramic Glazed Window Greenhouse, Structural Pillars, Anodized Window Mullions, Destination LED Matrix (`metaZ = 2.0`), & 4-Fan EV Roof Pod
    for (const sx of [-1.13, 1.13]) {
      for (const pz of [-3.74, -2.25, -0.75, 0.75, 2.22, 3.70]) {
        const pillar = new THREE.BoxGeometry(0.11, 1.06, 0.13);
        pillar.translate(sx, 2.08, pz);
        acc.addGeo(pillar, chassisDark, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.95);
      }
    }

    // Crisp Dark-Anodized Vertical Window Bay Mullions + Upper Hopper Transom Bar along both sides
    for (const mz of [-3.00, -1.50, 0.0, 1.48, 2.96]) {
      const mulR = new THREE.BoxGeometry(0.052, 0.98, 0.046);
      mulR.translate(-1.176, 2.08, mz);
      acc.addGeo(mulR, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.94);
    }
    for (const mz of [-3.00, -1.50, 0.0, 1.48]) {
      const mulL = new THREE.BoxGeometry(0.052, 0.98, 0.046);
      mulL.translate(1.176, 2.08, mz);
      acc.addGeo(mulL, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.94);
    }
    const transomR = new THREE.BoxGeometry(0.050, 0.032, 7.26);
    transomR.translate(-1.176, 2.34, 0.0);
    acc.addGeo(transomR, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.95);
    const transomL = new THREE.BoxGeometry(0.050, 0.032, 5.90);
    transomL.translate(1.176, 2.34, -0.72);
    acc.addGeo(transomL, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.95);

    const sideGlassR = new THREE.BoxGeometry(0.04, 0.98, 7.26);
    sideGlassR.translate(-1.175, 2.08, 0.0);
    acc.addGeo(sideGlassR, glassTint, HUMAN_MAT.VEHICLE_GLASS, BUS_J_CHASSIS, 1.0);

    const sideGlassL = new THREE.BoxGeometry(0.04, 0.98, 5.90);
    sideGlassL.translate(1.175, 2.08, -0.72);
    acc.addGeo(sideGlassL, glassTint, HUMAN_MAT.VEHICLE_GLASS, BUS_J_CHASSIS, 1.0);

    const frontWindshield = new THREE.BoxGeometry(2.18, 1.02, 0.06);
    frontWindshield.rotateX(-0.11);
    frontWindshield.translate(0.0, 2.07, 3.78);
    acc.addGeo(frontWindshield, glassTint, HUMAN_MAT.VEHICLE_GLASS, BUS_J_CHASSIS, 1.0);

    const rearGlass = new THREE.BoxGeometry(2.12, 0.88, 0.06);
    rearGlass.translate(0.0, 2.10, -3.81);
    acc.addGeo(rearGlass, glassTint, HUMAN_MAT.VEHICLE_GLASS, BUS_J_CHASSIS, 0.98);

    const headerCant = new THREE.BoxGeometry(2.38, 0.28, 7.66);
    headerCant.translate(0.0, 2.66, 0.0);
    acc.addGeo(headerCant, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 1.0);

    // Front Amber LED Dot-Matrix Route Display (`metaZ = 2.0`)
    const ledSignBezel = new THREE.BoxGeometry(1.64, 0.22, 0.035);
    ledSignBezel.translate(0.0, 2.65, 3.822);
    acc.addGeo(ledSignBezel, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.96);

    const ledRouteSign = new THREE.BoxGeometry(1.56, 0.18, 0.042);
    ledRouteSign.translate(0.0, 2.65, 3.83);
    acc.addGeo(ledRouteSign, [0.96, 0.66, 0.12], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 2.0);

    // Crowned Pearl-White Roof + Aerodynamic 4-Fan EV Battery/HVAC Pod + Orange HV Conduit (strictly peaking <= 2.96m!)
    const roofDeck = new THREE.BoxGeometry(2.26, 0.10, 7.52);
    roofDeck.translate(0.0, 2.83, 0.0);
    acc.addGeo(roofDeck, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 1.0);

    const evRoofPod = new THREE.BoxGeometry(1.64, 0.12, 3.95);
    evRoofPod.translate(0.0, 2.89, -0.35);
    acc.addGeo(evRoofPod, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 1.0);

    for (const fz of [-1.65, -0.35, 0.95]) {
      for (const fx of [-0.46, 0.46]) {
        const fanGrille = new THREE.CylinderGeometry(0.26, 0.28, 0.02, 12, 1);
        fanGrille.translate(fx, 2.95, fz);
        acc.addGeo(fanGrille, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.94);
      }
    }
    const hvCable = new THREE.CylinderGeometry(0.018, 0.018, 1.35, 8, 1);
    hvCable.rotateX(Math.PI * 0.5);
    hvCable.translate(-0.62, 2.89, -2.95);
    acc.addGeo(hvCable, hvOrange, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.96);

    // Iconic Curved Horn-Style Transit Side Mirrors ("Spion Tanduk") + Wide-Angle Spotter Mirrors
    for (const sx of [-1.0, 1.0]) {
      const hornArm = new THREE.CylinderGeometry(0.022, 0.026, 0.52, 8, 1);
      hornArm.rotateZ(sx * 0.72);
      hornArm.rotateY(-sx * 0.36);
      hornArm.translate(sx * 1.24, 2.48, 3.88);
      acc.addGeo(hornArm, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);

      const braceArm = new THREE.CylinderGeometry(0.014, 0.014, 0.34, 8, 1);
      braceArm.rotateZ(Math.PI * 0.5);
      braceArm.translate(sx * 1.26, 2.26, 3.92);
      acc.addGeo(braceArm, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.94);

      const mHousing = new THREE.BoxGeometry(0.11, 0.36, 0.16);
      mHousing.rotateY(sx * 0.22);
      mHousing.translate(sx * 1.41, 2.16, 3.98);
      acc.addGeo(mHousing, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);

      const mGlass = new THREE.BoxGeometry(0.012, 0.32, 0.135);
      mGlass.rotateY(sx * 0.22);
      mGlass.translate(sx * 1.39, 2.16, 3.92);
      acc.addGeo(mGlass, chromeSteel, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 1.0);

      const spotMirror = new THREE.BoxGeometry(0.09, 0.12, 0.13);
      spotMirror.rotateY(sx * 0.22);
      spotMirror.translate(sx * 1.41, 1.91, 3.98);
      acc.addGeo(spotMirror, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.95);
    }

    // 5. Dual-Projector LED Headlights, Corner Turn Lamps, High-Mount 3rd Brake Strip, & Ruby LED Taillights
    for (const sx of [-0.84, 0.84]) {
      const hlBezel = new THREE.BoxGeometry(0.44, 0.19, 0.05);
      hlBezel.translate(sx, 0.86, 3.83);
      acc.addGeo(hlBezel, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.92);

      for (const lx of [-0.09, 0.04]) {
        const proj = new THREE.CylinderGeometry(0.048, 0.048, 0.045, 12, 1);
        proj.rotateX(Math.PI * 0.5);
        proj.translate(sx + lx, 0.86, 3.85);
        acc.addGeo(proj, [0.92, 0.96, 1.0], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 0.0);
      }

      const cornerTurn = new THREE.BoxGeometry(0.085, 0.15, 0.045);
      cornerTurn.translate(sx + Math.sign(sx) * 0.165, 0.86, 3.845);
      acc.addGeo(cornerTurn, [0.95, 0.58, 0.08], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 0.0);

      const tailCluster = new THREE.BoxGeometry(0.30, 0.42, 0.05);
      tailCluster.translate(sx * 1.06, 1.04, -3.845);
      acc.addGeo(tailCluster, [0.58, 0.07, 0.07], HUMAN_MAT.VEHICLE_TAILLIGHT, BUS_J_CHASSIS, 1.0, 1.0);
    }
    const highBrakeBar = new THREE.BoxGeometry(0.92, 0.045, 0.035);
    highBrakeBar.translate(0.0, 2.66, -3.84);
    acc.addGeo(highBrakeBar, [0.64, 0.06, 0.06], HUMAN_MAT.VEHICLE_TAILLIGHT, BUS_J_CHASSIS, 1.0, 1.0);

    // 6. Four Articulated Heavy-Duty Alloy Transit Wheels (`BUS_J_WHEEL_FL..RR`, radius = 0.42m)
    //    - Front Axle: Single Steerable Tire + Ventilated Brake Rotor + Caliper + 10-Lug Hub
    //    - Rear Drive Axle: Authentic Dual Rear Tires ("Ban Belakang Ganda") + Deep-Dish Planetary Hub!
    const busWheels = [
      { joint: BUS_J_WHEEL_FL, pivot: BUS_BIND_PIVOTS[BUS_J_WHEEL_FL], side: 1, isRear: false },
      { joint: BUS_J_WHEEL_FR, pivot: BUS_BIND_PIVOTS[BUS_J_WHEEL_FR], side: -1, isRear: false },
      { joint: BUS_J_WHEEL_RL, pivot: BUS_BIND_PIVOTS[BUS_J_WHEEL_RL], side: 1, isRear: true },
      { joint: BUS_J_WHEEL_RR, pivot: BUS_BIND_PIVOTS[BUS_J_WHEEL_RR], side: -1, isRear: true }
    ];
    for (const ws of busWheels) {
      const [wx, wy, wz] = ws.pivot;
      const jId = ws.joint;

      if (!ws.isRear) {
        // Stationary front brake caliper + spinning ventilated rotor disc
        const caliper = new THREE.BoxGeometry(0.08, 0.14, 0.11);
        caliper.translate(wx - ws.side * 0.08, wy + 0.12, wz - 0.10);
        acc.addGeo(caliper, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.90);

        const rotor = new THREE.CylinderGeometry(0.21, 0.21, 0.026, 16, 1);
        rotor.rotateZ(Math.PI * 0.5);
        rotor.translate(wx - ws.side * 0.075, wy, wz);
        acc.addGeo(rotor, chromeSteel, HUMAN_MAT.METAL_ACCENT, jId, 0.88);
      }

      const tireOffsets = ws.isRear ? [0.02, -ws.side * 0.26] : [0.0];
      for (const txOff of tireOffsets) {
        const tire = new THREE.CylinderGeometry(BUS_WHEEL_RADIUS - 0.01, BUS_WHEEL_RADIUS - 0.01, 0.24, 22, 1);
        tire.rotateZ(Math.PI * 0.5);
        tire.translate(wx + txOff, wy, wz);
        acc.addGeo(tire, rubberTire, HUMAN_MAT.RUBBER_TIRE, jId, 0.92);

        for (let tb = 0; tb < 12; tb++) {
          const ang = (tb / 12) * Math.PI * 2.0;
          const lug = new THREE.BoxGeometry(0.246, 0.026, 0.074);
          lug.translate(0.0, BUS_WHEEL_RADIUS - 0.012, 0.0);
          lug.rotateX(ang);
          lug.translate(wx + txOff, wy, wz);
          acc.addGeo(lug, [0.13, 0.135, 0.145], HUMAN_MAT.RUBBER_TIRE, jId, 0.88);
        }
      }

      const rim = new THREE.CylinderGeometry(0.25, 0.25, 0.255, 18, 1);
      rim.rotateZ(Math.PI * 0.5);
      rim.translate(wx, wy, wz);
      acc.addGeo(rim, alloySilver, HUMAN_MAT.METAL_ACCENT, jId, 0.95);

      for (let sp = 0; sp < 8; sp++) {
        const spAng = (sp / 8) * Math.PI * 2.0;
        const spoke = new THREE.BoxGeometry(0.024, 0.22, 0.040);
        spoke.translate(0.0, 0.11, 0.0);
        spoke.rotateX(spAng);
        spoke.translate(wx + ws.side * 0.132, wy, wz);
        acc.addGeo(spoke, alloySilver, HUMAN_MAT.METAL_ACCENT, jId, 0.98);
      }

      // 10 Chrome Wheel Lug Studs + Planetary Center Hubcap
      for (let ln = 0; ln < 10; ln++) {
        const lAng = (ln / 10) * Math.PI * 2.0;
        const stud = new THREE.CylinderGeometry(0.011, 0.011, 0.024, 6, 1);
        stud.rotateZ(Math.PI * 0.5);
        stud.translate(wx + ws.side * 0.142, wy + Math.cos(lAng) * 0.125, wz + Math.sin(lAng) * 0.125);
        acc.addGeo(stud, chromeSteel, HUMAN_MAT.METAL_ACCENT, jId, 0.98);
      }

      const hub = new THREE.CylinderGeometry(0.082, 0.090, 0.068, 12, 1);
      hub.rotateZ(Math.PI * 0.5);
      hub.translate(wx + ws.side * 0.145, wy, wz);
      acc.addGeo(hub, makaraGold, HUMAN_MAT.CREST_GOLD, jId, 1.0);
    }

    return acc.toBundle();
  }

  /**
   * Builds the 16.4m x 6.2m UI Electric Passenger Catamaran Shuttle Ferry ("Kapal Feri Katamaran Listrik UI")
   * operating across Northern Selat UI between Dermaga Penyeberangan Utara Pulau U & Dermaga Pulau I:
   * - 2 Articulated Joints: `FERRY_J_HULL (0)` + `FERRY_J_RADAR (1)` spinning X-band open-array marine radar!
   * - True lofted wave-piercing aluminum catamaran demi-hulls (x = +-2.15m, waterline at y = 0.0m) + transom waterjets
   * - Caulked Sikaflex teak boarding promenade (`metaZ = 2.0`) at y = +1.23m matching both piers' landing stages
   * - Interior passenger seat rows, bow windlass/anchor/bollards, SOLAS liferafts & lifebuoys, upper sun-deck benches,
   *   bridge wing consoles, twin horn trumpets, VHF whip antennas, stern Merah-Putih ensign, & COLREGs lights!
   */
  static buildIslandShuttleFerry() {
    const acc = new RigidJointBundleAccumulator();

    const ferryWhite = [0.95, 0.96, 0.98];
    const uiYellow = [0.94, 0.70, 0.08];
    const royalNavy = [0.09, 0.20, 0.40];
    const teakDeck = [0.54, 0.36, 0.20];
    const stainless = [0.84, 0.87, 0.91];
    const darkSteel = [0.14, 0.16, 0.19];
    const solarBlue = [0.10, 0.22, 0.44];
    const cabinGlass = [0.16, 0.25, 0.34];
    const lifebuoyOrange = [0.94, 0.32, 0.07];
    const makaraGold = [0.91, 0.70, 0.15];

    // 1. Twin Lofted Wave-Piercing Aluminum Catamaran Demi-Hulls (+Z Bow, -Z Stern, x = +-2.15m) + Waterjet Nozzles
    for (const sx of [-2.15, 2.15]) {
      const demiLower = [
        { z: -7.55, keelY: -0.45, chineX: 0.56, chineY: -0.15, deckX: 0.70, deckY: 0.52 },
        { z: -4.20, keelY: -0.78, chineX: 0.64, chineY: -0.25, deckX: 0.74, deckY: 0.52 },
        { z: 0.50, keelY: -0.82, chineX: 0.66, chineY: -0.26, deckX: 0.76, deckY: 0.52 },
        { z: 4.60, keelY: -0.74, chineX: 0.52, chineY: -0.18, deckX: 0.64, deckY: 0.54 },
        { z: 6.80, keelY: -0.48, chineX: 0.24, chineY: -0.05, deckX: 0.34, deckY: 0.58 },
        { z: 8.15, keelY: -0.08, chineX: 0.05, chineY: 0.18, deckX: 0.08, deckY: 0.64 }
      ];
      acc.addGeo(buildLoftedHullStations(demiLower, sx, false), royalNavy, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 0.92);

      const demiUpper = [
        { z: -7.60, keelY: 0.50, chineX: 0.70, chineY: 0.78, deckX: 0.74, deckY: 1.06 },
        { z: -4.20, keelY: 0.50, chineX: 0.74, chineY: 0.80, deckX: 0.78, deckY: 1.06 },
        { z: 0.50, keelY: 0.50, chineX: 0.76, chineY: 0.82, deckX: 0.80, deckY: 1.06 },
        { z: 4.60, keelY: 0.52, chineX: 0.64, chineY: 0.84, deckX: 0.70, deckY: 1.08 },
        { z: 6.80, keelY: 0.56, chineX: 0.34, chineY: 0.88, deckX: 0.42, deckY: 1.12 },
        { z: 8.20, keelY: 0.62, chineX: 0.07, chineY: 0.92, deckX: 0.12, deckY: 1.16 }
      ];
      acc.addGeo(buildLoftedHullStations(demiUpper, sx, true), ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 0.96);

      // Transom Kamewa-style electric waterjet steering bucket & guard bar (-Z)
      const jetDuct = new THREE.CylinderGeometry(0.22, 0.28, 0.58, 12, 1);
      jetDuct.rotateX(Math.PI * 0.5);
      jetDuct.translate(sx, -0.10, -7.78);
      acc.addGeo(jetDuct, stainless, HUMAN_MAT.METAL_ACCENT, FERRY_J_HULL, 0.88);

      const jetStep = new THREE.BoxGeometry(1.36, 0.14, 0.86);
      jetStep.translate(sx, 0.42, -7.62);
      acc.addGeo(jetStep, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 0.92);
    }

    // 2. Cross-Deck Wet-Deck Bridging Box, Rubber Fenders, & Caulked Teak Boarding Promenade (`metaZ = 2.0`, y = 1.23m)
    const wetBridge = new THREE.BoxGeometry(5.96, 0.34, 14.30);
    wetBridge.translate(0.0, 1.04, -0.15);
    acc.addGeo(wetBridge, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 0.95);

    const yellowBelt = new THREE.BoxGeometry(6.14, 0.22, 14.45);
    yellowBelt.translate(0.0, 1.14, -0.15);
    acc.addGeo(yellowBelt, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 1.0);

    const rubStrake = new THREE.BoxGeometry(6.24, 0.14, 14.55);
    rubStrake.translate(0.0, 0.96, -0.15);
    acc.addGeo(rubStrake, [0.10, 0.11, 0.12], HUMAN_MAT.RUBBER_TIRE, FERRY_J_HULL, 0.90);

    const mainTeak = new THREE.BoxGeometry(5.86, 0.07, 14.15);
    mainTeak.translate(0.0, 1.23, -0.15);
    acc.addGeo(mainTeak, teakDeck, HUMAN_MAT.SHOE_BAG, FERRY_J_HULL, 0.97, 2.0);

    // Foredeck Electric Windlass & Cruciform Mooring Bollards at Bow & Stern Corners
    const ferryWindlass = new THREE.BoxGeometry(0.92, 0.36, 0.56);
    ferryWindlass.translate(0.0, 1.44, 6.25);
    acc.addGeo(ferryWindlass, darkSteel, HUMAN_MAT.METAL_ACCENT, FERRY_J_HULL, 0.94);

    for (const sx of [-2.55, 2.55]) {
      for (const bz of [-6.65, -0.85, 6.35]) {
        const bitt = new THREE.CylinderGeometry(0.085, 0.095, 0.32, 8, 1);
        bitt.translate(sx, 1.42, bz);
        acc.addGeo(bitt, stainless, HUMAN_MAT.METAL_ACCENT, FERRY_J_HULL, 0.96);
      }
    }

    // Port & Starboard Stainless Perimeter Guardrails + Sliding Boarding Gates + Banded SOLAS Lifebuoys + Liferaft Canisters
    for (const sx of [-2.88, 2.88]) {
      for (const seg of [
        [-6.9, -1.55],
        [1.15, 6.65]
      ]) {
        const len = seg[1] - seg[0];
        const midZ = 0.5 * (seg[0] + seg[1]);
        for (const ry of [2.12, 1.72]) {
          const rail = new THREE.CylinderGeometry(0.024, 0.024, len, 8, 1);
          rail.rotateX(Math.PI * 0.5);
          rail.translate(sx, ry, midZ);
          acc.addGeo(rail, stainless, HUMAN_MAT.METAL_ACCENT, FERRY_J_HULL, 0.98);
        }

        for (let pz = seg[0]; pz <= seg[1] + 0.01; pz += 1.70) {
          const post = new THREE.CylinderGeometry(0.022, 0.022, 0.88, 6, 1);
          post.translate(sx, 1.68, pz);
          acc.addGeo(post, stainless, HUMAN_MAT.METAL_ACCENT, FERRY_J_HULL, 0.96);
        }
      }

      // Broadside non-slip aluminum threshold plate at z = -0.20m
      const gateThreshold = new THREE.BoxGeometry(0.32, 0.03, 2.52);
      gateThreshold.translate(sx * 0.98, 1.275, -0.20);
      acc.addGeo(gateThreshold, stainless, HUMAN_MAT.METAL_ACCENT, FERRY_J_HULL, 0.98);

      // 2 Orange SOLAS Lifebuoys per Flank + White Reflective Bands
      for (const lz of [-3.65, 2.85]) {
        const ring = new THREE.TorusGeometry(0.27, 0.072, 8, 14);
        ring.rotateY(Math.PI * 0.5);
        ring.translate(sx * 0.84, 1.92, lz);
        acc.addGeo(ring, lifebuoyOrange, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 1.0);
      }

      // Aft SOLAS Cylindrical Inflatable Liferaft Canisters on Cradles
      const raftCan = new THREE.CylinderGeometry(0.28, 0.28, 1.05, 12, 1);
      raftCan.rotateX(Math.PI * 0.5);
      raftCan.translate(sx * 0.86, 1.58, -6.05);
      acc.addGeo(raftCan, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 0.99);
    }

    // 3. Panoramic Main Passenger Saloon Superstructure + Interior Commuter Seat Rows + UI Makara Badges
    const saloonLower = new THREE.BoxGeometry(4.65, 0.72, 9.80);
    saloonLower.translate(0.0, 1.62, -0.35);
    acc.addGeo(saloonLower, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 0.98);

    const saloonYellowBand = new THREE.BoxGeometry(4.70, 0.24, 9.84);
    saloonYellowBand.translate(0.0, 1.84, -0.35);
    acc.addGeo(saloonYellowBand, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 1.0);

    // Interior Passenger Seat Rows visible near windows
    for (let sz = -4.2; sz <= 3.4; sz += 1.25) {
      for (const ix of [-1.62, 1.62]) {
        const rowBench = new THREE.BoxGeometry(0.95, 0.48, 0.52);
        rowBench.translate(ix, 2.12, sz);
        acc.addGeo(rowBench, royalNavy, HUMAN_MAT.SHOE_BAG, FERRY_J_HULL, 0.88);
      }
    }

    // Framed Saloon Window Mullions + Warm Dusk/Night Illuminated Saloon Glass (`metaZ = 3.0`)
    const saloonGlass = new THREE.BoxGeometry(4.72, 0.72, 8.95);
    saloonGlass.translate(0.0, 2.32, -0.35);
    acc.addGeo(saloonGlass, cabinGlass, HUMAN_MAT.BOAT_LANTERN, FERRY_J_HULL, 1.0, 3.0);

    for (let mz = -4.4; mz <= 3.9; mz += 1.38) {
      const mullion = new THREE.BoxGeometry(4.76, 0.74, 0.09);
      mullion.translate(0.0, 2.32, mz);
      acc.addGeo(mullion, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 0.98);
    }

    const forwardSaloonGlass = new THREE.BoxGeometry(4.42, 0.82, 1.15);
    forwardSaloonGlass.rotateX(0.46);
    forwardSaloonGlass.translate(0.0, 2.26, 4.55);
    acc.addGeo(forwardSaloonGlass, cabinGlass, HUMAN_MAT.BOAT_LANTERN, FERRY_J_HULL, 1.0, 3.0);

    for (const sx of [-1.0, 1.0]) {
      const crest = new THREE.CylinderGeometry(0.22, 0.22, 0.03, 16, 1);
      crest.rotateZ(Math.PI * 0.5);
      crest.translate(sx * 2.37, 1.64, -0.20);
      acc.addGeo(crest, makaraGold, HUMAN_MAT.CREST_GOLD, FERRY_J_HULL, 1.0, 0.0, [0.5, 0.5]);
    }

    const upperDeckSlab = new THREE.BoxGeometry(4.92, 0.16, 10.60);
    upperDeckSlab.translate(0.0, 2.82, -0.45);
    acc.addGeo(upperDeckSlab, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 1.0);

    // 4. Raised Captain's Bridge Wheelhouse, Docking Bridge Wings, Aft Solar Canopy & Upper Sun-Deck Benches
    const bridgeCore = new THREE.BoxGeometry(3.35, 1.04, 3.45);
    bridgeCore.translate(0.0, 3.42, 1.85);
    acc.addGeo(bridgeCore, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 1.0);

    const bridgeWings = new THREE.BoxGeometry(4.85, 0.48, 1.18);
    bridgeWings.translate(0.0, 3.16, 2.15);
    acc.addGeo(bridgeWings, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 0.99);

    const bridgeGlass = new THREE.BoxGeometry(3.42, 0.52, 2.95);
    bridgeGlass.translate(0.0, 3.56, 1.98);
    acc.addGeo(bridgeGlass, cabinGlass, HUMAN_MAT.BOAT_LANTERN, FERRY_J_HULL, 1.0, 3.0);

    const bridgeRoof = new THREE.BoxGeometry(3.68, 0.14, 3.92);
    bridgeRoof.translate(0.0, 4.00, 1.85);
    acc.addGeo(bridgeRoof, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 1.0);

    // Twin Chrome Marine Air-Horn Trumpets on Bridge Roof
    for (const hx of [-0.92, 0.92]) {
      const horn = new THREE.ConeGeometry(0.065, 0.44, 10, 1);
      horn.rotateX(-Math.PI * 0.5);
      horn.translate(hx, 4.14, 3.55);
      acc.addGeo(horn, stainless, HUMAN_MAT.METAL_ACCENT, FERRY_J_HULL, 1.0);
    }

    // Aft Upper Sun-Deck Passenger Benches + Raised Solar PV Array Canopy on Stainless Struts
    for (const sx of [-1.45, 1.45]) {
      const deckBench = new THREE.BoxGeometry(0.55, 0.36, 3.80);
      deckBench.translate(sx, 3.08, -2.85);
      acc.addGeo(deckBench, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 0.96);

      for (const pz of [-4.8, -2.85, -0.9]) {
        const cStrut = new THREE.CylinderGeometry(0.032, 0.032, 0.92, 8, 1);
        cStrut.translate(sx * 1.32, 3.36, pz);
        acc.addGeo(cStrut, stainless, HUMAN_MAT.METAL_ACCENT, FERRY_J_HULL, 0.96);
      }
    }
    const solarPanel = new THREE.BoxGeometry(4.35, 0.08, 5.25);
    solarPanel.translate(0.0, 3.84, -2.85);
    acc.addGeo(solarPanel, solarBlue, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 0.99);

    // Stern Merah-Putih Indonesian Ensign Flagstaff
    const flagPole = new THREE.CylinderGeometry(0.022, 0.028, 1.65, 8, 1);
    flagPole.rotateX(-0.24);
    flagPole.translate(0.0, 2.02, -7.15);
    acc.addGeo(flagPole, stainless, HUMAN_MAT.METAL_ACCENT, FERRY_J_HULL, 0.98);

    const flagRed = new THREE.BoxGeometry(0.02, 0.24, 0.72);
    flagRed.translate(0.0, 2.66, -7.62);
    acc.addGeo(flagRed, [0.84, 0.12, 0.12], HUMAN_MAT.SHOE_BAG, FERRY_J_HULL, 1.0);

    const flagWhite = new THREE.BoxGeometry(0.02, 0.24, 0.72);
    flagWhite.translate(0.0, 2.42, -7.56);
    acc.addGeo(flagWhite, ferryWhite, HUMAN_MAT.SHOE_BAG, FERRY_J_HULL, 1.0);

    // 5. Radar Mast, VHF Antennas, COLREGs Lights, & ARTICULATED JOINT #1: Spinning X-Band Marine Radar (`FERRY_J_RADAR`)
    const mast = new THREE.CylinderGeometry(0.065, 0.095, 1.38, 8, 1);
    mast.translate(0.0, 4.74, 1.52);
    acc.addGeo(mast, ferryWhite, HUMAN_MAT.METAL_ACCENT, FERRY_J_HULL, 1.0);

    const radarPedestal = new THREE.CylinderGeometry(0.14, 0.18, 0.78, 10, 1);
    radarPedestal.translate(0.0, 4.46, 1.85);
    acc.addGeo(radarPedestal, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_HULL, 1.0);

    for (const vx of [-1.35, 1.35]) {
      const vhfWhip = new THREE.CylinderGeometry(0.012, 0.022, 1.55, 6, 1);
      vhfWhip.translate(vx, 4.82, 1.65);
      acc.addGeo(vhfWhip, ferryWhite, HUMAN_MAT.METAL_ACCENT, FERRY_J_HULL, 0.98);
    }

    // ARTICULATED SPINNING OPEN-ARRAY MARINE RADAR SCANNER (`FERRY_J_RADAR` at [0.0, 4.92, 1.85])
    const [rx, ry, rz] = FERRY_BIND_PIVOTS[FERRY_J_RADAR];
    const radarGearbox = new THREE.BoxGeometry(0.32, 0.14, 0.28);
    radarGearbox.translate(rx, ry - 0.04, rz);
    acc.addGeo(radarGearbox, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_RADAR, 1.0);

    const radarBar = new THREE.BoxGeometry(1.78, 0.11, 0.18);
    radarBar.translate(rx, ry + 0.06, rz);
    acc.addGeo(radarBar, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_RADAR, 1.0);

    const radarStripe = new THREE.BoxGeometry(1.79, 0.035, 0.185);
    radarStripe.translate(rx, ry + 0.06, rz);
    acc.addGeo(radarStripe, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, FERRY_J_RADAR, 1.0);

    const mastheadLight = new THREE.SphereGeometry(0.11, 10, 8);
    mastheadLight.translate(0.0, 5.48, 1.52);
    acc.addGeo(mastheadLight, [0.98, 0.96, 0.88], HUMAN_MAT.BOAT_LANTERN, FERRY_J_HULL, 1.0, 3.0);

    const stbdGreen = new THREE.SphereGeometry(0.10, 8, 6);
    stbdGreen.translate(-2.38, 3.48, 2.45);
    acc.addGeo(stbdGreen, [0.16, 0.90, 0.38], HUMAN_MAT.BOAT_LANTERN, FERRY_J_HULL, 1.0, 1.0);

    const portRed = new THREE.SphereGeometry(0.10, 8, 6);
    portRed.translate(2.38, 3.48, 2.45);
    acc.addGeo(portRed, [0.92, 0.16, 0.14], HUMAN_MAT.BOAT_LANTERN, FERRY_J_HULL, 1.0, 2.0);

    return acc.toBundle();
  }

  /**
   * Builds a Modern 2-Passenger Electric Helicopter Taxi ("Taxi Helicopter Listrik UI" — 2-Seat Multi-Rotor eVTOL):
   * - 7 Articulated Joints: 0 = Carbon-Pearl Aerodynamic Cabin/Skids/Halo, 1..6 = 6 Counter-Rotating Scimitar Electric Rotors
   * - Lofted Teardrop Carbon Pod, Gullwing Door Seams, NACA Battery Cooling Scoops, Oleo Skid Dampers, Tail Stinger
   * - Panoramic Synthetic-Vision PFD (`metaZ = 2.3`), Collective Lever, Sidestick, 4-Point Harnesses, Airfoil Canted V-Tail!
   */
  static buildElectricHeliTaxiEVTOL() {
    const acc = new RigidJointBundleAccumulator();

    const pearlWhite = [0.95, 0.96, 0.98];
    const uiGold = [0.94, 0.71, 0.10];
    const royalNavy = [0.09, 0.20, 0.40];
    const carbonDark = [0.11, 0.12, 0.14];
    const titanium = [0.68, 0.72, 0.77];
    const canopyGlass = [0.18, 0.28, 0.38];
    const seatCream = [0.87, 0.83, 0.75];
    const makaraGold = [0.92, 0.72, 0.16];

    // 1. Tubular Titanium-Carbon Landing Skids (bottom strictly at y = 0.0m!), Oleo Damper Collars, & Boarding Steps
    for (const sx of [-0.88, 0.88]) {
      const skid = new THREE.CylinderGeometry(0.036, 0.036, 2.92, 10, 1);
      skid.rotateX(Math.PI * 0.5);
      skid.translate(sx, 0.036, 0.06);
      acc.addGeo(skid, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.90);

      for (const wz of [-0.95, 0.05, 1.05]) {
        const wearShoe = new THREE.CylinderGeometry(0.042, 0.042, 0.16, 8, 1);
        wearShoe.rotateX(Math.PI * 0.5);
        wearShoe.translate(sx, 0.040, wz);
        acc.addGeo(wearShoe, titanium, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.88);
      }

      const skiTip = new THREE.CylinderGeometry(0.024, 0.036, 0.36, 8, 1);
      skiTip.rotateX(Math.PI * 0.5 - 0.48);
      skiTip.translate(sx, 0.11, 1.65);
      acc.addGeo(skiTip, uiGold, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 0.96);

      // Front & Rear cross-arch shock struts + titanium oleo damper sleeves
      for (const sz of [-0.66, 0.74]) {
        const strut = new THREE.CylinderGeometry(0.028, 0.032, 0.52, 8, 1);
        strut.rotateZ(-Math.sign(sx) * 0.52);
        strut.translate(sx * 0.82, 0.24, sz);
        acc.addGeo(strut, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.90);

        const oleoSleeve = new THREE.CylinderGeometry(0.038, 0.038, 0.16, 8, 1);
        oleoSleeve.rotateZ(-Math.sign(sx) * 0.52);
        oleoSleeve.translate(sx * 0.76, 0.30, sz);
        acc.addGeo(oleoSleeve, titanium, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.95);
      }

      const stepBar = new THREE.BoxGeometry(0.14, 0.025, 0.72);
      stepBar.translate(sx * 0.90, 0.23, 0.18);
      acc.addGeo(stepBar, titanium, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.94);
    }

    // 2. Lofted Aerodynamic Teardrop 2-Seat Carbon Cabin Pod (+Z Nose Forward) + NACA Cooling Scoops
    const evtolPodStations = [
      { z: -1.18, keelY: 0.54, chineX: 0.48, chineY: 0.76, deckX: 0.52, deckY: 1.02 },
      { z: -0.35, keelY: 0.40, chineX: 0.78, chineY: 0.68, deckX: 0.84, deckY: 0.98 },
      { z: 0.65, keelY: 0.39, chineX: 0.80, chineY: 0.66, deckX: 0.84, deckY: 0.96 },
      { z: 1.55, keelY: 0.45, chineX: 0.62, chineY: 0.65, deckX: 0.64, deckY: 0.90 },
      { z: 2.28, keelY: 0.58, chineX: 0.18, chineY: 0.68, deckX: 0.20, deckY: 0.78 }
    ];
    acc.addGeo(buildLoftedHullStations(evtolPodStations, 0.0, true), pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 0.98);

    const bellyKeel = new THREE.BoxGeometry(1.18, 0.12, 2.35);
    bellyKeel.translate(0.0, 0.42, 0.18);
    acc.addGeo(bellyKeel, carbonDark, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 0.90);

    // Nose Pitot Tube + Dual LED Landing Spotlights
    const pitotTube = new THREE.CylinderGeometry(0.008, 0.012, 0.26, 6, 1);
    pitotTube.rotateX(Math.PI * 0.5);
    pitotTube.translate(0.18, 0.68, 2.38);
    acc.addGeo(pitotTube, titanium, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.98);

    const noseSpotlight = new THREE.BoxGeometry(0.36, 0.08, 0.05);
    noseSpotlight.translate(0.0, 0.66, 2.28);
    acc.addGeo(noseSpotlight, [0.92, 0.96, 1.00], HUMAN_MAT.VEHICLE_HEADLIGHT, EVTOL_J_FUSELAGE, 1.0, 0.0);

    // UI Alma Mater Gold & Royal Navy Speed Cheatlines, NACA Battery Scoops, & Golden UI Makara Crests
    for (const sx of [-1.0, 1.0]) {
      const goldSide = new THREE.BoxGeometry(0.04, 0.095, 2.28);
      goldSide.translate(sx * 0.835, 0.84, 0.18);
      acc.addGeo(goldSide, uiGold, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 1.0);

      const navySide = new THREE.BoxGeometry(0.038, 0.05, 2.18);
      navySide.translate(sx * 0.832, 0.76, 0.16);
      acc.addGeo(navySide, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 0.98);

      const nacaScoop = new THREE.BoxGeometry(0.06, 0.11, 0.34);
      nacaScoop.translate(sx * 0.68, 0.92, -0.78);
      acc.addGeo(nacaScoop, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.86);

      const doorCrest = new THREE.CylinderGeometry(0.11, 0.11, 0.022, 14, 1);
      doorCrest.rotateZ(Math.PI * 0.5);
      doorCrest.translate(sx * 0.848, 0.84, 0.28);
      acc.addGeo(doorCrest, makaraGold, HUMAN_MAT.CREST_GOLD, EVTOL_J_FUSELAGE, 1.0, 0.0, [0.5, 0.5]);
    }

    // 3. Super-Detailed 2-Passenger Cockpit Interior: Panoramic Synthetic-Vision PFD (`metaZ = 2.3`), Collective, Sidestick, & 4-Point Harnesses
    const cabinFloor = new THREE.BoxGeometry(1.36, 0.06, 1.95);
    cabinFloor.translate(0.0, 0.56, 0.28);
    acc.addGeo(cabinFloor, [0.18, 0.19, 0.22], HUMAN_MAT.RUBBER_TIRE, EVTOL_J_FUSELAGE, 0.88);

    const centerPedestal = new THREE.BoxGeometry(0.24, 0.34, 0.58);
    centerPedestal.translate(0.0, 0.76, 0.76);
    acc.addGeo(centerPedestal, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.92);

    const flightScreenBezel = new THREE.BoxGeometry(0.46, 0.24, 0.022);
    flightScreenBezel.rotateX(-0.36);
    flightScreenBezel.translate(0.0, 1.02, 0.945);
    acc.addGeo(flightScreenBezel, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.96);

    const flightScreen = new THREE.BoxGeometry(0.42, 0.21, 0.026);
    flightScreen.rotateX(-0.36);
    flightScreen.translate(0.0, 1.02, 0.94);
    acc.addGeo(flightScreen, [0.10, 0.56, 0.74], HUMAN_MAT.VEHICLE_HEADLIGHT, EVTOL_J_FUSELAGE, 1.0, 2.3);

    // Pilot Right-Hand Fly-By-Wire Sidestick + Center Collective Pitch Lever
    const sideStick = new THREE.CylinderGeometry(0.016, 0.020, 0.16, 8, 1);
    sideStick.rotateX(0.18);
    sideStick.translate(0.62, 0.82, 0.42);
    acc.addGeo(sideStick, carbonDark, HUMAN_MAT.RUBBER_TIRE, EVTOL_J_FUSELAGE, 0.95);

    const collective = new THREE.CylinderGeometry(0.016, 0.018, 0.32, 8, 1);
    collective.rotateX(Math.PI * 0.5 - 0.38);
    collective.translate(0.12, 0.74, 0.36);
    acc.addGeo(collective, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.94);

    for (const [seatIdx, sx] of [[0, 0.37], [1, -0.37]]) {
      const seatBase = new THREE.BoxGeometry(0.46, 0.10, 0.46);
      seatBase.translate(sx, 0.66, 0.12);
      acc.addGeo(seatBase, seatCream, HUMAN_MAT.SHOE_BAG, EVTOL_J_FUSELAGE, 0.95);

      const seatBack = new THREE.BoxGeometry(0.44, 0.58, 0.09);
      seatBack.rotateX(-0.14);
      seatBack.translate(sx, 0.98, -0.10);
      acc.addGeo(seatBack, seatCream, HUMAN_MAT.SHOE_BAG, EVTOL_J_FUSELAGE, 0.96);

      const occJacket = seatIdx === 0 ? royalNavy : uiGold;
      const occTorso = new THREE.CylinderGeometry(0.14, 0.13, 0.44, 10, 1);
      occTorso.scale(1.12, 1.0, 0.78);
      occTorso.translate(sx, 0.94, 0.06);
      acc.addGeo(occTorso, occJacket, HUMAN_MAT.SHOE_BAG, EVTOL_J_FUSELAGE, 0.96);

      // 4-Point Aviation Safety Harness Straps over shoulders
      for (const hs of [-0.065, 0.065]) {
        const strap = new THREE.BoxGeometry(0.028, 0.34, 0.025);
        strap.rotateX(-0.12);
        strap.translate(sx + hs, 0.98, 0.16);
        acc.addGeo(strap, carbonDark, HUMAN_MAT.RUBBER_TIRE, EVTOL_J_FUSELAGE, 0.94);
      }

      const occHead = new THREE.SphereGeometry(0.096, 10, 8);
      occHead.translate(sx, 1.28, 0.04);
      acc.addGeo(occHead, [0.76, 0.56, 0.42], HUMAN_MAT.SHOE_BAG, EVTOL_J_FUSELAGE, 0.98);
    }

    // Panoramic Bubble Canopy Glass & Sleek Carbon Gullwing Door Arches
    const canopyDome = new THREE.SphereGeometry(0.76, 18, 12);
    canopyDome.scale(0.98, 0.76, 1.55);
    canopyDome.translate(0.0, 1.18, 0.36);
    acc.addGeo(canopyDome, canopyGlass, HUMAN_MAT.VEHICLE_GLASS, EVTOL_J_FUSELAGE, 1.0);

    for (const sx of [-0.74, 0.74]) {
      const gullArch = new THREE.TorusGeometry(0.56, 0.018, 6, 16, Math.PI * 0.85);
      gullArch.rotateY(Math.PI * 0.5);
      gullArch.translate(sx, 1.06, 0.24);
      acc.addGeo(gullArch, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.96);
    }

    const roofSpine = new THREE.BoxGeometry(0.38, 0.14, 1.72);
    roofSpine.translate(0.0, 1.72, -0.12);
    acc.addGeo(roofSpine, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 1.0);

    // 4. Twin Swept Carbon Tail Booms, Airfoil Canted V-Tail Ruddervators, Tail Bumper Stinger, & Aviation Strobes
    for (const sx of [-0.54, 0.54]) {
      const boom = new THREE.CylinderGeometry(0.048, 0.075, 1.92, 10, 1);
      boom.rotateX(Math.PI * 0.5 + 0.08);
      boom.translate(sx, 1.06, -1.78);
      acc.addGeo(boom, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 0.98);

      const vFin = buildAirfoilWingPanel({
        side: 1,
        span: 0.84,
        rootChord: 0.52,
        tipChord: 0.32,
        thicknessRatio: 0.11,
        camber: 0.0,
        sweepZ: -0.22
      });
      vFin.rotateZ(Math.PI * 0.5 - Math.sign(sx) * 0.30);
      vFin.translate(sx, 1.08, -2.52);
      acc.addGeo(vFin, uiGold, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 1.0);
    }

    const hTail = buildAirfoilWingPanel({
      side: 1,
      span: 1.12,
      rootChord: 0.38,
      tipChord: 0.38,
      thicknessRatio: 0.10,
      camber: 0.0
    });
    hTail.translate(-0.56, 1.12, -2.56);
    acc.addGeo(hTail, carbonDark, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 0.96);

    const tailStinger = new THREE.CylinderGeometry(0.012, 0.016, 0.36, 6, 1);
    tailStinger.rotateX(0.65);
    tailStinger.translate(0.0, 0.94, -2.62);
    acc.addGeo(tailStinger, titanium, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.92);

    const tailBeacon = new THREE.SphereGeometry(0.055, 8, 6);
    tailBeacon.translate(0.0, 1.22, -2.76);
    acc.addGeo(tailBeacon, [0.98, 0.96, 0.90], HUMAN_MAT.BOAT_LANTERN, EVTOL_J_FUSELAGE, 1.0, 3.0);

    // 5. Overhead Multi-Rotor Structural Carbon Halo Ring, Finned Motor Nacelles, & 6 Scimitar Rotors (JOINTS 1..6)
    for (const sx of [-0.28, 0.28]) {
      for (const sz of [-0.46, 0.16]) {
        const mast = new THREE.CylinderGeometry(0.036, 0.044, 0.58, 8, 1);
        mast.rotateZ(-sx * 0.35);
        mast.translate(sx * 1.2, 1.98, sz);
        acc.addGeo(mast, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.95);
      }
    }

    const haloRing = new THREE.TorusGeometry(EVTOL_ROTOR_RING_R, 0.042, 8, 36);
    haloRing.rotateX(Math.PI * 0.5);
    haloRing.translate(0.0, 2.24, 0.0);
    acc.addGeo(haloRing, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 1.0);

    const portNav = new THREE.SphereGeometry(0.06, 8, 6);
    portNav.translate(EVTOL_ROTOR_RING_R + 0.03, 2.24, 0.0);
    acc.addGeo(portNav, [0.95, 0.16, 0.14], HUMAN_MAT.BOAT_LANTERN, EVTOL_J_FUSELAGE, 1.0, 2.0);

    const stbdNav = new THREE.SphereGeometry(0.06, 8, 6);
    stbdNav.translate(-(EVTOL_ROTOR_RING_R + 0.03), 2.24, 0.0);
    acc.addGeo(stbdNav, [0.16, 0.92, 0.36], HUMAN_MAT.BOAT_LANTERN, EVTOL_J_FUSELAGE, 1.0, 1.0);

    for (let rIdx = 0; rIdx < 6; rIdx++) {
      const jointId = rIdx + 1;
      const [hx, hy, hz] = EVTOL_BIND_PIVOTS[jointId];
      const ang = (rIdx / 6) * Math.PI * 2.0;
      const spinSign = rIdx % 2 === 0 ? 1.0 : -1.0;

      const arm = new THREE.CylinderGeometry(0.028, 0.034, EVTOL_ROTOR_RING_R, 8, 1);
      arm.rotateZ(Math.PI * 0.5);
      arm.rotateY(-ang);
      arm.translate(hx * 0.5, 2.24, hz * 0.5);
      acc.addGeo(arm, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.96);

      const motorPod = new THREE.CylinderGeometry(0.10, 0.088, 0.17, 12, 1);
      motorPod.translate(hx, 2.245, hz);
      acc.addGeo(motorPod, uiGold, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 1.0);

      const coolingRing = new THREE.TorusGeometry(0.102, 0.012, 6, 14);
      coolingRing.rotateX(Math.PI * 0.5);
      coolingRing.translate(hx, 2.245, hz);
      acc.addGeo(coolingRing, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.92);

      // Articulated Spinning Rotor Spinner + 3 Twisted Tapered Scimitar Carbon Blades (`jointId = 1..6`)
      const spinner = new THREE.ConeGeometry(0.075, 0.10, 12, 1);
      spinner.translate(hx, hy + 0.035, hz);
      acc.addGeo(spinner, titanium, HUMAN_MAT.METAL_ACCENT, jointId, 1.0);

      for (let b = 0; b < 3; b++) {
        const bAng = (b / 3) * Math.PI * 2.0;
        const bladeInboard = new THREE.BoxGeometry(0.56, 0.013, 0.090);
        bladeInboard.rotateX(spinSign * 0.18);
        bladeInboard.translate(0.32, 0.0, 0.0);
        bladeInboard.rotateY(bAng);
        bladeInboard.translate(hx, hy, hz);
        acc.addGeo(bladeInboard, carbonDark, HUMAN_MAT.METAL_ACCENT, jointId, 0.98);

        const scimitarTip = new THREE.BoxGeometry(0.32, 0.012, 0.068);
        scimitarTip.rotateX(spinSign * 0.10);
        scimitarTip.rotateY(-spinSign * 0.24);
        scimitarTip.translate(0.73, 0.0, -spinSign * 0.035);
        scimitarTip.rotateY(bAng);
        scimitarTip.translate(hx, hy, hz);
        acc.addGeo(scimitarTip, uiGold, HUMAN_MAT.VEHICLE_GLOSS, jointId, 1.0);
      }
    }

    return acc.toBundle();
  }

  /**
   * Builds a Super-Detailed High-Wing Strut-Braced Cessna 172 Skyhawk Campus Shuttle Aircraft ("PK-UIA"):
   * - 2 Articulated Joints: 0 = Airframe/Cockpit/Gear, 1 = Spinning Twisted 2-Blade McCauley Propeller ([0, 1.18, 2.58])
   * - True NACA 2412 Cambered High Wing (11.0m span, dihedral & washout) + Flap-Track Fairings + Strut Cuffs + Pitot Tube
   * - Airfoil Horizontal Stabilizer & Swept Vertical Fin, Torque-Link Nose Gear, Lycoming Cowling Baffles & Exhausts
   * - Full Garmin G1000 Cockpit: Artificial Horizon PFD (`metaZ = 2.3`), Moving-Map MFD (`metaZ = 2.2`), Vernier Throttle/Mixture,
   *   Trim Wheel, Rudder Pedals, Standby Dials, Dual Yokes, & Unobstructed Captain Sightline at `[0.27, 1.54, 0.25]`!
   */
  static buildCessna172Aircraft() {
    const acc = new RigidJointBundleAccumulator();

    const aviationWhite = [0.95, 0.96, 0.98];
    const uiGold = [0.93, 0.70, 0.10];
    const royalNavy = [0.09, 0.20, 0.42];
    const darkMetal = [0.12, 0.13, 0.16];
    const rubberTire = [0.085, 0.088, 0.095];
    const chrome = [0.84, 0.87, 0.91];
    const glassTint = [0.20, 0.31, 0.42];
    const seatTan = [0.76, 0.68, 0.56];
    const makaraGold = [0.92, 0.72, 0.16];

    // 1. Tricycle Landing Gear + Torque Scissors + Aerodynamic Wheel Pants (bottom strictly at y = 0.0m!)
    const noseStrut = new THREE.CylinderGeometry(0.032, 0.028, 0.56, 10, 1);
    noseStrut.rotateX(-0.14);
    noseStrut.translate(0.0, 0.44, 1.80);
    acc.addGeo(noseStrut, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.94);

    for (const sc of [-1, 1]) {
      const scissor = new THREE.BoxGeometry(0.022, 0.11, 0.018);
      scissor.rotateX(sc * 0.52);
      scissor.translate(0.0, 0.38 + sc * 0.035, 1.75);
      acc.addGeo(scissor, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.92);
    }

    const noseTire = new THREE.CylinderGeometry(0.18, 0.18, 0.11, 16, 1);
    noseTire.rotateZ(Math.PI * 0.5);
    noseTire.translate(0.0, 0.18, 1.84);
    acc.addGeo(noseTire, rubberTire, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.90);

    const nosePant = new THREE.SphereGeometry(0.18, 12, 10);
    nosePant.scale(0.55, 0.72, 1.88);
    nosePant.translate(0.0, 0.23, 1.82);
    acc.addGeo(nosePant, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

    // Main Spring-Steel Landing Gear Legs, Boarding Steps, & Wheel Pants (x = +-1.28m, z = -0.16m)
    for (const sx of [-1.0, 1.0]) {
      const gearLeg = new THREE.BoxGeometry(0.86, 0.055, 0.12);
      gearLeg.rotateZ(-sx * 0.46);
      gearLeg.translate(sx * 0.86, 0.48, -0.16);
      acc.addGeo(gearLeg, aviationWhite, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.94);

      const crewStep = new THREE.BoxGeometry(0.16, 0.022, 0.08);
      crewStep.translate(sx * 0.74, 0.54, 0.24);
      acc.addGeo(crewStep, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.94);

      const mainTire = new THREE.CylinderGeometry(0.20, 0.20, 0.13, 16, 1);
      mainTire.rotateZ(Math.PI * 0.5);
      mainTire.translate(sx * 1.28, 0.20, -0.16);
      acc.addGeo(mainTire, rubberTire, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.90);

      const mainPant = new THREE.SphereGeometry(0.20, 12, 10);
      mainPant.scale(0.56, 0.74, 1.94);
      mainPant.translate(sx * 1.28, 0.25, -0.18);
      acc.addGeo(mainPant, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

      const pantStripe = new THREE.BoxGeometry(0.24, 0.04, 0.56);
      pantStripe.translate(sx * 1.28, 0.26, -0.18);
      acc.addGeo(pantStripe, uiGold, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);
    }

    // 2. Lofted Lycoming Engine Cowling (strictly below 1.36m glareshield!), Cabin Fuselage, & Empennage
    const cowlingStations = [
      { z: 1.24, keelY: 0.70, chineX: 0.56, chineY: 0.98, deckX: 0.54, deckY: 1.34 },
      { z: 1.88, keelY: 0.74, chineX: 0.52, chineY: 1.00, deckX: 0.48, deckY: 1.30 },
      { z: 2.48, keelY: 0.84, chineX: 0.42, chineY: 1.04, deckX: 0.36, deckY: 1.24 }
    ];
    acc.addGeo(buildLoftedHullStations(cowlingStations, 0.0, true), aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

    for (const sx of [-0.20, 0.20]) {
      const intake = new THREE.BoxGeometry(0.16, 0.10, 0.06);
      intake.translate(sx, 1.12, 2.49);
      acc.addGeo(intake, darkMetal, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.80);

      const exhaust = new THREE.CylinderGeometry(0.026, 0.026, 0.28, 8, 1);
      exhaust.rotateX(0.42);
      exhaust.translate(sx * 0.85, 0.65, 1.70);
      acc.addGeo(exhaust, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.90);
    }
    const oilCooler = new THREE.BoxGeometry(0.14, 0.08, 0.05);
    oilCooler.translate(0.0, 0.94, 2.49);
    acc.addGeo(oilCooler, darkMetal, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.82);

    // Hollow Lower Cabin Belly Floor & Side Fuselage Walls (100% open interior for Garmin G1000 & Yoke in FPS_PILOT!)
    const cabinFloor = new THREE.BoxGeometry(1.18, 0.14, 2.38);
    cabinFloor.translate(0.0, 0.73, 0.12);
    acc.addGeo(cabinFloor, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.96);

    const cabinAftBulkhead = new THREE.BoxGeometry(1.16, 0.72, 0.12);
    cabinAftBulkhead.translate(0.0, 1.02, -1.01);
    acc.addGeo(cabinAftBulkhead, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.96);

    for (const sx of [-1.0, 1.0]) {
      const cabinSideWall = new THREE.BoxGeometry(0.07, 0.72, 2.38);
      cabinSideWall.translate(sx * 0.555, 1.02, 0.12);
      acc.addGeo(cabinSideWall, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

      const doorHandle = new THREE.BoxGeometry(0.025, 0.028, 0.11);
      doorHandle.translate(sx * 0.598, 1.16, -0.05);
      acc.addGeo(doorHandle, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.98);

      const goldCheatSide = new THREE.BoxGeometry(0.04, 0.09, 5.45);
      goldCheatSide.rotateX(0.035);
      goldCheatSide.translate(sx * 0.588, 1.18, -0.82);
      acc.addGeo(goldCheatSide, uiGold, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

      const navyCheatSide = new THREE.BoxGeometry(0.038, 0.055, 5.35);
      navyCheatSide.rotateX(0.035);
      navyCheatSide.translate(sx * 0.586, 1.09, -0.85);
      acc.addGeo(navyCheatSide, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);
    }

    // Lofted Tapered Aft Tailcone (z = -1.05m .. -4.25m) + Rear Baggage Window + Tail Tie-Down Ring
    const tailConeStations = [
      { z: -4.25, keelY: 1.18, chineX: 0.12, chineY: 1.30, deckX: 0.11, deckY: 1.44 },
      { z: -2.65, keelY: 0.92, chineX: 0.34, chineY: 1.18, deckX: 0.32, deckY: 1.52 },
      { z: -1.02, keelY: 0.68, chineX: 0.56, chineY: 1.08, deckX: 0.54, deckY: 1.68 }
    ];
    acc.addGeo(buildLoftedHullStations(tailConeStations, 0.0, true), aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

    // Swept Dorsal Fin + Airfoil Vertical Stabilizer/Rudder + Airfoil Horizontal Stabilizer/Elevator
    const dorsalFillet = new THREE.BoxGeometry(0.07, 0.32, 1.38);
    dorsalFillet.rotateX(-0.24);
    dorsalFillet.translate(0.0, 1.58, -2.92);
    acc.addGeo(dorsalFillet, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

    const vStab = buildAirfoilWingPanel({
      side: 1,
      span: 1.38,
      rootChord: 1.12,
      tipChord: 0.58,
      thicknessRatio: 0.10,
      camber: 0.0,
      sweepZ: -0.46
    });
    vStab.rotateZ(Math.PI * 0.5);
    vStab.translate(0.0, 1.38, -3.72);
    acc.addGeo(vStab, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

    const tailBand = new THREE.BoxGeometry(0.095, 0.22, 0.68);
    tailBand.translate(0.0, 2.42, -4.02);
    acc.addGeo(tailBand, uiGold, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

    for (const sx of [-1.0, 1.0]) {
      const tailCrest = new THREE.CylinderGeometry(0.14, 0.14, 0.02, 14, 1);
      tailCrest.rotateZ(Math.PI * 0.5);
      tailCrest.translate(sx * 0.055, 2.02, -3.88);
      acc.addGeo(tailCrest, makaraGold, HUMAN_MAT.CREST_GOLD, CESSNA_J_AIRFRAME, 1.0, 0.0, [0.5, 0.5]);

      const hStab = buildAirfoilWingPanel({
        side: sx,
        span: 1.72,
        rootChord: 0.86,
        tipChord: 0.58,
        thicknessRatio: 0.10,
        camber: 0.0,
        sweepZ: -0.14
      });
      hStab.translate(0.0, 1.34, -3.92);
      acc.addGeo(hStab, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);
    }

    const tailBeacon = new THREE.SphereGeometry(0.055, 8, 6);
    tailBeacon.translate(0.0, 2.80, -4.16);
    acc.addGeo(tailBeacon, [0.96, 0.14, 0.12], HUMAN_MAT.BOAT_LANTERN, CESSNA_J_AIRFRAME, 1.0, 2.0);

    // 3. Super-Detailed 2-Seat Garmin G1000 Flight Cockpit (Captain eye at [0.27, 1.54, 0.25] kept 100% clear!)
    const glareshield = new THREE.BoxGeometry(1.12, 0.048, 0.28);
    glareshield.translate(0.0, 1.36, 1.06);
    acc.addGeo(glareshield, darkMetal, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.94);

    const wetCompass = new THREE.CylinderGeometry(0.028, 0.028, 0.042, 10, 1);
    wetCompass.translate(0.0, 1.405, 1.02);
    acc.addGeo(wetCompass, darkMetal, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.95);

    const instrumentPanel = new THREE.BoxGeometry(1.10, 0.34, 0.08);
    instrumentPanel.translate(0.0, 1.22, 0.92);
    acc.addGeo(instrumentPanel, [0.16, 0.18, 0.21], HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.94);

    // Garmin G1000 PFD (`metaZ = 2.3` Artificial Horizon!) & MFD (`metaZ = 2.2` Moving Map!) + Audio Panel + Standby Dials
    const g1000Pfd = new THREE.BoxGeometry(0.28, 0.19, 0.022);
    g1000Pfd.translate(0.27, 1.25, 0.875);
    acc.addGeo(g1000Pfd, [0.12, 0.56, 0.86], HUMAN_MAT.VEHICLE_HEADLIGHT, CESSNA_J_AIRFRAME, 1.0, 2.3);

    const audioPanel = new THREE.BoxGeometry(0.042, 0.19, 0.024);
    audioPanel.translate(0.105, 1.25, 0.874);
    acc.addGeo(audioPanel, darkMetal, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.96);

    const g1000Mfd = new THREE.BoxGeometry(0.28, 0.19, 0.022);
    g1000Mfd.translate(-0.06, 1.25, 0.875);
    acc.addGeo(g1000Mfd, [0.12, 0.64, 0.52], HUMAN_MAT.VEHICLE_HEADLIGHT, CESSNA_J_AIRFRAME, 1.0, 2.2);

    for (const stbyX of [0.34, 0.27, 0.20]) {
      const dial = new THREE.CylinderGeometry(0.024, 0.024, 0.022, 10, 1);
      dial.rotateX(Math.PI * 0.5);
      dial.translate(stbyX, 1.105, 0.876);
      acc.addGeo(dial, darkMetal, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.95);
    }

    // Black Throttle & Red Mixture Vernier Plungers + Pitch Trim Wheel + Flap Switch
    const throttleKnob = new THREE.CylinderGeometry(0.018, 0.018, 0.12, 8, 1);
    throttleKnob.rotateX(Math.PI * 0.5);
    throttleKnob.translate(0.03, 1.08, 0.84);
    acc.addGeo(throttleKnob, darkMetal, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.92);

    const mixtureKnob = new THREE.CylinderGeometry(0.017, 0.017, 0.12, 8, 1);
    mixtureKnob.rotateX(Math.PI * 0.5);
    mixtureKnob.translate(-0.035, 1.08, 0.84);
    acc.addGeo(mixtureKnob, [0.84, 0.16, 0.14], HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.96);

    const trimWheel = new THREE.CylinderGeometry(0.055, 0.055, 0.018, 12, 1);
    trimWheel.rotateZ(Math.PI * 0.5);
    trimWheel.translate(0.0, 0.98, 0.85);
    acc.addGeo(trimWheel, darkMetal, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.90);

    // Twin Sculpted Ram's-Horn Control Yokes, Rudder Pedals, & Contoured Leather Pilot/Co-Pilot Seats
    for (const [pIdx, sx] of [[0, 0.27], [1, -0.27]]) {
      for (const pedX of [-0.08, 0.08]) {
        const rPedal = new THREE.BoxGeometry(0.065, 0.11, 0.018);
        rPedal.rotateX(-0.35);
        rPedal.translate(sx + pedX, 0.85, 0.84);
        acc.addGeo(rPedal, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.88);
      }

      const yokeShaft = new THREE.CylinderGeometry(0.015, 0.015, 0.18, 8, 1);
      yokeShaft.rotateX(Math.PI * 0.5);
      yokeShaft.translate(sx, 1.18, 0.80);
      acc.addGeo(yokeShaft, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.95);

      const yokeBar = new THREE.BoxGeometry(0.22, 0.028, 0.026);
      yokeBar.translate(sx, 1.18, 0.71);
      acc.addGeo(yokeBar, darkMetal, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.96);

      for (const hx of [-0.095, 0.095]) {
        const horn = new THREE.BoxGeometry(0.028, 0.095, 0.028);
        horn.translate(sx + hx, 1.21, 0.71);
        acc.addGeo(horn, darkMetal, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.96);
      }

      const seatCush = new THREE.BoxGeometry(0.44, 0.10, 0.44);
      seatCush.translate(sx, 0.84, 0.15);
      acc.addGeo(seatCush, seatTan, HUMAN_MAT.SHOE_BAG, CESSNA_J_AIRFRAME, 0.94);

      const seatBack = new THREE.BoxGeometry(0.42, 0.56, 0.09);
      seatBack.rotateX(-0.12);
      seatBack.translate(sx, 1.16, -0.05);
      acc.addGeo(seatBack, seatTan, HUMAN_MAT.SHOE_BAG, CESSNA_J_AIRFRAME, 0.95);

      // Seated Captain (+0.27, head at z = +0.08m so FPS eye at z = +0.25m is 100% clear!) & Co-Pilot (-0.27)
      const pilotShirt = pIdx === 0 ? aviationWhite : uiGold;
      const pTorso = new THREE.CylinderGeometry(0.14, 0.13, 0.44, 10, 1);
      pTorso.scale(1.12, 1.0, 0.76);
      pTorso.translate(sx, 1.14, 0.09);
      acc.addGeo(pTorso, pilotShirt, HUMAN_MAT.SHOE_BAG, CESSNA_J_AIRFRAME, 0.96);

      const pHead = new THREE.SphereGeometry(0.095, 10, 8);
      pHead.translate(sx, 1.48, 0.08);
      acc.addGeo(pHead, [0.76, 0.56, 0.42], HUMAN_MAT.SHOE_BAG, CESSNA_J_AIRFRAME, 0.98);
    }

    // Slim Cockpit A-Pillars, Side Windows, & Backward-Raked Panoramic Windscreen
    for (const sx of [-0.56, 0.56]) {
      const aPillar = new THREE.CylinderGeometry(0.022, 0.024, 0.72, 8, 1);
      aPillar.rotateX(-0.56);
      aPillar.translate(sx, 1.66, 0.94);
      acc.addGeo(aPillar, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

      const bPillar = new THREE.BoxGeometry(0.05, 0.62, 0.08);
      bPillar.translate(sx, 1.66, -0.18);
      acc.addGeo(bPillar, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

      const sideWin = new THREE.BoxGeometry(0.025, 0.54, 1.68);
      sideWin.translate(sx * 1.01, 1.65, -0.02);
      acc.addGeo(sideWin, glassTint, HUMAN_MAT.VEHICLE_GLASS, CESSNA_J_AIRFRAME, 1.0);
    }
    const windscreen = new THREE.BoxGeometry(1.10, 0.72, 0.03);
    windscreen.rotateX(-0.56);
    windscreen.translate(0.0, 1.66, 0.95);
    acc.addGeo(windscreen, glassTint, HUMAN_MAT.VEHICLE_GLASS, CESSNA_J_AIRFRAME, 1.0);

    // 4. True NACA 2412 Cambered High Wing (11.0m Span), Flap Fairings, Wing Strut Cuffs, Pitot Tube, & Fuel Caps
    const wingCenter = buildAirfoilWingPanel({
      side: 1,
      span: 1.24,
      rootChord: 1.54,
      tipChord: 1.54,
      thicknessRatio: 0.125,
      camber: 0.025
    });
    wingCenter.translate(-0.62, 1.98, 0.08);
    acc.addGeo(wingCenter, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

    for (const sx of [-1.0, 1.0]) {
      const wingPanel = buildAirfoilWingPanel({
        side: sx,
        span: 4.90,
        rootChord: 1.52,
        tipChord: 1.10,
        thicknessRatio: 0.125,
        camber: 0.025,
        dihedralY: 0.14,
        washoutRad: -0.035
      });
      wingPanel.translate(sx * 0.60, 1.98, 0.08);
      acc.addGeo(wingPanel, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

      // Sculpted UI-Gold Wingtip Cap + Flap-Track Canoe Fairings + Fuel Tank Cap
      const tipCap = new THREE.SphereGeometry(0.14, 10, 8);
      tipCap.scale(1.2, 0.45, 3.8);
      tipCap.translate(sx * 5.52, 2.12, 0.08);
      acc.addGeo(tipCap, uiGold, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

      for (const fx of [1.45, 2.55]) {
        const flapCanoe = new THREE.ConeGeometry(0.038, 0.42, 8, 1);
        flapCanoe.rotateX(-Math.PI * 0.5);
        flapCanoe.translate(sx * fx, 1.94, -0.52);
        acc.addGeo(flapCanoe, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.94);
      }

      const fuelCap = new THREE.CylinderGeometry(0.045, 0.045, 0.015, 10, 1);
      fuelCap.translate(sx * 1.85, 2.13, 0.28);
      acc.addGeo(fuelCap, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 1.0);

      // Diagonal Wing Lift Strut + Aerodynamic Wing/Fuselage Strut Cuffs
      const strut = new THREE.CylinderGeometry(0.030, 0.030, 2.52, 8, 1);
      strut.rotateZ(-sx * 1.12);
      strut.translate(sx * 1.68, 1.42, 0.24);
      acc.addGeo(strut, aviationWhite, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.96);

      const navLight = new THREE.SphereGeometry(0.055, 8, 6);
      navLight.translate(sx * 5.58, 2.12, 0.42);
      acc.addGeo(
        navLight,
        sx > 0 ? [0.95, 0.16, 0.14] : [0.16, 0.92, 0.36],
        HUMAN_MAT.BOAT_LANTERN,
        CESSNA_J_AIRFRAME,
        1.0,
        sx > 0 ? 2.0 : 1.0
      );
    }

    // L-Shaped Underwing Pitot Tube on Captain (+X) Wing + Dual LED Wing Landing Light
    const pitotL = new THREE.CylinderGeometry(0.008, 0.010, 0.22, 6, 1);
    pitotL.rotateX(Math.PI * 0.5);
    pitotL.translate(2.05, 1.90, 0.56);
    acc.addGeo(pitotL, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.98);

    const wingLandingLight = new THREE.BoxGeometry(0.28, 0.07, 0.04);
    wingLandingLight.translate(2.35, 2.03, 0.83);
    acc.addGeo(wingLandingLight, [0.95, 0.98, 1.00], HUMAN_MAT.VEHICLE_HEADLIGHT, CESSNA_J_AIRFRAME, 1.0, 0.0);

    // 5. ARTICULATED JOINT #1: Chrome Spinner & Twisted 2-Blade McCauley Propeller (`CESSNA_J_PROP` at [0, 1.18, 2.58])
    const [px, py, pz] = CESSNA_BIND_PIVOTS[CESSNA_J_PROP];
    const propSpinner = new THREE.ConeGeometry(0.16, 0.30, 16, 1);
    propSpinner.rotateX(Math.PI * 0.5);
    propSpinner.translate(px, py, pz + 0.06);
    acc.addGeo(propSpinner, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_PROP, 1.0);

    for (const sBlade of [-1.0, 1.0]) {
      const bladeIn = new THREE.BoxGeometry(0.125, 0.52, 0.024);
      bladeIn.rotateY(0.30 * sBlade);
      bladeIn.translate(px, py + sBlade * 0.30, pz);
      acc.addGeo(bladeIn, darkMetal, HUMAN_MAT.METAL_ACCENT, CESSNA_J_PROP, 0.98);

      const bladeOut = new THREE.BoxGeometry(0.095, 0.42, 0.018);
      bladeOut.rotateY(0.16 * sBlade);
      bladeOut.translate(px, py + sBlade * 0.74, pz);
      acc.addGeo(bladeOut, darkMetal, HUMAN_MAT.METAL_ACCENT, CESSNA_J_PROP, 0.98);

      const bladeTip = new THREE.BoxGeometry(0.092, 0.11, 0.020);
      bladeTip.rotateY(0.14 * sBlade);
      bladeTip.translate(px, py + sBlade * 0.91, pz);
      acc.addGeo(bladeTip, uiGold, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_PROP, 1.0);
    }

    return acc.toBundle();
  }

  /**
   * Builds a Super-Detailed Swept-Wing T-Tail Citation Learjet Executive Shuttle Jet ("PK-UIB"):
   * - 6 Articulated Joints:
   *   `LEARJET_J_AIRFRAME (0)`, `LEARJET_J_FAN_L (1)`, `LEARJET_J_FAN_R (2)`,
   *   + Real Retractable Tricycle Landing Gear (`LEARJET_J_GEAR_NOSE (3)`, `LEARJET_J_GEAR_MAIN_L (4)`, `LEARJET_J_GEAR_MAIN_R (5)`)!
   * - Swept Supercritical Airfoil Wings (13.8m span) + Wing-Body Fairing + Flap Canoes + Canted UI-Gold Winglets
   * - Airfoil Swept T-Tail, Port Entry Door, 10 Framed Oval Windows, Aft Turbofans with 16 Twisted Titanium Fan Blades
   * - Collins Pro Line 21 Flight Deck: Captain & FO PFDs (`metaZ = 2.3`), Center EICAS/MFD (`metaZ = 2.2`), MCP Glareshield,
   *   Twin Thrust Levers, 3-Green Gear Indicator, Overhead Panel, & Unobstructed Captain Eye at `[0.32, 1.77, 3.72]`!
   */
  static buildCitationLearjetAircraft() {
    const acc = new RigidJointBundleAccumulator();

    const pearlWhite = [0.96, 0.97, 0.99];
    const titaniumBelly = [0.72, 0.75, 0.80];
    const uiGold = [0.93, 0.70, 0.10];
    const royalNavy = [0.08, 0.19, 0.40];
    const darkCarbon = [0.12, 0.13, 0.16];
    const rubberTire = [0.085, 0.088, 0.095];
    const chrome = [0.85, 0.88, 0.92];
    const glassTint = [0.18, 0.28, 0.38];
    const leatherCream = [0.85, 0.81, 0.73];
    const makaraGold = [0.92, 0.72, 0.16];

    // =========================================================================
    // 1. RETRACTABLE TRICYCLE LANDING GEAR RIGGED ON JOINTS 3, 4, & 5!
    //    (Bottom strictly at y = 0.0m when deployed; folds cleanly into bays in cruise!)
    // =========================================================================
    // Recessed dark wheel-well bays on airframe (`LEARJET_J_AIRFRAME`)
    const noseWell = new THREE.BoxGeometry(0.42, 0.12, 0.96);
    noseWell.translate(0.0, 0.84, 4.78);
    acc.addGeo(noseWell, darkCarbon, HUMAN_MAT.RUBBER_TIRE, LEARJET_J_AIRFRAME, 0.65);

    for (const sx of [-1.0, 1.0]) {
      const mainWell = new THREE.BoxGeometry(1.12, 0.12, 0.72);
      mainWell.translate(sx * 1.32, 0.82, -0.78);
      acc.addGeo(mainWell, darkCarbon, HUMAN_MAT.RUBBER_TIRE, LEARJET_J_AIRFRAME, 0.65);
    }

    // JOINT #3 (`LEARJET_J_GEAR_NOSE`, pivot `[0.0, 0.84, 4.42]`): Twin-Wheel Nose Gear + Torque Links + Taxi Light + Door
    const noseOleo = new THREE.CylinderGeometry(0.044, 0.038, 0.66, 10, 1);
    noseOleo.translate(0.0, 0.51, 4.42);
    acc.addGeo(noseOleo, chrome, HUMAN_MAT.METAL_ACCENT, LEARJET_J_GEAR_NOSE, 0.95);

    for (const sc of [-1, 1]) {
      const nLink = new THREE.BoxGeometry(0.024, 0.12, 0.020);
      nLink.rotateX(sc * 0.48);
      nLink.translate(0.0, 0.44 + sc * 0.04, 4.37);
      acc.addGeo(nLink, chrome, HUMAN_MAT.METAL_ACCENT, LEARJET_J_GEAR_NOSE, 0.92);
    }

    for (const wx of [-0.11, 0.11]) {
      const nTire = new THREE.CylinderGeometry(0.18, 0.18, 0.085, 16, 1);
      nTire.rotateZ(Math.PI * 0.5);
      nTire.translate(wx, 0.18, 4.42);
      acc.addGeo(nTire, rubberTire, HUMAN_MAT.RUBBER_TIRE, LEARJET_J_GEAR_NOSE, 0.90);

      const nRim = new THREE.CylinderGeometry(0.095, 0.095, 0.092, 12, 1);
      nRim.rotateZ(Math.PI * 0.5);
      nRim.translate(wx, 0.18, 4.42);
      acc.addGeo(nRim, pearlWhite, HUMAN_MAT.METAL_ACCENT, LEARJET_J_GEAR_NOSE, 0.96);
    }

    const taxiLight = new THREE.BoxGeometry(0.18, 0.075, 0.045);
    taxiLight.translate(0.0, 0.56, 4.47);
    acc.addGeo(taxiLight, [0.95, 0.98, 1.00], HUMAN_MAT.VEHICLE_HEADLIGHT, LEARJET_J_GEAR_NOSE, 1.0, 0.0);

    const noseDoor = new THREE.BoxGeometry(0.025, 0.48, 0.34);
    noseDoor.translate(0.20, 0.62, 4.52);
    acc.addGeo(noseDoor, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_GEAR_NOSE, 0.96);

    // JOINTS #4 & #5 (`LEARJET_J_GEAR_MAIN_L` at `+1.82` & `LEARJET_J_GEAR_MAIN_R` at `-1.82`): Trailing-Link Main Gear!
    for (const [mJoint, sx] of [[LEARJET_J_GEAR_MAIN_L, 1.82], [LEARJET_J_GEAR_MAIN_R, -1.82]]) {
      const sSign = Math.sign(sx);
      const mainStrut = new THREE.CylinderGeometry(0.052, 0.046, 0.64, 10, 1);
      mainStrut.translate(sx, 0.56, -0.74);
      acc.addGeo(mainStrut, chrome, HUMAN_MAT.METAL_ACCENT, mJoint, 0.94);

      const trailingArm = new THREE.BoxGeometry(0.055, 0.065, 0.24);
      trailingArm.rotateX(0.26);
      trailingArm.translate(sx, 0.28, -0.78);
      acc.addGeo(trailingArm, chrome, HUMAN_MAT.METAL_ACCENT, mJoint, 0.92);

      for (const wxOff of [-0.13, 0.13]) {
        const mTire = new THREE.CylinderGeometry(0.24, 0.24, 0.11, 16, 1);
        mTire.rotateZ(Math.PI * 0.5);
        mTire.translate(sx + wxOff, 0.24, -0.82);
        acc.addGeo(mTire, rubberTire, HUMAN_MAT.RUBBER_TIRE, mJoint, 0.90);

        const mRim = new THREE.CylinderGeometry(0.125, 0.125, 0.118, 12, 1);
        mRim.rotateZ(Math.PI * 0.5);
        mRim.translate(sx + wxOff, 0.24, -0.82);
        acc.addGeo(mRim, titaniumBelly, HUMAN_MAT.METAL_ACCENT, mJoint, 0.95);
      }

      const gearDoor = new THREE.BoxGeometry(0.028, 0.54, 0.44);
      gearDoor.translate(sx + sSign * 0.20, 0.62, -0.78);
      acc.addGeo(gearDoor, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, mJoint, 0.97);
    }

    // =========================================================================
    // 2. SLEEK PRESSURIZED EXECUTIVE FUSELAGE, RADOME, PROBES, & SWEPT T-TAIL
    // =========================================================================
    const fuselageMain = new THREE.CylinderGeometry(0.88, 0.88, 7.40, 24, 1);
    fuselageMain.rotateX(Math.PI * 0.5);
    fuselageMain.translate(0.0, 1.42, -0.15);
    acc.addGeo(fuselageMain, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    // Sculpted Wing-Body Aerodynamic Belly Canoe Fairing
    const bellyStations = [
      { z: -3.20, keelY: 0.62, chineX: 0.72, chineY: 0.74, deckX: 0.88, deckY: 0.98 },
      { z: -0.60, keelY: 0.56, chineX: 0.98, chineY: 0.70, deckX: 1.06, deckY: 0.98 },
      { z: 1.85, keelY: 0.60, chineX: 0.82, chineY: 0.74, deckX: 0.90, deckY: 0.98 }
    ];
    acc.addGeo(buildLoftedHullStations(bellyStations, 0.0, true), titaniumBelly, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.95);

    // Seamless 6-Station Lofted Forward Cockpit Fuselage & Aerodynamic Radome + Pitot/AOA Probes
    const noseStations = [
      { z: 3.55, keelY: 0.56, chineX: 0.78, chineY: 0.92, deckX: 0.87, deckY: 1.54, deckCrownY: 1.56 },
      { z: 4.18, keelY: 0.62, chineX: 0.72, chineY: 0.96, deckX: 0.78, deckY: 1.53, deckCrownY: 1.56 },
      { z: 4.78, keelY: 0.70, chineX: 0.60, chineY: 0.98, deckX: 0.64, deckY: 1.48, deckCrownY: 1.53 },
      { z: 5.55, keelY: 0.82, chineX: 0.42, chineY: 1.02, deckX: 0.44, deckY: 1.34, deckCrownY: 1.38 },
      { z: 6.28, keelY: 0.95, chineX: 0.22, chineY: 1.05, deckX: 0.22, deckY: 1.20, deckCrownY: 1.22 },
      { z: 6.65, keelY: 1.02, chineX: 0.11, chineY: 1.07, deckX: 0.11, deckY: 1.13, deckCrownY: 1.14 }
    ];
    acc.addGeo(buildLoftedHullStations(noseStations, 0.0, true), pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    const radomeTip = new THREE.ConeGeometry(0.15, 0.48, 16, 1);
    radomeTip.rotateX(Math.PI * 0.5 + 0.05);
    radomeTip.translate(0.0, 1.08, 6.84);
    acc.addGeo(radomeTip, darkCarbon, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.98);

    for (const sx of [-1.0, 1.0]) {
      const pitotProbe = new THREE.CylinderGeometry(0.008, 0.012, 0.24, 6, 1);
      pitotProbe.rotateX(Math.PI * 0.5);
      pitotProbe.translate(sx * 0.46, 1.12, 5.48);
      acc.addGeo(pitotProbe, chrome, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.98);
    }

    // Port (+X) Executive Clamshell Entry Door Frame & Handle
    const entryDoor = new THREE.BoxGeometry(0.028, 1.12, 0.68);
    entryDoor.translate(0.878, 1.48, 2.86);
    acc.addGeo(entryDoor, titaniumBelly, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.96);

    // Executive UI Gold & Royal Navy Side Fuselage Cheatlines + 10 Framed Oval Cabin Windows
    for (const sx of [-1.0, 1.0]) {
      const jetGoldSide = new THREE.BoxGeometry(0.045, 0.09, 7.65);
      jetGoldSide.translate(sx * 0.875, 1.38, -0.25);
      acc.addGeo(jetGoldSide, uiGold, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

      const jetNavySide = new THREE.BoxGeometry(0.042, 0.06, 7.55);
      jetNavySide.translate(sx * 0.873, 1.29, -0.28);
      acc.addGeo(jetNavySide, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.98);

      for (let w = 0; w < 5; w++) {
        const wz = 2.05 - w * 0.92;
        const winRim = new THREE.CylinderGeometry(0.17, 0.17, 0.036, 14, 1);
        winRim.rotateZ(Math.PI * 0.5);
        winRim.scale(1.0, 1.22, 0.85);
        winRim.translate(sx * 0.872, 1.64, wz);
        acc.addGeo(winRim, chrome, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.98);

        const ovalWin = new THREE.CylinderGeometry(0.145, 0.145, 0.042, 14, 1);
        ovalWin.rotateZ(Math.PI * 0.5);
        ovalWin.scale(1.0, 1.22, 0.85);
        ovalWin.translate(sx * 0.875, 1.64, wz);
        acc.addGeo(ovalWin, glassTint, HUMAN_MAT.BOAT_LANTERN, LEARJET_J_AIRFRAME, 1.0, 3.0);
      }
    }

    // Tapered Rear Tailcone, Dorsal Ram-Air Scoop, & Airfoil Swept T-Tail Empennage
    const rearCone = new THREE.CylinderGeometry(0.86, 0.24, 3.15, 18, 1);
    rearCone.rotateX(-Math.PI * 0.5 + 0.08);
    rearCone.translate(0.0, 1.54, -5.35);
    acc.addGeo(rearCone, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.98);

    const dorsalScoop = new THREE.BoxGeometry(0.18, 0.16, 0.65);
    dorsalScoop.translate(0.0, 2.18, -4.45);
    acc.addGeo(dorsalScoop, titaniumBelly, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.94);

    const tTailFin = buildAirfoilWingPanel({
      side: 1,
      span: 2.48,
      rootChord: 1.75,
      tipChord: 1.05,
      thicknessRatio: 0.095,
      camber: 0.0,
      sweepZ: -1.08
    });
    tTailFin.rotateZ(Math.PI * 0.5);
    tTailFin.translate(0.0, 1.68, -5.65);
    acc.addGeo(tTailFin, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    const tTailGold = new THREE.BoxGeometry(0.13, 0.32, 1.12);
    tTailGold.translate(0.0, 3.74, -6.58);
    acc.addGeo(tTailGold, uiGold, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    for (const sx of [-1.0, 1.0]) {
      const tailCrest = new THREE.CylinderGeometry(0.20, 0.20, 0.025, 14, 1);
      tailCrest.rotateZ(Math.PI * 0.5);
      tailCrest.translate(sx * 0.075, 2.98, -6.18);
      acc.addGeo(tailCrest, makaraGold, HUMAN_MAT.CREST_GOLD, LEARJET_J_AIRFRAME, 1.0, 0.0, [0.5, 0.5]);

      const hTail = buildAirfoilWingPanel({
        side: sx,
        span: 2.38,
        rootChord: 1.02,
        tipChord: 0.56,
        thicknessRatio: 0.09,
        camber: 0.0,
        sweepZ: -0.58,
        dihedralY: -0.06
      });
      hTail.translate(0.0, 4.12, -6.58);
      acc.addGeo(hTail, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);
    }

    const tTailBullet = new THREE.CylinderGeometry(0.11, 0.11, 1.72, 12, 1);
    tTailBullet.rotateX(Math.PI * 0.5);
    tTailBullet.translate(0.0, 4.12, -6.62);
    acc.addGeo(tTailBullet, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    const topStrobe = new THREE.SphereGeometry(0.065, 8, 6);
    topStrobe.translate(0.0, 4.26, -6.62);
    acc.addGeo(topStrobe, [0.96, 0.14, 0.12], HUMAN_MAT.BOAT_LANTERN, LEARJET_J_AIRFRAME, 1.0, 2.0);

    // =========================================================================
    // 3. COLLINS PRO LINE 21 EXECUTIVE JET FLIGHT DECK (FPS_PILOT eye `[0.32, 1.77, 3.72]`)
    // =========================================================================
    const cockpitFloor = new THREE.BoxGeometry(1.08, 0.06, 1.18);
    cockpitFloor.translate(0.0, 1.02, 3.86);
    acc.addGeo(cockpitFloor, darkCarbon, HUMAN_MAT.RUBBER_TIRE, LEARJET_J_AIRFRAME, 0.90);

    const jetGlareshield = new THREE.BoxGeometry(1.08, 0.048, 0.24);
    jetGlareshield.translate(0.0, 1.495, 4.31);
    acc.addGeo(jetGlareshield, darkCarbon, HUMAN_MAT.RUBBER_TIRE, LEARJET_J_AIRFRAME, 0.95);

    const mcpPanel = new THREE.BoxGeometry(0.54, 0.046, 0.04);
    mcpPanel.translate(0.0, 1.485, 4.21);
    acc.addGeo(mcpPanel, [0.20, 0.24, 0.28], HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.96);

    const jetPanel = new THREE.BoxGeometry(1.08, 0.32, 0.07);
    jetPanel.translate(0.0, 1.38, 4.26);
    acc.addGeo(jetPanel, [0.16, 0.18, 0.22], HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.95);

    // 3-Screen Pro Line 21 Glass Avionics: Captain PFD (`metaZ = 2.3`), Center EICAS/MFD (`metaZ = 2.2`), FO PFD (`metaZ = 2.3`)
    for (const [scrX, scrCol, mZ] of [
      [0.32, [0.10, 0.56, 0.88], 2.3],
      [0.0, [0.12, 0.68, 0.54], 2.2],
      [-0.32, [0.10, 0.56, 0.88], 2.3]
    ]) {
      const mfd = new THREE.BoxGeometry(0.27, 0.19, 0.022);
      mfd.translate(scrX, 1.41, 4.215);
      acc.addGeo(mfd, scrCol, HUMAN_MAT.VEHICLE_HEADLIGHT, LEARJET_J_AIRFRAME, 1.0, mZ);
    }

    // Center Pedestal: Dual FMS Keypads + Twin T-Handle Thrust Levers + Flap Lever + 3-Green Gear Indicator
    const thrustPedestal = new THREE.BoxGeometry(0.24, 0.32, 0.62);
    thrustPedestal.translate(0.0, 1.18, 4.08);
    acc.addGeo(thrustPedestal, darkCarbon, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.92);

    for (const tlX of [-0.042, 0.042]) {
      const thrustStalk = new THREE.BoxGeometry(0.018, 0.13, 0.024);
      thrustStalk.rotateX(0.28);
      thrustStalk.translate(tlX, 1.38, 4.08);
      acc.addGeo(thrustStalk, chrome, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.98);

      const thrustGrip = new THREE.CylinderGeometry(0.016, 0.016, 0.055, 8, 1);
      thrustGrip.rotateZ(Math.PI * 0.5);
      thrustGrip.translate(tlX, 1.44, 4.10);
      acc.addGeo(thrustGrip, darkCarbon, HUMAN_MAT.RUBBER_TIRE, LEARJET_J_AIRFRAME, 0.98);
    }

    for (const [pIdx, sx] of [[0, 0.32], [1, -0.32]]) {
      const yCol = new THREE.CylinderGeometry(0.020, 0.024, 0.42, 8, 1);
      yCol.rotateX(-0.14);
      yCol.translate(sx, 1.24, 4.18);
      acc.addGeo(yCol, darkCarbon, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.94);

      const yBar = new THREE.BoxGeometry(0.24, 0.032, 0.030);
      yBar.translate(sx, 1.44, 4.14);
      acc.addGeo(yBar, darkCarbon, HUMAN_MAT.RUBBER_TIRE, LEARJET_J_AIRFRAME, 0.96);

      for (const hx of [-0.105, 0.105]) {
        const yHorn = new THREE.BoxGeometry(0.030, 0.105, 0.030);
        yHorn.translate(sx + hx, 1.48, 4.14);
        acc.addGeo(yHorn, darkCarbon, HUMAN_MAT.RUBBER_TIRE, LEARJET_J_AIRFRAME, 0.96);
      }

      const jSeat = new THREE.BoxGeometry(0.46, 0.11, 0.46);
      jSeat.translate(sx, 1.14, 3.61);
      acc.addGeo(jSeat, leatherCream, HUMAN_MAT.SHOE_BAG, LEARJET_J_AIRFRAME, 0.95);

      const jBack = new THREE.BoxGeometry(0.44, 0.62, 0.10);
      jBack.rotateX(-0.12);
      jBack.translate(sx, 1.48, 3.39);
      acc.addGeo(jBack, leatherCream, HUMAN_MAT.SHOE_BAG, LEARJET_J_AIRFRAME, 0.96);

      // Seated Jet Captain (+0.32, head at z = +3.55m so FPS eye at z = +3.72m is 100% clear!) & First Officer (-0.32)
      const pTorso = new THREE.CylinderGeometry(0.145, 0.135, 0.46, 10, 1);
      pTorso.scale(1.12, 1.0, 0.76);
      pTorso.translate(sx, 1.44, 3.57);
      acc.addGeo(pTorso, pIdx === 0 ? pearlWhite : royalNavy, HUMAN_MAT.SHOE_BAG, LEARJET_J_AIRFRAME, 0.96);

      const pHead = new THREE.SphereGeometry(0.098, 10, 8);
      pHead.translate(sx, 1.77, 3.55);
      acc.addGeo(pHead, [0.76, 0.56, 0.42], HUMAN_MAT.SHOE_BAG, LEARJET_J_AIRFRAME, 0.98);
    }

    // V-Raked Dual-Pane Flush Cockpit Windscreen, Flush Side Glazing, Struts, & Lofted Roof Crown Fairing
    const centerPost = new THREE.CylinderGeometry(0.017, 0.020, 0.93, 8, 1);
    centerPost.rotateX(-0.888);
    centerPost.translate(0.0, 1.838, 4.28);
    acc.addGeo(centerPost, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.98);

    const buildFlushTrapezoidGlass = (p0, p1, p2, p3, halfT, outHint) => {
      const gPos = [];
      const gNorm = [];
      const gUv = [];
      const e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
      const e2 = [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]];
      let fnx = e1[1] * e2[2] - e1[2] * e2[1];
      let fny = e1[2] * e2[0] - e1[0] * e2[2];
      let fnz = e1[0] * e2[1] - e1[1] * e2[0];
      const fLen = Math.hypot(fnx, fny, fnz) || 1.0;
      fnx /= fLen; fny /= fLen; fnz /= fLen;
      if (fnx * outHint[0] + fny * outHint[1] + fnz * outHint[2] < 0.0) {
        fnx = -fnx; fny = -fny; fnz = -fnz;
      }
      const off = (pt, s) => [pt[0] + fnx * s, pt[1] + fny * s, pt[2] + fnz * s];
      const o = [off(p0, halfT), off(p1, halfT), off(p2, halfT), off(p3, halfT)];
      const n = [off(p0, -halfT), off(p1, -halfT), off(p2, -halfT), off(p3, -halfT)];
      const addTri = (a, b, c, ua, ub, uc, hx, hy, hz) => {
        const u1x = b[0] - a[0], u1y = b[1] - a[1], u1z = b[2] - a[2];
        const u2x = c[0] - a[0], u2y = c[1] - a[1], u2z = c[2] - a[2];
        let nx = u1y * u2z - u1z * u2y;
        let ny = u1z * u2x - u1x * u2z;
        let nz = u1x * u2y - u1y * u2x;
        const l = Math.hypot(nx, ny, nz);
        if (l < 1e-7) return;
        nx /= l; ny /= l; nz /= l;
        let va = a, vb = b, vc = c, tua = ua, tub = ub, tuc = uc;
        if (nx * hx + ny * hy + nz * hz < 0.0) {
          nx = -nx; ny = -ny; nz = -nz;
          vb = c; vc = b; tub = uc; tuc = ub;
        }
        gPos.push(va[0], va[1], va[2], vb[0], vb[1], vb[2], vc[0], vc[1], vc[2]);
        gNorm.push(nx, ny, nz, nx, ny, nz, nx, ny, nz);
        gUv.push(tua[0], tua[1], tub[0], tub[1], tuc[0], tuc[1]);
      };
      addTri(o[0], o[1], o[2], [0, 0], [1, 0], [1, 1], outHint[0], outHint[1], outHint[2]);
      addTri(o[0], o[2], o[3], [0, 0], [1, 1], [0, 1], outHint[0], outHint[1], outHint[2]);
      addTri(n[0], n[1], n[2], [0, 0], [1, 0], [1, 1], -outHint[0], -outHint[1], -outHint[2]);
      addTri(n[0], n[2], n[3], [0, 0], [1, 1], [0, 1], -outHint[0], -outHint[1], -outHint[2]);
      const hints = [[0, -1, 0], [outHint[0], 0.4, 0.8], [0, 1, 0], [-outHint[0], 0.0, -1.0]];
      for (let k = 0; k < 4; k++) {
        const kn = (k + 1) % 4;
        const h = hints[k];
        addTri(o[k], o[kn], n[kn], [0, 0], [1, 0], [1, 1], h[0], h[1], h[2]);
        addTri(o[k], n[kn], n[k], [0, 0], [1, 1], [0, 1], h[0], h[1], h[2]);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(gPos, 3));
      g.setAttribute('normal', new THREE.Float32BufferAttribute(gNorm, 3));
      g.setAttribute('uv', new THREE.Float32BufferAttribute(gUv, 2));
      computeCreaseSmoothNormals(g, 0.82);
      return g;
    };

    for (const sx of [-1.0, 1.0]) {
      const wsPane = buildFlushTrapezoidGlass(
        [0.0, 1.545, 4.64],
        [sx * 0.66, 1.505, 4.46],
        [sx * 0.554, 2.096, 3.769],
        [0.0, 2.13, 3.92],
        0.012,
        [sx * 0.35, 0.65, 0.65]
      );
      acc.addGeo(wsPane, glassTint, HUMAN_MAT.VEHICLE_GLASS, LEARJET_J_AIRFRAME, 1.0);

      const ckSideWin = buildFlushTrapezoidGlass(
        [sx * 0.865, 1.535, 3.55],
        [sx * 0.66, 1.505, 4.46],
        [sx * 0.554, 2.096, 3.769],
        [sx * 0.562, 2.096, 3.55],
        0.012,
        [sx, 0.45, 0.15]
      );
      acc.addGeo(ckSideWin, glassTint, HUMAN_MAT.VEHICLE_GLASS, LEARJET_J_AIRFRAME, 1.0);

      const ckAPillar = new THREE.CylinderGeometry(0.017, 0.021, 0.916, 8, 1);
      ckAPillar.rotateX(-0.8555);
      ckAPillar.rotateZ(sx * 0.1775);
      ckAPillar.translate(sx * 0.607, 1.8005, 4.1145);
      acc.addGeo(ckAPillar, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.99);
    }

    const cockpitRoofStations = [
      { z: 3.53, keelY: 2.085, chineX: 0.565, chineY: 2.096, deckX: 0.34, deckY: 2.23, deckCrownY: 2.30 },
      { z: 3.77, keelY: 2.090, chineX: 0.554, chineY: 2.096, deckX: 0.28, deckY: 2.18, deckCrownY: 2.22 },
      { z: 3.93, keelY: 2.115, chineX: 0.08, chineY: 2.125, deckX: 0.04, deckY: 2.13, deckCrownY: 2.135 }
    ];
    acc.addGeo(buildLoftedHullStations(cockpitRoofStations, 0.0, true), pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    // =========================================================================
    // 4. LOW SWEPT SUPERCRITICAL AIRFOIL WING (13.8m SPAN) & CANTED WINGLETS
    // =========================================================================
    for (const sx of [-1.0, 1.0]) {
      const wing = buildAirfoilWingPanel({
        side: sx,
        span: 6.02,
        rootChord: 2.35,
        tipChord: 0.98,
        thicknessRatio: 0.105,
        camber: 0.016,
        sweepZ: -1.48,
        dihedralY: 0.28,
        washoutRad: -0.032
      });
      wing.translate(sx * 0.80, 0.90, -0.18);
      acc.addGeo(wing, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

      // Underwing Flap-Track Canoe Fairings
      for (const [fx, fz] of [[2.25, -1.35], [3.85, -1.68]]) {
        const canoe = new THREE.ConeGeometry(0.048, 0.58, 8, 1);
        canoe.rotateX(-Math.PI * 0.5);
        canoe.translate(sx * fx, 0.95, fz);
        acc.addGeo(canoe, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.95);
      }

      // Iconic Blended Canted UI-Gold Learjet Winglet + Navigation Strobe
      const winglet = buildAirfoilWingPanel({
        side: 1,
        span: 1.14,
        rootChord: 0.86,
        tipChord: 0.42,
        thicknessRatio: 0.085,
        camber: 0.0,
        sweepZ: -0.44
      });
      winglet.rotateZ(Math.PI * 0.5 - sx * 0.18);
      winglet.translate(sx * 6.80, 1.16, -1.66);
      acc.addGeo(winglet, uiGold, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

      const navStrobe = new THREE.SphereGeometry(0.06, 8, 6);
      navStrobe.translate(sx * 6.98, 2.26, -1.96);
      acc.addGeo(
        navStrobe,
        sx > 0 ? [0.95, 0.16, 0.14] : [0.16, 0.92, 0.36],
        HUMAN_MAT.BOAT_LANTERN,
        LEARJET_J_AIRFRAME,
        1.0,
        sx > 0 ? 2.0 : 1.0
      );
    }

    // =========================================================================
    // 5. TWIN AFT HIGH-BYPASS TURBOFAN NACELLES & 16-BLADE TITANIUM FANS (JOINTS 1 & 2)
    // =========================================================================
    const turbofanSpecs = [
      { sx: 1.0, joint: LEARJET_J_FAN_L, pivot: LEARJET_BIND_PIVOTS[LEARJET_J_FAN_L] },
      { sx: -1.0, joint: LEARJET_J_FAN_R, pivot: LEARJET_BIND_PIVOTS[LEARJET_J_FAN_R] }
    ];
    for (const tf of turbofanSpecs) {
      const [fx, fy, fz] = tf.pivot;
      const pylon = buildAirfoilWingPanel({
        side: tf.sx,
        span: 0.58,
        rootChord: 1.48,
        tipChord: 1.22,
        thicknessRatio: 0.10,
        camber: 0.0,
        dihedralY: 0.12
      });
      pylon.translate(tf.sx * 0.72, 1.76, -3.45);
      acc.addGeo(pylon, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.96);

      // Open-ended outer cowl so the recessed 16-blade titanium fan stage is 100% unobstructed from the front!
      const nacelle = new THREE.CylinderGeometry(0.46, 0.42, 2.12, 20, 1, true);
      nacelle.rotateX(Math.PI * 0.5);
      nacelle.translate(fx, fy, -3.50);
      acc.addGeo(nacelle, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

      const nacelleBand = new THREE.CylinderGeometry(0.463, 0.460, 0.16, 20, 1, true);
      nacelleBand.rotateX(Math.PI * 0.5);
      nacelleBand.translate(fx, fy, -2.95);
      acc.addGeo(nacelleBand, uiGold, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

      const intakeLip = new THREE.TorusGeometry(0.435, 0.048, 10, 20);
      intakeLip.translate(fx, fy, fz + 0.05);
      acc.addGeo(intakeLip, chrome, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 1.0);

      const exhaustCone = new THREE.CylinderGeometry(0.36, 0.25, 0.48, 16, 1);
      exhaustCone.rotateX(-Math.PI * 0.5);
      exhaustCone.translate(fx, fy, -4.70);
      acc.addGeo(exhaustCone, titaniumBelly, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.92);

      const corePlug = new THREE.ConeGeometry(0.14, 0.36, 12, 1);
      corePlug.rotateX(-Math.PI * 0.5);
      corePlug.translate(fx, fy, -4.96);
      acc.addGeo(corePlug, darkCarbon, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.88);

      // Articulated Spinning Turbofan Hub Cone + 16 Twisted Wide-Chord Titanium Fan Blades (`joint = 1 or 2`)
      const fanDisc = new THREE.CylinderGeometry(0.39, 0.39, 0.04, 18, 1);
      fanDisc.rotateX(Math.PI * 0.5);
      fanDisc.translate(fx, fy, fz - 0.04);
      acc.addGeo(fanDisc, darkCarbon, HUMAN_MAT.METAL_ACCENT, tf.joint, 0.88);

      const fanCone = new THREE.ConeGeometry(0.115, 0.24, 14, 1);
      fanCone.rotateX(Math.PI * 0.5);
      fanCone.translate(fx, fy, fz + 0.055);
      acc.addGeo(fanCone, chrome, HUMAN_MAT.METAL_ACCENT, tf.joint, 1.0);

      for (let b = 0; b < 16; b++) {
        const bAng = (b / 16) * Math.PI * 2.0;
        const fBlade = new THREE.BoxGeometry(0.068, 0.36, 0.014);
        fBlade.rotateY(0.42);
        fBlade.translate(0.0, 0.19, 0.0);
        fBlade.rotateZ(bAng);
        fBlade.translate(fx, fy, fz - 0.01);
        acc.addGeo(fBlade, titaniumBelly, HUMAN_MAT.METAL_ACCENT, tf.joint, 0.96);
      }
    }

    return acc.toBundle();
  }
}
