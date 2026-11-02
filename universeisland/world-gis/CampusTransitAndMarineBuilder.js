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

// 6-Joint Articulated UI Inter-Island Electric Bus ("Bus Kuning Listrik Lintas Pulau UI" — 7.8m x 2.38m x 2.96m)
export const BUS_JOINTS_PER_VEHICLE = 6;
export const BUS_J_CHASSIS = 0;
export const BUS_J_WHEEL_FL = 1;
export const BUS_J_WHEEL_FR = 2;
export const BUS_J_WHEEL_RL = 3;
export const BUS_J_WHEEL_RR = 4;
export const BUS_J_STEER_WHEEL = 5;

export const BUS_WHEEL_RADIUS = 0.42;
export const BUS_WHEELBASE = 4.20;
export const BUS_TRACK_HALF_W = 1.04;

export const BUS_BIND_PIVOTS = [
  [0.0, 0.0, 0.0],                               // 0: BUS_J_CHASSIS
  [BUS_TRACK_HALF_W, BUS_WHEEL_RADIUS, 2.10],    // 1: BUS_J_WHEEL_FL (+X Left, +Z Front)
  [-BUS_TRACK_HALF_W, BUS_WHEEL_RADIUS, 2.10],   // 2: BUS_J_WHEEL_FR (-X Right, +Z Front)
  [BUS_TRACK_HALF_W, BUS_WHEEL_RADIUS, -2.10],   // 3: BUS_J_WHEEL_RL (+X Left, -Z Rear)
  [-BUS_TRACK_HALF_W, BUS_WHEEL_RADIUS, -2.10],  // 4: BUS_J_WHEEL_RR (-X Right, -Z Rear)
  [-0.68, 1.48, 3.36]                            // 5: BUS_J_STEER_WHEEL (RHD Driver Steering Hub)
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
  [0.0, 1.18, 2.58]      // 1: Spinning Propeller & Spinner Hub
];

// 3-Joint Citation Learjet Executive Campus Shuttle Jet ("PK-UIB")
export const LEARJET_JOINTS_PER_AIRCRAFT = 3;
export const LEARJET_J_AIRFRAME = 0;
export const LEARJET_J_FAN_L = 1;
export const LEARJET_J_FAN_R = 2;
export const LEARJET_BIND_PIVOTS = [
  [0.0, 0.0, 0.0],        // 0: Airframe (+Z Radome Nose, +Y Up, +X Left Wing)
  [1.46, 1.94, -2.48],    // 1: Port Turbofan Spinning Intake Cone
  [-1.46, 1.94, -2.48]    // 2: Starboard Turbofan Spinning Intake Cone
];

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

    // =========================================================================
    // 1. TUBULAR STEEL CHASSIS FRAME, BATTERY BAY, AXLES, & WISHBONE SUSPENSION
    // =========================================================================
    for (const sx of [-0.36, 0.36]) {
      const rail = new THREE.BoxGeometry(0.065, 0.075, 2.95);
      rail.translate(sx, 0.245, -0.05);
      acc.addGeo(rail, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.72);
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

    // Side rocker sills in UI Yellow along floorboard edges
    for (const sx of [-0.59, 0.59]) {
      const sill = new THREE.BoxGeometry(0.045, 0.09, 2.40);
      sill.translate(sx, 0.285, -0.06);
      acc.addGeo(sill, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.94);
    }

    // Front Dual A-Arm Wishbone Suspension, Coil-Over Springs, & Rear EV Motor Differential Axle
    const rearAxle = new THREE.CylinderGeometry(0.032, 0.032, 1.02, 12, 1);
    rearAxle.rotateZ(Math.PI * 0.5);
    rearAxle.translate(0.0, CART_WHEEL_RADIUS, -0.96);
    acc.addGeo(rearAxle, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.70);

    const diffGearbox = new THREE.SphereGeometry(0.095, 14, 10);
    diffGearbox.scale(1.15, 0.95, 1.1);
    diffGearbox.translate(0.06, CART_WHEEL_RADIUS, -0.96);
    acc.addGeo(diffGearbox, [0.20, 0.22, 0.25], HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.72);

    for (const sx of [-1, 1]) {
      // Front upper & lower A-arm wishbones
      for (const ay of [0.20, 0.27]) {
        const arm = new THREE.CylinderGeometry(0.014, 0.014, 0.32, 8, 1);
        arm.rotateZ(Math.PI * 0.5);
        arm.translate(sx * 0.37, ay, 1.18);
        acc.addGeo(arm, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.76);
      }
      // Front & Rear coil-over shock struts
      for (const wz of [1.18, -0.96]) {
        const strut = new THREE.CylinderGeometry(0.024, 0.024, 0.24, 10, 1);
        strut.rotateZ(sx * 0.22);
        strut.translate(sx * 0.43, 0.34, wz);
        acc.addGeo(strut, uiYellow, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.85);
      }
    }

    // =========================================================================
    // 2. SCULPTED UI YELLOW & PEARL-WHITE BODYWORK, COWL, FENDERS & BUMPERS
    // =========================================================================
    // Front lower bumper with rounded end caps
    const frontBumper = new THREE.BoxGeometry(1.22, 0.11, 0.14);
    frontBumper.translate(0.0, 0.27, 1.69);
    acc.addGeo(frontBumper, rubberDark, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.90);

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

    // Front honeycomb intake grille
    const frontGrille = new THREE.BoxGeometry(0.56, 0.13, 0.03);
    frontGrille.translate(0.0, 0.44, 1.645);
    acc.addGeo(frontGrille, chassisBlack, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.76);

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

    // Under-Seat EV Battery Body Tubs (Row 1 + Row 2/3) in Pearl-White with UI Yellow Side Accents & Makara Badges
    const tubFront = new THREE.BoxGeometry(1.10, 0.38, 0.56);
    tubFront.translate(0.0, 0.50, 0.32);
    acc.addGeo(tubFront, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.94);

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
    // 3. PHOTOCELL-COUPLED LED HEADLIGHTS & RUBY LED TAILLIGHTS
    // =========================================================================
    for (const sx of [-0.39, 0.39]) {
      const bezel = new THREE.BoxGeometry(0.22, 0.10, 0.04);
      bezel.translate(sx, 0.58, 1.64);
      acc.addGeo(bezel, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.90);

      const lens = new THREE.BoxGeometry(0.18, 0.072, 0.025);
      lens.translate(sx, 0.58, 1.655);
      acc.addGeo(lens, [0.88, 0.92, 0.96], HUMAN_MAT.VEHICLE_HEADLIGHT, CART_J_CHASSIS, 1.0, 0.0);
    }

    for (const sx of [-0.46, 0.46]) {
      const tailHousing = new THREE.BoxGeometry(0.18, 0.085, 0.035);
      tailHousing.translate(sx, 0.56, -1.41);
      acc.addGeo(tailHousing, [0.48, 0.07, 0.07], HUMAN_MAT.VEHICLE_TAILLIGHT, CART_J_CHASSIS, 1.0, 0.0);
    }

    // =========================================================================
    // 4. RIGHT-HAND-DRIVE EV COCKPIT, DASHBOARD, OLED SCREEN, PEDALS & STEERING
    // =========================================================================
    const dashBody = new THREE.BoxGeometry(1.10, 0.24, 0.26);
    dashBody.translate(0.0, 0.82, 0.95);
    acc.addGeo(dashBody, dashCharcoal, HUMAN_MAT.RUBBER_TIRE, CART_J_CHASSIS, 0.88);

    // Digital EV Speedometer & Battery SOC Display Screen (metaZ = 1.0)
    const oledScreen = new THREE.BoxGeometry(0.19, 0.09, 0.02);
    oledScreen.rotateX(-0.38);
    oledScreen.translate(-0.27, 0.95, 0.88);
    acc.addGeo(oledScreen, [0.12, 0.38, 0.46], HUMAN_MAT.VEHICLE_HEADLIGHT, CART_J_CHASSIS, 1.0, 1.0);

    // Center Dashboard Console: Recessed Twin Cupholders + UI Tumbler + F/N/R Drive Rocker & Key Switch
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

    // Windshield Pantograph Wiper Arm & Blade + Overhead Passenger Grab Handles
    const wiperArm = new THREE.BoxGeometry(0.014, 0.36, 0.012);
    wiperArm.rotateZ(-0.26);
    wiperArm.rotateX(-0.20);
    wiperArm.translate(-0.18, 1.56, 0.942);
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
    const steerTilt = 0.66; // ~38 deg from vertical
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

      // Center seam & side bolster piping
      const seam = new THREE.BoxGeometry(0.025, 0.11, 0.465);
      seam.translate(0.0, 0.688, cz);
      acc.addGeo(seam, seatPiping, HUMAN_MAT.SHOE_BAG, CART_J_CHASSIS, 0.86);

      // Curved hip restraint armrests on left & right sides
      for (const sx of [-0.555, 0.555]) {
        const armLoop = new THREE.BoxGeometry(0.032, 0.16, 0.36);
        armLoop.translate(sx, 0.80, cz - facingSign * 0.03);
        acc.addGeo(armLoop, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.90);
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
    // 6. SPLIT POLYCARBONATE WINDSHIELD, SIDE MIRRORS & 6-STRUT EXTENDED CANOPY
    // =========================================================================
    const aPillarTilt = -0.20;
    for (const sx of [-0.535, 0.535]) {
      // Front A-Pillar Struts
      const pA = new THREE.CylinderGeometry(0.019, 0.019, 1.02, 10, 1);
      pA.rotateX(aPillarTilt);
      pA.translate(sx, 1.35, 0.96);
      acc.addGeo(pA, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.94);

      // Side-View Mirrors mounted on A-pillars
      const mArm = new THREE.BoxGeometry(0.11, 0.02, 0.02);
      mArm.translate(sx + Math.sign(sx) * 0.06, 1.32, 0.97);
      acc.addGeo(mArm, chassisBlack, HUMAN_MAT.METAL_ACCENT, CART_J_CHASSIS, 0.92);

      const mHead = new THREE.BoxGeometry(0.04, 0.14, 0.09);
      mHead.rotateY(Math.sign(sx) * 0.25);
      mHead.translate(sx + Math.sign(sx) * 0.12, 1.33, 0.96);
      acc.addGeo(mHead, chassisBlack, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.95);

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

    // Center Panoramic Rear-View Mirror under front roof header
    const centerMirror = new THREE.BoxGeometry(0.42, 0.075, 0.04);
    centerMirror.translate(0.0, 1.77, 0.83);
    acc.addGeo(centerMirror, chassisBlack, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.90);

    // Extended 6-Seater Pearl-White Sun/Rain Canopy Roof with UI Yellow Perimeter Valance & Rain Gutter!
    const roofCore = new THREE.BoxGeometry(1.22, 0.055, 2.74);
    roofCore.translate(0.0, 1.865, -0.26);
    acc.addGeo(roofCore, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 1.0);

    const roofCrown = new THREE.BoxGeometry(1.06, 0.035, 2.58);
    roofCrown.translate(0.0, 1.895, -0.26);
    acc.addGeo(roofCrown, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 1.0);

    const roofValance = new THREE.BoxGeometry(1.25, 0.032, 2.77);
    roofValance.translate(0.0, 1.842, -0.26);
    acc.addGeo(roofValance, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, CART_J_CHASSIS, 0.98);

    // =========================================================================
    // 7. FOUR ARTICULATED 10-INCH 6-SPOKE ALLOY WHEELS + TREADED TIRES (JOINTS 2..5)
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

      // Main vulcanized rubber tire carcass (outer radius = 0.235m, width = 0.185m)
      const tireMain = new THREE.CylinderGeometry(CART_WHEEL_RADIUS - 0.008, CART_WHEEL_RADIUS - 0.008, 0.185, 24, 1);
      tireMain.rotateZ(Math.PI * 0.5);
      tireMain.translate(wx, wy, wz);
      acc.addGeo(tireMain, rubberDark, HUMAN_MAT.RUBBER_TIRE, jId, 0.92);

      // 12 Physical Raised Shoulder Tread Blocks around circumference so wheel rotation is visibly crisp!
      for (let tb = 0; tb < 12; tb++) {
        const ang = (tb / 12) * Math.PI * 2.0;
        const lug = new THREE.BoxGeometry(0.192, 0.022, 0.044);
        lug.translate(0.0, CART_WHEEL_RADIUS - 0.009, 0.0);
        lug.rotateX(ang);
        lug.translate(wx, wy, wz);
        acc.addGeo(lug, [0.12, 0.125, 0.135], HUMAN_MAT.RUBBER_TIRE, jId, 0.88);
      }

      // Machined 10-inch Alloy Rim Barrel & Outer Lip
      const rimBarrel = new THREE.CylinderGeometry(0.145, 0.145, 0.192, 20, 1);
      rimBarrel.rotateZ(Math.PI * 0.5);
      rimBarrel.translate(wx, wy, wz);
      acc.addGeo(rimBarrel, [0.22, 0.24, 0.27], HUMAN_MAT.METAL_ACCENT, jId, 0.85);

      const rimLip = new THREE.TorusGeometry(0.142, 0.014, 8, 20);
      rimLip.rotateY(Math.PI * 0.5);
      rimLip.translate(wx + ws.side * 0.092, wy, wz);
      acc.addGeo(rimLip, alloySilver, HUMAN_MAT.METAL_ACCENT, jId, 0.98);

      // 6 Sculpted Silver Alloy Spokes + UI Gold Center Hubcap (makes wheel spin unmistakable!)
      for (let sp = 0; sp < 6; sp++) {
        const spAng = (sp / 6) * Math.PI * 2.0;
        const spoke = new THREE.BoxGeometry(0.018, 0.132, 0.028);
        spoke.translate(0.0, 0.068, 0.0);
        spoke.rotateX(spAng);
        spoke.translate(wx + ws.side * 0.090, wy, wz);
        acc.addGeo(spoke, alloySilver, HUMAN_MAT.METAL_ACCENT, jId, 0.98);
      }

      const centerCap = new THREE.CylinderGeometry(0.040, 0.044, 0.032, 12, 1);
      centerCap.rotateZ(Math.PI * 0.5);
      centerCap.translate(wx + ws.side * 0.094, wy, wz);
      acc.addGeo(centerCap, makaraGold, HUMAN_MAT.CREST_GOLD, jId, 1.0);
    }

    return acc.toBundle();
  }

  /**
   * Builds a Traditional Distant Nusantara Outrigger Fishing Boat ("Perahu Nelayan Cadik", ~10.8m long):
   * - Double-ended curved wooden hull with high upswept Linggi bow & stern stemposts
   * - Vibrant coastal Nusantara hull paint stripes, Kajang cabin shelter, bamboo outrigger floats (cadik)
   * - Caping-hatted fisherman silhouette at stern & warm 2500K attractant lanterns at bow & masthead!
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
    const lanternGlass = [0.96, 0.82, 0.46];

    // Double-ended tapered Nusantara wooden hull (+Z bow, -Z stern, waterline at y = 0.0)
    const hullMid = new THREE.CylinderGeometry(0.86, 0.62, 6.8, 12, 1);
    hullMid.rotateX(Math.PI * 0.5);
    hullMid.scale(1.0, 0.78, 1.0);
    hullMid.translate(0.0, 0.34, 0.0);
    acc.addGeo(hullMid, pal.hull, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.95);

    const gunwaleStripe = new THREE.BoxGeometry(1.74, 0.18, 6.6);
    gunwaleStripe.translate(0.0, 0.78, 0.0);
    acc.addGeo(gunwaleStripe, pal.trim, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    // Tapered bow & stern cones + iconic upswept Nusantara Linggi stemposts!
    for (const sz of [-1, 1]) {
      const prow = new THREE.ConeGeometry(0.82, 2.4, 10, 1);
      prow.rotateX(sz * Math.PI * 0.5);
      prow.scale(0.95, 0.76, 1.0);
      prow.translate(0.0, 0.48, sz * 4.4);
      acc.addGeo(prow, pal.hull, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.96);

      const linggi = new THREE.BoxGeometry(0.18, 1.55, 0.32);
      linggi.rotateX(-sz * 0.28);
      linggi.translate(0.0, 1.18, sz * 5.25);
      acc.addGeo(linggi, pal.trim, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);
    }

    // Pitched Kajang Fisherman Cabin Shelter amidships
    const cabinBase = new THREE.BoxGeometry(1.36, 0.92, 2.65);
    cabinBase.translate(0.0, 1.24, -0.35);
    acc.addGeo(cabinBase, pal.cabin, HUMAN_MAT.SHOE_BAG, 0, 0.92);

    const cabinRoof = new THREE.CylinderGeometry(0.96, 0.96, 2.95, 3, 1);
    cabinRoof.rotateZ(Math.PI);
    cabinRoof.rotateX(Math.PI * 0.5);
    cabinRoof.scale(0.95, 0.48, 1.0);
    cabinRoof.translate(0.0, 1.82, -0.35);
    acc.addGeo(cabinRoof, woodTeak, HUMAN_MAT.SHOE_BAG, 0, 0.96);

    // Twin Cross-Booms (Katir) & Port/Starboard Bamboo Outrigger Floats (Cadik)
    for (const bz of [-1.75, 1.55]) {
      const katir = new THREE.CylinderGeometry(0.06, 0.06, 6.4, 8, 1);
      katir.rotateZ(Math.PI * 0.5);
      katir.translate(0.0, 0.86, bz);
      acc.addGeo(katir, woodTeak, HUMAN_MAT.SHOE_BAG, 0, 0.90);
    }
    for (const sx of [-3.15, 3.15]) {
      const cadik = new THREE.CylinderGeometry(0.14, 0.14, 7.6, 10, 1);
      cadik.rotateX(Math.PI * 0.5);
      cadik.translate(sx, 0.08, -0.1);
      acc.addGeo(cadik, bambooGold, HUMAN_MAT.SHOE_BAG, 0, 0.95);

      for (const bz of [-1.75, 1.55]) {
        const stanchion = new THREE.CylinderGeometry(0.045, 0.045, 0.80, 6, 1);
        stanchion.translate(sx, 0.46, bz);
        acc.addGeo(stanchion, woodTeak, HUMAN_MAT.SHOE_BAG, 0, 0.88);
      }
    }

    // Seated Nusantara Fisherman Silhouette with Conical Bamboo Caping Hat at Stern
    const fisherBody = new THREE.CylinderGeometry(0.20, 0.24, 0.78, 8, 1);
    fisherBody.translate(0.0, 1.18, -2.85);
    acc.addGeo(fisherBody, [0.22, 0.28, 0.36], HUMAN_MAT.SHOE_BAG, 0, 0.92);

    const caping = new THREE.ConeGeometry(0.42, 0.24, 12, 1);
    caping.translate(0.0, 1.72, -2.85);
    acc.addGeo(caping, bambooGold, HUMAN_MAT.SHOE_BAG, 0, 0.98);

    // Wooden Mast & Forward Fishing Light Boom + Warm 2500K Fisherman Lanterns (metaZ = 0.0)
    const mast = new THREE.CylinderGeometry(0.055, 0.08, 3.6, 8, 1);
    mast.translate(0.0, 2.55, 1.25);
    acc.addGeo(mast, woodTeak, HUMAN_MAT.SHOE_BAG, 0, 0.94);

    const mastLantern = new THREE.SphereGeometry(0.24, 10, 8);
    mastLantern.translate(0.0, 4.42, 1.25);
    acc.addGeo(mastLantern, lanternGlass, HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 0.0);

    const bowLantern = new THREE.SphereGeometry(0.22, 10, 8);
    bowLantern.translate(0.0, 2.02, 5.38);
    acc.addGeo(bowLantern, lanternGlass, HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 0.0);

    return acc.toBundle();
  }

  /**
   * Builds a Distant Small Coastal Ship ("Kapal Kecil Pesisir / Coaster", ~24.0m long):
   * - Flared steel displacement hull, red waterline boot-top, white 2-deck wheelhouse superstructure
   * - Illuminated bridge windows, radar mast, & COLREGs Red/Green/White marine navigation lights!
   */
  static buildDistantSmallShip(variant = 0) {
    const acc = new RigidJointBundleAccumulator();

    const hullColors = [
      [0.11, 0.22, 0.38], // Maritime Navy Steel
      [0.14, 0.34, 0.36], // Coastal Teal Steel
      [0.20, 0.24, 0.30], // Slate Patrol Grey
      [0.48, 0.16, 0.15]  // Burgundy Coaster Hull
    ];
    const hullCol = hullColors[variant % hullColors.length];
    const bootRed = [0.58, 0.14, 0.12];
    const superWhite = [0.92, 0.94, 0.96];
    const deckGrey = [0.28, 0.31, 0.34];
    const funnelAccent = variant % 2 === 0 ? [0.88, 0.62, 0.10] : [0.16, 0.42, 0.74];

    // Waterline red anti-fouling boot-top + Main steel hull + Flared bow
    const bootHull = new THREE.BoxGeometry(4.8, 0.55, 18.0);
    bootHull.translate(0.0, 0.22, -1.2);
    acc.addGeo(bootHull, bootRed, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.90);

    const midHull = new THREE.BoxGeometry(5.1, 1.75, 17.6);
    midHull.translate(0.0, 1.32, -1.4);
    acc.addGeo(midHull, hullCol, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.96);

    const bowWedge = new THREE.ConeGeometry(3.55, 6.8, 4, 1);
    bowWedge.rotateY(Math.PI * 0.25);
    bowWedge.rotateX(Math.PI * 0.5);
    bowWedge.scale(1.0, 0.46, 1.0);
    bowWedge.translate(0.0, 1.52, 10.2);
    acc.addGeo(bowWedge, hullCol, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    const mainDeck = new THREE.BoxGeometry(4.85, 0.18, 17.2);
    mainDeck.translate(0.0, 2.22, -1.3);
    acc.addGeo(mainDeck, deckGrey, HUMAN_MAT.RUBBER_TIRE, 0, 0.92);

    // 2-Deck White Marine Superstructure & Forward-Raked Wheelhouse Bridge
    const lowerDeckhouse = new THREE.BoxGeometry(4.0, 1.65, 8.6);
    lowerDeckhouse.translate(0.0, 3.12, -1.8);
    acc.addGeo(lowerDeckhouse, superWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    const bridgeHouse = new THREE.BoxGeometry(3.6, 1.55, 4.4);
    bridgeHouse.translate(0.0, 4.68, 0.1);
    acc.addGeo(bridgeHouse, superWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    // Wrap-around Bridge Windows (glowing softly warm-white metaZ = 3.0 at night!)
    const bridgeGlassFront = new THREE.BoxGeometry(3.45, 0.58, 0.18);
    bridgeGlassFront.translate(0.0, 4.86, 2.26);
    acc.addGeo(bridgeGlassFront, [0.22, 0.30, 0.38], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const bridgeGlassSides = new THREE.BoxGeometry(3.68, 0.54, 3.4);
    bridgeGlassSides.translate(0.0, 4.86, 0.25);
    acc.addGeo(bridgeGlassSides, [0.22, 0.30, 0.38], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const bridgeRoof = new THREE.BoxGeometry(3.95, 0.18, 4.9);
    bridgeRoof.translate(0.0, 5.52, 0.15);
    acc.addGeo(bridgeRoof, superWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    // Funnel Stack & Radar Mast
    const funnel = new THREE.CylinderGeometry(0.52, 0.62, 2.1, 12, 1);
    funnel.rotateX(-0.14);
    funnel.translate(0.0, 4.85, -4.1);
    acc.addGeo(funnel, funnelAccent, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.96);

    const radarMast = new THREE.CylinderGeometry(0.09, 0.15, 3.1, 8, 1);
    radarMast.translate(0.0, 7.05, 0.2);
    acc.addGeo(radarMast, superWhite, HUMAN_MAT.METAL_ACCENT, 0, 0.96);

    const radarScanner = new THREE.BoxGeometry(1.55, 0.16, 0.24);
    radarScanner.translate(0.0, 7.85, 0.2);
    acc.addGeo(radarScanner, superWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    // COLREGs Marine Navigation Lights (Masthead White = 3.0, Starboard (-X) Green = 1.0, Port (+X) Red = 2.0)
    const mastLight = new THREE.SphereGeometry(0.28, 10, 8);
    mastLight.translate(0.0, 8.68, 0.2);
    acc.addGeo(mastLight, [0.96, 0.94, 0.85], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const stbdGreen = new THREE.SphereGeometry(0.24, 10, 8);
    stbdGreen.translate(-1.95, 5.20, 1.4);
    acc.addGeo(stbdGreen, [0.20, 0.85, 0.35], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 1.0);

    const portRed = new THREE.SphereGeometry(0.24, 10, 8);
    portRed.translate(1.95, 5.20, 1.4);
    acc.addGeo(portRed, [0.88, 0.20, 0.16], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 2.0);

    return acc.toBundle();
  }

  /**
   * Builds a Sleek 44-ft (13.4m) Pearl-White & Royal-Navy Flybridge Mini Yacht ("Mini Yacht Teluk U"):
   * - Exact air draft = 5.05m above waterline (world top = +3.60m at WATER_LEVEL_Y = -1.45m),
   *   cruising smoothly underneath Jembatan Teksas UI's +7.80m soffit (9.25m air-draft clearance -> +4.20m spare headroom!).
   * - Deep-V planing hull, teak swim platform & aft lounge, foredeck sunpads, stainless bow pulpit rails,
   *   panoramic saloon windows (warm dusk/night cabin glow), upper flybridge helm, swept radar arch + radome, & COLREGs lights.
   */
  static buildFlybridgeMiniYacht() {
    const acc = new RigidJointBundleAccumulator();

    const yachtWhite = [0.95, 0.96, 0.98];
    const royalNavy = [0.09, 0.20, 0.40];
    const goldStripe = [0.90, 0.68, 0.14];
    const teakDeck = [0.54, 0.36, 0.20];
    const cushionCream = [0.89, 0.85, 0.76];
    const stainless = [0.84, 0.87, 0.91];
    const tintedGlass = [0.16, 0.24, 0.32];

    // 1. Deep-V Planing Hull (+Z Bow, -Z Stern, Waterline y = 0.0m, LOA = 13.4m, Beam = 4.05m)
    const lowerChine = new THREE.BoxGeometry(3.55, 0.52, 8.8);
    lowerChine.translate(0.0, 0.22, -0.8);
    acc.addGeo(lowerChine, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.92);

    const goldBoot = new THREE.BoxGeometry(3.72, 0.08, 9.0);
    goldBoot.translate(0.0, 0.50, -0.75);
    acc.addGeo(goldBoot, goldStripe, HUMAN_MAT.CREST_GOLD, 0, 0.98);

    const upperFreeboard = new THREE.BoxGeometry(3.96, 0.96, 8.9);
    upperFreeboard.translate(0.0, 1.00, -0.75);
    acc.addGeo(upperFreeboard, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    // Hull dark-tinted hullside porthole window strips
    const hullWindowStrip = new THREE.BoxGeometry(4.02, 0.22, 4.6);
    hullWindowStrip.translate(0.0, 0.98, 0.35);
    acc.addGeo(hullWindowStrip, tintedGlass, HUMAN_MAT.BOAT_LANTERN, 0, 0.98, 3.0);

    // Raked Flared Clipper Bow (+Z: z = +3.7m .. +6.7m)
    const bowLower = new THREE.ConeGeometry(2.52, 3.6, 4, 1);
    bowLower.rotateY(Math.PI * 0.25);
    bowLower.rotateX(Math.PI * 0.5);
    bowLower.scale(1.0, 0.42, 1.0);
    bowLower.translate(0.0, 0.72, 4.95);
    acc.addGeo(bowLower, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.95);

    const bowUpper = new THREE.ConeGeometry(2.80, 3.85, 4, 1);
    bowUpper.rotateY(Math.PI * 0.25);
    bowUpper.rotateX(Math.PI * 0.5);
    bowUpper.scale(1.0, 0.46, 1.0);
    bowUpper.translate(0.0, 1.18, 5.15);
    acc.addGeo(bowUpper, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    // 2. Teak Aft Swim Platform, Transom Steps, Aft Cockpit Sofa, & Foredeck Sunpads
    const swimPlatform = new THREE.BoxGeometry(3.68, 0.16, 1.48);
    swimPlatform.translate(0.0, 0.24, -5.88);
    acc.addGeo(swimPlatform, teakDeck, HUMAN_MAT.SHOE_BAG, 0, 0.95);

    const swimTrim = new THREE.BoxGeometry(3.76, 0.12, 1.54);
    swimTrim.translate(0.0, 0.18, -5.90);
    acc.addGeo(swimTrim, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.95);

    const mainTeakDeck = new THREE.BoxGeometry(3.78, 0.08, 9.2);
    mainTeakDeck.translate(0.0, 1.50, -0.55);
    acc.addGeo(mainTeakDeck, teakDeck, HUMAN_MAT.SHOE_BAG, 0, 0.96);

    const foreTeakDeck = new THREE.ConeGeometry(2.62, 3.55, 4, 1);
    foreTeakDeck.rotateY(Math.PI * 0.25);
    foreTeakDeck.rotateX(Math.PI * 0.5);
    foreTeakDeck.scale(1.0, 0.05, 1.0);
    foreTeakDeck.translate(0.0, 1.52, 5.05);
    acc.addGeo(foreTeakDeck, teakDeck, HUMAN_MAT.SHOE_BAG, 0, 0.98);

    // Twin Foredeck Luxury Sunpad Mattresses & Aft Cockpit U-Lounge
    for (const sx of [-0.52, 0.52]) {
      const sunpad = new THREE.BoxGeometry(0.86, 0.14, 2.05);
      sunpad.rotateX(0.05);
      sunpad.translate(sx, 1.60, 3.65);
      acc.addGeo(sunpad, cushionCream, HUMAN_MAT.SHOE_BAG, 0, 0.98);
    }
    const aftBench = new THREE.BoxGeometry(2.85, 0.42, 0.68);
    aftBench.translate(0.0, 1.72, -4.65);
    acc.addGeo(aftBench, cushionCream, HUMAN_MAT.SHOE_BAG, 0, 0.96);

    const aftTable = new THREE.BoxGeometry(1.45, 0.06, 0.68);
    aftTable.translate(0.0, 1.98, -3.95);
    acc.addGeo(aftTable, teakDeck, HUMAN_MAT.SHOE_BAG, 0, 0.96);

    // Stainless Steel Bow Pulpit Safety Guardrails
    for (const sx of [-1, 1]) {
      for (let pz = 1.6; pz <= 5.6; pz += 1.0) {
        const taper = Math.max(0.22, 1.0 - Math.max(0.0, pz - 3.4) * 0.34);
        const stanchion = new THREE.CylinderGeometry(0.018, 0.018, 0.58, 6, 1);
        stanchion.translate(sx * 1.78 * taper, 1.80, pz);
        acc.addGeo(stanchion, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.98);
      }
      const bowRail = new THREE.CylinderGeometry(0.020, 0.020, 4.4, 8, 1);
      bowRail.rotateX(Math.PI * 0.5);
      bowRail.rotateY(-sx * 0.25);
      bowRail.translate(sx * 1.24, 2.08, 3.65);
      acc.addGeo(bowRail, stainless, HUMAN_MAT.METAL_ACCENT, 0, 1.0);
    }

    // 3. Panoramic Main Saloon Cabin Superstructure (y = 1.52m .. 2.78m)
    const saloonCore = new THREE.BoxGeometry(3.05, 1.22, 5.25);
    saloonCore.translate(0.0, 2.13, -0.85);
    acc.addGeo(saloonCore, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    // Raked Forward Saloon Windshield & Wrap-Around Saloon Windows (warm interior glow metaZ = 3.0 at dusk/night!)
    const windshield = new THREE.BoxGeometry(2.88, 0.88, 1.35);
    windshield.rotateX(0.56);
    windshield.translate(0.0, 2.14, 1.95);
    acc.addGeo(windshield, tintedGlass, HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const sideSaloonGlass = new THREE.BoxGeometry(3.12, 0.62, 4.35);
    sideSaloonGlass.translate(0.0, 2.24, -0.68);
    acc.addGeo(sideSaloonGlass, tintedGlass, HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    // Overhanging Aft Cockpit Hardtop & Flybridge Sole Deck (y = 2.78m)
    const flySole = new THREE.BoxGeometry(3.22, 0.16, 6.15);
    flySole.translate(0.0, 2.80, -1.35);
    acc.addGeo(flySole, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    for (const sx of [-1.42, 1.42]) {
      const aftPillar = new THREE.CylinderGeometry(0.055, 0.065, 1.28, 8, 1);
      aftPillar.rotateX(-0.14);
      aftPillar.translate(sx, 2.14, -4.18);
      acc.addGeo(aftPillar, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.96);
    }

    // 4. Open Upper Flybridge Helm Console, Coaming Bulwark, Captain Seats, & Venturi Windscreen (y = 2.88m .. 3.72m)
    const flyCoaming = new THREE.BoxGeometry(2.76, 0.52, 3.95);
    flyCoaming.translate(0.0, 3.12, -1.05);
    acc.addGeo(flyCoaming, yachtWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    const flyWindscreen = new THREE.BoxGeometry(2.58, 0.34, 0.55);
    flyWindscreen.rotateX(0.52);
    flyWindscreen.translate(0.0, 3.48, 0.92);
    acc.addGeo(flyWindscreen, [0.36, 0.56, 0.72], HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    for (const sx of [-0.55, 0.55]) {
      const helmSeat = new THREE.BoxGeometry(0.54, 0.56, 0.52);
      helmSeat.translate(sx, 3.36, -0.05);
      acc.addGeo(helmSeat, cushionCream, HUMAN_MAT.SHOE_BAG, 0, 0.98);
    }
    const flyLounge = new THREE.BoxGeometry(2.25, 0.34, 1.25);
    flyLounge.translate(0.0, 3.18, -2.05);
    acc.addGeo(flyLounge, cushionCream, HUMAN_MAT.SHOE_BAG, 0, 0.98);

    // 5. Swept Aerodynamic FRP Radar Arch, Cylindrical Marine Radome, & COLREGs Navigation Lights
    //    (Strict Air Draft = 5.05m above waterline -> +4.20m clear headroom under Jembatan Teksas UI!)
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

    const lightMast = new THREE.CylinderGeometry(0.03, 0.04, 0.24, 8, 1);
    lightMast.translate(0.0, 4.90, -1.88);
    acc.addGeo(lightMast, stainless, HUMAN_MAT.METAL_ACCENT, 0, 1.0);

    // Top All-Round White Anchor/Masthead Light strictly peaking at y = 5.05m!
    const mastheadLight = new THREE.SphereGeometry(0.09, 10, 8);
    mastheadLight.translate(0.0, 4.96, -1.88);
    acc.addGeo(mastheadLight, [0.98, 0.96, 0.88], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    // Starboard (-X) Green & Port (+X) Red COLREGs Sidelights
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
   * - Overall Envelope: 7.80m Length x 2.38m Width x 2.96m Height (clears the 4.15m Underpass Bawah Laut bore by +1.19m!)
   * - 5 Articulated Joints (`BUS_J_CHASSIS` + 4 Ackermann/Rolling Heavy-Duty Alloy Wheels `BUS_J_WHEEL_FL..RR`)
   * - Authentic Bikun UI Yellow & Pearl-White Livery, Royal-Navy Lower Skirt, Golden UI Makara Crests,
   *   Amber LED Route Matrix Display ("PULAU U - UNDERPASS - PULAU I"), Rooftop EV Battery HVAC Pod,
   *   Panoramic Tinted Passenger Windows, Interior Seats, Twin Side Mirrors, & Photocell Headlights/Taillights!
   */
  static buildInterIslandElectricBus() {
    const acc = new RigidJointBundleAccumulator();

    const bikunYellow = [0.94, 0.70, 0.08];
    const pearlWhite = [0.94, 0.95, 0.97];
    const royalNavy = [0.09, 0.19, 0.38];
    const chassisDark = [0.11, 0.12, 0.14];
    const rubberTire = [0.085, 0.088, 0.095];
    const alloySilver = [0.78, 0.81, 0.85];
    const makaraGold = [0.91, 0.70, 0.15];
    const glassTint = [0.22, 0.32, 0.42];
    const seatBlue = [0.16, 0.32, 0.56];

    // 1. Underfloor EV Chassis Frame, Axles, & Navy Rocker Skirt (y = 0.34m .. 0.62m)
    const underChassis = new THREE.BoxGeometry(2.12, 0.24, 7.36);
    underChassis.translate(0.0, 0.44, 0.0);
    acc.addGeo(underChassis, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.74);

    for (const wz of [2.10, -2.10]) {
      const axle = new THREE.CylinderGeometry(0.055, 0.055, 1.96, 10, 1);
      axle.rotateZ(Math.PI * 0.5);
      axle.translate(0.0, BUS_WHEEL_RADIUS, wz);
      acc.addGeo(axle, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.72);
    }

    const lowerSkirt = new THREE.BoxGeometry(2.36, 0.34, 7.62);
    lowerSkirt.translate(0.0, 0.56, 0.0);
    acc.addGeo(lowerSkirt, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.92);

    // Front & Rear energy-absorbing bumpers
    for (const sz of [-1.0, 1.0]) {
      const bumper = new THREE.BoxGeometry(2.40, 0.24, 0.22);
      bumper.translate(0.0, 0.48, sz * 3.82);
      acc.addGeo(bumper, chassisDark, HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.90);
    }

    // 2. Main Iconic Bikun UI Alma Mater Yellow Hollow Perimeter Bodywork (y = 0.72m .. 1.56m) + Pearl-White Waistband
    //    (Hollow interior walls keep the driver cockpit floor, steering wheel, digital cluster, CCTV monitor, & seats 100% open in FPS_BUS_DRIVER!)
    for (const sx of [-1.0, 1.0]) {
      const sideWall = new THREE.BoxGeometry(0.11, 0.84, 7.68);
      sideWall.translate(sx * 1.135, 1.14, 0.0);
      acc.addGeo(sideWall, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);

      const sideWaist = new THREE.BoxGeometry(0.12, 0.11, 7.70);
      sideWaist.translate(sx * 1.14, 1.52, 0.0);
      acc.addGeo(sideWaist, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 1.0);
    }
    const rearWall = new THREE.BoxGeometry(2.38, 0.84, 0.14);
    rearWall.translate(0.0, 1.14, -3.77);
    acc.addGeo(rearWall, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);

    const rearWaist = new THREE.BoxGeometry(2.40, 0.11, 0.15);
    rearWaist.translate(0.0, 1.52, -3.775);
    acc.addGeo(rearWaist, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 1.0);

    const frontNoseApron = new THREE.BoxGeometry(2.38, 0.76, 0.16);
    frontNoseApron.translate(0.0, 1.10, 3.76);
    acc.addGeo(frontNoseApron, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);

    const frontNoseWaist = new THREE.BoxGeometry(2.40, 0.10, 0.17);
    frontNoseWaist.translate(0.0, 1.46, 3.765);
    acc.addGeo(frontNoseWaist, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 1.0);

    // Flared Wheel Arch Fenders
    for (const sx of [-1.0, 1.0]) {
      for (const wz of [2.10, -2.10]) {
        const arch = new THREE.TorusGeometry(0.47, 0.065, 8, 16, Math.PI);
        arch.rotateY(Math.PI * 0.5);
        arch.translate(sx * 1.17, 0.43, wz);
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
      sideCrest.translate(sx * 1.20, 1.16, 0.15);
      acc.addGeo(sideCrest, makaraGold, HUMAN_MAT.CREST_GOLD, BUS_J_CHASSIS, 1.0, 0.0, [0.5, 0.5]);
    }

    // 3. Interior Cabin Floor, 6 Rows of Contoured Blue Transit Seats, & Curbside (+X) Dual-Leaf Passenger Door
    const interiorCore = new THREE.BoxGeometry(2.14, 0.12, 7.10);
    interiorCore.translate(0.0, 0.78, 0.0);
    acc.addGeo(interiorCore, [0.22, 0.24, 0.27], HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.82);

    for (let rz = -2.85; rz <= 1.65; rz += 0.90) {
      for (const sx of [-0.68, 0.68]) {
        const seat = new THREE.BoxGeometry(0.74, 0.54, 0.46);
        seat.translate(sx, 1.14, rz);
        acc.addGeo(seat, seatBlue, HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.90);

        const headrest = new THREE.BoxGeometry(0.56, 0.22, 0.10);
        headrest.translate(sx, 1.49, rz - 0.16);
        acc.addGeo(headrest, [0.12, 0.24, 0.44], HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.92);
      }
    }

    // Interior UI-Yellow Vertical Passenger Grab Stanchions & Overhead Handrails
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
    }

    // =========================================================================
    // 3B. SUPER-DETAILED RIGHT-HAND-DRIVE (RHD, -X) BUS KUNING DRIVER COCKPIT
    // =========================================================================
    // Contoured Air-Suspension RHD Driver Seat (-0.68, 1.18, 2.96) & Safety Partition Screen
    const drvPedestal = new THREE.CylinderGeometry(0.14, 0.17, 0.32, 10, 1);
    drvPedestal.translate(-0.68, 0.98, 2.96);
    acc.addGeo(drvPedestal, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.86);

    const drvCushion = new THREE.BoxGeometry(0.54, 0.12, 0.50);
    drvCushion.translate(-0.68, 1.18, 2.98);
    acc.addGeo(drvCushion, [0.15, 0.18, 0.24], HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.94);

    const drvBack = new THREE.BoxGeometry(0.50, 0.62, 0.11);
    drvBack.rotateX(-0.10);
    drvBack.translate(-0.68, 1.54, 2.75);
    acc.addGeo(drvBack, [0.15, 0.18, 0.24], HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.95);

    const drvPartition = new THREE.BoxGeometry(0.86, 1.05, 0.04);
    drvPartition.translate(-0.66, 1.42, 2.62);
    acc.addGeo(drvPartition, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.90);

    const drvPartGlass = new THREE.BoxGeometry(0.80, 0.72, 0.025);
    drvPartGlass.translate(-0.66, 2.24, 2.62);
    acc.addGeo(drvPartGlass, glassTint, HUMAN_MAT.VEHICLE_GLASS, BUS_J_CHASSIS, 0.98);

    // Sculpted Uniformed Bikun Driver Torso, Shoulders, Arms, & Head (positioned at z = 2.92m so FPS camera at z = 3.16m is 100% unobstructed!)
    const drvTorso = new THREE.CylinderGeometry(0.17, 0.15, 0.48, 10, 1);
    drvTorso.scale(1.15, 1.0, 0.78);
    drvTorso.translate(-0.68, 1.46, 2.93);
    acc.addGeo(drvTorso, bikunYellow, HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.96);

    const drvHead = new THREE.SphereGeometry(0.105, 12, 10);
    drvHead.scale(0.92, 1.08, 0.98);
    drvHead.translate(-0.68, 1.83, 2.92);
    acc.addGeo(drvHead, [0.76, 0.56, 0.42], HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.98);

    const drvCap = new THREE.CylinderGeometry(0.106, 0.110, 0.065, 12, 1);
    drvCap.translate(-0.68, 1.90, 2.92);
    acc.addGeo(drvCap, royalNavy, HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.98);

    for (const armS of [-1, 1]) {
      const drvArm = new THREE.CylinderGeometry(0.044, 0.038, 0.42, 8, 1);
      drvArm.rotateX(1.22);
      drvArm.rotateY(-armS * 0.22);
      drvArm.translate(-0.68 + armS * 0.21, 1.50, 3.14);
      acc.addGeo(drvArm, bikunYellow, HUMAN_MAT.SHOE_BAG, BUS_J_CHASSIS, 0.95);
    }

    // Wrap-Around Ergonomic Transit Dashboard, Digital Cluster, CCTV/GPS Screen, & TapCash Card Validator
    const busDashMain = new THREE.BoxGeometry(2.16, 0.48, 0.42);
    busDashMain.translate(0.0, 1.18, 3.58);
    acc.addGeo(busDashMain, [0.14, 0.15, 0.18], HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.90);

    const busDashHood = new THREE.BoxGeometry(0.62, 0.16, 0.28);
    busDashHood.translate(-0.68, 1.45, 3.58);
    acc.addGeo(busDashHood, chassisDark, HUMAN_MAT.RUBBER_TIRE, BUS_J_CHASSIS, 0.94);

    // Glowing Digital Instrument Cluster & Center GPS/CCTV Monitor (metaZ = 1.0)
    const busClusterScreen = new THREE.BoxGeometry(0.38, 0.12, 0.02);
    busClusterScreen.rotateX(-0.28);
    busClusterScreen.translate(-0.68, 1.44, 3.47);
    acc.addGeo(busClusterScreen, [0.10, 0.42, 0.56], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 1.0);

    const busCctvMonitor = new THREE.BoxGeometry(0.24, 0.15, 0.025);
    busCctvMonitor.rotateY(0.28);
    busCctvMonitor.rotateX(-0.18);
    busCctvMonitor.translate(-0.22, 1.50, 3.48);
    acc.addGeo(busCctvMonitor, [0.14, 0.52, 0.44], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 1.0);

    // Curbside Front Boarding TapCash / UI SmartCard Fare Validator Terminal
    const tapPole = new THREE.CylinderGeometry(0.022, 0.022, 1.18, 8, 1);
    tapPole.translate(0.76, 1.32, 3.16);
    acc.addGeo(tapPole, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.95);

    const tapBox = new THREE.BoxGeometry(0.14, 0.21, 0.08);
    tapBox.rotateX(-0.35);
    tapBox.translate(0.76, 1.54, 3.16);
    acc.addGeo(tapBox, [0.12, 0.65, 0.85], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 1.0);

    // Angled Bus Steering Column + ARTICULATED JOINT #5: 4-Spoke Transit Steering Wheel (`BUS_J_STEER_WHEEL` at [-0.68, 1.48, 3.36])
    const busSteerTilt = 0.44;
    const busCol = new THREE.CylinderGeometry(0.036, 0.044, 0.52, 10, 1);
    busCol.rotateX(busSteerTilt);
    busCol.translate(-0.68, 1.25, 3.47);
    acc.addGeo(busCol, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.90);

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

    // Twin Pantograph Front Windshield Wipers (parked low & horizontal along lower windshield sill for unobstructed FPS view!), Emergency Roof Hatches, & EV Bus License Plates
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
    for (const [pz, ry] of [[3.935, 0.0], [-3.935, Math.PI]]) {
      const plate = new THREE.BoxGeometry(0.42, 0.12, 0.014);
      plate.rotateY(ry);
      plate.translate(0.0, 0.49, pz);
      acc.addGeo(plate, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.96);
    }

    // Curbside (+X Left Side in Indonesian LHT) Full-Height Glazed Folding Passenger Doors
    const curbDoorFrame = new THREE.BoxGeometry(0.05, 1.84, 1.16);
    curbDoorFrame.translate(1.185, 1.56, 2.86);
    acc.addGeo(curbDoorFrame, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.94);
    for (const dz of [-0.26, 0.26]) {
      const doorLeaf = new THREE.BoxGeometry(0.06, 1.72, 0.48);
      doorLeaf.translate(1.192, 1.56, 2.86 + dz);
      acc.addGeo(doorLeaf, glassTint, HUMAN_MAT.VEHICLE_GLASS, BUS_J_CHASSIS, 0.98);
    }

    // 4. Panoramic Glazed Window Greenhouse, Pillars, & Aerodynamic Bikun Yellow / Pearl-White Roof (y = 1.56m .. 2.96m)
    // Front & Rear Bulkhead Headers + Corner Pillars
    for (const sx of [-1.13, 1.13]) {
      for (const pz of [-3.74, -2.25, -0.75, 0.75, 2.22, 3.70]) {
        const pillar = new THREE.BoxGeometry(0.11, 1.06, 0.13);
        pillar.translate(sx, 2.08, pz);
        acc.addGeo(pillar, chassisDark, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.95);
      }
    }

    // Side Panoramic Tinted Passenger Glass Windows
    const sideGlassR = new THREE.BoxGeometry(0.04, 0.98, 7.26);
    sideGlassR.translate(-1.175, 2.08, 0.0);
    acc.addGeo(sideGlassR, glassTint, HUMAN_MAT.VEHICLE_GLASS, BUS_J_CHASSIS, 1.0);

    const sideGlassL = new THREE.BoxGeometry(0.04, 0.98, 5.90);
    sideGlassL.translate(1.175, 2.08, -0.72);
    acc.addGeo(sideGlassL, glassTint, HUMAN_MAT.VEHICLE_GLASS, BUS_J_CHASSIS, 1.0);

    // Swept Panoramic Front Windshield & Rear Window
    const frontWindshield = new THREE.BoxGeometry(2.18, 1.02, 0.06);
    frontWindshield.rotateX(-0.11);
    frontWindshield.translate(0.0, 2.07, 3.78);
    acc.addGeo(frontWindshield, glassTint, HUMAN_MAT.VEHICLE_GLASS, BUS_J_CHASSIS, 1.0);

    const rearGlass = new THREE.BoxGeometry(2.12, 0.88, 0.06);
    rearGlass.translate(0.0, 2.10, -3.81);
    acc.addGeo(rearGlass, glassTint, HUMAN_MAT.VEHICLE_GLASS, BUS_J_CHASSIS, 0.98);

    // Front Destination Header Box + Glowing Amber LED Route Display Matrix
    const headerCant = new THREE.BoxGeometry(2.38, 0.28, 7.66);
    headerCant.translate(0.0, 2.66, 0.0);
    acc.addGeo(headerCant, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 1.0);

    const ledRouteSign = new THREE.BoxGeometry(1.56, 0.18, 0.04);
    ledRouteSign.translate(0.0, 2.65, 3.83);
    acc.addGeo(ledRouteSign, [0.96, 0.66, 0.12], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 1.0);

    // Crowned Pearl-White Roof + Aerodynamic EV Battery & HVAC Pod (strictly peaking at y = 2.96m!)
    const roofDeck = new THREE.BoxGeometry(2.26, 0.10, 7.52);
    roofDeck.translate(0.0, 2.83, 0.0);
    acc.addGeo(roofDeck, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 1.0);

    const evRoofPod = new THREE.BoxGeometry(1.62, 0.14, 3.85);
    evRoofPod.translate(0.0, 2.91, -0.35);
    acc.addGeo(evRoofPod, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 1.0);

    // Twin Outrigger Bus Side-View Mirrors
    for (const sx of [-1.0, 1.0]) {
      const mArm = new THREE.CylinderGeometry(0.018, 0.018, 0.34, 8, 1);
      mArm.rotateZ(Math.PI * 0.5);
      mArm.translate(sx * 1.28, 2.18, 3.86);
      acc.addGeo(mArm, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.95);

      const mHousing = new THREE.BoxGeometry(0.10, 0.34, 0.15);
      mHousing.translate(sx * 1.42, 2.06, 3.92);
      acc.addGeo(mHousing, bikunYellow, HUMAN_MAT.VEHICLE_GLOSS, BUS_J_CHASSIS, 0.98);
    }

    // 5. Photocell-Coupled LED Projector Headlights, Daytime Running Strips, & Ruby LED Taillights
    for (const sx of [-0.84, 0.84]) {
      const hlBezel = new THREE.BoxGeometry(0.42, 0.18, 0.05);
      hlBezel.translate(sx, 0.86, 3.83);
      acc.addGeo(hlBezel, chassisDark, HUMAN_MAT.METAL_ACCENT, BUS_J_CHASSIS, 0.92);

      const hlLens = new THREE.BoxGeometry(0.36, 0.13, 0.04);
      hlLens.translate(sx, 0.86, 3.855);
      acc.addGeo(hlLens, [0.92, 0.95, 0.99], HUMAN_MAT.VEHICLE_HEADLIGHT, BUS_J_CHASSIS, 1.0, 0.0);

      const tailCluster = new THREE.BoxGeometry(0.32, 0.36, 0.05);
      tailCluster.translate(sx * 1.04, 1.02, -3.845);
      acc.addGeo(tailCluster, [0.56, 0.07, 0.07], HUMAN_MAT.VEHICLE_TAILLIGHT, BUS_J_CHASSIS, 1.0, 0.0);
    }

    // 6. Four Articulated Heavy-Duty 6-Spoke Alloy Transit Wheels (`BUS_J_WHEEL_FL..RR`, radius = 0.42m)
    const busWheels = [
      { joint: BUS_J_WHEEL_FL, pivot: BUS_BIND_PIVOTS[BUS_J_WHEEL_FL], side: 1 },
      { joint: BUS_J_WHEEL_FR, pivot: BUS_BIND_PIVOTS[BUS_J_WHEEL_FR], side: -1 },
      { joint: BUS_J_WHEEL_RL, pivot: BUS_BIND_PIVOTS[BUS_J_WHEEL_RL], side: 1 },
      { joint: BUS_J_WHEEL_RR, pivot: BUS_BIND_PIVOTS[BUS_J_WHEEL_RR], side: -1 }
    ];
    for (const ws of busWheels) {
      const [wx, wy, wz] = ws.pivot;
      const jId = ws.joint;
      const tire = new THREE.CylinderGeometry(BUS_WHEEL_RADIUS - 0.01, BUS_WHEEL_RADIUS - 0.01, 0.28, 22, 1);
      tire.rotateZ(Math.PI * 0.5);
      tire.translate(wx, wy, wz);
      acc.addGeo(tire, rubberTire, HUMAN_MAT.RUBBER_TIRE, jId, 0.92);

      for (let tb = 0; tb < 12; tb++) {
        const ang = (tb / 12) * Math.PI * 2.0;
        const lug = new THREE.BoxGeometry(0.286, 0.028, 0.075);
        lug.translate(0.0, BUS_WHEEL_RADIUS - 0.012, 0.0);
        lug.rotateX(ang);
        lug.translate(wx, wy, wz);
        acc.addGeo(lug, [0.13, 0.135, 0.145], HUMAN_MAT.RUBBER_TIRE, jId, 0.88);
      }

      const rim = new THREE.CylinderGeometry(0.25, 0.25, 0.292, 18, 1);
      rim.rotateZ(Math.PI * 0.5);
      rim.translate(wx, wy, wz);
      acc.addGeo(rim, alloySilver, HUMAN_MAT.METAL_ACCENT, jId, 0.95);

      for (let sp = 0; sp < 6; sp++) {
        const spAng = (sp / 6) * Math.PI * 2.0;
        const spoke = new THREE.BoxGeometry(0.025, 0.22, 0.045);
        spoke.translate(0.0, 0.11, 0.0);
        spoke.rotateX(spAng);
        spoke.translate(wx + ws.side * 0.142, wy, wz);
        acc.addGeo(spoke, alloySilver, HUMAN_MAT.METAL_ACCENT, jId, 0.98);
      }

      const hub = new THREE.CylinderGeometry(0.075, 0.082, 0.05, 12, 1);
      hub.rotateZ(Math.PI * 0.5);
      hub.translate(wx + ws.side * 0.148, wy, wz);
      acc.addGeo(hub, makaraGold, HUMAN_MAT.CREST_GOLD, jId, 1.0);
    }

    return acc.toBundle();
  }

  /**
   * Builds the 16.4m x 6.2m UI Electric Passenger Catamaran Shuttle Ferry ("Kapal Feri Katamaran Listrik UI")
   * operating across Northern Selat UI between Dermaga Penyeberangan Utara Pulau U & Dermaga Pulau I:
   * - Twin wave-piercing aluminum catamaran demi-hulls (x = +-2.15m, waterline at y = 0.0m)
   * - Main teak boarding deck at y = +1.23m above waterline (worldY = -0.22m matching both piers' landing stages!)
   * - UI Alma Mater Yellow, Pearl-White, & Royal-Navy Nusantara Livery + Golden UI Makara Crests
   * - Panoramic air-conditioned passenger saloon (warm dusk/night interior glow), raised wheelhouse bridge,
   *   rooftop solar PV canopy, port & starboard boarding gates, SOLAS lifebuoys, radar mast & COLREGs lights!
   */
  static buildIslandShuttleFerry() {
    const acc = new RigidJointBundleAccumulator();

    const ferryWhite = [0.95, 0.96, 0.98];
    const uiYellow = [0.94, 0.70, 0.08];
    const royalNavy = [0.09, 0.20, 0.40];
    const teakDeck = [0.54, 0.36, 0.20];
    const stainless = [0.84, 0.87, 0.91];
    const solarBlue = [0.10, 0.22, 0.44];
    const cabinGlass = [0.16, 0.25, 0.34];
    const lifebuoyOrange = [0.94, 0.32, 0.07];
    const makaraGold = [0.91, 0.70, 0.15];

    // 1. Twin Wave-Piercing Catamaran Demi-Hulls (+Z Bow, -Z Stern, x = +-2.15m, LOA = 16.4m, Waterline y = 0.0)
    for (const sx of [-2.15, 2.15]) {
      const hullLower = new THREE.BoxGeometry(1.42, 0.72, 12.2);
      hullLower.translate(sx, 0.24, -0.5);
      acc.addGeo(hullLower, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.92);

      const hullUpper = new THREE.BoxGeometry(1.52, 0.56, 12.4);
      hullUpper.translate(sx, 0.84, -0.45);
      acc.addGeo(hullUpper, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.96);

      // Wave-piercing raked bow cones (+Z: z = +5.6m .. +8.2m)
      const bow = new THREE.ConeGeometry(0.98, 3.10, 4, 1);
      bow.rotateY(Math.PI * 0.25);
      bow.rotateX(Math.PI * 0.5);
      bow.scale(0.82, 0.58, 1.0);
      bow.translate(sx, 0.62, 6.85);
      acc.addGeo(bow, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.96);

      // Transom electric azimuth thruster pods (-Z)
      const sternStep = new THREE.BoxGeometry(1.36, 0.38, 1.25);
      sternStep.translate(sx, 0.42, -7.10);
      acc.addGeo(sternStep, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.92);
    }

    // 2. Cross-Deck Wet-Deck Bridging Box, Rubber Side Fenders, & Teak Passenger Boarding Promenade (y = 1.23m)
    const wetBridge = new THREE.BoxGeometry(5.96, 0.34, 14.2);
    wetBridge.translate(0.0, 1.04, -0.20);
    acc.addGeo(wetBridge, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.95);

    const yellowBelt = new THREE.BoxGeometry(6.14, 0.22, 14.35);
    yellowBelt.translate(0.0, 1.14, -0.20);
    acc.addGeo(yellowBelt, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    // Perimeter black rubber rubbing strake fender
    const rubStrake = new THREE.BoxGeometry(6.24, 0.14, 14.45);
    rubStrake.translate(0.0, 0.96, -0.20);
    acc.addGeo(rubStrake, [0.10, 0.11, 0.12], HUMAN_MAT.RUBBER_TIRE, 0, 0.90);

    const mainTeak = new THREE.BoxGeometry(5.86, 0.07, 14.05);
    mainTeak.translate(0.0, 1.23, -0.20);
    acc.addGeo(mainTeak, teakDeck, HUMAN_MAT.SHOE_BAG, 0, 0.97);

    // Port & Starboard Stainless Perimeter Guardrails (leaving central 2.6m broadside boarding gate open at z = -0.2m!)
    for (const sx of [-2.88, 2.88]) {
      for (const seg of [
        [-6.9, -1.6],
        [1.2, 6.5]
      ]) {
        const len = seg[1] - seg[0];
        const midZ = 0.5 * (seg[0] + seg[1]);
        const topRail = new THREE.CylinderGeometry(0.026, 0.026, len, 8, 1);
        topRail.rotateX(Math.PI * 0.5);
        topRail.translate(sx, 2.12, midZ);
        acc.addGeo(topRail, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.98);

        for (let pz = seg[0]; pz <= seg[1] + 0.01; pz += 1.75) {
          const post = new THREE.CylinderGeometry(0.022, 0.022, 0.88, 6, 1);
          post.translate(sx, 1.68, pz);
          acc.addGeo(post, stainless, HUMAN_MAT.METAL_ACCENT, 0, 0.96);
        }
      }

      // Twin SOLAS Lifebuoys on Port & Starboard Cabin Coaming
      const ring = new THREE.TorusGeometry(0.28, 0.075, 8, 14);
      ring.rotateY(Math.PI * 0.5);
      ring.translate(sx * 0.84, 1.92, -3.6);
      acc.addGeo(ring, lifebuoyOrange, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);
    }

    // 3. Panoramic Main Passenger Saloon Superstructure (y = 1.26m .. 2.84m) + UI Makara Badges
    const saloonLower = new THREE.BoxGeometry(4.65, 0.72, 9.80);
    saloonLower.translate(0.0, 1.62, -0.35);
    acc.addGeo(saloonLower, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    const saloonYellowBand = new THREE.BoxGeometry(4.70, 0.24, 9.84);
    saloonYellowBand.translate(0.0, 1.84, -0.35);
    acc.addGeo(saloonYellowBand, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    // Warm Dusk/Night Illuminated Panoramic Saloon Windows (`HUMAN_MAT.BOAT_LANTERN`, `metaZ = 3.0`)
    const saloonGlass = new THREE.BoxGeometry(4.72, 0.72, 8.95);
    saloonGlass.translate(0.0, 2.32, -0.35);
    acc.addGeo(saloonGlass, cabinGlass, HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const forwardSaloonGlass = new THREE.BoxGeometry(4.42, 0.82, 1.15);
    forwardSaloonGlass.rotateX(0.46);
    forwardSaloonGlass.translate(0.0, 2.26, 4.55);
    acc.addGeo(forwardSaloonGlass, cabinGlass, HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    for (const sx of [-1.0, 1.0]) {
      const crest = new THREE.CylinderGeometry(0.22, 0.22, 0.03, 16, 1);
      crest.rotateZ(Math.PI * 0.5);
      crest.translate(sx * 2.37, 1.64, -0.20);
      acc.addGeo(crest, makaraGold, HUMAN_MAT.CREST_GOLD, 0, 1.0, 0.0, [0.5, 0.5]);
    }

    const upperDeckSlab = new THREE.BoxGeometry(4.92, 0.16, 10.60);
    upperDeckSlab.translate(0.0, 2.82, -0.45);
    acc.addGeo(upperDeckSlab, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    // 4. Raised Captain's Bridge Wheelhouse (y = 2.90m .. 4.02m) + Aft Rooftop Solar PV Array Canopy
    const bridgeCore = new THREE.BoxGeometry(3.35, 1.04, 3.45);
    bridgeCore.translate(0.0, 3.42, 1.85);
    acc.addGeo(bridgeCore, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    const bridgeGlass = new THREE.BoxGeometry(3.42, 0.52, 2.95);
    bridgeGlass.translate(0.0, 3.56, 1.98);
    acc.addGeo(bridgeGlass, cabinGlass, HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const bridgeRoof = new THREE.BoxGeometry(3.62, 0.14, 3.85);
    bridgeRoof.translate(0.0, 4.00, 1.85);
    acc.addGeo(bridgeRoof, uiYellow, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    // Rooftop Clean-Energy Solar PV Array over Aft Upper Deck
    const solarPanel = new THREE.BoxGeometry(4.25, 0.08, 5.20);
    solarPanel.translate(0.0, 2.95, -2.85);
    acc.addGeo(solarPanel, solarBlue, HUMAN_MAT.VEHICLE_GLOSS, 0, 0.98);

    // 5. Radar Mast, Marine Radome, & COLREGs Navigation Lights
    const mast = new THREE.CylinderGeometry(0.06, 0.08, 1.35, 8, 1);
    mast.translate(0.0, 4.72, 1.65);
    acc.addGeo(mast, ferryWhite, HUMAN_MAT.METAL_ACCENT, 0, 1.0);

    const radome = new THREE.CylinderGeometry(0.38, 0.40, 0.24, 14, 1);
    radome.translate(0.0, 4.56, 1.95);
    acc.addGeo(radome, ferryWhite, HUMAN_MAT.VEHICLE_GLOSS, 0, 1.0);

    const mastheadLight = new THREE.SphereGeometry(0.11, 10, 8);
    mastheadLight.translate(0.0, 5.45, 1.65);
    acc.addGeo(mastheadLight, [0.98, 0.96, 0.88], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 3.0);

    const stbdGreen = new THREE.SphereGeometry(0.10, 8, 6);
    stbdGreen.translate(-1.82, 3.96, 2.45);
    acc.addGeo(stbdGreen, [0.16, 0.90, 0.38], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 1.0);

    const portRed = new THREE.SphereGeometry(0.10, 8, 6);
    portRed.translate(1.82, 3.96, 2.45);
    acc.addGeo(portRed, [0.92, 0.16, 0.14], HUMAN_MAT.BOAT_LANTERN, 0, 1.0, 2.0);

    return acc.toBundle();
  }

  /**
   * Builds a Modern 2-Passenger Electric Helicopter Taxi ("Taxi Helicopter Listrik UI" — 2-Seat Multi-Rotor eVTOL):
   * - 7 Articulated Joints: 0 = Carbon-Pearl Aerodynamic Cabin/Skids/Halo, 1..6 = 6 Counter-Rotating 3-Blade Electric Rotors
   * - Panoramic Wrap-Around Tinted Canopy, 2 Contoured Bucket Seats (Strictly 2 Penumpang!), 2 Seated Occupants,
   *   Central Touchscreen Flight Avionics Display, Sidestick, Composite Landing Skids, Swept Twin-Boom V-Tail,
   *   UI Alma Mater Gold & Carbon Livery + Makara Medallions, and Aviation Navigation/Landing Lights!
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

    // 1. Tubular Titanium-Carbon Landing Skids (bottom strictly at y = 0.0m!) & Boarding Step Bars
    for (const sx of [-0.88, 0.88]) {
      const skid = new THREE.CylinderGeometry(0.036, 0.036, 2.86, 10, 1);
      skid.rotateX(Math.PI * 0.5);
      skid.translate(sx, 0.036, 0.06);
      acc.addGeo(skid, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.90);

      const skiTip = new THREE.CylinderGeometry(0.024, 0.036, 0.34, 8, 1);
      skiTip.rotateX(Math.PI * 0.5 - 0.48);
      skiTip.translate(sx, 0.11, 1.62);
      acc.addGeo(skiTip, uiGold, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 0.96);

      // Front & Rear curved composite cross-arch shock struts
      for (const sz of [-0.66, 0.74]) {
        const strut = new THREE.CylinderGeometry(0.028, 0.032, 0.52, 8, 1);
        strut.rotateZ(-Math.sign(sx) * 0.52);
        strut.translate(sx * 0.82, 0.24, sz);
        acc.addGeo(strut, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.90);
      }

      const stepBar = new THREE.BoxGeometry(0.14, 0.025, 0.68);
      stepBar.translate(sx * 0.90, 0.23, 0.18);
      acc.addGeo(stepBar, titanium, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.94);
    }

    // 2. Sculpted Aerodynamic Teardrop 2-Seat Cabin Pod (+Z Nose Forward, y = 0.38m .. 1.72m)
    const bellyTub = new THREE.BoxGeometry(1.48, 0.36, 2.48);
    bellyTub.translate(0.0, 0.54, 0.18);
    acc.addGeo(bellyTub, carbonDark, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 0.92);

    const lowerBody = new THREE.CylinderGeometry(0.76, 0.82, 2.52, 16, 1);
    lowerBody.rotateX(Math.PI * 0.5);
    lowerBody.scale(1.04, 0.62, 1.0);
    lowerBody.translate(0.0, 0.84, 0.20);
    acc.addGeo(lowerBody, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 0.98);

    const noseCone = new THREE.SphereGeometry(0.78, 16, 12);
    noseCone.scale(0.96, 0.62, 1.15);
    noseCone.translate(0.0, 0.80, 1.38);
    acc.addGeo(noseCone, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 1.0);

    // UI Alma Mater Gold & Royal Navy Speed Cheatlines + Golden UI Makara Door/Nose Crests
    const goldBelt = new THREE.BoxGeometry(1.66, 0.10, 2.35);
    goldBelt.translate(0.0, 0.86, 0.22);
    acc.addGeo(goldBelt, uiGold, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 1.0);

    const navyTrim = new THREE.BoxGeometry(1.64, 0.05, 2.25);
    navyTrim.translate(0.0, 0.78, 0.20);
    acc.addGeo(navyTrim, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 0.98);

    const noseSpotlight = new THREE.BoxGeometry(0.32, 0.08, 0.05);
    noseSpotlight.translate(0.0, 0.66, 2.24);
    acc.addGeo(noseSpotlight, [0.92, 0.96, 1.00], HUMAN_MAT.VEHICLE_HEADLIGHT, EVTOL_J_FUSELAGE, 1.0, 0.0);

    for (const sx of [-1.0, 1.0]) {
      const doorCrest = new THREE.CylinderGeometry(0.11, 0.11, 0.022, 14, 1);
      doorCrest.rotateZ(Math.PI * 0.5);
      doorCrest.translate(sx * 0.835, 0.86, 0.25);
      acc.addGeo(doorCrest, makaraGold, HUMAN_MAT.CREST_GOLD, EVTOL_J_FUSELAGE, 1.0, 0.0, [0.5, 0.5]);
    }

    // 3. Super-Detailed 2-Passenger Cockpit Interior (2 Bucket Seats, Flight Glass Display, & 2 Occupants!)
    const cabinFloor = new THREE.BoxGeometry(1.36, 0.06, 1.95);
    cabinFloor.translate(0.0, 0.56, 0.28);
    acc.addGeo(cabinFloor, [0.18, 0.19, 0.22], HUMAN_MAT.RUBBER_TIRE, EVTOL_J_FUSELAGE, 0.88);

    // Central Flight Avionics Glass Touchscreen Console & Sidestick
    const centerPedestal = new THREE.BoxGeometry(0.24, 0.34, 0.55);
    centerPedestal.translate(0.0, 0.76, 0.76);
    acc.addGeo(centerPedestal, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.92);

    const flightScreen = new THREE.BoxGeometry(0.38, 0.21, 0.025);
    flightScreen.rotateX(-0.36);
    flightScreen.translate(0.0, 1.02, 0.94);
    acc.addGeo(flightScreen, [0.10, 0.56, 0.74], HUMAN_MAT.VEHICLE_HEADLIGHT, EVTOL_J_FUSELAGE, 1.0, 1.0);

    for (const [seatIdx, sx] of [[0, 0.37], [1, -0.37]]) {
      const seatBase = new THREE.BoxGeometry(0.46, 0.10, 0.46);
      seatBase.translate(sx, 0.66, 0.12);
      acc.addGeo(seatBase, seatCream, HUMAN_MAT.SHOE_BAG, EVTOL_J_FUSELAGE, 0.95);

      const seatBack = new THREE.BoxGeometry(0.44, 0.58, 0.09);
      seatBack.rotateX(-0.14);
      seatBack.translate(sx, 0.98, -0.10);
      acc.addGeo(seatBack, seatCream, HUMAN_MAT.SHOE_BAG, EVTOL_J_FUSELAGE, 0.96);

      // Seated Pilot (+0.37) & Campus Passenger (-0.37) inside the 2-Seat eVTOL!
      const occJacket = seatIdx === 0 ? royalNavy : uiGold;
      const occTorso = new THREE.CylinderGeometry(0.14, 0.13, 0.44, 10, 1);
      occTorso.scale(1.12, 1.0, 0.78);
      occTorso.translate(sx, 0.94, 0.06);
      acc.addGeo(occTorso, occJacket, HUMAN_MAT.SHOE_BAG, EVTOL_J_FUSELAGE, 0.96);

      const occHead = new THREE.SphereGeometry(0.096, 10, 8);
      occHead.translate(sx, 1.28, 0.04);
      acc.addGeo(occHead, [0.76, 0.56, 0.42], HUMAN_MAT.SHOE_BAG, EVTOL_J_FUSELAGE, 0.98);
    }

    // Panoramic Bubble Canopy Glass & Sleek Carbon Gullwing Frames
    const canopyDome = new THREE.SphereGeometry(0.76, 18, 12);
    canopyDome.scale(0.98, 0.76, 1.55);
    canopyDome.translate(0.0, 1.18, 0.36);
    acc.addGeo(canopyDome, canopyGlass, HUMAN_MAT.VEHICLE_GLASS, EVTOL_J_FUSELAGE, 1.0);

    const roofSpine = new THREE.BoxGeometry(0.36, 0.14, 1.65);
    roofSpine.translate(0.0, 1.72, -0.12);
    acc.addGeo(roofSpine, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 1.0);

    // 4. Twin Swept Carbon Tail Booms, Canted V-Tail Stabilizers, & Aviation Strobes
    for (const sx of [-0.54, 0.54]) {
      const boom = new THREE.CylinderGeometry(0.048, 0.072, 1.85, 8, 1);
      boom.rotateX(Math.PI * 0.5 + 0.08);
      boom.translate(sx, 1.06, -1.78);
      acc.addGeo(boom, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 0.98);

      const vFin = new THREE.BoxGeometry(0.06, 0.82, 0.46);
      vFin.rotateX(-0.24);
      vFin.rotateZ(-Math.sign(sx) * 0.26);
      vFin.translate(sx * 1.18, 1.46, -2.56);
      acc.addGeo(vFin, uiGold, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 1.0);
    }
    const hTail = new THREE.BoxGeometry(1.36, 0.055, 0.36);
    hTail.translate(0.0, 1.14, -2.58);
    acc.addGeo(hTail, carbonDark, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 0.96);

    const tailBeacon = new THREE.SphereGeometry(0.055, 8, 6);
    tailBeacon.translate(0.0, 1.22, -2.74);
    acc.addGeo(tailBeacon, [0.98, 0.96, 0.90], HUMAN_MAT.BOAT_LANTERN, EVTOL_J_FUSELAGE, 1.0, 3.0);

    // 5. Overhead Multi-Rotor Structural Carbon Halo Ring, Twin Pylon Struts, & 6 Motor Nacelles
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

    // Port (+X) Red & Starboard (-X) Green Halo Aviation Navigation Lights
    const portNav = new THREE.SphereGeometry(0.06, 8, 6);
    portNav.translate(EVTOL_ROTOR_RING_R + 0.03, 2.24, 0.0);
    acc.addGeo(portNav, [0.95, 0.16, 0.14], HUMAN_MAT.BOAT_LANTERN, EVTOL_J_FUSELAGE, 1.0, 2.0);

    const stbdNav = new THREE.SphereGeometry(0.06, 8, 6);
    stbdNav.translate(-(EVTOL_ROTOR_RING_R + 0.03), 2.24, 0.0);
    acc.addGeo(stbdNav, [0.16, 0.92, 0.36], HUMAN_MAT.BOAT_LANTERN, EVTOL_J_FUSELAGE, 1.0, 1.0);

    // 6 Radial Support Arms, Brushless Electric Motor Pods, & 6 ARTICULATED 3-BLADE CARBON ROTORS (JOINTS 1..6)
    for (let rIdx = 0; rIdx < 6; rIdx++) {
      const jointId = rIdx + 1;
      const [hx, hy, hz] = EVTOL_BIND_PIVOTS[jointId];
      const ang = (rIdx / 6) * Math.PI * 2.0;

      const arm = new THREE.CylinderGeometry(0.028, 0.034, EVTOL_ROTOR_RING_R, 8, 1);
      arm.rotateZ(Math.PI * 0.5);
      arm.rotateY(-ang);
      arm.translate(hx * 0.5, 2.24, hz * 0.5);
      acc.addGeo(arm, carbonDark, HUMAN_MAT.METAL_ACCENT, EVTOL_J_FUSELAGE, 0.96);

      const motorPod = new THREE.CylinderGeometry(0.095, 0.085, 0.16, 12, 1);
      motorPod.translate(hx, 2.25, hz);
      acc.addGeo(motorPod, uiGold, HUMAN_MAT.VEHICLE_GLOSS, EVTOL_J_FUSELAGE, 1.0);

      // Articulated Spinning Rotor Spinner + 3 Whisper-Tip Carbon Rotor Blades (`jointId = 1..6`)
      const spinner = new THREE.ConeGeometry(0.072, 0.09, 10, 1);
      spinner.translate(hx, hy + 0.03, hz);
      acc.addGeo(spinner, titanium, HUMAN_MAT.METAL_ACCENT, jointId, 1.0);

      for (let b = 0; b < 3; b++) {
        const bAng = (b / 3) * Math.PI * 2.0;
        const blade = new THREE.BoxGeometry(0.84, 0.012, 0.082);
        blade.translate(0.44, 0.0, 0.0);
        blade.rotateY(bAng);
        blade.translate(hx, hy, hz);
        acc.addGeo(blade, carbonDark, HUMAN_MAT.METAL_ACCENT, jointId, 0.98);

        const bladeTip = new THREE.BoxGeometry(0.11, 0.014, 0.078);
        bladeTip.translate(0.82, 0.0, 0.0);
        bladeTip.rotateY(bAng);
        bladeTip.translate(hx, hy, hz);
        acc.addGeo(bladeTip, uiGold, HUMAN_MAT.VEHICLE_GLOSS, jointId, 1.0);
      }
    }

    return acc.toBundle();
  }

  /**
   * Builds a Super-Detailed High-Wing Strut-Braced Cessna 172 Skyhawk Campus Shuttle Aircraft ("PK-UIA"):
   * - 2 Articulated Joints: 0 = Airframe/Cockpit/Gear, 1 = Spinning 2-Blade McCauley Propeller ([0, 1.18, 2.58])
   * - Aviation Pearl-White Livery with UI Alma Mater Gold & Royal Navy Cheatlines + Makara Tailfin Crest
   * - Cambered 11.0m High Wing with N-Struts, Flap/Aileron Ribs, Tricycle Spring-Steel Gear + Wheel Spats
   * - Full 2-Seat Flight Cockpit for FPS Pilot View: Glareshield, Garmin G1000 Dual Glass Avionics (PFD/MFD),
   *   Throttle Quadrant, Twin Control Yokes, Seated Co-Pilot (-X) & Captain (+X, unobstructed eye at [+0.27, 1.50, +0.36])!
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

    // 1. Tricycle Landing Gear + Aerodynamic Wheel Fairings ("Wheel Pants", bottom strictly at y = 0.0m!)
    // Nose Gear (z = +1.84m)
    const noseStrut = new THREE.CylinderGeometry(0.032, 0.028, 0.56, 8, 1);
    noseStrut.rotateX(-0.14);
    noseStrut.translate(0.0, 0.44, 1.80);
    acc.addGeo(noseStrut, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.94);

    const noseTire = new THREE.CylinderGeometry(0.18, 0.18, 0.11, 16, 1);
    noseTire.rotateZ(Math.PI * 0.5);
    noseTire.translate(0.0, 0.18, 1.84);
    acc.addGeo(noseTire, rubberTire, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.90);

    const nosePant = new THREE.SphereGeometry(0.18, 12, 10);
    nosePant.scale(0.55, 0.72, 1.85);
    nosePant.translate(0.0, 0.23, 1.82);
    acc.addGeo(nosePant, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

    // Main Spring-Steel Landing Gear Legs & Wheel Pants (x = +-1.28m, z = -0.16m)
    for (const sx of [-1.0, 1.0]) {
      const gearLeg = new THREE.BoxGeometry(0.86, 0.055, 0.12);
      gearLeg.rotateZ(-sx * 0.46);
      gearLeg.translate(sx * 0.86, 0.48, -0.16);
      acc.addGeo(gearLeg, aviationWhite, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.94);

      const mainTire = new THREE.CylinderGeometry(0.20, 0.20, 0.13, 16, 1);
      mainTire.rotateZ(Math.PI * 0.5);
      mainTire.translate(sx * 1.28, 0.20, -0.16);
      acc.addGeo(mainTire, rubberTire, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.90);

      const mainPant = new THREE.SphereGeometry(0.20, 12, 10);
      mainPant.scale(0.56, 0.74, 1.92);
      mainPant.translate(sx * 1.28, 0.25, -0.18);
      acc.addGeo(mainPant, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

      const pantStripe = new THREE.BoxGeometry(0.24, 0.04, 0.54);
      pantStripe.translate(sx * 1.28, 0.26, -0.18);
      acc.addGeo(pantStripe, uiGold, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);
    }

    // 2. Sculpted Sloped Engine Cowling (flush below 1.41m glareshield for authentic panoramic FPS pilot sightline!), Cabin Fuselage, Tapered Tailcone, & Empennage (+Z Nose, -Z Tail)
    const cowling = new THREE.CylinderGeometry(0.46, 0.56, 1.26, 16, 1);
    cowling.rotateX(Math.PI * 0.5);
    cowling.scale(1.06, 0.62, 1.0);
    cowling.translate(0.0, 1.04, 1.88);
    acc.addGeo(cowling, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

    // Twin Lycoming Engine Cooling Intakes & Stainless Exhaust Pipes
    for (const sx of [-0.22, 0.22]) {
      const intake = new THREE.BoxGeometry(0.16, 0.11, 0.08);
      intake.translate(sx, 1.08, 2.49);
      acc.addGeo(intake, darkMetal, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.82);

      const exhaust = new THREE.CylinderGeometry(0.026, 0.026, 0.26, 8, 1);
      exhaust.rotateX(0.42);
      exhaust.translate(sx * 0.8, 0.66, 1.72);
      acc.addGeo(exhaust, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.90);
    }

    // Hollow Lower Cabin Belly Floor, Side Fuselage Walls, & Side Cheatlines (100% open interior for Garmin G1000 & Yoke in FPS_PILOT!)
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

      const goldCheatSide = new THREE.BoxGeometry(0.04, 0.09, 5.45);
      goldCheatSide.rotateX(0.035);
      goldCheatSide.translate(sx * 0.588, 1.18, -0.82);
      acc.addGeo(goldCheatSide, uiGold, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

      const navyCheatSide = new THREE.BoxGeometry(0.038, 0.055, 5.35);
      navyCheatSide.rotateX(0.035);
      navyCheatSide.translate(sx * 0.586, 1.09, -0.85);
      acc.addGeo(navyCheatSide, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);
    }

    // Tapered Aft Tailcone (z = -1.05m .. -4.20m)
    const tailCone = new THREE.CylinderGeometry(0.54, 0.18, 3.25, 14, 1);
    tailCone.rotateX(-Math.PI * 0.5 + 0.04);
    tailCone.scale(1.04, 1.08, 1.0);
    tailCone.translate(0.0, 1.24, -2.62);
    acc.addGeo(tailCone, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

    // Swept Dorsal Fin, Vertical Stabilizer, Rudder, Golden UI Makara Crest, & Horizontal Tailplane
    const dorsalFillet = new THREE.BoxGeometry(0.08, 0.34, 1.35);
    dorsalFillet.rotateX(-0.24);
    dorsalFillet.translate(0.0, 1.56, -2.95);
    acc.addGeo(dorsalFillet, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

    const vStab = new THREE.BoxGeometry(0.10, 1.36, 0.96);
    vStab.rotateX(-0.34);
    vStab.translate(0.0, 2.02, -3.86);
    acc.addGeo(vStab, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

    const tailBand = new THREE.BoxGeometry(0.11, 0.24, 0.88);
    tailBand.rotateX(-0.34);
    tailBand.translate(0.0, 2.34, -3.96);
    acc.addGeo(tailBand, uiGold, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

    for (const sx of [-1.0, 1.0]) {
      const tailCrest = new THREE.CylinderGeometry(0.14, 0.14, 0.02, 14, 1);
      tailCrest.rotateZ(Math.PI * 0.5);
      tailCrest.translate(sx * 0.058, 2.02, -3.84);
      acc.addGeo(tailCrest, makaraGold, HUMAN_MAT.CREST_GOLD, CESSNA_J_AIRFRAME, 1.0, 0.0, [0.5, 0.5]);
    }

    const tailBeacon = new THREE.SphereGeometry(0.055, 8, 6);
    tailBeacon.translate(0.0, 2.74, -4.12);
    acc.addGeo(tailBeacon, [0.96, 0.14, 0.12], HUMAN_MAT.BOAT_LANTERN, CESSNA_J_AIRFRAME, 1.0, 2.0);

    const hStab = new THREE.BoxGeometry(3.44, 0.075, 0.78);
    hStab.translate(0.0, 1.32, -3.92);
    acc.addGeo(hStab, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

    // 3. Super-Detailed 2-Seat Garmin G1000 Flight Cockpit (Captain on Left +X = +0.27m, Co-Pilot on Right -X = -0.27m!)
    const glareshield = new THREE.BoxGeometry(1.12, 0.05, 0.28);
    glareshield.translate(0.0, 1.36, 1.06);
    acc.addGeo(glareshield, darkMetal, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.94);

    const instrumentPanel = new THREE.BoxGeometry(1.10, 0.34, 0.08);
    instrumentPanel.translate(0.0, 1.22, 0.92);
    acc.addGeo(instrumentPanel, [0.16, 0.18, 0.21], HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.94);

    // Garmin G1000 Dual PFD (+0.27m) & MFD (-0.02m) Glowing Flight Displays + Throttle Quadrant
    const g1000Pfd = new THREE.BoxGeometry(0.28, 0.19, 0.02);
    g1000Pfd.translate(0.27, 1.25, 0.875);
    acc.addGeo(g1000Pfd, [0.12, 0.56, 0.86], HUMAN_MAT.VEHICLE_HEADLIGHT, CESSNA_J_AIRFRAME, 1.0, 1.0);

    const g1000Mfd = new THREE.BoxGeometry(0.28, 0.19, 0.02);
    g1000Mfd.translate(-0.04, 1.25, 0.875);
    acc.addGeo(g1000Mfd, [0.12, 0.64, 0.52], HUMAN_MAT.VEHICLE_HEADLIGHT, CESSNA_J_AIRFRAME, 1.0, 1.0);

    const throttleKnob = new THREE.CylinderGeometry(0.018, 0.018, 0.12, 8, 1);
    throttleKnob.rotateX(Math.PI * 0.5);
    throttleKnob.translate(0.0, 1.08, 0.84);
    acc.addGeo(throttleKnob, darkMetal, HUMAN_MAT.RUBBER_TIRE, CESSNA_J_AIRFRAME, 0.92);

    // Twin Sculpted Ram's-Horn Control Yokes & Contoured Leather Pilot/Co-Pilot Seats
    for (const [pIdx, sx] of [[0, 0.27], [1, -0.27]]) {
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
      seatCush.translate(sx, 0.84, 0.18);
      acc.addGeo(seatCush, seatTan, HUMAN_MAT.SHOE_BAG, CESSNA_J_AIRFRAME, 0.94);

      const seatBack = new THREE.BoxGeometry(0.42, 0.56, 0.09);
      seatBack.rotateX(-0.12);
      seatBack.translate(sx, 1.16, -0.02);
      acc.addGeo(seatBack, seatTan, HUMAN_MAT.SHOE_BAG, CESSNA_J_AIRFRAME, 0.95);

      // Seated Captain (+0.27, head at z = +0.12 so FPS eye at z = +0.36 is 100% clear!) & Co-Pilot (-0.27)
      const pilotShirt = pIdx === 0 ? aviationWhite : uiGold;
      const pTorso = new THREE.CylinderGeometry(0.14, 0.13, 0.44, 10, 1);
      pTorso.scale(1.12, 1.0, 0.76);
      pTorso.translate(sx, 1.14, 0.13);
      acc.addGeo(pTorso, pilotShirt, HUMAN_MAT.SHOE_BAG, CESSNA_J_AIRFRAME, 0.96);

      const pHead = new THREE.SphereGeometry(0.095, 10, 8);
      pHead.translate(sx, 1.48, 0.12);
      acc.addGeo(pHead, [0.76, 0.56, 0.42], HUMAN_MAT.SHOE_BAG, CESSNA_J_AIRFRAME, 0.98);
    }

    // Slim Cockpit A-Pillars, Side Windows, & Swept Panoramic Windscreen
    for (const sx of [-0.56, 0.56]) {
      const aPillar = new THREE.CylinderGeometry(0.022, 0.024, 0.68, 8, 1);
      aPillar.rotateX(0.58);
      aPillar.translate(sx, 1.66, 0.96);
      acc.addGeo(aPillar, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

      const bPillar = new THREE.BoxGeometry(0.05, 0.62, 0.08);
      bPillar.translate(sx, 1.66, -0.18);
      acc.addGeo(bPillar, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 0.98);

      const sideWin = new THREE.BoxGeometry(0.025, 0.54, 1.85);
      sideWin.translate(sx * 1.01, 1.65, 0.02);
      acc.addGeo(sideWin, glassTint, HUMAN_MAT.VEHICLE_GLASS, CESSNA_J_AIRFRAME, 1.0);
    }
    const windscreen = new THREE.BoxGeometry(1.10, 0.66, 0.03);
    windscreen.rotateX(0.60);
    windscreen.translate(0.0, 1.66, 0.98);
    acc.addGeo(windscreen, glassTint, HUMAN_MAT.VEHICLE_GLASS, CESSNA_J_AIRFRAME, 1.0);

    // 4. High Cambered 11.0m Wing, Twin Diagonal Struts, Nav/Strobe Lights, & LED Landing Light
    const wingCenter = new THREE.BoxGeometry(1.24, 0.13, 1.54);
    wingCenter.translate(0.0, 1.98, 0.08);
    acc.addGeo(wingCenter, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

    for (const sx of [-1.0, 1.0]) {
      const wingPanel = new THREE.BoxGeometry(4.92, 0.12, 1.48);
      wingPanel.rotateZ(sx * 0.028); // Authentic 1.6-deg high-wing dihedral
      wingPanel.translate(sx * 3.04, 2.05, 0.08);
      acc.addGeo(wingPanel, aviationWhite, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

      const wingTipStripe = new THREE.BoxGeometry(0.45, 0.125, 1.49);
      wingTipStripe.translate(sx * 5.26, 2.11, 0.08);
      acc.addGeo(wingTipStripe, uiGold, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_AIRFRAME, 1.0);

      // Diagonal Wing Lift Strut from lower fuselage sill to mid-wing
      const strut = new THREE.CylinderGeometry(0.032, 0.032, 2.52, 8, 1);
      strut.rotateZ(-sx * 1.12);
      strut.translate(sx * 1.68, 1.42, 0.24);
      acc.addGeo(strut, aviationWhite, HUMAN_MAT.METAL_ACCENT, CESSNA_J_AIRFRAME, 0.96);

      // Port (+X) Red & Starboard (-X) Green Wingtip Navigation Lights
      const navLight = new THREE.SphereGeometry(0.055, 8, 6);
      navLight.translate(sx * 5.52, 2.12, 0.42);
      acc.addGeo(
        navLight,
        sx > 0 ? [0.95, 0.16, 0.14] : [0.16, 0.92, 0.36],
        HUMAN_MAT.BOAT_LANTERN,
        CESSNA_J_AIRFRAME,
        1.0,
        sx > 0 ? 2.0 : 1.0
      );
    }

    const wingLandingLight = new THREE.BoxGeometry(0.28, 0.07, 0.04);
    wingLandingLight.translate(2.35, 2.03, 0.83);
    acc.addGeo(wingLandingLight, [0.95, 0.98, 1.00], HUMAN_MAT.VEHICLE_HEADLIGHT, CESSNA_J_AIRFRAME, 1.0, 0.0);

    // 5. ARTICULATED JOINT #1: Conical Propeller Spinner & 2-Blade Airfoil Propeller (`CESSNA_J_PROP` at [0, 1.18, 2.58])
    const [px, py, pz] = CESSNA_BIND_PIVOTS[CESSNA_J_PROP];
    const propSpinner = new THREE.ConeGeometry(0.16, 0.28, 14, 1);
    propSpinner.rotateX(Math.PI * 0.5);
    propSpinner.translate(px, py, pz + 0.06);
    acc.addGeo(propSpinner, chrome, HUMAN_MAT.METAL_ACCENT, CESSNA_J_PROP, 1.0);

    for (const sBlade of [-1.0, 1.0]) {
      const blade = new THREE.BoxGeometry(0.11, 0.92, 0.022);
      blade.rotateY(0.22 * sBlade);
      blade.translate(px, py + sBlade * 0.48, pz);
      acc.addGeo(blade, darkMetal, HUMAN_MAT.METAL_ACCENT, CESSNA_J_PROP, 0.98);

      const bladeTip = new THREE.BoxGeometry(0.105, 0.11, 0.024);
      bladeTip.rotateY(0.22 * sBlade);
      bladeTip.translate(px, py + sBlade * 0.89, pz);
      acc.addGeo(bladeTip, uiGold, HUMAN_MAT.VEHICLE_GLOSS, CESSNA_J_PROP, 1.0);
    }

    return acc.toBundle();
  }

  /**
   * Builds a Super-Detailed Swept-Wing T-Tail Citation Learjet Executive Shuttle Jet ("PK-UIB"):
   * - 3 Articulated Joints: 0 = Airframe/Cockpit/Gear, 1 & 2 = Twin Rear Turbofan Spinning Intake Fan Cones
   * - 14.6m Pressurized Executive Fuselage in Pearl-White, Titanium Belly, & UI Gold/Navy Executive Cheatlines
   * - Low Swept Supercritical Wing (13.8m span) with Iconic Upturned Canted Winglets, Swept T-Tail,
   *   Trailing-Link Landing Gear, 10 Illuminated Oval Executive Cabin Windows, & Golden UI Makara Tail Crest
   * - Full 2-Pilot Jet Flight Deck for FPS Pilot View: Glareshield, 3-Screen Pro Line Glass Avionics Suite,
   *   Center Thrust Pedestal, Dual Jet Control Yokes, First Officer (-X) & Captain (+X, eye at [+0.32, 1.78, +3.86])!
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

    // 1. Twin-Wheel Nose Gear & Trailing-Link Main Landing Gear (bottom strictly at y = 0.0m!)
    const noseOleo = new THREE.CylinderGeometry(0.045, 0.040, 0.68, 10, 1);
    noseOleo.translate(0.0, 0.48, 4.42);
    acc.addGeo(noseOleo, chrome, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.95);

    for (const wx of [-0.11, 0.11]) {
      const nTire = new THREE.CylinderGeometry(0.18, 0.18, 0.085, 16, 1);
      nTire.rotateZ(Math.PI * 0.5);
      nTire.translate(wx, 0.18, 4.42);
      acc.addGeo(nTire, rubberTire, HUMAN_MAT.RUBBER_TIRE, LEARJET_J_AIRFRAME, 0.90);
    }

    const taxiLight = new THREE.BoxGeometry(0.18, 0.08, 0.05);
    taxiLight.translate(0.0, 0.56, 4.48);
    acc.addGeo(taxiLight, [0.95, 0.98, 1.00], HUMAN_MAT.VEHICLE_HEADLIGHT, LEARJET_J_AIRFRAME, 1.0, 0.0);

    for (const sx of [-1.82, 1.82]) {
      const mainStrut = new THREE.CylinderGeometry(0.055, 0.048, 0.72, 10, 1);
      mainStrut.translate(sx, 0.54, -0.78);
      acc.addGeo(mainStrut, chrome, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.94);

      for (const wxOff of [-0.13, 0.13]) {
        const mTire = new THREE.CylinderGeometry(0.24, 0.24, 0.11, 16, 1);
        mTire.rotateZ(Math.PI * 0.5);
        mTire.translate(sx + wxOff, 0.24, -0.78);
        acc.addGeo(mTire, rubberTire, HUMAN_MAT.RUBBER_TIRE, LEARJET_J_AIRFRAME, 0.90);
      }
    }

    // 2. Sleek Pressurized Cylindrical Executive Fuselage, Radome Nose, & Tailcone (+Z Nose, -Z Tail)
    const fuselageMain = new THREE.CylinderGeometry(0.88, 0.88, 7.60, 20, 1);
    fuselageMain.rotateX(Math.PI * 0.5);
    fuselageMain.translate(0.0, 1.42, -0.10);
    acc.addGeo(fuselageMain, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    const bellyFairing = new THREE.BoxGeometry(1.78, 0.34, 5.20);
    bellyFairing.translate(0.0, 0.78, -0.35);
    acc.addGeo(bellyFairing, titaniumBelly, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.95);

    // Hollow Flight-Deck Cheek Walls (z = 3.55..4.42m) + Forward Nose Section Ahead of Instrument Panel (z = 4.38..5.42m)
    for (const sx of [-1.0, 1.0]) {
      const cheekWall = new THREE.BoxGeometry(0.10, 0.62, 0.96);
      cheekWall.rotateY(-sx * 0.14);
      cheekWall.translate(sx * 0.74, 1.30, 4.02);
      acc.addGeo(cheekWall, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.99);
    }
    const forwardLower = new THREE.CylinderGeometry(0.60, 0.78, 1.08, 18, 1);
    forwardLower.rotateX(Math.PI * 0.5);
    forwardLower.scale(1.0, 0.52, 1.0);
    forwardLower.translate(0.0, 1.14, 4.90);
    acc.addGeo(forwardLower, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    const radome = new THREE.ConeGeometry(0.60, 1.65, 18, 1);
    radome.rotateX(Math.PI * 0.5 + 0.08);
    radome.scale(1.0, 0.70, 1.0);
    radome.translate(0.0, 1.08, 6.10);
    acc.addGeo(radome, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    const radomeTip = new THREE.ConeGeometry(0.22, 0.52, 14, 1);
    radomeTip.rotateX(Math.PI * 0.5 + 0.08);
    radomeTip.translate(0.0, 1.08, 6.72);
    acc.addGeo(radomeTip, darkCarbon, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.98);

    // Executive UI Gold & Royal Navy Side Fuselage Cheatlines (hollow side-mounted so flight-deck interior stays 100% open!) + 5 Oval Cabin Windows per Side
    for (const sx of [-1.0, 1.0]) {
      const jetGoldSide = new THREE.BoxGeometry(0.045, 0.09, 7.65);
      jetGoldSide.translate(sx * 0.875, 1.38, -0.25);
      acc.addGeo(jetGoldSide, uiGold, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

      const jetNavySide = new THREE.BoxGeometry(0.042, 0.06, 7.55);
      jetNavySide.translate(sx * 0.873, 1.29, -0.28);
      acc.addGeo(jetNavySide, royalNavy, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.98);
    }

    for (const sx of [-1.0, 1.0]) {
      for (let w = 0; w < 5; w++) {
        const wz = 2.15 - w * 0.92;
        const ovalWin = new THREE.CylinderGeometry(0.15, 0.15, 0.04, 12, 1);
        ovalWin.rotateZ(Math.PI * 0.5);
        ovalWin.scale(1.0, 1.22, 0.85);
        ovalWin.translate(sx * 0.875, 1.64, wz);
        acc.addGeo(ovalWin, glassTint, HUMAN_MAT.BOAT_LANTERN, LEARJET_J_AIRFRAME, 1.0, 3.0);
      }
    }

    // Tapered Rear Tailcone & Iconic Swept T-Tail Empennage
    const rearCone = new THREE.CylinderGeometry(0.86, 0.24, 3.15, 16, 1);
    rearCone.rotateX(-Math.PI * 0.5 + 0.08);
    rearCone.translate(0.0, 1.54, -5.35);
    acc.addGeo(rearCone, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.98);

    const tTailFin = new THREE.BoxGeometry(0.14, 2.48, 1.55);
    tTailFin.rotateX(-0.42);
    tTailFin.translate(0.0, 2.92, -6.05);
    acc.addGeo(tTailFin, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    const tTailGold = new THREE.BoxGeometry(0.15, 0.34, 1.38);
    tTailGold.rotateX(-0.42);
    tTailGold.translate(0.0, 3.68, -6.36);
    acc.addGeo(tTailGold, uiGold, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    for (const sx of [-1.0, 1.0]) {
      const tailCrest = new THREE.CylinderGeometry(0.20, 0.20, 0.025, 14, 1);
      tailCrest.rotateZ(Math.PI * 0.5);
      tailCrest.translate(sx * 0.08, 2.98, -6.02);
      acc.addGeo(tailCrest, makaraGold, HUMAN_MAT.CREST_GOLD, LEARJET_J_AIRFRAME, 1.0, 0.0, [0.5, 0.5]);
    }

    const tTailBullet = new THREE.CylinderGeometry(0.11, 0.11, 1.65, 10, 1);
    tTailBullet.rotateX(Math.PI * 0.5);
    tTailBullet.translate(0.0, 4.12, -6.58);
    acc.addGeo(tTailBullet, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    for (const sx of [-1.0, 1.0]) {
      const hTail = new THREE.BoxGeometry(2.38, 0.08, 0.86);
      hTail.rotateY(-sx * 0.24);
      hTail.translate(sx * 1.18, 4.12, -6.72);
      acc.addGeo(hTail, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);
    }

    const topStrobe = new THREE.SphereGeometry(0.065, 8, 6);
    topStrobe.translate(0.0, 4.26, -6.58);
    acc.addGeo(topStrobe, [0.96, 0.14, 0.12], HUMAN_MAT.BOAT_LANTERN, LEARJET_J_AIRFRAME, 1.0, 2.0);

    // 3. Super-Detailed 2-Pilot Jet Flight Deck (Captain +X = +0.32m, First Officer -X = -0.32m!)
    const cockpitFloor = new THREE.BoxGeometry(1.42, 0.08, 1.55);
    cockpitFloor.translate(0.0, 1.02, 4.02);
    acc.addGeo(cockpitFloor, darkCarbon, HUMAN_MAT.RUBBER_TIRE, LEARJET_J_AIRFRAME, 0.90);

    const jetGlareshield = new THREE.BoxGeometry(1.36, 0.08, 0.44);
    jetGlareshield.translate(0.0, 1.61, 4.52);
    acc.addGeo(jetGlareshield, darkCarbon, HUMAN_MAT.RUBBER_TIRE, LEARJET_J_AIRFRAME, 0.95);

    const jetPanel = new THREE.BoxGeometry(1.34, 0.36, 0.08);
    jetPanel.translate(0.0, 1.44, 4.38);
    acc.addGeo(jetPanel, [0.16, 0.18, 0.22], HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.95);

    // 3-Screen Pro Line Glass Avionics Suite (Captain PFD, Center EICAS/MFD, FO PFD)
    for (const [scrX, scrCol] of [
      [0.34, [0.10, 0.56, 0.88]],
      [0.0, [0.12, 0.68, 0.54]],
      [-0.34, [0.10, 0.56, 0.88]]
    ]) {
      const mfd = new THREE.BoxGeometry(0.28, 0.20, 0.02);
      mfd.translate(scrX, 1.47, 4.335);
      acc.addGeo(mfd, scrCol, HUMAN_MAT.VEHICLE_HEADLIGHT, LEARJET_J_AIRFRAME, 1.0, 1.0);
    }

    const thrustPedestal = new THREE.BoxGeometry(0.22, 0.32, 0.58);
    thrustPedestal.translate(0.0, 1.18, 4.08);
    acc.addGeo(thrustPedestal, darkCarbon, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.92);

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
      jSeat.translate(sx, 1.14, 3.64);
      acc.addGeo(jSeat, leatherCream, HUMAN_MAT.SHOE_BAG, LEARJET_J_AIRFRAME, 0.95);

      const jBack = new THREE.BoxGeometry(0.44, 0.62, 0.10);
      jBack.rotateX(-0.12);
      jBack.translate(sx, 1.48, 3.42);
      acc.addGeo(jBack, leatherCream, HUMAN_MAT.SHOE_BAG, LEARJET_J_AIRFRAME, 0.96);

      // Seated Jet Captain (+0.32, head at z = +3.58 so FPS eye at z = +3.86 is 100% clear!) & First Officer (-0.32)
      const pTorso = new THREE.CylinderGeometry(0.145, 0.135, 0.46, 10, 1);
      pTorso.scale(1.12, 1.0, 0.76);
      pTorso.translate(sx, 1.44, 3.60);
      acc.addGeo(pTorso, pIdx === 0 ? pearlWhite : royalNavy, HUMAN_MAT.SHOE_BAG, LEARJET_J_AIRFRAME, 0.96);

      const pHead = new THREE.SphereGeometry(0.098, 10, 8);
      pHead.translate(sx, 1.77, 3.58);
      acc.addGeo(pHead, [0.76, 0.56, 0.42], HUMAN_MAT.SHOE_BAG, LEARJET_J_AIRFRAME, 0.98);
    }

    // Swept Multi-Pane Executive Jet Cockpit Windscreen & Center Post
    const centerPost = new THREE.CylinderGeometry(0.022, 0.022, 0.88, 8, 1);
    centerPost.rotateX(0.84);
    centerPost.translate(0.0, 1.88, 4.62);
    acc.addGeo(centerPost, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.98);

    const jetWindscreen = new THREE.BoxGeometry(1.32, 0.86, 0.035);
    jetWindscreen.rotateX(0.84);
    jetWindscreen.translate(0.0, 1.86, 4.60);
    acc.addGeo(jetWindscreen, glassTint, HUMAN_MAT.VEHICLE_GLASS, LEARJET_J_AIRFRAME, 1.0);

    const cockpitRoof = new THREE.CylinderGeometry(0.68, 0.84, 1.15, 14, 1);
    cockpitRoof.rotateX(Math.PI * 0.5);
    cockpitRoof.scale(1.0, 0.35, 1.0);
    cockpitRoof.translate(0.0, 2.08, 3.82);
    acc.addGeo(cockpitRoof, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

    // 4. Low Swept Supercritical Wing (13.8m Span) & Iconic Upturned Canted Winglets
    for (const sx of [-1.0, 1.0]) {
      const wing = new THREE.BoxGeometry(6.15, 0.14, 1.85);
      wing.rotateY(-sx * 0.24); // 14-deg aft wing sweep
      wing.rotateZ(sx * 0.045); // Dihedral
      wing.translate(sx * 3.75, 0.92, -0.78);
      acc.addGeo(wing, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

      // Iconic Canted Learjet Winglet + UI Gold Winglet Tip
      const winglet = new THREE.BoxGeometry(0.08, 1.12, 0.62);
      winglet.rotateX(-0.34);
      winglet.rotateZ(-sx * 0.18);
      winglet.translate(sx * 6.82, 1.54, -1.58);
      acc.addGeo(winglet, uiGold, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

      const navStrobe = new THREE.SphereGeometry(0.06, 8, 6);
      navStrobe.translate(sx * 6.92, 2.08, -1.72);
      acc.addGeo(
        navStrobe,
        sx > 0 ? [0.95, 0.16, 0.14] : [0.16, 0.92, 0.36],
        HUMAN_MAT.BOAT_LANTERN,
        LEARJET_J_AIRFRAME,
        1.0,
        sx > 0 ? 2.0 : 1.0
      );
    }

    // 5. Twin Rear-Fuselage High-Bypass Turbofan Nacelles & ARTICULATED SPINNING INTAKE FANS (JOINTS 1 & 2)
    const turbofanSpecs = [
      { sx: 1.0, joint: LEARJET_J_FAN_L, pivot: LEARJET_BIND_PIVOTS[LEARJET_J_FAN_L] },
      { sx: -1.0, joint: LEARJET_J_FAN_R, pivot: LEARJET_BIND_PIVOTS[LEARJET_J_FAN_R] }
    ];
    for (const tf of turbofanSpecs) {
      const [fx, fy, fz] = tf.pivot;
      const pylon = new THREE.BoxGeometry(0.56, 0.12, 1.25);
      pylon.rotateZ(tf.sx * 0.25);
      pylon.translate(tf.sx * 1.04, 1.80, -3.35);
      acc.addGeo(pylon, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 0.96);

      const nacelle = new THREE.CylinderGeometry(0.46, 0.42, 2.18, 18, 1);
      nacelle.rotateX(Math.PI * 0.5);
      nacelle.translate(fx, fy, -3.45);
      acc.addGeo(nacelle, pearlWhite, HUMAN_MAT.VEHICLE_GLOSS, LEARJET_J_AIRFRAME, 1.0);

      const intakeLip = new THREE.TorusGeometry(0.43, 0.045, 8, 18);
      intakeLip.translate(fx, fy, fz + 0.09);
      acc.addGeo(intakeLip, chrome, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 1.0);

      const exhaustCone = new THREE.CylinderGeometry(0.36, 0.26, 0.45, 14, 1);
      exhaustCone.rotateX(-Math.PI * 0.5);
      exhaustCone.translate(fx, fy, -4.68);
      acc.addGeo(exhaustCone, titaniumBelly, HUMAN_MAT.METAL_ACCENT, LEARJET_J_AIRFRAME, 0.92);

      // Articulated Spinning Turbofan Hub Cone + 10 Titanium Fan Blades (`joint = 1 or 2`)
      const fanDisc = new THREE.CylinderGeometry(0.39, 0.39, 0.04, 16, 1);
      fanDisc.rotateX(Math.PI * 0.5);
      fanDisc.translate(fx, fy, fz - 0.04);
      acc.addGeo(fanDisc, darkCarbon, HUMAN_MAT.METAL_ACCENT, tf.joint, 0.88);

      const fanCone = new THREE.ConeGeometry(0.11, 0.22, 12, 1);
      fanCone.rotateX(Math.PI * 0.5);
      fanCone.translate(fx, fy, fz + 0.05);
      acc.addGeo(fanCone, chrome, HUMAN_MAT.METAL_ACCENT, tf.joint, 1.0);

      for (let b = 0; b < 10; b++) {
        const bAng = (b / 10) * Math.PI * 2.0;
        const fBlade = new THREE.BoxGeometry(0.055, 0.36, 0.016);
        fBlade.translate(0.0, 0.19, 0.0);
        fBlade.rotateZ(bAng);
        fBlade.translate(fx, fy, fz - 0.01);
        acc.addGeo(fBlade, titaniumBelly, HUMAN_MAT.METAL_ACCENT, tf.joint, 0.96);
      }
    }

    return acc.toBundle();
  }
}
