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
}
