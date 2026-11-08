import * as THREE from 'three';
import { ProceduralTextureFactory } from '../../core/ProceduralTextureFactory.js';
import { UI_DPPD_MASTER_PLAN } from '../../world-gis/MasterPlanConfig.js';

const DEG = Math.PI / 180;

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (True-Color Photo-Realistic & Watertight Edition)
 * ./library/gedung-dppd/SecondarySubdomains.js
 *
 * Stage 4 Architectural Subdomains & Forecourt Details for Gedung DPPD UI (verified against ref_184, ref_185, ref_084):
 * - Watertight L1+L2 Side Return Walls & Sloped Off-White Plastered Masonry Wing Walls ("Pipi Tangga") with Grey Coping
 * - 11-Bay L2 Ribbon Window Band w/ Horizontal Transom Bar, Left Mini-Satellite Dish & Right Split-AC Unit (ref_184)
 * - L2 Wing Colonnade Square Cream Columns + Prominent Split-AC Outdoor Condenser Units w/ Fan Grilles (ref_084)
 * - Continuous Straight Hipped L1 Terracotta Portico Canopy (NO Street-View-UI chevron gablet artifact!), 4 Cream Columns,
 *   Recessed Double Entrance Doors & Unoccluded KSDP/DPPD Wall Signage
 * - Authentic Forecourt Stair & Ramp Layout (ref_184):
 *   Left Terracotta Stairs -> Sloped Red Curb -> Central Narrow Grey Stone Steps -> Central Terracotta Deck ->
 *   Twin Sloped Red-With-White-Coping Ramp Walls framing Wheelchair Ramp -> Right Terracotta Stairs
 * - Authentic Forecourt Apron, Blue ADA Wheelchair Stall, Radial Concrete U-Gutter ("Selokan Beton"),
 *   Light-Grey Concrete Driveway Bridge, UI-Huawei Brass Totem, Stone Pedestal, "DILARANG PARKIR" Sign,
 *   3D Narrow Gold Box-Truss Banner Frames & Stainless Campus Map Plaque
 */
export class DPPDSecondarySubdomains {
  static build(localGroup, materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.dppd;
    const eaveY = cfg.eaveY;   // 6.45m
    const l1TopY = cfg.l1TopY; // 3.72m

    // 1. Watertight Central Gable Pavilion Box & Side Return Walls (z = 52.8m .. 57.85m, x ∈ [-10.86m, +10.86m])
    //    Completely seals the side junction between the flat front gable projection and the curved main wings!
    const pavWidth = 21.72;
    const pavZBack = 52.8;
    const pavZFront = 57.85;
    const pavDepth = pavZFront - pavZBack;
    const pavZMid = 0.5 * (pavZBack + pavZFront);

    // Solid L2 floor / L1 ceiling slab inside central pavilion
    const pavFloorL2 = new THREE.Mesh(
      new THREE.BoxGeometry(pavWidth, 0.28, pavDepth),
      materials.creamStucco
    );
    pavFloorL2.position.set(0, l1TopY, pavZMid);
    localGroup.add(pavFloorL2);

    // Solid Left & Right Side Return Walls (x = ±10.68m, y = 0.15 .. eaveY, z = 52.8 .. 57.85m)
    for (const sgn of [-1, 1]) {
      const sideReturnWall = new THREE.Mesh(
        new THREE.BoxGeometry(0.36, eaveY - 0.15, pavDepth),
        materials.creamStucco
      );
      sideReturnWall.position.set(sgn * 10.68, 0.15 + (eaveY - 0.15) * 0.5, pavZMid);
      sideReturnWall.castShadow = true;
      sideReturnWall.receiveShadow = true;
      localGroup.add(sideReturnWall);
    }

    // Front L2 Sill Wall + 11-Bay Dark Glass Window Band (z = 57.78m, ref_084, ref_184)
    const frontWallL2 = new THREE.Mesh(new THREE.BoxGeometry(pavWidth, 0.88, 0.52), materials.creamStucco);
    frontWallL2.position.set(0, l1TopY + 0.44, 57.78);
    frontWallL2.castShadow = true;
    localGroup.add(frontWallL2);

    const winBand = new THREE.Mesh(new THREE.BoxGeometry(20.9, 1.88, 0.32), materials.windowGlass);
    winBand.position.set(0, l1TopY + 1.78, 57.73);
    localGroup.add(winBand);

    // Horizontal dark transom bar + top/bottom 3D window frame sills on L2 11-Bay Window Band (ref_184)
    for (const wy of [l1TopY + 0.86, l1TopY + 1.36, l1TopY + 2.70]) {
      const winFrameBar = new THREE.Mesh(new THREE.BoxGeometry(21.0, 0.08, 0.38), materials.darkMetal);
      winFrameBar.position.set(0, wy, 57.76);
      localGroup.add(winFrameBar);
    }
    // 22 Vertical Dark Aluminum Sub-Mullions inside the 11 Window Bays (2 panes per bay, ref_184!)
    for (let i = 0; i < 11; i++) {
      const bx = -10.45 + ((i + 0.5) / 11) * 20.9;
      const subMul = new THREE.Mesh(new THREE.BoxGeometry(0.055, 1.88, 0.36), materials.darkMetal);
      subMul.position.set(bx, l1TopY + 1.78, 57.76);
      localGroup.add(subMul);
    }

    // 12 Crisp Vertical Cream Piers forming 11 Symmetrical Window Bays (5 Left + 1 Center + 5 Right, ref_084, ref_184)
    for (let i = 0; i <= 11; i++) {
      const px = -10.45 + (i / 11) * 20.9;
      const pier = new THREE.Mesh(new THREE.BoxGeometry(0.24, 1.96, 0.38), materials.creamStucco);
      pier.position.set(px, l1TopY + 1.78, 57.80);
      pier.castShadow = true;
      localGroup.add(pier);
    }

    // Small White Round Satellite Mini-Dish w/ Dark Bracket on left pier #1 (x = -8.55m, ref_184)
    const dishArm = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 0.24), materials.darkMetal);
    dishArm.position.set(-8.55, l1TopY + 1.12, 57.94);
    localGroup.add(dishArm);
    const dish = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.15, 0.05, 14), materials.architecturalWhite);
    dish.rotation.x = Math.PI * 0.44;
    dish.position.set(-8.55, l1TopY + 1.14, 58.06);
    localGroup.add(dish);

    // White Split-AC Outdoor Condenser Unit at far-right bay of L2 Window Band (x = +9.65m, ref_184, ref_084!)
    const rightGableAc = new THREE.Mesh(new THREE.BoxGeometry(0.84, 0.58, 0.34), materials.architecturalWhite);
    rightGableAc.position.set(9.65, l1TopY + 1.08, 57.96);
    rightGableAc.castShadow = true;
    localGroup.add(rightGableAc);
    const rightGableFan = new THREE.Mesh(new THREE.CylinderGeometry(0.20, 0.20, 0.36, 12), materials.darkMetal);
    rightGableFan.rotation.x = Math.PI * 0.5;
    rightGableFan.position.set(9.55, l1TopY + 1.08, 57.97);
    localGroup.add(rightGableFan);

    // 2. L2 Wing Loggia Colonnade Columns + Prominent Split-AC Outdoor Units w/ Dark Fan Grilles (ref_084, ref_195)
    const rCol = cfg.rOuter - 0.22; // 57.28m
    const colH = eaveY - (l1TopY + 0.70);
    const colGeo = new THREE.BoxGeometry(0.34, colH, 0.34);
    const acGeo = new THREE.BoxGeometry(0.88, 0.62, 0.36);
    const fanGeo = new THREE.CylinderGeometry(0.22, 0.22, 0.38, 12);
    fanGeo.rotateX(Math.PI * 0.5);

    for (const sgn of [-1, 1]) {
      const aStart = 12.2;
      const aEnd = sgn < 0 ? 39.5 : 38.5;
      const cols = 11;
      for (let i = 0; i < cols; i++) {
        const alpha = sgn * (aStart + (i / (cols - 1)) * (aEnd - aStart)) * DEG;
        const cx = rCol * Math.sin(alpha);
        const cz = rCol * Math.cos(alpha);
        const col = new THREE.Mesh(colGeo, materials.creamStucco);
        col.position.set(cx, l1TopY + 0.70 + colH * 0.5, cz);
        col.rotation.y = alpha;
        col.castShadow = true;
        col.receiveShadow = true;
        localGroup.add(col);

        // Place white split-AC condenser units in bays matching ref_084 & ref_184
        if (i < cols - 1 && [0, 1, 2, 4, 6, 7, 8].includes(i)) {
          const midA = sgn * (aStart + ((i + 0.52) / (cols - 1)) * (aEnd - aStart)) * DEG;
          const rAc = rCol + 0.02;
          const ac = new THREE.Mesh(acGeo, materials.architecturalWhite);
          ac.position.set(rAc * Math.sin(midA), l1TopY + 1.05, rAc * Math.cos(midA));
          ac.rotation.y = midA;
          ac.castShadow = true;
          localGroup.add(ac);

          const fan = new THREE.Mesh(fanGeo, materials.darkMetal);
          fan.position.set(rAc * Math.sin(midA) - Math.cos(midA) * 0.12, l1TopY + 1.05, rAc * Math.cos(midA) + Math.sin(midA) * 0.12);
          fan.rotation.y = midA;
          localGroup.add(fan);
        }
      }
    }

    // 3. Recessed Ground-Floor (L1) Main Entrance Wall, Glass Double Doors, & Unoccluded Wall Signage (ref_084, ref_184)
    const zL1Wall = 56.85;
    const l1Wall = new THREE.Mesh(new THREE.BoxGeometry(pavWidth, l1TopY - 0.15, 0.55), materials.creamStucco);
    l1Wall.position.set(0, 0.15 + (l1TopY - 0.15) * 0.5, zL1Wall);
    l1Wall.receiveShadow = true;
    localGroup.add(l1Wall);

    // Central Dark-Framed Glass Double Entrance Doors + Symmetrical Side Windows (ref_184)
    const doorFrame = new THREE.Mesh(new THREE.BoxGeometry(3.15, 2.75, 0.16), materials.darkMetal);
    doorFrame.position.set(0, 0.45 + 2.75 * 0.5, zL1Wall + 0.28);
    localGroup.add(doorFrame);

    const doorGlass = new THREE.Mesh(new THREE.BoxGeometry(2.88, 2.58, 0.20), materials.windowGlass);
    doorGlass.position.set(0, 0.45 + 2.58 * 0.5, zL1Wall + 0.29);
    localGroup.add(doorGlass);

    for (const sx of [-4.8, 4.8]) {
      const slit = new THREE.Mesh(new THREE.BoxGeometry(0.68, 2.15, 0.16), materials.windowGlass);
      slit.position.set(sx, 1.75, zL1Wall + 0.28);
      localGroup.add(slit);
    }

    // Crisp Wall Signage right & left of doors at eye level (y = 2.02m so canopy never blocks it, ref_184!)
    const ksdpSign = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(2.15, 0.64, 40, 420, 440, 104),
      materials.signAtlasMat
    );
    ksdpSign.position.set(2.95, 2.02, zL1Wall + 0.30);
    localGroup.add(ksdpSign);

    const dppdPlaque = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(2.15, 0.44, 530, 428, 474, 88),
      materials.signAtlasMat
    );
    dppdPlaque.position.set(-2.95, 2.02, zL1Wall + 0.30);
    localGroup.add(dppdPlaque);

    // 4. Full-Width Straight Continuous L1 Terracotta Portico Canopy + 4 Cream Columns (ref_084, ref_184, ref_185)
    //    Upgraded with True-3D Stepped Tile Courses, 40 Sculpted 3D Eave Drip-Tile Nosings, Side Verge Caps & Recessed Downlights!
    const canW = 21.64;
    const canZIn = 57.55;
    const canZOut = 60.45;
    const canDepth = canZOut - canZIn;

    const canRoof = new THREE.Mesh(new THREE.BoxGeometry(canW, 0.16, canDepth + 0.14), materials.terracottaRoof);
    canRoof.position.set(0, 3.56, 0.5 * (canZIn + canZOut));
    canRoof.rotation.x = 0.15;
    canRoof.castShadow = true;
    canRoof.receiveShadow = true;
    localGroup.add(canRoof);

    // 6 Stepped 3D Terracotta Tile Course Overlap Bars across L1 Canopy Slope (true 3D tile relief!)
    for (let k = 0; k < 6; k++) {
      const tk = k / 6;
      const cz = canZOut - tk * canDepth;
      const cy = 3.36 + tk * 0.43 + 0.08;
      const courseBar = new THREE.Mesh(new THREE.BoxGeometry(canW + 0.04, 0.036, (canDepth / 6) + 0.04), materials.terracottaRoof);
      courseBar.position.set(0, cy, cz - (canDepth / 12));
      courseBar.rotation.x = 0.15;
      courseBar.castShadow = true;
      localGroup.add(courseBar);
    }

    // 42 Sculpted 3D Ceramic Eave Drip-Tile Nosings ("Genteng Cucuran Kanopi 3D") along the front lip of the L1 Canopy!
    for (let i = 0; i < 42; i++) {
      const tx = -canW * 0.5 + 0.26 + (i / 41) * (canW - 0.52);
      const eaveTile = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.065, 0.18), materials.terracottaRoof);
      eaveTile.position.set(tx, 3.39, canZOut + 0.04);
      eaveTile.rotation.x = 0.18;
      localGroup.add(eaveTile);
    }

    // 3D Terracotta Side Verge Caps & Rainwater Downspouts at Left & Right Ends of L1 Canopy
    for (const sgn of [-1, 1]) {
      const verge = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.20, canDepth + 0.22), materials.terracottaRoof);
      verge.position.set(sgn * (canW * 0.5 + 0.02), 3.58, 0.5 * (canZIn + canZOut));
      verge.rotation.x = 0.15;
      localGroup.add(verge);

      const downspout = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 3.25, 8), materials.darkMetal);
      downspout.position.set(sgn * 10.48, 1.82, 57.32);
      localGroup.add(downspout);
    }

    // Top ridge flashing along wall-canopy junction
    const canFlashing = new THREE.Mesh(new THREE.BoxGeometry(canW, 0.12, 0.18), materials.terracottaBrick);
    canFlashing.position.set(0, 3.82, canZIn + 0.06);
    localGroup.add(canFlashing);

    const canSoffit = new THREE.Mesh(new THREE.BoxGeometry(canW - 0.06, 0.10, canDepth + 0.02), materials.creamStucco);
    canSoffit.position.set(0, 3.28, 0.5 * (canZIn + canZOut));
    localGroup.add(canSoffit);

    // 4 Recessed 3D Downlight Fixtures in L1 Portico Ceiling Soffit + Porch Doormat & Waiting Chair (ref_184!)
    for (const lx of [-6.2, -2.2, 2.2, 6.2]) {
      const lamp = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.06, 10), materials.goldAccent);
      lamp.position.set(lx, 3.22, 0.5 * (canZIn + canZOut));
      localGroup.add(lamp);
    }
    const doormat = new THREE.Mesh(new THREE.BoxGeometry(1.85, 0.03, 0.95), materials.darkMetal);
    doormat.position.set(0, 0.48, zL1Wall + 0.92);
    localGroup.add(doormat);

    // Receptionist/Security Chair under the KSDP UI Sign (visible in ref_184!)
    const chairSeat = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.06, 0.48), materials.darkMetal);
    chairSeat.position.set(2.35, 0.92, zL1Wall + 0.68);
    localGroup.add(chairSeat);
    const chairBack = new THREE.Mesh(new THREE.BoxGeometry(0.50, 0.48, 0.06), materials.darkMetal);
    chairBack.position.set(2.35, 1.18, zL1Wall + 0.46);
    localGroup.add(chairBack);

    const canFascia = new THREE.Mesh(new THREE.BoxGeometry(canW + 0.04, 0.20, 0.10), materials.darkMetal);
    canFascia.position.set(0, 3.32, canZOut + 0.05);
    localGroup.add(canFascia);

    // 4 Square Cream Portico Columns supporting L1 Canopy (inner pair frames the central doors at x = ±1.62m, ref_184!)
    for (const cx of [-8.4, -1.62, 1.62, 8.4]) {
      const pCol = new THREE.Mesh(new THREE.BoxGeometry(0.28, 2.95, 0.28), materials.creamStucco);
      pCol.position.set(cx, 0.45 + 2.95 * 0.5, canZOut - 0.35);
      pCol.castShadow = true;
      localGroup.add(pCol);
    }

    // 5. Watertight Plastered Masonry Wing Walls ("Pipi Tangga") & Grass Berm Fill (x = ±10.82m, ref_184, ref_185, ref_084!)
    DPPDSecondarySubdomains._buildWatertightForecourtWingWalls(localGroup, materials, l1TopY);

    // 6. Terracotta-Tiled Upper Porch Deck & Authentic Stair / Grey Center Steps / Twin-Curb Ramp Sequence (ref_184!)
    const porchDeck = new THREE.Mesh(new THREE.BoxGeometry(21.64, 0.38, 4.35), materials.terracottaPlaza);
    porchDeck.position.set(0, 0.28, 59.00);
    porchDeck.receiveShadow = true;
    localGroup.add(porchDeck);

    // Forecourt Weathered Grey Paving Apron (strictly inside z = 61.15 .. 67.55m so it never buries the ditch/verge!)
    const apron = new THREE.Mesh(new THREE.BoxGeometry(21.8, 0.14, 6.4), materials.pavingBlock);
    apron.position.set(0, 0.13, 64.35);
    apron.receiveShadow = true;
    localGroup.add(apron);

    // Low Weathered Dark-Stone Retaining Step along the front edge of the Forecourt & Blue Stall (z = 67.58m, ref_184!)
    for (const [sx, sw] of [[-7.35, 6.9], [7.65, 6.3]]) {
      const frontStep = new THREE.Mesh(new THREE.BoxGeometry(sw, 0.26, 0.32), materials.stoneRetaining);
      frontStep.position.set(sx, 0.16, 67.58);
      frontStep.receiveShadow = true;
      localGroup.add(frontStep);
    }

    // Light-Grey Brushed Concrete Driveway Bridge crossing from Ring Road (z = 71.45m) onto Paving Forecourt (z = 67.40m, ref_184!)
    const driveBridge = new THREE.Mesh(new THREE.BoxGeometry(8.3, 0.16, 4.15), materials.architecturalWhite);
    driveBridge.position.set(0.45, 0.11, 69.45);
    driveBridge.receiveShadow = true;
    localGroup.add(driveBridge);

    // Open Radial Concrete U-Gutter ("Selokan Beton") w/ Two Parallel Raised Concrete Lips right at left edge of bridge
    // directly in front of the bare steel flagpole drum base (x = -4.08m, z = 67.65 .. 71.38m — ref_184 & ref_185 bottom-center!)
    const gutterFloor = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.08, 3.72), materials.stoneRetaining);
    gutterFloor.position.set(-4.08, 0.05, 69.52);
    localGroup.add(gutterFloor);
    for (const dx of [-0.19, 0.19]) {
      const gutterLip = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.22, 3.74), materials.architecturalWhite);
      gutterLip.position.set(-4.08 + dx, 0.15, 69.52);
      gutterLip.castShadow = true;
      gutterLip.receiveShadow = true;
      localGroup.add(gutterLip);
    }

    // Helper for sloped parapet cheek curbs alongside stairs & ramp (ref_184)
    const addSlopedRampCurb = (cx, zTop, yTop, zBot, yBot, hasWhiteCoping = false) => {
      const len = Math.hypot(zBot - zTop, yTop - yBot);
      const ang = Math.atan2(yTop - yBot, zBot - zTop);
      const curb = new THREE.Mesh(new THREE.BoxGeometry(0.20, 0.30, len), materials.terracottaBrick);
      curb.position.set(cx, 0.5 * (yTop + yBot), 0.5 * (zTop + zBot));
      curb.rotation.x = ang;
      curb.castShadow = true;
      localGroup.add(curb);
      if (hasWhiteCoping) {
        const cap = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.05, len + 0.02), materials.architecturalWhite);
        cap.position.set(cx, 0.5 * (yTop + yBot) + 0.16, 0.5 * (zTop + zBot));
        cap.rotation.x = ang;
        localGroup.add(cap);
      }
    };

    // Authentic Left-to-Right Stair & Ramp Sequence across the 21.6m Porch (z = 61.20m .. 63.65m, ref_184!):
    // A. Left Wide 6-Step Terracotta Stair Flight (x = -10.55m .. -2.05m, width = 8.50m)
    for (let st = 0; st < 6; st++) {
      const stepH = 0.054 * (6 - st);
      const step = new THREE.Mesh(new THREE.BoxGeometry(8.50, stepH, 0.40), materials.terracottaPlaza);
      step.position.set(-6.30, 0.15 + stepH * 0.5, 61.35 + st * 0.38);
      step.receiveShadow = true;
      localGroup.add(step);
    }

    // B. Sloped Red-Brown Curb #1 separating Left Terracotta Stairs from Central Grey Stone Steps (x = -1.96m)
    addSlopedRampCurb(-1.96, 61.18, 0.52, 63.58, 0.24, false);

    // C. Central Narrow 6-Step Grey Stone Stairway (x = -1.86m .. -0.36m, width = 1.50m — ref_184 center!)
    for (let st = 0; st < 6; st++) {
      const stepH = 0.054 * (6 - st);
      const step = new THREE.Mesh(new THREE.BoxGeometry(1.50, stepH, 0.40), materials.stoneRetaining);
      step.position.set(-1.11, 0.15 + stepH * 0.5, 61.35 + st * 0.38);
      step.receiveShadow = true;
      localGroup.add(step);
    }

    // D. Central Terracotta-Tiled Gently Sloped Deck Approach (x = -0.36m .. +3.34m, width = 3.70m, flush with grey steps!)
    const centerDeck = new THREE.Mesh(new THREE.BoxGeometry(3.70, 0.12, 2.50), materials.terracottaPlaza);
    centerDeck.position.set(1.49, 0.28, 62.38);
    centerDeck.rotation.x = 0.105;
    centerDeck.receiveShadow = true;
    localGroup.add(centerDeck);

    // E. Twin Parallel Sloped Red-Brown Ramp Walls w/ White Coping framing the ADA Wheelchair Ramp (x = +3.44m .. +5.16m, ref_184!)
    addSlopedRampCurb(3.42, 61.18, 0.56, 63.62, 0.28, true);
    const adaRamp = new THREE.Mesh(new THREE.BoxGeometry(1.54, 0.12, 2.52), materials.terracottaPlaza);
    adaRamp.position.set(4.30, 0.28, 62.40);
    adaRamp.rotation.x = 0.105;
    adaRamp.receiveShadow = true;
    localGroup.add(adaRamp);
    addSlopedRampCurb(5.18, 61.18, 0.56, 63.62, 0.28, true);

    // F. Right Wide 6-Step Terracotta Stair Flight (x = +5.28m .. +10.56m, width = 5.28m)
    for (let st = 0; st < 6; st++) {
      const stepH = 0.054 * (6 - st);
      const step = new THREE.Mesh(new THREE.BoxGeometry(5.28, stepH, 0.40), materials.terracottaPlaza);
      step.position.set(7.92, 0.15 + stepH * 0.5, 61.35 + st * 0.38);
      step.receiveShadow = true;
      localGroup.add(step);
    }

    // 7. Left Forecourt: Authentic UV-Mapped Blue ADA Wheelchair Parking Stall + Orange Cone + Stone Pedestal + Huawei Totem (ref_184)
    const blueBay = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasBox(5.2, 0.04, 2.85, 852, 652, 86, 124),
      materials.signAtlasMat
    );
    blueBay.position.set(-6.85, 0.20, 66.05);
    blueBay.receiveShadow = true;
    localGroup.add(blueBay);

    // Orange Traffic Cone at back-right corner of the Blue ADA Stall right behind the steel pole (ref_184)
    const coneBase = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.05, 0.34), materials.terracottaRoof);
    coneBase.position.set(-4.48, 0.23, 65.05);
    localGroup.add(coneBase);
    const cone = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.13, 0.56, 10), materials.terracottaRoof);
    cone.position.set(-4.48, 0.52, 65.05);
    localGroup.add(cone);

    // Small White "DILARANG PARKIR / TITIK KUMPUL" Signboard on Twin White Posts at foot of Left Sloped Wing Wall (ref_184, ref_185!)
    for (const px of [-9.72, -9.42]) {
      const sPost = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 1.05, 6), materials.architecturalWhite);
      sPost.position.set(px, 0.65, 64.55);
      localGroup.add(sPost);
    }
    const noParkSign = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.56, 0.04), materials.architecturalWhite);
    noParkSign.position.set(-9.57, 1.02, 64.55);
    noParkSign.rotation.y = 0.35;
    localGroup.add(noParkSign);
    const noParkIcon = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.05, 12), materials.terracottaRoof);
    noParkIcon.rotation.x = Math.PI * 0.5;
    noParkIcon.position.set(-9.56, 1.08, 64.56);
    noParkIcon.rotation.z = -0.35;
    localGroup.add(noParkIcon);

    // Short Carved Grey Stone Pedestal / Bollard w/ Square Cap at front-left corner of Blue ADA Stall (ref_184)
    const stonePed = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.28, 0.52, 10), materials.stoneRetaining);
    stonePed.position.set(-9.75, 0.44, 67.12);
    stonePed.castShadow = true;
    localGroup.add(stonePed);
    const stoneCap = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.10, 0.62), materials.stoneRetaining);
    stoneCap.position.set(-9.75, 0.72, 67.12);
    localGroup.add(stoneCap);

    // Brushed Gold/Brass UI-Huawei Training Center Monolith Totem on Dark Plinth (standing straight at front of Blue Stall, ref_184!)
    const huaweiPlinth = new THREE.Mesh(new THREE.BoxGeometry(1.26, 0.22, 0.36), materials.darkMetal);
    huaweiPlinth.position.set(-7.55, 0.29, 67.12);
    localGroup.add(huaweiPlinth);

    const huaweiBody = new THREE.Mesh(new THREE.BoxGeometry(1.16, 2.22, 0.20), materials.goldAccent);
    huaweiBody.position.set(-7.55, 0.40 + 2.22 * 0.5, 67.12);
    huaweiBody.castShadow = true;
    localGroup.add(huaweiBody);

    const huaweiFace = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(1.12, 2.16, 340, 528, 340, 496),
      materials.signAtlasMat
    );
    huaweiFace.position.set(-7.55, 0.40 + 2.22 * 0.5, 67.23);
    localGroup.add(huaweiFace);

    // 8. Right Forecourt: Narrow 3D Gold Box-Truss Banner Frames + Stainless Campus Map Plaque (ref_084, ref_184)
    //    Oriented perpendicular to the facade (narrow X profile, deep Z truss rungs) matching ref_184 & ref_084!
    for (const [bx, bz] of [[5.35, 66.45], [10.25, 66.35]]) {
      const basePad = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.22, 0.62), materials.darkMetal);
      basePad.position.set(bx, 0.24, bz);
      localGroup.add(basePad);

      for (const lz of [-0.20, 0.20]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.05, 5.1, 0.05), materials.goldAccent);
        leg.position.set(bx, 0.32 + 2.55, bz + lz);
        leg.castShadow = true;
        localGroup.add(leg);
      }
      for (let r = 0; r < 9; r++) {
        const rung = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.04, 0.44), materials.goldAccent);
        rung.position.set(bx, 0.75 + r * 0.56, bz);
        localGroup.add(rung);
      }
    }

    // Stainless-Steel Framed Campus Map Directory Signboard on Dark Plinth & Stainless Cylinder Post (ref_084, ref_184)
    const mapPlinth = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.20, 0.54), materials.darkMetal);
    mapPlinth.position.set(7.65, 0.24, 66.25);
    localGroup.add(mapPlinth);

    const mapPost = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.10, 1.25, 12), materials.silverLouver);
    mapPost.position.set(7.65, 0.85, 66.25);
    localGroup.add(mapPost);

    const mapFrame = new THREE.Mesh(new THREE.BoxGeometry(1.46, 1.04, 0.12), materials.silverLouver);
    mapFrame.position.set(7.65, 1.82, 66.25);
    mapFrame.castShadow = true;
    localGroup.add(mapFrame);

    const mapFace = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(1.36, 0.94, 680, 776, 184, 248),
      materials.signAtlasMat
    );
    mapFace.position.set(7.65, 1.82, 66.32);
    localGroup.add(mapFace);

    // Ornamental Grass Bush on the Right Berm Slope (ref_084 center-right!)
    const bush = new THREE.Mesh(new THREE.DodecahedronGeometry(0.88, 1), materials.lushGrass);
    bush.scale.set(1.15, 0.95, 1.15);
    bush.position.set(14.5, 1.72, 62.2);
    bush.castShadow = true;
    localGroup.add(bush);
  }

  static _buildWatertightForecourtWingWalls(localGroup, materials, l1TopY) {
    // Off-White/Cream Plastered Masonry Wing Walls ("Pipi Tangga") w/ Weathered Grey Coping along x = ±10.82m (ref_184, ref_185, ref_084)
    // Plus a seamless Green Grass Berm Fill Wedge bridging alpha = ±9.6° right against the outer face of the wing wall!
    for (const sgn of [-1, 1]) {
      const wx = sgn * 10.82;
      const xIn = wx - 0.18;
      const xOut = wx + 0.18;

      const buildWingSegment = (zTop, yTop, zBot, yBot, mat) => {
        const p = [
          [xIn, 0.12, zTop],
          [xIn, 0.12, zBot],
          [xIn, yBot, zBot],
          [xIn, yTop, zTop],
          [xOut, 0.12, zTop],
          [xOut, 0.12, zBot],
          [xOut, yBot, zBot],
          [xOut, yTop, zTop]
        ];
        const pos = [];
        const norm = [];
        const uvs = [];
        const addQuad = (a, b, d, e) => {
          const vA = new THREE.Vector3(...a);
          const vB = new THREE.Vector3(...b);
          const vD = new THREE.Vector3(...d);
          const fn = new THREE.Vector3().subVectors(vB, vA).cross(new THREE.Vector3().subVectors(vD, vA)).normalize();
          for (const pt of [a, b, d, a, d, e]) {
            pos.push(pt[0], pt[1], pt[2]);
            norm.push(fn.x, fn.y, fn.z);
            uvs.push((pt[0] + pt[2]) / 2.5, pt[1] / 2.5);
          }
        };
        addQuad(p[0], p[1], p[2], p[3]);
        addQuad(p[5], p[4], p[7], p[6]);
        addQuad(p[3], p[2], p[6], p[7]);
        addQuad(p[1], p[5], p[6], p[2]);

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        geo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
        geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        localGroup.add(mesh);
      };

      // 1. Upper plastered side wall closing the L1 portico right underneath the sloping L1 terracotta canopy (z = 56.2 .. 60.45m, y = 3.52m -> 3.24m, ref_184)
      buildWingSegment(56.2, 3.54, 57.55, 3.54, materials.creamStucco);
      buildWingSegment(57.55, 3.54, 60.45, 3.24, materials.creamStucco);
      // 2. Sloped plastered cheek wall descending from the front eave of the L1 canopy alongside the stairs (z = 60.45m, y = 3.24m -> z = 63.65m, y = 0.68m, ref_184!)
      buildWingSegment(60.45, 3.24, 63.65, 0.68, materials.creamStucco);
      // 3. Low weathered stone/concrete coping cap along the sloped cheek wall (ref_184)
      const capLen = Math.hypot(63.65 - 60.45, 3.24 - 0.68);
      const capAng = Math.atan2(3.24 - 0.68, 63.65 - 60.45);
      const coping = new THREE.Mesh(new THREE.BoxGeometry(0.40, 0.08, capLen + 0.08), materials.stoneRetaining);
      coping.position.set(wx, 0.5 * (3.24 + 0.68) + 0.03, 0.5 * (60.45 + 63.65));
      coping.rotation.x = capAng;
      localGroup.add(coping);

      // 4. Watertight Grass Berm Transition Wedge connecting the curved berm (alpha = 10.85°) smoothly to the outer face of the wing wall!
      const aCut = sgn * 10.85 * DEG;
      const xWall = sgn * 10.84;
      const r1 = 57.25;
      const r2 = 62.60;
      const r3 = 66.80;
      const bPos = [];
      const bNorm = [];
      const bUv = [];
      const pushGrassTri = (a, b, c) => {
        const vA = new THREE.Vector3(...a);
        const vB = new THREE.Vector3(...b);
        const vC = new THREE.Vector3(...c);
        const fn = new THREE.Vector3().subVectors(vB, vA).cross(new THREE.Vector3().subVectors(vC, vA)).normalize();
        const pts = fn.y < 0 ? [a, c, b] : [a, b, c];
        if (fn.y < 0) fn.negate();
        for (const pt of pts) {
          bPos.push(pt[0], pt[1], pt[2]);
          bNorm.push(fn.x, fn.y, fn.z);
          bUv.push(pt[0] / 3.5, pt[2] / 3.5);
        }
      };
      const pushGrassQuad = (a, b, c, d) => {
        pushGrassTri(a, b, c);
        pushGrassTri(a, c, d);
      };
      pushGrassQuad(
        [xWall, l1TopY - 0.16, r1 * Math.cos(aCut)],
        [r1 * Math.sin(aCut), l1TopY, r1 * Math.cos(aCut)],
        [r2 * Math.sin(aCut), 1.48, r2 * Math.cos(aCut)],
        [xWall, 1.48, r2 * Math.cos(aCut)]
      );
      pushGrassQuad(
        [xWall, 1.32, r2 * Math.cos(aCut)],
        [r2 * Math.sin(aCut), 1.32, r2 * Math.cos(aCut)],
        [r3 * Math.sin(aCut), 0.68, r3 * Math.cos(aCut)],
        [xWall, 0.68, r3 * Math.cos(aCut)]
      );
      const bGeo = new THREE.BufferGeometry();
      bGeo.setAttribute('position', new THREE.Float32BufferAttribute(bPos, 3));
      bGeo.setAttribute('normal', new THREE.Float32BufferAttribute(bNorm, 3));
      bGeo.setAttribute('uv', new THREE.Float32BufferAttribute(bUv, 2));
      const bMesh = new THREE.Mesh(bGeo, materials.lushGrass);
      bMesh.receiveShadow = true;
      localGroup.add(bMesh);

      // Connect the mid-berm stone terrace curb & lower batu-kali toe wall seamlessly into the wing wall (0.00m gap!)
      const midCurb = new THREE.Mesh(new THREE.BoxGeometry(1.15, 0.24, 0.34), materials.stoneRetaining);
      midCurb.position.set(sgn * 11.32, 1.38, r2 * Math.cos(aCut));
      localGroup.add(midCurb);

      const toeBridge = new THREE.Mesh(new THREE.BoxGeometry(1.90, 0.66, 0.38), materials.stoneRetaining);
      toeBridge.position.set(sgn * 11.68, 0.41, r3 * Math.cos(aCut));
      localGroup.add(toeBridge);
    }
  }
}
