import * as THREE from 'three';
import { NexusSSoSTKernel } from '../../core/MasterSurfaceKernels.js';
import { ProceduralTextureFactory } from '../../core/ProceduralTextureFactory.js';
import { UI_DPPD_MASTER_PLAN } from '../../world-gis/MasterPlanConfig.js';

const DEG = Math.PI / 180;

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./library/ilrc-drpm/index.js
 *
 * Class-A Procedural Building Module: Gedung ILRC & DRPM UI (Integrated Laboratory and Research Center)
 * Ground-Truth References: ref_002-084, ref_107-126, ref_154-155, ref_174, ref_206.
 *
 * Unified 4-Zone Manifold Crescent Architecture (th = -31° ENE to +83° SSE):
 * 1. East Curved Horizontal-Louvered Wing (-31°..+36°): Tall L1 white base, East drop-off portico,
 *    L4-L5 cantilevered 3-sided glass observation box (+8.5°), DRPM L1 open breezeway (+27°..+35°),
 *    "P PARKIR KHUSUS PIMPINAN" open iron gates, & grey Gardu utility substation at -32°.
 * 2. 5-Story Open-Air SE Atrium & Main Portal (+36°..+46°): Smooth burnt-terracotta flank walls (L2-L5),
 *    3 white corridor bridges (L3-L5), continuous curved white roof cornice w/ warm taupe-terracotta soffit,
 *    L1 white portal ("INTEGRATED LABORATORY AND RESEARCH CENTER / UNIVERSITAS INDONESIA" + Makara + KAN),
 *    and axial brown Koral-Sikat pebble-grid ramp w/ open black picket gates & blue ADA bay.
 * 3. South Curved Wing & Bamboo-Staked Dry-Leaf Mound (+46°..+75°): L1 white piers + Makara banners,
 *    L2 dark-mullioned glass curtain ribbon, L3-L5 horizontal silver louvers, & 1.65m brown dry-leaf mound
 *    retained by vertical bamboo stakes along the South ring road (zero magenta lens-flare bias!).
 * 4. 6-Story Terracotta / Burnt-Brick South End Tower (+75°..+83°, H = 25.2m + 27.2m Penthouse):
 *    Angled cantilevered SE sunshade slab fins (ref_084) + 3 stacked protruding hooded window boxes (ref_076).
 */
export class GedungILRCBuilder {
  static build(materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.ilrc;
    const group = new THREE.Group();
    group.name = 'Gedung_ILRC_DRPM_UI';

    const rOut = cfg.rOuter;          // 58.0m
    const rInE = cfg.rInnerEast;      // 37.5m (wider northern/eastern ILRC wing, ref_174)
    const rInS = cfg.rInnerSouth;     // 41.5m (stepped-in southeastern DRPM/South wing, ref_174)
    const corniceY = cfg.corniceTopY; // 21.20m

    const thStart = cfg.thStart;           // -31°
    const thEastEnd = cfg.thEastEnd;       // +36°
    const thAtriumEnd = cfg.thAtriumEnd;   // +46°
    const thSouthEnd = cfg.thSouthEnd;     // +75°
    const thTowerEnd = cfg.thTowerEnd;     // +83°

    // =========================================================================
    // 1. CONTINUOUS CURVED WHITE ROOF CORNICE BRIDGING ACROSS THE 5-STORY ATRIUM
    //    (Sweeps unbroken from -31° to +75°, with warm taupe-terracotta soffit over Atrium! ref_029, ref_030)
    // =========================================================================
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thStart, thSouthEnd, rOut - 1.2, rOut + 0.85, corniceY - 0.82, corniceY, 56,
      { uvMetersPerTile: 3.8 }
    );
    // Warm pinkish-taupe-terracotta underside soffit bridging across the Open Atrium (+36°..+46°, ref_029)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.warmTaupeSoffit, thEastEnd, thAtriumEnd, rInS + 6.5, rOut + 0.65, corniceY - 0.92, corniceY - 0.78, 10,
      { top: false, bottom: true, uvMetersPerTile: 3.5 }
    );

    // Flat Roof Decks on East Wing (-31°..+36°) & South Wing (+46°..+75°)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.pavingBlock, thStart, thEastEnd, rInE, rOut - 1.0, corniceY - 0.65, corniceY - 0.15, 32,
      { bottom: false, uvMetersPerTile: 5.0 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.pavingBlock, thAtriumEnd, thSouthEnd, rInS, rOut - 1.0, corniceY - 0.65, corniceY - 0.15, 18,
      { bottom: false, uvMetersPerTile: 5.0 }
    );

    // =========================================================================
    // 2. ZONE 1: EAST CURVED HORIZONTAL-LOUVERED WING & DRPM BREEZEWAY (-31° .. +36°)
    // =========================================================================
    // Inner courtyard wall & windows (East wing)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thStart, thEastEnd, rInE, rInE + 1.2, 0.15, corniceY - 0.6, 30,
      { bottom: false, uvMetersPerTile: 4.0 }
    );

    // L1 White Ground Floor (-31° .. +26.5° solid/lobby, +26.5° .. +36° open DRPM breezeway! ref_012, ref_018)
    const thBreezeway0 = 26.5 * DEG;
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thStart, thBreezeway0, rInE + 1.0, rOut - 0.75, 0.15, 5.15, 26,
      { bottom: false, uvMetersPerTile: 3.8 }
    );
    // 8-Pane Dark Ribbon Window north of DRPM breezeway (+16°..+25°, ref_012, ref_018)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.windowGlass, 16.0 * DEG, 25.2 * DEG, rOut - 0.82, rOut - 0.62, 1.35, 3.35, 8,
      { top: false, bottom: false, uvMetersPerTile: 2.2 }
    );
    // Tall Cylindrical White Columns at East Lobby (+6.5°) & DRPM Breezeway (+27.2°, +35.2°, ref_002, ref_012)
    for (const colDeg of [6.2, 11.2, 27.2, 31.2, 35.2]) {
      const cr = colDeg * DEG;
      const col = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 5.05, 14), materials.architecturalWhite);
      col.position.set((rOut - 0.65) * Math.cos(cr), 0.15 + 2.525, (rOut - 0.65) * Math.sin(cr));
      col.castShadow = true;
      group.add(col);
    }

    // East Vehicular Drop-Off Portico Canopy (+5.2° .. +11.8°, projecting to r = 63.4m, ref_002, ref_009)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, 5.2 * DEG, 11.8 * DEG, rOut - 0.7, 63.4, 4.45, 4.92, 6,
      { uvMetersPerTile: 3.5 }
    );
    for (const pDeg of [5.8, 11.2]) {
      const pr = pDeg * DEG;
      const pCol = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 4.45, 12), materials.architecturalWhite);
      pCol.position.set(62.8 * Math.cos(pr), 2.30, 62.8 * Math.sin(pr));
      pCol.castShadow = true;
      group.add(pCol);
    }

    // L2, L3, & Double-Height L4-L5 Horizontal Silver Aluminum Sun-Louvers (-31° .. +36°, ref_009, ref_019, ref_111)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.silverLouver, thStart + 0.01, thEastEnd - 0.01, rOut - 0.55, rOut - 0.08, 5.15, corniceY - 0.80, 34,
      { bottom: false, top: false, uvMetersPerTile: 2.8 }
    );
    // True-3D Projecting Horizontal Aluminum Sun-Louver Blades across East Wing L2-L5 (physical 0.22m depth!)
    for (let ly = 5.85; ly <= corniceY - 1.35; ly += 1.05) {
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.silverLouver, thStart + 0.01, thEastEnd - 0.01, rOut - 0.10, rOut + 0.14, ly - 0.035, ly + 0.035, 34,
        { bottom: true, top: true, uvMetersPerTile: 3.0 }
      );
    }
    // Projecting Curved White Concrete Floor-Slab Fascias (L2 base 5.15m full-depth ceiling soffit, L3 9.35m, L4 13.55m)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thStart, thEastEnd, rInE, rOut + 0.45, 5.15 - 0.20, 5.15 + 0.20, 36,
      { bottom: true, top: true, uvMetersPerTile: 3.8 }
    );
    for (const ySlab of [9.35, 13.55]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, thStart, thEastEnd, rOut - 0.8, rOut + 0.45, ySlab - 0.20, ySlab + 0.20, 36,
        { uvMetersPerTile: 3.8 }
      );
    }
    // Vertical White Structural Mullions dividing East Louvers into clean bays
    for (let deg = -28; deg <= 34; deg += 4.5) {
      const mr = deg * DEG;
      const mul = new THREE.Mesh(new THREE.BoxGeometry(0.16, corniceY - 5.95, 0.34), materials.architecturalWhite);
      mul.position.set((rOut + 0.05) * Math.cos(mr), 5.15 + (corniceY - 5.95) * 0.5, (rOut + 0.05) * Math.sin(mr));
      mul.rotation.y = -mr;
      group.add(mul);
    }

    // Cantilevered 3-Sided White-Framed Upper Glass Observation Box at L4–L5 (+6.2°..+10.8°, ref_009, ref_010)
    GedungILRCBuilder._buildCantileveredGlassBox(group, materials, 8.5 * DEG, rOut);

    // East Wing Roundabout Traffic Sign & Twin-Arm V-Post Garden Lamp (+9.2°..+11.5°, ref_009)
    {
      const sr = 11.5 * DEG;
      const signPost = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.05, 2.35, 8), materials.darkMetal);
      signPost.position.set(68.8 * Math.cos(sr), 1.25, 68.8 * Math.sin(sr));
      group.add(signPost);

      const signBoard = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasBox(0.72, 0.72, 0.06, 938, 716, 86, 60),
        materials.signAtlasMat
      );
      signBoard.position.set(68.8 * Math.cos(sr), 2.28, 68.8 * Math.sin(sr));
      signBoard.rotation.y = -sr + Math.PI * 0.35;
      group.add(signBoard);

      // Twin-arm V-post garden lamp near East Portico lawn (+9.2°, r = 64.2m, ref_009)
      const lr = 9.2 * DEG;
      const lx = 64.2 * Math.cos(lr);
      const lz = 64.2 * Math.sin(lr);
      const vPost = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 3.4, 8), materials.darkMetal);
      vPost.position.set(lx, 1.8, lz);
      group.add(vPost);
      for (const sgn of [-1, 1]) {
        const vArm = new THREE.Mesh(new THREE.BoxGeometry(0.85, 0.06, 0.06), materials.darkMetal);
        vArm.position.set(lx - Math.sin(lr) * sgn * 0.36, 3.65, lz + Math.cos(lr) * sgn * 0.36);
        vArm.rotation.y = -lr;
        vArm.rotation.z = sgn * 0.42;
        group.add(vArm);
        const vHead = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.10, 0.22), materials.architecturalWhite);
        vHead.position.set(lx - Math.sin(lr) * sgn * 0.72, 3.82, lz + Math.cos(lr) * sgn * 0.72);
        group.add(vHead);
      }
    }

    // Grey Masonry Gardu Utility Box at Mouth of STP–ILRC Seismic Gap (-32.4°, r = 60.4m, ref_116)
    {
      const gr = -32.4 * DEG;
      const gardu = new THREE.Mesh(new THREE.BoxGeometry(3.2, 2.85, 2.4), materials.pavingBlock);
      gardu.position.set(60.4 * Math.cos(gr), 1.50, 60.4 * Math.sin(gr));
      gardu.rotation.y = -gr;
      gardu.castShadow = true;
      group.add(gardu);
    }

    // SE DRPM Vehicular Gate + "P PARKIR KHUSUS PIMPINAN" Sign (+26.5°..+32.5°, ref_012, ref_015)
    GedungILRCBuilder._buildOpenPicketGate(group, materials, 29.5 * DEG, 67.5, 5.6, true);

    // =========================================================================
    // 3. ZONE 2: 5-STORY OPEN-AIR SE ATRIUM & MAIN ILRC PORTAL (+36° .. +46°, ref_023-030)
    // =========================================================================
    // Two Smooth Burnt-Terracotta / Red-Brick Flank Walls (NE Flank at +36°..+37.2°, SW Flank at +44.8°..+46°)
    for (const [f0, f1] of [[thEastEnd - 0.4 * DEG, thEastEnd + 1.1 * DEG], [thAtriumEnd - 1.1 * DEG, thAtriumEnd + 0.4 * DEG]]) {
      // White L1 supporting columns under terracotta flank walls (ref_024, ref_032)
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, f0, f1, rInS + 2.0, rOut - 0.3, 0.15, 4.85, 2,
        { bottom: false, uvMetersPerTile: 3.5 }
      );
      // Burnt-terracotta flank walls from L2 up to the curved roof soffit (y = 4.85 .. 20.45m)
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.terracottaBrick, f0, f1, rInS + 2.0, rOut - 0.15, 4.85, corniceY - 0.75, 2,
        { bottom: false, uvMetersPerTile: 3.0 }
      );
    }

    // Recessed Curved Inner Atrium Rear Gallery Wall (L2–L5 above the L1 ground portal opening, ref_024, ref_029)
    // Prevents seeing ITTC's red roof through the upper corridor bridges!
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thEastEnd - 0.3 * DEG, thAtriumEnd + 0.3 * DEG, rInS + 1.8, rInS + 6.8, 4.85, corniceY - 0.75, 10,
      { bottom: true, uvMetersPerTile: 3.8 }
    );
    for (const gy of [6.15, 10.15, 14.15, 18.05]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.windowGlass, thEastEnd + 0.4 * DEG, thAtriumEnd - 0.4 * DEG, rInS + 6.6, rInS + 7.05, gy, gy + 2.25, 8,
        { top: false, bottom: false, uvMetersPerTile: 2.5 }
      );
    }
    // Atrium ground floor paved lobby slab (r = rInS .. rOut, th = 36°..46°)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.pavingBlock, thEastEnd - 0.5 * DEG, thAtriumEnd + 0.5 * DEG, rInS, rOut - 0.2, 0.12, 0.22, 8,
      { bottom: false, uvMetersPerTile: 3.0 }
    );

    // L1 Main White Horizontal Portal Beam + Exact ILRC Signage (Makara + KAN + 2-Line Title, ref_024, ref_029)
    const thAtriumMid = 0.5 * (thEastEnd + thAtriumEnd); // +41.0°
    {
      const ca = Math.cos(thAtriumMid), sa = Math.sin(thAtriumMid);
      const portalGroup = new THREE.Group();
      portalGroup.position.set((rOut - 0.85) * ca, 0, (rOut - 0.85) * sa);
      portalGroup.rotation.y = Math.atan2(ca, sa);
      group.add(portalGroup);

      const spanW = 8.4;
      // Left & right white portal piers
      for (const px of [-3.85, 3.85]) {
        const pier = new THREE.Mesh(new THREE.BoxGeometry(0.72, 4.95, 0.95), materials.architecturalWhite);
        pier.position.set(px, 0.15 + 4.95 * 0.5, 0);
        pier.castShadow = true;
        portalGroup.add(pier);
      }
      // White portal lintel fascia beam (y = 3.75 .. 5.05m)
      const beam = new THREE.Mesh(new THREE.BoxGeometry(spanW, 1.30, 1.05), materials.architecturalWhite);
      beam.position.set(0, 4.40, 0);
      beam.castShadow = true;
      portalGroup.add(beam);

      // Crisp ILRC Portal Signage Plane (Tile 2 in Unified Signage Atlas!)
      const ilrcSign = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasSignPlane(spanW - 0.4, 1.18, 0, 272, 1024, 144),
        materials.signAtlasMat
      );
      ilrcSign.position.set(0, 4.40, 0.54);
      portalGroup.add(ilrcSign);

      // 3 Open-Air White Horizontal Corridor Bridges (L3, L4, L5) w/ 3D Tubular Handrails & Vertical Balusters (ref_024, ref_029)
      for (const by of [9.35, 13.35, 17.35]) {
        const brParapet = new THREE.Mesh(new THREE.BoxGeometry(spanW + 0.2, 1.12, 2.2), materials.architecturalWhite);
        brParapet.position.set(0, by, -2.1);
        brParapet.castShadow = true;
        brParapet.receiveShadow = true;
        portalGroup.add(brParapet);

        for (const ry of [by + 0.84, by + 0.68]) {
          const rail = new THREE.Mesh(new THREE.BoxGeometry(spanW + 0.1, 0.055, 0.055), materials.darkMetal);
          rail.position.set(0, ry, -1.08);
          portalGroup.add(rail);
        }
        for (let bx = -3.6; bx <= 3.6; bx += 1.2) {
          const post = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.36, 0.05), materials.darkMetal);
          post.position.set(bx, by + 0.70, -1.08);
          portalGroup.add(post);
        }
      }

      // Axial Brown Koral-Sikat Pebble-Grid Approach Ramp + Blue ADA Bay + Bollards (z = 0.5 .. 11.2m -> r = 57.5..68.4m)
      const koralRamp = new THREE.Mesh(new THREE.BoxGeometry(6.2, 0.16, 11.2), materials.koralSikat);
      koralRamp.position.set(0, 0.16, 6.0);
      koralRamp.receiveShadow = true;
      portalGroup.add(koralRamp);

      // Sky-Blue ADA Wheelchair Parking Stall w/ White Diagonal Hatching (Tile 6C, ref_029)
      const adaBay = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasBox(3.4, 0.05, 2.6, 852, 652, 86, 124),
        materials.signAtlasMat
      );
      adaBay.position.set(0, 0.25, 2.8);
      adaBay.receiveShadow = true;
      portalGroup.add(adaBay);

      for (const bx of [-3.25, 3.25]) {
        const curb = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.28, 11.2), materials.architecturalWhite);
        curb.position.set(bx, 0.22, 6.0);
        portalGroup.add(curb);

        const bollard = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.85, 8), materials.architecturalWhite);
        bollard.position.set(bx * 0.88, 0.55, 3.2);
        portalGroup.add(bollard);

        // Low black iron picket railing along both sides of the Koral-Sikat ramp bridge (ref_024, ref_029)
        const sideRailTop = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 7.6), materials.darkMetal);
        sideRailTop.position.set(bx, 0.92, 6.4);
        portalGroup.add(sideRailTop);
        const sideRailBot = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 7.6), materials.darkMetal);
        sideRailBot.position.set(bx, 0.42, 6.4);
        portalGroup.add(sideRailBot);
        for (let rz = 2.8; rz <= 10.0; rz += 0.9) {
          const rPost = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.68, 0.04), materials.darkMetal);
          rPost.position.set(bx, 0.62, rz);
          portalGroup.add(rPost);
        }
      }
    }
    // Open Black Picket Iron Gate at Entrance of Koral-Sikat Ramp (ref_023-025)
    GedungILRCBuilder._buildOpenPicketGate(group, materials, thAtriumMid, 67.4, 5.8, false);

    // =========================================================================
    // 4. ZONE 3: SOUTH CURVED WING & BAMBOO-STAKED DRY-LEAF MOUND (+46° .. +75°)
    // =========================================================================
    // L1 White Ground Base + Square Piers + 2 Navy/Gold UI Makara Vertical Banners (ref_033, ref_053)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thAtriumEnd, thSouthEnd, rInS, rOut - 1.1, 0.15, 4.95, 20,
      { bottom: false, uvMetersPerTile: 3.8 }
    );
    for (let deg = 48; deg <= 73; deg += 5.0) {
      const pr = deg * DEG;
      const pier = new THREE.Mesh(new THREE.BoxGeometry(0.45, 4.85, 0.55), materials.architecturalWhite);
      pier.position.set((rOut - 0.65) * Math.cos(pr), 0.15 + 2.425, (rOut - 0.65) * Math.sin(pr));
      pier.rotation.y = -pr;
      pier.castShadow = true;
      group.add(pier);
    }
    for (const banDeg of [53.5, 61.5]) {
      const br = banDeg * DEG;
      const ban = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasSignPlane(0.95, 2.15, 864, 776, 96, 248),
        materials.signAtlasMat
      );
      NexusSSoSTKernel.placeOnPolarCylinder(ban, rOut - 0.95, br, 2.55, 0.05, true);
      group.add(ban);
    }

    // L2 Full-Height Dark-Mullioned Glass Curtain Band (+46° .. +75°, y = 4.95 .. 9.25m, ref_033, ref_053)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.windowGlass, thAtriumEnd, thSouthEnd, rInS + 0.5, rOut - 0.35, 4.95, 9.25, 20,
      { bottom: false, uvMetersPerTile: 3.0 }
    );

    // L3–L5 Horizontal Silver Louvers + 3D Projecting Louver Blades + Curved White Floor Bands (y = 9.25 .. 20.45m, ref_028, ref_049)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.silverLouver, thAtriumEnd, thSouthEnd, rInS + 0.5, rOut - 0.15, 9.25, corniceY - 0.80, 20,
      { bottom: false, top: false, uvMetersPerTile: 2.8 }
    );
    for (let ly = 9.95; ly <= corniceY - 1.35; ly += 1.05) {
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.silverLouver, thAtriumEnd + 0.01, thSouthEnd - 0.01, rOut - 0.16, rOut + 0.12, ly - 0.035, ly + 0.035, 20,
        { bottom: true, top: true, uvMetersPerTile: 3.0 }
      );
    }
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thAtriumEnd, thSouthEnd, rInS, rOut + 0.42, 4.95 - 0.18, 4.95 + 0.18, 22,
      { bottom: true, top: true, uvMetersPerTile: 3.8 }
    );
    for (const ySlab of [9.25, 13.45, 17.05]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, thAtriumEnd, thSouthEnd, rOut - 0.8, rOut + 0.42, ySlab - 0.18, ySlab + 0.18, 22,
        { uvMetersPerTile: 3.8 }
      );
    }

    // South Dry-Brown Leaf-Litter Mound + Vertical Bamboo Stake Palisade (+45.5°..+76°, r = 61.2..68.15m, ref_028, ref_053)
    {
      const m0 = 45.5 * DEG;
      const m1 = 76.0 * DEG;
      const rCrest = 64.5;
      const rToe = 68.15;
      // Front slope rising from bamboo palisade (y = 0.45m) up to mound crest (y = 1.68m)
      const frontMound = new THREE.Mesh(
        NexusSSoSTKernel.createPolarSlopedRibbon(m0, m1, rCrest, 1.68, rToe, 0.45, 24, 3.0),
        materials.dryLeafSoil
      );
      frontMound.receiveShadow = true;
      group.add(frontMound);

      // Back slope dropping gently from crest (y = 1.68m) to L1 base (r = 60.5m, y = 0.25m)
      const backMound = new THREE.Mesh(
        NexusSSoSTKernel.createPolarSlopedRibbon(m0, m1, 60.5, 0.25, rCrest, 1.68, 24, 3.0),
        materials.dryLeafSoil
      );
      backMound.receiveShadow = true;
      group.add(backMound);

      // Tightly packed Bamboo-Stake Retaining Palisade along r = 68.15m (y = 0.10 .. 0.56m, ref_024-028, ref_053)
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.bambooWood, m0, m1, rToe - 0.14, rToe + 0.04, 0.10, 0.54, 36,
        { bottom: false, uvMetersPerTile: 1.2 }
      );
      for (let deg = 45.8; deg <= 75.8; deg += 0.9) {
        const sr = deg * DEG;
        const stake = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.05, 0.52, 5), materials.bambooWood);
        stake.position.set((rToe + 0.03) * Math.cos(sr), 0.34, (rToe + 0.03) * Math.sin(sr));
        group.add(stake);
      }
    }

    // =========================================================================
    // 5. ZONE 4: 6-STORY TERRACOTTA / BURNT-BRICK SOUTH END TOWER (+75° .. +83°, H = 25.2m + 27.2m Penthouse)
    //    Includes Angled SE Sunshade Slab Fins (ref_084) & 3 Stacked Protruding Hooded Windows (ref_076-077)!
    // =========================================================================
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.terracottaBrick, thSouthEnd, thTowerEnd, rInS + 1.0, rOut + 0.25, 0.15, cfg.towerTopY, 8,
      { bottom: false, uvMetersPerTile: 3.2 }
    );
    // Stepped Elevator Overrun Penthouse Bulkhead on top (reaching y = 27.20m — matching global height envelope! ref_077, ref_084)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.terracottaBrick, thSouthEnd + 1.8 * DEG, thTowerEnd - 1.2 * DEG, rInS + 4.5, rOut - 3.2, cfg.towerTopY - 0.1, cfg.penthouseTopY, 4,
      { bottom: false, uvMetersPerTile: 3.0 }
    );
    // White parapet coping strip at top of Brick Tower (y = 25.05..25.25m, ref_076)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thSouthEnd - 0.1 * DEG, thTowerEnd + 0.1 * DEG, rInS + 0.9, rOut + 0.35, cfg.towerTopY - 0.18, cfg.towerTopY + 0.05, 8,
      { uvMetersPerTile: 3.0 }
    );

    // Angled Cantilevered Grey/White Floor-Slab Sunshade Fins & Shadow Grooves on Outer SE Profile (ref_084, ref_076!)
    for (let fl = 1; fl <= 5; fl++) {
      const yf = 4.2 + fl * 3.8;
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, thSouthEnd, thTowerEnd - 0.5 * DEG, rOut + 0.1, rOut + 1.35, yf - 0.14, yf + 0.14, 6,
        { uvMetersPerTile: 3.0 }
      );
      // Recessed dark window slot bands between sunshade fins on SE outer face
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.windowGlass, thSouthEnd + 0.8 * DEG, thTowerEnd - 1.2 * DEG, rOut + 0.12, rOut + 0.34, yf - 2.3, yf - 0.7, 4,
        { top: false, bottom: false, uvMetersPerTile: 2.2 }
      );
    }

    // 3 Vertically Stacked Protruding Dark-Grilled Hooded Window Boxes on Upper WNW Face (L4, L5, L6 at th = +83°, ref_076-077)
    {
      const cT = Math.cos(thTowerEnd);
      const sT = Math.sin(thTowerEnd);
      // Outward normal of the +83° end cap points toward increasing theta: (-sin th, 0, cos th)!
      const nx = -sT;
      const nz = cT;
      const faceYaw = Math.atan2(nx, nz);
      const rWin = 0.5 * (rInS + 2.0 + rOut - 1.0); // ~50.2m radial center

      for (const wy of [13.6, 17.5, 21.4]) {
        const boxGroup = new THREE.Group();
        boxGroup.position.set(rWin * cT + nx * 0.45, wy, rWin * sT + nz * 0.45);
        boxGroup.rotation.y = faceYaw;
        group.add(boxGroup);

        // Protruding dark window cage
        const cage = new THREE.Mesh(new THREE.BoxGeometry(2.6, 1.65, 0.85), materials.darkMetal);
        cage.castShadow = true;
        boxGroup.add(cage);

        const glass = new THREE.Mesh(new THREE.BoxGeometry(2.35, 1.42, 0.90), materials.windowGlass);
        boxGroup.add(glass);

        // Sloped Silver/White Metal Awning Hood on top (ref_076-077)
        const hood = new THREE.Mesh(new THREE.BoxGeometry(2.95, 0.14, 1.15), materials.silverLouver);
        hood.position.set(0, 0.96, 0.12);
        hood.rotation.x = 0.24;
        hood.castShadow = true;
        boxGroup.add(hood);
      }
    }

    return group;
  }

  static _buildCantileveredGlassBox(parent, materials, thCenter, rOut) {
    const c = Math.cos(thCenter), s = Math.sin(thCenter);
    const boxGroup = new THREE.Group();
    boxGroup.position.set((rOut + 0.55) * c, 17.25, (rOut + 0.55) * s);
    boxGroup.rotation.y = Math.atan2(c, s);
    parent.add(boxGroup);

    const w = 4.6, h = 6.9, d = 1.85;
    // Stepped white concrete corbel soffit slab underneath (ref_009, ref_010)
    const corbel = new THREE.Mesh(new THREE.BoxGeometry(w + 0.3, 0.45, d + 0.2), materials.architecturalWhite);
    corbel.position.set(0, -h * 0.5 - 0.15, 0);
    corbel.castShadow = true;
    boxGroup.add(corbel);

    // 3-Sided Glazed Observation Box + Overhanging Cap Slab
    const glassBody = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), materials.windowGlass);
    glassBody.castShadow = true;
    boxGroup.add(glassBody);

    const cap = new THREE.Mesh(new THREE.BoxGeometry(w + 0.45, 0.34, d + 0.45), materials.architecturalWhite);
    cap.position.set(0, h * 0.5 + 0.15, 0.05);
    cap.castShadow = true;
    boxGroup.add(cap);

    // Crisp white corner mullions & 2 horizontal transoms (3 glass tiers, ref_009)
    for (const sx of [-w * 0.5, w * 0.5]) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(0.18, h, 0.18), materials.architecturalWhite);
      m.position.set(sx, 0, d * 0.5);
      boxGroup.add(m);
    }
    for (const ty of [-h * 0.17, h * 0.17]) {
      const tr = new THREE.Mesh(new THREE.BoxGeometry(w + 0.08, 0.14, d + 0.08), materials.architecturalWhite);
      tr.position.set(0, ty, 0);
      boxGroup.add(tr);
    }
  }

  static _buildOpenPicketGate(parent, materials, thCenter, rGate, widthMeters, includeDrpmSign = false) {
    const c = Math.cos(thCenter), s = Math.sin(thCenter);
    const gateGroup = new THREE.Group();
    gateGroup.position.set(rGate * c, 0.15, rGate * s);
    gateGroup.rotation.y = Math.atan2(c, s);
    parent.add(gateGroup);

    const leafW = widthMeters * 0.44;
    const gateH = 1.65;

    // Two gate leaves swung open ~78° inward along left & right curbs (ref_012, ref_024)
    for (const sgn of [-1, 1]) {
      const leaf = new THREE.Group();
      leaf.position.set(sgn * (widthMeters * 0.5), 0, 0);
      leaf.rotation.y = sgn * 1.38; // swung open inward!
      gateGroup.add(leaf);

      const lx = -sgn * leafW * 0.5;
      for (const ry of [0.22, gateH - 0.08]) {
        const rail = new THREE.Mesh(new THREE.BoxGeometry(leafW, 0.06, 0.06), materials.darkMetal);
        rail.position.set(lx, ry, 0);
        leaf.add(rail);
      }
      for (let p = 0; p <= 8; p++) {
        const px = -sgn * (p / 8) * leafW;
        const picket = new THREE.Mesh(new THREE.BoxGeometry(0.035, gateH, 0.035), materials.darkMetal);
        picket.position.set(px, gateH * 0.5, 0);
        leaf.add(picket);
      }

      // Mount "P / PARKIR KHUSUS PIMPINAN" plaque on south/left leaf of DRPM gate (ref_012, Tile 6 left!)
      if (includeDrpmSign && sgn === 1) {
        const sign = new THREE.Mesh(
          ProceduralTextureFactory.createAtlasSignPlane(0.68, 0.96, 680, 528, 172, 248),
          materials.signAtlasMat
        );
        sign.position.set(lx, 0.98, 0.05);
        leaf.add(sign);

        const signBack = new THREE.Mesh(
          ProceduralTextureFactory.createAtlasSignPlane(0.68, 0.96, 680, 528, 172, 248),
          materials.signAtlasMat
        );
        signBack.position.set(lx, 0.98, -0.05);
        signBack.rotation.y = Math.PI;
        leaf.add(signBack);
      }
    }
  }
}
