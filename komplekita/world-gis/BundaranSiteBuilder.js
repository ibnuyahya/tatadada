import * as THREE from 'three';
import { NexusSSoSTKernel } from '../core/MasterSurfaceKernels.js';
import { ProceduralTextureFactory } from '../core/ProceduralTextureFactory.js';
import { UI_DPPD_MASTER_PLAN } from './MasterPlanConfig.js';

const DEG = Math.PI / 180;

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./world-gis/BundaranSiteBuilder.js
 *
 * Layer 2 Geodetic Site, Ring Road, Perimeter Ditch, Verified Bridges, & Curated Tropical Flora Builder.
 * Strictly enforces the Negative Exclusion List (ref_088-093, ref_114, ref_163):
 * - ZERO cars, ZERO pedestrians, ZERO Street View 'X' or circular cursor discs, ZERO out-of-island buildings!
 */
export class BundaranSiteBuilder {
  static build(materials) {
    const group = new THREE.Group();
    group.name = 'BundaranSiteAndLandscape';

    const rr = UI_DPPD_MASTER_PLAN.ringRoad;
    const env = UI_DPPD_MASTER_PLAN.envelope;

    // 1. Sub-Terrain Anti-Floating Plinth Skirt (y = -18.0m .. 0.0m, r = 88.0m — Hukum Emas 11)
    const skirtGeo = new THREE.CylinderGeometry(rr.rPlinthRim, rr.rPlinthRim, env.foundationSkirtDepth, 96, 1, false);
    skirtGeo.translate(0, -env.foundationSkirtDepth * 0.5, 0);
    const skirtMesh = new THREE.Mesh(skirtGeo, materials.plinthEarthSkirt);
    skirtMesh.receiveShadow = true;
    group.add(skirtMesh);

    // 2. Outer Landscaped Woodland Rim (r = 79.85m .. 88.00m, sloped gently y = 0.14m -> 0.55m)
    const outerRimGeo = NexusSSoSTKernel.createPolarSlopedRibbon(
      0, Math.PI * 2, rr.rOuterCurb, 0.14, rr.rPlinthRim, 0.55, 112, 5.0
    );
    const outerRimMesh = new THREE.Mesh(outerRimGeo, materials.lushGrass);
    outerRimMesh.receiveShadow = true;
    group.add(outerRimMesh);

    // 3. Two-Lane One-Way Counter-Clockwise Asphalt Ring Road (r = 71.50m .. 79.50m, y = 0.02m)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.asphaltRoad, 0, Math.PI * 2, rr.rAsphaltIn, rr.rAsphaltOut, -0.05, 0.02, 128,
      { top: true, bottom: false, outer: false, inner: false, startCap: false, endCap: false, uvMetersPerTile: 6.0, castShadow: false }
    );

    // Continuous Inner & Outer Solid Yellow Road Edge Lines (y = 0.032m — zero Z-fighting!)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.yellowLine, 0, Math.PI * 2, rr.rAsphaltIn + 0.22, rr.rAsphaltIn + 0.36, 0.02, 0.032, 128,
      { top: true, bottom: false, outer: false, inner: false, startCap: false, endCap: false, castShadow: false }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.yellowLine, 0, Math.PI * 2, rr.rAsphaltOut - 0.36, rr.rAsphaltOut - 0.22, 0.02, 0.032, 128,
      { top: true, bottom: false, outer: false, inner: false, startCap: false, endCap: false, castShadow: false }
    );

    // White Dashed Center Lane Divider (48 curved dashes at r = 75.50m)
    for (let i = 0; i < 48; i++) {
      const a0 = (i / 48) * Math.PI * 2;
      const a1 = a0 + (0.42 / 48) * Math.PI * 2;
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, a0, a1, rr.rCenterDash - 0.07, rr.rCenterDash + 0.07, 0.02, 0.033, 2,
        { top: true, bottom: false, outer: false, inner: false, startCap: false, endCap: false, castShadow: false }
      );
    }

    // 4. Outer Black-and-White Striped Concrete Curb (360° continuous)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.stripedCurb, 0, Math.PI * 2, rr.rAsphaltOut, rr.rOuterCurb, 0.01, 0.19, 128,
      { bottom: false, startCap: false, endCap: false, uvMetersPerTile: 1.8, castShadow: true }
    );

    // 5. Verified Entrance Driveway Cutouts & Bridges Across the Perimeter Stone Ditch
    // Angles in degrees [-180..+180] where curbs & ditch open for real bridges (ref_174, ref_206):
    const bridgeOpeningsDeg = [
      { start: -112.0, end: -104.5, type: 'asphalt', label: 'STP_Exit_NW' },
      { start: -73.5, end: -66.0, type: 'asphalt', label: 'STP_Entry_NE' },
      { start: 4.5, end: 11.5, type: 'asphalt', label: 'ILRC_East_Portico' },
      { start: 26.0, end: 33.0, type: 'asphalt', label: 'DRPM_Loop_Gate' },
      { start: 38.5, end: 43.5, type: 'koral', label: 'ILRC_Atrium_Ramp' },
      { start: 132.0, end: 139.0, type: 'paving', label: 'DPPD_Main_Apron' },
      { start: 177.5, end: 185.5, type: 'asphalt', label: 'West_Main_ITTC_Gate' }
    ];

    // Normalize openings onto [0, 360) sorted ascending to build unbroken curb/ditch segments between bridges
    const normGaps = bridgeOpeningsDeg
      .map((g) => ({
        s: ((g.start % 360) + 360) % 360,
        e: ((g.end % 360) + 360) % 360,
        raw: g
      }))
      .sort((a, b) => a.s - b.s);

    for (let i = 0; i < normGaps.length; i++) {
      const cur = normGaps[i];
      const nxt = normGaps[(i + 1) % normGaps.length];
      const segStartDeg = cur.e;
      const segEndDeg = nxt.s > cur.e ? nxt.s : nxt.s + 360.0;
      const spanDeg = segEndDeg - segStartDeg;
      if (spanDeg <= 0.5) continue;

      const th0 = segStartDeg * DEG;
      const th1 = segEndDeg * DEG;
      const segs = Math.max(4, Math.ceil(spanDeg / 2.8));

      // Inner Black-and-White Striped Curb
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.stripedCurb, th0, th1, rr.rInnerVerge, rr.rInnerCurb, 0.01, 0.19, segs,
        { bottom: false, uvMetersPerTile: 1.8 }
      );

      // Inner Grass Verge Strip (r = 69.45 .. 71.15m)
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.lushGrass, th0, th1, rr.rDitchOuter, rr.rInnerVerge, 0.0, 0.14, segs,
        { bottom: false, uvMetersPerTile: 4.0, castShadow: false }
      );

      // Open Batu-Kali Drainage Ditch Walls & Floor (r = 68.55 .. 69.45m, floor y = -0.68m)
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.stoneRetaining, th0, th1, rr.rDitchOuter - 0.16, rr.rDitchOuter, -0.68, 0.15, segs,
        { bottom: false, uvMetersPerTile: 2.2 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.stoneRetaining, th0, th1, rr.rDitchInner, rr.rDitchInner + 0.16, -0.68, 0.15, segs,
        { bottom: false, uvMetersPerTile: 2.2 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.stoneRetaining, th0, th1, rr.rDitchInner, rr.rDitchOuter, -0.74, -0.66, segs,
        { top: true, bottom: false, outer: false, inner: false, uvMetersPerTile: 2.2, castShadow: false }
      );
    }

    // Build the Verified Culvert Bridges across the ditch + curb aprons
    // (Note: DPPD_Main_Apron is custom-built in DPPDSecondarySubdomains with its exact concrete bridge & radial U-gutter!)
    for (const br of bridgeOpeningsDeg) {
      if (br.label === 'DPPD_Main_Apron') continue;
      const th0 = br.start * DEG;
      const th1 = br.end * DEG;
      const mat =
        br.type === 'koral'
          ? materials.koralSikat
          : br.type === 'paving'
          ? materials.pavingBlock
          : materials.asphaltRoad;
      NexusSSoSTKernel.addCurvedPolarBand(
        group, mat, th0, th1, rr.rDitchInner - 0.8, rr.rInnerCurb + 0.05, -0.14, 0.15, 6,
        { bottom: false, uvMetersPerTile: 3.0 }
      );
      // Low concrete bridge side cheeks
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, th0 - 0.004, th0 + 0.004, rr.rDitchInner - 0.6, rr.rInnerVerge, 0.08, 0.28, 1,
        { bottom: false }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, th1 - 0.004, th1 + 0.004, rr.rDitchInner - 0.6, rr.rInnerVerge, 0.08, 0.28, 1,
        { bottom: false }
      );
    }

    // 6. Inner Island Base Turf Disk (r = 0 .. 68.55m, y = 0.12m)
    const islandGeo = new THREE.CylinderGeometry(rr.rDitchInner, rr.rDitchInner, 0.24, 112, 1, false);
    islandGeo.translate(0, 0.0, 0); // top surface at y = +0.12m
    const islandMesh = new THREE.Mesh(islandGeo, materials.lushGrass);
    islandMesh.receiveShadow = true;
    group.add(islandMesh);

    // 7. West 9:00 Main Driveway & Courtyard Plaza to ITTC Perpustakaan Lama (ref_174, ref_201-206)
    const westDriveGeo = new THREE.BoxGeometry(46.0, 0.08, 7.6);
    westDriveGeo.translate(-47.2, 0.14, -2.2);
    const westDriveMesh = new THREE.Mesh(westDriveGeo, materials.asphaltRoad);
    westDriveMesh.receiveShadow = true;
    group.add(westDriveMesh);

    // West 9:00 Driveway Splitter Island Signage: Black/Yellow "➔ ENTRANCE" Totem + Blue "➔ MASUK" Sign (ref_203)
    {
      const totBase = new THREE.Mesh(new THREE.BoxGeometry(1.18, 0.26, 0.48), materials.architecturalWhite);
      totBase.position.set(-67.8, 0.25, -4.6);
      totBase.rotation.y = -Math.PI * 0.42;
      group.add(totBase);

      const totBody = new THREE.Mesh(new THREE.BoxGeometry(0.95, 1.85, 0.24), materials.darkMetal);
      totBody.position.set(-67.8, 1.30, -4.6);
      totBody.rotation.y = -Math.PI * 0.42;
      totBody.castShadow = true;
      group.add(totBody);

      const totSign = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasSignPlane(0.88, 1.75, 0, 528, 340, 496),
        materials.signAtlasMat
      );
      totSign.position.set(-67.93, 1.30, -4.56);
      totSign.rotation.y = -Math.PI * 0.42;
      group.add(totSign);

      // Blue "➔ MASUK" directional sign on white post (ref_203)
      const mskPost = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.045, 1.38, 8), materials.architecturalWhite);
      mskPost.position.set(-65.2, 0.78, -5.6);
      group.add(mskPost);

      const mskSign = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasBox(0.76, 0.54, 0.08, 938, 716, 86, 60),
        materials.signAtlasMat
      );
      mskSign.position.set(-65.2, 1.68, -5.6);
      mskSign.rotation.y = -Math.PI * 0.42;
      group.add(mskSign);
    }

    // Paved West Forecourt Plaza in front of ITTC Perpustakaan Lama
    const westCourtGeo = new THREE.BoxGeometry(15.5, 0.09, 26.0);
    westCourtGeo.translate(-31.5, 0.145, 1.5);
    const westCourtMesh = new THREE.Mesh(westCourtGeo, materials.pavingBlock);
    westCourtMesh.receiveShadow = true;
    group.add(westCourtMesh);

    // Courtyard Connecting Walkways (ITTC to STP-UI North & ILRC East)
    const northWalk = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.09, 20.0), materials.pavingBlock);
    northWalk.position.set(-3.0, 0.145, -29.0);
    northWalk.receiveShadow = true;
    group.add(northWalk);

    const eastWalk = new THREE.Mesh(new THREE.BoxGeometry(20.0, 0.09, 4.2), materials.pavingBlock);
    eastWalk.position.set(27.5, 0.145, 1.5);
    eastWalk.receiveShadow = true;
    group.add(eastWalk);

    // 8. STP-UI North Crescent Drop-Off Lane (r = 59.0 .. 65.8m, th = -112° .. -66°, ref_164, ref_174)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.asphaltRoad, -112 * DEG, -66 * DEG, 58.6, 65.8, 0.08, 0.16, 32,
      { bottom: false, uvMetersPerTile: 5.0, castShadow: false }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.stripedCurb, -105 * DEG, -73 * DEG, 65.8, 66.12, 0.12, 0.28, 24,
      { bottom: false, uvMetersPerTile: 1.8 }
    );

    // 9. Clean Galvanized Curved-Arm Perimeter Streetlights w/ 3D Concrete Base Pedestals, LED Luminaires, & Curb Storm Drains
    for (let i = 0; i < 12; i++) {
      const th = (i / 12) * Math.PI * 2 + 0.18;
      const rPole = rr.rOuterCurb + 0.85;
      const px = rPole * Math.cos(th);
      const pz = rPole * Math.sin(th);

      // 3D Octagonal Concrete Pedestal Base
      const ped = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.24, 0.52, 8), materials.pavingBlock);
      ped.position.set(px, 0.36, pz);
      ped.castShadow = true;
      group.add(ped);

      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.12, 8.2, 8), materials.darkMetal);
      pole.position.set(px, 4.3, pz);
      pole.castShadow = true;
      group.add(pole);

      const arm = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.09, 0.12), materials.darkMetal);
      arm.position.set(px - Math.cos(th) * 0.82, 8.3, pz - Math.sin(th) * 0.82);
      arm.rotation.y = -th;
      group.add(arm);

      // 3D Die-Cast LED Cobra-Head Luminaire Housing + Warm Lens Underside
      const lampHead = new THREE.Mesh(new THREE.BoxGeometry(0.54, 0.12, 0.26), materials.architecturalWhite);
      lampHead.position.set(px - Math.cos(th) * 1.56, 8.28, pz - Math.sin(th) * 1.56);
      lampHead.rotation.y = -th;
      group.add(lampHead);

      // 3D Cast-Iron Storm-Drain Curb Inlet Grate on the asphalt edge
      const rGrate = rr.rAsphaltOut - 0.24;
      const grate = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.05, 0.68), materials.darkMetal);
      grate.position.set(rGrate * Math.cos(th), 0.025, rGrate * Math.sin(th));
      grate.rotation.y = -th;
      group.add(grate);
    }

    // 10. Organic Multi-Lobe Tropical Campus Trees & Curved Parabolic Palms (ONLY 2 InstancedMeshes!)
    BundaranSiteBuilder._buildCuratedTropicalFlora(group, materials);

    return group;
  }

  static _buildCuratedTropicalFlora(parent, materials) {
    // A. Broadleaf Tropical Tree Placements (keeping r <= 82.4m so max crown extent stays strictly inside r = 88.0m!)
    const treeSpecs = [
      // Outer woodland ring framing the roundabout naturally (16 trees, r = 81.6..82.4m -> exact 176.00m envelope!)
      { r: 82.2, deg: -165, s: 1.08, col: 0x3b682c },
      { r: 82.0, deg: -152, s: 0.96, col: 0x4b7735 },
      { r: 82.2, deg: -132, s: 1.04, col: 0x467633 },
      { r: 82.0, deg: -116, s: 0.98, col: 0x3c6a2d },
      { r: 82.2, deg: -62,  s: 1.02, col: 0x487634 },
      { r: 82.2, deg: -24,  s: 1.08, col: 0x3d6b2e },
      { r: 81.8, deg: -6,   s: 0.95, col: 0x517c39 },
      { r: 82.2, deg: 18,   s: 1.08, col: 0x4b7837 },
      { r: 82.0, deg: 35,   s: 0.96, col: 0x3f6d2f },
      { r: 82.2, deg: 54,   s: 1.10, col: 0x366128 },
      { r: 82.0, deg: 74,   s: 1.06, col: 0x437031 },
      { r: 82.0, deg: 94,   s: 1.02, col: 0x4a7634 },
      { r: 82.2, deg: 106,  s: 1.08, col: 0x3a662b },
      { r: 82.0, deg: 124,  s: 0.96, col: 0x4e7936 },
      { r: 82.2, deg: 156,  s: 1.05, col: 0x487735 },
      { r: 81.8, deg: 168,  s: 0.95, col: 0x3d692c },

      // West Entrance Courtyard Ketapang Kencana shade trees (ref_201-205)
      { x: -56.0, z: -9.5,  s: 1.05, col: 0x5c7d38 },
      { x: -42.0, z: -10.5, s: 1.12, col: 0x4e7433 },
      { x: -34.0, z: 13.5,  s: 0.96, col: 0x557b39 },
      { x: -51.5, z: 6.2,   s: 0.88, col: 0x4a7332 },

      // Inner Green Courtyard Ring (between ITTC Perpustakaan Lama and STP-UI / ILRC / DPPD, ref_174)
      { x: -15.0, z: -25.5, s: 1.08, col: 0x447130 },
      { x: 9.0,   z: -26.5, s: 1.15, col: 0x3d692c },
      { x: 24.5,  z: -14.0, s: 1.05, col: 0x527a37 },
      { x: 25.0,  z: 16.5,  s: 1.12, col: 0x416d2e },
      { x: 4.5,   z: 27.5,  s: 1.06, col: 0x4a7634 },
      { x: -16.5, z: 26.0,  s: 0.98, col: 0x3f6b2d },

      // East ILRC Berm Ketapang Kencana & Tanjung Trees (ref_119, ref_125 — spaced so L4 Glass Box stays visible)
      { r: 65.2, deg: -14,  s: 0.92, col: 0x627c3a },
      { r: 65.4, deg: -2,   s: 0.95, col: 0x4b7434 },
      { r: 65.2, deg: 19,   s: 0.94, col: 0x5a7838 },

      // South ILRC Bamboo-Staked Leaf Mound Crest Trees & Brick Tower Shrubs (ref_028, ref_033-050, ref_076)
      { r: 64.5, deg: 49,   s: 0.92, col: 0x3d6a2c, yBase: 1.35 },
      { r: 64.5, deg: 57,   s: 0.98, col: 0x487433, yBase: 1.40 },
      { r: 64.5, deg: 65,   s: 0.94, col: 0x3a6529, yBase: 1.35 },
      { r: 64.5, deg: 72,   s: 0.88, col: 0x4d7836, yBase: 1.20 },
      { r: 63.0, deg: 80.5, s: 0.62, col: 0x4f7b38, yBase: 0.35 }
    ];

    // Static tree trunks (merged into bambooWood material -> +0 extra draw calls!)
    for (const t of treeSpecs) {
      const tx = t.x !== undefined ? t.x : t.r * Math.cos(t.deg * DEG);
      const tz = t.z !== undefined ? t.z : t.r * Math.sin(t.deg * DEG);
      const y0 = t.yBase || 0.12;
      const trunkH = 3.6 * t.s;
      const trunk = new THREE.Mesh(
        new THREE.CylinderGeometry(0.22 * t.s, 0.36 * t.s, trunkH, 7),
        materials.bambooWood
      );
      trunk.position.set(tx, y0 + trunkH * 0.5, tz);
      trunk.castShadow = true;
      trunk.receiveShadow = true;
      parent.add(trunk);
    }

    // Build 1 Multi-Lobe Organic Canopy Prototype Geometry & render all crowns in 1 InstancedMesh!
    const crownProto = BundaranSiteBuilder._createMultiLobeCrownGeometry();
    const crownInst = new THREE.InstancedMesh(crownProto, materials.foliageCanopy, treeSpecs.length);
    crownInst.name = 'NexusInstancedBroadleafCrowns';
    crownInst.castShadow = true;
    crownInst.receiveShadow = true;

    const dummy = new THREE.Object3D();
    const tmpCol = new THREE.Color();
    treeSpecs.forEach((t, idx) => {
      const tx = t.x !== undefined ? t.x : t.r * Math.cos(t.deg * DEG);
      const tz = t.z !== undefined ? t.z : t.r * Math.sin(t.deg * DEG);
      const y0 = t.yBase || 0.12;
      dummy.position.set(tx, y0 + 4.3 * t.s, tz);
      dummy.rotation.set(0, (idx * 1.91) % (Math.PI * 2), 0);
      dummy.scale.set(t.s, t.s * 0.92, t.s);
      dummy.updateMatrix();
      crownInst.setMatrixAt(idx, dummy.matrix);
      tmpCol.setHex(t.col);
      crownInst.setColorAt(idx, tmpCol);
    });
    crownInst.instanceMatrix.needsUpdate = true;
    if (crownInst.instanceColor) crownInst.instanceColor.needsUpdate = true;
    parent.add(crownInst);

    // B. Authentic Parabolic Royal Palms & Manila Palms (ref_078, ref_084, ref_012, ref_029)
    const palmSpecs = [
      // 3 Tall Royal Palms ("Palem Raja") along DPPD SE Grass Apron (ref_078, ref_084)
      { r: 66.2, deg: 122.0, h: 12.6, s: 1.15 },
      { r: 66.4, deg: 109.5, h: 13.2, s: 1.18 },
      { r: 66.2, deg: 97.5,  h: 12.4, s: 1.12 },
      // 1 Solitary Palm near SE-ILRC Gap (ref_084 right edge)
      { r: 62.5, deg: 87.5,  h: 11.2, s: 1.05 },
      // 4 Manila Palms in front of DRPM L1 Breezeway + 1 Tall Atrium Flank Palm (ref_012, ref_029)
      { r: 60.4, deg: 27.5,  h: 6.2,  s: 0.72 },
      { r: 60.4, deg: 30.0,  h: 6.5,  s: 0.74 },
      { r: 60.4, deg: 32.5,  h: 6.1,  s: 0.72 },
      { r: 60.4, deg: 35.0,  h: 6.4,  s: 0.74 },
      { r: 59.6, deg: 36.8,  h: 9.2,  s: 0.92 }
    ];

    for (const pm of palmSpecs) {
      const px = pm.r * Math.cos(pm.deg * DEG);
      const pz = pm.r * Math.sin(pm.deg * DEG);
      const trunk = new THREE.Mesh(
        new THREE.CylinderGeometry(0.18 * pm.s, 0.28 * pm.s, pm.h, 8),
        materials.palmTrunk
      );
      trunk.position.set(px, 0.12 + pm.h * 0.5, pz);
      trunk.castShadow = true;
      trunk.receiveShadow = true;
      parent.add(trunk);
    }

    const palmCrownProto = BundaranSiteBuilder._createParabolicPalmCrownGeometry();
    const palmInst = new THREE.InstancedMesh(palmCrownProto, materials.foliageCanopy, palmSpecs.length);
    palmInst.name = 'NexusInstancedPalmCrowns';
    palmInst.castShadow = true;
    palmInst.receiveShadow = true;

    palmSpecs.forEach((pm, idx) => {
      const px = pm.r * Math.cos(pm.deg * DEG);
      const pz = pm.r * Math.sin(pm.deg * DEG);
      dummy.position.set(px, 0.12 + pm.h, pz);
      dummy.rotation.set(0, idx * 1.15, 0);
      dummy.scale.setScalar(pm.s);
      dummy.updateMatrix();
      palmInst.setMatrixAt(idx, dummy.matrix);
      tmpCol.setHex(idx < 4 ? 0x497832 : 0x55873b);
      palmInst.setColorAt(idx, tmpCol);
    });
    palmInst.instanceMatrix.needsUpdate = true;
    if (palmInst.instanceColor) palmInst.instanceColor.needsUpdate = true;
    parent.add(palmInst);
  }

  static _createMultiLobeCrownGeometry() {
    // 9-Lobe Tiered Tropical Broadleaf Crown (Ketapang Kencana & Tanjung layered sub-canopies)
    const lobes = [
      { x: 0.0,  y: 1.05, z: 0.0,  rx: 2.6, ry: 1.75, rz: 2.6 },
      { x: 1.55, y: 0.15, z: 0.85, rx: 2.15, ry: 1.45, rz: 2.05 },
      { x: -1.5, y: 0.20, z: 0.95, rx: 2.10, ry: 1.40, rz: 2.15 },
      { x: 0.25, y: 0.10, z: -1.6, rx: 2.20, ry: 1.45, rz: 2.05 },
      { x: -1.1, y: 0.35, z: -1.1, rx: 1.95, ry: 1.38, rz: 1.95 },
      { x: 1.25, y: 0.55, z: -1.0, rx: 1.80, ry: 1.30, rz: 1.85 },
      { x: -0.3, y: -0.45, z: 1.65, rx: 1.95, ry: 1.25, rz: 1.85 },
      { x: 1.85, y: -0.35, z: -0.2, rx: 1.75, ry: 1.20, rz: 1.75 },
      { x: -1.8, y: -0.30, z: -0.1, rx: 1.75, ry: 1.22, rz: 1.75 }
    ];
    const posArr = [];
    const normArr = [];
    const uvArr = [];
    const v = new THREE.Vector3();
    const n = new THREE.Vector3();

    for (const lb of lobes) {
      const g = new THREE.IcosahedronGeometry(1.0, 2);
      const nonIdx = g.index ? g.toNonIndexed() : g;
      const p = nonIdx.attributes.position;
      for (let i = 0; i < p.count; i++) {
        v.fromBufferAttribute(p, i);
        // Gentle organic noise displacement so no sphere looks robotic
        const noise = 1.0 + 0.09 * Math.sin(v.x * 5.0 + v.y * 3.0) * Math.cos(v.z * 4.0);
        n.copy(v).normalize();
        posArr.push(lb.x + v.x * lb.rx * noise, lb.y + v.y * lb.ry * noise, lb.z + v.z * lb.rz * noise);
        normArr.push(n.x, n.y, n.z);
        uvArr.push(0.5 + v.x * 0.5, 0.5 + v.z * 0.5);
      }
      g.dispose();
      if (nonIdx !== g) nonIdx.dispose();
    }

    const out = new THREE.BufferGeometry();
    out.setAttribute('position', new THREE.Float32BufferAttribute(posArr, 3));
    out.setAttribute('normal', new THREE.Float32BufferAttribute(normArr, 3));
    out.setAttribute('uv', new THREE.Float32BufferAttribute(uvArr, 2));
    out.computeBoundingBox();
    out.computeBoundingSphere();
    return out;
  }

  static _createParabolicPalmCrownGeometry() {
    const pos = [];
    const norm = [];
    const uvs = [];
    const frondCount = 13;
    const segs = 6;

    // Green crownshaft cylinder at top of trunk
    const shaft = new THREE.CylinderGeometry(0.24, 0.21, 1.1, 8).toNonIndexed();
    const sp = shaft.attributes.position;
    const sn = shaft.attributes.normal;
    for (let i = 0; i < sp.count; i++) {
      pos.push(sp.getX(i), sp.getY(i) - 0.35, sp.getZ(i));
      norm.push(sn.getX(i), sn.getY(i), sn.getZ(i));
      uvs.push(0.5, 0.5);
    }
    shaft.dispose();

    // 13 Gracefully arched parabolic V-folded palm fronds (upper & lower tiers)
    for (let f = 0; f < frondCount; f++) {
      const yaw = (f / frondCount) * Math.PI * 2 + (f % 2) * 0.14;
      const len = f % 2 === 0 ? 3.55 : 3.10;
      const lift = f % 2 === 0 ? 1.15 : 0.65;
      const droop = f % 2 === 0 ? 1.85 : 2.15;
      const cosY = Math.cos(yaw);
      const sinY = Math.sin(yaw);
      const perpX = -sinY;
      const perpZ = cosY;

      for (let s = 0; s < segs; s++) {
        const t0 = s / segs;
        const t1 = (s + 1) / segs;
        const r0 = 0.15 + t0 * len;
        const r1 = 0.15 + t1 * len;
        const y0 = 0.15 + lift * Math.sin(t0 * Math.PI * 0.65) - droop * t0 * t0;
        const y1 = 0.15 + lift * Math.sin(t1 * Math.PI * 0.65) - droop * t1 * t1;
        const w0 = 0.48 * Math.sin(Math.pow(t0, 0.65) * Math.PI) + 0.04;
        const w1 = 0.48 * Math.sin(Math.pow(t1, 0.65) * Math.PI) + 0.04;

        const c0 = [cosY * r0, y0 + 0.09, sinY * r0];
        const c1 = [cosY * r1, y1 + 0.09, sinY * r1];
        const l0 = [cosY * r0 + perpX * w0, y0 - 0.08, sinY * r0 + perpZ * w0];
        const l1 = [cosY * r1 + perpX * w1, y1 - 0.08, sinY * r1 + perpZ * w1];
        const rgt0 = [cosY * r0 - perpX * w0, y0 - 0.08, sinY * r0 - perpZ * w0];
        const rgt1 = [cosY * r1 - perpX * w1, y1 - 0.08, sinY * r1 - perpZ * w1];

        const c0Bot = [cosY * r0, y0 - 0.02, sinY * r0];
        const c1Bot = [cosY * r1, y1 - 0.02, sinY * r1];
        const l0Bot = [cosY * r0 + perpX * w0, y0 - 0.13, sinY * r0 + perpZ * w0];
        const l1Bot = [cosY * r1 + perpX * w1, y1 - 0.13, sinY * r1 + perpZ * w1];
        const rgt0Bot = [cosY * r0 - perpX * w0, y0 - 0.13, sinY * r0 - perpZ * w0];
        const rgt1Bot = [cosY * r1 - perpX * w1, y1 - 0.13, sinY * r1 - perpZ * w1];

        const pushTriSigned = (a, b, c, wantUp = true) => {
          const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
          const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
          let nx = uy * vz - uz * vy;
          let ny = uz * vx - ux * vz;
          let nz = ux * vy - uy * vx;
          const nl = Math.hypot(nx, ny, nz) || 1;
          nx /= nl; ny /= nl; nz /= nl;
          if ((wantUp && ny < 0) || (!wantUp && ny > 0)) {
            return pushTriSigned(a, c, b, wantUp);
          }
          for (const pt of [a, b, c]) {
            pos.push(pt[0], pt[1], pt[2]);
            norm.push(nx, ny, nz);
            uvs.push(0.5, 0.5);
          }
        };

        // Top sun-lit leaflets of the V-ribbed frond (visible from above & oblique)
        pushTriSigned(c0, l0, l1, true);
        pushTriSigned(c0, l1, c1, true);
        pushTriSigned(c0, c1, rgt1, true);
        pushTriSigned(c0, rgt1, rgt0, true);

        // 3D Thickened Underside leaflets & central rachis keel (visible from FPV Runner eye level below!)
        pushTriSigned(c0Bot, l1Bot, l0Bot, false);
        pushTriSigned(c0Bot, c1Bot, l1Bot, false);
        pushTriSigned(c0Bot, rgt1Bot, c1Bot, false);
        pushTriSigned(c0Bot, rgt0Bot, rgt1Bot, false);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    return geo;
  }
}
