import * as THREE from 'three';
import { NexusSSoSTKernel } from '../../core/MasterSurfaceKernels.js';
import { ProceduralTextureFactory } from '../../core/ProceduralTextureFactory.js';
import { UI_DPPD_MASTER_PLAN } from '../../world-gis/MasterPlanConfig.js';

const DEG = Math.PI / 180;

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./library/stp-ui/index.js
 *
 * Class-A Procedural Building Module: Gedung STP-UI (Science Techno Park Universitas Indonesia)
 * Ground-Truth References: ref_094-113, ref_115-117, ref_122-124, ref_141-174, ref_195-200.
 *
 * Key Architectural Features:
 * - 8-Story Curved Contemporary Hub (th = -150° WNW across -90° North to -34° ENE, H = 26.80m)
 * - Symmetrical Full-Height Metallic Copper ACP Corner Blades + 8-Story Copper-Fin Fire-Stair Cores at SW & NE
 * - L1 Glazed Cylindrical-Piloti Lobby (NE/Center) + Solid MEP Louver Wall (NW) + Silver Portico w/ Angled V-Columns
 * - L2-L3 Signature Facade: 24 Modules Alternating Opposing '/' & '\' Copper-Bronze ACP Trapezoids & Vertical Copper Louvers
 * - L4 Continuous Ribbon Windows; L5-L8 Smooth Cool-White Facade w/ 5 Continuous Curved White Sunshade Ledges & Box Windows
 * - L8 NE High-Level Branding: Gold UI Makara Crest + 2-Row White 3D Channel Letters ("SCIENCE TECHNO PARK / UNIVERSITAS INDONESIA")
 * - Curved Copper Roof Crown + 18 Radial Rooftop Solar PV Arrays + Sloped "STP-UI" Planter, Black/Yellow Totem & Red Hydrant Pipe
 */
export class GedungSTPUIBuilder {
  static build(materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.stp;
    const group = new THREE.Group();
    group.name = 'Gedung_STP_UI';

    const th0 = cfg.thStart; // -150°
    const th1 = cfg.thEnd;   // -34°
    const rIn = cfg.rInner;  // 39.0m
    const rOut = cfg.rOuter; // 57.8m

    const thBody0 = th0 + 2.4 * DEG; // -147.6°
    const thBody1 = th1 - 2.4 * DEG; // -36.4°

    // 1. Concrete Plinth & Inner Concave Courtyard Wall (8 stories w/ curved window ribbons & sunshade ledges, ref_195)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.pavingBlock, th0, th1, rIn - 0.5, rOut + 0.6, -0.2, 0.24, 48,
      { bottom: false, uvMetersPerTile: 4.0 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thBody0, thBody1, rIn, rIn + 0.9, 0.24, cfg.roofSlabY, 44,
      { bottom: false, uvMetersPerTile: 4.0 }
    );
    for (let fl = 1; fl < 8; fl++) {
      const yf = 1.2 + fl * 3.0;
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.windowGlass, thBody0 + 0.02, thBody1 - 0.02, rIn - 0.06, rIn + 0.15, yf, yf + 1.55, 40,
        { top: false, bottom: false, uvMetersPerTile: 3.5 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, thBody0, thBody1, rIn - 0.45, rIn + 0.05, yf + 1.55, yf + 1.82, 40,
        { bottom: true, uvMetersPerTile: 4.0 }
      );
    }

    // 2. Symmetrical 8-Story Copper ACP End Corner Blades & Scissor-Stair Louver Cores at SW (-150°) & NE (-34°) (ref_111, ref_148, ref_195)
    for (const [endStart, endStop, isNE] of [[th0, thBody0, false], [thBody1, th1, true]]) {
      // Front full-height vertical Copper-Bronze ACP corner blade rising to y = 27.10m
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.copperACP, endStart, endStop, rOut - 2.0, rOut + 0.48, 0.24, cfg.crownTopY + 0.30, 3,
        { bottom: false, uvMetersPerTile: 3.2 }
      );
      // Recessed white concrete scissor-stair core + vertical copper louver screen behind the front copper pier
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, endStart + 0.004, endStop - 0.004, rIn + 2.2, rOut - 2.0, 0.24, cfg.crownTopY - 0.3, 2,
        { bottom: false, uvMetersPerTile: 3.8 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.copperACP, endStart, endStop, rIn, rIn + 2.2, 0.24, cfg.crownTopY - 0.1, 2,
        { bottom: false, uvMetersPerTile: 3.2 }
      );
      // Vertical copper-fin stair screen tubes along outer flank of stair shaft
      const flankTh = isNE ? endStop - 0.002 : endStart + 0.002;
      const cF = Math.cos(flankTh);
      const sF = Math.sin(flankTh);
      for (let r = rIn + 3.0; r <= rOut - 2.4; r += 0.85) {
        const fin = new THREE.Mesh(new THREE.BoxGeometry(0.10, cfg.roofSlabY - 3.8, 0.22), materials.copperACP);
        fin.position.set(r * cF, 4.0 + (cfg.roofSlabY - 3.8) * 0.5, r * sF);
        fin.rotation.y = -flankTh;
        group.add(fin);
      }
    }

    // 3. Ground Floor L1: NW MEP Wall (-147.6°..-100°) + NE Glazed Lobby & Cylindrical Pilotis (-100°..-36.4°) (ref_111, ref_153)
    const thSplitL1 = -100.0 * DEG;
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thBody0, thSplitL1, rOut - 1.3, rOut - 0.55, 0.24, cfg.l1TopY, 20,
      { bottom: false, uvMetersPerTile: 3.5 }
    );
    // Clerestory windows & MEP intake louver panels on NW L1 wall
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.silverLouver, thBody0 + 0.05, thSplitL1 - 0.06, rOut - 0.62, rOut - 0.48, 0.65, 1.85, 16,
      { top: false, bottom: false, uvMetersPerTile: 2.0 }
    );

    // Glazed L1 Lobby + Thick Cylindrical White Concrete Pilotis across Center & NE
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.windowGlass, thSplitL1, thBody1 - 3.0 * DEG, rOut - 2.3, rOut - 1.7, 0.24, cfg.l1TopY, 24,
      { bottom: false, uvMetersPerTile: 3.2 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thBody1 - 3.0 * DEG, thBody1, rOut - 1.5, rOut - 0.55, 0.24, cfg.l1TopY, 4,
      { bottom: false, uvMetersPerTile: 3.2 }
    );
    for (let deg = -98; deg <= -44; deg += 6.75) {
      const rad = deg * DEG;
      const col = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, cfg.l1TopY - 0.2, 14), materials.architecturalWhite);
      col.position.set((rOut - 0.85) * Math.cos(rad), 0.2 + (cfg.l1TopY - 0.2) * 0.5, (rOut - 0.85) * Math.sin(rad));
      col.castShadow = true;
      group.add(col);
    }

    // 4. L1 North Main Drop-Off Portico Canopy w/ Angled V-Columns & Fascia Lettering (thPortico = -88°, ref_094, ref_164)
    GedungSTPUIBuilder._buildNorthPortico(group, materials, cfg.thPortico, rOut, cfg.l1TopY);

    // 5. L2–L3 Signature 2-Story Podium Screen (y = 4.30 .. 11.10m): Opposing '/' & '\' Copper ACP Trapezoids + Vertical Louvers
    // Inner dark glass wall behind screen
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.windowGlass, thBody0, thBody1, rOut - 0.95, rOut - 0.45, cfg.l1TopY, cfg.l3TopY, 44,
      { bottom: false, uvMetersPerTile: 3.2 }
    );
    // Projecting L2 soffit & L3 mid-rail & L4 base slab
    for (const yBand of [cfg.l1TopY, 7.70, cfg.l3TopY]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, thBody0, thBody1, rOut - 0.95, rOut + 0.25, yBand - 0.16, yBand + 0.16, 48,
        { uvMetersPerTile: 4.0 }
      );
    }

    const totalScreenSpan = thBody1 - thBody0;
    const screenBays = 24;
    for (let b = 0; b < screenBays; b++) {
      const b0 = thBody0 + (b / screenBays) * totalScreenSpan;
      const b1 = thBody0 + ((b + 1) / screenBays) * totalScreenSpan;
      const bMid = 0.5 * (b0 + b1);

      if (b % 2 === 0) {
        // Angled Trapezoidal Copper-Bronze ACP Panel (tilting '/' on L2 and '\' on L3 -> chevron geometry, ref_111, ref_153)
        const tilt = ((b / 2) % 2 === 0 ? 1 : -1) * 0.22 * (b1 - b0);
        const geoL2 = NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const wFactor = 0.88 - 0.26 * v;
            const th = bMid + (u - 0.5) * (b1 - b0) * wFactor + (v - 0.5) * tilt;
            const c = Math.cos(th), s = Math.sin(th);
            p.set((rOut + 0.22) * c, cfg.l1TopY + 0.16 + v * (3.40 - 0.32), (rOut + 0.22) * s);
            n.set(c, 0, s);
          },
          0, 1, 0, 1, 2, 1
        );
        const mL2 = new THREE.Mesh(geoL2, materials.copperACP);
        mL2.castShadow = true;
        group.add(mL2);

        const geoL3 = NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const wFactor = 0.62 + 0.26 * v;
            const th = bMid + (u - 0.5) * (b1 - b0) * wFactor - (v - 0.5) * tilt;
            const c = Math.cos(th), s = Math.sin(th);
            p.set((rOut + 0.22) * c, 7.70 + 0.16 + v * (3.40 - 0.32), (rOut + 0.22) * s);
            n.set(c, 0, s);
          },
          0, 1, 0, 1, 2, 1
        );
        const mL3 = new THREE.Mesh(geoL3, materials.copperACP);
        mL3.castShadow = true;
        group.add(mL3);
      } else {
        // Vertical Copper-Tube Fin Louver Bay (6 slender vertical tubes spanning L2-L3, ref_111, ref_153)
        for (let f = 1; f <= 6; f++) {
          const tf = b0 + (f / 7) * (b1 - b0);
          const fin = new THREE.Mesh(
            new THREE.BoxGeometry(0.11, cfg.l3TopY - cfg.l1TopY - 0.28, 0.16),
            materials.copperACP
          );
          fin.position.set(
            (rOut + 0.18) * Math.cos(tf),
            0.5 * (cfg.l1TopY + cfg.l3TopY),
            (rOut + 0.18) * Math.sin(tf)
          );
          fin.rotation.y = -tf;
          fin.castShadow = true;
          group.add(fin);
        }
      }
    }

    // 6. Upper Curved Architecture L4–L8 (y = 11.10 .. 25.40m):
    //    Core Wall + L4 Ribbon Windows + 5 Continuous Curved White Sunshade Ledges + Protruding White Box Window Frames!
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thBody0, thBody1, rIn + 0.85, rOut - 0.35, cfg.l1TopY, cfg.roofSlabY, 48,
      { bottom: false, inner: false, uvMetersPerTile: 4.0 }
    );

    // L4 Continuous Multi-Pane Dark Ribbon Window Band (y = 11.28 .. 13.80m, ref_111, ref_156)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.windowGlass, thBody0 + 3.5 * DEG, thBody1 - 3.5 * DEG, rOut - 0.42, rOut - 0.26, 11.32, 13.78, 42,
      { top: false, bottom: false, uvMetersPerTile: 3.0 }
    );

    // 5 Continuous Deep Curved White Sunshade Ledges (above L4, L5, L6, L7, & L8 Roof Cornice, ref_111, ref_148)
    const ledgeYs = [13.95, 16.80, 19.65, 22.50, cfg.roofSlabY];
    for (let li = 0; li < ledgeYs.length; li++) {
      const ly = ledgeYs[li];
      const overhang = li === 4 ? 1.05 : 0.78;
      const thick = li === 4 ? 0.38 : 0.24;
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, thBody0, thBody1, rOut - 0.45, rOut + overhang, ly - thick * 0.5, ly + thick * 0.5, 52,
        { uvMetersPerTile: 4.0 }
      );
    }

    // Stacked Copper AC Louver Cages in Outermost Bay 1 adjacent to SW & NE Copper Piers (L4-L7, ref_111, ref_148)
    for (const [c0, c1] of [[thBody0 + 0.3 * DEG, thBody0 + 3.4 * DEG], [thBody1 - 3.4 * DEG, thBody1 - 0.3 * DEG]]) {
      for (const yFloor of [11.15, 14.05, 16.90, 19.75]) {
        NexusSSoSTKernel.addCurvedPolarBand(
          group, materials.copperACP, c0, c1, rOut - 0.4, rOut + 0.18, yFloor + 0.1, yFloor + 2.55, 3,
          { uvMetersPerTile: 2.0 }
        );
      }
    }

    // Rhythmic Punched Windows w/ Protruding 3D White Box Surrounds across L5, L6, L7, & NW L8 (ref_111, ref_148, ref_150)
    const upperFloors = [
      { y0: 14.15, y1: 16.55, isL8: false },
      { y0: 17.00, y1: 19.40, isL8: false },
      { y0: 19.85, y1: 22.25, isL8: false },
      { y0: 22.70, y1: 25.05, isL8: true }
    ];
    for (let fi = 0; fi < upperFloors.length; fi++) {
      const fl = upperFloors[fi];
      const endLimitDeg = fl.isL8 ? -61.0 : -41.5; // Leave NE L8 (-60°..-36.5°) clear for Gold Makara + 3D Channel Letters!
      for (let deg = -142.0 + (fi % 2) * 1.8; deg <= endLimitDeg; deg += 5.4) {
        const rad = deg * DEG;
        const yc = 0.5 * (fl.y0 + fl.y1);
        const wh = fl.y1 - fl.y0 - 0.35;
        const ww = (deg > -80 && !fl.isL8 && fi % 2 === 0) ? 2.65 : 1.45;

        const c = Math.cos(rad);
        const s = Math.sin(rad);
        const yaw = Math.atan2(c, s);

        // Protruding 3D White Box Frame Surround (projecting 22cm proud of curved wall, ref_148)
        const surround = new THREE.Mesh(new THREE.BoxGeometry(ww + 0.34, wh + 0.32, 0.32), materials.architecturalWhite);
        surround.position.set((rOut - 0.25) * c, yc, (rOut - 0.25) * s);
        surround.rotation.y = yaw;
        surround.castShadow = true;
        group.add(surround);

        const pane = new THREE.Mesh(new THREE.BoxGeometry(ww, wh, 0.36), materials.windowGlass);
        pane.position.set((rOut - 0.24) * c, yc, (rOut - 0.24) * s);
        pane.rotation.y = yaw;
        group.add(pane);
      }
    }

    // 7. NE L8 High-Level Signage Frieze: Gold UI Makara + "SCIENCE TECHNO PARK / UNIVERSITAS INDONESIA" (th = -59.5°..-37°, ref_111)
    {
      const sTh0 = -59.5 * DEG;
      const sTh1 = -37.0 * DEG;
      const signGeo = NexusSSoSTKernel.sliceSubdomainFromMaster(
        (u, v, p, n) => {
          const th = sTh0 + u * (sTh1 - sTh0);
          const c = Math.cos(th), s = Math.sin(th);
          p.set((rOut - 0.16) * c, 22.68 + v * 2.42, (rOut - 0.16) * s);
          n.set(c, 0, s);
        },
        0, 1, 0, 1, 16, 1
      );
      // Map UVs cleanly onto Tile 0 (yPx: 0..160 -> v: 1.0 - 160/1024 .. 1.0) of the Unified Signage Atlas!
      const uv = signGeo.attributes.uv;
      const vBot = 1.0 - 158 / 1024;
      const vTop = 1.0 - 2 / 1024;
      for (let i = 0; i < uv.count; i++) {
        // Flip U (1 - u) because facing North-East from outside looks left-to-right from -37° toward -59.5°!
        // Let's verify! Standing at NE road (-42°) looking SW toward (-48°):
        // Left side of screen is East (-37°, next to NE Copper Pier!), right side of screen is West (-59.5°)!
        // At u = 1 (th = -37°), that's the LEFT side of the sign when viewed from outside!
        // Let's verify with cross product: viewer forward F = (-cos th, 0, -sin th); viewer UP = (0, 1, 0);
        // Viewer RIGHT = F x UP = (-cos th, 0, -sin th) x (0, 1, 0) = (+sin th, 0, -cos th)!
        // Derivative dP/d(th) = (-r sin th, 0, +r cos th) = -r * Viewer_RIGHT!
        // YES!! Increasing th goes LEFT in viewer screen space, so DECREASING th (from -37° to -59.5°) goes RIGHT!
        // Therefore u_text = 0 at th = -37° (u=1) and u_text = 1 at th = -59.5° (u=0) => u_atlas = 1.0 - u!
        const rawU = uv.getX(i);
        const rawV = uv.getY(i);
        uv.setXY(i, 0.004 + (1.0 - rawU) * 0.992, vBot + rawV * (vTop - vBot));
      }
      uv.needsUpdate = true;
      const signMesh = new THREE.Mesh(signGeo, materials.signAtlasMat);
      group.add(signMesh);
    }

    // 8. Curved Metallic Copper-Bronze Roof Crown Parapet + 18 Radial Rooftop Solar PV Strips & MEP Penthouse (ref_111, ref_174)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.copperACP, thBody0, thBody1, rOut - 0.55, rOut + 0.42, cfg.roofSlabY + 0.18, cfg.crownTopY, 48,
      { bottom: false, uvMetersPerTile: 3.5 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.pavingBlock, thBody0, thBody1, rIn + 0.5, rOut - 0.55, cfg.roofSlabY - 0.25, cfg.roofSlabY + 0.15, 44,
      { bottom: false, outer: false, inner: false, uvMetersPerTile: 5.0 }
    );

    // 18 Radial Rooftop Solar PV Panel Strips along Western-to-Central Roof Arc (-140° .. -66°, ref_174)
    for (let i = 0; i < 18; i++) {
      const rad = (-140.0 + i * 4.2) * DEG;
      const rMid = 0.5 * (rIn + rOut);
      const pv = new THREE.Mesh(new THREE.BoxGeometry(11.6, 0.14, 1.95), materials.solarPV);
      pv.position.set(rMid * Math.cos(rad), cfg.roofSlabY + 0.48, rMid * Math.sin(rad));
      pv.rotation.y = -rad;
      pv.rotation.z = 0.12;
      pv.castShadow = true;
      group.add(pv);
    }

    // Eastern Rooftop MEP / Chiller Penthouse Box (-58° .. -43°, ref_174)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, -58 * DEG, -43 * DEG, rIn + 4.5, rOut - 5.0, cfg.roofSlabY + 0.15, cfg.roofSlabY + 2.35, 10,
      { bottom: false, uvMetersPerTile: 3.5 }
    );

    // 9. North Forecourt: Sloped "STP-UI" Wedge Planter, Black/Yellow Directory Totem, & Exposed Red Fire-Hydrant Pipeline
    GedungSTPUIBuilder._buildNorthLandscapeProps(group, materials, cfg.thPortico);

    return group;
  }

  static _buildNorthPortico(parent, materials, thPortico, rOut, l1TopY) {
    const portGroup = new THREE.Group();
    const c = Math.cos(thPortico);
    const s = Math.sin(thPortico);
    portGroup.position.set(rOut * c, 0, rOut * s);
    // Outward normal (c, 0, s) = local +Z => rotation.y = atan2(c, s)
    portGroup.rotation.y = Math.atan2(c, s);
    parent.add(portGroup);

    const canW = 13.8;
    const canD = 6.6;
    const canY = l1TopY + 0.18;

    // Silver-white ACP flat-fascia canopy + warm wood/taupe underside soffit
    const canopy = new THREE.Mesh(new THREE.BoxGeometry(canW, 0.62, canD), materials.architecturalWhite);
    canopy.position.set(0, canY, canD * 0.5 - 0.4);
    canopy.castShadow = true;
    canopy.receiveShadow = true;
    portGroup.add(canopy);

    const soffit = new THREE.Mesh(new THREE.BoxGeometry(canW - 0.5, 0.08, canD - 0.5), materials.warmTaupeSoffit);
    soffit.position.set(0, canY - 0.30, canD * 0.5 - 0.4);
    portGroup.add(soffit);

    // Crisp Front Fascia Lettering: "SCIENCE TECHNO PARK UNIVERSITAS INDONESIA" (Tile 1 in Atlas!)
    const fasciaSign = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(canW - 0.6, 0.54, 0, 160, 1024, 112),
      materials.signAtlasMat
    );
    fasciaSign.position.set(0, canY, canD - 0.4 + 0.325);
    portGroup.add(fasciaSign);

    // Characteristic Angled 'V'-Shaped Composite Columns at Left & Right Outer Corners (ref_094, ref_156, ref_164)
    for (const sx of [-5.2, 5.2]) {
      for (const tiltZ of [-0.20, 0.20]) {
        const vLeg = new THREE.Mesh(new THREE.BoxGeometry(0.38, canY - 0.2, 0.38), materials.architecturalWhite);
        vLeg.position.set(sx, 0.15 + (canY - 0.2) * 0.5, canD - 1.35 + tiltZ * 1.8);
        vLeg.rotation.x = -tiltZ;
        vLeg.castShadow = true;
        portGroup.add(vLeg);
      }
    }
  }

  static _buildNorthLandscapeProps(parent, materials, thPortico) {
    // A. Sloped White Concrete Wedge Planter w/ "STP-UI" Topiary Badge directly in front of North Portico (ref_156, ref_164)
    const c = Math.cos(thPortico);
    const s = Math.sin(thPortico);
    const yaw = Math.atan2(c, s);

    const planterGroup = new THREE.Group();
    planterGroup.position.set(66.9 * c, 0.15, 66.9 * s);
    planterGroup.rotation.y = yaw;
    parent.add(planterGroup);

    const wedgeBase = new THREE.Mesh(new THREE.BoxGeometry(6.4, 0.72, 2.2), materials.architecturalWhite);
    wedgeBase.position.set(0, 0.36, 0);
    wedgeBase.castShadow = true;
    planterGroup.add(wedgeBase);

    const turfFace = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(5.8, 1.85, 852, 528, 172, 248),
      materials.signAtlasMat
    );
    turfFace.position.set(0, 0.76, 0.05);
    turfFace.rotation.x = -Math.PI * 0.36; // tilted 32° toward North Ring Road!
    planterGroup.add(turfFace);

    // B. Tall Matte-Black & Yellow STP Directory Totem just East of Portico (th = -75.5°, r = 67.2m, ref_105, ref_156)
    const thTot = -75.5 * DEG;
    const ct = Math.cos(thTot), st = Math.sin(thTot);
    const totGroup = new THREE.Group();
    totGroup.position.set(67.2 * ct, 0.15, 67.2 * st);
    totGroup.rotation.y = Math.atan2(ct, st) + 0.35;
    parent.add(totGroup);

    const totPed = new THREE.Mesh(new THREE.BoxGeometry(1.45, 0.42, 0.65), materials.pavingBlock);
    totPed.position.set(0, 0.21, 0);
    totGroup.add(totPed);

    const totBody = new THREE.Mesh(new THREE.BoxGeometry(1.26, 2.85, 0.32), materials.darkMetal);
    totBody.position.set(0, 0.42 + 2.85 * 0.5, 0);
    totBody.castShadow = true;
    totGroup.add(totBody);

    for (const zSide of [-0.17, 0.17]) {
      const face = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasSignPlane(1.20, 2.75, 0, 528, 340, 496),
        materials.signAtlasMat
      );
      face.position.set(0, 0.42 + 2.85 * 0.5, zSide);
      if (zSide < 0) face.rotation.y = Math.PI;
      totGroup.add(face);
    }

    // C. Exposed Red-Oxide Fire-Hydrant Pipeline & Twin Pillar Hydrants along Inner Ditch Bank (ref_096, ref_157)
    NexusSSoSTKernel.addCurvedPolarBand(
      parent, materials.redHydrant, -134 * DEG, -74 * DEG, 68.05, 68.21, 0.26, 0.40, 28,
      { bottom: false, uvMetersPerTile: 3.0 }
    );
    for (const hDeg of [-132, -76.5]) {
      const hr = hDeg * DEG;
      const hyd = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.16, 0.92, 10), materials.redHydrant);
      hyd.position.set(67.95 * Math.cos(hr), 0.58, 67.95 * Math.sin(hr));
      hyd.castShadow = true;
      parent.add(hyd);
    }
  }
}
