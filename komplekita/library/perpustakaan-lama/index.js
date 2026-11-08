import * as THREE from 'three';
import { NexusSSoSTKernel } from '../../core/MasterSurfaceKernels.js';
import { UI_DPPD_MASTER_PLAN } from '../../world-gis/MasterPlanConfig.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (True-Color Photo-Realistic & Watertight Edition)
 * ./library/perpustakaan-lama/index.js
 *
 * Class-A Procedural Building Module: ITTC Perpustakaan Lama Universitas Indonesia
 * Ground-Truth References: ref_096-100, ref_136-142, ref_174, ref_202-204, ref_206.
 *
 * Key Architectural Features:
 * - Strictly Cardinal N-S / E-W 42.0m x 42.0m Square Plan (NOT rotated 45°!)
 * - Solid High-Level Reinforced-Concrete Central Roof Deck ("Atap Coran / Dak Beton" at y = 13.95m, 16.0m x 16.0m)
 *   surrounded by the 4-Wing Hipped Terracotta Roof (eaveY = 14.20m, ridgeY = 18.80m) and topped by the
 *   Raised 9.6m x 9.6m Pyramidal Terracotta Monitor / Skylight Roof Cap (ref_174 — zero open pit to ground!)
 * - 4 Full Stories with Bold White Reinforced-Concrete Structural Column Grid (7 bays per side) & Spandrels
 * - Recessed L1 Colonnade w/ Solid Ceiling Soffit + West Portico Canopy facing the 9:00 Main Driveway
 * - 1-Story Flat-Roofed NW Courtyard Annex (13.0m x 7.5m, ref_174)
 */
export class PerpustakaanLamaBuilder {
  static build(materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.ittc;
    const group = new THREE.Group();
    group.name = 'ITTC_Perpustakaan_Lama_UI';

    const localRoot = new THREE.Group();
    localRoot.position.set(cfg.centerX, 0, cfg.centerZ);
    group.add(localRoot);

    const H = cfg.outerHalf;     // 21.0m
    const C = cfg.courtyardHalf; // 8.0m
    const eaveY = cfg.eaveY;     // 14.20m
    const floorHeights = [0.15, 3.85, 7.30, 10.75, eaveY];

    // 1. Solid Full-Footprint Concrete Plinth Base (42.8m x 42.8m, y = -0.20 .. 0.24m)
    const plinthSlab = new THREE.Mesh(
      new THREE.BoxGeometry((H + 0.4) * 2, 0.44, (H + 0.4) * 2),
      materials.pavingBlock
    );
    plinthSlab.position.set(0, 0.02, 0);
    plinthSlab.receiveShadow = true;
    localRoot.add(plinthSlab);

    // 2. Recessed L1 Core Walls (inset 2.1m behind outer colonnade columns -> half = 18.9m)
    const l1Core = new THREE.Mesh(
      new THREE.BoxGeometry((H - 2.1) * 2, floorHeights[1] - 0.24, (H - 2.1) * 2),
      materials.windowGlass
    );
    l1Core.position.set(0, 0.24 + (floorHeights[1] - 0.24) * 0.5, 0);
    l1Core.castShadow = true;
    l1Core.receiveShadow = true;
    localRoot.add(l1Core);

    // 3. Upper Stories L2, L3, L4 Recessed Window Curtain Box (half = H - 0.55m, completely watertight!)
    const upperCore = new THREE.Mesh(
      new THREE.BoxGeometry((H - 0.55) * 2, eaveY - 0.25 - floorHeights[1], (H - 0.55) * 2),
      materials.windowGlass
    );
    upperCore.position.set(0, floorHeights[1] + (eaveY - 0.25 - floorHeights[1]) * 0.5, 0);
    upperCore.castShadow = true;
    upperCore.receiveShadow = true;
    localRoot.add(upperCore);

    // 4. Crisp Projecting White Horizontal Spandrel & Floor Slabs at L2 (solid colonnade ceiling!), L3, L4, & Eave Ring Beam
    for (let f = 1; f <= 4; f++) {
      const ySlab = floorHeights[f];
      const spandrelH = f === 4 ? 0.58 : 0.92;
      const slab = new THREE.Mesh(
        new THREE.BoxGeometry((H + 0.08) * 2, spandrelH, (H + 0.08) * 2),
        materials.architecturalWhite
      );
      slab.position.set(0, ySlab - spandrelH * 0.15, 0);
      slab.castShadow = true;
      slab.receiveShadow = true;
      localRoot.add(slab);
    }

    // 5. Bold 7-Bay Structural White Concrete Columns (8 columns per side = 28 perimeter columns) & Window Mullions
    const bays = 7;
    const step = (2 * H) / bays; // 6.0m structural bay
    const colGeo = new THREE.BoxGeometry(0.56, eaveY - 0.15, 0.56);
    const mullionGeo = new THREE.BoxGeometry(0.10, eaveY - floorHeights[1] - 0.3, 0.16);

    for (let side = 0; side < 4; side++) {
      const sideGroup = new THREE.Group();
      sideGroup.rotation.y = side * (Math.PI * 0.5);
      localRoot.add(sideGroup);

      for (let i = 0; i <= bays; i++) {
        if (side > 0 && (i === 0 || i === bays)) {
          // Avoid duplicating the 4 corner columns
        } else {
          const x = -H + i * step;
          const col = new THREE.Mesh(colGeo, materials.architecturalWhite);
          col.position.set(x, 0.15 + (eaveY - 0.15) * 0.5, H - 0.24);
          col.castShadow = true;
          col.receiveShadow = true;
          sideGroup.add(col);
        }

        // 4 Vertical white/silver window mullions inside each 6.0m structural bay (ref_203)
        if (i < bays) {
          const bayStart = -H + i * step;
          for (let m = 1; m <= 4; m++) {
            const mx = bayStart + (m / 5) * step;
            const mul = new THREE.Mesh(mullionGeo, materials.architecturalWhite);
            mul.position.set(mx, floorHeights[1] + (eaveY - floorHeights[1] - 0.3) * 0.5, H - 0.48);
            sideGroup.add(mul);
          }
        }
      }
    }

    // 6. West Entrance Portico Canopy & Steps (facing 9:00 Main Driveway, ref_203)
    const westCanopy = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.42, 12.6), materials.architecturalWhite);
    westCanopy.position.set(-H - 1.75, 4.05, 0);
    westCanopy.castShadow = true;
    westCanopy.receiveShadow = true;
    localRoot.add(westCanopy);

    for (const pz of [-5.2, 5.2]) {
      const pCol = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 3.85, 12), materials.architecturalWhite);
      pCol.position.set(-H - 3.2, 2.0, pz);
      pCol.castShadow = true;
      localRoot.add(pCol);
    }

    const westSteps = new THREE.Mesh(new THREE.BoxGeometry(4.4, 0.22, 13.2), materials.terracottaPlaza);
    westSteps.position.set(-H - 1.9, 0.18, 0);
    westSteps.receiveShadow = true;
    localRoot.add(westSteps);

    // West Main Entrance Double Glass Doors, Dark Frame, & Tubular Step Handrails (ref_203!)
    const westDoorFrame = new THREE.Mesh(new THREE.BoxGeometry(0.24, 3.15, 3.60), materials.darkMetal);
    westDoorFrame.position.set(-H + 1.85, 1.72, 0);
    localRoot.add(westDoorFrame);
    const westDoorGlass = new THREE.Mesh(new THREE.BoxGeometry(0.28, 2.92, 3.28), materials.windowGlass);
    westDoorGlass.position.set(-H + 1.83, 1.72, 0);
    localRoot.add(westDoorGlass);
    for (const rz of [-6.2, 6.2]) {
      const railBar = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.06, 0.06), materials.darkMetal);
      railBar.position.set(-H - 1.9, 0.96, rz);
      localRoot.add(railBar);
      for (const rx of [-H - 3.6, -H - 0.3]) {
        const railPost = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.82, 0.06), materials.darkMetal);
        railPost.position.set(rx, 0.56, rz);
        localRoot.add(railPost);
      }
    }

    // 3D Outdoor Split-AC Condenser Units along L2 & L3 Spandrel Ledges (ref_202, ref_203)
    const acBoxGeo = new THREE.BoxGeometry(0.84, 0.58, 0.36);
    for (const side of [-1, 1]) {
      for (const posC of [-15.0, -3.0, 9.0]) {
        const acNS = new THREE.Mesh(acBoxGeo, materials.architecturalWhite);
        acNS.position.set(posC, 8.15, side * (H + 0.18));
        localRoot.add(acNS);
        const acEW = new THREE.Mesh(acBoxGeo, materials.architecturalWhite);
        acEW.rotation.y = Math.PI * 0.5;
        acEW.position.set(side * (H + 0.18), 11.60, posC);
        localRoot.add(acEW);
      }
    }

    // 7. High-Level Flat Reinforced-Concrete Central Roof Deck ("Atap Coran / Dak Beton" at y = 13.95m, ref_174!)
    //    Replaces the inaccurate 4-story open pit so the central well is a solid concrete roof deck!
    const roofOut = H + 1.25; // 22.25m outer eave overhang
    const roofIn = C - 0.85;  // 7.15m inner roof eave opening over the concrete roof deck

    const centralConcreteDeck = new THREE.Mesh(
      new THREE.BoxGeometry((C + 1.2) * 2, 0.45, (C + 1.2) * 2),
      materials.pavingBlock
    );
    centralConcreteDeck.position.set(0, eaveY - 0.32, 0);
    centralConcreteDeck.receiveShadow = true;
    localRoot.add(centralConcreteDeck);

    // White concrete parapet curb & gutter around the inner eave perimeter of the concrete roof deck
    PerpustakaanLamaBuilder._addHollowSquareSlab(
      localRoot,
      materials.architecturalWhite,
      roofIn + 0.38,
      roofIn - 0.18,
      eaveY - 0.25,
      eaveY + 0.16
    );

    // Raised Central Clerestory Monitor & Pyramidal Terracotta Roof Cap sitting ON TOP of the Concrete Roof Deck (ref_174)
    const sk = cfg.skylightHalf; // 4.8m
    const monBasePad = new THREE.Mesh(
      new THREE.BoxGeometry((sk + 0.25) * 2, 0.28, (sk + 0.25) * 2),
      materials.architecturalWhite
    );
    monBasePad.position.set(0, eaveY + 0.02, 0);
    monBasePad.castShadow = true;
    monBasePad.receiveShadow = true;
    localRoot.add(monBasePad);

    const pavWalls = new THREE.Mesh(
      new THREE.BoxGeometry(sk * 2 - 0.3, 0.95, sk * 2 - 0.3),
      materials.silverLouver
    );
    pavWalls.position.set(0, eaveY + 0.62, 0);
    pavWalls.castShadow = true;
    localRoot.add(pavWalls);

    const pavBeam = new THREE.Mesh(
      new THREE.BoxGeometry(sk * 2 + 0.15, 0.26, sk * 2 + 0.15),
      materials.architecturalWhite
    );
    pavBeam.position.set(0, eaveY + 1.15, 0);
    localRoot.add(pavBeam);

    // Pyramidal 3D Stepped Terracotta Roof Cap over the raised central monitor (y = 15.44 .. 17.49m) + 4 hip ridge caps!
    const pyrHalf = sk + 0.55;
    const pyrH = 2.05;
    const pyrBaseY = eaveY + 1.24;
    const pyrApex = [0, pyrBaseY + pyrH, 0];
    const mSW = [-pyrHalf, pyrBaseY, pyrHalf];
    const mSE = [pyrHalf, pyrBaseY, pyrHalf];
    const mNE = [pyrHalf, pyrBaseY, -pyrHalf];
    const mNW = [-pyrHalf, pyrBaseY, -pyrHalf];
    const pPos = [], pNorm = [], pUv = [];
    for (const [cA, cB] of [[mSW, mSE], [mSE, mNE], [mNE, mNW], [mNW, mSW]]) {
      NexusSSoSTKernel.pushThickSteppedRoofQuad(pPos, pNorm, pUv, cA, cB, pyrApex, pyrApex, {
        uvTileMeters: 1.7,
        courses: 7,
        stepHeight: 0.026,
        slabThickness: 0.10,
        eaveNosings: true
      });
    }
    const pyrGeo = new THREE.BufferGeometry();
    pyrGeo.setAttribute('position', new THREE.Float32BufferAttribute(pPos, 3));
    pyrGeo.setAttribute('normal', new THREE.Float32BufferAttribute(pNorm, 3));
    pyrGeo.setAttribute('uv', new THREE.Float32BufferAttribute(pUv, 2));
    pyrGeo.computeBoundingBox();
    pyrGeo.computeBoundingSphere();
    const pyrRoof = new THREE.Mesh(pyrGeo, materials.terracottaRoof);
    pyrRoof.castShadow = true;
    pyrRoof.receiveShadow = true;
    localRoot.add(pyrRoof);

    // 4 Ceramic hip ridge caps on the central pyramidal monitor roof
    const monCorner = sk + 0.52;
    const monApex = new THREE.Vector3(0, pyrBaseY + pyrH, 0);
    for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const cGeo = NexusSSoSTKernel.createRidgeCapStrip(
        [new THREE.Vector3(sx * monCorner, pyrBaseY, sz * monCorner), monApex],
        0.10,
        5
      );
      if (cGeo) {
        const cm = new THREE.Mesh(cGeo, materials.terracottaBrick);
        cm.castShadow = true;
        localRoot.add(cm);
      }
    }

    // 8. Main 4-Wing Courtyard Limasan / Hipped Terracotta Roof (Watertight SSoST)
    const { roofGeo, soffitGeo, ridgePointsList } = NexusSSoSTKernel.createSquareCourtyardLimasanRoof(
      roofOut,
      roofIn,
      eaveY,
      cfg.ridgeY,
      1.85
    );

    const mainRoof = new THREE.Mesh(roofGeo, materials.terracottaRoof);
    mainRoof.castShadow = true;
    mainRoof.receiveShadow = true;
    localRoot.add(mainRoof);

    const mainSoffit = new THREE.Mesh(soffitGeo, materials.creamStucco);
    mainSoffit.receiveShadow = true;
    localRoot.add(mainSoffit);

    // Dark timber/metal outer & inner eave fascia boards
    PerpustakaanLamaBuilder._addHollowSquareWalls(localRoot, materials.darkMetal, roofOut, roofIn, eaveY - 0.24, eaveY + 0.06);

    // 3D Ceramic Ridge Caps ("Nok Genteng") along all 12 ridge, hip, & valley lines
    for (const pts of ridgePointsList) {
      const capGeo = NexusSSoSTKernel.createRidgeCapStrip(pts, 0.13, 6);
      if (capGeo) {
        const cap = new THREE.Mesh(capGeo, materials.terracottaBrick);
        cap.castShadow = true;
        localRoot.add(cap);
      }
    }

    // 4 Corner Lightning Arrestor Spires along ridge loop
    const rRidge = 0.5 * (roofOut + roofIn);
    for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const spire = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.04, 1.6, 6), materials.darkMetal);
      spire.position.set(sx * rRidge, cfg.ridgeY + 0.8, sz * rRidge);
      localRoot.add(spire);
    }

    // 9. NW 1-Story Courtyard Annex Building (13.0m x 7.5m, ref_174 & ref_206)
    const anx = cfg.annexNW;
    const anxGroup = new THREE.Group();
    anxGroup.position.set(anx.x, 0, anx.z);
    group.add(anxGroup);

    const anxBody = new THREE.Mesh(new THREE.BoxGeometry(anx.w, anx.h - 0.35, anx.d), materials.architecturalWhite);
    anxBody.position.set(0, (anx.h - 0.35) * 0.5, 0);
    anxBody.castShadow = true;
    anxBody.receiveShadow = true;
    anxGroup.add(anxBody);

    const anxWin = new THREE.Mesh(new THREE.BoxGeometry(anx.w + 0.08, 1.45, anx.d - 1.4), materials.windowGlass);
    anxWin.position.set(0, 2.25, 0);
    anxGroup.add(anxWin);

    const anxRoof = new THREE.Mesh(new THREE.BoxGeometry(anx.w + 0.9, 0.38, anx.d + 0.9), materials.pavingBlock);
    anxRoof.position.set(0, anx.h, 0);
    anxRoof.castShadow = true;
    anxRoof.receiveShadow = true;
    anxGroup.add(anxRoof);

    return group;
  }

  static _addHollowSquareSlab(parent, mat, halfOut, halfIn, yBot, yTop) {
    const h = yTop - yBot;
    const yc = 0.5 * (yBot + yTop);
    const bandW = halfOut - halfIn;

    for (const signZ of [-1, 1]) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(halfOut * 2, h, bandW), mat);
      m.position.set(0, yc, signZ * (halfIn + bandW * 0.5));
      m.castShadow = true;
      m.receiveShadow = true;
      parent.add(m);
    }
    for (const signX of [-1, 1]) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(bandW, h, halfIn * 2), mat);
      m.position.set(signX * (halfIn + bandW * 0.5), yc, 0);
      m.castShadow = true;
      m.receiveShadow = true;
      parent.add(m);
    }
  }

  static _addHollowSquareWalls(parent, mat, halfOut, halfIn, yBot, yTop) {
    const h = yTop - yBot;
    const yc = 0.5 * (yBot + yTop);
    const t = 0.22;

    for (const r of [halfOut, halfIn]) {
      for (const sign of [-1, 1]) {
        const ns = new THREE.Mesh(new THREE.BoxGeometry(r * 2, h, t), mat);
        ns.position.set(0, yc, sign * r);
        ns.castShadow = true;
        ns.receiveShadow = true;
        parent.add(ns);

        const ew = new THREE.Mesh(new THREE.BoxGeometry(t, h, r * 2), mat);
        ew.position.set(sign * r, yc, 0);
        ew.castShadow = true;
        ew.receiveShadow = true;
        parent.add(ew);
      }
    }
  }
}
