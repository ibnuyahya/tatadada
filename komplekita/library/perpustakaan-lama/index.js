import * as THREE from 'three';
import { NexusSSoSTKernel } from '../../core/MasterSurfaceKernels.js';
import { UI_DPPD_MASTER_PLAN } from '../../world-gis/MasterPlanConfig.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./library/perpustakaan-lama/index.js
 *
 * Class-A Procedural Building Module: ITTC Perpustakaan Lama Universitas Indonesia
 * Ground-Truth References: ref_096-100, ref_136-142, ref_174, ref_202-204, ref_206.
 *
 * Key Architectural Features:
 * - Strictly Cardinal N-S / E-W 42.0m x 42.0m Square Plan (NOT rotated 45°!)
 * - 16.0m x 16.0m Open Central Courtyard with Nested 9.6m x 9.6m Pyramidal Skylight Pavilion (ref_174)
 * - 4 Full Stories with Bold White Reinforced-Concrete Structural Column Grid (7 bays per side) & Spandrels
 * - Recessed L1 Colonnade + West Portico Canopy facing the 9:00 Main Driveway + L2-L4 Mullioned Ribbon Windows
 * - Watertight Courtyard Limasan / Hipped Terracotta Roof (eaveY = 14.20m, ridgeY = 18.80m) + Ridge Caps
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

    const H = cfg.outerHalf;      // 21.0m
    const C = cfg.courtyardHalf;  // 8.0m
    const eaveY = cfg.eaveY;      // 14.20m
    const floorHeights = [0.15, 3.85, 7.30, 10.75, eaveY];

    // 1. Concrete Plinth Base Ring (42.8m x 42.8m minus 15.6m courtyard)
    PerpustakaanLamaBuilder._addHollowSquareSlab(localRoot, materials.pavingBlock, H + 0.4, C - 0.2, -0.2, 0.24);

    // 2. Recessed L1 Core Walls (inset 2.1m behind outer colonnade columns -> half = 18.9m)
    PerpustakaanLamaBuilder._addHollowSquareWalls(
      localRoot,
      materials.windowGlass,
      H - 2.1,
      C + 0.35,
      0.24,
      floorHeights[1]
    );

    // 3. Upper Stories L2, L3, L4 Recessed Window Curtain Walls (half = H - 0.55m) + Inner Courtyard Glass
    PerpustakaanLamaBuilder._addHollowSquareWalls(
      localRoot,
      materials.windowGlass,
      H - 0.55,
      C + 0.35,
      floorHeights[1],
      eaveY - 0.35
    );

    // 4. Crisp Projecting White Horizontal Spandrel & Sill Slabs at L2, L3, L4, & Eave Ring Beam
    for (let f = 1; f <= 4; f++) {
      const ySlab = floorHeights[f];
      const spandrelH = f === 4 ? 0.55 : 0.92;
      PerpustakaanLamaBuilder._addHollowSquareSlab(
        localRoot,
        materials.architecturalWhite,
        H + 0.08,
        C,
        ySlab - spandrelH * 0.65,
        ySlab + spandrelH * 0.35
      );
    }

    // 5. Bold 7-Bay Structural White Concrete Columns (8 columns per side = 28 perimeter columns + 16 courtyard piers)
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

      // Inner courtyard structural columns (5 columns per side)
      for (let ci = 0; ci < 4; ci++) {
        const cx = -C + ci * ((2 * C) / 4);
        const cCol = new THREE.Mesh(
          new THREE.BoxGeometry(0.42, eaveY - 0.15, 0.42),
          materials.architecturalWhite
        );
        cCol.position.set(cx, 0.15 + (eaveY - 0.15) * 0.5, C + 0.18);
        cCol.castShadow = true;
        sideGroup.add(cCol);
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

    // 7. Nested Central Courtyard Skylight Pavilion (9.6m x 9.6m inside 16m x 16m courtyard, ref_174)
    const sk = cfg.skylightHalf; // 4.8m
    const pavWalls = new THREE.Mesh(new THREE.BoxGeometry(sk * 2 - 0.6, 3.9, sk * 2 - 0.6), materials.windowGlass);
    pavWalls.position.set(0, 2.1, 0);
    localRoot.add(pavWalls);

    const pavBeam = new THREE.Mesh(new THREE.BoxGeometry(sk * 2 + 0.2, 0.36, sk * 2 + 0.2), materials.architecturalWhite);
    pavBeam.position.set(0, 4.12, 0);
    localRoot.add(pavBeam);

    // Pyramidal terracotta roof cap over the courtyard pavilion
    const pyrGeo = new THREE.ConeGeometry((sk + 0.65) * Math.SQRT2, 2.75, 4, 1);
    pyrGeo.rotateY(Math.PI * 0.25);
    const pyrRoof = new THREE.Mesh(pyrGeo, materials.terracottaRoof);
    pyrRoof.position.set(0, 4.28 + 2.75 * 0.5, 0);
    pyrRoof.castShadow = true;
    pyrRoof.receiveShadow = true;
    localRoot.add(pyrRoof);

    // 8. Main 4-Wing Courtyard Limasan / Hipped Terracotta Roof (Watertight SSoST)
    const roofOut = H + 1.25; // 22.25m eave overhang
    const roofIn = C - 0.85;  // 7.15m inner courtyard eave overhang
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

    // North & South full-width strips
    for (const signZ of [-1, 1]) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(halfOut * 2, h, bandW), mat);
      m.position.set(0, yc, signZ * (halfIn + bandW * 0.5));
      m.castShadow = true;
      m.receiveShadow = true;
      parent.add(m);
    }
    // East & West strips between North & South
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
