import * as THREE from 'three';
import { ProceduralTextureFactory } from '../../core/ProceduralTextureFactory.js';
import { UI_DPPD_MASTER_PLAN } from '../../world-gis/MasterPlanConfig.js';

const DEG = Math.PI / 180;

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (True-Color Photo-Realistic Edition)
 * ./library/gedung-dppd/SecondarySubdomains.js
 *
 * Stage 4 Architectural Subdomains & Forecourt Details for Gedung DPPD UI:
 * - 11-Bay / 12-Pane L2 Ribbon Window Band beneath Central Steep Gable + Mini White Satellite Dish
 * - L2 Wing Colonnade Square Cream Columns + Prominent Split-AC Outdoor Condenser Units w/ Fan Grilles (ref_084)
 * - Full-Width L1 Terracotta Portico Canopy (with Central Gablet, 4 Cream Columns, Double Doors & Unoccluded Wall Signage)
 * - Authentic Forecourt Stair & Ramp Layout (ref_184): Central 6-Step Grey Stone Stair, Right Terracotta ADA Ramp,
 *   Left & Right Wide Terracotta Stair Flights, UV-Mapped Blue ADA Wheelchair Stall w/ Hatching, Traffic Cone & Stone Bollard
 * - UI-Huawei Brass Totem (Left) + Twin Gold Banner Ladder Towers, Hanging Cloth Banner & Stainless Campus Map Plaque (Right)
 */
export class DPPDSecondarySubdomains {
  static build(localGroup, materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.dppd;
    const eaveY = cfg.eaveY; // 6.45m
    const l1TopY = cfg.l1TopY; // 3.72m

    // 1. Central Gable Front L2 Window Band (11 symmetrical bays across x ∈ [-10.45m, +10.45m] at z = 59.1m, ref_084)
    const frontWallL2 = new THREE.Mesh(new THREE.BoxGeometry(21.6, 0.85, 0.5), materials.creamStucco);
    frontWallL2.position.set(0, l1TopY + 0.42, 59.1);
    frontWallL2.castShadow = true;
    localGroup.add(frontWallL2);

    const winBand = new THREE.Mesh(new THREE.BoxGeometry(20.9, 1.88, 0.24), materials.windowGlass);
    winBand.position.set(0, l1TopY + 1.78, 59.05);
    localGroup.add(winBand);

    // 12 Crisp Vertical Cream Piers forming 11 Symmetrical Window Bays (5 Left + 1 Center + 5 Right, ref_084)
    for (let i = 0; i <= 11; i++) {
      const px = -10.45 + (i / 11) * 20.9;
      const pier = new THREE.Mesh(new THREE.BoxGeometry(0.24, 1.96, 0.36), materials.creamStucco);
      pier.position.set(px, l1TopY + 1.78, 59.12);
      pier.castShadow = true;
      localGroup.add(pier);
    }

    // Small White Round Satellite Mini-Dish w/ Dark Bracket on pier #2 from left (ref_084, ref_184)
    const dishArm = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 0.24), materials.darkMetal);
    dishArm.position.set(-6.65, l1TopY + 0.92, 59.26);
    localGroup.add(dishArm);
    const dish = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.15, 0.05, 14), materials.architecturalWhite);
    dish.rotation.x = Math.PI * 0.44;
    dish.position.set(-6.65, l1TopY + 0.94, 59.38);
    localGroup.add(dish);

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
    const zL1Wall = 58.65;
    const l1Wall = new THREE.Mesh(new THREE.BoxGeometry(22.2, l1TopY - 0.15, 0.5), materials.creamStucco);
    l1Wall.position.set(0, 0.15 + (l1TopY - 0.15) * 0.5, zL1Wall);
    l1Wall.receiveShadow = true;
    localGroup.add(l1Wall);

    // Central Dark-Framed Glass Double Entrance Doors + Symmetrical Side Slit Windows (ref_184)
    const doorFrame = new THREE.Mesh(new THREE.BoxGeometry(3.2, 2.75, 0.16), materials.darkMetal);
    doorFrame.position.set(0, 0.45 + 2.75 * 0.5, zL1Wall + 0.24);
    localGroup.add(doorFrame);

    const doorGlass = new THREE.Mesh(new THREE.BoxGeometry(2.92, 2.58, 0.20), materials.windowGlass);
    doorGlass.position.set(0, 0.45 + 2.58 * 0.5, zL1Wall + 0.25);
    localGroup.add(doorGlass);

    for (const sx of [-4.8, 4.8]) {
      const slit = new THREE.Mesh(new THREE.BoxGeometry(0.68, 2.15, 0.16), materials.windowGlass);
      slit.position.set(sx, 1.75, zL1Wall + 0.24);
      localGroup.add(slit);
    }

    // Crisp Wall Signage right of doors at eye level (y = 2.02m so canopy never blocks it, ref_184!)
    const ksdpSign = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(2.15, 0.64, 40, 420, 440, 104),
      materials.signAtlasMat
    );
    ksdpSign.position.set(2.95, 2.02, zL1Wall + 0.28);
    localGroup.add(ksdpSign);

    const dppdPlaque = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(2.15, 0.44, 530, 428, 474, 88),
      materials.signAtlasMat
    );
    dppdPlaque.position.set(-2.95, 2.02, zL1Wall + 0.28);
    localGroup.add(dppdPlaque);

    // 4. Full-Width L1 Terracotta Lean-To Portico Canopy + Central Gablet + 4 Slender Cream Columns (ref_084, ref_184)
    const canW = 21.8;
    const canZIn = zL1Wall + 0.1;
    const canZOut = 62.35;
    const canDepth = canZOut - canZIn;

    const canRoof = new THREE.Mesh(new THREE.BoxGeometry(canW, 0.18, canDepth + 0.25), materials.terracottaRoof);
    canRoof.position.set(0, 3.62, 0.5 * (canZIn + canZOut));
    canRoof.rotation.x = 0.16;
    canRoof.castShadow = true;
    canRoof.receiveShadow = true;
    localGroup.add(canRoof);

    const canSoffit = new THREE.Mesh(new THREE.BoxGeometry(canW - 0.2, 0.10, canDepth), materials.creamStucco);
    canSoffit.position.set(0, 3.30, 0.5 * (canZIn + canZOut));
    localGroup.add(canSoffit);

    const canFascia = new THREE.Mesh(new THREE.BoxGeometry(canW + 0.1, 0.22, 0.12), materials.darkMetal);
    canFascia.position.set(0, 3.34, canZOut + 0.06);
    localGroup.add(canFascia);

    // Central Entrance Pediment Gablet directly over main doors (ref_084)
    for (const sgn of [-1, 1]) {
      const gb = new THREE.Mesh(new THREE.BoxGeometry(2.35, 0.16, canDepth + 0.3), materials.terracottaRoof);
      gb.position.set(sgn * 1.05, 3.78, 0.5 * (canZIn + canZOut));
      gb.rotation.z = -sgn * 0.26;
      gb.castShadow = true;
      localGroup.add(gb);
    }

    // 4 Slender Cream Portico Columns supporting L1 Canopy
    for (const cx of [-8.4, -2.7, 2.7, 8.4]) {
      const pCol = new THREE.Mesh(new THREE.BoxGeometry(0.30, 3.05, 0.30), materials.creamStucco);
      pCol.position.set(cx, 0.45 + 3.05 * 0.5, canZOut - 0.38);
      pCol.castShadow = true;
      localGroup.add(pCol);
    }

    // 5. Terracotta-Tiled Upper Porch Deck, Authentic Central Grey Steps + Right Terracotta Ramp + Flanking Stairs (ref_184)
    const porchDeck = new THREE.Mesh(new THREE.BoxGeometry(21.8, 0.38, 3.9), materials.terracottaPlaza);
    porchDeck.position.set(0, 0.29, 60.5);
    porchDeck.receiveShadow = true;
    localGroup.add(porchDeck);

    // Forecourt Weathered Grey Paving Apron + Concrete Bridge Driveway (ref_184)
    const apron = new THREE.Mesh(new THREE.BoxGeometry(22.6, 0.14, 7.2), materials.pavingBlock);
    apron.position.set(0, 0.12, 65.8);
    apron.receiveShadow = true;
    localGroup.add(apron);

    // A. Central Narrow 6-Step Grey Stone Stairway (x = -0.85m, w = 1.65m — ref_184 center!)
    for (let st = 0; st < 6; st++) {
      const stepH = 0.056 * (6 - st);
      const step = new THREE.Mesh(new THREE.BoxGeometry(1.65, stepH, 0.38), materials.stoneRetaining);
      step.position.set(-0.85, 0.15 + stepH * 0.5, 62.50 + st * 0.36);
      step.receiveShadow = true;
      localGroup.add(step);
    }

    // B. Sloped Terracotta ADA Ramp immediately right of the central grey stairs (x = +1.65m, w = 3.1m — ref_184!)
    const ramp = new THREE.Mesh(new THREE.BoxGeometry(3.1, 0.12, 3.2), materials.terracottaPlaza);
    ramp.position.set(1.65, 0.28, 63.85);
    ramp.rotation.x = 0.095;
    ramp.receiveShadow = true;
    localGroup.add(ramp);

    // 3 Low Terracotta Parapet Cheek Walls dividing Left Stairs | Grey Center Stairs | Terracotta Ramp | Right Stairs (ref_184)
    for (const rx of [-1.78, 0.02, 3.28]) {
      const rCurb = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.44, 3.25), materials.terracottaBrick);
      rCurb.position.set(rx, 0.36, 63.85);
      rCurb.castShadow = true;
      localGroup.add(rCurb);
    }

    // C. Left Wide 6-Step Terracotta Stair Flight (x = -6.2m, w = 8.4m) & Right Wide 6-Step Flight (x = +6.9m, w = 6.8m)
    for (const [sx, sw] of [[-6.15, 8.4], [6.85, 6.8]]) {
      for (let st = 0; st < 6; st++) {
        const stepH = 0.056 * (6 - st);
        const step = new THREE.Mesh(new THREE.BoxGeometry(sw, stepH, 0.38), materials.terracottaPlaza);
        step.position.set(sx, 0.15 + stepH * 0.5, 62.50 + st * 0.36);
        step.receiveShadow = true;
        localGroup.add(step);
      }
    }

    // 6. Left Forecourt: Authentic UV-Mapped Blue ADA Wheelchair Parking Stall + Orange Cone + Stone Pedestal + Huawei Totem (ref_184)
    const blueBay = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasBox(4.4, 0.04, 4.6, 852, 652, 86, 124),
      materials.signAtlasMat
    );
    blueBay.position.set(-6.6, 0.18, 66.65);
    blueBay.receiveShadow = true;
    localGroup.add(blueBay);

    // Orange Traffic Cone at top-right corner of the Blue ADA Stall (ref_184)
    const coneBase = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.05, 0.36), materials.terracottaRoof);
    coneBase.position.set(-4.65, 0.22, 65.1);
    localGroup.add(coneBase);
    const cone = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.14, 0.58, 10), materials.terracottaRoof);
    cone.position.set(-4.65, 0.52, 65.1);
    localGroup.add(cone);

    // Short Carved Stone Bollard / Pedestal at bottom-left corner of Blue ADA Stall (ref_184)
    const stonePed = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.38, 0.56, 10), materials.stoneRetaining);
    stonePed.position.set(-10.2, 0.44, 67.2);
    stonePed.castShadow = true;
    localGroup.add(stonePed);

    // Brushed Gold/Brass UI-Huawei Training Center Monolith Totem (Tile 5 in Signage Atlas!)
    const huaweiBody = new THREE.Mesh(new THREE.BoxGeometry(1.18, 2.35, 0.22), materials.goldAccent);
    huaweiBody.position.set(-8.6, 0.35 + 2.35 * 0.5, 67.15);
    huaweiBody.rotation.y = 0.14;
    huaweiBody.castShadow = true;
    localGroup.add(huaweiBody);

    const huaweiFace = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(1.14, 2.28, 340, 528, 340, 496),
      materials.signAtlasMat
    );
    huaweiFace.position.set(-8.6 + Math.sin(0.14) * 0.12, 0.35 + 2.35 * 0.5, 67.15 + Math.cos(0.14) * 0.12);
    huaweiFace.rotation.y = 0.14;
    localGroup.add(huaweiFace);

    // 7. Right Forecourt: Twin Gold Open-Grid Ladder Banner Towers + Hanging Cloth Banner + Campus Map Plaque (ref_084, ref_184)
    for (const [idx, bx] of [[0, 6.0], [1, 10.6]]) {
      const basePad = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.26, 0.52), materials.darkMetal);
      basePad.position.set(bx, 0.25, 66.85);
      localGroup.add(basePad);

      for (const lx of [-0.22, 0.22]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.055, 5.2, 0.055), materials.goldAccent);
        leg.position.set(bx + lx, 0.35 + 2.6, 66.85);
        leg.castShadow = true;
        localGroup.add(leg);
      }
      for (let r = 0; r < 8; r++) {
        const rung = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.04, 0.045), materials.goldAccent);
        rung.position.set(bx, 0.95 + r * 0.62, 66.85);
        localGroup.add(rung);
      }

      // Vertical White/Blue/Yellow Hanging Cloth Banner on the Right Banner Frame (ref_084)
      if (idx === 1) {
        const clothBanner = new THREE.Mesh(
          ProceduralTextureFactory.createAtlasBox(0.56, 3.85, 0.06, 960, 776, 64, 248),
          materials.signAtlasMat
        );
        clothBanner.position.set(bx - 0.14, 2.35, 66.90);
        clothBanner.rotation.z = -0.04;
        localGroup.add(clothBanner);
      }
    }

    // Stainless-Steel Framed Campus Map Directory Signboard (ref_084, ref_184)
    const mapPost = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 1.35, 10), materials.silverLouver);
    mapPost.position.set(8.4, 0.80, 66.85);
    localGroup.add(mapPost);

    const mapFrame = new THREE.Mesh(new THREE.BoxGeometry(1.48, 1.05, 0.12), materials.silverLouver);
    mapFrame.position.set(8.4, 1.85, 66.85);
    mapFrame.castShadow = true;
    localGroup.add(mapFrame);

    const mapFace = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(1.38, 0.96, 680, 776, 184, 248),
      materials.signAtlasMat
    );
    mapFace.position.set(8.4, 1.85, 66.92);
    localGroup.add(mapFace);

    // Ornamental Grass Bush on the Right Berm Slope (ref_084 center-right!)
    const bush = new THREE.Mesh(new THREE.DodecahedronGeometry(0.88, 1), materials.lushGrass);
    bush.scale.set(1.15, 0.95, 1.15);
    bush.position.set(15.2, 1.85, 61.5);
    bush.castShadow = true;
    localGroup.add(bush);
  }
}
