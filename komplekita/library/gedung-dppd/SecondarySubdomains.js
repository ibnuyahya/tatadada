import * as THREE from 'three';
import { ProceduralTextureFactory } from '../../core/ProceduralTextureFactory.js';
import { UI_DPPD_MASTER_PLAN } from '../../world-gis/MasterPlanConfig.js';

const DEG = Math.PI / 180;

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./library/gedung-dppd/SecondarySubdomains.js
 *
 * Stage 4 Architectural Subdomains & Forecourt Details for Gedung DPPD UI:
 * - 11-Bay / 12-Pane L2 Ribbon Window Band beneath Central Steep Gable + Mini White Satellite Dish
 * - L2 Wing Colonnade Square Cream Columns + Rhythmic Split-AC Outdoor Condenser Units (ref_084)
 * - Full-Width L1 Terracotta Portico Canopy (with Central Gablet, 4 Cream Columns, Double Doors & Wall Signage)
 * - Terracotta-Tiled Forecourt Plaza, Flanking 5-Step Stairs, Central ADA Ramp, & Blue Parking Stall
 * - UI-Huawei Brass Totem (Left) + Twin Gold Banner Ladder Towers & Stainless Campus Map Plaque (Right)
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

    // Small White Round Satellite Mini-Dish on pier #2 from left (ref_084, ref_184)
    const dish = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.28, 0.08, 14), materials.architecturalWhite);
    dish.rotation.x = Math.PI * 0.42;
    dish.position.set(-6.65, l1TopY + 1.05, 59.38);
    localGroup.add(dish);

    // 2. L2 Wing Loggia Colonnade Columns + Split-AC Outdoor Units (NW & SE Wings, ref_079, ref_084, ref_194)
    const rCol = cfg.rOuter - 0.22; // 57.28m
    const colH = eaveY - (l1TopY + 0.70);
    const colGeo = new THREE.BoxGeometry(0.34, colH, 0.34);
    const acGeo = new THREE.BoxGeometry(0.78, 0.54, 0.32);

    // NW Wing: 10 open colonnade bays; SE Wing: 10 open colonnade bays
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

        // Place white split-AC condenser units in select bays matching ref_084 (Bays 1, 2, 4, 6, 7, 9)
        if (i < cols - 1 && [0, 1, 3, 5, 6, 8].includes(i)) {
          const midA = sgn * (aStart + ((i + 0.5) / (cols - 1)) * (aEnd - aStart)) * DEG;
          const ac = new THREE.Mesh(acGeo, materials.architecturalWhite);
          ac.position.set((rCol - 0.15) * Math.sin(midA), l1TopY + 0.98, (rCol - 0.15) * Math.cos(midA));
          ac.rotation.y = midA;
          ac.castShadow = true;
          localGroup.add(ac);
        }
      }
    }

    // 3. Recessed Ground-Floor (L1) Main Entrance Wall, Glass Double Doors, & Wall Signage (ref_084, ref_184)
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

    for (const sx of [-4.6, 4.6]) {
      const slit = new THREE.Mesh(new THREE.BoxGeometry(0.65, 2.15, 0.16), materials.windowGlass);
      slit.position.set(sx, 1.75, zL1Wall + 0.24);
      localGroup.add(slit);
    }

    // Crisp Wall Signage right of doors: "GEDUNG / KSDP UI" + "GEDUNG DPPD UI" Plaque (Tile 3 in Atlas!)
    const ksdpSign = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(2.1, 0.62, 40, 420, 440, 104),
      materials.signAtlasMat
    );
    ksdpSign.position.set(2.85, 2.38, zL1Wall + 0.27);
    localGroup.add(ksdpSign);

    const dppdPlaque = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(2.2, 0.44, 530, 428, 474, 88),
      materials.signAtlasMat
    );
    dppdPlaque.position.set(-2.85, 2.25, zL1Wall + 0.27);
    localGroup.add(dppdPlaque);

    // 4. Full-Width L1 Terracotta Lean-To Portico Canopy + Central Gablet + 4 Slender Cream Columns (ref_084, ref_184)
    const canW = 21.8;
    const canZIn = zL1Wall + 0.1;
    const canZOut = 62.35;
    const canDepth = canZOut - canZIn;

    // Pitched terracotta lean-to roof slab
    const canRoof = new THREE.Mesh(new THREE.BoxGeometry(canW, 0.18, canDepth + 0.25), materials.terracottaRoof);
    canRoof.position.set(0, 3.62, 0.5 * (canZIn + canZOut));
    canRoof.rotation.x = 0.16;
    canRoof.castShadow = true;
    canRoof.receiveShadow = true;
    localGroup.add(canRoof);

    const canSoffit = new THREE.Mesh(new THREE.BoxGeometry(canW - 0.2, 0.10, canDepth), materials.creamStucco);
    canSoffit.position.set(0, 3.30, 0.5 * (canZIn + canZOut));
    localGroup.add(canSoffit);

    const canFascia = new THREE.Mesh(new THREE.BoxGeometry(canW + 0.1, 0.24, 0.12), materials.darkMetal);
    canFascia.position.set(0, 3.34, canZOut + 0.06);
    localGroup.add(canFascia);

    // Subtle Central Entrance Pediment Gablet directly over main doors (ref_084)
    for (const sgn of [-1, 1]) {
      const gb = new THREE.Mesh(new THREE.BoxGeometry(2.35, 0.16, canDepth + 0.3), materials.terracottaRoof);
      gb.position.set(sgn * 1.05, 3.78, 0.5 * (canZIn + canZOut));
      gb.rotation.z = -sgn * 0.26;
      gb.castShadow = true;
      localGroup.add(gb);
    }

    // 4 Slender Cream Portico Columns supporting L1 Canopy
    for (const cx of [-8.4, -2.7, 2.7, 8.4]) {
      const pCol = new THREE.Mesh(new THREE.BoxGeometry(0.32, 3.05, 0.32), materials.creamStucco);
      pCol.position.set(cx, 0.45 + 3.05 * 0.5, canZOut - 0.38);
      pCol.castShadow = true;
      localGroup.add(pCol);
    }

    // 5. Terracotta-Tiled Upper Porch Deck, Flanking 5-Step Stairs, & Central ADA Ramp (ref_084, ref_184)
    const porchDeck = new THREE.Mesh(new THREE.BoxGeometry(21.8, 0.38, 3.9), materials.terracottaPlaza);
    porchDeck.position.set(0, 0.29, 60.5);
    porchDeck.receiveShadow = true;
    localGroup.add(porchDeck);

    // Forecourt Grey Paving Apron extending out to perimeter bridge (z = 62.4 .. 68.8m)
    const apron = new THREE.Mesh(new THREE.BoxGeometry(22.4, 0.14, 6.8), materials.pavingBlock);
    apron.position.set(0, 0.12, 65.6);
    apron.receiveShadow = true;
    localGroup.add(apron);

    // Left & Right 5-Step Terracotta Stair Flights + Central Sloped Terracotta ADA Ramp
    for (const sx of [-6.4, 6.4]) {
      for (let st = 0; st < 5; st++) {
        const stepH = 0.065 * (5 - st);
        const step = new THREE.Mesh(new THREE.BoxGeometry(6.8, stepH, 0.42), materials.terracottaPlaza);
        step.position.set(sx, 0.15 + stepH * 0.5, 62.55 + st * 0.40);
        step.receiveShadow = true;
        localGroup.add(step);
      }
    }

    // Central ADA Wheelchair Ramp + Low Red Parapet Cheek Walls (ref_084, ref_184)
    const ramp = new THREE.Mesh(new THREE.BoxGeometry(4.4, 0.12, 3.2), materials.terracottaPlaza);
    ramp.position.set(0, 0.28, 63.85);
    ramp.rotation.x = 0.095;
    ramp.receiveShadow = true;
    localGroup.add(ramp);

    for (const rx of [-2.3, 2.3]) {
      const rCurb = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.46, 3.25), materials.terracottaBrick);
      rCurb.position.set(rx, 0.38, 63.85);
      rCurb.castShadow = true;
      localGroup.add(rCurb);
    }

    // 6. Left Forecourt: Sky-Blue ADA Accessible Parking Bay + UI-Huawei Brass Totem (ref_184, ref_185)
    const blueBay = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.04, 4.8), materials.adaBlueBay);
    blueBay.position.set(-8.6, 0.18, 66.2);
    blueBay.receiveShadow = true;
    localGroup.add(blueBay);

    // Brushed Gold/Brass UI-Huawei Training Center Monolith Totem (Tile 5 in Signage Atlas!)
    const huaweiBody = new THREE.Mesh(new THREE.BoxGeometry(1.18, 2.35, 0.22), materials.goldAccent);
    huaweiBody.position.set(-10.8, 0.35 + 2.35 * 0.5, 66.8);
    huaweiBody.rotation.y = 0.18;
    huaweiBody.castShadow = true;
    localGroup.add(huaweiBody);

    const huaweiFace = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(1.14, 2.28, 340, 528, 340, 496),
      materials.signAtlasMat
    );
    huaweiFace.position.set(-10.8 + Math.sin(0.18) * 0.12, 0.35 + 2.35 * 0.5, 66.8 + Math.cos(0.18) * 0.12);
    huaweiFace.rotation.y = 0.18;
    localGroup.add(huaweiFace);

    // 7. Right Forecourt: Twin Gold Open-Grid Ladder Banner Towers + Stainless Campus Map Plaque (ref_084, ref_184)
    for (const bx of [8.6, 12.2]) {
      const basePad = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.28, 0.55), materials.darkMetal);
      basePad.position.set(bx, 0.26, 66.9);
      localGroup.add(basePad);

      for (const lx of [-0.22, 0.22]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.06, 5.2, 0.06), materials.goldAccent);
        leg.position.set(bx + lx, 0.35 + 2.6, 66.9);
        leg.castShadow = true;
        localGroup.add(leg);
      }
      for (let r = 0; r < 8; r++) {
        const rung = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.045, 0.05), materials.goldAccent);
        rung.position.set(bx, 0.95 + r * 0.62, 66.9);
        localGroup.add(rung);
      }
    }

    // Stainless-Steel Framed Campus Map Directory Signboard centered between the two gold towers (Tile 7 left!)
    const mapPost = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 1.35, 10), materials.silverLouver);
    mapPost.position.set(10.4, 0.80, 66.9);
    localGroup.add(mapPost);

    const mapFrame = new THREE.Mesh(new THREE.BoxGeometry(1.48, 1.05, 0.12), materials.silverLouver);
    mapFrame.position.set(10.4, 1.85, 66.9);
    mapFrame.castShadow = true;
    localGroup.add(mapFrame);

    const mapFace = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(1.38, 0.96, 680, 776, 184, 248),
      materials.signAtlasMat
    );
    mapFace.position.set(10.4, 1.85, 66.97);
    localGroup.add(mapFace);
  }
}
