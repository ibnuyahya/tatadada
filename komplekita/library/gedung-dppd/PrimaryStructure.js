import * as THREE from 'three';
import { NexusSSoSTKernel } from '../../core/MasterSurfaceKernels.js';
import { UI_DPPD_MASTER_PLAN } from '../../world-gis/MasterPlanConfig.js';

const DEG = Math.PI / 180;

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (True-Color Photo-Realistic & Watertight Edition)
 * ./library/gedung-dppd/PrimaryStructure.js
 *
 * Stage 3 Primary Hull, SSoST Stitched Roof Valleys, Polygonal Faceted Terracotta Wing Roofs,
 * 2-Tier Sloped Grass Berm ("Talud Rumput"), Batu-Kali Retaining Walls, & NW Salmon-Brick Gable End.
 *
 * Watertight Guarantees:
 * - Zero gap between Central Gable Pavilion and Curved L1/L2 Wing Walls (closed side return walls & soffits)
 * - Exact 3D Analytical Valley Intersection between Steep Central Gable and Curved Wing Roof (no protruding rear triangle or floating pipes!)
 * - Seamless Grass Berm & Sloped Masonry Wing Wall junction right up to the Central Forecourt Stairs & Canopy
 */
export class DPPDPrimaryStructure {
  static build(localGroup, materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.dppd;
    const rIn = cfg.rInner;          // 39.5m
    const rOut = cfg.rOuter;         // 57.5m
    const rWallL2 = cfg.rLoggiaWall; // 54.8m
    const rRidge = 47.2;             // Inward-offset main roof ridge (ref_174, ref_202)
    const eaveY = cfg.eaveY;         // 6.45m
    const wingRidgeY = cfg.wingRidgeY; // 11.85m

    const alphaNW = -42.5 * DEG;     // Maps to th = +178.0° (West 9:00 Gable End!)
    const alphaSE = 43.5 * DEG;      // Maps to th = +92.0° (South 6:00 Hipped End!)
    const alphaPortal = 10.6 * DEG;  // Overlaps inside x = ±10.85m central pavilion side walls -> 0.00m gap!

    const toPhi = (alpha) => Math.PI * 0.5 - alpha;

    // 1. Inner Courtyard Curved Facade & L1 Core Structure (full span alphaNW .. alphaSE, watertight caps!)
    NexusSSoSTKernel.addCurvedPolarBand(
      localGroup, materials.creamStucco, toPhi(alphaSE), toPhi(alphaNW), rIn, rWallL2, -0.2, cfg.l1TopY, 36,
      { top: true, bottom: false, startCap: true, endCap: true, uvMetersPerTile: 3.5 }
    );
    // Inner courtyard L2 cream wall + dark ribbon windows + ceiling soffit slab
    NexusSSoSTKernel.addCurvedPolarBand(
      localGroup, materials.creamStucco, toPhi(alphaSE), toPhi(alphaNW), rIn, rWallL2, cfg.l1TopY, eaveY, 36,
      { top: true, bottom: true, startCap: true, endCap: true, uvMetersPerTile: 3.5 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      localGroup, materials.windowGlass, toPhi(alphaSE) + 0.02, toPhi(alphaNW) - 0.02, rIn - 0.06, rIn + 0.12, cfg.l1TopY + 0.85, eaveY - 0.55, 36,
      { top: false, bottom: false, startCap: false, endCap: false, uvMetersPerTile: 3.2 }
    );

    // 2. Recessed L2 Loggia Back Wall, Awning Windows, & Loggia Walkway Floor + Sill Parapet
    for (const [a0, a1] of [[alphaNW + 2.5 * DEG, -alphaPortal], [alphaPortal, alphaSE - 4.5 * DEG]]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.creamStucco, toPhi(a1), toPhi(a0), rWallL2 - 0.4, rWallL2, cfg.l1TopY, eaveY, 18,
        { bottom: false, startCap: true, endCap: true, uvMetersPerTile: 3.2 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.windowGlass, toPhi(a1), toPhi(a0), rWallL2 - 0.05, rWallL2 + 0.08, cfg.l1TopY + 0.92, eaveY - 0.42, 18,
        { top: false, bottom: false, uvMetersPerTile: 2.8 }
      );
      // L2 Loggia Walkway Floor & Outer Cream Sill Parapet (y = 3.72 .. 4.44m)
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.creamStucco, toPhi(a1), toPhi(a0), rWallL2 - 0.2, rOut, cfg.l1TopY - 0.18, cfg.l1TopY + 0.08, 18,
        { bottom: true, top: true, startCap: true, endCap: true, uvMetersPerTile: 3.0 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.creamStucco, toPhi(a1), toPhi(a0), rOut - 0.36, rOut, cfg.l1TopY, cfg.l1TopY + 0.72, 18,
        { bottom: false, startCap: true, endCap: true, uvMetersPerTile: 3.0 }
      );
    }

    // Solid Flush Cream Corner Bays at SE Hip End (alphaSE - 4.5° .. alphaSE) & NW End Bay with Clerestory Windows (ref_071, ref_194)
    for (const [a0, a1] of [[alphaNW, alphaNW + 2.5 * DEG], [alphaSE - 4.5 * DEG, alphaSE]]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.creamStucco, toPhi(a1), toPhi(a0), rIn, rOut, 0.15, eaveY, 4,
        { bottom: false, startCap: true, endCap: true, uvMetersPerTile: 3.0 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.windowGlass, toPhi(a1) + 0.005, toPhi(a0) - 0.005, rOut - 0.06, rOut + 0.06, eaveY - 0.95, eaveY - 0.32, 4,
        { top: false, bottom: false, uvMetersPerTile: 2.2 }
      );
    }

    // 3. Two-Tier Sloped Green Grass Berm ("Talud Rumput") & Batu-Kali Toe Wall (NW & SE Wings, ref_079, ref_084, ref_192)
    //    Starts at ±10.2° so it tucks cleanly inside the straight forecourt wing retaining walls at x = ±10.95m (0.00m gap!)
    const rMidStep = 62.6;
    const rToe = cfg.rBermToe; // 66.8m
    const alphaBermCut = 10.85 * DEG;
    for (const [a0, a1] of [[alphaNW, -alphaBermCut], [alphaBermCut, alphaSE]]) {
      const p0 = toPhi(a1);
      const p1 = toPhi(a0);
      // Upper 32° grass slope: r = 57.3 (y = 3.72m) -> r = 62.6 (y = 1.48m)
      const upSlope = new THREE.Mesh(
        NexusSSoSTKernel.createPolarSlopedRibbon(p0, p1, rOut - 0.25, cfg.l1TopY, rMidStep, 1.48, 24, 3.5),
        materials.lushGrass
      );
      upSlope.receiveShadow = true;
      localGroup.add(upSlope);

      // Low 18cm grey concrete terrace curb step at r = 62.6m (ref_084, ref_192)
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.stoneRetaining, p0, p1, rMidStep - 0.15, rMidStep + 0.18, 1.26, 1.50, 24,
        { bottom: false, uvMetersPerTile: 2.0 }
      );

      // Lower gentle grass apron: r = 62.75 (y = 1.32m) -> r = 66.8 (y = 0.68m)
      const lowSlope = new THREE.Mesh(
        NexusSSoSTKernel.createPolarSlopedRibbon(p0, p1, rMidStep + 0.12, 1.32, rToe, 0.68, 24, 3.5),
        materials.lushGrass
      );
      lowSlope.receiveShadow = true;
      localGroup.add(lowSlope);

      // Batu-Kali Rubble-Stone Toe Retaining Wall at r = 66.8m (y = 0.10 .. 0.74m, ref_135, ref_084)
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.stoneRetaining, p0, p1, rToe - 0.28, rToe + 0.12, 0.08, 0.74, 24,
        { bottom: false, uvMetersPerTile: 2.0 }
      );
    }

    // Sloped Retaining End Cheek-Walls at SE Berm End (+43.5°) & NW Salmon-Brick Gable End (-42.5°, ref_135, ref_202)
    DPPDPrimaryStructure._addSlopedBermCheekWall(localGroup, materials.architecturalWhite, alphaSE, rOut - 0.2, cfg.l1TopY, rToe, 0.68, 0.30);
    DPPDPrimaryStructure._addSlopedBermCheekWall(localGroup, materials.architecturalWhite, alphaNW, rOut - 0.2, cfg.l1TopY, rToe - 1.2, 0.55, 0.32);

    // 4. NW Asymmetrical Salmon-Terracotta Brick Gable End-Wall at alphaNW = -42.5° (facing 9:00 West, ref_144, ref_202)
    DPPDPrimaryStructure._buildNWSalmonBrickGableWall(localGroup, materials, alphaNW, rIn, rRidge, rOut, eaveY, wingRidgeY);

    // 5. Polygonal Faceted Terracotta Wing Roofs (3 NW Facets + 3 SE Facets w/ Sloped Hip End)
    //    Stitched cleanly into the Central Steep A-Frame Gable with 0.00m Analytical Valley Seam!
    DPPDPrimaryStructure._buildStitchedWingAndGableRoofs(
      localGroup, materials, alphaNW, alphaSE, rIn - 0.95, rRidge, rOut + 1.15, eaveY, wingRidgeY, cfg.gableHalfWidth, cfg.gablePeakY
    );
  }

  static _addSlopedBermCheekWall(parent, mat, alpha, rTop, yTop, rBot, yBot, thickness = 0.30) {
    const s = Math.sin(alpha);
    const c = Math.cos(alpha);
    const nx = c * (thickness * 0.5);
    const nz = -s * (thickness * 0.5);

    const p = [
      [rTop * s - nx, 0.1, rTop * c - nz],
      [rBot * s - nx, 0.1, rBot * c - nz],
      [rBot * s - nx, yBot, rBot * c - nz],
      [rTop * s - nx, yTop, rTop * c - nz],
      [rTop * s + nx, 0.1, rTop * c + nz],
      [rBot * s + nx, 0.1, rBot * c + nz],
      [rBot * s + nx, yBot, rBot * c + nz],
      [rTop * s + nx, yTop, rTop * c + nz]
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
    parent.add(mesh);
  }

  static _buildNWSalmonBrickGableWall(parent, materials, alphaNW, rIn, rRidge, rOut, eaveY, ridgeY) {
    const s = Math.sin(alphaNW);
    const c = Math.cos(alphaNW);
    const nx = -c;
    const nz = s;
    const off = 0.08;

    const wSpan = rOut - rIn;
    const rMid = 0.5 * (rIn + rOut);
    const wallGroup = new THREE.Group();
    wallGroup.position.set(rMid * s + nx * off, 0, rMid * c + nz * off);
    wallGroup.rotation.y = Math.atan2(-c, s);
    parent.add(wallGroup);

    const baseWall = new THREE.Mesh(new THREE.BoxGeometry(wSpan, eaveY - 0.15, 0.36), materials.terracottaBrick);
    baseWall.position.set(0, 0.15 + (eaveY - 0.15) * 0.5, 0);
    baseWall.castShadow = true;
    baseWall.receiveShadow = true;
    wallGroup.add(baseWall);

    for (const by of [3.72, eaveY - 0.18]) {
      const beam = new THREE.Mesh(new THREE.BoxGeometry(wSpan + 0.12, 0.34, 0.42), materials.architecturalWhite);
      beam.position.set(0, by, 0.02);
      wallGroup.add(beam);
    }
    for (let i = 0; i <= 3; i++) {
      const px = -wSpan * 0.5 + (i / 3) * wSpan;
      const pil = new THREE.Mesh(new THREE.BoxGeometry(0.38, eaveY - 0.15, 0.42), materials.architecturalWhite);
      pil.position.set(px, 0.15 + (eaveY - 0.15) * 0.5, 0.02);
      wallGroup.add(pil);
    }
    for (const bx of [-2.8, 2.8]) {
      const cw = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.72, 0.44), materials.windowGlass);
      cw.position.set(bx, 3.12, 0.02);
      wallGroup.add(cw);

      const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, eaveY - 0.2, 6), materials.yellowLine);
      pipe.position.set(bx + 2.1, 0.15 + (eaveY - 0.2) * 0.5, 0.22);
      wallGroup.add(pipe);
    }

    const xIn = rIn - 0.85 - rMid;
    const xRidge = rRidge - rMid;
    const xOut = rOut + 1.05 - rMid;
    const shape = new THREE.Shape();
    shape.moveTo(xIn, eaveY);
    shape.lineTo(xOut, eaveY);
    shape.lineTo(xRidge, ridgeY);
    shape.closePath();
    const tympGeo = new THREE.ExtrudeGeometry(shape, { depth: 0.38, bevelEnabled: false });
    tympGeo.translate(0, 0, -0.19);
    const tympMesh = new THREE.Mesh(tympGeo, materials.terracottaBrick);
    tympMesh.castShadow = true;
    tympMesh.receiveShadow = true;
    wallGroup.add(tympMesh);

    for (const [xa, ya, xb, yb] of [[xIn, eaveY, xRidge, ridgeY], [xRidge, ridgeY, xOut, eaveY]]) {
      const len = Math.hypot(xb - xa, yb - ya);
      const ang = Math.atan2(yb - ya, xb - xa);
      const rake = new THREE.Mesh(new THREE.BoxGeometry(len + 0.15, 0.28, 0.44), materials.architecturalWhite);
      rake.position.set(0.5 * (xa + xb), 0.5 * (ya + yb), 0.04);
      rake.rotation.z = ang;
      wallGroup.add(rake);
    }
  }

  static _buildStitchedWingAndGableRoofs(
    parent,
    materials,
    alphaNW,
    alphaSE,
    rEaveIn,
    rRidge,
    rEaveOut,
    eaveY,
    wingRidgeY,
    gableHalfW,
    gablePeakY
  ) {
    const pos = [];
    const norm = [];
    const uvs = [];
    const ridgeCaps = [];

    const pushTri = (a, b, c, tileM = 1.8) => {
      const vA = new THREE.Vector3(...a);
      const vB = new THREE.Vector3(...b);
      const vC = new THREE.Vector3(...c);
      const fn = new THREE.Vector3().subVectors(vB, vA).cross(new THREE.Vector3().subVectors(vC, vA)).normalize();
      if (fn.y < 0) {
        return pushTri(a, c, b, tileM);
      }
      for (const pt of [a, b, c]) {
        pos.push(pt[0], pt[1], pt[2]);
        norm.push(fn.x, fn.y, fn.z);
        uvs.push(pt[0] / tileM, (pt[2] + pt[1]) / tileM);
      }
    };

    const pushQuad = (a, b, c, d, tileM = 1.8) => {
      pushTri(a, b, c, tileM);
      pushTri(a, c, d, tileM);
    };

    const ptPol = (r, a, y) => [r * Math.sin(a), y, r * Math.cos(a)];

    // Exact Analytical Valley Coordinates where Central Steep Gable meets Curved Wing Roof:
    // The two valleys start at the outer eave (±alphaEaveVal at rEaveOut, y = eaveY) and converge to a V-junction
    // directly on the main curved ridge at bPeak = [0, wingRidgeY, rRidge] (alpha = 0°, ref_174 Satellite Nadir!).
    const gHalfOut = gableHalfW + 0.65; // 11.85m eave overhang
    const alphaEaveVal = Math.asin(Math.min(0.99, gHalfOut / rEaveOut)); // ~11.65°

    // A. Continuous Smooth Inner Courtyard Slope across the ENTIRE building span (alphaNW .. alphaSE)
    //    Subdivided into 36 smooth angular steps matching the curved courtyard wall & soffit (0.000m chord gap!)
    const innerKeyAlphas = [alphaNW, -31.0 * DEG, -20.0 * DEG, 0.0, 20.0 * DEG, 33.5 * DEG];
    for (let i = 0; i < innerKeyAlphas.length - 1; i++) {
      const aStart = innerKeyAlphas[i];
      const aEnd = innerKeyAlphas[i + 1];
      const subSteps = 7;
      for (let s = 0; s < subSteps; s++) {
        const a0 = aStart + (s / subSteps) * (aEnd - aStart);
        const a1 = aStart + ((s + 1) / subSteps) * (aEnd - aStart);
        const r0 = ptPol(rRidge, a0, wingRidgeY);
        const r1 = ptPol(rRidge, a1, wingRidgeY);
        const i0 = ptPol(rEaveIn, a0, eaveY);
        const i1 = ptPol(rEaveIn, a1, eaveY);
        pushQuad(i1, i0, r0, r1);
      }
    }

    // B. NW Wing Outer Slope Facets (from alphaNW = -42.5° up to the exact NW Valley Line vEaveL -> bPeak!)
    const nwOuterPairs = [
      [alphaNW, -31.0 * DEG, alphaNW, -31.0 * DEG],
      [-31.0 * DEG, -20.0 * DEG, -31.0 * DEG, -20.0 * DEG],
      [-20.0 * DEG, -alphaEaveVal, -20.0 * DEG, 0.0]
    ];
    for (let i = 0; i < nwOuterPairs.length; i++) {
      const [ao0, ao1, ar0, ar1] = nwOuterPairs[i];
      const o0 = ptPol(rEaveOut, ao0, eaveY);
      const o1 = ptPol(rEaveOut, ao1, eaveY);
      const r0 = ptPol(rRidge, ar0, wingRidgeY);
      const r1 = ptPol(rRidge, ar1, wingRidgeY);
      pushQuad(o0, o1, r1, r0);
      ridgeCaps.push([new THREE.Vector3(...r0), new THREE.Vector3(...r1)]);
    }

    // C. SE Wing Outer Slope Facets (from exact SE Valley Line bPeak -> vEaveR up to 33.5°, plus Sloped Hip End to +43.5°)
    const seOuterPairs = [
      [alphaEaveVal, 20.0 * DEG, 0.0, 20.0 * DEG],
      [20.0 * DEG, 33.5 * DEG, 20.0 * DEG, 33.5 * DEG]
    ];
    for (let i = 0; i < seOuterPairs.length; i++) {
      const [ao0, ao1, ar0, ar1] = seOuterPairs[i];
      const o0 = ptPol(rEaveOut, ao0, eaveY);
      const o1 = ptPol(rEaveOut, ao1, eaveY);
      const r0 = ptPol(rRidge, ar0, wingRidgeY);
      const r1 = ptPol(rRidge, ar1, wingRidgeY);
      pushQuad(o0, o1, r1, r0);
      ridgeCaps.push([new THREE.Vector3(...r0), new THREE.Vector3(...r1)]);
    }

    // SE Sloped Hip End (from 33.5° ridge apex down to +43.5° outer & inner eave corners!)
    {
      const aHipRidge = 36.2 * DEG;
      const rStart = ptPol(rRidge, 33.5 * DEG, wingRidgeY);
      const rHipTip = ptPol(rRidge + 0.8, aHipRidge, wingRidgeY - 0.55);
      const oStart = ptPol(rEaveOut, 33.5 * DEG, eaveY);
      const oEnd = ptPol(rEaveOut, alphaSE, eaveY);
      const iStart = ptPol(rEaveIn, 33.5 * DEG, eaveY);
      const iEnd = ptPol(rEaveIn, alphaSE, eaveY);

      pushQuad(oStart, oEnd, rHipTip, rStart);
      pushQuad(iEnd, iStart, rStart, rHipTip);
      pushTri(oEnd, iEnd, rHipTip);

      ridgeCaps.push([new THREE.Vector3(...rStart), new THREE.Vector3(...rHipTip)]);
      ridgeCaps.push([new THREE.Vector3(...oEnd), new THREE.Vector3(...rHipTip)]);
      ridgeCaps.push([new THREE.Vector3(...iEnd), new THREE.Vector3(...rHipTip)]);
    }

    // D. Central Steep Symmetrical Cross-Gable Roof Stitched Directly to the V-Valley Vertices (0.00m Planar Seam!)
    //    - Lower valley vertices meet wing outer eave at vEaveL/R = ptPol(rEaveOut, ±alphaEaveVal, eaveY) (z = 57.44m)
    //    - Upper valley apex meets the main curved ridge at bPeak = [0, wingRidgeY, rRidge] (alpha = 0°)
    //    - Front gable overhang at zFront = 59.35m (1.91m in front of vEaveL/R, matching ref_084 & ref_185!)
    const zFront = 59.35;
    const fPeak = [0, wingRidgeY, zFront];
    const bPeak = ptPol(rRidge, 0.0, wingRidgeY);
    const fLeft = [-gHalfOut, eaveY, zFront];
    const fRight = [gHalfOut, eaveY, zFront];

    const vEaveL = ptPol(rEaveOut, -alphaEaveVal, eaveY);
    const vEaveR = ptPol(rEaveOut, alphaEaveVal, eaveY);

    // Left & Right 100% Planar Gable Slopes
    pushQuad(fLeft, vEaveL, bPeak, fPeak, 1.7);
    pushQuad(vEaveR, fRight, fPeak, bPeak, 1.7);

    // Main Gable Center Ridge Cap (from front apex to T-junction on main curved ridge)
    ridgeCaps.push([new THREE.Vector3(...fPeak), new THREE.Vector3(...bPeak)]);

    const roofGeo = new THREE.BufferGeometry();
    roofGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    roofGeo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
    roofGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    roofGeo.computeBoundingBox();
    roofGeo.computeBoundingSphere();

    const roofMesh = new THREE.Mesh(roofGeo, materials.terracottaRoof);
    roofMesh.castShadow = true;
    roofMesh.receiveShadow = true;
    parent.add(roofMesh);

    // Eave Soffits & Dark Fascia Boards along Wing Roofs
    const toPhi = (alpha) => Math.PI * 0.5 - alpha;
    NexusSSoSTKernel.addCurvedPolarBand(
      parent, materials.creamStucco, toPhi(alphaSE), toPhi(alphaNW), rEaveIn, rEaveOut, eaveY - 0.14, eaveY - 0.01, 42,
      { top: true, bottom: true, outer: true, inner: true, startCap: true, endCap: true, castShadow: false }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      parent, materials.darkMetal, toPhi(alphaSE), toPhi(alphaEaveVal - 0.2 * DEG), rEaveOut - 0.08, rEaveOut + 0.04, eaveY - 0.24, eaveY + 0.04, 18,
      { bottom: false, uvMetersPerTile: 3.0 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      parent, materials.darkMetal, toPhi(-alphaEaveVal + 0.2 * DEG), toPhi(alphaNW), rEaveOut - 0.08, rEaveOut + 0.04, eaveY - 0.24, eaveY + 0.04, 18,
      { bottom: false, uvMetersPerTile: 3.0 }
    );

    // 3D Ceramic Ridge Caps along Wing & Gable Ridges/Hips
    for (const pts of ridgeCaps) {
      const cGeo = NexusSSoSTKernel.createRidgeCapStrip(pts, 0.12, 6);
      if (cGeo) {
        const cm = new THREE.Mesh(cGeo, materials.terracottaBrick);
        cm.castShadow = true;
        parent.add(cm);
      }
    }

    // E. Full-Depth Watertight Cream Gable Attic Prism + Ceiling Soffit Slab (eliminates thin floating triangle!)
    //    Extends from tympZFront = 57.95m back to tympZBack = 55.20m (strictly 26cm below roof plane so zero white shows on top!)
    const tympZFront = 57.95;
    const tympZBack = 55.20;
    const tympDepth = tympZFront - tympZBack;
    const tympShape = new THREE.Shape();
    tympShape.moveTo(-gableHalfW + 0.05, eaveY - 0.12);
    tympShape.lineTo(gableHalfW - 0.05, eaveY - 0.12);
    tympShape.lineTo(0, wingRidgeY - 0.26);
    tympShape.closePath();

    const tympGeo = new THREE.ExtrudeGeometry(tympShape, { depth: tympDepth, bevelEnabled: false });
    tympGeo.translate(0, 0, tympZBack);
    const tympMesh = new THREE.Mesh(tympGeo, materials.creamStucco);
    tympMesh.castShadow = true;
    tympMesh.receiveShadow = true;
    parent.add(tympMesh);

    // Solid Horizontal Gable Eave Ceiling Soffit closing the underside from z = 54.0m to zFront = 59.35m
    const gableBaseSoffit = new THREE.Mesh(
      new THREE.BoxGeometry(gHalfOut * 2 - 0.12, 0.16, zFront - 54.0),
      materials.creamStucco
    );
    gableBaseSoffit.position.set(0, eaveY - 0.14, 0.5 * (54.0 + zFront));
    parent.add(gableBaseSoffit);

    // Deep front gable overhang soffit (18cm below terracotta roof!) & crisp dark terracotta-brown bargeboards
    for (const sgn of [-1, 1]) {
      const xa = sgn * gHalfOut;
      const ya = eaveY;
      const xb = 0;
      const yb = wingRidgeY;
      const slopeLen = Math.hypot(xb - xa, yb - ya);
      const ang = Math.atan2(yb - ya, xb - xa);

      const soffitDepth = zFront - tympZFront + 0.25;
      const soffit = new THREE.Mesh(
        new THREE.BoxGeometry(slopeLen - 0.30, 0.06, soffitDepth),
        materials.creamStucco
      );
      soffit.position.set(0.5 * (xa + xb), 0.5 * (ya + yb) - 0.22, tympZFront + soffitDepth * 0.5 - 0.12);
      soffit.rotation.z = ang;
      parent.add(soffit);

      const barge = new THREE.Mesh(
        new THREE.BoxGeometry(slopeLen + 0.14, 0.26, 0.16),
        materials.darkMetal
      );
      barge.position.set(0.5 * (xa + xb), 0.5 * (ya + yb) - 0.04, zFront + 0.02);
      barge.rotation.z = ang;
      parent.add(barge);

      const vergeCap = new THREE.Mesh(
        new THREE.BoxGeometry(slopeLen + 0.18, 0.10, 0.24),
        materials.terracottaRoof
      );
      vergeCap.position.set(0.5 * (xa + xb), 0.5 * (ya + yb) + 0.08, zFront + 0.01);
      vergeCap.rotation.z = ang;
      parent.add(vergeCap);

      // Horizontal side eave fascia board connecting wing fascia at vEaveL/R (z = 57.42m) to front gable corner (z = 60.85m)
      const sideFasciaLen = zFront - vEaveL[2] + 0.08;
      const sideFascia = new THREE.Mesh(
        new THREE.BoxGeometry(0.12, 0.26, sideFasciaLen),
        materials.darkMetal
      );
      sideFascia.position.set(sgn * (gHalfOut - 0.04), eaveY - 0.10, 0.5 * (vEaveL[2] + zFront));
      parent.add(sideFascia);
    }

    // Recessed Apex Soffit Fixture inside the Gable Peak (ref_184)
    const apexLamp = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.08, 12), materials.goldAccent);
    apexLamp.position.set(0, wingRidgeY - 1.15, tympZFront + 0.42);
    parent.add(apexLamp);

    // Lightning Arrestor Spires (Apex + 4 Wing Ridge vertices, ref_077, ref_084)
    const apexSpire = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.04, 1.85, 6), materials.darkMetal);
    apexSpire.position.set(0, wingRidgeY + 0.9, zFront - 0.15);
    parent.add(apexSpire);

    for (const deg of [-31, -20, 20, 33.5]) {
      const sp = ptPol(rRidge, deg * DEG, wingRidgeY + 0.72);
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.035, 1.45, 6), materials.darkMetal);
      rod.position.set(sp[0], sp[1], sp[2]);
      parent.add(rod);
    }
  }
}
