import * as THREE from 'three';
import { NexusSSoSTKernel } from '../../core/MasterSurfaceKernels.js';
import { UI_DPPD_MASTER_PLAN } from '../../world-gis/MasterPlanConfig.js';

const DEG = Math.PI / 180;

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./library/gedung-dppd/PrimaryStructure.js
 *
 * Stage 3 Primary Hull, SSoST Stitched Roof Valleys, Polygonal Faceted Terracotta Wing Roofs,
 * 2-Tier Sloped Grass Berm ("Talud Rumput"), Batu-Kali Retaining Walls, & NW Salmon-Brick Gable End.
 *
 * Local Coordinate Mapping:
 *   Group rotated by psi = (90° - thAxis) so local +Z points along thAxis = +135.5° (SW),
 *   local alpha < 0 maps to NW Wing (ending at th = +178° West),
 *   local alpha > 0 maps to SE Wing (ending at th = +92° South).
 */
export class DPPDPrimaryStructure {
  static build(localGroup, materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.dppd;
    const rIn = cfg.rInner;        // 39.5m
    const rOut = cfg.rOuter;       // 57.5m
    const rWallL2 = cfg.rLoggiaWall; // 54.8m
    const rRidge = 47.2;           // Slightly inward-offset ridge (ref_174, ref_202)
    const eaveY = cfg.eaveY;       // 6.45m
    const wingRidgeY = cfg.wingRidgeY; // 11.85m

    const alphaNW = -42.5 * DEG;   // Maps to th = 135.5° - (-42.5°) = +178.0° (West 9:00 Gable End!)
    const alphaSE = 43.5 * DEG;    // Maps to th = 135.5° - (+43.5°) = +92.0° (South 6:00 Hipped End!)
    const alphaPortal = 11.0 * DEG; // Central Steep A-Frame Gable & Entrance Cutout span [-11°, +11°]

    // Helper converting local (r, alpha) -> Polar angle for NexusSSoSTKernel (where x = r*cos(phi), z = r*sin(phi))
    // Since local x = r*sin(alpha), z = r*cos(alpha) => phi = pi/2 - alpha!
    const toPhi = (alpha) => Math.PI * 0.5 - alpha;

    // 1. Inner Courtyard Curved Facade & L1 Core Structure (full span alphaNW .. alphaSE)
    NexusSSoSTKernel.addCurvedPolarBand(
      localGroup, materials.creamStucco, toPhi(alphaSE), toPhi(alphaNW), rIn, rWallL2, -0.2, cfg.l1TopY, 36,
      { top: true, bottom: false, startCap: false, endCap: false, uvMetersPerTile: 3.5 }
    );
    // Inner courtyard L2 cream wall + dark ribbon windows
    NexusSSoSTKernel.addCurvedPolarBand(
      localGroup, materials.creamStucco, toPhi(alphaSE), toPhi(alphaNW), rIn, rIn + 0.45, cfg.l1TopY, eaveY, 36,
      { bottom: false, startCap: false, endCap: false, uvMetersPerTile: 3.5 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      localGroup, materials.windowGlass, toPhi(alphaSE) + 0.02, toPhi(alphaNW) - 0.02, rIn - 0.06, rIn + 0.12, cfg.l1TopY + 0.85, eaveY - 0.55, 36,
      { top: false, bottom: false, startCap: false, endCap: false, uvMetersPerTile: 3.2 }
    );

    // 2. Recessed L2 Loggia Back Wall + Reflective Awning Windows (eliminates black coal hole!)
    for (const [a0, a1] of [[alphaNW + 2.5 * DEG, -alphaPortal], [alphaPortal, alphaSE - 4.5 * DEG]]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.creamStucco, toPhi(a1), toPhi(a0), rWallL2 - 0.4, rWallL2, cfg.l1TopY, eaveY, 18,
        { bottom: false, uvMetersPerTile: 3.2 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.windowGlass, toPhi(a1), toPhi(a0), rWallL2 - 0.05, rWallL2 + 0.08, cfg.l1TopY + 0.92, eaveY - 0.42, 18,
        { top: false, bottom: false, uvMetersPerTile: 2.8 }
      );
      // L2 Loggia Walkway Floor & Outer Cream Sill Parapet (y = 3.72 .. 4.44m)
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.creamStucco, toPhi(a1), toPhi(a0), rWallL2, rOut, cfg.l1TopY - 0.18, cfg.l1TopY + 0.06, 18,
        { bottom: false, uvMetersPerTile: 3.0 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.creamStucco, toPhi(a1), toPhi(a0), rOut - 0.36, rOut, cfg.l1TopY, cfg.l1TopY + 0.72, 18,
        { bottom: false, uvMetersPerTile: 3.0 }
      );
    }

    // Solid Flush Cream Corner Bays at SE Hip End (alphaSE - 4.5° .. alphaSE) & NW End Bay with Clerestory Windows (ref_071, ref_194)
    for (const [a0, a1] of [[alphaNW, alphaNW + 2.5 * DEG], [alphaSE - 4.5 * DEG, alphaSE]]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.creamStucco, toPhi(a1), toPhi(a0), rWallL2, rOut, cfg.l1TopY, eaveY, 4,
        { bottom: false, uvMetersPerTile: 3.0 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.windowGlass, toPhi(a1) + 0.005, toPhi(a0) - 0.005, rOut - 0.06, rOut + 0.06, eaveY - 0.95, eaveY - 0.32, 4,
        { top: false, bottom: false, uvMetersPerTile: 2.2 }
      );
    }

    // 3. Two-Tier Sloped Green Grass Berm ("Talud Rumput") & Batu-Kali Toe Wall (NW & SE Wings, ref_079, ref_084, ref_192)
    const rMidStep = 62.6;
    const rToe = cfg.rBermToe; // 66.8m
    for (const [a0, a1] of [[alphaNW, -alphaPortal - 0.6 * DEG], [alphaPortal + 0.6 * DEG, alphaSE]]) {
      const p0 = toPhi(a1);
      const p1 = toPhi(a0);
      // Upper 32° grass slope: r = 57.5 (y = 3.72m) -> r = 62.6 (y = 1.48m)
      const upSlope = new THREE.Mesh(
        NexusSSoSTKernel.createPolarSlopedRibbon(p0, p1, rOut - 0.05, cfg.l1TopY, rMidStep, 1.48, 22, 3.5),
        materials.lushGrass
      );
      upSlope.receiveShadow = true;
      localGroup.add(upSlope);

      // Low 18cm grey concrete terrace curb step at r = 62.6m (ref_084, ref_192)
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.stoneRetaining, p0, p1, rMidStep - 0.15, rMidStep + 0.18, 1.26, 1.50, 22,
        { bottom: false, uvMetersPerTile: 2.0 }
      );

      // Lower gentle grass apron: r = 62.75 (y = 1.32m) -> r = 66.8 (y = 0.68m)
      const lowSlope = new THREE.Mesh(
        NexusSSoSTKernel.createPolarSlopedRibbon(p0, p1, rMidStep + 0.12, 1.32, rToe, 0.68, 22, 3.5),
        materials.lushGrass
      );
      lowSlope.receiveShadow = true;
      localGroup.add(lowSlope);

      // Batu-Kali Rubble-Stone Toe Retaining Wall at r = 66.8m (y = 0.10 .. 0.74m, ref_135, ref_084)
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.stoneRetaining, p0, p1, rToe - 0.28, rToe + 0.12, 0.08, 0.74, 22,
        { bottom: false, uvMetersPerTile: 2.0 }
      );
    }

    // Sloped Retaining End Cheek-Walls at SE Berm End (+43.5°) & NW Salmon-Brick Gable End (-42.5°, ref_135, ref_202)
    DPPDPrimaryStructure._addSlopedBermCheekWall(localGroup, materials.architecturalWhite, alphaSE, rOut, cfg.l1TopY, rToe, 0.68, 0.28);
    DPPDPrimaryStructure._addSlopedBermCheekWall(localGroup, materials.architecturalWhite, alphaNW, rOut, cfg.l1TopY, rToe - 1.2, 0.55, 0.32);

    // Two Sloped Batu-Kali Cheek Walls framing the Central Entrance Forecourt (alpha = ±11.6°, ref_084, ref_184)
    for (const sgn of [-1, 1]) {
      DPPDPrimaryStructure._addSlopedBermCheekWall(
        localGroup, materials.stoneRetaining, sgn * (alphaPortal + 0.55 * DEG), rOut - 0.4, cfg.l1TopY, rToe, 0.45, 0.38
      );
    }

    // 4. NW Asymmetrical Salmon-Terracotta Brick Gable End-Wall at alphaNW = -42.5° (facing 9:00 West, ref_144, ref_202)
    DPPDPrimaryStructure._buildNWSalmonBrickGableWall(localGroup, materials, alphaNW, rIn, rRidge, rOut, eaveY, wingRidgeY);

    // 5. Polygonal Faceted Terracotta Wing Roofs (3 NW Facets + 3 SE Facets w/ Sloped Hip End)
    //    Stitched cleanly into the Central Steep A-Frame Gable with 0.00m Gap!
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
    addQuad(p[3], p[2], p[6], p[7]); // sloped top coping
    addQuad(p[1], p[5], p[6], p[2]); // front toe

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
    // Outward normal facing NW (increasing negative alpha): (-c, 0, s)
    const nx = -c;
    const nz = s;
    const off = 0.08;

    // Lower 2-story rectangular salmon-brick wall [rIn .. rOut] x [0.15 .. eaveY]
    const wSpan = rOut - rIn;
    const rMid = 0.5 * (rIn + rOut);
    const wallGroup = new THREE.Group();
    wallGroup.position.set(rMid * s + nx * off, 0, rMid * c + nz * off);
    // Local X along radial (s, 0, c), local Z along outward normal (-c, 0, s) => rotation.y = atan2(-c, s) - pi/2?
    // Wait: if local X = (s, 0, c) and local Z = (-c, 0, s), Check X x Y = (s,0,c) x (0,1,0) = (-c, 0, s) = Z! Right-handed!
    // In Three.js rotation.y = psi: X_axis = (cos psi, 0, -sin psi) => cos psi = s, -sin psi = c => psi = atan2(-c, s)!
    wallGroup.rotation.y = Math.atan2(-c, s);
    parent.add(wallGroup);

    const baseWall = new THREE.Mesh(new THREE.BoxGeometry(wSpan, eaveY - 0.15, 0.34), materials.terracottaBrick);
    baseWall.position.set(0, 0.15 + (eaveY - 0.15) * 0.5, 0);
    baseWall.castShadow = true;
    baseWall.receiveShadow = true;
    wallGroup.add(baseWall);

    // White concrete L2 floor beam, eave tie-beam, & 4 structural white pilasters (3 bays, ref_144, ref_202)
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
    // High clerestory dark windows in middle bays + yellow utility downspout accents (ref_144)
    for (const bx of [-2.8, 2.8]) {
      const cw = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.72, 0.44), materials.windowGlass);
      cw.position.set(bx, 3.12, 0.02);
      wallGroup.add(cw);

      const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, eaveY - 0.2, 6), materials.yellowLine);
      pipe.position.set(bx + 2.1, 0.15 + (eaveY - 0.2) * 0.5, 0.22);
      wallGroup.add(pipe);
    }

    // Asymmetrical Triangular Salmon-Brick Gable Tympanum [xIn -> xRidge -> xOut]
    const xIn = rIn - 0.85 - rMid;
    const xRidge = rRidge - rMid;
    const xOut = rOut + 1.05 - rMid;
    const shape = new THREE.Shape();
    shape.moveTo(xIn, eaveY);
    shape.lineTo(xOut, eaveY);
    shape.lineTo(xRidge, ridgeY);
    shape.closePath();
    const tympGeo = new THREE.ExtrudeGeometry(shape, { depth: 0.32, bevelEnabled: false });
    tympGeo.translate(0, 0, -0.16);
    const tympMesh = new THREE.Mesh(tympGeo, materials.terracottaBrick);
    tympMesh.castShadow = true;
    tympMesh.receiveShadow = true;
    wallGroup.add(tympMesh);

    // Crisp White Sloping Bargeboards ("Lisplank Putih", ref_144, ref_202)
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

    // A. NW Wing Roof: 3 Polygonal Facets from alphaNW (-42.5° flush gable) to alphaValleyNW (-7.5° inside gable!)
    const nwAlphas = [alphaNW, -31.0 * DEG, -20.0 * DEG, -7.2 * DEG];
    for (let i = 0; i < nwAlphas.length - 1; i++) {
      const a0 = nwAlphas[i];
      const a1 = nwAlphas[i + 1];
      const o0 = ptPol(rEaveOut, a0, eaveY);
      const o1 = ptPol(rEaveOut, a1, eaveY);
      const r0 = ptPol(rRidge, a0, wingRidgeY);
      const r1 = ptPol(rRidge, a1, wingRidgeY);
      const i0 = ptPol(rEaveIn, a0, eaveY);
      const i1 = ptPol(rEaveIn, a1, eaveY);

      pushQuad(o0, o1, r1, r0);
      pushQuad(i1, i0, r0, r1);
      ridgeCaps.push([new THREE.Vector3(...r0), new THREE.Vector3(...r1)]);
      if (i > 0) {
        ridgeCaps.push([new THREE.Vector3(...o0), new THREE.Vector3(...r0)]);
      }
    }

    // B. SE Wing Roof: 2 Full Facets + 1 Sloped Hip-End Facet ending cleanly at alphaSE (+43.5°, ref_084, ref_174)
    const seAlphas = [7.2 * DEG, 20.0 * DEG, 33.5 * DEG];
    for (let i = 0; i < seAlphas.length - 1; i++) {
      const a0 = seAlphas[i];
      const a1 = seAlphas[i + 1];
      const o0 = ptPol(rEaveOut, a0, eaveY);
      const o1 = ptPol(rEaveOut, a1, eaveY);
      const r0 = ptPol(rRidge, a0, wingRidgeY);
      const r1 = ptPol(rRidge, a1, wingRidgeY);
      const i0 = ptPol(rEaveIn, a0, eaveY);
      const i1 = ptPol(rEaveIn, a1, eaveY);

      pushQuad(o0, o1, r1, r0);
      pushQuad(i1, i0, r0, r1);
      ridgeCaps.push([new THREE.Vector3(...r0), new THREE.Vector3(...r1)]);
      if (i > 0) {
        ridgeCaps.push([new THREE.Vector3(...o0), new THREE.Vector3(...r0)]);
      }
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
      // Triangular hipped end slope at SE tip!
      pushTri(oEnd, iEnd, rHipTip);

      ridgeCaps.push([new THREE.Vector3(...rStart), new THREE.Vector3(...rHipTip)]);
      ridgeCaps.push([new THREE.Vector3(...oEnd), new THREE.Vector3(...rHipTip)]);
      ridgeCaps.push([new THREE.Vector3(...iEnd), new THREE.Vector3(...rHipTip)]);
    }

    // C. Central Steep Symmetrical Cross-Gable Roof (x ∈ [-11.8, +11.8], zFront = 61.2 back to zBack = 38.6!)
    //    Analytical Valley Cap lines where steep gable meets the curved wing roofs!
    const gHalfOut = gableHalfW + 0.65; // 11.85m eave overhang
    const zFront = 61.25;
    const zRidgeBack = rRidge - 2.0; // 45.2m
    const zInnerEave = rEaveIn;      // 38.55m

    const fPeak = [0, gablePeakY, zFront];
    const bPeak = [0, gablePeakY - 1.15, zRidgeBack];
    const fLeft = [-gHalfOut, eaveY, zFront];
    const fRight = [gHalfOut, eaveY, zFront];
    const bLeft = [-gHalfOut, eaveY, zRidgeBack];
    const bRight = [gHalfOut, eaveY, zRidgeBack];
    const iCenter = [0, eaveY, zInnerEave];

    // Left & Right Steep Saddle Slopes + Rear Hip Transition into Inner Courtyard
    pushQuad(fLeft, bLeft, bPeak, fPeak, 1.7);
    pushQuad(bRight, fRight, fPeak, bPeak, 1.7);
    pushTri(bLeft, iCenter, bPeak, 1.7);
    pushTri(iCenter, bRight, bPeak, 1.7);

    // Main Gable Ridge & Valley Ridge Caps ("Nok Genteng & Jurai")
    ridgeCaps.push([new THREE.Vector3(...fPeak), new THREE.Vector3(...bPeak)]);
    ridgeCaps.push([new THREE.Vector3(...bPeak), new THREE.Vector3(...iCenter)]);

    // Analytical Valley Strips where Wing Roof Ridge (y = 11.85m) meets Steep Gable Plane:
    // Gable plane height y(x) = eaveY + (gablePeakY - eaveY) * (1 - |x| / gHalfOut)
    // At y = wingRidgeY => |x_valley| = gHalfOut * (1 - (wingRidgeY - eaveY) / (gablePeakY - eaveY))
    const xValTop = gHalfOut * (1.0 - (wingRidgeY - eaveY) / (gablePeakY - eaveY));
    for (const sgn of [-1, 1]) {
      ridgeCaps.push([
        new THREE.Vector3(sgn * gHalfOut, eaveY + 0.05, rEaveOut - 0.6),
        new THREE.Vector3(sgn * xValTop, wingRidgeY + 0.05, rRidge)
      ]);
    }

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
      parent, materials.creamStucco, toPhi(alphaSE), toPhi(alphaNW), rEaveIn, rEaveOut, eaveY - 0.12, eaveY - 0.02, 42,
      { top: false, bottom: true, outer: false, inner: false, startCap: false, endCap: false, castShadow: false }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      parent, materials.darkMetal, toPhi(alphaSE), toPhi(-10.2 * DEG), rEaveOut - 0.08, rEaveOut + 0.04, eaveY - 0.24, eaveY + 0.04, 18,
      { bottom: false, uvMetersPerTile: 3.0 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      parent, materials.darkMetal, toPhi(10.2 * DEG), toPhi(alphaNW), rEaveOut - 0.08, rEaveOut + 0.04, eaveY - 0.24, eaveY + 0.04, 18,
      { bottom: false, uvMetersPerTile: 3.0 }
    );

    // 3D Ceramic Ridge Caps along all Wing & Gable Ridges/Hips/Valleys
    for (const pts of ridgeCaps) {
      const cGeo = NexusSSoSTKernel.createRidgeCapStrip(pts, 0.13, 6);
      if (cGeo) {
        const cm = new THREE.Mesh(cGeo, materials.terracottaBrick);
        cm.castShadow = true;
        parent.add(cm);
      }
    }

    // D. Central Steep Cream Plaster Triangular Tympanum + Deep Overhanging Cream Bargeboards & Apex Finial (ref_084, ref_184)
    const tympZ = 59.35; // recessed 1.9m behind front overhang zFront = 61.25m!
    const tympShape = new THREE.Shape();
    tympShape.moveTo(-gableHalfW, eaveY - 0.1);
    tympShape.lineTo(gableHalfW, eaveY - 0.1);
    tympShape.lineTo(0, gablePeakY - 0.18);
    tympShape.closePath();

    const tympGeo = new THREE.ExtrudeGeometry(tympShape, { depth: 0.45, bevelEnabled: false });
    tympGeo.translate(0, 0, tympZ - 0.45);
    const tympMesh = new THREE.Mesh(tympGeo, materials.creamStucco);
    tympMesh.castShadow = true;
    tympMesh.receiveShadow = true;
    parent.add(tympMesh);

    // Deep front gable overhang soffit & crisp cream/dark bargeboards (NO inverted-V bug!)
    for (const sgn of [-1, 1]) {
      const xa = sgn * gHalfOut;
      const ya = eaveY;
      const xb = 0;
      const yb = gablePeakY;
      const slopeLen = Math.hypot(xb - xa, yb - ya);
      const ang = Math.atan2(yb - ya, xb - xa);

      // Deep overhang underside soffit slab from tympZ to zFront
      const soffitDepth = zFront - tympZ + 0.2;
      const soffit = new THREE.Mesh(
        new THREE.BoxGeometry(slopeLen, 0.14, soffitDepth),
        materials.creamStucco
      );
      soffit.position.set(0.5 * (xa + xb), 0.5 * (ya + yb) - 0.08, tympZ + soffitDepth * 0.5 - 0.1);
      soffit.rotation.z = ang;
      soffit.castShadow = true;
      parent.add(soffit);

      // Crisp dark terracotta-brown front bargeboard fascia + terracotta verge cap (ref_084, ref_184)
      const barge = new THREE.Mesh(
        new THREE.BoxGeometry(slopeLen + 0.14, 0.26, 0.16),
        materials.darkMetal
      );
      barge.position.set(0.5 * (xa + xb), 0.5 * (ya + yb) - 0.03, zFront + 0.02);
      barge.rotation.z = ang;
      parent.add(barge);

      const vergeCap = new THREE.Mesh(
        new THREE.BoxGeometry(slopeLen + 0.18, 0.11, 0.24),
        materials.terracottaRoof
      );
      vergeCap.position.set(0.5 * (xa + xb), 0.5 * (ya + yb) + 0.12, zFront + 0.01);
      vergeCap.rotation.z = ang;
      parent.add(vergeCap);
    }

    // Recessed Apex Soffit Fixture inside the Gable Peak (ref_184)
    const apexLamp = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.08, 12), materials.goldAccent);
    apexLamp.position.set(0, gablePeakY - 1.38, tympZ + 0.48);
    parent.add(apexLamp);

    // Lightning Arrestor Spires (Apex + 4 Wing Ridge vertices, ref_077, ref_084)
    const apexSpire = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.04, 1.85, 6), materials.darkMetal);
    apexSpire.position.set(0, gablePeakY + 0.9, zFront - 0.15);
    parent.add(apexSpire);

    for (const deg of [-31, -20, 20, 33.5]) {
      const sp = ptPol(rRidge, deg * DEG, wingRidgeY + 0.72);
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.035, 1.45, 6), materials.darkMetal);
      rod.position.set(sp[0], sp[1], sp[2]);
      parent.add(rod);
    }
  }
}
