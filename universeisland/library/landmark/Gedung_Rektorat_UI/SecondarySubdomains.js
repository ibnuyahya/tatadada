import * as THREE from 'three';
import { MasterSurfaceBuilder } from '../../../core/MasterSurfaceBuilder.js';
import {
  CORNER_QUADRANTS,
  getStationSegmentByZone
} from './AssetMetaAndEnvelope.js';
import { MasterSurfaceKernel } from './MasterSurfaceKernel.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.3 — 10x Ultra-High Level of Detail Edition)
 * ./library/landmark/Gedung_Rektorat_UI/SecondarySubdomains.js
 *
 * TAHAP 5 (Gate 3): Kavitas Sekunder, Shutlines, Flanges, & Normal-Offset Applique (10x Ultra-LOD):
 * 1. Bubungan Jurai Putih Berprofil 3D (Tajug Utama, 4 Limas Sudut, Pendopo Segidelapan, & 4 Paviliun Limas).
 * 2. 336 Mullion Jendela Fisik 3D + Transom Horizontal + Ambang Atas/Bawah pada 6 Lantai Pagoda & Blok Sayap.
 * 3. Rangka Tirai Kaca Berlian (X-Diamond Structural Mullion Lattice 3D) pada ke-4 Ceruk Atrium Utama & Skylight.
 * 4. Pahatan Tiga-Dimensi (3D Bas-Relief Sculpture) Emblem Kebesaran Makara UI & Pohon Kalpataru pada 4 Sisi Mahkota.
 * 5. Kolom Berprofil Plinth Bertingkat, Alur Pilaster, Kapital Korbel, Konsol Teritisan Pagoda, & Kasau/Gording Atap.
 * 6. Bak Tanaman Berceruk Tanah Hitam & Unit Louver Mekanikal Atap Datar Sayap Barat/Timur.
 */
export class SecondarySubdomains {
  static build(materials, detailScale = 1.0) {
    const group = new THREE.Group();
    group.name = 'Tahap5_SecondarySubdomains';

    const isUltra = detailScale >= 0.9;
    const evalMaster = MasterSurfaceKernel.evaluateMasterSurface;
    const duFacet = 0.25 / 7.0;

    const addMesh = (geo, mat, name, castShadow = true, receiveShadow = true) => {
      const m = new THREE.Mesh(geo, mat);
      m.name = name;
      m.castShadow = castShadow;
      m.receiveShadow = receiveShadow;
      group.add(m);
      return m;
    };

    // =========================================================================
    // 1. NORMAL-OFFSET CRISP WHITE HIP RIDGES ('BUBUNGAN JURAI PUTIH' 3D)
    //    ON STEEP MAIN TAJUG SPIRE (y = 33.10m .. 46.95m — Mavic Photos #1, #2, #3, #5)
    // =========================================================================
    const tajugSeg = getStationSegmentByZone('TAJUG_MAIN_SLOPE');
    if (tajugSeg) {
      const vSpan = Math.max(1e-6, tajugSeg.vMax - tajugSeg.vMin);
      for (let q = 0; q < 4; q++) {
        const uHipMin = q * 0.25 + 3.0 * duFacet;
        const uHipMax = q * 0.25 + 4.0 * duFacet;
        const uMid = 0.5 * (uHipMin + uHipMax);
        const uHalf = 0.5 * (uHipMax - uHipMin);
        const ridgeGeo = MasterSurfaceBuilder.buildNormalOffsetAppliqueBand(
          evalMaster,
          uHipMin,
          uHipMax,
          tajugSeg.vMin,
          tajugSeg.vMax,
          isUltra ? 6 : 2,
          isUltra ? 28 : 2,
          isUltra
            ? (u, v) => {
                const tV = (v - tajugSeg.vMin) / vSpan;
                const crownArch = 1.0 - 0.35 * Math.pow((u - uMid) / uHalf, 2.0);
                const tileCollar = 0.018 * Math.cos(2.0 * Math.PI * 22.0 * tV);
                return (0.115 + tileCollar) * crownArch;
              }
            : 0.11,
          { closeU: true, closeVBottom: true, closeVTop: true }
        );
        addMesh(ridgeGeo, materials.precastConcrete, `Applique_TajugWhiteHipRidge_Q${q}`);
      }
    }

    // =========================================================================
    // 2. NORMAL-OFFSET CRISP WHITE HIP RIDGES ON THE 4 BIG CORNER LIMAS ROOFS,
    //    OCTAGONAL PENDOPO, & 4 HIPPED PAVILIONS (Strictly inside 108x124x50m!)
    // =========================================================================
    if (detailScale >= 0.45) {
      for (const q of CORNER_QUADRANTS) {
        const evalCorner = MasterSurfaceKernel.createCornerTowerCrownEvaluator(q);
        for (let c = 0; c < 4; c++) {
          const uCorner = c * 0.25 + 0.125;
          const hipGeo = MasterSurfaceBuilder.buildNormalOffsetAppliqueBand(
            evalCorner,
            uCorner - 0.015,
            uCorner + 0.015,
            0.502,
            0.748,
            isUltra ? 4 : 2,
            isUltra ? 14 : 1,
            isUltra
              ? (u, v) => 0.088 + 0.012 * Math.cos(2.0 * Math.PI * 12.0 * ((v - 0.502) / 0.246))
              : 0.085,
            { closeU: true, closeVBottom: true, closeVTop: true }
          );
          addMesh(hipGeo, materials.precastConcrete, `Applique_CornerLimasHip_${q.id}_${c}`);
        }
      }
    }

    if (isUltra) {
      // 8 Crisp White Hip Ridges on the 45° Octagonal Skylit Pendopo
      const evalOct = MasterSurfaceKernel.createOctagonalSkylitRoofEvaluator({
        cx: 29.5,
        cz: 29.5,
        outerR: 8.2,
        skylightR: 2.25,
        eaveY: 3.75,
        fasciaY: 3.94,
        skylightBaseY: 6.85,
        skylightApexY: 8.15
      });
      for (let k = 0; k < 8; k++) {
        const uCorner = (k + 0.5) / 8.0;
        const octHip = MasterSurfaceBuilder.buildNormalOffsetAppliqueBand(
          evalOct,
          uCorner - 0.008,
          uCorner + 0.008,
          0.515,
          0.746,
          3,
          10,
          0.065,
          { closeU: true, closeVBottom: true, closeVTop: true }
        );
        addMesh(octHip, materials.precastConcrete, `Applique_OctPendopoHip_${k}`);
      }

      // 4 Crisp White Hip Ridges on each Hipped Pavilion (starting at v=0.535 >= 0.8m inside outer eave!)
      const hipPavilionSpecs = [
        { tag: 'Gatehouse', cx: -2.5, cz: 65.5, halfX: 6.5, halfZ: 6.5, rHX: 1.8, rHZ: 0.25, eY: 3.75, rY: 7.45 },
        { tag: 'WestHall', cx: -43.75, cz: -1.3, halfX: 10.25, halfZ: 12.2, rHX: 0.35, rHZ: 2.4, eY: 5.85, rY: 12.25 },
        { tag: 'EastHall', cx: 43.75, cz: -4.3, halfX: 10.25, halfZ: 12.2, rHX: 0.35, rHZ: 2.4, eY: 5.85, rY: 12.25 },
        { tag: 'NorthPav', cx: 5.0, cz: -43.5, halfX: 8.5, halfZ: 8.5, rHX: 0.25, rHZ: 0.25, eY: 3.75, rY: 8.55 }
      ];
      for (const hp of hipPavilionSpecs) {
        const evalHP = MasterSurfaceKernel.createHippedRoofEvaluator({
          cx: hp.cx,
          cz: hp.cz,
          halfX: hp.halfX,
          halfZ: hp.halfZ,
          ridgeHalfX: hp.rHX,
          ridgeHalfZ: hp.rHZ,
          eaveY: hp.eY,
          fasciaY: hp.eY + 0.22,
          ridgeY: hp.rY - 0.12,
          capTopY: hp.rY
        });
        for (let c = 0; c < 4; c++) {
          const uCorner = c * 0.25 + 0.125;
          const pHip = MasterSurfaceBuilder.buildNormalOffsetAppliqueBand(
            evalHP,
            uCorner - 0.012,
            uCorner + 0.012,
            0.535,
            0.746,
            3,
            10,
            0.075,
            { closeU: true, closeVBottom: true, closeVTop: true }
          );
          addMesh(pHip, materials.precastConcrete, `Applique_PavilionHip_${hp.tag}_${c}`);
        }
      }
    }

    // =========================================================================
    // 3. PHYSICAL 3D WINDOW MULLIONS, TRANSOMS, HEADERS, & SILLS ON 6 PAGODA FLOORS
    //    + 6 COURTYARD ANNEX BUILDINGS (10x Ultra-LOD)
    // =========================================================================
    if (isUltra) {
      const pLo = new THREE.Vector3();
      const pHi = new THREE.Vector3();
      const pMid = new THREE.Vector3();
      const nMid = new THREE.Vector3();

      for (let floor = 2; floor <= 7; floor++) {
        const spandrelSeg = getStationSegmentByZone(`FLOOR_${floor}_SPANDREL`);
        const glassSeg = getStationSegmentByZone(`FLOOR_${floor}_GLASS`);
        if (!spandrelSeg || !glassSeg) continue;

        for (let q = 0; q < 4; q++) {
          const qBase = q * 0.25;
          for (const [sA, sB, tag] of [
            [2.02, 2.98, 'FaceA'],
            [4.02, 4.98, 'FaceB']
          ]) {
            const uFaceMin = qBase + sA * duFacet;
            const uFaceMax = qBase + sB * duFacet;

            // A. Projecting Normal-Offset Window Sill Ledge
            const sillGeo = MasterSurfaceBuilder.buildNormalOffsetAppliqueBand(
              evalMaster,
              uFaceMin,
              uFaceMax,
              spandrelSeg.vMax - 0.0018,
              spandrelSeg.vMax,
              8,
              1,
              0.125,
              { closeU: true, closeVBottom: true, closeVTop: true }
            );
            addMesh(sillGeo, materials.precastConcrete, `Applique_WindowSill_F${floor}_Q${q}_${tag}`);

            // B. Projecting Normal-Offset Window Upper Lintel / Drip Header
            const headerGeo = MasterSurfaceBuilder.buildNormalOffsetAppliqueBand(
              evalMaster,
              uFaceMin,
              uFaceMax,
              glassSeg.vMax - 0.0014,
              glassSeg.vMax,
              8,
              1,
              0.095,
              { closeU: true, closeVBottom: true, closeVTop: true }
            );
            addMesh(headerGeo, materials.precastConcrete, `Applique_WindowHeader_F${floor}_Q${q}_${tag}`);

            // C. 7 Physical 3D Vertical Window Mullions (6 Window Bays per Face) + 1 Horizontal Transom
            evalMaster(0.5 * (uFaceMin + uFaceMax), glassSeg.vMin, pLo, nMid);
            evalMaster(0.5 * (uFaceMin + uFaceMax), glassSeg.vMax, pHi, nMid);
            const winH = Math.max(0.4, pHi.y - pLo.y);
            const winMidY = 0.5 * (pLo.y + pHi.y);
            const faceYaw = Math.atan2(nMid.x, nMid.z);

            const pStart = new THREE.Vector3();
            const pEnd = new THREE.Vector3();
            evalMaster(uFaceMin, 0.5 * (glassSeg.vMin + glassSeg.vMax), pStart, nMid);
            evalMaster(uFaceMax, 0.5 * (glassSeg.vMin + glassSeg.vMax), pEnd, nMid);
            const faceSpan = pStart.distanceTo(pEnd);

            // Horizontal upper transom bar at 70% window height
            pMid.lerpVectors(pStart, pEnd, 0.5);
            const hNorm = new THREE.Vector3(nMid.x, 0, nMid.z).normalize();
            const transomGeo = new THREE.BoxGeometry(faceSpan, 0.065, 0.075);
            SecondarySubdomains._transformGeometry(
              transomGeo,
              pMid.x + hNorm.x * 0.048,
              pLo.y + winH * 0.72,
              pMid.z + hNorm.z * 0.048,
              faceYaw
            );
            addMesh(transomGeo, materials.precastConcrete, `Applique_WinTransom_F${floor}_Q${q}_${tag}`);

            for (let b = 0; b <= 6; b++) {
              const uM = uFaceMin + (uFaceMax - uFaceMin) * (b / 6.0);
              evalMaster(uM, 0.5 * (glassSeg.vMin + glassSeg.vMax), pMid, nMid);
              const mNorm = new THREE.Vector3(nMid.x, 0, nMid.z).normalize();
              const mullionGeo = new THREE.BoxGeometry(0.085, winH, 0.095);
              SecondarySubdomains._transformGeometry(
                mullionGeo,
                pMid.x + mNorm.x * 0.052,
                winMidY,
                pMid.z + mNorm.z * 0.052,
                faceYaw
              );
              addMesh(mullionGeo, materials.precastConcrete, `Applique_WinMullion_F${floor}_Q${q}_${tag}_${b}`);
            }
          }
        }
      }

      // D. Physical 3D Vertical Window Mullions on the 4 Flat Annex Blocks & 2 Pyramidal Annex Halls
      const annexWindowBoxes = [
        { tag: 'WestNorthFlat', cx: -32.25, cz: -11.2, w: 22.14, d: 14.84, yMid: 4.3, h: 1.58 },
        { tag: 'WestSouthFlat', cx: -32.25, cz: 9.4, w: 22.14, d: 13.24, yMid: 4.3, h: 1.58 },
        { tag: 'EastNorthFlat', cx: 32.25, cz: -14.2, w: 22.14, d: 14.84, yMid: 4.3, h: 1.58 },
        { tag: 'EastSouthFlat', cx: 32.25, cz: 6.4, w: 22.14, d: 13.24, yMid: 4.3, h: 1.58 },
        { tag: 'WestHallWin', cx: -43.75, cz: -1.3, w: 18.05, d: 21.95, yMid: 4.35, h: 1.62 },
        { tag: 'EastHallWin', cx: 43.75, cz: -4.3, w: 18.05, d: 21.95, yMid: 4.35, h: 1.62 }
      ];
      for (const ab of annexWindowBoxes) {
        const nX = Math.max(3, Math.round(ab.w / 2.4));
        for (let ix = 0; ix <= nX; ix++) {
          const lx = -ab.w * 0.5 + (ab.w * ix) / nX;
          for (const sz of [-1, 1]) {
            const mZ = new THREE.BoxGeometry(0.08, ab.h, 0.1);
            mZ.translate(ab.cx + lx, ab.yMid, ab.cz + sz * (ab.d * 0.5 + 0.03));
            addMesh(mZ, materials.precastConcrete, `Applique_AnnexMullionZ_${ab.tag}_${ix}_${sz}`);
          }
        }
        const nZ = Math.max(3, Math.round(ab.d / 2.4));
        for (let iz = 1; iz < nZ; iz++) {
          const lz = -ab.d * 0.5 + (ab.d * iz) / nZ;
          for (const sx of [-1, 1]) {
            const mX = new THREE.BoxGeometry(0.1, ab.h, 0.08);
            mX.translate(ab.cx + sx * (ab.w * 0.5 + 0.03), ab.yMid, ab.cz + lz);
            addMesh(mX, materials.precastConcrete, `Applique_AnnexMullionX_${ab.tag}_${iz}_${sx}`);
          }
        }
      }
    }

    // =========================================================================
    // 4. PHYSICAL 3D X-DIAMOND ATRIUM CURTAIN-WALL LATTICE & OCTAGONAL SKYLIGHT RIBS
    // =========================================================================
    if (isUltra) {
      const atriumShafts = [
        { tag: 'South', cx: 0.0, cz: 13.06, yaw: 0.0 },
        { tag: 'East', cx: 13.06, cz: 0.0, yaw: Math.PI * 0.5 },
        { tag: 'North', cx: 0.0, cz: -13.06, yaw: Math.PI },
        { tag: 'West', cx: -13.06, cz: 0.0, yaw: -Math.PI * 0.5 }
      ];
      const slotSpanW = 4.44;
      const yAtriumBot = 3.45;
      const yAtriumTop = 25.05;
      const tiers = 6;
      const tierH = (yAtriumTop - yAtriumBot) / tiers;
      const diagLen = Math.hypot(slotSpanW, tierH);
      const diagAngle = Math.atan2(slotSpanW, tierH);

      for (const sh of atriumShafts) {
        // Portal architrave beam above ground lobby entrance
        const portalBeam = new THREE.BoxGeometry(slotSpanW + 0.24, 0.24, 0.22);
        SecondarySubdomains._transformGeometry(portalBeam, sh.cx, yAtriumBot - 0.1, sh.cz, sh.yaw);
        addMesh(portalBeam, materials.precastConcrete, `AtriumLattice_PortalBeam_${sh.tag}`);

        // Twin vertical border jambs + central vertical spine mullion
        for (const lx of [-slotSpanW * 0.5, 0.0, slotSpanW * 0.5]) {
          const vJamb = new THREE.BoxGeometry(lx === 0 ? 0.075 : 0.11, yAtriumTop - yAtriumBot, 0.11);
          vJamb.translate(lx, 0, 0);
          SecondarySubdomains._transformGeometry(
            vJamb,
            sh.cx,
            0.5 * (yAtriumBot + yAtriumTop),
            sh.cz,
            sh.yaw
          );
          addMesh(vJamb, materials.bronzeGold, `AtriumLattice_VJamb_${sh.tag}_${lx}`);
        }

        // 6 Tiers of Crisscrossing 3D X-Diamond Structural Mullions + Horizontal Transoms
        for (let t = 0; t < tiers; t++) {
          const yMid = yAtriumBot + (t + 0.5) * tierH;
          const yTop = yAtriumBot + (t + 1.0) * tierH;

          const hBar = new THREE.BoxGeometry(slotSpanW, 0.075, 0.095);
          SecondarySubdomains._transformGeometry(hBar, sh.cx, yTop, sh.cz, sh.yaw);
          addMesh(hBar, materials.bronzeGold, `AtriumLattice_Transom_${sh.tag}_${t}`);

          for (const sgn of [-1, 1]) {
            const diagStrut = new THREE.BoxGeometry(0.08, diagLen, 0.09);
            diagStrut.rotateZ(sgn * diagAngle);
            SecondarySubdomains._transformGeometry(diagStrut, sh.cx, yMid, sh.cz, sh.yaw);
            addMesh(diagStrut, materials.bronzeGold, `AtriumLattice_XDiag_${sh.tag}_${t}_${sgn}`);
          }
        }
      }

      // 8 Physical 3D Radial Structural Rafters on the 45° Octagonal Glass Pyramid Skylight (29.5, 29.5)
      const skyBaseR = 2.25;
      const skyTopR = 0.08;
      const skyBaseY = 6.85;
      const skyTopY = 8.15;
      const skyRafterLen = Math.hypot(skyBaseR - skyTopR, skyTopY - skyBaseY);
      const skyPitch = Math.atan2(skyBaseR - skyTopR, skyTopY - skyBaseY);
      for (let k = 0; k < 8; k++) {
        const ang = (k + 0.5) * (Math.PI * 0.25);
        const rMid = 0.5 * (skyBaseR + skyTopR);
        const yMid = 0.5 * (skyBaseY + skyTopY);
        const rafter = new THREE.BoxGeometry(0.075, skyRafterLen, 0.085);
        rafter.rotateX(-skyPitch);
        SecondarySubdomains._transformGeometry(
          rafter,
          29.5 + Math.sin(ang) * rMid,
          yMid + 0.02,
          29.5 + Math.cos(ang) * rMid,
          ang
        );
        addMesh(rafter, materials.bronzeGold, `OctSkylight_RadialRafter_${k}`);
      }
      const skyApexCap = new THREE.CylinderGeometry(0.05, 0.18, 0.24, 16, 1);
      skyApexCap.translate(29.5, skyTopY + 0.08, 29.5);
      addMesh(skyApexCap, materials.bronzeGold, 'OctSkylight_ApexFinial');
    }

    // =========================================================================
    // 5. TRUE 3D SCULPTED BAS-RELIEF MAKARA UI CREST (KALPATARU & PENTAGONAL SHIELD)
    //    ON ALL 4 CROWN WALLS (y = 26.2m .. 32.2m!)
    // =========================================================================
    const crownSeg = getStationSegmentByZone('CROWN_LOWER_WALL');
    if (crownSeg) {
      const vCrownMid = 0.5 * (crownSeg.vMin + crownSeg.vMax);
      const cardinalSides = [
        { uCenter: 0.0, tag: 'SouthRotunda', yaw: 0.0 },
        { uCenter: 0.25, tag: 'EastDanauKenanga', yaw: Math.PI * 0.5 },
        { uCenter: 0.5, tag: 'NorthCampus', yaw: Math.PI },
        { uCenter: 0.75, tag: 'WestAnnex', yaw: -Math.PI * 0.5 }
      ];

      // Build shared local-space 3D Bas-Relief Makara Kalpataru template once for LOD0
      const makara3DReliefTemplate = isUltra
        ? SecondarySubdomains._createMakara3DReliefGeometry()
        : null;

      for (const side of cardinalSides) {
        const pAnchor = new THREE.Vector3();
        const nAnchor = new THREE.Vector3();
        evalMaster(side.uCenter, vCrownMid, pAnchor, nAnchor);
        const horizNorm = new THREE.Vector3(nAnchor.x, 0, nAnchor.z).normalize();

        // Crisp off-white architectural vertical panel frame between the adjacent corner roofs
        const panelCenter = pAnchor.clone().addScaledVector(horizNorm, 0.09);
        panelCenter.y = 29.35;
        const backingPanel = new THREE.BoxGeometry(3.86, 5.45, 0.16);
        SecondarySubdomains._transformGeometry(
          backingPanel,
          panelCenter.x,
          panelCenter.y,
          panelCenter.z,
          side.yaw
        );
        addMesh(backingPanel, materials.precastConcrete, `Applique_MakaraPanelFrame_${side.tag}`);

        if (isUltra) {
          // Molded outer architrave surround & upper cornice cap around Makara panel
          for (const sx of [-1.85, 1.85]) {
            const sideMolding = new THREE.BoxGeometry(0.18, 5.55, 0.24);
            sideMolding.translate(sx, 0, 0);
            SecondarySubdomains._transformGeometry(
              sideMolding,
              panelCenter.x,
              panelCenter.y,
              panelCenter.z,
              side.yaw
            );
            addMesh(sideMolding, materials.precastConcrete, `Applique_MakaraSideMold_${side.tag}_${sx}`);
          }
          for (const [dy, hM, dM, mTag] of [
            [2.74, 0.22, 0.28, 'TopCornice'],
            [-2.72, 0.2, 0.26, 'BotPlinth']
          ]) {
            const hMold = new THREE.BoxGeometry(4.08, hM, dM);
            SecondarySubdomains._transformGeometry(
              hMold,
              panelCenter.x,
              panelCenter.y + dy,
              panelCenter.z,
              side.yaw
            );
            addMesh(hMold, materials.precastConcrete, `Applique_Makara${mTag}_${side.tag}`);
          }
        }

        // Non-mirrored Backlit Halo Plane (+0.185m outward along N_master, Hukum Emas 17)
        const emblemPos = pAnchor.clone().addScaledVector(horizNorm, 0.185);
        emblemPos.y = 29.65;
        const faceGeo = new THREE.PlaneGeometry(3.55, 3.55);
        SecondarySubdomains._transformGeometry(
          faceGeo,
          emblemPos.x,
          emblemPos.y,
          emblemPos.z,
          side.yaw
        );
        addMesh(faceGeo, materials.makaraEmblem, `Applique_MakaraEmblem_${side.tag}`);

        // True 3D Sculpted Bas-Relief Makara Kalpataru Crest proud of the backlit halo (+0.205m outward!)
        if (isUltra && makara3DReliefTemplate) {
          const reliefPos = pAnchor.clone().addScaledVector(horizNorm, 0.205);
          reliefPos.y = 29.65;
          const reliefGeo = makara3DReliefTemplate.clone();
          SecondarySubdomains._transformGeometry(
            reliefGeo,
            reliefPos.x,
            reliefPos.y,
            reliefPos.z,
            side.yaw
          );
          addMesh(reliefGeo, materials.bronzeGold, `Applique_Makara3DSculptedRelief_${side.tag}`);
        }
      }
    }

    // =========================================================================
    // 6. TALL 2-STORY SCULPTED GROUND-FLOOR COLONNADE PILLARS, PAGODA EAVE BRACKETS,
    //    & RECESSED COURTYARD PLANTER TROUGHS (Mavic Photo #2!)
    // =========================================================================
    const wingPillarCoords = [2.85, 5.33, 7.81, 10.29, 12.77, 15.25];
    const wingOutCoord = 15.42;
    const pillarBotY = 0.36;
    const pillarH = 5.32;

    for (const q of CORNER_QUADRANTS) {
      for (let i = 0; i < wingPillarCoords.length; i++) {
        const c = wingPillarCoords[i];
        SecondarySubdomains._addSculptedSquarePillar(
          addMesh,
          materials,
          q.sx * c,
          pillarBotY,
          q.sz * wingOutCoord,
          0.68,
          pillarH,
          0.72,
          `TowerColonnade_ZFace_${q.id}_${i}`,
          isUltra
        );

        if (i < wingPillarCoords.length - 1) {
          SecondarySubdomains._addSculptedSquarePillar(
            addMesh,
            materials,
            q.sx * wingOutCoord,
            pillarBotY,
            q.sz * c,
            0.72,
            pillarH,
            0.68,
            `TowerColonnade_XFace_${q.id}_${i}`,
            isUltra
          );
        }
      }

      // Sculpted Planter Troughs with Molded Coping Rim & Recessed Dark Soil Bed (y = 0.36 .. 0.96m)
      SecondarySubdomains._addPlanterTrough(
        addMesh,
        materials,
        q.sx * 9.05,
        q.sz * 15.15,
        12.2,
        0.85,
        `TowerBasePlanter_Z_${q.id}`,
        isUltra
      );
      SecondarySubdomains._addPlanterTrough(
        addMesh,
        materials,
        q.sx * 15.15,
        q.sz * 9.05,
        0.85,
        12.2,
        `TowerBasePlanter_X_${q.id}`,
        isUltra
      );
    }

    // Decorative 3D Nusantara Pagoda Eave Corbel Brackets ('Konsol Teritisan') under all 6 Pagoda Tiers + Tajug!
    if (isUltra) {
      const pBr = new THREE.Vector3();
      const nBr = new THREE.Vector3();
      for (let floor = 2; floor <= 7; floor++) {
        const soffitSeg = getStationSegmentByZone(`EAVE_${floor - 1}_SOFFIT`);
        if (!soffitSeg) continue;
        const vMid = 0.35 * soffitSeg.vMin + 0.65 * soffitSeg.vMax;
        for (let q = 0; q < 4; q++) {
          const qBase = q * 0.25;
          for (let b = 0; b < 12; b++) {
            const uB = qBase + (1.15 + (b / 11.0) * 4.7) * duFacet;
            evalMaster(uB, vMid, pBr, nBr);
            const outDir = new THREE.Vector3(pBr.x, 0, pBr.z).normalize();
            const yaw = Math.atan2(outDir.x, outDir.z);
            const corbel = new THREE.BoxGeometry(0.14, 0.2, 0.58);
            SecondarySubdomains._transformGeometry(
              corbel,
              pBr.x - outDir.x * 0.12,
              pBr.y - 0.08,
              pBr.z - outDir.z * 0.12,
              yaw
            );
            addMesh(corbel, materials.precastConcrete, `PagodaEaveCorbel_F${floor}_Q${q}_${b}`);
          }
        }
      }

      const tajugSoffit = getStationSegmentByZone('TAJUG_MAIN_SOFFIT');
      if (tajugSoffit) {
        const vMid = 0.4 * tajugSoffit.vMin + 0.6 * tajugSoffit.vMax;
        for (let b = 0; b < 36; b++) {
          const uB = (b + 0.5) / 36.0;
          evalMaster(uB, vMid, pBr, nBr);
          const outDir = new THREE.Vector3(pBr.x, 0, pBr.z).normalize();
          const yaw = Math.atan2(outDir.x, outDir.z);
          const corbel = new THREE.BoxGeometry(0.16, 0.24, 0.74);
          SecondarySubdomains._transformGeometry(
            corbel,
            pBr.x - outDir.x * 0.16,
            pBr.y - 0.1,
            pBr.z - outDir.z * 0.16,
            yaw
          );
          addMesh(corbel, materials.precastConcrete, `TajugEaveCorbel_${b}`);
        }
      }
    }

    // Main Tower +Z Ceremonial Lobby Entry Threshold Deck
    const entryDeck = new THREE.PlaneGeometry(9.6, 5.2, isUltra ? 8 : 1, isUltra ? 4 : 1);
    SecondarySubdomains._rotatePlaneToXZ(entryDeck);
    entryDeck.translate(-1.2, 0.37, 15.4);
    addMesh(entryDeck, materials.horizontalPlazaStone, 'TowerFrontEntry_StoneThreshold', false, true);

    const entryStep = new THREE.BoxGeometry(9.6, 0.55, 5.2);
    entryStep.translate(-1.2, 0.08, 15.4);
    addMesh(entryStep, materials.precastConcrete, 'TowerFrontEntry_Riser');

    // =========================================================================
    // 7. SELASAR AGUNG, 4-SIDED FLAG CLOISTER, & FRONT ROTUNDA GATEHOUSE COLONNADE
    //    + EXPOSED 3D CEILING RAFTERS & PURLINS ('KASAU & GORDING')
    // =========================================================================
    // 7A. Selasar Agung (x = -2.5 ± 3.35m, z = 17.5 .. 37.5m, y = 0.12 .. 4.90m)
    const selasarZList = [17.5, 21.5, 25.5, 29.5, 33.5, 37.5];
    for (let i = 0; i < selasarZList.length; i++) {
      const pz = selasarZList[i];
      for (const sx of [-1, 1]) {
        SecondarySubdomains._addSculptedSquarePillar(
          addMesh,
          materials,
          -2.5 + sx * 3.35,
          0.12,
          pz,
          0.54,
          4.78,
          0.54,
          `SelasarAgung_Col_${i}_${sx}`,
          isUltra
        );
      }
    }

    if (isUltra) {
      // Exposed 3D Ceiling Trusses, Tie-Beams, & Rafters under Selasar Agung High Gable Roof
      SecondarySubdomains._addGableCeilingRafters(addMesh, materials, {
        tag: 'SelasarAgung',
        cx: -2.5,
        cz: 27.5,
        length: 23.2,
        halfWidth: 3.95,
        eaveY: 4.92,
        ridgeY: 9.15,
        rafterPairs: 15,
        alongAxis: 'Z'
      });
    }

    // 7B. 4-Sided Rectangular Flag Cloister ('Pelataran Panji Fakultas UI')
    const cloisterXCols = [-15.8, -11.4, -7.0, -2.5, 2.0, 6.4, 10.8];
    for (let i = 0; i < cloisterXCols.length; i++) {
      const px = cloisterXCols[i];
      for (const pz of [43.3, 59.1]) {
        SecondarySubdomains._addSculptedSquarePillar(
          addMesh,
          materials,
          px,
          0.12,
          pz,
          0.42,
          3.48,
          0.42,
          `Cloister_NS_InnerCol_${i}_${pz}`,
          isUltra
        );
      }
    }
    const cloisterZCols = [46.8, 51.2, 55.6];
    for (let i = 0; i < cloisterZCols.length; i++) {
      const pz = cloisterZCols[i];
      for (const px of [-15.8, -20.0, 10.8, 15.0]) {
        SecondarySubdomains._addSculptedSquarePillar(
          addMesh,
          materials,
          px,
          0.12,
          pz,
          0.42,
          3.48,
          0.42,
          `Cloister_EW_Col_${i}_${px}`,
          isUltra
        );
      }
    }

    if (isUltra) {
      // Exposed 3D Ceiling Rafters under all 4 Flag Cloister Galleries
      SecondarySubdomains._addGableCeilingRafters(addMesh, materials, {
        tag: 'CloisterNorth', cx: -2.5, cz: 41.4, length: 35.2, halfWidth: 2.2, eaveY: 3.62, ridgeY: 5.38, rafterPairs: 20, alongAxis: 'X'
      });
      SecondarySubdomains._addGableCeilingRafters(addMesh, materials, {
        tag: 'CloisterSouth', cx: -2.5, cz: 61.1, length: 35.2, halfWidth: 2.2, eaveY: 3.62, ridgeY: 5.38, rafterPairs: 20, alongAxis: 'X'
      });
      SecondarySubdomains._addGableCeilingRafters(addMesh, materials, {
        tag: 'CloisterWest', cx: -18.1, cz: 51.25, length: 23.6, halfWidth: 2.2, eaveY: 3.62, ridgeY: 5.38, rafterPairs: 14, alongAxis: 'Z'
      });
      SecondarySubdomains._addGableCeilingRafters(addMesh, materials, {
        tag: 'CloisterEast', cx: 13.1, cz: 51.25, length: 23.6, halfWidth: 2.2, eaveY: 3.62, ridgeY: 5.38, rafterPairs: 14, alongAxis: 'Z'
      });
    }

    // 7C. Front Rotunda Gatehouse Pavilion ('Gerbang Depan', cx = -2.5, cz = 65.5, strictly z <= 71.90m!)
    const gateCols = [
      [-7.4, 60.8], [-2.5, 60.8], [2.4, 60.8],
      [-7.4, 65.5],               [2.4, 65.5],
      [-7.4, 70.2], [-4.4, 70.2], [-0.6, 70.2], [2.4, 70.2]
    ];
    for (let i = 0; i < gateCols.length; i++) {
      const [gx, gz] = gateCols[i];
      SecondarySubdomains._addSculptedSquarePillar(
        addMesh,
        materials,
        gx,
        0.14,
        gz,
        0.52,
        3.68,
        0.52,
        `FrontGatehouse_Col_${i}`,
        isUltra
      );
    }

    const gateDeck = new THREE.PlaneGeometry(11.2, 10.8, isUltra ? 8 : 1, isUltra ? 8 : 1);
    SecondarySubdomains._rotatePlaneToXZ(gateDeck);
    gateDeck.translate(-2.5, 0.24, 65.5);
    addMesh(gateDeck, materials.horizontalPlazaStone, 'FrontGatehouse_StoneFloor', false, true);

    const gateThresholdTread = new THREE.PlaneGeometry(11.6, 0.9);
    SecondarySubdomains._rotatePlaneToXZ(gateThresholdTread);
    gateThresholdTread.translate(-2.5, 0.13, 71.35);
    addMesh(gateThresholdTread, materials.horizontalPlazaStone, 'FrontGatehouse_StepTread', false, true);

    const gateThresholdRiser = new THREE.BoxGeometry(11.6, 0.35, 0.9);
    gateThresholdRiser.translate(-2.5, -0.05, 71.35);
    addMesh(gateThresholdRiser, materials.precastConcrete, 'FrontGatehouse_StepRiser');

    // =========================================================================
    // 8. 45° SE DIAGONAL BREEZEWAY, FLUTED OCTAGONAL PENDOPO, & NORTH PAVILION
    // =========================================================================
    // 8A. 45-Degree SE Breezeway Columns & Ceiling Rafters
    for (let s = 0; s < 4; s++) {
      const dist = -6.5 + s * 4.3;
      for (const side of [-1.75, 1.75]) {
        const cos45 = Math.SQRT1_2;
        const wx = 19.2 + side * cos45 + dist * cos45;
        const wz = 19.2 - side * cos45 + dist * cos45;
        SecondarySubdomains._addSculptedSquarePillar(
          addMesh,
          materials,
          wx,
          0.12,
          wz,
          0.4,
          3.56,
          0.4,
          `SE45Breezeway_Col_${s}_${side}`,
          isUltra,
          Math.PI * 0.25
        );
      }
    }
    if (isUltra) {
      SecondarySubdomains._addGableCeilingRafters(addMesh, materials, {
        tag: 'SE45Breezeway',
        cx: 19.2,
        cz: 19.2,
        length: 16.8,
        halfWidth: 2.15,
        eaveY: 3.7,
        ridgeY: 5.38,
        rafterPairs: 11,
        alongAxis: 'Z',
        rotY: Math.PI * 0.25
      });
    }

    // 8B. Octagonal Glass-Skylit Pendopo at (29.5, 29.5): 8 Fluted Classical Nusantara Round Columns
    if (!isUltra) {
      for (let k = 0; k < 8; k++) {
        const ang = (k + 0.5) * (Math.PI * 0.25);
        const px = 29.5 + 6.55 * Math.cos(ang);
        const pz = 29.5 + 6.55 * Math.sin(ang);
        const octCol = new THREE.CylinderGeometry(0.26, 0.29, 3.68, 8, 1);
        octCol.translate(px, 0.18 + 1.84, pz);
        addMesh(octCol, materials.precastConcrete, `OctSkylitPendopo_Col_${k}`);
      }
    } else {
      const flutedColTemplate = SecondarySubdomains._createFlutedRoundColumnGeometry(0.26, 0.30, 3.68);
      for (let k = 0; k < 8; k++) {
        const ang = (k + 0.5) * (Math.PI * 0.25);
        const px = 29.5 + 6.55 * Math.cos(ang);
        const pz = 29.5 + 6.55 * Math.sin(ang);
        const colGeo = flutedColTemplate.clone();
        colGeo.translate(px, 0.18, pz);
        addMesh(colGeo, materials.precastConcrete, `OctSkylitPendopo_FlutedCol_${k}`);

        // Radial ceiling tie-beam from each perimeter column inward to the octagonal skylight ring
        const rMid = 0.5 * (6.55 + 2.25);
        const beam = new THREE.BoxGeometry(0.18, 0.22, 6.55 - 2.25);
        SecondarySubdomains._transformGeometry(
          beam,
          29.5 + rMid * Math.sin(ang),
          3.78,
          29.5 + rMid * Math.cos(ang),
          ang
        );
        addMesh(beam, materials.precastConcrete, `OctSkylitPendopo_RadialBeam_${k}`);
      }
    }

    const octPlinth = new THREE.CylinderGeometry(7.25, 7.45, 2.8, isUltra ? 32 : 8, 1);
    octPlinth.rotateY(Math.PI / 8);
    octPlinth.translate(29.5, 0.22 - 1.4, 29.5);
    addMesh(octPlinth, materials.precastConcrete, 'OctSkylitPendopo_PlinthSkirt');

    const octFloor = new THREE.CylinderGeometry(7.15, 7.15, 0.04, isUltra ? 32 : 8, 1);
    octFloor.rotateY(Math.PI / 8);
    octFloor.translate(29.5, 0.23, 29.5);
    addMesh(octFloor, materials.horizontalPlazaStone, 'OctSkylitPendopo_StoneFloor', false, true);

    // 8C. North Standalone Pyramidal Pavilion Columns at (5.0, -43.5) — strictly z >= -50.8m > -52.0m!
    for (const lx of [-6.8, 0.0, 6.8]) {
      for (const lz of [-6.8, 0.0, 6.8]) {
        if (lx === 0.0 && lz === 0.0) continue;
        SecondarySubdomains._addSculptedSquarePillar(
          addMesh,
          materials,
          5.0 + lx,
          0.15,
          -43.5 + lz,
          0.46,
          3.65,
          0.46,
          `NorthPavilion_Col_${lx}_${lz}`,
          isUltra
        );
      }
    }

    // 8D. Crisp White Perimeter Curbs Framing the ◇◇◇ Diamond Courtyard Plaza
    const curbSpecs = [
      { w: 72.4, d: 0.55, x: 0.0, z: -35.0 },
      { w: 0.55, d: 74.0, x: -36.0, z: 2.0 },
      { w: 0.55, d: 74.0, x: 36.0, z: 2.0 }
    ];
    for (let i = 0; i < curbSpecs.length; i++) {
      const cs = curbSpecs[i];
      const curb = new THREE.BoxGeometry(cs.w, 0.34, cs.d);
      curb.translate(cs.x, 0.16, cs.z);
      addMesh(curb, materials.precastConcrete, `CourtyardPerimeterCurb_${i}`);
    }

    // =========================================================================
    // 9. ROOFTOP ARCHITECTURAL MECHANICAL LOUVER ENCLOSURES & ACCESS HATCHES
    //    ON THE 4 WEATHERED FLAT ANNEX ROOFS (Tucked inside 6.32m Parapets!)
    // =========================================================================
    if (isUltra) {
      const flatDeckUnits = [
        { tag: 'WestNorth', cx: -31.5, cz: -11.2, roofY: 5.77 },
        { tag: 'WestSouth', cx: -31.5, cz: 9.4, roofY: 5.77 },
        { tag: 'EastNorth', cx: 31.5, cz: -14.2, roofY: 5.77 },
        { tag: 'EastSouth', cx: 31.5, cz: 6.4, roofY: 5.77 }
      ];
      for (const u of flatDeckUnits) {
        // Low-profile architectural MEP Louver Screen (top y = 6.23m < 6.32m parapet top)
        const curbPad = new THREE.BoxGeometry(4.4, 0.12, 2.8);
        curbPad.translate(u.cx, u.roofY + 0.06, u.cz);
        addMesh(curbPad, materials.precastConcrete, `RoofMEP_CurbPad_${u.tag}`);

        const housing = new THREE.BoxGeometry(4.1, 0.34, 2.5);
        housing.translate(u.cx, u.roofY + 0.29, u.cz);
        addMesh(housing, materials.flatRoofSlab, `RoofMEP_Housing_${u.tag}`);

        // 7 Slatted Ventilation Louver Blades on top & sides
        for (let k = -3; k <= 3; k++) {
          const blade = new THREE.BoxGeometry(3.85, 0.035, 0.16);
          blade.rotateX(0.32);
          blade.translate(u.cx, u.roofY + 0.47, u.cz + k * 0.32);
          addMesh(blade, materials.bronzeGold, `RoofMEP_LouverBlade_${u.tag}_${k}`);
        }

        // Roof Access Stair/Service Hatch nearby
        const hatch = new THREE.BoxGeometry(1.35, 0.28, 1.15);
        hatch.translate(u.cx + 4.8 * (u.cx < 0 ? 1 : -1), u.roofY + 0.14, u.cz + 1.8);
        addMesh(hatch, materials.precastConcrete, `RoofMEP_AccessHatch_${u.tag}`);
      }

      // =========================================================================
      // 10. NEXUS v2.5 — 100x PHYSICAL 3D ARCHITECTURAL FACADE & SURFACE ARTICULATION
      //     (Precast Drip Stringcourses, Vertical Spandrel Pilaster Ribs, Stepped Eave Fascia
      //     Moldings, Coffered Ground Colonnade Ceilings + Recessed Downlights, 2-Story Lobby
      //     Curtain-Wall Mullions, 3-Tier Ceremonial Entry Staircase + Cheek-Walls + Grand Portal
      //     Doors, Tropical Annex Brise-Soleil Fins & Sun-Hoods, Crown Clerestory Louvers,
      //     Makara Floodlights, & Recessed Under-Eave Exposed Rafter Tails!)
      // =========================================================================
      SecondarySubdomains._addUltraArchitecturalFacadeArticulation(
        addMesh,
        materials,
        evalMaster,
        duFacet
      );
    }

    return group;
  }

  /**
   * NEXUS v2.5 (100x Surface & Architectural Facade Detail):
   * Adds >1,350 physical 3D architectural facade elements across Gedung Rektorat UI while strictly
   * preserving the exact 108.00 x 124.00 x 50.00m envelope and 13 batched Draw Calls.
   */
  static _addUltraArchitecturalFacadeArticulation(addMesh, materials, evalMaster, duFacet) {
    const pA = new THREE.Vector3();
    const pB = new THREE.Vector3();
    const pMid = new THREE.Vector3();
    const nMid = new THREE.Vector3();

    // -------------------------------------------------------------------------
    // A. 6-FLOOR PAGODA PRECAST DRIP STRINGCOURSES ('TALI AIR'), VERTICAL SPANDREL
    //    PILASTER RIBS (336 RIBS), & STEPPED EAVE FASCIA LISPLANK MOLDINGS
    // -------------------------------------------------------------------------
    for (let floor = 2; floor <= 7; floor++) {
      const tNum = floor - 1;
      const fasciaSeg = getStationSegmentByZone(`EAVE_${tNum}_FASCIA`);
      const spandrelSeg = getStationSegmentByZone(`FLOOR_${floor}_SPANDREL`);

      // A1. Crisp Stepped Nusantara Eave Fascia Trim ('Lisplank Bertingkat') around all 4 quadrants
      if (fasciaSeg) {
        for (let q = 0; q < 4; q++) {
          const uMin = q * 0.25 + 1.04 * duFacet;
          const uMax = q * 0.25 + 5.96 * duFacet;
          const fasciaTrim = MasterSurfaceBuilder.buildNormalOffsetAppliqueBand(
            evalMaster,
            uMin,
            uMax,
            fasciaSeg.vMin + 0.15 * (fasciaSeg.vMax - fasciaSeg.vMin),
            fasciaSeg.vMax,
            18,
            1,
            0.042,
            { closeU: true, closeVBottom: true, closeVTop: true }
          );
          addMesh(fasciaTrim, materials.precastConcrete, `FacadeV25_EaveFasciaTrim_F${floor}_Q${q}`);
        }
      }

      // A2. Horizontal Precast Drip-Reveal Stringcourse ('Tali Air') + 7 Vertical Spandrel Pilaster Ribs per Face
      if (spandrelSeg) {
        const vSpan = spandrelSeg.vMax - spandrelSeg.vMin;
        const vDripLo = spandrelSeg.vMin + 0.34 * vSpan;
        const vDripHi = spandrelSeg.vMin + 0.56 * vSpan;

        for (let q = 0; q < 4; q++) {
          const qBase = q * 0.25;
          for (const [sA, sB, tag] of [
            [2.02, 2.98, 'FaceA'],
            [4.02, 4.98, 'FaceB']
          ]) {
            const uFaceMin = qBase + sA * duFacet;
            const uFaceMax = qBase + sB * duFacet;

            // Normal-offset horizontal precast spandrel apron band
            const spandrelBand = MasterSurfaceBuilder.buildNormalOffsetAppliqueBand(
              evalMaster,
              uFaceMin,
              uFaceMax,
              vDripLo,
              vDripHi,
              10,
              1,
              0.046,
              { closeU: true, closeVBottom: true, closeVTop: true }
            );
            addMesh(spandrelBand, materials.precastConcrete, `FacadeV25_SpandrelBand_F${floor}_Q${q}_${tag}`);

            // 7 Vertical Precast Spandrel Joint Ribs aligned directly beneath the 7 window mullions
            evalMaster(0.5 * (uFaceMin + uFaceMax), spandrelSeg.vMin, pA, nMid);
            evalMaster(0.5 * (uFaceMin + uFaceMax), spandrelSeg.vMax, pB, nMid);
            const spandrelH = Math.max(0.24, pB.y - pA.y - 0.04);
            const spandrelMidY = 0.5 * (pA.y + pB.y);
            const faceYaw = Math.atan2(nMid.x, nMid.z);

            for (let b = 0; b <= 6; b++) {
              const uRib = uFaceMin + (uFaceMax - uFaceMin) * (b / 6.0);
              evalMaster(uRib, 0.5 * (spandrelSeg.vMin + spandrelSeg.vMax), pMid, nMid);
              const rNorm = new THREE.Vector3(nMid.x, 0, nMid.z).normalize();
              const ribGeo = new THREE.BoxGeometry(0.105, spandrelH, 0.056);
              SecondarySubdomains._transformGeometry(
                ribGeo,
                pMid.x + rNorm.x * 0.028,
                spandrelMidY,
                pMid.z + rNorm.z * 0.028,
                faceYaw
              );
              addMesh(ribGeo, materials.precastConcrete, `FacadeV25_SpandrelRib_F${floor}_Q${q}_${tag}_${b}`);
            }
          }
        }
      }
    }

    // Main Steep Tajug Spire Eave Fascia Lisplank Molding (y = 32.88 .. 33.10m)
    const tajugFasciaSeg = getStationSegmentByZone('TAJUG_MAIN_FASCIA');
    if (tajugFasciaSeg) {
      for (let q = 0; q < 4; q++) {
        const tajugTrim = MasterSurfaceBuilder.buildNormalOffsetAppliqueBand(
          evalMaster,
          q * 0.25,
          (q + 1) * 0.25,
          tajugFasciaSeg.vMin + 0.18 * (tajugFasciaSeg.vMax - tajugFasciaSeg.vMin),
          tajugFasciaSeg.vMax,
          16,
          1,
          0.052,
          { closeU: true, closeVBottom: true, closeVTop: true }
        );
        addMesh(tajugTrim, materials.precastConcrete, `FacadeV25_TajugFasciaTrim_Q${q}`);
      }
    }

    // -------------------------------------------------------------------------
    // B. 2-STORY GROUND COLONNADE COFFERED CEILING GRID, RECESSED SOFFIT DOWNLIGHT
    //    BEZELS, & RECESSED GROUND-FLOOR LOBBY CURTAIN-WALL MULLION LATTICE
    // -------------------------------------------------------------------------
    const colCoords = [2.85, 5.33, 7.81, 10.29, 12.77, 15.25];
    for (const q of CORNER_QUADRANTS) {
      // B1. Outer Perimeter Colonnade Architrave Tie-Beams (Z-face & X-face at y = 5.48m)
      const archBeamZ = new THREE.BoxGeometry(13.15, 0.34, 0.56);
      archBeamZ.translate(q.sx * 9.05, 5.48, q.sz * 15.42);
      addMesh(archBeamZ, materials.precastConcrete, `ColonnadeV25_ArchitraveZ_${q.id}`);

      const archBeamX = new THREE.BoxGeometry(0.56, 0.34, 13.15);
      archBeamX.translate(q.sx * 15.42, 5.48, q.sz * 9.05);
      addMesh(archBeamX, materials.precastConcrete, `ColonnadeV25_ArchitraveX_${q.id}`);

      // B2. Transverse Coffer Ribs connecting outer pillars inward to the recessed lobby wall
      for (let i = 0; i < colCoords.length; i++) {
        const c = colCoords[i];
        const cofferZ = new THREE.BoxGeometry(0.32, 0.26, 1.32);
        cofferZ.translate(q.sx * c, 5.52, q.sz * 14.86);
        addMesh(cofferZ, materials.precastConcrete, `ColonnadeV25_CofferZ_${q.id}_${i}`);

        const cofferX = new THREE.BoxGeometry(1.32, 0.26, 0.32);
        cofferX.translate(q.sx * 14.86, 5.52, q.sz * c);
        addMesh(cofferX, materials.precastConcrete, `ColonnadeV25_CofferX_${q.id}_${i}`);

        // B3. Recessed Architectural Soffit Downlight Bezels & Warm Lenses centered in each coffer bay
        if (i < colCoords.length - 1) {
          const cMid = 0.5 * (colCoords[i] + colCoords[i + 1]);
          for (const [dx, dz, bTag] of [
            [q.sx * cMid, q.sz * 14.88, 'ZBay'],
            [q.sx * 14.88, q.sz * cMid, 'XBay']
          ]) {
            const bezel = new THREE.CylinderGeometry(0.16, 0.18, 0.05, 14, 1);
            bezel.translate(dx, 5.61, dz);
            addMesh(bezel, materials.bronzeGold, `ColonnadeV25_DownlightBezel_${q.id}_${bTag}_${i}`);

            const lens = new THREE.CylinderGeometry(0.11, 0.11, 0.055, 12, 1);
            lens.translate(dx, 5.595, dz);
            addMesh(lens, materials.lanternGlass, `ColonnadeV25_DownlightLens_${q.id}_${bTag}_${i}`, false, false);
          }
        }
      }
    }

    // B4. 2-Story Recessed Ground-Floor Lobby Curtain-Wall Mullions & Mezzanine Transoms (y = 0.42 .. 5.55m)
    const lobbySeg = getStationSegmentByZone('LOBBY_GLASS');
    if (lobbySeg) {
      const vLobbyMid = 0.5 * (lobbySeg.vMin + lobbySeg.vMax);
      for (let q = 0; q < 4; q++) {
        const qBase = q * 0.25;
        for (const [sA, sB, tag] of [
          [2.03, 2.97, 'FaceA'],
          [4.03, 4.97, 'FaceB']
        ]) {
          const uFaceMin = qBase + sA * duFacet;
          const uFaceMax = qBase + sB * duFacet;
          evalMaster(uFaceMin, vLobbyMid, pA, nMid);
          evalMaster(uFaceMax, vLobbyMid, pB, nMid);
          const span = pA.distanceTo(pB);
          const faceYaw = Math.atan2(nMid.x, nMid.z);
          const hNorm = new THREE.Vector3(nMid.x, 0, nMid.z).normalize();
          pMid.lerpVectors(pA, pB, 0.5);

          // Mezzanine Structural Spandrel Beam at y = 3.12m & Upper Transom at y = 4.65m
          for (const [ty, th, tTag] of [
            [3.12, 0.16, 'MezzTransom'],
            [4.65, 0.085, 'UpperTransom']
          ]) {
            const tr = new THREE.BoxGeometry(span, th, 0.11);
            SecondarySubdomains._transformGeometry(
              tr,
              pMid.x + hNorm.x * 0.05,
              ty,
              pMid.z + hNorm.z * 0.05,
              faceYaw
            );
            addMesh(tr, materials.bronzeGold, `LobbyV25_${tTag}_Q${q}_${tag}`);
          }

          // 7 Tall 2-Story Vertical Curtain-Wall Mullions per Ground Wing Face
          for (let b = 0; b <= 6; b++) {
            const uM = uFaceMin + (uFaceMax - uFaceMin) * (b / 6.0);
            evalMaster(uM, vLobbyMid, pMid, nMid);
            const mNorm = new THREE.Vector3(nMid.x, 0, nMid.z).normalize();
            const lMul = new THREE.BoxGeometry(0.095, 5.12, 0.115);
            SecondarySubdomains._transformGeometry(
              lMul,
              pMid.x + mNorm.x * 0.055,
              2.98,
              pMid.z + mNorm.z * 0.055,
              faceYaw
            );
            addMesh(lMul, materials.bronzeGold, `LobbyV25_Mullion_Q${q}_${tag}_${b}`);
          }
        }
      }
    }

    // -------------------------------------------------------------------------
    // C. MONUMENTAL 3-TIER CEREMONIAL MAIN ENTRANCE STAIRCASE, CHEEK-WALLS ('PIPI TANGGA'),
    //    & GRAND DOUBLE-LEAF ATRIUM PORTAL DOORS (+Z South & -Z North Axes)
    // -------------------------------------------------------------------------
    const stepSpecs = [
      { yTop: 0.13, zMid: 18.55, depth: 0.72, w: 8.6 },
      { yTop: 0.25, zMid: 18.12, depth: 0.72, w: 8.9 },
      { yTop: 0.365, zMid: 17.72, depth: 0.68, w: 9.2 }
    ];
    for (let s = 0; s < stepSpecs.length; s++) {
      const st = stepSpecs[s];
      const riser = new THREE.BoxGeometry(st.w, st.yTop + 0.12, st.depth);
      riser.translate(-1.2, (st.yTop - 0.12) * 0.5, st.zMid);
      addMesh(riser, materials.precastConcrete, `EntryStairV25_Riser_${s}`);

      const tread = new THREE.PlaneGeometry(st.w - 0.06, st.depth - 0.04);
      SecondarySubdomains._rotatePlaneToXZ(tread);
      tread.translate(-1.2, st.yTop + 0.004, st.zMid);
      addMesh(tread, materials.horizontalPlazaStone, `EntryStairV25_Tread_${s}`, false, true);
    }

    // Twin Sculpted Precast Stair Cheek-Walls ('Pipi Tangga') + Bronze Entry Pedestal Lanterns
    for (const sx of [-1, 1]) {
      const cheekX = -1.2 + sx * 4.88;
      const cheekWall = new THREE.BoxGeometry(0.46, 0.72, 2.25);
      cheekWall.translate(cheekX, 0.32, 17.95);
      addMesh(cheekWall, materials.precastConcrete, `EntryStairV25_CheekWall_${sx}`);

      const cheekCap = new THREE.BoxGeometry(0.56, 0.09, 2.38);
      cheekCap.translate(cheekX, 0.72, 17.95);
      addMesh(cheekCap, materials.horizontalPlazaStone, `EntryStairV25_CheekCap_${sx}`);

      const pedLampBase = new THREE.BoxGeometry(0.34, 0.28, 0.34);
      pedLampBase.translate(cheekX, 0.90, 18.78);
      addMesh(pedLampBase, materials.bronzeGold, `EntryStairV25_PedestalBase_${sx}`);

      const pedLampGlass = new THREE.CylinderGeometry(0.11, 0.13, 0.24, 8, 1);
      pedLampGlass.translate(cheekX, 1.15, 18.78);
      addMesh(pedLampGlass, materials.lanternGlass, `EntryStairV25_PedestalLantern_${sx}`);
    }

    // Grand Double-Leaf Main Atrium Portal Door Frames & Vertical Tubular Bronze Handles (+Z & -Z)
    for (const [pz, yaw, pTag] of [
      [13.14, 0.0, 'SouthMainPortal'],
      [-13.14, Math.PI, 'NorthCourtyardPortal']
    ]) {
      const doorTransom = new THREE.BoxGeometry(4.48, 0.16, 0.18);
      SecondarySubdomains._transformGeometry(doorTransom, 0.0, 2.86, pz, yaw);
      addMesh(doorTransom, materials.bronzeGold, `PortalDoorV25_Transom_${pTag}`);

      for (const lx of [-1.42, -0.05, 0.05, 1.42]) {
        const stile = new THREE.BoxGeometry(0.085, 2.52, 0.14);
        stile.translate(lx, 0, 0);
        SecondarySubdomains._transformGeometry(stile, 0.0, 1.60, pz, yaw);
        addMesh(stile, materials.bronzeGold, `PortalDoorV25_Stile_${pTag}_${lx}`);
      }
      for (const hx of [-0.18, 0.18]) {
        const pull = new THREE.CylinderGeometry(0.022, 0.022, 0.96, 10, 1);
        pull.translate(hx, 0, 0.10);
        SecondarySubdomains._transformGeometry(pull, 0.0, 1.42, pz, yaw);
        addMesh(pull, materials.bronzeGold, `PortalDoorV25_PullHandle_${pTag}_${hx}`);
      }
    }

    // -------------------------------------------------------------------------
    // D. TROPICAL PRECAST BRISE-SOLEIL SUN-SHADING FINS, CANTILEVERED WINDOW EYEBROW
    //    HOODS ('KANOPI JENDELA'), SILL DRIP APRONS, PLINTH WATER-TABLES, & PARAPET COPINGS
    //    ON ALL 6 EAST & WEST COURTYARD ANNEX BUILDINGS (Strictly |x| <= 53.05m < 54.00m!)
    // -------------------------------------------------------------------------
    const annexArticulationSpecs = [
      { tag: 'WestNorthFlat', cx: -32.25, cz: -11.2, w: 22.14, d: 14.84, yMid: 4.3, h: 1.58, isFlat: true },
      { tag: 'WestSouthFlat', cx: -32.25, cz: 9.4, w: 22.14, d: 13.24, yMid: 4.3, h: 1.58, isFlat: true },
      { tag: 'EastNorthFlat', cx: 32.25, cz: -14.2, w: 22.14, d: 14.84, yMid: 4.3, h: 1.58, isFlat: true },
      { tag: 'EastSouthFlat', cx: 32.25, cz: 6.4, w: 22.14, d: 13.24, yMid: 4.3, h: 1.58, isFlat: true },
      { tag: 'WestHallWin', cx: -43.75, cz: -1.3, w: 18.05, d: 21.95, yMid: 4.35, h: 1.62, isFlat: false },
      { tag: 'EastHallWin', cx: 43.75, cz: -4.3, w: 18.05, d: 21.95, yMid: 4.35, h: 1.62, isFlat: false }
    ];

    for (const ab of annexArticulationSpecs) {
      const yHood = ab.yMid + ab.h * 0.5 + 0.065;
      const ySill = ab.yMid - ab.h * 0.5 - 0.055;
      const hoodProj = 0.25; // On WestHall/EastHall: 43.75 + 9.025 + 0.25 = 53.025m < 54.00m!
      const sillProj = 0.14;

      // D1. Continuous Cantilevered Precast Window Sun-Hoods, Sill Drip Aprons, & Ground Water-Table Plinths
      for (const sz of [-1, 1]) {
        const zWall = ab.cz + sz * (ab.d * 0.5);
        const hoodZ = new THREE.BoxGeometry(ab.w + hoodProj * 2.0, 0.11, hoodProj);
        hoodZ.translate(ab.cx, yHood, zWall + sz * (hoodProj * 0.5));
        addMesh(hoodZ, materials.precastConcrete, `AnnexV25_SunHoodZ_${ab.tag}_${sz}`);

        const sillZ = new THREE.BoxGeometry(ab.w + sillProj * 2.0, 0.09, sillProj);
        sillZ.translate(ab.cx, ySill, zWall + sz * (sillProj * 0.5));
        addMesh(sillZ, materials.precastConcrete, `AnnexV25_SillApronZ_${ab.tag}_${sz}`);

        const plinthZ = new THREE.BoxGeometry(ab.w + 0.18, 0.44, 0.09);
        plinthZ.translate(ab.cx, 0.22, zWall + sz * 0.045);
        addMesh(plinthZ, materials.precastConcrete, `AnnexV25_PlinthZ_${ab.tag}_${sz}`);
      }

      for (const sx of [-1, 1]) {
        const xWall = ab.cx + sx * (ab.w * 0.5);
        const hoodX = new THREE.BoxGeometry(hoodProj, 0.11, ab.d + hoodProj * 2.0);
        hoodX.translate(xWall + sx * (hoodProj * 0.5), yHood, ab.cz);
        addMesh(hoodX, materials.precastConcrete, `AnnexV25_SunHoodX_${ab.tag}_${sx}`);

        const sillX = new THREE.BoxGeometry(sillProj, 0.09, ab.d + sillProj * 2.0);
        sillX.translate(xWall + sx * (sillProj * 0.5), ySill, ab.cz);
        addMesh(sillX, materials.precastConcrete, `AnnexV25_SillApronX_${ab.tag}_${sx}`);

        const plinthX = new THREE.BoxGeometry(0.09, 0.44, ab.d + 0.18);
        plinthX.translate(xWall + sx * 0.045, 0.22, ab.cz);
        addMesh(plinthX, materials.precastConcrete, `AnnexV25_PlinthX_${ab.tag}_${sx}`);
      }

      // D2. Projecting Vertical Precast Brise-Soleil Sun-Shading Fins between Window Bays
      const finDepth = 0.20;
      const finH = ab.h + 0.14;
      const nX = Math.max(3, Math.round(ab.w / 2.4));
      for (let ix = 0; ix <= nX; ix++) {
        const lx = -ab.w * 0.5 + (ab.w * ix) / nX;
        for (const sz of [-1, 1]) {
          const finZ = new THREE.BoxGeometry(0.09, finH, finDepth);
          finZ.translate(ab.cx + lx, ab.yMid, ab.cz + sz * (ab.d * 0.5 + finDepth * 0.5));
          addMesh(finZ, materials.precastConcrete, `AnnexV25_BriseSoleilZ_${ab.tag}_${ix}_${sz}`);
        }
      }
      const nZ = Math.max(3, Math.round(ab.d / 2.4));
      for (let iz = 1; iz < nZ; iz++) {
        const lz = -ab.d * 0.5 + (ab.d * iz) / nZ;
        for (const sx of [-1, 1]) {
          const finX = new THREE.BoxGeometry(finDepth, finH, 0.09);
          finX.translate(ab.cx + sx * (ab.w * 0.5 + finDepth * 0.5), ab.yMid, ab.cz + lz);
          addMesh(finX, materials.precastConcrete, `AnnexV25_BriseSoleilX_${ab.tag}_${iz}_${sx}`);
        }
      }

      // D3. Molded Parapet Coping Capstones & Architectural Rainwater Scupper Boxes on Flat-Roof Annexes
      if (ab.isFlat) {
        const yCoping = 6.33;
        const copW = 0.34;
        for (const sz of [-1, 1]) {
          const copZ = new THREE.BoxGeometry(ab.w + 0.22, 0.075, copW);
          copZ.translate(ab.cx, yCoping, ab.cz + sz * (ab.d * 0.5 - 0.06));
          addMesh(copZ, materials.precastConcrete, `AnnexV25_ParapetCopingZ_${ab.tag}_${sz}`);

          // Twin Rainwater Scupper Collector Heads ('Kepala Talang Hujan') per North/South facade
          for (const scX of [-ab.w * 0.3, ab.w * 0.3]) {
            const scupper = new THREE.BoxGeometry(0.32, 0.36, 0.24);
            scupper.translate(ab.cx + scX, 5.72, ab.cz + sz * (ab.d * 0.5 + 0.11));
            addMesh(scupper, materials.bronzeGold, `AnnexV25_ScupperHead_${ab.tag}_${sz}_${scX}`);
          }
        }
        for (const sx of [-1, 1]) {
          const copX = new THREE.BoxGeometry(copW, 0.075, ab.d + 0.22);
          copX.translate(ab.cx + sx * (ab.w * 0.5 - 0.06), yCoping, ab.cz);
          addMesh(copX, materials.precastConcrete, `AnnexV25_ParapetCopingX_${ab.tag}_${sx}`);
        }
      }
    }

    // -------------------------------------------------------------------------
    // E. UPPER CROWN CLERESTORY LOUVER MULLIONS (y = 30.85 .. 32.78m) &
    //    ARCHITECTURAL MAKARA UPLIGHT FLOODLIGHT BRACKETS
    // -------------------------------------------------------------------------
    const clerestorySeg = getStationSegmentByZone('CROWN_CLERESTORY_BAND');
    if (clerestorySeg) {
      const vMid = 0.5 * (clerestorySeg.vMin + clerestorySeg.vMax);
      const clereH = 1.82;
      const cardinalCrowns = [
        { uC: 0.0, yaw: 0.0, tag: 'South' },
        { uC: 0.25, yaw: Math.PI * 0.5, tag: 'East' },
        { uC: 0.5, yaw: Math.PI, tag: 'North' },
        { uC: 0.75, yaw: -Math.PI * 0.5, tag: 'West' }
      ];

      for (const cs of cardinalCrowns) {
        evalMaster(cs.uC, vMid, pMid, nMid);
        const hNorm = new THREE.Vector3(nMid.x, 0, nMid.z).normalize();

        // Continuous Precast Clerestory Sill Ledge at y = 30.88m
        const cSill = new THREE.BoxGeometry(15.6, 0.14, 0.22);
        SecondarySubdomains._transformGeometry(
          cSill,
          pMid.x + hNorm.x * 0.09,
          30.88,
          pMid.z + hNorm.z * 0.09,
          cs.yaw
        );
        addMesh(cSill, materials.precastConcrete, `CrownV25_ClerestorySill_${cs.tag}`);

        // 11 Vertical Bronze Clerestory Louver Mullions per Crown Face
        for (let k = -5; k <= 5; k++) {
          const mul = new THREE.BoxGeometry(0.09, clereH, 0.14);
          mul.translate(k * 1.42, 0, 0);
          SecondarySubdomains._transformGeometry(
            mul,
            pMid.x + hNorm.x * 0.07,
            31.82,
            pMid.z + hNorm.z * 0.07,
            cs.yaw
          );
          addMesh(mul, materials.bronzeGold, `CrownV25_ClerestoryMullion_${cs.tag}_${k}`);
        }

        // Twin Architectural Floodlight Brackets aimed upward at the 3D Makara Crest (y = 26.55m)
        for (const fx of [-1.15, 1.15]) {
          const arm = new THREE.BoxGeometry(0.10, 0.08, 0.48);
          arm.translate(fx, 0, 0.24);
          SecondarySubdomains._transformGeometry(
            arm,
            pMid.x + hNorm.x * 0.08,
            26.55,
            pMid.z + hNorm.z * 0.08,
            cs.yaw
          );
          addMesh(arm, materials.bronzeGold, `CrownV25_MakaraFloodArm_${cs.tag}_${fx}`);

          const lampHead = new THREE.BoxGeometry(0.28, 0.16, 0.20);
          lampHead.rotateX(0.55);
          lampHead.translate(fx, 0.10, 0.44);
          SecondarySubdomains._transformGeometry(
            lampHead,
            pMid.x + hNorm.x * 0.08,
            26.55,
            pMid.z + hNorm.z * 0.08,
            cs.yaw
          );
          addMesh(lampHead, materials.bronzeGold, `CrownV25_MakaraFloodHead_${cs.tag}_${fx}`);
        }
      }
    }

    // -------------------------------------------------------------------------
    // F. RECESSED UNDER-EAVE EXPOSED RAFTER TAILS ('EKOR USUK / KASAU TERITISAN')
    //    ON THE 4 CORNER LIMAS ROOFS & 4 HIPPED PAVILIONS
    //    (Recessed 0.42m inside outer eaves -> strictly inside 108.00 x 124.00 x 50.00m!)
    // -------------------------------------------------------------------------
    const addPerimeterRafterTails = (tag, cx, cz, halfX, halfZ, soffitY, spacing = 1.45) => {
      const inset = 0.42;
      const tailLen = 0.48;
      const rx = Math.max(1.0, halfX - inset);
      const rz = Math.max(1.0, halfZ - inset);

      const countX = Math.max(3, Math.floor((rx * 2.0) / spacing));
      for (let ix = 0; ix <= countX; ix++) {
        const lx = -rx + (2.0 * rx * ix) / countX;
        for (const sz of [-1, 1]) {
          const tailZ = new THREE.BoxGeometry(0.085, 0.105, tailLen);
          tailZ.translate(cx + lx, soffitY, cz + sz * (halfZ - inset));
          addMesh(tailZ, materials.precastConcrete, `RafterTailV25_Z_${tag}_${ix}_${sz}`);
        }
      }

      const countZ = Math.max(3, Math.floor((rz * 2.0) / spacing));
      for (let iz = 1; iz < countZ; iz++) {
        const lz = -rz + (2.0 * rz * iz) / countZ;
        for (const sx of [-1, 1]) {
          const tailX = new THREE.BoxGeometry(tailLen, 0.105, 0.085);
          tailX.translate(cx + sx * (halfX - inset), soffitY, cz + lz);
          addMesh(tailX, materials.precastConcrete, `RafterTailV25_X_${tag}_${iz}_${sx}`);
        }
      }
    };

    // 4 Corner Tower Limas Roofs (centered at ±10.30, ±10.30, eaveHalf = 8.25m, eaveBaseY = 25.15m)
    for (const q of CORNER_QUADRANTS) {
      addPerimeterRafterTails(`CornerLimas_${q.id}`, q.sx * 10.3, q.sz * 10.3, 8.25, 8.25, 25.09, 1.48);
    }

    // 4 Hipped Pavilions (with 0.42m inset + 0.24m half-tail = 0.18m inside exact outer eave boundaries!)
    addPerimeterRafterTails('Gatehouse', -2.5, 65.5, 6.5, 6.5, 3.70, 1.35);
    addPerimeterRafterTails('WestHall', -43.75, -1.3, 10.25, 12.2, 5.80, 1.55);
    addPerimeterRafterTails('EastHall', 43.75, -4.3, 10.25, 12.2, 5.80, 1.55);
    addPerimeterRafterTails('NorthPav', 5.0, -43.5, 8.5, 8.5, 3.70, 1.45);
  }

  /**
   * Builds a sculpted square architectural column with 3-tier plinth base, fluted pilaster reveals,
   * and 2-tier stepped Nusantara corbel capital on LOD0 (or 1 clean box on LOD1).
   */
  static _addSculptedSquarePillar(
    addMesh,
    materials,
    cx,
    yBot,
    cz,
    wX,
    totalH,
    wZ,
    name,
    isUltra,
    rotY = 0.0
  ) {
    if (!isUltra) {
      const box = new THREE.BoxGeometry(wX, totalH, wZ);
      SecondarySubdomains._transformGeometry(box, cx, yBot + totalH * 0.5, cz, rotY);
      addMesh(box, materials.precastConcrete, name);
      return;
    }

    // 1. Stepped Base Plinth (lower shoe + upper torus molding)
    const baseH1 = 0.22;
    const baseH2 = 0.14;
    const capH1 = 0.14;
    const capH2 = 0.18;
    const shaftH = Math.max(0.5, totalH - baseH1 - baseH2 - capH1 - capH2);

    const base1 = new THREE.BoxGeometry(wX + 0.18, baseH1, wZ + 0.18);
    SecondarySubdomains._transformGeometry(base1, cx, yBot + baseH1 * 0.5, cz, rotY);
    addMesh(base1, materials.precastConcrete, `${name}_Plinth1`);

    const base2 = new THREE.BoxGeometry(wX + 0.09, baseH2, wZ + 0.09);
    SecondarySubdomains._transformGeometry(base2, cx, yBot + baseH1 + baseH2 * 0.5, cz, rotY);
    addMesh(base2, materials.precastConcrete, `${name}_Plinth2`);

    // 2. Main Shaft + Orthogonal Pilaster Fluting Strips
    const yShaftMid = yBot + baseH1 + baseH2 + shaftH * 0.5;
    const shaft = new THREE.BoxGeometry(wX * 0.92, shaftH, wZ * 0.92);
    SecondarySubdomains._transformGeometry(shaft, cx, yShaftMid, cz, rotY);
    addMesh(shaft, materials.precastConcrete, `${name}_Shaft`);

    const fluteX = new THREE.BoxGeometry(wX + 0.02, shaftH - 0.16, wZ * 0.52);
    SecondarySubdomains._transformGeometry(fluteX, cx, yShaftMid, cz, rotY);
    addMesh(fluteX, materials.precastConcrete, `${name}_FluteX`);

    const fluteZ = new THREE.BoxGeometry(wX * 0.52, shaftH - 0.16, wZ + 0.02);
    SecondarySubdomains._transformGeometry(fluteZ, cx, yShaftMid, cz, rotY);
    addMesh(fluteZ, materials.precastConcrete, `${name}_FluteZ`);

    // 3. Stepped Nusantara Corbel Capital (necking + abacus)
    const yCapStart = yBot + baseH1 + baseH2 + shaftH;
    const cap1 = new THREE.BoxGeometry(wX + 0.10, capH1, wZ + 0.10);
    SecondarySubdomains._transformGeometry(cap1, cx, yCapStart + capH1 * 0.5, cz, rotY);
    addMesh(cap1, materials.precastConcrete, `${name}_Cap1`);

    const cap2 = new THREE.BoxGeometry(wX + 0.22, capH2, wZ + 0.22);
    SecondarySubdomains._transformGeometry(cap2, cx, yCapStart + capH1 + capH2 * 0.5, cz, rotY);
    addMesh(cap2, materials.precastConcrete, `${name}_Cap2`);
  }

  /**
   * Builds a 14-flute lofted round Nusantara column geometry with molded torus base & stepped capital.
   */
  static _createFlutedRoundColumnGeometry(rTop, rBot, totalH) {
    const radialSegs = 28;
    const profile = [
      { y: 0.0, r: rBot + 0.14, flute: 0.0 },
      { y: 0.18, r: rBot + 0.14, flute: 0.0 },
      { y: 0.22, r: rBot + 0.08, flute: 0.0 },
      { y: 0.34, r: rBot + 0.05, flute: 0.0 },
      { y: 0.38, r: rBot, flute: 0.014 },
      { y: totalH * 0.5, r: 0.5 * (rBot + rTop) + 0.008, flute: 0.014 }, // Subtle classical entasis!
      { y: totalH - 0.34, r: rTop, flute: 0.014 },
      { y: totalH - 0.30, r: rTop + 0.05, flute: 0.0 },
      { y: totalH - 0.16, r: rTop + 0.11, flute: 0.0 },
      { y: totalH, r: rTop + 0.16, flute: 0.0 }
    ];

    const positions = [];
    const uvs = [];
    const indices = [];

    for (let ip = 0; ip < profile.length; ip++) {
      const st = profile[ip];
      const v = st.y / totalH;
      for (let ir = 0; ir <= radialSegs; ir++) {
        const u = ir / radialSegs;
        const th = u * Math.PI * 2.0;
        const r = st.r - st.flute * (0.5 + 0.5 * Math.cos(14.0 * th));
        positions.push(Math.cos(th) * r, st.y, Math.sin(th) * r);
        uvs.push(u * 2.0, v * 3.0);
      }
    }

    const stride = radialSegs + 1;
    for (let ip = 0; ip < profile.length - 1; ip++) {
      for (let ir = 0; ir < radialSegs; ir++) {
        const a = ip * stride + ir;
        const b = a + 1;
        const c = (ip + 1) * stride + ir + 1;
        const d = (ip + 1) * stride + ir;
        indices.push(a, d, c, a, c, b);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }

  /**
   * Builds a planter trough with molded concrete coping rim & recessed dark earth soil bed.
   */
  static _addPlanterTrough(addMesh, materials, cx, cz, wX, wZ, name, isUltra) {
    const baseBox = new THREE.BoxGeometry(wX, 0.56, wZ);
    baseBox.translate(cx, 0.64, cz);
    addMesh(baseBox, materials.precastConcrete, name);

    if (!isUltra) return;

    const rimT = 0.12;
    for (const sz of [-1, 1]) {
      const rimZ = new THREE.BoxGeometry(wX + 0.08, 0.08, rimT);
      rimZ.translate(cx, 0.95, cz + sz * (wZ * 0.5 - rimT * 0.5 + 0.02));
      addMesh(rimZ, materials.precastConcrete, `${name}_RimZ_${sz}`);
    }
    for (const sx of [-1, 1]) {
      const rimX = new THREE.BoxGeometry(rimT, 0.08, Math.max(0.1, wZ - rimT * 2));
      rimX.translate(cx + sx * (wX * 0.5 - rimT * 0.5 + 0.02), 0.95, cz);
      addMesh(rimX, materials.precastConcrete, `${name}_RimX_${sx}`);
    }

    const soil = new THREE.PlaneGeometry(Math.max(0.2, wX - 0.22), Math.max(0.2, wZ - 0.22));
    SecondarySubdomains._rotatePlaneToXZ(soil);
    soil.translate(cx, 0.93, cz);
    addMesh(soil, materials.foundationConcrete, `${name}_SoilBed`, false, true);
  }

  /**
   * Adds exposed 3D architectural ceiling rafters ('kasau'), ridge beam ('nok'), and purlins ('gording')
   * underneath a pitched gable gallery roof.
   */
  static _addGableCeilingRafters(
    addMesh,
    materials,
    { tag, cx, cz, length, halfWidth, eaveY, ridgeY, rafterPairs = 12, alongAxis = 'Z', rotY = 0.0 }
  ) {
    const baseRotY = alongAxis === 'X' ? Math.PI * 0.5 : 0.0;
    const totalRotY = baseRotY + rotY;
    const dy = ridgeY - eaveY;
    const slopeLen = Math.hypot(halfWidth, dy);
    const pitch = Math.atan2(dy, halfWidth);

    // Longitudinal Ridge Beam ('Balok Nok') + Twin Mid-Slope Purlins ('Gording')
    const ridgeBeam = new THREE.BoxGeometry(0.16, 0.22, length);
    ridgeBeam.translate(0, ridgeY - 0.14, 0);
    SecondarySubdomains._transformGeometry(ridgeBeam, cx, 0, cz, totalRotY);
    addMesh(ridgeBeam, materials.precastConcrete, `Ceiling_${tag}_RidgeBeam`);

    for (const sgn of [-1, 1]) {
      const purlin = new THREE.BoxGeometry(0.12, 0.14, length);
      purlin.rotateZ(-sgn * pitch);
      purlin.translate(sgn * halfWidth * 0.5, eaveY + dy * 0.5 - 0.08, 0);
      SecondarySubdomains._transformGeometry(purlin, cx, 0, cz, totalRotY);
      addMesh(purlin, materials.precastConcrete, `Ceiling_${tag}_Purlin_${sgn}`);
    }

    // Transverse Rafter Pairs ('Kasau')
    for (let i = 0; i < rafterPairs; i++) {
      const lz = -length * 0.5 + ((i + 0.5) / rafterPairs) * length;
      for (const sgn of [-1, 1]) {
        const rafter = new THREE.BoxGeometry(slopeLen, 0.1, 0.09);
        rafter.rotateZ(-sgn * pitch);
        rafter.translate(sgn * halfWidth * 0.5, eaveY + dy * 0.5 - 0.06, lz);
        SecondarySubdomains._transformGeometry(rafter, cx, 0, cz, totalRotY);
        addMesh(rafter, materials.precastConcrete, `Ceiling_${tag}_Rafter_${i}_${sgn}`);
      }
    }
  }

  /**
   * Constructs a unified local-space (+Z outward normal) 3D Bas-Relief Sculpture BufferGeometry
   * of the Universitas Indonesia Makara Crest (Outer Medallion Ring, Kalpataru Tree of Knowledge,
   * 7 Curving Branches, 9 Banyan Leaf Domes, & 5 Water-of-Knowledge Base Scrolls).
   */
  static _createMakara3DReliefGeometry() {
    const positions = [];
    const normals = [];
    const uvs = [];
    const indices = [];

    const appendGeo = (geo) => {
      if (!geo.attributes.normal) geo.computeVertexNormals();
      const p = geo.attributes.position.array;
      const n = geo.attributes.normal.array;
      const u = geo.attributes.uv ? geo.attributes.uv.array : null;
      const base = positions.length / 3;
      const vCount = geo.attributes.position.count;

      for (let i = 0; i < vCount; i++) {
        positions.push(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
        normals.push(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]);
        uvs.push(u ? u[i * 2] : 0.5, u ? u[i * 2 + 1] : 0.5);
      }

      if (geo.index) {
        const idx = geo.index.array;
        for (let i = 0; i < idx.length; i++) {
          indices.push(base + idx[i]);
        }
      } else {
        for (let i = 0; i < vCount; i++) {
          indices.push(base + i);
        }
      }
    };

    // 1. 3D Circular Outer Medallion Torus Ring (R = 1.48m, tube = 0.075m)
    const ringSegs = 48;
    const tubeSegs = 8;
    const ringBase = positions.length / 3;
    for (let i = 0; i <= ringSegs; i++) {
      const th = (i / ringSegs) * Math.PI * 2.0;
      const cTh = Math.cos(th);
      const sTh = Math.sin(th);
      for (let j = 0; j <= tubeSegs; j++) {
        const ph = -Math.PI * 0.5 + (j / tubeSegs) * Math.PI; // Front-facing half-torus relief!
        const cPh = Math.cos(ph);
        const sPh = Math.sin(ph);
        const rad = 1.48 + 0.075 * sPh;
        const px = cTh * rad;
        const py = sTh * rad;
        const pz = 0.075 * cPh;
        positions.push(px, py, pz);
        normals.push(cTh * sPh, sTh * sPh, cPh);
        uvs.push(i / ringSegs, j / tubeSegs);
      }
    }
    const rStride = tubeSegs + 1;
    for (let i = 0; i < ringSegs; i++) {
      for (let j = 0; j < tubeSegs; j++) {
        const a = ringBase + i * rStride + j;
        const b = ringBase + (i + 1) * rStride + j;
        const c = ringBase + (i + 1) * rStride + (j + 1);
        const d = ringBase + i * rStride + (j + 1);
        indices.push(a, b, c, a, c, d);
      }
    }

    // 2. Sculpted Central Kalpataru Tree Trunk & Flared Roots
    const trunk = new THREE.CylinderGeometry(0.11, 0.22, 1.32, 14, 4);
    trunk.scale(1.0, 1.0, 0.55);
    trunk.translate(0.0, -0.22, 0.04);
    appendGeo(trunk);

    // 3. 7 Curving 3D Relief Branches Radiating into the Crown
    const branchSpecs = [
      { x: -0.42, y: -0.05, len: 0.78, rotZ: 0.85 },
      { x: 0.42, y: -0.05, len: 0.78, rotZ: -0.85 },
      { x: -0.48, y: 0.34, len: 0.82, rotZ: 0.62 },
      { x: 0.48, y: 0.34, len: 0.82, rotZ: -0.62 },
      { x: -0.32, y: 0.68, len: 0.62, rotZ: 0.38 },
      { x: 0.32, y: 0.68, len: 0.62, rotZ: -0.38 },
      { x: 0.0, y: 0.74, len: 0.58, rotZ: 0.0 }
    ];
    for (const br of branchSpecs) {
      const bGeo = new THREE.CylinderGeometry(0.042, 0.075, br.len, 10, 1);
      bGeo.scale(1.0, 1.0, 0.5);
      bGeo.rotateZ(br.rotZ);
      bGeo.translate(br.x, br.y, 0.035);
      appendGeo(bGeo);
    }

    // 4. 9 Sculpted 3D Banyan Leaf-Cluster Relief Domes (Makara Kalpataru Canopy)
    const foliageNodes = [
      { x: 0.0, y: 1.06, r: 0.34 },
      { x: -0.44, y: 0.94, r: 0.28 },
      { x: 0.44, y: 0.94, r: 0.28 },
      { x: -0.78, y: 0.62, r: 0.27 },
      { x: 0.78, y: 0.62, r: 0.27 },
      { x: -0.86, y: 0.16, r: 0.25 },
      { x: 0.86, y: 0.16, r: 0.25 },
      { x: -0.38, y: 0.48, r: 0.26 },
      { x: 0.38, y: 0.48, r: 0.26 }
    ];
    for (const fn of foliageNodes) {
      const dome = new THREE.SphereGeometry(fn.r, 14, 8);
      dome.scale(1.0, 0.88, 0.34);
      dome.translate(fn.x, fn.y, 0.03);
      appendGeo(dome);
    }

    // 5. 5 Sculpted 3D Water-of-Knowledge Base Scroll Waves beneath the Kalpataru Roots
    for (let w = 0; w < 5; w++) {
      const wy = -0.96 - w * 0.095;
      const ww = 1.55 - w * 0.22;
      const scroll = new THREE.BoxGeometry(ww, 0.055, 0.085);
      scroll.translate(0.0, wy, 0.04);
      appendGeo(scroll);
    }

    const out = new THREE.BufferGeometry();
    out.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    out.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));
    out.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    out.setIndex(indices);
    return out;
  }

  static _transformGeometry(geo, tx, ty, tz, rotY = 0) {
    if (rotY !== 0) {
      if (typeof geo.rotateY === 'function') {
        geo.rotateY(rotY);
      } else {
        const m = new THREE.Matrix4().makeRotationY(rotY);
        geo.applyMatrix4(m);
      }
    }
    geo.translate(tx, ty, tz);
  }

  static _rotatePlaneToXZ(geo) {
    const pos = geo.attributes.position;
    const norm = geo.attributes.normal;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      pos.setXYZ(i, x, 0, -y);
      norm.setXYZ(i, 0, 1, 0);
    }
  }
}
