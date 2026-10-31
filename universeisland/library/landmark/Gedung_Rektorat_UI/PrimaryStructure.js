import * as THREE from 'three';
import { MasterSurfaceBuilder } from '../../../core/MasterSurfaceBuilder.js';
import {
  MASTER_PROFILE_STATIONS,
  CORNER_QUADRANTS
} from './AssetMetaAndEnvelope.js';
import { MasterSurfaceKernel } from './MasterSurfaceKernel.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.3 — 10x Ultra-High Level of Detail Edition)
 * ./library/landmark/Gedung_Rektorat_UI/PrimaryStructure.js
 *
 * TAHAP 4 (Gate 3): Pemotongan Subdomain SSoST (0.0000m Gap) & Konstruksi Kompleks Rektorat UI (10x Ultra-LOD):
 * 1. Menara Utama Pagoda 6-Teritisan Berprofil Genteng Keramik Jawa 3D + Ceruk Atrium Kaca Berlian Tengah +
 *    4 Atap Limas Sudut Berkontur Genteng 3D + Leher Putih Makara UI + Atap Puncak Limas Tajug Curam (50.00m).
 * 2. Pelataran Keramik Abu-Putih Motif Berlian (◇◇◇) + Skirt Fondasi -18.00m.
 * 3. Sumbu +Z Rotunda: Selasar Agung Beratap Pelana Genteng 3D & Bubungan Silinder -> Kloister Panji -> Gerbang Depan (+72.00m).
 * 4. Sayap Barat (-54.00m) & Sayap Timur Tepi Danau Kenanga (+54.00m): Blok Dek Beton Datar Berkop Parapet + Aula Limas Merah.
 * 5. Sumbu 45° Tenggara: Selasar Diagonal & Pendopo Segidelapan Ber-Skylight Kaca Puncak.
 * 6. Sumbu -Z Utara: Pendopo Limas Utara (-52.00m).
 */
export class PrimaryStructure {
  static build(materials, detailScale = 1.0) {
    const group = new THREE.Group();
    group.name = 'Tahap4_PrimaryStructure';

    const isUltra = detailScale >= 0.9;
    const stepsPerFacet = isUltra ? 14 : detailScale >= 0.45 ? 2 : 1;
    const fullUSteps = 28 * stepsPerFacet;
    const subdomainBoundaries = [];

    const addMesh = (geo, mat, name, castShadow = true, receiveShadow = true) => {
      const m = new THREE.Mesh(geo, mat);
      m.name = name;
      m.castShadow = castShadow;
      m.receiveShadow = receiveShadow;
      group.add(m);
      return m;
    };

    const evalMaster = MasterSurfaceKernel.evaluateMasterSurface;
    const stations = MASTER_PROFILE_STATIONS;
    const duFacet = 0.25 / 7.0;

    // =========================================================================
    // 1. MAIN TOWER SSoST SUBDOMAIN SLICING (0.0000m SEAM GAP — 10x ULTRA-LOD)
    // =========================================================================
    for (let i = 0; i < stations.length - 1; i++) {
      const s0 = stations[i];
      const s1 = stations[i + 1];
      const vMin = s0.v;
      const vMax = s1.v;
      const zone = s0.zone;
      subdomainBoundaries.push(vMax);

      // A. Sub-Terrain Skirt & Base Plinth
      if (zone === 'BOTTOM_CAP' || zone === 'FOUNDATION_SKIRT') {
        const geo = MasterSurfaceBuilder.sliceSubdomain(
          evalMaster,
          0.0,
          1.0,
          vMin,
          vMax,
          isUltra ? 112 : fullUSteps,
          zone === 'FOUNDATION_SKIRT' && isUltra ? 3 : 1,
          null,
          { uvMode: 'facadeNormalized', uvScaleU: 6.0, uvScaleV: 2.0, yMinNorm: -18.0, totalHeightNorm: 18.0 }
        );
        const fCols = new Float32Array(geo.attributes.position.count * 3);
        for (let ci = 0; ci < fCols.length; ci += 3) {
          fCols[ci] = 0.52;
          fCols[ci + 1] = 0.49;
          fCols[ci + 2] = 0.45;
        }
        geo.setAttribute('color', new THREE.BufferAttribute(fCols, 3));
        addMesh(geo, materials.foundationConcrete, `Primary_${zone}`, false, true);
        continue;
      }

      if (zone === 'TOWER_BASE_STEP1' || zone === 'TOWER_BASE_DECK') {
        const geo = MasterSurfaceBuilder.sliceSubdomain(
          evalMaster,
          0.0,
          1.0,
          vMin,
          vMax,
          fullUSteps,
          isUltra ? 2 : 1,
          null,
          { uvMode: 'worldHorizontal', uvScaleU: 0.14, uvScaleV: 0.14 }
        );
        addMesh(
          geo,
          zone === 'TOWER_BASE_DECK' ? materials.horizontalPlazaStone : materials.precastConcrete,
          `Primary_${zone}`
        );
        continue;
      }

      // B. Cruciform 4-Wing Zone (y = 0.42m .. 25.15m):
      // Keep seg 0 & seg 6 in every quadrant as UNBROKEN Dark X-Diamond Central Atrium Glass!
      const isCruciformWingZone = s0.towerActive > 0.5 && s1.towerActive > 0.5;
      if (isCruciformWingZone) {
        const isEaveSlope = zone.endsWith('_SLOPE');
        const isGlassBand = zone.endsWith('_GLASS');
        const vStepsEave = isEaveSlope ? (isUltra ? 14 : 2) : 1;
        const vStepsWall = isUltra ? (isGlassBand ? 3 : 2) : 1;

        for (let q = 0; q < 4; q++) {
          const qBase = q * 0.25;

          // Central Atrium Slot Halves (seg 0..1 and seg 6..7) -> Continuous Dark X-Diamond Atrium Glass!
          for (const [sA, sB, tag] of [
            [0, 1, 'AtriumSlotA'],
            [6, 7, 'AtriumSlotB']
          ]) {
            const atriumGeo = MasterSurfaceBuilder.sliceSubdomain(
              evalMaster,
              qBase + sA * duFacet,
              qBase + sB * duFacet,
              vMin,
              vMax,
              stepsPerFacet,
              isUltra ? 2 : 1,
              null,
              { uvMode: 'facadeNormalized', uvScaleU: 2.0, uvScaleV: 1.0, yMinNorm: 0.42, totalHeightNorm: 24.73 }
            );
            addMesh(atriumGeo, materials.atriumGlass, `Primary_${zone}_Q${q}_${tag}`);
          }

          // Corner Wing Perimeter (seg 1..6)
          if (isEaveSlope) {
            const uStart = qBase + 1 * duFacet;
            const uEnd = qBase + 6 * duFacet;
            const slopeGeo = PrimaryStructure._sliceCorrugatedTileSubdomain(
              evalMaster,
              uStart,
              uEnd,
              vMin,
              vMax,
              5 * stepsPerFacet,
              vStepsEave,
              isUltra
                ? {
                    wavesU: 168,
                    coursesV: 7,
                    amp: 0.021,
                    taperU: true
                  }
                : null,
              { uvMode: 'parametric', uvScaleU: 3.0, uvScaleV: 14.0 }
            );
            addMesh(slopeGeo, materials.terracottaTile, `Primary_${zone}_Q${q}_WingEave`);
          } else if (isGlassBand && detailScale >= 0.45) {
            // Crisp white return flanks & corner pier framing the 6-bay dark ribbon window faces (seg 2..3 & 4..5)
            const wingSubFacets = [
              { a: 1, b: 2, mat: materials.precastConcrete, tag: 'ReturnA', uvU: 1.0 },
              { a: 2, b: 3, mat: materials.ribbonGlass, tag: 'WinFaceA', uvU: 7.0 },
              { a: 3, b: 4, mat: materials.precastConcrete, tag: 'CornerPier', uvU: 1.0 },
              { a: 4, b: 5, mat: materials.ribbonGlass, tag: 'WinFaceB', uvU: 7.0 },
              { a: 5, b: 6, mat: materials.precastConcrete, tag: 'ReturnB', uvU: 1.0 }
            ];
            for (const wf of wingSubFacets) {
              const wGeo = MasterSurfaceBuilder.sliceSubdomain(
                evalMaster,
                qBase + wf.a * duFacet,
                qBase + wf.b * duFacet,
                vMin,
                vMax,
                stepsPerFacet,
                vStepsWall,
                null,
                { uvMode: 'parametric', uvScaleU: wf.uvU, uvScaleV: 22.0 }
              );
              addMesh(wGeo, wf.mat, `Primary_${zone}_Q${q}_${wf.tag}`);
            }
          } else {
            const wingMat = isGlassBand ? materials.ribbonGlass : materials.precastConcrete;
            const cGeo = MasterSurfaceBuilder.sliceSubdomain(
              evalMaster,
              qBase + 1 * duFacet,
              qBase + 6 * duFacet,
              vMin,
              vMax,
              5 * stepsPerFacet,
              vStepsWall,
              null,
              { uvMode: 'facadeNormalized', uvScaleU: 4.0, uvScaleV: 3.0, yMinNorm: 0.0, totalHeightNorm: 26.0 }
            );
            addMesh(cGeo, wingMat, `Primary_${zone}_Q${q}_WingConcrete`);
          }
        }
        continue;
      }

      // C. Upper Crown Clerestory Band (y = 30.85m .. 32.78m):
      // Central seg 0..1 & 6..7 stay Crisp Off-White Makara UI Panel, while seg 1..6 are Dark Clerestory Windows!
      if (zone === 'CROWN_CLERESTORY_BAND') {
        for (let q = 0; q < 4; q++) {
          const qBase = q * 0.25;
          for (const [sA, sB, mat, tag] of [
            [0, 1, materials.precastConcrete, 'MakaraWallA'],
            [1, 6, materials.lanternGlass, 'ClerestoryGlass'],
            [6, 7, materials.precastConcrete, 'MakaraWallB']
          ]) {
            const clGeo = MasterSurfaceBuilder.sliceSubdomain(
              evalMaster,
              qBase + sA * duFacet,
              qBase + sB * duFacet,
              vMin,
              vMax,
              (sB - sA) * stepsPerFacet,
              isUltra ? 3 : 1,
              null,
              { uvMode: 'parametric', uvScaleU: 4.0, uvScaleV: 12.0 }
            );
            addMesh(clGeo, mat, `Primary_${zone}_Q${q}_${tag}`);
          }
        }
        continue;
      }

      // D. Monumental Steep 4-Sided Pyramidal Terracotta Tajug Spire (y = 33.10m .. 46.95m)
      //    10x Ultra-LOD: 392 x 46 grid with true 3D Javanese clay tile ribs & 28 overlapping tile courses!
      if (zone === 'TAJUG_MAIN_SLOPE') {
        const tajugGeo = PrimaryStructure._sliceCorrugatedTileSubdomain(
          evalMaster,
          0.0,
          1.0,
          vMin,
          vMax,
          fullUSteps,
          isUltra ? 46 : 4,
          isUltra
            ? {
                wavesU: 112,
                coursesV: 28,
                amp: 0.024,
                taperU: false
              }
            : null,
          { uvMode: 'parametric', uvScaleU: 7.0, uvScaleV: 24.0 }
        );
        addMesh(tajugGeo, materials.terracottaTile, `Primary_${zone}`);
        continue;
      }

      // E. Crisp White Pyramidal Apex Cap (Mustaka Putih) & Fluted Lightning Finial
      if (zone === 'MUSTAKA_WHITE_CAP') {
        const capGeo = MasterSurfaceBuilder.sliceSubdomain(
          evalMaster,
          0.0,
          1.0,
          vMin,
          vMax,
          isUltra ? 112 : 28,
          isUltra ? 6 : 2
        );
        addMesh(capGeo, materials.precastConcrete, `Primary_${zone}`);
        continue;
      }

      if (zone === 'LIGHTNING_SPIRE') {
        const spireGeo = MasterSurfaceBuilder.sliceSubdomain(
          evalMaster,
          0.0,
          1.0,
          vMin,
          vMax,
          isUltra ? 56 : 16,
          isUltra ? 10 : 2
        );
        addMesh(spireGeo, materials.bronzeGold, `Primary_${zone}`);
        continue;
      }

      // F. Default Off-White Architectural Concrete (Crown Drum Walls, Soffits, & Fascias)
      const defGeo = MasterSurfaceBuilder.sliceSubdomain(
        evalMaster,
        0.0,
        1.0,
        vMin,
        vMax,
        fullUSteps,
        isUltra ? 2 : 1,
        null,
        { uvMode: 'facadeNormalized', uvScaleU: 5.0, uvScaleV: 3.0, yMinNorm: 0.0, totalHeightNorm: 48.0 }
      );
      addMesh(defGeo, materials.precastConcrete, `Primary_${zone}`);
    }

    // =========================================================================
    // 2. 4 BIG CORNER PYRAMIDAL/LIMAS ROOFS (TIER 7 — y = 25.15m .. 30.85m)
    // =========================================================================
    const cornerSteps = isUltra ? 112 : 16;
    for (const q of CORNER_QUADRANTS) {
      const evalCorner = MasterSurfaceKernel.createCornerTowerCrownEvaluator(q);
      // Underside white soffit + fascia (v: 0.0 -> 0.50)
      addMesh(
        MasterSurfaceBuilder.sliceSubdomain(
          evalCorner,
          0.0,
          1.0,
          0.0,
          0.5,
          cornerSteps,
          isUltra ? 3 : 2
        ),
        materials.precastConcrete,
        `CornerLimas_SoffitFascia_${q.id}`
      );
      // Steep terracotta hipped roof slope (v: 0.50 -> 0.75) with 3D Javanese clay tile courses
      addMesh(
        PrimaryStructure._sliceCorrugatedTileSubdomain(
          evalCorner,
          0.0,
          1.0,
          0.5,
          0.75,
          cornerSteps,
          isUltra ? 16 : 2,
          isUltra
            ? {
                wavesU: 64,
                coursesV: 12,
                amp: 0.02,
                taperU: false
              }
            : null,
          { uvMode: 'parametric', uvScaleU: 3.5, uvScaleV: 6.0 }
        ),
        materials.terracottaTile,
        `CornerLimas_TileSlope_${q.id}`
      );
      // Crisp off-white flat ridge cap at y = 30.85m (v: 0.75 -> 1.00)
      addMesh(
        MasterSurfaceBuilder.sliceSubdomain(
          evalCorner,
          0.0,
          1.0,
          0.75,
          1.0,
          cornerSteps,
          isUltra ? 2 : 1
        ),
        materials.precastConcrete,
        `CornerLimas_WhiteCap_${q.id}`
      );
    }

    // =========================================================================
    // 3. SIGNATURE UI DIAMOND-INLAID COURTYARD PLAZA PLINTH (◇◇◇ PAVING + -18m SKIRT)
    // =========================================================================
    PrimaryStructure._addHorizontalDiamondPlaza(
      addMesh,
      materials,
      -36.0,
      36.0,
      -35.0,
      39.0,
      0.12,
      'MainDiamondCourtyard',
      true,
      detailScale
    );
    // Flag Cloister Inner Ceremonial Paved Court
    PrimaryStructure._addHorizontalDiamondPlaza(
      addMesh,
      materials,
      -16.0,
      11.0,
      43.2,
      59.0,
      0.14,
      'CloisterFlagCourt',
      false,
      detailScale
    );

    // =========================================================================
    // 4. +Z FRONT CEREMONIAL SEQUENCE: SELASAR AGUNG -> FLAG CLOISTER -> GATEHOUSE (+72.00m)
    // =========================================================================
    // 4A. High-Gabled Ceremonial Breezeway ('Selasar Agung', x = -2.50m, z = 15.6 .. 39.4m)
    PrimaryStructure._addPitchedGableRoof(addMesh, materials, {
      name: 'SelasarAgung_Roof',
      cx: -2.5,
      cz: 27.5,
      length: 24.0,
      halfWidth: 4.2,
      eaveY: 4.85,
      ridgeY: 9.35,
      alongAxis: 'Z',
      detailScale
    });

    // Small gable skylight hood at bottom of Front Central Atrium Slot (seen in Mavic Photo #1!)
    PrimaryStructure._addPitchedGableRoof(addMesh, materials, {
      name: 'FrontAtrium_EntryGable',
      cx: 0.0,
      cz: 14.8,
      length: 3.6,
      halfWidth: 2.35,
      eaveY: 6.1,
      ridgeY: 7.55,
      alongAxis: 'Z',
      detailScale
    });

    // 4B. 4-Sided Rectangular Terracotta-Roofed Cloister ('Pelataran Panji Fakultas UI')
    PrimaryStructure._addPitchedGableRoof(addMesh, materials, {
      name: 'Cloister_NorthGalleryRoof',
      cx: -2.5,
      cz: 41.4,
      length: 36.0,
      halfWidth: 2.4,
      eaveY: 3.55,
      ridgeY: 5.55,
      alongAxis: 'X',
      detailScale
    });
    PrimaryStructure._addPitchedGableRoof(addMesh, materials, {
      name: 'Cloister_SouthGalleryRoof',
      cx: -2.5,
      cz: 61.1,
      length: 36.0,
      halfWidth: 2.4,
      eaveY: 3.55,
      ridgeY: 5.55,
      alongAxis: 'X',
      detailScale
    });
    PrimaryStructure._addPitchedGableRoof(addMesh, materials, {
      name: 'Cloister_WestGalleryRoof',
      cx: -18.1,
      cz: 51.25,
      length: 24.5,
      halfWidth: 2.4,
      eaveY: 3.55,
      ridgeY: 5.55,
      alongAxis: 'Z',
      detailScale
    });
    PrimaryStructure._addPitchedGableRoof(addMesh, materials, {
      name: 'Cloister_EastGalleryRoof',
      cx: 13.1,
      cz: 51.25,
      length: 24.5,
      halfWidth: 2.4,
      eaveY: 3.55,
      ridgeY: 5.55,
      alongAxis: 'Z',
      detailScale
    });

    // Low off-white outer perimeter walls of the Cloister with molded top coping (Mavic Photo #1 foreground!)
    const southWallL = new THREE.BoxGeometry(11.5, 2.55, 0.45);
    southWallL.translate(-14.5, 1.35, 62.6);
    addMesh(southWallL, materials.precastConcrete, 'Cloister_FrontWallL');
    const southWallR = new THREE.BoxGeometry(11.5, 2.55, 0.45);
    southWallR.translate(9.5, 1.35, 62.6);
    addMesh(southWallR, materials.precastConcrete, 'Cloister_FrontWallR');
    if (isUltra) {
      for (const [wx, tag] of [
        [-14.5, 'L'],
        [9.5, 'R']
      ]) {
        const cap = new THREE.BoxGeometry(11.66, 0.14, 0.58);
        cap.translate(wx, 2.66, 62.6);
        addMesh(cap, materials.precastConcrete, `Cloister_FrontWallCoping_${tag}`);
      }
    }

    // 4C. Front Rotunda Hipped Gatehouse Pavilion ('Gerbang Pendopo Depan', maxZ = +72.00m EXACT!)
    PrimaryStructure._addHippedPavilion(addMesh, materials, {
      name: 'FrontRotunda_Gatehouse',
      cx: -2.5,
      cz: 65.5,
      halfX: 6.5,
      halfZ: 6.5, // 65.5 + 6.5 = +72.00m EXACT!
      ridgeHalfX: 1.8,
      ridgeHalfZ: 0.25,
      eaveY: 3.75,
      ridgeY: 7.45,
      stepsU: cornerSteps,
      detailScale
    });

    // =========================================================================
    // 5. TWIN WEST (-54.00m) & EAST LAKESIDE (+54.00m) COURTYARD ANNEX WINGS
    // =========================================================================
    // 5A. West Courtyard Annex Wing (minX = -54.00m EXACT!)
    PrimaryStructure._addFlatRoofCourtyardBlock(addMesh, materials, {
      name: 'WestAnnex_NorthFlatBlock',
      minX: -43.5,
      maxX: -21.0,
      minZ: -18.8,
      maxZ: -3.6,
      roofY: 5.75,
      parapetTopY: 6.32,
      detailScale
    });
    PrimaryStructure._addFlatRoofCourtyardBlock(addMesh, materials, {
      name: 'WestAnnex_SouthFlatBlock',
      minX: -43.5,
      maxX: -21.0,
      minZ: 2.6,
      maxZ: 16.2,
      roofY: 5.75,
      parapetTopY: 6.32,
      detailScale
    });
    PrimaryStructure._addPitchedGableRoof(addMesh, materials, {
      name: 'WestAnnex_TowerBridge',
      cx: -18.4,
      cz: -0.5,
      length: 6.4,
      halfWidth: 2.6,
      eaveY: 4.4,
      ridgeY: 6.1,
      alongAxis: 'X',
      detailScale
    });
    PrimaryStructure._addHippedPavilion(addMesh, materials, {
      name: 'WestAnnex_PyramidalHall',
      cx: -43.75,
      cz: -1.3,
      halfX: 10.25, // -43.75 - 10.25 = -54.00m EXACT!
      halfZ: 12.2,
      ridgeHalfX: 0.35,
      ridgeHalfZ: 2.4,
      eaveY: 5.85,
      ridgeY: 12.25,
      stepsU: cornerSteps,
      includeBodyWalls: true,
      detailScale
    });

    // 5B. East Lakeside Courtyard Annex Wing (maxX = +54.00m EXACT, bordering Danau Kenanga!)
    PrimaryStructure._addFlatRoofCourtyardBlock(addMesh, materials, {
      name: 'EastAnnex_NorthFlatBlock',
      minX: 21.0,
      maxX: 43.5,
      minZ: -21.8,
      maxZ: -6.6,
      roofY: 5.75,
      parapetTopY: 6.32,
      detailScale
    });
    PrimaryStructure._addFlatRoofCourtyardBlock(addMesh, materials, {
      name: 'EastAnnex_SouthFlatBlock',
      minX: 21.0,
      maxX: 43.5,
      minZ: -0.4,
      maxZ: 13.2,
      roofY: 5.75,
      parapetTopY: 6.32,
      detailScale
    });
    PrimaryStructure._addPitchedGableRoof(addMesh, materials, {
      name: 'EastAnnex_TowerBridge',
      cx: 18.4,
      cz: -3.5,
      length: 6.4,
      halfWidth: 2.6,
      eaveY: 4.4,
      ridgeY: 6.1,
      alongAxis: 'X',
      detailScale
    });
    PrimaryStructure._addHippedPavilion(addMesh, materials, {
      name: 'EastAnnex_LakesideHall',
      cx: 43.75,
      cz: -4.3,
      halfX: 10.25, // +43.75 + 10.25 = +54.00m EXACT!
      halfZ: 12.2,
      ridgeHalfX: 0.35,
      ridgeHalfZ: 2.4,
      eaveY: 5.85,
      ridgeY: 12.25,
      stepsU: cornerSteps,
      includeBodyWalls: true,
      detailScale
    });

    // =========================================================================
    // 6. 45° SE DIAGONAL BREEZEWAY & OCTAGONAL GLASS-SKYLIT LAKESIDE PENDOPO
    // =========================================================================
    PrimaryStructure._addPitchedGableRoof(addMesh, materials, {
      name: 'SE45_DiagonalBreezewayRoof',
      cx: 19.2,
      cz: 19.2,
      length: 17.5,
      halfWidth: 2.35,
      eaveY: 3.65,
      ridgeY: 5.55,
      alongAxis: 'Z',
      rotY: Math.PI * 0.25,
      detailScale
    });

    const evalOctPendopo = MasterSurfaceKernel.createOctagonalSkylitRoofEvaluator({
      cx: 29.5,
      cz: 29.5,
      outerR: 8.2,
      skylightR: 2.25,
      eaveY: 3.75,
      fasciaY: 3.94,
      skylightBaseY: 6.85,
      skylightApexY: 8.15
    });
    const octSteps = isUltra ? 112 : 16;
    addMesh(
      MasterSurfaceBuilder.sliceSubdomain(
        evalOctPendopo,
        0.0,
        1.0,
        0.0,
        0.5,
        octSteps,
        isUltra ? 3 : 2
      ),
      materials.precastConcrete,
      'OctSkylitPendopo_SoffitFascia'
    );
    addMesh(
      PrimaryStructure._sliceCorrugatedTileSubdomain(
        evalOctPendopo,
        0.0,
        1.0,
        0.5,
        0.75,
        octSteps,
        isUltra ? 14 : 3,
        isUltra
          ? {
              wavesU: 64,
              coursesV: 10,
              amp: 0.018,
              taperU: false
            }
          : null,
        { uvMode: 'parametric', uvScaleU: 3.0, uvScaleV: 4.0 }
      ),
      materials.terracottaTile,
      'OctSkylitPendopo_TerracottaSlope'
    );
    addMesh(
      MasterSurfaceBuilder.sliceSubdomain(
        evalOctPendopo,
        0.0,
        1.0,
        0.75,
        1.0,
        octSteps,
        isUltra ? 6 : 2
      ),
      materials.atriumGlass,
      'OctSkylitPendopo_GlassPyramidSkylight'
    );

    // =========================================================================
    // 7. -Z NORTH STANDALONE PYRAMIDAL PAVILION ('PENDOPO UTARA', minZ = -52.00m EXACT!)
    // =========================================================================
    PrimaryStructure._addHippedPavilion(addMesh, materials, {
      name: 'North_PyramidalPavilion',
      cx: 5.0,
      cz: -43.5,
      halfX: 8.5,
      halfZ: 8.5, // -43.5 - 8.5 = -52.00m EXACT!
      ridgeHalfX: 0.25,
      ridgeHalfZ: 0.25,
      eaveY: 3.75,
      ridgeY: 8.55,
      stepsU: cornerSteps,
      includeLowWalls: true,
      detailScale
    });

    return { group, subdomainBoundaries };
  }

  /**
   * Slices a terracotta roof subdomain with physical 3D Javanese clay tile wave ribs & overlapping
   * horizontal tile courses while strictly zeroing normal displacement at vMin/vMax (and optional uMin/uMax)
   * so SSoST 0.0000m seam continuity is 100% mathematically preserved.
   */
  static _sliceCorrugatedTileSubdomain(
    evaluateFn,
    uMin,
    uMax,
    vMin,
    vMax,
    uSteps,
    vSteps,
    corrugationCfg = null,
    options = {}
  ) {
    let offsetFn = null;
    if (corrugationCfg) {
      const { wavesU = 64, coursesV = 10, amp = 0.02, taperU = false } = corrugationCfg;
      const vSpan = Math.max(1e-6, vMax - vMin);
      const uSpan = Math.max(1e-6, uMax - uMin);
      offsetFn = (u, v) => {
        const tV = Math.max(0.0, Math.min(1.0, (v - vMin) / vSpan));
        const envV = Math.sin(Math.PI * tV);
        if (envV <= 1e-5) return 0.0;
        let envU = 1.0;
        if (taperU) {
          const tU = Math.max(0.0, Math.min(1.0, (u - uMin) / uSpan));
          envU = Math.sin(Math.PI * tU);
        }
        const rib = Math.cos(2.0 * Math.PI * wavesU * u);
        const courseFrac = (tV * coursesV) % 1.0;
        const courseStep = (0.5 - courseFrac) * 0.72;
        return amp * envV * envU * (0.68 * rib + 0.32 * courseStep);
      };
    }

    const geo = MasterSurfaceBuilder.sliceSubdomain(
      evaluateFn,
      uMin,
      uMax,
      vMin,
      vMax,
      uSteps,
      vSteps,
      offsetFn,
      options
    );

    if (offsetFn) {
      geo.computeVertexNormals();
      PrimaryStructure._sanitizeNormals(geo);
    }
    return geo;
  }

  /**
   * Ensures every vertex normal is finite and unit-length after 3D relief normal recomputation.
   */
  static _sanitizeNormals(geo) {
    const normAttr = geo.attributes.normal;
    if (!normAttr) return;
    const arr = normAttr.array;
    for (let i = 0; i < arr.length; i += 3) {
      const nx = arr[i];
      const ny = arr[i + 1];
      const nz = arr[i + 2];
      const len = Math.hypot(nx, ny, nz);
      if (!Number.isFinite(len) || len < 1e-6) {
        arr[i] = 0.0;
        arr[i + 1] = 1.0;
        arr[i + 2] = 0.0;
      } else {
        arr[i] = nx / len;
        arr[i + 1] = ny / len;
        arr[i + 2] = nz / len;
      }
    }
  }

  /**
   * Builds a horizontal courtyard deck using the signature UI Grey-Grid + White-Diamond (◇◇◇) stone texture
   * strictly separated from its vertical sub-terrain retaining skirt (Hukum Emas 5 & 10).
   */
  static _addHorizontalDiamondPlaza(
    addMesh,
    materials,
    minX,
    maxX,
    minZ,
    maxZ,
    topY,
    name,
    withDeepSkirt,
    detailScale = 1.0
  ) {
    const w = maxX - minX;
    const d = maxZ - minZ;
    const cx = 0.5 * (minX + maxX);
    const cz = 0.5 * (minZ + maxZ);
    const isUltra = detailScale >= 0.9;

    const segsX = isUltra ? Math.max(12, Math.round(w * 0.45)) : 1;
    const segsZ = isUltra ? Math.max(12, Math.round(d * 0.45)) : 1;
    const deck = new THREE.PlaneGeometry(w, d, segsX, segsZ);
    PrimaryStructure._rotatePlaneToXZ(deck);
    deck.translate(cx, topY, cz);
    // World-planar UV scale so each ◇ diamond cell spans ~3.2m like the real Rektorat UI plaza!
    const pos = deck.attributes.position;
    const uv = deck.attributes.uv;
    for (let i = 0; i < pos.count; i++) {
      uv.array[i * 2] = pos.getX(i) * 0.042;
      uv.array[i * 2 + 1] = pos.getZ(i) * 0.042;
    }
    addMesh(deck, materials.horizontalPlazaStone, `${name}_DiamondTop`, false, true);

    const skirtDepth = withDeepSkirt ? 18.0 : 2.6;
    const skirtH = topY + skirtDepth;
    const skirtBox = new THREE.BoxGeometry(w, skirtH, d);
    skirtBox.translate(cx, topY - skirtH * 0.5 - 0.005, cz);
    const sCols = new Float32Array(skirtBox.attributes.position.count * 3);
    for (let ci = 0; ci < sCols.length; ci += 3) {
      sCols[ci] = 0.52;
      sCols[ci + 1] = 0.49;
      sCols[ci + 2] = 0.45;
    }
    skirtBox.setAttribute('color', new THREE.BufferAttribute(sCols, 3));
    addMesh(skirtBox, materials.foundationConcrete, `${name}_Skirt`, false, true);
  }

  /**
   * Builds a watertight Pitched Gable/Saddle Terracotta Roof + White Underside Beam/Soffit + White Triangular Pediments.
   * On 10x Ultra-LOD (detailScale >= 0.9), generates subdivided 3D corrugated Javanese clay tile courses,
   * a 3D cylindrical half-round ridge cap (bubungan atas), and molded gable pediment bargeboards!
   */
  static _addPitchedGableRoof(
    addMesh,
    materials,
    { name, cx, cz, length, halfWidth, eaveY, ridgeY, alongAxis = 'Z', rotY = 0.0, detailScale = 1.0 }
  ) {
    const baseRotY = alongAxis === 'X' ? Math.PI * 0.5 : 0.0;
    const totalRotY = baseRotY + rotY;
    const hl = length * 0.5;
    const hw = halfWidth;
    const fasciaH = 0.22;
    const isUltra = detailScale >= 0.9;

    const yEaveTop = eaveY + fasciaH;
    const dy = ridgeY - yEaveTop;
    const slopeLen = Math.hypot(hw, dy);
    const uScale = length * 0.14;
    const vScale = slopeLen * 0.22;

    if (!isUltra) {
      // Lightweight LOD1/LOD2 representation
      const slopePositions = new Float32Array([
        0, ridgeY, hl,    hw, yEaveTop, hl,    hw, yEaveTop, -hl,
        0, ridgeY, hl,    hw, yEaveTop, -hl,   0, ridgeY, -hl,
        -hw, yEaveTop, hl,   0, ridgeY, hl,    0, ridgeY, -hl,
        -hw, yEaveTop, hl,   0, ridgeY, -hl,   -hw, yEaveTop, -hl
      ]);
      const tiledUvs = new Float32Array([
        0, 0,       0, vScale,  uScale, vScale,   0, 0,       uScale, vScale, uScale, 0,
        0, vScale,  0, 0,       uScale, 0,        0, vScale,  uScale, 0,      uScale, vScale
      ]);
      const roofGeo = new THREE.BufferGeometry();
      roofGeo.setAttribute('position', new THREE.BufferAttribute(slopePositions, 3));
      roofGeo.setAttribute('uv', new THREE.BufferAttribute(tiledUvs, 2));
      roofGeo.computeVertexNormals();
      PrimaryStructure._transformGeometry(roofGeo, cx, 0, cz, totalRotY);
      addMesh(roofGeo, materials.terracottaTile, `${name}_Tiles`);
    } else {
      // 10x Ultra-LOD: Subdivided 3D Corrugated Javanese Clay Tile Courses + 3D Cylindrical Ridge Roll!
      const zSegs = Math.max(18, Math.min(72, Math.round(length * 2.2)));
      const sSegs = 12;
      const tileWaves = Math.max(10, Math.round(length * 2.6));
      const tileCourses = Math.max(6, Math.round(slopeLen * 2.8));

      const positions = [];
      const uvs = [];
      const indices = [];

      const buildSlopeSide = (sideSign) => {
        const nLen = Math.hypot(dy, hw) || 1.0;
        const nx = (sideSign * dy) / nLen;
        const ny = hw / nLen;
        const baseIdx = positions.length / 3;

        for (let iz = 0; iz <= zSegs; iz++) {
          const tz = iz / zSegs;
          const lz = hl - tz * length;
          const envZ = Math.sin(Math.PI * tz);
          const rib = Math.cos(2.0 * Math.PI * tileWaves * tz);

          for (let is = 0; is <= sSegs; is++) {
            const ts = is / sSegs; // 0 at ridge -> 1 at eave
            const envS = Math.sin(Math.PI * ts);
            const courseFrac = (ts * tileCourses) % 1.0;
            const courseStep = (0.5 - courseFrac) * 0.75;
            const disp = 0.019 * envZ * envS * (0.68 * rib + 0.32 * courseStep);

            const lx = sideSign * hw * ts + nx * disp;
            const ly = ridgeY - dy * ts + ny * disp;

            positions.push(lx, ly, lz);
            uvs.push(tz * uScale, ts * vScale);
          }
        }

        const stride = sSegs + 1;
        for (let iz = 0; iz < zSegs; iz++) {
          for (let is = 0; is < sSegs; is++) {
            const a = baseIdx + iz * stride + is;
            const b = baseIdx + (iz + 1) * stride + is;
            const c = baseIdx + (iz + 1) * stride + (is + 1);
            const d = baseIdx + iz * stride + (is + 1);
            if (sideSign > 0) {
              // +X slope: a->d goes (+X, -Y), a->b goes (-Z); (+X, -Y) x (-Z) = (+dy, +hw, 0) OUTWARD!
              indices.push(a, d, c, a, c, b);
            } else {
              // -X slope: a->b goes (-Z), a->d goes (-X, -Y); (-Z) x (-X, -Y) = (-dy, +hw, 0) OUTWARD!
              indices.push(a, b, c, a, c, d);
            }
          }
        }
      };

      buildSlopeSide(1);
      buildSlopeSide(-1);

      // 3D Cylindrical Half-Round Ridge Roll ('Bubungan Genteng Atas') with Mortar Collar Rings
      const rSegs = 10;
      const ridgeBase = positions.length / 3;
      const collarWaves = Math.max(8, Math.round(length * 1.8));
      for (let iz = 0; iz <= zSegs; iz++) {
        const tz = iz / zSegs;
        const lz = hl - tz * length;
        const collar = 1.0 + 0.14 * Math.pow(0.5 + 0.5 * Math.cos(2.0 * Math.PI * collarWaves * tz), 4.0);
        const rad = 0.115 * collar;

        for (let ir = 0; ir <= rSegs; ir++) {
          const tr = ir / rSegs;
          const ang = -Math.PI * 0.55 + tr * Math.PI * 1.1;
          const lx = Math.sin(ang) * rad;
          const ly = ridgeY - 0.03 + Math.cos(ang) * rad;
          positions.push(lx, ly, lz);
          uvs.push(tz * uScale, tr * 0.35);
        }
      }
      const rStride = rSegs + 1;
      for (let iz = 0; iz < zSegs; iz++) {
        for (let ir = 0; ir < rSegs; ir++) {
          const a = ridgeBase + iz * rStride + ir;
          const b = ridgeBase + (iz + 1) * rStride + ir;
          const c = ridgeBase + (iz + 1) * rStride + (ir + 1);
          const d = ridgeBase + iz * rStride + (ir + 1);
          indices.push(a, d, c, a, c, b);
        }
      }

      const roofGeo = new THREE.BufferGeometry();
      roofGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
      roofGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
      roofGeo.setIndex(indices);
      roofGeo.computeVertexNormals();
      PrimaryStructure._sanitizeNormals(roofGeo);
      PrimaryStructure._transformGeometry(roofGeo, cx, 0, cz, totalRotY);
      addMesh(roofGeo, materials.terracottaTile, `${name}_Tiles`);

      // Molded 3D White Gable Bargeboards ('Lisplang Pelana') along front (+hl) and back (-hl)
      const bargeLen = slopeLen + 0.08;
      const slopePitch = Math.atan2(dy, hw);
      for (const zEnd of [-hl + 0.09, hl - 0.09]) {
        for (const sgn of [-1, 1]) {
          const barge = new THREE.BoxGeometry(bargeLen, 0.16, 0.14);
          barge.rotateZ(-sgn * slopePitch);
          barge.translate(sgn * hw * 0.5, yEaveTop + dy * 0.5 - 0.06, zEnd);
          PrimaryStructure._transformGeometry(barge, cx, 0, cz, totalRotY);
          addMesh(barge, materials.precastConcrete, `${name}_Bargeboard_${zEnd > 0 ? 'F' : 'B'}_${sgn}`);
        }
      }
    }

    // 2. Crisp Off-White Triangular Gable Pediments (+hl and -hl) + Horizontal Underside Soffit Slab
    const soffitSlab = new THREE.BoxGeometry(hw * 2.0, fasciaH, length);
    PrimaryStructure._transformGeometry(soffitSlab, cx, eaveY + fasciaH * 0.5, cz, totalRotY);
    addMesh(soffitSlab, materials.precastConcrete, `${name}_Soffit`);

    const pedWallDepth = length - 0.35;
    const pedPositions = new Float32Array([
      -hw * 0.94, yEaveTop, pedWallDepth * 0.5,
      hw * 0.94, yEaveTop, pedWallDepth * 0.5,
      0, ridgeY - 0.08, pedWallDepth * 0.5,
      hw * 0.94, yEaveTop, -pedWallDepth * 0.5,
      -hw * 0.94, yEaveTop, -pedWallDepth * 0.5,
      0, ridgeY - 0.08, -pedWallDepth * 0.5
    ]);
    const pedUvs = new Float32Array([0, 0, 1, 0, 0.5, 1, 0, 0, 1, 0, 0.5, 1]);
    const pedGeo = new THREE.BufferGeometry();
    pedGeo.setAttribute('position', new THREE.BufferAttribute(pedPositions, 3));
    pedGeo.setAttribute('uv', new THREE.BufferAttribute(pedUvs, 2));
    pedGeo.computeVertexNormals();
    PrimaryStructure._transformGeometry(pedGeo, cx, 0, cz, totalRotY);
    addMesh(pedGeo, materials.precastConcrete, `${name}_WhitePediments`);
  }

  /**
   * Builds a Hipped / Pyramidal Nusantara Pavilion Roof + optional Annex Hall Walls & Plinth.
   */
  static _addHippedPavilion(
    addMesh,
    materials,
    {
      name,
      cx,
      cz,
      halfX,
      halfZ,
      ridgeHalfX = 0.25,
      ridgeHalfZ = 0.25,
      eaveY = 3.75,
      ridgeY = 8.5,
      stepsU = 24,
      includeBodyWalls = false,
      includeLowWalls = false,
      detailScale = 1.0
    }
  ) {
    const isUltra = detailScale >= 0.9;
    const evalRoof = MasterSurfaceKernel.createHippedRoofEvaluator({
      cx,
      cz,
      halfX,
      halfZ,
      ridgeHalfX,
      ridgeHalfZ,
      eaveY,
      fasciaY: eaveY + 0.22,
      ridgeY: ridgeY - 0.12,
      capTopY: ridgeY
    });

    addMesh(
      MasterSurfaceBuilder.sliceSubdomain(
        evalRoof,
        0.0,
        1.0,
        0.0,
        0.5,
        stepsU,
        isUltra ? 3 : 2
      ),
      materials.precastConcrete,
      `${name}_SoffitFascia`
    );
    addMesh(
      PrimaryStructure._sliceCorrugatedTileSubdomain(
        evalRoof,
        0.0,
        1.0,
        0.5,
        0.75,
        stepsU,
        isUltra ? 14 : 3,
        isUltra
          ? {
              wavesU: 84,
              coursesV: 12,
              amp: 0.018,
              taperU: false
            }
          : null,
        { uvMode: 'parametric', uvScaleU: 4.0, uvScaleV: 6.0 }
      ),
      materials.terracottaTile,
      `${name}_TileSlope`
    );
    addMesh(
      MasterSurfaceBuilder.sliceSubdomain(
        evalRoof,
        0.0,
        1.0,
        0.75,
        1.0,
        stepsU,
        isUltra ? 2 : 1
      ),
      materials.precastConcrete,
      `${name}_RidgeCap`
    );

    // Base plinth + -2.8m skirt (never floating!)
    const wallW = (halfX - 1.1) * 2.0;
    const wallD = (halfZ - 1.1) * 2.0;
    const plinth = new THREE.BoxGeometry(wallW + 0.8, 3.0, wallD + 0.8);
    plinth.translate(cx, 0.22 - 1.5, cz);
    addMesh(plinth, materials.precastConcrete, `${name}_PlinthSkirt`);

    if (includeBodyWalls) {
      // Enclosed 2-story off-white annex hall body + dark horizontal ribbon window band
      const lowerWall = new THREE.BoxGeometry(wallW, 3.4, wallD);
      lowerWall.translate(cx, 1.82, cz);
      addMesh(lowerWall, materials.precastConcrete, `${name}_LowerWall`);

      if (isUltra) {
        const sillMolding = new THREE.BoxGeometry(wallW + 0.24, 0.16, wallD + 0.24);
        sillMolding.translate(cx, 3.48, cz);
        addMesh(sillMolding, materials.precastConcrete, `${name}_WindowSillMolding`);
      }

      const winBand = new THREE.BoxGeometry(wallW - 0.25, 1.65, wallD - 0.25);
      winBand.translate(cx, 4.35, cz);
      addMesh(winBand, materials.ribbonGlass, `${name}_RibbonWin`);

      const upperBeam = new THREE.BoxGeometry(wallW + 0.1, 0.7, wallD + 0.1);
      upperBeam.translate(cx, 5.52, cz);
      addMesh(upperBeam, materials.precastConcrete, `${name}_UpperBeam`);
    } else if (includeLowWalls) {
      const lowWall = new THREE.BoxGeometry(wallW * 0.78, 3.55, wallD * 0.78);
      lowWall.translate(cx, 1.9, cz);
      addMesh(lowWall, materials.precastConcrete, `${name}_CoreRoom`);
    }
  }

  /**
   * Builds a Flat-Roofed Courtyard Annex Block with Weathered Grey Concrete Roof Slab
   * recessed inside Crisp Off-White Concrete Parapet Walls (Mavic Nadir Photos #2, #3, & #5!).
   */
  static _addFlatRoofCourtyardBlock(
    addMesh,
    materials,
    { name, minX, maxX, minZ, maxZ, roofY = 5.75, parapetTopY = 6.32, detailScale = 1.0 }
  ) {
    const w = maxX - minX;
    const d = maxZ - minZ;
    const cx = 0.5 * (minX + maxX);
    const cz = 0.5 * (minZ + maxZ);
    const isUltra = detailScale >= 0.9;

    // 1. Buried foundation skirt + lower off-white architectural wall
    const lowerH = 3.5 + 3.2; // from y = -3.2m skirt to y = +3.5m
    const lowerBox = new THREE.BoxGeometry(w, lowerH, d);
    lowerBox.translate(cx, 3.5 - lowerH * 0.5, cz);
    addMesh(lowerBox, materials.precastConcrete, `${name}_BaseAndSkirt`);

    if (isUltra) {
      // Ground-level water-table plinth belt & window sill ledge
      const plinthBelt = new THREE.BoxGeometry(w + 0.18, 0.36, d + 0.18);
      plinthBelt.translate(cx, 0.36, cz);
      addMesh(plinthBelt, materials.precastConcrete, `${name}_PlinthBelt`);

      const sillLedge = new THREE.BoxGeometry(w + 0.2, 0.14, d + 0.2);
      sillLedge.translate(cx, 3.46, cz);
      addMesh(sillLedge, materials.precastConcrete, `${name}_WindowSill`);
    }

    // 2. Recessed upper ribbon window band (y = 3.5m .. 5.1m)
    const winBox = new THREE.BoxGeometry(w - 0.36, 1.6, d - 0.36);
    winBox.translate(cx, 4.3, cz);
    addMesh(winBox, materials.ribbonGlass, `${name}_Windows`);

    // 3. Upper off-white concrete cornice & 4 rooftop parapet walls (y = 5.1m .. parapetTopY)
    const cornice = new THREE.BoxGeometry(w + 0.16, roofY - 5.1, d + 0.16);
    cornice.translate(cx, 5.1 + (roofY - 5.1) * 0.5, cz);
    addMesh(cornice, materials.precastConcrete, `${name}_Cornice`);

    const pH = parapetTopY - roofY;
    const pT = 0.55;
    for (const sz of [-1, 1]) {
      const pZ = new THREE.BoxGeometry(w + 0.16, pH, pT);
      pZ.translate(cx, roofY + pH * 0.5, cz + sz * (d * 0.5 - pT * 0.5 + 0.08));
      addMesh(pZ, materials.precastConcrete, `${name}_ParapetZ_${sz}`);
      if (isUltra) {
        const capZ = new THREE.BoxGeometry(w + 0.24, 0.09, pT + 0.12);
        capZ.translate(cx, parapetTopY + 0.045, cz + sz * (d * 0.5 - pT * 0.5 + 0.08));
        addMesh(capZ, materials.precastConcrete, `${name}_ParapetCapZ_${sz}`);
      }
    }
    for (const sx of [-1, 1]) {
      const pX = new THREE.BoxGeometry(pT, pH, d - pT * 2 + 0.16);
      pX.translate(cx + sx * (w * 0.5 - pT * 0.5 + 0.08), roofY + pH * 0.5, cz);
      addMesh(pX, materials.precastConcrete, `${name}_ParapetX_${sx}`);
      if (isUltra) {
        const capX = new THREE.BoxGeometry(pT + 0.12, 0.09, d - pT * 2 + 0.24);
        capX.translate(cx + sx * (w * 0.5 - pT * 0.5 + 0.08), parapetTopY + 0.045, cz);
        addMesh(capX, materials.precastConcrete, `${name}_ParapetCapX_${sx}`);
      }
    }

    // 4. Horizontal Weathered Grey Waterproof Concrete Flat Roof Deck (Hukum Emas 10!)
    const flatDeck = new THREE.PlaneGeometry(
      w - pT * 1.6,
      d - pT * 1.6,
      isUltra ? 12 : 1,
      isUltra ? 8 : 1
    );
    PrimaryStructure._rotatePlaneToXZ(flatDeck);
    flatDeck.translate(cx, roofY + 0.02, cz);
    addMesh(flatDeck, materials.flatRoofSlab, `${name}_WeatheredFlatDeck`, false, true);
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
