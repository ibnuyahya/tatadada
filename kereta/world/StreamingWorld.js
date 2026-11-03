import * as THREE from 'three';
import { GAME_PREMISE_CARTRIDGE } from '../data/PremiseCartridge.js';
import { EndlessWorldMathematics, createTrackFrameStruct } from './WorldMath.js';
import { ProceduralAssetFactory } from '../assets/ProceduralTextures.js';
import { GeometryBatcher } from '../engine/GeometryBatcher.js';
import { ChunkGeometryCore } from '../workers/ChunkGeometryCore.js';

/**
 * ============================================================================
 * NEXT-GEN STREAMING RAILWAY WORLD (WEB WORKER + FLOATING ORIGIN + GPU BATCHED)
 * - Offloads chunk spline/terrain/scenery geometry + normal synthesis to an ES
 *   Module Web Worker (`ChunkBuilderWorker.js`) using zero-copy Transferables.
 * - Anchors every chunk in chunk-local space (`originS = sStart`) and shifts
 *   `group.position.z` in O(1) time during Floating-Origin rebasing.
 * - Pre-bakes static multi-part infrastructure (catenary portals, viaducts,
 *   tunnels, signals, crossings, mega-station structures & rigid NPC bone
 *   segments) into shared GPU BufferGeometries with up to 90% fewer draw calls.
 * ============================================================================
 */
const _rotMat = new THREE.Matrix4();
const _negTangent = new THREE.Vector3();
const _frameTmp = createTrackFrameStruct();

export class StreamingRailwayWorld {
  constructor(scene, textures, envMap) {
    this.scene = scene;
    this.textures = textures;
    this.envMap = envMap;

    this.chunks = new Map();
    this.pendingChunks = new Set();
    this.completedQueue = [];

    this.levelCrossings = [];
    this.blockSignals = [];
    this.stationPassengers = [];
    this.stackPsdDoors = [];
    this.windTurbineRotors = [];

    this.stationNpcCount = 104;

    this._initMaterials();
    this._initSharedInfrastructureGeometries();
    this._initSharedNPCAssets();
    this._initWorker();
  }

  _initMaterials() {
    const textures = this.textures;

    this.profiledRailMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      metalness: 0.92,
      roughness: 0.16,
      envMapIntensity: 1.45
    });
    this.wireMat = new THREE.MeshBasicMaterial({
      color: 0x1e293b,
      side: THREE.DoubleSide
    });
    this.ballastMat = new THREE.MeshStandardMaterial({
      map: textures.ballast,
      bumpMap: textures.microBump,
      bumpScale: 0.14,
      vertexColors: true,
      roughness: 0.86,
      metalness: 0.08
    });
    this.chunkRiverTex = textures.waterRipple.clone();
    this.chunkRiverTex.wrapS = THREE.RepeatWrapping;
    this.chunkRiverTex.wrapT = THREE.RepeatWrapping;
    this.chunkRiverTex.needsUpdate = true;
    this.chunkWaterMat = new THREE.MeshPhysicalMaterial({
      vertexColors: true,
      map: this.chunkRiverTex,
      bumpMap: this.chunkRiverTex,
      bumpScale: 0.14,
      roughness: 0.08,
      metalness: 0.68,
      clearcoat: 0.92,
      clearcoatRoughness: 0.06,
      envMapIntensity: 1.5,
      transparent: true,
      opacity: 0.92
    });
    this.terrainMat = new THREE.MeshStandardMaterial({
      map: textures.terrainDetail,
      bumpMap: textures.microBump,
      bumpScale: 0.11,
      vertexColors: true,
      roughness: 0.90,
      metalness: 0.02
    });
    this.sceneryBatchMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      bumpMap: textures.microBump,
      bumpScale: 0.06,
      roughness: 0.78,
      metalness: 0.08
    });

    // Shared Vertex-Colored PBR Materials for Batched Infrastructure
    this.infraPbrMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.52,
      metalness: 0.42,
      envMapIntensity: 1.15
    });
    this.infraMetalMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.26,
      metalness: 0.78,
      envMapIntensity: 1.3
    });
    this.infraConcreteMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      bumpMap: textures.microBump,
      bumpScale: 0.08,
      roughness: 0.82,
      metalness: 0.06
    });
    this.infraEmissiveMat = new THREE.MeshBasicMaterial({
      vertexColors: true
    });

    this.concreteMat = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      bumpMap: textures.microBump,
      bumpScale: 0.10,
      roughness: 0.82
    });
    this.insulatorMat = new THREE.MeshStandardMaterial({
      color: 0xf8fafc,
      roughness: 0.22,
      metalness: 0.12
    });
    this.platformTopMat = new THREE.MeshStandardMaterial({
      map: textures.platformTop,
      bumpMap: textures.microBump,
      bumpScale: 0.06,
      roughness: 0.68
    });
    this.platformMat = new THREE.MeshStandardMaterial({
      map: textures.platform,
      bumpMap: textures.microBump,
      bumpScale: 0.08,
      roughness: 0.78
    });
    this.roofDarkMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      metalness: 0.72,
      roughness: 0.24,
      envMapIntensity: 1.2
    });
    this.psdDoorGlassMat = new THREE.MeshPhysicalMaterial({
      color: 0xbae6fd,
      metalness: 0.15,
      roughness: 0.08,
      clearcoat: 1.0,
      clearcoatRoughness: 0.05,
      envMapIntensity: 1.4,
      transparent: true,
      opacity: 0.48
    });
    this.stationGlassMat = new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8,
      metalness: 0.2,
      roughness: 0.10,
      clearcoat: 0.9,
      envMapIntensity: 1.35,
      transparent: true,
      opacity: 0.52
    });
    this.tunnelMat = new THREE.MeshStandardMaterial({
      map: textures.tunnel,
      bumpMap: textures.microBump,
      bumpScale: 0.16,
      roughness: 0.85,
      side: THREE.DoubleSide
    });
    this.skyscraperMat = new THREE.MeshPhysicalMaterial({
      map: textures.skyscraper,
      roughness: 0.18,
      metalness: 0.68,
      clearcoat: 0.65,
      envMapIntensity: 1.4
    });

    this.sigRedOffMat = new THREE.MeshBasicMaterial({ color: 0x3b0712 });
    this.sigRedOnMat = new THREE.MeshBasicMaterial({ color: 0xf43f5e });
    this.sigYelOffMat = new THREE.MeshBasicMaterial({ color: 0x451a03 });
    this.sigYelOnMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });
    this.sigGrnOffMat = new THREE.MeshBasicMaterial({ color: 0x064e3b });
    this.sigGrnOnMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
    this.beaconRedMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });

    textures.platformTop.repeat.set(1, 38);
    textures.platform.repeat.set(28, 1);

    this.platMats = [
      this.platformMat,
      this.platformMat,
      this.platformTopMat,
      this.concreteMat,
      this.concreteMat,
      this.concreteMat
    ];

    this.rockMatCache = new Map();
    this.facadeMatCache = new Map();
    this.signMatCache = new Map();
  }

  _getRockShellMat(hexColor) {
    if (!this.rockMatCache.has(hexColor)) {
      this.rockMatCache.set(
        hexColor,
        new THREE.MeshStandardMaterial({
          color: hexColor,
          roughness: 0.92,
          side: THREE.DoubleSide
        })
      );
    }
    return this.rockMatCache.get(hexColor);
  }

  _getStationFacadeMat(stName) {
    if (!this.facadeMatCache.has(stName)) {
      const tex = ProceduralAssetFactory.createStationFacadeTexture(stName);
      this.facadeMatCache.set(
        stName,
        new THREE.MeshStandardMaterial({
          map: tex,
          roughness: 0.25,
          metalness: 0.55,
          envMapIntensity: 1.25
        })
      );
    }
    return this.facadeMatCache.get(stName);
  }

  _getStationSignMat(stName, elevM) {
    const key = `${stName}_${elevM}`;
    if (!this.signMatCache.has(key)) {
      const tex = ProceduralAssetFactory.createStationSignTexture(stName, elevM);
      this.signMatCache.set(
        key,
        new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide })
      );
    }
    return this.signMatCache.get(key);
  }

  _initSharedInfrastructureGeometries() {
    const B = GeometryBatcher;
    this.sharedSigLampGeo = new THREE.SphereGeometry(0.14, 10, 10);
    this.sharedSigLampGeo.userData.sharedGeo = true;
    this.sharedCxLampGeo = new THREE.SphereGeometry(0.18, 8, 8);
    this.sharedCxLampGeo.userData.sharedGeo = true;

    // 1. Merged Double-Track Catenary Portal (12 parts -> 1 single draw call!)
    {
      const mast = new THREE.BoxGeometry(0.24, 6.8, 0.24);
      const beam = new THREE.BoxGeometry(9.8, 0.22, 0.22);
      const ins = new THREE.CylinderGeometry(0.08, 0.08, 0.55, 6);
      const foot = new THREE.BoxGeometry(0.65, 2.2, 0.65);
      const steady = new THREE.CylinderGeometry(0.025, 0.025, 1.35, 6);
      const regClamp = new THREE.CylinderGeometry(0.022, 0.022, 0.96, 6);
      const weight = new THREE.CylinderGeometry(0.14, 0.14, 0.95, 8);
      const hvSign = new THREE.BoxGeometry(0.32, 0.32, 0.28);

      this.mergedCatenaryPortalGeo = B.mergeParts([
        B.bakePart(mast, { position: [-4.8, 2.6, 0], color: 0x475569 }),
        B.bakePart(mast, { position: [4.8, 2.6, 0], color: 0x475569 }),
        B.bakePart(foot, { position: [-4.8, -1.2, 0], color: 0x64748b }),
        B.bakePart(foot, { position: [4.8, -1.2, 0], color: 0x64748b }),
        B.bakePart(beam, { position: [0, 6.28, 0], color: 0x475569 }),
        B.bakePart(beam, { position: [0, 5.85, 0], color: 0x475569 }),
        B.bakePart(ins, { position: [-2.2, 5.75, 0], color: 0xf8fafc }),
        B.bakePart(ins, { position: [2.2, 5.75, 0], color: 0xf8fafc }),
        B.bakePart(regClamp, { position: [-2.2, 5.80, 0], color: 0x94a3b8 }),
        B.bakePart(regClamp, { position: [2.2, 5.80, 0], color: 0x94a3b8 }),
        B.bakePart(steady, { position: [-2.75, 5.45, 0], rotation: [0, 0, 1.18], color: 0x475569 }),
        B.bakePart(steady, { position: [2.75, 5.45, 0], rotation: [0, 0, -1.18], color: 0x475569 }),
        B.bakePart(weight, { position: [-5.05, 1.85, 0], color: 0x64748b }),
        B.bakePart(hvSign, { position: [-4.8, 3.2, 0.05], color: 0xfacc15 })
      ]);
      [mast, beam, ins, foot, steady, regClamp, weight, hvSign].forEach(g => g.dispose());
    }

    // 2. Merged Double-Track Viaduct Bays, Concrete Box-Girder Approaches, Waterline Cutwaters & Bridge Abutments!
    {
      const bayLen = 10.25;
      const deckGeo = new THREE.BoxGeometry(11.2, 0.58, bayLen);
      const boxSpineGeo = new THREE.BoxGeometry(7.8, 1.65, bayLen);
      const parapetWallGeo = new THREE.BoxGeometry(0.28, 1.18, bayLen);
      const walkwayLipGeo = new THREE.BoxGeometry(0.65, 0.22, bayLen);
      const pierCapGeo = new THREE.BoxGeometry(9.6, 1.35, 3.2);
      const bearingPadGeo = new THREE.BoxGeometry(1.15, 0.28, 1.15);
      const pierShaftGeo = new THREE.BoxGeometry(7.6, 68.0, 2.5);

      const baseDeckParts = [
        B.bakePart(deckGeo, { position: [0, -0.46, 0], color: 0x94a3b8 }),
        B.bakePart(boxSpineGeo, { position: [0, -1.48, 0], color: 0x64748b }),
        B.bakePart(walkwayLipGeo, { position: [-5.55, -0.28, 0], color: 0x94a3b8 }),
        B.bakePart(walkwayLipGeo, { position: [5.55, -0.28, 0], color: 0x94a3b8 }),
        B.bakePart(parapetWallGeo, { position: [-5.68, 0.38, 0], color: 0xcbd5e1 }),
        B.bakePart(parapetWallGeo, { position: [5.68, 0.38, 0], color: 0xcbd5e1 })
      ];

      this.mergedViaductDeckNoPierGeo = B.mergeParts(baseDeckParts);
      this.mergedViaductDeckWithPierGeo = B.mergeParts([
        ...baseDeckParts,
        B.bakePart(bearingPadGeo, { position: [-2.8, -2.38, 0], color: 0x1e293b }),
        B.bakePart(bearingPadGeo, { position: [2.8, -2.38, 0], color: 0x1e293b }),
        B.bakePart(pierCapGeo, { position: [0, -3.12, 0], color: 0x94a3b8 }),
        B.bakePart(pierShaftGeo, { position: [0, -37.2, 0], color: 0x64748b })
      ]);

      // Hydrodynamic Pointed River Cutwater / Starling Pier Fender (placed dynamically right at river water level!)
      const cutBody = new THREE.BoxGeometry(8.6, 3.8, 3.4);
      const cutNose = new THREE.ConeGeometry(2.4, 3.8, 4);
      const foamCollar = new THREE.BoxGeometry(9.3, 0.24, 4.1);
      this.mergedPierCutwaterGeo = B.mergeParts([
        B.bakePart(cutBody, { position: [0, -0.6, 0], color: 0x475569 }),
        B.bakePart(cutNose, { position: [-5.1, -0.6, 0], rotation: [0, Math.PI / 4, 0], scale: [1.0, 1.0, 0.72], color: 0x334155 }),
        B.bakePart(cutNose, { position: [5.1, -0.6, 0], rotation: [0, Math.PI / 4, 0], scale: [1.0, 1.0, 0.72], color: 0x334155 }),
        B.bakePart(foamCollar, { position: [0, 0.04, 0], color: 0xe0f2fe })
      ]);

      // Approach Viaduct Steel Handrails (for concrete box-girder spans before main steel truss)
      const railBar = new THREE.BoxGeometry(0.08, 0.08, bayLen);
      const railPost = new THREE.BoxGeometry(0.10, 0.75, 0.10);
      const approachRailParts = [];
      [-5.68, 5.68].forEach((sx) => {
        approachRailParts.push(
          B.bakePart(railBar, { position: [sx, 1.28, 0], color: 0x0284c7 }),
          B.bakePart(railBar, { position: [sx, 1.04, 0], color: 0x94a3b8 }),
          B.bakePart(railPost, { position: [sx, 1.02, -bayLen * 0.35], color: 0x475569 }),
          B.bakePart(railPost, { position: [sx, 1.02, bayLen * 0.35], color: 0x475569 })
        );
      });
      this.mergedApproachRailGeo = B.mergeParts(approachRailParts);

      // Reinforced Concrete Bridge Abutments + Flared Wing Walls (anchors BOTH Railway Viaduct & Parallel Highway Bridge!)
      const railHeadwall = new THREE.BoxGeometry(12.4, 8.5, 2.6);
      const railWing = new THREE.BoxGeometry(1.1, 7.2, 6.4);
      const hwyHeadwall = new THREE.BoxGeometry(10.8, 7.8, 2.4);
      const hwyWing = new THREE.BoxGeometry(0.95, 6.5, 5.4);
      this.mergedBridgeAbutmentGeo = B.mergeParts([
        B.bakePart(railHeadwall, { position: [0, -4.35, 0], color: 0x64748b }),
        B.bakePart(railWing, { position: [-6.4, -3.8, 2.6], rotation: [0, -0.26, 0], color: 0x475569 }),
        B.bakePart(railWing, { position: [6.4, -3.8, 2.6], rotation: [0, 0.26, 0], color: 0x475569 }),
        B.bakePart(hwyHeadwall, { position: [19.5, -4.45, 0], color: 0x64748b }),
        B.bakePart(hwyWing, { position: [14.0, -3.9, 2.2], rotation: [0, -0.24, 0], color: 0x475569 }),
        B.bakePart(hwyWing, { position: [25.0, -3.9, 2.2], rotation: [0, 0.24, 0], color: 0x475569 })
      ]);

      const topChord = new THREE.BoxGeometry(0.36, 0.36, bayLen);
      const botChord = new THREE.BoxGeometry(0.42, 0.45, bayLen);
      const vPost = new THREE.BoxGeometry(0.32, 6.65, 0.32);
      const diag = new THREE.BoxGeometry(0.24, 12.2, 0.24);
      const topCross = new THREE.BoxGeometry(10.36, 0.32, 0.36);
      const ins = new THREE.CylinderGeometry(0.08, 0.08, 0.55, 6);

      const trussParts = [
        B.bakePart(topCross, { position: [0, 6.2, -bayLen * 0.5], color: 0x0284c7 }),
        B.bakePart(ins, { position: [-2.2, 5.75, -bayLen * 0.5], color: 0xf8fafc }),
        B.bakePart(ins, { position: [2.2, 5.75, -bayLen * 0.5], color: 0xf8fafc })
      ];
      [-5.0, 5.0].forEach((sx) => {
        trussParts.push(
          B.bakePart(topChord, { position: [sx, 6.2, 0], color: 0x0284c7 }),
          B.bakePart(botChord, { position: [sx, -0.15, 0], color: 0x0284c7 }),
          B.bakePart(vPost, { position: [sx, 3.02, -bayLen * 0.5], color: 0x0284c7 }),
          B.bakePart(diag, { position: [sx, 3.02, 0], rotation: [0.98, 0, 0], color: 0x0284c7 })
        );
      });
      this.mergedViaductTrussGeo = B.mergeParts(trussParts);

      [
        deckGeo, boxSpineGeo, parapetWallGeo, walkwayLipGeo, pierCapGeo, bearingPadGeo, pierShaftGeo,
        cutBody, cutNose, foamCollar, railBar, railPost, railHeadwall, railWing, hwyHeadwall, hwyWing,
        topChord, botChord, vPost, diag, topCross, ins
      ].forEach(g => g.dispose());
    }

    // 3. Shared & Merged Tunnel Segment Geometries (12 parts -> 3 draw calls!)
    {
      const bayLen = 12.7;
      this.sharedTunnelArchGeo = new THREE.CylinderGeometry(6.9, 6.9, bayLen, 16, 1, true, -Math.PI * 0.54, Math.PI * 1.08);
      this.sharedTunnelArchGeo.rotateX(Math.PI / 2);
      this.sharedTunnelArchGeo.translate(0, 1.4, 0);
      this.sharedTunnelArchGeo.userData.sharedGeo = true;

      this.sharedTunnelRockGeo = new THREE.CylinderGeometry(11.5, 13.5, bayLen, 10, 1, true, -Math.PI * 0.5, Math.PI);
      this.sharedTunnelRockGeo.rotateX(Math.PI / 2);
      this.sharedTunnelRockGeo.translate(0, 1.4, 0);
      this.sharedTunnelRockGeo.userData.sharedGeo = true;

      const ribGeo = new THREE.CylinderGeometry(6.78, 6.78, 0.55, 16, 1, true, -Math.PI * 0.54, Math.PI * 1.08);
      ribGeo.rotateX(Math.PI / 2);
      const walkGeo = new THREE.BoxGeometry(1.15, 0.35, bayLen);
      const railGeo = new THREE.BoxGeometry(0.06, 0.85, bayLen);
      const fanGeo = new THREE.CylinderGeometry(0.38, 0.38, 1.85, 10);
      fanGeo.rotateX(Math.PI / 2);
      const sconceGeo = new THREE.BoxGeometry(0.18, 0.22, 1.4);

      this.mergedTunnelHardwareGeo = B.mergeParts([
        B.bakePart(ribGeo, { position: [0, 1.4, 0], color: 0x64748b }),
        B.bakePart(walkGeo, { position: [-6.15, 0.15, 0], color: 0x64748b }),
        B.bakePart(walkGeo, { position: [6.15, 0.15, 0], color: 0x64748b }),
        B.bakePart(railGeo, { position: [-5.77, 0.65, 0], color: 0xfbbf24 }),
        B.bakePart(railGeo, { position: [5.77, 0.65, 0], color: 0xfbbf24 }),
        B.bakePart(fanGeo, { position: [-1.6, 7.35, 0], color: 0x475569 }),
        B.bakePart(fanGeo, { position: [1.6, 7.35, 0], color: 0x475569 })
      ]);
      this.mergedTunnelSconcesGeo = B.mergeParts([
        B.bakePart(sconceGeo, { position: [-6.45, 3.85, 0], color: 0xfde047 }),
        B.bakePart(sconceGeo, { position: [6.45, 3.85, 0], color: 0xfde047 })
      ]);

      // Concrete Tunnel Portal Headwall Collar (Arched cutout R=6.85m centered at y=1.4m + retaining wing walls)
      const pPillar = new THREE.BoxGeometry(5.8, 12.2, 2.4);
      const pWing = new THREE.BoxGeometry(1.4, 9.8, 5.6);
      const pVoussoir = new THREE.BoxGeometry(1.75, 2.8, 2.4);
      const pCoping = new THREE.BoxGeometry(26.4, 0.95, 2.8);
      const pAccent = new THREE.BoxGeometry(25.8, 0.34, 2.9);
      const portalParts = [
        B.bakePart(pPillar, { position: [-9.8, 4.8, 0], color: 0x64748b }),
        B.bakePart(pPillar, { position: [9.8, 4.8, 0], color: 0x64748b }),
        B.bakePart(pWing, { position: [-12.6, 3.6, 0], rotation: [0, 0.22, 0], color: 0x475569 }),
        B.bakePart(pWing, { position: [12.6, 3.6, 0], rotation: [0, -0.22, 0], color: 0x475569 }),
        B.bakePart(pCoping, { position: [0, 10.75, 0], color: 0xe2e8f0 }),
        B.bakePart(pAccent, { position: [0, 10.12, 0], color: 0xdc2626 })
      ];
      for (let k = 0; k < 14; k++) {
        const th = ((k + 0.5) / 14) * Math.PI;
        portalParts.push(
          B.bakePart(pVoussoir, {
            position: [Math.cos(th) * 8.25, 1.4 + Math.sin(th) * 8.25, 0],
            rotation: [0, 0, th - Math.PI * 0.5],
            color: 0x64748b
          })
        );
      }
      this.mergedTunnelPortalGeo = B.mergeParts(portalParts);
      [ribGeo, walkGeo, railGeo, fanGeo, sconceGeo, pPillar, pWing, pVoussoir, pCoping, pAccent].forEach(g => g.dispose());
    }

    // 4. Merged Block Signal Static Structure (11 static parts -> 1 draw call + 3 aspect lamps!)
    {
      const foot = new THREE.BoxGeometry(0.65, 2.2, 0.65);
      const relay = new THREE.BoxGeometry(0.68, 1.35, 0.48);
      const pole = new THREE.CylinderGeometry(0.11, 0.13, 4.6, 10);
      const ladder = new THREE.BoxGeometry(0.28, 4.4, 0.06);
      const backboard = new THREE.BoxGeometry(0.66, 1.54, 0.06);
      const housing = new THREE.BoxGeometry(0.52, 1.38, 0.32);
      const visor = new THREE.CylinderGeometry(0.16, 0.16, 0.28, 10, 1, true, 0, Math.PI);
      visor.rotateX(Math.PI / 2);
      const balise = new THREE.BoxGeometry(0.42, 0.08, 0.56);

      this.mergedBlockSignalGeo = B.mergeParts([
        B.bakePart(foot, { position: [-2.6, -0.8, 0], color: 0x64748b }),
        B.bakePart(relay, { position: [-3.25, 0.42, 0.25], color: 0xf8fafc }),
        B.bakePart(pole, { position: [-2.6, 2.2, 0], color: 0x475569 }),
        B.bakePart(ladder, { position: [-2.6, 2.2, -0.20], color: 0x475569 }),
        B.bakePart(backboard, { position: [-2.6, 4.35, 0.02], color: 0xf8fafc }),
        B.bakePart(housing, { position: [-2.6, 4.35, 0.14], color: 0x090d16 }),
        B.bakePart(visor, { position: [-2.6, 4.82, 0.38], color: 0x475569 }),
        B.bakePart(visor, { position: [-2.6, 4.39, 0.38], color: 0x475569 }),
        B.bakePart(visor, { position: [-2.6, 3.96, 0.38], color: 0x475569 }),
        B.bakePart(balise, { position: [0, 0.02, 0], color: 0xfacc15 }),
        B.bakePart(balise, { position: [0, 0.02, -1.6], color: 0xfacc15 })
      ]);
      [foot, relay, pole, ladder, backboard, housing, visor, balise].forEach(g => g.dispose());
    }

    // 5. Merged Level Crossing Static Structure (Graded Asphalt Ramps + Bodan Rail Pads + Parallel Highway T-Junction!)
    {
      const trackDeck = new THREE.BoxGeometry(8.8, 0.16, 7.8);
      const rampGeo = new THREE.BoxGeometry(7.4, 0.16, 7.8);
      const villageRoad = new THREE.BoxGeometry(14.5, 0.12, 7.8);
      const hwyJunction = new THREE.BoxGeometry(13.2, 0.12, 11.4);
      const stopLine = new THREE.BoxGeometry(0.38, 0.06, 7.4);
      const centerDash = new THREE.BoxGeometry(2.6, 0.05, 0.18);
      const rumbleStrip = new THREE.BoxGeometry(0.22, 0.05, 7.2);
      const bodanGaugePad = new THREE.BoxGeometry(1.24, 0.19, 7.6);
      const bodanInterPad = new THREE.BoxGeometry(2.74, 0.19, 7.6);
      const guardHouse = new THREE.BoxGeometry(2.8, 2.5, 2.8);
      const guardRoof = new THREE.ConeGeometry(2.4, 1.1, 4);
      const guardWin = new THREE.BoxGeometry(2.86, 0.85, 1.6);
      const post = new THREE.CylinderGeometry(0.12, 0.14, 3.2, 8);
      const xBar = new THREE.BoxGeometry(0.95, 0.14, 0.06);
      // Waiting village MPV stopped behind the west barrier line (clear of tracks & parallel toll road!)
      const waitCarBody = new THREE.BoxGeometry(4.1, 1.15, 1.82);
      const waitCarCab = new THREE.BoxGeometry(2.3, 0.72, 1.64);
      const waitWheel = new THREE.CylinderGeometry(0.32, 0.32, 1.92, 10);
      waitWheel.rotateX(Math.PI / 2);

      const cxParts = [
        // Central level crossing deck + rubber Bodan rail flangeway filler pads (leaving 4 steel rail heads gleaming!)
        B.bakePart(trackDeck, { position: [0, -0.01, 0], color: 0x334155 }),
        B.bakePart(bodanGaugePad, { position: [-2.2, 0.005, 0], color: 0x1e293b }),
        B.bakePart(bodanGaugePad, { position: [2.2, 0.005, 0], color: 0x1e293b }),
        B.bakePart(bodanInterPad, { position: [0.0, 0.005, 0], color: 0x1e293b }),
        // Smooth 8.5% graded asphalt approach ramps descending from rail head (+0.08m) to ground/highway (-0.54m)
        B.bakePart(rampGeo, { position: [-8.0, -0.27, 0], rotation: [0, 0, 0.086], color: 0x1e293b }),
        B.bakePart(rampGeo, { position: [8.0, -0.27, 0], rotation: [0, 0, -0.086], color: 0x1e293b }),
        // Ground-level village road [-25..-11.6m] & flush T-junction into parallel highway [+11.6..+24.8m]
        B.bakePart(villageRoad, { position: [-18.8, -0.54, 0], color: 0x1e293b }),
        B.bakePart(hwyJunction, { position: [18.2, -0.54, 0], color: 0x1e293b }),
        B.bakePart(stopLine, { position: [-6.6, -0.14, 0], rotation: [0, 0, 0.086], color: 0xf8fafc }),
        B.bakePart(stopLine, { position: [6.6, -0.14, 0], rotation: [0, 0, -0.086], color: 0xf8fafc }),
        B.bakePart(rumbleStrip, { position: [-9.8, -0.39, 0], rotation: [0, 0, 0.086], color: 0xfacc15 }),
        B.bakePart(rumbleStrip, { position: [9.8, -0.39, 0], rotation: [0, 0, -0.086], color: 0xfacc15 }),
        B.bakePart(centerDash, { position: [-15.2, -0.49, 0], color: 0xf8fafc }),
        B.bakePart(centerDash, { position: [-21.0, -0.49, 0], color: 0xf8fafc }),
        B.bakePart(centerDash, { position: [13.4, -0.49, 0], color: 0xf8fafc }),
        // JPL Guard House (grounded at -0.54m!) & St. Andrew's Warning Posts
        B.bakePart(guardHouse, { position: [-9.2, 0.72, -6.2], color: 0xe2e8f0 }),
        B.bakePart(guardRoof, { position: [-9.2, 2.52, -6.2], rotation: [0, Math.PI / 4, 0], color: 0xc2410c }),
        B.bakePart(guardWin, { position: [-9.2, 0.96, -6.2], color: 0x38bdf8 }),
        B.bakePart(post, { position: [-6.2, 1.35, 4.1], color: 0xe2e8f0 }),
        B.bakePart(post, { position: [6.2, 1.35, -4.1], color: 0xe2e8f0 }),
        B.bakePart(xBar, { position: [-6.2, 2.75, 4.24], rotation: [0, 0, Math.PI / 4], color: 0xf8fafc }),
        B.bakePart(xBar, { position: [-6.2, 2.75, 4.25], rotation: [0, 0, -Math.PI / 4], color: 0xef4444 }),
        B.bakePart(xBar, { position: [6.2, 2.75, -4.24], rotation: [0, 0, Math.PI / 4], color: 0xf8fafc }),
        B.bakePart(xBar, { position: [6.2, 2.75, -4.25], rotation: [0, 0, -Math.PI / 4], color: 0xef4444 }),
        // Waiting MPV on village approach lane (Indonesian left-hand lane z = +1.85m facing +X)
        B.bakePart(waitCarBody, { position: [-14.2, 0.22, 1.85], color: 0xf8fafc }),
        B.bakePart(waitCarCab, { position: [-14.4, 1.05, 1.85], color: 0x38bdf8 }),
        B.bakePart(waitWheel, { position: [-12.9, -0.18, 1.85], color: 0x0f172a }),
        B.bakePart(waitWheel, { position: [-15.5, -0.18, 1.85], color: 0x0f172a })
      ];
      this.mergedCrossingStaticGeo = B.mergeParts(cxParts);

      const bar = new THREE.BoxGeometry(5.6, 0.15, 0.1);
      const stripe = new THREE.BoxGeometry(1.1, 0.16, 0.11);
      this.mergedBoomLeftGeo = B.mergeParts([
        B.bakePart(bar, { position: [2.6, 0, 0], color: 0xef4444 }),
        B.bakePart(stripe, { position: [1.6, 0, 0], color: 0xf8fafc }),
        B.bakePart(stripe, { position: [3.8, 0, 0], color: 0xf8fafc })
      ]);
      this.mergedBoomRightGeo = B.mergeParts([
        B.bakePart(bar, { position: [-2.6, 0, 0], color: 0xef4444 }),
        B.bakePart(stripe, { position: [-1.6, 0, 0], color: 0xf8fafc }),
        B.bakePart(stripe, { position: [-3.8, 0, 0], color: 0xf8fafc })
      ]);

      [
        trackDeck, rampGeo, villageRoad, hwyJunction, stopLine, centerDash, rumbleStrip,
        bodanGaugePad, bodanInterPad, guardHouse, guardRoof, guardWin, post, xBar,
        waitCarBody, waitCarCab, waitWheel, bar, stripe
      ].forEach(g => g.dispose());
    }

    // 6. Merged Wind Turbine Static Tower+Nacelle & 1-Draw-Call 3-Blade Rotor
    {
      const tower = new THREE.CylinderGeometry(0.48, 1.22, 32, 12);
      const nacelle = new THREE.BoxGeometry(1.65, 1.75, 3.8);
      this.mergedTurbineStaticGeo = B.mergeParts([
        B.bakePart(tower, { position: [0, 15.0, 0], color: 0xf8fafc }),
        B.bakePart(nacelle, { position: [0, 30.8, -0.4], color: 0xf8fafc })
      ]);

      const hub = new THREE.ConeGeometry(0.82, 1.65, 12);
      hub.rotateX(Math.PI / 2);
      const blade = new THREE.BoxGeometry(0.48, 14.5, 0.16);
      blade.translate(0, 7.2, 0);

      const rotorParts = [B.bakePart(hub, { position: [0, 0, 0.45], color: 0xf8fafc })];
      for (let b = 0; b < 3; b++) {
        const bClone = blade.clone();
        bClone.rotateY(0.18);
        bClone.rotateZ((b * Math.PI * 2) / 3);
        rotorParts.push(B.bakePart(bClone, { color: 0xf8fafc }));
        bClone.dispose();
      }
      this.mergedTurbineRotorGeo = B.mergeParts(rotorParts);
      [tower, nacelle, hub, blade].forEach(g => g.dispose());
    }

    // 7. Pre-Baked Mega-Station Static Architecture (~215 meshes -> 3 shared GPU meshes + textured/animated parts!)
    this._buildSharedStationGeometries();
  }

  _buildSharedStationGeometries() {
    const B = GeometryBatcher;
    const platLen = 276;
    const platCenterZ = 52;

    this.sharedPlatBoxGeo = new THREE.BoxGeometry(6.8, 1.15, platLen);
    this.sharedPlatBoxGeo.userData.sharedGeo = true;
    this.sharedTerminalBoxGeo = new THREE.BoxGeometry(10.5, 11.5, 128.0);
    this.sharedTerminalBoxGeo.userData.sharedGeo = true;
    this.sharedSignBoardGeo = new THREE.BoxGeometry(2.6, 0.68, 0.08);
    this.sharedSignBoardGeo.userData.sharedGeo = true;
    this.sharedPsdDoorPanelGeo = new THREE.BoxGeometry(0.05, 2.00, 1.05);
    this.sharedPsdDoorPanelGeo.userData.sharedGeo = true;

    const opaqueParts = [];
    const glassParts = [];
    const emissiveParts = [];

    const addBox = (target, size, pos, col, rot = null) => {
      const g = new THREE.BoxGeometry(size[0], size[1], size[2]);
      target.push(B.bakePart(g, { position: pos, rotation: rot, color: col }));
      g.dispose();
    };
    const addCyl = (target, args, pos, col, rot = null, scl = null) => {
      const g = new THREE.CylinderGeometry(...args);
      target.push(B.bakePart(g, { position: pos, rotation: rot, scale: scl, color: col }));
      g.dispose();
    };

    // Under-platform lips, main canopies & red fascias (covering full 276m 10-car consist!)
    addBox(opaqueParts, [0.25, 0.75, platLen], [-1.90, 0.20, platCenterZ], 0x0f172a);
    addBox(opaqueParts, [0.25, 0.75, platLen], [6.30, 0.20, platCenterZ], 0x0f172a);
    addBox(opaqueParts, [7.4, 0.28, platLen - 4], [-5.35, 4.85, platCenterZ], 0x0f172a);
    addBox(opaqueParts, [7.4, 0.28, platLen - 4], [9.75, 4.85, platCenterZ], 0x0f172a);
    addBox(opaqueParts, [0.18, 0.36, platLen - 4], [-1.68, 4.88, platCenterZ], 0xdc2626);
    addBox(opaqueParts, [0.18, 0.36, platLen - 4], [6.08, 4.88, platCenterZ], 0xdc2626);
    addBox(glassParts, [19.6, 0.18, platLen - 16], [2.2, 8.95, platCenterZ], null);

    // 18 bays of columns, Mega Mendung wave arches, canopy beams & benches along [-75 .. +180]
    const waveArchBase = new THREE.CylinderGeometry(9.6, 9.6, 0.42, 24, 1, true, -Math.PI * 0.48, Math.PI * 0.96);
    waveArchBase.rotateX(Math.PI / 2);

    for (let z = -75; z <= 180; z += 15) {
      addCyl(opaqueParts, [0.20, 0.24, 4.0, 12], [-5.35, 2.9, z], 0x475569);
      addCyl(opaqueParts, [0.20, 0.24, 4.0, 12], [9.75, 2.9, z], 0x475569);
      opaqueParts.push(B.bakePart(waveArchBase, { position: [2.2, 1.4, z], scale: [1.0, 0.82, 1.35], color: 0xe2e8f0 }));
      addBox(opaqueParts, [6.6, 0.26, 0.32], [-5.35, 4.65, z], 0x475569);
      addBox(opaqueParts, [6.6, 0.26, 0.32], [9.75, 4.65, z], 0x475569);

      if (z <= 165) {
        addBox(opaqueParts, [0.55, 0.08, 2.2], [-5.8, 1.32, z + 5], 0x0284c7);
        addBox(opaqueParts, [0.08, 0.45, 2.2], [-6.02, 1.56, z + 5], 0x0284c7);
        addBox(opaqueParts, [0.55, 0.08, 2.2], [10.2, 1.32, z + 5], 0x0284c7);
        addBox(opaqueParts, [0.08, 0.45, 2.2], [10.42, 1.56, z + 5], 0x0284c7);
      }
    }
    waveArchBase.dispose();

    // Fixed PSD pillars, continuous header beams & non-overlapping fixed glass barrier walls aligned 1:1 with stopped Coaches 1..8!
    const doorZList = [-34.2, -18.6, -12.6, 3.0, 9.0, 24.6, 30.6, 46.2, 52.2, 67.8, 73.8, 89.4, 95.4, 111.0, 117.0];
    [-2.08, 6.48].forEach((px) => {
      addBox(opaqueParts, [0.18, 0.18, 165.0], [px, 3.16, 41.4], 0xdc2626);
      doorZList.forEach((dz) => {
        addBox(opaqueParts, [0.14, 2.25, 0.24], [px, 2.10, dz - 1.20], 0x1e293b);
        addBox(opaqueParts, [0.14, 2.25, 0.24], [px, 2.10, dz + 1.20], 0x1e293b);
      });
      [-26.4, -4.8, 16.8, 38.4, 60.0, 81.6, 103.2].forEach((midCarZ) => {
        addBox(glassParts, [0.06, 2.05, 13.0], [px, 2.02, midCarZ], null);
      });
      [-15.6, 6.0, 27.6, 49.2, 70.8, 92.4, 114.0].forEach((gangwayZ) => {
        addBox(glassParts, [0.06, 2.05, 3.36], [px, 2.02, gangwayZ], null);
      });
      addBox(glassParts, [0.06, 2.05, 5.4], [px, 2.02, -38.2], null);
      addBox(glassParts, [0.06, 2.05, 5.4], [px, 2.02, 121.0], null);
    });

    // Terminal wing roof, entrance canopy, escalators, kiosks, PIDS, TVMs & skybridge
    addBox(opaqueParts, [13.2, 0.65, 136.0], [-13.8, 11.45, 22], 0xe2e8f0);
    addBox(opaqueParts, [4.5, 0.45, 36.0], [-20.2, 4.8, 22], 0xdc2626);

    [-5.6, 10.0].forEach((ex) => {
      addBox(opaqueParts, [1.65, 0.45, 9.8], [ex, 4.3, 16.8], 0x475569, [0.68, 0, 0]);
      addBox(glassParts, [1.72, 0.85, 9.6], [ex, 4.85, 16.8], null, [0.68, 0, 0]);
    });

    [-28, 36, 86].forEach((kz) => {
      addBox(opaqueParts, [2.4, 2.6, 4.8], [-7.1, 2.25, kz], 0x0f172a);
      addBox(emissiveParts, [0.12, 0.48, 4.4], [-5.88, 3.2, kz], 0xf59e0b);
      addBox(emissiveParts, [0.10, 1.4, 3.8], [-5.89, 2.1, kz], 0xe0f2fe);
    });

    [-46, -10, 26, 62, 98].forEach((pz) => {
      addBox(opaqueParts, [0.22, 0.48, 2.1], [-3.4, 3.95, pz], 0x090d16);
      addBox(emissiveParts, [0.24, 0.32, 1.9], [-3.4, 3.95, pz], 0xf59e0b);
      addBox(opaqueParts, [0.75, 1.85, 0.85], [-7.4, 1.88, pz - 4], 0x0284c7);
    });

    addBox(opaqueParts, [24.5, 0.55, 8.4], [-1.2, 7.6, 24], 0x64748b);
    addBox(glassParts, [24.2, 2.8, 8.0], [-1.2, 9.25, 24], null);
    addBox(opaqueParts, [25.2, 0.42, 8.8], [-1.2, 10.8, 24], 0x0f172a);
    addBox(opaqueParts, [3.2, 7.8, 6.6], [-6.4, 4.7, 24], 0x64748b);
    addBox(opaqueParts, [3.2, 7.8, 6.6], [10.6, 4.7, 24], 0x64748b);

    // Siding track, hydraulic buffer stop & KPJ Plasser & Theurer maintenance vehicle
    addBox(opaqueParts, [3.4, 0.45, platLen + 28], [14.5, -0.25, platCenterZ], 0x64748b);
    addBox(opaqueParts, [0.09, 0.14, platLen + 24], [14.5 - 0.717, 0.05, platCenterZ], 0x475569);
    addBox(opaqueParts, [0.09, 0.14, platLen + 24], [14.5 + 0.717, 0.05, platCenterZ], 0x475569);
    addBox(opaqueParts, [1.9, 1.15, 1.2], [14.5, 0.58, platCenterZ - (platLen * 0.5 + 10)], 0xdc2626);
    addBox(emissiveParts, [2.1, 0.35, 0.35], [14.5, 0.92, platCenterZ - (platLen * 0.5 + 9.3)], 0xfacc15);

    const mx = 14.5, mz0 = -24;
    addBox(opaqueParts, [2.68, 0.38, 15.4], [mx, 0.95, mz0], 0x475569);
    addBox(opaqueParts, [2.65, 2.15, 14.6], [mx, 2.18, mz0], 0xf59e0b);
    addBox(opaqueParts, [2.70, 0.28, 14.66], [mx, 1.55, mz0], 0xdc2626);
    addBox(opaqueParts, [2.72, 0.72, 13.8], [mx, 2.58, mz0], 0x0f172a);
    [-7.32, 7.32].forEach((mz) => {
      addBox(glassParts, [2.35, 0.72, 0.12], [mx, 2.58, mz0 + mz], null);
      addBox(opaqueParts, [2.55, 0.42, 0.35], [mx, 0.62, mz0 + mz], 0xdc2626);
      emissiveParts.push(B.bakePart(this.sharedSigLampGeo, { position: [mx, 3.42, mz0 + mz * 0.88], color: 0xfacc15 }));
    });
    addBox(opaqueParts, [2.2, 0.12, 8.6], [mx, 3.32, mz0], 0x475569);
    addBox(emissiveParts, [0.06, 0.65, 8.4], [mx - 1.05, 3.68, mz0], 0xfacc15);
    addBox(emissiveParts, [0.06, 0.65, 8.4], [mx + 1.05, 3.68, mz0], 0xfacc15);
    [-4.8, 4.8].forEach((bz) => {
      addBox(opaqueParts, [2.2, 0.32, 2.4], [mx, 0.55, mz0 + bz], 0x0f172a);
      [-0.85, 0.85].forEach((az) => {
        addCyl(opaqueParts, [0.42, 0.42, 1.65, 12], [mx, 0.42, mz0 + bz + az], 0x475569, [0, 0, Math.PI / 2]);
      });
    });

    // Platform LED light strip & precision green stop target positioned at z = -61.5m (directly visible ahead of Cab 3D windshield when stopped at 0.00m!)
    addBox(emissiveParts, [0.35, 0.08, platLen - 10], [-3.9, 4.52, platCenterZ], 0xfef08a);
    addCyl(opaqueParts, [0.055, 0.065, 1.35, 8], [-2.35, 1.48, -61.5], 0x475569);
    addBox(emissiveParts, [0.95, 0.95, 0.14], [-2.35, 2.28, -61.5], 0x10b981, [0, 0.18, Math.PI / 4]);
    addBox(emissiveParts, [3.8, 0.09, 0.48], [0, 0.08, -60.2], 0x34d399);

    this.mergedStationOpaqueGeo = B.mergeParts(opaqueParts);
    this.mergedStationGlassGeo = B.mergeParts(glassParts);
    this.mergedStationEmissiveGeo = B.mergeParts(emissiveParts);
  }

  _initSharedNPCAssets() {
    const B = GeometryBatcher;
    const raw = {
      pelvis: new THREE.CylinderGeometry(0.148, 0.138, 0.18, 14),
      waist: new THREE.CylinderGeometry(0.138, 0.148, 0.16, 14),
      torso: new THREE.CylinderGeometry(0.182, 0.140, 0.38, 16),
      chestSculpt: new THREE.SphereGeometry(0.145, 12, 10),
      lapel: new THREE.BoxGeometry(0.065, 0.28, 0.035),
      idBadge: new THREE.BoxGeometry(0.052, 0.072, 0.012),
      shoulderClavicle: new THREE.CylinderGeometry(0.064, 0.064, 0.44, 12),
      shoulderDeltoid: new THREE.SphereGeometry(0.068, 10, 8),
      collar: new THREE.CylinderGeometry(0.082, 0.105, 0.075, 12),
      neck: new THREE.CylinderGeometry(0.052, 0.060, 0.11, 12),
      head: new THREE.SphereGeometry(0.126, 20, 16),
      nose3D: new THREE.ConeGeometry(0.022, 0.052, 8),
      ear: new THREE.SphereGeometry(0.024, 8, 8),
      glassesBridge: new THREE.BoxGeometry(0.15, 0.012, 0.015),
      glassesRim: new THREE.BoxGeometry(0.058, 0.038, 0.016),
      hairCrown: new THREE.SphereGeometry(0.133, 16, 12, 0, Math.PI * 2, 0, Math.PI * 0.56),
      hairBackLong: new THREE.CylinderGeometry(0.118, 0.132, 0.26, 12, 1, false, Math.PI * 0.55, Math.PI * 0.90),
      hairSideFringe: new THREE.BoxGeometry(0.22, 0.055, 0.075),
      hairBun: new THREE.SphereGeometry(0.064, 10, 10),
      hijabHood: new THREE.SphereGeometry(0.135, 16, 14, Math.PI * 0.78, Math.PI * 1.44, 0, Math.PI * 0.86),
      hijabDrape: new THREE.CylinderGeometry(0.105, 0.175, 0.18, 14),
      capCrown: new THREE.CylinderGeometry(0.132, 0.132, 0.088, 14),
      capVisor: new THREE.BoxGeometry(0.175, 0.016, 0.135),
      skirt: new THREE.CylinderGeometry(0.142, 0.188, 0.38, 14),
      thigh: new THREE.CylinderGeometry(0.074, 0.058, 0.36, 12),
      knee: new THREE.SphereGeometry(0.056, 10, 8),
      calf: new THREE.CylinderGeometry(0.056, 0.042, 0.35, 12),
      shoe: new THREE.BoxGeometry(0.098, 0.072, 0.225),
      shoeToe: new THREE.SphereGeometry(0.050, 10, 8),
      sole: new THREE.BoxGeometry(0.104, 0.024, 0.235),
      upperArm: new THREE.CylinderGeometry(0.052, 0.044, 0.27, 12),
      elbow: new THREE.SphereGeometry(0.043, 8, 8),
      forearm: new THREE.CylinderGeometry(0.042, 0.034, 0.25, 12),
      wristwatch: new THREE.CylinderGeometry(0.038, 0.038, 0.024, 10),
      hand: new THREE.BoxGeometry(0.038, 0.068, 0.056),
      thumb: new THREE.BoxGeometry(0.018, 0.034, 0.024),
      phone: new THREE.BoxGeometry(0.046, 0.090, 0.009),
      phoneScreen: new THREE.PlaneGeometry(0.040, 0.080),
      coffeeCup: new THREE.CylinderGeometry(0.036, 0.026, 0.092, 10),
      briefcase: new THREE.BoxGeometry(0.075, 0.27, 0.36),
      backpack: new THREE.BoxGeometry(0.25, 0.34, 0.13),
      backpackPocket: new THREE.BoxGeometry(0.20, 0.16, 0.045),
      suitcaseBody: new THREE.BoxGeometry(0.22, 0.46, 0.32),
      suitcaseRod: new THREE.CylinderGeometry(0.011, 0.011, 0.42, 8),
      suitcaseGrip: new THREE.BoxGeometry(0.14, 0.024, 0.028),
      suitcaseWheel: new THREE.CylinderGeometry(0.036, 0.036, 0.028, 10)
    };

    raw.head.rotateY(-Math.PI * 0.5);
    raw.head.scale(0.92, 1.08, 0.98);
    raw.head.translate(0, 0.145, 0);
    raw.head.userData.sharedGeo = true;

    raw.pelvis.userData.sharedGeo = true;
    raw.skirt.translate(0, -0.12, 0);
    raw.skirt.userData.sharedGeo = true;

    raw.thigh.translate(0, -0.18, 0);
    raw.thigh.userData.sharedGeo = true;

    raw.elbow.userData.sharedGeo = true;
    raw.wristwatch.translate(0, -0.21, 0);
    raw.wristwatch.userData.sharedGeo = true;
    raw.briefcase.translate(0, -0.40, 0.02);
    raw.briefcase.userData.sharedGeo = true;
    raw.coffeeCup.translate(0, -0.27, 0.045);
    raw.coffeeCup.userData.sharedGeo = true;
    raw.suitcaseBody.userData.sharedGeo = true;

    raw.chestSculpt.scale(1.18, 0.82, 0.68);
    raw.nose3D.rotateX(Math.PI * 0.38);
    raw.ear.scale(0.55, 1.15, 0.75);
    raw.shoeToe.scale(0.96, 0.68, 1.25);
    raw.shoulderClavicle.rotateZ(Math.PI * 0.5);
    raw.calf.translate(0, -0.175, 0);
    raw.upperArm.translate(0, -0.135, 0);
    raw.forearm.translate(0, -0.125, 0);
    raw.suitcaseWheel.rotateZ(Math.PI / 2);

    // Bone-Level Merged Sub-Assemblies (Cuts ~28 redundant draw calls per NPC!)
    this.sharedPaxGeos = {
      pelvis: raw.pelvis,
      skirt: raw.skirt,
      head: raw.head,
      thigh: raw.thigh,
      elbow: raw.elbow,
      wristwatch: raw.wristwatch,
      briefcase: raw.briefcase,
      coffeeCup: raw.coffeeCup,
      suitcaseBody: raw.suitcaseBody,

      // UV-preserved merged torso + waist + chest
      torsoFull: B.mergeParts([
        B.bakePart(raw.waist, { position: [0, 0.14, 0], keepUV: true }),
        B.bakePart(raw.torso, { position: [0, 0.35, 0], keepUV: true }),
        B.bakePart(raw.chestSculpt, { position: [0, 0.39, 0.055], keepUV: true })
      ]),
      // Merged lapels + clavicle
      lapelsClavicle: B.mergeParts([
        B.bakePart(raw.shoulderClavicle, { position: [0, 0.51, 0] }),
        B.bakePart(raw.lapel, { position: [-0.055, 0.42, 0.142], rotation: [0, 0, -0.18] }),
        B.bakePart(raw.lapel, { position: [0.055, 0.42, 0.142], rotation: [0, 0, 0.18] })
      ]),
      // Merged collar + ID badge
      collarBadge: B.mergeParts([
        B.bakePart(raw.collar, { position: [0, 0.555, 0] }),
        B.bakePart(raw.idBadge, { position: [0.065, 0.34, 0.155] })
      ]),
      // Merged neck + 3D nose + left/right ears
      neckNoseEars: B.mergeParts([
        B.bakePart(raw.neck, { position: [0, 0.03, 0] }),
        B.bakePart(raw.nose3D, { position: [0, 0.142, 0.128] }),
        B.bakePart(raw.ear, { position: [-0.118, 0.142, -0.005] }),
        B.bakePart(raw.ear, { position: [0.118, 0.142, -0.005] })
      ]),
      // Merged eyeglasses
      glassesFull: B.mergeParts([
        B.bakePart(raw.glassesBridge, { position: [0, 0.162, 0.124] }),
        B.bakePart(raw.glassesRim, { position: [-0.044, 0.160, 0.125] }),
        B.bakePart(raw.glassesRim, { position: [0.044, 0.160, 0.125] })
      ]),
      // Merged hairstyles, hijab & officer cap
      hairShort: B.mergeParts([
        B.bakePart(raw.hairCrown, { position: [0, 0.156, -0.006], rotation: [-0.22, 0, 0] }),
        B.bakePart(raw.hairSideFringe, { position: [0, 0.238, 0.072], rotation: [0.15, 0, 0] })
      ]),
      hairFemale: B.mergeParts([
        B.bakePart(raw.hairCrown, { position: [0, 0.156, -0.006], rotation: [-0.22, 0, 0] }),
        B.bakePart(raw.hairSideFringe, { position: [0, 0.238, 0.072], rotation: [0.15, 0, 0] }),
        B.bakePart(raw.hairBackLong, { position: [0, 0.06, -0.02] }),
        B.bakePart(raw.hairBun, { position: [0, 0.155, -0.138] })
      ]),
      hijabFull: B.mergeParts([
        B.bakePart(raw.hijabHood, { position: [0, 0.148, -0.008] }),
        B.bakePart(raw.hijabDrape, { position: [0, 0.025, -0.018] })
      ]),
      officerCapFull: B.mergeParts([
        B.bakePart(raw.capCrown, { position: [0, 0.245, 0], color: 0x0f172a }),
        B.bakePart(raw.capVisor, { position: [0, 0.212, 0.105], rotation: [0.16, 0, 0], color: 0x111827 }),
        B.bakePart(raw.idBadge, { position: [0, 0.248, 0.134], scale: 0.45, color: 0xfacc15 })
      ]),
      // Merged lower leg & shoe
      kneeCalf: B.mergeParts([
        B.bakePart(raw.knee),
        B.bakePart(raw.calf)
      ]),
      shoeUpper: B.mergeParts([
        B.bakePart(raw.shoe, { position: [0, -0.35, 0.036] }),
        B.bakePart(raw.shoeToe, { position: [0, -0.356, 0.132] })
      ]),
      soleOffset: B.mergeParts([
        B.bakePart(raw.sole, { position: [0, -0.382, 0.040] })
      ]),
      // Merged upper arm + deltoid & forearm + hand + thumb
      upperArmFull: B.mergeParts([
        B.bakePart(raw.shoulderDeltoid),
        B.bakePart(raw.upperArm)
      ]),
      forearmHandL: B.mergeParts([
        B.bakePart(raw.forearm),
        B.bakePart(raw.hand, { position: [0, -0.26, 0.01] }),
        B.bakePart(raw.thumb, { position: [0.022, -0.245, 0.034] })
      ]),
      forearmHandR: B.mergeParts([
        B.bakePart(raw.forearm),
        B.bakePart(raw.hand, { position: [0, -0.26, 0.01] }),
        B.bakePart(raw.thumb, { position: [-0.022, -0.245, 0.034] })
      ]),
      // Merged backpack, phone & suitcase hardware
      backpackFull: B.mergeParts([
        B.bakePart(raw.backpack, { position: [0, 0.36, -0.145] }),
        B.bakePart(raw.backpackPocket, { position: [0, 0.30, -0.215] })
      ]),
      phoneFull: B.mergeParts([
        B.bakePart(raw.phone, { position: [0, -0.28, 0.048], rotation: [-0.75, 0, 0], color: 0x090d16 }),
        B.bakePart(raw.phoneScreen, { position: [0, -0.278, 0.054], rotation: [-0.75, 0, 0], color: 0x7dd3fc })
      ]),
      suitcaseHardware: B.mergeParts([
        B.bakePart(raw.suitcaseRod, { position: [-0.055, 0.36, -0.10], color: 0xe2e8f0 }),
        B.bakePart(raw.suitcaseRod, { position: [0.055, 0.36, -0.10], color: 0xe2e8f0 }),
        B.bakePart(raw.suitcaseGrip, { position: [0, 0.56, -0.10], color: 0x111827 }),
        B.bakePart(raw.suitcaseWheel, { position: [0, -0.24, 0.11], color: 0x111827 }),
        B.bakePart(raw.suitcaseWheel, { position: [0, -0.24, -0.11], color: 0x111827 })
      ])
    };

    // Dispose temporary raw geometries that were consumed into merged parts
    [
      raw.waist, raw.torso, raw.chestSculpt, raw.lapel, raw.idBadge, raw.shoulderClavicle,
      raw.shoulderDeltoid, raw.collar, raw.neck, raw.nose3D, raw.ear, raw.glassesBridge,
      raw.glassesRim, raw.hairCrown, raw.hairBackLong, raw.hairSideFringe, raw.hairBun,
      raw.hijabHood, raw.hijabDrape, raw.capCrown, raw.capVisor, raw.knee, raw.calf,
      raw.shoe, raw.shoeToe, raw.sole, raw.upperArm, raw.forearm, raw.hand, raw.thumb,
      raw.phone, raw.phoneScreen, raw.backpack, raw.backpackPocket, raw.suitcaseRod,
      raw.suitcaseGrip, raw.suitcaseWheel
    ].forEach(g => g.dispose());

    this.sharedPaxMats = {
      faces: [0, 1, 2, 3, 4, 5].map(i => new THREE.MeshStandardMaterial({
        map: ProceduralAssetFactory.createNPCFaceTexture(i),
        roughness: 0.52,
        metalness: 0.04
      })),
      skins: ['#f5d0b5', '#eab68c', '#d99b72', '#f8d9c0', '#c88a62', '#e5b895'].map(c => new THREE.MeshStandardMaterial({
        color: c,
        roughness: 0.52
      })),
      torsos: [
        ['#1e3a8a', '#dc2626', 0],
        ['#0f172a', '#0284c7', 0],
        ['#be123c', '#f8fafc', 1],
        ['#047857', '#facc15', 1],
        ['#334155', '#38bdf8', 0],
        ['#7c2d12', '#fde047', 1],
        ['#4338ca', '#f43f5e', 1],
        ['#1e293b', '#eab308', 2]
      ].map(([p, s, st]) => new THREE.MeshStandardMaterial({
        map: ProceduralAssetFactory.createNPCTorsoTexture(p, s, st),
        roughness: 0.58
      })),
      sleeves: [0x1e3a8a, 0x0f172a, 0xbe123c, 0x047857, 0x334155, 0x7c2d12, 0x4338ca, 0x1e293b].map(
        c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.60 })
      ),
      pants: [0x0f172a, 0x1e293b, 0x334155, 0x1e3a5f, 0x292524, 0x475569].map(
        c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.68 })
      ),
      hairs: [0x111827, 0x271c19, 0x3f2e21, 0x090d16, 0x4a3525].map(
        c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.75 })
      ),
      hijabs: [0x991b1b, 0x0f766e, 0x1e40af, 0xd97706, 0x6d28d9, 0xbe185d].map(
        c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.58, side: THREE.DoubleSide })
      ),
      shoeDark: new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.38, metalness: 0.25 }),
      shoeBrown: new THREE.MeshStandardMaterial({ color: 0x451a03, roughness: 0.42, metalness: 0.18 }),
      soleWhite: new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.55 }),
      collarWhite: new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.45 }),
      bagMats: [0x334155, 0x7c2d12, 0x0f172a, 0x1e3a8a].map(
        c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.52, metalness: 0.22 })
      ),
      luggageMats: [0xe11d48, 0x0284c7, 0x475569, 0xd97706, 0x059669].map(
        c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.22, metalness: 0.68, envMapIntensity: 1.3 })
      ),
      gold: new THREE.MeshStandardMaterial({ color: 0xfacc15, metalness: 0.85, roughness: 0.20 })
    };
  }

  _initWorker() {
    this.worker = null;
    try {
      this.worker = new Worker(new URL('../workers/ChunkBuilderWorker.js', import.meta.url), {
        type: 'module'
      });
      this.worker.onmessage = (e) => {
        const msg = e.data;
        if (!msg) return;
        if (msg.type === 'CHUNK_READY' && msg.data) {
          this.pendingChunks.delete(msg.data.chunkIdx);
          if (!this.chunks.has(msg.data.chunkIdx)) {
            this.completedQueue.push(msg.data);
          }
        } else if (msg.type === 'CHUNK_ERROR') {
          this.pendingChunks.delete(msg.chunkIdx);
          console.warn('[StreamingWorld] Worker chunk fallback:', msg.error);
        }
      };
      this.worker.onerror = (err) => {
        console.warn('[StreamingWorld] Worker runtime fallback to main thread:', err);
        this.worker = null;
        this.pendingChunks.clear();
      };
    } catch (err) {
      console.warn('[StreamingWorld] Worker unavailable, using synchronous chunk core:', err);
      this.worker = null;
    }
  }

  /**
   * Synchronously populates initial spawn chunks around Stasiun Halim so Frame 1
   * renders immediately with zero pop-in.
   */
  warmupInitialChunks(initialS, originS) {
    const chunkLen = GAME_PREMISE_CARTRIDGE.railway.chunkLength;
    const curIdx = Math.floor(initialS / chunkLen);
    for (let idx = curIdx - 2; idx <= curIdx + 5; idx++) {
      if (!this.chunks.has(idx)) {
        const data = ChunkGeometryCore.buildChunkBuffers(idx);
        const grp = this.assembleChunkFromBuffers(data, originS);
        this.chunks.set(idx, grp);
        this.scene.add(grp);
      }
    }
  }

  /**
   * Shifts all active chunk root groups in O(1) time when Floating-Origin rebases!
   */
  applyOriginRebase(originS) {
    for (const group of this.chunks.values()) {
      group.position.z = -(group.userData.sStart - originS);
    }
  }

  setEnvMap(envMap) {
    this.envMap = envMap;
  }

  createDetailedHumanNPC(variantIdx = 0, role = 'BOARDING') {
    const PG = this.sharedPaxGeos;
    const PM = this.sharedPaxMats;
    const isOfficer = (role === 'OFFICER');
    const isSeated = (role === 'SEATED_BENCH' || role === 'SEATED_COACH');
    const isFemale = (!isOfficer && (variantIdx % 3 === 1 || variantIdx % 5 === 2));

    const skinMat = PM.skins[variantIdx % PM.skins.length];
    const faceMat = PM.faces[variantIdx % PM.faces.length];
    const torsoIdx = isOfficer ? 7 : (variantIdx % 7);
    const torsoMat = PM.torsos[torsoIdx];
    const sleeveMat = PM.sleeves[torsoIdx];
    const pantMat = PM.pants[variantIdx % PM.pants.length];
    const hairMat = PM.hairs[variantIdx % PM.hairs.length];
    const shoeMat = (variantIdx % 3 === 0) ? PM.shoeBrown : PM.shoeDark;

    const root = new THREE.Group();
    const heightScale = 0.95 + (variantIdx % 5) * 0.022;
    root.scale.setScalar(heightScale);

    // 1. Upper Body Hierarchy
    const upperBody = new THREE.Group();
    upperBody.position.y = isSeated ? 0.52 : 0.78;
    root.add(upperBody);

    const pelvis = new THREE.Mesh(PG.pelvis, pantMat);
    pelvis.userData.sharedGeo = true;
    const torsoFull = new THREE.Mesh(PG.torsoFull, torsoMat);
    const lapelsClavicle = new THREE.Mesh(PG.lapelsClavicle, sleeveMat);
    const collarBadge = new THREE.Mesh(PG.collarBadge, PM.collarWhite);
    torsoFull.castShadow = true;
    upperBody.add(pelvis, torsoFull, lapelsClavicle, collarBadge);

    if (isFemale && !isSeated && variantIdx % 4 === 1) {
      const skirt = new THREE.Mesh(PG.skirt, pantMat);
      skirt.userData.sharedGeo = true;
      upperBody.add(skirt);
    }

    // Head & Neck Pivot Group
    const headGroup = new THREE.Group();
    headGroup.position.set(0, 0.60, 0);
    upperBody.add(headGroup);

    const head = new THREE.Mesh(PG.head, faceMat);
    head.userData.sharedGeo = true;
    const neckNoseEars = new THREE.Mesh(PG.neckNoseEars, skinMat);
    headGroup.add(head, neckNoseEars);

    if (variantIdx % 3 === 1) {
      const gMat = (variantIdx % 2 === 0) ? PM.shoeDark : PM.gold;
      headGroup.add(new THREE.Mesh(PG.glassesFull, gMat));
    }

    if (isOfficer) {
      headGroup.add(new THREE.Mesh(PG.officerCapFull, this.infraPbrMat));
    } else if (variantIdx % 5 === 2) {
      const hMat = PM.hijabs[variantIdx % PM.hijabs.length];
      headGroup.add(new THREE.Mesh(PG.hijabFull, hMat));
    } else if (isFemale) {
      headGroup.add(new THREE.Mesh(PG.hairFemale, hairMat));
    } else {
      headGroup.add(new THREE.Mesh(PG.hairShort, hairMat));
    }

    // 2. Articulated Two-Segment Left & Right Legs
    const buildLeg = (sideX) => {
      const hipPivot = new THREE.Group();
      hipPivot.position.set(sideX, isSeated ? 0.52 : 0.76, 0);

      const thigh = new THREE.Mesh(PG.thigh, pantMat);
      thigh.userData.sharedGeo = true;
      hipPivot.add(thigh);

      const kneePivot = new THREE.Group();
      kneePivot.position.set(0, -0.36, 0);
      hipPivot.add(kneePivot);

      const kneeCalf = new THREE.Mesh(PG.kneeCalf, pantMat);
      const shoeUpper = new THREE.Mesh(PG.shoeUpper, shoeMat);
      const sole = new THREE.Mesh(PG.soleOffset, PM.soleWhite);
      kneePivot.add(kneeCalf, shoeUpper, sole);

      root.add(hipPivot);
      return { hipPivot, kneePivot };
    };

    const legL = buildLeg(-0.086);
    const legR = buildLeg(0.086);

    if (isSeated) {
      legL.hipPivot.rotation.x = -Math.PI * 0.48;
      legR.hipPivot.rotation.x = -Math.PI * 0.48;
      legL.kneePivot.rotation.x = Math.PI * 0.48;
      legR.kneePivot.rotation.x = Math.PI * 0.48;
    }

    // 3. Articulated Two-Segment Left & Right Arms
    const buildArm = (sideX, isLeft) => {
      const shoulderPivot = new THREE.Group();
      shoulderPivot.position.set(sideX, 0.50, 0);
      shoulderPivot.rotation.z = isLeft ? 0.07 : -0.07;
      upperBody.add(shoulderPivot);

      const upperArm = new THREE.Mesh(PG.upperArmFull, sleeveMat);
      shoulderPivot.add(upperArm);

      const elbowPivot = new THREE.Group();
      elbowPivot.position.set(0, -0.26, 0);
      shoulderPivot.add(elbowPivot);

      const elbowJoint = new THREE.Mesh(PG.elbow, sleeveMat);
      elbowJoint.userData.sharedGeo = true;
      const forearmHand = new THREE.Mesh(isLeft ? PG.forearmHandL : PG.forearmHandR, skinMat);
      elbowPivot.add(elbowJoint, forearmHand);

      if (isLeft) {
        const watch = new THREE.Mesh(PG.wristwatch, PM.gold);
        watch.userData.sharedGeo = true;
        elbowPivot.add(watch);
      }
      return { shoulderPivot, elbowPivot };
    };

    const armL = buildArm(-0.242, true);
    const armR = buildArm(0.242, false);

    // 4. Accessories
    let suitcase = null;
    let hasPhone = false;
    const bagMat = PM.bagMats[variantIdx % PM.bagMats.length];
    const lugMat = PM.luggageMats[variantIdx % PM.luggageMats.length];

    if (!isOfficer && !isSeated) {
      if (variantIdx % 3 === 0) {
        upperBody.add(new THREE.Mesh(PG.backpackFull, bagMat));
      } else if (variantIdx % 4 === 2) {
        const bcase = new THREE.Mesh(PG.briefcase, bagMat);
        bcase.userData.sharedGeo = true;
        armL.elbowPivot.add(bcase);
      }

      if ((role === 'BOARDING' || role === 'ALIGHTING' || role === 'STROLLER') && variantIdx % 2 === 0) {
        suitcase = new THREE.Group();
        suitcase.position.set(-0.29, 0.25, -0.16);
        const sBody = new THREE.Mesh(PG.suitcaseBody, lugMat);
        sBody.userData.sharedGeo = true;
        const sHard = new THREE.Mesh(PG.suitcaseHardware, this.infraMetalMat);
        suitcase.add(sBody, sHard);
        root.add(suitcase);
      }
    }

    if (!isOfficer && variantIdx % 2 === 1) {
      hasPhone = true;
      armR.elbowPivot.add(new THREE.Mesh(PG.phoneFull, this.infraEmissiveMat));
    } else if (!isOfficer && variantIdx % 5 === 0) {
      const cup = new THREE.Mesh(PG.coffeeCup, PM.collarWhite);
      cup.userData.sharedGeo = true;
      armR.elbowPivot.add(cup);
    }

    return {
      root,
      upperBody,
      headGroup,
      legL,
      legR,
      armL,
      armR,
      suitcase,
      hasPhone
    };
  }

  update(trainS, doorsOpen, dt, activeTier = null, originS = EndlessWorldMathematics.originS, aiTrainS = 999999) {
    if (activeTier && activeTier.stationNpcCount) {
      this.stationNpcCount = activeTier.stationNpcCount;
    }

    const chunkLen = GAME_PREMISE_CARTRIDGE.railway.chunkLength;
    const chunksBehind = activeTier ? activeTier.chunksBehind : GAME_PREMISE_CARTRIDGE.railway.chunksBehind;
    const chunksAhead = activeTier ? activeTier.chunksAhead : GAME_PREMISE_CARTRIDGE.railway.chunksAhead;

    const curIdx = Math.floor(trainS / chunkLen);
    const minIdx = curIdx - chunksBehind;
    const maxIdx = curIdx + chunksAhead;

    // 1. Unload distant chunks & deterministically dispose chunk-unique BufferGeometries
    for (const [idx, group] of this.chunks.entries()) {
      if (idx < minIdx || idx > maxIdx) {
        this.scene.remove(group);
        group.traverse((o) => {
          if (o.geometry && !o.userData.sharedGeo && !o.geometry.userData?.sharedGeo) {
            o.geometry.dispose();
          }
          if (o.material && o.userData.ownMat) {
            o.material.dispose();
          }
        });
        this.chunks.delete(idx);
      }
    }

    // 2. Mount up to 2 worker-completed chunks per frame (zero main-thread normal/terrain math!)
    let mountedCount = 0;
    while (this.completedQueue.length > 0 && mountedCount < 2) {
      const data = this.completedQueue.shift();
      if (data.chunkIdx < minIdx || data.chunkIdx > maxIdx || this.chunks.has(data.chunkIdx)) {
        continue;
      }
      const grp = this.assembleChunkFromBuffers(data, originS);
      this.chunks.set(data.chunkIdx, grp);
      this.scene.add(grp);
      mountedCount++;
    }

    // 3. Request missing chunks (prioritizing nearest ahead of train)
    for (let offset = -chunksBehind; offset <= chunksAhead; offset++) {
      const idx = curIdx + offset;
      if (!this.chunks.has(idx) && !this.pendingChunks.has(idx)) {
        if (this.worker) {
          if (this.pendingChunks.size < 4) {
            this.pendingChunks.add(idx);
            this.worker.postMessage({ type: 'BUILD_CHUNK', chunkIdx: idx });
          }
        } else {
          // Synchronous fallback (max 1 chunk per frame)
          const data = ChunkGeometryCore.buildChunkBuffers(idx);
          const grp = this.assembleChunkFromBuffers(data, originS);
          this.chunks.set(idx, grp);
          this.scene.add(grp);
          break;
        }
      }
    }

    // 3B. Animate 3D River Current & Rapids UV Flow across active valley gorges
    if (this.chunkRiverTex) {
      this.chunkRiverTex.offset.x = (this.chunkRiverTex.offset.x - dt * 0.26) % 1;
      this.chunkRiverTex.offset.y = (this.chunkRiverTex.offset.y + dt * 0.05) % 1;
    }

    // 4. Animate active level crossing barriers & flashing lamps (triggered by BOTH Player Train & Oncoming AI Train!)
    this.levelCrossings = this.levelCrossings.filter(c => c.parent && c.parent.parent !== null);
    const tNow = performance.now() * 0.006;
    for (const cx of this.levelCrossings) {
      const distPlayer = Math.abs(trainS - cx.sCoord);
      const distAi = Math.abs(aiTrainS - cx.sCoord);
      const isOccupied = distPlayer < 340 || distAi < 340;
      const closedTarget = isOccupied ? 0.0 : -Math.PI * 0.46;
      cx.armLeft.rotation.z += (closedTarget - cx.armLeft.rotation.z) * Math.min(1, dt * 3.5);
      cx.armRight.rotation.z += (-closedTarget - cx.armRight.rotation.z) * Math.min(1, dt * 3.5);
      if (isOccupied) {
        const flash = Math.sin(tNow) > 0;
        cx.lampA.material.color.setHex(flash ? 0xf43f5e : 0x330000);
        cx.lampB.material.color.setHex(!flash ? 0xf43f5e : 0x330000);
      } else {
        cx.lampA.material.color.setHex(0x330000);
        cx.lampB.material.color.setHex(0x330000);
      }
    }

    // 4B. Dynamic 3D Wayside Automatic Block Signal Aspects (Red = Occupied Block, Yellow = Caution/Approach, Green = Clear)
    this.blockSignals = this.blockSignals.filter(s => s.parent && s.parent.parent !== null);
    const stInt = GAME_PREMISE_CARTRIDGE.railway.stationInterval;
    for (const sig of this.blockSignals) {
      const passedBy = trainS - sig.sCoord;
      const nextStStopS = Math.max(1, Math.ceil(sig.sCoord / stInt)) * stInt + 48;
      const distSigToStop = nextStStopS - sig.sCoord;

      let aspect = 2; // 0: RED, 1: YELLOW, 2: GREEN
      if ((passedBy >= -8 && passedBy < 450) || (doorsOpen && Math.abs(sig.sCoord - trainS) < 220)) {
        aspect = 0;
      } else if ((passedBy >= 450 && passedBy < 900) || (distSigToStop > 0 && distSigToStop < 620)) {
        aspect = 1;
      }
      sig.lampRed.material = aspect === 0 ? this.sigRedOnMat : this.sigRedOffMat;
      sig.lampYel.material = aspect === 1 ? this.sigYelOnMat : this.sigYelOffMat;
      sig.lampGrn.material = aspect === 2 ? this.sigGrnOnMat : this.sigGrnOffMat;
    }

    // 5. Animate Platform Screen Doors (PSD) synchronized with train doors
    this.stackPsdDoors = this.stackPsdDoors.filter(d => d.parent && d.parent.parent !== null);
    for (const psd of this.stackPsdDoors) {
      const distSt = Math.abs(trainS - (psd.stationS + 48));
      const targetSlide = (doorsOpen && distSt < 95) ? psd.dir * 0.96 : 0.0;
      psd.mesh.position.z = psd.baseZ + (targetSlide - (psd.mesh.position.z - psd.baseZ)) * Math.min(1, dt * 5.0);
    }

    // 6. Animate 3D Wind Turbine Rotors on hilltops
    this.windTurbineRotors = this.windTurbineRotors.filter(r => r.parent && r.parent.parent !== null);
    for (const wt of this.windTurbineRotors) {
      wt.rotor.rotation.z += dt * 1.65;
    }

    // 7. Animate Articulated 3D Station NPCs (with Smart Distance Culling for Blazing-Fast Draw Calls!)
    this.stationPassengers = this.stationPassengers.filter(p => p.parent && p.parent.parent !== null);
    const cullDist = (activeTier && activeTier.npcCullDist) ? activeTier.npcCullDist : 85;
    const tSec = performance.now() * 0.001;
    const lerpYaw = (cur, tgt, rate) => {
      let diff = ((tgt - cur + Math.PI) % (Math.PI * 2)) - Math.PI;
      if (diff < -Math.PI) diff += Math.PI * 2;
      return cur + diff * Math.min(1, rate);
    };

    for (const pax of this.stationPassengers) {
      const distTrainToSt = Math.abs(trainS - pax.stationS);
      if (distTrainToSt > cullDist + 55) {
        if (pax.mesh.visible) pax.mesh.visible = false;
        continue;
      }
      if (pax.role !== 'BOARDING' && pax.role !== 'ALIGHTING') {
        if (!pax.mesh.visible) pax.mesh.visible = true;
      }

      const rig = pax.rig;
      const phase = tSec * pax.walkFreq + pax.seed;
      let isWalking = false;
      let moveSpeed = 0;

      if (pax.role === 'SEATED_BENCH') {
        rig.upperBody.position.y = 0.52 + Math.sin(tSec * 2.2 + pax.seed) * 0.006;
        rig.armR.shoulderPivot.rotation.x = -0.65;
        rig.armR.elbowPivot.rotation.x = -1.15 + Math.sin(tSec * 1.8 + pax.seed) * 0.08;
        rig.armL.shoulderPivot.rotation.x = -0.35;
        rig.armL.elbowPivot.rotation.x = -0.75;
        rig.headGroup.rotation.x = 0.28 + Math.sin(tSec * 1.1 + pax.seed) * 0.08;
        rig.headGroup.rotation.y = Math.sin(tSec * 0.7 + pax.seed) * 0.35;
        continue;
      }

      if (pax.role === 'OFFICER') {
        if (distTrainToSt < 95) {
          pax.mesh.rotation.y = lerpYaw(pax.mesh.rotation.y, Math.PI * 0.5, dt * 6);
          rig.armR.shoulderPivot.rotation.x = -2.15;
          rig.armR.shoulderPivot.rotation.z = -0.35;
          rig.armR.elbowPivot.rotation.x = -1.15;
          rig.armL.shoulderPivot.rotation.x = 0.08;
          rig.armL.elbowPivot.rotation.x = -0.15;
          rig.headGroup.rotation.y = Math.sin(tSec * 1.5) * 0.25;
          rig.headGroup.rotation.x = -0.05;
          rig.legL.hipPivot.rotation.x = 0;
          rig.legR.hipPivot.rotation.x = 0;
          rig.legL.kneePivot.rotation.x = 0;
          rig.legR.kneePivot.rotation.x = 0;
          continue;
        } else {
          isWalking = true;
          moveSpeed = 0.95;
          pax.mesh.position.z += pax.patrolDir * moveSpeed * dt;
          if (pax.mesh.position.z > pax.patrolMaxZ) pax.patrolDir = -1;
          if (pax.mesh.position.z < pax.patrolMinZ) pax.patrolDir = 1;
          const targetY = pax.patrolDir > 0 ? 0 : Math.PI;
          pax.mesh.rotation.y = lerpYaw(pax.mesh.rotation.y, targetY, dt * 6);
          rig.armR.shoulderPivot.rotation.z = 0;
        }
      } else if (pax.role === 'STROLLER') {
        if (pax.pauseTimer > 0) {
          pax.pauseTimer -= dt;
          isWalking = false;
        } else {
          isWalking = true;
          moveSpeed = pax.speed;
          pax.mesh.position.z += pax.patrolDir * moveSpeed * dt;
          if (pax.mesh.position.z > pax.patrolMaxZ) {
            pax.patrolDir = -1;
            pax.pauseTimer = 1.5 + (pax.seed % 2.5);
          } else if (pax.mesh.position.z < pax.patrolMinZ) {
            pax.patrolDir = 1;
            pax.pauseTimer = 1.5 + (pax.seed % 2.5);
          }
          if (Math.random() < 0.002) pax.pauseTimer = 2.0;
          const targetY = pax.patrolDir > 0 ? 0 : Math.PI;
          pax.mesh.rotation.y = lerpYaw(pax.mesh.rotation.y, targetY, dt * 6.5);
        }
      } else if (pax.role === 'CONVERSING') {
        const gestureWave = Math.sin(tSec * 3.4 + pax.seed);
        rig.upperBody.position.y = 0.78 + Math.sin(tSec * 2.0 + pax.seed) * 0.007;
        rig.upperBody.rotation.y = Math.sin(tSec * 1.1 + pax.seed) * 0.08;
        rig.headGroup.rotation.x = Math.sin(tSec * 2.8 + pax.seed) * 0.12;
        rig.headGroup.rotation.y = Math.cos(tSec * 1.4 + pax.seed) * 0.18;
        rig.armR.shoulderPivot.rotation.x = -0.45 + gestureWave * 0.25;
        rig.armR.elbowPivot.rotation.x = -0.95 + Math.cos(tSec * 3.4 + pax.seed) * 0.30;
        rig.armL.shoulderPivot.rotation.x = -0.25 - gestureWave * 0.15;
        rig.armL.elbowPivot.rotation.x = -0.55;
        continue;
      } else if (pax.role === 'BOARDING' || pax.role === 'ALIGHTING') {
        if (doorsOpen && distTrainToSt < 120) {
          pax.timer += dt;
          if (pax.timer >= pax.delay) {
            if (pax.role === 'ALIGHTING' && !pax.hasSpawnedOut) {
              pax.hasSpawnedOut = true;
              pax.mesh.visible = true;
            }
            const dx = pax.targetX - pax.mesh.position.x;
            const dz = pax.targetZ - pax.mesh.position.z;
            const dist = Math.hypot(dx, dz);

            if (dist > 0.14) {
              isWalking = true;
              moveSpeed = pax.speed;
              const step = Math.min(dist, moveSpeed * dt);
              pax.mesh.position.x += (dx / dist) * step;
              pax.mesh.position.z += (dz / dist) * step;
              const desiredYaw = Math.atan2(dx, dz);
              pax.mesh.rotation.y = lerpYaw(pax.mesh.rotation.y, desiredYaw, dt * 8.5);
            } else {
              isWalking = false;
              if (pax.role === 'BOARDING') {
                pax.mesh.visible = false;
              } else if (pax.role === 'ALIGHTING') {
                pax.targetZ = pax.homeZ + (pax.seed % 2 > 1 ? 26 : -26);
                pax.targetX = -4.6 - (pax.seed % 1.4);
              }
            }
          }
        } else {
          pax.timer = 0;
          pax.hasSpawnedOut = false;
          if (pax.role === 'BOARDING') {
            pax.mesh.visible = true;
            const dx = pax.homeX - pax.mesh.position.x;
            const dz = pax.homeZ - pax.mesh.position.z;
            const dHome = Math.hypot(dx, dz);
            if (dHome > 0.18) {
              isWalking = true;
              moveSpeed = 1.0;
              pax.mesh.position.x += (dx / dHome) * moveSpeed * dt;
              pax.mesh.position.z += (dz / dHome) * moveSpeed * dt;
              pax.mesh.rotation.y = lerpYaw(pax.mesh.rotation.y, Math.atan2(dx, dz), dt * 6);
            } else {
              pax.mesh.rotation.y = lerpYaw(pax.mesh.rotation.y, Math.PI * 0.5, dt * 4);
            }
          } else if (pax.role === 'ALIGHTING') {
            pax.mesh.position.set(pax.homeX, 1.02, pax.homeZ);
            pax.mesh.visible = false;
          }
        }
      }

      if (isWalking) {
        const strideAmp = THREE.MathUtils.clamp(moveSpeed * 0.42, 0.35, 0.68);
        const sinP = Math.sin(phase);
        const cosP = Math.cos(phase);

        rig.legL.hipPivot.rotation.x = sinP * strideAmp;
        rig.legR.hipPivot.rotation.x = -sinP * strideAmp;
        rig.legL.kneePivot.rotation.x = Math.max(0, -cosP) * strideAmp * 1.35 + 0.08;
        rig.legR.kneePivot.rotation.x = Math.max(0, cosP) * strideAmp * 1.35 + 0.08;

        rig.armL.shoulderPivot.rotation.x = -sinP * strideAmp * 0.75;
        rig.armL.elbowPivot.rotation.x = -0.35 - Math.max(0, sinP) * 0.42;

        if (rig.suitcase) {
          rig.armL.shoulderPivot.rotation.x = 0.28;
          rig.armL.elbowPivot.rotation.x = -0.45;
          rig.suitcase.position.z = -0.14 + sinP * 0.02;
        }

        rig.armR.shoulderPivot.rotation.x = sinP * strideAmp * 0.75;
        rig.armR.elbowPivot.rotation.x = -0.35 - Math.max(0, -sinP) * 0.42;

        const bob = Math.abs(cosP) * 0.042;
        rig.upperBody.position.y = 0.77 + bob;
        rig.upperBody.rotation.y = -sinP * 0.08;
        rig.upperBody.rotation.z = cosP * 0.025;
        rig.headGroup.rotation.y = sinP * 0.06;
        rig.headGroup.rotation.x = 0.04;
      } else {
        const breath = Math.sin(tSec * 2.1 + pax.seed);
        const weightShift = Math.sin(tSec * 0.85 + pax.seed);

        rig.upperBody.position.y = 0.78 + breath * 0.006;
        rig.upperBody.rotation.z = weightShift * 0.022;
        rig.upperBody.rotation.y = Math.sin(tSec * 0.6 + pax.seed) * 0.06;

        rig.legL.hipPivot.rotation.x = 0;
        rig.legR.hipPivot.rotation.x = 0.04;
        rig.legL.kneePivot.rotation.x = 0.02;
        rig.legR.kneePivot.rotation.x = 0.06;

        rig.armL.shoulderPivot.rotation.x = 0.04 + breath * 0.02;
        rig.armL.elbowPivot.rotation.x = -0.18;

        if (rig.hasPhone) {
          const checkCycle = Math.sin(tSec * 0.55 + pax.seed);
          if (checkCycle > -0.2) {
            rig.armR.shoulderPivot.rotation.x = -0.58;
            rig.armR.elbowPivot.rotation.x = -1.22 + breath * 0.04;
            rig.headGroup.rotation.x = 0.32;
            rig.headGroup.rotation.y = -0.12 + Math.sin(tSec * 1.4 + pax.seed) * 0.08;
          } else {
            rig.armR.shoulderPivot.rotation.x = -0.15;
            rig.armR.elbowPivot.rotation.x = -0.45;
            rig.headGroup.rotation.x = 0.0;
            rig.headGroup.rotation.y = Math.sin(tSec * 0.9 + pax.seed) * 0.35;
          }
        } else {
          rig.armR.shoulderPivot.rotation.x = -0.04 - breath * 0.02;
          rig.armR.elbowPivot.rotation.x = -0.22;
          rig.headGroup.rotation.x = Math.sin(tSec * 0.7 + pax.seed) * 0.06;
          rig.headGroup.rotation.y = Math.sin(tSec * 0.8 + pax.seed) * 0.38;
        }
      }
    }
  }

  /**
   * Assembles a Three.js chunk `Group` directly from precomputed Worker/Core
   * typed arrays + placement metadata, anchored at `sStart` for O(1) rebasing.
   */
  assembleChunkFromBuffers(data, originS) {
    const group = new THREE.Group();
    const { chunkIdx, sStart, sEnd, biomeGroundColor } = data;
    group.userData.chunkIdx = chunkIdx;
    group.userData.sStart = sStart;
    group.position.set(0, 0, -(sStart - originS));

    const secondTrackOff = GAME_PREMISE_CARTRIDGE.railway.doubleTrackOffset;

    // 1. Ballast / Track Formation Bed Mesh
    const bGeo = new THREE.BufferGeometry();
    bGeo.setAttribute('position', new THREE.BufferAttribute(data.ballast.pos, 3));
    bGeo.setAttribute('normal', new THREE.BufferAttribute(data.ballast.nrm, 3));
    bGeo.setAttribute('uv', new THREE.BufferAttribute(data.ballast.uv, 2));
    if (data.ballast.col && data.ballast.col.length > 0) {
      bGeo.setAttribute('color', new THREE.BufferAttribute(data.ballast.col, 3));
    }
    bGeo.setIndex(new THREE.BufferAttribute(data.ballast.idx, 1));
    const ballastMesh = new THREE.Mesh(bGeo, this.ballastMat);
    ballastMesh.receiveShadow = true;
    group.add(ballastMesh);

    // 2. 4 Profiled UIC-60 Steel Rails + Bridge Guard Rails + Station Crossover Turnouts Mesh
    const rGeo = new THREE.BufferGeometry();
    rGeo.setAttribute('position', new THREE.BufferAttribute(data.rails.pos, 3));
    rGeo.setAttribute('normal', new THREE.BufferAttribute(data.rails.nrm, 3));
    rGeo.setAttribute('color', new THREE.BufferAttribute(data.rails.col, 3));
    rGeo.setIndex(new THREE.BufferAttribute(data.rails.idx, 1));
    const railMesh = new THREE.Mesh(rGeo, this.profiledRailMat);
    railMesh.receiveShadow = true;
    group.add(railMesh);

    // 3. 3D Overhead Catenary & Dropper Wires
    const wGeo = new THREE.BufferGeometry();
    wGeo.setAttribute('position', new THREE.BufferAttribute(data.wires.pos, 3));
    wGeo.setIndex(new THREE.BufferAttribute(data.wires.idx, 1));
    group.add(new THREE.Mesh(wGeo, this.wireMat));

    // 4. Multi-Band Terraced Terrain Strip
    const tGeo = new THREE.BufferGeometry();
    tGeo.setAttribute('position', new THREE.BufferAttribute(data.terrain.pos, 3));
    tGeo.setAttribute('normal', new THREE.BufferAttribute(data.terrain.nrm, 3));
    tGeo.setAttribute('uv', new THREE.BufferAttribute(data.terrain.uv, 2));
    tGeo.setAttribute('color', new THREE.BufferAttribute(data.terrain.col, 3));
    tGeo.setIndex(new THREE.BufferAttribute(data.terrain.idx, 1));
    const terrainMesh = new THREE.Mesh(tGeo, this.terrainMat);
    terrainMesh.receiveShadow = true;
    group.add(terrainMesh);

    // 4B. Chunk-Local 3D Flowing River & Coastal Ocean Mesh (Zero Pop-In, Meandering Shoreline & Pier Rapids!)
    if (data.water && data.water.pos && data.water.pos.length > 0) {
      const watGeo = new THREE.BufferGeometry();
      watGeo.setAttribute('position', new THREE.BufferAttribute(data.water.pos, 3));
      watGeo.setAttribute('normal', new THREE.BufferAttribute(data.water.nrm, 3));
      watGeo.setAttribute('uv', new THREE.BufferAttribute(data.water.uv, 2));
      watGeo.setAttribute('color', new THREE.BufferAttribute(data.water.col, 3));
      watGeo.setIndex(new THREE.BufferAttribute(data.water.idx, 1));
      const chunkWaterMesh = new THREE.Mesh(watGeo, this.chunkWaterMat);
      chunkWaterMesh.receiveShadow = true;
      group.add(chunkWaterMesh);
    }

    // 5. Batched Scenery, Multi-Species Trees, Landmarks, SUTET & Villages (1 Draw Call!)
    if (data.scenery.pos.length > 0) {
      const sGeo = new THREE.BufferGeometry();
      sGeo.setAttribute('position', new THREE.BufferAttribute(data.scenery.pos, 3));
      sGeo.setAttribute('normal', new THREE.BufferAttribute(data.scenery.nrm, 3));
      sGeo.setAttribute('color', new THREE.BufferAttribute(data.scenery.col, 3));
      const sceneryMesh = new THREE.Mesh(sGeo, this.sceneryBatchMat);
      sceneryMesh.castShadow = true;
      sceneryMesh.receiveShadow = true;
      group.add(sceneryMesh);
    }

    // 6. Tunnels, Concrete Portals, Viaducts & Catenary Portals (Chunk-Local via `sStart`)
    for (let i = 0; i < data.tunnelSpans.length; i++) {
      EndlessWorldMathematics.getTrackFrame(data.tunnelSpans[i], secondTrackOff * 0.5, sStart, _frameTmp);
      this.buildTunnelSegment(group, _frameTmp, biomeGroundColor);
    }

    if (data.tunnelPortals) {
      for (let i = 0; i < data.tunnelPortals.length; i++) {
        EndlessWorldMathematics.getTrackFrame(data.tunnelPortals[i].s, secondTrackOff * 0.5, sStart, _frameTmp);
        const portalMesh = new THREE.Mesh(this.mergedTunnelPortalGeo, this.infraConcreteMat);
        this._orientGroupToFrame(portalMesh, _frameTmp);
        portalMesh.castShadow = true;
        portalMesh.receiveShadow = true;
        group.add(portalMesh);
      }
    }

    if (data.bridgeAbutments) {
      for (let i = 0; i < data.bridgeAbutments.length; i++) {
        const ab = data.bridgeAbutments[i];
        EndlessWorldMathematics.getTrackFrame(ab.s, secondTrackOff * 0.5, sStart, _frameTmp);
        const abGrp = new THREE.Group();
        this._orientGroupToFrame(abGrp, _frameTmp);
        const abMesh = new THREE.Mesh(this.mergedBridgeAbutmentGeo, this.infraConcreteMat);
        if (ab.isExit) abMesh.rotation.y = Math.PI;
        abMesh.castShadow = true;
        abMesh.receiveShadow = true;
        abGrp.add(abMesh);
        group.add(abGrp);
      }
    }

    for (let i = 0; i < data.viaductSpans.length; i++) {
      const sp = data.viaductSpans[i];
      EndlessWorldMathematics.getTrackFrame(sp.s, secondTrackOff * 0.5, sStart, _frameTmp);
      this.buildDoubleTrackViaduct(group, _frameTmp, sp);
    }

    for (let i = 0; i < data.catenarySpans.length; i++) {
      EndlessWorldMathematics.getTrackFrame(data.catenarySpans[i], secondTrackOff * 0.5, sStart, _frameTmp);
      this.buildDoubleTrackCatenary(group, _frameTmp);
    }

    // 7. Metropolis Glass Skyscrapers
    for (let i = 0; i < data.metropolisBuildings.length; i++) {
      const b = data.metropolisBuildings[i];
      const bldg = new THREE.Mesh(new THREE.BoxGeometry(b.w, b.h, b.d), this.skyscraperMat);
      bldg.position.set(b.x, b.y, b.z);
      bldg.castShadow = true;
      bldg.receiveShadow = true;
      group.add(bldg);
    }

    // 8. Animated Hilltop Wind Turbines
    for (let i = 0; i < data.windTurbines.length; i++) {
      const wt = data.windTurbines[i];
      const staticMesh = new THREE.Mesh(this.mergedTurbineStaticGeo, this.insulatorMat);
      staticMesh.position.set(wt.x, wt.y, wt.z);
      staticMesh.castShadow = true;

      const beacon = new THREE.Mesh(this.sharedSigLampGeo, this.beaconRedMat);
      beacon.position.set(wt.x, wt.y + 31.85, wt.z - 1.2);

      const rotor = new THREE.Mesh(this.mergedTurbineRotorGeo, this.insulatorMat);
      rotor.position.set(wt.x, wt.y + 30.8, wt.z + 1.55);
      rotor.castShadow = true;

      group.add(staticMesh, beacon, rotor);
      this.windTurbineRotors.push({ parent: group, rotor });
    }

    // 9. Station Complex Check
    const stInt = GAME_PREMISE_CARTRIDGE.railway.stationInterval;
    const stIdx = Math.round(sStart / stInt);
    const stCenterS = stIdx * stInt;
    if (stIdx >= 0 && stCenterS >= sStart && stCenterS < sEnd) {
      this.buildStationComplex(group, stCenterS, stIdx, sStart);
    }

    // 10. Block Signal Check
    const sigInt = GAME_PREMISE_CARTRIDGE.railway.signalInterval;
    const sigS = Math.ceil(sStart / sigInt) * sigInt;
    if (
      sigS >= sStart &&
      sigS < sEnd &&
      !EndlessWorldMathematics.isTunnelZone(sigS) &&
      EndlessWorldMathematics.getViaductFactor(sigS) < 0.02 &&
      !EndlessWorldMathematics.isNearStation(sigS, 85)
    ) {
      this.buildBlockSignal(group, sigS, sStart);
    }

    // 11. Level Crossing Check
    const cxInt = GAME_PREMISE_CARTRIDGE.railway.crossingInterval;
    const cxS = Math.ceil(sStart / cxInt) * cxInt;
    if (
      cxS > 300 &&
      cxS >= sStart &&
      cxS < sEnd &&
      Math.abs(cxS - stCenterS) > 350 &&
      !EndlessWorldMathematics.isTunnelZone(cxS) &&
      !EndlessWorldMathematics.isViaductZone(cxS)
    ) {
      this.buildLevelCrossing(group, cxS, sStart);
    }

    return group;
  }

  _orientGroupToFrame(grp, frame) {
    grp.position.copy(frame.position);
    _negTangent.copy(frame.tangent).negate();
    _rotMat.makeBasis(frame.right, frame.up, _negTangent);
    grp.quaternion.setFromRotationMatrix(_rotMat);
  }

  buildTunnelSegment(group, midFrame, biomeGroundColor) {
    const tGrp = new THREE.Group();
    this._orientGroupToFrame(tGrp, midFrame);

    const archMesh = new THREE.Mesh(this.sharedTunnelArchGeo, this.tunnelMat);
    const rockShell = new THREE.Mesh(this.sharedTunnelRockGeo, this._getRockShellMat(biomeGroundColor || 0x334155));
    const hardware = new THREE.Mesh(this.mergedTunnelHardwareGeo, this.infraConcreteMat);
    const sconces = new THREE.Mesh(this.mergedTunnelSconcesGeo, this.infraEmissiveMat);

    tGrp.add(archMesh, rockShell, hardware, sconces);
    group.add(tGrp);
  }

  buildDoubleTrackViaduct(group, midFrame, sp = {}) {
    const hasConcretePier = sp.hasPier !== undefined ? sp.hasPier : true;
    const isMainTruss = !sp.mode || sp.mode === 'TRUSS';
    const vGrp = new THREE.Group();
    this._orientGroupToFrame(vGrp, midFrame);

    const deckMesh = new THREE.Mesh(
      hasConcretePier ? this.mergedViaductDeckWithPierGeo : this.mergedViaductDeckNoPierGeo,
      this.infraConcreteMat
    );
    deckMesh.castShadow = true;
    deckMesh.receiveShadow = true;
    vGrp.add(deckMesh);

    if (hasConcretePier && sp.inWater && sp.riverLocalY !== undefined) {
      const cutwater = new THREE.Mesh(this.mergedPierCutwaterGeo, this.infraConcreteMat);
      cutwater.position.y = sp.riverLocalY;
      cutwater.receiveShadow = true;
      vGrp.add(cutwater);
    }

    if (isMainTruss) {
      const trussMesh = new THREE.Mesh(this.mergedViaductTrussGeo, this.infraMetalMat);
      trussMesh.castShadow = true;
      trussMesh.receiveShadow = true;
      vGrp.add(trussMesh);
    } else {
      const railMesh = new THREE.Mesh(this.mergedApproachRailGeo, this.infraMetalMat);
      vGrp.add(railMesh);
      if (sp.hasCatenaryMast) {
        const mastMesh = new THREE.Mesh(this.mergedCatenaryPortalGeo, this.infraMetalMat);
        mastMesh.castShadow = true;
        vGrp.add(mastMesh);
      }
    }

    group.add(vGrp);
  }

  buildDoubleTrackCatenary(group, midFrame) {
    const cMesh = new THREE.Mesh(this.mergedCatenaryPortalGeo, this.infraMetalMat);
    this._orientGroupToFrame(cMesh, midFrame);
    cMesh.castShadow = true;
    group.add(cMesh);
  }

  buildBlockSignal(group, sigS, sStart) {
    EndlessWorldMathematics.getTrackFrame(sigS, 0, sStart, _frameTmp);
    const sigGrp = new THREE.Group();
    this._orientGroupToFrame(sigGrp, _frameTmp);

    const staticBody = new THREE.Mesh(this.mergedBlockSignalGeo, this.infraPbrMat);
    staticBody.castShadow = true;

    const lampRed = new THREE.Mesh(this.sharedSigLampGeo, this.sigRedOffMat);
    lampRed.position.set(-2.6, 4.78, 0.28);
    const lampYel = new THREE.Mesh(this.sharedSigLampGeo, this.sigYelOffMat);
    lampYel.position.set(-2.6, 4.35, 0.28);
    const lampGrn = new THREE.Mesh(this.sharedSigLampGeo, this.sigGrnOnMat);
    lampGrn.position.set(-2.6, 3.92, 0.28);

    sigGrp.add(staticBody, lampRed, lampYel, lampGrn);
    group.add(sigGrp);
    this.blockSignals.push({ parent: group, sCoord: sigS, lampRed, lampYel, lampGrn });
  }

  buildLevelCrossing(group, cxS, sStart) {
    EndlessWorldMathematics.getTrackFrame(cxS, 2.2, sStart, _frameTmp);
    const cxGrp = new THREE.Group();
    this._orientGroupToFrame(cxGrp, _frameTmp);

    const staticRoad = new THREE.Mesh(this.mergedCrossingStaticGeo, this.infraPbrMat);
    staticRoad.receiveShadow = true;
    staticRoad.castShadow = true;

    // Individual materials per crossing so flashing lamps don't cross-talk
    const lampA = new THREE.Mesh(this.sharedCxLampGeo, new THREE.MeshBasicMaterial({ color: 0xf43f5e }));
    lampA.position.set(-6.2, 2.25, 4.25);
    const lampB = new THREE.Mesh(this.sharedCxLampGeo, new THREE.MeshBasicMaterial({ color: 0x330000 }));
    lampB.position.set(6.2, 2.25, -4.25);
    lampA.userData.ownMat = true;
    lampB.userData.ownMat = true;

    const armLeft = new THREE.Group();
    armLeft.position.set(-6.2, 1.1, 4.1);
    armLeft.add(new THREE.Mesh(this.mergedBoomLeftGeo, this.infraPbrMat));

    const armRight = new THREE.Group();
    armRight.position.set(6.2, 1.1, -4.1);
    armRight.add(new THREE.Mesh(this.mergedBoomRightGeo, this.infraPbrMat));

    cxGrp.add(staticRoad, lampA, lampB, armLeft, armRight);
    group.add(cxGrp);

    this.levelCrossings.push({ parent: group, sCoord: cxS, armLeft, armRight, lampA, lampB });
  }

  buildStationComplex(group, stCenterS, stIdx, sStart) {
    EndlessWorldMathematics.getTrackFrame(stCenterS, 0, sStart, _frameTmp);
    const stGrp = new THREE.Group();
    this._orientGroupToFrame(stGrp, _frameTmp);
    const platCenterZ = 52;

    // 1. Textured 276m Island Platforms (covering full 10-car consist from z=-86m to z=+190m!)
    const platL = new THREE.Mesh(this.sharedPlatBoxGeo, this.platMats);
    platL.position.set(-5.35, 0.40, platCenterZ);
    platL.receiveShadow = true;
    const platR = new THREE.Mesh(this.sharedPlatBoxGeo, this.platMats);
    platR.position.set(9.75, 0.40, platCenterZ);
    platR.receiveShadow = true;
    stGrp.add(platL, platR);

    // 2. Pre-Baked Mega-Station Opaque, Glass & Emissive Architecture (3 GPU Draw Calls!)
    const stOpaque = new THREE.Mesh(this.mergedStationOpaqueGeo, this.infraPbrMat);
    stOpaque.castShadow = true;
    stOpaque.receiveShadow = true;
    const stGlass = new THREE.Mesh(this.mergedStationGlassGeo, this.stationGlassMat);
    const stEmissive = new THREE.Mesh(this.mergedStationEmissiveGeo, this.infraEmissiveMat);
    stGrp.add(stOpaque, stGlass, stEmissive);

    // 3. Synchronized Sliding PSD Glass Doors on Mainline 1 Platform (Aligned 1:1 with stopped Coaches 1..8!)
    const doorZList = [-34.2, -18.6, -12.6, 3.0, 9.0, 24.6, 30.6, 46.2, 52.2, 67.8, 73.8, 89.4, 95.4, 111.0, 117.0];
    doorZList.forEach((dz) => {
      const px = -2.08;
      const slideL = new THREE.Mesh(this.sharedPsdDoorPanelGeo, this.psdDoorGlassMat);
      slideL.position.set(px, 2.00, dz - 0.55);
      const slideR = new THREE.Mesh(this.sharedPsdDoorPanelGeo, this.psdDoorGlassMat);
      slideR.position.set(px, 2.00, dz + 0.55);
      stGrp.add(slideL, slideR);

      this.stackPsdDoors.push(
        { parent: group, mesh: slideL, baseZ: dz - 0.55, dir: -1, stationS: stCenterS },
        { parent: group, mesh: slideR, baseZ: dz + 0.55, dir: 1, stationS: stCenterS }
      );
    });

    // 4. Multi-Story Glass Terminal Concourse Building & 3D Platform Nameboards
    const stName = GAME_PREMISE_CARTRIDGE.stations[Math.max(0, stIdx) % GAME_PREMISE_CARTRIDGE.stations.length];
    const facadeMat = this._getStationFacadeMat(stName);
    const mainTerminal = new THREE.Mesh(this.sharedTerminalBoxGeo, [
      facadeMat, facadeMat, this.roofDarkMat, this.concreteMat, this.concreteMat, this.concreteMat
    ]);
    mainTerminal.position.set(-13.8, 5.5, 22);
    mainTerminal.castShadow = true;
    mainTerminal.receiveShadow = true;
    stGrp.add(mainTerminal);

    const signMat = this._getStationSignMat(stName, 680 + (stIdx * 27) % 240);
    [-52, -16, 20, 56, 92].forEach((sz) => {
      const board = new THREE.Mesh(this.sharedSignBoardGeo, signMat);
      board.position.set(-3.9, 2.55, sz);
      board.rotation.y = Math.PI / 2;
      stGrp.add(board);
    });

    // 5. Articulated 3D Humans Scaled Cleanly by Active Quality Tier
    const benchZList = [-70, -40, -10, 20, 50, 80, 110, 140];
    const totalNpcs = this.stationNpcCount || 104;

    // Sample evenly across the 128 canonical NPC slots so every role (Boarding, Alighting, Stroller, Bench, Conversing, Skybridge, Officer) is represented!
    for (let k = 0; k < totalNpcs; k++) {
      const p = Math.min(127, Math.floor((k * 128) / totalNpcs));
      let role = 'BOARDING';
      if (p < 28) role = 'BOARDING';
      else if (p < 46) role = 'ALIGHTING';
      else if (p < 78) role = 'STROLLER';
      else if (p < 94) role = 'SEATED_BENCH';
      else if (p < 110) role = 'CONVERSING';
      else if (p < 124) role = 'SKYBRIDGE_COMMUTER';
      else role = 'OFFICER';

      const rig = this.createDetailedHumanNPC(p + stIdx * 7, role === 'SKYBRIDGE_COMMUTER' ? 'STROLLER' : role);
      const pax = rig.root;

      const doorZ = doorZList[p % doorZList.length];
      let homeX = -3.6, homeZ = doorZ, targetX = -1.45, targetZ = doorZ;
      let baseY = 1.02;
      let patrolMinZ = -72, patrolMaxZ = 142, patrolDir = (p % 2 === 0 ? 1 : -1);

      if (role === 'BOARDING') {
        const queueRow = Math.floor(p / 13);
        homeX = -2.85 - queueRow * 0.58;
        homeZ = doorZ + ((p % 2 === 0 ? 1 : -1) * (0.62 + (p % 3) * 0.28));
        targetX = -1.45;
        targetZ = doorZ;
        pax.rotation.y = Math.PI * 0.5;
        pax.visible = true;
      } else if (role === 'ALIGHTING') {
        homeX = -1.45;
        homeZ = doorZ;
        targetX = -3.9 - (p % 5) * 0.65;
        targetZ = doorZ + ((p % 2 === 0 ? 1 : -1) * (2.2 + (p % 4) * 0.95));
        pax.rotation.y = -Math.PI * 0.5;
        pax.visible = false;
      } else if (role === 'STROLLER') {
        const idxStroll = p - 46;
        const onRightPlat = (idxStroll % 2 === 1);
        homeX = onRightPlat ? (7.8 + (idxStroll % 5) * 0.52) : (-3.6 - (idxStroll % 5) * 0.52);
        homeZ = -68 + idxStroll * 6.2;
        targetX = homeX;
        targetZ = homeZ;
        pax.rotation.y = patrolDir > 0 ? 0 : Math.PI;
      } else if (role === 'SEATED_BENCH') {
        const bIdx = (p - 78) % benchZList.length;
        const onRight = (p >= 86);
        homeX = onRight ? 10.02 : -5.62;
        homeZ = benchZList[bIdx] + ((p % 2 === 0) ? -0.46 : 0.46);
        pax.position.set(homeX, 0.98, homeZ);
        pax.rotation.y = onRight ? -Math.PI * 0.5 : Math.PI * 0.5;
        stGrp.add(pax);
        this.stationPassengers.push({
          parent: group,
          mesh: pax,
          rig,
          role,
          stationS: stCenterS,
          homeX,
          homeZ,
          targetX: homeX,
          targetZ: homeZ,
          delay: 0,
          timer: 0,
          speed: 0,
          walkFreq: 5.5,
          seed: p * 1.7
        });
        continue;
      } else if (role === 'CONVERSING') {
        const pairIdx = Math.floor((p - 94) / 2);
        const isFirst = (p % 2 === 0);
        const onRight = (pairIdx % 2 === 1);
        homeX = onRight ? (8.6 + (pairIdx % 3) * 0.45) : (-4.5 - (pairIdx % 3) * 0.45);
        homeZ = -54 + pairIdx * 22.0 + (isFirst ? -0.42 : 0.42);
        pax.rotation.y = isFirst ? 0 : Math.PI;
      } else if (role === 'SKYBRIDGE_COMMUTER') {
        const sbIdx = p - 110;
        homeX = -9.5 + sbIdx * 1.45;
        baseY = 7.90;
        homeZ = 22.2 + (sbIdx % 3) * 1.5;
        pax.position.set(homeX, baseY, homeZ);
        pax.rotation.y = (sbIdx % 2 === 0) ? Math.PI * 0.5 : -Math.PI * 0.5;
        stGrp.add(pax);
        this.stationPassengers.push({
          parent: group,
          mesh: pax,
          rig,
          role: 'CONVERSING',
          stationS: stCenterS,
          homeX,
          homeZ,
          targetX: homeX,
          targetZ: homeZ,
          delay: 0,
          timer: 0,
          speed: 0,
          walkFreq: 5.2,
          seed: p * 1.3
        });
        continue;
      } else if (role === 'OFFICER') {
        const offIdx = p - 124;
        const isRightOff = (offIdx % 2 === 1);
        homeX = isRightOff ? 7.05 : -2.65;
        homeZ = offIdx < 2 ? -36 : 56;
        patrolMinZ = homeZ - 22;
        patrolMaxZ = homeZ + 22;
        pax.rotation.y = isRightOff ? -Math.PI * 0.5 : Math.PI * 0.5;
      }

      pax.position.set(homeX, baseY, homeZ);
      stGrp.add(pax);

      this.stationPassengers.push({
        parent: group,
        mesh: pax,
        rig,
        role,
        stationS: stCenterS,
        homeX,
        homeZ,
        targetX,
        targetZ,
        patrolMinZ,
        patrolMaxZ,
        patrolDir,
        pauseTimer: 0,
        hasSpawnedOut: false,
        delay: (p % 6) * 0.28,
        timer: 0,
        speed: 1.18 + (p % 4) * 0.18,
        walkFreq: 5.8 + (p % 3) * 0.6,
        seed: p * 1.37
      });
    }

    group.add(stGrp);
  }
}
