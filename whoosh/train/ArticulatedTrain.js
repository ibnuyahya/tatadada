import * as THREE from 'three';
import { GAME_PREMISE_CARTRIDGE } from '../data/PremiseCartridge.js';
import { EndlessWorldMathematics, createTrackFrameStruct } from '../world/WorldMath.js';
import { ProceduralAssetFactory } from '../assets/ProceduralTextures.js';
import { GeometryBatcher } from '../engine/GeometryBatcher.js';

/**
 * ============================================================================
 * NEXT-GEN ARTICULATED MULTI-BOGIE TRAIN + CLEARCOAT PBR + ZERO-GC KINEMATICS
 * - Automotive `MeshPhysicalMaterial` clearcoat on CR400AF Whoosh & KAI rolling stock
 * - Pre-baked GPU bogie assemblies (52 meshes -> 3 meshes per bogie!)
 * - Pre-baked GPU highway vehicles (~45 meshes -> 3 meshes per vehicle!)
 * - Cached material & geometry pools (zero VRAM leaks when switching liveries/AI)
 * - Zero-GC spline chord kinematics + Floating-Origin particle rebasing
 * ============================================================================
 */
const _fFront = createTrackFrameStruct();
const _fRear = createTrackFrameStruct();
const _fCenter = createTrackFrameStruct();
const _hFrame = createTrackFrameStruct();

const _midPos = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _negFwd = new THREE.Vector3();
const _upWorld = new THREE.Vector3(0, 1, 0);
const _right = new THREE.Vector3();
const _up = new THREE.Vector3();
const _basisMat = new THREE.Matrix4();
const _locoPos = new THREE.Vector3();

export class ArticulatedTrainSystem {
  constructor(scene, envMap, microBump, worldRef = null) {
    this.scene = scene;
    this.envMap = envMap;
    this.microBump = microBump;
    this.world = worldRef;

    this.rootGroup = new THREE.Group();
    this.scene.add(this.rootGroup);

    // High-Resolution Triple-Panel Glass Cockpit MFD Canvas Texture
    this.mfdCanvas = document.createElement('canvas');
    this.mfdCanvas.width = 1024;
    this.mfdCanvas.height = 256;
    this.mfdCtx = this.mfdCanvas.getContext('2d');
    this.mfdTexture = new THREE.CanvasTexture(this.mfdCanvas);
    this.mfdTexture.colorSpace = THREE.SRGBColorSpace;
    this.mfdNeedsInitialDraw = true;

    this.liveryKey = 'WHOOSH';
    this.numCoaches = 8;
    this.vehicles = [];
    this.interactiveCabMeshes = [];

    this.doorsOpen = false;
    this.doorAnim = 0;
    this.pantoRaised = true;
    this.pantoAnim = 1.0;
    this.wipersActive = false;
    this.wiperPhase = 0;
    this.lightsActive = true;
    this.wheelSpinPhase = 0;
    this.aiWheelSpinPhase = 0;
    this.lastWheelSlip = false;

    this.playerMatCache = new Map();
    this.aiMatCache = new Map();
    this.trimGeoCache = new Map();

    // Persistent locomotive headlight SpotLight (re-parented across livery changes)
    this.spotLight = new THREE.SpotLight(0xfffbeb, 38.0, 280, Math.PI / 5, 0.45, 1.25);
    this.spotLightTarget = new THREE.Object3D();

    this._initSharedRollingStockAssets();

    this.particleGroup = new THREE.Group();
    this.scene.add(this.particleGroup);
    this.smokePool = [];
    this.sparkPool = [];
    this.initParticlePools();

    // Articulated Oncoming Parallel Track AI Train
    this.aiTrainGroup = new THREE.Group();
    this.scene.add(this.aiTrainGroup);
    this.aiVehicles = [];
    this.aiPoolKeys = [
      'NUSANTARA',
      'FREIGHT',
      'KAI_STAINLESS',
      'TANKER',
      'WHOOSH',
      'COAL_FREIGHT',
      'KRL_COMMUTER',
      'WHOOSH_CIT'
    ];
    this.aiPoolIdx = 0;
    this.currentAiLiveryKey = 'NUSANTARA';
    this.aiTrainS = 780;
    this.aiSpeedMps = 44.0;
    this.passedPlayer = false;
    this.buildOncomingAITrain(this.currentAiLiveryKey);

    // Animated 3D Highway Vehicles Pool on Parallel Toll Road
    this.highwayGroup = new THREE.Group();
    this.scene.add(this.highwayGroup);
    this.highwayVehicles = [];
    this.initHighwayTrafficPool();

    this.rebuildPlayerConsist('WHOOSH', 8);
  }

  _disposeGroupNonShared(grp) {
    while (grp.children.length > 0) {
      const child = grp.children[0];
      grp.remove(child);
      child.traverse((o) => {
        if (o.geometry && !o.userData.sharedGeo && !o.geometry.userData?.sharedGeo) {
          o.geometry.dispose();
        }
      });
    }
  }

  _initSharedRollingStockAssets() {
    const B = GeometryBatcher;

    this.bogieVertexMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      metalness: 0.80,
      roughness: 0.28,
      envMapIntensity: 1.35
    });
    this.wheelVertexMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      metalness: 0.90,
      roughness: 0.16,
      envMapIntensity: 1.5
    });
    this.highwayBodyMat = new THREE.MeshPhysicalMaterial({
      vertexColors: true,
      metalness: 0.45,
      roughness: 0.24,
      clearcoat: 0.72,
      clearcoatRoughness: 0.12,
      envMapIntensity: 1.35
    });
    this.highwayGlassMat = new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8,
      metalness: 0.85,
      roughness: 0.06,
      clearcoat: 1.0,
      envMapIntensity: 1.6
    });
    this.highwayEmissiveMat = new THREE.MeshBasicMaterial({
      vertexColors: true
    });
    this.sharedLampWhiteMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    this.sharedTailRedMat = new THREE.MeshBasicMaterial({ color: 0xf43f5e });
    this.sharedDrlCyanMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    this.sharedChromeMat = new THREE.MeshStandardMaterial({
      color: 0xf8fafc,
      metalness: 0.92,
      roughness: 0.12,
      envMapIntensity: 1.5
    });
    this.sharedPorcelainMat = new THREE.MeshStandardMaterial({
      color: 0xf8fafc,
      roughness: 0.18,
      metalness: 0.15
    });
    this.sharedCopperRedMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      metalness: 0.65,
      roughness: 0.25
    });
    this.sharedBellowMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.82,
      metalness: 0.15,
      side: THREE.DoubleSide
    });
    this.sharedCowcatcherRedMat = new THREE.MeshStandardMaterial({
      color: 0xdc2626,
      roughness: 0.38,
      metalness: 0.35
    });

    // Pre-Bake Bogie Static Frame (38 parts -> 1 GPU Geometry!) & Wheelset (13 parts -> 1 GPU Geometry!)
    const bolster = new THREE.BoxGeometry(2.10, 0.24, 0.92);
    const sideFrame = new THREE.BoxGeometry(0.12, 0.18, 3.15);
    const airBag = new THREE.CylinderGeometry(0.28, 0.30, 0.22, 14);
    const coilSpring = new THREE.CylinderGeometry(0.085, 0.085, 0.24, 10);
    const axleBox = new THREE.BoxGeometry(0.14, 0.20, 0.28);
    const bearingCap = new THREE.CylinderGeometry(0.09, 0.09, 0.16, 10);
    bearingCap.rotateZ(Math.PI / 2);
    const yawDamper = new THREE.CylinderGeometry(0.045, 0.045, 1.05, 8);
    yawDamper.rotateX(Math.PI / 2);
    const vertDamper = new THREE.CylinderGeometry(0.036, 0.036, 0.28, 8);
    const tractionMotor = new THREE.CylinderGeometry(0.28, 0.28, 0.78, 12);
    tractionMotor.rotateZ(Math.PI / 2);
    const gearbox = new THREE.BoxGeometry(0.22, 0.42, 0.52);
    const antiRollBar = new THREE.CylinderGeometry(0.032, 0.032, 1.88, 8);
    antiRollBar.rotateZ(Math.PI / 2);
    const sanderPipe = new THREE.CylinderGeometry(0.018, 0.018, 0.44, 6);
    sanderPipe.rotateX(0.35);
    const caliper = new THREE.BoxGeometry(0.18, 0.24, 0.16);

    // Wheel tread radius = 0.460m; profiled rail head top = +0.105m => exact axle contact centerline y = 0.565m!
    const frameParts = [
      B.bakePart(bolster, { position: [0, 0.72, 0], color: 0x1e293b }),
      B.bakePart(antiRollBar, { position: [0, 0.60, 0], color: 0xf59e0b }),
      B.bakePart(airBag, { position: [-0.86, 0.85, 0], color: 0x0f172a }),
      B.bakePart(airBag, { position: [0.86, 0.85, 0], color: 0x0f172a })
    ];

    [-1.18, 1.18].forEach((az) => {
      frameParts.push(
        B.bakePart(tractionMotor, { position: [0, 0.58, az * 0.68], color: 0x1e293b }),
        B.bakePart(gearbox, { position: [0.38, 0.565, az], color: 0x475569 })
      );
    });

    [-0.88, 0.88].forEach((sx) => {
      frameParts.push(
        B.bakePart(sideFrame, { position: [sx, 0.75, 0], color: 0x1e293b }),
        B.bakePart(yawDamper, { position: [sx * 1.04, 0.77, 0], color: 0xf59e0b })
      );
      [-1.18, 1.18].forEach((az) => {
        frameParts.push(
          B.bakePart(axleBox, { position: [sx, 0.565, az], color: 0x1e293b }),
          B.bakePart(bearingCap, { position: [sx * 1.03, 0.565, az], color: 0xf59e0b }),
          B.bakePart(coilSpring, { position: [sx, 0.66, az - 0.16], color: 0x475569 }),
          B.bakePart(coilSpring, { position: [sx, 0.66, az + 0.16], color: 0x475569 }),
          B.bakePart(vertDamper, { position: [sx * 1.06, 0.66, az + (az < 0 ? -0.26 : 0.26)], color: 0xf59e0b }),
          B.bakePart(sanderPipe, { position: [sx * 0.82, 0.34, az + (az < 0 ? -0.54 : 0.54)], color: 0x475569 }),
          B.bakePart(caliper, { position: [sx * 0.45, 0.565, az + (az < 0 ? 0.24 : -0.24)], color: 0xef4444 })
        );
      });
    });
    this.mergedBogieFrameGeo = B.mergeParts(frameParts);

    const axleShaft = new THREE.CylinderGeometry(0.095, 0.095, 1.72, 12);
    axleShaft.rotateZ(Math.PI / 2);
    const wheelTread = new THREE.CylinderGeometry(0.46, 0.46, 0.135, 22);
    wheelTread.rotateZ(Math.PI / 2);
    const wheelFlange = new THREE.CylinderGeometry(0.50, 0.50, 0.032, 22);
    wheelFlange.rotateZ(Math.PI / 2);
    const wheelHub = new THREE.CylinderGeometry(0.16, 0.16, 0.16, 12);
    wheelHub.rotateZ(Math.PI / 2);
    const wheelSpokeBar = new THREE.BoxGeometry(0.155, 0.68, 0.065);
    const brakeDisc = new THREE.CylinderGeometry(0.36, 0.36, 0.07, 16);
    brakeDisc.rotateZ(Math.PI / 2);

    const wsParts = [B.bakePart(axleShaft, { color: 0x1e293b })];
    [-0.72, 0.72].forEach((wx) => {
      wsParts.push(
        B.bakePart(wheelTread, { position: [wx, 0, 0], color: 0xe2e8f0 }),
        B.bakePart(wheelFlange, { position: [wx + (wx < 0 ? 0.06 : -0.06), 0, 0], color: 0xe2e8f0 }),
        B.bakePart(wheelHub, { position: [wx, 0, 0], color: 0x334155 }),
        B.bakePart(wheelSpokeBar, { position: [wx, 0, 0], color: 0x334155 }),
        B.bakePart(wheelSpokeBar, { position: [wx, 0, 0], rotation: [Math.PI * 0.5, 0, 0], color: 0x334155 }),
        B.bakePart(brakeDisc, { position: [wx * 0.55, 0, 0], color: 0x94a3b8 })
      );
    });
    this.mergedWheelsetGeo = B.mergeParts(wsParts);

    [
      bolster, sideFrame, airBag, coilSpring, axleBox, bearingCap, yawDamper,
      vertDamper, tractionMotor, gearbox, antiRollBar, sanderPipe, caliper,
      axleShaft, wheelTread, wheelFlange, wheelHub, wheelSpokeBar, brakeDisc
    ].forEach(g => g.dispose());
  }

  createBogie() {
    const bogie = new THREE.Group();
    const frameMesh = new THREE.Mesh(this.mergedBogieFrameGeo, this.bogieVertexMat);
    frameMesh.castShadow = true;

    const axleF = new THREE.Mesh(this.mergedWheelsetGeo, this.wheelVertexMat);
    axleF.position.set(0, 0.565, -1.18);
    const axleR = new THREE.Mesh(this.mergedWheelsetGeo, this.wheelVertexMat);
    axleR.position.set(0, 0.565, 1.18);

    bogie.add(frameMesh, axleF, axleR);
    bogie.userData = { axleF, axleR };
    return bogie;
  }

  initHighwayTrafficPool() {
    const B = GeometryBatcher;
    const colors = [0xef4444, 0x0284c7, 0x10b981, 0xf59e0b, 0xf8fafc, 0x7c3aed, 0x0ea5e9, 0xe11d48];

    const wheelGeo = new THREE.CylinderGeometry(0.48, 0.48, 0.34, 16);
    wheelGeo.rotateZ(Math.PI / 2);
    const hubGeo = new THREE.CylinderGeometry(0.30, 0.30, 0.36, 12);
    hubGeo.rotateZ(Math.PI / 2);
    const discGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.35, 10);
    discGeo.rotateZ(Math.PI / 2);
    const lugCapGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.39, 8);
    lugCapGeo.rotateZ(Math.PI / 2);
    const spokeGeo = new THREE.BoxGeometry(0.38, 0.54, 0.055);
    const mudflapGeo = new THREE.BoxGeometry(0.38, 0.46, 0.04);
    const mirrorArmGeo = new THREE.BoxGeometry(0.32, 0.04, 0.04);
    const mirrorHeadGeo = new THREE.BoxGeometry(0.08, 0.28, 0.14);
    const licensePlateGeo = new THREE.BoxGeometry(0.52, 0.14, 0.04);

    // Pre-bake 1 single wheel+rim+disc+3spokes unit so adding 4-8 wheels is super fast
    const singleWheelUnit = B.mergeParts([
      B.bakePart(wheelGeo, { color: 0x111827 }),
      B.bakePart(hubGeo, { color: 0xe2e8f0 }),
      B.bakePart(discGeo, { color: 0x0f172a }),
      B.bakePart(lugCapGeo, { color: 0x0f172a }),
      B.bakePart(spokeGeo, { color: 0xe2e8f0 }),
      B.bakePart(spokeGeo, { rotation: [Math.PI / 3, 0, 0], color: 0xe2e8f0 }),
      B.bakePart(spokeGeo, { rotation: [-Math.PI / 3, 0, 0], color: 0xe2e8f0 })
    ]);

    const addBox = (arr, size, pos, col = null, rot = null) => {
      const g = new THREE.BoxGeometry(size[0], size[1], size[2]);
      arr.push(B.bakePart(g, { position: pos, rotation: rot, color: col }));
      g.dispose();
    };
    const addCyl = (arr, args, pos, col, rot = null, scl = null) => {
      const g = new THREE.CylinderGeometry(...args);
      arr.push(B.bakePart(g, { position: pos, rotation: rot, scale: scl, color: col }));
      g.dispose();
    };
    const pushWheels = (bodyParts, xSpan, zPositions, scale = 1.0, hasMudflaps = false) => {
      zPositions.forEach((wz) => {
        [-xSpan, xSpan].forEach((wx) => {
          const wClone = singleWheelUnit.clone();
          wClone.scale(scale, scale, scale);
          wClone.translate(wx, 0.48 * scale, wz);
          bodyParts.push(wClone);
          if (hasMudflaps) {
            bodyParts.push(B.bakePart(mudflapGeo, {
              position: [wx, 0.36 * scale, wz + 0.56 * scale],
              scale,
              color: 0x0f172a
            }));
          }
        });
      });
    };

    for (let i = 0; i < 28; i++) {
      const vType = i % 6;
      const paintHex = colors[i % colors.length];
      const bodyParts = [];
      const glassParts = [];
      const emissiveParts = [];

      if (vType === 0) {
        addBox(bodyParts, [2.38, 2.42, 2.45], [0, 1.72, -4.25], 0xf8fafc);
        addBox(bodyParts, [2.32, 0.65, 1.55], [0, 3.12, -3.95], 0xf8fafc, [0.32, 0, 0]);
        addBox(bodyParts, [2.42, 0.16, 0.28], [0, 2.76, -5.52], 0x0f172a);
        addBox(glassParts, [2.22, 0.86, 0.12], [0, 2.25, -5.46]);
        addBox(glassParts, [2.42, 0.76, 1.45], [0, 2.24, -4.45]);
        addBox(bodyParts, [1.72, 0.96, 0.14], [0, 1.28, -5.46], 0xe2e8f0);
        for (let gy = 0.92; gy <= 1.62; gy += 0.18) {
          addBox(bodyParts, [1.62, 0.05, 0.18], [0, gy, -5.47], 0x0f172a);
        }
        addBox(bodyParts, [2.46, 0.44, 0.35], [0, 0.64, -5.42], 0x0f172a);
        [-1.28, 1.28].forEach((mx) => {
          bodyParts.push(
            B.bakePart(mirrorArmGeo, { position: [mx, 2.35, -5.15], color: 0x0f172a }),
            B.bakePart(mirrorHeadGeo, { position: [mx * 1.08, 2.28, -5.15], color: 0xe2e8f0 })
          );
          addCyl(bodyParts, [0.08, 0.08, 2.85, 8], [mx * 0.88, 2.25, -2.92], 0xe2e8f0);
          addCyl(bodyParts, [0.32, 0.32, 1.65, 10], [mx * 0.78, 0.72, -2.6], 0xe2e8f0, [Math.PI / 2, 0, 0]);
        });
        addBox(bodyParts, [2.08, 0.38, 11.4], [0, 0.72, 0.1], 0x0f172a);
        addBox(bodyParts, [1.95, 0.68, 0.18], [0, 0.48, -1.05], 0x0f172a);
        addBox(bodyParts, [2.36, 0.24, 3.6], [0, 0.68, 0.85], 0xe2e8f0);
        addBox(bodyParts, [2.44, 2.68, 8.5], [0, 2.22, 1.55], paintHex);
        addBox(bodyParts, [2.48, 0.38, 6.2], [0, 2.35, 1.55], 0xf8fafc);
        for (let rz = -2.2; rz <= 5.2; rz += 0.95) {
          addBox(bodyParts, [2.48, 2.52, 0.10], [0, 2.22, rz], paintHex);
        }
        [-0.75, -0.25, 0.25, 0.75].forEach((lx) => {
          addBox(bodyParts, [0.05, 2.56, 0.08], [lx, 2.22, 5.82], 0xe2e8f0);
        });
        pushWheels(bodyParts, 1.12, [-4.35, -1.65, 3.45, 4.68], 1.0, true);
      } else if (vType === 1) {
        addBox(bodyParts, [2.46, 3.02, 11.2], [0, 2.02, 0], paintHex);
        addBox(bodyParts, [2.49, 0.28, 10.8], [0, 1.65, 0], 0xf8fafc, [0.04, 0, 0]);
        addBox(bodyParts, [2.50, 1.12, 10.6], [0, 2.54, 0], 0x0f172a);
        addBox(glassParts, [2.52, 0.96, 10.1], [0, 2.56, 0]);
        addBox(glassParts, [2.34, 0.92, 0.14], [0, 2.06, -5.58]);
        addBox(bodyParts, [2.48, 0.24, 0.22], [0, 2.64, -5.56], paintHex);
        addBox(emissiveParts, [1.45, 0.14, 0.24], [0, 2.64, -5.57], 0xf59e0b);
        addBox(glassParts, [2.28, 0.56, 0.14], [0, 3.04, -5.54], null, [0.18, 0, 0]);
        addBox(bodyParts, [2.54, 1.32, 0.42], [0, 2.48, -3.65], 0xe2e8f0, [-0.38, 0, 0]);
        addBox(bodyParts, [2.53, 0.16, 4.8], [0, 3.12, -1.4], 0xe2e8f0);
        for (let bz = -2.2; bz <= 1.2; bz += 1.65) {
          addBox(bodyParts, [2.49, 0.78, 1.48], [0, 1.12, bz], 0x0f172a);
          addBox(bodyParts, [2.50, 0.72, 1.42], [0, 1.12, bz], paintHex);
        }
        addBox(bodyParts, [1.82, 0.34, 2.8], [0, 3.66, -1.6], 0xf8fafc);
        addBox(bodyParts, [1.82, 0.34, 2.8], [0, 3.66, 2.1], 0xf8fafc);
        addBox(bodyParts, [2.44, 0.24, 0.45], [0, 3.56, 5.48], paintHex);
        addBox(emissiveParts, [1.65, 0.08, 0.48], [0, 3.58, 5.49], 0xf43f5e);
        addBox(bodyParts, [1.85, 0.85, 0.14], [0, 1.25, 5.58], 0x0f172a);
        [-1.32, 1.32].forEach((mx) => {
          addBox(bodyParts, [0.24, 0.12, 0.55], [mx, 2.48, -5.72], paintHex);
          addBox(bodyParts, [0.14, 0.44, 0.22], [mx * 1.06, 2.32, -5.92], 0x0f172a);
        });
        pushWheels(bodyParts, 1.15, [-3.55, 2.55, 3.82], 1.0, true);
      } else if (vType === 3) {
        addBox(bodyParts, [2.15, 2.05, 1.95], [0, 1.48, -3.25], 0xfacc15);
        addBox(glassParts, [1.98, 0.76, 0.10], [0, 1.88, -4.22]);
        addBox(glassParts, [2.19, 0.68, 1.15], [0, 1.86, -3.35]);
        addBox(bodyParts, [2.10, 0.52, 1.2], [0, 2.68, -3.1], 0xfacc15, [0.28, 0, 0]);
        addBox(bodyParts, [2.32, 2.48, 6.2], [0, 1.96, 1.05], 0xe2e8f0);
        addBox(bodyParts, [2.36, 0.32, 5.8], [0, 1.85, 1.05], paintHex);
        addBox(bodyParts, [1.88, 0.32, 8.2], [0, 0.65, -0.05], 0x0f172a);
        pushWheels(bodyParts, 1.02, [-3.25, 1.85, 2.95], 0.88, true);
      } else if (vType === 4) {
        addBox(bodyParts, [2.25, 2.18, 2.15], [0, 1.56, -3.65], 0xdc2626);
        addBox(glassParts, [2.08, 0.78, 0.10], [0, 1.96, -4.72]);
        addCyl(bodyParts, [1.12, 1.12, 6.8, 18], [0, 1.92, 1.15], 0xf8fafc, [Math.PI / 2, 0, 0], [1.08, 1.0, 0.88]);
        addBox(bodyParts, [2.44, 0.36, 6.6], [0, 1.92, 1.15], 0xdc2626);
        addBox(bodyParts, [0.55, 0.08, 6.4], [0, 2.94, 1.15], 0xe2e8f0);
        addBox(bodyParts, [1.96, 0.34, 8.8], [0, 0.68, -0.1], 0x0f172a);
        pushWheels(bodyParts, 1.06, [-3.65, 2.15, 3.35], 0.92, true);
      } else {
        const isPolice = (vType === 5);
        const mainCol = isPolice ? 0xf8fafc : paintHex;
        addBox(bodyParts, [1.92, 0.74, 4.62], [0, 0.72, 0], mainCol);
        addBox(bodyParts, [1.78, 0.14, 1.45], [0, 1.10, -1.48], isPolice ? 0x3b82f6 : mainCol, [0.08, 0, 0]);
        addBox(bodyParts, [1.68, 0.66, 2.55], [0, 1.38, 0.28], mainCol);
        addBox(glassParts, [1.72, 0.52, 2.35], [0, 1.36, 0.26]);
        addBox(glassParts, [1.62, 0.58, 0.08], [0, 1.34, -0.96], null, [-0.52, 0, 0]);
        addBox(glassParts, [1.58, 0.54, 0.08], [0, 1.34, 1.52], null, [0.48, 0, 0]);
        [-0.68, 0.68].forEach((rx) => {
          addBox(bodyParts, [0.06, 0.06, 2.1], [rx, 1.74, 0.32], 0xe2e8f0);
        });
        if (isPolice) {
          addBox(emissiveParts, [1.95, 0.22, 4.4], [0, 0.76, 0], 0x3b82f6);
          addBox(bodyParts, [1.22, 0.06, 0.28], [0, 1.76, 0.05], 0xe2e8f0);
          addBox(emissiveParts, [0.52, 0.10, 0.24], [-0.30, 1.82, 0.05], 0x3b82f6);
          addBox(emissiveParts, [0.52, 0.10, 0.24], [0.30, 1.82, 0.05], 0xf43f5e);
        }
        addBox(bodyParts, [0.06, 0.14, 0.22], [0, 1.76, 1.25], mainCol);
        [-1.02, 1.02].forEach((mx) => {
          addBox(bodyParts, [0.18, 0.12, 0.14], [mx, 1.16, -0.68], mainCol);
          addBox(bodyParts, [0.04, 0.05, 0.22], [mx * 0.96, 1.02, -0.15], 0xe2e8f0);
          addBox(bodyParts, [0.04, 0.05, 0.22], [mx * 0.96, 1.02, 0.72], 0xe2e8f0);
        });
        addBox(bodyParts, [1.32, 0.38, 0.12], [0, 0.72, -2.28], 0x0f172a);
        addBox(bodyParts, [1.28, 0.06, 0.14], [0, 0.78, -2.29], 0xe2e8f0);
        addBox(bodyParts, [1.94, 0.12, 0.24], [0, 0.38, -2.24], 0x0f172a);
        addBox(bodyParts, [1.88, 0.22, 0.18], [0, 0.44, 2.26], 0x0f172a);
        pushWheels(bodyParts, 0.94, [-1.45, 1.45], 0.74, false);
      }

      const zFront = (vType === 2 || vType === 5) ? -2.31 : (vType === 3 ? -4.23 : (vType === 4 ? -4.73 : -5.58));
      const zRear = (vType === 2 || vType === 5) ? 2.31 : (vType === 3 ? 4.16 : (vType === 4 ? 4.55 : 5.58));
      emissiveParts.push(
        B.bakePart(licensePlateGeo, { position: [0, 0.54, zFront - 0.02], color: 0xf8fafc }),
        B.bakePart(licensePlateGeo, { position: [0, 0.62, zRear + 0.02], color: 0xf8fafc })
      );
      [-0.74, 0.74].forEach((lx) => {
        addBox(emissiveParts, [0.36, 0.18, 0.08], [lx, 0.82, zFront], 0xfef08a);
        addBox(emissiveParts, [0.38, 0.05, 0.09], [lx, 0.92, zFront], 0x38bdf8);
        addBox(emissiveParts, [0.42, 0.20, 0.08], [lx, 0.86, zRear], 0xf43f5e);
      });

      const grp = new THREE.Group();
      const bodyGeo = B.mergeParts(bodyParts);
      const glassGeo = B.mergeParts(glassParts);
      const emGeo = B.mergeParts(emissiveParts);

      const bMesh = new THREE.Mesh(bodyGeo, this.highwayBodyMat);
      bMesh.castShadow = true;
      grp.add(bMesh);
      if (glassGeo) grp.add(new THREE.Mesh(glassGeo, this.highwayGlassMat));
      if (emGeo) grp.add(new THREE.Mesh(emGeo, this.highwayEmissiveMat));

      // Indonesian Left-Hand Traffic (`Lajur Kiri`): dir=+1 (-Z) at +17.4m, oncoming dir=-1 (+Z) at +21.6m from corridor center!
      const dir = i % 2 === 0 ? 1 : -1;
      const baseSpeed = (vType === 0 || vType === 4 ? 21 : (vType === 1 || vType === 3 ? 27 : 33));
      this.highwayGroup.add(grp);
      this.highwayVehicles.push({
        mesh: grp,
        sCoord: i * 58 + Math.random() * 18,
        dir,
        laneFromMid: dir > 0 ? 17.4 : 21.6,
        baseSpeedMps: baseSpeed,
        speedMps: baseSpeed * dir
      });
    }

    [wheelGeo, hubGeo, discGeo, lugCapGeo, spokeGeo, mudflapGeo, mirrorArmGeo, mirrorHeadGeo, licensePlateGeo, singleWheelUnit].forEach(g => g.dispose());
  }

  _getLiveryMaterialSet(liveryKey, isPlayer = false) {
    const cache = isPlayer ? this.playerMatCache : this.aiMatCache;
    if (cache.has(liveryKey)) return cache.get(liveryKey);

    const cfg = GAME_PREMISE_CARTRIDGE.liveries[liveryKey] || GAME_PREMISE_CARTRIDGE.liveries.WHOOSH;
    const locoTex = ProceduralAssetFactory.createRollingStockTexture(cfg, true);
    const coachTex = ProceduralAssetFactory.createRollingStockTexture(cfg, false);
    const noseTexFwd = ProceduralAssetFactory.createBulletNoseTexture(cfg, false);
    const noseTexTail = ProceduralAssetFactory.createBulletNoseTexture(cfg, true);

    const set = {
      locoBodyMat: new THREE.MeshPhysicalMaterial({
        map: locoTex,
        bumpMap: this.microBump,
        bumpScale: 0.014,
        metalness: 0.56,
        roughness: 0.16,
        clearcoat: 0.88,
        clearcoatRoughness: 0.09,
        envMapIntensity: 1.45,
        side: THREE.DoubleSide
      }),
      coachBodyMat: new THREE.MeshPhysicalMaterial({
        map: coachTex,
        bumpMap: this.microBump,
        bumpScale: 0.014,
        metalness: 0.56,
        roughness: 0.16,
        clearcoat: 0.88,
        clearcoatRoughness: 0.09,
        envMapIntensity: 1.45,
        side: THREE.DoubleSide
      }),
      noseSurfaceMatFwd: new THREE.MeshPhysicalMaterial({
        map: noseTexFwd,
        metalness: 0.58,
        roughness: 0.14,
        clearcoat: 0.92,
        clearcoatRoughness: 0.07,
        envMapIntensity: 1.5,
        side: THREE.DoubleSide
      }),
      noseSurfaceMatTail: new THREE.MeshPhysicalMaterial({
        map: noseTexTail,
        metalness: 0.58,
        roughness: 0.14,
        clearcoat: 0.92,
        clearcoatRoughness: 0.07,
        envMapIntensity: 1.5,
        side: THREE.DoubleSide
      }),
      noseMat: new THREE.MeshPhysicalMaterial({
        color: cfg.bodyColor,
        metalness: 0.58,
        roughness: 0.15,
        clearcoat: 0.85,
        clearcoatRoughness: 0.10,
        envMapIntensity: 1.4
      }),
      stripe1Mat: new THREE.MeshPhysicalMaterial({
        color: cfg.stripe1,
        metalness: 0.54,
        roughness: 0.18,
        clearcoat: 0.80,
        envMapIntensity: 1.35
      }),
      stripe2Mat: new THREE.MeshStandardMaterial({
        color: cfg.stripe2,
        metalness: 0.55,
        roughness: 0.22,
        envMapIntensity: 1.25
      }),
      frameMat: new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        metalness: 0.80,
        roughness: 0.30,
        envMapIntensity: 1.2
      }),
      glassMat: new THREE.MeshPhysicalMaterial({
        color: 0x050e1c,
        metalness: 0.90,
        roughness: 0.04,
        clearcoat: 1.0,
        clearcoatRoughness: 0.03,
        envMapIntensity: 1.8
      })
    };

    cache.set(liveryKey, set);
    return set;
  }

  createContainerFlatcarWagon(bogieMat, wagonIdx) {
    const B = GeometryBatcher;
    const parts = [];
    const addBox = (size, pos, col, rot = null) => {
      const g = new THREE.BoxGeometry(size[0], size[1], size[2]);
      parts.push(B.bakePart(g, { position: pos, rotation: rot, color: col }));
      g.dispose();
    };
    const addCyl = (args, pos, col, rot = null) => {
      const g = new THREE.CylinderGeometry(...args);
      parts.push(B.bakePart(g, { position: pos, rotation: rot, color: col }));
      g.dispose();
    };

    addBox([2.52, 0.34, 19.6], [0, 1.04, 0], 0x334155);
    addBox([0.14, 0.42, 19.8], [-1.24, 1.06, 0], 0xfacc15);
    addBox([0.14, 0.42, 19.8], [1.24, 1.06, 0], 0xfacc15);
    addCyl([0.22, 0.22, 1.85, 10], [0.45, 0.74, 0], 0xdc2626, [Math.PI / 2, 0, 0]);
    addCyl([0.28, 0.28, 0.06, 12], [-1.34, 1.15, 8.8], 0xfacc15, [0, 0, Math.PI / 2]);
    [-8.8, -3.2, 0, 3.2, 8.8].forEach((tz) => {
      addBox([2.58, 0.22, 0.24], [0, 1.02, tz], 0x334155);
    });

    const palettes = [
      { base: 0x0284c7, stripe: 0xf97316 },
      { base: 0xdc2626, stripe: 0xf8fafc },
      { base: 0x15803d, stripe: 0xfacc15 },
      { base: 0x0ea5e9, stripe: 0x0f172a },
      { base: 0xd97706, stripe: 0xf8fafc }
    ];

    const addContainer = (lenZ, zPos, pal) => {
      const y0 = 2.48;
      addBox([2.44, 2.54, lenZ], [0, y0, zPos], pal.base);
      addBox([2.49, 0.38, lenZ * 0.72], [0, y0 + 0.25, zPos], pal.stripe);
      for (let rz = -lenZ * 0.44; rz <= lenZ * 0.44; rz += 0.85) {
        addBox([2.48, 2.38, 0.08], [0, y0, zPos + rz], pal.base);
      }
      [-1.18, 1.18].forEach((cx) => {
        [-lenZ * 0.49, lenZ * 0.49].forEach((cz) => {
          addBox([0.16, 0.16, 0.18], [cx, y0 + 1.22, zPos + cz], 0xfacc15);
        });
      });
      [-0.72, -0.24, 0.24, 0.72].forEach((lx) => {
        addBox([0.045, 2.42, 0.06], [lx, y0, zPos + lenZ * 0.50], 0xe2e8f0);
      });
    };

    if (wagonIdx % 2 === 0) {
      addContainer(12.2, 0, palettes[wagonIdx % palettes.length]);
    } else {
      addContainer(6.05, -3.4, palettes[wagonIdx % palettes.length]);
      addContainer(6.05, 3.4, palettes[(wagonIdx + 2) % palettes.length]);
    }

    const merged = B.mergeParts(parts);
    merged.userData.sharedGeo = false; // Disposed cleanly when consist changes
    const mesh = new THREE.Mesh(merged, this.highwayBodyMat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  createTankerWagon(bogieMat, wagonIdx) {
    const B = GeometryBatcher;
    const parts = [];
    const addBox = (size, pos, col, rot = null) => {
      const g = new THREE.BoxGeometry(size[0], size[1], size[2]);
      parts.push(B.bakePart(g, { position: pos, rotation: rot, color: col }));
      g.dispose();
    };
    const addCyl = (args, pos, col, rot = null) => {
      const g = new THREE.CylinderGeometry(...args);
      parts.push(B.bakePart(g, { position: pos, rotation: rot, color: col }));
      g.dispose();
    };

    addBox([2.45, 0.32, 18.8], [0, 1.02, 0], 0x1e293b);
    [-6.2, 0, 6.2].forEach((sz) => {
      addBox([2.48, 0.65, 0.45], [0, 1.42, sz], 0x1e293b);
      addCyl([1.31, 1.31, 0.14, 20], [0, 2.42, sz], 0x1e293b, [Math.PI / 2, 0, 0]);
    });

    addCyl([1.28, 1.28, 16.4, 24], [0, 2.42, 0], 0xf8fafc, [Math.PI / 2, 0, 0]);
    const capGeo = new THREE.SphereGeometry(1.28, 18, 14);
    parts.push(
      B.bakePart(capGeo, { position: [0, 2.42, -8.2], scale: [1, 1, 0.45], color: 0x16a34a }),
      B.bakePart(capGeo, { position: [0, 2.42, 8.2], scale: [1, 1, 0.45], color: 0x16a34a })
    );
    capGeo.dispose();

    const bandCol = wagonIdx % 2 === 0 ? 0x16a34a : 0x0284c7;
    addBox([2.58, 0.46, 15.8], [0, 2.42, 0], bandCol);
    addBox([2.62, 0.55, 0.55], [0, 2.42, 0], 0xdc2626);
    addCyl([0.45, 0.48, 0.46, 14], [0, 3.76, 0], 0x1e293b);
    addCyl([0.10, 0.12, 0.35, 8], [0, 3.85, -1.2], 0xdc2626);
    addCyl([0.10, 0.12, 0.35, 8], [0, 3.85, 1.2], 0xfacc15);
    addBox([0.72, 0.08, 14.8], [0, 3.72, 0], 0x1e293b);
    [-0.36, 0.36].forEach((rx) => {
      addBox([0.04, 0.45, 14.6], [rx, 3.96, 0], 0xfacc15);
    });
    addBox([0.05, 2.75, 0.05], [-1.34, 2.40, -0.22], 0xfacc15);
    addBox([0.05, 2.75, 0.05], [-1.34, 2.40, 0.22], 0xfacc15);
    for (let ly = 1.2; ly <= 3.6; ly += 0.34) {
      addBox([0.06, 0.04, 0.44], [-1.34, ly, 0], 0xfacc15);
    }
    addCyl([0.18, 0.18, 0.55, 10], [0, 0.78, 0], 0xdc2626);

    const merged = B.mergeParts(parts);
    merged.userData.sharedGeo = false;
    const mesh = new THREE.Mesh(merged, this.highwayBodyMat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  createCoalHopperWagon() {
    const B = GeometryBatcher;
    const parts = [];
    const addBox = (size, pos, col) => {
      const g = new THREE.BoxGeometry(size[0], size[1], size[2]);
      parts.push(B.bakePart(g, { position: pos, color: col }));
      g.dispose();
    };

    addBox([2.68, 2.15, 18.2], [0, 2.12, 0], 0x7c2d12);
    addBox([2.78, 0.14, 18.4], [0, 3.22, 0], 0x0f172a);
    addBox([2.72, 0.16, 17.8], [0, 1.32, 0], 0xfacc15);
    for (let rz = -7.5; rz <= 7.5; rz += 2.15) {
      addBox([2.76, 2.18, 0.18], [0, 2.12, rz], 0x0f172a);
    }
    const chuteGeo = new THREE.ConeGeometry(1.15, 0.72, 4);
    [-4.5, 0, 4.5].forEach((cz) => {
      parts.push(B.bakePart(chuteGeo, { position: [0, 0.76, cz], rotation: [Math.PI, Math.PI / 4, 0], color: 0x0f172a }));
    });
    chuteGeo.dispose();

    addBox([2.48, 0.46, 17.6], [0, 3.30, 0], 0x111827);
    const peakGeo = new THREE.ConeGeometry(1.22, 0.62, 7);
    [-5.2, 0, 5.2].forEach((mz) => {
      parts.push(B.bakePart(peakGeo, { position: [0, 3.64, mz], scale: [1.0, 1.0, 2.2], color: 0x111827 }));
    });
    peakGeo.dispose();

    const merged = B.mergeParts(parts);
    merged.userData.sharedGeo = false;
    const mesh = new THREE.Mesh(merged, this.bogieVertexMat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  createPanoramicCoach(stripeMat, bogieMat, glassMat, carLen) {
    const B = GeometryBatcher;
    const pGrp = new THREE.Group();
    const parts = [];
    const emParts = [];

    const addBox = (arr, size, pos, col) => {
      const g = new THREE.BoxGeometry(size[0], size[1], size[2]);
      arr.push(B.bakePart(g, { position: pos, color: col }));
      g.dispose();
    };

    addBox(parts, [2.68, 0.42, carLen - 1.2], [0, 0.96, 0], 0x1e293b);
    addBox(parts, [2.86, 1.35, carLen - 0.55], [0, 1.55, 0], 0x0f172a);
    addBox(parts, [2.90, 0.15, carLen - 0.52], [0, 2.12, 0], 0xfacc15);
    addBox(parts, [2.88, 0.08, carLen - 0.52], [0, 1.05, 0], 0xfacc15);
    [-carLen * 0.42, carLen * 0.42].forEach((az) => {
      addBox(parts, [1.85, 0.24, 2.2], [0, 3.88, az], 0x0f172a);
    });

    const lampGeo = new THREE.SphereGeometry(0.10, 8, 8);
    for (let sz = -6.0; sz <= 6.0; sz += 2.0) {
      addBox(parts, [0.82, 0.85, 0.65], [-0.82, 2.45, sz], 0x991b1b);
      addBox(parts, [0.82, 0.85, 0.65], [0.82, 2.45, sz], 0x991b1b);
      emParts.push(B.bakePart(lampGeo, { position: [0, 3.62, sz], color: 0xfef08a }));
    }
    lampGeo.dispose();

    for (let mz = -7.5; mz <= 7.5; mz += 2.5) {
      addBox(parts, [2.82, 1.66, 0.14], [0, 3.02, mz], 0xfacc15);
    }

    const bodyGeo = B.mergeParts(parts);
    bodyGeo.userData.sharedGeo = false;
    const emGeo = B.mergeParts(emParts);
    emGeo.userData.sharedGeo = false;

    const bodyMesh = new THREE.Mesh(bodyGeo, this.highwayBodyMat);
    bodyMesh.castShadow = true;
    bodyMesh.receiveShadow = true;

    const glassVault = new THREE.Mesh(new THREE.BoxGeometry(2.76, 1.62, carLen - 0.8), glassMat);
    glassVault.position.set(0, 3.02, 0);

    pGrp.add(bodyMesh, new THREE.Mesh(emGeo, this.highwayEmissiveMat), glassVault);
    return pGrp;
  }

  initParticlePools() {
    const smokeGeo = new THREE.SphereGeometry(0.45, 7, 7);
    smokeGeo.userData.sharedGeo = true;
    for (let i = 0; i < 28; i++) {
      const mat = new THREE.MeshBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.0, depthWrite: false });
      const mesh = new THREE.Mesh(smokeGeo, mat);
      mesh.visible = false;
      this.particleGroup.add(mesh);
      this.smokePool.push({ mesh, life: 0, maxLife: 1.2 + Math.random() * 0.6, vel: new THREE.Vector3() });
    }

    const sparkGeo = new THREE.BoxGeometry(0.08, 0.08, 0.28);
    sparkGeo.userData.sharedGeo = true;
    const sparkMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24 });
    for (let i = 0; i < 24; i++) {
      const mesh = new THREE.Mesh(sparkGeo, sparkMat);
      mesh.visible = false;
      this.particleGroup.add(mesh);
      this.sparkPool.push({ mesh, life: 0, vel: new THREE.Vector3() });
    }
  }

  applyOriginRebase(deltaZ) {
    for (const p of this.smokePool) {
      if (p.life > 0) p.mesh.position.z += deltaZ;
    }
    for (const s of this.sparkPool) {
      if (s.life > 0) s.mesh.position.z += deltaZ;
    }
  }

  buildOncomingAITrain(liveryKey = 'WHOOSH_CIT') {
    this.currentAiLiveryKey = liveryKey;
    const aiCfg = GAME_PREMISE_CARTRIDGE.liveries[liveryKey] || GAME_PREMISE_CARTRIDGE.liveries.WHOOSH;

    this._disposeGroupNonShared(this.aiTrainGroup);
    this.aiVehicles = [];

    const isBullet = (aiCfg.noseType === 'BULLET');
    // Match AI cruise speed realistically to each rolling stock class's maxSpeedKmh!
    this.aiSpeedMps = Math.min(84.0, ((aiCfg.maxSpeedKmh || 150) / 3.6) * 0.86);

    const mats = this._getLiveryMaterialSet(liveryKey, false);
    const {
      locoBodyMat: locoMat,
      coachBodyMat: coachMat,
      noseSurfaceMatFwd: noseSurfaceMat,
      noseMat,
      stripe1Mat: stripeMat,
      stripe2Mat,
      frameMat: bogieMat,
      glassMat
    } = mats;

    const numAiCars = 6;
    const carLen = 20.5;
    const carGap = 1.0;

    for (let i = 0; i < numAiCars; i++) {
      const isLead = (i === 0);
      const isTail = (i === numAiCars - 1);
      const vGroup = new THREE.Group();
      const bodyGroup = new THREE.Group();
      vGroup.add(bodyGroup);

      const bogieFront = this.createBogie();
      const bogieRear = this.createBogie();
      this.aiTrainGroup.add(bogieFront, bogieRear);

      if (isLead && !isBullet) {
        const hoodMesh = new THREE.Mesh(new THREE.BoxGeometry(2.58, 2.58, 12.2), locoMat);
        hoodMesh.position.set(0, 2.52, -1.2);
        hoodMesh.castShadow = true;
        const hoodCrown = new THREE.Mesh(new THREE.BoxGeometry(2.18, 0.28, 12.0), bogieMat);
        hoodCrown.position.set(0, 3.92, -1.2);
        const radWing = new THREE.Mesh(new THREE.BoxGeometry(2.88, 0.52, 4.4), stripeMat);
        radWing.position.set(0, 3.45, -4.6);

        [-4.8, -2.6, -0.4].forEach((fz) => {
          const fan = new THREE.Mesh(new THREE.CylinderGeometry(0.64, 0.68, 0.18, 12), bogieMat);
          fan.position.set(0, 4.08, fz);
          bodyGroup.add(fan);
        });

        const cabMeshF = new THREE.Mesh(new THREE.BoxGeometry(2.90, 2.68, 3.8), noseMat);
        cabMeshF.position.set(0, 2.56, 6.8);
        cabMeshF.castShadow = true;
        const cabStripeF = new THREE.Mesh(new THREE.BoxGeometry(2.94, 0.56, 3.85), stripeMat);
        cabStripeF.position.set(0, 1.68, 6.8);
        const frontNoseHood = new THREE.Mesh(new THREE.BoxGeometry(2.86, 1.34, 1.35), stripeMat);
        frontNoseHood.position.set(0, 1.72, 9.25);
        frontNoseHood.rotation.x = -0.16;

        [-0.66, 0.66].forEach((wx) => {
          const winPane = new THREE.Mesh(new THREE.BoxGeometry(1.06, 0.80, 0.16), glassMat);
          winPane.position.set(wx, 3.10, 8.72);
          winPane.rotation.x = 0.18;
          const ditchLamp = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), this.sharedLampWhiteMat);
          ditchLamp.position.set(wx * 1.25, 1.28, 9.88);
          bodyGroup.add(winPane, ditchLamp);
        });
        const headLightMain = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.24, 0.14), this.sharedLampWhiteMat);
        headLightMain.position.set(0, 3.68, 8.75);

        const fuelTank = new THREE.Mesh(new THREE.BoxGeometry(2.64, 0.72, 6.2), bogieMat);
        fuelTank.position.set(0, 0.65, -0.8);
        [-1.36, 1.36].forEach((hx) => {
          const walkway = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.10, 18.8), bogieMat);
          walkway.position.set(hx, 1.26, 0);
          const handrail = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.045, 13.5), noseMat);
          handrail.position.set(hx, 2.08, -1.2);
          bodyGroup.add(walkway, handrail);
        });
        const cowcatcher = new THREE.Mesh(new THREE.BoxGeometry(2.68, 0.56, 0.68), this.sharedCowcatcherRedMat);
        cowcatcher.position.set(0, 0.78, 9.68);
        bodyGroup.add(
          hoodMesh, hoodCrown, radWing, cabMeshF, cabStripeF, frontNoseHood,
          headLightMain, fuelTank, cowcatcher
        );
      } else if (aiCfg.consistType === 'FREIGHT_CONTAINER') {
        bodyGroup.add(this.createContainerFlatcarWagon(bogieMat, i));
      } else if (aiCfg.consistType === 'FREIGHT_TANKER') {
        bodyGroup.add(this.createTankerWagon(bogieMat, i));
      } else if (aiCfg.consistType === 'FREIGHT_COAL') {
        bodyGroup.add(this.createCoalHopperWagon());
      } else if (aiCfg.consistType === 'KAI_PASSENGER' && i === 1) {
        bodyGroup.add(this.createPanoramicCoach(stripeMat, bogieMat, glassMat, carLen));
      } else {
        const bodyLen = (isBullet && (isLead || isTail)) ? 14.5 : (carLen - 0.45);
        const bodyZOff = isBullet ? (isLead ? -3.0 : (isTail ? 3.0 : 0)) : 0;

        const shellGrp = this.createHighDetailCarbodyShell(
          coachMat, noseMat, stripeMat, stripe2Mat, bogieMat, glassMat, bodyLen, false
        );
        shellGrp.position.z = bodyZOff;
        bodyGroup.add(shellGrp);

        if (isBullet && isLead) {
          const noseCone = this.createSculptedBulletNose(noseSurfaceMat, stripeMat, stripe2Mat, glassMat, this.sharedLampWhiteMat, 1);
          noseCone.position.set(0, 0, 4.25);
          bodyGroup.add(noseCone);
        } else if (isBullet && isTail) {
          const tailCone = this.createSculptedBulletNose(noseSurfaceMat, stripeMat, stripe2Mat, glassMat, this.sharedTailRedMat, -1);
          tailCone.position.set(0, 0, -4.25);
          bodyGroup.add(tailCone);
        }
      }

      // Raised pantograph on oncoming Electric AI train kissing Track 2's 5.32m contact wire
      if (isLead && aiCfg.tractionType === 'ELECTRIC') {
        const aiPanto = new THREE.Group();
        aiPanto.position.set(0, 4.02, -3.8);
        const pArmLow = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 0.96, 8), this.sharedChromeMat);
        pArmLow.position.set(0, 0.48, 0.24);
        pArmLow.rotation.x = -0.56;
        const pArmUp = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.04, 0.96, 8), this.sharedChromeMat);
        pArmUp.position.set(0, 0.98, 0.06);
        pArmUp.rotation.x = 0.50;
        const pStrip = new THREE.Mesh(new THREE.BoxGeometry(1.68, 0.05, 0.22), bogieMat);
        pStrip.position.set(0, 1.28, 0);
        aiPanto.add(pArmLow, pArmUp, pStrip);
        bodyGroup.add(aiPanto);
      }

      // Tail red marker lights on non-bullet oncoming AI train
      if (isTail && !isBullet) {
        const tlL = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), this.sharedTailRedMat);
        const tlR = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), this.sharedTailRedMat);
        tlL.position.set(-1.05, 1.95, -carLen * 0.5 + 0.3);
        tlR.position.set(1.05, 1.95, -carLen * 0.5 + 0.3);
        bodyGroup.add(tlL, tlR);
      }

      if (!isTail) {
        if (aiCfg.consistType?.startsWith('FREIGHT_')) {
          const coupler = this.createFreightCoupler();
          coupler.position.set(0, 0, -carLen * 0.5 - 0.50);
          bodyGroup.add(coupler);
        } else {
          const gangway = this.createInterCarGangway(noseMat);
          gangway.position.set(0, 0, -carLen * 0.5 - 0.35);
          bodyGroup.add(gangway);
        }
      }

      this.aiTrainGroup.add(vGroup);
      this.aiVehicles.push({
        vGroup,
        bodyGroup,
        bogieFront,
        bogieRear,
        offsetS: i * (carLen + carGap),
        bogieSpan: 12.6
      });
    }
  }

  getHSRCrossSectionPoint(vNorm, uNose = 0) {
    const theta = vNorm * Math.PI * 2;
    const sinT = Math.sin(theta);
    const cosT = -Math.cos(theta);

    const uEff = Math.max(0, (uNose - 0.12) / 0.88);
    const widthTaper = Math.pow(1.0 - Math.pow(uNose, 1.85), 0.58);
    const halfW = THREE.MathUtils.lerp(0.05, 1.49, widthTaper);

    const canopyArch = Math.sin(Math.min(1.0, uNose / 0.50) * Math.PI) * 0.16 * (1.0 - uNose);
    const topY = THREE.MathUtils.lerp(3.96, 1.22, Math.pow(uNose, 1.28)) + canopyArch;
    const botY = THREE.MathUtils.lerp(0.76, 0.92, Math.pow(uEff, 1.15));

    const midY = (topY + botY) * 0.5;
    const halfH = Math.max(0.04, (topY - botY) * 0.5);

    const expX = THREE.MathUtils.lerp(0.68, 0.95, uNose);
    const expY = THREE.MathUtils.lerp(0.64, 0.95, uNose);

    const sx = Math.sign(sinT) * Math.pow(Math.abs(sinT), expX);
    const sy = Math.sign(cosT) * Math.pow(Math.abs(cosT), expY);

    const px = sx * halfW;
    const py = THREE.MathUtils.clamp(midY + sy * halfH, botY, topY);
    const hNorm = (topY - botY) > 0.001 ? THREE.MathUtils.clamp((py - botY) / (topY - botY), 0.0, 1.0) : 0.5;
    return { x: px, y: py, hNorm };
  }

  createLoftedCarbodyGeometry(bodyLen) {
    const key = `loftBodyFixed_${bodyLen.toFixed(2)}`;
    if (this[key]) return this[key];

    const lengthSegs = 6;
    const radialSegs = 36;
    const verts = [];
    const uvs = [];
    const indices = [];

    for (let i = 0; i <= lengthSegs; i++) {
      const u = i / lengthSegs;
      const z = (u - 0.5) * bodyLen;
      for (let j = 0; j <= radialSegs; j++) {
        const v = j / radialSegs;
        const pt = this.getHSRCrossSectionPoint(v, 0);
        verts.push(pt.x, pt.y, z);
        const uTex = v <= 0.5 ? (1.0 - u) : u;
        uvs.push(uTex, pt.hNorm);
      }
    }

    const ringStride = radialSegs + 1;
    for (let i = 0; i < lengthSegs; i++) {
      for (let j = 0; j < radialSegs; j++) {
        const a = i * ringStride + j;
        const b = (i + 1) * ringStride + j;
        const c = (i + 1) * ringStride + (j + 1);
        const d = i * ringStride + (j + 1);
        indices.push(a, d, b, b, d, c);
      }
    }

    const addBulkheadCap = (zPos, isFrontCap) => {
      const centerIdx = verts.length / 3;
      verts.push(0, 2.36, zPos);
      uvs.push(0.02, 0.5);
      for (let j = 0; j <= radialSegs; j++) {
        const v = j / radialSegs;
        const pt = this.getHSRCrossSectionPoint(v, 0);
        verts.push(pt.x, pt.y, zPos);
        uvs.push(0.02, pt.hNorm);
      }
      for (let j = 0; j < radialSegs; j++) {
        if (isFrontCap) {
          indices.push(centerIdx, centerIdx + 1 + j + 1, centerIdx + 1 + j);
        } else {
          indices.push(centerIdx, centerIdx + 1 + j, centerIdx + 1 + j + 1);
        }
      }
    };
    addBulkheadCap(-bodyLen * 0.5, true);
    addBulkheadCap(bodyLen * 0.5, false);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    geo.userData.sharedGeo = true;
    this[key] = geo;
    return geo;
  }

  createLoftedBulletNoseGeometry(dirZ = -1) {
    const key = dirZ < 0 ? 'loftNoseFwdV4' : 'loftNoseRevV4';
    if (this[key]) return this[key];

    const stations = 32;
    const radialSegs = 36;
    const totalLen = 7.15;
    const verts = [];
    const uvs = [];
    const indices = [];

    for (let i = 0; i <= stations; i++) {
      const u = i / stations;
      const z = u * totalLen * dirZ;
      for (let j = 0; j <= radialSegs; j++) {
        const v = j / radialSegs;
        const pt = this.getHSRCrossSectionPoint(v, u);
        verts.push(pt.x, pt.y, z);
        uvs.push(u, pt.hNorm);
      }
    }

    const ringStride = radialSegs + 1;
    for (let i = 0; i < stations; i++) {
      for (let j = 0; j < radialSegs; j++) {
        const a = i * ringStride + j;
        const b = (i + 1) * ringStride + j;
        const c = (i + 1) * ringStride + (j + 1);
        const d = i * ringStride + (j + 1);
        if (dirZ < 0) {
          indices.push(a, b, d, b, c, d);
        } else {
          indices.push(a, d, b, b, d, c);
        }
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    geo.userData.sharedGeo = true;
    this[key] = geo;
    return geo;
  }

  createSculptedBulletNose(noseSurfaceMat, stripeMat, stripe2Mat, glassMat, lampMat, dirZ = -1) {
    const B = GeometryBatcher;
    const noseGrp = new THREE.Group();
    const drlGlowMat = dirZ < 0 ? this.sharedDrlCyanMat : this.sharedTailRedMat;

    const loftGeo = this.createLoftedBulletNoseGeometry(dirZ);
    const mainNoseMesh = new THREE.Mesh(loftGeo, noseSurfaceMat);
    mainNoseMesh.castShadow = true;
    mainNoseMesh.receiveShadow = true;
    noseGrp.add(mainNoseMesh);

    const cacheKey = `noseTrim_${dirZ}`;
    let cachedTrim = this.trimGeoCache.get(cacheKey);
    if (!cachedTrim) {
      const parts = [];
      const drlParts = [];
      const addBox = (arr, size, pos, col, rot = null) => {
        const g = new THREE.BoxGeometry(size[0], size[1], size[2]);
        arr.push(B.bakePart(g, { position: pos, rotation: rot, color: col }));
        g.dispose();
      };
      const addCyl = (arr, args, pos, col, rot = null) => {
        const g = new THREE.CylinderGeometry(...args);
        arr.push(B.bakePart(g, { position: pos, rotation: rot, color: col }));
        g.dispose();
      };

      addBox(parts, [1.72, 0.30, 2.35], [0, 0.74, dirZ * 1.42], 0x1e293b);
      addBox(parts, [1.52, 0.08, 1.65], [0, 0.58, dirZ * 2.15], 0xe2e8f0);
      addCyl(parts, [0.28, 0.34, 0.42, 16], [0, 1.24, dirZ * 6.82], 0xf8fafc, [Math.PI / 2, 0, 0]);

      [-0.28, 0.28].forEach((wx) => {
        addBox(parts, [0.024, 0.72, 0.024], [wx, 2.86, dirZ * 2.32], 0x0f172a, [dirZ * 0.72, 0, 0.25]);
      });

      const lensGeo = new THREE.SphereGeometry(0.045, 8, 8);
      [-1.34, 1.34].forEach((cx) => {
        addBox(parts, [0.16, 0.10, 0.34], [cx, 2.68, dirZ * 0.65], 0xe2e8f0);
        parts.push(B.bakePart(lensGeo, { position: [cx * 1.04, 2.68, dirZ * 0.52], color: 0x0f172a }));
      });
      lensGeo.dispose();

      [-0.76, 0.76].forEach((lx) => {
        addBox(drlParts, [0.08, 0.06, 0.82], [lx, 1.62, dirZ * 4.85], 0xffffff, [0, lx < 0 ? dirZ * 0.24 : -dirZ * 0.24, 0]);
      });

      cachedTrim = {
        trimGeo: B.mergeParts(parts),
        drlGeo: B.mergeParts(drlParts)
      };
      this.trimGeoCache.set(cacheKey, cachedTrim);
    }

    noseGrp.add(
      new THREE.Mesh(cachedTrim.trimGeo, this.bogieVertexMat),
      new THREE.Mesh(cachedTrim.drlGeo, drlGlowMat)
    );
    return noseGrp;
  }

  createHighDetailCarbodyShell(bodyMat, noseMat, stripe1Mat, stripe2Mat, frameMat, glassMat, bodyLen) {
    const B = GeometryBatcher;
    const shell = new THREE.Group();

    const bodyGeo = this.createLoftedCarbodyGeometry(bodyLen);
    const monocoqueMesh = new THREE.Mesh(bodyGeo, bodyMat);
    monocoqueMesh.castShadow = true;
    monocoqueMesh.receiveShadow = true;
    shell.add(monocoqueMesh);

    const cacheKey = `carbodyTrim_${bodyLen.toFixed(2)}`;
    let trimGeo = this.trimGeoCache.get(cacheKey);
    if (!trimGeo) {
      const parts = [];
      const addBox = (size, pos, col, rot = null) => {
        const g = new THREE.BoxGeometry(size[0], size[1], size[2]);
        parts.push(B.bakePart(g, { position: pos, rotation: rot, color: col }));
        g.dispose();
      };
      const addCyl = (args, pos, col, rot = null) => {
        const g = new THREE.CylinderGeometry(...args);
        parts.push(B.bakePart(g, { position: pos, rotation: rot, color: col }));
        g.dispose();
      };

      for (let ez = -bodyLen * 0.28; ez <= bodyLen * 0.28; ez += bodyLen * 0.28) {
        addBox([1.82, 0.44, 2.6], [0, 0.94, ez], 0x1e293b);
        addBox([1.88, 0.26, 1.4], [0, 0.94, ez], 0x334155);
        addCyl([0.16, 0.16, 1.8, 10], [0.68, 0.86, ez], 0x94a3b8, [Math.PI / 2, 0, 0]);
      }

      [-0.88, 0.88].forEach((rx) => {
        addBox([0.11, 0.15, bodyLen - 1.0], [rx, 3.86, 0], 0xe2e8f0, [0, 0, rx < 0 ? -0.18 : 0.18]);
      });

      addCyl([0.025, 0.025, bodyLen - 0.6, 8], [0.38, 4.02, 0], 0xdc2626, [Math.PI / 2, 0, 0]);

      for (let iz = -bodyLen * 0.36; iz <= bodyLen * 0.36; iz += 3.2) {
        addCyl([0.055, 0.065, 0.16, 8], [0.38, 3.92, iz], 0xf8fafc);
      }
      addCyl([0.11, 0.11, 0.55, 10], [0.12, 3.98, 0], 0xf8fafc, [0, 0, Math.PI / 2]);
      addBox([0.24, 0.14, 0.65], [-0.38, 3.94, -bodyLen * 0.38], 0xe2e8f0);

      [-bodyLen * 0.24, bodyLen * 0.24].forEach((hz) => {
        addBox([1.42, 0.11, 2.55], [0, 3.92, hz], 0xe2e8f0);
        addBox([1.22, 0.13, 1.95], [0, 3.93, hz], 0x1e293b);
        [-0.45, 0.45].forEach((fz) => {
          addCyl([0.36, 0.36, 0.14, 12], [0, 3.93, hz + fz], 0x334155);
        });
      });

      trimGeo = B.mergeParts(parts);
      this.trimGeoCache.set(cacheKey, trimGeo);
    }

    shell.add(new THREE.Mesh(trimGeo, this.bogieVertexMat));
    return shell;
  }

  createInterCarGangway(noseMat) {
    const gw = new THREE.Group();
    const bellowGeo = this.createLoftedCarbodyGeometry(1.18);
    const bellowMesh = new THREE.Mesh(bellowGeo, this.sharedBellowMat);
    bellowMesh.scale.set(0.95, 0.96, 1.0);
    gw.add(bellowMesh);

    const ringMesh = new THREE.Mesh(bellowGeo, noseMat);
    ringMesh.scale.set(0.985, 0.985, 0.28);
    gw.add(ringMesh);
    return gw;
  }

  createFreightCoupler() {
    if (!this._sharedFreightCouplerGeo) {
      const B = GeometryBatcher;
      const parts = [];
      const shank = new THREE.BoxGeometry(0.24, 0.20, 1.45);
      const knuckleA = new THREE.BoxGeometry(0.38, 0.28, 0.34);
      const knuckleB = new THREE.BoxGeometry(0.38, 0.28, 0.34);
      const airHose = new THREE.CylinderGeometry(0.035, 0.035, 1.25, 8);
      parts.push(
        B.bakePart(shank, { position: [0, 0.92, 0], color: 0x1e293b }),
        B.bakePart(knuckleA, { position: [0, 0.92, -0.22], color: 0x334155 }),
        B.bakePart(knuckleB, { position: [0, 0.92, 0.22], color: 0x334155 }),
        B.bakePart(airHose, { position: [0.24, 0.68, 0], rotation: [Math.PI / 2 + 0.16, 0, 0], color: 0x0f172a })
      );
      shank.dispose();
      knuckleA.dispose();
      knuckleB.dispose();
      airHose.dispose();
      this._sharedFreightCouplerGeo = B.mergeParts(parts);
      this._sharedFreightCouplerGeo.userData.sharedGeo = true;
    }
    return new THREE.Mesh(this._sharedFreightCouplerGeo, this.bogieVertexMat);
  }

  rebuildPlayerConsist(liveryKey, numCoaches) {
    this.liveryKey = liveryKey;
    this.numCoaches = numCoaches;
    this.mfdNeedsInitialDraw = true;
    const cfg = GAME_PREMISE_CARTRIDGE.liveries[liveryKey];

    if (this.spotLight && this.spotLight.parent) this.spotLight.parent.remove(this.spotLight);
    if (this.spotLightTarget && this.spotLightTarget.parent) this.spotLightTarget.parent.remove(this.spotLightTarget);

    this._disposeGroupNonShared(this.rootGroup);
    this.vehicles = [];
    this.interactiveCabMeshes = [];

    const mats = this._getLiveryMaterialSet(liveryKey, true);
    const {
      locoBodyMat,
      coachBodyMat,
      noseSurfaceMatFwd,
      noseSurfaceMatTail,
      noseMat,
      stripe1Mat,
      stripe2Mat,
      frameMat,
      glassMat
    } = mats;

    const carLen = 20.5;
    const carGap = 1.1;

    for (let i = 0; i <= numCoaches; i++) {
      const isLoco = (i === 0);
      const isTailBullet = (i === numCoaches && cfg.noseType === 'BULLET');
      const vGroup = new THREE.Group();
      const bodyGroup = new THREE.Group();
      vGroup.add(bodyGroup);

      const bogieFront = this.createBogie();
      const bogieRear = this.createBogie();
      this.rootGroup.add(bogieFront, bogieRear);

      const bodyMat = isLoco ? locoBodyMat : coachBodyMat;

      const chassisLen = (isLoco || isTailBullet) ? 13.5 : (carLen - 1.2);
      const chassisZ = isLoco ? 3.2 : (isTailBullet ? -3.2 : 0);
      const chassis = new THREE.Mesh(new THREE.BoxGeometry(1.88, 0.32, chassisLen), frameMat);
      chassis.position.set(0, 0.96, chassisZ);
      bodyGroup.add(chassis);

      if (isLoco) {
        this.extCabShell = new THREE.Group();
        bodyGroup.add(this.extCabShell);
      }

      const bodyLen = (isLoco || isTailBullet) ? 14.5 : (carLen - 0.55);
      const bodyZOffset = isLoco ? 3.0 : (isTailBullet ? -3.0 : 0);

      if (isLoco && cfg.noseType === 'HOOD') {
        const hoodMaterials = [bodyMat, bodyMat, frameMat, frameMat, noseMat, noseMat];
        const longHood = new THREE.Mesh(new THREE.BoxGeometry(2.54, 2.58, bodyLen), hoodMaterials);
        longHood.position.set(0, 2.55, bodyZOffset);
        longHood.castShadow = true;

        const hoodCrown = new THREE.Mesh(new THREE.BoxGeometry(2.16, 0.32, bodyLen - 0.2), frameMat);
        hoodCrown.position.set(0, 3.96, bodyZOffset);
        for (let fz = 1.0; fz <= 5.8; fz += 2.2) {
          const fanRing = new THREE.Mesh(new THREE.CylinderGeometry(0.68, 0.72, 0.18, 12), frameMat);
          fanRing.position.set(0, 4.14, bodyZOffset + fz);
          const fanCore = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.20, 8), stripe2Mat);
          fanCore.position.set(0, 4.15, bodyZOffset + fz);
          bodyGroup.add(fanRing, fanCore);
        }

        const radiatorWing = new THREE.Mesh(new THREE.BoxGeometry(2.90, 0.58, 5.2), stripe1Mat);
        radiatorWing.position.set(0, 3.48, bodyZOffset + 3.8);
        const radiatorGrille = new THREE.Mesh(new THREE.BoxGeometry(2.94, 0.38, 4.6), frameMat);
        radiatorGrille.position.set(0, 3.48, bodyZOffset + 3.8);

        const fuelTank = new THREE.Mesh(new THREE.BoxGeometry(2.68, 0.72, 6.8), frameMat);
        fuelTank.position.set(0, 0.64, 1.2);
        [-1.36, 1.36].forEach((tx) => {
          const airTank = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 3.2, 10), stripe2Mat);
          airTank.rotation.x = Math.PI / 2;
          airTank.position.set(tx, 0.96, 1.2);
          bodyGroup.add(airTank);
        });

        const stack = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.28, 0.52, 12), frameMat);
        stack.position.set(0, 4.22, 1.8);
        const hornCluster = new THREE.Group();
        hornCluster.position.set(0, 4.18, -1.2);
        const brassMat = new THREE.MeshStandardMaterial({ color: 0xfacc15, metalness: 0.85, roughness: 0.2 });
        [-0.18, 0, 0.18].forEach((hx, hIdx) => {
          const trumpet = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.03, 0.36, 8), brassMat);
          trumpet.rotation.x = Math.PI / 2;
          trumpet.position.set(hx, hIdx === 1 ? 0.08 : 0, 0);
          hornCluster.add(trumpet);
        });
        bodyGroup.add(longHood, hoodCrown, radiatorWing, radiatorGrille, fuelTank, stack, hornCluster);

        [-1.38, 1.38].forEach((hx) => {
          const walkway = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.08, bodyLen), frameMat);
          walkway.position.set(hx, 1.28, bodyZOffset);
          const topRail = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.045, bodyLen - 0.6), noseMat);
          topRail.position.set(hx, 2.12, bodyZOffset);
          bodyGroup.add(walkway, topRail);

          for (let sz = -bodyLen * 0.44; sz <= bodyLen * 0.44; sz += 2.2) {
            const stanchion = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.86, 0.04), noseMat);
            stanchion.position.set(hx, 1.70, bodyZOffset + sz);
            bodyGroup.add(stanchion);
          }
        });

        const cabModule = new THREE.Mesh(new THREE.BoxGeometry(2.94, 2.68, 4.2), noseMat);
        cabModule.position.set(0, 2.58, -6.85);
        cabModule.castShadow = true;
        const cabStripe = new THREE.Mesh(new THREE.BoxGeometry(2.96, 0.56, 4.22), stripe1Mat);
        cabStripe.position.set(0, 1.68, -6.85);
        const cabRoof = new THREE.Mesh(new THREE.BoxGeometry(2.68, 0.28, 4.1), noseMat);
        cabRoof.position.set(0, 4.02, -6.85);
        const browVisor = new THREE.Mesh(new THREE.BoxGeometry(2.86, 0.14, 0.65), stripe1Mat);
        browVisor.position.set(0, 3.62, -9.05);
        browVisor.rotation.x = 0.22;

        const frontNoseHood = new THREE.Mesh(new THREE.BoxGeometry(2.90, 1.36, 1.45), stripe1Mat);
        frontNoseHood.position.set(0, 1.74, -9.55);
        frontNoseHood.rotation.x = 0.16;
        const noseChevron = new THREE.Mesh(new THREE.BoxGeometry(2.92, 0.52, 1.48), stripe2Mat);
        noseChevron.position.set(0, 1.48, -9.58);

        [-0.68, 0.68].forEach((wx) => {
          const winPane = new THREE.Mesh(new THREE.BoxGeometry(1.08, 0.82, 0.16), glassMat);
          winPane.position.set(wx, 3.12, -8.92);
          winPane.rotation.x = -0.18;
          winPane.rotation.y = wx < 0 ? 0.08 : -0.08;
          this.extCabShell.add(winPane);
        });
        this.extCabShell.add(cabModule, cabStripe, cabRoof, browVisor, frontNoseHood, noseChevron);
      } else if (!isLoco && cfg.consistType === 'FREIGHT_CONTAINER') {
        bodyGroup.add(this.createContainerFlatcarWagon(frameMat, i));
      } else if (!isLoco && cfg.consistType === 'FREIGHT_TANKER') {
        bodyGroup.add(this.createTankerWagon(frameMat, i));
      } else if (!isLoco && cfg.consistType === 'FREIGHT_COAL') {
        bodyGroup.add(this.createCoalHopperWagon());
      } else if (!isLoco && cfg.consistType === 'KAI_PASSENGER' && i === 1) {
        bodyGroup.add(this.createPanoramicCoach(stripe1Mat, frameMat, glassMat, carLen));
      } else {
        const shellGrp = this.createHighDetailCarbodyShell(
          bodyMat, noseMat, stripe1Mat, stripe2Mat, frameMat, glassMat, bodyLen
        );
        shellGrp.position.z = bodyZOffset;

        if (isLoco) {
          const bulletNose = this.createSculptedBulletNose(
            noseSurfaceMatFwd, stripe1Mat, stripe2Mat, glassMat, this.sharedLampWhiteMat, -1
          );
          bulletNose.position.set(0, 0, -4.25);
          this.extCabShell.add(shellGrp, bulletNose);
        } else {
          bodyGroup.add(shellGrp);
          if (i === Math.min(2, numCoaches)) {
            this.coachViewShell = shellGrp;
            this.interiorCoachGroup = this.build3DCoachInterior(carLen, cfg);
            this.interiorCoachGroup.visible = false;
            bodyGroup.add(this.interiorCoachGroup);
          }
          if (isTailBullet) {
            const bulletTail = this.createSculptedBulletNose(
              noseSurfaceMatTail, stripe1Mat, stripe2Mat, glassMat, this.sharedTailRedMat, 1
            );
            bulletTail.position.set(0, 0, 4.25);
            bodyGroup.add(bulletTail);
          }
        }
      }

      if (i < numCoaches) {
        if (cfg.consistType?.startsWith('FREIGHT_')) {
          const coupler = this.createFreightCoupler();
          coupler.position.set(0, 0, carLen * 0.5 + 0.55);
          bodyGroup.add(coupler);
        } else {
          const gangway = this.createInterCarGangway(noseMat);
          gangway.position.set(0, 0, carLen * 0.5 + 0.35);
          bodyGroup.add(gangway);
        }
      }

      if (i === numCoaches && cfg.noseType !== 'BULLET') {
        const tlL = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), this.sharedTailRedMat);
        const tlR = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), this.sharedTailRedMat);
        tlL.position.set(-1.05, 1.95, carLen * 0.5 - 0.3);
        tlR.position.set(1.05, 1.95, carLen * 0.5 - 0.3);
        bodyGroup.add(tlL, tlR);
      }

      // Animated Flush-Seated Sliding Plug-Doors (Always visible when closed; pop outward + slide longitudinally when opened!)
      const doors = [];
      if (!isLoco && !cfg.consistType?.startsWith('FREIGHT_')) {
        [-1.445, 1.445].forEach((sx) => {
          const doorZPositions = (i === numCoaches && cfg.noseType === 'BULLET')
            ? [-carLen * 0.38]
            : [-carLen * 0.38, carLen * 0.38];
          doorZPositions.forEach((ez) => {
            const recess = new THREE.Mesh(new THREE.BoxGeometry(0.025, 1.70, 1.20), this.sharedBellowMat);
            recess.position.set(sx * 0.985, 2.22, ez);
            bodyGroup.add(recess);

            const dGrp = new THREE.Group();
            dGrp.position.set(sx, 2.22, ez);

            const dPanel = new THREE.Mesh(new THREE.BoxGeometry(0.04, 1.68, 1.18), noseMat);
            const dStripe = new THREE.Mesh(new THREE.BoxGeometry(0.046, 0.44, 1.18), stripe1Mat);
            dStripe.position.y = -0.26;
            const dWin = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.56, 0.62), glassMat);
            dWin.position.y = 0.32;

            dGrp.add(dPanel, dStripe, dWin);
            dGrp.userData = { baseZ: ez, sideX: sx };
            bodyGroup.add(dGrp);
            doors.push(dGrp);
          });
        });
      }

      if (isLoco) {
        if (cfg.noseType === 'HOOD') {
          const pilotL = new THREE.Mesh(new THREE.BoxGeometry(1.52, 0.54, 0.24), this.sharedCowcatcherRedMat);
          pilotL.position.set(-0.66, 0.76, -carLen * 0.5 + 0.14);
          pilotL.rotation.y = 0.24;
          const pilotR = new THREE.Mesh(new THREE.BoxGeometry(1.52, 0.54, 0.24), this.sharedCowcatcherRedMat);
          pilotR.position.set(0.66, 0.76, -carLen * 0.5 + 0.14);
          pilotR.rotation.y = -0.24;
          const coupler = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.28, 0.72), frameMat);
          coupler.position.set(0, 0.90, -carLen * 0.5 - 0.22);
          bodyGroup.add(pilotL, pilotR, coupler);
        }

        if (cfg.tractionType === 'ELECTRIC') {
          this.pantoGroup = new THREE.Group();
          this.pantoGroup.position.set(0, 4.02, 3.8);

          // Static aerodynamic roof shrouds, porcelain HV insulators & base frame (never squashed!)
          [-1.08, 1.08].forEach((sx) => {
            const pShroud = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.38, 3.4), noseMat);
            pShroud.position.set(sx, 0.08, 0);
            pShroud.rotation.z = sx < 0 ? -0.22 : 0.22;
            this.pantoGroup.add(pShroud);
          });

          [[-0.48, -0.52], [0.48, -0.52], [-0.48, 0.52], [0.48, 0.52]].forEach(([ix, iz]) => {
            const ins = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 0.28, 10), this.sharedPorcelainMat);
            ins.position.set(ix, 0.04, iz);
            this.pantoGroup.add(ins);
          });

          const pBase = new THREE.Mesh(new THREE.BoxGeometry(1.18, 0.10, 1.25), frameMat);
          pBase.position.y = 0.16;
          const pCyl = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.65, 8), stripe1Mat);
          pCyl.rotation.x = Math.PI / 2;
          pCyl.position.set(0, 0.26, 0.1);

          // Articulated single-arm linkage pivoted cleanly at y=0.22 above roof base
          this.pantoArmsGroup = new THREE.Group();
          this.pantoArmsGroup.position.set(0, 0.22, 0);

          const lowerArm = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.055, 0.96, 8), this.sharedChromeMat);
          lowerArm.position.set(0, 0.28, -0.24);
          lowerArm.rotation.x = 0.56;

          const upperArmL = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.04, 0.96, 8), this.sharedChromeMat);
          upperArmL.position.set(-0.22, 0.78, -0.06);
          upperArmL.rotation.x = -0.50;
          const upperArmR = upperArmL.clone();
          upperArmR.position.x = 0.22;
          this.pantoArmsGroup.add(lowerArm, upperArmL, upperArmR);

          // Rigid horizontal collector panhead (4.02 + 1.28 = 5.30m, kissing 5.32m contact wire!)
          this.pantoPanHead = new THREE.Group();
          this.pantoPanHead.position.set(0, 1.28, 0);
          const stripF = new THREE.Mesh(new THREE.BoxGeometry(1.68, 0.045, 0.08), frameMat);
          stripF.position.z = -0.14;
          const stripR = new THREE.Mesh(new THREE.BoxGeometry(1.68, 0.045, 0.08), frameMat);
          stripR.position.z = 0.14;
          [-0.92, 0.92].forEach((hx) => {
            const horn = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.035, 0.36), this.sharedChromeMat);
            horn.position.set(hx, -0.04, 0);
            horn.rotation.z = hx < 0 ? 0.35 : -0.35;
            this.pantoPanHead.add(horn);
          });
          this.pantoPanHead.add(stripF, stripR);

          this.pantoArcSpark = new THREE.Mesh(
            new THREE.SphereGeometry(0.32, 8, 8),
            this.sharedDrlCyanMat
          );
          this.pantoArcSpark.position.set(0, 1.32, 0);
          this.pantoArcSpark.visible = false;

          this.pantoGroup.add(pBase, pCyl, this.pantoArmsGroup, this.pantoPanHead, this.pantoArcSpark);
          bodyGroup.add(this.pantoGroup);
        } else {
          this.pantoGroup = null;
          this.pantoArmsGroup = null;
          this.pantoPanHead = null;
          this.pantoArcSpark = null;
        }

        this.hlL = new THREE.Mesh(new THREE.SphereGeometry(0.11, 10, 10), this.sharedLampWhiteMat);
        this.hlR = new THREE.Mesh(new THREE.SphereGeometry(0.11, 10, 10), this.sharedLampWhiteMat);
        if (cfg.noseType === 'BULLET') {
          this.hlL.scale.set(0.6, 0.5, 1.4);
          this.hlR.scale.set(0.6, 0.5, 1.4);
          this.hlL.position.set(-0.74, 1.56, -9.25);
          this.hlR.position.set(0.74, 1.56, -9.25);
        } else {
          this.hlL.position.set(-0.82, 1.56, -carLen * 0.5 + 0.05);
          this.hlR.position.set(0.82, 1.56, -carLen * 0.5 + 0.05);
        }
        bodyGroup.add(this.hlL, this.hlR);

        this.spotLight.position.set(0, 1.95, -carLen * 0.5 - 1.65);
        this.spotLightTarget.position.set(0, 0.8, -carLen * 0.5 - 95);
        bodyGroup.add(this.spotLightTarget);
        this.spotLight.target = this.spotLightTarget;
        bodyGroup.add(this.spotLight);

        this.build3DDriverCab(bodyGroup, carLen);
      }

      this.rootGroup.add(vGroup);
      this.vehicles.push({
        isLoco,
        vGroup,
        bodyGroup,
        bogieFront,
        bogieRear,
        doors,
        offsetS: i * (carLen + carGap),
        bogieSpan: (isLoco || isTailBullet) ? 11.8 : 13.8
      });
    }
  }

  build3DCoachInterior(carLen, cfg) {
    const B = GeometryBatcher;
    const interior = new THREE.Group();

    const staticParts = [];
    const addBox = (size, pos, col, rot = null) => {
      const g = new THREE.BoxGeometry(size[0], size[1], size[2]);
      staticParts.push(B.bakePart(g, { position: pos, rotation: rot, color: col }));
      g.dispose();
    };

    const seatColor = cfg.noseType === 'BULLET' ? 0x991b1b : 0x1e3a8a;
    addBox([2.74, 0.12, 16.0], [0, 1.32, 0], 0x1e293b);
    addBox([0.58, 0.13, 15.8], [0, 1.325, 0], 0x0f172a);
    addBox([2.62, 0.12, 16.0], [0, 3.68, 0], 0xe2e8f0);
    addBox([0.55, 0.08, 15.6], [-1.12, 3.22, 0], 0x334155, [0, 0, -0.12]);
    addBox([0.55, 0.08, 15.6], [1.12, 3.22, 0], 0x334155, [0, 0, 0.12]);

    const ledStripL = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.04, 15.2), new THREE.MeshBasicMaterial({ color: 0xfef3c7 }));
    ledStripL.position.set(-0.78, 3.61, 0);
    const ledStripR = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.04, 15.2), new THREE.MeshBasicMaterial({ color: 0xfef3c7 }));
    ledStripR.position.set(0.78, 3.61, 0);
    interior.add(ledStripL, ledStripR);

    const worldRef = this.world || (window.apexRailSim && window.apexRailSim.world);
    let seatCounter = 0;

    for (let rz = -5.2; rz <= 5.2; rz += 1.05) {
      [-1.08, -0.56, 0.56, 1.08].forEach((sx) => {
        seatCounter++;
        const by = 1.38;
        addBox([0.18, 0.34, 0.24], [sx, by + 0.17, rz], 0x334155);
        addBox([0.48, 0.12, 0.48], [sx, by + 0.36, rz], seatColor);
        addBox([0.46, 0.72, 0.10], [sx, by + 0.74, rz + 0.19], seatColor, [0.14, 0, 0]);
        addBox([0.36, 0.18, 0.12], [sx, by + 1.04, rz + 0.23], 0xf8fafc, [0.14, 0, 0]);
        addBox([0.06, 0.06, 0.42], [sx - 0.24, by + 0.52, rz + 0.02], 0x334155);
        addBox([0.06, 0.06, 0.42], [sx + 0.24, by + 0.52, rz + 0.02], 0x334155);

        const isPlayerCameraSeat = (Math.abs(rz) < 0.65 && sx < -0.8);
        if (!isPlayerCameraSeat && (seatCounter % 3 !== 0) && worldRef) {
          const paxRig = worldRef.createDetailedHumanNPC(seatCounter + 3, 'SEATED_COACH');
          paxRig.root.position.set(sx, by - 0.10, rz - 0.02);
          paxRig.root.rotation.y = Math.PI;
          interior.add(paxRig.root);
        }
      });
    }

    // Tray table & coffee cup in front of player window seat
    addBox([0.42, 0.025, 0.28], [-1.05, 2.02, -0.42], 0xcbd5e1);
    const cupGeo = new THREE.CylinderGeometry(0.038, 0.030, 0.09, 12);
    const sleeveGeo = new THREE.CylinderGeometry(0.039, 0.033, 0.045, 12);
    staticParts.push(
      B.bakePart(cupGeo, { position: [-1.14, 2.08, -0.42], color: 0xf8fafc }),
      B.bakePart(sleeveGeo, { position: [-1.14, 2.075, -0.42], color: 0xb45309 })
    );
    cupGeo.dispose();
    sleeveGeo.dispose();

    const cabinMerged = B.mergeParts(staticParts);
    cabinMerged.userData.sharedGeo = false;
    interior.add(new THREE.Mesh(cabinMerged, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.58, metalness: 0.15 })));

    return interior;
  }

  build3DDriverCab(locoBody, carLen) {
    const cab = new THREE.Group();
    cab.position.set(0, 2.88, -carLen * 0.5 + 1.05);
    this.interiorCabGroup = cab;

    const darkShellMat = new THREE.MeshStandardMaterial({ color: 0x0b111e, roughness: 0.55, metalness: 0.35 });
    const wallPanelMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.48, metalness: 0.25 });
    const deskLeatherMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.38, metalness: 0.30 });
    const binnacleMat = new THREE.MeshStandardMaterial({ color: 0x060a12, roughness: 0.28, metalness: 0.45 });
    const brushedAlumMat = new THREE.MeshStandardMaterial({ color: 0xcbd5e1, roughness: 0.22, metalness: 0.85, envMapIntensity: 1.35 });

    const cabFloor = new THREE.Mesh(new THREE.BoxGeometry(2.84, 0.12, 2.75), darkShellMat);
    cabFloor.position.set(0, -0.78, -0.35);
    cab.add(cabFloor);

    const ceilingDome = new THREE.Mesh(new THREE.BoxGeometry(2.76, 0.14, 2.45), wallPanelMat);
    ceilingDome.position.set(0, 1.02, -0.25);
    ceilingDome.rotation.x = 0.08;
    const frontVisor = new THREE.Mesh(new THREE.BoxGeometry(2.68, 0.18, 0.68), darkShellMat);
    frontVisor.position.set(0, 0.88, -1.28);
    frontVisor.rotation.x = 0.28;
    const ceilingLight = new THREE.Mesh(new THREE.PlaneGeometry(0.95, 0.35), new THREE.MeshBasicMaterial({ color: 0xe0f2fe }));
    ceilingLight.position.set(0, 0.94, -0.15);
    ceilingLight.rotation.x = Math.PI / 2;
    cab.add(ceilingDome, frontVisor, ceilingLight);

    [-1, 1].forEach((side) => {
      const sx = side * 1.34;
      const sideWallLow = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.86, 2.55), wallPanelMat);
      sideWallLow.position.set(sx, -0.35, -0.32);

      const sideLedge = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.08, 2.35), deskLeatherMat);
      sideLedge.position.set(side * 1.24, 0.06, -0.32);

      const sideWallTop = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.26, 2.45), wallPanelMat);
      sideWallTop.position.set(sx * 0.96, 0.86, -0.32);
      sideWallTop.rotation.z = -side * 0.15;

      const aPillar = new THREE.Mesh(new THREE.BoxGeometry(0.16, 1.18, 0.22), darkShellMat);
      aPillar.position.set(side * 1.18, 0.42, -1.28);
      aPillar.rotation.x = -0.34;
      aPillar.rotation.z = -side * 0.14;

      const bPillar = new THREE.Mesh(new THREE.BoxGeometry(0.16, 1.15, 0.38), wallPanelMat);
      bPillar.position.set(sx, 0.42, 0.58);

      cab.add(sideWallLow, sideLedge, sideWallTop, aPillar, bPillar);
    });

    const cabGlassMat = new THREE.MeshPhysicalMaterial({
      color: 0xbae6fd,
      metalness: 0.2,
      roughness: 0.04,
      clearcoat: 1.0,
      envMapIntensity: 1.45,
      transparent: true,
      opacity: 0.09,
      depthWrite: false
    });
    const windshieldPane = new THREE.Mesh(new THREE.PlaneGeometry(2.54, 1.12), cabGlassMat);
    windshieldPane.position.set(0, 0.42, -1.42);
    windshieldPane.rotation.x = -0.26;
    cab.add(windshieldPane);

    const deskGroup = new THREE.Group();
    deskGroup.position.set(0, -0.26, -1.04);

    const centerDesk = new THREE.Mesh(new THREE.BoxGeometry(1.16, 0.46, 0.68), deskLeatherMat);
    centerDesk.position.set(0, -0.18, -0.06);
    const centerTop = new THREE.Mesh(new THREE.BoxGeometry(1.18, 0.06, 0.64), binnacleMat);
    centerTop.position.set(0, 0.05, -0.05);
    centerTop.rotation.x = 0.14;

    const leftWing = new THREE.Group();
    leftWing.position.set(-0.78, 0, 0.06);
    leftWing.rotation.y = 0.34;
    const lwBase = new THREE.Mesh(new THREE.BoxGeometry(0.68, 0.46, 0.62), deskLeatherMat);
    lwBase.position.y = -0.18;
    const lwTop = new THREE.Mesh(new THREE.BoxGeometry(0.70, 0.06, 0.60), binnacleMat);
    lwTop.position.y = 0.05;
    lwTop.rotation.x = 0.12;
    leftWing.add(lwBase, lwTop);

    const rightWing = new THREE.Group();
    rightWing.position.set(0.78, 0, 0.06);
    rightWing.rotation.y = -0.34;
    const rwBase = new THREE.Mesh(new THREE.BoxGeometry(0.68, 0.46, 0.62), deskLeatherMat);
    rwBase.position.y = -0.18;
    const rwTop = new THREE.Mesh(new THREE.BoxGeometry(0.70, 0.06, 0.60), binnacleMat);
    rwTop.position.y = 0.05;
    rwTop.rotation.x = 0.12;
    rightWing.add(rwBase, rwTop);

    const trimCenter = new THREE.Mesh(new THREE.BoxGeometry(1.18, 0.035, 0.05), brushedAlumMat);
    trimCenter.position.set(0, 0.02, 0.26);
    deskGroup.add(centerDesk, centerTop, leftWing, rightWing, trimCenter);

    const binnacleHousing = new THREE.Mesh(new THREE.BoxGeometry(1.12, 0.29, 0.18), binnacleMat);
    binnacleHousing.position.set(0, 0.18, -0.14);
    binnacleHousing.rotation.x = -0.28;

    const glareHoodTop = new THREE.Mesh(new THREE.BoxGeometry(1.16, 0.025, 0.24), darkShellMat);
    glareHoodTop.position.set(0, 0.32, -0.14);
    glareHoodTop.rotation.x = -0.12;

    const mfdMat = new THREE.MeshBasicMaterial({ map: this.mfdTexture });
    const mfdScreen = new THREE.Mesh(new THREE.PlaneGeometry(1.06, 0.25), mfdMat);
    mfdScreen.position.set(0, 0.18, -0.046);
    mfdScreen.rotation.x = -0.28;
    deskGroup.add(binnacleHousing, glareHoodTop, mfdScreen);

    // Master Traction T-Handle Controller
    const thrBase = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.045, 0.28), brushedAlumMat);
    thrBase.position.set(-0.56, 0.08, 0.04);
    thrBase.rotation.y = 0.22;
    const thrSlot = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.05, 0.22), binnacleMat);
    thrSlot.position.set(-0.56, 0.085, 0.04);
    thrSlot.rotation.y = 0.22;
    deskGroup.add(thrBase, thrSlot);

    this.leverThrottle = new THREE.Group();
    this.leverThrottle.position.set(-0.56, 0.08, 0.04);
    this.leverThrottle.rotation.y = 0.22;
    const tStem = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.014, 0.13, 10), this.sharedChromeMat);
    tStem.position.y = 0.065;
    const tCrossbar = new THREE.Mesh(new THREE.CylinderGeometry(0.020, 0.020, 0.13, 14), binnacleMat);
    tCrossbar.rotation.z = Math.PI / 2;
    tCrossbar.position.y = 0.13;
    const tAccent = new THREE.Mesh(new THREE.CylinderGeometry(0.021, 0.021, 0.045, 14), new THREE.MeshStandardMaterial({ color: 0x06b6d4, roughness: 0.25 }));
    tAccent.rotation.z = Math.PI / 2;
    tAccent.position.y = 0.13;
    tCrossbar.userData = { cabAction: 'THROTTLE' };
    tAccent.userData = { cabAction: 'THROTTLE' };
    this.interactiveCabMeshes.push(tCrossbar, tAccent);
    this.leverThrottle.add(tStem, tCrossbar, tAccent);
    deskGroup.add(this.leverThrottle);

    // Pneumatic Brake Controller
    const brkBase = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.045, 0.28), brushedAlumMat);
    brkBase.position.set(0.56, 0.08, 0.04);
    brkBase.rotation.y = -0.22;
    const brkSlot = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.05, 0.22), binnacleMat);
    brkSlot.position.set(0.56, 0.085, 0.04);
    brkSlot.rotation.y = -0.22;
    deskGroup.add(brkBase, brkSlot);

    this.leverBrake = new THREE.Group();
    this.leverBrake.position.set(0.56, 0.08, 0.04);
    this.leverBrake.rotation.y = -0.22;
    const bStem = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.014, 0.13, 10), this.sharedChromeMat);
    bStem.position.y = 0.065;
    const bGrip = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.022, 0.12, 14), binnacleMat);
    bGrip.rotation.z = Math.PI / 2;
    bGrip.position.y = 0.13;
    const bAccent = new THREE.Mesh(new THREE.CylinderGeometry(0.023, 0.023, 0.04, 14), new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.25 }));
    bAccent.rotation.z = Math.PI / 2;
    bAccent.position.y = 0.13;
    bGrip.userData = { cabAction: 'BRAKE' };
    bAccent.userData = { cabAction: 'BRAKE' };
    this.interactiveCabMeshes.push(bGrip, bAccent);
    this.leverBrake.add(bStem, bGrip, bAccent);
    deskGroup.add(this.leverBrake);

    // Console Pushbuttons & Emergency Stop Mushroom Plunger
    const hornBezel = new THREE.Mesh(new THREE.CylinderGeometry(0.034, 0.038, 0.02, 14), brushedAlumMat);
    hornBezel.position.set(-0.78, 0.07, 0.12);
    const hornBtn3D = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.026, 0.028, 14), new THREE.MeshStandardMaterial({ color: 0xfacc15, emissive: 0xca8a04, emissiveIntensity: 0.4 }));
    hornBtn3D.position.set(-0.78, 0.075, 0.12);
    hornBtn3D.userData = { cabAction: 'HORN' };
    this.interactiveCabMeshes.push(hornBtn3D);

    const sifaBezel = new THREE.Mesh(new THREE.CylinderGeometry(0.034, 0.038, 0.02, 14), brushedAlumMat);
    sifaBezel.position.set(0.78, 0.07, 0.12);
    this.sifaBtn3D = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.026, 0.028, 14), new THREE.MeshBasicMaterial({ color: 0x10b981 }));
    this.sifaBtn3D.position.set(0.78, 0.075, 0.12);
    this.sifaBtn3D.userData = { cabAction: 'SIFA' };
    this.interactiveCabMeshes.push(this.sifaBtn3D);

    const eStopCollar = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.048, 0.022, 16), new THREE.MeshStandardMaterial({ color: 0xfacc15 }));
    eStopCollar.position.set(0.82, 0.07, -0.06);
    const eStopHead = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.028, 0.038, 16), new THREE.MeshStandardMaterial({ color: 0xef4444, roughness: 0.3 }));
    eStopHead.position.set(0.82, 0.095, -0.06);
    eStopHead.userData = { cabAction: 'BRAKE' };
    this.interactiveCabMeshes.push(eStopHead);

    deskGroup.add(hornBezel, hornBtn3D, sifaBezel, this.sifaBtn3D, eStopCollar, eStopHead);
    cab.add(deskGroup);
    locoBody.add(cab);
  }

  update3DMFDTexture(speedKmh, limitKmh, throttlePct, brakePipeBar, distNextSt, sifaWarn) {
    // Skip expensive 1024x256 GPU texture upload when outside Cab 3D camera
    if (!this.mfdNeedsInitialDraw && this.interiorCabGroup && !this.interiorCabGroup.visible) {
      return;
    }
    this.mfdNeedsInitialDraw = false;

    const ctx = this.mfdCtx;
    ctx.fillStyle = '#030712';
    ctx.fillRect(0, 0, 1024, 256);

    const panels = [
      { x: 10, w: 300, title: 'CTCS-3 / ETCS TRACK PROFILE' },
      { x: 322, w: 380, title: sifaWarn ? '⚠ SIFA ALERTER - PRESS [Q]!' : 'WHOOSH CR400AF PRIMARY DISPLAY' },
      { x: 714, w: 300, title: 'TRACTION & PNEUMATIC TELEMETRY' }
    ];

    panels.forEach((p, idx) => {
      ctx.fillStyle = (idx === 1 && sifaWarn) ? '#3b0712' : '#081121';
      ctx.fillRect(p.x, 10, p.w, 236);
      ctx.strokeStyle = (idx === 1 && sifaWarn) ? '#f43f5e' : '#0284c7';
      ctx.lineWidth = 3;
      ctx.strokeRect(p.x, 10, p.w, 236);

      ctx.fillStyle = '#0c1e38';
      ctx.fillRect(p.x + 2, 12, p.w - 4, 26);
      ctx.fillStyle = '#38bdf8';
      ctx.font = 'bold 13px JetBrains Mono';
      ctx.textAlign = 'left';
      ctx.fillText(p.title, p.x + 12, 30);
    });

    const cfg = GAME_PREMISE_CARTRIDGE.liveries[this.liveryKey];
    const isElectric = cfg && cfg.tractionType === 'ELECTRIC';
    const voltStr = isElectric
      ? (this.pantoRaised ? '25.0 kV AC 50Hz' : '0.0 kV (PANTO DOWN)')
      : 'GE 7FDL DIESEL-ELEC';
    const sigStr = this.doorsOpen
      ? 'RESTRICTED / STOP'
      : (distNextSt < 550 ? 'APPROACH (YELLOW)' : 'CLEAR ASPECT (GREEN)');

    ctx.fillStyle = this.doorsOpen ? '#f43f5e' : (distNextSt < 550 ? '#f59e0b' : '#10b981');
    ctx.beginPath();
    ctx.arc(46, 76, 14, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 18px JetBrains Mono';
    ctx.fillText(`NEXT STN: ${distNextSt} m`, 74, 82);

    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 13px JetBrains Mono';
    ctx.fillText(`ATP TARGET SPEED: ${limitKmh} KM/H`, 30, 122);
    ctx.fillText(`POWER BUS: ${voltStr}`, 30, 148);
    ctx.fillText(`CAB SIGNAL: ${sigStr}`, 30, 174);

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(30, 194, 260, 22);
    ctx.fillStyle = speedKmh > limitKmh ? '#f43f5e' : '#10b981';
    ctx.fillRect(30, 194, Math.min(1, speedKmh / Math.max(1, limitKmh)) * 260, 22);

    const cx = 512, cy = 152, r = 82;
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI * 0.80, Math.PI * 2.20);
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 14;
    ctx.stroke();

    const gaugeMax = Math.max(160, (cfg ? cfg.maxSpeedKmh : 350) + 10);
    const spdRatio = Math.min(1.0, speedKmh / gaugeMax);
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI * 0.80, Math.PI * (0.80 + spdRatio * 1.40));
    ctx.strokeStyle = speedKmh > limitKmh ? '#f43f5e' : '#06b6d4';
    ctx.lineWidth = 14;
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 56px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(`${speedKmh}`, cx, cy + 12);
    ctx.font = 'bold 14px JetBrains Mono';
    ctx.fillStyle = '#38bdf8';
    ctx.fillText(`KM/H • LIMIT ${limitKmh}`, cx, cy + 40);

    ctx.textAlign = 'left';
    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 13px JetBrains Mono';
    ctx.fillText(`TRACTION EFFORT (${Math.round(throttlePct)}%)`, 732, 68);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(732, 76, 262, 20);
    ctx.fillStyle = '#06b6d4';
    ctx.fillRect(732, 76, (throttlePct / 100) * 262, 20);

    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`BRAKE PIPE (${brakePipeBar.toFixed(1)} BAR)`, 732, 126);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(732, 134, 262, 20);
    ctx.fillStyle = '#f59e0b';
    ctx.fillRect(732, 134, (brakePipeBar / 5.0) * 262, 20);

    ctx.fillStyle = this.lastWheelSlip ? '#f43f5e' : '#34d399';
    ctx.font = 'bold 14px JetBrains Mono';
    ctx.fillText(this.lastWheelSlip ? '⚠ WSP: WHEEL SLIP ACTIVE!' : 'BOGIE WSP & TRACTION: NORMAL', 732, 192);
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(`CONSIST: ${this.numCoaches}-CAR READY`, 732, 218);

    this.mfdTexture.needsUpdate = true;
  }

  updateKinematics(leadS, speedMps, throttlePct, brakePct, wheelSlipping, dt, audioEngine) {
    const cfg = GAME_PREMISE_CARTRIDGE.liveries[this.liveryKey];
    const originS = EndlessWorldMathematics.originS;
    this.lastWheelSlip = Boolean(wheelSlipping);

    const doorTarget = this.doorsOpen ? 1.0 : 0.0;
    this.doorAnim += (doorTarget - this.doorAnim) * Math.min(1, dt * 4.5);

    // Articulated Pantograph Linkage: roof shrouds & porcelain insulators remain rigid!
    const pantoTarget = this.pantoRaised ? 1.0 : 0.14;
    this.pantoAnim += (pantoTarget - this.pantoAnim) * Math.min(1, dt * 4.0);
    if (this.pantoGroup && this.pantoArmsGroup && this.pantoPanHead) {
      const fold = THREE.MathUtils.clamp((this.pantoAnim - 0.14) / 0.86, 0.0, 1.0);
      this.pantoArmsGroup.scale.y = THREE.MathUtils.lerp(0.14, 1.0, fold);
      this.pantoPanHead.position.y = 0.28 + 1.00 * fold;
      const isElectricFast = cfg.tractionType === 'ELECTRIC' && this.pantoRaised && Math.abs(speedMps) > 8 && throttlePct > 15;
      if (this.pantoArcSpark) {
        this.pantoArcSpark.position.y = this.pantoPanHead.position.y + 0.04;
        this.pantoArcSpark.visible = isElectricFast && Math.random() > 0.72;
      }
    }

    if (this.leverThrottle) this.leverThrottle.rotation.x = THREE.MathUtils.lerp(0.35, -0.55, throttlePct / 100);
    if (this.leverBrake) this.leverBrake.rotation.x = THREE.MathUtils.lerp(-0.45, 0.45, brakePct / 100);

    if (this.wipersActive) {
      this.wiperPhase += dt * 5.2;
    }

    if (this.spotLight) {
      const frameNow = EndlessWorldMathematics.getTrackFrame(leadS, 0, originS, _fCenter);
      const beamPower = frameNow.inTunnel ? 58.0 : 38.0;
      this.spotLight.intensity = this.lightsActive ? beamPower : 0.0;
      this.hlL.visible = this.lightsActive;
      this.hlR.visible = this.lightsActive;
    }

    const tNow = performance.now() * 0.001;
    const speedNorm = Math.min(1.0, Math.abs(speedMps) / 42.0);
    const wheelR = 0.46; // True UIC 920mm wheelset radius
    const dTheta = (speedMps / wheelR) * dt;
    const slipOverspin = wheelSlipping ? (18.5 * dt * Math.sign(speedMps || 1)) : 0.0;
    this.wheelSpinPhase = (this.wheelSpinPhase + dTheta) % (Math.PI * 2);
    this.poweredSpinPhase = ((this.poweredSpinPhase || 0) + dTheta + slipOverspin) % (Math.PI * 2);

    const isEmu = cfg.consistType === 'EMU';
    const plugOut = Math.min(1.0, this.doorAnim * 3.2) * 0.068;
    const slideAmt = Math.max(0.0, (this.doorAnim - 0.18) / 0.82) * 1.14;

    // 1. Player Consist Multi-Bogie Chord Kinematics (Zero GC!)
    for (let i = 0; i < this.vehicles.length; i++) {
      const v = this.vehicles[i];
      const centerS = leadS - v.offsetS;
      const sF = centerS + v.bogieSpan * 0.5;
      const sR = centerS - v.bogieSpan * 0.5;

      EndlessWorldMathematics.getTrackFrame(sF, 0, originS, _fFront);
      EndlessWorldMathematics.getTrackFrame(sR, 0, originS, _fRear);
      EndlessWorldMathematics.getTrackFrame(centerS, 0, originS, _fCenter);

      this.placeOnFrame(v.bogieFront, _fFront);
      this.placeOnFrame(v.bogieRear, _fRear);

      const isMotoredCar = (i === 0) || (isEmu && (i % 2 === 1));
      const spinAngle = isMotoredCar ? this.poweredSpinPhase : this.wheelSpinPhase;
      v.bogieFront.userData.axleF.rotation.x = -spinAngle;
      v.bogieFront.userData.axleR.rotation.x = -spinAngle;
      v.bogieRear.userData.axleF.rotation.x = -spinAngle;
      v.bogieRear.userData.axleR.rotation.x = -spinAngle;

      _midPos.addVectors(_fFront.position, _fRear.position).multiplyScalar(0.5);
      v.vGroup.position.copy(_midPos);

      _fwd.subVectors(_fFront.position, _fRear.position).normalize();
      _right.crossVectors(_fwd, _upWorld).normalize();
      _up.crossVectors(_right, _fwd).normalize();
      _negFwd.copy(_fwd).negate();

      _basisMat.makeBasis(_right, _up, _negFwd);
      v.vGroup.quaternion.setFromRotationMatrix(_basisMat);

      const roll = Math.sin(tNow * 6.8 + i * 1.3) * 0.007 * speedNorm;
      const pitch = Math.cos(tNow * 8.4 + i * 0.9) * 0.0035 * speedNorm;
      v.bodyGroup.rotation.z = _fCenter.cantAngle + roll;
      v.bodyGroup.rotation.x = pitch;

      // Flush-seated plug doors: always visible; pop laterally outward + slide longitudinally when open
      for (let dIdx = 0; dIdx < v.doors.length; dIdx++) {
        const d = v.doors[dIdx];
        const sideSign = d.userData.sideX >= 0 ? 1 : -1;
        const dir = d.userData.baseZ > 0 ? -1 : 1;
        d.position.x = d.userData.sideX + sideSign * plugOut;
        d.position.z = d.userData.baseZ + dir * slideAmt;
        d.visible = true;
      }
    }

    // 2. Articulated Oncoming AI Train on Parallel Track 2
    this.aiTrainS -= this.aiSpeedMps * dt;
    if (this.aiTrainS < leadS - 320) {
      this.spawnNextOncomingTrain(leadS);
    }
    if (!this.passedPlayer && Math.abs(this.aiTrainS - leadS) < 28) {
      this.passedPlayer = true;
      if (audioEngine) audioEngine.playPassingTrainWhoosh();
    }

    // 3. Highway Traffic Vehicles along Parallel Toll Road, River Bridges & Mountain Road Tunnels!
    for (let i = 0; i < this.highwayVehicles.length; i++) {
      const hv = this.highwayVehicles[i];
      let targetSpd = hv.baseSpeedMps !== undefined ? hv.baseSpeedMps * hv.dir : hv.speedMps;

      // Same-lane car-following headway (prevent fast cars from ghosting through trucks ahead)
      for (let j = 0; j < this.highwayVehicles.length; j++) {
        if (i === j) continue;
        const other = this.highwayVehicles[j];
        if (other.dir !== hv.dir || Math.abs(other.laneFromMid - hv.laneFromMid) > 0.5) continue;
        const gapAhead = (other.sCoord - hv.sCoord) * hv.dir;
        if (gapAhead > 0 && gapAhead < 22.0) {
          if (gapAhead < 14.0) {
            hv.sCoord = other.sCoord - hv.dir * 14.0;
          }
          if (hv.dir > 0) targetSpd = Math.min(targetSpd, other.speedMps);
          else targetSpd = Math.max(targetSpd, other.speedMps);
        }
      }
      hv.speedMps += (targetSpd - hv.speedMps) * Math.min(1, dt * 4.0);
      hv.sCoord += hv.speedMps * dt;

      if (hv.sCoord < leadS - 220) {
        hv.sCoord = leadS + 720 + Math.random() * 250;
        if (hv.baseSpeedMps !== undefined) hv.speedMps = hv.baseSpeedMps * hv.dir;
      } else if (hv.sCoord > leadS + 980) {
        hv.sCoord = leadS - 180 - Math.random() * 60;
        if (hv.baseSpeedMps !== undefined) hv.speedMps = hv.baseSpeedMps * hv.dir;
      }

      EndlessWorldMathematics.getTrackFrame(hv.sCoord, 2.2, originS, _hFrame);
      const hx = _hFrame.position.x + _hFrame.right.x * hv.laneFromMid;
      const hz = _hFrame.position.z + _hFrame.right.z * hv.laneFromMid;
      const hy = EndlessWorldMathematics.getHighwayElevation(hv.sCoord, _hFrame.position.y);
      hv.mesh.position.set(hx, hy + 0.08, hz);

      // Right-handed basis (det = +1) for both forward and oncoming Indonesian left-hand highway lanes
      if (hv.dir > 0) {
        _right.copy(_hFrame.right);
        _fwd.copy(_hFrame.tangent).negate();
      } else {
        _right.copy(_hFrame.right).negate();
        _fwd.copy(_hFrame.tangent);
      }
      _basisMat.makeBasis(_right, _hFrame.up, _fwd);
      hv.mesh.quaternion.setFromRotationMatrix(_basisMat);
      hv.mesh.visible = true;
    }

    const track2Off = GAME_PREMISE_CARTRIDGE.railway.doubleTrackOffset;
    this.aiWheelSpinPhase = (this.aiWheelSpinPhase + (this.aiSpeedMps / wheelR) * dt) % (Math.PI * 2);

    for (let i = 0; i < this.aiVehicles.length; i++) {
      const av = this.aiVehicles[i];
      const cS = this.aiTrainS + av.offsetS;
      const sF = cS + av.bogieSpan * 0.5;
      const sR = cS - av.bogieSpan * 0.5;

      EndlessWorldMathematics.getTrackFrame(sF, track2Off, originS, _fFront);
      EndlessWorldMathematics.getTrackFrame(sR, track2Off, originS, _fRear);
      EndlessWorldMathematics.getTrackFrame(cS, track2Off, originS, _fCenter);

      this.placeOnFrame(av.bogieFront, _fFront);
      this.placeOnFrame(av.bogieRear, _fRear);

      av.bogieFront.userData.axleF.rotation.x = this.aiWheelSpinPhase;
      av.bogieFront.userData.axleR.rotation.x = this.aiWheelSpinPhase;
      av.bogieRear.userData.axleF.rotation.x = this.aiWheelSpinPhase;
      av.bogieRear.userData.axleR.rotation.x = this.aiWheelSpinPhase;

      _midPos.addVectors(_fFront.position, _fRear.position).multiplyScalar(0.5);
      av.vGroup.position.copy(_midPos);

      _fwd.subVectors(_fFront.position, _fRear.position).normalize();
      _right.crossVectors(_fwd, _upWorld).normalize();
      _up.crossVectors(_right, _fwd).normalize();
      _negFwd.copy(_fwd).negate();

      _basisMat.makeBasis(_right, _up, _negFwd);
      av.vGroup.quaternion.setFromRotationMatrix(_basisMat);
      av.bodyGroup.rotation.z = _fCenter.cantAngle;
    }

    this.updateParticles(cfg, throttlePct, brakePct, speedMps, wheelSlipping, dt);
  }

  spawnNextOncomingTrain(leadS, distanceAhead = 880) {
    this.aiPoolIdx = (this.aiPoolIdx + 1) % this.aiPoolKeys.length;
    const nextLivery = this.aiPoolKeys[this.aiPoolIdx];
    this.buildOncomingAITrain(nextLivery);
    this.aiTrainS = leadS + distanceAhead + Math.random() * 240;
    this.passedPlayer = false;
  }

  updateParticles(cfg, throttlePct, brakePct, speedMps, wheelSlipping, dt) {
    const loco = this.vehicles[0];
    if (!loco) return;
    loco.bodyGroup.updateMatrixWorld();

    if (cfg.tractionType === 'DIESEL') {
      for (let i = 0; i < this.smokePool.length; i++) {
        const p = this.smokePool[i];
        if (p.life <= 0 && Math.random() < (0.18 + (throttlePct / 100) * 0.65)) {
          p.life = p.maxLife;
          p.mesh.visible = true;
          // Emit exhaust directly from the CC206 roof stack (0, 4.46, 1.8) in world space!
          _locoPos.set((Math.random() - 0.5) * 0.28, 4.46, 1.8 + (Math.random() - 0.5) * 0.25);
          _locoPos.applyMatrix4(loco.bodyGroup.matrixWorld);
          p.mesh.position.copy(_locoPos);
          p.mesh.scale.setScalar(0.45);
          p.vel.set((Math.random() - 0.5) * 1.4, 2.2 + (throttlePct / 100) * 3.6, 1.8 + Math.abs(speedMps) * 0.35);
        } else if (p.life > 0) {
          p.life -= dt;
          p.mesh.position.addScaledVector(p.vel, dt);
          const prog = 1 - (p.life / p.maxLife);
          p.mesh.scale.setScalar(0.55 + prog * 2.4);
          p.mesh.material.opacity = (1 - prog) * (0.16 + (throttlePct / 100) * 0.26);
          if (p.life <= 0) p.mesh.visible = false;
        }
      }
    } else {
      for (let i = 0; i < this.smokePool.length; i++) this.smokePool[i].mesh.visible = false;
    }

    const emitSparks = (wheelSlipping || (brakePct > 70 && Math.abs(speedMps) > 8));
    for (let i = 0; i < this.sparkPool.length; i++) {
      const s = this.sparkPool[i];
      if (s.life <= 0 && emitSparks && Math.random() < 0.48) {
        s.life = 0.22 + Math.random() * 0.20;
        s.mesh.visible = true;
        const side = Math.random() > 0.5 ? 0.7175 : -0.7175;
        const bogieZ = (Math.random() > 0.5 ? -6.3 : 6.3) + (Math.random() - 0.5) * 2.4;
        _locoPos.set(side, 0.14, bogieZ);
        _locoPos.applyMatrix4(loco.bodyGroup.matrixWorld);
        s.mesh.position.copy(_locoPos);
        s.vel.set((Math.random() - 0.5) * 3.2, 1.2 + Math.random() * 2.4, 4.5 + Math.abs(speedMps) * 0.35);
      } else if (s.life > 0) {
        s.life -= dt;
        s.vel.y -= 9.8 * dt;
        s.mesh.position.addScaledVector(s.vel, dt);
        if (s.life <= 0) s.mesh.visible = false;
      }
    }
  }

  placeOnFrame(obj, frame) {
    obj.position.copy(frame.position);
    _negFwd.copy(frame.tangent).negate();
    _basisMat.makeBasis(frame.right, frame.up, _negFwd);
    obj.quaternion.setFromRotationMatrix(_basisMat);
    obj.rotateZ(frame.cantAngle);
  }
}
