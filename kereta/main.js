import * as THREE from 'three';
import { GAME_PREMISE_CARTRIDGE } from './data/PremiseCartridge.js';
import { FloatingOriginManager } from './engine/FloatingOrigin.js';
import { HardwareTelemetryGovernor } from './engine/HardwareGovernor.js';
import { HybridRenderer } from './engine/HybridRenderer.js';
import { EndlessWorldMathematics, createTrackFrameStruct } from './world/WorldMath.js';
import { ProceduralAssetFactory } from './assets/ProceduralTextures.js';
import { MultiBusRailAudioEngine } from './audio/RailAudioEngine.js';
import { StreamingRailwayWorld } from './world/StreamingWorld.js';
import { ArticulatedTrainSystem } from './train/ArticulatedTrain.js';
import { WindshieldFX } from './ui/WindshieldFX.js';

/**
 * ============================================================================
 * NEXUS-3D v6.0 "UNREAL WEB" MASTER SIMULATION KERNEL
 * - Unified WebGPU / WebGL2 Hybrid Renderer + TSL Sky & HDR Bloom
 * - IEEE-754 64-bit Floating-Origin Rebasing (Zero Float32 jitter forever!)
 * - Web Worker Background Chunk Streaming + GPU Merged Static Infrastructure
 * - Hardware Telemetry Governor (AUTO / ULTRA / HIGH / BALANCED / ECO)
 * - Sub-Stepped Velocity Verlet Longitudinal Rail Physics + Multi-Bus Doppler
 * ============================================================================
 */

// Pre-allocated Zero-GC math scratchpad for per-frame physics, camera & Doppler
const _physFrame = createTrackFrameStruct();
const _camFrame = createTrackFrameStruct();
const _aiFrame = createTrackFrameStruct();
const _cxFrame = createTrackFrameStruct();

const _locoPos = new THREE.Vector3();
const _locoQuat = new THREE.Quaternion();
const _coachPos = new THREE.Vector3();
const _coachQuat = new THREE.Quaternion();
const _tmpVecA = new THREE.Vector3();
const _tmpVecB = new THREE.Vector3();
const _tmpVecC = new THREE.Vector3();
const _playerVel = new THREE.Vector3();
const _aiVel = new THREE.Vector3();
const _lookEuler = new THREE.Euler(0, 0, 0, 'YXZ');
const _lookQuat = new THREE.Quaternion();
const _ribbonPt = new THREE.Vector3();
const _ndcMouse = new THREE.Vector2();

export class ApexRailSimulatorKernel {
  constructor() {
    this.canvas = document.getElementById('webgl-viewport');
    this.rainCanvas = document.getElementById('windshield-fx-canvas');
    this.ribbonCanvas = document.getElementById('track-ribbon-canvas');
    this.ribbonCtx = this.ribbonCanvas ? this.ribbonCanvas.getContext('2d') : null;

    // 1. Floating-Origin Manager (64-bit longitudinal `distanceS` -> local Float32 Z)
    this.floatingOrigin = new FloatingOriginManager(2000.0, 100.0);

    // 2. Sub-Stepped Longitudinal Physics State (Starts at Stasiun Halim KCIC Platform)
    this.distanceS = 48.0;
    this.floatingOrigin.forceRebase(this.distanceS);
    EndlessWorldMathematics.setOriginS(this.floatingOrigin.originS);

    this.lastJointIdx = Math.floor(this.distanceS / GAME_PREMISE_CARTRIDGE.railway.jointInterval);
    this.speedMps = 0.0;
    this.accel = 0.0;
    this.prevAccel = 0.0;
    this.reverser = 1;
    this.throttlePct = 0;
    this.actualThrottlePct = 0;
    this.brakePct = 0;
    this.pipeBar = 5.0;
    this.cylBar = 0.0;
    this.eqBar = 5.0;
    this.currentTractionKn = 0;
    this.sanderActive = false;
    this.wheelSlipping = false;
    this.autopilot = false;
    this.hudCollapsed = false;

    // 3. SIFA Vigilance Alerter & Career Timetable State
    this.sifaTimer = 0.0;
    this.sifaWarning = false;
    this.boardingProgress = 0.0;
    this.score = 5000;
    this.comfort = 100.0;
    this.lastEvalStIdx = 0;
    this.aiStationStage = 'CRUISE';
    this.aiStageTimer = 0;

    // 4. 7-Camera Suite + Walk/Drone Mode & 3D Velocity Doppler State
    this.cameraMode = 0; // 0:Cab, 1:Orbit, 2:FlyBy, 3:Bogie, 4:Coach, 5:Walk, 6:Drone
    this.cabYaw = 0.0;
    this.cabPitch = -0.04;
    this.orbitYaw = 0.58;
    this.orbitPitch = 0.24;
    this.orbitDist = 28.0;
    this.flyByAnchorS = -999999;
    this.flyByPos = new THREE.Vector3();
    this.walkOffset = new THREE.Vector3(-4.2, 2.2, 0);
    this.prevCamPos = new THREE.Vector3();
    this.camVel = new THREE.Vector3();
    this.lastPlayerDoppler = 1.0;
    this.dragging = false;
    this.lastPtrX = 0;
    this.lastPtrY = 0;
    this.raycaster = new THREE.Raycaster();

    // 5. Weather, Clock & Windshield FX
    this.weather = 'MORNING';
    this.sunOffset = new THREE.Vector3(70, 55, 40);
    this.clockMins = 7 * 60 + 30;
    this.windshieldFX = new WindshieldFX(this.rainCanvas);

    this.keys = {};
    this.booted = false;

    // 6. Hardware Telemetry Governor (AUTO / ULTRA / HIGH / BALANCED / ECO)
    this.governor = new HardwareTelemetryGovernor((newTier) => {
      if (this.hybridRenderer) {
        this.hybridRenderer.applyQualityTier(newTier);
      }
      this.updateEngineTelemetryUI(this.lastMeasuredFps || 60);
    });
  }

  _setBootStatus(msg) {
    const el = document.getElementById('boot-status-text');
    if (el) el.textContent = msg;
  }

  async boot() {
    try {
      this._setBootStatus('Menginisialisasi Hybrid WebGPU / WebGL2 Renderer & TSL Shaders...');
      await this.initSceneAndRenderer();

      this._setBootStatus('Menyiapkan Multi-Bus WebAudio Doppler Synthesizer...');
      this.audio = new MultiBusRailAudioEngine();

      this._setBootStatus('Memuat Streaming World Worker & Infrastruktur Stasiun Halim...');
      this.world = new StreamingRailwayWorld(this.scene, this.textures, this.envMap);
      this.world.stationNpcCount = this.governor.activeTier.stationNpcCount;
      this.world.warmupInitialChunks(this.distanceS, this.floatingOrigin.originS);

      this._setBootStatus('Merakit Rangkaian Kereta Cepat KCIC Whoosh CR400AF PBR...');
      this.train = new ArticulatedTrainSystem(this.scene, this.envMap, this.textures.microBump, this.world);

      this.bindControls();
      this.setWeatherPreset('MORNING');

      // Position train, shadows & camera before GPU shader precompilation
      this.train.updateKinematics(this.distanceS, 0, 0, 0, false, 0.016, this.audio);
      this.updateCameraSystem(0.016);
      this.updateHUD();
      this.updateEngineTelemetryUI(60);

      // Pre-warm both interior cab & exterior Clearcoat carbody shaders so camera switching has 0ms hitch
      this._setBootStatus('Mengompilasi Pipeline Shader GPU...');
      if (this.train.extCabShell) this.train.extCabShell.visible = true;
      await this.hybridRenderer.precompileScene(this.scene, this.camera);
      if (this.train.extCabShell) this.train.extCabShell.visible = (this.cameraMode !== 0);

      this.lastTime = performance.now();
      this.fpsFrames = 0;
      this.fpsTimer = this.lastTime;
      this.lastMeasuredFps = 60;
      this.hudTick = 0;
      this.booted = true;

      // Dismiss boot splash smoothly
      const loader = document.getElementById('boot-loader');
      if (loader) {
        loader.style.opacity = '0';
        loader.style.pointerEvents = 'none';
        setTimeout(() => loader.remove(), 420);
      }

      this.animate = this.animate.bind(this);
      this.hybridRenderer.setAnimationLoop(this.animate);
    } catch (err) {
      console.error('[ApexRailSimulatorKernel] Boot error:', err);
      this._setBootStatus(`Gagal memuat engine: ${err && err.message ? err.message : err}`);
    }
  }

  async initSceneAndRenderer() {
    const initialTier = this.governor.activeTier;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x7dd3fc);
    this.scene.fog = new THREE.FogExp2(0x93c5fd, 0.00062);

    this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.18, 2600);

    // Hemisphere + Texel-Snapped Real-Time Directional Sun Light
    this.hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0x334155, 0.88);
    this.scene.add(this.hemiLight);

    this.sunLight = new THREE.DirectionalLight(0xfffbeb, 1.45);
    this.sunLight.castShadow = initialTier.shadows;
    this.sunLight.shadow.mapSize.set(initialTier.shadowMapSize, initialTier.shadowMapSize);
    this.sunLight.shadow.bias = -0.00035;
    this.sunLight.shadow.normalBias = 0.025;
    this.sunLight.shadow.camera.near = 4.0;
    this.sunLight.shadow.camera.far = 260.0;
    const r = initialTier.shadowRadius;
    this.sunLight.shadow.camera.left = -r;
    this.sunLight.shadow.camera.right = r;
    this.sunLight.shadow.camera.top = r;
    this.sunLight.shadow.camera.bottom = -r;
    this.sunLight.shadow.camera.updateProjectionMatrix();
    this.scene.add(this.sunLight, this.sunLight.target);

    // Initialize Unified WebGPU / WebGL2 Renderer + TSL Sky Dome + TSL Bloom
    this.hybridRenderer = new HybridRenderer(this.canvas);
    await this.hybridRenderer.init(this.scene, this.camera, initialTier);

    // Physically Filtered PMREM Environment Map
    this.envMap = this.hybridRenderer.getOrCreateWeatherEnvMap(
      'MORNING',
      (p) => ProceduralAssetFactory.createEnvironmentCubeMap(p)
    );
    this.scene.environment = this.envMap;

    // Shared Procedural PBR Textures (Cached deterministically in ProceduralAssetFactory)
    this.textures = {
      microBump: ProceduralAssetFactory.createMicroBumpTexture(),
      ballast: ProceduralAssetFactory.createBallastAndSleeperTexture(),
      terrainDetail: ProceduralAssetFactory.createTerrainDetailTexture(),
      waterRipple: ProceduralAssetFactory.createWaterRippleTexture(),
      platformTop: ProceduralAssetFactory.createPlatformTopTexture(),
      platform: ProceduralAssetFactory.createStationPlatformTexture(),
      tunnel: ProceduralAssetFactory.createTunnelWallTexture(),
      skyscraper: ProceduralAssetFactory.createSkyscraperTexture()
    };
    this.textures.waterRipple.repeat.set(60, 60);
    this.textures.microBump.repeat.set(8, 8);

    // 1. Distant 360-Degree Horizon Mountain Silhouette Ring
    this.horizonRing = this.createDistantMountainRing();
    this.scene.add(this.horizonRing);

    // 2. Global Endless Valley Ground Floor Plane
    const vFloorGeo = new THREE.PlaneGeometry(3800, 3800);
    vFloorGeo.rotateX(-Math.PI / 2);
    this.valleyFloorMesh = new THREE.Mesh(
      vFloorGeo,
      new THREE.MeshStandardMaterial({
        color: 0x24471e,
        map: this.textures.terrainDetail,
        roughness: 0.92,
        metalness: 0.02
      })
    );
    this.valleyFloorMesh.position.y = -3.6;
    this.valleyFloorMesh.receiveShadow = true;
    this.scene.add(this.valleyFloorMesh);

    // 3. Global Shimmering Water Plane (Coastal Ocean & River Viaduct Gorge)
    const wGeo = new THREE.PlaneGeometry(2800, 2800);
    wGeo.rotateX(-Math.PI / 2);
    this.waterMesh = new THREE.Mesh(
      wGeo,
      new THREE.MeshPhysicalMaterial({
        color: 0x0284c7,
        map: this.textures.waterRipple,
        bumpMap: this.textures.waterRipple,
        bumpScale: 0.18,
        roughness: 0.07,
        metalness: 0.82,
        clearcoat: 0.9,
        clearcoatRoughness: 0.06,
        envMapIntensity: 1.55,
        transparent: true,
        opacity: 0.92
      })
    );
    this.waterMesh.position.y = -28.0;
    this.waterMesh.visible = false;
    this.scene.add(this.waterMesh);

    // 4. 3D Storm Rain Particles (Outside Train Cabin Envelope)
    const rGeo = new THREE.BufferGeometry();
    const rPos = new Float32Array(2400 * 3);
    for (let i = 0; i < 2400; i++) {
      let rx = (Math.random() - 0.5) * 140;
      const rz = (Math.random() - 0.5) * 140;
      if (Math.abs(rx) < 4.5 && rz > -26.0 && rz < 26.0) {
        rx = (rx >= 0 ? 1 : -1) * (4.8 + Math.random() * 38);
      }
      rPos[i * 3] = rx;
      rPos[i * 3 + 1] = Math.random() * 55;
      rPos[i * 3 + 2] = rz;
    }
    rGeo.setAttribute('position', new THREE.BufferAttribute(rPos, 3));
    this.rainPoints = new THREE.Points(
      rGeo,
      new THREE.PointsMaterial({ color: 0xbae6fd, size: 0.16, transparent: true, opacity: 0.42 })
    );
    this.rainPoints.visible = false;
    this.scene.add(this.rainPoints);
  }

  createDistantMountainRing() {
    const radius = 1950;
    const segs = 96;
    const geo = new THREE.BufferGeometry();
    const verts = [];
    const cols = [];
    const indices = [];

    const colBase = new THREE.Color(0x334155);
    const colPeak = new THREE.Color(0x93c5fd);

    for (let i = 0; i <= segs; i++) {
      const theta = (i / segs) * Math.PI * 2;
      const x = Math.cos(theta) * radius;
      const z = Math.sin(theta) * radius;
      const h = 85 + Math.sin(theta * 5) * 55 + Math.cos(theta * 13) * 32 + Math.sin(theta * 23) * 16;

      verts.push(x, -35, z);
      cols.push(colBase.r, colBase.g, colBase.b);
      verts.push(x, h, z);
      cols.push(colPeak.r, colPeak.g, colPeak.b);

      if (i < segs) {
        const b = i * 2;
        indices.push(b, b + 1, b + 2, b + 1, b + 3, b + 2);
      }
    }
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
    geo.setIndex(indices);
    return new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, fog: true })
    );
  }

  setWeatherPreset(preset) {
    this.weather = preset;
    const map = {
      MORNING: { skyTop: 0x0b2447, skyHor: 0x7dd3fc, fog: 0x93c5fd, sun: 2.25, hemi: 1.08, pos: [70, 55, 40], rain: false, clouds: 0.82, mins: 7 * 60 + 30 },
      NOON:    { skyTop: 0x0284c7, skyHor: 0xbae6fd, fog: 0xbae6fd, sun: 2.55, hemi: 1.15, pos: [20, 110, 10], rain: false, clouds: 0.65, mins: 12 * 60 + 15 },
      SUNSET:  { skyTop: 0x1e1b4b, skyHor: 0xf97316, fog: 0xfdba74, sun: 1.95, hemi: 0.82, pos: [-90, 22, -40], rain: false, clouds: 0.90, mins: 17 * 60 + 45 },
      STORM:   { skyTop: 0x020617, skyHor: 0x0f172a, fog: 0x0f172a, sun: 0.36, hemi: 0.38, pos: [-30, 60, -20], rain: true,  clouds: 0.98, mins: 21 * 60 + 40 }
    };
    const cfg = map[preset] || map.MORNING;
    this.weatherCfg = cfg;
    this.scene.background.setHex(cfg.skyHor);
    this.scene.fog.color.setHex(cfg.fog);
    this.sunLight.intensity = cfg.sun;
    this.hemiLight.intensity = cfg.hemi;
    this.sunOffset.set(cfg.pos[0], cfg.pos[1], cfg.pos[2]);
    this.rainPoints.visible = cfg.rain;
    this.clockMins = cfg.mins;

    if (this.hybridRenderer) {
      this.hybridRenderer.updateSkyWeather(cfg, this.sunOffset);
      this.envMap = this.hybridRenderer.getOrCreateWeatherEnvMap(
        preset,
        (p) => ProceduralAssetFactory.createEnvironmentCubeMap(p)
      );
      this.scene.environment = this.envMap;
    }

    if (this.world) this.world.setEnvMap(this.envMap);
  }

  resetSifa() {
    this.sifaTimer = 0.0;
    this.sifaWarning = false;
    const btn = document.getElementById('btn-sifa');
    const lbl = document.getElementById('lbl-sifa');
    if (btn) btn.classList.remove('active-rose');
    if (lbl) lbl.textContent = 'SIFA OK';
    if (this.train && this.train.sifaBtn3D) {
      this.train.sifaBtn3D.material.color.setHex(0x10b981);
    }
  }

  /**
   * ========================================================================
   * SUB-STEPPED VELOCITY VERLET LONGITUDINAL PHYSICS (N=2 SUBSTEPS)
   * ========================================================================
   */
  stepPhysics(dt) {
    const substeps = 2;
    const subDt = dt / substeps;
    const originS = this.floatingOrigin.originS;

    for (let step = 0; step < substeps; step++) {
      const cfg = GAME_PREMISE_CARTRIDGE.liveries[this.train.liveryKey];
      const massKg = (cfg.locoMassTons + this.train.numCoaches * cfg.carMassTons) * 1000;
      const frame = EndlessWorldMathematics.getTrackFrame(this.distanceS, 0, originS, _physFrame);

      if (this.autopilot) this.runAICoDriver(frame, subDt);

      if (this.keys['KeyW'] || this.keys['KeyA']) {
        this.setThrottle(this.throttlePct + 28 * subDt);
        this.resetSifa();
      }
      if (this.keys['KeyS'] || this.keys['KeyD']) {
        if (this.throttlePct > 0) {
          this.setThrottle(this.throttlePct - 35 * subDt);
        } else {
          this.setBrake(this.brakePct + 28 * subDt);
        }
        this.resetSifa();
      }
      if (this.keys['Quote']) {
        this.setBrake(this.brakePct + 32 * subDt);
        this.resetSifa();
      }
      if (this.keys['Semicolon']) {
        this.setBrake(this.brakePct - 32 * subDt);
        this.resetSifa();
      }

      // 1. Pneumatic Brake Pipe & Cylinder Propagation
      this.eqBar = 5.0 - (this.brakePct / 100) * 1.8;
      this.pipeBar += (this.eqBar - this.pipeBar) * Math.min(1, subDt * 1.85);
      const targetCyl = (this.brakePct / 100) * 3.8;
      const cylRate = this.brakePct > 85 ? 2.8 : 1.45;
      this.cylBar += (targetCyl - this.cylBar) * Math.min(1, subDt * cylRate);

      // 2.CTCS-3 / ATP Overspeed Protection & Jerk-Limited Torque Ramp
      const effectiveLimit = Math.min(frame.speedLimit, cfg.maxSpeedKmh);
      const absV = Math.abs(this.speedMps);
      const vKmh = absV * 3.6;

      if (!this.autopilot && vKmh > effectiveLimit + 6.0) {
        this.atpIntervention = true;
        if (this.throttlePct > 0) this.setThrottle(0);
        if (this.brakePct < 55) this.setBrake(55);
      } else if (this.atpIntervention && vKmh <= effectiveLimit - 1.5) {
        this.atpIntervention = false;
      }

      const pantoOk = cfg.tractionType !== 'ELECTRIC' || this.train.pantoRaised;
      const targetThr = (!this.train.doorsOpen && pantoOk && this.brakePct < 8 && !this.atpIntervention)
        ? this.throttlePct
        : 0;
      const rampRate = targetThr > this.actualThrottlePct ? 24.0 : 48.0;
      if (this.actualThrottlePct < targetThr) {
        this.actualThrottlePct = Math.min(targetThr, this.actualThrottlePct + rampRate * subDt);
      } else {
        this.actualThrottlePct = Math.max(targetThr, this.actualThrottlePct - rampRate * subDt);
      }

      // 3. Livery-Specific Power Curve + Curtius-Kniffler Rail-Wheel Adhesion Physics
      let tractionN = 0;
      this.wheelSlipping = false;
      const isEmu = (cfg.consistType === 'EMU');
      const emuScale = isEmu ? ((1 + this.train.numCoaches) / 9.0) : 1.0;

      if (!this.train.doorsOpen && pantoOk && this.reverser !== 0 && this.actualThrottlePct > 0.2) {
        const maxStartForceN = cfg.maxTractionKn * 1000 * emuScale;
        const maxPowerWatts = (cfg.powerMw || 3.6) * 1e6 * emuScale;
        const powerLimitedForceN = maxPowerWatts / Math.max(absV, 12.5);
        const requestedForceN = (this.actualThrottlePct / 100) * Math.min(maxStartForceN, powerLimitedForceN);

        // Curtius-Kniffler empirical adhesion law: mu_dry = 7.5 / (v_kmh + 44) + 0.161
        const ckDry = 7.5 / (vKmh + 44.0) + 0.161;
        let muRail = (this.weather === 'STORM') ? (ckDry * 0.56) : ckDry;
        if (this.sanderActive) muRail += 0.085;

        const adhesiveMassKg = isEmu
          ? (massKg * (cfg.motoredAxleRatio || 0.50))
          : (cfg.locoMassTons * 1000);
        const maxAdhesionN = adhesiveMassKg * 9.81 * muRail;

        let deliveredForceN = requestedForceN;
        if (requestedForceN > maxAdhesionN) {
          this.wheelSlipping = true;
          deliveredForceN = maxAdhesionN * 0.76; // Kinetic slip loss until throttle eased or Sander [X] engaged
        }
        tractionN = this.reverser * deliveredForceN;
      }

      // 4. Mass-Scaled Davis Drag + 1.48x Tunnel Piston Effect + Röckl Curve Drag + Grade Gravity
      const massTons = massKg / 1000;
      const rollingN = (14.5 + 0.22 * vKmh) * massTons;
      const baseAero = (cfg.noseType === 'BULLET')
        ? (5.2 + 0.34 * this.train.numCoaches)
        : (11.8 + 0.85 * this.train.numCoaches);
      const tunnelPistonMult = frame.inTunnel ? 1.48 : 1.0;
      const aeroN = baseAero * tunnelPistonMult * absV * absV;

      const absCurv = Math.abs(frame.curvature);
      const curveDragN = absCurv > 0.00005 ? (massTons * 9.81 * (absCurv * 65.0)) : 0.0;
      const dragN = (rollingN + aeroN + curveDragN) * (this.speedMps >= 0 ? 1 : -1);
      const gradeN = massKg * 9.81 * (frame.gradePermille / 1000);

      const brkRatio = this.cylBar / 3.8;
      const maxBrakeDecel = this.brakePct > 85 ? 1.18 : 0.78;
      const brakeN = brkRatio * (massKg * maxBrakeDecel);

      if (tractionN !== 0) {
        this.currentTractionKn = Math.round(Math.abs(tractionN) / 1000);
      } else if (cfg.tractionType === 'ELECTRIC' && pantoOk && this.brakePct > 5 && absV > 3.0) {
        // Regenerative electro-dynamic braking fed back into 25kV AC overhead catenary
        this.currentTractionKn = -Math.round(Math.min(brakeN * 0.42, 260000 * emuScale) / 1000);
      } else {
        this.currentTractionKn = 0;
      }

      const motiveN = tractionN - gradeN - dragN;
      this.prevAccel = this.accel;

      if (absV < 0.05 && this.brakePct > 5 && Math.abs(tractionN) < brakeN) {
        this.speedMps = 0;
        this.accel = 0;
      } else {
        const bDir = this.speedMps > 0 ? 1 : (this.speedMps < 0 ? -1 : Math.sign(motiveN));
        const netF = motiveN - bDir * brakeN;
        this.accel = netF / massKg;
        this.speedMps += this.accel * subDt;
        if (bDir > 0 && this.speedMps < 0 && tractionN <= 0) this.speedMps = 0;
        if (bDir < 0 && this.speedMps > 0 && tractionN >= 0) this.speedMps = 0;
      }

      const maxMps = (cfg.maxSpeedKmh / 3.6) * 1.03;
      this.speedMps = THREE.MathUtils.clamp(this.speedMps, -15, maxMps);
      this.distanceS = Math.max(10, this.distanceS + this.speedMps * subDt);
    }

    // SIFA Vigilance Alerter Check
    if (!this.autopilot && Math.abs(this.speedMps) > 9.7) {
      this.sifaTimer += dt;
      if (this.sifaTimer > 22.0 && !this.sifaWarning) {
        this.sifaWarning = true;
        this.audio.playSifaBeep();
        const btn = document.getElementById('btn-sifa');
        const lbl = document.getElementById('lbl-sifa');
        if (btn) btn.classList.add('active-rose');
        if (lbl) lbl.textContent = 'TEKAN [Q]!';
        if (this.train && this.train.sifaBtn3D) {
          this.train.sifaBtn3D.material.color.setHex(0xf43f5e);
        }
      }
      if (this.sifaTimer > 30.0) {
        this.setThrottle(0);
        this.setBrake(100);
        this.resetSifa();
      }
    } else {
      this.sifaTimer = 0.0;
    }

    // Rail Joint Acoustic Trigger
    const frame = EndlessWorldMathematics.getTrackFrame(this.distanceS, 0, originS, _physFrame);
    const curJoint = Math.floor(this.distanceS / GAME_PREMISE_CARTRIDGE.railway.jointInterval);
    const inResonantStructure = frame.inTunnel || EndlessWorldMathematics.isViaductZone(this.distanceS);
    if (curJoint !== this.lastJointIdx) {
      this.lastJointIdx = curJoint;
      this.audio.playRailJointClack(this.speedMps, inResonantStructure, this.lastPlayerDoppler || 1.0);
    }

    // Comfort Evaluation
    const cfgNow = GAME_PREMISE_CARTRIDGE.liveries[this.train.liveryKey];
    const effLimitNow = Math.min(frame.speedLimit, cfgNow.maxSpeedKmh);
    const jerk = Math.abs(this.accel - this.prevAccel) / Math.max(0.008, dt);
    const speedKmh = Math.abs(this.speedMps) * 3.6;
    if (jerk > 2.6 || speedKmh > effLimitNow + 5) {
      this.comfort = Math.max(40, this.comfort - dt * 3.2);
    } else {
      this.comfort = Math.min(100, this.comfort + dt * 0.7);
    }

    this.evaluateStationStop(dt);
  }

  runAICoDriver(frame, dt) {
    const stInt = GAME_PREMISE_CARTRIDGE.railway.stationInterval;
    const nextStIdx = Math.max(1, Math.ceil((this.distanceS - 18) / stInt));
    const targetS = nextStIdx * stInt + 48;
    const distToStop = targetS - this.distanceS;

    if (this.aiStationStage !== 'CRUISE') {
      this.speedMps = 0;
      this.setThrottle(0);
      this.setBrake(65);
      this.aiStageTimer -= dt;

      if (this.aiStationStage === 'STOPPING_HOLD') {
        if (this.aiStageTimer <= 0) {
          this.toggleDoors(true);
          this.aiStationStage = 'BOARDING';
          this.aiStageTimer = 8.5;
        }
      } else if (this.aiStationStage === 'BOARDING') {
        if (this.aiStageTimer <= 0 && this.boardingProgress >= 100) {
          this.toggleDoors(false);
          this.audio.playChime('DEPART');
          this.aiStationStage = 'DOORS_CLOSING';
          this.aiStageTimer = 2.0;
        }
      } else if (this.aiStationStage === 'DOORS_CLOSING') {
        if (this.aiStageTimer <= 0) {
          this.audio.init();
          this.audio.setHorn(true);
          this.aiStationStage = 'HORN_DEPART';
          this.aiStageTimer = 1.3;
          document.getElementById('eval-boarding-status').innerHTML =
            `<i class="ph-fill ph-megaphone text-amber-400 text-base"></i><span>Semboyan 35! Pintu terkunci, kereta diberangkatkan...</span>`;
        }
      } else if (this.aiStationStage === 'HORN_DEPART') {
        if (this.aiStageTimer <= 0) {
          this.audio.setHorn(false);
          this.lastEvalStIdx = nextStIdx + 0.5;
          this.aiStationStage = 'CRUISE';
          this.setBrake(0);
          this.setThrottle(35);
        }
      }
      return;
    }

    if (this.lastEvalStIdx < nextStIdx && distToStop <= 0.45 && distToStop > -18.0 && Math.abs(this.speedMps) < 1.8) {
      this.speedMps = 0;
      this.distanceS = targetS;
      this.setThrottle(0);
      this.setBrake(70);
      this.aiStationStage = 'STOPPING_HOLD';
      this.aiStageTimer = 1.6;
      return;
    }

    if (this.train.doorsOpen) {
      this.setThrottle(0);
      this.setBrake(60);
      return;
    }

    const cfg = GAME_PREMISE_CARTRIDGE.liveries[this.train.liveryKey];
    if (cfg.tractionType === 'ELECTRIC' && !this.train.pantoRaised) {
      this.train.pantoRaised = true;
      const pBtn = document.getElementById('btn-panto');
      if (pBtn) pBtn.classList.add('active');
    }
    this.reverser = 1;
    const effLimit = Math.min(frame.speedLimit, cfg.maxSpeedKmh);
    const maxSafeMps = (effLimit - 4) / 3.6;
    let targetMps = maxSafeMps;

    if (distToStop > 0 && distToStop < 680 && this.lastEvalStIdx < nextStIdx) {
      const decelProfile = Math.sqrt(Math.max(0, 2 * 0.72 * Math.max(0, distToStop - 0.15)));
      targetMps = Math.min(maxSafeMps, Math.max(0.85, decelProfile));
      if (distToStop < 0.35) targetMps = 0;
    }

    const err = targetMps - this.speedMps;
    if (targetMps === 0) {
      this.setThrottle(0);
      this.setBrake(70);
    } else if (err > 0.4) {
      this.setBrake(0);
      this.setThrottle(THREE.MathUtils.clamp(err * 20, 14, 88));
    } else if (err < -0.25) {
      this.setThrottle(0);
      this.setBrake(THREE.MathUtils.clamp(-err * 28, 12, 85));
    } else {
      this.setThrottle(14);
      this.setBrake(0);
    }
  }

  evaluateStationStop(dt) {
    const stInt = GAME_PREMISE_CARTRIDGE.railway.stationInterval;
    const nearIdx = Math.max(1, Math.round(this.distanceS / stInt));
    const stopS = nearIdx * stInt + 48;
    const errM = this.distanceS - stopS;

    const dockPanel = document.getElementById('docking-guide-panel');
    if (Math.abs(errM) < 130) {
      dockPanel.classList.remove('hidden');
      dockPanel.classList.add('flex');
      const sign = errM >= 0 ? '+' : '';
      document.getElementById('docking-dist-readout').textContent = `${sign}${errM.toFixed(2)} m`;
      const pct = THREE.MathUtils.clamp(50 + (errM / 60) * 45, 4, 96);
      document.getElementById('docking-nose-marker').style.left = `${pct}%`;
    } else {
      dockPanel.classList.add('hidden');
      dockPanel.classList.remove('flex');
    }

    if (Math.abs(this.speedMps) < 0.08 && Math.abs(errM) < 45 && Math.floor(this.lastEvalStIdx) !== nearIdx) {
      this.lastEvalStIdx = nearIdx;
      this.boardingProgress = 0;
      const absE = Math.abs(errM);
      let grade = 'B GOOD';
      let pts = 750;
      if (absE <= 0.55) { grade = 'S+ MASTER!'; pts = 2000; }
      else if (absE <= 2.2) { grade = 'A EXCELLENT'; pts = 1400; }

      this.score += pts;
      this.audio.playChime('STATION');

      const stName = GAME_PREMISE_CARTRIDGE.stations[(nearIdx - 1) % GAME_PREMISE_CARTRIDGE.stations.length];
      document.getElementById('eval-st-name').textContent = stName;
      document.getElementById('eval-rank-pill').textContent = grade;
      document.getElementById('eval-err-m').textContent = `${errM >= 0 ? '+' : ''}${errM.toFixed(2)} m`;
      document.getElementById('eval-comfort-score').textContent = `${Math.round(this.comfort)}%`;
      document.getElementById('eval-pts-earned').textContent = `+${pts.toLocaleString()}`;
      document.getElementById('eval-boarding-progress').style.width = '0%';
      document.getElementById('station-eval-modal').classList.remove('hidden');
    } else if (Math.abs(errM) > 65 && Math.abs(this.speedMps) > 4.0) {
      document.getElementById('station-eval-modal').classList.add('hidden');
    }

    if (this.train.doorsOpen && !document.getElementById('station-eval-modal').classList.contains('hidden')) {
      this.boardingProgress = Math.min(100, this.boardingProgress + dt * 13.5);
      document.getElementById('eval-boarding-progress').style.width = `${Math.round(this.boardingProgress)}%`;
      if (this.boardingProgress < 45) {
        document.getElementById('eval-boarding-status').innerHTML =
          `<i class="ph-fill ph-users-three text-amber-400 text-base"></i><span>Penumpang turun & naik gerbong (${Math.round(this.boardingProgress)}%)...</span>`;
      } else if (this.boardingProgress < 100) {
        document.getElementById('eval-boarding-status').innerHTML =
          `<i class="ph-fill ph-users-three text-cyan-400 text-base"></i><span>Penumpang sedang naik ke gerbong (${Math.round(this.boardingProgress)}%)...</span>`;
      } else {
        document.getElementById('eval-boarding-status').innerHTML =
          `<i class="ph-fill ph-check-circle text-emerald-400 text-base"></i><span>Penumpang selesai naik (100%)! Tutup <strong>[DOORS / O]</strong>, klakson <strong>[SPACE]</strong> & jalan.</span>`;
      }
    }
  }

  /**
   * ========================================================================
   * 7-CAMERA SUITE + TEXEL-SNAPPED SHADOWS + ZERO-GC 3D DOPPLER SOLVER
   * ========================================================================
   */
  updateCameraSystem(dt) {
    const loco = this.train.vehicles[0];
    if (!loco) {
      return { playerDoppler: 1.0, aiDoppler: 1.0, aiDist: 999, crossingDist: 999, crossingDoppler: 1.0 };
    }

    if (this.train.extCabShell) this.train.extCabShell.visible = (this.cameraMode !== 0);
    if (this.train.interiorCabGroup) this.train.interiorCabGroup.visible = (this.cameraMode === 0);
    if (this.train.coachViewShell) this.train.coachViewShell.visible = (this.cameraMode !== 4);
    if (this.train.interiorCoachGroup) this.train.interiorCoachGroup.visible = (this.cameraMode === 4);

    loco.bodyGroup.getWorldPosition(_locoPos);
    loco.bodyGroup.getWorldQuaternion(_locoQuat);

    const originS = this.floatingOrigin.originS;
    const aiDistS = Math.abs(this.train.aiTrainS - this.distanceS);
    const passShake = aiDistS < 42 ? (1 - aiDistS / 42) * 0.035 : 0;

    if (this.cameraMode === 0) {
      _tmpVecA.set((Math.random() - 0.5) * passShake, 3.26 + (Math.random() - 0.5) * passShake, -9.08);
      _tmpVecA.applyQuaternion(_locoQuat);
      this.camera.position.copy(_locoPos).add(_tmpVecA);
      _lookEuler.set(this.cabPitch, this.cabYaw, 0, 'YXZ');
      _lookQuat.setFromEuler(_lookEuler);
      this.camera.quaternion.copy(_locoQuat).multiply(_lookQuat);
    } else if (this.cameraMode === 1) {
      _tmpVecA.copy(_locoPos);
      _tmpVecA.y += 2.4;
      const ox = Math.sin(this.orbitYaw) * Math.cos(this.orbitPitch) * this.orbitDist;
      const oy = Math.sin(this.orbitPitch) * this.orbitDist + 2.4;
      const oz = Math.cos(this.orbitYaw) * Math.cos(this.orbitPitch) * this.orbitDist;
      _tmpVecB.set(ox, Math.max(1.2, oy), oz).applyQuaternion(_locoQuat);
      this.camera.position.copy(_tmpVecA).add(_tmpVecB);
      this.camera.lookAt(_tmpVecA);
    } else if (this.cameraMode === 2) {
      if (Math.abs(this.distanceS - this.flyByAnchorS) > 135) {
        this.flyByAnchorS = this.distanceS + (this.speedMps >= 0 ? 95 : -75);
        const f = EndlessWorldMathematics.getTrackFrame(this.flyByAnchorS, 0, originS, _camFrame);
        this.flyByPos.copy(f.position).addScaledVector(f.right, -5.8).addScaledVector(f.up, 2.3);
      }
      this.camera.position.copy(this.flyByPos);
      _tmpVecA.copy(_locoPos);
      _tmpVecA.y += 2.2;
      this.camera.lookAt(_tmpVecA);
    } else if (this.cameraMode === 3) {
      _tmpVecA.set(-1.62, 0.64, -2.8).applyQuaternion(_locoQuat);
      _tmpVecB.set(-0.72, 0.565, -6.3).applyQuaternion(_locoQuat);
      this.camera.position.copy(_locoPos).add(_tmpVecA);
      _tmpVecC.copy(_locoPos).add(_tmpVecB);
      this.camera.lookAt(_tmpVecC);
    } else if (this.cameraMode === 4) {
      const coach = this.train.vehicles[Math.min(2, this.train.vehicles.length - 1)];
      coach.bodyGroup.getWorldPosition(_coachPos);
      coach.bodyGroup.getWorldQuaternion(_coachQuat);
      _tmpVecA.set(-0.98, 2.42, 0.0).applyQuaternion(_coachQuat);
      this.camera.position.copy(_coachPos).add(_tmpVecA);
      _lookEuler.set(this.cabPitch - 0.04, 1.12 + this.cabYaw * 0.65, 0, 'YXZ');
      _lookQuat.setFromEuler(_lookEuler);
      this.camera.quaternion.copy(_coachQuat).multiply(_lookQuat);
    } else if (this.cameraMode === 5) {
      const walkSpeed = 7.0 * dt;
      if (this.keys['ArrowUp']) this.walkOffset.z -= walkSpeed;
      if (this.keys['ArrowDown']) this.walkOffset.z += walkSpeed;
      if (this.keys['ArrowLeft']) this.walkOffset.x -= walkSpeed;
      if (this.keys['ArrowRight']) this.walkOffset.x += walkSpeed;

      const wf = EndlessWorldMathematics.getTrackFrame(this.distanceS - this.walkOffset.z, 0, originS, _camFrame);
      const wx = wf.position.x + wf.right.x * this.walkOffset.x;
      const wz = wf.position.z + wf.right.z * this.walkOffset.x;
      const walkWorldS = this.distanceS - this.walkOffset.z;
      const nearPlat = EndlessWorldMathematics.isNearStation(walkWorldS, 85);
      let groundY = wf.position.y - 0.50;
      if (nearPlat && ((this.walkOffset.x >= -8.75 && this.walkOffset.x <= -1.95) || (this.walkOffset.x >= 6.35 && this.walkOffset.x <= 13.15))) {
        groundY = wf.position.y + 1.00;
      } else if (this.walkOffset.x >= 16.5 && this.walkOffset.x <= 26.8) {
        groundY = EndlessWorldMathematics.getHighwayElevation(walkWorldS, wf.position.y);
      } else if (this.walkOffset.x < -6.2 || this.walkOffset.x > 26.8) {
        groundY = EndlessWorldMathematics.getTerrainHeightAtS(wx, walkWorldS);
      }
      this.camera.position.set(wx, groundY + 1.68, wz);
      const trackYaw = Math.atan2(-wf.tangent.x, -wf.tangent.z);
      _lookEuler.set(this.cabPitch, trackYaw + this.cabYaw, 0, 'YXZ');
      this.camera.quaternion.setFromEuler(_lookEuler);
    } else if (this.cameraMode === 6) {
      const t = performance.now() * 0.00035;
      _tmpVecA.set(Math.sin(t) * 34.0, 18.0 + Math.cos(t * 2) * 5.0, -24.0 + Math.cos(t) * 28.0);
      _tmpVecA.applyQuaternion(_locoQuat);
      this.camera.position.copy(_locoPos).add(_tmpVecA);
      _tmpVecB.set(_locoPos.x, _locoPos.y + 2.5, _locoPos.z + 4);
      this.camera.lookAt(_tmpVecB);
    }

    // Compute true 3D Camera Velocity & Multi-Source Acoustic Doppler Shift
    const safeDt = Math.max(0.005, dt);
    if (this.prevCamPos.lengthSq() > 0) {
      this.camVel.subVectors(this.camera.position, this.prevCamPos).divideScalar(safeDt);
    }
    this.prevCamPos.copy(this.camera.position);

    const cSound = 343.0;
    const playerFrame = EndlessWorldMathematics.getTrackFrame(this.distanceS, 0, originS, _camFrame);
    _playerVel.copy(playerFrame.tangent).multiplyScalar(this.speedMps);

    // 1. Player Train Doppler
    let playerDoppler = 1.0;
    _tmpVecA.subVectors(this.camera.position, _locoPos);
    const distPlayerToCam = _tmpVecA.length();
    if (this.cameraMode === 2 || this.cameraMode === 5 || this.cameraMode === 6) {
      _tmpVecB.copy(_tmpVecA).divideScalar(Math.max(0.5, distPlayerToCam));
      const vSrcRadial = THREE.MathUtils.clamp(_playerVel.dot(_tmpVecB), -120, 120);
      const vObsRadial = (this.cameraMode === 2) ? 0 : THREE.MathUtils.clamp(this.camVel.dot(_tmpVecB), -120, 120);
      playerDoppler = THREE.MathUtils.clamp((cSound - vObsRadial) / (cSound - vSrcRadial), 0.72, 1.38);
    }
    this.lastPlayerDoppler = playerDoppler;

    // 2. Oncoming AI Train Doppler
    const track2Off = GAME_PREMISE_CARTRIDGE.railway.doubleTrackOffset;
    const aiFrame = EndlessWorldMathematics.getTrackFrame(this.train.aiTrainS, track2Off, originS, _aiFrame);
    _aiVel.copy(aiFrame.tangent).multiplyScalar(-this.train.aiSpeedMps);
    _tmpVecA.subVectors(this.camera.position, aiFrame.position);
    const aiDist3D = _tmpVecA.length();
    _tmpVecB.copy(_tmpVecA).divideScalar(Math.max(1.0, aiDist3D));
    const vAiSrcRadial = THREE.MathUtils.clamp(_aiVel.dot(_tmpVecB), -120, 120);
    const vCamObsRadial = (this.cameraMode === 2) ? 0 : THREE.MathUtils.clamp(_playerVel.dot(_tmpVecB), -120, 120);
    const aiDoppler = THREE.MathUtils.clamp((cSound - vCamObsRadial) / (cSound - vAiSrcRadial), 0.68, 1.48);

    // 3. Nearest Level Crossing Bell Doppler
    const cxInt = GAME_PREMISE_CARTRIDGE.railway.crossingInterval;
    const nearCxS = Math.round(this.distanceS / cxInt) * cxInt;
    const cxFrame = EndlessWorldMathematics.getTrackFrame(nearCxS, 2.2, originS, _cxFrame);
    _tmpVecA.subVectors(this.camera.position, cxFrame.position);
    const crossingDist = _tmpVecA.length();
    _tmpVecB.copy(_tmpVecA).divideScalar(Math.max(1.0, crossingDist));
    const vObsCx = (this.cameraMode === 2) ? 0 : THREE.MathUtils.clamp(_playerVel.dot(_tmpVecB), -110, 110);
    const crossingDoppler = THREE.MathUtils.clamp((cSound - vObsCx) / cSound, 0.78, 1.25);

    // 4. Texel-Snapped Directional Sun Shadow Update (Zero shadow shimmer!)
    this.governor.updateDirectionalShadow(
      this.sunLight,
      _locoPos,
      this.sunOffset
    );

    // 5. Update TSL Sky Dome & Horizon Silhouette Ring
    this.hybridRenderer.updateSkyFrame(this.camera.position, performance.now() * 0.001);
    if (this.horizonRing) {
      this.horizonRing.position.x = this.camera.position.x;
      this.horizonRing.position.z = this.camera.position.z;
    }

    const { biome } = EndlessWorldMathematics.getBiomeAtDistance(this.distanceS);
    const camWorldS = originS - this.camera.position.z;
    const camInTunnel = EndlessWorldMathematics.isTunnelZone(camWorldS);
    const underCanopy = EndlessWorldMathematics.isNearStation(this.distanceS, 125) && (this.cameraMode === 0 || this.cameraMode === 4);

    // Smooth Tunnel Sunlight, Sky & Fog Occlusion (prevents bright sunlight piercing solid mountain rock!)
    const targetOcc = camInTunnel ? 1.0 : 0.0;
    this.tunnelOcclusion = (this.tunnelOcclusion || 0) + (targetOcc - (this.tunnelOcclusion || 0)) * Math.min(1, dt * 5.0);
    const occ = this.tunnelOcclusion;
    const wCfg = this.weatherCfg || { sun: 2.25, hemi: 1.08, fog: 0x93c5fd, skyHor: 0x7dd3fc, rain: false };
    this.sunLight.intensity = wCfg.sun * (1.0 - 0.94 * occ);
    this.hemiLight.intensity = wCfg.hemi * (1.0 - 0.70 * occ);
    this.scene.fog.color.setHex(wCfg.fog).lerp(new THREE.Color(0x0b101b), occ * 0.92);
    if (this.hybridRenderer && this.hybridRenderer.skyDome) {
      this.hybridRenderer.skyDome.visible = occ < 0.96;
    }

    if (this.valleyFloorMesh) {
      this.valleyFloorMesh.position.x = _locoPos.x;
      this.valleyFloorMesh.position.z = _locoPos.z;
      this.valleyFloorMesh.material.color.setHex(biome.groundColor);
      // Steady below deepest -12.2m river thalweg so chunk-local 3D rivers never pop or z-fight!
      this.valleyFloorMesh.position.y = -13.8;
      this.valleyFloorMesh.visible = (biome.id !== 'COASTAL');
    }

    // Distant horizon ocean plane for COASTAL biome (chunk-local 3D water meshes render rivers & surf!)
    this.waterMesh.visible = (biome.id === 'COASTAL');
    if (this.waterMesh.visible) {
      this.waterMesh.position.x = _locoPos.x;
      this.waterMesh.position.z = _locoPos.z;
      if (this.textures.waterRipple) {
        this.textures.waterRipple.offset.x = (performance.now() * 0.00008) % 1;
        this.textures.waterRipple.offset.y = (performance.now() * 0.00006) % 1;
      }
      this.waterMesh.position.y = -2.46;
    }

    this.rainPoints.visible = Boolean(wCfg.rain && !camInTunnel && !underCanopy);
    if (this.rainPoints.visible) {
      this.rainPoints.position.copy(_locoPos);
      const attr = this.rainPoints.geometry.attributes.position;
      for (let i = 0; i < attr.count; i++) {
        let y = attr.getY(i) - 38 * dt;
        if (y < -4) y = 46;
        attr.setY(i, y);
      }
      attr.needsUpdate = true;
    }

    // 6. Laminated Windshield & Coach Panoramic Window Overlay
    this.windshieldFX.render({
      dt,
      cameraMode: this.cameraMode,
      distanceS: this.distanceS,
      speedMps: this.speedMps,
      weather: this.weather,
      cabYaw: this.cabYaw,
      cabPitch: this.cabPitch,
      curvature: playerFrame.curvature,
      inTunnel: camInTunnel,
      underCanopy,
      wipersActive: this.train.wipersActive,
      wiperPhase: this.train.wiperPhase,
      maxDpr: this.governor.activeTier.maxDpr
    });

    return { playerDoppler, aiDoppler, aiDist: aiDist3D, crossingDist, crossingDoppler };
  }

  /**
   * ========================================================================
   * HUD, GPU ENGINE BADGE & TRACK PROFILE RIBBON RENDERING
   * ========================================================================
   */
  updateEngineTelemetryUI(fps) {
    const fpsEl = document.getElementById('hud-fps');
    if (fpsEl) fpsEl.textContent = `${fps} FPS`;

    if (!this.hybridRenderer) return;
    const tel = this.hybridRenderer.getTelemetry();
    const tier = this.governor.activeTier;
    const modeLabel = this.governor.mode === 'AUTO' ? `AUTO(${tier.id})` : tier.id;

    const badgeEl = document.getElementById('hud-engine-badge');
    if (badgeEl) {
      badgeEl.textContent = `${tel.backend} • ${modeLabel} • ${tel.drawCalls} DC`;
    }

    const hangarStatus = document.getElementById('lbl-quality-status');
    if (hangarStatus) {
      hangarStatus.textContent = `${tel.backend} • ${tier.label} • ${tel.drawCalls} Draw Calls`;
    }
  }

  updateHUD() {
    const speedKmh = Math.round(Math.abs(this.speedMps) * 3.6);
    const frame = EndlessWorldMathematics.getTrackFrame(
      this.distanceS,
      0,
      this.floatingOrigin.originS,
      _physFrame
    );
    const { biome } = EndlessWorldMathematics.getBiomeAtDistance(this.distanceS);
    const cfg = GAME_PREMISE_CARTRIDGE.liveries[this.train.liveryKey];
    const effectiveLimit = Math.min(frame.speedLimit, cfg.maxSpeedKmh);

    document.getElementById('hud-speed').textContent = speedKmh;
    document.getElementById('hud-limit').textContent = effectiveLimit;
    document.getElementById('hud-odometer').textContent = `${(this.distanceS / 1000).toFixed(2)} km`;
    document.getElementById('hud-biome-name').textContent = biome.name;

    const hh = Math.floor((this.clockMins / 60) % 24).toString().padStart(2, '0');
    const mm = Math.floor(this.clockMins % 60).toString().padStart(2, '0');
    document.getElementById('hud-clock').textContent = `${hh}:${mm}`;

    const stInt = GAME_PREMISE_CARTRIDGE.railway.stationInterval;
    const nextStIdx = Math.max(1, Math.ceil((this.distanceS - 12) / stInt));
    const targetS = nextStIdx * stInt + 48;
    const distM = Math.max(0, Math.round(targetS - this.distanceS));
    const stName = GAME_PREMISE_CARTRIDGE.stations[(nextStIdx - 1) % GAME_PREMISE_CARTRIDGE.stations.length];

    document.getElementById('hud-next-station').textContent = stName;
    document.getElementById('hud-station-dist').textContent = `${distM.toLocaleString()} m`;
    document.getElementById('hud-next-station-mob').textContent = stName;
    document.getElementById('hud-station-dist-mob').textContent = `${distM.toLocaleString()} m`;

    const oncomingEl = document.getElementById('hud-oncoming-info');
    if (oncomingEl) {
      const aiCfg = GAME_PREMISE_CARTRIDGE.liveries[this.train.currentAiLiveryKey];
      const aiDistM = Math.round(this.train.aiTrainS - this.distanceS);
      const distLabel = aiDistM > 0 ? `${aiDistM}m` : 'LEWAT';
      oncomingEl.textContent = `${aiCfg ? aiCfg.name : this.train.currentAiLiveryKey} (${distLabel})`;
    }

    const etaSec = this.speedMps > 1.5 ? Math.round(distM / this.speedMps) : 99;
    const etaM = Math.floor(etaSec / 60).toString().padStart(2, '0');
    const etaS = (etaSec % 60).toString().padStart(2, '0');
    document.getElementById('hud-eta').textContent = `${etaM}:${etaS}`;

    const gStr = `${frame.gradePermille >= 0 ? '+' : ''}${frame.gradePermille.toFixed(1)}‰`;
    document.getElementById('hud-grade').textContent = gStr;
    document.getElementById('hud-grade-mob').textContent = gStr;
    const absC = Math.abs(frame.curvature);
    const curveEl = document.getElementById('hud-curve');
    if (curveEl) {
      curveEl.textContent = absC < 0.00010 ? 'R-∞' : `R-${Math.round(1 / absC)}m`;
    }

    const lineStatusEl = document.getElementById('hud-line-status');
    if (lineStatusEl) {
      if (frame.inTunnel) lineStatusEl.textContent = 'HSR TUNNEL BORING';
      else if (EndlessWorldMathematics.isViaductZone(this.distanceS)) lineStatusEl.textContent = 'ELEVATED VIADUCT';
      else if (absC >= 0.00010) lineStatusEl.textContent = 'SUPERELEVATED CURVE';
      else lineStatusEl.textContent = `HSR ${effectiveLimit} KM/H STRAIGHT`;
    }

    const notch = Math.round((this.throttlePct / 100) * 8);
    document.getElementById('hud-thr-badge').textContent = notch === 0 ? 'IDLE (N0)' : `NOTCH N${notch} (${Math.round(this.throttlePct)}%)`;
    const rpmVal = cfg.tractionType === 'ELECTRIC'
      ? Math.round(Math.abs(this.speedMps) * 58)
      : Math.round(450 + (notch / 8) * 600);
    document.getElementById('hud-rpm').textContent = rpmVal;
    document.getElementById('hud-kn').textContent = this.currentTractionKn < 0
      ? `${this.currentTractionKn} kN (REGEN)`
      : `${this.currentTractionKn} kN`;

    const slipEl = document.getElementById('hud-slip-indicator');
    slipEl.textContent = this.wheelSlipping ? 'WHEEL SLIP!' : 'GRIP OK';
    slipEl.className = this.wheelSlipping ? 'text-rose-400 font-extrabold' : 'text-emerald-400 font-bold';

    document.getElementById('hud-brk-badge').textContent = `${this.brakePct < 3 ? 'REL' : (this.brakePct > 88 ? 'EMR' : 'SRV')} (${Math.round(this.brakePct)}%)`;
    document.getElementById('hud-bp').textContent = `${this.pipeBar.toFixed(1)} BAR`;
    document.getElementById('hud-bc').textContent = `${this.cylBar.toFixed(1)} BAR`;
    document.getElementById('hud-eq').textContent = `${this.eqBar.toFixed(1)} BAR`;

    document.getElementById('hud-score').textContent = this.score.toLocaleString();
    const cPct = Math.round(this.comfort);
    document.getElementById('hud-comfort-pct').textContent = `${cPct}%`;
    document.getElementById('hud-comfort-bar').style.width = `${cPct}%`;
    const rankEl = document.getElementById('hud-rank');
    if (rankEl) {
      rankEl.textContent = cPct >= 94 ? 'S+ MASTER' : (cPct >= 82 ? 'A PRO' : (cPct >= 65 ? 'B FAIR' : 'C ROUGH'));
    }

    this.train.update3DMFDTexture(speedKmh, effectiveLimit, this.throttlePct, this.pipeBar, distM, this.sifaWarning);

    // Signal Aspect Lamps
    const rL = document.getElementById('sig-red');
    const yL = document.getElementById('sig-yellow');
    const gL = document.getElementById('sig-green');
    rL.className = 'w-3 h-3 rounded-full bg-rose-950 border border-rose-800/50';
    yL.className = 'w-3 h-3 rounded-full bg-amber-950 border border-amber-800/50';
    gL.className = 'w-3 h-3 rounded-full bg-emerald-950 border border-emerald-800/50';
    if (this.train.doorsOpen) {
      rL.className = 'w-3 h-3 rounded-full bg-rose-500 shadow-[0_0_10px_#f43f5e]';
    } else if (distM < 550) {
      yL.className = 'w-3 h-3 rounded-full bg-amber-400 shadow-[0_0_10px_#f59e0b]';
    } else {
      gL.className = 'w-3 h-3 rounded-full bg-emerald-400 shadow-[0_0_10px_#10b981]';
    }

    // Alert Banner
    const alertBox = document.getElementById('hud-alert-banner');
    const alertTxt = document.getElementById('hud-alert-text');
    if (this.sifaWarning) {
      alertBox.classList.remove('hidden');
      alertTxt.textContent = 'PERINGATAN SIFA VIGILANCE: TEKAN [Q] ATAU TOMBOL SIFA SEGERA!';
    } else if (this.atpIntervention || speedKmh > effectiveLimit + 3) {
      alertBox.classList.remove('hidden');
      alertTxt.textContent = this.atpIntervention
        ? `INTERVENSI REM OTOMATIS ATP: KECEPATAN MELEBIHI BATAS ${effectiveLimit} KM/H!`
        : `PERINGATAN ATP: OVERSPEED (${speedKmh}/${effectiveLimit} KM/H)! KURANGI KECEPATAN!`;
    } else if (this.wheelSlipping) {
      alertBox.classList.remove('hidden');
      alertTxt.textContent = 'RODA SELIP DI REL BASAH! AKTIFKAN SANDER [X] ATAU TURUNKAN THROTTLE!';
    } else if (this.train.doorsOpen && this.throttlePct > 5) {
      alertBox.classList.remove('hidden');
      alertTxt.textContent = 'INTERLOCK KEAMANAN: TUTUP PINTU PENUMPANG [O] SEBELUM BERANGKAT!';
    } else if (cfg.tractionType === 'ELECTRIC' && !this.train.pantoRaised && this.throttlePct > 5) {
      alertBox.classList.remove('hidden');
      alertTxt.textContent = 'PANTOGRAF TURUN: NAIKKAN PANTOGRAF [U] UNTUK MENARIK ARUS LAA!';
    } else {
      alertBox.classList.add('hidden');
    }

    this.renderTrackRibbon(targetS);
  }

  renderTrackRibbon(targetStopS) {
    const c = this.ribbonCanvas;
    if (!c || !this.ribbonCtx || c.offsetParent === null) return;
    if (c.width !== c.clientWidth || c.height !== c.clientHeight) {
      c.width = c.clientWidth;
      c.height = c.clientHeight;
    }
    const ctx = this.ribbonCtx;
    const w = c.width;
    const h = c.height;
    ctx.clearRect(0, 0, w, h);

    const spanM = 1200;
    const originS = this.floatingOrigin.originS;
    ctx.beginPath();
    ctx.moveTo(28, h);
    for (let x = 28; x <= w; x += 8) {
      const frac = (x - 28) / Math.max(1, w - 28);
      const s = this.distanceS + frac * spanM;
      const p = EndlessWorldMathematics.getTrackPosition(s, 0, originS, _ribbonPt);
      const py = h - 4 - THREE.MathUtils.clamp((p.y - 2.0) * 1.15, 2, h - 7);
      ctx.lineTo(x, py);
      if (EndlessWorldMathematics.isTunnelZone(s)) {
        ctx.fillStyle = 'rgba(148, 163, 184, 0.35)';
        ctx.fillRect(x - 4, 0, 8, 4);
      } else if (EndlessWorldMathematics.isViaductZone(s)) {
        ctx.fillStyle = 'rgba(56, 189, 248, 0.32)';
        ctx.fillRect(x - 4, h - 3, 8, 3);
      }
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(6, 182, 212, 0.24)';
    ctx.fill();

    const aiRel = this.train.aiTrainS - this.distanceS;
    if (aiRel >= 0 && aiRel <= spanM) {
      const ax = 28 + (aiRel / spanM) * (w - 28);
      ctx.fillStyle = '#f43f5e';
      ctx.fillRect(ax - 6, h * 0.35, 12, 5);
      ctx.font = 'bold 8px JetBrains Mono';
      ctx.fillStyle = '#fda4af';
      ctx.fillText(this.train.currentAiLiveryKey, Math.max(30, Math.min(w - 58, ax - 18)), h * 0.30);
    }

    const stRel = targetStopS - this.distanceS;
    if (stRel >= 0 && stRel <= spanM) {
      const sx = 28 + (stRel / spanM) * (w - 28);
      ctx.fillStyle = '#10b981';
      ctx.fillRect(sx - 1.5, 2, 3, h - 4);
      ctx.font = 'bold 9px JetBrains Mono';
      ctx.fillStyle = '#34d399';
      ctx.fillText('STN', Math.min(w - 24, sx + 4), 11);
    }

    ctx.fillStyle = '#38bdf8';
    ctx.fillRect(6, h * 0.5 - 4, 18, 8);
  }

  /**
   * ========================================================================
   * UI, 3D CAB RAYCASTER & KEYBOARD EVENT BINDINGS
   * ========================================================================
   */
  setThrottle(v) {
    this.throttlePct = THREE.MathUtils.clamp(v, 0, 100);
    document.getElementById('slider-throttle').value = Math.round(this.throttlePct);

    if (this.throttlePct > 2) {
      if (this.reverser === 0) this.setReverser(1);
      if (this.brakePct > 0 && !this.atpIntervention) this.setBrake(0);
    }
  }

  setBrake(v) {
    const prev = this.brakePct;
    this.brakePct = THREE.MathUtils.clamp(v, 0, 100);
    document.getElementById('slider-brake').value = Math.round(this.brakePct);
    if (prev > 25 && this.brakePct < 5) this.audio.playPneumaticHiss();
  }

  setReverser(dir) {
    if (Math.abs(this.speedMps) > 0.5 && dir !== this.reverser && dir !== 0) return;
    this.reverser = dir;
    const map = { '1': 'btn-rev-f', '0': 'btn-rev-n', '-1': 'btn-rev-r' };
    ['btn-rev-f', 'btn-rev-n', 'btn-rev-r'].forEach((id) => {
      const el = document.getElementById(id);
      if (id === map[String(dir)]) {
        el.className = 'px-2 py-0.5 rounded text-[10px] font-mono-tech font-bold bg-emerald-500/30 text-emerald-300 border border-emerald-400/50';
      } else {
        el.className = 'px-2 py-0.5 rounded text-[10px] font-mono-tech font-bold text-slate-400 hover:text-white';
      }
    });
  }

  toggleDoors(force) {
    if (Math.abs(this.speedMps) > 0.4) return;
    this.train.doorsOpen = typeof force === 'boolean' ? force : !this.train.doorsOpen;
    document.getElementById('btn-doors').classList.toggle('active-amber', this.train.doorsOpen);
    document.getElementById('lbl-doors').textContent = this.train.doorsOpen ? 'OPENED' : 'DOORS';
    this.audio.playPneumaticHiss();

    if (this.train.doorsOpen) {
      document.getElementById('eval-boarding-status').innerHTML =
        `<i class="ph-fill ph-users-three text-cyan-400 text-base"></i><span>Penumpang sedang naik ke gerbong...</span>`;
    }
  }

  setCamera(mode) {
    this.cameraMode = mode;
    if (mode === 2) this.flyByAnchorS = -999999;
    document.querySelectorAll('.cam-btn').forEach((b) => {
      b.classList.toggle('active', parseInt(b.dataset.cam, 10) === mode);
    });
    const walkHud = document.getElementById('walk-mode-hud');
    walkHud.classList.toggle('hidden', mode !== 5);
    walkHud.classList.toggle('flex', mode === 5);
  }

  handle3DCabClick(clientX, clientY) {
    if (this.cameraMode !== 0) return false;
    _ndcMouse.set(
      (clientX / window.innerWidth) * 2 - 1,
      -(clientY / window.innerHeight) * 2 + 1
    );
    this.raycaster.setFromCamera(_ndcMouse, this.camera);
    const hits = this.raycaster.intersectObjects(this.train.interactiveCabMeshes, true);
    if (hits.length > 0) {
      const action = hits[0].object.userData.cabAction;
      if (action === 'THROTTLE') {
        this.setThrottle((this.throttlePct + 25) % 125);
        return true;
      } else if (action === 'BRAKE') {
        this.setBrake(this.brakePct > 10 ? 0 : 45);
        return true;
      } else if (action === 'HORN') {
        this.audio.init();
        this.audio.setHorn(true);
        setTimeout(() => this.audio.setHorn(false), 650);
        return true;
      } else if (action === 'SIFA') {
        this.resetSifa();
        this.audio.playChime('DEPART');
        return true;
      }
    }
    return false;
  }

  bindControls() {
    const unlock = () => this.audio.init();
    window.addEventListener('pointerdown', unlock, { once: true });
    window.addEventListener('keydown', unlock, { once: true });

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.hybridRenderer.resize(window.innerWidth, window.innerHeight);
    });

    this.canvas.addEventListener('pointerdown', (e) => {
      if (this.handle3DCabClick(e.clientX, e.clientY)) return;
      this.dragging = true;
      this.lastPtrX = e.clientX;
      this.lastPtrY = e.clientY;
    });

    window.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastPtrX;
      const dy = e.clientY - this.lastPtrY;
      this.lastPtrX = e.clientX;
      this.lastPtrY = e.clientY;

      if (this.cameraMode === 0 || this.cameraMode === 4 || this.cameraMode === 5) {
        this.cabYaw = THREE.MathUtils.clamp(this.cabYaw - dx * 0.004, -1.25, 1.25);
        this.cabPitch = THREE.MathUtils.clamp(this.cabPitch - dy * 0.004, -0.55, 0.45);
      } else if (this.cameraMode === 1) {
        this.orbitYaw -= dx * 0.006;
        this.orbitPitch = THREE.MathUtils.clamp(this.orbitPitch + dy * 0.006, 0.05, 1.35);
      }
    });

    window.addEventListener('pointerup', () => { this.dragging = false; });

    this.canvas.addEventListener('wheel', (e) => {
      if (this.cameraMode === 1) {
        this.orbitDist = THREE.MathUtils.clamp(this.orbitDist + e.deltaY * 0.03, 9, 90);
      }
    }, { passive: true });

    document.getElementById('slider-throttle').addEventListener('input', (e) => { this.setThrottle(parseFloat(e.target.value)); this.resetSifa(); });
    document.getElementById('btn-thr-inc').addEventListener('click', () => { this.setThrottle(this.throttlePct + 12.5); this.resetSifa(); });
    document.getElementById('btn-thr-dec').addEventListener('click', () => { this.setThrottle(this.throttlePct - 12.5); this.resetSifa(); });

    document.getElementById('slider-brake').addEventListener('input', (e) => { this.setBrake(parseFloat(e.target.value)); this.resetSifa(); });
    document.getElementById('btn-brk-inc').addEventListener('click', () => { this.setBrake(this.brakePct + 15); this.resetSifa(); });
    document.getElementById('btn-brk-dec').addEventListener('click', () => { this.setBrake(this.brakePct - 15); this.resetSifa(); });

    document.getElementById('btn-ebrake').addEventListener('click', () => {
      this.setThrottle(0);
      this.setBrake(100);
      this.audio.playPneumaticHiss();
    });

    document.getElementById('btn-rev-f').addEventListener('click', () => this.setReverser(1));
    document.getElementById('btn-rev-n').addEventListener('click', () => this.setReverser(0));
    document.getElementById('btn-rev-r').addEventListener('click', () => this.setReverser(-1));

    const hornBtn = document.getElementById('btn-horn');
    hornBtn.addEventListener('pointerdown', () => { this.audio.init(); this.audio.setHorn(true); this.resetSifa(); });
    hornBtn.addEventListener('pointerup', () => this.audio.setHorn(false));
    hornBtn.addEventListener('pointerleave', () => this.audio.setHorn(false));

    document.getElementById('btn-doors').addEventListener('click', () => this.toggleDoors());
    document.getElementById('btn-lights').addEventListener('click', () => {
      this.train.lightsActive = !this.train.lightsActive;
      document.getElementById('btn-lights').classList.toggle('active', this.train.lightsActive);
    });
    document.getElementById('btn-sander').addEventListener('click', () => {
      this.sanderActive = !this.sanderActive;
      document.getElementById('btn-sander').classList.toggle('active-amber', this.sanderActive);
    });
    document.getElementById('btn-wipers').addEventListener('click', () => {
      this.train.wipersActive = !this.train.wipersActive;
      document.getElementById('btn-wipers').classList.toggle('active', this.train.wipersActive);
    });
    document.getElementById('btn-panto').addEventListener('click', () => {
      this.train.pantoRaised = !this.train.pantoRaised;
      document.getElementById('btn-panto').classList.toggle('active', this.train.pantoRaised);
      this.audio.playPneumaticHiss();
    });
    document.getElementById('btn-sifa').addEventListener('click', () => {
      this.audio.init();
      this.resetSifa();
      this.audio.playChime('DEPART');
    });
    document.getElementById('btn-autopilot').addEventListener('click', () => {
      this.autopilot = !this.autopilot;
      document.getElementById('btn-autopilot').classList.toggle('active-emerald', this.autopilot);
      document.getElementById('lbl-autopilot').textContent = this.autopilot ? 'AI ON' : 'AI OFF';
    });

    const spawnAiBtn = document.getElementById('btn-spawn-oncoming');
    if (spawnAiBtn) {
      spawnAiBtn.addEventListener('click', () => {
        this.train.spawnNextOncomingTrain(this.distanceS, 520);
      });
    }

    // Walk mode buttons
    document.getElementById('btn-walk-fwd').addEventListener('click', () => { this.walkOffset.z -= 2.2; });
    document.getElementById('btn-walk-bwd').addEventListener('click', () => { this.walkOffset.z += 2.2; });
    document.getElementById('btn-walk-left').addEventListener('click', () => { this.walkOffset.x -= 1.2; });
    document.getElementById('btn-walk-right').addEventListener('click', () => { this.walkOffset.x += 1.2; });

    document.querySelectorAll('.cam-btn').forEach((b) => {
      b.addEventListener('click', () => this.setCamera(parseInt(b.dataset.cam, 10)));
    });

    const toggleHud = () => {
      this.hudCollapsed = !this.hudCollapsed;
      const dock = document.getElementById('bottom-control-dock');
      const camBar = document.getElementById('right-cam-bar');
      const btnHud = document.getElementById('btn-toggle-hud');
      const icon = document.getElementById('icon-toggle-hud');
      if (dock) {
        dock.classList.toggle('translate-y-44', this.hudCollapsed);
        dock.classList.toggle('opacity-0', this.hudCollapsed);
        dock.classList.toggle('pointer-events-none', this.hudCollapsed);
      }
      if (camBar) {
        camBar.classList.toggle('translate-x-20', this.hudCollapsed);
        camBar.classList.toggle('opacity-0', this.hudCollapsed);
      }
      if (btnHud) btnHud.classList.toggle('active', !this.hudCollapsed);
      if (icon) {
        icon.className = this.hudCollapsed
          ? 'ph-bold ph-eye-slash text-base text-amber-300'
          : 'ph-bold ph-eye text-base text-cyan-300';
      }
    };
    document.getElementById('btn-toggle-hud').addEventListener('click', toggleHud);

    document.getElementById('btn-toggle-audio').addEventListener('click', () => {
      this.audio.init();
      this.audio.enabled = !this.audio.enabled;
      if (this.audio.masterGain) this.audio.masterGain.gain.value = this.audio.enabled ? 0.72 : 0;
      document.getElementById('btn-toggle-audio').classList.toggle('active', this.audio.enabled);
      document.getElementById('icon-audio').className = this.audio.enabled
        ? 'ph-fill ph-speaker-high text-base'
        : 'ph-fill ph-speaker-slash text-base';
    });

    const mHangar = document.getElementById('modal-hangar');
    const mManual = document.getElementById('modal-manual');
    document.getElementById('btn-open-hangar').addEventListener('click', () => { mHangar.classList.remove('hidden'); mHangar.classList.add('flex'); });
    document.getElementById('btn-close-hangar').addEventListener('click', () => { mHangar.classList.add('hidden'); mHangar.classList.remove('flex'); });
    document.getElementById('btn-apply-hangar').addEventListener('click', () => { mHangar.classList.add('hidden'); mHangar.classList.remove('flex'); });

    document.getElementById('btn-open-manual').addEventListener('click', () => { mManual.classList.remove('hidden'); mManual.classList.add('flex'); });
    document.getElementById('btn-close-manual').addEventListener('click', () => { mManual.classList.add('hidden'); mManual.classList.remove('flex'); });
    document.getElementById('btn-ok-manual').addEventListener('click', () => { mManual.classList.add('hidden'); mManual.classList.remove('flex'); });

    let chosenLivery = this.train.liveryKey;
    document.querySelectorAll('.livery-card').forEach((btn) => {
      btn.addEventListener('click', () => {
        chosenLivery = btn.dataset.livery;
        document.querySelectorAll('.livery-card').forEach(b => b.classList.toggle('active', b === btn));
        this.reconfigureConsist(chosenLivery, parseInt(document.getElementById('slider-consist-len').value, 10));
      });
    });

    document.getElementById('slider-consist-len').addEventListener('input', (e) => {
      this.reconfigureConsist(chosenLivery, parseInt(e.target.value, 10));
    });

    document.querySelectorAll('.weather-card').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.weather-card').forEach(b => b.classList.toggle('active', b === btn));
        this.setWeatherPreset(btn.dataset.weather);
      });
    });

    // Instant Biome Teleport + Immediate Floating-Origin Rebase + Synchronous Chunk Warmup
    document.querySelectorAll('.biome-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.biome, 10);
        this.distanceS = idx * 4400 + 80;
        this.lastJointIdx = Math.floor(this.distanceS / GAME_PREMISE_CARTRIDGE.railway.jointInterval);

        this.floatingOrigin.forceRebase(this.distanceS);
        EndlessWorldMathematics.setOriginS(this.floatingOrigin.originS);
        this.world.applyOriginRebase(this.floatingOrigin.originS);
        this.world.warmupInitialChunks(this.distanceS, this.floatingOrigin.originS);

        this.train.spawnNextOncomingTrain(this.distanceS, 680);
        for (let i = 0; i < this.train.highwayVehicles.length; i++) {
          this.train.highwayVehicles[i].sCoord = this.distanceS - 120 + i * 38;
        }
        this.flyByAnchorS = -999999;
        this.prevCamPos.setScalar(0);
      });
    });

    // Next-Gen Graphics Quality Governor Selector (AUTO / ULTRA / HIGH / BALANCED / ECO)
    document.querySelectorAll('.quality-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const qMode = btn.dataset.quality;
        document.querySelectorAll('.quality-btn').forEach(b => b.classList.toggle('active', b === btn));
        this.governor.setMode(qMode);
        this.hybridRenderer.applyQualityTier(this.governor.activeTier);
        this.updateEngineTelemetryUI(this.lastMeasuredFps || 60);
      });
    });

    window.addEventListener('keydown', (e) => {
      this.keys[e.code] = true;
      if (e.code === 'Tab') { e.preventDefault(); document.getElementById('btn-toggle-hud').click(); }
      if (e.code === 'Space') { e.preventDefault(); this.audio.setHorn(true); this.resetSifa(); }
      if (e.code === 'KeyO') this.toggleDoors();
      if (e.code === 'KeyH') document.getElementById('btn-lights').click();
      if (e.code === 'KeyV') document.getElementById('btn-wipers').click();
      if (e.code === 'KeyX') document.getElementById('btn-sander').click();
      if (e.code === 'KeyU') document.getElementById('btn-panto').click();
      if (e.code === 'KeyQ' || e.code === 'KeyB') document.getElementById('btn-sifa').click();
      if (e.code === 'KeyP') document.getElementById('btn-autopilot').click();
      if (e.code === 'Backspace') document.getElementById('btn-ebrake').click();
      if (e.code === 'Digit1') this.setCamera(0);
      if (e.code === 'Digit2') this.setCamera(1);
      if (e.code === 'Digit3') this.setCamera(2);
      if (e.code === 'Digit4') this.setCamera(3);
      if (e.code === 'Digit5') this.setCamera(4);
      if (e.code === 'Digit6' || e.code === 'KeyF') this.setCamera(5);
      if (e.code === 'Digit7') this.setCamera(6);
    });

    window.addEventListener('keyup', (e) => {
      this.keys[e.code] = false;
      if (e.code === 'Space') this.audio.setHorn(false);
    });
  }

  reconfigureConsist(liveryKey, numCars) {
    const cfg = GAME_PREMISE_CARTRIDGE.liveries[liveryKey];
    const tons = cfg.locoMassTons + numCars * cfg.carMassTons;
    document.getElementById('lbl-consist-len').textContent = `${numCars} Gerbong (${tons} Ton)`;
    document.getElementById('hud-livery-badge').textContent = cfg.name;
    document.getElementById('hud-mass').textContent = `${tons}t`;

    const pantoBtn = document.getElementById('btn-panto');
    if (pantoBtn) {
      const isElectric = (cfg.tractionType === 'ELECTRIC');
      this.train.pantoRaised = isElectric;
      pantoBtn.classList.toggle('active', isElectric);
      pantoBtn.classList.toggle('opacity-40', !isElectric);
      pantoBtn.querySelector('span').textContent = isElectric ? 'PANTO' : 'DIESEL';
    }

    this.train.rebuildPlayerConsist(liveryKey, numCars);
  }

  animate() {
    const now = performance.now();
    const rawDt = (now - this.lastTime) / 1000;
    const dt = Math.min(0.05, Math.max(0.001, rawDt));
    this.lastTime = now;
    this.clockMins += dt * 0.25;

    // 1. Hardware Telemetry Governor & True Uncapped FPS Counter (60Hz / 120Hz / 144Hz+)
    this.governor.sampleFrame(rawDt);
    this.fpsFrames++;
    if (now - this.fpsTimer > 500) {
      const fps = Math.round((this.fpsFrames * 1000) / (now - this.fpsTimer));
      this.lastMeasuredFps = fps;
      this.updateEngineTelemetryUI(fps);
      this.fpsFrames = 0;
      this.fpsTimer = now;
    }

    // 2. Sub-Stepped Longitudinal Physics
    this.stepPhysics(dt);

    // 3. IEEE-754 64-bit Floating-Origin Rebasing (Snapped cleanly on 100m chunk grid)
    this.floatingOrigin.rebaseIfNeeded(this.distanceS, (deltaZ, newOriginS) => {
      EndlessWorldMathematics.setOriginS(newOriginS);
      this.world.applyOriginRebase(newOriginS);
      this.train.applyOriginRebase(deltaZ);
      this.camera.position.z += deltaZ;
      this.prevCamPos.z += deltaZ;
      this.flyByPos.z += deltaZ;
    });

    // 4. Stream Double-Track World Chunks via Background Web Worker + Animate Station NPCs & Signals
    this.world.update(
      this.distanceS,
      this.train.doorsOpen,
      dt,
      this.governor.activeTier,
      this.floatingOrigin.originS,
      this.train.aiTrainS
    );

    // 5. Update Multi-Bogie Consist + Exhaust/Sparks + Oncoming AI Train + Highway Traffic
    this.train.updateKinematics(
      this.distanceS,
      this.speedMps,
      this.throttlePct,
      this.brakePct,
      this.wheelSlipping,
      dt,
      this.audio
    );

    // 6. Update Camera, Texel-Snapped Sun Shadows & Multi-Source 3D Doppler
    const dopplerData = this.updateCameraSystem(dt);

    // 7. Update Multi-Bus WebAudio Synthesizer
    const f = EndlessWorldMathematics.getTrackFrame(
      this.distanceS,
      0,
      this.floatingOrigin.originS,
      _physFrame
    );
    const cfgActive = GAME_PREMISE_CARTRIDGE.liveries[this.train.liveryKey];
    const resonant = f.inTunnel || EndlessWorldMathematics.isViaductZone(this.distanceS);
    const aiApproaching = this.train.aiTrainS > (this.distanceS - 8.0);
    this.audio.update(
      this.speedMps,
      this.throttlePct,
      f.curvature,
      dopplerData.playerDoppler,
      resonant,
      dopplerData.aiDoppler,
      dopplerData.aiDist,
      dopplerData.crossingDist,
      dopplerData.crossingDoppler,
      cfgActive ? cfgActive.tractionType : 'ELECTRIC',
      this.wheelSlipping,
      this.brakePct,
      aiApproaching
    );

    // 8. Update Telemetry HUD & 3D MFD Texture (Throttled every 3 frames)
    this.hudTick = (this.hudTick + 1) % 3;
    if (this.hudTick === 0) {
      this.updateHUD();
    }

    // 9. Render Next-Gen WebGPU / WebGL2 + TSL HDR Bloom Pipeline
    this.hybridRenderer.render(this.scene, this.camera);
  }
}

const sim = new ApexRailSimulatorKernel();
window.apexRailSim = sim;
sim.boot();
