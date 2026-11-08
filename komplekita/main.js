import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { ProceduralTextureFactory } from './core/ProceduralTextureFactory.js';
import { NexusBatchOptimizer } from './core/BatchRenderer.js';
import { VerificationGateRunner } from './core/VerificationGateRunner.js';
import { CameraDirector } from './core/CameraDirector.js';
import { WeatherOpticsEngine } from './core/WeatherOpticsEngine.js';

import { UI_DPPD_MASTER_PLAN, BUNDARAN_CAMERA_PRESETS } from './world-gis/MasterPlanConfig.js';
import { BundaranSiteBuilder } from './world-gis/BundaranSiteBuilder.js';

import { PerpustakaanLamaBuilder } from './library/perpustakaan-lama/index.js';
import { GedungDPPDBuilder } from './library/gedung-dppd/index.js';
import { GedungSTPUIBuilder } from './library/stp-ui/index.js';
import { GedungILRCBuilder } from './library/ilrc-drpm/index.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./main.js
 *
 * Unified Class-A Orchestrator for Kompleks Bundaran Gedung DPPD (eks-KSDP),
 * ITTC Perpustakaan Lama, Science Techno Park (STP-UI), & ILRC/DRPM Universitas Indonesia Depok.
 */
export class BundaranKompleksAsset {
  constructor(materials) {
    this.materials = materials;
    this.meta = {
      id: UI_DPPD_MASTER_PLAN.id,
      name: UI_DPPD_MASTER_PLAN.name,
      dimensions: {
        width: UI_DPPD_MASTER_PLAN.envelope.width,
        length: UI_DPPD_MASTER_PLAN.envelope.length,
        height: UI_DPPD_MASTER_PLAN.envelope.height
      },
      foundationSkirtDepth: UI_DPPD_MASTER_PLAN.envelope.foundationSkirtDepth,
      orthoValidationPresets: BUNDARAN_CAMERA_PRESETS
    };

    this.root = new THREE.Group();
    this.root.name = 'NexusBundaranKompleksRoot';
    this.dppdRig = null;

    this._buildAndOptimize();
  }

  _buildAndOptimize() {
    const siteGroup = BundaranSiteBuilder.build(this.materials);
    const ittcGroup = PerpustakaanLamaBuilder.build(this.materials);
    const dppdResult = GedungDPPDBuilder.build(this.materials);
    const stpGroup = GedungSTPUIBuilder.build(this.materials);
    const ilrcGroup = GedungILRCBuilder.build(this.materials);

    this.dppdRig = dppdResult.rig;

    this.root.add(siteGroup);
    this.root.add(ittcGroup);
    this.root.add(dppdResult.group);
    this.root.add(stpGroup);
    this.root.add(ilrcGroup);

    // Single-Pass Hierarchy Batching -> merges static materials across all 4 buildings + site into <= 21 Draw Calls
    NexusBatchOptimizer.optimizeHierarchy(this.root, { disposeSourceGeometries: true });
  }

  /**
   * Gate 2 SSoST Master Surface Evaluator:
   * Evaluates closed polar ring manifold u in [0,1] -> theta in [-pi, +pi], v in [0,1] -> elevation.
   */
  evaluateMasterSurface(u, v, targetPos, targetNormal) {
    const theta = (u * 2.0 - 1.0) * Math.PI;
    const r = UI_DPPD_MASTER_PLAN.ringRoad.rCenterDash;
    const y = (v - 0.5) * UI_DPPD_MASTER_PLAN.envelope.height;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    targetPos.set(r * c, y, r * s);
    targetNormal.set(c, 0, s).normalize();
  }

  /**
   * Gate 3 Zero-Gap SSoST Subdomain Seam Continuity Check (0.00m Gap):
   * Verifies that adjacent polar roof/wall subdomains share exact boundary coordinates.
   */
  verifySubdomainContinuity() {
    const ilrc = UI_DPPD_MASTER_PLAN.sectors.ilrc;
    const rOut = ilrc.rOuter;
    // Check East Wing -> Atrium -> South Wing -> South Tower angular boundary continuity
    const seam1 = Math.hypot(
      rOut * Math.cos(ilrc.thEastEnd) - rOut * Math.cos(ilrc.thAtriumStart),
      rOut * Math.sin(ilrc.thEastEnd) - rOut * Math.sin(ilrc.thAtriumStart)
    );
    const seam2 = Math.abs(ilrc.thSouthEnd - 75.0 * (Math.PI / 180));
    return Math.max(seam1, seam2);
  }

  updateKinematics(dt, state = {}) {
    if (this.dppdRig && typeof this.dppdRig.update === 'function') {
      this.dppdRig.update(dt, state);
    }
  }
}

class NexusStudioApp {
  constructor() {
    this.canvasContainer = document.getElementById('viewport-container') || document.body;
    this.sharedUniforms = ProceduralTextureFactory.createSharedShaderUniforms();

    this.stationKeys = [
      'sv_dppd_front',
      'sv_dppd_totem',
      'sv_dppd_se_wing',
      'sv_dppd_nw_gable',
      'sv_west_ittc',
      'sv_stp_sw_core',
      'sv_stp_portico',
      'sv_stp_ne_makara',
      'sv_ilrc_east_box',
      'sv_ilrc_drpm_gate',
      'sv_ilrc_atrium',
      'sv_ilrc_mound',
      'sv_ilrc_brick_tower'
    ];
    this.currentStationIndex = 0;
    this.activePresetKey = 'sv_dppd_front';

    this._initRendererAndScene();
    this._buildWorld();
    this._bindUI();
    this._startLoop();
  }

  _initRendererAndScene() {
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
      stencil: false
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.02;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.canvasContainer.appendChild(this.renderer.domElement);

    // Self-healing WebGL context guards
    this.renderer.domElement.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      console.warn('[NEXUS] WebGL context lost — awaiting automatic restoration.');
    });
    this.renderer.domElement.addEventListener('webglcontextrestored', () => {
      console.info('[NEXUS] WebGL context restored — recompiling shaders.');
      this.renderer.compile(this.scene, this.camera);
    });

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(
      58,
      Math.max(1, window.innerWidth) / Math.max(1, window.innerHeight),
      0.35,
      650.0
    );

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.065;
    this.controls.maxDistance = 310.0;
    this.controls.minDistance = 3.0;
    this.controls.maxPolarAngle = Math.PI * 0.492;

    this.cameraDirector = new CameraDirector(this.camera, this.controls, this.renderer.domElement);
    // Register soft anti-clipping collision solids for both Orbit and 6-DOF Drone flight
    this.cameraDirector.registerCollisionBoxes([
      { minX: -25.2, maxX: 19.2, minY: 0.0, maxY: 19.5, minZ: -20.8, maxZ: 23.8 }, // ITTC central square mass
      { minX: -29.8, maxX: -15.2, minY: 0.0, maxY: 5.2, minZ: -25.5, maxZ: -16.5 } // ITTC NW Annex
    ]);

    window.addEventListener('resize', () => {
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.cameraDirector.applyNonOccludingViewportOffset(this.cameraDirector.drawerCollapsed);
    });
  }

  _buildWorld() {
    // 1. Procedural 14:00 WIB Afternoon Environment Reflection Map (PMREM)
    const envMap = ProceduralTextureFactory.createAfternoonEnvironmentMap(this.renderer);
    this.scene.environment = envMap;

    // 2. Shared PBR Material Bank + Unified 1024x1024 Signage Atlas
    this.materials = ProceduralTextureFactory.createComplexMaterialBank(this.sharedUniforms);

    // 3. Weather & 14:00 WIB Daylight Optics Engine
    this.weatherEngine = new WeatherOpticsEngine(this.scene, this.renderer, this.materials);

    // 4. Build & Batch Complete 4-Building Roundabout Asset
    this.asset = new BundaranKompleksAsset(this.materials);
    this.scene.add(this.asset.root);

    // 5. Initial Default Camera View: Proportional Human-Scale FPV Runner Mode (1.68m person -> 1.57m eye level)
    //    Jogging around the Bundaran complex while looking around at the buildings (with Drawer Collapsed — Hukum Emas 20)
    this.cameraDirector.setMode('runner');
    this.cameraDirector.applyNonOccludingViewportOffset(true);

    // 6. Pre-warm GPU Shaders & Execute Automated 4-Gate Verification after first render pass
    this.renderer.compile(this.scene, this.camera);
    this.renderer.render(this.scene, this.camera);

    this.gateReport = VerificationGateRunner.runAllGates(this.asset, this.renderer, {
      maxDrawCalls: 25
    });

    // Expose forensic inspection handle on window for automated headless testing
    window.__NEXUS_APP__ = this;
    window.__NEXUS_GATE_REPORT__ = this.gateReport;

    this._renderGateReportUI();
  }

  _renderGateReportUI() {
    const r = this.gateReport;
    if (!r) return;

    const pillBadge = document.getElementById('pill-gate-badge');
    if (pillBadge) {
      pillBadge.textContent = r.summaryBadge;
      pillBadge.className = `gate-badge ${r.allPassed ? 'pass' : 'warn'}`;
    }

    const g1 = r.gates.gate1;
    const g2 = r.gates.gate2;
    const g3 = r.gates.gate3;
    const g4 = r.gates.gate4;

    const setGateRow = (id, passed, detail) => {
      const el = document.getElementById(id);
      if (!el) return;
      const statusEl = el.querySelector('.gate-status');
      const detailEl = el.querySelector('.gate-detail');
      if (statusEl) {
        statusEl.textContent = passed ? 'PASS' : 'FAIL';
        statusEl.className = `gate-status ${passed ? 'pass' : 'fail'}`;
      }
      if (detailEl) detailEl.textContent = detail;
    };

    setGateRow(
      'gate-row-1',
      g1.passed,
      `${g1.measuredSize.width}×${g1.measuredSize.length}×${g1.measuredSize.height}m | Skirt -${g1.measuredSkirtDepth}m | ${g1.nanCount} NaN`
    );
    setGateRow(
      'gate-row-2',
      g2.passed,
      `Outward Normals Valid | SSoST Gap ${g2.closureGapMeters.toFixed(4)}m`
    );
    setGateRow(
      'gate-row-3',
      g3.passed,
      `Seam Gap ${g3.maxSeamGapMeters.toFixed(4)}m | Kinematic Rig Safe`
    );
    setGateRow(
      'gate-row-4',
      g4.passed,
      `${g4.liveRenderCalls || g4.meshDrawNodes}/25 Draw Calls | ${g1.triangleCount.toLocaleString('en-US')} Tris`
    );
  }

  _bindUI() {
    const drawer = document.getElementById('inspector-drawer');
    const launcherBtn = document.getElementById('launcher-pill');
    const closeDrawerBtn = document.getElementById('drawer-close-btn');

    const setDrawerCollapsed = (collapsed) => {
      if (!drawer) return;
      drawer.classList.toggle('collapsed', collapsed);
      if (launcherBtn) {
        launcherBtn.setAttribute('aria-expanded', String(!collapsed));
      }
      this.cameraDirector.applyNonOccludingViewportOffset(collapsed);
    };

    if (launcherBtn) {
      launcherBtn.addEventListener('click', () => {
        const nextCollapsed = !drawer.classList.contains('collapsed');
        setDrawerCollapsed(nextCollapsed);
      });
    }
    if (closeDrawerBtn) {
      closeDrawerBtn.addEventListener('click', () => setDrawerCollapsed(true));
    }

    // Preset selector helper
    const activatePreset = (presetKey, duration = 0.72) => {
      const preset = BUNDARAN_CAMERA_PRESETS[presetKey];
      if (!preset) return;
      this.activePresetKey = presetKey;
      this.cameraDirector.applyPreset(preset, duration);

      // Sync UI buttons & station dropdown
      document.querySelectorAll('[data-preset]').forEach((btn) => {
        btn.classList.toggle('active', btn.getAttribute('data-preset') === presetKey);
      });

      const svSelect = document.getElementById('sv-station-select');
      if (svSelect && this.stationKeys.includes(presetKey)) {
        svSelect.value = presetKey;
        this.currentStationIndex = this.stationKeys.indexOf(presetKey);
      }

      const activeViewLabel = document.getElementById('active-view-label');
      if (activeViewLabel) {
        activeViewLabel.textContent = preset.label;
      }

      // Sync Orbit/Drone toggle visual
      document.querySelectorAll('[data-cam-mode]').forEach((btn) => {
        btn.classList.toggle('active', btn.getAttribute('data-cam-mode') === this.cameraDirector.mode);
      });
    };

    this.activatePreset = activatePreset;

    document.querySelectorAll('[data-preset]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const key = btn.getAttribute('data-preset');
        activatePreset(key, 0.72);
      });
    });

    const svSelect = document.getElementById('sv-station-select');
    if (svSelect) {
      svSelect.addEventListener('change', (e) => {
        activatePreset(e.target.value, 0.72);
      });
    }

    const prevBtn = document.getElementById('sv-prev-btn');
    const nextBtn = document.getElementById('sv-next-btn');
    if (prevBtn) {
      prevBtn.addEventListener('click', () => {
        this.currentStationIndex =
          (this.currentStationIndex - 1 + this.stationKeys.length) % this.stationKeys.length;
        activatePreset(this.stationKeys[this.currentStationIndex], 0.68);
      });
    }
    if (nextBtn) {
      nextBtn.addEventListener('click', () => {
        this.currentStationIndex = (this.currentStationIndex + 1) % this.stationKeys.length;
        activatePreset(this.stationKeys[this.currentStationIndex], 0.68);
      });
    }

    // Inspection Shader & Lighting Mode buttons
    document.querySelectorAll('[data-inspect-mode]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = btn.getAttribute('data-inspect-mode');
        document.querySelectorAll('[data-inspect-mode]').forEach((b) => b.classList.toggle('active', b === btn));

        if (mode === 'realistic_1400') {
          this.sharedUniforms.uInspectMode.value = 0;
          this.weatherEngine.setLightingMode('afternoon_1400');
        } else if (mode === 'clay') {
          this.sharedUniforms.uInspectMode.value = 1;
          this.weatherEngine.setLightingMode('afternoon_1400');
        } else if (mode === 'zebra') {
          this.sharedUniforms.uInspectMode.value = 2;
          this.weatherEngine.setLightingMode('afternoon_1400');
        } else if (mode === 'night_gi') {
          this.sharedUniforms.uInspectMode.value = 0;
          this.weatherEngine.setLightingMode('night_gi');
        }
      });
    });

      // Camera Mode (FPV Runner vs Orbit vs 6-DOF Drone)
    document.querySelectorAll('[data-cam-mode]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = btn.getAttribute('data-cam-mode');
        this.cameraDirector.setMode(mode);
        document.querySelectorAll('[data-cam-mode]').forEach((b) => b.classList.toggle('active', b === btn));
        if (mode === 'runner') {
          document.querySelectorAll('[data-preset]').forEach((b) => b.classList.remove('active'));
        }
      });
    });
  }

  _startLoop() {
    let lastTime = performance.now();
    let frameCount = 0;
    let fpsAccum = 0;

    const fpsEl = document.getElementById('hud-fps');
    const dcEl = document.getElementById('hud-dc');
    const trisEl = document.getElementById('hud-tris');
    const viewLabelEl = document.getElementById('active-view-label');

    const animate = (now) => {
      requestAnimationFrame(animate);
      const dt = Math.min(0.05, Math.max(0.0005, (now - lastTime) * 0.001));
      lastTime = now;
      const timeSec = now * 0.001;

      this.cameraDirector.update(dt);
      this.asset.updateKinematics(dt, { timeSec, windSpeed: 3.6 });
      this.weatherEngine.update(timeSec);

      this.renderer.render(this.scene, this.camera);

      frameCount++;
      fpsAccum += dt;
      if (fpsAccum >= 0.4) {
        const fps = Math.min(120, Math.round(frameCount / fpsAccum));
        const calls = this.renderer.info.render.calls;
        const tris = this.renderer.info.render.triangles;
        if (fpsEl) fpsEl.textContent = `${fps} FPS`;
        if (dcEl) dcEl.textContent = `${calls} DC`;
        if (trisEl) trisEl.textContent = `${(tris / 1000).toFixed(1)}k Tris`;
        if (viewLabelEl && this.cameraDirector.mode === 'runner') {
          viewLabelEl.textContent = this.cameraDirector.runner.activeLandmarkLabel;
        }
        frameCount = 0;
        fpsAccum = 0;
      }
    };

    requestAnimationFrame(animate);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  new NexusStudioApp();
});
