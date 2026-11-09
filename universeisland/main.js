import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.9 — Live Kampus UI Depok & 20–45s Physical Transitions)
 * ./main.js — Primary Simulator Orchestrator (Kampus UI Depok: Pulau Rektorat 360°, Danau Kenanga, & Helipad VIP)
 *
 * Strictly adheres to:
 * - Default Real-Time Clock (Asia/Jakarta WIB) & Live Weather of Kampus UI, Kota Depok
 * - Mandatory 20–45s Continuous Physical Weather & Diurnal Time Transitions + Button Progress Visualization
 * - Frame-Yielded 9-Stage Progressive Preloader + GPU Shader Pre-Warming (0ms Frame-1 Stutter)
 * - Strict Relative Paths (./), <= 25 Total Scene Draw Calls, & Default-Collapsed Non-Occluding UI
 */

const WEATHER_LABELS = {
  CLEAR: 'Cerah',
  PARTLY_CLOUDY: 'Cerah Berawan',
  OVERCAST: 'Mendung',
  WINDY: 'Angin Kencang',
  FOGGY: 'Kabut Embun',
  POST_RAIN: 'Seusai Hujan',
  MIST: 'Gerimis',
  SHOWER: 'Hujan Ringan',
  RAIN: 'Hujan Deras',
  STORM: 'Badai Petir'
};

const TOD_LABELS = {
  DAWN: 'Fajar',
  MORNING: 'Pagi',
  DAY: 'Siang',
  AFTERNOON: 'Sore',
  DUSK: 'Senja',
  NIGHT: 'Malam'
};

if (typeof window !== 'undefined') {
  const preventContextMenu = (e) => {
    e.preventDefault();
    return false;
  };
  window.addEventListener('contextmenu', preventContextMenu, { capture: true });
  if (typeof document !== 'undefined') {
    document.addEventListener('contextmenu', preventContextMenu, { capture: true });
  }
}

const TOTAL_EXPECTED_TRIS = 4655186;

function yieldToBrowserFrame() {
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    const fallbackTimer = setTimeout(finish, 34);
    requestAnimationFrame(() => {
      clearTimeout(fallbackTimer);
      setTimeout(finish, 0);
    });
  });
}

function scheduleAsyncBackgroundTask(fn, delayMs = 60) {
  if (typeof window !== 'undefined' && typeof window.requestIdleCallback === 'function') {
    setTimeout(() => {
      window.requestIdleCallback(() => fn(), { timeout: 600 });
    }, delayMs);
  } else {
    setTimeout(fn, delayMs);
  }
}

function createPreloaderController() {
  const rootEl = document.getElementById('nexus-preloader');
  const stageEl = document.getElementById('loader-stage-text');
  const pctEl = document.getElementById('loader-pct');
  const barEl = document.getElementById('loader-bar-fill');
  const triEl = document.getElementById('loader-tri-count');
  const weatherBadgeEl = document.getElementById('loader-weather-badge');
  const pips = rootEl ? Array.from(rootEl.querySelectorAll('.loader-step-pip')) : [];

  let displayedTris = 0;
  let targetTris = 0;
  let rafId = 0;

  function renderTriLabel(val) {
    if (!triEl) return;
    triEl.textContent = `${val.toLocaleString('id-ID')} / ${TOTAL_EXPECTED_TRIS.toLocaleString('id-ID')} segitiga`;
  }

  function tickTriCounter() {
    if (!triEl) return;
    const diff = targetTris - displayedTris;
    if (Math.abs(diff) <= 1200) {
      displayedTris = targetTris;
    } else {
      displayedTris += Math.ceil(diff * 0.45);
    }
    renderTriLabel(displayedTris);
    if (displayedTris !== targetTris) {
      rafId = requestAnimationFrame(tickTriCounter);
    } else {
      rafId = 0;
    }
  }

  return {
    setWeatherBadge(text) {
      if (weatherBadgeEl) weatherBadgeEl.textContent = text;
    },
    async advance(stepIdx, pct, label, trisCount) {
      if (stageEl) stageEl.textContent = label;
      if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;
      if (barEl) barEl.style.width = `${Math.max(4, Math.min(100, pct))}%`;
      pips.forEach((pip, idx) => {
        pip.classList.toggle('done', idx < stepIdx);
        pip.classList.toggle('active', idx === stepIdx);
      });
      if (typeof trisCount === 'number') {
        // Ensure displayed triangle count never lags behind completed procedural stages
        displayedTris = Math.max(displayedTris, trisCount >= TOTAL_EXPECTED_TRIS ? TOTAL_EXPECTED_TRIS : Math.round(trisCount * 0.96));
        targetTris = trisCount;
        renderTriLabel(displayedTris);
        if (displayedTris !== targetTris && !rafId) {
          rafId = requestAnimationFrame(tickTriCounter);
        }
      }
      await yieldToBrowserFrame();
    },
    dismiss() {
      if (!rootEl) return;
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      displayedTris = TOTAL_EXPECTED_TRIS;
      targetTris = TOTAL_EXPECTED_TRIS;
      renderTriLabel(TOTAL_EXPECTED_TRIS);
      if (barEl) barEl.style.width = '100%';
      if (pctEl) pctEl.textContent = '100%';
      pips.forEach((pip) => {
        pip.classList.remove('active');
        pip.classList.add('done');
      });
      rootEl.classList.add('is-finishing');
      rootEl.setAttribute('aria-hidden', 'true');
      setTimeout(() => {
        rootEl.style.display = 'none';
      }, 420);
    }
  };
}

async function bootstrapNexusSimulator() {
  const _nowBoot = () => (typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now());
  const _bootOriginMs =
    typeof window !== 'undefined' &&
    window.NexusForensic &&
    typeof window.NexusForensic.getOriginPerfMs === 'function'
      ? window.NexusForensic.getOriginPerfMs()
      : 0.0;
  const _bootstrapStartRelMs = Math.max(0.0, _nowBoot() - _bootOriginMs);
  if (typeof window !== 'undefined' && window.NexusForensic) {
    window.NexusForensic.markMilestone('bootstrap_start', 'Memulai bootstrap Kepulauan UI v6.46');
  }
  const preloader = createPreloaderController();

  // Stage 0 & Stage 1 Unified Parallel Module Pipeline:
  // Launch all 17 ES6 module imports simultaneously at t=0 so network/disk + parse overlap 100%,
  // while immediately activating Depok WIB clock & Open-Meteo telemetry as soon as DepokLiveWeatherService resolves!
  const _tS01Wall0 = _nowBoot();
  const _tS01Import0 = _nowBoot();
  const _depokWeatherModPromise = import('./core/DepokLiveWeatherService.js');
  const _engineModulesPromise = Promise.all([
    import('./core/CameraDirector.js'),
    import('./core/WeatherOpticsEngine.js'),
    import('./core/AudioSynthesizer.js'),
    import('./core/ProceduralTextureFactory.js'),
    import('./core/VideotronYouTubeEngine.js'),
    import('./core/RotundaDancingFountain.js'),
    import('./core/HardwareTelemetryGovernor.js'),
    import('./core/CinematicPostFX.js'),
    import('./core/ForensicLogSystem.js'),
    import('./world-gis/FloatingOriginManager.js'),
    import('./world-gis/TerrainDEMLoader.js'),
    import('./world-gis/SplineCorridorBuilder.js'),
    import('./world-gis/AutogenBiomeSpawner.js'),
    import('./world-gis/TropicalGrassField.js'),
    import('./world-gis/DynamicTrafficAI.js'),
    import('./library/landmark/Gedung_Rektorat_UI/index.js')
  ]);

  await preloader.advance(0, 10, 'Mengaktifkan Jam WIB Depok & Sinkronisasi Cuaca Asinkron...', 0);
  const { DepokLiveWeatherService } = await _depokWeatherModPromise;
  const _tS01Cpu0 = _nowBoot();

  let latestDepokClock = DepokLiveWeatherService.getDepokWIBClock();
  let latestDepokWeather = DepokLiveWeatherService.buildOfflineDepokFallback();
  let liveWeatherSettled = false;

  preloader.setWeatherBadge(
    `${latestDepokClock.formattedHM} WIB · ${latestDepokWeather.labelId.replace(' (Estimasi Lokal)', '')} (Sinkron Asinkron...)`
  );
  const _tLiveWeatherStartRelMs = Math.max(0.0, _nowBoot() - _bootOriginMs);
  let _bootProfilerRef = null;

  // Reference weather telemetry streams asynchronously in background — NEVER blocks world loading!
  const liveWeatherPromise = DepokLiveWeatherService.fetchLiveDepokWeather({ timeoutMs: 4500 }).then(
    (data) => {
      latestDepokWeather = data;
      liveWeatherSettled = true;
      if (_bootProfilerRef) _bootProfilerRef.recordLiveWeatherSettled(data);
      preloader.setWeatherBadge(
        `${data.labelId.replace(' (Estimasi Lokal)', '')} · ${data.temperatureC}°C · RH ${data.humidityPct}%`
      );
      return data;
    }
  );
  const _tS01End = _nowBoot();
  const _s01CpuMs = Math.max(0.0, _tS01End - _tS01Cpu0);
  const _s01ImportMs = Math.max(0.0, _tS01Cpu0 - _tS01Import0);

  // Stage 1: Await Remaining Parallel Engine & World Modules (Already In-Flight Since t=0!)
  const _tS02Wall0 = _nowBoot();
  await preloader.advance(1, 20, 'Memuat Modul Mesin Prosedural NEXUS v5.0 & Shader Ultra...', 0);
  const _tS02Import0 = _nowBoot();
  const [
    { CameraDirector, applyNonOccludingViewportOffset, PhysicsKernel },
    { WeatherOpticsEngine },
    { AudioSynthesizer },
    { createSharedSurfaceUniforms },
    { VideotronYouTubeEngine },
    { RotundaDancingFountain },
    { HardwareTelemetryGovernor },
    { CinematicPostFX },
    { NexusRolling30sRecorder, NexusBootForensicProfiler, createForensicReadOnlyProbe },
    { FloatingOriginManager },
    { TerrainDEMLoader },
    { SplineCorridorBuilder },
    { AutogenBiomeSpawner },
    { TropicalGrassField },
    { DynamicTrafficAI },
    { GedungRektoratUIAsset }
  ] = await _engineModulesPromise;
  const _tS02End = _nowBoot();

  const bootProfiler = new NexusBootForensicProfiler({ originPerfMs: _bootOriginMs });
  _bootProfilerRef = bootProfiler;
  bootProfiler.deferredTasks.liveDepokWeatherFetch.startedAtMs = +_tLiveWeatherStartRelMs.toFixed(2);
  if (liveWeatherSettled) {
    bootProfiler.recordLiveWeatherSettled(latestDepokWeather);
  }
  bootProfiler.recordPreBootstrapSpan(0.1, _bootstrapStartRelMs);
  bootProfiler.stages.push({
    id: 'S01_depok_clock_and_weather_kickoff',
    category: 'MODULE_LOAD',
    label: 'Jam WIB Depok & Kickoff Open-Meteo Asinkron',
    startMs: +Math.max(0.0, _tS01Wall0 - _bootOriginMs).toFixed(2),
    endMs: +Math.max(0.0, _tS01End - _bootOriginMs).toFixed(2),
    wallMs: +Math.max(0.0, _tS01End - _tS01Wall0).toFixed(2),
    cpuMs: +_s01CpuMs.toFixed(2),
    waitOrYieldMs: +Math.max(0.0, _tS01End - _tS01Wall0 - _s01CpuMs).toFixed(2),
    addedPrograms: 0,
    addedGeometries: 0,
    addedTextures: 0,
    detail: `Import DepokLiveWeatherService ${_s01ImportMs.toFixed(1)}ms`
  });
  bootProfiler.stages.push({
    id: 'S02_engine_modules_parallel_import',
    category: 'MODULE_LOAD',
    label: 'Import Paralel 16 Modul Mesin Prosedural & Dunia',
    startMs: +Math.max(0.0, _tS02Wall0 - _bootOriginMs).toFixed(2),
    endMs: +Math.max(0.0, _tS02End - _bootOriginMs).toFixed(2),
    wallMs: +Math.max(0.0, _tS02End - _tS02Wall0).toFixed(2),
    cpuMs: null,
    waitOrYieldMs: +Math.max(0.0, _tS02End - _tS02Wall0).toFixed(2),
    addedPrograms: 0,
    addedGeometries: 0,
    addedTextures: 0,
    detail: `Promise.all 16 modul ESM (${Math.max(0.0, _tS02End - _tS02Import0).toFixed(1)}ms net/parse)`
  });

  // Stage 2: WebGL2 Human-Eye Physiological HDR Renderer, Instant RTC Spatial Anchor, & PMREM Sky IBL
  const _tokS03 = bootProfiler.beginStage(
    'S03_webgl2_context_and_rtc',
    'RENDERER_INIT',
    'Konteks WebGL2/ANGLE, Videotron CSS3D & Floating Origin RTC'
  );
  await preloader.advance(2, 30, 'Menginisialisasi WebGL2 Human-Eye Photoreceptor HDR & Kubah Langit...', 28856);
  const _tS03Cpu0 = _nowBoot();
  const container = document.getElementById('sim-canvas-wrap');
  const isMobilePlatform = HardwareTelemetryGovernor.detectMobilePlatform();
  const nativeDpr = window.devicePixelRatio || 1;
  const renderer = new THREE.WebGLRenderer({
    antialias: !isMobilePlatform || nativeDpr < 1.35,
    alpha: true,
    premultipliedAlpha: true,
    powerPreference: 'high-performance'
  });
  renderer.setClearColor(0x070b12, 1.0);
  renderer.setPixelRatio(Math.min(nativeDpr, isMobilePlatform ? 1.05 : 2.0));
  renderer.setSize(window.innerWidth, window.innerHeight);
  // Unified in-shader HumanEyeHDROptics (Naka-Rushton + Hunt + Stevens + Purkinje) handles 100% of tone-mapping
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  let webglContextLost = false;
  let hwGovernorRef = null;
  renderer.domElement.addEventListener(
    'webglcontextlost',
    (e) => {
      e.preventDefault();
      webglContextLost = true;
      hwGovernorRef?.logEvent('WEBGL', 'Context lost');
      if (typeof window !== 'undefined' && window.NexusForensic) {
        window.NexusForensic.recordWebGLContextEvent(true);
      }
    },
    false
  );
  renderer.domElement.addEventListener(
    'webglcontextrestored',
    () => {
      webglContextLost = false;
      renderer.shadowMap.needsUpdate = true;
      hwGovernorRef?.logEvent('WEBGL', 'Context restored');
      if (typeof window !== 'undefined' && window.NexusForensic) {
        window.NexusForensic.recordWebGLContextEvent(false);
      }
    },
    false
  );
  container.appendChild(renderer.domElement);

  let syncVideotronUIHook = null;
  const videotron = new VideotronYouTubeEngine({
    initialInput: VideotronYouTubeEngine.DEFAULT_EMBED_HTML,
    onStateChange: (state, errMsg) => {
      if (syncVideotronUIHook) syncVideotronUIHook(state, errMsg);
    }
  });
  videotron.mount(container);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0xa8c6df, 0.0);

  // Boot immediately with VFS / canonical Kampus UI Depok RTC anchor; reconcile JSON reference asynchronously!
  const vfsCampusContext =
    typeof window !== 'undefined' && window.__NEXUS_JSON_VFS__
      ? window.__NEXUS_JSON_VFS__['data-routes/kampus-ui-context.json']
      : null;
  const rtcManager = new FloatingOriginManager(vfsCampusContext || FloatingOriginManager.DEFAULT_RTC_CENTER);
  const worldRoot = new THREE.Group();
  worldRoot.name = 'KampusUIDepok_RTCWorldRoot';
  rtcManager.anchorGroup(worldRoot);
  scene.add(worldRoot);

  FloatingOriginManager.loadCampusContext('./data-routes/kampus-ui-context.json').then((asyncCtx) => {
    if (asyncCtx && asyncCtx.rtcCenterWorld) {
      rtcManager.setCenter(asyncCtx.rtcCenterWorld);
      if (Number.isFinite(asyncCtx.lat)) rtcManager.lat = Number(asyncCtx.lat);
      if (Number.isFinite(asyncCtx.lon)) rtcManager.lon = Number(asyncCtx.lon);
      worldRoot.userData.rtcCenterWorld = rtcManager.getCenterWorld();
    }
  });

  const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.18, 5600);
  const presets = GedungRektoratUIAsset.meta.orthoValidationPresets;
  camera.position.set(presets.oblique45.pos.x, presets.oblique45.pos.y, presets.oblique45.pos.z);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  // Allow tilting upward into the night sky (Moon, Milky Way, Constellations); CameraDirector soft floor prevents underground clip!
  controls.maxPolarAngle = Math.PI * 0.86;
  controls.minDistance = 0.55;
  controls.maxDistance = 2150;
  controls.target.set(presets.oblique45.target.x, presets.oblique45.target.y, presets.oblique45.target.z);
  controls.update();
  bootProfiler.attachRenderer(renderer);
  bootProfiler.endStage(_tokS03, {
    cpuMs: Math.max(0.0, _nowBoot() - _tS03Cpu0),
    detail: `Canvas ${renderer.domElement.width}×${renderer.domElement.height} · DPR ${renderer.getPixelRatio().toFixed(2)}×`
  });

  const _tokS04 = bootProfiler.beginStage(
    'S04_shared_textures_governor',
    'RENDERER_INIT',
    'Tekstur Prosedural Bersama, Hardware Governor & Blackbox 30d'
  );
  const sharedUniforms = createSharedSurfaceUniforms();
  const hwGovernor = new HardwareTelemetryGovernor(renderer, sharedUniforms);
  hwGovernorRef = hwGovernor;
  const forensicRecorder = new NexusRolling30sRecorder();
  bootProfiler.endStage(_tokS04, {
    detail: `${hwGovernor.gpuCleanName || 'WebGL2'} · ${hwGovernor.activeTierId || 'TIER_S'}`
  });
  if (typeof window !== 'undefined' && window.NexusForensic) {
    window.NexusForensic.markMilestone(
      'webgl2_renderer_ready',
      `${hwGovernor.gpuCleanName || 'WebGL2'} · ${hwGovernor.activeTierId || 'TIER_S'}`
    );
  }

  const _tokS05 = bootProfiler.beginStage(
    'S05_weather_optics_and_water',
    'WORLD_BUILD',
    'Mesin Optik Atmosfer/PMREM, Material Air & Shell AudioSynthesizer'
  );
  const weather = new WeatherOpticsEngine(scene, {
    shadowMapSize: hwGovernor.getActiveTierConfig().shadowMapSize
  });
  if (sharedUniforms.uGenshinMode) {
    weather.uniforms.uGenshinMode = sharedUniforms.uGenshinMode;
    if (weather.skyMesh && weather.skyMesh.material && weather.skyMesh.material.uniforms) {
      weather.skyMesh.material.uniforms.uGenshinMode = sharedUniforms.uGenshinMode;
    }
  }
  weather.attachRenderer(renderer);
  const waterMaterial = weather.createDanauKenangaWaterMaterial();
  const audioSynth = new AudioSynthesizer();
  audioSynth.setHardwareEfficiencyHint(
    hwGovernor.isMobileDevice ||
      hwGovernor.activeTierId === 'TIER_C' ||
      /IGPU|INTEGRATED|MOBILE|SOFTWARE/i.test(String(hwGovernor.gpuClass || ''))
  );
  forensicRecorder.attachAudioForensics(audioSynth, hwGovernor);
  bootProfiler.endStage(_tokS05, {
    detail: `ShadowMap ${hwGovernor.getActiveTierConfig().shadowMapSize}px · PMREM Sky IBL`
  });

  // Stage 3: Build Gedung Rektorat UI Class-A Monument (8-Sided Tower + 3-Tier Tajug Pagoda)
  const _tokS06 = bootProfiler.beginStage(
    'S06_gedung_rektorat_4gates',
    'WORLD_BUILD',
    'Gedung Rektorat UI Kelas-A (Menara Segi-8, Tajug 3-Tumpang & Audit 4-Gate)'
  );
  await preloader.advance(3, 44, 'Menyusun Menara Segi-8 & Atap Tajug 3-Tumpang Rektorat UI...', 377118);
  const _tS06Cpu0 = _nowBoot();
  const rektoratAsset = GedungRektoratUIAsset.build({ sharedUniforms });
  const _s06CpuMs = Math.max(0.0, _nowBoot() - _tS06Cpu0);
  const _rekGateRep = rektoratAsset?.userData?.gateReport || null;
  bootProfiler.subBuilders.gedungRektorat4Gates = {
    cpuMs: +_s06CpuMs.toFixed(2),
    gate1_EnvelopeAndNaN: Boolean(_rekGateRep?.gate1_EnvelopeAndNaN),
    gate2_ManifoldNormals: Boolean(_rekGateRep?.gate2_ManifoldNormals),
    gate3_ZeroGapAndKinematics: Boolean(_rekGateRep?.gate3_ZeroGapAndKinematics),
    childCount: Array.isArray(rektoratAsset?.children) ? rektoratAsset.children.length : 0
  };
  bootProfiler.endStage(_tokS06, {
    cpuMs: _s06CpuMs,
    detail: `Menara Segi-8 & Tajug 3-Tumpang (${_s06CpuMs.toFixed(1)}ms CPU)`
  });
  if (typeof window !== 'undefined' && window.NexusForensic) {
    window.NexusForensic.markMilestone('rektorat_4gates_built', 'Menara Segi-8 & Tajug 3-Tumpang selesai');
  }

  // Stage 4: Build Kepulauan UI (Mata Air, Sungai Berkelok, Danau Kecil, & Air Terjun Niagara Mini)
  const _tokS07 = bootProfiler.beginStage(
    'S07_terrain_dem_and_ocean',
    'WORLD_BUILD',
    'DEM Topografi Pulau U & I, Danau, Sungai, Air Terjun & Samudra Jawa'
  );
  await preloader.advance(4, 58, 'Membentuk Mata Air, Sungai Berkelok Pulau U & Air Terjun Niagara Mini Pulau I...', 1002000);
  const _tS07Cpu0 = _nowBoot();
  const { group: terrainGroup, groundHeightFn } = TerrainDEMLoader.buildCampusTerrain(
    waterMaterial,
    sharedUniforms
  );
  const _s07CpuMs = Math.max(0.0, _nowBoot() - _tS07Cpu0);
  bootProfiler.endStage(_tokS07, {
    cpuMs: _s07CpuMs,
    detail: `Terrain Pulau U & I + Samudra (${_s07CpuMs.toFixed(1)}ms CPU)`
  });

  // Stage 5: Build Plaza Rotunda, Kompleks DPPD–ILRC–STP, Air Mancur Dancing, Jogging/Calisthenics, Jembatan Teksas UI, Underpass, & Dermaga
  const _tokS08 = bootProfiler.beginStage(
    'S08_corridors_dppd_and_fountain',
    'WORLD_BUILD',
    'Plaza Rotunda, Kompleks DPPD–STP, Jembatan Teksas, Underpass & Air Mancur'
  );
  await preloader.advance(5, 70, 'Merakit Kompleks DPPD–STP, Air Mancur Dancing, Jogging Track & Jaringan Jalan Pulau U...', 1205000);
  const _tS08Cpu0 = _nowBoot();
  const corridorGroup = SplineCorridorBuilder.buildRotundaAndPromenades(sharedUniforms);
  const _tS08Fountain0 = _nowBoot();
  const rotundaFountain = new RotundaDancingFountain();
  const _tS08End = _nowBoot();
  const _s08CpuMs = Math.max(0.0, _tS08End - _tS08Cpu0);
  bootProfiler.endStage(_tokS08, {
    cpuMs: _s08CpuMs,
    detail: `Koridor/DPPD ${(_tS08Fountain0 - _tS08Cpu0).toFixed(1)}ms + Air Mancur ${(_tS08End - _tS08Fountain0).toFixed(1)}ms`
  });

  // Stage 6: Build Dense Southern Highland Rainforest (Pulau U), Sparse Montane Forest (Pulau I) & 64 Photocell Streetlamps
  const _tokS09a = bootProfiler.beginStage(
    'S09a_biome_trees_and_lamps',
    'WORLD_BUILD',
    'Hutan Hujan Tropis Pulau U, Lereng Gunung Pulau I & 64 Lampu Sensor'
  );
  await preloader.advance(6, 82, 'Menumbuhkan Hutan Lebat Pulau U, Lereng Gunung Pulau I, & 64 Lampu Sensor Cahaya...', 3951674);
  const _tS09aCpu0 = _nowBoot();
  const biomeGroup = AutogenBiomeSpawner.buildCampusBiome(groundHeightFn, sharedUniforms);
  const _s09aCpuMs = Math.max(0.0, _nowBoot() - _tS09aCpu0);
  bootProfiler.endStage(_tokS09a, {
    cpuMs: _s09aCpuMs,
    detail: `Bioma hutan & 64 lampu (${_s09aCpuMs.toFixed(1)}ms CPU)`
  });

  const _tokS09b = bootProfiler.beginStage(
    'S09b_world_solid_colliders',
    'WORLD_BUILD',
    'Kompilasi Collider Fisika Padat 3D Seluruh Kepulauan'
  );
  const solidWorld = SplineCorridorBuilder.buildWorldSolidColliders(biomeGroup);
  bootProfiler.subBuilders.solidWorldCollidersCount = Array.isArray(solidWorld?.staticColliders)
    ? solidWorld.staticColliders.length
    : 0;
  bootProfiler.endStage(_tokS09b, {
    detail: `${bootProfiler.subBuilders.solidWorldCollidersCount} collider padat statis`
  });

  // Stage 7: Build 69 22-Joint Sivitas Akademika NPCs + Taxi Heli Listrik eVTOL + Cessna & Learjet 3x Sehari + Bus Kuning + Feri + 48 Cats
  const _tokS10 = bootProfiler.beginStage(
    'S10_sivitas_npcs_cats_fleet',
    'WORLD_BUILD',
    '69 Sivitas 22-Sendi, 48 Kucing, Bikun, eVTOL, Pesawat & Armada Laut'
  );
  await preloader.advance(7, 90, 'Menghadirkan 69 Sivitas Akademika, Taxi Heli Listrik, Pesawat 3x Sehari, & 48 Kucing...', TOTAL_EXPECTED_TRIS);
  const _tS10Cpu0 = _nowBoot();
  const sivitasAI = new DynamicTrafficAI(groundHeightFn, sharedUniforms);
  sivitasAI.setSolidWorld(solidWorld);
  const sivitasGroup = sivitasAI.build();
  worldRoot.add(terrainGroup, corridorGroup, rotundaFountain.mesh, biomeGroup, rektoratAsset, sivitasGroup);
  const _s10CpuMs = Math.max(0.0, _nowBoot() - _tS10Cpu0);
  bootProfiler.endStage(_tokS10, {
    cpuMs: _s10CpuMs,
    detail: `${sivitasAI.npcs.length} sivitas, ${sivitasAI.cats.length} kucing, armada darat/laut/udara (${_s10CpuMs.toFixed(1)}ms CPU)`
  });
  if (typeof window !== 'undefined' && window.NexusForensic) {
    window.NexusForensic.markMilestone(
      'archipelago_and_sivitas_built',
      `${sivitasAI.npcs.length} sivitas, ${sivitasAI.cats.length} kucing, armada darat/laut/udara siap`
    );
  }

  // Seamless 1-Draw-Call PBR + Shadow-Mapped Manicured Tropical Campus Turf + View-Aware Dual-Perspective PostFX
  const _tokS11 = bootProfiler.beginStage(
    'S11_tropical_grass_3d',
    'WORLD_BUILD',
    'Bake Tekstur Masking Rumput 896×1344 (3-Pass) & Instancing Rumput 3D'
  );
  await preloader.advance(7, 92, 'Menumbuhkan 288.000 Helai Rumput Gajah Mini & Masking Presisi...', TOTAL_EXPECTED_TRIS);
  const oceanWaterMesh = terrainGroup.children.find((c) => c && c.material === waterMaterial) || null;
  const initTierName = hwGovernor.activeTierId || hwGovernor.activeTier || 'TIER_S';
  const grassField = await TropicalGrassField.createAsync(
    {
      bladeCount: hwGovernor.isMobileDevice ? 16000 : initTierName === 'TIER_S' ? 48000 : 34000,
      sharedUniforms
    },
    yieldToBrowserFrame
  );
  scene.add(grassField.mesh);
  const _gfInit = grassField.initForensics || null;
  bootProfiler.subBuilders.tropicalGrass3D = _gfInit;
  bootProfiler.endStage(_tokS11, {
    cpuMs: _gfInit ? _gfInit.totalCpuMs : undefined,
    waitMs: _gfInit ? _gfInit.yieldWaitMs : undefined,
    detail: _gfInit
      ? `${_gfInit.bladeCount} rumpun · Bake CPU ${_gfInit.bakeCpuMs}ms (P1 ${_gfInit.pass1DemSdfCpuMs} + P2 ${_gfInit.pass2LawnMaskCpuMs} + P3 ${_gfInit.pass3ErosionCpuMs}) + Inst ${_gfInit.instancePopulateCpuMs}ms · ${_gfInit.yieldCount} yield`
      : `${grassField.bladeCount} rumpun rumput 3D`
  });

  const _tokS12 = bootProfiler.beginStage(
    'S12_postfx_and_initial_sim',
    'RENDERER_INIT',
    'Alokasi FBO PostFX/MSAA, Ephemeris Awal & Pengarah Kamera'
  );
  const _tS12Cpu0 = _nowBoot();
  const planarReflHiddenList = [oceanWaterMesh, grassField.mesh].filter(Boolean);
  const drawBufSize = new THREE.Vector2();
  renderer.getDrawingBufferSize(drawBufSize);
  const postFX = new CinematicPostFX(renderer, drawBufSize.x, drawBufSize.y, {
    msaaSamples: hwGovernor.isMobileDevice ? 0 : 4,
    planarReflSize: hwGovernor.isMobileDevice ? 384 : 512,
    planarEnabled: !hwGovernor.isMobileDevice
  });
  const _s12PostFxFboCpuMs = Math.max(0.0, _nowBoot() - _tS12Cpu0);

  weather.onThunder = (distanceMeters, intensity, immediate = false) => {
    audioSynth.triggerThunder(distanceMeters, intensity, weather.uniforms.uLightningDir.value, {
      immediate: Boolean(immediate)
    });
  };

  // Stage 8: Apply Depok WIB Clock & Real-Time Ephemeris Immediately & Compile Specialized GPU Shaders Asynchronously
  await preloader.advance(8, 93, 'Kompilasi Inkremental Shader GPU & Membuka Kepulauan UI...', TOTAL_EXPECTED_TRIS);
  const _tS12SimCpu0 = _nowBoot();

  let isLiveDepok = true;
  latestDepokClock = DepokLiveWeatherService.getDepokWIBClock();
  weather.setReferenceAstronomyDate(new Date(), sharedUniforms);
  weather.setSolarHour(latestDepokClock.solarHour, sharedUniforms, { immediate: true });
  weather.setWeather(latestDepokWeather.weatherMode, {
    immediate: true,
    cloudCoverNorm: latestDepokWeather.cloudCoverNorm
  });
  weather.update(0.016, camera, sharedUniforms);
  const initSunElevDeg =
    (Math.asin(Math.max(-1.0, Math.min(1.0, Number(weather.uniforms.uTrueSunDir.value.y) || 0.0))) * 180.0) / Math.PI;
  rotundaFountain.update(0.016, initSunElevDeg, sharedUniforms, weather.uniforms.uSunDir.value);
  GedungRektoratUIAsset.setNightIllumination(rektoratAsset, weather.nightIlluminationLevel);
  sivitasAI.update(0.016, 0.016, camera.position);

  // Full 3D Solid-Body Non-Penetration World Colliders (Buildings, Columns, Roofs, Trees, Lamps, Benches, Piers, Bridges, Underpass, DPPD, ATC)
  const walkAndDroneColliders = solidWorld.staticColliders;

  const sampleWalkableElevation = (x, z) => {
    const underpassY = TerrainDEMLoader.sampleUnderpassDeckY(x, z);
    if (underpassY !== null) {
      const isEnclosedBore =
        z >= TerrainDEMLoader.UNDERPASS_WEST_PORTAL_Z &&
        z <= TerrainDEMLoader.UNDERPASS_EAST_PORTAL_Z;
      if (!isEnclosedBore || camera.position.y < -2.2) {
        const onSidewalk =
          Math.abs(x - TerrainDEMLoader.UNDERPASS_X) >=
          TerrainDEMLoader.UNDERPASS_CARRIAGEWAY_HALF_W - 0.05;
        return underpassY + (onSidewalk ? 0.19 : 0.02);
      }
    }
    const ferryPierY = TerrainDEMLoader.sampleFerryPierDeckY(x, z);
    if (ferryPierY !== null) {
      return ferryPierY + 0.02;
    }
    const teksasDeckY = TerrainDEMLoader.sampleTeksasBridgeDeckY(x, z);
    if (teksasDeckY !== null) {
      const teksasSoffitY = TerrainDEMLoader.sampleTeksasBridgeSoffitY(z);
      if (teksasSoffitY === null || camera.position.y >= teksasSoffitY - 0.15) {
        return teksasDeckY + 0.02;
      }
    }
    const rDPPD = Math.hypot(x - TerrainDEMLoader.DPPD_CX, z - TerrainDEMLoader.DPPD_CZ);
    if (rDPPD <= 39.55) {
      const rLocal = rDPPD / TerrainDEMLoader.DPPD_SCALE;
      if (rLocal >= 53.1 && rLocal <= 65.1) {
        return TerrainDEMLoader.DPPD_PLATEAU_Y + 0.08 * TerrainDEMLoader.DPPD_SCALE;
      }
      return TerrainDEMLoader.DPPD_PLATEAU_Y + 0.24 * TerrainDEMLoader.DPPD_SCALE;
    }
    if (x >= -36.0 && x <= 36.0 && z >= -35.0 && z <= 39.0) {
      return 0.13;
    }
    if (x >= -16.0 && x <= 11.0 && z >= 39.0 && z <= 71.5) {
      return 0.15;
    }
    const distHeli = Math.hypot(x - TerrainDEMLoader.HELIPAD_CX, z - TerrainDEMLoader.HELIPAD_CZ);
    if (distHeli <= 14.6) {
      return TerrainDEMLoader.HELIPAD_DECK_Y + 0.015;
    }
    if (x >= -90.2 && x <= -81.2 && Math.abs(z - TerrainDEMLoader.HELIPAD_CZ) <= 4.4) {
      const rampT = Math.max(0.0, Math.min(1.0, (-81.2 - x) / 9.0));
      return 0.032 + rampT * (TerrainDEMLoader.HELIPAD_DECK_Y - 0.032);
    }
    if (Math.abs(x + 84.8) <= 2.55 && Math.abs(z - 118.2) <= 2.55) {
      return 0.30;
    }
    // Lapangan Rotunda UI (-2.5, 106.5): exact circular zebra curbs, cambered asphalt, & EPDM jogging track
    const rRotW = Math.hypot(x - -2.5, z - 106.5);
    if (rRotW >= 23.34 && rRotW <= 38.36) {
      if ((rRotW >= 23.34 && rRotW <= 23.91) || (rRotW >= 33.69 && rRotW <= 34.26)) {
        const cutW = Math.hypot(x - -36.45, z - 106.5) < 5.2;
        const cutT = Math.hypot(x - -20.6, z - 135.2) < 2.9;
        const cutS = Math.hypot(x - -2.5, z - 72.55) < 2.5;
        if (!cutW && !cutT && !cutS) return 0.186;
      }
      if (rRotW > 23.91 && rRotW < 33.69) {
        const fR = (rRotW - 28.8) / 4.9;
        return 0.035 + Math.max(0.0, 1.0 - fR * fR) * 0.016;
      }
      if (rRotW >= 35.95 && rRotW <= 38.36) {
        return 0.038;
      }
    }
    // Pulau I Straight ICAO Runway RWY 36/18, Blast Pads, Stub Taxiways & West Aircraft Parking Apron
    if (x >= -382.6 && x <= -127.4 && Math.abs(z - TerrainDEMLoader.RUNWAY_CZ) <= 12.5) {
      return TerrainDEMLoader.RUNWAY_DECK_Y + 0.02;
    }
    if (x >= -284.6 && x <= -223.4 && z >= 659.5 && z <= 697.5) {
      return TerrainDEMLoader.RUNWAY_DECK_Y + 0.02;
    }
    const fwY = TerrainDEMLoader.sampleWaterSurfaceY(x, z);
    if (fwY !== null) {
      return Math.max(fwY + 0.08, groundHeightFn(x, z));
    }
    return Math.max(-0.2, groundHeightFn(x, z));
  };

  const camOceanWaveScratch = {
    eta: 0.0,
    surgeX: 0.0,
    surgeZ: 0.0,
    nx: 0.0,
    ny: 1.0,
    nz: 0.0,
    dShore: 100.0,
    diffractionKd: 1.0,
    bathymetryH: 22.0,
    breakingDiss: 0.0
  };
  const sampleLiveOceanWaveY = (x, z) => {
    WeatherOpticsEngine.evaluateOceanWaveField(
      x,
      z,
      weather.uniforms.uTime.value,
      weather.getSeaWaveHeightMeters(),
      camOceanWaveScratch
    );
    return TerrainDEMLoader.WATER_LEVEL_Y + camOceanWaveScratch.eta;
  };

  const camDirector = new CameraDirector(camera, controls, walkAndDroneColliders);
  camDirector.setSharedUniforms(sharedUniforms);
  camDirector.setFPSAnchorProvider((mode, subjectOffset) => sivitasAI.getFPSAnchor(mode, subjectOffset));
  camDirector.setWorldPhysicsContext({
    staticColliders: solidWorld.staticColliders,
    getDynamicColliders: () => sivitasAI.getDynamicSolidColliders(),
    underpassConfig: solidWorld.underpassConfig,
    groundHeightFn,
    walkableHeightFn: sampleWalkableElevation,
    coastDistFn: TerrainDEMLoader.getIslandCoastDistance,
    waterSurfaceFn: TerrainDEMLoader.sampleWaterSurfaceY,
    oceanWaterY: TerrainDEMLoader.WATER_LEVEL_Y,
    oceanWaveSurfaceFn: sampleLiveOceanWaveY,
    bridgeDeckFn: TerrainDEMLoader.sampleTeksasBridgeDeckY,
    bridgeSoffitFn: TerrainDEMLoader.sampleTeksasBridgeSoffitY,
    bridgeCenterX: TerrainDEMLoader.TEKSAS_BRIDGE_X,
    bridgeHalfW: 2.6,
    ferryPierDeckFn: TerrainDEMLoader.sampleFerryPierDeckY
  });
  camDirector.bindKeyboard(window);

  // Default Startup View: Mavic 3 Pro Drone View Circling Pulau U & Kepulauan UI
  let arrivalEaseClock = 0.0;
  const arrivalStartY = presets.oblique45.pos.y + 7.5;
  const arrivalEndY = presets.oblique45.pos.y;
  let arrivalActive = false;
  camDirector.setMode('DRONE_6DOF', { startIslandTour: true });
  camDirector.syncPreviousCameraPosition();

  // Frame-Yielded Incremental Parallel GPU Shader Pre-Compilation (2 objects/slice + rAF yield -> ZERO multi-tab GPU hang!)
  weather.uniforms.uCamPos.value.copy(camera.position);
  if (weather.skyMesh) {
    weather.skyMesh.position.copy(camera.position);
  }
  const _s12SimCpuMs = Math.max(0.0, _nowBoot() - _tS12SimCpu0);
  const _s12TotalCpuMs = _s12PostFxFboCpuMs + _s12SimCpuMs;
  await yieldToBrowserFrame();
  bootProfiler.endStage(_tokS12, {
    cpuMs: _s12TotalCpuMs,
    detail: `FBO PostFX (${postFX.msaaSamples}× MSAA, Refl ${postFX.planarReflSize}px) ${_s12PostFxFboCpuMs.toFixed(1)}ms + Sim Awal ${_s12SimCpuMs.toFixed(1)}ms`
  });

  const _tokS13 = bootProfiler.beginStage(
    'S13_gpu_shader_prewarm',
    'SHADER_WARMUP',
    'Pre-Warm Inkremental Shader GPU, PostFX Quad, Planar Refl & Shadow Pass'
  );
  const tWarm0 = performance.now();
  // Bind the EXACT target WebGLRenderTarget during shader compilation and poll KHR_parallel_shader_compile
  // directly via renderer.info.programs + gl.flush() (avoids Three.js r170 compileAsync's infinite setTimeout loop
  // when a material has currentProgram === undefined, while keeping 100% parallel driver compilation!)
  const _glWarm = typeof renderer.getContext === 'function' ? renderer.getContext() : null;
  const _khrParallelExt =
    _glWarm && typeof _glWarm.getExtension === 'function'
      ? _glWarm.getExtension('KHR_parallel_shader_compile')
      : null;
  const _waitParallelProgramsReady = async (startProgIdx = 0, maxWaitMs = 1800.0) => {
    if (!_glWarm || !_khrParallelExt || !Array.isArray(renderer.info?.programs)) return;
    const progs0 = renderer.info.programs;
    if (progs0.length <= startProgIdx) return;
    if (typeof _glWarm.flush === 'function') _glWarm.flush();
    const COMP_ENUM = _khrParallelExt.COMPLETION_STATUS_KHR || 0x91b1;
    const deadline = _nowBoot() + maxWaitMs;
    while (_nowBoot() < deadline) {
      const progs = renderer.info.programs;
      let allReady = true;
      for (let pi = startProgIdx; pi < progs.length; pi++) {
        const pr = progs[pi];
        if (!pr) continue;
        // Skip disposed programs (e.g. PMREMGenerator temporary cubemap shaders) whose GL handle is no longer live!
        if (pr.program && typeof _glWarm.isProgram === 'function' && !_glWarm.isProgram(pr.program)) {
          continue;
        }
        let ready = true;
        if (pr.program && typeof _glWarm.getProgramParameter === 'function') {
          const st = _glWarm.getProgramParameter(pr.program, COMP_ENUM);
          ready = st === null ? true : Boolean(st);
        } else if (typeof pr.isReady === 'function') {
          ready = Boolean(pr.isReady());
        }
        if (!ready) {
          allReady = false;
          break;
        }
      }
      if (allReady) return;
      if (typeof _glWarm.flush === 'function') _glWarm.flush();
      await new Promise((r) => setTimeout(r, 5));
    }
  };

  const compileAsyncSlice = async (targetScene, targetCam, targetRT = null) => {
    const tSliceWall0 = _nowBoot();
    let syncCallCpuMs = 0.0;
    const prevRT = renderer.getRenderTarget();
    const progsBefore = Array.isArray(renderer.info?.programs) ? renderer.info.programs.length : 0;
    renderer.setRenderTarget(targetRT);
    try {
      if (typeof renderer.compile === 'function') {
        const tSync0 = _nowBoot();
        renderer.compile(targetScene, targetCam);
        if (_glWarm && typeof _glWarm.flush === 'function') _glWarm.flush();
        syncCallCpuMs = Math.max(0.0, _nowBoot() - tSync0);
        if (!hwGovernor.isMobileDevice && _khrParallelExt) {
          await _waitParallelProgramsReady(progsBefore, 8000.0);
        }
      }
    } finally {
      renderer.setRenderTarget(prevRT);
    }
    return {
      wallMs: Math.max(0.0, _nowBoot() - tSliceWall0),
      syncCallCpuMs
    };
  };

  const _describeWarmNode = (node) => {
    if (!node) return 'Node';
    const rawName = node.name ? String(node.name).trim() : '';
    const mats = Array.isArray(node.material) ? node.material : [node.material];
    const m0 = mats[0];
    const cKey =
      m0 && typeof m0.customProgramCacheKey === 'function'
        ? String(m0.customProgramCacheKey() || '').slice(0, 28)
        : m0?.type || 'Mat';
    const kind = node.isInstancedMesh ? 'Inst' : node.isPoints ? 'Pts' : 'Mesh';
    return rawName ? `${rawName.slice(0, 32)} (${kind}:${cKey})` : `${kind}:${cKey}`;
  };

  // Temporarily expose conditional weather/grass meshes and non-empty InstancedMesh count
  // so KHR_parallel_shader_compile pre-warms 100% of active GPU programs without compiling unused hidden helpers!
  const transientWarmMeshes = [
    weather.rainAndDripMesh,
    weather.splashAndMistMesh,
    weather.fireflyPoints,
    weather.volcanicPlumeMesh,
    grassField.mesh
  ].filter(Boolean);
  const savedTransientVis = transientWarmMeshes.map((m) => m.visible);
  for (let i = 0; i < transientWarmMeshes.length; i++) {
    transientWarmMeshes[i].visible = true;
  }
  const savedEmptyInstMeshes = [];
  scene.traverse((n) => {
    if (n && n.visible !== false && n.isInstancedMesh && n.count === 0) {
      savedEmptyInstMeshes.push(n);
      n.count = 1;
    }
  });

  // Collect all active renderables and deduplicate by exact WebGLProgram archetype (unified nexus-omni-v54-uber -> ~11 programs total!)
  const visibleRenderables = [];
  const uniqueWarmRenderables = [];
  const seenWarmKeys = new Set();
  const collectVisibleRenderables = (node) => {
    if (!node || node.visible === false) return;
    if (node.isMesh || node.isPoints || node.isLine) {
      visibleRenderables.push(node);
      const mats = Array.isArray(node.material) ? node.material : [node.material];
      const matKey = mats
        .map((m) => {
          if (!m) return '';
          const cKey =
            typeof m.customProgramCacheKey === 'function' ? m.customProgramCacheKey() : m.uuid;
          return `${m.type}:${cKey}:${m.flatShading ? 1 : 0}:${m.side}`;
        })
        .join('|');
      const instKey = node.isInstancedMesh ? 'I' : 'M';
      const vColKey = node.geometry?.attributes?.color ? 'C' : 'N';
      const uvKey = node.geometry?.attributes?.uv ? 'U' : '0';
      const nrmKey = node.geometry?.attributes?.normal ? 'N' : '0';
      const tanKey = node.geometry?.attributes?.tangent ? 'T' : '0';
      const skinKey = node.geometry?.attributes?.aBoneIndices ? 'S' : 'R';
      const depthKey = node.customDepthMaterial
        ? typeof node.customDepthMaterial.customProgramCacheKey === 'function'
          ? node.customDepthMaterial.customProgramCacheKey()
          : node.customDepthMaterial.uuid
        : '0';
      const shKey = `${node.receiveShadow ? 1 : 0}${node.castShadow ? 1 : 0}`;
      const key = `${matKey}:${instKey}:${vColKey}:${uvKey}:${nrmKey}:${tanKey}:${skinKey}:${depthKey}:${shKey}`;
      if (!seenWarmKeys.has(key)) {
        seenWarmKeys.add(key);
        uniqueWarmRenderables.push(node);
      }
    }
    const ch = node.children;
    if (Array.isArray(ch)) {
      for (let i = 0; i < ch.length; i++) {
        collectVisibleRenderables(ch[i]);
      }
    }
  };
  const _tCollect0 = _nowBoot();
  collectVisibleRenderables(scene);
  let _totalWarmCpuMs = Math.max(0.0, _nowBoot() - _tCollect0);

  bootProfiler.beginShaderWarmup({
    compileMode:
      !hwGovernor.isMobileDevice && typeof renderer.compileAsync === 'function'
        ? 'KHR_parallel_compileAsync'
        : 'sync_renderer_compile',
    msaaSamples: postFX ? postFX.msaaSamples : 0,
    planarReflSize: postFX ? postFX.planarReflSize : 0,
    uniqueArchetypesCount: uniqueWarmRenderables.length
  });

  const mainSceneRT = postFX && postFX.msaaTarget ? postFX.msaaTarget : null;
  if (visibleRenderables.length > 0) {
    for (let i = 0; i < visibleRenderables.length; i++) {
      visibleRenderables[i].visible = false;
    }
    const SLICE_SIZE = hwGovernor.isMobileDevice ? 3 : 4;
    const totalSlices = Math.ceil(uniqueWarmRenderables.length / SLICE_SIZE);
    for (let sIdx = 0; sIdx < totalSlices; sIdx++) {
      const startI = sIdx * SLICE_SIZE;
      const endI = Math.min(uniqueWarmRenderables.length, startI + SLICE_SIZE);
      const sliceArchetypes = [];
      for (let i = startI; i < endI; i++) {
        uniqueWarmRenderables[i].visible = true;
        sliceArchetypes.push(_describeWarmNode(uniqueWarmRenderables[i]));
      }
      const pct = Math.min(99, 94 + Math.floor(((sIdx + 1) / Math.max(1, totalSlices)) * 5));
      const tPreAdv0 = _nowBoot();
      await preloader.advance(
        8,
        pct,
        `Kompilasi Paralel Shader GPU (${endI}/${uniqueWarmRenderables.length})...`,
        TOTAL_EXPECTED_TRIS
      );
      const preAdvWaitMs = Math.max(0.0, _nowBoot() - tPreAdv0);
      const sliceRes = await compileAsyncSlice(scene, camera, mainSceneRT);
      const tYield0 = _nowBoot();
      await yieldToBrowserFrame();
      const postSliceYieldMs = Math.max(0.0, _nowBoot() - tYield0);
      _totalWarmCpuMs += sliceRes.syncCallCpuMs;
      bootProfiler.recordShaderWarmupStep({
        stepId: `slice_${sIdx + 1}_of_${totalSlices}`,
        phase: 'world_archetype_slice',
        archetypes: sliceArchetypes,
        wallMs: sliceRes.wallMs,
        cpuMs: sliceRes.syncCallCpuMs,
        yieldAfterMs: preAdvWaitMs + postSliceYieldMs
      });
      for (let i = startI; i < endI; i++) {
        uniqueWarmRenderables[i].visible = false;
      }
    }
    for (let i = 0; i < visibleRenderables.length; i++) {
      visibleRenderables[i].visible = true;
    }
  } else {
    const fallbackSlice = await compileAsyncSlice(scene, camera, mainSceneRT);
    _totalWarmCpuMs += fallbackSlice.syncCallCpuMs;
    bootProfiler.recordShaderWarmupStep({
      stepId: 'slice_fallback_full_scene',
      phase: 'world_archetype_slice',
      archetypes: ['FullScene'],
      wallMs: fallbackSlice.wallMs,
      cpuMs: fallbackSlice.syncCallCpuMs,
      yieldAfterMs: 0
    });
  }

  // Pre-warm fullscreen PostFX quad programs in a single parallel batch with exact runtime RenderTargets
  if (postFX && postFX._fsScene && postFX._fsCam && postFX._fsQuad && typeof renderer.compile === 'function') {
    const tPfxWall0 = _nowBoot();
    let pfxCpuMs = 0.0;
    const prevRT = renderer.getRenderTarget();
    const pfxProgStart = Array.isArray(renderer.info?.programs) ? renderer.info.programs.length : 0;
    const bloomRT = postFX.bloomHalfA || null;
    try {
      const tPfxSync0 = _nowBoot();
      if (postFX._bloomDownMat) {
        renderer.setRenderTarget(bloomRT);
        postFX._fsQuad.material = postFX._bloomDownMat;
        renderer.compile(postFX._fsScene, postFX._fsCam);
      }
      if (postFX._bloomUpMat) {
        renderer.setRenderTarget(bloomRT);
        postFX._fsQuad.material = postFX._bloomUpMat;
        renderer.compile(postFX._fsScene, postFX._fsCam);
      }
      if (postFX._compositeMat) {
        renderer.setRenderTarget(null);
        postFX._fsQuad.material = postFX._compositeMat;
        renderer.compile(postFX._fsScene, postFX._fsCam);
      }
      if (_glWarm && typeof _glWarm.flush === 'function') _glWarm.flush();
      pfxCpuMs = Math.max(0.0, _nowBoot() - tPfxSync0);
      if (!hwGovernor.isMobileDevice && _khrParallelExt) {
        await _waitParallelProgramsReady(pfxProgStart, 4000.0);
      }
    } finally {
      renderer.setRenderTarget(prevRT);
    }
    const pfxWallMs = Math.max(0.0, _nowBoot() - tPfxWall0);
    const tPfxYield0 = _nowBoot();
    await yieldToBrowserFrame();
    const pfxYieldMs = Math.max(0.0, _nowBoot() - tPfxYield0);
    _totalWarmCpuMs += pfxCpuMs;
    bootProfiler.recordShaderWarmupStep({
      stepId: 'postfx_fullscreen_quads',
      phase: 'postfx_compile',
      archetypes: ['BloomDown', 'BloomUp', 'CompositeHVS'],
      wallMs: pfxWallMs,
      cpuMs: pfxCpuMs,
      yieldAfterMs: pfxYieldMs
    });
  }

  const isDroneCinemaView = () =>
    camDirector.mode !== 'WALK_SIVITAS' &&
    !camDirector.isFPSMode() &&
    !String(camDirector.mode || '').startsWith('FPS_');

  const _tWarmPrep0 = _nowBoot();
  videotron.update(camera, window.innerWidth, window.innerHeight, sharedUniforms);
  grassField.update(camera, { windVec: weather.uniforms.uWindVec.value }, sharedUniforms);
  postFX.updateMode(10.0, isDroneCinemaView(), {
    duskFactor: weather.uniforms.uDuskFactor.value,
    nightFactor: weather.uniforms.uNightFactor.value
  });
  // Temporarily disable frustum culling and force visibility on all collected renderables
  // during hidden warmup passes so WebGLShadowMap & planar reflection compile 100% of GPU programs before boot completes!
  const savedWarmFrustumCulled = visibleRenderables.map((n) => n.frustumCulled);
  const savedPostPrepVis = visibleRenderables.map((n) => n.visible);
  for (let i = 0; i < visibleRenderables.length; i++) {
    visibleRenderables[i].frustumCulled = false;
    visibleRenderables[i].visible = true;
  }
  _totalWarmCpuMs += Math.max(0.0, _nowBoot() - _tWarmPrep0);

  const _tWarmRefl0 = _nowBoot();
  postFX.renderPlanarWaterReflection(
    scene,
    camera,
    weather,
    TerrainDEMLoader.WATER_LEVEL_Y,
    planarReflHiddenList
  );
  const _warmReflCpuMs = Math.max(0.0, _nowBoot() - _tWarmRefl0);
  const _tWarmReflYield0 = _nowBoot();
  await yieldToBrowserFrame();
  const _warmReflYieldMs = Math.max(0.0, _nowBoot() - _tWarmReflYield0);
  _totalWarmCpuMs += _warmReflCpuMs;
  bootProfiler.recordShaderWarmupStep({
    stepId: 'hidden_planar_reflection_warm_pass',
    phase: 'planar_reflection_render',
    archetypes: ['PlanarWaterMirrorPass'],
    wallMs: _warmReflCpuMs,
    cpuMs: _warmReflCpuMs,
    yieldAfterMs: _warmReflYieldMs
  });

  const _tWarmMain0 = _nowBoot();
  // Ensure sunLight shadow pass compiles all MeshDepthMaterial variants even if simulator boots at night WIB!
  const _sunWarm = weather.sunLight || null;
  const _savedSunVis = _sunWarm ? _sunWarm.visible : true;
  const _savedSunCast = _sunWarm ? _sunWarm.castShadow : true;
  const _savedSunInt = _sunWarm ? _sunWarm.intensity : 1.0;
  if (_sunWarm) {
    _sunWarm.visible = true;
    _sunWarm.castShadow = true;
    if (_sunWarm.intensity < 0.1) _sunWarm.intensity = 0.1;
  }
  if (renderer.shadowMap) {
    renderer.shadowMap.needsUpdate = true;
  }
  postFX.render(scene, camera);
  if (_sunWarm) {
    _sunWarm.visible = _savedSunVis;
    _sunWarm.castShadow = _savedSunCast;
    _sunWarm.intensity = _savedSunInt;
  }
  for (let i = 0; i < visibleRenderables.length; i++) {
    visibleRenderables[i].frustumCulled = savedWarmFrustumCulled[i];
    visibleRenderables[i].visible = savedPostPrepVis[i];
  }
  for (let i = 0; i < savedEmptyInstMeshes.length; i++) {
    savedEmptyInstMeshes[i].count = 0;
  }
  for (let i = 0; i < transientWarmMeshes.length; i++) {
    transientWarmMeshes[i].visible = savedTransientVis[i];
  }
  weather._syncParticleVisibility();
  const _warmMainCpuMs = Math.max(0.0, _nowBoot() - _tWarmMain0);
  const _tWarmMainYield0 = _nowBoot();
  await yieldToBrowserFrame();
  const _warmMainYieldMs = Math.max(0.0, _nowBoot() - _tWarmMainYield0);
  _totalWarmCpuMs += _warmMainCpuMs;
  bootProfiler.recordShaderWarmupStep({
    stepId: 'hidden_main_and_shadow_warm_pass',
    phase: 'main_and_shadow_render',
    archetypes: ['MainMSAAScene+PCFShadowMap'],
    wallMs: _warmMainCpuMs,
    cpuMs: _warmMainCpuMs,
    yieldAfterMs: _warmMainYieldMs
  });

  // Measure a true steady-state post-compile frame (never feed one-time shader compile time into hwGovernor!)
  const tSteady0 = performance.now();
  postFX.render(scene, camera);
  const steadyFrameMs = performance.now() - tSteady0;
  if (steadyFrameMs <= 35.0) {
    hwGovernor.recordWarmupSample(steadyFrameMs);
  }
  _totalWarmCpuMs += Math.max(0.0, steadyFrameMs);
  bootProfiler.recordShaderWarmupStep({
    stepId: 'steady_calibration_render_pass',
    phase: 'steady_governor_calibration',
    archetypes: ['SteadyPostCompileRender'],
    wallMs: steadyFrameMs,
    cpuMs: steadyFrameMs,
    yieldAfterMs: 0
  });
  const warmupMs = performance.now() - tWarm0;
  const _warmedProgsCount = Array.isArray(renderer.info?.programs) ? renderer.info.programs.length : 0;
  bootProfiler.endShaderWarmup(warmupMs, _totalWarmCpuMs);
  bootProfiler.endStage(_tokS13, {
    cpuMs: _totalWarmCpuMs,
    waitMs: Math.max(0.0, warmupMs - _totalWarmCpuMs),
    detail: `Warmup ${warmupMs.toFixed(1)}ms (CPU ${_totalWarmCpuMs.toFixed(1)}ms) · Programs ${_warmedProgsCount}`
  });

  const liveForensicProbe = createForensicReadOnlyProbe({
    recorder: forensicRecorder,
    bootProfiler,
    renderer,
    camera,
    postFX,
    grassField,
    weather,
    audioSynth,
    videotron,
    rotundaFountain,
    hwGovernor,
    camDirector,
    sivitasAI,
    rektoratAsset,
    getWebglContextLost: () => webglContextLost,
    getIsLiveDepok: () => isLiveDepok,
    getLatestDepokWeather: () => latestDepokWeather,
    getLatestDepokClock: () => latestDepokClock,
    isDroneCinemaViewFn: isDroneCinemaView
  });
  if (typeof window !== 'undefined' && window.NexusForensic) {
    window.NexusForensic.registerEngineProbe(liveForensicProbe);
    window.NexusForensic.markMilestone(
      'shaders_prewarmed_ready',
      `Warmup ${warmupMs.toFixed(1)}ms · Programs ${_warmedProgsCount}`
    );
  }

  const _tokS14 = bootProfiler.beginStage(
    'S14_preloader_100_gpu_flush',
    'FIRST_FRAME_GAP',
    'Yield Preloader 100%, Flush Antrean GPU Pasca-Warmup & Buka Tirai'
  );
  const _tS14Adv0 = _nowBoot();
  await preloader.advance(8, 100, 'Selamat Datang di Kepulauan Universitas Indonesia (Pulau U & I), Depok.', TOTAL_EXPECTED_TRIS);
  const _s14WaitMs = Math.max(0.0, _nowBoot() - _tS14Adv0);
  const _tS14Dis0 = _nowBoot();
  preloader.dismiss();
  const _s14DisCpuMs = Math.max(0.0, _nowBoot() - _tS14Dis0);
  const _s14Entry = bootProfiler.endStage(_tokS14, {
    cpuMs: _s14DisCpuMs,
    waitMs: _s14WaitMs,
    detail: `rAF/GPU flush ${_s14WaitMs.toFixed(1)}ms + dismiss ${_s14DisCpuMs.toFixed(1)}ms`
  });
  if (_s14Entry) {
    bootProfiler.firstFrameGap.preloader100YieldAndGpuFlushMs = _s14Entry.wallMs;
  }

  const _tokS15 = bootProfiler.beginStage(
    'S15_ui_dom_and_viewport_bind',
    'FIRST_FRAME_GAP',
    'Inisialisasi Kontrol UI, Sinkronisasi Viewport Awal & DOM Telemetri'
  );
  const _tS15Cpu0 = _nowBoot();

  const useTouchControls = () =>
    Boolean(isMobilePlatform || (window.matchMedia && window.matchMedia('(pointer: coarse)').matches));

  // Non-blocking Arrival Coach Toast (Auto-fades or dismisses on first interaction)
  const welcomeToastEl = document.getElementById('nexus-welcome-toast');
  const welcomeTextEl = document.getElementById('nexus-welcome-text');
  let welcomeDismissed = false;
  let welcomeTimer = 0;
  function dismissWelcomeToast() {
    if (welcomeDismissed) return;
    welcomeDismissed = true;
    if (welcomeTimer) clearTimeout(welcomeTimer);
    welcomeToastEl?.classList.add('is-hidden');
  }
  if (welcomeToastEl) {
    if (welcomeTextEl && useTouchControls()) {
      welcomeTextEl.innerHTML =
        'Geser layar menoleh &middot; Cubit 2 jari zoom &middot; Coba <strong>&#9889; Avata 2</strong> di atas';
    }
    setTimeout(() => {
      if (!welcomeDismissed) welcomeToastEl.classList.remove('is-hidden');
    }, 300);
    welcomeTimer = setTimeout(dismissWelcomeToast, 5800);
  }

  // Pointer drag + 2-Finger Pinch Optical Zoom + Optional Pointer-Lock FPV look
  let dragging = false;
  let lastPX = 0;
  let lastPY = 0;
  const activeTouches = new Map();
  let prevPinchDist = 0;

  renderer.domElement.addEventListener('pointerdown', (e) => {
    dismissWelcomeToast();
    arrivalActive = false;
    camDirector.cancelTransition();
    if (e.pointerType === 'touch') {
      activeTouches.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (activeTouches.size === 2) {
        const pts = Array.from(activeTouches.values());
        prevPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        dragging = false;
        return;
      }
    }
    dragging = true;
    lastPX = e.clientX;
    lastPY = e.clientY;
  });
  renderer.domElement.addEventListener(
    'wheel',
    (e) => {
      dismissWelcomeToast();
      arrivalActive = false;
      camDirector.cancelTransition();
      if (camDirector.mode === 'DRONE_6DOF' || camDirector.mode === 'AUTO_CINEMATIC') {
        const step = Math.sign(e.deltaY || 0) * 2.2;
        if (step !== 0) {
          camDirector.adjustMavicCinemaZoom(step);
          syncViewport();
          syncQuickDockAndContextDOM();
        }
      }
    },
    { passive: true }
  );
  renderer.domElement.addEventListener('dblclick', () => {
    if (camDirector.mode === 'DRONE_6DOF' || camDirector.mode === 'AUTO_CINEMATIC') {
      camDirector.resetMavicGimbalAndZoom();
      syncViewport();
      syncQuickDockAndContextDOM();
      audioSynth.playUiChime(560);
    } else if (camDirector.isFPSMode()) {
      camDirector.resetFPSHeadLook();
      audioSynth.playUiChime(560);
    }
  });
  window.addEventListener('pointermove', (e) => {
    if (e.pointerType === 'touch' && activeTouches.has(e.pointerId)) {
      activeTouches.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (activeTouches.size >= 2) {
        const pts = Array.from(activeTouches.values());
        const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        if (prevPinchDist > 8 && (camDirector.mode === 'DRONE_6DOF' || camDirector.mode === 'AUTO_CINEMATIC')) {
          const dFov = (prevPinchDist - dist) * 0.11;
          if (Math.abs(dFov) > 0.05) {
            camDirector.adjustMavicCinemaZoom(dFov);
            syncViewport();
            syncQuickDockAndContextDOM();
          }
        }
        prevPinchDist = dist;
        return;
      }
    }
    if (document.pointerLockElement === renderer.domElement && camDirector.isAvataFPVMode()) {
      camDirector.lookDelta(e.movementX || 0, e.movementY || 0, 0.0021, true);
      return;
    }
    if (!dragging) return;
    camDirector.lookDelta(e.clientX - lastPX, e.clientY - lastPY, 0.0024, false);
    lastPX = e.clientX;
    lastPY = e.clientY;
  });
  const clearPointerTouch = (e) => {
    if (e && e.pointerType === 'touch') {
      activeTouches.delete(e.pointerId);
      if (activeTouches.size < 2) prevPinchDist = 0;
    }
    if (activeTouches.size === 0) dragging = false;
  };
  window.addEventListener('pointerup', clearPointerTouch);
  window.addEventListener('pointercancel', clearPointerTouch);

  // Non-Occluding Responsive UI Drawer (Default Collapsed — Hukum Emas 9)
  const pillBtn = document.getElementById('sim-pill-toggle');
  const pillLabelEl = document.getElementById('sim-pill-label');
  const drawer = document.getElementById('sim-drawer');
  const closeBtn = document.getElementById('sim-drawer-close');
  const avataHudEl = document.getElementById('avata-fpv-hud');
  const quickDockEl = document.getElementById('nexus-quick-dock');
  const contextBarEl = document.getElementById('nexus-context-bar');
  const exploreTouchPadEl = document.getElementById('explore-touch-pad');
  const helpPopoverEl = document.getElementById('nexus-help-popover');
  let isPanelExpanded = false;

  function syncViewport() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h);
    renderer.getDrawingBufferSize(drawBufSize);
    postFX.setSize(drawBufSize.x, drawBufSize.y);
    const rect = drawer.getBoundingClientRect();
    const baseFov = camDirector.getActiveBaseVerticalFov();
    applyNonOccludingViewportOffset(
      camera,
      w,
      h,
      isPanelExpanded,
      {
        width: rect.width || 276,
        height: rect.height || Math.min(h * 0.32, h * 0.34),
        side: 'right'
      },
      baseFov
    );
    if (avataHudEl) {
      avataHudEl.classList.toggle('drawer-open', isPanelExpanded);
    }
  }

  let activeDrawerTab = 'explore';
  let syncForensicDOMHook = null;

  function setPanelExpanded(expanded) {
    if (expanded) dismissWelcomeToast();
    isPanelExpanded = expanded;
    drawer.classList.toggle('collapsed', !expanded);
    drawer.setAttribute('aria-expanded', String(expanded));
    pillBtn.classList.toggle('hidden', expanded);
    if (expanded && document.pointerLockElement === renderer.domElement && document.exitPointerLock) {
      document.exitPointerLock();
    }
    if (expanded && activeDrawerTab === 'audit' && syncForensicDOMHook) {
      syncForensicDOMHook();
    }
    syncQuickDockAndContextDOM();
    requestAnimationFrame(syncViewport);
  }

  pillBtn.addEventListener('click', () => setPanelExpanded(true));
  closeBtn.addEventListener('click', () => setPanelExpanded(false));
  window.addEventListener('resize', syncViewport);
  const _prevPfxW = postFX.width;
  const _prevPfxH = postFX.height;
  const _tInitSyncVp0 = _nowBoot();
  syncViewport();
  bootProfiler.firstFrameGap.initialSyncViewportCpuMs = +Math.max(0.0, _nowBoot() - _tInitSyncVp0).toFixed(2);
  bootProfiler.firstFrameGap.postFxResizedOnViewportSync =
    postFX.width !== _prevPfxW || postFX.height !== _prevPfxH;

  // Tabs
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      activeDrawerTab = btn.dataset.tab || 'explore';
      document.getElementById(`tab-${activeDrawerTab}`).classList.add('active');
      if (activeDrawerTab === 'audit' && syncForensicDOMHook) {
        syncForensicDOMHook();
      }
      requestAnimationFrame(syncViewport);
    });
  });

  // --- DJI AVATA 2 ACTION FPV GOGGLES 3 OSD, ZONE LAUNCHPAD & DUAL TOUCH STICKS ---
  const avataOsdModeEl = document.getElementById('avata-osd-mode');
  const avataOsdZoneEl = document.getElementById('avata-osd-zone');
  const avataOsdAttitudeEl = document.getElementById('avata-osd-attitude');
  const avataHorizonGroupEl = document.getElementById('avata-horizon-group');
  const avataOsdSpdEl = document.getElementById('avata-osd-spd');
  const avataOsdMsEl = document.getElementById('avata-osd-ms');
  const avataOsdThrEl = document.getElementById('avata-osd-thr');
  const avataThrFillEl = document.getElementById('avata-thr-fill');
  const avataOsdAltEl = document.getElementById('avata-osd-alt');
  const avataOsdHomeEl = document.getElementById('avata-osd-home');
  const avataOsdFovEl = document.getElementById('avata-osd-fov');
  const avataFreeBtn = document.getElementById('btn-avata-free-toggle');
  const avataProfileBtn = document.getElementById('btn-avata-profile');
  const avataRollLBtn = document.getElementById('btn-avata-roll-l');
  const avataRollRBtn = document.getElementById('btn-avata-roll-r');
  const avataSplitSBtn = document.getElementById('btn-avata-splits');
  const avataMouseLockBtn = document.getElementById('btn-avata-mouse-lock');
  const avataOsdToggleBtn = document.getElementById('btn-avata-osd-toggle');
  const avataExitBtn = document.getElementById('btn-avata-exit');
  const avataZoneBtns = Array.from(document.querySelectorAll('[data-avata-zone]'));
  let avataOsdMinimal = false;

  function resolveActiveAvataZoneKey(maneuverId = '', zoneLabel = '') {
    const m = String(maneuverId || '');
    const z = String(zoneLabel || '').toUpperCase();
    if (m.startsWith('M01_') || m.startsWith('M18_') || z.includes('LORONG')) return 'LORONG_TEKSAS';
    if (m.startsWith('M02_') || m.startsWith('M09B_') || m.startsWith('M16_') || z.includes('KOLONG')) {
      return 'KOLONG_TEKSAS';
    }
    if (m.startsWith('M10_') || m.startsWith('M15_') || z.includes('UNDERPASS')) return 'UNDERPASS';
    if (
      m.startsWith('M11_') ||
      m.startsWith('M12_') ||
      z.includes('KAWAH') ||
      z.includes('NIAGARA') ||
      z.includes('GUNUNG')
    ) {
      return 'GUNUNG_NIAGARA';
    }
    if (m.startsWith('M04_') || m.startsWith('M07_') || z.includes('STP') || z.includes('DPPD')) {
      return 'DPPD_CANYON';
    }
    if (m.startsWith('M21_') || m.startsWith('M22_') || z.includes('REKTORAT') || z.includes('TAJUG')) {
      return 'REKTORAT';
    }
    return '';
  }

  function syncAvataHudDOM(fastFrameOnly = false) {
    if (!avataHudEl) return;
    const active = camDirector.isAvataFPVMode();
    if (avataHudEl.classList.contains('active') !== active) {
      avataHudEl.classList.toggle('active', active);
      avataHudEl.setAttribute('aria-hidden', String(!active));
      if (!active && document.pointerLockElement === renderer.domElement && document.exitPointerLock) {
        document.exitPointerLock();
      }
    }
    if (!active) return;

    const tel = camDirector.getAvataTelemetry();
    if (avataHorizonGroupEl) {
      const pitchShift = Math.max(-38, Math.min(38, tel.pitchDeg * 0.72));
      avataHorizonGroupEl.setAttribute(
        'transform',
        `translate(0 ${pitchShift.toFixed(1)}) rotate(${(-tel.rollDeg).toFixed(1)} 110 80)`
      );
    }
    const rushAlpha = Math.max(0, Math.min(0.85, (tel.speedKmh - 40) / 28));
    avataHudEl.style.setProperty('--avata-rush', rushAlpha.toFixed(2));
    if (fastFrameOnly) return;

    if (avataOsdModeEl) {
      const modeTxt = tel.autoFreestyle
        ? 'PRO-PILOT AI'
        : tel.profile === 'ACRO'
          ? 'BEBAS · ACRO+'
          : 'BEBAS · SPORT';
      avataOsdModeEl.textContent = modeTxt;
      avataOsdModeEl.classList.toggle('is-acro', tel.profile === 'ACRO');
    }
    if (avataOsdZoneEl) {
      avataOsdZoneEl.textContent = tel.apasActive
        ? `APAS 360° · ${tel.zoneLabel}`
        : tel.zoneLabel || 'KEPULAUAN UI';
      avataOsdZoneEl.classList.toggle('apas-active', Boolean(tel.apasActive));
    }
    if (avataOsdAttitudeEl) {
      const hdg = String(tel.yawDeg).padStart(3, '0');
      const bankVal = -tel.rollDeg;
      const bankSign = bankVal > 0 ? '+' : '';
      avataOsdAttitudeEl.textContent = `HDG ${hdg}° · ${tel.gForce.toFixed(1)}G · BANK ${bankSign}${bankVal}°`;
    }
    if (avataOsdSpdEl) avataOsdSpdEl.textContent = String(tel.speedKmh);
    if (avataOsdMsEl) avataOsdMsEl.textContent = `${tel.speedMs.toFixed(1)} m/s`;
    if (avataOsdThrEl) avataOsdThrEl.textContent = `THR ${tel.throttlePct}%`;
    if (avataThrFillEl) avataThrFillEl.style.width = `${tel.throttlePct}%`;
    if (avataOsdAltEl) avataOsdAltEl.textContent = tel.altitudeAGL.toFixed(1);
    if (avataOsdHomeEl) avataOsdHomeEl.textContent = `H ${tel.homeDistM} m`;
    if (avataOsdFovEl) avataOsdFovEl.textContent = `FOV ${tel.fovDeg}°`;

    if (avataFreeBtn) {
      avataFreeBtn.textContent = tel.autoFreestyle ? 'Auto Pro-Pilot' : 'Terbang Bebas';
      avataFreeBtn.classList.toggle('active', tel.autoFreestyle);
      avataFreeBtn.classList.toggle('gold-active', !tel.autoFreestyle);
    }
    if (avataProfileBtn) {
      avataProfileBtn.textContent = tel.profile === 'ACRO' ? 'ACRO+' : 'SPORT';
      avataProfileBtn.classList.toggle('gold-active', tel.profile === 'ACRO');
    }
    if (avataMouseLockBtn) {
      const locked = document.pointerLockElement === renderer.domElement;
      avataMouseLockBtn.textContent = locked ? 'Esc Lepas Mouse' : 'Kunci Mouse';
      avataMouseLockBtn.classList.toggle('active', locked);
    }
    if (avataZoneBtns.length > 0) {
      const activeZoneKey = tel.autoFreestyle
        ? resolveActiveAvataZoneKey(camDirector._avataActiveManeuverId, tel.zoneLabel)
        : '';
      avataZoneBtns.forEach((b) => {
        b.classList.toggle('active', b.dataset.avataZone !== 'RANDOM' && b.dataset.avataZone === activeZoneKey);
      });
    }
  }

  if (avataFreeBtn) {
    avataFreeBtn.addEventListener('click', () => {
      camDirector.setAvataAutoFreestyle(!camDirector.avataAutoFreestyle);
      audioSynth.playUiChime(580);
      syncAvataHudDOM();
    });
  }
  if (avataProfileBtn) {
    avataProfileBtn.addEventListener('click', () => {
      camDirector.setAvataFlightProfile(camDirector.avataFlightProfile === 'SPORT' ? 'ACRO' : 'SPORT');
      audioSynth.playUiChime(620);
      syncAvataHudDOM();
    });
  }
  if (avataRollLBtn) {
    avataRollLBtn.addEventListener('click', () => camDirector.triggerAvataStunt('ROLL_LEFT'));
  }
  if (avataRollRBtn) {
    avataRollRBtn.addEventListener('click', () => camDirector.triggerAvataStunt('ROLL_RIGHT'));
  }
  if (avataSplitSBtn) {
    avataSplitSBtn.addEventListener('click', () => camDirector.triggerAvataStunt('SPLIT_S'));
  }
  if (avataMouseLockBtn) {
    avataMouseLockBtn.addEventListener('click', () => {
      if (document.pointerLockElement === renderer.domElement) {
        document.exitPointerLock?.();
      } else {
        setPanelExpanded(false);
        renderer.domElement.requestPointerLock?.();
      }
      setTimeout(() => syncAvataHudDOM(), 60);
    });
  }
  document.addEventListener('pointerlockchange', () => syncAvataHudDOM());
  if (avataOsdToggleBtn) {
    avataOsdToggleBtn.addEventListener('click', () => {
      avataOsdMinimal = !avataOsdMinimal;
      avataHudEl?.classList.toggle('osd-minimal', avataOsdMinimal);
      avataOsdToggleBtn.classList.toggle('active', avataOsdMinimal);
    });
  }
  avataZoneBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      dismissWelcomeToast();
      arrivalActive = false;
      camDirector.jumpAvataToZone(btn.dataset.avataZone || 'RANDOM');
      audioSynth.playUiChime(660);
      syncFPSStatusDOM();
      requestAnimationFrame(syncViewport);
    });
  });

  function bindAvataVirtualStick(zoneId, nubId, onAxes) {
    const zone = document.getElementById(zoneId);
    const nub = document.getElementById(nubId);
    if (!zone || !nub) return;
    let activeId = null;
    const maxR = 34;
    const updateFromEvent = (e) => {
      const r = zone.getBoundingClientRect();
      const cx = r.left + r.width * 0.5;
      const cy = r.top + r.height * 0.5;
      let dx = e.clientX - cx;
      let dy = e.clientY - cy;
      const dist = Math.hypot(dx, dy);
      if (dist > maxR) {
        dx = (dx / dist) * maxR;
        dy = (dy / dist) * maxR;
      }
      nub.style.transform = `translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px)`;
      onAxes(dx / maxR, -dy / maxR);
    };
    const release = (e) => {
      if (activeId === null || (e && e.pointerId !== activeId)) return;
      activeId = null;
      nub.style.transform = 'translate(0px, 0px)';
      onAxes(0, 0);
    };
    zone.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
      activeId = e.pointerId;
      zone.setPointerCapture?.(e.pointerId);
      updateFromEvent(e);
    });
    zone.addEventListener('pointermove', (e) => {
      if (e.pointerId !== activeId) return;
      e.stopPropagation();
      updateFromEvent(e);
    });
    zone.addEventListener('pointerup', release);
    zone.addEventListener('pointercancel', release);
  }

  bindAvataVirtualStick('avata-stick-left', 'avata-nub-left', (nx, ny) => {
    camDirector.setAvataStickInput({ moveX: nx, moveY: ny });
  });
  bindAvataVirtualStick('avata-stick-right', 'avata-nub-right', (nx, ny) => {
    camDirector.setAvataStickInput({ lookX: nx, lookY: ny });
  });

  function bindAvataHoldPad(btnId, onHold) {
    const el = document.getElementById(btnId);
    if (!el) return;
    const end = () => {
      el.classList.remove('pressed');
      onHold(false);
    };
    el.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
      el.setPointerCapture?.(e.pointerId);
      el.classList.add('pressed');
      onHold(true);
    });
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('lostpointercapture', end);
  }
  bindAvataHoldPad('avata-pad-boost', (down) => camDirector.setAvataStickInput({ boost: down }));
  bindAvataHoldPad('avata-pad-up', (down) => camDirector.setAvataStickInput({ climbY: down ? 1 : 0 }));
  bindAvataHoldPad('avata-pad-down', (down) => camDirector.setAvataStickInput({ climbY: down ? -1 : 0 }));

  // Mobile/Tablet Virtual Thumbstick for WALK_SIVITAS & Manual DRONE_6DOF
  const exploreStickLabelEl = document.getElementById('explore-stick-label');
  const explorePadUpEl = document.getElementById('explore-pad-up');
  const explorePadDownEl = document.getElementById('explore-pad-down');
  bindAvataVirtualStick('explore-stick-left', 'explore-nub-left', (nx, ny) => {
    camDirector.setExploreStickInput({ moveX: nx, moveY: ny });
  });
  bindAvataHoldPad('explore-pad-boost', (down) => camDirector.setExploreStickInput({ boost: down }));
  bindAvataHoldPad('explore-pad-up', (down) => camDirector.setExploreStickInput({ climbY: down ? 1 : 0 }));
  bindAvataHoldPad('explore-pad-down', (down) => camDirector.setExploreStickInput({ climbY: down ? -1 : 0 }));

  // --- QUICK-EXPERIENCE DOCK, FLOATING CONTEXT MINI-BAR, & UNIFIED MODE SWITCHER ---
  const FPS_ROLE_ORDER = [
    'FPS_BUS_DRIVER',
    'FPS_JOGGER',
    'FPS_GOLF_DRIVER',
    'FPS_PILOT',
    'FPS_CHAT_NPC',
    'FPS_CAT'
  ];
  let lastFpsRoleIdx = 0;

  const qdockModeBtns = Array.from(document.querySelectorAll('[data-qmode]'));
  const drawerCamModeBtns = Array.from(document.querySelectorAll('[data-cammode]'));
  const ctxTitleEl = document.getElementById('ctx-bar-title');
  const ctxPrevBtn = document.getElementById('ctx-btn-prev');
  const ctxNextBtn = document.getElementById('ctx-btn-next');
  const ctxRoleBtn = document.getElementById('ctx-btn-role');
  const ctxResetBtn = document.getElementById('ctx-btn-reset');
  const ctxResumeTourBtn = document.getElementById('ctx-btn-resume-tour');
  const ctxExitBtn = document.getElementById('ctx-btn-exit');

  function syncQuickDockAndContextDOM() {
    const mode = camDirector.mode;
    const isAvata = camDirector.isAvataFPVMode();
    const isFps = camDirector.isFPSMode();

    if (quickDockEl) {
      quickDockEl.classList.toggle('hidden', isAvata);
    }

    qdockModeBtns.forEach((btn) => {
      const qm = btn.dataset.qmode;
      const active =
        (qm === 'DRONE_6DOF' && (mode === 'DRONE_6DOF' || mode === 'AUTO_CINEMATIC')) ||
        (qm === 'DRONE_AVATA2_FPV' && isAvata) ||
        (qm === 'ORBIT' && mode === 'ORBIT') ||
        (qm === 'WALK_SIVITAS' && mode === 'WALK_SIVITAS') ||
        (qm === 'FPS_CYCLE' && isFps);
      btn.classList.toggle('active', active);
    });

    drawerCamModeBtns.forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.cammode === mode);
    });

    const isMavicManual = Boolean(mode === 'DRONE_6DOF' && camDirector._droneManualEngaged);
    const isMavicZoomed = Boolean(
      (mode === 'DRONE_6DOF' || mode === 'AUTO_CINEMATIC') &&
        Math.abs((Number(camDirector._mavicCinemaFov) || 43.0) - 43.0) > 0.6
    );

    if (contextBarEl) {
      const showCtx = !isPanelExpanded && !isAvata && (isFps || mode === 'WALK_SIVITAS' || isMavicManual || isMavicZoomed);
      contextBarEl.classList.toggle('active', showCtx);
      if (showCtx) {
        if (isFps) {
          const st = camDirector.lastFPSStatus || sivitasAI.getFPSAnchor(mode, camDirector.fpsSubjectOffset);
          if (ctxTitleEl) ctxTitleEl.textContent = st?.label || mode;
          if (ctxPrevBtn) ctxPrevBtn.style.display = '';
          if (ctxNextBtn) ctxNextBtn.style.display = '';
          if (ctxRoleBtn) ctxRoleBtn.style.display = '';
          if (ctxResetBtn) ctxResetBtn.style.display = '';
          if (ctxResumeTourBtn) ctxResumeTourBtn.style.display = 'none';
          if (ctxExitBtn) ctxExitBtn.style.display = '';
        } else if (mode === 'WALK_SIVITAS') {
          if (ctxTitleEl) {
            ctxTitleEl.textContent = useTouchControls()
              ? 'Jalan Sivitas · Stik Kiri & Geser'
              : 'Jalan Sivitas · WASD + Shift';
          }
          if (ctxPrevBtn) ctxPrevBtn.style.display = 'none';
          if (ctxNextBtn) ctxNextBtn.style.display = 'none';
          if (ctxRoleBtn) ctxRoleBtn.style.display = 'none';
          if (ctxResetBtn) ctxResetBtn.style.display = 'none';
          if (ctxResumeTourBtn) ctxResumeTourBtn.style.display = 'none';
          if (ctxExitBtn) ctxExitBtn.style.display = '';
        } else {
          const fovRound = Math.round(Number(camDirector._mavicCinemaFov) || 43);
          if (ctxTitleEl) {
            ctxTitleEl.textContent = isMavicManual
              ? `Mavic Manual · FOV ${fovRound}°`
              : `Zoom Optik Mavic · FOV ${fovRound}°`;
          }
          if (ctxPrevBtn) ctxPrevBtn.style.display = 'none';
          if (ctxNextBtn) ctxNextBtn.style.display = 'none';
          if (ctxRoleBtn) ctxRoleBtn.style.display = 'none';
          if (ctxResetBtn) ctxResetBtn.style.display = isMavicZoomed ? '' : 'none';
          if (ctxResumeTourBtn) ctxResumeTourBtn.style.display = isMavicManual ? '' : 'none';
          if (ctxExitBtn) ctxExitBtn.style.display = 'none';
        }
      }
    }

    if (exploreTouchPadEl) {
      const showTouchWalkOrDrone =
        useTouchControls() && !isPanelExpanded && !isAvata && (mode === 'WALK_SIVITAS' || isMavicManual);
      exploreTouchPadEl.classList.toggle('active', showTouchWalkOrDrone);
      exploreTouchPadEl.setAttribute('aria-hidden', String(!showTouchWalkOrDrone));
      if (showTouchWalkOrDrone) {
        const isWalk = mode === 'WALK_SIVITAS';
        if (exploreStickLabelEl) exploreStickLabelEl.textContent = isWalk ? 'Jalan / Geser' : 'Terbang / Geser';
        if (explorePadUpEl) explorePadUpEl.style.display = isWalk ? 'none' : '';
        if (explorePadDownEl) explorePadDownEl.style.display = isWalk ? 'none' : '';
      }
    }
  }

  function activateModeFromUI(targetMode, opts = {}) {
    dismissWelcomeToast();
    arrivalActive = false;
    if (targetMode !== 'ORBIT') {
      document.querySelectorAll('[data-preset]').forEach((b) => b.classList.remove('active'));
    }
    if (
      (targetMode === 'DRONE_6DOF' ||
        targetMode === 'DRONE_AVATA2_FPV' ||
        targetMode === 'WALK_SIVITAS' ||
        camDirector.isFPSMode(targetMode)) &&
      videotron.interactiveScreenClick
    ) {
      videotron.setInteractiveScreenClick(false, renderer.domElement);
    }

    if (camDirector.isFPSMode(targetMode)) {
      const idx = FPS_ROLE_ORDER.indexOf(targetMode);
      if (idx >= 0) lastFpsRoleIdx = idx;
    }

    const prevCamMode = camDirector.mode;
    if (camDirector.mode === targetMode && camDirector.isFPSMode(targetMode) && !opts.forceReset) {
      camDirector.cycleFPSSubject(1);
    } else if (camDirector.mode === 'DRONE_AVATA2_FPV' && targetMode === 'DRONE_AVATA2_FPV' && !opts.fromQuickDock) {
      camDirector.setAvataAutoFreestyle(!camDirector.avataAutoFreestyle);
    } else if (targetMode === 'DRONE_6DOF') {
      camDirector.setMode('DRONE_6DOF', { startIslandTour: true, ...opts });
    } else {
      camDirector.setMode(targetMode, opts);
    }

    if (prevCamMode !== camDirector.mode) {
      forensicRecorder.onCameraSwitch(prevCamMode, camDirector.mode);
      window.NexusForensic?.logBreadcrumb('camera_switch', `${prevCamMode} -> ${camDirector.mode}`);
    }
    hwGovernor.logEvent('CAM', camDirector.mode);
    audioSynth.playUiChime(590);
    syncFPSStatusDOM();
    requestAnimationFrame(syncViewport);
  }

  qdockModeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const qm = btn.dataset.qmode;
      if (qm === 'FPS_CYCLE') {
        if (camDirector.isFPSMode()) {
          lastFpsRoleIdx = (lastFpsRoleIdx + 1) % FPS_ROLE_ORDER.length;
          activateModeFromUI(FPS_ROLE_ORDER[lastFpsRoleIdx], { forceReset: true });
        } else {
          activateModeFromUI(FPS_ROLE_ORDER[lastFpsRoleIdx], { forceReset: true });
        }
        return;
      }
      if (qm === 'DRONE_AVATA2_FPV' && camDirector.isAvataFPVMode()) {
        camDirector.jumpAvataToZone('RANDOM');
        audioSynth.playUiChime(650);
        syncFPSStatusDOM();
        return;
      }
      activateModeFromUI(qm, { fromQuickDock: true });
    });
  });

  if (avataExitBtn) {
    avataExitBtn.addEventListener('click', () => activateModeFromUI('DRONE_6DOF', { startIslandTour: true }));
  }
  if (ctxExitBtn) {
    ctxExitBtn.addEventListener('click', () => activateModeFromUI('DRONE_6DOF', { startIslandTour: true }));
  }
  if (ctxPrevBtn) {
    ctxPrevBtn.addEventListener('click', () => {
      camDirector.cycleFPSSubject(-1);
      audioSynth.playUiChime(540);
      syncFPSStatusDOM();
    });
  }
  if (ctxNextBtn) {
    ctxNextBtn.addEventListener('click', () => {
      camDirector.cycleFPSSubject(1);
      audioSynth.playUiChime(610);
      syncFPSStatusDOM();
    });
  }
  if (ctxRoleBtn) {
    ctxRoleBtn.addEventListener('click', () => {
      lastFpsRoleIdx = (lastFpsRoleIdx + 1) % FPS_ROLE_ORDER.length;
      activateModeFromUI(FPS_ROLE_ORDER[lastFpsRoleIdx], { forceReset: true });
    });
  }
  if (ctxResetBtn) {
    ctxResetBtn.addEventListener('click', () => {
      if (camDirector.isFPSMode()) {
        camDirector.resetFPSHeadLook();
      } else {
        camDirector.resetMavicGimbalAndZoom();
        syncViewport();
      }
      audioSynth.playUiChime(560);
      syncFPSStatusDOM();
    });
  }
  if (ctxResumeTourBtn) {
    ctxResumeTourBtn.addEventListener('click', () => {
      camDirector.resumeMavicIslandTour();
      audioSynth.playUiChime(630);
      syncFPSStatusDOM();
      requestAnimationFrame(syncViewport);
    });
  }

  // Quick Dock Utility Actions: Clean Cinema Mode, Fullscreen, & Shortcuts Help Popover
  const qdockCleanBtn = document.getElementById('btn-qdock-clean');
  const restoreUiBtn = document.getElementById('btn-restore-ui');
  const qdockFsBtn = document.getElementById('btn-qdock-fullscreen');
  const qdockHelpBtn = document.getElementById('btn-qdock-help');
  const helpCloseBtn = document.getElementById('btn-help-close');

  function toggleCleanCinemaUI(forceState) {
    const next =
      typeof forceState === 'boolean' ? forceState : !document.body.classList.contains('clean-cinema-ui');
    document.body.classList.toggle('clean-cinema-ui', next);
    if (next) helpPopoverEl?.classList.remove('active');
    audioSynth.playUiChime(next ? 520 : 620);
  }
  if (qdockCleanBtn) qdockCleanBtn.addEventListener('click', () => toggleCleanCinemaUI(true));
  if (restoreUiBtn) restoreUiBtn.addEventListener('click', () => toggleCleanCinemaUI(false));
  if (qdockFsBtn) {
    qdockFsBtn.addEventListener('click', () => {
      if (document.fullscreenElement) {
        document.exitFullscreen?.();
      } else {
        document.documentElement.requestFullscreen?.().catch(() => {});
      }
    });
  }
  if (qdockHelpBtn && helpPopoverEl) {
    qdockHelpBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      dismissWelcomeToast();
      const open = !helpPopoverEl.classList.contains('active');
      helpPopoverEl.classList.toggle('active', open);
      qdockHelpBtn.classList.toggle('active', open);
    });
  }
  if (helpCloseBtn && helpPopoverEl) {
    helpCloseBtn.addEventListener('click', () => {
      helpPopoverEl.classList.remove('active');
      qdockHelpBtn?.classList.remove('active');
    });
  }
  document.addEventListener('pointerdown', (e) => {
    if (
      helpPopoverEl &&
      helpPopoverEl.classList.contains('active') &&
      !helpPopoverEl.contains(e.target) &&
      e.target !== qdockHelpBtn
    ) {
      helpPopoverEl.classList.remove('active');
      qdockHelpBtn?.classList.remove('active');
    }
  });

  // Live FPS Telemetry Status Card & Subject Cycle Controls (+ 6-Chip Cockpit/Bridge Quick-Picker)
  const fpsLiveCardEl = document.getElementById('fps-live-card');
  const fpsLiveTitleEl = document.getElementById('fps-live-title');
  const fpsLiveSubEl = document.getElementById('fps-live-sub');
  const fpsPilotPickerEl = document.getElementById('fps-pilot-picker');
  const fpsPilotChipEls = Array.from(document.querySelectorAll('[data-pilot-slot]'));
  const fpsPrevBtn = document.getElementById('btn-fps-prev-subject');
  const fpsNextBtn = document.getElementById('btn-fps-next-subject');
  const fpsResetBtn = document.getElementById('btn-fps-reset-look');

  function syncFPSStatusDOM() {
    syncAvataHudDOM();
    syncQuickDockAndContextDOM();
    if (!fpsLiveCardEl) return;
    const active = camDirector.isFPSMode();
    fpsLiveCardEl.style.display = active ? 'flex' : 'none';
    if (!active) {
      if (fpsPilotPickerEl) fpsPilotPickerEl.style.display = 'none';
      return;
    }
    const isPilotMode = camDirector.mode === 'FPS_PILOT';
    if (fpsPilotPickerEl) {
      fpsPilotPickerEl.style.display = isPilotMode ? 'grid' : 'none';
    }
    if (isPilotMode && fpsPilotChipEls.length > 0) {
      const activeSlot = ((camDirector.fpsSubjectOffset % 6) + 6) % 6;
      for (let i = 0; i < fpsPilotChipEls.length; i++) {
        const chip = fpsPilotChipEls[i];
        const slot = Number(chip.getAttribute('data-pilot-slot'));
        chip.classList.toggle('active', slot === activeSlot);
      }
    }
    const st = camDirector.lastFPSStatus || sivitasAI.getFPSAnchor(camDirector.mode, camDirector.fpsSubjectOffset);
    if (st) {
      if (fpsLiveTitleEl) fpsLiveTitleEl.textContent = st.label || camDirector.mode;
      if (fpsLiveSubEl) fpsLiveSubEl.textContent = st.subLabel || 'Geser layar untuk menoleh';
    }
  }
  syncFPSStatusDOM();

  for (let i = 0; i < fpsPilotChipEls.length; i++) {
    const chip = fpsPilotChipEls[i];
    chip.addEventListener('click', () => {
      const slot = Number(chip.getAttribute('data-pilot-slot')) || 0;
      const prevCam = camDirector.mode;
      if (camDirector.mode !== 'FPS_PILOT') {
        camDirector.setMode('FPS_PILOT', { subjectOffset: slot });
        forensicRecorder.onCameraSwitch(prevCam, 'FPS_PILOT');
      } else {
        camDirector.setFPSSubjectOffset(slot);
      }
      window.NexusForensic?.logBreadcrumb('pilot_slot', `Slot ${slot} (${chip.textContent || ''})`);
      syncFPSStatusDOM();
    });
  }

  if (fpsPrevBtn) {
    fpsPrevBtn.addEventListener('click', () => {
      camDirector.cycleFPSSubject(-1);
      syncFPSStatusDOM();
    });
  }
  if (fpsNextBtn) {
    fpsNextBtn.addEventListener('click', () => {
      camDirector.cycleFPSSubject(1);
      syncFPSStatusDOM();
    });
  }
  if (fpsResetBtn) {
    fpsResetBtn.addEventListener('click', () => {
      camDirector.resetFPSHeadLook();
    });
  }

  // Spot Category Filter Pills (All / Ikonik / Pulau U / Pulau I & Laut)
  const presetBtns = Array.from(document.querySelectorAll('[data-preset]'));
  const presetFilterBtns = Array.from(document.querySelectorAll('[data-pfilter]'));
  const presetCountPillEl = document.getElementById('preset-count-pill');
  presetFilterBtns.forEach((fbtn) => {
    fbtn.addEventListener('click', () => {
      const cat = fbtn.dataset.pfilter || 'ALL';
      presetFilterBtns.forEach((b) => b.classList.toggle('active', b === fbtn));
      let shown = 0;
      presetBtns.forEach((pbtn) => {
        const cats = (pbtn.dataset.pcat || '').split(/\s+/).filter(Boolean);
        const match = cat === 'ALL' || cats.includes(cat);
        pbtn.classList.toggle('preset-filtered-out', !match);
        if (match) shown++;
      });
      if (presetCountPillEl) presetCountPillEl.textContent = `${shown} Spot`;
    });
  });

  // Tri-Angle & Live Patrol Camera Presets
  presetBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      dismissWelcomeToast();
      arrivalActive = false;
      const prevCam = camDirector.mode;
      presetBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const notePresetSwitch = () => {
        if (prevCam !== camDirector.mode) {
          forensicRecorder.onCameraSwitch(prevCam, camDirector.mode);
        }
        window.NexusForensic?.logBreadcrumb('spot_preset', btn.dataset.preset);
      };
      if (btn.dataset.preset === 'mobilGolfUI') {
        const cartPose = sivitasAI.getGolfCartPose();
        if (cartPose) {
          const sy = Math.sin(cartPose.yaw);
          const cy = Math.cos(cartPose.yaw);
          const liveCartPreset = {
            pos: {
              x: cartPose.worldX + sy * 5.8 + cy * 4.5,
              y: cartPose.worldY + 2.45,
              z: cartPose.worldZ + cy * 5.8 - sy * 4.5
            },
            target: {
              x: cartPose.worldX,
              y: cartPose.worldY + 1.08,
              z: cartPose.worldZ
            }
          };
          camDirector.applyOrthoPreset('mobilGolfUI', { ...presets, mobilGolfUI: liveCartPreset });
          notePresetSwitch();
          syncFPSStatusDOM();
          requestAnimationFrame(syncViewport);
          return;
        }
      }
      if (btn.dataset.preset === 'taxiHeliListrik') {
        const heliPose = sivitasAI.getHeliTaxiPose();
        if (heliPose) {
          const sy = Math.sin(heliPose.yaw);
          const cy = Math.cos(heliPose.yaw);
          const liveHeliPreset = {
            pos: {
              x: heliPose.worldX + sy * 10.5 - cy * 8.2,
              y: heliPose.worldY + 4.2,
              z: heliPose.worldZ + cy * 10.5 + sy * 8.2
            },
            target: {
              x: heliPose.worldX,
              y: heliPose.worldY + 1.45,
              z: heliPose.worldZ
            }
          };
          camDirector.applyOrthoPreset('taxiHeliListrik', { ...presets, taxiHeliListrik: liveHeliPreset });
          notePresetSwitch();
          syncFPSStatusDOM();
          requestAnimationFrame(syncViewport);
          return;
        }
      }
      if (btn.dataset.preset === 'underpassBawahLautInterior' && camera.position.y > -2.0) {
        camDirector.applyOrthoPreset(btn.dataset.preset, presets, { immediate: true });
        notePresetSwitch();
        syncFPSStatusDOM();
        requestAnimationFrame(syncViewport);
        return;
      }
      camDirector.applyOrthoPreset(btn.dataset.preset, presets);
      notePresetSwitch();
      syncFPSStatusDOM();
      requestAnimationFrame(syncViewport);
    });
  });

  // Exploration, DJI Avata 2 Action FPV, & 6 Living FPS Camera Modes (Drawer Buttons)
  drawerCamModeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      activateModeFromUI(btn.dataset.cammode);
    });
  });

  // --- OUTDOOR CAMPUS LED VIDEOTRON YOUTUBE / PLAYLIST / 2-LAYER SHORTS CONTROLS ---
  const vtBadgeEl = document.getElementById('vt-detected-badge');
  const vtTitleEl = document.getElementById('vt-active-title');
  const vtFocusCamBtn = document.getElementById('btn-vt-focus-cam');
  const vtUrlInput = document.getElementById('vt-url-input');
  const vtApplyBtn = document.getElementById('btn-vt-apply');
  const vtDefaultBtn = document.getElementById('btn-vt-default');
  const vtStatusMsgEl = document.getElementById('vt-status-msg');
  const vtModeBtns = Array.from(document.querySelectorAll('[data-vtmode]'));
  const vtPlayPauseBtn = document.getElementById('btn-vt-playpause');
  const vtMuteBtn = document.getElementById('btn-vt-mute');
  const vtVolSlider = document.getElementById('slider-vt-vol');
  const vtVolLabel = document.getElementById('label-vt-vol');
  const vtPrevBtn = document.getElementById('btn-vt-prev');
  const vtNextBtn = document.getElementById('btn-vt-next');
  const vtInteractBtn = document.getElementById('btn-vt-interact');

  function syncVideotronUI(state, errMsg = null) {
    if (!state) return;
    if (vtBadgeEl) {
      vtBadgeEl.textContent = (state.badgeLabelId || 'SINGLE 16:9').toUpperCase();
    }
    if (vtTitleEl) {
      vtTitleEl.textContent = state.title || 'Videotron Kampus UI';
    }
    vtModeBtns.forEach((b) => {
      b.classList.toggle('active', b.dataset.vtmode === state.formatMode);
    });
    if (vtPlayPauseBtn) {
      const span = vtPlayPauseBtn.querySelector('.btn-title') || vtPlayPauseBtn;
      span.textContent = state.isPlaying ? 'Jeda Video' : 'Putar Video';
      vtPlayPauseBtn.classList.toggle('active', !state.isPlaying);
    }
    if (vtMuteBtn) {
      const span = vtMuteBtn.querySelector('.btn-title') || vtMuteBtn;
      span.textContent = state.isMuted ? 'Suara YT: MUTE' : 'Suara YT: ON';
      vtMuteBtn.classList.toggle('active', !state.isMuted);
    }
    if (vtVolLabel) {
      vtVolLabel.textContent = state.isMuted ? 'MUTE' : `${state.volume}%`;
    }
    if (vtVolSlider && document.activeElement !== vtVolSlider) {
      vtVolSlider.value = String(state.volume);
    }
    if (vtInteractBtn) {
      const span = vtInteractBtn.querySelector('.btn-title') || vtInteractBtn;
      span.textContent = state.interactiveScreenClick ? 'Klik 3D: ON' : 'Klik 3D: OFF';
      vtInteractBtn.classList.toggle('active', state.interactiveScreenClick);
    }
    if (vtStatusMsgEl) {
      if (errMsg) {
        vtStatusMsgEl.classList.add('is-error');
        vtStatusMsgEl.textContent = errMsg;
      } else {
        vtStatusMsgEl.classList.remove('is-error');
        const fileNote =
          window.location.protocol === 'file:'
            ? ' (Tips: jalankan via HTTP/HTTPS/localhost agar YouTube tidak terkena blokir Referer file://).'
            : '';
        if (state.activeLayout === 'SHORT_2LAYER_BLUR') {
          vtStatusMsgEl.textContent = 'Short terdeteksi: mode 2-lapis (belakang full-width blur) aktif.' + fileNote;
        } else if (state.isPlaylist) {
          vtStatusMsgEl.textContent = 'Playlist YouTube aktif pada layar Videotron 3D.' + fileNote;
        } else {
          vtStatusMsgEl.textContent = 'Video YouTube 16:9 aktif pada layar Videotron 3D.' + fileNote;
        }
      }
    }
  }

  syncVideotronUIHook = syncVideotronUI;
  syncVideotronUI(videotron.getState());

  if (vtFocusCamBtn) {
    vtFocusCamBtn.addEventListener('click', () => {
      dismissWelcomeToast();
      arrivalActive = false;
      document.querySelectorAll('[data-preset]').forEach((b) =>
        b.classList.toggle('active', b.dataset.preset === 'videotronUI')
      );
      camDirector.applyOrthoPreset('videotronUI', presets);
      syncFPSStatusDOM();
      requestAnimationFrame(syncViewport);
    });
  }

  async function applyVideotronInput() {
    if (!vtUrlInput) return;
    const res = await videotron.loadSource(vtUrlInput.value);
    if (res.ok && res.state && res.state.normalizedUrl && !/<iframe\b/i.test(vtUrlInput.value)) {
      vtUrlInput.value = res.state.normalizedUrl;
    }
  }

  if (vtApplyBtn) {
    vtApplyBtn.addEventListener('click', applyVideotronInput);
  }
  if (vtUrlInput) {
    vtUrlInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        applyVideotronInput();
      }
    });
  }
  if (vtDefaultBtn) {
    vtDefaultBtn.addEventListener('click', async () => {
      if (vtUrlInput) vtUrlInput.value = VideotronYouTubeEngine.DEFAULT_EMBED_URL;
      await videotron.resetToDefault();
    });
  }
  vtModeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      videotron.setFormatMode(btn.dataset.vtmode);
    });
  });
  if (vtPlayPauseBtn) {
    vtPlayPauseBtn.addEventListener('click', () => videotron.togglePlayPause());
  }
  if (vtMuteBtn) {
    vtMuteBtn.addEventListener('click', () => videotron.toggleMute());
  }
  if (vtVolSlider) {
    vtVolSlider.addEventListener('input', () => {
      videotron.setVolume(Number(vtVolSlider.value));
    });
  }
  if (vtPrevBtn) {
    vtPrevBtn.addEventListener('click', () => videotron.previousTrack());
  }
  if (vtNextBtn) {
    vtNextBtn.addEventListener('click', () => videotron.nextTrack());
  }
  if (vtInteractBtn) {
    vtInteractBtn.addEventListener('click', () => {
      videotron.setInteractiveScreenClick(!videotron.interactiveScreenClick, renderer.domElement);
    });
  }

  // --- LIVE KAMPUS UI DEPOK TELEMETRY & CONTINUOUS SPECTRUM SLIDERS (TIME, WEATHER, SEA STATE 0-2M) ---
  const liveCardEl = document.getElementById('live-depok-card');
  const liveBadgeTextEl = document.getElementById('ui-depok-badge-text');
  const liveClockEl = document.getElementById('ui-depok-clock');
  const liveWeatherEl = document.getElementById('ui-depok-weather');
  const liveHvsEyeEl = document.getElementById('ui-hvs-eye');
  const liveAstroEl = document.getElementById('ui-astro-moon');
  const liveBtnEl = document.getElementById('btn-live-depok');
  const sliderDurEl = document.getElementById('slider-trans-dur');
  const labelDurEl = document.getElementById('label-trans-dur');
  const sliderTimeSpectrumEl = document.getElementById('slider-time-spectrum');
  const sliderWeatherSpectrumEl = document.getElementById('slider-weather-spectrum');
  const sliderSeaStateEl = document.getElementById('slider-sea-state');
  const btnSeaAutoEl = document.getElementById('btn-sea-auto');
  const activeTodPillEl = document.getElementById('ui-active-tod-pill');
  const activeWeatherPillEl = document.getElementById('ui-active-weather-pill');
  const activeSeaPillEl = document.getElementById('ui-active-sea-pill');
  const skipTransBtn = document.getElementById('btn-skip-transition');
  const moodButtons = Array.from(document.querySelectorAll('[data-mood-tod]'));
  const todButtons = Array.from(document.querySelectorAll('[data-tod]'));
  const weatherButtons = Array.from(document.querySelectorAll('[data-weather]'));

  function updateDepokTelemetryDOM() {
    if (liveClockEl) {
      if (isLiveDepok) {
        liveClockEl.textContent = `${latestDepokClock.formattedHMS} · ${latestDepokClock.periodLabelId}`;
      } else {
        const simClock = DepokLiveWeatherService.formatSolarHourClock(weather.currentSolarHour);
        liveClockEl.textContent = `Simulasi ${simClock.formattedHM} · ${simClock.periodLabelId}`;
      }
    }
    if (liveWeatherEl && latestDepokWeather) {
      if (isLiveDepok) {
        const srcTag = latestDepokWeather.isLiveApi ? 'Open-Meteo Depok' : 'Klimatologi Depok';
        liveWeatherEl.textContent =
          `${latestDepokWeather.labelId.replace(' (Estimasi Lokal)', '')} · ` +
          `${latestDepokWeather.temperatureC}°C · RH ${latestDepokWeather.humidityPct}% · ` +
          `Angin ${latestDepokWeather.windSpeedKmh} km/j (${srcTag})`;
      } else {
        const spec = WeatherOpticsEngine.evaluateWeatherSpectrum(weather.getWeatherSpectrum());
        const seaH = weather.getSeaWaveHeightMeters();
        liveWeatherEl.textContent =
          `Simulasi: ${spec.labelId} (${spec.percent}%) · Ombak ${seaH.toFixed(2)}m · ` +
          `Rill Depok: ${latestDepokClock.formattedHM} (${latestDepokWeather.temperatureC}°C)`;
      }
    }
  }
  updateDepokTelemetryDOM();

  // Continuous 24-Hour Solar/Lunar Time Spectrum Slider (00:00 – 24:00 WIB, Proportional 45s End-to-End Transition)
  if (sliderTimeSpectrumEl) {
    sliderTimeSpectrumEl.addEventListener('input', () => {
      dismissWelcomeToast();
      isLiveDepok = false;
      weather.setContinuousTimeHours(Number(sliderTimeSpectrumEl.value), sharedUniforms, {
        immediate: false,
        proportional45s: true
      });
      syncTransitionButtonsUI();
    });
    sliderTimeSpectrumEl.addEventListener('change', () => {
      const tgtH = weather.getTargetSliderSolarHour();
      hwGovernor.logEvent(
        'WAKTU',
        `Target ${WeatherOpticsEngine.formatSolarHourWIB(tgtH)} WIB`
      );
    });
  }

  // Continuous Physical Weather Spectrum Slider (0% Cerah Murni -> 100% Badai Petir, Proportional 45s End-to-End Transition)
  if (sliderWeatherSpectrumEl) {
    sliderWeatherSpectrumEl.addEventListener('input', () => {
      dismissWelcomeToast();
      isLiveDepok = false;
      const u01 = Math.max(0.0, Math.min(1.0, Number(sliderWeatherSpectrumEl.value) / 100.0));
      weather.setWeatherSpectrum(u01, sharedUniforms, {
        immediate: false,
        proportional45s: true
      });
      syncTransitionButtonsUI();
    });
    sliderWeatherSpectrumEl.addEventListener('change', () => {
      const spec = WeatherOpticsEngine.evaluateWeatherSpectrum(weather.getTargetWeatherSpectrum());
      hwGovernor.logEvent('CUACA', `Target ${spec.labelId} (${spec.percent}%)`);
      forensicRecorder.recordSubstantiveEvent('state_change', `Cuaca → ${spec.labelId} (${spec.percent}%)`);
    });
  }

  // Continuous Sea State & Significant Wave Height Slider (0.00 m Laut Tenang -> 3.00 m Gelora Besar, Proportional 45s End-to-End Transition)
  if (sliderSeaStateEl) {
    sliderSeaStateEl.addEventListener('input', () => {
      dismissWelcomeToast();
      const waveM = Math.max(0.0, Math.min(3.0, Number(sliderSeaStateEl.value)));
      weather.setSeaWaveHeightMeters(waveM, {
        immediate: false,
        proportional45s: true
      });
      syncTransitionButtonsUI();
    });
    sliderSeaStateEl.addEventListener('change', () => {
      const hTgt = weather.getTargetSeaWaveHeightMeters();
      hwGovernor.logEvent(
        'LAUT',
        `Target ${WeatherOpticsEngine.describeSeaStateLabel(hTgt)} (${hTgt.toFixed(2)} m)`
      );
      forensicRecorder.recordSubstantiveEvent('state_change', `Ombak → ${hTgt.toFixed(2)} m`);
    });
  }

  if (btnSeaAutoEl) {
    btnSeaAutoEl.addEventListener('click', () => {
      dismissWelcomeToast();
      weather.setSeaWaveHeightMeters(null, {
        auto: true,
        immediate: false,
        proportional45s: true
      });
      audioSynth.playUiChime(600);
      hwGovernor.logEvent('LAUT', 'Auto mengikuti cuaca');
      syncTransitionButtonsUI();
    });
  }

  // Legacy Transition Duration Slider (if present)
  if (sliderDurEl) {
    sliderDurEl.addEventListener('input', () => {
      const clamped = weather.setTransitionDuration(Number(sliderDurEl.value));
      if (labelDurEl) labelDurEl.textContent = `${Math.round(clamped)} dtk`;
    });
  }

  todButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      isLiveDepok = false;
      weather.setTimeOfDay(btn.dataset.tod, sharedUniforms);
      syncTransitionButtonsUI();
    });
  });

  weatherButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      isLiveDepok = false;
      weather.setWeather(btn.dataset.weather);
      audioSynth.playUiChime(560);
      syncTransitionButtonsUI();
    });
  });

  moodButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      isLiveDepok = false;
      if (btn.dataset.moodTod) weather.setTimeOfDay(btn.dataset.moodTod, sharedUniforms);
      if (btn.dataset.moodW) weather.setWeather(btn.dataset.moodW);
      audioSynth.playUiChime(620);
      syncTransitionButtonsUI();
    });
  });

  // Optional Instant Transition Skip (Applies Active Time, Weather, & Sea Wave Transitions Immediately)
  if (skipTransBtn) {
    skipTransBtn.addEventListener('click', () => {
      weather.skipActiveTransitions(sharedUniforms);
      sivitasAI.syncOceanState(weather.getSeaWaveHeightMeters(), weather.uniforms.uTime.value);
      audioSynth.playUiChime(700);
      syncTransitionButtonsUI();
    });
  }

  // Return to Live Kampus UI Depok Clock & Weather (Smooth 20–45s Transition!)
  if (liveBtnEl) {
    liveBtnEl.addEventListener('click', () => {
      isLiveDepok = true;
      weather.setSeaWaveHeightMeters(null, {
        auto: true,
        immediate: false,
        proportional45s: true
      });
      latestDepokClock = DepokLiveWeatherService.getDepokWIBClock();
      updateDepokTelemetryDOM();
      weather.setReferenceAstronomyDate(new Date(), sharedUniforms);
      weather.setSolarHour(latestDepokClock.solarHour, sharedUniforms);
      weather.setWeather(latestDepokWeather.weatherMode, {
        cloudCoverNorm: latestDepokWeather.cloudCoverNorm
      });
      hwGovernor.logEvent('ATMOS', 'Sinkron Live Depok');
      syncTransitionButtonsUI();

      // Refresh live Open-Meteo telemetry in background if older than 45 seconds
      if (Date.now() - (latestDepokWeather.fetchedAtMs || 0) > 45000) {
        DepokLiveWeatherService.fetchLiveDepokWeather().then((fresh) => {
          latestDepokWeather = fresh;
          updateDepokTelemetryDOM();
          if (isLiveDepok) {
            weather.setWeather(fresh.weatherMode, {
              cloudCoverNorm: fresh.cloudCoverNorm
            });
          }
        });
      }
    });
  }

  // Periodic 5-Minute Live Depok Weather Auto-Sync
  setInterval(async () => {
    const fresh = await DepokLiveWeatherService.fetchLiveDepokWeather();
    latestDepokWeather = fresh;
    updateDepokTelemetryDOM();
    if (isLiveDepok) {
      weather.setWeather(fresh.weatherMode, {
        cloudCoverNorm: fresh.cloudCoverNorm
      });
    }
  }, DepokLiveWeatherService.REFRESH_INTERVAL_MS);

  function renderAxisButtonStates(buttons, attrName, axisState) {
    if (!buttons || buttons.length === 0) return;
    const pct = Math.round(axisState.progress * 100);
    const remSec = Math.max(1, Math.ceil(axisState.remainingSec));
    const targetCode = axisState.toMode || axisState.toTod;
    const fromCode = axisState.fromMode || axisState.fromTod;

    for (const btn of buttons) {
      const code = btn.dataset[attrName];
      const subEl = btn.querySelector('.btn-sub');
      const idleHint = btn.dataset.hint || '';
      const compactSub = btn.dataset.compactSub === '1';
      btn.classList.remove('active', 'is-transitioning', 'is-departing');
      btn.style.setProperty('--trans-pct', '0%');
      if (subEl) subEl.textContent = idleHint;

      if (axisState.active) {
        if (code === targetCode) {
          btn.classList.add('is-transitioning');
          btn.style.setProperty('--trans-pct', `${pct}%`);
          if (subEl) {
            subEl.textContent = compactSub ? `${pct}%` : `${pct}%·${remSec}d`;
          }
        } else if (code === fromCode && code !== targetCode) {
          btn.classList.add('is-departing');
          if (subEl) {
            subEl.textContent = compactSub ? `${Math.max(0, 100 - pct)}%` : `-${pct}%`;
          }
        }
      } else if (code === targetCode) {
        btn.classList.add('active');
      }
    }
  }

  function syncTransitionButtonsUI() {
    const ts = weather.getTransitionState();
    renderAxisButtonStates(weatherButtons, 'weather', ts.weather);
    renderAxisButtonStates(todButtons, 'tod', ts.time);

    // Keep slider thumbs anchored to the user's selected target during proportional transitions (never snap back on blur!)
    const targetSliderHour = ts.time.active ? ts.time.targetSliderHour : weather.getSliderSolarHour();
    if (sliderTimeSpectrumEl && document.activeElement !== sliderTimeSpectrumEl) {
      sliderTimeSpectrumEl.value = targetSliderHour.toFixed(2);
    }

    const liveWeatherSpec01 = ts.weather.active && !ts.weather.isSpectrumSweep
      ? (1.0 - ts.weather.progress) * WeatherOpticsEngine.weatherModeToSpectrum(ts.weather.fromMode) +
        ts.weather.progress * WeatherOpticsEngine.weatherModeToSpectrum(ts.weather.toMode)
      : weather.getWeatherSpectrum();
    const targetWeatherSpec01 = ts.weather.active && !ts.weather.isSpectrumSweep
      ? WeatherOpticsEngine.weatherModeToSpectrum(ts.weather.toMode)
      : weather.getTargetWeatherSpectrum();

    if (sliderWeatherSpectrumEl && document.activeElement !== sliderWeatherSpectrumEl) {
      sliderWeatherSpectrumEl.value = (targetWeatherSpec01 * 100.0).toFixed(1);
    }

    const seaWaveM = weather.getSeaWaveHeightMeters();
    const targetSeaWaveM = weather.isManualSeaState()
      ? weather.getTargetSeaWaveHeightMeters()
      : ts.sea.active
        ? ts.sea.targetWaveM
        : seaWaveM;
    if (sliderSeaStateEl && document.activeElement !== sliderSeaStateEl) {
      sliderSeaStateEl.value = targetSeaWaveM.toFixed(2);
    }
    if (btnSeaAutoEl) {
      btnSeaAutoEl.classList.toggle('active', !weather.isManualSeaState());
    }

    if (activeTodPillEl) {
      const simClk = DepokLiveWeatherService.formatSolarHourClock(weather.currentSolarHour);
      const liveWibHm = WeatherOpticsEngine.formatSolarHourWIB(weather.getSliderSolarHour());
      if (ts.time.active) {
        const tgtWibHm = WeatherOpticsEngine.formatSolarHourWIB(ts.time.targetSliderHour);
        const remT = Math.max(1, Math.ceil(ts.time.remainingSec));
        activeTodPillEl.textContent = `${liveWibHm} \u2192 ${tgtWibHm} · ${remT}d`;
      } else {
        activeTodPillEl.textContent = `${simClk.periodLabelId} · ${liveWibHm}`;
      }
    }

    const specEval = WeatherOpticsEngine.evaluateWeatherSpectrum(liveWeatherSpec01);
    const tgtSpecEval = WeatherOpticsEngine.evaluateWeatherSpectrum(targetWeatherSpec01);
    if (activeWeatherPillEl) {
      if (ts.weather.active) {
        const wLbl = ts.weather.isSpectrumSweep
          ? tgtSpecEval.labelId
          : WEATHER_LABELS[ts.weather.toMode] || tgtSpecEval.labelId;
        const remW = Math.max(1, Math.ceil(ts.weather.remainingSec));
        activeWeatherPillEl.textContent = `${specEval.percent}% \u2192 ${wLbl} · ${remW}d`;
      } else {
        activeWeatherPillEl.textContent = `${specEval.labelId} · ${specEval.percent}%`;
      }
    }

    if (activeSeaPillEl) {
      if (ts.sea.active) {
        const remS = Math.max(1, Math.ceil(ts.sea.remainingSec));
        activeSeaPillEl.textContent = `${seaWaveM.toFixed(2)} \u2192 ${ts.sea.targetWaveM.toFixed(2)}m · ${remS}d`;
      } else {
        activeSeaPillEl.textContent = `${WeatherOpticsEngine.describeSeaStateLabel(seaWaveM)} · ${seaWaveM.toFixed(2)} m`;
      }
    }

    const anyTrans = ts.weather.active || ts.time.active || ts.sea.active;
    if (skipTransBtn) {
      skipTransBtn.classList.toggle('visible', Boolean(anyTrans));
    }

    const activeAxes = [];
    if (ts.weather.active) {
      const wName = ts.weather.isSpectrumSweep
        ? tgtSpecEval.labelId
        : WEATHER_LABELS[ts.weather.toMode] || tgtSpecEval.labelId;
      activeAxes.push({
        pct: Math.round(ts.weather.progress * 100),
        rem: Math.max(1, Math.ceil(ts.weather.remainingSec)),
        rawRem: ts.weather.remainingSec,
        label: wName
      });
    }
    if (ts.time.active) {
      const tName = ts.time.isProportionalSlider
        ? `${WeatherOpticsEngine.formatSolarHourWIB(ts.time.targetSliderHour)} WIB`
        : TOD_LABELS[ts.time.toTod];
      activeAxes.push({
        pct: Math.round(ts.time.progress * 100),
        rem: Math.max(1, Math.ceil(ts.time.remainingSec)),
        rawRem: ts.time.remainingSec,
        label: tName
      });
    }
    if (ts.sea.active) {
      activeAxes.push({
        pct: Math.round(ts.sea.progress * 100),
        rem: Math.max(1, Math.ceil(ts.sea.remainingSec)),
        rawRem: ts.sea.remainingSec,
        label: `Ombak ${ts.sea.targetWaveM.toFixed(2)}m`
      });
    }
    activeAxes.sort((a, b) => b.rawRem - a.rawRem);
    const dominantAxis =
      activeAxes.length > 0
        ? {
            pct: activeAxes[0].pct,
            rem: activeAxes[0].rem,
            targetLabel:
              activeAxes.length > 1
                ? `${activeAxes[0].label} + ${activeAxes[1].label}`
                : activeAxes[0].label
          }
        : null;

    if (liveCardEl && liveBtnEl) {
      liveCardEl.classList.toggle('is-manual', !isLiveDepok);
      const titleEl = liveBtnEl.querySelector('.btn-title');
      const subEl = liveBtnEl.querySelector('.btn-sub');
      liveBtnEl.classList.remove('active', 'is-transitioning');
      liveBtnEl.style.setProperty('--trans-pct', '0%');

      if (isLiveDepok) {
        if (liveBadgeTextEl) {
          liveBadgeTextEl.textContent = latestDepokWeather.isLiveApi
            ? 'LIVE KAMPUS UI DEPOK'
            : 'WAKTU RILL DEPOK (WIB)';
        }
        if (anyTrans && dominantAxis) {
          liveBtnEl.classList.add('is-transitioning');
          liveBtnEl.style.setProperty('--trans-pct', `${dominantAxis.pct}%`);
          if (titleEl) titleEl.textContent = 'Menyelaraskan ke Kondisi Rill Depok...';
          if (subEl) subEl.textContent = `${dominantAxis.pct}% · Sisa ${dominantAxis.rem} dtk`;
        } else {
          liveBtnEl.classList.add('active');
          if (titleEl) titleEl.textContent = '● Sinkron Rill Kampus UI Depok';
          if (subEl) subEl.textContent = 'Jam WIB & Cuaca Langsung Aktif';
        }
      } else {
        if (liveBadgeTextEl) liveBadgeTextEl.textContent = 'MODE SPEKTRUM MANUAL';
        if (titleEl) titleEl.textContent = '↻ Kembali ke Waktu & Cuaca Rill Depok';
        if (subEl) {
          subEl.textContent =
            anyTrans && dominantAxis
              ? `Transisi aktif · Sisa ${dominantAxis.rem} dtk (maks 45 dtk)`
              : 'Transisi proporsional hingga 45 detik';
        }
        updateDepokTelemetryDOM();
      }
    }

    if (pillBtn && pillLabelEl) {
      if (anyTrans && dominantAxis) {
        pillBtn.classList.add('is-transitioning');
        pillBtn.style.setProperty('--pill-pct', `${dominantAxis.pct}%`);
        pillLabelEl.textContent = `Menuju ${dominantAxis.targetLabel} · ${dominantAxis.pct}% (${dominantAxis.rem}d)`;
      } else {
        pillBtn.classList.remove('is-transitioning');
        pillBtn.style.setProperty('--pill-pct', '0%');
        if (isLiveDepok) {
          const cleanWeather = latestDepokWeather.labelId.replace(' (Estimasi Lokal)', '');
          pillLabelEl.textContent = `LIVE Depok · ${latestDepokClock.formattedHM} · ${cleanWeather}`;
        } else {
          const wibHm = WeatherOpticsEngine.formatSolarHourWIB(weather.currentSolarHour);
          pillLabelEl.textContent = `Kepulauan UI · ${wibHm} WIB · ${specEval.labelId}`;
        }
      }
    }

    const hvs = weather.getHumanEyeHDRState();
    const regimeId =
      hvs.visionRegime === 'SCOTOPIC'
        ? 'Skotopik (Sel Batang)'
        : hvs.visionRegime === 'MESOPIC'
          ? 'Mesopik (Adaptif)'
          : 'Fotopik (Sel Kerucut)';
    if (liveHvsEyeEl) {
      liveHvsEyeEl.textContent =
        `HDR Mata Manusia · ${regimeId} · Pupil ${hvs.pupilDiameterMm.toFixed(1)} mm · Gain ${hvs.eyeGain.toFixed(2)}×`;
    }
    if (liveAstroEl) {
      const astro = weather.getAstronomicalState();
      const moonPct = Math.round(astro.moonIlluminatedFraction * 100);
      const lampTxt =
        astro.streetlampSensorLevel > 0.05
          ? `ON ${Math.round(astro.streetlampSensorLevel * 100)}%`
          : 'OFF';
      liveAstroEl.textContent =
        `Bulan: ${astro.moonPhaseLabelId} (${moonPct}%) · Bortle ${astro.bortleClass} · Sensor Lampu: ${lampTxt}`;
    }
    const elHvsAudit = document.getElementById('sim-hvs-eye');
    if (elHvsAudit) {
      elHvsAudit.textContent = `${hvs.visionRegime} · ${hvs.pupilDiameterMm.toFixed(1)} mm`;
    }
  }
  syncTransitionButtonsUI();

  // SSoST Visual Preset & Zebra Inspection Mode (PBR Natural | 🏎️ Forza Horizon 6 | ✨ Genshin Stylized HDR | Zebra)
  const qdockGenshinBtn = document.getElementById('btn-qdock-genshin');
  const inspectButtons = Array.from(document.querySelectorAll('[data-inspect]'));
  let permalinkDirty = false;

  function setVisualInspectPreset(modeKey, playChime = false) {
    const key = String(modeKey);
    const isGenshin = key === 'GENSHIN';
    const isForza = key === 'FORZA';
    const isZebra = key === '1' || key === '2';
    const modeVal = isGenshin ? 1.0 : isForza ? -1.0 : 0.0;
    if (sharedUniforms.uGenshinMode) {
      sharedUniforms.uGenshinMode.value = modeVal;
    }
    if (weather.uniforms && weather.uniforms.uGenshinMode) {
      weather.uniforms.uGenshinMode.value = modeVal;
    }
    sharedUniforms.uZebraInspectMode.value = isZebra ? Number(key) : 0;
    weather._applyContinuousCelestialAndFog(sharedUniforms);

    inspectButtons.forEach((b) => {
      b.classList.toggle('active', b.dataset.inspect === key);
    });
    if (qdockGenshinBtn) {
      qdockGenshinBtn.classList.toggle('active', isGenshin || isForza);
      qdockGenshinBtn.textContent = isForza ? '\uD83C\uDFCE\uFE0F Forza' : '\u2728 Genshin';
    }
    permalinkDirty = true;
    if (playChime) {
      audioSynth.playUiChime(isGenshin ? 680 : isForza ? 610 : 540);
    }
  }

  inspectButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      setVisualInspectPreset(btn.dataset.inspect, true);
    });
  });

  if (qdockGenshinBtn) {
    qdockGenshinBtn.addEventListener('click', () => {
      const curVal = sharedUniforms.uGenshinMode ? sharedUniforms.uGenshinMode.value : 0.0;
      const nextKey = curVal > 0.5 ? 'FORZA' : curVal < -0.5 ? '0' : 'GENSHIN';
      setVisualInspectPreset(nextKey, true);
    });
  }

  // 3D Binaural HRTF & Physical Acoustic Engine (Unified Across Drawer, Quick Dock, & Avata 2 HUD)
  const audioBtn = document.getElementById('btn-audio');
  const qdockAudioBtn = document.getElementById('btn-qdock-audio');
  const avataAudioBtn = document.getElementById('btn-avata-audio');
  const audioVolSlider = document.getElementById('slider-audio-vol');
  const audioVolLabel = document.getElementById('label-audio-vol');
  const acousticStatusEl = document.getElementById('ui-acoustic-status');
  const audioMixBtns = Array.from(document.querySelectorAll('[data-audiomix]'));
  const spatialModeBtn = document.getElementById('btn-spatial-mode');
  const bellAuditionBtn = document.getElementById('btn-bell-audition');
  const thunderAuditionBtn = document.getElementById('btn-thunder-audition');
  const spectrumBarEls = Array.from(document.querySelectorAll('#acoustic-spectrum-bars > i'));
  const spectrumScratch8 = new Uint8Array(8);
  const atmosRadarCanvas = document.getElementById('atmos-radar-canvas');
  const atmosRadarCtx = atmosRadarCanvas ? atmosRadarCanvas.getContext('2d') : null;
  const atmosBadgeEl = document.getElementById('ui-atmos-badge');
  const atmosPerspEl = document.getElementById('ui-atmos-perspective');
  const barAtmosEar = document.getElementById('bar-atmos-ear');
  const barAtmosHeight = document.getElementById('bar-atmos-height');
  const barAtmosLfe = document.getElementById('bar-atmos-lfe');
  const valAtmosEar = document.getElementById('val-atmos-ear');
  const valAtmosHeight = document.getElementById('val-atmos-height');
  const valAtmosLfe = document.getElementById('val-atmos-lfe');

  function updateAcousticWorldNow(dtStep) {
    camera.updateMatrixWorld();
    const cx = camera.position.x;
    const cz = camera.position.z;
    const dUCam = TerrainDEMLoader.getIslandUCoastDistance(cx, cz);
    const dICam = TerrainDEMLoader.getIslandICoastDistance(cx, cz);
    const coastDist = Math.min(dUCam, dICam);
    const seaDirX =
      TerrainDEMLoader.getIslandCoastDistance(cx + 4.0, cz) -
      TerrainDEMLoader.getIslandCoastDistance(cx - 4.0, cz);
    const seaDirZ =
      TerrainDEMLoader.getIslandCoastDistance(cx, cz + 4.0) -
      TerrainDEMLoader.getIslandCoastDistance(cx, cz - 4.0);
    const wVec = weather.uniforms.uWindVec.value;
    const windSpeed = wVec.length();
    const wzComp = Number.isFinite(wVec.z) && Math.abs(wVec.z) > 1e-5 ? wVec.z : wVec.y;
    const apasActive = (camDirector._avataApasTimer || 0) > 0.01;
    const inCorridor = (camDirector._avataMeta?.corridorWeight || 0) > 0.35;
    audioSynth.updateAcousticWorld(dtStep, {
      camera,
      camMode: camDirector.mode,
      fpsSubjectKind: camDirector.lastFPSSubjectKind || undefined,
      isRoofSheltered: DynamicTrafficAI._isShelteredByRoof(cx, cz),
      solarHour: weather.currentSolarHour,
      temperatureC: latestDepokWeather?.temperatureC,
      humidityPct: latestDepokWeather?.humidityPct,
      rainIntensity: weather.uniforms.uRainIntensity.value,
      stormFactor: weather.uniforms.uStormFactor.value,
      windSpeed,
      windDirX: windSpeed > 1e-3 ? wVec.x / windSpeed : 0.78,
      windDirZ: windSpeed > 1e-3 ? wzComp / windSpeed : 0.62,
      wetness: weather.uniforms.uWetness.value,
      seaWaveHeightM: weather.getSeaWaveHeightMeters(),
      nightFactor: weather.uniforms.uNightFactor?.value ?? weather.nightIlluminationLevel,
      lightningDir: weather.uniforms.uLightningDir.value,
      fountainNightBlend: rotundaFountain.nightBlend || 0.0,
      coastDist,
      seaDirX,
      seaDirZ,
      groundY: TerrainDEMLoader.sampleGroundHeightWithCoast(cx, cz, dUCam, dICam),
      riverZ: TerrainDEMLoader.evalPulauURiverCenterZ(cx),
      videotronPlaying: Boolean(videotron && !videotron.isMuted && videotron.isPlaying),
      sivitasAI,
      avataThrottle01: camDirector._avataThrottle01 || 0.0,
      avataTurnRate: Math.abs(camDirector._avataYawRateSmooth || 0.0),
      avataSignedYawRate: camDirector._avataYawRateSmooth || 0.0,
      avataGForce: camDirector._avataGForce || 1.0,
      avataStuntActive: Boolean(camDirector._avataStunt?.active),
      avataApasActive: apasActive,
      obstacleClearanceM: apasActive ? 2.2 : inCorridor ? 2.8 : 99.0,
      mavicFov: camDirector._mavicCinemaFov || 43.0
    });
  }

  function syncAudioButtonsDOM() {
    const active = Boolean(audioSynth.enabled);
    if (audioBtn) {
      const titleSpan = audioBtn.querySelector('.btn-title') || audioBtn;
      titleSpan.textContent = active ? 'Suara: ON' : 'Suara: OFF';
      audioBtn.classList.toggle('active', active);
    }
    if (qdockAudioBtn) {
      qdockAudioBtn.textContent = active ? '\u{1F50A}' : '\u{1F508}';
      qdockAudioBtn.classList.toggle('sound-on', active);
    }
    if (avataAudioBtn) {
      avataAudioBtn.textContent = active ? '\u{1F50A} ON' : '\u{1F508} Suara';
      avataAudioBtn.classList.toggle('active', active);
    }
    const curMix = audioSynth.getMixProfile ? audioSynth.getMixProfile() : 'BALANCED';
    audioMixBtns.forEach((b) => {
      b.classList.toggle('active', b.dataset.audiomix === curMix);
    });
    if (spatialModeBtn) {
      const spMode = audioSynth.getSpatialOutputMode ? audioSynth.getSpatialOutputMode() : 'BINAURAL';
      spatialModeBtn.textContent = spMode === 'SPEAKER' ? 'Speaker' : 'Atmos 3D';
      spatialModeBtn.classList.toggle('active', spMode !== 'SPEAKER');
    }
  }

  function renderAtmos714Radar(stage, isAudioLive) {
    if (!atmosRadarCtx || !atmosRadarCanvas || !stage) return;
    const w = atmosRadarCanvas.width;
    const h = atmosRadarCanvas.height;
    const cx = w * 0.5;
    const cy = h * 0.5;
    const rOuter = w * 0.44;
    const rHeight = w * 0.23;
    const ctx2d = atmosRadarCtx;

    ctx2d.clearRect(0, 0, w, h);

    // Background acoustic sphere
    ctx2d.beginPath();
    ctx2d.arc(cx, cy, rOuter + 2, 0, Math.PI * 2);
    ctx2d.fillStyle = isAudioLive ? 'rgba(10, 20, 36, 0.92)' : 'rgba(10, 16, 26, 0.78)';
    ctx2d.fill();

    // Outer 7.0 Ear-Level Ring
    ctx2d.beginPath();
    ctx2d.arc(cx, cy, rOuter, 0, Math.PI * 2);
    ctx2d.strokeStyle = isAudioLive ? 'rgba(56, 217, 150, 0.34)' : 'rgba(148, 163, 184, 0.22)';
    ctx2d.lineWidth = 1.3;
    ctx2d.stroke();

    // Inner .4 Overhead Height Dome Ring
    ctx2d.save();
    ctx2d.setLineDash([2.5, 2.5]);
    ctx2d.beginPath();
    ctx2d.arc(cx, cy, rHeight, 0, Math.PI * 2);
    ctx2d.strokeStyle = 'rgba(56, 189, 248, 0.36)';
    ctx2d.lineWidth = 1.1;
    ctx2d.stroke();
    ctx2d.restore();

    // Crosshairs + Forward Nose Heading Pointer (0 deg Azimuth at Top)
    ctx2d.beginPath();
    ctx2d.moveTo(cx, cy - rOuter);
    ctx2d.lineTo(cx, cy + rOuter);
    ctx2d.moveTo(cx - rOuter, cy);
    ctx2d.lineTo(cx + rOuter, cy);
    ctx2d.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx2d.lineWidth = 1;
    ctx2d.stroke();

    // 12 Dolby Atmos 7.1.4 Virtual Speaker Nodes
    const layout = AudioSynthesizer.ATMOS_SPEAKER_LAYOUT_7_1_4;
    const beds = stage.bedLevels12;
    if (layout && beds) {
      const deg = Math.PI / 180.0;
      for (let s = 0; s < layout.length; s++) {
        const sp = layout[s];
        if (sp.layer === 'LFE') continue;
        const az = sp.az * deg;
        const rr = sp.layer === 'HEIGHT' ? rHeight : rOuter;
        const sx = cx + Math.sin(az) * rr;
        const sy = cy - Math.cos(az) * rr;
        const lv = beds[s] || 0.0;
        ctx2d.beginPath();
        ctx2d.arc(sx, sy, 1.5 + lv * 2.1, 0, Math.PI * 2);
        ctx2d.fillStyle =
          sp.layer === 'HEIGHT'
            ? `rgba(56, 189, 248, ${(0.28 + lv * 0.68).toFixed(2)})`
            : `rgba(56, 217, 150, ${(0.25 + lv * 0.68).toFixed(2)})`;
        ctx2d.fill();
      }
    }

    // Active 3D Point-of-View Spatial Sound Objects (Azimuth + Elevation + Volumetric Spread)
    const objs = stage.activeObjects || [];
    const deg = Math.PI / 180.0;
    for (let i = objs.length - 1; i >= 0; i--) {
      const o = objs[i];
      const az = (o.azDeg || 0) * deg;
      const el = o.elDeg || 0;
      const rad =
        el > 0
          ? rOuter * (1.0 - Math.min(0.75, el / 90.0) * 0.56)
          : rOuter * (0.86 + Math.min(1.0, -el / 90.0) * 0.11);
      const px = cx + Math.sin(az) * rad;
      const py = cy - Math.cos(az) * rad;
      const g = Math.max(0.08, Math.min(1.0, o.gain01 || 0.1));
      const colorRgb =
        o.kind === 'VOICE'
          ? '56, 217, 150'
          : el >= 16.0
            ? '56, 189, 248'
            : '242, 194, 48';

      if ((o.spread01 || 0) > 0.18) {
        ctx2d.beginPath();
        ctx2d.arc(px, py, 3.2 + o.spread01 * 7.5, 0, Math.PI * 2);
        ctx2d.fillStyle = `rgba(${colorRgb}, ${(0.08 + g * 0.18).toFixed(2)})`;
        ctx2d.fill();
      }
      ctx2d.beginPath();
      ctx2d.arc(px, py, 2.1 + g * 2.6, 0, Math.PI * 2);
      ctx2d.fillStyle = `rgba(${colorRgb}, ${(0.48 + g * 0.50).toFixed(2)})`;
      ctx2d.fill();
    }

    // Listener Head Nose Triangle at Center
    ctx2d.beginPath();
    ctx2d.moveTo(cx, cy - 4.2);
    ctx2d.lineTo(cx - 3.0, cy + 3.0);
    ctx2d.lineTo(cx + 3.0, cy + 3.0);
    ctx2d.closePath();
    ctx2d.fillStyle = '#f8fafc';
    ctx2d.fill();
  }

  function syncAcousticTelemetryDOM() {
    const tel = audioSynth.getAcousticTelemetry();
    const volPct = Math.round(tel.masterVolume * 100);
    if (audioVolLabel) {
      audioVolLabel.textContent = `${volPct}%`;
    }
    if (audioVolSlider && document.activeElement !== audioVolSlider) {
      audioVolSlider.value = String(volPct);
    }
    if (spectrumBarEls.length === 8 && audioSynth.sampleSpectrum8Bars) {
      audioSynth.sampleSpectrum8Bars(spectrumScratch8);
      for (let i = 0; i < 8; i++) {
        const v01 = tel.enabled ? spectrumScratch8[i] / 255.0 : 0.0;
        const hPx = tel.enabled ? Math.max(2, Math.round(2 + v01 * 8)) : 2;
        spectrumBarEls[i].style.height = `${hPx}px`;
        spectrumBarEls[i].style.opacity = tel.enabled ? String((0.42 + v01 * 0.58).toFixed(2)) : '0.32';
      }
    }
    if (audioSynth.getAtmosStageSnapshot) {
      const stage = audioSynth.getAtmosStageSnapshot();
      if (atmosBadgeEl) {
        atmosBadgeEl.textContent = stage.spatialMode === 'SPEAKER' ? '7.1.4 SPK' : 'ATMOS 7.1.4';
      }
      if (atmosPerspEl) {
        atmosPerspEl.textContent = stage.perspectiveLabel || tel.perspectiveLabel || 'Titik Kamera 3D';
      }
      const earPct = Math.round((stage.earEnergy01 || 0) * 100);
      const hgtPct = Math.round((stage.heightEnergy01 || 0) * 100);
      const lfePct = Math.round((stage.lfeEnergy01 || 0) * 100);
      if (barAtmosEar) barAtmosEar.style.width = `${Math.max(4, earPct)}%`;
      if (barAtmosHeight) barAtmosHeight.style.width = `${Math.max(4, hgtPct)}%`;
      if (barAtmosLfe) barAtmosLfe.style.width = `${Math.max(3, lfePct)}%`;
      if (valAtmosEar) valAtmosEar.textContent = `${earPct}%`;
      if (valAtmosHeight) valAtmosHeight.textContent = `${hgtPct}%`;
      if (valAtmosLfe) valAtmosLfe.textContent = `${lfePct}%`;
      renderAtmos714Radar(stage, Boolean(tel.enabled));
    }
    if (acousticStatusEl) {
      const contextLabel =
        tel.zoneId === 'OPEN_AIR' || tel.zoneId === 'HIGH_ALTITUDE'
          ? tel.microBiomeLabel || tel.zoneLabel
          : tel.zoneLabel;
      if (tel.enabled) {
        const doppTxt =
          Math.abs(tel.dominantDoppler - 1.0) >= 0.008
            ? ` (${tel.dominantDoppler.toFixed(2)}\u00D7)`
            : '';
        const azTxt = `Az ${tel.dominantAzDeg >= 0 ? '+' : ''}${tel.dominantAzDeg}\u00B0`;
        const focusTxt = tel.activeSfxBanner
          ? `SFX: ${tel.activeSfxBanner}`
          : `${tel.dominantSourceLabel}${doppTxt} \u00B7 ${azTxt}`;
        acousticStatusEl.textContent = `${contextLabel} \u00B7 Gema ${tel.reverbSendPct}% \u00B7 ${focusTxt}`;
        acousticStatusEl.style.color = 'var(--ok-green)';
      } else {
        acousticStatusEl.textContent = `Atmos 7.1.4 \u00B7 ${tel.perspectiveLabel || contextLabel} \u00B7 Klik Suara / G`;
        acousticStatusEl.style.color = 'var(--text-muted)';
      }
    }
  }

  function toggleMasterAudio() {
    dismissWelcomeToast();
    if (!audioSynth.enabled && audioSynth.getMasterVolume() < 0.05) {
      audioSynth.setMasterVolume(0.8);
      if (audioVolSlider) audioVolSlider.value = '80';
    }
    const active = audioSynth.toggle();
    updateAcousticWorldNow(0.016);
    if (active) audioSynth.playUiChime(640);
    syncAudioButtonsDOM();
    syncAcousticTelemetryDOM();
  }

  if (audioVolSlider) {
    audioVolSlider.addEventListener('input', () => {
      dismissWelcomeToast();
      const vol01 = Math.max(0.0, Math.min(1.0, Number(audioVolSlider.value) / 100.0));
      if (vol01 <= 0.005) {
        audioSynth.setMasterVolume(0.0);
        if (audioSynth.enabled) audioSynth.toggle(false);
      } else {
        audioSynth.setMasterVolume(vol01);
        if (!audioSynth.enabled) {
          audioSynth.toggle(true);
          updateAcousticWorldNow(0.016);
        }
      }
      syncAudioButtonsDOM();
      syncAcousticTelemetryDOM();
    });
  }

  audioMixBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      dismissWelcomeToast();
      audioSynth.setMixProfile(btn.dataset.audiomix);
      if (!audioSynth.enabled) {
        audioSynth.toggle(true);
      }
      updateAcousticWorldNow(0.016);
      audioSynth.playUiChime(btn.dataset.audiomix === 'NATURE' ? 528 : btn.dataset.audiomix === 'TRANSIT' ? 740 : 640);
      syncAudioButtonsDOM();
      syncAcousticTelemetryDOM();
    });
  });

  if (spatialModeBtn) {
    spatialModeBtn.addEventListener('click', () => {
      dismissWelcomeToast();
      const cur = audioSynth.getSpatialOutputMode ? audioSynth.getSpatialOutputMode() : 'BINAURAL';
      const next = cur === 'BINAURAL' ? 'SPEAKER' : 'BINAURAL';
      if (audioSynth.setSpatialOutputMode) audioSynth.setSpatialOutputMode(next);
      if (audioSynth.enabled) audioSynth.playUiChime(next === 'BINAURAL' ? 680 : 540);
      syncAudioButtonsDOM();
      syncAcousticTelemetryDOM();
    });
  }

  if (bellAuditionBtn) {
    bellAuditionBtn.addEventListener('click', () => {
      dismissWelcomeToast();
      if (!audioSynth.enabled) {
        if (audioSynth.getMasterVolume() < 0.05) {
          audioSynth.setMasterVolume(0.8);
          if (audioVolSlider) audioVolSlider.value = '80';
        }
        audioSynth.toggle(true);
      }
      updateAcousticWorldNow(0.016);
      if (audioSynth.triggerCampusBell) audioSynth.triggerCampusBell(4);
      syncAudioButtonsDOM();
      syncAcousticTelemetryDOM();
    });
  }

  if (thunderAuditionBtn) {
    thunderAuditionBtn.addEventListener('click', () => {
      dismissWelcomeToast();
      if (!audioSynth.enabled) {
        if (audioSynth.getMasterVolume() < 0.05) {
          audioSynth.setMasterVolume(0.8);
          if (audioVolSlider) audioVolSlider.value = '80';
        }
        audioSynth.toggle(true);
      }
      updateAcousticWorldNow(0.016);
      if (weather && typeof weather.triggerManualLightningStrike === 'function') {
        weather.triggerManualLightningStrike(camera);
      } else {
        audioSynth.triggerThunder(195, 1.48, weather.uniforms.uLightningDir.value, { immediate: true });
      }
      window.NexusForensic?.logBreadcrumb('thunder_audition', '195m_5layer_shockwave');
      syncAudioButtonsDOM();
      syncAcousticTelemetryDOM();
    });
  }

  if (audioBtn) audioBtn.addEventListener('click', toggleMasterAudio);
  if (qdockAudioBtn) qdockAudioBtn.addEventListener('click', toggleMasterAudio);
  if (avataAudioBtn) avataAudioBtn.addEventListener('click', toggleMasterAudio);
  updateAcousticWorldNow(0.016);
  syncAudioButtonsDOM();
  syncAcousticTelemetryDOM();

  // Global Quick Keyboard Shortcuts (Non-Intrusive When Typing in Inputs)
  const QUICK_VIEW_CYCLE = ['DRONE_6DOF', 'DRONE_AVATA2_FPV', 'ORBIT', 'WALK_SIVITAS', 'FPS_CYCLE'];
  window.addEventListener('keydown', (e) => {
    const t = e.target;
    const editable =
      t &&
      (t.tagName === 'INPUT' ||
        t.tagName === 'TEXTAREA' ||
        t.tagName === 'SELECT' ||
        t.isContentEditable);
    if (editable || e.ctrlKey || e.metaKey || e.altKey) return;

    dismissWelcomeToast();

    if (e.code === 'Escape') {
      if (helpPopoverEl && helpPopoverEl.classList.contains('active')) {
        helpPopoverEl.classList.remove('active');
        qdockHelpBtn?.classList.remove('active');
        return;
      }
      if (document.body.classList.contains('clean-cinema-ui')) {
        toggleCleanCinemaUI(false);
        return;
      }
      if (isPanelExpanded) {
        setPanelExpanded(false);
      }
      return;
    }

    if (e.repeat) return;

    if (e.code === 'Tab') {
      e.preventDefault();
      setPanelExpanded(!isPanelExpanded);
      return;
    }
    if (e.key === '?') {
      if (helpPopoverEl) {
        const open = !helpPopoverEl.classList.contains('active');
        helpPopoverEl.classList.toggle('active', open);
        qdockHelpBtn?.classList.toggle('active', open);
      }
      return;
    }
    if (e.code === 'KeyG') {
      toggleMasterAudio();
      return;
    }
    if (e.code === 'KeyH') {
      toggleCleanCinemaUI();
      return;
    }
    if (e.code === 'KeyV') {
      const curKey = camDirector.isFPSMode()
        ? 'FPS_CYCLE'
        : camDirector.mode === 'AUTO_CINEMATIC'
          ? 'DRONE_6DOF'
          : camDirector.mode;
      const nextIdx = (QUICK_VIEW_CYCLE.indexOf(curKey) + 1) % QUICK_VIEW_CYCLE.length;
      const nextKey = QUICK_VIEW_CYCLE[nextIdx];
      if (nextKey === 'FPS_CYCLE') {
        activateModeFromUI(FPS_ROLE_ORDER[lastFpsRoleIdx], { forceReset: true });
      } else {
        activateModeFromUI(nextKey, { fromQuickDock: true });
      }
      return;
    }
    if (e.code === 'KeyN') {
      if (camDirector.isAvataFPVMode()) {
        camDirector.jumpAvataToZone('RANDOM');
        audioSynth.playUiChime(660);
        syncFPSStatusDOM();
      } else if (camDirector.isFPSMode()) {
        camDirector.cycleFPSSubject(1);
        audioSynth.playUiChime(610);
        syncFPSStatusDOM();
      }
      return;
    }
    if (e.code === 'KeyR' && (camDirector.mode === 'DRONE_6DOF' || camDirector.mode === 'AUTO_CINEMATIC')) {
      camDirector.resetMavicGimbalAndZoom();
      syncViewport();
      syncQuickDockAndContextDOM();
      audioSynth.playUiChime(560);
      return;
    }
    if (e.code === 'KeyL') {
      copyPermalinkToClipboard();
    }
  });

  // --- ROBUST BIDIRECTIONAL PERMALINK ENGINE & AVIATION NFZ / NO-WANDERING HUD TOAST ---
  const geoToastEl = document.getElementById('nexus-geo-toast');
  const qdockPermalinkBtn = document.getElementById('btn-qdock-permalink');
  let geoToastOverrideUntilMs = 0;
  let lastGeoAlertShownMs = 0;
  let activeSpotPresetKey = '';
  let lastPermalinkSyncMs = 0;
  let isApplyingPermalink = false;

  presetBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      activeSpotPresetKey = btn.dataset.preset || '';
      permalinkDirty = true;
    });
  });

  function showGeoOrPermalinkToast(text, isPermalink = false, durationMs = 3200) {
    if (!geoToastEl) return;
    const now = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
    geoToastEl.textContent = text;
    geoToastEl.classList.toggle('is-permalink', Boolean(isPermalink));
    geoToastEl.classList.remove('is-hidden');
    if (isPermalink) {
      geoToastOverrideUntilMs = now + durationMs;
    }
  }

  function syncGeoZoneWarningDOM() {
    if (!geoToastEl) return;
    const now = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
    if (now < geoToastOverrideUntilMs) return;
    const st = PhysicsKernel?.lastGeoZoneStatus;
    if (st && st.active && now - st.timestampMs < 1600) {
      geoToastEl.textContent = st.labelId;
      geoToastEl.classList.remove('is-permalink', 'is-hidden');
      if (now - lastGeoAlertShownMs > 1800) {
        lastGeoAlertShownMs = now;
        if (audioSynth && audioSynth.enabled) {
          audioSynth.playUiChime(320);
        }
      }
    } else {
      geoToastEl.classList.add('is-hidden');
    }
  }

  function buildPermalinkHash() {
    const params = new URLSearchParams();
    const mode = camDirector.mode || 'DRONE_6DOF';
    params.set('m', mode);
    if (activeSpotPresetKey && mode === 'ORBIT') {
      params.set('spot', activeSpotPresetKey);
    }
    const px = Number(camera.position.x) || 0;
    const py = Number(camera.position.y) || 0;
    const pz = Number(camera.position.z) || 0;
    params.set('p', `${px.toFixed(2)},${py.toFixed(2)},${pz.toFixed(2)}`);

    if (mode === 'ORBIT' && controls && controls.target) {
      const tx = Number(controls.target.x) || 0;
      const ty = Number(controls.target.y) || 0;
      const tz = Number(controls.target.z) || 0;
      params.set('t', `${tx.toFixed(2)},${ty.toFixed(2)},${tz.toFixed(2)}`);
    } else if (!camDirector.isFPSMode()) {
      const yaw = Number(camDirector.yaw) || 0;
      const pitch = Number(camDirector.pitch) || 0;
      const roll = Number(camDirector.roll) || 0;
      params.set('a', `${yaw.toFixed(3)},${pitch.toFixed(3)},${roll.toFixed(3)}`);
      if (mode === 'DRONE_AVATA2_FPV') {
        params.set('free', camDirector.avataAutoFreestyle ? '0' : '1');
        if (camDirector.avataFlightProfile === 'ACRO') params.set('acro', '1');
      } else if (mode === 'DRONE_6DOF') {
        params.set('free', camDirector._droneManualEngaged ? '1' : '0');
      }
    } else {
      params.set('sub', String(camDirector.fpsSubjectOffset || 0));
    }

    params.set('live', isLiveDepok ? '1' : '0');
    if (!isLiveDepok) {
      params.set('tm', weather.currentSolarHour.toFixed(2));
      params.set('wx', (weather.getWeatherSpectrum() * 100.0).toFixed(1));
    }
    if (weather.isManualSeaState()) {
      params.set('wv', weather.getSeaWaveHeightMeters().toFixed(2));
    }

    const gModeVal = Number(sharedUniforms.uGenshinMode?.value || 0);
    const isGenshin = gModeVal > 0.5;
    const isForza = gModeVal < -0.5;
    const zebraMode = Number(sharedUniforms.uZebraInspectMode?.value || 0);
    const visTag = isGenshin
      ? 'genshin'
      : isForza
        ? 'forza'
        : zebraMode === 1
          ? 'zebra1'
          : zebraMode === 2
            ? 'zebra2'
            : 'pbr';
    params.set('vis', visTag);

    const vtState = videotron && typeof videotron.getState === 'function' ? videotron.getState() : null;
    if (vtState && vtState.videoId && vtState.videoId !== 'N5vSW33FwQQ') {
      params.set('vt', vtState.videoId);
    }
    return params.toString();
  }

  function getPermalinkURL() {
    const hash = buildPermalinkHash();
    if (typeof window === 'undefined' || !window.location) return `#${hash}`;
    const base = `${window.location.origin}${window.location.pathname}${window.location.search}`;
    return `${base}#${hash}`;
  }

  function syncPermalinkReplaceState(force = false) {
    if (isApplyingPermalink || typeof window === 'undefined' || !window.history?.replaceState) return;
    const now = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
    if (!force && (!permalinkDirty || now - lastPermalinkSyncMs < 1200)) return;
    permalinkDirty = false;
    lastPermalinkSyncMs = now;
    const hash = buildPermalinkHash();
    try {
      window.history.replaceState(null, '', `#${hash}`);
    } catch (_e) {
      // Ignore file:// or sandboxed history restrictions
    }
  }

  function applyPermalinkFromURL(customInput = null) {
    if (typeof window === 'undefined') return false;
    let raw = '';
    if (typeof customInput === 'string' && customInput.trim().length > 0) {
      const s = customInput.trim();
      const hIdx = s.indexOf('#');
      const qIdx = s.indexOf('?');
      if (hIdx >= 0) raw = s.slice(hIdx + 1);
      else if (qIdx >= 0) raw = s.slice(qIdx + 1);
      else raw = s.replace(/^[#?]/, '');
    } else {
      const h = (window.location.hash || '').replace(/^#/, '');
      const q = (window.location.search || '').replace(/^\?/, '');
      raw = h && h.includes('=') ? h : q;
    }
    if (!raw || !raw.includes('=')) return false;

    const params = new URLSearchParams(raw);
    const hasAnyKey =
      params.has('m') ||
      params.has('spot') ||
      params.has('p') ||
      params.has('live') ||
      params.has('tm') ||
      params.has('wx') ||
      params.has('wv') ||
      params.has('vis') ||
      params.has('vt');
    if (!hasAnyKey) return false;

    isApplyingPermalink = true;
    try {
      arrivalActive = false;
      dismissWelcomeToast();

      // 1. Visual Preset (pbr | forza | genshin | zebra1 | zebra2)
      if (params.has('vis')) {
        const v = String(params.get('vis') || '').toLowerCase();
        if (v === 'genshin' || v === 'anime' || v === '1') {
          setVisualInspectPreset('GENSHIN', false);
        } else if (v === 'forza' || v === 'fh6' || v === 'forza6') {
          setVisualInspectPreset('FORZA', false);
        } else if (v === 'zebra1' || v === 'zebra') {
          setVisualInspectPreset('1', false);
        } else if (v === 'zebra2') {
          setVisualInspectPreset('2', false);
        } else {
          setVisualInspectPreset('0', false);
        }
      }

      // 2. Live Depok vs Manual Solar Time, Weather Spectrum, & Sea Wave Height
      if (params.has('live')) {
        isLiveDepok = params.get('live') === '1' || params.get('live') === 'true';
      } else if (params.has('tm') || params.has('wx')) {
        isLiveDepok = false;
      }

      if (!isLiveDepok) {
        if (params.has('tm')) {
          const tmVal = Number(params.get('tm'));
          if (Number.isFinite(tmVal)) {
            weather.setContinuousTimeHours(Math.max(0.0, Math.min(24.0, tmVal)), sharedUniforms, {
              immediate: true
            });
          }
        }
        if (params.has('wx')) {
          const wxVal = Number(params.get('wx'));
          if (Number.isFinite(wxVal)) {
            const u01 = Math.max(0.0, Math.min(1.0, wxVal > 1.001 ? wxVal / 100.0 : wxVal));
            weather.setWeatherSpectrum(u01, sharedUniforms, { immediate: true });
          }
        }
      }

      if (params.has('wv')) {
        const wvRaw = String(params.get('wv') || '').toLowerCase();
        if (wvRaw === 'auto') {
          weather.setSeaWaveHeightMeters(null, { auto: true, immediate: true });
        } else {
          const wvVal = Number(wvRaw);
          if (Number.isFinite(wvVal)) {
            weather.setSeaWaveHeightMeters(Math.max(0.0, Math.min(3.0, wvVal)), { immediate: true });
          }
        }
      }

      weather.update(0.016, camera, sharedUniforms);
      GedungRektoratUIAsset.setNightIllumination(rektoratAsset, weather.nightIlluminationLevel);
      updateDepokTelemetryDOM();
      syncTransitionButtonsUI();

      // 3. Camera Mode, Spot Preset, & Exact 3D Coordinates / Angles
      const spotKey = params.get('spot') || '';
      const modeParam = params.get('m') || (spotKey ? 'ORBIT' : '');
      if (spotKey && presets[spotKey]) {
        activeSpotPresetKey = spotKey;
        presetBtns.forEach((b) => b.classList.toggle('active', b.dataset.preset === spotKey));
        camDirector.applyOrthoPreset(spotKey, presets, { immediate: true });
      } else if (modeParam) {
        const subOff = Number(params.get('sub')) || 0;
        camDirector.setMode(modeParam, { immediate: true, subjectOffset: subOff });
      }

      if (params.has('p')) {
        const parts = String(params.get('p'))
          .split(',')
          .map((n) => Number(n));
        if (parts.length === 3 && parts.every((n) => Number.isFinite(n))) {
          camDirector.cancelTransition();
          camera.position.set(parts[0], parts[1], parts[2]);
          camDirector.syncPreviousCameraPosition();
        }
      }

      if (params.has('t') && controls && controls.target) {
        const tParts = String(params.get('t'))
          .split(',')
          .map((n) => Number(n));
        if (tParts.length === 3 && tParts.every((n) => Number.isFinite(n))) {
          controls.target.set(tParts[0], tParts[1], tParts[2]);
          camera.lookAt(controls.target);
          controls.update();
        }
      }

      if (params.has('a') && camDirector.mode !== 'ORBIT' && !camDirector.isFPSMode()) {
        const aParts = String(params.get('a'))
          .split(',')
          .map((n) => Number(n));
        if (aParts.length >= 2 && aParts.every((n) => Number.isFinite(n))) {
          camDirector.yaw = aParts[0];
          camDirector.pitch = aParts[1];
          camDirector.roll = Number.isFinite(aParts[2]) ? aParts[2] : 0.0;
          if (camDirector.mode === 'DRONE_6DOF') {
            camDirector._droneManualEngaged = params.get('free') !== '0';
          } else if (camDirector.mode === 'DRONE_AVATA2_FPV') {
            camDirector.avataAutoFreestyle = params.get('free') === '0';
            if (params.get('acro') === '1') camDirector.avataFlightProfile = 'ACRO';
          }
          const eul = new THREE.Euler(camDirector.pitch, camDirector.yaw, camDirector.roll, 'YXZ');
          camera.quaternion.setFromEuler(eul);
        }
      }

      // 4. Custom Videotron YouTube Video ID / URL
      if (params.has('vt') && videotron && typeof videotron.loadSource === 'function') {
        const vtVal = String(params.get('vt') || '').trim();
        if (vtVal) {
          videotron.loadSource(vtVal);
        }
      }

      syncQuickDockAndContextDOM();
      syncFPSStatusDOM();
      syncViewport();
      return true;
    } finally {
      isApplyingPermalink = false;
    }
  }

  async function copyPermalinkToClipboard() {
    syncPermalinkReplaceState(true);
    const fullUrl = getPermalinkURL();
    let copied = false;
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(fullUrl);
        copied = true;
      } catch (_e) {
        copied = false;
      }
    }
    if (audioSynth && audioSynth.enabled) {
      audioSynth.playUiChime(720);
    }
    showGeoOrPermalinkToast(
      copied
        ? '🔗 Permalink Disalin! (Posisi Kamera 3D, Cuaca, Waktu & Shader Tersimpan di URL)'
        : '🔗 URL Permalink Diperbarui pada Bilah Alamat Browser',
      true,
      3400
    );
    return fullUrl;
  }

  if (qdockPermalinkBtn) {
    qdockPermalinkBtn.addEventListener('click', () => {
      copyPermalinkToClipboard();
    });
  }

  window.addEventListener('hashchange', () => {
    if (!isApplyingPermalink) {
      applyPermalinkFromURL();
    }
  });

  // Apply initial URL Permalink if user opened a shared link (#m=... or ?m=...)
  applyPermalinkFromURL();

  // Hardware Profiler, Substantive 30s Forensic Blackbox, 60s Ring Buffer, & Adaptive 120-FPS Governor Controls
  const elHwTier = document.getElementById('sim-hw-tier');
  const elHwDetail = document.getElementById('sim-hw-detail');
  const elHwGpuName = document.getElementById('sim-hw-gpu-name');
  const elForensicGpuMeta = document.getElementById('forensic-gpu-meta');
  const elForensicSummary = document.getElementById('forensic-60s-summary');
  const elForensicSpark = document.getElementById('forensic-sparkline');
  const elForensicDetail = document.getElementById('forensic-60s-detail');
  const elForensicBootV25 = document.getElementById('forensic-boot-v25');
  const elForensicSoundV24 = document.getElementById('forensic-sound-v24');
  const elForensic30sDiag = document.getElementById('drawer-forensic-30s-diag');
  const elForensicVerdict = document.getElementById('drawer-forensic-verdict');
  const elForensicEventLog = document.getElementById('forensic-event-log');
  const btnCopyForensic = document.getElementById('btn-copy-forensic');

  function buildForensicContext() {
    const acTel = audioSynth.getAcousticTelemetry();
    return {
      camMode: camDirector.mode,
      atmosLayout: acTel.atmosLayout || '7.1.4',
      atmosPerspective: acTel.perspectiveId || 'MAVIC_AERIAL',
      solarHourWIB: +weather.currentSolarHour.toFixed(2),
      weatherSpectrumPct: Math.round(weather.getWeatherSpectrum() * 100),
      seaWaveHeightM: +weather.getSeaWaveHeightMeters().toFixed(2),
      seaManual: weather.isManualSeaState()
    };
  }

  function syncForensicDOM() {
    if (!elForensicGpuMeta) return;
    const snap = hwGovernor.getForensicSnapshot(renderer, buildForensicContext());
    const g = snap.gpu;
    const w = snap.window60s;
    elForensicGpuMeta.textContent =
      `${g.cleanName} [${g.gpuClass}] · ${g.drawingBuffer} (DPR ${g.pixelRatio.toFixed(2)}×) · ` +
      `Shadow ${g.shadowMapSize}px · MaxTex ${g.maxTextureSize} · MSAA ${g.maxSamplesMSAA}× · Aniso ${g.maxAnisotropy}×`;
    if (elForensicSummary) {
      elForensicSummary.textContent =
        `${w.samplesSec || 1}d: Avg ${w.avgFps} · P5 ${w.low5Fps} · Min ${w.minFps} FPS`;
    }
    if (elForensicSpark) {
      elForensicSpark.textContent = w.sparkline || '▅▅▅';
      elForensicSpark.style.color =
        w.avgFps >= 50 ? 'var(--ok-green)' : w.avgFps >= 28 ? 'var(--ui-gold)' : '#ff7b72';
    }
    if (elForensicDetail) {
      const heapTxt = w.jsHeapMB !== null ? ` · Heap ${w.jsHeapMB}MB` : '';
      elForensicDetail.textContent =
        `Frame ${w.avgFrameMs}ms (P95 ${w.p95FrameMs}ms, Peak ${w.peakFrameMs}ms) · CPU ${w.avgCpuMs}ms · ` +
        `${w.bottleneck} · Prog ${w.programs} · Tex ${w.textures}${heapTxt}`;
    }
    if (elForensicBootV25) {
      const bf = bootProfiler.buildSnapshot({ audioSynth, weather });
      const bs = bf.summary;
      const topW = bs.heaviestWallStage ? `${bs.heaviestWallStage.id} (${bs.heaviestWallStage.wallMs}ms)` : 'n/a';
      const rainSt = bf.deferredBackgroundTasks.rainCollisionZBufferBake;
      const rainTxt =
        rainSt.status === 'completed' && rainSt.totalCpuMs !== null
          ? `RainBake ${rainSt.totalCpuMs}ms CPU`
          : `RainBake ${rainSt.status}`;
      elForensicBootV25.textContent =
        `Boot v2.5 [${bf.verdict}] · 1stFrm ${bs.timeToFirstFrameSubmittedMs}ms (CPU ${bs.totalBootBlockingCpuMs}ms) · ` +
        `Top #1 ${topW} · Gap ${bf.firstFrameGapBreakdown.totalGapMs}ms · ${rainTxt}`;
    }
    if (elForensicSoundV24 && typeof audioSynth.getForensicSnapshot === 'function') {
      const sf = audioSynth.getForensicSnapshot(hwGovernor);
      const hc = sf.hardwareAndContext;
      const dg = sf.dspGraphInventory;
      const rh = sf.realtimeHealth;
      const srTxt = hc.initialized ? `${hc.sampleRateHz}Hz` : 'Siaga';
      elForensicSoundV24.textContent =
        `Sound v2.4 [${sf.verdict}] · ${srTxt} · DSP ${dg.totalPersistentNodes}n ` +
        `(${dg.activeAudibleSubBuses} aktif / ${dg.hibernatedSubBuses || 0} hib / ${dg.silentRunningSubBuses} diam, ${dg.uncleanedTransientAuxNodes} aux) · ` +
        `CPU ${rh.emaUpdateUs}µs · Drift ${rh.lastClockDriftMs}ms (Und ${rh.underrunSuspectCount}) · Lim ${rh.currentLimiterReductionDb}dB`;
    }

    const r30 = forensicRecorder.buildRolling30sSnapshot();
    const syn = r30.forensicSynthesis;
    if (elForensic30sDiag && syn) {
      const ov = syn.overall30s;
      const pe = syn.pipelineEfficiency;
      const diag0 = syn.bottleneckFindings && syn.bottleneckFindings[0] ? syn.bottleneckFindings[0] : 'Optimal';
      elForensic30sDiag.textContent =
        `30d (${r30.capturedSpanSec}d): ${ov.avgFps} FPS · Calls ${pe.avgMainDrawCalls}/25 · Refl ${pe.planarReflPassActivePct}% · ${diag0}`;
    }
    if (elForensicVerdict && typeof window !== 'undefined' && window.NexusForensic) {
      const repLite = window.NexusForensic.buildReport('drawer_poll', false);
      const verd = repLite.meta.healthVerdict;
      elForensicVerdict.textContent = verd;
      elForensicVerdict.className =
        'forensic-verdict-pill ' +
        (verd === 'HEALTHY' ? 'healthy' : verd === 'DEGRADED' ? 'degraded' : 'critical');
    }

    if (elForensicEventLog) {
      const lines = [];
      const subEv = r30.substantiveEvents || [];
      for (let i = subEv.length - 1; i >= Math.max(0, subEv.length - 6); i--) {
        const se = subEv[i];
        if (se.kind === 'camera_switch') {
          lines.push(`[${se.tRelSec}d] CAM · ${se.from} → ${se.to} (${se.switchFrameMs}ms, ${se.callsAfter}c)`);
        } else if (se.kind === 'frame_spike') {
          lines.push(`[${se.tRelSec}d] SPIKE · ${se.frameMs}ms (${se.probableCause})`);
        } else if (se.kind === 'shader_compile') {
          lines.push(`[${se.tRelSec}d] SHADER · +${se.addedPrograms} prog (${se.frameMs}ms)`);
        } else {
          lines.push(`[${se.tRelSec}d] ${se.detail || se.kind}`);
        }
      }
      if (snap.events && snap.events.length > 0) {
        const startIdx = Math.max(0, snap.events.length - 8);
        for (let i = snap.events.length - 1; i >= startIdx; i--) {
          const ev = snap.events[i];
          const sec = Math.max(0, Math.floor(ev.tSec || 0));
          const mm = String(Math.floor(sec / 60)).padStart(2, '0');
          const ss = String(sec % 60).padStart(2, '0');
          lines.push(`[${mm}:${ss}] ${ev.tag} · ${ev.msg}`);
        }
      }
      if (lines.length > 0) {
        elForensicEventLog.textContent = lines.slice(0, 12).join('\n');
      }
    }
  }
  syncForensicDOMHook = syncForensicDOM;

  if (btnCopyForensic) {
    btnCopyForensic.addEventListener('click', () => {
      const lbl = document.getElementById('btn-copy-forensic-label') || btnCopyForensic;
      if (typeof window !== 'undefined' && window.NexusForensic) {
        window.NexusForensic.copyJSON(true).then(() => {
          lbl.textContent = '✓ Tersalin!';
          setTimeout(() => {
            lbl.textContent = '📋 Salin JSON';
          }, 1800);
        });
      } else {
        const jsonStr = hwGovernor.exportForensicLogJSON(renderer, buildForensicContext());
        if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
          navigator.clipboard
            .writeText(jsonStr)
            .then(() => {
              lbl.textContent = '✓ Tersalin!';
              setTimeout(() => {
                lbl.textContent = '📋 Salin JSON';
              }, 1800);
            })
            .catch(() => {});
        }
      }
      audioSynth.playUiChime(680);
    });
  }

  function syncHardwareTelemetryDOM() {
    const hw = hwGovernor.getTelemetryState();
    const pr = Number.isFinite(hw.pixelRatio) ? hw.pixelRatio : Number(hw.livePixelRatio) || 1.0;
    const cDist = Number.isFinite(hw.closeDerivativeDist) ? hw.closeDerivativeDist : 18.0;
    if (elHwTier) {
      elHwTier.textContent = `${hw.tier || hw.activeTierId || 'TIER_S'} · ${hw.tierShort || hw.badgeLabel || 'Auto'}`;
    }
    if (elHwDetail) {
      elHwDetail.textContent = `GPU BoneTex · ${pr.toFixed(2)}× · ${cDist.toFixed(0)}m`;
    }
    if (elHwGpuName) {
      elHwGpuName.textContent = `GPU: ${hw.gpuRenderer || hw.gpuModel || 'WebGL2 GPU'} (${hw.cpuCores || 8} Core, ${hw.deviceMemoryGB || 8} GB, Target ${hw.targetFps || 120} FPS)`;
    }
  }
  const _tInitForDom0 = _nowBoot();
  syncHardwareTelemetryDOM();
  syncForensicDOM();
  bootProfiler.firstFrameGap.initialSyncForensicDomCpuMs = +Math.max(0.0, _nowBoot() - _tInitForDom0).toFixed(2);

  document.querySelectorAll('[data-hwmode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-hwmode]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      hwGovernor.setMode(btn.dataset.hwmode);
      forensicRecorder.recordSubstantiveEvent('state_change', `Governor → ${btn.dataset.hwmode}`);
      syncHardwareTelemetryDOM();
      syncForensicDOM();
    });
  });

  // Main 120 FPS Ultra Render & Telemetry Loop
  const elFps = document.getElementById('sim-fps');
  const elCalls = document.getElementById('sim-calls');
  const elTris = document.getElementById('sim-tris');

  let prevTime = 0.0;
  let elapsed = 0.0;
  let frameCount = 0;
  let statTimer = 0.0;
  let clockTickAccum = 0.0;
  let uiTransAccum = 0.0;
  let forensicUiAccum = 0.0;
  let lastCpuMs = 2.0;
  let lastDrawCalls = 18;
  let lastAppliedNightIllum = -1.0;
  let firstFrameMarked = false;
  let secondFrameMarked = false;
  let _tokS16 = null;
  let _tokS16Closed = false;
  const cullBiomePlanarRefl = (reflProjView, waterY) =>
    AutogenBiomeSpawner.cullForPlanarReflection(biomeGroup, reflProjView, waterY);

  function renderLoop(now) {
    requestAnimationFrame(renderLoop);
    if (webglContextLost) return;
    if (!_tokS16Closed && _tokS16) {
      _tokS16Closed = true;
      const _s16Entry = bootProfiler.endStage(_tokS16, {
        pureAsyncIo: true,
        detail: 'Tunggu rAF pertama dari akhir bootstrap menuju eksekusi renderLoop #1'
      });
      if (_s16Entry) {
        bootProfiler.firstFrameGap.rafQueueWaitToFrame1Ms = _s16Entry.wallMs;
      }
    } else if (firstFrameMarked && !secondFrameMarked) {
      secondFrameMarked = true;
      bootProfiler.recordSecondFrameEntered();
    }
    const _tokS17 = !firstFrameMarked
      ? bootProfiler.beginStage(
          'S17_first_frame_cpu_submit',
          'FIRST_FRAME_GAP',
          'Simulasi, Planar Reflection & Submit PostFX Frame Pertama'
        )
      : null;
    if (prevTime <= 0.0) prevTime = now - 16.667;
    const rawWallSec = (now - prevTime) * 0.001;
    const rawDt =
      typeof document !== 'undefined' && document.hidden
        ? 0.016
        : Math.min(1.0, Math.max(0.0005, rawWallSec));
    const dt = Math.min(0.05, Math.max(0.001, rawDt));
    prevTime = now;
    elapsed += dt;
    const tCpuStart = performance.now();

    try {
      // Gentle 1.05s arrival camera ease-out right after preloader curtain lift
      if (arrivalActive) {
        arrivalEaseClock += dt;
        const tArr = Math.min(1.0, arrivalEaseClock / 1.05);
        const easeOut = 1.0 - Math.pow(1.0 - tArr, 3.0);
        camera.position.y = arrivalStartY + (arrivalEndY - arrivalStartY) * easeOut;
        if (tArr >= 1.0) arrivalActive = false;
      }

      // Real-time 1 Hz Kampus UI Depok WIB astronomical clock tracking (preserves smooth HVS pupil kinetics!)
      clockTickAccum += dt;
      if (clockTickAccum >= 1.0) {
        clockTickAccum = 0.0;
        latestDepokClock = DepokLiveWeatherService.getDepokWIBClock();
        updateDepokTelemetryDOM();
        if (isLiveDepok && !weather.getTransitionState().time.active) {
          weather.setSolarHour(latestDepokClock.solarHour, sharedUniforms, {
            immediate: true,
            preserveAdaptation: true
          });
        }
      }

      sharedUniforms.uTime.value = elapsed;
      weather.update(dt, camera, sharedUniforms);
      const liveSunElevDeg =
        (Math.asin(Math.max(-1.0, Math.min(1.0, Number(weather.uniforms.uTrueSunDir.value.y) || 0.0))) * 180.0) /
        Math.PI;
      rotundaFountain.update(elapsed, liveSunElevDeg, sharedUniforms, weather.uniforms.uSunDir.value);

      // Continuous C^1 Architectural Window, Garden Lamp, & Helipad Night Illumination + 3D HRTF Physical Acoustic Coupling (cached when steady!)
      if (Math.abs(weather.nightIlluminationLevel - lastAppliedNightIllum) > 0.0015) {
        lastAppliedNightIllum = weather.nightIlluminationLevel;
        GedungRektoratUIAsset.setNightIllumination(rektoratAsset, lastAppliedNightIllum);
      }

      sivitasAI.syncOceanState(weather.getSeaWaveHeightMeters(), weather.uniforms.uTime.value);
      sivitasAI.update(dt, elapsed, camera.position);

      // Feed 4 Live GPU 19.47° Lord Kelvin Ship Wake + Stern Propeller Wash Slots (0 GC!)
      if (sivitasAI.shuttleFerry) {
        const f = sivitasAI.shuttleFerry;
        weather.setVesselWakeSlot(0, f.worldX, f.worldZ, f.yaw, +(f.speed || 0.0));
      } else {
        weather.setVesselWakeSlot(0, 0, 0, 0, 0);
      }
      if (sivitasAI.miniYacht) {
        const y = sivitasAI.miniYacht;
        weather.setVesselWakeSlot(1, y.worldX, y.worldZ, y.yaw, -(y.speed || 0.0));
      } else {
        weather.setVesselWakeSlot(1, 0, 0, 0, 0);
      }
      if (Array.isArray(sivitasAI.marineVessels) && sivitasAI.marineVessels.length > 0) {
        let bestIdxA = -1;
        let bestIdxB = -1;
        let bestDistA = Infinity;
        let bestDistB = Infinity;
        const cxCam = camera.position.x;
        const czCam = camera.position.z;
        for (let vi = 0; vi < sivitasAI.marineVessels.length; vi++) {
          const mv = sivitasAI.marineVessels[vi];
          const dxM = mv.worldX - cxCam;
          const dzM = mv.worldZ - czCam;
          const d2 = dxM * dxM + dzM * dzM;
          if (d2 < bestDistA) {
            bestDistB = bestDistA;
            bestIdxB = bestIdxA;
            bestDistA = d2;
            bestIdxA = vi;
          } else if (d2 < bestDistB) {
            bestDistB = d2;
            bestIdxB = vi;
          }
        }
        if (bestIdxA >= 0) {
          const vA = sivitasAI.marineVessels[bestIdxA];
          const signedSpdA = vA.type === 'SMALL_SHIP' ? +(vA.speed || 0.0) : -(vA.speed || 0.0);
          weather.setVesselWakeSlot(2, vA.worldX, vA.worldZ, vA.yaw, signedSpdA);
        } else {
          weather.setVesselWakeSlot(2, 0, 0, 0, 0);
        }
        if (bestIdxB >= 0) {
          const vB = sivitasAI.marineVessels[bestIdxB];
          const signedSpdB = vB.type === 'SMALL_SHIP' ? +(vB.speed || 0.0) : -(vB.speed || 0.0);
          weather.setVesselWakeSlot(3, vB.worldX, vB.worldZ, vB.yaw, signedSpdB);
        } else {
          weather.setVesselWakeSlot(3, 0, 0, 0, 0);
        }
      }
      camDirector.update(dt, sampleWalkableElevation);
      if (camDirector.isAvataFPVMode()) {
        syncAvataHudDOM(true);
      }
      updateAcousticWorldNow(dt);
      GedungRektoratUIAsset.updateKinematics(rektoratAsset, dt, {
        isNight: weather.nightIlluminationLevel > 0.35,
        nightLevel: weather.nightIlluminationLevel,
        windMultiplier: weather.getWindGustMultiplier()
      });

      hwGovernor.update(dt, camera, controls.enabled ? controls.target : null, weather, {
        rawDt,
        cpuMs: lastCpuMs,
        calls: lastDrawCalls
      });
      videotron.update(camera, window.innerWidth, window.innerHeight, sharedUniforms);
      grassField.update(camera, { windVec: weather.uniforms.uWindVec.value }, sharedUniforms);
      const droneCinemaNow = isDroneCinemaView();
      postFX.updateMode(
        dt,
        droneCinemaNow,
        {
          duskFactor: weather.uniforms.uDuskFactor.value,
          nightFactor: weather.uniforms.uNightFactor.value
        },
        sharedUniforms.uGenshinMode ? sharedUniforms.uGenshinMode.value : 0.0
      );
      renderer.getDrawingBufferSize(drawBufSize);
      if (drawBufSize.x !== postFX.width || drawBufSize.y !== postFX.height) {
        postFX.setSize(drawBufSize.x, drawBufSize.y);
      }
      const shadowRefreshedThisFrame = Boolean(renderer.shadowMap && renderer.shadowMap.needsUpdate);
      const savedPortalAlpha = sharedUniforms.uVideotronPortalAlpha
        ? sharedUniforms.uVideotronPortalAlpha.value
        : 0.0;
      if (sharedUniforms.uVideotronPortalAlpha) {
        sharedUniforms.uVideotronPortalAlpha.value = 0.0;
      }
      // Single scene-graph matrixWorld update shared across Planar Reflection + Shadow + Main passes!
      scene.updateMatrixWorld();
      camera.updateMatrixWorld();
      const prevSceneAutoUpdate = scene.matrixWorldAutoUpdate;
      scene.matrixWorldAutoUpdate = false;
      let reflRanThisFrame = false;
      let reflCallsThisFrame = 0;
      let _tF1Refl0 = 0.0;
      let _tF1Main0 = 0.0;
      try {
        _tF1Refl0 = _tokS17 ? performance.now() : 0.0;
        postFX.renderPlanarWaterReflection(
          scene,
          camera,
          weather,
          TerrainDEMLoader.WATER_LEVEL_Y,
          planarReflHiddenList,
          cullBiomePlanarRefl
        );
        reflRanThisFrame = Boolean(postFX.lastPlanarReflRan);
        reflCallsThisFrame = reflRanThisFrame ? postFX.lastPlanarReflCalls || 0 : 0;
        if (sharedUniforms.uVideotronPortalAlpha) {
          sharedUniforms.uVideotronPortalAlpha.value = savedPortalAlpha;
        }
        _tF1Main0 = _tokS17 ? performance.now() : 0.0;
        AutogenBiomeSpawner.cullForMainAndShadow(
          biomeGroup,
          camera,
          weather.sunLight,
          shadowRefreshedThisFrame
        );
        postFX.render(scene, camera);
      } finally {
        scene.matrixWorldAutoUpdate = prevSceneAutoUpdate;
      }
      const _tF1Done = performance.now();
      lastCpuMs = _tF1Done - tCpuStart;
      lastDrawCalls = renderer.info.render.calls;

      if (!firstFrameMarked) {
        firstFrameMarked = true;
        const _f1SimMs = Math.max(0.0, _tF1Refl0 - tCpuStart);
        const _f1ReflMs = Math.max(0.0, _tF1Main0 - _tF1Refl0);
        const _f1MainMs = Math.max(0.0, _tF1Done - _tF1Main0);
        if (_tokS17) {
          bootProfiler.endStage(_tokS17, {
            cpuMs: lastCpuMs,
            detail: `${lastDrawCalls} calls · Sim ${_f1SimMs.toFixed(1)}ms + Refl ${_f1ReflMs.toFixed(1)}ms + Main ${_f1MainMs.toFixed(1)}ms`
          });
        }
        bootProfiler.recordFirstFrameSubmitted({
          simAndKinematicsCpuMs: _f1SimMs,
          planarReflectionCpuMs: _f1ReflMs,
          planarReflectionRan: reflRanThisFrame,
          planarReflectionCalls: reflCallsThisFrame,
          mainPostFxRenderCpuMs: _f1MainMs,
          shadowUpdated: shadowRefreshedThisFrame,
          mainCalls: postFX.lastWorldDrawCalls || lastDrawCalls,
          mainTriangles: postFX.lastWorldTriangles || renderer.info.render.triangles,
          totalCpuMs: lastCpuMs
        });
        if (typeof window !== 'undefined' && window.NexusForensic) {
          window.NexusForensic.markMilestone(
            'first_frame_rendered',
            `${lastDrawCalls} draw calls · ${renderer.info.render.triangles.toLocaleString('id-ID')} tris`
          );
        }
      }

      // Zero-GC 30-second rolling forensic blackbox sample
      forensicRecorder.recordRenderedFrame({
        frameMs: Math.max(0.5, rawWallSec * 1000.0),
        cpuMs: lastCpuMs,
        mainCalls: postFX.lastWorldDrawCalls || lastDrawCalls,
        mainTris: postFX.lastWorldTriangles || renderer.info.render.triangles,
        reflRan: reflRanThisFrame,
        reflCalls: reflCallsThisFrame,
        grassVisible: Boolean(grassField.mesh && grassField.mesh.visible),
        shadowUpdated: shadowRefreshedThisFrame,
        programsCount: Array.isArray(renderer.info?.programs) ? renderer.info.programs.length : 0,
        camMode: camDirector.mode,
        pilotKind: camDirector.lastFPSStatus?.vehicleKind || '',
        isDroneCinema: droneCinemaNow,
        posX: camera.position.x,
        posY: camera.position.y,
        posZ: camera.position.z,
        speedMS: Number(camDirector.lastSpeedMS || camDirector._avataSpeed || 0),
        dpr: hwGovernor.livePixelRatio || renderer.getPixelRatio()
      });

      // Smooth ~12 Hz UI slider & status refresh
      uiTransAccum += dt;
      if (uiTransAccum >= 0.085) {
        uiTransAccum = 0.0;
        syncTransitionButtonsUI();
        syncFPSStatusDOM();
        syncAcousticTelemetryDOM();
        syncGeoZoneWarningDOM();
        if (!arrivalActive) {
          permalinkDirty = true;
          syncPermalinkReplaceState(false);
        }
      }

      // Strictly 1 Hz Forensic GPU + 30s/60s Ring Buffer DOM update ONLY when Forensik tab is open (Zero Bloat!)
      forensicUiAccum += rawDt;
      if (forensicUiAccum >= 1.0) {
        forensicUiAccum = 0.0;
        if (isPanelExpanded && activeDrawerTab === 'audit') {
          syncForensicDOM();
        }
      }

      frameCount++;
      statTimer += rawDt;
      if (statTimer >= 0.5) {
        const fps = Math.round(frameCount / Math.max(0.001, statTimer));
        const calls = renderer.info.render.calls;
        if (elFps) elFps.textContent = `${fps} FPS`;
        if (elCalls) elCalls.textContent = `${calls} / 25`;
        if (elTris) elTris.textContent = renderer.info.render.triangles.toLocaleString('id-ID');
        syncHardwareTelemetryDOM();

        const assetReport = rektoratAsset.userData.gateReport;
        const sg1 = document.getElementById('sg1');
        const sg2 = document.getElementById('sg2');
        const sg3 = document.getElementById('sg3');
        const sg4 = document.getElementById('sg4');
        if (sg1) sg1.textContent = assetReport?.gate1_EnvelopeAndNaN ? 'PASS' : 'WARN';
        if (sg2) sg2.textContent = assetReport?.gate2_ManifoldNormals ? 'PASS' : 'WARN';
        if (sg3) sg3.textContent = assetReport?.gate3_ZeroGapAndKinematics ? 'PASS' : 'WARN';
        if (sg4) sg4.textContent = calls <= 25 ? 'PASS' : 'WARN';

        frameCount = 0;
        statTimer = 0.0;
      }
    } catch (frameErr) {
      console.error('[NEXUS RenderLoop Error]', frameErr);
    }
  }

  if (typeof window !== 'undefined') {
    window.__NEXUS_SIM__ = {
      scene,
      camera,
      renderer,
      postFX,
      grassField,
      weather,
      audioSynth,
      videotron,
      rotundaFountain,
      hwGovernor,
      forensicRecorder,
      bootProfiler,
      forensic: window.NexusForensic || null,
      camDirector,
      sivitasAI,
      solidWorld,
      rektoratAsset,
      sharedUniforms,
      presets,
      syncTransitionButtonsUI,
      syncFPSStatusDOM,
      syncAcousticTelemetryDOM,
      syncHardwareTelemetryDOM,
      syncForensicDOM,
      updateDepokTelemetryDOM,
      setLiveDepok(flag) {
        isLiveDepok = !!flag;
      },
      setSimClock(sec) {
        const t = Number(sec) || 0.0;
        elapsed = t;
        sharedUniforms.uTime.value = t;
      },
      stopArrival() {
        arrivalActive = false;
      },
      setGenshinMode(enabled) {
        setVisualInspectPreset(enabled ? 'GENSHIN' : '0', false);
      },
      buildPermalinkHash,
      getPermalinkURL,
      applyPermalinkFromURL,
      copyPermalinkToClipboard
    };
  }

  const _s15CpuMs = Math.max(0.0, _nowBoot() - _tS15Cpu0);
  const _s15Entry = bootProfiler.endStage(_tokS15, {
    cpuMs: _s15CpuMs,
    detail: `DOM/UI bind ${_s15CpuMs.toFixed(1)}ms (syncViewport ${bootProfiler.firstFrameGap.initialSyncViewportCpuMs}ms, FBO resize=${bootProfiler.firstFrameGap.postFxResizedOnViewportSync})`
  });
  if (_s15Entry) {
    bootProfiler.firstFrameGap.uiDomAndViewportBindCpuMs = _s15Entry.cpuMs;
  }
  _tokS16 = bootProfiler.beginStage(
    'S16_raf_wait_to_render_loop',
    'FIRST_FRAME_GAP',
    'Antrean Browser rAF Menuju Eksekusi Frame Pertama'
  );
  requestAnimationFrame(renderLoop);

  // --- ASYNCHRONOUS BACKGROUND REFERENCE PIPELINE (RUNS AFTER WORLD IS ALREADY LIVE!) ---
  // 1. Reconcile Live Open-Meteo Kota Depok Telemetry Asynchronously Whenever Network Responds
  liveWeatherPromise.then((liveData) => {
    if (!liveData) return;
    latestDepokWeather = liveData;
    updateDepokTelemetryDOM();
    if (isLiveDepok) {
      weather.setWeather(liveData.weatherMode, {
        immediate: elapsed < 0.35,
        cloudCoverNorm: liveData.cloudCoverNorm
      });
    }
    syncTransitionButtonsUI();
  });

  // 2. Stream & Bake 768x768 3D Rain Roof/Canopy Collision Z-Buffer & 1,600 Drip Anchors Asynchronously (< 1.5ms once per VSync frame, zero frame drops!)
  scheduleAsyncBackgroundTask(() => {
    bootProfiler.recordRainBakeStarted();
    const pBake = weather.bakeSceneCollisionFieldAsync(
      [worldRoot],
      {
        size: 768,
        sliceBudgetMs: 1.1,
        groundHeightFn,
        groundHeightWithCoastFn: TerrainDEMLoader.sampleGroundHeightWithCoast,
        freshwaterSurfaceYFn: TerrainDEMLoader.sampleWaterSurfaceY,
        shorelineXFn: TerrainDEMLoader.getShorelinePromenadeX,
        islandCoastDistanceFn: TerrainDEMLoader.getIslandCoastDistance,
        islandUCoastFn: TerrainDEMLoader.getIslandUCoastDistance,
        islandICoastFn: TerrainDEMLoader.getIslandICoastDistance,
        skipRootMatrixUpdate: true,
        waterLevelY: TerrainDEMLoader.WATER_LEVEL_Y
      },
      sharedUniforms
    );
    if (pBake && typeof pBake.then === 'function') {
      pBake.then((bakeRes) => {
        const finalBake = bakeRes || weather.collisionBake || null;
        bootProfiler.recordRainBakeCompleted(finalBake);
        if (typeof window !== 'undefined' && window.NexusForensic) {
          const bf = finalBake && finalBake.bakeForensics ? finalBake.bakeForensics : null;
          const det = bf
            ? `768x768 Z-buffer siap (CPU ${bf.totalCpuMs}ms · ${bf.sliceCount} slice · maks ${bf.maxSliceCpuMs}ms/slice · wall ${bf.wallElapsedMs}ms)`
            : '768x768 Z-buffer atap & kanopi siap (<1.5ms/frame)';
          window.NexusForensic.markMilestone('rain_collision_zbuffer_baked', det);
        }
      });
    }
  }, 45);
}

bootstrapNexusSimulator().catch((err) => {
  console.error('[NEXUS Bootstrap Error]', err);
  if (typeof window !== 'undefined' && typeof window.__showNexusBootError === 'function') {
    window.__showNexusBootError(err);
  }
});
