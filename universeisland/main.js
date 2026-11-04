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

const TOTAL_EXPECTED_TRIS = 4469864;

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
  const preloader = createPreloaderController();

  // Stage 0: Instant Local Depok WIB Clock + Fire 100% Asynchronous Open-Meteo Telemetry (0ms Blocking Wait!)
  await preloader.advance(0, 10, 'Mengaktifkan Jam WIB Depok & Sinkronisasi Cuaca Asinkron...', 0);
  const { DepokLiveWeatherService } = await import('./core/DepokLiveWeatherService.js');

  let latestDepokClock = DepokLiveWeatherService.getDepokWIBClock();
  let latestDepokWeather = DepokLiveWeatherService.buildOfflineDepokFallback();
  let liveWeatherSettled = false;

  preloader.setWeatherBadge(
    `${latestDepokClock.formattedHM} WIB · ${latestDepokWeather.labelId.replace(' (Estimasi Lokal)', '')} (Sinkron Asinkron...)`
  );

  // Reference weather telemetry streams asynchronously in background — NEVER blocks world loading!
  const liveWeatherPromise = DepokLiveWeatherService.fetchLiveDepokWeather({ timeoutMs: 4500 }).then(
    (data) => {
      latestDepokWeather = data;
      liveWeatherSettled = true;
      preloader.setWeatherBadge(
        `${data.labelId.replace(' (Estimasi Lokal)', '')} · ${data.temperatureC}°C · RH ${data.humidityPct}%`
      );
      return data;
    }
  );

  // Stage 1: Dynamic Parallel Import of NEXUS v5.0 Procedural Modules
  await preloader.advance(1, 20, 'Memuat Modul Mesin Prosedural NEXUS v5.0 & Shader Ultra...', 0);
  const [
    { CameraDirector, applyNonOccludingViewportOffset },
    { WeatherOpticsEngine },
    { AudioSynthesizer },
    { createSharedSurfaceUniforms },
    { VideotronYouTubeEngine },
    { RotundaDancingFountain },
    { HardwareTelemetryGovernor },
    { FloatingOriginManager },
    { TerrainDEMLoader },
    { SplineCorridorBuilder },
    { AutogenBiomeSpawner },
    { DynamicTrafficAI },
    { GedungRektoratUIAsset }
  ] = await Promise.all([
    import('./core/CameraDirector.js'),
    import('./core/WeatherOpticsEngine.js'),
    import('./core/AudioSynthesizer.js'),
    import('./core/ProceduralTextureFactory.js'),
    import('./core/VideotronYouTubeEngine.js'),
    import('./core/RotundaDancingFountain.js'),
    import('./core/HardwareTelemetryGovernor.js'),
    import('./world-gis/FloatingOriginManager.js'),
    import('./world-gis/TerrainDEMLoader.js'),
    import('./world-gis/SplineCorridorBuilder.js'),
    import('./world-gis/AutogenBiomeSpawner.js'),
    import('./world-gis/DynamicTrafficAI.js'),
    import('./library/landmark/Gedung_Rektorat_UI/index.js')
  ]);

  // Stage 2: WebGL2 Human-Eye Physiological HDR Renderer, Instant RTC Spatial Anchor, & PMREM Sky IBL
  await preloader.advance(2, 30, 'Menginisialisasi WebGL2 Human-Eye Photoreceptor HDR & Kubah Langit...', 28856);
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
  renderer.domElement.addEventListener(
    'webglcontextlost',
    (e) => {
      e.preventDefault();
      webglContextLost = true;
    },
    false
  );
  renderer.domElement.addEventListener(
    'webglcontextrestored',
    () => {
      webglContextLost = false;
      renderer.shadowMap.needsUpdate = true;
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

  const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.18, 4200);
  const presets = GedungRektoratUIAsset.meta.orthoValidationPresets;
  camera.position.set(presets.oblique45.pos.x, presets.oblique45.pos.y, presets.oblique45.pos.z);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  // Allow tilting upward into the night sky (Moon, Milky Way, Constellations); CameraDirector soft floor prevents underground clip!
  controls.maxPolarAngle = Math.PI * 0.86;
  controls.minDistance = 0.55;
  controls.maxDistance = 1550;
  controls.target.set(presets.oblique45.target.x, presets.oblique45.target.y, presets.oblique45.target.z);
  controls.update();

  const sharedUniforms = createSharedSurfaceUniforms();
  const hwGovernor = new HardwareTelemetryGovernor(renderer, sharedUniforms);
  const weather = new WeatherOpticsEngine(scene, {
    shadowMapSize: hwGovernor.getActiveTierConfig().shadowMapSize
  });
  weather.attachRenderer(renderer);
  const waterMaterial = weather.createDanauKenangaWaterMaterial();
  const audioSynth = new AudioSynthesizer();

  // Stage 3: Build Gedung Rektorat UI Class-A Monument (8-Sided Tower + 3-Tier Tajug Pagoda)
  await preloader.advance(3, 44, 'Menyusun Menara Segi-8 & Atap Tajug 3-Tumpang Rektorat UI...', 377118);
  const rektoratAsset = GedungRektoratUIAsset.build({ sharedUniforms });

  // Stage 4: Build Kepulauan UI (Mata Air, Sungai Berkelok, Danau Kecil, & Air Terjun Niagara Mini)
  await preloader.advance(4, 58, 'Membentuk Mata Air, Sungai Berkelok Pulau U & Air Terjun Niagara Mini Pulau I...', 1002000);
  const { group: terrainGroup, groundHeightFn } = TerrainDEMLoader.buildCampusTerrain(
    waterMaterial,
    sharedUniforms
  );

  // Stage 5: Build Plaza Rotunda, Kompleks DPPD–ILRC–STP, Air Mancur Dancing, Jogging/Calisthenics, Jembatan Teksas UI, Underpass, & Dermaga
  await preloader.advance(5, 70, 'Merakit Kompleks DPPD–STP, Air Mancur Dancing, Jogging Track & Jaringan Jalan Pulau U...', 1205000);
  const corridorGroup = SplineCorridorBuilder.buildRotundaAndPromenades(sharedUniforms);
  const rotundaFountain = new RotundaDancingFountain();

  // Stage 6: Build Dense Southern Highland Rainforest (Pulau U), Sparse Montane Forest (Pulau I) & 64 Photocell Streetlamps
  await preloader.advance(6, 82, 'Menumbuhkan Hutan Lebat Pulau U, Lereng Gunung Pulau I, & 64 Lampu Sensor Cahaya...', 5020000);
  const biomeGroup = AutogenBiomeSpawner.buildCampusBiome(groundHeightFn, sharedUniforms);
  const solidWorld = SplineCorridorBuilder.buildWorldSolidColliders(biomeGroup);

  // Stage 7: Build 69 22-Joint Sivitas Akademika NPCs + Taxi Heli Listrik eVTOL + Cessna & Learjet 3x Sehari + Bus Kuning + Feri + 48 Cats
  await preloader.advance(7, 92, 'Menghadirkan 69 Sivitas Akademika, Taxi Heli Listrik, Pesawat 3x Sehari, & 48 Kucing...', TOTAL_EXPECTED_TRIS);
  const sivitasAI = new DynamicTrafficAI(groundHeightFn, sharedUniforms);
  sivitasAI.setSolidWorld(solidWorld);
  const sivitasGroup = sivitasAI.build();

  worldRoot.add(terrainGroup, corridorGroup, rotundaFountain.mesh, biomeGroup, rektoratAsset, sivitasGroup);

  weather.onThunder = (distanceMeters, intensity) => {
    audioSynth.triggerThunder(distanceMeters, intensity, weather.uniforms.uLightningDir.value);
  };

  // Stage 8: Apply Depok WIB Clock & Real-Time Ephemeris Immediately & Compile Specialized GPU Shaders Asynchronously
  await preloader.advance(8, 97, 'Kompilasi Paralel Shader GPU & Membuka Kepulauan UI...', TOTAL_EXPECTED_TRIS);

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

  // Non-blocking Parallel GPU Shader Pre-Compilation (KHR_parallel_shader_compile on Desktop; deterministic synchronous compile on Mobile)
  await yieldToBrowserFrame();
  if (!hwGovernor.isMobileDevice && typeof renderer.compileAsync === 'function') {
    await Promise.race([
      renderer.compileAsync(scene, camera).catch(() => {}),
      new Promise((resolve) => setTimeout(resolve, 2500))
    ]);
  } else if (typeof renderer.compile === 'function') {
    renderer.compile(scene, camera);
  }
  videotron.update(camera, window.innerWidth, window.innerHeight, sharedUniforms);
  const tWarm0 = performance.now();
  renderer.render(scene, camera);
  hwGovernor.recordWarmupSample(performance.now() - tWarm0);
  await preloader.advance(8, 100, 'Selamat Datang di Kepulauan Universitas Indonesia (Pulau U & I), Depok.', TOTAL_EXPECTED_TRIS);
  preloader.dismiss();

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

  function setPanelExpanded(expanded) {
    if (expanded) dismissWelcomeToast();
    isPanelExpanded = expanded;
    drawer.classList.toggle('collapsed', !expanded);
    drawer.setAttribute('aria-expanded', String(expanded));
    pillBtn.classList.toggle('hidden', expanded);
    if (expanded && document.pointerLockElement === renderer.domElement && document.exitPointerLock) {
      document.exitPointerLock();
    }
    syncQuickDockAndContextDOM();
    requestAnimationFrame(syncViewport);
  }

  pillBtn.addEventListener('click', () => setPanelExpanded(true));
  closeBtn.addEventListener('click', () => setPanelExpanded(false));
  window.addEventListener('resize', syncViewport);
  syncViewport();

  // Tabs
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
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

    if (camDirector.mode === targetMode && camDirector.isFPSMode(targetMode) && !opts.forceReset) {
      camDirector.cycleFPSSubject(1);
    } else if (camDirector.mode === 'DRONE_AVATA2_FPV' && targetMode === 'DRONE_AVATA2_FPV' && !opts.fromQuickDock) {
      camDirector.setAvataAutoFreestyle(!camDirector.avataAutoFreestyle);
    } else if (targetMode === 'DRONE_6DOF') {
      camDirector.setMode('DRONE_6DOF', { startIslandTour: true, ...opts });
    } else {
      camDirector.setMode(targetMode, opts);
    }

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

  // Live FPS Telemetry Status Card & Subject Cycle Controls
  const fpsLiveCardEl = document.getElementById('fps-live-card');
  const fpsLiveTitleEl = document.getElementById('fps-live-title');
  const fpsLiveSubEl = document.getElementById('fps-live-sub');
  const fpsPrevBtn = document.getElementById('btn-fps-prev-subject');
  const fpsNextBtn = document.getElementById('btn-fps-next-subject');
  const fpsResetBtn = document.getElementById('btn-fps-reset-look');

  function syncFPSStatusDOM() {
    syncAvataHudDOM();
    syncQuickDockAndContextDOM();
    if (!fpsLiveCardEl) return;
    const active = camDirector.isFPSMode();
    fpsLiveCardEl.style.display = active ? 'flex' : 'none';
    if (!active) return;
    const st = camDirector.lastFPSStatus || sivitasAI.getFPSAnchor(camDirector.mode, camDirector.fpsSubjectOffset);
    if (st) {
      if (fpsLiveTitleEl) fpsLiveTitleEl.textContent = st.label || camDirector.mode;
      if (fpsLiveSubEl) fpsLiveSubEl.textContent = st.subLabel || 'Geser layar untuk menoleh';
    }
  }
  syncFPSStatusDOM();

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
      presetBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
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
          syncFPSStatusDOM();
          requestAnimationFrame(syncViewport);
          return;
        }
      }
      if (btn.dataset.preset === 'underpassBawahLautInterior' && camera.position.y > -2.0) {
        camDirector.applyOrthoPreset(btn.dataset.preset, presets, { immediate: true });
        syncFPSStatusDOM();
        requestAnimationFrame(syncViewport);
        return;
      }
      camDirector.applyOrthoPreset(btn.dataset.preset, presets);
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

  // --- LIVE KAMPUS UI DEPOK TELEMETRY & 20–45s SMOOTH TRANSITION CONTROLS ---
  const liveCardEl = document.getElementById('live-depok-card');
  const liveBadgeTextEl = document.getElementById('ui-depok-badge-text');
  const liveClockEl = document.getElementById('ui-depok-clock');
  const liveWeatherEl = document.getElementById('ui-depok-weather');
  const liveHvsEyeEl = document.getElementById('ui-hvs-eye');
  const liveAstroEl = document.getElementById('ui-astro-moon');
  const liveBtnEl = document.getElementById('btn-live-depok');
  const sliderDurEl = document.getElementById('slider-trans-dur');
  const labelDurEl = document.getElementById('label-trans-dur');
  const activeTodPillEl = document.getElementById('ui-active-tod-pill');
  const activeWeatherPillEl = document.getElementById('ui-active-weather-pill');
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
        const tsNow = weather.getTransitionState();
        liveWeatherEl.textContent =
          `Cuaca Simulasi: ${WEATHER_LABELS[tsNow.weather.toMode]} · ` +
          `Rill Depok: ${latestDepokClock.formattedHM} (${latestDepokWeather.temperatureC}°C)`;
      }
    }
  }
  updateDepokTelemetryDOM();

  // Transition Duration Slider (Strictly 20s – 45s)
  if (sliderDurEl) {
    sliderDurEl.addEventListener('input', () => {
      const clamped = weather.setTransitionDuration(Number(sliderDurEl.value));
      if (labelDurEl) labelDurEl.textContent = `${Math.round(clamped)} dtk`;
    });
  }

  // Manual Time of Day Override -> Mandatory 20–45s Diurnal Sweep (Zero Sudden Day/Night Jump!)
  todButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      isLiveDepok = false;
      weather.setTimeOfDay(btn.dataset.tod, sharedUniforms);
      syncTransitionButtonsUI();
    });
  });

  // Manual Weather Override -> Mandatory 20–45s Staged Meteorological Choreography!
  weatherButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      isLiveDepok = false;
      weather.setWeather(btn.dataset.weather);
      audioSynth.playUiChime(560);
      syncTransitionButtonsUI();
    });
  });

  // 1-Click Cinematic Mood Combos (Golden Dusk, Night Storm, Post-Rain Rainbow, Misty Dawn)
  moodButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      isLiveDepok = false;
      if (btn.dataset.moodTod) weather.setTimeOfDay(btn.dataset.moodTod, sharedUniforms);
      if (btn.dataset.moodW) weather.setWeather(btn.dataset.moodW);
      audioSynth.playUiChime(620);
      syncTransitionButtonsUI();
    });
  });

  // Optional Instant Transition Skip (When User Wants Immediate Weather/Time Inspection)
  if (skipTransBtn) {
    skipTransBtn.addEventListener('click', () => {
      const ts = weather.getTransitionState();
      if (ts.time.active) {
        weather.setTimeOfDay(ts.time.toTod, sharedUniforms, { immediate: true });
      }
      if (ts.weather.active) {
        weather.setWeather(ts.weather.toMode, { immediate: true });
      }
      audioSynth.playUiChime(700);
      syncTransitionButtonsUI();
    });
  }

  // Return to Live Kampus UI Depok Clock & Weather (Smooth 20–45s Transition!)
  if (liveBtnEl) {
    liveBtnEl.addEventListener('click', () => {
      isLiveDepok = true;
      latestDepokClock = DepokLiveWeatherService.getDepokWIBClock();
      updateDepokTelemetryDOM();
      weather.setReferenceAstronomyDate(new Date(), sharedUniforms);
      weather.setSolarHour(latestDepokClock.solarHour, sharedUniforms);
      weather.setWeather(latestDepokWeather.weatherMode, {
        cloudCoverNorm: latestDepokWeather.cloudCoverNorm
      });
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

    if (activeTodPillEl) {
      const simClk = DepokLiveWeatherService.formatSolarHourClock(weather.currentSolarHour);
      const todLbl = TOD_LABELS[ts.time.toTod] || simClk.periodLabelId;
      if (ts.time.active) {
        const remT = Math.max(1, Math.ceil(ts.time.remainingSec));
        activeTodPillEl.textContent = `\u2192 ${todLbl} · ${remT}d`;
      } else {
        activeTodPillEl.textContent = `${todLbl} · ${simClk.formattedHM.replace(' WIB', '')}`;
      }
    }

    if (activeWeatherPillEl) {
      const wLbl = WEATHER_LABELS[ts.weather.toMode] || ts.weather.toMode;
      if (ts.weather.active) {
        const remW = Math.max(1, Math.ceil(ts.weather.remainingSec));
        activeWeatherPillEl.textContent = `\u2192 ${wLbl} · ${remW}d`;
      } else {
        activeWeatherPillEl.textContent = wLbl;
      }
    }

    const anyTrans = ts.weather.active || ts.time.active;
    if (skipTransBtn) {
      skipTransBtn.classList.toggle('visible', Boolean(anyTrans));
    }
    const dominantAxis =
      ts.weather.active && ts.time.active
        ? ts.weather.remainingSec >= ts.time.remainingSec
          ? {
              pct: Math.round(ts.weather.progress * 100),
              rem: Math.max(1, Math.ceil(ts.weather.remainingSec)),
              targetLabel: `${WEATHER_LABELS[ts.weather.toMode]} & ${TOD_LABELS[ts.time.toTod]}`
            }
          : {
              pct: Math.round(ts.time.progress * 100),
              rem: Math.max(1, Math.ceil(ts.time.remainingSec)),
              targetLabel: `${TOD_LABELS[ts.time.toTod]} & ${WEATHER_LABELS[ts.weather.toMode]}`
            }
        : ts.weather.active
          ? {
              pct: Math.round(ts.weather.progress * 100),
              rem: Math.max(1, Math.ceil(ts.weather.remainingSec)),
              targetLabel: WEATHER_LABELS[ts.weather.toMode]
            }
          : ts.time.active
            ? {
                pct: Math.round(ts.time.progress * 100),
                rem: Math.max(1, Math.ceil(ts.time.remainingSec)),
                targetLabel: TOD_LABELS[ts.time.toTod]
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
        if (liveBadgeTextEl) liveBadgeTextEl.textContent = 'MODE SIMULASI MANUAL';
        if (titleEl) titleEl.textContent = '↻ Kembali ke Waktu & Cuaca Rill Depok';
        if (subEl) subEl.textContent = `Transisi halus ${Math.round(ts.durationSec)} detik`;
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
        pillLabelEl.textContent = `Rektorat UI · ${TOD_LABELS[ts.time.toTod]} · ${WEATHER_LABELS[ts.weather.toMode]}`;
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

  // SSoST Zebra Inspection Mode
  document.querySelectorAll('[data-inspect]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-inspect]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      sharedUniforms.uZebraInspectMode.value = Number(btn.dataset.inspect);
    });
  });

  // 3D Binaural HRTF & Physical Acoustic Engine (Unified Across Drawer, Quick Dock, & Avata 2 HUD)
  const audioBtn = document.getElementById('btn-audio');
  const qdockAudioBtn = document.getElementById('btn-qdock-audio');
  const avataAudioBtn = document.getElementById('btn-avata-audio');
  const audioVolSlider = document.getElementById('slider-audio-vol');
  const audioVolLabel = document.getElementById('label-audio-vol');
  const acousticStatusEl = document.getElementById('ui-acoustic-status');
  const audioMixBtns = Array.from(document.querySelectorAll('[data-audiomix]'));

  function updateAcousticWorldNow(dtStep) {
    camera.updateMatrixWorld();
    const apasActive = (camDirector._avataApasTimer || 0) > 0.01;
    const inCorridor = (camDirector._avataMeta?.corridorWeight || 0) > 0.35;
    audioSynth.updateAcousticWorld(dtStep, {
      camera,
      camMode: camDirector.mode,
      isRoofSheltered: DynamicTrafficAI._isShelteredByRoof(camera.position.x, camera.position.z),
      solarHour: weather.currentSolarHour,
      temperatureC: latestDepokWeather?.temperatureC,
      humidityPct: latestDepokWeather?.humidityPct,
      rainIntensity: weather.uniforms.uRainIntensity.value,
      stormFactor: weather.uniforms.uStormFactor.value,
      windSpeed: weather.uniforms.uWindVec.value.length(),
      wetness: weather.uniforms.uWetness.value,
      nightFactor: weather.uniforms.uNightFactor?.value ?? weather.nightIlluminationLevel,
      lightningDir: weather.uniforms.uLightningDir.value,
      fountainNightBlend: rotundaFountain.nightBlend || 0.0,
      coastDist: TerrainDEMLoader.getIslandCoastDistance(camera.position.x, camera.position.z),
      riverZ: TerrainDEMLoader.evalPulauURiverCenterZ(camera.position.x),
      videotronPlaying: Boolean(videotron && !videotron.isMuted && videotron.isPlaying),
      sivitasAI,
      avataThrottle01: camDirector._avataThrottle01 || 0.0,
      avataTurnRate: Math.abs(camDirector._avataYawRateSmooth || 0.0),
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
        const focusTxt = tel.activeSfxBanner
          ? `SFX: ${tel.activeSfxBanner}`
          : `${tel.dominantSourceLabel}${doppTxt}`;
        acousticStatusEl.textContent = `${contextLabel} \u00B7 Gema ${tel.reverbSendPct}% \u00B7 ${focusTxt}`;
        acousticStatusEl.style.color = 'var(--ok-green)';
      } else {
        acousticStatusEl.textContent = `HRTF 3D \u00B7 ${contextLabel} \u00B7 Klik Suara / G`;
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
    }
  });

  // Hardware Profiler & Adaptive 120-FPS Governor Controls
  const elHwTier = document.getElementById('sim-hw-tier');
  const elHwDetail = document.getElementById('sim-hw-detail');
  const elHwGpuName = document.getElementById('sim-hw-gpu-name');

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
  syncHardwareTelemetryDOM();

  document.querySelectorAll('[data-hwmode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-hwmode]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      hwGovernor.setMode(btn.dataset.hwmode);
      syncHardwareTelemetryDOM();
    });
  });

  // Main 120 FPS Ultra Render & Telemetry Loop
  const elFps = document.getElementById('sim-fps');
  const elCalls = document.getElementById('sim-calls');
  const elTris = document.getElementById('sim-tris');

  let prevTime = performance.now();
  let elapsed = 0.0;
  let frameCount = 0;
  let statTimer = 0.0;
  let clockTickAccum = 0.0;
  let uiTransAccum = 0.0;

  function renderLoop(now) {
    requestAnimationFrame(renderLoop);
    if (webglContextLost) return;
    const dt = Math.min(0.05, Math.max(0.001, (now - prevTime) * 0.001));
    prevTime = now;
    elapsed += dt;

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

      // Continuous C^1 Architectural Window, Garden Lamp, & Helipad Night Illumination + 3D HRTF Physical Acoustic Coupling
      GedungRektoratUIAsset.setNightIllumination(rektoratAsset, weather.nightIlluminationLevel);

      sivitasAI.update(dt, elapsed, camera.position);
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

      hwGovernor.update(dt, camera, controls.enabled ? controls.target : null, weather);
      videotron.update(camera, window.innerWidth, window.innerHeight, sharedUniforms);
      renderer.render(scene, camera);

      // Smooth ~12 Hz UI button progress bar & countdown refresh
      uiTransAccum += dt;
      if (uiTransAccum >= 0.085) {
        uiTransAccum = 0.0;
        syncTransitionButtonsUI();
        syncFPSStatusDOM();
        syncAcousticTelemetryDOM();
      }

      frameCount++;
      statTimer += dt;
      if (statTimer >= 0.5) {
        const fps = Math.round(frameCount / statTimer);
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
      weather,
      audioSynth,
      videotron,
      rotundaFountain,
      hwGovernor,
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
      }
    };
  }

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

  // 2. Stream & Bake 512x512 3D Rain Roof/Canopy Collision Z-Buffer & 1,600 Drip Anchors Asynchronously (< 6ms/frame slices!)
  scheduleAsyncBackgroundTask(() => {
    weather.bakeSceneCollisionFieldAsync(
      [worldRoot],
      {
        groundHeightFn,
        freshwaterSurfaceYFn: TerrainDEMLoader.sampleWaterSurfaceY,
        shorelineXFn: TerrainDEMLoader.getShorelinePromenadeX,
        islandCoastDistanceFn: TerrainDEMLoader.getIslandCoastDistance,
        waterLevelY: TerrainDEMLoader.WATER_LEVEL_Y
      },
      sharedUniforms
    );
  }, 45);
}

bootstrapNexusSimulator().catch((err) => {
  console.error('[NEXUS Bootstrap Error]', err);
  if (typeof window !== 'undefined' && typeof window.__showNexusBootError === 'function') {
    window.__showNexusBootError(err);
  }
});
