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
  MIST: 'Gerimis',
  RAIN: 'Hujan Deras',
  STORM: 'Badai Petir'
};

const TOD_LABELS = {
  DAY: 'Siang',
  DUSK: 'Sore',
  NIGHT: 'Malam'
};

const TOTAL_EXPECTED_TRIS = 5541566;

function yieldToBrowserFrame() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => setTimeout(resolve, 0));
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

  // Stage 1: Dynamic Parallel Import of NEXUS v3.1 Procedural Modules
  await preloader.advance(1, 20, 'Memuat Modul Mesin Prosedural NEXUS v3.1 & Shader Ultra...', 0);
  const [
    { CameraDirector, applyNonOccludingViewportOffset },
    { WeatherOpticsEngine },
    { AudioSynthesizer },
    { createSharedSurfaceUniforms },
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
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: 'high-performance'
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  // Unified in-shader HumanEyeHDROptics (Naka-Rushton + Hunt + Stevens + Purkinje) handles 100% of tone-mapping
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

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
  const weather = new WeatherOpticsEngine(scene);
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

  // Stage 5: Build Plaza Rotunda Makara UI, Jembatan Teksas UI, VIP Helipad (Pulau U), & Landasan Pacu RWY 36/18 (Pulau I)
  await preloader.advance(5, 70, 'Merakit Plaza Rotunda, Jembatan Teksas UI, Helipad VIP & Landasan Pacu Pulau I...', 1118000);
  const corridorGroup = SplineCorridorBuilder.buildRotundaAndPromenades(sharedUniforms);

  // Stage 6: Build Dense Southern Highland Rainforest (Pulau U), Sparse Montane Forest (Pulau I) & 44 Photocell Streetlamps
  await preloader.advance(6, 82, 'Menumbuhkan Hutan Lebat Pulau U, Lereng Gunung Pulau I, & 44 Lampu Sensor Cahaya...', 4899000);
  const biomeGroup = AutogenBiomeSpawner.buildCampusBiome(groundHeightFn, sharedUniforms);

  // Stage 7: Build 48 22-Joint Sivitas Akademika NPCs + Patrolling Electric Golf Cart + 48 Cats + Mini Yacht + 12 Marine Vessels
  await preloader.advance(7, 92, 'Menghadirkan 48 Sivitas Akademika, Mobil Golf UI, Mini Yacht Teluk U, & 48 Kucing...', TOTAL_EXPECTED_TRIS);
  const sivitasAI = new DynamicTrafficAI(groundHeightFn, sharedUniforms);
  const sivitasGroup = sivitasAI.build();

  worldRoot.add(terrainGroup, corridorGroup, biomeGroup, rektoratAsset, sivitasGroup);

  weather.onThunder = (distanceMeters, intensity) => {
    audioSynth.triggerThunder(distanceMeters, intensity);
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
  GedungRektoratUIAsset.setNightIllumination(rektoratAsset, weather.nightIlluminationLevel);
  sivitasAI.update(0.016, 0.016, camera.position);

  // Combine obstacle colliders: keep tower, annex wings, & pavilions solid while cushioning the 360° island shoreline
  const walkAndDroneColliders = [
    ...GedungRektoratUIAsset.meta.obstacleColliders.slice(1),
    {
      type: 'ISLAND_PERIMETER',
      cx: TerrainDEMLoader.ISLAND_CX,
      cz: TerrainDEMLoader.ISLAND_CZ,
      coastDistFn: TerrainDEMLoader.getIslandCoastDistance,
      bridgeDeckFn: TerrainDEMLoader.sampleTeksasBridgeDeckY,
      bridgeCenterX: TerrainDEMLoader.TEKSAS_BRIDGE_X,
      bridgeWalkHalfW: 2.35,
      margin: 0.95,
      minY: -10.0,
      maxY: 12.5
    }
  ];

  const sampleWalkableElevation = (x, z) => {
    const teksasDeckY = TerrainDEMLoader.sampleTeksasBridgeDeckY(x, z);
    if (teksasDeckY !== null) {
      return teksasDeckY + 0.02;
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
  camDirector.bindKeyboard(window);

  // Subtle 1.05s arrival ease-out on curtain lift (cancelled immediately on user interaction)
  let arrivalEaseClock = 0.0;
  const arrivalStartY = presets.oblique45.pos.y + 7.5;
  const arrivalEndY = presets.oblique45.pos.y;
  let arrivalActive = true;
  camera.position.y = arrivalStartY;
  controls.update();

  // Non-blocking Parallel GPU Shader Pre-Compilation (KHR_parallel_shader_compile) + 1 Warm-Up Render Frame
  if (typeof renderer.compileAsync === 'function') {
    await Promise.race([
      renderer.compileAsync(scene, camera).catch(() => {}),
      new Promise((resolve) => setTimeout(resolve, 2500))
    ]);
  } else if (typeof renderer.compile === 'function') {
    renderer.compile(scene, camera);
  }
  renderer.render(scene, camera);
  await preloader.advance(8, 100, 'Selamat Datang di Kepulauan Universitas Indonesia (Pulau U & I), Depok.', TOTAL_EXPECTED_TRIS);
  preloader.dismiss();

  // Pointer drag look in DRONE_6DOF and WALK_SIVITAS modes
  let dragging = false;
  let lastPX = 0;
  let lastPY = 0;
  renderer.domElement.addEventListener('pointerdown', (e) => {
    arrivalActive = false;
    camDirector.cancelTransition();
    dragging = true;
    lastPX = e.clientX;
    lastPY = e.clientY;
  });
  renderer.domElement.addEventListener('wheel', () => {
    arrivalActive = false;
    camDirector.cancelTransition();
  }, { passive: true });
  window.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    camDirector.lookDelta(e.clientX - lastPX, e.clientY - lastPY);
    lastPX = e.clientX;
    lastPY = e.clientY;
  });
  window.addEventListener('pointerup', () => {
    dragging = false;
  });

  // Non-Occluding Responsive UI Drawer (Default Collapsed — Hukum Emas 9)
  const pillBtn = document.getElementById('sim-pill-toggle');
  const pillLabelEl = document.getElementById('sim-pill-label');
  const drawer = document.getElementById('sim-drawer');
  const closeBtn = document.getElementById('sim-drawer-close');
  let isPanelExpanded = false;

  function syncViewport() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h);
    const rect = drawer.getBoundingClientRect();
    applyNonOccludingViewportOffset(camera, w, h, isPanelExpanded, {
      width: rect.width || 276,
      height: rect.height || Math.min(h * 0.32, h * 0.34),
      side: 'right'
    });
  }

  function setPanelExpanded(expanded) {
    isPanelExpanded = expanded;
    drawer.classList.toggle('collapsed', !expanded);
    drawer.setAttribute('aria-expanded', String(expanded));
    pillBtn.classList.toggle('hidden', expanded);
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

  // Tri-Angle & Live Patrol Camera Presets
  document.querySelectorAll('[data-preset]').forEach((btn) => {
    btn.addEventListener('click', () => {
      arrivalActive = false;
      document.querySelectorAll('[data-preset]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('[data-cammode]').forEach((b) =>
        b.classList.toggle('active', b.dataset.cammode === 'ORBIT')
      );
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
          return;
        }
      }
      camDirector.applyOrthoPreset(btn.dataset.preset, presets);
    });
  });

  // Exploration Camera Modes
  document.querySelectorAll('[data-cammode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      arrivalActive = false;
      document.querySelectorAll('[data-cammode]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      camDirector.setMode(btn.dataset.cammode);
    });
  });

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
      syncTransitionButtonsUI();
    });
  });

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

    for (const btn of buttons) {
      const code = btn.dataset[attrName];
      const subEl = btn.querySelector('.btn-sub');
      btn.classList.remove('active', 'is-transitioning', 'is-departing');
      btn.style.setProperty('--trans-pct', '0%');
      if (subEl) subEl.textContent = '';

      if (axisState.active) {
        if (code === axisState.toMode || code === axisState.toTod) {
          btn.classList.add('is-transitioning');
          btn.style.setProperty('--trans-pct', `${pct}%`);
          if (subEl) subEl.textContent = `${pct}% · ${remSec} dtk`;
        } else if (
          (code === axisState.fromMode || code === axisState.fromTod) &&
          code !== (axisState.toMode || axisState.toTod)
        ) {
          btn.classList.add('is-departing');
          if (subEl) subEl.textContent = `Mereda ${Math.max(0, 100 - pct)}%`;
        }
      } else if (code === (axisState.toMode || axisState.toTod)) {
        btn.classList.add('active');
      }
    }
  }

  function syncTransitionButtonsUI() {
    const ts = weather.getTransitionState();
    renderAxisButtonStates(weatherButtons, 'weather', ts.weather);
    renderAxisButtonStates(todButtons, 'tod', ts.time);

    const anyTrans = ts.weather.active || ts.time.active;
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

  // Web Audio Soundscape (Continuously Coupled to Live 20–45s Transition Uniforms)
  const audioBtn = document.getElementById('btn-audio');
  audioBtn.addEventListener('click', () => {
    const active = audioSynth.toggle();
    audioSynth.syncContinuousWeatherAudio(
      weather.uniforms.uRainIntensity.value,
      weather.uniforms.uStormFactor.value,
      weather.uniforms.uWindVec.value.length()
    );
    const titleSpan = audioBtn.querySelector('.btn-title') || audioBtn;
    titleSpan.textContent = active ? 'Suara: ON' : 'Suara: OFF';
    audioBtn.classList.toggle('active', active);
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
    const dt = Math.min(0.05, (now - prevTime) * 0.001);
    prevTime = now;
    elapsed += dt;

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

    // Continuous C^1 Architectural Window, Garden Lamp, & Helipad Night Illumination + Live Audio Coupling
    GedungRektoratUIAsset.setNightIllumination(rektoratAsset, weather.nightIlluminationLevel);
    audioSynth.syncContinuousWeatherAudio(
      weather.uniforms.uRainIntensity.value,
      weather.uniforms.uStormFactor.value,
      weather.uniforms.uWindVec.value.length()
    );

    camDirector.update(dt, sampleWalkableElevation);
    GedungRektoratUIAsset.updateKinematics(rektoratAsset, dt, {
      isNight: weather.nightIlluminationLevel > 0.35,
      nightLevel: weather.nightIlluminationLevel,
      windMultiplier: weather.getWindGustMultiplier()
    });
    sivitasAI.update(dt, elapsed, camera.position);

    renderer.render(scene, camera);

    // Smooth ~12 Hz UI button progress bar & countdown refresh
    uiTransAccum += dt;
    if (uiTransAccum >= 0.085) {
      uiTransAccum = 0.0;
      syncTransitionButtonsUI();
    }

    frameCount++;
    statTimer += dt;
    if (statTimer >= 0.5) {
      const fps = Math.round(frameCount / statTimer);
      const calls = renderer.info.render.calls;
      elFps.textContent = `${fps} FPS`;
      elCalls.textContent = `${calls} / 25`;
      elTris.textContent = renderer.info.render.triangles.toLocaleString('id-ID');

      const assetReport = rektoratAsset.userData.gateReport;
      document.getElementById('sg1').textContent = assetReport?.gate1_EnvelopeAndNaN ? 'PASS' : 'WARN';
      document.getElementById('sg2').textContent = assetReport?.gate2_ManifoldNormals ? 'PASS' : 'WARN';
      document.getElementById('sg3').textContent = assetReport?.gate3_ZeroGapAndKinematics ? 'PASS' : 'WARN';
      document.getElementById('sg4').textContent = calls <= 25 ? 'PASS' : 'WARN';

      frameCount = 0;
      statTimer = 0.0;
    }
  }

  if (typeof window !== 'undefined') {
    window.__NEXUS_SIM__ = {
      scene,
      camera,
      renderer,
      weather,
      camDirector,
      sivitasAI,
      rektoratAsset,
      sharedUniforms,
      presets,
      syncTransitionButtonsUI,
      updateDepokTelemetryDOM,
      setLiveDepok(flag) {
        isLiveDepok = !!flag;
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

bootstrapNexusSimulator();
