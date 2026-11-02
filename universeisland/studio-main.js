import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.9 — Live Kampus UI Depok & 20–45s Smooth Transitions)
 * ./studio-main.js — Class-A 3D Inspection Workbench Orchestrator for studio.html
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

async function bootstrapStudioWorkbench() {
  const [
    { DepokLiveWeatherService },
    { CameraDirector, applyNonOccludingViewportOffset },
    { PhysicsKernel },
    { WeatherOpticsEngine },
    { AudioSynthesizer },
    {
      SURFACE_PROFILE,
      createSharedSurfaceUniforms,
      createHorizontalStonePaverTexture,
      injectOmniScaleSurfaceShader
    },
    { GedungRektoratUIAsset },
    { SplineCorridorBuilder }
  ] = await Promise.all([
    import('./core/DepokLiveWeatherService.js'),
    import('./core/CameraDirector.js'),
    import('./core/PhysicsKernel.js'),
    import('./core/WeatherOpticsEngine.js'),
    import('./core/AudioSynthesizer.js'),
    import('./core/ProceduralTextureFactory.js'),
    import('./library/landmark/Gedung_Rektorat_UI/index.js'),
    import('./world-gis/SplineCorridorBuilder.js')
  ]);

  const liveWeatherPromise = DepokLiveWeatherService.fetchLiveDepokWeather({ timeoutMs: 3600 });

  const wrap = document.getElementById('studio-canvas-wrap');
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  // Unified in-shader HumanEyeHDROptics handles 100% of physiological photoreceptor HDR tone-mapping
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  wrap.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0xa8c6df, 0.0);

  const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.35, 1100);
  const presets = GedungRektoratUIAsset.meta.orthoValidationPresets;
  camera.position.set(presets.oblique45.pos.x, presets.oblique45.pos.y, presets.oblique45.pos.z);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.maxPolarAngle = Math.PI * 0.492;
  controls.minDistance = 14;
  controls.maxDistance = 260;
  controls.target.set(presets.oblique45.target.x, presets.oblique45.target.y, presets.oblique45.target.z);
  controls.update();

  const sharedUniforms = createSharedSurfaceUniforms();
  const weather = new WeatherOpticsEngine(scene);
  weather.attachRenderer(renderer);
  const audioSynth = new AudioSynthesizer();

  // Calibrated High-Detail 224-Segment Studio Turntable Plinth supporting the 108 x 124 m Complex (1 Draw Call)
  const plinthGeo = new THREE.CylinderGeometry(86, 89, 0.6, 224, 6);
  plinthGeo.translate(0, -0.3, 10.0);
  const plinthMat = injectOmniScaleSurfaceShader(
    new THREE.MeshStandardMaterial({
      color: 0x62666c,
      map: createHorizontalStonePaverTexture(),
      roughness: 0.86,
      metalness: 0.05,
      envMapIntensity: 0.12
    }),
    {
      ySegmentStart: 120.0,
      segmentCount: 0.0,
      surfaceProfile: SURFACE_PROFILE.UI_DIAMOND_PLAZA,
      bumpScale: 0.72,
      sharedUniforms
    }
  );
  const plinthMesh = new THREE.Mesh(plinthGeo, plinthMat);
  plinthMesh.name = 'StudioInspectionPlinth';
  plinthMesh.receiveShadow = true;
  scene.add(plinthMesh);

  // Build Verified Class-A Asset
  const rektoratAsset = GedungRektoratUIAsset.build({ sharedUniforms });
  scene.add(rektoratAsset);

  const studioGroundFn = (x, z) => (Math.hypot(x, z - 10.0) <= 88.5 ? 0.0 : -5.5);

  // Bake 512x512 3D Roof/Eave/Plinth Collision Z-Buffer & Extract Exact 3D Eave Drip Anchors
  weather.bakeSceneCollisionField(
    [plinthMesh, rektoratAsset],
    {
      groundHeightFn: studioGroundFn
    },
    sharedUniforms
  );
  weather.onThunder = (distanceMeters, intensity) => {
    audioSynth.triggerThunder(distanceMeters, intensity);
  };

  // Boot into Real-Time Kampus UI Depok WIB Clock & Live Weather by Default
  let isLiveDepok = true;
  let latestDepokClock = DepokLiveWeatherService.getDepokWIBClock();
  let latestDepokWeather = await Promise.race([
    liveWeatherPromise,
    new Promise((resolve) =>
      setTimeout(() => resolve(DepokLiveWeatherService.buildOfflineDepokFallback()), 1400)
    )
  ]);

  weather.setReferenceAstronomyDate(new Date(), sharedUniforms);
  weather.setSolarHour(latestDepokClock.solarHour, sharedUniforms, { immediate: true });
  weather.setWeather(latestDepokWeather.weatherMode, {
    immediate: true,
    cloudCoverNorm: latestDepokWeather.cloudCoverNorm
  });
  weather.update(0.016, camera, sharedUniforms);
  GedungRektoratUIAsset.setNightIllumination(rektoratAsset, weather.nightIlluminationLevel);
  if (typeof renderer.compile === 'function') {
    renderer.compile(scene, camera);
  }

  const REKTORAT_STUDIO_PREFIXES = [
    'Rektorat_',
    'WestAnnex_',
    'EastAnnex_',
    'SelasarAgung_',
    'Cloister_',
    'FrontGatehouse_',
    'SE45Breezeway_',
    'OctSkylitPendopo_',
    'NorthPavilion_'
  ];
  const studioSolidColliders = [
    ...SplineCorridorBuilder.buildWorldSolidColliders(null).staticColliders.filter(
      (c) => c.id && REKTORAT_STUDIO_PREFIXES.some((prefix) => c.id.startsWith(prefix))
    ),
    PhysicsKernel.prepareSolidCollider({
      id: 'StudioInspectionPlinth',
      type: 'CYL',
      x: 0.0,
      z: 10.0,
      r: 89.0,
      minY: -0.62,
      maxY: 0.0
    })
  ];

  const camDirector = new CameraDirector(
    camera,
    controls,
    studioSolidColliders
  );
  camDirector.setWorldPhysicsContext({
    staticColliders: studioSolidColliders,
    groundHeightFn: studioGroundFn,
    walkableHeightFn: studioGroundFn,
    oceanWaterY: -5.5
  });
  camDirector.bindKeyboard(window);

  // Pointer drag look support when in DRONE_6DOF
  let dragging = false;
  let prevX = 0;
  let prevY = 0;
  renderer.domElement.addEventListener('pointerdown', (e) => {
    camDirector.cancelTransition();
    dragging = true;
    prevX = e.clientX;
    prevY = e.clientY;
  });
  renderer.domElement.addEventListener(
    'wheel',
    () => {
      camDirector.cancelTransition();
    },
    { passive: true }
  );
  window.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    camDirector.lookDelta(e.clientX - prevX, e.clientY - prevY);
    prevX = e.clientX;
    prevY = e.clientY;
  });
  window.addEventListener('pointerup', () => {
    dragging = false;
  });

  // --- NON-OCCLUDING PANEL & PORTRAIT ERGONOMICS (DEFAULT COLLAPSED) ---
  const pillBtn = document.getElementById('pill-toggle');
  const pillLabelEl = document.getElementById('studio-pill-label');
  const drawer = document.getElementById('inspector-drawer');
  const closeBtn = document.getElementById('drawer-close');
  let isPanelExpanded = false;

  function syncViewportAndOffset() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h);
    const rect = drawer.getBoundingClientRect();
    applyNonOccludingViewportOffset(camera, w, h, isPanelExpanded, {
      width: rect.width || 274,
      height: rect.height || Math.min(h * 0.32, h * 0.34),
      side: 'right'
    });
  }

  function setPanelExpanded(expanded) {
    isPanelExpanded = expanded;
    drawer.classList.toggle('collapsed', !expanded);
    drawer.setAttribute('aria-expanded', String(expanded));
    pillBtn.classList.toggle('hidden', expanded);
    requestAnimationFrame(syncViewportAndOffset);
  }

  pillBtn.addEventListener('click', () => setPanelExpanded(true));
  closeBtn.addEventListener('click', () => setPanelExpanded(false));
  window.addEventListener('resize', syncViewportAndOffset);
  syncViewportAndOffset();

  // Tabs
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
      requestAnimationFrame(syncViewportAndOffset);
    });
  });

  // Tri-Angle Camera Presets
  document.querySelectorAll('[data-preset]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-preset]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('[data-cammode]').forEach((b) =>
        b.classList.toggle('active', b.dataset.cammode === 'ORBIT')
      );
      camDirector.applyOrthoPreset(btn.dataset.preset, presets);
    });
  });

  // Camera Mode
  document.querySelectorAll('[data-cammode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-cammode]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      camDirector.setMode(btn.dataset.cammode);
    });
  });

  // Studio Zebra / Clay Inspection
  document.querySelectorAll('[data-inspect]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-inspect]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      sharedUniforms.uZebraInspectMode.value = Number(btn.dataset.inspect);
    });
  });

  // --- LIVE DEPOK & 20–45s SMOOTH TRANSITION CONTROLS ---
  const liveCardEl = document.getElementById('studio-depok-card');
  const liveBadgeEl = document.getElementById('studio-depok-badge');
  const liveClockEl = document.getElementById('studio-depok-clock');
  const liveWeatherEl = document.getElementById('studio-depok-weather');
  const liveBtnEl = document.getElementById('studio-btn-live-depok');
  const sliderDurEl = document.getElementById('studio-slider-trans-dur');
  const labelDurEl = document.getElementById('studio-label-trans-dur');
  const todButtons = Array.from(document.querySelectorAll('[data-tod]'));
  const weatherButtons = Array.from(document.querySelectorAll('[data-weather]'));

  function updateDepokDOM() {
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
        liveWeatherEl.textContent =
          `${latestDepokWeather.labelId.replace(' (Estimasi Lokal)', '')} · ` +
          `${latestDepokWeather.temperatureC}°C · RH ${latestDepokWeather.humidityPct}%`;
      } else {
        const tsNow = weather.getTransitionState();
        liveWeatherEl.textContent =
          `Simulasi: ${WEATHER_LABELS[tsNow.weather.toMode]} · Rill ${latestDepokClock.formattedHM}`;
      }
    }
  }
  updateDepokDOM();

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
      syncStudioTransitionUI();
    });
  });

  weatherButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      isLiveDepok = false;
      weather.setWeather(btn.dataset.weather);
      syncStudioTransitionUI();
    });
  });

  if (liveBtnEl) {
    liveBtnEl.addEventListener('click', () => {
      isLiveDepok = true;
      latestDepokClock = DepokLiveWeatherService.getDepokWIBClock();
      updateDepokDOM();
      weather.setReferenceAstronomyDate(new Date(), sharedUniforms);
      weather.setSolarHour(latestDepokClock.solarHour, sharedUniforms);
      weather.setWeather(latestDepokWeather.weatherMode, {
        cloudCoverNorm: latestDepokWeather.cloudCoverNorm
      });
      syncStudioTransitionUI();
    });
  }

  setInterval(async () => {
    const fresh = await DepokLiveWeatherService.fetchLiveDepokWeather();
    latestDepokWeather = fresh;
    updateDepokDOM();
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

  function syncStudioTransitionUI() {
    const ts = weather.getTransitionState();
    renderAxisButtonStates(weatherButtons, 'weather', ts.weather);
    renderAxisButtonStates(todButtons, 'tod', ts.time);

    const anyTrans = ts.weather.active || ts.time.active;
    const pct = ts.weather.active
      ? Math.round(ts.weather.progress * 100)
      : Math.round(ts.time.progress * 100);
    const rem = ts.weather.active
      ? Math.max(1, Math.ceil(ts.weather.remainingSec))
      : Math.max(1, Math.ceil(ts.time.remainingSec));
    const targetLabel = ts.weather.active
      ? WEATHER_LABELS[ts.weather.toMode]
      : TOD_LABELS[ts.time.toTod];

    if (liveCardEl && liveBtnEl) {
      liveCardEl.classList.toggle('is-manual', !isLiveDepok);
      const titleEl = liveBtnEl.querySelector('.btn-title');
      const subEl = liveBtnEl.querySelector('.btn-sub');
      liveBtnEl.classList.remove('active', 'is-transitioning');
      liveBtnEl.style.setProperty('--trans-pct', '0%');

      if (isLiveDepok) {
        if (liveBadgeEl) liveBadgeEl.textContent = 'LIVE KAMPUS UI DEPOK';
        if (anyTrans) {
          liveBtnEl.classList.add('is-transitioning');
          liveBtnEl.style.setProperty('--trans-pct', `${pct}%`);
          if (titleEl) titleEl.textContent = 'Menyelaraskan ke Rill Depok...';
          if (subEl) subEl.textContent = `${pct}% · Sisa ${rem} dtk`;
        } else {
          liveBtnEl.classList.add('active');
          if (titleEl) titleEl.textContent = '● Sinkron Rill Kampus UI Depok';
          if (subEl) subEl.textContent = 'Jam WIB & Cuaca Langsung Aktif';
        }
      } else {
        if (liveBadgeEl) liveBadgeEl.textContent = 'MODE MANUAL STUDIO';
        if (titleEl) titleEl.textContent = '↻ Kembali ke Waktu & Cuaca Rill Depok';
        if (subEl) subEl.textContent = `Transisi halus ${Math.round(ts.durationSec)} detik`;
        updateDepokDOM();
      }
    }

    if (pillBtn && pillLabelEl) {
      if (anyTrans) {
        pillBtn.classList.add('is-transitioning');
        pillBtn.style.setProperty('--pill-pct', `${pct}%`);
        pillLabelEl.textContent = `Menuju ${targetLabel} · ${pct}% (${rem}d)`;
      } else {
        pillBtn.classList.remove('is-transitioning');
        pillBtn.style.setProperty('--pill-pct', '0%');
        pillLabelEl.textContent = isLiveDepok
          ? `LIVE Depok · ${latestDepokClock.formattedHM} · Studio`
          : `Studio · ${TOD_LABELS[ts.time.toTod]} · ${WEATHER_LABELS[ts.weather.toMode]}`;
      }
    }
  }
  syncStudioTransitionUI();

  const studioAudioBtn = document.getElementById('studio-btn-audio');
  if (studioAudioBtn) {
    studioAudioBtn.addEventListener('click', () => {
      const active = audioSynth.toggle();
      audioSynth.syncContinuousWeatherAudio(
        weather.uniforms.uRainIntensity.value,
        weather.uniforms.uStormFactor.value,
        weather.uniforms.uWindVec.value.length()
      );
      const titleSpan = studioAudioBtn.querySelector('.btn-title') || studioAudioBtn;
      titleSpan.textContent = active ? 'Suara Akustik: ON' : 'Suara Akustik: OFF';
      studioAudioBtn.classList.toggle('active', active);
    });
  }

  // Populate initial 4-Gate Verification telemetry
  const initReport = rektoratAsset.userData.gateReport;
  const elFps = document.getElementById('stat-fps');
  const elCalls = document.getElementById('stat-calls');
  const elTris = document.getElementById('stat-tris');
  const elDims = document.getElementById('stat-dims');
  if (initReport && initReport.measuredSize) {
    const ms = initReport.measuredSize;
    elDims.textContent = `${ms.width.toFixed(1)}×${ms.length.toFixed(1)}×${ms.height.toFixed(1)}`;
  }

  let lastTime = performance.now();
  let frameAcc = 0;
  let timeAcc = 0;
  let clockAcc = 0;
  let uiAcc = 0;

  function animate(now) {
    requestAnimationFrame(animate);
    const dt = Math.min(0.05, (now - lastTime) * 0.001);
    lastTime = now;

    clockAcc += dt;
    if (clockAcc >= 1.0) {
      clockAcc = 0;
      latestDepokClock = DepokLiveWeatherService.getDepokWIBClock();
      updateDepokDOM();
      if (isLiveDepok && !weather.getTransitionState().time.active) {
        weather.setSolarHour(latestDepokClock.solarHour, sharedUniforms, {
          immediate: true,
          preserveAdaptation: true
        });
      }
    }

    sharedUniforms.uTime.value += dt;
    weather.update(dt, camera, sharedUniforms);
    GedungRektoratUIAsset.setNightIllumination(rektoratAsset, weather.nightIlluminationLevel);
    audioSynth.syncContinuousWeatherAudio(
      weather.uniforms.uRainIntensity.value,
      weather.uniforms.uStormFactor.value,
      weather.uniforms.uWindVec.value.length()
    );

    camDirector.update(dt, studioGroundFn);
    GedungRektoratUIAsset.updateKinematics(rektoratAsset, dt, {
      isNight: weather.nightIlluminationLevel > 0.35,
      nightLevel: weather.nightIlluminationLevel,
      windMultiplier: weather.getWindGustMultiplier()
    });

    renderer.render(scene, camera);

    uiAcc += dt;
    if (uiAcc >= 0.085) {
      uiAcc = 0;
      syncStudioTransitionUI();
    }

    frameAcc++;
    timeAcc += dt;
    if (timeAcc >= 0.45) {
      const fps = Math.round(frameAcc / timeAcc);
      const calls = renderer.info.render.calls;
      elFps.textContent = `${fps} FPS`;
      elCalls.textContent = `${calls} / 25`;
      elTris.textContent = renderer.info.render.triangles.toLocaleString('id-ID');

      document.getElementById('g1-badge').textContent = initReport?.gate1_EnvelopeAndNaN ? 'PASS' : 'WARN';
      document.getElementById('g2-badge').textContent = initReport?.gate2_ManifoldNormals ? 'PASS' : 'WARN';
      document.getElementById('g3-badge').textContent = initReport?.gate3_ZeroGapAndKinematics ? 'PASS' : 'WARN';
      document.getElementById('g4-badge').textContent = calls <= 25 ? 'PASS' : 'WARN';

      frameAcc = 0;
      timeAcc = 0;
    }
  }

  if (typeof window !== 'undefined') {
    window.__NEXUS_STUDIO__ = {
      scene,
      camera,
      renderer,
      weather,
      camDirector,
      studioSolidColliders,
      rektoratAsset,
      sharedUniforms,
      presets,
      syncStudioTransitionUI,
      updateDepokDOM,
      setLiveDepok(flag) {
        isLiveDepok = !!flag;
      }
    };
  }

  requestAnimationFrame(animate);
}

bootstrapStudioWorkbench();
