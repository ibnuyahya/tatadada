/**
 * Kepulauan UI Depok — Read-Only Forensic Telemetry & 30-Second Rolling Blackbox System
 * ./core/ForensicLogSystem.js
 *
 * Strictly Read-Only Telemetry Module (0% modification or invocation of any 3D render pass):
 * 1. NexusRolling30sRecorder:
 *    - 30 pre-allocated TypedArray 1-second buckets + live partial-second bucket (zero GC, zero raw frame spam).
 *    - Bounded 56-slot substantive event ring capturing camera-switch frame cost, runtime shader compilations,
 *      genuine GPU/CPU frame spikes (separated from OS window blur/tab pauses), sustained heavy load hints,
 *      and camera/medium/optics transitions.
 *    - Automated forensicSynthesis with per-camera-mode & per-group metrics, render-pass efficiency,
 *      and bottleneck diagnosis.
 *
 * 2. createForensicReadOnlyProbe:
 *    - Pure read-only snapshot of WebGL2 state, GPU capabilities, 4-Gate invariants, active camera,
 *      Depok WIB weather/astronomy, and Dolby Atmos 7.1.4 telemetry for JSON Copy & .JSON Download.
 */

export const NEXUS_FORENSIC_CAM_MODES = [
  'DRONE_6DOF',
  'DRONE_AVATA2_FPV',
  'ORBIT',
  'AUTO_CINEMATIC',
  'WALK_SIVITAS',
  'FPS_JOGGER',
  'FPS_BUS_DRIVER',
  'FPS_GOLF_DRIVER',
  'FPS_CHAT_NPC',
  'FPS_CAT',
  'FPS_PILOT'
];

export const NEXUS_FORENSIC_CAM_GROUPS = [
  'mavic_aerial',
  'avata2_fpv',
  'human_eye_fps',
  'orbit_cinema'
];

export const NEXUS_FORENSIC_MEDIUMS = [
  'udara_drone',
  'udara_pesawat_evtol',
  'jalan_kaki',
  'kabin_darat',
  'kapal_laut',
  'orbit_bebas'
];

export const NEXUS_FORENSIC_OPTICS = [
  'HUMAN_EYE_TRUE_COLOR',
  'DRONE_MAVIC3_CINEMA'
];

export function camModeToForensicGroup(mode) {
  if (mode === 'DRONE_AVATA2_FPV') return 'avata2_fpv';
  if (mode === 'ORBIT' || mode === 'AUTO_CINEMATIC') return 'orbit_cinema';
  if (
    mode === 'WALK_SIVITAS' ||
    mode === 'FPS_JOGGER' ||
    mode === 'FPS_BUS_DRIVER' ||
    mode === 'FPS_GOLF_DRIVER' ||
    mode === 'FPS_CHAT_NPC' ||
    mode === 'FPS_CAT' ||
    mode === 'FPS_PILOT'
  ) {
    return 'human_eye_fps';
  }
  return 'mavic_aerial';
}

export function detectKepulauanMediumIdx(camMode, pilotKind = '') {
  if (camMode === 'ORBIT') return 5;
  if (camMode === 'WALK_SIVITAS' || camMode === 'FPS_JOGGER' || camMode === 'FPS_CHAT_NPC' || camMode === 'FPS_CAT') {
    return 2;
  }
  if (camMode === 'FPS_BUS_DRIVER' || camMode === 'FPS_GOLF_DRIVER') {
    return 3;
  }
  if (camMode === 'FPS_PILOT') {
    const pk = String(pilotKind || '').toUpperCase();
    if (pk.includes('FERRY') || pk.includes('FERI') || pk.includes('YACHT') || pk.includes('SHIP') || pk.includes('BOAT')) {
      return 4;
    }
    return 1;
  }
  return 0;
}

export class NexusRolling30sRecorder {
  constructor(options = {}) {
    this.BUCKET_CAP = 30;
    this.EVENT_CAP = 56;
    this.WINDOW_SEC = 30.0;
    this._customNowSec = typeof options.nowSecFn === 'function' ? options.nowSecFn : null;

    const cap = this.BUCKET_CAP;
    this.bEndSec = new Float64Array(cap);
    this.bFrames = new Uint16Array(cap);
    this.bAvgMs = new Float32Array(cap);
    this.bMaxMs = new Float32Array(cap);
    this.bCpuMs = new Float32Array(cap);
    this.bHitches25 = new Uint16Array(cap);
    this.bStalls50 = new Uint16Array(cap);
    this.bMainCalls = new Uint16Array(cap);
    this.bMainTrisK = new Float32Array(cap);
    this.bReflRuns = new Uint16Array(cap);
    this.bReflCalls = new Uint16Array(cap);
    this.bGrassRuns = new Uint16Array(cap);
    this.bShadowRuns = new Uint16Array(cap);
    this.bPrograms = new Uint16Array(cap);
    this.bCamIdx = new Uint8Array(cap);
    this.bMedIdx = new Uint8Array(cap);
    this.bOpticsIdx = new Uint8Array(cap);
    this.bPosX = new Int16Array(cap);
    this.bPosY = new Int16Array(cap);
    this.bPosZ = new Int16Array(cap);
    this.bSpeed10 = new Uint16Array(cap);
    this.bDpr100 = new Uint16Array(cap);

    this.bHead = 0;
    this.bFilled = 0;

    this.curWindowStartSec = -1.0;
    this.accFrames = 0;
    this.accMsSum = 0.0;
    this.accMaxMs = 0.0;
    this.accCpuSum = 0.0;
    this.accHitch25 = 0;
    this.accStall50 = 0;
    this.accMainCallsSum = 0;
    this.accMainTrisSum = 0;
    this.accReflRuns = 0;
    this.accReflCallsSum = 0;
    this.accGrassRuns = 0;
    this.accShadowRuns = 0;
    this.accLastPrograms = 0;
    this.accCamIdx = 0;
    this.accMedIdx = 0;
    this.accOpticsIdx = 1;
    this.accPosX = 0;
    this.accPosY = 0;
    this.accPosZ = 0;
    this.accSpeedMS = 0.0;
    this.accDpr = 1.0;

    this.dtRing = new Float32Array(240);
    this.dtCursor = 0;
    this.totalFrames = 0;
    this.emaFrameMs = 16.67;
    this.lastMeasuredFps = 60;
    this.lastReflRan = false;
    this.lastReflCalls = 0;

    this.lastKnownPrograms = -1;
    this.lastMainCalls = 0;
    this.lastMedIdx = -1;
    this.lastOpticsIdx = -1;
    this.lastSpikeLoggedSec = -10.0;
    this.lastHeavyLoadSec = -10.0;

    this.windowUnfocused = false;
    this.skipSpikeFramesAfterFocus = 0;

    this.pendingCamSwitch = false;
    this.pendingCamFrom = 'DRONE_6DOF';
    this.pendingCamTo = 'DRONE_6DOF';
    this.pendingCamCallsBefore = 0;
    this.pendingCamProgsBefore = 0;

    this.eventsRing = new Array(this.EVENT_CAP);
    this.evHead = 0;
    this.evFilled = 0;

    this._simStartPerfMs =
      typeof performance !== 'undefined' && performance.now ? performance.now() : 0.0;
    this._virtualClockSec = 0.0;

    // Forensik Sound v2.4 (Tahap 1) 1Hz Passive Audio Blackbox Trackers
    this.audioSynth = options.audioSynth || null;
    this.hwGovernor = options.hwGovernor || null;
    this.lastAudioCtxState = 'uninitialized';
    this.lastAudioMuted = true;
    this.lastAudioUnderruns = 0;
    this.lastAudioHeavyLimits = 0;
    this.lastAudioErrors = 0;

    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
      window.addEventListener('blur', () => {
        this.windowUnfocused = true;
        this.skipSpikeFramesAfterFocus = 2;
      });
      window.addEventListener('focus', () => {
        this.windowUnfocused = false;
        this.skipSpikeFramesAfterFocus = 2;
      });
      if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
        document.addEventListener('visibilitychange', () => {
          if (document.hidden) this.windowUnfocused = true;
          this.skipSpikeFramesAfterFocus = 2;
        });
      }
    }
  }

  nowUptimeSec() {
    if (this._customNowSec) return this._customNowSec();
    if (typeof performance !== 'undefined' && performance.now) {
      return Math.max(this._virtualClockSec, (performance.now() - this._simStartPerfMs) * 0.001);
    }
    return this._virtualClockSec;
  }

  _camModeToIdx(mode) {
    const idx = NEXUS_FORENSIC_CAM_MODES.indexOf(mode);
    return idx >= 0 ? idx : 0;
  }

  _pushEventObj(evObj) {
    this.eventsRing[this.evHead] = evObj;
    this.evHead = (this.evHead + 1) % this.EVENT_CAP;
    if (this.evFilled < this.EVENT_CAP) this.evFilled++;
  }

  onCameraSwitch(fromMode, toMode) {
    if (fromMode === toMode) return;
    this.pendingCamSwitch = true;
    this.pendingCamFrom = fromMode || 'DRONE_6DOF';
    this.pendingCamTo = toMode || 'DRONE_6DOF';
    this.pendingCamCallsBefore = this.lastMainCalls;
    this.pendingCamProgsBefore = this.lastKnownPrograms >= 0 ? this.lastKnownPrograms : 0;
  }

  attachAudioForensics(audioSynth, hwGovernor = null) {
    if (audioSynth) {
      this.audioSynth = audioSynth;
      this.lastAudioCtxState =
        audioSynth.ctx && audioSynth.ctx.state ? String(audioSynth.ctx.state) : 'uninitialized';
      this.lastAudioMuted = !Boolean(audioSynth.enabled);
    }
    if (hwGovernor) {
      this.hwGovernor = hwGovernor;
    }
  }

  clearRollingHistory() {
    this.bHead = 0;
    this.bFilled = 0;
    this.evHead = 0;
    this.evFilled = 0;
    this.accFrames = 0;
    this.accMsSum = 0.0;
    this.accMaxMs = 0.0;
    this.accCpuSum = 0.0;
    this.accHitch25 = 0;
    this.accStall50 = 0;
    this.accMainCallsSum = 0;
    this.accMainTrisSum = 0;
    this.accReflRuns = 0;
    this.accReflCallsSum = 0;
    this.accGrassRuns = 0;
    this.accShadowRuns = 0;
    this.dtRing.fill(0);
    this.dtCursor = 0;
    this.totalFrames = 0;
    this.lastAudioUnderruns = 0;
    this.lastAudioHeavyLimits = 0;
    this.lastAudioErrors = 0;
    if (this.audioSynth && typeof this.audioSynth.resetRollingForensics === 'function') {
      this.audioSynth.resetRollingForensics();
    }
  }

  recordSubstantiveEvent(kind, detail, extra = null) {
    const tSec = this.nowUptimeSec();
    this._pushEventObj(Object.assign({ tSec, kind, detail }, extra || {}));
  }

  _flushOneSecondBucket(nowSec) {
    if (this.accFrames <= 0) {
      this.curWindowStartSec = nowSec;
      return;
    }
    const slot = this.bHead;
    const fr = this.accFrames;
    const avgMs = this.accMsSum / fr;
    const avgCpu = this.accCpuSum / fr;
    const avgMainCalls = Math.round(this.accMainCallsSum / fr);
    const avgReflCalls = this.accReflRuns > 0 ? Math.round(this.accReflCallsSum / this.accReflRuns) : 0;
    const estFps = avgMs > 0.25 ? Math.min(240, Math.round(1000.0 / avgMs)) : 60;
    this.lastMeasuredFps = estFps;

    this.bEndSec[slot] = nowSec;
    this.bFrames[slot] = Math.min(65535, fr);
    this.bAvgMs[slot] = avgMs;
    this.bMaxMs[slot] = this.accMaxMs;
    this.bCpuMs[slot] = avgCpu;
    this.bHitches25[slot] = this.accHitch25;
    this.bStalls50[slot] = this.accStall50;
    this.bMainCalls[slot] = Math.min(65535, avgMainCalls);
    this.bMainTrisK[slot] = (this.accMainTrisSum / fr) * 0.001;
    this.bReflRuns[slot] = this.accReflRuns;
    this.bReflCalls[slot] = Math.min(65535, avgReflCalls);
    this.bGrassRuns[slot] = this.accGrassRuns;
    this.bShadowRuns[slot] = this.accShadowRuns;
    this.bPrograms[slot] = this.accLastPrograms;
    this.bCamIdx[slot] = this.accCamIdx;
    this.bMedIdx[slot] = this.accMedIdx;
    this.bOpticsIdx[slot] = this.accOpticsIdx;
    this.bPosX[slot] = Math.max(-32767, Math.min(32767, Math.round(this.accPosX)));
    this.bPosY[slot] = Math.max(-32767, Math.min(32767, Math.round(this.accPosY)));
    this.bPosZ[slot] = Math.max(-32767, Math.min(32767, Math.round(this.accPosZ)));
    this.bSpeed10[slot] = Math.min(65535, Math.max(0, Math.round(this.accSpeedMS * 10.0)));
    this.bDpr100[slot] = Math.min(65535, Math.max(25, Math.round(this.accDpr * 100.0)));

    this.bHead = (this.bHead + 1) % this.BUCKET_CAP;
    if (this.bFilled < this.BUCKET_CAP) this.bFilled++;

    if (fr >= 5 && estFps < 38 && nowSec - this.lastHeavyLoadSec >= 3.0) {
      this.lastHeavyLoadSec = nowSec;
      let hint = 'gpu_fillrate_msaa_or_postfx';
      if (avgCpu > avgMs * 0.58) hint = 'cpu_simulation_or_ai_tick';
      else if (avgMainCalls > 25) hint = 'draw_call_budget_exceeded';
      else if (this.accReflRuns >= fr && avgReflCalls > 22) hint = 'planar_water_reflection_pass';
      else if (this.accShadowRuns >= fr && fr >= 12) hint = 'frequent_shadow_map_refresh';
      this._pushEventObj({
        tSec: nowSec,
        kind: 'sustained_heavy_load',
        avgFps: estFps,
        avgMs: +avgMs.toFixed(2),
        cpuMs: +avgCpu.toFixed(2),
        cam: NEXUS_FORENSIC_CAM_MODES[this.accCamIdx],
        mainCalls: avgMainCalls,
        reflCalls: avgReflCalls,
        bottleneckHint: hint
      });
    }

    // Passive 1Hz Sound Forensics v2.4 Event Audit (Zero audio-graph mutation)
    if (this.audioSynth && this.audioSynth._forensic) {
      const af = this.audioSynth._forensic;
      const curAudState =
        this.audioSynth.ctx && this.audioSynth.ctx.state
          ? String(this.audioSynth.ctx.state)
          : 'uninitialized';
      const curAudMuted = !Boolean(this.audioSynth.enabled);
      if (curAudState !== this.lastAudioCtxState) {
        this._pushEventObj({
          tSec: nowSec,
          kind: 'audio_state_change',
          detail: `Web Audio context ${this.lastAudioCtxState} → ${curAudState}`,
          from: this.lastAudioCtxState,
          to: curAudState
        });
        this.lastAudioCtxState = curAudState;
      }
      if (curAudMuted !== this.lastAudioMuted) {
        this.lastAudioMuted = curAudMuted;
        this._pushEventObj({
          tSec: nowSec,
          kind: 'audio_state_change',
          detail: curAudMuted ? 'Audio dibisukan pengguna (mute)' : 'Audio diaktifkan pengguna (unmute)'
        });
      }
      if (af.clock && af.clock.underrunSuspectCount > this.lastAudioUnderruns) {
        this.lastAudioUnderruns = af.clock.underrunSuspectCount;
        this._pushEventObj({
          tSec: nowSec,
          kind: 'audio_underrun_lag',
          lagMs: af.clock.lastUnderrunLagMs,
          count: this.lastAudioUnderruns,
          detail: `Lag clock Web Audio ${af.clock.lastUnderrunLagMs}ms (>24ms gejala underrun ke-${this.lastAudioUnderruns})`
        });
      }
      if (af.limiter && af.limiter.heavyLimitingEvents > this.lastAudioHeavyLimits) {
        this.lastAudioHeavyLimits = af.limiter.heavyLimitingEvents;
        this._pushEventObj({
          tSec: nowSec,
          kind: 'audio_limiter_overload',
          reductionDb: af.limiter.currentReductionDb,
          detail: `Master compressor/limiter kompresi berat (${af.limiter.currentReductionDb} dB)`
        });
      }
      if (af.perf && af.perf.updateErrorsCount > this.lastAudioErrors) {
        this.lastAudioErrors = af.perf.updateErrorsCount;
        this._pushEventObj({
          tSec: nowSec,
          kind: 'audio_error',
          detail: `Eror runtime audio: ${af.perf.lastUpdateError}`
        });
      }
    }

    this.curWindowStartSec = nowSec;
    this.accFrames = 0;
    this.accMsSum = 0.0;
    this.accMaxMs = 0.0;
    this.accCpuSum = 0.0;
    this.accHitch25 = 0;
    this.accStall50 = 0;
    this.accMainCallsSum = 0;
    this.accMainTrisSum = 0;
    this.accReflRuns = 0;
    this.accReflCallsSum = 0;
    this.accGrassRuns = 0;
    this.accShadowRuns = 0;
  }

  recordRenderedFrame(sample) {
    let frameMs = Number(sample.frameMs) || 16.67;
    const cpuMs = Number(sample.cpuMs) || 0.0;
    const mainCalls = Number(sample.mainCalls) || 0;
    const mainTris = Number(sample.mainTris) || 0;
    const reflRan = Boolean(sample.reflRan);
    const reflCalls = Number(sample.reflCalls) || 0;
    const grassVisible = Boolean(sample.grassVisible);
    const shadowUpdated = Boolean(sample.shadowUpdated);
    const programsCount = Number(sample.programsCount) || 0;
    const camMode = sample.camMode || 'DRONE_6DOF';
    const pilotKind = sample.pilotKind || '';
    const isDroneCinema = Boolean(sample.isDroneCinema);
    const posX = Number(sample.posX) || 0;
    const posY = Number(sample.posY) || 0;
    const posZ = Number(sample.posZ) || 0;
    const speedMS = Number(sample.speedMS) || 0;
    const dpr = Number(sample.dpr) || 1.0;

    this.lastReflRan = reflRan;
    this.lastReflCalls = reflRan ? reflCalls : 0;

    this._virtualClockSec += Math.max(0.0005, Math.min(1.0, frameMs * 0.001));
    const nowSec = sample.explicitNowSec !== undefined ? Number(sample.explicitNowSec) : this.nowUptimeSec();
    if (this.curWindowStartSec < 0.0) this.curWindowStartSec = nowSec;

    this.dtRing[this.dtCursor] = frameMs;
    this.dtCursor = (this.dtCursor + 1) % 240;
    this.totalFrames++;

    const cIdx = this._camModeToIdx(camMode);
    const mIdx = detectKepulauanMediumIdx(camMode, pilotKind);
    const oIdx = isDroneCinema ? 1 : 0;

    let progsDelta = 0;
    if (this.lastKnownPrograms < 0 || this.totalFrames <= 3) {
      this.lastKnownPrograms = programsCount;
    } else if (programsCount > this.lastKnownPrograms) {
      progsDelta = programsCount - this.lastKnownPrograms;
      this.lastKnownPrograms = programsCount;
      this._pushEventObj({
        tSec: nowSec,
        kind: 'shader_compile',
        addedPrograms: progsDelta,
        totalPrograms: programsCount,
        frameMs: +frameMs.toFixed(2),
        cam: NEXUS_FORENSIC_CAM_MODES[cIdx]
      });
    } else if (programsCount < this.lastKnownPrograms) {
      this.lastKnownPrograms = programsCount;
    }

    const wasCamSwitchFrame = this.pendingCamSwitch;
    if (this.pendingCamSwitch) {
      this.pendingCamSwitch = false;
      this._pushEventObj({
        tSec: nowSec,
        kind: 'camera_switch',
        from: this.pendingCamFrom,
        to: this.pendingCamTo,
        optics: NEXUS_FORENSIC_OPTICS[oIdx],
        switchFrameMs: +frameMs.toFixed(2),
        callsBefore: this.pendingCamCallsBefore,
        callsAfter: mainCalls,
        trisAfterK: +(mainTris * 0.001).toFixed(1),
        reflCallsAfter: reflRan ? reflCalls : 0,
        programsDelta: Math.max(0, programsCount - this.pendingCamProgsBefore)
      });
    }

    const isFocusPause = (this.windowUnfocused || this.skipSpikeFramesAfterFocus > 0) && frameMs > 280.0;
    if (this.skipSpikeFramesAfterFocus > 0) this.skipSpikeFramesAfterFocus--;
    if (isFocusPause) {
      this._pushEventObj({
        tSec: nowSec,
        kind: 'window_focus_pause',
        pauseMs: +frameMs.toFixed(1),
        detail: 'Jeda fokus jendela / tab OS (bukan lag render GPU)'
      });
      frameMs = this.emaFrameMs;
    } else {
      const spikeThreshold = Math.max(36.0, this.emaFrameMs * 2.15);
      if (frameMs > spikeThreshold && this.totalFrames > 8 && nowSec - this.lastSpikeLoggedSec >= 0.45) {
        this.lastSpikeLoggedSec = nowSec;
        let cause = 'browser_or_gc_jank';
        if (progsDelta > 0) cause = 'shader_compile';
        else if (wasCamSwitchFrame) cause = 'camera_view_transition';
        else if (cpuMs > frameMs * 0.55 && cpuMs > 14.0) cause = 'cpu_simulation_spike';
        else if (shadowUpdated && reflRan) cause = 'shadow_plus_planar_reflection';
        else if (mainCalls > 25) cause = 'high_draw_calls';
        this._pushEventObj({
          tSec: nowSec,
          kind: 'frame_spike',
          frameMs: +frameMs.toFixed(2),
          cpuMs: +cpuMs.toFixed(2),
          baselineMs: +this.emaFrameMs.toFixed(2),
          cam: NEXUS_FORENSIC_CAM_MODES[cIdx],
          mainCalls,
          reflCalls: reflRan ? reflCalls : 0,
          trisK: +(mainTris * 0.001).toFixed(1),
          shadowUpdated,
          programsDelta: progsDelta,
          probableCause: cause
        });
      }
      this.emaFrameMs = this.emaFrameMs * 0.92 + Math.min(frameMs, 120.0) * 0.08;
    }

    if (this.lastMedIdx >= 0 && mIdx !== this.lastMedIdx && this.totalFrames > 4) {
      const prevMed = NEXUS_FORENSIC_MEDIUMS[this.lastMedIdx];
      const nextMed = NEXUS_FORENSIC_MEDIUMS[mIdx];
      this.lastMedIdx = mIdx;
      this._pushEventObj({
        tSec: nowSec,
        kind: 'state_change',
        detail: `Medium ${prevMed}→${nextMed}`,
        from: prevMed,
        to: nextMed,
        pos: [Math.round(posX), Math.round(posY), Math.round(posZ)]
      });
    } else {
      this.lastMedIdx = mIdx;
    }

    if (this.lastOpticsIdx >= 0 && oIdx !== this.lastOpticsIdx && this.totalFrames > 4) {
      const nextOpt = NEXUS_FORENSIC_OPTICS[oIdx];
      this.lastOpticsIdx = oIdx;
      this._pushEventObj({
        tSec: nowSec,
        kind: 'state_change',
        detail: `Pipa Optik → ${oIdx === 1 ? 'Drone Mavic 3 Cinema Grade' : 'Human-Eye True-Color (0% LUT)'}`,
        optics: nextOpt
      });
    } else {
      this.lastOpticsIdx = oIdx;
    }

    this.accFrames++;
    this.accMsSum += frameMs;
    if (frameMs > this.accMaxMs) this.accMaxMs = frameMs;
    this.accCpuSum += cpuMs;
    if (frameMs > 25.0) this.accHitch25++;
    if (frameMs > 50.0) this.accStall50++;
    this.accMainCallsSum += mainCalls;
    this.accMainTrisSum += mainTris;
    if (reflRan) {
      this.accReflRuns++;
      this.accReflCallsSum += reflCalls;
    }
    if (grassVisible) this.accGrassRuns++;
    if (shadowUpdated) this.accShadowRuns++;
    this.accLastPrograms = programsCount;
    this.accCamIdx = cIdx;
    this.accMedIdx = mIdx;
    this.accOpticsIdx = oIdx;
    this.accPosX = posX;
    this.accPosY = posY;
    this.accPosZ = posZ;
    this.accSpeedMS = speedMS;
    this.accDpr = dpr;
    this.lastMainCalls = mainCalls;

    if (nowSec - this.curWindowStartSec >= 1.0) {
      this._flushOneSecondBucket(nowSec);
    }
  }

  getInstantPerformanceStats() {
    const sampleCount = Math.min(this.totalFrames, 240);
    let sumMs = 0.0;
    let maxMs = 0.0;
    const sorted = [];
    for (let i = 0; i < sampleCount; i++) {
      const v = this.dtRing[i];
      if (v > 0) {
        sumMs += v;
        if (v > maxMs) maxMs = v;
        sorted.push(v);
      }
    }
    sorted.sort((a, b) => a - b);
    const avgMs = sorted.length ? +(sumMs / sorted.length).toFixed(2) : 16.67;
    const p95Ms = sorted.length
      ? +sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))].toFixed(2)
      : 16.67;
    const avgFps = avgMs > 0.25 ? Math.min(240, Math.round(1000.0 / avgMs)) : this.lastMeasuredFps;
    const minFps = maxMs > 0.5 ? Math.max(1, Math.round(1000.0 / maxMs)) : avgFps;
    return {
      avgFps,
      minFps,
      avgFrameMs: avgMs,
      p95FrameMs: p95Ms,
      maxFrameMs: +maxMs.toFixed(2),
      totalFramesRendered: this.totalFrames
    };
  }

  buildRolling30sSnapshot() {
    const nowSec = this.nowUptimeSec();
    const cutoffSec = nowSec - this.WINDOW_SEC;

    const timeline1s = [];
    const startOffset = this.bFilled < this.BUCKET_CAP ? 0 : this.bHead;
    for (let i = 0; i < this.bFilled; i++) {
      const slot = (startOffset + i) % this.BUCKET_CAP;
      const tEnd = this.bEndSec[slot];
      if (tEnd < cutoffSec) continue;
      const fr = this.bFrames[slot];
      if (fr <= 0) continue;
      const avgM = this.bAvgMs[slot];
      timeline1s.push({
        tRelSec: +Math.max(-30.0, tEnd - nowSec).toFixed(1),
        frames: fr,
        fps: avgM > 0.25 ? Math.min(240, Math.round(1000.0 / avgM)) : 60,
        avgMs: +avgM.toFixed(2),
        maxMs: +this.bMaxMs[slot].toFixed(2),
        cpuMs: +this.bCpuMs[slot].toFixed(2),
        hitches25ms: this.bHitches25[slot],
        stalls50ms: this.bStalls50[slot],
        mainCalls: this.bMainCalls[slot],
        mainTrisK: +this.bMainTrisK[slot].toFixed(1),
        reflRuns: this.bReflRuns[slot],
        reflCalls: this.bReflCalls[slot],
        grassRuns: this.bGrassRuns[slot],
        shadowUpdates: this.bShadowRuns[slot],
        programs: this.bPrograms[slot],
        cam: NEXUS_FORENSIC_CAM_MODES[this.bCamIdx[slot]],
        optics: NEXUS_FORENSIC_OPTICS[this.bOpticsIdx[slot]],
        medium: NEXUS_FORENSIC_MEDIUMS[this.bMedIdx[slot]],
        pos: [this.bPosX[slot], this.bPosY[slot], this.bPosZ[slot]],
        speedMS: +(this.bSpeed10[slot] * 0.1).toFixed(1),
        dpr: +(this.bDpr100[slot] * 0.01).toFixed(2)
      });
    }

    if (this.accFrames > 0 && this.curWindowStartSec >= cutoffSec) {
      const fr = this.accFrames;
      const avgM = this.accMsSum / fr;
      timeline1s.push({
        tRelSec: 0.0,
        partialSec: true,
        frames: fr,
        fps: avgM > 0.25 ? Math.min(240, Math.round(1000.0 / avgM)) : 60,
        avgMs: +avgM.toFixed(2),
        maxMs: +this.accMaxMs.toFixed(2),
        cpuMs: +(this.accCpuSum / fr).toFixed(2),
        hitches25ms: this.accHitch25,
        stalls50ms: this.accStall50,
        mainCalls: Math.round(this.accMainCallsSum / fr),
        mainTrisK: +((this.accMainTrisSum / fr) * 0.001).toFixed(1),
        reflRuns: this.accReflRuns,
        reflCalls: this.accReflRuns > 0 ? Math.round(this.accReflCallsSum / this.accReflRuns) : 0,
        grassRuns: this.accGrassRuns,
        shadowUpdates: this.accShadowRuns,
        programs: this.accLastPrograms,
        cam: NEXUS_FORENSIC_CAM_MODES[this.accCamIdx],
        optics: NEXUS_FORENSIC_OPTICS[this.accOpticsIdx],
        medium: NEXUS_FORENSIC_MEDIUMS[this.accMedIdx],
        pos: [Math.round(this.accPosX), Math.round(this.accPosY), Math.round(this.accPosZ)],
        speedMS: +this.accSpeedMS.toFixed(1),
        dpr: +this.accDpr.toFixed(2)
      });
    }

    const substantiveEvents = [];
    const evStart = this.evFilled < this.EVENT_CAP ? 0 : this.evHead;
    for (let i = 0; i < this.evFilled; i++) {
      const ev = this.eventsRing[(evStart + i) % this.EVENT_CAP];
      if (!ev || ev.tSec < cutoffSec) continue;
      const cleanEv = Object.assign({}, ev);
      cleanEv.tRelSec = +Math.max(-30.0, ev.tSec - nowSec).toFixed(2);
      delete cleanEv.tSec;
      substantiveEvents.push(cleanEv);
    }

    const makeStatSlot = () => ({
      frames: 0,
      buckets1s: 0,
      avgFps: 0,
      avgFrameMs: 0,
      maxFrameMs: 0,
      avgMainCalls: 0,
      avgReflCalls: 0,
      avgTrisK: 0,
      _msW: 0,
      _callsW: 0,
      _reflW: 0,
      _trisW: 0
    });

    const byCameraMode = {};
    for (let c = 0; c < NEXUS_FORENSIC_CAM_MODES.length; c++) {
      byCameraMode[NEXUS_FORENSIC_CAM_MODES[c]] = makeStatSlot();
    }
    const byCameraGroup = {};
    for (let g = 0; g < NEXUS_FORENSIC_CAM_GROUPS.length; g++) {
      byCameraGroup[NEXUS_FORENSIC_CAM_GROUPS[g]] = makeStatSlot();
    }

    let totalFrames30s = 0;
    let totalWeightedMs = 0;
    let totalWeightedCpuMs = 0;
    let worstFrameMs = 0;
    let min1sFps = 999;
    let totalHitches25 = 0;
    let totalStalls50 = 0;
    let sustainedLowFpsSeconds = 0;
    let totalMainCallsW = 0;
    let totalMainTrisKW = 0;
    let totalReflRuns = 0;
    let totalReflCallsW = 0;
    let totalGrassRuns = 0;
    let totalShadowUpdates = 0;

    const accumSlot = (slotObj, b) => {
      if (!slotObj) return;
      const fr = b.frames;
      slotObj.frames += fr;
      slotObj.buckets1s += 1;
      slotObj._msW += b.avgMs * fr;
      slotObj._callsW += b.mainCalls * fr;
      slotObj._reflW += b.reflCalls * fr;
      slotObj._trisW += b.mainTrisK * fr;
      if (b.maxMs > slotObj.maxFrameMs) slotObj.maxFrameMs = b.maxMs;
    };

    const finalizeSlots = (mapObj, keys) => {
      for (let i = 0; i < keys.length; i++) {
        const s = mapObj[keys[i]];
        if (s.frames > 0) {
          const mMs = s._msW / s.frames;
          s.avgFrameMs = +mMs.toFixed(2);
          s.avgFps = mMs > 0.25 ? Math.min(240, Math.round(1000.0 / mMs)) : 60;
          s.maxFrameMs = +s.maxFrameMs.toFixed(2);
          s.avgMainCalls = Math.round(s._callsW / s.frames);
          s.avgReflCalls = Math.round(s._reflW / s.frames);
          s.avgTrisK = +(s._trisW / s.frames).toFixed(1);
        }
        delete s._msW;
        delete s._callsW;
        delete s._reflW;
        delete s._trisW;
      }
    };

    for (let i = 0; i < timeline1s.length; i++) {
      const b = timeline1s[i];
      const fr = b.frames;
      totalFrames30s += fr;
      totalWeightedMs += b.avgMs * fr;
      totalWeightedCpuMs += b.cpuMs * fr;
      if (b.maxMs > worstFrameMs) worstFrameMs = b.maxMs;
      if (!b.partialSec && b.fps < min1sFps) min1sFps = b.fps;
      if (!b.partialSec && b.fps < 38) sustainedLowFpsSeconds++;
      totalHitches25 += b.hitches25ms;
      totalStalls50 += b.stalls50ms;
      totalMainCallsW += b.mainCalls * fr;
      totalMainTrisKW += b.mainTrisK * fr;
      totalReflRuns += b.reflRuns;
      totalReflCallsW += b.reflCalls * b.reflRuns;
      totalGrassRuns += b.grassRuns;
      totalShadowUpdates += b.shadowUpdates;

      accumSlot(byCameraMode[b.cam], b);
      accumSlot(byCameraGroup[camModeToForensicGroup(b.cam)], b);
    }

    finalizeSlots(byCameraMode, NEXUS_FORENSIC_CAM_MODES);
    finalizeSlots(byCameraGroup, NEXUS_FORENSIC_CAM_GROUPS);

    const overallAvgMs = totalFrames30s > 0 ? totalWeightedMs / totalFrames30s : 16.67;
    const overallAvgCpuMs = totalFrames30s > 0 ? totalWeightedCpuMs / totalFrames30s : 2.0;
    const overallAvgFps =
      totalFrames30s > 0 && overallAvgMs > 0.25
        ? Math.min(240, Math.round(1000.0 / overallAvgMs))
        : this.lastMeasuredFps;
    if (min1sFps === 999) min1sFps = overallAvgFps;

    let camSwitchCount = 0;
    let runtimeShaderCompiles = 0;
    let maxCamSwitchMs = 0;
    for (let i = 0; i < substantiveEvents.length; i++) {
      const ev = substantiveEvents[i];
      if (ev.kind === 'camera_switch') {
        camSwitchCount++;
        if ((ev.switchFrameMs || 0) > maxCamSwitchMs) maxCamSwitchMs = ev.switchFrameMs;
      } else if (ev.kind === 'shader_compile') {
        runtimeShaderCompiles += ev.addedPrograms || 1;
      }
    }

    const avgMainCalls30s = totalFrames30s > 0 ? Math.round(totalMainCallsW / totalFrames30s) : 0;
    const avgMainTrisK30s = totalFrames30s > 0 ? +(totalMainTrisKW / totalFrames30s).toFixed(1) : 0;
    const avgReflCalls30s = totalReflRuns > 0 ? Math.round(totalReflCallsW / totalReflRuns) : 0;
    const reflActivePct = totalFrames30s > 0 ? Math.round((totalReflRuns / totalFrames30s) * 100) : 0;
    const grassActivePct = totalFrames30s > 0 ? Math.round((totalGrassRuns / totalFrames30s) * 100) : 0;
    const shadowUpdatePct = totalFrames30s > 0 ? Math.round((totalShadowUpdates / totalFrames30s) * 100) : 0;

    const bottleneckFindings = [];
    const improvementOpportunities = [];

    if (runtimeShaderCompiles > 0) {
      bottleneckFindings.push(
        `Terdeteksi ${runtimeShaderCompiles} kompilasi shader runtime di tengah simulasi.`
      );
      improvementOpportunities.push(
        'Pastikan varian material baru ikut di-prewarm pada tahap Preloader ke-8 sebelum tirai dibuka.'
      );
    }
    if (avgMainCalls30s > 25) {
      bottleneckFindings.push(
        `Rata-rata draw call utama (${avgMainCalls30s} calls/frame) melampaui pagu Gate-4 (<= 25 calls).`
      );
      improvementOpportunities.push(
        'Gabungkan mesh statis ke BatchRenderer atau gunakan InstancedMesh tunggal.'
      );
    }
    if (camSwitchCount > 0 && maxCamSwitchMs > 42.0) {
      bottleneckFindings.push(
        `Pergantian perspektif kamera memicu frame puncak ${maxCamSwitchMs}ms.`
      );
    }
    if (totalStalls50 >= 2 || worstFrameMs > 85.0) {
      bottleneckFindings.push(
        `Terdeteksi ${totalStalls50} frame stall >50ms (frame terburuk ${worstFrameMs.toFixed(1)}ms, min 1d ${min1sFps} FPS).`
      );
    }
    if ((overallAvgFps < 45 || sustainedLowFpsSeconds >= 2) && totalFrames30s >= 15) {
      const domBound = overallAvgCpuMs > overallAvgMs * 0.58 ? 'CPU Simulasi JS' : 'GPU Fillrate / MSAA';
      bottleneckFindings.push(
        `Rata-rata 30d ${overallAvgFps} FPS (${sustainedLowFpsSeconds}d < 38 FPS, dominan ${domBound}).`
      );
      improvementOpportunities.push(
        'Gunakan Penakar Hardware Otomatis / Tier B untuk menyesuaikan DPR & frekuensi bayangan.'
      );
    }
    const audSnap30s =
      this.audioSynth && typeof this.audioSynth.getForensicSnapshot === 'function'
        ? this.audioSynth.getForensicSnapshot(this.hwGovernor || null)
        : null;
    if (audSnap30s) {
      if (audSnap30s.verdict === 'CRITICAL_AUDIO_ERROR') {
        bottleneckFindings.push(`Mesin Web Audio mengalami eror: ${audSnap30s.advisory}`);
      } else if (audSnap30s.realtimeHealth.underrunSuspectCount > 0) {
        bottleneckFindings.push(
          `Terdeteksi ${audSnap30s.realtimeHealth.underrunSuspectCount} gejala audio clock lag/underrun (maks ${audSnap30s.realtimeHealth.maxClockLagMs}ms) dengan ${audSnap30s.dspGraphInventory.totalPersistentNodes} node DSP kontinu.`
        );
        improvementOpportunities.push(
          `Kandidat optimasi Tahap 2: sistem Tier Audio adaptif & pemutusan ${audSnap30s.dspGraphInventory.silentRunningSubBuses} sub-bus DSP diam.`
        );
      } else if (audSnap30s.realtimeHealth.emaUpdateMs > 1.5) {
        bottleneckFindings.push(
          `Update parameter audio main-thread tinggi (${audSnap30s.realtimeHealth.emaUpdateMs}ms rata-rata, maks ${audSnap30s.realtimeHealth.maxUpdateMs}ms).`
        );
      }
    }
    if (bottleneckFindings.length === 0) {
      bottleneckFindings.push(
        `0 bottleneck terdeteksi: pagu Gate-4 terjaga (${avgMainCalls30s}/25 calls), shader pre-warm stabil, dan pipa optik berjalan mulus.`
      );
      improvementOpportunities.push(
        'Pipeline optimal (Single-Pass Batching + Toroidal 3D Grass Culling + Planar Mirror Reflection + Adaptive Governor aktif).'
      );
    }
    if (
      audSnap30s &&
      audSnap30s.hardwareAndContext.initialized &&
      audSnap30s.dspGraphInventory.silentRunningSubBuses >= 15 &&
      audSnap30s.realtimeHealth.underrunSuspectCount === 0
    ) {
      improvementOpportunities.push(
        `Profil DSP Audio (v2.4): ${audSnap30s.dspGraphInventory.totalPersistentNodes} node kontinu (${audSnap30s.dspGraphInventory.activeAudibleSubBuses} bus berbunyi, ${audSnap30s.dspGraphInventory.hibernatedSubBuses || 0} bus hibernasi, ${audSnap30s.dspGraphInventory.silentRunningSubBuses} bus diam berputar, ${audSnap30s.dspGraphInventory.uncleanedTransientAuxNodes} aux transien tersisa).`
      );
    }

    const oldestRel = timeline1s.length > 0 ? Math.abs(timeline1s[0].tRelSec) : 0;
    const capturedSpanSec = Math.min(30, Math.max(timeline1s.length, Math.round(oldestRel)));

    return {
      windowSec: 30,
      capturedSpanSec,
      recordingDesign:
        'Substantive-only 30s rolling blackbox: 1s aggregated TypedArray buckets + high-signal transition/anomaly events (zero raw per-frame spam)',
      forensicSynthesis: {
        overall30s: {
          totalFrames: totalFrames30s,
          avgFps: overallAvgFps,
          avgFrameMs: +overallAvgMs.toFixed(2),
          avgCpuMs: +overallAvgCpuMs.toFixed(2),
          min1sFps,
          worstFrameMs: +worstFrameMs.toFixed(2),
          hitchesOver25ms: totalHitches25,
          stallsOver50ms: totalStalls50,
          sustainedLowFpsSeconds,
          cameraSwitches: camSwitchCount,
          maxCameraSwitchFrameMs: +maxCamSwitchMs.toFixed(2),
          runtimeShaderCompiles
        },
        byCameraGroup,
        byCameraMode,
        pipelineEfficiency: {
          avgMainDrawCalls: avgMainCalls30s,
          drawCallBudgetCeiling: 25,
          drawCallHeadroom: Math.max(0, 25 - avgMainCalls30s),
          avgMainTrianglesK: avgMainTrisK30s,
          avgPlanarReflDrawCalls: avgReflCalls30s,
          planarReflPassActivePct: reflActivePct,
          grass3DVisiblePct: grassActivePct,
          shadowMapRefreshPct: shadowUpdatePct,
          activeShaderPrograms: this.lastKnownPrograms >= 0 ? this.lastKnownPrograms : 0
        },
        audioEfficiency: audSnap30s
          ? {
              verdict: audSnap30s.verdict,
              initialized: audSnap30s.hardwareAndContext.initialized,
              muted: audSnap30s.hardwareAndContext.muted,
              latencyHint: audSnap30s.hardwareAndContext.latencyHint || 'interactive',
              contextState: audSnap30s.hardwareAndContext.contextState,
              sampleRateHz: audSnap30s.hardwareAndContext.sampleRateHz,
              emaUpdateUs: audSnap30s.realtimeHealth.emaUpdateUs,
              maxUpdateMs: audSnap30s.realtimeHealth.maxUpdateMs,
              skippedRedundantParamOps: audSnap30s.realtimeHealth.skippedRedundantParamOps || 0,
              persistentNodes: audSnap30s.dspGraphInventory.totalPersistentNodes,
              activeAudibleSubBuses: audSnap30s.dspGraphInventory.activeAudibleSubBuses,
              hibernatedSubBuses: audSnap30s.dspGraphInventory.hibernatedSubBuses || 0,
              silentRunningSubBuses: audSnap30s.dspGraphInventory.silentRunningSubBuses,
              uncleanedTransientAuxNodes: audSnap30s.dspGraphInventory.uncleanedTransientAuxNodes,
              underrunSuspects: audSnap30s.realtimeHealth.underrunSuspectCount,
              peakLimiterReductionDb: audSnap30s.realtimeHealth.peakLimiterReductionDb
            }
          : null,
        bottleneckFindings,
        improvementOpportunities
      },
      substantiveEvents,
      timeline1s
    };
  }
}

/**
 * Forensik Inisiasi & Boot Granular v2.5 (100% Passive, Zero Runtime Frame Overhead):
 * Breaks down every initiation stage (HTML/importmap -> WebGL2 -> World Sub-Builders ->
 * 3D Grass Bake -> Per-Slice & Per-Pass Shader Compilation -> First-Frame Gap -> Deferred Rain/Audio)
 * into wallMs, blocking cpuMs, yield/GPU waitMs, GPU resource deltas, and automatic bottleneck ranking.
 */
export class NexusBootForensicProfiler {
  constructor(options = {}) {
    this._nowRaw = () =>
      typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
    this._originPerfMs =
      typeof options.originPerfMs === 'number' && Number.isFinite(options.originPerfMs)
        ? options.originPerfMs
        : typeof window !== 'undefined' &&
            window.NexusForensic &&
            typeof window.NexusForensic.getOriginPerfMs === 'function'
          ? window.NexusForensic.getOriginPerfMs()
          : this._nowRaw();
    this.renderer = options.renderer || null;
    this.stages = [];
    this.shaderSlices = [];
    this.shaderMeta = {
      compileMode: 'unknown',
      msaaSamplesDuringWarmup: 0,
      planarReflSizeDuringWarmup: 0,
      uniqueArchetypesCount: 0,
      programsBeforeWarmup: 0,
      programsAfterWarmup: 0,
      totalWarmupWallMs: 0,
      totalWarmupCpuMs: 0
    };
    this.firstFrameGap = {
      prewarmReadyAtMs: 0,
      firstFrameMarkedAtMs: 0,
      secondFrameStartAtMs: 0,
      totalGapMs: 0,
      preloader100YieldAndGpuFlushMs: 0,
      uiDomAndViewportBindCpuMs: 0,
      initialSyncViewportCpuMs: 0,
      postFxResizedOnViewportSync: false,
      initialSyncForensicDomCpuMs: 0,
      rafQueueWaitToFrame1Ms: 0,
      frame1SimAndKinematicsCpuMs: 0,
      frame1PlanarReflectionCpuMs: 0,
      frame1PlanarReflectionRan: false,
      frame1PlanarReflectionCalls: 0,
      frame1MainPostFxRenderCpuMs: 0,
      frame1ShadowMapUpdated: false,
      frame1MainCalls: 0,
      frame1MainTriangles: 0,
      frame1TotalCpuMs: 0,
      programsAtPrewarmEnd: 0,
      programsAfterFrame1: 0,
      lateCompiledProgramsCount: 0,
      lateCompiledPrograms: [],
      frame1ToFrame2GpuPresentWallMs: null
    };
    this.subBuilders = {
      gedungRektorat4Gates: null,
      tropicalGrass3D: null,
      solidWorldCollidersCount: 0
    };
    this.deferredTasks = {
      rainCollisionZBufferBake: {
        status: 'pending',
        startedAtMs: null,
        completedAtMs: null,
        wallDurationMs: null,
        totalCpuMs: null,
        sliceCount: null,
        avgSliceCpuMs: null,
        maxSliceCpuMs: null,
        sliceBudgetMs: null,
        gridSize: null,
        staticMeshCount: null,
        canopyMeshCount: null,
        rawDripCandidateCount: null
      },
      liveDepokWeatherFetch: {
        status: 'pending',
        startedAtMs: null,
        resolvedAtMs: null,
        elapsedMs: null,
        isLiveApi: null,
        labelId: null
      }
    };
    this._knownProgramIds = new Set();
  }

  relMs() {
    return Math.max(0.0, +(this._nowRaw() - this._originPerfMs).toFixed(2));
  }

  attachRenderer(renderer) {
    this.renderer = renderer || null;
    this._snapshotProgramDiff(false);
  }

  _readGpuResourceCounts() {
    const info = this.renderer && this.renderer.info ? this.renderer.info : null;
    return {
      programs: info && Array.isArray(info.programs) ? info.programs.length : 0,
      geometries: info && info.memory ? info.memory.geometries || 0 : 0,
      textures: info && info.memory ? info.memory.textures || 0 : 0
    };
  }

  _snapshotProgramDiff(recordAsKnown = true) {
    const progs =
      this.renderer && this.renderer.info && Array.isArray(this.renderer.info.programs)
        ? this.renderer.info.programs
        : [];
    const added = [];
    for (let i = 0; i < progs.length; i++) {
      const p = progs[i];
      if (!p) continue;
      const pid = p.id !== undefined ? p.id : i;
      if (!this._knownProgramIds.has(pid)) {
        if (recordAsKnown) this._knownProgramIds.add(pid);
        const rawKey = typeof p.cacheKey === 'string' ? p.cacheKey : '';
        added.push({
          id: pid,
          name: String(p.name || p.type || 'WebGLProgram').slice(0, 72),
          cacheKeyHead: rawKey ? rawKey.slice(0, 64) : ''
        });
      }
    }
    if (!recordAsKnown) {
      for (let i = 0; i < progs.length; i++) {
        const p = progs[i];
        if (p) this._knownProgramIds.add(p.id !== undefined ? p.id : i);
      }
    }
    return added;
  }

  _peekPostFrame1Programs() {
    if (!(this.firstFrameGap.firstFrameMarkedAtMs > 0)) return [];
    const progs =
      this.renderer && this.renderer.info && Array.isArray(this.renderer.info.programs)
        ? this.renderer.info.programs
        : [];
    const postFrame1 = [];
    for (let i = 0; i < progs.length; i++) {
      const p = progs[i];
      if (!p) continue;
      const pid = p.id !== undefined ? p.id : i;
      if (!this._knownProgramIds.has(pid)) {
        const rawKey = typeof p.cacheKey === 'string' ? p.cacheKey : '';
        postFrame1.push({
          id: pid,
          name: String(p.name || p.type || 'WebGLProgram').slice(0, 72),
          cacheKeyHead: rawKey ? rawKey.slice(0, 64) : ''
        });
      }
    }
    return postFrame1;
  }

  recordPreBootstrapSpan(collectorInitRelMs, bootstrapStartRelMs, detail = '') {
    const s = Math.max(0.0, Number(collectorInitRelMs) || 0.0);
    const e = Math.max(s, Number(bootstrapStartRelMs) || this.relMs());
    const w = +(e - s).toFixed(2);
    this.stages.push({
      id: 'S00_html_and_importmap',
      category: 'MODULE_LOAD',
      label: 'HTML, Early Collector & Three.js Importmap',
      startMs: +s.toFixed(2),
      endMs: +e.toFixed(2),
      wallMs: w,
      cpuMs: null,
      waitOrYieldMs: null,
      addedPrograms: 0,
      addedGeometries: 0,
      addedTextures: 0,
      detail: detail || 'Parse HTML + muat Three.js CDN & modul entry main.js'
    });
  }

  beginStage(id, category, label) {
    const startRel = this.relMs();
    const startRaw = this._nowRaw();
    const gpuBefore = this._readGpuResourceCounts();
    return {
      id,
      category,
      label,
      startRel,
      startRaw,
      gpuBefore
    };
  }

  endStage(token, opts = {}) {
    if (!token) return null;
    const endRel = this.relMs();
    const wallMs = Math.max(0.0, +(this._nowRaw() - token.startRaw).toFixed(2));
    const explicitCpu =
      typeof opts.cpuMs === 'number' && Number.isFinite(opts.cpuMs) ? Math.max(0.0, opts.cpuMs) : null;
    const explicitWait =
      typeof opts.waitMs === 'number' && Number.isFinite(opts.waitMs) ? Math.max(0.0, opts.waitMs) : null;
    let cpuMs = explicitCpu !== null ? +Math.min(wallMs, explicitCpu).toFixed(2) : wallMs;
    let waitOrYieldMs =
      explicitWait !== null
        ? +Math.min(wallMs, explicitWait).toFixed(2)
        : explicitCpu !== null
          ? +Math.max(0.0, wallMs - cpuMs).toFixed(2)
          : 0.0;
    if (opts.pureAsyncIo === true && explicitCpu === null) {
      cpuMs = null;
      waitOrYieldMs = wallMs;
    }
    const gpuAfter = this._readGpuResourceCounts();
    const entry = {
      id: token.id,
      category: token.category,
      label: token.label,
      startMs: token.startRel,
      endMs: endRel,
      wallMs,
      cpuMs,
      waitOrYieldMs,
      addedPrograms: Math.max(0, gpuAfter.programs - token.gpuBefore.programs),
      addedGeometries: Math.max(0, gpuAfter.geometries - token.gpuBefore.geometries),
      addedTextures: Math.max(0, gpuAfter.textures - token.gpuBefore.textures),
      detail: opts.detail ? String(opts.detail).slice(0, 220) : ''
    };
    this.stages.push(entry);
    return entry;
  }

  beginShaderWarmup(meta = {}) {
    const gpuCounts = this._readGpuResourceCounts();
    this._snapshotProgramDiff(false);
    this.shaderMeta.compileMode = meta.compileMode || 'unknown';
    this.shaderMeta.msaaSamplesDuringWarmup = Number(meta.msaaSamples) || 0;
    this.shaderMeta.planarReflSizeDuringWarmup = Number(meta.planarReflSize) || 0;
    this.shaderMeta.uniqueArchetypesCount = Number(meta.uniqueArchetypesCount) || 0;
    this.shaderMeta.programsBeforeWarmup = gpuCounts.programs;
  }

  recordShaderWarmupStep(stepObj) {
    const newProgs = this._snapshotProgramDiff(true);
    const compileWallMs = Math.max(0.0, +(Number(stepObj.wallMs) || 0.0).toFixed(2));
    const cpuMs = Math.max(0.0, +(Number(stepObj.cpuMs) || 0.0).toFixed(2));
    const yieldAfterMs = Math.max(0.0, +(Number(stepObj.yieldAfterMs) || 0.0).toFixed(2));
    this.shaderSlices.push({
      stepId: String(stepObj.stepId || `step_${this.shaderSlices.length}`),
      phase: String(stepObj.phase || 'compile_slice'),
      archetypes: Array.isArray(stepObj.archetypes) ? stepObj.archetypes.slice(0, 8) : [],
      wallMs: compileWallMs,
      cpuMs,
      yieldAfterMs,
      effectiveCostMs: +(compileWallMs + yieldAfterMs).toFixed(2),
      addedProgramsCount: newProgs.length,
      newPrograms: newProgs
    });
  }

  endShaderWarmup(totalWallMs, totalCpuMs) {
    const gpuCounts = this._readGpuResourceCounts();
    this.shaderMeta.programsAfterWarmup = gpuCounts.programs;
    this.shaderMeta.totalWarmupWallMs = Math.max(0.0, +(Number(totalWallMs) || 0.0).toFixed(2));
    this.shaderMeta.totalWarmupCpuMs = Math.max(0.0, +(Number(totalCpuMs) || 0.0).toFixed(2));
    this.firstFrameGap.prewarmReadyAtMs = this.relMs();
    this.firstFrameGap.programsAtPrewarmEnd = gpuCounts.programs;
  }

  recordFirstFrameSubmitted(metrics = {}) {
    const nowRel = this.relMs();
    const lateProgs = this._snapshotProgramDiff(true);
    const gpuCounts = this._readGpuResourceCounts();
    const g = this.firstFrameGap;
    g.firstFrameMarkedAtMs = nowRel;
    g.totalGapMs = +Math.max(0.0, nowRel - (g.prewarmReadyAtMs || 0)).toFixed(2);
    if (metrics.simAndKinematicsCpuMs !== undefined) {
      g.frame1SimAndKinematicsCpuMs = +Math.max(0.0, Number(metrics.simAndKinematicsCpuMs) || 0).toFixed(2);
    }
    if (metrics.planarReflectionCpuMs !== undefined) {
      g.frame1PlanarReflectionCpuMs = +Math.max(0.0, Number(metrics.planarReflectionCpuMs) || 0).toFixed(2);
    }
    g.frame1PlanarReflectionRan = Boolean(metrics.planarReflectionRan);
    g.frame1PlanarReflectionCalls = Number(metrics.planarReflectionCalls) || 0;
    if (metrics.mainPostFxRenderCpuMs !== undefined) {
      g.frame1MainPostFxRenderCpuMs = +Math.max(0.0, Number(metrics.mainPostFxRenderCpuMs) || 0).toFixed(2);
    }
    g.frame1ShadowMapUpdated = Boolean(metrics.shadowUpdated);
    g.frame1MainCalls = Number(metrics.mainCalls) || 0;
    g.frame1MainTriangles = Number(metrics.mainTriangles) || 0;
    if (metrics.totalCpuMs !== undefined) {
      g.frame1TotalCpuMs = +Math.max(0.0, Number(metrics.totalCpuMs) || 0).toFixed(2);
    }
    g.programsAfterFrame1 = gpuCounts.programs;
    g.lateCompiledProgramsCount = lateProgs.length;
    g.lateCompiledPrograms = lateProgs;
    this._frame1SubmittedRaw = this._nowRaw();
  }

  recordSecondFrameEntered() {
    if (this.firstFrameGap.frame1ToFrame2GpuPresentWallMs !== null) return;
    const nowRaw = this._nowRaw();
    const presentMs =
      typeof this._frame1SubmittedRaw === 'number'
        ? Math.max(0.0, +(nowRaw - this._frame1SubmittedRaw).toFixed(2))
        : 0.0;
    this.firstFrameGap.secondFrameStartAtMs = this.relMs();
    this.firstFrameGap.frame1ToFrame2GpuPresentWallMs = presentMs;
    this.stages.push({
      id: 'S18_frame1_gpu_present',
      category: 'FIRST_FRAME_GAP',
      label: 'Eksekusi GPU & Presentasi Swapchain Frame Pertama',
      startMs: this.firstFrameGap.firstFrameMarkedAtMs,
      endMs: this.firstFrameGap.secondFrameStartAtMs,
      wallMs: presentMs,
      cpuMs: 0.0,
      waitOrYieldMs: presentMs,
      addedPrograms: 0,
      addedGeometries: 0,
      addedTextures: 0,
      detail: 'Interval rAF Frame #1 -> Frame #2 (waktu nyata piksel pertama tampil di layar)'
    });
  }

  recordRainBakeStarted() {
    this.deferredTasks.rainCollisionZBufferBake.status = 'running';
    this.deferredTasks.rainCollisionZBufferBake.startedAtMs = this.relMs();
  }

  recordRainBakeCompleted(bakeResult) {
    const rb = this.deferredTasks.rainCollisionZBufferBake;
    rb.status = 'completed';
    rb.completedAtMs = this.relMs();
    const bf = bakeResult && bakeResult.bakeForensics ? bakeResult.bakeForensics : null;
    if (bf) {
      rb.wallDurationMs = bf.wallElapsedMs ?? null;
      rb.totalCpuMs = bf.totalCpuMs ?? null;
      rb.sliceCount = bf.sliceCount ?? null;
      rb.avgSliceCpuMs = bf.avgSliceCpuMs ?? null;
      rb.maxSliceCpuMs = bf.maxSliceCpuMs ?? null;
      rb.sliceBudgetMs = bf.sliceBudgetMs ?? null;
      rb.gridSize = bf.gridSize ?? null;
      rb.staticMeshCount = bf.staticMeshCount ?? null;
      rb.canopyMeshCount = bf.canopyMeshCount ?? null;
      rb.rawDripCandidateCount = bf.rawDripCandidateCount ?? null;
    }
  }

  recordLiveWeatherStarted() {
    this.deferredTasks.liveDepokWeatherFetch.startedAtMs = this.relMs();
  }

  recordLiveWeatherSettled(liveData) {
    const lw = this.deferredTasks.liveDepokWeatherFetch;
    lw.resolvedAtMs = this.relMs();
    lw.elapsedMs =
      lw.startedAtMs !== null ? +Math.max(0.0, lw.resolvedAtMs - lw.startedAtMs).toFixed(2) : null;
    lw.status = liveData && liveData.isLiveApi ? 'live_api_ok' : 'offline_fallback';
    lw.isLiveApi = Boolean(liveData && liveData.isLiveApi);
    lw.labelId = liveData && liveData.labelId ? String(liveData.labelId) : null;
  }

  buildSnapshot(extra = {}) {
    const audioSynth = extra.audioSynth || null;
    const firstFrameMs =
      this.firstFrameGap.firstFrameMarkedAtMs > 0
        ? this.firstFrameGap.firstFrameMarkedAtMs
        : this.stages.length > 0
          ? this.stages[this.stages.length - 1].endMs
          : this.relMs();
    let maxStageWallBeforePresent = 0.0;
    for (let i = 0; i < this.stages.length; i++) {
      const st0 = this.stages[i];
      if (st0.id !== 'S18_frame1_gpu_present') {
        if (st0.endMs > maxStageWallBeforePresent) maxStageWallBeforePresent = st0.endMs;
        if (st0.wallMs > maxStageWallBeforePresent) maxStageWallBeforePresent = st0.wallMs;
      }
    }
    const denomWall = Math.max(1.0, firstFrameMs, maxStageWallBeforePresent);

    let totalBlockingCpuMs = 0.0;
    let totalAsyncWaitMs = 0.0;
    let longMainThreadBlocksOver50ms = 0;
    let worstCpuStage = null;

    const preFrame2Stages = [];
    for (let i = 0; i < this.stages.length; i++) {
      const st = this.stages[i];
      if (st.id !== 'S18_frame1_gpu_present') {
        if (typeof st.cpuMs === 'number' && Number.isFinite(st.cpuMs)) {
          totalBlockingCpuMs += st.cpuMs;
          if (st.cpuMs >= 50.0) longMainThreadBlocksOver50ms++;
          if (!worstCpuStage || st.cpuMs > worstCpuStage.cpuMs) {
            worstCpuStage = st;
          }
        }
        if (typeof st.waitOrYieldMs === 'number' && Number.isFinite(st.waitOrYieldMs)) {
          totalAsyncWaitMs += st.waitOrYieldMs;
        }
        preFrame2Stages.push(st);
      }
    }

    const denomCpu = Math.max(1.0, totalBlockingCpuMs);
    const enrichedStages = this.stages.map((st) => ({
      id: st.id,
      category: st.category,
      label: st.label,
      startMs: st.startMs,
      endMs: st.endMs,
      wallMs: st.wallMs,
      cpuMs: st.cpuMs,
      waitOrYieldMs: st.waitOrYieldMs,
      pctOfFirstFrameWall:
        st.id === 'S18_frame1_gpu_present' ? null : +((st.wallMs / denomWall) * 100.0).toFixed(1),
      pctOfBlockingCpu:
        st.id !== 'S18_frame1_gpu_present' && typeof st.cpuMs === 'number'
          ? +((st.cpuMs / denomCpu) * 100.0).toFixed(1)
          : null,
      addedPrograms: st.addedPrograms,
      addedGeometries: st.addedGeometries,
      addedTextures: st.addedTextures,
      detail: st.detail
    }));

    const rankedByWall = enrichedStages
      .filter((s) => s.id !== 'S18_frame1_gpu_present')
      .slice()
      .sort((a, b) => b.wallMs - a.wallMs)
      .slice(0, 6)
      .map((s, idx) => ({
        rank: idx + 1,
        id: s.id,
        category: s.category,
        label: s.label,
        wallMs: s.wallMs,
        cpuMs: s.cpuMs,
        waitOrYieldMs: s.waitOrYieldMs,
        pctOfFirstFrameWall: s.pctOfFirstFrameWall,
        addedPrograms: s.addedPrograms
      }));

    const rankedByCpu = enrichedStages
      .filter((s) => s.id !== 'S18_frame1_gpu_present' && typeof s.cpuMs === 'number' && s.cpuMs > 0)
      .slice()
      .sort((a, b) => b.cpuMs - a.cpuMs)
      .slice(0, 6)
      .map((s, idx) => ({
        rank: idx + 1,
        id: s.id,
        category: s.category,
        label: s.label,
        cpuMs: s.cpuMs,
        wallMs: s.wallMs,
        pctOfBlockingCpu: s.pctOfBlockingCpu
      }));

    const rankedShaderSteps = this.shaderSlices
      .slice()
      .sort((a, b) => b.effectiveCostMs - a.effectiveCostMs)
      .slice(0, 6);

    const audForensic =
      audioSynth && typeof audioSynth.getForensicSnapshot === 'function'
        ? audioSynth.getForensicSnapshot(null)
        : null;
    const audHc = audForensic && audForensic.hardwareAndContext ? audForensic.hardwareAndContext : null;

    const presentExtraMs = this.firstFrameGap.frame1ToFrame2GpuPresentWallMs || 0;
    const trueInteractiveMs = +(firstFrameMs + presentExtraMs).toFixed(2);

    let verdict = 'FAST_BOOT';
    if (trueInteractiveMs >= 20000 || totalBlockingCpuMs >= 3000) {
      verdict = 'HEAVY_BOOT';
    } else if (trueInteractiveMs >= 8000 || totalBlockingCpuMs >= 1500) {
      verdict = 'MODERATE_BOOT';
    }

    const topWall = rankedByWall[0] || null;
    const topCpu = rankedByCpu[0] || null;
    const topShader = rankedShaderSteps[0] || null;
    let dominantBottleneck = 'BALANCED_STARTUP';
    if (topWall) {
      if (topWall.id === 'S13_gpu_shader_prewarm') {
        dominantBottleneck = 'GPU_SHADER_DRIVER_COMPILE';
      } else if (topWall.id.startsWith('S14') || topWall.id.startsWith('S16') || topWall.id.startsWith('S17')) {
        dominantBottleneck = 'POST_PREWARM_GPU_FLUSH_OR_FIRST_FRAME';
      } else if (topWall.id === 'S11_tropical_grass_3d') {
        dominantBottleneck = 'GRASS_LAWN_TEXTURE_BAKE_AND_YIELDS';
      } else if (topWall.category === 'WORLD_BUILD') {
        dominantBottleneck = 'CPU_PROCEDURAL_WORLD_GEOMETRY';
      }
    }

    const findings = [];
    if (topWall) {
      findings.push(
        `Tahap jam-dinding terberat #1: ${topWall.label} (${topWall.wallMs}ms / ${topWall.pctOfFirstFrameWall}% dari waktu menuju frame pertama ${firstFrameMs}ms).`
      );
    }
    if (topCpu) {
      findings.push(
        `Blokir CPU main-thread terbesar #1: ${topCpu.label} (${topCpu.cpuMs}ms dari total CPU inisiasi ${totalBlockingCpuMs.toFixed(1)}ms; ${longMainThreadBlocksOver50ms} tahap >50ms).`
      );
    }
    if (topShader) {
      const pNames = topShader.newPrograms.map((p) => p.name).join(', ') || 'tanpa program baru';
      findings.push(
        `Sub-langkah prewarm shader terlambat #1: ${topShader.stepId} (${topShader.effectiveCostMs}ms = compile ${topShader.wallMs}ms + yield ${topShader.yieldAfterMs}ms; +${topShader.addedProgramsCount} prog: ${pNames}).`
      );
    }
    if (this.firstFrameGap.totalGapMs > 250 || this.firstFrameGap.lateCompiledProgramsCount > 0) {
      const lateNames = this.firstFrameGap.lateCompiledPrograms.map((p) => p.name).join(', ');
      findings.push(
        `Jeda pasca-prewarm -> frame 1: ${this.firstFrameGap.totalGapMs}ms (flush preloader ${this.firstFrameGap.preloader100YieldAndGpuFlushMs}ms, UI/viewport ${this.firstFrameGap.uiDomAndViewportBindCpuMs}ms, tunggu rAF ${this.firstFrameGap.rafQueueWaitToFrame1Ms}ms, CPU frame-1 ${this.firstFrameGap.frame1TotalCpuMs}ms, shader telat: +${this.firstFrameGap.lateCompiledProgramsCount}${lateNames ? ' [' + lateNames + ']' : ''}).`
      );
    }
    const postFrame1LateProgs = this._peekPostFrame1Programs();
    if (postFrame1LateProgs.length > 0) {
      const pfNames = postFrame1LateProgs.map((p) => `${p.name}#${p.id}`).join(', ');
      findings.push(
        `Kompilasi shader susulan pasca-Frame-1 (luput dari prewarm): +${postFrame1LateProgs.length} program [${pfNames}].`
      );
    }
    const rb = this.deferredTasks.rainCollisionZBufferBake;
    if (rb.status === 'completed' && rb.totalCpuMs !== null) {
      findings.push(
        `Rain Z-buffer (${rb.gridSize}): total CPU murni ${rb.totalCpuMs}ms dalam ${rb.sliceCount} slice (maks ${rb.maxSliceCpuMs}ms/slice, bentang dinding ${rb.wallDurationMs}ms).`
      );
    }
    if (audHc && audHc.initialized) {
      findings.push(
        `Inisialisasi Web Audio saat diaktifkan: blokir CPU ${audHc.initDurationMs}ms untuk ${audForensic.dspGraphInventory.totalPersistentNodes} node DSP.`
      );
    }

    const findStageEnd = (id) => {
      for (let i = 0; i < enrichedStages.length; i++) {
        if (enrichedStages[i].id === id) return enrichedStages[i].endMs;
      }
      return null;
    };

    return {
      forensicVersion: 'v2.5',
      verdict,
      dominantBootBottleneck: dominantBottleneck,
      bootComplete: this.firstFrameGap.firstFrameMarkedAtMs > 0,
      summary: {
        timeToBootstrapStartMs: findStageEnd('S00_html_and_importmap'),
        timeToWebGL2ReadyMs: findStageEnd('S04_shared_textures_governor'),
        timeToWorldGeometryBuiltMs: findStageEnd('S10_sivitas_npcs_cats_fleet'),
        timeToGrassAndPostFxReadyMs: findStageEnd('S12_postfx_and_initial_sim'),
        timeToShadersPrewarmedMs: this.firstFrameGap.prewarmReadyAtMs || findStageEnd('S13_gpu_shader_prewarm'),
        timeToFirstFrameSubmittedMs: +firstFrameMs.toFixed(2),
        firstFrameToSecondFramePresentMs: this.firstFrameGap.frame1ToFrame2GpuPresentWallMs,
        trueTimeToInteractivePixelsMs: trueInteractiveMs,
        totalBootBlockingCpuMs: +totalBlockingCpuMs.toFixed(2),
        totalBootAsyncAndGpuWaitMs: +totalAsyncWaitMs.toFixed(2),
        longMainThreadBlocksOver50ms,
        heaviestWallStage: topWall,
        heaviestCpuStage: topCpu
      },
      rankedHeaviestByWallMs: rankedByWall,
      rankedHeaviestByCpuMs: rankedByCpu,
      stages: enrichedStages,
      shaderWarmupBreakdown: {
        compileMode: this.shaderMeta.compileMode,
        msaaSamplesDuringWarmup: this.shaderMeta.msaaSamplesDuringWarmup,
        planarReflSizeDuringWarmup: this.shaderMeta.planarReflSizeDuringWarmup,
        uniqueArchetypesCount: this.shaderMeta.uniqueArchetypesCount,
        programsBeforeWarmup: this.shaderMeta.programsBeforeWarmup,
        programsAfterWarmup: this.shaderMeta.programsAfterWarmup,
        totalCompiledPrograms: Math.max(
          0,
          this.shaderMeta.programsAfterWarmup - this.shaderMeta.programsBeforeWarmup
        ),
        totalWarmupWallMs: this.shaderMeta.totalWarmupWallMs,
        totalWarmupCpuMs: this.shaderMeta.totalWarmupCpuMs,
        postFrame1LateCompiledCount: postFrame1LateProgs.length,
        postFrame1LateCompiledPrograms: postFrame1LateProgs,
        rankedSlowestSteps: rankedShaderSteps,
        steps: this.shaderSlices
      },
      firstFrameGapBreakdown: Object.assign({}, this.firstFrameGap),
      subBuilderDetails: {
        gedungRektorat4Gates: this.subBuilders.gedungRektorat4Gates,
        tropicalGrass3D: this.subBuilders.tropicalGrass3D,
        solidWorldCollidersCount: this.subBuilders.solidWorldCollidersCount
      },
      deferredBackgroundTasks: {
        rainCollisionZBufferBake: Object.assign({}, this.deferredTasks.rainCollisionZBufferBake),
        liveDepokWeatherFetch: Object.assign({}, this.deferredTasks.liveDepokWeatherFetch),
        webAudioLazyGraphInit: audHc
          ? {
              initialized: audHc.initialized,
              contextState: audHc.contextState,
              totalBlockingCpuMs: audHc.initDurationMs,
              breakdownMs: audHc.initBreakdownMs || null,
              persistentNodes: audForensic?.dspGraphInventory?.totalPersistentNodes || 0,
              proceduralBufferMemoryKB: audForensic?.dspGraphInventory?.proceduralBufferMemoryKB || 0
            }
          : null
      },
      findings
    };
  }
}

/**
 * Pure Read-Only Engine Telemetry Snapshot (NEVER invokes any 3D render pass or mutates 3D state).
 */
export function createForensicReadOnlyProbe(deps) {
  const {
    recorder,
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
    getWebglContextLost,
    getIsLiveDepok,
    getLatestDepokWeather,
    getLatestDepokClock,
    isDroneCinemaViewFn
  } = deps;

  if (recorder && typeof recorder.attachAudioForensics === 'function' && !recorder.audioSynth && audioSynth) {
    recorder.attachAudioForensics(audioSynth, hwGovernor || null);
  }

  return function readOnlyForensicSnapshot(opts = null) {
    if (opts && opts.clearRolling && recorder && typeof recorder.clearRollingHistory === 'function') {
      recorder.clearRollingHistory();
      return { cleared: true };
    }
    const contextLost = typeof getWebglContextLost === 'function' ? Boolean(getWebglContextLost()) : false;
    const perfNow = recorder.getInstantPerformanceStats();
    const hwSnap = hwGovernor ? hwGovernor.getForensicSnapshot(renderer, {}) : null;
    const rInfo = renderer && renderer.info ? renderer.info : null;

    let glErrStr = 'NO_ERROR (0)';
    let unmaskedGpu = (hwSnap && hwSnap.gpu && hwSnap.gpu.cleanName) || 'WebGL2 GPU';

    try {
      const gl = renderer && typeof renderer.getContext === 'function' ? renderer.getContext() : null;
      if (gl) {
        const errCode = typeof gl.getError === 'function' ? gl.getError() : 0;
        glErrStr = errCode === 0 ? 'NO_ERROR (0)' : `GL_ERR_${errCode}`;
        if (typeof gl.getExtension === 'function') {
          const dbgInfo = gl.getExtension('WEBGL_debug_renderer_info');
          if (dbgInfo && typeof gl.getParameter === 'function') {
            unmaskedGpu = gl.getParameter(dbgInfo.UNMASKED_RENDERER_WEBGL) || unmaskedGpu;
          }
        }
      }
    } catch (_err) {
      // Non-fatal in headless / mock contexts
    }

    const gateRep = (rektoratAsset && rektoratAsset.userData && rektoratAsset.userData.gateReport) || {};
    const liveCalls = postFX
      ? postFX.lastWorldDrawCalls || (rInfo?.render?.calls ?? 0)
      : rInfo?.render?.calls ?? 0;
    const gate1Ok = gateRep.gate1_EnvelopeAndNaN !== false;
    const gate2Ok = gateRep.gate2_ManifoldNormals !== false;
    const gate3Ok = gateRep.gate3_ZeroGapAndKinematics !== false;
    const gate4Ok = liveCalls <= 25;
    const npcCount = Array.isArray(sivitasAI?.npcs) ? sivitasAI.npcs.length : 0;
    const catCount = Array.isArray(sivitasAI?.cats) ? sivitasAI.cats.length : 0;
    const busCount = Array.isArray(sivitasAI?.buses) ? sivitasAI.buses.length : 0;
    const golfCount = Array.isArray(sivitasAI?.golfCarts) ? sivitasAI.golfCarts.length : 0;
    const shipCount = Array.isArray(sivitasAI?.marineVessels) ? sivitasAI.marineVessels.length : 0;
    const planeCount = Array.isArray(sivitasAI?.aircraft) ? sivitasAI.aircraft.length : 0;

    const camMode = camDirector ? camDirector.mode : 'DRONE_6DOF';
    const fpsStatus = camDirector && camDirector.isFPSMode && camDirector.isFPSMode()
      ? camDirector.lastFPSStatus || null
      : null;
    const pilotKind = fpsStatus && fpsStatus.vehicleKind ? fpsStatus.vehicleKind : '';
    const mediumKey = NEXUS_FORENSIC_MEDIUMS[detectKepulauanMediumIdx(camMode, pilotKind)];
    const droneCinemaActive = typeof isDroneCinemaViewFn === 'function' ? Boolean(isDroneCinemaViewFn()) : true;
    const hvsState = weather && weather.hvsOptics && typeof weather.hvsOptics.getTelemetry === 'function'
      ? weather.hvsOptics.getTelemetry()
      : weather && weather.hvsOptics
        ? {
            regime: weather.hvsOptics.regime || 'PHOTOPIC',
            pupilDiameterMM: +(weather.hvsOptics.pupilDiameterMM || 2.5).toFixed(2),
            exposureGain: +(weather.hvsOptics.currentGain || 1.0).toFixed(3)
          }
        : null;

    const depokWeather = typeof getLatestDepokWeather === 'function' ? getLatestDepokWeather() : null;
    const depokClock = typeof getLatestDepokClock === 'function' ? getLatestDepokClock() : null;
    const liveDepokSync = typeof getIsLiveDepok === 'function' ? Boolean(getIsLiveDepok()) : true;
    const atmosTel = audioSynth && typeof audioSynth.getAcousticTelemetry === 'function'
      ? audioSynth.getAcousticTelemetry()
      : null;
    const atmosStage = audioSynth && typeof audioSynth.getAtmosStageSnapshot === 'function'
      ? audioSynth.getAtmosStageSnapshot()
      : null;
    const soundForensics =
      audioSynth && typeof audioSynth.getForensicSnapshot === 'function'
        ? audioSynth.getForensicSnapshot(hwGovernor || null)
        : null;

    const sunDirY = weather?.uniforms?.uTrueSunDir?.value?.y ?? 0;
    const sunElevDeg = +((Math.asin(Math.max(-1, Math.min(1, sunDirY))) * 180) / Math.PI).toFixed(2);
    const moonDirY = weather?.uniforms?.uMoonDir?.value?.y ?? 0;
    const moonElevDeg = +((Math.asin(Math.max(-1, Math.min(1, moonDirY))) * 180) / Math.PI).toFixed(2);

    const domEl = renderer?.domElement;
    const drawingBufStr = hwSnap?.gpu?.drawingBuffer || `${domEl?.width || 1280}×${domEl?.height || 720}`;

    const bootForensics =
      bootProfiler && typeof bootProfiler.buildSnapshot === 'function'
        ? bootProfiler.buildSnapshot({ audioSynth, weather })
        : null;

    return {
      bootForensics,
      rolling30s: recorder.buildRolling30sSnapshot(),
      soundForensics,
      webgl: {
        contextAvailable: !contextLost,
        isWebGL2: Boolean(renderer?.capabilities?.isWebGL2 ?? true),
        unmaskedRenderer: unmaskedGpu,
        gpuClass: hwSnap?.gpu?.gpuClass || 'DESKTOP',
        activeGovernorTier: hwSnap?.gpu?.activeTierId || 'TIER_S',
        governorMode: hwSnap?.gpu?.mode || 'AUTO',
        glError: glErrStr,
        canvasCssSize: {
          w: domEl?.clientWidth || (typeof window !== 'undefined' ? window.innerWidth : 1280),
          h: domEl?.clientHeight || (typeof window !== 'undefined' ? window.innerHeight : 720)
        },
        drawingBuffer: drawingBufStr,
        pixelRatio: hwSnap?.gpu?.pixelRatio ?? 1,
        msaaSamples: postFX ? postFX.msaaSamples : 4,
        planarReflResolution: postFX ? `${postFX.planarReflSize}×${postFX.planarReflSize}` : '768×768',
        shadowMapSize: hwSnap?.gpu?.shadowMapSize || 2048,
        renderStats: rInfo
          ? {
              worldCalls: liveCalls,
              planarReflCalls: recorder.lastReflCalls || 0,
              triangles: postFX?.lastWorldTriangles || rInfo.render?.triangles || 0,
              geometries: rInfo.memory?.geometries || 0,
              textures: rInfo.memory?.textures || 0,
              programs: Array.isArray(rInfo.programs) ? rInfo.programs.length : 0
            }
          : null
      },
      performance: {
        avgFps: perfNow.avgFps,
        minFps: perfNow.minFps,
        low5Fps60s: hwSnap?.window60s?.low5Fps ?? perfNow.minFps,
        avgFrameMs: perfNow.avgFrameMs,
        p95FrameMs: perfNow.p95FrameMs,
        maxFrameMs: perfNow.maxFrameMs,
        avgCpuMs60s: hwSnap?.window60s?.avgCpuMs ?? 0,
        bottleneck60s: hwSnap?.window60s?.bottleneck || 'Optimal / VSync',
        sparkline60s: hwSnap?.window60s?.sparkline || '▅▅▅',
        totalFramesRendered: perfNow.totalFramesRendered
      },
      worldLogicalAudit: {
        auditPassed: gate1Ok && gate2Ok && gate3Ok && gate4Ok && npcCount >= 60,
        gates: {
          gate1_Envelope108x124x50m: gate1Ok,
          gate2_WatertightManifold: gate2Ok,
          gate3_ZeroSubdomainGap: gate3Ok,
          gate4_DrawCallsWithin25: gate4Ok,
          liveWorldDrawCalls: liveCalls,
          budgetCeiling: 25
        },
        counts: {
          sivitasHumans18Joint: npcCount,
          campusCats: catCount,
          bikunBuses: busCount,
          golfCarts: golfCount,
          shuttleFerry: sivitasAI?.shuttleFerry ? 1 : 0,
          miniYacht: sivitasAI?.miniYacht ? 1 : 0,
          offshoreShips: shipCount,
          scheduledAircraft: planeCount,
          electricHeliTaxi: sivitasAI?.heliTaxi ? 1 : 0,
          grassPatches: grassField?.bladeCount || 0,
          grassCurvedBlades: (grassField?.bladeCount || 0) * 6
        }
      },
      engineAndWorld: {
        cameraAndOptics: {
          camMode,
          cameraGroup: camModeToForensicGroup(camMode),
          medium: mediumKey,
          fpsSubjectLabel: fpsStatus?.label || null,
          pilotSlot: camDirector?.fpsPilotPreferredSlot ?? 0,
          fovDeg: camera ? +camera.fov.toFixed(1) : 45,
          pos: camera
            ? {
                x: +camera.position.x.toFixed(2),
                y: +camera.position.y.toFixed(2),
                z: +camera.position.z.toFixed(2)
              }
            : { x: 0, y: 0, z: 0 },
          speedMS: camDirector ? +(Number(camDirector.lastSpeedMS || camDirector._avataSpeed || 0)).toFixed(2) : 0,
          dualOpticsMode: postFX ? postFX.activeModeLabel : droneCinemaActive ? 'DRONE_MAVIC3_CINEMA_GRADE' : 'HUMAN_EYE_TRUE_COLOR',
          droneCinemaBlend: postFX ? +postFX.droneCinemaBlend.toFixed(3) : droneCinemaActive ? 1 : 0,
          planarWaterReflectionActive: Boolean(recorder.lastReflRan),
          grassFieldVisible: Boolean(grassField?.mesh?.visible),
          humanEyeAdaptation: hvsState
        },
        astronomyAndDepokWeather: {
          liveDepokSync,
          wibClock: depokClock?.formattedHMS || depokClock?.formattedHM || null,
          solarHourWIB: weather ? +weather.currentSolarHour.toFixed(3) : 12,
          sunElevationDeg: sunElevDeg,
          moonElevationDeg: moonElevDeg,
          nightIlluminationLevel: weather ? +weather.nightIlluminationLevel.toFixed(3) : 0,
          weatherMode: weather?.currentWeather || 'CLEAR',
          weatherSpectrumPct: weather ? Math.round(weather.getWeatherSpectrum() * 100) : 0,
          seaWaveHeightM: weather ? +weather.getSeaWaveHeightMeters().toFixed(2) : 0.18,
          seaManualOverride: weather ? Boolean(weather.isManualSeaState()) : false,
          rainCollisionBaked: Boolean(weather?.collisionBake),
          depokStationSource: depokWeather
            ? {
                isLiveOpenMeteo: Boolean(depokWeather.isLiveApi),
                label: depokWeather.labelId,
                tempC: depokWeather.temperatureC,
                humidityPct: depokWeather.humidityPct,
                windKmh: depokWeather.windSpeedKmh
              }
            : null
        },
        landmarksAndMedia: {
          fountainProgram: rotundaFountain?.activeModeLabel || 'ACTIVE',
          videotronMode: videotron?.displayMode || 'AUTO',
          videotronDetectedFormat: videotron?.detectedKind || 'SINGLE_16_9',
          videotronMuted: videotron ? Boolean(videotron.isMuted) : true
        },
        spatialAudioAtmos714: atmosTel
          ? {
              enabled: Boolean(atmosTel.enabled),
              layout: atmosTel.atmosLayout || '7.1.4',
              spatialOutputMode: atmosStage?.spatialMode || 'BINAURAL',
              perspectiveId: atmosTel.perspectiveId,
              zoneId: atmosTel.zoneId,
              reverbSendPct: atmosTel.reverbSendPct,
              dominantSource: atmosTel.dominantSourceLabel,
              dominantDoppler: atmosTel.dominantDoppler,
              earBedPct: Math.round((atmosStage?.earEnergy01 || 0) * 100),
              heightDomePct: Math.round((atmosStage?.heightEnergy01 || 0) * 100),
              subLfePct: Math.round((atmosStage?.lfeEnergy01 || 0) * 100)
            }
          : null
      },
      governor60sRing: hwSnap ? hwSnap.window60s : null
    };
  };
}
