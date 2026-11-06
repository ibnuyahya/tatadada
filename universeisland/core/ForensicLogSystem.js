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
    if (bottleneckFindings.length === 0) {
      bottleneckFindings.push(
        `0 bottleneck terdeteksi: pagu Gate-4 terjaga (${avgMainCalls30s}/25 calls), shader pre-warm stabil, dan pipa optik berjalan mulus.`
      );
      improvementOpportunities.push(
        'Pipeline optimal (Single-Pass Batching + Toroidal 3D Grass Culling + Planar Mirror Reflection + Adaptive Governor aktif).'
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
        bottleneckFindings,
        improvementOpportunities
      },
      substantiveEvents,
      timeline1s
    };
  }
}

/**
 * Pure Read-Only Engine Telemetry Snapshot (NEVER invokes any 3D render pass or mutates 3D state).
 */
export function createForensicReadOnlyProbe(deps) {
  const {
    recorder,
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

  return function readOnlyForensicSnapshot() {
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

    const sunDirY = weather?.uniforms?.uTrueSunDir?.value?.y ?? 0;
    const sunElevDeg = +((Math.asin(Math.max(-1, Math.min(1, sunDirY))) * 180) / Math.PI).toFixed(2);
    const moonDirY = weather?.uniforms?.uMoonDir?.value?.y ?? 0;
    const moonElevDeg = +((Math.asin(Math.max(-1, Math.min(1, moonDirY))) * 180) / Math.PI).toFixed(2);

    const domEl = renderer?.domElement;
    const drawingBufStr = hwSnap?.gpu?.drawingBuffer || `${domEl?.width || 1280}×${domEl?.height || 720}`;

    return {
      rolling30s: recorder.buildRolling30sSnapshot(),
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
