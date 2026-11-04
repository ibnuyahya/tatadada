/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v6.14 — 3D Binaural HRTF Physical Acoustic Engine)
 * ./core/AudioSynthesizer.js
 *
 * 100% Procedural Web Audio API 3D Spatial & Physical Sound Engine for Kepulauan UI Depok:
 * - Real-time 3D AudioListener pose & orthonormal orientation coupled to Three.js camera
 * - Binaural HRTF PannerNode pool for dynamic vehicles, aircraft, vessels, & hydrological landmarks
 * - Relativistic Acoustic Doppler Shift (c = 343.0 m/s) & ISO 9613-1 Distance Air Absorption
 * - Procedural Stereo Convolution Reverb & Architectural Roof/Tunnel/Cockpit Occlusion:
 *   Subsea Underpass Bore, Jembatan Teksas Steel-Polycarbonate Truss, Sheltered Colonnades, & FPS Cabins
 * - Multi-Emitter Propulsion Synthesis: Bis Kuning (+ pneumatic brake hiss), Mobil Golf PMSM Inverter,
 *   PK-UIE Hexa-Rotor eVTOL BVI thump, Cessna 172 piston prop, Citation Learjet twin-turbofan, & Ferry/Yacht
 * - Dual-Drone Aero-Acoustics: Mavic 3 Pro low-RPM hum vs DJI Avata 2 asymmetric quad + obstacle whoosh
 * - Material-Aware Footsteps (Stone, Asphalt, EPDM Track, Steel/Wood Deck, Concrete Bore, Grass + Wet Splash)
 * - Diurnal Nusantara Bio-Acoustics (Daytime tropical songbirds, Dusk/Night crickets & frogs, Campus Cats)
 * - Directional 3D HRTF Speed-of-Sound Thunder (delta_t = d / 343 m/s + N-wave plasma crack + cloud echoes)
 */

const SPEED_OF_SOUND_MS = 343.0;

function clamp(val, min, max) {
  return val < min ? min : val > max ? max : val;
}

function smooth01(edge0, edge1, x) {
  const t = clamp((x - edge0) / Math.max(1e-5, edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

export class AudioSynthesizer {
  static SPEED_OF_SOUND = SPEED_OF_SOUND_MS;

  /**
   * Computes exact acoustic Doppler frequency ratio & ISO 9613-1 distance air-absorption lowpass cutoff.
   */
  static computeDopplerAndAbsorption(
    srcX,
    srcY,
    srcZ,
    srcVx,
    srcVy,
    srcVz,
    lisX,
    lisY,
    lisZ,
    lisVx,
    lisVy,
    lisVz
  ) {
    const dx = lisX - srcX;
    const dy = lisY - srcY;
    const dz = lisZ - srcZ;
    const dist = Math.hypot(dx, dy, dz);
    const invD = 1.0 / Math.max(0.65, dist);
    const ux = dx * invD;
    const uy = dy * invD;
    const uz = dz * invD;

    // Radial component of source velocity toward listener (+ = approaching)
    const vS = clamp(srcVx * ux + srcVy * uy + srcVz * uz, -118.0, 118.0);
    // Radial component of listener velocity toward source (+ = approaching)
    const vL = clamp(-(lisVx * ux + lisVy * uy + lisVz * uz), -118.0, 118.0);

    const rawDoppler = (SPEED_OF_SOUND_MS + vL) / Math.max(120.0, SPEED_OF_SOUND_MS - vS);
    const doppler = clamp(rawDoppler, 0.64, 1.54);

    // Humid tropical air high-frequency absorption (~28°C, 75% RH Depok atmosphere)
    const airCutoffHz = clamp(16800.0 / (1.0 + dist / 46.0), 360.0, 16800.0);

    return { dist, doppler, airCutoffHz, radialApproachMs: vS + vL };
  }

  /**
   * Classifies the listener's 3D architectural & environmental acoustic space.
   */
  static classifyAcousticZone(x, y, z, camMode = 'DRONE_6DOF', isRoofSheltered = false) {
    const inUnderpassCorridorX = Math.abs(x - -356.0) <= 5.45;
    const inUnderpassBoreZ = z >= 322.0 && z <= 628.0;
    const inUnderpassRampZ = (z >= 270.0 && z < 322.0) || (z > 628.0 && z <= 672.0);

    const isCockpitClosed = camMode === 'FPS_BUS_DRIVER' || camMode === 'FPS_PILOT';
    const isGolfCanopy = camMode === 'FPS_GOLF_DRIVER';

    if (inUnderpassCorridorX && inUnderpassBoreZ && y < -1.85) {
      return {
        zoneId: 'SUBSEA_TUNNEL',
        labelId: isCockpitClosed ? 'Kabin dalam Terowongan Laut' : 'Terowongan Bawah Laut',
        reverbSend: isCockpitClosed ? 0.48 : 0.64,
        weatherOcclusionGain: 0.08,
        weatherLowpassHz: 230.0,
        tunnelHum: 1.0,
        patterBoost: 0.0
      };
    }

    if (
      Math.abs(x - -28.0) <= 2.85 &&
      z >= 168.0 &&
      z <= 323.0 &&
      y >= 3.4 &&
      y <= 12.25
    ) {
      return {
        zoneId: 'TEKSAS_TRUSS',
        labelId: 'Lorong Jembatan Teksas',
        reverbSend: 0.38,
        weatherOcclusionGain: 0.54,
        weatherLowpassHz: 1420.0,
        tunnelHum: 0.0,
        patterBoost: 1.32
      };
    }

    if (isCockpitClosed) {
      return {
        zoneId: 'COCKPIT_INTERIOR',
        labelId: camMode === 'FPS_PILOT' ? 'Kokpit Kedap Pesawat/eVTOL' : 'Kabin Bis Kuning UI',
        reverbSend: 0.20,
        weatherOcclusionGain: 0.26,
        weatherLowpassHz: 520.0,
        tunnelHum: 0.0,
        patterBoost: 0.55
      };
    }

    if (isGolfCanopy) {
      return {
        zoneId: 'GOLF_CANOPY',
        labelId: 'Kanopi Mobil Golf UI',
        reverbSend: 0.16,
        weatherOcclusionGain: 0.68,
        weatherLowpassHz: 1850.0,
        tunnelHum: 0.0,
        patterBoost: 1.18
      };
    }

    if (inUnderpassCorridorX && inUnderpassRampZ && y < 0.65) {
      return {
        zoneId: 'UNDERPASS_RAMP',
        labelId: 'Ramp Beton Underpass',
        reverbSend: 0.34,
        weatherOcclusionGain: 0.78,
        weatherLowpassHz: 2800.0,
        tunnelHum: 0.35,
        patterBoost: 0.92
      };
    }

    if (isRoofSheltered && y < 8.8) {
      return {
        zoneId: 'ROOF_SHELTERED',
        labelId: 'Kolonade / Selasar Beratap',
        reverbSend: 0.29,
        weatherOcclusionGain: 0.45,
        weatherLowpassHz: 1120.0,
        tunnelHum: 0.0,
        patterBoost: 1.24
      };
    }

    if (y > 40.0) {
      return {
        zoneId: 'HIGH_ALTITUDE',
        labelId: 'Udara Terbuka Tinggi',
        reverbSend: 0.04,
        weatherOcclusionGain: 0.92,
        weatherLowpassHz: 9600.0,
        tunnelHum: 0.0,
        patterBoost: 0.15
      };
    }

    return {
      zoneId: 'OPEN_AIR',
      labelId: 'Ruang Terbuka HRTF',
      reverbSend: 0.11,
      weatherOcclusionGain: 1.0,
      weatherLowpassHz: 13500.0,
      tunnelHum: 0.0,
      patterBoost: 1.0
    };
  }

  constructor() {
    this.ctx = null;
    this.enabled = false;
    this.masterVolume = 0.80; // 0.0 .. 1.0 user slider

    // Master & Space Routing Nodes
    this.masterGain = null;
    this.masterCompressor = null;
    this.worldSumGain = null;
    this.convolverNode = null;
    this.reverbSendGain = null;
    this.reverbReturnGain = null;

    // Weather & Environmental Buses
    this.weatherOccFilter = null;
    this.weatherOccGain = null;
    this.breezeGain = null;
    this.breezeFilter = null;
    this.rainRoarGain = null;
    this.rainRoarFilter = null;
    this.rainPatterGain = null;
    this.rainPatterFilter = null;
    this.surfFilter = null;
    this.surfGain = null;
    this.tunnelHumGain = null;
    this.cricketGain = null;

    // Dual-Drone Buses (DJI Avata 2 Cinewhoop + DJI Mavic 3 Pro + Obstacle Proximity Whoosh)
    this.avataMotorOscA = null;
    this.avataMotorOscB = null;
    this.avataMotorOscC = null;
    this.avataMotorFilter = null;
    this.avataMotorGain = null;
    this.avataWashFilter = null;
    this.avataWashGain = null;
    this.proxWhooshFilter = null;
    this.proxWhooshGain = null;

    this.mavicOscA = null;
    this.mavicOscB = null;
    this.mavicFilter = null;
    this.mavicGain = null;

    // Feline Purr Bus (when in FPS_CAT or right next to a lounging cat)
    this.catPurrGain = null;
    this.catPurrOsc = null;
    this.catPurrLfo = null;

    // Legacy aliases & 3D HRTF Emitter Slots
    this.fountainFilter = null;
    this.fountainGain = null;
    this.waterfallFilter = null;
    this.waterfallGain = null;
    this._emitters = null;

    // Procedural Noise Buffers
    this._pinkBuffer = null;
    this._patterBuffer = null;
    this._thunderTimers = [];

    // Zero-GC Listener & Entity Kinematics State
    this._lis = {
      x: 0.0,
      y: 22.0,
      z: 100.0,
      vx: 0.0,
      vy: 0.0,
      vz: 0.0,
      fwdX: 0.0,
      fwdY: 0.0,
      fwdZ: -1.0,
      upX: 0.0,
      upY: 1.0,
      upZ: 0.0,
      speedMs: 0.0,
      init: false
    };
    this._weatherState = {
      rain: 0.0,
      storm: 0.0,
      windSpeed: 0.85,
      wetness: 0.0,
      nightFactor: 0.0,
      lightningDirX: 0.6,
      lightningDirY: 0.34,
      lightningDirZ: 0.7
    };
    this._zoneState = AudioSynthesizer.classifyAcousticZone(0, 22, 100, 'DRONE_6DOF', false);

    this._entityTracks = {
      bus: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, prevSpd: 0, init: false, brakeCooldown: 0 },
      cart: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, prevSpd: 0, init: false },
      heli: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, prevSpd: 0, init: false },
      air: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, prevSpd: 0, init: false, id: '' },
      marine: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, prevSpd: 0, init: false, id: '' }
    };

    // Stride & Bio-Acoustic Timers
    this._strideAccumM = 0.0;
    this._stepFootSide = 1;
    this._birdTimer = 1.8;
    this._frogTimer = 2.6;
    this._catMeowTimer = 5.5;
    this._whooshCooldown = 0.0;
    this._simClock = 0.0;

    // Live Public Telemetry Snapshot
    this._telemetry = {
      enabled: false,
      masterVolume: 0.80,
      masterVolumePct: 80,
      zoneId: 'OPEN_AIR',
      zoneLabel: 'Ruang Terbuka HRTF',
      zoneLabelId: 'Ruang Terbuka HRTF',
      reverbSendPct: 11,
      weatherOcclusionPct: 0,
      active3DSources: 7,
      dominantSourceLabel: 'Air Mancur & Alam',
      dominantDoppler: 1.0,
      listenerSpeedMs: 0.0,
      surfaceMaterial: 'AIR'
    };
  }

  _targetMasterLinearGain() {
    if (!this.enabled) return 0.0;
    // Perceptual curve: 80% slider -> 0.32 linear headroom into safety compressor
    const v = clamp(this.masterVolume, 0.0, 1.0);
    return Math.pow(v, 1.35) * 0.44;
  }

  setMasterVolume(vol01) {
    this.masterVolume = clamp(Number(vol01) || 0.0, 0.0, 1.0);
    this._telemetry.masterVolume = this.masterVolume;
    this._telemetry.masterVolumePct = Math.round(this.masterVolume * 100);
    if (this.ctx && this.masterGain) {
      this.masterGain.gain.setTargetAtTime(this._targetMasterLinearGain(), this.ctx.currentTime, 0.06);
    }
    return this.masterVolume;
  }

  getMasterVolume() {
    return this.masterVolume;
  }

  getAcousticTelemetry() {
    this._telemetry.enabled = this.enabled;
    this._telemetry.masterVolume = this.masterVolume;
    this._telemetry.masterVolumePct = Math.round(this.masterVolume * 100);
    return this._telemetry;
  }

  toggle(forceState = undefined) {
    if (!this.ctx) {
      this._initContext();
    }
    if (!this.ctx) return false;

    this.enabled = typeof forceState === 'boolean' ? forceState : !this.enabled;
    this._telemetry.enabled = this.enabled;

    if (this.enabled && this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
    const target = this._targetMasterLinearGain();
    this.masterGain.gain.setTargetAtTime(target, this.ctx.currentTime, 0.14);
    return this.enabled;
  }

  _setParam(param, value, now, tau = 0.06, instant = false) {
    if (!param || !Number.isFinite(value)) return;
    if (instant) {
      if (typeof param.cancelScheduledValues === 'function') {
        param.cancelScheduledValues(now);
      }
      param.setValueAtTime(value, now);
    } else {
      param.setTargetAtTime(value, now, tau);
    }
  }

  _setPannerPosition(panner, x, y, z, now, instant = false) {
    if (!panner || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
    if (panner.positionX && panner.positionY && panner.positionZ) {
      this._setParam(panner.positionX, x, now, 0.032, instant);
      this._setParam(panner.positionY, y, now, 0.032, instant);
      this._setParam(panner.positionZ, z, now, 0.032, instant);
    } else if (typeof panner.setPosition === 'function') {
      panner.setPosition(x, y, z);
    }
  }

  _create3DEmitter(opts = {}) {
    const airLP = this.ctx.createBiquadFilter();
    airLP.type = 'lowpass';
    airLP.frequency.value = opts.initCutoff || 8500;
    airLP.Q.value = 0.5;

    const gain = this.ctx.createGain();
    gain.gain.value = 0.0;

    const panner = this.ctx.createPanner();
    panner.panningModel = 'HRTF';
    panner.distanceModel = 'inverse';
    panner.refDistance = opts.refDistance || 9.0;
    panner.maxDistance = opts.maxDistance || 1400.0;
    panner.rolloffFactor = opts.rolloffFactor || 1.05;
    panner.coneInnerAngle = 360;
    panner.coneOuterAngle = 360;

    const revSend = this.ctx.createGain();
    revSend.gain.value = opts.reverbWeight ?? 0.35;

    airLP.connect(gain);
    gain.connect(panner);
    panner.connect(this.worldSumGain);
    panner.connect(revSend);
    revSend.connect(this.reverbSendGain);

    return { input: airLP, airLP, gain, panner, revSend };
  }

  /**
   * Builds a physically plausible stereo Impulse Response (early concrete/stone reflections +
   * exponentially damped diffuse tail) for tunnels, bridge trusses, and colonnades.
   */
  _buildStereoImpulseResponse(sr) {
    const durationSec = 1.55;
    const len = Math.floor(sr * durationSec);
    const ir = this.ctx.createBuffer(2, len, sr);
    const left = ir.getChannelData(0);
    const right = ir.getChannelData(1);

    // Discrete early wall/vault reflections (11..67 ms) with stereo decorrelation
    const earlyTaps = [
      { ms: 11.2, gL: 0.52, gR: 0.34 },
      { ms: 18.7, gL: -0.36, gR: 0.49 },
      { ms: 28.4, gL: 0.33, gR: -0.38 },
      { ms: 41.9, gL: -0.26, gR: 0.29 },
      { ms: 56.5, gL: 0.21, gR: -0.24 },
      { ms: 73.0, gL: 0.16, gR: 0.18 }
    ];
    for (let t = 0; t < earlyTaps.length; t++) {
      const idx = Math.min(len - 1, Math.floor((earlyTaps[t].ms * 0.001) * sr));
      left[idx] += earlyTaps[t].gL;
      right[idx] += earlyTaps[t].gR;
    }

    let lpL = 0.0;
    let lpR = 0.0;
    for (let i = 0; i < len; i++) {
      const tSec = i / sr;
      // Progressive HF absorption in concrete/stone enclosure
      const hfAlpha = clamp(0.46 - tSec * 0.22, 0.12, 0.46);
      const wL = Math.random() * 2.0 - 1.0;
      const wR = Math.random() * 2.0 - 1.0;
      lpL += (wL - lpL) * hfAlpha;
      lpR += (wR - lpR) * hfAlpha;
      const env = Math.exp(-tSec * 3.85) * (1.0 - Math.exp(-tSec * 45.0));
      left[i] += lpL * env * 0.26;
      right[i] += lpR * env * 0.26;
    }
    return ir;
  }

  _initContext() {
    if (typeof window === 'undefined') return;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;

    this.ctx = new AudioCtx();
    const sr = this.ctx.sampleRate;

    // 0. Master Safety Mastering Compressor -> Master Volume Gain -> Destination
    this.worldSumGain = this.ctx.createGain();
    this.worldSumGain.gain.value = 1.0;

    this.masterCompressor = this.ctx.createDynamicsCompressor();
    this.masterCompressor.threshold.value = -9.0;
    this.masterCompressor.knee.value = 12.0;
    this.masterCompressor.ratio.value = 4.5;
    this.masterCompressor.attack.value = 0.004;
    this.masterCompressor.release.value = 0.16;

    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = 0.0;

    this.worldSumGain.connect(this.masterCompressor);
    this.masterCompressor.connect(this.masterGain);
    this.masterGain.connect(this.ctx.destination);

    // 0b. Shared Stereo Convolution Reverb Bus (Tunnels, Bridge Trusses, Colonnades, Thunder Echoes)
    this.reverbSendGain = this.ctx.createGain();
    this.reverbSendGain.gain.value = 0.11;

    this.convolverNode = this.ctx.createConvolver();
    this.convolverNode.buffer = this._buildStereoImpulseResponse(sr);

    this.reverbReturnGain = this.ctx.createGain();
    this.reverbReturnGain.gain.value = 0.92;

    this.reverbSendGain.connect(this.convolverNode);
    this.convolverNode.connect(this.reverbReturnGain);
    this.reverbReturnGain.connect(this.worldSumGain);

    // 1. Procedural 4-Second Seamless Pink-Noise & Micro-Droplet Patter Buffers
    const bufferSize = sr * 4;
    const pinkBuffer = this.ctx.createBuffer(1, bufferSize, sr);
    const pinkOut = pinkBuffer.getChannelData(0);
    let b0 = 0.0;
    let b1 = 0.0;
    let b2 = 0.0;
    for (let i = 0; i < bufferSize; i++) {
      const white = Math.random() * 2.0 - 1.0;
      b0 = 0.99765 * b0 + white * 0.099046;
      b1 = 0.963 * b1 + white * 0.2965164;
      b2 = 0.57 * b2 + white * 1.0526913;
      pinkOut[i] = (b0 + b1 + b2 + white * 0.1848) * 0.075;
    }
    // Crossfade final 512 samples for zero loop-seam click
    for (let k = 0; k < 512; k++) {
      const a = k / 512;
      pinkOut[bufferSize - 512 + k] = pinkOut[bufferSize - 512 + k] * (1.0 - a) + pinkOut[k] * a;
    }
    this._pinkBuffer = pinkBuffer;

    const patterBuffer = this.ctx.createBuffer(1, bufferSize, sr);
    const patterOut = patterBuffer.getChannelData(0);
    let env = 0.0;
    for (let i = 0; i < bufferSize; i++) {
      if (Math.random() < 0.0042) {
        env = Math.min(1.0, env + 0.35 + Math.random() * 0.65);
      }
      env *= 0.965;
      patterOut[i] = (Math.random() * 2.0 - 1.0) * env * 0.26;
    }
    this._patterBuffer = patterBuffer;

    // 2. Architectural Weather Occlusion Sub-Bus (Muffles outdoor wind/rain inside tunnels & cockpits!)
    this.weatherOccFilter = this.ctx.createBiquadFilter();
    this.weatherOccFilter.type = 'lowpass';
    this.weatherOccFilter.frequency.value = 13500;

    this.weatherOccGain = this.ctx.createGain();
    this.weatherOccGain.gain.value = 1.0;

    this.weatherOccFilter.connect(this.weatherOccGain);
    this.weatherOccGain.connect(this.worldSumGain);

    // Bus 1: Campus Breeze, Canopy Rustle & Turbulent Storm Wind Squall
    const breezeSrc = this.ctx.createBufferSource();
    breezeSrc.buffer = pinkBuffer;
    breezeSrc.loop = true;
    this.breezeFilter = this.ctx.createBiquadFilter();
    this.breezeFilter.type = 'lowpass';
    this.breezeFilter.frequency.value = 390;
    this.breezeGain = this.ctx.createGain();
    this.breezeGain.gain.value = 0.44;
    breezeSrc.connect(this.breezeFilter);
    this.breezeFilter.connect(this.breezeGain);
    this.breezeGain.connect(this.weatherOccFilter);
    breezeSrc.start();

    // Bus 2: Tropical Rain Downpour Wash
    const roarSrc = this.ctx.createBufferSource();
    roarSrc.buffer = pinkBuffer;
    roarSrc.loop = true;
    this.rainRoarFilter = this.ctx.createBiquadFilter();
    this.rainRoarFilter.type = 'bandpass';
    this.rainRoarFilter.frequency.value = 960;
    this.rainRoarFilter.Q.value = 0.62;
    this.rainRoarGain = this.ctx.createGain();
    this.rainRoarGain.gain.value = 0.0;
    roarSrc.connect(this.rainRoarFilter);
    this.rainRoarFilter.connect(this.rainRoarGain);
    this.rainRoarGain.connect(this.weatherOccFilter);
    roarSrc.start();

    // Bus 3: Crisp Nearby Raindrop Impact & Terracotta / Polycarbonate Eave-Drip Patter
    const patterSrc = this.ctx.createBufferSource();
    patterSrc.buffer = patterBuffer;
    patterSrc.loop = true;
    this.rainPatterFilter = this.ctx.createBiquadFilter();
    this.rainPatterFilter.type = 'highpass';
    this.rainPatterFilter.frequency.value = 2100;
    this.rainPatterGain = this.ctx.createGain();
    this.rainPatterGain.gain.value = 0.0;
    patterSrc.connect(this.rainPatterFilter);
    this.rainPatterFilter.connect(this.rainPatterGain);
    this.rainPatterGain.connect(this.worldSumGain);
    patterSrc.start();

    // Bus 4: 360° Archipelago Shoreline Surf & Low-Skim Sea Swell Wash
    const surfSrc = this.ctx.createBufferSource();
    surfSrc.buffer = pinkBuffer;
    surfSrc.loop = true;
    surfSrc.playbackRate.value = 0.82;
    this.surfFilter = this.ctx.createBiquadFilter();
    this.surfFilter.type = 'bandpass';
    this.surfFilter.frequency.value = 420;
    this.surfFilter.Q.value = 0.55;
    this.surfGain = this.ctx.createGain();
    this.surfGain.gain.value = 0.0;
    surfSrc.connect(this.surfFilter);
    this.surfFilter.connect(this.surfGain);
    this.surfGain.connect(this.weatherOccFilter);
    surfSrc.start();

    // Bus 5: Subsea Underpass Jet-Fan Ventilation & Hydrostatic Bore Ambiance
    const tunOscA = this.ctx.createOscillator();
    tunOscA.type = 'sine';
    tunOscA.frequency.value = 62.5;
    const tunOscB = this.ctx.createOscillator();
    tunOscB.type = 'triangle';
    tunOscB.frequency.value = 125.2;
    const tunNoise = this.ctx.createBufferSource();
    tunNoise.buffer = pinkBuffer;
    tunNoise.loop = true;
    const tunLp = this.ctx.createBiquadFilter();
    tunLp.type = 'lowpass';
    tunLp.frequency.value = 210;
    this.tunnelHumGain = this.ctx.createGain();
    this.tunnelHumGain.gain.value = 0.0;
    const tunSubGain = this.ctx.createGain();
    tunSubGain.gain.value = 0.28;
    tunOscA.connect(tunSubGain);
    tunOscB.connect(tunSubGain);
    tunSubGain.connect(tunLp);
    tunNoise.connect(tunLp);
    tunLp.connect(this.tunnelHumGain);
    this.tunnelHumGain.connect(this.worldSumGain);
    this.tunnelHumGain.connect(this.reverbSendGain);
    tunOscA.start();
    tunOscB.start();
    tunNoise.start();

    // Bus 6: Diurnal Nocturnal Nusantara Cricket Stridulation Chorus (Dusk & Night)
    const crikOscA = this.ctx.createOscillator();
    crikOscA.type = 'sine';
    crikOscA.frequency.value = 4140;
    const crikOscB = this.ctx.createOscillator();
    crikOscB.type = 'sine';
    crikOscB.frequency.value = 4186;
    const crikPulse = this.ctx.createGain();
    crikPulse.gain.value = 0.5;
    const crikLfo = this.ctx.createOscillator();
    crikLfo.type = 'sine';
    crikLfo.frequency.value = 13.5;
    const crikLfoDepth = this.ctx.createGain();
    crikLfoDepth.gain.value = 0.48;
    crikLfo.connect(crikLfoDepth);
    crikLfoDepth.connect(crikPulse.gain);
    this.cricketGain = this.ctx.createGain();
    this.cricketGain.gain.value = 0.0;
    crikOscA.connect(crikPulse);
    crikOscB.connect(crikPulse);
    crikPulse.connect(this.cricketGain);
    this.cricketGain.connect(this.weatherOccFilter);
    crikOscA.start();
    crikOscB.start();
    crikLfo.start();

    // Bus 7: DJI Avata 2 5-Blade Ducted Cinewhoop Asymmetric Motor Bank + Prop-Wash + Proximity Aero-Whoosh
    this.avataMotorOscA = this.ctx.createOscillator();
    this.avataMotorOscA.type = 'sawtooth';
    this.avataMotorOscA.frequency.value = 235;

    this.avataMotorOscB = this.ctx.createOscillator();
    this.avataMotorOscB.type = 'triangle';
    this.avataMotorOscB.frequency.value = 472;

    this.avataMotorOscC = this.ctx.createOscillator();
    this.avataMotorOscC.type = 'sawtooth';
    this.avataMotorOscC.frequency.value = 354;

    const avataOscCGain = this.ctx.createGain();
    avataOscCGain.gain.value = 0.42;

    this.avataMotorFilter = this.ctx.createBiquadFilter();
    this.avataMotorFilter.type = 'bandpass';
    this.avataMotorFilter.frequency.value = 540;
    this.avataMotorFilter.Q.value = 1.55;

    this.avataMotorGain = this.ctx.createGain();
    this.avataMotorGain.gain.value = 0.0;

    this.avataMotorOscA.connect(this.avataMotorFilter);
    this.avataMotorOscB.connect(this.avataMotorFilter);
    this.avataMotorOscC.connect(avataOscCGain);
    avataOscCGain.connect(this.avataMotorFilter);
    this.avataMotorFilter.connect(this.avataMotorGain);
    this.avataMotorGain.connect(this.worldSumGain);
    this.avataMotorGain.connect(this.reverbSendGain);
    this.avataMotorOscA.start();
    this.avataMotorOscB.start();
    this.avataMotorOscC.start();

    const washSrc = this.ctx.createBufferSource();
    washSrc.buffer = pinkBuffer;
    washSrc.loop = true;
    this.avataWashFilter = this.ctx.createBiquadFilter();
    this.avataWashFilter.type = 'bandpass';
    this.avataWashFilter.frequency.value = 720;
    this.avataWashFilter.Q.value = 0.85;
    this.avataWashGain = this.ctx.createGain();
    this.avataWashGain.gain.value = 0.0;
    washSrc.connect(this.avataWashFilter);
    this.avataWashFilter.connect(this.avataWashGain);
    this.avataWashGain.connect(this.worldSumGain);
    washSrc.start();

    // Proximity Obstacle Pass-By Aerodynamic Compression Whoosh
    const proxSrc = this.ctx.createBufferSource();
    proxSrc.buffer = pinkBuffer;
    proxSrc.loop = true;
    proxSrc.playbackRate.value = 1.35;
    this.proxWhooshFilter = this.ctx.createBiquadFilter();
    this.proxWhooshFilter.type = 'bandpass';
    this.proxWhooshFilter.frequency.value = 920;
    this.proxWhooshFilter.Q.value = 1.9;
    this.proxWhooshGain = this.ctx.createGain();
    this.proxWhooshGain.gain.value = 0.0;
    proxSrc.connect(this.proxWhooshFilter);
    this.proxWhooshFilter.connect(this.proxWhooshGain);
    this.proxWhooshGain.connect(this.worldSumGain);
    this.proxWhooshGain.connect(this.reverbSendGain);
    proxSrc.start();

    // Bus 8: DJI Mavic 3 Pro Low-Noise Folding-Propeller Aerial Hum
    this.mavicOscA = this.ctx.createOscillator();
    this.mavicOscA.type = 'triangle';
    this.mavicOscA.frequency.value = 144;
    this.mavicOscB = this.ctx.createOscillator();
    this.mavicOscB.type = 'sine';
    this.mavicOscB.frequency.value = 289;
    this.mavicFilter = this.ctx.createBiquadFilter();
    this.mavicFilter.type = 'lowpass';
    this.mavicFilter.frequency.value = 420;
    this.mavicGain = this.ctx.createGain();
    this.mavicGain.gain.value = 0.0;
    this.mavicOscA.connect(this.mavicFilter);
    this.mavicOscB.connect(this.mavicFilter);
    this.mavicFilter.connect(this.mavicGain);
    this.mavicGain.connect(this.worldSumGain);
    this.mavicOscA.start();
    this.mavicOscB.start();

    // Bus 9: Feline Purr Resonator (Active in FPS_CAT or close cat contemplation)
    this.catPurrOsc = this.ctx.createOscillator();
    this.catPurrOsc.type = 'triangle';
    this.catPurrOsc.frequency.value = 26.8;
    const purrHarm = this.ctx.createOscillator();
    purrHarm.type = 'sine';
    purrHarm.frequency.value = 53.6;
    const purrBreathLfo = this.ctx.createOscillator();
    purrBreathLfo.type = 'sine';
    purrBreathLfo.frequency.value = 0.42;
    const purrMod = this.ctx.createGain();
    purrMod.gain.value = 0.55;
    const purrLfoDepth = this.ctx.createGain();
    purrLfoDepth.gain.value = 0.38;
    purrBreathLfo.connect(purrLfoDepth);
    purrLfoDepth.connect(purrMod.gain);
    const purrLp = this.ctx.createBiquadFilter();
    purrLp.type = 'lowpass';
    purrLp.frequency.value = 165;
    this.catPurrGain = this.ctx.createGain();
    this.catPurrGain.gain.value = 0.0;
    this.catPurrOsc.connect(purrMod);
    purrHarm.connect(purrMod);
    purrMod.connect(purrLp);
    purrLp.connect(this.catPurrGain);
    this.catPurrGain.connect(this.worldSumGain);
    this.catPurrOsc.start();
    purrHarm.start();
    purrBreathLfo.start();

    // =========================================================================
    // 7 DEDICATED 3D BINAURAL HRTF POINT-SOURCE EMITTERS
    // =========================================================================
    this._emitters = {};

    // E1. Rotunda Dancing Fountain 3D HRTF Emitter (-2.5, 0.9, 106.5)
    const eFountain = this._create3DEmitter({ refDistance: 10.0, rolloffFactor: 1.15, reverbWeight: 0.25 });
    const fountShimmer = this.ctx.createBufferSource();
    fountShimmer.buffer = patterBuffer;
    fountShimmer.loop = true;
    fountShimmer.playbackRate.value = 1.24;
    const fountWash = this.ctx.createBufferSource();
    fountWash.buffer = pinkBuffer;
    fountWash.loop = true;
    const fountBp = this.ctx.createBiquadFilter();
    fountBp.type = 'bandpass';
    fountBp.frequency.value = 1750;
    fountBp.Q.value = 0.78;
    fountShimmer.connect(fountBp);
    fountWash.connect(fountBp);
    fountBp.connect(eFountain.input);
    this._setPannerPosition(eFountain.panner, -2.5, 0.9, 106.5, this.ctx.currentTime, true);
    fountShimmer.start();
    fountWash.start();
    eFountain.shimmerSrc = fountShimmer;
    eFountain.bp = fountBp;
    this.fountainFilter = fountBp;
    this.fountainGain = eFountain.gain;
    this._emitters.fountain = eFountain;

    // E2. Pulau I Niagara Waterfall (-44, 9, 643) & Pulau U Meandering River 3D HRTF Emitter
    const eWaterfall = this._create3DEmitter({ refDistance: 16.0, rolloffFactor: 0.95, reverbWeight: 0.32 });
    const wfSrc = this.ctx.createBufferSource();
    wfSrc.buffer = pinkBuffer;
    wfSrc.loop = true;
    const wfBp = this.ctx.createBiquadFilter();
    wfBp.type = 'bandpass';
    wfBp.frequency.value = 620;
    wfBp.Q.value = 0.52;
    wfSrc.connect(wfBp);
    wfBp.connect(eWaterfall.input);
    this._setPannerPosition(eWaterfall.panner, -44.0, 9.0, 643.0, this.ctx.currentTime, true);
    wfSrc.start();
    eWaterfall.bp = wfBp;
    this.waterfallFilter = wfBp;
    this.waterfallGain = eWaterfall.gain;
    this._emitters.waterfall = eWaterfall;

    // E3. Bus Kuning Lintas Pulau UI (Hybrid/EV Traction Hum + Asphalt Tire Roll + Tunnel Reverb)
    const eBus = this._create3DEmitter({ refDistance: 7.5, rolloffFactor: 1.05, reverbWeight: 0.55 });
    const busOscA = this.ctx.createOscillator();
    busOscA.type = 'sawtooth';
    busOscA.frequency.value = 88;
    const busOscB = this.ctx.createOscillator();
    busOscB.type = 'triangle';
    busOscB.frequency.value = 176;
    const busToneLp = this.ctx.createBiquadFilter();
    busToneLp.type = 'lowpass';
    busToneLp.frequency.value = 310;
    const busTire = this.ctx.createBufferSource();
    busTire.buffer = pinkBuffer;
    busTire.loop = true;
    const busTireBp = this.ctx.createBiquadFilter();
    busTireBp.type = 'bandpass';
    busTireBp.frequency.value = 420;
    busTireBp.Q.value = 0.9;
    const busTireGain = this.ctx.createGain();
    busTireGain.gain.value = 0.55;
    busOscA.connect(busToneLp);
    busOscB.connect(busToneLp);
    busToneLp.connect(eBus.input);
    busTire.connect(busTireBp);
    busTireBp.connect(busTireGain);
    busTireGain.connect(eBus.input);
    busOscA.start();
    busOscB.start();
    busTire.start();
    eBus.oscA = busOscA;
    eBus.oscB = busOscB;
    eBus.tireBp = busTireBp;
    eBus.tireGain = busTireGain;
    this._emitters.bus = eBus;

    // E4. Mobil Golf Listrik UI (PMSM Electric Inverter Whine + Stone Paver Tire Roll)
    const eCart = this._create3DEmitter({ refDistance: 5.5, rolloffFactor: 1.15, reverbWeight: 0.25 });
    const cartOscA = this.ctx.createOscillator();
    cartOscA.type = 'sine';
    cartOscA.frequency.value = 240;
    const cartOscB = this.ctx.createOscillator();
    cartOscB.type = 'triangle';
    cartOscB.frequency.value = 480;
    const cartOscGain = this.ctx.createGain();
    cartOscGain.gain.value = 0.35;
    const cartRoll = this.ctx.createBufferSource();
    cartRoll.buffer = pinkBuffer;
    cartRoll.loop = true;
    const cartRollBp = this.ctx.createBiquadFilter();
    cartRollBp.type = 'bandpass';
    cartRollBp.frequency.value = 560;
    cartRollBp.Q.value = 1.1;
    cartOscA.connect(cartOscGain);
    cartOscB.connect(cartOscGain);
    cartOscGain.connect(eCart.input);
    cartRoll.connect(cartRollBp);
    cartRollBp.connect(eCart.input);
    cartOscA.start();
    cartOscB.start();
    cartRoll.start();
    eCart.oscA = cartOscA;
    eCart.oscB = cartOscB;
    eCart.rollBp = cartRollBp;
    this._emitters.cart = eCart;

    // E5. Taxi Helicopter Listrik eVTOL (PK-UIE Hexa-Rotor BVI Blade Thump + Coaxial Turbine Whine)
    const eHeli = this._create3DEmitter({ refDistance: 13.0, rolloffFactor: 0.92, reverbWeight: 0.28 });
    const heliNoise = this.ctx.createBufferSource();
    heliNoise.buffer = pinkBuffer;
    heliNoise.loop = true;
    const heliBladeLp = this.ctx.createBiquadFilter();
    heliBladeLp.type = 'lowpass';
    heliBladeLp.frequency.value = 280;
    const heliThumpGate = this.ctx.createGain();
    heliThumpGate.gain.value = 0.55;
    const heliBladeLfo = this.ctx.createOscillator();
    heliBladeLfo.type = 'sine';
    heliBladeLfo.frequency.value = 21.0; // Blade-passage frequency!
    const heliLfoDepth = this.ctx.createGain();
    heliLfoDepth.gain.value = 0.44;
    heliBladeLfo.connect(heliLfoDepth);
    heliLfoDepth.connect(heliThumpGate.gain);

    const heliTurbine = this.ctx.createOscillator();
    heliTurbine.type = 'triangle';
    heliTurbine.frequency.value = 520;
    const heliTurbGain = this.ctx.createGain();
    heliTurbGain.gain.value = 0.22;

    heliNoise.connect(heliBladeLp);
    heliBladeLp.connect(heliThumpGate);
    heliThumpGate.connect(eHeli.input);
    heliTurbine.connect(heliTurbGain);
    heliTurbGain.connect(eHeli.input);
    heliNoise.start();
    heliBladeLfo.start();
    heliTurbine.start();
    eHeli.bladeLfo = heliBladeLfo;
    eHeli.turbineOsc = heliTurbine;
    this._emitters.heli = eHeli;

    // E6. Active Fixed-Wing Aviation Emitter (Cessna 172 Piston Prop vs Citation Learjet Twin-Turbofan)
    const eAir = this._create3DEmitter({ refDistance: 22.0, rolloffFactor: 0.82, maxDistance: 2200.0, reverbWeight: 0.30 });
    const airPropOscA = this.ctx.createOscillator();
    airPropOscA.type = 'sawtooth';
    airPropOscA.frequency.value = 104;
    const airPropOscB = this.ctx.createOscillator();
    airPropOscB.type = 'triangle';
    airPropOscB.frequency.value = 208;
    const airPropGain = this.ctx.createGain();
    airPropGain.gain.value = 0.55;

    const airJetWhine = this.ctx.createOscillator();
    airJetWhine.type = 'sine';
    airJetWhine.frequency.value = 1280;
    const airJetWhineGain = this.ctx.createGain();
    airJetWhineGain.gain.value = 0.0;

    const airJetRoar = this.ctx.createBufferSource();
    airJetRoar.buffer = pinkBuffer;
    airJetRoar.loop = true;
    const airRoarBp = this.ctx.createBiquadFilter();
    airRoarBp.type = 'bandpass';
    airRoarBp.frequency.value = 520;
    airRoarBp.Q.value = 0.65;
    const airRoarGain = this.ctx.createGain();
    airRoarGain.gain.value = 0.45;

    airPropOscA.connect(airPropGain);
    airPropOscB.connect(airPropGain);
    airPropGain.connect(eAir.input);
    airJetWhine.connect(airJetWhineGain);
    airJetWhineGain.connect(eAir.input);
    airJetRoar.connect(airRoarBp);
    airRoarBp.connect(airRoarGain);
    airRoarGain.connect(eAir.input);
    airPropOscA.start();
    airPropOscB.start();
    airJetWhine.start();
    airJetRoar.start();
    eAir.propOscA = airPropOscA;
    eAir.propOscB = airPropOscB;
    eAir.propGain = airPropGain;
    eAir.jetWhine = airJetWhine;
    eAir.jetWhineGain = airJetWhineGain;
    eAir.roarBp = airRoarBp;
    eAir.roarGain = airRoarGain;
    this._emitters.air = eAir;

    // E7. Marine Fleet Emitter (Kapal Feri Katamaran Selat UI & Flybridge Mini Yacht Teluk U)
    const eMarine = this._create3DEmitter({ refDistance: 12.0, rolloffFactor: 1.0, reverbWeight: 0.22 });
    const marOsc = this.ctx.createOscillator();
    marOsc.type = 'triangle';
    marOsc.frequency.value = 72;
    const marOscGain = this.ctx.createGain();
    marOscGain.gain.value = 0.35;
    const marWake = this.ctx.createBufferSource();
    marWake.buffer = pinkBuffer;
    marWake.loop = true;
    const marWakeBp = this.ctx.createBiquadFilter();
    marWakeBp.type = 'bandpass';
    marWakeBp.frequency.value = 540;
    marWakeBp.Q.value = 0.8;
    marOsc.connect(marOscGain);
    marOscGain.connect(eMarine.input);
    marWake.connect(marWakeBp);
    marWakeBp.connect(eMarine.input);
    marOsc.start();
    marWake.start();
    eMarine.osc = marOsc;
    eMarine.wakeBp = marWakeBp;
    this._emitters.marine = eMarine;
  }

  setWeatherAudio(weatherMode) {
    if (!this.ctx || !this.rainRoarGain) return;
    const now = this.ctx.currentTime;

    let roar = 0.0;
    let patter = 0.0;
    let windVol = 0.44;
    let windFreq = 390;

    if (weatherMode === 'STORM') {
      roar = 1.22;
      patter = 0.90;
      windVol = 1.12;
      windFreq = 640;
    } else if (weatherMode === 'RAIN') {
      roar = 0.90;
      patter = 0.70;
      windVol = 0.70;
      windFreq = 490;
    } else if (weatherMode === 'SHOWER') {
      roar = 0.48;
      patter = 0.50;
      windVol = 0.56;
      windFreq = 445;
    } else if (weatherMode === 'MIST') {
      roar = 0.20;
      patter = 0.28;
      windVol = 0.48;
      windFreq = 410;
    } else if (weatherMode === 'WINDY') {
      roar = 0.0;
      patter = 0.0;
      windVol = 0.92;
      windFreq = 585;
    }

    this.rainRoarGain.gain.setTargetAtTime(roar, now, 0.45);
    this.rainPatterGain.gain.setTargetAtTime(patter * (this._zoneState?.patterBoost ?? 1.0), now, 0.45);
    this.breezeGain.gain.setTargetAtTime(windVol, now, 0.55);
    if (this.breezeFilter) {
      this.breezeFilter.frequency.setTargetAtTime(windFreq, now, 0.55);
    }
  }

  /**
   * Couples Web Audio wind, downpour roar, and eave-drip patter continuously 1:1
   * to the live 20–45s meteorological transition state + listener altitude & roof shelter!
   */
  syncContinuousWeatherAudio(rainIntensity = 0.0, stormFactor = 0.0, windSpeed = 0.85) {
    this._weatherState.rain = clamp(Number(rainIntensity) || 0.0, 0.0, 1.5);
    this._weatherState.storm = clamp(Number(stormFactor) || 0.0, 0.0, 1.0);
    this._weatherState.windSpeed = Math.max(0.0, Number(windSpeed) || 0.85);

    if (!this.ctx || !this.rainRoarGain) return;
    const now = this.ctx.currentTime;
    const rain = this._weatherState.rain;
    const storm = this._weatherState.storm;
    const wNorm = clamp((this._weatherState.windSpeed - 0.7) / 3.0, 0.0, 1.6);

    // High-altitude aerodynamic wind boost & ground-patter height attenuation
    const altY = Math.max(0.0, this._lis.y);
    const altWindBoost = clamp(altY / 85.0, 0.0, 0.45);
    const altPatterAtten = 1.0 / (1.0 + Math.max(0.0, altY - 12.0) * 0.045);
    const patBoost = this._zoneState ? this._zoneState.patterBoost : 1.0;

    const roar = Math.min(1.26, Math.pow(rain / 1.42, 1.15) * 1.24);
    const patter =
      Math.min(0.96, Math.pow(Math.min(1.0, rain * 1.05), 0.85) * (0.70 + 0.22 * storm)) *
      altPatterAtten *
      patBoost;
    const windVol = 0.40 + wNorm * 0.52 + storm * 0.22 + altWindBoost * 0.28;
    const windFreq = 375 + wNorm * 170 + storm * 95 + altWindBoost * 160;

    this.rainRoarGain.gain.setTargetAtTime(roar, now, 0.16);
    this.rainPatterGain.gain.setTargetAtTime(patter, now, 0.16);
    this.breezeGain.gain.setTargetAtTime(windVol, now, 0.20);
    if (this.breezeFilter) {
      this.breezeFilter.frequency.setTargetAtTime(windFreq, now, 0.20);
    }
  }

  /**
   * Modulates the DJI Avata 2 5-blade ducted cinewhoop asymmetric 4-motor harmonic bank,
   * cornering prop-wash, G-load bite, and obstacle proximity aerodynamic whoosh.
   */
  syncAvataDroneAudio(
    active = false,
    throttle01 = 0.0,
    speedMs = 0.0,
    turnRate = 0.0,
    gForce = 1.0,
    obstacleClearanceM = 99.0
  ) {
    if (!this.ctx || !this.avataMotorGain || !this.avataWashGain) return;
    const now = this.ctx.currentTime;
    if (!active || !this.enabled) {
      this.avataMotorGain.gain.setTargetAtTime(0.0, now, 0.10);
      this.avataWashGain.gain.setTargetAtTime(0.0, now, 0.10);
      if (this.proxWhooshGain) this.proxWhooshGain.gain.setTargetAtTime(0.0, now, 0.10);
      return;
    }

    const thr = clamp(throttle01, 0.0, 1.0);
    const spd = clamp(speedMs, 0.0, 32.0);
    const turn = clamp(turnRate, 0.0, 5.0);
    const gLoad = clamp((Number(gForce) || 1.0) - 1.0, 0.0, 2.5);

    // Asymmetric motor differential during yaw/roll maneuvers creates signature FPV cinewhoop chord!
    const baseHz = 212 + thr * 278 + (spd / 27.0) * 62 + gLoad * 28;
    const asymDetune = 1.0 + turn * 0.032 + gLoad * 0.018;
    this.avataMotorOscA.frequency.setTargetAtTime(baseHz, now, 0.05);
    this.avataMotorOscB.frequency.setTargetAtTime(baseHz * 2.008 * asymDetune, now, 0.05);
    if (this.avataMotorOscC) {
      this.avataMotorOscC.frequency.setTargetAtTime(baseHz * 1.498 * (2.0 - asymDetune * 0.98), now, 0.05);
    }
    this.avataMotorFilter.frequency.setTargetAtTime(450 + thr * 680 + gLoad * 180, now, 0.07);
    this.avataMotorGain.gain.setTargetAtTime(0.050 + thr * 0.105 + gLoad * 0.022, now, 0.07);

    const washGain = 0.036 + (spd / 27.0) * 0.175 + turn * 0.042;
    this.avataWashFilter.frequency.setTargetAtTime(600 + (spd / 27.0) * 740 + turn * 145, now, 0.07);
    this.avataWashGain.gain.setTargetAtTime(Math.min(0.30, washGain), now, 0.07);

    // Aerodynamic air-compression whoosh when skimming fast past bridge trusses, trees, or canyon walls!
    if (this.proxWhooshGain && this.proxWhooshFilter) {
      const proxFactor =
        spd > 5.0 && obstacleClearanceM < 6.8
          ? smooth01(6.8, 1.4, obstacleClearanceM) * clamp(spd / 20.0, 0.25, 1.15)
          : 0.0;
      this.proxWhooshFilter.frequency.setTargetAtTime(520 + proxFactor * 1150, now, 0.05);
      this.proxWhooshGain.gain.setTargetAtTime(Math.min(0.28, proxFactor * 0.24), now, 0.05);
    }
  }

  /**
   * 3D Binaural HRTF Rotunda Dancing Fountain, Niagara Waterfall, Pulau U River, & Shoreline Surf
   */
  syncSpatialWaterAudio(camX = 0.0, camY = 20.0, camZ = 100.0, nightShowBlend = 0.0, coastDist = -40.0, riverZ = null) {
    if (!this.ctx || !this._emitters) return;
    const now = this.ctx.currentTime;
    if (!this.enabled) {
      this.fountainGain.gain.setTargetAtTime(0.0, now, 0.14);
      this.waterfallGain.gain.setTargetAtTime(0.0, now, 0.14);
      if (this.surfGain) this.surfGain.gain.setTargetAtTime(0.0, now, 0.14);
      return;
    }

    // 1. 3D HRTF Rotunda Dancing Fountain at (-2.5, 0.9, 106.5)
    const eFount = this._emitters.fountain;
    const fountMet = AudioSynthesizer.computeDopplerAndAbsorption(
      -2.5, 0.9, 106.5, 0, 0, 0,
      camX, camY, camZ, this._lis.vx, this._lis.vy, this._lis.vz
    );
    const showBoost = 1.0 + clamp(nightShowBlend, 0.0, 1.0) * 0.38;
    const fountProx = clamp(1.0 - fountMet.dist / 92.0, 0.0, 1.0);
    eFount.airLP.frequency.setTargetAtTime(fountMet.airCutoffHz, now, 0.12);
    eFount.bp.frequency.setTargetAtTime((1650 + 380 * nightShowBlend) * clamp(fountMet.doppler, 0.88, 1.14), now, 0.12);
    eFount.gain.gain.setTargetAtTime(fountProx * fountProx * 0.56 * showBoost, now, 0.14);

    // 2. 3D HRTF Pulau I Niagara Waterfall (-44, 9, 643) OR Nearest Pulau U Meandering River Riffle
    const eWf = this._emitters.waterfall;
    const dNiagara = Math.hypot(camX - -44.0, camY - 9.0, camZ - 643.0);
    let wfX = -44.0;
    let wfY = 9.0;
    let wfZ = 643.0;
    let wfLevel = 0.0;
    let wfCenterHz = 620.0;

    const niagProx = clamp(1.0 - dNiagara / 165.0, 0.0, 1.0);
    let riverProx = 0.0;
    if (riverZ !== null && Number.isFinite(riverZ) && camX >= -295.0 && camX <= -98.0 && camZ < 420.0) {
      const dRiv = Math.hypot(camY - 1.2, camZ - riverZ);
      riverProx = clamp(1.0 - dRiv / 54.0, 0.0, 1.0);
    }

    if (riverProx * 0.55 > niagProx) {
      wfX = clamp(camX, -288.0, -104.0);
      wfY = 0.8;
      wfZ = riverZ;
      wfLevel = riverProx * riverProx * 0.42;
      wfCenterHz = 980.0 + riverProx * 420.0;
    } else {
      wfLevel = niagProx * niagProx * 0.78;
      wfCenterHz = 460.0 + niagProx * 440.0;
    }

    const wfMet = AudioSynthesizer.computeDopplerAndAbsorption(
      wfX, wfY, wfZ, 0, 0, 0,
      camX, camY, camZ, this._lis.vx, this._lis.vy, this._lis.vz
    );
    this._setPannerPosition(eWf.panner, wfX, wfY, wfZ, now, false);
    eWf.airLP.frequency.setTargetAtTime(wfMet.airCutoffHz, now, 0.14);
    eWf.bp.frequency.setTargetAtTime(wfCenterHz * clamp(wfMet.doppler, 0.88, 1.14), now, 0.14);
    eWf.gain.gain.setTargetAtTime(wfLevel, now, 0.16);

    // 3. Rhythmic Coastal Wave Surf around Pulau U, Pulau I, Teluk U & Selat UI
    if (this.surfGain && this.surfFilter) {
      const shoreNear = Math.abs(coastDist) < 48.0 ? 1.0 - Math.abs(coastDist) / 48.0 : 0.0;
      const openWaterLow = coastDist > 0.0 && camY > -1.8 && camY < 22.0 ? (22.0 - camY) / 22.0 : 0.0;
      const altFade = camY > -1.8 ? clamp(1.0 - Math.max(0.0, camY - 6.0) / 36.0, 0.0, 1.0) : 0.0;
      const waveSurge = 0.68 + 0.32 * Math.sin(this._simClock * 0.88) * Math.cos(this._simClock * 0.37 + 0.8);
      const surfVol = Math.max(shoreNear * 0.24, openWaterLow * 0.20) * altFade * waveSurge;
      this.surfFilter.frequency.setTargetAtTime(360 + waveSurge * 240, now, 0.18);
      this.surfGain.gain.setTargetAtTime(surfVol, now, 0.18);
    }
  }

  _updateTrackVelocity(track, wx, wy, wz, fallbackYaw, fallbackSpd, dt) {
    if (!track.init) {
      track.x = wx;
      track.y = wy;
      track.z = wz;
      track.vx = Math.sin(fallbackYaw) * fallbackSpd;
      track.vy = 0.0;
      track.vz = Math.cos(fallbackYaw) * fallbackSpd;
      track.prevSpd = fallbackSpd;
      track.init = true;
      return true;
    }
    const dx = wx - track.x;
    const dy = wy - track.y;
    const dz = wz - track.z;
    const jump = Math.hypot(dx, dy, dz);
    if (jump > 36.0 || dt <= 1e-5) {
      track.x = wx;
      track.y = wy;
      track.z = wz;
      track.vx = Math.sin(fallbackYaw) * fallbackSpd;
      track.vy = 0.0;
      track.vz = Math.cos(fallbackYaw) * fallbackSpd;
      track.prevSpd = fallbackSpd;
      return true;
    }
    const alpha = 1.0 - Math.exp(-dt * 12.0);
    track.vx += (dx / dt - track.vx) * alpha;
    track.vy += (dy / dt - track.vy) * alpha;
    track.vz += (dz / dt - track.vz) * alpha;
    track.x = wx;
    track.y = wy;
    track.z = wz;
    return false;
  }

  /**
   * Master Real-Time 3D Physical Acoustic World Orchestrator (60–120 Hz, Zero GC).
   * Updates 3D Binaural Listener orientation, Architectural Zone Reverb/Occlusion,
   * all 7 HRTF 3D Dynamic Vehicle/Aircraft/Marine/Water Emitters with True Doppler,
   * Mavic/Avata Propulsion, Surface-Aware Footsteps, and Diurnal Tropical Bio-Acoustics.
   */
  updateAcousticWorld(dt, state = {}) {
    const safeDt = clamp(Number(dt) || 0.016, 0.001, 0.1);
    this._simClock += safeDt;

    const camera = state.camera || null;
    const camX = camera ? camera.position.x : Number(state.camX) || 0.0;
    const camY = camera ? camera.position.y : Number(state.camY) || 20.0;
    const camZ = camera ? camera.position.z : Number(state.camZ) || 100.0;

    // 1. Update Listener 3D Position, Velocity, & Orthonormal Orientation (Zero Vector Allocations!)
    let teleported = false;
    if (!this._lis.init) {
      this._lis.x = camX;
      this._lis.y = camY;
      this._lis.z = camZ;
      this._lis.init = true;
      teleported = true;
    } else {
      const dx = camX - this._lis.x;
      const dy = camY - this._lis.y;
      const dz = camZ - this._lis.z;
      const stepDist = Math.hypot(dx, dy, dz);
      if (stepDist > 28.0) {
        this._lis.vx = 0.0;
        this._lis.vy = 0.0;
        this._lis.vz = 0.0;
        teleported = true;
      } else {
        const vAlpha = 1.0 - Math.exp(-safeDt * 14.0);
        this._lis.vx += (dx / safeDt - this._lis.vx) * vAlpha;
        this._lis.vy += (dy / safeDt - this._lis.vy) * vAlpha;
        this._lis.vz += (dz / safeDt - this._lis.vz) * vAlpha;
      }
      this._lis.x = camX;
      this._lis.y = camY;
      this._lis.z = camZ;
    }
    this._lis.speedMs = Math.hypot(this._lis.vx, this._lis.vy, this._lis.vz);
    this._telemetry.listenerSpeedMs = Number(this._lis.speedMs.toFixed(1));

    if (camera && camera.matrixWorld && camera.matrixWorld.elements) {
      const e = camera.matrixWorld.elements;
      let fx = -e[8];
      let fy = -e[9];
      let fz = -e[10];
      const fLen = Math.hypot(fx, fy, fz) || 1.0;
      fx /= fLen;
      fy /= fLen;
      fz /= fLen;

      let ux = e[4];
      let uy = e[5];
      let uz = e[6];
      // Gram-Schmidt orthogonalization against forward vector for glitch-free HRTF convolution
      const dotFU = fx * ux + fy * uy + fz * uz;
      ux -= dotFU * fx;
      uy -= dotFU * fy;
      uz -= dotFU * fz;
      const uLen = Math.hypot(ux, uy, uz);
      if (uLen > 1e-4) {
        ux /= uLen;
        uy /= uLen;
        uz /= uLen;
      } else {
        ux = 0.0;
        uy = 1.0;
        uz = 0.0;
      }
      this._lis.fwdX = fx;
      this._lis.fwdY = fy;
      this._lis.fwdZ = fz;
      this._lis.upX = ux;
      this._lis.upY = uy;
      this._lis.upZ = uz;
    }

    // 2. Classify Architectural Space, Roof Shelter, & Enclosed Reverb
    const camMode = state.camMode || 'DRONE_6DOF';
    const isRoofSheltered = Boolean(state.isRoofSheltered);
    const zone = AudioSynthesizer.classifyAcousticZone(camX, camY, camZ, camMode, isRoofSheltered);
    this._zoneState = zone;
    this._telemetry.zoneId = zone.zoneId;
    this._telemetry.zoneLabel = zone.labelId;
    this._telemetry.zoneLabelId = zone.labelId;
    this._telemetry.reverbSendPct = Math.round(zone.reverbSend * 100);
    this._telemetry.weatherOcclusionPct = Math.round((1.0 - zone.weatherOcclusionGain) * 100);

    // Cache meteorological & astronomical state
    const rainInt = clamp(Number(state.rainIntensity) || 0.0, 0.0, 1.5);
    const stormFac = clamp(Number(state.stormFactor) || 0.0, 0.0, 1.0);
    const windSpd = Math.max(0.0, Number(state.windSpeed) || 0.85);
    const wetness = clamp(Number(state.wetness) || 0.0, 0.0, 1.0);
    const nightFac = clamp(Number(state.nightFactor) || 0.0, 0.0, 1.0);
    this._weatherState.wetness = wetness;
    this._weatherState.nightFactor = nightFac;
    if (state.lightningDir && Number.isFinite(state.lightningDir.x)) {
      this._weatherState.lightningDirX = state.lightningDir.x;
      this._weatherState.lightningDirY = state.lightningDir.y;
      this._weatherState.lightningDirZ = state.lightningDir.z;
    }

    // Resolve surface material for telemetry & footsteps
    const surfMat = this._resolveSurfaceMaterial(camX, camY, camZ, camMode);
    this._telemetry.surfaceMaterial = surfMat;

    // Update 3D Vehicle/Aircraft Kinematics & Compute Dominant Acoustic Source + Live Doppler EVEN before unmuting!
    const sivitasAI = state.sivitasAI || null;
    this._updateDynamicFleetAcoustics(safeDt, sivitasAI);

    if (!this.ctx) return;
    const now = this.ctx.currentTime;

    // Bind Web Audio 3D Listener Pose & Orientation
    const listener = this.ctx.listener;
    if (listener) {
      if (listener.positionX && listener.forwardX && listener.upX) {
        this._setParam(listener.positionX, camX, now, 0.024, teleported);
        this._setParam(listener.positionY, camY, now, 0.024, teleported);
        this._setParam(listener.positionZ, camZ, now, 0.024, teleported);
        this._setParam(listener.forwardX, this._lis.fwdX, now, 0.024, true);
        this._setParam(listener.forwardY, this._lis.fwdY, now, 0.024, true);
        this._setParam(listener.forwardZ, this._lis.fwdZ, now, 0.024, true);
        this._setParam(listener.upX, this._lis.upX, now, 0.024, true);
        this._setParam(listener.upY, this._lis.upY, now, 0.024, true);
        this._setParam(listener.upZ, this._lis.upZ, now, 0.024, true);
      } else if (typeof listener.setPosition === 'function') {
        listener.setPosition(camX, camY, camZ);
        if (typeof listener.setOrientation === 'function') {
          listener.setOrientation(
            this._lis.fwdX, this._lis.fwdY, this._lis.fwdZ,
            this._lis.upX, this._lis.upY, this._lis.upZ
          );
        }
      }
    }

    // Apply Convolution Reverb Send, Architectural Weather Occlusion, & Tunnel Ventilation Hum
    this.reverbSendGain.gain.setTargetAtTime(zone.reverbSend, now, 0.14);
    this.weatherOccGain.gain.setTargetAtTime(zone.weatherOcclusionGain, now, 0.16);
    this.weatherOccFilter.frequency.setTargetAtTime(zone.weatherLowpassHz, now, 0.16);
    if (this.tunnelHumGain) {
      this.tunnelHumGain.gain.setTargetAtTime(this.enabled ? zone.tunnelHum * 0.24 : 0.0, now, 0.18);
    }

    // Sync Weather, Spatial Water,Surf, & Dual-Drone Propulsion
    this.syncContinuousWeatherAudio(rainInt, stormFac, windSpd);
    this.syncSpatialWaterAudio(
      camX,
      camY,
      camZ,
      Number(state.fountainNightBlend) || 0.0,
      Number.isFinite(state.coastDist) ? state.coastDist : -40.0,
      state.riverZ ?? null
    );

    const isAvata = camMode === 'DRONE_AVATA2_FPV';
    const isMavic = camMode === 'DRONE_6DOF' || camMode === 'AUTO_CINEMATIC';
    if (isAvata) {
      this.syncAvataDroneAudio(
        true,
        Number(state.avataThrottle01) || 0.35,
        this._lis.speedMs,
        Number(state.avataTurnRate) || 0.0,
        Number(state.avataGForce) || 1.0,
        Number.isFinite(state.obstacleClearanceM) ? state.obstacleClearanceM : 99.0
      );
    } else {
      this.syncAvataDroneAudio(false, 0, 0, 0, 1.0, 99.0);
    }

    if (this.mavicGain) {
      if (isMavic && this.enabled) {
        const mavSpd = clamp(this._lis.speedMs / 18.0, 0.0, 1.2);
        const mavHz = 136.0 + mavSpd * 48.0;
        this.mavicOscA.frequency.setTargetAtTime(mavHz, now, 0.09);
        this.mavicOscB.frequency.setTargetAtTime(mavHz * 2.006, now, 0.09);
        this.mavicFilter.frequency.setTargetAtTime(340 + mavSpd * 280, now, 0.10);
        this.mavicGain.gain.setTargetAtTime(0.022 + mavSpd * 0.028, now, 0.12);
      } else {
        this.mavicGain.gain.setTargetAtTime(0.0, now, 0.10);
      }
    }

    // Nocturnal Cricket Chorus (Dusk/Night outdoor trees & lawns, ducked in heavy storm or deep underground)
    if (this.cricketGain) {
      const rainDuck = clamp(1.0 - rainInt * 0.85, 0.12, 1.0);
      const altDuck = clamp(1.0 - Math.max(0.0, camY - 8.0) / 34.0, 0.0, 1.0);
      const nightActive = smooth01(0.28, 0.65, nightFac) * rainDuck * altDuck;
      this.cricketGain.gain.setTargetAtTime(this.enabled ? nightActive * 0.016 : 0.0, now, 0.25);
    }

    if (!this.enabled || this.ctx.state !== 'running') return;

    // 3. Surface-Aware Gait Footsteps (WALK_SIVITAS, FPS_JOGGER, FPS_CAT)
    const isGaitMode = camMode === 'WALK_SIVITAS' || camMode === 'FPS_JOGGER' || camMode === 'FPS_CAT';
    if (isGaitMode && !teleported) {
      const horizSpeed = Math.hypot(this._lis.vx, this._lis.vz);
      if (horizSpeed > 0.25 && horizSpeed < 12.0) {
        const strideLen =
          camMode === 'FPS_CAT'
            ? 0.30
            : camMode === 'FPS_JOGGER' || horizSpeed > 2.4
              ? 1.04
              : 0.74;
        this._strideAccumM += horizSpeed * safeDt;
        if (this._strideAccumM >= strideLen) {
          this._strideAccumM %= strideLen;
          this._stepFootSide = -this._stepFootSide;
          this._triggerFootstepGrain(
            surfMat,
            horizSpeed,
            wetness,
            isRoofSheltered || zone.zoneId === 'SUBSEA_TUNNEL',
            camMode === 'FPS_CAT'
          );
        }
      } else {
        this._strideAccumM = Math.min(this._strideAccumM, 0.35);
      }
    } else {
      this._strideAccumM = 0.0;
    }

    // 4. Campus Feline Purr & Proximity 3D HRTF Meow + Diurnal Songbirds & Lakeside Frogs
    this._updateBioAcousticMicroEvents(safeDt, now, camMode, sivitasAI, nightFac, rainInt, zone.zoneId);
  }

  _updateDynamicFleetAcoustics(dt, sivitasAI) {
    if (!sivitasAI) return;
    const now = this.ctx ? this.ctx.currentTime : 0.0;
    const audioReady = Boolean(this.ctx && this._emitters && this.enabled);
    const lx = this._lis.x;
    const ly = this._lis.y;
    const lz = this._lis.z;
    const lvx = this._lis.vx;
    const lvy = this._lis.vy;
    const lvz = this._lis.vz;

    let bestLabel = 'Air Mancur & Alam';
    let bestScore = 0.08;
    let bestDoppler = 1.0;

    // 1. Bus Kuning Lintas Pulau UI
    const b = sivitasAI.interIslandBus;
    if (b) {
      const tr = this._entityTracks.bus;
      const jumped = this._updateTrackVelocity(tr, b.worldX, b.worldY + 1.15, b.worldZ, b.yaw || 0, b.speed || 0, dt);
      const accel = dt > 1e-4 ? ((b.speed || 0) - tr.prevSpd) / dt : 0.0;
      tr.prevSpd = b.speed || 0;
      tr.brakeCooldown = Math.max(0.0, tr.brakeCooldown - dt);

      const met = AudioSynthesizer.computeDopplerAndAbsorption(
        tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz
      );
      const busProx = clamp(1.0 - met.dist / 135.0, 0.0, 1.0);
      const spdNorm = clamp((b.speed || 0) / 6.5, 0.0, 1.25);
      const score = busProx * (0.45 + 0.55 * spdNorm);
      if (score > bestScore) {
        bestScore = score;
        bestLabel = 'Bis Kuning UI';
        bestDoppler = met.doppler;
      }

      if (audioReady) {
        const eb = this._emitters.bus;
        this._setPannerPosition(eb.panner, tr.x, tr.y, tr.z, now, jumped);
        const baseHz = (68.0 + spdNorm * 64.0) * met.doppler;
        eb.oscA.frequency.setTargetAtTime(baseHz, now, 0.05);
        eb.oscB.frequency.setTargetAtTime(baseHz * 2.0, now, 0.05);
        eb.tireBp.frequency.setTargetAtTime((280 + spdNorm * 380) * met.doppler, now, 0.06);
        eb.tireGain.gain.setTargetAtTime(0.15 + spdNorm * 0.65, now, 0.06);
        eb.airLP.frequency.setTargetAtTime(met.airCutoffHz, now, 0.08);
        // Stronger reverberation when the bus itself rolls inside the Subsea Underpass!
        const busInTunnel = b.worldY < -2.2 ? 0.85 : 0.35;
        eb.revSend.gain.setTargetAtTime(busInTunnel, now, 0.12);
        eb.gain.gain.setTargetAtTime(busProx > 0 ? 0.08 + spdNorm * 0.28 : 0.0, now, 0.08);

        if (accel < -2.6 && met.dist < 58.0 && tr.brakeCooldown <= 0.0 && this.ctx.state === 'running') {
          tr.brakeCooldown = 6.5;
          this._triggerAirBrakeHiss(tr.x, tr.y - 0.5, tr.z);
        }
      }
    }

    // 2. Mobil Golf Listrik UI
    const c = sivitasAI.golfCart;
    if (c) {
      const tr = this._entityTracks.cart;
      const jumped = this._updateTrackVelocity(tr, c.worldX, c.worldY + 0.85, c.worldZ, c.yaw || 0, c.speed || 0, dt);
      const met = AudioSynthesizer.computeDopplerAndAbsorption(
        tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz
      );
      const cartProx = clamp(1.0 - met.dist / 78.0, 0.0, 1.0);
      const spdNorm = clamp((c.speed || 0) / 2.7, 0.0, 1.2);
      const score = cartProx * (0.35 + 0.65 * spdNorm);
      if (score > bestScore) {
        bestScore = score;
        bestLabel = 'Mobil Golf Listrik';
        bestDoppler = met.doppler;
      }

      if (audioReady) {
        const ec = this._emitters.cart;
        this._setPannerPosition(ec.panner, tr.x, tr.y, tr.z, now, jumped);
        const invHz = (175.0 + spdNorm * 295.0) * met.doppler;
        ec.oscA.frequency.setTargetAtTime(invHz, now, 0.05);
        ec.oscB.frequency.setTargetAtTime(invHz * 2.005, now, 0.05);
        ec.rollBp.frequency.setTargetAtTime((360 + spdNorm * 320) * met.doppler, now, 0.06);
        ec.airLP.frequency.setTargetAtTime(met.airCutoffHz, now, 0.08);
        ec.gain.gain.setTargetAtTime(cartProx > 0 ? spdNorm * 0.22 + 0.015 : 0.0, now, 0.08);
      }
    }

    // 3. Taxi Helicopter Listrik eVTOL (PK-UIE)
    const h = sivitasAI.heliTaxi;
    if (h) {
      const tr = this._entityTracks.heli;
      const jumped = this._updateTrackVelocity(tr, h.worldX, h.worldY + 1.35, h.worldZ, h.yaw || 0, h.speed || 0, dt);
      const met = AudioSynthesizer.computeDopplerAndAbsorption(
        tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz
      );
      const heliProx = clamp(1.0 - met.dist / 240.0, 0.0, 1.0);
      const omegaNorm = clamp((h.rotorOmega || 20.0) / 62.0, 0.2, 1.15);
      const vtolLoad =
        h.phase && (h.phase.startsWith('TAKEOFF') || h.phase.startsWith('LANDING')) ? 1.25 : 1.0;
      const score = heliProx * omegaNorm * 0.95;
      if (score > bestScore) {
        bestScore = score;
        bestLabel = 'Taxi Heli eVTOL';
        bestDoppler = met.doppler;
      }

      if (audioReady) {
        const eh = this._emitters.heli;
        this._setPannerPosition(eh.panner, tr.x, tr.y, tr.z, now, jumped);
        eh.bladeLfo.frequency.setTargetAtTime(clamp((12.0 + omegaNorm * 13.5) * met.doppler, 8.0, 34.0), now, 0.06);
        eh.turbineOsc.frequency.setTargetAtTime((340.0 + omegaNorm * 390.0) * met.doppler, now, 0.06);
        eh.airLP.frequency.setTargetAtTime(met.airCutoffHz, now, 0.08);
        const hGain = heliProx > 0 ? (0.04 + omegaNorm * 0.34) * vtolLoad : 0.0;
        eh.gain.gain.setTargetAtTime(hGain, now, 0.09);
      }
    }

    // 4. Fixed-Wing Campus Fleet: Cessna 172 Skyhawk (PK-UIA) & Citation Learjet (PK-UIB)
    const cessna = sivitasAI.cessnaAircraft;
    const learjet = sivitasAI.learjetAircraft;
    let activeAc = null;
    let isJet = false;
    if (learjet && learjet.phase !== 'APRON_PARKED') {
      activeAc = learjet;
      isJet = true;
    } else if (cessna && cessna.phase !== 'APRON_PARKED') {
      activeAc = cessna;
      isJet = false;
    } else if (learjet && cessna) {
      const dC = Math.hypot(lx - cessna.worldX, ly - cessna.worldY, lz - cessna.worldZ);
      const dL = Math.hypot(lx - learjet.worldX, ly - learjet.worldY, lz - learjet.worldZ);
      activeAc = dL < dC ? learjet : cessna;
      isJet = activeAc === learjet;
    }

    if (activeAc) {
      const tr = this._entityTracks.air;
      const switched = tr.id !== activeAc.registration;
      tr.id = activeAc.registration;
      if (switched) tr.init = false;
      const jumped = this._updateTrackVelocity(
        tr,
        activeAc.worldX,
        activeAc.worldY + 1.5,
        activeAc.worldZ,
        activeAc.yaw || 0,
        activeAc.speed || 0,
        dt
      );
      const met = AudioSynthesizer.computeDopplerAndAbsorption(
        tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz
      );
      const maxHear = isJet ? 560.0 : 380.0;
      const acProx = clamp(1.0 - met.dist / maxHear, 0.0, 1.0);
      const parked = activeAc.phase === 'APRON_PARKED';
      const propNorm = parked ? 0.0 : clamp((activeAc.propOmega || 0.0) / 76.0, 0.0, 1.15);
      const score = acProx * propNorm * 1.15;
      if (score > bestScore) {
        bestScore = score;
        bestLabel = isJet ? 'Citation Learjet' : 'Cessna 172';
        bestDoppler = met.doppler;
      }

      if (audioReady) {
        const ea = this._emitters.air;
        this._setPannerPosition(ea.panner, tr.x, tr.y, tr.z, now, jumped || switched);
        ea.airLP.frequency.setTargetAtTime(met.airCutoffHz, now, 0.08);
        if (parked || acProx <= 0.0) {
          ea.gain.gain.setTargetAtTime(0.0, now, 0.12);
        } else if (isJet) {
          // Twin-Turbofan Bypass Whine + Jet Core Exhaust Roar with 343 m/s Doppler!
          ea.propGain.gain.setTargetAtTime(0.0, now, 0.08);
          ea.jetWhine.frequency.setTargetAtTime((780 + propNorm * 1120) * met.doppler, now, 0.05);
          ea.jetWhineGain.gain.setTargetAtTime(0.18 + propNorm * 0.24, now, 0.08);
          ea.roarBp.frequency.setTargetAtTime((310 + propNorm * 520) * met.doppler, now, 0.06);
          ea.roarGain.gain.setTargetAtTime(0.45 + propNorm * 0.65, now, 0.08);
          ea.gain.gain.setTargetAtTime(0.12 + propNorm * 0.56, now, 0.08);
        } else {
          // Lycoming IO-360 4-Cylinder Piston & 2-Blade Prop Harmonic with 343 m/s Doppler!
          const pHz = (68.0 + propNorm * 74.0) * met.doppler;
          ea.propOscA.frequency.setTargetAtTime(pHz, now, 0.05);
          ea.propOscB.frequency.setTargetAtTime(pHz * 2.0, now, 0.05);
          ea.propGain.gain.setTargetAtTime(0.42 + propNorm * 0.35, now, 0.08);
          ea.jetWhineGain.gain.setTargetAtTime(0.0, now, 0.08);
          ea.roarBp.frequency.setTargetAtTime((240 + propNorm * 280) * met.doppler, now, 0.06);
          ea.roarGain.gain.setTargetAtTime(0.18 + propNorm * 0.26, now, 0.08);
          ea.gain.gain.setTargetAtTime(0.08 + propNorm * 0.42, now, 0.08);
        }
      }
    }

    // 5. Marine Fleet: Northern Selat UI Shuttle Ferry vs Teluk U Flybridge Mini Yacht
    const ferry = sivitasAI.shuttleFerry;
    const yacht = sivitasAI.miniYacht;
    let vessel = null;
    let vesselId = '';
    if (ferry && yacht) {
      const dF = Math.hypot(lx - ferry.worldX, ly - ferry.worldY, lz - ferry.worldZ);
      const dY = Math.hypot(lx - yacht.worldX, ly - yacht.worldY, lz - yacht.worldZ);
      if (dY < dF) {
        vessel = yacht;
        vesselId = 'Mini Yacht Teluk U';
      } else {
        vessel = ferry;
        vesselId = 'Feri Katamaran UI';
      }
    } else {
      vessel = ferry || yacht;
      vesselId = ferry ? 'Feri Katamaran UI' : 'Mini Yacht Teluk U';
    }

    if (vessel) {
      const tr = this._entityTracks.marine;
      const switched = tr.id !== vesselId;
      tr.id = vesselId;
      if (switched) tr.init = false;
      const jumped = this._updateTrackVelocity(
        tr,
        vessel.worldX,
        vessel.worldY + 1.0,
        vessel.worldZ,
        vessel.yaw || 0,
        vessel.speed || 0,
        dt
      );
      const met = AudioSynthesizer.computeDopplerAndAbsorption(
        tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz
      );
      const mProx = clamp(1.0 - met.dist / 135.0, 0.0, 1.0);
      const mSpd = clamp((vessel.speed || 0) / 3.4, 0.0, 1.15);
      const score = mProx * (0.28 + 0.55 * mSpd);
      if (score > bestScore) {
        bestScore = score;
        bestLabel = vesselId;
        bestDoppler = met.doppler;
      }

      if (audioReady) {
        const em = this._emitters.marine;
        this._setPannerPosition(em.panner, tr.x, tr.y, tr.z, now, jumped || switched);
        em.osc.frequency.setTargetAtTime((54.0 + mSpd * 38.0) * met.doppler, now, 0.07);
        em.wakeBp.frequency.setTargetAtTime((380.0 + mSpd * 340.0) * met.doppler, now, 0.08);
        em.airLP.frequency.setTargetAtTime(met.airCutoffHz, now, 0.09);
        em.gain.gain.setTargetAtTime(mProx > 0 ? 0.03 + mSpd * 0.24 : 0.0, now, 0.10);
      }
    }

    this._telemetry.dominantSourceLabel = bestLabel;
    this._telemetry.dominantDoppler = Number(bestDoppler.toFixed(3));
  }

  _resolveSurfaceMaterial(x, y, z, camMode) {
    if (camMode === 'DRONE_6DOF' || camMode === 'DRONE_AVATA2_FPV' || camMode === 'AUTO_CINEMATIC') {
      return 'AIR';
    }
    if (Math.abs(x - -356.0) <= 5.4 && z >= 272.0 && z <= 670.0 && y < 0.7) {
      return 'TUNNEL_CONCRETE';
    }
    if (Math.abs(x - -28.0) <= 2.85 && z >= 150.0 && z <= 342.0 && y >= 0.4) {
      return 'BRIDGE_STEEL_WOOD';
    }
    if (
      (Math.abs(x - -48.0) <= 14.5 && z >= 452.0 && z <= 502.0) ||
      (Math.abs(x - -98.0) <= 14.5 && z >= 572.0 && z <= 621.0)
    ) {
      return 'BRIDGE_STEEL_WOOD';
    }
    // Dedicated Teluk U EPDM Rubberized Jogging Track Oval at (-62.0, 136.5)
    const jx = (x - -62.0) / 16.5;
    const jz = (z - 136.5) / 7.8;
    const jr = Math.hypot(jx, jz);
    if (jr >= 0.72 && jr <= 1.18) {
      return 'EPDM_TRACK';
    }
    // Rektorat Courtyard, Rotunda Plaza, & DPPD Stone Terraces
    if (
      (x >= -38.0 && x <= 38.0 && z >= -36.0 && z <= 73.0) ||
      Math.hypot(x - -2.5, z - 106.5) <= 38.0 ||
      Math.hypot(x - -255.0, z - 329.0) <= 39.5
    ) {
      return 'STONE_PAVER';
    }
    // Pulau I Runway / Apron or Campus Roads
    if (
      (z >= 658.0 && z <= 722.0 && x >= -384.0 && x <= -126.0) ||
      Math.hypot(x - -104.0, z - 106.5) <= 16.5
    ) {
      return 'ASPHALT';
    }
    return 'GRASS_SOIL';
  }

  _triggerFootstepGrain(surfaceMat, speedMs, wetness, isDrySheltered, isCatPaw = false) {
    if (!this.ctx || !this._pinkBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const velScale = clamp(speedMs / 2.2, 0.45, 1.35) * (isCatPaw ? 0.32 : 1.0);
    const pitchJitter = 0.94 + Math.random() * 0.12 + (this._stepFootSide > 0 ? 0.03 : -0.03);

    const src = this.ctx.createBufferSource();
    src.buffer = this._pinkBuffer;
    src.playbackRate.value = 1.35 * pitchJitter;

    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';

    let centerHz = 820;
    let qVal = 1.4;
    let dur = 0.075;
    let peak = 0.12 * velScale;
    let thudHz = 0;

    if (surfaceMat === 'BRIDGE_STEEL_WOOD') {
      centerHz = 640 * pitchJitter;
      qVal = 2.2;
      dur = 0.095;
      peak = 0.15 * velScale;
      thudHz = 142 * pitchJitter;
    } else if (surfaceMat === 'TUNNEL_CONCRETE') {
      centerHz = 1350 * pitchJitter;
      qVal = 1.8;
      dur = 0.068;
      peak = 0.14 * velScale;
    } else if (surfaceMat === 'EPDM_TRACK') {
      centerHz = 480 * pitchJitter;
      qVal = 1.1;
      dur = 0.082;
      peak = 0.10 * velScale;
      thudHz = 108 * pitchJitter;
    } else if (surfaceMat === 'STONE_PAVER') {
      centerHz = 1220 * pitchJitter;
      qVal = 1.6;
      dur = 0.065;
      peak = 0.12 * velScale;
    } else if (surfaceMat === 'ASPHALT') {
      centerHz = 980 * pitchJitter;
      qVal = 1.2;
      dur = 0.072;
      peak = 0.115 * velScale;
    } else {
      // GRASS_SOIL leaf/turf rustle
      centerHz = 1650 * pitchJitter;
      qVal = 0.75;
      dur = 0.095;
      peak = 0.085 * velScale;
    }

    bp.frequency.setValueAtTime(centerHz, t0);
    bp.Q.setValueAtTime(qVal, t0);

    const env = this.ctx.createGain();
    env.gain.setValueAtTime(0.0008, t0);
    env.gain.linearRampToValueAtTime(peak, t0 + 0.008);
    env.gain.exponentialRampToValueAtTime(0.0008, t0 + dur);

    src.connect(bp);
    bp.connect(env);
    env.connect(this.worldSumGain);
    if (surfaceMat === 'TUNNEL_CONCRETE' || surfaceMat === 'BRIDGE_STEEL_WOOD') {
      env.connect(this.reverbSendGain);
    }
    src.onended = () => {
      try {
        env.disconnect();
      } catch (_e) {}
    };
    src.start(t0);
    src.stop(t0 + dur + 0.01);

    if (thudHz > 0) {
      const osc = this.ctx.createOscillator();
      const og = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(thudHz, t0);
      osc.frequency.exponentialRampToValueAtTime(thudHz * 0.58, t0 + 0.075);
      og.gain.setValueAtTime(0.001, t0);
      og.gain.linearRampToValueAtTime(peak * 0.55, t0 + 0.006);
      og.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.08);
      osc.connect(og);
      og.connect(this.worldSumGain);
      osc.onended = () => {
        try {
          og.disconnect();
        } catch (_e) {}
      };
      osc.start(t0);
      osc.stop(t0 + 0.09);
    }

    // Wet Puddle Micro-Splash Layer when ground is wet & unsheltered!
    if (!isDrySheltered && wetness > 0.26) {
      const spSrc = this.ctx.createBufferSource();
      spSrc.buffer = this._patterBuffer || this._pinkBuffer;
      spSrc.playbackRate.value = 1.6 + Math.random() * 0.35;
      const hp = this.ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency.setValueAtTime(2550, t0);
      const sg = this.ctx.createGain();
      const spPeak = peak * smooth01(0.25, 0.85, wetness) * 0.82;
      sg.gain.setValueAtTime(0.0008, t0);
      sg.gain.linearRampToValueAtTime(spPeak, t0 + 0.012);
      sg.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.095);
      spSrc.connect(hp);
      hp.connect(sg);
      sg.connect(this.worldSumGain);
      spSrc.onended = () => {
        try {
          sg.disconnect();
        } catch (_e) {}
      };
      spSrc.start(t0);
      spSrc.stop(t0 + 0.105);
    }
  }

  _triggerAirBrakeHiss(bx, by, bz) {
    if (!this.ctx || !this._pinkBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = this._pinkBuffer;
    src.playbackRate.value = 1.85;

    const hp = this.ctx.createBiquadFilter();
    hp.type = 'bandpass';
    hp.frequency.setValueAtTime(3400, t0);
    hp.Q.setValueAtTime(1.1, t0);

    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.001, t0);
    g.gain.linearRampToValueAtTime(0.22, t0 + 0.03);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.42);

    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 6.0;
    pan.rolloffFactor = 1.1;
    this._setPannerPosition(pan, bx, by, bz, t0, true);

    src.connect(hp);
    hp.connect(g);
    g.connect(pan);
    pan.connect(this.worldSumGain);
    pan.connect(this.reverbSendGain);
    src.onended = () => {
      try {
        pan.disconnect();
      } catch (_e) {}
    };
    src.start(t0);
    src.stop(t0 + 0.44);
  }

  _updateBioAcousticMicroEvents(dt, now, camMode, sivitasAI, nightFac, rainInt, zoneId) {
    const lx = this._lis.x;
    const ly = this._lis.y;
    const lz = this._lis.z;

    // A. Campus Cat Purr & Soft 3D HRTF Meow
    let nearestCat = null;
    let nearestCatD = 999.0;
    if (sivitasAI && Array.isArray(sivitasAI.cats) && ly < 8.0) {
      for (let i = 0; i < sivitasAI.cats.length; i++) {
        const c = sivitasAI.cats[i];
        const d = Math.hypot(lx - c.worldX, ly - c.worldY, lz - c.worldZ);
        if (d < nearestCatD) {
          nearestCatD = d;
          nearestCat = c;
        }
      }
    }
    if (this.catPurrGain) {
      const purrTarget =
        camMode === 'FPS_CAT'
          ? 0.085
          : nearestCatD < 2.8
            ? (1.0 - nearestCatD / 2.8) * 0.045
            : 0.0;
      this.catPurrGain.gain.setTargetAtTime(purrTarget, now, 0.20);
    }

    this._catMeowTimer -= dt;
    if (this._catMeowTimer <= 0.0) {
      this._catMeowTimer = 8.5 + Math.random() * 11.0;
      if ((camMode === 'FPS_CAT' || (nearestCat && nearestCatD < 6.5)) && rainInt < 0.65) {
        const cx = nearestCat ? nearestCat.worldX : lx;
        const cy = nearestCat ? nearestCat.worldY + 0.25 : ly - 0.1;
        const cz = nearestCat ? nearestCat.worldZ : lz;
        const isKitten = nearestCat?.phenotype?.ageBuildClass === 'KITTEN';
        this._triggerCatMeow3D(cx, cy, cz, isKitten);
      }
    }

    // Suppress outdoor wildlife chirps inside Subsea Tunnel or at high drone altitude
    if (zoneId === 'SUBSEA_TUNNEL' || ly > 36.0) return;

    // B. Daytime Tropical Kampus UI Songbirds (Kutilang / Pipit FM Syrinx Chirps)
    if (nightFac < 0.42 && rainInt < 0.45) {
      this._birdTimer -= dt;
      if (this._birdTimer <= 0.0) {
        this._birdTimer = 2.4 + Math.random() * 4.6;
        const az = Math.random() * Math.PI * 2.0;
        const r = 11.0 + Math.random() * 20.0;
        this._triggerSongbirdChirp3D(lx + Math.cos(az) * r, Math.max(3.5, ly + 2.5 + Math.random() * 6.0), lz + Math.sin(az) * r);
      }
    }

    // C. Dusk / Night Lakeside Frogs near Danau Kenanga, Danau Kecil, or Pulau U River
    if (nightFac > 0.48 && rainInt < 0.95) {
      this._frogTimer -= dt;
      if (this._frogTimer <= 0.0) {
        this._frogTimer = 3.2 + Math.random() * 5.4;
        const nearLake =
          (lx > 18.0 && Math.abs(lz) < 85.0) ||
          Math.hypot(lx - -78.0, lz - -22.0) < 48.0 ||
          (lx < -105.0 && lx > -295.0 && Math.abs(lz - 55.0) < 65.0);
        if (nearLake) {
          const az = Math.random() * Math.PI * 2.0;
          const r = 8.0 + Math.random() * 16.0;
          this._triggerFrogCroak3D(lx + Math.cos(az) * r, 0.2, lz + Math.sin(az) * r);
        }
      }
    }
  }

  _triggerSongbirdChirp3D(bx, by, bz) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const notes = 2 + Math.floor(Math.random() * 3);
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 7.0;
    pan.rolloffFactor = 1.05;
    this._setPannerPosition(pan, bx, by, bz, t0, true);
    pan.connect(this.weatherOccFilter);
    pan.connect(this.reverbSendGain);

    const baseHz = 2250 + Math.random() * 980;
    let cursor = t0;
    for (let n = 0; n < notes; n++) {
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      const fStart = baseHz * (0.92 + 0.18 * ((n + 1) % 2));
      const fEnd = fStart * (n % 2 === 0 ? 1.28 : 0.84);
      const dur = 0.065 + Math.random() * 0.045;
      osc.frequency.setValueAtTime(fStart, cursor);
      osc.frequency.exponentialRampToValueAtTime(fEnd, cursor + dur);
      g.gain.setValueAtTime(0.0005, cursor);
      g.gain.linearRampToValueAtTime(0.055, cursor + 0.014);
      g.gain.exponentialRampToValueAtTime(0.0005, cursor + dur);
      osc.connect(g);
      g.connect(pan);
      osc.start(cursor);
      osc.stop(cursor + dur + 0.01);
      if (n === notes - 1) {
        osc.onended = () => {
          try {
            pan.disconnect();
          } catch (_e) {}
        };
      }
      cursor += dur + 0.035 + Math.random() * 0.03;
    }
  }

  _triggerFrogCroak3D(fx, fy, fz) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const bp = this.ctx.createBiquadFilter();
    const g = this.ctx.createGain();
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 6.5;
    this._setPannerPosition(pan, fx, fy, fz, t0, true);

    osc.type = 'sawtooth';
    const f0 = 195 + Math.random() * 55;
    osc.frequency.setValueAtTime(f0, t0);
    osc.frequency.linearRampToValueAtTime(f0 * 1.18, t0 + 0.09);
    osc.frequency.linearRampToValueAtTime(f0 * 0.92, t0 + 0.19);

    bp.type = 'bandpass';
    bp.frequency.setValueAtTime(680, t0);
    bp.Q.setValueAtTime(3.2, t0);

    g.gain.setValueAtTime(0.0008, t0);
    g.gain.linearRampToValueAtTime(0.075, t0 + 0.025);
    g.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.21);

    osc.connect(bp);
    bp.connect(g);
    g.connect(pan);
    pan.connect(this.weatherOccFilter);
    osc.onended = () => {
      try {
        pan.disconnect();
      } catch (_e) {}
    };
    osc.start(t0);
    osc.stop(t0 + 0.23);
  }

  _triggerCatMeow3D(cx, cy, cz, isKitten = false) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const bp = this.ctx.createBiquadFilter();
    const g = this.ctx.createGain();
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 3.5;
    this._setPannerPosition(pan, cx, cy, cz, t0, true);

    const base = isKitten ? 760 : 510;
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(base * 0.88, t0);
    osc.frequency.exponentialRampToValueAtTime(base * 1.24, t0 + 0.11);
    osc.frequency.exponentialRampToValueAtTime(base * 0.76, t0 + 0.34);

    // Formant sweep ("m-ee-ow")
    bp.type = 'bandpass';
    bp.Q.setValueAtTime(2.4, t0);
    bp.frequency.setValueAtTime(900, t0);
    bp.frequency.exponentialRampToValueAtTime(1850, t0 + 0.12);
    bp.frequency.exponentialRampToValueAtTime(780, t0 + 0.34);

    g.gain.setValueAtTime(0.0008, t0);
    g.gain.linearRampToValueAtTime(0.085, t0 + 0.05);
    g.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.36);

    osc.connect(bp);
    bp.connect(g);
    g.connect(pan);
    pan.connect(this.worldSumGain);
    osc.onended = () => {
      try {
        pan.disconnect();
      } catch (_e) {}
    };
    osc.start(t0);
    osc.stop(t0 + 0.38);
  }

  /**
   * Soft subtle harmonic tactile chime for stunt/zone confirmation when audio is enabled
   */
  playUiChime(freqHz = 580) {
    if (!this.enabled || !this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freqHz, t0);
    osc.frequency.exponentialRampToValueAtTime(freqHz * 1.26, t0 + 0.09);
    g.gain.setValueAtTime(0.001, t0);
    g.gain.linearRampToValueAtTime(0.075, t0 + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.13);
    osc.connect(g);
    g.connect(this.worldSumGain || this.masterGain);
    osc.onended = () => {
      try {
        g.disconnect();
      } catch (_e) {}
    };
    osc.start(t0);
    osc.stop(t0 + 0.14);
  }

  /**
   * Synthesizes 3D Binaural HRTF rolling thunder propagating at the speed of sound (c = 343 m/s)
   * from the exact lightning discharge azimuth (`dirVec` or cached `uLightningDir`), with close-range
   * N-wave plasma crack, atmospheric high-frequency absorption, cloud echoes, & tunnel occlusion!
   */
  triggerThunder(distanceMeters = 650, intensity = 1.0, dirVec = null) {
    if (!this.enabled || !this.ctx || !this._pinkBuffer) return;

    const lx = dirVec && Number.isFinite(dirVec.x) ? dirVec.x : this._weatherState.lightningDirX;
    const lz = dirVec && Number.isFinite(dirVec.z) ? dirVec.z : this._weatherState.lightningDirZ;
    const delaySec = clamp(distanceMeters / SPEED_OF_SOUND_MS, 0.18, 4.8);

    const timerId = setTimeout(() => {
      if (!this.enabled || !this.ctx || this.ctx.state !== 'running') return;
      const t0 = this.ctx.currentTime;
      const occMult = this._zoneState ? clamp(0.28 + 0.72 * this._zoneState.weatherOcclusionGain, 0.28, 1.0) : 1.0;
      const distAtten = clamp(740.0 / Math.max(250.0, distanceMeters), 0.28, 1.20);
      const peakGain = Math.min(1.75, intensity * distAtten * 1.42) * occMult;

      // 3D HRTF Strike Origin in Sky Dome
      const pan = this.ctx.createPanner();
      pan.panningModel = 'HRTF';
      pan.distanceModel = 'inverse';
      pan.refDistance = 180.0;
      pan.maxDistance = 4000.0;
      pan.rolloffFactor = 0.35;
      const strikeX = this._lis.x + lx * distanceMeters;
      const strikeY = this._lis.y + Math.max(85.0, distanceMeters * 0.28);
      const strikeZ = this._lis.z + lz * distanceMeters;
      this._setPannerPosition(pan, strikeX, strikeY, strikeZ, t0, true);
      pan.connect(this.worldSumGain);
      pan.connect(this.reverbSendGain);

      // 0. Close-Strike High-Voltage N-Wave Plasma Crack (< 620 m)
      if (distanceMeters < 620 && occMult > 0.45) {
        const crackSrc = this.ctx.createBufferSource();
        crackSrc.buffer = this._patterBuffer || this._pinkBuffer;
        crackSrc.playbackRate.value = 1.45;
        const crackHp = this.ctx.createBiquadFilter();
        crackHp.type = 'bandpass';
        crackHp.frequency.setValueAtTime(1450, t0);
        crackHp.Q.setValueAtTime(0.9, t0);
        const crackGain = this.ctx.createGain();
        crackGain.gain.setValueAtTime(0.001, t0);
        crackGain.gain.linearRampToValueAtTime(peakGain * 0.72, t0 + 0.012);
        crackGain.gain.exponentialRampToValueAtTime(0.001, t0 + 0.22);
        crackSrc.connect(crackHp);
        crackHp.connect(crackGain);
        crackGain.connect(pan);
        crackSrc.start(t0);
        crackSrc.stop(t0 + 0.24);
      }

      // 1. Filtered Cumulonimbus Acoustic Rumble & Secondary Cloud-Layer Echo Swell
      const noiseSrc = this.ctx.createBufferSource();
      noiseSrc.buffer = this._pinkBuffer;
      noiseSrc.playbackRate.value = 0.54 + Math.random() * 0.24;

      const lp = this.ctx.createBiquadFilter();
      lp.type = 'lowpass';
      const initialCutoff = (distanceMeters < 580 ? 1680 : 660) * occMult;
      lp.frequency.setValueAtTime(Math.max(180, initialCutoff), t0);
      lp.frequency.exponentialRampToValueAtTime(88, t0 + 2.9);

      const rumbleGain = this.ctx.createGain();
      rumbleGain.gain.setValueAtTime(0.001, t0);
      rumbleGain.gain.linearRampToValueAtTime(peakGain, t0 + 0.045);
      rumbleGain.gain.exponentialRampToValueAtTime(peakGain * 0.46, t0 + 0.38);
      rumbleGain.gain.linearRampToValueAtTime(peakGain * 0.64, t0 + 0.74);
      rumbleGain.gain.exponentialRampToValueAtTime(0.0008, t0 + 3.45);

      noiseSrc.connect(lp);
      lp.connect(rumbleGain);
      rumbleGain.connect(pan);
      noiseSrc.onended = () => {
        try {
          pan.disconnect();
        } catch (_e) {}
      };
      noiseSrc.start(t0);
      noiseSrc.stop(t0 + 3.55);

      // 2. Sub-Bass Acoustic Shockwave Thump (34–68 Hz)
      const osc = this.ctx.createOscillator();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(66, t0);
      osc.frequency.exponentialRampToValueAtTime(34, t0 + 1.45);

      const subGain = this.ctx.createGain();
      subGain.gain.setValueAtTime(0.001, t0);
      subGain.gain.linearRampToValueAtTime(peakGain * 0.44, t0 + 0.06);
      subGain.gain.exponentialRampToValueAtTime(0.0008, t0 + 1.55);

      osc.connect(subGain);
      subGain.connect(this.worldSumGain);
      osc.onended = () => {
        try {
          subGain.disconnect();
        } catch (_e) {}
      };
      osc.start(t0);
      osc.stop(t0 + 1.62);
    }, delaySec * 1000);

    this._thunderTimers.push(timerId);
    if (this._thunderTimers.length > 12) {
      this._thunderTimers.shift();
    }
  }
}
