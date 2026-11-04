/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.2 Super-Realistic Physical Weather Edition)
 * ./core/AudioSynthesizer.js
 *
 * Pure Web Audio API multi-bus physical soundscape for Kampus UI Depok:
 * - Bus 1: Danau Kenanga breeze & turbulent storm wind squall bus
 * - Bus 2: Heavy tropical rain downpour acoustic wash bus
 * - Bus 3: Crisp nearby raindrop impact & terracotta eave-drip patter bus
 * - Bus 4: Speed-of-sound delayed rolling thunder synthesizer (delta_t = distance / 343 m/s)
 */
export class AudioSynthesizer {
  constructor() {
    this.ctx = null;
    this.enabled = false;
    this.masterGain = null;
    this.breezeGain = null;
    this.breezeFilter = null;
    this.rainRoarGain = null;
    this.rainPatterGain = null;
    this.avataMotorOscA = null;
    this.avataMotorOscB = null;
    this.avataMotorFilter = null;
    this.avataMotorGain = null;
    this.avataWashFilter = null;
    this.avataWashGain = null;
    this.fountainFilter = null;
    this.fountainGain = null;
    this.waterfallFilter = null;
    this.waterfallGain = null;
    this._pinkBuffer = null;
    this._thunderTimers = [];
  }

  toggle() {
    if (!this.ctx) {
      this._initContext();
    }
    if (!this.ctx) return false;

    this.enabled = !this.enabled;
    if (this.enabled && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
    const target = this.enabled ? 0.24 : 0.0;
    this.masterGain.gain.setTargetAtTime(target, this.ctx.currentTime, 0.18);
    return this.enabled;
  }

  _initContext() {
    if (typeof window === 'undefined') return;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;

    this.ctx = new AudioCtx();
    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = 0.0;
    this.masterGain.connect(this.ctx.destination);

    const sr = this.ctx.sampleRate;
    const bufferSize = sr * 4;

    // 1. Pink-noise buffer for natural wind, lake water, & rain downpour wash
    const pinkBuffer = this.ctx.createBuffer(1, bufferSize, sr);
    const pinkOut = pinkBuffer.getChannelData(0);
    let b0 = 0, b1 = 0, b2 = 0;
    for (let i = 0; i < bufferSize; i++) {
      const white = Math.random() * 2 - 1;
      b0 = 0.99765 * b0 + white * 0.099046;
      b1 = 0.963 * b1 + white * 0.2965164;
      b2 = 0.57 * b2 + white * 1.0526913;
      pinkOut[i] = (b0 + b1 + b2 + white * 0.1848) * 0.075;
    }
    this._pinkBuffer = pinkBuffer;

    // 2. Impulsive micro-droplet patter buffer (Poisson droplet impacts on terracotta tiles & puddles)
    const patterBuffer = this.ctx.createBuffer(1, bufferSize, sr);
    const patterOut = patterBuffer.getChannelData(0);
    let env = 0.0;
    for (let i = 0; i < bufferSize; i++) {
      if (Math.random() < 0.0042) {
        env = Math.min(1.0, env + 0.35 + Math.random() * 0.65);
      }
      env *= 0.965;
      patterOut[i] = (Math.random() * 2 - 1) * env * 0.26;
    }

    // Bus 1: Campus Breeze & Storm Squall Wind Bus
    const breezeSrc = this.ctx.createBufferSource();
    breezeSrc.buffer = pinkBuffer;
    breezeSrc.loop = true;

    this.breezeFilter = this.ctx.createBiquadFilter();
    this.breezeFilter.type = 'lowpass';
    this.breezeFilter.frequency.value = 390;

    this.breezeGain = this.ctx.createGain();
    this.breezeGain.gain.value = 0.5;

    breezeSrc.connect(this.breezeFilter);
    this.breezeFilter.connect(this.breezeGain);
    this.breezeGain.connect(this.masterGain);
    breezeSrc.start();

    // Bus 2: Tropical Rain Downpour Roar Bus
    const roarSrc = this.ctx.createBufferSource();
    roarSrc.buffer = pinkBuffer;
    roarSrc.loop = true;

    const roarFilter = this.ctx.createBiquadFilter();
    roarFilter.type = 'bandpass';
    roarFilter.frequency.value = 980;
    roarFilter.Q.value = 0.62;

    this.rainRoarGain = this.ctx.createGain();
    this.rainRoarGain.gain.value = 0.0;

    roarSrc.connect(roarFilter);
    roarFilter.connect(this.rainRoarGain);
    this.rainRoarGain.connect(this.masterGain);
    roarSrc.start();

    // Bus 3: Crisp Nearby Droplet & Eave-Drip Patter Bus
    const patterSrc = this.ctx.createBufferSource();
    patterSrc.buffer = patterBuffer;
    patterSrc.loop = true;

    const patterFilter = this.ctx.createBiquadFilter();
    patterFilter.type = 'highpass';
    patterFilter.frequency.value = 2150;

    this.rainPatterGain = this.ctx.createGain();
    this.rainPatterGain.gain.value = 0.0;

    patterSrc.connect(patterFilter);
    patterFilter.connect(this.rainPatterGain);
    this.rainPatterGain.connect(this.masterGain);
    patterSrc.start();

    // Bus 5: DJI Avata 2 5-Blade Ducted Cinewhoop Brushless Quad Motor & Prop-Wash Bus
    this.avataMotorOscA = this.ctx.createOscillator();
    this.avataMotorOscA.type = 'sawtooth';
    this.avataMotorOscA.frequency.value = 235;

    this.avataMotorOscB = this.ctx.createOscillator();
    this.avataMotorOscB.type = 'triangle';
    this.avataMotorOscB.frequency.value = 472;

    this.avataMotorFilter = this.ctx.createBiquadFilter();
    this.avataMotorFilter.type = 'bandpass';
    this.avataMotorFilter.frequency.value = 540;
    this.avataMotorFilter.Q.value = 1.65;

    this.avataMotorGain = this.ctx.createGain();
    this.avataMotorGain.gain.value = 0.0;

    this.avataMotorOscA.connect(this.avataMotorFilter);
    this.avataMotorOscB.connect(this.avataMotorFilter);
    this.avataMotorFilter.connect(this.avataMotorGain);
    this.avataMotorGain.connect(this.masterGain);
    this.avataMotorOscA.start();
    this.avataMotorOscB.start();

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
    this.avataWashGain.connect(this.masterGain);
    washSrc.start();

    // Bus 6: Spatial Rotunda Dancing Fountain Shimmer & Pulau I Niagara Waterfall / Shoreline Surf Bus
    const fountainSrc = this.ctx.createBufferSource();
    fountainSrc.buffer = patterBuffer;
    fountainSrc.loop = true;
    fountainSrc.playbackRate.value = 1.24;

    this.fountainFilter = this.ctx.createBiquadFilter();
    this.fountainFilter.type = 'bandpass';
    this.fountainFilter.frequency.value = 1850;
    this.fountainFilter.Q.value = 0.85;

    this.fountainGain = this.ctx.createGain();
    this.fountainGain.gain.value = 0.0;

    fountainSrc.connect(this.fountainFilter);
    this.fountainFilter.connect(this.fountainGain);
    this.fountainGain.connect(this.masterGain);
    fountainSrc.start();

    const waterfallSrc = this.ctx.createBufferSource();
    waterfallSrc.buffer = pinkBuffer;
    waterfallSrc.loop = true;

    this.waterfallFilter = this.ctx.createBiquadFilter();
    this.waterfallFilter.type = 'bandpass';
    this.waterfallFilter.frequency.value = 640;
    this.waterfallFilter.Q.value = 0.55;

    this.waterfallGain = this.ctx.createGain();
    this.waterfallGain.gain.value = 0.0;

    waterfallSrc.connect(this.waterfallFilter);
    this.waterfallFilter.connect(this.waterfallGain);
    this.waterfallGain.connect(this.masterGain);
    waterfallSrc.start();
  }

  setWeatherAudio(weatherMode) {
    if (!this.ctx || !this.rainRoarGain) return;
    const now = this.ctx.currentTime;

    let roar = 0.0;
    let patter = 0.0;
    let windVol = 0.48;
    let windFreq = 390;

    if (weatherMode === 'STORM') {
      roar = 1.25;
      patter = 0.92;
      windVol = 1.15;
      windFreq = 640;
    } else if (weatherMode === 'RAIN') {
      roar = 0.92;
      patter = 0.72;
      windVol = 0.72;
      windFreq = 490;
    } else if (weatherMode === 'MIST') {
      roar = 0.22;
      patter = 0.30;
      windVol = 0.52;
      windFreq = 410;
    }

    this.rainRoarGain.gain.setTargetAtTime(roar, now, 0.45);
    this.rainPatterGain.gain.setTargetAtTime(patter, now, 0.45);
    this.breezeGain.gain.setTargetAtTime(windVol, now, 0.55);
    if (this.breezeFilter) {
      this.breezeFilter.frequency.setTargetAtTime(windFreq, now, 0.55);
    }
  }

  /**
   * Couples Web Audio wind, downpour roar, and eave-drip patter continuously 1:1
   * to the live 20–45s meteorological transition state (zero abrupt audio jumps!).
   */
  syncContinuousWeatherAudio(rainIntensity = 0.0, stormFactor = 0.0, windSpeed = 0.85) {
    if (!this.ctx || !this.enabled || !this.rainRoarGain) return;
    const now = this.ctx.currentTime;
    const rain = Math.max(0.0, Math.min(1.5, rainIntensity));
    const storm = Math.max(0.0, Math.min(1.0, stormFactor));
    const wNorm = Math.max(0.0, Math.min(1.5, (windSpeed - 0.7) / 3.0));

    const roar = Math.min(1.28, Math.pow(rain / 1.42, 1.15) * 1.25);
    const patter = Math.min(0.95, Math.pow(Math.min(1.0, rain * 1.05), 0.85) * (0.72 + 0.22 * storm));
    const windVol = 0.46 + wNorm * 0.52 + storm * 0.22;
    const windFreq = 385 + wNorm * 165 + storm * 95;

    this.rainRoarGain.gain.setTargetAtTime(roar, now, 0.18);
    this.rainPatterGain.gain.setTargetAtTime(patter, now, 0.18);
    this.breezeGain.gain.setTargetAtTime(windVol, now, 0.22);
    if (this.breezeFilter) {
      this.breezeFilter.frequency.setTargetAtTime(windFreq, now, 0.22);
    }
  }

  /**
   * Modulates the DJI Avata 2 5-blade ducted cinewhoop motor harmonic & cornering prop-wash
   * in real time from throttle (0..1), airspeed (m/s), and yaw/bank turn rate.
   */
  syncAvataDroneAudio(active = false, throttle01 = 0.0, speedMs = 0.0, turnRate = 0.0) {
    if (!this.ctx || !this.avataMotorGain || !this.avataWashGain) return;
    const now = this.ctx.currentTime;
    if (!active || !this.enabled) {
      this.avataMotorGain.gain.setTargetAtTime(0.0, now, 0.12);
      this.avataWashGain.gain.setTargetAtTime(0.0, now, 0.12);
      return;
    }
    const thr = Math.max(0.0, Math.min(1.0, throttle01));
    const spd = Math.max(0.0, Math.min(30.0, speedMs));
    const turn = Math.max(0.0, Math.min(4.0, turnRate));
    const baseHz = 215 + thr * 285 + (spd / 27.0) * 65;
    this.avataMotorOscA.frequency.setTargetAtTime(baseHz, now, 0.06);
    this.avataMotorOscB.frequency.setTargetAtTime(baseHz * 2.012, now, 0.06);
    this.avataMotorFilter.frequency.setTargetAtTime(460 + thr * 680, now, 0.08);
    this.avataMotorGain.gain.setTargetAtTime(0.055 + thr * 0.115, now, 0.08);

    const washGain = 0.04 + (spd / 27.0) * 0.18 + turn * 0.045;
    this.avataWashFilter.frequency.setTargetAtTime(620 + (spd / 27.0) * 740 + turn * 140, now, 0.08);
    this.avataWashGain.gain.setTargetAtTime(Math.min(0.32, washGain), now, 0.08);
  }

  /**
   * 3D Proximity-Attenuated Rotunda Dancing Fountain, Niagara Waterfall, & Low-Altitude Coastal Waves
   */
  syncSpatialWaterAudio(camX = 0.0, camY = 20.0, camZ = 100.0) {
    if (!this.ctx || !this.fountainGain || !this.waterfallGain) return;
    const now = this.ctx.currentTime;
    if (!this.enabled) {
      this.fountainGain.gain.setTargetAtTime(0.0, now, 0.15);
      this.waterfallGain.gain.setTargetAtTime(0.0, now, 0.15);
      return;
    }

    // Rotunda Dancing Fountain at (-2.5, 0.3, 106.5)
    const dFount = Math.hypot(camX - -2.5, camY - 2.0, camZ - 106.5);
    const fountProx = Math.max(0.0, 1.0 - dFount / 72.0);
    this.fountainGain.gain.setTargetAtTime(fountProx * fountProx * 0.42, now, 0.16);

    // Pulau I Niagara Waterfall (-44, 9, 643) + Low-Skim Teluk/Selat Wave Wash
    const dNiagara = Math.hypot(camX - -44.0, camY - 9.0, camZ - 643.0);
    const niagProx = Math.max(0.0, 1.0 - dNiagara / 115.0);
    const lowSeaProx = camY > -2.2 && camY < 11.5 ? Math.max(0.0, (11.5 - camY) / 11.5) * 0.16 : 0.0;
    const surfLevel = Math.min(0.62, niagProx * niagProx * 0.56 + lowSeaProx);
    this.waterfallFilter.frequency.setTargetAtTime(480 + niagProx * 420, now, 0.18);
    this.waterfallGain.gain.setTargetAtTime(surfLevel, now, 0.18);
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
    g.gain.linearRampToValueAtTime(0.08, t0 + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.13);
    osc.connect(g);
    g.connect(this.masterGain);
    osc.start(t0);
    osc.stop(t0 + 0.14);
  }

  /**
   * Synthesizes physically delayed rolling thunder propagating at the speed of sound (c = 343 m/s)
   * with atmospheric high-frequency absorption proportional to strike distance.
   */
  triggerThunder(distanceMeters = 650, intensity = 1.0) {
    if (!this.enabled || !this.ctx || !this._pinkBuffer) return;

    const delaySec = Math.max(0.18, Math.min(4.8, distanceMeters / 343.0));
    const timerId = setTimeout(() => {
      if (!this.enabled || !this.ctx || this.ctx.state !== 'running') return;
      const t0 = this.ctx.currentTime;
      const distAtten = Math.max(0.28, Math.min(1.15, 720 / Math.max(260, distanceMeters)));
      const peakGain = Math.min(1.8, intensity * distAtten * 1.45);

      // 1. Filtered noise crack & rolling Cumulonimbus echo tail
      const noiseSrc = this.ctx.createBufferSource();
      noiseSrc.buffer = this._pinkBuffer;
      noiseSrc.playbackRate.value = 0.55 + Math.random() * 0.25;

      const lp = this.ctx.createBiquadFilter();
      lp.type = 'lowpass';
      const initialCutoff = distanceMeters < 580 ? 1550 : 620;
      lp.frequency.setValueAtTime(initialCutoff, t0);
      lp.frequency.exponentialRampToValueAtTime(92, t0 + 2.8);

      const rumbleGain = this.ctx.createGain();
      rumbleGain.gain.setValueAtTime(0.001, t0);
      rumbleGain.gain.linearRampToValueAtTime(peakGain, t0 + 0.045);
      rumbleGain.gain.exponentialRampToValueAtTime(peakGain * 0.48, t0 + 0.38);
      rumbleGain.gain.linearRampToValueAtTime(peakGain * 0.62, t0 + 0.72); // Secondary cloud echo swell
      rumbleGain.gain.exponentialRampToValueAtTime(0.001, t0 + 3.4);

      noiseSrc.connect(lp);
      lp.connect(rumbleGain);
      rumbleGain.connect(this.masterGain);
      noiseSrc.start(t0);
      noiseSrc.stop(t0 + 3.5);

      // 2. Low sub-bass acoustic shockwave thump (42-68 Hz)
      const osc = this.ctx.createOscillator();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(66, t0);
      osc.frequency.exponentialRampToValueAtTime(34, t0 + 1.4);

      const subGain = this.ctx.createGain();
      subGain.gain.setValueAtTime(0.001, t0);
      subGain.gain.linearRampToValueAtTime(peakGain * 0.42, t0 + 0.06);
      subGain.gain.exponentialRampToValueAtTime(0.001, t0 + 1.5);

      osc.connect(subGain);
      subGain.connect(this.masterGain);
      osc.start(t0);
      osc.stop(t0 + 1.6);
    }, delaySec * 1000);

    this._thunderTimers.push(timerId);
    if (this._thunderTimers.length > 12) {
      this._thunderTimers.shift();
    }
  }
}
