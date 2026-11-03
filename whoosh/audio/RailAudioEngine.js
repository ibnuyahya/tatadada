/**
 * ============================================================================
 * LAYER 3: MULTI-BUS REAL-TIME WEB AUDIO SYNTHESIZER (3D DOPPLER + REVERB)
 * Synthesizes Prime Mover, VVVF IGBT Inverter, Welded Rail Joint Clacks,
 * Nathan K5LA Tri-Chord Semboyan 35 Horn, Oncoming AI Pass-By Doppler & Chimes.
 * ============================================================================
 */
export class MultiBusRailAudioEngine {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.initialized = false;
    this.hornActive = false;
  }

  init() {
    if (this.initialized) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioCtx();

      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 0.72;
      this.masterGain.connect(this.ctx.destination);

      // 1. Prime Mover / Traction Motor Hum Bus
      this.engineOsc1 = this.ctx.createOscillator();
      this.engineOsc2 = this.ctx.createOscillator();
      this.engineOsc1.type = 'sawtooth';
      this.engineOsc2.type = 'triangle';
      this.engineFilter = this.ctx.createBiquadFilter();
      this.engineFilter.type = 'lowpass';
      this.engineFilter.frequency.value = 180;
      this.engineGain = this.ctx.createGain();
      this.engineGain.gain.value = 0.05;

      this.engineOsc1.connect(this.engineFilter);
      this.engineOsc2.connect(this.engineFilter);
      this.engineFilter.connect(this.engineGain);
      this.engineGain.connect(this.masterGain);
      this.engineOsc1.start();
      this.engineOsc2.start();

      // 2. VVVF IGBT Inverter Harmonic Bus
      this.vvvfOsc = this.ctx.createOscillator();
      this.vvvfOsc.type = 'sine';
      this.vvvfGain = this.ctx.createGain();
      this.vvvfGain.gain.value = 0.0;
      this.vvvfOsc.connect(this.vvvfGain);
      this.vvvfGain.connect(this.masterGain);
      this.vvvfOsc.start();

      // 3. White Noise Buffer for Rail Friction, Pneumatic Hiss & Joint Clacks
      const bufLen = this.ctx.sampleRate * 2;
      this.noiseBuffer = this.ctx.createBuffer(1, bufLen, this.ctx.sampleRate);
      const data = this.noiseBuffer.getChannelData(0);
      for (let i = 0; i < bufLen; i++) data[i] = Math.random() * 2 - 1;

      this.rollNoise = this.ctx.createBufferSource();
      this.rollNoise.buffer = this.noiseBuffer;
      this.rollNoise.loop = true;
      this.rollFilter = this.ctx.createBiquadFilter();
      this.rollFilter.type = 'bandpass';
      this.rollFilter.frequency.value = 240;
      this.rollFilter.Q.value = 1.5;
      this.rollGain = this.ctx.createGain();
      this.rollGain.gain.value = 0.0;

      this.rollNoise.connect(this.rollFilter);
      this.rollFilter.connect(this.rollGain);
      this.rollGain.connect(this.masterGain);
      this.rollNoise.start();

      // 4. Curve Flange Squeal & Wet Wheel-Slip Metallic Screech Bus
      this.squealNoise = this.ctx.createBufferSource();
      this.squealNoise.buffer = this.noiseBuffer;
      this.squealNoise.loop = true;
      this.squealFilter = this.ctx.createBiquadFilter();
      this.squealFilter.type = 'bandpass';
      this.squealFilter.frequency.value = 2680;
      this.squealFilter.Q.value = 11.5;
      this.squealGain = this.ctx.createGain();
      this.squealGain.gain.value = 0.0;

      this.squealNoise.connect(this.squealFilter);
      this.squealFilter.connect(this.squealGain);
      this.squealGain.connect(this.masterGain);
      this.squealNoise.start();

      // 5. Nathan K5LA Tri-Chord Air Horn Bus
      this.hornOscs = [311.13, 370.0, 466.16].map((freq) => {
        const osc = this.ctx.createOscillator();
        osc.type = 'sawtooth';
        osc.frequency.value = freq;
        osc.start();
        return osc;
      });
      this.hornFilter = this.ctx.createBiquadFilter();
      this.hornFilter.type = 'lowpass';
      this.hornFilter.frequency.value = 1550;
      this.hornGain = this.ctx.createGain();
      this.hornGain.gain.value = 0.0;

      this.hornOscs.forEach((o) => o.connect(this.hornFilter));
      this.hornFilter.connect(this.hornGain);
      this.hornGain.connect(this.masterGain);

      // 6. Oncoming AI Train Doppler Pass-By Bus (Separated Rolling Hum vs Pre-Pass Horn!)
      this.aiHumOsc = this.ctx.createOscillator();
      this.aiHumOsc.type = 'sawtooth';
      this.aiHumOsc.frequency.value = 160;
      this.aiHumGain = this.ctx.createGain();
      this.aiHumGain.gain.value = 1.0;

      this.aiHornOsc = this.ctx.createOscillator();
      this.aiHornOsc.type = 'sawtooth';
      this.aiHornOsc.frequency.value = 392;
      this.aiHornGain = this.ctx.createGain();
      this.aiHornGain.gain.value = 0.0;

      this.aiFilter = this.ctx.createBiquadFilter();
      this.aiFilter.type = 'lowpass';
      this.aiFilter.frequency.value = 950;
      this.aiGain = this.ctx.createGain();
      this.aiGain.gain.value = 0.0;

      this.aiHumOsc.connect(this.aiHumGain);
      this.aiHumGain.connect(this.aiFilter);
      this.aiHornOsc.connect(this.aiHornGain);
      this.aiHornGain.connect(this.aiFilter);
      this.aiFilter.connect(this.aiGain);
      this.aiGain.connect(this.masterGain);
      this.aiHumOsc.start();
      this.aiHornOsc.start();

      // 7. Level Crossing Bell Doppler Bus
      this.cxBellOsc = this.ctx.createOscillator();
      this.cxBellOsc.type = 'sine';
      this.cxBellOsc.frequency.value = 1180;
      this.cxBellGain = this.ctx.createGain();
      this.cxBellGain.gain.value = 0.0;
      this.cxBellOsc.connect(this.cxBellGain);
      this.cxBellGain.connect(this.masterGain);
      this.cxBellOsc.start();

      this.initialized = true;
    } catch (e) {
      console.warn('WebAudio skipped:', e);
    }
  }

  update(
    speedMps,
    throttlePct,
    curvature,
    doppler = 1.0,
    inTunnelOrBridge = false,
    aiDoppler = 1.0,
    aiDist = 999,
    crossingDist = 999,
    crossingDoppler = 1.0,
    tractionType = 'ELECTRIC',
    wheelSlipping = false,
    brakePct = 0,
    aiApproaching = false
  ) {
    if (!this.initialized || !this.enabled) return;
    if (this.ctx.state === 'suspended') this.ctx.resume();

    const now = this.ctx.currentTime;
    const absV = Math.abs(speedMps);
    const isElectric = (tractionType === 'ELECTRIC');

    if (isElectric) {
      // Electric EMU (CR400AF Whoosh / KRL / Glacier): quiet 50/100Hz auxiliary transformer hum + multi-stage VVVF IGBT sweep
      const auxFreq = (50 + absV * 2.1) * doppler;
      this.engineOsc1.frequency.setTargetAtTime(auxFreq * 2.0, now, 0.08);
      this.engineOsc2.frequency.setTargetAtTime(auxFreq, now, 0.08);
      this.engineFilter.frequency.setTargetAtTime(140 + absV * 3.0, now, 0.08);
      this.engineGain.gain.setTargetAtTime(0.012 + Math.min(0.028, absV * 0.0005), now, 0.08);

      const regenActive = (brakePct > 8 && absV > 3.5);
      const invEffort = regenActive ? (brakePct * 0.65) : throttlePct;
      let carrierHz = 220;
      if (absV < 14.0) {
        carrierHz = 240 + (absV / 14.0) * 620; // Asynchronous PWM carrier melody
      } else {
        carrierHz = Math.min(1720, 95 + absV * 13.8); // Synchronous 3-phase traction motor fundamental
      }
      this.vvvfOsc.frequency.setTargetAtTime(carrierHz * doppler, now, 0.04);
      const vvvfVol = (absV > 0.35 && invEffort > 2)
        ? Math.min(0.068, 0.012 + (invEffort / 100) * 0.052)
        : 0.0;
      this.vvvfGain.gain.setTargetAtTime(vvvfVol, now, 0.05);
    } else {
      // Diesel-Electric GE 7FDL V12 (CC206 / CC205): 8-notch cylinder firing rumble + turbocharger whistle
      const notch = Math.round((throttlePct / 100) * 8);
      const fBase = (34 + notch * 7.8) * doppler;
      this.engineOsc1.frequency.setTargetAtTime(fBase, now, 0.07);
      this.engineOsc2.frequency.setTargetAtTime(fBase * 0.5, now, 0.07);
      this.engineFilter.frequency.setTargetAtTime(165 + notch * 52, now, 0.07);
      this.engineGain.gain.setTargetAtTime(0.038 + (notch / 8) * 0.078, now, 0.07);

      const turboFreq = (920 + (notch / 8) * 1180) * doppler;
      this.vvvfOsc.frequency.setTargetAtTime(turboFreq, now, 0.08);
      const turboVol = notch >= 2 ? (0.006 + (notch / 8) * 0.024) : 0.0;
      this.vvvfGain.gain.setTargetAtTime(turboVol, now, 0.08);
    }

    // Rolling rail noise + quadratic high-speed aerodynamic slipstream roar (v^2 above 150 km/h)
    const acousticMult = inTunnelOrBridge ? 1.72 : 1.0;
    const aeroQuad = Math.pow(Math.min(1.0, absV / 97.2), 2.0) * 0.11;
    const rollVol = Math.min(0.28, ((absV / 65) * 0.11 + aeroQuad) * acousticMult);
    const rollFreq = (inTunnelOrBridge ? 175 : 245) + absV * 9.8;
    this.rollFilter.frequency.setTargetAtTime(rollFreq * doppler, now, 0.06);
    this.rollGain.gain.setTargetAtTime(rollVol, now, 0.06);

    // Metallic flange squeal on tight curves & wet-rail wheel-slip screech
    if (this.squealGain) {
      const absCurv = Math.abs(curvature || 0);
      let squealVol = 0.0;
      if (wheelSlipping && absV > 0.4) {
        squealVol = 0.042;
      } else if (absCurv > 0.00014 && absV > 12.0) {
        squealVol = Math.min(0.032, (absCurv - 0.00014) * 140.0 * Math.min(1.0, absV / 45.0));
      }
      this.squealFilter.frequency.setTargetAtTime((wheelSlipping ? 2950 : 2580) * doppler, now, 0.05);
      this.squealGain.gain.setTargetAtTime(squealVol, now, 0.06);
    }

    if (this.hornActive) {
      const baseFreqs = [311.13, 370.0, 466.16];
      this.hornOscs.forEach((osc, i) => {
        osc.frequency.setTargetAtTime(baseFreqs[i] * doppler, now, 0.03);
      });
    }

    // Oncoming AI Train Doppler: blow Semboyan 35 warning horn ONLY prior to cab pass-by!
    if (aiDist < 280) {
      const atten = Math.pow(Math.max(0, 1.0 - aiDist / 280), 2.0);
      this.aiHumOsc.frequency.setTargetAtTime(148 * aiDoppler, now, 0.03);
      this.aiHornOsc.frequency.setTargetAtTime(392 * aiDoppler, now, 0.03);
      if (this.aiHornGain) {
        const blowAiHorn = (aiApproaching && aiDist < 230 && aiDist > 12) ? 0.68 : 0.0;
        this.aiHornGain.gain.setTargetAtTime(blowAiHorn, now, 0.04);
      }
      this.aiGain.gain.setTargetAtTime(atten * 0.14, now, 0.04);
    } else {
      if (this.aiHornGain) this.aiHornGain.gain.setTargetAtTime(0.0, now, 0.04);
      this.aiGain.gain.setTargetAtTime(0.0, now, 0.08);
    }

    if (crossingDist < 190) {
      const pulse = Math.sin(performance.now() * 0.022) > 0 ? 1.0 : 0.15;
      const cxAtten = Math.pow(Math.max(0, 1.0 - crossingDist / 190), 1.8);
      this.cxBellOsc.frequency.setTargetAtTime(1180 * crossingDoppler, now, 0.03);
      this.cxBellGain.gain.setTargetAtTime(cxAtten * 0.07 * pulse, now, 0.02);
    } else {
      this.cxBellGain.gain.setTargetAtTime(0.0, now, 0.06);
    }
  }

  playRailJointClack(speedMps, inTunnelOrBridge = false, doppler = 1.0) {
    if (!this.initialized || !this.enabled || !this.noiseBuffer) return;
    const absV = Math.abs(speedMps);
    if (absV < 0.8) return;

    const triggerTap = (delay, pitch) => {
      const t = this.ctx.currentTime + delay;
      const src = this.ctx.createBufferSource();
      src.buffer = this.noiseBuffer;
      const flt = this.ctx.createBiquadFilter();
      flt.type = 'bandpass';
      flt.frequency.value = (inTunnelOrBridge ? 440 : 720) * pitch * doppler;
      flt.Q.value = 3.8;
      const g = this.ctx.createGain();
      const vol = Math.min(0.22, 0.04 + (absV / 55) * 0.14);
      g.gain.setValueAtTime(vol, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.055);
      src.connect(flt);
      flt.connect(g);
      g.connect(this.masterGain);
      src.start(t);
      src.stop(t + 0.06);
    };

    const wheelDelay = Math.max(0.024, Math.min(0.24, 2.50 / absV));
    triggerTap(0, 1.0);
    triggerTap(wheelDelay, 0.88);
  }

  playPassingTrainWhoosh() {
    if (!this.initialized || !this.enabled || !this.noiseBuffer) return;
    const now = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noiseBuffer;

    const flt = this.ctx.createBiquadFilter();
    flt.type = 'bandpass';
    flt.frequency.setValueAtTime(720, now);
    flt.frequency.exponentialRampToValueAtTime(160, now + 0.85);
    flt.Q.value = 2.0;

    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.01, now);
    g.gain.linearRampToValueAtTime(0.28, now + 0.22);
    g.gain.exponentialRampToValueAtTime(0.001, now + 0.9);

    src.connect(flt);
    flt.connect(g);
    g.connect(this.masterGain);
    src.start(now);
    src.stop(now + 0.95);
  }

  setHorn(active, doppler = 1.0) {
    this.hornActive = active;
    if (!this.initialized || !this.enabled) return;
    const now = this.ctx.currentTime;
    const baseFreqs = [311.13, 370.0, 466.16];
    this.hornOscs.forEach((osc, i) => {
      osc.frequency.setTargetAtTime(baseFreqs[i] * doppler, now, 0.03);
    });
    this.hornGain.gain.setTargetAtTime(active ? 0.18 : 0.0, now, active ? 0.04 : 0.12);
  }

  playPneumaticHiss() {
    if (!this.initialized || !this.enabled || !this.noiseBuffer) return;
    const now = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noiseBuffer;
    const flt = this.ctx.createBiquadFilter();
    flt.type = 'highpass';
    flt.frequency.value = 2300;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.12, now);
    g.gain.exponentialRampToValueAtTime(0.001, now + 0.42);
    src.connect(flt);
    flt.connect(g);
    g.connect(this.masterGain);
    src.start(now);
    src.stop(now + 0.45);
  }

  playSifaBeep() {
    if (!this.initialized || !this.enabled) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = 'square';
    osc.frequency.setValueAtTime(1480, now);
    g.gain.setValueAtTime(0.12, now);
    g.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
    osc.connect(g);
    g.connect(this.masterGain);
    osc.start(now);
    osc.stop(now + 0.38);
  }

  playChime(type = 'STATION') {
    if (!this.initialized || !this.enabled) return;
    const freqs = type === 'STATION' ? [523.25, 659.25, 783.99, 1046.5] : [880, 659.25];
    freqs.forEach((f, idx) => {
      const t = this.ctx.currentTime + idx * 0.13;
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = f;
      g.gain.setValueAtTime(0.11, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.55);
      osc.connect(g);
      g.connect(this.masterGain);
      osc.start(t);
      osc.stop(t + 0.6);
    });
  }
}
