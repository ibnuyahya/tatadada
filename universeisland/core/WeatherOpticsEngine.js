import * as THREE from 'three';
import { RainCollisionBake } from './RainCollisionBake.js';
import { HumanEyeHDROptics } from './HumanEyeHDROptics.js';
import {
  computeBaseJD0WIB,
  evaluateDepokAstronomy,
  createRealStarCatalogGridTexture
} from './AstronomicalSkyCatalog.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.2 Super-Realistic Physical Weather Edition)
 * ./core/WeatherOpticsEngine.js
 *
 * Implements Hukum Emas 8, 12, 13, & 14 + Full Meteorological & Fluid-Impact Physics:
 * 1. Analytical Sky Dome with:
 *    - Exact Sphere Sun/Moon Projection (z_sphere = sqrt(max(0.0, 1.0 - r_norm^2)))
 *    - Radial Euclidean SDF Starfield
 *    - 2-Layer Parallax 6-Octave Domain-Rotated FBM Cumulonimbus Clouds (R = mat2(0.8, 0.6, -0.6, 0.8))
 *      with Beer-Lambert Self-Shadowing, Horizon Virga Rain Shafts, & Multi-Branch Forked Plasma Lightning!
 * 2. 100% GPU Physics-Driven Raindrops & Newtonian Roof/Canopy Drip Curtains (1 Draw Call, 6,800 Quads):
 *    - Marshall-Palmer Drop-Size Distribution (D in [0.95, 4.65] mm)
 *    - Gunn-Kinzer Terminal Velocity v_t(D) = 9.65 - 10.3 * exp(-0.6 * D)
 *    - Boundary-Layer Wind-Shear Profile W(y) ~ (y / 10)^0.22 + Aerodynamic Drag Coupling
 *    - Exact 512x512 3D Scene Collision Z-Buffer Interception (keeping covered walkways/interiors 100% dry)
 *    - Pendant-Bead Accumulation + Newtonian Free-Fall y(t) = y_eave - 0.5 * g * t^2 along 3D Roof Eaves & Trees
 * 3. 100% GPU 3D Surface Impact Splashes, Parabolic Rebound Micro-Droplets, & Roof Spray Mist (1 Draw Call, 5,200 Quads):
 *    - Rayleigh-Worthington Crown Rings aligned to exact 3D surface normal N_hit(x, z)
 *    - Ballistic Rebound Micro-Droplets P(t) = P_hit + v_0 * t + 0.5 * g * t^2
 *    - Wind-Drifting Impact Aerosol Spray Mist over roofs & plazas
 * 4. Danau Kenanga Rotated 6-Octave FBM Water with Multi-Layer Capillary Raindrop Ripples & Storm Chop.
 */
export class WeatherOpticsEngine {
  static SKY_DROP_COUNT = 5200;
  static EAVE_DRIP_COUNT = 2000;
  static SPLASH_CROWN_COUNT = 2600;
  static REBOUND_DROP_COUNT = 2000;
  static SPRAY_MIST_COUNT = 600;

  static MIN_TRANSITION_SEC = 20.0;
  static MAX_TRANSITION_SEC = 45.0;
  static DEFAULT_TRANSITION_SEC = 25.0;
  static SLIDER_FULL_SPAN_SEC = 45.0;
  static MAX_SEA_WAVE_M = 3.0;
  static SEA_STATE_NORM_DIV = 0.854;

  // 8-Component Dispersive Gravity-Swell & Wind-Sea Modes (Time-Invariant k_i & \omega_i = \sqrt{g k_i} -> 0 phase-jump during 45s transitions!)
  // Modes 1 & 2 (L=82m & L=65.5m crossing at 17 deg) generate natural 5-wave ocean swell groups ("wave sets")!
  static OCEAN_WAVE_MODES = [
    { k: 0.076624, w: 0.8670, dx: 0.8660, dz: 0.5000, phase0: 0.00, q: 0.34, L: 82.0 },
    { k: 0.095926, w: 0.9701, dx: 0.9744, dz: 0.2250, phase0: 1.19, q: 0.33, L: 65.5 },
    { k: 0.161107, w: 1.2572, dx: 0.5299, dz: 0.8480, phase0: 2.47, q: 0.31, L: 39.0 },
    { k: 0.237101, w: 1.5251, dx: 0.9613, dz: -0.2756, phase0: 3.83, q: 0.29, L: 26.5 },
    { k: 0.352988, w: 1.8609, dx: 0.7193, dz: 0.6947, phase0: 5.11, q: 0.27, L: 17.8 },
    { k: 0.551157, w: 2.3253, dx: 0.8910, dz: -0.4540, phase0: 0.73, q: 0.24, L: 11.4 },
    { k: 0.884956, w: 2.9464, dx: 0.1908, dz: 0.9816, phase0: 4.29, q: 0.21, L: 7.1 },
    { k: 1.427997, w: 3.7428, dx: 0.9455, dz: 0.3256, phase0: 2.08, q: 0.18, L: 4.4 }
  ];

  static TOD_PROFILES = {
    DAWN: { id: 'DAWN', labelId: 'Fajar', timeText: '05:45', solarHour: 5.75 },
    MORNING: { id: 'MORNING', labelId: 'Pagi', timeText: '08:00', solarHour: 8.0 },
    DAY: { id: 'DAY', labelId: 'Siang', timeText: '11:51', solarHour: 11.85 },
    AFTERNOON: { id: 'AFTERNOON', labelId: 'Sore', timeText: '15:30', solarHour: 15.5 },
    DUSK: { id: 'DUSK', labelId: 'Senja', timeText: '17:51', solarHour: 17.85 },
    NIGHT: { id: 'NIGHT', labelId: 'Malam', timeText: '22:00', solarHour: 22.0 }
  };

  static WEATHER_PROFILES = {
    CLEAR: { id: 'CLEAR', labelId: 'Cerah', rain: 0.0, wet: 0.0, cloud: 0.22, storm: 0.0, fog: 0.0, wind: 0.82 },
    PARTLY_CLOUDY: { id: 'PARTLY_CLOUDY', labelId: 'Cerah Berawan', rain: 0.0, wet: 0.0, cloud: 0.48, storm: 0.0, fog: 0.0, wind: 1.05 },
    OVERCAST: { id: 'OVERCAST', labelId: 'Mendung', rain: 0.0, wet: 0.06, cloud: 0.88, storm: 0.18, fog: 0.16, wind: 1.4 },
    WINDY: { id: 'WINDY', labelId: 'Angin Kencang', rain: 0.0, wet: 0.0, cloud: 0.52, storm: 0.12, fog: 0.0, wind: 3.25 },
    FOGGY: { id: 'FOGGY', labelId: 'Kabut Embun', rain: 0.0, wet: 0.34, cloud: 0.76, storm: 0.02, fog: 0.92, wind: 0.46 },
    POST_RAIN: { id: 'POST_RAIN', labelId: 'Seusai Hujan', rain: 0.0, wet: 0.92, cloud: 0.34, storm: 0.0, fog: 0.18, wind: 0.86 },
    MIST: { id: 'MIST', labelId: 'Gerimis', rain: 0.34, wet: 0.58, cloud: 0.74, storm: 0.05, fog: 0.0, wind: 1.2 },
    SHOWER: { id: 'SHOWER', labelId: 'Hujan Ringan', rain: 0.66, wet: 0.82, cloud: 0.84, storm: 0.14, fog: 0.08, wind: 1.62 },
    RAIN: { id: 'RAIN', labelId: 'Hujan Deras', rain: 1.0, wet: 0.96, cloud: 0.92, storm: 0.28, fog: 0.0, wind: 2.05 },
    STORM: { id: 'STORM', labelId: 'Badai Petir', rain: 1.42, wet: 1.0, cloud: 1.0, storm: 1.0, fog: 0.0, wind: 3.55 }
  };

  // Continuous [0.00 .. 1.00] Meteorological Spectrum Knots (C1-smooth slider interpolation)
  static WEATHER_SPECTRUM_KNOTS = [
    { u: 0.00, mode: 'CLEAR', labelId: 'Cerah Murni', rain: 0.0, wet: 0.0, cloud: 0.18, storm: 0.0, fog: 0.0, wind: 0.78, waveM: 0.18 },
    { u: 0.15, mode: 'PARTLY_CLOUDY', labelId: 'Cerah Berawan', rain: 0.0, wet: 0.0, cloud: 0.48, storm: 0.0, fog: 0.0, wind: 1.05, waveM: 0.30 },
    { u: 0.30, mode: 'OVERCAST', labelId: 'Mendung', rain: 0.0, wet: 0.06, cloud: 0.86, storm: 0.16, fog: 0.14, wind: 1.42, waveM: 0.48 },
    { u: 0.44, mode: 'FOGGY', labelId: 'Kabut Embun', rain: 0.0, wet: 0.34, cloud: 0.78, storm: 0.04, fog: 0.88, wind: 0.62, waveM: 0.24 },
    { u: 0.58, mode: 'MIST', labelId: 'Gerimis', rain: 0.34, wet: 0.58, cloud: 0.76, storm: 0.06, fog: 0.16, wind: 1.24, waveM: 0.64 },
    { u: 0.72, mode: 'SHOWER', labelId: 'Hujan Ringan', rain: 0.68, wet: 0.82, cloud: 0.85, storm: 0.16, fog: 0.08, wind: 1.66, waveM: 1.04 },
    { u: 0.86, mode: 'RAIN', labelId: 'Hujan Deras', rain: 1.02, wet: 0.96, cloud: 0.93, storm: 0.34, fog: 0.03, wind: 2.18, waveM: 1.58 },
    { u: 1.00, mode: 'STORM', labelId: 'Badai Petir', rain: 1.42, wet: 1.00, cloud: 1.00, storm: 1.00, fog: 0.00, wind: 3.55, waveM: 2.35 }
  ];

  static weatherModeToSpectrum(mode) {
    switch (mode) {
      case 'CLEAR': return 0.00;
      case 'PARTLY_CLOUDY': return 0.15;
      case 'POST_RAIN': return 0.24;
      case 'OVERCAST': return 0.30;
      case 'WINDY': return 0.36;
      case 'FOGGY': return 0.44;
      case 'MIST': return 0.58;
      case 'SHOWER': return 0.72;
      case 'RAIN': return 0.86;
      case 'STORM': return 1.00;
      default: return 0.00;
    }
  }

  static evaluateWeatherSpectrum(u01) {
    const u = Math.max(0.0, Math.min(1.0, Number(u01) || 0.0));
    const knots = WeatherOpticsEngine.WEATHER_SPECTRUM_KNOTS;
    let k0 = knots[0];
    let k1 = knots[knots.length - 1];
    for (let i = 0; i < knots.length - 1; i++) {
      if (u >= knots[i].u && u <= knots[i + 1].u) {
        k0 = knots[i];
        k1 = knots[i + 1];
        break;
      }
    }
    const span = Math.max(1e-5, k1.u - k0.u);
    const linT = Math.max(0.0, Math.min(1.0, (u - k0.u) / span));
    const s = linT * linT * (3.0 - 2.0 * linT);
    const near = linT < 0.5 ? k0 : k1;
    return {
      u,
      percent: Math.round(u * 100.0),
      mode: near.mode,
      labelId: near.labelId,
      rain: k0.rain + (k1.rain - k0.rain) * s,
      wet: k0.wet + (k1.wet - k0.wet) * s,
      cloud: k0.cloud + (k1.cloud - k0.cloud) * s,
      storm: k0.storm + (k1.storm - k0.storm) * s,
      fog: k0.fog + (k1.fog - k0.fog) * s,
      wind: k0.wind + (k1.wind - k0.wind) * s,
      autoWaveM: k0.waveM + (k1.waveM - k0.waveM) * s
    };
  }

  static formatSolarHourWIB(hour24) {
    const raw = Math.max(0.0, Math.min(24.0, Number(hour24) || 0.0));
    if (raw >= 23.992) return '24:00';
    const wrapped = ((raw % 24.0) + 24.0) % 24.0;
    const totalMin = Math.round(wrapped * 60.0) % 1440;
    const hh = String(Math.floor(totalMin / 60)).padStart(2, '0');
    const mm = String(totalMin % 60).padStart(2, '0');
    return `${hh}:${mm}`;
  }

  static describeSeaStateLabel(waveHeightM) {
    const h = Math.max(0.0, Number(waveHeightM) || 0.0);
    if (h < 0.08) return 'Laut Tenang';
    if (h < 0.40) return 'Riak Halus';
    if (h < 0.95) return 'Ombak Kecil';
    if (h < 1.65) return 'Ombak Sedang';
    if (h < 2.35) return 'Ombak Besar';
    return 'Gelora Besar';
  }

  /**
   * Exact C^1 Trapezoidal Cosine-Ramp Proportional Integrator Fraction E(t) in [0, 1]:
   * - Guarantees E(0) = 0 and E(duration) = 1 (so 0 -> FullSpan takes EXACTLY 45.000s, half-span takes EXACTLY 22.500s!).
   * - Preserves non-zero cruise velocity across continuous 60 Hz slider drag events via launchRampSec -> 0 once moving!
   */
  static _evalProportionalEase(elapsed, duration, launchRampSec = 0.35, landingRampSec = 0.35) {
    const D = Math.max(1e-5, duration);
    const t = Math.max(0.0, Math.min(D, elapsed));
    if (t >= D) return 1.0;
    if (t <= 0.0) return 0.0;
    const tL = Math.max(0.0, Math.min(launchRampSec, 0.25 * D));
    const tR = Math.max(0.0, Math.min(landingRampSec, 0.25 * D));
    const totalW = Math.max(1e-6, D - 0.5 * tL - 0.5 * tR);

    let area = 0.0;
    if (t <= tL && tL > 1e-6) {
      area = 0.5 * (t - (tL / Math.PI) * Math.sin((Math.PI * t) / tL));
    } else {
      area += 0.5 * tL;
      const cEnd = D - tR;
      const tCruise = Math.min(t, cEnd) - tL;
      if (tCruise > 0.0) area += tCruise;
      if (t > cEnd && tR > 1e-6) {
        const u = t - cEnd;
        area += 0.5 * (u + (tR / Math.PI) * Math.sin((Math.PI * u) / tR));
      }
    }
    return Math.max(0.0, Math.min(1.0, area / totalW));
  }

  /**
   * Computes exact 8-mode Hasselmann (1973) JONSWAP + Phillips/Miche steepness-saturated offshore wave amplitudes
   * calibrated so Significant Wave Height H_s = 4 * sqrt(m_0) = 4 * sqrt(0.5 * sum(a_i^2)) === waveHeightM in [0..3.0m]!
   */
  static computeJonswap8Amplitudes(waveHeightM, outArr8 = null) {
    const out = outArr8 || new Float32Array(8);
    const Hs = Math.max(0.0, Math.min(WeatherOpticsEngine.MAX_SEA_WAVE_M, Number(waveHeightM) || 0.0));
    if (Hs <= 1e-5) {
      for (let i = 0; i < 8; i++) out[i] = 0.0;
      return out;
    }

    const modes = WeatherOpticsEngine.OCEAN_WAVE_MODES;
    // Peak period T_p shifts from 2.35s (capillary-gravity ripples) -> 7.29s (82m storm rollers at H_s = 3.0m)
    const Tp = 2.35 + 2.85 * Math.sqrt(Hs);
    const wp = (2.0 * Math.PI) / Tp;

    let sumSq = 0.0;
    for (let i = 0; i < 8; i++) {
      const wi = modes[i].w;
      const r = wi / wp;
      const rInv4 = 1.0 / (r * r * r * r);
      const pm = Math.pow(r, -5.0) * Math.exp(-1.25 * rInv4);
      const sig = r <= 1.0 ? 0.07 : 0.09;
      const enh = Math.pow(3.3, Math.exp(-((r - 1.0) * (r - 1.0)) / (2.0 * sig * sig)));
      // Logarithmic frequency bin width d_omega ~ omega_i -> energy bin S(omega_i) * omega_i
      const rawA = Math.sqrt(Math.max(1e-12, pm * enh * wi));
      out[i] = rawA;
      sumSq += rawA * rawA;
    }

    // Exact m_0 = 0.5 * sum(a_i^2), H_s = 4 * sqrt(m_0) = 2 * sqrt(2 * sum(a_i^2))
    const normScale = Hs / (2.0 * Math.sqrt(2.0 * Math.max(1e-12, sumSq)));

    // Apply Phillips equilibrium / Miche breaking steepness cap (k_i * a_i <= 0.135) & redistribute any clipped tail energy
    let clippedEnergy = 0.0;
    let freeEnergy = 0.0;
    for (let i = 0; i < 8; i++) {
      const aUnclipped = out[i] * normScale;
      const aMaxMiche = 0.135 / modes[i].k;
      if (aUnclipped > aMaxMiche) {
        out[i] = aMaxMiche;
        clippedEnergy += aUnclipped * aUnclipped - aMaxMiche * aMaxMiche;
      } else {
        out[i] = aUnclipped;
        freeEnergy += aUnclipped * aUnclipped;
      }
    }
    if (clippedEnergy > 1e-8 && freeEnergy > 1e-8) {
      const boost = Math.sqrt(1.0 + clippedEnergy / freeEnergy);
      for (let i = 0; i < 8; i++) {
        const aMaxMiche = 0.135 / modes[i].k;
        if (out[i] < aMaxMiche - 1e-6) {
          out[i] = Math.min(aMaxMiche, out[i] * boost);
        }
      }
    }
    return out;
  }

  static _evalIslandRadiusAnalyticJS(theta) {
    const eastCove = -46.0 * Math.pow(0.5 + 0.5 * Math.cos(theta + 0.25), 4.4);
    const neHeadland = 23.5 * Math.pow(0.5 + 0.5 * Math.cos(theta + 1.04), 8.5);
    const southRotundaLobe = 15.5 * Math.pow(0.5 + 0.5 * Math.cos(theta - 1.12), 2.3);
    const swHelipadPeninsula = 20.5 * Math.pow(0.5 + 0.5 * Math.cos(theta - 2.44), 6.5);
    const organicMeander =
      2.15 * Math.sin(3.0 * theta + 0.40) +
      1.20 * Math.cos(5.0 * theta - 0.70) +
      0.55 * Math.sin(8.0 * theta + 1.10);
    return 121.0 + eastCove + neHeadland + southRotundaLobe + swHelipadPeninsula + organicMeander;
  }

  static _ensureOceanCoastLut() {
    if (WeatherOpticsEngine._oceanRadiusLut) return;
    const N = 8192;
    const TWO_PI = Math.PI * 2.0;
    const step = TWO_PI / N;
    const rLut = new Float64Array(N + 1);
    const invNormLut = new Float64Array(N + 1);
    const eps = 0.008;
    for (let i = 0; i <= N; i++) {
      const th = -Math.PI + i * step;
      const r0 = WeatherOpticsEngine._evalIslandRadiusAnalyticJS(th);
      const rP = WeatherOpticsEngine._evalIslandRadiusAnalyticJS(th + eps);
      const rM = WeatherOpticsEngine._evalIslandRadiusAnalyticJS(th - eps);
      const slopeRatio = (rP - rM) / (2.0 * eps * Math.max(20.0, r0));
      rLut[i] = r0;
      invNormLut[i] = 1.0 / Math.sqrt(1.0 + slopeRatio * slopeRatio);
    }
    WeatherOpticsEngine._oceanLutInvStep = N / TWO_PI;
    WeatherOpticsEngine._oceanRadiusLut = rLut;
    WeatherOpticsEngine._oceanInvNormLut = invNormLut;
  }

  static _islandRadiusAtAngleJS(theta) {
    WeatherOpticsEngine._ensureOceanCoastLut();
    const TWO_PI = Math.PI * 2.0;
    let u = theta + Math.PI;
    if (u < 0.0 || u >= TWO_PI) {
      u = ((u % TWO_PI) + TWO_PI) % TWO_PI;
    }
    const f = u * WeatherOpticsEngine._oceanLutInvStep;
    const idx = f | 0;
    const frac = f - idx;
    const rLut = WeatherOpticsEngine._oceanRadiusLut;
    return rLut[idx] + (rLut[idx + 1] - rLut[idx]) * frac;
  }

  static _evalArchipelagoCoastRawJS(x, z) {
    let distUCenter;
    if (x <= -145.0) {
      distUCenter = Math.abs(Math.hypot(x + 145.0, z - 207.0) - 185.0);
    } else {
      const dLegL = Math.hypot(Math.max(0.0, x + 62.0), z - 22.0);
      const dLegR = Math.hypot(Math.max(0.0, x - 10.0), z - 392.0);
      distUCenter = Math.min(dLegL, dLegR);
    }
    const meanderU = 2.0 * Math.sin(x * 0.035 + z * 0.025) + 1.1 * Math.cos(z * 0.048 - x * 0.031);
    const dUHorseshoe = distUCenter - 74.0 - meanderU;
    let dU = dUHorseshoe;
    if (!(z > 250.0 || x < -195.0)) {
      const dx = x - -18.0;
      const dz = z - 34.0;
      const r = Math.hypot(dx, dz) + 1e-4;
      if (r - 166.5 < dUHorseshoe + 28.0) {
        WeatherOpticsEngine._ensureOceanCoastLut();
        const theta = Math.atan2(dz, dx);
        const f = (theta + Math.PI) * WeatherOpticsEngine._oceanLutInvStep;
        const idx = Math.max(0, Math.min(8191, f | 0));
        const frac = f - idx;
        const rLut = WeatherOpticsEngine._oceanRadiusLut;
        const invLut = WeatherOpticsEngine._oceanInvNormLut;
        const r0 = rLut[idx] + (rLut[idx + 1] - rLut[idx]) * frac;
        const invNorm = invLut[idx] + (invLut[idx + 1] - invLut[idx]) * frac;
        const dCampusHead = (r - r0) * invNorm;
        const h = Math.max(0.0, 28.0 - Math.abs(dCampusHead - dUHorseshoe)) / 28.0;
        dU = Math.min(dCampusHead, dUHorseshoe) - h * h * 28.0 * 0.25;
      }
    }

    const qx = Math.abs(x + 160.0) - 188.0;
    const qz = Math.abs(z - 694.0) - 32.0;
    const outDist = Math.hypot(Math.max(qx, 0.0), Math.max(qz, 0.0));
    const inDist = Math.min(Math.max(qx, qz), 0.0);
    const meanderI = 1.9 * Math.sin(x * 0.039 - z * 0.027) + 1.05 * Math.cos(x * 0.061 + z * 0.033);
    const dI = outDist + inDist - 56.0 - meanderI;
    return Math.min(dU, dI);
  }

  /**
   * Exact 1:1 CPU Mirror of the GPU 8-Mode JONSWAP + Headland Diffraction + Finite-Depth Shoaling
   * + Battjes-Janssen Surf Breaking + 2nd-Order Stokes Ocean Hydrodynamics Field.
   * Used by DynamicTrafficAI (4-point hull buoyancy) and PhysicsKernel (3D wave-crest drone clearance).
   */
  static evaluateOceanWaveField(x, z, timeSec, seaWaveHeightM, outObj = null) {
    const res = outObj || {
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
    const Hs = Math.max(0.0, Math.min(WeatherOpticsEngine.MAX_SEA_WAVE_M, Number(seaWaveHeightM) || 0.0));
    const drPerp = WeatherOpticsEngine._evalArchipelagoCoastRawJS(x, z);
    const dShore = drPerp - 3.52;
    res.dShore = dShore;

    const sm = WeatherOpticsEngine._smoothstep;
    // 1. Archipelago Headland Diffraction & Sheltering (Teluk U, Selat UI, Dermaga Piers, & Danau Kenanga Cove)
    const telukShelter =
      (1.0 - sm(-45.0, 35.0, x)) *
      sm(-275.0, -210.0, x) *
      sm(82.0, 140.0, z) *
      (1.0 - sm(285.0, 345.0, z));
    const straitMask =
      (1.0 - sm(-70.0, 10.0, x)) *
      sm(-345.0, -265.0, x) *
      sm(420.0, 465.0, z) *
      (1.0 - sm(595.0, 640.0, z));
    const dPierU = Math.hypot(x + 90.0, z - 444.0);
    const dPierI = Math.hypot(x + 90.0, z - 616.0);
    const pierLee = Math.max(1.0 - sm(8.0, 34.0, dPierU), 1.0 - sm(8.0, 34.0, dPierI));
    const coveMask =
      (1.0 - sm(65.0, 135.0, x)) * sm(20.0, 42.0, x) * (1.0 - sm(42.0, 82.0, Math.abs(z)));

    const Kd =
      (1.0 - 0.72 * telukShelter) *
      (1.0 - 0.54 * straitMask) *
      (1.0 - 0.48 * pierLee) *
      (1.0 - 0.42 * coveMask);
    res.diffractionKd = Kd;

    // 2. Shelf Bathymetry & Battjes-Janssen / Miche Depth-Limited Breaking Envelope (H_b <= 0.76 * h)
    const dPos = Math.max(0.0, dShore);
    const hBathym = Math.min(22.0, Math.max(0.24, 0.24 + 0.16 * dPos + 0.0042 * dPos * dPos));
    res.bathymetryH = hBathym;

    if (Hs <= 1e-4) {
      res.eta = 0.0;
      res.surgeX = 0.0;
      res.surgeZ = 0.0;
      res.nx = 0.0;
      res.ny = 1.0;
      res.nz = 0.0;
      res.breakingDiss = 0.0;
      return res;
    }

    const HbreakMax = 0.76 * hBathym;
    const HsLocal = Math.max(1e-4, Hs * Kd);
    const rawBreakScale = HbreakMax / Math.sqrt(HbreakMax * HbreakMax + HsLocal * HsLocal);
    const deepRestore = sm(36.0, 78.0, dShore);
    const surfBreakScale = rawBreakScale + (1.0 - rawBreakScale) * deepRestore;
    res.breakingDiss = Math.max(0.0, Math.min(1.0, (1.0 - surfBreakScale) * sm(0.05, 0.65, Hs)));

    if (!WeatherOpticsEngine._cpuAmpScratch8) {
      WeatherOpticsEngine._cpuAmpScratch8 = new Float32Array(8);
    }
    const amps = WeatherOpticsEngine.computeJonswap8Amplitudes(Hs, WeatherOpticsEngine._cpuAmpScratch8);
    const modes = WeatherOpticsEngine.OCEAN_WAVE_MODES;
    const t = Number(timeSec) || 0.0;

    let eta = 0.0;
    let surgeX = 0.0;
    let surgeZ = 0.0;
    let negGradX = 0.0;
    let negGradZ = 0.0;
    let jacobianY = 1.0;

    for (let i = 0; i < 8; i++) {
      const m = modes[i];
      const kh = m.k * hBathym;
      // Green's Law shoaling amplification peaking at intermediate depth kh ~ 0.85 before surf breaking
      const shoalBoost = 1.0 + 0.16 * Math.exp(-Math.pow((kh - 0.85) / 0.55, 2.0)) * (1.0 - deepRestore);
      const ai = amps[i] * Kd * shoalBoost * surfBreakScale;
      // Stationary nearshore shoaling phase integral (zero time-frequency Doppler shift!)
      const shoalPhase = (1.0 - sm(0.0, 38.0, dPos)) * m.k * 3.8;
      const ph = (m.dx * x + m.dz * z) * m.k + shoalPhase - m.w * t + m.phase0;
      const s = Math.sin(ph);
      const c = Math.cos(ph);
      // 2nd-Order Stokes crest sharpening + broad rounded troughs: +0.5 * k * a^2 * cos(2*ph)
      const stokes2 = 0.5 * m.k * ai * ai;
      const c2 = c * c - s * s;
      const s2 = 2.0 * s * c;

      eta += ai * c + stokes2 * c2;
      surgeX -= m.dx * (m.q * ai * s);
      surgeZ -= m.dz * (m.q * ai * s);
      const dEta = m.k * ai * s + 2.0 * m.k * stokes2 * s2;
      negGradX += m.dx * dEta;
      negGradZ += m.dz * dEta;
      jacobianY -= m.q * m.k * ai * c;
    }

    // 360° Refracted Nearshore Surf Rollers inside dShore < 42m
    const coastScale = sm(0.0, 0.22, Hs / 0.854);
    if (dShore < 42.0 && coastScale > 0.002) {
      const eC = 0.38;
      const gx =
        WeatherOpticsEngine._evalArchipelagoCoastRawJS(x + eC, z) -
        WeatherOpticsEngine._evalArchipelagoCoastRawJS(x - eC, z);
      const gz =
        WeatherOpticsEngine._evalArchipelagoCoastRawJS(x, z + eC) -
        WeatherOpticsEngine._evalArchipelagoCoastRawJS(x, z - eC);
      const gLen = Math.hypot(gx, gz) || 1.0;
      const cnx = gx / gLen;
      const cnz = gz / gLen;
      const refCx = z > 536.0 ? -160.0 : -155.0;
      const refCz = z > 536.0 ? 680.0 : 207.0;
      const thetaIsl = Math.atan2(z - refCz, x - refCx);

      const dClamp = Math.max(-1.8, dShore);
      const phiShoal = 0.36 * dClamp + 4.9 * Math.log(1.0 + Math.max(0.0, dClamp + 1.8) / 7.5);
      const kLocal = 0.36 + 4.9 / (7.5 + Math.max(0.0, dClamp + 1.8));
      const shoalEnv = Math.exp(-Math.max(0.0, dShore - 2.0) * 0.068) * (1.0 - sm(24.0, 42.0, dShore));

      const psi1 = phiShoal + 1.10 * t + 1.7 * Math.sin(thetaIsl - 0.5) + 0.65 * Math.cos(2.0 * thetaIsl + 0.8);
      const psi2 = phiShoal * 0.74 + 0.79 * t - 1.4 * Math.cos(thetaIsl + 0.4) + 0.55 * Math.sin(3.0 * thetaIsl);
      const wSurf1 = Math.cos(psi1) + 0.26 * Math.cos(2.0 * psi1);
      const dSurf1 = Math.sin(psi1) + 0.52 * Math.sin(2.0 * psi1);
      const wSurf2 = Math.cos(psi2) + 0.22 * Math.cos(2.0 * psi2);
      const dSurf2 = Math.sin(psi2) + 0.44 * Math.sin(2.0 * psi2);

      const sCoast = Math.min(Hs / 0.854, 1.35) * (0.45 + 0.55 * Kd);
      const aR1 = (0.036 + 0.095 * sCoast) * coastScale * shoalEnv;
      const aR2 = (0.024 + 0.062 * sCoast) * coastScale * shoalEnv;

      eta += aR1 * wSurf1 + aR2 * wSurf2;
      const rSurge = (aR1 * Math.sin(psi1) + aR2 * Math.sin(psi2)) * 0.42;
      surgeX -= cnx * rSurge;
      surgeZ -= cnz * rSurge;
      const rSlope = kLocal * (aR1 * dSurf1 + aR2 * 0.74 * dSurf2);
      negGradX += cnx * rSlope;
      negGradZ += cnz * rSlope;
    }

    const rCenter = Math.hypot(x + 155.0, z - 330.0);
    const horizonTaper = 1.0 - sm(1150.0, 1630.0, rCenter);

    res.eta = eta * horizonTaper;
    res.surgeX = surgeX * horizonTaper;
    res.surgeZ = surgeZ * horizonTaper;
    const nyRaw = Math.max(0.28, jacobianY);
    const nLen = Math.hypot(negGradX * horizonTaper, nyRaw, negGradZ * horizonTaper) || 1.0;
    res.nx = (negGradX * horizonTaper) / nLen;
    res.ny = nyRaw / nLen;
    res.nz = (negGradZ * horizonTaper) / nLen;
    return res;
  }

  constructor(scene, options = {}) {
    this.scene = scene;
    this._shadowMapSize = Math.max(1024, Math.min(4096, Number(options.shadowMapSize) || 4096));
    this.timeOfDay = 'DAY'; // 'DAWN' | 'MORNING' | 'DAY' | 'AFTERNOON' | 'DUSK' | 'NIGHT'
    this.weatherMode = 'CLEAR'; // 10 physical regimes in WeatherOpticsEngine.WEATHER_PROFILES
    this.weatherSpectrum = 0.0; // Live continuous [0, 1] weather spectrum coordinate
    this.targetWeatherSpectrum = 0.0; // Target [0, 1] weather spectrum coordinate
    this.manualSeaWaveHeightM = null; // null = weather-coupled auto, or [0.00 .. 3.00] m manual slider override
    this.seaWaveHeightM = 0.24; // Live significant ocean wave height H_s in meters [0.00 .. 3.00]
    this.currentWindAngleRad = 0.48;
    this._ampScratch8 = new Float32Array(8);

    this.transitionDurationSec = WeatherOpticsEngine.DEFAULT_TRANSITION_SEC;
    this.currentSolarHour = 11.85; // Continuous 24h Kampus UI Depok WIB solar clock [0, 24)
    this._sliderSolarHour = 11.85; // Linear [0.00 .. 24.00] slider coordinate (preserves 0->24 = 45s full sweep!)
    this.continuousSolarHour = 11.85; // Unwrapped monotonic WIB hours since _baseJD0WIB (0.000° midnight jump!)
    this._baseJD0WIB = computeBaseJD0WIB(new Date());
    this._astroState = null;
    this.nightIlluminationLevel = 0.0; // Continuous [0, 1] architectural & aviation night glow
    this.streetlampSensorLevel = 0.0; // Continuous [0, 1] ambient-lux photocell streetlamp power

    this.targetRainIntensity = 0.0;
    this.targetWetness = 0.0;
    this.targetCloudCover = 0.22;
    this.targetStormFactor = 0.0;
    this.targetFogDensity = 0.0;
    this.baseWindSpeed = 0.82;
    this.targetWindSpeed = 0.82;
    this.windGustMultiplier = 1.0;

    this._weatherTrans = {
      active: false,
      isSpectrumSweep: false,
      fromMode: 'CLEAR',
      toMode: 'CLEAR',
      startSpectrum: 0.0,
      targetSpectrum: 0.0,
      dir: 0,
      launchClock: 0.0,
      launchRampSec: 0.35,
      elapsed: 0.0,
      duration: WeatherOpticsEngine.DEFAULT_TRANSITION_SEC,
      progress: 1.0,
      startRain: 0.0,
      startWet: 0.0,
      startCloud: 0.22,
      startStorm: 0.0,
      startFog: 0.0,
      startWind: 0.82
    };

    this._timeTrans = {
      active: false,
      isProportionalSlider: false,
      fromTod: 'DAY',
      toTod: 'DAY',
      dir: 0,
      launchClock: 0.0,
      launchRampSec: 0.35,
      elapsed: 0.0,
      duration: WeatherOpticsEngine.DEFAULT_TRANSITION_SEC,
      progress: 1.0,
      startHour: 11.85,
      startSliderHour: 11.85,
      targetSliderHour: 11.85,
      startContinuousHour: 11.85,
      deltaHour: 0.0
    };

    this._seaTrans = {
      active: false,
      dir: 0,
      launchClock: 0.0,
      launchRampSec: 0.35,
      elapsed: 0.0,
      duration: 0.0,
      progress: 1.0,
      startWaveM: 0.24,
      targetWaveM: 0.24
    };

    // Multi-stroke plasma lightning state (active when uStormFactor > 0.60)
    this.nextLightningTimer = 2.4;
    this.lightningClock = -1.0;
    this.lightningStrokeProfile = null;
    this.onThunder = null; // Optional callback: (distanceMeters, intensity) => void

    this._baseSunPos = new THREE.Vector3(68, 96, 62);
    this._baseSunColor = new THREE.Color(0xfff5e4);
    this._baseSunIntensity = 2.35;
    this._baseHemiIntensity = 0.72;
    this._baseFillPos = new THREE.Vector3(-55, 38, -48);
    this._baseFillIntensity = 0.35;
    this._flashColor = new THREE.Color(0xc8deff);
    this._tmpSunColor = new THREE.Color();
    this._colDaySun = new THREE.Color(0xfff5e4);
    this._colDuskSun = new THREE.Color(0xff9c52);
    this._colNightMoon = new THREE.Color(0x8fb0e8);
    this._colStormTint = new THREE.Color(0x9aa9bc);
    this._colHemiSky = new THREE.Color();
    this._colHemiGnd = new THREE.Color();
    this._colFog = new THREE.Color();
    this._colTmpA = new THREE.Color();
    this._colTmpB = new THREE.Color();

    const fallbackCollisionTex = RainCollisionBake.createFallbackTexture();
    this.hvsOptics = new HumanEyeHDROptics();
    this._lastSharedSurfaceUniforms = null;
    this._bakedCollision = null;
    this._smoothedSolarDirectTrans = 1.0;
    this._solarTransInitialized = false;
    this._smoothedCamSunVis = 1.0;
    const hvsUniforms = HumanEyeHDROptics.createUniforms();

    const fallbackPlanarData = new Uint8Array([140, 178, 218, 255, 140, 178, 218, 255, 140, 178, 218, 255, 140, 178, 218, 255]);
    const fallbackPlanarTex = new THREE.DataTexture(fallbackPlanarData, 2, 2, THREE.RGBAFormat);
    fallbackPlanarTex.needsUpdate = true;

    this.uniforms = {
      uTime: { value: 0.0 },
      uCamPos: { value: new THREE.Vector3() },
      uSunDir: { value: new THREE.Vector3(0.52, 0.72, 0.45).normalize() },
      uTrueSunDir: { value: new THREE.Vector3(0.52, 0.72, 0.45).normalize() },
      uSolarDirectTrans: { value: 1.0 },
      uMoonDir: { value: new THREE.Vector3(-0.42, 0.62, -0.52).normalize() },
      uMoonPhaseSunDir: { value: new THREE.Vector3(0.42, -0.56, 0.52).normalize() },
      uMoonIllumFraction: { value: 0.85 },
      uWorldToEqMat: { value: new THREE.Matrix3() },
      uStarGridTex: { value: createRealStarCatalogGridTexture() },
      uStreetlampPower: { value: 0.0 },
      uSunColor: { value: new THREE.Color(0xfff5e4) },
      uNightFactor: { value: 0.0 },
      uDuskFactor: { value: 0.0 },
      uRainIntensity: { value: 0.0 },
      uWetness: { value: 0.0 },
      uCloudCover: { value: 0.22 },
      uStormFactor: { value: 0.0 },
      uFogFactor: { value: 0.0 },
      uLightningFlash: { value: 0.0 },
      uLightningDir: { value: new THREE.Vector3(-0.45, 0.34, -0.82).normalize() },
      uLightningSeed: { value: 17.0 },
      uWindVec: { value: new THREE.Vector3(0.65, 0.0, 0.35) },
      uWindPhase: { value: new THREE.Vector2(0.0, 0.0) },
      uSeaState: { value: 0.24 / WeatherOpticsEngine.SEA_STATE_NORM_DIV },
      uWaveAmpA: { value: new THREE.Vector4(0.0, 0.0, 0.0, 0.0) },
      uWaveAmpB: { value: new THREE.Vector4(0.0, 0.0, 0.0, 0.0) },
      uVesselWakes: {
        value: [
          new THREE.Vector4(0, 0, 0, 0),
          new THREE.Vector4(0, 0, 0, 0),
          new THREE.Vector4(0, 0, 0, 0),
          new THREE.Vector4(0, 0, 0, 0)
        ]
      },
      uFireflyIntensity: { value: 0.0 },
      uCollisionMap: { value: fallbackCollisionTex },
      uCollisionHalfExtent: { value: RainCollisionBake.DEFAULT_HALF_EXTENT },
      uCollisionBounds: {
        value: new THREE.Vector4(
          RainCollisionBake.DEFAULT_MIN_X,
          RainCollisionBake.DEFAULT_MIN_Z,
          RainCollisionBake.DEFAULT_MAX_X - RainCollisionBake.DEFAULT_MIN_X,
          RainCollisionBake.DEFAULT_MAX_Z - RainCollisionBake.DEFAULT_MIN_Z
        )
      },
      uCollisionHeightRange: {
        value: new THREE.Vector2(
          RainCollisionBake.HEIGHT_MIN_Y,
          RainCollisionBake.HEIGHT_MAX_Y - RainCollisionBake.HEIGHT_MIN_Y
        )
      },
      uCollisionReady: { value: 0.0 },
      uHvsParams: hvsUniforms.uHvsParams,
      uHvsGlare: hvsUniforms.uHvsGlare,
      uGenshinMode: hvsUniforms.uGenshinMode,
      uPlanarReflTex: { value: fallbackPlanarTex },
      uPlanarReflMatrix: { value: new THREE.Matrix4() },
      uPlanarReflActive: { value: 0.0 },
      uRenderingPlanarRefl: { value: 0.0 }
    };
    this._syncOceanSpectrumUniforms();

    this.hemiLight = null;
    this.sunLight = null;
    this.fillLight = null;
    this.skyMesh = null;
    this.rainAndDripMesh = null;
    this.splashAndMistMesh = null;
    this.fireflyPoints = null;
    this.volcanicPlumeMesh = null;

    this._dripAnchorAttr = null;
    this._dripLandingAttr = null;
    this._dripOutDirAttr = null;
    this._splashAnchorAttr = null;
    this._envMaps = null;

    this._initLighting();
    this._initAnalyticalSkyDome();
    this._initPhysicalRainAndDripMesh();
    this._initPhysicalSplashAndMistMesh();
    this._initGPUFireflyParticles();
    this._initVolcanicSmokePlumeMesh();
    this.setTimeOfDay('DAY', null, { immediate: true });
    this.setWeather('CLEAR', { immediate: true });
  }

  _syncOceanSpectrumUniforms() {
    const Hs = Math.max(0.0, Math.min(WeatherOpticsEngine.MAX_SEA_WAVE_M, Number(this.seaWaveHeightM) || 0.0));
    this.seaWaveHeightM = Hs;
    this.uniforms.uSeaState.value = Hs / WeatherOpticsEngine.SEA_STATE_NORM_DIV;
    const amps = WeatherOpticsEngine.computeJonswap8Amplitudes(Hs, this._ampScratch8);
    this.uniforms.uWaveAmpA.value.set(amps[0], amps[1], amps[2], amps[3]);
    this.uniforms.uWaveAmpB.value.set(amps[4], amps[5], amps[6], amps[7]);
  }

  /**
   * Updates one of the 4 live GPU 19.47-deg Lord Kelvin Ship Wake + Stern Propeller Wash slots (0 GC).
   * @param {number} slotIdx 0..3
   * @param {number} worldX
   * @param {number} worldZ
   * @param {number} yawRad
   * @param {number} signedSpeedAndClass >0 for wide catamaran/ship hull, <0 for sleek planing yacht/fishing hull, 0 to disable
   */
  setVesselWakeSlot(slotIdx, worldX, worldZ, yawRad, signedSpeedAndClass) {
    const arr = this.uniforms.uVesselWakes?.value;
    if (!arr || slotIdx < 0 || slotIdx >= arr.length) return;
    arr[slotIdx].set(worldX, worldZ, yawRad, signedSpeedAndClass);
  }

  _initLighting() {
    this.hemiLight = new THREE.HemisphereLight(0xdbeeff, 0x5c6358, 0.78);
    this.hemiLight.position.set(0, 80, 0);
    this.scene.add(this.hemiLight);

    this.sunLight = new THREE.DirectionalLight(0xfff5e4, 2.45);
    this.sunLight.position.set(68, 96, 62);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.width = this._shadowMapSize || 4096;
    this.sunLight.shadow.mapSize.height = this._shadowMapSize || 4096;
    this.sunLight.shadow.camera.near = 10;
    this.sunLight.shadow.camera.far = 260;
    const d = 104;
    this.sunLight.shadow.camera.left = -d;
    this.sunLight.shadow.camera.right = d;
    this.sunLight.shadow.camera.top = d;
    this.sunLight.shadow.camera.bottom = -d;
    this.sunLight.shadow.bias = -0.00018;
    this.sunLight.shadow.normalBias = 0.016;
    this.sunLight.shadow.radius = 1.85;
    this.scene.add(this.sunLight);
    if (this.sunLight.target) {
      this.scene.add(this.sunLight.target);
    }

    this.fillLight = new THREE.DirectionalLight(0x9ec5eb, 0.42);
    this.fillLight.position.set(-55, 38, -48);
    this.scene.add(this.fillLight);
  }

  /**
   * NEXUS v2.6 ULTRA HDR IMAGE-BASED LIGHTING (PMREM):
   * Pre-convolves 3 HDR Physical Sky + Campus Bounce PMREM Cubemap Environment Textures
   * (DAY, DUSK, NIGHT) once at startup (< 8ms, 0 extra runtime Draw Calls!), activating Three.js r170
   * Split-Sum GGX Specular IBL + Kulla-Conty Multiscattering Energy Compensation across all PBR surfaces.
   */
  attachRenderer(renderer) {
    if (!renderer || typeof THREE.PMREMGenerator !== 'function') return;
    try {
      const pmrem = new THREE.PMREMGenerator(renderer);
      if (typeof pmrem.compileCubemapShader === 'function') {
        pmrem.compileCubemapShader();
      }

      const envScene = new THREE.Scene();
      const envGeo = new THREE.SphereGeometry(24, 32, 18);
      const envUniforms = {
        uZenith: { value: new THREE.Color(0x1f5ec4) },
        uHorizon: { value: new THREE.Color(0x9ec6ea) },
        uGround: { value: new THREE.Color(0x3a4834) },
        uTerracottaBounce: { value: new THREE.Color(0x8e4c3e) },
        uSunDir: { value: new THREE.Vector3(0.52, 0.72, 0.45).normalize() },
        uSunRadiance: { value: new THREE.Color(8.5, 7.8, 6.6) }
      };
      const envMat = new THREE.ShaderMaterial({
        side: THREE.BackSide,
        depthWrite: false,
        toneMapped: false,
        uniforms: envUniforms,
        vertexShader: /* glsl */ `
          varying vec3 vDir;
          void main() {
            vDir = normalize(position);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: /* glsl */ `
          varying vec3 vDir;
          uniform vec3 uZenith;
          uniform vec3 uHorizon;
          uniform vec3 uGround;
          uniform vec3 uTerracottaBounce;
          uniform vec3 uSunDir;
          uniform vec3 uSunRadiance;
          void main() {
            vec3 d = normalize(vDir);
            vec3 col;
            if (d.y >= 0.0) {
              col = mix(uHorizon, uZenith, pow(clamp(d.y, 0.0, 1.0), 0.56));
              float cosS = max(dot(d, normalize(uSunDir)), 0.0);
              float mieHalo = pow(cosS, 14.0) * 0.28 + pow(cosS, 140.0) * 1.65;
              col += uSunRadiance * mieHalo;
            } else {
              float down = clamp(-d.y, 0.0, 1.0);
              vec3 bounce = mix(uTerracottaBounce, uGround, smoothstep(0.0, 0.45, down));
              col = mix(uHorizon * 0.62, bounce, smoothstep(0.0, 0.28, down));
            }
            gl_FragColor = vec4(col, 1.0);
          }
        `
      });
      const envMesh = new THREE.Mesh(envGeo, envMat);
      envScene.add(envMesh);

      const bakeRegime = (zen, hor, gnd, bounce, sDir, sRad) => {
        envUniforms.uZenith.value.setHex(zen);
        envUniforms.uHorizon.value.setHex(hor);
        envUniforms.uGround.value.setHex(gnd);
        envUniforms.uTerracottaBounce.value.setHex(bounce);
        envUniforms.uSunDir.value.copy(sDir).normalize();
        envUniforms.uSunRadiance.value.setRGB(sRad[0], sRad[1], sRad[2]);
        return pmrem.fromScene(envScene, 0.025).texture;
      };

      // 3 Pre-Convolved Physical Sky + Campus-Bounce PMREM Regimes (DAY, DUSK, STORM_NIGHT — 0 runtime baking!)
      const dayTex = bakeRegime(
        0x2e5fa8,
        0xa6c6e2,
        0x34422e,
        0x7d4c3c,
        new THREE.Vector3(0.24, 0.86, 0.44),
        [4.2, 3.9, 3.4]
      );
      const duskTex = bakeRegime(
        0x26325e,
        0xd87d46,
        0x282924,
        0x703c2a,
        new THREE.Vector3(-0.78, 0.22, 0.58),
        [4.6, 2.25, 0.96]
      );
      const stormNightTex = bakeRegime(
        0x1c283a,
        0x4a596c,
        0x1d2321,
        0x2e2726,
        new THREE.Vector3(-0.35, 0.65, -0.52),
        [1.05, 1.22, 1.48]
      );
      this._envMaps = {
        BROADBAND: dayTex,
        DAY: dayTex,
        DUSK: duskTex,
        STORM_NIGHT: stormNightTex
      };
      this.scene.environment = dayTex;

      envGeo.dispose();
      envMat.dispose();
      pmrem.dispose();

      this._applyEnvironmentIBL(0.0);
    } catch (_err) {
      this._envMaps = null;
    }
  }

  _applyEnvironmentIBL(lightningFlash = 0.0) {
    if (!this._envMaps || !this.scene) return;
    const nightF = this.uniforms.uNightFactor.value;
    const duskF = this.uniforms.uDuskFactor.value;
    const cloudF = this.uniforms.uCloudCover.value;
    const rainF = this.uniforms.uRainIntensity.value;
    const stormF = this.uniforms.uStormFactor.value;

    let targetEnv = this._envMaps.DAY;
    if (nightF > 0.56 || stormF > 0.64 || rainF > 0.68) {
      targetEnv = this._envMaps.STORM_NIGHT || this._envMaps.DAY;
    } else if (duskF > 0.42) {
      targetEnv = this._envMaps.DUSK || this._envMaps.DAY;
    }
    if (this.scene.environment !== targetEnv) {
      this.scene.environment = targetEnv;
    }

    const overcastDim =
      1.0 -
      Math.min(
        0.48,
        cloudF * 0.12 + rainF * 0.22 + stormF * 0.16
      );
    const baseEnvInt =
      0.96 * (1.0 - duskF * 0.16) * (1.0 - nightF) + 0.24 * nightF;
    if ('environmentIntensity' in this.scene) {
      this.scene.environmentIntensity = baseEnvInt * overcastDim + lightningFlash * 0.68;
    }
  }

  /**
   * Bakes the 512x512 3D world collision height+normal+material field from scene roots
   * and binds the extracted 3D roof-eave & tree-canopy drip coordinates directly to GPU buffers.
   */
  bakeSceneCollisionField(roots, options = {}, sharedSurfaceUniforms = null) {
    const bake = RainCollisionBake.bakeFromHierarchy(roots, {
      ...options,
      maxDrips: WeatherOpticsEngine.EAVE_DRIP_COUNT
    });
    return this._applyCollisionBakeResult(bake, sharedSurfaceUniforms);
  }

  /**
   * Time-sliced asynchronous 512x512 3D Rain Collision Z-Buffer & Drip Anchor baker (< 6ms per frame slice).
   * Streams reference collision data in the background while the 3D world runs interactively at 60-120 FPS!
   */
  async bakeSceneCollisionFieldAsync(roots, options = {}, sharedSurfaceUniforms = null) {
    const budgetMs =
      options && options.sliceBudgetMs !== undefined ? Number(options.sliceBudgetMs) : 1.5;
    const bake = await RainCollisionBake.bakeFromHierarchyAsync(
      roots,
      {
        ...options,
        maxDrips: WeatherOpticsEngine.EAVE_DRIP_COUNT
      },
      budgetMs
    );
    return this._applyCollisionBakeResult(bake, sharedSurfaceUniforms);
  }

  _applyCollisionBakeResult(bake, sharedSurfaceUniforms = null) {
    this._bakedCollision = bake;
    this.collisionBake = bake;
    this.uniforms.uCollisionMap.value = bake.texture;
    this.uniforms.uCollisionReady.value = 1.0;
    this.uniforms.uCollisionHalfExtent.value = bake.halfExtent;
    const minX = bake.minX !== undefined ? bake.minX : -bake.halfExtent;
    const minZ = bake.minZ !== undefined ? bake.minZ : -bake.halfExtent;
    const spanX = bake.spanX !== undefined ? bake.spanX : bake.halfExtent * 2.0;
    const spanZ = bake.spanZ !== undefined ? bake.spanZ : bake.halfExtent * 2.0;
    const minY = bake.minY !== undefined ? bake.minY : RainCollisionBake.HEIGHT_MIN_Y;
    const spanY =
      bake.spanY !== undefined
        ? bake.spanY
        : RainCollisionBake.HEIGHT_MAX_Y - RainCollisionBake.HEIGHT_MIN_Y;
    this.uniforms.uCollisionBounds.value.set(minX, minZ, spanX, spanZ);
    this.uniforms.uCollisionHeightRange.value.set(minY, spanY);

    if (sharedSurfaceUniforms) {
      if (sharedSurfaceUniforms.uCollisionMap) {
        sharedSurfaceUniforms.uCollisionMap.value = bake.texture;
      }
      if (sharedSurfaceUniforms.uCollisionBounds) {
        sharedSurfaceUniforms.uCollisionBounds.value.set(minX, minZ, spanX, spanZ);
      }
      if (sharedSurfaceUniforms.uCollisionHeightRange) {
        sharedSurfaceUniforms.uCollisionHeightRange.value.set(minY, spanY);
      }
      if (sharedSurfaceUniforms.uCollisionReady) {
        sharedSurfaceUniforms.uCollisionReady.value = 1.0;
      }
    }

    // Bind exact 3D roof-eave & tree-canopy overhang coordinates + outward eave pitch direction to the 2,000 drip/runoff quads
    if (this._dripAnchorAttr && this._dripLandingAttr && bake.dripAnchors.length > 0) {
      const skyVerts = WeatherOpticsEngine.SKY_DROP_COUNT * 4;
      const anchorArr = this._dripAnchorAttr.array;
      const landingArr = this._dripLandingAttr.array;
      const outDirArr = this._dripOutDirAttr ? this._dripOutDirAttr.array : null;
      const anchors = bake.dripAnchors;

      for (let i = 0; i < WeatherOpticsEngine.EAVE_DRIP_COUNT; i++) {
        const a = anchors[i % anchors.length];
        const ox = Number.isFinite(a.outX) ? a.outX : 0.0;
        const oz = Number.isFinite(a.outZ) ? a.outZ : 1.0;
        for (let c = 0; c < 4; c++) {
          const vi = skyVerts + i * 4 + c;
          anchorArr[vi * 3] = a.x;
          anchorArr[vi * 3 + 1] = a.y;
          anchorArr[vi * 3 + 2] = a.z;
          landingArr[vi * 2] = a.landingY;
          landingArr[vi * 2 + 1] = a.isCanopy;
          if (outDirArr) {
            outDirArr[vi * 2] = ox;
            outDirArr[vi * 2 + 1] = oz;
          }
        }
      }
      this._dripAnchorAttr.needsUpdate = true;
      this._dripLandingAttr.needsUpdate = true;
      if (this._dripOutDirAttr) this._dripOutDirAttr.needsUpdate = true;

      // Also anchor 600 splash crowns directly along the 3D eave-drip & parabolic genting-runoff landing lines!
      if (this._splashAnchorAttr) {
        const spArr = this._splashAnchorAttr.array;
        const eaveSplashStart = 2000; // Last 600 of the 2,600 crown splashes
        for (let i = 0; i < 600; i++) {
          const a = anchors[(i * 3) % anchors.length];
          const ox = Number.isFinite(a.outX) ? a.outX : 0.0;
          const oz = Number.isFinite(a.outZ) ? a.outZ : 0.0;
          const outThrow = a.isCanopy > 0.5 ? 0.0 : (i % 2 === 0 ? 0.42 : 0.05);
          const qBase = (eaveSplashStart + i) * 4;
          for (let c = 0; c < 4; c++) {
            const vi = qBase + c;
            spArr[vi * 3] = a.x + ox * outThrow;
            spArr[vi * 3 + 1] = a.z + oz * outThrow;
            spArr[vi * 3 + 2] = 1.0; // Flag: world-pinned eave-drip / runoff landing splash!
          }
        }
        this._splashAnchorAttr.needsUpdate = true;
      }
    }

    return bake;
  }

  /**
   * Shared Dual-Island Kepulauan UI ("U" & "I") Analytical SDF GLSL Kernel
   * Synchronized 1:1 with TerrainDEMLoader.getIslandUCoastDistance & getIslandICoastDistance.
   */
  static _getArchipelagoSdfGLSL() {
    return /* glsl */ `
      const vec2 ISLAND_CENTER = vec2(-18.0, 34.0);
      const vec2 ARCHIPELAGO_CENTER = vec2(-155.0, 330.0);

      float sminArch(float a, float b, float k) {
        float h = max(0.0, k - abs(a - b)) / k;
        return min(a, b) - h * h * k * 0.25;
      }

      // Exact 360-degree closed polar campus-head shoreline R(theta) synchronized 1:1 with TerrainDEMLoader.getIslandRadiusAtAngle
      float islandRadiusAtAngle(float theta) {
        float eastCove = -46.0 * pow(0.5 + 0.5 * cos(theta + 0.25), 4.4);
        float neHeadland = 23.5 * pow(0.5 + 0.5 * cos(theta + 1.04), 8.5);
        float southRotundaLobe = 15.5 * pow(0.5 + 0.5 * cos(theta - 1.12), 2.3);
        float swHelipadPeninsula = 20.5 * pow(0.5 + 0.5 * cos(theta - 2.44), 6.5);
        float organicMeander =
          2.15 * sin(3.0 * theta + 0.40) +
          1.20 * cos(5.0 * theta - 0.70) +
          0.55 * sin(8.0 * theta + 1.10);
        return 121.0 + eastCove + neHeadland + southRotundaLobe + swHelipadPeninsula + organicMeander;
      }

      // Pulau "U" Horseshoe Signed Coast Distance dU(x, z)
      float evalPulauUCoast(vec2 xz) {
        float distUCenter;
        if (xz.x <= -145.0) {
          distUCenter = abs(length(xz - vec2(-145.0, 207.0)) - 185.0);
        } else {
          float dLegL = length(vec2(max(0.0, xz.x + 62.0), xz.y - 22.0));
          float dLegR = length(vec2(max(0.0, xz.x - 10.0), xz.y - 392.0));
          distUCenter = min(dLegL, dLegR);
        }
        float meanderU = 2.0 * sin(xz.x * 0.035 + xz.y * 0.025) + 1.1 * cos(xz.y * 0.048 - xz.x * 0.031);
        float dUHorseshoe = distUCenter - 74.0 - meanderU;

        vec2 d = xz - ISLAND_CENTER;
        if (xz.y > 250.0 || xz.x < -195.0) {
          return dUHorseshoe;
        }
        float r = length(d) + 1e-4;
        float theta = atan(d.y, d.x);
        float r0 = islandRadiusAtAngle(theta);
        float eps = 0.008;
        float dRdTh = (islandRadiusAtAngle(theta + eps) - islandRadiusAtAngle(theta - eps)) / (2.0 * eps);
        float slopeRatio = dRdTh / max(20.0, r0);
        float dCampusHead = (r - r0) * inversesqrt(1.0 + slopeRatio * slopeRatio);
        return sminArch(dCampusHead, dUHorseshoe, 28.0);
      }

      // Pulau "I" Volcano & Runway Island Signed Coast Distance dI(x, z)
      // Synchronized 1:1 with TerrainDEMLoader: I_CENTER=(-160, 694), I_HALF=(244, 88), I_CORNER_R=56
      float evalPulauICoast(vec2 xz) {
        vec2 q = abs(xz - vec2(-160.0, 694.0)) - vec2(188.0, 32.0);
        float outsideDist = length(max(q, vec2(0.0)));
        float insideDist = min(max(q.x, q.y), 0.0);
        float meanderI = 1.9 * sin(xz.x * 0.039 - xz.y * 0.027) + 1.05 * cos(xz.x * 0.061 + xz.y * 0.033);
        return outsideDist + insideDist - 56.0 - meanderI;
      }

      float evalArchipelagoCoastRaw(vec2 xz) {
        return min(evalPulauUCoast(xz), evalPulauICoast(xz));
      }

      // Analytical terrain + volcano + runway elevation outside the central 320m x 320m Rectorat bake texture
      float evalArchipelagoAnalyticalHeight(vec2 xz, float dCoast) {
        if (dCoast > 3.52) return -1.45;
        if (dCoast > 0.0) return -1.45 * smoothstep(0.0, 3.6, dCoast);
        float coastCollar = smoothstep(-1.2, -16.5, dCoast);
        if (xz.y > 536.0) {
          // Pulau I: ICAO Runway RWY 36/18 + West Apron + Active Stratovolcano at (-32, 698)
          if (xz.x > -383.0 && xz.x < -127.0 && abs(xz.y - 708.0) < 12.6) {
            return 0.58;
          }
          if (xz.x > -285.0 && xz.x < -223.0 && xz.y > 659.5 && xz.y < 697.5) {
            return 0.58;
          }
          float hBase = 0.44 * smoothstep(-1.2, -11.0, dCoast);
          vec2 dv = xz - vec2(-32.0, 698.0);
          float rV = length(dv);
          if (rV < 118.0) {
            float u = max(0.0, (98.0 - rV) / 78.5);
            float cone = 58.0 * pow(min(1.0, u), 1.38) * smoothstep(0.0, 0.14, u);
            float crater = -14.2 * smoothstep(19.5, 5.2, rV);
            float southClear = smoothstep(-126.0, -96.0, xz.x);
            return hBase + max(0.0, (cone + crater) * southClear * coastCollar);
          }
          return hBase;
        }
        // Pulau U Rolling Hills outside central Rectorat lowland
        float flatGuard = smoothstep(125.0, 185.0, length(xz - vec2(-12.0, 48.0)));
        float hHills =
          18.6 * exp(-dot(xz - vec2(-336.0, 236.0), xz - vec2(-336.0, 236.0)) / 9200.0) +
          16.8 * exp(-dot(xz - vec2(-68.0, 392.0), xz - vec2(-68.0, 392.0)) / 7800.0) +
          14.6 * exp(-dot(xz - vec2(-228.0, 384.0), xz - vec2(-228.0, 384.0)) / 6800.0) +
          14.2 * exp(-dot(xz - vec2(-248.0, 24.0), xz - vec2(-248.0, 24.0)) / 5900.0);
        return (0.45 + hHills * flatGuard) * coastCollar;
      }
    `;
  }

  /**
   * Shared Analytical Spherical-Earth Java Sea Radiance Kernel (R_E = 6,371,000 m):
   * Evaluated identically by both oceanMesh (at r = 2,020..2,680m outer rim) and skyMesh
   * (from r = 2,680m out to the spherical horizon dip angle -sqrt(2 * H_cam / R_E)),
   * guaranteeing a 0.000-RGB-step boundless ocean horizon from 1.5m eye level to 2,000m altitude!
   */
  static _getSphericalFarSeaGLSL() {
    return /* glsl */ `
      float farSeaHash21(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
      }

      float farSeaNoise2(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(
          mix(farSeaHash21(i), farSeaHash21(i + vec2(1.0, 0.0)), u.x),
          mix(farSeaHash21(i + vec2(0.0, 1.0)), farSeaHash21(i + vec2(1.0, 1.0)), u.x),
          u.y
        );
      }

      float farSeaFbm3(vec2 p) {
        const mat2 R_SEA = mat2(0.80, 0.60, -0.60, 0.80);
        float s = 0.0;
        float a = 0.54;
        for (int i = 0; i < 3; i++) {
          s += a * farSeaNoise2(p);
          p = R_SEA * p * 2.07 + vec2(17.1, 31.7);
          a *= 0.48;
        }
        return s;
      }

      vec3 evalSphericalFarSeaHDR(
        vec3 dir,
        float Hcam,
        float dSeaM,
        vec3 activeSunDir,
        vec3 trueSunDir,
        vec3 moonDir,
        float duskFactor,
        float nightFactor,
        float cloudCover,
        float rainIntensity,
        float stormFactor,
        float fogFactor,
        float lightningFlash,
        vec2 windPhase,
        float solarDirectTrans
      ) {
        vec2 dirXZ = normalize(dir.xz + vec2(1e-5, 0.0));
        vec2 sunXZ = normalize(trueSunDir.xz + vec2(1e-5, 0.0));
        float cosSunAz = dot(dirXZ, sunXZ);

        // Local incidence angle on curved Earth ocean surface (accounts for Earth tangent tilt dSeaM / R_E)
        float nDotVSea = clamp(-dir.y - dSeaM / 6371000.0, 0.0015, 1.0);
        float roughSpread = 0.10 + 0.14 * clamp(stormFactor + rainIntensity * 0.5, 0.0, 1.0);
        float fresnelExpS = mix(4.6, 2.7, roughSpread);
        float fresSea = clamp(0.036 + 0.88 * pow(1.0 - nDotVSea, fresnelExpS), 0.045, 0.88);

        // Azimuth-matched horizon & zenith sky reflection (unified via nexusEvalHorizonSkySRGB -> 0.000 horizon seam!)
        float lunarSeaScatter = clamp(moonDir.y, 0.0, 1.0) * 0.55 * (1.0 - clamp(cloudCover * 0.5 + rainIntensity * 0.5, 0.0, 1.0));
        vec3 sZen = mix(
          mix(vec3(0.13, 0.37, 0.71), vec3(0.11, 0.16, 0.38), duskFactor),
          vec3(0.056, 0.086, 0.145) + vec3(0.038, 0.058, 0.098) * lunarSeaScatter,
          nightFactor
        );
        vec3 sHor = nexusEvalHorizonSkySRGB(
          vec3(dir.x, 0.0, dir.z),
          trueSunDir,
          duskFactor,
          nightFactor,
          cloudCover,
          rainIntensity,
          stormFactor,
          fogFactor,
          lunarSeaScatter
        );
        float reflEl = clamp(nDotVSea * 0.85 + 0.052, 0.02, 1.0);
        vec3 seaSkyRefl = mix(sHor, sZen, pow(reflEl, 0.58));

        // Reflected tropical cumulus bands advected monotonically via uWindPhase
        vec2 cUV = (dir.xz / (reflEl + 0.26)) * 1.38 + windPhase * 0.013;
        float cVal = farSeaFbm3(cUV);
        float cThr = mix(0.56, 0.22, clamp(cloudCover, 0.0, 1.0));
        float cMsk = smoothstep(cThr, cThr + 0.34, cVal) * smoothstep(0.015, 0.20, reflEl);
        vec3 cReflCol = mix(
          mix(vec3(0.92, 0.95, 0.99), vec3(0.95, 0.56, 0.32), duskFactor),
          vec3(0.075, 0.105, 0.168),
          nightFactor
        );
        float stormGrey = clamp(rainIntensity * 0.65 + stormFactor * 0.35, 0.0, 0.90);
        cReflCol = mix(cReflCol, vec3(0.24, 0.28, 0.34) * mix(1.0, 0.24, nightFactor), stormGrey);
        seaSkyRefl = mix(seaSkyRefl, cReflCol, cMsk * 0.72);
        seaSkyRefl = mix(seaSkyRefl, vec3(0.25, 0.29, 0.35) * mix(1.0, 0.22, nightFactor), stormGrey * 0.92);
        seaSkyRefl += vec3(0.76, 0.88, 1.0) * lightningFlash * 1.10;

        // Deep Pelagic Jerlov IB Java Sea volume + subtle wind-swell mottling before horizon extinction
        float overcastDesatS = clamp(rainIntensity * 0.65 + stormFactor * 0.25, 0.0, 1.0);
        float gModeFar = clamp(uGenshinMode, 0.0, 1.0);
        vec3 pelagicDayPBR = mix(vec3(0.015, 0.064, 0.118), vec3(0.011, 0.038, 0.075), overcastDesatS);
        vec3 pelagicDayGenshin = mix(vec3(0.022, 0.125, 0.265), vec3(0.016, 0.075, 0.165), overcastDesatS);
        vec3 pelagicDay = mix(pelagicDayPBR, pelagicDayGenshin, gModeFar * 0.85);
        vec3 pelagicCol = mix(pelagicDay, mix(vec3(0.010, 0.020, 0.042), vec3(0.016, 0.045, 0.105), gModeFar * 0.75), nightFactor);
        vec2 worldSeaXZ = dirXZ * dSeaM;
        float swellTex = (farSeaFbm3(worldSeaXZ * 0.0045 - windPhase * 0.025) - 0.5)
                       * (1.0 - smoothstep(3500.0, 24000.0, dSeaM));
        pelagicCol *= (1.0 + swellTex * 0.18) * mix(mix(1.16, 0.88, duskFactor), 0.76, nightFactor);

        vec3 farColSRGB = mix(pelagicCol, seaSkyRefl, fresSea);
        vec3 hdrSea = nexusSRGBToLinear(max(farColSRGB, vec3(0.0))) * mix(1.16, 0.76, nightFactor);

        // Physically occluded wind-sheared far-sea solar / lunar horizon shimmer
        float sunAboveS = smoothstep(-0.024, 0.026, trueSunDir.y) * clamp(solarDirectTrans, 0.0, 1.0);
        vec3 viewToCam = -dir;
        vec3 hSunS = normalize(trueSunDir + viewToCam);
        float specSunS = pow(max(hSunS.y, 0.0), 42.0) * smoothstep(0.15, 0.96, cosSunAz) * sunAboveS;
        float moonAboveS = smoothstep(-0.018, 0.035, moonDir.y) * nightFactor * (1.0 - stormGrey);
        vec3 hMoonS = normalize(moonDir + viewToCam);
        float specMoonS = pow(max(hMoonS.y, 0.0), 48.0)
                        * smoothstep(0.20, 0.96, dot(dirXZ, normalize(moonDir.xz + vec2(1e-5, 0.0))))
                        * moonAboveS * 0.12;
        float specDistAtten = 1.0 - smoothstep(9000.0, 56000.0, dSeaM);
        vec3 glitColS = nexusSRGBToLinear(
          mix(mix(vec3(1.0, 0.94, 0.80), vec3(1.0, 0.58, 0.26), duskFactor), vec3(0.45, 0.56, 0.78), nightFactor)
        );
        hdrSea += glitColS * (specSunS * 0.24 + specMoonS) * specDistAtten;
        hdrSea += vec3(0.75, 0.95, 1.45) * lightningFlash * 0.55;

        // Exact Boundary-Layer Aerial Perspective matching oceanMesh at r = 2,660m + unified horizon merger
        float rayDist3D = dSeaM / max(0.08, length(dir.xz));
        vec3 virtCam = vec3(0.0, Hcam - 1.45, 0.0);
        vec3 virtPos = virtCam + dir * rayDist3D;
        vec3 hazeSunDir = normalize(mix(trueSunDir, activeSunDir, smoothstep(0.65, 0.96, nightFactor)));
        hdrSea = nexusApplyBoundaryLayerAerialPerspective(
          hdrSea,
          virtCam,
          virtPos,
          hazeSunDir,
          duskFactor,
          nightFactor,
          cloudCover,
          rainIntensity,
          lightningFlash,
          fogFactor
        );

        // Long-range spherical marine tropospheric extinction (4.2 km -> 46 km seamless horizon merger)
        float longMarineMask = smoothstep(4200.0, 46000.0, dSeaM);
        if (longMarineMask > 0.001) {
          vec3 hdrHorMarine = nexusSRGBToLinear(max(sHor, vec3(0.0))) * mix(1.16, 0.76, nightFactor);
          hdrSea = mix(hdrSea, hdrHorMarine, longMarineMask * 0.92);
        }
        return hdrSea;
      }
    `;
  }

  /**
   * Physical 3D Ray-World Solar/Lunar Occlusion Kernel for Kepulauan UI Water Surfaces:
   * Combines exact O(1) 3D solid intersection (with physical 0.53° solar limb penumbra) against
   * Gedung Rektorat UI (Cruciform Tower + Steep Tajug Pyramid + Annexes + Rotunda Gatehouse),
   * Kompleks DPPD-STP UI, Jembatan Teksas UI (hollow arched deck + A-frame pylons), and Gunung Gede Bogor,
   * PLUS a uniform-flow 8-tap uCollisionMap & analytical terrain raymarch for trees, hills, & shorelines!
   */
  static _getWaterSolarOcclusionGLSL() {
    return /* glsl */ `
      float evalSoftBoxSolarOcc(vec3 ro, vec3 invRd, vec3 bMin, vec3 bMax) {
        vec3 t0 = (bMin - ro) * invRd;
        vec3 t1 = (bMax - ro) * invRd;
        vec3 tSm = min(t0, t1);
        vec3 tBg = max(t0, t1);
        float tNear = max(max(tSm.x, tSm.y), max(tSm.z, 0.25));
        float tFar = min(min(tBg.x, tBg.y), tBg.z);
        float chord = tFar - tNear;
        return smoothstep(0.0, 1.25 + 0.028 * max(0.0, tNear), chord);
      }

      float evalWaterCelestialVisibility3D(vec3 wp, vec3 rayDir) {
        // 1. Horizon limb cutoff (solar/lunar angular radius ~ 0.024 rad)
        float horVis = smoothstep(-0.024, 0.024, rayDir.y);
        if (horVis <= 0.001) return 0.0;

        vec3 safeRd = vec3(
          abs(rayDir.x) > 1e-5 ? rayDir.x : (rayDir.x >= 0.0 ? 1e-5 : -1e-5),
          abs(rayDir.y) > 1e-5 ? rayDir.y : 1e-5,
          abs(rayDir.z) > 1e-5 ? rayDir.z : (rayDir.z >= 0.0 ? 1e-5 : -1e-5)
        );
        vec3 invRd = 1.0 / safeRd;
        float occ = 0.0;

        // Exact O(1) Archipelago Bounding-Cylinder Early-Out (C = (-145, 345), R = 575m):
        // Over 90% of open Java Sea pixels either point away from Kepulauan UI or miss its 575m cylinder completely!
        float rdXZ2 = max(1e-4, dot(rayDir.xz, rayDir.xz));
        vec2 relArch = wp.xz - vec2(-145.0, 345.0);
        float r2Arch = dot(relArch, relArch);
        if (r2Arch > 330625.0) {
          float bArch = dot(relArch, rayDir.xz);
          if (bArch >= 0.0) return horVis;
          float d2Arch = r2Arch - (bArch * bArch) / rdXZ2;
          if (d2Arch > 330625.0) return horVis;
          float tEnterArch = max(0.0, (-bArch / sqrt(rdXZ2)) - 575.0);
          if (wp.y + rayDir.y * tEnterArch > 60.0) return horVis;
        }

        // 2A. Gedung Pusat Administrasi / Rektorat UI Depok (x = 0, z = 0)
        //     Cruciform 8-Story Tower + Steep 3-Tier Javanese Tajug Spire + Courtyard Podium & Annex Wings
        occ = max(occ, evalSoftBoxSolarOcc(wp, invRd, vec3(-18.8, -0.5, -18.8), vec3(18.8, 33.4, 18.8)));
        occ = max(occ, evalSoftBoxSolarOcc(wp, invRd, vec3(-54.2, -0.5, -20.0), vec3(-20.0, 12.5, 16.8)));
        occ = max(occ, evalSoftBoxSolarOcc(wp, invRd, vec3( 20.0, -0.5, -22.8), vec3( 54.2, 12.5, 13.8)));
        occ = max(occ, evalSoftBoxSolarOcc(wp, invRd, vec3(-36.2, -0.5, -35.2), vec3( 36.2,  9.4, 39.8)));
        occ = max(occ, evalSoftBoxSolarOcc(wp, invRd, vec3(-20.6, -0.5,  39.5), vec3( 15.6,  7.6, 72.0)));
        occ = max(occ, evalSoftBoxSolarOcc(wp, invRd, vec3( -3.8, -0.5, -52.2), vec3( 13.8,  8.8,-35.0)));

        // Exact Steep Pyramidal Tajug Roof Silhouette (y: 32.2m -> 50.2m tapering from 19.6m -> 0.55m)
        float tTajug = -dot(wp.xz, rayDir.xz) / rdXZ2;
        if (tTajug > 0.3) {
          vec3 pT = wp + rayDir * tTajug;
          if (pT.y >= 31.5 && pT.y <= 50.6) {
            float uPyr = clamp((pT.y - 32.4) / 17.6, 0.0, 1.0);
            float halfW = mix(19.6, 0.55, uPyr);
            float penT = 0.35 + 0.015 * tTajug;
            float cheb = max(abs(pT.x), abs(pT.z));
            float hMask = smoothstep(31.5, 32.6, pT.y) * (1.0 - smoothstep(49.6, 50.6, pT.y));
            occ = max(occ, (1.0 - smoothstep(halfW - penT, halfW + penT, cheb)) * hMask);
          }
        }

        // 2B. Kompleks Gedung DPPD & STP UI (-298, 224)
        vec2 relDPPD = wp.xz - vec2(-298.0, 224.0);
        float tDPPD = -dot(relDPPD, rayDir.xz) / rdXZ2;
        if (tDPPD > 0.3) {
          vec3 pD = wp + rayDir * tDPPD;
          if (pD.y >= -0.5 && pD.y <= 26.8) {
            float rD = length(pD.xz - vec2(-298.0, 224.0));
            float penD = 0.50 + 0.016 * tDPPD;
            float topFadeD = 1.0 - smoothstep(24.8, 26.8, pD.y);
            occ = max(occ, (1.0 - smoothstep(38.5 - penD, 38.5 + penD, rD)) * topFadeD);
          }
        }

        // 2C. Jembatan Teksas UI (x = -28, z in [168, 322]): Hollow Arched Deck + Twin A-Frame Pylons
        occ = max(occ, evalSoftBoxSolarOcc(wp, invRd, vec3(-31.2, -1.0, 192.6), vec3(-24.8, 35.5, 199.4)));
        occ = max(occ, evalSoftBoxSolarOcc(wp, invRd, vec3(-31.2, -1.0, 291.6), vec3(-24.8, 35.5, 298.4)));
        float tBr = (-28.0 - wp.x) * invRd.x;
        if (tBr > 0.3) {
          vec3 pB = wp + rayDir * tBr;
          if (pB.z >= 168.0 && pB.z <= 322.0) {
            float uArch = clamp((pB.z - 245.65) / 74.0, -1.0, 1.0);
            float deckTop = 4.15 + 4.95 * max(0.0, 1.0 - uArch * uArch) + 0.85;
            float deckBot = deckTop - 2.35;
            float penB = 0.25 + 0.014 * tBr;
            float inSlab = smoothstep(deckBot - penB, deckBot + penB, pB.y)
                         * (1.0 - smoothstep(deckTop - penB, deckTop + penB, pB.y));
            occ = max(occ, inSlab * 0.94);
          }
        }

        // 2D. Gunung Gede Bogor on Pulau I (-32, 698, conical peak y = 58.0m)
        vec2 relVolc = wp.xz - vec2(-32.0, 698.0);
        float tVolc = -dot(relVolc, rayDir.xz) / rdXZ2;
        if (tVolc > 0.5) {
          vec3 pV = wp + rayDir * tVolc;
          if (pV.y >= -0.5 && pV.y <= 57.8 && pV.z > 590.0) {
            float rCone = mix(94.0, 15.5, clamp(pV.y / 57.5, 0.0, 1.0));
            float rHitV = length(pV.xz - vec2(-32.0, 698.0));
            float penV = 0.85 + 0.018 * tVolc;
            float topV = 1.0 - smoothstep(55.5, 57.8, pV.y);
            occ = max(occ, (1.0 - smoothstep(rCone - penV, rCone + penV, rHitV)) * topV);
          }
        }

        // 2E. Exact Analytical Pulau U Hill Massifs matching evalArchipelagoAnalyticalHeight 1:1!
        float tH1 = -dot(wp.xz - vec2(-336.0, 236.0), rayDir.xz) / rdXZ2;
        if (tH1 > 1.0) {
          vec3 pH1 = wp + rayDir * tH1;
          float fg1 = smoothstep(125.0, 185.0, length(pH1.xz - vec2(-12.0, 48.0)));
          float h1 = 18.6 * exp(-dot(pH1.xz - vec2(-336.0, 236.0), pH1.xz - vec2(-336.0, 236.0)) / 9200.0) * fg1;
          if (h1 > 0.6) occ = max(occ, smoothstep(pH1.y - 0.8, pH1.y + 0.8, h1));
        }
        float tH2 = -dot(wp.xz - vec2(-68.0, 392.0), rayDir.xz) / rdXZ2;
        if (tH2 > 1.0) {
          vec3 pH2 = wp + rayDir * tH2;
          float fg2 = smoothstep(125.0, 185.0, length(pH2.xz - vec2(-12.0, 48.0)));
          float h2 = 16.8 * exp(-dot(pH2.xz - vec2(-68.0, 392.0), pH2.xz - vec2(-68.0, 392.0)) / 7800.0) * fg2;
          if (h2 > 0.6) occ = max(occ, smoothstep(pH2.y - 0.8, pH2.y + 0.8, h2));
        }
        float tH3 = -dot(wp.xz - vec2(-228.0, 384.0), rayDir.xz) / rdXZ2;
        if (tH3 > 1.0) {
          vec3 pH3 = wp + rayDir * tH3;
          float fg3 = smoothstep(125.0, 185.0, length(pH3.xz - vec2(-12.0, 48.0)));
          float h3 = 14.6 * exp(-dot(pH3.xz - vec2(-228.0, 384.0), pH3.xz - vec2(-228.0, 384.0)) / 6800.0) * fg3;
          if (h3 > 0.6) occ = max(occ, smoothstep(pH3.y - 0.8, pH3.y + 0.8, h3));
        }
        float tH4 = -dot(wp.xz - vec2(-248.0, 24.0), rayDir.xz) / rdXZ2;
        if (tH4 > 1.0) {
          vec3 pH4 = wp + rayDir * tH4;
          float fg4 = smoothstep(125.0, 185.0, length(pH4.xz - vec2(-12.0, 48.0)));
          float h4 = 14.2 * exp(-dot(pH4.xz - vec2(-248.0, 24.0), pH4.xz - vec2(-248.0, 24.0)) / 5900.0) * fg4;
          if (h4 > 0.6) occ = max(occ, smoothstep(pH4.y - 0.8, pH4.y + 0.8, h4));
        }

        // 3. Lightweight 6-Tap Dual-Island uCollisionMap Raymarch (Trees, Canopy, ATC, Hanggar, & Baked Terrain)
        if (uCollisionReady > 0.5 && occ < 0.99) {
          for (int si = 1; si <= 6; si++) {
            float fsi = float(si);
            float tM = 3.8 * fsi * fsi + 2.0;
            vec3 pM = wp + rayDir * tM;
            if (pM.y < 54.0) {
              vec2 cuv = (pM.xz - uCollisionBounds.xy) / uCollisionBounds.zw;
              if (cuv.x > 0.002 && cuv.x < 0.998 && cuv.y > 0.002 && cuv.y < 0.998) {
#if __VERSION__ == 300
                vec4 cTex = textureLod(uCollisionMap, cuv, 0.0);
#else
                vec4 cTex = texture2D(uCollisionMap, cuv);
#endif
                float yCol = uCollisionHeightRange.x + ((cTex.r * 255.0 * 256.0 + cTex.g * 255.0) / 65535.0) * uCollisionHeightRange.y;
                if (yCol > -0.6) {
                  float penM = 0.32 + 0.016 * tM;
                  float isCanopy = step(0.98, cTex.a);
                  float hitCol = smoothstep(pM.y - penM, pM.y + penM, yCol);
                  occ = max(occ, hitCol * mix(1.0, 0.88, isCanopy));
                }
              }
            }
          }
        }

        return horVis * clamp(1.0 - occ, 0.0, 1.0);
      }
    `;
  }

  /**
   * Hukum Emas 14: Analytical Sphere Sun/Moon + Radial Euclidean SDF Stars +
   * 2-Layer 6-Octave Rotated FBM Cumulonimbus Storm Clouds + Forked Plasma Lightning
   */
  _initAnalyticalSkyDome() {
    const skyGeo = new THREE.SphereGeometry(4200, 64, 36);
    const skyMat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      extensions: { derivatives: true },
      uniforms: {
        uTime: this.uniforms.uTime,
        uCamPos: this.uniforms.uCamPos,
        uSunDir: this.uniforms.uSunDir,
        uTrueSunDir: this.uniforms.uTrueSunDir,
        uSolarDirectTrans: this.uniforms.uSolarDirectTrans,
        uMoonDir: this.uniforms.uMoonDir,
        uMoonPhaseSunDir: this.uniforms.uMoonPhaseSunDir,
        uMoonIllumFraction: this.uniforms.uMoonIllumFraction,
        uWorldToEqMat: this.uniforms.uWorldToEqMat,
        uStarGridTex: this.uniforms.uStarGridTex,
        uSunColor: this.uniforms.uSunColor,
        uNightFactor: this.uniforms.uNightFactor,
        uDuskFactor: this.uniforms.uDuskFactor,
        uRainIntensity: this.uniforms.uRainIntensity,
        uCloudCover: this.uniforms.uCloudCover,
        uStormFactor: this.uniforms.uStormFactor,
        uFogFactor: this.uniforms.uFogFactor,
        uLightningFlash: this.uniforms.uLightningFlash,
        uLightningDir: this.uniforms.uLightningDir,
        uLightningSeed: this.uniforms.uLightningSeed,
        uWindVec: this.uniforms.uWindVec,
        uWindPhase: this.uniforms.uWindPhase,
        uHvsParams: this.uniforms.uHvsParams,
        uHvsGlare: this.uniforms.uHvsGlare,
        uGenshinMode: this.uniforms.uGenshinMode,
        uRenderingPlanarRefl: this.uniforms.uRenderingPlanarRefl
      },
      vertexShader: /* glsl */ `
        varying vec3 vWorldDir;
        void main() {
          vWorldDir = normalize(position);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec3 vWorldDir;
        uniform float uTime;
        uniform vec3 uCamPos;
        uniform vec3 uSunDir;
        uniform vec3 uTrueSunDir;
        uniform float uSolarDirectTrans;
        uniform vec3 uMoonDir;
        uniform vec3 uMoonPhaseSunDir;
        uniform float uMoonIllumFraction;
        uniform mat3 uWorldToEqMat;
        uniform sampler2D uStarGridTex;
        uniform vec3 uSunColor;
        uniform float uNightFactor;
        uniform float uDuskFactor;
        uniform float uRainIntensity;
        uniform float uCloudCover;
        uniform float uStormFactor;
        uniform float uFogFactor;
        uniform float uLightningFlash;
        uniform vec3 uLightningDir;
        uniform float uLightningSeed;
        uniform vec3 uWindVec;
        uniform vec2 uWindPhase;
        uniform vec4 uHvsParams;
        uniform vec4 uHvsGlare;
        uniform float uRenderingPlanarRefl;

        ${HumanEyeHDROptics.getGLSLKernel()}
        ${WeatherOpticsEngine._getSphericalFarSeaGLSL()}

        // Hukum Emas 14: Strict domain rotation matrix preventing periodic axial striping
        const mat2 ROT_FBM = mat2(0.8, 0.6, -0.6, 0.8);

        // Exact IAU J2000.0 Equatorial -> Galactic Rotation Matrix (Column-Major for GLSL):
        // Row 0 -> Galactic Center (RA 17h45m37.2s, Dec -28.9362°)
        // Row 2 -> North Galactic Pole (RA 12h51m26.3s, Dec +27.1283°)
        const mat3 GAL_FROM_EQ = mat3(
          -0.05487556,  0.49410943, -0.86766615,
          -0.87343709, -0.44482963, -0.19807637,
          -0.48383502,  0.74698224,  0.45598378
        );

        float hash21(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        float noise2(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          vec2 u = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
            mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
            u.y
          );
        }

        float fbmGal4(vec2 p) {
          float v = 0.0;
          float a = 0.54;
          for (int i = 0; i < 4; i++) {
            v += a * noise2(p);
            p = ROT_FBM * p * 2.11 + vec2(13.7, 27.3);
            a *= 0.48;
          }
          return v;
        }

        // 6-Octave Domain-Rotated FBM for billowing tropical Cumulonimbus structure
        float fbmCloud6(vec2 p) {
          float v = 0.0;
          float a = 0.52;
          for (int i = 0; i < 6; i++) {
            v += a * noise2(p);
            p = ROT_FBM * p * 2.04 + vec2(17.1, 31.7);
            a *= 0.5;
          }
          return v;
        }

        // Fast 3-Octave & 2-Octave Cloud Shadow / Crepuscular Shaft Evaluators (saves 22 noise2 octaves/pixel & halves D3D11 fxc unroll!)
        float fbmCloud3(vec2 p) {
          float v = 0.52 * noise2(p);
          p = ROT_FBM * p * 2.04 + vec2(17.1, 31.7);
          v += 0.26 * noise2(p);
          p = ROT_FBM * p * 2.04 + vec2(17.1, 31.7);
          v += 0.13 * noise2(p);
          return v * 1.08;
        }

        float fbmCloud2(vec2 p) {
          float v = 0.58 * noise2(p);
          p = ROT_FBM * p * 2.09 + vec2(13.7, 27.3);
          v += 0.32 * noise2(p);
          return v * 1.06;
        }

        float fbmBolt4(vec2 p) {
          float v = 0.0;
          float a = 0.55;
          for (int i = 0; i < 3; i++) {
            v += a * (noise2(p) - 0.5);
            p = ROT_FBM * p * 2.35 + vec2(9.3, 21.4);
            a *= 0.5;
          }
          return v;
        }

        // Physical Planck B-V Color Index -> Stellar Photosphere RGB Chromaticity
        vec3 evalPlanckStellarColor(float bv) {
          vec3 hotOBA = vec3(0.68, 0.82, 1.00);   // Rigel, Acrux, Spica, Orion Belt (B-V < 0)
          vec3 pureAF = vec3(0.96, 0.98, 1.00);   // Sirius, Vega, Altair, Fomalhaut (B-V ~ 0.0..0.3)
          vec3 solarGK = vec3(1.00, 0.91, 0.72);  // Alpha Centauri, Capella, Pollux (B-V ~ 0.7..1.0)
          vec3 rubyKM = vec3(1.00, 0.60, 0.32);   // Betelgeuse, Antares, Aldebaran, Gacrux (B-V > 1.4)
          vec3 c = mix(hotOBA, pureAF, smoothstep(-0.28, 0.14, bv));
          c = mix(c, solarGK, smoothstep(0.18, 0.88, bv));
          c = mix(c, rubyKM, smoothstep(0.95, 1.75, bv));
          return c;
        }

        // Procedural Tidally-Locked Nearside Selenographic Albedo & Micro-Crater Relief
        // (Anorthosite Lunar Highlands vs Dark Basalt Maria Basins + Southern Tycho Ejecta Rays)
        vec2 evalLunarMariaSurface(vec2 sDisk, float zMoon) {
          // True 3D spherical selenographic coordinates centered on sub-Earth Sinus Medii (0°, 0°),
          // naturally foreshortening Mare Crisium & Oceanus Procellarum toward the lunar limb!
          float sLon = atan(sDisk.x, max(0.06, zMoon));
          float sLat = asin(clamp(sDisk.y, -1.0, 1.0));
          vec2 sUV = vec2(sLon, sLat);

          float craterMicro = fbmGal4(sUV * 11.5 + vec2(3.1, 7.4));
          float mareMacro = fbmGal4(sUV * 3.1 + vec2(11.2, 5.8));

          // Canonical Earth-facing nearside Lunar Maria basins (rad):
          // Mare Imbrium (-18°W, +33°N), Mare Serenitatis (+18°E, +28°N), Mare Tranquillitatis (+31°E, +9°N),
          // Mare Crisium (+58°E, +17°N), Mare Fecunditatis (+51°E, -8°S), Oceanus Procellarum (-46°W, +10°N),
          // Mare Nubium (-17°W, -20°S), Mare Humorum (-39°W, -24°S)
          float mImbrium   = exp(-dot(sUV - vec2(-0.31,  0.56), sUV - vec2(-0.31,  0.56)) * 6.2);
          float mSeren     = exp(-dot(sUV - vec2( 0.31,  0.48), sUV - vec2( 0.31,  0.48)) * 9.5);
          float mTranquil  = exp(-dot(sUV - vec2( 0.54,  0.15), sUV - vec2( 0.54,  0.15)) * 7.8);
          float mCrisium   = exp(-dot(sUV - vec2( 1.01,  0.30), sUV - vec2( 1.01,  0.30)) * 22.0);
          float mFecund    = exp(-dot(sUV - vec2( 0.88, -0.14), sUV - vec2( 0.88, -0.14)) * 10.5);
          float mProcell   = exp(-dot(sUV - vec2(-0.78,  0.16), sUV - vec2(-0.78,  0.16)) * 3.6);
          float mNubium    = exp(-dot(sUV - vec2(-0.30, -0.35), sUV - vec2(-0.30, -0.35)) * 7.4);
          float mHumorum   = exp(-dot(sUV - vec2(-0.68, -0.42), sUV - vec2(-0.68, -0.42)) * 13.0);

          float mariaRaw = mImbrium * 1.05 + mSeren * 0.98 + mTranquil * 0.96 + mCrisium * 0.92
                         + mFecund * 0.82 + mProcell * 0.90 + mNubium * 0.84 + mHumorum * 0.80;
          float mariaMask = clamp(mariaRaw * (0.72 + 0.54 * mareMacro), 0.0, 1.0);

          // Southern Highland Tycho crater (-11°W, -43°S) & Copernican bright ejecta rays
          vec2 dTycho = sUV - vec2(-0.19, -0.75);
          float rTycho = length(dTycho) + 1e-4;
          float rayAng = atan(dTycho.y, dTycho.x);
          float tychoRays = exp(-rTycho * 2.9) * (0.42 + 0.58 * pow(0.5 + 0.5 * sin(rayAng * 9.0 + mareMacro * 2.4), 3.0));

          float highland = 0.98 + 0.16 * craterMicro + 0.24 * tychoRays;
          float basaltMare = 0.14 + 0.07 * craterMicro;
          float albedo = mix(highland, basaltMare, smoothstep(0.18, 0.62, mariaMask));
          return vec2(albedo, craterMicro);
        }

        void main() {
          vec3 dir = normalize(vWorldDir);
          // Uniform-control-flow angular pixel footprint (radians/px) for subpixel-stable constellation stars
          float skyPxRad = clamp(max(length(dFdx(dir)), length(dFdy(dir))), 0.00050, 0.00340);
          float h = clamp(dir.y, 0.0, 1.0);
          float overcast = clamp(uCloudCover * 0.62 + uRainIntensity * 0.48, 0.0, 1.0);

          vec3 trueSunDir = normalize(uTrueSunDir);
          vec3 moonDir = normalize(uMoonDir);
          vec3 celestialDir = normalize(uSunDir);
          float cosTheta = dot(dir, celestialDir);
          float cosSun = dot(dir, trueSunDir);
          float cosMoon = dot(dir, moonDir);

          // 1. FULL-FEATURE RAYLEIGH + OZONE CHAPPUIS + BORTLE CLASS 1-2 ISLAND NIGHT ATMOSPHERE
          //    Zero urban orange skyglow pollution! Only pure dark oceanic cosmos, subtle 557.7nm natural
          //    airglow near the marine horizon, and physical lunar Rayleigh scatter when the Moon is risen.
          float gSkyMode = clamp(uGenshinMode, 0.0, 1.0);
          vec3 dayZenith = mix(vec3(0.12, 0.36, 0.74), vec3(0.09, 0.35, 0.82), gSkyMode * 0.88);
          vec3 dayHorizon = mix(vec3(0.62, 0.78, 0.92), vec3(0.55, 0.80, 0.96), gSkyMode * 0.85);

          // Stratospheric Ozone Chappuis absorption deepens twilight zenith to royal cobalt-indigo
          vec3 duskZenith = mix(vec3(0.10, 0.15, 0.38), vec3(0.24, 0.18, 0.56), gSkyMode * 0.85);
          vec3 duskHorizonPBR = mix(vec3(0.52, 0.56, 0.68), vec3(0.94, 0.47, 0.20), smoothstep(-0.35, 0.85, cosSun));
          vec3 duskHorizonGenshin = mix(vec3(0.58, 0.49, 0.86), vec3(1.00, 0.52, 0.28), smoothstep(-0.35, 0.85, cosSun));
          vec3 duskHorizon = mix(duskHorizonPBR, duskHorizonGenshin, gSkyMode * 0.88);

          float lunarSkyScatter = clamp(moonDir.y, 0.0, 1.0) * uMoonIllumFraction * (1.0 - overcast * 0.75);
          // Calibrated IEC-sRGB tropical midnight sapphire-navy sky (maps through sRGB->Linear->HVS to natural deep navy #111A2A, NEVER pitch-black (0,0,0)!)
          vec3 bortle1Zenith = mix(vec3(0.056, 0.086, 0.145), vec3(0.068, 0.125, 0.265), gSkyMode * 0.75) + vec3(0.038, 0.058, 0.098) * lunarSkyScatter;
          vec3 bortle1Horizon = mix(vec3(0.084, 0.122, 0.192), vec3(0.095, 0.168, 0.345), gSkyMode * 0.78) + vec3(0.042, 0.062, 0.104) * lunarSkyScatter;

          vec3 zenith = mix(mix(dayZenith, duskZenith, uDuskFactor), bortle1Zenith, uNightFactor);
          vec3 horizon = mix(mix(dayHorizon, duskHorizon, uDuskFactor), bortle1Horizon, uNightFactor);
          vec3 skyCol = mix(horizon, zenith, pow(h, 0.56));

          // Anti-Solar Belt of Venus ("Sabuk Venus") & Earth's Shadow Arch ("Bayangan Bumi") at Dawn/Dusk!
          if (uDuskFactor > 0.01 && overcast < 0.88) {
            vec2 dirXZ = normalize(dir.xz + vec2(1e-5, 0.0));
            vec2 sunXZ = normalize(trueSunDir.xz + vec2(1e-5, 0.0));
            float antiSun = clamp(dot(dirXZ, -sunXZ) * 0.5 + 0.5, 0.0, 1.0);
            float shadowTop = clamp(0.03 + max(0.0, -trueSunDir.y) * 0.55, 0.02, 0.14);
            float earthShadow = (1.0 - smoothstep(shadowTop * 0.45, shadowTop + 0.05, h)) * pow(antiSun, 1.4);
            float beltOfVenus = exp(-pow((h - (shadowTop + 0.055)) / 0.068, 2.0)) * pow(antiSun, 1.25);
            float clearTwilight = uDuskFactor * (1.0 - overcast * 0.85) * (1.0 - uNightFactor * 0.85);
            skyCol = mix(skyCol, vec3(0.11, 0.14, 0.25), earthShadow * clearTwilight * 0.62);
            skyCol += vec3(0.42, 0.20, 0.26) * beltOfVenus * clearTwilight * 0.55;
          }

          // Deep atmospheric Rayleigh/Mie storm overcast gradient
          vec3 stormHorizonDay = mix(vec3(0.35, 0.39, 0.44), vec3(0.18, 0.21, 0.26), uStormFactor);
          vec3 stormZenithDay = mix(vec3(0.20, 0.24, 0.29), vec3(0.09, 0.11, 0.15), uStormFactor);
          vec3 stormBase = mix(stormHorizonDay, stormZenithDay, pow(h, 0.55));
          stormBase *= mix(1.0, 0.22, uNightFactor);
          skyCol = mix(skyCol, stormBase, clamp(uRainIntensity * 0.74 + uStormFactor * 0.34, 0.0, 0.92));

          // 2. NATURAL PHOTOGRAPHIC MILKY WAY ("BIMASAKTI"), GREAT RIFT DUST LANES,
          //    CRISP 1-PIXEL PINPOINT CONSTELLATION STARS, & PLANAR-REFLECTION FAST-PATH
          vec3 eqDir = normalize(uWorldToEqMat * dir);
          vec3 galDir = GAL_FROM_EQ * eqDir;
          float starClearVis = smoothstep(0.12, 0.84, uNightFactor)
                             * clamp(1.0 - uRainIntensity * 1.12 - uStormFactor * 0.68, 0.0, 1.0);
          vec3 hdrCosmos = vec3(0.0);

          // Zodiacal Light Cone ("Cahaya Zodiak" — subtle interplanetary dust glow in post-dusk/pre-dawn)
          if (uNightFactor > 0.30 && trueSunDir.y < -0.08 && trueSunDir.y > -0.54 && dir.y > 0.0) {
            float zodAlong = max(0.0, dot(normalize(dir.xz + vec2(1e-5, 0.0)), normalize(trueSunDir.xz + vec2(1e-5, 0.0))));
            float zodCone = pow(zodAlong, 5.2) * exp(-h * 2.6) * smoothstep(-0.54, -0.14, trueSunDir.y);
            hdrCosmos += vec3(0.0018, 0.0022, 0.0030) * zodCone * uNightFactor * (1.0 - overcast) * (1.0 - lunarSkyScatter * 0.65);
          }

          if (starClearVis > 0.005 && dir.y > -0.01) {
            float horizonExtinction = smoothstep(-0.005, 0.085, dir.y);
            float moonGlareWash = 1.0 - 0.32 * lunarSkyScatter - 0.56 * pow(max(cosMoon, 0.0), 18.0) * lunarSkyScatter;
            float darkSkyVis = starClearVis * horizonExtinction * clamp(moonGlareWash, 0.28, 1.0);

            // 2A. Delicate True J2000 Galactic Milky Way Band & Great Rift Dust Lanes (calibrated linear radiance!)
            float sinB = galDir.z;
            float coreProximity = clamp(galDir.x * 0.5 + 0.5, 0.0, 1.0);
            float bulgeWidth = mix(0.105, 0.215, pow(coreProximity, 3.0));
            float mwWarp = (fbmGal4(galDir.xy * 2.4 + vec2(sinB * 3.8, -sinB * 3.4)) - 0.5) * 0.065;
            float mwProfile = exp(-pow((sinB + mwWarp) / bulgeWidth, 2.0));

            float starCloudTex = fbmGal4(galDir.xy * 5.2 + vec2(sinB * 10.2 + 2.1, -sinB * 9.6 - 1.3));
            float fineCloudTex = uRenderingPlanarRefl < 0.5
              ? fbmGal4(galDir.xy * 13.5 + vec2(-sinB * 21.0 - 4.7, sinB * 20.0 + 3.8))
              : 0.45;
            float mwLuminance = mwProfile * (0.42 + 0.78 * starCloudTex + 0.34 * fineCloudTex);
            mwLuminance *= mix(0.62, 1.48, pow(coreProximity, 2.0));

            // Great Rift ("Celah Besar") & Southern Cross Coalsack ("Kantung Arang") Dark Molecular Dust Lanes
            float riftNoise = fbmGal4(galDir.xy * 6.8 + vec2(sinB * 17.5 + 9.4, -sinB * 16.4 + 2.2));
            float riftBand = exp(-pow((sinB - 0.012 + (riftNoise - 0.5) * 0.055) / 0.044, 2.0));
            float riftStrength = smoothstep(0.34, 0.72, riftNoise) * smoothstep(-0.35, 0.55, galDir.x);
            float dCoalsack = length(galDir - vec3(0.54012, -0.84118, -0.02600));
            float coalsack = exp(-(dCoalsack * dCoalsack) / 0.0068) * 0.80;
            float dustTransmittance = clamp(1.0 - riftBand * riftStrength * 0.78 - coalsack, 0.16, 1.0);

            vec3 mwOuterLin = vec3(0.0062, 0.0092, 0.0150);
            vec3 mwCoreLin = vec3(0.0180, 0.0156, 0.0130);
            vec3 mwColorLin = mix(mwOuterLin, mwCoreLin, pow(coreProximity, 2.0) * (0.55 + 0.45 * starCloudTex));
            hdrCosmos += mwColorLin * mwLuminance * dustTransmittance * darkSkyVis;

            // Skip sub-arcminute pinpoint stars, nebulae, & meteors inside low-res planar water reflection
            // (eliminates white star-blob/speckle artifacts on dark ocean waves AND makes planar sky 3x faster!)
            if (uRenderingPlanarRefl < 0.5) {
              // Subtle Southern Deep-Sky Wisps (whisper-faint naked-eye realistic surface brightness, zero blobs!)
              vec3 lmcEq = vec3(0.0549, 0.3416, -0.9382);
              vec3 smcEq = vec3(0.2878, 0.0674, -0.9553);
              vec3 m42Eq = vec3(0.1066, 0.9898, -0.0940);
              float dLMC = length(eqDir - lmcEq);
              float dSMC = length(eqDir - smcEq);
              float dM42 = length(eqDir - m42Eq);
              if (dLMC < 0.075) {
                float lmcTex = 0.52 + 0.48 * fbmGal4(eqDir.xy * 42.0);
                hdrCosmos += vec3(0.00065, 0.00092, 0.00145) * exp(-pow(dLMC / 0.024, 2.0)) * lmcTex * darkSkyVis;
              }
              if (dSMC < 0.055) {
                hdrCosmos += vec3(0.00048, 0.00068, 0.00110) * exp(-pow(dSMC / 0.016, 2.0)) * darkSkyVis;
              }
              if (dM42 < 0.0075) {
                hdrCosmos += vec3(0.0020, 0.0026, 0.0036) * exp(-pow(dM42 / 0.0016, 2.0)) * darkSkyVis;
              }

              // 2B. Tier-A Faint Equatorial-Locked Galactic Pinpoint Background Stars (~4,200 fine 1px Bortle stars)
              vec3 bP3 = eqDir * 104.0;
              vec3 bBase = floor(bP3 - 0.5);
              float galStarThresh = mix(0.87, 0.62, exp(-abs(sinB) * 6.5));
              for (int oz = 0; oz < 2; oz++) {
                for (int oy = 0; oy < 2; oy++) {
                  for (int ox = 0; ox < 2; ox++) {
                    vec3 bCell = bBase + vec3(float(ox), float(oy), float(oz));
                    float h0 = fract(sin(dot(bCell, vec3(127.1, 311.7, 74.7))) * 43758.5453);
                    if (h0 > galStarThresh) {
                      float h1 = fract(sin(dot(bCell, vec3(269.5, 183.3, 246.1))) * 43758.5453);
                      float h2 = fract(sin(dot(bCell, vec3(113.5, 271.9, 124.6))) * 43758.5453);
                      float h3 = fract(sin(dot(bCell, vec3(419.2, 371.9, 168.3))) * 43758.5453);
                      vec3 bStarPos = bCell + vec3(0.05 + 0.90 * h1, 0.05 + 0.90 * h2, 0.05 + 0.90 * h3);
                      if (abs(length(bStarPos) - 104.0) < 0.165) {
                        vec3 bStarDir = normalize(bStarPos);
                        float dAng = length(eqDir - bStarDir);
                        float rAng = max(skyPxRad * 0.78, mix(0.00055, 0.00092, h2));
                        if (dAng < rAng) {
                          float twkBg = 0.84 + 0.16 * sin(uTime * (1.8 + h3 * 2.5) + h0 * 50.0);
                          float mBg = smoothstep(rAng, 0.0, dAng) * mix(0.018, 0.062, h3 * h3);
                          vec3 cBg = evalPlanckStellarColor(mix(-0.15, 1.45, fract((h1 + h2) * 7.3)));
                          hdrCosmos += cBg * mBg * twkBg * darkSkyVis;
                        }
                      }
                    }
                  }
                }
              }

              // 2C. Tier-B O(1) REAL J2000 CONSTELLATION STAR CATALOG (212 Crisp 1px Diamond Pinpoint Stars!)
              float eqRA = atan(eqDir.y, eqDir.x);
              float eqDec = asin(clamp(eqDir.z, -1.0, 1.0));
              float raPos = mod(eqRA + 6.28318530718, 6.28318530718);
              float cx = clamp(floor((raPos / 6.28318530718) * 128.0), 0.0, 127.0);
              float cy = clamp(floor((eqDec / 3.14159265359 + 0.5) * 64.0), 0.0, 63.0);

              for (int slot = 0; slot < 4; slot++) {
                vec2 sTexUV = vec2((cx * 4.0 + float(slot) + 0.5) / 512.0, (cy + 0.5) / 64.0);
#if __VERSION__ == 300
                vec4 stData = textureLod(uStarGridTex, sTexUV, 0.0);
#else
                vec4 stData = texture2D(uStarGridTex, sTexUV);
#endif
                if (stData.w > 0.0) {
                  vec3 stEq = normalize(stData.xyz);
                  float theta = length(eqDir - stEq);
                  if (theta < 0.0044) {
                    float magCode = floor(stData.w);
                    float vmag = magCode * 0.01 - 2.0;
                    float bv = fract(stData.w) * 2.6 - 0.4;
                    vec3 starRGB = evalPlanckStellarColor(bv);

                    float radialWindow = pow(clamp(1.0 - theta / 0.0044, 0.0, 1.0), 2.5);
                    // Strictly sub-bloom-threshold (< 0.65 < 0.77) linear flux -> razor-sharp 1px diamond stars, ZERO 16px bloom golf-balls!
                    float rawFlux = pow(2.05, 2.65 - vmag);
                    float coreFlux = clamp((0.26 * rawFlux) / (1.0 + 0.36 * rawFlux), 0.0, 0.64);
                    float haloFlux = clamp((0.045 * rawFlux) / (1.0 + 0.52 * rawFlux), 0.0, 0.082);

                    float starSeed = fract(magCode * 0.173 + stEq.x * 13.7 + stEq.y * 29.1);
                    float lowAltScint = mix(0.24, 0.05, smoothstep(0.05, 0.55, h));
                    float twinkle = 1.0 + lowAltScint * sin(uTime * (3.6 + starSeed * 4.2) + starSeed * 62.8)
                                        * cos(uTime * (2.1 + starSeed * 2.9) - starSeed * 19.4);

                    float magProm = clamp((3.9 - vmag) / 5.2, 0.0, 1.0);
                    float coreRad = max(skyPxRad * mix(0.54, 0.82, magProm), mix(0.00038, 0.00068, magProm));
                    float airyCore = exp(-pow(theta / coreRad, 2.0));
                    float innerAureole = exp(-theta / max(coreRad * 2.1, 0.0011)) * 0.18;

                    vec3 coreCol = mix(starRGB, vec3(1.0), 0.42 * airyCore);
                    vec3 sContrib = (coreCol * airyCore * coreFlux + starRGB * innerAureole * haloFlux)
                                  * radialWindow * twinkle * darkSkyVis;
                    hdrCosmos += sContrib;
                  }
                }
              }

              // 2D. Occasional Subtle Tropical Island Shooting Star / Meteor ("Bintang Jatuh" every ~15s)
              if (h > 0.18 && overcast < 0.42) {
                float mCycle = floor(uTime / 15.0);
                float mPhase = fract(uTime / 15.0);
                if (mPhase < 0.048) {
                  float pNorm = mPhase / 0.048;
                  float mAz = hash21(vec2(mCycle, 3.7)) * 6.2831853;
                  float mEl = 0.34 + 0.36 * hash21(vec2(mCycle, 9.1));
                  vec3 mStart = normalize(vec3(cos(mAz) * cos(mEl), sin(mEl), sin(mAz) * cos(mEl)));
                  vec3 mVel = normalize(vec3(-sin(mAz) * 0.72, -0.52, cos(mAz) * 0.45));
                  vec3 mHead = normalize(mStart + mVel * (pNorm * 0.20));
                  vec3 mTail = normalize(mStart + mVel * (max(0.0, pNorm - 0.32) * 0.20));
                  vec3 seg = mHead - mTail;
                  float segLen2 = max(1e-6, dot(seg, seg));
                  float tProj = clamp(dot(dir - mTail, seg) / segLen2, 0.0, 1.0);
                  float dSeg = length((mTail + seg * tProj) - dir);
                  float meteor = exp(-dSeg * 1450.0) * tProj * sin(pNorm * 3.14159265) * darkSkyVis;
                  hdrCosmos += vec3(0.70, 0.84, 1.0) * meteor * 0.36;
                }
              }
            }
          }

          // 3. CONTINUOUS DUAL-BODY CELESTIAL PROJECTION: SOLAR PHOTOSPHERE & REAL-TIME SELENOGRAPHIC MOON!
          //    Spatial clouds occlude both discs via cloudTrans; discVis only attenuates during heavy rain/storm sheets.
          float discVis = clamp(1.0 - uRainIntensity * 0.90 - uStormFactor * 0.65, 0.0, 1.0);
          float sunAboveHor = smoothstep(-0.045, 0.045, trueSunDir.y);
          float sunVis = discVis * sunAboveHor;
          float sunDiscRadius = 0.024;
          float sinSun = sqrt(max(0.0, 1.0 - cosSun * cosSun));
          float rNormSun = sinSun / sunDiscRadius;
          if (cosSun > 0.0 && rNormSun < 1.0 && sunVis > 0.005) {
            float zSphere = sqrt(max(0.0, 1.0 - rNormSun * rNormSun));
            vec3 sunBodyCol = mix(vec3(1.0, 0.95, 0.82), vec3(1.0, 0.64, 0.34), uDuskFactor);
            skyCol = mix(skyCol, sunBodyCol * (0.72 + 0.38 * zSphere), sunVis);
          }

          // Henyey-Greenstein (g = 0.76) Physical Anisotropic Mie Solar Aureole
          float solarHaloVis = smoothstep(-0.14, 0.06, trueSunDir.y);
          float sunHalo = pow(max(cosSun, 0.0), 32.0) * (1.0 - overcast * 0.88) * solarHaloVis;
          float gMie = 0.76;
          float hgPhase = (1.0 - gMie * gMie) / (4.0 * 3.14159265 * pow(max(0.04, 1.0 + gMie * gMie - 2.0 * gMie * max(cosSun, -0.5)), 1.5));
          float mieAureole = clamp(hgPhase * 0.14, 0.0, 0.75) * (1.0 - overcast * 0.82) * solarHaloVis;
          vec3 solarMieTint = mix(vec3(0.52, 0.42, 0.26), vec3(0.96, 0.48, 0.18), uDuskFactor);
          skyCol += (sunHalo + mieAureole) * solarMieTint;

          // Real-Time Selenographic Moon Disc (Evaluated in True Linear-HDR so Lunar Maria & Phase Terminator stay razor-sharp!)
          float moonAboveHor = smoothstep(-0.020, 0.040, moonDir.y);
          float moonDayGain = mix(0.32, 1.0, smoothstep(-0.05, 0.70, uNightFactor + uDuskFactor * 0.55));
          float moonVis = moonDayGain * moonAboveHor * discVis;
          float moonDiscRadius = 0.0215;
          float sinMoon = sqrt(max(0.0, 1.0 - cosMoon * cosMoon));
          float rNormMoon = sinMoon / moonDiscRadius;
          vec3 hdrMoonBody = vec3(0.0);
          float moonOccultMask = 0.0;

          if (cosMoon > 0.0 && rNormMoon < 1.0 && moonAboveHor > 0.005) {
            // Tidally-locked Earth-facing selenographic frame aligned with J2000 Celestial North!
            // As the Moon rises in the East and sets in the West over Depok (-6.37° S), the familiar
            // nearside Lunar Maria rotate naturally with Southern-Hemisphere parallactic angle.
            vec3 celNorthWorld = normalize(vec3(0.0, 0.0, 1.0) * uWorldToEqMat);
            vec3 moonNorthProj = celNorthWorld - moonDir * dot(celNorthWorld, moonDir);
            vec3 mUpFallback = abs(moonDir.y) < 0.94 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
            vec3 moonNorthUp = length(moonNorthProj) > 1e-4 ? normalize(moonNorthProj) : normalize(mUpFallback - moonDir * dot(mUpFallback, moonDir));
            vec3 moonSelRight = normalize(cross(moonDir, moonNorthUp));

            vec2 sDisk = vec2(dot(dir, moonSelRight), dot(dir, moonNorthUp)) / moonDiscRadius;
            float zMoon = sqrt(max(0.0, 1.0 - dot(sDisk, sDisk)));

            // Outward 3D Lunar Sphere Normal in Depok World Space (sub-Earth center is -moonDir!)
            vec3 moonSurfWorldN = normalize(sDisk.x * moonSelRight + sDisk.y * moonNorthUp - zMoon * moonDir);
            vec3 sunFromMoon = normalize(uMoonPhaseSunDir);

            vec2 lunarSurf = evalLunarMariaSurface(sDisk, zMoon);
            float mariaAlbedo = lunarSurf.x;
            float craterRelief = (lunarSurf.y - 0.5) * 0.12;

            // Exact 3D Real-Time Lunar Phase Terminator with Micro-Crater Relief & Lommel-Seeliger Regolith Reflectance!
            float ndotl = dot(moonSurfWorldN, sunFromMoon);
            float terminator = smoothstep(-0.032, 0.055, ndotl + craterRelief);
            float mu0 = max(0.0, ndotl + craterRelief * 0.45);
            float mu = max(0.08, zMoon);
            float lommelSeeliger = (mu0 / (mu0 + mu + 0.06)) * 1.56;
            float limbSoft = smoothstep(1.0, 0.960, rNormMoon);

            // Solid lunar body occults background stars even across the dark unlit limb!
            moonOccultMask = limbSoft * moonAboveHor;

            // Calibrated Linear-HDR regolith (Dark Basalt Maria ~ 118/255 sRGB, Bright Highlands ~ 232/255 sRGB!)
            vec3 sunlitRegolithLin = vec3(0.76, 0.73, 0.65) * mariaAlbedo * lommelSeeliger * terminator;
            vec3 earthshineLin = vec3(0.0058, 0.0086, 0.0145) * (0.32 + 0.68 * mariaAlbedo)
                               * (1.0 - terminator * 0.88) * (0.55 + 0.45 * zMoon) * uNightFactor;
            hdrMoonBody = (sunlitRegolithLin + earthshineLin) * limbSoft * moonVis;
          }

          // Tight, crisp atmospheric lunar corona right around the Moon limb (ZERO wide blurry cotton-ball halo!)
          float moonHaloStrength = uMoonIllumFraction * smoothstep(0.12, 0.82, uNightFactor) * moonAboveHor * discVis;
          float moonTightCorona = (pow(max(cosMoon, 0.0), 540.0) * 0.13 + pow(max(cosMoon, 0.0), 68.0) * 0.018)
                                * (1.0 - overcast * 0.85) * moonHaloStrength;

          // 4. 3-LAYER FULL-FEATURE VOLUMETRIC CLOUDS (HIGH ICE CIRRUS + CUMULUS + CUMULONIMBUS)
          //    Clouds physically occlude stars, Milky Way, and Moon behind them!
          float cloudDensityTotal = 0.0;
          float cloudCoverMask = 0.0;
          if (dir.y > 0.01) {
            vec2 windAdv = uWindPhase * 0.011;
            // Layer 0: High-Altitude Fibrous Tropical Ice Cirrus (8-11 km, glowing coral-gold deep into twilight!)
            vec2 projCirrus = (dir.xz / (dir.y + 0.34)) * 0.78 + windAdv * 0.28;
            float cirrusStreak = fbmGal4(vec2(projCirrus.x * 2.6 + projCirrus.y * 0.8, projCirrus.y * 0.65));
            float cirrusFine = noise2(vec2(projCirrus.x * 11.0 + cirrusStreak * 2.4, projCirrus.y * 3.2));
            float dCirrus = smoothstep(0.56, 0.86, cirrusStreak) * (0.55 + 0.45 * cirrusFine) * (1.0 - uStormFactor * 0.65);

            vec2 projHigh = (dir.xz / (dir.y + 0.26)) * 1.15 + windAdv * 0.55;
            vec2 projLow = (dir.xz / (dir.y + 0.16)) * 1.85 + windAdv * 1.35 + vec2(5.7, -3.4);

            float fHigh = fbmCloud6(projHigh);
            float fLowBase = fbmCloud6(projLow);
            // Sculpted convective cauliflower turret billows along cumulus crowns
            float fTurret = fbmGal4(projLow * 3.65 - windAdv * 0.62 + vec2(4.1, 8.7));
            float fLow = clamp(fLowBase * 0.84 + (fTurret - 0.36) * 0.26, 0.0, 1.0);

            // Tropical clear nights naturally dissipate daytime convective cumulus, opening wide starry island skies!
            float dryNightClearBias = 0.085 * uNightFactor * (1.0 - clamp(uRainIntensity * 2.5 + uStormFactor, 0.0, 1.0));
            float thresh = mix(0.58, 0.20, clamp(uCloudCover, 0.0, 1.0)) + dryNightClearBias;
            float dHigh = smoothstep(thresh, thresh + 0.34, fHigh);
            float dLow = smoothstep(thresh - 0.04, thresh + 0.26, fLow) * (0.48 + 0.52 * uCloudCover);
            cloudDensityTotal = clamp(dCirrus * 0.22 + dHigh * 0.58 + dLow * 0.82, 0.0, 1.0);

            // High cirrus catches upper-stratosphere sunset light slightly longer than low cumulus
            vec3 cirrusDay = mix(vec3(0.96, 0.98, 1.0), vec3(1.0, 0.54, 0.30), clamp(uDuskFactor * 1.25, 0.0, 1.0));
            vec3 cirrusNight = vec3(0.052, 0.078, 0.132) * (0.65 + 0.55 * lunarSkyScatter);
            float horizonFade = smoothstep(0.005, 0.16, dir.y);
            skyCol = mix(skyCol, mix(cirrusDay, cirrusNight, uNightFactor), dCirrus * 0.28 * horizonFade * (1.0 - dLow));

            // Directional Beer-Lambert optical thickness + Schneider Powder-Sugar Multiple Scattering
            vec2 sunStep = normalize(celestialDir.xz + vec2(1e-4, 0.0)) * 0.14;
            float fShadow = fbmCloud3(projLow + sunStep);
            float fShadow2 = fbmCloud2((projLow + sunStep * 2.1) * 1.85);
            float selfLit = clamp(
              1.0 - max(0.0, fShadow - fLowBase) * 3.35 - max(0.0, fShadow2 - fLow) * 1.15 - dLow * (0.30 + 0.40 * uStormFactor),
              0.10,
              1.0
            );
            // Tahap 3: Genshin Painterly Cel-Sculpted Cumulus Billows
            selfLit = mix(selfLit, smoothstep(0.24, 0.66, selfLit) * 0.88 + 0.12, gSkyMode * 0.65);
            float powderEffect = 1.0 - exp(-cloudDensityTotal * 3.4 * (1.0 - uStormFactor * 0.45));
            float multiScatter = 0.21 * powderEffect * (0.38 + 0.62 * selfLit) * (1.0 - uStormFactor * 0.55);
            float forwardScatter = pow(max(cosTheta, 0.0), 8.0) * 0.38 * (1.0 - dLow * 0.66);
            // Crisp diffraction silver-lining rim around sunlit cauliflower cumulus edges!
            float silverEdge = pow(max(cosTheta, 0.0), mix(14.0, 9.0, gSkyMode)) * smoothstep(0.04, 0.34, cloudDensityTotal) * exp(-cloudDensityTotal * 2.35) * (0.84 + 0.72 * gSkyMode);

            vec3 litCloudDay = mix(mix(vec3(0.98, 0.99, 1.0), vec3(1.00, 0.98, 0.94), gSkyMode * 0.65), vec3(0.99, 0.62, 0.36), uDuskFactor);
            vec3 darkBasePBR = mix(vec3(0.47, 0.53, 0.61), vec3(0.13, 0.16, 0.22), clamp(uRainIntensity * 0.65 + uStormFactor * 0.55, 0.0, 1.0));
            vec3 darkBaseGenshin = mix(vec3(0.56, 0.66, 0.88), vec3(0.22, 0.28, 0.44), clamp(uRainIntensity * 0.65 + uStormFactor * 0.55, 0.0, 1.0));
            vec3 darkBaseDay = mix(darkBasePBR, darkBaseGenshin, gSkyMode * 0.78);
            darkBaseDay = mix(darkBaseDay, mix(vec3(0.36, 0.23, 0.27), vec3(0.52, 0.34, 0.58), gSkyMode * 0.80), uDuskFactor * (1.0 - uStormFactor * 0.6));

            vec3 scatterTint = mix(vec3(1.0, 0.96, 0.86), vec3(1.0, 0.64, 0.32), uDuskFactor);
            vec3 cloudColDay = mix(darkBaseDay, litCloudDay, clamp(selfLit + multiScatter, 0.0, 1.0)) + scatterTint * (forwardScatter + silverEdge);
            float moonCloudLit = 0.20 + 0.80 * lunarSkyScatter;
            vec3 nightCloudDark = vec3(0.046, 0.070, 0.122) + vec3(0.022, 0.034, 0.058) * lunarSkyScatter;
            vec3 nightCloudLit = vec3(0.064, 0.094, 0.158) + vec3(0.085, 0.125, 0.195) * lunarSkyScatter;
            vec3 cloudColNight = mix(nightCloudDark, nightCloudLit, selfLit)
                               + vec3(0.20, 0.28, 0.42) * silverEdge * 0.48 * moonCloudLit;
            vec3 cloudCol = mix(cloudColDay, cloudColNight, uNightFactor);

            float mainCloudDens = clamp(dHigh * 0.58 + dLow * 0.80, 0.0, 1.0);
            cloudCoverMask = clamp(mainCloudDens * horizonFade * (0.68 + 0.30 * uCloudCover), 0.0, 0.985);
            skyCol = mix(skyCol, cloudCol, cloudCoverMask);

            // True Cloud-Gap-Occluded Radial Crepuscular God-Rays (3-tap volumetric march toward the Sun; skipped in planar refl!)
            if (uRenderingPlanarRefl < 0.5 && cosSun > 0.12 && solarHaloVis > 0.01 && uRainIntensity < 0.78 && uNightFactor < 0.85) {
              float shaftTrans = 0.0;
              for (int gi = 1; gi <= 3; gi++) {
                float gFrac = float(gi) * 0.28;
                vec3 sampleRay = normalize(mix(dir, trueSunDir, gFrac));
                if (sampleRay.y > 0.015) {
                  vec2 pGap = (sampleRay.xz / (sampleRay.y + 0.16)) * 1.85 + windAdv * 1.35 + vec2(5.7, -3.4);
                  float cGap = fbmCloud3(pGap * 1.02);
                  float gapOpen = 1.0 - smoothstep(thresh - 0.06, thresh + 0.22, cGap);
                  shaftTrans += gapOpen * 0.3333;
                }
              }
              float shaftContrast = smoothstep(0.18, 0.82, shaftTrans) * (1.0 - cloudDensityTotal * 0.58);
              float godRay = pow(max(0.0, cosSun), mix(4.8, 3.6, gSkyMode)) * shaftContrast * smoothstep(0.03, 0.48, uCloudCover);
              vec3 rayCol = mix(vec3(0.48, 0.42, 0.26), vec3(0.96, 0.50, 0.20), uDuskFactor);
              skyCol += rayCol * godRay * solarHaloVis * (1.0 - uRainIntensity) * (1.0 - uNightFactor) * (0.34 + 0.28 * gSkyMode);
            }
          }

          // Distant Horizon Virga / Falling Rain Shafts beneath storm cloud base
          if (uRainIntensity > 0.15 && dir.y > -0.02 && dir.y < 0.32) {
            float az = atan(dir.z, dir.x);
            float shaftNoise = noise2(vec2(az * 7.5 - uTime * 0.09, dir.y * 2.2 + uTime * 0.8));
            float shaftStreaks = noise2(vec2(az * 34.0 + dir.y * 4.5, uTime * 2.2));
            float virgaMask = smoothstep(0.36, 0.82, shaftNoise) * (0.65 + 0.35 * shaftStreaks);
            virgaMask *= smoothstep(0.32, 0.08, dir.y) * smoothstep(-0.02, 0.04, dir.y);
            vec3 virgaCol = mix(vec3(0.46, 0.52, 0.58), vec3(0.06, 0.09, 0.14), uNightFactor);
            skyCol = mix(skyCol, virgaCol, virgaMask * min(1.0, uRainIntensity) * 0.42);
          }

          // Procedural Multi-Branch Forked Plasma Lightning + In-Cloud Volumetric Scattering
          if (uLightningFlash > 0.008 && dir.y > 0.0) {
            vec3 lDir = normalize(uLightningDir);
            float angCos = dot(normalize(vec3(dir.x, 0.0, dir.z) + vec3(1e-5, 0.0, 0.0)), normalize(vec3(lDir.x, 0.0, lDir.z) + vec3(1e-5, 0.0, 0.0)));
            float azDiff = acos(clamp(angCos, -1.0, 1.0));
            float crossSign = sign(dir.x * lDir.z - dir.z * lDir.x);
            float signedAz = azDiff * crossSign;

            float cloudScatter = exp(-azDiff * 2.6) * exp(-abs(dir.y - 0.34) * 3.2);
            skyCol += vec3(0.54, 0.68, 1.0) * cloudScatter * (0.45 + 0.75 * cloudDensityTotal) * uLightningFlash * 1.45;

            float seedT = uLightningSeed;
            if (azDiff < 0.42 && dir.y < 0.56) {
              float jagMain = fbmBolt4(vec2(dir.y * 13.5, seedT * 7.13)) * 0.13
                            + fbmBolt4(vec2(dir.y * 41.0, seedT * 3.71)) * 0.032;
              float mainX = jagMain + (0.28 - dir.y) * 0.08;
              float dMain = abs(signedAz - mainX);
              float topMask = smoothstep(0.55, 0.42, dir.y) * smoothstep(0.0, 0.02, dir.y);

              float core = exp(-dMain * 340.0) * topMask;
              float corona = exp(-dMain * 38.0) * topMask * 0.55;

              float b1Y = clamp(0.36 - dir.y, 0.0, 0.32);
              float jagB1 = fbmBolt4(vec2(dir.y * 22.0 + 4.2, seedT * 5.3)) * 0.06;
              float dB1 = abs(signedAz - (mainX + b1Y * 0.52 + jagB1));
              float mB1 = step(dir.y, 0.36) * smoothstep(0.06, 0.16, dir.y);
              core += exp(-dB1 * 420.0) * mB1 * 0.72;
              corona += exp(-dB1 * 48.0) * mB1 * 0.32;

              float b2Y = clamp(0.25 - dir.y, 0.0, 0.22);
              float jagB2 = fbmBolt4(vec2(dir.y * 26.0 - 8.1, seedT * 9.7)) * 0.05;
              float dB2 = abs(signedAz - (mainX - b2Y * 0.46 + jagB2));
              float mB2 = step(dir.y, 0.25) * smoothstep(0.03, 0.11, dir.y);
              core += exp(-dB2 * 440.0) * mB2 * 0.65;
              corona += exp(-dB2 * 52.0) * mB2 * 0.28;

              skyCol += (vec3(0.96, 0.98, 1.0) * core * 2.6 + vec3(0.48, 0.68, 1.0) * corona) * uLightningFlash;
            }
          }

          // 5. SPHERICAL JAVA SEA HORIZON & GEODETIC 3-TIER WEST JAVA MAINLAND + RIDGED STRATOVOLCANO CORDILLERA
          //    True R_E = 6,371 km Earth curvature + real West Java hypsometry (mdpl):
          //    - Boundless Java Sea seamlessly continuing oceanMesh (r = 2,680m) to spherical horizon slopeHor = -sqrt(2*H/R_E)
          //    - Curved North Java Coastline (Tanjung Kait - Teluk Jakarta - Muara Gembong, D ~ 22.5..30 km) grounded on the sea
          //    - Tier 1: Receding Jabodetabek lowland plain (25..48 km)
          //    - Tier 2: Dissected Bogor-Sentul-Hambalang-Cigudeg folded foothills (48..54 km, 480..940 mdpl)
          //    - Tier 3: Ridged Halimun-Salak (2,211 mdpl) & Pangrango-Gede-Jonggol (3,019 / 2,958 mdpl) at 71..84 km
          float Hcam = max(1.2, uCamPos.y + 1.45);
          float horizLen = max(0.08, length(dir.xz));
          float raySlope = dir.y / horizLen;
          float slopeHor = -sqrt(2.0 * Hcam / 6371000.0);
          float dHorM = sqrt(2.0 * 6371000.0 * Hcam);

          float horizonFogBand = clamp(uFogFactor, 0.0, 1.0) * (1.0 - smoothstep(-0.04, 0.28, dir.y));
          if (horizonFogBand > 0.002) {
            vec3 fogSkyS = mix(mix(vec3(0.79, 0.85, 0.90), vec3(0.90, 0.66, 0.48), uDuskFactor), vec3(0.085, 0.115, 0.175), uNightFactor);
            skyCol = mix(skyCol, fogSkyS, horizonFogBand * 0.82);
          }
          float cloudTrans = (1.0 - cloudCoverMask) * (1.0 - horizonFogBand * 0.78);
          vec3 hdrTropoSky = nexusSRGBToLinear(max(skyCol, vec3(0.0))) * mix(1.16, 0.76, uNightFactor);
          hdrTropoSky += nexusSRGBToLinear(max(solarMieTint, vec3(0.0))) * (sunHalo * 1.45 + mieAureole * 0.85) * (0.25 + 0.75 * cloudTrans);
          if (uLightningFlash > 0.008 && dir.y > 0.0) {
            vec3 lDirH = normalize(uLightningDir);
            float angCosH = dot(normalize(vec3(dir.x, 0.0, dir.z) + vec3(1e-5, 0.0, 0.0)), normalize(vec3(lDirH.x, 0.0, lDirH.z) + vec3(1e-5, 0.0, 0.0)));
            float azDiffH = acos(clamp(angCosH, -1.0, 1.0));
            float boltBoost = exp(-azDiffH * 3.2) * exp(-abs(dir.y - 0.30) * 2.8);
            hdrTropoSky += vec3(0.75, 0.95, 1.55) * boltBoost * uLightningFlash * 1.35;
          }

          // 5A. Boundless Spherical Java Sea below slopeHor (matches oceanMesh r = 2,680m rim 1:1 & merges seamlessly into hdrTropoSky!)
          float seaSolidMask = smoothstep(slopeHor + 0.00055, slopeHor - 0.00110, raySlope);
          float pSea = max(-slopeHor + 1e-6, -raySlope);
          float dSeaM = (2.0 * Hcam) / (pSea + sqrt(max(0.0, pSea * pSea - slopeHor * slopeHor)));
          vec3 hdrSeaLayer = hdrTropoSky;
          if (seaSolidMask > 0.001) {
            hdrSeaLayer = evalSphericalFarSeaHDR(
              dir,
              Hcam,
              dSeaM,
              celestialDir,
              trueSunDir,
              moonDir,
              uDuskFactor,
              uNightFactor,
              uCloudCover,
              uRainIntensity,
              uStormFactor,
              uFogFactor,
              uLightningFlash,
              uWindPhase,
              uSolarDirectTrans
            );
            // Exact C^1-smooth radiative convergence to hdrTropoSky at the spherical horizon dHorM (0.000000 horizon seam!)
            float horMerge = smoothstep(dHorM * 0.42, dHorM * 0.995, dSeaM);
            hdrSeaLayer = mix(hdrSeaLayer, hdrTropoSky, horMerge * 0.96);
          }

          // 5B. Geodetically Grounded 3-Tier Pulau Jawa Mainland & Ridged Stratovolcano Cordillera
          //     Placed at true Southern Kepulauan Seribu <-> Pantai Utara Jawa distance (~11.5-18.5 km),
          //     so the curved North Java coastline, Tanjung Pasir/Kait & Tanjung Priok capes, Teluk Jakarta bay,
          //     littoral shoals, and coastal surf sit VISIBLY IN FRONT OF the spherical sea horizon (dCoastM < dHorM)!
          float javaSolidMask = 0.0;
          vec3 hdrJavaLayer = hdrTropoSky;
          if (dir.x < -0.015 && raySlope > -0.095 && raySlope < 0.085) {
            float azSouth = atan(dir.z, -dir.x); // 0 = South (-X), - = SSW/Banten-Salak, + = SSE/Gede-Jonggol
            float absAz = abs(azSouth);
            float islandEnv = 1.0 - smoothstep(0.68, 1.02, absAz);
            if (islandEnv > 0.001) {
              float marineClarity = clamp(1.0 - uRainIntensity * 0.86 - uFogFactor * 0.88 - uStormFactor * 0.52, 0.05, 1.0);

              // Curved 3D Pantai Utara Jawa Planform Distance from Southern Kepulauan Seribu (~11.5 - 18.8 km):
              // Advancing Tanjung Pasir / Kait (-0.255, ~11.8 km) & Tanjung Priok / Muara Gembong (+0.275, ~12.2 km) capes,
              // concave Teluk Jakarta bay (~17.5 km), curving smoothly back along Banten & Karawang wings toward the horizon!
              float bayRecess =  2900.0 * exp(-pow(azSouth / 0.195, 2.0));
              float capeKait  = -2800.0 * exp(-pow((azSouth + 0.255) / 0.098, 2.0));
              float capeGemb  = -2400.0 * exp(-pow((azSouth - 0.275) / 0.108, 2.0));
              float coastIrr  = (noise2(vec2(azSouth * 14.5, 2.3)) - 0.5) * 1450.0
                              + (noise2(vec2(azSouth * 39.0, 7.1)) - 0.5) * 580.0;
              float wingCurve = 46000.0 * pow(smoothstep(0.28, 0.88, absAz), 1.85);
              float dCoastM = max(11200.0, 14600.0 + bayRecess + capeKait + capeGemb + coastIrr + wingCurve);

              float slopeCoast = (-Hcam / dCoastM) - (dCoastM / 12742000.0);
              float visCoastSlope = (dCoastM < dHorM) ? slopeCoast : slopeHor;

              // Shallow North Java Littoral Shelf & Subtle Breaking Coastal Surf Line in the Java Sea right along Pantai Utara Jawa!
              if (seaSolidMask > 0.01 && dCoastM < dHorM * 0.97) {
                float seaBelowCoast = visCoastSlope - raySlope;
                if (seaBelowCoast > -0.0006 && seaBelowCoast < 0.0032) {
                  float coastHazeS = clamp(1.0 - exp(-dCoastM * 0.000028), 0.30, 0.82);
                  float shoalBand = exp(-max(0.0, seaBelowCoast) / 0.00125) * smoothstep(-0.0005, 0.0001, seaBelowCoast);
                  vec3 shoalTintS = mix(mix(vec3(0.045, 0.165, 0.175), vec3(0.18, 0.13, 0.12), uDuskFactor), vec3(0.012, 0.025, 0.042), uNightFactor);
                  vec3 shoalHDR = mix(nexusSRGBToLinear(shoalTintS) * mix(1.12, 0.74, uNightFactor), hdrTropoSky, coastHazeS);
                  hdrSeaLayer = mix(hdrSeaLayer, shoalHDR, shoalBand * marineClarity * 0.55);

                  float surfNoiseJ = 0.45 + 0.55 * noise2(vec2(azSouth * 165.0, uTime * 0.35));
                  float surfLine = exp(-pow(abs(seaBelowCoast - 0.00014) / 0.00016, 2.0)) * surfNoiseJ;
                  vec3 surfTintS = mix(mix(vec3(0.76, 0.83, 0.88), vec3(0.84, 0.58, 0.42), uDuskFactor), vec3(0.09, 0.13, 0.20), uNightFactor);
                  vec3 surfHDR = mix(nexusSRGBToLinear(surfTintS), hdrTropoSky, coastHazeS);
                  hdrSeaLayer = mix(hdrSeaLayer, surfHDR, surfLine * marineClarity * 0.42);
                }
              }

              // Multi-scale ridged multifractal harmonics for dissected tropical mountain arêtes & barrancos
              float rN1 = 1.0 - abs(noise2(vec2(azSouth * 24.0, 4.1)) * 2.0 - 1.0);
              float rN2 = 1.0 - abs(noise2(vec2(azSouth * 58.0 + 3.7, 8.9)) * 2.0 - 1.0);
              float rN3 = 1.0 - abs(noise2(vec2(azSouth * 136.0 - 5.2, 13.4)) * 2.0 - 1.0);
              float ridgeCrestDetail = (rN1 * 0.56 + rN2 * rN1 * 0.30 + rN3 * rN2 * 0.14 - 0.42);
              float azWarped = azSouth + (noise2(vec2(azSouth * 17.5, 6.3)) - 0.5) * 0.026
                                       + (rN2 - 0.5) * 0.011;

              // Tier 1: North Java Coastal Mangrove/Alluvial Lowland & Jabodetabek Plain (D = dCoastM -> dFootM ~ 34 km)
              float dFootM = dCoastM + 19500.0;
              float hJavaCoast = (155.0 + 48.0 * noise2(vec2(azSouth * 9.5, 1.7)) + 28.0 * rN1) * islandEnv;
              float rawPlainTop = ((hJavaCoast - Hcam) / dFootM) - (dFootM / 12742000.0);
              float slopePlainTop = (dCoastM < dHorM) ? max(rawPlainTop, slopeHor + 0.00045 * islandEnv) : rawPlainTop;

              // Tier 2: Folded Bogor-Sentul-Hambalang-Cigudeg Foothill Belt (~34-42 km, 460-960 mdpl)
              float footEnv = (1.0 - smoothstep(0.46, 0.86, absAz)) * islandEnv;
              float sentulMass = 540.0 * pow(clamp(1.0 - abs(azWarped - 0.08) / 0.35, 0.0, 1.0), 1.35);
              float cigudegMass = 500.0 * pow(clamp(1.0 - abs(azWarped + 0.31) / 0.30, 0.0, 1.0), 1.40);
              float hFootMeters = (265.0 + max(sentulMass, cigudegMass) + 235.0 * rN1 * rN1 + 95.0 * rN2) * footEnv;
              float slopeFootTop = ((hFootMeters - Hcam) / dFootM) - (dFootM / 12742000.0);

              // Tier 3: West Java Volcanic Cordillera (~62-72 km: Halimun-Salak & Pangrango-Gede-Jonggol)
              float dVolcM = 62500.0 + 4800.0 * smoothstep(-0.15, 0.25, azSouth) + 18000.0 * pow(smoothstep(0.38, 0.92, absAz), 1.6);

              // SSW: Pegunungan Halimun-Kendeng (-0.45) & Broad Dissected Gunung Salak Massif (Salak I 2,211m & Salak II 2,180m)
              float qHal1  = clamp(1.0 - sqrt(pow((azWarped + 0.465) / 0.165, 2.0) + 0.018), 0.0, 1.0);
              float qHal2  = clamp(1.0 - sqrt(pow((azWarped + 0.355) / 0.130, 2.0) + 0.022), 0.0, 1.0);
              float qSal2  = clamp(1.0 - sqrt(pow((azWarped + 0.268) / 0.098, 2.0) + 0.011), 0.0, 1.0);
              float qSal1  = clamp(1.0 - sqrt(pow((azWarped + 0.206) / 0.108, 2.0) + 0.009), 0.0, 1.0);
              float qSalAp = clamp(1.0 - abs(azWarped + 0.245) / 0.265, 0.0, 1.0);

              float hHalimun = max(1790.0 * pow(qHal1, 1.48), 1620.0 * pow(qHal2, 1.52));
              float hSalakII = 2165.0 * pow(qSal2, 1.54);
              float hSalakI  = 2211.0 * pow(qSal1, 1.50);
              // Ciapus / Kawah Ratu caldera saddle between Salak I & Salak II
              float salakBreach = 135.0 * exp(-pow((azWarped + 0.238) / 0.020, 2.0));
              float hSalakMassif = (max(hSalakI, hSalakII) - salakBreach) + 460.0 * pow(qSalAp, 1.75);

              // SSE: Broad Conical Gunung Pangrango (3,019 mdpl) + Kandang Badak Saddle + Cratered Gunung Gede (2,958 mdpl) + Jonggol
              float qPang   = clamp(1.0 - sqrt(pow((azWarped - 0.152) / 0.102, 2.0) + 0.0075), 0.0, 1.0);
              float qGede   = clamp(1.0 - sqrt(pow((azWarped - 0.228) / 0.116, 2.0) + 0.0110), 0.0, 1.0);
              float qGedeAp = clamp(1.0 - abs(azWarped - 0.194) / 0.275, 0.0, 1.0);
              float qMega   = clamp(1.0 - sqrt(pow((azWarped - 0.068) / 0.105, 2.0) + 0.020), 0.0, 1.0);
              float qJong1  = clamp(1.0 - sqrt(pow((azWarped - 0.395) / 0.155, 2.0) + 0.018), 0.0, 1.0);
              float qJong2  = clamp(1.0 - sqrt(pow((azWarped - 0.520) / 0.160, 2.0) + 0.022), 0.0, 1.0);

              float hPangrango = 3019.0 * pow(qPang, 1.48);
              // Truncated Kawah Ratu / Suryakencana summit plateau on Gunung Gede
              float gedeCraterNotch = 105.0 * exp(-pow((azWarped - 0.232) / 0.019, 2.0));
              float hGede = 2958.0 * pow(qGede, 1.46) - gedeCraterNotch;
              float kandangBadakBridge = 2420.0 * exp(-pow((azWarped - 0.190) / 0.044, 2.0));
              float hMega = 1480.0 * pow(qMega, 1.52);
              float hJonggol = max(1680.0 * pow(qJong1, 1.50), 1420.0 * pow(qJong2, 1.56));
              float hGedeMassif = max(max(max(hPangrango, hGede), kandangBadakBridge) + 490.0 * pow(qGedeAp, 1.75), max(hMega, hJonggol));

              // Combine cordillera with multi-scale ridged erosion spurs across mid flanks & summits
              float rawCordilleraM = max(hHalimun, max(hSalakMassif, hGedeMassif));
              float erosionAmp = (115.0 + 0.068 * rawCordilleraM) * smoothstep(220.0, 950.0, rawCordilleraM);
              float hVolcMeters = max(0.0, (rawCordilleraM + erosionAmp * ridgeCrestDetail) * islandEnv);
              float slopeVolcTop = ((hVolcMeters - Hcam) / dVolcM) - (dVolcM / 12742000.0);

              // Unified silhouette top slope + WIDE organic interior transitions (zero horizontal band!)
              float slopeJavaMaxTop = max(slopePlainTop, max(slopeFootTop, slopeVolcTop));
              float aaFeather = max(skyPxRad * 0.95, 0.00065);
              float baseMask = smoothstep(visCoastSlope - aaFeather * 1.6, visCoastSlope + aaFeather * 1.6, raySlope);
              float topMask  = smoothstep(slopeJavaMaxTop + aaFeather * 1.25, slopeJavaMaxTop - aaFeather * 1.25, raySlope);
              float anyJava  = baseMask * topMask * step(visCoastSlope + 0.0003, slopeJavaMaxTop);
              // Wide 0.0045-rad interior blend between coastal plain, misty foothills, and volcanic massif (no sharp horizontal line!)
              float wFootLayer = smoothstep(slopeFootTop + 0.0042, slopeFootTop - 0.0042, raySlope)
                               * step(visCoastSlope + 0.0004, slopeFootTop);
              float wPlainLayer = smoothstep(slopePlainTop + 0.0045, slopePlainTop - 0.0045, raySlope)
                                * step(visCoastSlope + 0.0003, slopePlainTop);

              if (anyJava > 0.001) {
                // 3D Azimuthal Macro-Slope + Radial Barranco Flank Normals for True Volumetric Sunlight & Alpenglow!
                float macroGradSalak = -((azSouth + 0.235) / 0.095) * smoothstep(900.0, 2000.0, hSalakMassif);
                float macroGradGede  = -((azSouth - 0.188) / 0.105) * smoothstep(900.0, 2400.0, hGedeMassif);
                float macroGradHal   = -((azSouth + 0.420) / 0.140) * smoothstep(700.0, 1700.0, hHalimun);
                float macroGradJong  = -((azSouth - 0.430) / 0.150) * smoothstep(600.0, 1500.0, hJonggol);
                float macroAzGrad = clamp(macroGradSalak + macroGradGede + macroGradHal + macroGradJong, -2.2, 2.2);

                float vVolc = clamp((raySlope - max(visCoastSlope, slopeFootTop * 0.65)) / max(0.003, slopeVolcTop - visCoastSlope), 0.0, 1.0);
                // Radial barranco ridges fanning down the volcano flanks from summit to apron
                vec2 barrancoUV = vec2(azSouth * 46.0 + (1.0 - vVolc) * macroAzGrad * 0.55, vVolc * 5.2);
                float barrancoA = 1.0 - abs(noise2(barrancoUV) * 2.0 - 1.0);
                float barrancoB = 1.0 - abs(noise2(barrancoUV * 2.35 + vec2(4.7, 1.9)) * 2.0 - 1.0);
                float ridgeFacet = barrancoA * 0.65 + barrancoB * 0.35;
                float barrancoGrad = (noise2(barrancoUV + vec2(0.18, 0.0)) - noise2(barrancoUV - vec2(0.18, 0.0))) * 2.8;

                // World-space 3D outward mountain normal (+X = North facing Kepulauan UI, +Z = East, -Z = West)
                float nzFlank = clamp((macroAzGrad * 0.48 + barrancoGrad * 0.36), -0.88, 0.88);
                vec3 nMount = normalize(vec3(0.72, 0.48 + 0.25 * vVolc, nzFlank));
                vec3 activeLightM = uNightFactor > 0.65 ? moonDir : trueSunDir;
                float sunFacing = clamp(dot(nMount, activeLightM), -0.35, 1.0);
                float directBeamM = clamp(uSolarDirectTrans, 0.0, 1.0) * smoothstep(-0.025, 0.08, trueSunDir.y) * (1.0 - uNightFactor);

                // Intrinsic tropical montane dipterocarp forest + upper-elevation subalpine rock tint
                vec3 forestShadeDay = vec3(0.042, 0.095, 0.098);
                vec3 forestSunDay   = vec3(0.098, 0.185, 0.118);
                vec3 forestDuskShade = vec3(0.055, 0.052, 0.095);
                vec3 forestDuskSun   = vec3(0.320, 0.155, 0.082); // Warm golden-amber sunset alpenglow on western flanks!
                vec3 forestNight     = vec3(0.003, 0.006, 0.013) * (0.45 + 0.55 * lunarSkyScatter);

                float litFrac = clamp(0.32 + 0.68 * max(0.0, sunFacing) * directBeamM + 0.14 * ridgeFacet, 0.0, 1.0);
                vec3 dayFlankS  = mix(forestShadeDay, forestSunDay, litFrac);
                vec3 duskFlankS = mix(forestDuskShade, forestDuskSun, clamp(0.22 + 0.85 * max(0.0, sunFacing) * directBeamM + 0.12 * ridgeFacet, 0.0, 1.0));
                vec3 intrinsicVolcHDR = nexusSRGBToLinear(mix(mix(dayFlankS, duskFlankS, uDuskFactor), forestNight, uNightFactor));

                // Koschmieder Rayleigh-Mie Tropospheric Extinction T(D) = exp(-sigma_eff * D)
                // Grounded depth layering: Coast (12-18 km) ~38-52% veiled, Foothills (34 km) ~64-74% veiled, Volcanoes (64 km) ~76-88% veiled!
                float sigmaTropo = 0.0000245 + 0.000028 * clamp(uFogFactor * 1.4 + uRainIntensity * 1.1 + uCloudCover * 0.25, 0.0, 1.5);

                float altClear = 1.0 - 0.16 * smoothstep(0.35, 0.95, vVolc);
                float hazeVolc = clamp(1.0 - exp(-dVolcM * sigmaTropo * altClear), 0.72, 0.92);
                vec3 javaInteriorHDR = mix(intrinsicVolcHDR, hdrTropoSky, hazeVolc);

                if (wFootLayer > 0.001) {
                  float vFoot = clamp((raySlope - visCoastSlope) / max(0.002, slopeFootTop - visCoastSlope), 0.0, 1.0);
                  float valleyMist = exp(-pow(max(0.0, 1.0 - vFoot) * 2.2, 2.0)) * 0.06 * (1.0 - uNightFactor * 0.75);
                  float footFacet = 0.85 + 0.30 * rN2;
                  float hazeFoot = clamp(1.0 - exp(-dFootM * sigmaTropo * 1.02), 0.56, 0.88);
                  vec3 footHDR = mix(intrinsicVolcHDR * (0.88 * footFacet), hdrTropoSky * (1.0 + valleyMist), hazeFoot);
                  javaInteriorHDR = mix(javaInteriorHDR, footHDR, wFootLayer);
                }

                if (wPlainLayer > 0.001) {
                  float uPlain = clamp((raySlope - visCoastSlope) / max(0.0012, slopePlainTop - visCoastSlope), 0.0, 1.0);
                  float dPlainRay = mix(dCoastM, dFootM, uPlain);
                  float plainTex = 0.84 + 0.32 * noise2(vec2(azSouth * 58.0, uPlain * 14.0));
                  vec3 coastDayS = mix(vec3(0.044, 0.106, 0.088), vec3(0.068, 0.135, 0.098), plainTex * 0.6);
                  vec3 coastDuskS = mix(vec3(0.052, 0.054, 0.082), vec3(0.185, 0.108, 0.072), directBeamM * 0.65 + 0.22);
                  vec3 plainIntrinsicHDR = nexusSRGBToLinear(mix(mix(coastDayS, coastDuskS, uDuskFactor), forestNight * 0.92, uNightFactor));
                  float hazePlain = clamp(1.0 - exp(-dPlainRay * sigmaTropo * 1.15), 0.38, 0.82);
                  vec3 plainHDR = mix(plainIntrinsicHDR, hdrTropoSky, hazePlain);
                  javaInteriorHDR = mix(javaInteriorHDR, plainHDR, wPlainLayer);
                }

                hdrJavaLayer = javaInteriorHDR;
                javaSolidMask = clamp(anyJava * marineClarity, 0.0, 1.0);
              }

              // Subtle Dusk & Night Teluk Jakarta / Pantai Utara Jawa coastal glow & harbor lights anchored at visCoastSlope!
              float jakartaSpan = exp(-pow(azSouth / 0.34, 2.0)) * (1.0 - smoothstep(22000.0, 48000.0, dCoastM));
              float nightGlowPwr = clamp(uNightFactor * 0.95 + uDuskFactor * 0.38, 0.0, 1.0) * marineClarity;
              float relAboveCoast = raySlope - visCoastSlope;
              if (nightGlowPwr > 0.01 && relAboveCoast > -0.0015 && relAboveCoast < 0.030) {
                float domeH = exp(-max(0.0, relAboveCoast) / 0.0090) * jakartaSpan;
                hdrJavaLayer += nexusSRGBToLinear(vec3(0.24, 0.15, 0.06)) * domeH * nightGlowPwr * 0.16;
                if (relAboveCoast > -0.0004 && relAboveCoast < 0.0042) {
                  float citySpark = smoothstep(0.68, 0.95, noise2(vec2(azSouth * 420.0, relAboveCoast * 850.0)));
                  hdrJavaLayer += nexusSRGBToLinear(vec3(0.52, 0.36, 0.15)) * citySpark * jakartaSpan * nightGlowPwr * 0.24;
                }
              }
            }
          }

          // Composite Exoatmospheric Cosmos, Sun Disc, & Moon Disc BEHIND solid Java mountains & spherical Sea,
          // then composite foreground Spherical Sea & Geodetic Pulau Jawa cleanly!
          float solidEarthMask = clamp(max(seaSolidMask, javaSolidMask), 0.0, 1.0);
          vec3 hdrCelestial = hdrCosmos * (1.0 - moonOccultMask) * (cloudTrans * cloudTrans);
          if (cosSun > 0.0 && rNormSun < 1.0 && sunVis > 0.005) {
            float zSphS = sqrt(max(0.0, 1.0 - rNormSun * rNormSun));
            vec3 hdrSunBody = mix(vec3(3.4, 3.0, 2.3), vec3(3.1, 1.55, 0.68), uDuskFactor);
            hdrCelestial += hdrSunBody * (0.45 + 0.95 * zSphS) * sunVis * cloudTrans;
          }
          hdrCelestial += hdrMoonBody * cloudTrans;
          hdrCelestial += vec3(0.55, 0.68, 0.92) * moonTightCorona * (1.0 - moonOccultMask * 0.65) * (0.25 + 0.75 * cloudTrans);

          vec3 hdrSky = hdrTropoSky + hdrCelestial * (1.0 - solidEarthMask);
          if (seaSolidMask > 0.001) {
            hdrSky = mix(hdrSky, hdrSeaLayer, seaSolidMask);
          }
          if (javaSolidMask > 0.001) {
            hdrSky = mix(hdrSky, hdrJavaLayer, javaSolidMask);
          }

          vec3 hvsSkyLin = nexusApplyHumanEyeHDR(
            hdrSky,
            dir,
            celestialDir,
            uSunColor,
            uHvsParams,
            uHvsGlare
          );
          gl_FragColor = vec4(nexusLinearToSRGB(hvsSkyLin), 1.0);
        }
      `
    });

    this.skyMesh = new THREE.Mesh(skyGeo, skyMat);
    this.skyMesh.name = 'NexusAnalyticalSkyDome';
    this.skyMesh.frustumCulled = false;
    this.scene.add(this.skyMesh);
  }

  /**
   * Hukum Emas 8 & 12: 100% GPU Physics-Driven Raindrops (Marshall-Palmer + Gunn-Kinzer + Wind Shear)
   * & Newtonian Gravity Roof-Eave / Canopy Drip Curtains in STRICTLY 1 SINGLE DRAW CALL!
   */
  _initPhysicalRainAndDripMesh() {
    const skyCount = WeatherOpticsEngine.SKY_DROP_COUNT;
    const dripCount = WeatherOpticsEngine.EAVE_DRIP_COUNT;
    const totalQuads = skyCount + dripCount;
    const totalVerts = totalQuads * 4;

    const positions = new Float32Array(totalVerts * 3);
    const normals = new Float32Array(totalVerts * 3);
    const uvs = new Float32Array(totalVerts * 2);

    const aKind = new Float32Array(totalVerts); // 0 = Sky Marshall-Palmer Drop, 1 = Eave/Genting Drip & Limpahan
    const aDropDiam = new Float32Array(totalVerts); // Base Marshall-Palmer diameter seed in mm [0.65 .. 4.65]
    const aPhase = new Float32Array(totalVerts);
    const aAnchor = new Float32Array(totalVerts * 3); // Base (x, y, z) seed or exact 3D eave overhang coord
    const aLanding = new Float32Array(totalVerts * 2); // (landingY, isCanopy)
    const aOutDir = new Float32Array(totalVerts * 2); // Outward roof-eave overhang normal (outX, outZ) for genting limpahan
    const indices = new Uint32Array(totalQuads * 6);

    const cornerUVs = [
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0]
    ];

    for (let q = 0; q < totalQuads; q++) {
      const isDrip = q >= skyCount;
      // Deterministic low-discrepancy Halton-like sequence (0 NaN, 0 clustering)
      const u1 = ((q * 733 + 19) % 10007) / 10007;
      const u2 = ((q * 1499 + 53) % 10009) / 10009;
      const u3 = ((q * 2311 + 97) % 9973) / 9973;
      const u4 = ((q * 3571 + 131) % 9967) / 9967;

      // Marshall-Palmer exponential drop-size inverse CDF: N(D) = N_0 * exp(-Lambda * D)
      const diamMm = isDrip
        ? 3.2 + u4 * 1.35 // Pendant roof-tile/leaf drips are ~3.2-4.55 mm capillary drops
        : Math.min(4.65, 0.85 - Math.log(1.0 - u4 * 0.965) * 1.12);

      const ax = (u1 - 0.5) * 134.0;
      const ay = u2 * 58.0;
      const az = (u3 - 0.5) * 134.0;
      const outAng = u1 * Math.PI * 2.0;
      const defOx = Math.cos(outAng);
      const defOz = Math.sin(outAng);

      for (let c = 0; c < 4; c++) {
        const vi = q * 4 + c;
        positions[vi * 3] = 0.0;
        positions[vi * 3 + 1] = 0.0;
        positions[vi * 3 + 2] = 0.0;

        normals[vi * 3] = 0.0;
        normals[vi * 3 + 1] = 1.0;
        normals[vi * 3 + 2] = 0.0;

        uvs[vi * 2] = cornerUVs[c][0];
        uvs[vi * 2 + 1] = cornerUVs[c][1];

        aKind[vi] = isDrip ? 1.0 : 0.0;
        aDropDiam[vi] = diamMm;
        aPhase[vi] = u2;

        aAnchor[vi * 3] = ax;
        aAnchor[vi * 3 + 1] = isDrip ? 6.0 : ay;
        aAnchor[vi * 3 + 2] = az;

        aLanding[vi * 2] = 0.14;
        aLanding[vi * 2 + 1] = 0.0;

        aOutDir[vi * 2] = defOx;
        aOutDir[vi * 2 + 1] = defOz;
      }

      const b = q * 4;
      const io = q * 6;
      indices[io] = b;
      indices[io + 1] = b + 1;
      indices[io + 2] = b + 2;
      indices[io + 3] = b;
      indices[io + 4] = b + 2;
      indices[io + 5] = b + 3;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setAttribute('aKind', new THREE.BufferAttribute(aKind, 1));
    geo.setAttribute('aDropDiam', new THREE.BufferAttribute(aDropDiam, 1));
    geo.setAttribute('aPhase', new THREE.BufferAttribute(aPhase, 1));

    this._dripAnchorAttr = new THREE.BufferAttribute(aAnchor, 3);
    this._dripLandingAttr = new THREE.BufferAttribute(aLanding, 2);
    this._dripOutDirAttr = new THREE.BufferAttribute(aOutDir, 2);
    geo.setAttribute('aAnchor', this._dripAnchorAttr);
    geo.setAttribute('aLanding', this._dripLandingAttr);
    geo.setAttribute('aOutDir', this._dripOutDirAttr);
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
      uniforms: {
        uTime: this.uniforms.uTime,
        uCamPos: this.uniforms.uCamPos,
        uSunDir: this.uniforms.uSunDir,
        uNightFactor: this.uniforms.uNightFactor,
        uDuskFactor: this.uniforms.uDuskFactor,
        uRainIntensity: this.uniforms.uRainIntensity,
        uWetness: this.uniforms.uWetness,
        uLightningFlash: this.uniforms.uLightningFlash,
        uWindVec: this.uniforms.uWindVec,
        uWindPhase: this.uniforms.uWindPhase,
        uCollisionMap: this.uniforms.uCollisionMap,
        uCollisionHalfExtent: this.uniforms.uCollisionHalfExtent,
        uCollisionBounds: this.uniforms.uCollisionBounds,
        uCollisionHeightRange: this.uniforms.uCollisionHeightRange
      },
      vertexShader: /* glsl */ `
        attribute float aKind;
        attribute float aDropDiam;
        attribute float aPhase;
        attribute vec3 aAnchor;
        attribute vec2 aLanding;
        attribute vec2 aOutDir;

        uniform float uTime;
        uniform vec3 uCamPos;
        uniform float uRainIntensity;
        uniform float uWetness;
        uniform vec3 uWindVec;
        uniform vec2 uWindPhase;
        uniform sampler2D uCollisionMap;
        uniform float uCollisionHalfExtent;
        uniform vec4 uCollisionBounds;
        uniform vec2 uCollisionHeightRange;

        varying vec2 vUv;
        varying float vAlpha;
        varying float vBeadness;

        ${WeatherOpticsEngine._getArchipelagoSdfGLSL()}

        float decodeCollisionHeight(vec2 xz) {
          float hOut = 0.0;
          vec2 cuv = (xz - uCollisionBounds.xy) / uCollisionBounds.zw;
          if (cuv.x < 0.001 || cuv.x > 0.999 || cuv.y < 0.001 || cuv.y > 0.999) {
            float dCoast = evalArchipelagoCoastRaw(xz);
            hOut = evalArchipelagoAnalyticalHeight(xz, dCoast);
          } else {
            vec4 cTex = texture2D(uCollisionMap, cuv);
            hOut = uCollisionHeightRange.x + ((cTex.r * 255.0 * 256.0 + cTex.g * 255.0) / 65535.0) * uCollisionHeightRange.y;
          }
          return hOut;
        }

        void main() {
          vUv = uv;
          vAlpha = 0.0;
          vBeadness = 0.0;

          // Cull helper
          vec4 culledClip = vec4(2.0, 2.0, 2.0, 1.0);

          if (aKind < 0.5) {
            // --- MODE A: EXACT 1:1 WORLD-SCALE MARSHALL-PALMER + GUNN-KINZER SKY RAINDROP ---
            // Filter drop spectrum by rain intensity (drizzle = fine ~0.8-1.4mm drops at ~3.4-5.2 m/s,
            // heavy tropical rain/storm = ~1.8-4.2mm drops at ~6.2-8.7 m/s)
            float activeGate = fract(aPhase * 31.73);
            if (activeGate > clamp(uRainIntensity * 0.92, 0.0, 1.0)) {
              gl_Position = culledClip;
              return;
            }
            if (aDropDiam > 1.55 + uRainIntensity * 2.9) {
              gl_Position = culledClip;
              return;
            }

            float rainNorm = clamp(uRainIntensity / 1.40, 0.08, 1.05);
            float effDiam = clamp(
              mix(0.68, 1.65, rainNorm) + (aDropDiam - 0.85) * mix(0.28, 0.92, rainNorm),
              0.55,
              4.65
            );

            // Exact 1:1 Meter/Sec Gunn-Kinzer (1949) terminal velocity: v_t(D) = 9.65 - 10.3 * exp(-0.6 * D_mm) m/s
            // (Zero artificial speed inflation: 1 world unit === 1.000 physical meter!)
            float vTerminal = clamp(9.65 - 10.3 * exp(-0.6 * effDiam), 2.55, 8.85);

            // Wrap 134m x 134m rain volume smoothly around camera horizontal position
            float boxSpanXZ = 134.0;
            float halfSpanXZ = 67.0;
            float colMinY = -2.5;
            float colSpanY = 58.5;

            float fallDist = uTime * vTerminal + aPhase * colSpanY;
            float cycleId = floor((aAnchor.y - fallDist) / colSpanY);
            float pyTop = colMinY + mod(aAnchor.y - fallDist, colSpanY);

            // Slight per-cycle horizontal decorrelation so drops never repeat on exact same vertical line
            float jx = sin(cycleId * 12.9898 + aPhase * 78.233) * 2.8;
            float jz = cos(cycleId * 39.346 + aPhase * 41.135) * 2.8;

            // Altitude boundary-layer wind shear W(y) ~ (y / 10)^0.22 + monotonic uWindPhase advection (0 gust-slosh!)
            float altFactor = pow(clamp(max(1.5, pyTop) / 10.0, 0.35, 1.75), 0.22);
            float dragCouple = mix(1.18, 0.78, clamp((effDiam - 0.65) / 3.9, 0.0, 1.0));

            vec2 baseXZ = vec2(
              uCamPos.x + mod(aAnchor.x + jx + uWindPhase.x * 0.85 - uCamPos.x + halfSpanXZ, boxSpanXZ) - halfSpanXZ,
              uCamPos.z + mod(aAnchor.z + jz + uWindPhase.y * 0.85 - uCamPos.z + halfSpanXZ, boxSpanXZ) - halfSpanXZ
            );

            float gust = 0.88 + 0.18 * sin(uTime * 1.65 - dot(baseXZ, vec2(0.055, 0.045)));
            vec2 windHoriz = uWindVec.xz * altFactor * dragCouple * gust * 0.96;
            vec3 vel3D = vec3(windHoriz.x, -vTerminal, windHoriz.y);
            vec3 streakDir = normalize(vel3D);

            // Human-eye persistence streak (~1/37s = 27ms natural foveal integration):
            // 0.09m - 0.26m streak length proportional to 1.70m humans, 0.25m cats, & genting roof tiles!
            float speed3D = length(vel3D);
            float streakLen = clamp(speed3D * 0.027, 0.075, 0.34);

            vec3 pHead = vec3(baseXZ.x, pyTop, baseXZ.y) + streakDir * streakLen;
            vec3 pTail = vec3(baseXZ.x, pyTop, baseXZ.y);

            // Exact 3D Scene Collision Interception (Roofs, Eaves, Walkways, Tree Canopies, & Ground!)
            float yHit = decodeCollisionHeight(pHead.xz);
            if (pTail.y <= yHit + 0.02) {
              // Entire raindrop is already below the intercepting 3D roof/canopy/ground -> 100% dry interior!
              gl_Position = culledClip;
              return;
            }
            if (pHead.y < yHit) {
              // Clip the leading tip cleanly right onto the impact surface!
              float frac = clamp((pTail.y - yHit) / max(0.02, pTail.y - pHead.y), 0.05, 1.0);
              pHead = mix(pTail, pHead, frac);
            }

            vec3 midP = 0.5 * (pHead + pTail);
            vec3 toCam = uCamPos - midP;
            float distCam = length(toCam) + 1e-4;
            if (distCam > 82.0) {
              gl_Position = culledClip;
              return;
            }

            vec3 sideDir = normalize(cross(toCam / distCam, streakDir) + vec3(1e-5, 0.0, 0.0));
            // Physical millimeter drop width + subtle anti-aliasing angular floor at distance
            float halfW = clamp(0.0032 + effDiam * 0.0020 + distCam * 0.00034, 0.0035, 0.018);

            // uv.y = 0 -> upper wake tail; uv.y = 1 -> lower leading meniscus head
            vec3 worldVert = mix(pTail, pHead, uv.y) + sideDir * ((uv.x - 0.5) * 2.0 * halfW);
            float distFade = smoothstep(82.0, 26.0, distCam) * smoothstep(0.6, 2.2, distCam);
            float sizeAlpha = mix(0.24, 0.58, clamp((effDiam - 0.65) / 3.8, 0.0, 1.0));
            vAlpha = sizeAlpha * distFade * clamp(uRainIntensity * 0.96, 0.28, 1.15);
            vBeadness = 0.16;

            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);
          } else {
            // --- MODE B: AERODYNAMIC-DRAG-CAPPED GENTING CAPILLARY DRIPS (TETESAN)
            //             & HEAVY-RAIN PARABOLIC TILE-VALLEY RUNOFF STREAMS (LIMPAHAN) ---
            // Eaves keep dripping residual runoff naturally even as rain tapers!
            float runoffLevel = max(uRainIntensity, uWetness * 0.45);
            float dripGate = fract(aPhase * 19.37);
            if (runoffLevel < 0.08 || dripGate > clamp(runoffLevel * 1.05, 0.0, 1.0)) {
              gl_Position = culledClip;
              return;
            }

            vec3 eavePos = aAnchor;
            float distCam = length(uCamPos - eavePos);
            if (distCam > 78.0) {
              gl_Position = culledClip;
              return;
            }

            float landingY = aLanding.x;
            float isCanopy = aLanding.y;
            float isHardRoof = 1.0 - step(0.5, isCanopy);
            float dropHeight = max(0.35, eavePos.y - landingY);

            // Exact Aerodynamic-Drag-Capped Free-Fall Kinematics (g = 9.81 m/s^2, v_t = 8.35 m/s for ~3.8mm pendant drop):
            // dy(t) = (v_t^2 / g) * ln(cosh(g*t / v_t)), vy(t) = v_t * tanh(g*t / v_t)
            // Overflow-safe GLSL ES 1.00 closed-form fall duration u_max = acosh(exp(w)):
            float g = 9.81;
            float vTermDrip = 8.35;
            float wFall = (g * dropHeight) / (vTermDrip * vTermDrip);
            float uFallMax = wFall + log(1.0 + sqrt(max(0.0, 1.0 - exp(-2.0 * wFall))));
            float tFall = (vTermDrip / g) * uFallMax;

            // Dual Hydrological Regime on Architectural Genting Eaves:
            // 1) Tetesan Genting (light rain / post-rain): capillary pendant bead hangs under terracotta lip (tHold ~ 0.20-0.56s)
            // 2) Limpahan Cucuran Genting (moderate-to-heavy rain): continuous tile-valley runoff stream packets
            //    shooting off the roof pitch direction (aOutDir) in a realistic parabolic arc!
            float isRunoffSlot = isHardRoof * step(0.42, fract(aPhase * 7.13));
            float limpahanStrength = isRunoffSlot * smoothstep(0.28, 0.82, max(uRainIntensity * 0.92, runoffLevel * 0.85));

            float tHold = mix(
              mix(0.56, 0.20, clamp(runoffLevel, 0.0, 1.0)) + isCanopy * 0.14,
              0.024,
              limpahanStrength
            );
            float tPause = mix(
              mix(0.50, 0.10, clamp(runoffLevel, 0.0, 1.0)),
              0.042,
              limpahanStrength
            );
            float period = tHold + tFall + tPause;

            float tau = mod(uTime + aPhase * period * 3.7, period);
            if (tau > tHold + tFall) {
              gl_Position = culledClip;
              return;
            }

            vec3 pHead;
            vec3 pTail;
            vec3 streakDir = vec3(0.0, -1.0, 0.0);

            if (tau < tHold) {
              // Growing capillary pendant water bead hanging under terracotta tile lip / leaf tip!
              float grow = tau / max(0.01, tHold);
              float beadR = 0.011 + 0.013 * grow;
              pTail = eavePos + vec3(0.0, 0.005, 0.0);
              pHead = eavePos - vec3(0.0, beadR * 2.2, 0.0);
              vBeadness = 1.0;
            } else {
              // Detached drag-coupled gravity free-fall + parabolic outward genting runoff pitch trajectory:
              float tf = tau - tHold;
              float uG = (g * tf) / vTermDrip;
              float eNeg2u = exp(-2.0 * uG);
              float dy = ((vTermDrip * vTermDrip) / g) * (uG + log(0.5 * (1.0 + eNeg2u)));
              float vy = vTermDrip * ((1.0 - eNeg2u) / (1.0 + eNeg2u));

              vec2 outDirN = length(aOutDir) > 0.05 ? normalize(aOutDir) : vec2(0.0, 1.0);
              float vOut0 = limpahanStrength * (0.34 + 0.54 * clamp(uRainIntensity, 0.0, 1.35)) * (0.85 + 0.30 * fract(aPhase * 13.7));
              float dragTauH = 1.35;
              float outDisp = vOut0 * dragTauH * (1.0 - exp(-tf / dragTauH));
              float vOutNow = vOut0 * exp(-tf / dragTauH);
              vec2 windDrift = uWindVec.xz * (0.11 * tf * tf / (1.0 + 0.35 * tf));
              vec2 horizDisp = outDirN * outDisp + windDrift;

              vec3 vel = vec3(
                outDirN.x * vOutNow + uWindVec.x * 0.22 * min(1.5, tf),
                -max(0.35, vy),
                outDirN.y * vOutNow + uWindVec.z * 0.22 * min(1.5, tf)
              );
              streakDir = normalize(vel);

              float rawStreakLen = clamp(0.032 + length(vel) * mix(0.025, 0.038, limpahanStrength), 0.032, 0.36);
              float streakLen = min(rawStreakLen, max(0.020, dy + 0.016));
              pHead = vec3(eavePos.x + horizDisp.x, max(landingY, eavePos.y - dy), eavePos.z + horizDisp.y);
              pTail = pHead - streakDir * streakLen;
              if (pTail.y < landingY + 0.01) {
                gl_Position = culledClip;
                return;
              }
              vBeadness = mix(exp(-tf * 4.5), 0.22, limpahanStrength);
            }

            vec3 midP = 0.5 * (pHead + pTail);
            vec3 toCam = normalize(uCamPos - midP + vec3(1e-5));
            vec3 sideDir = normalize(cross(toCam, streakDir) + vec3(1e-5, 0.0, 0.0));
            float halfW = max(
              mix(0.0085, 0.0155, max(vBeadness, limpahanStrength * 0.72)),
              distCam * 0.00046
            );

            vec3 worldVert = mix(pTail, pHead, uv.y) + sideDir * ((uv.x - 0.5) * 2.0 * halfW);
            vAlpha = smoothstep(78.0, 22.0, distCam) * mix(0.50, 0.78, max(vBeadness, limpahanStrength * 0.65));

            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);
          }
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec2 vUv;
        varying float vAlpha;
        varying float vBeadness;

        uniform float uNightFactor;
        uniform float uDuskFactor;
        uniform float uLightningFlash;

        void main() {
          if (vAlpha < 0.01) discard;

          // Cylindrical refractive cross-section across drop width x in [-1, +1]
          float x = (vUv.x - 0.5) * 2.0;
          float radialMask = sqrt(max(0.0, 1.0 - x * x));

          // Tear-drop / motion-blurred axial profile: denser meniscus at bottom head (vUv.y -> 1), soft wake tail (vUv.y -> 0)
          float axialStreak = smoothstep(0.0, 0.42, vUv.y) * smoothstep(1.0, 0.84, vUv.y);
          float beadDist = length(vec2(x, (vUv.y - 0.58) * 1.9));
          float axialBead = smoothstep(1.0, 0.15, beadDist);
          float shape = mix(axialStreak * radialMask, axialBead, vBeadness);

          if (shape < 0.03) discard;

          // Physical water-drop optics (n = 1.333): Fresnel rim + central internal caustic focal line + lower meniscus glint
          float fresnelRim = pow(1.0 - radialMask, 2.4) * 0.65;
          float causticCore = exp(-x * x * 10.0) * (0.45 + 0.55 * smoothstep(0.45, 0.92, vUv.y));

          vec3 dayTint = mix(vec3(0.76, 0.85, 0.94), vec3(0.94, 0.72, 0.56), uDuskFactor * 0.65);
          // At night, raindrops catch warm campus/window glow + cool moonlit/lightning refraction
          vec3 nightTint = vec3(0.46, 0.56, 0.72);
          vec3 dropCol = mix(dayTint, nightTint, uNightFactor * 0.72);
          dropCol += vec3(0.24, 0.28, 0.34) * (fresnelRim + causticCore) * (1.0 - 0.4 * uNightFactor);
          dropCol += vec3(0.82, 0.90, 1.0) * uLightningFlash * 0.95;

          float outAlpha = clamp(shape * vAlpha * (0.68 + 0.45 * causticCore), 0.0, 0.88);
          gl_FragColor = vec4(dropCol, outAlpha);
        }
      `
    });

    this.rainAndDripMesh = new THREE.Mesh(geo, mat);
    this.rainAndDripMesh.name = 'NexusPhysicalRainAndDrips';
    this.rainAndDripMesh.userData.dynamicKinematic = true;
    this.rainAndDripMesh.frustumCulled = false;
    this.rainAndDripMesh.visible = false;
    this.scene.add(this.rainAndDripMesh);
  }

  /**
   * Hukum Emas 8 & 12: 100% GPU 3D Surface-Normal Impact Crown Splashes +
   * Newtonian Parabolic Ballistic Rebound Micro-Droplets + Roof Aerosol Spray Mist
   * in STRICTLY 1 SINGLE DRAW CALL!
   */
  _initPhysicalSplashAndMistMesh() {
    const crownCount = WeatherOpticsEngine.SPLASH_CROWN_COUNT;
    const reboundCount = WeatherOpticsEngine.REBOUND_DROP_COUNT;
    const mistCount = WeatherOpticsEngine.SPRAY_MIST_COUNT;
    const totalQuads = crownCount + reboundCount + mistCount;
    const totalVerts = totalQuads * 4;

    const positions = new Float32Array(totalVerts * 3);
    const normals = new Float32Array(totalVerts * 3);
    const uvs = new Float32Array(totalVerts * 2);

    // aSplashMeta: (type: 0=crown, 1=parabolic rebound, 2=aerosol mist, phase, randAzimuth, randSpeed)
    const aSplashMeta = new Float32Array(totalVerts * 4);
    // aSplashAnchor: (xOffsetOrWorldX, zOffsetOrWorldZ, isWorldPinned)
    const aSplashAnchor = new Float32Array(totalVerts * 3);
    const indices = new Uint32Array(totalQuads * 6);

    const cornerUVs = [
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0]
    ];

    for (let q = 0; q < totalQuads; q++) {
      let sType = 0.0;
      if (q >= crownCount && q < crownCount + reboundCount) sType = 1.0;
      else if (q >= crownCount + reboundCount) sType = 2.0;

      // Pair sets of 2 rebound droplets to share nearby impact origins for realistic multi-droplet shatter!
      const seedIdx = sType === 1.0 ? Math.floor((q - crownCount) / 2) + 5000 : q;
      const u1 = ((seedIdx * 887 + 31) % 10007) / 10007;
      const u2 = ((seedIdx * 1627 + 71) % 10009) / 10009;
      const uPhase = ((seedIdx * 2539 + 113) % 9973) / 9973;

      const uAz = ((q * 3301 + 47) % 9967) / 9967;
      const uSpd = ((q * 4129 + 89) % 9949) / 9949;

      const span = sType === 2.0 ? 96.0 : 62.0;
      const ax = (u1 - 0.5) * span;
      const az = (u2 - 0.5) * span;

      for (let c = 0; c < 4; c++) {
        const vi = q * 4 + c;
        positions[vi * 3] = 0.0;
        positions[vi * 3 + 1] = 0.0;
        positions[vi * 3 + 2] = 0.0;

        normals[vi * 3] = 0.0;
        normals[vi * 3 + 1] = 1.0;
        normals[vi * 3 + 2] = 0.0;

        uvs[vi * 2] = cornerUVs[c][0];
        uvs[vi * 2 + 1] = cornerUVs[c][1];

        aSplashMeta[vi * 4] = sType;
        aSplashMeta[vi * 4 + 1] = uPhase;
        aSplashMeta[vi * 4 + 2] = uAz * Math.PI * 2.0;
        aSplashMeta[vi * 4 + 3] = uSpd;

        aSplashAnchor[vi * 3] = ax;
        aSplashAnchor[vi * 3 + 1] = az;
        aSplashAnchor[vi * 3 + 2] = 0.0;
      }

      const b = q * 4;
      const io = q * 6;
      indices[io] = b;
      indices[io + 1] = b + 1;
      indices[io + 2] = b + 2;
      indices[io + 3] = b;
      indices[io + 4] = b + 2;
      indices[io + 5] = b + 3;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setAttribute('aSplashMeta', new THREE.BufferAttribute(aSplashMeta, 4));
    this._splashAnchorAttr = new THREE.BufferAttribute(aSplashAnchor, 3);
    geo.setAttribute('aSplashAnchor', this._splashAnchorAttr);
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
      uniforms: {
        uTime: this.uniforms.uTime,
        uCamPos: this.uniforms.uCamPos,
        uNightFactor: this.uniforms.uNightFactor,
        uDuskFactor: this.uniforms.uDuskFactor,
        uRainIntensity: this.uniforms.uRainIntensity,
        uLightningFlash: this.uniforms.uLightningFlash,
        uWindVec: this.uniforms.uWindVec,
        uCollisionMap: this.uniforms.uCollisionMap,
        uCollisionHalfExtent: this.uniforms.uCollisionHalfExtent,
        uCollisionBounds: this.uniforms.uCollisionBounds,
        uCollisionHeightRange: this.uniforms.uCollisionHeightRange
      },
      vertexShader: /* glsl */ `
        attribute vec4 aSplashMeta;
        attribute vec3 aSplashAnchor;

        uniform float uTime;
        uniform vec3 uCamPos;
        uniform float uRainIntensity;
        uniform vec3 uWindVec;
        uniform sampler2D uCollisionMap;
        uniform float uCollisionHalfExtent;
        uniform vec4 uCollisionBounds;
        uniform vec2 uCollisionHeightRange;

        varying vec2 vUv;
        varying float vType;
        varying float vLife;
        varying float vAlpha;
        varying float vSurfClass;

        ${WeatherOpticsEngine._getArchipelagoSdfGLSL()}

        // Decodes yHit, exact 3D surface normal N_hit, and surfaceClass (0=Water, 1=FlatHard, 2=SlopedRoof, 3=Canopy)
        void sampleCollisionSurface(vec2 xz, out float yHit, out vec3 nHit, out float surfClass) {
          vec2 cuv = (xz - uCollisionBounds.xy) / uCollisionBounds.zw;
          if (cuv.x < 0.001 || cuv.x > 0.999 || cuv.y < 0.001 || cuv.y > 0.999) {
            float dCoast = evalArchipelagoCoastRaw(xz);
            if (dCoast > 3.52) {
              yHit = -1.45;
              nHit = vec3(0.0, 1.0, 0.0);
              surfClass = 0.0; // Ocean / Teluk U / Selat UI Water
            } else {
              yHit = evalArchipelagoAnalyticalHeight(xz, dCoast);
              float epsH = 1.25;
              float hx = evalArchipelagoAnalyticalHeight(xz + vec2(epsH, 0.0), evalArchipelagoCoastRaw(xz + vec2(epsH, 0.0)));
              float hz = evalArchipelagoAnalyticalHeight(xz + vec2(0.0, epsH), evalArchipelagoCoastRaw(xz + vec2(0.0, epsH)));
              nHit = normalize(vec3(yHit - hx, epsH, yHit - hz));
              surfClass = nHit.y < 0.88 ? 2.0 : 1.0;
            }
            return;
          }
          vec4 cTex = texture2D(uCollisionMap, cuv);
          yHit = uCollisionHeightRange.x + ((cTex.r * 255.0 * 256.0 + cTex.g * 255.0) / 65535.0) * uCollisionHeightRange.y;
          float aByte = cTex.a * 255.0;
          if (aByte < 5.0) {
            surfClass = 0.0; // Danau Kenanga Water
            nHit = vec3(0.0, 1.0, 0.0);
          } else if (aByte > 251.0) {
            surfClass = 3.0; // Botanical Tree Canopy
            float nx = cTex.b * 2.0 - 1.0;
            float ny = sqrt(max(0.2, 1.0 - nx * nx));
            nHit = normalize(vec3(nx, ny, 0.0));
          } else {
            float nx = cTex.b * 2.0 - 1.0;
            float nz = clamp((aByte - 128.0) / 116.0, -1.0, 1.0);
            float ny = sqrt(max(0.04, 1.0 - nx * nx - nz * nz));
            nHit = normalize(vec3(nx, ny, nz));
            surfClass = ny < 0.88 ? 2.0 : 1.0;
          }
        }

        void main() {
          vUv = uv;
          float sType = aSplashMeta.x;
          float phase = aSplashMeta.y;
          float azAngle = aSplashMeta.z;
          float rndSpd = aSplashMeta.w;
          vType = sType;
          vAlpha = 0.0;
          vLife = 0.0;
          vSurfClass = 1.0;

          vec4 culledClip = vec4(2.0, 2.0, 2.0, 1.0);

          float gate = fract(phase * 23.41 + rndSpd * 7.1);
          if (gate > clamp(uRainIntensity * 0.98, 0.0, 1.0)) {
            gl_Position = culledClip;
            return;
          }

          if (sType < 0.5) {
            // --- SUB-POPULATION A: 3D SURFACE-NORMAL RAYLEIGH-WORTHINGTON CROWN SPLASH ---
            float period = mix(0.46, 0.28, rndSpd);
            float tLocal = uTime + phase * 11.3;
            float cycle = floor(tLocal / period);
            float life = fract(tLocal / period);
            vLife = life;

            vec2 hitXZ;
            if (aSplashAnchor.z > 0.5) {
              // World-pinned along extracted 3D roof-eave drip landing lines!
              vec2 j = vec2(sin(cycle * 17.3 + phase * 43.1), cos(cycle * 29.1 + phase * 19.7)) * 0.26;
              hitXZ = aSplashAnchor.xy + j;
            } else {
              // Camera-wrapped 62m x 62m high-density splash field
              float span = 62.0;
              float halfS = 31.0;
              vec2 j = vec2(sin(cycle * 13.7 + phase * 53.9), cos(cycle * 31.7 + phase * 27.3)) * 1.9;
              hitXZ = vec2(
                uCamPos.x + mod(aSplashAnchor.x + j.x - uCamPos.x + halfS, span) - halfS,
                uCamPos.z + mod(aSplashAnchor.y + j.y - uCamPos.z + halfS, span) - halfS
              );
            }

            float yHit;
            vec3 nHit;
            float surfClass;
            sampleCollisionSurface(hitXZ, yHit, nHit, surfClass);
            vSurfClass = surfClass;

            vec3 hitPos = vec3(hitXZ.x, yHit, hitXZ.y) + nHit * 0.022;
            float distCam = length(uCamPos - hitPos);
            if (distCam > 48.0) {
              gl_Position = culledClip;
              return;
            }

            // Orthonormal tangent plane (t1, t2) strictly perpendicular to 3D surface normal nHit!
            vec3 upRef = abs(nHit.y) < 0.93 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
            vec3 t1 = normalize(cross(upRef, nHit));
            vec3 t2 = cross(nHit, t1);

            float maxRad = surfClass < 0.5
              ? mix(0.16, 0.34, rndSpd)  // Larger capillary wave ring on Danau Kenanga water
              : mix(0.085, 0.19, rndSpd) * (0.85 + 0.25 * min(1.4, uRainIntensity));
            float rad = maxRad * (0.20 + 0.80 * pow(life, 0.62));
            // Upward Rayleigh liquid crown lift during early impact [0 .. 0.35]
            float crownLift = sin(min(1.0, life * 2.4) * 3.14159) * maxRad * (surfClass < 0.5 ? 0.18 : 0.36);

            vec2 qOff = (uv - 0.5) * 2.0 * rad;
            vec3 worldVert = hitPos + t1 * qOff.x + t2 * qOff.y + nHit * crownLift;

            vAlpha = (1.0 - life) * smoothstep(0.0, 0.08, life) * smoothstep(48.0, 14.0, distCam)
                   * mix(0.58, 0.85, clamp(uRainIntensity, 0.0, 1.0));
            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);

          } else if (sType < 1.5) {
            // --- SUB-POPULATION B: NEWTONIAN PARABOLIC BALLISTIC REBOUND MICRO-DROPLETS ---
            // P(tau) = P_hit + v_0 * tau + 0.5 * g * tau^2
            float period = mix(0.42, 0.26, rndSpd);
            float tLocal = uTime + phase * 11.3;
            float cycle = floor(tLocal / period);
            float life = fract(tLocal / period);
            float tau = life * period;
            vLife = life;

            float span = 50.0;
            float halfS = 25.0;
            vec2 j = vec2(sin(cycle * 13.7 + phase * 53.9), cos(cycle * 31.7 + phase * 27.3)) * 1.9;
            vec2 hitXZ = vec2(
              uCamPos.x + mod(aSplashAnchor.x + j.x - uCamPos.x + halfS, span) - halfS,
              uCamPos.z + mod(aSplashAnchor.y + j.y - uCamPos.z + halfS, span) - halfS
            );

            float yHit;
            vec3 nHit;
            float surfClass;
            sampleCollisionSurface(hitXZ, yHit, nHit, surfClass);
            vSurfClass = surfClass;

            // Soft foliage canopies absorb momentum with fewer ballistic rebounds than hard tiles/pavers
            if (surfClass > 2.5 && rndSpd < 0.65) {
              gl_Position = culledClip;
              return;
            }

            vec3 hitPos = vec3(hitXZ.x, yHit + 0.015, hitXZ.y);
            float distCam = length(uCamPos - hitPos);
            if (distCam > 38.0) {
              gl_Position = culledClip;
              return;
            }

            vec3 upRef = abs(nHit.y) < 0.93 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
            vec3 t1 = normalize(cross(upRef, nHit));
            vec3 t2 = cross(nHit, t1);

            // Ejection cone around 3D surface normal N_hit (tilts down-slope automatically on 48.7° Tajug roof!)
            float coneAngle = mix(0.38, 0.98, fract(rndSpd * 13.7));
            float vMag = mix(1.35, 3.05, rndSpd) * (0.78 + 0.25 * min(1.4, uRainIntensity));
            float az = azAngle + cycle * 1.7;
            vec3 ejectDir = normalize(
              nHit * cos(coneAngle) + (t1 * cos(az) + t2 * sin(az)) * sin(coneAngle)
            );
            vec3 v0 = ejectDir * vMag + vec3(uWindVec.x * 0.25, 0.0, uWindVec.z * 0.25);
            vec3 gVec = vec3(0.0, -9.81, 0.0);

            vec3 posTau = hitPos + v0 * tau + 0.5 * gVec * tau * tau;
            if (tau > 0.05 && posTau.y < yHit + 0.005) {
              gl_Position = culledClip;
              return;
            }

            vec3 velTau = v0 + gVec * tau;
            vec3 dirTau = normalize(velTau + vec3(0.0, 1e-4, 0.0));
            vec3 toCam = normalize(uCamPos - posTau + vec3(1e-5));
            vec3 sideDir = normalize(cross(toCam, dirTau) + vec3(1e-5, 0.0, 0.0));

            float dropLen = clamp(0.022 + length(velTau) * 0.014, 0.020, 0.085);
            float dropW = max(0.010, distCam * 0.00048);

            vec3 worldVert = posTau + dirTau * ((uv.y - 0.5) * dropLen) + sideDir * ((uv.x - 0.5) * 2.0 * dropW);
            vAlpha = (1.0 - life * 0.85) * smoothstep(38.0, 10.0, distCam) * 0.78;
            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);

          } else {
            // --- SUB-POPULATION C: IMPACT AEROSOL SPRAY MIST HOVERING OVER ROOFS & COURTYARDS ---
            if (uRainIntensity < 0.35) {
              gl_Position = culledClip;
              return;
            }

            float period = mix(3.4, 2.2, rndSpd);
            float tLocal = uTime + phase * 17.9;
            float cycle = floor(tLocal / period);
            float life = fract(tLocal / period);
            vLife = life;

            float span = 96.0;
            float halfS = 48.0;
            vec2 baseXZ = vec2(
              uCamPos.x + mod(aSplashAnchor.x + sin(cycle * 9.1) * 3.0 - uCamPos.x + halfS, span) - halfS,
              uCamPos.z + mod(aSplashAnchor.y + cos(cycle * 7.3) * 3.0 - uCamPos.z + halfS, span) - halfS
            );

            float yHit;
            vec3 nHit;
            float surfClass;
            sampleCollisionSurface(baseXZ, yHit, nHit, surfClass);

            // Spray mist rises 0.45..1.65m above wet terracotta roofs & stone plazas and drifts with wind
            vec2 drift = uWindVec.xz * (life * 2.4);
            float puffRadius = mix(1.15, 2.45, rndSpd) * (0.75 + 0.45 * life);
            vec3 center = vec3(baseXZ.x + drift.x, yHit + 0.55 + life * 0.95, baseXZ.y + drift.y);

            float distCam = length(uCamPos - center);
            if (distCam > 88.0 || distCam < 3.5) {
              gl_Position = culledClip;
              return;
            }

            // Camera-facing billboard quad
            vec3 forward = normalize(uCamPos - center);
            vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward) + vec3(1e-5, 0.0, 0.0));
            vec3 up = cross(forward, right);

            vec2 qOff = (uv - 0.5) * 2.0 * puffRadius;
            vec3 worldVert = center + right * qOff.x + up * qOff.y;

            float env = sin(life * 3.14159);
            vAlpha = env * smoothstep(88.0, 28.0, distCam) * smoothstep(3.5, 9.5, distCam)
                   * clamp((uRainIntensity - 0.30) * 0.13, 0.0, 0.14);
            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);
          }
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec2 vUv;
        varying float vType;
        varying float vLife;
        varying float vAlpha;
        varying float vSurfClass;

        uniform float uNightFactor;
        uniform float uDuskFactor;
        uniform float uLightningFlash;

        void main() {
          if (vAlpha < 0.004) discard;

          vec2 p = (vUv - 0.5) * 2.0;
          float r = length(p);
          if (r > 1.0) discard;

          vec3 baseCol = mix(vec3(0.84, 0.90, 0.97), vec3(0.94, 0.72, 0.56), uDuskFactor * 0.6);
          baseCol = mix(baseCol, vec3(0.45, 0.55, 0.70), uNightFactor * 0.72);
          baseCol += vec3(0.82, 0.90, 1.0) * uLightningFlash * 0.95;

          if (vType < 0.5) {
            // Rayleigh-Worthington Hollow Crown Ring + Inner Capillary Ripple + 7-Lobe Rim Beading
            float theta = atan(p.y, p.x);
            float outerRing = exp(-pow((r - 0.82) * 9.5, 2.0));
            float innerRing = exp(-pow((r - 0.44 * vLife) * 11.0, 2.0)) * 0.45;
            float crownBeads = 0.68 + 0.32 * cos(theta * 7.0 + vLife * 6.0);
            float mask = (outerRing * crownBeads + innerRing);
            if (mask < 0.04) discard;
            gl_FragColor = vec4(baseCol, clamp(mask * vAlpha, 0.0, 0.82));
          } else if (vType < 1.5) {
            // Glistening Spherical/Ellipsoidal Ballistic Rebound Micro-Droplet
            float core = smoothstep(1.0, 0.1, r);
            float spec = exp(-r * r * 5.0) * 0.45;
            gl_FragColor = vec4(baseCol + vec3(spec), clamp(core * vAlpha, 0.0, 0.85));
          } else {
            // Soft Gaussian Impact Aerosol Spray Mist
            float puff = exp(-r * r * 3.4) * (1.0 - r * r);
            gl_FragColor = vec4(baseCol, clamp(puff * vAlpha, 0.0, 0.16));
          }
        }
      `
    });

    this.splashAndMistMesh = new THREE.Mesh(geo, mat);
    this.splashAndMistMesh.name = 'NexusPhysicalSplashAndMist';
    this.splashAndMistMesh.userData.dynamicKinematic = true;
    this.splashAndMistMesh.frustumCulled = false;
    this.splashAndMistMesh.visible = false;
    this.scene.add(this.splashAndMistMesh);
  }

  /**
   * Hukum Emas 12 & 14: 100% GPU Vertex Lakeside Fireflies / Evening Spores around Danau Kenanga
   */
  _initGPUFireflyParticles() {
    const count = 180;
    const positions = new Float32Array(count * 3);
    const speeds = new Float32Array(count);
    const phases = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2.0;
      const rad = 28.0 + ((i * 37) % 52);
      positions[i * 3] = 52.0 + Math.abs(Math.cos(angle)) * rad;
      positions[i * 3 + 1] = 1.2 + ((i * 19) % 45) * 0.12;
      positions[i * 3 + 2] = Math.sin(angle) * 75.0;
      speeds[i] = 0.6 + ((i * 29) % 10) * 0.12;
      phases[i] = ((i * 61) % 100) * 0.0628;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('aSpeed', new THREE.BufferAttribute(speeds, 1));
    geo.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: this.uniforms.uTime,
        uFireflyIntensity: this.uniforms.uFireflyIntensity
      },
      vertexShader: /* glsl */ `
        attribute float aSpeed;
        attribute float aPhase;
        uniform float uTime;
        uniform float uFireflyIntensity;
        varying float vGlow;

        void main() {
          vec3 p = position;
          p.x += sin(uTime * aSpeed + aPhase) * 1.8;
          p.y += sin(uTime * aSpeed * 1.4 + aPhase * 2.0) * 0.65;
          p.z += cos(uTime * aSpeed * 0.9 + aPhase) * 1.8;

          vGlow = uFireflyIntensity * (0.35 + 0.65 * max(0.0, sin(uTime * 2.6 + aPhase * 3.0)));
          vec4 mvPos = modelViewMatrix * vec4(p, 1.0);
          gl_PointSize = clamp(48.0 / max(1.0, -mvPos.z), 2.0, 9.0);
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: /* glsl */ `
        varying float vGlow;
        void main() {
          if (vGlow < 0.02) discard;
          float d = length(gl_PointCoord - vec2(0.5));
          if (d > 0.5) discard;
          float radial = smoothstep(0.5, 0.0, d);
          gl_FragColor = vec4(1.0, 0.86, 0.35, radial * vGlow);
        }
      `
    });

    this.fireflyPoints = new THREE.Points(geo, mat);
    this.fireflyPoints.name = 'NexusGPUFireflies';
    this.fireflyPoints.userData.dynamicKinematic = true;
    this.fireflyPoints.visible = false;
    this.scene.add(this.fireflyPoints);
  }

  /**
   * Hukum Emas 8, 12, & 14 (NEXUS v3.6 Gunung Gede Bogor Kawah Ratu Thin Fumarole Wisp + Mini-Niagara Mist):
   * 100% GPU Physics-Driven Quiet Hydrothermal Fumarole Steam Wisp + Mini-Niagara Waterfall Mist
   * on Pulau I in STRICTLY 1 SINGLE DRAW CALL!
   * - Ultra-thin, faint white fumarole steam wisp rising gently from Kawah Ratu (-32, 46.8, 698), ZERO red fire/sparks!
   * - 96 Billowing Sunlit Aerosol Mist/Spray Puffs rising from Mini-Niagara Plunge Pool (-44, 2.05, 643)
   */
  _initVolcanicSmokePlumeMesh() {
    const smokeQuads = 232;
    const sparkQuads = 56;
    const mistQuads = 96;
    const totalQuads = smokeQuads + sparkQuads + mistQuads;
    const totalVerts = totalQuads * 4;

    const positions = new Float32Array(totalVerts * 3);
    const normals = new Float32Array(totalVerts * 3);
    const uvs = new Float32Array(totalVerts * 2);
    const aPlumeKind = new Float32Array(totalVerts);
    const aPlumeSeed = new Float32Array(totalVerts * 4);
    const indices = new Uint32Array(totalQuads * 6);

    const cornerUVs = [
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0]
    ];

    for (let q = 0; q < totalQuads; q++) {
      const kind = q < smokeQuads ? 0.0 : q < smokeQuads + sparkQuads ? 1.0 : 2.0;
      const s1 = ((q * 613 + 37) % 9973) / 9973;
      const s2 = ((q * 1427 + 91) % 9967) / 9967;
      const s3 = ((q * 2719 + 149) % 9949) / 9949;
      const s4 = ((q * 3989 + 211) % 9941) / 9941;

      for (let c = 0; c < 4; c++) {
        const vi = q * 4 + c;
        // Anchor rest position near crater / waterfall for valid bounding sphere fallback
        positions[vi * 3] = kind > 1.5 ? -44.0 : -32.0;
        positions[vi * 3 + 1] = kind > 1.5 ? 6.0 : 56.0;
        positions[vi * 3 + 2] = kind > 1.5 ? 643.0 : 698.0;

        normals[vi * 3] = 0.0;
        normals[vi * 3 + 1] = 1.0;
        normals[vi * 3 + 2] = 0.0;

        uvs[vi * 2] = cornerUVs[c][0];
        uvs[vi * 2 + 1] = cornerUVs[c][1];

        aPlumeKind[vi] = kind;
        aPlumeSeed[vi * 4] = s1;
        aPlumeSeed[vi * 4 + 1] = s2;
        aPlumeSeed[vi * 4 + 2] = s3;
        aPlumeSeed[vi * 4 + 3] = s4;
      }

      const b = q * 4;
      const io = q * 6;
      indices[io] = b;
      indices[io + 1] = b + 1;
      indices[io + 2] = b + 2;
      indices[io + 3] = b;
      indices[io + 4] = b + 2;
      indices[io + 5] = b + 3;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setAttribute('aPlumeKind', new THREE.BufferAttribute(aPlumeKind, 1));
    geo.setAttribute('aPlumeSeed', new THREE.BufferAttribute(aPlumeSeed, 4));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
      uniforms: {
        uTime: this.uniforms.uTime,
        uCamPos: this.uniforms.uCamPos,
        uSunDir: this.uniforms.uSunDir,
        uNightFactor: this.uniforms.uNightFactor,
        uDuskFactor: this.uniforms.uDuskFactor,
        uRainIntensity: this.uniforms.uRainIntensity,
        uLightningFlash: this.uniforms.uLightningFlash,
        uWindVec: this.uniforms.uWindVec
      },
      vertexShader: /* glsl */ `
        attribute float aPlumeKind;
        attribute vec4 aPlumeSeed;

        uniform float uTime;
        uniform vec3 uCamPos;
        uniform vec3 uWindVec;

        varying vec2 vUv;
        varying float vKind;
        varying float vLife;
        varying float vAlpha;
        varying float vRot;
        varying float vSeed;
        varying vec3 vToCam;

        const vec3 VENT_ORIGIN = vec3(-32.0, 46.8, 698.0);
        const vec3 NIAGARA_POOL_CENTER = vec3(-44.0, 2.05, 643.0);

        void main() {
          vUv = uv;
          vKind = aPlumeKind;
          vSeed = aPlumeSeed.z;

          vec2 windH = uWindVec.xz;
          float windSpd = length(windH);
          vec2 windDir = windSpd > 0.02 ? windH / windSpd : vec2(0.88, 0.46);

          if (aPlumeKind < 0.5) {
            // --- SUB-POPULATION A: GUNUNG GEDE BOGOR (KAWAH RATU) ULTRA-THIN WHITE FUMAROLE STEAM WISP ---
            // Cull excess quads so only a delicate, whisper-thin hydrothermal steam ribbon rises gently from Kawah Ratu
            if (aPlumeSeed.x > 0.28) {
              vAlpha = 0.0;
              gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
              return;
            }
            float period = mix(18.0, 27.0, aPlumeSeed.y);
            float tLocal = uTime + aPlumeSeed.x * 41.7;
            float cycle = floor(tLocal / period);
            float life = fract(tLocal / period);
            vLife = life;

            float riseY = pow(life, 0.88) * mix(18.0, 34.0, aPlumeSeed.z);
            // Gentle downwind drift clearing the ~58m Kawah Ratu grey-andesite rim
            float rimClear = smoothstep(0.08, 0.36, life);
            vec2 downwind = windDir * (pow(life, 1.25) * (6.5 + windSpd * 16.0) * rimClear);

            float swirlAng = aPlumeSeed.w * 6.28318 + cycle * 2.1 + life * 2.4 * (aPlumeSeed.y - 0.5);
            float radialSpread = mix(1.1, 5.2, pow(life, 0.72)) * (0.45 + 0.55 * aPlumeSeed.z);
            vec2 eddy = vec2(cos(swirlAng), sin(swirlAng)) * radialSpread;

            vec3 center = VENT_ORIGIN + vec3(downwind.x + eddy.x, riseY, downwind.y + eddy.y);
            vec3 toCamVec = uCamPos - center;
            float distCam = length(toCamVec) + 1e-4;
            vToCam = toCamVec / distCam;

            vec3 forward = vToCam;
            vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward) + vec3(1e-5, 0.0, 0.0));
            vec3 up = cross(forward, right);

            float puffRadius = mix(1.8, 5.8, pow(life, 0.70)) * mix(0.84, 1.16, aPlumeSeed.y);
            vRot = aPlumeSeed.w * 6.28318 + uTime * 0.08 * (aPlumeSeed.x - 0.5);

            vec2 qOff = (uv - 0.5) * 2.0 * puffRadius;
            vec3 worldVert = center + right * qOff.x + up * qOff.y;

            float fadeIn = smoothstep(0.0, 0.16, life);
            float fadeOut = pow(1.0 - life, 1.45);
            float nearGuard = smoothstep(6.0, 22.0, distCam);
            vAlpha = fadeIn * fadeOut * nearGuard * 0.052;
            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);
          } else if (aPlumeKind < 1.5) {
            // --- SUB-POPULATION B: DISABLED ON QUIET GUNUNG GEDE BOGOR (ZERO INCANDESCENT FIRE / SPARKS!) ---
            vLife = 0.0;
            vRot = 0.0;
            vToCam = vec3(0.0, 1.0, 0.0);
            vAlpha = 0.0;
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            return;
          } else {
            // --- SUB-POPULATION C: MINI-NIAGARA BILLOWING HORSESHOE WATERFALL MIST & SPRAY CLOUD ---
            float period = mix(4.0, 7.4, aPlumeSeed.y);
            float tLocal = uTime + aPlumeSeed.z * 23.7;
            float cycle = floor(tLocal / period);
            float life = fract(tLocal / period);
            vLife = life;

            float arcAlpha = (fract(aPlumeSeed.x + cycle * 0.17) * 2.0 - 1.0) * 1.15;
            float impR = mix(10.6, 13.8, aPlumeSeed.w);
            vec2 arcDir = vec2(sin(arcAlpha), cos(arcAlpha));
            float riseY = pow(life, 0.72) * mix(4.8, 14.2, aPlumeSeed.z);
            vec2 amphitheaterRoll = -arcDir * (pow(life, 0.62) * mix(2.4, 7.2, aPlumeSeed.y));
            vec2 windDrift = windDir * (pow(life, 1.18) * (1.1 + windSpd * 3.4));

            vec3 center = NIAGARA_POOL_CENTER + vec3(
              arcDir.x * impR + amphitheaterRoll.x + windDrift.x,
              riseY,
              arcDir.y * impR + amphitheaterRoll.y + windDrift.y
            );
            vec3 toCamVec = uCamPos - center;
            float distCam = length(toCamVec) + 1e-4;
            vToCam = toCamVec / distCam;

            vec3 forward = vToCam;
            vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward) + vec3(1e-5, 0.0, 0.0));
            vec3 up = cross(forward, right);

            float puffRadius = mix(3.5, 9.4, pow(life, 0.64)) * mix(0.84, 1.24, aPlumeSeed.w);
            vRot = aPlumeSeed.w * 6.28318 + uTime * 0.22 * (aPlumeSeed.x - 0.5);

            vec2 qOff = (uv - 0.5) * 2.0 * puffRadius;
            vec3 worldVert = center + right * qOff.x + up * qOff.y;

            float fadeIn = smoothstep(0.0, 0.14, life);
            float fadeOut = pow(1.0 - life, 1.38);
            float nearGuard = smoothstep(4.5, 16.0, distCam) * smoothstep(880.0, 280.0, distCam);
            vAlpha = fadeIn * fadeOut * nearGuard * 0.38;
            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);
          }
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec2 vUv;
        varying float vKind;
        varying float vLife;
        varying float vAlpha;
        varying float vRot;
        varying float vSeed;
        varying vec3 vToCam;

        uniform float uTime;
        uniform vec3 uSunDir;
        uniform float uNightFactor;
        uniform float uDuskFactor;
        uniform float uRainIntensity;
        uniform float uLightningFlash;

        float plumeHash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        float plumeNoise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(plumeHash(i), plumeHash(i + vec2(1.0, 0.0)), f.x),
            mix(plumeHash(i + vec2(0.0, 1.0)), plumeHash(i + vec2(1.0, 1.0)), f.x),
            f.y
          );
        }

        float plumeFbm(vec2 p) {
          const mat2 rotM = mat2(0.8, 0.6, -0.6, 0.8);
          float v = 0.0;
          v += 0.5333 * plumeNoise(p); p = rotM * p * 2.03;
          v += 0.2667 * plumeNoise(p); p = rotM * p * 2.05;
          v += 0.1333 * plumeNoise(p); p = rotM * p * 2.02;
          v += 0.0667 * plumeNoise(p);
          return v;
        }

        void main() {
          if (vAlpha < 0.004) discard;
          vec2 p = (vUv - 0.5) * 2.0;
          float r = length(p);
          if (r > 1.0) discard;

          if (vKind > 0.5 && vKind < 1.5) {
            discard;
          }

          // Rotate 2D billboard coordinates & sculpt cauliflower FBM volcanic puff density
          float cs = cos(vRot);
          float sn = sin(vRot);
          vec2 pr = vec2(cs * p.x - sn * p.y, sn * p.x + cs * p.y);
          vec2 nCoord = pr * 1.85 + vec2(vSeed * 13.7, -uTime * 0.07 + vSeed * 7.3);
          float fbmVal = plumeFbm(nCoord);
          float fbmDx = plumeFbm(nCoord + vec2(0.14, 0.0)) - fbmVal;
          float fbmDy = plumeFbm(nCoord + vec2(0.0, 0.14)) - fbmVal;

          float radialMask = smoothstep(1.0, 0.26, r + (fbmVal - 0.5) * 0.44);
          if (radialMask < 0.015) discard;

          // Reconstruct 3D pseudo-spherical volumetric puff normal in world space
          vec3 forward = normalize(vToCam);
          vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward) + vec3(1e-5, 0.0, 0.0));
          vec3 up = cross(forward, right);
          float sphereZ = sqrt(max(0.05, 1.0 - min(0.95, r * r)));
          vec3 puffNorm = normalize(
            right * (p.x * 0.78 - fbmDx * 1.85) +
            up * (p.y * 0.78 - fbmDy * 1.85) +
            forward * sphereZ
          );

          vec3 sunL = normalize(uSunDir);
          float sunLit = clamp(dot(puffNorm, sunL) * 0.5 + 0.5, 0.0, 1.0);
          float silverLining = pow(max(dot(-forward, sunL), 0.0), 4.0) * (1.0 - radialMask * 0.55) * 0.35;

          if (vKind > 1.5) {
            // Sunlit Aerosol Spray & Mist Cloud rising from Mini-Niagara Horseshoe Plunge Pool
            vec3 mistDay = mix(vec3(0.58, 0.70, 0.78), vec3(0.96, 0.99, 1.0), pow(sunLit, 0.78))
                         + vec3(0.92, 0.98, 1.0) * silverLining * 1.15;
            vec3 mistDusk = mix(vec3(0.32, 0.28, 0.36), vec3(0.98, 0.64, 0.40), pow(sunLit, 0.85))
                          + vec3(1.0, 0.66, 0.35) * silverLining;
            vec3 mistNight = mix(vec3(0.07, 0.10, 0.16), vec3(0.20, 0.27, 0.38), sunLit);
            vec3 mistCol = mix(mix(mistDay, mistDusk, uDuskFactor), mistNight, uNightFactor);
            mistCol *= mix(1.0, 0.72, clamp(uRainIntensity * 0.55, 0.0, 0.55));
            mistCol += vec3(0.72, 0.85, 1.0) * uLightningFlash * 0.85;
            gl_FragColor = vec4(mistCol, clamp(radialMask * vAlpha, 0.0, 0.52));
            return;
          }

          // Quiet Gunung Gede Bogor (Kawah Ratu) Pure Hydrothermal White Water-Vapor Fumarole Wisp (ZERO ash, ZERO red fire!)
          vec3 dayLitCol = vec3(0.95, 0.97, 0.99);
          vec3 dayShadowCol = vec3(0.70, 0.75, 0.81);
          vec3 dayCol = mix(dayShadowCol, dayLitCol, pow(sunLit, 0.80)) + vec3(0.96, 0.98, 1.0) * silverLining;

          vec3 duskCol = mix(vec3(0.42, 0.36, 0.44), vec3(0.98, 0.68, 0.44), pow(sunLit, 0.88)) + vec3(1.0, 0.66, 0.36) * silverLining;
          vec3 nightCol = mix(vec3(0.08, 0.10, 0.15), vec3(0.20, 0.25, 0.34), sunLit);

          vec3 steamCol = mix(dayCol, duskCol, uDuskFactor);
          steamCol = mix(steamCol, nightCol, uNightFactor);
          steamCol *= mix(1.0, 0.74, clamp(uRainIntensity * 0.55, 0.0, 0.55));
          steamCol += vec3(0.70, 0.82, 1.0) * uLightningFlash * (0.45 + 0.55 * puffNorm.y);

          gl_FragColor = vec4(steamCol, clamp(radialMask * vAlpha, 0.0, 0.075));
        }
      `
    });

    this.volcanicPlumeMesh = new THREE.Mesh(geo, mat);
    this.volcanicPlumeMesh.name = 'NexusVolcanicSmokePlume';
    this.volcanicPlumeMesh.userData.dynamicKinematic = true;
    this.volcanicPlumeMesh.frustumCulled = false;
    this.volcanicPlumeMesh.renderOrder = 3;
    this.volcanicPlumeMesh.visible = true;
    this.scene.add(this.volcanicPlumeMesh);
  }

  /**
   * Hukum Emas 14 (v3.8 Ultra-Realistic 0–3.0m Physical Ocean, Kelvin Wake & Spectral Seawater Edition):
   * Creates the Physical Dielectric Ocean & Lake Water Material (IOR 1.337 Seawater, F0 = 0.0208) with:
   * - 8-Mode Hasselmann (1973) JONSWAP + Phillips/Miche Steepness-Saturated Swell & Wind-Sea Spectrum (\omega_i = \sqrt{g k_i})
   * - 2nd-Order Stokes Crest Sharpening, Archipelago Headland/Strait Diffraction (K_d), Green's Shoaling, & Battjes-Janssen Surf Breaking
   * - Spatial Nyquist Vertex Anti-Aliasing (aRiverDetail.y = gridSpacingM) -> 0 offshore aliasing, 100% crisp per-pixel normals
   * - 4-Slot Live 19.47° Lord Kelvin Ship V-Wakes + Stern Propeller Wash & Submerged Milky-Aquamarine Bubble Trails
   * - Jerlov Type IB–II 4-Zone Seawater Volume, Wave-Trough Sky Occlusion (cavityAO), Low-Sun Wave Self-Shadowing, & Thin-Crest Jade SSS
   * - Ross-Cardone / Bruneton Slope-Variance Fresnel, Anisotropic GGX Sun/Moon Glitter, & 4-Layer Foam + Storm Spindrift Mist (0 Hexagons!)
   */
  createDanauKenangaWaterMaterial() {
    const hydroCommonGLSL = /* glsl */ `
      ${WeatherOpticsEngine._getArchipelagoSdfGLSL()}

      // Evaluates 360-degree coastline outward unit normal once per ocean vertex (eliminates 4x per-pixel SDF finite diffs!)
      vec2 evalCoastNormalVertex(vec2 baseXZ, float dShore) {
        if (dShore >= 46.0) return vec2(1.0, 0.0);
        float eC = 0.38;
        float gx = evalArchipelagoCoastRaw(baseXZ + vec2(eC, 0.0)) - evalArchipelagoCoastRaw(baseXZ - vec2(eC, 0.0));
        float gz = evalArchipelagoCoastRaw(baseXZ + vec2(0.0, eC)) - evalArchipelagoCoastRaw(baseXZ - vec2(0.0, eC));
        return normalize(vec2(gx, gz) + vec2(1e-5, 0.0));
      }

      void accumOceanWaveMode(
        vec2 baseXZ,
        float t,
        vec2 d,
        float k,
        float w,
        float phase0,
        float q,
        float L,
        float ampRaw,
        float Kd,
        float hBathym,
        float deepRestore,
        float surfBreakScale,
        float shoalPhaseBase,
        float gridSpacingM,
        float crestWeight,
        inout vec2 horizSurge,
        inout float eta,
        inout vec2 negGradEta,
        inout float jacobianY,
        inout float crestAccum,
        inout float troughAccum
      ) {
        if (ampRaw <= 1e-5) return;
        float nyq = gridSpacingM > 0.01 ? (1.0 - smoothstep(0.22 * L, 0.48 * L, gridSpacingM)) : 1.0;
        if (nyq <= 0.002) return;
        float kh = k * hBathym;
        float shoalArg = (kh - 0.85) / 0.55;
        float shoalBoost = 1.0 + 0.16 * exp(-shoalArg * shoalArg) * (1.0 - deepRestore);
        float ai = ampRaw * Kd * shoalBoost * surfBreakScale * nyq;
        float ph = dot(d, baseXZ) * k + shoalPhaseBase * k - w * t + phase0;
        float s = sin(ph);
        float c = cos(ph);
        // 2nd-Order Stokes crest sharpening + broad rounded troughs: +0.5 * k * a^2 * cos(2*ph)
        float stokes2 = 0.5 * k * ai * ai;
        float c2 = c * c - s * s;
        float s2 = 2.0 * s * c;

        eta += ai * c + stokes2 * c2;
        horizSurge -= d * (q * ai * s);
        float dEta = k * ai * s + 2.0 * k * stokes2 * s2;
        negGradEta += d * dEta;
        jacobianY -= q * k * ai * c;
        crestAccum += max(0.0, c + 0.25 * c2) * crestWeight;
        troughAccum += max(0.0, -c) * crestWeight;
      }

      // Synchronized 1:1 with CPU WeatherOpticsEngine.evaluateOceanWaveField across H_s in [0.00 m .. 3.00 m]!
      void evalGravityHydrodynamics(
        vec2 baseXZ,
        float t,
        float seaState,
        vec4 ampA,
        vec4 ampB,
        float gridSpacingM,
        vec2 coastNormalHint,
        out vec3 dispOut,
        out vec3 gradNormalOut,
        out float dShoreOut,
        out float crestFactorOut,
        out float rollerSwashOut,
        out float hBathymOut,
        out float surfBreakOut,
        out float cavityOut
      ) {
        float drPerp = evalArchipelagoCoastRaw(baseXZ);
        // Stone turap boulder berm crosses static water datum y = -1.45m at drPerp = 3.52m
        float dShore = drPerp - 3.52;
        dShoreOut = dShore;

        float S = clamp(seaState, 0.0, 3.55);
        float Hs = clamp(S * 0.854, 0.0, 3.0);

        // 1. Archipelago Headland Diffraction & Sheltering (Teluk U, Selat UI, Dermaga Piers, & Danau Kenanga Cove)
        float telukShelter =
          (1.0 - smoothstep(-45.0, 35.0, baseXZ.x)) *
          smoothstep(-275.0, -210.0, baseXZ.x) *
          smoothstep(82.0, 140.0, baseXZ.y) *
          (1.0 - smoothstep(285.0, 345.0, baseXZ.y));
        float straitMask =
          (1.0 - smoothstep(-70.0, 10.0, baseXZ.x)) *
          smoothstep(-345.0, -265.0, baseXZ.x) *
          smoothstep(420.0, 465.0, baseXZ.y) *
          (1.0 - smoothstep(595.0, 640.0, baseXZ.y));
        float dPierU = length(baseXZ - vec2(-90.0, 444.0));
        float dPierI = length(baseXZ - vec2(-90.0, 616.0));
        float pierLee = max(1.0 - smoothstep(8.0, 34.0, dPierU), 1.0 - smoothstep(8.0, 34.0, dPierI));
        float coveMask =
          (1.0 - smoothstep(65.0, 135.0, baseXZ.x)) *
          smoothstep(20.0, 42.0, baseXZ.x) *
          (1.0 - smoothstep(42.0, 82.0, abs(baseXZ.y)));

        float Kd =
          (1.0 - 0.72 * telukShelter) *
          (1.0 - 0.54 * straitMask) *
          (1.0 - 0.48 * pierLee) *
          (1.0 - 0.42 * coveMask);

        // 2. Shelf Bathymetry & Battjes-Janssen / Miche Depth-Limited Breaking Envelope (H_b <= 0.76 * h)
        float dPos = max(0.0, dShore);
        float hBathym = clamp(0.24 + 0.16 * dPos + 0.0042 * dPos * dPos, 0.24, 22.0);
        hBathymOut = hBathym;

        float HbreakMax = 0.76 * hBathym;
        float HsLocal = max(1e-4, Hs * Kd);
        float rawBreakScale = HbreakMax * inversesqrt(HbreakMax * HbreakMax + HsLocal * HsLocal);
        float deepRestore = smoothstep(36.0, 78.0, dShore);
        float surfBreakScale = mix(rawBreakScale, 1.0, deepRestore);
        surfBreakOut = clamp((1.0 - surfBreakScale) * smoothstep(0.05, 0.65, Hs), 0.0, 1.0);

        vec2 horizSurge = vec2(0.0);
        float eta = 0.0;
        vec2 negGradEta = vec2(0.0);
        float jacobianY = 1.0;
        float crestAccum = 0.0;
        float troughAccum = 0.0;

        float shoalPhaseBase = (1.0 - smoothstep(0.0, 38.0, dPos)) * 3.8;

        // 8-Mode JONSWAP Swell & Wind-Sea Spectrum (Modes 1 & 2 at L=82m & L=65.5m form natural 5-wave ocean swell sets!)
        accumOceanWaveMode(baseXZ, t, vec2(0.8660,  0.5000), 0.076624, 0.8670, 0.00, 0.34, 82.0, ampA.x, Kd, hBathym, deepRestore, surfBreakScale, shoalPhaseBase, gridSpacingM, 0.22, horizSurge, eta, negGradEta, jacobianY, crestAccum, troughAccum);
        accumOceanWaveMode(baseXZ, t, vec2(0.9744,  0.2250), 0.095926, 0.9701, 1.19, 0.33, 65.5, ampA.y, Kd, hBathym, deepRestore, surfBreakScale, shoalPhaseBase, gridSpacingM, 0.20, horizSurge, eta, negGradEta, jacobianY, crestAccum, troughAccum);
        accumOceanWaveMode(baseXZ, t, vec2(0.5299,  0.8480), 0.161107, 1.2572, 2.47, 0.31, 39.0, ampA.z, Kd, hBathym, deepRestore, surfBreakScale, shoalPhaseBase, gridSpacingM, 0.17, horizSurge, eta, negGradEta, jacobianY, crestAccum, troughAccum);
        accumOceanWaveMode(baseXZ, t, vec2(0.9613, -0.2756), 0.237101, 1.5251, 3.83, 0.29, 26.5, ampA.w, Kd, hBathym, deepRestore, surfBreakScale, shoalPhaseBase, gridSpacingM, 0.14, horizSurge, eta, negGradEta, jacobianY, crestAccum, troughAccum);
        accumOceanWaveMode(baseXZ, t, vec2(0.7193,  0.6947), 0.352988, 1.8609, 5.11, 0.27, 17.8, ampB.x, Kd, hBathym, deepRestore, surfBreakScale, shoalPhaseBase, gridSpacingM, 0.11, horizSurge, eta, negGradEta, jacobianY, crestAccum, troughAccum);
        accumOceanWaveMode(baseXZ, t, vec2(0.8910, -0.4540), 0.551157, 2.3253, 0.73, 0.24, 11.4, ampB.y, Kd, hBathym, deepRestore, surfBreakScale, shoalPhaseBase, gridSpacingM, 0.07, horizSurge, eta, negGradEta, jacobianY, crestAccum, troughAccum);
        accumOceanWaveMode(baseXZ, t, vec2(0.1908,  0.9816), 0.884956, 2.9464, 4.29, 0.21,  7.1, ampB.z, Kd, hBathym, deepRestore, surfBreakScale, shoalPhaseBase, gridSpacingM, 0.05, horizSurge, eta, negGradEta, jacobianY, crestAccum, troughAccum);
        accumOceanWaveMode(baseXZ, t, vec2(0.9455,  0.3256), 1.427997, 3.7428, 2.08, 0.18,  4.4, ampB.w, Kd, hBathym, deepRestore, surfBreakScale, shoalPhaseBase, gridSpacingM, 0.04, horizSurge, eta, negGradEta, jacobianY, crestAccum, troughAccum);

        // 3. 360-Degree Snell's-Law Refracted Shoaling Coastal Surf Rollers (Pulau U & Pulau I)
        float swashCrest = 0.0;
        float coastScale = smoothstep(0.0, 0.22, S);
        float nyqRoller = gridSpacingM > 0.01 ? (1.0 - smoothstep(3.8, 9.5, gridSpacingM)) : 1.0;
        if (dShore < 42.0 && coastScale > 0.002 && nyqRoller > 0.002) {
          vec2 coastN = normalize(coastNormalHint + vec2(1e-5, 0.0));
          vec2 refC = baseXZ.y > 536.0 ? vec2(-160.0, 680.0) : vec2(-155.0, 207.0);
          float thetaIsl = atan(baseXZ.y - refC.y, baseXZ.x - refC.x);

          float dClamp = max(-1.8, dShore);
          float phiShoal = 0.36 * dClamp + 4.9 * log(1.0 + max(0.0, dClamp + 1.8) / 7.5);
          float kLocal = 0.36 + 4.9 / (7.5 + max(0.0, dClamp + 1.8));
          float shoalEnv = exp(-max(0.0, dShore - 2.0) * 0.068) * smoothstep(42.0, 24.0, dShore) * nyqRoller;

          float psi1 = phiShoal + 1.10 * t + 1.7 * sin(thetaIsl - 0.5) + 0.65 * cos(2.0 * thetaIsl + 0.8);
          float psi2 = phiShoal * 0.74 + 0.79 * t - 1.4 * cos(thetaIsl + 0.4) + 0.55 * sin(3.0 * thetaIsl);

          float wSurf1 = cos(psi1) + 0.26 * cos(2.0 * psi1);
          float dSurf1 = sin(psi1) + 0.52 * sin(2.0 * psi1);
          float wSurf2 = cos(psi2) + 0.22 * cos(2.0 * psi2);
          float dSurf2 = sin(psi2) + 0.44 * sin(2.0 * psi2);

          float S_coast = min(S, 1.35) * (0.45 + 0.55 * Kd);
          float aR1 = (0.036 + 0.095 * S_coast) * coastScale * shoalEnv;
          float aR2 = (0.024 + 0.062 * S_coast) * coastScale * shoalEnv;

          eta += aR1 * wSurf1 + aR2 * wSurf2;
          horizSurge -= coastN * ((aR1 * sin(psi1) + aR2 * sin(psi2)) * 0.42);
          negGradEta += coastN * (kLocal * (aR1 * dSurf1 + aR2 * 0.74 * dSurf2));
          swashCrest = clamp(0.5 + 0.38 * wSurf1 + 0.24 * wSurf2, 0.0, 1.0) * shoalEnv * coastScale;

          if (dShore > -0.6 && dShore < 4.8 && gridSpacingM < 1.5) {
            float reflEnv = (1.0 - clamp(dShore / 4.8, 0.0, 1.0)) * smoothstep(-0.6, 0.1, dShore);
            float psiRefl = dShore * 3.1 - 1.92 * t + sin(thetaIsl * 6.0) * 1.1;
            float aRefl = 0.015 * (0.30 + 0.65 * S_coast) * coastScale * reflEnv;
            eta += aRefl * cos(psiRefl);
            negGradEta += coastN * (3.1 * aRefl * sin(psiRefl));
          }
        }

        // Smoothly taper 3D vertex displacement at outer 2680m Kepulauan Seribu ocean horizon + calm coral reef lagoons!
        // Unrolled zero-array evaluation prevents D3D11 fxc indexable temporary register stalls during shader compile!
        float rCenter = length(baseXZ - ARCHIPELAGO_CENTER);
        float seribuCalm = 0.0;
        if (rCenter > 880.0) {
          seribuCalm = max(seribuCalm, 1.0 - smoothstep(118.0, 215.0, length(baseXZ - vec2( 1120.0,  295.0))));
          seribuCalm = max(seribuCalm, 1.0 - smoothstep( 96.0, 168.0, length(baseXZ - vec2(  995.0,  520.0))));
          seribuCalm = max(seribuCalm, 1.0 - smoothstep(128.0, 235.0, length(baseXZ - vec2( 1040.0, -260.0))));
          seribuCalm = max(seribuCalm, 1.0 - smoothstep( 98.0, 195.0, length(baseXZ - vec2( 1480.0,  -40.0))));
          seribuCalm = max(seribuCalm, 1.0 - smoothstep(152.0, 260.0, length(baseXZ - vec2( 1185.0,  920.0))));
          seribuCalm = max(seribuCalm, 1.0 - smoothstep(158.0, 245.0, length(baseXZ - vec2( 1660.0,  580.0))));
          seribuCalm = max(seribuCalm, 1.0 - smoothstep(108.0, 205.0, length(baseXZ - vec2(  640.0, -760.0))));
          seribuCalm = max(seribuCalm, 1.0 - smoothstep( 92.0, 175.0, length(baseXZ - vec2(  180.0, -960.0))));
          seribuCalm = max(seribuCalm, 1.0 - smoothstep(108.0, 215.0, length(baseXZ - vec2( -110.0, 1540.0))));
          seribuCalm = max(seribuCalm, 1.0 - smoothstep( 94.0, 185.0, length(baseXZ - vec2( -540.0, -860.0))));
          seribuCalm = max(seribuCalm, 1.0 - smoothstep( 88.0, 180.0, length(baseXZ - vec2( -980.0, -690.0))));
          seribuCalm = max(seribuCalm, 1.0 - smoothstep(106.0, 205.0, length(baseXZ - vec2( -920.0, 1280.0))));
        }
        float horizonTaper = (1.0 - smoothstep(1880.0, 2655.0, rCenter)) * (1.0 - 0.94 * seribuCalm);

        dispOut = vec3(horizSurge.x, eta, horizSurge.y) * horizonTaper;
        gradNormalOut = normalize(vec3(negGradEta.x * horizonTaper, max(0.28, jacobianY), negGradEta.y * horizonTaper));
        float jacobianPinch = clamp((1.0 - jacobianY) * 1.65, 0.0, 1.0);
        crestFactorOut = clamp(crestAccum * 0.72 + jacobianPinch * 0.42, 0.0, 1.0) * horizonTaper;
        rollerSwashOut = swashCrest;
        cavityOut = clamp(troughAccum * clamp(Hs / 1.75, 0.0, 1.18), 0.0, 0.88) * horizonTaper;
      }

      void accumSeribuReef(vec2 xz, float cx, float cz, float rIn, float rOut, float sd, inout float lagoon, inout float reefSurf) {
        vec2 dv = xz - vec2(cx, cz);
        float maxBox = rOut * 1.48;
        if (abs(dv.x) < maxBox && abs(dv.y) < maxBox) {
          float dRaw = length(dv);
          if (dRaw < maxBox) {
            float ang = atan(dv.y, dv.x);
            // Organic multi-lobed coral flat contour aligned with elongated E-W / NE-SW cay orientation (ZERO bullseye rings!)
            float lobe = 1.0
                       + 0.24 * cos(2.0 * (ang - 1.42 - sin(sd) * 0.38))
                       + 0.14 * sin(3.0 * ang + sd)
                       + 0.09 * cos(5.0 * ang - sd * 1.7)
                       + 0.06 * sin(8.0 * ang + sd * 2.3);
            float dEff = dRaw / max(0.55, lobe);
            lagoon = max(lagoon, 1.0 - smoothstep(rIn * 0.72, rOut * 1.04, dEff));
            // Broken windward coral-crest surf arcs (strongest facing North-East Java Sea swell, zero closed circle!)
            float windwardArc = smoothstep(-0.15, 0.72, cos(ang - 0.32))
                              * smoothstep(0.36, 0.84, 0.5 + 0.5 * sin(6.0 * ang + sd * 3.1));
            float uEdge = (dEff - (rOut - 14.0)) / 13.0;
            reefSurf = max(reefSurf, exp(-(uEdge * uEdge)) * windwardArc);
          }
        }
      }

      // Evaluates Kepulauan Seribu Coral Patch-Reef Lagoon Mask (x) & Windward Reef-Edge Surf Arcs (y)
      vec2 evalKepulauanSeribuReef(vec2 xz) {
        float rArch = length(xz - ARCHIPELAGO_CENTER);
        if (rArch < 880.0) return vec2(0.0);
        float lagoon = 0.0;
        float reefSurf = 0.0;
        accumSeribuReef(xz,  1120.0,  295.0, 108.0, 215.0,  0.60, lagoon, reefSurf);
        accumSeribuReef(xz,   995.0,  520.0,  88.0, 168.0,  2.77, lagoon, reefSurf);
        accumSeribuReef(xz,  1040.0, -260.0, 118.0, 235.0,  4.94, lagoon, reefSurf);
        accumSeribuReef(xz,  1480.0,  -40.0,  88.0, 195.0,  7.11, lagoon, reefSurf);
        accumSeribuReef(xz,  1185.0,  920.0, 146.0, 260.0,  9.28, lagoon, reefSurf);
        accumSeribuReef(xz,  1660.0,  580.0, 152.0, 245.0, 11.45, lagoon, reefSurf);
        accumSeribuReef(xz,   640.0, -760.0,  98.0, 205.0, 13.62, lagoon, reefSurf);
        accumSeribuReef(xz,   180.0, -960.0,  84.0, 175.0, 15.79, lagoon, reefSurf);
        accumSeribuReef(xz,  -110.0, 1540.0,  98.0, 215.0, 17.96, lagoon, reefSurf);
        accumSeribuReef(xz,  -540.0, -860.0,  84.0, 185.0, 20.13, lagoon, reefSurf);
        accumSeribuReef(xz,  -980.0, -690.0,  80.0, 180.0, 22.30, lagoon, reefSurf);
        accumSeribuReef(xz,  -920.0, 1280.0,  96.0, 205.0, 24.47, lagoon, reefSurf);
        return vec2(lagoon, reefSurf);
      }
    `;

    return new THREE.ShaderMaterial({
      extensions: { derivatives: true },
      uniforms: {
        uTime: this.uniforms.uTime,
        uSunDir: this.uniforms.uSunDir,
        uTrueSunDir: this.uniforms.uTrueSunDir,
        uSolarDirectTrans: this.uniforms.uSolarDirectTrans,
        uMoonDir: this.uniforms.uMoonDir,
        uMoonIllumFraction: this.uniforms.uMoonIllumFraction,
        uSunColor: this.uniforms.uSunColor,
        uNightFactor: this.uniforms.uNightFactor,
        uDuskFactor: this.uniforms.uDuskFactor,
        uRainIntensity: this.uniforms.uRainIntensity,
        uCloudCover: this.uniforms.uCloudCover,
        uStormFactor: this.uniforms.uStormFactor,
        uFogFactor: this.uniforms.uFogFactor,
        uLightningFlash: this.uniforms.uLightningFlash,
        uWindVec: this.uniforms.uWindVec,
        uWindPhase: this.uniforms.uWindPhase,
        uSeaState: this.uniforms.uSeaState,
        uWaveAmpA: this.uniforms.uWaveAmpA,
        uWaveAmpB: this.uniforms.uWaveAmpB,
        uVesselWakes: this.uniforms.uVesselWakes,
        uHvsParams: this.uniforms.uHvsParams,
        uHvsGlare: this.uniforms.uHvsGlare,
        uGenshinMode: this.uniforms.uGenshinMode,
        uPlanarReflTex: this.uniforms.uPlanarReflTex,
        uPlanarReflMatrix: this.uniforms.uPlanarReflMatrix,
        uPlanarReflActive: this.uniforms.uPlanarReflActive,
        uCollisionMap: this.uniforms.uCollisionMap,
        uCollisionBounds: this.uniforms.uCollisionBounds,
        uCollisionHeightRange: this.uniforms.uCollisionHeightRange,
        uCollisionReady: this.uniforms.uCollisionReady
      },
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
      vertexShader: /* glsl */ `
        attribute vec4 aHydroMeta;
        attribute vec4 aRiverDetail;

        uniform float uTime;
        uniform float uSeaState;
        uniform vec4 uWaveAmpA;
        uniform vec4 uWaveAmpB;

        varying vec3 vWorldPos;
        varying vec2 vBaseXZ;
        varying float vWaveElev;
        varying vec4 vHydroMeta;
        varying vec4 vRiverDetail;
        varying vec2 vFlowDir;
        varying vec3 vGeomNormal;
        varying vec2 vCoastN;

        ${hydroCommonGLSL}

        void main() {
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vBaseXZ = wp.xz;
          vHydroMeta = aHydroMeta;
          vRiverDetail = aRiverDetail;
          vFlowDir = uv;
          vGeomNormal = normalize(mat3(modelMatrix) * normal);

          if (aHydroMeta.x < 0.5) {
            float dShoreVert = evalArchipelagoCoastRaw(wp.xz) - 3.52;
            vec2 coastN = evalCoastNormalVertex(wp.xz, dShoreVert);
            vCoastN = coastN;
            vec3 disp;
            vec3 gNorm;
            float dShore;
            float crestF;
            float swashF;
            float hBathymV;
            float surfBreakV;
            float cavityV;
            float gridSp = max(2.4, aRiverDetail.y);
            evalGravityHydrodynamics(
              wp.xz,
              uTime,
              uSeaState,
              uWaveAmpA,
              uWaveAmpB,
              gridSp,
              coastN,
              disp,
              gNorm,
              dShore,
              crestF,
              swashF,
              hBathymV,
              surfBreakV,
              cavityV
            );
            wp.xyz += disp;
            vWaveElev = disp.y;
          } else if (aHydroMeta.x < 1.5) {
            vCoastN = vec2(1.0, 0.0);
            // Natural Fluvial Stream, Artesian Spring, Small Lake, & Plunge Pool:
            // Depth-locked at 0.00m shoreline contact + subtle standing bow-swell over emergent 3D boulders!
            float depthLock = smoothstep(0.0, 0.24, max(0.0, aRiverDetail.x));
            float localSpd = max(0.22, aRiverDetail.y);
            float bowSwell = 0.024 * clamp(aRiverDetail.z, 0.0, 1.35);
            float rip = 0.011 * sin(aHydroMeta.z * 1.55 - uTime * (1.1 + min(2.2, localSpd) * 1.45))
                      + 0.007 * cos(wp.x * 1.9 + wp.z * 1.7 - uTime * 2.2);
            float dyFresh = (rip + bowSwell) * depthLock;
            wp.y += dyFresh;
            vWaveElev = dyFresh;
          } else {
            vCoastN = vec2(1.0, 0.0);
            // Mini-Niagara 3D Curved Horseshoe Falling Cataract Curtain: hydraulic sheet flutter, seam-locked at 0 & 1!
            float tDrop = aHydroMeta.w;
            float seamEnv = sin(3.14159265 * clamp(tDrop, 0.0, 1.0));
            float flutter = 0.075 * sin(aHydroMeta.z * 1.65 + uTime * 7.2) * cos(tDrop * 12.0 - uTime * 9.6) * seamEnv;
            wp.xyz += vGeomNormal * flutter;
            vWaveElev = flutter;
          }

          vWorldPos = wp.xyz;
          gl_Position = projectionMatrix * viewMatrix * wp;
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec3 vWorldPos;
        varying vec2 vBaseXZ;
        varying float vWaveElev;
        varying vec4 vHydroMeta;
        varying vec4 vRiverDetail;
        varying vec2 vFlowDir;
        varying vec3 vGeomNormal;
        varying vec2 vCoastN;

        uniform float uTime;
        uniform vec3 uSunDir;
        uniform vec3 uTrueSunDir;
        uniform float uSolarDirectTrans;
        uniform vec3 uMoonDir;
        uniform float uMoonIllumFraction;
        uniform vec3 uSunColor;
        uniform float uNightFactor;
        uniform float uDuskFactor;
        uniform float uRainIntensity;
        uniform float uCloudCover;
        uniform float uStormFactor;
        uniform float uFogFactor;
        uniform float uLightningFlash;
        uniform vec3 uWindVec;
        uniform vec2 uWindPhase;
        uniform float uSeaState;
        uniform vec4 uWaveAmpA;
        uniform vec4 uWaveAmpB;
        uniform vec4 uVesselWakes[4];
        uniform vec4 uHvsParams;
        uniform vec4 uHvsGlare;
        uniform sampler2D uPlanarReflTex;
        uniform mat4 uPlanarReflMatrix;
        uniform float uPlanarReflActive;
        uniform sampler2D uCollisionMap;
        uniform vec4 uCollisionBounds;
        uniform vec2 uCollisionHeightRange;
        uniform float uCollisionReady;

        ${HumanEyeHDROptics.getGLSLKernel()}

        // Hukum Emas 14: Domain rotation matrices preventing periodic axial striping
        const mat2 R = mat2(0.80, 0.60, -0.60, 0.80);
        const mat2 R_INV = mat2(0.80, -0.60, 0.60, 0.80);

        ${hydroCommonGLSL}
        ${WeatherOpticsEngine._getSphericalFarSeaGLSL()}
        ${WeatherOpticsEngine._getWaterSolarOcclusionGLSL()}

        float hash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        vec2 hash2(vec2 p) {
          return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453);
        }

        float vnoise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          vec2 u = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
            mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
            u.y
          );
        }

        // Cellular Voronoi distance evaluator -> vec2(edgeRidgeDist, cellCenterDist) for focal caustics & lacy sea-foam
        vec2 voronoiCell(vec2 x) {
          vec2 n = floor(x);
          vec2 f = fract(x);
          float md1 = 8.0;
          float md2 = 8.0;
          for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
              vec2 g = vec2(float(i), float(j));
              vec2 o = 0.14 + 0.72 * hash2(n + g);
              vec2 r = g + o - f;
              float d = dot(r, r);
              if (d < md1) {
                md2 = md1;
                md1 = d;
              } else if (d < md2) {
                md2 = d;
              }
            }
          }
          float s1 = sqrt(md1);
          return vec2(sqrt(md2) - s1, s1);
        }

        // Dual Counter-Crossed Dispersive Capillary-Gravity Wind Wavelets:
        // Two crossing wave fans (+24 deg & -27 deg) advancing monotonically via uWindPhase & dispersive uTime,
        // so capillary facets continuously form, interfere, sparkle, and dissolve (ZERO sliding-rubber-sheet look!)
        float evalCrossedWavelets(vec2 xz, float t, vec2 windPh) {
          vec2 pA = R * (xz * 0.34);
          vec2 pB = R_INV * (xz * 0.39) + vec2(19.3, 7.1);
          vec2 driftA = windPh * 0.075 + vec2(0.42, 0.19) * t;
          vec2 driftB = windPh * 0.068 + vec2(0.37, -0.24) * t;

          float sum = 0.0;
          float amp = 0.28;
          for (int i = 0; i < 4; i++) {
            float dispRate = 1.0 + float(i) * 0.36;
            float nA = vnoise(pA - driftA * dispRate);
            float nB = vnoise(pB - driftB * dispRate);
            sum += amp * (nA + nB - 1.0);
            pA = R * pA * 2.09 + vec2(13.7, 29.3);
            pB = R_INV * pB * 2.13 + vec2(23.1, 11.9);
            amp *= 0.50;
          }
          return sum;
        }

        float fbmCloudRefl(vec2 p) {
          float sum = 0.0;
          float amp = 0.52;
          for (int i = 0; i < 4; i++) {
            sum += amp * vnoise(p);
            p = R * p * 2.05 + vec2(17.1, 31.7);
            amp *= 0.5;
          }
          return sum;
        }

        // True Counter-Propagating Voronoi Focal-Envelope Water Caustics (zero grid-aligned noise diamonds!)
        float evalCaustics(vec2 p, float t) {
          vec2 v1 = voronoiCell(R * (p * 0.92) + vec2(t * 0.21, -t * 0.16));
          vec2 v2 = voronoiCell(R_INV * (p * 1.18) + vec2(-t * 0.18, t * 0.23));
          float ridge1 = exp(-v1.x * 8.5);
          float ridge2 = exp(-v2.x * 9.0);
          return clamp(pow(ridge1 * 0.62 + ridge2 * 0.58, 1.85) * 1.35, 0.0, 1.35);
        }

        // Anisotropic Poiseuille Flow-Sheared Fluvial Capillary Wavelets & 3D Boulder Wake Chop
        // (100% organic multi-scale crossed wave fans stretched along local river current — ZERO Voronoi grid!)
        float evalFluvialRipples(vec2 xz, vec2 F, vec2 C, float localSpd, float boulderWake, float t) {
          float sAlong = dot(xz, F);
          float sCross = dot(xz, C);

          // Crossed anisotropic current-stretched streamline wavelets (+15 deg & -18 deg relative to flow)
          vec2 q1 = vec2(sAlong * 1.05 + sCross * 0.34 - t * localSpd * 1.28, sCross * 2.85 - sAlong * 0.22 - t * 0.32);
          vec2 q2 = vec2(sAlong * 1.22 - sCross * 0.38 - t * localSpd * 1.42 + 13.7, sCross * 3.15 + sAlong * 0.26 + t * 0.29 + 7.1);
          float wLong = (vnoise(q1) + vnoise(q2) - 1.0) * 0.42;

          // Fine dispersive surface-tension capillary facets & subtle bank-eddy curl
          vec2 q3 = R * vec2(sAlong * 2.75 - t * localSpd * 1.95, sCross * 4.95) + vec2(19.3, 5.4);
          vec2 q4 = R_INV * vec2(sAlong * 3.40 - t * localSpd * 2.25, sCross * 5.60) + vec2(8.2, 23.1);
          float wCap = (vnoise(q3) + vnoise(q4) - 1.0) * 0.24;

          // Standing hydraulic riffle & 3D boulder Kelvin-wake chop
          float chopAmp = clamp((localSpd - 0.85) * 0.22 + boulderWake * 0.48, 0.0, 0.65);
          float wChop = chopAmp * 0.16 * sin(sAlong * 8.4 - t * (3.2 + localSpd * 2.8) + vnoise(xz * 1.9) * 2.6)
                      * cos(sCross * 7.2 + sin(sAlong * 2.1 - t * 1.5) * 0.8);

          return wLong + wCap + wChop;
        }

        void main() {
          float distCam = length(cameraPosition - vWorldPos);

          // === REGIME 2: PULAU "I" MINI-NIAGARA 3D CONCAVE HORSESHOE FALLING CATARACT CURTAIN ===
          if (vHydroMeta.x > 1.5) {
            float alphaNorm = vHydroMeta.y;
            float arcM = vHydroMeta.z;
            float tDrop = clamp(vHydroMeta.w, 0.0, 1.0);
            vec3 viewDirF = normalize(cameraPosition - vWorldPos);
            vec3 sunDirF = normalize(uSunDir);
            vec3 trueSunF = normalize(uTrueSunDir);
            vec3 moonDirF = normalize(uMoonDir);
            vec3 celDirF = uNightFactor > 0.55 ? moonDirF : trueSunF;
            float directTransF = uNightFactor > 0.55
              ? clamp(1.0 - uRainIntensity * 0.85 - uStormFactor * 0.65 - uCloudCover * 0.45, 0.0, 1.0)
              : clamp(uSolarDirectTrans, 0.0, 1.0);
            float sunVisF = evalWaterCelestialVisibility3D(vWorldPos, celDirF) * directTransF;

            // Accelerating gravitational free-fall coordinate (v = sqrt(2 g h)) -> ZERO static texture slide, ZERO Voronoi grid!
            float fallV = pow(tDrop, 0.68) * 11.5 - uTime * 6.8;
            float chuteA = vnoise(vec2(arcM * 1.45, fallV * 0.82));
            float chuteB = vnoise(vec2(arcM * 3.85 + 13.7, fallV * 1.65 - uTime * 2.5));
            float chuteMicro = vnoise(R * vec2(arcM * 9.2 + chuteA * 1.8, fallV * 3.4 - uTime * 4.2));
            float sprayPlume = smoothstep(0.32, 0.82, chuteA * 0.45 + chuteB * 0.35 + chuteMicro * 0.20);

            // Turbulent white-water splitter wakes streaming below the 2 basalt "Goat Island" lip boulders!
            float splitWake1 = exp(-pow((alphaNorm + 0.532) / 0.095, 2.0));
            float splitWake2 = exp(-pow((alphaNorm - 0.463) / 0.095, 2.0));
            float hornEdgeFoam = smoothstep(0.76, 0.99, abs(alphaNorm)) * 0.55;

            // Emerald-jade translucent rollover brink (tDrop < 0.22) -> aerated roaring white cataract -> plunge boil!
            vec3 jadeBrink = mix(vec3(0.045, 0.295, 0.275), vec3(0.115, 0.485, 0.445), chuteA);
            jadeBrink = mix(jadeBrink, mix(vec3(0.085, 0.540, 0.620), vec3(0.180, 0.720, 0.780), chuteA), clamp(uGenshinMode, 0.0, 1.0) * 0.85);
            jadeBrink = mix(jadeBrink, vec3(0.018, 0.045, 0.068), uNightFactor);

            float aeration = clamp(
              smoothstep(0.04, 0.42, tDrop) * (0.42 + 0.34 * chuteA + 0.28 * chuteB) +
              sprayPlume * 0.44 * smoothstep(0.06, 0.38, tDrop) +
              (splitWake1 + splitWake2) * 0.64 +
              hornEdgeFoam +
              smoothstep(0.68, 0.98, tDrop) * 0.56,
              0.0,
              1.0
            );

            vec3 baseNormF = gl_FrontFacing ? normalize(vGeomNormal) : -normalize(vGeomNormal);
            vec3 pertNormF = normalize(baseNormF + vec3((chuteB - 0.5) * 0.32, (chuteA - 0.5) * 0.22, (chuteMicro - 0.5) * 0.28));
            float sunDiffF = clamp(dot(pertNormF, sunDirF) * 0.5 + 0.5, 0.14, 1.0);

            vec3 whiteDay = mix(vec3(0.64, 0.77, 0.84), vec3(0.97, 0.99, 1.0), sunDiffF);
            vec3 whiteDusk = mix(vec3(0.36, 0.30, 0.38), vec3(0.98, 0.66, 0.42), sunDiffF);
            vec3 whiteNight = mix(vec3(0.08, 0.12, 0.19), vec3(0.22, 0.30, 0.42), sunDiffF);
            vec3 cataractFoam = mix(mix(whiteDay, whiteDusk, uDuskFactor), whiteNight, uNightFactor);

            vec3 fallCol = mix(jadeBrink, cataractFoam, aeration);

            // Sunlit waterfall crest sparkle & mist sheen in True Linear-HDR -> HVS Photoreceptor Pipeline
            vec3 halfVecF = normalize(celDirF + viewDirF);
            float specF = pow(max(dot(pertNormF, halfVecF), 0.0), 48.0) * 0.95 * sunVisF;
            vec3 specTintF = mix(mix(vec3(1.0, 0.96, 0.86), vec3(1.0, 0.60, 0.30), uDuskFactor), vec3(0.45, 0.58, 0.80), uNightFactor);
            vec3 hdrFall = nexusSRGBToLinear(max(fallCol, vec3(0.0))) * mix(1.18, 0.74, uNightFactor);
            hdrFall += nexusSRGBToLinear(specTintF) * specF * 1.95 * (1.0 - 0.38 * aeration);
            hdrFall += vec3(0.85, 1.05, 1.50) * uLightningFlash * 1.25;

            hdrFall = nexusApplyBoundaryLayerAerialPerspective(
              hdrFall,
              cameraPosition,
              vWorldPos,
              sunDirF,
              uDuskFactor,
              uNightFactor,
              uCloudCover,
              uRainIntensity,
              uLightningFlash,
              uFogFactor
            );

            vec3 hvsFallLin = nexusApplyHumanEyeHDR(
              hdrFall,
              normalize(vWorldPos - cameraPosition),
              sunDirF,
              uSunColor,
              uHvsParams,
              uHvsGlare
            );
            gl_FragColor = vec4(nexusLinearToSRGB(hvsFallLin), 1.0);
            return;
          }

          // === REGIME 1: HUMAN-EYE LINEAR-HDR NATURAL FLUVIAL RIVER, MATA AIR SPRING, DANAU KECIL REKTORAT, & PLUNGE POOL ===
          if (vHydroMeta.x > 0.5) {
            float crossU = vHydroMeta.y;
            float flowArc = vHydroMeta.z;
            float subMode = vHydroMeta.w;
            bool isSpring = (subMode > 2.8 && subMode < 3.8);
            bool isPlungePool = (subMode > 4.0);
            bool isSmallLake = (!isSpring && !isPlungePool && subMode < 0.35);

            // Exact sub-triangle physical water depth (meters), Poiseuille local flow speed, 3D boulder wake, & bend asymmetry
            float rawSignedDepth = vRiverDetail.x;
            float localSpd = max(0.18, vRiverDetail.y);
            float boulderWake = max(0.0, vRiverDetail.z);
            float bendSide = vRiverDetail.w;

            // Smooth low-frequency organic pebble shoreline contour (ZERO high-frequency jagged pixel teeth!)
            float shoreContour = (vnoise(R * (vWorldPos.xz * 1.85)) - 0.5) * 0.011
                               * smoothstep(0.18, 0.0, max(0.0, rawSignedDepth));
            float effDepth = rawSignedDepth + vWaveElev * 0.44 + shoreContour;
            if (!isPlungePool && effDepth < 0.0015) discard;
            effDepth = max(0.0015, effDepth);
            float meniscusFade = smoothstep(0.0015, 0.095, effDepth);

            vec3 viewDirR = normalize(cameraPosition - vWorldPos);
            vec3 sunDirR = normalize(uSunDir);
            vec3 trueSunR = normalize(uTrueSunDir);
            vec3 moonDirR = normalize(uMoonDir);
            vec3 celDirR = uNightFactor > 0.55 ? moonDirR : trueSunR;
            float directTransR = uNightFactor > 0.55
              ? clamp(1.0 - uRainIntensity * 0.88 - uStormFactor * 0.65 - uCloudCover * 0.45, 0.0, 1.0)
              : clamp(uSolarDirectTrans, 0.0, 1.0);
            float directCelVisR = evalWaterCelestialVisibility3D(vWorldPos, celDirR) * directTransR;
            float sunVisR = clamp(1.0 - uRainIntensity * 0.74 - uStormFactor * 0.40 - uCloudCover * 0.22, 0.08, 1.0) * directCelVisR;
            float overcastR = clamp(uCloudCover * 0.52 + uRainIntensity * 0.38 + uStormFactor * 0.28, 0.0, 0.88);
            float clearSunMask = sunVisR * clamp(1.0 - uRainIntensity * 0.90 - uCloudCover * 0.52, 0.0, 1.0);

            // Live Diurnal & Weather-Coupled Linear-HDR Fluvial Irradiance (solar + sky dome + lightning)
            float dayIrradR = mix(1.34, 0.72, overcastR);
            float duskIrradR = mix(0.92, 0.52, overcastR);
            float nightIrradR = mix(0.088, 0.052, overcastR);
            float surfIrrad = mix(mix(dayIrradR, duskIrradR, uDuskFactor), nightIrradR, uNightFactor) + uLightningFlash * 1.55;

            vec2 F = length(vFlowDir) > 0.05 ? normalize(vFlowDir) : vec2(1.0, 0.0);
            vec2 C = vec2(-F.y, F.x);

            // Anisotropic flow-sheared capillary ripples + Nyquist distance & shallow-film damping
            float epsR = mix(0.055, 0.20, smoothstep(10.0, 150.0, distCam));
            float fWave0 = evalFluvialRipples(vWorldPos.xz, F, C, localSpd, boulderWake, uTime);
            float fWaveX = evalFluvialRipples(vWorldPos.xz + vec2(epsR, 0.0), F, C, localSpd, boulderWake, uTime);
            float fWaveZ = evalFluvialRipples(vWorldPos.xz + vec2(0.0, epsR), F, C, localSpd, boulderWake, uTime);

            float shallowWaveDamp = mix(0.24, 1.0, smoothstep(0.002, 0.14, effDepth));
            float rGain = (0.028 + 0.019 * min(2.0, localSpd) + 0.016 * min(1.2, boulderWake))
                        * shallowWaveDamp
                        * (1.0 - smoothstep(52.0, 265.0, distCam));
            float rNx = ((fWave0 - fWaveX) / epsR) * rGain;
            float rNz = ((fWave0 - fWaveZ) / epsR) * rGain;
            float freshFoam = 0.0;

            if (isSpring) {
              // Concentric bubbling artesian spring upwelling rings at Mata Air Hutan Selatan (-286, +48)
              vec2 dSpr = vWorldPos.xz - vec2(-286.0, 48.0);
              float rSpr = length(dSpr) + 1e-4;
              float ringW = sin(rSpr * 4.4 - uTime * 3.6) * exp(-rSpr * 0.26);
              rNx += (dSpr.x / rSpr) * ringW * 0.048 * meniscusFade;
              rNz += (dSpr.y / rSpr) * ringW * 0.048 * meniscusFade;
              float upwellBoil = vnoise(R * (vWorldPos.xz * 3.4 - vec2(uTime * 0.6))) * exp(-rSpr * 0.36);
              freshFoam += max(0.0, ringW * 0.25 + upwellBoil * 0.28) * exp(-rSpr * 0.28);
            }

            if (isPlungePool) {
              // Horseshoe waterfall impact shockwave rings & organic churning aerated plunge-pool boil
              vec2 dPlg = vWorldPos.xz - vec2(-44.0, 643.0);
              float rPlg = length(dPlg) + 1e-4;
              float aPlg = atan(dPlg.x, dPlg.y);
              float arcMask = 1.0 - smoothstep(0.98, 1.32, abs(aPlg));
              float impBand = exp(-pow((rPlg - 12.3) / 2.35, 2.0)) * arcMask;
              float outWave = sin((rPlg - 12.2) * 3.2 + uTime * 4.6) * exp(-abs(rPlg - 11.0) * 0.16);
              rNx += (dPlg.x / rPlg) * outWave * 0.072;
              rNz += (dPlg.y / rPlg) * outWave * 0.072;
              float boilA = vnoise(R * (vWorldPos.xz * 2.1 + dPlg * 0.12 - vec2(0.0, uTime * 0.75)));
              float boilB = vnoise(R_INV * (vWorldPos.xz * 4.7 - vec2(uTime * 0.55, uTime * 0.95)));
              float boilPlume = smoothstep(0.32, 0.78, boilA * 0.6 + boilB * 0.4);
              freshFoam += clamp(impBand * (0.58 + 0.52 * boilPlume) + smoothstep(14.5, 6.2, rPlg) * arcMask * boilPlume * 0.34, 0.0, 0.94);
            }

            // Concentric capillary raindrop impact rings on river, lake, & spring during rain
            if (uRainIntensity > 0.03 && distCam < 135.0) {
              float fadeR = smoothstep(135.0, 26.0, distCam) * shallowWaveDamp;
              for (int rl = 0; rl < 2; rl++) {
                float gSc = rl == 0 ? 1.45 : 2.65;
                vec2 gp = vWorldPos.xz * gSc;
                vec2 gi = floor(gp);
                vec2 gf = fract(gp) - 0.5;
                float sd = hash(gi + float(rl) * 37.0);
                if (sd > clamp(uRainIntensity * 0.85, 0.15, 1.0)) continue;
                float lf = fract(uTime * (1.2 + float(rl) * 0.45) + sd);
                float d = length(gf) + 1e-4;
                float fr = lf * 0.42;
                float wv = sin((d - fr) * 88.0) * exp(-abs(d - fr) * 25.0) * (1.0 - lf) * min(1.3, uRainIntensity) * fadeR;
                rNx += (gf.x / d) * wv * 0.34;
                rNz += (gf.y / d) * wv * 0.34;
                freshFoam += max(0.0, wv) * 0.26;
              }
            }

            vec3 normR = normalize(vec3(vGeomNormal.x + rNx, max(0.45, vGeomNormal.y), vGeomNormal.z + rNz));

            // 1. Crystal-Clear Sunlit Alluvial Riverbed in True Linear-HDR (matches wet terrain shore seamlessly!)
            vec2 refrP = vWorldPos.xz - normR.xz * min(0.44, effDepth * 0.36);
            float cobbleL = vnoise(R * (refrP * 3.1));
            float cobbleM = vnoise(R_INV * (refrP * 7.6 + vec2(11.3, 5.7)));
            float sandGrain = vnoise(R * (refrP * 19.2));
            float mossPatch = smoothstep(0.42, 0.78, vnoise(R_INV * (refrP * 1.25 + vec2(3.1, 9.4))));
            float innerSandMix = clamp(-bendSide * 1.45, 0.0, 1.0) * smoothstep(0.85, 0.0, effDepth);

            vec3 darkWetCobble = vec3(0.155, 0.175, 0.145);
            vec3 warmWetPebble = vec3(0.315, 0.278, 0.205);
            vec3 olivePeriphyton = vec3(0.132, 0.228, 0.116);
            vec3 wetPointBarSand = vec3(0.375, 0.328, 0.232);

            float stoneBlend = smoothstep(0.20, 0.78, cobbleL * 0.60 + cobbleM * 0.40);
            vec3 bedAlbedo = mix(darkWetCobble, warmWetPebble, stoneBlend);
            bedAlbedo = mix(bedAlbedo, olivePeriphyton, mossPatch * 0.46 * smoothstep(0.08, 0.48, effDepth));
            bedAlbedo = mix(bedAlbedo, wetPointBarSand, max(innerSandMix * 0.78, (1.0 - meniscusFade) * 0.55));
            bedAlbedo *= (0.90 + 0.22 * sandGrain);

            vec3 bedTintIrrad = mix(mix(vec3(1.0, 0.98, 0.92), vec3(1.0, 0.66, 0.38), uDuskFactor), vec3(0.68, 0.82, 1.08), uNightFactor);
            vec3 pebbleBed = bedAlbedo * bedTintIrrad * surfIrrad * (0.76 + 0.36 * clearSunMask);

            if (distCam < 145.0 && effDepth < 1.45 && clearSunMask > 0.04) {
              vec2 cUV = refrP - sunDirR.xz * (effDepth / max(0.30, sunDirR.y)) * 0.30 + normR.xz * 0.24;
              float cw1 = sin(dot(R * cUV, vec2(4.1, 2.8)) - uTime * 2.0 + vnoise(cUV * 2.2) * 3.1);
              float cw2 = sin(dot(R_INV * cUV, vec2(-3.6, 4.5)) + uTime * 1.7 + vnoise(R * cUV * 2.6) * 2.9);
              float softCaustic = pow(max(0.0, cw1 * 0.5 + cw2 * 0.5), 2.8);
              float cDepthEnv = smoothstep(0.02, 0.18, effDepth) * (1.0 - smoothstep(0.75, 1.45, effDepth));
              vec3 cTint = mix(vec3(1.05, 0.96, 0.74), vec3(1.10, 0.56, 0.24), uDuskFactor);
              pebbleBed += cTint * softCaustic * cDepthEnv * clearSunMask * smoothstep(145.0, 28.0, distCam) * 0.52;
            }

            // 2. Luminous Clear Tropical Stream & Campus Lake Volume Scattering + Subsurface Glow (ZERO dark swamp ink!)
            float optDepth = effDepth * (1.0 + 0.34 / max(0.28, viewDirR.y));
            float gModeR = clamp(uGenshinMode, 0.0, 1.0);
            vec3 shallowScatter = isSpring ? vec3(0.072, 0.355, 0.332)
                                : isPlungePool ? vec3(0.076, 0.380, 0.362)
                                : vec3(0.070, 0.302, 0.258);
            vec3 deepScatter = isSpring ? vec3(0.036, 0.205, 0.212)
                             : isPlungePool ? vec3(0.040, 0.228, 0.236)
                             : vec3(0.038, 0.182, 0.180);
            shallowScatter = mix(shallowScatter, vec3(0.095, 0.585, 0.620), gModeR * 0.85);
            deepScatter = mix(deepScatter, vec3(0.045, 0.365, 0.480), gModeR * 0.85);

            // Natural rainstorm alluvial silt warming (stays luminous silvery-olive, never pitch-black!)
            if (!isSpring && uRainIntensity > 0.05) {
              float siltMix = clamp(uRainIntensity * 0.30 + uStormFactor * 0.16, 0.0, 0.42);
              shallowScatter = mix(shallowScatter, vec3(0.118, 0.205, 0.152), siltMix);
              deepScatter = mix(deepScatter, vec3(0.068, 0.128, 0.098), siltMix);
            }

            // Forward & upward sunlit subsurface water-column scattering
            float sssAlignR = pow(max(0.0, dot(-viewDirR, normalize(sunDirR - normR * 0.44))), 2.6);
            vec3 sssTintR = mix(vec3(0.058, 0.235, 0.205), vec3(0.250, 0.140, 0.058), uDuskFactor);
            vec3 waterSSS = sssTintR * (0.34 + 0.66 * sssAlignR) * clearSunMask * (1.0 - exp(-optDepth * 1.15)) * (1.0 - uNightFactor);

            vec3 extinctCoeff = (isSpring || isPlungePool) ? vec3(0.92, 0.36, 0.38) : vec3(1.02, 0.43, 0.48);
            vec3 transR = exp(-extinctCoeff * optDepth);
            vec3 volR = mix(deepScatter, shallowScatter, exp(-optDepth * 0.72)) * surfIrrad + waterSSS;
            vec3 underR = mix(volR, pebbleBed, transR);

            // 3. True Open-Sky + Billowing Cloud Fresnel Mirror Reflection in Linear-HDR
            float ndvR = max(dot(viewDirR, normR), 0.0);
            float fresR = 0.028 + (1.0 - 0.028) * pow(1.0 - ndvR, 4.4);
            vec3 reflDirR = reflect(-viewDirR, normR);
            float rEl = clamp(reflDirR.y, 0.0, 1.0);

            float cosReflAzR = dot(normalize(reflDirR.xz + vec2(1e-5, 0.0)), normalize(trueSunR.xz + vec2(1e-5, 0.0)));
            vec3 duskHorR = mix(vec3(0.52, 0.56, 0.68), vec3(0.92, 0.47, 0.22), smoothstep(-0.35, 0.85, cosReflAzR));
            vec3 sZenSRGB = mix(mix(vec3(0.14, 0.39, 0.75), vec3(0.16, 0.19, 0.40), uDuskFactor), vec3(0.020, 0.032, 0.072), uNightFactor);
            vec3 sHorSRGB = mix(mix(vec3(0.58, 0.76, 0.90), duskHorR, uDuskFactor), vec3(0.052, 0.080, 0.148), uNightFactor);
            vec3 skyR = nexusSRGBToLinear(mix(sHorSRGB, sZenSRGB, pow(rEl, 0.58))) * mix(1.18, 0.76, uNightFactor);

            // Reflected billowing clouds advected monotonically via uWindPhase
            vec2 rCloudUV = (reflDirR.xz / (rEl + 0.22)) * 1.42 + uWindPhase * 0.013;
            float cFieldR = fbmCloudRefl(rCloudUV);
            float cThreshR = mix(0.56, 0.22, clamp(uCloudCover, 0.0, 1.0));
            float cMaskR = smoothstep(cThreshR, cThreshR + 0.32, cFieldR) * smoothstep(0.0, 0.18, rEl);
            vec3 cColSRGB = mix(mix(vec3(0.94, 0.96, 1.0), vec3(0.96, 0.58, 0.34), uDuskFactor), vec3(0.08, 0.11, 0.18), uNightFactor);
            skyR = mix(skyR, nexusSRGBToLinear(cColSRGB) * mix(1.18, 0.76, uNightFactor), cMaskR * 0.78);

            // Silvery overcast sky mirror during mist, rain, & storm (real rain-swept rivers mirror the bright cloud deck!)
            vec3 greySkyDay = mix(vec3(0.56, 0.61, 0.66), vec3(0.36, 0.40, 0.46), clamp(uStormFactor + uRainIntensity * 0.32, 0.0, 1.0));
            vec3 greySkySRGB = mix(mix(greySkyDay, vec3(0.34, 0.30, 0.34), uDuskFactor), vec3(0.042, 0.055, 0.082), uNightFactor);
            skyR = mix(skyR, nexusSRGBToLinear(greySkySRGB) * (0.92 + 0.22 * cFieldR), overcastR * 0.85);

            // Overhanging Canopy Reflection ONLY at low grazing horizon angles inside dense southern rainforest (x < -175)
            float denseForestZone = isPlungePool ? 0.34 : smoothstep(-172.0, -246.0, vWorldPos.x);
            float bankGrazing = smoothstep(0.70, 0.98, abs(crossU));
            float canopyElevCut = 0.06 + 0.17 * denseForestZone + 0.09 * bankGrazing;
            vec2 canopyProj = vWorldPos.xz + reflDirR.xz * (4.0 / max(0.16, rEl + 0.10));
            float leafDapple = vnoise(R * (canopyProj * 0.52)) * 0.62 + vnoise(R_INV * (canopyProj * 1.45)) * 0.38;
            float canopyHit = (1.0 - smoothstep(canopyElevCut - 0.10, canopyElevCut + 0.12, rEl + (leafDapple - 0.5) * 0.14))
                            * (0.14 + 0.46 * max(denseForestZone, bankGrazing * 0.55));
            vec3 foliageReflHDR = mix(vec3(0.052, 0.165, 0.048), vec3(0.135, 0.315, 0.095), leafDapple) * surfIrrad;
            skyR = mix(skyR, foliageReflHDR, clamp(canopyHit, 0.0, 0.46));
            skyR += vec3(0.85, 1.05, 1.50) * uLightningFlash * 1.20;

            float reflMix = clamp((0.11 + fresR * 0.88) * mix(0.12, 1.0, meniscusFade), 0.0, 0.95);
            vec3 freshCol = mix(underR, skyR, reflMix);

            // Earned Aerated Micro-Bubble Filaments ONLY around 3D Boulders & Fast Shallow Riffles (ZERO Voronoi grid!)
            if (!isSpring && !isPlungePool && !isSmallLake) {
              float sAl = dot(vWorldPos.xz, F);
              float sCr = dot(vWorldPos.xz, C);
              float strA = vnoise(vec2(sAl * 1.9 - uTime * localSpd * 2.2, sCr * 9.2));
              float strB = vnoise(vec2(sAl * 4.8 - uTime * localSpd * 3.1 + 7.3, sCr * 18.5));
              float bubbleStreak = smoothstep(0.48, 0.86, strA * 0.56 + strB * 0.44);

              float rockWakeFoam = clamp(boulderWake, 0.0, 1.35) * (0.28 + 0.66 * bubbleStreak);
              float rapidMask = smoothstep(1.20, 1.65, localSpd) * smoothstep(0.62, 0.12, effDepth);
              float rapidFoam = rapidMask * smoothstep(0.64, 0.93, strA * 0.50 + strB * 0.50) * 0.36;
              float swashFilm = smoothstep(0.032, 0.006, effDepth) * smoothstep(0.80, 1.40, localSpd) * smoothstep(0.60, 0.90, strB) * 0.16;

              freshFoam += (rockWakeFoam + rapidFoam) * meniscusFade + swashFilm;
            }

            // Dual-Lobe GGX Liquid Sheen + Crisp Solar/Lunar Sparkle on Rippling Stream & Danau Kecil Rektorat
            vec3 hVecR = normalize(celDirR + viewDirR);
            float ndhR = max(dot(normR, hVecR), 0.0);
            float specTightR = pow(ndhR, 185.0) * 1.95 * clearSunMask;
            float specBroadR = pow(ndhR, 28.0) * 0.24 * sunVisR;
            vec3 glitTintR = nexusSRGBToLinear(
              mix(mix(vec3(1.0, 0.96, 0.84), vec3(1.0, 0.58, 0.26), uDuskFactor), vec3(0.48, 0.60, 0.82), uNightFactor)
            );
            freshCol += glitTintR * (specTightR + specBroadR) * meniscusFade;

            // Danau Kecil Rektorat (West Parkland Pond) Shimmering Architectural Reflection (Day, Dusk, & Golden Night!)
            if (isSmallLake && vWorldPos.x > -96.0) {
              float rekAlign = exp(-pow((vWorldPos.z + 18.0) / 21.0, 2.0)) * smoothstep(-95.0, -70.5, vWorldPos.x) * meniscusFade;
              float shim = 0.56 + 0.44 * sin(normR.x * 26.0 - uTime * 2.3);
              if (uNightFactor < 0.85 && reflDirR.x > 0.12) {
                float tajugBand = smoothstep(0.18, 0.42, rEl) * (1.0 - smoothstep(0.45, 0.66, rEl));
                vec3 dayArchHDR = mix(vec3(0.48, 0.45, 0.40), vec3(0.44, 0.16, 0.11), tajugBand) * surfIrrad * 0.22;
                freshCol = mix(freshCol, dayArchHDR * (0.85 + 0.30 * shim), rekAlign * (1.0 - uNightFactor) * 0.28);
              }
              if (uNightFactor > 0.08) {
                freshCol += vec3(1.15, 0.68, 0.18) * rekAlign * shim * uNightFactor * 0.32;
              }
            }

            if (freshFoam > 0.005) {
              vec3 fDaySRGB = mix(vec3(0.88, 0.93, 0.92), vec3(0.70, 0.75, 0.78), overcastR * 0.55);
              vec3 fColSRGB = mix(mix(fDaySRGB, vec3(0.92, 0.68, 0.52), uDuskFactor), vec3(0.20, 0.27, 0.37), uNightFactor);
              vec3 fColHDR = nexusSRGBToLinear(fColSRGB) * mix(1.16, 0.78, uNightFactor) + vec3(0.80, 0.98, 1.40) * uLightningFlash;
              freshCol = mix(freshCol, fColHDR, clamp(freshFoam, 0.0, 0.88));
            }

            freshCol = nexusApplyBoundaryLayerAerialPerspective(
              freshCol,
              cameraPosition,
              vWorldPos,
              sunDirR,
              uDuskFactor,
              uNightFactor,
              uCloudCover,
              uRainIntensity,
              uLightningFlash,
              uFogFactor
            );

            vec3 hvsFreshLin = nexusApplyHumanEyeHDR(
              freshCol,
              normalize(vWorldPos - cameraPosition),
              sunDirR,
              uSunColor,
              uHvsParams,
              uHvsGlare
            );
            gl_FragColor = vec4(nexusLinearToSRGB(hvsFreshLin), 1.0);
            return;
          }

          // 1. EXACT PER-PIXEL 8-MODE JONSWAP GRAVITY HYDRODYNAMICS, KELVIN SHIP WAKES, & SWELL-MODULATED CAPILLARY WAVELETS
          vec3 gDisp;
          vec3 gravNorm;
          float dShore;
          float crestFactor;
          float rollerSwash;
          float hBathym;
          float surfBreak;
          float cavityAO;
          float pxFootprintM = clamp(max(length(dFdx(vBaseXZ)), length(dFdy(vBaseXZ))) * 1.65, 0.0, 48.0);
          evalGravityHydrodynamics(
            vBaseXZ,
            uTime,
            uSeaState,
            uWaveAmpA,
            uWaveAmpB,
            pxFootprintM,
            vCoastN,
            gDisp,
            gravNorm,
            dShore,
            crestFactor,
            rollerSwash,
            hBathym,
            surfBreak,
            cavityAO
          );

          // Live 3D waterline & True Shelf-to-Pelagic Optical Depth (0.04m swash edge -> 22.5m deep strait/pelagic sea)
          float HsLive = clamp(uSeaState * 0.854, 0.0, 3.0);
          float liveWaterline = max(0.0, dShore + gDisp.y * 1.85);
          float waterDepth = clamp(max(0.04, hBathym - 0.18 + gDisp.y * 0.72), 0.04, 22.5);

          // 1B. 4-Slot Live 19.47° Lord Kelvin Ship V-Wakes + Stern Propeller Wash & Submerged Bubble Plumes
          vec2 wakeNormXZ = vec2(0.0);
          float wakeFoam = 0.0;
          float wakeBubble = 0.0;
          if (distCam < 760.0 && liveWaterline > 0.6) {
            float wakeDistFade = smoothstep(760.0, 140.0, distCam);
            for (int wi = 0; wi < 4; wi++) {
              vec4 wk = uVesselWakes[wi];
              float wSpd = abs(wk.w);
              if (wSpd > 0.12) {
                bool isWideHull = wk.w > 0.0;
                float halfL = isWideHull ? 8.2 : 5.4;
                vec2 wFwd = vec2(sin(wk.z), cos(wk.z));
                vec2 wPerp = vec2(cos(wk.z), -sin(wk.z));
                vec2 wRel = vWorldPos.xz - wk.xy;
                float sBehind = halfL - dot(wRel, wFwd);
                float uCrossSigned = dot(wRel, wPerp);
                float uCross = abs(uCrossSigned);
                float spdNorm = clamp(wSpd / 6.2, 0.18, 1.30);
                float wakeMaxLen = mix(26.0, 78.0, clamp(spdNorm, 0.0, 1.0));

                if (sBehind > -1.4 && sBehind < wakeMaxLen && uCross < 4.8 + 0.44 * max(0.0, sBehind)) {
                  float sPos = max(0.0, sBehind);
                  // Exact Lord Kelvin gravity-wave wedge half-angle: tan(19.4712 deg) = 1 / sqrt(8) = 0.353553
                  float uKelvin = (isWideHull ? 1.55 : 0.90) + 0.353553 * sPos;
                  float armW = 1.05 + 0.068 * sPos;
                  float cuspEnv = exp(-pow((uCross - uKelvin) / armW, 2.0))
                                * smoothstep(-1.0, 2.4, sBehind)
                                * exp(-sPos / wakeMaxLen);
                  float phDiv = (uCross - uKelvin) * 2.42 - sPos * 0.66 - uTime * 3.85;
                  float signSide = uCrossSigned >= 0.0 ? 1.0 : -1.0;
                  vec2 cuspDir = normalize(wPerp * signSide - wFwd * 0.353553);
                  float cuspSlope = cos(phDiv) * cuspEnv * 0.25 * spdNorm;

                  float transEnv = smoothstep(uKelvin + 0.6, uKelvin * 0.48, uCross)
                                 * smoothstep(2.6, 8.8, sBehind)
                                 * exp(-sPos / (wakeMaxLen * 0.82));
                  float transSlope = sin(sPos * 1.18 - uTime * 3.2) * transEnv * 0.11 * spdNorm;

                  wakeNormXZ += (cuspDir * cuspSlope + wFwd * transSlope) * wakeDistFade;

                  float sStern = sBehind - 2.0 * halfL;
                  if (sStern > -2.2) {
                    float sStPos = max(0.0, sStern);
                    float propLat = isWideHull ? abs(uCross - 1.52) : uCross;
                    float washW = (isWideHull ? 1.15 : 0.92) + 0.056 * sStPos;
                    float washCore = exp(-pow(propLat / washW, 2.0)) * smoothstep(-2.0, 1.4, sStern);
                    float turbN = vnoise(R * (vWorldPos.xz * 2.75 - wFwd * (uTime * 2.45)));
                    float spdGain = clamp(spdNorm, 0.24, 1.0);
                    wakeFoam += washCore * exp(-sStPos / mix(11.0, 25.0, spdGain)) * (0.46 + 0.62 * turbN) * spdGain * wakeDistFade;
                    wakeBubble += washCore * exp(-sStPos / mix(22.0, 58.0, spdGain)) * (0.54 + 0.46 * turbN) * spdGain * wakeDistFade;
                  }
                  wakeFoam += cuspEnv * smoothstep(0.32, 0.96, sin(phDiv)) * 0.26 * clamp(spdNorm - 0.14, 0.0, 1.0) * wakeDistFade;
                }
              }
            }
          }

          // Swell-modulated capillary-gravity wind wavelets (windward/crest roughness boost + sheltered trough calm!)
          float capDetailFade = 1.0 - smoothstep(48.0, 320.0, distCam);
          float shoalCapDamp = mix(0.42, 1.0, smoothstep(0.0, 3.2, liveWaterline));
          float swellCapMod = (0.72 + 0.46 * crestFactor) * (1.0 - 0.26 * cavityAO);
          float calmCap = mix(0.08, 0.68 + 0.50 * min(uSeaState, 3.2), smoothstep(0.0, 0.26, uSeaState));
          float capGain = calmCap * shoalCapDamp * swellCapMod * capDetailFade;

          float c0 = 0.0;
          float capNx = wakeNormXZ.x;
          float capNz = wakeNormXZ.y;
          if (capGain > 0.006) {
            float epsC = mix(0.065, 0.25, smoothstep(14.0, 195.0, distCam));
            c0 = evalCrossedWavelets(vWorldPos.xz, uTime, uWindPhase);
            float cx = evalCrossedWavelets(vWorldPos.xz + vec2(epsC, 0.0), uTime, uWindPhase);
            float cz = evalCrossedWavelets(vWorldPos.xz + vec2(0.0, epsC), uTime, uWindPhase);
            capNx += ((c0 - cx) / epsC) * 0.056 * capGain;
            capNz += ((c0 - cz) / epsC) * 0.056 * capGain;
          }

          float rainRingFoam = 0.0;
          // 3-Layer Overlapping Concentric Capillary Raindrop Impact Rings
          if (uRainIntensity > 0.03 && distCam < 145.0) {
            float fadeDist = smoothstep(145.0, 32.0, distCam);
            for (int rl = 0; rl < 3; rl++) {
              float gScale = rl == 0 ? 1.25 : (rl == 1 ? 2.15 : 3.45);
              vec2 gp = vWorldPos.xz * gScale;
              vec2 gi = floor(gp);
              vec2 gf = fract(gp) - 0.5;
              float seed = hash(gi + float(rl) * 41.0);
              if (seed > clamp(uRainIntensity * 0.85, 0.15, 1.0)) continue;

              vec2 center = (vec2(hash(gi + 7.3), hash(gi + 19.1)) - 0.5) * 0.50;
              float life = fract(uTime * (1.15 + float(rl) * 0.42) + seed);
              vec2 dv = gf - center;
              float d = length(dv) + 1e-4;
              float front = life * 0.45;
              float env = exp(-abs(d - front) * 26.0) * (1.0 - life) * smoothstep(0.0, 0.05, life);
              float wave = sin((d - front) * 92.0) * env * min(1.35, uRainIntensity) * fadeDist;
              capNx += (dv.x / d) * wave * 0.45;
              capNz += (dv.y / d) * wave * 0.45;
              rainRingFoam += max(0.0, wave) * 0.42;
            }
          }

          vec3 normal = normalize(vec3(gravNorm.x + capNx, gravNorm.y, gravNorm.z + capNz));
          vec3 viewDir = normalize(cameraPosition - vWorldPos);
          vec3 sunDir = normalize(uSunDir);
          vec3 trueSunO = normalize(uTrueSunDir);
          vec3 moonDirO = normalize(uMoonDir);
          vec3 specLightDirO = uNightFactor > 0.55 ? moonDirO : trueSunO;
          float sunVis = clamp(1.0 - uRainIntensity * 0.78 - uStormFactor * 0.42 - uCloudCover * 0.22, 0.06, 1.0);

          // Moving Cumulus Cloud Shadows across Ocean, Teluk U, & Kepulauan Seribu Lagoons
          vec2 oceanCloudPos = (vWorldPos.xz - specLightDirO.xz * (340.0 / max(0.20, specLightDirO.y))) * 0.0036 + uWindPhase * 0.0085;
          float oceanCloudField = fbmCloudRefl(oceanCloudPos * 1.45);
          float oceanCloudThresh = mix(0.60, 0.26, clamp(uCloudCover, 0.0, 1.0));
          float oceanCloudShadow = 1.0 - smoothstep(oceanCloudThresh, oceanCloudThresh + 0.26, oceanCloudField)
                                       * clamp(uCloudCover * 0.85, 0.0, 0.52)
                                       * smoothstep(0.06, 0.24, specLightDirO.y)
                                       * (1.0 - uNightFactor * 0.85);
          sunVis *= mix(1.0, oceanCloudShadow, 0.85);

          // True 3D Celestial Ray-World Occlusion (Single-Pass Unified Sun/Moon Ray -> 2x Faster D3D11 fxc Compile!)
          float useMoonCelO = step(0.55, uNightFactor);
          vec3 activeCelDirO = useMoonCelO > 0.5 ? moonDirO : trueSunO;
          float rawCelVisO = evalWaterCelestialVisibility3D(vWorldPos, activeCelDirO);
          float celAtmWeightO = useMoonCelO > 0.5
            ? clamp(1.0 - uRainIntensity * 0.88 - uStormFactor * 0.65 - uCloudCover * 0.45, 0.0, 1.0)
              * smoothstep(0.22, 0.82, uNightFactor)
              * (0.025 + 0.32 * clamp(uMoonIllumFraction, 0.0, 1.0))
            : clamp(uSolarDirectTrans, 0.0, 1.0) * (1.0 - smoothstep(0.50, 0.88, uNightFactor));
          float directCelVisO = rawCelVisO * celAtmWeightO * mix(1.0, oceanCloudShadow, 0.88);

          // Kepulauan Seribu Shallow Biogenic Coral-Sand Patch-Reef Lagoon Bathymetry
          vec2 seribuReef = evalKepulauanSeribuReef(vBaseXZ);
          if (seribuReef.x > 0.004) {
            waterDepth = mix(waterDepth, mix(2.10, 0.42, seribuReef.x), seribuReef.x * 0.94);
          }

          // Low-sun grazing wave-slope self-shadowing (steep leeward backs of 1.5–3.0m swells cast soft shadow troughs!)
          float lowSunWeight = 1.0 - smoothstep(0.18, 0.62, specLightDirO.y);
          float waveSunNdotL = dot(gravNorm, specLightDirO);
          float waveSelfShadow = mix(1.0, smoothstep(-0.14, 0.22, waveSunNdotL), lowSunWeight * smoothstep(0.35, 1.80, HsLive) * 0.72);

          // 2. JERLOV TYPE IB–II 4-ZONE SEAWATER VOLUME, SEABED CAUSTICS, SUBMERGED BUBBLE PLUMES, & THIN-CREST JADE SSS
          //    Fast-path: skip 3-layer seabed pebble noise in deep pelagic water (waterDepth >= 8.5m -> >99.6% opaque!)
          vec2 refrXZ = vWorldPos.xz - normal.xz * min(0.62, waterDepth * 0.40);
          vec3 submergedBed = vec3(0.165, 0.168, 0.148);
          if (waterDepth < 8.5) {
            float sandMacro = fbmCloudRefl(refrXZ * 0.65);
            float pebMid = vnoise(R * (refrXZ * 4.4 + sandMacro * 0.8));
            float pebFine = vnoise(R_INV * (refrXZ * 12.5));
            submergedBed = mix(
              vec3(0.118, 0.128, 0.116),
              vec3(0.295, 0.276, 0.228),
              clamp(sandMacro * 0.55 + pebMid * 0.30 + pebFine * 0.15, 0.0, 1.0)
            );
            if (seribuReef.x > 0.004) {
              float coralPatch = smoothstep(0.28, 0.72, fbmCloudRefl(refrXZ * 0.22 + vec2(7.3, 13.1)));
              vec3 coralSandBed = mix(vec3(0.18, 0.28, 0.25), vec3(0.82, 0.80, 0.71), coralPatch);
              submergedBed = mix(submergedBed, coralSandBed, seribuReef.x * 0.92);
            }
          }
          float nightBedScale = 0.012 + 0.072 * clamp(moonDirO.y, 0.0, 1.0) * clamp(uMoonIllumFraction, 0.0, 1.0);
          submergedBed *= mix(1.0, nightBedScale, uNightFactor);

          if (waterDepth < 3.15 && distCam < 220.0 && directCelVisO > 0.005) {
            vec2 causticUV = refrXZ - specLightDirO.xz * (waterDepth / max(0.28, specLightDirO.y)) * 0.38 + normal.xz * 0.24;
            float caust = evalCaustics(causticUV, uTime) * smoothstep(3.15, 0.10, waterDepth) * smoothstep(220.0, 48.0, distCam);
            vec3 caustTint = mix(mix(vec3(1.0, 0.95, 0.76), vec3(1.0, 0.62, 0.32), uDuskFactor), vec3(0.22, 0.32, 0.48), uNightFactor);
            submergedBed += caustTint * caust * directCelVisO * waveSelfShadow * 0.58;
          }

          float overcastDesat = clamp(uRainIntensity * 0.65 + uStormFactor * 0.25, 0.0, 1.0);
          float gModeW = clamp(uGenshinMode, 0.0, 1.0);
          vec3 shallowTurquoise = mix(
            mix(vec3(0.068, 0.285, 0.262), vec3(0.088, 0.465, 0.500), gModeW * 0.86),
            vec3(0.045, 0.155, 0.162),
            overcastDesat * (1.0 - gModeW * 0.55)
          );
          if (seribuReef.x > 0.004) {
            vec3 reefTurquoise = mix(
              mix(vec3(0.078, 0.415, 0.388), vec3(0.110, 0.550, 0.580), gModeW * 0.86),
              vec3(0.048, 0.215, 0.218),
              overcastDesat * (1.0 - gModeW * 0.55)
            );
            shallowTurquoise = mix(shallowTurquoise, reefTurquoise, seribuReef.x * 0.88);
          }
          vec3 shelfEmerald     = mix(
            mix(vec3(0.028, 0.152, 0.194), vec3(0.042, 0.310, 0.430), gModeW * 0.85),
            vec3(0.021, 0.088, 0.118),
            overcastDesat * (1.0 - gModeW * 0.55)
          );
          vec3 straitTeal       = mix(
            mix(vec3(0.018, 0.094, 0.152), vec3(0.028, 0.205, 0.360), gModeW * 0.85),
            vec3(0.014, 0.058, 0.094),
            overcastDesat * (1.0 - gModeW * 0.55)
          );
          vec3 deepPelagic      = mix(
            mix(vec3(0.013, 0.058, 0.114), vec3(0.020, 0.125, 0.265), gModeW * 0.85),
            vec3(0.010, 0.036, 0.072),
            overcastDesat * (1.0 - gModeW * 0.55)
          );

          shallowTurquoise = mix(shallowTurquoise, vec3(0.015, 0.030, 0.052), uNightFactor);
          shelfEmerald     = mix(shelfEmerald,     vec3(0.012, 0.024, 0.046), uNightFactor);
          straitTeal       = mix(straitTeal,       vec3(0.011, 0.021, 0.043), uNightFactor);
          deepPelagic      = mix(deepPelagic,      vec3(0.010, 0.020, 0.042), uNightFactor);

          // Wave-trough cavity occlusion deepens heavy swell troughs smoothly while keeping crests luminous
          float troughDarken = 1.0 - cavityAO * 0.24;

          // Translucent Backlit Jade-Emerald Subsurface Scattering (SSS) glowing through raised 0.5–3.0m wave crests!
          float waveSlope = clamp(1.0 - gravNorm.y, 0.0, 1.0);
          float crestElevNorm = clamp(max(0.0, gDisp.y) / max(0.32, 0.46 * HsLive + 0.12), 0.0, 1.45);
          float thinCrest = clamp(crestElevNorm * 0.78 + crestFactor * 0.58 + rollerSwash * 0.44 + waveSlope * 0.68, 0.0, 1.95)
                          * smoothstep(0.08, 0.52, HsLive);
          float sssForward = pow(max(0.0, dot(-viewDir, normalize(specLightDirO - normal * 0.52))), 2.4) * directCelVisO;
          float sssAmbient = (max(0.0, gravNorm.y) * 0.28 + waveSlope * 0.36) * sunVis;
          float crestSSS = thinCrest * (sssAmbient + 1.05 * sssForward) * (1.0 - uNightFactor * 0.88);
          vec3 sssTint = mix(vec3(0.058, 0.335, 0.275), vec3(0.29, 0.185, 0.078), uDuskFactor);

          // Submerged Milky-Aquamarine Micro-Bubble Plumes (fast-path gated to wakes, nearshore surf, & heavy >1.05m swells!)
          float submergedBubble = 0.0;
          if (distCam < 780.0 && (wakeBubble > 0.004 || (surfBreak > 0.004 && liveWaterline < 14.0) || HsLive > 1.05)) {
            float bubbleNoise = fbmCloudRefl(vWorldPos.xz * 0.85 - normal.xz * 0.45 + vec2(uTime * 0.11, -uTime * 0.09));
            submergedBubble = clamp(
              wakeBubble * 0.86 +
              surfBreak * (0.26 + 0.42 * bubbleNoise) * smoothstep(14.0, 0.6, liveWaterline) +
              crestFactor * waveSlope * smoothstep(1.05, 2.75, HsLive) * bubbleNoise * 0.36,
              0.0,
              0.84
            ) * smoothstep(780.0, 95.0, distCam);
          }
          vec3 bubbleTint = mix(mix(vec3(0.16, 0.44, 0.45), vec3(0.30, 0.25, 0.21), uDuskFactor), vec3(0.035, 0.075, 0.11), uNightFactor);
          bubbleTint *= mix(1.0, 0.68, overcastDesat);

          // Beer-Lambert spectral seawater attenuation + upward sky-irradiance volume backscatter across 4 Jerlov zones
          float volSkyIrrad = mix(mix(1.18, 0.88, uDuskFactor), 0.76, uNightFactor) * mix(1.0, 0.78, overcastDesat) * mix(1.0, oceanCloudShadow, 0.34);
          vec3 opticalPath = vec3(2.15, 0.78, 0.62) * waterDepth * (1.0 + 0.38 * (1.0 - max(0.22, viewDir.y)));
          vec3 transmittance = exp(-opticalPath);
          vec3 midToShallow = mix(shelfEmerald, shallowTurquoise, exp(-waterDepth * 0.72));
          vec3 deepToMid = mix(deepPelagic, mix(straitTeal, midToShallow, exp(-waterDepth * 0.24)), exp(-waterDepth * 0.095));
          vec3 waterVolume = deepToMid * volSkyIrrad * troughDarken + sssTint * crestSSS * 0.58;
          waterVolume = mix(waterVolume, bubbleTint, submergedBubble * 0.58);
          vec3 underwaterCol = mix(waterVolume, submergedBed, transmittance);

          // 3. ROSS-CARDONE / BRUNETON SLOPE-VARIANCE EFFECTIVE DIELECTRIC FRESNEL (IOR 1.337 -> F0 = 0.0208)
          float distSlopeSpread = smoothstep(42.0, 360.0, distCam);
          float roughSigma2 = 0.0015 + 0.024 * clamp(uSeaState, 0.0, 3.55) * (0.34 + 0.66 * distSlopeSpread) + uRainIntensity * 0.008;
          vec3 fresNormal = normalize(mix(gravNorm, normal, 0.36));
          float NdotV = max(dot(viewDir, fresNormal), 0.0);
          float fresnelExp = mix(5.0, 2.65, clamp(roughSigma2 * 18.5, 0.0, 1.0));
          float fresnelRaw = 0.0208 + (1.0 - 0.0208) * pow(1.0 - NdotV, fresnelExp);
          float subfacetScatterFloor = 0.042 + 0.115 * clamp(roughSigma2 * 15.0, 0.0, 1.0);
          float fresnel = max(fresnelRaw, subfacetScatterFloor);
          float horizonMasking = mix(0.96, 0.76, clamp(roughSigma2 * 16.0, 0.0, 1.0)) * (1.0 - cavityAO * 0.12);
          float reflWeight = clamp(0.065 + fresnel * 0.90, 0.0, horizonMasking);

          // Unified sky reflection matching skyMesh & evalSphericalFarSeaHDR 1:1 via nexusEvalHorizonSkySRGB!
          vec3 reflNormal = normalize(mix(gravNorm, normal, 0.62));
          vec3 reflRay = reflect(-viewDir, reflNormal);
          float reflElev = clamp(max(reflRay.y, 0.045 + 0.065 * waveSlope), 0.0, 1.0);
          float lunarOceanScatter = clamp(moonDirO.y, 0.0, 1.0) * clamp(uMoonIllumFraction, 0.0, 1.0) * (1.0 - overcastDesat * 0.75);
          vec3 skyZen = mix(
            mix(vec3(0.13, 0.37, 0.71), vec3(0.11, 0.16, 0.38), uDuskFactor),
            vec3(0.056, 0.086, 0.145) + vec3(0.038, 0.058, 0.098) * lunarOceanScatter,
            uNightFactor
          );
          vec3 skyHor = nexusEvalHorizonSkySRGB(
            vec3(reflRay.x, 0.0, reflRay.z),
            trueSunO,
            uDuskFactor,
            uNightFactor,
            uCloudCover,
            uRainIntensity,
            uStormFactor,
            uFogFactor,
            lunarOceanScatter
          );
          vec3 skyRefl = mix(skyHor, skyZen, pow(reflElev, 0.58));

          // Reflected Cumulonimbus / tropical clouds advected monotonically via uWindPhase
          vec2 reflCloudUV = (reflRay.xz / (reflElev + 0.26)) * 1.38 + uWindPhase * 0.013;
          float cField = fbmCloudRefl(reflCloudUV);
          float cThresh = mix(0.56, 0.22, clamp(uCloudCover, 0.0, 1.0));
          float cMask = smoothstep(cThresh, cThresh + 0.34, cField) * smoothstep(0.02, 0.22, reflElev);
          vec3 cloudReflCol = mix(mix(vec3(0.92, 0.95, 0.99), vec3(0.95, 0.56, 0.32), uDuskFactor), vec3(0.075, 0.105, 0.168), uNightFactor);
          cloudReflCol = mix(cloudReflCol, vec3(0.24, 0.28, 0.34) * mix(1.0, 0.25, uNightFactor), clamp(uRainIntensity * 0.68 + uStormFactor * 0.35, 0.0, 0.92));
          skyRefl = mix(skyRefl, cloudReflCol, cMask * 0.74);
          skyRefl = mix(skyRefl, vec3(0.25, 0.29, 0.35) * mix(1.0, 0.22, uNightFactor), clamp(uRainIntensity * 0.62 + uStormFactor * 0.30, 0.0, 0.90));
          skyRefl *= (1.0 - cavityAO * 0.16);
          skyRefl += vec3(0.76, 0.88, 1.0) * uLightningFlash * 1.15;

          // 3B. REAL PLANAR MIRROR REFLECTION (Gedung Rektorat UI, Trees, Jembatan Teksas UI, & Islands within 680m!)
          //     Scoped to 220-680m so open-ocean sky reflection beyond the islands is NEVER double-tonemapped!
          if (uPlanarReflActive > 0.5 && distCam < 685.0) {
            vec4 reflClip = uPlanarReflMatrix * vec4(vWorldPos.x, -1.45, vWorldPos.z, 1.0);
            if (reflClip.w > 0.05) {
              vec2 reflBaseUV = reflClip.xy / reflClip.w;
              vec2 edgeDist = min(reflBaseUV, 1.0 - reflBaseUV);
              float edgeFeather = smoothstep(0.008, 0.095, min(edgeDist.x, edgeDist.y))
                                * smoothstep(680.0, 220.0, distCam);
              if (edgeFeather > 0.004) {
                vec2 waveDistort = reflNormal.xz
                                 * mix(0.032, 0.009, smoothstep(16.0, 320.0, distCam))
                                 * clamp(0.26 + HsLive * 0.40, 0.20, 1.12)
                                 * mix(1.0, 0.76, uNightFactor);
                vec2 sampleReflUV = clamp(reflBaseUV + waveDistort, vec2(0.004), vec2(0.996));
                vec4 planarSample = texture2D(uPlanarReflTex, sampleReflUV);
                vec3 planarRGB = mix(planarSample.rgb, min(planarSample.rgb, vec3(0.62)), uNightFactor * 0.85);
                float planarClarity = (1.0 - clamp(roughSigma2 * 8.5, 0.0, 0.32)) * (1.0 - cavityAO * 0.14);
                skyRefl = mix(skyRefl, planarRGB * planarClarity, edgeFeather * planarSample.a * 0.90);
              }
            }
          }

          // Physical Overhead Reflection & Soft Ambient Shadow of Jembatan Teksas UI directly beneath the span!
          if (vWorldPos.z > 169.0 && vWorldPos.z < 322.0 && abs(vWorldPos.x + 28.0) < 8.5) {
            float uArchB = clamp((vWorldPos.z - 245.65) / 74.0, -1.0, 1.0);
            float bridgeDeckH = 4.15 + 4.95 * max(0.0, 1.0 - uArchB * uArchB) - 1.25;
            float dxRefl = clamp(reflRay.x * (max(1.5, bridgeDeckH - vWorldPos.y) / max(0.32, reflElev + 0.22)), -3.6, 3.6);
            float hitX = vWorldPos.x + dxRefl;
            float lateralProx = exp(-pow((vWorldPos.x + 28.0) / 4.8, 2.0));
            float reflBridgeMask = (1.0 - smoothstep(2.2, 3.8, abs(hitX + 28.0))) * lateralProx * smoothstep(170.0, 175.0, vWorldPos.z) * (1.0 - smoothstep(316.0, 321.0, vWorldPos.z));
            float underBridgeAO = (1.0 - smoothstep(2.4, 6.2, abs(vWorldPos.x + 28.0))) * smoothstep(170.0, 175.0, vWorldPos.z) * (1.0 - smoothstep(316.0, 321.0, vWorldPos.z));
            vec3 bridgeSoffitRefl = mix(vec3(0.14, 0.16, 0.19), vec3(0.28, 0.11, 0.09), smoothstep(1.8, 2.7, abs(hitX + 28.0))) * mix(1.0, 0.25, uNightFactor);
            if (uNightFactor > 0.12) {
              bridgeSoffitRefl += vec3(0.45, 0.34, 0.16) * uNightFactor * 0.45;
            }
            skyRefl = mix(skyRefl, bridgeSoffitRefl, reflBridgeMask * 0.46);
            underwaterCol *= (1.0 - underBridgeAO * 0.26);
          }

          vec3 col = mix(underwaterCol, skyRefl, reflWeight);

          // 4. ANISOTROPIC WIND-SHEARED DUAL-LOBE GGX MICROFACET SUN / MOON GLITTER PATH (3D SOLID-OCCLUDED!)
          vec2 wDir2 = length(uWindVec.xz) > 0.02 ? normalize(uWindVec.xz) : vec2(0.78, 0.62);
          vec2 wPerp2 = vec2(-wDir2.y, wDir2.x);
          vec3 H = normalize(specLightDirO + viewDir);
          // Cox-Munk anisotropic slope distribution: cross-wind variance is narrower than upwind/downwind variance,
          // naturally stretching solar/lunar glitter into a shimmering path toward the horizon!
          vec3 T_upwind = normalize(vec3(wDir2.x, -dot(normal.xz, wDir2) / max(0.25, normal.y), wDir2.y));
          vec3 B_cross  = normalize(cross(normal, T_upwind));
          float hDotT = dot(H, T_upwind);
          float hDotB = dot(H, B_cross);
          float hDotN = max(dot(normal, H), 1e-4);
          float alphaUp = clamp(0.032 + 0.055 * sqrt(max(0.0, uSeaState)) + 0.085 * distSlopeSpread, 0.03, 0.34);
          float alphaCr = alphaUp * 0.64;
          float anisoDenom = (hDotT * hDotT) / (alphaUp * alphaUp) + (hDotB * hDotB) / (alphaCr * alphaCr) + hDotN * hDotN;
          float ggxGlint = 1.0 / (3.14159265 * alphaUp * alphaCr * anisoDenom * anisoDenom);
          float specTight = pow(hDotN, mix(360.0, 95.0, distSlopeSpread)) * mix(1.45, 0.62, distSlopeSpread) * waveSelfShadow;
          float specGlitter = min(2.8, ggxGlint * 0.036) * waveSelfShadow;
          vec3 glitterTint = mix(mix(vec3(1.0, 0.94, 0.80), vec3(1.0, 0.58, 0.26), uDuskFactor), vec3(0.45, 0.56, 0.78), uNightFactor);
          vec3 glitter = glitterTint * (specTight + specGlitter) * directCelVisO;
          col += glitter / (1.0 + glitter * 0.42);

          // 4B. GENSHIN IMPACT / FONTAINE PROCEDURAL ANIME 4-POINT STAR SUN-GLINT SPARKLES
          if (gModeW > 0.001 && distCam < 520.0) {
            vec2 spGrid = vWorldPos.xz * 2.15 + vec2(uTime * 0.16, -uTime * 0.13);
            vec2 spId = floor(spGrid);
            vec2 spUv = fract(spGrid) - 0.5;
            float spHash = fract(sin(dot(spId, vec2(127.1, 311.7))) * 43758.5453);
            float spFlicker = max(0.0, sin(uTime * (3.2 + spHash * 2.8) + spHash * 6.2831));
            spFlicker = pow(spFlicker, 5.0) * step(0.62, spHash);
            float starCross = max(
              (1.0 - smoothstep(0.0, 0.42, abs(spUv.x) * 4.5 + abs(spUv.y))) * (1.0 - smoothstep(0.0, 0.09, abs(spUv.x))),
              (1.0 - smoothstep(0.0, 0.42, abs(spUv.y) * 4.5 + abs(spUv.x))) * (1.0 - smoothstep(0.0, 0.09, abs(spUv.y)))
            );
            float starCore = 1.0 - smoothstep(0.0, 0.16, length(spUv));
            float sunGlintMask = smoothstep(0.55, 0.96, hDotN) * directCelVisO * waveSelfShadow * smoothstep(520.0, 95.0, distCam);
            float gSparkle = (starCross * 0.85 + starCore * 1.15) * spFlicker * sunGlintMask * gModeW;
            col += mix(vec3(1.0, 0.98, 0.86), vec3(0.75, 0.96, 1.0), 0.35) * gSparkle * 1.45;
          }

          // 5. 4-LAYER PHOTOGRAPHIC SEA FOAM, BREAKING BORES, LANGMUIR WINDROWS, & STORM SPINDRIFT SPRAY (ZERO HEXAGONS!)
          float foamBand = 1.0 - smoothstep(150.0, 560.0, distCam);
          float totalFoam = clamp(wakeFoam, 0.0, 0.92);
          float spindriftMist = 0.0;

          // Kepulauan Seribu Fringing Reef Outer Breaking Surf Ring
          if (seribuReef.y > 0.004 && distCam < 2350.0) {
            float reefLace = vnoise(R * (vWorldPos.xz * 1.65 - uWindPhase * 0.22));
            totalFoam = max(
              totalFoam,
              seribuReef.y * (0.42 + 0.52 * reefLace) * smoothstep(2350.0, 620.0, distCam) * 0.82
            );
          }

          // 5A. Shoreline Breaking Surf Bore & Receding Aerated Turap Swash Lace
          if (liveWaterline < 6.4 && distCam < 1380.0) {
            float alongCoast = dot(vWorldPos.xz, vec2(-vCoastN.y, vCoastN.x));
            float swashMacro = fbmCloudRefl(vec2(liveWaterline * 0.85 - uTime * 0.18, alongCoast * 0.55) + normal.xz * 0.45);
            float swashMeso = vnoise(R * (vWorldPos.xz * 4.2 + swashMacro * 1.35 + vec2(uTime * 0.24, -uTime * 0.18)));
            float swashMicro = vnoise(R_INV * (vWorldPos.xz * 13.8 - uTime * 0.35));
            float frothyLace = mix(
              0.66,
              clamp(smoothstep(0.24, 0.74, swashMacro * 0.48 + swashMeso * 0.35 + swashMicro * 0.17) * (0.52 + 0.68 * swashMeso), 0.0, 1.0),
              foamBand
            );
            float contactSwash = smoothstep(1.15 + rollerSwash * 0.95, 0.0, liveWaterline);
            float rollerRibbon = smoothstep(5.6, 0.35, liveWaterline) * (pow(rollerSwash, 1.55) * 0.62 + surfBreak * 0.48);
            float altFade = smoothstep(1380.0, 380.0, distCam) * 0.65 + 0.35 * smoothstep(1380.0, 920.0, distCam);
            float seaFoamDrive = smoothstep(0.03, 0.28, uSeaState) * (0.44 + 0.30 * clamp(uSeaState, 0.0, 3.0));
            totalFoam = max(
              totalFoam,
              clamp(contactSwash * (0.38 + 0.72 * frothyLace) + rollerRibbon * frothyLace, 0.0, 0.94) * seaFoamDrive * altFade
            );
          }

          // 5B. Monahan Steepness-Driven Open-Sea Plunging/Spilling Whitecaps, Langmuir Windrows, & Leeward Spindrift Mist (0.35m .. 3.00m!)
          if (HsLive > 0.30 && distCam < 860.0 && liveWaterline > 0.75) {
            float seaNorm = clamp((HsLive - 0.30) / 2.70, 0.0, 1.0);
            float sAlongW = dot(vWorldPos.xz, wDir2) - length(uWindPhase) * 0.28;
            float sCrossW = dot(vWorldPos.xz, wPerp2);
            float warpF = fbmCloudRefl(vWorldPos.xz * 0.22 - wDir2 * (uTime * 0.08));

            float breakPatch = fbmCloudRefl(vec2(sAlongW * 0.38 + warpF * 1.65, sCrossW * 0.34 - warpF * 1.25));
            float frothDetail = vnoise(R * (vWorldPos.xz * 3.5 - wDir2 * (uTime * 0.46) + breakPatch * 1.4));
            float bubbleGrain = mix(0.65, vnoise(R_INV * (vWorldPos.xz * 12.2 + normal.xz * 1.25)), foamBand);

            // Asymmetric spilling crest whitecap + decaying bubble mat on wave back
            float leeTrail = max(0.0, -dot(gravNorm.xz, wDir2));
            float steepDrive = (crestFactor * (0.58 + 2.05 * waveSlope) + surfBreak * 0.45 + leeTrail * crestFactor * 0.34)
                             * (0.52 + 0.78 * breakPatch);
            float breakThresh = mix(0.78, 0.19, pow(seaNorm, 0.78));
            float crestPlume = smoothstep(breakThresh, breakThresh + 0.30, steepDrive)
                             * smoothstep(0.22, 0.78, frothDetail * 0.64 + bubbleGrain * 0.36);

            // Organic Wind-Aligned Langmuir Bubble Windrows (domain-warped & broken into lacy frothy trails, ZERO ruled lines!)
            float langmuir = 0.0;
            if (HsLive > 0.85) {
              vec2 warpLang = vec2(
                fbmCloudRefl(vWorldPos.xz * 0.14 + vec2(3.7, 9.1) - wDir2 * (uTime * 0.05)),
                fbmCloudRefl(R * (vWorldPos.xz * 0.17) + vec2(14.2, 2.8) - wDir2 * (uTime * 0.06))
              ) - 0.5;
              float sAlWarp = sAlongW * 0.18 + warpLang.x * 4.2;
              float sCrWarp = sCrossW * 0.72 + warpLang.y * 5.8 + gDisp.y * 0.65;
              float rowMacro = fbmCloudRefl(vec2(sAlWarp, sCrWarp));
              float rowLace = vnoise(R * (vWorldPos.xz * 2.6 - wDir2 * (uTime * 0.32) + warpLang * 3.5));
              float rowBreakup = smoothstep(0.42, 0.82, breakPatch * 0.58 + frothDetail * 0.42);
              langmuir = smoothstep(0.66, 0.88, rowMacro)
                       * smoothstep(0.36, 0.84, rowLace)
                       * rowBreakup
                       * (0.35 + 0.65 * crestFactor)
                       * smoothstep(0.85, 2.75, HsLive) * 0.38;
            }

            // High-Energy Storm Spindrift Spray Mist shearing downwind off 1.65m .. 3.00m wave crests!
            if (HsLive > 1.60) {
              float spinStreak = vnoise(R * vec2(sAlongW * 0.42 - uTime * 1.85, sCrossW * 1.85 + warpF * 2.4));
              float spinFine = vnoise(R_INV * vec2(sAlongW * 1.05 - uTime * 3.1, sCrossW * 3.8 + warpF * 1.8));
              float spinEnv = smoothstep(1.60, 2.95, HsLive) * smoothstep(0.42, 0.94, crestFactor + leeTrail * 0.42);
              spindriftMist = spinEnv * smoothstep(0.40, 0.86, spinStreak * 0.62 + spinFine * 0.38)
                            * smoothstep(680.0, 90.0, distCam) * 0.42;
            }

            float openSeaFade = smoothstep(860.0, 140.0, distCam) * smoothstep(0.75, 3.2, liveWaterline);
            totalFoam = max(totalFoam, clamp(crestPlume * mix(0.68, 0.95, seaNorm) + langmuir, 0.0, 0.95) * openSeaFade);
          }

          if (totalFoam > 0.004) {
            // Genshin Fontaine crisp cel-edged foam ribbon when uGenshinMode > 0
            totalFoam = mix(totalFoam, smoothstep(0.18, 0.34, totalFoam) * 0.94, gModeW * 0.62);
            // Natural pearlescent aerated bubble lace blending smoothly into crisp sunlit white plunging crest core (ZERO neon cyan!)
            float denseCore = smoothstep(0.18, 0.68, totalFoam);
            float foamSunLit = mix(0.76, 1.0, waveSelfShadow);
            vec3 aeratedLace = mix(col * 0.32 + vec3(0.50, 0.58, 0.62), vec3(0.95, 0.97, 0.98), denseCore) * foamSunLit;
            aeratedLace = mix(aeratedLace, mix(vec3(0.72, 0.92, 0.98), vec3(0.99, 1.0, 1.0), smoothstep(0.25, 0.58, totalFoam)), gModeW * 0.75);
            vec3 foamCol = mix(mix(aeratedLace, vec3(0.94, 0.68, 0.50), uDuskFactor), vec3(0.18, 0.24, 0.33) * (0.55 + 0.45 * denseCore), uNightFactor);
            foamCol = mix(foamCol, foamCol * 0.76, clamp(uRainIntensity * 0.45 + uStormFactor * 0.25, 0.0, 0.45));
            foamCol += vec3(0.70, 0.82, 1.0) * uLightningFlash * 0.85;
            col = mix(col, foamCol, clamp(totalFoam, 0.0, 0.92));
          }

          if (spindriftMist > 0.004) {
            vec3 mistTint = mix(mix(vec3(0.84, 0.91, 0.96), vec3(0.96, 0.68, 0.46), uDuskFactor), vec3(0.14, 0.20, 0.29), uNightFactor);
            mistTint += vec3(0.72, 0.85, 1.0) * uLightningFlash * 0.90;
            col = mix(col, mistTint, clamp(spindriftMist, 0.0, 0.55));
          }

          col += vec3(0.74, 0.83, 0.92) * rainRingFoam * (0.42 - 0.22 * uNightFactor + 0.55 * uLightningFlash);

          // 6. SUBTLE SHORELINE PROMENADE LAMP GLINT (when uPlanarReflActive > 0.5, Gedung Rektorat UI reflects physically via uPlanarReflTex;
          //    eliminates the static cellular c0 brown-yellow oil-slick stain across East Danau Kenanga Cove!)
          if (uNightFactor > 0.08) {
            if (liveWaterline < 9.5 && vWorldPos.x > 36.0 && vWorldPos.z > -62.0 && vWorldPos.z < 56.0) {
              float lampShimmer = 0.65 + 0.35 * sin(normal.x * 16.0 + normal.z * 14.0 - uTime * 2.2);
              float shoreProx = exp(-liveWaterline * 0.38);
              col += vec3(0.88, 0.58, 0.24) * shoreProx * lampShimmer * uNightFactor * 0.14;
            } else if (uPlanarReflActive < 0.5 && vWorldPos.x > 38.0) {
              float towerAlign = exp(-pow(vWorldPos.z / 20.0, 2.0)) * smoothstep(115.0, 56.0, vWorldPos.x);
              float waveShimmer = 0.68 + 0.32 * sin(normal.x * 18.0 - uTime * 2.4);
              col += vec3(0.88, 0.60, 0.26) * towerAlign * waveShimmer * uNightFactor * 0.16;
            }
          }

          // Subtle outer sea-rim horizon blend complementing 3D boundary-layer aerial perspective (2,680m Kepulauan Seribu rim)
          if (seribuReef.y > 0.01) {
            float surfN = 0.45 + 0.55 * vnoise(R * (vWorldPos.xz * 0.42 - uWindPhase * 0.08));
            totalFoam = max(totalFoam, seribuReef.y * surfN * 0.44 * smoothstep(2450.0, 520.0, distCam));
          }
          // Unify Ocean & Danau Kenanga Cove into True Scene-Referred Linear-HDR -> Boundary-Layer Aerial Perspective
          // -> 0.000-Seam Spherical Far-Sea Rim Merger (r: 1,250m -> 2,640m) -> HVS!
          vec3 hdrOcean = nexusSRGBToLinear(max(col, vec3(0.0))) * mix(1.16, 0.76, uNightFactor);
          hdrOcean += nexusSRGBToLinear(glitterTint) * (specTight * 0.82 + specGlitter * 0.42) * directCelVisO;
          hdrOcean += vec3(0.75, 0.95, 1.45) * uLightningFlash * 0.55;

          vec3 oceanHazeSun = normalize(mix(trueSunO, sunDir, smoothstep(0.65, 0.96, uNightFactor)));
          hdrOcean = nexusApplyBoundaryLayerAerialPerspective(
            hdrOcean,
            cameraPosition,
            vWorldPos,
            oceanHazeSun,
            uDuskFactor,
            uNightFactor,
            uCloudCover,
            uRainIntensity,
            uLightningFlash,
            uFogFactor
          );

          float farRimBlend = smoothstep(1250.0, 2640.0, length(vBaseXZ - ARCHIPELAGO_CENTER));
          if (farRimBlend > 0.001) {
            vec3 rayDirO = normalize(vWorldPos - cameraPosition);
            float HcamO = max(1.2, cameraPosition.y + 1.45);
            float dSeaRimM = length(vWorldPos.xz - cameraPosition.xz);
            vec3 hdrRimSea = evalSphericalFarSeaHDR(
              rayDirO,
              HcamO,
              dSeaRimM,
              sunDir,
              trueSunO,
              moonDirO,
              uDuskFactor,
              uNightFactor,
              uCloudCover,
              uRainIntensity,
              uStormFactor,
              uFogFactor,
              uLightningFlash,
              uWindPhase,
              uSolarDirectTrans
            );
            hdrOcean = mix(hdrOcean, hdrRimSea, farRimBlend);
          }

          vec3 hvsOceanLin = nexusApplyHumanEyeHDR(
            hdrOcean,
            normalize(vWorldPos - cameraPosition),
            sunDir,
            uSunColor,
            uHvsParams,
            uHvsGlare
          );
          gl_FragColor = vec4(nexusLinearToSRGB(hvsOceanLin), 1.0);
        }
      `
    });
  }

  static _smoothstep(e0, e1, x) {
    const t = Math.max(0.0, Math.min(1.0, (x - e0) / (e1 - e0 || 1e-6)));
    return t * t * (3.0 - 2.0 * t);
  }

  static _smootherstep(e0, e1, x) {
    const t = Math.max(0.0, Math.min(1.0, (x - e0) / (e1 - e0 || 1e-6)));
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
  }

  /**
   * Sets user-configurable weather & diurnal transition duration, strictly clamped to [20, 45] seconds.
   */
  setTransitionDuration(seconds) {
    const clamped = Math.min(
      WeatherOpticsEngine.MAX_TRANSITION_SEC,
      Math.max(WeatherOpticsEngine.MIN_TRANSITION_SEC, Number(seconds) || WeatherOpticsEngine.DEFAULT_TRANSITION_SEC)
    );
    this.transitionDurationSec = clamped;

    // Rescale non-proportional preset transitions proportionally so current progress p stays 100% jump-free!
    if (this._weatherTrans.active && !this._weatherTrans.isSpectrumSweep) {
      const p = this._weatherTrans.progress;
      this._weatherTrans.duration = clamped;
      this._weatherTrans.elapsed = p * clamped;
    }
    if (this._timeTrans.active && !this._timeTrans.isProportionalSlider) {
      const p = this._timeTrans.progress;
      this._timeTrans.duration = clamped;
      this._timeTrans.elapsed = p * clamped;
    }
    return clamped;
  }

  getTransitionState() {
    const seaTarget =
      this.manualSeaWaveHeightM !== null
        ? this.manualSeaWaveHeightM
        : this._seaTrans.active
          ? this._seaTrans.targetWaveM
          : this.seaWaveHeightM;
    return {
      durationSec: this.transitionDurationSec,
      anyActive: Boolean(this._weatherTrans.active || this._timeTrans.active || this._seaTrans.active),
      weather: {
        active: this._weatherTrans.active,
        isSpectrumSweep: Boolean(this._weatherTrans.isSpectrumSweep),
        fromMode: this._weatherTrans.fromMode,
        toMode: this.weatherMode,
        liveSpectrum: this.weatherSpectrum,
        targetSpectrum: this.targetWeatherSpectrum,
        duration: this._weatherTrans.duration,
        progress: this._weatherTrans.progress,
        remainingSec: this._weatherTrans.active
          ? Math.max(0.0, this._weatherTrans.duration - this._weatherTrans.elapsed)
          : 0.0
      },
      time: {
        active: this._timeTrans.active,
        isProportionalSlider: Boolean(this._timeTrans.isProportionalSlider),
        fromTod: this._timeTrans.fromTod,
        toTod: this.timeOfDay,
        duration: this._timeTrans.duration,
        progress: this._timeTrans.progress,
        remainingSec: this._timeTrans.active
          ? Math.max(0.0, this._timeTrans.duration - this._timeTrans.elapsed)
          : 0.0,
        solarHour: this.currentSolarHour,
        sliderHour: this._sliderSolarHour,
        targetSliderHour: this._timeTrans.active ? this._timeTrans.targetSliderHour : this._sliderSolarHour,
        continuousSolarHour: this.continuousSolarHour
      },
      sea: {
        active: this._seaTrans.active,
        manualOverride: this.manualSeaWaveHeightM !== null,
        startWaveM: this._seaTrans.startWaveM,
        targetWaveM: seaTarget,
        liveWaveM: this.seaWaveHeightM,
        duration: this._seaTrans.duration,
        progress: this._seaTrans.progress,
        remainingSec: this._seaTrans.active
          ? Math.max(0.0, this._seaTrans.duration - this._seaTrans.elapsed)
          : 0.0
      },
      nightIlluminationLevel: this.nightIlluminationLevel,
      streetlampSensorLevel: this.streetlampSensorLevel
    };
  }

  /**
   * Updates the reference calendar date for real-time Depok lunar ephemeris & sidereal sky.
   */
  setReferenceAstronomyDate(refDate = new Date(), sharedSurfaceUniforms = null) {
    this._baseJD0WIB = computeBaseJD0WIB(refDate);
    this._applyContinuousCelestialAndFog(sharedSurfaceUniforms);
    return this.getAstronomicalState();
  }

  /**
   * Returns live Jean Meeus lunar phase, LST, Bortle sky class, & photocell streetlamp telemetry.
   */
  getAstronomicalState() {
    if (!this._astroState) {
      this._astroState = evaluateDepokAstronomy(
        this._baseJD0WIB,
        this.continuousSolarHour,
        this.uniforms.uWorldToEqMat.value,
        this.uniforms.uMoonDir.value,
        this.uniforms.uMoonPhaseSunDir.value
      );
    }
    return {
      ...this._astroState,
      moonIlluminatedFraction: this._astroState.illuminatedFraction,
      moonPhaseLabelId: this._astroState.phaseNameId,
      illuminatedPercent: +(this._astroState.illuminatedFraction * 100.0).toFixed(1),
      bortleClass: '1–2',
      streetlampSensorLevel: +this.streetlampSensorLevel.toFixed(4),
      continuousSolarHour: +this.continuousSolarHour.toFixed(4)
    };
  }

  _syncParticleVisibility() {
    const rainNow = this.uniforms.uRainIntensity.value;
    const nightNow = this.uniforms.uNightFactor.value;
    const duskNow = this.uniforms.uDuskFactor.value;
    const raining = rainNow > 0.012;

    if (this.rainAndDripMesh) this.rainAndDripMesh.visible = raining;
    if (this.splashAndMistMesh) this.splashAndMistMesh.visible = raining;
    if (this.volcanicPlumeMesh) this.volcanicPlumeMesh.visible = !raining;

    // Strict mutual exclusion (!raining) guarantees total scene draw calls <= 24 <= 25 at EVERY transition millisecond!
    const twilightGlow = Math.max(
      WeatherOpticsEngine._smoothstep(0.08, 0.55, nightNow),
      duskNow * 0.36
    );
    const dryFade = 1.0 - WeatherOpticsEngine._smoothstep(0.002, 0.012, rainNow);
    this.uniforms.uFireflyIntensity.value = twilightGlow * dryFade;

    if (this.fireflyPoints) {
      this.fireflyPoints.visible = !raining && this.uniforms.uFireflyIntensity.value > 0.02;
    }
  }

  static _todModeToSolarHour(mode) {
    const profile = WeatherOpticsEngine.TOD_PROFILES[mode];
    if (profile) return profile.solarHour;
    return WeatherOpticsEngine.TOD_PROFILES.DAY.solarHour;
  }

  static _solarHourToTodMode(solarHour) {
    const h = ((solarHour % 24.0) + 24.0) % 24.0;
    if (h >= 18.5 || h < 5.25) return 'NIGHT';
    if (h >= 5.25 && h < 6.58) return 'DAWN';
    if (h >= 6.58 && h < 10.25) return 'MORNING';
    if (h >= 10.25 && h < 14.5) return 'DAY';
    if (h >= 14.5 && h < 17.08) return 'AFTERNOON';
    return 'DUSK';
  }

  static _hash21JS(px, py) {
    const s = Math.sin(px * 127.1 + py * 311.7) * 43758.5453123;
    return s - Math.floor(s);
  }

  static _noise2JS(px, py) {
    const ix = Math.floor(px);
    const iy = Math.floor(py);
    const fx = px - ix;
    const fy = py - iy;
    const ux = fx * fx * (3.0 - 2.0 * fx);
    const uy = fy * fy * (3.0 - 2.0 * fy);
    const a = WeatherOpticsEngine._hash21JS(ix, iy);
    const b = WeatherOpticsEngine._hash21JS(ix + 1.0, iy);
    const c = WeatherOpticsEngine._hash21JS(ix, iy + 1.0);
    const d = WeatherOpticsEngine._hash21JS(ix + 1.0, iy + 1.0);
    return (a + (b - a) * ux) + ((c + (d - c) * ux) - (a + (b - a) * ux)) * uy;
  }

  static _fbmCloudJS(px, py, octaves = 6, amp0 = 0.52, mult = 2.04, gain = 0.5, ox = 17.1, oy = 31.7) {
    let v = 0.0;
    let a = amp0;
    let x = px;
    let y = py;
    for (let i = 0; i < octaves; i++) {
      v += a * WeatherOpticsEngine._noise2JS(x, y);
      // GLSL mat2(0.8, 0.6, -0.6, 0.8) * p = (0.8*x - 0.6*y, 0.6*x + 0.8*y)
      const rx = (0.8 * x - 0.6 * y) * mult + ox;
      const ry = (0.6 * x + 0.8 * y) * mult + oy;
      x = rx;
      y = ry;
      a *= gain;
    }
    return v;
  }

  /**
   * Evaluates direct solar beam transmittance [0..1] through both the meteorological overcast/rain/fog deck
   * AND the live 3D sky-dome cumulus cloud field directly along uTrueSunDir (3-point solar disc limb average).
   */
  _computeRawSolarDirectTrans(sunDir) {
    const sm = WeatherOpticsEngine._smoothstep;
    const cloudNow = this.uniforms.uCloudCover.value;
    const rainNow = this.uniforms.uRainIntensity.value;
    const stormNow = this.uniforms.uStormFactor.value;
    const fogNow = this.uniforms.uFogFactor.value;

    const cloudBlock = sm(0.54, 0.88, cloudNow);
    const precipFogBlock = sm(0.08, 0.58, rainNow * 0.85 + stormNow * 0.75 + fogNow * 0.82);
    const weatherBeamTrans = (1.0 - 0.96 * cloudBlock) * (1.0 - 0.98 * precipFogBlock);

    if (cloudNow <= 0.28 || sunDir.y <= 0.012 || weatherBeamTrans <= 0.02) {
      return weatherBeamTrans;
    }

    const wx = this.uniforms.uWindPhase.value.x * 0.011;
    const wz = this.uniforms.uWindPhase.value.y * 0.011;
    const nightF = this.uniforms.uNightFactor.value;
    const dryNightClearBias = 0.085 * nightF * (1.0 - Math.min(1.0, Math.max(0.0, rainNow * 2.5 + stormNow)));
    const thresh = 0.58 + (0.20 - 0.58) * Math.max(0.0, Math.min(1.0, cloudNow)) + dryNightClearBias;

    const sampleDiscPoint = (dx, dy, dz) => {
      const len = Math.hypot(dx, dy, dz) || 1.0;
      const sx = dx / len;
      const sy = dy / len;
      const sz = dz / len;
      if (sy <= 0.012) return 0.0;
      const phx = (sx / (sy + 0.26)) * 1.15 + wx * 0.55;
      const phz = (sz / (sy + 0.26)) * 1.15 + wz * 0.55;
      const plx = (sx / (sy + 0.16)) * 1.85 + wx * 1.35 + 5.7;
      const plz = (sz / (sy + 0.16)) * 1.85 + wz * 1.35 - 3.4;
      const fHigh = WeatherOpticsEngine._fbmCloudJS(phx, phz, 6, 0.52, 2.04, 0.5, 17.1, 31.7);
      const fLowBase = WeatherOpticsEngine._fbmCloudJS(plx, plz, 6, 0.52, 2.04, 0.5, 17.1, 31.7);
      const fTurret = WeatherOpticsEngine._fbmCloudJS(
        plx * 3.65 - wx * 0.62 + 4.1,
        plz * 3.65 - wz * 0.62 + 8.7,
        4,
        0.54,
        2.11,
        0.48,
        13.7,
        27.3
      );
      const fLow = Math.max(0.0, Math.min(1.0, fLowBase * 0.84 + (fTurret - 0.36) * 0.26));
      const dHigh = sm(thresh, thresh + 0.34, fHigh);
      const dLow = sm(thresh - 0.04, thresh + 0.26, fLow) * (0.48 + 0.52 * cloudNow);
      const mainCloudDens = Math.max(0.0, Math.min(1.0, dHigh * 0.58 + dLow * 0.80));
      const horizonFade = sm(0.005, 0.16, sy);
      return Math.min(0.985, mainCloudDens * horizonFade * (0.68 + 0.30 * cloudNow));
    };

    const c0 = sampleDiscPoint(sunDir.x, sunDir.y, sunDir.z);
    const c1 = sampleDiscPoint(sunDir.x - sunDir.z * 0.016, sunDir.y + 0.010, sunDir.z + sunDir.x * 0.016);
    const c2 = sampleDiscPoint(sunDir.x + sunDir.z * 0.016, sunDir.y - 0.010, sunDir.z - sunDir.x * 0.016);
    const avgMask = (c0 * 0.50 + c1 * 0.25 + c2 * 0.25);
    const skyDiscCloudTrans = 1.0 - 0.90 * sm(0.42, 0.84, avgMask);
    return weatherBeamTrans * skyDiscCloudTrans;
  }

  static _evalSoftBoxOccJS(ox, oy, oz, invRx, invRy, invRz, minX, minY, minZ, maxX, maxY, maxZ) {
    const tx0 = (minX - ox) * invRx;
    const tx1 = (maxX - ox) * invRx;
    const ty0 = (minY - oy) * invRy;
    const ty1 = (maxY - oy) * invRy;
    const tz0 = (minZ - oz) * invRz;
    const tz1 = (maxZ - oz) * invRz;
    const tNear = Math.max(Math.min(tx0, tx1), Math.min(ty0, ty1), Math.min(tz0, tz1), 0.25);
    const tFar = Math.min(Math.max(tx0, tx1), Math.max(ty0, ty1), Math.max(tz0, tz1));
    const chord = tFar - tNear;
    return WeatherOpticsEngine._smoothstep(0.0, 1.25 + 0.028 * Math.max(0.0, tNear), chord);
  }

  /**
   * Exact 1:1 CPU/JS Mirror of GPU evalWaterCelestialVisibility3D(wp, rayDir):
   * Evaluates 3D ray-world celestial occlusion from any world point (camera or water surface)
   * against horizon, Gedung Rektorat UI (tower + Tajug pyramid + wings), DPPD-STP, Jembatan Teksas,
   * Gunung Gede Bogor, subsea underpass bore, and baked 2048x2048 collision heightmap!
   */
  evalPointCelestialVisibility3DJS(ox, oy, oz, rx, ry, rz) {
    const sm = WeatherOpticsEngine._smoothstep;
    const horVis = sm(-0.024, 0.024, ry);
    if (horVis <= 0.001 || oy < -1.85) return 0.0;

    const safeRx = Math.abs(rx) > 1e-5 ? rx : rx >= 0.0 ? 1e-5 : -1e-5;
    const safeRy = Math.abs(ry) > 1e-5 ? ry : 1e-5;
    const safeRz = Math.abs(rz) > 1e-5 ? rz : rz >= 0.0 ? 1e-5 : -1e-5;
    const invRx = 1.0 / safeRx;
    const invRy = 1.0 / safeRy;
    const invRz = 1.0 / safeRz;
    const rdXZ2 = Math.max(1e-4, rx * rx + rz * rz);

    let occ = 0.0;
    const box = WeatherOpticsEngine._evalSoftBoxOccJS;
    // 1. Gedung Rektorat UI Depok (Cruciform Tower + Steep Tajug Pyramid + Podium & Annexes)
    occ = Math.max(occ, box(ox, oy, oz, invRx, invRy, invRz, -18.8, -0.5, -18.8, 18.8, 33.4, 18.8));
    occ = Math.max(occ, box(ox, oy, oz, invRx, invRy, invRz, -54.2, -0.5, -20.0, -20.0, 12.5, 16.8));
    occ = Math.max(occ, box(ox, oy, oz, invRx, invRy, invRz,  20.0, -0.5, -22.8,  54.2, 12.5, 13.8));
    occ = Math.max(occ, box(ox, oy, oz, invRx, invRy, invRz, -36.2, -0.5, -35.2,  36.2,  9.4, 39.8));
    occ = Math.max(occ, box(ox, oy, oz, invRx, invRy, invRz, -20.6, -0.5,  39.5,  15.6,  7.6, 72.0));
    occ = Math.max(occ, box(ox, oy, oz, invRx, invRy, invRz,  -3.8, -0.5, -52.2,  13.8,  8.8, -35.0));

    const tTajug = -(ox * rx + oz * rz) / rdXZ2;
    if (tTajug > 0.3) {
      const pyT = oy + ry * tTajug;
      if (pyT >= 31.5 && pyT <= 50.6) {
        const pxT = ox + rx * tTajug;
        const pzT = oz + rz * tTajug;
        const uPyr = Math.max(0.0, Math.min(1.0, (pyT - 32.4) / 17.6));
        const halfW = 19.6 + (0.55 - 19.6) * uPyr;
        const penT = 0.35 + 0.015 * tTajug;
        const cheb = Math.max(Math.abs(pxT), Math.abs(pzT));
        const hMask = sm(31.5, 32.6, pyT) * (1.0 - sm(49.6, 50.6, pyT));
        occ = Math.max(occ, (1.0 - sm(halfW - penT, halfW + penT, cheb)) * hMask);
      }
    }

    // 2. Kompleks DPPD & STP UI (-298, 224)
    const tDPPD = -((ox + 298.0) * rx + (oz - 224.0) * rz) / rdXZ2;
    if (tDPPD > 0.3) {
      const pyD = oy + ry * tDPPD;
      if (pyD >= -0.5 && pyD <= 26.8) {
        const rD = Math.hypot(ox + rx * tDPPD + 298.0, oz + rz * tDPPD - 224.0);
        const penD = 0.50 + 0.016 * tDPPD;
        occ = Math.max(occ, (1.0 - sm(38.5 - penD, 38.5 + penD, rD)) * (1.0 - sm(24.8, 26.8, pyD)));
      }
    }

    // 3. Jembatan Teksas UI (Pylons + Arched Deck)
    occ = Math.max(occ, box(ox, oy, oz, invRx, invRy, invRz, -31.2, -1.0, 192.6, -24.8, 35.5, 199.4));
    occ = Math.max(occ, box(ox, oy, oz, invRx, invRy, invRz, -31.2, -1.0, 291.6, -24.8, 35.5, 298.4));
    const tBr = (-28.0 - ox) * invRx;
    if (tBr > 0.3) {
      const pzB = oz + rz * tBr;
      if (pzB >= 168.0 && pzB <= 322.0) {
        const pyB = oy + ry * tBr;
        const uArch = Math.max(-1.0, Math.min(1.0, (pzB - 245.65) / 74.0));
        const deckTop = 4.15 + 4.95 * Math.max(0.0, 1.0 - uArch * uArch) + 0.85;
        const deckBot = deckTop - 2.35;
        const penB = 0.25 + 0.014 * tBr;
        occ = Math.max(occ, sm(deckBot - penB, deckBot + penB, pyB) * (1.0 - sm(deckTop - penB, deckTop + penB, pyB)) * 0.94);
      }
    }

    // 4. Gunung Gede Bogor on Pulau I (-32, 698, y = 58m)
    const tVolc = -((ox + 32.0) * rx + (oz - 698.0) * rz) / rdXZ2;
    if (tVolc > 0.5) {
      const pyV = oy + ry * tVolc;
      const pzV = oz + rz * tVolc;
      if (pyV >= -0.5 && pyV <= 57.8 && pzV > 590.0) {
        const rCone = 94.0 + (15.5 - 94.0) * Math.max(0.0, Math.min(1.0, pyV / 57.5));
        const rHitV = Math.hypot(ox + rx * tVolc + 32.0, pzV - 698.0);
        const penV = 0.85 + 0.018 * tVolc;
        occ = Math.max(occ, (1.0 - sm(rCone - penV, rCone + penV, rHitV)) * (1.0 - sm(55.5, 57.8, pyV)));
      }
    }

    // 5. Baked 2048x2048 Collision Heightfield March (Trees, Hills, ATC, Hanggar)
    const bake = this._bakedCollision;
    if (bake && bake.heightBuffer && bake.bounds && bake.cols > 0 && bake.rows > 0) {
      const bMinX = bake.bounds.x;
      const bMinZ = bake.bounds.y;
      const bSpanX = bake.bounds.z;
      const bSpanZ = bake.bounds.w;
      for (let si = 1; si <= 8; si++) {
        const tM = 2.35 * si * si + 1.5;
        const pyM = oy + ry * tM;
        if (pyM >= 54.0) break;
        const pxM = ox + rx * tM;
        const pzM = oz + rz * tM;
        const u = (pxM - bMinX) / bSpanX;
        const v = (pzM - bMinZ) / bSpanZ;
        if (u > 0.002 && u < 0.998 && v > 0.002 && v < 0.998) {
          const ci = Math.min(bake.cols - 1, Math.max(0, Math.floor(u * bake.cols)));
          const ri = Math.min(bake.rows - 1, Math.max(0, Math.floor(v * bake.rows)));
          const idx = ri * bake.cols + ci;
          const yCol = bake.heightBuffer[idx];
          if (yCol > -0.8) {
            const penM = 0.30 + 0.016 * tM;
            const isCanopy = bake.typeBuffer && bake.typeBuffer[idx] === 2 ? 0.88 : 1.0;
            occ = Math.max(occ, sm(pyM - penM, pyM + penM, yCol) * isCanopy);
          }
        }
      }
    }

    return horVis * Math.max(0.0, Math.min(1.0, 1.0 - occ));
  }

  /**
   * Evaluates continuous 24-hour Kampus UI Depok (Lat -6.3658, UTC+7 WIB) celestial lighting,
   * twilight colors, fog, and architectural night-illumination level with zero discrete jumps.
   */
  _applyContinuousCelestialAndFog(sharedSurfaceUniforms = null, maxTransStep = Infinity) {
    const h = ((this.currentSolarHour % 24.0) + 24.0) % 24.0;
    // Kepulauan UI / Kepulauan Seribu True-North Compass (+X = North, -X = South/Jawa, +Z = East Sunrise, -Z = West Sunset):
    // Sunrise ~05:55 WIB (+Z East over Pulau I), Zenith ~11:58 WIB, Sunset ~18:00 WIB (-Z West over Pulau U & Kepulauan Seribu Barat)
    const sunPhase = ((h - 5.92) / 12.08) * Math.PI;
    const sunElev = Math.sin(sunPhase);
    const sunEastWest = Math.cos(sunPhase);

    const nightF = 1.0 - WeatherOpticsEngine._smoothstep(-0.22, 0.11, sunElev);
    const duskF =
      WeatherOpticsEngine._smoothstep(-0.20, 0.03, sunElev) *
      (1.0 - WeatherOpticsEngine._smoothstep(0.08, 0.42, sunElev));

    this.uniforms.uNightFactor.value = nightF;
    this.uniforms.uDuskFactor.value = duskF;

    // Continuous 24h True Sun Vector & Lunar Vector (+Z = East, -Z = West, +X = North equatorial declination)
    const trueSx = 0.34;
    const trueSy = sunElev * 0.86;
    const trueSz = sunEastWest * 0.86;
    this.uniforms.uTrueSunDir.value.set(trueSx, trueSy, trueSz).normalize();

    const sx = 0.34;
    const sy = Math.max(0.04, sunElev * 0.86);
    const sz = sunEastWest * 0.86;
    const sLen = Math.hypot(sx, sy, sz) || 1.0;

    // v3.7 Real-Time Jean Meeus Depok Lunar Ephemeris & Unwrapped Sidereal Sky Matrix (0.000° midnight jump!)
    // Memoized across sub-second steps (< 0.00025h = 0.9s = 0.0037° sidereal rotation) while updating instantly on any slider drag!
    let astro = this._astroState;
    if (
      !astro ||
      this._lastAstroBaseJD0 !== this._baseJD0WIB ||
      !Number.isFinite(this._lastAstroContHour) ||
      Math.abs(this.continuousSolarHour - this._lastAstroContHour) >= 0.00025
    ) {
      astro = evaluateDepokAstronomy(
        this._baseJD0WIB,
        this.continuousSolarHour,
        this.uniforms.uWorldToEqMat.value,
        this.uniforms.uMoonDir.value,
        this.uniforms.uMoonPhaseSunDir.value
      );
      this._astroState = astro;
      this._lastAstroBaseJD0 = this._baseJD0WIB;
      this._lastAstroContHour = this.continuousSolarHour;
    }
    this.uniforms.uMoonIllumFraction.value = astro.illuminatedFraction;

    // Smooth shadow-map lunar director (tracks real Moon azimuth while keeping shadow frustum elevation >= 0.42)
    const realM = this.uniforms.uMoonDir.value;
    const moonAbove = WeatherOpticsEngine._smoothstep(-0.08, 0.28, realM.y);
    const mx = realM.x * (0.45 + 0.55 * moonAbove);
    const my = 0.54 + 0.30 * Math.max(0.0, realM.y);
    const mz = realM.z * (0.45 + 0.55 * moonAbove) - 0.25 * (1.0 - moonAbove);
    const mLen = Math.hypot(mx, my, mz) || 1.0;

    // Physical Horizon Cutoff + Zero-Intensity Twilight Shadow Handoff:
    // Direct solar beam & cast shadows reach 0.000 right as the solar limb sinks below the sea horizon (sunElev <= -0.026),
    // BEFORE lunar directional light rises (sunElev: -0.21 -> -0.46).
    // Across sunElev in [-0.21, -0.05], _baseSunIntensity === 0.000 while sunLight.position
    // repositions from western sunset to moon -> ZERO shadow-map jump & ZERO post-sunset solar shadows!
    const solarFade = WeatherOpticsEngine._smoothstep(-0.026, 0.34, sunElev);
    const moonBlend = 1.0 - WeatherOpticsEngine._smoothstep(-0.21, -0.05, sunElev);
    const lunarPhaseGain = 0.38 + 0.62 * astro.illuminatedFraction * moonAbove;
    const lunarFade = (1.0 - WeatherOpticsEngine._smoothstep(-0.46, -0.21, sunElev)) * lunarPhaseGain;

    const dirX = (sx / sLen) * (1.0 - moonBlend) + (mx / mLen) * moonBlend;
    const dirY = (sy / sLen) * (1.0 - moonBlend) + (my / mLen) * moonBlend;
    const dirZ = (sz / sLen) * (1.0 - moonBlend) + (mz / mLen) * moonBlend;
    this.uniforms.uSunDir.value.set(dirX, dirY, dirZ).normalize();
    this._baseSunPos.copy(this.uniforms.uSunDir.value).multiplyScalar(128.0);

    // Continuous overcast, rain, fog, & sky-dome cloud direct solar beam transmittance (C^1 rate-limited!)
    const rainNow = this.uniforms.uRainIntensity.value;
    const stormNow = this.uniforms.uStormFactor.value;
    const cloudNow = this.uniforms.uCloudCover.value;
    const stormMix = Math.min(1.0, rainNow * 0.45 + stormNow * 0.55);
    const overcastAtten = Math.max(
      0.32,
      1.0 - rainNow * 0.36 - stormNow * 0.24 - Math.max(0.0, cloudNow - 0.30) * 0.16
    );

    const rawSolarTrans = this._computeRawSolarDirectTrans(this.uniforms.uTrueSunDir.value);
    if (!this._solarTransInitialized || !Number.isFinite(maxTransStep)) {
      this._smoothedSolarDirectTrans = rawSolarTrans;
      this._solarTransInitialized = true;
    } else {
      const dTr = rawSolarTrans - this._smoothedSolarDirectTrans;
      const clampedStep = Math.max(-maxTransStep, Math.min(maxTransStep, dTr));
      this._smoothedSolarDirectTrans = Math.max(0.0, Math.min(1.0, this._smoothedSolarDirectTrans + clampedStep));
    }
    const sunAboveLimb = WeatherOpticsEngine._smoothstep(-0.026, 0.026, sunElev);
    this.uniforms.uSolarDirectTrans.value = this._smoothedSolarDirectTrans * sunAboveLimb;

    const gMode = Math.max(
      0.0,
      Math.min(
        1.0,
        Number(
          (sharedSurfaceUniforms && sharedSurfaceUniforms.uGenshinMode
            ? sharedSurfaceUniforms.uGenshinMode.value
            : this.uniforms.uGenshinMode
              ? this.uniforms.uGenshinMode.value
              : 0.0) || 0.0
        )
      )
    );
    if (this.uniforms.uGenshinMode) {
      this.uniforms.uGenshinMode.value = gMode;
    }

    // Sun / Dusk / Moon directional color
    this._colTmpA.copy(this._colDaySun).lerp(this._colDuskSun, duskF);
    if (gMode > 0.001) {
      this._colTmpB.setHex(0xfff4d6);
      this._colTmpA.lerp(this._colTmpB, gMode * (1.0 - duskF * 0.65));
    }
    this._colTmpA.lerp(this._colStormTint, stormMix * 0.65);
    this._baseSunColor.copy(this._colTmpA).lerp(this._colNightMoon, moonBlend);

    // Direct cast shadows vanish physically when the Sun is covered by thick clouds/overcast/fog/rain,
    // while cloud-scattered solar flux transfers smoothly into diffuse hemisphere & sky fill irradiance!
    const directSunFactor = 0.015 + 0.985 * this._smoothedSolarDirectTrans;
    const lunarClearFactor = Math.max(0.05, 1.0 - rainNow * 0.85 - stormNow * 0.65 - Math.max(0.0, cloudNow - 0.45) * 1.10);
    const dayDuskInt = (2.52 * (1.0 - duskF) + 1.92 * duskF) * solarFade * directSunFactor;
    this._baseSunIntensity = (dayDuskInt * overcastAtten + 0.34 * lunarFade * overcastAtten * lunarClearFactor) * (1.0 + 0.12 * gMode);

    // Hemisphere Sky & Ground colors (Hukum Emas 13: dark night bounce prevents washed-out passive concrete)
    this._colHemiSky.setHex(0xdbeeff);
    if (gMode > 0.001) {
      this._colTmpB.setHex(0x7ecfff);
      this._colHemiSky.lerp(this._colTmpB, gMode * 0.78);
    }
    this._colTmpB.setHex(0xffc99e);
    this._colHemiSky.lerp(this._colTmpB, duskF);
    this._colTmpB.setHex(0x768396);
    this._colHemiSky.lerp(this._colTmpB, stormMix * 0.78);
    this._colTmpB.setHex(0x233454);
    this._colHemiSky.lerp(this._colTmpB, nightF);
    this.hemiLight.color.copy(this._colHemiSky);

    this._colHemiGnd.setHex(0x4c5349);
    if (gMode > 0.001) {
      this._colTmpB.setHex(0x3a6a7d);
      this._colHemiGnd.lerp(this._colTmpB, gMode * 0.74);
    }
    this._colTmpB.setHex(0x3b332e);
    this._colHemiGnd.lerp(this._colTmpB, duskF);
    this._colTmpB.setHex(0x0c1017);
    this._colHemiGnd.lerp(this._colTmpB, nightF);
    this.hemiLight.groundColor.copy(this._colHemiGnd);

    const cloudDiffusedSun = solarFade * (1.0 - this._smoothedSolarDirectTrans) * overcastAtten;
    const dayHemi = (0.66 * (1.0 - duskF) + 0.42 * duskF) * (0.68 + 0.32 * overcastAtten) + 0.22 * cloudDiffusedSun;
    this._baseHemiIntensity = (dayHemi * (1.0 - nightF) + 0.20 * nightF) * (1.0 + 0.26 * gMode * (1.0 - nightF));
    this._baseFillIntensity = (((0.35 * overcastAtten * (1.0 - duskF * 0.42)) + 0.14 * cloudDiffusedSun) * (1.0 - nightF) + 0.10 * nightF) * (1.0 + 0.48 * gMode * (1.0 - nightF));
    this.fillLight.position.copy(this._baseFillPos);
    this.fillLight.color.setHex(0x9ec5eb);
    if (gMode > 0.001) {
      this._colTmpB.setHex(0x66b8ff);
      this.fillLight.color.lerp(this._colTmpB, gMode * 0.82);
    }
    this.fillLight.intensity = this._baseFillIntensity;

    // Isotropic FogExp2 disabled (density = 0.0) in favor of unified 3D Boundary-Layer Aerial Perspective!
    if (this.scene.fog) {
      this._colFog.setHex(0xa8c6df);
      this._colTmpB.setHex(0x6b7b8c);
      this._colFog.lerp(this._colTmpB, Math.min(1.0, rainNow * 0.85));
      this._colTmpB.setHex(0x454f5c);
      this._colFog.lerp(this._colTmpB, stormNow * 0.85);
      this._colTmpB.setHex(0x9e6a4b);
      this._colFog.lerp(this._colTmpB, duskF * (1.0 - stormMix * 0.65));
      this._colTmpB.setHex(0x090f1b);
      this._colFog.lerp(this._colTmpB, nightF);
      this.scene.fog.color.copy(this._colFog);
      this.scene.fog.density = 0.0;
    }

    // Continuous C1-Smooth Architectural & Helipad Night Illumination Level (strictly 0.0 at DAY, zero threshold jump!)
    this.nightIlluminationLevel = 1.0 - WeatherOpticsEngine._smoothstep(-0.26, 0.14, sunElev);

    // Physical NEMA Ambient-Lux Streetlamp Photocell Sensor:
    // - Strictly 0.0000 in clear/misty daytime (uStreetlampPower === 0.000 -> zero daytime emissive!)
    // - Ignites smoothly during golden-hour dusk/dawn twilight AND during heavy dark storm clouds!
    const daySunLux = WeatherOpticsEngine._smoothstep(-0.24, 0.24, sunElev);
    const stormSkyDarkening =
      WeatherOpticsEngine._smoothstep(0.68, 0.96, cloudNow) *
      WeatherOpticsEngine._smoothstep(0.55, 1.15, rainNow + stormNow * 0.65);
    const ambientLuxNorm = daySunLux * (1.0 - 0.94 * stormSkyDarkening);
    this.streetlampSensorLevel = 1.0 - ambientLuxNorm;
    this.uniforms.uStreetlampPower.value = this.streetlampSensorLevel;

    // Shadow-casting sunLight ALWAYS stays on its continuous celestial position (never freezes or teleports!)
    this.sunLight.position.copy(this._baseSunPos);
    this.sunLight.color.copy(this._baseSunColor);
    this.sunLight.intensity = this._baseSunIntensity;
    this.hemiLight.intensity = this._baseHemiIntensity;
    this.uniforms.uSunColor.value.copy(this.sunLight.color);

    this._applyEnvironmentIBL(Math.max(0.0, this.uniforms.uLightningFlash.value));

    if (sharedSurfaceUniforms) {
      this._lastSharedSurfaceUniforms = sharedSurfaceUniforms;
      if (sharedSurfaceUniforms.uIsNight) {
        sharedSurfaceUniforms.uIsNight.value = this.nightIlluminationLevel;
      }
      if (sharedSurfaceUniforms.uStreetlampPower) {
        sharedSurfaceUniforms.uStreetlampPower.value = this.streetlampSensorLevel;
      }
      if (sharedSurfaceUniforms.uNightFactor) {
        sharedSurfaceUniforms.uNightFactor.value = nightF;
      }
      if (sharedSurfaceUniforms.uDuskFactor) {
        sharedSurfaceUniforms.uDuskFactor.value = duskF;
      }
      if (sharedSurfaceUniforms.uCloudCover) {
        sharedSurfaceUniforms.uCloudCover.value = this.uniforms.uCloudCover.value;
      }
      if (sharedSurfaceUniforms.uFogFactor) {
        sharedSurfaceUniforms.uFogFactor.value = this.uniforms.uFogFactor.value;
      }
      if (sharedSurfaceUniforms.uSunDir) {
        sharedSurfaceUniforms.uSunDir.value.copy(this.uniforms.uSunDir.value);
      }
      if (sharedSurfaceUniforms.uSunColor) {
        sharedSurfaceUniforms.uSunColor.value.copy(this.sunLight.color);
      }
      if (sharedSurfaceUniforms.uSolarDirectTrans) {
        sharedSurfaceUniforms.uSolarDirectTrans.value = this.uniforms.uSolarDirectTrans.value;
      }
    }
  }

  /**
   * Synchronizes Human Visual System (HVS) pupil & photoreceptor HDR adaptation uniforms
   * across WeatherOpticsEngine (Sky + Ocean/Fluvial Water) and SharedSurfaceUniforms (100% of Surfaces),
   * including physical 3D solid-body + cloud occlusion of Spencer-Shirley intraocular solar lens glare!
   */
  _syncHumanEyeHDR(dt, camera = null, sharedSurfaceUniforms = null, immediate = false) {
    if (sharedSurfaceUniforms) {
      this._lastSharedSurfaceUniforms = sharedSurfaceUniforms;
    }
    const targetShared = sharedSurfaceUniforms || this._lastSharedSurfaceUniforms;
    const hvsOut = this.hvsOptics.update(
      dt,
      {
        nightFactor: this.uniforms.uNightFactor.value,
        duskFactor: this.uniforms.uDuskFactor.value,
        cloudCover: this.uniforms.uCloudCover.value,
        rainIntensity: this.uniforms.uRainIntensity.value,
        stormFactor: this.uniforms.uStormFactor.value,
        lightningFlash: this.uniforms.uLightningFlash.value,
        sunDir: this.uniforms.uTrueSunDir.value,
        camera,
        immediate
      },
      this.uniforms,
      targetShared
    );

    // Physical 3D Camera-to-Sun Glare Occlusion:
    // Intraocular ciliary solar glare (uHvsGlare.x) NEVER bleeds through Gedung Rektorat UI, walls, volcano, or thick clouds!
    const ts = this.uniforms.uTrueSunDir.value;
    const solarTrans = this.uniforms.uSolarDirectTrans.value;
    const rawCamVis =
      camera && camera.position
        ? this.evalPointCelestialVisibility3DJS(
            camera.position.x,
            camera.position.y,
            camera.position.z,
            ts.x,
            ts.y,
            ts.z
          ) * solarTrans
        : WeatherOpticsEngine._smoothstep(-0.024, 0.024, ts.y) * solarTrans;

    if (immediate || !camera) {
      this._smoothedCamSunVis = rawCamVis;
    } else {
      const kGlare = 1.0 - Math.exp(-Math.max(0.0, dt) * 14.0);
      this._smoothedCamSunVis += (rawCamVis - this._smoothedCamSunVis) * kGlare;
    }
    const glareScale = Math.max(0.0, Math.min(1.0, this._smoothedCamSunVis));
    this.uniforms.uHvsGlare.value.x *= glareScale;
    if (targetShared && targetShared.uHvsGlare && targetShared.uHvsGlare.value) {
      targetShared.uHvsGlare.value.x = this.uniforms.uHvsGlare.value.x;
    }
    return hvsOut;
  }

  /**
   * Returns live Human-Eye Physiological HDR adaptation telemetry for HUD & 4-Gate photometric verification.
   */
  getHumanEyeHDRState() {
    return {
      adaptedLuminance: this.hvsOptics.adaptedLuminance,
      targetLuminance: this.hvsOptics.targetLuminance,
      pupilDiameterMm: this.hvsOptics.pupilDiameterMm,
      visionRegime: this.hvsOptics.visionRegime,
      eyeGain: this.uniforms.uHvsParams.value.x,
      huntChroma: this.uniforms.uHvsParams.value.y,
      stevensContrast: this.uniforms.uHvsParams.value.z,
      purkinjeMix: this.uniforms.uHvsParams.value.w,
      sunGlareStrength: this.uniforms.uHvsGlare.value.x,
      lightningVeil: this.uniforms.uHvsGlare.value.y
    };
  }

  /**
   * Sets or smoothly transitions (20–45s) to an exact Kampus UI Depok WIB solar hour [0, 24).
   */
  setSolarHour(targetSolarHour, sharedSurfaceUniforms = null, options = {}) {
    const normTarget = ((Number(targetSolarHour) % 24.0) + 24.0) % 24.0;
    const nextTod = options.todOverride || WeatherOpticsEngine._solarHourToTodMode(normTarget);
    const prevTod = this.timeOfDay;
    this.timeOfDay = nextTod;

    const startH = ((this.continuousSolarHour % 24.0) + 24.0) % 24.0;
    let delta = ((normTarget - startH + 36.0) % 24.0) - 12.0;
    // Natural forward dawn sunrise sweep when transitioning from NIGHT to DAY!
    if (prevTod === 'NIGHT' && nextTod === 'DAY' && delta < 0.0) {
      delta += 24.0;
    }

    if (options.immediate === true) {
      if (options.preserveAdaptation === true) {
        this.continuousSolarHour += delta;
      } else {
        const dayIndex = Math.floor(this.continuousSolarHour / 24.0);
        this.continuousSolarHour = dayIndex * 24.0 + normTarget;
      }
      this.currentSolarHour = normTarget;
      this._sliderSolarHour = normTarget;
      this._timeTrans.active = false;
      this._timeTrans.isProportionalSlider = false;
      this._timeTrans.fromTod = nextTod;
      this._timeTrans.toTod = nextTod;
      this._timeTrans.startSliderHour = normTarget;
      this._timeTrans.targetSliderHour = normTarget;
      this._timeTrans.progress = 1.0;
      this._timeTrans.elapsed = 0.0;
      this._applyContinuousCelestialAndFog(sharedSurfaceUniforms);
      if (!options.preserveAdaptation) {
        this._syncHumanEyeHDR(0.0, null, sharedSurfaceUniforms, true);
      }
      this._syncParticleVisibility();
      return;
    }

    const dur = Math.min(
      WeatherOpticsEngine.MAX_TRANSITION_SEC,
      Math.max(
        WeatherOpticsEngine.MIN_TRANSITION_SEC,
        Number(options.durationSec ?? this.transitionDurationSec)
      )
    );

    if (Math.abs(delta) < 0.015) {
      this.continuousSolarHour += delta;
      this.currentSolarHour = normTarget;
      this._sliderSolarHour = normTarget;
      this._timeTrans.active = false;
      this._timeTrans.isProportionalSlider = false;
      this._timeTrans.targetSliderHour = normTarget;
      this._timeTrans.progress = 1.0;
      this._applyContinuousCelestialAndFog(sharedSurfaceUniforms);
      if (!options.preserveAdaptation) {
        this._syncHumanEyeHDR(0.0, null, sharedSurfaceUniforms, true);
      }
      this._syncParticleVisibility();
      return;
    }

    this._timeTrans.active = true;
    this._timeTrans.isProportionalSlider = false;
    this._timeTrans.fromTod = prevTod;
    this._timeTrans.toTod = nextTod;
    this._timeTrans.elapsed = 0.0;
    this._timeTrans.duration = dur;
    this._timeTrans.progress = 0.0;
    this._timeTrans.startHour = startH;
    this._timeTrans.startSliderHour = startH;
    this._timeTrans.targetSliderHour = normTarget;
    this._timeTrans.startContinuousHour = this.continuousSolarHour;
    this._timeTrans.deltaHour = delta;
  }

  /**
   * Switches or smoothly transitions (20–45s by default!) Time of Day ('DAY' | 'DUSK' | 'NIGHT').
   * Pass { immediate: true } ONLY for initial startup boot or instant headless test matrix snapshots.
   */
  setTimeOfDay(mode, sharedSurfaceUniforms = null, options = {}) {
    const targetHour =
      options.solarHour !== undefined
        ? Number(options.solarHour)
        : WeatherOpticsEngine._todModeToSolarHour(mode);
    this.setSolarHour(targetHour, sharedSurfaceUniforms, {
      ...options,
      todOverride: mode
    });
  }

  static _getWeatherPresetTargets(weatherMode, options = {}) {
    const profile = WeatherOpticsEngine.WEATHER_PROFILES[weatherMode];
    if (profile && weatherMode !== 'CLEAR') {
      return {
        rain: profile.rain,
        wet: profile.wet,
        cloud:
          options.cloudCoverNorm !== undefined && profile.rain === 0.0
            ? Math.max(profile.cloud * 0.75, Math.min(0.96, Number(options.cloudCoverNorm)))
            : profile.cloud,
        storm: profile.storm,
        fog: profile.fog,
        wind: profile.wind
      };
    }
    // CLEAR (supports live Depok Open-Meteo cloudCoverNorm override!)
    const liveCloud =
      options.cloudCoverNorm !== undefined
        ? Math.max(0.14, Math.min(0.86, Number(options.cloudCoverNorm)))
        : 0.22;
    return {
      rain: 0.0,
      wet: 0.0,
      cloud: liveCloud,
      storm: 0.0,
      fog: 0.0,
      wind: 0.82
    };
  }

  _computeAutoSeaWaveHeightM() {
    const windChopBoost = Math.max(0.0, this.baseWindSpeed - 1.35) * 0.22;
    return Math.max(
      0.0,
      Math.min(
        WeatherOpticsEngine.MAX_SEA_WAVE_M,
        0.18 +
          this.uniforms.uRainIntensity.value * 0.62 +
          this.uniforms.uStormFactor.value * 0.88 +
          windChopBoost
      )
    );
  }

  /**
   * Continuous [0.00 .. 24.00] WIB solar/lunar spectrum slider control:
   * - Default (options.immediate !== false && !options.proportional45s): instant apply (headless snapshot tests).
   * - Proportional 45s slider mode ({ immediate: false } or { proportional45s: true }):
   *   Duration T = 45.0 * |targetHour - liveSliderHour| / 24.0 seconds (0->24 = 45.0s, 0->12 = 22.5s),
   *   preserving cruise velocity across continuous 60 Hz slider drag events!
   */
  setContinuousTimeHours(hour24, sharedSurfaceUniforms = null, options = {}) {
    if (sharedSurfaceUniforms) {
      this._lastSharedSurfaceUniforms = sharedSurfaceUniforms;
    }
    const clamped = Math.max(0.0, Math.min(24.0, Number(hour24) || 0.0));
    const normTarget = clamped >= 24.0 ? 23.998 : clamped;
    const useProportional = options.proportional45s === true || options.immediate === false;

    if (!useProportional) {
      const dayIndex = Math.floor(this.continuousSolarHour / 24.0);
      this.continuousSolarHour = dayIndex * 24.0 + clamped;
      this.currentSolarHour = normTarget;
      this._sliderSolarHour = clamped;
      this.timeOfDay = WeatherOpticsEngine._solarHourToTodMode(normTarget);
      this._timeTrans.active = false;
      this._timeTrans.isProportionalSlider = false;
      this._timeTrans.fromTod = this.timeOfDay;
      this._timeTrans.toTod = this.timeOfDay;
      this._timeTrans.startSliderHour = clamped;
      this._timeTrans.targetSliderHour = clamped;
      this._timeTrans.progress = 1.0;
      this._timeTrans.elapsed = 0.0;
      this._applyContinuousCelestialAndFog(sharedSurfaceUniforms);
      if (options.immediateAdaptation === true) {
        this._syncHumanEyeHDR(0.0, null, sharedSurfaceUniforms, true);
      }
      this._syncParticleVisibility();
      return this.currentSolarHour;
    }

    const liveSliderH = Number.isFinite(this._sliderSolarHour) ? this._sliderSolarHour : this.currentSolarHour;
    const delta = clamped - liveSliderH;
    const dur = WeatherOpticsEngine.SLIDER_FULL_SPAN_SEC * (Math.abs(delta) / 24.0);
    const nextTod = WeatherOpticsEngine._solarHourToTodMode(normTarget);

    if (Math.abs(delta) < 1e-4 || dur < 0.015) {
      this.continuousSolarHour += delta;
      this._sliderSolarHour = clamped;
      this.currentSolarHour = normTarget;
      this.timeOfDay = nextTod;
      this._timeTrans.active = false;
      this._timeTrans.isProportionalSlider = false;
      this._timeTrans.fromTod = nextTod;
      this._timeTrans.toTod = nextTod;
      this._timeTrans.startSliderHour = clamped;
      this._timeTrans.targetSliderHour = clamped;
      this._timeTrans.progress = 1.0;
      this._timeTrans.elapsed = 0.0;
      this._applyContinuousCelestialAndFog(sharedSurfaceUniforms);
      this._syncParticleVisibility();
      return this.currentSolarHour;
    }

    const newDir = delta >= 0.0 ? 1 : -1;
    const keepCruise = this._timeTrans.active && this._timeTrans.dir === newDir;
    const prevLaunchClock = keepCruise ? this._timeTrans.launchClock : 0.0;

    this._timeTrans.active = true;
    this._timeTrans.isProportionalSlider = true;
    this._timeTrans.fromTod = WeatherOpticsEngine._solarHourToTodMode(this.currentSolarHour);
    this._timeTrans.toTod = nextTod;
    this.timeOfDay = nextTod;
    this._timeTrans.dir = newDir;
    this._timeTrans.launchClock = prevLaunchClock;
    this._timeTrans.launchRampSec = Math.max(0.0, 0.35 - prevLaunchClock);
    this._timeTrans.elapsed = 0.0;
    this._timeTrans.duration = dur;
    this._timeTrans.progress = 0.0;
    this._timeTrans.startHour = this.currentSolarHour;
    this._timeTrans.startSliderHour = liveSliderH;
    this._timeTrans.targetSliderHour = clamped;
    this._timeTrans.startContinuousHour = this.continuousSolarHour;
    this._timeTrans.deltaHour = delta;
    return this.currentSolarHour;
  }

  getSliderSolarHour() {
    return this._sliderSolarHour;
  }

  getTargetSliderSolarHour() {
    return this._timeTrans.active ? this._timeTrans.targetSliderHour : this._sliderSolarHour;
  }

  /**
   * Continuous [0.00 .. 1.00] physical weather spectrum control (Cerah -> Mendung -> Kabut -> Hujan -> Badai):
   * - Default (options.immediate !== false && !options.proportional45s): instant apply.
   * - Proportional 45s slider mode ({ immediate: false } or { proportional45s: true }):
   *   Duration T = 45.0 * |targetU - liveU| seconds (0% -> 100% = 45.0s, 50% span = 22.5s),
   *   sweeping smoothly across all intermediate meteorological regimes with zero initial discontinuity!
   */
  setWeatherSpectrum(u01, sharedSurfaceUniforms = null, options = {}) {
    if (sharedSurfaceUniforms) {
      this._lastSharedSurfaceUniforms = sharedSurfaceUniforms;
    }
    const spec = WeatherOpticsEngine.evaluateWeatherSpectrum(u01);
    const useProportional = options.proportional45s === true || options.immediate === false;

    this.targetWeatherSpectrum = spec.u;
    this.weatherMode = spec.mode;
    this.targetRainIntensity = spec.rain;
    this.targetWetness = spec.wet;
    this.targetCloudCover = spec.cloud;
    this.targetStormFactor = spec.storm;
    this.targetFogDensity = spec.fog;
    this.targetWindSpeed = spec.wind;

    if (!useProportional) {
      this.weatherSpectrum = spec.u;
      this._weatherTrans.active = false;
      this._weatherTrans.isSpectrumSweep = false;
      this._weatherTrans.fromMode = spec.mode;
      this._weatherTrans.toMode = spec.mode;
      this._weatherTrans.startSpectrum = spec.u;
      this._weatherTrans.targetSpectrum = spec.u;
      this._weatherTrans.progress = 1.0;
      this._weatherTrans.elapsed = 0.0;

      this.uniforms.uRainIntensity.value = spec.rain;
      this.uniforms.uWetness.value = spec.wet;
      this.uniforms.uCloudCover.value = spec.cloud;
      this.uniforms.uStormFactor.value = spec.storm;
      this.uniforms.uFogFactor.value = spec.fog;
      this.baseWindSpeed = spec.wind;
      if (this.scene.fog) {
        this.scene.fog.density = 0.0;
      }
      if (spec.storm > 0.62) {
        this.nextLightningTimer = Math.min(this.nextLightningTimer, 0.85);
      } else {
        this.uniforms.uLightningFlash.value = 0.0;
        this.lightningClock = -1.0;
      }
      if (this.manualSeaWaveHeightM === null) {
        this.seaWaveHeightM = spec.autoWaveM;
        this._seaTrans.active = false;
        this._seaTrans.startWaveM = spec.autoWaveM;
        this._seaTrans.targetWaveM = spec.autoWaveM;
        this._syncOceanSpectrumUniforms();
      }
      this._applyContinuousCelestialAndFog(sharedSurfaceUniforms);
      if (options.immediateAdaptation === true) {
        this._syncHumanEyeHDR(0.0, null, sharedSurfaceUniforms, true);
      }
      this._syncParticleVisibility();
      return spec;
    }

    const liveU = Number.isFinite(this.weatherSpectrum) ? this.weatherSpectrum : 0.0;
    const deltaU = spec.u - liveU;
    const dur = WeatherOpticsEngine.SLIDER_FULL_SPAN_SEC * Math.abs(deltaU);

    if (Math.abs(deltaU) < 1e-4 || dur < 0.015) {
      this.weatherSpectrum = spec.u;
      this._weatherTrans.active = false;
      this._weatherTrans.isSpectrumSweep = false;
      this._weatherTrans.fromMode = spec.mode;
      this._weatherTrans.toMode = spec.mode;
      this._weatherTrans.startSpectrum = spec.u;
      this._weatherTrans.targetSpectrum = spec.u;
      this._weatherTrans.progress = 1.0;
      this._weatherTrans.elapsed = 0.0;
      this.uniforms.uRainIntensity.value = spec.rain;
      this.uniforms.uWetness.value = spec.wet;
      this.uniforms.uCloudCover.value = spec.cloud;
      this.uniforms.uStormFactor.value = spec.storm;
      this.uniforms.uFogFactor.value = spec.fog;
      this.baseWindSpeed = spec.wind;
      if (this.manualSeaWaveHeightM === null && !this._seaTrans.active) {
        this.seaWaveHeightM = spec.autoWaveM;
        this._syncOceanSpectrumUniforms();
      }
      this._applyContinuousCelestialAndFog(sharedSurfaceUniforms);
      this._syncParticleVisibility();
      return spec;
    }

    const startKnotSpec = WeatherOpticsEngine.evaluateWeatherSpectrum(liveU);
    const newDir = deltaU >= 0.0 ? 1 : -1;
    const keepCruise = this._weatherTrans.active && this._weatherTrans.dir === newDir;
    const prevLaunchClock = keepCruise ? this._weatherTrans.launchClock : 0.0;

    this._weatherTrans.active = true;
    this._weatherTrans.isSpectrumSweep = true;
    this._weatherTrans.fromMode = startKnotSpec.mode;
    this._weatherTrans.toMode = spec.mode;
    this._weatherTrans.startSpectrum = liveU;
    this._weatherTrans.targetSpectrum = spec.u;
    this._weatherTrans.dir = newDir;
    this._weatherTrans.launchClock = prevLaunchClock;
    this._weatherTrans.launchRampSec = Math.max(0.0, 0.35 - prevLaunchClock);
    this._weatherTrans.elapsed = 0.0;
    this._weatherTrans.duration = dur;
    this._weatherTrans.progress = 0.0;
    this._weatherTrans.startRain = this.uniforms.uRainIntensity.value;
    this._weatherTrans.startWet = this.uniforms.uWetness.value;
    this._weatherTrans.startCloud = this.uniforms.uCloudCover.value;
    this._weatherTrans.startStorm = this.uniforms.uStormFactor.value;
    this._weatherTrans.startFog = this.uniforms.uFogFactor.value;
    this._weatherTrans.startWind = this.baseWindSpeed;
    // Residual offsets from knot curve at t=0 ensure 0.000000 jump on mid-flight retargeting AND exact convergence at t=duration!
    this._weatherTrans.resRain = this.uniforms.uRainIntensity.value - startKnotSpec.rain;
    this._weatherTrans.resWet = this.uniforms.uWetness.value - startKnotSpec.wet;
    this._weatherTrans.resCloud = this.uniforms.uCloudCover.value - startKnotSpec.cloud;
    this._weatherTrans.resStorm = this.uniforms.uStormFactor.value - startKnotSpec.storm;
    this._weatherTrans.resFog = this.uniforms.uFogFactor.value - startKnotSpec.fog;
    this._weatherTrans.resWind = this.baseWindSpeed - startKnotSpec.wind;
    this._weatherTrans.startAutoWaveM = this.seaWaveHeightM;
    return spec;
  }

  getWeatherSpectrum() {
    return this.weatherSpectrum;
  }

  getTargetWeatherSpectrum() {
    return this._weatherTrans.active && this._weatherTrans.isSpectrumSweep
      ? this._weatherTrans.targetSpectrum
      : this.targetWeatherSpectrum;
  }

  /**
   * Continuous [0.00 .. 3.00] meter ocean wave height slider control (pass null or { auto: true } for weather-coupled auto):
   * - Default (options.immediate !== false && !options.proportional45s): instant apply.
   * - Proportional 45s slider mode ({ immediate: false } or { proportional45s: true }):
   *   Duration T = 45.0 * |targetM - liveWaveM| / 3.0 seconds (0.0m -> 3.0m = 45.0s, 0.0m -> 1.5m = 22.5s)!
   */
  setSeaWaveHeightMeters(meters, options = {}) {
    const useProportional = options.proportional45s === true || options.immediate === false;
    const isAutoRequest = meters === null || meters === undefined || options.auto === true;

    if (isAutoRequest) {
      this.manualSeaWaveHeightM = null;
      const autoTarget = this._weatherTrans.active && this._weatherTrans.isSpectrumSweep
        ? WeatherOpticsEngine.evaluateWeatherSpectrum(this._weatherTrans.targetSpectrum).autoWaveM
        : this._computeAutoSeaWaveHeightM();

      if (!useProportional) {
        this._seaTrans.active = false;
        this._seaTrans.progress = 1.0;
        this._seaTrans.elapsed = 0.0;
        this._seaTrans.startWaveM = autoTarget;
        this._seaTrans.targetWaveM = autoTarget;
        if (options.immediate === true) {
          this.seaWaveHeightM = autoTarget;
          this._syncOceanSpectrumUniforms();
        }
        return this.seaWaveHeightM;
      }

      const deltaAuto = autoTarget - this.seaWaveHeightM;
      const durAuto =
        WeatherOpticsEngine.SLIDER_FULL_SPAN_SEC *
        (Math.abs(deltaAuto) / WeatherOpticsEngine.MAX_SEA_WAVE_M);
      if (Math.abs(deltaAuto) < 1e-4 || durAuto < 0.015) {
        this.seaWaveHeightM = autoTarget;
        this._seaTrans.active = false;
        this._seaTrans.progress = 1.0;
        this._seaTrans.startWaveM = autoTarget;
        this._seaTrans.targetWaveM = autoTarget;
        this._syncOceanSpectrumUniforms();
        return this.seaWaveHeightM;
      }

      const dirAuto = deltaAuto >= 0.0 ? 1 : -1;
      const keepCruiseAuto = this._seaTrans.active && this._seaTrans.dir === dirAuto;
      const prevClockAuto = keepCruiseAuto ? this._seaTrans.launchClock : 0.0;
      this._seaTrans.active = true;
      this._seaTrans.dir = dirAuto;
      this._seaTrans.launchClock = prevClockAuto;
      this._seaTrans.launchRampSec = Math.max(0.0, 0.35 - prevClockAuto);
      this._seaTrans.elapsed = 0.0;
      this._seaTrans.duration = durAuto;
      this._seaTrans.progress = 0.0;
      this._seaTrans.startWaveM = this.seaWaveHeightM;
      this._seaTrans.targetWaveM = autoTarget;
      return this.seaWaveHeightM;
    }

    const h = Math.max(0.0, Math.min(WeatherOpticsEngine.MAX_SEA_WAVE_M, Number(meters) || 0.0));
    this.manualSeaWaveHeightM = h;

    if (!useProportional) {
      this.seaWaveHeightM = h;
      this._seaTrans.active = false;
      this._seaTrans.startWaveM = h;
      this._seaTrans.targetWaveM = h;
      this._seaTrans.progress = 1.0;
      this._seaTrans.elapsed = 0.0;
      this._syncOceanSpectrumUniforms();
      return this.seaWaveHeightM;
    }

    const deltaH = h - this.seaWaveHeightM;
    const dur =
      WeatherOpticsEngine.SLIDER_FULL_SPAN_SEC *
      (Math.abs(deltaH) / WeatherOpticsEngine.MAX_SEA_WAVE_M);
    if (Math.abs(deltaH) < 1e-4 || dur < 0.015) {
      this.seaWaveHeightM = h;
      this._seaTrans.active = false;
      this._seaTrans.startWaveM = h;
      this._seaTrans.targetWaveM = h;
      this._seaTrans.progress = 1.0;
      this._seaTrans.elapsed = 0.0;
      this._syncOceanSpectrumUniforms();
      return this.seaWaveHeightM;
    }

    const newDir = deltaH >= 0.0 ? 1 : -1;
    const keepCruise = this._seaTrans.active && this._seaTrans.dir === newDir;
    const prevLaunchClock = keepCruise ? this._seaTrans.launchClock : 0.0;

    this._seaTrans.active = true;
    this._seaTrans.dir = newDir;
    this._seaTrans.launchClock = prevLaunchClock;
    this._seaTrans.launchRampSec = Math.max(0.0, 0.35 - prevLaunchClock);
    this._seaTrans.elapsed = 0.0;
    this._seaTrans.duration = dur;
    this._seaTrans.progress = 0.0;
    this._seaTrans.startWaveM = this.seaWaveHeightM;
    this._seaTrans.targetWaveM = h;
    return this.seaWaveHeightM;
  }

  getSeaWaveHeightMeters() {
    return this.seaWaveHeightM;
  }

  getTargetSeaWaveHeightMeters() {
    if (this.manualSeaWaveHeightM !== null) return this.manualSeaWaveHeightM;
    if (this._seaTrans.active) return this._seaTrans.targetWaveM;
    return this.seaWaveHeightM;
  }

  isManualSeaState() {
    return this.manualSeaWaveHeightM !== null;
  }

  /**
   * Immediately completes any active Time, Weather, and Sea transitions at their exact user targets.
   */
  skipActiveTransitions(sharedSurfaceUniforms = null) {
    const targetShared = sharedSurfaceUniforms || this._lastSharedSurfaceUniforms;
    if (this._timeTrans.active) {
      if (this._timeTrans.isProportionalSlider) {
        this.setContinuousTimeHours(this._timeTrans.targetSliderHour, targetShared, {
          immediate: true,
          immediateAdaptation: true
        });
      } else {
        this.setSolarHour(this._timeTrans.targetSliderHour, targetShared, {
          immediate: true,
          todOverride: this.timeOfDay
        });
      }
    }
    if (this._weatherTrans.active) {
      if (this._weatherTrans.isSpectrumSweep) {
        this.setWeatherSpectrum(this._weatherTrans.targetSpectrum, targetShared, {
          immediate: true,
          immediateAdaptation: true
        });
      } else {
        this.setWeather(this.weatherMode, { immediate: true });
      }
    }
    if (this._seaTrans.active) {
      if (this.manualSeaWaveHeightM !== null) {
        this.setSeaWaveHeightMeters(this.manualSeaWaveHeightM, { immediate: true });
      } else {
        this.setSeaWaveHeightMeters(null, { auto: true, immediate: true });
      }
    }
    this._syncOceanSpectrumUniforms();
    this._applyContinuousCelestialAndFog(targetShared);
    this._syncHumanEyeHDR(0.0, null, targetShared, true);
    this._syncParticleVisibility();
  }

  /**
   * Switches active meteorological regime with mandatory 20–45s staged physical transition by default!
   * - Clouds, fog & wind lead precipitation onset; ground wetness builds only as raindrops fall;
   * - When clearing, rain tapers gradually to drizzle -> 0, then wet ground evaporates slowly;
   * - Pass { immediate: true } ONLY for initial startup boot or instant headless test matrix snapshots.
   */
  setWeather(weatherMode, options = {}) {
    const prevMode = this.weatherMode;
    this.weatherMode = WeatherOpticsEngine.WEATHER_PROFILES[weatherMode] ? weatherMode : 'CLEAR';
    const specU = WeatherOpticsEngine.weatherModeToSpectrum(this.weatherMode);
    this.weatherSpectrum = specU;
    this.targetWeatherSpectrum = specU;
    const preset = WeatherOpticsEngine._getWeatherPresetTargets(this.weatherMode, options);

    this.targetRainIntensity = preset.rain;
    this.targetWetness = preset.wet;
    this.targetCloudCover = preset.cloud;
    this.targetStormFactor = preset.storm;
    this.targetFogDensity = preset.fog;
    this.targetWindSpeed = preset.wind;

    if (options.immediate === true) {
      this._weatherTrans.active = false;
      this._weatherTrans.isSpectrumSweep = false;
      this._weatherTrans.fromMode = this.weatherMode;
      this._weatherTrans.toMode = this.weatherMode;
      this._weatherTrans.startSpectrum = specU;
      this._weatherTrans.targetSpectrum = specU;
      this._weatherTrans.progress = 1.0;
      this._weatherTrans.elapsed = 0.0;

      this.uniforms.uRainIntensity.value = preset.rain;
      if (!options.preserveWetness) {
        this.uniforms.uWetness.value = preset.wet;
      }
      this.uniforms.uCloudCover.value = preset.cloud;
      this.uniforms.uStormFactor.value = preset.storm;
      this.uniforms.uFogFactor.value = preset.fog;
      this.baseWindSpeed = preset.wind;
      if (this.scene.fog) {
        this.scene.fog.density = 0.0;
      }
      if (this.weatherMode === 'STORM') {
        this.nextLightningTimer = Math.min(this.nextLightningTimer, 0.85);
      } else {
        this.uniforms.uLightningFlash.value = 0.0;
        this.lightningClock = -1.0;
      }
      if (this.manualSeaWaveHeightM === null) {
        this.seaWaveHeightM = this._computeAutoSeaWaveHeightM();
        this._seaTrans.active = false;
        this._syncOceanSpectrumUniforms();
      }
      this._applyContinuousCelestialAndFog();
      this._syncHumanEyeHDR(0.0, null, null, true);
      this._syncParticleVisibility();
      return;
    }

    const dur = Math.min(
      WeatherOpticsEngine.MAX_TRANSITION_SEC,
      Math.max(
        WeatherOpticsEngine.MIN_TRANSITION_SEC,
        Number(options.durationSec ?? this.transitionDurationSec)
      )
    );

    const deltaSum =
      Math.abs(this.uniforms.uRainIntensity.value - preset.rain) +
      Math.abs(this.uniforms.uWetness.value - preset.wet) +
      Math.abs(this.uniforms.uCloudCover.value - preset.cloud) +
      Math.abs(this.uniforms.uStormFactor.value - preset.storm) +
      Math.abs(this.uniforms.uFogFactor.value - preset.fog) +
      Math.abs(this.baseWindSpeed - preset.wind) * 0.25;

    if (deltaSum < 0.015 && !this._weatherTrans.active) {
      this._weatherTrans.fromMode = this.weatherMode;
      this._weatherTrans.toMode = this.weatherMode;
      this._weatherTrans.progress = 1.0;
      return;
    }

    this._weatherTrans.active = true;
    this._weatherTrans.isSpectrumSweep = false;
    this._weatherTrans.fromMode = prevMode;
    this._weatherTrans.toMode = this.weatherMode;
    this._weatherTrans.elapsed = 0.0;
    this._weatherTrans.duration = dur;
    this._weatherTrans.progress = 0.0;
    this._weatherTrans.startRain = this.uniforms.uRainIntensity.value;
    this._weatherTrans.startWet = this.uniforms.uWetness.value;
    this._weatherTrans.startCloud = this.uniforms.uCloudCover.value;
    this._weatherTrans.startStorm = this.uniforms.uStormFactor.value;
    this._weatherTrans.startFog = this.uniforms.uFogFactor.value;
    this._weatherTrans.startWind = this.baseWindSpeed;
  }

  getWindGustMultiplier() {
    return this.windGustMultiplier;
  }

  _triggerLightningDischarge(timeSec) {
    const az = ((Math.sin(timeSec * 12.73) * 0.5 + 0.5) * Math.PI * 1.8) - Math.PI * 0.9;
    this.uniforms.uLightningDir.value.set(Math.cos(az), 0.34, Math.sin(az)).normalize();
    this.uniforms.uLightningSeed.value = 1.0 + (Math.floor(Math.abs(timeSec * 17.37) + 1.0) % 991);
    this.lightningClock = 0.0;

    const isTriple = Math.cos(timeSec * 7.9) > 0.05;
    const peakIntensity = 0.98 + 0.38 * (0.5 + 0.5 * Math.sin(timeSec * 19.1));
    this.lightningStrokeProfile = { isTriple, peakIntensity, az };

    const distMeters = 185 + Math.abs(Math.sin(timeSec * 23.4)) * 540;
    if (typeof this.onThunder === 'function') {
      this.onThunder(distMeters, peakIntensity, false);
    }
  }

  /**
   * Triggers an immediate on-demand lightning bolt in front of the active camera
   * and dispatches an instant close-range thunder detonation (for UI audition & testing).
   */
  triggerManualLightningStrike(camera = null) {
    const nowSec = performance.now() * 0.001;
    let dx = 0.55;
    let dz = -0.82;
    if (camera && camera.matrixWorld && camera.matrixWorld.elements) {
      const e = camera.matrixWorld.elements;
      dx = -e[8] + (Math.sin(nowSec * 11.3) * 0.18);
      dz = -e[10] + (Math.cos(nowSec * 11.3) * 0.18);
    }
    this.uniforms.uLightningDir.value.set(dx, 0.36, dz).normalize();
    this.uniforms.uLightningSeed.value = 1.0 + (Math.floor(Math.abs(nowSec * 37.13) + 7.0) % 991);
    this.lightningClock = 0.0;
    this.lightningStrokeProfile = { isTriple: true, peakIntensity: 1.38, az: Math.atan2(dz, dx) };
    this.uniforms.uLightningFlash.value = 1.22;
    if (typeof this.onThunder === 'function') {
      this.onThunder(195, 1.48, true);
    }
  }

  _evaluateLightningEnvelope(t) {
    if (!this.lightningStrokeProfile || t < 0.0 || t > 0.52) return 0.0;
    const p = this.lightningStrokeProfile.peakIntensity;
    let env = Math.exp(-Math.pow((t - 0.025) / 0.022, 2.0)) * p;
    env += Math.exp(-Math.pow((t - 0.135) / 0.028, 2.0)) * p * 0.74;
    if (this.lightningStrokeProfile.isTriple) {
      env += Math.exp(-Math.pow((t - 0.265) / 0.036, 2.0)) * p * 0.48;
    }
    env += Math.exp(-t * 7.5) * 0.18 * p;
    return Math.min(1.25, env);
  }

  update(dt, camera, sharedSurfaceUniforms = null) {
    const safeDt = Math.min(0.1, Math.max(0.0, dt));
    const transDt = Math.min(2.0, Math.max(0.0, dt));
    this.uniforms.uTime.value += safeDt;
    const t = this.uniforms.uTime.value;

    // 1. Staged Physical Meteorological Transition OR Proportional 45s Spectrum Sweep (ZERO sudden jumps!)
    if (this._weatherTrans.active) {
      const s = this._weatherTrans;
      s.elapsed += transDt;
      s.launchClock += transDt;
      const durW = Math.max(0.015, s.duration);
      const p = Math.min(1.0, s.elapsed / durW);
      s.progress = p;

      if (s.isSpectrumSweep) {
        const easeW = WeatherOpticsEngine._evalProportionalEase(
          s.elapsed,
          durW,
          s.launchRampSec,
          0.35
        );
        const liveU = s.startSpectrum + (s.targetSpectrum - s.startSpectrum) * easeW;
        this.weatherSpectrum = liveU;
        const liveSpec = WeatherOpticsEngine.evaluateWeatherSpectrum(liveU);
        const decay = 1.0 - easeW;

        this.uniforms.uCloudCover.value = Math.max(
          0.0,
          Math.min(1.0, liveSpec.cloud + (s.resCloud || 0.0) * decay)
        );
        this.baseWindSpeed = Math.max(0.2, liveSpec.wind + (s.resWind || 0.0) * decay);
        const nextRain = Math.max(0.0, liveSpec.rain + (s.resRain || 0.0) * decay);
        this.uniforms.uRainIntensity.value = nextRain < 0.004 && liveSpec.rain === 0.0 ? 0.0 : nextRain;
        this.uniforms.uStormFactor.value = Math.max(
          0.0,
          Math.min(1.0, liveSpec.storm + (s.resStorm || 0.0) * decay)
        );
        this.uniforms.uFogFactor.value = Math.max(
          0.0,
          Math.min(1.0, liveSpec.fog + (s.resFog || 0.0) * decay)
        );
        this.uniforms.uWetness.value = Math.max(
          0.0,
          Math.min(1.0, liveSpec.wet + (s.resWet || 0.0) * decay)
        );
        if (this.manualSeaWaveHeightM === null && !this._seaTrans.active) {
          const resSea = (s.startAutoWaveM ?? liveSpec.autoWaveM) - WeatherOpticsEngine.evaluateWeatherSpectrum(s.startSpectrum).autoWaveM;
          this.seaWaveHeightM = Math.max(
            0.0,
            Math.min(WeatherOpticsEngine.MAX_SEA_WAVE_M, liveSpec.autoWaveM + resSea * decay)
          );
        }
        if (this.scene.fog) {
          this.scene.fog.density = 0.0;
        }
        if (p >= 1.0) {
          this.weatherSpectrum = s.targetSpectrum;
          this.weatherMode = liveSpec.mode;
          s.active = false;
          s.isSpectrumSweep = false;
          s.fromMode = this.weatherMode;
        }
      } else {
        const gettingWetter = this.targetRainIntensity >= s.startRain;
        const pCloud = gettingWetter
          ? WeatherOpticsEngine._smootherstep(0.0, 0.58, p)
          : WeatherOpticsEngine._smootherstep(0.18, 0.92, p);
        const pWind = WeatherOpticsEngine._smootherstep(0.04, 0.80, p);
        const pRain = gettingWetter
          ? WeatherOpticsEngine._smootherstep(0.16, 0.88, p)
          : WeatherOpticsEngine._smootherstep(0.0, 0.64, p);
        const pStorm = gettingWetter
          ? WeatherOpticsEngine._smootherstep(0.22, 0.94, p)
          : WeatherOpticsEngine._smootherstep(0.0, 0.50, p);
        const pWet = gettingWetter
          ? WeatherOpticsEngine._smootherstep(0.28, 1.0, p)
          : WeatherOpticsEngine._smootherstep(0.30, 1.0, p);
        const pFog = WeatherOpticsEngine._smootherstep(0.06, 0.86, p);

        this.uniforms.uCloudCover.value = s.startCloud + (this.targetCloudCover - s.startCloud) * pCloud;
        this.baseWindSpeed = s.startWind + (this.targetWindSpeed - s.startWind) * pWind;
        this.uniforms.uRainIntensity.value = s.startRain + (this.targetRainIntensity - s.startRain) * pRain;
        if (this.targetRainIntensity === 0.0 && this.uniforms.uRainIntensity.value < 0.004) {
          this.uniforms.uRainIntensity.value = 0.0;
        }
        this.uniforms.uStormFactor.value = s.startStorm + (this.targetStormFactor - s.startStorm) * pStorm;
        this.uniforms.uFogFactor.value = s.startFog + (this.targetFogDensity - s.startFog) * pFog;
        this.uniforms.uWetness.value = s.startWet + (this.targetWetness - s.startWet) * pWet;

        if (this.scene.fog) {
          this.scene.fog.density = 0.0;
        }

        if (p >= 1.0) {
          s.active = false;
          s.fromMode = this.weatherMode;
        }
      }
    } else {
      // Settled physical maintenance & post-rain evaporation hysteresis
      const lerpRate = 1.0 - Math.exp(-safeDt * 1.8);
      this.uniforms.uRainIntensity.value +=
        (this.targetRainIntensity - this.uniforms.uRainIntensity.value) * lerpRate;
      if (this.targetRainIntensity === 0.0 && this.uniforms.uRainIntensity.value < 0.008) {
        this.uniforms.uRainIntensity.value = 0.0;
      }
      this.uniforms.uCloudCover.value +=
        (this.targetCloudCover - this.uniforms.uCloudCover.value) * lerpRate;
      this.uniforms.uStormFactor.value +=
        (this.targetStormFactor - this.uniforms.uStormFactor.value) * lerpRate;
      this.uniforms.uFogFactor.value +=
        (this.targetFogDensity - this.uniforms.uFogFactor.value) * lerpRate;
      this.baseWindSpeed += (this.targetWindSpeed - this.baseWindSpeed) * lerpRate;
      if (this.scene.fog) {
        this.scene.fog.density = 0.0;
      }

      const rainNow = this.uniforms.uRainIntensity.value;
      if (rainNow > 0.03) {
        const targetWet = Math.min(1.0, 0.35 + rainNow * 0.72);
        const wetRate = 1.0 - Math.exp(-safeDt * (0.45 + rainNow * 0.65));
        this.uniforms.uWetness.value += (targetWet - this.uniforms.uWetness.value) * wetRate;
      } else if (this.targetWetness > 0.01) {
        const wetHoldRate = 1.0 - Math.exp(-safeDt * 0.85);
        this.uniforms.uWetness.value += (this.targetWetness - this.uniforms.uWetness.value) * wetHoldRate;
      } else if (this.uniforms.uWetness.value > 0.0) {
        const dryPerSec = 1.0 / Math.max(20.0, this.transitionDurationSec * 0.85);
        this.uniforms.uWetness.value = Math.max(0.0, this.uniforms.uWetness.value - safeDt * dryPerSec);
      }
    }

    // 2. Continuous Proportional 45s Slider OR 20–45s Preset Diurnal Solar/Lunar/Sidereal Arc Transition
    if (this._timeTrans.active) {
      const sT = this._timeTrans;
      sT.elapsed += transDt;
      sT.launchClock += transDt;
      const durT = Math.max(0.015, sT.duration);
      const pT = Math.min(1.0, sT.elapsed / durT);
      sT.progress = pT;
      const easeT = sT.isProportionalSlider
        ? WeatherOpticsEngine._evalProportionalEase(sT.elapsed, durT, sT.launchRampSec, 0.35)
        : WeatherOpticsEngine._smoothstep(0.0, 1.0, pT);

      this.continuousSolarHour = sT.startContinuousHour + sT.deltaHour * easeT;
      if (sT.isProportionalSlider) {
        this._sliderSolarHour = sT.startSliderHour + sT.deltaHour * easeT;
        const normH =
          this._sliderSolarHour >= 24.0
            ? 23.998
            : ((this._sliderSolarHour % 24.0) + 24.0) % 24.0;
        this.currentSolarHour = normH;
      } else {
        this.currentSolarHour = ((this.continuousSolarHour % 24.0) + 24.0) % 24.0;
        this._sliderSolarHour = this.currentSolarHour;
      }

      if (pT >= 1.0) {
        sT.active = false;
        sT.isProportionalSlider = false;
        sT.fromTod = this.timeOfDay;
      }
    }

    // 3. Turbulent 3D Wind Vector, Monotonically Integrated Wind Phase, & Proportional 45s / Auto Ocean Swell Dynamics
    const gustOsc =
      0.90 + 0.18 * Math.sin(t * 0.34) + 0.11 * Math.cos(t * 0.78) + 0.05 * Math.sin(t * 1.52);
    const windSpeed = this.baseWindSpeed * gustOsc;
    const windDirAngle = 0.48 + 0.14 * Math.sin(t * 0.19);
    this.currentWindAngleRad = windDirAngle;
    this.uniforms.uWindVec.value.set(
      Math.cos(windDirAngle) * windSpeed,
      0.0,
      Math.sin(windDirAngle) * windSpeed
    );
    // Strict monotonic time integral Phi(t) = \int v_wind(tau) d_tau (ZERO backward phase reversal!)
    this.uniforms.uWindPhase.value.x += this.uniforms.uWindVec.value.x * safeDt;
    this.uniforms.uWindPhase.value.y += this.uniforms.uWindVec.value.z * safeDt;

    if (this._seaTrans.active) {
      const sSea = this._seaTrans;
      sSea.elapsed += transDt;
      sSea.launchClock += transDt;
      const durS = Math.max(0.015, sSea.duration);
      const pS = Math.min(1.0, sSea.elapsed / durS);
      sSea.progress = pS;
      const easeS = WeatherOpticsEngine._evalProportionalEase(
        sSea.elapsed,
        durS,
        sSea.launchRampSec,
        0.35
      );
      const liveSeaTarget =
        this.manualSeaWaveHeightM !== null ? sSea.targetWaveM : this._computeAutoSeaWaveHeightM();
      sSea.targetWaveM = liveSeaTarget;
      this.seaWaveHeightM = sSea.startWaveM + (liveSeaTarget - sSea.startWaveM) * easeS;
      if (pS >= 1.0) {
        this.seaWaveHeightM = liveSeaTarget;
        sSea.active = false;
      }
    } else if (this.manualSeaWaveHeightM !== null) {
      this.seaWaveHeightM = this.manualSeaWaveHeightM;
    } else if (!this._weatherTrans.isSpectrumSweep) {
      const autoWaveM = this._computeAutoSeaWaveHeightM();
      const seaRate = 1.0 - Math.exp(-safeDt * 0.55);
      this.seaWaveHeightM += (autoWaveM - this.seaWaveHeightM) * seaRate;
    }
    this._syncOceanSpectrumUniforms();

    this.windGustMultiplier = Math.max(0.75, windSpeed / 0.85);

    // 4. Thunderstorm Multi-Stroke Plasma Lightning (only fires once storm maturity uStormFactor > 0.62!)
    if (this.uniforms.uStormFactor.value > 0.62) {
      this.nextLightningTimer -= safeDt;
      if (this.nextLightningTimer <= 0.0) {
        this._triggerLightningDischarge(t);
        this.nextLightningTimer = 2.6 + (0.5 + 0.5 * Math.sin(t * 17.3)) * 4.4;
      }
    }

    let flash = 0.0;
    if (this.lightningClock >= 0.0) {
      this.lightningClock += safeDt;
      flash = this._evaluateLightningEnvelope(this.lightningClock);
      if (this.lightningClock > 0.54) {
        this.lightningClock = -1.0;
        flash = 0.0;
      }
    }
    this.uniforms.uLightningFlash.value = flash;

    // Evaluate continuous 24h Depok solar/lunar light, twilight colors, fog, & PMREM IBL
    this._applyContinuousCelestialAndFog(sharedSurfaceUniforms, safeDt * 0.40);

    if (flash > 0.005) {
      // v3.6 Non-Teleporting Lightning Flash: shadow-casting sunLight stays strictly on _baseSunPos
      // (ZERO shadow-map jump!), while non-shadow-casting fillLight + hemiLight carry directional storm flash!
      const lDir = this.uniforms.uLightningDir.value;
      const wF = Math.min(1.0, flash * 1.45);
      this.fillLight.position.set(
        this._baseFillPos.x * (1.0 - wF) + lDir.x * 115.0 * wF,
        this._baseFillPos.y * (1.0 - wF) + 76.0 * wF,
        this._baseFillPos.z * (1.0 - wF) + lDir.z * 115.0 * wF
      );
      this._tmpSunColor.setHex(0x9ec5eb).lerp(this._flashColor, wF);
      this.fillLight.color.copy(this._tmpSunColor);
      this.fillLight.intensity = this._baseFillIntensity + flash * 3.45;

      this._tmpSunColor.copy(this._baseSunColor).lerp(this._flashColor, Math.min(1.0, flash * 0.72));
      this.sunLight.color.copy(this._tmpSunColor);
      this.sunLight.intensity = this._baseSunIntensity;
      this.hemiLight.intensity = this._baseHemiIntensity + flash * 1.45;
      this.uniforms.uSunColor.value.copy(this.sunLight.color);
      if (sharedSurfaceUniforms && sharedSurfaceUniforms.uSunColor) {
        sharedSurfaceUniforms.uSunColor.value.copy(this.sunLight.color);
      }
    }

    // 5. Evaluate Dynamic Biological Pupil & Retinal Photoreceptor HDR Adaptation across All Weathers & Times
    this._syncHumanEyeHDR(safeDt, camera, sharedSurfaceUniforms, false);

    // 6. Sync Remaining Shared Surface Uniforms for Ultra Physical Shaders
    if (sharedSurfaceUniforms) {
      if (sharedSurfaceUniforms.uWetness) {
        sharedSurfaceUniforms.uWetness.value = this.uniforms.uWetness.value;
      }
      if (sharedSurfaceUniforms.uRainIntensity) {
        sharedSurfaceUniforms.uRainIntensity.value = this.uniforms.uRainIntensity.value;
      }
      if (sharedSurfaceUniforms.uFogFactor) {
        sharedSurfaceUniforms.uFogFactor.value = this.uniforms.uFogFactor.value;
      }
      if (sharedSurfaceUniforms.uLightningFlash) {
        sharedSurfaceUniforms.uLightningFlash.value = flash;
      }
      if (sharedSurfaceUniforms.uWindVec) {
        sharedSurfaceUniforms.uWindVec.value.copy(this.uniforms.uWindVec.value);
      }
      if (sharedSurfaceUniforms.uCloudShadowPhase) {
        sharedSurfaceUniforms.uCloudShadowPhase.value.copy(this.uniforms.uWindPhase.value);
      }
    }

    this._syncParticleVisibility();

    if (camera) {
      this.uniforms.uCamPos.value.copy(camera.position);
      if (this.skyMesh) {
        this.skyMesh.position.copy(camera.position);
      }
    }
  }
}
