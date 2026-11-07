/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v6.17 — Iterasi 3: Dynamic World Event SFX, Fleet/NPC/Drone SFX & Sub-Mix)
 * ./core/AudioSynthesizer.js
 *
 * 100% Procedural Web Audio API 3D Spatial & Physical Sound Engine for Kepulauan UI Depok:
 * - Real-time 3D AudioListener pose & orthonormal orientation coupled to Three.js camera
 * - Binaural HRTF PannerNode pool + Directional Radiation Cones (Exhaust, PA Array, Transit)
 * - Temperature & Humidity-Coupled Speed of Sound / ISO 9613-1 Air Absorption + 3D Obstacle Occlusion
 * - Dual-Convolver Reverb (2.35s Concrete Vault IR + 0.68s Metallic Truss/Cabin Early-Reflection IR)
 * - Roof-Material-Specific Rain Acoustics (Terracotta Clay, Polycarbonate Truss, Cabin Windshield, Canopy)
 * - 16 Location-Context Micro-Biomes + 6-Phase Diurnal Nusantara Bio-Acoustics
 * - Real-Time State-Transition World SFX (Bus Door/Spray/Suspension, Golf Courtesy Bell, Ferry Marine Horn,
 *   Runway Tire Chirp & Thrust Reverser, eVTOL Skid Thud, NPC Camera Shutter & Umbrella Pop, Avata Acro/APAS)
 * - 3-Mode Acoustic Sub-Mix Profiles ('BALANCED' | 'NATURE' | 'TRANSIT') & 3D HRTF Thunder (delta_t = d / c)
 */

const SPEED_OF_SOUND_MS = 343.0;

function clamp(val, min, max) {
  return val < min ? min : val > max ? max : val;
}

function smooth01(edge0, edge1, x) {
  const span = edge1 - edge0;
  if (Math.abs(span) < 1e-6) return x >= edge1 ? 1.0 : 0.0;
  const t = clamp((x - edge0) / span, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

export class AudioSynthesizer {
  static SPEED_OF_SOUND = SPEED_OF_SOUND_MS;

  static MIX_PROFILES = {
    BALANCED: { id: 'BALANCED', labelId: 'Seimbang', natureMult: 1.0, transitMult: 1.0, sfxMult: 1.0 },
    NATURE: { id: 'NATURE', labelId: 'Alam & Ruang', natureMult: 1.38, transitMult: 0.44, sfxMult: 0.86 },
    TRANSIT: { id: 'TRANSIT', labelId: 'Aksi & Armada', natureMult: 0.66, transitMult: 1.32, sfxMult: 1.24 }
  };

  /**
   * Computes exact temperature-coupled speed of sound in m/s: c(T) = 331.3 * sqrt(1 + T / 273.15).
   */
  static computeSpeedOfSound(tempC = 20.0) {
    const t = clamp(Number.isFinite(tempC) ? tempC : 20.0, -10.0, 48.0);
    return 331.3 * Math.sqrt(1.0 + t / 273.15);
  }

  /**
   * Computes 2-tap specular topographic & monumental early-reflection echoes off:
   *  1) Gunung Api Pulau I volcanic flank (-32, 42, 698)
   *  2) Menara Gedung Rektorat UI concrete tower (0, 28, 0)
   */
  static computeMultiPathEchoes(srcX, srcY, srcZ, lisX, lisY, lisZ, tempC = 27.5) {
    const cSound = AudioSynthesizer.computeSpeedOfSound(tempC);
    const dDirect = Math.max(1.0, Math.hypot(lisX - srcX, lisY - srcY, lisZ - srcZ));
    const reflectors = [
      { id: 'GUNUNG_API_PULAU_I', rx: -32.0, ry: 42.0, rz: 698.0, reflCoeff: 0.42, baseCutoff: 420.0 },
      { id: 'MENARA_REKTORAT_UI', rx: 0.0, ry: 28.0, rz: 0.0, reflCoeff: 0.36, baseCutoff: 560.0 }
    ];
    const echoes = [];
    for (let i = 0; i < reflectors.length; i++) {
      const ref = reflectors[i];
      const d1 = Math.hypot(ref.rx - srcX, ref.ry - srcY, ref.rz - srcZ);
      const d2 = Math.hypot(lisX - ref.rx, lisY - ref.ry, lisZ - ref.rz);
      if (d2 > 1150.0) continue;
      const extraPathM = Math.max(28.0, d1 + d2 - dDirect);
      const extraDelaySec = clamp(extraPathM / cSound, 0.09, 3.6);
      const spreadAtten = clamp(320.0 / Math.max(120.0, d2), 0.14, 0.88);
      echoes.push({
        id: ref.id,
        rx: ref.rx,
        ry: ref.ry,
        rz: ref.rz,
        extraDelaySec,
        gainMult: Number((ref.reflCoeff * spreadAtten).toFixed(4)),
        cutoffHz: Math.round(clamp(ref.baseCutoff * spreadAtten + 140.0, 180.0, 680.0))
      });
    }
    return echoes;
  }

  /**
   * Computes exact acoustic Doppler frequency ratio & ISO 9613-1 temperature/humidity-coupled
   * distance air-absorption lowpass cutoff.
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
    lisVz,
    tempC = 27.5,
    humidityPct = 76.0
  ) {
    const dx = lisX - srcX;
    const dy = lisY - srcY;
    const dz = lisZ - srcZ;
    const dist = Math.hypot(dx, dy, dz);
    const invD = 1.0 / Math.max(0.65, dist);
    const ux = dx * invD;
    const uy = dy * invD;
    const uz = dz * invD;

    const cSound = AudioSynthesizer.computeSpeedOfSound(tempC);

    // Radial component of source velocity toward listener (+ = approaching)
    const vS = clamp(srcVx * ux + srcVy * uy + srcVz * uz, -118.0, 118.0);
    // Radial component of listener velocity toward source (+ = approaching)
    const vL = clamp(-(lisVx * ux + lisVy * uy + lisVz * uz), -118.0, 118.0);

    const rawDoppler = (cSound + vL) / Math.max(120.0, cSound - vS);
    const doppler = clamp(rawDoppler, 0.64, 1.54);

    // ISO 9613-1 molecular O2/N2 relaxation attenuation coupled to temperature & relative humidity
    const rh = clamp(Number.isFinite(humidityPct) ? humidityPct : 76.0, 25.0, 100.0);
    const tempFactor = 1.0 + (clamp(tempC, 15.0, 38.0) - 27.5) * 0.008;
    const rhRelaxFactor = 0.88 + 0.32 * (Math.abs(rh - 74.0) / 50.0);
    const charDistM = 46.0 / (tempFactor * rhRelaxFactor);
    const airCutoffHz = clamp(16800.0 / (1.0 + dist / charDistM), 320.0, 16800.0);

    return { dist, doppler, airCutoffHz, radialApproachMs: vS + vL, speedOfSoundMs: cSound };
  }

  /**
   * Computes physical ground/water-surface specular reflection interference (Lloyd's Mirror Effect):
   * Path difference delta_R = sqrt(d_xz^2 + (h_s + h_L)^2) - sqrt(d_xz^2 + (h_s - h_L)^2),
   * first destructive comb notch f_0 = c(T) / (2 * delta_R), and surface-dependent notch depth.
   */
  static computeGroundReflectionEffect(
    srcX,
    srcY,
    srcZ,
    lisX,
    lisY,
    lisZ,
    groundY = 0.0,
    surfaceRefl = 0.82,
    tempC = 27.5,
    out = null
  ) {
    const dx = lisX - srcX;
    const dz = lisZ - srcZ;
    const dXZ2 = dx * dx + dz * dz;
    const hS = Math.max(0.25, srcY - groundY);
    const hL = Math.max(0.25, lisY - groundY);
    const rDirect = Math.sqrt(dXZ2 + (hS - hL) * (hS - hL));
    const rReflected = Math.sqrt(dXZ2 + (hS + hL) * (hS + hL));
    const deltaPathM = Math.max(0.018, rReflected - rDirect);
    const cSound = AudioSynthesizer.computeSpeedOfSound(tempC);
    const rawNotchHz = cSound / (2.0 * deltaPathM);
    const combNotchHz = clamp(rawNotchHz, 48.0, 9200.0);

    const refl = clamp(Number.isFinite(surfaceRefl) ? surfaceRefl : 0.82, 0.0, 0.98);
    const spreadRatio = rDirect / Math.max(0.5, rReflected);
    const bandWeight = smooth01(42.0, 95.0, rawNotchHz) * smooth01(7600.0, 3800.0, rawNotchHz);
    const combGainDb = -clamp(spreadRatio * refl * bandWeight * 9.2, 0.0, 8.8);

    const res = out || {};
    res.deltaPathM = deltaPathM;
    res.combNotchHz = combNotchHz;
    res.combGainDb = combGainDb;
    res.surfaceRefl = refl;
    return res;
  }

  /**
   * Computes ISO 226 Fletcher-Munson dynamic equal-loudness shelf compensation so low-volume
   * listening preserves warm sub-bass body (95 Hz) and crisp Nusantara air sparkle (6.8 kHz).
   */
  static computeEqualLoudnessCompensation(volume01 = 0.80) {
    const v = clamp(Number.isFinite(volume01) ? volume01 : 0.80, 0.0, 1.0);
    const quietness = clamp((0.82 - Math.max(0.05, v)) / 0.77, 0.0, 1.0);
    return {
      lowShelfGainDb: +(quietness * 5.8).toFixed(3),
      highShelfGainDb: +(quietness * 3.6).toFixed(3),
      quietness: +quietness.toFixed(3)
    };
  }

  /**
   * Computes 3D line-of-sight geometric acoustic occlusion & diffraction attenuation across
   * major Kepulauan UI structures and terrain masses:
   * - Subsea Underpass Bore rock/seawater isolation (`x = -356, z = 322..628, y < -1.85`)
   * - Gunung Api Pulau I volcanic cone (`(-32, 0..60, 698)`)
   * - Menara Gedung Rektorat UI brutalist concrete tower (`(0, 0..44, 0)`)
   * - Kompleks Bundar DPPD–STP (`(-255, 0..22, 329)`)
   */
  static computeGeometricOcclusion(srcX, srcY, srcZ, lisX, lisY, lisZ, out = null) {
    let gainMult = 1.0;
    let cutoffMult = 1.0;
    let occluder = 'NONE';

    const srcInSubsea =
      Math.abs(srcX - -356.0) <= 6.2 && srcZ >= 322.0 && srcZ <= 628.0 && srcY < -1.85;
    const lisInSubsea =
      Math.abs(lisX - -356.0) <= 6.2 && lisZ >= 322.0 && lisZ <= 628.0 && lisY < -1.85;

    // 1. Subsea Tunnel Solid Seabed / Portal Diffraction Isolation
    if (srcInSubsea !== lisInSubsea) {
      const inZ = srcInSubsea ? srcZ : lisZ;
      const outX = srcInSubsea ? lisX : srcX;
      const outY = srcInSubsea ? lisY : srcY;
      const outZ = srcInSubsea ? lisZ : srcZ;
      const outDistS = Math.hypot(
        outX - -356.0,
        Math.max(0.0, 270.0 - outZ) + Math.max(0.0, outZ - 323.0)
      );
      const outDistN = Math.hypot(
        outX - -356.0,
        Math.max(0.0, 627.0 - outZ) + Math.max(0.0, outZ - 675.0)
      );
      const leakS = smooth01(42.0, 1.5, Math.abs(inZ - 322.0)) * smooth01(34.0, 4.0, outDistS);
      const leakN = smooth01(42.0, 1.5, Math.abs(inZ - 628.0)) * smooth01(34.0, 4.0, outDistN);
      const portalLeak = Math.max(leakS, leakN) * (outY < 12.0 ? 1.0 : 0.45);
      gainMult *= 0.06 + 0.66 * portalLeak;
      cutoffMult *= 0.13 + 0.58 * portalLeak;
      occluder = 'SUBSEA_ROCK';
    }

    const segDx = lisX - srcX;
    const segDy = lisY - srcY;
    const segDz = lisZ - srcZ;
    const segLenXZ2 = segDx * segDx + segDz * segDz;

    if (segLenXZ2 > 144.0) {
      // Helper: closest parameter t in [0, 1] in XZ to obstacle center (cx, cz)
      // 2. Gunung Api Pulau I Basalt Cone (-32, 698, peak ~58m, base R ~64m)
      const tVol = (( -32.0 - srcX) * segDx + (698.0 - srcZ) * segDz) / segLenXZ2;
      if (tVol > 0.08 && tVol < 0.92) {
        const px = srcX + segDx * tVol;
        const py = srcY + segDy * tVol;
        const pz = srcZ + segDz * tVol;
        const rV = Math.hypot(px - -32.0, pz - 698.0);
        if (rV < 60.0) {
          const hCone = 58.0 * Math.max(0.0, 1.0 - Math.pow(rV / 65.0, 1.38));
          if (py < hCone - 2.2) {
            const depth01 = clamp((hCone - py) / 26.0, 0.0, 1.0);
            gainMult *= 1.0 - 0.72 * depth01;
            cutoffMult *= 1.0 - 0.75 * depth01;
            if (occluder === 'NONE') occluder = 'GUNUNG_API';
          }
        }
      }

      // 3. Menara Gedung Rektorat UI Concrete Tower (0, 0, half-width ~15m, height ~44m)
      const srcOutsideRekt = Math.max(Math.abs(srcX), Math.abs(srcZ)) > 16.5;
      const lisOutsideRekt = Math.max(Math.abs(lisX), Math.abs(lisZ)) > 16.5;
      if (srcOutsideRekt && lisOutsideRekt) {
        const tRek = ((0.0 - srcX) * segDx + (0.0 - srcZ) * segDz) / segLenXZ2;
        if (tRek > 0.08 && tRek < 0.92) {
          const px = srcX + segDx * tRek;
          const py = srcY + segDy * tRek;
          const pz = srcZ + segDz * tRek;
          if (Math.max(Math.abs(px), Math.abs(pz)) < 14.6 && py < 42.0) {
            const boxDepth = clamp((42.0 - py) / 18.0, 0.25, 1.0);
            gainMult *= 1.0 - 0.56 * boxDepth;
            cutoffMult *= 1.0 - 0.60 * boxDepth;
            if (occluder === 'NONE') occluder = 'GEDUNG_REKTORAT';
          }
        }
      }

      // 4. Kompleks Bundar DPPD–STP (-255, 329, R ~30m, height ~21m)
      const srcOutsideDppd = Math.hypot(srcX - -255.0, srcZ - 329.0) > 32.0;
      const lisOutsideDppd = Math.hypot(lisX - -255.0, lisZ - 329.0) > 32.0;
      if (srcOutsideDppd && lisOutsideDppd) {
        const tDp = ((-255.0 - srcX) * segDx + (329.0 - srcZ) * segDz) / segLenXZ2;
        if (tDp > 0.08 && tDp < 0.92) {
          const px = srcX + segDx * tDp;
          const py = srcY + segDy * tDp;
          const pz = srcZ + segDz * tDp;
          if (Math.hypot(px - -255.0, pz - 329.0) < 28.5 && py < 20.5) {
            gainMult *= 0.52;
            cutoffMult *= 0.46;
            if (occluder === 'NONE') occluder = 'KOMPLEKS_DPPD';
          }
        }
      }

      // 5. Bukit Hutan Selatan Pulau U Terrain Ridge (-135, -118, R ~56m, crest ~15.8m)
      const srcOutsideHill = Math.hypot(srcX - -135.0, srcZ - -118.0) > 54.0;
      const lisOutsideHill = Math.hypot(lisX - -135.0, lisZ - -118.0) > 54.0;
      if (srcOutsideHill && lisOutsideHill) {
        const tHill = ((-135.0 - srcX) * segDx + (-118.0 - srcZ) * segDz) / segLenXZ2;
        if (tHill > 0.08 && tHill < 0.92) {
          const px = srcX + segDx * tHill;
          const py = srcY + segDy * tHill;
          const pz = srcZ + segDz * tHill;
          const rH = Math.hypot(px - -135.0, pz - -118.0);
          if (rH < 52.0) {
            const hRidge = 15.8 * Math.max(0.0, 1.0 - Math.pow(rH / 56.0, 1.6));
            if (py < hRidge - 1.6) {
              const depth01 = clamp((hRidge - py) / 11.0, 0.2, 1.0);
              gainMult *= 1.0 - 0.54 * depth01;
              cutoffMult *= 1.0 - 0.58 * depth01;
              if (occluder === 'NONE') occluder = 'BUKIT_PULAU_U';
            }
          }
        }
      }
    }

    // 6. Jembatan Teksas Solid Steel Box-Girder Deck & Arch Soffit Occlusion (x in [-30.8, -25.2], z in [171.65, 319.65])
    if (Math.abs(segDx) > 0.5 || Math.abs(segDy) > 1.2) {
      const belowDeckSrc = srcY < 4.35;
      const belowDeckLis = lisY < 4.35;
      const aboveDeckSrc = srcY > 6.65;
      const aboveDeckLis = lisY > 6.65;
      if ((belowDeckSrc && aboveDeckLis) || (aboveDeckSrc && belowDeckLis)) {
        const tDeck = clamp((5.5 - srcY) / (segDy || 1e-4), 0.0, 1.0);
        const ix = srcX + segDx * tDeck;
        const iz = srcZ + segDz * tDeck;
        if (ix >= -31.0 && ix <= -25.0 && iz >= 171.5 && iz <= 319.8) {
          gainMult *= 0.56;
          cutoffMult *= 0.48;
          if (occluder === 'NONE') occluder = 'JEMBATAN_TEKSAS';
        }
      } else if (Math.abs(segDx) > 6.0 && (srcX - -28.0) * (lisX - -28.0) < 0.0) {
        const tSpan = (-28.0 - srcX) / segDx;
        if (tSpan > 0.08 && tSpan < 0.92) {
          const py = srcY + segDy * tSpan;
          const pz = srcZ + segDz * tSpan;
          if (pz >= 172.0 && pz <= 319.5 && py >= 4.4 && py <= 7.6) {
            gainMult *= 0.62;
            cutoffMult *= 0.54;
            if (occluder === 'NONE') occluder = 'JEMBATAN_TEKSAS';
          }
        }
      }
    }

    const res = out || {};
    res.occluded = gainMult < 0.94;
    res.gainMult = clamp(gainMult, 0.04, 1.0);
    res.cutoffMult = clamp(cutoffMult, 0.10, 1.0);
    res.occluder = occluder;
    return res;
  }

  /**
   * Classifies the listener's 3D architectural & environmental acoustic space, Dual-IR reverb split
   * (`cathedralSend` vs `metallicSend`), and overhead roof rain-impact material (`roofMaterial`).
   */
  static classifyAcousticZone(x, y, z, camMode = 'DRONE_6DOF', isRoofSheltered = false, fpsSubjectKind = '') {
    const inUnderpassCorridorX = Math.abs(x - -356.0) <= 5.45;
    const inUnderpassBoreZ = z >= 322.0 && z <= 628.0;
    const inUnderpassRampZ = (z >= 270.0 && z < 322.0) || (z > 628.0 && z <= 672.0);
    const inTeksasSpan =
      Math.abs(x - -28.0) <= 2.85 && z >= 168.0 && z <= 323.0 && y >= 3.4 && y <= 12.25;

    const isYachtOpenHelm = camMode === 'FPS_PILOT' && fpsSubjectKind === 'yacht';
    const isCockpitClosed = camMode === 'FPS_BUS_DRIVER' || (camMode === 'FPS_PILOT' && !isYachtOpenHelm);
    const isGolfCanopy = camMode === 'FPS_GOLF_DRIVER' || isYachtOpenHelm;

    if (inUnderpassCorridorX && inUnderpassBoreZ && y < -1.85) {
      const rev = isCockpitClosed ? 0.48 : 0.66;
      return {
        zoneId: 'SUBSEA_TUNNEL',
        labelId: isCockpitClosed ? 'Kabin dalam Terowongan Laut' : 'Terowongan Bawah Laut',
        reverbSend: rev,
        cathedralSend: rev * 0.88,
        metallicSend: rev * (isCockpitClosed ? 0.34 : 0.16),
        reverbCharacter: 'CONCRETE_VAULT_IR',
        roofMaterial: 'CONCRETE_VAULT',
        weatherOcclusionGain: 0.07,
        weatherLowpassHz: 220.0,
        tunnelHum: 1.0,
        patterBoost: 0.0
      };
    }

    if (inTeksasSpan) {
      return {
        zoneId: 'TEKSAS_TRUSS',
        labelId: 'Lorong Jembatan Teksas',
        reverbSend: 0.40,
        cathedralSend: 0.11,
        metallicSend: 0.36,
        reverbCharacter: 'STEEL_TRUSS_IR',
        roofMaterial: 'POLYCARBONATE_TRUSS',
        weatherOcclusionGain: 0.54,
        weatherLowpassHz: 1420.0,
        tunnelHum: 0.0,
        patterBoost: 1.36
      };
    }

    if (isCockpitClosed) {
      const isPressurizedJet = camMode === 'FPS_PILOT' && fpsSubjectKind === 'learjet';
      const isFerryBridge = camMode === 'FPS_PILOT' && fpsSubjectKind === 'ferry';
      const isEvtol = camMode === 'FPS_PILOT' && (fpsSubjectKind === 'evtol' || fpsSubjectKind === 'heli');
      return {
        zoneId: 'COCKPIT_INTERIOR',
        labelId: isPressurizedJet
          ? 'Kokpit Bertekanan Learjet PK-UIB'
          : isFerryBridge
            ? 'Anjungan Kedap Feri Katamaran'
            : isEvtol
              ? 'Kokpit Kedap Heli eVTOL'
              : camMode === 'FPS_PILOT'
                ? 'Kokpit Kedap Pesawat/eVTOL'
                : 'Kabin Bis Kuning UI',
        reverbSend: isPressurizedJet ? 0.18 : 0.21,
        cathedralSend: 0.04,
        metallicSend: 0.20,
        reverbCharacter: 'CABIN_EARLY_IR',
        roofMaterial: 'CABIN_WINDSHIELD',
        weatherOcclusionGain: isPressurizedJet ? 0.16 : isFerryBridge ? 0.23 : isEvtol ? 0.22 : 0.25,
        weatherLowpassHz: isPressurizedJet ? 390.0 : isFerryBridge ? 480.0 : isEvtol ? 495.0 : 510.0,
        tunnelHum: 0.0,
        patterBoost: isPressurizedJet ? 0.42 : 0.58
      };
    }

    if (isGolfCanopy) {
      return {
        zoneId: 'GOLF_CANOPY',
        labelId: isYachtOpenHelm ? 'Kanopi Anjungan Mini Yacht' : 'Kanopi Mobil Golf UI',
        reverbSend: 0.17,
        cathedralSend: 0.05,
        metallicSend: 0.15,
        reverbCharacter: 'CANOPY_EARLY_IR',
        roofMaterial: 'CANOPY_FABRIC',
        weatherOcclusionGain: 0.68,
        weatherLowpassHz: isYachtOpenHelm ? 2150.0 : 1850.0,
        tunnelHum: 0.0,
        patterBoost: 1.20
      };
    }

    if (inUnderpassCorridorX && inUnderpassRampZ && y < 0.65) {
      return {
        zoneId: 'UNDERPASS_RAMP',
        labelId: 'Ramp Beton Underpass',
        reverbSend: 0.36,
        cathedralSend: 0.31,
        metallicSend: 0.10,
        reverbCharacter: 'CONCRETE_RAMP_IR',
        roofMaterial: 'OPEN_FOLIAGE_STONE',
        weatherOcclusionGain: 0.78,
        weatherLowpassHz: 2800.0,
        tunnelHum: 0.35,
        patterBoost: 0.94
      };
    }

    if (isRoofSheltered && y < 8.8) {
      return {
        zoneId: 'ROOF_SHELTERED',
        labelId: 'Kolonade / Selasar Beratap',
        reverbSend: 0.30,
        cathedralSend: 0.24,
        metallicSend: 0.11,
        reverbCharacter: 'STONE_COLONNADE_IR',
        roofMaterial: 'TERRACOTTA_CLAY',
        weatherOcclusionGain: 0.44,
        weatherLowpassHz: 1120.0,
        tunnelHum: 0.0,
        patterBoost: 1.28
      };
    }

    if (y > 40.0) {
      return {
        zoneId: 'HIGH_ALTITUDE',
        labelId: 'Udara Terbuka Tinggi',
        reverbSend: 0.04,
        cathedralSend: 0.03,
        metallicSend: 0.01,
        reverbCharacter: 'OPEN_SKY_IR',
        roofMaterial: 'OPEN_FOLIAGE_STONE',
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
      cathedralSend: 0.09,
      metallicSend: 0.04,
      reverbCharacter: 'OPEN_COURTYARD_IR',
      roofMaterial: 'OPEN_FOLIAGE_STONE',
      weatherOcclusionGain: 1.0,
      weatherLowpassHz: 13500.0,
      tunnelHum: 0.0,
      patterBoost: 1.0
    };
  }

  /**
   * Classifies the 6 Diurnal Nusantara Bio-Acoustic Phases aligned 1:1 with the 6 Solar Presets:
   * - DAWN (04:45–06:30): Fajar (Kokok Ayam Fajar + Perkutut + Morning Dew)
   * - MORNING (06:30–10:30): Pagi (Lively Kutilang & Tekukur + Campus Awakening)
   * - DAY (10:30–14:30): Siang (Vibrant Academic Courtyard Murmur + Songbirds)
   * - AFTERNOON (14:30–17:00): Sore (Forest Tonggeret Cicada Chorus + Warm Breeze)
   * - DUSK (17:00–18:45): Senja (Golden-Hour Tonggeret + Early Dusk Crickets)
   * - NIGHT (18:45–04:45): Malam (Stereo Crickets + Lakeside Frogs + Nocturnal Owl/Tokek)
   */
  static classifyDiurnalBioPhase(solarHour = 12.0) {
    const h = (((Number(solarHour) || 0.0) % 24.0) + 24.0) % 24.0;
    if (h >= 4.75 && h < 6.50) {
      return {
        phaseId: 'DAWN',
        labelId: 'Fajar Nusantara',
        birdWeight: 0.90,
        tonggeretWeight: 0.06,
        cricketWeight: 0.22,
        socialWeight: 0.16
      };
    }
    if (h >= 6.50 && h < 10.50) {
      return {
        phaseId: 'MORNING',
        labelId: 'Pagi Kampus',
        birdWeight: 1.0,
        tonggeretWeight: 0.10,
        cricketWeight: 0.0,
        socialWeight: 0.82
      };
    }
    if (h >= 10.50 && h < 14.50) {
      return {
        phaseId: 'DAY',
        labelId: 'Siang Akademik',
        birdWeight: 0.68,
        tonggeretWeight: 0.30,
        cricketWeight: 0.0,
        socialWeight: 1.0
      };
    }
    if (h >= 14.50 && h < 17.00) {
      return {
        phaseId: 'AFTERNOON',
        labelId: 'Sore Teduh',
        birdWeight: 0.72,
        tonggeretWeight: 0.94,
        cricketWeight: 0.08,
        socialWeight: 0.88
      };
    }
    if (h >= 17.00 && h < 18.75) {
      return {
        phaseId: 'DUSK',
        labelId: 'Senja Keemasan',
        birdWeight: 0.36,
        tonggeretWeight: 1.0,
        cricketWeight: 0.56,
        socialWeight: 0.54
      };
    }
    return {
      phaseId: 'NIGHT',
      labelId: 'Malam Tropis',
      birdWeight: 0.0,
      tonggeretWeight: 0.0,
      cricketWeight: 1.0,
      socialWeight: 0.10
    };
  }

  /**
   * Resolves the listener's exact Kepulauan UI Location-Context Micro-Biome (16 Distinct Contexts)
   * + smooth continuous 3D spatial blend weights for click-free environmental ambience transitions.
   */
  static classifyMicroContextBiome(x, y, z, opts = {}) {
    const coastDist = Number.isFinite(opts.coastDist) ? opts.coastDist : -40.0;
    const riverZ = opts.riverZ !== undefined ? opts.riverZ : null;

    // Continuous 3D proximity weights (used for seamless audio crossfading across zone borders)
    const dFlag = Math.hypot(x - -2.5, z - 52.0);
    const dVideotron = Math.hypot(x - 28.4, z - 78.2);
    const dRotunda = Math.hypot(x - -2.5, z - 106.5);
    const dSports = Math.hypot(x - -62.0, z - 136.5);
    const dDppd = Math.hypot(x - -255.0, z - 329.0);
    const dDermagaU = Math.hypot(x - -48.0, z - 480.0);
    const dDermagaI = Math.hypot(x - -98.0, z - 595.0);
    const dCraterXZ = Math.hypot(x - -32.0, z - 698.0);
    const dNiagara = Math.hypot(x - -44.0, z - 643.0);
    const dKiosk = Math.min(
      Math.hypot(x - -39.5, z - 134.5),
      Math.hypot(x - -16.5, z - 345.0),
      Math.hypot(x - -213.0, z - 326.0),
      Math.hypot(x - -116.5, z - 619.5)
    );
    const dSpring = Math.hypot(x - -265.0, z - -95.0);
    const dRektorat = Math.max(Math.abs(x), Math.abs(z));

    const altGroundFade = clamp(1.0 - Math.max(0.0, y - 10.0) / 36.0, 0.0, 1.0);
    const flagWeight = smooth01(25.0, 6.0, dFlag) * altGroundFade;
    const fumaroleWeight =
      smooth01(48.0, 12.0, dCraterXZ) * smooth01(18.0, 38.0, y) * smooth01(95.0, 68.0, y);
    const trussSpanWeight =
      smooth01(9.0, 2.5, Math.abs(x - -28.0)) *
      smooth01(158.0, 174.0, z) *
      smooth01(332.0, 316.0, z) *
      smooth01(24.0, 12.0, y);

    const nearDanauKenanga = x >= 20.0 && x <= 106.0 && Math.abs(z - 10.0) <= 74.0;
    const nearDanauKecil = Math.hypot(x - -78.0, z - -22.0) <= 40.0;
    const nearTurapCoast = Math.abs(coastDist) <= 18.0 && y > -1.8;
    const turapLapWeight =
      (nearDanauKenanga || nearDanauKecil
        ? 0.92
        : nearTurapCoast
          ? smooth01(18.0, 3.0, Math.abs(coastDist))
          : 0.0) * altGroundFade;

    const socialProx = Math.max(
      smooth01(22.0, 5.0, dKiosk) * 1.0,
      smooth01(32.0, 9.0, dRotunda) * 0.78,
      smooth01(38.0, 12.0, dRektorat) * 0.72,
      smooth01(40.0, 14.0, dDppd) * 0.75,
      smooth01(22.0, 7.0, dVideotron) * 0.65
    );
    const socialWeight = socialProx * altGroundFade;

    // 1. Subsea Underpass Bore
    if (Math.abs(x - -356.0) <= 6.2 && z >= 322.0 && z <= 628.0 && y < -1.85) {
      return {
        biomeId: 'SUBSEA_UNDERPASS_BORE',
        labelId: 'Terowongan Bawah Laut UI',
        canopyWeight: 0.0,
        socialWeight: 0.0,
        flagWeight: 0.0,
        fumaroleWeight: 0.0,
        turapLapWeight: 0.0,
        trussAeolianWeight: 0.0
      };
    }

    // 2. Jembatan Teksas Span
    if (Math.abs(x - -28.0) <= 5.5 && z >= 165.0 && z <= 325.0 && y < 18.5) {
      return {
        biomeId: 'TEKSAS_BRIDGE_SPAN',
        labelId: 'Bentang Jembatan Teksas',
        canopyWeight: 0.08,
        socialWeight: 0.22 * altGroundFade,
        flagWeight: 0.0,
        fumaroleWeight: 0.0,
        turapLapWeight: 0.45 * altGroundFade,
        trussAeolianWeight: Math.max(0.65, trussSpanWeight)
      };
    }

    // 3. Kawah Gunung Api Pulau I
    if (dCraterXZ <= 42.0 && y >= 24.0 && y <= 86.0) {
      return {
        biomeId: 'VOLCANO_CRATER_RIM',
        labelId: 'Kawah Gunung Api Pulau I',
        canopyWeight: 0.05,
        socialWeight: 0.0,
        flagWeight: 0.0,
        fumaroleWeight: Math.max(0.72, fumaroleWeight),
        turapLapWeight: 0.0,
        trussAeolianWeight: 0.0
      };
    }

    // 4. Tebing Air Terjun Niagara Pulau I
    if (dNiagara <= 42.0 && y < 38.0) {
      return {
        biomeId: 'NIAGARA_WATERFALL_CLIFF',
        labelId: 'Tebing Air Terjun Niagara',
        canopyWeight: 0.55 * altGroundFade,
        socialWeight: 0.0,
        flagWeight: 0.0,
        fumaroleWeight,
        turapLapWeight: 0.55 * altGroundFade,
        trussAeolianWeight: 0.0
      };
    }

    // 5. Bandara & Menara ATC Pulau I
    if (x >= -390.0 && x <= -120.0 && z >= 648.0 && z <= 732.0 && y < 38.0) {
      return {
        biomeId: 'AIRPORT_ATC_APRON',
        labelId: 'Apron & Menara ATC Bandara',
        canopyWeight: 0.14 * altGroundFade,
        socialWeight: 0.18 * altGroundFade,
        flagWeight: 0.0,
        fumaroleWeight: 0.0,
        turapLapWeight: 0.12 * altGroundFade,
        trussAeolianWeight: 0.18
      };
    }

    // 6. Kios & Taman Bangku Kampus UI (checked before Dermaga/Rotunda so kiosk corners resolve cleanly)
    if (dKiosk <= 16.5 && y < 10.5) {
      return {
        biomeId: 'CAMPUS_KIOSK_BENCH',
        labelId: 'Kios & Taman Bangku UI',
        canopyWeight: 0.42 * altGroundFade,
        socialWeight: Math.max(0.85, socialWeight),
        flagWeight,
        fumaroleWeight: 0.0,
        turapLapWeight,
        trussAeolianWeight: 0.0
      };
    }

    // 7. Dermaga Feri Selat UI (Pulau U & Pulau I Piers)
    if ((dDermagaU <= 32.0 || dDermagaI <= 32.0) && y < 18.0) {
      return {
        biomeId: 'DERMAGA_HARBOUR',
        labelId: 'Dermaga Feri Selat UI',
        canopyWeight: 0.12 * altGroundFade,
        socialWeight: 0.36 * altGroundFade,
        flagWeight: 0.0,
        fumaroleWeight: 0.0,
        turapLapWeight: 0.88 * altGroundFade,
        trussAeolianWeight: 0.15
      };
    }

    // 8. Lapangan 15 Bendera Fakultas
    if (dFlag <= 18.5 && y < 18.5) {
      return {
        biomeId: 'FLAG_CLOISTER',
        labelId: 'Lapangan 15 Bendera UI',
        canopyWeight: 0.28 * altGroundFade,
        socialWeight: Math.max(0.32, socialWeight),
        flagWeight: Math.max(0.85, flagWeight),
        fumaroleWeight: 0.0,
        turapLapWeight: 0.0,
        trussAeolianWeight: 0.0
      };
    }

    // 9. Plaza Videotron UI
    if (dVideotron <= 20.0 && y < 18.5) {
      return {
        biomeId: 'VIDEOTRON_PLAZA',
        labelId: 'Plaza Videotron UI',
        canopyWeight: 0.30 * altGroundFade,
        socialWeight: Math.max(0.55, socialWeight),
        flagWeight,
        fumaroleWeight: 0.0,
        turapLapWeight,
        trussAeolianWeight: 0.0
      };
    }

    // 10. Plaza Air Mancur Rotunda
    if (dRotunda <= 28.0 && y < 22.0) {
      return {
        biomeId: 'ROTUNDA_FOUNTAIN_PLAZA',
        labelId: 'Plaza Air Mancur Rotunda',
        canopyWeight: 0.26 * altGroundFade,
        socialWeight: Math.max(0.68, socialWeight),
        flagWeight,
        fumaroleWeight: 0.0,
        turapLapWeight: 0.35 * altGroundFade,
        trussAeolianWeight: 0.0
      };
    }

    // 11. Lintasan Lari & Kalistenik Teluk U
    if (dSports <= 24.0 && y < 14.0) {
      return {
        biomeId: 'SPORTS_TRACK_TELUK',
        labelId: 'Lintasan Lari & Kalistenik',
        canopyWeight: 0.45 * altGroundFade,
        socialWeight: 0.44 * altGroundFade,
        flagWeight: 0.0,
        fumaroleWeight: 0.0,
        turapLapWeight: Math.max(0.35, turapLapWeight),
        trussAeolianWeight: 0.0
      };
    }

    // 12. Pelataran Gedung Rektorat UI
    if (dRektorat <= 36.0 && y < 28.0) {
      return {
        biomeId: 'REKTORAT_COURTYARD',
        labelId: 'Pelataran Gedung Rektorat',
        canopyWeight: 0.24 * altGroundFade,
        socialWeight: Math.max(0.62, socialWeight),
        flagWeight,
        fumaroleWeight: 0.0,
        turapLapWeight: 0.0,
        trussAeolianWeight: 0.0
      };
    }

    // 13. Kompleks Bundar DPPD–STP
    if (dDppd <= 38.0 && y < 26.0) {
      return {
        biomeId: 'DPPD_STP_ACADEMIC',
        labelId: 'Kompleks Bundar DPPD–STP',
        canopyWeight: 0.38 * altGroundFade,
        socialWeight: Math.max(0.66, socialWeight),
        flagWeight: 0.0,
        fumaroleWeight: 0.0,
        turapLapWeight: 0.0,
        trussAeolianWeight: 0.0
      };
    }

    // 14. Mata Air & Sungai Berbatu Pulau U
    const onPulauURiver =
      riverZ !== null &&
      Number.isFinite(riverZ) &&
      x >= -295.0 &&
      x <= -98.0 &&
      z < 420.0 &&
      Math.abs(z - riverZ) <= 22.0;
    if ((dSpring <= 35.0 || onPulauURiver) && y < 20.0) {
      return {
        biomeId: 'RIVER_SPRING_RIFFLE',
        labelId: 'Mata Air & Sungai Pulau U',
        canopyWeight: 0.78 * altGroundFade,
        socialWeight: 0.0,
        flagWeight: 0.0,
        fumaroleWeight: 0.0,
        turapLapWeight: 0.48 * altGroundFade,
        trussAeolianWeight: 0.0
      };
    }

    // 15. Tepian Danau Kenanga, Danau Kecil, & Turap Batu Pantai
    if ((nearDanauKenanga || nearDanauKecil || nearTurapCoast) && y < 16.0) {
      return {
        biomeId: 'DANAU_TURAP_PROMENADE',
        labelId: 'Tepian Danau & Turap Batu',
        canopyWeight: 0.48 * altGroundFade,
        socialWeight: 0.25 * altGroundFade,
        flagWeight: 0.0,
        fumaroleWeight: 0.0,
        turapLapWeight: Math.max(0.75, turapLapWeight),
        trussAeolianWeight: 0.0
      };
    }

    // 16. Hutan Kota Tropis UI (South Pulau U Rainforest & Pulau I Green Hills)
    const overLand = coastDist <= 4.0;
    return {
      biomeId: 'TROPICAL_RAINFOREST',
      labelId: 'Hutan Kota Tropis UI',
      canopyWeight: (overLand ? 0.92 : 0.15) * altGroundFade,
      socialWeight,
      flagWeight,
      fumaroleWeight,
      turapLapWeight,
      trussAeolianWeight: trussSpanWeight
    };
  }

  // ===========================================================================
  // DOLBY ATMOS 7.1.4 POINT-OF-VIEW SPATIAL SOUNDFIELD PHYSICS (v633, Zero GC)
  // ===========================================================================

  /**
   * Canonical 12-Channel ITU-R BS.2051 / Dolby Atmos 7.1.4 Hemispherical Layout:
   * - 7 Ear-Level Bed Channels: L (-30°), R (+30°), C (0°), Ls (-90°), Rs (+90°), Lrs (-145°), Rrs (+145°)
   * - 1 LFE (.1) Sub-Bass Channel: 24–110 Hz
   * - 4 Overhead Height Dome (.4) Channels: Ltf (-45°, +45°), Rtf (+45°, +45°), Ltr (-135°, +45°), Rtr (+135°, +45°)
   */
  static get ATMOS_SPEAKER_LAYOUT_7_1_4() {
    if (!this._cachedAtmos714Layout) {
      const deg = Math.PI / 180.0;
      const mk = (id, azDeg, elDeg, layer) => {
        const az = azDeg * deg;
        const el = elDeg * deg;
        const ce = Math.cos(el);
        return {
          id,
          azDeg,
          elDeg,
          layer,
          ux: Math.sin(az) * ce,
          uy: Math.sin(el),
          uz: Math.cos(az) * ce
        };
      };
      this._cachedAtmos714Layout = [
        mk('L', -30, 0, 'EAR'),
        mk('R', 30, 0, 'EAR'),
        mk('C', 0, 0, 'EAR'),
        mk('LFE', 0, -35, 'LFE'),
        mk('Ls', -90, 0, 'EAR'),
        mk('Rs', 90, 0, 'EAR'),
        mk('Lrs', -145, 0, 'EAR'),
        mk('Rrs', 145, 0, 'EAR'),
        mk('Ltf', -45, 45, 'HEIGHT'),
        mk('Rtf', 45, 45, 'HEIGHT'),
        mk('Ltr', -135, 45, 'HEIGHT'),
        mk('Rtr', 135, 45, 'HEIGHT')
      ];
    }
    return this._cachedAtmos714Layout;
  }

  /**
   * Projects a 3D world-space coordinate `(srcX, srcY, srcZ)` into the active view's exact
   * orthonormal 3D listener head frame (`Right`, `Up`, `Front`) -> `(azDeg, elDeg, distM)`.
   * - `azDeg`: `0°` dead ahead, `+90°` right ear, `-90°` left ear, `±180°` directly behind.
   * - `elDeg`: `+90°` overhead zenith (.4 Height Dome), `0°` ear horizon, `-90°` nadir below feet/drone.
   */
  static projectWorldToListenerSpherical(srcX, srcY, srcZ, lisPose = {}, out = null) {
    const res = out || {};
    const lx = Number.isFinite(lisPose.x) ? lisPose.x : 0.0;
    const ly = Number.isFinite(lisPose.y) ? lisPose.y : 1.7;
    const lz = Number.isFinite(lisPose.z) ? lisPose.z : 0.0;

    let fx = Number.isFinite(lisPose.fwdX) ? lisPose.fwdX : 0.0;
    let fy = Number.isFinite(lisPose.fwdY) ? lisPose.fwdY : 0.0;
    let fz = Number.isFinite(lisPose.fwdZ) ? lisPose.fwdZ : -1.0;
    const fLen = Math.hypot(fx, fy, fz) || 1.0;
    fx /= fLen;
    fy /= fLen;
    fz /= fLen;

    const u0x = Number.isFinite(lisPose.upX) ? lisPose.upX : 0.0;
    const u0y = Number.isFinite(lisPose.upY) ? lisPose.upY : 1.0;
    const u0z = Number.isFinite(lisPose.upZ) ? lisPose.upZ : 0.0;

    // Right = normalize(Forward × Up)
    let rx = fy * u0z - fz * u0y;
    let ry = fz * u0x - fx * u0z;
    let rz = fx * u0y - fy * u0x;
    const rLen = Math.hypot(rx, ry, rz);
    if (rLen > 1e-5) {
      rx /= rLen;
      ry /= rLen;
      rz /= rLen;
    } else {
      rx = 1.0;
      ry = 0.0;
      rz = 0.0;
    }

    // Orthonormal Up = Right × Forward
    const ux = ry * fz - rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy - ry * fx;

    const dx = srcX - lx;
    const dy = srcY - ly;
    const dz = srcZ - lz;
    const distM = Math.hypot(dx, dy, dz);
    if (distM < 1e-4) {
      res.localRight = 0.0;
      res.localUp = 0.0;
      res.localFront = 1.0;
      res.distM = 0.0;
      res.azDeg = 0.0;
      res.elDeg = 0.0;
      res.frontBackCos = 1.0;
      res.height01 = 0.0;
      res.nadir01 = 0.0;
      return res;
    }

    const invD = 1.0 / distM;
    const localRight = clamp((dx * rx + dy * ry + dz * rz) * invD, -1.0, 1.0);
    const localUp = clamp((dx * ux + dy * uy + dz * uz) * invD, -1.0, 1.0);
    const localFront = clamp((dx * fx + dy * fy + dz * fz) * invD, -1.0, 1.0);
    const horizLen = Math.hypot(localRight, localFront);

    res.localRight = localRight;
    res.localUp = localUp;
    res.localFront = localFront;
    res.distM = distM;
    res.azDeg = horizLen > 1e-4 ? (Math.atan2(localRight, localFront) * 180.0) / Math.PI : 0.0;
    res.elDeg = (Math.asin(localUp) * 180.0) / Math.PI;
    res.frontBackCos = horizLen > 1e-4 ? clamp(localFront / horizLen, -1.0, 1.0) : 1.0;
    res.height01 = clamp(localUp, 0.0, 1.0);
    res.nadir01 = clamp(-localUp, 0.0, 1.0);
    return res;
  }

  /**
   * Computes Dolby Atmos Volumetric Object Size (`spread01`), Anthropometric Pinna Rear-Concha
   * Shadow (`rearShelfDb` @ 4.2 kHz), Blauert Overhead Elevation Presence (`elevationPeakDb` @ 7.8 kHz),
   * and `.4` Overhead Height Dome Send (`heightSendGain`).
   */
  static computeAtmosPinnaAndSpreadCue(
    azDeg = 0.0,
    elDeg = 0.0,
    distM = 12.0,
    sourceRadiusM = 2.0,
    spatialMode = 'BINAURAL',
    out = null
  ) {
    const res = out || {};
    const R = Math.max(0.2, Number(sourceRadiusM) || 1.5);
    const D = Math.max(0.35, Number(distM) || 1.0);
    const spread01 = clamp((2.0 / Math.PI) * Math.atan(Math.pow(R / D, 1.32)), 0.0, 0.92);

    const absAz = Math.abs(azDeg);
    const elRad = (elDeg * Math.PI) / 180.0;
    const sinEl = Math.sin(elRad);
    const rearFactor = smooth01(82.0, 174.0, absAz) * (1.0 - 0.45 * Math.abs(sinEl));
    const isSpeaker = spatialMode === 'SPEAKER';

    const rearShelfDb = (isSpeaker ? -1.8 : -3.6) * rearFactor;
    const elevationPeakDb = sinEl >= 0.0 ? sinEl * (isSpeaker ? 2.2 : 3.8) : sinEl * 1.8;
    const heightSendGain = smooth01(14.0, 72.0, elDeg) * (0.18 + 0.34 * (1.0 - spread01 * 0.35));

    const spreadEnergy = 0.52 * spread01;
    res.spread01 = spread01;
    res.directGainMult = Math.sqrt(Math.max(0.18, 1.0 - spreadEnergy));
    res.spreadSendGain = Math.sqrt(spreadEnergy) * 0.78;
    res.rearShelfDb = rearShelfDb;
    res.elevationPeakDb = elevationPeakDb;
    res.heightSendGain = heightSendGain;
    return res;
  }

  /**
   * Computes true world-space 3D coordinates for every environmental spatial bed around `(camX, camY, camZ)`
   * so rotating the camera in ANY view counter-rotates coastline surf, lakeside lap, upwind breeze,
   * overhead forest canopy, flags, truss arch, and volcano fumarole with zero head-locked drift.
   */
  static computeWorldEnvironmentAnchors(camX, camY, camZ, state = {}, out = null) {
    const res = out || {
      sea: { x: 0, y: 0, z: 0 },
      lap: { x: 0, y: 0, z: 0 },
      wind: { x: 0, y: 0, z: 0 },
      canopy: { x: 0, y: 0, z: 0 },
      tonggeret: { x: 0, y: 0, z: 0 },
      cricket: { x: 0, y: 0, z: 0 },
      flag: { x: 0, y: 0, z: 0 },
      truss: { x: 0, y: 0, z: 0 },
      fumarole: { x: 0, y: 0, z: 0 },
      social: { x: 0, y: 0, z: 0 },
      skyFront: { x: 0, y: 0, z: 0 },
      skyRear: { x: 0, y: 0, z: 0 }
    };

    // 1. True Coastline / Ocean Surf Vector
    let sdx = Number(state.seaDirX);
    let sdz = Number(state.seaDirZ);
    if (!Number.isFinite(sdx) || !Number.isFinite(sdz) || Math.hypot(sdx, sdz) < 1e-3) {
      // Physical fallback toward Selat UI (z ≈ 505), Teluk U, or outer archipelago sea
      if (camZ > 360.0 && camZ < 630.0) {
        sdx = 0.0;
        sdz = camZ < 505.0 ? 1.0 : -1.0;
      } else if (camZ >= 120.0 && camZ <= 360.0 && camX > -95.0) {
        sdx = 1.0;
        sdz = 0.25;
      } else {
        const cxIsland = camZ > 515.0 ? -160.0 : -85.0;
        const czIsland = camZ > 515.0 ? 690.0 : 120.0;
        sdx = camX - cxIsland;
        sdz = camZ - czIsland;
      }
    }
    const sLen = Math.hypot(sdx, sdz) || 1.0;
    sdx /= sLen;
    sdz /= sLen;

    const coastDist = Number.isFinite(state.coastDist) ? state.coastDist : -40.0;
    const dSurf = clamp(coastDist < 0.0 ? Math.max(10.0, -coastDist) : 14.0, 10.0, 68.0);
    res.sea.x = camX + sdx * dSurf;
    res.sea.y = -1.38;
    res.sea.z = camZ + sdz * dSurf;

    // 2. Nearest Freshwater Lake / River / Stone Turap Lapping Coordinate
    const dKenanga = Math.hypot(Math.max(0.0, 32.0 - camX), Math.max(0.0, Math.abs(camZ) - 72.0));
    const dKecil = Math.hypot(camX - -78.0, camZ - -22.0);
    const riverZ = state.riverZ;
    const dRiver =
      Number.isFinite(riverZ) && camX >= -295.0 && camX <= -98.0 && camZ < 420.0
        ? Math.abs(camZ - riverZ)
        : 999.0;
    if (dRiver < 28.0 && dRiver <= dKenanga && dRiver <= dKecil) {
      res.lap.x = clamp(camX, -288.0, -104.0);
      res.lap.y = 0.65;
      res.lap.z = riverZ;
    } else if (dKenanga < 38.0 && dKenanga <= dKecil) {
      res.lap.x = Math.max(camX + 6.0, 34.0);
      res.lap.y = 0.12;
      res.lap.z = clamp(camZ, -68.0, 68.0);
    } else if (dKecil < 44.0) {
      res.lap.x = -78.0;
      res.lap.y = 0.15;
      res.lap.z = -22.0;
    } else {
      res.lap.x = res.sea.x;
      res.lap.y = -1.05;
      res.lap.z = res.sea.z;
    }

    // 3. Upwind Campus Breeze Origin
    let wdx = Number.isFinite(state.windDirX) ? state.windDirX : 0.78;
    let wdz = Number.isFinite(state.windDirZ) ? state.windDirZ : 0.62;
    const wLen = Math.hypot(wdx, wdz) || 1.0;
    wdx /= wLen;
    wdz /= wLen;
    res.wind.x = camX - wdx * 18.0;
    res.wind.y = camY + 2.4;
    res.wind.z = camZ - wdz * 18.0;

    // 4. Overhead Rainforest Tree-Crown Canopy & Antiphonal Wildlife Anchors
    const groundY = Number.isFinite(state.groundY) ? state.groundY : Math.min(camY, 0.2);
    const crownY = Math.max(groundY + 10.8, 8.8);
    res.canopy.x = camX + wdx * 5.2;
    res.canopy.y = crownY;
    res.canopy.z = camZ + wdz * 5.2;

    res.tonggeret.x = camX - 6.8;
    res.tonggeret.y = crownY - 1.1;
    res.tonggeret.z = camZ + 7.4;

    res.cricket.x = camX + 6.2;
    res.cricket.y = groundY + 0.32;
    res.cricket.z = camZ - 5.4;

    // 5. Landmark Micro-Biome World Emitters
    res.flag.x = -2.5;
    res.flag.y = 5.4;
    res.flag.z = 52.0;

    res.truss.x = -28.0;
    res.truss.y = clamp(camY + 3.4, 6.8, 13.8);
    res.truss.z = clamp(camZ, 152.0, 340.0);

    res.fumarole.x = -32.0;
    res.fumarole.y = 52.0;
    res.fumarole.z = 698.0;

    // Nearest Campus Social Plaza Center
    const dRotunda = Math.hypot(camX - -2.5, camZ - 106.5);
    const dRektorat = Math.hypot(camX - 0.0, camZ - 26.0);
    const dDppd = Math.hypot(camX - -255.0, camZ - 329.0);
    if (dDppd < dRotunda && dDppd < dRektorat) {
      res.social.x = -255.0;
      res.social.y = groundY + 1.5;
      res.social.z = 329.0;
    } else if (dRektorat < dRotunda) {
      res.social.x = 0.0;
      res.social.y = groundY + 1.5;
      res.social.z = 26.0;
    } else {
      res.social.x = -2.5;
      res.social.y = groundY + 1.5;
      res.social.z = 106.5;
    }

    // 6. Overhead .4 Atmos Height Dome Ceiling Anchors
    const fx = Number.isFinite(state.fwdX) ? state.fwdX : 0.0;
    const fz = Number.isFinite(state.fwdZ) ? state.fwdZ : -1.0;
    const fhLen = Math.hypot(fx, fz);
    const hfx = fhLen > 1e-3 ? fx / fhLen : 0.0;
    const hfz = fhLen > 1e-3 ? fz / fhLen : -1.0;
    res.skyFront.x = camX + hfx * 5.6;
    res.skyFront.y = camY + 8.6;
    res.skyFront.z = camZ + hfz * 5.6;
    res.skyRear.x = camX - hfx * 5.6;
    res.skyRear.y = camY + 8.6;
    res.skyRear.z = camZ - hfz * 5.6;

    return res;
  }

  /**
   * Resolves the exact point-of-view acoustic simulation profile for whichever camera view (`camMode`)
   * is currently active, identifying co-moving vehicles/aircraft (to eliminate artificial self-Doppler
   * and position engines at true cockpit-relative 3D coordinates) and ear-height perspective rules.
   */
  static resolveViewPerspectiveAcoustics(camMode, camX, camY, camZ, sivitasAI = null, state = {}, out = null) {
    const res = out || {};
    res.coMovingEntity = null;
    res.isPurePointMic = false;
    res.earHeightClass = 'HUMAN_EAR';

    if (camMode === 'ORBIT') {
      res.perspectiveId = 'ORBIT_POINT_MIC';
      res.perspectiveLabel = 'Orbit 3D (Titik Murni)';
      res.isPurePointMic = true;
      res.earHeightClass = camY > 14.0 ? 'AERIAL' : 'HUMAN_EAR';
      return res;
    }
    if (camMode === 'DRONE_AVATA2_FPV') {
      res.perspectiveId = 'AVATA_CINEWHOOP';
      res.perspectiveLabel = 'Cinewhoop Avata 2 FPV';
      res.earHeightClass = 'AERIAL';
      return res;
    }
    if (camMode === 'DRONE_6DOF' || camMode === 'AUTO_CINEMATIC') {
      res.perspectiveId = 'MAVIC_AERIAL';
      res.perspectiveLabel = 'Udara Mavic 3 Pro';
      res.earHeightClass = 'AERIAL';
      return res;
    }
    if (camMode === 'WALK_SIVITAS') {
      res.perspectiveId = 'HUMAN_WALK_EAR';
      res.perspectiveLabel = 'Telinga Jalan Sivitas';
      res.earHeightClass = 'HUMAN_EAR';
      return res;
    }
    if (camMode === 'FPS_JOGGER') {
      res.perspectiveId = 'JOGGER_TRACK_EAR';
      res.perspectiveLabel = 'Pelari Lintasan UI';
      res.earHeightClass = 'HUMAN_EAR';
      return res;
    }
    if (camMode === 'FPS_CHAT_NPC') {
      res.perspectiveId = 'SOCIAL_CIRCLE_EAR';
      res.perspectiveLabel = 'Dialog Lingkar Sivitas';
      res.earHeightClass = 'HUMAN_EAR';
      return res;
    }
    if (camMode === 'FPS_CAT') {
      res.perspectiveId = 'FELINE_GROUND_EAR';
      res.perspectiveLabel = 'Telinga Kucing (0.28m)';
      res.earHeightClass = 'GROUND_FELINE';
      return res;
    }
    if (camMode === 'FPS_BUS_DRIVER') {
      res.perspectiveId = 'BUS_CABIN_DRIVER';
      res.perspectiveLabel = 'Kokpit Sopir Bis UI';
      res.coMovingEntity = 'bus';
      res.earHeightClass = 'VEHICLE_CABIN';
      return res;
    }
    if (camMode === 'FPS_GOLF_DRIVER') {
      res.perspectiveId = 'GOLF_OPEN_SHUTTLE';
      res.perspectiveLabel = 'Kemudi Shuttle Golf';
      res.coMovingEntity = 'cart';
      res.earHeightClass = 'VEHICLE_OPEN';
      return res;
    }
    if (camMode === 'FPS_PILOT') {
      let kind = state.fpsSubjectKind || '';
      if (kind === 'heli') kind = 'evtol';
      const validKind =
        kind === 'cessna' || kind === 'learjet' || kind === 'evtol' || kind === 'ferry' || kind === 'yacht';
      if (!validKind && sivitasAI) {
        const cs = sivitasAI.cessnaAircraft;
        const lj = sivitasAI.learjetAircraft;
        const ev = sivitasAI.heliTaxi;
        const fe = sivitasAI.shuttleFerry;
        const ya = sivitasAI.miniYacht;
        const dCs = cs ? Math.hypot(camX - cs.worldX, camY - cs.worldY, camZ - cs.worldZ) : 9999.0;
        const dLj = lj ? Math.hypot(camX - lj.worldX, camY - lj.worldY, camZ - lj.worldZ) : 9999.0;
        const dEv = ev ? Math.hypot(camX - ev.worldX, camY - ev.worldY, camZ - ev.worldZ) : 9999.0;
        const dFe = fe ? Math.hypot(camX - fe.worldX, camY - fe.worldY, camZ - fe.worldZ) : 9999.0;
        const dYa = ya ? Math.hypot(camX - ya.worldX, camY - ya.worldY, camZ - ya.worldZ) : 9999.0;
        let minD = dCs;
        kind = 'cessna';
        if (dLj < minD && dLj < 25.0) {
          minD = dLj;
          kind = 'learjet';
        }
        if (dEv < minD && dEv < 25.0) {
          minD = dEv;
          kind = 'evtol';
        }
        if (dFe < minD && dFe < 25.0) {
          minD = dFe;
          kind = 'ferry';
        }
        if (dYa < minD && dYa < 25.0) {
          minD = dYa;
          kind = 'yacht';
        }
      }
      if (kind === 'learjet') {
        res.perspectiveId = 'PILOT_COCKPIT_LEARJET';
        res.perspectiveLabel = 'Kokpit Learjet PK-UIB';
        res.coMovingEntity = 'learjet';
        res.earHeightClass = 'AIRCRAFT_COCKPIT';
      } else if (kind === 'evtol') {
        res.perspectiveId = 'PILOT_COCKPIT_EVTOL';
        res.perspectiveLabel = 'Kokpit Heli eVTOL PK-UIE';
        res.coMovingEntity = 'heli';
        res.earHeightClass = 'AIRCRAFT_COCKPIT';
      } else if (kind === 'ferry') {
        res.perspectiveId = 'CAPTAIN_BRIDGE_FERRY';
        res.perspectiveLabel = 'Anjungan Feri Katamaran UI';
        res.coMovingEntity = 'ferry';
        res.earHeightClass = 'MARINE_BRIDGE';
      } else if (kind === 'yacht') {
        res.perspectiveId = 'CAPTAIN_HELM_YACHT';
        res.perspectiveLabel = 'Anjungan Kemudi Mini Yacht';
        res.coMovingEntity = 'yacht';
        res.earHeightClass = 'MARINE_OPEN';
      } else {
        res.perspectiveId = 'PILOT_COCKPIT_CESSNA';
        res.perspectiveLabel = 'Kokpit Cessna PK-UIA';
        res.coMovingEntity = 'cessna';
        res.earHeightClass = 'AIRCRAFT_COCKPIT';
      }
      return res;
    }

    res.perspectiveId = 'OPEN_POINT_MIC';
    res.perspectiveLabel = 'Titik Kamera 3D';
    return res;
  }

  /**
   * Projects active 3D spatial objects and hemispherical beds onto the 12 Dolby Atmos `7.1.4`
   * speaker channels `[L, R, C, LFE, Ls, Rs, Lrs, Rrs, Ltf, Rtf, Ltr, Rtr]` in `[0..1]`.
   */
  static projectAtmos714ChannelEnergies(activeObjects = [], envSummary = {}, out12 = null) {
    const ch = out12 && out12.length >= 12 ? out12 : new Float32Array(12);
    const layout = AudioSynthesizer.ATMOS_SPEAKER_LAYOUT_7_1_4;

    const baseEar = clamp(Number(envSummary.baseEarBed) || 0.04, 0.0, 0.45);
    const baseHeight = clamp(Number(envSummary.baseHeightBed) || 0.02, 0.0, 0.55);
    const baseLfe = clamp(Number(envSummary.baseLfe) || 0.0, 0.0, 0.75);

    for (let s = 0; s < 12; s++) {
      const layer = layout[s].layer;
      ch[s] = layer === 'LFE' ? baseLfe : layer === 'HEIGHT' ? baseHeight : baseEar;
    }

    const deg = Math.PI / 180.0;
    for (let i = 0; i < activeObjects.length; i++) {
      const obj = activeObjects[i];
      const g = clamp(Number(obj.gain01) || 0.0, 0.0, 1.5);
      if (g <= 0.002) continue;
      const az = (Number(obj.azDeg) || 0.0) * deg;
      const el = (Number(obj.elDeg) || 0.0) * deg;
      const ce = Math.cos(el);
      const sx = Math.sin(az) * ce;
      const sy = Math.sin(el);
      const sz = Math.cos(az) * ce;
      const spread = clamp(Number(obj.spread01) || 0.0, 0.0, 0.95);
      const focusExp = 4.2 - spread * 3.1; // Sharp VBAP lobe (4.2) -> wide enveloping bed (1.1)
      const lfeW = clamp(Number(obj.lfeWeight) || 0.0, 0.0, 1.0);

      ch[3] += g * lfeW * 0.85;
      for (let s = 0; s < 12; s++) {
        if (s === 3) continue;
        const sp = layout[s];
        const dot = sx * sp.ux + sy * sp.uy + sz * sp.uz;
        const dirW = dot > 0.0 ? Math.pow(dot, focusExp) : 0.0;
        const ambW = spread * (sp.layer === 'HEIGHT' ? 0.14 : 0.18);
        ch[s] += g * (dirW * (1.0 - spread * 0.42) + ambW);
      }
    }

    let earSum = 0.0;
    let heightSum = 0.0;
    for (let s = 0; s < 12; s++) {
      ch[s] = clamp(ch[s], 0.0, 1.0);
      if (layout[s].layer === 'EAR') earSum += ch[s];
      else if (layout[s].layer === 'HEIGHT') heightSum += ch[s];
    }

    return {
      channels12: ch,
      earEnergy01: clamp(earSum / 4.2, 0.0, 1.0),
      heightEnergy01: clamp(heightSum / 2.4, 0.0, 1.0),
      lfeEnergy01: ch[3]
    };
  }

  constructor() {
    this.ctx = null;
    this.enabled = false;
    this.masterVolume = 0.80; // 0.0 .. 1.0 user slider

    // Master, ISO 226 Equal-Loudness, Priority Ducking, & Dual-Convolver Space Routing Nodes
    this.masterGain = null;
    this.masterCompressor = null;
    this.subsonicFilter = null;
    this.isoLowShelf = null;
    this.isoHighShelf = null;
    this.ambientDuckGain = null;
    this.worldSumGain = null;
    this.convolverNode = null;
    this.convolverCathedral = null;
    this.convolverMetallic = null;
    this.reverbSendGain = null;
    this.cathedralSendGain = null;
    this.metallicSendGain = null;
    this.reverbReturnGain = null;

    // Binaural Stereo / 3D World-Anchored Bed Panners & Dolby Atmos 7.1.4 Buses
    this.surfPan = null;
    this.canopyPan = null;
    this.cricketPan = null;
    this.tonggeretPan = null;
    this._bedPanners = null;
    this.lfeSumGain = null;
    this.lfeMasterGain = null;
    this.atmosSpreadIn = null;
    this.heightDomeIn = null;
    this.heightFrontPanner = null;
    this.heightRearPanner = null;
    this.heightRevGain = null;
    this.cockpitAirGain = null;
    this.cockpitAirFilter = null;
    this.cockpitAvionicsGain = null;
    this.cockpitPanner = null;

    // Weather, Roof-Material Resonance & Environmental Buses
    this.weatherOccFilter = null;
    this.weatherOccGain = null;
    this.breezeGain = null;
    this.breezeFilter = null;
    this.rainRoarGain = null;
    this.rainRoarFilter = null;
    this.rainPatterGain = null;
    this.rainPatterFilter = null;
    this.roofResonanceFilter = null;
    this.roofResonanceGain = null;
    this.surfFilter = null;
    this.surfGain = null;
    this.tunnelHumGain = null;
    this.cricketGain = null;

    // Iterasi 2 Location-Context Micro-Biome Continuous Ambience Buses
    this.canopyRustleFilter = null;
    this.canopyRustleGain = null;
    this.tonggeretGain = null;
    this.socialMurmurGain = null;
    this.flagFlutterFilter = null;
    this.flagFlutterGain = null;
    this.trussAeolianGain = null;
    this.fumaroleGain = null;
    this.turapLapFilter = null;
    this.turapLapGain = null;

    // Dual-Drone Buses (DJI Avata 2 Cinewhoop + Stereo Roll Pan + Duct Resonance + DJI Mavic 3 Pro)
    this.avataMotorOscA = null;
    this.avataMotorOscB = null;
    this.avataMotorOscC = null;
    this.avataMotorFilter = null;
    this.avataMotorGain = null;
    this.avataDuctFilter = null;
    this.avataDuctGain = null;
    this.avataPan = null;
    this.avataWashFilter = null;
    this.avataWashGain = null;
    this.proxWhooshFilter = null;
    this.proxWhooshGain = null;
    this._warmShaperCurve = null;

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
      rollRad: 0.0,
      speedMs: 0.0,
      init: false
    };
    this._weatherState = {
      rain: 0.0,
      storm: 0.0,
      windSpeed: 0.85,
      wetness: 0.0,
      nightFactor: 0.0,
      tempC: 27.5,
      humidityPct: 76.0,
      lightningDirX: 0.6,
      lightningDirY: 0.34,
      lightningDirZ: 0.7
    };
    this._zoneState = AudioSynthesizer.classifyAcousticZone(0, 22, 100, 'DRONE_6DOF', false);
    this._biomeState = AudioSynthesizer.classifyMicroContextBiome(0, 22, 100, {});
    this._diurnalState = AudioSynthesizer.classifyDiurnalBioPhase(11.85);
    this._occScratch = { occluded: false, gainMult: 1.0, cutoffMult: 1.0, occluder: 'NONE' };
    this._combScratch = { deltaPathM: 1.0, combNotchHz: 1200.0, combGainDb: 0.0, surfaceRefl: 0.8 };
    this._thunderDuckTimer = 0.0;
    this._foregroundPressure01 = 0.0;

    this._entityTracks = {
      bus: {
        x: 0,
        y: 0,
        z: 0,
        vx: 0,
        vy: 0,
        vz: 0,
        prevSpd: 0,
        prevPitch: 0,
        wasMoving: true,
        init: false,
        brakeCooldown: 0,
        doorCooldown: 0,
        suspCooldown: 0
      },
      cart: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, prevSpd: 0, prevPhase: '', init: false, bellCooldown: 0 },
      heli: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, prevSpd: 0, prevPhase: '', init: false },
      cessna: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, prevSpd: 0, init: false },
      learjet: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, prevSpd: 0, init: false },
      ferry: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, prevSpd: 0, init: false },
      yacht: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, prevSpd: 0, init: false },
      coastalBoat: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, prevSpd: 0, init: false, idx: -1 }
    };

    // Iterasi 3: Sub-Mix Profile, Spatial Output Mode & Live State-Transition World SFX Tracker
    this.mixProfile = 'BALANCED';
    this._mixWeights = AudioSynthesizer.MIX_PROFILES.BALANCED;
    this.spatialOutputMode = 'BINAURAL'; // 'BINAURAL' | 'SPEAKER'
    this._spectrumBins = new Uint8Array(32);
    this._sfxTracker = {
      cessnaPhase: '',
      learjetPhase: '',
      learjetGearPrev: -1.0,
      learjetGearSfxCooldown: 0.0,
      ferryPhase: '',
      prevRain: 0.0,
      rainInit: false,
      prevAvataStunt: false,
      prevAvataApas: false,
      apasCooldown: 0.0,
      prevMavicFov: 43.0,
      mavicFovInit: false,
      npcShutterTimer: 5.5,
      sfxBannerTimer: 0.0,
      prevBellSlot: null,
      bellCooldown: 0.0,
      eaveDripCooldown: 1.4,
      hullSlapCooldown: 1.8,
      prevZoneId: '',
      portalCooldown: 0.0
    };

    // Stride, Diurnal Bio-Acoustic & Micro-Biome Timers
    this._strideAccumM = 0.0;
    this._stepFootSide = 1;
    this._birdTimer = 1.8;
    this._doveTimer = 4.2;
    this._roosterTimer = 3.5;
    this._nightOwlTimer = 6.5;
    this._frogTimer = 2.6;
    this._catMeowTimer = 5.5;
    this._contextMicroTimer = 2.4;
    this._atcSquelchTimer = 5.2;
    this._jogBreathAccum = 0.0;
    this._whooshCooldown = 0.0;
    this._simClock = 0.0;

    // v633: Zero-GC Dolby Atmos 7.1.4 Point-of-View Spatial Soundfield State & NPC Trackers
    this._npcStepHalfPhase = new Int8Array(96);
    this._npcStepHalfPhase.fill(-1);
    this._npcFootstepCooldown = 0.0;
    this._chatCandidates = [null, null, null, null];
    this._sphScratch = {
      localRight: 0.0,
      localUp: 0.0,
      localFront: 1.0,
      distM: 1.0,
      azDeg: 0.0,
      elDeg: 0.0,
      frontBackCos: 1.0,
      height01: 0.0,
      nadir01: 0.0
    };
    this._cueScratch = {
      spread01: 0.0,
      directGainMult: 1.0,
      spreadSendGain: 0.0,
      rearShelfDb: 0.0,
      elevationPeakDb: 0.0,
      heightSendGain: 0.0
    };
    this._atmosAnchors = AudioSynthesizer.computeWorldEnvironmentAnchors(0, 22, 100, {});
    this._viewPerspective = AudioSynthesizer.resolveViewPerspectiveAcoustics('DRONE_6DOF', 0, 22, 100, null, {});
    this._atmosObjectPool = [];
    for (let i = 0; i < 20; i++) {
      this._atmosObjectPool.push({
        id: '',
        label: '',
        kind: 'OBJECT',
        azDeg: 0.0,
        elDeg: 0.0,
        distM: 0.0,
        gain01: 0.0,
        spread01: 0.0,
        lfeWeight: 0.0
      });
    }
    this._atmosActiveObjects = [];
    this._atmosBedLevels12 = new Float32Array(12);
    this._atmosEnvScratch = { baseEarBed: 0.05, baseHeightBed: 0.03, baseLfe: 0.01 };
    this._atmosStage = {
      layout: '7.1.4',
      spatialMode: 'BINAURAL',
      perspectiveId: 'MAVIC_AERIAL',
      perspectiveLabel: 'Udara Mavic 3 Pro',
      coMovingEntity: null,
      isPurePointMic: false,
      earEnergy01: 0.06,
      heightEnergy01: 0.04,
      lfeEnergy01: 0.02,
      bedLevels12: this._atmosBedLevels12,
      activeObjects: this._atmosActiveObjects,
      objectCount: 0
    };

    // Live Public Telemetry Snapshot
    this._telemetry = {
      enabled: false,
      masterVolume: 0.80,
      masterVolumePct: 80,
      mixProfile: 'BALANCED',
      mixProfileLabel: 'Seimbang',
      spatialOutputMode: 'BINAURAL',
      atmosLayout: '7.1.4',
      perspectiveId: 'MAVIC_AERIAL',
      perspectiveLabel: 'Udara Mavic 3 Pro',
      earEnergyPct: 6,
      heightEnergyPct: 4,
      lfeEnergyPct: 2,
      zoneId: 'OPEN_AIR',
      zoneLabel: 'Ruang Terbuka HRTF',
      zoneLabelId: 'Ruang Terbuka HRTF',
      microBiomeId: 'TROPICAL_RAINFOREST',
      microBiomeLabel: 'Hutan Kota Tropis UI',
      diurnalPhase: 'DAY',
      diurnalLabel: 'Siang Akademik',
      reverbCharacter: 'OPEN_COURTYARD_IR',
      roofMaterial: 'OPEN_FOLIAGE_STONE',
      reverbSendPct: 11,
      weatherOcclusionPct: 0,
      active3DSources: 12,
      occluded3DSources: 0,
      dominantSourceLabel: 'Air Mancur & Alam',
      dominantDoppler: 1.0,
      dominantAzDeg: 0,
      dominantElDeg: 0,
      lastSfxId: 'NONE',
      lastSfxLabel: '',
      activeSfxBanner: '',
      sfxCount: 0,
      listenerSpeedMs: 0.0,
      surfaceMaterial: 'AIR'
    };

    // Zero-GC Sound Engine & Web Audio DSP Forensic Blackbox + Adaptive Sub-Bus Hibernation State (Tahap 2)
    this._preferPlaybackBuffer = false;
    this._hibernationSlots = [];
    this._hibernationSlotByGain = new Map();
    this._forensic = {
      initAttempts: 0,
      initDurationMs: 0.0,
      initBreakdownMs: null,
      initErrors: [],
      resumeAttempts: 0,
      resumeFailures: 0,
      stateTransitionsCount: 0,
      lastStateChange: 'uninitialized',
      unlockedMilestoneSent: false,
      requestedLatencyHint: 'interactive',
      lastSimPaused: false,
      _inGraphBuild: false,
      nodes: {
        persistentTotal: 0,
        oscillators: 0,
        loopBufferSources: 0,
        biquadFilters: 0,
        gainNodes: 0,
        panners3D: 0,
        stereoPanners: 0,
        delayLines: 0,
        convolvers: 0,
        waveShapers: 0,
        compressors: 0,
        analysers: 0,
        channelMergers: 0,
        audioBuffersCount: 0,
        audioBufferBytes: 0,
        transientSourceSpawns: 0,
        liveTransientSources: 0,
        transientAuxCreated: 0,
        transientAuxDisconnected: 0
      },
      perf: {
        updateCalls: 0,
        emaUpdateMs: 0.0,
        maxUpdateMs: 0.0,
        overBudgetFrames: 0,
        skippedRedundantParamOps: 0,
        updateErrorsCount: 0,
        lastUpdateError: ''
      },
      clock: {
        accumSec: 0.0,
        prevPerfMs: 0.0,
        prevCtxSec: -1.0,
        checksCount: 0,
        lastDriftMs: 0.0,
        maxLagMs: 0.0,
        underrunSuspectCount: 0,
        lastUnderrunLagMs: 0.0
      },
      limiter: {
        currentReductionDb: 0.0,
        peakReductionDb: 0.0,
        heavyLimitingEvents: 0
      }
    };
  }

  getAtmosStageSnapshot() {
    this._atmosStage.spatialMode = this.spatialOutputMode;
    return this._atmosStage;
  }

  _pushAtmosObject(id, label, kind, srcX, srcY, srcZ, gain01, radiusM = 2.0, lfeWeight = 0.0) {
    if (gain01 <= 0.004 || this._atmosActiveObjects.length >= this._atmosObjectPool.length) return null;
    const idx = this._atmosActiveObjects.length;
    const slot = this._atmosObjectPool[idx];
    const sph = AudioSynthesizer.projectWorldToListenerSpherical(srcX, srcY, srcZ, this._lis, this._sphScratch);
    const cue = AudioSynthesizer.computeAtmosPinnaAndSpreadCue(
      sph.azDeg,
      sph.elDeg,
      sph.distM,
      radiusM,
      this.spatialOutputMode,
      this._cueScratch
    );
    slot.id = id;
    slot.label = label;
    slot.kind = kind;
    slot.azDeg = +sph.azDeg.toFixed(1);
    slot.elDeg = +sph.elDeg.toFixed(1);
    slot.distM = +sph.distM.toFixed(1);
    slot.gain01 = clamp(gain01, 0.0, 1.0);
    slot.spread01 = +cue.spread01.toFixed(3);
    slot.lfeWeight = clamp(lfeWeight, 0.0, 1.0);
    this._atmosActiveObjects.push(slot);
    return slot;
  }

  setMixProfile(profileId = 'BALANCED') {
    const key = String(profileId || 'BALANCED').toUpperCase();
    const prof = AudioSynthesizer.MIX_PROFILES[key] || AudioSynthesizer.MIX_PROFILES.BALANCED;
    this.mixProfile = prof.id;
    this._mixWeights = prof;
    this._telemetry.mixProfile = prof.id;
    this._telemetry.mixProfileLabel = prof.labelId;
    return prof.id;
  }

  getMixProfile() {
    return this.mixProfile;
  }

  setSpatialOutputMode(mode = 'BINAURAL') {
    const raw = String(mode || 'BINAURAL').toUpperCase();
    const m = raw === 'SPEAKER' || raw === 'STEREO' ? 'SPEAKER' : 'BINAURAL';
    this.spatialOutputMode = m;
    this._telemetry.spatialOutputMode = m;
    this._atmosStage.spatialMode = m;
    const model = m === 'SPEAKER' ? 'equalpower' : 'HRTF';
    if (this._emitters) {
      for (const key of Object.keys(this._emitters)) {
        const em = this._emitters[key];
        if (em && em.panner) {
          em.panner.panningModel = model;
        }
      }
    }
    if (this._bedPanners) {
      for (const key of Object.keys(this._bedPanners)) {
        const bp = this._bedPanners[key];
        if (bp) bp.panningModel = model;
      }
    }
    if (this.heightFrontPanner) this.heightFrontPanner.panningModel = model;
    if (this.heightRearPanner) this.heightRearPanner.panningModel = model;
    if (this.cockpitPanner) this.cockpitPanner.panningModel = model;
    return m;
  }

  getSpatialOutputMode() {
    return this.spatialOutputMode;
  }

  /**
   * Zero-allocation 8-band octave spectrum sampler from master AnalyserNode (0..255 per bar).
   */
  sampleSpectrum8Bars(out8) {
    const target = out8 && out8.length >= 8 ? out8 : new Uint8Array(8);
    if (!this.enabled || !this.ctx || !this.masterAnalyser || this.ctx.state !== 'running') {
      for (let i = 0; i < 8; i++) target[i] = 0;
      return target;
    }
    this.masterAnalyser.getByteFrequencyData(this._spectrumBins);
    const bandSpans = [
      [0, 1],
      [1, 2],
      [2, 4],
      [4, 6],
      [6, 9],
      [9, 13],
      [13, 19],
      [19, 28]
    ];
    for (let b = 0; b < 8; b++) {
      const [i0, i1] = bandSpans[b];
      let sum = 0;
      let cnt = 0;
      for (let k = i0; k < i1 && k < this._spectrumBins.length; k++) {
        sum += this._spectrumBins[k];
        cnt++;
      }
      target[b] = cnt > 0 ? Math.min(255, Math.round((sum / cnt) * 1.18)) : 0;
    }
    return target;
  }

  _recordSfxEvent(sfxId, labelId) {
    this._telemetry.lastSfxId = sfxId;
    this._telemetry.lastSfxLabel = labelId;
    this._telemetry.activeSfxBanner = labelId;
    this._telemetry.sfxCount = (this._telemetry.sfxCount || 0) + 1;
    this._sfxTracker.sfxBannerTimer = 4.2;
  }

  _targetMasterLinearGain() {
    if (!this.enabled) return 0.0;
    // Perceptual curve: 80% slider -> 0.32 linear headroom into safety compressor
    const v = clamp(this.masterVolume, 0.0, 1.0);
    return Math.pow(v, 1.35) * 0.44;
  }

  _applyEqualLoudnessContour(now) {
    const eq = AudioSynthesizer.computeEqualLoudnessCompensation(this.masterVolume);
    this._telemetry.isoLoudnessLowDb = eq.lowShelfGainDb;
    this._telemetry.isoLoudnessHighDb = eq.highShelfGainDb;
    if (this.ctx && this.isoLowShelf && this.isoHighShelf) {
      const t = Number.isFinite(now) ? now : this.ctx.currentTime;
      this.isoLowShelf.gain.setTargetAtTime(eq.lowShelfGainDb, t, 0.08);
      this.isoHighShelf.gain.setTargetAtTime(eq.highShelfGainDb, t, 0.08);
    }
  }

  setMasterVolume(vol01) {
    this.masterVolume = clamp(Number(vol01) || 0.0, 0.0, 1.0);
    this._telemetry.masterVolume = this.masterVolume;
    this._telemetry.masterVolumePct = Math.round(this.masterVolume * 100);
    this._applyEqualLoudnessContour();
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

  _markUnlockedMilestoneIfRunning() {
    const f = this._forensic;
    if (this.ctx && this.ctx.state === 'running' && !f.unlockedMilestoneSent) {
      f.unlockedMilestoneSent = true;
      if (
        typeof window !== 'undefined' &&
        window.NexusForensic &&
        typeof window.NexusForensic.markMilestone === 'function'
      ) {
        window.NexusForensic.markMilestone(
          'web_audio_unlocked',
          `${this.ctx.sampleRate || 0}Hz · ${f.nodes.persistentTotal} DSP nodes · Atmos 7.1.4`
        );
      }
    }
  }

  setHardwareEfficiencyHint(preferPlaybackBuffer = false) {
    this._preferPlaybackBuffer = Boolean(preferPlaybackBuffer);
    if (!this.ctx) {
      this._forensic.requestedLatencyHint = this._preferPlaybackBuffer ? 'playback' : 'interactive';
    }
  }

  _wakeSubBusSlot(slot, nowSec = 0.0) {
    if (!slot || !slot.hibernated) return;
    const gParam = slot.gainNode && slot.gainNode.gain ? slot.gainNode.gain : null;
    if (gParam) {
      if (typeof gParam._origSetValueAtTime === 'function') {
        try {
          gParam._origSetValueAtTime(0.0, nowSec);
        } catch (_e) {}
      } else {
        gParam.value = 0.0;
      }
      gParam._lastTau = -1.0;
    }
    const links = slot.links;
    for (let i = 0; i < links.length; i++) {
      const pair = links[i];
      if (pair[0] && pair[1]) {
        try {
          pair[0].connect(pair[1]);
        } catch (_e) {}
      }
    }
    slot.hibernated = false;
    slot.silentSec = 0.0;
  }

  _hibernateSubBusSlot(slot) {
    if (!slot || slot.hibernated) return;
    const sources = slot.uniqueSources;
    for (let i = 0; i < sources.length; i++) {
      const s = sources[i];
      if (s && typeof s.disconnect === 'function') {
        try {
          s.disconnect();
        } catch (_e) {}
      }
    }
    slot.hibernated = true;
  }

  _registerHibernationSlot(name, kind, gainNode, links) {
    if (!gainNode || !gainNode.gain || !Array.isArray(links) || links.length === 0) return;
    const validLinks = [];
    const uniqueSources = [];
    for (let i = 0; i < links.length; i++) {
      const pair = links[i];
      if (pair && pair[0] && pair[1]) {
        validLinks.push(pair);
        if (uniqueSources.indexOf(pair[0]) === -1) {
          uniqueSources.push(pair[0]);
        }
      }
    }
    if (validLinks.length === 0) return;
    const initVal = Number(gainNode.gain.value) || 0.0;
    const slot = {
      name,
      kind,
      gainNode,
      links: validLinks,
      uniqueSources,
      hibernated: false,
      // Sub-buses initialized at exact 0.0 have zero ringing tail and can hibernate on frame 1 if unused
      silentSec: initVal <= 0.0002 ? 0.65 : 0.0
    };
    this._hibernationSlots.push(slot);
    this._hibernationSlotByGain.set(gainNode.gain, slot);
  }

  _tickSubBusHibernation(safeDt, nowSec) {
    const slots = this._hibernationSlots;
    if (!slots || slots.length === 0) return;
    for (let i = 0; i < slots.length; i++) {
      const slot = slots[i];
      const gp = slot.gainNode.gain;
      const rawVal = gp._lastTgt !== undefined ? gp._lastTgt : gp.value;
      const wantAudible = this.enabled && Number.isFinite(rawVal) && rawVal > 0.00025;
      if (wantAudible) {
        slot.silentSec = 0.0;
        if (slot.hibernated) {
          this._wakeSubBusSlot(slot, nowSec);
        }
      } else {
        slot.silentSec += safeDt;
        if (!slot.hibernated && slot.silentSec >= 0.62) {
          this._hibernateSubBusSlot(slot);
        }
      }
    }
  }

  _cleanupTransientNodes(sourceNode, ...auxNodes) {
    if (!sourceNode) return;
    sourceNode.onended = () => {
      if (typeof sourceNode.disconnect === 'function') {
        try {
          sourceNode.disconnect();
        } catch (_e) {}
      }
      for (let i = 0; i < auxNodes.length; i++) {
        const n = auxNodes[i];
        if (n && typeof n.disconnect === 'function') {
          try {
            n.disconnect();
          } catch (_e) {}
        }
      }
    };
  }

  _installForensicNodeCounters(ctx) {
    if (!ctx || ctx.__nexusSoundForensicWrapped) return;
    ctx.__nexusSoundForensicWrapped = true;
    const f = this._forensic;
    const ns = f.nodes;

    // Markovian AudioParam deduplicator: skips mathematically identical setTargetAtTime(T, t, tau)
    // calls on persistent nodes while waking hibernated sub-buses synchronously at 0.0 gain before ramp-up!
    const wrapPersistentParam = (param, eps = 0.00025, isGainParam = false) => {
      if (!param || param.__nexusForensicWrapped) return;
      param.__nexusForensicWrapped = true;
      param._lastTau = -1.0;
      param._lastStart = -1.0;
      if (typeof param.setValueAtTime === 'function') {
        const origVal = param.setValueAtTime.bind(param);
        param._origSetValueAtTime = origVal;
        param.setValueAtTime = (value, startTime) => {
          if (isGainParam && value > 0.00025 && this.enabled) {
            const slot = this._hibernationSlotByGain.get(param);
            if (slot && slot.hibernated) {
              this._wakeSubBusSlot(slot, Number(startTime) || 0.0);
            }
          }
          param._lastTgt = value;
          param._lastTau = -1.0;
          param._lastStart = Number(startTime) || 0.0;
          return origVal(value, startTime);
        };
      }
      if (typeof param.setTargetAtTime === 'function') {
        const origTgt = param.setTargetAtTime.bind(param);
        param.setTargetAtTime = (target, startTime, timeConstant) => {
          const nowT = Number(startTime) || 0.0;
          const tau = Number(timeConstant) || 0.05;
          if (isGainParam && target > 0.00025 && this.enabled) {
            const slot = this._hibernationSlotByGain.get(param);
            if (slot && slot.hibernated) {
              this._wakeSubBusSlot(slot, nowT);
            }
          }
          const prev = param._lastTgt;
          if (
            prev !== undefined &&
            Math.abs(target - prev) <= eps &&
            param._lastTau > 0.0 &&
            Math.abs(tau - param._lastTau) <= 0.005 &&
            nowT >= param._lastStart
          ) {
            f.perf.skippedRedundantParamOps++;
            return param;
          }
          param._lastTgt = target;
          param._lastTau = tau;
          param._lastStart = nowT;
          return origTgt(target, startTime, timeConstant);
        };
      }
      if (typeof param.cancelScheduledValues === 'function') {
        const origCancel = param.cancelScheduledValues.bind(param);
        param.cancelScheduledValues = (cancelTime) => {
          param._lastTau = -1.0;
          return origCancel(cancelTime);
        };
      }
      if (typeof param.linearRampToValueAtTime === 'function') {
        const origLin = param.linearRampToValueAtTime.bind(param);
        param.linearRampToValueAtTime = (val, endTime) => {
          param._lastTgt = val;
          param._lastTau = -1.0;
          return origLin(val, endTime);
        };
      }
      if (typeof param.exponentialRampToValueAtTime === 'function') {
        const origExp = param.exponentialRampToValueAtTime.bind(param);
        param.exponentialRampToValueAtTime = (val, endTime) => {
          param._lastTgt = val;
          param._lastTau = -1.0;
          return origExp(val, endTime);
        };
      }
    };

    const wrapFactory = (methodName, persistentKey, kind) => {
      if (typeof ctx[methodName] !== 'function') return;
      const orig = ctx[methodName].bind(ctx);
      ctx[methodName] = (...args) => {
        const node = orig(...args);
        if (f._inGraphBuild) {
          ns.persistentTotal++;
          if (persistentKey && ns[persistentKey] !== undefined) ns[persistentKey]++;
          if (node) {
            if (methodName === 'createGain' && node.gain) {
              wrapPersistentParam(node.gain, 0.00022, true);
            } else if (methodName === 'createBiquadFilter') {
              wrapPersistentParam(node.frequency, 0.25, false);
              wrapPersistentParam(node.Q, 0.005, false);
              wrapPersistentParam(node.gain, 0.02, false);
            } else if (methodName === 'createOscillator') {
              wrapPersistentParam(node.frequency, 0.15, false);
            } else if (methodName === 'createStereoPanner' && node.pan) {
              wrapPersistentParam(node.pan, 0.002, false);
            } else if (methodName === 'createPanner') {
              wrapPersistentParam(node.positionX, 0.015, false);
              wrapPersistentParam(node.positionY, 0.015, false);
              wrapPersistentParam(node.positionZ, 0.015, false);
              wrapPersistentParam(node.orientationX, 0.003, false);
              wrapPersistentParam(node.orientationY, 0.003, false);
              wrapPersistentParam(node.orientationZ, 0.003, false);
            }
          }
        } else if (kind === 'source') {
          ns.transientSourceSpawns++;
          ns.liveTransientSources++;
          if (node) {
            let decremented = false;
            const onSourceEnded = () => {
              if (!decremented) {
                decremented = true;
                if (ns.liveTransientSources > 0) ns.liveTransientSources--;
              }
            };
            if (typeof node.addEventListener === 'function') {
              node.addEventListener('ended', onSourceEnded, { once: true });
            } else {
              node._forensicOnEnded = onSourceEnded;
            }
          }
        } else if (kind === 'aux') {
          ns.transientAuxCreated++;
          if (node && typeof node.disconnect === 'function') {
            const origDisc = node.disconnect.bind(node);
            let countedDisc = false;
            node.disconnect = (...dArgs) => {
              if (!countedDisc) {
                countedDisc = true;
                ns.transientAuxDisconnected++;
              }
              return origDisc(...dArgs);
            };
          }
        }
        return node;
      };
    };

    wrapFactory('createOscillator', 'oscillators', 'source');
    wrapFactory('createBufferSource', 'loopBufferSources', 'source');
    wrapFactory('createBiquadFilter', 'biquadFilters', 'aux');
    wrapFactory('createGain', 'gainNodes', 'aux');
    wrapFactory('createPanner', 'panners3D', 'aux');
    wrapFactory('createStereoPanner', 'stereoPanners', 'aux');
    wrapFactory('createDelay', 'delayLines', 'aux');
    wrapFactory('createWaveShaper', 'waveShapers', 'aux');
    wrapFactory('createDynamicsCompressor', 'compressors', 'aux');
    wrapFactory('createConvolver', 'convolvers', 'aux');
    wrapFactory('createAnalyser', 'analysers', 'aux');
    wrapFactory('createChannelMerger', 'channelMergers', 'aux');

    if (typeof ctx.createBuffer === 'function') {
      const origBuf = ctx.createBuffer.bind(ctx);
      ctx.createBuffer = (numChannels, length, sampleRate) => {
        const buf = origBuf(numChannels, length, sampleRate);
        ns.audioBuffersCount++;
        ns.audioBufferBytes += Math.max(0, (Number(numChannels) || 1) * (Number(length) || 0) * 4);
        return buf;
      };
    }
  }

  toggle(forceState = undefined) {
    if (!this.ctx) {
      this._initContext();
    }
    if (!this.ctx) return false;

    this.enabled = typeof forceState === 'boolean' ? forceState : !this.enabled;
    this._telemetry.enabled = this.enabled;

    if (this.enabled && this.ctx.state === 'suspended') {
      this._forensic.resumeAttempts++;
      try {
        const p = this.ctx.resume();
        if (p && typeof p.then === 'function') {
          p.then(() => {
            this._markUnlockedMilestoneIfRunning();
          }).catch((err) => {
            this._forensic.resumeFailures++;
            this.recordExternalError(err, 'ctx.resume');
          });
        }
      } catch (err) {
        this._forensic.resumeFailures++;
        this.recordExternalError(err, 'ctx.resume');
      }
    } else if (this.enabled && this.ctx.state === 'running') {
      this._markUnlockedMilestoneIfRunning();
    }
    if (this.enabled && this._hibernationSlots && this._hibernationSlots.length > 0) {
      const nowWake = this.ctx.currentTime;
      for (let i = 0; i < this._hibernationSlots.length; i++) {
        const slot = this._hibernationSlots[i];
        const gp = slot.gainNode.gain;
        const rawVal = gp._lastTgt !== undefined ? gp._lastTgt : gp.value;
        if (slot.hibernated && Number.isFinite(rawVal) && rawVal > 0.00025) {
          this._wakeSubBusSlot(slot, nowWake);
          gp.setTargetAtTime(rawVal, nowWake, 0.10);
        }
      }
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

  _setPannerOrientation(panner, ox, oy, oz, now, instant = false) {
    if (!panner || !Number.isFinite(ox) || !Number.isFinite(oy) || !Number.isFinite(oz)) return;
    const len = Math.hypot(ox, oy, oz);
    if (len < 1e-4) return;
    const nx = ox / len;
    const ny = oy / len;
    const nz = oz / len;
    if (panner.orientationX && panner.orientationY && panner.orientationZ) {
      this._setParam(panner.orientationX, nx, now, 0.04, instant);
      this._setParam(panner.orientationY, ny, now, 0.04, instant);
      this._setParam(panner.orientationZ, nz, now, 0.04, instant);
    } else if (typeof panner.setOrientation === 'function') {
      panner.setOrientation(nx, ny, nz);
    }
  }

  _createStereoPannerSafe(initialPan = 0.0) {
    if (!this.ctx || typeof this.ctx.createStereoPanner !== 'function') return null;
    const sp = this.ctx.createStereoPanner();
    sp.pan.value = clamp(initialPan, -1.0, 1.0);
    return sp;
  }

  _createWarmWaveshaper(drive = 1.9) {
    if (!this.ctx || typeof this.ctx.createWaveShaper !== 'function') return null;
    if (!this._warmShaperCurve) {
      const n = 257;
      const curve = new Float32Array(n);
      const norm = Math.tanh(drive) || 1.0;
      for (let i = 0; i < n; i++) {
        const x = (i * 2.0) / (n - 1) - 1.0;
        curve[i] = Math.tanh(x * drive) / norm;
      }
      this._warmShaperCurve = curve;
    }
    const ws = this.ctx.createWaveShaper();
    ws.curve = this._warmShaperCurve;
    ws.oversample = 'none';
    return ws;
  }

  _createBed3DPanner(refDistance = 12.0, rolloffFactor = 0.65, maxDistance = 900.0) {
    const pan = this.ctx.createPanner();
    pan.panningModel = this.spatialOutputMode === 'SPEAKER' ? 'equalpower' : 'HRTF';
    pan.distanceModel = 'inverse';
    pan.refDistance = refDistance;
    pan.maxDistance = maxDistance;
    pan.rolloffFactor = rolloffFactor;
    return pan;
  }

  _create3DEmitter(opts = {}) {
    const airLP = this.ctx.createBiquadFilter();
    airLP.type = 'lowpass';
    airLP.frequency.value = opts.initCutoff || 8500;
    airLP.Q.value = 0.5;

    let groundComb = null;
    if (opts.enableGroundComb) {
      groundComb = this.ctx.createBiquadFilter();
      groundComb.type = 'peaking';
      groundComb.frequency.value = 850;
      groundComb.Q.value = 2.35;
      groundComb.gain.value = 0.0;
      groundComb.connect(airLP);
    }

    // Dolby Atmos Binaural Pinna Rear-Concha Shadow (4.2 kHz) & Blauert Overhead Elevation Presence (7.8 kHz)
    const pinnaRearShelf = this.ctx.createBiquadFilter();
    pinnaRearShelf.type = 'highshelf';
    pinnaRearShelf.frequency.value = 4200;
    pinnaRearShelf.gain.value = 0.0;

    const pinnaElevPeak = this.ctx.createBiquadFilter();
    pinnaElevPeak.type = 'peaking';
    pinnaElevPeak.frequency.value = 7800;
    pinnaElevPeak.Q.value = 1.25;
    pinnaElevPeak.gain.value = 0.0;

    const gain = this.ctx.createGain();
    gain.gain.value = 0.0;

    const directGain = this.ctx.createGain();
    directGain.gain.value = 1.0;

    const spreadSend = this.ctx.createGain();
    spreadSend.gain.value = 0.0;

    const heightSend = this.ctx.createGain();
    heightSend.gain.value = 0.0;

    const lfeSend = this.ctx.createGain();
    lfeSend.gain.value = opts.lfeWeight ?? 0.0;

    const panner = this.ctx.createPanner();
    panner.panningModel = this.spatialOutputMode === 'SPEAKER' ? 'equalpower' : 'HRTF';
    panner.distanceModel = 'inverse';
    panner.refDistance = opts.refDistance || 9.0;
    panner.maxDistance = opts.maxDistance || 1400.0;
    panner.rolloffFactor = opts.rolloffFactor || 1.05;
    panner.coneInnerAngle = opts.coneInnerAngle ?? 360;
    panner.coneOuterAngle = opts.coneOuterAngle ?? 360;
    panner.coneOuterGain = opts.coneOuterGain ?? 1.0;

    const revSend = this.ctx.createGain();
    revSend.gain.value = opts.reverbWeight ?? 0.35;

    airLP.connect(pinnaRearShelf);
    pinnaRearShelf.connect(pinnaElevPeak);
    pinnaElevPeak.connect(gain);

    gain.connect(directGain);
    directGain.connect(panner);
    panner.connect(this.worldSumGain);
    panner.connect(revSend);
    revSend.connect(this.reverbSendGain);

    if (this.atmosSpreadIn) {
      gain.connect(spreadSend);
      spreadSend.connect(this.atmosSpreadIn);
    }
    if (this.heightDomeIn) {
      gain.connect(heightSend);
      heightSend.connect(this.heightDomeIn);
    }
    if (this.lfeSumGain) {
      gain.connect(lfeSend);
      lfeSend.connect(this.lfeSumGain);
    }

    return {
      input: groundComb || airLP,
      groundComb,
      airLP,
      pinnaRearShelf,
      pinnaElevPeak,
      gain,
      directGain,
      spreadSend,
      heightSend,
      lfeSend,
      panner,
      revSend,
      sourceRadiusM: opts.sourceRadiusM || 2.0,
      lfeWeight: opts.lfeWeight || 0.0
    };
  }

  _applyAtmosCueToEmitter(em, srcX, srcY, srcZ, now, tau = 0.08) {
    if (!em) return;
    const sph = AudioSynthesizer.projectWorldToListenerSpherical(srcX, srcY, srcZ, this._lis, this._sphScratch);
    const cue = AudioSynthesizer.computeAtmosPinnaAndSpreadCue(
      sph.azDeg,
      sph.elDeg,
      sph.distM,
      em.sourceRadiusM || 2.0,
      this.spatialOutputMode,
      this._cueScratch
    );
    if (em.pinnaRearShelf) {
      em.pinnaRearShelf.gain.setTargetAtTime(cue.rearShelfDb, now, tau);
    }
    if (em.pinnaElevPeak) {
      em.pinnaElevPeak.gain.setTargetAtTime(cue.elevationPeakDb, now, tau);
    }
    if (em.directGain) {
      em.directGain.gain.setTargetAtTime(cue.directGainMult, now, tau);
    }
    if (em.spreadSend) {
      em.spreadSend.gain.setTargetAtTime(cue.spreadSendGain, now, tau);
    }
    if (em.heightSend) {
      em.heightSend.gain.setTargetAtTime(cue.heightSendGain * 0.48, now, tau);
    }
    if (em.lfeSend) {
      const distLfe = clamp(1.0 - sph.distM / 220.0, 0.0, 1.0);
      em.lfeSend.gain.setTargetAtTime((em.lfeWeight || 0.0) * (0.28 + 0.72 * distLfe), now, tau);
    }
  }

  /**
   * IR 1 (2.35s): Long Concrete Bore & Stone Colonnade Cathedral Impulse Response
   * Models Subsea Underpass 9.5m concrete vault lateral slapbacks + warm progressive HF damping.
   */
  _buildCathedralTunnelIR(sr) {
    const durationSec = 2.35;
    const len = Math.floor(sr * durationSec);
    const ir = this.ctx.createBuffer(2, len, sr);
    const left = ir.getChannelData(0);
    const right = ir.getChannelData(1);

    const earlyTaps = [
      { ms: 13.8, gL: 0.54, gR: 0.32 },
      { ms: 24.5, gL: -0.39, gR: 0.51 },
      { ms: 38.2, gL: 0.35, gR: -0.37 },
      { ms: 54.6, gL: -0.27, gR: 0.30 },
      { ms: 76.4, gL: 0.22, gR: -0.24 },
      { ms: 104.0, gL: 0.16, gR: 0.18 }
    ];
    for (let t = 0; t < earlyTaps.length; t++) {
      const idx = Math.min(len - 1, Math.floor(earlyTaps[t].ms * 0.001 * sr));
      left[idx] += earlyTaps[t].gL;
      right[idx] += earlyTaps[t].gR;
    }

    let lpL = 0.0;
    let lpR = 0.0;
    const invSr = 1.0 / sr;
    const expDecayStep = Math.exp(-2.55 * invSr);
    const expAttackStep = Math.exp(-42.0 * invSr);
    let eDecay = 1.0;
    let eAttack = 1.0;
    let seed = 0x51a7c9e3 | 0;
    const invNorm = 1.0 / 2147483648.0;
    for (let i = 0; i < len; i++) {
      const tSec = i * invSr;
      const hfAlpha = tSec >= 2.0625 ? 0.09 : 0.42 - tSec * 0.16;
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      const wL = seed * invNorm;
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      const wR = seed * invNorm;
      lpL += (wL - lpL) * hfAlpha;
      lpR += (wR - lpR) * hfAlpha;
      const env = eDecay * (1.0 - eAttack) * 0.24;
      left[i] += lpL * env;
      right[i] += lpR * env;
      eDecay *= expDecayStep;
      eAttack *= expAttackStep;
    }
    return ir;
  }

  /**
   * IR 2 (0.68s): Crisp Steel-Arch Truss, Polycarbonate Canopy & Vehicle Cabin Impulse Response
   * Models tight 4.8..36ms early reflections inside Jembatan Teksas UI, Bus/Pilot cabins, & canopies.
   */
  _buildMetallicCanopyIR(sr) {
    const durationSec = 0.68;
    const len = Math.floor(sr * durationSec);
    const ir = this.ctx.createBuffer(2, len, sr);
    const left = ir.getChannelData(0);
    const right = ir.getChannelData(1);

    const earlyTaps = [
      { ms: 4.9, gL: 0.58, gR: 0.42 },
      { ms: 9.8, gL: -0.44, gR: 0.54 },
      { ms: 16.2, gL: 0.36, gR: -0.39 },
      { ms: 24.7, gL: -0.25, gR: 0.28 },
      { ms: 35.5, gL: 0.18, gR: -0.20 }
    ];
    for (let t = 0; t < earlyTaps.length; t++) {
      const idx = Math.min(len - 1, Math.floor(earlyTaps[t].ms * 0.001 * sr));
      left[idx] += earlyTaps[t].gL;
      right[idx] += earlyTaps[t].gR;
    }

    const invSr = 1.0 / sr;
    const expDecayStep = Math.exp(-8.4 * invSr);
    const expAttackStep = Math.exp(-90.0 * invSr);
    const w1 = 2.0 * Math.PI * 1180.0 * invSr;
    const w2 = 2.0 * Math.PI * 1890.0 * invSr;
    let eDecay = 1.0;
    let eAttack = 1.0;
    let seed = 0x3b8f264d | 0;
    const invNorm = 0.24 / 2147483648.0;
    for (let i = 0; i < len; i++) {
      const modalShimmer = 0.12 * Math.sin(w1 * i) + 0.08 * Math.sin(w2 * i);
      const env = eDecay * (1.0 - eAttack);
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      const nL = seed * invNorm;
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      const nR = seed * invNorm;
      left[i] += (nL + modalShimmer) * env;
      right[i] += (nR - modalShimmer) * env;
      eDecay *= expDecayStep;
      eAttack *= expAttackStep;
    }
    return ir;
  }

  _initContext() {
    const f = this._forensic;
    const hostObj =
      typeof window !== 'undefined' ? window : typeof globalThis !== 'undefined' ? globalThis : null;
    const AudioCtx = hostObj ? hostObj.AudioContext || hostObj.webkitAudioContext : null;
    if (!AudioCtx) {
      if (typeof window !== 'undefined' && f.initErrors.length < 8) {
        f.initErrors.push('Web Audio API (AudioContext) tidak didukung browser');
      }
      return;
    }

    f.initAttempts++;
    const nowPerf = () => (typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now());
    const tInit0 = nowPerf();
    const latencyHint = this._preferPlaybackBuffer ? 'playback' : 'interactive';
    f.requestedLatencyHint = latencyHint;
    try {
      try {
        this.ctx = new AudioCtx({ latencyHint });
      } catch (_optErr) {
        this.ctx = new AudioCtx();
      }
    } catch (ctxErr) {
      const msg = String(ctxErr && ctxErr.message ? ctxErr.message : ctxErr).slice(0, 220);
      if (f.initErrors.length < 8) f.initErrors.push(msg);
      if (
        typeof window !== 'undefined' &&
        window.NexusForensic &&
        typeof window.NexusForensic.recordCustomError === 'function'
      ) {
        window.NexusForensic.recordCustomError('webaudio_init_error', msg, 'AudioSynthesizer._initContext');
      }
      return;
    }
    const tCtxCreated = nowPerf();

    this._installForensicNodeCounters(this.ctx);
    f.lastStateChange = this.ctx.state || 'suspended';
    if (typeof this.ctx.addEventListener === 'function') {
      this.ctx.addEventListener('statechange', () => {
        if (!this.ctx) return;
        const st = this.ctx.state || 'unknown';
        f.stateTransitionsCount++;
        f.lastStateChange = st;
        f.clock.prevPerfMs = 0.0;
        f.clock.prevCtxSec = -1.0;
        if (st === 'running') {
          this._markUnlockedMilestoneIfRunning();
        }
      });
    }

    f._inGraphBuild = true;
    f._tmpCtxCreateMs = Math.max(0.0, tCtxCreated - tInit0);
    try {
      this._buildPersistentGraph(this.ctx.sampleRate || 48000);
    } catch (graphErr) {
      const msg = String(graphErr && graphErr.message ? graphErr.message : graphErr).slice(0, 220);
      if (f.initErrors.length < 8) f.initErrors.push(`graph_build: ${msg}`);
      this.recordExternalError(graphErr, '_buildPersistentGraph');
    } finally {
      f._inGraphBuild = false;
      const tInit1 = nowPerf();
      f.initDurationMs = Math.max(0.0, +(tInit1 - tInit0).toFixed(2));
      this._markUnlockedMilestoneIfRunning();
    }
  }

  _buildPersistentGraph(sr) {
    const _nowP = () => (typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now());
    const _tG0 = _nowP();

    // 0. Master Subsonic Guard (24 Hz) -> ISO 226 Fletcher-Munson Equal-Loudness -> Safety Compressor -> Destination
    this.worldSumGain = this.ctx.createGain();
    this.worldSumGain.gain.value = 1.0;

    this.subsonicFilter = this.ctx.createBiquadFilter();
    this.subsonicFilter.type = 'highpass';
    this.subsonicFilter.frequency.value = 24.0;
    this.subsonicFilter.Q.value = 0.707;

    this.isoLowShelf = this.ctx.createBiquadFilter();
    this.isoLowShelf.type = 'lowshelf';
    this.isoLowShelf.frequency.value = 95.0;
    this.isoLowShelf.gain.value = 0.0;

    this.isoHighShelf = this.ctx.createBiquadFilter();
    this.isoHighShelf.type = 'highshelf';
    this.isoHighShelf.frequency.value = 6800.0;
    this.isoHighShelf.gain.value = 0.0;
    this._applyEqualLoudnessContour(this.ctx.currentTime);

    this.masterCompressor = this.ctx.createDynamicsCompressor();
    this.masterCompressor.threshold.value = -9.0;
    this.masterCompressor.knee.value = 12.0;
    this.masterCompressor.ratio.value = 4.5;
    this.masterCompressor.attack.value = 0.004;
    this.masterCompressor.release.value = 0.16;

    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = 0.0;

    this.worldSumGain.connect(this.subsonicFilter);
    this.subsonicFilter.connect(this.isoLowShelf);
    this.isoLowShelf.connect(this.isoHighShelf);
    this.isoHighShelf.connect(this.masterCompressor);
    this.masterCompressor.connect(this.masterGain);
    if (typeof this.ctx.createAnalyser === 'function') {
      this.masterAnalyser = this.ctx.createAnalyser();
      this.masterAnalyser.fftSize = 64;
      this.masterAnalyser.smoothingTimeConstant = 0.78;
      this.masterGain.connect(this.masterAnalyser);
      this.masterAnalyser.connect(this.ctx.destination);
    } else {
      this.masterGain.connect(this.ctx.destination);
    }

    // 0b. Dual-Character Stereo Convolution Reverb Engine (Concrete Vault IR + Metallic Truss/Cabin IR)
    this.reverbSendGain = this.ctx.createGain();
    this.reverbSendGain.gain.value = 1.0;

    this.cathedralSendGain = this.ctx.createGain();
    this.cathedralSendGain.gain.value = 0.09;
    this.convolverCathedral = this.ctx.createConvolver();
    this.convolverCathedral.buffer = this._buildCathedralTunnelIR(sr);
    this.convolverNode = this.convolverCathedral; // Backward-compatible reference

    this.metallicSendGain = this.ctx.createGain();
    this.metallicSendGain.gain.value = 0.04;
    this.convolverMetallic = this.ctx.createConvolver();
    this.convolverMetallic.buffer = this._buildMetallicCanopyIR(sr);

    this.reverbReturnGain = this.ctx.createGain();
    this.reverbReturnGain.gain.value = 0.92;

    this.reverbSendGain.connect(this.cathedralSendGain);
    this.cathedralSendGain.connect(this.convolverCathedral);
    this.convolverCathedral.connect(this.reverbReturnGain);

    this.reverbSendGain.connect(this.metallicSendGain);
    this.metallicSendGain.connect(this.convolverMetallic);
    this.convolverMetallic.connect(this.reverbReturnGain);

    this.reverbReturnGain.connect(this.worldSumGain);
    const _tG1 = _nowP();

    // 0c. Dolby Atmos 7.1.4 Sub-Buses: (.1) LFE Sub-Bass + Volumetric Object Spread + (.4) Overhead Height Dome
    this.lfeSumGain = this.ctx.createGain();
    this.lfeSumGain.gain.value = 1.0;
    const lfeLpA = this.ctx.createBiquadFilter();
    lfeLpA.type = 'lowpass';
    lfeLpA.frequency.value = 108.0;
    lfeLpA.Q.value = 0.707;
    const lfeLpB = this.ctx.createBiquadFilter();
    lfeLpB.type = 'lowpass';
    lfeLpB.frequency.value = 108.0;
    lfeLpB.Q.value = 0.707;
    this.lfeMasterGain = this.ctx.createGain();
    this.lfeMasterGain.gain.value = 0.78;
    this.lfeSumGain.connect(lfeLpA);
    lfeLpA.connect(lfeLpB);
    lfeLpB.connect(this.lfeMasterGain);
    this.lfeMasterGain.connect(this.worldSumGain);

    this.atmosSpreadIn = this.ctx.createGain();
    this.atmosSpreadIn.gain.value = 1.0;
    const spreadHp = this.ctx.createBiquadFilter();
    spreadHp.type = 'highpass';
    spreadHp.frequency.value = 155.0;
    const spreadLp = this.ctx.createBiquadFilter();
    spreadLp.type = 'lowpass';
    spreadLp.frequency.value = 7200.0;
    this.atmosSpreadIn.connect(spreadHp);
    spreadHp.connect(spreadLp);
    if (typeof this.ctx.createDelay === 'function') {
      const dL = this.ctx.createDelay(0.05);
      dL.delayTime.value = 0.0073;
      const dR = this.ctx.createDelay(0.05);
      dR.delayTime.value = 0.0119;
      const pL = this._createStereoPannerSafe(-0.88);
      const pR = this._createStereoPannerSafe(0.88);
      const gSide = this.ctx.createGain();
      gSide.gain.value = 0.56;
      spreadLp.connect(dL);
      spreadLp.connect(dR);
      if (pL && pR) {
        dL.connect(pL);
        dR.connect(pR);
        pL.connect(gSide);
        pR.connect(gSide);
      } else {
        dL.connect(gSide);
        dR.connect(gSide);
      }
      gSide.connect(this.worldSumGain);
    } else {
      spreadLp.connect(this.worldSumGain);
    }

    // (.4) Overhead Atmos Height Dome Ceiling Panners + 3D Vault/Arch Ceiling Reverb Tap
    this.heightDomeIn = this.ctx.createGain();
    this.heightDomeIn.gain.value = 1.0;
    this.heightFrontPanner = this._createBed3DPanner(7.5, 0.52, 600.0);
    this.heightRearPanner = this._createBed3DPanner(7.5, 0.52, 600.0);
    const hFrontGain = this.ctx.createGain();
    hFrontGain.gain.value = 0.58;
    const hRearGain = this.ctx.createGain();
    hRearGain.gain.value = 0.52;
    this.heightDomeIn.connect(hFrontGain);
    this.heightDomeIn.connect(hRearGain);
    hFrontGain.connect(this.heightFrontPanner);
    hRearGain.connect(this.heightRearPanner);
    this.heightFrontPanner.connect(this.worldSumGain);
    this.heightRearPanner.connect(this.worldSumGain);

    const heightRevHp = this.ctx.createBiquadFilter();
    heightRevHp.type = 'highpass';
    heightRevHp.frequency.value = 360.0;
    this.heightRevGain = this.ctx.createGain();
    this.heightRevGain.gain.value = 0.34;
    this.reverbReturnGain.connect(heightRevHp);
    if (typeof this.ctx.createDelay === 'function') {
      const hRevDelay = this.ctx.createDelay(0.05);
      hRevDelay.delayTime.value = 0.0115;
      heightRevHp.connect(hRevDelay);
      hRevDelay.connect(this.heightRevGain);
    } else {
      heightRevHp.connect(this.heightRevGain);
    }
    this.heightRevGain.connect(this.heightDomeIn);

    this._bedPanners = {
      breeze: this._createBed3DPanner(15.0, 0.42, 900.0),
      rainFront: this._createBed3DPanner(8.5, 0.48, 600.0),
      rainRear: this._createBed3DPanner(8.5, 0.48, 600.0),
      surf: this._createBed3DPanner(18.0, 0.52, 1200.0),
      cricket: this._createBed3DPanner(8.5, 0.76, 500.0),
      canopy: this._createBed3DPanner(11.5, 0.60, 700.0),
      tonggeret: this._createBed3DPanner(11.5, 0.68, 700.0),
      social: this._createBed3DPanner(14.0, 0.72, 600.0),
      flag: this._createBed3DPanner(8.5, 0.92, 450.0),
      truss: this._createBed3DPanner(8.0, 0.88, 450.0),
      fumarole: this._createBed3DPanner(18.0, 0.80, 900.0),
      turapLap: this._createBed3DPanner(9.0, 0.82, 500.0)
    };
    const _tG2 = _nowP();

    // 1. Procedural 4-Second Seamless Binaural Stereo-Decorrelated Pink-Noise & Droplet Patter Buffers
    //    (Synthesized via deterministic 32-bit xorshift PRNG for sub-10ms zero-allocation buffer init)
    const bufferSize = sr * 4;
    const pinkBuffer = this.ctx.createBuffer(2, bufferSize, sr);
    const invRandSigned = 1.0 / 2147483648.0;
    const invRandUnit = 1.0 / 4294967296.0;
    for (let ch = 0; ch < 2; ch++) {
      const pinkOut = pinkBuffer.getChannelData(ch);
      let b0 = 0.0;
      let b1 = 0.0;
      let b2 = 0.0;
      let seed = (ch === 0 ? 0x6d2b79f5 : 0x9e3779b9) | 0;
      for (let i = 0; i < bufferSize; i++) {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        const white = seed * invRandSigned;
        b0 = 0.99765 * b0 + white * 0.099046;
        b1 = 0.963 * b1 + white * 0.2965164;
        b2 = 0.57 * b2 + white * 1.0526913;
        pinkOut[i] = (b0 + b1 + b2 + white * 0.1848) * 0.075;
      }
      // Crossfade final 512 samples per channel for zero loop-seam click
      for (let k = 0; k < 512; k++) {
        const a = k / 512;
        pinkOut[bufferSize - 512 + k] = pinkOut[bufferSize - 512 + k] * (1.0 - a) + pinkOut[k] * a;
      }
    }
    this._pinkBuffer = pinkBuffer;

    const patterBuffer = this.ctx.createBuffer(2, bufferSize, sr);
    for (let ch = 0; ch < 2; ch++) {
      const patterOut = patterBuffer.getChannelData(ch);
      let env = 0.0;
      let seed = (ch === 0 ? 0x27d4eb2d : 0x85ebca6b) | 0;
      for (let i = 0; i < bufferSize; i++) {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        const uProb = (seed >>> 0) * invRandUnit;
        if (uProb < 0.0042) {
          seed ^= seed << 13;
          seed ^= seed >>> 17;
          seed ^= seed << 5;
          const uAmp = (seed >>> 0) * invRandUnit;
          env = Math.min(1.0, env + 0.35 + uAmp * 0.65);
        }
        env *= 0.965;
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        patterOut[i] = seed * invRandSigned * env * 0.26;
      }
      for (let k = 0; k < 256; k++) {
        const a = k / 256;
        patterOut[bufferSize - 256 + k] = patterOut[bufferSize - 256 + k] * (1.0 - a) + patterOut[k] * a;
      }
    }
    this._patterBuffer = patterBuffer;
    const _tG3 = _nowP();

    // 2. Architectural Weather Occlusion + Priority Sidechain Ducking Sub-Bus
    this.weatherOccFilter = this.ctx.createBiquadFilter();
    this.weatherOccFilter.type = 'lowpass';
    this.weatherOccFilter.frequency.value = 13500;

    this.weatherOccGain = this.ctx.createGain();
    this.weatherOccGain.gain.value = 1.0;

    this.ambientDuckGain = this.ctx.createGain();
    this.ambientDuckGain.gain.value = 1.0;
    this.ambientDuckGain.connect(this.weatherOccFilter);

    this.weatherOccFilter.connect(this.weatherOccGain);
    this.weatherOccGain.connect(this.worldSumGain);

    // Bus 1: Campus Breeze, Canopy Rustle & Turbulent Storm Wind Squall (3D Upwind Anchor)
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
    this.breezeGain.connect(this._bedPanners.breeze);
    this._bedPanners.breeze.connect(this.weatherOccFilter);
    breezeSrc.start();

    // Bus 2: Tropical Rain Downpour Wash (3D .4 Overhead Sky Canopy Front + Rear)
    const roarSrc = this.ctx.createBufferSource();
    roarSrc.buffer = pinkBuffer;
    roarSrc.loop = true;
    this.rainRoarFilter = this.ctx.createBiquadFilter();
    this.rainRoarFilter.type = 'bandpass';
    this.rainRoarFilter.frequency.value = 960;
    this.rainRoarFilter.Q.value = 0.62;
    this.rainRoarGain = this.ctx.createGain();
    this.rainRoarGain.gain.value = 0.0;
    const rainFrontSplit = this.ctx.createGain();
    rainFrontSplit.gain.value = 0.62;
    const rainRearSplit = this.ctx.createGain();
    rainRearSplit.gain.value = 0.56;
    roarSrc.connect(this.rainRoarFilter);
    this.rainRoarFilter.connect(this.rainRoarGain);
    this.rainRoarGain.connect(rainFrontSplit);
    this.rainRoarGain.connect(rainRearSplit);
    rainFrontSplit.connect(this._bedPanners.rainFront);
    rainRearSplit.connect(this._bedPanners.rainRear);
    this._bedPanners.rainFront.connect(this.weatherOccFilter);
    this._bedPanners.rainRear.connect(this.weatherOccFilter);
    roarSrc.start();

    // Bus 3: Crisp Nearby Raindrop Impact + Roof-Material Resonance Filter (Terracotta / Polycarbonate / Windshield)
    const patterSrc = this.ctx.createBufferSource();
    patterSrc.buffer = patterBuffer;
    patterSrc.loop = true;
    this.rainPatterFilter = this.ctx.createBiquadFilter();
    this.rainPatterFilter.type = 'highpass';
    this.rainPatterFilter.frequency.value = 2100;
    this.rainPatterGain = this.ctx.createGain();
    this.rainPatterGain.gain.value = 0.0;

    this.roofResonanceFilter = this.ctx.createBiquadFilter();
    this.roofResonanceFilter.type = 'bandpass';
    this.roofResonanceFilter.frequency.value = 780;
    this.roofResonanceFilter.Q.value = 2.1;
    this.roofResonanceGain = this.ctx.createGain();
    this.roofResonanceGain.gain.value = 0.0;

    patterSrc.connect(this.rainPatterFilter);
    this.rainPatterFilter.connect(this.rainPatterGain);
    this.rainPatterGain.connect(this.worldSumGain);

    patterSrc.connect(this.roofResonanceFilter);
    this.roofResonanceFilter.connect(this.roofResonanceGain);
    this.roofResonanceGain.connect(this.worldSumGain);
    this.roofResonanceGain.connect(this.reverbSendGain);
    patterSrc.start();

    // Bus 4: 3D Coastline-Anchored Archipelago Shoreline Surf + (.1) LFE Ocean Swell Body
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
    const surfLfeSend = this.ctx.createGain();
    surfLfeSend.gain.value = 0.36;
    const surfSpreadSend = this.ctx.createGain();
    surfSpreadSend.gain.value = 0.42;
    surfSrc.connect(this.surfFilter);
    this.surfFilter.connect(this.surfGain);
    this.surfGain.connect(this._bedPanners.surf);
    this._bedPanners.surf.connect(this.weatherOccFilter);
    this.surfGain.connect(surfLfeSend);
    surfLfeSend.connect(this.lfeSumGain);
    this.surfGain.connect(surfSpreadSend);
    surfSpreadSend.connect(this.atmosSpreadIn);
    surfSrc.start();

    // Bus 5: Subsea Underpass Jet-Fan Ventilation & Hydrostatic Bore Ambiance + (.1) LFE Bore Sub
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
    const tunLfeSend = this.ctx.createGain();
    tunLfeSend.gain.value = 0.48;
    tunOscA.connect(tunSubGain);
    tunOscB.connect(tunSubGain);
    tunSubGain.connect(tunLp);
    tunNoise.connect(tunLp);
    tunLp.connect(this.tunnelHumGain);
    this.tunnelHumGain.connect(this.worldSumGain);
    this.tunnelHumGain.connect(this.reverbSendGain);
    this.tunnelHumGain.connect(tunLfeSend);
    tunLfeSend.connect(this.lfeSumGain);
    tunOscA.start();
    tunOscB.start();
    tunNoise.start();

    // Bus 6: Diurnal Nocturnal Nusantara Cricket Stridulation Chorus (3D Ground-Foliage Anchor)
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
    this.cricketGain.connect(this._bedPanners.cricket);
    this._bedPanners.cricket.connect(this.ambientDuckGain);
    crikOscA.start();
    crikOscB.start();
    crikLfo.start();

    // Bus 7: DJI Avata 2 5-Blade Ducted Cinewhoop Asymmetric Motor Bank + Warm Shaper + Duct Helmholtz + Stereo Roll Pan
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

    this.avataDuctFilter = this.ctx.createBiquadFilter();
    this.avataDuctFilter.type = 'bandpass';
    this.avataDuctFilter.frequency.value = 1180;
    this.avataDuctFilter.Q.value = 3.1;
    this.avataDuctGain = this.ctx.createGain();
    this.avataDuctGain.gain.value = 0.28;

    const avataShaper = this._createWarmWaveshaper(1.85);
    this.avataMotorGain = this.ctx.createGain();
    this.avataMotorGain.gain.value = 0.0;
    this.avataPan = this._createStereoPannerSafe(0.0);

    this.avataMotorOscA.connect(this.avataMotorFilter);
    this.avataMotorOscB.connect(this.avataMotorFilter);
    this.avataMotorOscC.connect(avataOscCGain);
    avataOscCGain.connect(this.avataMotorFilter);
    this.avataMotorOscA.connect(this.avataDuctFilter);
    this.avataDuctFilter.connect(this.avataDuctGain);
    if (avataShaper) {
      this.avataMotorFilter.connect(avataShaper);
      this.avataDuctGain.connect(avataShaper);
      avataShaper.connect(this.avataMotorGain);
    } else {
      this.avataMotorFilter.connect(this.avataMotorGain);
      this.avataDuctGain.connect(this.avataMotorGain);
    }
    if (this.avataPan) {
      this.avataMotorGain.connect(this.avataPan);
      this.avataPan.connect(this.worldSumGain);
      this.avataPan.connect(this.reverbSendGain);
    } else {
      this.avataMotorGain.connect(this.worldSumGain);
      this.avataMotorGain.connect(this.reverbSendGain);
    }
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

    // Bus 9: Feline Purr Resonator (Active in FPS_CAT or close cat contemplation + .1 LFE Chest Warmth)
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
    const purrLfeSend = this.ctx.createGain();
    purrLfeSend.gain.value = 0.55;
    this.catPurrOsc.connect(purrMod);
    purrHarm.connect(purrMod);
    purrMod.connect(purrLp);
    purrLp.connect(this.catPurrGain);
    this.catPurrGain.connect(this.worldSumGain);
    this.catPurrGain.connect(purrLfeSend);
    purrLfeSend.connect(this.lfeSumGain);
    this.catPurrOsc.start();
    purrHarm.start();
    purrBreathLfo.start();

    // =========================================================================
    // ITERASI 2 + v633 ATMOS 3D WORLD-ANCHORED MICRO-BIOME AMBIENCE SUB-BUSES
    // =========================================================================

    // Bus 10: Tropical Rainforest Canopy Leaf-Rustle Wave (3D Overhead Tree-Crown Anchor)
    const canopySrc = this.ctx.createBufferSource();
    canopySrc.buffer = pinkBuffer;
    canopySrc.loop = true;
    canopySrc.playbackRate.value = 1.18;
    this.canopyRustleFilter = this.ctx.createBiquadFilter();
    this.canopyRustleFilter.type = 'bandpass';
    this.canopyRustleFilter.frequency.value = 2150;
    this.canopyRustleFilter.Q.value = 0.85;
    this.canopyRustleGain = this.ctx.createGain();
    this.canopyRustleGain.gain.value = 0.0;
    canopySrc.connect(this.canopyRustleFilter);
    this.canopyRustleFilter.connect(this.canopyRustleGain);
    this.canopyRustleGain.connect(this._bedPanners.canopy);
    this._bedPanners.canopy.connect(this.ambientDuckGain);
    canopySrc.start();

    // Bus 11: Nusantara Afternoon/Dusk Forest Tonggeret (Cicada) Chorus (3D Overhead Tree-Crown Anchor)
    const tongOscA = this.ctx.createOscillator();
    tongOscA.type = 'sawtooth';
    tongOscA.frequency.value = 3910;
    const tongOscB = this.ctx.createOscillator();
    tongOscB.type = 'triangle';
    tongOscB.frequency.value = 3972;
    const tongBp = this.ctx.createBiquadFilter();
    tongBp.type = 'bandpass';
    tongBp.frequency.value = 3940;
    tongBp.Q.value = 5.2;
    const tongGate = this.ctx.createGain();
    tongGate.gain.value = 0.52;
    const tongLfo = this.ctx.createOscillator();
    tongLfo.type = 'sine';
    tongLfo.frequency.value = 23.5;
    const tongLfoDepth = this.ctx.createGain();
    tongLfoDepth.gain.value = 0.44;
    tongLfo.connect(tongLfoDepth);
    tongLfoDepth.connect(tongGate.gain);
    this.tonggeretGain = this.ctx.createGain();
    this.tonggeretGain.gain.value = 0.0;
    tongOscA.connect(tongBp);
    tongOscB.connect(tongBp);
    tongBp.connect(tongGate);
    tongGate.connect(this.tonggeretGain);
    this.tonggeretGain.connect(this._bedPanners.tonggeret);
    this._bedPanners.tonggeret.connect(this.ambientDuckGain);
    tongOscA.start();
    tongOscB.start();
    tongLfo.start();

    // Bus 12: Campus Academic Plaza & Kiosk Conversational Formant Murmur (3D Nearest Plaza Anchor)
    const socSrc = this.ctx.createBufferSource();
    socSrc.buffer = pinkBuffer;
    socSrc.loop = true;
    socSrc.playbackRate.value = 0.94;
    const socFormA = this.ctx.createBiquadFilter();
    socFormA.type = 'bandpass';
    socFormA.frequency.value = 510;
    socFormA.Q.value = 2.6;
    const socFormB = this.ctx.createBiquadFilter();
    socFormB.type = 'bandpass';
    socFormB.frequency.value = 1160;
    socFormB.Q.value = 2.9;
    const socCadence = this.ctx.createGain();
    socCadence.gain.value = 0.62;
    const socLfo = this.ctx.createOscillator();
    socLfo.type = 'sine';
    socLfo.frequency.value = 2.7;
    const socLfoDepth = this.ctx.createGain();
    socLfoDepth.gain.value = 0.28;
    socLfo.connect(socLfoDepth);
    socLfoDepth.connect(socCadence.gain);
    this.socialMurmurGain = this.ctx.createGain();
    this.socialMurmurGain.gain.value = 0.0;
    socSrc.connect(socFormA);
    socSrc.connect(socFormB);
    socFormA.connect(socCadence);
    socFormB.connect(socCadence);
    socCadence.connect(this.socialMurmurGain);
    this.socialMurmurGain.connect(this._bedPanners.social);
    this._bedPanners.social.connect(this.ambientDuckGain);
    this._bedPanners.social.connect(this.reverbSendGain);
    socSrc.start();
    socLfo.start();

    // Bus 13: 3D Lapangan 15 Bendera Silk Cloth Flutter + 3D Jembatan Teksas Overhead Steel-Arch Hum
    const flagSrc = this.ctx.createBufferSource();
    flagSrc.buffer = patterBuffer;
    flagSrc.loop = true;
    flagSrc.playbackRate.value = 0.72;
    this.flagFlutterFilter = this.ctx.createBiquadFilter();
    this.flagFlutterFilter.type = 'bandpass';
    this.flagFlutterFilter.frequency.value = 340;
    this.flagFlutterFilter.Q.value = 1.4;
    this.flagFlutterGain = this.ctx.createGain();
    this.flagFlutterGain.gain.value = 0.0;
    flagSrc.connect(this.flagFlutterFilter);
    this.flagFlutterFilter.connect(this.flagFlutterGain);
    this.flagFlutterGain.connect(this._bedPanners.flag);
    this._bedPanners.flag.connect(this.weatherOccFilter);
    flagSrc.start();

    const aeoOscA = this.ctx.createOscillator();
    aeoOscA.type = 'sine';
    aeoOscA.frequency.value = 218.0;
    const aeoOscB = this.ctx.createOscillator();
    aeoOscB.type = 'triangle';
    aeoOscB.frequency.value = 327.2;
    const aeoLp = this.ctx.createBiquadFilter();
    aeoLp.type = 'lowpass';
    aeoLp.frequency.value = 460.0;
    this.trussAeolianGain = this.ctx.createGain();
    this.trussAeolianGain.gain.value = 0.0;
    aeoOscA.connect(aeoLp);
    aeoOscB.connect(aeoLp);
    aeoLp.connect(this.trussAeolianGain);
    this.trussAeolianGain.connect(this._bedPanners.truss);
    this._bedPanners.truss.connect(this.worldSumGain);
    this._bedPanners.truss.connect(this.reverbSendGain);
    aeoOscA.start();
    aeoOscB.start();

    // Bus 14: 3D Kawah Gunung Api Pulau I Geothermal Sub-Bass Fumarole & Steam Vent Hiss + (.1) LFE
    const fumSubOsc = this.ctx.createOscillator();
    fumSubOsc.type = 'triangle';
    fumSubOsc.frequency.value = 46.5;
    const fumSubGain = this.ctx.createGain();
    fumSubGain.gain.value = 0.44;
    const fumSteamSrc = this.ctx.createBufferSource();
    fumSteamSrc.buffer = pinkBuffer;
    fumSteamSrc.loop = true;
    const fumSteamBp = this.ctx.createBiquadFilter();
    fumSteamBp.type = 'bandpass';
    fumSteamBp.frequency.value = 1320.0;
    fumSteamBp.Q.value = 1.15;
    this.fumaroleGain = this.ctx.createGain();
    this.fumaroleGain.gain.value = 0.0;
    const fumLfeSend = this.ctx.createGain();
    fumLfeSend.gain.value = 0.54;
    fumSubOsc.connect(fumSubGain);
    fumSubGain.connect(this.fumaroleGain);
    fumSteamSrc.connect(fumSteamBp);
    fumSteamBp.connect(this.fumaroleGain);
    this.fumaroleGain.connect(this._bedPanners.fumarole);
    this._bedPanners.fumarole.connect(this.worldSumGain);
    this.fumaroleGain.connect(fumLfeSend);
    fumLfeSend.connect(this.lfeSumGain);
    fumSubOsc.start();
    fumSteamSrc.start();

    // Bus 15: 3D Tepian Danau Kenanga, Danau Kecil, & Stone Turap Seawall Water-Lapping
    const lapSrc = this.ctx.createBufferSource();
    lapSrc.buffer = patterBuffer;
    lapSrc.loop = true;
    lapSrc.playbackRate.value = 0.88;
    this.turapLapFilter = this.ctx.createBiquadFilter();
    this.turapLapFilter.type = 'bandpass';
    this.turapLapFilter.frequency.value = 780.0;
    this.turapLapFilter.Q.value = 1.35;
    this.turapLapGain = this.ctx.createGain();
    this.turapLapGain.gain.value = 0.0;
    lapSrc.connect(this.turapLapFilter);
    this.turapLapFilter.connect(this.turapLapGain);
    this.turapLapGain.connect(this._bedPanners.turapLap);
    this._bedPanners.turapLap.connect(this.weatherOccFilter);
    lapSrc.start();

    // Bus 16 (v633): Cockpit Windshield Airframe Slipstream & 400Hz Avionics Inverter Bus (FPS_PILOT)
    const cockAirSrc = this.ctx.createBufferSource();
    cockAirSrc.buffer = pinkBuffer;
    cockAirSrc.loop = true;
    cockAirSrc.playbackRate.value = 1.12;
    this.cockpitAirFilter = this.ctx.createBiquadFilter();
    this.cockpitAirFilter.type = 'bandpass';
    this.cockpitAirFilter.frequency.value = 680.0;
    this.cockpitAirFilter.Q.value = 0.85;
    this.cockpitAirGain = this.ctx.createGain();
    this.cockpitAirGain.gain.value = 0.0;

    const avioOsc = this.ctx.createOscillator();
    avioOsc.type = 'sine';
    avioOsc.frequency.value = 400.0;
    const avioFanBp = this.ctx.createBiquadFilter();
    avioFanBp.type = 'bandpass';
    avioFanBp.frequency.value = 520.0;
    avioFanBp.Q.value = 1.6;
    const avioOscGain = this.ctx.createGain();
    avioOscGain.gain.value = 0.14;
    this.cockpitAvionicsGain = this.ctx.createGain();
    this.cockpitAvionicsGain.gain.value = 0.0;

    this.cockpitPanner = this._createBed3DPanner(2.4, 0.75, 40.0);
    cockAirSrc.connect(this.cockpitAirFilter);
    this.cockpitAirFilter.connect(this.cockpitAirGain);
    this.cockpitAirGain.connect(this.cockpitPanner);

    cockAirSrc.connect(avioFanBp);
    avioOsc.connect(avioOscGain);
    avioOscGain.connect(this.cockpitAvionicsGain);
    avioFanBp.connect(this.cockpitAvionicsGain);
    this.cockpitAvionicsGain.connect(this.cockpitPanner);

    this.cockpitPanner.connect(this.worldSumGain);
    this.cockpitPanner.connect(this.reverbSendGain);
    cockAirSrc.start();
    avioOsc.start();

    // =========================================================================
    // 11 DEDICATED SIMULTANEOUS 3D BINAURAL HRTF POINT-SOURCE EMITTERS
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

    // E3. Bus Kuning Lintas Pulau UI (Warm Soft-Saturated Traction Hum + Asphalt Tire Roll + Ground Comb)
    const eBus = this._create3DEmitter({ refDistance: 7.5, rolloffFactor: 1.05, reverbWeight: 0.55, enableGroundComb: true });
    const busOscA = this.ctx.createOscillator();
    busOscA.type = 'sawtooth';
    busOscA.frequency.value = 88;
    const busOscB = this.ctx.createOscillator();
    busOscB.type = 'triangle';
    busOscB.frequency.value = 176;
    const busShaper = this._createWarmWaveshaper(2.1);
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
    if (busShaper) {
      busOscA.connect(busShaper);
      busOscB.connect(busShaper);
      busShaper.connect(busToneLp);
    } else {
      busOscA.connect(busToneLp);
      busOscB.connect(busToneLp);
    }
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

    // E5. Taxi Helicopter Listrik eVTOL (PK-UIE Hexa-Rotor BVI Blade Thump + Coaxial Turbine Whine + Ground Comb)
    const eHeli = this._create3DEmitter({ refDistance: 13.0, rolloffFactor: 0.92, reverbWeight: 0.28, enableGroundComb: true });
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
    heliBladeLfo.frequency.value = 21.0;
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

    // E6. Dedicated Cessna 172 Skyhawk (PK-UIA) Lycoming 4-Cylinder Piston Prop + Exhaust Formant + Ground Comb
    const eCessna = this._create3DEmitter({
      refDistance: 20.0,
      rolloffFactor: 0.84,
      maxDistance: 1800.0,
      reverbWeight: 0.28,
      coneInnerAngle: 130,
      coneOuterAngle: 275,
      coneOuterGain: 0.58,
      enableGroundComb: true
    });
    const cesOscA = this.ctx.createOscillator();
    cesOscA.type = 'sawtooth';
    cesOscA.frequency.value = 96;
    const cesOscB = this.ctx.createOscillator();
    cesOscB.type = 'triangle';
    cesOscB.frequency.value = 192;
    const cesShaper = this._createWarmWaveshaper(2.2);
    const cesFormant = this.ctx.createBiquadFilter();
    cesFormant.type = 'lowpass';
    cesFormant.frequency.value = 560;
    const cesPropGain = this.ctx.createGain();
    cesPropGain.gain.value = 0.52;
    const cesWash = this.ctx.createBufferSource();
    cesWash.buffer = pinkBuffer;
    cesWash.loop = true;
    const cesWashBp = this.ctx.createBiquadFilter();
    cesWashBp.type = 'bandpass';
    cesWashBp.frequency.value = 340;
    cesWashBp.Q.value = 0.8;
    const cesWashGain = this.ctx.createGain();
    cesWashGain.gain.value = 0.28;
    if (cesShaper) {
      cesOscA.connect(cesShaper);
      cesOscB.connect(cesShaper);
      cesShaper.connect(cesFormant);
    } else {
      cesOscA.connect(cesFormant);
      cesOscB.connect(cesFormant);
    }
    cesFormant.connect(cesPropGain);
    cesPropGain.connect(eCessna.input);
    cesWash.connect(cesWashBp);
    cesWashBp.connect(cesWashGain);
    cesWashGain.connect(eCessna.input);
    cesOscA.start();
    cesOscB.start();
    cesWash.start();
    eCessna.propOscA = cesOscA;
    eCessna.propOscB = cesOscB;
    eCessna.formant = cesFormant;
    eCessna.propGain = cesPropGain;
    eCessna.roarBp = cesWashBp;
    eCessna.roarGain = cesWashGain;
    this._emitters.cessna = eCessna;

    // E7. Dedicated Citation Learjet (PK-UIB) Twin-Turbofan N1 Whine + Jet Core Roar + Ground Comb
    const eLearjet = this._create3DEmitter({
      refDistance: 24.0,
      rolloffFactor: 0.80,
      maxDistance: 2400.0,
      reverbWeight: 0.32,
      coneInnerAngle: 112,
      coneOuterAngle: 252,
      coneOuterGain: 0.45,
      enableGroundComb: true
    });
    const jetWhineA = this.ctx.createOscillator();
    jetWhineA.type = 'sine';
    jetWhineA.frequency.value = 1280;
    const jetWhineB = this.ctx.createOscillator();
    jetWhineB.type = 'triangle';
    jetWhineB.frequency.value = 1924;
    const jetWhineGain = this.ctx.createGain();
    jetWhineGain.gain.value = 0.22;
    const jetRoar = this.ctx.createBufferSource();
    jetRoar.buffer = pinkBuffer;
    jetRoar.loop = true;
    const jetRoarBp = this.ctx.createBiquadFilter();
    jetRoarBp.type = 'bandpass';
    jetRoarBp.frequency.value = 540;
    jetRoarBp.Q.value = 0.62;
    const jetRoarGain = this.ctx.createGain();
    jetRoarGain.gain.value = 0.55;
    jetWhineA.connect(jetWhineGain);
    jetWhineB.connect(jetWhineGain);
    jetWhineGain.connect(eLearjet.input);
    jetRoar.connect(jetRoarBp);
    jetRoarBp.connect(jetRoarGain);
    jetRoarGain.connect(eLearjet.input);
    jetWhineA.start();
    jetWhineB.start();
    jetRoar.start();
    eLearjet.jetWhine = jetWhineA;
    eLearjet.jetWhineB = jetWhineB;
    eLearjet.jetWhineGain = jetWhineGain;
    eLearjet.roarBp = jetRoarBp;
    eLearjet.roarGain = jetRoarGain;
    this._emitters.learjet = eLearjet;

    // E8. Dedicated Kapal Feri Katamaran Selat UI (Twin Marine Diesel + Deep Catamaran Wake)
    const eFerry = this._create3DEmitter({
      refDistance: 13.5,
      rolloffFactor: 0.96,
      reverbWeight: 0.22,
      coneInnerAngle: 150,
      coneOuterAngle: 300,
      coneOuterGain: 0.68
    });
    const ferOscA = this.ctx.createOscillator();
    ferOscA.type = 'triangle';
    ferOscA.frequency.value = 62;
    const ferOscB = this.ctx.createOscillator();
    ferOscB.type = 'sawtooth';
    ferOscB.frequency.value = 124;
    const ferLp = this.ctx.createBiquadFilter();
    ferLp.type = 'lowpass';
    ferLp.frequency.value = 260;
    const ferOscGain = this.ctx.createGain();
    ferOscGain.gain.value = 0.36;
    const ferWake = this.ctx.createBufferSource();
    ferWake.buffer = pinkBuffer;
    ferWake.loop = true;
    const ferWakeBp = this.ctx.createBiquadFilter();
    ferWakeBp.type = 'bandpass';
    ferWakeBp.frequency.value = 480;
    ferWakeBp.Q.value = 0.78;
    ferOscA.connect(ferLp);
    ferOscB.connect(ferLp);
    ferLp.connect(ferOscGain);
    ferOscGain.connect(eFerry.input);
    ferWake.connect(ferWakeBp);
    ferWakeBp.connect(eFerry.input);
    ferOscA.start();
    ferOscB.start();
    ferWake.start();
    eFerry.osc = ferOscA;
    eFerry.oscB = ferOscB;
    eFerry.wakeBp = ferWakeBp;
    this._emitters.ferry = eFerry;

    // E9. Dedicated Flybridge Mini Yacht Teluk U (Smooth Twin-Inboard Purr + Crisp Planing Spray)
    const eYacht = this._create3DEmitter({
      refDistance: 10.5,
      rolloffFactor: 1.04,
      reverbWeight: 0.20,
      coneInnerAngle: 155,
      coneOuterAngle: 305,
      coneOuterGain: 0.70
    });
    const yacOsc = this.ctx.createOscillator();
    yacOsc.type = 'triangle';
    yacOsc.frequency.value = 84;
    const yacOscGain = this.ctx.createGain();
    yacOscGain.gain.value = 0.28;
    const yacWake = this.ctx.createBufferSource();
    yacWake.buffer = pinkBuffer;
    yacWake.loop = true;
    const yacWakeBp = this.ctx.createBiquadFilter();
    yacWakeBp.type = 'bandpass';
    yacWakeBp.frequency.value = 680;
    yacWakeBp.Q.value = 0.85;
    yacOsc.connect(yacOscGain);
    yacOscGain.connect(eYacht.input);
    yacWake.connect(yacWakeBp);
    yacWakeBp.connect(eYacht.input);
    yacOsc.start();
    yacWake.start();
    eYacht.osc = yacOsc;
    eYacht.wakeBp = yacWakeBp;
    this._emitters.yacht = eYacht;

    // E10. Dedicated Active Coastal Vessel Emitter (SMALL_SHIP Diesel Chug vs FISHING_BOAT Outboard Put-Put)
    const eCoastal = this._create3DEmitter({
      refDistance: 9.5,
      rolloffFactor: 1.08,
      reverbWeight: 0.18,
      coneInnerAngle: 160,
      coneOuterAngle: 310,
      coneOuterGain: 0.72
    });
    const cbOsc = this.ctx.createOscillator();
    cbOsc.type = 'sawtooth';
    cbOsc.frequency.value = 58;
    const cbChugGate = this.ctx.createGain();
    cbChugGate.gain.value = 0.55;
    const cbChugLfo = this.ctx.createOscillator();
    cbChugLfo.type = 'sine';
    cbChugLfo.frequency.value = 8.5;
    const cbChugDepth = this.ctx.createGain();
    cbChugDepth.gain.value = 0.40;
    cbChugLfo.connect(cbChugDepth);
    cbChugDepth.connect(cbChugGate.gain);
    const cbShaper = this._createWarmWaveshaper(2.0);
    const cbLp = this.ctx.createBiquadFilter();
    cbLp.type = 'lowpass';
    cbLp.frequency.value = 390;
    const cbWake = this.ctx.createBufferSource();
    cbWake.buffer = pinkBuffer;
    cbWake.loop = true;
    const cbWakeBp = this.ctx.createBiquadFilter();
    cbWakeBp.type = 'bandpass';
    cbWakeBp.frequency.value = 620;
    cbWakeBp.Q.value = 0.9;
    const cbWakeGain = this.ctx.createGain();
    cbWakeGain.gain.value = 0.34;
    cbOsc.connect(cbChugGate);
    if (cbShaper) {
      cbChugGate.connect(cbShaper);
      cbShaper.connect(cbLp);
    } else {
      cbChugGate.connect(cbLp);
    }
    cbLp.connect(eCoastal.input);
    cbWake.connect(cbWakeBp);
    cbWakeBp.connect(cbWakeGain);
    cbWakeGain.connect(eCoastal.input);
    cbOsc.start();
    cbChugLfo.start();
    cbWake.start();
    eCoastal.osc = cbOsc;
    eCoastal.chugLfo = cbChugLfo;
    eCoastal.lp = cbLp;
    eCoastal.wakeBp = cbWakeBp;
    this._emitters.coastalBoat = eCoastal;

    // E11. Directional Videotron Outdoor Line-Array Speaker & Cabinet Fan Emitter at (28.4, 7.6, 78.2)
    // Outward normal facing Lapangan Rotunda (-2.5, 106.5): yaw = -0.8293 -> (-0.7374, -0.06, +0.6755)
    const eVideotron = this._create3DEmitter({
      refDistance: 8.5,
      rolloffFactor: 1.15,
      reverbWeight: 0.24,
      coneInnerAngle: 78,
      coneOuterAngle: 196,
      coneOuterGain: 0.22
    });
    const vtOscA = this.ctx.createOscillator();
    vtOscA.type = 'sine';
    vtOscA.frequency.value = 196.0;
    const vtOscB = this.ctx.createOscillator();
    vtOscB.type = 'triangle';
    vtOscB.frequency.value = 293.66;
    const vtPadGain = this.ctx.createGain();
    vtPadGain.gain.value = 0.20;
    const vtFanNoise = this.ctx.createBufferSource();
    vtFanNoise.buffer = pinkBuffer;
    vtFanNoise.loop = true;
    const vtFanBp = this.ctx.createBiquadFilter();
    vtFanBp.type = 'bandpass';
    vtFanBp.frequency.value = 440.0;
    vtFanBp.Q.value = 1.4;
    const vtFanGain = this.ctx.createGain();
    vtFanGain.gain.value = 0.32;
    vtOscA.connect(vtPadGain);
    vtOscB.connect(vtPadGain);
    vtPadGain.connect(eVideotron.input);
    vtFanNoise.connect(vtFanBp);
    vtFanBp.connect(vtFanGain);
    vtFanGain.connect(eVideotron.input);
    this._setPannerPosition(eVideotron.panner, 28.4, 7.6, 78.2, this.ctx.currentTime, true);
    this._setPannerOrientation(eVideotron.panner, -0.7374, -0.06, 0.6755, this.ctx.currentTime, true);
    vtOscA.start();
    vtOscB.start();
    vtFanNoise.start();
    eVideotron.oscA = vtOscA;
    eVideotron.oscB = vtOscB;
    eVideotron.padGain = vtPadGain;
    eVideotron.fanGain = vtFanGain;
    this._emitters.videotron = eVideotron;

    // Configure directional radiation cones, Atmos Volumetric Object Sizes (sourceRadiusM), & (.1) LFE Weights
    this._emitters.fountain.sourceRadiusM = 10.5;
    this._emitters.fountain.lfeWeight = 0.14;

    this._emitters.waterfall.sourceRadiusM = 17.5;
    this._emitters.waterfall.lfeWeight = 0.44;

    this._emitters.bus.panner.coneInnerAngle = 135;
    this._emitters.bus.panner.coneOuterAngle = 285;
    this._emitters.bus.panner.coneOuterGain = 0.64;
    this._emitters.bus.sourceRadiusM = 4.2;
    this._emitters.bus.lfeWeight = 0.46;

    this._emitters.cart.panner.coneInnerAngle = 140;
    this._emitters.cart.panner.coneOuterAngle = 290;
    this._emitters.cart.panner.coneOuterGain = 0.68;
    this._emitters.cart.sourceRadiusM = 2.2;
    this._emitters.cart.lfeWeight = 0.08;

    this._emitters.heli.sourceRadiusM = 8.5;
    this._emitters.heli.lfeWeight = 0.54;

    this._emitters.cessna.sourceRadiusM = 5.5;
    this._emitters.cessna.lfeWeight = 0.36;

    this._emitters.learjet.sourceRadiusM = 9.5;
    this._emitters.learjet.lfeWeight = 0.60;

    this._emitters.ferry.sourceRadiusM = 8.5;
    this._emitters.ferry.lfeWeight = 0.48;

    this._emitters.yacht.sourceRadiusM = 5.0;
    this._emitters.yacht.lfeWeight = 0.24;

    this._emitters.coastalBoat.sourceRadiusM = 4.0;
    this._emitters.coastalBoat.lfeWeight = 0.22;

    this._emitters.videotron.sourceRadiusM = 4.5;
    this._emitters.videotron.lfeWeight = 0.18;

    // E12 (v633): Dedicated 3D Conversational Sivitas Partner Voice Object (FPS_CHAT_NPC & Nearby Discussion Circles)
    const eVoice = this._create3DEmitter({
      refDistance: 1.85,
      rolloffFactor: 1.18,
      maxDistance: 65.0,
      reverbWeight: 0.32,
      coneInnerAngle: 120,
      coneOuterAngle: 260,
      coneOuterGain: 0.48,
      sourceRadiusM: 0.35,
      lfeWeight: 0.0
    });
    const vocGlottal = this.ctx.createOscillator();
    vocGlottal.type = 'sawtooth';
    vocGlottal.frequency.value = 152.0;
    const vocSub = this.ctx.createOscillator();
    vocSub.type = 'triangle';
    vocSub.frequency.value = 304.0;
    const vocSubGain = this.ctx.createGain();
    vocSubGain.gain.value = 0.36;
    const vocBreath = this.ctx.createBufferSource();
    vocBreath.buffer = pinkBuffer;
    vocBreath.loop = true;
    const vocBreathGain = this.ctx.createGain();
    vocBreathGain.gain.value = 0.22;

    const vocForm1 = this.ctx.createBiquadFilter();
    vocForm1.type = 'bandpass';
    vocForm1.frequency.value = 540.0;
    vocForm1.Q.value = 4.2;
    const vocForm2 = this.ctx.createBiquadFilter();
    vocForm2.type = 'bandpass';
    vocForm2.frequency.value = 1480.0;
    vocForm2.Q.value = 4.8;
    const vocSyllable = this.ctx.createGain();
    vocSyllable.gain.value = 0.0;

    vocGlottal.connect(vocForm1);
    vocGlottal.connect(vocForm2);
    vocSub.connect(vocSubGain);
    vocSubGain.connect(vocForm1);
    vocBreath.connect(vocBreathGain);
    vocBreathGain.connect(vocForm2);
    vocForm1.connect(vocSyllable);
    vocForm2.connect(vocSyllable);
    vocSyllable.connect(eVoice.input);
    vocGlottal.start();
    vocSub.start();
    vocBreath.start();
    eVoice.glottal = vocGlottal;
    eVoice.sub = vocSub;
    eVoice.form1 = vocForm1;
    eVoice.form2 = vocForm2;
    eVoice.syllable = vocSyllable;
    this._emitters.npcVoice = eVoice;

    // Register all 33 persistent sub-buses for click-free silent hibernation (zero routing/character change when audible!)
    this._hibernationSlots.length = 0;
    this._hibernationSlotByGain.clear();
    const bp = this._bedPanners;
    this._registerHibernationSlot('campus_breeze', 'env', this.breezeGain, [
      [this.breezeGain, bp.breeze],
      [bp.breeze, this.weatherOccFilter]
    ]);
    this._registerHibernationSlot('tropical_rain_roar', 'env', this.rainRoarGain, [
      [this.rainRoarGain, rainFrontSplit],
      [this.rainRoarGain, rainRearSplit],
      [bp.rainFront, this.weatherOccFilter],
      [bp.rainRear, this.weatherOccFilter]
    ]);
    this._registerHibernationSlot('raindrop_patter', 'env', this.rainPatterGain, [
      [this.rainPatterGain, this.worldSumGain]
    ]);
    this._registerHibernationSlot('roof_rain_resonance', 'env', this.roofResonanceGain, [
      [this.roofResonanceGain, this.worldSumGain],
      [this.roofResonanceGain, this.reverbSendGain]
    ]);
    this._registerHibernationSlot('archipelago_surf', 'env', this.surfGain, [
      [this.surfGain, bp.surf],
      [this.surfGain, surfLfeSend],
      [this.surfGain, surfSpreadSend],
      [bp.surf, this.weatherOccFilter]
    ]);
    this._registerHibernationSlot('subsea_tunnel_hum', 'env', this.tunnelHumGain, [
      [this.tunnelHumGain, this.worldSumGain],
      [this.tunnelHumGain, this.reverbSendGain],
      [this.tunnelHumGain, tunLfeSend]
    ]);
    this._registerHibernationSlot('nocturnal_cricket', 'env', this.cricketGain, [
      [this.cricketGain, bp.cricket],
      [bp.cricket, this.ambientDuckGain]
    ]);
    this._registerHibernationSlot(
      'avata2_cinewhoop_motor',
      'env',
      this.avataMotorGain,
      this.avataPan
        ? [
            [this.avataMotorGain, this.avataPan],
            [this.avataPan, this.worldSumGain],
            [this.avataPan, this.reverbSendGain]
          ]
        : [
            [this.avataMotorGain, this.worldSumGain],
            [this.avataMotorGain, this.reverbSendGain]
          ]
    );
    this._registerHibernationSlot('avata2_prop_wash', 'env', this.avataWashGain, [
      [this.avataWashGain, this.worldSumGain]
    ]);
    this._registerHibernationSlot('avata2_prox_whoosh', 'env', this.proxWhooshGain, [
      [this.proxWhooshGain, this.worldSumGain],
      [this.proxWhooshGain, this.reverbSendGain]
    ]);
    this._registerHibernationSlot('mavic3_aerial_hum', 'env', this.mavicGain, [
      [this.mavicGain, this.worldSumGain]
    ]);
    this._registerHibernationSlot('feline_purr_resonator', 'env', this.catPurrGain, [
      [this.catPurrGain, this.worldSumGain],
      [this.catPurrGain, purrLfeSend]
    ]);
    this._registerHibernationSlot('rainforest_canopy_rustle', 'env', this.canopyRustleGain, [
      [this.canopyRustleGain, bp.canopy],
      [bp.canopy, this.ambientDuckGain]
    ]);
    this._registerHibernationSlot('afternoon_tonggeret', 'env', this.tonggeretGain, [
      [this.tonggeretGain, bp.tonggeret],
      [bp.tonggeret, this.ambientDuckGain]
    ]);
    this._registerHibernationSlot('campus_social_murmur', 'env', this.socialMurmurGain, [
      [this.socialMurmurGain, bp.social],
      [bp.social, this.ambientDuckGain],
      [bp.social, this.reverbSendGain]
    ]);
    this._registerHibernationSlot('rotunda_flag_flutter', 'env', this.flagFlutterGain, [
      [this.flagFlutterGain, bp.flag],
      [bp.flag, this.weatherOccFilter]
    ]);
    this._registerHibernationSlot('teksas_truss_aeolian', 'env', this.trussAeolianGain, [
      [this.trussAeolianGain, bp.truss],
      [bp.truss, this.worldSumGain],
      [bp.truss, this.reverbSendGain]
    ]);
    this._registerHibernationSlot('volcano_fumarole_vent', 'env', this.fumaroleGain, [
      [this.fumaroleGain, bp.fumarole],
      [this.fumaroleGain, fumLfeSend],
      [bp.fumarole, this.worldSumGain]
    ]);
    this._registerHibernationSlot('danau_turap_lapping', 'env', this.turapLapGain, [
      [this.turapLapGain, bp.turapLap],
      [bp.turapLap, this.weatherOccFilter]
    ]);
    this._registerHibernationSlot('cockpit_windshield_air', 'env', this.cockpitAirGain, [
      [this.cockpitAirGain, this.cockpitPanner]
    ]);
    this._registerHibernationSlot('cockpit_400hz_avionics', 'env', this.cockpitAvionicsGain, [
      [this.cockpitAvionicsGain, this.cockpitPanner]
    ]);

    const emitterDefs = [
      ['3d_rotunda_fountain', this._emitters.fountain],
      ['3d_niagara_waterfall', this._emitters.waterfall],
      ['3d_bikun_yellow_bus', this._emitters.bus],
      ['3d_golf_cart_ev', this._emitters.cart],
      ['3d_heli_evtol_pkuie', this._emitters.heli],
      ['3d_cessna_172_pkuia', this._emitters.cessna],
      ['3d_citation_learjet_pkuib', this._emitters.learjet],
      ['3d_catamaran_ferry', this._emitters.ferry],
      ['3d_flybridge_yacht', this._emitters.yacht],
      ['3d_coastal_vessel', this._emitters.coastalBoat],
      ['3d_videotron_line_array', this._emitters.videotron],
      ['3d_sivitas_npc_voice', this._emitters.npcVoice]
    ];
    for (let i = 0; i < emitterDefs.length; i++) {
      const [eName, emObj] = emitterDefs[i];
      if (!emObj || !emObj.gain) continue;
      const eLinks = [
        [emObj.gain, emObj.directGain],
        [emObj.panner, this.worldSumGain],
        [emObj.panner, this.reverbSendGain]
      ];
      if (this.atmosSpreadIn && emObj.spreadSend) eLinks.push([emObj.gain, emObj.spreadSend]);
      if (this.heightDomeIn && emObj.heightSend) eLinks.push([emObj.gain, emObj.heightSend]);
      if (this.lfeSumGain && emObj.lfeSend) eLinks.push([emObj.gain, emObj.lfeSend]);
      this._registerHibernationSlot(eName, 'emitter', emObj.gain, eLinks);
    }

    const _tG4 = _nowP();
    this._forensic.initBreakdownMs = {
      audioCtxCreateMs: +(this._forensic._tmpCtxCreateMs || 0).toFixed(2),
      masterAndDualConvolverIrMs: +Math.max(0.0, _tG1 - _tG0).toFixed(2),
      atmos714AndBedPannersMs: +Math.max(0.0, _tG2 - _tG1).toFixed(2),
      pinkAndPatterBuffersMs: +Math.max(0.0, _tG3 - _tG2).toFixed(2),
      worldVoicesAndEmittersMs: +Math.max(0.0, _tG4 - _tG3).toFixed(2)
    };
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
   * Couples Web Audio wind, downpour roar, and roof-material-specific raindrop impact resonance
   * (`TERRACOTTA_CLAY`, `POLYCARBONATE_TRUSS`, `CABIN_WINDSHIELD`, `CANOPY_FABRIC`, `OPEN_FOLIAGE_STONE`)
   * continuously 1:1 to the live 20–45s meteorological transition state + listener altitude & shelter!
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

    const altY = Math.max(0.0, this._lis.y);
    const altWindBoost = clamp(altY / 85.0, 0.0, 0.45);
    const altPatterAtten = 1.0 / (1.0 + Math.max(0.0, altY - 12.0) * 0.045);
    const patBoost = this._zoneState ? this._zoneState.patterBoost : 1.0;
    const roofMat = this._zoneState ? this._zoneState.roofMaterial : 'OPEN_FOLIAGE_STONE';

    const roar = Math.min(1.26, Math.pow(rain / 1.42, 1.15) * 1.24);
    const basePatter =
      Math.min(0.96, Math.pow(Math.min(1.0, rain * 1.05), 0.85) * (0.70 + 0.22 * storm)) *
      altPatterAtten *
      patBoost;
    const windVol = 0.40 + wNorm * 0.52 + storm * 0.22 + altWindBoost * 0.28;
    const windFreq = 375 + wNorm * 170 + storm * 95 + altWindBoost * 160;

    // Roof-material acoustic timbre split (Crisp HF splash vs resonant structural roof drum)
    let crispHpHz = 2250.0;
    let crispShare = 0.82;
    let resHz = 1850.0;
    let resQ = 0.9;
    let resShare = 0.18;

    if (roofMat === 'TERRACOTTA_CLAY') {
      // Warm Nusantara terracotta clay roof tiles + eave drip in Selasar Agung & Walkways
      crispHpHz = 1850.0;
      crispShare = 0.52;
      resHz = 760.0;
      resQ = 2.15;
      resShare = 0.68;
    } else if (roofMat === 'POLYCARBONATE_TRUSS') {
      // Hollow polycarbonate sunroof & steel arch drum inside Lorong Jembatan Teksas UI
      crispHpHz = 1650.0;
      crispShare = 0.48;
      resHz = 1480.0;
      resQ = 2.75;
      resShare = 0.84;
    } else if (roofMat === 'CABIN_WINDSHIELD') {
      // Laminated glass windshield & damped metal roof inside Bis Kuning / Aircraft Cockpit
      crispHpHz = 1280.0;
      crispShare = 0.26;
      resHz = 430.0;
      resQ = 1.65;
      resShare = 0.72;
    } else if (roofMat === 'CANOPY_FABRIC') {
      // Taut fiberglass/canvas canopy on Mobil Golf Listrik UI
      crispHpHz = 1720.0;
      crispShare = 0.46;
      resHz = 960.0;
      resQ = 1.95;
      resShare = 0.70;
    } else if (roofMat === 'CONCRETE_VAULT') {
      crispShare = 0.0;
      resShare = 0.0;
    }

    const natM = this._mixWeights ? this._mixWeights.natureMult : 1.0;
    this.rainRoarGain.gain.setTargetAtTime(roar * natM, now, 0.16);
    this.rainPatterFilter.frequency.setTargetAtTime(crispHpHz, now, 0.16);
    this.rainPatterGain.gain.setTargetAtTime(basePatter * crispShare * natM, now, 0.16);
    if (this.roofResonanceFilter && this.roofResonanceGain) {
      this.roofResonanceFilter.frequency.setTargetAtTime(resHz, now, 0.14);
      this.roofResonanceFilter.Q.setTargetAtTime(resQ, now, 0.14);
      this.roofResonanceGain.gain.setTargetAtTime(
        this.enabled ? basePatter * resShare * natM : 0.0,
        now,
        0.16
      );
    }
    this.breezeGain.gain.setTargetAtTime(windVol * natM, now, 0.20);
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
    obstacleClearanceM = 99.0,
    rollRad = 0.0,
    signedYawRate = 0.0
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

    const trnM = this._mixWeights ? this._mixWeights.transitMult : 1.0;
    const baseHz = 212 + thr * 278 + (spd / 27.0) * 62 + gLoad * 28;
    const asymDetune = 1.0 + turn * 0.032 + gLoad * 0.018;
    this.avataMotorOscA.frequency.setTargetAtTime(baseHz, now, 0.05);
    this.avataMotorOscB.frequency.setTargetAtTime(baseHz * 2.008 * asymDetune, now, 0.05);
    if (this.avataMotorOscC) {
      this.avataMotorOscC.frequency.setTargetAtTime(baseHz * 1.498 * (2.0 - asymDetune * 0.98), now, 0.05);
    }
    this.avataMotorFilter.frequency.setTargetAtTime(450 + thr * 680 + gLoad * 180, now, 0.07);
    if (this.avataDuctFilter && this.avataDuctGain) {
      this.avataDuctFilter.frequency.setTargetAtTime(1060 + thr * 340 + gLoad * 95, now, 0.06);
      this.avataDuctGain.gain.setTargetAtTime(0.14 + thr * 0.24 + gLoad * 0.08, now, 0.07);
    }
    if (this.avataPan) {
      const bankPan = clamp(-rollRad * 0.52 + signedYawRate * 0.12, -0.55, 0.55);
      this.avataPan.pan.setTargetAtTime(bankPan, now, 0.06);
    }
    this.avataMotorGain.gain.setTargetAtTime((0.050 + thr * 0.105 + gLoad * 0.022) * trnM, now, 0.07);

    const washGain = (0.036 + (spd / 27.0) * 0.175 + turn * 0.042) * trnM;
    this.avataWashFilter.frequency.setTargetAtTime(600 + (spd / 27.0) * 740 + turn * 145, now, 0.07);
    this.avataWashGain.gain.setTargetAtTime(Math.min(0.32, washGain), now, 0.07);

    if (this.proxWhooshGain && this.proxWhooshFilter) {
      const proxFactor =
        spd > 5.0 && obstacleClearanceM < 6.8
          ? smooth01(6.8, 1.4, obstacleClearanceM) * clamp(spd / 20.0, 0.25, 1.15)
          : 0.0;
      this.proxWhooshFilter.frequency.setTargetAtTime(520 + proxFactor * 1150, now, 0.05);
      this.proxWhooshGain.gain.setTargetAtTime(Math.min(0.30, proxFactor * 0.24 * trnM), now, 0.05);
    }
  }

  /**
   * 3D Binaural HRTF Rotunda Dancing Fountain, Niagara Waterfall, Pulau U River, & Shoreline Surf
   * with 3D Geometric Obstacle Occlusion & Temperature/Humidity Air Absorption.
   */
  syncSpatialWaterAudio(
    camX = 0.0,
    camY = 20.0,
    camZ = 100.0,
    nightShowBlend = 0.0,
    coastDist = -40.0,
    riverZ = null,
    seaWaveHeightM = 0.28
  ) {
    if (!this.ctx || !this._emitters) return;
    const now = this.ctx.currentTime;
    if (!this.enabled) {
      this.fountainGain.gain.setTargetAtTime(0.0, now, 0.14);
      this.waterfallGain.gain.setTargetAtTime(0.0, now, 0.14);
      if (this.surfGain) this.surfGain.gain.setTargetAtTime(0.0, now, 0.14);
      return;
    }

    const tempC = this._weatherState.tempC;
    const rh = this._weatherState.humidityPct;

    // 1. 3D HRTF Rotunda Dancing Fountain at (-2.5, 0.9, 106.5) + Tower/Tunnel Occlusion
    const eFount = this._emitters.fountain;
    const fountMet = AudioSynthesizer.computeDopplerAndAbsorption(
      -2.5, 0.9, 106.5, 0, 0, 0,
      camX, camY, camZ, this._lis.vx, this._lis.vy, this._lis.vz,
      tempC, rh
    );
    const fountOcc = AudioSynthesizer.computeGeometricOcclusion(
      -2.5, 0.9, 106.5, camX, camY, camZ, this._occScratch
    );
    const natM = this._mixWeights ? this._mixWeights.natureMult : 1.0;
    const showBoost = 1.0 + clamp(nightShowBlend, 0.0, 1.0) * 0.38;
    const fountProx = clamp(1.0 - fountMet.dist / 92.0, 0.0, 1.0);
    this._applyAtmosCueToEmitter(eFount, -2.5, 0.9, 106.5, now, 0.12);
    eFount.airLP.frequency.setTargetAtTime(Math.max(220.0, fountMet.airCutoffHz * fountOcc.cutoffMult), now, 0.12);
    eFount.bp.frequency.setTargetAtTime((1650 + 380 * nightShowBlend) * clamp(fountMet.doppler, 0.88, 1.14), now, 0.12);
    eFount.gain.gain.setTargetAtTime(fountProx * fountProx * 0.56 * showBoost * fountOcc.gainMult * natM, now, 0.14);

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

    const useRiver = riverProx * 0.55 > niagProx;
    if (useRiver) {
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
      camX, camY, camZ, this._lis.vx, this._lis.vy, this._lis.vz,
      tempC, rh
    );
    const wfOcc = AudioSynthesizer.computeGeometricOcclusion(
      wfX, wfY, wfZ, camX, camY, camZ, this._occScratch
    );
    eWf.sourceRadiusM = useRiver ? 6.5 : 16.5;
    eWf.lfeWeight = useRiver ? 0.08 : 0.42;
    this._setPannerPosition(eWf.panner, wfX, wfY, wfZ, now, false);
    this._applyAtmosCueToEmitter(eWf, wfX, wfY, wfZ, now, 0.14);
    eWf.airLP.frequency.setTargetAtTime(Math.max(220.0, wfMet.airCutoffHz * wfOcc.cutoffMult), now, 0.14);
    eWf.bp.frequency.setTargetAtTime(wfCenterHz * clamp(wfMet.doppler, 0.88, 1.14), now, 0.14);
    eWf.gain.gain.setTargetAtTime(wfLevel * wfOcc.gainMult * natM, now, 0.16);

    // 3. Rhythmic Coastal Wave Surf around Pulau U, Pulau I, Teluk U & Selat UI (3D anchored to true coastline!)
    if (this.surfGain && this.surfFilter) {
      const waveM = clamp(Number.isFinite(seaWaveHeightM) ? seaWaveHeightM : 0.28, 0.0, 3.0);
      const waveDrive = 0.04 + 2.35 * Math.pow(waveM / 3.0, 0.84);
      const shoreNear = Math.abs(coastDist) < 62.0 ? 1.0 - Math.abs(coastDist) / 62.0 : 0.0;
      const openWaterLow = coastDist > 0.0 && camY > -1.8 && camY < 32.0 ? (32.0 - camY) / 32.0 : 0.0;
      const altFade = camY > -1.8 ? clamp(1.0 - Math.max(0.0, camY - 6.5) / 44.0, 0.0, 1.0) : 0.0;
      const waveSurge = 0.66 + 0.34 * Math.sin(this._simClock * 0.86) * Math.cos(this._simClock * 0.35 + 0.8);
      const surfVol =
        Math.min(0.56, Math.max(shoreNear * 0.28, openWaterLow * 0.24) * altFade * waveSurge * waveDrive) * natM;
      const surfCutoff = (240 + (1.0 - waveM / 3.6) * 145) + waveSurge * (210 + waveM * 105);
      this.surfFilter.frequency.setTargetAtTime(surfCutoff, now, 0.18);
      this.surfGain.gain.setTargetAtTime(surfVol, now, 0.18);
      if (this._bedPanners && this._bedPanners.surf && this._atmosAnchors) {
        const seaAnc = this._atmosAnchors.sea;
        this._setPannerPosition(this._bedPanners.surf, seaAnc.x, seaAnc.y, seaAnc.z, now, false);
      }
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
   */
  updateAcousticWorld(dt, state = {}) {
    const _tUpd0 = typeof performance !== 'undefined' && performance.now ? performance.now() : 0.0;
    const safeDt = clamp(Number(dt) || 0.016, 0.001, 0.1);
    const simPaused = Boolean(state && state.simPaused);
    this._forensic.lastSimPaused = simPaused;

    try {
      this._updateAcousticWorldCore(safeDt, state);
    } catch (err) {
      this.recordExternalError(err, 'updateAcousticWorld');
    }

    const _tUpd1 = typeof performance !== 'undefined' && performance.now ? performance.now() : _tUpd0;
    const _updMs = Math.max(0.0, _tUpd1 - _tUpd0);
    const fp = this._forensic.perf;
    fp.updateCalls++;
    fp.emaUpdateMs = fp.updateCalls === 1 ? _updMs : fp.emaUpdateMs * 0.92 + _updMs * 0.08;
    if (_updMs > fp.maxUpdateMs) fp.maxUpdateMs = _updMs;
    if (_updMs > 1.5) fp.overBudgetFrames++;

    const ctxNowSec = this.ctx ? Number(this.ctx.currentTime) || 0.0 : 0.0;
    if (this.ctx) {
      this._tickSubBusHibernation(safeDt, ctxNowSec);
    }
    this._samplePassiveClockAndLimiter(safeDt, ctxNowSec, _tUpd1, simPaused);
  }

  _updateAcousticWorldCore(safeDt, state = {}) {
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
      // Camera right-axis Y component e[1] vs up-axis Y component e[5] yields roll tilt in radians
      this._lis.rollRad = Math.atan2(Number(e[1]) || 0.0, Math.max(0.15, Math.abs(Number(e[5]) || 1.0)));
    } else if (Number.isFinite(state.fwdX) && Number.isFinite(state.fwdZ)) {
      let fx = Number(state.fwdX);
      let fy = Number.isFinite(state.fwdY) ? Number(state.fwdY) : 0.0;
      let fz = Number(state.fwdZ);
      const fLen = Math.hypot(fx, fy, fz) || 1.0;
      fx /= fLen;
      fy /= fLen;
      fz /= fLen;
      let ux = Number.isFinite(state.upX) ? Number(state.upX) : 0.0;
      let uy = Number.isFinite(state.upY) ? Number(state.upY) : 1.0;
      let uz = Number.isFinite(state.upZ) ? Number(state.upZ) : 0.0;
      const dotFU = fx * ux + fy * uy + fz * uz;
      ux -= dotFU * fx;
      uy -= dotFU * fy;
      uz -= dotFU * fz;
      const uLen = Math.hypot(ux, uy, uz);
      this._lis.fwdX = fx;
      this._lis.fwdY = fy;
      this._lis.fwdZ = fz;
      this._lis.upX = uLen > 1e-4 ? ux / uLen : 0.0;
      this._lis.upY = uLen > 1e-4 ? uy / uLen : 1.0;
      this._lis.upZ = uLen > 1e-4 ? uz / uLen : 0.0;
      this._lis.rollRad = Number(state.rollRad) || 0.0;
    }

    // 2. Resolve Point-of-View Perspective Profile first so enclosed/open fleet cabin zone is exact
    const camMode = state.camMode || 'DRONE_6DOF';
    const isRoofSheltered = Boolean(state.isRoofSheltered);
    const sivitasAI = state.sivitasAI || null;
    AudioSynthesizer.resolveViewPerspectiveAcoustics(
      camMode,
      camX,
      camY,
      camZ,
      sivitasAI,
      state,
      this._viewPerspective
    );
    this._telemetry.perspectiveId = this._viewPerspective.perspectiveId;
    this._telemetry.perspectiveLabel = this._viewPerspective.perspectiveLabel;

    // Classify Architectural Space, Dual-IR Reverb Split, & Roof Material
    const resolvedFpsKind =
      state.fpsSubjectKind ||
      (camMode === 'FPS_PILOT' ? this._viewPerspective.coMovingEntity || '' : '');
    const zone = AudioSynthesizer.classifyAcousticZone(
      camX,
      camY,
      camZ,
      camMode,
      isRoofSheltered,
      resolvedFpsKind
    );
    this._zoneState = zone;
    this._telemetry.zoneId = zone.zoneId;
    this._telemetry.zoneLabel = zone.labelId;
    this._telemetry.zoneLabelId = zone.labelId;
    this._telemetry.reverbCharacter = zone.reverbCharacter;
    this._telemetry.roofMaterial = zone.roofMaterial;
    this._telemetry.reverbSendPct = Math.round(zone.reverbSend * 100);
    this._telemetry.weatherOcclusionPct = Math.round((1.0 - zone.weatherOcclusionGain) * 100);

    // Cache meteorological & astronomical state (including temperature & relative humidity coupling)
    const rainInt = clamp(Number(state.rainIntensity) || 0.0, 0.0, 1.5);
    const stormFac = clamp(Number(state.stormFactor) || 0.0, 0.0, 1.0);
    const windSpd = Math.max(0.0, Number(state.windSpeed) || 0.85);
    const wetness = clamp(Number(state.wetness) || 0.0, 0.0, 1.0);
    const nightFac = clamp(Number(state.nightFactor) || 0.0, 0.0, 1.0);
    this._weatherState.wetness = wetness;
    this._weatherState.nightFactor = nightFac;
    this._weatherState.tempC = Number.isFinite(state.temperatureC)
      ? clamp(state.temperatureC, 18.0, 38.0)
      : 30.5 - nightFac * 5.2 - rainInt * 2.4;
    this._weatherState.humidityPct = Number.isFinite(state.humidityPct)
      ? clamp(state.humidityPct, 35.0, 100.0)
      : clamp(68.0 + wetness * 24.0 + rainInt * 10.0, 45.0, 99.0);
    if (state.lightningDir && Number.isFinite(state.lightningDir.x)) {
      this._weatherState.lightningDirX = state.lightningDir.x;
      this._weatherState.lightningDirY = state.lightningDir.y;
      this._weatherState.lightningDirZ = state.lightningDir.z;
    }

    // Classify 6-Phase Diurnal Bio-Acoustic Cycle & 16 Location-Context Micro-Biomes
    const inferredSolarHour = Number.isFinite(state.solarHour)
      ? state.solarHour
      : nightFac > 0.62
        ? 22.0
        : nightFac > 0.24
          ? 17.85
          : 11.85;
    const diurnal = AudioSynthesizer.classifyDiurnalBioPhase(inferredSolarHour);
    const coastDistVal = Number.isFinite(state.coastDist) ? state.coastDist : -40.0;
    const biome = AudioSynthesizer.classifyMicroContextBiome(camX, camY, camZ, {
      coastDist: coastDistVal,
      riverZ: state.riverZ ?? null
    });
    this._diurnalState = diurnal;
    this._biomeState = biome;
    this._telemetry.diurnalPhase = diurnal.phaseId;
    this._telemetry.diurnalLabel = diurnal.labelId;
    this._telemetry.microBiomeId = biome.biomeId;
    this._telemetry.microBiomeLabel = biome.labelId;

    // Resolve surface material for telemetry & footsteps
    const surfMat = this._resolveSurfaceMaterial(camX, camY, camZ, camMode);
    this._telemetry.surfaceMaterial = surfMat;

    AudioSynthesizer.computeWorldEnvironmentAnchors(
      camX,
      camY,
      camZ,
      {
        coastDist: coastDistVal,
        riverZ: state.riverZ ?? null,
        seaDirX: state.seaDirX,
        seaDirZ: state.seaDirZ,
        windDirX: state.windDirX,
        windDirZ: state.windDirZ,
        groundY: state.groundY,
        fwdX: this._lis.fwdX,
        fwdZ: this._lis.fwdZ
      },
      this._atmosAnchors
    );

    // Update 3D Vehicle/Aircraft/Videotron/NPC Voice Kinematics, State-Transition SFX, Occlusion, & Live Doppler
    if (this._sfxTracker.sfxBannerTimer > 0.0) {
      this._sfxTracker.sfxBannerTimer = Math.max(0.0, this._sfxTracker.sfxBannerTimer - safeDt);
      if (this._sfxTracker.sfxBannerTimer <= 0.0) {
        this._telemetry.activeSfxBanner = '';
      }
    }
    this._updateDynamicFleetAcoustics(safeDt, sivitasAI, state);
    this._updateWorldEventSfx(safeDt, state, sivitasAI, camMode, rainInt);
    this._updateAtmosStageSnapshot(state, sivitasAI, camMode, rainInt, windSpd);

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

    // Anchor 3D Hemispherical Environmental Beds & .4 Overhead Height Dome in True World Space
    const anc = this._atmosAnchors;
    if (this.heightFrontPanner && this.heightRearPanner && anc) {
      this._setPannerPosition(this.heightFrontPanner, anc.skyFront.x, anc.skyFront.y, anc.skyFront.z, now, teleported);
      this._setPannerPosition(this.heightRearPanner, anc.skyRear.x, anc.skyRear.y, anc.skyRear.z, now, teleported);
    }
    if (this._bedPanners && anc) {
      const bp = this._bedPanners;
      this._setPannerPosition(bp.breeze, anc.wind.x, anc.wind.y, anc.wind.z, now, teleported);
      this._setPannerPosition(bp.rainFront, anc.skyFront.x, anc.skyFront.y - 2.6, anc.skyFront.z, now, teleported);
      this._setPannerPosition(bp.rainRear, anc.skyRear.x, anc.skyRear.y - 2.6, anc.skyRear.z, now, teleported);
      this._setPannerPosition(bp.surf, anc.sea.x, anc.sea.y, anc.sea.z, now, teleported);
      this._setPannerPosition(bp.cricket, anc.cricket.x, anc.cricket.y, anc.cricket.z, now, teleported);
      this._setPannerPosition(bp.canopy, anc.canopy.x, anc.canopy.y, anc.canopy.z, now, teleported);
      this._setPannerPosition(bp.tonggeret, anc.tonggeret.x, anc.tonggeret.y, anc.tonggeret.z, now, teleported);
      this._setPannerPosition(bp.social, anc.social.x, anc.social.y, anc.social.z, now, teleported);
      this._setPannerPosition(bp.flag, anc.flag.x, anc.flag.y, anc.flag.z, now, teleported);
      this._setPannerPosition(bp.truss, anc.truss.x, anc.truss.y, anc.truss.z, now, teleported);
      this._setPannerPosition(bp.fumarole, anc.fumarole.x, anc.fumarole.y, anc.fumarole.z, now, teleported);
      this._setPannerPosition(bp.turapLap, anc.lap.x, anc.lap.y, anc.lap.z, now, teleported);
    }

    // Apply Dual-Character Convolution Reverb Crossfade, .4 Vault Ceiling Reflections, & Tunnel Hum
    this.reverbSendGain.gain.setTargetAtTime(1.0, now, 0.08);
    if (this.cathedralSendGain) {
      this.cathedralSendGain.gain.setTargetAtTime(zone.cathedralSend, now, 0.14);
    }
    if (this.metallicSendGain) {
      this.metallicSendGain.gain.setTargetAtTime(zone.metallicSend, now, 0.12);
    }
    if (this.heightRevGain) {
      const ceilingRefl = clamp(0.18 + (zone.cathedralSend + zone.metallicSend) * 0.42, 0.18, 0.62);
      this.heightRevGain.gain.setTargetAtTime(this.enabled ? ceilingRefl : 0.0, now, 0.14);
    }
    if (this.lfeMasterGain) {
      this.lfeMasterGain.gain.setTargetAtTime(this.enabled ? 0.92 : 0.0, now, 0.12);
    }
    this.weatherOccGain.gain.setTargetAtTime(zone.weatherOcclusionGain, now, 0.16);
    this.weatherOccFilter.frequency.setTargetAtTime(zone.weatherLowpassHz, now, 0.16);
    if (this.tunnelHumGain) {
      this.tunnelHumGain.gain.setTargetAtTime(this.enabled ? zone.tunnelHum * 0.24 : 0.0, now, 0.18);
    }

    // Sync Weather, Spatial Water, Surf, & Dual-Drone Propulsion
    this.syncContinuousWeatherAudio(rainInt, stormFac, windSpd);
    this.syncSpatialWaterAudio(
      camX,
      camY,
      camZ,
      Number(state.fountainNightBlend) || 0.0,
      coastDistVal,
      state.riverZ ?? null,
      Number.isFinite(state.seaWaveHeightM) ? state.seaWaveHeightM : 0.28
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
        Number.isFinite(state.obstacleClearanceM) ? state.obstacleClearanceM : 99.0,
        Number.isFinite(state.avataRollRad) ? state.avataRollRad : this._lis.rollRad,
        Number(state.avataSignedYawRate) || 0.0
      );
    } else {
      this.syncAvataDroneAudio(false, 0, 0, 0, 1.0, 99.0, 0.0, 0.0);
    }

    if (this.mavicGain) {
      if (isMavic && this.enabled) {
        const trnM = this._mixWeights ? this._mixWeights.transitMult : 1.0;
        const mavSpd = clamp(this._lis.speedMs / 18.0, 0.0, 1.2);
        const mavHz = 136.0 + mavSpd * 48.0;
        this.mavicOscA.frequency.setTargetAtTime(mavHz, now, 0.09);
        this.mavicOscB.frequency.setTargetAtTime(mavHz * 2.006, now, 0.09);
        this.mavicFilter.frequency.setTargetAtTime(340 + mavSpd * 280, now, 0.10);
        this.mavicGain.gain.setTargetAtTime((0.022 + mavSpd * 0.028) * trnM, now, 0.12);
      } else {
        this.mavicGain.gain.setTargetAtTime(0.0, now, 0.10);
      }
    }

    // Bus 16: Pilot Cockpit Windshield Slipstream & 400 Hz Avionics Bus (Aircraft/eVTOL cockpits in FPS_PILOT only!)
    if (this.cockpitAirGain && this.cockpitAvionicsGain && this.cockpitPanner) {
      const coMove = this._viewPerspective.coMovingEntity;
      const isAeroCockpit =
        camMode === 'FPS_PILOT' && (coMove === 'learjet' || coMove === 'cessna' || coMove === 'heli');
      if (isAeroCockpit && this.enabled) {
        const trnM = this._mixWeights ? this._mixWeights.transitMult : 1.0;
        const acSpd =
          coMove === 'learjet'
            ? Number(sivitasAI?.learjetAircraft?.speed) || this._lis.speedMs
            : coMove === 'heli'
              ? Number(sivitasAI?.heliTaxi?.speed) || this._lis.speedMs
              : Number(sivitasAI?.cessnaAircraft?.speed) || this._lis.speedMs;
        const refSpd = coMove === 'learjet' ? 58.0 : coMove === 'heli' ? 18.0 : 38.0;
        const ias01 = clamp(acSpd / refSpd, 0.0, 1.25);
        this._setPannerPosition(
          this.cockpitPanner,
          camX + this._lis.fwdX * 1.15,
          camY + this._lis.fwdY * 1.15 + 0.08,
          camZ + this._lis.fwdZ * 1.15,
          now,
          true
        );
        if (this.cockpitAirFilter) {
          this.cockpitAirFilter.frequency.setTargetAtTime(480.0 + ias01 * 920.0, now, 0.08);
        }
        this.cockpitAirGain.gain.setTargetAtTime(ias01 * 0.135 * trnM, now, 0.09);
        this.cockpitAvionicsGain.gain.setTargetAtTime(0.032 * trnM, now, 0.12);
      } else {
        this.cockpitAirGain.gain.setTargetAtTime(0.0, now, 0.09);
        this.cockpitAvionicsGain.gain.setTargetAtTime(0.0, now, 0.09);
      }
    }

    // Smart Priority Sidechain Ducking across Outdoor Bio/Canopy Beds
    if (this._thunderDuckTimer > 0.0) {
      this._thunderDuckTimer = Math.max(0.0, this._thunderDuckTimer - safeDt);
    }
    if (this.ambientDuckGain) {
      const thunderDuck = this._thunderDuckTimer > 0.0 ? 0.48 * clamp(this._thunderDuckTimer / 1.8, 0.0, 1.0) : 0.0;
      const fgDuck = clamp(this._foregroundPressure01 * 0.42 + thunderDuck, 0.0, 0.62);
      this.ambientDuckGain.gain.setTargetAtTime(1.0 - fgDuck, now, 0.16);
    }

    // Nocturnal Cricket Chorus + Iterasi 2 Location-Context Micro-Biome Ambience Buses
    // (Physically attenuated inside enclosed pressurized/glazed cabins & subsea tunnel!)
    const cabinBioInsul =
      zone.zoneId === 'SUBSEA_TUNNEL'
        ? 0.0
        : zone.zoneId === 'COCKPIT_INTERIOR'
          ? this._viewPerspective.coMovingEntity === 'learjet'
            ? 0.12
            : 0.22
          : 1.0;
    const natM = (this._mixWeights ? this._mixWeights.natureMult : 1.0) * cabinBioInsul;
    const rainBioDuck = clamp(1.0 - rainInt * 0.85, 0.10, 1.0);
    const altBioDuck = clamp(1.0 - Math.max(0.0, camY - 8.0) / 34.0, 0.0, 1.0);
    const windNormBio = clamp((windSpd - 0.5) / 3.2, 0.08, 1.5);

    if (this.cricketGain) {
      const crikLevel =
        Math.max(smooth01(0.28, 0.65, nightFac), diurnal.cricketWeight) *
        clamp(0.35 + 0.65 * biome.canopyWeight, 0.0, 1.0) *
        rainBioDuck *
        altBioDuck;
      this.cricketGain.gain.setTargetAtTime(this.enabled ? crikLevel * 0.018 * natM : 0.0, now, 0.25);
    }

    if (this.canopyRustleGain && this.canopyRustleFilter) {
      const waveGust = 0.72 + 0.28 * Math.sin(this._simClock * 0.64) * Math.cos(this._simClock * 0.27);
      const canopyLvl = biome.canopyWeight * (0.28 + 0.72 * windNormBio) * waveGust;
      this.canopyRustleFilter.frequency.setTargetAtTime(1750 + windNormBio * 760 * waveGust, now, 0.22);
      this.canopyRustleGain.gain.setTargetAtTime(this.enabled ? canopyLvl * 0.13 * natM : 0.0, now, 0.24);
    }

    if (this.tonggeretGain) {
      const phraseSwell = 0.65 + 0.35 * Math.sin(this._simClock * 0.41 + 1.1);
      const tongLvl =
        diurnal.tonggeretWeight *
        smooth01(0.18, 0.65, biome.canopyWeight) *
        rainBioDuck *
        altBioDuck *
        phraseSwell;
      this.tonggeretGain.gain.setTargetAtTime(this.enabled ? tongLvl * 0.014 * natM : 0.0, now, 0.30);
    }

    if (this.socialMurmurGain) {
      const dryCampus = clamp(1.0 - rainInt * 1.35, 0.0, 1.0);
      const socLvl = biome.socialWeight * diurnal.socialWeight * dryCampus;
      this.socialMurmurGain.gain.setTargetAtTime(this.enabled ? socLvl * 0.085 * natM : 0.0, now, 0.28);
    }

    if (this.flagFlutterGain && this.flagFlutterFilter) {
      const snapMod = 0.70 + 0.30 * Math.sin(this._simClock * 2.9);
      const flagLvl = biome.flagWeight * (0.30 + 0.85 * windNormBio) * snapMod;
      this.flagFlutterFilter.frequency.setTargetAtTime(280 + windNormBio * 210, now, 0.16);
      this.flagFlutterGain.gain.setTargetAtTime(this.enabled ? flagLvl * 0.16 * natM : 0.0, now, 0.20);
    }

    if (this.trussAeolianGain) {
      const aeoLvl = biome.trussAeolianWeight * clamp((windSpd - 0.6) / 3.4, 0.05, 1.2);
      this.trussAeolianGain.gain.setTargetAtTime(this.enabled ? aeoLvl * 0.032 * natM : 0.0, now, 0.24);
    }

    if (this.fumaroleGain) {
      const ventPulse = 0.78 + 0.22 * Math.sin(this._simClock * 0.92);
      this.fumaroleGain.gain.setTargetAtTime(
        this.enabled ? biome.fumaroleWeight * ventPulse * 0.19 * natM : 0.0,
        now,
        0.24
      );
    }

    if (this.turapLapGain && this.turapLapFilter) {
      const lapCycle = 0.62 + 0.38 * Math.sin(this._simClock * 1.45);
      this.turapLapFilter.frequency.setTargetAtTime(640 + lapCycle * 280, now, 0.18);
      this.turapLapGain.gain.setTargetAtTime(
        this.enabled ? biome.turapLapWeight * lapCycle * 0.11 * natM : 0.0,
        now,
        0.22
      );
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

  _samplePassiveClockAndLimiter(safeDt, ctxNowSec, perfNowMs, simPaused = false) {
    const f = this._forensic;
    const ck = f.clock;
    ck.accumSec += safeDt;
    if (ck.accumSec < 1.0) return;
    ck.accumSec = 0.0;

    // 1. Passively sample native Web Audio DynamicsCompressorNode gain reduction (dB, <= 0)
    const comp = this.masterCompressor;
    if (comp && typeof comp.reduction === 'number' && Number.isFinite(comp.reduction)) {
      const redDb = comp.reduction;
      f.limiter.currentReductionDb = +redDb.toFixed(2);
      if (redDb < f.limiter.peakReductionDb) {
        f.limiter.peakReductionDb = +redDb.toFixed(2);
      }
      if (redDb < -6.0) {
        f.limiter.heavyLimitingEvents++;
      }
    }

    // 2. Compare AudioContext hardware render clock vs wall-clock ONLY when actively running & tab visible
    const isRunning = Boolean(this.ctx && this.ctx.state === 'running' && this.enabled && !simPaused);
    const isTabVisible = typeof document === 'undefined' || !document.hidden;
    if (!isRunning || !isTabVisible || perfNowMs <= 0.0) {
      ck.prevPerfMs = 0.0;
      ck.prevCtxSec = -1.0;
      return;
    }

    if (ck.prevPerfMs > 0.0 && ck.prevCtxSec >= 0.0) {
      const wallDtSec = (perfNowMs - ck.prevPerfMs) * 0.001;
      const ctxDtSec = ctxNowSec - ck.prevCtxSec;
      if (wallDtSec >= 0.75 && wallDtSec <= 2.2 && ctxDtSec >= 0.0) {
        ck.checksCount++;
        const lagMs = (wallDtSec - ctxDtSec) * 1000.0;
        ck.lastDriftMs = +lagMs.toFixed(2);
        if (lagMs > ck.maxLagMs) {
          ck.maxLagMs = +lagMs.toFixed(2);
        }
        // >24ms audio render-clock stall over 1s foreground playback indicates OS audio callback starvation / underrun
        if (lagMs > 24.0) {
          ck.underrunSuspectCount++;
          ck.lastUnderrunLagMs = +lagMs.toFixed(2);
        }
      }
    }
    ck.prevPerfMs = perfNowMs;
    ck.prevCtxSec = ctxNowSec;
  }

  recordExternalError(err, where = 'audio_runtime') {
    const fp = this._forensic.perf;
    fp.updateErrorsCount++;
    const msg = String(err && err.message ? err.message : err).slice(0, 220);
    fp.lastUpdateError = `${where}: ${msg}`;
  }

  resetRollingForensics() {
    const f = this._forensic;
    f.perf.updateCalls = 0;
    f.perf.emaUpdateMs = 0.0;
    f.perf.maxUpdateMs = 0.0;
    f.perf.overBudgetFrames = 0;
    f.perf.updateErrorsCount = 0;
    f.perf.lastUpdateError = '';
    f.clock.accumSec = 0.0;
    f.clock.prevPerfMs = 0.0;
    f.clock.prevCtxSec = -1.0;
    f.clock.checksCount = 0;
    f.clock.lastDriftMs = 0.0;
    f.clock.maxLagMs = 0.0;
    f.clock.underrunSuspectCount = 0;
    f.clock.lastUnderrunLagMs = 0.0;
    f.limiter.peakReductionDb = f.limiter.currentReductionDb || 0.0;
    f.limiter.heavyLimitingEvents = 0;
  }

  getForensicSnapshot(hwGovernor = null) {
    const f = this._forensic;
    const ns = f.nodes;
    const ctx = this.ctx;
    const hostObj =
      typeof window !== 'undefined' ? window : typeof globalThis !== 'undefined' ? globalThis : null;
    const webAudioSupported = Boolean(hostObj && (hostObj.AudioContext || hostObj.webkitAudioContext));
    const initialized = Boolean(ctx);
    const ctxState = ctx && ctx.state ? String(ctx.state) : 'uninitialized';
    const muted = !this.enabled;

    let baseLatencyMs = null;
    let outputLatencyMs = null;
    let outputTimestampGapMs = null;
    let maxChannelCount = null;
    let channelCount = null;
    let channelInterp = null;

    if (ctx) {
      if (typeof ctx.baseLatency === 'number' && Number.isFinite(ctx.baseLatency)) {
        baseLatencyMs = +(ctx.baseLatency * 1000.0).toFixed(2);
      }
      if (typeof ctx.outputLatency === 'number' && Number.isFinite(ctx.outputLatency)) {
        outputLatencyMs = +(ctx.outputLatency * 1000.0).toFixed(2);
      }
      if (typeof ctx.getOutputTimestamp === 'function' && ctxState === 'running') {
        try {
          const ts = ctx.getOutputTimestamp();
          const pNow = typeof performance !== 'undefined' && performance.now ? performance.now() : 0;
          if (ts && typeof ts.performanceTime === 'number' && ts.performanceTime > 0 && pNow >= ts.performanceTime) {
            outputTimestampGapMs = +(pNow - ts.performanceTime).toFixed(2);
          }
        } catch (_e) {}
      }
      if (ctx.destination) {
        maxChannelCount = ctx.destination.maxChannelCount || 2;
        channelCount = ctx.destination.channelCount || 2;
        channelInterp = ctx.destination.channelInterpretation || 'speakers';
      }
    }

    // Inspect all 33 persistent DSP sub-buses (21 continuous beds/propulsion + 12 3D Binaural HRTF/Atmos emitters)
    const em = this._emitters || {};
    const envBusEntries = [
      ['campus_breeze', this.breezeGain],
      ['tropical_rain_roar', this.rainRoarGain],
      ['raindrop_patter', this.rainPatterGain],
      ['roof_rain_resonance', this.roofResonanceGain],
      ['archipelago_surf', this.surfGain],
      ['subsea_tunnel_hum', this.tunnelHumGain],
      ['nocturnal_cricket', this.cricketGain],
      ['avata2_cinewhoop_motor', this.avataMotorGain],
      ['avata2_prop_wash', this.avataWashGain],
      ['avata2_prox_whoosh', this.proxWhooshGain],
      ['mavic3_aerial_hum', this.mavicGain],
      ['feline_purr_resonator', this.catPurrGain],
      ['rainforest_canopy_rustle', this.canopyRustleGain],
      ['afternoon_tonggeret', this.tonggeretGain],
      ['campus_social_murmur', this.socialMurmurGain],
      ['rotunda_flag_flutter', this.flagFlutterGain],
      ['teksas_truss_aeolian', this.trussAeolianGain],
      ['volcano_fumarole_vent', this.fumaroleGain],
      ['danau_turap_lapping', this.turapLapGain],
      ['cockpit_windshield_air', this.cockpitAirGain],
      ['cockpit_400hz_avionics', this.cockpitAvionicsGain]
    ];
    const emitterBusEntries = [
      ['3d_rotunda_fountain', em.fountain ? em.fountain.gain : null],
      ['3d_niagara_waterfall', em.waterfall ? em.waterfall.gain : null],
      ['3d_bikun_yellow_bus', em.bus ? em.bus.gain : null],
      ['3d_golf_cart_ev', em.cart ? em.cart.gain : null],
      ['3d_heli_evtol_pkuie', em.heli ? em.heli.gain : null],
      ['3d_cessna_172_pkuia', em.cessna ? em.cessna.gain : null],
      ['3d_citation_learjet_pkuib', em.learjet ? em.learjet.gain : null],
      ['3d_catamaran_ferry', em.ferry ? em.ferry.gain : null],
      ['3d_flybridge_yacht', em.yacht ? em.yacht.gain : null],
      ['3d_coastal_vessel', em.coastalBoat ? em.coastalBoat.gain : null],
      ['3d_videotron_line_array', em.videotron ? em.videotron.gain : null],
      ['3d_sivitas_npc_voice', em.npcVoice ? em.npcVoice.gain : null]
    ];

    let activeEnvSubBuses = 0;
    let hibernatedEnvSubBuses = 0;
    let silentEnvSubBuses = 0;
    let active3DEmitterBuses = 0;
    let hibernated3DEmitterBuses = 0;
    let silent3DEmitterBuses = 0;
    const activeBusTop = [];
    const hibernatedBusSample = [];
    const silentBusSample = [];

    if (initialized) {
      for (let i = 0; i < envBusEntries.length; i++) {
        const [name, gNode] = envBusEntries[i];
        if (!gNode || !gNode.gain) continue;
        const rawVal = gNode.gain._lastTgt !== undefined ? gNode.gain._lastTgt : gNode.gain.value;
        const slot = this._hibernationSlotByGain.get(gNode.gain);
        if (this.enabled && Number.isFinite(rawVal) && rawVal > 0.0005) {
          activeEnvSubBuses++;
          if (activeBusTop.length < 10) activeBusTop.push(`${name}:${rawVal.toFixed(3)}`);
        } else if (slot && slot.hibernated) {
          hibernatedEnvSubBuses++;
          if (hibernatedBusSample.length < 10) hibernatedBusSample.push(name);
        } else {
          silentEnvSubBuses++;
          if (silentBusSample.length < 10) silentBusSample.push(name);
        }
      }
      for (let i = 0; i < emitterBusEntries.length; i++) {
        const [name, gNode] = emitterBusEntries[i];
        if (!gNode || !gNode.gain) continue;
        const rawVal = gNode.gain._lastTgt !== undefined ? gNode.gain._lastTgt : gNode.gain.value;
        const slot = this._hibernationSlotByGain.get(gNode.gain);
        if (this.enabled && Number.isFinite(rawVal) && rawVal > 0.0005) {
          active3DEmitterBuses++;
          if (activeBusTop.length < 10) activeBusTop.push(`${name}:${rawVal.toFixed(3)}`);
        } else if (slot && slot.hibernated) {
          hibernated3DEmitterBuses++;
          if (hibernatedBusSample.length < 10) hibernatedBusSample.push(name);
        } else {
          silent3DEmitterBuses++;
          if (silentBusSample.length < 10) silentBusSample.push(name);
        }
      }
    }

    const activeAudibleSubBuses = activeEnvSubBuses + active3DEmitterBuses;
    const hibernatedSubBuses = hibernatedEnvSubBuses + hibernated3DEmitterBuses;
    const silentRunningSubBuses = silentEnvSubBuses + silent3DEmitterBuses;
    const uncleanedAux = Math.max(0, ns.transientAuxCreated - ns.transientAuxDisconnected);

    const gClass = String(hwGovernor?.gpuClass || '');
    const aTier = String(hwGovernor?.activeTierId || '');
    const isLowPowerDevice = Boolean(
      hwGovernor && (hwGovernor.isMobileDevice || aTier === 'TIER_C' || /INTEGRATED|MOBILE|SOFTWARE/i.test(gClass))
    );

    let verdict = 'UNINITIALIZED_AWAITING_GESTURE';
    let advisory = 'Klik tombol Suara di dock bawah untuk mengaktifkan mesin audio Web Audio API & Dolby Atmos 7.1.4.';

    if (
      (typeof window !== 'undefined' && !webAudioSupported) ||
      f.initErrors.length > 0 ||
      f.perf.updateErrorsCount > 0 ||
      (initialized && ctxState === 'closed')
    ) {
      verdict = 'CRITICAL_AUDIO_ERROR';
      advisory = f.initErrors[0] || f.perf.lastUpdateError || 'Konteks Web Audio tertutup atau gagal diinisialisasi.';
    } else if (!initialized) {
      verdict = 'UNINITIALIZED_AWAITING_GESTURE';
    } else if (muted) {
      verdict = 'MUTED_BY_USER';
      advisory = `Audio dibisukan pengguna (${hibernatedSubBuses} sub-bus dihibernasi hemat CPU, ${silentRunningSubBuses} meluruh).`;
    } else if (ctxState === 'suspended' || ctxState === 'interrupted') {
      verdict = 'AUTOPLAY_SUSPENDED';
      advisory = 'Konteks Web Audio tertahan kebijakan Autoplay browser — klik kontrol/kanvas sekali untuk membuka kunci suara.';
    } else if (f.clock.underrunSuspectCount >= 2 || f.perf.emaUpdateMs > 1.8) {
      verdict = 'AUDIO_UNDERRUN_OR_OVERLOAD';
      advisory = `Terdeteksi ${f.clock.underrunSuspectCount} gejala audio clock lag/underrun (maks ${f.clock.maxLagMs}ms; ${hibernatedSubBuses} sub-bus dihibernasi).`;
    } else if (isLowPowerDevice && silentRunningSubBuses >= 15) {
      verdict = 'MONOLITHIC_DSP_UNOPTIMIZED_FOR_DEVICE';
      advisory = `Audio berjalan lancar, namun ${silentRunningSubBuses} sub-bus diam belum dihibernasi.`;
    } else {
      verdict = 'OPTIMAL_AUDIO_RUNNING';
      advisory = `Web Audio berjalan sehat (${ctx ? ctx.sampleRate : 0}Hz, latency ${f.requestedLatencyHint}, update ${Math.round(f.perf.emaUpdateMs * 1000)}µs, ${activeAudibleSubBuses} bus berbunyi / ${hibernatedSubBuses} dihibernasi / ${silentRunningSubBuses} transisi, ${uncleanedAux} aux sisa).`;
    }

    const acTel = this.getAcousticTelemetry();
    const atmosStage = this.getAtmosStageSnapshot();

    return {
      forensicVersion: 'v2.4',
      stage: 'STAGE_2_ADAPTIVE_HIBERNATION_ZERO_CHARACTER_LOSS',
      verdict,
      tierProfile: {
        tierSystemImplemented: true,
        activeSoundTier: this._preferPlaybackBuffer ? 'ADAPTIVE_ATMOS_PLAYBACK_BUFFER' : 'ADAPTIVE_ATMOS_INTERACTIVE',
        mixProfile: this.mixProfile,
        spatialOutputMode: this.spatialOutputMode,
        panningModel: this.spatialOutputMode === 'SPEAKER' ? 'equalpower' : 'HRTF',
        atmosLayout: '7.1.4',
        requestedLatencyHint: f.requestedLatencyHint,
        recommendedDeviceSoundTier: hwGovernor
          ? isLowPowerDevice
            ? 'ADAPTIVE_ATMOS_PLAYBACK_BUFFER'
            : aTier === 'TIER_S'
              ? 'ULTRA_ATMOS_HRTF'
              : 'HIGH_SPATIAL'
          : 'AUTO'
      },
      hardwareAndContext: {
        webAudioSupported,
        initialized,
        muted,
        simPaused: Boolean(f.lastSimPaused),
        contextState: ctxState,
        currentTimeSec: ctx ? +ctx.currentTime.toFixed(2) : 0,
        sampleRateHz: ctx ? ctx.sampleRate || 0 : 0,
        baseLatencyMs,
        outputLatencyMs,
        outputTimestampGapMs,
        maxChannelCount,
        channelCount,
        channelInterpretation: channelInterp,
        initDurationMs: f.initDurationMs,
        initBreakdownMs: f.initBreakdownMs,
        initAttempts: f.initAttempts,
        initErrors: f.initErrors.slice(),
        resumeAttempts: f.resumeAttempts,
        resumeFailures: f.resumeFailures,
        stateTransitionsCount: f.stateTransitionsCount,
        lastStateChange: f.lastStateChange
      },
      dspGraphInventory: {
        totalPersistentNodes: ns.persistentTotal,
        oscillators: ns.oscillators,
        loopBufferSources: ns.loopBufferSources,
        biquadFilters: ns.biquadFilters,
        gainNodes: ns.gainNodes,
        panners3D: ns.panners3D,
        stereoPanners: ns.stereoPanners,
        delayLines: ns.delayLines,
        convolvers: ns.convolvers,
        waveShapers: ns.waveShapers,
        compressors: ns.compressors,
        analysers: ns.analysers,
        channelMergers: ns.channelMergers,
        proceduralAudioBuffers: ns.audioBuffersCount,
        proceduralBufferMemoryKB: +(ns.audioBufferBytes / 1024).toFixed(1),
        totalSubBuses: activeAudibleSubBuses + hibernatedSubBuses + silentRunningSubBuses,
        activeAudibleSubBuses,
        hibernatedSubBuses,
        silentRunningSubBuses,
        activeEnvSubBuses,
        hibernatedEnvSubBuses,
        silentEnvSubBuses,
        active3DEmitterBuses,
        hibernated3DEmitterBuses,
        silent3DEmitterBuses,
        activeSubBusSample: activeBusTop,
        hibernatedSubBusSample: hibernatedBusSample,
        silentSubBusSample: silentBusSample,
        transientSourceSpawns: ns.transientSourceSpawns,
        liveTransientSources: ns.liveTransientSources,
        transientAuxCreated: ns.transientAuxCreated,
        transientAuxDisconnected: ns.transientAuxDisconnected,
        uncleanedTransientAuxNodes: uncleanedAux
      },
      realtimeHealth: {
        updateCalls: f.perf.updateCalls,
        emaUpdateUs: Math.round(f.perf.emaUpdateMs * 1000.0),
        emaUpdateMs: +f.perf.emaUpdateMs.toFixed(3),
        maxUpdateMs: +f.perf.maxUpdateMs.toFixed(3),
        overBudgetFramesOver1p5ms: f.perf.overBudgetFrames,
        skippedRedundantParamOps: f.perf.skippedRedundantParamOps,
        updateErrorsCount: f.perf.updateErrorsCount,
        lastUpdateError: f.perf.lastUpdateError || null,
        clockChecks1s: f.clock.checksCount,
        lastClockDriftMs: f.clock.lastDriftMs,
        maxClockLagMs: f.clock.maxLagMs,
        underrunSuspectCount: f.clock.underrunSuspectCount,
        lastUnderrunLagMs: f.clock.lastUnderrunLagMs,
        masterVolumePct: Math.round(this.masterVolume * 100),
        masterGainTarget: +this._targetMasterLinearGain().toFixed(3),
        currentLimiterReductionDb: f.limiter.currentReductionDb,
        peakLimiterReductionDb: f.limiter.peakReductionDb,
        heavyLimitingEvents: f.limiter.heavyLimitingEvents
      },
      spatialAtmos: {
        layout: atmosStage.layout || '7.1.4',
        spatialMode: atmosStage.spatialMode || this.spatialOutputMode,
        perspectiveId: acTel.perspectiveId,
        perspectiveLabel: acTel.perspectiveLabel,
        zoneId: acTel.zoneId,
        microBiomeId: acTel.microBiomeId,
        diurnalPhase: acTel.diurnalPhase,
        reverbCharacter: acTel.reverbCharacter,
        reverbSendPct: acTel.reverbSendPct,
        weatherOcclusionPct: acTel.weatherOcclusionPct,
        activeAtmosObjects: atmosStage.objectCount || 0,
        occluded3DSources: acTel.occluded3DSources || 0,
        dominantSource: acTel.dominantSourceLabel,
        dominantDoppler: acTel.dominantDoppler,
        earBedPct: Math.round((atmosStage.earEnergy01 || 0) * 100),
        heightDomePct: Math.round((atmosStage.heightEnergy01 || 0) * 100),
        subLfePct: Math.round((atmosStage.lfeEnergy01 || 0) * 100),
        sfxTriggeredCount: acTel.sfxCount || 0,
        lastSfxId: acTel.lastSfxId || 'NONE'
      },
      advisory
    };
  }

  _updateDynamicFleetAcoustics(dt, sivitasAI, state = {}) {
    const now = this.ctx ? this.ctx.currentTime : 0.0;
    const audioReady = Boolean(this.ctx && this._emitters && this.enabled);
    const sfxLive = Boolean(audioReady && this.ctx.state === 'running');
    const trnM = this._mixWeights ? this._mixWeights.transitMult : 1.0;
    const natM = this._mixWeights ? this._mixWeights.natureMult : 1.0;
    const lx = this._lis.x;
    const ly = this._lis.y;
    const lz = this._lis.z;
    const lvx = this._lis.vx;
    const lvy = this._lis.vy;
    const lvz = this._lis.vz;
    const tempC = this._weatherState.tempC;
    const rh = this._weatherState.humidityPct;
    const wetness = this._weatherState.wetness;
    const coMove = this._viewPerspective ? this._viewPerspective.coMovingEntity : null;

    const vs =
      this._voiceStageState ||
      (this._voiceStageState = { active: false, x: 0, y: 0, z: 0, gain01: 0, speakerIdx: -1 });
    vs.active = false;
    vs.gain01 = 0.0;

    let bestLabel = 'Air Mancur & Alam';
    let bestScore = 0.08;
    let bestDoppler = 1.0;
    let occludedCount = 0;

    // 0. Directional Videotron Outdoor Speaker Array & Cooling Cabinet at (28.4, 7.6, 78.2)
    {
      const vtMet = AudioSynthesizer.computeDopplerAndAbsorption(
        28.4, 7.6, 78.2, 0, 0, 0, lx, ly, lz, lvx, lvy, lvz, tempC, rh
      );
      const vtOcc = AudioSynthesizer.computeGeometricOcclusion(
        28.4, 7.6, 78.2, lx, ly, lz, this._occScratch
      );
      if (vtOcc.occluded) occludedCount++;
      const vtProx = clamp(1.0 - vtMet.dist / 68.0, 0.0, 1.0);
      const vtPlaying = state.videotronPlaying !== false;
      const vtScore = vtProx * (vtPlaying ? 0.55 : 0.25) * vtOcc.gainMult;
      this._vtStageGain01 = clamp(vtScore, 0.0, 1.0);
      if (vtScore > bestScore && vtProx > 0.25) {
        bestScore = vtScore;
        bestLabel = 'Videotron UI (Array)';
        bestDoppler = 1.0;
      }
      if (audioReady && this._emitters.videotron) {
        const ev = this._emitters.videotron;
        this._applyAtmosCueToEmitter(ev, 28.4, 7.6, 78.2, now, 0.10);
        const modChord = 1.0 + 0.015 * Math.sin(this._simClock * 0.72);
        ev.oscA.frequency.setTargetAtTime(196.0 * modChord, now, 0.14);
        ev.oscB.frequency.setTargetAtTime(293.66 * modChord, now, 0.14);
        ev.padGain.gain.setTargetAtTime(vtPlaying ? 0.20 : 0.03, now, 0.14);
        ev.airLP.frequency.setTargetAtTime(Math.max(240.0, vtMet.airCutoffHz * vtOcc.cutoffMult), now, 0.10);
        ev.gain.gain.setTargetAtTime(vtProx > 0 ? vtProx * vtProx * 0.16 * vtOcc.gainMult * trnM : 0.0, now, 0.12);
      }
    }

    if (!sivitasAI) {
      if (audioReady && this._emitters.npcVoice) {
        this._emitters.npcVoice.gain.gain.setTargetAtTime(0.0, now, 0.10);
      }
      this._telemetry.occluded3DSources = occludedCount;
      this._telemetry.dominantSourceLabel = bestLabel;
      this._telemetry.dominantDoppler = Number(bestDoppler.toFixed(3));
      return;
    }

    // 1. Bus Kuning Lintas Pulau UI (+ Co-Moving RHD Cabin Driver Perspective in FPS_BUS_DRIVER)
    const b = sivitasAI.interIslandBus;
    if (b) {
      const tr = this._entityTracks.bus;
      const yaw = b.yaw || 0.0;
      const curSpd = b.speed || 0.0;
      const curPitch = b.pitch || 0.0;
      const wasInit = tr.init;
      const prevPitch = tr.prevPitch || 0.0;
      const jumped = this._updateTrackVelocity(tr, b.worldX, b.worldY + 1.15, b.worldZ, yaw, curSpd, dt);
      const accel = dt > 1e-4 ? (curSpd - tr.prevSpd) / dt : 0.0;
      tr.prevSpd = curSpd;
      tr.prevPitch = curPitch;
      tr.brakeCooldown = Math.max(0.0, tr.brakeCooldown - dt);
      tr.doorCooldown = Math.max(0.0, tr.doorCooldown - dt);
      tr.suspCooldown = Math.max(0.0, tr.suspCooldown - dt);

      const isBusCabin = coMove === 'bus';
      let srcX = tr.x;
      let srcY = tr.y;
      let srcZ = tr.z;
      let doppler = 1.0;
      let airCutHz = 9500.0;
      let occGain = 1.0;
      let occCut = 1.0;
      let distM = 3.5;

      if (isBusCabin) {
        // Inside RHD Bus Cockpit: Rear underfloor traction motor is 3.4m BEHIND and 0.85m BELOW driver's ears; zero self-Doppler!
        srcX = lx - Math.sin(yaw) * 3.4;
        srcY = ly - 0.85;
        srcZ = lz - Math.cos(yaw) * 3.4;
        airCutHz = 5200.0;
      } else {
        const met = AudioSynthesizer.computeDopplerAndAbsorption(
          tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz, tempC, rh
        );
        const occ = AudioSynthesizer.computeGeometricOcclusion(
          tr.x, tr.y, tr.z, lx, ly, lz, this._occScratch
        );
        if (occ.occluded) occludedCount++;
        doppler = met.doppler;
        airCutHz = met.airCutoffHz;
        occGain = occ.gainMult;
        occCut = occ.cutoffMult;
        distM = met.dist;
      }

      // State-transition SFX: Bus Stop / Depart Pneumatic Door & 2-Note Campus Chime
      const isMoving = curSpd >= 0.35;
      if (wasInit && isMoving !== tr.wasMoving && distM < 68.0 && tr.doorCooldown <= 0.0) {
        tr.doorCooldown = 4.2;
        this._recordSfxEvent('BUS_DOOR_CHIME', 'Pintu & Chime Bis UI');
        if (sfxLive) {
          const doorX = isBusCabin ? lx - Math.cos(yaw) * 1.15 + Math.sin(yaw) * 1.25 : tr.x;
          const doorY = isBusCabin ? ly - 0.35 : tr.y;
          const doorZ = isBusCabin ? lz + Math.sin(yaw) * 1.15 + Math.cos(yaw) * 1.25 : tr.z;
          this._triggerBusDoorChime3D(doorX, doorY, doorZ, isMoving);
        }
      }
      tr.wasMoving = isMoving;

      // State-transition SFX: Underpass Ramp Grade-Change Suspension Thump
      if (wasInit && Math.abs(curPitch - prevPitch) > 0.036 && curSpd > 1.8 && distM < 52.0 && tr.suspCooldown <= 0.0) {
        tr.suspCooldown = 3.5;
        this._recordSfxEvent('BUS_SUSPENSION_THUMP', 'Suspensi Ramp Bis UI');
        if (sfxLive) this._triggerSuspensionThump3D(srcX, srcY - 0.35, srcZ);
      }

      // State-transition SFX: Pneumatic Air-Brake Hiss (during active deceleration while rolling)
      if (wasInit && curSpd >= 0.40 && accel < -2.6 && distM < 58.0 && tr.brakeCooldown <= 0.0) {
        tr.brakeCooldown = 6.5;
        this._recordSfxEvent('BUS_AIR_BRAKE', 'Rem Angin Bis UI');
        if (sfxLive) this._triggerAirBrakeHiss(srcX, srcY - 0.25, srcZ);
      }

      const busProx = isBusCabin ? 1.0 : clamp(1.0 - distM / 135.0, 0.0, 1.0);
      const spdNorm = clamp(curSpd / 6.5, 0.0, 1.25);
      const score = busProx * (0.45 + 0.55 * spdNorm) * occGain;
      tr.emitX = srcX;
      tr.emitY = srcY;
      tr.emitZ = srcZ;
      tr.emitGain01 = clamp(busProx * (0.20 + spdNorm * 0.58) * occGain, 0.0, 1.0);
      if (score > bestScore) {
        bestScore = score;
        bestLabel = isBusCabin ? 'Kabina Bis Kuning UI' : 'Bis Kuning UI';
        bestDoppler = doppler;
      }

      if (audioReady) {
        const eb = this._emitters.bus;
        this._setPannerPosition(eb.panner, srcX, srcY, srcZ, now, jumped || isBusCabin);
        this._setPannerOrientation(eb.panner, Math.sin(yaw), 0.0, Math.cos(yaw), now, jumped || isBusCabin);
        this._applyAtmosCueToEmitter(eb, srcX, srcY, srcZ, now, 0.07);
        const baseHz = (68.0 + spdNorm * 64.0) * doppler;
        eb.oscA.frequency.setTargetAtTime(baseHz, now, 0.05);
        eb.oscB.frequency.setTargetAtTime(baseHz * 2.0, now, 0.05);
        const wetSpray = b.worldY > -1.8 ? smooth01(0.22, 0.85, wetness) : 0.0;
        eb.tireBp.frequency.setTargetAtTime((280 + spdNorm * 380 + wetSpray * 820) * doppler, now, 0.06);
        eb.tireGain.gain.setTargetAtTime(0.15 + spdNorm * (0.65 + wetSpray * 0.42), now, 0.06);
        if (eb.groundComb) {
          if (isBusCabin || b.worldY < -2.2) {
            eb.groundComb.gain.setTargetAtTime(0.0, now, 0.09);
          } else {
            const gComb = AudioSynthesizer.computeGroundReflectionEffect(
              srcX, srcY, srcZ, lx, ly, lz, b.worldY, 0.84, tempC, this._combScratch
            );
            eb.groundComb.frequency.setTargetAtTime(gComb.combNotchHz, now, 0.07);
            eb.groundComb.gain.setTargetAtTime(gComb.combGainDb * 0.65, now, 0.09);
          }
        }
        eb.airLP.frequency.setTargetAtTime(Math.max(220.0, airCutHz * occCut), now, 0.08);
        const busInTunnel = b.worldY < -2.2 ? 0.88 : 0.35;
        eb.revSend.gain.setTargetAtTime(busInTunnel, now, 0.12);
        eb.gain.gain.setTargetAtTime(
          busProx > 0 ? (0.08 + spdNorm * 0.28) * (isBusCabin ? 0.82 : 1.0) * occGain * trnM : 0.0,
          now,
          0.08
        );
      }
    }

    // 2. Mobil Golf Listrik UI (+ Co-Moving RHD Open Shuttle Perspective in FPS_GOLF_DRIVER)
    const c = sivitasAI.golfCart;
    if (c) {
      const tr = this._entityTracks.cart;
      const yaw = c.yaw || 0.0;
      const curSpd = c.speed || 0.0;
      const wasInit = tr.init;
      const jumped = this._updateTrackVelocity(tr, c.worldX, c.worldY + 0.85, c.worldZ, yaw, curSpd, dt);
      const accel = dt > 1e-4 ? (curSpd - tr.prevSpd) / dt : 0.0;
      tr.prevSpd = curSpd;
      tr.bellCooldown = Math.max(0.0, tr.bellCooldown - dt);

      const isCartDriver = coMove === 'cart';
      let srcX = tr.x;
      let srcY = tr.y;
      let srcZ = tr.z;
      let doppler = 1.0;
      let airCutHz = 11000.0;
      let occGain = 1.0;
      let occCut = 1.0;
      let distM = 1.5;

      if (isCartDriver) {
        // Rear PMSM electric transaxle is 1.35m BEHIND and 0.62m BELOW shuttle driver's ears; zero self-Doppler!
        srcX = lx - Math.sin(yaw) * 1.35;
        srcY = ly - 0.62;
        srcZ = lz - Math.cos(yaw) * 1.35;
      } else {
        const met = AudioSynthesizer.computeDopplerAndAbsorption(
          tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz, tempC, rh
        );
        const occ = AudioSynthesizer.computeGeometricOcclusion(
          tr.x, tr.y, tr.z, lx, ly, lz, this._occScratch
        );
        if (occ.occluded) occludedCount++;
        doppler = met.doppler;
        airCutHz = met.airCutoffHz;
        occGain = occ.gainMult;
        occCut = occ.cutoffMult;
        distM = met.dist;
      }

      const curPhase = c.phase || 'CRUISE';
      const phaseChanged = wasInit && tr.prevPhase && tr.prevPhase !== curPhase;
      tr.prevPhase = curPhase;
      if ((phaseChanged || (wasInit && accel < -1.55)) && distM < 58.0 && tr.bellCooldown <= 0.0) {
        tr.bellCooldown = 4.0;
        this._recordSfxEvent('CART_COURTESY_BELL', 'Bel Mobil Golf UI');
        if (sfxLive) {
          const bellX = isCartDriver ? lx + Math.sin(yaw) * 0.95 : tr.x;
          const bellY = isCartDriver ? ly - 0.25 : tr.y + 0.25;
          const bellZ = isCartDriver ? lz + Math.cos(yaw) * 0.95 : tr.z;
          this._triggerCourtesyBell3D(bellX, bellY, bellZ);
        }
      }

      const cartProx = isCartDriver ? 1.0 : clamp(1.0 - distM / 78.0, 0.0, 1.0);
      const spdNorm = clamp(curSpd / 2.7, 0.0, 1.2);
      const score = cartProx * (0.35 + 0.65 * spdNorm) * occGain;
      tr.emitX = srcX;
      tr.emitY = srcY;
      tr.emitZ = srcZ;
      tr.emitGain01 = clamp(cartProx * (0.14 + spdNorm * 0.48) * occGain, 0.0, 1.0);
      if (score > bestScore) {
        bestScore = score;
        bestLabel = 'Mobil Golf Listrik';
        bestDoppler = doppler;
      }

      if (audioReady) {
        const ec = this._emitters.cart;
        this._setPannerPosition(ec.panner, srcX, srcY, srcZ, now, jumped || isCartDriver);
        this._setPannerOrientation(ec.panner, Math.sin(yaw), 0.0, Math.cos(yaw), now, jumped || isCartDriver);
        this._applyAtmosCueToEmitter(ec, srcX, srcY, srcZ, now, 0.07);
        const invHz = (175.0 + spdNorm * 295.0) * doppler;
        ec.oscA.frequency.setTargetAtTime(invHz, now, 0.05);
        ec.oscB.frequency.setTargetAtTime(invHz * 2.005, now, 0.05);
        ec.rollBp.frequency.setTargetAtTime((360 + spdNorm * 320) * doppler, now, 0.06);
        ec.airLP.frequency.setTargetAtTime(Math.max(220.0, airCutHz * occCut), now, 0.08);
        ec.gain.gain.setTargetAtTime(
          cartProx > 0 ? (spdNorm * 0.22 + 0.018) * (isCartDriver ? 0.78 : 1.0) * occGain * trnM : 0.0,
          now,
          0.08
        );
      }
    }

    // 3. Taxi Helicopter Listrik eVTOL (PK-UIE + Helipad Skid Touchdown Thump + Co-Moving Overhead Rotor Dome)
    const h = sivitasAI.heliTaxi;
    if (h) {
      const tr = this._entityTracks.heli;
      const wasInit = tr.init;
      const yaw = h.yaw || 0.0;
      const jumped = this._updateTrackVelocity(tr, h.worldX, h.worldY + 1.35, h.worldZ, yaw, h.speed || 0, dt);
      const isEvtolCockpit = coMove === 'heli';
      let srcX = tr.x;
      let srcY = tr.y;
      let srcZ = tr.z;
      let doppler = 1.0;
      let airCutHz = 8400.0;
      let occGain = 1.0;
      let occCut = 1.0;
      let distM = 1.3;

      if (isEvtolCockpit) {
        // Inside eVTOL PK-UIE Cockpit: Coaxial rotor mast is +1.28m ABOVE & 0.16m AFT of pilot's head (el > +65° into .4 Height Dome!); zero self-Doppler!
        srcX = lx - Math.sin(yaw) * 0.16;
        srcY = ly + 1.28;
        srcZ = lz - Math.cos(yaw) * 0.16;
        airCutHz = 5100.0;
      } else {
        const met = AudioSynthesizer.computeDopplerAndAbsorption(
          tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz, tempC, rh
        );
        const occ = AudioSynthesizer.computeGeometricOcclusion(
          tr.x, tr.y, tr.z, lx, ly, lz, this._occScratch
        );
        if (occ.occluded) occludedCount++;
        doppler = met.doppler;
        airCutHz = met.airCutoffHz;
        occGain = occ.gainMult;
        occCut = occ.cutoffMult;
        distM = met.dist;
      }

      const curHeliPhase = h.phase || '';
      if (
        wasInit &&
        tr.prevPhase &&
        tr.prevPhase.startsWith('LANDING_') &&
        curHeliPhase.startsWith('DWELL_') &&
        (isEvtolCockpit || distM < 175.0)
      ) {
        this._recordSfxEvent('EVTOL_SKID_TOUCHDOWN', 'Touchdown Helipad eVTOL');
        if (sfxLive) this._triggerHeliSkidTouchdown3D(h.worldX, h.worldY + 0.15, h.worldZ);
      }
      tr.prevPhase = curHeliPhase;

      const heliProx = isEvtolCockpit ? 1.0 : clamp(1.0 - distM / 240.0, 0.0, 1.0);
      const omegaNorm = clamp((h.rotorOmega || 20.0) / 62.0, 0.2, 1.15);
      const vtolLoad =
        curHeliPhase && (curHeliPhase.startsWith('TAKEOFF') || curHeliPhase.startsWith('LANDING')) ? 1.25 : 1.0;
      const score = heliProx * omegaNorm * (isEvtolCockpit ? 1.12 : 0.95) * occGain;
      tr.emitX = srcX;
      tr.emitY = srcY;
      tr.emitZ = srcZ;
      tr.emitGain01 = clamp(heliProx * omegaNorm * 0.68 * occGain, 0.0, 1.0);
      if (score > bestScore) {
        bestScore = score;
        bestLabel = isEvtolCockpit ? 'Rotor Atas Heli eVTOL' : 'Taxi Heli eVTOL';
        bestDoppler = doppler;
      }

      if (audioReady) {
        const eh = this._emitters.heli;
        this._setPannerPosition(eh.panner, srcX, srcY, srcZ, now, jumped || isEvtolCockpit);
        this._applyAtmosCueToEmitter(eh, srcX, srcY, srcZ, now, 0.07);
        eh.bladeLfo.frequency.setTargetAtTime(clamp((12.0 + omegaNorm * 13.5) * doppler, 8.0, 34.0), now, 0.06);
        eh.turbineOsc.frequency.setTargetAtTime((340.0 + omegaNorm * 390.0) * doppler, now, 0.06);
        if (eh.groundComb) {
          if (isEvtolCockpit) {
            eh.groundComb.gain.setTargetAtTime(0.0, now, 0.09);
          } else {
            const overSea = tr.z > 430.0 && tr.z < 610.0;
            const gComb = AudioSynthesizer.computeGroundReflectionEffect(
              srcX, srcY, srcZ, lx, ly, lz, overSea ? -1.42 : 0.0, overSea ? 0.94 : 0.78, tempC, this._combScratch
            );
            eh.groundComb.frequency.setTargetAtTime(gComb.combNotchHz, now, 0.07);
            eh.groundComb.gain.setTargetAtTime(gComb.combGainDb, now, 0.09);
          }
        }
        eh.airLP.frequency.setTargetAtTime(Math.max(220.0, airCutHz * occCut), now, 0.08);
        const cabScale = isEvtolCockpit ? 0.64 : 1.0;
        const hGain = heliProx > 0 ? (0.04 + omegaNorm * 0.34) * vtolLoad * cabScale * occGain * trnM : 0.0;
        eh.gain.gain.setTargetAtTime(hGain, now, 0.09);
      }
    }

    // 4. Simultaneous Fixed-Wing Campus Fleet: Cessna 172 (PK-UIA) & Citation Learjet (PK-UIB)
    const cessna = sivitasAI.cessnaAircraft;
    if (cessna) {
      const prevP = this._sfxTracker.cessnaPhase;
      const curP = cessna.phase || '';
      if (prevP && prevP !== curP) {
        const dC = Math.hypot(lx - cessna.worldX, ly - cessna.worldY, lz - cessna.worldZ);
        if (prevP === 'FINAL_APPROACH' && curP === 'LANDING_ROLLOUT' && dC < 460.0) {
          this._recordSfxEvent('AIRCRAFT_TOUCHDOWN', 'Touchdown Roda Cessna');
          if (sfxLive) this._triggerRunwayTouchdownChirp3D(cessna.worldX, cessna.worldY + 0.2, cessna.worldZ, false);
        } else if (prevP === 'LINEUP_RWY27' && curP === 'TAKEOFF_CLIMB' && dC < 460.0) {
          this._recordSfxEvent('AIRCRAFT_TAKEOFF_SPOOL', 'Spool Lepas Landas Cessna');
          if (sfxLive) this._triggerTakeoffSpoolSurge3D(cessna.worldX, cessna.worldY + 1.0, cessna.worldZ, false);
        }
      }
      this._sfxTracker.cessnaPhase = curP;

      const tr = this._entityTracks.cessna;
      const yaw = cessna.yaw || 0.0;
      const jumped = this._updateTrackVelocity(tr, cessna.worldX, cessna.worldY + 1.4, cessna.worldZ, yaw, cessna.speed || 0, dt);
      const isCessnaCockpit = coMove === 'cessna';
      let srcX = tr.x;
      let srcY = tr.y;
      let srcZ = tr.z;
      let doppler = 1.0;
      let airCutHz = 8500.0;
      let occGain = 1.0;
      let occCut = 1.0;
      let distM = 2.0;

      if (isCessnaCockpit) {
        // Inside Cessna 172 Captain Seat: Nose Lycoming propeller is +1.95m IN FRONT of cockpit (az ≈ 0°); zero self-Doppler!
        srcX = lx + Math.sin(yaw) * 1.95;
        srcY = ly - 0.18;
        srcZ = lz + Math.cos(yaw) * 1.95;
        airCutHz = 4800.0;
      } else {
        const met = AudioSynthesizer.computeDopplerAndAbsorption(
          tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz, tempC, rh
        );
        const occ = AudioSynthesizer.computeGeometricOcclusion(
          tr.x, tr.y, tr.z, lx, ly, lz, this._occScratch
        );
        if (occ.occluded) occludedCount++;
        doppler = met.doppler;
        airCutHz = met.airCutoffHz;
        occGain = occ.gainMult;
        occCut = occ.cutoffMult;
        distM = met.dist;
      }

      const acProx = isCessnaCockpit ? 1.0 : clamp(1.0 - distM / 390.0, 0.0, 1.0);
      const parked = curP === 'APRON_PARKED';
      const rawPropNorm = parked ? 0.0 : clamp((cessna.propOmega || 0.0) / 74.0, 0.0, 1.15);
      const effPropNorm = isCessnaCockpit ? (parked ? 0.18 : Math.max(0.24, rawPropNorm)) : rawPropNorm;
      const score = acProx * effPropNorm * 1.12 * occGain;
      tr.emitX = srcX;
      tr.emitY = srcY;
      tr.emitZ = srcZ;
      tr.emitGain01 = !isCessnaCockpit && parked ? 0.0 : clamp(acProx * (0.18 + effPropNorm * 0.65) * occGain, 0.0, 1.0);
      if (score > bestScore) {
        bestScore = score;
        bestLabel = isCessnaCockpit ? 'Propeler Depan Cessna' : 'Cessna 172 Skyhawk';
        bestDoppler = doppler;
      }

      if (audioReady && this._emitters.cessna) {
        const ec = this._emitters.cessna;
        this._setPannerPosition(ec.panner, srcX, srcY, srcZ, now, jumped || isCessnaCockpit);
        this._setPannerOrientation(
          ec.panner,
          isCessnaCockpit ? -Math.sin(yaw) : Math.sin(yaw),
          0.0,
          isCessnaCockpit ? -Math.cos(yaw) : Math.cos(yaw),
          now,
          jumped || isCessnaCockpit
        );
        this._applyAtmosCueToEmitter(ec, srcX, srcY, srcZ, now, 0.07);
        if (ec.groundComb) {
          if (isCessnaCockpit) {
            ec.groundComb.gain.setTargetAtTime(0.0, now, 0.09);
          } else {
            const overRunway = tr.z >= 648.0 && tr.z <= 732.0 && tr.x >= -390.0 && tr.x <= -120.0;
            const gComb = AudioSynthesizer.computeGroundReflectionEffect(
              srcX, srcY, srcZ, lx, ly, lz, overRunway ? 0.6 : -1.42, overRunway ? 0.85 : 0.94, tempC, this._combScratch
            );
            ec.groundComb.frequency.setTargetAtTime(gComb.combNotchHz, now, 0.07);
            ec.groundComb.gain.setTargetAtTime(gComb.combGainDb, now, 0.09);
          }
        }
        ec.airLP.frequency.setTargetAtTime(Math.max(240.0, airCutHz * occCut), now, 0.08);
        if ((!isCessnaCockpit && parked) || acProx <= 0.0) {
          ec.gain.gain.setTargetAtTime(0.0, now, 0.12);
        } else {
          const pHz = (68.0 + effPropNorm * 74.0) * doppler;
          ec.propOscA.frequency.setTargetAtTime(pHz, now, 0.05);
          ec.propOscB.frequency.setTargetAtTime(pHz * 2.0, now, 0.05);
          ec.formant.frequency.setTargetAtTime(380 + effPropNorm * 340, now, 0.07);
          ec.propGain.gain.setTargetAtTime(0.42 + effPropNorm * 0.34, now, 0.08);
          ec.roarBp.frequency.setTargetAtTime((240 + effPropNorm * 280) * doppler, now, 0.06);
          ec.roarGain.gain.setTargetAtTime(0.18 + effPropNorm * 0.25, now, 0.08);
          const cabScale = isCessnaCockpit ? (parked ? 0.34 : 0.68) : 1.0;
          ec.gain.gain.setTargetAtTime((0.08 + effPropNorm * 0.40) * cabScale * occGain * trnM, now, 0.08);
        }
      }
    }

    const learjet = sivitasAI.learjetAircraft;
    if (learjet) {
      const prevP = this._sfxTracker.learjetPhase;
      const curP = learjet.phase || '';
      if (prevP && prevP !== curP) {
        const dL = Math.hypot(lx - learjet.worldX, ly - learjet.worldY, lz - learjet.worldZ);
        if (prevP === 'FINAL_APPROACH' && curP === 'LANDING_ROLLOUT' && dL < 520.0) {
          this._recordSfxEvent('JET_THRUST_REVERSER', 'Touchdown & Reverser Learjet');
          if (sfxLive) {
            this._triggerRunwayTouchdownChirp3D(learjet.worldX, learjet.worldY + 0.25, learjet.worldZ, true);
            this._triggerThrustReverserDeploy3D(learjet.worldX, learjet.worldY + 1.4, learjet.worldZ);
          }
        } else if (prevP === 'LINEUP_RWY27' && curP === 'TAKEOFF_CLIMB' && dL < 520.0) {
          this._recordSfxEvent('AIRCRAFT_TAKEOFF_SPOOL', 'Spool Lepas Landas Learjet');
          if (sfxLive) this._triggerTakeoffSpoolSurge3D(learjet.worldX, learjet.worldY + 1.4, learjet.worldZ, true);
        }
      }
      this._sfxTracker.learjetPhase = curP;

      const tr = this._entityTracks.learjet;
      const yaw = learjet.yaw || 0.0;
      const jumped = this._updateTrackVelocity(tr, learjet.worldX, learjet.worldY + 1.5, learjet.worldZ, yaw, learjet.speed || 0, dt);
      const isLearjetCockpit = coMove === 'learjet';

      // Learjet Landing Gear Electro-Hydraulic Pump Whine + Uplock/Downlock Clunk on gearDeploy transition
      this._sfxTracker.learjetGearSfxCooldown = Math.max(0.0, (this._sfxTracker.learjetGearSfxCooldown || 0.0) - dt);
      if (Number.isFinite(learjet.gearDeploy)) {
        const gDep = learjet.gearDeploy;
        const prevG = this._sfxTracker.learjetGearPrev;
        this._sfxTracker.learjetGearPrev = gDep;
        if (prevG >= 0.0 && this._sfxTracker.learjetGearSfxCooldown <= 0.0) {
          const retracting = prevG >= 0.96 && gDep < 0.96;
          const extending = prevG <= 0.04 && gDep > 0.04;
          if (retracting || extending) {
            const dGear = Math.hypot(lx - learjet.worldX, ly - learjet.worldY, lz - learjet.worldZ);
            if (isLearjetCockpit || dGear < 380.0) {
              this._sfxTracker.learjetGearSfxCooldown = 3.5;
              this._recordSfxEvent('LEARJET_GEAR_HYDRAULIC', retracting ? 'Retraksi Roda Learjet' : 'Ekstensi Roda Learjet');
              if (sfxLive) {
                const gx = isLearjetCockpit ? lx + Math.sin(yaw) * 0.65 : learjet.worldX;
                const gy = isLearjetCockpit ? ly - 0.85 : learjet.worldY + 0.55;
                const gz = isLearjetCockpit ? lz + Math.cos(yaw) * 0.65 : learjet.worldZ;
                this._triggerLandingGearActuator3D(gx, gy, gz, retracting, isLearjetCockpit);
              }
            }
          }
        }
      }
      let srcX = tr.x;
      let srcY = tr.y;
      let srcZ = tr.z;
      let doppler = 1.0;
      let airCutHz = 8800.0;
      let occGain = 1.0;
      let occCut = 1.0;
      let distM = 5.8;

      if (isLearjetCockpit) {
        // Inside Citation Learjet Captain Seat: Twin turbofans are -5.8m BEHIND cockpit (|az| > 150°); zero self-Doppler!
        srcX = lx - Math.sin(yaw) * 5.8;
        srcY = ly + 0.16;
        srcZ = lz - Math.cos(yaw) * 5.8;
        airCutHz = 4600.0;
      } else {
        const met = AudioSynthesizer.computeDopplerAndAbsorption(
          tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz, tempC, rh
        );
        const occ = AudioSynthesizer.computeGeometricOcclusion(
          tr.x, tr.y, tr.z, lx, ly, lz, this._occScratch
        );
        if (occ.occluded) occludedCount++;
        doppler = met.doppler;
        airCutHz = met.airCutoffHz;
        occGain = occ.gainMult;
        occCut = occ.cutoffMult;
        distM = met.dist;
      }

      const acProx = isLearjetCockpit ? 1.0 : clamp(1.0 - distM / 580.0, 0.0, 1.0);
      const parked = curP === 'APRON_PARKED';
      const rawPropNorm = parked ? 0.0 : clamp((learjet.propOmega || 0.0) / 76.0, 0.0, 1.15);
      const effPropNorm = isLearjetCockpit ? (parked ? 0.16 : Math.max(0.24, rawPropNorm)) : rawPropNorm;
      const reverserActive = curP === 'LANDING_ROLLOUT' && (learjet.speed || 0) > 16.0;
      const revBoost = reverserActive ? 1.38 : 1.0;
      const score = acProx * effPropNorm * 1.18 * revBoost * occGain;
      tr.emitX = srcX;
      tr.emitY = srcY;
      tr.emitZ = srcZ;
      tr.emitGain01 = !isLearjetCockpit && parked ? 0.0 : clamp(acProx * (0.20 + effPropNorm * 0.72) * revBoost * occGain, 0.0, 1.0);
      if (score > bestScore) {
        bestScore = score;
        bestLabel = isLearjetCockpit ? 'Turbofan Belakang Learjet' : 'Citation Learjet';
        bestDoppler = doppler;
      }

      if (audioReady && this._emitters.learjet) {
        const el = this._emitters.learjet;
        this._setPannerPosition(el.panner, srcX, srcY, srcZ, now, jumped || isLearjetCockpit);
        const coneSign = isLearjetCockpit ? 1.0 : !reverserActive ? -1.0 : 1.0;
        this._setPannerOrientation(el.panner, coneSign * Math.sin(yaw), 0.0, coneSign * Math.cos(yaw), now, jumped || isLearjetCockpit);
        this._applyAtmosCueToEmitter(el, srcX, srcY, srcZ, now, 0.07);
        if (el.groundComb) {
          if (isLearjetCockpit) {
            el.groundComb.gain.setTargetAtTime(0.0, now, 0.09);
          } else {
            const overRunway = tr.z >= 648.0 && tr.z <= 732.0 && tr.x >= -390.0 && tr.x <= -120.0;
            const gComb = AudioSynthesizer.computeGroundReflectionEffect(
              srcX, srcY, srcZ, lx, ly, lz, overRunway ? 0.6 : -1.42, overRunway ? 0.85 : 0.94, tempC, this._combScratch
            );
            el.groundComb.frequency.setTargetAtTime(gComb.combNotchHz, now, 0.07);
            el.groundComb.gain.setTargetAtTime(gComb.combGainDb, now, 0.09);
          }
        }
        el.airLP.frequency.setTargetAtTime(Math.max(240.0, airCutHz * occCut), now, 0.08);
        if ((!isLearjetCockpit && parked) || acProx <= 0.0) {
          el.gain.gain.setTargetAtTime(0.0, now, 0.12);
        } else {
          const n1Hz = (780 + effPropNorm * 1120) * doppler;
          el.jetWhine.frequency.setTargetAtTime(n1Hz, now, 0.05);
          el.jetWhineB.frequency.setTargetAtTime(n1Hz * 1.503, now, 0.05);
          el.jetWhineGain.gain.setTargetAtTime(0.16 + effPropNorm * 0.22, now, 0.08);
          el.roarBp.frequency.setTargetAtTime((310 + effPropNorm * 520 + (reverserActive ? 220 : 0)) * doppler, now, 0.06);
          el.roarGain.gain.setTargetAtTime((0.45 + effPropNorm * 0.65) * revBoost, now, 0.08);
          const cabScale = isLearjetCockpit ? (parked ? 0.30 : 0.62) : 1.0;
          el.gain.gain.setTargetAtTime((0.12 + effPropNorm * 0.56) * cabScale * revBoost * occGain * trnM, now, 0.08);
        }
      }
    }

    // 5. Simultaneous Marine Fleet: Shuttle Ferry + Mini Yacht + Nearest Active Coastal Boat + Wave Hull Slap!
    const seaWaveM = Number.isFinite(state.seaWaveHeightM) ? state.seaWaveHeightM : 0.28;
    let closestHullVessel = null;
    let closestHullDist = 85.0;

    const ferry = sivitasAI.shuttleFerry;
    if (ferry) {
      const prevFP = this._sfxTracker.ferryPhase;
      const curFP = ferry.phase || '';
      if (prevFP && prevFP !== curFP) {
        const dF = Math.hypot(lx - ferry.worldX, ly - ferry.worldY, lz - ferry.worldZ);
        if (dF < 280.0) {
          this._recordSfxEvent('FERRY_MARINE_HORN', 'Klakson Feri Selat UI');
          if (sfxLive) this._triggerMarineHorn3D(ferry.worldX, ferry.worldY + 3.2, ferry.worldZ, curFP === 'CRUISE');
        }
      }
      this._sfxTracker.ferryPhase = curFP;

      const tr = this._entityTracks.ferry;
      const yaw = ferry.yaw || 0.0;
      const jumped = this._updateTrackVelocity(tr, ferry.worldX, ferry.worldY + 1.0, ferry.worldZ, yaw, ferry.speed || 0, dt);
      const isFerryBridge = coMove === 'ferry';
      let srcX = tr.x;
      let srcY = tr.y;
      let srcZ = tr.z;
      let doppler = 1.0;
      let airCutHz = 8200.0;
      let occGain = 1.0;
      let occCut = 1.0;
      let distM = 6.0;

      if (isFerryBridge) {
        // Inside Catamaran Forward Wheelhouse: Twin diesel waterjets are 5.6m BEHIND & 2.2m BELOW Captain's bridge; zero self-Doppler!
        srcX = lx - Math.sin(yaw) * 5.6;
        srcY = ly - 2.2;
        srcZ = lz - Math.cos(yaw) * 5.6;
        airCutHz = 4400.0;
        closestHullDist = 2.2;
        closestHullVessel = tr;
      } else {
        const met = AudioSynthesizer.computeDopplerAndAbsorption(
          tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz, tempC, rh
        );
        const occ = AudioSynthesizer.computeGeometricOcclusion(
          tr.x, tr.y, tr.z, lx, ly, lz, this._occScratch
        );
        if (occ.occluded) occludedCount++;
        doppler = met.doppler;
        airCutHz = met.airCutoffHz;
        occGain = occ.gainMult;
        occCut = occ.cutoffMult;
        distM = met.dist;
        if (distM < closestHullDist) {
          closestHullDist = distM;
          closestHullVessel = tr;
        }
      }

      const fProx = isFerryBridge ? 1.0 : clamp(1.0 - distM / 155.0, 0.0, 1.0);
      const fSpd = clamp((ferry.speed || 0) / 3.4, 0.0, 1.15);
      const score = fProx * (0.28 + 0.56 * fSpd) * (isFerryBridge ? 1.18 : 1.0) * occGain;
      tr.emitX = srcX;
      tr.emitY = srcY;
      tr.emitZ = srcZ;
      tr.emitGain01 = clamp(score, 0.0, 1.0);
      if (score > bestScore) {
        bestScore = score;
        bestLabel = isFerryBridge ? 'Waterjet Buritan Feri UI' : 'Feri Katamaran UI';
        bestDoppler = doppler;
      }
      if (audioReady && this._emitters.ferry) {
        const ef = this._emitters.ferry;
        this._setPannerPosition(ef.panner, srcX, srcY, srcZ, now, jumped || isFerryBridge);
        const fSign = isFerryBridge ? 1.0 : -1.0;
        this._setPannerOrientation(ef.panner, fSign * Math.sin(yaw), 0.0, fSign * Math.cos(yaw), now, jumped || isFerryBridge);
        this._applyAtmosCueToEmitter(ef, srcX, srcY, srcZ, now, 0.08);
        const fHz = (52.0 + fSpd * 36.0) * doppler;
        ef.osc.frequency.setTargetAtTime(fHz, now, 0.07);
        ef.oscB.frequency.setTargetAtTime(fHz * 2.0, now, 0.07);
        ef.wakeBp.frequency.setTargetAtTime((360.0 + fSpd * 330.0 + seaWaveM * 85.0) * doppler, now, 0.08);
        ef.airLP.frequency.setTargetAtTime(Math.max(220.0, airCutHz * occCut), now, 0.09);
        const cabScale = isFerryBridge ? 0.68 : 1.0;
        ef.gain.gain.setTargetAtTime(fProx > 0 ? (0.035 + fSpd * 0.24) * cabScale * occGain * trnM : 0.0, now, 0.10);
      }
    }

    const yacht = sivitasAI.miniYacht;
    if (yacht) {
      const tr = this._entityTracks.yacht;
      const yaw = yacht.yaw || 0.0;
      const jumped = this._updateTrackVelocity(tr, yacht.worldX, yacht.worldY + 0.9, yacht.worldZ, yaw, yacht.speed || 0, dt);
      const isYachtHelm = coMove === 'yacht';
      let srcX = tr.x;
      let srcY = tr.y;
      let srcZ = tr.z;
      let doppler = 1.0;
      let airCutHz = 8600.0;
      let occGain = 1.0;
      let occCut = 1.0;
      let distM = 3.7;

      if (isYachtHelm) {
        // On Mini Yacht Open Flybridge Helm: Twin inboard stern drives are 3.4m BEHIND & 1.4m BELOW helm; zero self-Doppler!
        srcX = lx - Math.sin(yaw) * 3.4;
        srcY = ly - 1.4;
        srcZ = lz - Math.cos(yaw) * 3.4;
        airCutHz = 7200.0;
        closestHullDist = 1.8;
        closestHullVessel = tr;
      } else {
        const met = AudioSynthesizer.computeDopplerAndAbsorption(
          tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz, tempC, rh
        );
        const occ = AudioSynthesizer.computeGeometricOcclusion(
          tr.x, tr.y, tr.z, lx, ly, lz, this._occScratch
        );
        if (occ.occluded) occludedCount++;
        doppler = met.doppler;
        airCutHz = met.airCutoffHz;
        occGain = occ.gainMult;
        occCut = occ.cutoffMult;
        distM = met.dist;
        if (distM < closestHullDist) {
          closestHullDist = distM;
          closestHullVessel = tr;
        }
      }

      const yProx = isYachtHelm ? 1.0 : clamp(1.0 - distM / 120.0, 0.0, 1.0);
      const ySpd = clamp((yacht.speed || 0) / 3.2, 0.0, 1.15);
      const score = yProx * (0.25 + 0.54 * ySpd) * (isYachtHelm ? 1.18 : 1.0) * occGain;
      tr.emitX = srcX;
      tr.emitY = srcY;
      tr.emitZ = srcZ;
      tr.emitGain01 = clamp(score, 0.0, 1.0);
      if (score > bestScore) {
        bestScore = score;
        bestLabel = isYachtHelm ? 'Mesin Buritan Mini Yacht' : 'Mini Yacht Teluk U';
        bestDoppler = doppler;
      }
      if (audioReady && this._emitters.yacht) {
        const ey = this._emitters.yacht;
        this._setPannerPosition(ey.panner, srcX, srcY, srcZ, now, jumped || isYachtHelm);
        const ySign = isYachtHelm ? 1.0 : -1.0;
        this._setPannerOrientation(ey.panner, ySign * Math.sin(yaw), 0.0, ySign * Math.cos(yaw), now, jumped || isYachtHelm);
        this._applyAtmosCueToEmitter(ey, srcX, srcY, srcZ, now, 0.08);
        ey.osc.frequency.setTargetAtTime((72.0 + ySpd * 48.0) * doppler, now, 0.07);
        ey.wakeBp.frequency.setTargetAtTime((520.0 + ySpd * 380.0 + seaWaveM * 95.0) * doppler, now, 0.08);
        ey.airLP.frequency.setTargetAtTime(Math.max(220.0, airCutHz * occCut), now, 0.09);
        const cabScale = isYachtHelm ? 0.78 : 1.0;
        ey.gain.gain.setTargetAtTime(yProx > 0 ? (0.025 + ySpd * 0.21) * cabScale * occGain * trnM : 0.0, now, 0.10);
      }
    }

    // Nearest Coastal Archipelago Boat (`sivitasAI.marineVessels`: `SMALL_SHIP` Diesel Chug vs `FISHING_BOAT` Outboard)
    if (Array.isArray(sivitasAI.marineVessels) && sivitasAI.marineVessels.length > 0) {
      let nearestBoat = null;
      let nearestIdx = -1;
      let nearestD = 165.0;
      for (let i = 0; i < sivitasAI.marineVessels.length; i++) {
        const mv = sivitasAI.marineVessels[i];
        const d = Math.hypot(lx - mv.worldX, ly - (mv.worldY || -1.4), lz - mv.worldZ);
        if (d < nearestD) {
          nearestD = d;
          nearestBoat = mv;
          nearestIdx = i;
        }
      }
      if (nearestBoat) {
        const tr = this._entityTracks.coastalBoat;
        const switched = tr.idx !== nearestIdx;
        tr.idx = nearestIdx;
        if (switched) tr.init = false;
        const yaw = nearestBoat.yaw || 0.0;
        const jumped = this._updateTrackVelocity(
          tr,
          nearestBoat.worldX,
          (nearestBoat.worldY || -1.4) + 0.65,
          nearestBoat.worldZ,
          yaw,
          nearestBoat.speed || 0,
          dt
        );
        const met = AudioSynthesizer.computeDopplerAndAbsorption(
          tr.x, tr.y, tr.z, tr.vx, tr.vy, tr.vz, lx, ly, lz, lvx, lvy, lvz, tempC, rh
        );
        const occ = AudioSynthesizer.computeGeometricOcclusion(
          tr.x, tr.y, tr.z, lx, ly, lz, this._occScratch
        );
        if (occ.occluded) occludedCount++;
        if (met.dist < closestHullDist) {
          closestHullDist = met.dist;
          closestHullVessel = tr;
        }

        const isFishing =
          nearestBoat.type === 'FISHING_BOAT' || nearestBoat.kind === 'FISHING_BOAT';
        const cbProx = clamp(1.0 - met.dist / 135.0, 0.0, 1.0);
        const cbSpd = clamp((nearestBoat.speed || 0) / 3.5, 0.1, 1.2);
        const score = cbProx * (0.26 + 0.52 * cbSpd) * occ.gainMult;
        tr.emitX = tr.x;
        tr.emitY = tr.y;
        tr.emitZ = tr.z;
        tr.emitGain01 = clamp(score, 0.0, 1.0);
        if (score > bestScore) {
          bestScore = score;
          bestLabel = isFishing ? 'Perahu Nelayan UI' : 'Kapal Patroli Pesisir';
          bestDoppler = met.doppler;
        }
        if (audioReady && this._emitters.coastalBoat) {
          const ecb = this._emitters.coastalBoat;
          this._setPannerPosition(ecb.panner, tr.x, tr.y, tr.z, now, jumped || switched);
          this._setPannerOrientation(ecb.panner, -Math.sin(yaw), 0.0, -Math.cos(yaw), now, jumped || switched);
          this._applyAtmosCueToEmitter(ecb, tr.x, tr.y, tr.z, now, 0.08);
          const motorHz = (isFishing ? 86.0 + cbSpd * 58.0 : 44.0 + cbSpd * 28.0) * met.doppler;
          const chugRate = clamp((isFishing ? 13.5 + cbSpd * 8.5 : 6.8 + cbSpd * 4.8) * met.doppler, 4.0, 28.0);
          ecb.osc.frequency.setTargetAtTime(motorHz, now, 0.06);
          ecb.chugLfo.frequency.setTargetAtTime(chugRate, now, 0.06);
          ecb.lp.frequency.setTargetAtTime(isFishing ? 640 : 320, now, 0.08);
          ecb.wakeBp.frequency.setTargetAtTime((440 + cbSpd * 320 + seaWaveM * 90) * met.doppler, now, 0.08);
          ecb.airLP.frequency.setTargetAtTime(Math.max(220.0, met.airCutoffHz * occ.cutoffMult), now, 0.09);
          ecb.gain.gain.setTargetAtTime((0.025 + cbSpd * 0.19) * cbProx * occ.gainMult * trnM, now, 0.09);
        }
      } else {
        this._entityTracks.coastalBoat.emitGain01 = 0.0;
        if (audioReady && this._emitters.coastalBoat) {
          this._emitters.coastalBoat.gain.gain.setTargetAtTime(0.0, now, 0.12);
        }
      }
    }

    // 6. Conversing Sivitas Partner 3D Voice (`npcVoice`) & Passing NPC 3D Gait Footsteps
    this._npcFootstepCooldown = Math.max(0.0, (this._npcFootstepCooldown || 0.0) - dt);
    if (Array.isArray(sivitasAI.npcs) && sivitasAI.npcs.length > 0 && ly < 24.0) {
      const rainInt = Number(state.rainIntensity) || 0.0;
      const camMode = state.camMode || 'DRONE_6DOF';
      const cands = this._chatCandidates;
      cands[0] = null;
      cands[1] = null;
      cands[2] = null;
      cands[3] = null;
      let d0 = 999.0;
      let d1 = 999.0;
      let d2 = 999.0;
      let d3 = 999.0;

      for (let i = 0; i < sivitasAI.npcs.length; i++) {
        const npc = sivitasAI.npcs[i];
        const ny = Number(npc.worldY) || 0.0;
        const dh = Math.hypot(lx - npc.worldX, lz - npc.worldZ);
        const d3d = Math.hypot(dh, ly - (ny + 1.45));

        // 6A. Passing NPC / Jogger 3D Footsteps (excludes self-avatar at dh < 0.48m)
        const nSpd = Number(npc.liveSpeed) || 0.0;
        if (dh >= 0.48 && d3d < 14.5 && ly < 11.5 && nSpd > 0.45 && i < this._npcStepHalfPhase.length) {
          const halfStep = Math.floor(((Number(npc.gaitCycleU) || 0.0) * 2.0) % 2.0);
          const prevHalf = this._npcStepHalfPhase[i];
          this._npcStepHalfPhase[i] = halfStep;
          if (prevHalf >= 0 && prevHalf !== halfStep && this._npcFootstepCooldown <= 0.0) {
            this._npcFootstepCooldown = 0.065;
            if (sfxLive) {
              this._triggerNpcFootstep3D(npc.worldX, ny + 0.06, npc.worldZ, nSpd, npc.behavior === 'JOGGER_LOOP');
            }
          }
        }

        // 6B. Conversing Sivitas Partner Candidates (skips dh < 0.42m self-head in FPS_CHAT_NPC!)
        if (dh >= 0.42 && d3d < 22.0 && rainInt < 0.78) {
          const beh = npc.behavior || '';
          const isChatting =
            beh === 'SEMINAR_CIRCLE' ||
            beh === 'PAIR_ROTUNDA' ||
            beh === 'PAIR_SE_WALKWAY' ||
            beh === 'BENCH_SEATED' ||
            (camMode === 'FPS_GOLF_DRIVER' && beh === 'GOLF_CART_SHUTTLE_RIDER' && (npc.seatedBlend || 0) > 0.5);
          if (isChatting) {
            if (d3d < d0) {
              d3 = d2; cands[3] = cands[2];
              d2 = d1; cands[2] = cands[1];
              d1 = d0; cands[1] = cands[0];
              d0 = d3d; cands[0] = npc;
            } else if (d3d < d1) {
              d3 = d2; cands[3] = cands[2];
              d2 = d1; cands[2] = cands[1];
              d1 = d3d; cands[1] = npc;
            } else if (d3d < d2) {
              d3 = d2; cands[3] = cands[2];
              d2 = d3d; cands[2] = npc;
            } else if (d3d < d3) {
              d3 = d3d; cands[3] = npc;
            }
          }
        }
      }

      if (cands[0]) {
        let groupCnt = 1;
        if (d0 < 5.5) {
          if (cands[1] && d1 < d0 + 3.0) groupCnt = 2;
          if (cands[2] && d2 < d0 + 3.0 && groupCnt === 2) groupCnt = 3;
          if (cands[3] && d3 < d0 + 3.0 && groupCnt === 3) groupCnt = 4;
        }
        const turnSlot = groupCnt > 1 ? Math.floor(this._simClock / 3.4) % groupCnt : 0;
        const sp = cands[turnSlot] || cands[0];
        const spIdx = Number.isFinite(sp.index) ? sp.index : turnSlot;
        const isSeated = sp.behavior === 'BENCH_SEATED' || (sp.seatedBlend || 0) > 0.5;
        const mouthY = (Number(sp.worldY) || 0.0) + (isSeated ? 1.18 : 1.52) * (sp.heightScale || 1.0);
        const vOcc = AudioSynthesizer.computeGeometricOcclusion(
          sp.worldX, mouthY, sp.worldZ, lx, ly, lz, this._occScratch
        );
        const dMouth = Math.hypot(lx - sp.worldX, ly - mouthY, lz - sp.worldZ);
        const vProx = clamp(1.0 - dMouth / 22.0, 0.0, 1.0);
        const socialFocusBoost = camMode === 'FPS_CHAT_NPC' ? 1.38 : 1.0;
        const vGain01 = clamp(vProx * vProx * 0.76 * socialFocusBoost * vOcc.gainMult, 0.0, 1.0);

        if (vGain01 > 0.012) {
          vs.active = true;
          vs.x = sp.worldX;
          vs.y = mouthY;
          vs.z = sp.worldZ;
          vs.gain01 = vGain01;
          vs.speakerIdx = spIdx;
        }
        const vScore = vProx * 0.90 * socialFocusBoost * vOcc.gainMult;
        if (vScore > bestScore && vProx > 0.28) {
          bestScore = vScore;
          bestLabel = 'Dialog Sivitas 3D';
          bestDoppler = 1.0;
        }

        if (audioReady && this._emitters.npcVoice) {
          const ev = this._emitters.npcVoice;
          const spYaw = sp.yaw || 0.0;
          this._setPannerPosition(ev.panner, sp.worldX, mouthY, sp.worldZ, now, false);
          this._setPannerOrientation(ev.panner, Math.sin(spYaw), 0.0, Math.cos(spYaw), now, false);
          this._applyAtmosCueToEmitter(ev, sp.worldX, mouthY, sp.worldZ, now, 0.06);
          const isFemale = spIdx % 2 === 1;
          const f0 =
            (isFemale ? 188.0 + (spIdx % 5) * 7.5 : 122.0 + (spIdx % 5) * 6.0) *
            (1.0 + 0.045 * Math.sin(this._simClock * 2.7 + spIdx));
          const vowelMod = 0.5 + 0.5 * Math.sin(this._simClock * 5.2 + spIdx * 1.3);
          const sylEnv = clamp(
            Math.sin(this._simClock * 11.4 + spIdx * 1.7) * 0.62 +
              Math.sin(this._simClock * 4.3 + spIdx * 0.9) * 0.48,
            0.0,
            1.0
          );
          ev.glottal.frequency.setTargetAtTime(f0, now, 0.05);
          ev.sub.frequency.setTargetAtTime(f0 * 0.5, now, 0.05);
          ev.form1.frequency.setTargetAtTime(420.0 + vowelMod * 340.0 + (isFemale ? 75.0 : 0.0), now, 0.06);
          ev.form2.frequency.setTargetAtTime(1180.0 + (1.0 - vowelMod) * 880.0 + (isFemale ? 160.0 : 0.0), now, 0.06);
          ev.syllable.gain.setTargetAtTime(0.16 + sylEnv * 0.84, now, 0.035);
          ev.airLP.frequency.setTargetAtTime(Math.max(380.0, 6800.0 * vOcc.cutoffMult), now, 0.08);
          ev.gain.gain.setTargetAtTime(vProx * vProx * 0.16 * socialFocusBoost * vOcc.gainMult * natM, now, 0.07);
        }
      } else if (audioReady && this._emitters.npcVoice) {
        this._emitters.npcVoice.gain.gain.setTargetAtTime(0.0, now, 0.10);
      }
    } else if (audioReady && this._emitters.npcVoice) {
      this._emitters.npcVoice.gain.gain.setTargetAtTime(0.0, now, 0.10);
    }

    // Sea-Wave-Coupled Boat Hull Wave-Slap SFX when seaWaveHeightM >= 0.80 m (scales up to 3.00 m!)
    this._sfxTracker.hullSlapCooldown = Math.max(0.0, this._sfxTracker.hullSlapCooldown - dt);
    if (seaWaveM >= 0.80 && closestHullVessel && this._sfxTracker.hullSlapCooldown <= 0.0) {
      this._sfxTracker.hullSlapCooldown = clamp(3.4 - (seaWaveM - 0.8) * 1.05, 1.05, 3.6);
      this._recordSfxEvent('BOAT_HULL_WAVE_SLAP', 'Hempasan Ombak Lambung Kapal');
      if (sfxLive) {
        this._triggerHullWaveSlap3D(closestHullVessel.x, closestHullVessel.y - 0.4, closestHullVessel.z, seaWaveM);
      }
    }

    this._foregroundPressure01 = clamp((bestScore - 0.32) * 1.35, 0.0, 1.0);
    this._telemetry.occluded3DSources = occludedCount;
    this._telemetry.dominantSourceLabel = bestLabel;
    this._telemetry.dominantDoppler = Number(bestDoppler.toFixed(3));
  }

  /**
   * Populates the live zero-GC Dolby Atmos 7.1.4 Point-of-View Spatial Stage (`this._atmosStage`)
   * from the exact active camera coordinate, 3D head orientation, surrounding 3D entities, and beds.
   * Runs even when Web Audio `this.ctx === null` so HUD radar & headless physics verification stay 100% live.
   */
  _updateAtmosStageSnapshot(state = {}, sivitasAI = null, camMode = 'DRONE_6DOF', rainInt = 0.0, windSpd = 0.85) {
    this._atmosActiveObjects.length = 0;
    const lx = this._lis.x;
    const ly = this._lis.y;
    const lz = this._lis.z;
    const anc = this._atmosAnchors;
    const biome = this._biomeState;
    const diurnal = this._diurnalState;
    const zone = this._zoneState;

    // 1. Conversing Sivitas Partner 3D Voice (highest priority near-field human object)
    if (this._voiceStageState && this._voiceStageState.active) {
      this._pushAtmosObject(
        'NPC_VOICE',
        'Dialog Sivitas 3D',
        'VOICE',
        this._voiceStageState.x,
        this._voiceStageState.y,
        this._voiceStageState.z,
        this._voiceStageState.gain01,
        0.35,
        0.0
      );
    }

    // 2. Rotunda Fountain (-2.5, 0.9, 106.5) & Niagara Waterfall / Pulau U River
    const dFount = Math.hypot(lx - -2.5, ly - 0.9, lz - 106.5);
    if (dFount < 92.0) {
      const fOcc = AudioSynthesizer.computeGeometricOcclusion(-2.5, 0.9, 106.5, lx, ly, lz, this._occScratch);
      const fProx = clamp(1.0 - dFount / 92.0, 0.0, 1.0);
      this._pushAtmosObject('FOUNTAIN', 'Air Mancur Rotunda', 'WATER', -2.5, 0.9, 106.5, fProx * fProx * 0.68 * fOcc.gainMult, 5.6, 0.08);
    }

    const dNiagara = Math.hypot(lx - -44.0, ly - 9.0, lz - 643.0);
    const niagProx = clamp(1.0 - dNiagara / 165.0, 0.0, 1.0);
    const riverZ = state.riverZ;
    let riverProx = 0.0;
    if (riverZ !== null && Number.isFinite(riverZ) && lx >= -295.0 && lx <= -98.0 && lz < 420.0) {
      riverProx = clamp(1.0 - Math.hypot(ly - 1.2, lz - riverZ) / 54.0, 0.0, 1.0);
    }
    if (riverProx * 0.55 > niagProx && riverProx > 0.02) {
      const rx = clamp(lx, -288.0, -104.0);
      this._pushAtmosObject('RIVER', 'Sungai Pulau U', 'WATER', rx, 0.8, riverZ, riverProx * riverProx * 0.52, 6.5, 0.08);
    } else if (niagProx > 0.02) {
      const wOcc = AudioSynthesizer.computeGeometricOcclusion(-44.0, 9.0, 643.0, lx, ly, lz, this._occScratch);
      this._pushAtmosObject(
        'NIAGARA',
        'Air Terjun Niagara',
        'WATER',
        -44.0,
        9.0,
        643.0,
        niagProx * niagProx * 0.85 * wOcc.gainMult,
        16.5,
        0.42
      );
    }

    // 3. Coastline Surf Bed Object (anchored along true coastline gradient)
    const coastDist = Number.isFinite(state.coastDist) ? state.coastDist : -40.0;
    const waveM = clamp(Number.isFinite(state.seaWaveHeightM) ? state.seaWaveHeightM : 0.28, 0.0, 3.0);
    const shoreNear = Math.abs(coastDist) < 62.0 ? 1.0 - Math.abs(coastDist) / 62.0 : 0.0;
    const openWaterLow = coastDist > 0.0 && ly > -1.8 && ly < 32.0 ? (32.0 - ly) / 32.0 : 0.0;
    const surfAlt = ly > -1.8 ? clamp(1.0 - Math.max(0.0, ly - 6.5) / 44.0, 0.0, 1.0) : 0.0;
    const surfG = Math.max(shoreNear, openWaterLow * 0.85) * surfAlt * (0.18 + 0.82 * Math.pow(waveM / 3.0, 0.72));
    if (surfG > 0.02 && anc) {
      this._pushAtmosObject('SURF', 'Ombak Pesisir', 'WATER', anc.sea.x, anc.sea.y, anc.sea.z, surfG * 0.72, 18.0, clamp(waveM / 3.0, 0.12, 0.55));
    }

    // 4. Videotron Speaker Array & Dynamic Transit / Aviation / Marine Fleet
    if ((this._vtStageGain01 || 0) > 0.02) {
      this._pushAtmosObject('VIDEOTRON', 'Videotron UI', 'OBJECT', 28.4, 7.6, 78.2, this._vtStageGain01, 3.6, 0.12);
    }
    if (sivitasAI) {
      const trs = this._entityTracks;
      if (sivitasAI.interIslandBus && (trs.bus.emitGain01 || 0) > 0.01) {
        this._pushAtmosObject('BUS', 'Bis Kuning UI', 'TRANSIT', trs.bus.emitX, trs.bus.emitY, trs.bus.emitZ, trs.bus.emitGain01, 3.2, 0.34);
      }
      if (sivitasAI.golfCart && (trs.cart.emitGain01 || 0) > 0.01) {
        this._pushAtmosObject('CART', 'Mobil Golf UI', 'TRANSIT', trs.cart.emitX, trs.cart.emitY, trs.cart.emitZ, trs.cart.emitGain01, 1.8, 0.05);
      }
      if (sivitasAI.heliTaxi && (trs.heli.emitGain01 || 0) > 0.01) {
        this._pushAtmosObject('HELI', 'Heli eVTOL', 'AIR', trs.heli.emitX, trs.heli.emitY, trs.heli.emitZ, trs.heli.emitGain01, 4.2, 0.38);
      }
      if (sivitasAI.cessnaAircraft && (trs.cessna.emitGain01 || 0) > 0.01) {
        this._pushAtmosObject('CESSNA', 'Cessna PK-UIA', 'AIR', trs.cessna.emitX, trs.cessna.emitY, trs.cessna.emitZ, trs.cessna.emitGain01, 3.8, 0.26);
      }
      if (sivitasAI.learjetAircraft && (trs.learjet.emitGain01 || 0) > 0.01) {
        this._pushAtmosObject('LEARJET', 'Learjet PK-UIB', 'AIR', trs.learjet.emitX, trs.learjet.emitY, trs.learjet.emitZ, trs.learjet.emitGain01, 5.2, 0.52);
      }
      if (sivitasAI.shuttleFerry && (trs.ferry.emitGain01 || 0) > 0.01) {
        this._pushAtmosObject('FERRY', 'Feri Selat UI', 'MARINE', trs.ferry.emitX, trs.ferry.emitY, trs.ferry.emitZ, trs.ferry.emitGain01, 6.5, 0.44);
      }
      if (sivitasAI.miniYacht && (trs.yacht.emitGain01 || 0) > 0.01) {
        this._pushAtmosObject('YACHT', 'Mini Yacht', 'MARINE', trs.yacht.emitX, trs.yacht.emitY, trs.yacht.emitZ, trs.yacht.emitGain01, 3.4, 0.22);
      }
      if ((trs.coastalBoat.emitGain01 || 0) > 0.01) {
        this._pushAtmosObject('BOAT', 'Perahu Pesisir', 'MARINE', trs.coastalBoat.emitX, trs.coastalBoat.emitY, trs.coastalBoat.emitZ, trs.coastalBoat.emitGain01, 2.6, 0.16);
      }
    }

    // 5. Overhead Canopy, Wildlife & Landmark Micro-Biome Beds (projected from true 3D coordinates!)
    if (anc && biome && zone.zoneId !== 'SUBSEA_TUNNEL') {
      const cabinInsul =
        zone.zoneId === 'COCKPIT_INTERIOR'
          ? this._viewPerspective.coMovingEntity === 'learjet'
            ? 0.12
            : 0.22
          : 1.0;
      const windNorm = clamp((windSpd - 0.5) / 3.2, 0.08, 1.4);
      const canopyG = biome.canopyWeight * (0.20 + 0.45 * windNorm) * zone.weatherOcclusionGain * cabinInsul;
      if (canopyG > 0.03) {
        this._pushAtmosObject('CANOPY', 'Tajuk Hutan Atas', 'BED', anc.canopy.x, anc.canopy.y, anc.canopy.z, canopyG, 9.5, 0.0);
      }
      const altBio = clamp(1.0 - Math.max(0.0, ly - 8.0) / 34.0, 0.0, 1.0) * cabinInsul;
      const tongG = (diurnal ? diurnal.tonggeretWeight : 0.0) * smooth01(0.18, 0.65, biome.canopyWeight) * altBio * 0.42;
      if (tongG > 0.03) {
        this._pushAtmosObject('TONGGERET', 'Tonggeret Kanopi', 'BED', anc.tonggeret.x, anc.tonggeret.y, anc.tonggeret.z, tongG, 5.0, 0.0);
      }
      const crikG = Math.max(smooth01(0.28, 0.65, this._weatherState.nightFactor), diurnal ? diurnal.cricketWeight : 0.0) * altBio * 0.36;
      if (crikG > 0.03) {
        this._pushAtmosObject('CRICKET', 'Jangkrik Rumput', 'BED', anc.cricket.x, anc.cricket.y, anc.cricket.z, crikG, 4.5, 0.0);
      }
      if (biome.fumaroleWeight * cabinInsul > 0.05) {
        this._pushAtmosObject('FUMAROLE', 'Kawah Gunung Api', 'BED', anc.fumarole.x, anc.fumarole.y, anc.fumarole.z, biome.fumaroleWeight * 0.65 * cabinInsul, 14.0, 0.48);
      }
      if (biome.flagWeight * cabinInsul > 0.08) {
        this._pushAtmosObject('FLAG', 'Tiang Bendera UI', 'BED', anc.flag.x, anc.flag.y, anc.flag.z, biome.flagWeight * 0.45 * cabinInsul, 3.5, 0.0);
      }
      if (biome.trussAeolianWeight * cabinInsul > 0.08) {
        this._pushAtmosObject('TRUSS', 'Rangka Teksas UI', 'BED', anc.truss.x, anc.truss.y, anc.truss.z, biome.trussAeolianWeight * 0.42 * cabinInsul, 4.8, 0.0);
      }
    }

    // 6. Project onto 12-Channel Dolby Atmos 7.1.4 Bed + Height + LFE Stage
    const windNormEnv = clamp((windSpd - 0.6) / 3.2, 0.0, 1.3);
    this._atmosEnvScratch.baseEarBed = clamp(
      0.03 + windNormEnv * 0.07 + rainInt * 0.14 * zone.weatherOcclusionGain,
      0.02,
      0.40
    );
    this._atmosEnvScratch.baseHeightBed = clamp(
      0.02 + rainInt * 0.18 * zone.weatherOcclusionGain + (zone.cathedralSend + zone.metallicSend) * 0.16,
      0.01,
      0.52
    );
    this._atmosEnvScratch.baseLfe = clamp(
      zone.tunnelHum * 0.28 + this._weatherState.storm * 0.26 + (camMode === 'FPS_CAT' ? 0.12 : 0.0),
      0.0,
      0.72
    );

    const proj = AudioSynthesizer.projectAtmos714ChannelEnergies(
      this._atmosActiveObjects,
      this._atmosEnvScratch,
      this._atmosBedLevels12
    );

    const vp = this._viewPerspective;
    this._atmosStage.spatialMode = this.spatialOutputMode;
    this._atmosStage.perspectiveId = vp.perspectiveId;
    this._atmosStage.perspectiveLabel = vp.perspectiveLabel;
    this._atmosStage.coMovingEntity = vp.coMovingEntity;
    this._atmosStage.isPurePointMic = vp.isPurePointMic;
    this._atmosStage.earEnergy01 = +proj.earEnergy01.toFixed(3);
    this._atmosStage.heightEnergy01 = +proj.heightEnergy01.toFixed(3);
    this._atmosStage.lfeEnergy01 = +proj.lfeEnergy01.toFixed(3);
    this._atmosStage.objectCount = this._atmosActiveObjects.length;

    let topObj = null;
    let topGain = -1.0;
    for (let i = 0; i < this._atmosActiveObjects.length; i++) {
      const o = this._atmosActiveObjects[i];
      if (o.gain01 > topGain) {
        topGain = o.gain01;
        topObj = o;
      }
    }
    this._telemetry.active3DSources = this._atmosActiveObjects.length;
    this._telemetry.earEnergyPct = Math.round(proj.earEnergy01 * 100);
    this._telemetry.heightEnergyPct = Math.round(proj.heightEnergy01 * 100);
    this._telemetry.lfeEnergyPct = Math.round(proj.lfeEnergy01 * 100);
    this._telemetry.dominantAzDeg = topObj ? Math.round(topObj.azDeg) : 0;
    this._telemetry.dominantElDeg = topObj ? Math.round(topObj.elDeg) : 0;
  }

  _triggerNpcFootstep3D(nx, ny, nz, speedMs = 1.4, isJogger = false) {
    if (!this.ctx || !this._pinkBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const surf = this._resolveSurfaceMaterial(nx, ny, nz, 'WALK_SIVITAS');
    const pan = this.ctx.createPanner();
    pan.panningModel = this.spatialOutputMode === 'SPEAKER' ? 'equalpower' : 'HRTF';
    pan.distanceModel = 'inverse';
    pan.refDistance = 2.2;
    pan.rolloffFactor = 1.35;
    this._setPannerPosition(pan, nx, ny, nz, t0, true);
    pan.connect(this.weatherOccFilter);
    if (surf === 'TUNNEL_CONCRETE' || surf === 'BRIDGE_STEEL_WOOD') {
      pan.connect(this.reverbSendGain);
    }

    const src = this.ctx.createBufferSource();
    src.buffer = this._pinkBuffer;
    src.playbackRate.value = 1.25 + Math.random() * 0.22;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    const centerHz =
      surf === 'BRIDGE_STEEL_WOOD'
        ? 620
        : surf === 'EPDM_TRACK'
          ? 510
          : surf === 'GRASS_SOIL'
            ? 1580
            : 1080;
    bp.frequency.setValueAtTime(centerHz, t0);
    bp.Q.setValueAtTime(1.5, t0);
    const g = this.ctx.createGain();
    const peak = clamp(speedMs / 2.6, 0.35, 1.2) * (isJogger ? 0.068 : 0.048) * sfxM;
    g.gain.setValueAtTime(0.0004, t0);
    g.gain.linearRampToValueAtTime(peak, t0 + 0.007);
    g.gain.exponentialRampToValueAtTime(0.0004, t0 + 0.068);
    src.connect(bp);
    bp.connect(g);
    g.connect(pan);
    this._cleanupTransientNodes(src, bp, g, pan);
    src.start(t0);
    src.stop(t0 + 0.08);
  }

  /**
   * Iterasi 3: Real-Time World Event SFX Tracker for Sivitas NPCs (Umbrella Pop, Camera Shutter),
   * DJI Avata 2 FPV (Acro High-G Air-Rip, APAS Radar Tick), & DJI Mavic 3 Pro (Optical Zoom Servo).
   */
  _updateWorldEventSfx(dt, state = {}, sivitasAI = null, camMode = 'DRONE_6DOF', rainInt = 0.0) {
    const trk = this._sfxTracker;
    const sfxLive = Boolean(this.ctx && this.enabled && this.ctx.state === 'running');
    const lx = this._lis.x;
    const ly = this._lis.y;
    const lz = this._lis.z;

    trk.apasCooldown = Math.max(0.0, trk.apasCooldown - dt);

    // 1. DJI Avata 2 Acro Barrel-Roll / Split-S High-G Air-Rip & APAS Proximity Radar Tick
    const stuntNow = Boolean(state.avataStuntActive) && camMode === 'DRONE_AVATA2_FPV';
    if (stuntNow && !trk.prevAvataStunt) {
      this._recordSfxEvent('AVATA_ACRO_RIP', 'Manuver Acro Avata 2');
      if (sfxLive) this._triggerAvataAcroRip();
    }
    trk.prevAvataStunt = stuntNow;

    const apasNow = Boolean(state.avataApasActive) && camMode === 'DRONE_AVATA2_FPV';
    if (apasNow && !trk.prevAvataApas && trk.apasCooldown <= 0.0) {
      trk.apasCooldown = 3.6;
      this._recordSfxEvent('AVATA_APAS_TICK', 'Sensor APAS Avata 2');
      if (sfxLive) this._triggerApasRadarTick();
    }
    trk.prevAvataApas = apasNow;

    // 2. DJI Mavic 3 Pro Hasselblad Optical Zoom Servo Whisper
    if ((camMode === 'DRONE_6DOF' || camMode === 'AUTO_CINEMATIC') && Number.isFinite(state.mavicFov)) {
      const fov = Number(state.mavicFov);
      if (!trk.mavicFovInit) {
        trk.prevMavicFov = fov;
        trk.mavicFovInit = true;
      } else if (Math.abs(fov - trk.prevMavicFov) > 2.2) {
        const zoomIn = fov < trk.prevMavicFov;
        trk.prevMavicFov = fov;
        this._recordSfxEvent('MAVIC_ZOOM_SERVO', 'Servo Lensa Mavic');
        if (sfxLive) this._triggerMavicZoomWhisper(zoomIn);
      }
    }

    // 3. Sivitas Akademika NPCs: Rain Onset Umbrella Canopy Pop-Open & Landmark Camera Shutter Click
    if (sivitasAI && Array.isArray(sivitasAI.npcs) && sivitasAI.npcs.length > 0) {
      if (!trk.rainInit) {
        trk.prevRain = rainInt;
        trk.rainInit = true;
      } else {
        if (trk.prevRain < 0.15 && rainInt >= 0.15 && ly < 24.0) {
          let bestNpc = null;
          let bestD = 26.0;
          for (let i = 0; i < sivitasAI.npcs.length; i++) {
            const npc = sivitasAI.npcs[i];
            const d = Math.hypot(lx - npc.worldX, ly - (npc.worldY || 0), lz - npc.worldZ);
            if (d < bestD) {
              bestD = d;
              bestNpc = npc;
            }
          }
          if (bestNpc) {
            this._recordSfxEvent('NPC_UMBRELLA_OPEN', 'Buka Payung Sivitas');
            if (sfxLive) this._triggerUmbrellaOpen3D(bestNpc.worldX, (bestNpc.worldY || 0) + 1.65, bestNpc.worldZ);
          }
        }
        trk.prevRain = rainInt;
      }

      if (rainInt < 0.18 && ly < 16.0) {
        trk.npcShutterTimer -= dt;
        if (trk.npcShutterTimer <= 0.0) {
          trk.npcShutterTimer = 6.5 + Math.random() * 8.5;
          let photoNpc = null;
          let bestPhotoD = 16.5;
          for (let i = 0; i < sivitasAI.npcs.length; i++) {
            const npc = sivitasAI.npcs[i];
            if (npc.behavior !== 'PHOTO_LANDMARK') continue;
            const d = Math.hypot(lx - npc.worldX, ly - (npc.worldY || 0), lz - npc.worldZ);
            if (d < bestPhotoD) {
              bestPhotoD = d;
              photoNpc = npc;
            }
          }
          if (photoNpc) {
            this._recordSfxEvent('NPC_CAMERA_SHUTTER', 'Klik Kamera Sivitas');
            if (sfxLive) {
              this._triggerCameraShutter3D(photoNpc.worldX, (photoNpc.worldY || 0) + 1.48, photoNpc.worldZ);
            }
          }
        }
      }
    }

    // 4. Menara Gedung Rektorat UI Bronze Carillon Bell on Half-Hour / Full-Hour Solar Clock Crossings (06:00..21:00 WIB)
    trk.bellCooldown = Math.max(0.0, trk.bellCooldown - dt);
    if (Number.isFinite(state.solarHour)) {
      const h = (((Number(state.solarHour) % 24.0) + 24.0) % 24.0);
      const halfSlot = Math.floor(h * 2.0);
      if (trk.prevBellSlot === null) {
        trk.prevBellSlot = halfSlot;
        trk.prevSolarHourVal = h;
      } else {
        const hourJump = Math.abs(h - (trk.prevSolarHourVal ?? h));
        trk.prevSolarHourVal = h;
        if (halfSlot !== trk.prevBellSlot) {
          trk.prevBellSlot = halfSlot;
          const dTower = Math.hypot(lx, lz);
          if (hourJump < 0.60 && h >= 6.0 && h <= 21.5 && dTower < 680.0 && trk.bellCooldown <= 0.0) {
            trk.bellCooldown = 8.0;
            const isFullHour = halfSlot % 2 === 0;
            this.triggerCampusBell(isFullHour ? 3 : 1);
          }
        }
      }
    }

    // 5. Post-Rain Eave & Canopy Water Droplet Plink when Rain Clears Over Wet Campus Surfaces
    trk.eaveDripCooldown = Math.max(0.0, trk.eaveDripCooldown - dt);
    const wetness = this._weatherState.wetness;
    if (
      rainInt < 0.08 &&
      wetness > 0.28 &&
      ly > -1.2 &&
      ly < 14.0 &&
      trk.eaveDripCooldown <= 0.0 &&
      (Boolean(state.isRoofSheltered) || (this._biomeState && this._biomeState.canopyWeight > 0.32))
    ) {
      trk.eaveDripCooldown = 1.6 + Math.random() * 2.4;
      this._recordSfxEvent('POST_RAIN_EAVE_DRIP', 'Tetesan Embun Atap/Tajuk');
      if (sfxLive) {
        const az = Math.random() * Math.PI * 2.0;
        const r = 1.8 + Math.random() * 3.8;
        this._triggerPostRainEaveDrip3D(lx + Math.cos(az) * r, Math.max(0.15, ly - 0.8), lz + Math.sin(az) * r);
      }
    }

    // 6. Subsea Underpass Portal Aerodynamic Pressure Whoomp on SUBSEA_TUNNEL Entry / Exit
    trk.portalCooldown = Math.max(0.0, trk.portalCooldown - dt);
    const curZoneId = this._zoneState ? this._zoneState.zoneId : 'OPEN_AIR';
    if (!trk.prevZoneId) {
      trk.prevZoneId = curZoneId;
    } else if (curZoneId !== trk.prevZoneId) {
      const enteringTunnel = curZoneId === 'SUBSEA_TUNNEL' && trk.prevZoneId !== 'SUBSEA_TUNNEL';
      const exitingTunnel = trk.prevZoneId === 'SUBSEA_TUNNEL' && curZoneId !== 'SUBSEA_TUNNEL';
      trk.prevZoneId = curZoneId;
      if ((enteringTunnel || exitingTunnel) && trk.portalCooldown <= 0.0) {
        trk.portalCooldown = 1.5;
        this._recordSfxEvent('PORTAL_PRESSURE_WHOOMP', enteringTunnel ? 'Masuk Portal Terowongan' : 'Keluar Portal Terowongan');
        if (sfxLive) this._triggerPortalPressureWhoomp(enteringTunnel);
      }
    }
  }

  _triggerPortalPressureWhoomp(entering = true) {
    if (!this.ctx || !this._pinkBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const src = this.ctx.createBufferSource();
    src.buffer = this._pinkBuffer;
    src.playbackRate.value = entering ? 0.82 : 1.08;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.setValueAtTime(entering ? 320 : 185, t0);
    bp.frequency.exponentialRampToValueAtTime(entering ? 115 : 420, t0 + 0.36);
    bp.Q.setValueAtTime(1.35, t0);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0008, t0);
    g.gain.linearRampToValueAtTime(0.16 * sfxM, t0 + 0.06);
    g.gain.exponentialRampToValueAtTime(0.0006, t0 + 0.42);
    src.connect(bp);
    bp.connect(g);
    g.connect(this.worldSumGain);
    g.connect(this.reverbSendGain);
    if (this.lfeSumGain) g.connect(this.lfeSumGain);
    this._cleanupTransientNodes(src, bp, g);
    src.start(t0);
    src.stop(t0 + 0.45);
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

    // 3D Grounded Nadir Footstep Panner (placed beneath L/R foot along listener Right = Fwd x Up, el ≈ -72°!)
    const side = this._stepFootSide > 0 ? 1.0 : -1.0;
    const rx = this._lis.fwdY * this._lis.upZ - this._lis.fwdZ * this._lis.upY;
    const rz = this._lis.fwdX * this._lis.upY - this._lis.fwdY * this._lis.upX;
    const rLen = Math.hypot(rx, rz) || 1.0;
    const latM = side * (isCatPaw ? 0.08 : 0.17);
    const stepX = this._lis.x + (rx / rLen) * latM + this._lis.fwdX * 0.18;
    const stepY = this._lis.y - (isCatPaw ? 0.24 : 1.52);
    const stepZ = this._lis.z + (rz / rLen) * latM + this._lis.fwdZ * 0.18;

    const stepPan = this.ctx.createPanner();
    stepPan.panningModel = this.spatialOutputMode === 'SPEAKER' ? 'equalpower' : 'HRTF';
    stepPan.distanceModel = 'inverse';
    stepPan.refDistance = 1.9;
    stepPan.rolloffFactor = 0.85;
    this._setPannerPosition(stepPan, stepX, stepY, stepZ, t0, true);
    env.connect(stepPan);
    stepPan.connect(this.worldSumGain);
    if (surfaceMat === 'TUNNEL_CONCRETE' || surfaceMat === 'BRIDGE_STEEL_WOOD') {
      stepPan.connect(this.reverbSendGain);
    }
    this._cleanupTransientNodes(src, bp, env, stepPan);
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
      this._cleanupTransientNodes(osc, og);
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
      this._cleanupTransientNodes(spSrc, hp, sg);
      spSrc.start(t0);
      spSrc.stop(t0 + 0.105);
    }
  }

  _triggerLandingGearActuator3D(gx, gy, gz, retracting = true, inCockpit = false) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const pan = this.ctx.createPanner();
    pan.panningModel = this.spatialOutputMode === 'SPEAKER' ? 'equalpower' : 'HRTF';
    pan.refDistance = inCockpit ? 2.5 : 18.0;
    pan.rolloffFactor = inCockpit ? 0.85 : 1.05;
    this._setPannerPosition(pan, gx, gy, gz, t0, true);
    pan.connect(this.worldSumGain);

    let gearPartsDone = 0;
    const finishGearPan = () => {
      gearPartsDone++;
      if (gearPartsDone >= 2) {
        try {
          pan.disconnect();
        } catch (_e) {}
      }
    };

    // 1. 400 Hz Electro-Hydraulic Pump Whine (1.12s pressure load sweep)
    const pump = this.ctx.createOscillator();
    const pumpBp = this.ctx.createBiquadFilter();
    const pumpGain = this.ctx.createGain();
    pump.type = 'sawtooth';
    pump.frequency.setValueAtTime(retracting ? 385.0 : 495.0, t0);
    pump.frequency.linearRampToValueAtTime(retracting ? 515.0 : 365.0, t0 + 1.05);
    pumpBp.type = 'bandpass';
    pumpBp.frequency.setValueAtTime(860.0, t0);
    pumpBp.Q.setValueAtTime(3.2, t0);
    const wPeak = (inCockpit ? 0.085 : 0.055) * sfxM;
    pumpGain.gain.setValueAtTime(0.0005, t0);
    pumpGain.gain.linearRampToValueAtTime(wPeak, t0 + 0.08);
    pumpGain.gain.setValueAtTime(wPeak * 0.92, t0 + 0.92);
    pumpGain.gain.exponentialRampToValueAtTime(0.0005, t0 + 1.14);
    pump.connect(pumpBp);
    pumpBp.connect(pumpGain);
    pumpGain.connect(pan);
    pump.onended = () => {
      try {
        pump.disconnect();
        pumpBp.disconnect();
        pumpGain.disconnect();
      } catch (_e) {}
      finishGearPan();
    };
    pump.start(t0);
    pump.stop(t0 + 1.16);

    // 2. Mechanical Uplock / Downlock Strut Latch Clunk + LFE Airframe Thud
    const tLock = t0 + (retracting ? 1.04 : 0.98);
    const thud = this.ctx.createOscillator();
    const thudGain = this.ctx.createGain();
    thud.type = 'sine';
    thud.frequency.setValueAtTime(124.0, tLock);
    thud.frequency.exponentialRampToValueAtTime(52.0, tLock + 0.11);
    const cPeak = (inCockpit ? 0.16 : 0.09) * sfxM;
    thudGain.gain.setValueAtTime(0.0008, tLock);
    thudGain.gain.linearRampToValueAtTime(cPeak, tLock + 0.008);
    thudGain.gain.exponentialRampToValueAtTime(0.0006, tLock + 0.13);
    thud.connect(thudGain);
    thudGain.connect(pan);
    if (this.lfeSumGain) thudGain.connect(this.lfeSumGain);
    thud.onended = () => {
      try {
        thud.disconnect();
        thudGain.disconnect();
      } catch (_e) {}
      finishGearPan();
    };
    thud.start(tLock);
    thud.stop(tLock + 0.15);
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

    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.001, t0);
    g.gain.linearRampToValueAtTime(0.22 * sfxM, t0 + 0.03);
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
    this._cleanupTransientNodes(src, hp, g, pan);
    src.start(t0);
    src.stop(t0 + 0.44);
  }

  _triggerBusDoorChime3D(bx, by, bz, departing = false) {
    if (!this.ctx || !this._pinkBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 6.5;
    this._setPannerPosition(pan, bx, by, bz, t0, true);
    pan.connect(this.worldSumGain);
    pan.connect(this.reverbSendGain);

    // Pneumatic door cylinder hiss + 2-note campus transit chime (E5->G#5 on depart, G#5->E5 on stop)
    const hiss = this.ctx.createBufferSource();
    hiss.buffer = this._pinkBuffer;
    hiss.playbackRate.value = 1.95;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.setValueAtTime(2850, t0);
    bp.Q.setValueAtTime(1.4, t0);
    const hg = this.ctx.createGain();
    hg.gain.setValueAtTime(0.0008, t0);
    hg.gain.linearRampToValueAtTime(0.11 * sfxM, t0 + 0.025);
    hg.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.28);
    hiss.connect(bp);
    bp.connect(hg);
    hg.connect(pan);
    this._cleanupTransientNodes(hiss, bp, hg);
    hiss.start(t0);
    hiss.stop(t0 + 0.30);

    const freqs = departing ? [659.25, 830.61] : [830.61, 659.25];
    for (let i = 0; i < 2; i++) {
      const dt = 0.04 + i * 0.14;
      const osc = this.ctx.createOscillator();
      const cg = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freqs[i], t0 + dt);
      cg.gain.setValueAtTime(0.0005, t0 + dt);
      cg.gain.linearRampToValueAtTime(0.085 * sfxM, t0 + dt + 0.012);
      cg.gain.exponentialRampToValueAtTime(0.0005, t0 + dt + 0.24);
      osc.connect(cg);
      cg.connect(pan);
      if (i === 1) {
        this._cleanupTransientNodes(osc, cg, pan);
      } else {
        this._cleanupTransientNodes(osc, cg);
      }
      osc.start(t0 + dt);
      osc.stop(t0 + dt + 0.26);
    }
  }

  _triggerSuspensionThump3D(sx, sy, sz) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 6.0;
    this._setPannerPosition(pan, sx, sy, sz, t0, true);
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(92, t0);
    osc.frequency.exponentialRampToValueAtTime(42, t0 + 0.14);
    g.gain.setValueAtTime(0.001, t0);
    g.gain.linearRampToValueAtTime(0.16 * sfxM, t0 + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.16);
    osc.connect(g);
    g.connect(pan);
    pan.connect(this.worldSumGain);
    pan.connect(this.reverbSendGain);
    this._cleanupTransientNodes(osc, g, pan);
    osc.start(t0);
    osc.stop(t0 + 0.18);
  }

  _triggerCourtesyBell3D(cx, cy, cz) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 5.5;
    this._setPannerPosition(pan, cx, cy, cz, t0, true);
    pan.connect(this.worldSumGain);
    pan.connect(this.reverbSendGain);

    // Friendly brass double courtesy bell ("ting-ting", fundamental + inharmonic bell partial)
    for (let k = 0; k < 2; k++) {
      const dt = k * 0.135;
      const osc1 = this.ctx.createOscillator();
      const osc2 = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc1.type = 'sine';
      osc2.type = 'sine';
      osc1.frequency.setValueAtTime(1760, t0 + dt);
      osc2.frequency.setValueAtTime(1760 * 2.42, t0 + dt);
      g.gain.setValueAtTime(0.0005, t0 + dt);
      g.gain.linearRampToValueAtTime(0.075 * sfxM, t0 + dt + 0.005);
      g.gain.exponentialRampToValueAtTime(0.0005, t0 + dt + 0.22);
      osc1.connect(g);
      osc2.connect(g);
      g.connect(pan);
      this._cleanupTransientNodes(osc1);
      if (k === 1) {
        this._cleanupTransientNodes(osc2, g, pan);
      } else {
        this._cleanupTransientNodes(osc2, g);
      }
      osc1.start(t0 + dt);
      osc2.start(t0 + dt);
      osc1.stop(t0 + dt + 0.24);
      osc2.stop(t0 + dt + 0.24);
    }
  }

  _triggerMarineHorn3D(mx, my, mz, isDeparture = true) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 22.0;
    pan.rolloffFactor = 0.85;
    this._setPannerPosition(pan, mx, my, mz, t0, true);
    pan.connect(this.worldSumGain);
    pan.connect(this.reverbSendGain);

    // Deep twin-tone marine catamaran horn (C#3 138.6 Hz + F3 174.6 Hz)
    const dur = isDeparture ? 1.15 : 0.78;
    const oscA = this.ctx.createOscillator();
    const oscB = this.ctx.createOscillator();
    const lp = this.ctx.createBiquadFilter();
    const g = this.ctx.createGain();
    oscA.type = 'sawtooth';
    oscB.type = 'sawtooth';
    oscA.frequency.setValueAtTime(138.59, t0);
    oscB.frequency.setValueAtTime(174.61, t0);
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(520, t0);
    g.gain.setValueAtTime(0.0008, t0);
    g.gain.linearRampToValueAtTime(0.16 * sfxM, t0 + 0.08);
    g.gain.setValueAtTime(0.15 * sfxM, t0 + dur - 0.16);
    g.gain.exponentialRampToValueAtTime(0.0008, t0 + dur);
    oscA.connect(lp);
    oscB.connect(lp);
    lp.connect(g);
    g.connect(pan);
    this._cleanupTransientNodes(oscB);
    this._cleanupTransientNodes(oscA, lp, g, pan);
    oscA.start(t0);
    oscB.start(t0);
    oscA.stop(t0 + dur + 0.03);
    oscB.stop(t0 + dur + 0.03);
  }

  _triggerRunwayTouchdownChirp3D(ax, ay, az, isJet = false) {
    if (!this.ctx || !this._pinkBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 24.0;
    pan.rolloffFactor = 0.85;
    this._setPannerPosition(pan, ax, ay, az, t0, true);
    pan.connect(this.worldSumGain);
    pan.connect(this.reverbSendGain);

    // Rubber tire spin-up chirp + oleo strut touchdown thump
    const chirpOsc = this.ctx.createOscillator();
    const cg = this.ctx.createGain();
    chirpOsc.type = 'sawtooth';
    const fStart = isJet ? 1420 : 1180;
    chirpOsc.frequency.setValueAtTime(fStart, t0);
    chirpOsc.frequency.exponentialRampToValueAtTime(fStart * 0.52, t0 + 0.15);
    cg.gain.setValueAtTime(0.001, t0);
    cg.gain.linearRampToValueAtTime((isJet ? 0.18 : 0.13) * sfxM, t0 + 0.015);
    cg.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.17);

    const noise = this.ctx.createBufferSource();
    noise.buffer = this._pinkBuffer;
    noise.playbackRate.value = 1.75;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.setValueAtTime(1650, t0);
    bp.Q.setValueAtTime(2.2, t0);
    const ng = this.ctx.createGain();
    ng.gain.setValueAtTime(0.001, t0);
    ng.gain.linearRampToValueAtTime(0.19 * sfxM, t0 + 0.012);
    ng.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.19);

    chirpOsc.connect(bp);
    noise.connect(bp);
    bp.connect(ng);
    chirpOsc.connect(cg);
    cg.connect(pan);
    ng.connect(pan);
    this._cleanupTransientNodes(chirpOsc, cg);
    this._cleanupTransientNodes(noise, bp, ng, pan);
    chirpOsc.start(t0);
    chirpOsc.stop(t0 + 0.18);
    noise.start(t0);
    noise.stop(t0 + 0.21);
  }

  _triggerThrustReverserDeploy3D(jx, jy, jz) {
    if (!this.ctx || !this._pinkBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 28.0;
    pan.rolloffFactor = 0.80;
    this._setPannerPosition(pan, jx, jy, jz, t0, true);
    pan.connect(this.worldSumGain);
    pan.connect(this.reverbSendGain);

    const src = this.ctx.createBufferSource();
    src.buffer = this._pinkBuffer;
    src.playbackRate.value = 0.92;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.setValueAtTime(280, t0);
    bp.frequency.linearRampToValueAtTime(720, t0 + 0.35);
    bp.frequency.exponentialRampToValueAtTime(340, t0 + 1.45);
    bp.Q.setValueAtTime(0.75, t0);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.001, t0);
    g.gain.linearRampToValueAtTime(0.28 * sfxM, t0 + 0.22);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + 1.55);
    src.connect(bp);
    bp.connect(g);
    g.connect(pan);
    this._cleanupTransientNodes(src, bp, g, pan);
    src.start(t0);
    src.stop(t0 + 1.60);
  }

  _triggerTakeoffSpoolSurge3D(ax, ay, az, isJet = false) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 24.0;
    pan.rolloffFactor = 0.82;
    this._setPannerPosition(pan, ax, ay, az, t0, true);
    pan.connect(this.worldSumGain);

    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = isJet ? 'sine' : 'sawtooth';
    osc.frequency.setValueAtTime(isJet ? 620 : 84, t0);
    osc.frequency.exponentialRampToValueAtTime(isJet ? 1540 : 148, t0 + 0.85);
    g.gain.setValueAtTime(0.001, t0);
    g.gain.linearRampToValueAtTime(0.14 * sfxM, t0 + 0.25);
    g.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.95);
    osc.connect(g);
    g.connect(pan);
    this._cleanupTransientNodes(osc, g, pan);
    osc.start(t0);
    osc.stop(t0 + 0.98);
  }

  _triggerHeliSkidTouchdown3D(hx, hy, hz) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 10.0;
    this._setPannerPosition(pan, hx, hy, hz, t0, true);
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(124, t0);
    osc.frequency.exponentialRampToValueAtTime(48, t0 + 0.13);
    g.gain.setValueAtTime(0.001, t0);
    g.gain.linearRampToValueAtTime(0.16 * sfxM, t0 + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.15);
    osc.connect(g);
    g.connect(pan);
    pan.connect(this.worldSumGain);
    this._cleanupTransientNodes(osc, g, pan);
    osc.start(t0);
    osc.stop(t0 + 0.17);
  }

  _triggerCameraShutter3D(px, py, pz) {
    if (!this.ctx || !this._patterBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 4.0;
    this._setPannerPosition(pan, px, py, pz, t0, true);
    pan.connect(this.weatherOccFilter);

    // Crisp mechanical double-curtain shutter click ("k-chik")
    for (let k = 0; k < 2; k++) {
      const dt = k * 0.038;
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(k === 0 ? 3200 : 4150, t0 + dt);
      g.gain.setValueAtTime(0.0005, t0 + dt);
      g.gain.linearRampToValueAtTime(0.042 * sfxM, t0 + dt + 0.003);
      g.gain.exponentialRampToValueAtTime(0.0005, t0 + dt + 0.028);
      osc.connect(g);
      g.connect(pan);
      if (k === 1) {
        this._cleanupTransientNodes(osc, g, pan);
      } else {
        this._cleanupTransientNodes(osc, g);
      }
      osc.start(t0 + dt);
      osc.stop(t0 + dt + 0.034);
    }
  }

  _triggerUmbrellaOpen3D(ux, uy, uz) {
    if (!this.ctx || !this._pinkBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 5.5;
    this._setPannerPosition(pan, ux, uy, uz, t0, true);
    pan.connect(this.weatherOccFilter);

    // Taut nylon canopy pop-open whoosh + latch click
    const src = this.ctx.createBufferSource();
    src.buffer = this._pinkBuffer;
    src.playbackRate.value = 1.65;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.setValueAtTime(680, t0);
    bp.frequency.exponentialRampToValueAtTime(2200, t0 + 0.14);
    bp.Q.setValueAtTime(1.5, t0);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0008, t0);
    g.gain.linearRampToValueAtTime(0.095 * sfxM, t0 + 0.05);
    g.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.19);
    src.connect(bp);
    bp.connect(g);
    g.connect(pan);
    this._cleanupTransientNodes(src, bp, g, pan);
    src.start(t0);
    src.stop(t0 + 0.21);
  }

  /**
   * Public & automatic trigger for Menara Gedung Rektorat UI Bronze Carillon Bell at (0, 34, 0)
   */
  triggerCampusBell(noteCount = 2) {
    const count = clamp(Math.round(Number(noteCount) || 2), 1, 4);
    this._recordSfxEvent('REKTORAT_CARILLON_BELL', 'Lonceng Menara Rektorat UI');
    if (this.enabled && this.ctx && this.ctx.state === 'running') {
      this._triggerRektoratCarillon3D(count);
    }
  }

  _triggerRektoratCarillon3D(noteCount = 2) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const bx = 0.0;
    const by = 34.0;
    const bz = 0.0;
    const occ = AudioSynthesizer.computeGeometricOcclusion(
      bx,
      by,
      bz,
      this._lis.x,
      this._lis.y,
      this._lis.z
    );
    const doppAbs = AudioSynthesizer.computeDopplerAndAbsorption(
      bx,
      by,
      bz,
      0.0,
      0.0,
      0.0,
      this._lis.x,
      this._lis.y,
      this._lis.z,
      this._lis.vx,
      this._lis.vy,
      this._lis.vz,
      this._weatherState.tempC,
      this._weatherState.humidityPct
    );

    const pan = this.ctx.createPanner();
    pan.panningModel = this.spatialOutputMode === 'SPEAKER' ? 'equalpower' : 'HRTF';
    pan.distanceModel = 'inverse';
    pan.refDistance = 42.0;
    pan.maxDistance = 1400.0;
    pan.rolloffFactor = 0.72;
    this._setPannerPosition(pan, bx, by, bz, t0, true);

    const lp = this.ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(clamp(doppAbs.airCutoffHz * occ.cutoffMult, 240.0, 16000.0), t0);
    lp.connect(pan);
    pan.connect(this.worldSumGain);
    pan.connect(this.reverbSendGain);

    // Nusantara Bronze Carillon Motif (E4 - G4 - B4 - E5) with authentic cast-bronze partials (1 : 1.518 : 2.414 : 3.86)
    const motifHz = [329.63, 392.0, 493.88, 659.25];
    const partials = [
      { ratio: 1.0, amp: 1.0, decay: 2.35 },
      { ratio: 1.518, amp: 0.56, decay: 1.85 },
      { ratio: 2.004, amp: 0.42, decay: 1.55 },
      { ratio: 2.414, amp: 0.28, decay: 1.15 },
      { ratio: 3.86, amp: 0.16, decay: 0.68 }
    ];
    const totalNotes = clamp(noteCount, 1, 4);
    for (let n = 0; n < totalNotes; n++) {
      const noteStart = t0 + n * 0.62;
      const f0 = motifHz[n % motifHz.length];
      const basePeak = 0.088 * sfxM * occ.gainMult;

      for (let p = 0; p < partials.length; p++) {
        const pt = partials[p];
        const osc = this.ctx.createOscillator();
        const g = this.ctx.createGain();
        osc.type = p === 0 ? 'triangle' : 'sine';
        osc.frequency.setValueAtTime(f0 * pt.ratio, noteStart);
        g.gain.setValueAtTime(0.0004, noteStart);
        g.gain.linearRampToValueAtTime(basePeak * pt.amp, noteStart + 0.008);
        g.gain.exponentialRampToValueAtTime(0.0004, noteStart + pt.decay);
        osc.connect(g);
        g.connect(lp);
        if (n === totalNotes - 1 && p === 0) {
          this._cleanupTransientNodes(osc, g, lp, pan);
        } else {
          this._cleanupTransientNodes(osc, g);
        }
        osc.start(noteStart);
        osc.stop(noteStart + pt.decay + 0.04);
      }
    }
  }

  _triggerPostRainEaveDrip3D(dx, dy, dz) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const pan = this.ctx.createPanner();
    pan.panningModel = this.spatialOutputMode === 'SPEAKER' ? 'equalpower' : 'HRTF';
    pan.refDistance = 3.2;
    pan.rolloffFactor = 1.15;
    this._setPannerPosition(pan, dx, dy, dz, t0, true);
    pan.connect(this.weatherOccFilter);
    pan.connect(this.reverbSendGain);

    // Minnaert air-cavity water droplet upward pitch-sweep plink (f0 = 1580..1950 Hz -> 1.48x in 34 ms)
    const fStart = 1580 + Math.random() * 380;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(fStart, t0);
    osc.frequency.exponentialRampToValueAtTime(fStart * 1.48, t0 + 0.036);
    g.gain.setValueAtTime(0.0004, t0);
    g.gain.linearRampToValueAtTime(0.052 * sfxM, t0 + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0004, t0 + 0.046);
    osc.connect(g);
    g.connect(pan);
    this._cleanupTransientNodes(osc, g, pan);
    osc.start(t0);
    osc.stop(t0 + 0.055);
  }

  _triggerHullWaveSlap3D(bx, by, bz, waveM = 1.0) {
    if (!this.ctx || !this._pinkBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const waveScale = clamp((waveM - 0.65) / 2.35, 0.28, 1.25);
    const pan = this.ctx.createPanner();
    pan.panningModel = this.spatialOutputMode === 'SPEAKER' ? 'equalpower' : 'HRTF';
    pan.refDistance = 12.0;
    pan.rolloffFactor = 0.95;
    this._setPannerPosition(pan, bx, by, bz, t0, true);
    pan.connect(this.weatherOccFilter);
    pan.connect(this.reverbSendGain);

    // 1. Hollow hull-chine impact thump (96 Hz -> 48 Hz)
    const thud = this.ctx.createOscillator();
    const tg = this.ctx.createGain();
    thud.type = 'triangle';
    thud.frequency.setValueAtTime(96, t0);
    thud.frequency.exponentialRampToValueAtTime(48, t0 + 0.16);
    tg.gain.setValueAtTime(0.0008, t0);
    tg.gain.linearRampToValueAtTime(0.14 * waveScale * sfxM, t0 + 0.014);
    tg.gain.exponentialRampToValueAtTime(0.0006, t0 + 0.19);
    thud.connect(tg);
    tg.connect(pan);
    this._cleanupTransientNodes(thud, tg);
    thud.start(t0);
    thud.stop(t0 + 0.21);

    // 2. Whitecap spray splash burst (bandpass 680 -> 260 Hz)
    const spray = this.ctx.createBufferSource();
    spray.buffer = this._patterBuffer || this._pinkBuffer;
    spray.playbackRate.value = 1.18 + Math.random() * 0.22;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.setValueAtTime(680, t0);
    bp.frequency.exponentialRampToValueAtTime(260, t0 + 0.34);
    bp.Q.setValueAtTime(1.15, t0);
    const sg = this.ctx.createGain();
    sg.gain.setValueAtTime(0.0008, t0);
    sg.gain.linearRampToValueAtTime(0.11 * waveScale * sfxM, t0 + 0.028);
    sg.gain.exponentialRampToValueAtTime(0.0006, t0 + 0.38);
    spray.connect(bp);
    bp.connect(sg);
    sg.connect(pan);
    this._cleanupTransientNodes(spray, bp, sg, pan);
    spray.start(t0);
    spray.stop(t0 + 0.41);
  }

  _triggerAvataAcroRip() {
    if (!this.ctx || !this._pinkBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;

    // High-G 5-blade prop-vortex rip + motor surge during Barrel Roll / Split-S
    const src = this.ctx.createBufferSource();
    src.buffer = this._pinkBuffer;
    src.playbackRate.value = 1.55;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.setValueAtTime(740, t0);
    bp.frequency.exponentialRampToValueAtTime(1950, t0 + 0.22);
    bp.frequency.exponentialRampToValueAtTime(620, t0 + 0.58);
    bp.Q.setValueAtTime(1.85, t0);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.001, t0);
    g.gain.linearRampToValueAtTime(0.18 * sfxM, t0 + 0.12);
    g.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.62);
    src.connect(bp);
    bp.connect(g);
    g.connect(this.worldSumGain);
    g.connect(this.reverbSendGain);
    this._cleanupTransientNodes(src, bp, g);
    src.start(t0);
    src.stop(t0 + 0.65);
  }

  _triggerApasRadarTick() {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(1920, t0);
    osc.frequency.exponentialRampToValueAtTime(2360, t0 + 0.045);
    g.gain.setValueAtTime(0.0005, t0);
    g.gain.linearRampToValueAtTime(0.036 * sfxM, t0 + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0005, t0 + 0.065);
    osc.connect(g);
    g.connect(this.worldSumGain);
    this._cleanupTransientNodes(osc, g);
    osc.start(t0);
    osc.stop(t0 + 0.075);
  }

  _triggerMavicZoomWhisper(zoomIn = true) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const sfxM = this._mixWeights ? this._mixWeights.sfxMult : 1.0;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(zoomIn ? 880 : 1180, t0);
    osc.frequency.exponentialRampToValueAtTime(zoomIn ? 1240 : 840, t0 + 0.09);
    g.gain.setValueAtTime(0.0005, t0);
    g.gain.linearRampToValueAtTime(0.024 * sfxM, t0 + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0005, t0 + 0.10);
    osc.connect(g);
    g.connect(this.worldSumGain);
    this._cleanupTransientNodes(osc, g);
    osc.start(t0);
    osc.stop(t0 + 0.11);
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
    if (zoneId === 'SUBSEA_TUNNEL' || ly > 42.0) return;

    const phaseId = this._diurnalState ? this._diurnalState.phaseId : 'DAY';
    const biomeId = this._biomeState ? this._biomeState.biomeId : 'TROPICAL_RAINFOREST';

    // Enclosed pressurized/glazed cockpits & cabins insulate outdoor birds, frogs, and plaza clinks
    if (zoneId !== 'COCKPIT_INTERIOR') {
      // B1. DAWN (Fajar): Distant Nusantara Rooster Crow (Kokok Ayam Fajar) + Perkutut Wood-Dove
      if (phaseId === 'DAWN' && rainInt < 0.45) {
        this._roosterTimer -= dt;
        if (this._roosterTimer <= 0.0) {
          this._roosterTimer = 9.5 + Math.random() * 12.0;
          const az = Math.random() * Math.PI * 2.0;
          const r = 22.0 + Math.random() * 24.0;
          this._triggerDawnRooster3D(lx + Math.cos(az) * r, Math.max(1.2, ly), lz + Math.sin(az) * r);
        }
      }

      // B2. Daytime / Morning / Afternoon Tropical Songbirds (Kutilang FM Syrinx + Perkutut/Tekukur Coo)
      if (phaseId !== 'NIGHT' && rainInt < 0.45) {
        const birdRate = this._diurnalState ? this._diurnalState.birdWeight : 0.7;
        this._birdTimer -= dt * (0.55 + 0.75 * birdRate);
        if (this._birdTimer <= 0.0) {
          this._birdTimer = 2.6 + Math.random() * 4.8;
          const az = Math.random() * Math.PI * 2.0;
          const r = 11.0 + Math.random() * 20.0;
          this._triggerSongbirdChirp3D(
            lx + Math.cos(az) * r,
            Math.max(3.5, ly + 2.5 + Math.random() * 6.0),
            lz + Math.sin(az) * r
          );
        }

        this._doveTimer -= dt * (0.45 + 0.65 * birdRate);
        if (this._doveTimer <= 0.0) {
          this._doveTimer = 6.8 + Math.random() * 9.5;
          const az = Math.random() * Math.PI * 2.0;
          const r = 13.0 + Math.random() * 22.0;
          this._triggerDoveCoo3D(
            lx + Math.cos(az) * r,
            Math.max(4.0, ly + 3.0 + Math.random() * 5.0),
            lz + Math.sin(az) * r
          );
        }
      }

      // B3. NIGHT (Malam): Nocturnal Burung Hantu (Owl) Hoot & Tokek Canopy Call
      if (phaseId === 'NIGHT' && rainInt < 0.55) {
        this._nightOwlTimer -= dt;
        if (this._nightOwlTimer <= 0.0) {
          this._nightOwlTimer = 8.5 + Math.random() * 13.0;
          const az = Math.random() * Math.PI * 2.0;
          const r = 15.0 + Math.random() * 24.0;
          this._triggerNightOwlOrTokek3D(
            lx + Math.cos(az) * r,
            Math.max(4.5, ly + 3.5 + Math.random() * 5.5),
            lz + Math.sin(az) * r
          );
        }
      }

      // C. Dusk / Night Lakeside Frogs near Danau Kenanga, Danau Kecil, or Pulau U River
      if ((phaseId === 'NIGHT' || phaseId === 'DUSK' || nightFac > 0.45) && rainInt < 0.95) {
        this._frogTimer -= dt;
        if (this._frogTimer <= 0.0) {
          this._frogTimer = 3.2 + Math.random() * 5.4;
          const nearLake =
            biomeId === 'DANAU_TURAP_PROMENADE' ||
            biomeId === 'RIVER_SPRING_RIFFLE' ||
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

      // D. Location-Context Micro-Biome Signature Events (Flag Halyard, Dermaga Fender, Gym Bar, Kiosk Cup)
      this._contextMicroTimer -= dt;
      if (this._contextMicroTimer <= 0.0) {
        this._contextMicroTimer = 4.2 + Math.random() * 5.8;
        if (biomeId === 'FLAG_CLOISTER') {
          const fx = -2.5 + (Math.random() * 2.0 - 1.0) * 8.0;
          const fz = 52.0 + (Math.random() * 2.0 - 1.0) * 6.0;
          this._triggerFlagHalyardClink3D(fx, 4.8, fz);
        } else if (biomeId === 'DERMAGA_HARBOUR') {
          const isNorthPier = Math.hypot(lx - -98.0, lz - 595.0) < Math.hypot(lx - -48.0, lz - 480.0);
          this._triggerHarbourFenderCreak3D(isNorthPier ? -98.0 : -48.0, 0.3, isNorthPier ? 582.0 : 492.0);
        } else if (biomeId === 'SPORTS_TRACK_TELUK' && phaseId !== 'NIGHT' && rainInt < 0.35) {
          this._triggerGymBarClink3D(-62.0 + (Math.random() - 0.5) * 8.0, 1.9, 136.5 + (Math.random() - 0.5) * 5.0);
        } else if (biomeId === 'CAMPUS_KIOSK_BENCH' && phaseId !== 'NIGHT' && rainInt < 0.45) {
          this._triggerKioskCupClink3D(lx + (Math.random() - 0.5) * 5.0, 1.2, lz + (Math.random() - 0.5) * 5.0);
        }
      }
    }

    // E. Bandara Pulau I ATC VHF Radio Squelch Burst (Near ATC Tower/Apron or inside FPS_PILOT)
    if (biomeId === 'AIRPORT_ATC_APRON' || camMode === 'FPS_PILOT') {
      this._atcSquelchTimer -= dt;
      if (this._atcSquelchTimer <= 0.0) {
        this._atcSquelchTimer = 7.5 + Math.random() * 9.5;
        const rx = camMode === 'FPS_PILOT' ? lx + this._lis.fwdX * 0.8 : -222.5;
        const ry = camMode === 'FPS_PILOT' ? ly - 0.15 : 13.5;
        const rz = camMode === 'FPS_PILOT' ? lz + this._lis.fwdZ * 0.8 : 665.5;
        this._triggerAtcRadioSquelch3D(rx, ry, rz, camMode === 'FPS_PILOT');
      }
    }

    // F. Athletic Breathing Cadence during Active Jogging in FPS_JOGGER
    if (camMode === 'FPS_JOGGER') {
      const runSpd = Math.hypot(this._lis.vx, this._lis.vz);
      if (runSpd > 2.3) {
        this._jogBreathAccum += dt * clamp(runSpd / 3.4, 0.75, 1.35);
        if (this._jogBreathAccum >= 1.85) {
          this._jogBreathAccum = 0.0;
          this._triggerJoggerBreathGrain(clamp(runSpd / 4.8, 0.4, 1.0));
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
      if (n === notes - 1) {
        this._cleanupTransientNodes(osc, g, pan);
      } else {
        this._cleanupTransientNodes(osc, g);
      }
      osc.start(cursor);
      osc.stop(cursor + dur + 0.01);
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
    this._cleanupTransientNodes(osc, bp, g, pan);
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
    this._cleanupTransientNodes(osc, bp, g, pan);
    osc.start(t0);
    osc.stop(t0 + 0.38);
  }

  _triggerDoveCoo3D(dx, dy, dz) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 7.5;
    this._setPannerPosition(pan, dx, dy, dz, t0, true);
    pan.connect(this.weatherOccFilter);
    pan.connect(this.reverbSendGain);

    // Warm Perkutut / Tekukur 3-note hollow coo ("kur-ru-kuk")
    const notes = [
      { dt: 0.0, dur: 0.13, f0: 560, f1: 635 },
      { dt: 0.17, dur: 0.11, f0: 615, f1: 545 },
      { dt: 0.32, dur: 0.22, f0: 585, f1: 495 }
    ];
    for (let i = 0; i < notes.length; i++) {
      const n = notes[i];
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(n.f0, t0 + n.dt);
      osc.frequency.exponentialRampToValueAtTime(n.f1, t0 + n.dt + n.dur);
      g.gain.setValueAtTime(0.0005, t0 + n.dt);
      g.gain.linearRampToValueAtTime(0.048, t0 + n.dt + 0.025);
      g.gain.exponentialRampToValueAtTime(0.0005, t0 + n.dt + n.dur);
      osc.connect(g);
      g.connect(pan);
      if (i === notes.length - 1) {
        this._cleanupTransientNodes(osc, g, pan);
      } else {
        this._cleanupTransientNodes(osc, g);
      }
      osc.start(t0 + n.dt);
      osc.stop(t0 + n.dt + n.dur + 0.015);
    }
  }

  _triggerDawnRooster3D(rx, ry, rz) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 12.0;
    pan.rolloffFactor = 1.1;
    this._setPannerPosition(pan, rx, ry, rz, t0, true);
    pan.connect(this.weatherOccFilter);
    pan.connect(this.reverbSendGain);

    // Distant Nusantara "ku-ku-ru-yuuk" 4-syllable formant melody
    const syl = [
      { dt: 0.0, dur: 0.10, f0: 660, f1: 720 },
      { dt: 0.13, dur: 0.10, f0: 710, f1: 780 },
      { dt: 0.26, dur: 0.12, f0: 770, f1: 880 },
      { dt: 0.41, dur: 0.46, f0: 870, f1: 590 }
    ];
    for (let i = 0; i < syl.length; i++) {
      const s = syl[i];
      const osc = this.ctx.createOscillator();
      const bp = this.ctx.createBiquadFilter();
      const g = this.ctx.createGain();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(s.f0, t0 + s.dt);
      osc.frequency.exponentialRampToValueAtTime(s.f1, t0 + s.dt + s.dur);
      bp.type = 'bandpass';
      bp.frequency.setValueAtTime(1320, t0 + s.dt);
      bp.Q.setValueAtTime(2.8, t0 + s.dt);
      g.gain.setValueAtTime(0.0005, t0 + s.dt);
      g.gain.linearRampToValueAtTime(0.042, t0 + s.dt + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0005, t0 + s.dt + s.dur);
      osc.connect(bp);
      bp.connect(g);
      g.connect(pan);
      if (i === syl.length - 1) {
        this._cleanupTransientNodes(osc, bp, g, pan);
      } else {
        this._cleanupTransientNodes(osc, bp, g);
      }
      osc.start(t0 + s.dt);
      osc.stop(t0 + s.dt + s.dur + 0.02);
    }
  }

  _triggerNightOwlOrTokek3D(ox, oy, oz, forceTokek = undefined) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const pan = this.ctx.createPanner();
    pan.panningModel = this.spatialOutputMode === 'SPEAKER' ? 'equalpower' : 'HRTF';
    pan.refDistance = 9.5;
    this._setPannerPosition(pan, ox, oy, oz, t0, true);
    pan.connect(this.weatherOccFilter);
    pan.connect(this.reverbSendGain);

    const useTokek =
      forceTokek !== undefined ? Boolean(forceTokek) : (this._tokekToggle = !this._tokekToggle);

    if (useTokek) {
      // Authentic Nusantara Tokek ("tok-keee-ek!") 2-syllable vocal call repeated twice
      const reps = 2;
      for (let r = 0; r < reps; r++) {
        const baseT = t0 + r * 0.56;
        const atten = r === 0 ? 1.0 : 0.76;

        // Syllable 1: Crisp glottal click "tok"
        const clickOsc = this.ctx.createOscillator();
        const clickGain = this.ctx.createGain();
        clickOsc.type = 'triangle';
        clickOsc.frequency.setValueAtTime(980, baseT);
        clickOsc.frequency.exponentialRampToValueAtTime(520, baseT + 0.042);
        clickGain.gain.setValueAtTime(0.0005, baseT);
        clickGain.gain.linearRampToValueAtTime(0.046 * atten, baseT + 0.006);
        clickGain.gain.exponentialRampToValueAtTime(0.0005, baseT + 0.048);
        clickOsc.connect(clickGain);
        clickGain.connect(pan);
        this._cleanupTransientNodes(clickOsc, clickGain);
        clickOsc.start(baseT);
        clickOsc.stop(baseT + 0.055);

        // Syllable 2: Nasal formant bark "keee-ek"
        const barkT = baseT + 0.11;
        const barkOsc = this.ctx.createOscillator();
        const barkBp = this.ctx.createBiquadFilter();
        const barkGain = this.ctx.createGain();
        barkOsc.type = 'sawtooth';
        barkOsc.frequency.setValueAtTime(610, barkT);
        barkOsc.frequency.exponentialRampToValueAtTime(435, barkT + 0.22);
        barkBp.type = 'bandpass';
        barkBp.frequency.setValueAtTime(1480, barkT);
        barkBp.frequency.exponentialRampToValueAtTime(980, barkT + 0.22);
        barkBp.Q.setValueAtTime(2.6, barkT);
        barkGain.gain.setValueAtTime(0.0005, barkT);
        barkGain.gain.linearRampToValueAtTime(0.052 * atten, barkT + 0.024);
        barkGain.gain.exponentialRampToValueAtTime(0.0005, barkT + 0.24);
        barkOsc.connect(barkBp);
        barkBp.connect(barkGain);
        barkGain.connect(pan);
        if (r === reps - 1) {
          this._cleanupTransientNodes(barkOsc, barkBp, barkGain, pan);
        } else {
          this._cleanupTransientNodes(barkOsc, barkBp, barkGain);
        }
        barkOsc.start(barkT);
        barkOsc.stop(barkT + 0.26);
      }
      return;
    }

    // Hollow tropical owl double-hoot ("hoo... hoo-oo")
    const hoots = [
      { dt: 0.0, dur: 0.24, f0: 410, f1: 375 },
      { dt: 0.36, dur: 0.34, f0: 392, f1: 342 }
    ];
    for (let i = 0; i < hoots.length; i++) {
      const h = hoots[i];
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(h.f0, t0 + h.dt);
      osc.frequency.exponentialRampToValueAtTime(h.f1, t0 + h.dt + h.dur);
      g.gain.setValueAtTime(0.0005, t0 + h.dt);
      g.gain.linearRampToValueAtTime(0.045, t0 + h.dt + 0.035);
      g.gain.exponentialRampToValueAtTime(0.0005, t0 + h.dt + h.dur);
      osc.connect(g);
      g.connect(pan);
      if (i === hoots.length - 1) {
        this._cleanupTransientNodes(osc, g, pan);
      } else {
        this._cleanupTransientNodes(osc, g);
      }
      osc.start(t0 + h.dt);
      osc.stop(t0 + h.dt + h.dur + 0.02);
    }
  }

  _triggerFlagHalyardClink3D(fx, fy, fz) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 5.0;
    this._setPannerPosition(pan, fx, fy, fz, t0, true);
    pan.connect(this.weatherOccFilter);

    // Metallic bronze halyard clip tapping hollow aluminum flagpole (two quick taps)
    for (let k = 0; k < 2; k++) {
      const dt = k * (0.11 + Math.random() * 0.06);
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(2480 + k * 340 + Math.random() * 180, t0 + dt);
      g.gain.setValueAtTime(0.0005, t0 + dt);
      g.gain.linearRampToValueAtTime(0.038, t0 + dt + 0.005);
      g.gain.exponentialRampToValueAtTime(0.0005, t0 + dt + 0.085);
      osc.connect(g);
      g.connect(pan);
      if (k === 1) {
        this._cleanupTransientNodes(osc, g, pan);
      } else {
        this._cleanupTransientNodes(osc, g);
      }
      osc.start(t0 + dt);
      osc.stop(t0 + dt + 0.095);
    }
  }

  _triggerHarbourFenderCreak3D(hx, hy, hz) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const bp = this.ctx.createBiquadFilter();
    const g = this.ctx.createGain();
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 7.5;
    this._setPannerPosition(pan, hx, hy, hz, t0, true);

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(118, t0);
    osc.frequency.linearRampToValueAtTime(164, t0 + 0.22);
    osc.frequency.linearRampToValueAtTime(104, t0 + 0.48);
    bp.type = 'bandpass';
    bp.frequency.setValueAtTime(420, t0);
    bp.Q.setValueAtTime(3.6, t0);
    g.gain.setValueAtTime(0.0008, t0);
    g.gain.linearRampToValueAtTime(0.055, t0 + 0.08);
    g.gain.exponentialRampToValueAtTime(0.0008, t0 + 0.50);

    osc.connect(bp);
    bp.connect(g);
    g.connect(pan);
    pan.connect(this.weatherOccFilter);
    this._cleanupTransientNodes(osc, bp, g, pan);
    osc.start(t0);
    osc.stop(t0 + 0.52);
  }

  _triggerGymBarClink3D(gx, gy, gz) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 5.5;
    this._setPannerPosition(pan, gx, gy, gz, t0, true);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(1680 + Math.random() * 220, t0);
    g.gain.setValueAtTime(0.0005, t0);
    g.gain.linearRampToValueAtTime(0.036, t0 + 0.005);
    g.gain.exponentialRampToValueAtTime(0.0005, t0 + 0.14);
    osc.connect(g);
    g.connect(pan);
    pan.connect(this.weatherOccFilter);
    this._cleanupTransientNodes(osc, g, pan);
    osc.start(t0);
    osc.stop(t0 + 0.15);
  }

  _triggerKioskCupClink3D(kx, ky, kz) {
    if (!this.ctx || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = 4.5;
    this._setPannerPosition(pan, kx, ky, kz, t0, true);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(2940 + Math.random() * 380, t0);
    g.gain.setValueAtTime(0.0005, t0);
    g.gain.linearRampToValueAtTime(0.032, t0 + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0005, t0 + 0.075);
    osc.connect(g);
    g.connect(pan);
    pan.connect(this.weatherOccFilter);
    this._cleanupTransientNodes(osc, g, pan);
    osc.start(t0);
    osc.stop(t0 + 0.085);
  }

  _triggerAtcRadioSquelch3D(rx, ry, rz, inCockpit = false) {
    if (!this.ctx || !this._pinkBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const pan = this.ctx.createPanner();
    pan.panningModel = 'HRTF';
    pan.refDistance = inCockpit ? 2.0 : 12.0;
    this._setPannerPosition(pan, rx, ry, rz, t0, true);
    pan.connect(this.worldSumGain);

    // VHF Aviation band-limited (350–2600 Hz) short ATC carrier + crisp squelch tail
    const noise = this.ctx.createBufferSource();
    noise.buffer = this._pinkBuffer;
    noise.playbackRate.value = 1.6;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.setValueAtTime(1420, t0);
    bp.Q.setValueAtTime(1.8, t0);
    const g = this.ctx.createGain();
    const lvl = inCockpit ? 0.055 : 0.038;
    g.gain.setValueAtTime(0.0005, t0);
    g.gain.linearRampToValueAtTime(lvl * 0.65, t0 + 0.015);
    g.gain.linearRampToValueAtTime(lvl * 0.38, t0 + 0.24);
    g.gain.linearRampToValueAtTime(lvl, t0 + 0.27);
    g.gain.exponentialRampToValueAtTime(0.0005, t0 + 0.34);

    const tone = this.ctx.createOscillator();
    const tg = this.ctx.createGain();
    tone.type = 'sawtooth';
    tone.frequency.setValueAtTime(195, t0 + 0.03);
    tone.frequency.linearRampToValueAtTime(225, t0 + 0.14);
    tone.frequency.linearRampToValueAtTime(178, t0 + 0.24);
    tg.gain.setValueAtTime(0.0005, t0 + 0.03);
    tg.gain.linearRampToValueAtTime(lvl * 0.42, t0 + 0.06);
    tg.gain.exponentialRampToValueAtTime(0.0005, t0 + 0.25);

    noise.connect(bp);
    tone.connect(tg);
    tg.connect(bp);
    bp.connect(g);
    g.connect(pan);
    this._cleanupTransientNodes(tone, tg);
    this._cleanupTransientNodes(noise, bp, g, pan);
    noise.start(t0);
    noise.stop(t0 + 0.36);
    tone.start(t0 + 0.03);
    tone.stop(t0 + 0.26);
  }

  _triggerJoggerBreathGrain(intensity01 = 0.6) {
    if (!this.ctx || !this._pinkBuffer || this.ctx.state !== 'running') return;
    const t0 = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = this._pinkBuffer;
    src.playbackRate.value = 1.25;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.setValueAtTime(1150, t0);
    bp.Q.setValueAtTime(1.5, t0);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0005, t0);
    g.gain.linearRampToValueAtTime(0.028 * intensity01, t0 + 0.06);
    g.gain.exponentialRampToValueAtTime(0.0005, t0 + 0.24);
    src.connect(bp);
    bp.connect(g);
    g.connect(this.worldSumGain);
    this._cleanupTransientNodes(src, bp, g);
    src.start(t0);
    src.stop(t0 + 0.26);
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
    this._cleanupTransientNodes(osc, g);
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
      this._thunderDuckTimer = Math.max(this._thunderDuckTimer, clamp(2.4 * distAtten, 0.8, 2.6));

      // 3D HRTF Strike Origin in Sky Dome + .4 Overhead Height Dome + .1 LFE Shockwave
      const pan = this.ctx.createPanner();
      pan.panningModel = this.spatialOutputMode === 'SPEAKER' ? 'equalpower' : 'HRTF';
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
      if (this.heightDomeIn) pan.connect(this.heightDomeIn);

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
        this._cleanupTransientNodes(crackSrc, crackHp, crackGain);
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
      this._cleanupTransientNodes(noiseSrc, lp, rumbleGain, pan);
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
      if (this.lfeSumGain) subGain.connect(this.lfeSumGain);
      this._cleanupTransientNodes(osc, subGain);
      osc.start(t0);
      osc.stop(t0 + 1.62);

      // 3. Iterasi 3: Multi-Path Topographic & Monumental 3D HRTF Thunder Echoes
      //    (Gunung Api Pulau I flank & Menara Gedung Rektorat UI concrete spire)
      const echoes = AudioSynthesizer.computeMultiPathEchoes(
        strikeX,
        strikeY,
        strikeZ,
        this._lis.x,
        this._lis.y,
        this._lis.z,
        this._weatherState.tempC
      );
      for (let eIdx = 0; eIdx < echoes.length; eIdx++) {
        const ec = echoes[eIdx];
        const te = t0 + ec.extraDelaySec;
        const ePan = this.ctx.createPanner();
        ePan.panningModel = this.spatialOutputMode === 'SPEAKER' ? 'equalpower' : 'HRTF';
        ePan.distanceModel = 'inverse';
        ePan.refDistance = 140.0;
        ePan.rolloffFactor = 0.32;
        this._setPannerPosition(ePan, ec.rx, ec.ry, ec.rz, te, true);
        ePan.connect(this.worldSumGain);

        const eSrc = this.ctx.createBufferSource();
        eSrc.buffer = this._pinkBuffer;
        eSrc.playbackRate.value = 0.48 + eIdx * 0.08;
        const eLp = this.ctx.createBiquadFilter();
        eLp.type = 'lowpass';
        eLp.frequency.setValueAtTime(ec.cutoffHz * occMult, te);
        eLp.frequency.exponentialRampToValueAtTime(82, te + 2.1);
        const eGain = this.ctx.createGain();
        const ePeak = peakGain * ec.gainMult * 0.55;
        eGain.gain.setValueAtTime(0.0005, te);
        eGain.gain.linearRampToValueAtTime(Math.max(0.002, ePeak), te + 0.14);
        eGain.gain.exponentialRampToValueAtTime(0.0005, te + 2.25);
        eSrc.connect(eLp);
        eLp.connect(eGain);
        eGain.connect(ePan);
        this._cleanupTransientNodes(eSrc, eLp, eGain, ePan);
        eSrc.start(te);
        eSrc.stop(te + 2.32);
      }
    }, delaySec * 1000);

    this._thunderTimers.push(timerId);
    if (this._thunderTimers.length > 12) {
      this._thunderTimers.shift();
    }
  }
}
