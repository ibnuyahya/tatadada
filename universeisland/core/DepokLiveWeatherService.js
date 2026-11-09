/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.9 — Real-Time Kampus UI Depok Live Telemetry Edition)
 * ./core/DepokLiveWeatherService.js
 *
 * Provides real-time Kampus Universitas Indonesia (Kota Depok, Jawa Barat) astronomical clock
 * (Asia/Jakarta WIB, UTC+7) and live meteorological telemetry pinned to Gedung Rektorat UI
 * (6°21'57" S, 106°49'43" E / lat -6.3658, lon 106.8286) via CORS-open Open-Meteo WMO API,
 * with non-blocking timeout protection and diurnal Depok climatology offline fallback.
 */

export class DepokLiveWeatherService {
  static KAMPUS_UI_LAT = -6.3658;
  static KAMPUS_UI_LON = 106.8286;
  static TIMEZONE = 'Asia/Jakarta';
  static REFRESH_INTERVAL_MS = 5 * 60 * 1000; // Refresh live Depok weather every 5 minutes

  /**
   * Computes exact real-time Kota Depok (Asia/Jakarta, UTC+7 WIB) clock & solar hour [0, 24),
   * independent of the viewer device's local OS timezone.
   *
   * @param {Date} [nowDate]
   */
  static getDepokWIBClock(nowDate = new Date()) {
    // Asia/Jakarta is strictly UTC+7 year-round (no DST)
    const utcMs = nowDate.getTime();
    const wibDate = new Date(utcMs + 7 * 3600 * 1000);

    const hour = wibDate.getUTCHours();
    const minute = wibDate.getUTCMinutes();
    const second = wibDate.getUTCSeconds();
    const ms = wibDate.getUTCMilliseconds();

    const solarHour = hour + minute / 60.0 + (second + ms * 0.001) / 3600.0;
    const hh = String(hour).padStart(2, '0');
    const mm = String(minute).padStart(2, '0');
    const ss = String(second).padStart(2, '0');

    let todRegime = 'DAY';
    let periodLabelId = 'Siang';
    if (solarHour >= 18.5 || solarHour < 5.25) {
      todRegime = 'NIGHT';
      periodLabelId = solarHour < 5.25 ? 'Dini Hari' : 'Malam';
    } else if (solarHour >= 5.25 && solarHour < 6.58) {
      todRegime = 'DAWN';
      periodLabelId = 'Fajar';
    } else if (solarHour >= 6.58 && solarHour < 10.25) {
      todRegime = 'MORNING';
      periodLabelId = 'Pagi';
    } else if (solarHour >= 14.5 && solarHour < 17.08) {
      todRegime = 'AFTERNOON';
      periodLabelId = 'Sore';
    } else if (solarHour >= 17.08 && solarHour < 18.5) {
      todRegime = 'DUSK';
      periodLabelId = 'Senja';
    }

    return {
      hour,
      minute,
      second,
      solarHour,
      formattedHM: `${hh}:${mm} WIB`,
      formattedHMS: `${hh}:${mm}:${ss} WIB`,
      todRegime,
      periodLabelId
    };
  }

  /**
   * Formats any continuous 24h solarHour [0, 24) into WIB clock & Indonesian period label
   * so Manual Simulation Mode displays the exact simulated solar clock during 20–45s sweeps!
   */
  static formatSolarHourClock(solarHour = 10.35) {
    const hWrap = ((Number(solarHour) % 24.0) + 24.0) % 24.0;
    const totalSec = Math.floor(hWrap * 3600.0) % 86400;
    const hour = Math.floor(totalSec / 3600);
    const minute = Math.floor((totalSec % 3600) / 60);
    const second = totalSec % 60;
    const hh = String(hour).padStart(2, '0');
    const mm = String(minute).padStart(2, '0');
    const ss = String(second).padStart(2, '0');

    let todRegime = 'DAY';
    let periodLabelId = 'Siang';
    if (hWrap >= 18.5 || hWrap < 5.25) {
      todRegime = 'NIGHT';
      periodLabelId = hWrap < 5.25 ? 'Dini Hari' : 'Malam';
    } else if (hWrap >= 5.25 && hWrap < 6.58) {
      todRegime = 'DAWN';
      periodLabelId = 'Fajar';
    } else if (hWrap >= 6.58 && hWrap < 10.25) {
      todRegime = 'MORNING';
      periodLabelId = 'Pagi';
    } else if (hWrap >= 14.5 && hWrap < 17.08) {
      todRegime = 'AFTERNOON';
      periodLabelId = 'Sore';
    } else if (hWrap >= 17.08 && hWrap < 18.5) {
      todRegime = 'DUSK';
      periodLabelId = 'Senja';
    }

    return {
      hour,
      minute,
      second,
      solarHour: hWrap,
      formattedHM: `${hh}:${mm} WIB`,
      formattedHMS: `${hh}:${mm}:${ss} WIB`,
      todRegime,
      periodLabelId
    };
  }

  /**
   * Maps WMO weather code + live precipitation (mm) + cloud cover (%) into NEXUS physical weather regime.
   */
  static classifyWMOCondition(
    wmoCode = 1,
    precipitationMm = 0.0,
    cloudCoverPct = 30,
    windSpeedKmh = 8.0
  ) {
    const code = Number(wmoCode) || 0;
    const precip = Math.max(0.0, Number(precipitationMm) || 0.0);
    const clouds = Math.max(0.0, Math.min(100.0, Number(cloudCoverPct) || 0.0));
    const windKmh = Math.max(0.0, Number(windSpeedKmh) || 0.0);

    // WMO 95, 96, 99: Thunderstorm
    if (code >= 95) {
      return {
        weatherMode: 'STORM',
        labelId: 'Badai Petir Tropis',
        cloudCoverNorm: Math.max(0.92, clouds / 100.0)
      };
    }

    // WMO 63..67 (Moderate/Heavy Rain), 81..82 (Heavy Showers), or significant precipitation >= 2.2 mm
    if ((code >= 63 && code <= 67) || (code >= 81 && code <= 82) || precip >= 2.2) {
      return {
        weatherMode: 'RAIN',
        labelId: code >= 65 || code === 82 || precip >= 4.5 ? 'Hujan Deras Pulau' : 'Hujan Tropis',
        cloudCoverNorm: Math.max(0.86, clouds / 100.0)
      };
    }

    // WMO 61 (Light Rain), 80 (Slight Rain Showers), or moderate precipitation 0.85..2.2 mm
    if (code === 61 || code === 62 || code === 80 || precip >= 0.85) {
      return {
        weatherMode: 'SHOWER',
        labelId: 'Hujan Ringan',
        cloudCoverNorm: Math.max(0.78, clouds / 100.0)
      };
    }

    // WMO 51..57 (Drizzle) or light measurable precipitation 0.08..0.85 mm
    if ((code >= 51 && code <= 57) || precip >= 0.08) {
      return {
        weatherMode: 'MIST',
        labelId: 'Gerimis Tropis',
        cloudCoverNorm: Math.max(0.68, clouds / 100.0)
      };
    }

    // WMO 45..48 (Radiation / Valley Fog & Dew Mist without rain)
    if (code >= 45 && code <= 48) {
      return {
        weatherMode: 'FOGGY',
        labelId: 'Kabut Pesisir',
        cloudCoverNorm: Math.max(0.72, clouds / 100.0)
      };
    }

    // Brisk dry monsoon wind >= 24 km/h
    if (windKmh >= 24.0) {
      return {
        weatherMode: 'WINDY',
        labelId: 'Angin Kencang Selat',
        cloudCoverNorm: Math.max(0.36, Math.min(0.68, clouds / 100.0))
      };
    }

    // WMO 3 or thick dry cloud deck >= 82% -> Overcast (Mendung)
    if (code === 3 || clouds >= 82) {
      return {
        weatherMode: 'OVERCAST',
        labelId: 'Mendung Tropis',
        cloudCoverNorm: Math.max(0.82, Math.min(0.94, clouds / 100.0))
      };
    }

    // Clear / Fair / Partly Cloudy (WMO 0, 1, 2)
    let labelId = 'Cerah';
    let maxDryCloud = 0.25;
    if (code === 2 || clouds >= 38) {
      labelId = 'Cerah Berawan';
      maxDryCloud = 0.46;
    }

    return {
      weatherMode: 'CLEAR',
      labelId,
      cloudCoverNorm: Math.max(0.12, Math.min(maxDryCloud, (clouds / 100.0) * 0.80))
    };
  }

  /**
   * Realistic Kota Depok diurnal tropical climatology fallback when browser is completely offline.
   */
  static buildOfflineDepokFallback(nowDate = new Date()) {
    const clock = DepokLiveWeatherService.getDepokWIBClock(nowDate);
    const h = clock.solarHour;
    // Diurnal temperature curve of Depok: ~24.2°C pre-dawn -> ~32.2°C at 13:30 WIB -> ~26.5°C night
    const diurnal = Math.sin(((h - 8.5) / 24.0) * Math.PI * 2.0);
    const temperatureC = +(28.2 + 3.8 * diurnal).toFixed(1);
    const humidityPct = Math.round(76 - 16 * diurnal);
    const cloudCoverPct = h >= 13.5 && h <= 18.0 ? 54 : 28;
    const classified = DepokLiveWeatherService.classifyWMOCondition(2, 0.0, cloudCoverPct);

    return {
      ok: false,
      isLiveApi: false,
      source: 'DEPOK_CLIMATOLOGY_FALLBACK',
      locationName: 'Universe Island',
      weatherMode: classified.weatherMode,
      labelId: `${classified.labelId} (Estimasi Lokal)`,
      wmoCode: 2,
      temperatureC,
      humidityPct,
      precipitationMm: 0.0,
      cloudCoverPct,
      cloudCoverNorm: classified.cloudCoverNorm,
      windSpeedKmh: 7.5,
      windDirectionDeg: 215,
      fetchedAtMs: Date.now()
    };
  }

  /**
   * Fetches real-time meteorological telemetry directly for Kampus UI, Kota Depok
   * from Open-Meteo (CORS *, zero API key, 4.5s timeout).
   */
  static async fetchLiveDepokWeather({ timeoutMs = 4500 } = {}) {
    if (typeof fetch !== 'function') {
      return DepokLiveWeatherService.buildOfflineDepokFallback();
    }

    const url =
      `https://api.open-meteo.com/v1/forecast` +
      `?latitude=${DepokLiveWeatherService.KAMPUS_UI_LAT}` +
      `&longitude=${DepokLiveWeatherService.KAMPUS_UI_LON}` +
      `&current=temperature_2m,relative_humidity_2m,precipitation,rain,showers,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m` +
      `&timezone=Asia%2FJakarta`;

    const controller = typeof AbortController === 'function' ? new AbortController() : null;
    const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;

    try {
      const res = await fetch(url, {
        method: 'GET',
        mode: 'cors',
        cache: 'no-store',
        signal: controller ? controller.signal : undefined
      });
      if (timer) clearTimeout(timer);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const json = await res.json();
      const cur = json && json.current ? json.current : null;
      if (!cur) {
        throw new Error('Malformed Open-Meteo payload');
      }

      const wmoCode = Number(cur.weather_code ?? 1);
      const precip = Math.max(
        Number(cur.precipitation || 0),
        Number(cur.rain || 0) + Number(cur.showers || 0)
      );
      const cloudCoverPct = Number(cur.cloud_cover ?? 30);
      const windSpeedKmh = +(Number(cur.wind_speed_10m) || 0).toFixed(1);
      const classified = DepokLiveWeatherService.classifyWMOCondition(
        wmoCode,
        precip,
        cloudCoverPct,
        windSpeedKmh
      );

      return {
        ok: true,
        isLiveApi: true,
        source: 'LIVE_OPEN_METEO',
        locationName: 'Universe Island',
        weatherMode: classified.weatherMode,
        labelId: classified.labelId,
        wmoCode,
        temperatureC: +(Number(cur.temperature_2m ?? 29.5)).toFixed(1),
        humidityPct: Math.round(Number(cur.relative_humidity_2m ?? 74)),
        precipitationMm: +precip.toFixed(2),
        cloudCoverPct: Math.round(cloudCoverPct),
        cloudCoverNorm: classified.cloudCoverNorm,
        windSpeedKmh: +(Number(cur.wind_speed_10m ?? 8.0)).toFixed(1),
        windDirectionDeg: Math.round(Number(cur.wind_direction_10m ?? 210)),
        fetchedAtMs: Date.now()
      };
    } catch (_err) {
      if (timer) clearTimeout(timer);
      return DepokLiveWeatherService.buildOfflineDepokFallback();
    }
  }
}
