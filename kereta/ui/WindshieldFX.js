/**
 * ============================================================================
 * REALISTIC LAMINATED WINDSHIELD GLASS, REFLECTION, RAIN & PANTOGRAPH WIPER FX
 * - Active in First-Person Behind-Glass Views: CAM 0 (Cab 3D) & CAM 4 (Coach)
 * - Smart high-DPI backing canvas scaling (`ctx.setTransform`) for crisp visuals
 *   with minimal 2D fillrate overhead on 2K/4K displays.
 * - Preserves 3-layer refractive raindrops (MICRO mist beads, LENS droplets,
 *   RIVULET slipstream trails), spatial polar wiper collision clearing, and
 *   20x twin pantograph wiper assemblies.
 * ============================================================================
 */
export class WindshieldFX {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas ? canvas.getContext('2d') : null;
    this.prevWiperSweep = 1.18;
    this.wetFilmAlpha = 0.0;
    this._logicalW = 0;
    this._logicalH = 0;
    this._appliedDpr = 1;

    this.rainDrops = Array.from({ length: 240 }, (_, idx) => {
      const kind = idx < 115 ? 'MICRO' : (idx < 195 ? 'LENS' : 'RIVULET');
      const x = 0.03 + Math.random() * 0.94;
      const y = 0.04 + Math.random() * 0.92;
      return {
        kind,
        x,
        y,
        r: kind === 'MICRO'
          ? (1.1 + Math.random() * 1.4)
          : (kind === 'LENS' ? (2.4 + Math.random() * 2.5) : (2.8 + Math.random() * 2.2)),
        v: kind === 'MICRO'
          ? 0.0
          : (kind === 'LENS' ? (0.015 + Math.random() * 0.045) : (0.11 + Math.random() * 0.16)),
        seed: Math.random() * 100,
        alpha: 0.72 + Math.random() * 0.28,
        trail: [
          { x, y },
          { x, y: Math.max(0, y - 0.012) },
          { x, y: Math.max(0, y - 0.024) },
          { x, y: Math.max(0, y - 0.036) }
        ]
      };
    });
  }

  render({
    dt,
    cameraMode,
    distanceS,
    speedMps,
    weather,
    cabYaw,
    cabPitch,
    curvature,
    inTunnel,
    underCanopy = false,
    wipersActive,
    wiperPhase,
    maxDpr = 1.25
  }) {
    if (!this.canvas || !this.ctx) return;

    const isCabGlass = (cameraMode === 0);
    const isCoachGlass = (cameraMode === 4);
    const active = isCabGlass || isCoachGlass;

    this.canvas.classList.toggle('hidden', !active);
    if (!active) return;

    const w = window.innerWidth;
    const h = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, Math.min(1.35, maxDpr));
    const targetBackingW = Math.max(1, Math.round(w * dpr));
    const targetBackingH = Math.max(1, Math.round(h * dpr));

    if (
      this.canvas.width !== targetBackingW ||
      this.canvas.height !== targetBackingH ||
      this._appliedDpr !== dpr
    ) {
      this.canvas.width = targetBackingW;
      this.canvas.height = targetBackingH;
      this._logicalW = w;
      this._logicalH = h;
      this._appliedDpr = dpr;
    }

    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Parallax shift from camera look angle + track curvature
    const shiftX = (cabYaw * 240) + (curvature * 9000);
    const shiftY = (cabPitch * 180);

    if (isCabGlass) {
      // 1. Laminated Safety Glass Polarized Dichroic / Fresnel Sheen at Edges
      const fresnelGrad = ctx.createRadialGradient(
        w * 0.5 + shiftX * 0.25,
        h * 0.48 + shiftY * 0.25,
        Math.min(w, h) * 0.22,
        w * 0.5,
        h * 0.5,
        Math.max(w, h) * 0.74
      );
      fresnelGrad.addColorStop(0.0, 'rgba(255, 255, 255, 0.0)');
      fresnelGrad.addColorStop(0.68, 'rgba(56, 189, 248, 0.035)');
      fresnelGrad.addColorStop(0.90, 'rgba(14, 165, 233, 0.11)');
      fresnelGrad.addColorStop(1.0, 'rgba(2, 6, 23, 0.45)');
      ctx.fillStyle = fresnelGrad;
      ctx.fillRect(0, 0, w, h);

      // 2. Dynamic Diagonal Specular Sky & Cloud Reflection Streak across Curved Windshield
      ctx.save();
      ctx.translate(w * 0.5 + shiftX * 0.55, h * 0.35 + shiftY * 0.45);
      ctx.rotate(-0.24);
      const sheenGrad = ctx.createLinearGradient(0, -h * 0.45, 0, h * 0.45);
      const skyRefAlpha = inTunnel ? 0.025 : (weather === 'STORM' ? 0.04 : 0.095);
      sheenGrad.addColorStop(0.0, 'rgba(255, 255, 255, 0)');
      sheenGrad.addColorStop(0.32, `rgba(224, 242, 254, ${skyRefAlpha})`);
      sheenGrad.addColorStop(0.45, `rgba(255, 255, 255, ${skyRefAlpha * 1.35})`);
      sheenGrad.addColorStop(0.56, 'rgba(186, 230, 253, 0.015)');
      sheenGrad.addColorStop(1.0, 'rgba(255, 255, 255, 0)');
      ctx.fillStyle = sheenGrad;
      ctx.fillRect(-w, -h * 0.45, w * 2, h * 0.9);

      // Passing Overhead Catenary Portal / Tunnel Light Reflection Flash on Glass
      const portalPhase = ((distanceS % 50) + 50) % 50 / 50;
      const bandY = (portalPhase - 0.5) * h * 1.3;
      const portalRef = ctx.createLinearGradient(0, bandY - 28, 0, bandY + 28);
      const pAlpha = inTunnel ? 0.08 : 0.045;
      portalRef.addColorStop(0, 'rgba(255, 255, 255, 0)');
      portalRef.addColorStop(
        0.5,
        inTunnel ? `rgba(251, 191, 36, ${pAlpha})` : `rgba(255, 255, 255, ${pAlpha})`
      );
      portalRef.addColorStop(1, 'rgba(255, 255, 255, 0)');
      ctx.fillStyle = portalRef;
      ctx.fillRect(-w, bandY - 28, w * 2, 56);
      ctx.restore();

      // 3. Solar Lens Glare & Micro-Scratches Halo on Glass
      if (!inTunnel && weather !== 'STORM') {
        const sunX = w * (weather === 'SUNSET' ? 0.25 : 0.72) + shiftX * 0.7;
        const sunY = h * 0.18 + shiftY * 0.7;
        const sunGlare = ctx.createRadialGradient(sunX, sunY, 4, sunX, sunY, Math.min(w, h) * 0.34);
        sunGlare.addColorStop(0.0, 'rgba(255, 251, 235, 0.16)');
        sunGlare.addColorStop(0.3, 'rgba(254, 240, 138, 0.06)');
        sunGlare.addColorStop(1.0, 'rgba(255, 255, 255, 0)');
        ctx.fillStyle = sunGlare;
        ctx.fillRect(0, 0, w, h);

        ctx.strokeStyle = 'rgba(255, 251, 235, 0.045)';
        ctx.lineWidth = 1.2;
        for (let r = 36; r <= 132; r += 32) {
          ctx.beginPath();
          ctx.arc(sunX, sunY, r, -0.6, 0.8);
          ctx.stroke();
        }
      }

      // 4. Internal Cockpit Dashboard & Glowing Cyan MFD Reflection on Lower Sloped Windshield Glass
      const dashRefY = h * 0.68 + shiftY * 0.25;
      const dashRefGrad = ctx.createLinearGradient(0, dashRefY, 0, h);
      dashRefGrad.addColorStop(0.0, 'rgba(6, 182, 212, 0.0)');
      dashRefGrad.addColorStop(0.35, 'rgba(6, 182, 212, 0.055)');
      dashRefGrad.addColorStop(0.75, 'rgba(15, 23, 42, 0.18)');
      dashRefGrad.addColorStop(1.0, 'rgba(2, 6, 23, 0.38)');
      ctx.fillStyle = dashRefGrad;
      ctx.fillRect(0, dashRefY, w, h - dashRefY);
    } else if (isCoachGlass) {
      // COACH PASSENGER DOUBLE-GLAZED PANORAMIC WINDOW FRAME & REFLECTION
      const sheenGrad = ctx.createLinearGradient(0, 0, w, h);
      sheenGrad.addColorStop(0.0, 'rgba(186, 230, 253, 0.08)');
      sheenGrad.addColorStop(0.45, 'rgba(255, 255, 255, 0.02)');
      sheenGrad.addColorStop(0.55, 'rgba(255, 255, 255, 0.11)');
      sheenGrad.addColorStop(1.0, 'rgba(14, 165, 233, 0.05)');
      ctx.fillStyle = sheenGrad;
      ctx.fillRect(0, 0, w, h);
    }

    // 5. Compute Exact Twin Pantograph Wiper Geometry & Sweep Interval
    const pivLX = w * 0.33;
    const pivRX = w * 0.65;
    const pivY = h * 1.02;
    const bladeR = Math.min(w, h) * 0.78;
    const curSw = wipersActive ? Math.sin(wiperPhase) * 0.68 : 1.18;
    const prevSw = this.prevWiperSweep !== undefined ? this.prevWiperSweep : curSw;
    this.prevWiperSweep = curSw;

    const angPrev = -Math.PI * 0.5 - prevSw;
    const angCurr = -Math.PI * 0.5 - curSw;
    const minSweepAng = Math.min(angPrev, angCurr) - 0.045;
    const maxSweepAng = Math.max(angPrev, angCurr) + 0.045;

    if (wipersActive) {
      this.wetFilmAlpha = Math.min(0.18, this.wetFilmAlpha + dt * 0.45);
    } else {
      this.wetFilmAlpha = Math.max(0.0, this.wetFilmAlpha - dt * 0.15);
    }

    // 6. Photorealistic 3-Layer Refractive Windshield Water Droplets & Spatial Wiper Clearing
    const inOpenRain = weather === 'STORM' && !inTunnel && !underCanopy;
    if (inOpenRain) {
      this.rainResidual = 1.0;
    } else if (this.rainResidual > 0) {
      const dryRate = 0.16 + Math.abs(speedMps) * 0.008 + (wipersActive ? 0.55 : 0.0);
      this.rainResidual = Math.max(0.0, this.rainResidual - dt * dryRate);
    }

    if (inOpenRain || (this.rainResidual || 0) > 0.01) {
      const speedFactor = 1.0 + Math.abs(speedMps) * 0.028;
      const tSec = performance.now() * 0.001;

      if (isCabGlass && (wipersActive || this.wetFilmAlpha > 0.01)) {
        ctx.save();
        ctx.strokeStyle = `rgba(186, 230, 253, ${0.04 + this.wetFilmAlpha * 0.35})`;
        ctx.lineWidth = bladeR * 0.68;
        ctx.lineCap = 'round';
        [pivLX, pivRX].forEach((pivX) => {
          ctx.beginPath();
          ctx.arc(pivX, pivY, bladeR * 0.64, -Math.PI * 0.5 - 0.68, -Math.PI * 0.5 + 0.68);
          ctx.stroke();
        });
        ctx.restore();
      }

      const checkWiperHit = (px, py, pivX) => {
        const dx = px - pivX;
        const dy = py - pivY;
        const dist = Math.hypot(dx, dy);
        if (dist >= bladeR * 0.24 && dist <= bladeR * 1.02) {
          const ptAng = Math.atan2(dy, dx);
          if (ptAng >= minSweepAng && ptAng <= maxSweepAng) return true;
        }
        return false;
      };

      for (let i = 0; i < this.rainDrops.length; i++) {
        const d = this.rainDrops[i];
        if (!inOpenRain && d.alpha <= 0.01) continue;

        if (d.kind !== 'MICRO') {
          const windX = (d.x - 0.5) * 0.045 * dt * speedFactor;
          const wiggle = Math.sin(tSec * 5.5 + d.seed) * 0.0008;
          d.x += windX + wiggle;
          d.y += d.v * dt * (1.0 + Math.abs(speedMps) * 0.015);

          if (d.kind === 'RIVULET' && d.trail) {
            for (let k = d.trail.length - 1; k > 0; k--) {
              d.trail[k].x += (d.trail[k - 1].x - d.trail[k].x) * Math.min(1, dt * 18);
              d.trail[k].y += (d.trail[k - 1].y - d.trail[k].y) * Math.min(1, dt * 18);
            }
            d.trail[0].x = d.x;
            d.trail[0].y = d.y;
          }
        } else if (inOpenRain) {
          d.alpha = Math.min(1.0, d.alpha + dt * 0.35);
        } else {
          d.alpha = Math.max(0.0, d.alpha - dt * 0.35);
        }

        const px = d.x * w;
        const py = d.y * h;

        let wipedByBlade = false;
        if (isCabGlass && wipersActive) {
          wipedByBlade = checkWiperHit(px, py, pivLX) || checkWiperHit(px, py, pivRX);
        }

        if (wipedByBlade || d.y > 0.98 || d.x < 0.01 || d.x > 0.99) {
          if (!inOpenRain) {
            d.alpha = 0.0;
            continue;
          }
          d.x = 0.03 + Math.random() * 0.94;
          d.y = wipedByBlade ? (0.04 + Math.random() * 0.88) : (0.03 + Math.random() * 0.35);
          d.alpha = wipedByBlade ? 0.05 : 0.75;
          if (d.trail) {
            for (let k = 0; k < d.trail.length; k++) {
              d.trail[k].x = d.x;
              d.trail[k].y = d.y;
            }
          }
          if (wipedByBlade) continue;
        }

        if (inOpenRain) {
          d.alpha = Math.min(0.95, d.alpha + dt * 0.9);
        } else {
          d.alpha = Math.max(0.0, d.alpha - dt * 0.22);
        }
        const a = d.alpha * (inOpenRain ? 1.0 : (this.rainResidual || 0));
        if (a <= 0.01) continue;

        if (d.kind === 'MICRO') {
          ctx.fillStyle = `rgba(15, 23, 42, ${0.28 * a})`;
          ctx.beginPath();
          ctx.arc(px, py, d.r + 0.5, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = `rgba(224, 242, 254, ${0.72 * a})`;
          ctx.beginPath();
          ctx.arc(px - 0.3, py - 0.3, d.r * 0.65, 0, Math.PI * 2);
          ctx.fill();
        } else if (d.kind === 'LENS') {
          const rx = d.r * 1.08;
          const ry = d.r * 1.25;

          ctx.fillStyle = `rgba(2, 6, 23, ${0.34 * a})`;
          ctx.beginPath();
          ctx.ellipse(px, py, rx + 0.8, ry + 0.8, 0, 0, Math.PI * 2);
          ctx.fill();

          const dropGrad = ctx.createRadialGradient(
            px + rx * 0.28,
            py + ry * 0.32,
            rx * 0.1,
            px,
            py,
            ry
          );
          dropGrad.addColorStop(0.0, `rgba(224, 242, 254, ${0.82 * a})`);
          dropGrad.addColorStop(0.55, `rgba(125, 211, 252, ${0.38 * a})`);
          dropGrad.addColorStop(1.0, `rgba(15, 23, 42, ${0.42 * a})`);
          ctx.fillStyle = dropGrad;
          ctx.beginPath();
          ctx.ellipse(px, py, rx, ry, 0, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = `rgba(255, 255, 255, ${0.94 * a})`;
          ctx.beginPath();
          ctx.arc(px - rx * 0.34, py - ry * 0.36, Math.max(0.9, rx * 0.30), 0, Math.PI * 2);
          ctx.fill();
        } else if (d.kind === 'RIVULET' && d.trail) {
          for (let k = d.trail.length - 1; k >= 1; k--) {
            const tFrac = 1.0 - k / d.trail.length;
            const tx = d.trail[k].x * w;
            const ty = d.trail[k].y * h;
            const tr = d.r * tFrac * 0.72;
            ctx.fillStyle = `rgba(186, 230, 253, ${0.42 * tFrac * a})`;
            ctx.beginPath();
            ctx.arc(tx, ty, Math.max(0.8, tr), 0, Math.PI * 2);
            ctx.fill();
          }

          const rx = d.r * 1.05;
          const ry = d.r * 1.38;
          ctx.fillStyle = `rgba(2, 6, 23, ${0.36 * a})`;
          ctx.beginPath();
          ctx.ellipse(px, py, rx + 0.8, ry + 0.8, 0, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = `rgba(224, 242, 254, ${0.78 * a})`;
          ctx.beginPath();
          ctx.ellipse(px, py, rx, ry, 0, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = `rgba(255, 255, 255, ${0.95 * a})`;
          ctx.beginPath();
          ctx.arc(px - rx * 0.32, py - ry * 0.35, Math.max(1.0, rx * 0.32), 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // 7. Single Unified 20x Detailed Pantograph Windshield Wiper System
    if (isCabGlass) {
      const rotAngle = angCurr + Math.PI * 0.5;
      const moveDir = Math.sign(angCurr - angPrev);

      [pivLX, pivRX].forEach((pivX) => {
        ctx.save();
        ctx.translate(pivX, pivY);
        ctx.rotate(rotAngle);

        // 1. Soft Cast Shadow on Laminated Glass
        ctx.save();
        ctx.translate(6, 5);
        ctx.strokeStyle = 'rgba(2, 6, 23, 0.38)';
        ctx.lineWidth = 6.5;
        ctx.beginPath();
        ctx.moveTo(-9, 0); ctx.lineTo(-9, -bladeR * 0.62);
        ctx.moveTo(9, 0);  ctx.lineTo(9, -bladeR * 0.60);
        ctx.moveTo(0, -bladeR * 0.24); ctx.lineTo(0, -bladeR * 0.98);
        ctx.stroke();
        ctx.restore();

        // 2. Water-Surge Bead Line Pushed Along Moving Squeegee Edge during Storm
        if (weather === 'STORM' && wipersActive && Math.abs(angCurr - angPrev) > 0.001) {
          const surgeX = moveDir >= 0 ? 5.5 : -5.5;
          const surgeGrad = ctx.createLinearGradient(0, -bladeR * 0.98, 0, -bladeR * 0.25);
          surgeGrad.addColorStop(0, 'rgba(224, 242, 254, 0.85)');
          surgeGrad.addColorStop(0.5, 'rgba(255, 255, 255, 0.95)');
          surgeGrad.addColorStop(1, 'rgba(186, 230, 253, 0.75)');
          ctx.strokeStyle = surgeGrad;
          ctx.lineWidth = 3.2;
          ctx.beginPath();
          ctx.moveTo(surgeX, -bladeR * 0.25);
          ctx.lineTo(surgeX, -bladeR * 0.98);
          ctx.stroke();
        }

        // 3. Dual Parallel Pantograph Arms with Metallic Bevel Highlight
        ctx.strokeStyle = '#090d16';
        ctx.lineWidth = 5.5;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(-9, 0); ctx.lineTo(-9, -bladeR * 0.62);
        ctx.moveTo(9, 0);  ctx.lineTo(9, -bladeR * 0.60);
        ctx.stroke();

        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(-10.2, -10); ctx.lineTo(-10.2, -bladeR * 0.61);
        ctx.moveTo(7.8, -10);   ctx.lineTo(7.8, -bladeR * 0.59);
        ctx.stroke();

        // 4. Pantograph Cross-Linkage Bridge & Twin Chrome Rivet Pins
        ctx.fillStyle = '#0f172a';
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 1.5;
        ctx.fillRect(-14, -bladeR * 0.635, 28, 10);
        ctx.strokeRect(-14, -bladeR * 0.635, 28, 10);
        ctx.fillStyle = '#e2e8f0';
        ctx.beginPath();
        ctx.arc(-9, -bladeR * 0.635 + 5, 2.4, 0, Math.PI * 2);
        ctx.arc(9, -bladeR * 0.635 + 5, 2.4, 0, Math.PI * 2);
        ctx.fill();

        // 5. High-Speed Aerodynamic Wind Deflector Spoiler Fin
        const finGrad = ctx.createLinearGradient(-14, 0, -2, 0);
        finGrad.addColorStop(0, '#090d16');
        finGrad.addColorStop(0.5, '#334155');
        finGrad.addColorStop(1, '#0f172a');
        ctx.fillStyle = finGrad;
        ctx.beginPath();
        ctx.moveTo(-13, -bladeR * 0.58);
        ctx.lineTo(-4, -bladeR * 0.62);
        ctx.lineTo(-4, -bladeR * 0.30);
        ctx.lineTo(-13, -bladeR * 0.27);
        ctx.closePath();
        ctx.fill();

        // 6. Articulated Whippletree Multi-Point Pressure Yokes
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 2.6;
        [-0.86, -0.68, -0.50, -0.34].forEach((cFrac) => {
          const cy = -bladeR * cFrac;
          ctx.fillStyle = '#334155';
          ctx.fillRect(-5.5, cy - 3, 11.0, 6.0);
          ctx.fillStyle = '#cbd5e1';
          ctx.beginPath();
          ctx.arc(0, cy, 1.6, 0, Math.PI * 2);
          ctx.fill();
        });

        // 7. Primary Graphite Rubber Squeegee Blade & Stainless Spine Backing
        ctx.fillStyle = '#020617';
        ctx.fillRect(-3.2, -bladeR * 0.99, 6.4, bladeR * 0.74);
        ctx.fillStyle = '#94a3b8';
        ctx.fillRect(-1.1, -bladeR * 0.98, 2.2, bladeR * 0.72);

        // 8. Base Motor Drive Housing & Chrome Hex Nut
        ctx.fillStyle = '#090d16';
        ctx.beginPath();
        ctx.arc(0, 0, 15, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = '#e2e8f0';
        ctx.beginPath();
        ctx.arc(0, 0, 5.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
      });

      // 8. Ceramic Frit Black Border & Curved Cockpit Window Gasket Seal
      ctx.save();
      ctx.strokeStyle = 'rgba(9, 13, 22, 0.88)';
      ctx.lineWidth = Math.max(18, w * 0.015);
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(4, 4, w - 8, h - 8, 32);
      } else {
        ctx.rect(4, 4, w - 8, h - 8);
      }
      ctx.stroke();

      ctx.strokeStyle = 'rgba(148, 163, 184, 0.18)';
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(16, 16, w - 32, h - 32, 24);
      } else {
        ctx.rect(16, 16, w - 32, h - 32);
      }
      ctx.stroke();
      ctx.restore();
    } else if (isCoachGlass) {
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.92)';
      ctx.lineWidth = Math.max(32, w * 0.032);
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(10, 10, w - 20, h - 20, 48);
      } else {
        ctx.rect(10, 10, w - 20, h - 20);
      }
      ctx.stroke();
    }
  }
}
