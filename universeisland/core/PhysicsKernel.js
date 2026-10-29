/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./core/PhysicsKernel.js
 *
 * Differential motion integrators, exponential frame-rate-independent damping,
 * tropical breeze wind-field harmonics, and smooth kinematic oscillators (Hukum Emas 11 & 15).
 */
export class PhysicsKernel {
  static expDamp(current, target, lambda, dt) {
    const alpha = 1.0 - Math.exp(-lambda * Math.max(0.0, dt));
    return current + (target - current) * alpha;
  }

  /**
   * Evaluates smooth multi-frequency tropical Depok breeze vector at world (x, z, t).
   */
  static sampleWindField(x, z, timeSeconds, gustMultiplier = 1.0) {
    const phaseA = timeSeconds * 1.85 + x * 0.06 + z * 0.04;
    const phaseB = timeSeconds * 3.4 - x * 0.03 + z * 0.08;
    const strength = (0.55 + 0.32 * Math.sin(phaseA) + 0.13 * Math.cos(phaseB)) * gustMultiplier;
    return {
      wx: strength * 0.88,
      wz: strength * 0.46,
      magnitude: strength
    };
  }

  /**
   * Smooth quintic S-curve interpolation C2-continuous on [0, 1] (supports both edge1 > edge0 and edge1 < edge0)
   */
  static smootherstep(edge0, edge1, x) {
    const span = edge1 - edge0;
    const denom = Math.abs(span) < 1e-6 ? 1e-6 : span;
    const t = Math.max(0.0, Math.min(1.0, (x - edge0) / denom));
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
  }
}
