/**
 * ============================================================================
 * FLOATING-ORIGIN REBASING MANAGER (IEEE-754 FLOAT64 -> LOCAL FLOAT32)
 * Eliminates WebGL/WebGPU Float32 vertex precision jitter at huge distances
 * (e.g. > 50 km - 10,000 km) by keeping active camera & train coordinates
 * anchored within [-2500m, +2500m] of world origin (0, 0, 0).
 * ============================================================================
 */
export class FloatingOriginManager {
  constructor(rebaseThresholdMeters = 2000.0, snapGridMeters = 100.0) {
    this.originS = 0.0; // IEEE-754 64-bit longitudinal origin offset (meters)
    this.rebaseThreshold = rebaseThresholdMeters;
    this.snapGrid = snapGridMeters;
    this.rebaseCount = 0;
  }

  /**
   * Convert 64-bit longitudinal track coordinate `s` to local Float32 Z coordinate.
   * Since track runs toward -Z as `s` increases, `localZ = -(s - originS)`.
   */
  worldSToLocalZ(s) {
    return -(s - this.originS);
  }

  /**
   * Convert local Z coordinate back to 64-bit longitudinal `s`.
   */
  localZToWorldS(localZ) {
    return this.originS - localZ;
  }

  /**
   * Force rebase immediately (e.g. on instant biome teleport).
   * Returns `deltaZ` applied to existing local Z positions (`newLocalZ = oldLocalZ + deltaZ`).
   */
  forceRebase(distanceS) {
    const newOriginS = Math.round(distanceS / this.snapGrid) * this.snapGrid;
    const deltaOriginS = newOriginS - this.originS;
    if (deltaOriginS === 0) return 0;
    this.originS = newOriginS;
    this.rebaseCount++;
    // Since localZ = -(s - originS) = -s + originS, increasing originS by +D shifts localZ by +D!
    return deltaOriginS;
  }

  /**
   * Check if `distanceS` has drifted beyond `rebaseThreshold` from `originS`.
   * If so, shifts `originS` cleanly on `snapGrid` (100m chunk boundary) and invokes callback.
   */
  rebaseIfNeeded(distanceS, onRebaseDeltaZ) {
    if (Math.abs(distanceS - this.originS) < this.rebaseThreshold) {
      return 0;
    }
    const deltaZ = this.forceRebase(distanceS);
    if (deltaZ !== 0 && typeof onRebaseDeltaZ === 'function') {
      onRebaseDeltaZ(deltaZ, this.originS);
    }
    return deltaZ;
  }
}
