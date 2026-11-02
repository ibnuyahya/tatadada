/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v4.0 — Strict Solid-Body Non-Penetration Physics Edition)
 * ./core/PhysicsKernel.js
 *
 * Differential motion integrators, exponential frame-rate-independent damping,
 * tropical breeze wind-field harmonics, smooth kinematic oscillators (Hukum Emas 11 & 15),
 * and Unified Zero-GC 3D Rigid / OBB / Capsule / Topological-Tunnel Solid-Body Non-Penetration Solver.
 */

const _mtvScratch = { nx: 0.0, ny: 0.0, nz: 0.0, depth: 0.0 };
const _satAxes = [
  [1.0, 0.0],
  [0.0, 1.0],
  [1.0, 0.0],
  [0.0, 1.0]
];
const _cornersA = [
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0]
];
const _cornersB = [
  [0, 0],
  [0, 0],
  [0, 0],
  [0, 0]
];

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

  /**
   * Precomputes world-space AABB broadphase bounds on a solid collider (BOX/AABB, CYL, OBB) in-place.
   */
  static prepareSolidCollider(col) {
    if (!col) return col;
    const kind = col.kind || col.type;
    col.kind = kind;
    if (kind === 'BOX' || kind === 'AABB') {
      col.bMinX = col.minX;
      col.bMaxX = col.maxX;
      col.bMinY = col.minY;
      col.bMaxY = col.maxY;
      col.bMinZ = col.minZ;
      col.bMaxZ = col.maxZ;
    } else if (kind === 'CYL') {
      col.bMinX = col.x - col.r;
      col.bMaxX = col.x + col.r;
      col.bMinY = col.minY;
      col.bMaxY = col.maxY;
      col.bMinZ = col.z - col.r;
      col.bMaxZ = col.z + col.r;
    } else if (kind === 'OBB') {
      const yaw = col.yaw || 0.0;
      const sinY = Math.sin(yaw);
      const cosY = Math.cos(yaw);
      col._sinY = sinY;
      col._cosY = cosY;
      const fL = col.frontL !== undefined ? col.frontL : col.halfL;
      const bL = col.backL !== undefined ? col.backL : col.halfL;
      const midOff = 0.5 * (fL - bL);
      const halfL = 0.5 * (fL + bL);
      col._effHalfL = halfL;
      col._effCx = col.cx + sinY * midOff;
      col._effCz = col.cz + cosY * midOff;
      const extX = Math.abs(cosY) * col.halfW + Math.abs(sinY) * halfL;
      const extZ = Math.abs(sinY) * col.halfW + Math.abs(cosY) * halfL;
      col.bMinX = col._effCx - extX;
      col.bMaxX = col._effCx + extX;
      col.bMinY = col.minY;
      col.bMaxY = col.maxY;
      col.bMinZ = col._effCz - extZ;
      col.bMaxZ = col._effCz + extZ;
    }
    return col;
  }

  /**
   * Computes exact 2D Minimum Translation Vector (MTV) pushing circle (px, pz, radius)
   * out of oriented rectangle centered at (cx, cz) with half-extents (halfW, halfL) and heading `yaw`.
   * Returns penetration depth (> 0 if overlapping, <= 0 if separated) and writes unit outward normal to `out`.
   */
  static computeOBB2DCircleMTV(cx, cz, halfW, halfL, sinY, cosY, px, pz, radius, out = _mtvScratch) {
    const dx = px - cx;
    const dz = pz - cz;
    // Local frame: +X_local = (cosY, -sinY), +Z_local = (sinY, cosY)
    const lx = dx * cosY - dz * sinY;
    const lz = dx * sinY + dz * cosY;

    const qx = Math.max(-halfW, Math.min(halfW, lx));
    const qz = Math.max(-halfL, Math.min(halfL, lz));
    const rx = lx - qx;
    const rz = lz - qz;
    const distSq = rx * rx + rz * rz;

    if (distSq > 1e-12) {
      const dist = Math.sqrt(distSq);
      const pen = radius - dist;
      if (pen <= 0.0) {
        out.depth = 0.0;
        return 0.0;
      }
      const lnx = rx / dist;
      const lnz = rz / dist;
      out.nx = lnx * cosY + lnz * sinY;
      out.ny = 0.0;
      out.nz = -lnx * sinY + lnz * cosY;
      out.depth = pen;
      return pen;
    }

    // Circle center is strictly inside OBB -> push out along nearest OBB face
    const penX = halfW - Math.abs(lx);
    const penZ = halfL - Math.abs(lz);
    if (penX <= penZ) {
      const sx = lx >= 0.0 ? 1.0 : -1.0;
      out.nx = sx * cosY;
      out.ny = 0.0;
      out.nz = -sx * sinY;
      out.depth = radius + penX;
      return out.depth;
    }
    const sz = lz >= 0.0 ? 1.0 : -1.0;
    out.nx = sz * sinY;
    out.ny = 0.0;
    out.nz = sz * cosY;
    out.depth = radius + penZ;
    return out.depth;
  }

  /**
   * Computes exact 3D Sphere vs Solid Collider (BOX/AABB, CYL, OBB) Minimum Translation Vector (MTV).
   * Handles lateral wall pushes AND vertical top-roof / underside-soffit pushes seamlessly!
   */
  static computeSphere3DColliderMTV(px, py, pz, radius, col, out = _mtvScratch) {
    const kind = col.kind || col.type;
    if (col.bMinX !== undefined) {
      if (
        px + radius <= col.bMinX ||
        px - radius >= col.bMaxX ||
        py + radius <= col.bMinY ||
        py - radius >= col.bMaxY ||
        pz + radius <= col.bMinZ ||
        pz - radius >= col.bMaxZ
      ) {
        out.depth = 0.0;
        return 0.0;
      }
    }

    if (kind === 'BOX' || kind === 'AABB') {
      const qx = Math.max(col.minX, Math.min(col.maxX, px));
      const qy = Math.max(col.minY, Math.min(col.maxY, py));
      const qz = Math.max(col.minZ, Math.min(col.maxZ, pz));
      const rx = px - qx;
      const ry = py - qy;
      const rz = pz - qz;
      const dSq = rx * rx + ry * ry + rz * rz;
      if (dSq > 1e-12) {
        const d = Math.sqrt(dSq);
        const pen = radius - d;
        if (pen <= 0.0) {
          out.depth = 0.0;
          return 0.0;
        }
        out.nx = rx / d;
        out.ny = ry / d;
        out.nz = rz / d;
        out.depth = pen;
        return pen;
      }
      // Sphere center strictly inside AABB: choose minimum-escape face among 6 faces
      const dL = px - col.minX;
      const dR = col.maxX - px;
      const dB = py - col.minY;
      const dT = col.maxY - py;
      const dF = pz - col.minZ;
      const dK = col.maxZ - pz;
      const m = Math.min(dL, dR, dB, dT, dF, dK);
      out.nx = 0.0;
      out.ny = 0.0;
      out.nz = 0.0;
      if (m === dT) out.ny = 1.0;
      else if (m === dB) out.ny = -1.0;
      else if (m === dL) out.nx = -1.0;
      else if (m === dR) out.nx = 1.0;
      else if (m === dF) out.nz = -1.0;
      else out.nz = 1.0;
      out.depth = radius + m;
      return out.depth;
    }

    if (kind === 'CYL') {
      if (py + radius <= col.minY || py - radius >= col.maxY) {
        out.depth = 0.0;
        return 0.0;
      }
      const dx = px - col.x;
      const dz = pz - col.z;
      const rHoriz = Math.hypot(dx, dz);
      const qy = Math.max(col.minY, Math.min(col.maxY, py));
      const ry = py - qy;
      const radialExcess = Math.max(0.0, rHoriz - col.r);
      const dSq = radialExcess * radialExcess + ry * ry;

      if (dSq > 1e-12) {
        const d = Math.sqrt(dSq);
        const pen = radius - d;
        if (pen <= 0.0) {
          out.depth = 0.0;
          return 0.0;
        }
        if (radialExcess > 1e-6 && rHoriz > 1e-6) {
          out.nx = ((dx / rHoriz) * radialExcess) / d;
          out.nz = ((dz / rHoriz) * radialExcess) / d;
        } else {
          out.nx = 0.0;
          out.nz = 0.0;
        }
        out.ny = ry / d;
        out.depth = pen;
        return pen;
      }
      // Sphere center strictly inside solid cylinder
      const penR = col.r - rHoriz;
      const penTop = col.maxY - py;
      const penBot = py - col.minY;
      if (penTop <= penR && penTop <= penBot) {
        out.nx = 0.0;
        out.ny = 1.0;
        out.nz = 0.0;
        out.depth = radius + penTop;
        return out.depth;
      }
      if (penBot <= penR && penBot <= penTop) {
        out.nx = 0.0;
        out.ny = -1.0;
        out.nz = 0.0;
        out.depth = radius + penBot;
        return out.depth;
      }
      if (rHoriz > 1e-6) {
        out.nx = dx / rHoriz;
        out.nz = dz / rHoriz;
      } else {
        out.nx = 1.0;
        out.nz = 0.0;
      }
      out.ny = 0.0;
      out.depth = radius + penR;
      return out.depth;
    }

    if (kind === 'OBB') {
      if (col._effCx === undefined) PhysicsKernel.prepareSolidCollider(col);
      const dx = px - col._effCx;
      const dz = pz - col._effCz;
      const sinY = col._sinY;
      const cosY = col._cosY;
      const lx = dx * cosY - dz * sinY;
      const lz = dx * sinY + dz * cosY;

      const qx = Math.max(-col.halfW, Math.min(col.halfW, lx));
      const qy = Math.max(col.minY, Math.min(col.maxY, py));
      const qz = Math.max(-col._effHalfL, Math.min(col._effHalfL, lz));

      const rx = lx - qx;
      const ry = py - qy;
      const rz = lz - qz;
      const dSq = rx * rx + ry * ry + rz * rz;
      if (dSq > 1e-12) {
        const d = Math.sqrt(dSq);
        const pen = radius - d;
        if (pen <= 0.0) {
          out.depth = 0.0;
          return 0.0;
        }
        const lnx = rx / d;
        const lnz = rz / d;
        out.nx = lnx * cosY + lnz * sinY;
        out.ny = ry / d;
        out.nz = -lnx * sinY + lnz * cosY;
        out.depth = pen;
        return pen;
      }
      // Inside 3D OBB
      const penX = col.halfW - Math.abs(lx);
      const penZ = col._effHalfL - Math.abs(lz);
      const penTop = col.maxY - py;
      const penBot = py - col.minY;
      const m = Math.min(penX, penZ, penTop, penBot);
      if (m === penTop) {
        out.nx = 0.0;
        out.ny = 1.0;
        out.nz = 0.0;
      } else if (m === penBot) {
        out.nx = 0.0;
        out.ny = -1.0;
        out.nz = 0.0;
      } else if (m === penX) {
        const sx = lx >= 0.0 ? 1.0 : -1.0;
        out.nx = sx * cosY;
        out.ny = 0.0;
        out.nz = -sx * sinY;
      } else {
        const sz = lz >= 0.0 ? 1.0 : -1.0;
        out.nx = sz * sinY;
        out.ny = 0.0;
        out.nz = sz * cosY;
      }
      out.depth = radius + m;
      return out.depth;
    }

    out.depth = 0.0;
    return 0.0;
  }

  /**
   * Hard 2D/Vertical Vertical-Capsule depenetration for terrestrial agents (NPCs & Cats) against a solid collider.
   * Pushes horizontally (x, z) out of any overlapping solid volume spanning [minY, maxY].
   */
  static depenetrateHorizontalCylinder(agentPos, minY, maxY, radius, col) {
    if (col.bMinY !== undefined && (maxY <= col.bMinY || minY >= col.bMaxY)) return false;
    if (col.bMinX !== undefined) {
      if (
        agentPos.x + radius <= col.bMinX ||
        agentPos.x - radius >= col.bMaxX ||
        agentPos.z + radius <= col.bMinZ ||
        agentPos.z - radius >= col.bMaxZ
      ) {
        return false;
      }
    }

    const kind = col.kind || col.type;
    if (kind === 'CYL') {
      const dx = agentPos.x - col.x;
      const dz = agentPos.z - col.z;
      const minR = radius + col.r;
      const dSq = dx * dx + dz * dz;
      if (dSq >= minR * minR) return false;
      const d = Math.sqrt(dSq);
      if (d > 1e-6) {
        const push = (minR - d) / d;
        agentPos.x += dx * push;
        agentPos.z += dz * push;
      } else {
        agentPos.x += minR;
      }
      return true;
    }

    if (kind === 'BOX' || kind === 'AABB') {
      const cx = 0.5 * (col.minX + col.maxX);
      const cz = 0.5 * (col.minZ + col.maxZ);
      const hw = 0.5 * (col.maxX - col.minX);
      const hl = 0.5 * (col.maxZ - col.minZ);
      const pen = PhysicsKernel.computeOBB2DCircleMTV(
        cx,
        cz,
        hw,
        hl,
        0.0,
        1.0,
        agentPos.x,
        agentPos.z,
        radius,
        _mtvScratch
      );
      if (pen <= 0.0) return false;
      agentPos.x += _mtvScratch.nx * pen;
      agentPos.z += _mtvScratch.nz * pen;
      return true;
    }

    if (kind === 'OBB') {
      if (col._effCx === undefined) PhysicsKernel.prepareSolidCollider(col);
      const pen = PhysicsKernel.computeOBB2DCircleMTV(
        col._effCx,
        col._effCz,
        col.halfW,
        col._effHalfL,
        col._sinY,
        col._cosY,
        agentPos.x,
        agentPos.z,
        radius,
        _mtvScratch
      );
      if (pen <= 0.0) return false;
      agentPos.x += _mtvScratch.nx * pen;
      agentPos.z += _mtvScratch.nz * pen;
      return true;
    }

    return false;
  }

  /**
   * 2D SAT Minimum Translation Vector (MTV) pushing dynamic OBB `a` out of static collider `col` (CYL, BOX/AABB, OBB).
   */
  static depenetrateOBB2DVsCollider(obb, col) {
    if (obb.maxY !== undefined && col.minY !== undefined) {
      if (obb.maxY <= col.minY || obb.minY >= col.maxY) return false;
    }
    const kind = col.kind || col.type;
    const fLA = obb.frontL !== undefined ? obb.frontL : obb.halfL;
    const bLA = obb.backL !== undefined ? obb.backL : obb.halfL;
    const midOffA = 0.5 * (fLA - bLA);
    const halfLA = 0.5 * (fLA + bLA);
    const sinA = Math.sin(obb.yaw || 0.0);
    const cosA = Math.cos(obb.yaw || 0.0);
    const cxA = obb.cx + sinA * midOffA;
    const czA = obb.cz + cosA * midOffA;

    if (kind === 'CYL') {
      // Circle vs OBB gives MTV pushing circle out of OBB -> negate to push OBB out of static cylinder!
      const pen = PhysicsKernel.computeOBB2DCircleMTV(
        cxA,
        czA,
        obb.halfW,
        halfLA,
        sinA,
        cosA,
        col.x,
        col.z,
        col.r,
        _mtvScratch
      );
      if (pen <= 0.0) return false;
      obb.cx -= _mtvScratch.nx * pen;
      obb.cz -= _mtvScratch.nz * pen;
      return true;
    }

    let cxB, czB, hwB, hlB, sinB, cosB;
    if (kind === 'BOX' || kind === 'AABB') {
      cxB = 0.5 * (col.minX + col.maxX);
      czB = 0.5 * (col.minZ + col.maxZ);
      hwB = 0.5 * (col.maxX - col.minX);
      hlB = 0.5 * (col.maxZ - col.minZ);
      sinB = 0.0;
      cosB = 1.0;
    } else if (kind === 'OBB') {
      if (col._effCx === undefined) PhysicsKernel.prepareSolidCollider(col);
      cxB = col._effCx;
      czB = col._effCz;
      hwB = col.halfW;
      hlB = col._effHalfL;
      sinB = col._sinY;
      cosB = col._cosY;
    } else {
      return false;
    }

    _satAxes[0][0] = cosA;
    _satAxes[0][1] = -sinA;
    _satAxes[1][0] = sinA;
    _satAxes[1][1] = cosA;
    _satAxes[2][0] = cosB;
    _satAxes[2][1] = -sinB;
    _satAxes[3][0] = sinB;
    _satAxes[3][1] = cosB;

    const fillCorners = (out, cx, cz, hw, hl, sY, cY) => {
      const rx = cY * hw;
      const rz = -sY * hw;
      const fx = sY * hl;
      const fz = cY * hl;
      out[0][0] = cx - rx - fx;
      out[0][1] = cz - rz - fz;
      out[1][0] = cx + rx - fx;
      out[1][1] = cz + rz - fz;
      out[2][0] = cx + rx + fx;
      out[2][1] = cz + rz + fz;
      out[3][0] = cx - rx + fx;
      out[3][1] = cz - rz + fz;
    };

    fillCorners(_cornersA, cxA, czA, obb.halfW, halfLA, sinA, cosA);
    fillCorners(_cornersB, cxB, czB, hwB, hlB, sinB, cosB);

    let minOverlap = Infinity;
    let bestAx = 0.0;
    let bestAz = 0.0;

    for (let i = 0; i < 4; i++) {
      const ax = _satAxes[i][0];
      const az = _satAxes[i][1];
      let minA = Infinity;
      let maxA = -Infinity;
      let minB = Infinity;
      let maxB = -Infinity;
      for (let k = 0; k < 4; k++) {
        const pA = _cornersA[k][0] * ax + _cornersA[k][1] * az;
        if (pA < minA) minA = pA;
        if (pA > maxA) maxA = pA;
        const pB = _cornersB[k][0] * ax + _cornersB[k][1] * az;
        if (pB < minB) minB = pB;
        if (pB > maxB) maxB = pB;
      }
      const overlap = Math.min(maxA, maxB) - Math.max(minA, minB);
      if (overlap <= 0.0) return false;
      if (overlap < minOverlap) {
        minOverlap = overlap;
        const dir = (cxA - cxB) * ax + (czA - czB) * az >= 0.0 ? 1.0 : -1.0;
        bestAx = ax * dir;
        bestAz = az * dir;
      }
    }

    obb.cx += bestAx * minOverlap;
    obb.cz += bestAz * minOverlap;
    return true;
  }

  /**
   * Exact Two-Regime Topological Solid Solver for Underpass Bawah Laut Selatan U–I:
   * - Inside sunken ramp trench or subsea tunnel bore: strictly blocks lateral passage through concrete walls
   *   (|x - ux| <= boreHalfW - radius) and vertical passage through floor or arched subsea vault ceiling!
   * - Outside trench / on hillside or over Selat UI ocean: strictly blocks sideways clipping through outer
   *   retaining walls into the sunken ramp and blocks downward diving through portal headwalls or seabed!
   */
  static resolveUnderpassTopologicalSolids(pos, prevPos, vel, radius, topOffset, bottomOffset, cfg) {
    if (!cfg) return false;
    const ux = cfg.ux; // -356.0
    const boreHW = cfg.boreHalfW; // 4.55
    const parapetOuterHW = boreHW + 0.72; // 5.27 (outer retaining face + thickness)
    const zW0 = cfg.zWestStart; // 272.0
    const zWP = cfg.zWestPortal; // 326.0
    const zEP = cfg.zEastPortal; // 624.0
    const zE1 = cfg.zEastEnd; // 670.0
    const floorY = cfg.floorY; // -8.20
    const clearH = cfg.clearH; // 4.15 -> crown -4.05

    // Determine whether the body was/is legitimately inside the underpass channel/bore
    const curDx = Math.abs(pos.x - ux);
    const prevDx = prevPos ? Math.abs(prevPos.x - ux) : curDx;
    const prevZ = prevPos ? prevPos.z : pos.z;
    const prevY = prevPos ? prevPos.y : pos.y;

    if (Math.max(pos.z, prevZ) < zW0 - radius - 1.0 || Math.min(pos.z, prevZ) > zE1 + radius + 1.0) return false;
    if (Math.min(curDx, prevDx) > parapetOuterHW + radius + 6.0) return false;

    let modified = false;

    const wasInSubseaBore =
      prevZ >= zWP - 1.5 &&
      prevZ <= zEP + 1.5 &&
      prevDx <= boreHW + 0.25 &&
      prevY <= floorY + clearH + 0.35;

    const wasInOpenRamp =
      ((prevZ >= zW0 - 1.0 && prevZ <= zWP + 1.5) || (prevZ >= zEP - 1.5 && prevZ <= zE1 + 1.0)) &&
      prevDx <= boreHW + 0.15;

    // Regime 1: Inside Enclosed Subsea Tunnel Bore (z in [326, 624])
    if (wasInSubseaBore || (pos.z >= zWP && pos.z <= zEP && Math.abs(pos.x - ux) <= boreHW && pos.y < floorY + clearH - 0.15)) {
      if (pos.z >= zWP - 0.2 && pos.z <= zEP + 0.2) {
        const maxInX = Math.max(0.5, boreHW - radius);
        const dx = pos.x - ux;
        if (Math.abs(dx) > maxInX) {
          const sgn = dx >= 0.0 ? 1.0 : -1.0;
          pos.x = ux + sgn * maxInX;
          if (vel && vel.x * sgn > 0) vel.x = 0.0;
          modified = true;
        }
        // Arched concrete vault ceiling y_vault(x)
        const normX = Math.min(0.98, Math.abs(pos.x - ux) / boreHW);
        const vaultY = floorY + 2.20 + Math.sqrt(Math.max(0.0, 1.0 - normX * normX)) * (clearH - 2.20);
        const maxCamY = vaultY - Math.max(radius, topOffset);
        if (pos.y > maxCamY) {
          pos.y = maxCamY;
          if (vel && vel.y > 0) vel.y = 0.0;
          modified = true;
        }
        // Solid tunnel floor + sidewalk
        const onSidewalk = Math.abs(pos.x - ux) > cfg.roadHalfW;
        const minCamY = floorY + (onSidewalk ? 0.18 : 0.0) + Math.max(radius * 0.5, bottomOffset);
        if (pos.y < minCamY) {
          pos.y = minCamY;
          if (vel && vel.y < 0) vel.y = 0.0;
          modified = true;
        }
        return modified;
      }
    }

    // Regime 2: Open-Cut Approach Ramps (West z in [272, 326], East z in [624, 670])
    const inWestRampZ = pos.z >= zW0 && pos.z <= zWP;
    const inEastRampZ = pos.z >= zEP && pos.z <= zE1;
    if (inWestRampZ || inEastRampZ) {
      const rampDeckY = cfg.sampleRampY(ux, pos.z);
      const surfSideY = Math.max(
        0.55,
        cfg.sampleGroundY(ux + (pos.x >= ux ? 1.0 : -1.0) * 6.5, pos.z)
      );
      const parapetCrownY = surfSideY + 0.52;

      if (wasInOpenRamp || wasInSubseaBore) {
        // Entity is INSIDE the open-cut ramp trench: if below parapet crown, retaining walls block lateral exit!
        if (pos.y - bottomOffset < parapetCrownY) {
          const maxInX = Math.max(0.5, boreHW - radius);
          const dx = pos.x - ux;
          if (Math.abs(dx) > maxInX) {
            const sgn = dx >= 0.0 ? 1.0 : -1.0;
            pos.x = ux + sgn * maxInX;
            if (vel && vel.x * sgn > 0) vel.x = 0.0;
            modified = true;
          }
        }
        // Portal Headwall Lintel right at zWP / zEP: if above tunnel arch crown (-4.05m), block crossing into subsea Z!
        const archMaxY = floorY + clearH - Math.max(radius, topOffset);
        if (inWestRampZ && pos.z > zWP - radius && pos.y > archMaxY) {
          pos.z = zWP - radius;
          if (vel && vel.z > 0) vel.z = 0.0;
          modified = true;
        } else if (inEastRampZ && pos.z < zEP + radius && pos.y > archMaxY) {
          pos.z = zEP + radius;
          if (vel && vel.z < 0) vel.z = 0.0;
          modified = true;
        }
        if (rampDeckY !== null) {
          const onSidewalk = Math.abs(pos.x - ux) > cfg.roadHalfW;
          const minY = rampDeckY + (onSidewalk ? 0.18 : 0.0) + bottomOffset;
          if (pos.y < minY) {
            pos.y = minY;
            if (vel && vel.y < 0) vel.y = 0.0;
            modified = true;
          }
        }
      } else {
        // Entity is OUTSIDE on the hillside: cannot clip sideways through retaining wall below parapetCrownY!
        const minOutX = parapetOuterHW + radius;
        const dx = pos.x - ux;
        if (Math.abs(dx) < minOutX && pos.y - bottomOffset < parapetCrownY + 0.15) {
          const sgn = (prevPos ? prevPos.x - ux : dx) >= 0.0 ? 1.0 : -1.0;
          pos.x = ux + sgn * minOutX;
          if (vel && vel.x * sgn < 0) vel.x = 0.0;
          modified = true;
        }
      }
    }

    return modified;
  }

  /**
   * Hard Iterative 3D Camera & Drone Non-Penetration Solver (3 Gauss-Seidel passes + Sub-Stepped Sweep).
   * Guarantees 0.000m penetration into buildings, roofs, columns, trees, lamps, bridges, underpass walls/vault,
   * terrain/volcano/hills, ocean/freshwater surfaces, and live dynamic vehicles/vessels!
   */
  static resolveHardCameraCollision3D(pos, prevPos, vel, options = {}) {
    const {
      staticColliders = [],
      dynamicColliders = [],
      radius = 0.9,
      bottomOffset = radius,
      topOffset = radius,
      isWalk = false,
      groundHeightFn = null,
      walkableHeightFn = null,
      coastDistFn = null,
      waterSurfaceFn = null,
      oceanWaterY = -1.45,
      underpassConfig = null,
      bridgeSoffitFn = null,
      bridgeDeckFn = null,
      bridgeCenterX = -28.0,
      bridgeHalfW = 2.6,
      ferryPierDeckFn = null
    } = options;

    let hitAny = false;

    const cancelInwardVel = (nx, ny, nz) => {
      if (!vel) return;
      const vn = vel.x * nx + vel.y * ny + vel.z * nz;
      if (vn < 0.0) {
        vel.x -= vn * nx;
        vel.y -= vn * ny;
        vel.z -= vn * nz;
      }
    };

    const solveSinglePosition = (p, pPrev) => {
      for (let iter = 0; iter < 3; iter++) {
        // 1. Underpass Bawah Laut Two-Regime Topological Solids (Retaining Walls, Headwalls, Bore Walls, Vault Ceiling)
        if (
          underpassConfig &&
          PhysicsKernel.resolveUnderpassTopologicalSolids(
            p,
            pPrev,
            vel,
            radius,
            topOffset,
            bottomOffset,
            underpassConfig
          )
        ) {
          hitAny = true;
        }

        const inEnclosedSubseaBore =
          underpassConfig &&
          p.z >= underpassConfig.zWestPortal - 0.3 &&
          p.z <= underpassConfig.zEastPortal + 0.3 &&
          Math.abs(p.x - underpassConfig.ux) <= underpassConfig.boreHalfW + 0.1 &&
          p.y <= underpassConfig.floorY + underpassConfig.clearH + 0.1;

        const inOpenUnderpassRamp =
          !inEnclosedSubseaBore &&
          underpassConfig &&
          p.z >= underpassConfig.zWestStart &&
          p.z <= underpassConfig.zEastEnd &&
          Math.abs(p.x - underpassConfig.ux) <= underpassConfig.boreHalfW + 0.05 &&
          p.y < 1.8;

        // 2. Jembatan Teksas UI 3D Box-Girder Slab & Overhead Sunroof Canopy Solid Separation
        if (!isWalk && bridgeDeckFn && Math.abs(p.x - bridgeCenterX) < bridgeHalfW + radius + 0.35) {
          const bDeckY = bridgeDeckFn(bridgeCenterX, p.z);
          if (bDeckY !== null) {
            const bSoffitY = bridgeSoffitFn ? (bridgeSoffitFn(p.z) ?? bDeckY - 1.3) : bDeckY - 1.3;
            // Solid steel box girder slab [bSoffitY, bDeckY]
            if (p.y + topOffset > bSoffitY && p.y - bottomOffset < bDeckY) {
              const midSlab = 0.5 * (bSoffitY + bDeckY);
              if ((pPrev ? pPrev.y : p.y) >= midSlab) {
                p.y = bDeckY + bottomOffset;
                cancelInwardVel(0, 1, 0);
              } else {
                p.y = bSoffitY - topOffset;
                cancelInwardVel(0, -1, 0);
              }
              hitAny = true;
            }
            // Overhead 30-unit Blue Wave Sunroof shell on main arch span [bDeckY + 2.95, bDeckY + 3.55]
            if (p.z >= 174.5 && p.z <= 316.8 && Math.abs(p.x - bridgeCenterX) < 2.52 + radius) {
              const roofBot = bDeckY + 2.95;
              const roofTop = bDeckY + 3.58;
              if (p.y + topOffset > roofBot && p.y - bottomOffset < roofTop) {
                if ((pPrev ? pPrev.y : p.y) >= 0.5 * (roofBot + roofTop)) {
                  p.y = roofTop + bottomOffset;
                  cancelInwardVel(0, 1, 0);
                } else {
                  p.y = roofBot - topOffset;
                  cancelInwardVel(0, -1, 0);
                }
                hitAny = true;
              }
            }
          }
        }

        // 3. Static Solid World Colliders (Buildings, Columns, Spires, Roofs, Trees, Lamps, Benches, Piers, Pylons)
        for (let i = 0; i < staticColliders.length; i++) {
          const c = staticColliders[i];
          if (c.type === 'ISLAND_PERIMETER') {
            if (!isWalk) continue;
            // Hard pedestrian railing/shoreline containment in WALK_SIVITAS mode!
            if (bridgeDeckFn && bridgeDeckFn(p.x, p.z) !== null) {
              const maxHW = c.bridgeWalkHalfW || 2.25;
              const dxB = p.x - bridgeCenterX;
              if (Math.abs(dxB) > maxHW) {
                const sgn = dxB >= 0 ? 1 : -1;
                p.x = bridgeCenterX + sgn * maxHW;
                cancelInwardVel(-sgn, 0, 0);
                hitAny = true;
              }
              continue;
            }
            if (inEnclosedSubseaBore || inOpenUnderpassRamp) continue;
            if (ferryPierDeckFn && ferryPierDeckFn(p.x, p.z) !== null) continue;
            if (typeof c.coastDistFn === 'function') {
              const dC = c.coastDistFn(p.x, p.z);
              const maxEdge = -(c.margin || 0.9);
              if (dC > maxEdge) {
                const eps = 0.35;
                const gx = c.coastDistFn(p.x + eps, p.z) - c.coastDistFn(p.x - eps, p.z);
                const gz = c.coastDistFn(p.x, p.z + eps) - c.coastDistFn(p.x, p.z - eps);
                const gL = Math.hypot(gx, gz) || 1.0;
                const nxIn = -gx / gL;
                const nzIn = -gz / gL;
                const push = dC - maxEdge;
                p.x += nxIn * push;
                p.z += nzIn * push;
                cancelInwardVel(nxIn, 0, nzIn);
                hitAny = true;
              }
            }
            continue;
          }

          if (isWalk) {
            // Walk capsule spans [feetY + 0.12 .. eyeY + 0.22]
            const feetY = p.y - 1.56;
            const headY = p.y + 0.24;
            if (c.bMinY !== undefined && (headY <= c.bMinY || feetY >= c.bMaxY)) continue;
            if (PhysicsKernel.depenetrateHorizontalCylinder(p, feetY, headY, radius, c)) {
              hitAny = true;
            }
          } else {
            const pen = PhysicsKernel.computeSphere3DColliderMTV(p.x, p.y, p.z, radius, c, _mtvScratch);
            if (pen > 0.0) {
              p.x += _mtvScratch.nx * pen;
              p.y += _mtvScratch.ny * pen;
              p.z += _mtvScratch.nz * pen;
              cancelInwardVel(_mtvScratch.nx, _mtvScratch.ny, _mtvScratch.nz);
              hitAny = true;
            }
          }
        }

        // 4. Live Dynamic Solid Colliders (Bus Kuning UI, Mobil Golf UI, Feri Katamaran UI, Mini Yacht, NPCs)
        for (let i = 0; i < dynamicColliders.length; i++) {
          const dc = dynamicColliders[i];
          if (!dc || dc.enabled === false) continue;
          if (isWalk) {
            const feetY = p.y - 1.56;
            const headY = p.y + 0.24;
            if (dc.bMinY !== undefined && (headY <= dc.bMinY || feetY >= dc.bMaxY)) continue;
            if (PhysicsKernel.depenetrateHorizontalCylinder(p, feetY, headY, radius, dc)) {
              hitAny = true;
            }
          } else {
            const pen = PhysicsKernel.computeSphere3DColliderMTV(p.x, p.y, p.z, radius, dc, _mtvScratch);
            if (pen > 0.0) {
              p.x += _mtvScratch.nx * pen;
              p.y += _mtvScratch.ny * pen;
              p.z += _mtvScratch.nz * pen;
              cancelInwardVel(_mtvScratch.nx, _mtvScratch.ny, _mtvScratch.nz);
              hitAny = true;
            }
          }
        }

        // 5. Solid Terrain Heightfield & Solid Ocean / Freshwater Surface Non-Penetration!
        if (!inEnclosedSubseaBore && !inOpenUnderpassRamp) {
          if (isWalk && walkableHeightFn) {
            // In walk mode, keep feet Out of freshwater lakes/rivers too!
            if (waterSurfaceFn) {
              const fwY = waterSurfaceFn(p.x, p.z);
              const rawG = groundHeightFn ? groundHeightFn(p.x, p.z) : 0.0;
              if (fwY !== null && rawG < fwY - 0.04 && pPrev) {
                p.x = pPrev.x;
                p.z = pPrev.z;
                if (vel) {
                  vel.x = 0.0;
                  vel.z = 0.0;
                }
                hitAny = true;
              }
            }
            const wFloor = walkableHeightFn(p.x, p.z) + 1.68;
            if (p.y < wFloor) {
              p.y = wFloor;
              cancelInwardVel(0, 1, 0);
              hitAny = true;
            }
          } else {
            let minSolidSurfaceY = groundHeightFn ? groundHeightFn(p.x, p.z) : 0.0;
            // Solid Dermaga Pier Deck when flying right over pier slab
            if (ferryPierDeckFn) {
              const pierY = ferryPierDeckFn(p.x, p.z);
              if (pierY !== null && (pPrev ? pPrev.y : p.y) >= pierY - 0.25) {
                minSolidSurfaceY = Math.max(minSolidSurfaceY, pierY);
              }
            }
            // Solid Ocean / Strait / Teluk U Water Surface Barrier (prevents diving underwater!)
            if (coastDistFn) {
              const dC = coastDistFn(p.x, p.z);
              if (dC > -0.5) {
                minSolidSurfaceY = Math.max(minSolidSurfaceY, oceanWaterY);
              }
            } else if (minSolidSurfaceY < oceanWaterY) {
              minSolidSurfaceY = oceanWaterY;
            }
            // Solid Freshwater Spring / River / Plunge Pool Water Surface Barrier
            if (waterSurfaceFn) {
              const fwY = waterSurfaceFn(p.x, p.z);
              if (fwY !== null) {
                minSolidSurfaceY = Math.max(minSolidSurfaceY, fwY);
              }
            }
            const reqY = minSolidSurfaceY + bottomOffset;
            if (p.y < reqY) {
              p.y = reqY;
              cancelInwardVel(0, 1, 0);
              hitAny = true;
            }
          }
        }
      }
    };

    // Sub-stepped continuous sweep from prevPos -> pos whenever displacement exceeds 0.55 * radius
    // so fast 28 m/s drones or sharp Orbit swings can NEVER tunnel across walls or columns!
    if (prevPos) {
      const dx = pos.x - prevPos.x;
      const dy = pos.y - prevPos.y;
      const dz = pos.z - prevPos.z;
      const moveDist = Math.hypot(dx, dy, dz);
      const maxStep = Math.max(0.28, radius * 0.6);
      if (moveDist > maxStep && moveDist < 260.0) {
        const steps = Math.min(256, Math.ceil(moveDist / maxStep));
        let curX = prevPos.x;
        let curY = prevPos.y;
        let curZ = prevPos.z;
        const stepX = dx / steps;
        const stepY = dy / steps;
        const stepZ = dz / steps;
        const subPrev = { x: curX, y: curY, z: curZ };
        for (let s = 1; s <= steps; s++) {
          subPrev.x = curX;
          subPrev.y = curY;
          subPrev.z = curZ;
          pos.x = curX + stepX;
          pos.y = curY + stepY;
          pos.z = curZ + stepZ;
          solveSinglePosition(pos, subPrev);
          curX = pos.x;
          curY = pos.y;
          curZ = pos.z;
        }
        return hitAny;
      }
    }

    solveSinglePosition(pos, prevPos);
    return hitAny;
  }
}
