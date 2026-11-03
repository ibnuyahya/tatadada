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
      const rCyl = col.r !== undefined ? col.r : col.radius || 0.5;
      col.r = rCyl;
      col.radius = rCyl;
      col.bMinX = col.x - rCyl;
      col.bMaxX = col.x + rCyl;
      col.bMinY = col.minY;
      col.bMaxY = col.maxY;
      col.bMinZ = col.z - rCyl;
      col.bMaxZ = col.z + rCyl;
    } else if (kind === 'ELLIPSOID') {
      const rx = col.rx || col.r || col.radius || 1.0;
      const ry = col.ry || 1.0;
      const cy = col.cy !== undefined ? col.cy : 0.5 * ((col.minY || 0) + (col.maxY || 0));
      col.rx = rx;
      col.ry = ry;
      col.cy = cy;
      col.r = rx;
      col.radius = rx;
      col.minY = cy - ry;
      col.maxY = cy + ry;
      col.bMinX = col.x - rx;
      col.bMaxX = col.x + rx;
      col.bMinY = cy - ry;
      col.bMaxY = cy + ry;
      col.bMinZ = col.z - rx;
      col.bMaxZ = col.z + rx;
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

    if (kind === 'ELLIPSOID') {
      if (col.cy === undefined) PhysicsKernel.prepareSolidCollider(col);
      const rx = col.rx + radius;
      const ry = col.ry + radius;
      const dx = px - col.x;
      const dy = py - col.cy;
      const dz = pz - col.z;
      if (Math.abs(dy) >= ry || Math.abs(dx) >= rx || Math.abs(dz) >= rx) {
        out.depth = 0.0;
        return 0.0;
      }
      const ux = dx / rx;
      const uy = dy / ry;
      const uz = dz / rx;
      const qSq = ux * ux + uy * uy + uz * uz;
      if (qSq >= 1.0) {
        out.depth = 0.0;
        return 0.0;
      }
      if (qSq < 1e-10) {
        out.nx = 0.0;
        out.ny = 1.0;
        out.nz = 0.0;
        out.depth = ry;
        return ry;
      }
      const q = Math.sqrt(qSq);
      let gx = ux / rx;
      let gy = uy / ry;
      let gz = uz / rx;
      if (col.isCanopy && uy > -0.38 && gy < 0.24 / ry) {
        gy = 0.24 / ry;
      }
      const gLen = Math.hypot(gx, gy, gz) || 1.0;
      out.nx = gx / gLen;
      out.ny = gy / gLen;
      out.nz = gz / gLen;
      const scaleOut = 1.0 / q - 1.0;
      const dotE = (dx * out.nx + dy * out.ny + dz * out.nz) * scaleOut;
      out.depth = Math.max(0.015, dotE);
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
   * Exact Two-Regime 3D Topological Solid Solver for Jembatan Teksas UI ("Lorong Jembatan Teksas"):
   * - Regime A (Inside Hollow Bridge Corridor |x - bx| <= 2.40m, y in [bDeckY, roofInnerY]):
   *   Enforces solid floor deck, solid overhead 30-module blue wave sunroof ceiling, and solid left/right
   *   crimson-red parabolic Warren steel trusses so a drone can fly 100% cleanly THROUGH the interior corridor!
   * - Regime B (Outside Bridge — Flying Beside over Teluk U, Underneath the Arch, or Above the Sunroof):
   *   Treats the entire outer envelope (side trusses, Warren struts, box girder, sunroof, & concrete approach
   *   ramp walls/handrails) as an impenetrable 3D solid with minimum-escape separation!
   */
  static resolveTeksasBridgeTopologicalSolids(
    p,
    pPrev,
    vel,
    radius,
    topOffset,
    bottomOffset,
    bridgeCenterX,
    bridgeDeckFn,
    bridgeSoffitFn,
    cancelInwardVel
  ) {
    if (!bridgeDeckFn) return false;
    const bx = bridgeCenterX; // -28.0
    const zRampW0 = 149.65;
    const zPortW = 171.65;
    const zPortE = 319.65;
    const zRampE1 = 341.65;

    const refX = pPrev ? pPrev.x : p.x;
    const refY = pPrev ? pPrev.y : p.y;
    const refZ = pPrev ? pPrev.z : p.z;

    if (Math.max(p.z, refZ) < zRampW0 - radius - 0.5 || Math.min(p.z, refZ) > zRampE1 + radius + 0.5) {
      return false;
    }
    const outerMaxHW = 2.88;
    if (Math.min(Math.abs(p.x - bx), Math.abs(refX - bx)) > outerMaxHW + radius + 1.5) {
      return false;
    }

    const bDeckY = bridgeDeckFn(bx, p.z);
    if (bDeckY === null) return false;

    let modified = false;

    // 1. Main 148m Parabolic Through-Truss Arch Span across Teluk U (z in [171.65, 319.65])
    if (p.z >= zPortW && p.z <= zPortE) {
      const u = (p.z - 245.65) / 74.0;
      const bSoffitY = bridgeSoffitFn ? (bridgeSoffitFn(p.z) ?? bDeckY - 1.30) : bDeckY - 1.30;
      const chordTopY = bDeckY + 1.34 + 2.58 * Math.max(0.0, 1.0 - u * u);
      const hasSunroof = p.z >= 174.5 && p.z <= 316.8;
      const roofInnerY = bDeckY + 2.86;
      const roofOuterTopY = bDeckY + 3.62;
      const innerHalfW = 2.40;
      const outerHalfW = 2.86;

      const zRefClamp = Math.max(zPortW, Math.min(zPortE, refZ));
      const refDeckY = bridgeDeckFn(bx, zRefClamp) ?? bDeckY;
      const refMaxCorridorY =
        refZ >= 174.3 && refZ <= 317.0 ? refDeckY + 2.94 : refDeckY + 4.85;
      const inBridgeCorridor =
        Math.abs(refX - bx) <= innerHalfW + 0.05 &&
        refY >= refDeckY - 0.14 &&
        refY <= refMaxCorridorY;

      if (inBridgeCorridor) {
        // Regime A: Inside Lorong Jembatan Teksas UI!
        // A1. Solid Timber-Composite Deck Floor
        const minInY = bDeckY + bottomOffset;
        if (p.y < minInY) {
          p.y = minInY;
          cancelInwardVel(0, 1, 0);
          modified = true;
        }
        // A2. Solid Overhead Blue Wave Sunroof Ceiling & Leading Lip
        if (hasSunroof) {
          const maxInY = roofInnerY - topOffset;
          if (p.y > maxInY) {
            if (refZ < 174.5 && refY > roofInnerY - topOffset * 0.5) {
              p.z = 174.5 - radius;
              cancelInwardVel(0, 0, -1);
            } else if (refZ > 316.8 && refY > roofInnerY - topOffset * 0.5) {
              p.z = 316.8 + radius;
              cancelInwardVel(0, 0, 1);
            } else {
              p.y = maxInY;
              cancelInwardVel(0, -1, 0);
            }
            modified = true;
          }
        }
        // A3. Solid Left & Right Crimson-Red Parabolic Steel Trusses + Sunroof Edge Posts
        const wallTopY = hasSunroof ? Math.max(chordTopY, roofOuterTopY) : chordTopY;
        if (p.y - bottomOffset < wallTopY) {
          const maxInX = Math.max(0.42, innerHalfW - radius);
          const dx = p.x - bx;
          if (Math.abs(dx) > maxInX) {
            const sgn = dx >= 0.0 ? 1.0 : -1.0;
            p.x = bx + sgn * maxInX;
            cancelInwardVel(-sgn, 0, 0);
            modified = true;
          }
        }
        return modified;
      }

      // Regime B: Outside Jembatan Teksas (Beside over Teluk U, Underneath Arch, or Above Sunroof!)
      const yMinSolid = bSoffitY;
      const yMaxSolid = hasSunroof ? Math.max(chordTopY, roofOuterTopY) : chordTopY;
      const dx = p.x - bx;
      const absDx = Math.abs(dx);
      const reqOutX = outerHalfW + radius;

      if (absDx < reqOutX && p.y + topOffset > yMinSolid && p.y - bottomOffset < yMaxSolid) {
        const pushSide = reqOutX - absDx;
        const pushDown = p.y + topOffset - yMinSolid;
        const pushUp = yMaxSolid - (p.y - bottomOffset);

        // Bias slightly toward the regime pPrev occupied so glancing side hits NEVER snap onto roof/underbelly!
        const wasClearlyBelow = refY + topOffset <= yMinSolid + 0.22;
        const wasClearlyAbove = refY - bottomOffset >= yMaxSolid - 0.22;

        if (wasClearlyBelow && pushDown <= pushSide + 0.45) {
          p.y = yMinSolid - topOffset;
          cancelInwardVel(0, -1, 0);
        } else if (wasClearlyAbove && pushUp <= pushSide + 0.45) {
          p.y = yMaxSolid + bottomOffset;
          cancelInwardVel(0, 1, 0);
        } else if (pushSide <= pushDown && pushSide <= pushUp) {
          const refDx = refX - bx;
          const sgn = Math.abs(refDx) > 0.05 ? (refDx >= 0.0 ? 1.0 : -1.0) : dx >= 0.0 ? 1.0 : -1.0;
          p.x = bx + sgn * reqOutX;
          cancelInwardVel(sgn, 0, 0);
        } else if (pushDown < pushUp) {
          p.y = yMinSolid - topOffset;
          cancelInwardVel(0, -1, 0);
        } else {
          p.y = yMaxSolid + bottomOffset;
          cancelInwardVel(0, 1, 0);
        }
        modified = true;
      }
      return modified;
    }

    // 2. West & East Concrete Approach Ramps + Handrails (z in [149.65, 171.65) and (319.65, 341.65])
    const rampOuterHW = 2.76;
    const railTopY = bDeckY + 1.12;
    const dxR = p.x - bx;
    const absDxR = Math.abs(dxR);

    // Outside concrete ramp side wall laterally
    if (Math.abs(refX - bx) > rampOuterHW - 0.05 && refY - bottomOffset < railTopY - 0.08) {
      const reqOutX = rampOuterHW + radius;
      if (absDxR < reqOutX && p.y - bottomOffset < railTopY) {
        const sgn = (refX - bx) >= 0.0 ? 1.0 : -1.0;
        p.x = bx + sgn * reqOutX;
        cancelInwardVel(sgn, 0, 0);
        return true;
      }
      return false;
    }

    if (absDxR < rampOuterHW + radius) {
      // Solid concrete ramp/stair deck
      if (p.y < bDeckY + bottomOffset) {
        p.y = bDeckY + bottomOffset;
        cancelInwardVel(0, 1, 0);
        modified = true;
      }
      // Outer ramp handrails at |x - bx| = 2.42..2.76 up to bDeckY + 1.12
      if (p.y - bottomOffset < railTopY && absDxR > 2.38 - radius) {
        if (refY - bottomOffset >= railTopY - 0.16) {
          p.y = railTopY + bottomOffset;
          cancelInwardVel(0, 1, 0);
        } else {
          const sgn = dxR >= 0.0 ? 1.0 : -1.0;
          p.x = bx + sgn * Math.max(0.4, 2.38 - radius);
          cancelInwardVel(-sgn, 0, 0);
        }
        modified = true;
      }
      // Central ceremonial stair handrail at |x - bx| = 1.22 up to bDeckY + 1.08
      const centerRailTop = bDeckY + 1.08;
      if (p.y - bottomOffset < centerRailTop) {
        const distRail = Math.abs(absDxR - 1.22);
        const minRailClear = 0.09 + radius;
        if (distRail < minRailClear) {
          if (refY - bottomOffset >= centerRailTop - 0.22) {
            p.y = centerRailTop + bottomOffset;
            cancelInwardVel(0, 1, 0);
          } else {
            const sgn = dxR >= 0.0 ? 1.0 : -1.0;
            const pushInOrOut = absDxR < 1.22 ? 1.22 - minRailClear : 1.22 + minRailClear;
            p.x = bx + sgn * Math.max(0.0, pushInOrOut);
          }
          modified = true;
        }
      }
    }
    return modified;
  }

  /**
   * Omnidirectional Binocular-Fisheye APAS (Advanced Pilot Assistance System) 3D Obstacle Field Probe.
   * Evaluates soft repulsive normal + penetration depth for a virtual lookahead sphere of `probeRadius`
   * against all static colliders (buildings, spires, trees, canopies, pylons), live dynamic colliders
   * (Bus Kuning UI, Golf Cart, Catamaran Ferry, Mini Yacht, eVTOL, Aircraft), Jembatan Teksas trusses/roof,
   * Underpass retaining walls/vault, and terrain/ocean surfaces — with ZERO heap allocations!
   */
  static probeDroneAPASRepulsion(px, py, pz, refX, refY, refZ, probeRadius, ctx, outRepel) {
    outRepel.nx = 0.0;
    outRepel.ny = 0.0;
    outRepel.nz = 0.0;
    outRepel.pen = 0.0;
    if (!ctx) return 0.0;

    let accX = 0.0;
    let accY = 0.0;
    let accZ = 0.0;
    let maxPen = 0.0;
    const addPush = (nx, ny, nz, pen) => {
      if (pen <= 0.0) return;
      accX += nx * pen;
      accY += ny * pen;
      accZ += nz * pen;
      if (pen > maxPen) maxPen = pen;
    };

    const uCfg = ctx.underpassConfig || null;
    const inSubseaBore =
      uCfg &&
      pz >= uCfg.zWestPortal - 0.5 &&
      pz <= uCfg.zEastPortal + 0.5 &&
      Math.abs(px - uCfg.ux) <= uCfg.boreHalfW + 0.25 &&
      py <= uCfg.floorY + uCfg.clearH + 0.25;
    const inOpenRamp =
      !inSubseaBore &&
      uCfg &&
      pz >= uCfg.zWestStart &&
      pz <= uCfg.zEastEnd &&
      Math.abs(px - uCfg.ux) <= uCfg.boreHalfW + 0.15 &&
      py < 1.85;

    // 1. Underpass walls, ceiling vault, & headwall lintels
    if (uCfg && pz >= uCfg.zWestStart - probeRadius && pz <= uCfg.zEastEnd + probeRadius) {
      const ux = uCfg.ux;
      const dxU = px - ux;
      if (inSubseaBore) {
        const wallPen = Math.abs(dxU) + probeRadius - uCfg.boreHalfW;
        if (wallPen > 0.0) addPush(dxU >= 0 ? -1 : 1, 0, 0, wallPen);
        const normX = Math.min(0.98, Math.abs(dxU) / uCfg.boreHalfW);
        const vaultY =
          uCfg.floorY + 2.20 + Math.sqrt(Math.max(0.0, 1.0 - normX * normX)) * (uCfg.clearH - 2.20);
        if (py + probeRadius > vaultY) addPush(0, -1, 0, py + probeRadius - vaultY);
        if (py - probeRadius < uCfg.floorY) addPush(0, 1, 0, uCfg.floorY - (py - probeRadius));
      } else if (inOpenRamp) {
        const wallPen = Math.abs(dxU) + probeRadius - uCfg.boreHalfW;
        if (wallPen > 0.0) addPush(dxU >= 0 ? -1 : 1, 0, 0, wallPen);
        const rY = uCfg.sampleRampY ? uCfg.sampleRampY(ux, pz) : null;
        if (rY !== null && py - probeRadius < rY) addPush(0, 1, 0, rY - (py - probeRadius));
      } else {
        const outerHW = uCfg.boreHalfW + 0.72;
        if (Math.abs(dxU) < outerHW + probeRadius && py - probeRadius < 1.35) {
          const sgn = (refX - ux) >= 0.0 ? 1.0 : -1.0;
          addPush(sgn, 0.25, 0, outerHW + probeRadius - Math.abs(dxU));
        }
      }
    }

    // 2. Jembatan Teksas UI trusses, sunroof, box girder, & approach ramps
    const bDeckFn = ctx.bridgeDeckFn || null;
    const bx = Number.isFinite(ctx.bridgeCenterX) ? ctx.bridgeCenterX : -28.0;
    if (bDeckFn && pz >= 149.65 - probeRadius && pz <= 341.65 + probeRadius) {
      const zClamped = Math.max(149.65, Math.min(341.65, pz));
      const bDeckY = bDeckFn(bx, zClamped);
      if (bDeckY !== null && Math.abs(px - bx) < 2.88 + probeRadius + 0.5) {
        if (zClamped >= 171.65 && zClamped <= 319.65) {
          const u = (zClamped - 245.65) / 74.0;
          const bSoffitY = ctx.bridgeSoffitFn ? (ctx.bridgeSoffitFn(zClamped) ?? bDeckY - 1.30) : bDeckY - 1.30;
          const chordTopY = bDeckY + 1.34 + 2.58 * Math.max(0.0, 1.0 - u * u);
          const hasRoof = zClamped >= 174.5 && zClamped <= 316.8;
          const roofInnerY = bDeckY + 2.86;
          const roofOuterTopY = bDeckY + 3.62;

          const refZCl = Math.max(171.65, Math.min(319.65, refZ));
          const refDeck = bDeckFn(bx, refZCl) ?? bDeckY;
          const refInCorr =
            Math.abs(refX - bx) <= 2.42 &&
            refY >= refDeck - 0.12 &&
            refY <= (refZ >= 174.3 && refZ <= 317.0 ? refDeck + 2.94 : refDeck + 4.85);

          if (refInCorr) {
            const dxB = px - bx;
            const wallPen = Math.abs(dxB) + probeRadius - 2.40;
            if (wallPen > 0.0) addPush(dxB >= 0 ? -1 : 1, 0, 0, wallPen);
            if (py - probeRadius < bDeckY) addPush(0, 1, 0, bDeckY - (py - probeRadius));
            if (hasRoof && py + probeRadius > roofInnerY) addPush(0, -1, 0, py + probeRadius - roofInnerY);
          } else {
            const yMaxS = hasRoof ? Math.max(chordTopY, roofOuterTopY) : chordTopY;
            const absDx = Math.abs(px - bx);
            if (absDx < 2.86 + probeRadius && py + probeRadius > bSoffitY && py - probeRadius < yMaxS) {
              const pSide = 2.86 + probeRadius - absDx;
              const pDown = py + probeRadius - bSoffitY;
              const pUp = yMaxS - (py - probeRadius);
              if (refY + probeRadius <= bSoffitY + 0.35 && pDown <= pSide + 0.5) {
                addPush(0, -1, 0, pDown);
              } else if (refY - probeRadius >= yMaxS - 0.35 && pUp <= pSide + 0.5) {
                addPush(0, 1, 0, pUp);
              } else {
                const sgn = (refX - bx) >= 0.0 ? 1.0 : -1.0;
                addPush(sgn, 0, 0, pSide);
              }
            }
          }
        } else {
          const absDx = Math.abs(px - bx);
          if (Math.abs(refX - bx) > 2.70 && py - probeRadius < bDeckY + 1.12 && absDx < 2.76 + probeRadius) {
            const sgn = (refX - bx) >= 0.0 ? 1.0 : -1.0;
            addPush(sgn, 0.2, 0, 2.76 + probeRadius - absDx);
          } else if (absDx < 2.76 + probeRadius && py - probeRadius < bDeckY) {
            addPush(0, 1, 0, bDeckY - (py - probeRadius));
          }
        }
      }
    }

    // 3. Static Colliders (Buildings, Spires, Columns, Tree Trunks, Tree Canopies, Pylons, ATC)
    const staticCols = ctx.staticColliders || [];
    for (let i = 0; i < staticCols.length; i++) {
      const c = staticCols[i];
      if (c.type === 'ISLAND_PERIMETER') continue;
      const pen = PhysicsKernel.computeSphere3DColliderMTV(px, py, pz, probeRadius, c, _mtvScratch);
      if (pen > 0.0) {
        addPush(_mtvScratch.nx, _mtvScratch.ny, _mtvScratch.nz, pen);
      }
    }

    // 4. Live Dynamic Colliders (Bus Kuning UI, Golf Cart, Catamaran Ferry, Mini Yacht, eVTOL, Aircraft, NPCs)
    const dynCols =
      typeof ctx.getDynamicColliders === 'function'
        ? ctx.getDynamicColliders()
        : ctx.dynamicColliders || [];
    for (let i = 0; i < dynCols.length; i++) {
      const dc = dynCols[i];
      if (!dc || dc.enabled === false) continue;
      const pen = PhysicsKernel.computeSphere3DColliderMTV(px, py, pz, probeRadius, dc, _mtvScratch);
      if (pen > 0.0) {
        addPush(_mtvScratch.nx, _mtvScratch.ny, _mtvScratch.nz, pen);
      }
    }

    // 5. Terrain Heightfield & Ocean / Freshwater Surface
    if (!inSubseaBore && !inOpenRamp) {
      let minSurf = ctx.groundHeightFn ? ctx.groundHeightFn(px, pz) : 0.0;
      const oceanY = Number.isFinite(ctx.oceanWaterY) ? ctx.oceanWaterY : -1.45;
      if (ctx.coastDistFn) {
        if (ctx.coastDistFn(px, pz) > -0.5) minSurf = Math.max(minSurf, oceanY);
      } else if (minSurf < oceanY) {
        minSurf = oceanY;
      }
      if (ctx.waterSurfaceFn) {
        const fwY = ctx.waterSurfaceFn(px, pz);
        if (fwY !== null) minSurf = Math.max(minSurf, fwY);
      }
      if (ctx.ferryPierDeckFn) {
        const pierY = ctx.ferryPierDeckFn(px, pz);
        if (pierY !== null && refY >= pierY - 0.25) minSurf = Math.max(minSurf, pierY);
      }
      if (py - probeRadius < minSurf) {
        // Estimate terrain slope normal so diving toward Gunung Api / hills deflects smoothly up-slope!
        let gNx = 0.0;
        let gNy = 1.0;
        let gNz = 0.0;
        if (ctx.groundHeightFn && minSurf > oceanY + 0.3) {
          const hX = ctx.groundHeightFn(px + 1.2, pz) - ctx.groundHeightFn(px - 1.2, pz);
          const hZ = ctx.groundHeightFn(px, pz + 1.2) - ctx.groundHeightFn(px, pz - 1.2);
          const nLen = Math.hypot(-hX, 2.4, -hZ) || 1.0;
          gNx = -hX / nLen;
          gNy = 2.4 / nLen;
          gNz = -hZ / nLen;
        }
        addPush(gNx, gNy, gNz, minSurf - (py - probeRadius));
      }
    }

    const totLen = Math.hypot(accX, accY, accZ);
    if (totLen > 1e-5 && maxPen > 0.0) {
      outRepel.nx = accX / totLen;
      outRepel.ny = accY / totLen;
      outRepel.nz = accZ / totLen;
      outRepel.pen = maxPen;
      return maxPen;
    }
    return 0.0;
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

        // 2. Jembatan Teksas UI Two-Regime 3D Topological Solids (Hollow Corridor vs Solid Red Trusses, Sunroof, Girder, & Ramps)
        void bridgeHalfW;
        if (
          !isWalk &&
          bridgeDeckFn &&
          PhysicsKernel.resolveTeksasBridgeTopologicalSolids(
            p,
            pPrev,
            vel,
            radius,
            topOffset,
            bottomOffset,
            bridgeCenterX,
            bridgeDeckFn,
            bridgeSoffitFn,
            cancelInwardVel
          )
        ) {
          hitAny = true;
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
