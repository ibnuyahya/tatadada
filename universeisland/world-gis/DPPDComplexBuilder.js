import * as THREE from 'three';
import { TerrainDEMLoader } from './TerrainDEMLoader.js';

/**
 * NEXUS Omni-Scale Procedural Engine v6.46 — Kepulauan UI Edition (1:1 Komplekita Transplant)
 * ./world-gis/DPPDComplexBuilder.js
 *
 * Exact 1:1 Architectural & Site Transplant of Kompleks Bundaran DPPD – ITTC – STP-UI – ILRC/DRPM
 * from `komplekita` (206-Photo Ground-Truth Calibration) onto Pulau U:
 * 1. Gedung DPPD UI (eks-KSDP, 2 Lantai — L1 buried behind 2-tier sloped green grass berm & batu-kali toe wall,
 *    L2 under low-slung Nusantara terracotta roof eaveY=6.45m ridgeY=13.55m, NW salmon-brick triangular gable end wall,
 *    clean cream-white front gable triangle w/ apex lamp, L1 terracotta porch canopy w/ GEDUNG DPPD UI & GEDUNG KSDP UI
 *    plaques, dual 3D gold-yellow space-frame ladder trusses + hanging UI DEPOK cloth banner, UI-Huawei brass totem,
 *    blue ADA stall + bare steel pole, campus map directory board, & 3 royal palms)
 * 2. ITTC Perpustakaan Lama UI (Central 42m x 42m 4-Story Square Courtyard Heritage Building, eaveY=14.20m ridgeY=18.80m,
 *    Atap Coran central concrete deck + raised pyramidal terracotta skylight monitor, West portico + NW 1-story annex)
 * 3. Gedung STP-UI (Science Techno Park 8 Lantai, 116° wide curved sweep -150°..-34°, crownTopY=26.80m,
 *    24-bay copper-bronze L2-L3 chevron & louver podium, white L5-L8 sunshade ledges, white curved courtyard wall
 *    w/ 7 dark ribbon window bands, cream stair cores, copper side blades, & 18 radial blue rooftop solar PV panels)
 * 4. Gedung ILRC & DRPM UI (Integrated Laboratory & Research Center, 114° wide curved sweep -31°..+83°, corniceTopY=21.20m,
 *    6-story solid burnt-terracotta red-brick South Tower +75°..+83° w/ 3 stacked dark-hooded window boxes & 5 white
 *    cantilevered sunshade fins, smooth cream-white inner courtyard wall, & 5-story open atrium portal)
 * 5. Bundaran Site & Tropical Flora (Stone-lined perimeter drainage moat + 5 entry bridges, West Main ITTC Driveway
 *    w/ splitter island, 30 multi-lobe tropical broadleaf trees + 6 parabolic royal palms, & Pulau U Ring Road integration)
 *
 * Batched directly into SplineCorridorBuilder's { horizontalGeo, verticalGeo } pair (+0 extra Draw Calls!).
 */

const DEG = Math.PI / 180;

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./world-gis/MasterPlanConfig.js
 *
 * Single Source of Spatial Truth (MasterPlanSSoST) for Kompleks Bundaran Gedung DPPD,
 * ITTC Perpustakaan Lama, STP-UI, & ILRC/DRPM Universitas Indonesia, Depok.
 *
 * Geodetically calibrated against:
 * - Satellite Nadir 90° (ref_174.jpg, ref_206.jpg, ref_207.jpg)
 * - Street View GPS Stations around Jl. Prof. Dr. Nugroho Notosusanto (151.0m road centerline diameter)
 *
 * World Coordinate Convention (1.0 unit = 1.0 meter):
 *   +X = East (0°), +Z = South (+90°), -X = West (±180°), -Z = North (-90°), +Y = Up
 */

const UI_DPPD_MASTER_PLAN = {
  id: 'UI_DEPOK_BUNDARAN_DPPD_STP_ILRC_V2',
  name: 'Kompleks Bundaran Gedung DPPD (eks-KSDP), ITTC, STP-UI & ILRC Universitas Indonesia',
  gpsOrigin: {
    lat: -6.365429,
    lon: 106.826801,
    elevationMdpl: 73.0
  },
  envelope: {
    width: 176.0,
    length: 176.0,
    height: 27.2,
    foundationSkirtDepth: 18.0
  },
  ringRoad: {
    rDitchInner: 68.55,
    rDitchOuter: 69.45,
    ditchFloorY: -0.68,
    rInnerVerge: 71.15,
    rInnerCurb: 71.50,
    rAsphaltIn: 71.50,
    rCenterDash: 75.50,
    rAsphaltOut: 79.50,
    rOuterCurb: 79.85,
    rPlinthRim: 88.00,
    trafficFlow: 'counter-clockwise'
  },
  sectors: {
    // 1. ITTC Perpustakaan Lama — Cardinal N-S / E-W Square at Center
    ittc: {
      centerX: -3.0,
      centerZ: 1.5,
      outerHalf: 21.0,      // 42.0m x 42.0m footprint
      courtyardHalf: 8.0,   // 16.0m x 16.0m courtyard
      skylightHalf: 4.8,    // 9.6m x 9.6m nested skylight roof cap (ref_174)
      floors: 4,
      eaveY: 14.20,
      ridgeY: 18.80,
      annexNW: { x: -22.5, z: -21.0, w: 13.0, d: 7.5, h: 4.2 }
    },

    // 2. Gedung DPPD (eks-KSDP) — SW Quadrant (+92° South to +178° West, Main Axis +135.5° SW)
    dppd: {
      thStart: 92.0 * DEG,
      thEnd: 178.0 * DEG,
      thAxis: 135.5 * DEG,
      rInner: 39.5,
      rOuter: 57.5,
      rLoggiaWall: 54.8,
      rBermToe: 66.8,
      l1TopY: 3.72,
      eaveY: 6.45,
      wingRidgeY: 13.55,
      gableHalfWidth: 11.2,
      gablePeakY: 13.55
    },

    // 3. Gedung STP-UI (Science Techno Park) — North Arc (-150° WNW across -90° North to -34° ENE)
    stp: {
      thStart: -150.0 * DEG,
      thEnd: -34.0 * DEG,
      thPortico: -88.0 * DEG,
      rInner: 39.0,
      rOuter: 57.8,
      floors: 8,
      l1TopY: 4.30,
      l3TopY: 11.10,
      roofSlabY: 25.40,
      crownTopY: 26.80
    },

    // 4. Gedung ILRC & DRPM UI — East-to-SSE Arc (-31° ENE across 0° East & +41° Atrium to +83° SSE Brick Tower)
    ilrc: {
      thStart: -31.0 * DEG,
      thEastEnd: 36.0 * DEG,
      thAtriumStart: 36.0 * DEG,
      thAtriumEnd: 46.0 * DEG,
      thSouthEnd: 75.0 * DEG,
      thTowerEnd: 83.0 * DEG,
      rInnerEast: 37.5,
      rInnerSouth: 41.5,
      rOuter: 58.0,
      corniceTopY: 21.20,
      towerTopY: 25.20,
      penthouseTopY: 27.20
    }
  }
};

/**
 * Helper to create polar camera preset at radius rCam, angleDeg, height yCam looking at (rTgt, tgtDeg, yTgt).
 */
function polarPreset(label, sector, rCam, camDeg, yCam, rTgt, tgtDeg, yTgt, fov = 56) {
  const cRad = camDeg * DEG;
  const tRad = tgtDeg * DEG;
  return {
    label,
    sector,
    fov,
    pos: [+(rCam * Math.cos(cRad)).toFixed(2), yCam, +(rCam * Math.sin(cRad)).toFixed(2)],
    target: [+(rTgt * Math.cos(tRad)).toFixed(2), yTgt, +(rTgt * Math.sin(tRad)).toFixed(2)]
  };
}

const BUNDARAN_CAMERA_PRESETS = {
  // Mandatory Tri-Angle Validation Presets (Gate 4)
  elevasi_horisontal: polarPreset('Elevasi 0° — Fasad Utama DPPD', 'tri', 82.0, 135.5, 3.6, 54.0, 135.5, 6.8, 54),
  oblique_45: {
    label: 'Oblique 45° — Ikhtisar Bundaran',
    sector: 'tri',
    fov: 48,
    pos: [-112.0, 96.0, 112.0],
    target: [0.0, 6.0, 0.0]
  },
  nadir_zenith_90: {
    label: 'Nadir 90° — Satelit Orthografik',
    sector: 'tri',
    isNadir: true,
    altitude: 198.0,
    pos: [0.0, 198.0, 0.45],
    target: [0.0, 0.0, 0.0]
  },

  // Sector Overviews
  overview_dppd: polarPreset('Gedung DPPD (Barat Daya)', 'dppd', 84.0, 135.5, 6.5, 52.0, 135.5, 7.2, 54),
  overview_ittc: {
    label: 'ITTC Perpustakaan Lama (Tengah)',
    sector: 'ittc',
    fov: 52,
    pos: [-66.0, 26.0, 8.0],
    target: [-3.0, 8.5, 1.5]
  },
  overview_stp: polarPreset('Gedung STP-UI (Utara)', 'stp', 88.0, -86.0, 11.0, 52.0, -88.0, 13.5, 58),
  overview_ilrc: polarPreset('Gedung ILRC & Atrium (Tenggara)', 'ilrc', 86.0, 41.0, 9.0, 52.0, 41.0, 10.5, 56),

  // Calibrated Street View Node Stations (Clockwise / Counter-Clockwise Ring Road Tour)
  sv_dppd_front: polarPreset('SV-01 • Depan Pelana Utama DPPD (ref_084)', 'dppd', 76.5, 135.5, 2.3, 55.0, 135.5, 7.2, 62),
  sv_dppd_totem: polarPreset('SV-02 • Kanopi L1, Ramp & Totem Huawei (ref_184)', 'dppd', 73.5, 142.0, 2.2, 57.0, 136.5, 4.5, 58),
  sv_dppd_se_wing: polarPreset('SV-03 • Sayap Kanan SE & 3 Palem Raja (ref_078)', 'dppd', 76.5, 112.0, 2.3, 55.0, 118.0, 6.0, 60),
  sv_dppd_nw_gable: polarPreset('SV-04 • Dinding Pelana Bata NW & Akses Barat (ref_144)', 'dppd', 76.5, 174.0, 2.4, 54.0, 168.0, 6.2, 58),
  sv_west_ittc: {
    label: 'SV-05 • Pintu Masuk Barat → ITTC Perpustakaan Lama (ref_203)',
    sector: 'ittc',
    fov: 62,
    pos: [-76.8, 2.3, -2.8],
    target: [-24.0, 6.8, -0.5]
  },
  sv_stp_sw_core: polarPreset('SV-06 • Sudut SW Tembaga & Tangga STP-UI (ref_148)', 'stp', 77.5, -146.0, 2.6, 54.0, -135.0, 13.0, 64),
  sv_stp_portico: polarPreset('SV-07 • Portico Utara & Planter STP-UI (ref_164)', 'stp', 78.5, -90.5, 2.2, 56.0, -89.0, 7.8, 68),
  sv_stp_ne_makara: polarPreset('SV-08 • Puncak Makara Emas & Sirip NE STP-UI (ref_111)', 'stp', 77.5, -40.0, 2.6, 54.0, -52.0, 15.0, 68),
  sv_ilrc_east_box: polarPreset('SV-09 • Sayap Louver Timur & Kotak Kaca ILRC (ref_009)', 'ilrc', 77.0, 6.0, 2.4, 55.0, 8.0, 9.8, 60),
  sv_ilrc_drpm_gate: polarPreset('SV-10 • Gerbang Parkir Pimpinan & DRPM (ref_012)', 'ilrc', 77.0, 28.0, 2.3, 55.0, 30.0, 7.5, 60),
  sv_ilrc_atrium: polarPreset('SV-11 • Atrium Terbuka 5 Lantai & Portal ILRC (ref_029)', 'ilrc', 76.5, 41.0, 2.4, 52.0, 41.0, 10.5, 62),
  sv_ilrc_mound: polarPreset('SV-12 • Sayap Selatan & Gundukan Bambu ILRC (ref_033)', 'ilrc', 76.5, 60.0, 2.3, 55.0, 58.0, 8.5, 60),
  sv_ilrc_brick_tower: polarPreset('SV-13 • Menara Bata 6 Lantai & 3 Jendela Tudung (ref_076)', 'ilrc', 76.8, 86.0, 2.4, 54.0, 79.0, 12.5, 60)
};

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./core/MasterSurfaceKernels.js
 *
 * Layer 1 Mathematical SSoST Kernels:
 * - Kernel 1: Parametric Surface Slicing & Polar-Geodetic Annular Lofting (0.00m seam, outward winding guarantee)
 * - Kernel 2: Analytical Roof Valley / Courtyard Hip-Limasan Slicing & CSG Smooth SDF Operators
 * - Surface Applique Normal-Offset Helper (Zero Z-Fighting on flat & curved facades)
 */
class NexusSSoSTKernel {
  /**
   * Smooth CSG union & subtraction primitives (Kernel 2 SDF helpers)
   */
  static opSmoothUnion(d1, d2, k = 0.15) {
    const kk = Math.max(1e-5, k);
    const h = THREE.MathUtils.clamp(0.5 + (0.5 * (d2 - d1)) / kk, 0.0, 1.0);
    return THREE.MathUtils.lerp(d2, d1, h) - kk * h * (1.0 - h);
  }

  static opSmoothSubtraction(d1, d2, k = 0.15) {
    const kk = Math.max(1e-5, k);
    const h = THREE.MathUtils.clamp(0.5 - (0.5 * (d2 + d1)) / kk, 0.0, 1.0);
    return THREE.MathUtils.lerp(d2, -d1, h) + kk * h * (1.0 - h);
  }

  /**
   * Sanitizes a Float32Array in-place so zero NaN/Infinity values can ever reach WebGL buffers.
   */
  static sanitizeFloatArray(arr, fallback = 0.0) {
    for (let i = 0, len = arr.length; i < len; i++) {
      if (!Number.isFinite(arr[i])) arr[i] = fallback;
    }
    return arr;
  }

  /**
   * Slices a continuous subdomain [uMin..uMax] x [vMin..vMax] from an analytical Master Surface S(u,v)->(p,n).
   * Guarantees outward counter-clockwise triangle winding matching analytical normal n(u,v) and finite buffers.
   */
  static sliceSubdomainFromMaster(
    evaluateFn,
    uMin = 0.0,
    uMax = 1.0,
    vMin = 0.0,
    vMax = 1.0,
    uSegs = 24,
    vSegs = 8,
    uvScaleU = 1.0,
    uvScaleV = 1.0
  ) {
    const us = Math.max(1, Math.floor(uSegs));
    const vs = Math.max(1, Math.floor(vSegs));
    const vertCount = (us + 1) * (vs + 1);
    const positions = new Float32Array(vertCount * 3);
    const normals = new Float32Array(vertCount * 3);
    const uvs = new Float32Array(vertCount * 2);
    const indices = [];

    const p = new THREE.Vector3();
    const n = new THREE.Vector3();
    let pPtr = 0;
    let uvPtr = 0;

    for (let iv = 0; iv <= vs; iv++) {
      const tv = iv / vs;
      const v = vMin + (vMax - vMin) * tv;
      for (let iu = 0; iu <= us; iu++) {
        const tu = iu / us;
        const u = uMin + (uMax - uMin) * tu;
        evaluateFn(u, v, p, n);

        const px = Number.isFinite(p.x) ? p.x : 0.0;
        const py = Number.isFinite(p.y) ? p.y : 0.0;
        const pz = Number.isFinite(p.z) ? p.z : 0.0;

        let nx = Number.isFinite(n.x) ? n.x : 0.0;
        let ny = Number.isFinite(n.y) ? n.y : 1.0;
        let nz = Number.isFinite(n.z) ? n.z : 0.0;
        const nLen = Math.hypot(nx, ny, nz);
        if (nLen > 1e-6) {
          nx /= nLen;
          ny /= nLen;
          nz /= nLen;
        } else {
          nx = 0.0;
          ny = 1.0;
          nz = 0.0;
        }

        positions[pPtr] = px;
        positions[pPtr + 1] = py;
        positions[pPtr + 2] = pz;

        normals[pPtr] = nx;
        normals[pPtr + 1] = ny;
        normals[pPtr + 2] = nz;

        uvs[uvPtr] = u * uvScaleU;
        uvs[uvPtr + 1] = v * uvScaleV;

        pPtr += 3;
        uvPtr += 2;
      }
    }

    const stride = us + 1;
    const vA = new THREE.Vector3();
    const vB = new THREE.Vector3();
    const vC = new THREE.Vector3();
    const e1 = new THREE.Vector3();
    const e2 = new THREE.Vector3();
    const faceNorm = new THREE.Vector3();
    const refNorm = new THREE.Vector3();

    for (let iv = 0; iv < vs; iv++) {
      for (let iu = 0; iu < us; iu++) {
        const a = iv * stride + iu;
        const b = a + 1;
        const c = (iv + 1) * stride + iu;
        const d = c + 1;

        vA.fromArray(positions, a * 3);
        vB.fromArray(positions, b * 3);
        vC.fromArray(positions, c * 3);
        refNorm.fromArray(normals, a * 3);

        e1.subVectors(vB, vA);
        e2.subVectors(vC, vA);
        faceNorm.crossVectors(e1, e2);

        if (faceNorm.dot(refNorm) >= 0.0) {
          indices.push(a, b, c, b, d, c);
        } else {
          indices.push(a, c, b, b, c, d);
        }
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    return geo;
  }

  /**
   * Builds a watertight polar-cylindrical sector band [th0..th1] x [rIn..rOut] x [yBot..yTop]
   * with metric world-space UVs so brick, plaster, louvers, and concrete bands never stretch.
   */
  static addCurvedPolarBand(parent, mat, th0, th1, rIn, rOut, yBot, yTop, segs = 32, opts = {}) {
    const {
      top = true,
      bottom = yBot > 0.2,
      outer = true,
      inner = true,
      startCap = true,
      endCap = true,
      uvMetersPerTile = 4.0,
      castShadow = true,
      receiveShadow = true
    } = opts;

    const spanRad = th1 - th0;
    const midR = 0.5 * (rIn + rOut);
    const arcMeters = Math.max(0.1, Math.abs(spanRad) * midR);
    const heightMeters = Math.max(0.05, Math.abs(yTop - yBot));
    const radialMeters = Math.max(0.05, Math.abs(rOut - rIn));
    const uvU = arcMeters / uvMetersPerTile;
    const uvH = heightMeters / uvMetersPerTile;
    const uvR = radialMeters / uvMetersPerTile;

    const addFace = (geo) => {
      const m = new THREE.Mesh(geo, mat);
      m.castShadow = castShadow;
      m.receiveShadow = receiveShadow;
      parent.add(m);
      return m;
    };

    if (outer) {
      addFace(
        NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const th = th0 + u * spanRad;
            const c = Math.cos(th);
            const s = Math.sin(th);
            p.set(rOut * c, yBot + v * (yTop - yBot), rOut * s);
            n.set(c, 0, s);
          },
          0, 1, 0, 1, segs, 1, uvU, uvH
        )
      );
    }

    if (inner) {
      addFace(
        NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const th = th0 + u * spanRad;
            const c = Math.cos(th);
            const s = Math.sin(th);
            p.set(rIn * c, yBot + v * (yTop - yBot), rIn * s);
            n.set(-c, 0, -s);
          },
          0, 1, 0, 1, segs, 1, uvU, uvH
        )
      );
    }

    if (top) {
      addFace(
        NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const th = th0 + u * spanRad;
            const r = rIn + v * (rOut - rIn);
            p.set(r * Math.cos(th), yTop, r * Math.sin(th));
            n.set(0, 1, 0);
          },
          0, 1, 0, 1, segs, 1, uvU, uvR
        )
      );
    }

    if (bottom) {
      addFace(
        NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const th = th0 + u * spanRad;
            const r = rIn + v * (rOut - rIn);
            p.set(r * Math.cos(th), yBot, r * Math.sin(th));
            n.set(0, -1, 0);
          },
          0, 1, 0, 1, segs, 1, uvU, uvR
        )
      );
    }

    const signSpan = spanRad >= 0 ? 1 : -1;
    if (startCap) {
      const c0 = Math.cos(th0);
      const s0 = Math.sin(th0);
      addFace(
        NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const r = rIn + u * (rOut - rIn);
            const y = yBot + v * (yTop - yBot);
            p.set(r * c0, y, r * s0);
            n.set(s0 * signSpan, 0, -c0 * signSpan);
          },
          0, 1, 0, 1, 1, 1, uvR, uvH
        )
      );
    }

    if (endCap) {
      const c1 = Math.cos(th1);
      const s1 = Math.sin(th1);
      addFace(
        NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const r = rIn + u * (rOut - rIn);
            const y = yBot + v * (yTop - yBot);
            p.set(r * c1, y, r * s1);
            n.set(-s1 * signSpan, 0, c1 * signSpan);
          },
          0, 1, 0, 1, 1, 1, uvR, uvH
        )
      );
    }
  }

  /**
   * Builds a sloped polar-conical annular surface (for sloped grass berms, sloped stone retaining walls,
   * ditch side walls, and conical curved soffits) with exact analytical surface normals.
   */
  static createPolarSlopedRibbon(th0, th1, r0, y0, r1, y1, segs = 48, uvMetersPerTile = 3.5, flipNormal = false) {
    const spanRad = th1 - th0;
    const midR = 0.5 * (r0 + r1);
    const arcLen = Math.max(0.1, Math.abs(spanRad) * midR);
    const profLen = Math.max(0.05, Math.hypot(r1 - r0, y1 - y0));
    const dr = r1 - r0;
    const dy = y1 - y0;
    const signFlip = flipNormal ? -1.0 : 1.0;

    return NexusSSoSTKernel.sliceSubdomainFromMaster(
      (u, v, p, n) => {
        const th = th0 + u * spanRad;
        const r = r0 + v * dr;
        const y = y0 + v * dy;
        const c = Math.cos(th);
        const s = Math.sin(th);
        p.set(r * c, y, r * s);
        // Tangent along theta: (-r*s, 0, r*c); tangent along v: (dr*c, dy, dr*s)
        // Outward/upward normal proportional to (-dy*c, dr, -dy*s) or (+dy*c, -dr, +dy*s)
        let nx = -dy * c * signFlip;
        let ny = dr * signFlip;
        let nz = -dy * s * signFlip;
        if (ny < 0 && !flipNormal && Math.abs(dr) > 1e-4) {
          nx = -nx;
          ny = -ny;
          nz = -nz;
        }
        n.set(nx, ny, nz).normalize();
      },
      0,
      1,
      0,
      1,
      segs,
      2,
      arcLen / uvMetersPerTile,
      profLen / uvMetersPerTile
    );
  }

  /**
   * Creates a 3D cylindrical/hexagonal ceramic ridge-cap ("nok genteng keramik") tube along a polyline of 3D points.
   */
  static createRidgeCapStrip(points, radius = 0.14, radialSegs = 6) {
    if (!points || points.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.02);
    const tubularSegs = Math.max(2, (points.length - 1) * 4);
    const geo = new THREE.TubeGeometry(curve, tubularSegs, radius, radialSegs, false);
    return geo;
  }

  /**
   * Pushes a True-3D Thickened & Stepped-Course Terracotta Roof Quad (or Triangle when p2==p3)
   * into (pos, norm, uvs) buffers.
   * - Generates 3D overlapping horizontal tile courses with physical step lips (stepHeight = 0.028m)
   * - Generates 3D solid slab thickness along the eave edge, ridge edge, and side verges (slabThickness = 0.11m)
   * - Generates 3D corrugated ceramic eave drip-tile nosings ("profil gelombang genteng cucuran 3D") along the eave edge!
   */
  static pushThickSteppedRoofQuad(pos, norm, uvs, inP0, inP1, inP2, inP3, opts = {}) {
    const tileM = opts.uvTileMeters || 1.8;
    const courses = opts.courses || 9;
    const stepH = opts.stepHeight ?? 0.028;
    const slabThick = opts.slabThickness ?? 0.10;
    const addEaveNosings = opts.eaveNosings !== false;

    let p0 = inP0;
    let p1 = inP1;
    let p2 = inP2;
    let p3 = inP3;

    const v0 = new THREE.Vector3(...p0);
    const v1 = new THREE.Vector3(...p1);
    const v2 = new THREE.Vector3(...p2);
    const v3 = new THREE.Vector3(...p3);

    // Ensure upward normal (+Y) while keeping (p0, p1) at the lower eave and (p3, p2) at the upper ridge
    const e1 = new THREE.Vector3().subVectors(v1, v0);
    const e2 = new THREE.Vector3().subVectors(v3, v0);
    if (e2.lengthSq() < 1e-6) e2.subVectors(v2, v0);
    const fn = new THREE.Vector3().crossVectors(e1, e2).normalize();
    if (fn.y < 0) {
      p0 = inP1;
      p1 = inP0;
      p2 = inP3;
      p3 = inP2;
      fn.negate();
    }

    const downDir = new THREE.Vector3(
      0.5 * (p0[0] + p1[0] - p2[0] - p3[0]),
      0.5 * (p0[1] + p1[1] - p2[1] - p3[1]),
      0.5 * (p0[2] + p1[2] - p2[2] - p3[2])
    ).normalize();

    const emitTri = (a, b, c, nVec) => {
      for (const pt of [a, b, c]) {
        pos.push(pt[0], pt[1], pt[2]);
        norm.push(nVec.x, nVec.y, nVec.z);
        uvs.push(pt[0] / tileM, (pt[2] + pt[1]) / tileM);
      }
    };
    const emitQuad = (a, b, c, d, nVec) => {
      emitTri(a, b, c, nVec);
      emitTri(a, c, d, nVec);
    };

    const lerp3 = (a, b, t) => [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t
    ];
    const offset3 = (a, n, dist) => [
      a[0] + n.x * dist,
      a[1] + n.y * dist,
      a[2] + n.z * dist
    ];

    // 1. Stepped 3D overlapping tile courses from eave (t=0) to ridge (t=1)
    for (let k = 0; k < courses; k++) {
      const t0 = k / courses;
      const t1 = (k + 1) / courses;
      const lBotBase = lerp3(p0, p3, t0);
      const rBotBase = lerp3(p1, p2, t0);
      const lTopBase = lerp3(p0, p3, t1);
      const rTopBase = lerp3(p1, p2, t1);

      const lBotUp = offset3(lBotBase, fn, stepH);
      const rBotUp = offset3(rBotBase, fn, stepH);
      const lTopUp = offset3(lTopBase, fn, 0.005);
      const rTopUp = offset3(rTopBase, fn, 0.005);

      // Upper weathered ceramic tile slope of course k
      emitQuad(lBotUp, rBotUp, rTopUp, lTopUp, fn);

      // 3D vertical front overlap lip of course k (gives visible depth to every tile row!)
      const lBotLow = offset3(lBotBase, fn, k === 0 ? -slabThick : 0.003);
      const rBotLow = offset3(rBotBase, fn, k === 0 ? -slabThick : 0.003);
      emitQuad(lBotLow, rBotLow, rBotUp, lBotUp, downDir);
    }

    // 2. Solid 3D Slab Side Verges & Underside Back-Face Thickness
    const b0 = offset3(p0, fn, -slabThick);
    const b1 = offset3(p1, fn, -slabThick);
    const b2 = offset3(p2, fn, -slabThick);
    const b3 = offset3(p3, fn, -slabThick);
    const t0 = offset3(p0, fn, stepH);
    const t1 = offset3(p1, fn, stepH);
    const t2 = offset3(p2, fn, 0.005);
    const t3 = offset3(p3, fn, 0.005);

    const botNorm = fn.clone().negate();
    emitQuad(b0, b3, b2, b1, botNorm);

    const sideLNorm = new THREE.Vector3().crossVectors(fn, new THREE.Vector3().subVectors(new THREE.Vector3(...p3), new THREE.Vector3(...p0))).normalize();
    const sideRNorm = new THREE.Vector3().crossVectors(new THREE.Vector3().subVectors(new THREE.Vector3(...p2), new THREE.Vector3(...p1)), fn).normalize();
    emitQuad(b3, b0, t0, t3, sideLNorm);
    emitQuad(b1, b2, t2, t1, sideRNorm);

    // 3. 3D Corrugated Ceramic Eave Drip-Tile Nosings ("genteng list cucuran 3D") along the eave edge (p0 -> p1)
    const eaveLen = Math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
    if (addEaveNosings && eaveLen > 0.6) {
      const tilePitch = 0.34;
      const nTiles = Math.min(64, Math.max(2, Math.floor(eaveLen / tilePitch)));
      const eaveTang = new THREE.Vector3(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]).normalize();
      for (let i = 0; i < nTiles; i++) {
        const uA = (i + 0.14) / nTiles;
        const uB = (i + 0.86) / nTiles;
        const eA = lerp3(p0, p1, uA);
        const eB = lerp3(p0, p1, uB);
        const eBackA = lerp3(eA, lerp3(p3, p2, 0.5 * (uA + uB)), 0.055);
        const eBackB = lerp3(eB, lerp3(p3, p2, 0.5 * (uA + uB)), 0.055);

        const crownH = stepH + 0.034;
        const fA = offset3(eA, fn, crownH);
        const fB = offset3(eB, fn, crownH);
        const fBotA = offset3(eA, fn, -0.02);
        const fBotB = offset3(eB, fn, -0.02);
        const bkA = offset3(eBackA, fn, stepH * 0.6);
        const bkB = offset3(eBackB, fn, stepH * 0.6);

        // Front lip, top arch, and side cheeks of each 3D ceramic eave tile
        emitQuad(fBotA, fBotB, fB, fA, downDir);
        emitQuad(fA, fB, bkB, bkA, fn);
        emitTri(fBotA, fA, bkA, eaveTang.clone().negate());
        emitTri(fBotB, bkB, fB, eaveTang);
      }
    }
  }

  /**
   * Builds a watertight 4-wing Courtyard Hip/Limasan Roof (for ITTC Perpustakaan Lama 42m x 42m w/ 16m courtyard).
   * Outer eave square: [-halfOut, +halfOut], Inner courtyard eave square: [-halfIn, +halfIn],
   * Continuous square ridge loop at halfRidge = 0.5*(halfOut + halfIn) at y = ridgeY, eaves at y = eaveY.
   * Includes 3D stepped tile courses, 3D slab thickness, 3D corrugated eave tile nosings, and flat underside soffit.
   */
  static createSquareCourtyardLimasanRoof(halfOut, halfIn, eaveY, ridgeY, uvTileMeters = 2.0) {
    const halfRidge = 0.5 * (halfOut + halfIn);
    const pos = [];
    const norm = [];
    const uvs = [];

    const pushQuad = (p0, p1, p2, p3) => {
      NexusSSoSTKernel.pushThickSteppedRoofQuad(pos, norm, uvs, p0, p1, p2, p3, {
        uvTileMeters,
        courses: 10,
        stepHeight: 0.030,
        slabThickness: 0.12,
        eaveNosings: true
      });
    };

    // 4 cardinal directions: South (+Z), East (+X), North (-Z), West (-X)
    // Outer corners (CCW from top view):
    const oSW = [-halfOut, eaveY, halfOut];
    const oSE = [halfOut, eaveY, halfOut];
    const oNE = [halfOut, eaveY, -halfOut];
    const oNW = [-halfOut, eaveY, -halfOut];

    const rSW = [-halfRidge, ridgeY, halfRidge];
    const rSE = [halfRidge, ridgeY, halfRidge];
    const rNE = [halfRidge, ridgeY, -halfRidge];
    const rNW = [-halfRidge, ridgeY, -halfRidge];

    const iSW = [-halfIn, eaveY, halfIn];
    const iSE = [halfIn, eaveY, halfIn];
    const iNE = [halfIn, eaveY, -halfIn];
    const iNW = [-halfIn, eaveY, -halfIn];

    // 4 Outer trapezoidal hip slopes
    pushQuad(oSW, oSE, rSE, rSW); // South outer slope
    pushQuad(oSE, oNE, rNE, rSE); // East outer slope
    pushQuad(oNE, oNW, rNW, rNE); // North outer slope
    pushQuad(oNW, oSW, rSW, rNW); // West outer slope

    // 4 Inner trapezoidal courtyard valley slopes
    pushQuad(iSE, iSW, rSW, rSE); // South inner courtyard slope
    pushQuad(iNE, iSE, rSE, rNE); // East inner courtyard slope
    pushQuad(iNW, iNE, rNE, rNW); // North inner courtyard slope
    pushQuad(iSW, iNW, rNW, rSW); // West inner courtyard slope

    const roofGeo = new THREE.BufferGeometry();
    roofGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    roofGeo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
    roofGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    roofGeo.computeBoundingBox();
    roofGeo.computeBoundingSphere();

    // Flat underside eave soffit ring [halfIn..halfOut] at eaveY
    const sPos = [];
    const sNorm = [];
    const sUv = [];
    const addSoffitTrap = (a, b, c, d) => {
      const tris = [a, c, b, a, d, c];
      for (const pt of tris) {
        sPos.push(pt[0], eaveY - 0.12, pt[2]);
        sNorm.push(0, -1, 0);
        sUv.push(pt[0] / 3.0, pt[2] / 3.0);
      }
    };
    addSoffitTrap(oSW, oSE, iSE, iSW);
    addSoffitTrap(oSE, oNE, iNE, iSE);
    addSoffitTrap(oNE, oNW, iNW, iNE);
    addSoffitTrap(oNW, oSW, iSW, iNW);

    const soffitGeo = new THREE.BufferGeometry();
    soffitGeo.setAttribute('position', new THREE.Float32BufferAttribute(sPos, 3));
    soffitGeo.setAttribute('normal', new THREE.Float32BufferAttribute(sNorm, 3));
    soffitGeo.setAttribute('uv', new THREE.Float32BufferAttribute(sUv, 2));
    soffitGeo.computeBoundingBox();
    soffitGeo.computeBoundingSphere();

    const toV3 = (arr) => new THREE.Vector3(arr[0], arr[1] + 0.055, arr[2]);
    const ridgePointsList = [
      // Main top ridge loop (4 segments)
      [toV3(rSW), toV3(rSE)],
      [toV3(rSE), toV3(rNE)],
      [toV3(rNE), toV3(rNW)],
      [toV3(rNW), toV3(rSW)],
      // 4 outer diagonal hip ridges
      [toV3(oSW), toV3(rSW)],
      [toV3(oSE), toV3(rSE)],
      [toV3(oNE), toV3(rNE)],
      [toV3(oNW), toV3(rNW)],
      // 4 inner diagonal valley caps
      [toV3(iSW), toV3(rSW)],
      [toV3(iSE), toV3(rSE)],
      [toV3(iNE), toV3(rNE)],
      [toV3(iNW), toV3(rNW)]
    ];

    return { roofGeo, soffitGeo, ridgePointsList };
  }

  /**
   * Positions a flat plaque/signage mesh cleanly offset by deltaMeters along a curved polar cylinder's outward normal
   * so signage never Z-fights or sinks into curved walls.
   */
  static placeOnPolarCylinder(mesh, radius, thetaRad, y, deltaMeters = 0.025, faceOutward = true) {
    const r = radius + (faceOutward ? deltaMeters : -deltaMeters);
    const c = Math.cos(thetaRad);
    const s = Math.sin(thetaRad);
    mesh.position.set(r * c, y, r * s);
    // Outward normal is (c, 0, s); Three.js PlaneGeometry faces +Z by default, so rotation.y = atan2(c, s)
    const yaw = Math.atan2(c, s) + (faceOutward ? 0 : Math.PI);
    mesh.rotation.set(0, yaw, 0);
    return mesh;
  }
}

/**
 * Multi-Color 3D Relief Sign & Plaque Geometry Generator (Replaces 2D Canvas Sign Atlas with
 * per-vertex colored 3D relief geometry so all Komplekita signs bake into horizontalGeo with +0 Draw Calls!).
 */
class ProceduralTextureFactory {
  static _GLYPHS_5X7 = {
    A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
    B: ['11110', '10001', '10001', '11110', '10001', '10001', '11110'],
    C: ['01111', '10000', '10000', '10000', '10000', '10000', '01111'],
    D: ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
    E: ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
    F: ['11111', '10000', '10000', '11110', '10000', '10000', '10000'],
    G: ['01111', '10000', '10000', '10111', '10001', '10001', '01111'],
    H: ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
    I: ['01110', '00100', '00100', '00100', '00100', '00100', '01110'],
    J: ['00111', '00010', '00010', '00010', '10010', '10010', '01100'],
    K: ['10001', '10010', '10100', '11000', '10100', '10010', '10001'],
    L: ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
    M: ['10001', '11011', '10101', '10101', '10001', '10001', '10001'],
    N: ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
    O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
    P: ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
    Q: ['01110', '10001', '10001', '10001', '10101', '10010', '01101'],
    R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
    S: ['01111', '10000', '10000', '01110', '00001', '00001', '11110'],
    T: ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
    U: ['10001', '10001', '10001', '10001', '10001', '10001', '01110'],
    V: ['10001', '10001', '10001', '10001', '10001', '01010', '00100'],
    W: ['10001', '10001', '10001', '10101', '10101', '11011', '10001'],
    X: ['10001', '10001', '01010', '00100', '01010', '10001', '10001'],
    Y: ['10001', '10001', '01010', '00100', '00100', '00100', '00100'],
    Z: ['11111', '00001', '00010', '00100', '01000', '10000', '11111'],
    '-': ['00000', '00000', '00000', '01110', '00000', '00000', '00000'],
    '>': ['00100', '00010', '11111', '00010', '00100', '00000', '00000']
  };

  static _pushColoredBox(pos, norm, uvs, col, cx, cy, cz, sx, sy, sz, rgb) {
    const g = new THREE.BoxGeometry(sx, sy, sz);
    const nonIdx = g.index ? g.toNonIndexed() : g;
    const p = nonIdx.attributes.position;
    const n = nonIdx.attributes.normal;
    for (let i = 0; i < p.count; i++) {
      pos.push(cx + p.getX(i), cy + p.getY(i), cz + p.getZ(i));
      norm.push(n.getX(i), n.getY(i), n.getZ(i));
      uvs.push(0.5, 0.5);
      col.push(rgb[0], rgb[1], rgb[2]);
    }
    g.dispose();
    if (nonIdx !== g) nonIdx.dispose();
  }

  static _push3DText(pos, norm, uvs, col, text, cx, cy, cz, totalW, letterH, depth, rgb) {
    const chars = String(text || '').toUpperCase().split('');
    if (!chars.length) return;
    const totalCols = chars.length * 6 - 1;
    const pxW = totalW / Math.max(1, totalCols);
    const pxH = letterH / 7.0;
    let curX = cx - totalW * 0.5;
    for (const ch of chars) {
      const grid = ProceduralTextureFactory._GLYPHS_5X7[ch];
      if (grid) {
        for (let r = 0; r < 7; r++) {
          let c = 0;
          while (c < 5) {
            if (grid[r][c] !== '1') {
              c++;
              continue;
            }
            let cEnd = c;
            while (cEnd + 1 < 5 && grid[r][cEnd + 1] === '1') cEnd++;
            const runLen = cEnd - c + 1;
            const midCol = (c + cEnd) * 0.5;
            ProceduralTextureFactory._pushColoredBox(
              pos, norm, uvs, col,
              curX + (midCol + 0.5) * pxW,
              cy + (3.0 - r) * pxH,
              cz,
              pxW * runLen * 1.04,
              pxH * 1.06,
              depth,
              rgb
            );
            c = cEnd + 1;
          }
        }
      }
      curX += pxW * 6.0;
    }
  }

  static createAtlasSignPlane(w, h, x, y, sw, sh, depthMeters = 0.048) {
    const pos = [];
    const norm = [];
    const uvs = [];
    const col = [];

    const cCream = [0.89, 0.85, 0.76];
    const cMaroon = [0.36, 0.11, 0.09];
    const cBrass = [0.83, 0.69, 0.42];
    const cSandMap = [0.78, 0.66, 0.49];
    const cWhite = [0.94, 0.95, 0.96];
    const cSilver = [0.84, 0.87, 0.90];
    const cDark = [0.12, 0.14, 0.17];
    const cGold = [0.90, 0.70, 0.14];
    const cBlue = [0.12, 0.34, 0.72];
    const cNavy = [0.08, 0.13, 0.22];
    const cGrass = [0.22, 0.39, 0.15];

    if (y >= 410 && y <= 440 && x < 200) {
      // Tile 3A: DPPD L1 Entrance Cream Wall Lettering ("GEDUNG / KSDP UI")
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cCream);
      ProceduralTextureFactory._push3DText(pos, norm, uvs, col, 'GEDUNG', 0, h * 0.18, depthMeters * 0.6, w * 0.66, h * 0.28, 0.028, cDark);
      ProceduralTextureFactory._push3DText(pos, norm, uvs, col, 'KSDP UI', 0, -h * 0.20, depthMeters * 0.6, w * 0.76, h * 0.28, 0.028, cDark);
    } else if (y >= 410 && y <= 440 && x >= 500) {
      // Tile 3B: DPPD Maroon/Gold Plaque ("GEDUNG DPPD UI")
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cMaroon);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, -w * 0.36, 0, depthMeters * 0.6, h * 0.46, h * 0.54, 0.025, cGold);
      ProceduralTextureFactory._push3DText(pos, norm, uvs, col, 'GEDUNG DPPD UI', w * 0.08, h * 0.08, depthMeters * 0.6, w * 0.72, h * 0.30, 0.028, cGold);
    } else if (y >= 150 && y <= 200) {
      // Tile 1: STP-UI L1 Silver Portico Fascia ("SCIENCE TECHNO PARK UNIVERSITAS INDONESIA")
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cSilver);
      ProceduralTextureFactory._push3DText(pos, norm, uvs, col, 'SCIENCE TECHNO PARK UI', 0, 0, depthMeters * 0.6, w * 0.88, h * 0.42, 0.028, cDark);
    } else if (y >= 265 && y <= 360) {
      // Tile 2: ILRC Atrium White Portal Beam ("INTEGRATED LABORATORY AND RESEARCH CENTER")
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cWhite);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, -w * 0.14, h * 0.24, depthMeters * 0.6, h * 0.32, h * 0.32, 0.025, cGold);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, w * 0.14, h * 0.24, depthMeters * 0.6, h * 0.44, h * 0.26, 0.025, cBlue);
      ProceduralTextureFactory._push3DText(pos, norm, uvs, col, 'INTEGRATED LABORATORY', 0, -h * 0.06, depthMeters * 0.6, w * 0.86, h * 0.20, 0.026, cDark);
      ProceduralTextureFactory._push3DText(pos, norm, uvs, col, 'UNIVERSITAS INDONESIA', 0, -h * 0.32, depthMeters * 0.6, w * 0.72, h * 0.18, 0.024, cDark);
    } else if (x === 0 && y === 528) {
      // Tile 4: Black & Yellow STP Directory Totem
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cDark);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, -h * 0.38, depthMeters * 0.55, w * 0.98, h * 0.22, 0.02, cGold);
      ProceduralTextureFactory._push3DText(pos, norm, uvs, col, 'STP UI', 0, h * 0.34, depthMeters * 0.6, w * 0.72, h * 0.11, 0.022, cWhite);
    } else if (x >= 335 && x <= 355 && y === 528) {
      // Tile 5: Brushed Gold/Brass UI-Huawei Training Center Totem
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cBrass);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, -h * 0.42, depthMeters * 0.55, w, h * 0.15, 0.02, cDark);
      ProceduralTextureFactory._push3DText(pos, norm, uvs, col, 'UI - HUAWEI', 0, h * 0.12, depthMeters * 0.6, w * 0.80, h * 0.08, 0.02, cDark);
    } else if (x === 680 && y === 528) {
      // Tile 6A: DRPM "P PARKIR KHUSUS PIMPINAN" Sign
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cWhite);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, h * 0.14, depthMeters * 0.55, w * 0.86, h * 0.54, 0.02, cBlue);
      ProceduralTextureFactory._push3DText(pos, norm, uvs, col, 'P', 0, h * 0.14, depthMeters * 0.65, w * 0.42, h * 0.38, 0.024, cWhite);
    } else if (x === 852 && y === 528) {
      // Tile 6B: Planter Topiary "STP-UI" on Lush Hedge Green + Brass 3D Letters
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cGrass);
      ProceduralTextureFactory._push3DText(pos, norm, uvs, col, 'STP-UI', 0, 0, depthMeters * 0.65, w * 0.82, h * 0.54, 0.055, cGold);
    } else if (x === 852 && y === 652) {
      // Tile 6C: Blue ADA Wheelchair Parking Stall
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cBlue);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, depthMeters * 0.55, w * 0.92, h * 0.92, 0.015, cWhite);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, depthMeters * 0.60, w * 0.84, h * 0.84, 0.018, cBlue);
    } else if (x === 938 && y >= 710) {
      // Tile 6D-bot: Blue "-> MASUK" Sign
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cWhite);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, depthMeters * 0.55, w * 0.90, h * 0.86, 0.02, cBlue);
      ProceduralTextureFactory._push3DText(pos, norm, uvs, col, '> MASUK', 0, 0, depthMeters * 0.65, w * 0.76, h * 0.34, 0.022, cWhite);
    } else if (x === 680 && y === 776) {
      // Tile 7A: DPPD Relief Campus Map Directory Board
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cSandMap);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, -h * 0.05, depthMeters * 0.6, h * 0.38, h * 0.38, 0.025, cMaroon);
      ProceduralTextureFactory._push3DText(pos, norm, uvs, col, 'PETA KAMPUS UI', 0, h * 0.34, depthMeters * 0.6, w * 0.78, h * 0.14, 0.02, cDark);
    } else if (x === 864 && y === 776) {
      // Tile 7B: ILRC Vertical Navy/Black Banner with Gold UI Makara
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cNavy);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, h * 0.12, depthMeters * 0.6, w * 0.58, w * 0.58, 0.022, cGold);
    } else {
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, depthMeters, cWhite);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    return geo;
  }

  static createAtlasBox(w, h, d, x, y, sw, sh) {
    const pos = [];
    const norm = [];
    const uvs = [];
    const col = [];
    if (x === 938 && y === 652) {
      // Rooftop Solar PV Silicon Cell Array (#152c4a w/ silver frame)
      const cPv = [0.09, 0.18, 0.32];
      const cFrame = [0.72, 0.77, 0.82];
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, d, cFrame);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, h * 0.15, 0, w * 0.96, h * 0.85, d * 0.94, cPv);
    } else if (x === 852 && y === 652) {
      // Blue ADA Wheelchair Parking Stall Slab
      const cBlue = [0.14, 0.38, 0.88];
      const cWhite = [0.95, 0.96, 0.98];
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, d, cWhite);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, h * 0.12, 0, w * 0.90, h * 0.90, d * 0.92, cBlue);
    } else {
      // Traffic/Warning Sign Box
      const cBlue = [0.12, 0.34, 0.72];
      const cWhite = [0.95, 0.96, 0.98];
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w, h, d, cWhite);
      ProceduralTextureFactory._pushColoredBox(pos, norm, uvs, col, 0, 0, 0, w * 0.90, h * 0.88, d * 1.06, cBlue);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    return geo;
  }
}

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./world-gis/BundaranSiteBuilder.js
 *
 * Layer 2 Geodetic Site, Ring Road, Perimeter Ditch, Verified Bridges, & Curated Tropical Flora Builder.
 * Strictly enforces the Negative Exclusion List (ref_088-093, ref_114, ref_163):
 * - ZERO cars, ZERO pedestrians, ZERO Street View 'X' or circular cursor discs, ZERO out-of-island buildings!
 */
class BundaranSiteBuilder {
  static build(materials) {
    const group = new THREE.Group();
    group.name = 'BundaranSiteAndLandscape';

    const rr = UI_DPPD_MASTER_PLAN.ringRoad;
    const env = UI_DPPD_MASTER_PLAN.envelope;

    // 1. Sub-Terrain Anti-Floating Plinth Skirt (y = -18.0m .. 0.0m, r = 88.0m — Hukum Emas 11)
    const skirtGeo = new THREE.CylinderGeometry(rr.rPlinthRim, rr.rPlinthRim, env.foundationSkirtDepth, 96, 1, false);
    skirtGeo.translate(0, -env.foundationSkirtDepth * 0.5, 0);
    const skirtMesh = new THREE.Mesh(skirtGeo, materials.plinthEarthSkirt);
    skirtMesh.receiveShadow = true;
    group.add(skirtMesh);

    // 2. Outer Landscaped Woodland Rim (r = 79.85m .. 88.00m, sloped gently y = 0.14m -> 0.55m)
    const outerRimGeo = NexusSSoSTKernel.createPolarSlopedRibbon(
      0, Math.PI * 2, rr.rOuterCurb, 0.14, rr.rPlinthRim, 0.55, 112, 5.0
    );
    const outerRimMesh = new THREE.Mesh(outerRimGeo, materials.lushGrass);
    outerRimMesh.receiveShadow = true;
    group.add(outerRimMesh);

    // 3. Two-Lane One-Way Counter-Clockwise Asphalt Ring Road (r = 71.50m .. 79.50m, y = 0.02m)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.asphaltRoad, 0, Math.PI * 2, rr.rAsphaltIn, rr.rAsphaltOut, -0.05, 0.02, 128,
      { top: true, bottom: false, outer: false, inner: false, startCap: false, endCap: false, uvMetersPerTile: 6.0, castShadow: false }
    );

    // Continuous Inner & Outer Solid Yellow Road Edge Lines (y = 0.032m — zero Z-fighting!)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.yellowLine, 0, Math.PI * 2, rr.rAsphaltIn + 0.22, rr.rAsphaltIn + 0.36, 0.02, 0.032, 128,
      { top: true, bottom: false, outer: false, inner: false, startCap: false, endCap: false, castShadow: false }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.yellowLine, 0, Math.PI * 2, rr.rAsphaltOut - 0.36, rr.rAsphaltOut - 0.22, 0.02, 0.032, 128,
      { top: true, bottom: false, outer: false, inner: false, startCap: false, endCap: false, castShadow: false }
    );

    // White Dashed Center Lane Divider (48 curved dashes at r = 75.50m)
    for (let i = 0; i < 48; i++) {
      const a0 = (i / 48) * Math.PI * 2;
      const a1 = a0 + (0.42 / 48) * Math.PI * 2;
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, a0, a1, rr.rCenterDash - 0.07, rr.rCenterDash + 0.07, 0.02, 0.033, 2,
        { top: true, bottom: false, outer: false, inner: false, startCap: false, endCap: false, castShadow: false }
      );
    }

    // 4. Outer Black-and-White Striped Concrete Curb (360° continuous)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.stripedCurb, 0, Math.PI * 2, rr.rAsphaltOut, rr.rOuterCurb, 0.01, 0.19, 128,
      { bottom: false, startCap: false, endCap: false, uvMetersPerTile: 1.8, castShadow: true }
    );

    // 5. Verified Entrance Driveway Cutouts & Bridges Across the Perimeter Stone Ditch
    // Angles in degrees [-180..+180] where curbs & ditch open for real bridges (ref_174, ref_206):
    const bridgeOpeningsDeg = [
      { start: -112.0, end: -104.5, type: 'asphalt', label: 'STP_Exit_NW' },
      { start: -73.5, end: -66.0, type: 'asphalt', label: 'STP_Entry_NE' },
      { start: 4.5, end: 11.5, type: 'asphalt', label: 'ILRC_East_Portico' },
      { start: 26.0, end: 33.0, type: 'asphalt', label: 'DRPM_Loop_Gate' },
      { start: 38.5, end: 43.5, type: 'koral', label: 'ILRC_Atrium_Ramp' },
      { start: 132.0, end: 139.0, type: 'paving', label: 'DPPD_Main_Apron' },
      { start: 177.5, end: 185.5, type: 'asphalt', label: 'West_Main_ITTC_Gate' }
    ];

    // Normalize openings onto [0, 360) sorted ascending to build unbroken curb/ditch segments between bridges
    const normGaps = bridgeOpeningsDeg
      .map((g) => ({
        s: ((g.start % 360) + 360) % 360,
        e: ((g.end % 360) + 360) % 360,
        raw: g
      }))
      .sort((a, b) => a.s - b.s);

    for (let i = 0; i < normGaps.length; i++) {
      const cur = normGaps[i];
      const nxt = normGaps[(i + 1) % normGaps.length];
      const segStartDeg = cur.e;
      const segEndDeg = nxt.s > cur.e ? nxt.s : nxt.s + 360.0;
      const spanDeg = segEndDeg - segStartDeg;
      if (spanDeg <= 0.5) continue;

      const th0 = segStartDeg * DEG;
      const th1 = segEndDeg * DEG;
      const segs = Math.max(4, Math.ceil(spanDeg / 2.8));

      // Inner Black-and-White Striped Curb
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.stripedCurb, th0, th1, rr.rInnerVerge, rr.rInnerCurb, 0.01, 0.19, segs,
        { bottom: false, uvMetersPerTile: 1.8 }
      );

      // Inner Grass Verge Strip (r = 69.45 .. 71.15m)
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.lushGrass, th0, th1, rr.rDitchOuter, rr.rInnerVerge, 0.0, 0.14, segs,
        { bottom: false, uvMetersPerTile: 4.0, castShadow: false }
      );

      // Open Batu-Kali Drainage Ditch Walls & Floor (r = 68.55 .. 69.45m, floor y = -0.68m)
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.stoneRetaining, th0, th1, rr.rDitchOuter - 0.16, rr.rDitchOuter, -0.68, 0.15, segs,
        { bottom: false, uvMetersPerTile: 2.2 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.stoneRetaining, th0, th1, rr.rDitchInner, rr.rDitchInner + 0.16, -0.68, 0.15, segs,
        { bottom: false, uvMetersPerTile: 2.2 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.stoneRetaining, th0, th1, rr.rDitchInner, rr.rDitchOuter, -0.74, -0.66, segs,
        { top: true, bottom: false, outer: false, inner: false, uvMetersPerTile: 2.2, castShadow: false }
      );
    }

    // Build the Verified Culvert Bridges across the ditch + curb aprons
    // (Note: DPPD_Main_Apron is custom-built in DPPDSecondarySubdomains with its exact concrete bridge & radial U-gutter!)
    for (const br of bridgeOpeningsDeg) {
      if (br.label === 'DPPD_Main_Apron') continue;
      const th0 = br.start * DEG;
      const th1 = br.end * DEG;
      const mat =
        br.type === 'koral'
          ? materials.koralSikat
          : br.type === 'paving'
          ? materials.pavingBlock
          : materials.asphaltRoad;
      NexusSSoSTKernel.addCurvedPolarBand(
        group, mat, th0, th1, rr.rDitchInner - 0.8, rr.rInnerCurb + 0.05, -0.14, 0.15, 6,
        { bottom: false, uvMetersPerTile: 3.0 }
      );
      // Low concrete bridge side cheeks
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, th0 - 0.004, th0 + 0.004, rr.rDitchInner - 0.6, rr.rInnerVerge, 0.08, 0.28, 1,
        { bottom: false }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, th1 - 0.004, th1 + 0.004, rr.rDitchInner - 0.6, rr.rInnerVerge, 0.08, 0.28, 1,
        { bottom: false }
      );
    }

    // 6. Inner Island Base Turf Disk (r = 0 .. 68.55m, y = 0.12m)
    const islandGeo = new THREE.CylinderGeometry(rr.rDitchInner, rr.rDitchInner, 0.24, 112, 1, false);
    islandGeo.translate(0, 0.0, 0); // top surface at y = +0.12m
    const islandMesh = new THREE.Mesh(islandGeo, materials.lushGrass);
    islandMesh.receiveShadow = true;
    group.add(islandMesh);

    // 7. West 9:00 Main Driveway & Courtyard Plaza to ITTC Perpustakaan Lama (ref_174, ref_201-206)
    const westDriveGeo = new THREE.BoxGeometry(46.0, 0.08, 7.6);
    westDriveGeo.translate(-47.2, 0.14, -2.2);
    const westDriveMesh = new THREE.Mesh(westDriveGeo, materials.asphaltRoad);
    westDriveMesh.receiveShadow = true;
    group.add(westDriveMesh);

    // West 9:00 Driveway Splitter Island Signage: Black/Yellow "➔ ENTRANCE" Totem + Blue "➔ MASUK" Sign (ref_203)
    {
      const totBase = new THREE.Mesh(new THREE.BoxGeometry(1.18, 0.26, 0.48), materials.architecturalWhite);
      totBase.position.set(-67.8, 0.25, -4.6);
      totBase.rotation.y = -Math.PI * 0.42;
      group.add(totBase);

      const totBody = new THREE.Mesh(new THREE.BoxGeometry(0.95, 1.85, 0.24), materials.darkMetal);
      totBody.position.set(-67.8, 1.30, -4.6);
      totBody.rotation.y = -Math.PI * 0.42;
      totBody.castShadow = true;
      group.add(totBody);

      const totSign = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasSignPlane(0.88, 1.75, 0, 528, 340, 496),
        materials.signAtlasMat
      );
      totSign.position.set(-67.93, 1.30, -4.56);
      totSign.rotation.y = -Math.PI * 0.42;
      group.add(totSign);

      // Blue "➔ MASUK" directional sign on white post (ref_203)
      const mskPost = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.045, 1.38, 8), materials.architecturalWhite);
      mskPost.position.set(-65.2, 0.78, -5.6);
      group.add(mskPost);

      const mskSign = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasBox(0.76, 0.54, 0.08, 938, 716, 86, 60),
        materials.signAtlasMat
      );
      mskSign.position.set(-65.2, 1.68, -5.6);
      mskSign.rotation.y = -Math.PI * 0.42;
      group.add(mskSign);
    }

    // Paved West Forecourt Plaza in front of ITTC Perpustakaan Lama
    const westCourtGeo = new THREE.BoxGeometry(15.5, 0.09, 26.0);
    westCourtGeo.translate(-31.5, 0.145, 1.5);
    const westCourtMesh = new THREE.Mesh(westCourtGeo, materials.pavingBlock);
    westCourtMesh.receiveShadow = true;
    group.add(westCourtMesh);

    // Courtyard Connecting Walkways (ITTC to STP-UI North & ILRC East)
    const northWalk = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.09, 20.0), materials.pavingBlock);
    northWalk.position.set(-3.0, 0.145, -29.0);
    northWalk.receiveShadow = true;
    group.add(northWalk);

    const eastWalk = new THREE.Mesh(new THREE.BoxGeometry(20.0, 0.09, 4.2), materials.pavingBlock);
    eastWalk.position.set(27.5, 0.145, 1.5);
    eastWalk.receiveShadow = true;
    group.add(eastWalk);

    // 8. STP-UI North Crescent Drop-Off Lane (r = 59.0 .. 65.8m, th = -112° .. -66°, ref_164, ref_174)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.asphaltRoad, -112 * DEG, -66 * DEG, 58.6, 65.8, 0.08, 0.16, 32,
      { bottom: false, uvMetersPerTile: 5.0, castShadow: false }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.stripedCurb, -105 * DEG, -73 * DEG, 65.8, 66.12, 0.12, 0.28, 24,
      { bottom: false, uvMetersPerTile: 1.8 }
    );

    // 9. Clean Galvanized Curved-Arm Perimeter Streetlights w/ 3D Concrete Base Pedestals, LED Luminaires, & Curb Storm Drains
    for (let i = 0; i < 12; i++) {
      const th = (i / 12) * Math.PI * 2 + 0.18;
      const rPole = rr.rOuterCurb + 0.85;
      const px = rPole * Math.cos(th);
      const pz = rPole * Math.sin(th);

      // 3D Octagonal Concrete Pedestal Base
      const ped = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.24, 0.52, 8), materials.pavingBlock);
      ped.position.set(px, 0.36, pz);
      ped.castShadow = true;
      group.add(ped);

      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.12, 8.2, 8), materials.darkMetal);
      pole.position.set(px, 4.3, pz);
      pole.castShadow = true;
      group.add(pole);

      const arm = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.09, 0.12), materials.darkMetal);
      arm.position.set(px - Math.cos(th) * 0.82, 8.3, pz - Math.sin(th) * 0.82);
      arm.rotation.y = -th;
      group.add(arm);

      // 3D Die-Cast LED Cobra-Head Luminaire Housing + Warm Lens Underside
      const lampHead = new THREE.Mesh(new THREE.BoxGeometry(0.54, 0.12, 0.26), materials.architecturalWhite);
      lampHead.position.set(px - Math.cos(th) * 1.56, 8.28, pz - Math.sin(th) * 1.56);
      lampHead.rotation.y = -th;
      group.add(lampHead);

      // 3D Cast-Iron Storm-Drain Curb Inlet Grate on the asphalt edge
      const rGrate = rr.rAsphaltOut - 0.24;
      const grate = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.05, 0.68), materials.darkMetal);
      grate.position.set(rGrate * Math.cos(th), 0.025, rGrate * Math.sin(th));
      grate.rotation.y = -th;
      group.add(grate);
    }

    // 10. Organic Multi-Lobe Tropical Campus Trees & Curved Parabolic Palms (ONLY 2 InstancedMeshes!)
    BundaranSiteBuilder._buildCuratedTropicalFlora(group, materials);

    return group;
  }

  static _buildCuratedTropicalFlora(parent, materials) {
    // A. Broadleaf Tropical Tree Placements (keeping r <= 82.4m so max crown extent stays strictly inside r = 88.0m!)
    const treeSpecs = [
      // Outer woodland ring framing the roundabout naturally (16 trees, r = 81.6..82.4m -> exact 176.00m envelope!)
      { r: 82.2, deg: -165, s: 1.08, col: 0x3b682c },
      { r: 82.0, deg: -152, s: 0.96, col: 0x4b7735 },
      { r: 82.2, deg: -132, s: 1.04, col: 0x467633 },
      { r: 82.0, deg: -116, s: 0.98, col: 0x3c6a2d },
      { r: 82.2, deg: -62,  s: 1.02, col: 0x487634 },
      { r: 82.2, deg: -24,  s: 1.08, col: 0x3d6b2e },
      { r: 81.8, deg: -6,   s: 0.95, col: 0x517c39 },
      { r: 82.2, deg: 18,   s: 1.08, col: 0x4b7837 },
      { r: 82.0, deg: 35,   s: 0.96, col: 0x3f6d2f },
      { r: 82.2, deg: 54,   s: 1.10, col: 0x366128 },
      { r: 82.0, deg: 74,   s: 1.06, col: 0x437031 },
      { r: 82.0, deg: 94,   s: 1.02, col: 0x4a7634 },
      { r: 82.2, deg: 106,  s: 1.08, col: 0x3a662b },
      { r: 82.0, deg: 124,  s: 0.96, col: 0x4e7936 },
      { r: 82.2, deg: 156,  s: 1.05, col: 0x487735 },
      { r: 81.8, deg: 168,  s: 0.95, col: 0x3d692c },

      // West Entrance Courtyard Ketapang Kencana shade trees (ref_201-205)
      { x: -56.0, z: -9.5,  s: 1.05, col: 0x5c7d38 },
      { x: -42.0, z: -10.5, s: 1.12, col: 0x4e7433 },
      { x: -34.0, z: 13.5,  s: 0.96, col: 0x557b39 },
      { x: -51.5, z: 6.2,   s: 0.88, col: 0x4a7332 },

      // Inner Green Courtyard Ring (between ITTC Perpustakaan Lama and STP-UI / ILRC / DPPD, ref_174)
      { x: -15.0, z: -25.5, s: 1.08, col: 0x447130 },
      { x: 9.0,   z: -26.5, s: 1.15, col: 0x3d692c },
      { x: 24.5,  z: -14.0, s: 1.05, col: 0x527a37 },
      { x: 25.0,  z: 16.5,  s: 1.12, col: 0x416d2e },
      { x: 4.5,   z: 27.5,  s: 1.06, col: 0x4a7634 },
      { x: -16.5, z: 26.0,  s: 0.98, col: 0x3f6b2d },

      // East ILRC Berm Ketapang Kencana & Tanjung Trees (ref_119, ref_125 — spaced so L4 Glass Box stays visible)
      { r: 65.2, deg: -14,  s: 0.92, col: 0x627c3a },
      { r: 65.4, deg: -2,   s: 0.95, col: 0x4b7434 },
      { r: 65.2, deg: 19,   s: 0.94, col: 0x5a7838 },

      // South ILRC Bamboo-Staked Leaf Mound Crest Trees & Brick Tower Shrubs (ref_028, ref_033-050, ref_076)
      { r: 64.5, deg: 49,   s: 0.92, col: 0x3d6a2c, yBase: 1.35 },
      { r: 64.5, deg: 57,   s: 0.98, col: 0x487433, yBase: 1.40 },
      { r: 64.5, deg: 65,   s: 0.94, col: 0x3a6529, yBase: 1.35 },
      { r: 64.5, deg: 72,   s: 0.88, col: 0x4d7836, yBase: 1.20 },
      { r: 63.0, deg: 80.5, s: 0.62, col: 0x4f7b38, yBase: 0.35 }
    ];

    // Static tree trunks (merged into bambooWood material -> +0 extra draw calls!)
    for (const t of treeSpecs) {
      const tx = t.x !== undefined ? t.x : t.r * Math.cos(t.deg * DEG);
      const tz = t.z !== undefined ? t.z : t.r * Math.sin(t.deg * DEG);
      const y0 = t.yBase || 0.12;
      const trunkH = 3.6 * t.s;
      const trunk = new THREE.Mesh(
        new THREE.CylinderGeometry(0.22 * t.s, 0.36 * t.s, trunkH, 7),
        materials.bambooWood
      );
      trunk.position.set(tx, y0 + trunkH * 0.5, tz);
      trunk.castShadow = true;
      trunk.receiveShadow = true;
      parent.add(trunk);
    }

    // Build 1 Multi-Lobe Organic Canopy Prototype Geometry & render all crowns in 1 InstancedMesh!
    const crownProto = BundaranSiteBuilder._createMultiLobeCrownGeometry();
    const crownInst = new THREE.InstancedMesh(crownProto, materials.foliageCanopy, treeSpecs.length);
    crownInst.name = 'NexusInstancedBroadleafCrowns';
    crownInst.castShadow = true;
    crownInst.receiveShadow = true;

    const dummy = new THREE.Object3D();
    const tmpCol = new THREE.Color();
    treeSpecs.forEach((t, idx) => {
      const tx = t.x !== undefined ? t.x : t.r * Math.cos(t.deg * DEG);
      const tz = t.z !== undefined ? t.z : t.r * Math.sin(t.deg * DEG);
      const y0 = t.yBase || 0.12;
      dummy.position.set(tx, y0 + 4.3 * t.s, tz);
      dummy.rotation.set(0, (idx * 1.91) % (Math.PI * 2), 0);
      dummy.scale.set(t.s, t.s * 0.92, t.s);
      dummy.updateMatrix();
      crownInst.setMatrixAt(idx, dummy.matrix);
      tmpCol.setHex(t.col);
      crownInst.setColorAt(idx, tmpCol);
    });
    crownInst.instanceMatrix.needsUpdate = true;
    if (crownInst.instanceColor) crownInst.instanceColor.needsUpdate = true;
    parent.add(crownInst);

    // B. Authentic Parabolic Royal Palms & Manila Palms (ref_078, ref_084, ref_012, ref_029)
    const palmSpecs = [
      // 3 Tall Royal Palms ("Palem Raja") along DPPD SE Grass Apron (ref_078, ref_084)
      { r: 66.2, deg: 122.0, h: 12.6, s: 1.15 },
      { r: 66.4, deg: 109.5, h: 13.2, s: 1.18 },
      { r: 66.2, deg: 97.5,  h: 12.4, s: 1.12 },
      // 1 Solitary Palm near SE-ILRC Gap (ref_084 right edge)
      { r: 62.5, deg: 87.5,  h: 11.2, s: 1.05 },
      // 4 Manila Palms in front of DRPM L1 Breezeway + 1 Tall Atrium Flank Palm (ref_012, ref_029)
      { r: 60.4, deg: 27.5,  h: 6.2,  s: 0.72 },
      { r: 60.4, deg: 30.0,  h: 6.5,  s: 0.74 },
      { r: 60.4, deg: 32.5,  h: 6.1,  s: 0.72 },
      { r: 60.4, deg: 35.0,  h: 6.4,  s: 0.74 },
      { r: 59.6, deg: 36.8,  h: 9.2,  s: 0.92 }
    ];

    for (const pm of palmSpecs) {
      const px = pm.r * Math.cos(pm.deg * DEG);
      const pz = pm.r * Math.sin(pm.deg * DEG);
      const trunk = new THREE.Mesh(
        new THREE.CylinderGeometry(0.18 * pm.s, 0.28 * pm.s, pm.h, 8),
        materials.palmTrunk
      );
      trunk.position.set(px, 0.12 + pm.h * 0.5, pz);
      trunk.castShadow = true;
      trunk.receiveShadow = true;
      parent.add(trunk);
    }

    const palmCrownProto = BundaranSiteBuilder._createParabolicPalmCrownGeometry();
    const palmInst = new THREE.InstancedMesh(palmCrownProto, materials.foliageCanopy, palmSpecs.length);
    palmInst.name = 'NexusInstancedPalmCrowns';
    palmInst.castShadow = true;
    palmInst.receiveShadow = true;

    palmSpecs.forEach((pm, idx) => {
      const px = pm.r * Math.cos(pm.deg * DEG);
      const pz = pm.r * Math.sin(pm.deg * DEG);
      dummy.position.set(px, 0.12 + pm.h, pz);
      dummy.rotation.set(0, idx * 1.15, 0);
      dummy.scale.setScalar(pm.s);
      dummy.updateMatrix();
      palmInst.setMatrixAt(idx, dummy.matrix);
      tmpCol.setHex(idx < 4 ? 0x497832 : 0x55873b);
      palmInst.setColorAt(idx, tmpCol);
    });
    palmInst.instanceMatrix.needsUpdate = true;
    if (palmInst.instanceColor) palmInst.instanceColor.needsUpdate = true;
    parent.add(palmInst);
  }

  static _createMultiLobeCrownGeometry() {
    // 9-Lobe Tiered Tropical Broadleaf Crown (Ketapang Kencana & Tanjung layered sub-canopies)
    const lobes = [
      { x: 0.0,  y: 1.05, z: 0.0,  rx: 2.6, ry: 1.75, rz: 2.6 },
      { x: 1.55, y: 0.15, z: 0.85, rx: 2.15, ry: 1.45, rz: 2.05 },
      { x: -1.5, y: 0.20, z: 0.95, rx: 2.10, ry: 1.40, rz: 2.15 },
      { x: 0.25, y: 0.10, z: -1.6, rx: 2.20, ry: 1.45, rz: 2.05 },
      { x: -1.1, y: 0.35, z: -1.1, rx: 1.95, ry: 1.38, rz: 1.95 },
      { x: 1.25, y: 0.55, z: -1.0, rx: 1.80, ry: 1.30, rz: 1.85 },
      { x: -0.3, y: -0.45, z: 1.65, rx: 1.95, ry: 1.25, rz: 1.85 },
      { x: 1.85, y: -0.35, z: -0.2, rx: 1.75, ry: 1.20, rz: 1.75 },
      { x: -1.8, y: -0.30, z: -0.1, rx: 1.75, ry: 1.22, rz: 1.75 }
    ];
    const posArr = [];
    const normArr = [];
    const uvArr = [];
    const v = new THREE.Vector3();
    const n = new THREE.Vector3();

    for (const lb of lobes) {
      const g = new THREE.IcosahedronGeometry(1.0, 1);
      const nonIdx = g.index ? g.toNonIndexed() : g;
      const p = nonIdx.attributes.position;
      for (let i = 0; i < p.count; i++) {
        v.fromBufferAttribute(p, i);
        // Gentle organic noise displacement so no sphere looks robotic
        const noise = 1.0 + 0.09 * Math.sin(v.x * 5.0 + v.y * 3.0) * Math.cos(v.z * 4.0);
        n.copy(v).normalize();
        posArr.push(lb.x + v.x * lb.rx * noise, lb.y + v.y * lb.ry * noise, lb.z + v.z * lb.rz * noise);
        normArr.push(n.x, n.y, n.z);
        uvArr.push(0.5 + v.x * 0.5, 0.5 + v.z * 0.5);
      }
      g.dispose();
      if (nonIdx !== g) nonIdx.dispose();
    }

    const out = new THREE.BufferGeometry();
    out.setAttribute('position', new THREE.Float32BufferAttribute(posArr, 3));
    out.setAttribute('normal', new THREE.Float32BufferAttribute(normArr, 3));
    out.setAttribute('uv', new THREE.Float32BufferAttribute(uvArr, 2));
    out.computeBoundingBox();
    out.computeBoundingSphere();
    return out;
  }

  static _createParabolicPalmCrownGeometry() {
    const pos = [];
    const norm = [];
    const uvs = [];
    const frondCount = 13;
    const segs = 6;

    // Green crownshaft cylinder at top of trunk
    const shaft = new THREE.CylinderGeometry(0.24, 0.21, 1.1, 8).toNonIndexed();
    const sp = shaft.attributes.position;
    const sn = shaft.attributes.normal;
    for (let i = 0; i < sp.count; i++) {
      pos.push(sp.getX(i), sp.getY(i) - 0.35, sp.getZ(i));
      norm.push(sn.getX(i), sn.getY(i), sn.getZ(i));
      uvs.push(0.5, 0.5);
    }
    shaft.dispose();

    // 13 Gracefully arched parabolic V-folded palm fronds (upper & lower tiers)
    for (let f = 0; f < frondCount; f++) {
      const yaw = (f / frondCount) * Math.PI * 2 + (f % 2) * 0.14;
      const len = f % 2 === 0 ? 3.55 : 3.10;
      const lift = f % 2 === 0 ? 1.15 : 0.65;
      const droop = f % 2 === 0 ? 1.85 : 2.15;
      const cosY = Math.cos(yaw);
      const sinY = Math.sin(yaw);
      const perpX = -sinY;
      const perpZ = cosY;

      for (let s = 0; s < segs; s++) {
        const t0 = s / segs;
        const t1 = (s + 1) / segs;
        const r0 = 0.15 + t0 * len;
        const r1 = 0.15 + t1 * len;
        const y0 = 0.15 + lift * Math.sin(t0 * Math.PI * 0.65) - droop * t0 * t0;
        const y1 = 0.15 + lift * Math.sin(t1 * Math.PI * 0.65) - droop * t1 * t1;
        const w0 = 0.48 * Math.sin(Math.pow(t0, 0.65) * Math.PI) + 0.04;
        const w1 = 0.48 * Math.sin(Math.pow(t1, 0.65) * Math.PI) + 0.04;

        const c0 = [cosY * r0, y0 + 0.09, sinY * r0];
        const c1 = [cosY * r1, y1 + 0.09, sinY * r1];
        const l0 = [cosY * r0 + perpX * w0, y0 - 0.08, sinY * r0 + perpZ * w0];
        const l1 = [cosY * r1 + perpX * w1, y1 - 0.08, sinY * r1 + perpZ * w1];
        const rgt0 = [cosY * r0 - perpX * w0, y0 - 0.08, sinY * r0 - perpZ * w0];
        const rgt1 = [cosY * r1 - perpX * w1, y1 - 0.08, sinY * r1 - perpZ * w1];

        const c0Bot = [cosY * r0, y0 - 0.02, sinY * r0];
        const c1Bot = [cosY * r1, y1 - 0.02, sinY * r1];
        const l0Bot = [cosY * r0 + perpX * w0, y0 - 0.13, sinY * r0 + perpZ * w0];
        const l1Bot = [cosY * r1 + perpX * w1, y1 - 0.13, sinY * r1 + perpZ * w1];
        const rgt0Bot = [cosY * r0 - perpX * w0, y0 - 0.13, sinY * r0 - perpZ * w0];
        const rgt1Bot = [cosY * r1 - perpX * w1, y1 - 0.13, sinY * r1 - perpZ * w1];

        const pushTriSigned = (a, b, c, wantUp = true) => {
          const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
          const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
          let nx = uy * vz - uz * vy;
          let ny = uz * vx - ux * vz;
          let nz = ux * vy - uy * vx;
          const nl = Math.hypot(nx, ny, nz) || 1;
          nx /= nl; ny /= nl; nz /= nl;
          if ((wantUp && ny < 0) || (!wantUp && ny > 0)) {
            return pushTriSigned(a, c, b, wantUp);
          }
          for (const pt of [a, b, c]) {
            pos.push(pt[0], pt[1], pt[2]);
            norm.push(nx, ny, nz);
            uvs.push(0.5, 0.5);
          }
        };

        // Top sun-lit leaflets of the V-ribbed frond (visible from above & oblique)
        pushTriSigned(c0, l0, l1, true);
        pushTriSigned(c0, l1, c1, true);
        pushTriSigned(c0, c1, rgt1, true);
        pushTriSigned(c0, rgt1, rgt0, true);

        // 3D Thickened Underside leaflets & central rachis keel (visible from FPV Runner eye level below!)
        pushTriSigned(c0Bot, l1Bot, l0Bot, false);
        pushTriSigned(c0Bot, c1Bot, l1Bot, false);
        pushTriSigned(c0Bot, rgt1Bot, c1Bot, false);
        pushTriSigned(c0Bot, rgt0Bot, rgt1Bot, false);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    return geo;
  }
}

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (True-Color Photo-Realistic & Watertight Edition)
 * ./library/perpustakaan-lama/index.js
 *
 * Class-A Procedural Building Module: ITTC Perpustakaan Lama Universitas Indonesia
 * Ground-Truth References: ref_096-100, ref_136-142, ref_174, ref_202-204, ref_206.
 *
 * Key Architectural Features:
 * - Strictly Cardinal N-S / E-W 42.0m x 42.0m Square Plan (NOT rotated 45°!)
 * - Solid High-Level Reinforced-Concrete Central Roof Deck ("Atap Coran / Dak Beton" at y = 13.95m, 16.0m x 16.0m)
 *   surrounded by the 4-Wing Hipped Terracotta Roof (eaveY = 14.20m, ridgeY = 18.80m) and topped by the
 *   Raised 9.6m x 9.6m Pyramidal Terracotta Monitor / Skylight Roof Cap (ref_174 — zero open pit to ground!)
 * - 4 Full Stories with Bold White Reinforced-Concrete Structural Column Grid (7 bays per side) & Spandrels
 * - Recessed L1 Colonnade w/ Solid Ceiling Soffit + West Portico Canopy facing the 9:00 Main Driveway
 * - 1-Story Flat-Roofed NW Courtyard Annex (13.0m x 7.5m, ref_174)
 */
class PerpustakaanLamaBuilder {
  static build(materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.ittc;
    const group = new THREE.Group();
    group.name = 'ITTC_Perpustakaan_Lama_UI';

    const localRoot = new THREE.Group();
    localRoot.position.set(cfg.centerX, 0, cfg.centerZ);
    group.add(localRoot);

    const H = cfg.outerHalf;     // 21.0m
    const C = cfg.courtyardHalf; // 8.0m
    const eaveY = cfg.eaveY;     // 14.20m
    const floorHeights = [0.15, 3.85, 7.30, 10.75, eaveY];

    // 1. Solid Full-Footprint Concrete Plinth Base (42.8m x 42.8m, y = -0.20 .. 0.24m)
    const plinthSlab = new THREE.Mesh(
      new THREE.BoxGeometry((H + 0.4) * 2, 0.44, (H + 0.4) * 2),
      materials.pavingBlock
    );
    plinthSlab.position.set(0, 0.02, 0);
    plinthSlab.receiveShadow = true;
    localRoot.add(plinthSlab);

    // 2. Recessed L1 Core Walls (inset 2.1m behind outer colonnade columns -> half = 18.9m)
    const l1Core = new THREE.Mesh(
      new THREE.BoxGeometry((H - 2.1) * 2, floorHeights[1] - 0.24, (H - 2.1) * 2),
      materials.windowGlass
    );
    l1Core.position.set(0, 0.24 + (floorHeights[1] - 0.24) * 0.5, 0);
    l1Core.castShadow = true;
    l1Core.receiveShadow = true;
    localRoot.add(l1Core);

    // 3. Upper Stories L2, L3, L4 Recessed Window Curtain Box (half = H - 0.55m, completely watertight!)
    const upperCore = new THREE.Mesh(
      new THREE.BoxGeometry((H - 0.55) * 2, eaveY - 0.25 - floorHeights[1], (H - 0.55) * 2),
      materials.windowGlass
    );
    upperCore.position.set(0, floorHeights[1] + (eaveY - 0.25 - floorHeights[1]) * 0.5, 0);
    upperCore.castShadow = true;
    upperCore.receiveShadow = true;
    localRoot.add(upperCore);

    // 4. Crisp Projecting White Horizontal Spandrel & Floor Slabs at L2 (solid colonnade ceiling!), L3, L4, & Eave Ring Beam
    for (let f = 1; f <= 4; f++) {
      const ySlab = floorHeights[f];
      const spandrelH = f === 4 ? 0.58 : 0.92;
      const slab = new THREE.Mesh(
        new THREE.BoxGeometry((H + 0.08) * 2, spandrelH, (H + 0.08) * 2),
        materials.architecturalWhite
      );
      slab.position.set(0, ySlab - spandrelH * 0.15, 0);
      slab.castShadow = true;
      slab.receiveShadow = true;
      localRoot.add(slab);
    }

    // 5. Bold 7-Bay Structural White Concrete Columns (8 columns per side = 28 perimeter columns) & Window Mullions
    const bays = 7;
    const step = (2 * H) / bays; // 6.0m structural bay
    const colGeo = new THREE.BoxGeometry(0.56, eaveY - 0.15, 0.56);
    const mullionGeo = new THREE.BoxGeometry(0.10, eaveY - floorHeights[1] - 0.3, 0.16);

    for (let side = 0; side < 4; side++) {
      const sideGroup = new THREE.Group();
      sideGroup.rotation.y = side * (Math.PI * 0.5);
      localRoot.add(sideGroup);

      for (let i = 0; i <= bays; i++) {
        if (side > 0 && (i === 0 || i === bays)) {
          // Avoid duplicating the 4 corner columns
        } else {
          const x = -H + i * step;
          const col = new THREE.Mesh(colGeo, materials.architecturalWhite);
          col.position.set(x, 0.15 + (eaveY - 0.15) * 0.5, H - 0.24);
          col.castShadow = true;
          col.receiveShadow = true;
          sideGroup.add(col);
        }

        // 4 Vertical white/silver window mullions inside each 6.0m structural bay (ref_203)
        if (i < bays) {
          const bayStart = -H + i * step;
          for (let m = 1; m <= 4; m++) {
            const mx = bayStart + (m / 5) * step;
            const mul = new THREE.Mesh(mullionGeo, materials.architecturalWhite);
            mul.position.set(mx, floorHeights[1] + (eaveY - floorHeights[1] - 0.3) * 0.5, H - 0.48);
            sideGroup.add(mul);
          }
        }
      }
    }

    // 6. West Entrance Portico Canopy & Steps (facing 9:00 Main Driveway, ref_203)
    const westCanopy = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.42, 12.6), materials.architecturalWhite);
    westCanopy.position.set(-H - 1.75, 4.05, 0);
    westCanopy.castShadow = true;
    westCanopy.receiveShadow = true;
    localRoot.add(westCanopy);

    for (const pz of [-5.2, 5.2]) {
      const pCol = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 3.85, 12), materials.architecturalWhite);
      pCol.position.set(-H - 3.2, 2.0, pz);
      pCol.castShadow = true;
      localRoot.add(pCol);
    }

    const westSteps = new THREE.Mesh(new THREE.BoxGeometry(4.4, 0.22, 13.2), materials.terracottaPlaza);
    westSteps.position.set(-H - 1.9, 0.18, 0);
    westSteps.receiveShadow = true;
    localRoot.add(westSteps);

    // West Main Entrance Double Glass Doors, Dark Frame, & Tubular Step Handrails (ref_203!)
    const westDoorFrame = new THREE.Mesh(new THREE.BoxGeometry(0.24, 3.15, 3.60), materials.darkMetal);
    westDoorFrame.position.set(-H + 1.85, 1.72, 0);
    localRoot.add(westDoorFrame);
    const westDoorGlass = new THREE.Mesh(new THREE.BoxGeometry(0.28, 2.92, 3.28), materials.windowGlass);
    westDoorGlass.position.set(-H + 1.83, 1.72, 0);
    localRoot.add(westDoorGlass);
    for (const rz of [-6.2, 6.2]) {
      const railBar = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.06, 0.06), materials.darkMetal);
      railBar.position.set(-H - 1.9, 0.96, rz);
      localRoot.add(railBar);
      for (const rx of [-H - 3.6, -H - 0.3]) {
        const railPost = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.82, 0.06), materials.darkMetal);
        railPost.position.set(rx, 0.56, rz);
        localRoot.add(railPost);
      }
    }

    // 3D Outdoor Split-AC Condenser Units along L2 & L3 Spandrel Ledges (ref_202, ref_203)
    const acBoxGeo = new THREE.BoxGeometry(0.84, 0.58, 0.36);
    for (const side of [-1, 1]) {
      for (const posC of [-15.0, -3.0, 9.0]) {
        const acNS = new THREE.Mesh(acBoxGeo, materials.architecturalWhite);
        acNS.position.set(posC, 8.15, side * (H + 0.18));
        localRoot.add(acNS);
        const acEW = new THREE.Mesh(acBoxGeo, materials.architecturalWhite);
        acEW.rotation.y = Math.PI * 0.5;
        acEW.position.set(side * (H + 0.18), 11.60, posC);
        localRoot.add(acEW);
      }
    }

    // 7. High-Level Flat Reinforced-Concrete Central Roof Deck ("Atap Coran / Dak Beton" at y = 13.95m, ref_174!)
    //    Replaces the inaccurate 4-story open pit so the central well is a solid concrete roof deck!
    const roofOut = H + 1.25; // 22.25m outer eave overhang
    const roofIn = C - 0.85;  // 7.15m inner roof eave opening over the concrete roof deck

    const centralConcreteDeck = new THREE.Mesh(
      new THREE.BoxGeometry((C + 1.2) * 2, 0.45, (C + 1.2) * 2),
      materials.pavingBlock
    );
    centralConcreteDeck.position.set(0, eaveY - 0.32, 0);
    centralConcreteDeck.receiveShadow = true;
    localRoot.add(centralConcreteDeck);

    // White concrete parapet curb & gutter around the inner eave perimeter of the concrete roof deck
    PerpustakaanLamaBuilder._addHollowSquareSlab(
      localRoot,
      materials.architecturalWhite,
      roofIn + 0.38,
      roofIn - 0.18,
      eaveY - 0.25,
      eaveY + 0.16
    );

    // Raised Central Clerestory Monitor & Pyramidal Terracotta Roof Cap sitting ON TOP of the Concrete Roof Deck (ref_174)
    const sk = cfg.skylightHalf; // 4.8m
    const monBasePad = new THREE.Mesh(
      new THREE.BoxGeometry((sk + 0.25) * 2, 0.28, (sk + 0.25) * 2),
      materials.architecturalWhite
    );
    monBasePad.position.set(0, eaveY + 0.02, 0);
    monBasePad.castShadow = true;
    monBasePad.receiveShadow = true;
    localRoot.add(monBasePad);

    const pavWalls = new THREE.Mesh(
      new THREE.BoxGeometry(sk * 2 - 0.3, 0.95, sk * 2 - 0.3),
      materials.silverLouver
    );
    pavWalls.position.set(0, eaveY + 0.62, 0);
    pavWalls.castShadow = true;
    localRoot.add(pavWalls);

    const pavBeam = new THREE.Mesh(
      new THREE.BoxGeometry(sk * 2 + 0.15, 0.26, sk * 2 + 0.15),
      materials.architecturalWhite
    );
    pavBeam.position.set(0, eaveY + 1.15, 0);
    localRoot.add(pavBeam);

    // Pyramidal 3D Stepped Terracotta Roof Cap over the raised central monitor (y = 15.44 .. 17.49m) + 4 hip ridge caps!
    const pyrHalf = sk + 0.55;
    const pyrH = 2.05;
    const pyrBaseY = eaveY + 1.24;
    const pyrApex = [0, pyrBaseY + pyrH, 0];
    const mSW = [-pyrHalf, pyrBaseY, pyrHalf];
    const mSE = [pyrHalf, pyrBaseY, pyrHalf];
    const mNE = [pyrHalf, pyrBaseY, -pyrHalf];
    const mNW = [-pyrHalf, pyrBaseY, -pyrHalf];
    const pPos = [], pNorm = [], pUv = [];
    for (const [cA, cB] of [[mSW, mSE], [mSE, mNE], [mNE, mNW], [mNW, mSW]]) {
      NexusSSoSTKernel.pushThickSteppedRoofQuad(pPos, pNorm, pUv, cA, cB, pyrApex, pyrApex, {
        uvTileMeters: 1.7,
        courses: 7,
        stepHeight: 0.026,
        slabThickness: 0.10,
        eaveNosings: true
      });
    }
    const pyrGeo = new THREE.BufferGeometry();
    pyrGeo.setAttribute('position', new THREE.Float32BufferAttribute(pPos, 3));
    pyrGeo.setAttribute('normal', new THREE.Float32BufferAttribute(pNorm, 3));
    pyrGeo.setAttribute('uv', new THREE.Float32BufferAttribute(pUv, 2));
    pyrGeo.computeBoundingBox();
    pyrGeo.computeBoundingSphere();
    const pyrRoof = new THREE.Mesh(pyrGeo, materials.terracottaRoof);
    pyrRoof.castShadow = true;
    pyrRoof.receiveShadow = true;
    localRoot.add(pyrRoof);

    // 4 Ceramic hip ridge caps on the central pyramidal monitor roof
    const monCorner = sk + 0.52;
    const monApex = new THREE.Vector3(0, pyrBaseY + pyrH, 0);
    for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const cGeo = NexusSSoSTKernel.createRidgeCapStrip(
        [new THREE.Vector3(sx * monCorner, pyrBaseY, sz * monCorner), monApex],
        0.10,
        5
      );
      if (cGeo) {
        const cm = new THREE.Mesh(cGeo, materials.terracottaBrick);
        cm.castShadow = true;
        localRoot.add(cm);
      }
    }

    // 8. Main 4-Wing Courtyard Limasan / Hipped Terracotta Roof (Watertight SSoST)
    const { roofGeo, soffitGeo, ridgePointsList } = NexusSSoSTKernel.createSquareCourtyardLimasanRoof(
      roofOut,
      roofIn,
      eaveY,
      cfg.ridgeY,
      1.85
    );

    const mainRoof = new THREE.Mesh(roofGeo, materials.terracottaRoof);
    mainRoof.castShadow = true;
    mainRoof.receiveShadow = true;
    localRoot.add(mainRoof);

    const mainSoffit = new THREE.Mesh(soffitGeo, materials.creamStucco);
    mainSoffit.receiveShadow = true;
    localRoot.add(mainSoffit);

    // Dark timber/metal outer & inner eave fascia boards
    PerpustakaanLamaBuilder._addHollowSquareWalls(localRoot, materials.darkMetal, roofOut, roofIn, eaveY - 0.24, eaveY + 0.06);

    // 3D Ceramic Ridge Caps ("Nok Genteng") along all 12 ridge, hip, & valley lines
    for (const pts of ridgePointsList) {
      const capGeo = NexusSSoSTKernel.createRidgeCapStrip(pts, 0.13, 6);
      if (capGeo) {
        const cap = new THREE.Mesh(capGeo, materials.terracottaBrick);
        cap.castShadow = true;
        localRoot.add(cap);
      }
    }

    // 4 Corner Lightning Arrestor Spires along ridge loop
    const rRidge = 0.5 * (roofOut + roofIn);
    for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const spire = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.04, 1.6, 6), materials.darkMetal);
      spire.position.set(sx * rRidge, cfg.ridgeY + 0.8, sz * rRidge);
      localRoot.add(spire);
    }

    // 9. NW 1-Story Courtyard Annex Building (13.0m x 7.5m, ref_174 & ref_206)
    const anx = cfg.annexNW;
    const anxGroup = new THREE.Group();
    anxGroup.position.set(anx.x, 0, anx.z);
    group.add(anxGroup);

    const anxBody = new THREE.Mesh(new THREE.BoxGeometry(anx.w, anx.h - 0.35, anx.d), materials.architecturalWhite);
    anxBody.position.set(0, (anx.h - 0.35) * 0.5, 0);
    anxBody.castShadow = true;
    anxBody.receiveShadow = true;
    anxGroup.add(anxBody);

    const anxWin = new THREE.Mesh(new THREE.BoxGeometry(anx.w + 0.08, 1.45, anx.d - 1.4), materials.windowGlass);
    anxWin.position.set(0, 2.25, 0);
    anxGroup.add(anxWin);

    const anxRoof = new THREE.Mesh(new THREE.BoxGeometry(anx.w + 0.9, 0.38, anx.d + 0.9), materials.pavingBlock);
    anxRoof.position.set(0, anx.h, 0);
    anxRoof.castShadow = true;
    anxRoof.receiveShadow = true;
    anxGroup.add(anxRoof);

    return group;
  }

  static _addHollowSquareSlab(parent, mat, halfOut, halfIn, yBot, yTop) {
    const h = yTop - yBot;
    const yc = 0.5 * (yBot + yTop);
    const bandW = halfOut - halfIn;

    for (const signZ of [-1, 1]) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(halfOut * 2, h, bandW), mat);
      m.position.set(0, yc, signZ * (halfIn + bandW * 0.5));
      m.castShadow = true;
      m.receiveShadow = true;
      parent.add(m);
    }
    for (const signX of [-1, 1]) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(bandW, h, halfIn * 2), mat);
      m.position.set(signX * (halfIn + bandW * 0.5), yc, 0);
      m.castShadow = true;
      m.receiveShadow = true;
      parent.add(m);
    }
  }

  static _addHollowSquareWalls(parent, mat, halfOut, halfIn, yBot, yTop) {
    const h = yTop - yBot;
    const yc = 0.5 * (yBot + yTop);
    const t = 0.22;

    for (const r of [halfOut, halfIn]) {
      for (const sign of [-1, 1]) {
        const ns = new THREE.Mesh(new THREE.BoxGeometry(r * 2, h, t), mat);
        ns.position.set(0, yc, sign * r);
        ns.castShadow = true;
        ns.receiveShadow = true;
        parent.add(ns);

        const ew = new THREE.Mesh(new THREE.BoxGeometry(t, h, r * 2), mat);
        ew.position.set(sign * r, yc, 0);
        ew.castShadow = true;
        ew.receiveShadow = true;
        parent.add(ew);
      }
    }
  }
}

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (True-Color Photo-Realistic & Watertight Edition)
 * ./library/gedung-dppd/PrimaryStructure.js
 *
 * Stage 3 Primary Hull, SSoST Stitched Roof Valleys, Polygonal Faceted Terracotta Wing Roofs,
 * 2-Tier Sloped Grass Berm ("Talud Rumput"), Batu-Kali Retaining Walls, & NW Salmon-Brick Gable End.
 *
 * Watertight Guarantees:
 * - Zero gap between Central Gable Pavilion and Curved L1/L2 Wing Walls (closed side return walls & soffits)
 * - Exact 3D Analytical Valley Intersection between Steep Central Gable and Curved Wing Roof (no protruding rear triangle or floating pipes!)
 * - Seamless Grass Berm & Sloped Masonry Wing Wall junction right up to the Central Forecourt Stairs & Canopy
 */
class DPPDPrimaryStructure {
  static build(localGroup, materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.dppd;
    const rIn = cfg.rInner;          // 39.5m
    const rOut = cfg.rOuter;         // 57.5m
    const rWallL2 = cfg.rLoggiaWall; // 54.8m
    const rRidge = 47.2;             // Inward-offset main roof ridge (ref_174, ref_202)
    const eaveY = cfg.eaveY;         // 6.45m
    const wingRidgeY = cfg.wingRidgeY; // 11.85m

    const alphaNW = -42.5 * DEG;     // Maps to th = +178.0° (West 9:00 Gable End!)
    const alphaSE = 43.5 * DEG;      // Maps to th = +92.0° (South 6:00 Hipped End!)
    const alphaPortal = 10.6 * DEG;  // Overlaps inside x = ±10.85m central pavilion side walls -> 0.00m gap!

    const toPhi = (alpha) => Math.PI * 0.5 - alpha;

    // 1. Inner Courtyard Curved Facade & L1 Core Structure (full span alphaNW .. alphaSE, watertight caps!)
    NexusSSoSTKernel.addCurvedPolarBand(
      localGroup, materials.creamStucco, toPhi(alphaSE), toPhi(alphaNW), rIn, rWallL2, -0.2, cfg.l1TopY, 36,
      { top: true, bottom: false, startCap: true, endCap: true, uvMetersPerTile: 3.5 }
    );
    // Inner courtyard L2 cream wall + dark ribbon windows + ceiling soffit slab
    NexusSSoSTKernel.addCurvedPolarBand(
      localGroup, materials.creamStucco, toPhi(alphaSE), toPhi(alphaNW), rIn, rWallL2, cfg.l1TopY, eaveY, 36,
      { top: true, bottom: true, startCap: true, endCap: true, uvMetersPerTile: 3.5 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      localGroup, materials.windowGlass, toPhi(alphaSE) + 0.02, toPhi(alphaNW) - 0.02, rIn - 0.06, rIn + 0.12, cfg.l1TopY + 0.85, eaveY - 0.55, 36,
      { top: false, bottom: false, startCap: false, endCap: false, uvMetersPerTile: 3.2 }
    );

    // 2. Recessed L2 Loggia Back Wall, Awning Windows, & Loggia Walkway Floor + Sill Parapet
    for (const [a0, a1] of [[alphaNW + 2.5 * DEG, -alphaPortal], [alphaPortal, alphaSE - 4.5 * DEG]]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.creamStucco, toPhi(a1), toPhi(a0), rWallL2 - 0.4, rWallL2, cfg.l1TopY, eaveY, 18,
        { bottom: false, startCap: true, endCap: true, uvMetersPerTile: 3.2 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.windowGlass, toPhi(a1), toPhi(a0), rWallL2 - 0.05, rWallL2 + 0.08, cfg.l1TopY + 0.92, eaveY - 0.42, 18,
        { top: false, bottom: false, uvMetersPerTile: 2.8 }
      );
      // L2 Loggia Walkway Floor & Outer Cream Sill Parapet (y = 3.72 .. 4.44m)
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.creamStucco, toPhi(a1), toPhi(a0), rWallL2 - 0.2, rOut, cfg.l1TopY - 0.18, cfg.l1TopY + 0.08, 18,
        { bottom: true, top: true, startCap: true, endCap: true, uvMetersPerTile: 3.0 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.creamStucco, toPhi(a1), toPhi(a0), rOut - 0.36, rOut, cfg.l1TopY, cfg.l1TopY + 0.72, 18,
        { bottom: false, startCap: true, endCap: true, uvMetersPerTile: 3.0 }
      );
    }

    // Solid Flush Cream Corner Bays at SE Hip End (alphaSE - 4.5° .. alphaSE) & NW End Bay with Clerestory Windows (ref_071, ref_194)
    for (const [a0, a1] of [[alphaNW, alphaNW + 2.5 * DEG], [alphaSE - 4.5 * DEG, alphaSE]]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.creamStucco, toPhi(a1), toPhi(a0), rIn, rOut, 0.15, eaveY, 4,
        { bottom: false, startCap: true, endCap: true, uvMetersPerTile: 3.0 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.windowGlass, toPhi(a1) + 0.005, toPhi(a0) - 0.005, rOut - 0.06, rOut + 0.06, eaveY - 0.95, eaveY - 0.32, 4,
        { top: false, bottom: false, uvMetersPerTile: 2.2 }
      );
    }

    // 3. Two-Tier Sloped Green Grass Berm ("Talud Rumput") & Batu-Kali Toe Wall (NW & SE Wings, ref_079, ref_084, ref_192)
    //    Starts at ±10.2° so it tucks cleanly inside the straight forecourt wing retaining walls at x = ±10.95m (0.00m gap!)
    const rMidStep = 62.6;
    const rToe = cfg.rBermToe; // 66.8m
    const alphaBermCut = 10.85 * DEG;
    for (const [a0, a1] of [[alphaNW, -alphaBermCut], [alphaBermCut, alphaSE]]) {
      const p0 = toPhi(a1);
      const p1 = toPhi(a0);
      // Upper 32° grass slope: r = 57.3 (y = 3.72m) -> r = 62.6 (y = 1.48m)
      const upSlope = new THREE.Mesh(
        NexusSSoSTKernel.createPolarSlopedRibbon(p0, p1, rOut - 0.25, cfg.l1TopY, rMidStep, 1.48, 24, 3.5),
        materials.lushGrass
      );
      upSlope.receiveShadow = true;
      localGroup.add(upSlope);

      // Low 18cm grey concrete terrace curb step at r = 62.6m (ref_084, ref_192)
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.stoneRetaining, p0, p1, rMidStep - 0.15, rMidStep + 0.18, 1.26, 1.50, 24,
        { bottom: false, uvMetersPerTile: 2.0 }
      );

      // Lower gentle grass apron: r = 62.75 (y = 1.32m) -> r = 66.8 (y = 0.68m)
      const lowSlope = new THREE.Mesh(
        NexusSSoSTKernel.createPolarSlopedRibbon(p0, p1, rMidStep + 0.12, 1.32, rToe, 0.68, 24, 3.5),
        materials.lushGrass
      );
      lowSlope.receiveShadow = true;
      localGroup.add(lowSlope);

      // Batu-Kali Rubble-Stone Toe Retaining Wall at r = 66.8m (y = 0.10 .. 0.74m, ref_135, ref_084)
      NexusSSoSTKernel.addCurvedPolarBand(
        localGroup, materials.stoneRetaining, p0, p1, rToe - 0.28, rToe + 0.12, 0.08, 0.74, 24,
        { bottom: false, uvMetersPerTile: 2.0 }
      );
    }

    // Sloped Retaining End Cheek-Walls at SE Berm End (+43.5°) & NW Salmon-Brick Gable End (-42.5°, ref_135, ref_202)
    DPPDPrimaryStructure._addSlopedBermCheekWall(localGroup, materials.architecturalWhite, alphaSE, rOut - 0.2, cfg.l1TopY, rToe, 0.68, 0.30);
    DPPDPrimaryStructure._addSlopedBermCheekWall(localGroup, materials.architecturalWhite, alphaNW, rOut - 0.2, cfg.l1TopY, rToe - 1.2, 0.55, 0.32);

    // 4. NW Asymmetrical Salmon-Terracotta Brick Gable End-Wall at alphaNW = -42.5° (facing 9:00 West, ref_144, ref_202)
    DPPDPrimaryStructure._buildNWSalmonBrickGableWall(localGroup, materials, alphaNW, rIn, rRidge, rOut, eaveY, wingRidgeY);

    // 5. Polygonal Faceted Terracotta Wing Roofs (3 NW Facets + 3 SE Facets w/ Sloped Hip End)
    //    Stitched cleanly into the Central Steep A-Frame Gable with 0.00m Analytical Valley Seam!
    DPPDPrimaryStructure._buildStitchedWingAndGableRoofs(
      localGroup, materials, alphaNW, alphaSE, rIn - 0.95, rRidge, rOut + 1.15, eaveY, wingRidgeY, cfg.gableHalfWidth, cfg.gablePeakY
    );
  }

  static _addSlopedBermCheekWall(parent, mat, alpha, rTop, yTop, rBot, yBot, thickness = 0.30) {
    const s = Math.sin(alpha);
    const c = Math.cos(alpha);
    const nx = c * (thickness * 0.5);
    const nz = -s * (thickness * 0.5);

    const p = [
      [rTop * s - nx, 0.1, rTop * c - nz],
      [rBot * s - nx, 0.1, rBot * c - nz],
      [rBot * s - nx, yBot, rBot * c - nz],
      [rTop * s - nx, yTop, rTop * c - nz],
      [rTop * s + nx, 0.1, rTop * c + nz],
      [rBot * s + nx, 0.1, rBot * c + nz],
      [rBot * s + nx, yBot, rBot * c + nz],
      [rTop * s + nx, yTop, rTop * c + nz]
    ];

    const pos = [];
    const norm = [];
    const uvs = [];
    const addQuad = (a, b, d, e) => {
      const vA = new THREE.Vector3(...a);
      const vB = new THREE.Vector3(...b);
      const vD = new THREE.Vector3(...d);
      const fn = new THREE.Vector3().subVectors(vB, vA).cross(new THREE.Vector3().subVectors(vD, vA)).normalize();
      for (const pt of [a, b, d, a, d, e]) {
        pos.push(pt[0], pt[1], pt[2]);
        norm.push(fn.x, fn.y, fn.z);
        uvs.push((pt[0] + pt[2]) / 2.5, pt[1] / 2.5);
      }
    };
    addQuad(p[0], p[1], p[2], p[3]);
    addQuad(p[5], p[4], p[7], p[6]);
    addQuad(p[3], p[2], p[6], p[7]);
    addQuad(p[1], p[5], p[6], p[2]);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
  }

  static _buildNWSalmonBrickGableWall(parent, materials, alphaNW, rIn, rRidge, rOut, eaveY, ridgeY) {
    const s = Math.sin(alphaNW);
    const c = Math.cos(alphaNW);
    const nx = -c;
    const nz = s;
    const off = 0.08;

    const wSpan = rOut - rIn;
    const rMid = 0.5 * (rIn + rOut);
    const wallGroup = new THREE.Group();
    wallGroup.position.set(rMid * s + nx * off, 0, rMid * c + nz * off);
    wallGroup.rotation.y = Math.atan2(-c, s);
    parent.add(wallGroup);

    const baseWall = new THREE.Mesh(new THREE.BoxGeometry(wSpan, eaveY - 0.15, 0.36), materials.terracottaBrick);
    baseWall.position.set(0, 0.15 + (eaveY - 0.15) * 0.5, 0);
    baseWall.castShadow = true;
    baseWall.receiveShadow = true;
    wallGroup.add(baseWall);

    for (const by of [3.72, eaveY - 0.18]) {
      const beam = new THREE.Mesh(new THREE.BoxGeometry(wSpan + 0.12, 0.34, 0.42), materials.architecturalWhite);
      beam.position.set(0, by, 0.02);
      wallGroup.add(beam);
    }
    for (let i = 0; i <= 3; i++) {
      const px = -wSpan * 0.5 + (i / 3) * wSpan;
      const pil = new THREE.Mesh(new THREE.BoxGeometry(0.38, eaveY - 0.15, 0.42), materials.architecturalWhite);
      pil.position.set(px, 0.15 + (eaveY - 0.15) * 0.5, 0.02);
      wallGroup.add(pil);
    }
    for (const bx of [-2.8, 2.8]) {
      const cw = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.72, 0.44), materials.windowGlass);
      cw.position.set(bx, 3.12, 0.02);
      wallGroup.add(cw);

      const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, eaveY - 0.2, 6), materials.yellowLine);
      pipe.position.set(bx + 2.1, 0.15 + (eaveY - 0.2) * 0.5, 0.22);
      wallGroup.add(pipe);
    }

    const xIn = rIn - 0.85 - rMid;
    const xRidge = rRidge - rMid;
    const xOut = rOut + 1.05 - rMid;
    const shape = new THREE.Shape();
    shape.moveTo(xIn, eaveY);
    shape.lineTo(xOut, eaveY);
    shape.lineTo(xRidge, ridgeY);
    shape.closePath();
    const tympGeo = new THREE.ExtrudeGeometry(shape, { depth: 0.38, bevelEnabled: false });
    tympGeo.translate(0, 0, -0.19);
    const tympMesh = new THREE.Mesh(tympGeo, materials.terracottaBrick);
    tympMesh.castShadow = true;
    tympMesh.receiveShadow = true;
    wallGroup.add(tympMesh);

    for (const [xa, ya, xb, yb] of [[xIn, eaveY, xRidge, ridgeY], [xRidge, ridgeY, xOut, eaveY]]) {
      const len = Math.hypot(xb - xa, yb - ya);
      const ang = Math.atan2(yb - ya, xb - xa);
      const rake = new THREE.Mesh(new THREE.BoxGeometry(len + 0.15, 0.28, 0.44), materials.architecturalWhite);
      rake.position.set(0.5 * (xa + xb), 0.5 * (ya + yb), 0.04);
      rake.rotation.z = ang;
      wallGroup.add(rake);
    }
  }

  static _buildStitchedWingAndGableRoofs(
    parent,
    materials,
    alphaNW,
    alphaSE,
    rEaveIn,
    rRidge,
    rEaveOut,
    eaveY,
    wingRidgeY,
    gableHalfW,
    gablePeakY
  ) {
    const pos = [];
    const norm = [];
    const uvs = [];
    const ridgeCaps = [];

    const pushQuad = (a, b, c, d, tileM = 1.8, opts = {}) => {
      NexusSSoSTKernel.pushThickSteppedRoofQuad(pos, norm, uvs, a, b, c, d, {
        uvTileMeters: tileM,
        courses: opts.courses || 10,
        stepHeight: opts.stepHeight ?? 0.028,
        slabThickness: opts.slabThickness ?? 0.08,
        eaveNosings: opts.eaveNosings !== false
      });
    };

    const pushTri = (a, b, c, tileM = 1.8, opts = {}) => {
      pushQuad(a, b, c, c, tileM, opts);
    };

    const ptPol = (r, a, y) => [r * Math.sin(a), y, r * Math.cos(a)];

    // Exact Analytical Valley Coordinates where Central Steep Gable meets Curved Wing Roof:
    // The two valleys start at the outer eave (±alphaEaveVal at rEaveOut, y = eaveY) and converge to a V-junction
    // directly on the main curved ridge at bPeak = [0, wingRidgeY, rRidge] (alpha = 0°, ref_174 Satellite Nadir!).
    const gHalfOut = gableHalfW + 0.65; // 11.85m eave overhang
    const alphaEaveVal = Math.asin(Math.min(0.99, gHalfOut / rEaveOut)); // ~11.65°

    // A. Continuous Smooth Inner Courtyard Slope across the ENTIRE building span (alphaNW .. alphaSE)
    //    Subdivided into 36 smooth angular steps matching the curved courtyard wall & soffit (0.000m chord gap!)
    const innerKeyAlphas = [alphaNW, -31.0 * DEG, -20.0 * DEG, 0.0, 20.0 * DEG, 33.5 * DEG];
    for (let i = 0; i < innerKeyAlphas.length - 1; i++) {
      const aStart = innerKeyAlphas[i];
      const aEnd = innerKeyAlphas[i + 1];
      const subSteps = 7;
      for (let s = 0; s < subSteps; s++) {
        const a0 = aStart + (s / subSteps) * (aEnd - aStart);
        const a1 = aStart + ((s + 1) / subSteps) * (aEnd - aStart);
        const r0 = ptPol(rRidge, a0, wingRidgeY);
        const r1 = ptPol(rRidge, a1, wingRidgeY);
        const i0 = ptPol(rEaveIn, a0, eaveY);
        const i1 = ptPol(rEaveIn, a1, eaveY);
        pushQuad(i1, i0, r0, r1);
      }
    }

    // B. NW Wing Outer Slope Facets (from alphaNW = -42.5° up to the exact NW Valley Line vEaveL -> bPeak!)
    const nwOuterPairs = [
      [alphaNW, -31.0 * DEG, alphaNW, -31.0 * DEG],
      [-31.0 * DEG, -20.0 * DEG, -31.0 * DEG, -20.0 * DEG],
      [-20.0 * DEG, -alphaEaveVal, -20.0 * DEG, 0.0]
    ];
    for (let i = 0; i < nwOuterPairs.length; i++) {
      const [ao0, ao1, ar0, ar1] = nwOuterPairs[i];
      const o0 = ptPol(rEaveOut, ao0, eaveY);
      const o1 = ptPol(rEaveOut, ao1, eaveY);
      const r0 = ptPol(rRidge, ar0, wingRidgeY);
      const r1 = ptPol(rRidge, ar1, wingRidgeY);
      pushQuad(o0, o1, r1, r0);
      ridgeCaps.push([new THREE.Vector3(...r0), new THREE.Vector3(...r1)]);
    }

    // C. SE Wing Outer Slope Facets (from exact SE Valley Line bPeak -> vEaveR up to 33.5°, plus Sloped Hip End to +43.5°)
    const seOuterPairs = [
      [alphaEaveVal, 20.0 * DEG, 0.0, 20.0 * DEG],
      [20.0 * DEG, 33.5 * DEG, 20.0 * DEG, 33.5 * DEG]
    ];
    for (let i = 0; i < seOuterPairs.length; i++) {
      const [ao0, ao1, ar0, ar1] = seOuterPairs[i];
      const o0 = ptPol(rEaveOut, ao0, eaveY);
      const o1 = ptPol(rEaveOut, ao1, eaveY);
      const r0 = ptPol(rRidge, ar0, wingRidgeY);
      const r1 = ptPol(rRidge, ar1, wingRidgeY);
      pushQuad(o0, o1, r1, r0);
      ridgeCaps.push([new THREE.Vector3(...r0), new THREE.Vector3(...r1)]);
    }

    // SE Sloped Hip End (from 33.5° ridge apex down to +43.5° outer & inner eave corners!)
    {
      const aHipRidge = 36.2 * DEG;
      const rStart = ptPol(rRidge, 33.5 * DEG, wingRidgeY);
      const rHipTip = ptPol(rRidge + 0.8, aHipRidge, wingRidgeY - 0.55);
      const oStart = ptPol(rEaveOut, 33.5 * DEG, eaveY);
      const oEnd = ptPol(rEaveOut, alphaSE, eaveY);
      const iStart = ptPol(rEaveIn, 33.5 * DEG, eaveY);
      const iEnd = ptPol(rEaveIn, alphaSE, eaveY);

      pushQuad(oStart, oEnd, rHipTip, rStart);
      pushQuad(iEnd, iStart, rStart, rHipTip);
      pushTri(oEnd, iEnd, rHipTip);

      ridgeCaps.push([new THREE.Vector3(...rStart), new THREE.Vector3(...rHipTip)]);
      ridgeCaps.push([new THREE.Vector3(...oEnd), new THREE.Vector3(...rHipTip)]);
      ridgeCaps.push([new THREE.Vector3(...iEnd), new THREE.Vector3(...rHipTip)]);
    }

    // D. Central Steep Symmetrical Cross-Gable Roof Stitched Directly to the V-Valley Vertices (0.00m Planar Seam!)
    //    - Lower valley vertices meet wing outer eave at vEaveL/R = ptPol(rEaveOut, ±alphaEaveVal, eaveY) (z = 57.44m)
    //    - Upper valley apex meets the main curved ridge at bPeak = [0, wingRidgeY, rRidge] (alpha = 0°)
    //    - Front gable overhang at zFront = 59.35m (1.91m in front of vEaveL/R, matching ref_084 & ref_185!)
    const zFront = 59.35;
    const fPeak = [0, wingRidgeY, zFront];
    const bPeak = ptPol(rRidge, 0.0, wingRidgeY);
    const fLeft = [-gHalfOut, eaveY, zFront];
    const fRight = [gHalfOut, eaveY, zFront];

    const vEaveL = ptPol(rEaveOut, -alphaEaveVal, eaveY);
    const vEaveR = ptPol(rEaveOut, alphaEaveVal, eaveY);

    // Left & Right 100% Planar Gable Slopes with 12 3D Stepped Tile Courses & Corrugated Eave Nosings
    pushQuad(fLeft, vEaveL, bPeak, fPeak, 1.7, { courses: 12, stepHeight: 0.030, slabThickness: 0.08, eaveNosings: true });
    pushQuad(vEaveR, fRight, fPeak, bPeak, 1.7, { courses: 12, stepHeight: 0.030, slabThickness: 0.08, eaveNosings: true });

    // Main Gable Center Ridge Cap (from front apex to T-junction on main curved ridge)
    ridgeCaps.push([new THREE.Vector3(...fPeak), new THREE.Vector3(...bPeak)]);

    const roofGeo = new THREE.BufferGeometry();
    roofGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    roofGeo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
    roofGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    roofGeo.computeBoundingBox();
    roofGeo.computeBoundingSphere();

    const roofMesh = new THREE.Mesh(roofGeo, materials.terracottaRoof);
    roofMesh.castShadow = true;
    roofMesh.receiveShadow = true;
    parent.add(roofMesh);

    // Eave Soffits & Dark Fascia Boards along Wing Roofs
    const toPhi = (alpha) => Math.PI * 0.5 - alpha;
    NexusSSoSTKernel.addCurvedPolarBand(
      parent, materials.creamStucco, toPhi(alphaSE), toPhi(alphaNW), rEaveIn, rEaveOut, eaveY - 0.18, eaveY - 0.08, 42,
      { top: true, bottom: true, outer: true, inner: true, startCap: true, endCap: true, castShadow: false }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      parent, materials.darkMetal, toPhi(alphaSE), toPhi(alphaEaveVal - 0.2 * DEG), rEaveOut - 0.08, rEaveOut + 0.04, eaveY - 0.24, eaveY + 0.02, 18,
      { bottom: false, uvMetersPerTile: 3.0 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      parent, materials.darkMetal, toPhi(-alphaEaveVal + 0.2 * DEG), toPhi(alphaNW), rEaveOut - 0.08, rEaveOut + 0.04, eaveY - 0.24, eaveY + 0.02, 18,
      { bottom: false, uvMetersPerTile: 3.0 }
    );

    // 3D Ceramic Ridge Caps along Wing & Gable Ridges/Hips
    for (const pts of ridgeCaps) {
      const raisedPts = pts.map((p) => new THREE.Vector3(p.x, p.y + 0.035, p.z));
      const cGeo = NexusSSoSTKernel.createRidgeCapStrip(raisedPts, 0.14, 6);
      if (cGeo) {
        const cm = new THREE.Mesh(cGeo, materials.terracottaBrick);
        cm.castShadow = true;
        parent.add(cm);
      }
    }

    // E. Full-Depth Watertight Cream Gable Attic Prism + Ceiling Soffit Slab (eliminates thin floating triangle!)
    //    Extends from tympZFront = 57.95m back to tympZBack = 55.20m (strictly 32cm below roof plane so zero white shows on top!)
    const tympZFront = 57.95;
    const tympZBack = 55.20;
    const tympDepth = tympZFront - tympZBack;
    const tympShape = new THREE.Shape();
    tympShape.moveTo(-gableHalfW + 0.05, eaveY - 0.16);
    tympShape.lineTo(gableHalfW - 0.05, eaveY - 0.16);
    tympShape.lineTo(0, wingRidgeY - 0.32);
    tympShape.closePath();

    const tympGeo = new THREE.ExtrudeGeometry(tympShape, { depth: tympDepth, bevelEnabled: false });
    tympGeo.translate(0, 0, tympZBack);
    const tympMesh = new THREE.Mesh(tympGeo, materials.creamStucco);
    tympMesh.castShadow = true;
    tympMesh.receiveShadow = true;
    parent.add(tympMesh);

    // Solid Horizontal Gable Eave Ceiling Soffit closing the underside strictly from vEaveL[2] (57.44m) to zFront (59.35m)
    const gableBaseSoffit = new THREE.Mesh(
      new THREE.BoxGeometry(gHalfOut * 2 - 0.16, 0.14, zFront - vEaveL[2]),
      materials.creamStucco
    );
    gableBaseSoffit.position.set(0, eaveY - 0.18, 0.5 * (vEaveL[2] + zFront));
    parent.add(gableBaseSoffit);

    // Deep front gable overhang soffit (24cm below terracotta roof!) & 3D ceramic verge tile caps + dark bargeboards
    for (const sgn of [-1, 1]) {
      const xa = sgn * gHalfOut;
      const ya = eaveY;
      const xb = 0;
      const yb = wingRidgeY;
      const slopeLen = Math.hypot(xb - xa, yb - ya);
      const ang = Math.atan2(yb - ya, xb - xa);

      const soffitDepth = zFront - tympZFront + 0.25;
      const soffit = new THREE.Mesh(
        new THREE.BoxGeometry(slopeLen - 0.30, 0.06, soffitDepth),
        materials.creamStucco
      );
      soffit.position.set(0.5 * (xa + xb), 0.5 * (ya + yb) - 0.24, tympZFront + soffitDepth * 0.5 - 0.12);
      soffit.rotation.z = ang;
      parent.add(soffit);

      const barge = new THREE.Mesh(
        new THREE.BoxGeometry(slopeLen + 0.14, 0.24, 0.16),
        materials.darkMetal
      );
      barge.position.set(0.5 * (xa + xb), 0.5 * (ya + yb) - 0.11, zFront + 0.01);
      barge.rotation.z = ang;
      parent.add(barge);

      const vergeCap = new THREE.Mesh(
        new THREE.BoxGeometry(slopeLen + 0.18, 0.12, 0.28),
        materials.terracottaRoof
      );
      vergeCap.position.set(0.5 * (xa + xb), 0.5 * (ya + yb) + 0.06, zFront + 0.02);
      vergeCap.rotation.z = ang;
      vergeCap.castShadow = true;
      parent.add(vergeCap);

      // Horizontal side eave fascia board connecting wing fascia at vEaveL/R (z = 57.42m) to front gable corner (z = 59.35m)
      const sideFasciaLen = zFront - vEaveL[2] + 0.08;
      const sideFascia = new THREE.Mesh(
        new THREE.BoxGeometry(0.12, 0.24, sideFasciaLen),
        materials.darkMetal
      );
      sideFascia.position.set(sgn * (gHalfOut - 0.04), eaveY - 0.12, 0.5 * (vEaveL[2] + zFront));
      parent.add(sideFascia);
    }

    // Recessed Apex Soffit Fixture inside the Gable Peak (ref_184)
    const apexLamp = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.08, 12), materials.goldAccent);
    apexLamp.position.set(0, wingRidgeY - 1.15, tympZFront + 0.42);
    parent.add(apexLamp);

    // Lightning Arrestor Spires (Apex + 4 Wing Ridge vertices, ref_077, ref_084)
    const apexSpire = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.04, 1.85, 6), materials.darkMetal);
    apexSpire.position.set(0, wingRidgeY + 0.9, zFront - 0.15);
    parent.add(apexSpire);

    for (const deg of [-31, -20, 20, 33.5]) {
      const sp = ptPol(rRidge, deg * DEG, wingRidgeY + 0.72);
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.035, 1.45, 6), materials.darkMetal);
      rod.position.set(sp[0], sp[1], sp[2]);
      parent.add(rod);
    }
  }
}

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (True-Color Photo-Realistic & Watertight Edition)
 * ./library/gedung-dppd/SecondarySubdomains.js
 *
 * Stage 4 Architectural Subdomains & Forecourt Details for Gedung DPPD UI (verified against ref_184, ref_185, ref_084):
 * - Watertight L1+L2 Side Return Walls & Sloped Off-White Plastered Masonry Wing Walls ("Pipi Tangga") with Grey Coping
 * - 11-Bay L2 Ribbon Window Band w/ Horizontal Transom Bar, Left Mini-Satellite Dish & Right Split-AC Unit (ref_184)
 * - L2 Wing Colonnade Square Cream Columns + Prominent Split-AC Outdoor Condenser Units w/ Fan Grilles (ref_084)
 * - Continuous Straight Hipped L1 Terracotta Portico Canopy (NO Street-View-UI chevron gablet artifact!), 4 Cream Columns,
 *   Recessed Double Entrance Doors & Unoccluded KSDP/DPPD Wall Signage
 * - Authentic Forecourt Stair & Ramp Layout (ref_184):
 *   Left Terracotta Stairs -> Sloped Red Curb -> Central Narrow Grey Stone Steps -> Central Terracotta Deck ->
 *   Twin Sloped Red-With-White-Coping Ramp Walls framing Wheelchair Ramp -> Right Terracotta Stairs
 * - Authentic Forecourt Apron, Blue ADA Wheelchair Stall, Radial Concrete U-Gutter ("Selokan Beton"),
 *   Light-Grey Concrete Driveway Bridge, UI-Huawei Brass Totem, Stone Pedestal, "DILARANG PARKIR" Sign,
 *   3D Narrow Gold Box-Truss Banner Frames & Stainless Campus Map Plaque
 */
class DPPDSecondarySubdomains {
  static build(localGroup, materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.dppd;
    const eaveY = cfg.eaveY;   // 6.45m
    const l1TopY = cfg.l1TopY; // 3.72m

    // 1. Watertight Central Gable Pavilion Box & Side Return Walls (z = 52.8m .. 57.85m, x ∈ [-10.86m, +10.86m])
    //    Completely seals the side junction between the flat front gable projection and the curved main wings!
    const pavWidth = 21.72;
    const pavZBack = 52.8;
    const pavZFront = 57.85;
    const pavDepth = pavZFront - pavZBack;
    const pavZMid = 0.5 * (pavZBack + pavZFront);

    // Solid L2 floor / L1 ceiling slab inside central pavilion
    const pavFloorL2 = new THREE.Mesh(
      new THREE.BoxGeometry(pavWidth, 0.28, pavDepth),
      materials.creamStucco
    );
    pavFloorL2.position.set(0, l1TopY, pavZMid);
    localGroup.add(pavFloorL2);

    // Solid Left & Right Side Return Walls (x = ±10.68m, y = 0.15 .. eaveY, z = 52.8 .. 57.85m)
    for (const sgn of [-1, 1]) {
      const sideReturnWall = new THREE.Mesh(
        new THREE.BoxGeometry(0.36, eaveY - 0.15, pavDepth),
        materials.creamStucco
      );
      sideReturnWall.position.set(sgn * 10.68, 0.15 + (eaveY - 0.15) * 0.5, pavZMid);
      sideReturnWall.castShadow = true;
      sideReturnWall.receiveShadow = true;
      localGroup.add(sideReturnWall);
    }

    // Front L2 Sill Wall + 11-Bay Dark Glass Window Band (z = 57.78m, ref_084, ref_184)
    const frontWallL2 = new THREE.Mesh(new THREE.BoxGeometry(pavWidth, 0.88, 0.52), materials.creamStucco);
    frontWallL2.position.set(0, l1TopY + 0.44, 57.78);
    frontWallL2.castShadow = true;
    localGroup.add(frontWallL2);

    const winBand = new THREE.Mesh(new THREE.BoxGeometry(20.9, 1.88, 0.32), materials.windowGlass);
    winBand.position.set(0, l1TopY + 1.78, 57.73);
    localGroup.add(winBand);

    // Horizontal dark transom bar + top/bottom 3D window frame sills on L2 11-Bay Window Band (ref_184)
    for (const wy of [l1TopY + 0.86, l1TopY + 1.36, l1TopY + 2.70]) {
      const winFrameBar = new THREE.Mesh(new THREE.BoxGeometry(21.0, 0.08, 0.38), materials.darkMetal);
      winFrameBar.position.set(0, wy, 57.76);
      localGroup.add(winFrameBar);
    }
    // 22 Vertical Dark Aluminum Sub-Mullions inside the 11 Window Bays (2 panes per bay, ref_184!)
    for (let i = 0; i < 11; i++) {
      const bx = -10.45 + ((i + 0.5) / 11) * 20.9;
      const subMul = new THREE.Mesh(new THREE.BoxGeometry(0.055, 1.88, 0.36), materials.darkMetal);
      subMul.position.set(bx, l1TopY + 1.78, 57.76);
      localGroup.add(subMul);
    }

    // 12 Crisp Vertical Cream Piers forming 11 Symmetrical Window Bays (5 Left + 1 Center + 5 Right, ref_084, ref_184)
    for (let i = 0; i <= 11; i++) {
      const px = -10.45 + (i / 11) * 20.9;
      const pier = new THREE.Mesh(new THREE.BoxGeometry(0.24, 1.96, 0.38), materials.creamStucco);
      pier.position.set(px, l1TopY + 1.78, 57.80);
      pier.castShadow = true;
      localGroup.add(pier);
    }

    // Small White Round Satellite Mini-Dish w/ Dark Bracket on left pier #1 (x = -8.55m, ref_184)
    const dishArm = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 0.24), materials.darkMetal);
    dishArm.position.set(-8.55, l1TopY + 1.12, 57.94);
    localGroup.add(dishArm);
    const dish = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.15, 0.05, 14), materials.architecturalWhite);
    dish.rotation.x = Math.PI * 0.44;
    dish.position.set(-8.55, l1TopY + 1.14, 58.06);
    localGroup.add(dish);

    // White Split-AC Outdoor Condenser Unit at far-right bay of L2 Window Band (x = +9.65m, ref_184, ref_084!)
    const rightGableAc = new THREE.Mesh(new THREE.BoxGeometry(0.84, 0.58, 0.34), materials.architecturalWhite);
    rightGableAc.position.set(9.65, l1TopY + 1.08, 57.96);
    rightGableAc.castShadow = true;
    localGroup.add(rightGableAc);
    const rightGableFan = new THREE.Mesh(new THREE.CylinderGeometry(0.20, 0.20, 0.36, 12), materials.darkMetal);
    rightGableFan.rotation.x = Math.PI * 0.5;
    rightGableFan.position.set(9.55, l1TopY + 1.08, 57.97);
    localGroup.add(rightGableFan);

    // 2. L2 Wing Loggia Colonnade Columns + Prominent Split-AC Outdoor Units w/ Dark Fan Grilles (ref_084, ref_195)
    const rCol = cfg.rOuter - 0.22; // 57.28m
    const colH = eaveY - (l1TopY + 0.70);
    const colGeo = new THREE.BoxGeometry(0.34, colH, 0.34);
    const acGeo = new THREE.BoxGeometry(0.88, 0.62, 0.36);
    const fanGeo = new THREE.CylinderGeometry(0.22, 0.22, 0.38, 12);
    fanGeo.rotateX(Math.PI * 0.5);

    for (const sgn of [-1, 1]) {
      const aStart = 12.2;
      const aEnd = sgn < 0 ? 39.5 : 38.5;
      const cols = 11;
      for (let i = 0; i < cols; i++) {
        const alpha = sgn * (aStart + (i / (cols - 1)) * (aEnd - aStart)) * DEG;
        const cx = rCol * Math.sin(alpha);
        const cz = rCol * Math.cos(alpha);
        const col = new THREE.Mesh(colGeo, materials.creamStucco);
        col.position.set(cx, l1TopY + 0.70 + colH * 0.5, cz);
        col.rotation.y = alpha;
        col.castShadow = true;
        col.receiveShadow = true;
        localGroup.add(col);

        // Place white split-AC condenser units in bays matching ref_084 & ref_184
        if (i < cols - 1 && [0, 1, 2, 4, 6, 7, 8].includes(i)) {
          const midA = sgn * (aStart + ((i + 0.52) / (cols - 1)) * (aEnd - aStart)) * DEG;
          const rAc = rCol + 0.02;
          const ac = new THREE.Mesh(acGeo, materials.architecturalWhite);
          ac.position.set(rAc * Math.sin(midA), l1TopY + 1.05, rAc * Math.cos(midA));
          ac.rotation.y = midA;
          ac.castShadow = true;
          localGroup.add(ac);

          const fan = new THREE.Mesh(fanGeo, materials.darkMetal);
          fan.position.set(rAc * Math.sin(midA) - Math.cos(midA) * 0.12, l1TopY + 1.05, rAc * Math.cos(midA) + Math.sin(midA) * 0.12);
          fan.rotation.y = midA;
          localGroup.add(fan);
        }
      }
    }

    // 3. Recessed Ground-Floor (L1) Main Entrance Wall, Glass Double Doors, & Unoccluded Wall Signage (ref_084, ref_184)
    const zL1Wall = 56.85;
    const l1Wall = new THREE.Mesh(new THREE.BoxGeometry(pavWidth, l1TopY - 0.15, 0.55), materials.creamStucco);
    l1Wall.position.set(0, 0.15 + (l1TopY - 0.15) * 0.5, zL1Wall);
    l1Wall.receiveShadow = true;
    localGroup.add(l1Wall);

    // Central Dark-Framed Glass Double Entrance Doors + Symmetrical Side Windows (ref_184)
    const doorFrame = new THREE.Mesh(new THREE.BoxGeometry(3.15, 2.75, 0.16), materials.darkMetal);
    doorFrame.position.set(0, 0.45 + 2.75 * 0.5, zL1Wall + 0.28);
    localGroup.add(doorFrame);

    const doorGlass = new THREE.Mesh(new THREE.BoxGeometry(2.88, 2.58, 0.20), materials.windowGlass);
    doorGlass.position.set(0, 0.45 + 2.58 * 0.5, zL1Wall + 0.29);
    localGroup.add(doorGlass);

    for (const sx of [-4.8, 4.8]) {
      const slit = new THREE.Mesh(new THREE.BoxGeometry(0.68, 2.15, 0.16), materials.windowGlass);
      slit.position.set(sx, 1.75, zL1Wall + 0.28);
      localGroup.add(slit);
    }

    // Crisp Wall Signage right & left of doors at eye level (y = 2.02m so canopy never blocks it, ref_184!)
    const ksdpSign = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(2.15, 0.64, 40, 420, 440, 104),
      materials.signAtlasMat
    );
    ksdpSign.position.set(2.95, 2.02, zL1Wall + 0.30);
    localGroup.add(ksdpSign);

    const dppdPlaque = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(2.15, 0.44, 530, 428, 474, 88),
      materials.signAtlasMat
    );
    dppdPlaque.position.set(-2.95, 2.02, zL1Wall + 0.30);
    localGroup.add(dppdPlaque);

    // 4. Full-Width Straight Continuous L1 Terracotta Portico Canopy + 4 Cream Columns (ref_084, ref_184, ref_185)
    //    Upgraded with True-3D Stepped Tile Courses, 40 Sculpted 3D Eave Drip-Tile Nosings, Side Verge Caps & Recessed Downlights!
    const canW = 21.64;
    const canZIn = 57.55;
    const canZOut = 60.45;
    const canDepth = canZOut - canZIn;

    const canRoof = new THREE.Mesh(new THREE.BoxGeometry(canW, 0.16, canDepth + 0.14), materials.terracottaRoof);
    canRoof.position.set(0, 3.56, 0.5 * (canZIn + canZOut));
    canRoof.rotation.x = 0.15;
    canRoof.castShadow = true;
    canRoof.receiveShadow = true;
    localGroup.add(canRoof);

    // 6 Stepped 3D Terracotta Tile Course Overlap Bars across L1 Canopy Slope (true 3D tile relief!)
    for (let k = 0; k < 6; k++) {
      const tk = k / 6;
      const cz = canZOut - tk * canDepth;
      const cy = 3.36 + tk * 0.43 + 0.08;
      const courseBar = new THREE.Mesh(new THREE.BoxGeometry(canW + 0.04, 0.036, (canDepth / 6) + 0.04), materials.terracottaRoof);
      courseBar.position.set(0, cy, cz - (canDepth / 12));
      courseBar.rotation.x = 0.15;
      courseBar.castShadow = true;
      localGroup.add(courseBar);
    }

    // 42 Sculpted 3D Ceramic Eave Drip-Tile Nosings ("Genteng Cucuran Kanopi 3D") along the front lip of the L1 Canopy!
    for (let i = 0; i < 42; i++) {
      const tx = -canW * 0.5 + 0.26 + (i / 41) * (canW - 0.52);
      const eaveTile = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.065, 0.18), materials.terracottaRoof);
      eaveTile.position.set(tx, 3.39, canZOut + 0.04);
      eaveTile.rotation.x = 0.18;
      localGroup.add(eaveTile);
    }

    // 3D Terracotta Side Verge Caps & Rainwater Downspouts at Left & Right Ends of L1 Canopy
    for (const sgn of [-1, 1]) {
      const verge = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.20, canDepth + 0.22), materials.terracottaRoof);
      verge.position.set(sgn * (canW * 0.5 + 0.02), 3.58, 0.5 * (canZIn + canZOut));
      verge.rotation.x = 0.15;
      localGroup.add(verge);

      const downspout = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 3.25, 8), materials.darkMetal);
      downspout.position.set(sgn * 10.48, 1.82, 57.32);
      localGroup.add(downspout);
    }

    // Top ridge flashing along wall-canopy junction
    const canFlashing = new THREE.Mesh(new THREE.BoxGeometry(canW, 0.12, 0.18), materials.terracottaBrick);
    canFlashing.position.set(0, 3.82, canZIn + 0.06);
    localGroup.add(canFlashing);

    const canSoffit = new THREE.Mesh(new THREE.BoxGeometry(canW - 0.06, 0.10, canDepth + 0.02), materials.creamStucco);
    canSoffit.position.set(0, 3.28, 0.5 * (canZIn + canZOut));
    localGroup.add(canSoffit);

    // 4 Recessed 3D Downlight Fixtures in L1 Portico Ceiling Soffit + Porch Doormat & Waiting Chair (ref_184!)
    for (const lx of [-6.2, -2.2, 2.2, 6.2]) {
      const lamp = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.06, 10), materials.goldAccent);
      lamp.position.set(lx, 3.22, 0.5 * (canZIn + canZOut));
      localGroup.add(lamp);
    }
    const doormat = new THREE.Mesh(new THREE.BoxGeometry(1.85, 0.03, 0.95), materials.darkMetal);
    doormat.position.set(0, 0.48, zL1Wall + 0.92);
    localGroup.add(doormat);

    // Receptionist/Security Chair under the KSDP UI Sign (visible in ref_184!)
    const chairSeat = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.06, 0.48), materials.darkMetal);
    chairSeat.position.set(2.35, 0.92, zL1Wall + 0.68);
    localGroup.add(chairSeat);
    const chairBack = new THREE.Mesh(new THREE.BoxGeometry(0.50, 0.48, 0.06), materials.darkMetal);
    chairBack.position.set(2.35, 1.18, zL1Wall + 0.46);
    localGroup.add(chairBack);

    const canFascia = new THREE.Mesh(new THREE.BoxGeometry(canW + 0.04, 0.20, 0.10), materials.darkMetal);
    canFascia.position.set(0, 3.32, canZOut + 0.05);
    localGroup.add(canFascia);

    // 4 Square Cream Portico Columns supporting L1 Canopy (inner pair frames the central doors at x = ±1.62m, ref_184!)
    for (const cx of [-8.4, -1.62, 1.62, 8.4]) {
      const pCol = new THREE.Mesh(new THREE.BoxGeometry(0.28, 2.95, 0.28), materials.creamStucco);
      pCol.position.set(cx, 0.45 + 2.95 * 0.5, canZOut - 0.35);
      pCol.castShadow = true;
      localGroup.add(pCol);
    }

    // 5. Watertight Plastered Masonry Wing Walls ("Pipi Tangga") & Grass Berm Fill (x = ±10.82m, ref_184, ref_185, ref_084!)
    DPPDSecondarySubdomains._buildWatertightForecourtWingWalls(localGroup, materials, l1TopY);

    // 6. Terracotta-Tiled Upper Porch Deck & Authentic Stair / Grey Center Steps / Twin-Curb Ramp Sequence (ref_184!)
    const porchDeck = new THREE.Mesh(new THREE.BoxGeometry(21.64, 0.38, 4.35), materials.terracottaPlaza);
    porchDeck.position.set(0, 0.28, 59.00);
    porchDeck.receiveShadow = true;
    localGroup.add(porchDeck);

    // Forecourt Weathered Grey Paving Apron (strictly inside z = 61.15 .. 67.55m so it never buries the ditch/verge!)
    const apron = new THREE.Mesh(new THREE.BoxGeometry(21.8, 0.14, 6.4), materials.pavingBlock);
    apron.position.set(0, 0.13, 64.35);
    apron.receiveShadow = true;
    localGroup.add(apron);

    // Low Weathered Dark-Stone Retaining Step along the front edge of the Forecourt & Blue Stall (z = 67.58m, ref_184!)
    for (const [sx, sw] of [[-7.35, 6.9], [7.65, 6.3]]) {
      const frontStep = new THREE.Mesh(new THREE.BoxGeometry(sw, 0.26, 0.32), materials.stoneRetaining);
      frontStep.position.set(sx, 0.16, 67.58);
      frontStep.receiveShadow = true;
      localGroup.add(frontStep);
    }

    // Light-Grey Brushed Concrete Driveway Bridge crossing from Ring Road (z = 71.45m) onto Paving Forecourt (z = 67.40m, ref_184!)
    const driveBridge = new THREE.Mesh(new THREE.BoxGeometry(8.3, 0.16, 4.15), materials.architecturalWhite);
    driveBridge.position.set(0.45, 0.11, 69.45);
    driveBridge.receiveShadow = true;
    localGroup.add(driveBridge);

    // Open Radial Concrete U-Gutter ("Selokan Beton") w/ Two Parallel Raised Concrete Lips right at left edge of bridge
    // directly in front of the bare steel flagpole drum base (x = -4.08m, z = 67.65 .. 71.38m — ref_184 & ref_185 bottom-center!)
    const gutterFloor = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.08, 3.72), materials.stoneRetaining);
    gutterFloor.position.set(-4.08, 0.05, 69.52);
    localGroup.add(gutterFloor);
    for (const dx of [-0.19, 0.19]) {
      const gutterLip = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.22, 3.74), materials.architecturalWhite);
      gutterLip.position.set(-4.08 + dx, 0.15, 69.52);
      gutterLip.castShadow = true;
      gutterLip.receiveShadow = true;
      localGroup.add(gutterLip);
    }

    // Helper for sloped parapet cheek curbs alongside stairs & ramp (ref_184)
    const addSlopedRampCurb = (cx, zTop, yTop, zBot, yBot, hasWhiteCoping = false) => {
      const len = Math.hypot(zBot - zTop, yTop - yBot);
      const ang = Math.atan2(yTop - yBot, zBot - zTop);
      const curb = new THREE.Mesh(new THREE.BoxGeometry(0.20, 0.30, len), materials.terracottaBrick);
      curb.position.set(cx, 0.5 * (yTop + yBot), 0.5 * (zTop + zBot));
      curb.rotation.x = ang;
      curb.castShadow = true;
      localGroup.add(curb);
      if (hasWhiteCoping) {
        const cap = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.05, len + 0.02), materials.architecturalWhite);
        cap.position.set(cx, 0.5 * (yTop + yBot) + 0.16, 0.5 * (zTop + zBot));
        cap.rotation.x = ang;
        localGroup.add(cap);
      }
    };

    // Authentic Left-to-Right Stair & Ramp Sequence across the 21.6m Porch (z = 61.20m .. 63.65m, ref_184!):
    // A. Left Wide 6-Step Terracotta Stair Flight (x = -10.55m .. -2.05m, width = 8.50m)
    for (let st = 0; st < 6; st++) {
      const stepH = 0.054 * (6 - st);
      const step = new THREE.Mesh(new THREE.BoxGeometry(8.50, stepH, 0.40), materials.terracottaPlaza);
      step.position.set(-6.30, 0.15 + stepH * 0.5, 61.35 + st * 0.38);
      step.receiveShadow = true;
      localGroup.add(step);
    }

    // B. Sloped Red-Brown Curb #1 separating Left Terracotta Stairs from Central Grey Stone Steps (x = -1.96m)
    addSlopedRampCurb(-1.96, 61.18, 0.52, 63.58, 0.24, false);

    // C. Central Narrow 6-Step Grey Stone Stairway (x = -1.86m .. -0.36m, width = 1.50m — ref_184 center!)
    for (let st = 0; st < 6; st++) {
      const stepH = 0.054 * (6 - st);
      const step = new THREE.Mesh(new THREE.BoxGeometry(1.50, stepH, 0.40), materials.stoneRetaining);
      step.position.set(-1.11, 0.15 + stepH * 0.5, 61.35 + st * 0.38);
      step.receiveShadow = true;
      localGroup.add(step);
    }

    // D. Central Terracotta-Tiled Gently Sloped Deck Approach (x = -0.36m .. +3.34m, width = 3.70m, flush with grey steps!)
    const centerDeck = new THREE.Mesh(new THREE.BoxGeometry(3.70, 0.12, 2.50), materials.terracottaPlaza);
    centerDeck.position.set(1.49, 0.28, 62.38);
    centerDeck.rotation.x = 0.105;
    centerDeck.receiveShadow = true;
    localGroup.add(centerDeck);

    // E. Twin Parallel Sloped Red-Brown Ramp Walls w/ White Coping framing the ADA Wheelchair Ramp (x = +3.44m .. +5.16m, ref_184!)
    addSlopedRampCurb(3.42, 61.18, 0.56, 63.62, 0.28, true);
    const adaRamp = new THREE.Mesh(new THREE.BoxGeometry(1.54, 0.12, 2.52), materials.terracottaPlaza);
    adaRamp.position.set(4.30, 0.28, 62.40);
    adaRamp.rotation.x = 0.105;
    adaRamp.receiveShadow = true;
    localGroup.add(adaRamp);
    addSlopedRampCurb(5.18, 61.18, 0.56, 63.62, 0.28, true);

    // F. Right Wide 6-Step Terracotta Stair Flight (x = +5.28m .. +10.56m, width = 5.28m)
    for (let st = 0; st < 6; st++) {
      const stepH = 0.054 * (6 - st);
      const step = new THREE.Mesh(new THREE.BoxGeometry(5.28, stepH, 0.40), materials.terracottaPlaza);
      step.position.set(7.92, 0.15 + stepH * 0.5, 61.35 + st * 0.38);
      step.receiveShadow = true;
      localGroup.add(step);
    }

    // 7. Left Forecourt: Authentic UV-Mapped Blue ADA Wheelchair Parking Stall + Orange Cone + Stone Pedestal + Huawei Totem (ref_184)
    const blueBay = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasBox(5.2, 0.04, 2.85, 852, 652, 86, 124),
      materials.signAtlasMat
    );
    blueBay.position.set(-6.85, 0.20, 66.05);
    blueBay.receiveShadow = true;
    localGroup.add(blueBay);

    // Orange Traffic Cone at back-right corner of the Blue ADA Stall right behind the steel pole (ref_184)
    const coneBase = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.05, 0.34), materials.terracottaRoof);
    coneBase.position.set(-4.48, 0.23, 65.05);
    localGroup.add(coneBase);
    const cone = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.13, 0.56, 10), materials.terracottaRoof);
    cone.position.set(-4.48, 0.52, 65.05);
    localGroup.add(cone);

    // Small White "DILARANG PARKIR / TITIK KUMPUL" Signboard on Twin White Posts at foot of Left Sloped Wing Wall (ref_184, ref_185!)
    for (const px of [-9.72, -9.42]) {
      const sPost = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 1.05, 6), materials.architecturalWhite);
      sPost.position.set(px, 0.65, 64.55);
      localGroup.add(sPost);
    }
    const noParkSign = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.56, 0.04), materials.architecturalWhite);
    noParkSign.position.set(-9.57, 1.02, 64.55);
    noParkSign.rotation.y = 0.35;
    localGroup.add(noParkSign);
    const noParkIcon = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.05, 12), materials.terracottaRoof);
    noParkIcon.rotation.x = Math.PI * 0.5;
    noParkIcon.position.set(-9.56, 1.08, 64.56);
    noParkIcon.rotation.z = -0.35;
    localGroup.add(noParkIcon);

    // Short Carved Grey Stone Pedestal / Bollard w/ Square Cap at front-left corner of Blue ADA Stall (ref_184)
    const stonePed = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.28, 0.52, 10), materials.stoneRetaining);
    stonePed.position.set(-9.75, 0.44, 67.12);
    stonePed.castShadow = true;
    localGroup.add(stonePed);
    const stoneCap = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.10, 0.62), materials.stoneRetaining);
    stoneCap.position.set(-9.75, 0.72, 67.12);
    localGroup.add(stoneCap);

    // Brushed Gold/Brass UI-Huawei Training Center Monolith Totem on Dark Plinth (standing straight at front of Blue Stall, ref_184!)
    const huaweiPlinth = new THREE.Mesh(new THREE.BoxGeometry(1.26, 0.22, 0.36), materials.darkMetal);
    huaweiPlinth.position.set(-7.55, 0.29, 67.12);
    localGroup.add(huaweiPlinth);

    const huaweiBody = new THREE.Mesh(new THREE.BoxGeometry(1.16, 2.22, 0.20), materials.goldAccent);
    huaweiBody.position.set(-7.55, 0.40 + 2.22 * 0.5, 67.12);
    huaweiBody.castShadow = true;
    localGroup.add(huaweiBody);

    const huaweiFace = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(1.12, 2.16, 340, 528, 340, 496),
      materials.signAtlasMat
    );
    huaweiFace.position.set(-7.55, 0.40 + 2.22 * 0.5, 67.23);
    localGroup.add(huaweiFace);

    // 8. Right Forecourt: Narrow 3D Gold Box-Truss Banner Frames + Stainless Campus Map Plaque (ref_084, ref_184)
    //    Oriented perpendicular to the facade (narrow X profile, deep Z truss rungs) matching ref_184 & ref_084!
    for (const [bx, bz] of [[5.35, 66.45], [10.25, 66.35]]) {
      const basePad = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.22, 0.62), materials.darkMetal);
      basePad.position.set(bx, 0.24, bz);
      localGroup.add(basePad);

      for (const lz of [-0.20, 0.20]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.05, 5.1, 0.05), materials.goldAccent);
        leg.position.set(bx, 0.32 + 2.55, bz + lz);
        leg.castShadow = true;
        localGroup.add(leg);
      }
      for (let r = 0; r < 9; r++) {
        const rung = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.04, 0.44), materials.goldAccent);
        rung.position.set(bx, 0.75 + r * 0.56, bz);
        localGroup.add(rung);
      }
    }

    // Stainless-Steel Framed Campus Map Directory Signboard on Dark Plinth & Stainless Cylinder Post (ref_084, ref_184)
    const mapPlinth = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.20, 0.54), materials.darkMetal);
    mapPlinth.position.set(7.65, 0.24, 66.25);
    localGroup.add(mapPlinth);

    const mapPost = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.10, 1.25, 12), materials.silverLouver);
    mapPost.position.set(7.65, 0.85, 66.25);
    localGroup.add(mapPost);

    const mapFrame = new THREE.Mesh(new THREE.BoxGeometry(1.46, 1.04, 0.12), materials.silverLouver);
    mapFrame.position.set(7.65, 1.82, 66.25);
    mapFrame.castShadow = true;
    localGroup.add(mapFrame);

    const mapFace = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(1.36, 0.94, 680, 776, 184, 248),
      materials.signAtlasMat
    );
    mapFace.position.set(7.65, 1.82, 66.32);
    localGroup.add(mapFace);

    // Ornamental Grass Bush on the Right Berm Slope (ref_084 center-right!)
    const bush = new THREE.Mesh(new THREE.DodecahedronGeometry(0.88, 1), materials.lushGrass);
    bush.scale.set(1.15, 0.95, 1.15);
    bush.position.set(14.5, 1.72, 62.2);
    bush.castShadow = true;
    localGroup.add(bush);
  }

  static _buildWatertightForecourtWingWalls(localGroup, materials, l1TopY) {
    // Off-White/Cream Plastered Masonry Wing Walls ("Pipi Tangga") w/ Weathered Grey Coping along x = ±10.82m (ref_184, ref_185, ref_084)
    // Plus a seamless Green Grass Berm Fill Wedge bridging alpha = ±9.6° right against the outer face of the wing wall!
    for (const sgn of [-1, 1]) {
      const wx = sgn * 10.82;
      const xIn = wx - 0.18;
      const xOut = wx + 0.18;

      const buildWingSegment = (zTop, yTop, zBot, yBot, mat) => {
        const p = [
          [xIn, 0.12, zTop],
          [xIn, 0.12, zBot],
          [xIn, yBot, zBot],
          [xIn, yTop, zTop],
          [xOut, 0.12, zTop],
          [xOut, 0.12, zBot],
          [xOut, yBot, zBot],
          [xOut, yTop, zTop]
        ];
        const pos = [];
        const norm = [];
        const uvs = [];
        const addQuad = (a, b, d, e) => {
          const vA = new THREE.Vector3(...a);
          const vB = new THREE.Vector3(...b);
          const vD = new THREE.Vector3(...d);
          const fn = new THREE.Vector3().subVectors(vB, vA).cross(new THREE.Vector3().subVectors(vD, vA)).normalize();
          for (const pt of [a, b, d, a, d, e]) {
            pos.push(pt[0], pt[1], pt[2]);
            norm.push(fn.x, fn.y, fn.z);
            uvs.push((pt[0] + pt[2]) / 2.5, pt[1] / 2.5);
          }
        };
        addQuad(p[0], p[1], p[2], p[3]);
        addQuad(p[5], p[4], p[7], p[6]);
        addQuad(p[3], p[2], p[6], p[7]);
        addQuad(p[1], p[5], p[6], p[2]);

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        geo.setAttribute('normal', new THREE.Float32BufferAttribute(norm, 3));
        geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        localGroup.add(mesh);
      };

      // 1. Upper plastered side wall closing the L1 portico right underneath the sloping L1 terracotta canopy (z = 56.2 .. 60.45m, y = 3.52m -> 3.24m, ref_184)
      buildWingSegment(56.2, 3.54, 57.55, 3.54, materials.creamStucco);
      buildWingSegment(57.55, 3.54, 60.45, 3.24, materials.creamStucco);
      // 2. Sloped plastered cheek wall descending from the front eave of the L1 canopy alongside the stairs (z = 60.45m, y = 3.24m -> z = 63.65m, y = 0.68m, ref_184!)
      buildWingSegment(60.45, 3.24, 63.65, 0.68, materials.creamStucco);
      // 3. Low weathered stone/concrete coping cap along the sloped cheek wall (ref_184)
      const capLen = Math.hypot(63.65 - 60.45, 3.24 - 0.68);
      const capAng = Math.atan2(3.24 - 0.68, 63.65 - 60.45);
      const coping = new THREE.Mesh(new THREE.BoxGeometry(0.40, 0.08, capLen + 0.08), materials.stoneRetaining);
      coping.position.set(wx, 0.5 * (3.24 + 0.68) + 0.03, 0.5 * (60.45 + 63.65));
      coping.rotation.x = capAng;
      localGroup.add(coping);

      // 4. Watertight Grass Berm Transition Wedge connecting the curved berm (alpha = 10.85°) smoothly to the outer face of the wing wall!
      const aCut = sgn * 10.85 * DEG;
      const xWall = sgn * 10.84;
      const r1 = 57.25;
      const r2 = 62.60;
      const r3 = 66.80;
      const bPos = [];
      const bNorm = [];
      const bUv = [];
      const pushGrassTri = (a, b, c) => {
        const vA = new THREE.Vector3(...a);
        const vB = new THREE.Vector3(...b);
        const vC = new THREE.Vector3(...c);
        const fn = new THREE.Vector3().subVectors(vB, vA).cross(new THREE.Vector3().subVectors(vC, vA)).normalize();
        const pts = fn.y < 0 ? [a, c, b] : [a, b, c];
        if (fn.y < 0) fn.negate();
        for (const pt of pts) {
          bPos.push(pt[0], pt[1], pt[2]);
          bNorm.push(fn.x, fn.y, fn.z);
          bUv.push(pt[0] / 3.5, pt[2] / 3.5);
        }
      };
      const pushGrassQuad = (a, b, c, d) => {
        pushGrassTri(a, b, c);
        pushGrassTri(a, c, d);
      };
      pushGrassQuad(
        [xWall, l1TopY - 0.16, r1 * Math.cos(aCut)],
        [r1 * Math.sin(aCut), l1TopY, r1 * Math.cos(aCut)],
        [r2 * Math.sin(aCut), 1.48, r2 * Math.cos(aCut)],
        [xWall, 1.48, r2 * Math.cos(aCut)]
      );
      pushGrassQuad(
        [xWall, 1.32, r2 * Math.cos(aCut)],
        [r2 * Math.sin(aCut), 1.32, r2 * Math.cos(aCut)],
        [r3 * Math.sin(aCut), 0.68, r3 * Math.cos(aCut)],
        [xWall, 0.68, r3 * Math.cos(aCut)]
      );
      const bGeo = new THREE.BufferGeometry();
      bGeo.setAttribute('position', new THREE.Float32BufferAttribute(bPos, 3));
      bGeo.setAttribute('normal', new THREE.Float32BufferAttribute(bNorm, 3));
      bGeo.setAttribute('uv', new THREE.Float32BufferAttribute(bUv, 2));
      const bMesh = new THREE.Mesh(bGeo, materials.lushGrass);
      bMesh.receiveShadow = true;
      localGroup.add(bMesh);

      // Connect the mid-berm stone terrace curb & lower batu-kali toe wall seamlessly into the wing wall (0.00m gap!)
      const midCurb = new THREE.Mesh(new THREE.BoxGeometry(1.15, 0.24, 0.34), materials.stoneRetaining);
      midCurb.position.set(sgn * 11.32, 1.38, r2 * Math.cos(aCut));
      localGroup.add(midCurb);

      const toeBridge = new THREE.Mesh(new THREE.BoxGeometry(1.90, 0.66, 0.38), materials.stoneRetaining);
      toeBridge.position.set(sgn * 11.68, 0.41, r3 * Math.cos(aCut));
      localGroup.add(toeBridge);
    }
  }
}

class DPPDKinematicRig {
  static build(localGroup, materials) {
    // 1. Authentic Circular Concrete Drum Base + Bare Galvanized Steel Pole at Corner of Blue ADA Stall (ref_184, ref_185, ref_082)
    const poleX = -4.25;
    const poleZ = 67.25;

    const drum = new THREE.Mesh(new THREE.CylinderGeometry(0.52, 0.58, 0.36, 16), materials.pavingBlock);
    drum.position.set(poleX, 0.28, poleZ);
    localGroup.add(drum);

    const poleH = 9.2;
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.085, poleH, 10), materials.silverLouver);
    pole.position.set(poleX, 0.44 + poleH * 0.5, poleZ);
    localGroup.add(pole);

    const finial = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.05, 0.12, 8), materials.darkMetal);
    finial.position.set(poleX, 0.44 + poleH + 0.05, poleZ);
    localGroup.add(finial);

    // 2. Vertical Hanging Cloth Banner on Right Gold Ladder Frame (x = 10.22m, y = 2.32m, z = 66.58m, ref_084, ref_082)
    //    Baked with per-vertex colors (Gold top band, White body, Royal-Blue lower band) + 3D breeze fold!
    const bannerGeo = new THREE.BoxGeometry(0.58, 3.85, 0.032, 4, 14, 1).toNonIndexed();
    const pos = bannerGeo.attributes.position;
    const colArr = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const py = pos.getY(i);
      const normY = Math.max(0.0, Math.min(1.0, (py + 1.925) / 3.85));
      const drape = 1.0 - normY;
      const wave = Math.sin(normY * 6.5) * 0.06 * drape;
      pos.setZ(i, pos.getZ(i) + wave);
      pos.setX(i, pos.getX(i) + 0.08 * drape * drape);
      let r = 0.94, g = 0.95, b = 0.97;
      if (normY > 0.86) {
        r = 0.88; g = 0.68; b = 0.15;
      } else if (normY < 0.34 && normY > 0.10) {
        r = 0.11; g = 0.31; b = 0.65;
      } else if (normY <= 0.10) {
        r = 0.88; g = 0.68; b = 0.15;
      }
      colArr[i * 3] = r;
      colArr[i * 3 + 1] = g;
      colArr[i * 3 + 2] = b;
    }
    bannerGeo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
    bannerGeo.computeVertexNormals();
    const bannerMesh = new THREE.Mesh(bannerGeo, materials.signAtlasMat);
    bannerMesh.position.set(10.22, 2.32, 66.58);
    localGroup.add(bannerMesh);

    for (const ry of [4.24, 0.42]) {
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.68, 8), materials.goldAccent);
      rod.rotation.z = Math.PI * 0.5;
      rod.position.set(10.25, ry, 66.58);
      localGroup.add(rod);
    }
  }
}

class GedungDPPDBuilder {
  static build(materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.dppd;
    const root = new THREE.Group();
    root.name = 'Gedung_DPPD_UI';

    const localGroup = new THREE.Group();
    localGroup.rotation.y = Math.PI * 0.5 - cfg.thAxis;
    root.add(localGroup);

    DPPDPrimaryStructure.build(localGroup, materials);
    DPPDSecondarySubdomains.build(localGroup, materials);
    DPPDKinematicRig.build(localGroup, materials);

    return { group: root };
  }
}

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./library/stp-ui/index.js
 *
 * Class-A Procedural Building Module: Gedung STP-UI (Science Techno Park Universitas Indonesia)
 * Ground-Truth References: ref_094-113, ref_115-117, ref_122-124, ref_141-174, ref_195-200.
 *
 * Key Architectural Features:
 * - 8-Story Curved Contemporary Hub (th = -150° WNW across -90° North to -34° ENE, H = 26.80m)
 * - Symmetrical Full-Height Metallic Copper ACP Corner Blades + 8-Story Copper-Fin Fire-Stair Cores at SW & NE
 * - L1 Glazed Cylindrical-Piloti Lobby (NE/Center) + Solid MEP Louver Wall (NW) + Silver Portico w/ Angled V-Columns
 * - L2-L3 Signature Facade: 24 Modules Alternating Opposing '/' & '\' Copper-Bronze ACP Trapezoids & Vertical Copper Louvers
 * - L4 Continuous Ribbon Windows; L5-L8 Smooth Cool-White Facade w/ 5 Continuous Curved White Sunshade Ledges & Box Windows
 * - L8 NE High-Level Branding: Gold UI Makara Crest + 2-Row White 3D Channel Letters ("SCIENCE TECHNO PARK / UNIVERSITAS INDONESIA")
 * - Curved Copper Roof Crown + 18 Radial Rooftop Solar PV Arrays + Sloped "STP-UI" Planter, Black/Yellow Totem & Red Hydrant Pipe
 */
class GedungSTPUIBuilder {
  static build(materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.stp;
    const group = new THREE.Group();
    group.name = 'Gedung_STP_UI';

    const th0 = cfg.thStart; // -150°
    const th1 = cfg.thEnd;   // -34°
    const rIn = cfg.rInner;  // 39.0m
    const rOut = cfg.rOuter; // 57.8m

    const thBody0 = th0 + 2.4 * DEG; // -147.6°
    const thBody1 = th1 - 2.4 * DEG; // -36.4°

    // 1. Concrete Plinth & Inner Concave Courtyard Wall (8 stories w/ curved window ribbons & sunshade ledges, ref_195)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.pavingBlock, th0, th1, rIn - 0.5, rOut + 0.6, -0.2, 0.24, 48,
      { bottom: false, uvMetersPerTile: 4.0 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thBody0, thBody1, rIn, rIn + 0.9, 0.24, cfg.roofSlabY, 44,
      { bottom: false, uvMetersPerTile: 4.0 }
    );
    for (let fl = 1; fl < 8; fl++) {
      const yf = 1.2 + fl * 3.0;
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.windowGlass, thBody0 + 0.02, thBody1 - 0.02, rIn - 0.06, rIn + 0.15, yf, yf + 1.55, 40,
        { top: false, bottom: false, uvMetersPerTile: 3.5 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, thBody0, thBody1, rIn - 0.45, rIn + 0.05, yf + 1.55, yf + 1.82, 40,
        { bottom: true, uvMetersPerTile: 4.0 }
      );
    }

    // 2. Symmetrical 8-Story Copper ACP End Corner Blades & Scissor-Stair Louver Cores at SW (-150°) & NE (-34°) (ref_111, ref_148, ref_195)
    for (const [endStart, endStop, isNE] of [[th0, thBody0, false], [thBody1, th1, true]]) {
      // Front full-height vertical Copper-Bronze ACP corner blade rising to y = 27.10m
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.copperACP, endStart, endStop, rOut - 2.0, rOut + 0.48, 0.24, cfg.crownTopY + 0.30, 3,
        { bottom: false, uvMetersPerTile: 3.2 }
      );
      // Recessed white concrete scissor-stair core + vertical copper louver screen behind the front copper pier
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, endStart + 0.004, endStop - 0.004, rIn + 2.2, rOut - 2.0, 0.24, cfg.crownTopY - 0.3, 2,
        { bottom: false, uvMetersPerTile: 3.8 }
      );
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.copperACP, endStart, endStop, rIn, rIn + 2.2, 0.24, cfg.crownTopY - 0.1, 2,
        { bottom: false, uvMetersPerTile: 3.2 }
      );
      // Vertical copper-fin stair screen tubes along outer flank of stair shaft
      const flankTh = isNE ? endStop - 0.002 : endStart + 0.002;
      const cF = Math.cos(flankTh);
      const sF = Math.sin(flankTh);
      for (let r = rIn + 3.0; r <= rOut - 2.4; r += 0.85) {
        const fin = new THREE.Mesh(new THREE.BoxGeometry(0.10, cfg.roofSlabY - 3.8, 0.22), materials.copperACP);
        fin.position.set(r * cF, 4.0 + (cfg.roofSlabY - 3.8) * 0.5, r * sF);
        fin.rotation.y = -flankTh;
        group.add(fin);
      }
    }

    // 3. Ground Floor L1: NW MEP Wall (-147.6°..-100°) + NE Glazed Lobby & Cylindrical Pilotis (-100°..-36.4°) (ref_111, ref_153)
    const thSplitL1 = -100.0 * DEG;
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thBody0, thSplitL1, rOut - 1.3, rOut - 0.55, 0.24, cfg.l1TopY, 20,
      { bottom: false, uvMetersPerTile: 3.5 }
    );
    // Clerestory windows & MEP intake louver panels on NW L1 wall
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.silverLouver, thBody0 + 0.05, thSplitL1 - 0.06, rOut - 0.62, rOut - 0.48, 0.65, 1.85, 16,
      { top: false, bottom: false, uvMetersPerTile: 2.0 }
    );

    // Glazed L1 Lobby + Thick Cylindrical White Concrete Pilotis across Center & NE
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.windowGlass, thSplitL1, thBody1 - 3.0 * DEG, rOut - 2.3, rOut - 1.7, 0.24, cfg.l1TopY, 24,
      { bottom: false, uvMetersPerTile: 3.2 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thBody1 - 3.0 * DEG, thBody1, rOut - 1.5, rOut - 0.55, 0.24, cfg.l1TopY, 4,
      { bottom: false, uvMetersPerTile: 3.2 }
    );
    for (let deg = -98; deg <= -44; deg += 6.75) {
      const rad = deg * DEG;
      const col = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, cfg.l1TopY - 0.2, 14), materials.architecturalWhite);
      col.position.set((rOut - 0.85) * Math.cos(rad), 0.2 + (cfg.l1TopY - 0.2) * 0.5, (rOut - 0.85) * Math.sin(rad));
      col.castShadow = true;
      group.add(col);
    }

    // 4. L1 North Main Drop-Off Portico Canopy w/ Angled V-Columns & Fascia Lettering (thPortico = -88°, ref_094, ref_164)
    GedungSTPUIBuilder._buildNorthPortico(group, materials, cfg.thPortico, rOut, cfg.l1TopY);

    // 5. L2–L3 Signature 2-Story Podium Screen (y = 4.30 .. 11.10m): Opposing '/' & '\' Copper ACP Trapezoids + Vertical Louvers
    // Inner dark glass wall behind screen
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.windowGlass, thBody0, thBody1, rOut - 0.95, rOut - 0.45, cfg.l1TopY, cfg.l3TopY, 44,
      { bottom: false, uvMetersPerTile: 3.2 }
    );
    // Projecting L2 soffit (full radial depth rIn .. rOut+0.25 so L1 piloti ceiling is 100% closed!) & L3 mid-rail & L4 base slab
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thBody0, thBody1, rIn, rOut + 0.25, cfg.l1TopY - 0.16, cfg.l1TopY + 0.16, 48,
      { bottom: true, top: true, uvMetersPerTile: 4.0 }
    );
    for (const yBand of [7.70, cfg.l3TopY]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, thBody0, thBody1, rOut - 0.95, rOut + 0.25, yBand - 0.16, yBand + 0.16, 48,
        { uvMetersPerTile: 4.0 }
      );
    }

    const totalScreenSpan = thBody1 - thBody0;
    const screenBays = 24;
    for (let b = 0; b < screenBays; b++) {
      const b0 = thBody0 + (b / screenBays) * totalScreenSpan;
      const b1 = thBody0 + ((b + 1) / screenBays) * totalScreenSpan;
      const bMid = 0.5 * (b0 + b1);

      if (b % 2 === 0) {
        // Angled Trapezoidal Copper-Bronze ACP Panel (tilting '/' on L2 and '\' on L3 -> chevron geometry, ref_111, ref_153)
        const tilt = ((b / 2) % 2 === 0 ? 1 : -1) * 0.22 * (b1 - b0);
        const geoL2 = NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const wFactor = 0.88 - 0.26 * v;
            const th = bMid + (u - 0.5) * (b1 - b0) * wFactor + (v - 0.5) * tilt;
            const c = Math.cos(th), s = Math.sin(th);
            p.set((rOut + 0.22) * c, cfg.l1TopY + 0.16 + v * (3.40 - 0.32), (rOut + 0.22) * s);
            n.set(c, 0, s);
          },
          0, 1, 0, 1, 2, 1
        );
        const mL2 = new THREE.Mesh(geoL2, materials.copperACP);
        mL2.castShadow = true;
        group.add(mL2);

        const geoL3 = NexusSSoSTKernel.sliceSubdomainFromMaster(
          (u, v, p, n) => {
            const wFactor = 0.62 + 0.26 * v;
            const th = bMid + (u - 0.5) * (b1 - b0) * wFactor - (v - 0.5) * tilt;
            const c = Math.cos(th), s = Math.sin(th);
            p.set((rOut + 0.22) * c, 7.70 + 0.16 + v * (3.40 - 0.32), (rOut + 0.22) * s);
            n.set(c, 0, s);
          },
          0, 1, 0, 1, 2, 1
        );
        const mL3 = new THREE.Mesh(geoL3, materials.copperACP);
        mL3.castShadow = true;
        group.add(mL3);
      } else {
        // Vertical Copper-Tube Fin Louver Bay (8 slender vertical tubes spanning L2-L3 + dark window backing, ref_111, ref_164)
        NexusSSoSTKernel.addCurvedPolarBand(
          group, materials.windowGlass, b0 + 0.3 * DEG, b1 - 0.3 * DEG, rOut - 0.36, rOut - 0.20, cfg.l1TopY + 0.55, cfg.l3TopY - 0.45, 2,
          { top: false, bottom: false }
        );
        for (let f = 1; f <= 8; f++) {
          const tf = b0 + (f / 9) * (b1 - b0);
          const fin = new THREE.Mesh(
            new THREE.BoxGeometry(0.085, cfg.l3TopY - cfg.l1TopY - 0.24, 0.16),
            materials.copperACP
          );
          fin.position.set(
            (rOut + 0.18) * Math.cos(tf),
            0.5 * (cfg.l1TopY + cfg.l3TopY),
            (rOut + 0.18) * Math.sin(tf)
          );
          fin.rotation.y = -tf;
          fin.castShadow = true;
          group.add(fin);
        }
      }
    }

    // 6. Upper Curved Architecture L4–L8 (y = 11.10 .. 25.40m):
    //    Core Wall + L4 Ribbon Windows + 5 Continuous Curved White Sunshade Ledges + Protruding White Box Window Frames!
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thBody0, thBody1, rIn + 0.85, rOut - 0.35, cfg.l1TopY, cfg.roofSlabY, 48,
      { bottom: false, inner: false, uvMetersPerTile: 4.0 }
    );

    // L4 Continuous Multi-Pane Dark Ribbon Window Band w/ Vertical Mullion Piers (y = 11.28 .. 13.80m, ref_111, ref_164)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.windowGlass, thBody0 + 3.8 * DEG, thBody1 - 3.8 * DEG, rOut - 0.42, rOut - 0.22, 11.32, 13.78, 44,
      { top: false, bottom: false, uvMetersPerTile: 2.6 }
    );
    for (let deg = -142.0; deg <= -42.0; deg += 4.5) {
      const mr = deg * DEG;
      const mul = new THREE.Mesh(new THREE.BoxGeometry(0.18, 2.52, 0.28), materials.architecturalWhite);
      mul.position.set((rOut - 0.24) * Math.cos(mr), 12.55, (rOut - 0.24) * Math.sin(mr));
      mul.rotation.y = -mr;
      group.add(mul);
    }

    // 5 Continuous Deep Curved Light-Silver/White Sunshade Ledges (above L4, L5, L6, L7, & L8 Roof Cornice, ref_111, ref_148)
    const ledgeYs = [13.95, 16.80, 19.65, 22.50, cfg.roofSlabY];
    for (let li = 0; li < ledgeYs.length; li++) {
      const ly = ledgeYs[li];
      const overhang = li === 4 ? 1.08 : 0.82;
      const thick = li === 4 ? 0.36 : 0.22;
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, thBody0, thBody1, rOut - 0.45, rOut + overhang, ly - thick * 0.5, ly + thick * 0.5, 52,
        { uvMetersPerTile: 4.0 }
      );
    }

    // Open Vertical Copper-Tube Fin Louver Screens in Outermost Bay 1 adjacent to SW & NE Copper Blades (L4-L8, ref_111, ref_148!)
    for (const [c0, c1] of [[thBody0 + 0.3 * DEG, thBody0 + 3.8 * DEG], [thBody1 - 3.8 * DEG, thBody1 - 0.3 * DEG]]) {
      // Dark recessed shadow backing
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.darkMetal, c0, c1, rOut - 0.55, rOut - 0.28, 11.15, 25.15, 3,
        { bottom: false }
      );
      // 7 vertical copper louver fins spanning L4-L8
      for (let f = 1; f <= 7; f++) {
        const tf = c0 + (f / 8) * (c1 - c0);
        const fin = new THREE.Mesh(new THREE.BoxGeometry(0.08, 14.0, 0.22), materials.copperACP);
        fin.position.set((rOut + 0.06) * Math.cos(tf), 18.15, (rOut + 0.06) * Math.sin(tf));
        fin.rotation.y = -tf;
        fin.castShadow = true;
        group.add(fin);
      }
    }

    // Authentic L5, L6, L7 & NW L8 Window Rhythm (High Horizontal Vents near SW/NE ends + Wide 4-Pane & Paired Vertical Windows, ref_111, ref_148, ref_164)
    const upperFloors = [
      { y0: 14.15, y1: 16.55, isL8: false },
      { y0: 17.00, y1: 19.40, isL8: false },
      { y0: 19.85, y1: 22.25, isL8: false },
      { y0: 22.70, y1: 25.05, isL8: true }
    ];
    for (let fi = 0; fi < upperFloors.length; fi++) {
      const fl = upperFloors[fi];
      // High horizontal transom slot vents near SW (-141°..-134°) & NE (-46°..-40°) ends (ref_111, ref_148)
      const ventAngles = fl.isL8 ? [-140.5, -136.5] : [-140.5, -136.5, -45.5, -41.5];
      for (const vDeg of ventAngles) {
        const rad = vDeg * DEG;
        const c = Math.cos(rad), s = Math.sin(rad);
        const yaw = Math.atan2(c, s);
        const vy = fl.y1 - 0.55;
        const vBox = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.56, 0.28), materials.architecturalWhite);
        vBox.position.set((rOut - 0.26) * c, vy, (rOut - 0.26) * s);
        vBox.rotation.y = yaw;
        group.add(vBox);
        const vGlass = new THREE.Mesh(new THREE.BoxGeometry(1.55, 0.38, 0.32), materials.windowGlass);
        vGlass.position.set((rOut - 0.25) * c, vy, (rOut - 0.25) * s);
        vGlass.rotation.y = yaw;
        group.add(vGlass);
      }

      // Main punched windows across central arc (-131° .. -50°)
      const endLimitDeg = fl.isL8 ? -61.5 : -49.5;
      for (let idx = 0, deg = -131.0; deg <= endLimitDeg; idx++, deg += 4.6) {
        const rad = deg * DEG;
        const yc = 0.5 * (fl.y0 + fl.y1);
        const wh = fl.y1 - fl.y0 - 0.36;
        const isWide = (idx % 3 === 1) && !fl.isL8;
        const ww = isWide ? 2.85 : 1.18;

        const c = Math.cos(rad);
        const s = Math.sin(rad);
        const yaw = Math.atan2(c, s);

        const surround = new THREE.Mesh(new THREE.BoxGeometry(ww + 0.34, wh + 0.30, 0.42), materials.architecturalWhite);
        surround.position.set((rOut - 0.20) * c, yc, (rOut - 0.20) * s);
        surround.rotation.y = yaw;
        surround.castShadow = true;
        group.add(surround);

        const pane = new THREE.Mesh(new THREE.BoxGeometry(ww, wh, 0.34), materials.windowGlass);
        pane.position.set((rOut - 0.22) * c, yc, (rOut - 0.22) * s);
        pane.rotation.y = yaw;
        group.add(pane);

        // 3D Horizontal Awning Transom Bar inside the protruding window box
        const transom = new THREE.Mesh(new THREE.BoxGeometry(ww + 0.04, 0.06, 0.38), materials.darkMetal);
        transom.position.set((rOut - 0.20) * c, yc + wh * 0.16, (rOut - 0.20) * s);
        transom.rotation.y = yaw;
        group.add(transom);
      }
    }

    // 7. NE L8 High-Level Signage Frieze: Gold UI Makara + "SCIENCE TECHNO PARK / UNIVERSITAS INDONESIA" (th = -59.5°..-37°, ref_111)
    //    Includes 3D Curved Backing Slab, Top/Bottom 3D Trim Mouldings, & 3D Raised Gold Makara Disc Relief!
    {
      const sTh0 = -59.5 * DEG;
      const sTh1 = -37.0 * DEG;
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, sTh0 - 0.2 * DEG, sTh1 + 0.2 * DEG, rOut - 0.42, rOut - 0.17, 22.62, 25.14, 16,
        { bottom: true, top: true, uvMetersPerTile: 3.5 }
      );
      // 3D Raised Gold UI Makara Relief Emblem on the right/east end of the frieze (th = -39.1°, y = 23.88m, ref_111)
      const makTh = -39.1 * DEG;
      const makDisc = new THREE.Mesh(new THREE.CylinderGeometry(0.82, 0.86, 0.12, 16), materials.goldAccent);
      makDisc.rotation.x = Math.PI * 0.5;
      makDisc.rotation.z = -Math.atan2(Math.cos(makTh), Math.sin(makTh));
      makDisc.position.set((rOut - 0.13) * Math.cos(makTh), 23.88, (rOut - 0.13) * Math.sin(makTh));
      group.add(makDisc);
      const signGeo = NexusSSoSTKernel.sliceSubdomainFromMaster(
        (u, v, p, n) => {
          const th = sTh0 + u * (sTh1 - sTh0);
          const c = Math.cos(th), s = Math.sin(th);
          p.set((rOut - 0.16) * c, 22.68 + v * 2.42, (rOut - 0.16) * s);
          n.set(c, 0, s);
        },
        0, 1, 0, 1, 16, 1
      );
      // Map UVs cleanly onto Tile 0 (yPx: 0..160 -> v: 1.0 - 160/1024 .. 1.0) of the Unified Signage Atlas!
      const uv = signGeo.attributes.uv;
      const vBot = 1.0 - 158 / 1024;
      const vTop = 1.0 - 2 / 1024;
      for (let i = 0; i < uv.count; i++) {
        // Flip U (1 - u) because facing North-East from outside looks left-to-right from -37° toward -59.5°!
        // Let's verify! Standing at NE road (-42°) looking SW toward (-48°):
        // Left side of screen is East (-37°, next to NE Copper Pier!), right side of screen is West (-59.5°)!
        // At u = 1 (th = -37°), that's the LEFT side of the sign when viewed from outside!
        // Let's verify with cross product: viewer forward F = (-cos th, 0, -sin th); viewer UP = (0, 1, 0);
        // Viewer RIGHT = F x UP = (-cos th, 0, -sin th) x (0, 1, 0) = (+sin th, 0, -cos th)!
        // Derivative dP/d(th) = (-r sin th, 0, +r cos th) = -r * Viewer_RIGHT!
        // YES!! Increasing th goes LEFT in viewer screen space, so DECREASING th (from -37° to -59.5°) goes RIGHT!
        // Therefore u_text = 0 at th = -37° (u=1) and u_text = 1 at th = -59.5° (u=0) => u_atlas = 1.0 - u!
        const rawU = uv.getX(i);
        const rawV = uv.getY(i);
        uv.setXY(i, 0.004 + (1.0 - rawU) * 0.992, vBot + rawV * (vTop - vBot));
      }
      uv.needsUpdate = true;
      const signMesh = new THREE.Mesh(signGeo, materials.silverLouver);
      group.add(signMesh);
    }

    // 8. Curved Metallic Copper-Bronze Roof Crown Parapet + 18 Radial Rooftop Solar PV Strips & MEP Penthouse (ref_111, ref_174)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.copperACP, thBody0, thBody1, rOut - 0.55, rOut + 0.42, cfg.roofSlabY + 0.18, cfg.crownTopY, 48,
      { bottom: false, uvMetersPerTile: 3.5 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.pavingBlock, thBody0, thBody1, rIn + 0.5, rOut - 0.55, cfg.roofSlabY - 0.25, cfg.roofSlabY + 0.15, 44,
      { bottom: false, outer: false, inner: false, uvMetersPerTile: 5.0 }
    );

    // 18 Radial Rooftop Solar PV Panel Strips along Western-to-Central Roof Arc (-140° .. -66°, ref_174)
    const pvGeo = ProceduralTextureFactory.createAtlasBox(11.6, 0.14, 1.95, 938, 652, 86, 64);
    for (let i = 0; i < 18; i++) {
      const rad = (-140.0 + i * 4.2) * DEG;
      const rMid = 0.5 * (rIn + rOut);
      const pv = new THREE.Mesh(pvGeo, materials.signAtlasMat);
      pv.position.set(rMid * Math.cos(rad), cfg.roofSlabY + 0.48, rMid * Math.sin(rad));
      pv.rotation.y = -rad;
      pv.rotation.z = 0.12;
      pv.castShadow = true;
      group.add(pv);
    }

    // Eastern Rooftop MEP / Chiller Penthouse Box (-58° .. -43°, ref_174)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, -58 * DEG, -43 * DEG, rIn + 4.5, rOut - 5.0, cfg.roofSlabY + 0.15, cfg.roofSlabY + 2.35, 10,
      { bottom: false, uvMetersPerTile: 3.5 }
    );

    // 9. North Forecourt: Sloped "STP-UI" Wedge Planter, Black/Yellow Directory Totem, & Exposed Red Fire-Hydrant Pipeline
    GedungSTPUIBuilder._buildNorthLandscapeProps(group, materials, cfg.thPortico);

    return group;
  }

  static _buildNorthPortico(parent, materials, thPortico, rOut, l1TopY) {
    const portGroup = new THREE.Group();
    const c = Math.cos(thPortico);
    const s = Math.sin(thPortico);
    portGroup.position.set(rOut * c, 0, rOut * s);
    portGroup.rotation.y = Math.atan2(c, s);
    parent.add(portGroup);

    const canW = 14.2;
    const canD = 5.4;
    const canY = l1TopY + 0.12;

    // Silver-white ACP flat-fascia canopy + warm underside soffit w/ recessed downlights (ref_164)
    const canopy = new THREE.Mesh(new THREE.BoxGeometry(canW, 0.64, canD), materials.architecturalWhite);
    canopy.position.set(0, canY, canD * 0.5 - 0.4);
    canopy.castShadow = true;
    canopy.receiveShadow = true;
    portGroup.add(canopy);

    const soffit = new THREE.Mesh(new THREE.BoxGeometry(canW - 0.5, 0.08, canD - 0.5), materials.warmTaupeSoffit);
    soffit.position.set(0, canY - 0.30, canD * 0.5 - 0.4);
    portGroup.add(soffit);

    for (const lx of [-4.2, -1.4, 1.4, 4.2]) {
      const lamp = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.06, 10), materials.goldAccent);
      lamp.position.set(lx, canY - 0.33, canD * 0.5 - 0.2);
      portGroup.add(lamp);
    }

    // Crisp Front Fascia Lettering: "SCIENCE TECHNO PARK UNIVERSITAS INDONESIA" (Tile 1 in Atlas!)
    const fasciaSign = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(canW - 0.6, 0.56, 0, 160, 1024, 112),
      materials.signAtlasMat
    );
    fasciaSign.position.set(0, canY, canD - 0.4 + 0.33);
    portGroup.add(fasciaSign);

    // Large White Cylindrical Piloti Columns + Angled V-Strut Supports at Outer Corners (ref_164)
    for (const sx of [-5.5, 5.5]) {
      const mainCyl = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.36, canY - 0.2, 14), materials.architecturalWhite);
      mainCyl.position.set(sx * 0.72, 0.15 + (canY - 0.2) * 0.5, canD - 1.6);
      mainCyl.castShadow = true;
      portGroup.add(mainCyl);

      for (const tiltZ of [-0.18, 0.18]) {
        const vLeg = new THREE.Mesh(new THREE.BoxGeometry(0.32, canY - 0.2, 0.32), materials.architecturalWhite);
        vLeg.position.set(sx, 0.15 + (canY - 0.2) * 0.5, canD - 1.2 + tiltZ * 1.5);
        vLeg.rotation.x = -tiltZ;
        vLeg.castShadow = true;
        portGroup.add(vLeg);
      }
    }

    // L1 North Main Lobby Double Glass Entrance Doors, Dark Anodized Frame, & 4 Stainless Bollards (ref_164!)
    const lobbyDoorFrame = new THREE.Mesh(new THREE.BoxGeometry(4.2, 3.25, 0.28), materials.darkMetal);
    lobbyDoorFrame.position.set(0, 1.78, -0.32);
    portGroup.add(lobbyDoorFrame);
    const lobbyDoorGlass = new THREE.Mesh(new THREE.BoxGeometry(3.85, 3.02, 0.32), materials.windowGlass);
    lobbyDoorGlass.position.set(0, 1.78, -0.30);
    portGroup.add(lobbyDoorGlass);
    for (const bx of [-4.2, -1.8, 1.8, 4.2]) {
      const bollard = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.08, 0.82, 10), materials.silverLouver);
      bollard.position.set(bx, 0.52, canD - 0.55);
      portGroup.add(bollard);
    }
  }

  static _buildNorthLandscapeProps(parent, materials, thPortico) {
    // A. Sloped White Concrete Wedge Planter w/ 3D Raised Rim Walls, "STP-UI" Topiary Mound & Pedestrian Ramp Bridge (ref_164!)
    const thPlanter = thPortico - 3.5 * DEG; // Slightly left/west of portico center as seen in ref_164!
    const c = Math.cos(thPlanter);
    const s = Math.sin(thPlanter);
    const yaw = Math.atan2(c, s);

    const planterGroup = new THREE.Group();
    planterGroup.position.set(65.6 * c, 0.15, 65.6 * s);
    planterGroup.rotation.y = yaw;
    parent.add(planterGroup);

    // White concrete sloped retaining walls + 3D raised side/front/back rim curbs framing the "STP-UI" hedge mound (ref_164)
    const wedgeBase = new THREE.Mesh(new THREE.BoxGeometry(7.6, 0.62, 3.2), materials.architecturalWhite);
    wedgeBase.position.set(0, 0.31, 0);
    wedgeBase.castShadow = true;
    planterGroup.add(wedgeBase);
    for (const sx of [-3.68, 3.68]) {
      const sideRim = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.42, 3.2), materials.architecturalWhite);
      sideRim.position.set(sx, 0.68, 0);
      sideRim.rotation.x = 0.20;
      planterGroup.add(sideRim);
    }

    const mound = new THREE.Mesh(new THREE.BoxGeometry(6.8, 0.54, 2.6), materials.lushGrass);
    mound.position.set(0, 0.56, 0.08);
    mound.rotation.x = 0.22;
    planterGroup.add(mound);

    const turfFace = new THREE.Mesh(
      ProceduralTextureFactory.createAtlasSignPlane(5.6, 1.45, 852, 528, 172, 124),
      materials.signAtlasMat
    );
    turfFace.position.set(0, 0.88, 0.32);
    turfFace.rotation.x = -Math.PI * 0.35;
    planterGroup.add(turfFace);

    // Sloped White Concrete Walkway & Black Steel Grate Bridge right next to the planter (ref_164 foreground!)
    const walkRamp = new THREE.Mesh(new THREE.BoxGeometry(1.85, 0.22, 3.4), materials.pavingBlock);
    walkRamp.position.set(-4.6, 0.22, 1.2);
    walkRamp.rotation.y = -0.35;
    planterGroup.add(walkRamp);

    const steelBridge = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.14, 1.8), materials.darkMetal);
    steelBridge.position.set(-5.2, 0.18, 2.8);
    planterGroup.add(steelBridge);

    // B. Tall Matte-Black & Yellow STP Directory Totem just East of Portico (th = -75.5°, r = 67.2m, ref_105, ref_156)
    const thTot = -75.5 * DEG;
    const ct = Math.cos(thTot), st = Math.sin(thTot);
    const totGroup = new THREE.Group();
    totGroup.position.set(67.2 * ct, 0.15, 67.2 * st);
    totGroup.rotation.y = Math.atan2(ct, st) + 0.35;
    parent.add(totGroup);

    const totPed = new THREE.Mesh(new THREE.BoxGeometry(1.45, 0.42, 0.65), materials.pavingBlock);
    totPed.position.set(0, 0.21, 0);
    totGroup.add(totPed);

    const totBody = new THREE.Mesh(new THREE.BoxGeometry(1.26, 2.85, 0.32), materials.darkMetal);
    totBody.position.set(0, 0.42 + 2.85 * 0.5, 0);
    totBody.castShadow = true;
    totGroup.add(totBody);

    for (const zSide of [-0.17, 0.17]) {
      const face = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasSignPlane(1.20, 2.75, 0, 528, 340, 496),
        materials.signAtlasMat
      );
      face.position.set(0, 0.42 + 2.85 * 0.5, zSide);
      if (zSide < 0) face.rotation.y = Math.PI;
      totGroup.add(face);
    }

    // C. Exposed Red-Oxide Fire-Hydrant Pipeline & Twin Pillar Hydrants along Inner Ditch Bank (ref_096, ref_157)
    NexusSSoSTKernel.addCurvedPolarBand(
      parent, materials.redHydrant, -134 * DEG, -74 * DEG, 68.05, 68.21, 0.26, 0.40, 28,
      { bottom: false, uvMetersPerTile: 3.0 }
    );
    for (const hDeg of [-132, -76.5]) {
      const hr = hDeg * DEG;
      const hyd = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.16, 0.92, 10), materials.redHydrant);
      hyd.position.set(67.95 * Math.cos(hr), 0.58, 67.95 * Math.sin(hr));
      hyd.castShadow = true;
      parent.add(hyd);
    }
  }
}

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./library/ilrc-drpm/index.js
 *
 * Class-A Procedural Building Module: Gedung ILRC & DRPM UI (Integrated Laboratory and Research Center)
 * Ground-Truth References: ref_002-084, ref_107-126, ref_154-155, ref_174, ref_206.
 *
 * Unified 4-Zone Manifold Crescent Architecture (th = -31° ENE to +83° SSE):
 * 1. East Curved Horizontal-Louvered Wing (-31°..+36°): Tall L1 white base, East drop-off portico,
 *    L4-L5 cantilevered 3-sided glass observation box (+8.5°), DRPM L1 open breezeway (+27°..+35°),
 *    "P PARKIR KHUSUS PIMPINAN" open iron gates, & grey Gardu utility substation at -32°.
 * 2. 5-Story Open-Air SE Atrium & Main Portal (+36°..+46°): Smooth burnt-terracotta flank walls (L2-L5),
 *    3 white corridor bridges (L3-L5), continuous curved white roof cornice w/ warm taupe-terracotta soffit,
 *    L1 white portal ("INTEGRATED LABORATORY AND RESEARCH CENTER / UNIVERSITAS INDONESIA" + Makara + KAN),
 *    and axial brown Koral-Sikat pebble-grid ramp w/ open black picket gates & blue ADA bay.
 * 3. South Curved Wing & Bamboo-Staked Dry-Leaf Mound (+46°..+75°): L1 white piers + Makara banners,
 *    L2 dark-mullioned glass curtain ribbon, L3-L5 horizontal silver louvers, & 1.65m brown dry-leaf mound
 *    retained by vertical bamboo stakes along the South ring road (zero magenta lens-flare bias!).
 * 4. 6-Story Terracotta / Burnt-Brick South End Tower (+75°..+83°, H = 25.2m + 27.2m Penthouse):
 *    Angled cantilevered SE sunshade slab fins (ref_084) + 3 stacked protruding hooded window boxes (ref_076).
 */
class GedungILRCBuilder {
  static build(materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.ilrc;
    const group = new THREE.Group();
    group.name = 'Gedung_ILRC_DRPM_UI';

    const rOut = cfg.rOuter;          // 58.0m
    const rInE = cfg.rInnerEast;      // 37.5m (wider northern/eastern ILRC wing, ref_174)
    const rInS = cfg.rInnerSouth;     // 41.5m (stepped-in southeastern DRPM/South wing, ref_174)
    const corniceY = cfg.corniceTopY; // 21.20m

    const thStart = cfg.thStart;           // -31°
    const thEastEnd = cfg.thEastEnd;       // +36°
    const thAtriumEnd = cfg.thAtriumEnd;   // +46°
    const thSouthEnd = cfg.thSouthEnd;     // +75°
    const thTowerEnd = cfg.thTowerEnd;     // +83°

    // =========================================================================
    // 1. CONTINUOUS CURVED WHITE ROOF CORNICE BRIDGING ACROSS THE 5-STORY ATRIUM
    //    (Sweeps unbroken from -31° to +75°, with warm taupe-terracotta soffit over Atrium! ref_029, ref_030)
    // =========================================================================
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thStart, thSouthEnd, rOut - 1.2, rOut + 0.85, corniceY - 0.82, corniceY, 56,
      { uvMetersPerTile: 3.8 }
    );
    // Warm pinkish-taupe-terracotta underside soffit bridging across the Open Atrium (+36°..+46°, ref_029)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.warmTaupeSoffit, thEastEnd, thAtriumEnd, rInS + 6.5, rOut + 0.65, corniceY - 0.92, corniceY - 0.78, 10,
      { top: false, bottom: true, uvMetersPerTile: 3.5 }
    );

    // Flat Roof Decks on East Wing (-31°..+36°) & South Wing (+46°..+75°)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.pavingBlock, thStart, thEastEnd, rInE, rOut - 1.0, corniceY - 0.65, corniceY - 0.15, 32,
      { bottom: false, uvMetersPerTile: 5.0 }
    );
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.pavingBlock, thAtriumEnd, thSouthEnd, rInS, rOut - 1.0, corniceY - 0.65, corniceY - 0.15, 18,
      { bottom: false, uvMetersPerTile: 5.0 }
    );

    // =========================================================================
    // 2. ZONE 1: EAST CURVED HORIZONTAL-LOUVERED WING & DRPM BREEZEWAY (-31° .. +36°)
    // =========================================================================
    // Inner courtyard wall & windows (East wing)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thStart, thEastEnd, rInE, rInE + 1.2, 0.15, corniceY - 0.6, 30,
      { bottom: false, uvMetersPerTile: 4.0 }
    );

    // L1 White Ground Floor (-31° .. +26.5° solid/lobby, +26.5° .. +36° open DRPM breezeway! ref_012, ref_018)
    const thBreezeway0 = 26.5 * DEG;
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thStart, thBreezeway0, rInE + 1.0, rOut - 0.75, 0.15, 5.15, 26,
      { bottom: false, uvMetersPerTile: 3.8 }
    );
    // 8-Pane Dark Ribbon Window north of DRPM breezeway (+16°..+25°, ref_012, ref_018)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.windowGlass, 16.0 * DEG, 25.2 * DEG, rOut - 0.82, rOut - 0.62, 1.35, 3.35, 8,
      { top: false, bottom: false, uvMetersPerTile: 2.2 }
    );
    // Tall Cylindrical White Columns at East Lobby (+6.5°) & DRPM Breezeway (+27.2°, +35.2°, ref_002, ref_012)
    for (const colDeg of [6.2, 11.2, 27.2, 31.2, 35.2]) {
      const cr = colDeg * DEG;
      const col = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 5.05, 14), materials.architecturalWhite);
      col.position.set((rOut - 0.65) * Math.cos(cr), 0.15 + 2.525, (rOut - 0.65) * Math.sin(cr));
      col.castShadow = true;
      group.add(col);
    }

    // East Vehicular Drop-Off Portico Canopy (+5.2° .. +11.8°, projecting to r = 63.4m, ref_002, ref_009)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, 5.2 * DEG, 11.8 * DEG, rOut - 0.7, 63.4, 4.45, 4.92, 6,
      { uvMetersPerTile: 3.5 }
    );
    for (const pDeg of [5.8, 11.2]) {
      const pr = pDeg * DEG;
      const pCol = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 4.45, 12), materials.architecturalWhite);
      pCol.position.set(62.8 * Math.cos(pr), 2.30, 62.8 * Math.sin(pr));
      pCol.castShadow = true;
      group.add(pCol);
    }

    // L2, L3, & Double-Height L4-L5 Horizontal Silver Aluminum Sun-Louvers (-31° .. +36°, ref_009, ref_019, ref_111)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.silverLouver, thStart + 0.01, thEastEnd - 0.01, rOut - 0.55, rOut - 0.08, 5.15, corniceY - 0.80, 34,
      { bottom: false, top: false, uvMetersPerTile: 2.8 }
    );
    // True-3D Projecting Horizontal Aluminum Sun-Louver Blades across East Wing L2-L5 (physical 0.22m depth!)
    for (let ly = 5.85; ly <= corniceY - 1.35; ly += 1.05) {
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.silverLouver, thStart + 0.01, thEastEnd - 0.01, rOut - 0.10, rOut + 0.14, ly - 0.035, ly + 0.035, 34,
        { bottom: true, top: true, uvMetersPerTile: 3.0 }
      );
    }
    // Projecting Curved White Concrete Floor-Slab Fascias (L2 base 5.15m full-depth ceiling soffit, L3 9.35m, L4 13.55m)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thStart, thEastEnd, rInE, rOut + 0.45, 5.15 - 0.20, 5.15 + 0.20, 36,
      { bottom: true, top: true, uvMetersPerTile: 3.8 }
    );
    for (const ySlab of [9.35, 13.55]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, thStart, thEastEnd, rOut - 0.8, rOut + 0.45, ySlab - 0.20, ySlab + 0.20, 36,
        { uvMetersPerTile: 3.8 }
      );
    }
    // Vertical White Structural Mullions dividing East Louvers into clean bays
    for (let deg = -28; deg <= 34; deg += 4.5) {
      const mr = deg * DEG;
      const mul = new THREE.Mesh(new THREE.BoxGeometry(0.16, corniceY - 5.95, 0.34), materials.architecturalWhite);
      mul.position.set((rOut + 0.05) * Math.cos(mr), 5.15 + (corniceY - 5.95) * 0.5, (rOut + 0.05) * Math.sin(mr));
      mul.rotation.y = -mr;
      group.add(mul);
    }

    // Cantilevered 3-Sided White-Framed Upper Glass Observation Box at L4–L5 (+6.2°..+10.8°, ref_009, ref_010)
    GedungILRCBuilder._buildCantileveredGlassBox(group, materials, 8.5 * DEG, rOut);

    // East Wing Roundabout Traffic Sign & Twin-Arm V-Post Garden Lamp (+9.2°..+11.5°, ref_009)
    {
      const sr = 11.5 * DEG;
      const signPost = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.048, 1.88, 8), materials.darkMetal);
      signPost.position.set(68.8 * Math.cos(sr), 1.02, 68.8 * Math.sin(sr));
      group.add(signPost);

      const signBoard = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasBox(0.72, 0.72, 0.08, 938, 716, 86, 60),
        materials.signAtlasMat
      );
      signBoard.position.set(68.8 * Math.cos(sr), 2.28, 68.8 * Math.sin(sr));
      signBoard.rotation.y = -sr + Math.PI * 0.35;
      group.add(signBoard);

      // Twin-arm V-post garden lamp near East Portico lawn (+9.2°, r = 64.2m, ref_009)
      const lr = 9.2 * DEG;
      const lx = 64.2 * Math.cos(lr);
      const lz = 64.2 * Math.sin(lr);
      const vPost = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 3.4, 8), materials.darkMetal);
      vPost.position.set(lx, 1.8, lz);
      group.add(vPost);
      for (const sgn of [-1, 1]) {
        const vArm = new THREE.Mesh(new THREE.BoxGeometry(0.85, 0.06, 0.06), materials.darkMetal);
        vArm.position.set(lx - Math.sin(lr) * sgn * 0.36, 3.65, lz + Math.cos(lr) * sgn * 0.36);
        vArm.rotation.y = -lr;
        vArm.rotation.z = sgn * 0.42;
        group.add(vArm);
        const vHead = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.10, 0.22), materials.architecturalWhite);
        vHead.position.set(lx - Math.sin(lr) * sgn * 0.72, 3.82, lz + Math.cos(lr) * sgn * 0.72);
        group.add(vHead);
      }
    }

    // Grey Masonry Gardu Utility Box at Mouth of STP–ILRC Seismic Gap (-32.4°, r = 60.4m, ref_116)
    {
      const gr = -32.4 * DEG;
      const gardu = new THREE.Mesh(new THREE.BoxGeometry(3.2, 2.85, 2.4), materials.pavingBlock);
      gardu.position.set(60.4 * Math.cos(gr), 1.50, 60.4 * Math.sin(gr));
      gardu.rotation.y = -gr;
      gardu.castShadow = true;
      group.add(gardu);
    }

    // SE DRPM Vehicular Gate + "P PARKIR KHUSUS PIMPINAN" Sign (+26.5°..+32.5°, ref_012, ref_015)
    GedungILRCBuilder._buildOpenPicketGate(group, materials, 29.5 * DEG, 67.5, 5.6, true);

    // =========================================================================
    // 3. ZONE 2: 5-STORY OPEN-AIR SE ATRIUM & MAIN ILRC PORTAL (+36° .. +46°, ref_023-030)
    // =========================================================================
    // Two Smooth Burnt-Terracotta / Red-Brick Flank Walls (NE Flank at +36°..+37.2°, SW Flank at +44.8°..+46°)
    for (const [f0, f1] of [[thEastEnd - 0.4 * DEG, thEastEnd + 1.1 * DEG], [thAtriumEnd - 1.1 * DEG, thAtriumEnd + 0.4 * DEG]]) {
      // White L1 supporting columns under terracotta flank walls (ref_024, ref_032)
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, f0, f1, rInS + 2.0, rOut - 0.3, 0.15, 4.85, 2,
        { bottom: false, uvMetersPerTile: 3.5 }
      );
      // Burnt-terracotta flank walls from L2 up to the curved roof soffit (y = 4.85 .. 20.45m)
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.terracottaBrick, f0, f1, rInS + 2.0, rOut - 0.15, 4.85, corniceY - 0.75, 2,
        { bottom: false, uvMetersPerTile: 3.0 }
      );
    }

    // Recessed Curved Inner Atrium Rear Gallery Wall (L2–L5 above the L1 ground portal opening, ref_024, ref_029)
    // Prevents seeing ITTC's red roof through the upper corridor bridges!
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thEastEnd - 0.3 * DEG, thAtriumEnd + 0.3 * DEG, rInS + 1.8, rInS + 6.8, 4.85, corniceY - 0.75, 10,
      { bottom: true, uvMetersPerTile: 3.8 }
    );
    for (const gy of [6.15, 10.15, 14.15, 18.05]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.windowGlass, thEastEnd + 0.4 * DEG, thAtriumEnd - 0.4 * DEG, rInS + 6.6, rInS + 7.05, gy, gy + 2.25, 8,
        { top: false, bottom: false, uvMetersPerTile: 2.5 }
      );
    }
    // Atrium ground floor paved lobby slab (r = rInS .. rOut, th = 36°..46°)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.pavingBlock, thEastEnd - 0.5 * DEG, thAtriumEnd + 0.5 * DEG, rInS, rOut - 0.2, 0.12, 0.22, 8,
      { bottom: false, uvMetersPerTile: 3.0 }
    );

    // L1 Main White Horizontal Portal Beam + Exact ILRC Signage (Makara + KAN + 2-Line Title, ref_024, ref_029)
    const thAtriumMid = 0.5 * (thEastEnd + thAtriumEnd); // +41.0°
    {
      const ca = Math.cos(thAtriumMid), sa = Math.sin(thAtriumMid);
      const portalGroup = new THREE.Group();
      portalGroup.position.set((rOut - 0.85) * ca, 0, (rOut - 0.85) * sa);
      portalGroup.rotation.y = Math.atan2(ca, sa);
      group.add(portalGroup);

      const spanW = 8.4;
      // Left & right white portal piers
      for (const px of [-3.85, 3.85]) {
        const pier = new THREE.Mesh(new THREE.BoxGeometry(0.72, 4.95, 0.95), materials.architecturalWhite);
        pier.position.set(px, 0.15 + 4.95 * 0.5, 0);
        pier.castShadow = true;
        portalGroup.add(pier);
      }
      // White portal lintel fascia beam (y = 3.75 .. 5.05m)
      const beam = new THREE.Mesh(new THREE.BoxGeometry(spanW, 1.30, 1.05), materials.architecturalWhite);
      beam.position.set(0, 4.40, 0);
      beam.castShadow = true;
      portalGroup.add(beam);

      // Crisp ILRC Portal Signage Plane (Tile 2 in Unified Signage Atlas!)
      const ilrcSign = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasSignPlane(spanW - 0.4, 1.18, 0, 272, 1024, 144),
        materials.signAtlasMat
      );
      ilrcSign.position.set(0, 4.40, 0.54);
      portalGroup.add(ilrcSign);

      // 3 Open-Air White Horizontal Corridor Bridges (L3, L4, L5) w/ 3D Tubular Handrails & Vertical Balusters (ref_024, ref_029)
      for (const by of [9.35, 13.35, 17.35]) {
        const brParapet = new THREE.Mesh(new THREE.BoxGeometry(spanW + 0.2, 1.12, 2.2), materials.architecturalWhite);
        brParapet.position.set(0, by, -2.1);
        brParapet.castShadow = true;
        brParapet.receiveShadow = true;
        portalGroup.add(brParapet);

        for (const ry of [by + 0.84, by + 0.68]) {
          const rail = new THREE.Mesh(new THREE.BoxGeometry(spanW + 0.1, 0.055, 0.055), materials.darkMetal);
          rail.position.set(0, ry, -1.08);
          portalGroup.add(rail);
        }
        for (let bx = -3.6; bx <= 3.6; bx += 1.2) {
          const post = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.36, 0.05), materials.darkMetal);
          post.position.set(bx, by + 0.70, -1.08);
          portalGroup.add(post);
        }
      }

      // Axial Brown Koral-Sikat Pebble-Grid Approach Ramp + Blue ADA Bay + Bollards (z = 0.5 .. 11.2m -> r = 57.5..68.4m)
      const koralRamp = new THREE.Mesh(new THREE.BoxGeometry(6.2, 0.16, 11.2), materials.koralSikat);
      koralRamp.position.set(0, 0.16, 6.0);
      koralRamp.receiveShadow = true;
      portalGroup.add(koralRamp);

      // Sky-Blue ADA Wheelchair Parking Stall w/ White Diagonal Hatching (Tile 6C, ref_029)
      const adaBay = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasBox(3.4, 0.05, 2.6, 852, 652, 86, 124),
        materials.signAtlasMat
      );
      adaBay.position.set(0, 0.25, 2.8);
      adaBay.receiveShadow = true;
      portalGroup.add(adaBay);

      for (const bx of [-3.25, 3.25]) {
        const curb = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.28, 11.2), materials.architecturalWhite);
        curb.position.set(bx, 0.22, 6.0);
        portalGroup.add(curb);

        const bollard = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.85, 8), materials.architecturalWhite);
        bollard.position.set(bx * 0.88, 0.55, 3.2);
        portalGroup.add(bollard);

        // Low black iron picket railing along both sides of the Koral-Sikat ramp bridge (ref_024, ref_029)
        const sideRailTop = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 7.6), materials.darkMetal);
        sideRailTop.position.set(bx, 0.92, 6.4);
        portalGroup.add(sideRailTop);
        const sideRailBot = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 7.6), materials.darkMetal);
        sideRailBot.position.set(bx, 0.42, 6.4);
        portalGroup.add(sideRailBot);
        for (let rz = 2.8; rz <= 10.0; rz += 0.9) {
          const rPost = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.68, 0.04), materials.darkMetal);
          rPost.position.set(bx, 0.62, rz);
          portalGroup.add(rPost);
        }
      }
    }
    // Open Black Picket Iron Gate at Entrance of Koral-Sikat Ramp (ref_023-025)
    GedungILRCBuilder._buildOpenPicketGate(group, materials, thAtriumMid, 67.4, 5.8, false);

    // =========================================================================
    // 4. ZONE 3: SOUTH CURVED WING & BAMBOO-STAKED DRY-LEAF MOUND (+46° .. +75°)
    // =========================================================================
    // L1 White Ground Base + Square Piers + 2 Navy/Gold UI Makara Vertical Banners (ref_033, ref_053)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thAtriumEnd, thSouthEnd, rInS, rOut - 1.1, 0.15, 4.95, 20,
      { bottom: false, uvMetersPerTile: 3.8 }
    );
    for (let deg = 48; deg <= 73; deg += 5.0) {
      const pr = deg * DEG;
      const pier = new THREE.Mesh(new THREE.BoxGeometry(0.45, 4.85, 0.55), materials.architecturalWhite);
      pier.position.set((rOut - 0.65) * Math.cos(pr), 0.15 + 2.425, (rOut - 0.65) * Math.sin(pr));
      pier.rotation.y = -pr;
      pier.castShadow = true;
      group.add(pier);
    }
    for (const banDeg of [53.5, 61.5]) {
      const br = banDeg * DEG;
      const ban = new THREE.Mesh(
        ProceduralTextureFactory.createAtlasSignPlane(0.95, 2.15, 864, 776, 96, 248),
        materials.signAtlasMat
      );
      NexusSSoSTKernel.placeOnPolarCylinder(ban, rOut - 0.95, br, 2.55, 0.05, true);
      group.add(ban);
    }

    // L2 Full-Height Dark-Mullioned Glass Curtain Band (+46° .. +75°, y = 4.95 .. 9.25m, ref_033, ref_053)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.windowGlass, thAtriumEnd, thSouthEnd, rInS + 0.5, rOut - 0.35, 4.95, 9.25, 20,
      { bottom: false, uvMetersPerTile: 3.0 }
    );

    // L3–L5 Horizontal Silver Louvers + 3D Projecting Louver Blades + Curved White Floor Bands (y = 9.25 .. 20.45m, ref_028, ref_049)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.silverLouver, thAtriumEnd, thSouthEnd, rInS + 0.5, rOut - 0.15, 9.25, corniceY - 0.80, 20,
      { bottom: false, top: false, uvMetersPerTile: 2.8 }
    );
    for (let ly = 9.95; ly <= corniceY - 1.35; ly += 1.05) {
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.silverLouver, thAtriumEnd + 0.01, thSouthEnd - 0.01, rOut - 0.16, rOut + 0.12, ly - 0.035, ly + 0.035, 20,
        { bottom: true, top: true, uvMetersPerTile: 3.0 }
      );
    }
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thAtriumEnd, thSouthEnd, rInS, rOut + 0.42, 4.95 - 0.18, 4.95 + 0.18, 22,
      { bottom: true, top: true, uvMetersPerTile: 3.8 }
    );
    for (const ySlab of [9.25, 13.45, 17.05]) {
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, thAtriumEnd, thSouthEnd, rOut - 0.8, rOut + 0.42, ySlab - 0.18, ySlab + 0.18, 22,
        { uvMetersPerTile: 3.8 }
      );
    }

    // South Dry-Brown Leaf-Litter Mound + Vertical Bamboo Stake Palisade (+45.5°..+76°, r = 61.2..68.15m, ref_028, ref_053)
    {
      const m0 = 45.5 * DEG;
      const m1 = 76.0 * DEG;
      const rCrest = 64.5;
      const rToe = 68.15;
      // Front slope rising from bamboo palisade (y = 0.45m) up to mound crest (y = 1.68m)
      const frontMound = new THREE.Mesh(
        NexusSSoSTKernel.createPolarSlopedRibbon(m0, m1, rCrest, 1.68, rToe, 0.45, 24, 3.0),
        materials.dryLeafSoil
      );
      frontMound.receiveShadow = true;
      group.add(frontMound);

      // Back slope dropping gently from crest (y = 1.68m) to L1 base (r = 60.5m, y = 0.25m)
      const backMound = new THREE.Mesh(
        NexusSSoSTKernel.createPolarSlopedRibbon(m0, m1, 60.5, 0.25, rCrest, 1.68, 24, 3.0),
        materials.dryLeafSoil
      );
      backMound.receiveShadow = true;
      group.add(backMound);

      // Tightly packed Bamboo-Stake Retaining Palisade along r = 68.15m (y = 0.10 .. 0.56m, ref_024-028, ref_053)
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.bambooWood, m0, m1, rToe - 0.14, rToe + 0.04, 0.10, 0.54, 36,
        { bottom: false, uvMetersPerTile: 1.2 }
      );
      for (let deg = 45.8; deg <= 75.8; deg += 0.9) {
        const sr = deg * DEG;
        const stake = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.05, 0.52, 5), materials.bambooWood);
        stake.position.set((rToe + 0.03) * Math.cos(sr), 0.34, (rToe + 0.03) * Math.sin(sr));
        group.add(stake);
      }
    }

    // =========================================================================
    // 5. ZONE 4: 6-STORY TERRACOTTA / BURNT-BRICK SOUTH END TOWER (+75° .. +83°, H = 25.2m + 27.2m Penthouse)
    //    Includes Angled SE Sunshade Slab Fins (ref_084) & 3 Stacked Protruding Hooded Windows (ref_076-077)!
    // =========================================================================
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.terracottaBrick, thSouthEnd, thTowerEnd, rInS + 1.0, rOut + 0.25, 0.15, cfg.towerTopY, 8,
      { bottom: false, uvMetersPerTile: 3.2 }
    );
    // Stepped Elevator Overrun Penthouse Bulkhead on top (reaching y = 27.20m — matching global height envelope! ref_077, ref_084)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.terracottaBrick, thSouthEnd + 1.8 * DEG, thTowerEnd - 1.2 * DEG, rInS + 4.5, rOut - 3.2, cfg.towerTopY - 0.1, cfg.penthouseTopY, 4,
      { bottom: false, uvMetersPerTile: 3.0 }
    );
    // White parapet coping strip at top of Brick Tower (y = 25.05..25.25m, ref_076)
    NexusSSoSTKernel.addCurvedPolarBand(
      group, materials.architecturalWhite, thSouthEnd - 0.1 * DEG, thTowerEnd + 0.1 * DEG, rInS + 0.9, rOut + 0.35, cfg.towerTopY - 0.18, cfg.towerTopY + 0.05, 8,
      { uvMetersPerTile: 3.0 }
    );

    // Angled Cantilevered Grey/White Floor-Slab Sunshade Fins & Shadow Grooves on Outer SE Profile (ref_084, ref_076!)
    for (let fl = 1; fl <= 5; fl++) {
      const yf = 4.2 + fl * 3.8;
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.architecturalWhite, thSouthEnd, thTowerEnd - 0.5 * DEG, rOut + 0.1, rOut + 1.35, yf - 0.14, yf + 0.14, 6,
        { uvMetersPerTile: 3.0 }
      );
      // Recessed dark window slot bands between sunshade fins on SE outer face
      NexusSSoSTKernel.addCurvedPolarBand(
        group, materials.windowGlass, thSouthEnd + 0.8 * DEG, thTowerEnd - 1.2 * DEG, rOut + 0.12, rOut + 0.34, yf - 2.3, yf - 0.7, 4,
        { top: false, bottom: false, uvMetersPerTile: 2.2 }
      );
    }

    // 3 Vertically Stacked Protruding Dark-Grilled Hooded Window Boxes on Upper WNW Face (L4, L5, L6 at th = +83°, ref_076-077)
    {
      const cT = Math.cos(thTowerEnd);
      const sT = Math.sin(thTowerEnd);
      // Outward normal of the +83° end cap points toward increasing theta: (-sin th, 0, cos th)!
      const nx = -sT;
      const nz = cT;
      const faceYaw = Math.atan2(nx, nz);
      const rWin = 0.5 * (rInS + 2.0 + rOut - 1.0); // ~50.2m radial center

      for (const wy of [13.6, 17.5, 21.4]) {
        const boxGroup = new THREE.Group();
        boxGroup.position.set(rWin * cT + nx * 0.45, wy, rWin * sT + nz * 0.45);
        boxGroup.rotation.y = faceYaw;
        group.add(boxGroup);

        // Protruding dark window cage
        const cage = new THREE.Mesh(new THREE.BoxGeometry(2.6, 1.65, 0.85), materials.darkMetal);
        cage.castShadow = true;
        boxGroup.add(cage);

        const glass = new THREE.Mesh(new THREE.BoxGeometry(2.35, 1.42, 0.90), materials.windowGlass);
        boxGroup.add(glass);

        // Sloped Silver/White Metal Awning Hood on top (ref_076-077)
        const hood = new THREE.Mesh(new THREE.BoxGeometry(2.95, 0.14, 1.15), materials.silverLouver);
        hood.position.set(0, 0.96, 0.12);
        hood.rotation.x = 0.24;
        hood.castShadow = true;
        boxGroup.add(hood);
      }
    }

    return group;
  }

  static _buildCantileveredGlassBox(parent, materials, thCenter, rOut) {
    const c = Math.cos(thCenter), s = Math.sin(thCenter);
    const boxGroup = new THREE.Group();
    boxGroup.position.set((rOut + 0.55) * c, 17.25, (rOut + 0.55) * s);
    boxGroup.rotation.y = Math.atan2(c, s);
    parent.add(boxGroup);

    const w = 4.6, h = 6.9, d = 1.85;
    // Stepped white concrete corbel soffit slab underneath (ref_009, ref_010)
    const corbel = new THREE.Mesh(new THREE.BoxGeometry(w + 0.3, 0.45, d + 0.2), materials.architecturalWhite);
    corbel.position.set(0, -h * 0.5 - 0.15, 0);
    corbel.castShadow = true;
    boxGroup.add(corbel);

    // 3-Sided Glazed Observation Box + Overhanging Cap Slab
    const glassBody = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), materials.windowGlass);
    glassBody.castShadow = true;
    boxGroup.add(glassBody);

    const cap = new THREE.Mesh(new THREE.BoxGeometry(w + 0.45, 0.34, d + 0.45), materials.architecturalWhite);
    cap.position.set(0, h * 0.5 + 0.15, 0.05);
    cap.castShadow = true;
    boxGroup.add(cap);

    // Crisp white corner mullions & 2 horizontal transoms (3 glass tiers, ref_009)
    for (const sx of [-w * 0.5, w * 0.5]) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(0.18, h, 0.18), materials.architecturalWhite);
      m.position.set(sx, 0, d * 0.5);
      boxGroup.add(m);
    }
    for (const ty of [-h * 0.17, h * 0.17]) {
      const tr = new THREE.Mesh(new THREE.BoxGeometry(w + 0.08, 0.14, d + 0.08), materials.architecturalWhite);
      tr.position.set(0, ty, 0);
      boxGroup.add(tr);
    }
  }

  static _buildOpenPicketGate(parent, materials, thCenter, rGate, widthMeters, includeDrpmSign = false) {
    const c = Math.cos(thCenter), s = Math.sin(thCenter);
    const gateGroup = new THREE.Group();
    gateGroup.position.set(rGate * c, 0.15, rGate * s);
    gateGroup.rotation.y = Math.atan2(c, s);
    parent.add(gateGroup);

    const leafW = widthMeters * 0.44;
    const gateH = 1.65;

    // Two gate leaves swung open ~78° inward along left & right curbs (ref_012, ref_024)
    for (const sgn of [-1, 1]) {
      const leaf = new THREE.Group();
      leaf.position.set(sgn * (widthMeters * 0.5), 0, 0);
      leaf.rotation.y = sgn * 1.38; // swung open inward!
      gateGroup.add(leaf);

      const lx = -sgn * leafW * 0.5;
      for (const ry of [0.22, gateH - 0.08]) {
        const rail = new THREE.Mesh(new THREE.BoxGeometry(leafW, 0.06, 0.06), materials.darkMetal);
        rail.position.set(lx, ry, 0);
        leaf.add(rail);
      }
      for (let p = 0; p <= 8; p++) {
        const px = -sgn * (p / 8) * leafW;
        const picket = new THREE.Mesh(new THREE.BoxGeometry(0.035, gateH, 0.035), materials.darkMetal);
        picket.position.set(px, gateH * 0.5, 0);
        leaf.add(picket);
      }

      // Mount "P / PARKIR KHUSUS PIMPINAN" plaque on south/left leaf of DRPM gate (ref_012, Tile 6 left!)
      if (includeDrpmSign && sgn === 1) {
        const sign = new THREE.Mesh(
          ProceduralTextureFactory.createAtlasSignPlane(0.68, 0.96, 680, 528, 172, 248),
          materials.signAtlasMat
        );
        sign.position.set(lx, 0.98, 0.05);
        leaf.add(sign);

        const signBack = new THREE.Mesh(
          ProceduralTextureFactory.createAtlasSignPlane(0.68, 0.96, 680, 528, 172, 248),
          materials.signAtlasMat
        );
        signBack.position.set(lx, 0.98, -0.05);
        signBack.rotation.y = Math.PI;
        leaf.add(signBack);
      }
    }
  }
}

export class DPPDComplexBuilder {
  // Exact scale mapping Komplekita's rInnerCurb (71.50m) to Pulau U's inner ring-road curb (52.88m)
  static KOMPLEKITA_SCALE = 52.88 / 71.50;
  // +0.5° yaw adjustment so Komplekita's thAxis (+135.5°) aligns to +135.0° (-pi/4) in dppdGroup
  static KOMPLEKITA_YAW_OFFSET = 0.5 * DEG;

  static _sliceSubdomain(evaluateFn, uMin, uMax, vMin, vMax, uSegs = 24, vSegs = 12) {
    return NexusSSoSTKernel.sliceSubdomainFromMaster(evaluateFn, uMin, uMax, vMin, vMax, uSegs, vSegs);
  }

  static _createPalette() {
    const mk = (hex, kind = 'H', uvSentinel = null) => {
      const c = new THREE.Color(hex);
      const m = new THREE.MeshBasicMaterial({ color: c });
      m.userData.batchKind = kind;
      m.userData.rgb = [c.r, c.g, c.b];
      m.userData.uvSentinel = uvSentinel;
      return m;
    };
    const mkLin = (rgb, kind = 'H', uvSentinel = null) => {
      const m = new THREE.MeshBasicMaterial({ color: 0xffffff });
      m.userData.batchKind = kind;
      m.userData.rgb = rgb;
      m.userData.uvSentinel = uvSentinel;
      return m;
    };

    return {
      // Pulau U Ring Road & Spur Collar Linear RGBs
      asphaltCrown: mkLin([0.195, 0.205, 0.225], 'H', [0.0, -420.0]),
      asphaltGutter: mkLin([0.155, 0.165, 0.185], 'H', [0.0, -419.15]),
      uiYellowLine: mkLin([0.86, 0.64, 0.10]),
      whiteEdgeLine: mkLin([0.89, 0.90, 0.92]),
      concreteCurb: mkLin([0.88, 0.86, 0.82]),
      campusGrass: mkLin([0.148, 0.312, 0.080]),
      sidewalkPaver: mkLin([0.45, 0.44, 0.41]),
      courtyardGranite: mkLin([0.52, 0.51, 0.48]),
      curbStriped: mkLin([0.92, 0.93, 0.95], 'Z'),
      granitePlinth: mk(0x5e6166),
      darkIron: mk(0x1f232a),
      darkAnodized: mk(0x23272f),
      uiYellowFascia: mk(0xdca81c),
      teakBench: mk(0x825428),
      warmLuminaire: mk(0xfff1c4, 'H', [0.5, -890.0]),

      // Exact Komplekita Material Keys (ProceduralTextureFactory.createComplexMaterialBank)
      terracottaRoof: mk(0xa8422d),
      terracottaBrick: mk(0xa63e2b),
      salmonBrick: mk(0xbd4f3a),
      architecturalWhite: mk(0xeeede8),
      creamStucco: mk(0xebe2ce),
      copperACP: mk(0xb5693d),
      copperBronze: mk(0xb5693d),
      warmTaupeSoffit: mk(0xb5693d),
      silverLouver: mk(0xc4cbd4),
      windowGlass: mk(0x132232, 'H', [0.5, -720.0]),
      solarPanel: mk(0x162d54),
      solarPV: mkLin([1.0, 1.0, 1.0]),
      asphalt: mkLin([0.195, 0.205, 0.225], 'H', [0.0, -420.0]),
      asphaltRoad: mkLin([0.195, 0.205, 0.225], 'H', [0.0, -420.0]),
      plinthEarthSkirt: mkLin([0.195, 0.205, 0.225], 'H', [0.0, -420.0]),
      stripedCurb: mkLin([0.92, 0.93, 0.95], 'Z'),
      pavingBlock: mk(0x7c7972),
      terracottaPlaza: mk(0xa63e2b),
      stoneRetaining: mk(0x52504c),
      palmTrunk: mk(0x52504c),
      darkDitchBed: mk(0x25282c),
      dryLeafSoil: mk(0x584232),
      bambooWood: mk(0x7a623c),
      koralSikat: mk(0x735844),
      lushGrass: mkLin([0.158, 0.328, 0.086]),
      canopyFoliage: mkLin([1.0, 1.0, 1.0]),
      foliageCanopy: mkLin([1.0, 1.0, 1.0]),
      treeBark: mk(0x584232),
      goldAccent: mk(0xdca81c),
      yellowLine: mkLin([0.86, 0.64, 0.10]),
      yellowCurb: mkLin([0.86, 0.64, 0.10]),
      whiteLine: mkLin([0.89, 0.90, 0.92]),
      redHydrant: mk(0xb82824),
      blueParking: mk(0x1e56b8),
      adaBlueBay: mkLin([1.0, 1.0, 1.0]),
      darkMetal: mk(0x22262c),
      signAtlasMat: mkLin([1.0, 1.0, 1.0])
    };
  }

  static buildForCorridor() {
    const mats = DPPDComplexBuilder._createPalette();
    const root = new THREE.Group();
    root.position.set(
      TerrainDEMLoader.DPPD_CX,
      TerrainDEMLoader.DPPD_PLATEAU_Y,
      TerrainDEMLoader.DPPD_CZ
    );
    root.rotation.y = TerrainDEMLoader.DPPD_Y_ROT;
    root.scale.set(
      TerrainDEMLoader.DPPD_SCALE,
      TerrainDEMLoader.DPPD_SCALE,
      TerrainDEMLoader.DPPD_SCALE
    );

    const dppdGroup = new THREE.Group();
    const contextGroup = new THREE.Group();
    root.add(dppdGroup, contextGroup);

    // =========================================================================
    // 1. PULAU U PLATEAU LAWN, 4-STRIP CAMBERED RING ROAD (r = 53.20 .. 64.80),
    //    FLARED SPUR BELL-MOUTH APRONS (-4.6° & -158.3°), & 360° OUTER SIDEWALK
    // =========================================================================
    const campusLawnPad = new THREE.Mesh(
      new THREE.CylinderGeometry(52.90, 53.10, 0.80, 96),
      mats.campusGrass
    );
    campusLawnPad.position.y = -0.39;
    dppdGroup.add(campusLawnPad);

    const outerGrassSkirt = new THREE.Mesh(
      new THREE.CylinderGeometry(68.05, 71.80, 2.80, 96, 1, true),
      mats.campusGrass
    );
    outerGrassSkirt.position.y = -1.36;
    dppdGroup.add(outerGrassSkirt);

    const evalSpurMouthFactor = (rLocal, th, halfMouthM = 4.9, fadeM = 2.6) => {
      const wp = DPPDComplexBuilder.localToWorld(rLocal * Math.cos(th), 0.0, rLocal * Math.sin(th), 0.0);
      const dSouth = Math.hypot(wp.x - -292.1, wp.z - 336.8);
      const dEast = Math.hypot(wp.x - -218.3, wp.z - 338.4);
      const dMin = Math.min(dSouth, dEast);
      const t = Math.max(0.0, Math.min(1.0, (dMin - halfMouthM) / fadeM));
      return t * t * (3.0 - 2.0 * t);
    };

    // Inner curb-cut factor across Komplekita's 5 entry bridges/driveways:
    // West ITTC Gate (181.5°), STP Portico (-88.5°), ILRC Atrium (+41°), DRPM Gate (+62.8°), DPPD Portico (+135°)
    const innerBridgeThroats = [
      [181.0, 5.5],
      [-88.5, 7.0],
      [41.0, 7.5],
      [62.8, 4.5],
      [135.0, 10.5]
    ];
    const evalInnerMouthFactor = (th) => {
      const deg = ((th * 180.0) / Math.PI + 540.0) % 360.0 - 180.0;
      let minFactor = 1.0;
      for (const [cDeg, halfSpan] of innerBridgeThroats) {
        const d = Math.abs((((deg - cDeg) + 540.0) % 360.0) - 180.0);
        const t = Math.max(0.0, Math.min(1.0, (d - halfSpan) / 2.5));
        const f = t * t * (3.0 - 2.0 * t);
        if (f < minFactor) minFactor = f;
      }
      return minFactor;
    };

    const ringStrips = [
      { rA: 53.20, rB: 55.10, yA: 0.065, yB: 0.082, mat: mats.asphaltGutter },
      { rA: 55.10, rB: 59.00, yA: 0.082, yB: 0.092, mat: mats.asphaltCrown },
      { rA: 59.00, rB: 62.90, yA: 0.092, yB: 0.082, mat: mats.asphaltCrown },
      { rA: 62.90, rB: 64.80, yA: 0.082, yB: 0.065, mat: mats.asphaltGutter }
    ];
    for (const st of ringStrips) {
      const stripGeo = DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const th = u * Math.PI * 2.0;
          const r = st.rA + v * (st.rB - st.rA);
          const y = st.yA + v * (st.yB - st.yA);
          p.set(r * Math.cos(th), y, r * Math.sin(th));
          n.set(0, 1, 0);
        },
        0, 1, 0, 1, 128, 1
      );
      dppdGroup.add(new THREE.Mesh(stripGeo, st.mat));
    }

    for (const spurDeg of [-4.6, -158.3]) {
      const thC = (spurDeg * Math.PI) / 180.0;
      const halfSpan = (11.5 * Math.PI) / 180.0;
      const apronGeo = DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const flare = 1.0 + 0.38 * (1.0 - v);
          const th = thC + (u - 0.5) * 2.0 * halfSpan * flare;
          const r = 64.55 + v * (69.60 - 64.55);
          const edgeDrop = Math.abs(u - 0.5) > 0.36 ? -0.010 : 0.0;
          p.set(r * Math.cos(th), 0.072 + edgeDrop, r * Math.sin(th));
          n.set(0, 1, 0);
        },
        0, 1, 0, 1, 20, 4
      );
      dppdGroup.add(new THREE.Mesh(apronGeo, mats.asphaltCrown));

      for (let cw = -3; cw <= 3; cw++) {
        const thS = thC + cw * 0.024;
        const stripe = new THREE.Mesh(new THREE.BoxGeometry(2.15, 0.032, 0.56), mats.whiteEdgeLine);
        stripe.position.set(66.6 * Math.cos(thS), 0.082, 66.6 * Math.sin(thS));
        stripe.rotation.y = -thS;
        dppdGroup.add(stripe);
      }
    }

    const outerSidewalkGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = u * Math.PI * 2.0;
        const r = 65.16 + v * (68.10 - 65.16);
        const cf = evalSpurMouthFactor(r, th, 5.2, 2.4);
        const y = 0.055 + 0.135 * cf;
        p.set(r * Math.cos(th), y, r * Math.sin(th));
        n.set(0, 1, 0);
      },
      0, 1, 0, 1, 144, 1
    );
    dppdGroup.add(new THREE.Mesh(outerSidewalkGeo, mats.sidewalkPaver));

    const outerRimGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = u * Math.PI * 2.0;
        const r = 68.05 + v * 0.34;
        const cf = evalSpurMouthFactor(r, th, 5.4, 2.4);
        p.set(r * Math.cos(th), 0.052 + 0.155 * cf, r * Math.sin(th));
        n.set(0, 1, 0);
      },
      0, 1, 0, 1, 144, 1
    );
    dppdGroup.add(new THREE.Mesh(outerRimGeo, mats.concreteCurb));

    const buildRingZebraCurb = (rIn, rOut, yTop, mode) => {
      const segs = 144;
      const circum = Math.PI * (rIn + rOut) * TerrainDEMLoader.DPPD_SCALE;
      const zebraRepeats = Math.max(16, Math.round(circum * 0.26));
      const evalYTop = (th) => {
        let cf = 1.0;
        if (mode === 'OUTER') cf = evalSpurMouthFactor(0.5 * (rIn + rOut), th, 5.1, 2.3);
        else if (mode === 'INNER') cf = evalInnerMouthFactor(th);
        return 0.068 + (yTop - 0.068) * cf;
      };

      const geos = [
        DPPDComplexBuilder._sliceSubdomain(
          (u, v, p, n) => {
            const th = u * Math.PI * 2.0;
            const r = rIn + v * (rOut - rIn);
            p.set(r * Math.cos(th), evalYTop(th), r * Math.sin(th));
            n.set(0, 1, 0);
          },
          0, 1, 0, 1, segs, 1
        ),
        DPPDComplexBuilder._sliceSubdomain(
          (u, v, p, n) => {
            const th = u * Math.PI * 2.0;
            const yt = evalYTop(th);
            p.set(rIn * Math.cos(th), 0.06 + v * (yt - 0.06), rIn * Math.sin(th));
            n.set(-Math.cos(th), 0, -Math.sin(th));
          },
          0, 1, 0, 1, segs, 1
        ),
        DPPDComplexBuilder._sliceSubdomain(
          (u, v, p, n) => {
            const th = u * Math.PI * 2.0;
            const yt = evalYTop(th);
            p.set(rOut * Math.cos(th), -0.25 + v * (yt + 0.25), rOut * Math.sin(th));
            n.set(Math.cos(th), 0, Math.sin(th));
          },
          0, 1, 0, 1, segs, 1
        )
      ];

      for (const g of geos) {
        const uv = g.attributes.uv;
        for (let i = 0; i < uv.count; i++) {
          uv.array[i * uv.itemSize] = uv.getX(i) * zebraRepeats;
        }
        const m = new THREE.Mesh(g, mats.curbStriped);
        m.userData.keepCurbUV = true;
        dppdGroup.add(m);
      }
    };

    buildRingZebraCurb(52.84, 53.22, 0.22, 'INNER');
    buildRingZebraCurb(64.78, 65.16, 0.22, 'OUTER');

    for (const [rLine, mode] of [
      [53.52, 'INNER'],
      [64.30, 'OUTER']
    ]) {
      const edgeLineGeo = DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const th = u * Math.PI * 2.0;
          const cf =
            mode === 'OUTER'
              ? evalSpurMouthFactor(rLine, th, 5.0, 1.8)
              : evalInnerMouthFactor(th);
          const r = rLine + v * 0.18;
          p.set(r * Math.cos(th), cf > 0.45 ? 0.088 : 0.030, r * Math.sin(th));
          n.set(0, 1, 0);
        },
        0, 1, 0, 1, 144, 1
      );
      dppdGroup.add(new THREE.Mesh(edgeLineGeo, mats.whiteEdgeLine));
    }

    // Authentic Komplekita Zebra Crosswalk angles (147.5° West of DPPD portico, +41° ILRC Atrium, -88.5° STP Portico)
    const crossAnglesDeg = [147.5, 41.0, -88.5];
    for (let d = 0; d < 56; d++) {
      const deg = (d / 56) * 360.0 - 180.0;
      const nearCross = crossAnglesDeg.some(
        (cd) => Math.abs((((deg - cd) + 540.0) % 360.0) - 180.0) < 4.8
      );
      if (nearCross) continue;
      const th = (deg * Math.PI) / 180.0;
      const dash = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.032, 2.35), mats.uiYellowLine);
      dash.position.set(59.0 * Math.cos(th), 0.104, 59.0 * Math.sin(th));
      dash.rotation.y = -th;
      dppdGroup.add(dash);
    }

    for (const crossDeg of crossAnglesDeg) {
      const thC = (crossDeg * Math.PI) / 180.0;
      for (let st = -4; st <= 4; st++) {
        const rSt = 59.0 + st * 1.12;
        const stripe = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.034, 3.30), mats.whiteEdgeLine);
        stripe.position.set(rSt * Math.cos(thC), 0.105, rSt * Math.sin(thC));
        stripe.rotation.y = -thC;
        dppdGroup.add(stripe);
      }
    }

    // =========================================================================
    // 2. EXACT 1:1 KOMPLEKITA INNER ISLAND & 4 BUILDINGS TRANSPLANT
    //    (BundaranSiteBuilder + PerpustakaanLamaBuilder + GedungDPPDBuilder
    //     + GedungSTPUIBuilder + GedungILRCBuilder)
    // =========================================================================
    const komplekitaIslandGroup = new THREE.Group();
    komplekitaIslandGroup.name = 'Komplekita_Exact_Transplant_Root';
    komplekitaIslandGroup.scale.setScalar(DPPDComplexBuilder.KOMPLEKITA_SCALE);
    komplekitaIslandGroup.rotation.y = DPPDComplexBuilder.KOMPLEKITA_YAW_OFFSET;
    dppdGroup.add(komplekitaIslandGroup);

    const siteGroup = BundaranSiteBuilder.build(mats);
    const ittcGroup = PerpustakaanLamaBuilder.build(mats);
    const { group: dppdBuildingGroup } = GedungDPPDBuilder.build(mats);
    const stpBuildingGroup = GedungSTPUIBuilder.build(mats);
    const ilrcBuildingGroup = GedungILRCBuilder.build(mats);

    komplekitaIslandGroup.add(
      siteGroup,
      ittcGroup,
      dppdBuildingGroup,
      stpBuildingGroup,
      ilrcBuildingGroup
    );

    // =========================================================================
    // 3. HALTE BUS KUNING (BIKUN) "HALTE DPPD - ILRC - STP" AT th = -112° (r = 70.2)
    // =========================================================================
    {
      const thHalte = (-112.0 * Math.PI) / 180.0;
      const halte = new THREE.Group();
      halte.position.set(70.20 * Math.cos(thHalte), 0.18, 70.20 * Math.sin(thHalte));
      halte.rotation.y = -thHalte - Math.PI * 0.5;

      const pad = new THREE.Mesh(new THREE.BoxGeometry(7.6, 1.10, 4.2), mats.granitePlinth);
      pad.position.set(0, -0.46, 0.25);
      halte.add(pad);
      const deck = new THREE.Mesh(new THREE.BoxGeometry(7.4, 0.08, 4.0), mats.courtyardGranite);
      deck.position.set(0, 0.10, 0.25);
      halte.add(deck);

      for (const px of [-3.1, 0.0, 3.1]) {
        const post = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 3.35, 10), mats.darkIron);
        post.position.set(px, 1.72, -0.95);
        halte.add(post);
      }
      const backGlass = new THREE.Mesh(new THREE.BoxGeometry(6.1, 2.35, 0.10), mats.windowGlass);
      backGlass.position.set(0, 1.55, -0.95);
      halte.add(backGlass);

      const roof = new THREE.Mesh(new THREE.BoxGeometry(7.6, 0.22, 3.1), mats.darkAnodized);
      roof.position.set(0, 3.45, -0.15);
      roof.rotation.x = 0.06;
      halte.add(roof);

      const yellowFascia = new THREE.Mesh(new THREE.BoxGeometry(7.68, 0.42, 0.16), mats.uiYellowFascia);
      yellowFascia.position.set(0, 3.55, 1.38);
      halte.add(yellowFascia);

      const lightStrip = new THREE.Mesh(new THREE.BoxGeometry(5.8, 0.06, 0.28), mats.warmLuminaire);
      lightStrip.position.set(0, 3.32, 0.25);
      halte.add(lightStrip);

      const benchSeat = new THREE.Mesh(new THREE.BoxGeometry(4.6, 0.12, 0.58), mats.teakBench);
      benchSeat.position.set(0, 0.62, -0.45);
      halte.add(benchSeat);
      for (const lx of [-1.8, 0.0, 1.8]) {
        const bLeg = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.54, 0.50), mats.darkIron);
        bLeg.position.set(lx, 0.32, -0.45);
        halte.add(bLeg);
      }
      contextGroup.add(halte);
    }

    // =========================================================================
    // 4. BAKE ENTIRE HIERARCHY (MESH + INSTANCEDMESH + VERTEX COLORS)
    //    INTO { horizontalGeo, verticalGeo } (+0 EXTRA DRAW CALLS!)
    // =========================================================================
    root.updateMatrixWorld(true);

    const hPos = [];
    const hNorm = [];
    const hUv = [];
    const hCol = [];
    const hIdx = [];
    let hCount = 0;

    const vPos = [];
    const vNorm = [];
    const vUv = [];
    const vCol = [];
    const vIdx = [];
    let vCount = 0;

    const tmpV = new THREE.Vector3();
    const tmpN = new THREE.Vector3();
    const instMat = new THREE.Matrix4();
    const worldMat = new THREE.Matrix4();
    const instCol = new THREE.Color();

    const emitGeometryWithMatrix = (g, mat, matrixWorld, instanceTint = null, keepCurbUV = false) => {
      const isZebra = mat.userData.batchKind === 'Z';
      const baseRgb = mat.userData.rgb || [0.85, 0.85, 0.85];
      const uvSentinel = mat.userData.uvSentinel;

      const posAttr = g.attributes.position;
      if (!posAttr) return;
      if (!g.attributes.normal) g.computeVertexNormals();
      const normAttr = g.attributes.normal;
      const uvAttr = g.attributes.uv;
      const colAttr = g.attributes.color;
      const indexAttr = g.index;

      const base = isZebra ? vCount : hCount;
      const tPos = isZebra ? vPos : hPos;
      const tNorm = isZebra ? vNorm : hNorm;
      const tUv = isZebra ? vUv : hUv;
      const tCol = isZebra ? vCol : hCol;
      const tIdx = isZebra ? vIdx : hIdx;

      for (let i = 0; i < posAttr.count; i++) {
        tmpV.set(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i)).applyMatrix4(matrixWorld);
        tmpN.set(normAttr.getX(i), normAttr.getY(i), normAttr.getZ(i)).transformDirection(matrixWorld);

        tPos.push(tmpV.x, tmpV.y, tmpV.z);
        tNorm.push(tmpN.x, tmpN.y, tmpN.z);

        let cr = baseRgb[0];
        let cg = baseRgb[1];
        let cb = baseRgb[2];
        if (colAttr) {
          cr *= colAttr.getX(i);
          cg *= colAttr.getY(i);
          cb *= colAttr.getZ(i);
        }
        if (instanceTint) {
          cr *= instanceTint.r;
          cg *= instanceTint.g;
          cb *= instanceTint.b;
        }
        tCol.push(cr, cg, cb);

        if (uvSentinel) {
          tUv.push(uvSentinel[0], uvSentinel[1]);
        } else if (isZebra && keepCurbUV && uvAttr) {
          tUv.push(uvAttr.getX(i), uvAttr.getY(i));
        } else if (isZebra) {
          tUv.push(0.055, 0.5);
        } else if (uvAttr) {
          tUv.push(uvAttr.getX(i), Math.max(0.0, uvAttr.getY(i)));
        } else {
          tUv.push(0.5, 0.5);
        }
      }

      if (indexAttr) {
        for (let i = 0; i < indexAttr.count; i++) {
          tIdx.push(base + indexAttr.getX(i));
        }
      } else {
        for (let i = 0; i < posAttr.count; i++) {
          tIdx.push(base + i);
        }
      }

      if (isZebra) {
        vCount += posAttr.count;
      } else {
        hCount += posAttr.count;
      }
    };

    root.traverse((obj) => {
      if (!obj.isMesh || !obj.geometry || !obj.material) return;
      const keepCurbUV = Boolean(obj.userData.keepCurbUV);
      if (obj.isInstancedMesh) {
        for (let idx = 0; idx < obj.count; idx++) {
          obj.getMatrixAt(idx, instMat);
          worldMat.multiplyMatrices(obj.matrixWorld, instMat);
          let tint = null;
          if (obj.instanceColor) {
            obj.getColorAt(idx, instCol);
            tint = instCol;
          }
          emitGeometryWithMatrix(obj.geometry, obj.material, worldMat, tint, keepCurbUV);
        }
      } else {
        emitGeometryWithMatrix(obj.geometry, obj.material, obj.matrixWorld, null, keepCurbUV);
      }
    });

    const horizontalGeo = new THREE.BufferGeometry();
    horizontalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(hPos), 3));
    horizontalGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(hNorm), 3));
    horizontalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(hUv), 2));
    horizontalGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(hCol), 3));
    horizontalGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(hIdx), 1));

    const verticalGeo = new THREE.BufferGeometry();
    verticalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vPos), 3));
    verticalGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(vNorm), 3));
    verticalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(vUv), 2));
    verticalGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(vCol), 3));
    verticalGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(vIdx), 1));

    return { horizontalGeo, verticalGeo };
  }

  /**
   * Transforms local dppdGroup coordinates (lx, ly, lz) to Pulau U world coordinates.
   */
  static localToWorld(lx, ly, lz, extraRotY = 0.0) {
    const s = TerrainDEMLoader.DPPD_SCALE;
    const totalRot = TerrainDEMLoader.DPPD_Y_ROT + extraRotY;
    const c = Math.cos(totalRot);
    const sn = Math.sin(totalRot);
    return {
      x: TerrainDEMLoader.DPPD_CX + s * (lx * c + lz * sn),
      y: TerrainDEMLoader.DPPD_PLATEAU_Y + s * ly,
      z: TerrainDEMLoader.DPPD_CZ + s * (-lx * sn + lz * c)
    };
  }

  /**
   * Transforms Komplekita plan coordinates (kx, ky, kz) directly to Pulau U world coordinates.
   */
  static komplekitaToWorld(kx, ky, kz, extraRotY = 0.0) {
    const ks = DPPDComplexBuilder.KOMPLEKITA_SCALE;
    return DPPDComplexBuilder.localToWorld(
      kx * ks,
      ky * ks,
      kz * ks,
      DPPDComplexBuilder.KOMPLEKITA_YAW_OFFSET + extraRotY
    );
  }

  /**
   * Returns exact world-space solid-body colliders for the 1:1 Komplekita buildings & trees on Pulau U.
   */
  static getSolidColliders() {
    const cols = [];
    const s = TerrainDEMLoader.DPPD_SCALE;
    const ks = DPPDComplexBuilder.KOMPLEKITA_SCALE;
    const netS = s * ks;
    const baseY = TerrainDEMLoader.DPPD_PLATEAU_Y;
    const rotDppd = Math.PI * 0.5 - UI_DPPD_MASTER_PLAN.sectors.dppd.thAxis;

    // 1. ITTC Perpustakaan Lama UI (42m x 42m Square Courtyard Building, ridgeY = 18.80m) + NW 1-Story Annex
    cols.push({
      id: 'DPPD_Courtyard_CentralMonument',
      type: 'CYL',
      x: TerrainDEMLoader.DPPD_CX,
      z: TerrainDEMLoader.DPPD_CZ,
      r: 21.5 * netS,
      minY: baseY,
      maxY: baseY + 18.80 * netS
    });
    const nwAnnexW = DPPDComplexBuilder.komplekitaToWorld(-12.5, 0, -26.2, 0.0);
    cols.push({
      id: 'DPPD_ITTC_NWAnnex',
      type: 'CYL',
      x: nwAnnexW.x,
      z: nwAnnexW.z,
      r: 5.5 * netS,
      minY: baseY,
      maxY: baseY + 4.20 * netS
    });

    // 2. Gedung DPPD UI (2-Story Curved Heritage Wing + Front Gable & Portico, ridgeY = 13.55m)
    const dppdCfg = UI_DPPD_MASTER_PLAN.sectors.dppd;
    for (let k = 0; k < 8; k++) {
      const th = dppdCfg.thStart + ((k + 0.5) / 8) * (dppdCfg.thEnd - dppdCfg.thStart);
      const rMid = 0.5 * (dppdCfg.rInner + dppdCfg.rOuter);
      const wp = DPPDComplexBuilder.komplekitaToWorld(rMid * Math.cos(th), 0, rMid * Math.sin(th), 0.0);
      cols.push({
        id: `DPPD_2Story_HeritageWing_${k}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 9.2 * netS,
        minY: baseY,
        maxY: baseY + 13.55 * netS
      });
    }
    const porchW = DPPDComplexBuilder.komplekitaToWorld(
      58.0 * Math.cos(dppdCfg.thAxis),
      0,
      58.0 * Math.sin(dppdCfg.thAxis),
      0.0
    );
    cols.push({
      id: 'DPPD_2Story_FrontCentralGableAndPorch',
      type: 'CYL',
      x: porchW.x,
      z: porchW.z,
      r: 8.5 * netS,
      minY: baseY,
      maxY: baseY + 13.85 * netS
    });

    // 3. Gedung STP-UI (8-Story Curved Innovation Building, -150° .. -34°, crownTopY = 26.80m) + Portico
    const stpCfg = UI_DPPD_MASTER_PLAN.sectors.stp;
    for (let k = 0; k < 9; k++) {
      const th = stpCfg.thStart + ((k + 0.5) / 9) * (stpCfg.thEnd - stpCfg.thStart);
      const rMid = 0.5 * (stpCfg.rInner + stpCfg.rOuter);
      const wp = DPPDComplexBuilder.komplekitaToWorld(rMid * Math.cos(th), 0, rMid * Math.sin(th), 0.0);
      cols.push({
        id: `DPPD_STPUI_8Story_Sector_${k}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 8.2 * netS,
        minY: baseY,
        maxY: baseY + 26.80 * netS
      });
    }
    const porticoW = DPPDComplexBuilder.komplekitaToWorld(
      64.5 * Math.cos(stpCfg.thPortico),
      0,
      64.5 * Math.sin(stpCfg.thPortico),
      0.0
    );
    cols.push({
      id: 'DPPD_STPUI_Portico',
      type: 'CYL',
      x: porticoW.x,
      z: porticoW.z,
      r: 4.8 * netS,
      minY: baseY,
      maxY: baseY + 5.60 * netS
    });

    // 4. Gedung ILRC & DRPM UI (-31° .. +83°, corniceTopY = 21.20m, South Brick Tower = 27.20m)
    const ilrcCfg = UI_DPPD_MASTER_PLAN.sectors.ilrc;
    for (let k = 0; k < 9; k++) {
      const th = ilrcCfg.thStart + ((k + 0.5) / 9) * (ilrcCfg.thTowerEnd - ilrcCfg.thStart);
      const rMid = 0.5 * (ilrcCfg.rInnerEast + ilrcCfg.rOuter);
      const wp = DPPDComplexBuilder.komplekitaToWorld(rMid * Math.cos(th), 0, rMid * Math.sin(th), 0.0);
      const topH = k >= 7 ? 27.20 : 21.20;
      cols.push({
        id: `DPPD_ILRC_6Story_Sector_${k}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 7.8 * netS,
        minY: baseY,
        maxY: baseY + topH * netS
      });
    }

    // 5. Halte Bikun Shelter at th = -112° (r = 70.6 in local space)
    const thHalte = (-112.0 * Math.PI) / 180.0;
    const halteW = DPPDComplexBuilder.localToWorld(70.60 * Math.cos(thHalte), 0, 70.60 * Math.sin(thHalte), 0.0);
    cols.push({
      id: 'DPPD_HalteBikun',
      type: 'CYL',
      x: halteW.x,
      z: halteW.z,
      r: 2.4 * s,
      minY: baseY,
      maxY: baseY + 3.7 * s
    });

    return cols;
  }
}
