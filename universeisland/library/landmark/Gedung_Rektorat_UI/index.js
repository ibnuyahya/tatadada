import * as THREE from 'three';
import { NexusBatchOptimizer } from '../../../core/BatchRenderer.js';
import { GateVerifier } from '../../../core/GateVerifier.js';
import { createRibbonGlassTextures } from '../../../core/ProceduralTextureFactory.js';
import { GEDUNG_REKTORAT_UI_META } from './AssetMetaAndEnvelope.js';
import { MasterSurfaceKernel } from './MasterSurfaceKernel.js';
import { PrimaryStructure } from './PrimaryStructure.js';
import { SecondarySubdomains } from './SecondarySubdomains.js';
import { KinematicRig } from './KinematicRig.js';
import { MaterialShaderBank } from './MaterialShaderBank.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./library/landmark/Gedung_Rektorat_UI/index.js
 *
 * Perakit Akhir Aset Kelas-A sesuai Kontrak `StandardNexusAsset` (v2.0):
 * Gedung Pusat Administrasi / Rektorat Universitas Indonesia (Kampus Depok)
 * Karya Prof. Ir. Gunawan Tjahjono, M.Arch., Ph.D.
 */
export class GedungRektoratUIAsset {
  static meta = GEDUNG_REKTORAT_UI_META;

  /**
   * Fungsi Single Source of Spatial Truth (SSoST):
   * Mengembalikan { position: THREE.Vector3, normal: THREE.Vector3 } pada domain (u, v) in [0, 1]^2.
   */
  static evaluateMasterSurface(
    u,
    v,
    targetPos = new THREE.Vector3(),
    targetNorm = new THREE.Vector3()
  ) {
    return MasterSurfaceKernel.evaluateMasterSurface(u, v, targetPos, targetNorm);
  }

  /**
   * Membangun objek dengan 3 tingkat Auto-LOD + Single-Pass Batching (<= 25 Draw Calls)
   * dan menjalankan verifikasi otomatis 4-Gate (`GateVerifier.runAllGates`).
   */
  static build(sharedMaterials = null, sharedGeometries = null) {
    const materials =
      sharedMaterials && sharedMaterials.precastConcrete
        ? sharedMaterials
        : MaterialShaderBank.createMaterials(sharedMaterials?.sharedUniforms || null);

    const lod = new THREE.LOD();
    lod.name = 'GEDUNG_REKTORAT_UI_DEPOK_LOD';

    // --- LOD 0: Ultra Class-A Inspection & Near View (0m .. 160m) ---
    const lod0Group = new THREE.Group();
    lod0Group.name = 'RektoratUI_LOD0';
    const primary0 = PrimaryStructure.build(materials, 1.0);
    const secondary0 = SecondarySubdomains.build(materials, 1.0);
    const kinematic0 = KinematicRig.build(materials);
    lod0Group.add(primary0.group, secondary0, kinematic0);
    NexusBatchOptimizer.optimizeHierarchy(lod0Group);
    lod.addLevel(lod0Group, 0);

    // --- LOD 1: Medium Campus View (160m .. 310m) ---
    const lod1Group = new THREE.Group();
    lod1Group.name = 'RektoratUI_LOD1';
    lod1Group.visible = false;
    const primary1 = PrimaryStructure.build(materials, 0.5);
    const secondary1 = SecondarySubdomains.build(materials, 0.5);
    lod1Group.add(primary1.group, secondary1);
    NexusBatchOptimizer.optimizeHierarchy(lod1Group);
    lod.addLevel(lod1Group, 160);

    // --- LOD 2: Far Horizon Silhouette (>= 310m) ---
    const lod2Group = new THREE.Group();
    lod2Group.name = 'RektoratUI_LOD2';
    lod2Group.visible = false;
    const primary2 = PrimaryStructure.build(materials, 0.25);
    lod2Group.add(primary2.group);
    NexusBatchOptimizer.optimizeHierarchy(lod2Group);
    lod.addLevel(lod2Group, 310);

    const rigCtrl = kinematic0.userData.kinematicController;
    lod.userData.materials = materials;
    lod.userData.kinematicController = rigCtrl;
    lod.userData.subdomainBoundaries = primary0.subdomainBoundaries;

    // Run Automated 4-Gate Verification ("Verified by Design")
    const gateReport = GateVerifier.runAllGates(lod0Group, this.meta.realDimensionsMeters, null, {
      foundationSkirtDepth: this.meta.foundationSkirtDepth,
      evaluateMasterSurface: MasterSurfaceKernel.evaluateMasterSurface,
      subdomainBoundaries: primary0.subdomainBoundaries,
      kinematicSweepFn: KinematicRig.runKinematicSweepCheck(rigCtrl)
    });
    lod.userData.gateReport = gateReport;

    return lod;
  }

  /**
   * Menghasilkan tekstur True Emissive Mask (bagian pasif #000000 mutlak, 0% bleed).
   */
  static emissiveMaskGenerator(canvasWidth = 2048, canvasHeight = 512) {
    return createRibbonGlassTextures().emissiveMap;
  }

  /**
   * Mengatur iluminasi malam hari dengan disiplin True Emissive Masking (Hukum Emas 13).
   */
  static setNightIllumination(meshRoot, isNight) {
    const mats = meshRoot?.userData?.materials;
    if (!mats) return;

    const nightLevel = Math.max(0.0, Math.min(1.0, typeof isNight === 'number' ? isNight : isNight ? 1.0 : 0.0));
    if (mats.sharedUniforms && mats.sharedUniforms.uIsNight) {
      mats.sharedUniforms.uIsNight.value = nightLevel <= 0.015 ? 0.0 : nightLevel;
    }

    // Smooth C0-continuous twilight ramp (strictly 0.0 at DAY nightLevel <= 0.015 per Hukum Emas 10!)
    const smoothRamp = nightLevel <= 0.015 ? 0.0 : (nightLevel - 0.015) / 0.985;
    mats.ribbonGlass.emissiveIntensity = 1.35 * smoothRamp;
    if (mats.atriumGlass) {
      mats.atriumGlass.emissiveIntensity = 1.25 * smoothRamp;
    }
    mats.lanternGlass.emissiveIntensity = smoothRamp > 0.0 ? 0.03 + 1.72 * smoothRamp : 0.03;
    mats.makaraEmblem.emissiveIntensity = 1.35 * smoothRamp;
  }

  /**
   * Memperbarui kinematika bendera upacara, pintu lobi, dan lampu suar puncak.
   */
  static updateKinematics(meshRoot, dt, telemetryState = {}) {
    const rigCtrl = meshRoot?.userData?.kinematicController;
    if (!rigCtrl) return;
    KinematicRig.update(rigCtrl, dt, telemetryState);
  }
}

export { GedungRektoratUIAsset as StandardNexusAsset };
export default GedungRektoratUIAsset;
