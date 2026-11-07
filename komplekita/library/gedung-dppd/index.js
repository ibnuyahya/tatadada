import * as THREE from 'three';
import { UI_DPPD_MASTER_PLAN } from '../../world-gis/MasterPlanConfig.js';
import { DPPDPrimaryStructure } from './PrimaryStructure.js';
import { DPPDSecondarySubdomains } from './SecondarySubdomains.js';
import { DPPDKinematicRig } from './KinematicRig.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./library/gedung-dppd/index.js
 *
 * Complete Class-A Building Module Orchestrator for Gedung DPPD (eks-KSDP) UI Depok.
 */
export class GedungDPPDBuilder {
  static build(materials) {
    const cfg = UI_DPPD_MASTER_PLAN.sectors.dppd;
    const root = new THREE.Group();
    root.name = 'Gedung_DPPD_UI';

    // Rotate local frame so local +Z points along thAxis = +135.5° SW: psi = pi/2 - thAxis
    const localGroup = new THREE.Group();
    localGroup.rotation.y = Math.PI * 0.5 - cfg.thAxis;
    root.add(localGroup);

    DPPDPrimaryStructure.build(localGroup, materials);
    DPPDSecondarySubdomains.build(localGroup, materials);
    const rig = DPPDKinematicRig.build(localGroup, materials);

    return { group: root, rig };
  }
}
