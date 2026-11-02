import * as THREE from 'three';
import {
  SURFACE_PROFILE,
  createSharedSurfaceUniforms,
  injectOmniScaleSurfaceShader,
  createPrecastConcreteTexture,
  createWeatheredFlatRoofTexture,
  createTerracottaBrickTexture,
  createJavaneseClayTileTexture,
  createHorizontalStonePaverTexture,
  createRibbonGlassTextures,
  createAtriumDiamondGlassTextures,
  createMakaraEmblemTextures
} from '../../../core/ProceduralTextureFactory.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.5 — 100x Architectural & World Surface Detail Edition)
 * ./library/landmark/Gedung_Rektorat_UI/MaterialShaderBank.js
 *
 * TAHAP 6B (Gate 4): Material PBR Linear-Safe Terkalibrasi 5 Foto DJI Mavic,
 * 100x Analytical Screen-Space Micro-Normal Relief, 3D Ray-Box Parallax Interior Rooms,
 * True Emissive Masking (#000000 Pasif), Analytical Contact AO, & Studio Clay/Zebra Inspection.
 */
export class MaterialShaderBank {
  static createMaterials(externalUniforms = null) {
    const sharedUniforms = externalUniforms || createSharedSurfaceUniforms();

    const concreteTex = createPrecastConcreteTexture();
    const flatRoofTex = createWeatheredFlatRoofTexture();
    const brickTex = createTerracottaBrickTexture();
    const clayTileTex = createJavaneseClayTileTexture();
    const plazaStoneTex = createHorizontalStonePaverTexture();
    const ribbonGlassTex = createRibbonGlassTextures();
    const atriumGlassTex = createAtriumDiamondGlassTextures();
    const makaraTex = createMakaraEmblemTextures();

    const shaderCfg = {
      ySegmentStart: 5.65,
      segmentStepH: 3.25,
      segmentCount: 6.0,
      sharedUniforms
    };

    // 1 & 3. Unified Weathered Tropical Flat Concrete Roof Deck Slab & Vertex-Tinted Sub-Terrain Foundation / Planter Soil
    //        (Shares SURFACE_PROFILE.WEATHERED_FLAT_ROOF via vertexColors -> saves 1 WebGL Draw Call with 0 visual loss!)
    const flatRoofSlab = injectOmniScaleSurfaceShader(
      new THREE.MeshStandardMaterial({
        name: 'WeatheredFlatRoofAndFoundationSlab',
        color: 0xb2b6b8,
        vertexColors: true,
        map: flatRoofTex,
        roughness: 0.91,
        metalness: 0.02,
        envMapIntensity: 0.09
      }),
      { ...shaderCfg, surfaceProfile: SURFACE_PROFILE.WEATHERED_FLAT_ROOF, bumpScale: 0.72 }
    );
    const foundationConcrete = flatRoofSlab;

    // 2. Crisp Off-White / Warm Cream Architectural Precast Concrete (Mavic Photo #1 & #2)
    const precastConcrete = injectOmniScaleSurfaceShader(
      new THREE.MeshStandardMaterial({
        name: 'PrecastConcrete',
        color: 0xf4f0e8,
        map: concreteTex,
        roughness: 0.82,
        metalness: 0.03,
        envMapIntensity: 0.12
      }),
      { ...shaderCfg, surfaceProfile: SURFACE_PROFILE.PRECAST_CONCRETE, bumpScale: 0.72 }
    );

    // 4. Warm Terracotta Trim / Ridge & Brick Accent
    const terracottaBrick = injectOmniScaleSurfaceShader(
      new THREE.MeshStandardMaterial({
        name: 'TerracottaBrick',
        color: 0xba6654,
        map: brickTex,
        roughness: 0.82,
        metalness: 0.03,
        envMapIntensity: 0.1
      }),
      { ...shaderCfg, surfaceProfile: SURFACE_PROFILE.TERRACOTTA_MASONRY, bumpScale: 0.75 }
    );

    // 5. Authentic Weathered Javanese Terracotta Red Clay Roof Tiles (Mavic Photos #1–#5)
    const terracottaTile = injectOmniScaleSurfaceShader(
      new THREE.MeshStandardMaterial({
        name: 'JavaneseClayTile',
        color: 0xc9624f,
        map: clayTileTex,
        roughness: 0.75,
        metalness: 0.03,
        envMapIntensity: 0.13
      }),
      { ...shaderCfg, surfaceProfile: SURFACE_PROFILE.JAVANESE_CLAY_TILE, bumpScale: 0.85 }
    );

    // 6. Signature Rektorat UI Courtyard Stone Pavers (Grey-Olive Grid + White Diamond ◇◇◇)
    const horizontalPlazaStone = injectOmniScaleSurfaceShader(
      new THREE.MeshStandardMaterial({
        name: 'HorizontalPlazaStone',
        color: 0xc8c6be,
        map: plazaStoneTex,
        roughness: 0.84,
        metalness: 0.03,
        envMapIntensity: 0.11
      }),
      { ...shaderCfg, surfaceProfile: SURFACE_PROFILE.UI_DIAMOND_PLAZA, bumpScale: 0.75 }
    );

    // 7. 6-Bay White-Mullioned Dark Horizontal Ribbon Glass + 3D Ray-Box Parallax Interior Office Rooms
    const ribbonGlass = injectOmniScaleSurfaceShader(
      new THREE.MeshStandardMaterial({
        name: 'RibbonCurtainGlass',
        color: 0xffffff,
        map: ribbonGlassTex.map,
        emissive: new THREE.Color(0xffcf7c),
        emissiveMap: ribbonGlassTex.emissiveMap,
        emissiveIntensity: 0.0,
        roughness: 0.12,
        metalness: 0.24,
        envMapIntensity: 0.86
      }),
      { ...shaderCfg, surfaceProfile: SURFACE_PROFILE.RIBBON_INTERIOR_GLASS, bumpScale: 0.0 }
    );

    // 8. Central Recessed Vertical X-Diamond Atrium Glass & Skylight Glass + 3D Parallax Atrium Gallery
    const atriumGlass = injectOmniScaleSurfaceShader(
      new THREE.MeshStandardMaterial({
        name: 'AtriumDiamondGlass',
        color: 0xffffff,
        map: atriumGlassTex.map,
        emissive: new THREE.Color(0xffca75),
        emissiveMap: atriumGlassTex.emissiveMap,
        emissiveIntensity: 0.0,
        roughness: 0.10,
        metalness: 0.28,
        envMapIntensity: 0.92
      }),
      { ...shaderCfg, surfaceProfile: SURFACE_PROFILE.ATRIUM_DIAMOND_GLASS, bumpScale: 0.0 }
    );

    // 9. Upper Clerestory & Skylit Pendopo Lantern Glass + 3D Parallax Crown Hall
    const lanternGlass = injectOmniScaleSurfaceShader(
      new THREE.MeshStandardMaterial({
        name: 'CrownClerestoryGlass',
        color: 0x162228,
        emissive: new THREE.Color(0xffb454),
        emissiveIntensity: 0.03,
        roughness: 0.13,
        metalness: 0.22,
        envMapIntensity: 0.80
      }),
      { ...shaderCfg, surfaceProfile: SURFACE_PROFILE.ATRIUM_DIAMOND_GLASS, bumpScale: 0.0 }
    );

    // 10. White/Metallic Apex Finial, Ridge Trim, & Architectural Bronze Mullion Hardware
    const bronzeGold = injectOmniScaleSurfaceShader(
      new THREE.MeshStandardMaterial({
        name: 'ApexAndMastTrim',
        color: 0xe6e2d8,
        roughness: 0.30,
        metalness: 0.54,
        envMapIntensity: 0.72
      }),
      { ...shaderCfg, surfaceProfile: SURFACE_PROFILE.MAKARA_STONE_BRONZE, bumpScale: 0.85 }
    );

    // 11. Sculpted Bas-Relief Makara UI on Crisp Off-White Concrete Panel (#000000 Night Mask!)
    const makaraEmblem = injectOmniScaleSurfaceShader(
      new THREE.MeshStandardMaterial({
        name: 'MakaraUIEmblem',
        color: 0xf4f0e8,
        map: makaraTex.map,
        emissive: new THREE.Color(0xffd24c),
        emissiveMap: makaraTex.emissiveMap,
        emissiveIntensity: 0.0,
        roughness: 0.64,
        metalness: 0.14,
        envMapIntensity: 0.22
      }),
      { ...shaderCfg, surfaceProfile: SURFACE_PROFILE.MAKARA_STONE_BRONZE, bumpScale: 0.95 }
    );

    // 12. Dynamic Ceremonial Flag Fabric (Merah-Putih + 14 Colorful UI Faculty Panji Flags!)
    const flagFabric = injectOmniScaleSurfaceShader(
      new THREE.MeshStandardMaterial({
        name: 'CeremonialFlagFabric',
        vertexColors: true,
        side: THREE.DoubleSide,
        roughness: 0.76,
        metalness: 0.02,
        envMapIntensity: 0.1
      }),
      {
        ...shaderCfg,
        segmentCount: 0.0,
        enableWindSway: false,
        surfaceProfile: SURFACE_PROFILE.DEFAULT,
        bumpScale: 0.16
      }
    );

    // 13. Summit Aviation Warning Beacon
    const aviationBeacon = injectOmniScaleSurfaceShader(
      new THREE.MeshStandardMaterial({
        name: 'AviationWarningBeacon',
        color: 0x661111,
        emissive: new THREE.Color(0xff261b),
        emissiveIntensity: 0.15,
        roughness: 0.25,
        metalness: 0.1
      }),
      {
        ...shaderCfg,
        segmentCount: 0.0,
        enableWindSway: false,
        surfaceProfile: SURFACE_PROFILE.DEFAULT,
        bumpScale: 0.05
      }
    );

    return {
      sharedUniforms,
      foundationConcrete,
      precastConcrete,
      flatRoofSlab,
      terracottaBrick,
      terracottaTile,
      horizontalPlazaStone,
      ribbonGlass,
      atriumGlass,
      lanternGlass,
      bronzeGold,
      makaraEmblem,
      flagFabric,
      aviationBeacon
    };
  }
}
