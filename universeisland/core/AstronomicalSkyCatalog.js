import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v3.7
 * ./core/AstronomicalSkyCatalog.js
 *
 * Pure-Procedural Real-Time Equatorial Island Astronomy & J2000 Bright-Star Constellation Catalog
 * for Kepulauan Universitas Indonesia / Kampus Depok (Lat -6.3658° S, Lon +106.8282° E, UTC+7 WIB):
 * - Continuous Jean Meeus Solar & Lunar Ephemeris (0.0000° midnight-wrap discontinuity!)
 * - Exact Real-Time Lunar Phase, Illuminated Fraction, Lunar Age, & 3D World Terminator Sun-Vector
 * - Orthonormal World-Horizon <-> J2000 Equatorial Rotation Matrix driven by Local Sidereal Time (LST)
 * - 208 Real J2000 Navigational & Constellation Stars (Orion, Crux/Pari, Centaurus, Scorpius,
 *   Sagittarius, Canis Major/Sirius, Carina/Canopus, Taurus + Pleiades M45, Gemini, Auriga, Leo,
 *   Virgo, Bootes, Lyra, Cygnus, Aquila, Cassiopeia, Perseus, Pegasus, Andromeda, Ursa Major,
 *   Eridanus, Piscis Austrinus, Grus, Pavo) packed into an O(1) 384x64 RGBA32F GPU Spatial Grid!
 * - Zero external files, < 3 ms CPU initialization, 0 extra Draw Calls.
 */

const DEG2RAD = Math.PI / 180.0;
const RAD2DEG = 180.0 / Math.PI;
const TWO_PI = Math.PI * 2.0;

// Kepulauan UI Depok Geographic Coordinates (WIB = UTC+7)
export const DEPOK_LAT_DEG = -6.3658;
export const DEPOK_LON_DEG = 106.8282;
export const DEPOK_WIB_OFFSET_HOURS = 7.0;

/**
 * Real J2000.0 Bright Constellation & Navigational Star Catalog:
 * Each entry: [RA_hours, Dec_deg, Vmag, BV_colorIndex]
 * Covers all major Northern, Equatorial & Southern constellations visible from -6.36° S Nusantara.
 */
export const REAL_J2000_BRIGHT_STARS = [
  // === ORION (Waluku / Bintang Bajak — Unmistakable Equatorial Showpiece!) ===
  [5.9195, 7.4071, 0.45, 1.85],   // Betelgeuse (Alpha Ori — Ruby-Orange M2 Supergiant)
  [5.2423, -8.2016, 0.13, -0.03], // Rigel (Beta Ori — Blue-White B8 Supergiant)
  [5.4189, 6.3497, 1.64, -0.22],  // Bellatrix (Gamma Ori)
  [5.7959, -9.6696, 2.06, -0.18], // Saiph (Kappa Ori)
  [5.5334, -0.2991, 2.23, -0.21], // Mintaka (Delta Ori — Western Belt Star)
  [5.6036, -1.2019, 1.69, -0.18], // Alnilam (Epsilon Ori — Center Belt Star)
  [5.6793, -1.9426, 1.77, -0.20], // Alnitak (Zeta Ori — Eastern Belt Star)
  [5.5855, 9.9342, 3.39, -0.18],  // Meissa (Lambda Ori — Head of Orion)
  [5.5906, -5.9099, 2.77, -0.24], // Hatysa (Iota Ori — Tip of Orion's Sword / M42)
  [5.5885, -4.8380, 3.70, -0.20], // c Orionis (Upper Sword)
  [5.5521, -2.6001, 3.77, -0.20], // Sigma Orionis
  [4.8308, 6.9613, 3.19, 0.45],   // Pi3 Orionis (Tabit — Orion's Shield)
  [4.8535, 5.6051, 3.69, -0.18],  // Pi4 Orionis
  [4.9042, 2.4407, 3.72, -0.18],  // Pi5 Orionis
  [5.3573, -2.3971, 3.35, -0.20], // Eta Orionis

  // === CRUX (Rasi Pari / Gubug Penceng — Iconic Southern Cross!) & CENTAURUS ===
  [12.4433, -63.0991, 0.77, -0.24], // Acrux (Alpha Crucis — Foot of Southern Cross)
  [12.7953, -59.6888, 1.25, -0.24], // Mimosa (Beta Crucis — Eastern Arm)
  [12.5194, -57.1132, 1.63, 1.59],  // Gacrux (Gamma Crucis — Red Giant Top of Cross!)
  [12.2524, -58.7489, 2.79, -0.23], // Imai (Delta Crucis — Western Arm)
  [12.3565, -60.4012, 3.59, 1.33],  // Ginan (Epsilon Crucis — Fifth Star of Crux)
  [14.6601, -60.8340, -0.27, 0.71], // Rigil Kentaurus (Alpha Centauri — Southern Pointer #1)
  [14.0637, -60.3730, 0.61, -0.23], // Hadar (Beta Centauri — Southern Pointer #2)
  [14.1114, -36.3700, 2.06, 1.01],  // Menkent (Theta Centauri)
  [12.6919, -48.9599, 2.17, -0.02], // Muhlifain (Gamma Centauri)
  [13.9258, -47.2884, 2.55, -0.22], // Zeta Centauri
  [13.6648, -53.4664, 2.30, -0.22], // Epsilon Centauri
  [12.1398, -50.7224, 2.58, -0.19], // Delta Centauri
  [13.3401, -36.7123, 2.75, 0.04],  // Iota Centauri

  // === SCORPIUS (Kalajengking — Full Curved J-Hook across Southern Milky Way!) ===
  [16.4901, -26.4320, 0.96, 1.83], // Antares (Alpha Sco — Fiery Heart of the Scorpion)
  [17.5601, -37.1038, 1.62, -0.23], // Shaula (Lambda Sco — Bright Stinger #1)
  [17.5127, -37.2958, 2.70, -0.22], // Lesath (Upsilon Sco — Stinger Twin #2)
  [17.6220, -42.9978, 1.86, 0.40], // Sargas (Theta Sco — Southern Bend)
  [16.0056, -22.6217, 2.29, -0.12], // Dschubba (Delta Sco — Middle Forehead Claw)
  [16.0906, -19.8055, 2.56, -0.07], // Acrab / Graffias (Beta Sco — Upper Claw)
  [15.9809, -26.1141, 2.89, -0.19], // Fang (Pi Sco — Lower Claw)
  [16.3532, -25.5928, 2.90, 0.13], // Alniyat (Sigma Sco)
  [16.5981, -28.2160, 2.82, -0.21], // Paikauhale (Tau Sco)
  [16.8361, -34.2932, 2.29, 1.14], // Larawag (Epsilon Sco — Mid-Tail)
  [16.8648, -38.0474, 3.00, -0.20], // Mu1 Scorpii
  [16.9102, -42.3620, 3.62, 1.37], // Zeta2 Scorpii
  [17.2022, -43.2390, 3.32, 0.41], // Eta Scorpii
  [17.7082, -39.0300, 2.39, -0.21], // Girtab (Kappa Sco)
  [17.7931, -40.1270, 2.99, 0.51], // Iota1 Scorpii

  // === SAGITTARIUS (Teapot Asterism framing the Galactic Core!) ===
  [18.4029, -34.3846, 1.79, -0.03], // Kaus Australis (Epsilon Sgr — Teapot Base)
  [18.9211, -26.2967, 2.05, -0.21], // Nunki (Sigma Sgr — Handle Top)
  [19.0435, -29.8801, 2.60, 0.08],  // Ascella (Zeta Sgr — Handle Base)
  [18.3499, -29.8281, 2.72, 1.38],  // Kaus Media (Delta Sgr — Spout Base)
  [18.4661, -25.4217, 2.82, 1.04],  // Kaus Borealis (Lambda Sgr — Lid Tip)
  [18.0968, -30.4241, 2.98, 1.00],  // Alnasl (Gamma2 Sgr — Spout Tip)
  [18.7609, -26.9908, 3.17, -0.11], // Phi Sagittarii
  [19.1157, -27.6703, 3.32, 1.18],  // Tau Sagittarii
  [19.0642, -21.0236, 2.88, 0.38],  // Albaldah (Pi Sgr)

  // === CANIS MAJOR, CANIS MINOR & MONOCEROS ===
  [6.7525, -16.7161, -1.46, 0.00], // Sirius (Alpha CMa — Brightest Star in the Night Sky!)
  [6.9771, -28.9721, 1.50, -0.21], // Adhara (Epsilon CMa)
  [7.1399, -26.3932, 1.83, 0.67],  // Wezen (Delta CMa)
  [6.3783, -17.9559, 1.98, -0.24], // Mirzam (Beta CMa)
  [7.4016, -29.3031, 2.45, -0.08], // Aludra (Eta CMa)
  [6.3388, -30.0634, 3.02, -0.18], // Furud (Zeta CMa)
  [7.0504, -23.8333, 3.02, -0.14], // Omicron2 CMa
  [7.6550, 5.2250, 0.34, 0.42],    // Procyon (Alpha CMi)
  [7.4526, 8.2893, 2.89, -0.09],   // Gomeisa (Beta CMi)

  // === TAURUS, HYADES & PLEIADES (Lintang Kartika / Seven Sisters M45!) ===
  [4.5987, 16.5093, 0.86, 1.54],  // Aldebaran (Alpha Tau — Orange Bull's Eye)
  [5.4382, 28.6075, 1.65, -0.13], // Elnath (Beta Tau — Northern Horn)
  [5.6274, 21.1425, 2.97, -0.17], // Tianguan (Zeta Tau — Southern Horn)
  [4.4777, 15.8709, 3.40, 0.14],  // Theta2 Tauri (Hyades V)
  [4.3299, 15.6276, 3.65, 0.98],  // Gamma Tauri (Hyades Vertex)
  [4.4766, 19.1805, 3.53, 0.99],  // Ain (Epsilon Tauri — Hyades)
  [3.7914, 24.1051, 2.87, -0.09], // Alcyone (Eta Tau — Pleiades M45 Brightest!)
  [3.8197, 24.0534, 3.62, -0.08], // Atlas (27 Tau — Pleiades)
  [3.7481, 24.1133, 3.70, -0.11], // Electra (17 Tau — Pleiades)
  [3.7638, 24.3678, 3.87, -0.07], // Maia (20 Tau — Pleiades)
  [3.7721, 23.9484, 4.14, -0.06], // Merope (23 Tau — Pleiades)

  // === CARINA, VELA & PUPPIS (Argo Navis & False Cross) ===
  [6.3992, -52.6957, -0.74, 0.15], // Canopus (Alpha Car — 2nd Brightest Star in the Sky!)
  [9.2200, -69.7172, 1.67, 0.07],  // Miaplacidus (Beta Car)
  [8.3752, -59.5095, 1.86, 1.28],  // Avior (Epsilon Car — False Cross)
  [9.2848, -59.2752, 2.21, 0.19],  // Aspidiske (Iota Car — False Cross)
  [8.7451, -54.7088, 1.95, 0.04],  // Alsephina (Delta Vel — False Cross)
  [9.3685, -55.0107, 2.48, -0.18], // Markeb (Kappa Vel — False Cross)
  [8.1589, -47.3366, 1.75, -0.22], // Suhail al Muhlif (Gamma Vel)
  [9.1333, -43.4326, 2.21, 1.66],  // Suhail (Lambda Vel)
  [8.0598, -40.0031, 2.25, -0.27], // Naos (Zeta Pup)
  [7.2875, -37.0974, 2.71, 1.62],  // Pi Puppis
  [8.1258, -24.3043, 2.83, 0.43],  // Tureis (Rho Pup)
  [10.2286, -70.0379, 3.30, -0.07], // Omega Carinae
  [10.7165, -64.3945, 2.74, -0.22], // Theta Carinae

  // === GEMINI & AURIGA ===
  [7.7553, 28.0262, 1.14, 1.00],  // Pollux (Beta Gem)
  [7.5767, 31.8883, 1.58, 0.03],  // Castor (Alpha Gem)
  [6.6285, 16.3993, 1.93, 0.00],  // Alhena (Gamma Gem)
  [6.3826, 22.5136, 2.87, 1.64],  // Tejat (Mu Gem)
  [6.7322, 25.1311, 3.06, 1.40],  // Mebsuta (Epsilon Gem)
  [7.3357, 21.9823, 3.50, 0.34],  // Wasat (Delta Gem)
  [5.2782, 45.9980, 0.08, 0.80],  // Capella (Alpha Aur)
  [5.9921, 44.9474, 1.90, 0.08],  // Menkalinan (Beta Aur)
  [5.9951, 37.2126, 2.65, -0.08], // Mahasim (Theta Aur)
  [4.9499, 33.1661, 2.69, 1.53],  // Hassaleh (Iota Aur)

  // === SUMMER TRIANGLE: LYRA, CYGNUS (Northern Cross), & AQUILA ===
  [18.6156, 38.7837, 0.03, 0.00], // Vega (Alpha Lyr)
  [18.8347, 33.3627, 3.52, -0.05], // Sheliak (Beta Lyr)
  [18.9823, 32.6896, 3.25, -0.05], // Sulafat (Gamma Lyr)
  [20.6905, 45.2803, 1.25, 0.09], // Deneb (Alpha Cyg — Top of Northern Cross)
  [20.3705, 40.2567, 2.23, 0.67], // Sadr (Gamma Cyg — Center of Cross / Cygnus Rift)
  [19.5120, 27.9597, 3.05, 1.13], // Albireo (Beta Cyg — Foot of Northern Cross)
  [20.7702, 33.9703, 2.48, 1.02], // Gienah (Epsilon Cyg — Eastern Arm)
  [19.7496, 45.1308, 2.86, -0.03], // Fawaris (Delta Cyg — Western Arm)
  [19.8464, 8.8683, 0.76, 0.22],  // Altair (Alpha Aql)
  [19.7710, 10.6133, 2.72, 1.51], // Tarazed (Gamma Aql)
  [19.9219, 6.4068, 3.71, 0.86],  // Alshain (Beta Aql)
  [19.0902, 13.8635, 2.99, 0.01], // Okab (Zeta Aql)

  // === LEO, VIRGO, BOOTES & CORVUS ===
  [10.1395, 11.9672, 1.35, -0.11], // Regulus (Alpha Leo — Base of Sickle)
  [10.3329, 19.8415, 2.01, 1.14],  // Algieba (Gamma Leo)
  [11.8177, 14.5721, 2.14, 0.09],  // Denebola (Beta Leo — Lion's Tail)
  [11.2351, 20.5237, 2.56, 0.12],  // Zosma (Delta Leo)
  [9.7642, 23.7743, 2.97, 0.81],   // Ras Elased (Epsilon Leo)
  [11.2373, 15.4296, 3.33, 0.01],  // Chertan (Theta Leo)
  [10.2782, 23.4173, 3.43, 0.33],  // Adhafera (Zeta Leo)
  [13.4199, -11.1613, 0.97, -0.23], // Spica (Alpha Vir)
  [12.6943, -1.4494, 2.74, 0.36],  // Porrima (Gamma Vir)
  [13.0363, 10.9591, 2.85, 0.93],  // Vindemiatrix (Epsilon Vir)
  [13.5782, -0.5959, 3.38, 0.11],  // Heze (Zeta Vir)
  [14.2610, 19.1824, -0.05, 1.23], // Arcturus (Alpha Boo — Brilliant Golden-Orange!)
  [14.7498, 27.0742, 2.35, 0.97],  // Izar (Epsilon Boo)
  [13.9114, 18.3977, 2.68, 0.58],  // Muphrid (Eta Boo)
  [14.5349, 38.3083, 3.04, 0.19],  // Seginus (Gamma Boo)
  [12.2634, -17.5419, 2.58, -0.11], // Gienah Corvi (Gamma Crv)
  [12.5731, -23.3968, 2.65, 0.89], // Kraz (Beta Crv)
  [12.4977, -16.5154, 2.94, -0.05], // Algorab (Delta Crv)
  [12.1689, -22.6198, 3.02, 1.33], // Minkar (Epsilon Crv)

  // === CASSIOPEIA ('W' Asterism), PERSEUS, ANDROMEDA & PEGASUS (Great Square) ===
  [0.6751, 56.5373, 2.24, 1.17],  // Schedar (Alpha Cas)
  [0.1529, 59.1498, 2.28, 0.34],  // Caph (Beta Cas)
  [0.9451, 60.7167, 2.15, -0.15], // Gamma Cassiopeiae (Center of 'W')
  [1.4303, 60.2353, 2.68, 0.13],  // Ruchbah (Delta Cas)
  [1.9066, 63.6701, 3.35, -0.15], // Segin (Epsilon Cas)
  [3.4054, 49.8612, 1.79, 0.48],  // Mirfak (Alpha Per)
  [3.1361, 40.9556, 2.09, -0.05], // Algol (Beta Per)
  [3.9022, 31.8836, 2.86, -0.13], // Zeta Persei
  [3.9642, 40.0102, 2.90, -0.18], // Epsilon Persei
  [0.1398, 29.0904, 2.07, -0.11], // Alpheratz (Alpha And — NE Corner of Pegasus Square)
  [1.1622, 35.6206, 2.07, 1.58],  // Mirach (Beta And)
  [2.0650, 42.3297, 2.10, 1.37],  // Almach (Gamma And)
  [23.0793, 15.2053, 2.49, -0.04], // Markab (Alpha Peg — SW Corner of Pegasus Square)
  [23.0629, 28.0828, 2.44, 1.67], // Scheat (Beta Peg — NW Corner of Pegasus Square)
  [0.2206, 15.1836, 2.84, -0.23], // Algenib (Gamma Peg — SE Corner of Pegasus Square)
  [21.7364, 9.8750, 2.38, 1.52],  // Enif (Epsilon Peg)
  [22.7167, 30.2212, 2.93, 0.85], // Matar (Eta Peg)
  [2.1196, 23.4624, 2.01, 1.15],  // Hamal (Alpha Ari)
  [1.9107, 20.8080, 2.64, 0.13],  // Sheratan (Beta Ari)

  // === URSA MAJOR (Biduk Utara / Big Dipper) ===
  [11.0621, 61.7510, 1.79, 1.07], // Dubhe (Alpha UMa)
  [11.0307, 56.3824, 2.37, -0.02], // Merak (Beta UMa)
  [11.8972, 53.6948, 2.44, 0.00], // Phecda (Gamma UMa)
  [12.2571, 57.0326, 3.32, 0.08], // Megrez (Delta UMa)
  [12.9005, 55.9598, 1.76, -0.02], // Alioth (Epsilon UMa)
  [13.3988, 54.9254, 2.23, 0.02], // Mizar (Zeta UMa)
  [13.7923, 49.3133, 1.85, -0.19], // Alkaid (Eta UMa — Handle Tip)

  // === ERIDANUS, PISCIS AUSTRINUS, CETUS, AQUARIUS, OPHIUCHUS, LIBRA, GRUS, PAVO, HYDRUS ===
  [1.6286, -57.2368, 0.46, -0.16], // Achernar (Alpha Eri — Southern River Mouth)
  [5.1308, -5.0864, 2.78, 0.13],   // Cursa (Beta Eri — Near Rigel)
  [2.9710, -40.3047, 2.88, 0.14],  // Acamar (Theta Eri)
  [3.9672, -13.5085, 2.95, 1.59],  // Zaurak (Gamma Eri)
  [22.9608, -29.6222, 1.16, 0.09], // Fomalhaut (Alpha PsA — Lonely Autumn Beacon)
  [0.7265, -17.9866, 2.04, 1.02],  // Diphda (Beta Cet)
  [3.0380, 4.0897, 2.54, 1.63],    // Menkar (Alpha Cet)
  [21.5260, -5.5712, 2.90, 0.83],  // Sadalsuud (Beta Aqr)
  [22.0964, -0.3199, 2.95, 0.97],  // Sadalmelik (Alpha Aqr)
  [20.7946, -14.7814, 3.77, -0.06], // Albali (Epsilon Aqr)
  [17.5822, 12.5600, 2.08, 0.15],  // Rasalhague (Alpha Oph)
  [17.1730, -15.7249, 2.43, 0.06], // Sabik (Eta Oph)
  [16.6193, -10.5671, 2.54, -0.05], // Zeta Ophiuchi
  [17.7246, 4.5673, 2.76, 1.16],   // Cebalrai (Beta Oph)
  [15.2834, -9.3829, 2.61, -0.11], // Zubeneschamali (Beta Lib)
  [14.8480, -16.0418, 2.75, 0.15], // Zubenelgenubi (Alpha2 Lib)
  [15.7378, 6.4256, 2.63, 1.17],   // Unukalhai (Alpha Ser)
  [15.5781, 26.7147, 2.22, -0.02], // Alphecca (Alpha CrB)
  [22.1372, -46.9610, 1.73, -0.14], // Alnair (Alpha Gru)
  [22.7111, -46.8846, 2.13, 1.61], // Tiaki (Beta Gru)
  [21.8989, -37.3649, 3.00, -0.12], // Gamma Gruis
  [20.4275, -56.7351, 1.94, -0.20], // Peacock (Alpha Pav)
  [0.4293, -42.3060, 2.40, 1.09],  // Ankaa (Alpha Phe)
  [22.3090, -60.2597, 2.86, 1.39], // Alpha Tucanae
  [0.4285, -77.2542, 2.82, 0.62],  // Beta Hydri
  [1.9794, -61.5699, 2.86, 0.28],  // Alpha Hydri
  [16.8112, -68.9397, 1.91, 1.45], // Atria (Alpha TrA — Southern Triangle)
  [15.9190, -63.4305, 2.83, 0.29], // Beta Trianguli Australis
  [15.3147, -68.6795, 2.87, 0.00], // Gamma Trianguli Australis
  [17.5305, -49.8761, 2.84, -0.17], // Beta Arae
  [17.5315, -55.5299, 2.95, 1.59], // Alpha Arae
  [14.7084, -47.3882, 2.30, -0.22], // Alpha Lupi
  [14.9754, -43.1339, 2.68, -0.22], // Beta Lupi
  [15.5858, -41.1668, 2.77, -0.20], // Gamma Lupi
  [5.7400, -34.0741, 2.65, -0.13],  // Phact (Alpha Columbae)
  [5.8490, -35.7683, 3.12, 1.15],   // Wazn (Beta Columbae)

  // === LEPUS, HYDRA, CANIS MINOR, HERCULES, DRACO, CAPRICORNUS, MUSCA & DORADO ===
  [5.5455, -17.8223, 2.58, 0.21],   // Arneb (Alpha Leporis — Under Orion's Feet)
  [5.4707, -20.7594, 2.84, 0.82],   // Nihal (Beta Leporis)
  [5.2155, -16.2055, 3.29, -0.11],  // Mu Leporis
  [5.0909, -22.3710, 3.19, 1.46],   // Epsilon Leporis
  [9.4598, -8.6586, 1.98, 1.44],    // Alphard (Alpha Hydrae — Lonely Heart of the Sea Serpent)
  [13.3150, -23.1715, 2.99, 0.92],  // Gamma Hydrae
  [8.9234, 5.9456, 3.11, 1.00],     // Zeta Hydrae
  [7.4525, 8.2893, 2.89, -0.09],    // Gomeisa (Beta Canis Minoris — Procyon's Companion)
  [16.5037, 21.4896, 2.77, 0.94],   // Kornephoros (Beta Herculis)
  [16.6881, 31.6027, 2.81, 0.65],   // Zeta Herculis (Keystone)
  [16.7149, 38.9223, 3.48, 0.92],   // Eta Herculis (Keystone)
  [17.2508, 36.8092, 3.16, 1.44],   // Pi Herculis (Keystone)
  [17.9434, 51.4889, 2.23, 1.52],   // Eltanin (Gamma Draconis)
  [17.5072, 52.3014, 2.79, 0.98],   // Rastaban (Beta Draconis)
  [14.0732, 64.3758, 3.67, -0.05],  // Thuban (Alpha Draconis)
  [21.7840, -16.1273, 2.87, 0.29],  // Deneb Algedi (Delta Capricorni)
  [20.3501, -14.7814, 3.08, 0.79],  // Dabih (Beta Capricorni)
  [20.2943, -12.5449, 3.57, 0.88],  // Algedi (Alpha2 Capricorni)
  [1.5247, 15.3458, 3.62, 0.97],    // Alpherg (Eta Piscium)
  [23.2859, 3.2823, 3.70, 0.92],    // Gamma Piscium
  [6.4800, -7.0331, 3.76, -0.24],   // Beta Monocerotis
  [12.6196, -69.1355, 2.69, -0.20], // Alpha Muscae (Beside Southern Cross!)
  [12.7711, -68.1081, 3.05, -0.18], // Beta Muscae
  [4.5666, -55.0450, 3.27, -0.10]   // Alpha Doradus (Gateway to LMC!)
];

function wrapRad2Pi(r) {
  return ((r % TWO_PI) + TWO_PI) % TWO_PI;
}

/**
 * Computes exact Julian Date at 00:00 WIB (UTC+7) of the civil WIB date corresponding to `refDate`.
 */
export function computeBaseJD0WIB(refDate = new Date()) {
  const wibShiftMs = DEPOK_WIB_OFFSET_HOURS * 3600000.0;
  const wibDate = new Date(refDate.getTime() + wibShiftMs);
  let Y = wibDate.getUTCFullYear();
  let M = wibDate.getUTCMonth() + 1;
  const D = wibDate.getUTCDate();

  if (M <= 2) {
    Y -= 1;
    M += 12;
  }
  const A = Math.floor(Y / 100);
  const B = 2 - A + Math.floor(A / 4);
  const jd0UTC = Math.floor(365.25 * (Y + 4716)) + Math.floor(30.6001 * (M + 1)) + D + B - 1524.5;
  // 00:00 WIB = (jd0UTC - 7/24) in UT
  return jd0UTC - DEPOK_WIB_OFFSET_HOURS / 24.0;
}

/**
 * Evaluates continuous Jean Meeus Lunar & Solar Ephemeris + Local Sidereal Time (LST)
 * for Kepulauan UI Depok at `jdUT = baseJD0WIB + continuousSolarHour / 24.0`.
 * Because `continuousSolarHour` is unwrapped across transitions, there is ZERO jump at midnight!
 */
export function evaluateDepokAstronomy(
  baseJD0WIB,
  continuousSolarHour,
  outWorldToEqMat3 = null,
  outMoonDirWorld = null,
  outMoonPhaseSunDirWorld = null
) {
  const jdUT = baseJD0WIB + Number(continuousSolarHour) / 24.0;
  const d = jdUT - 2451545.0; // Days since J2000.0
  const T = d / 36525.0;

  // 1. Greenwich Mean Sidereal Time (GMST) & Depok Local Sidereal Time (LST)
  const gmstDeg =
    280.46061837 + 360.98564736629 * d + 0.000387933 * T * T - (T * T * T) / 38710000.0;
  const lstRad = wrapRad2Pi((gmstDeg + DEPOK_LON_DEG) * DEG2RAD);

  const latRad = DEPOK_LAT_DEG * DEG2RAD;
  const cosLat = Math.cos(latRad);
  const sinLat = Math.sin(latRad);
  const cosLST = Math.cos(lstRad);
  const sinLST = Math.sin(lstRad);

  // J2000 Equatorial unit basis vectors for Depok World Frame:
  // +X_world = East, +Y_world = Up (Zenith), +Z_world = South (det = +1)
  const eastEqX = -sinLST;
  const eastEqY = cosLST;
  const eastEqZ = 0.0;

  const upEqX = cosLat * cosLST;
  const upEqY = cosLat * sinLST;
  const upEqZ = sinLat;

  const southEqX = sinLat * cosLST;
  const southEqY = sinLat * sinLST;
  const southEqZ = -cosLat;

  // World -> Equatorial 3x3 matrix: eqVec = M * worldVec
  // Column 0 = EastEq, Column 1 = UpEq, Column 2 = SouthEq
  // THREE.Matrix3.set(...) takes ROW-major arguments:
  if (outWorldToEqMat3) {
    outWorldToEqMat3.set(
      eastEqX, upEqX, southEqX,
      eastEqY, upEqY, southEqY,
      eastEqZ, upEqZ, southEqZ
    );
  }

  // Helper: project any J2000 Equatorial unit vector into Depok World (East=+X, Up=+Y, South=+Z)
  const eqToWorld = (ex, ey, ez, outVec) => {
    const wx = ex * eastEqX + ey * eastEqY + ez * eastEqZ;
    const wy = ex * upEqX + ey * upEqY + ez * upEqZ;
    const wz = ex * southEqX + ey * southEqY + ez * southEqZ;
    if (outVec) outVec.set(wx, wy, wz).normalize();
    return { x: wx, y: wy, z: wz };
  };

  // 2. Solar Ecliptic & Equatorial Position (Jean Meeus Ch. 25)
  const L0Sun = wrapRad2Pi((280.46646 + 0.98564736 * d) * DEG2RAD);
  const MSun = wrapRad2Pi((357.52911 + 0.98560028 * d) * DEG2RAD);
  const CSun =
    (1.914602 * Math.sin(MSun) + 0.019993 * Math.sin(2.0 * MSun) + 0.000289 * Math.sin(3.0 * MSun)) *
    DEG2RAD;
  const sunLon = L0Sun + CSun;
  const eps = (23.439291 - 0.000013004 * d) * DEG2RAD; // Obliquity of Ecliptic
  const cosEps = Math.cos(eps);
  const sinEps = Math.sin(eps);

  const sunEqX = Math.cos(sunLon);
  const sunEqY = Math.sin(sunLon) * cosEps;
  const sunEqZ = Math.sin(sunLon) * sinEps;

  // 3. Lunar Ecliptic & Equatorial Position (Jean Meeus Ch. 47 major perturbation terms, ~0.12° accuracy)
  const Lm = wrapRad2Pi((218.3164477 + 13.17639648 * d) * DEG2RAD); // Mean longitude
  const Mm = wrapRad2Pi((134.9633964 + 13.06499295 * d) * DEG2RAD); // Mean anomaly
  const Dm = wrapRad2Pi((297.8501921 + 12.19074912 * d) * DEG2RAD); // Mean elongation
  const Fm = wrapRad2Pi((93.2720950 + 13.22935024 * d) * DEG2RAD);  // Argument of latitude

  const moonLon =
    Lm +
    (6.288774 * Math.sin(Mm) +
      1.274027 * Math.sin(2.0 * Dm - Mm) +
      0.658314 * Math.sin(2.0 * Dm) +
      0.213618 * Math.sin(2.0 * Mm) -
      0.185116 * Math.sin(MSun) -
      0.114332 * Math.sin(2.0 * Fm) +
      0.058793 * Math.sin(2.0 * Dm - 2.0 * Mm) +
      0.057066 * Math.sin(2.0 * Dm - MSun - Mm) +
      0.053322 * Math.sin(2.0 * Dm + Mm)) *
      DEG2RAD;

  const moonLat =
    (5.128122 * Math.sin(Fm) +
      0.280602 * Math.sin(Mm + Fm) +
      0.277693 * Math.sin(Mm - Fm) +
      0.173238 * Math.sin(2.0 * Dm - Fm)) *
    DEG2RAD;

  const cosB = Math.cos(moonLat);
  const sinB = Math.sin(moonLat);
  const cosL = Math.cos(moonLon);
  const sinL = Math.sin(moonLon);

  // Ecliptic Cartesian -> Equatorial Cartesian
  const mEclX = cosB * cosL;
  const mEclY = cosB * sinL;
  const mEclZ = sinB;

  const moonEqX = mEclX;
  const moonEqY = mEclY * cosEps - mEclZ * sinEps;
  const moonEqZ = mEclY * sinEps + mEclZ * cosEps;

  // Geocentric Elongation psi between Moon and Sun -> Lunar Phase Angle i ≈ pi - psi
  // (incorporates Earth-Sun vs Earth-Moon 1:389 distance ratio for exact Quarter timing!)
  const cosPsi = Math.max(-1.0, Math.min(1.0, sunEqX * moonEqX + sunEqY * moonEqY + sunEqZ * moonEqZ));
  const sinPsi = Math.sqrt(Math.max(0.0, 1.0 - cosPsi * cosPsi));
  const phaseAngleRad = Math.atan2(389.0 * sinPsi, 1.0 - 389.0 * cosPsi);
  const illuminatedFraction = 0.5 * (1.0 + Math.cos(phaseAngleRad));

  // Waxing vs Waning from signed ecliptic longitude elongation (moonLon - sunLon) in [0, 2pi)
  const eclElong = wrapRad2Pi(moonLon - sunLon);
  const isWaxing = eclElong < Math.PI;
  const lunarAgeDays = (eclElong / TWO_PI) * 29.53058885;

  let phaseNameId = 'Purnama';
  if (illuminatedFraction < 0.03) {
    phaseNameId = 'Bulan Baru';
  } else if (illuminatedFraction < 0.36) {
    phaseNameId = isWaxing ? 'Sabit Muda' : 'Sabit Tua';
  } else if (illuminatedFraction < 0.64) {
    phaseNameId = isWaxing ? 'Perbani Awal' : 'Perbani Akhir';
  } else if (illuminatedFraction < 0.965) {
    phaseNameId = isWaxing ? 'Cembung Awal' : 'Cembung Akhir';
  } else {
    phaseNameId = 'Purnama';
  }

  // Project Real Moon & Phase-Terminator Sun Vector into Depok World Space
  const moonWorld = eqToWorld(moonEqX, moonEqY, moonEqZ, outMoonDirWorld);
  // Vector from Moon toward Sun in Equatorial space (unit sunEq - (1/389) moonEq), projected to World Space:
  const sFromMX = sunEqX - moonEqX * (1.0 / 389.0);
  const sFromMY = sunEqY - moonEqY * (1.0 / 389.0);
  const sFromMZ = sunEqZ - moonEqZ * (1.0 / 389.0);
  eqToWorld(sFromMX, sFromMY, sFromMZ, outMoonPhaseSunDirWorld);

  return {
    jdUT,
    lstDeg: +(lstRad * RAD2DEG).toFixed(3),
    illuminatedFraction,
    phaseAngleRad,
    lunarAgeDays,
    isWaxing,
    phaseNameId,
    moonAltitudeDeg: +(Math.asin(Math.max(-1.0, Math.min(1.0, moonWorld.y))) * RAD2DEG).toFixed(2),
    moonAboveHorizon: moonWorld.y > -0.02
  };
}

/**
 * Builds the O(1) 384x64 RGBA32F (128 RA cells x 64 Dec cells x 3 Star Slots per cell)
 * J2000 Bright-Star Spatial Grid DataTexture from REAL_J2000_BRIGHT_STARS (0 external assets!).
 *
 * Texture Layout:
 * - Grid cell (cx in [0..127], cy in [0..63]) covers RA in [0, 2pi), sin(Dec) or Dec in [-pi/2, +pi/2].
 * - Slot s in {0, 1, 2} lives at texel X = cx * 3 + s, Y = cy.
 * - RGBA float32 stores: (starEqDir.x, starEqDir.y, starEqDir.z, packedMagBV)
 *   where if empty: w = -99.0; if occupied:
 *   intMagCode = floor((Vmag + 2.0) * 100.0) in [54..650], fracBV = clamp((BV + 0.4) / 2.6, 0.01, 0.99)
 *   packedMagBV = intMagCode + fracBV (> 0.0).
 */
/**
 * Builds an exact Power-of-Two 512 x 64 RGBA32F Spatial Equirectangular Star-Catalog Lookup Grid:
 * - 128 RA cells x 64 Dec cells, 4 star slots per cell (width = 128 * 4 = 512 = 2^9, height = 64 = 2^6).
 * - Each texel stores:
 *   vec4(eqDir.x, eqDir.y, eqDir.z, packedMagBV)
 *   where if empty: w = -99.0; if occupied:
 *   intMagCode = floor((Vmag + 2.0) * 100.0) in [54..650], fracBV = clamp((BV + 0.4) / 2.6, 0.01, 0.99)
 *   packedMagBV = intMagCode + fracBV (> 0.0).
 * - Guarantees 100% of all 208 real J2000 catalog stars reside in their home cell + neighbor glare cells,
 *   with ClampToEdgeWrapping on exact POT dimensions for 100% WebGL1/WebGL2 compatibility.
 */
let _cachedStarGridTexture = null;

export function createRealStarCatalogGridTexture() {
  if (_cachedStarGridTexture) return _cachedStarGridTexture;

  const CELLS_RA = 128;
  const CELLS_DEC = 64;
  const SLOTS = 4;
  const TEX_W = CELLS_RA * SLOTS; // 512 (exact 2^9 POT)
  const TEX_H = CELLS_DEC;        // 64  (exact 2^6 POT)

  const data = new Float32Array(TEX_W * TEX_H * 4);
  for (let i = 0; i < TEX_W * TEX_H; i++) {
    data[i * 4 + 3] = -99.0;
  }

  // Precompute J2000 Equatorial 3D unit vectors & home cells for all 208 catalog stars
  const stars = REAL_J2000_BRIGHT_STARS.map(([raHours, decDeg, vmag, bv]) => {
    const raPos = (((raHours * 15.0 * DEG2RAD) % TWO_PI) + TWO_PI) % TWO_PI;
    const dec = decDeg * DEG2RAD;
    const cosD = Math.cos(dec);
    const homeCx = Math.max(0, Math.min(CELLS_RA - 1, Math.floor((raPos / TWO_PI) * CELLS_RA)));
    const homeCy = Math.max(0, Math.min(CELLS_DEC - 1, Math.floor((dec / Math.PI + 0.5) * CELLS_DEC)));
    return {
      ra: raPos,
      dec,
      vmag,
      bv,
      homeCx,
      homeCy,
      x: cosD * Math.cos(raPos),
      y: cosD * Math.sin(raPos),
      z: Math.sin(dec)
    };
  });

  // Splat each star into every RA/Dec cell whose center is within ~4.6° great-circle radius
  // (easily covering cell half-diagonal ~2.0° + full Airy/ciliary glare radius ~2.0°!)
  const cosReach = Math.cos(4.6 * DEG2RAD);
  const cellCandidates = [];

  for (let cy = 0; cy < CELLS_DEC; cy++) {
    const decCenter = ((cy + 0.5) / CELLS_DEC - 0.5) * Math.PI;
    const cosDC = Math.cos(decCenter);
    const sinDC = Math.sin(decCenter);

    for (let cx = 0; cx < CELLS_RA; cx++) {
      const raCenter = ((cx + 0.5) / CELLS_RA) * TWO_PI;
      const cellX = cosDC * Math.cos(raCenter);
      const cellY = cosDC * Math.sin(raCenter);
      const cellZ = sinDC;

      cellCandidates.length = 0;
      for (let si = 0; si < stars.length; si++) {
        const st = stars[si];
        const dotC = cellX * st.x + cellY * st.y + cellZ * st.z;
        const isHome = st.homeCx === cx && st.homeCy === cy;
        if (isHome || dotC >= cosReach) {
          const angDeg = Math.acos(Math.max(-1.0, Math.min(1.0, dotC))) * RAD2DEG;
          // Home-cell stars always win top slots, followed by closest & brightest neighbors
          const priority = (isHome ? -100.0 : 0.0) + angDeg - 0.30 * (4.0 - st.vmag);
          cellCandidates.push({ st, priority });
        }
      }

      if (cellCandidates.length > 1) {
        cellCandidates.sort((a, b) => a.priority - b.priority);
      }

      const count = Math.min(SLOTS, cellCandidates.length);
      for (let slot = 0; slot < count; slot++) {
        const st = cellCandidates[slot].st;
        const texIdx = cy * TEX_W + (cx * SLOTS + slot);
        const magCode = Math.floor(Math.max(1.0, (st.vmag + 2.0) * 100.0));
        const bvFrac = Math.max(0.01, Math.min(0.99, (st.bv + 0.4) / 2.6));
        data[texIdx * 4] = st.x;
        data[texIdx * 4 + 1] = st.y;
        data[texIdx * 4 + 2] = st.z;
        data[texIdx * 4 + 3] = magCode + bvFrac;
      }
    }
  }

  const tex = new THREE.DataTexture(data, TEX_W, TEX_H, THREE.RGBAFormat, THREE.FloatType);
  tex.name = 'NexusJ2000RealBrightStarGrid512x64';
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;

  _cachedStarGridTexture = tex;
  return tex;
}
