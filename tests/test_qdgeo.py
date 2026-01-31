"""Test suite for QDGeo rigid-body molecular geometry construction"""

import os
import numpy as np
import qdgeo
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

# Create output directory for test geometries
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Helper Functions ---

def write_sdf(coords, mol, filename, title="Molecule"):
    """Write molecular geometry to SDF format using RDKit."""
    from rdkit.Chem import SDWriter
    mol_copy = Chem.Mol(mol)
    conf = Chem.Conformer(mol_copy.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, (x, y, z))
    mol_copy.AddConformer(conf, assignId=True)
    mol_copy.SetProp("_Name", title)
    filepath = os.path.join(OUTPUT_DIR, filename)
    writer = SDWriter(filepath)
    writer.write(mol_copy)
    writer.close()
    print(f"  → Saved: {filepath}")


def coords_to_conformer(coords, mol):
    """Create temporary conformer from coordinates."""
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, (x, y, z))
    mol.AddConformer(conf, assignId=True)
    return conf


def get_dihedral(coords, mol, i, j, k, l):
    """Calculate dihedral angle from coordinates using RDKit."""
    conf = coords_to_conformer(coords, mol)
    angle = rdMolTransforms.GetDihedralDeg(conf, i, j, k, l)
    mol.RemoveConformer(conf.GetId())
    return angle


def get_bond_length(coords, i, j):
    """Calculate bond length between two atoms."""
    return np.linalg.norm(coords[i] - coords[j])


def get_angle(coords, i, j, k):
    """Calculate angle i-j-k in degrees."""
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))


def dihedral_diff(angle1_deg, angle2_deg):
    """Calculate minimum difference between two dihedral angles accounting for periodicity."""
    diff = abs(angle1_deg - angle2_deg)
    diff = min(diff, 360 - diff)
    return diff


def check_geometry(coords, mol, bond_tol=0.1, angle_tol=5.0, clash_dist=1.0):
    """Check that geometry is reasonable. Returns list of issues."""
    issues = []
    n = mol.GetNumAtoms()
    
    # Check bond lengths
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dist = np.linalg.norm(coords[i] - coords[j])
        si = mol.GetAtomWithIdx(i).GetSymbol()
        sj = mol.GetAtomWithIdx(j).GetSymbol()
        # Expected bond lengths (approximate) - key is sorted alphabetically
        expected = {'BrC': 1.94, 'CC': 1.54, 'CH': 1.09, 'CCl': 1.79, 'CO': 1.43, 
                    'CN': 1.47, 'HO': 0.96, 'HN': 1.01, 'CF': 1.35, 'CS': 1.82}
        key = ''.join(sorted([si, sj]))
        exp = expected.get(key, 1.5)
        if abs(dist - exp) > bond_tol + 0.3:  # Allow flexibility for bond order/aromaticity
            issues.append(f"Bond {si}({i})-{sj}({j}): {dist:.2f}Å (expected ~{exp:.2f})")
    
    # Check bond angles
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
        if len(neighbors) < 2:
            continue
        hyb = atom.GetHybridization()
        if hyb == Chem.HybridizationType.SP3:
            expected_angle = 109.5
        elif hyb == Chem.HybridizationType.SP2:
            expected_angle = 120.0
        elif hyb == Chem.HybridizationType.SP:
            expected_angle = 180.0
        else:
            expected_angle = 109.5
        
        for j in range(len(neighbors)):
            for k in range(j+1, len(neighbors)):
                n1, n2 = neighbors[j], neighbors[k]
                ang = get_angle(coords, n1, i, n2)
                if abs(ang - expected_angle) > angle_tol:
                    s1 = mol.GetAtomWithIdx(n1).GetSymbol()
                    s2 = mol.GetAtomWithIdx(n2).GetSymbol()
                    sc = atom.GetSymbol()
                    issues.append(f"Angle {s1}({n1})-{sc}({i})-{s2}({n2}): {ang:.1f}° (expected ~{expected_angle:.1f}°)")
    
    # Check for clashes (non-bonded atoms too close)
    for i in range(n):
        for j in range(i+1, n):
            if mol.GetBondBetweenAtoms(i, j):
                continue
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < clash_dist:
                si = mol.GetAtomWithIdx(i).GetSymbol()
                sj = mol.GetAtomWithIdx(j).GetSymbol()
                issues.append(f"Clash {si}({i})-{sj}({j}): {dist:.2f}Å")
    
    return issues


def get_atoms_by_symbol(mol, symbol):
    """Get list of atom indices with given symbol."""
    return [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetSymbol() == symbol]


# --- Basic Molecule Tests ---

def test_water():
    mol = Chem.AddHs(Chem.MolFromSmiles('O'))
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (3, 3)
    
    # Check O-H bond lengths (~0.96 Å)
    o_idx = get_atoms_by_symbol(mol, 'O')[0]
    h_indices = get_atoms_by_symbol(mol, 'H')
    for h_idx in h_indices:
        bond_len = get_bond_length(coords, o_idx, h_idx)
        assert 0.9 < bond_len < 1.1, f"O-H bond length {bond_len:.2f} out of range"
    
    # Check H-O-H angle (~104-109°)
    angle = get_angle(coords, h_indices[0], o_idx, h_indices[1])
    assert 100 < angle < 115, f"H-O-H angle {angle:.1f}° out of range"
    
    write_sdf(coords, mol, "water.sdf", "Water")


def test_ethane():
    mol = Chem.AddHs(Chem.MolFromSmiles('CC'))
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (8, 3)
    
    # Check C-C bond length (~1.54 Å)
    c_atoms = get_atoms_by_symbol(mol, 'C')
    cc_len = get_bond_length(coords, c_atoms[0], c_atoms[1])
    assert 1.4 < cc_len < 1.7, f"C-C bond length {cc_len:.2f} out of range"
    
    write_sdf(coords, mol, "ethane.sdf", "Ethane")


def test_propane():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCC'))
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (11, 3)
    write_sdf(coords, mol, "propane.sdf", "Propane")


def test_cyclopropane():
    mol = Chem.AddHs(Chem.MolFromSmiles('C1CC1'))
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (9, 3)
    
    # Check that ring carbons form approximate equilateral triangle
    c_atoms = get_atoms_by_symbol(mol, 'C')
    d01 = get_bond_length(coords, c_atoms[0], c_atoms[1])
    d12 = get_bond_length(coords, c_atoms[1], c_atoms[2])
    d20 = get_bond_length(coords, c_atoms[2], c_atoms[0])
    assert abs(d01 - d12) < 0.2, "Ring bonds not equal"
    assert abs(d12 - d20) < 0.2, "Ring bonds not equal"
    
    write_sdf(coords, mol, "cyclopropane.sdf", "Cyclopropane")


def test_butane():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (14, 3)
    write_sdf(coords, mol, "butane.sdf", "Butane")


def test_cyclohexane():
    mol = Chem.AddHs(Chem.MolFromSmiles('C1CCCCC1'))
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (18, 3)
    write_sdf(coords, mol, "cyclohexane.sdf", "Cyclohexane")


def test_ethanol():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (9, 3)
    write_sdf(coords, mol, "ethanol.sdf", "Ethanol")


def test_benzene():
    mol = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (12, 3)
    
    # Check planarity - all carbons should be coplanar
    c_atoms = get_atoms_by_symbol(mol, 'C')
    c_coords = coords[c_atoms]
    
    # Fit a plane and check deviation
    centroid = c_coords.mean(axis=0)
    centered = c_coords - centroid
    _, s, vh = np.linalg.svd(centered)
    # Smallest singular value corresponds to out-of-plane direction
    max_deviation = s[-1]
    assert max_deviation < 0.5, f"Benzene not planar, deviation={max_deviation:.2f}"
    
    write_sdf(coords, mol, "benzene.sdf", "Benzene")


# --- Dihedral Constraint Tests ---

def test_butane_dihedral():
    """Test butane with specified dihedral angle."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    
    # Build with gauche conformation (60°)
    target_dihedral = 60.0
    torsions = {(c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): target_dihedral}
    coords = qdgeo.build_mol(mol, torsions=torsions, verbose=1)
    
    assert coords.shape == (14, 3)
    
    # Check dihedral angle
    actual_dihedral = get_dihedral(coords, mol, *c_atoms)
    assert dihedral_diff(actual_dihedral, target_dihedral) < 15.0, \
        f"Dihedral {actual_dihedral:.1f}° differs from target {target_dihedral}°"
    
    write_sdf(coords, mol, "butane_gauche.sdf", "Butane Gauche")


def test_butane_anti():
    """Test butane with anti (180°) dihedral."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    
    # Build with anti conformation (180°) - this is the default
    coords = qdgeo.build_mol(mol, verbose=1)
    
    actual_dihedral = get_dihedral(coords, mol, *c_atoms)
    # Should be close to 180° (anti) by default
    assert dihedral_diff(actual_dihedral, 180.0) < 30.0, \
        f"Default dihedral {actual_dihedral:.1f}° should be near 180°"
    
    write_sdf(coords, mol, "butane_anti.sdf", "Butane Anti")


def test_br_cc_cl_default():
    """Test Br-C-C-Cl default geometry (anti conformation)."""
    mol = Chem.AddHs(Chem.MolFromSmiles('BrCCCl'))
    br = get_atoms_by_symbol(mol, 'Br')[0]
    c_atoms = get_atoms_by_symbol(mol, 'C')
    cl = get_atoms_by_symbol(mol, 'Cl')[0]
    
    coords = qdgeo.build_mol(mol, verbose=1)
    
    # Default should be anti (180°)
    actual = get_dihedral(coords, mol, br, c_atoms[0], c_atoms[1], cl)
    assert dihedral_diff(actual, 180.0) < 15.0, f"Dihedral {actual:.1f}° != 180°"
    
    # Check full geometry - all bond angles must be tetrahedral
    issues = check_geometry(coords, mol, angle_tol=3.0)
    assert len(issues) == 0, f"Geometry issues:\n" + "\n".join(issues)
    
    write_sdf(coords, mol, "br_cc_cl_anti.sdf", "Br-C-C-Cl (anti)")


def test_br_cc_cl_dihedral():
    """Test Br-C-C-Cl with dihedral constraint and full geometry check."""
    mol = Chem.AddHs(Chem.MolFromSmiles('BrCCCl'))
    br = get_atoms_by_symbol(mol, 'Br')[0]
    c_atoms = get_atoms_by_symbol(mol, 'C')
    cl = get_atoms_by_symbol(mol, 'Cl')[0]
    
    target = 60.0
    torsions = {(br, c_atoms[0], c_atoms[1], cl): target}
    coords = qdgeo.build_mol(mol, torsions=torsions, verbose=1)
    
    # Check dihedral
    actual = get_dihedral(coords, mol, br, c_atoms[0], c_atoms[1], cl)
    assert dihedral_diff(actual, target) < 15.0, f"Dihedral {actual:.1f}° != {target:.1f}°"
    
    # Check full geometry - all bond angles must be correct
    issues = check_geometry(coords, mol, angle_tol=3.0)
    assert len(issues) == 0, f"Geometry issues:\n" + "\n".join(issues)
    
    write_sdf(coords, mol, "br_cc_cl.sdf", "Br-C-C-Cl")


def test_pentane_multiple_dihedrals():
    """Test pentane with multiple dihedral constraints."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCCC'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    
    torsions = {
        (c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 60.0,
        (c_atoms[1], c_atoms[2], c_atoms[3], c_atoms[4]): 120.0,
    }
    coords = qdgeo.build_mol(mol, torsions=torsions, verbose=1)
    
    d1 = get_dihedral(coords, mol, *c_atoms[:4])
    d2 = get_dihedral(coords, mol, *c_atoms[1:])
    
    assert dihedral_diff(d1, 60.0) < 20.0
    assert dihedral_diff(d2, 120.0) < 20.0
    write_sdf(coords, mol, "pentane_dihedrals.sdf", "Pentane Multiple Dihedrals")


def test_hexane_all_dihedrals():
    """Test hexane with constraints on all three C-C-C-C dihedrals."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCCCC'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    
    targets = [180.0, 60.0, -60.0]
    torsions = {
        (c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): targets[0],
        (c_atoms[1], c_atoms[2], c_atoms[3], c_atoms[4]): targets[1],
        (c_atoms[2], c_atoms[3], c_atoms[4], c_atoms[5]): targets[2],
    }
    coords = qdgeo.build_mol(mol, torsions=torsions, verbose=1)
    
    for idx, (i, j, k, l) in enumerate([(0,1,2,3), (1,2,3,4), (2,3,4,5)]):
        d = get_dihedral(coords, mol, c_atoms[i], c_atoms[j], c_atoms[k], c_atoms[l])
        assert dihedral_diff(d, targets[idx]) < 25.0
    write_sdf(coords, mol, "hexane_all_dihedrals.sdf", "Hexane All Dihedrals")


def test_butanol_dihedrals():
    """Test 1-butanol with dihedral constraints."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCCO'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    o_atoms = get_atoms_by_symbol(mol, 'O')
    
    torsions = {
        (c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 180.0,
        (c_atoms[1], c_atoms[2], c_atoms[3], o_atoms[0]): 60.0,
    }
    coords = qdgeo.build_mol(mol, torsions=torsions, verbose=1)
    
    d1 = get_dihedral(coords, mol, *c_atoms)
    d2 = get_dihedral(coords, mol, c_atoms[1], c_atoms[2], c_atoms[3], o_atoms[0])
    assert dihedral_diff(d1, 180.0) < 20.0
    assert dihedral_diff(d2, 60.0) < 20.0
    write_sdf(coords, mol, "butanol_dihedrals.sdf", "Butanol Dihedrals")


def test_butadiene_conjugated():
    """Test 1,3-butadiene with dihedral constraint."""
    mol = Chem.AddHs(Chem.MolFromSmiles('C=CC=C'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    
    # s-cis conformation
    torsions = {(c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 0.0}
    coords = qdgeo.build_mol(mol, torsions=torsions, verbose=1)
    
    d = get_dihedral(coords, mol, *c_atoms)
    assert dihedral_diff(d, 0.0) < 35.0
    write_sdf(coords, mol, "butadiene_cis.sdf", "Butadiene s-cis")


def test_propanediol_multiple_oh():
    """Test 1,3-propanediol with dihedral constraints on both C-O bonds."""
    mol = Chem.AddHs(Chem.MolFromSmiles('OCCCO'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    o_atoms = get_atoms_by_symbol(mol, 'O')
    
    torsions = {
        (o_atoms[0], c_atoms[0], c_atoms[1], c_atoms[2]): 60.0,
        (c_atoms[0], c_atoms[1], c_atoms[2], o_atoms[1]): -60.0,
    }
    coords = qdgeo.build_mol(mol, torsions=torsions, verbose=1)
    
    d1 = get_dihedral(coords, mol, o_atoms[0], c_atoms[0], c_atoms[1], c_atoms[2])
    d2 = get_dihedral(coords, mol, c_atoms[0], c_atoms[1], c_atoms[2], o_atoms[1])
    assert dihedral_diff(d1, 60.0) < 20.0
    assert dihedral_diff(d2, -60.0) < 20.0
    write_sdf(coords, mol, "propanediol_dihedrals.sdf", "Propanediol Dihedrals")


# --- Implicit Hydrogen Tests ---

def test_ethane_implicit_hydrogens():
    mol = Chem.MolFromSmiles('CC')  # Don't add hydrogens
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (2, 3)
    write_sdf(coords, mol, "ethane_implicit.sdf", "Ethane (Implicit H)")


def test_benzene_implicit_hydrogens():
    mol = Chem.MolFromSmiles('c1ccccc1')  # Don't add hydrogens
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (6, 3)
    write_sdf(coords, mol, "benzene_implicit.sdf", "Benzene (Implicit H)")


def test_butane_implicit_hydrogens():
    mol = Chem.MolFromSmiles('CCCC')  # Don't add hydrogens
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (4, 3)
    write_sdf(coords, mol, "butane_implicit.sdf", "Butane (Implicit H)")


def test_butane_dihedral_implicit_hydrogens():
    """Test butane with dihedral constraint and implicit hydrogens."""
    mol = Chem.MolFromSmiles('CCCC')
    c_atoms = list(range(mol.GetNumAtoms()))
    
    torsions = {(c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 60.0}
    coords = qdgeo.build_mol(mol, torsions=torsions, verbose=1)
    
    assert coords.shape == (4, 3)
    dihedral_deg = get_dihedral(coords, mol, *c_atoms)
    assert dihedral_diff(dihedral_deg, 60.0) < 20.0
    write_sdf(coords, mol, "butane_dihedral_implicit.sdf", "Butane with Dihedral (Implicit H)")


# --- Ring System Tests ---

def test_pyrrole():
    """Test pyrrole (5-membered aromatic ring)."""
    mol = Chem.AddHs(Chem.MolFromSmiles('c1cc[nH]c1'))
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (10, 3)
    write_sdf(coords, mol, "pyrrole.sdf", "Pyrrole")


def test_isopentane_branched():
    """Test isopentane (2-methylbutane) - branched molecule."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CC(C)CC'))
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (17, 3)  # 5 C + 12 H
    write_sdf(coords, mol, "isopentane.sdf", "Isopentane")


# --- Large Molecule Tests ---

def test_cholesterol():
    """Test cholesterol (C27H46O, MW ~387) - a steroid."""
    smiles = 'CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C'
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol.GetNumAtoms() == 74
    
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (74, 3)
    write_sdf(coords, mol, "cholesterol.sdf", "Cholesterol")


def test_simvastatin():
    """Test simvastatin (C25H38O5, MW ~418) - a statin drug."""
    smiles = 'CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C'
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol.GetNumAtoms() == 68
    
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (68, 3)
    write_sdf(coords, mol, "simvastatin.sdf", "Simvastatin")


def test_cortisol():
    """Test cortisol (C21H30O5, MW ~362) - a stress hormone."""
    smiles = 'CC12CCC(=O)C=C1CCC3C2C(CC4(C3CCC4(C(=O)CO)O)C)O'
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol.GetNumAtoms() == 56
    
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (56, 3)
    write_sdf(coords, mol, "cortisol.sdf", "Cortisol")


def test_tamoxifen():
    """Test tamoxifen (C26H29NO, MW ~371) - a breast cancer drug."""
    smiles = 'CC/C(=C(\\c1ccccc1)/c2ccc(cc2)OCCN(C)C)/c3ccccc3'
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol.GetNumAtoms() == 57
    
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (57, 3)
    write_sdf(coords, mol, "tamoxifen.sdf", "Tamoxifen")


def test_caffeine():
    """Test caffeine (C8H10N4O2, MW ~194)."""
    smiles = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol.GetNumAtoms() == 24
    
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (24, 3)
    write_sdf(coords, mol, "caffeine.sdf", "Caffeine")


def test_squalane():
    """Test squalane (C30H62, MW ~423) - highly branched acyclic molecule."""
    smiles = 'CC(C)CCCC(C)CCCC(C)CCCCC(C)CCCC(C)CCCC(C)C'
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol.GetNumAtoms() == 92
    
    coords = qdgeo.build_mol(mol, verbose=1)
    assert coords.shape == (92, 3)
    write_sdf(coords, mol, "squalane.sdf", "Squalane")


# --- Low-Level API Test ---

def test_low_level_api():
    """Test using the low-level C++ API directly."""
    # Build a simple water molecule
    n_atoms = 3  # O, H, H
    bonds = [
        (0, 1, 0.96),  # O-H
        (0, 2, 0.96),  # O-H
    ]
    angles = [
        (1, 0, 2, np.deg2rad(104.5)),  # H-O-H angle
    ]
    
    coords = qdgeo.build_molecule(n_atoms, bonds, angles)
    assert coords.shape == (3, 3)
    
    # Check bond lengths
    d1 = np.linalg.norm(coords[0] - coords[1])
    d2 = np.linalg.norm(coords[0] - coords[2])
    assert abs(d1 - 0.96) < 0.1
    assert abs(d2 - 0.96) < 0.1


def test_builder_class():
    """Test using the MoleculeBuilder class directly."""
    builder = qdgeo.MoleculeBuilder(4)  # 4 atoms for butane backbone
    
    # Add C-C bonds
    builder.add_bond(0, 1, 1.54)
    builder.add_bond(1, 2, 1.54)
    builder.add_bond(2, 3, 1.54)
    
    # Set angles
    tetrahedral = np.arccos(-1/3)
    builder.set_angle(0, 1, 2, tetrahedral)
    builder.set_angle(1, 2, 3, tetrahedral)
    
    # Set torsion (gauche)
    builder.set_torsion(0, 1, 2, 3, np.deg2rad(60))
    
    coords = builder.build()
    assert coords.shape == (4, 3)
