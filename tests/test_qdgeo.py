"""Test suite for QDGeo molecular geometry optimization"""

import os
import numpy as np
import qdgeo
from qdgeo import optimize_mol
from qdgeo.optimize_mol import BOND_LENGTHS, ANGLE
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdFMCS

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
    return np.deg2rad(angle)


def dihedral_diff(angle1_deg, angle2_deg):
    """Calculate minimum difference between two dihedral angles accounting for periodicity."""
    diff = abs(angle1_deg - angle2_deg)
    diff = min(diff, 360 - diff)
    diff_180 = abs(angle1_deg + 180 - angle2_deg)
    diff_180 = min(diff_180, 360 - diff_180)
    return min(diff, diff_180)


def get_atoms_by_symbol(mol, symbol):
    """Get list of atom indices with given symbol."""
    return [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetSymbol() == symbol]


# --- Basic Molecule Tests ---

def test_water():
    mol = Chem.AddHs(Chem.MolFromSmiles('O'))
    coords = qdgeo.optimize_mol(mol, verbose=0)
    assert coords.shape == (3, 3)
    write_sdf(coords, mol, "water.sdf", "Water")


def test_ethane():
    mol = Chem.AddHs(Chem.MolFromSmiles('CC'))
    coords = qdgeo.optimize_mol(mol, verbose=0)
    assert coords.shape == (8, 3)
    write_sdf(coords, mol, "ethane.sdf", "Ethane")


def test_propane():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCC'))
    coords = qdgeo.optimize_mol(mol, verbose=0)
    assert coords.shape == (11, 3)
    write_sdf(coords, mol, "propane.sdf", "Propane")


def test_cyclopropane():
    mol = Chem.AddHs(Chem.MolFromSmiles('C1CC1'))
    coords = qdgeo.optimize_mol(mol, verbose=1)
    assert coords.shape == (9, 3)
    write_sdf(coords, mol, "cyclopropane.sdf", "Cyclopropane")


def test_butane():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
    coords = qdgeo.optimize_mol(mol, repulsion_k=0.1, repulsion_cutoff=3.0, verbose=1)
    assert coords.shape == (14, 3)
    write_sdf(coords, mol, "butane.sdf", "Butane")


def test_cyclohexane():
    mol = Chem.AddHs(Chem.MolFromSmiles('C1CCCCC1'))
    coords = qdgeo.optimize_mol(mol, repulsion_k=0.2, repulsion_cutoff=3.5, verbose=1)
    assert coords.shape == (18, 3)
    write_sdf(coords, mol, "cyclohexane.sdf", "Cyclohexane")


def test_ethanol():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    coords = qdgeo.optimize_mol(mol, verbose=1)
    assert coords.shape == (9, 3)
    write_sdf(coords, mol, "ethanol.sdf", "Ethanol")


def test_benzene():
    mol = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))
    coords = qdgeo.optimize_mol(mol, verbose=1, repulsion_k=0.1, repulsion_cutoff=3.0, maxeval=10000)
    assert coords.shape == (12, 3)
    write_sdf(coords, mol, "benzene.sdf", "Benzene")
    
    # Check planarity
    c_atoms = get_atoms_by_symbol(mol, 'C')
    if len(c_atoms) >= 4:
        conf = coords_to_conformer(coords, mol)
        dihedral = rdMolTransforms.GetDihedralDeg(conf, c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3])
        mol.RemoveConformer(conf.GetId())
        dihedral_abs = abs(dihedral) % 180
        assert dihedral_abs < 60.0 or abs(dihedral_abs - 180) < 60.0


# --- Dihedral Constraint Tests ---

def test_butane_dihedral():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    dihedral_dict = {(c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 60.0}
    
    coords = qdgeo.optimize_mol(mol, dihedral=dihedral_dict, dihedral_k=5.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0)
    assert coords.shape == (14, 3)
    
    dihedral_deg = np.rad2deg(get_dihedral(coords, mol, *c_atoms))
    assert dihedral_diff(dihedral_deg, 60.0) < 15.0
    write_sdf(coords, mol, "butane_dihedral.sdf", "Butane with Dihedral")


def test_ethanol_dihedral():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    o_atoms = get_atoms_by_symbol(mol, 'O')
    
    if len(c_atoms) >= 2 and len(o_atoms) >= 1:
        h_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(c_atoms[0]).GetNeighbors() 
                       if mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() == 'H']
        if h_neighbors:
            dihedral_dict = {(h_neighbors[0], c_atoms[0], c_atoms[1], o_atoms[0]): 180.0}
            coords = qdgeo.optimize_mol(mol, dihedral=dihedral_dict, dihedral_k=5.0,
                                         repulsion_k=0.1, repulsion_cutoff=3.0)
            assert coords.shape == (9, 3)
            write_sdf(coords, mol, "ethanol_dihedral.sdf", "Ethanol with Dihedral")


def test_br_cc_cl_dihedral():
    mol = Chem.AddHs(Chem.MolFromSmiles('BrCCCl'))
    br, c1, c2, cl = get_atoms_by_symbol(mol, 'Br')[0], *get_atoms_by_symbol(mol, 'C'), get_atoms_by_symbol(mol, 'Cl')[0]
    
    dihedral_dict = {(br, c1, c2, cl): 60.0}
    coords = qdgeo.optimize_mol(mol, dihedral=dihedral_dict, dihedral_k=5.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=0)
    
    dihedral_deg = np.rad2deg(get_dihedral(coords, mol, br, c1, c2, cl))
    assert dihedral_diff(dihedral_deg, 60.0) < 15.0
    write_sdf(coords, mol, "br_cc_cl.sdf", "Br-C-C-Cl")


def test_pentane_multiple_dihedrals():
    """Test pentane with multiple dihedral constraints."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCCC'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    
    dihedral_dict = {
        (c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 60.0,
        (c_atoms[1], c_atoms[2], c_atoms[3], c_atoms[4]): 120.0,
    }
    coords = qdgeo.optimize_mol(mol, dihedral=dihedral_dict, dihedral_k=5.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=0)
    
    d1 = np.rad2deg(get_dihedral(coords, mol, *c_atoms[:4]))
    d2 = np.rad2deg(get_dihedral(coords, mol, *c_atoms[1:]))
    assert dihedral_diff(d1, 60.0) < 15.0
    assert dihedral_diff(d2, 120.0) < 15.0
    write_sdf(coords, mol, "pentane_dihedrals.sdf", "Pentane Multiple Dihedrals")


def test_hexane_all_dihedrals():
    """Test hexane with constraints on all three C-C-C-C dihedrals."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCCCC'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    
    targets = [180.0, 60.0, -60.0]
    dihedral_dict = {
        (c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): targets[0],
        (c_atoms[1], c_atoms[2], c_atoms[3], c_atoms[4]): targets[1],
        (c_atoms[2], c_atoms[3], c_atoms[4], c_atoms[5]): targets[2],
    }
    coords = qdgeo.optimize_mol(mol, dihedral=dihedral_dict, dihedral_k=5.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=0)
    
    for idx, (i, j, k, l) in enumerate([(0,1,2,3), (1,2,3,4), (2,3,4,5)]):
        d = np.rad2deg(get_dihedral(coords, mol, c_atoms[i], c_atoms[j], c_atoms[k], c_atoms[l]))
        assert dihedral_diff(d, targets[idx]) < 15.0
    write_sdf(coords, mol, "hexane_all_dihedrals.sdf", "Hexane All Dihedrals")


def test_butanol_dihedrals():
    """Test 1-butanol with dihedral constraints on both C-C and C-O bonds."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCCO'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    o_atoms = get_atoms_by_symbol(mol, 'O')
    
    dihedral_dict = {
        (c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 180.0,
        (c_atoms[1], c_atoms[2], c_atoms[3], o_atoms[0]): 60.0,
    }
    coords = qdgeo.optimize_mol(mol, dihedral=dihedral_dict, dihedral_k=5.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=0)
    
    d1 = np.rad2deg(get_dihedral(coords, mol, *c_atoms))
    d2 = np.rad2deg(get_dihedral(coords, mol, c_atoms[1], c_atoms[2], c_atoms[3], o_atoms[0]))
    assert dihedral_diff(d1, 180.0) < 15.0
    assert dihedral_diff(d2, 60.0) < 15.0
    write_sdf(coords, mol, "butanol_dihedrals.sdf", "Butanol Dihedrals")


def test_butadiene_conjugated():
    """Test 1,3-butadiene with dihedral constraint across conjugated system."""
    mol = Chem.AddHs(Chem.MolFromSmiles('C=CC=C'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    
    dihedral_dict = {(c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 0.0}
    coords = qdgeo.optimize_mol(mol, dihedral=dihedral_dict, dihedral_k=10.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=0)
    
    d = np.rad2deg(get_dihedral(coords, mol, *c_atoms))
    assert dihedral_diff(d, 0.0) < 35.0
    write_sdf(coords, mol, "butadiene_cis.sdf", "Butadiene s-cis")


def test_propanediol_multiple_oh():
    """Test 1,3-propanediol with dihedral constraints on both C-O bonds."""
    mol = Chem.AddHs(Chem.MolFromSmiles('OCCCO'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    o_atoms = get_atoms_by_symbol(mol, 'O')
    
    dihedral_dict = {
        (o_atoms[0], c_atoms[0], c_atoms[1], c_atoms[2]): 60.0,
        (c_atoms[0], c_atoms[1], c_atoms[2], o_atoms[1]): -60.0,
    }
    coords = qdgeo.optimize_mol(mol, dihedral=dihedral_dict, dihedral_k=5.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=0)
    
    d1 = np.rad2deg(get_dihedral(coords, mol, o_atoms[0], c_atoms[0], c_atoms[1], c_atoms[2]))
    d2 = np.rad2deg(get_dihedral(coords, mol, c_atoms[0], c_atoms[1], c_atoms[2], o_atoms[1]))
    assert dihedral_diff(d1, 60.0) < 15.0
    assert dihedral_diff(d2, -60.0) < 15.0
    write_sdf(coords, mol, "propanediol_dihedrals.sdf", "Propanediol Dihedrals")


# --- Template Restraint Tests ---

def test_template_same_molecule():
    """Test template restraints with same molecule (butane -> butane)."""
    template = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
    AllChem.EmbedMolecule(template, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(template)
    
    c_atoms_template = get_atoms_by_symbol(template, 'C')
    template_dihedral_deg = np.rad2deg(get_dihedral(
        template.GetConformer().GetPositions(), template, *c_atoms_template))
    
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
    coords = qdgeo.optimize_mol(mol, template=template, template_coordinate_k=10.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=0)
    
    c_atoms = get_atoms_by_symbol(mol, 'C')
    result_dihedral_deg = np.rad2deg(get_dihedral(coords, mol, *c_atoms))
    assert dihedral_diff(result_dihedral_deg, template_dihedral_deg) < 20.0
    write_sdf(coords, mol, "butane_template.sdf", "Butane with Template")


def test_template_substructure():
    """Test template with substructure (propane template -> pentane target)."""
    template = Chem.AddHs(Chem.MolFromSmiles('CCC'))
    AllChem.EmbedMolecule(template, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(template)
    
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCCC'))
    coords = qdgeo.optimize_mol(mol, template=template, template_coordinate_k=8.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=1)
    
    assert coords.shape == (mol.GetNumAtoms(), 3)
    write_sdf(coords, mol, "pentane_propane_template.sdf", "Pentane with Propane Template")


def test_template_with_heteroatoms():
    """Test template with heteroatoms (ethanol template -> butanol target)."""
    template = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    AllChem.EmbedMolecule(template, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(template)
    
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCCO'))
    coords = qdgeo.optimize_mol(mol, template=template, template_coordinate_k=8.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=1)
    
    assert coords.shape == (mol.GetNumAtoms(), 3)
    write_sdf(coords, mol, "butanol_ethanol_template.sdf", "Butanol with Ethanol Template")


def test_template_without_conformer():
    """Test that template without conformer is handled gracefully."""
    template = Chem.AddHs(Chem.MolFromSmiles('CCC'))  # No EmbedMolecule call
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
    
    coords = qdgeo.optimize_mol(mol, template=template, template_coordinate_k=5.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=1)
    assert coords.shape == (mol.GetNumAtoms(), 3)


def test_template_no_substructure_match():
    """Test that non-matching template is handled gracefully."""
    template = Chem.AddHs(Chem.MolFromSmiles('CCN'))
    AllChem.EmbedMolecule(template, randomSeed=42)
    
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))  # No nitrogen
    coords = qdgeo.optimize_mol(mol, template=template, template_coordinate_k=5.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=1)
    assert coords.shape == (mol.GetNumAtoms(), 3)


def test_mcs_butane_to_hexane():
    """Test MCS template matching: butane template -> hexane target."""
    template = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
    AllChem.EmbedMolecule(template, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(template)
    
    c_atoms_template = get_atoms_by_symbol(template, 'C')
    template_dihedral_deg = rdMolTransforms.GetDihedralDeg(
        template.GetConformer(), *c_atoms_template)
    
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCCCC'))
    coords = qdgeo.optimize_mol(mol, template=template, template_coordinate_k=10.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=1)
    
    assert coords.shape == (mol.GetNumAtoms(), 3)
    write_sdf(coords, mol, "hexane_butane_mcs.sdf", "Hexane with Butane MCS Template")


def test_mcs_benzene_to_toluene():
    """Test MCS template matching: benzene template -> toluene target."""
    template = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))
    AllChem.EmbedMolecule(template, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(template)
    
    mol = Chem.AddHs(Chem.MolFromSmiles('Cc1ccccc1'))
    coords = qdgeo.optimize_mol(mol, template=template, template_coordinate_k=10.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=1)
    
    assert coords.shape == (mol.GetNumAtoms(), 3)
    write_sdf(coords, mol, "toluene_benzene_mcs.sdf", "Toluene with Benzene MCS Template")


# --- Implicit Hydrogen Tests ---

def test_ethane_implicit_hydrogens():
    mol = Chem.MolFromSmiles('CC')  # Don't add hydrogens
    coords = qdgeo.optimize_mol(mol, verbose=1)
    assert coords.shape == (2, 3)
    write_sdf(coords, mol, "ethane_implicit.sdf", "Ethane (Implicit H)")


def test_benzene_implicit_hydrogens():
    mol = Chem.MolFromSmiles('c1ccccc1')  # Don't add hydrogens
    coords = qdgeo.optimize_mol(mol, verbose=1, repulsion_k=0.1, repulsion_cutoff=3.0)
    assert coords.shape == (6, 3)
    write_sdf(coords, mol, "benzene_implicit.sdf", "Benzene (Implicit H)")


def test_butane_implicit_hydrogens():
    mol = Chem.MolFromSmiles('CCCC')  # Don't add hydrogens
    coords = qdgeo.optimize_mol(mol, verbose=1)
    assert coords.shape == (4, 3)
    write_sdf(coords, mol, "butane_implicit.sdf", "Butane (Implicit H)")


def test_butane_dihedral_implicit_hydrogens():
    """Test butane with dihedral constraint and implicit hydrogens."""
    mol = Chem.MolFromSmiles('CCCC')  # Don't add hydrogens
    c_atoms = list(range(mol.GetNumAtoms()))
    
    dihedral_dict = {(c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 60.0}
    coords = qdgeo.optimize_mol(mol, dihedral=dihedral_dict, dihedral_k=5.0,
                                 repulsion_k=0.1, repulsion_cutoff=3.0, verbose=1)
    
    assert coords.shape == (4, 3)
    dihedral_deg = np.rad2deg(get_dihedral(coords, mol, *c_atoms))
    assert dihedral_diff(dihedral_deg, 60.0) < 15.0
    write_sdf(coords, mol, "butane_dihedral_implicit.sdf", "Butane with Dihedral (Implicit H)")


# --- Special Geometry Tests ---

def test_pyrrole_planarity():
    """Test pyrrole with explicit hydrogens - verify correct angles for 5-membered ring."""
    mol = Chem.AddHs(Chem.MolFromSmiles('c1cc[nH]c1'))
    coords = qdgeo.optimize_mol(mol, verbose=1, repulsion_k=0.1, repulsion_cutoff=3.0, maxeval=10000)
    assert coords.shape == (10, 3)
    
    conf = coords_to_conformer(coords, mol)
    ring_info = mol.GetRingInfo()
    five_rings = [r for r in ring_info.AtomRings() if len(r) == 5]
    five_ring_atoms = set(a for r in five_rings for a in r)
    
    max_error = 0.0
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetHybridization() != Chem.HybridizationType.SP2:
            continue
        neighbors = [a.GetIdx() for a in atom.GetNeighbors()]
        if len(neighbors) != 3 or atom_idx not in five_ring_atoms:
            continue
        
        h_neighbors = [n for n in neighbors if mol.GetAtomWithIdx(n).GetSymbol() == 'H']
        if not h_neighbors:
            continue
        
        h_idx = h_neighbors[0]
        pos_center = np.array(conf.GetAtomPosition(atom_idx))
        pos_h = np.array(conf.GetAtomPosition(h_idx))
        
        for other in neighbors:
            if other == h_idx:
                continue
            pos_other = np.array(conf.GetAtomPosition(other))
            v1, v2 = pos_other - pos_center, pos_h - pos_center
            cos_angle = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
            angle_deg = np.rad2deg(np.arccos(cos_angle))
            max_error = max(max_error, abs(angle_deg - 126.0))  # Non-ring angles should be 126°
    
    mol.RemoveConformer(conf.GetId())
    assert max_error < 1.0, f"Angle error too large: {max_error:.2f}°"
    write_sdf(coords, mol, "pyrrole_planar.sdf", "Pyrrole Planar")


def test_isopentane_branched():
    """Test isopentane (2-methylbutane) with dihedral constraint on branched molecule."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CC(C)CC'))
    c_atoms = get_atoms_by_symbol(mol, 'C')
    
    # Find central carbon (3 carbon neighbors)
    central_c = None
    for c in c_atoms:
        c_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() 
                       if mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() == 'C']
        if len(c_neighbors) == 3:
            central_c = c
            break
    
    if central_c is not None:
        neighbor_carbons = [n.GetIdx() for n in mol.GetAtomWithIdx(central_c).GetNeighbors() 
                           if mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() == 'C']
        
        # Find 4-atom chain for dihedral
        fourth_c = second_c = None
        for nc in neighbor_carbons:
            nc_carbons = [n.GetIdx() for n in mol.GetAtomWithIdx(nc).GetNeighbors() 
                         if n.GetSymbol() == 'C' and n.GetIdx() != central_c]
            if nc_carbons:
                second_c, fourth_c = nc, nc_carbons[0]
                break
        
        if fourth_c is not None:
            target_c = neighbor_carbons[0] if neighbor_carbons[0] != second_c else neighbor_carbons[1]
            dihedral_dict = {(fourth_c, second_c, central_c, target_c): 120.0}
            
            coords = qdgeo.optimize_mol(mol, dihedral=dihedral_dict, dihedral_k=5.0,
                                         repulsion_k=0.1, repulsion_cutoff=3.0, verbose=0)
            
            d = np.rad2deg(get_dihedral(coords, mol, fourth_c, second_c, central_c, target_c))
            assert dihedral_diff(d, 120.0) < 15.0
            write_sdf(coords, mol, "isopentane_dihedrals.sdf", "Isopentane Dihedrals")


# --- Low-Level API Test ---

def test_ethanol_dihedral_low_level():
    """Test ethanol using the low-level optimize API."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    n = mol.GetNumAtoms()
    
    bonds = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        key = tuple(sorted([mol.GetAtomWithIdx(a1).GetSymbol(), mol.GetAtomWithIdx(a2).GetSymbol()]))
        length = BOND_LENGTHS.get(key, 1.5)
        if bond.GetBondTypeAsDouble() == 2.0:
            length *= 0.9
        elif bond.GetBondTypeAsDouble() == 3.0:
            length *= 0.85
        bonds.append((a1, a2, length))
    
    angles = []
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
        if len(neighbors) < 2:
            continue
        angle_val = ANGLE.get(atom.GetHybridization(), ANGLE[Chem.HybridizationType.SP3])
        for j in range(len(neighbors)):
            for k in range(j + 1, len(neighbors)):
                angles.append((neighbors[j], i, neighbors[k], angle_val))
    
    c_atoms = get_atoms_by_symbol(mol, 'C')
    o_atoms = get_atoms_by_symbol(mol, 'O')
    h_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(c_atoms[0]).GetNeighbors() 
                   if mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() == 'H']
    
    if h_neighbors:
        dihedrals = [(h_neighbors[0], c_atoms[0], c_atoms[1], o_atoms[0], np.deg2rad(180.0))]
        coords, converged, energy = qdgeo.optimize(
            n_atoms=n, bonds=bonds, angles=angles, dihedrals=dihedrals,
            bond_force_constant=1.5, angle_force_constant=2.0, dihedral_force_constant=5.0,
            repulsion_force_constant=0.1, repulsion_cutoff=3.0)
        
        assert converged
        assert coords.shape == (9, 3)
        write_sdf(coords, mol, "ethanol_low_level.sdf", "Ethanol Low-Level API")


# --- Large Molecule Tests (MW ~400, for profiling) ---

def test_cholesterol():
    """Test cholesterol (C27H46O, MW ~387) - a steroid hormone."""
    # Cholesterol SMILES
    smiles = 'CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C'
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol.GetNumAtoms() == 74  # 27 C + 46 H + 1 O
    
    coords = qdgeo.optimize_mol(mol, verbose=1, repulsion_k=0.1, repulsion_cutoff=3.0,
                                 n_starts=5, maxeval=10000)
    assert coords.shape == (74, 3)
    write_sdf(coords, mol, "cholesterol.sdf", "Cholesterol")


def test_simvastatin():
    """Test simvastatin (C25H38O5, MW ~418) - a statin drug."""
    # Simvastatin SMILES
    smiles = 'CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C'
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol.GetNumAtoms() == 68  # 25 C + 38 H + 5 O
    
    coords = qdgeo.optimize_mol(mol, verbose=1, repulsion_k=0.1, repulsion_cutoff=3.0,
                                 n_starts=5, maxeval=10000)
    assert coords.shape == (68, 3)
    write_sdf(coords, mol, "simvastatin.sdf", "Simvastatin")


def test_cortisol():
    """Test cortisol (C21H30O5, MW ~362) - a stress hormone."""
    # Cortisol SMILES
    smiles = 'CC12CCC(=O)C=C1CCC3C2C(CC4(C3CCC4(C(=O)CO)O)C)O'
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol.GetNumAtoms() == 56  # 21 C + 30 H + 5 O
    
    coords = qdgeo.optimize_mol(mol, verbose=1, repulsion_k=0.1, repulsion_cutoff=3.0,
                                 n_starts=5, maxeval=10000)
    assert coords.shape == (56, 3)
    write_sdf(coords, mol, "cortisol.sdf", "Cortisol")


def test_tamoxifen():
    """Test tamoxifen (C26H29NO, MW ~371) - a breast cancer drug."""
    # Tamoxifen SMILES
    smiles = 'CC/C(=C(\\c1ccccc1)/c2ccc(cc2)OCCN(C)C)/c3ccccc3'
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol.GetNumAtoms() == 57  # 26 C + 29 H + 1 N + 1 O
    
    coords = qdgeo.optimize_mol(mol, verbose=1, repulsion_k=0.1, repulsion_cutoff=3.0,
                                 n_starts=5, maxeval=10000)
    assert coords.shape == (57, 3)
    write_sdf(coords, mol, "tamoxifen.sdf", "Tamoxifen")


def test_caffeine():
    """Test caffeine (C8H10N4O2, MW ~194) - a smaller reference molecule."""
    smiles = 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol.GetNumAtoms() == 24  # 8 C + 10 H + 4 N + 2 O
    
    coords = qdgeo.optimize_mol(mol, verbose=1, repulsion_k=0.1, repulsion_cutoff=3.0,
                                 n_starts=5, maxeval=10000)
    assert coords.shape == (24, 3)
    write_sdf(coords, mol, "caffeine.sdf", "Caffeine")
