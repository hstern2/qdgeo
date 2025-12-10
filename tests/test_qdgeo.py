"""Test suite for QDGeo molecular geometry optimization"""

import os
import numpy as np
import qdgeo
from qdgeo import optimize_mol
from qdgeo.optimize_mol import BOND_LENGTHS, ANGLE
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


def write_sdf(coords, mol, filename, title="Molecule"):
    """Write molecular geometry to SDF format using RDKit."""
    from rdkit.Chem import SDWriter
    
    # Create a copy of the molecule and add conformer
    mol_copy = Chem.Mol(mol)
    conf = Chem.Conformer(mol_copy.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, (x, y, z))
    mol_copy.AddConformer(conf, assignId=True)
    
    # Set molecule name
    mol_copy.SetProp("_Name", title)
    
    # Write using RDKit's SDF writer
    writer = SDWriter(filename)
    writer.write(mol_copy)
    writer.close()


class TestWater:
    def test_water(self):
        mol = Chem.MolFromSmiles('O')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, verbose=1)
        assert coords.shape == (3, 3)
        write_sdf(coords, mol, os.path.join(os.path.dirname(__file__), "water.sdf"), "Water")


class TestEthane:
    def test_ethane(self):
        mol = Chem.MolFromSmiles('CC')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, verbose=1)
        assert coords.shape == (8, 3)
        write_sdf(coords, mol, os.path.join(os.path.dirname(__file__), "ethane.sdf"), "Ethane")


class TestPropane:
    def test_propane(self):
        mol = Chem.MolFromSmiles('CCC')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, verbose=1)
        assert coords.shape == (11, 3)
        write_sdf(coords, mol, os.path.join(os.path.dirname(__file__), "propane.sdf"), "Propane")


class TestCyclopropane:
    def test_cyclopropane(self):
        mol = Chem.MolFromSmiles('C1CC1')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, verbose=1)
        assert coords.shape == (9, 3)
        write_sdf(coords, mol, os.path.join(os.path.dirname(__file__), "cyclopropane.sdf"), "Cyclopropane")


def _coords_to_conformer(coords, mol):
    """Create temporary conformer from coordinates."""
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, (x, y, z))
    mol.AddConformer(conf, assignId=True)
    return conf

def get_dihedral(coords, mol, i, j, k, l):
    """Calculate dihedral angle from coordinates using RDKit."""
    conf = _coords_to_conformer(coords, mol)
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


class TestButane:
    def test_butane(self):
        mol = Chem.MolFromSmiles('CCCC')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, repulsion_k=0.1, repulsion_cutoff=3.0, verbose=1)
        assert coords.shape == (14, 3)
        write_sdf(coords, mol, os.path.join(os.path.dirname(__file__), "butane.sdf"), "Butane")


class TestButaneWithDihedral:
    def test_butane_dihedral_mol_interface(self):
        mol = Chem.MolFromSmiles('CCCC')
        mol = Chem.AddHs(mol)
        n = mol.GetNumAtoms()
        
        c_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        assert len(c_atoms) >= 4
        
        dihedral_dict = {(c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 60.0}
        
        coords = qdgeo.optimize_mol(
            mol, dihedral=dihedral_dict, dihedral_k=5.0,
            repulsion_k=0.1, repulsion_cutoff=3.0,
            tolerance=1e-6, maxeval=5000
        )
        
        assert coords.shape == (14, 3)
        
        dihedral_angle = get_dihedral(coords, mol, c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3])
        dihedral_deg = np.rad2deg(dihedral_angle)
        target_deg = 60.0
        
        diff = dihedral_diff(dihedral_deg, target_deg)
        assert diff < 15.0
        write_sdf(coords, mol, os.path.join(os.path.dirname(__file__), "butane_dihedral_mol.sdf"), "Butane with Dihedral (Mol Interface)")


class TestCyclohexane:
    def test_cyclohexane(self):
        mol = Chem.MolFromSmiles('C1CCCCC1')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, repulsion_k=0.2, repulsion_cutoff=3.5, verbose=1)
        assert coords.shape == (18, 3)
        write_sdf(coords, mol, os.path.join(os.path.dirname(__file__), "cyclohexane.sdf"), "Cyclohexane")


class TestEthanol:
    def test_ethanol(self):
        mol = Chem.MolFromSmiles('CCO')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, verbose=1)
        assert coords.shape == (9, 3)
        write_sdf(coords, mol, os.path.join(os.path.dirname(__file__), "ethanol.sdf"), "Ethanol")


class TestEthanolWithDihedral:
    def test_ethanol_dihedral_constraint(self):
        mol = Chem.MolFromSmiles('CCO')
        mol = Chem.AddHs(mol)
        n = mol.GetNumAtoms()
        
        bonds = []
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            key = tuple(sorted([mol.GetAtomWithIdx(a1).GetSymbol(), 
                               mol.GetAtomWithIdx(a2).GetSymbol()]))
            length = BOND_LENGTHS.get(key, 1.5)
            order = bond.GetBondTypeAsDouble()
            if order == 2.0:
                length *= 0.9
            elif order == 3.0:
                length *= 0.85
            bonds.append((a1, a2, length))
        
        angles = []
        for i in range(n):
            atom = mol.GetAtomWithIdx(i)
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neighbors) < 2:
                continue
            angle_val = ANGLE.get(atom.GetHybridization(), 
                                  ANGLE[Chem.HybridizationType.SP3])
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    angles.append((neighbors[j], i, neighbors[k], angle_val))
        
        c_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        o_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'O']
        
        if len(c_atoms) >= 2 and len(o_atoms) >= 1:
            h_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(c_atoms[0]).GetNeighbors() 
                           if mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() == 'H']
            if h_neighbors:
                dihedrals = [(h_neighbors[0], c_atoms[0], c_atoms[1], o_atoms[0], np.deg2rad(180.0))]
                
                coords, converged, energy = qdgeo.optimize(
                    n_atoms=n, bonds=bonds, angles=angles, dihedrals=dihedrals,
                    bond_force_constant=1.5, angle_force_constant=2.0,
                    dihedral_force_constant=5.0,
                    repulsion_force_constant=0.1, repulsion_cutoff=3.0,
                    tolerance=1e-6, maxeval=5000
                )
                
                assert converged
                assert coords.shape == (9, 3)
                write_sdf(coords, mol, os.path.join(os.path.dirname(__file__), "ethanol_dihedral.sdf"), "Ethanol with Dihedral")


class TestEthanolWithDihedralMolInterface:
    def test_ethanol_dihedral_mol_interface(self):
        mol = Chem.MolFromSmiles('CCO')
        mol = Chem.AddHs(mol)
        n = mol.GetNumAtoms()
        
        c_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        o_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'O']
        
        if len(c_atoms) >= 2 and len(o_atoms) >= 1:
            h_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(c_atoms[0]).GetNeighbors() 
                           if mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() == 'H']
            if h_neighbors:
                dihedral_dict = {(h_neighbors[0], c_atoms[0], c_atoms[1], o_atoms[0]): 180.0}
                
                coords = qdgeo.optimize_mol(
                    mol, dihedral=dihedral_dict, dihedral_k=5.0,
                    repulsion_k=0.1, repulsion_cutoff=3.0,
                    tolerance=1e-6, maxeval=5000
                )
                
                assert coords.shape == (9, 3)
                write_sdf(coords, mol, os.path.join(os.path.dirname(__file__), "ethanol_dihedral_mol.sdf"), "Ethanol with Dihedral (Mol Interface)")


class TestBrCCCl:
    def test_br_cc_cl_dihedral(self):
        mol = Chem.MolFromSmiles('BrCCCl')
        mol = Chem.AddHs(mol)
        n = mol.GetNumAtoms()
        
        br_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'Br']
        c_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        cl_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'Cl']
        
        assert len(br_atoms) == 1 and len(c_atoms) == 2 and len(cl_atoms) == 1
        
        br = br_atoms[0]
        c1, c2 = c_atoms[0], c_atoms[1]
        cl = cl_atoms[0]
        
        dihedral_dict = {(br, c1, c2, cl): 60.0}
        
        coords = qdgeo.optimize_mol(
            mol, dihedral=dihedral_dict, dihedral_k=5.0,
            repulsion_k=0.1, repulsion_cutoff=3.0,
            tolerance=1e-6, maxeval=5000, verbose=0
        )
        
        assert coords.shape == (n, 3)
        
        dihedral_angle = get_dihedral(coords, mol, br, c1, c2, cl)
        dihedral_deg = np.rad2deg(dihedral_angle)
        target_deg = 60.0
        
        diff = dihedral_diff(dihedral_deg, target_deg)
        assert diff < 15.0
        write_sdf(coords, mol, os.path.join(os.path.dirname(__file__), "br_cc_cl.sdf"), "Br-C-C-Cl")


class TestBenzene:
    def test_benzene(self):
        mol = Chem.MolFromSmiles('c1ccccc1')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, verbose=1, repulsion_k=0.1, repulsion_cutoff=3.0)
        assert coords.shape == (12, 3)
        write_sdf(coords, mol, os.path.join(os.path.dirname(__file__), "benzene.sdf"), "Benzene")
        
        # Check that aromatic dihedrals are ~0° (planar)
        c_atoms = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        if len(c_atoms) >= 4:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for idx, (x, y, z) in enumerate(coords):
                conf.SetAtomPosition(idx, (x, y, z))
            mol.AddConformer(conf, assignId=True)
            dihedral = rdMolTransforms.GetDihedralDeg(conf, c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3])
            mol.RemoveConformer(conf.GetId())
            
            dihedral_abs = abs(dihedral)
            while dihedral_abs > 180:
                dihedral_abs -= 360
            dihedral_abs = abs(dihedral_abs)
            assert dihedral_abs < 30.0 or abs(dihedral_abs - 180) < 30.0

