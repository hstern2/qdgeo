"""Test suite for QDGeo molecular geometry optimization"""

import os
import qdgeo
from rdkit import Chem


def write_sdf(coords, mol, filename, title="Molecule"):
    """Write molecular geometry to SDF format."""
    n_atoms = coords.shape[0]
    n_bonds = mol.GetNumBonds()
    
    with open(filename, 'w') as f:
        f.write(f"{title}\n  QDGeo\n\n")
        f.write(f"{n_atoms:3d}{n_bonds:3d}  0  0  0  0  0  0  0  0999 V2000\n")
        for i, (x, y, z) in enumerate(coords):
            atom = mol.GetAtomWithIdx(i)
            f.write(f"{x:10.4f}{y:10.4f}{z:10.4f} {atom.GetSymbol():>2s}  0  0  0  0  0  0  0  0  0  0  0  0\n")
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            order = int(bond.GetBondTypeAsDouble())
            f.write(f"{a1+1:3d}{a2+1:3d}{order:3d}  0  0  0  0\n")
        f.write("M  END\n$$$$\n")


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

