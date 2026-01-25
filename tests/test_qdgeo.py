"""Test suite for QDGeo molecular geometry optimization"""

import os
import numpy as np
import qdgeo
from qdgeo import optimize_mol
from qdgeo.optimize_mol import BOND_LENGTHS, ANGLE
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# Create output directory for test geometries
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    filepath = os.path.join(OUTPUT_DIR, filename)
    writer = SDWriter(filepath)
    writer.write(mol_copy)
    writer.close()
    print(f"  → Saved: {filepath}")


class TestWater:
    def test_water(self):
        mol = Chem.MolFromSmiles('O')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, verbose=0)
        assert coords.shape == (3, 3)
        write_sdf(coords, mol, "water.sdf", "Water")


class TestEthane:
    def test_ethane(self):
        mol = Chem.MolFromSmiles('CC')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, verbose=0)
        assert coords.shape == (8, 3)
        write_sdf(coords, mol, "ethane.sdf", "Ethane")


class TestPropane:
    def test_propane(self):
        mol = Chem.MolFromSmiles('CCC')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, verbose=0)
        assert coords.shape == (11, 3)
        write_sdf(coords, mol, "propane.sdf", "Propane")


class TestCyclopropane:
    def test_cyclopropane(self):
        mol = Chem.MolFromSmiles('C1CC1')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, verbose=1)
        assert coords.shape == (9, 3)
        write_sdf(coords, mol, "cyclopropane.sdf", "Cyclopropane")


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
        write_sdf(coords, mol, "butane.sdf", "Butane")


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
        write_sdf(coords, mol, "butane_dihedral_mol.sdf", "Butane with Dihedral (Mol Interface)")


class TestCyclohexane:
    def test_cyclohexane(self):
        mol = Chem.MolFromSmiles('C1CCCCC1')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, repulsion_k=0.2, repulsion_cutoff=3.5, verbose=1)
        assert coords.shape == (18, 3)
        write_sdf(coords, mol, "cyclohexane.sdf", "Cyclohexane")


class TestEthanol:
    def test_ethanol(self):
        mol = Chem.MolFromSmiles('CCO')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, verbose=1)
        assert coords.shape == (9, 3)
        write_sdf(coords, mol, "ethanol.sdf", "Ethanol")


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
                write_sdf(coords, mol, "ethanol_dihedral.sdf", "Ethanol with Dihedral")


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
                write_sdf(coords, mol, "ethanol_dihedral_mol.sdf", "Ethanol with Dihedral (Mol Interface)")


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
        write_sdf(coords, mol, "br_cc_cl.sdf", "Br-C-C-Cl")


class TestBenzene:
    def test_benzene(self):
        mol = Chem.MolFromSmiles('c1ccccc1')
        mol = Chem.AddHs(mol)
        coords = qdgeo.optimize_mol(mol, verbose=1, repulsion_k=0.1, repulsion_cutoff=3.0)
        assert coords.shape == (12, 3)
        write_sdf(coords, mol, "benzene.sdf", "Benzene")
        
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


class TestPentaneMultipleDihedrals:
    def test_pentane_multiple_dihedrals(self):
        """Test pentane with multiple dihedral constraints on different bonds."""
        mol = Chem.MolFromSmiles('CCCCC')
        mol = Chem.AddHs(mol)
        n = mol.GetNumAtoms()
        
        # Get carbon atoms
        c_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        assert len(c_atoms) == 5
        
        # Constrain two different dihedrals
        dihedral_dict = {
            (c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 60.0,   # First dihedral: 60°
            (c_atoms[1], c_atoms[2], c_atoms[3], c_atoms[4]): 120.0,  # Second dihedral: 120°
        }
        
        coords = qdgeo.optimize_mol(
            mol, dihedral=dihedral_dict, dihedral_k=5.0,
            repulsion_k=0.1, repulsion_cutoff=3.0,
            tolerance=1e-6, maxeval=5000, verbose=0
        )
        
        assert coords.shape == (n, 3)
        
        # Check first dihedral (should be ~60°)
        dihedral1 = get_dihedral(coords, mol, c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3])
        dihedral1_deg = np.rad2deg(dihedral1)
        diff1 = dihedral_diff(dihedral1_deg, 60.0)
        print(f"Pentane dihedral 1: target=60°, actual={dihedral1_deg:.2f}°, diff={diff1:.2f}°")
        assert diff1 < 15.0, f"Dihedral 1 constraint failed: target=60°, actual={dihedral1_deg:.2f}°"
        
        # Check second dihedral (should be ~120°)
        dihedral2 = get_dihedral(coords, mol, c_atoms[1], c_atoms[2], c_atoms[3], c_atoms[4])
        dihedral2_deg = np.rad2deg(dihedral2)
        diff2 = dihedral_diff(dihedral2_deg, 120.0)
        print(f"Pentane dihedral 2: target=120°, actual={dihedral2_deg:.2f}°, diff={diff2:.2f}°")
        assert diff2 < 15.0, f"Dihedral 2 constraint failed: target=120°, actual={dihedral2_deg:.2f}°"
        
        write_sdf(coords, mol, "pentane_dihedrals.sdf", "Pentane Multiple Dihedrals")


class TestHexaneAllDihedrals:
    def test_hexane_all_dihedrals(self):
        """Test hexane with constraints on all three C-C-C-C dihedrals."""
        mol = Chem.MolFromSmiles('CCCCCC')
        mol = Chem.AddHs(mol)
        n = mol.GetNumAtoms()
        
        # Get carbon atoms
        c_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        assert len(c_atoms) == 6
        
        # Constrain all three backbone dihedrals to different values
        dihedral_dict = {
            (c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 180.0,  # Trans
            (c_atoms[1], c_atoms[2], c_atoms[3], c_atoms[4]): 60.0,   # Gauche
            (c_atoms[2], c_atoms[3], c_atoms[4], c_atoms[5]): -60.0,  # Gauche-
        }
        
        coords = qdgeo.optimize_mol(
            mol, dihedral=dihedral_dict, dihedral_k=5.0,
            repulsion_k=0.1, repulsion_cutoff=3.0,
            tolerance=1e-6, maxeval=5000, verbose=0
        )
        
        assert coords.shape == (n, 3)
        
        # Check all three dihedrals
        targets = [180.0, 60.0, -60.0]
        for idx, (i, j, k, l) in enumerate([
            (c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]),
            (c_atoms[1], c_atoms[2], c_atoms[3], c_atoms[4]),
            (c_atoms[2], c_atoms[3], c_atoms[4], c_atoms[5])
        ]):
            dihedral = get_dihedral(coords, mol, i, j, k, l)
            dihedral_deg = np.rad2deg(dihedral)
            diff = dihedral_diff(dihedral_deg, targets[idx])
            print(f"Hexane dihedral {idx+1}: target={targets[idx]:.0f}°, actual={dihedral_deg:.2f}°, diff={diff:.2f}°")
            assert diff < 15.0, f"Dihedral {idx+1} constraint failed: target={targets[idx]:.0f}°, actual={dihedral_deg:.2f}°"
        
        write_sdf(coords, mol, "hexane_all_dihedrals.sdf", "Hexane All Dihedrals")


class TestButanolDihedrals:
    def test_butanol_multiple_constraints(self):
        """Test 1-butanol with dihedral constraints on both C-C and C-O bonds."""
        mol = Chem.MolFromSmiles('CCCCO')
        mol = Chem.AddHs(mol)
        n = mol.GetNumAtoms()
        
        # Get carbon and oxygen atoms
        c_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        o_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'O']
        assert len(c_atoms) == 4 and len(o_atoms) == 1
        
        # Constrain C-C-C-C and C-C-C-O dihedrals
        dihedral_dict = {
            (c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 180.0,  # C-C-C-C trans
            (c_atoms[1], c_atoms[2], c_atoms[3], o_atoms[0]): 60.0,   # C-C-C-O gauche
        }
        
        coords = qdgeo.optimize_mol(
            mol, dihedral=dihedral_dict, dihedral_k=5.0,
            repulsion_k=0.1, repulsion_cutoff=3.0,
            tolerance=1e-6, maxeval=5000, verbose=0
        )
        
        assert coords.shape == (n, 3)
        
        # Check C-C-C-C dihedral
        dihedral1 = get_dihedral(coords, mol, c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3])
        dihedral1_deg = np.rad2deg(dihedral1)
        diff1 = dihedral_diff(dihedral1_deg, 180.0)
        print(f"Butanol C-C-C-C: target=180°, actual={dihedral1_deg:.2f}°, diff={diff1:.2f}°")
        assert diff1 < 15.0, f"C-C-C-C dihedral failed: target=180°, actual={dihedral1_deg:.2f}°"
        
        # Check C-C-C-O dihedral
        dihedral2 = get_dihedral(coords, mol, c_atoms[1], c_atoms[2], c_atoms[3], o_atoms[0])
        dihedral2_deg = np.rad2deg(dihedral2)
        diff2 = dihedral_diff(dihedral2_deg, 60.0)
        print(f"Butanol C-C-C-O: target=60°, actual={dihedral2_deg:.2f}°, diff={diff2:.2f}°")
        assert diff2 < 15.0, f"C-C-C-O dihedral failed: target=60°, actual={dihedral2_deg:.2f}°"
        
        write_sdf(coords, mol, "butanol_dihedrals.sdf", "Butanol Dihedrals")


class TestIsopentaneBranched:
    def test_isopentane_branched_dihedrals(self):
        """Test isopentane (2-methylbutane) with dihedral constraints on branched molecule."""
        mol = Chem.MolFromSmiles('CC(C)CC')  # 2-methylbutane
        mol = Chem.AddHs(mol)
        n = mol.GetNumAtoms()
        
        # Get carbon atoms
        c_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        assert len(c_atoms) == 5
        
        # Find the central carbon (has 3 carbon neighbors)
        central_c = None
        for c in c_atoms:
            neighbors = mol.GetAtomWithIdx(c).GetNeighbors()
            c_neighbors = [n.GetIdx() for n in neighbors if n.GetSymbol() == 'C']
            if len(c_neighbors) == 3:
                central_c = c
                break
        
        assert central_c is not None
        
        # Get the three carbons bonded to central carbon
        neighbors = mol.GetAtomWithIdx(central_c).GetNeighbors()
        neighbor_carbons = [n.GetIdx() for n in neighbors if n.GetSymbol() == 'C']
        assert len(neighbor_carbons) == 3
        
        # Find one more carbon connected to one of the neighbors (for a 4-atom dihedral)
        fourth_c = None
        second_c = None
        for nc in neighbor_carbons:
            nc_neighbors = mol.GetAtomWithIdx(nc).GetNeighbors()
            nc_carbons = [n.GetIdx() for n in nc_neighbors if n.GetSymbol() == 'C' and n.GetIdx() != central_c]
            if len(nc_carbons) > 0:
                second_c = nc
                fourth_c = nc_carbons[0]
                break
        
        if fourth_c is not None:
            # Constrain one dihedral along the main chain
            dihedral_dict = {
                (fourth_c, second_c, central_c, neighbor_carbons[0] if neighbor_carbons[0] != second_c else neighbor_carbons[1]): 120.0,
            }
            
            coords = qdgeo.optimize_mol(
                mol, dihedral=dihedral_dict, dihedral_k=5.0,
                repulsion_k=0.1, repulsion_cutoff=3.0,
                tolerance=1e-6, maxeval=5000, verbose=0
            )
            
            assert coords.shape == (n, 3)
            
            # Check dihedral
            target_c = neighbor_carbons[0] if neighbor_carbons[0] != second_c else neighbor_carbons[1]
            dihedral = get_dihedral(coords, mol, fourth_c, second_c, central_c, target_c)
            dihedral_deg = np.rad2deg(dihedral)
            diff = dihedral_diff(dihedral_deg, 120.0)
            print(f"Isopentane dihedral: target=120°, actual={dihedral_deg:.2f}°, diff={diff:.2f}°")
            assert diff < 15.0, f"Isopentane dihedral failed: target=120°, actual={dihedral_deg:.2f}°"
            
            write_sdf(coords, mol, "isopentane_dihedrals.sdf", "Isopentane Dihedrals")


class TestButadiene:
    def test_butadiene_conjugated(self):
        """Test 1,3-butadiene with dihedral constraint across conjugated system."""
        mol = Chem.MolFromSmiles('C=CC=C')
        mol = Chem.AddHs(mol)
        n = mol.GetNumAtoms()
        
        # Get carbon atoms
        c_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        assert len(c_atoms) == 4
        
        # Constrain the C=C-C=C dihedral (force s-trans vs s-cis)
        dihedral_dict = {
            (c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3]): 0.0,  # Force s-cis conformation
        }
        
        coords = qdgeo.optimize_mol(
            mol, dihedral=dihedral_dict, dihedral_k=10.0,  # Higher force constant for double bonds
            repulsion_k=0.1, repulsion_cutoff=3.0,
            tolerance=1e-6, maxeval=5000, verbose=0
        )
        
        assert coords.shape == (n, 3)
        
        # Check dihedral
        dihedral = get_dihedral(coords, mol, c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3])
        dihedral_deg = np.rad2deg(dihedral)
        diff = dihedral_diff(dihedral_deg, 0.0)
        print(f"Butadiene C=C-C=C: target=0°, actual={dihedral_deg:.2f}°, diff={diff:.2f}°")
        assert diff < 35.0, f"Butadiene dihedral failed: target=0°, actual={dihedral_deg:.2f}°"
        
        write_sdf(coords, mol, "butadiene_cis.sdf", "Butadiene s-cis")


class TestPropanediol:
    def test_propanediol_multiple_oh(self):
        """Test 1,3-propanediol with dihedral constraints on both C-O bonds."""
        mol = Chem.MolFromSmiles('OCCCO')
        mol = Chem.AddHs(mol)
        n = mol.GetNumAtoms()
        
        # Get carbon and oxygen atoms
        c_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        o_atoms = [i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == 'O']
        assert len(c_atoms) == 3 and len(o_atoms) == 2
        
        # Constrain O-C-C-C and C-C-C-O dihedrals
        dihedral_dict = {
            (o_atoms[0], c_atoms[0], c_atoms[1], c_atoms[2]): 60.0,
            (c_atoms[0], c_atoms[1], c_atoms[2], o_atoms[1]): -60.0,
        }
        
        coords = qdgeo.optimize_mol(
            mol, dihedral=dihedral_dict, dihedral_k=5.0,
            repulsion_k=0.1, repulsion_cutoff=3.0,
            tolerance=1e-6, maxeval=5000, verbose=0
        )
        
        assert coords.shape == (n, 3)
        
        # Check first O-C-C-C dihedral
        dihedral1 = get_dihedral(coords, mol, o_atoms[0], c_atoms[0], c_atoms[1], c_atoms[2])
        dihedral1_deg = np.rad2deg(dihedral1)
        diff1 = dihedral_diff(dihedral1_deg, 60.0)
        print(f"Propanediol O-C-C-C: target=60°, actual={dihedral1_deg:.2f}°, diff={diff1:.2f}°")
        assert diff1 < 15.0, f"First O-C-C-C dihedral failed: target=60°, actual={dihedral1_deg:.2f}°"
        
        # Check second C-C-C-O dihedral
        dihedral2 = get_dihedral(coords, mol, c_atoms[0], c_atoms[1], c_atoms[2], o_atoms[1])
        dihedral2_deg = np.rad2deg(dihedral2)
        diff2 = dihedral_diff(dihedral2_deg, -60.0)
        print(f"Propanediol C-C-C-O: target=-60°, actual={dihedral2_deg:.2f}°, diff={diff2:.2f}°")
        assert diff2 < 15.0, f"Second C-C-C-O dihedral failed: target=-60°, actual={dihedral2_deg:.2f}°"
        
        write_sdf(coords, mol, "propanediol_dihedrals.sdf", "Propanediol Dihedrals")


class TestTemplateButaneToButane:
    def test_template_same_molecule(self):
        """Test template restraints with same molecule (butane -> butane)."""
        from rdkit.Chem import AllChem
        
        # Create template butane with specific conformation
        template = Chem.MolFromSmiles('CCCC')
        template = Chem.AddHs(template)
        AllChem.EmbedMolecule(template, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(template)
        
        # Get template dihedral
        c_atoms_template = [i for i in range(template.GetNumAtoms()) 
                           if template.GetAtomWithIdx(i).GetSymbol() == 'C']
        template_dihedral = get_dihedral(
            template.GetConformer().GetPositions(),
            template,
            c_atoms_template[0], c_atoms_template[1], 
            c_atoms_template[2], c_atoms_template[3]
        )
        template_dihedral_deg = np.rad2deg(template_dihedral)
        
        # Create target molecule (same structure)
        mol = Chem.MolFromSmiles('CCCC')
        mol = Chem.AddHs(mol)
        
        # Optimize with template
        coords = qdgeo.optimize_mol(
            mol, template=template, template_k=10.0,
            repulsion_k=0.1, repulsion_cutoff=3.0,
            verbose=0
        )
        
        assert coords.shape == (mol.GetNumAtoms(), 3)
        
        # Check that optimized dihedral matches template
        c_atoms = [i for i in range(mol.GetNumAtoms()) 
                  if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        result_dihedral = get_dihedral(coords, mol, c_atoms[0], c_atoms[1], c_atoms[2], c_atoms[3])
        result_dihedral_deg = np.rad2deg(result_dihedral)
        
        diff = dihedral_diff(result_dihedral_deg, template_dihedral_deg)
        print(f"Template butane dihedral: template={template_dihedral_deg:.2f}°, result={result_dihedral_deg:.2f}°, diff={diff:.2f}°")
        assert diff < 20.0, f"Template restraint failed: template={template_dihedral_deg:.2f}°, result={result_dihedral_deg:.2f}°"
        
        write_sdf(coords, mol, "butane_template.sdf", "Butane with Template")


class TestTemplatePropaneToPentane:
    def test_template_substructure(self):
        """Test template with substructure (propane template -> pentane target)."""
        from rdkit.Chem import AllChem
        
        # Create template propane with specific conformation
        template = Chem.MolFromSmiles('CCC')
        template = Chem.AddHs(template)
        AllChem.EmbedMolecule(template, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(template)
        
        # Get template C-C-C-C dihedral
        c_atoms_template = [i for i in range(template.GetNumAtoms()) 
                           if template.GetAtomWithIdx(i).GetSymbol() == 'C']
        
        # Create target molecule (pentane - larger than template)
        mol = Chem.MolFromSmiles('CCCCC')
        mol = Chem.AddHs(mol)
        
        # Optimize with template
        coords = qdgeo.optimize_mol(
            mol, template=template, template_k=8.0,
            repulsion_k=0.1, repulsion_cutoff=3.0,
            verbose=1
        )
        
        assert coords.shape == (mol.GetNumAtoms(), 3)
        
        # Template should constrain matching part of molecule
        write_sdf(coords, mol, "pentane_propane_template.sdf", "Pentane with Propane Template")


class TestTemplateEthanolToButanol:
    def test_template_with_heteroatoms(self):
        """Test template with heteroatoms (ethanol template -> butanol target)."""
        from rdkit.Chem import AllChem
        
        # Create template ethanol with specific conformation
        template = Chem.MolFromSmiles('CCO')
        template = Chem.AddHs(template)
        AllChem.EmbedMolecule(template, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(template)
        
        # Create target molecule (butanol)
        mol = Chem.MolFromSmiles('CCCCO')
        mol = Chem.AddHs(mol)
        
        # Optimize with template
        coords = qdgeo.optimize_mol(
            mol, template=template, template_k=8.0,
            repulsion_k=0.1, repulsion_cutoff=3.0,
            verbose=1
        )
        
        assert coords.shape == (mol.GetNumAtoms(), 3)
        
        # Template should constrain the C-C-C-O end of butanol
        c_atoms = [i for i in range(mol.GetNumAtoms()) 
                  if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        o_atoms = [i for i in range(mol.GetNumAtoms()) 
                  if mol.GetAtomWithIdx(i).GetSymbol() == 'O']
        
        # The last two carbons and oxygen should match template geometry
        assert len(c_atoms) >= 2 and len(o_atoms) >= 1
        
        write_sdf(coords, mol, "butanol_ethanol_template.sdf", "Butanol with Ethanol Template")


class TestTemplateNoConformer:
    def test_template_without_conformer(self):
        """Test that template without conformer is handled gracefully."""
        # Create template without conformer
        template = Chem.MolFromSmiles('CCC')
        template = Chem.AddHs(template)
        # Note: No EmbedMolecule call, so no conformer
        
        # Create target molecule
        mol = Chem.MolFromSmiles('CCCC')
        mol = Chem.AddHs(mol)
        
        # Should work without error, just ignoring template
        coords = qdgeo.optimize_mol(
            mol, template=template, template_k=5.0,
            repulsion_k=0.1, repulsion_cutoff=3.0,
            verbose=1
        )
        
        assert coords.shape == (mol.GetNumAtoms(), 3)


class TestTemplateNoMatch:
    def test_template_no_substructure_match(self):
        """Test that non-matching template is handled gracefully."""
        from rdkit.Chem import AllChem
        
        # Create template with nitrogen
        template = Chem.MolFromSmiles('CCN')
        template = Chem.AddHs(template)
        AllChem.EmbedMolecule(template, randomSeed=42)
        
        # Create target molecule with no nitrogen
        mol = Chem.MolFromSmiles('CCCC')
        mol = Chem.AddHs(mol)
        
        # Should work without error, just ignoring template
        coords = qdgeo.optimize_mol(
            mol, template=template, template_k=5.0,
            repulsion_k=0.1, repulsion_cutoff=3.0,
            verbose=1
        )
        
        assert coords.shape == (mol.GetNumAtoms(), 3)

