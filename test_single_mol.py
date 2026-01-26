#!/usr/bin/env python3
"""Test a single molecule from a.smi with verbose output."""

import sys
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import qdgeo
import numpy as np

# Read template
template_path = '/Users/hstern/Desktop/xfep/qdg/ORO-0001020.sdf'
supplier = Chem.SDMolSupplier(template_path, removeHs=False)
template = next(supplier)
print(f"Template: {template.GetNumAtoms()} atoms")
print(f"Template has {template.GetNumConformers()} conformers")

# First molecule from a.smi
smiles = 'c1cc(ccc1c1c[nH]c(c1Cl)Cl)NC(=O)[C@H](C(C1CC1)C1CC1)NC(=O)c1ccnn1C(C)C'
mol_name = 'ORO-0000717'

print(f"\nProcessing {mol_name}...")
print(f"SMILES: {smiles}")

# Parse and add H
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
print(f"Molecule: {mol.GetNumAtoms()} atoms (with H)")

# Generate conformer
print("\nGenerating conformer with template...")
try:
    coords = qdgeo.optimize_mol(
        mol,
        template=template,
        template_coordinate_k=10.0,
        repulsion_k=0.1,
        repulsion_cutoff=3.0,
        verbose=2,  # More verbose
        maxeval=5000
    )
    print(f"\nSUCCESS: Generated coordinates shape: {coords.shape}")
    
    # Check MCS RMSD
    from qdgeo.optimize_mol import _find_mcs_mapping
    atom_map = _find_mcs_mapping(mol, template, verbose=1)
    if atom_map:
        # Get MCS coordinates from optimized molecule
        mol_conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            mol_conf.SetAtomPosition(i, (coords[i, 0], coords[i, 1], coords[i, 2]))
        mol.AddConformer(mol_conf)
        
        # Get template conformer
        template_conf = template.GetConformer()
        
        # Calculate RMSD for MCS atoms
        mol_indices = list(atom_map.keys())
        template_indices = list(atom_map.values())
        
        rmsd = rdMolAlign.AlignMol(mol, template, atomMap=list(zip(mol_indices, template_indices)))
        print(f"\nMCS RMSD: {rmsd:.4f} Å")
        
        # Check individual atom distances
        max_dist = 0.0
        for mol_idx, template_idx in atom_map.items():
            mol_pos = np.array(mol_conf.GetAtomPosition(mol_idx))
            template_pos = np.array(template_conf.GetAtomPosition(template_idx))
            dist = np.linalg.norm(mol_pos - template_pos)
            max_dist = max(max_dist, dist)
            if dist > 1.0:  # Report large distances
                print(f"  Atom {mol_idx} -> {template_idx}: {dist:.4f} Å")
        print(f"Max MCS atom distance: {max_dist:.4f} Å")
        
        mol.RemoveConformer(mol_conf.GetId())
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
