"""RDKit wrapper for QDGeo optimization"""

import numpy as np
from typing import Optional
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolTransforms
from . import optimize

BOND_LENGTHS = {
    ("C", "C"): 1.54, ("C", "H"): 1.09, ("C", "O"): 1.43,
    ("C", "N"): 1.47, ("O", "H"): 0.96, ("N", "H"): 1.01,
    ("O", "O"): 1.48, ("N", "N"): 1.45,
    ("C", "Br"): 1.94, ("C", "Cl"): 1.79,
}

ANGLE = {
    Chem.HybridizationType.SP3: np.arccos(-1.0/3.0),
    Chem.HybridizationType.SP2: np.deg2rad(120.0),
    Chem.HybridizationType.SP: np.deg2rad(180.0),
}


def _get_bond_length(mol, bond):
    """Get ideal bond length from bond type and atom symbols."""
    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    key = tuple(sorted([mol.GetAtomWithIdx(a1).GetSymbol(), 
                       mol.GetAtomWithIdx(a2).GetSymbol()]))
    length = BOND_LENGTHS.get(key, 1.5)
    order = bond.GetBondTypeAsDouble()
    
    # Adjust for bond order
    if order == 1.5:  # Aromatic
        length *= 0.92
    elif order == 2.0:  # Double
        length *= 0.87
    elif order == 3.0:  # Triple
        length *= 0.80
    
    return length


def _get_dihedral_atoms(mol, j, k):
    """Get first valid neighbor pair for dihedral (i, j, k, l)."""
    atom_j = mol.GetAtomWithIdx(j)
    atom_k = mol.GetAtomWithIdx(k)
    neighbors_j = [n.GetIdx() for n in atom_j.GetNeighbors() if n.GetIdx() != k]
    neighbors_k = [n.GetIdx() for n in atom_k.GetNeighbors() if n.GetIdx() != j]
    if neighbors_j and neighbors_k:
        return neighbors_j[0], neighbors_k[0]
    return None, None


def _apply_template_restraints(mol, template, explicit_dihedrals, constrained_bonds, verbose):
    """Apply dihedral restraints from template molecule using substructure match.
    
    Args:
        mol: Target molecule
        template: Template molecule with conformer
        explicit_dihedrals: Dictionary to populate with dihedral restraints (modified in place)
        constrained_bonds: Set of bonds with dihedral constraints (modified in place)
        verbose: Verbosity level
    
    Returns:
        atom_map: Tuple mapping mol atom indices to template atom indices (or None if no match)
    """
    # Check if template has a conformer
    if template.GetNumConformers() == 0:
        if verbose > 0:
            print("Warning: Template has no conformer, skipping template restraints")
        return None
    
    # Find substructure match (template in mol)
    # Try with and without useChirality for better matching
    match = mol.GetSubstructMatch(template)
    
    if not match:
        # Try matching without explicit hydrogens consideration
        # Create query molecule from template
        match = mol.GetSubstructMatch(template, useChirality=False)
    
    if not match:
        if verbose > 0:
            print("Warning: Could not find template as substructure in molecule, skipping template restraints")
        return None
    
    if verbose > 0:
        print(f"\nTemplate restraints: Found substructure match with {len(match)} atoms")
        print(f"  Atom mapping (mol -> template): {list(enumerate(match))}")
    
    # Get template conformer
    template_conf = template.GetConformer()
    
    # Extract dihedrals from template for all rotatable bonds
    num_template_dihedrals = 0
    for bond in template.GetBonds():
        t_j, t_k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Skip ring bonds in template
        if bond.IsInRing():
            continue
        
        # Find the corresponding atoms in mol
        mol_j = match[t_j]
        mol_k = match[t_k]
        
        # Get dihedral atoms in template
        t_i, t_l = _get_dihedral_atoms(template, t_j, t_k)
        if t_i is None:
            continue
        
        # Map to mol indices
        mol_i = match[t_i]
        mol_l = match[t_l]
        
        # Calculate dihedral angle from template conformer
        template_dihedral_deg = rdMolTransforms.GetDihedralDeg(template_conf, t_i, t_j, t_k, t_l)
        template_dihedral_rad = np.deg2rad(template_dihedral_deg)
        
        # Add restraint
        dihedral_tuple = (mol_i, mol_j, mol_k, mol_l)
        explicit_dihedrals[dihedral_tuple] = template_dihedral_rad
        constrained_bonds.add((min(mol_j, mol_k), max(mol_j, mol_k)))
        num_template_dihedrals += 1
        
        if verbose > 0:
            mol_syms = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in [mol_i, mol_j, mol_k, mol_l]]
            print(f"  Template dihedral: {'-'.join(mol_syms)} ({mol_i}, {mol_j}, {mol_k}, {mol_l}): {template_dihedral_deg:.2f}°")
    
    if verbose > 0:
        print(f"  Total template dihedrals applied: {num_template_dihedrals}\n")
    
    return match


def optimize_mol(mol, bond_k=1.5, angle_k=2.0, tolerance=1e-6, maxeval=5000, verbose=0,
                 dihedral: Optional[dict[tuple[int, int, int, int], float]] = None,
                 dihedral_k=5.0, repulsion_k=0.1, repulsion_cutoff=3.0, n_starts=10,
                 template: Optional[Chem.Mol] = None, template_k=5.0,
                 planarity_k=0.5):
    """Optimize molecular geometry using QDGeo.
    
    Args:
        mol: RDKit molecule object
        bond_k: Bond force constant (default: 1.5)
        angle_k: Angle force constant (default: 2.0)
        tolerance: Convergence tolerance (default: 1e-6)
        maxeval: Maximum function evaluations (default: 5000)
        verbose: If > 0, prints bond lengths and angles (default: 0)
        dihedral: Optional dictionary of dihedral angle constraints.
                  Keys are tuples of 4 atom indices (i, j, k, l), values are target angles in degrees.
        dihedral_k: Dihedral force constant (default: 5.0)
        repulsion_k: Non-bonded repulsion force constant (default: 0.1)
        repulsion_cutoff: Repulsion cutoff distance in Angstroms (default: 3.0)
        n_starts: Number of random starting points to try (default: 10)
        template: Optional RDKit molecule to use as template. Will find maximum substructure match
                  and apply restraints from template geometry (default: None)
        template_k: Force constant for template dihedral restraints (default: 5.0)
        planarity_k: Force constant for planarity restraints (default: 0.5)
    
    Returns:
        Optimized coordinates array, shape (n_atoms, 3)
    """
    n = mol.GetNumAtoms()
    
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), _get_bond_length(mol, bond))
             for bond in mol.GetBonds()]
    
    angles = []
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) >= 2:
            angle_val = ANGLE.get(atom.GetHybridization(), ANGLE[Chem.HybridizationType.SP3])
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    angles.append((neighbors[j], i, neighbors[k], angle_val))
    
    explicit_dihedrals = {}
    constrained_bonds = set()
    
    # Add planarity constraints for sp2 centers with exactly 3 neighbors
    # This keeps the sp2 center in the plane of its 3 neighbors
    # For aromatic rings with explicit H, this forces H to be coplanar with the ring
    planarities = []
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        
        if atom.GetHybridization() == Chem.HybridizationType.SP2 and len(neighbors) == 3:
            # Atom i should be planar with its three neighbors
            planarities.append((i, neighbors[0], neighbors[1], neighbors[2]))
            if verbose > 0:
                sym = atom.GetSymbol()
                neighbor_syms = [mol.GetAtomWithIdx(n).GetSymbol() for n in neighbors]
                print(f"  Planarity: {sym}{i} with {neighbor_syms[0]}{neighbors[0]}, {neighbor_syms[1]}{neighbors[1]}, {neighbor_syms[2]}{neighbors[2]}")
    
    # Apply template restraints if provided
    atom_map = None  # Maps mol atom indices to template atom indices
    if template is not None:
        atom_map = _apply_template_restraints(mol, template, explicit_dihedrals, 
                                               constrained_bonds, verbose)
    
    if dihedral is not None:
        for (i, j, k, l), angle_deg in dihedral.items():
            if not mol.GetBondBetweenAtoms(j, k).IsInRing():
                explicit_dihedrals[(i, j, k, l)] = np.deg2rad(angle_deg)
                constrained_bonds.add((min(j, k), max(j, k)))
    
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_key = (min(j, k), max(j, k))
        if bond_key in constrained_bonds:
            continue
        
        # For aromatic ring bonds, add 0° dihedral to enforce planarity
        if bond.IsInRing():
            atom_j = mol.GetAtomWithIdx(j)
            atom_k = mol.GetAtomWithIdx(k)
            if (atom_j.GetHybridization() == Chem.HybridizationType.SP2 and
                atom_k.GetHybridization() == Chem.HybridizationType.SP2):
                i, l = _get_dihedral_atoms(mol, j, k)
                if i is not None and (i, j, k, l) not in explicit_dihedrals:
                    explicit_dihedrals[(i, j, k, l)] = 0.0  # 0° for planar rings
                    if verbose > 0:
                        atom_i = mol.GetAtomWithIdx(i)
                        atom_l = mol.GetAtomWithIdx(l)
                        print(f"  Dihedral: {atom_i.GetSymbol()}{i}-{atom_j.GetSymbol()}{j}-{atom_k.GetSymbol()}{k}-{atom_l.GetSymbol()}{l} = 0.0°")
            continue
        
        # Only add dihedral for rotatable single bonds (non-ring)
        if bond.GetBondTypeAsDouble() == 1.0:
            i, l = _get_dihedral_atoms(mol, j, k)
            if i is not None and (i, j, k, l) not in explicit_dihedrals:
                explicit_dihedrals[(i, j, k, l)] = np.deg2rad(180.0)
    
    dihedrals = [(i, j, k, l, phi) for (i, j, k, l), phi in explicit_dihedrals.items()]
    
    # Use template_k if template provided, otherwise use dihedral_k
    effective_dihedral_k = template_k if (template is not None and atom_map is not None) else dihedral_k
    
    if verbose > 0:
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else 'Unnamed'
        print(f"\nMolecule: {name}")
        print(f"Atoms: {n}, Bonds: {len(bonds)}, Angles: {len(angles)}, Dihedrals: {len(dihedrals)}, Planarities: {len(planarities)}")
        if template is not None and atom_map is not None:
            print(f"Using template_k={template_k}")
        print()
    
    coords, converged, energy = optimize(
        n_atoms=n, bonds=bonds, angles=angles, dihedrals=dihedrals,
        bond_force_constant=bond_k, angle_force_constant=angle_k,
        dihedral_force_constant=effective_dihedral_k,
        repulsion_force_constant=repulsion_k, repulsion_cutoff=repulsion_cutoff,
        planarities=planarities, planarity_force_constant=planarity_k,
        tolerance=tolerance, maxeval=maxeval, verbose=verbose, n_starts=n_starts
    )
    
    if verbose > 0:
        print(f"Optimization converged: {converged}, Final energy: {energy:.6e}\n")
    
    return coords
