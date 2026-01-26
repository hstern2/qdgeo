"""RDKit wrapper for QDGeo optimization"""

import numpy as np
from typing import Optional
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolTransforms
from rdkit.Chem import rdFMCS
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


def _find_mcs_mapping(mol, template, verbose, debug=False):
    """Find maximum common substructure (MCS) between mol and template and return atom mapping.
    
    Args:
        mol: Target molecule
        template: Template molecule
        verbose: Verbosity level
        debug: If True, print detailed debug information
    
    Returns:
        atom_map: Dictionary mapping mol atom indices to template atom indices (or None if no match)
    """
    if debug:
        print(f"[DEBUG MCS] Starting MCS search: mol has {mol.GetNumAtoms()} atoms, template has {template.GetNumAtoms()} atoms")
    
    # Find MCS between mol and template
    mcs_params = rdFMCS.MCSParameters()
    mcs_params.AtomCompareParameters.MatchValences = False
    mcs_params.AtomCompareParameters.RingMatchesRingOnly = False
    mcs_params.BondCompareParameters.RingMatchesRingOnly = False
    mcs_params.BondCompareParameters.CompleteRingsOnly = False
    mcs_params.Timeout = 10  # 10 second timeout to prevent hanging
    mcs_params.MaximizeBonds = True
    
    if debug:
        print(f"[DEBUG MCS] Calling rdFMCS.FindMCS() with timeout={mcs_params.Timeout}s...")
    
    mcs_result = rdFMCS.FindMCS([mol, template], mcs_params)
    
    if debug:
        print(f"[DEBUG MCS] FindMCS() returned: numAtoms={mcs_result.numAtoms}")
    
    if mcs_result.numAtoms < 2:
        if verbose > 0:
            print("Warning: MCS has fewer than 2 atoms, skipping template restraints")
        return None
    
    if debug:
        print(f"[DEBUG MCS] Creating MCS molecule from SMARTS: {mcs_result.smartsString}")
    
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    if mcs_mol is None:
        if verbose > 0:
            print("Warning: Could not create MCS molecule from SMARTS, skipping template restraints")
        return None
    
    if debug:
        print(f"[DEBUG MCS] Finding substructure matches in mol and template...")
    
    mol_matches = mol.GetSubstructMatches(mcs_mol)
    template_matches = template.GetSubstructMatches(mcs_mol)
    
    if debug:
        print(f"[DEBUG MCS] Found {len(mol_matches)} matches in mol, {len(template_matches)} in template")
    
    if not mol_matches or not template_matches:
        if verbose > 0:
            print("Warning: Could not find MCS matches in molecules, skipping template restraints")
        return None
    
    # Use first match (could be improved to find best match)
    mol_match = mol_matches[0]
    template_match = template_matches[0]
    
    # Create mapping: mol_idx -> template_idx
    atom_map = {}
    for mcs_idx, (mol_idx, template_idx) in enumerate(zip(mol_match, template_match)):
        atom_map[mol_idx] = template_idx
    
    if verbose > 0:
        print(f"\nMCS template restraints: Found MCS with {len(atom_map)} atoms")
        print(f"  Atom mapping (mol -> template): {sorted(atom_map.items())}")
    
    return atom_map


def _apply_template_restraints(mol, template, explicit_dihedrals, constrained_bonds, 
                               bond_lengths, verbose):
    """Apply restraints from template molecule using MCS matching.
    
    Args:
        mol: Target molecule
        template: Template molecule with conformer
        explicit_dihedrals: Dictionary to populate with dihedral restraints (modified in place)
        constrained_bonds: Set of bonds with constraints (modified in place)
        bond_lengths: Dictionary mapping (atom1, atom2) -> target length (modified in place)
        verbose: Verbosity level
    
    Returns:
        atom_map: Dictionary mapping mol atom indices to template atom indices (or None if no match)
    """
    # Check if template has a conformer
    if template.GetNumConformers() == 0:
        if verbose > 0:
            print("Warning: Template has no conformer, skipping template restraints")
        return None
    
    # Find MCS and atom mapping
    atom_map = _find_mcs_mapping(mol, template, verbose)
    if atom_map is None:
        return None
    
    # Get template conformer
    template_conf = template.GetConformer()
    
    # Update bond lengths for bonds in MCS to match template
    num_template_bonds = 0
    for bond in template.GetBonds():
        t_j, t_k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Check if both atoms are in the mapping
        if t_j not in atom_map.values() or t_k not in atom_map.values():
            continue
        
        # Find corresponding atoms in mol (reverse mapping)
        mol_j = None
        mol_k = None
        for mol_idx, template_idx in atom_map.items():
            if template_idx == t_j:
                mol_j = mol_idx
            if template_idx == t_k:
                mol_k = mol_idx
        
        if mol_j is None or mol_k is None:
            continue
        
        # Calculate bond length from template
        t_pos_j = template_conf.GetAtomPosition(t_j)
        t_pos_k = template_conf.GetAtomPosition(t_k)
        template_bond_length = np.linalg.norm(np.array([t_pos_j.x, t_pos_j.y, t_pos_j.z]) - 
                                              np.array([t_pos_k.x, t_pos_k.y, t_pos_k.z]))
        
        # Update bond length constraint
        bond_key = (min(mol_j, mol_k), max(mol_j, mol_k))
        bond_lengths[bond_key] = template_bond_length
        num_template_bonds += 1
        
        if verbose > 0:
            mol_sym_j = mol.GetAtomWithIdx(mol_j).GetSymbol()
            mol_sym_k = mol.GetAtomWithIdx(mol_k).GetSymbol()
            print(f"  Template bond: {mol_sym_j}{mol_j}-{mol_sym_k}{mol_k} = {template_bond_length:.3f} Å")
    
    # Extract dihedrals from template for rotatable bonds in MCS
    num_template_dihedrals = 0
    for bond in template.GetBonds():
        t_j, t_k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Check if both atoms are in the mapping
        if t_j not in atom_map.values() or t_k not in atom_map.values():
            continue
        
        # Skip ring bonds
        if bond.IsInRing():
            continue
        
        # Find corresponding atoms in mol
        mol_j = None
        mol_k = None
        for mol_idx, template_idx in atom_map.items():
            if template_idx == t_j:
                mol_j = mol_idx
            if template_idx == t_k:
                mol_k = mol_idx
        
        if mol_j is None or mol_k is None:
            continue
        
        # Get dihedral atoms in template
        t_i, t_l = _get_dihedral_atoms(template, t_j, t_k)
        if t_i is None or t_l is None:
            continue
        
        # Check if all four atoms are in mapping
        if (t_i not in atom_map.values() or t_l not in atom_map.values()):
            continue
        
        # Find corresponding atoms in mol
        mol_i = None
        mol_l = None
        for mol_idx, template_idx in atom_map.items():
            if template_idx == t_i:
                mol_i = mol_idx
            if template_idx == t_l:
                mol_l = mol_idx
        
        if mol_i is None or mol_l is None:
            continue
        
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
        print(f"  Total template bonds: {num_template_bonds}, dihedrals: {num_template_dihedrals}\n")
    
    return atom_map


def optimize_mol(mol, bond_k=1.5, angle_k=2.0, tolerance=1e-6, maxeval=5000, verbose=0,
                 dihedral: Optional[dict[tuple[int, int, int, int], float]] = None,
                 dihedral_k=5.0, repulsion_k=0.1, repulsion_cutoff=3.0, n_starts=10,
                 template: Optional[Chem.Mol] = None, 
                 template_coordinate_k=10.0, debug=False):
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
        template_coordinate_k: Force constant for template coordinate restraints (MCS atoms to template positions) (default: 10.0)
        debug: If True, print detailed debug information (default: False)
    
    Returns:
        Optimized coordinates array, shape (n_atoms, 3)
    """
    n = mol.GetNumAtoms()
    
    # Find MCS mapping if template provided
    atom_map = None
    if template is not None:
        if template.GetNumConformers() == 0:
            if verbose > 0:
                print("Warning: Template has no conformer, skipping template restraints")
        else:
            atom_map = _find_mcs_mapping(mol, template, verbose, debug=debug)
    
    # Create bonds list
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), _get_bond_length(mol, bond))
             for bond in mol.GetBonds()]
    
    # Get ring information for detecting 5-membered rings
    ring_info = mol.GetRingInfo()
    five_membered_rings = [ring for ring in ring_info.AtomRings() if len(ring) == 5]
    five_membered_atoms = set()
    for ring in five_membered_rings:
        five_membered_atoms.update(ring)
    
    # Create angles list
    angles = []
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) >= 2:
            # Check if this is an sp2 center in a 5-membered ring with 3 neighbors
            is_sp2_in_5ring = (atom.GetHybridization() == Chem.HybridizationType.SP2 and 
                              len(neighbors) == 3 and i in five_membered_atoms)
            
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    # Check if this angle is part of a 5-membered ring
                    # An angle is in a ring if both neighbor atoms and the center are in the same 5-membered ring
                    angle_in_5ring = False
                    if is_sp2_in_5ring:
                        for ring in five_membered_rings:
                            if (i in ring and neighbors[j] in ring and neighbors[k] in ring):
                                angle_in_5ring = True
                                break
                    
                    if angle_in_5ring:
                        # Ring angle in 5-membered ring: 108°
                        angle_val = np.deg2rad(108.0)
                        if verbose > 0:
                            sym = atom.GetSymbol()
                            n1_sym = mol.GetAtomWithIdx(neighbors[j]).GetSymbol()
                            n2_sym = mol.GetAtomWithIdx(neighbors[k]).GetSymbol()
                            print(f"  Angle (5-ring): {n1_sym}{neighbors[j]}-{sym}{i}-{n2_sym}{neighbors[k]} = 108.0°")
                    elif is_sp2_in_5ring:
                        # Non-ring angle for sp2 in 5-membered ring: 126°
                        angle_val = np.deg2rad(126.0)
                        if verbose > 0:
                            sym = atom.GetSymbol()
                            n1_sym = mol.GetAtomWithIdx(neighbors[j]).GetSymbol()
                            n2_sym = mol.GetAtomWithIdx(neighbors[k]).GetSymbol()
                            print(f"  Angle (non-ring): {n1_sym}{neighbors[j]}-{sym}{i}-{n2_sym}{neighbors[k]} = 126.0°")
                    else:
                        # Standard angle based on hybridization
                        angle_val = ANGLE.get(atom.GetHybridization(), ANGLE[Chem.HybridizationType.SP3])
                    
                    angles.append((neighbors[j], i, neighbors[k], angle_val))
    
    # Set up dihedrals and coordinate constraints
    explicit_dihedrals = {}
    constrained_bonds = set()
    coordinate_constraints = []
    
    # Apply template coordinate restraints if provided
    if template is not None and atom_map is not None:
        template_conf = template.GetConformer()
        for mol_idx, template_idx in atom_map.items():
            t_pos = template_conf.GetAtomPosition(template_idx)
            coordinate_constraints.append((mol_idx, t_pos.x, t_pos.y, t_pos.z))
        
        if verbose > 0:
            print(f"  Template coordinate restraints: {len(coordinate_constraints)} atoms\n")
    
    # Add user-provided dihedral constraints
    if dihedral is not None:
        for (i, j, k, l), angle_deg in dihedral.items():
            if not mol.GetBondBetweenAtoms(j, k).IsInRing():
                explicit_dihedrals[(i, j, k, l)] = np.deg2rad(angle_deg)
                constrained_bonds.add((min(j, k), max(j, k)))
    
    dihedrals = [(i, j, k, l, phi) for (i, j, k, l), phi in explicit_dihedrals.items()]
    
    if verbose > 0:
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else 'Unnamed'
        print(f"\nMolecule: {name}")
        print(f"Atoms: {n}, Bonds: {len(bonds)}, Angles: {len(angles)}, Dihedrals: {len(dihedrals)}")
        if template is not None and atom_map is not None:
            print(f"Template coordinate restraints: {len(coordinate_constraints)} atoms")
        print()
    
    if debug:
        print("[DEBUG] Creating optimizer and starting optimization...")
    
    coords, converged, energy = optimize(
        n_atoms=n, bonds=bonds, angles=angles, dihedrals=dihedrals,
        bond_force_constant=bond_k, angle_force_constant=angle_k,
        dihedral_force_constant=dihedral_k,
        repulsion_force_constant=repulsion_k, repulsion_cutoff=repulsion_cutoff,
        coordinates=coordinate_constraints, coordinate_force_constant=template_coordinate_k,
        tolerance=tolerance, maxeval=maxeval, verbose=verbose, n_starts=n_starts
    )
    
    if verbose > 0:
        print(f"Optimization converged: {converged}, Final energy: {energy:.6e}\n")
    
    return coords
