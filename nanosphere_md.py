
"""
Workflow for generating, relaxing, and running MD on nanospheres using Packmol and MatterSim.

Workflow:
1. Generate a nanosphere (default 1.5 nm diameter) with a given chemical composition (e.g. "Rh3P2")
   by filling the sphere with atoms.
   Packmol is used to pack atoms while preserving the composition.
   The number of atoms generated is logged.
2. Repeat 1. 100 times; for each, relax the structure for a given number of steps (default 50) 
   using the Mace-MP force field, then compute the energy per atom (eV/atom).
   Sort the 100 nanospheres by energy per atom and select the 10 lowest–energy ones.
3. For each of these 10 nanospheres, embed it in a cubic vacuum cell whose side length is:
      cell_side = (nanosphere_diameter) + 2*2 Å. (Too big vacuum cause errors in DFT calculation)
   Save each structure as a CIF file.
4. Run an NVT MD simulation on each embedded nanosphere using the ASE environment with:
   - Langevin thermostat (default; user–option available)
   - MatterSim force field (default; user–option available)
   - 1 fs timestep (default; option available)
   - Velocity–Verlet integration (default; option available)
   MD is run in three segments:
      (a) Heating: from T_start (330 K default) to T_end (800 K default) over t_heat (100 ps default),
          with a constant heating rate.
      (b) Hold at 800 K for t_hold (20 ps default).
      (c) Cooling: from 800 K to T_final (300 K default) over t_cool (30 ps default).
   (Here we assume a constant cooling rate for simplicity.)
5. Save the final MD structure for each nanosphere as a CIF file named MD_composition_i.cif.

Usage (via Fire):
    python nanosphere_md.py --composition Rh3P2

Author: Dongin Kim
Date: 11/Aug/2025
"""

import os
import subprocess
import tempfile
import math
import shutil
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from ase import Atoms, io
from ase.optimize import BFGS
from ase.md.langevin import Langevin
from ase import units

# You can use Fire to create a command-line interface
import fire
from tqdm import tqdm

#########################################
# Helper functions
#########################################

def parse_composition(comp_str: str) -> Dict[str, int]:
    """
    Parse a composition string like "Rh3P2" into a dictionary: {"Rh":3, "P":2}
    """
    import re
    pattern = re.compile(r"([A-Z][a-z]*)(\d*)")
    comp = {}
    for (elem, count) in pattern.findall(comp_str):
        comp[elem] = int(count) if count != "" else 1
    return comp

def compute_total_atoms(sphere_diameter: float, atomic_volume: float = 15.0) -> int:
    """
    Compute total number of atoms in a sphere of given diameter (in Å).
    """
    radius = sphere_diameter / 2.0
    volume = (4.0/3.0) * math.pi * (radius**3)
    total_atoms = int(round(volume / atomic_volume))
    return total_atoms

def create_single_atom_xyz(element: str, filename: Path):
    """
    Create a simple xyz file containing one atom of the given element at (0,0,0).
    Includes a comment line as required by XYZ format.
    """
    with open(filename, "w") as f:
        f.write("1\n")                      # number of atoms
        f.write(f"{element} atom\n")        # comment line (can be anything)
        f.write(f"{element} 0.0 0.0 0.0\n") # atomic coordinates

def run_packmol(packmol_input: Path, packmol_exe: str = "packmol"):
    """
    Run Packmol with the specified input file.
    """
    try:
        subprocess.run([packmol_exe, "<", str(packmol_input)], shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Packmol failed: {e}")

def generate_nanosphere(composition: str,
                         sphere_diameter: float = 15.0,
                         atomic_volume: float = 15.0,
                         packmol_exe: str = "packmol") -> Tuple[Atoms, int]:
    """
    Generate a nanosphere with given composition using Packmol.
    
    Args:
        composition: Chemical composition string (e.g. "Rh3P2").
        sphere_diameter: Diameter in Å (default 15 Å for 1.5 nm).
        atomic_volume: Atomic volume in Å³/atom (default 15).
        packmol_exe: Path to packmol executable (default "packmol").
        
    Returns:
        A tuple (atoms, total_atoms) where atoms is an ASE Atoms object
        representing the nanosphere, and total_atoms is the number of atoms.
    """
    comp_dict = parse_composition(composition)
    total_parts = sum(comp_dict.values())
    total_atoms = compute_total_atoms(sphere_diameter, atomic_volume)
    
    # Determine number of atoms per element
    element_counts = {elem: int(round(total_atoms * count / total_parts)) for elem, count in comp_dict.items()}
    # Update total_atoms to reflect the sum of rounded values
    total_atoms = sum(element_counts.values())
    
    print(f"[Packmol] Composition: {composition} parsed as {comp_dict}")
    print(f"[Packmol] Target sphere diameter: {sphere_diameter} Å, estimated total atoms: {total_atoms}")
    print(f"[Packmol] Atom counts: {element_counts}")
    
    # Create a temporary working directory
    temp_dir = Path(tempfile.mkdtemp(prefix="packmol_"))
    # Create single-atom xyz files for each element in the temp directory
    for elem in element_counts.keys():
        create_single_atom_xyz(elem, temp_dir / f"{elem}.xyz")
    
    # Write the Packmol input file
    packmol_input = temp_dir / "packmol_input.inp"
    with open(packmol_input, "w") as f:
        f.write("tolerance 2.0\n")
        f.write("filetype xyz\n")
        f.write(f"output nanosphere.xyz\n\n")
        for elem, count in element_counts.items():
            f.write(f"structure {temp_dir / f'{elem}.xyz'}\n")
            f.write(f"  number {count}\n")
            f.write(f"  inside sphere 0. 0. 0. {sphere_diameter/2.0}\n")
            f.write("end structure\n\n")
    
    # Run Packmol (using shell redirection since packmol reads from stdin)
    cmd = f"{packmol_exe} < {packmol_input}"
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=temp_dir)
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"Packmol execution failed: {e}")
    
    # Read the output nanosphere file
    nanosphere_file = temp_dir / "nanosphere.xyz"
    atoms = io.read(str(nanosphere_file))
    
    print(f"[Packmol] Generated nanosphere with {len(atoms)} atoms.")
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    return atoms, len(atoms)

def relax_structure(atoms: Atoms,
                    relax_steps: int = 50,
                    calculator=None) -> Atoms:
    """
    Relax the structure for a given number of steps using BFGS.
    The user must provide a calculator (default is MatterSim).
    """
    if calculator is None:
        # Import the MACE calculator
        from mace.calculators import mace_mp
        calculator = mace_mp(device="cpu", default_dtype="float32")
    atoms.calc = calculator
    dyn = BFGS(atoms, logfile="relax.log")
    dyn.run(fmax=0.05, steps=relax_steps)
    energy = atoms.get_potential_energy()
    energy_per_atom = energy / len(atoms)
    print(f"[Relax] Energy: {energy:.3f} eV, Energy/atom: {energy_per_atom:.6f} eV/atom")
    return atoms

def embed_in_vacuum(atoms: Atoms, nanosphere_diameter: float = 15.0, buffer: float = 2.0) -> Atoms:
    """
    Embed the nanosphere in a cubic vacuum cell with the nanosphere at the center.
    The cell side length is: nanosphere_diameter + 2*buffer.
    """
    cell_side = nanosphere_diameter + 2 * buffer
    atoms.set_cell([cell_side, cell_side, cell_side])
    atoms.center()  # center atoms inside the unit cell
    atoms.set_pbc([True, True, True])
    return atoms


def run_md_simulation(atoms,
                      dt_fs=0.25,
                      t_heat_ps=10.0,
                      t_hold_ps=12.0,
                      t_cool_ps=6.0,
                      T_start=330.0,
                      T_end=800.0,
                      T_final=300.0,
                      calculator=None,
                      logfile="md.log"):
    # calculator
    if calculator is None:
        from mattersim.forcefield import MatterSimCalculator
        calculator = MatterSimCalculator(
            load_path="/home/intern_02/Dongin/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth",
            device="cpu"
        )
    atoms.calc = calculator

    # init velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_start)
    Stationary(atoms)

    # steps
    n_heat = max(1, int(t_heat_ps * 1000.0 / dt_fs))
    n_hold = max(1, int(t_hold_ps * 1000.0 / dt_fs))
    n_cool = max(1, int(t_cool_ps * 1000.0 / dt_fs))

    # Langevin (note friction units!)
    dyn = Langevin(atoms,
                   dt_fs * units.fs,
                   temperature_K=T_start,
                   friction=0.02 / units.fs,   # e.g., 0.02 fs^-1
                   logfile=logfile)
    # trajectory
    traj = Trajectory("md.traj", "w", atoms)

    # Heating
    for step in range(n_heat):
        T_target = T_start + (T_end - T_start) * ((step + 1) / n_heat)
        dyn.set_temperature(temperature_K=T_target)   # <-- FIX
        dyn.run(1)
        if step % 10 == 0:
            traj.write()

    # Hold
    dyn.set_temperature(temperature_K=T_end)          # <-- FIX
    dyn.run(n_hold)

    # Cooling
    for step in range(n_cool):
        T_target = T_end - (T_end - T_final) * ((step + 1) / n_cool)
        dyn.set_temperature(temperature_K=T_target)   # <-- FIX
        dyn.run(1)
        if step % 10 == 0:
            traj.write()

    traj.close()
    return atoms
                         
#Main Function Below

def run_md_simulation(atoms,
                      dt_fs=0.5,
                      t_heat_ps=10.0,
                      t_hold_ps=12.0,
                      t_cool_ps=6.0,
                      T_start=330.0,
                      T_end=800.0,
                      T_final=300.0,
                      thermostat="langevin",
                      integrator="velocity_verlet",
                      calculator=None,
                      logfile="md.log"):

    if calculator is None:
        from mattersim.forcefield import MatterSimCalculator
        calculator = MatterSimCalculator(load_path="/home/intern_02/Dongin/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth", device="cpu") 
    atoms.calc = calculator
    #change the load_path according to where you downloaded the mattersim force field.
    #the device was selected as cpu for default as this calculation does not require extensive computational power. However gpu cuda can be applied as well.

                         
    # Assign initial velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_start)

    # Convert times to steps
    n_heat = int(t_heat_ps * 1000 / dt_fs)
    n_hold = int(t_hold_ps * 1000 / dt_fs)
    n_cool = int(t_cool_ps * 1000 / dt_fs)

    dyn = Langevin(atoms, dt_fs * units.fs, temperature_K=T_start, friction=0.0001, logfile=logfile)

    print(f"[MD] Starting MD simulation with {n_heat + n_hold + n_cool} steps")

    # Heating phase
    for step in range(n_heat):
        T_target = T_start + (T_end - T_start) * (step / n_heat)
        rescale_velocities(atoms, T_target)
        dyn.set_temperature(T_target)
        dyn.run(1)
        if step % 10 == 0:
            print(f"[Heating] Step {step}, T_target = {T_target:.1f} K")
    

    # Hold at T_end
    dyn.set_temperature(T_end)
    for step in range(n_hold):
        rescale_velocities(atoms, T_end)
        dyn.run(1)


    # Cooling phase
    for step in range(n_cool):
        T_target = T_end - (T_end - T_final) * (step / n_cool)
        rescale_velocities(atoms, T_target)
        dyn.set_temperature(T_target)
        dyn.run(1)
        if step % 10 == 0:
            print(f"[Heating] Step {step}, T_target = {T_target:.1f} K")

    print("[MD] Simulation complete")
    return atoms

#########################################
# Main workflow function
#########################################

def main(composition: str = "Rh3P2",
         sphere_diameter: float = 15.0,       # in Å (1.5 nm)
         relax_steps: int = 50,
         num_initial: int = 100,
         num_select: int = 10,
         cube_buffer: float = 2.0,          # buffer in Å
         dt_fs: float = 0.5,                 # timestep in fs
         t_heat_ps: float = 10.0,
         t_hold_ps: float = 12.0,
         t_cool_ps: float = 6.0,
         T_start: float = 330.0,
         T_end: float = 800.0,
         T_final: float = 300.0,
         output_dir: str = None,  # If None, defaults to the directory of the execution file.
         packmol_exe: str = "packmol"):
    """
    Main workflow:
      1. Generate 'num_initial' nanospheres using Packmol with the given composition.
      2. Relax each nanosphere for 'relax_steps' steps using MatterSim force field.
      3. Compute energy per atom and select 'num_select' lowest–energy nanospheres.
      4. Embed each selected nanosphere in a cubic vacuum cell.
      5. Run an NVT MD simulation (heating, hold, cooling) for each.
      6. Save the final MD structure as a CIF file.

    If output_dir is not provided, the output files will be saved in the same directory as the execution file.
    """
    # Determine output directory: default to the execution file directory if not provided
    if output_dir is None:
        output_path = Path(__file__).parent
    else:
        output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    nanosphere_list: List[Tuple[Atoms, float]] = []  # (atoms, energy_per_atom)
    print(f"[Workflow] Generating {num_initial} nanospheres for composition {composition} ...")
    
    # Loop to generate and relax nanospheres
    for i in tqdm(range(num_initial), desc="Generating nanospheres"):
        # 1. Generate nanosphere using Packmol
        atoms, natoms = generate_nanosphere(composition, sphere_diameter, atomic_volume=15.0, packmol_exe=packmol_exe)
        # 2. Relax structure (using MatterSim force field)
        atoms = relax_structure(atoms, relax_steps=relax_steps)
        energy = atoms.get_potential_energy()
        energy_per_atom = energy / len(atoms)
        nanosphere_list.append((atoms, energy_per_atom))
    
    # Sort by energy per atom (lowest first)
    nanosphere_list.sort(key=lambda x: x[1])
    selected = nanosphere_list[:num_select]
    print(f"[Workflow] Selected top {num_select} nanospheres with lowest energy per atom:")
    for idx, (atoms, energy_per_atom) in enumerate(selected, start=1):
        print(f"   Nanosphere {idx}: {len(atoms)} atoms, Energy/atom = {energy_per_atom:.6f} eV")
    
   # For each selected nanosphere, embed in cubic cell and run MD simulation
    for idx, (atoms, _) in enumerate(selected, start=1):
        # Embed in cubic cell: cell side = sphere_diameter + 2*buffer
        atoms = embed_in_vacuum(atoms, sphere_diameter, cube_buffer)
        # Save initial relaxed structure for reference (optional)
        init_filename = output_path / f"nanosphere_{composition}_{idx}_init.cif"
        io.write(str(init_filename), atoms)
        print(f"[Embed] Saved initial embedded nanosphere to {init_filename}")
        # === DIAGNOSTIC BLOCK START ===
        from ase.optimize import BFGS
        import numpy as np
        from mace.calculators import mace_mp
        calculator = mace_mp(device="cpu", default_dtype="float32")

        atoms.calc = calculator  # reuse the same force field

        # Print initial energy and forces
        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        print(f"[Diagnostic] Initial potential energy: {initial_energy:.4f} eV")
        print(f"[Diagnostic] Max force before relaxation: {np.abs(initial_forces).max():.4f} eV/Å")

        # Run 20-step BFGS relaxation
        print("[Diagnostic] Running 20-step BFGS relaxation...")
        relax = BFGS(atoms, logfile='relax_diagnostic.log')
        relax.run(fmax=0.01, steps=1000)

        # Print results
        relaxed_energy = atoms.get_potential_energy()
        relaxed_forces = atoms.get_forces()
        print(f"[Diagnostic] Energy after relaxation: {relaxed_energy:.4f} eV")
        print(f"[Diagnostic] Max force after relaxation: {np.abs(relaxed_forces).max():.4f} eV/Å")
        print(f"[Diagnostic] Energy change: {relaxed_energy - initial_energy:.4f} eV")
        # === DIAGNOSTIC BLOCK END ===
    
        relaxed_filename = output_path / f"relaxed_nanosphere_{composition}_{idx}.cif"
        io.write(str(relaxed_filename), atoms)
        print(f"[BFGS] Saved relaxed structure to: {relaxed_filename}")

        # Run MD simulation
        atoms_md = run_md_simulation(atoms.copy(), dt_fs=dt_fs,
                                     t_heat_ps=t_heat_ps,
                                     t_hold_ps=t_hold_ps,
                                     t_cool_ps=t_cool_ps,
                                     T_start=T_start,
                                     T_end=T_end,
                                     T_final=T_final,
                                     logfile=f"md_{composition}_{idx}.log")

        # Save final MD structure as CIF file with naming convention MD_composition_idx.cif
        final_filename = output_path / f"MD_{composition}_{idx}.cif"
        io.write(str(final_filename), atoms_md)
        print(f"[MD] Saved final MD structure to {final_filename}")

if __name__ == "__main__":
    # Use Fire to expose command-line interface.
    fire.Fire(main)
