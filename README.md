# Nanosphere-Generation
# Nanosphere MD Workflow & Surface Rh Analysis

This repository provides three Python scripts for **nanoparticle generation/relaxation/molecular dynamics (MD)**, **surface Rh atom identification**, and **dependency checking**.  
You can simply download the `.py` files into any folder on your machine and run them with Python.

---

## Files in This Repository
- **`nanosphere_md.py`** – Generates spherical nanoparticles (Packmol) for a given composition, relaxes them, selects low-energy structures, inserts into vacuum, and runs heating–holding–cooling MD simulations.
- **`Surface_Rh.py`** – Reads a VASP `CONTCAR`, estimates surface Rh atoms by neighbor count, and visualizes them with `nglview`.
- **`check_imports.py`** – Scans the scripts for imported modules and checks if they are installed.

---

## Requirements
Before running the scripts, ensure you have:
- **Python 3.9+** installed  
- Required Python packages:
  pip install numpy matplotlib ase pymatgen tqdm fire nglview
Packmol installed and available in your system PATH:

which packmol
(Optional) mace-torch or mattersim if you want to use ML force fields for MD.

## How to Use

1. **Download the scripts**  

   Either:  
   - Download each file from GitHub and place them in the same folder, **OR**  
   - Use `git clone` to copy the entire repository:  
     ```bash
     git clone https://github.com/831kdi/Nanosphere-Generation.git
     cd Nanosphere-Generation
     ```

2. **Check dependencies**  

   Run:
   ```bash
   python check_imports.py
   ```
   It will list any missing Python packages. Install them with:
   ```bash
   pip install <package>
   ```

3. **Run a nanoparticle MD simulation**

   Example:
   ```bash
   python nanosphere_md.py --composition Rh3P2
   ```
   This will:
   - Generate ~1.5 nm spheres via Packmol
   - Relax with BFGS
   - Select low-energy structures
   - Insert into vacuum
   - Run heating–holding–cooling MD
   - Save CIF and log files in the current folder
   See the top of nanosphere_md.py for adjustable parameters.

4. **Identify surface Rh atoms**

   Copy the .py script and make a .ipynb file and paste the script.

   Open it in a IDE of your own choice (jupyter notebook or visual studio code recommended) to visualise the surface Rh atoms
   
   This will:
   - Convert CONTCAR → CIF
   - Count neighbors (default cutoff = 3.1 Å)
   - Print surface Rh atom indices (neighbor count < 10)
   - Open an nglview visualization (surface Rh marked as O)

