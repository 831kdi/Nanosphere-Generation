import numpy as np
from matplotlib import pyplot as plt
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.visualize import view
from pymatgen.core.structure import Structure

#reading coordinates from CONTCAR
st = Structure.from_file("CONTCAR")  #put your CONTCAR file name
atoms = st.to_ase_atoms()
write("tmp1.cif", atoms) #write a cif file named tmp1. Change according to your preference

#Counting the number of neighbours for each Rh atoms
read("tmp1.cif").get_scaled_positions()
view(atoms, viewer="ngl") #Visualise if correct CONTCAR file was transformed as the cif file
cutoff = 3.1 #cutoff should be changed manually for better surface Rh filtering (In most cases 2.9 ~ 3.1 shows good performance)
tmp_atoms = atoms.copy()
rh_indices = tmp_atoms.symbols == "Rh" # only Rh atoms
# print("Rh indices:", rh_indices) 
# distance matrix
dist_mat = atoms.get_all_distances(mic=True)
num_neigbor_atoms = (dist_mat < cutoff).sum(axis=1)
print("Number of neighbor atoms within cutoff:", num_neigbor_atoms)

#Visualise histogram to plot number of neighbours of each Rh atom
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(9, 8))
values = num_neigbor_atoms
bins = range(2, 16)
ax.hist(values, bins=bins, align="left", rwidth=0.8)
ax.set_xticks(np.arange(min(bins), max(bins)+1, 1))
plt.rcParams["font.family"] = "Arial"
plt.tight_layout()
plt.show()

# target_indices are the surface Rh atoms
target_indices = np.where((rh_indices) & (num_neigbor_atoms < 10))[0] #cutoff of the number of neighbours should be manually changed after visualising the atoms chosen as the surface Rh atoms
print(target_indices+1)
# Add target atoms with dummy atom type to visualize
for i in target_indices:
    tmp_atoms[i].symbol = "O"
view(tmp_atoms, viewer="ngl") #This visualise atoms that has been chosen as a surface Rh atom to be in a smaller red dot



