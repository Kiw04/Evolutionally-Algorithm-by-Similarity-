#pkg_resource error fix by pip install setup tools 
from cryspy.interactive import action
from ase.calculators.emt import EMT
import ipywidgets as widgets
from dscribe.descriptors import SOAP
import numpy as np
from cryspy.interactive import action
import pickle
import os
from ase.io import read
import re
from ase import Atoms
import glob
from ase.io import read
from scipy.spatial.distance import euclidean, cosine
import numpy as np
import math



def struc_gen():
    action.clean(skip_yes=True)
    #Type yes to clean data
    action.initialize()
    # ---------- EMT in ASE
    calculator = EMT()
    widgets.IntProgress()
    # ---------- structure optimization
    action.restart(
    njob=0,    # njob=0: njob in cryspy.in will be used
    calculator=calculator,
    optimizer='FIRE',    # 'FIRE', 'BFGS' or 'LBFGS'
    symmetry=True,       # default: True
    fmax=0.01,           # default: 0.01
    steps=2000,          # default: 2000
    )
    return None


def split_poscar_file(input_filename, output_folder):
    """
    Reads a file containing multiple POSCAR structures, creates a specified
    output folder, and writes each structure to a separate file within it.

    Args:
        input_filename (str): The name of the input file to read.
        output_folder (str): The name of the folder to save the files in.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output will be saved in the '{output_folder}' directory.")

    try:
        with open(input_filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
        return
    # Use a regular expression to split the file content by 'ID_X' lines
    structures = re.split(r'(?=ID_\d+)', content)

    # The first split element might be empty if the file starts with the delimiter
    if structures and not structures[0].strip():
        structures.pop(0)

    for i, structure_str in enumerate(structures):
        # Clean up the string to remove the ID line
        lines = structure_str.strip().split('\n')
        
        # Get everything after the ID line and prepend the new element line
        poscar_content = 'Cu Au\n' + '\n'.join(lines[1:]) # <-- MODIFIED LINE

        # Construct the full path for the output file
        output_filename = os.path.join(output_folder, f'{i}_POSCAR')
        
        # Write the content to the new POSCAR file
        with open(output_filename, 'w') as f_out:
            f_out.write(poscar_content)
        
        print(f"  -> Created {output_filename}")
    return None

def SOAP_Calculation(r_cut,n_max,l_max):
    # --- 1. Configuration ---
    # Specify the path to the folder containing your POSCAR files
    poscar_folder = './gen_1_structure/'
    
    # SOAP descriptor hyperparameters (adjust as needed)
 
    periodic = True

    # --- 2. Dynamically Determine All Species ---
    # It's crucial to provide SOAP with a list of all possible chemical species
    # that can appear in your structures. We'll find them automatically.
    print("Scanning files to determine all chemical species...")
    all_species = set()
    global file_list
    file_list = [f for f in os.listdir(poscar_folder) if f.endswith('POSCAR')]

    for filename in file_list:
        try:
            filepath = os.path.join(poscar_folder, filename)
            atoms = read(filepath, format='vasp')
            all_species.update(atoms.get_chemical_symbols())
        except Exception as e:
            print(f"Could not read or process {filename}. Error: {e}")

    # Convert the set to a sorted list for consistent ordering
    species = sorted(list(all_species))
    if not species:
        raise ValueError("No valid POSCAR files found or no species detected in the folder.")
    print(f"Detected species: {species}")


    # --- 3. Initialize the SOAP Descriptor ---
    # The descriptor is initialized once with all parameters.
    global average_soap
    average_soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average="inner",  # This creates the global average descriptor
        sparse=False,
        periodic=periodic
    )
    
    print("\nSOAP descriptor initialized.")


    # --- 4. Calculate SOAP Vectors for Each Structure ---
    soap_vectors = []
    all_atoms = []
    umap_atom_color = []

    print(f"\nProcessing {len(file_list)} structures...")
    for filename in file_list:
        filepath = os.path.join(poscar_folder, filename)
        try:
            # Read the structure from the POSCAR file using ASE
            atoms = read(filepath, format='vasp')

            # Create the average SOAP vector for the structure and store it
            soap_vector = average_soap.create(atoms)
            soap_vectors.append(soap_vector)
            if filename == "CuAu_Stn2_POSCAR":
                umap_atom_color.append("red")
            elif filename == "CuAu_Stn3_POSCAR":
                umap_atom_color.append("green")
            else:
                umap_atom_color.append("black")
            # Optionally keep the ASE Atoms object
            all_atoms.append(atoms)
            print(f"  - Successfully processed {filename}")

        except Exception as e:
            print(f"  - Skipping {filename} due to an error: {e}")

    # Convert the list of vectors into a single NumPy array
    global soap_array
    soap_array = np.array(soap_vectors)

    # --- 5. Final Output ---
    print("\n-------------------------------------------")
    print(f"Total {len(soap_array)} structures successfully processed.")
    print(f"Shape of the final SOAP descriptor array: {soap_array.shape}")
    print("-------------------------------------------")
    return None
        
def cos_sim(p, q):
    return 1-(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)))


def smallest(arr,n):
  smallest_items = []
  for i, number in enumerate(arr):
    # If the list isn't full yet, just add the new item.
    if len(smallest_items) < n:
      smallest_items.append((number, i))
      # Sort this small list after adding to keep the largest at the end
      # A simple bubble sort is used here to avoid the built-in sorted()
      for j in range(len(smallest_items) - 1, 0, -1):
        if smallest_items[j][0] < smallest_items[j-1][0]:
          smallest_items[j], smallest_items[j-1] = smallest_items[j-1], smallest_items[j]
        else:
          break
      continue

    # If the list is full, check if the new number is smaller than the largest item in our list
    # The largest item will be at the end of our sorted list (index n-1)
    if number < smallest_items[n-1][0]:
      # Replace the largest item with the new, smaller item
      smallest_items[n-1] = (number, i)
      # Re-sort the list to maintain order.
      for j in range(len(smallest_items) - 1, 0, -1):
        if smallest_items[j][0] < smallest_items[j-1][0]:
            smallest_items[j], smallest_items[j-1] = smallest_items[j-1], smallest_items[j]
        else:
            break
    result_indices = [item[1] for item in smallest_items]
    '''
    print("Cosine ranking:")
    for number,i in enumerate(smallest(cosine_similarities,7)):
        print(file_list[i],smallest_items[number][0])
    '''
  return result_indices

def cosine_sim(path: str):
    # 1. Read the standard structure and generate its SOAP vector
    atoms_bm_standard = read(path)
    average_bm = average_soap.create(atoms_bm_standard)

    # 2. Iterate through soap_array (a list or array of SOAP vectors)
    cosine_similarities = []

    for i, soap_vec in enumerate(soap_array):
        # Ensure vectors are flattened if necessary
        vec = np.ravel(soap_vec)
        ref = np.ravel(average_bm)

        # Euclidean distance
        euclid = euclidean(ref, vec)

        cosine_sim = cos_sim(ref, vec)
        cosine_similarities.append(cosine_sim)

    # Optional: Convert to NumPy arrays for easy plotting/processing
    cosine_similarities = np.array(cosine_similarities)

    print('Cosine Value:',len(cosine_similarities))
    return cosine_similarities

if __name__ == '__main__':
    input_file = './data/opt_POSCARS'  
    output_dir = 'gen_1_structure'        
    split_poscar_file(input_file, output_dir)
    
