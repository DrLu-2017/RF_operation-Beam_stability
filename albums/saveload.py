"""
Module for save/load and utility functions.
"""
import numpy as np
import h5py as hp

# %% Utility Functions
def save_hdf5(filename, data_dict):
    """Saves data dictionary to an HDF5 file."""
    with hp.File(filename, "a") as f:
        for key, value in data_dict.items():
            f[key] = value

def load_hdf5(filename, keys):
    """Loads data from an HDF5 file given a list of keys."""
    with hp.File(filename, "r") as f:
        return {key: np.array(f[key]) for key in keys}

# %% Save/Load Functions
def save_out(name, out):
    """Save output data to an HDF5 file."""
    keys = [
        "zero_freq_coup", "robinson_coup", "modes_coup", "HOM_coup",
        "converged_coup", "PTBL_coup", "bl", "xi", "R"
    ]
    save_hdf5(f"{name}.hdf5", dict(zip(keys, out)))

def load_out(file):
    """Load output data from an HDF5 file."""
    keys = [
        "zero_freq_coup", "robinson_coup", "modes_coup", "HOM_coup",
        "converged_coup", "PTBL_coup", "bl", "xi", "R"
    ]
    data = load_hdf5(file, keys)
    return tuple(data[key] for key in keys)

def save_out_opti(name, out):
    """Save optimization results to an HDF5 file."""
    keys = ["psi_result", "bl_result", "xi_result", "R_result"]
    save_hdf5(f"{name}.hdf5", dict(zip(keys, out)))

def load_out_opti(file):
    """Load optimization results from an HDF5 file."""
    keys = ["psi_result", "bl_result", "xi_result", "R_result"]
    data = load_hdf5(file, keys)
    return tuple(data[key] for key in keys)
