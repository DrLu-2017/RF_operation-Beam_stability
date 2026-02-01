"""
Module where parameter scans and optimization scans are defined.
"""
import numpy as np
from albums.robinson import RobinsonModes
from tqdm import tqdm
from mpi4py import MPI
from albums.saveload import save_out, save_out_opti
from albums.plot_func import __plot_modes, __plot_opti, __plot 
from albums.optimiser import __get_vals, __get_vals_equilibrium
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Helper function to initialize arrays
def initialize_arrays(var1, var2, mode_coupling):
    N = 2 if mode_coupling else 4
    shape_2d = (len(var1), len(var2))
    shape_3d = (len(var1), len(var2), N)
    
    return {
        "zero_freq_coup": np.zeros(shape_2d, dtype=bool),
        "robinson_coup": np.zeros((*shape_2d, 4), dtype=bool),
        "modes_coup": np.zeros(shape_3d, dtype=float),
        "HOM_coup": np.zeros(shape_2d, dtype=bool),
        "xi": np.zeros(shape_2d, dtype=float),
        "converged_coup": np.zeros(shape_3d, dtype=bool),
        "PTBL_coup": np.zeros(shape_2d, dtype=bool),
        "bl": np.zeros(shape_2d, dtype=float),
        "R": np.zeros(shape_2d, dtype=float),
    }

def initialize_arrays_1D(var1, mode_coupling):
    """Initialize arrays for 1D scans (single parameter sweep)."""
    N = 2 if mode_coupling else 4
    
    return {
        "zero_freq_coup": np.zeros(len(var1), dtype=bool),
        "robinson_coup": np.zeros((len(var1), 4), dtype=bool),
        "modes_coup": np.zeros((len(var1), N), dtype=float),
        "HOM_coup": np.zeros(len(var1), dtype=bool),
        "xi": np.zeros(len(var1), dtype=float),
        "converged_coup": np.zeros((len(var1), N), dtype=bool),
        "PTBL_coup": np.zeros(len(var1), dtype=bool),
        "bl": np.zeros(len(var1), dtype=float),
        "R": np.zeros(len(var1), dtype=float),
    }

# Generalized scan function
def __scan(func, var1, var2, **kwargs):
    if size not in (1, len(var1)):
        raise ValueError("len(var1) != size")
    mode_coupling = kwargs['other_kwargs'].get("mode_coupling", True)
    skip = kwargs.get("skip", False)

    var1_local = np.array_split(var1, size)[rank]
    arrays_local = initialize_arrays(var1_local, var2, mode_coupling)

    for i, v1 in enumerate(var1_local):
        var_skip = False
        for j, v2 in enumerate(tqdm(var2, desc=f'rank={rank}', position=rank)):
            results = func(v1, v2, **kwargs)
            bunch_length, zero_frequency, robinson, HOM, Omega, PTBL, converged, xi, R = results
            if not np.asarray(converged).any():
                if skip:
                    if var_skip:
                        break
                    var_skip = True
                continue
            var_skip = False

            arrays_local["zero_freq_coup"][i, j] = zero_frequency
            arrays_local["robinson_coup"][i, j, :] = robinson
            arrays_local["modes_coup"][i, j, :] = Omega
            arrays_local["HOM_coup"][i, j] = HOM
            arrays_local["converged_coup"][i, j, :] = converged
            arrays_local["PTBL_coup"][i, j] = PTBL
            arrays_local["bl"][i, j] = bunch_length * 1e12
            arrays_local["xi"][i, j] = xi
            arrays_local["R"][i, j] = R

    results = {key: None for key in arrays_local} if rank != 0 else initialize_arrays(var1, var2, mode_coupling)
    for key, local_array in arrays_local.items():
        comm.Gather(local_array, results[key], root=0)
    return (results['zero_freq_coup'], results['robinson_coup'],
    results['modes_coup'], results['HOM_coup'], results['converged_coup'],
            results['PTBL_coup'], results['bl'], results['xi'], results["R"])

# Scan in one dimension
def __scan_1D(MC, HC, ring, psi_HC_vals, current, mode_coupling, tau_boundary, method, **kwargs):
    arrays = initialize_arrays_1D(psi_HC_vals, mode_coupling)

    for j, psi_HC in enumerate(tqdm(psi_HC_vals)):
        HC.psi = psi_HC * np.pi / 180
        solver = RobinsonModes(ring, [MC, HC], current, tau_boundary=tau_boundary)
        results = solver.solve(method=method, mode_coupling=mode_coupling, **kwargs)
        (
            bunch_length, zero_frequency, robinson, HOM, Omega, PTBL,
            converged
        ) = results

        if not np.asarray(converged).any():
            continue

        arrays["zero_freq_coup"][j] = zero_frequency
        arrays["robinson_coup"][j, :] = robinson
        arrays["modes_coup"][j, :] = Omega
        arrays["HOM_coup"][j] = HOM
        arrays["converged_coup"][j, :] = converged
        arrays["PTBL_coup"][j] = PTBL
        arrays["bl"][j] = bunch_length * 1e12
        arrays["xi"][j] = solver.xi
        arrays["R"][j] = solver.R_factor(method)

    return tuple(arrays.values())

#%% follow modes at given current
def scan_modes(MC,
             HC,
             ring,
             psi_HC_vals,
             current,
             mode_coupling,
             tau_boundary,
             method,
             **kwargs):
    mode_coupling_states = [mode_coupling] if mode_coupling is not None else [False, True]

    for mc in mode_coupling_states:
        results = __scan_1D(MC, HC, ring, psi_HC_vals, current, mc, tau_boundary, method, **kwargs)
        __plot_modes(results, psi_HC_vals, mc)



def __scan_opti(func,
                var1,
                var2,
                **kwargs):
    if size not in (1, len(var1)):
        raise ValueError("len(var1) != size")

    shape = (len(var1), len(var2))
    results = {key: np.zeros(shape, dtype=float) for key in ["psi", "bl", "xi", "R"]}

    local_var1 = np.array_split(var1, size)[rank]

    local_results = {key: np.zeros((len(local_var1), len(var2)), dtype=float) for key in results}

    for i, v1 in enumerate(local_var1):
        for j, v2 in enumerate(tqdm(var2, desc=f"rank={rank}", position=rank)):
            psi, bunch_length, xi_val, R = func(v1, v2, **kwargs)
            local_results["psi"][i, j] = psi
            local_results["bl"][i, j] = bunch_length * 1e12
            local_results["xi"][i, j] = xi_val
            local_results["R"][i, j] = R

    for key in results:
        comm.Allgather(local_results[key], results[key])

    comm.Barrier()

    return tuple(results.values())

def __scan_after_opti(func,
                      psi,
                      var1,
                      var2,
                      psi_add,
                      **kwargs):
    if size not in (1, len(var1)):
        raise ValueError("len(var1) != size")

    mode_coupling = kwargs.get("other_kwargs", {}).get("mode_coupling", True)

    var1_local = np.array_split(var1, size)[rank]

    local_arrays = initialize_arrays(var1_local, var2, mode_coupling)

    for i, v1 in enumerate(var1_local):
        for j, v2 in enumerate(tqdm(var2, desc=f'rank={rank}', position=rank)):
            idx = rank if size > 1 else i
            results = func(psi[idx, j] + psi_add, v1, v2, **kwargs)
            (
                bunch_length, zero_frequency, robinson, HOM, Omega, PTBL,
                converged, xi, R
            ) = results

            if not np.asarray(converged).any():
                continue

            local_arrays["zero_freq_coup"][i, j] = zero_frequency
            local_arrays["robinson_coup"][i, j, :] = robinson
            local_arrays["modes_coup"][i, j, :] = Omega
            local_arrays["HOM_coup"][i, j] = HOM
            local_arrays["converged_coup"][i, j, :] = converged
            local_arrays["PTBL_coup"][i, j] = PTBL
            local_arrays["bl"][i, j] = bunch_length * 1e12
            local_arrays["xi"][i, j] = xi
            local_arrays["R"][i, j] = R

    gathered_results = initialize_arrays(var1, var2, mode_coupling) if rank == 0 else {
        key: None for key in local_arrays
    }

    for key in local_arrays:
        comm.Gather(local_arrays[key], gathered_results[key], root=0)

    return (gathered_results['zero_freq_coup'], 
            gathered_results['robinson_coup'],
            gathered_results['modes_coup'], 
            gathered_results['HOM_coup'], 
            gathered_results['converged_coup'],
            gathered_results['PTBL_coup'], 
            gathered_results['bl'], 
            gathered_results['xi'], 
            gathered_results["R"])

def __psi_I0(psi, I0, **kwargs):
    kwargs["HC"].psi = psi * np.pi / 180
    solver = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], I0, tau_boundary=kwargs["tau_boundary"])
    results = solver.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    return (*results, solver.xi, solver.R_factor(kwargs["method"]))

# Scan psi and current
def scan_psi_I0(name, MC, HC, ring, psi_HC_vals, currents, method, tau_boundary=None, save=True, **other_kwargs):
    kwargs = {
        "MC": MC,
        "HC": HC,
        "tau_boundary": tau_boundary,
        "method": method,
        "ring": ring,
        "name": name,
        "other_kwargs": other_kwargs
    }

    output = __scan(__psi_I0, psi_HC_vals, currents, **kwargs)

    if rank == 0:
        if save:
            save_out(f"{name}_{method}", output)
        plot_type = other_kwargs.get("plot_type", False)
        if not plot_type:
            for param in ["xi", "bunch_length", "R"]:
                __plot(output, psi_HC_vals, currents, 1, 1e3, 'Tuning angle [°]', 'Current [mA]', param, save, **kwargs)
        else:
            __plot(output, psi_HC_vals, currents, 1, 1e3, 'Tuning angle [°]', 'Current [mA]', plot_type, save, **kwargs)
    
    # Return results for Streamlit/programmatic access
    return output
   
# Function for psi and R/Q scan
def __psi_RoQ(psi, RoQ, **kwargs):
    kwargs["HC"].Rs = RoQ * kwargs["HC"].Q
    kwargs["HC"].psi = psi * np.pi / 180

    solver = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], kwargs["I0"], tau_boundary=kwargs["tau_boundary"])
    results = solver.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    
    return (*results, solver.xi, solver.R_factor(kwargs["method"]))

# Scan psi and R/Q
def scan_psi_RoQ(name, MC, HC, ring, psi_HC_vals, RoQ_vals, I0, method, tau_boundary=None, save=True, **other_kwargs):
    kwargs = {
        "MC": MC,
        "HC": HC,
        "tau_boundary": tau_boundary,
        "method": method,
        "ring": ring,
        "name": f'{name}_RoQ_{np.floor(I0*1e3)}mA',
        "I0": I0,
        "other_kwargs": other_kwargs
    }

    output = __scan(__psi_RoQ, psi_HC_vals, RoQ_vals, **kwargs)

    if rank == 0:
        if save:
            save_out(name + "_" + method, output)
        for param in ["xi", "bunch_length", "R"]:
            __plot(output, psi_HC_vals, RoQ_vals, 1, 1, 'Tuning angle [°]', 'R/Q [ohm]', param, save, **kwargs)
    
    # Return results for Streamlit/programmatic access
    return output

        
#%% opti RoQ/I

def __opti_I0_RoQ(I0, RoQ, **kwargs):

        kwargs["HC"].Rs = RoQ*kwargs["HC"].Q
        (psi, bunch_length, xi_val, R) = __get_vals(kwargs["ring"], 
                                                    kwargs["MC"], 
                                                    kwargs["HC"], 
                                                    I0,
                                                    kwargs["psi0_HC"], 
                                                    kwargs["tau_boundary"],
                                                    kwargs["method"], 
                                                    kwargs["bounds"],
                                                    **kwargs["other_kwargs"])
        return (psi, bunch_length, xi_val, R)
      
def scan_RoQ_I0(name,
            MC,
            HC,
            ring,
            current_vals,
            RoQ_vals,
            psi0_HC,
            bounds,
            tau_boundary,
            method,
            plot_2D=None,
            save=True,
            scan_after_opti=True,
            psi_add_after_opti=-0.1,
            **other_kwargs):
    
    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_RoQ_I0',
              "psi0_HC":psi0_HC,
              "bounds":bounds,
              "other_kwargs":other_kwargs}

    out = __scan_opti(__opti_I0_RoQ,
                    current_vals,
                    RoQ_vals,
                    **kwargs)
    
    if rank == 0:
        if save:
            save_out_opti(name + "_" + method, out)
        if plot_2D is None:
            for _plot_2D in ['xi', 'bunch_length', 'psi', 'R']:
                __plot_opti(out,
                            current_vals,
                            RoQ_vals,
                            1e3,
                            1,
                            'Current [mA]',
                            'R/Q [ohm]',
                            _plot_2D,
                            save,
                            **kwargs)
        else:
            __plot_opti(out,
                        current_vals,
                        RoQ_vals,
                        1e3,
                        1,
                        'Current [mA]',
                        'R/Q [ohm]',
                        plot_2D,
                        save,
                        **kwargs)

    comm.Barrier()

    if scan_after_opti:
        out2 = __scan_after_opti(__psi_I0_RoQ,
                              out[0],
                              current_vals,
                              RoQ_vals,
                              psi_add=psi_add_after_opti,
                              **kwargs)

        if rank == 0:
            if save:
                name = name +"_" + "after_opti"
                kwargs["name"] = name
                save_out(name + "_" + method, out2)
            
            if plot_2D is None:
                for _plot_2D in ['xi', 'bunch_length', 'R']:
                        __plot(out2,
                                current_vals,
                                RoQ_vals,   
                                1e3,
                                1,
                                'Current [mA]',
                                'R/Q [ohm]',
                                _plot_2D,
                               save,
                                **kwargs)
            else:
                __plot(out2,
                        current_vals,
                        RoQ_vals,   
                        1e3,
                        1,
                        'Current [mA]',
                        'R/Q [ohm]',
                        plot_2D,
                        save,
                        **kwargs)
                
#%% opti RoQ/I0 equilibrium

def __opti_I0_RoQ_equilirium(I0, RoQ, **kwargs):
        kwargs["HC"].Rs = RoQ*kwargs["HC"].Q
        (psi, bunch_length, xi_val, R) = __get_vals_equilibrium(kwargs["ring"], 
                                                    kwargs["MC"], 
                                                    kwargs["HC"], 
                                                    I0, 
                                                    kwargs["psi0_HC"], 
                                                    kwargs["tau_boundary"],
                                                    kwargs["method"], 
                                                    kwargs["bounds"],
                                                    **kwargs["other_kwargs"])
        
        return (psi, bunch_length, xi_val, R)
    
def scan_RoQ_I0_equilirium(name,
            MC,
            HC,
            ring,
            current_vals,
            RoQ_vals,
            psi0_HC,
            bounds,
            tau_boundary,
            method,
            plot_2D=None,
            save=True,
            # scan_after_opti=True,
            # psi_add_after_opti=-0.1,
            **other_kwargs):
    # for passive HC ! Set Q0=QL
    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_RoQ_I0',
              "psi0_HC":psi0_HC,
              "bounds":bounds,
              "other_kwargs":other_kwargs}

    out = __scan_opti(__opti_I0_RoQ_equilirium,
                    current_vals,
                    RoQ_vals,
                    **kwargs)
    
    if rank == 0:
        if save:
            save_out_opti(name + "_" + method, out)
        if plot_2D is None:
            for _plot_2D in ['xi', 'bunch_length', 'psi', 'R']:
                __plot_opti(out,
                            current_vals,
                            RoQ_vals,
                            1e3,
                            1,
                            'Current [mA]',
                            'R/Q [ohm]',
                            _plot_2D,
                            save,
                            **kwargs)
        else:
            __plot_opti(out,
                        current_vals,
                        RoQ_vals,
                        1e3,
                        1,
                        'Current [mA]',
                        'R/Q [ohm]',
                        plot_2D,
                        save,
                        **kwargs)
            
#%% opti RoQ/Q0

def __opti_Q0_RoQ(Q0, RoQ, **kwargs):
        kwargs["HC"].Q = Q0
        kwargs["HC"].QL = Q0
        kwargs["HC"].Rs = RoQ*kwargs["HC"].Q
        (psi, bunch_length, xi_val, R) = __get_vals(kwargs["ring"], 
                                                    kwargs["MC"], 
                                                    kwargs["HC"], 
                                                    kwargs["I0"], 
                                                    kwargs["psi0_HC"], 
                                                    kwargs["tau_boundary"],
                                                    kwargs["method"], 
                                                    kwargs["bounds"],
                                                    **kwargs["other_kwargs"])
        
        return (psi, bunch_length, xi_val, R)
    
def scan_RoQ_Q0(name,
            MC,
            HC,
            ring,
            Q0_vals,
            RoQ_vals,
            I0,
            psi0_HC,
            bounds,
            tau_boundary,
            method,
            plot_2D=None,
            save=True,
            scan_after_opti=True,
            psi_add_after_opti=-0.1,
            **other_kwargs):
    # for passive HC ! Set Q0=QL
    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_RoQ_Q0_{int(I0*1e3)}mA',
              "psi0_HC":psi0_HC,
              "bounds":bounds,
              "I0":I0,
              "other_kwargs":other_kwargs}

    out = __scan_opti(__opti_Q0_RoQ,
                    Q0_vals,
                    RoQ_vals,
                    **kwargs)
    
    if rank == 0:
        if save:
            save_out_opti(name + "_" + method, out)
        if plot_2D is None:
            for _plot_2D in ['xi', 'bunch_length', 'psi', 'R']:
                __plot_opti(out,
                            Q0_vals,
                            RoQ_vals,
                            1,
                            1,
                            'Harmonic cavity Q0',
                            'R/Q [ohm]',
                            _plot_2D,
                            save,
                            **kwargs)
        else:
            __plot_opti(out,
                        Q0_vals,
                        RoQ_vals,
                        1,
                        1,
                        'Harmonic cavity Q0',
                        'R/Q [ohm]',
                        plot_2D,
                        save,
                        **kwargs)

    comm.Barrier()

    if scan_after_opti:
        out2 = __scan_after_opti(__psi_Q0_RoQ,
                              out[0],
                              Q0_vals,
                              RoQ_vals,
                              psi_add=psi_add_after_opti,
                              **kwargs)

        if rank == 0:
            if save:
                name = name +"_" + "after_opti"
                kwargs["name"] = name
                save_out(name + "_" + method, out2)
            
            if plot_2D is None:
                for _plot_2D in ['xi', 'bunch_length', 'R']:
                    __plot(out2,
                            Q0_vals,
                            RoQ_vals,   
                            1,
                            1,
                            'Harmonic cavity Q0',
                            'R/Q [ohm]',
                            _plot_2D,
                            save,
                            **kwargs)
            else:
                __plot(out2,
                        Q0_vals,
                        RoQ_vals,   
                        1,
                        1,
                        'Harmonic cavity Q0',
                        'R/Q [ohm]',
                        plot_2D,
                        save,
                        **kwargs)
                
#%% opti RoQ/Q0 equilibrium

def __opti_Q0_RoQ_equilirium(Q0, RoQ, **kwargs):
        kwargs["HC"].Q = Q0
        kwargs["HC"].QL = Q0
        kwargs["HC"].Rs = RoQ*kwargs["HC"].Q
        (psi, bunch_length, xi_val, R) = __get_vals_equilibrium(kwargs["ring"], 
                                                    kwargs["MC"], 
                                                    kwargs["HC"], 
                                                    kwargs["I0"], 
                                                    kwargs["psi0_HC"], 
                                                    kwargs["tau_boundary"],
                                                    kwargs["method"], 
                                                    kwargs["bounds"],
                                                    **kwargs["other_kwargs"])
        
        return (psi, bunch_length, xi_val, R)
    
def scan_RoQ_Q0_equilirium(name,
            MC,
            HC,
            ring,
            Q0_vals,
            RoQ_vals,
            I0,
            psi0_HC,
            bounds,
            tau_boundary,
            method,
            plot_2D=None,
            save=True,
            # scan_after_opti=True,
            # psi_add_after_opti=-0.1,
            **other_kwargs):
    # for passive HC ! Set Q0=QL
    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_RoQ_Q0_{int(I0*1e3)}mA',
              "psi0_HC":psi0_HC,
              "bounds":bounds,
              "I0":I0,
              "other_kwargs":other_kwargs}

    out = __scan_opti(__opti_Q0_RoQ_equilirium,
                    Q0_vals,
                    RoQ_vals,
                    **kwargs)
    
    if rank == 0:
        if save:
            save_out_opti(name + "_" + method, out)
        if plot_2D is None:
            for _plot_2D in ['xi', 'bunch_length', 'psi', 'R']:
                __plot_opti(out,
                            Q0_vals,
                            RoQ_vals,
                            1,
                            1,
                            'Harmonic cavity Q0',
                            'R/Q [ohm]',
                            _plot_2D,
                            save,
                            **kwargs)
        else:
            __plot_opti(out,
                        Q0_vals,
                        RoQ_vals,
                        1,
                        1,
                        'Harmonic cavity Q0',
                        'R/Q [ohm]',
                        plot_2D,
                        save,
                        **kwargs)

#%% generetic find intability after opti


def __psi_I0_RoQ(psi, I0, RoQ, **kwargs):
    kwargs["HC"].Rs = RoQ * kwargs["HC"].Q
    kwargs["HC"].psi = psi*np.pi/180   
    B = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], I0, tau_boundary=kwargs["tau_boundary"])
    out = B.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    out = (*out, B.xi, B.R_factor(kwargs["method"]))
    return out

def __psi_Q0_RoQ(psi, Q0, RoQ, **kwargs):
    kwargs["HC"].Q = Q0
    kwargs["HC"].QL = Q0
    kwargs["HC"].Rs = RoQ * Q0
    kwargs["HC"].psi = psi*np.pi/180   
    B = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], kwargs["I0"], tau_boundary=kwargs["tau_boundary"])
    out = B.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    out = (*out, B.xi, B.R_factor(kwargs["method"]))
    return out



#%% scan_psi_MCQL

def __psi_QL(psi, QL, **kwargs):
    kwargs["MC"].QL = QL
    kwargs["HC"].psi = psi*np.pi/180
    B = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], kwargs["I0"], tau_boundary=kwargs["tau_boundary"])
    out = B.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    out = (*out, B.xi, B.R_factor(kwargs["method"]))
    
    return out
        
def scan_psi_QL(name,
                MC,
                HC,
                ring,
                psi_HC_vals,
                QL_vals,
                I0,
                method,
                tau_boundary=None,
                save=True,
                **other_kwargs):

    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_QL_{np.floor(I0*1e3)}mA',
              "I0":I0,
              "other_kwargs":other_kwargs}
    
    out = __scan(__psi_QL,
                 psi_HC_vals,
                 QL_vals,
                 **kwargs)
    
    if rank == 0:
        if save:
            save_out(name + "_" + method, out)
        plot_list = other_kwargs.get("plot_list", ['xi', 'bunch_length', 'R'])
        for _plot_2D in plot_list:
            __plot(out,
                    psi_HC_vals,
                    QL_vals,
                    1,
                    1,
                    'Tuning angle [°]',
                    'Main cavity QL',
                    _plot_2D,
                    save,
                    **kwargs)
#%% scan_psi_MCdet

def __psi_MCdet(psi, det, **kwargs):
    kwargs["MC"].detune = det
    kwargs["HC"].psi = psi*np.pi/180
    B = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], kwargs["I0"], tau_boundary=kwargs["tau_boundary"])
    out = B.solve_passive(method=kwargs["method"], optimal_tunning=False, **kwargs["other_kwargs"])
    out = (*out, B.xi, B.R_factor(kwargs["method"]))
    
    return out
        
def scan_psi_MCdet(name,
                MC,
                HC,
                ring,
                psi_HC_vals,
                MCdet_vals,
                I0,
                method,
                tau_boundary=None,
                save=True,
                **other_kwargs):

    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_MCdet_{np.floor(I0*1e3)}mA',
              "I0":I0,
              "other_kwargs":other_kwargs}
    
    out = __scan(__psi_MCdet,
                 psi_HC_vals,
                 MCdet_vals,
                 **kwargs)
    
    if rank == 0:
        if save:
            save_out(name + "_" + method, out)
        plot_list = other_kwargs.get("plot_list", ['xi', 'bunch_length', 'R'])
        for _plot_2D in plot_list:
            __plot(out,
                    psi_HC_vals,
                    MCdet_vals,
                    1,
                    1e-3,
                    'Tuning angle [°]',
                    'Main cavity detuning [kHz]',
                    _plot_2D,
                    save,
                    **kwargs)
#%% scan_psi_MC_Rs

def __psi_MCRs(psi, Rs, **kwargs):
    kwargs["MC"].Rs_per_cavity = Rs
    kwargs["HC"].psi = psi*np.pi/180
    B = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], kwargs["I0"], tau_boundary=kwargs["tau_boundary"])
    out = B.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    out = (*out, B.xi, B.R_factor(kwargs["method"]))
    
    return out
        
def scan_psi_MC_Rs(name,
                    MC,
                    HC,
                    ring,
                    psi_HC_vals,
                    MC_Rs_vals,
                    I0,
                    method,
                    tau_boundary=None,
                    save=True,
                    **other_kwargs):

    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_MC_Rs_{np.floor(I0*1e3)}mA',
              "I0":I0,
              "other_kwargs":other_kwargs}
    
    out = __scan(__psi_MCRs,
                 psi_HC_vals,
                 MC_Rs_vals,
                 **kwargs)
    
    if rank == 0:
        if save:
            save_out(name + "_" + method, out)
        plot_list = other_kwargs.get("plot_list", ['xi', 'bunch_length', 'R'])
        for _plot_2D in plot_list:
            __plot(out,
                    psi_HC_vals,
                    MC_Rs_vals,
                    1,
                    1e-6,
                    'Tuning angle [°]',
                    'Main cavity Rs per cavity [MOhm]',
                    _plot_2D,
                    save,
                    **kwargs)
#%% scan_psi_MC_beta

def __psi_MC_beta(psi, beta, **kwargs):
    kwargs["MC"].beta = beta
    kwargs["HC"].psi = psi*np.pi/180
    B = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], kwargs["I0"], tau_boundary=kwargs["tau_boundary"])
    out = B.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    out = (*out, B.xi, B.R_factor(kwargs["method"]))
    
    return out
        
def scan_psi_MC_beta(name,
                    MC,
                    HC,
                    ring,
                    psi_HC_vals,
                    MC_beta_vals,
                    I0,
                    method,
                    tau_boundary=None,
                    save=True,
                    **other_kwargs):

    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_MC_beta_{np.floor(I0*1e3)}mA',
              "I0":I0,
              "other_kwargs":other_kwargs}
    
    out = __scan(__psi_MC_beta,
                 psi_HC_vals,
                 MC_beta_vals,
                 **kwargs)
    
    if rank == 0:
        if save:
            save_out(name + "_" + method, out)
        plot_list = other_kwargs.get("plot_list", ['xi', 'bunch_length', 'R'])
        for _plot_2D in plot_list:
            __plot(out,
                    psi_HC_vals,
                    MC_beta_vals,
                    1,
                    1,
                    'Tuning angle [°]',
                    'Main cavity beta',
                    _plot_2D,
                    save,
                    **kwargs)
        
#%% scan_psi_HC_Q0

def __psi_HC_Q0(psi, Q0, **kwargs):
    kwargs["HC"].Q = Q0
    kwargs["HC"].QL = Q0
    kwargs["HC"].Rs = Q0 * kwargs["RoQ"]
    kwargs["HC"].psi = psi*np.pi/180
    B = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], kwargs["I0"], tau_boundary=kwargs["tau_boundary"])
    out = B.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    out = (*out, B.xi, B.R_factor(kwargs["method"]))
    
    return out
        
def scan_psi_HC_Q0(name,
                    MC,
                    HC,
                    ring,
                    psi_HC_vals,
                    HC_Q0_vals,
                    RoQ,
                    I0,
                    method,
                    tau_boundary=None,
                    save=True,
                    **other_kwargs):
    # for passive HC ! Set Q0=QL
    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_HC_Q0_{np.floor(I0*1e3)}mA',
              "I0":I0,
              "RoQ":RoQ,
              "other_kwargs":other_kwargs}
    
    out = __scan(__psi_HC_Q0,
                 psi_HC_vals,
                 HC_Q0_vals,
                 **kwargs)
    
    if rank == 0:
        if save:
            save_out(name + "_" + method, out)
        plot_list = other_kwargs.get("plot_list", ['xi', 'bunch_length', 'R'])
        for _plot_2D in plot_list:
            __plot(out,
                    psi_HC_vals,
                    HC_Q0_vals,
                    1,
                    1,
                    'Tuning angle [°]',
                    'Harmonic cavity Q0',
                    _plot_2D,
                    save,
                    **kwargs)
        
#%% scan_MC_Vc_HC_Vc_active

def __MC_Vc_HC_Vc_active(MC_Vc, HC_Vc, **kwargs):
    kwargs["MC"].Vc = MC_Vc
    kwargs["MC"].theta = np.arccos(kwargs["ring"].U0/kwargs["MC"].Vc)
    kwargs["MC"].set_optimal_detune(kwargs["I0"])
    # kwargs["MC"].psi = kwargs["MC"].psi - 30/180*np.pi
    kwargs["MC"].set_generator(kwargs["I0"])
    
    kwargs["HC"].Vc = HC_Vc
    kwargs["HC"].theta = -np.pi/2
    kwargs["HC"].set_optimal_detune(kwargs["I0"])
    # kwargs["HC"].psi = kwargs["HC"].psi - 30/180*np.pi
    kwargs["HC"].set_generator(kwargs["I0"])
    B = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], kwargs["I0"], tau_boundary=kwargs["tau_boundary"])
    out = B.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    out = (*out, B.xi, B.R_factor(kwargs["method"]))
    
    return out
        
def scan_MC_Vc_HC_Vc_active(name,
                    MC,
                    HC,
                    ring,
                    MC_Vc_vals,
                    HC_Vc_vals,
                    I0,
                    method,
                    tau_boundary=None,
                    save=True,
                    **other_kwargs):

    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_MC_Vc_HC_Vc_{np.floor(I0*1e3)}mA',
              "I0":I0,
              "other_kwargs":other_kwargs}
    
    out = __scan(__MC_Vc_HC_Vc_active,
                 MC_Vc_vals,
                 HC_Vc_vals,
                 **kwargs)
    
    if rank == 0:
        if save:
            save_out(name + "_" + method, out)
        plot_list = other_kwargs.get("plot_list", ['xi', 'bunch_length', 'R'])
        for _plot_2D in plot_list:
            __plot(out,
                    MC_Vc_vals,
                    HC_Vc_vals,
                    1e-6,
                    1e-3,
                    'MC voltage [MV]',
                    'HC voltage [kV]',
                    _plot_2D,
                    save,
                    **kwargs)
#%% scan_MC_Vc_MC_psi_active

def __MC_Vc_MC_psi_active(MC_Vc, MC_psi, **kwargs):
    kwargs["MC"].Vc = MC_Vc
    kwargs["MC"].theta = np.arccos(kwargs["ring"].U0/kwargs["MC"].Vc)
    kwargs["MC"].set_optimal_detune(kwargs["I0"])
    kwargs["MC"].psi = kwargs["MC"].psi + MC_psi/180*np.pi
    kwargs["MC"].set_generator(kwargs["I0"])
    B = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], kwargs["I0"], tau_boundary=kwargs["tau_boundary"])
    out = B.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    out = (*out, B.xi, B.R_factor(kwargs["method"]))
    
    return out
        
def scan_MC_Vc_MC_psi_active(name,
                    MC,
                    HC,
                    ring,
                    MC_Vc_vals,
                    MC_psi_vals,
                    I0,
                    method,
                    tau_boundary=None,
                    save=True,
                    **other_kwargs):

    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_MC_Vc_MC_psi_{np.floor(I0*1e3)}mA',
              "I0":I0,
              "other_kwargs":other_kwargs}
    
    out = __scan(__MC_Vc_MC_psi_active,
                 MC_Vc_vals,
                 MC_psi_vals,
                 **kwargs)
    
    if rank == 0:
        if save:
            save_out(name + "_" + method, out)
        plot_list = other_kwargs.get("plot_list", ['xi', 'bunch_length', 'R'])
        for _plot_2D in plot_list:
            __plot(out,
                    MC_Vc_vals,
                    MC_psi_vals,
                    1e-6,
                    1,
                    'MC voltage [MV]',
                    'MC delta psi from optimal [°]',
                    _plot_2D,
                    save,
                    **kwargs)
#%% scan_MC_psi_HC_psi_active

def __MC_psi_HC_psi_active(MC_psi, HC_psi, **kwargs):
    kwargs["MC"].set_optimal_detune(kwargs["I0"])
    kwargs["MC"].psi = kwargs["MC"].psi + MC_psi/180*np.pi
    kwargs["MC"].set_generator(kwargs["I0"])

    kwargs["HC"].set_optimal_detune(kwargs["I0"])
    kwargs["HC"].psi = kwargs["HC"].psi + HC_psi/180*np.pi
    kwargs["HC"].set_generator(kwargs["I0"])
    B = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], kwargs["I0"], tau_boundary=kwargs["tau_boundary"])
    out = B.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    out = (*out, B.xi, B.R_factor(kwargs["method"]))
    
    return out
        
def scan_MC_psi_HC_psi_active(name,
                    MC,
                    HC,
                    ring,
                    MC_psi_vals,
                    HC_psi_vals,
                    I0,
                    method,
                    tau_boundary=None,
                    save=True,
                    **other_kwargs):

    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_MC_psi_HC_psi_{np.floor(I0*1e3)}mA',
              "I0":I0,
              "other_kwargs":other_kwargs}
    
    out = __scan(__MC_psi_HC_psi_active,
                 MC_psi_vals,
                 HC_psi_vals,
                 **kwargs)
    
    if rank == 0:
        if save:
            save_out(name + "_" + method, out)
        plot_list = other_kwargs.get("plot_list", ['xi', 'bunch_length', 'R'])
        for _plot_2D in plot_list:
            __plot(out,
                    MC_psi_vals,
                    HC_psi_vals,
                    1,
                    1,
                    'MC delta psi from optimal [°]',
                    'HC delta psi from optimal [°]',
                    _plot_2D,
                    save,
                    **kwargs)
        
#%% scan_HC_beta_I0_active

def __HC_beta_I0_active(HC_beta, I0, **kwargs):
    kwargs["MC"].set_optimal_detune(I0)
    kwargs["MC"].set_generator(I0)
    
    kwargs["HC"].beta = HC_beta
    kwargs["HC"].set_optimal_detune(I0)
    kwargs["HC"].set_generator(I0)
    B = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], I0, tau_boundary=kwargs["tau_boundary"])
    out = B.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    out = (*out, B.xi, B.R_factor(kwargs["method"]))
    
    return out
        
def scan_HC_beta_I0_active(name,
                    MC,
                    HC,
                    ring,
                    HC_beta_vals,
                    I0_vals,
                    method,
                    tau_boundary=None,
                    save=True,
                    **other_kwargs):

    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_HC_beta_I0',
              "other_kwargs":other_kwargs}
    out = __scan(__HC_beta_I0_active,
                 HC_beta_vals,
                 I0_vals,
                 **kwargs)
    if rank == 0:
        if save:
            save_out(name + "_" + method, out)
        plot_list = other_kwargs.get("plot_list", ['xi', 'bunch_length', 'R'])
        for _plot_2D in plot_list:
            __plot(out,
                    HC_beta_vals,
                    I0_vals,
                    1,
                    1e3,
                    'HC beta',
                    'Current [mA]',
                    _plot_2D,
                    save,
                    **kwargs)
        
#%% scan_MC_psi_I0

def __MC_psi_I0(psi, I0, **kwargs):
    kwargs["MC"].psi = psi*np.pi/180
    kwargs["MC"].set_generator(I0)
    B = RobinsonModes(kwargs["ring"], [kwargs["MC"], kwargs["HC"]], I0, tau_boundary=kwargs["tau_boundary"])
    out = B.solve(method=kwargs["method"], **kwargs["other_kwargs"])
    out = (*out, B.xi, B.R_factor(kwargs["method"]))
    
    return out
        
def scan_MC_psi_I0(name,
                    MC,
                    HC,
                    ring,
                    psi_MC_vals,
                    I0_vals,
                    method,
                    tau_boundary=None,
                    save=True,
                    **other_kwargs):

    kwargs = {"MC":MC,
              "HC":HC,
              "tau_boundary":tau_boundary,
              "method":method,
              "ring":ring,
              "name":f'{name}_MC_psi_I0',
              "other_kwargs":other_kwargs}
    
    out = __scan(__MC_psi_I0,
                 psi_MC_vals,
                 I0_vals,
                 **kwargs)
    
    if rank == 0:
        if save:
            save_out(name + "_" + method, out)
        plot_list = other_kwargs.get("plot_list", ['xi', 'bunch_length', 'R'])
        for _plot_2D in plot_list:
            __plot(out,
                    psi_MC_vals,
                    I0_vals,
                    1,
                    1e3,
                    'MC Tuning angle [°]',
                    'Current [mA]',
                    _plot_2D,
                    save,
                    **kwargs)
        
