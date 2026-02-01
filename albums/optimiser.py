"""
Module where the optimization functions are defined.
"""
from scipy.optimize import Bounds, minimize
import numpy as np
from albums.robinson import RobinsonModes
# Global counter for evaluations
eval_num = 0

def evaluate_R(B, method, is_equilibrium=False):
    """
    Evaluate the R-factor or equilibrium solution.

    Parameters:
    - B: RobinsonModes instance
    - method: The method used for solving
    - is_equilibrium: Boolean indicating if equilibrium-only evaluation is performed

    Returns:
    - Evaluation score (negative R-factor if successful, penalized score otherwise)
    """
    try:
        if is_equilibrium:
            _, R, _, converged = B.solve_equilibrium_only(method=method)
            if not converged:
                return 10
            return -R if not np.isnan(R) else 10
        else:
            out = B.solve(method=method)
            _, _, _, _, _, PTBL, converged = out
            if not converged.any() or PTBL:
                return 10
            return -B.R_factor(method)
    except:
        return 10

def optimize_R(ring, MC, HC, I0, psi0_HC, tau_boundary, bounds, is_equilibrium=False, **kwargs):
    """
    General optimization for the R-factor or equilibrium.

    Parameters:
    - ring: Ring configuration
    - MC: Main cavity parameters
    - HC: Harmonic cavity parameters
    - I0: Beam current
    - psi0_HC: Initial guess for psi_HC
    - tau_boundary: Boundary for tau
    - bounds: Bounds for psi_HC
    - is_equilibrium: Boolean flag for equilibrium-only optimization
    - kwargs: Additional configuration for the optimizer

    Returns:
    - Optimal value of psi_HC or a fallback value (90 on failure)
    """
    method_opti = kwargs.get("method_opti", "COBYLA")
    tol_opti = kwargs.get("tol_opti", 0.01)
    maxiter_opti = kwargs.get("maxiter_opti", 1000)
    rhobeg_opti = kwargs.get("rhobeg_opti", 0.1)
    method = kwargs.get("method", "default_method")
    
    def to_eval(psi_HC):
        global eval_num
        eval_num += 1
        HC.psi = psi_HC * np.pi / 180
        B = RobinsonModes(ring, [MC, HC], I0, tau_boundary=tau_boundary)
        return evaluate_R(B, method, is_equilibrium)

    bd = Bounds([bounds[0]], [bounds[1]])
    res = minimize(fun=to_eval, x0=[psi0_HC], bounds=bd, method=method_opti, 
                   tol=tol_opti, options={"maxiter": maxiter_opti, "rhobeg": rhobeg_opti})
    return res.x[0] if res.success else 90

def minimize_psi(ring, MC, HC, I0, psi0_HC, tau_boundary, bounds, **kwargs):
    """
    Minimize the psi parameter for stability.

    Parameters:
    - ring: Ring configuration
    - MC: Main cavity parameters
    - HC: Harmonic cavity parameters
    - I0: Beam current
    - psi0_HC: Initial guess for psi_HC
    - tau_boundary: Boundary for tau
    - bounds: Bounds for psi_HC
    - kwargs: Additional configuration for the optimizer

    Returns:
    - Optimal value of psi_HC or np.nan on failure
    """
    method = kwargs.get("method", "Nelder-Mead")
    tol = kwargs.get("tol", 0.05)

    def to_eval(psi_HC):
        HC.psi = psi_HC * np.pi / 180
        B = RobinsonModes(ring, [MC, HC], I0, tau_boundary=tau_boundary)
        try:
            out = B.solve(method=method, **kwargs)
            _, zero_frequency, robinson, _, _, PTBL, converged = out
            if (not converged.any() or robinson.any() or PTBL or zero_frequency):
                return psi_HC + 100
        except:
            return psi_HC + 100
        return psi_HC

    bd = Bounds([bounds[0]], [bounds[1]])
    res = minimize(fun=to_eval, x0=[psi0_HC], bounds=bd, method=method, tol=tol)
    return res.x[0] if res.success else np.nan

def __get_vals(ring, MC, HC, I0, psi0_HC, 
               tau_boundary, method, bounds, **kwargs):
    
    debug = kwargs.get("debug", False)
    loop_option = kwargs.get("loop_option", False)
    add_psi_loop = kwargs.get("add_psi_loop", 0.02)
    auto_psi_input = kwargs.get("auto_psi_input", False)
    xi_init_input = kwargs.get("xi_init_input", 0.8)
    
    if auto_psi_input:
        xi = xi_init_input
        HC_det = I0*HC.Rs/HC.Q*ring.f1/MC.Vc*HC.m**2/xi
        HC_fr = HC_det + HC.m * ring.f1
        HC_psi = np.arctan(HC.QL * (HC_fr / (HC.m * ring.f1) -
                                         (HC.m * ring.f1) / HC_fr)) * 180 / np.pi
        psi0_HC = HC_psi
        if debug:
            print(f"psi0 input  = {psi0_HC}")
    
    psi = maximize_R(ring, MC, HC, I0, psi0_HC, 
                   tau_boundary, method, bounds, **kwargs)
    
    if debug:
        print(f"psi out of maximize_R = {psi}")
    
    if loop_option:
    
        count = 0
        add_psi = add_psi_loop
        while(True):
            psi = (psi + count*add_psi)
            HC.psi = psi*np.pi/180
            B = RobinsonModes(ring, [MC, HC], I0, tau_boundary=tau_boundary)
            out = B.solve(method=method, **kwargs)
            (bunch_length, zero_frequency, robinson, _, _, PTBL, converged) = out
            
            try:
                cond = not converged.any()
                is_stable = not (cond or robinson.any() or PTBL or zero_frequency)
            except AttributeError:
                cond = not converged
                is_stable = False
            
            if is_stable:
                if debug:
                    print(f"psi sent to result = {psi}")
                    print(out)
                return (psi, bunch_length, B.xi, B.R_factor(method))
            else:
                if debug:
                    print(f"unstable: {count}")
                    print(f"psi unstable = {psi}")
                    print(out)
            
            if count == 200:
                raise ValueError()
            count += 1
            
    else:
        HC.psi = psi*np.pi/180
        B = RobinsonModes(ring, [MC, HC], I0, tau_boundary=tau_boundary)
        out = B.solve(method=method, **kwargs)
        (bunch_length, zero_frequency, robinson, _, _, PTBL, converged) = out
        
        if debug:
            print(out)
        
        return (psi, bunch_length, B.xi, B.R_factor(method))
    
def __get_vals_equilibrium(ring, MC, HC, I0, psi0_HC, 
               tau_boundary, method, bounds, **kwargs):
    
    debug = kwargs.get("debug", False)
    loop_option = kwargs.get("loop_option", False)
    add_psi_loop = kwargs.get("add_psi_loop", 0.02)
    auto_psi_input = kwargs.get("auto_psi_input", False)
    xi_init_input = kwargs.get("xi_init_input", 0.8)
    
    if auto_psi_input:
        xi = xi_init_input
        HC_det = I0*HC.Rs/HC.Q*ring.f1/MC.Vc*HC.m**2/xi
        HC_fr = HC_det + HC.m * ring.f1
        HC_psi = np.arctan(HC.QL * (HC_fr / (HC.m * ring.f1) -
                                         (HC.m * ring.f1) / HC_fr)) * 180 / np.pi
        psi0_HC = HC_psi
        if debug:
            print(f"psi0 input  = {psi0_HC}")
    
    psi = maximize_R_equilibrium(ring, MC, HC, I0, psi0_HC, 
                   tau_boundary, method, bounds, **kwargs)
    
    if debug:
        print(f"psi out of maximize_R = {psi}")
    
    if loop_option:
    
        count = 0
        add_psi = add_psi_loop
        while(True):
            psi = (psi + count*add_psi)
            HC.psi = psi*np.pi/180
            B = RobinsonModes(ring, [MC, HC], I0, tau_boundary=tau_boundary)
            out = B.solve_equilibrium_only(method=method, **kwargs)
            (bunch_length, _, xi, converged) = out  
            
            is_stable = converged
            
            if is_stable:
                if debug:
                    print(f"psi sent to result = {psi}")
                    print(out)
                return (psi, bunch_length, B.xi, B.R_factor(method))
            else:
                if debug:
                    print(f"unstable: {count}")
                    print(f"psi unstable = {psi}")
                    print(out)
            
            if count == 200:
                raise ValueError()
            count += 1
            
    else:
        HC.psi = psi*np.pi/180
        B = RobinsonModes(ring, [MC, HC], I0, tau_boundary=tau_boundary)
        out = B.solve_equilibrium_only(method=method, **kwargs)
        (bunch_length, _, xi, converged) = out
        
        if debug:
            print(out)
        
        return (psi, bunch_length, B.xi, B.R_factor(method))

def maximize_R(ring, MC, HC, I0, psi0_HC, 
               tau_boundary, method, bounds, **kwargs):
    
    method_opti = kwargs.get("method_opti", "COBYLA")
    tol_opti = kwargs.get("tol_opti", 0.01)
    maxiter_opti = kwargs.get("maxiter_opti", 1000)
    rhobeg_opti = kwargs.get("rhobeg_opti", 0.1)
    
    # Define objective function properly (no vectorize)
    def to_eval(x):   
        global eval_num
        eval_num += 1 
        
        # Handle input: x might be an array-like from minimize
        try:
            psi_HC = float(x[0]) if hasattr(x, '__getitem__') else float(x)
        except:
            psi_HC = float(psi0_HC)
            
        res = 10.0
        try:
            HC.psi = psi_HC * np.pi / 180.0
            B = RobinsonModes(ring, [MC, HC], I0, tau_boundary=tau_boundary)
            
            out = B.solve(method=method, **kwargs)
            (bunch_length, zero_frequency, robinson, HOM, Omega, PTBL, converged) = out  
            
            # Check convergence and stability
            is_stable = False
            try:
                cond = not converged.any() if hasattr(converged, 'any') else not converged
                instability = (robinson.any() if hasattr(robinson, 'any') else robinson) or PTBL or zero_frequency
                is_stable = not (cond or instability)
            except:
                is_stable = False
            
            if is_stable:
                try:
                    r_val = B.R_factor(method)
                    res = -float(r_val)
                    if np.isnan(res):
                        res = 10.0
                except:
                    res = 10.0
            else:
                res = 10.0
                
        except Exception as e:
            # print(f"Error in evaluation: {e}")
            res = 10.0
            
        return res
    
    try:
        # Use Bounds class correctly
        bd = Bounds([bounds[0]], [bounds[1]])

        res = minimize(fun=to_eval, x0=[psi0_HC], bounds=bd, 
                        method=method_opti, 
                        tol=tol_opti,
                        options={"maxiter": maxiter_opti, "rhobeg": rhobeg_opti})
        
        if res.success:
            return res.x[0]
        else:
            return 90.0
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Return initial guess or default if optimization crashes
        return psi0_HC if 'res' not in locals() else 90.0

def maximize_R_equilibrium(ring, MC, HC, I0, psi0_HC, 
               tau_boundary, method, bounds, **kwargs):
    
    method_opti = kwargs.get("method_opti", "COBYLA")
    tol_opti = kwargs.get("tol_opti", 0.01)
    maxiter_opti = kwargs.get("maxiter_opti", 1000)
    rhobeg_opti = kwargs.get("rhobeg_opti", 0.1)
           
    def to_eval(x):   
        global eval_num
        eval_num += 1 
        
        try:
            psi_HC = float(x[0]) if hasattr(x, '__getitem__') else float(x)
        except:
            psi_HC = float(psi0_HC)
            
        res = 10.0
        try:
            HC.psi = psi_HC * np.pi / 180.0
            B = RobinsonModes(ring, [MC, HC], I0, tau_boundary=tau_boundary)
            
            out = B.solve_equilibrium_only(method=method, **kwargs)
            (bunch_length, R, xi, converged) = out  
            
            if converged:
                res = -float(R)
                if np.isnan(res):
                    res = 10.0
            else:
                res = 10.0
                
        except Exception:
            res = 10.0
            
        return res
    
    try:
        bd = Bounds([bounds[0]], [bounds[1]])

        res = minimize(fun=to_eval, x0=[psi0_HC], bounds=bd, 
                        method=method_opti, 
                        tol=tol_opti,
                        options={"maxiter": maxiter_opti, "rhobeg": rhobeg_opti})
        
        if res.success:
            return res.x[0]
        else:
            return 90.0
    except Exception as e:
        print(f"Optimization failed: {e}")
        return psi0_HC if 'res' not in locals() else 90.0
