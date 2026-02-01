# -*- coding: utf-8 -*-
"""
Module to compute longitudinal instabilities in a double RF system.
"""

import numpy as np
from math import factorial
from copy import deepcopy
from scipy.constants import c
from scipy.integrate import quad, trapezoid
from scipy.optimize import root, root_scalar
from mbtrack2 import BeamLoadingEquilibrium, gaussian_bunch
from pycolleff.longitudinal_equilibrium import LongitudinalEquilibrium, ImpedanceSource

from albums.mbtrack2_to_pycolleff import synchrotron_to_pycolleff, cavityresonator_to_pycolleff, impedance_to_pycolleff

class RobinsonModes():
    """
    Class to compute instabilities in a double RF system mainly based on the
    theory and algorithm developped in [1].

    The original algorithm has been adapted to work with other methods to solve
    the self-consistent problem (i.e. Haissinski equation) [3,4,5,6] and to give
    an estimate of the threshold of l=1/PTBL instability [7].

    See the solve method for details.

    Parameters
    ----------
    ring : Synchrotron object
        Ring parameters.
    cavity_list : list of CavityResonator objects
        A list which contains:
            - a CavityResonator for the main cavity in 1st position.
            - a CavityResonator for the harmonic cavity in 2nd position.
    I0 : float
        Beam current in [A].
    tau_boundary : float, optinal
        Integration boundary in [s].
        If None, 0.1 * RF period is used.
        The default is None.

    Attributes
    ----------
    xi : float

    Methods
    -------
    solve()
        Solve for a given settings of the double RF system.
    R_factor(method)
        Touschek lifetime ratio.

    References
    ----------
    [1] : Bosch, R. A., K. J. Kleman, and J. J. Bisognano. "Robinson
    instabilities with a higher-harmonic cavity." Physical Review Special
    Topics-Accelerators and Beams 4.7 (2001): 074401.

    [2] : Bosch, R. A., and C. S. Hsue. "Suppression of longitudinal
    coupled-bunch instabilities by a passive higher harmonic cavity."
    Proceedings of International Conference on Particle Accelerators. IEEE, 1993.

    [3] : Gamelin, A., Yamamoto, N. (2021). Equilibrium bunch density
    distribution with multiple active and passive RF cavities.
    IPAC'21 (MOPAB069).

    [4] : Venturini, M. (2018). Passive higher-harmonic rf cavities with general
    settings and multibunch instabilities in electron storage rings.
    Physical Review Accelerators and Beams, 21(11), 114404.

    [5] : Alves, Murilo B., and Fernando H. de Sá. "Equilibrium of longitudinal
    bunch distributions in electron storage rings with arbitrary impedance
    sources and generic filling patterns." Physical Review Accelerators and
    Beams 26.9 (2023): 094402.

    [6] : de Sá, F., & Alves, M. (2023). pycolleff and cppcolleff: modules for
    impedance analysis and wake-field induced instabilities evaluation.
    (Version 0.1.0) [Computer software]. https://doi.org/10.5281/zenodo.7974571

    [7] : He, Tianlong. "Novel perturbation method for judging the stability of
    the equilibrium solution in the presence of passive harmonic cavities."
    Physical Review Accelerators and Beams 25.9 (2022): 094402.
    """
    def __init__(self,
                 ring,
                 cavity_list,
                 I0,
                 tau_boundary=None):
        self.ring = ring
        self.cavity_list = []
        for item in cavity_list:
            self.cavity_list.append(deepcopy(item))
        
        # Add ring reference to each cavity so they can access ring properties
        # (needed for cavityresonator_to_pycolleff conversion)
        for cavity in self.cavity_list:
            cavity.ring = ring
        
        self.I0 = I0
        self.n_cavity = len(cavity_list)
        if tau_boundary is None:
            self.tau_boundary = self.ring.T1*0.1
        else:
            self.tau_boundary = tau_boundary

    def solve(self,
                passive_harmonic_cavity=True,
                mode_coupling=True,
                auto_set_MC_theta=True,
                optimal_tunning=True,
                f_HOM=0,
                Z_HOM=0,
                method="Venturini",
                **kwargs):
        """
        Solve for a given settings of the double RF system.

        Parameters
        ----------
        passive_harmonic_cavity : bool, optional
            If True, solve for a passive harmonic cavity.
            If False, solve for an active harmonic cavity.
            The default is True.
        mode_coupling : bool, optional
            If True, compute Robinson instabilities taking into account
            dipole-quadrupole mode coupling (only dipole and quadrupole modes).
            If False, compute Robinson instabilities without taking into
            account mode coupling (modes from dipole to octupole).
            The default is True.
        auto_set_MC_theta : bool, optional
            For passive harmonic cavity only:
                -if True, set the main cavity phase to cancel the losses in the
                harmonic cavity.
                -if False, keep input main cavity phase setting.
            The default is True.
        optimal_tunning : bool, optional
            For passive harmonic cavity:
                -if True, set main cavity tuning to optimal.
                -if False, keep input main cavity tuning setting.
            For active harmonic cavity:
                -if True, set main and harmonic cavity tuning to optimal.
                -if False, keep input main and harmonic cavity tuning setting.
                The default is True.
        f_HOM : float or list of float, optional
            Frequencies of the HOMs to evaluate in [Hz].
            The default is 0.
        Z_HOM : float or list of float, optional
            Impedances of the HOMs at f_HOM in [ohm].
            The default is 0.
        method : {"Bosch","Venturini", "Alves"}, optional
            Choose the method used to solve the self-consistent problem (i.e.
            Haissinski equation):
                - "Bosch" corresponds to an implementation of the original
                algorithm in [1]. It computes only the amplitude form factor
                and bunch length. The result can depend heavily on tau_boundary.
                - "Venturini" corresponds to the method described in [3-4].
                It can deal with passive and active HC.
                - "Alves" corresponds to the method described in [5].
                It can deal with arbitrary filling pattern and broad band impedance.
                The pycolleff package is needed for this method [6].
                Only for passive HC.
            For both "Venturini" and "Alves", the bunch profile is computed
            which gives both amplitude and phase form factors.
            The default is "Venturini".

        Kwargs for method "Bosch"
        -------------------------
        Passive harmonic cavity:
            max_counter : int
                Maximum number of iteration for the computation of form factors.
                Default is 200.
        Active harmonic cavity:
            auto_set_for_xi : bool
                If True, set harmonic cavity setting to achive xi value.
                specified as in [1].
                If False, keep input main and harmonic cavity setting.
                Default is False.
            xi : float
                Value to achive if auto_set_for_xi is True.
                Default is None.

        Kwargs for method "Venturini"
        -----------------------------
        F_init : array-like of float of shape (2,)
            Initial form factors (MC, HC) to consider for synchronous phase
            calculation if auto_set_MC_theta is True.
            Default is [1,1].
        set_MC_phase_HCpassive : bool
            If True, set the main cavity phase to cancel the losses in the 
            harmonic cavity using approxmiate relation.
            Default is False.

        Kwargs for method "Alves"
        -------------------------
        filling : array-like of float of shape (ring.h,)
            Filling pattern of the beam.
            Sum must be equal to one.
            Default is None, corresponding to uniform filling.
        impedance : Impedance object
            Additional (broad band) longitudinal impedance to consider for the
            Haissiski equation.
            Default is None.
        F_init : array-like of float of shape (2,)
            Initial form factors (MC, HC) to consider for synchronous phase
            calculation if auto_set_MC_theta is True.
            Default is [1,1].
        set_MC_phase_HCpassive : bool
            If True, set the main cavity phase to cancel the losses in the 
            harmonic cavity using approximate relation.
            Default is False.
        use_GaussLMCI : bool
            If True, the Gaussian LMCI method is used to compute Robinson
            instabilities instead of Bosch's equations.
            Default is False.
        use_PTBL_He : bool
            If True, the He criteria is used to compute PTBL instability 
            instead of Gaussian LMCI method.
            Default is False.
        **kwargs passed to LongitudinalEquilibrium.calc_longitudinal_equilibrium:
            - niter
            - tol
            - beta
            - m
            - print_flag
        **kwargs passed to LongitudinalEquilibrium.calc_mode_coupling:
            - max_azi
            - max_rad

        Returns
        -------
        bunch_length : float or None
            RMS bunch length in [s].
        zero_frequency : bool or None
            True if zero frequency instability is predicted.
        robinson : array of bool of shape (4,) or None
            Bool of the arrays are True if the following instability is predicted:
            If mode_coupling:
                - Coupled dipole mode
                - Coupled quadrupole mode
                - Fast mode coupling by dipole mode
                - Fast mode coupling by quadrupole mode
            else:
                - Dipole mode
                - Quadrupole mode
                - Sextupole mode
                - Octupole mode
        HOM : bool
            True if HOM instability is predicted.
        Omega : array of float or None
            If mode_coupling:
                - Coupled dipole angular frequency in unit of 2*pi*[Hz]
                - Coupled quadrupole angular frequency in unit of 2*pi*[Hz]
            else:
                - Dipole angular frequency in unit of 2*pi*[Hz]
                - Quadrupole angular frequency in unit of 2*pi*[Hz]
                - Sextupole angular frequency in unit of 2*pi*[Hz]
                - Octupole angular frequency in unit of 2*pi*[Hz]
        PTBL : bool
            True if PTBL instability is predicted.
            None is returned if method is "Bosch".
        converged : bool or array of bool
            Return True values if calculation has converged. In details:
            If form factor calculation or Haissinski equation resolution fails:
                return False and other results are None
            If zero_frequency is True:
                return True and other instability results than zero_frequency
                are None
            If mode_coupling is True:
                return array of bool of shape (2,) for coupled dipole and
                quadrupole modes.
            If mode_coupling is False:
                return array of bool of shape (4,) for dipole to octupole modes.

        """
        (bunch_length, a, b, c, converged) = self._converge_form_factors(passive_harmonic_cavity, auto_set_MC_theta, optimal_tunning, method, **kwargs)

        if converged:
            (zero_frequency, robinson, HOM, Omega, converged) = self._solve_core(bunch_length, a, b, c, mode_coupling, f_HOM, Z_HOM, method, **kwargs)
        else:
            return (None, None, None, None, None, None, converged)

        if method == "Bosch":
            omega_r = self._robinson_frequency()
            landau = self._landau_threshold(omega_r, c, b)
            coupled_mode1, _, _ = self._coupled_bunch_mode1(omega_r, bunch_length, landau)
            return (bunch_length, zero_frequency, robinson, HOM, Omega, np.any(coupled_mode1), converged)
        elif method == "Venturini":
            (eta, _, _) = self.BLE.PTBL_threshold(self.I0)
            if eta > 1:
                PTBL = True
            else:
                PTBL = False
            return (bunch_length, zero_frequency, robinson, HOM, Omega, PTBL, converged)
        elif method == "Alves":
            use_PTBL_He = kwargs.get("use_PTBL_He", False)
            if use_PTBL_He:
                PTBL = self._PTBL_He()
            else:
                PTBL = self._PTBL_alves(**kwargs)
            return (bunch_length, zero_frequency, robinson, HOM, Omega, PTBL, converged)
        
    def solve_equilibrium_only(self,
                               passive_harmonic_cavity=True,
                               auto_set_MC_theta=True,
                               optimal_tunning=True,
                               method="Venturini",
                               **kwargs):
        
        (bunch_length, _, _, _, converged) = self._converge_form_factors(passive_harmonic_cavity, auto_set_MC_theta, optimal_tunning, method, **kwargs)
        
        return (bunch_length, self.R_factor(method), self.xi, converged)

    def R_factor(self, method):
        """
        Touschek lifetime ratio as defined in [1].

        Parameters
        ----------
        method : {"Bosch","Venturini", "Alves"}
            Method used in self.solve.

        Returns
        -------
        R : float
            Touschek lifetime ratio for methods "Veturini" or "Alves".
            1 if method is "Bosch" or solve has not converged.

        Reference
        ---------
        [1] : Byrd, J. M., and M. Georgsson. "Lifetime increase using passive
        harmonic cavities in synchrotron light sources." Physical Review
        Special Topics-Accelerators and Beams 4.3 (2001): 030701.

        """
        if method == "Bosch":
            R = 1
        elif method == "Venturini":
            try:
                R = self.BLE.R_factor
            except AttributeError:
                R = 1
        elif method == "Alves":
            try:
                rho0 = gaussian_bunch(self.LE.zgrid/c, self.ring.sigma_0)/c
                R = trapezoid(rho0**2, self.LE.zgrid)/trapezoid(self.LE.distributions[0]**2, self.LE.zgrid)
            except AttributeError:
                R = 1
        else:
            raise AttributeError
        return R

    @property
    def xi(self):
        """
        Ratio of the harmonic cavity 'force' over the main cavity one.
        Near Eq. (7) in [1].
        """
        MC = self.cavity_list[0]
        HC = self.cavity_list[1]
        return -1 * HC.m * HC.Vc * np.sin(HC.theta) / (MC.Vc * np.sin(MC.theta))

    def _potential_decomposition(self):
        # Eq. (4-6) in [1]

        a_sum = 0
        b_sum = 0
        c_sum = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            a_sum += cavity.m*cavity.Vc*np.sin(cavity.theta)
            b_sum += cavity.m**2*cavity.Vc*np.cos(cavity.theta)
            c_sum += cavity.m**3*cavity.Vc*np.sin(cavity.theta)

        a = self.ring.ac * self.ring.omega1 / (2 * self.ring.E0 * self.ring.T0) * a_sum
        b = self.ring.ac * self.ring.omega1**2 / (6 * self.ring.E0 * self.ring.T0) * b_sum
        c = - self.ring.ac * self.ring.omega1**3 / (24 * self.ring.E0 * self.ring.T0) * c_sum

        return (a, b, c)

    def _bunch_length(self, a, b, c):
        # Eq. (8) in [1]

        U0 = self.ring.ac**2 * self.ring.sigma_delta**2/2

        U = lambda tau: a*tau**2 + b*tau**3 + c*tau**4

        numerator = lambda tau: tau**2*np.exp(-U(tau)/(2*U0))
        denominator = lambda tau: np.exp(-U(tau)/(2.*U0))

        num = quad(numerator, -self.tau_boundary, self.tau_boundary)[0]
        den = quad(denominator, -self.tau_boundary, self.tau_boundary)[0]

        bunch_length = np.sqrt(num/den)

        return bunch_length

    def _form_factors(self, f, m, bunch_length):

        omega = 2*np.pi*f
        F = np.exp(-m**2*omega**2*bunch_length**2/2)

        return F

    def _robinson_frequency(self):
        # Eq. (10) in [1]

        sum_val = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            sum_val += cavity.m*self.F[i]*cavity.Vc*np.sin(cavity.theta)
        omega2 = self.ring.ac * self.ring.omega1 / (self.ring.E0 * self.ring.T0) * sum_val
        # Protect against negative value under sqrt (can happen with certain cavity phase configurations)
        if omega2 < 0:
            return np.nan  # Invalid Robinson frequency
        return np.sqrt(omega2)

    def _dipole_coupled_bunch_growth_rate(self, f_HOM, Z_HOM, bunch_length, omega_r):
        # Eq. (22) in [1]

        from_factor = self._form_factors(f_HOM, 1, bunch_length)
        gr = ((self.I0 * self.ring.eta() * 2 * np.pi * f_HOM * Z_HOM * from_factor**2) /
              (2 * self.ring.E0 * self.ring.T0 * omega_r))

        return gr

    def _dipole_coupled_bunch(self, f_HOM, Z_HOM, bunch_length, omega_r, landau):
        HOM = False
        if isinstance(f_HOM, (int, float)):
            f_HOM = [f_HOM]
        if isinstance(Z_HOM, (int, float)):
            Z_HOM = [Z_HOM]

        if len(f_HOM) != len(Z_HOM):
            raise ValueError("f_HOM and Z_HOM must have the same length.")
        else:
            N = len(f_HOM)

        for i in range(N):
            gr_HOM = self._dipole_coupled_bunch_growth_rate(f_HOM[i], Z_HOM[i], bunch_length, omega_r)
            # Check if growth rate is stronger than synchrotron damping
            gr = gr_HOM - 1/self.ring.tau[2]
            if gr > 0:
                if gr_HOM > landau[0]:
                    HOM = True
                    break
        return HOM

    def _zero_frequency_no_coupling(self, mu, bunch_length):
        # Eq. (14) in [1]
        coef1 = self.ring.ac * self.ring.omega1 * self.I0 / (self.ring.E0 * self.ring.T0)
        coef2 = (self.ring.omega1 * bunch_length)**(2*mu-2)/(2**(mu-1)*factorial(mu))
        sum_val = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            sum_val += cavity.m**(2*mu - 1) * cavity.RL * self.F[i]**2 * np.sin(-2*cavity.psi)
        return coef1 * coef2 * sum_val

    def _zero_frequency_instability(self, omega_r, bunch_length, mode_coupling):
        # Eq. (14) in [1]
        if mode_coupling is False:
            if omega_r**2 < self._zero_frequency_no_coupling(1, bunch_length):
                return True
            elif omega_r**2 < self._zero_frequency_no_coupling(2, bunch_length):
                return True
            elif omega_r**2 < self._zero_frequency_no_coupling(3, bunch_length):
                return True
            elif omega_r**2 < self._zero_frequency_no_coupling(4, bunch_length):
                return True
            else:
                return False
        else:
            # These conditions are the same for no coupling and with coupling (comparison with A_tilde and B_tilde results in same formula) except for addition of coupled zero-frequency condition
            if omega_r**2 - self._A_tilde(0) < 0:
                return True
            elif 4*omega_r**2 - self._B_tilde(0, bunch_length) < 0:
                return True
            # Eq. (B.12) in [1]
            elif (omega_r**2 - self._A_tilde(0))*(4*omega_r**2 - self._B_tilde(0, bunch_length)) + self._d_tilde(0, bunch_length) < 0:
                return True
            else:
                return False

    def _landau_threshold(self, omega_r, c, b):
        # Eq. (19) in [1]
        dipole_landau = 0.78*self.ring.ac**2*self.ring.sigma_delta**2/omega_r*np.abs(3*c/omega_r**2 - (3*b/omega_r**2)**2)

        landau_threshold = np.zeros((4,1))
        landau_threshold[0] = dipole_landau
        landau_threshold[1] = 2.24/0.78*dipole_landau
        landau_threshold[2] = 4.12/0.78*dipole_landau
        landau_threshold[3] = 6.36/0.78*dipole_landau

        return landau_threshold

    def _robinson_damping_rate(self, Omega, mu, bunch_length):
        # Eq. (16) in [1]
        coef1 = 8 * self.ring.ac * self.I0 / (self.ring.E0 * self.ring.T0)
        coef2 = mu * (self.ring.omega1 * bunch_length)**(2*mu - 2) / (2**mu * factorial(mu-1))
        sum_val = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            prod = np.tan(-cavity.psi) * np.cos(self._phi_pm(cavity, Omega, "+"))**2 * np.cos(self._phi_pm(cavity, Omega, "-"))**2
            sum_val += cavity.m**(2*mu - 2) * self.F[i]**2  * cavity.RL * cavity.QL * prod
        return coef1 * coef2 * sum_val

    def _coupled_bunch_mode1_damping_rate(self, Omega, mu, bunch_length):
        # Eq. (15) in [1]
        coef1 = self.ring.ac * self.ring.omega1 * self.I0 / (Omega * self.ring.E0 * self.ring.T0)
        coef2 = mu*(self.ring.omega1 * bunch_length)**(2*mu - 2) / (2**mu * factorial(mu-1))
        sum_val = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            diff = np.cos(self._phi_pm(cavity, Omega, "-"))**2 - np.cos(self._phi_pm(cavity, Omega, "+"))**2
            sum_val += cavity.m**(2*mu - 1) * self.F[i]**2  * cavity.RL * diff
        return coef1 * coef2 * sum_val

    def _robinson_no_coupling(self, omega_r, bunch_length, landau_threshold):
        # Follow III.A.(vii) in [1].

        # Calculate AC robinson growth rates
        modes = np.array([1,2,3,4])

        # Output array
        ac_robinson = np.zeros(len(modes),dtype=bool)
        Omega_modes = np.zeros(len(modes),dtype=float)
        converged = np.zeros(len(modes),dtype=bool)

        # Calculate the Robinson frequency
        for mu in modes:

            # Initial guess based on zero current
            Omega0 = mu*omega_r

            # Eq. (13) in [1]
            def f(Omega):
                coef1 = self.ring.ac * self.ring.omega1 * self.I0 / (self.ring.E0 * self.ring.T0)
                coef2 = mu * (self.ring.omega1 * bunch_length)**(2*mu - 2) / (2**mu * factorial(mu-1))
                sum_val = 0
                for i in range(self.n_cavity):
                    cavity = self.cavity_list[i]
                    brackets = np.sin(2*self._phi_pm(cavity, Omega, "-")) + np.sin(2*self._phi_pm(cavity, Omega, "+"))
                    sum_val += cavity.m**(2*mu - 1) * cavity.RL * self.F[i]**2 * brackets
                inner = (mu * omega_r)**2 - coef1 * coef2 * sum_val
                return Omega - np.sqrt(inner)

            sol = root_scalar(f, x0=Omega0, xtol=1e-6)

            if sol.converged:
                # Save the final result
                Omega = sol.root
                Omega_modes[mu-1] = Omega
                converged[mu-1] = True

                # When the Robinson frequency is found calculate the damping rate - Eq. (16) in [1]
                alpha_r = self._robinson_damping_rate(Omega, mu, bunch_length)

                # Add radiation damping to the damping rate
                alpha_r_incl_rad_damp = alpha_r + mu/self.ring.tau[2]

                # Complex frequency shift
                delta_Omega = Omega - 1j*alpha_r - mu*omega_r

                # Check if the instability occur
                if alpha_r_incl_rad_damp < 0:
                    # Check if Landau damping is overcome
                    if np.abs(delta_Omega) > landau_threshold[mu-1]:
                        ac_robinson[mu-1] = True
            else:
                converged[mu-1] = False

        return ac_robinson, Omega_modes, converged

    def _coupled_bunch_mode1(self, omega_r, bunch_length, landau_threshold):
        # Follow III.A.(viii) in [1].

        # Calculate coupled bunch mode +-1 growth rates
        modes = ["+1","-1"]
        mu = 1

        # Output array
        coupled_mode1 = np.zeros(len(modes),dtype=bool)
        mode1_modes = np.zeros(len(modes),dtype=float)
        converged = np.zeros(len(modes),dtype=bool)

        # Calculate the coupled bunch mode +-1 frequency
        for j,  mode in enumerate(modes):
            if mode == "+1":
                added = self.ring.omega0
            elif mode == "-1":
                added = - self.ring.omega0

            # Initial guess based on zero current
            Omega0 = mu*omega_r

            # Eq.13 with Omega -> Omega +- omega0
            def f(Omega):
                coef1 = self.ring.ac * self.ring.omega1 * self.I0 / (self.ring.E0 * self.ring.T0)
                coef2 = mu * (self.ring.omega1 * bunch_length)**(2*mu - 2) / (2**mu * factorial(mu-1))
                sum_val = 0
                for i in range(self.n_cavity):
                    cavity = self.cavity_list[i]
                    brackets = np.sin(2*self._phi_pm(cavity, Omega + added, "-")) + np.sin(2*self._phi_pm(cavity, Omega + added, "+"))
                    sum_val += cavity.m**(2*mu - 1) * cavity.RL * self.F[i]**2 * brackets
                inner = (mu * omega_r)**2 - coef1 * coef2 * sum_val
                return Omega - np.sqrt(inner)

            sol = root_scalar(f, x0=Omega0, xtol=1e-6)

            if sol.converged:
                # Save the final result
                Omega = sol.root
                mode1_modes[j] = Omega
                converged[j] = True

                # When the coupled bunch mode +-1 frequency is found calculate the damping rate - Eq. 15
                alpha_r = self._coupled_bunch_mode1_damping_rate(Omega + added, mu, bunch_length)

                # Add radiation damping to the damping rate
                alpha_r_incl_rad_damp = alpha_r + mu/self.ring.tau[2]

                # Complex frequency shift
                delta_Omega = Omega - 1j*alpha_r - mu*omega_r

                # Check if the instability occur
                if alpha_r_incl_rad_damp < 0:
                    # Check if Landau damping is overcome
                    if np.abs(delta_Omega) > landau_threshold[mu-1]:
                        coupled_mode1[j] = True
            else:
                converged[j] = False

        return coupled_mode1, mode1_modes, converged

    def _robinson_coupling(self, omega_r, bunch_length, landau_threshold, abs_val=True):
        # Follow III.A.(ix) in [1].

        methods = ["hybr", "broyden1"] # These two methods seem enough to reach converge in most cases.
        coupled_dipole = False
        coupled_quadrupole = False
        fast_mode_coupling_bydip = False
        fast_mode_coupling_byquad = False
        Omega_dip = np.nan
        Omega_quad = np.nan

        # Coupled dipole
        def f_dipole(x):
            omega_b13 = self._Omega_B13(x[0], x[1], bunch_length, omega_r, "-", abs_val)
            # Protect against negative sqrt
            if omega_b13 < 0:
                y0 = np.nan
            else:
                y0 = x[0] - np.sqrt(omega_b13)
            y1 = x[1] - self._ar_B11(x[0], bunch_length, omega_r)
            return np.array([y0, y1])

        # Initial guess based on zero current
        x0 = np.array([omega_r, 1/self.ring.tau[2]])

        for method in methods:
            try:
                sol = root(f_dipole, x0, tol=1e-6, method=method)
            except ValueError:
                sol = {"success":False}
            if sol["success"]:
                break

        if sol["success"]:
            # Save the final result
            Omega_dip = sol.x[0]
            alpha_r_incl_rad_damp_dip = sol.x[1]
            delta_Omega_dip = Omega_dip - 1j*alpha_r_incl_rad_damp_dip - omega_r
            converged_dip = True

            # Check fast mode-coupling instability Eq. inside square root of B13
            root_val = self._Omega_B13(Omega_dip, alpha_r_incl_rad_damp_dip, bunch_length, omega_r, "-", return_root=True)
            if root_val < 0:
                fast_mode_coupling_bydip = True

            # Check if coupled-dipole instability exists
            if alpha_r_incl_rad_damp_dip < 0:
                # Check if Landau damping is overcome
                if np.abs(delta_Omega_dip) > landau_threshold[0]:
                    coupled_dipole = True
        else:
            converged_dip = False

        # Coupled quadrupole
        def f_quadrupole(x):
            omega_b13 = self._Omega_B13(x[0], x[1], bunch_length, omega_r, "+", abs_val)
            # Protect against negative sqrt
            if omega_b13 < 0:
                y0 = np.nan
            else:
                y0 = x[0] - np.sqrt(omega_b13)
            y1 = x[1] - self._ar_B11(x[0], bunch_length, omega_r)
            return np.array([y0, y1])

        # Initial guess based on zero current
        x0 = np.array([2*omega_r, 2/self.ring.tau[2]])

        for method in methods:
            try:
                sol = root(f_quadrupole, x0, tol=1e-6, method=method)
            except ValueError:
                sol = {"success":False}
            if sol["success"]:
                break

        if sol["success"]:
            # Save the final result
            Omega_quad = sol.x[0]
            alpha_r_incl_rad_damp_quad = sol.x[1]
            delta_Omega_quad = Omega_quad - 1j*alpha_r_incl_rad_damp_quad - 2*omega_r
            converged_quad = True

            # Check fast mode-coupling instability Eq. inside square root of B13
            root_val = self._Omega_B13(Omega_quad, alpha_r_incl_rad_damp_quad, bunch_length, omega_r, "+", return_root=True)
            if root_val < 0:
                fast_mode_coupling_byquad = True

            # Check if coupled-quadrupole instability exists
            if alpha_r_incl_rad_damp_quad < 0:
                # Check if Landau damping is overcome
                if np.abs(delta_Omega_quad) > landau_threshold[1]:
                    coupled_quadrupole = True
        else:
            converged_quad = False

        robinson = np.array([coupled_dipole, coupled_quadrupole, fast_mode_coupling_bydip, fast_mode_coupling_byquad])
        Omega = np.array([Omega_dip, Omega_quad])
        converged = np.array([converged_dip, converged_quad])

        return (robinson, Omega, converged)

    def _converge_form_factors(self,
                              passive_harmonic_cavity=True,
                              auto_set_MC_theta=True,
                              optimal_tunning=True,
                              method="Venturini",
                              **kwargs
                              ):
        if method == "Bosch":
            if passive_harmonic_cavity:
                out = self._conv_ff_passive_bosch(auto_set_MC_theta, optimal_tunning, **kwargs)
            else:
                out = self._conv_ff_active_bosch(optimal_tunning, **kwargs)
        elif method == "Venturini":
            out = self._conv_ff_venturini(passive_harmonic_cavity, auto_set_MC_theta, optimal_tunning, **kwargs)
        elif method == "Alves":
            out = self._conv_ff_alves(passive_harmonic_cavity, auto_set_MC_theta, optimal_tunning, **kwargs)
        else:
            raise ValueError(f"method {method} is not correct.")
        (bunch_length, a, b, c, converged) = out
        return (bunch_length, a, b, c, converged)

    def _conv_ff_passive_bosch(self,
                                auto_set_MC_theta=True,
                                optimal_tunning=True,
                                **kwargs
                                ):
        # Follow III.A. from (i) to (v) in [1] with added information from [2].

        max_counter = kwargs.get("max_counter", 200)

        self.F = np.zeros(self.n_cavity)
        self.F[0] = 1
        self.F[1:] = 0.1

        MC = self.cavity_list[0]

        diff = 1
        counter = 0
        max_counter = max_counter
        converged = True
        old_F = np.zeros_like(self.F)
        while(diff > 0.01):
            counter += 1
            if counter > max_counter:
                converged = False
                break

            if auto_set_MC_theta:
                # Calculate synchronous phase
                delta = 0
                for i in range(self.n_cavity):
                    if i == 0:
                        continue
                    cavity = self.cavity_list[i]
                    delta += cavity.Vb(self.I0) * self.F[i] * np.cos(cavity.psi)
                MC.theta = np.arccos((self.ring.U0 + delta) / MC.Vc)

            if optimal_tunning:
                # Optimal detuning for main cavity
                MC.set_optimal_detune(self.I0, self.F[0])

            # Get theta & Vc (from (iii) near Eq. (21) in [1])
            for i in range(self.n_cavity):
                if i == 0:
                    continue
                cavity = self.cavity_list[i]
                cavity.theta = np.arctan(np.sin(-2*cavity.psi)/(-2*np.cos(cavity.psi)**2))
                cavity.Vc = -2*self.I0*self.F[i]*cavity.RL*np.cos(cavity.psi)**2 / np.cos(cavity.theta)

            # Taylor expansion coeff
            a, b, c = self._potential_decomposition()

            # Bunch length
            bunch_length = self._bunch_length(a, b, c)

            # Form factors
            diff = 0
            for i in range(self.n_cavity):
                cavity = self.cavity_list[i]
                old_F[i] = self.F[i]
                self.F[i] = self._form_factors(self.ring.f1, cavity.m, bunch_length)
                diff += np.abs((self.F[i] - old_F[i])/self.F[i])

            if diff > 0.01:
                for i in range(self.n_cavity):
                    cavity = self.cavity_list[i]
                    # From 4. in [2].
                    self.F[i] = (self.F[i]*0.1 + old_F[i]*0.9)

        return (bunch_length, a, b, c, converged)

    def _conv_ff_active_bosch(self, optimal_tunning=True, **kwargs):
        # Follow V.A. from (i) to (iv) in [1].
        # kwargs: auto_set_for_xi, xi

        auto_set_for_xi = kwargs.get("auto_set_for_xi", False)

        self.F = np.zeros(self.n_cavity)
        self.F[0] = 1
        self.F[1:] = 0.1

        if auto_set_for_xi:
            xi = kwargs.get("xi", None)
            if xi is None:
                raise ValueError("xi must be given if auto_set_for_xi=True")
            MC = self.cavity_list[0]
            HC = self.cavity_list[1]
            MC.theta = np.arccos(self.ring.U0/(MC.Vc*(1 - 1/HC.m**2)))
            HC.theta = np.arctan(HC.m * xi * np.tan(MC.theta)) - np.pi
            HC.Vc = - xi * MC.Vc * np.sin(MC.theta) / (HC.m * np.sin(HC.theta))

        (a, b, c) = self._potential_decomposition()
        bunch_length = self._bunch_length(a, b, c)
        for i, cavity in enumerate(self.cavity_list):
            self.F[i] = self._form_factors(self.ring.f1, cavity.m, bunch_length)

        if optimal_tunning:
            for i, cavity in enumerate(self.cavity_list):
                cavity.set_optimal_detune(self.I0, self.F[i])

        return (bunch_length, a, b, c, True)

    def _conv_ff_venturini(self,
                            passive_harmonic_cavity=True,
                            auto_set_MC_theta=True,
                            optimal_tunning=True,
                            **kwargs):

        self.F = kwargs.get("F_init", np.ones(self.n_cavity))
        set_MC_phase_HCpassive = kwargs.get("set_MC_phase_HCpassive", False)

        MC = self.cavity_list[0]
        HC = self.cavity_list[1]

        if set_MC_phase_HCpassive:
            # Calculate synchronous phase
            delta = 0
            for i in range(self.n_cavity):
                if i == 0:
                    continue
                cavity = self.cavity_list[i]
                delta += cavity.Vb(self.I0) * self.F[i] * np.cos(cavity.psi)
            MC.theta = np.arccos((self.ring.U0 + delta) / MC.Vc)

        if optimal_tunning:
            # Optimal detuning for main cavity
            MC.set_optimal_detune(self.I0)

        if optimal_tunning and not passive_harmonic_cavity:
            # Optimal detuning for active harmonic cavity
            HC.set_optimal_detune(self.I0)
            
        MC.set_generator(self.I0)
        if passive_harmonic_cavity:
            HC.Vg = 0
            HC.theta_g = 0
        else:
            HC.set_generator(self.I0)

        BLE = BeamLoadingEquilibrium(self.ring, self.cavity_list, self.I0,
                                   auto_set_MC_theta=auto_set_MC_theta,
                                   B1=-1*self.tau_boundary*c,
                                   B2=self.tau_boundary*c)

        # Enhanced solver with multiple methods for better convergence
        # This helps with SOLEIL II and similar configurations at high psi/current
        methods_to_try = ['hybr', 'lm', 'broyden1', 'krylov']
        tolerances_to_try = [1e-4, 1e-3, 1e-2]
        
        converged = False
        sol = None
        
        # Generate initial guesses based on psi angle for better convergence
        # For high psi (60-90 deg), form factors can be quite different from 1
        psi_rad = getattr(HC, 'psi', 0)
        psi_deg = np.rad2deg(abs(psi_rad)) if psi_rad != 0 else 0
        
        # Create multiple initial guesses
        initial_guesses = []
        # Standard guess
        initial_guesses.append(None)  # None means use default [1,0,1,0,...]
        # For high psi angles, try reduced form factors
        if psi_deg > 45:
            # At high psi, form factors may be reduced
            for f_init in [0.9, 0.8, 0.7, 0.5]:
                initial_guesses.append([f_init, 0] * self.n_cavity + ([MC.theta] if auto_set_MC_theta else []))

        try:
            # Try each method with each tolerance and initial guess
            for method in methods_to_try:
                if converged:
                    break
                for tol in tolerances_to_try:
                    if converged:
                        break
                    for x0 in initial_guesses:
                        if converged:
                            break
                        try:
                            sol = BLE.beam_equilibrium(x0=x0, method=method, tol=tol)
                            if sol.success:
                                converged = True
                                break
                        except Exception:
                            continue
            
            if converged and sol is not None:
                bunch_length = BLE.std_rho()/c
                self.F = BLE.F
                self.PHI = BLE.PHI
                self.BLE = BLE
                
                # Get theta & Vc from phasor addition to take into account F&PHI and generator voltage
                for i, cavity in enumerate(self.cavity_list):
                    Vc_phasor = -1*cavity.Vb(self.I0) * self.F[i] * np.exp(1j*(cavity.psi - self.PHI[i])) + cavity.Vg * np.exp(1j*cavity.theta_g)
                    cavity.Vc = np.abs(Vc_phasor)
                    cavity.theta = np.angle(Vc_phasor)
                
                # Taylor expansion coeff
                a, b, cc = self._potential_decomposition()
            else:
                bunch_length = None
                a = None
                b = None
                cc = None
        except:
            converged = False
            bunch_length = None
            a = None
            b = None
            cc = None

        return (bunch_length, a, b, cc, converged)

    def _conv_ff_alves(self,
                        passive_harmonic_cavity=True,
                        auto_set_MC_theta=True,
                        optimal_tunning=True,
                        **kwargs):

        if passive_harmonic_cavity is False:
            raise NotImplementedError("Alves method with active harmonic cavities is not implemented.")

        # kwargs
        niter = kwargs.get('niter', 200)
        tol = kwargs.get('tol', 1e-8)
        beta = kwargs.get('beta', 0.1)
        m = kwargs.get('m', 3)
        print_flag = kwargs.get('print_flag', False)
        filling = kwargs.get("filling", None)
        impedance = kwargs.get("impedance", None)
        self.F = kwargs.get("F_init", np.ones(self.n_cavity))
        self.PHI = np.zeros(self.n_cavity)
        set_MC_phase_HCpassive = kwargs.get("set_MC_phase_HCpassive", False)

        MC = self.cavity_list[0]
        HC = self.cavity_list[1]
        
        if set_MC_phase_HCpassive:
            # Calculate synchronous phase
            delta = 0
            for i in range(self.n_cavity):
                if i == 0:
                    continue
                cavity = self.cavity_list[i]
                delta += cavity.Vb(self.I0) * self.F[i] * np.cos(cavity.psi)
            MC.theta = np.arccos((self.ring.U0 + delta) / MC.Vc)

        if optimal_tunning:
            # Optimal detuning for main cavity
            MC.set_optimal_detune(self.I0)

        MC.set_generator(self.I0)
        HC.Vg = 0
        HC.theta_g = 0

        if filling is None:
            filling = np.ones(self.ring.h) / self.ring.h
            identical_bunches = True
        else:
            identical_bunches = False
        
        nb = sum(filling != 0)

        ring_pce = synchrotron_to_pycolleff(self.ring, self.I0, MC.Vc, nb)
        mc_pce = cavityresonator_to_pycolleff(MC, Impedance=False)
        hc_pce = cavityresonator_to_pycolleff(HC, Impedance=True)
        sources = [mc_pce, hc_pce]

        if impedance is not None:
            impedance_pce = impedance_to_pycolleff(impedance)
            sources += [impedance_pce]

        longeq = LongitudinalEquilibrium(ring=ring_pce,
                                         impedance_sources=sources,
                                         fillpattern=filling)
        longeq.identical_bunches = identical_bunches

        longeq.feedback_on = True
        longeq.feedback_method = 0 # 0 = Phasor, 1 = LS
        
        # Numerical parameters from M. Alves example
        longeq.zgrid = np.linspace(-1, 1, 2001) * ring_pce.rf_lamb / 2
        longeq.max_mode = 1000*ring_pce.harm_num
        longeq.min_mode0_ratio = 1e-10
        try:
            _, converged = longeq.calc_longitudinal_equilibrium(niter,
                                                                   tol,
                                                                   beta,
                                                                   m,
                                                                   print_flag)
            z0, sigmaz = longeq.calc_moments(longeq.zgrid, longeq.distributions)
            bunch_length = sigmaz[0]/c

            if auto_set_MC_theta:
                ring_pce = ring_pce = synchrotron_to_pycolleff(self.ring, self.I0, MC.Vc, nb)
                longeq = LongitudinalEquilibrium(ring=ring_pce,
                                                 impedance_sources=sources,
                                                 fillpattern=filling)
                longeq.identical_bunches = identical_bunches

                longeq.main_ref_phase_offset = z0[0]/c*self.ring.omega1
                longeq.feedback_on = True
                longeq.feedback_method = 0 # 0 = Phasor, 1 = LS

                longeq.zgrid = np.linspace(-1, 1, 2001) * ring_pce.rf_lamb / 2
                longeq.max_mode = 1000*ring_pce.harm_num
                longeq.min_mode0_ratio = 1e-10
                _, converged = longeq.calc_longitudinal_equilibrium(niter,
                                                                       tol,
                                                                       beta,
                                                                       m,
                                                                       print_flag)
        except:
            converged = False

        if converged:
            z0, sigmaz = longeq.calc_moments(longeq.zgrid, longeq.distributions)
            bunch_length = sigmaz[0]/c
            self.LE = longeq
            self.LE.ring.bunlen = sigmaz[0]
            # FF
            zgrid = longeq.zgrid - z0[0]
            rho = longeq.distributions[0]
            for i, cavity in enumerate(self.cavity_list):
                exp = np.exp(1j * self.ring.k1 * cavity.m * zgrid)
                FF = trapezoid(exp * rho, x=zgrid)
                self.F[i] = np.real(FF)
                self.PHI[i] = np.imag(FF)

            # Get Vg and theta_g
            MC.Vg = longeq.main_gen_amp_mon
            MC.theta_g = -1*longeq.main_gen_phase_mon


            # Get theta & Vc from phasor addition to take into account F&PHI and generator voltage
            # (instead of (iii) near Eq. 21)
            for i, cavity in enumerate(self.cavity_list):
                Vc_phasor = -1*cavity.Vb(self.I0) * self.F[i] * np.exp(1j*(cavity.psi - self.PHI[i])) + cavity.Vg * np.exp(1j*cavity.theta_g)
                cavity.Vc = np.abs(Vc_phasor)
                cavity.theta = np.angle(Vc_phasor)

            # Taylor expansion coeff
            a, b, cc = self._potential_decomposition()
        else:
            bunch_length = None
            a = None
            b = None
            cc = None

        return (bunch_length, a, b, cc, converged)

    def _solve_core(self, bunch_length, a, b, c, mode_coupling, f_HOM, Z_HOM, method, **kwargs):
        
        use_GaussLMCI = kwargs.get('use_GaussLMCI', False)
        zero_frequency = None
        robinson = None
        HOM = None
        Omega = None

        omega_r = self._robinson_frequency()
        landau = self._landau_threshold(omega_r, c, b)

        HOM = self._dipole_coupled_bunch(f_HOM, Z_HOM, bunch_length, omega_r, landau)

        zero_frequency = self._zero_frequency_instability(omega_r, bunch_length, mode_coupling)
        if zero_frequency:
            return (zero_frequency, robinson, HOM, Omega, True)
        
        if method == "Alves" and use_GaussLMCI:
            robinson, Omega, converged = self._FMCI_alves(**kwargs)
        else:
            if mode_coupling:
                robinson, Omega, converged = self._robinson_coupling(omega_r, bunch_length, landau)
            else:
                robinson, Omega, converged = self._robinson_no_coupling(omega_r, bunch_length, landau)

        return (zero_frequency, robinson, HOM, Omega, converged)

    def _omega_n(self, cavity):
        # cavity resonant frequency - in text near II.A in [1]
        return 2 * cavity.QL * cavity.m * self.ring.omega1 / (2 * cavity.QL + np.tan(- cavity.psi))

    def _phi_pm(self, cavity, Omega, sign):
        # Near Eq. (13) in [1]
        omega_n = self._omega_n(cavity)
        if sign == "+":
            return np.arctan(2*cavity.QL*(cavity.m*self.ring.omega1+Omega-omega_n)/omega_n)
        elif sign == "-":
            return np.arctan(2*cavity.QL*(cavity.m*self.ring.omega1-Omega-omega_n)/omega_n)

    def _A_tilde(self, Omega):
        # Eq. (B8) in [1]
        coef = self.ring.ac * self.ring.omega1 * self.I0 / (2* self.ring.E0 * self.ring.T0)
        sum_val = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            brackets = np.sin(2*self._phi_pm(cavity, Omega, "-")) + np.sin(2*self._phi_pm(cavity, Omega, "+"))
            sum_val += cavity.m * cavity.RL * self.F[i]**2 * brackets
        return coef * sum_val

    def _B_tilde(self, Omega, bunch_length):
        # Eq. (B8) in [1]
        coef1 = self.ring.ac * self.ring.omega1 * self.I0 / (2* self.ring.E0 * self.ring.T0)
        coef2 = (self.ring.omega1 * bunch_length)**2
        sum_val = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            brackets = np.sin(2*self._phi_pm(cavity, Omega, "-")) + np.sin(2*self._phi_pm(cavity, Omega, "+"))
            sum_val += cavity.m**3 * cavity.RL * self.F[i]**2 * brackets
        return coef1 * coef2 * sum_val

    def _D_tilde(self, Omega, bunch_length):
        # Eq. (B8) in [1]
        coef2 = self.ring.ac * self.ring.omega1 * self.I0 / (2* self.ring.E0 * self.ring.T0)
        coef1 = (self.ring.omega1 * bunch_length)
        sum_val = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            brackets = np.sin(2*self._phi_pm(cavity, Omega, "-")) - np.sin(2*self._phi_pm(cavity, Omega, "+"))
            sum_val += cavity.m**2 * cavity.RL * self.F[i]**2 * brackets
        return coef1 * coef2 * sum_val

    def _a_tilde(self, Omega):
        # Eq. (B8) in [1]
        coef = self.ring.ac * self.ring.omega1 * self.I0 / (self.ring.E0 * self.ring.T0)
        sum_val = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            brackets = np.cos(self._phi_pm(cavity, Omega, "-"))**2 - np.cos(self._phi_pm(cavity, Omega, "+"))**2
            sum_val += cavity.m * cavity.RL * self.F[i]**2 * brackets
        return coef * sum_val + 2 * Omega / self.ring.tau[2]

    def _b_tilde(self, Omega, bunch_length):
        # Eq. (B8) in [1]
        coef1 = self.ring.ac * self.ring.omega1 * self.I0 / (self.ring.E0 * self.ring.T0)
        coef2 = (self.ring.omega1 * bunch_length)**2
        sum_val = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            brackets = np.cos(self._phi_pm(cavity, Omega, "-"))**2 - np.cos(self._phi_pm(cavity, Omega, "+"))**2
            sum_val += cavity.m**3 * cavity.RL * self.F[i]**2 * brackets
        return coef1 * coef2 * sum_val + 4 * Omega / self.ring.tau[2]

    def _d_tilde(self, Omega, bunch_length):
        # Eq. (B8) in [1]
        coef2 = self.ring.ac * self.ring.omega1 * self.I0 / (self.ring.E0 * self.ring.T0)
        coef1 = (self.ring.omega1 * bunch_length)
        sum_val = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            brackets = np.cos(self._phi_pm(cavity, Omega, "-"))**2 + np.cos(self._phi_pm(cavity, Omega, "+"))**2
            sum_val += cavity.m**2 * cavity.RL * self.F[i]**2 * brackets
        return coef1 * coef2 * sum_val

    def _ar_B11(self, Omega, bunch_length, omega_r):
        # Eq. (B11) in [1]
        a = self._a_tilde(Omega)
        b = self._b_tilde(Omega, bunch_length)
        d = self._d_tilde(Omega, bunch_length)
        A = self._A_tilde(Omega)
        B = self._B_tilde(Omega, bunch_length)
        D = self._D_tilde(Omega, bunch_length)
        num = a * (Omega**2 - (2 * omega_r)**2 + B) + b*(Omega**2 - omega_r**2 + A) - 2*D*d
        den = 2 * Omega * (2 * Omega**2 - 5*omega_r**2 + A + B)
        return num/den

    def _Omega_B13(self, Omega, alpha_r, bunch_length, omega_r, sign, abs_val=True, return_root=False):
        # Eq. (B13) in [1]
        a = self._a_tilde(Omega)
        b = self._b_tilde(Omega, bunch_length)
        d = self._d_tilde(Omega, bunch_length)
        A = self._A_tilde(Omega)
        B = self._B_tilde(Omega, bunch_length)
        D = self._D_tilde(Omega, bunch_length)
        part1 = (5*omega_r**2 - A - B)/2
        root_val = (3*omega_r**2 + A - B)**2 / 4 + D**2 - d**2 + (a - 2*Omega*alpha_r)*(b - 2*Omega*alpha_r)
        if return_root:
            return root_val
        if abs_val:
            root_val = np.abs(root_val)
        if sign == "+":
            return part1 + np.sqrt(root_val)
        elif sign == "-":
            return part1 - np.sqrt(root_val)

    def _PTBL_alves(self, **kwargs):
        # kwargs
        max_azi = kwargs.get("max_azi", 10)
        max_rad = kwargs.get("max_rad", 10)
        
        # Synchrotron tune from bunch length and maching condition
        sync_freq = self.ring.sigma_delta * self.ring.eta() * c
        sync_freq /= self.LE.ring.bunlen * 2 * np.pi
        self.LE.ring.sync_tune = sync_freq / self.ring.f0
        
        # All impedance source as ImpedanceDFT
        for idx, _ in enumerate(self.LE.impedance_sources):
            self.LE.impedance_sources[idx].calc_method = ImpedanceSource.Methods.ImpedanceDFT

        # Calculate Vlasov's equation eigen-frequencies:
        eigenfreq, *_ = self.LE.calc_mode_coupling(
            w=[-10 * self.ring.omega1, +10 * self.ring.omega1],
            cbmode=1,
            max_azi=max_azi,
            max_rad=max_rad,
            use_fokker=False,
            delete_m0=True,
            delete_m0k0=True,
            reduced=True,
        )

        # Find most unstable mode:
        idx = np.argmax(eigenfreq.imag)
        grate = eigenfreq.imag[idx]

        # Subtract radiation damping rate
        grate -= 1 / self.ring.tau[2]

        PTBL = grate > 0

        return PTBL
    
    def _FMCI_alves(self, **kwargs):
        # kwargs
        max_azi = kwargs.get("max_azi", 2)
        max_rad = kwargs.get("max_rad", 0)
        
        # Synchrotron tune from bunch length and maching condition
        sync_freq = self.ring.sigma_delta * self.ring.eta() * c
        sync_freq /= self.LE.ring.bunlen * 2 * np.pi
        self.LE.ring.sync_tune = sync_freq / self.ring.f0
        
        # All impedance source as ImpedanceDFT
        for idx, _ in enumerate(self.LE.impedance_sources):
            self.LE.impedance_sources[idx].calc_method = ImpedanceSource.Methods.ImpedanceDFT

        # Calculate Vlasov's equation eigen-frequencies:
        eigenfreq, *_ = self.LE.calc_mode_coupling(
            w=[-300 * self.ring.omega1, +300 * self.ring.omega1],
            cbmode=0,
            max_azi=max_azi,
            max_rad=max_rad,
            use_fokker=False,
            delete_m0=True,
            delete_m0k0=True,
            reduced=True,
        )

        # Find most unstable mode:
        idx = np.argmax(eigenfreq.imag)
        grate = eigenfreq.imag[idx]

        # Subtract radiation damping rate
        grate -= 1 / self.ring.tau[2]

        FMCI = grate > 0
            
        robinson = np.array([False, False, FMCI, FMCI])
        Omega = np.array([eigenfreq.real[0], eigenfreq.real[1]])
        converged = np.array([True, True])

        return (robinson, Omega, converged)
    
    def _PTBL_He(self):
        
        m = self.ring.h

        MC = self.cavity_list[0]
        HC = self.cavity_list[1]
        a = np.exp(-HC.wr * self.ring.T0 / (2 * HC.Q))
        theta = HC.detune * 2 * np.pi * self.ring.T0
        dtheta = np.arcsin((1-a) * np.cos(theta / 2) /
                           (np.sqrt(1 + a**2 - 2 * a * np.cos(theta))))

        k = np.arange(1, m)
        d_k = np.exp(-1 * HC.wr * self.ring.T0 * (k-1) / (2 * HC.Q * m))
        theta_k = (theta/2 + dtheta - (k-1) / m * theta)
        eps_k = 1 - np.sin(np.pi / 2 - k * 2 * np.pi / m)

        num = np.sum(eps_k * d_k * np.cos(theta_k))
        f = num / (m * np.sqrt(1 + a**2 - 2 * a * np.cos(theta)))

        eta = (2 * np.pi * HC.m**2 * self.F[1] * self.ring.h * self.I0 *
               HC.Rs / HC.Q * f /
               (MC.Vc * np.sin(MC.theta - self.PHI[1] / HC.m)))
        
        if eta > 1:
            PTBL = True
        else:
            PTBL = False
            
        return PTBL
