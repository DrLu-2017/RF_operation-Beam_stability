# -*- coding: utf-8 -*-
"""
Module with conversion function to go from mbtrack2 to pycolleff.
"""
import numpy as np
from scipy.constants import c
from mbtrack2.utilities.misc import double_sided_impedance
from pycolleff.colleff import Ring
from pycolleff.longitudinal_equilibrium import ImpedanceSource

def synchrotron_to_pycolleff(synchrotron, I0, Vrf, bunch_number):
    """
    Return a pycolleff Ring element from the synchrotron element data.

    Parameters
    ----------
    synchrotron : mbtrack2.tracking.synchrotron.Synchrotron
        Parameters description.
    I0 : float
        Beam current in [A].
    Vrf : float
        RF voltage in [V].
    bunch_number : int
        Total number of bunches filled.

    Returns
    -------
    ring : pycolleff.colleff.Ring
        pycolleff Ring object.

    """
    ring = Ring()
    ring.version = 'from_mbtrack2'
    ring.rf_freq = synchrotron.f1
    ring.mom_comp = synchrotron.ac  # momentum compaction factor
    ring.energy = synchrotron.E0  # energy [eV]
    ring.tuney = synchrotron.tune[1]  # vertical tune
    ring.tunex = synchrotron.tune[0]  # horizontal tune
    ring.chromx = synchrotron.chro[0]  # horizontal chromaticity
    ring.chromy = synchrotron.chro[1]  # vertical chromaticity
    ring.harm_num = synchrotron.h  # harmonic Number
    ring.num_bun = bunch_number  # number of bunches filled
    ring.total_current = I0  # total current [A]
    ring.sync_tune = synchrotron.synchrotron_tune(Vrf)  # synchrotron tune
    ring.espread = synchrotron.sigma_delta
    ring.bunlen = synchrotron.sigma_0 * c  # [m]
    ring.damptx = synchrotron.tau[0]  # [s]
    ring.dampty = synchrotron.tau[1]  # [s]
    ring.dampte = synchrotron.tau[2]  # [s]
    ring.en_lost_rad = synchrotron.U0  # [eV]
    ring.gap_voltage = Vrf  # [V]

    return ring

def cavityresonator_to_pycolleff(cavres, Impedance=True):
    """
    Convenience method to export a CavityResonator to pycolleff.

    Parameters
    ----------
    cavres : mbtrack2.tracking.rf.CavityResonator
        CavityResonator object.
    Impedance : bool, optional
        If True, export as impedance (i.e. ImpedanceSource.Methods.ImpedanceDFT).
        If False, export as wake (i.e. ImpedanceSource.Methods.Wake).
        Default is True.

    Returns
    -------
    cav : pycolleff ImpedanceSource object

    """
    cav = ImpedanceSource()
    cav.harm_rf = cavres.m
    cav.Q = cavres.QL
    RoverQ = cavres.RL / cavres.QL
    cav.shunt_impedance = RoverQ * cav.Q
    cav.ang_freq_rf = cavres.ring.omega1
    cav.ang_freq = cav.harm_rf * cav.ang_freq_rf
    cav.detune_w = 2 * np.pi * cavres.detune
    if cavres.Vg != 0:
        cav.active_passive = ImpedanceSource.ActivePassive.Active
        if Impedance:
            raise NotImplementedError()
        else:
            cav.calc_method = ImpedanceSource.Methods.Wake
    else:
        cav.active_passive = ImpedanceSource.ActivePassive.Passive
        if Impedance:
            cav.calc_method = ImpedanceSource.Methods.ImpedanceDFT
        else:
            cav.calc_method = ImpedanceSource.Methods.Wake
    return cav

def impedance_to_pycolleff(impedance):
    """
    Convenience method to export impedance to pycolleff.
    Only implemented for longitudinal impedance.

    Parameters
    ----------
    impedance : mbtrack2.impedance.wakefield.Impedance
        Impedance object.

    Returns
    -------
    imp : pycolleff.longitudinal_equilibrium.ImpedanceSource
        pycolleff ImpedanceSource object
    """
    imp = ImpedanceSource()
    if impedance.component_type == "long":
        double_sided_impedance(impedance)
        # Negative imag sign convention !
        imp.zl_table = impedance.data["real"].to_numpy(
        ) - 1j * impedance.data["imag"].to_numpy()
        imp.ang_freq_table = impedance.data.index.to_numpy() * 2 * np.pi
    else:
        raise NotImplementedError()

    imp.calc_method = ImpedanceSource.Methods.ImpedanceDFT
    imp.active_passive = ImpedanceSource.ActivePassive.Passive

    return imp
