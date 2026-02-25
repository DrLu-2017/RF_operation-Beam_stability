import numpy as np

# ============================================================
# 1. Storage Ring Parameters
# ============================================================

E0_GeV = 2.75
f0 = 352.2e6                 # Hz
h = 416
omega0 = 2 * np.pi * f0

I_min = 0.01e-3              # A
I_max = 500e-3               # A

# Energy loss (Phase 2)
U0 = (487e3 + 359e3)          # eV
U0 *= 1.602e-19              # J


# ============================================================
# 2. Fundamental Cavities
# ============================================================

ncav = 4
Vf_total = 1700e3            # V
Vf_cav = Vf_total / ncav

Q0 = 35000
Qext = 6364
beta = Q0 / Qext
Ql = Q0 / (1 + beta)

Rsh = 5e6                    # Ohm

tau_cav = 2 * Ql / omega0   # s


# ============================================================
# 3. Harmonic Cavities
# ============================================================

nh = 4
nhcav = 2

fh0 = nh * f0
omega_h0 = 2 * np.pi * fh0

Qh0 = 31000
Rh = 0.92e6                  # Ohm

# Flat potential theoretical voltage (simplified)
Vh_opt = 2 * U0 / (nh * Vf_total)

# Actually adopted
Vh = 400e3                   # V


# ============================================================
# 4. Harmonic Cavity Detuning
# ============================================================

delta_fh = (
    1 / 4 * nhcav * Vh / (I_max * Rh) * Qh0 / fh0
)

# Angular frequency
omega_hc = 2 * np.pi * (fh0 + delta_fh)


# ============================================================
# 5. Harmonic Cavity Admittance & Voltage
# ============================================================

Yh = (1 / Rh) * (
    1 + 1j * Qh0 * (omega_hc / omega_h0 - omega_h0 / omega_hc)
)

# Induced voltage per harmonic cavity
Vh_cav = I_max / (2 * np.abs(Yh))

# Total harmonic voltage
Vh_total = nhcav * Vh_cav


# ============================================================
# 6. Harmonic Cavity Power
# ============================================================

Ph_diss = Vh_cav**2 / (2 * Rh)
PhT_diss = nhcav * Ph_diss

Uh0 = PhT_diss / I_max       # J
UT0 = U0 + Uh0               # J


# ============================================================
# 7. Synchronous Phase (Fundamental)
# ============================================================

phi_s = np.arcsin(UT0 / Vf_total)   # rad
phi_s_deg = np.degrees(phi_s)


# ============================================================
# 8. Fundamental Cavity Detuning (I_max)
# ============================================================

delta_f = (
    Vf_cav * np.sin(phi_s)
    / (Q0 * Rsh * I_max)
    * f0
)


# ============================================================
# 9. Power (Fundamental)
# ============================================================

Pbeam = UT0 * I_max
Pf_diss = Vf_cav**2 / (2 * Rsh)

# ============================================================
# 10. Output Results
# ============================================================

print("========== RF Cavity Summary ==========")
print(f"I_max                = {I_max*1e3:.1f} mA")
print(f"Vf_total             = {Vf_total/1e3:.1f} kV")
print(f"Vh_total             = {Vh_total/1e3:.2f} kV")
print(f"delta_fh             = {delta_fh/1e3:.2f} kHz")
print(f"delta_f (fund.)      = {delta_f/1e3:.2f} kHz")
print(f"phi_s                = {phi_s_deg:.2f} deg")
print(f"Pbeam                = {Pbeam/1e3:.2f} kW")
print(f"Pf_diss (per cav)    = {Pf_diss/1e3:.2f} kW")
print(f"PhT_diss (total)     = {PhT_diss/1e3:.2f} kW")
print("=======================================")

# ============================================================
# 11. RF Feedback (Fundamental Cavities)
# ============================================================

# -------- Feedback parameters --------
gain = 1.3                 # RF Feedback gain
delay = 1.999e-6           # s (≈ 704 RF periods)

# -------- Bare cavity bandwidth --------
BP = f0 / Ql               # Hz
BP_fb = (1 + gain) * BP    # Hz

# -------- Maximum stable gain --------
g_max = delay * omega0**2 / (2 * np.pi * Ql)

# -------- Frequency axis --------
f = np.linspace(f0 - 200e3, f0 + 200e3, 2001)
omega = 2 * np.pi * f

# -------- Bare cavity impedance --------
Zc = Rsh / (
    1 + 1j * Ql * (f / f0 - f0 / f)
)

# -------- Feedback transfer function --------
Gc = gain * np.exp(-1j * 2 * np.pi * f * delay)

# -------- Closed-loop cavity impedance --------
Zcfb = Zc / (1 + Zc * Gc)

# ============================================================
# 12. Diagnostics (PDF-style quantities)
# ============================================================

# Equivalent shunt impedance at resonance
idx_f0 = np.argmin(np.abs(f - f0))

R_eq_no_fb = np.real(Zc[idx_f0])
R_eq_fb = np.real(Zcfb[idx_f0])

print("========== RF Feedback ==========")
print(f"Gain                  = {gain:.2f}")
print(f"Delay                 = {delay*1e6:.3f} us")
print(f"Bare bandwidth        = {BP/1e3:.2f} kHz")
print(f"Feedback bandwidth    = {BP_fb/1e3:.2f} kHz")
print(f"Max stable gain       = {g_max:.2f}")
print(f"Req / no feedback     = {R_eq_no_fb/1e6:.2f} MOhm")
print(f"Req / with feedback  = {R_eq_fb/1e6:.2f} MOhm")
print("=================================")
