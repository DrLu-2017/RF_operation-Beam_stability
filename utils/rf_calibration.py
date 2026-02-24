import numpy as np

class RFVoltageCalibrator:
    """
    RF Voltage Calibrator based on Beam Synchrotron Frequency Measurement.
    
    This class implements the reverse calculation of cavity voltage from measured 
    synchrotron frequency fs, and provides methods for calculating theoretical 
    fs and calibration factors.
    """
    def __init__(self, f0=352.197e6, h=416, alpha=4.24e-4, E0=2.7391e9, U0=932e3):
        """
        Initialize with machine physical constants (SI units).
        
        :param f0: RF Frequency (Hz)
        :param h: Harmonic number
        :param alpha: Momentum compaction factor
        :param E0: Beam energy (eV)
        :param U0: Energy loss per turn (V)
        """
        self.f0 = f0                 # RF 频率 (Hz)
        self.h = h                    # 谐波数
        self.alpha = alpha            # 动量压缩因子
        self.E0 = E0                  # 电子束流能量 (eV)
        self.U0 = U0                  # 单圈能量损失 (V)
        self.f_rev = self.f0 / self.h # 周转频率

    def calculate_theoretical_fs(self, v_total_v):
        """根据给定的总电压计算理论同步频率 fs"""
        if v_total_v <= self.U0:
            raise ValueError("总电压必须大于单圈能量损失 U0。")
        sin_phi_s = self.U0 / v_total_v
        cos_phi_s = np.sqrt(max(0, 1 - sin_phi_s**2))
        term = (v_total_v * cos_phi_s * self.alpha * self.h) / (2 * np.pi * self.E0)
        return self.f_rev * np.sqrt(term)

    def calculate_calibrated_voltage(self, fs_measured_hz):
        """根据实测频率推算实际腔体电压"""
        k = (2 * np.pi * (fs_measured_hz**2) * self.h * self.E0) / ((self.f0**2) * self.alpha)
        return np.sqrt(k**2 + self.U0**2)

    def get_calibration_factor(self, v_measured_sum_v, fs_measured_hz):
        """计算校准系数 (Gain)"""
        v_actual = self.calculate_calibrated_voltage(fs_measured_hz)
        return v_actual / v_measured_sum_v

class DecayPowerCalibrator:
    """
    RF Voltage Calibration based on RF Decay and Forward Power.
    Commonly used for Pulsed SRF linear accelerators (DESY Method).
    """
    def __init__(self, f0=1.3e9, r_over_q=1036.0):
        """
        :param f0: Resonance frequency (Hz)
        :param r_over_q: Geometric shunt impedance (Ohms, Linac definition V^2/P)
        """
        self.f0 = f0
        self.roq = r_over_q

    def calculate_ql(self, tau_us):
        """
        Calculate Loaded Q from decay time constant tau.
        :param tau_us: Decay time constant (micro-seconds)
        :return: QL
        """
        omega0 = 2 * np.pi * self.f0
        tau = tau_us * 1e-6
        return (omega0 * tau) / 2.0

    def calculate_voltage(self, tau_us, p_for_kw):
        """
        Calculate cavity voltage using QL and Forward Power.
        Formula: V_cav = 2 * sqrt( (R/Q) * Q_L * P_for )
        
        :param tau_us: Decay time constant (us)
        :param p_for_kw: Incident forward power (kW)
        :return: V_phys (Volts)
        """
        ql = self.calculate_ql(tau_us)
        p_for = p_for_kw * 1000.0
        v_cav = 2.0 * np.sqrt(self.roq * ql * p_for)
        return v_cav

    def get_calibration_constant(self, tau_us, p_for_kw, adc_raw):
        """
        Calculate the calibration factor Kt = V_phys / A_raw.
        
        :param tau_us: Decay time constant (us)
        :param p_for_kw: Incident forward power (kW)
        :param adc_raw: Voltage reading from Probe ADC
        """
        v_phys = self.calculate_voltage(tau_us, p_for_kw)
        return v_phys / adc_raw
