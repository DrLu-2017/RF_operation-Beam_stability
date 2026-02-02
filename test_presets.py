#!/usr/bin/env python3
"""
Test script to verify all presets load correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.presets import get_preset, get_preset_names

def test_preset(preset_name):
    """Test loading a single preset"""
    try:
        preset = get_preset(preset_name)
        
        # Check ring parameters
        ring = preset.get("ring", {})
        assert "energy" in ring, f"{preset_name}: Missing 'energy' in ring"
        assert "harmonic_number" in ring, f"{preset_name}: Missing 'harmonic_number' in ring"
        
        # Check main cavity parameters
        mc = preset.get("main_cavity", {})
        assert "voltage" in mc, f"{preset_name}: Missing 'voltage' in main_cavity"
        assert "frequency" in mc, f"{preset_name}: Missing 'frequency' in main_cavity"
        assert "Q0" in mc or "Q" in mc, f"{preset_name}: Missing 'Q0' or 'Q' in main_cavity"
        
        # Check harmonic cavity parameters
        hc = preset.get("harmonic_cavity", {})
        assert "voltage" in hc, f"{preset_name}: Missing 'voltage' in harmonic_cavity"
        assert "frequency" in hc, f"{preset_name}: Missing 'frequency' in harmonic_cavity"
        
        print(f"✓ {preset_name:30s} - OK")
        print(f"  Energy: {ring.get('energy')} GeV, h={ring.get('harmonic_number')}, U0={ring.get('energy_loss_per_turn', 0)*1e6:.1f} keV")
        return True
        
    except Exception as e:
        print(f"✗ {preset_name:30s} - FAILED: {e}")
        return False

def main():
    print("=" * 70)
    print("Testing All Presets")
    print("=" * 70)
    print()
    
    preset_names = get_preset_names()
    print(f"Found {len(preset_names)} presets to test\n")
    
    results = {}
    for name in preset_names:
        results[name] = test_preset(name)
        print()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if failed == 0:
        print("\n✅ All presets loaded successfully!")
        return 0
    else:
        print("\n❌ Some presets failed to load")
        return 1

if __name__ == "__main__":
    sys.exit(main())
