"""
Quick test to verify preset scan parameter synchronization
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.presets import get_preset, get_preset_names

def test_preset_scan_params():
    """Test that all presets have scan_params defined"""
    print("Testing Preset Scan Parameter Synchronization")
    print("=" * 60)
    
    preset_names = get_preset_names()
    
    for name in preset_names:
        preset = get_preset(name)
        print(f"\n📋 Preset: {name}")
        print("-" * 60)
        
        # Check if scan_params exists
        if "scan_params" in preset:
            scan_params = preset["scan_params"]
            print(f"  ✅ Scan parameters found:")
            print(f"     • Psi Min:    {scan_params.get('psi_min', 'N/A')}°")
            print(f"     • Psi Max:    {scan_params.get('psi_max', 'N/A')}°")
            print(f"     • Psi Points: {scan_params.get('psi_points', 'N/A')}")
        else:
            print(f"  ❌ No scan_params found!")
        
        # Also show other key parameters for context
        print(f"\n  Other parameters:")
        print(f"     • Ring Energy:    {preset['ring'].get('energy', 'N/A')} GeV")
        print(f"     • Circumference:  {preset['ring'].get('circumference', 'N/A')} m")
        print(f"     • Beam Current:   {preset.get('current', 'N/A')} A")
        print(f"     • Passive HC:     {preset.get('passive_hc', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")

if __name__ == "__main__":
    test_preset_scan_params()
