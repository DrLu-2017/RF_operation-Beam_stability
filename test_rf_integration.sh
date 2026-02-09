#!/bin/bash

# RF System Integration Test Script
# Quick test to verify the new Double RF System page integration

echo "🔧 RF System Integration Test"
echo "=============================="
echo ""

# Check if we're in the correct directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: Please run this script from the DRFB directory"
    exit 1
fi

echo "✓ Found streamlit_app.py"

# Check if required files exist
echo ""
echo "Checking integration files..."

if [ -f "utils/rf_calculations.py" ]; then
    echo "✓ utils/rf_calculations.py exists"
else
    echo "❌ utils/rf_calculations.py missing"
    exit 1
fi

if [ -f "pages/0_🔧_Double_RF_System.py" ]; then
    echo "✓ pages/0_🔧_Double_RF_System.py exists"
else
    echo "❌ pages/0_🔧_Double_RF_System.py missing"
    exit 1
fi

if [ -f "utils/presets.py" ]; then
    echo "✓ utils/presets.py exists"
else
    echo "❌ utils/presets.py missing"
    exit 1
fi

echo ""
echo "✅ All integration files present!"
echo ""
echo "📋 Integration Summary:"
echo "  - Created: utils/rf_calculations.py (RF physics engine)"
echo "  - Updated: utils/presets.py (added RF parameters)"
echo "  - Rewritten: pages/0_🔧_Double_RF_System.py (full integration)"
echo ""
echo "🚀 To start the application, run:"
echo "   streamlit run streamlit_app.py"
echo ""
echo "📖 For detailed documentation, see:"
echo "   .gemini/rf_system_integration_summary.md"
echo ""
