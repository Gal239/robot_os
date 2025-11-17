#!/bin/bash
# Fix NVIDIA GPU - Load kernel modules
# This script must be run with sudo

echo "=========================================="
echo "NVIDIA GPU FIX - Load Kernel Modules"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ ERROR: This script must be run with sudo"
    echo ""
    echo "Usage: sudo bash fix_nvidia_gpu.sh"
    exit 1
fi

echo "1. Current module status:"
echo "   Checking loaded NVIDIA modules..."
lsmod | grep nvidia
if [ $? -ne 0 ]; then
    echo "   ❌ No NVIDIA modules loaded"
else
    echo "   ✅ NVIDIA modules already loaded"
fi
echo ""

echo "2. Loading NVIDIA kernel modules..."
echo "   Loading nvidia..."
modprobe nvidia
if [ $? -eq 0 ]; then
    echo "   ✅ nvidia loaded"
else
    echo "   ❌ Failed to load nvidia"
    echo "   Checking dmesg for errors:"
    dmesg | grep nvidia | tail -10
    exit 1
fi

echo "   Loading nvidia-uvm..."
modprobe nvidia-uvm
if [ $? -eq 0 ]; then
    echo "   ✅ nvidia-uvm loaded"
else
    echo "   ⚠️  Failed to load nvidia-uvm (non-critical)"
fi

echo "   Loading nvidia-modeset..."
modprobe nvidia-modeset
if [ $? -eq 0 ]; then
    echo "   ✅ nvidia-modeset loaded"
else
    echo "   ⚠️  Failed to load nvidia-modeset (non-critical)"
fi

echo ""
echo "3. Verifying modules loaded:"
lsmod | grep nvidia
echo ""

echo "4. Testing nvidia-smi:"
nvidia-smi
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! NVIDIA GPU is now working!"
else
    echo ""
    echo "❌ nvidia-smi still not working. Check dmesg for errors:"
    dmesg | grep nvidia | tail -20
    exit 1
fi

echo ""
echo "=========================================="
echo "NEXT STEPS:"
echo "=========================================="
echo "1. Run the test again:"
echo "   PYTHONPATH=\$PWD python3 simulation_center/core/tests/test_3_scenarios_subprocess.py"
echo ""
echo "2. Expected results:"
echo "   - vision_rl should reach 1.8-2.2x (or higher with RTX 3090!)"
echo "   - MuJoCo will use NVIDIA GPU instead of Intel"
echo ""
echo "3. To make this permanent (load on boot):"
echo "   The modules should auto-load, but if not:"
echo "   echo 'nvidia' >> /etc/modules-load.d/nvidia.conf"
echo "   echo 'nvidia-uvm' >> /etc/modules-load.d/nvidia.conf"
echo "=========================================="
