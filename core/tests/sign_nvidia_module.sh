#!/bin/bash
# Sign NVIDIA kernel modules for Secure Boot
# This allows NVIDIA GPU to work while keeping Secure Boot enabled

echo "=========================================="
echo "Sign NVIDIA Modules for Secure Boot"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ ERROR: This script must be run with sudo"
    echo ""
    echo "Usage: sudo bash sign_nvidia_module.sh"
    exit 1
fi

echo "1. Checking Secure Boot status..."
mokutil --sb-state
echo ""

echo "2. Checking if MOK keys exist..."
if [ -f /var/lib/shim-signed/mok/MOK.priv ] && [ -f /var/lib/shim-signed/mok/MOK.der ]; then
    echo "   ✅ MOK keys found"
else
    echo "   ⚠️  MOK keys not found. Need to generate them."
    echo ""
    echo "   Generating MOK keys..."

    # Create directory
    mkdir -p /var/lib/shim-signed/mok
    cd /var/lib/shim-signed/mok

    # Generate key
    openssl req -new -x509 -newkey rsa:2048 -keyout MOK.priv -outform DER -out MOK.der -days 36500 -subj "/CN=Machine Owner Key/" -nodes

    if [ $? -eq 0 ]; then
        echo "   ✅ MOK keys generated"
        chmod 600 MOK.priv
        chmod 644 MOK.der
    else
        echo "   ❌ Failed to generate MOK keys"
        exit 1
    fi
fi

echo ""
echo "3. Enrolling MOK key (if not already enrolled)..."
mokutil --import /var/lib/shim-signed/mok/MOK.der
if [ $? -eq 0 ]; then
    echo "   ✅ MOK enrollment initiated"
    echo "   ⚠️  IMPORTANT: On next reboot, you'll see a blue MOK Management screen"
    echo "      1. Select 'Enroll MOK'"
    echo "      2. Select 'Continue'"
    echo "      3. Select 'Yes'"
    echo "      4. Enter the password you'll be prompted to create"
    echo "      5. Reboot"
else
    echo "   ℹ️  MOK may already be enrolled (this is OK)"
fi

echo ""
echo "4. Signing NVIDIA kernel modules..."

KERNEL_VERSION=$(uname -r)
MODULE_DIR="/lib/modules/$KERNEL_VERSION/updates/dkms"

if [ ! -d "$MODULE_DIR" ]; then
    echo "   ❌ NVIDIA module directory not found: $MODULE_DIR"
    exit 1
fi

SIGN_SCRIPT="/lib/modules/$KERNEL_VERSION/build/scripts/sign-file"

if [ ! -f "$SIGN_SCRIPT" ]; then
    echo "   ⚠️  Kernel signing script not found. Installing kernel headers..."
    apt-get install -y linux-headers-$KERNEL_VERSION

    if [ ! -f "$SIGN_SCRIPT" ]; then
        echo "   ❌ Still can't find signing script"
        exit 1
    fi
fi

# Sign each NVIDIA module
for module in nvidia nvidia-modeset nvidia-drm nvidia-uvm nvidia-peermem; do
    MODULE_FILE="$MODULE_DIR/${module}.ko"

    if [ -f "$MODULE_FILE" ]; then
        echo "   Signing $module.ko..."
        $SIGN_SCRIPT sha256 /var/lib/shim-signed/mok/MOK.priv /var/lib/shim-signed/mok/MOK.der "$MODULE_FILE"

        if [ $? -eq 0 ]; then
            echo "   ✅ Signed $module.ko"
        else
            echo "   ❌ Failed to sign $module.ko"
        fi
    else
        echo "   ⚠️  $module.ko not found (may be optional)"
    fi
done

echo ""
echo "5. Updating initramfs..."
update-initramfs -u -k $KERNEL_VERSION

echo ""
echo "=========================================="
echo "NEXT STEPS:"
echo "=========================================="
echo ""
echo "1. REBOOT your system:"
echo "   sudo reboot"
echo ""
echo "2. During boot, you'll see a BLUE SCREEN (MOK Management):"
echo "   - Select: Enroll MOK"
echo "   - Select: Continue"
echo "   - Select: Yes"
echo "   - Enter the password when prompted"
echo "   - Select: Reboot"
echo ""
echo "3. After reboot, test NVIDIA:"
echo "   nvidia-smi"
echo ""
echo "4. If nvidia-smi works, test MuJoCo performance:"
echo "   cd /home/gal-labs/PycharmProjects/echo_robot"
echo "   PYTHONPATH=\$PWD python3 simulation_center/core/tests/test_3_scenarios_subprocess.py"
echo ""
echo "Expected results:"
echo "   - vision_rl: 1.8-2.2x (or higher!)"
echo "   - Secure Boot: Still enabled ✅"
echo "   - Battlefield: Still works ✅"
echo ""
echo "=========================================="
