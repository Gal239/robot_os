# SECURE BOOT BLOCKING NVIDIA GPU

## Problem
Your RTX 3090 GPU cannot be used because Secure Boot is blocking the NVIDIA kernel module.

Error: `modprobe: ERROR: could not insert 'nvidia': Key was rejected by service`

## Solutions

### Option 1: Disable Secure Boot (RECOMMENDED - Easiest)

**Steps:**
1. Reboot your computer
2. Press **DEL** or **F2** or **F12** during boot to enter BIOS/UEFI
3. Navigate to **Security** or **Boot** menu
4. Find **Secure Boot** setting
5. Change from **Enabled** to **Disabled**
6. Save and Exit (usually F10)

**After reboot:**
```bash
cd /home/gal-labs/PycharmProjects/echo_robot
sudo modprobe nvidia
nvidia-smi  # Should now work!

# Test performance
PYTHONPATH=$PWD python3 simulation_center/core/tests/test_3_scenarios_subprocess.py
```

**Expected results:**
- nvidia-smi shows RTX 3090
- vision_rl: **1.8-2.2x** (or higher!)
- rl_core: **3-5x**

---

### Option 2: Sign NVIDIA Module (Keep Secure Boot Enabled)

This is more complex but keeps Secure Boot security.

**Automatic signing (easiest):**
```bash
# Reinstall driver with signing
sudo apt-get install --reinstall nvidia-driver-580
sudo reboot
```

During reboot, you'll see a blue **MOK Management** screen:
1. Select **Enroll MOK**
2. Select **Continue**
3. Enter the password shown on screen
4. Reboot

**Manual signing (if automatic fails):**
```bash
# Install signing tools
sudo apt-get install mokutil

# Check if MOK key exists
ls -la /var/lib/shim-signed/mok/

# If MOK key exists, sign the module
for module in nvidia nvidia-modeset nvidia-drm nvidia-uvm; do
    sudo /lib/modules/$(uname -r)/build/scripts/sign-file sha256 \
        /var/lib/shim-signed/mok/MOK.priv \
        /var/lib/shim-signed/mok/MOK.der \
        /lib/modules/$(uname -r)/updates/dkms/${module}.ko
done

sudo reboot
```

---

### Option 3: Check Current Status

**Before any changes, verify the issue:**
```bash
# Check Secure Boot status
mokutil --sb-state
# Output: SecureBoot enabled

# Try loading module
sudo modprobe nvidia
# Error: Key was rejected by service

# Check if module file exists
ls -la /lib/modules/$(uname -r)/updates/dkms/nvidia.ko
# Output: File exists

# This confirms Secure Boot is blocking it
```

---

## Why This Happens

- **Secure Boot** requires all kernel modules to be signed with a trusted key
- **NVIDIA driver** is built by DKMS (Dynamic Kernel Module Support)
- DKMS-built modules are **not signed** by default
- Linux kernel rejects unsigned modules when Secure Boot is on

## Recommendation

For development/ML work, **Option 1 (disable Secure Boot)** is recommended:
- Immediate fix (no reboot into special menus)
- No complexity with key signing
- Common practice for ML/GPU workstations
- You can re-enable it later if needed

For production systems where security is critical, use **Option 2**.

---

## After Fix - Test Performance

Once NVIDIA GPU is working:

```bash
cd /home/gal-labs/PycharmProjects/echo_robot

# Verify GPU works
nvidia-smi

# Run diagnostic
PYTHONPATH=$PWD python3 simulation_center/core/tests/diagnose_gpu.py

# Test all 3 scenarios
PYTHONPATH=$PWD python3 simulation_center/core/tests/test_3_scenarios_subprocess.py
```

**Expected performance with RTX 3090:**
- **rl_core**: 3-5x real-time (currently 2.68x with Intel GPU)
- **vision_rl**: 1.8-2.2x (currently 1.54x with Intel GPU)
- May even reach **2.5-3x** with RTX 3090's power!

---

## Summary

**Current State:**
- ✅ RTX 3090 hardware present
- ✅ NVIDIA driver 580 installed
- ❌ Secure Boot blocking kernel module
- ⚠️  Falling back to Intel iGPU (slower)

**After Fix:**
- ✅ RTX 3090 fully operational
- ✅ nvidia-smi working
- ✅ MuJoCo using RTX 3090
- ✅ 1.8-2.2x+ performance for vision_rl
