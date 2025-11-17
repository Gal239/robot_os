/**
 * Echo Robotics Lab - Robot Icon Component
 * Premium robot with mouse-tracking eyes
 */

class RobotIcon {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.render();
        this.setupEyeTracking();
        this.startBlinking();
    }

    render() {
        this.container.innerHTML = `
            <div class="robot-icon">
                <div class="robot-head">
                    <div class="robot-eyes">
                        <div class="robot-eye robot-eye-left"></div>
                        <div class="robot-eye robot-eye-right"></div>
                    </div>
                </div>
                <span class="robot-title">Echo Robot</span>
            </div>
        `;

        this.robotHead = this.container.querySelector('.robot-head');
        this.leftEye = this.container.querySelector('.robot-eye-left');
        this.rightEye = this.container.querySelector('.robot-eye-right');
    }

    setupEyeTracking() {
        document.addEventListener('mousemove', (e) => {
            this.trackMouse(e.clientX, e.clientY);
        });
    }

    trackMouse(mouseX, mouseY) {
        if (!this.robotHead || !this.leftEye || !this.rightEye) return;

        // Get robot head position
        const headRect = this.robotHead.getBoundingClientRect();
        const headCenterX = headRect.left + headRect.width / 2;
        const headCenterY = headRect.top + headRect.height / 2;

        // Calculate angle from robot head to mouse
        const angle = Math.atan2(mouseY - headCenterY, mouseX - headCenterX);

        // Calculate eye movement (max 8px radius for more sensitivity)
        const maxMove = 8;
        const moveX = Math.cos(angle) * maxMove;
        const moveY = Math.sin(angle) * maxMove;

        // Apply transform to both eyes
        this.leftEye.style.transform = `translate(${moveX}px, ${moveY}px)`;
        this.rightEye.style.transform = `translate(${moveX}px, ${moveY}px)`;
    }

    startBlinking() {
        const blink = () => {
            if (!this.leftEye || !this.rightEye) return;

            // Blink animation
            this.leftEye.classList.add('blinking');
            this.rightEye.classList.add('blinking');

            setTimeout(() => {
                this.leftEye.classList.remove('blinking');
                this.rightEye.classList.remove('blinking');
            }, 150);

            // Schedule next blink (3-6 seconds)
            const nextBlinkTime = 3000 + Math.random() * 3000;
            setTimeout(blink, nextBlinkTime);
        };

        // Start first blink after 2 seconds
        setTimeout(blink, 2000);
    }
}

// Export for use by AppHeader
window.RobotIcon = RobotIcon;
