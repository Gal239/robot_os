/**
 * Echo Robotics Lab - Lava Lamp Particle Effect
 * Continuous rising LED particles with smooth random pulsing
 */

class LavaLamp {
    /**
     * Create lava lamp particle effect
     * @param {HTMLElement} container - Container element for particles
     * @param {number} count - Number of particles (default: 30)
     */
    static create(container, count = 30) {
        const colors = ['#00FF00', '#FF0044', '#00FFFF', '#FF8800'];
        const pulseAnimations = ['lavaPulse1', 'lavaPulse2', 'lavaPulse3', 'lavaPulse4'];

        for (let i = 0; i < count; i++) {
            const particle = document.createElement('div');
            particle.className = 'lava-particle';

            const color = colors[Math.floor(Math.random() * colors.length)];
            const size = 2 + Math.random() * 2; // 2-4px
            const left = 5 + Math.random() * 90; // 5-95%
            const duration = 15 + Math.random() * 15; // 15-30s
            const delay = Math.random() * 30; // 0-30s staggered start
            const pulseAnim = pulseAnimations[Math.floor(Math.random() * pulseAnimations.length)];
            const pulseDuration = 3 + Math.random() * 4; // 3-7s pulse cycle

            particle.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                background: ${color};
                border-radius: 50%;
                left: ${left}%;
                top: 110%;
                box-shadow: 0 0 ${size * 2}px ${color};
                opacity: 0;
                animation: riseLavaParticle ${duration}s linear ${delay}s infinite, ${pulseAnim} ${pulseDuration}s ease-in-out infinite;
                pointer-events: none;
                z-index: 0;
            `;

            container.appendChild(particle);
        }
    }
}

// Export for use in components
window.LavaLamp = LavaLamp;
