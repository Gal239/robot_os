/**
 * New Dream Button Component
 * Professional button for starting a new session
 */

class NewDreamButton {
    /**
     * Create a new dream button element
     * @param {Object} options - Configuration options
     * @param {Function} options.onClick - Click handler function
     * @returns {HTMLElement} - Button element
     */
    static create(options = {}) {
        const { onClick } = options;

        const button = document.createElement('button');
        button.className = 'new-dream-button';
        button.id = 'new-dream-btn';
        button.title = 'Start New Dream';

        // Icon
        const icon = document.createElement('i');
        icon.setAttribute('data-feather', 'plus');
        icon.className = 'new-dream-button-icon';
        button.appendChild(icon);

        // Text
        const text = document.createElement('span');
        text.className = 'new-dream-button-text';
        text.textContent = 'New Dream';
        button.appendChild(text);

        // Attach click handler
        if (onClick) {
            button.addEventListener('click', onClick);
        }

        return button;
    }

    /**
     * Render button in container
     * @param {string} containerId - Container element ID
     * @param {Object} options - Configuration options
     */
    static render(containerId, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[NewDreamButton] Container ${containerId} not found`);
            return null;
        }

        const button = this.create(options);
        container.appendChild(button);

        // Initialize Feather icons for this button
        if (typeof feather !== 'undefined') {
            feather.replace();
        }

        return button;
    }
}

// Export for use in app.js
window.NewDreamButton = NewDreamButton;
