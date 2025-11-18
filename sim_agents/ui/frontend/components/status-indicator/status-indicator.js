/**
 * Status Indicator Component
 * Shows current system status with LED neon effects
 */

class StatusIndicator {
    /**
     * Create a status indicator element
     * @param {Object} options - Configuration options
     * @param {string} options.status - Status ('ready', 'thinking', 'building', 'error', 'not_started')
     * @param {string} options.id - Element ID
     * @returns {HTMLElement} - Status indicator element
     */
    static create(options = {}) {
        const {
            status = 'not_started',
            id = 'status-indicator'
        } = options;

        const indicator = document.createElement('div');
        indicator.className = `status-indicator ${status}`;
        indicator.id = id;

        // Icon
        const icon = document.createElement('i');
        icon.setAttribute('data-feather', 'activity');
        icon.className = 'status-indicator-icon';
        indicator.appendChild(icon);

        // Dot
        const dot = document.createElement('span');
        dot.className = 'status-indicator-dot';
        indicator.appendChild(dot);

        // Text
        const text = document.createElement('span');
        text.className = 'status-indicator-text';
        text.textContent = this.getTextForStatus(status);
        indicator.appendChild(text);

        return indicator;
    }

    /**
     * Update status indicator
     * @param {string} elementId - Status indicator element ID
     * @param {string} status - New status
     */
    static update(elementId, status) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.error(`[StatusIndicator] Element ${elementId} not found`);
            return;
        }

        // Remove all status classes
        element.classList.remove('ready', 'thinking', 'building', 'error', 'not_started');

        // Add new status class
        element.classList.add(status);

        // Update text
        const textEl = element.querySelector('.status-indicator-text');
        if (textEl) {
            textEl.textContent = this.getTextForStatus(status);
        }

        // Re-initialize Feather icons
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    }

    /**
     * Get text for status
     */
    static getTextForStatus(status) {
        const statusTexts = {
            'ready': 'Ready',
            'thinking': 'Thinking',
            'building': 'Building',
            'error': 'Error',
            'not_started': 'Not Started'
        };

        return statusTexts[status] || status;
    }

    /**
     * Render status indicator in container
     * @param {string} containerId - Container element ID
     * @param {Object} options - Configuration options
     */
    static render(containerId, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[StatusIndicator] Container ${containerId} not found`);
            return null;
        }

        const indicator = this.create(options);
        container.appendChild(indicator);

        // Initialize Feather icons
        if (typeof feather !== 'undefined') {
            feather.replace();
        }

        return indicator;
    }
}

// Export for use in app.js
window.StatusIndicator = StatusIndicator;
