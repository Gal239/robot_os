/**
 * Echo Robotics Lab - Status Indicator Component
 * Animated status display with dot and text
 */

class StatusIndicator {
    /**
     * Create a status indicator element
     * @param {Object} options - Configuration options
     * @param {string} options.status - Status ('ready', 'thinking', 'building', 'error', 'not_started')
     * @param {string} options.size - Size ('compact', 'normal', 'large')
     * @param {boolean} options.showIcon - Show icon (optional)
     * @returns {HTMLElement} - Status indicator element
     */
    static create(options = {}) {
        const {
            status = 'not_started',
            size = 'normal',
            showIcon = false
        } = options;

        const indicator = document.createElement('div');
        indicator.className = `status-indicator ${status}`;

        if (size !== 'normal') {
            indicator.classList.add(size);
        }

        // Dot
        const dot = document.createElement('span');
        dot.className = 'status-indicator-dot';
        indicator.appendChild(dot);

        // Icon (optional)
        if (showIcon) {
            const icon = document.createElement('span');
            icon.className = 'status-indicator-icon';
            icon.textContent = this.getIconForStatus(status);
            indicator.appendChild(icon);
        }

        // Text
        const text = document.createElement('span');
        text.className = 'status-indicator-text';
        text.textContent = this.getTextForStatus(status);
        indicator.appendChild(text);

        return indicator;
    }

    /**
     * Update status indicator
     * @param {HTMLElement} element - Status indicator element
     * @param {string} status - New status
     */
    static update(element, status) {
        if (!element) return;

        // Remove all status classes
        element.classList.remove('ready', 'thinking', 'building', 'error', 'not_started');

        // Add new status class
        element.classList.add(status);

        // Update text
        const textEl = element.querySelector('.status-indicator-text');
        if (textEl) {
            textEl.textContent = this.getTextForStatus(status);
        }

        // Update icon if present
        const iconEl = element.querySelector('.status-indicator-icon');
        if (iconEl) {
            iconEl.textContent = this.getIconForStatus(status);
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
     * Get icon for status
     */
    static getIconForStatus(status) {
        const statusIcons = {
            'ready': '✓',
            'thinking': '●',
            'building': '⚙',
            'error': '✕',
            'not_started': '○'
        };

        return statusIcons[status] || '○';
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
        container.innerHTML = '';
        container.appendChild(indicator);

        return indicator;
    }
}

// Export for use in app.js
window.StatusIndicator = StatusIndicator;
