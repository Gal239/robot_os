/**
 * Echo Robotics Lab - App Header Component
 * Professional branded header with status indicator
 */

class AppHeader {
    constructor() {
        this.currentStatus = 'not_started';

        this.init();
    }

    /**
     * Initialize the app header
     */
    init() {
        this.render();
    }

    /**
     * Render the app header
     */
    render() {
        const container = document.getElementById('app-header-container');
        if (!container) {
            console.error('[AppHeader] Container not found');
            return;
        }

        container.innerHTML = `
            <header class="app-header">
                <!-- Left: Robot Icon -->
                <div class="app-header-left">
                    <div id="robot-icon-container"></div>
                </div>

                <!-- Center: Empty -->
                <div class="app-header-center">
                </div>

                <!-- Right: Session + Theme Toggle -->
                <div class="app-header-right">
                    <button class="app-header-session-btn" title="Session">
                        <i data-feather="layers"></i>
                    </button>
                    <div id="theme-toggle-container"></div>
                </div>
            </header>
        `;

        // Initialize robot icon
        if (typeof RobotIcon !== 'undefined') {
            new RobotIcon('robot-icon-container');
        }

        // Render theme toggle button
        if (typeof ThemeToggle !== 'undefined') {
            ThemeToggle.render('theme-toggle-container');
        }

        // Initialize Feather icons
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    }

    /**
     * Update status indicator
     * @param {string} status - 'ready', 'thinking', 'building', 'error', 'not_started'
     */
    setStatus(status) {
        this.currentStatus = status;

        const statusEl = document.getElementById('app-header-status');
        if (!statusEl) return;

        // Remove all status classes
        statusEl.className = 'app-header-status';

        // Add current status class
        statusEl.classList.add(status);

        // Update status text
        const statusText = {
            'ready': 'Ready',
            'thinking': 'Thinking',
            'building': 'Building',
            'error': 'Error',
            'not_started': 'Not Started'
        };

        const textEl = statusEl.querySelector('.app-header-status-text');
        if (textEl) {
            textEl.textContent = statusText[status] || status;
        }
    }
}

// Export for use in app.js
window.AppHeader = AppHeader;
