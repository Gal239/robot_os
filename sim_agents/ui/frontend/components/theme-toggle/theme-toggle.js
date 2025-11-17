/**
 * Echo Robotics Lab - Theme Toggle Component
 * Smooth toggle between light and PyCharm Darcula dark mode
 */

class ThemeToggle {
    constructor() {
        this.currentTheme = this.getInitialTheme();
        this.applyTheme(this.currentTheme, false); // Apply without transition on load
    }

    /**
     * Get initial theme (from localStorage or system preference)
     */
    getInitialTheme() {
        // Check localStorage first
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            return savedTheme;
        }

        // Check system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return 'dark';
        }

        return 'light'; // Default to light theme
    }

    /**
     * Apply theme to document
     * @param {string} theme - 'light' or 'dark'
     * @param {boolean} withTransition - Enable smooth transition
     */
    applyTheme(theme, withTransition = true) {
        const html = document.documentElement;

        // Add transition class for smooth color changes
        if (withTransition) {
            html.classList.add('theme-transitioning');
        }

        // Set theme attribute
        html.setAttribute('data-theme', theme);

        // Remove transition class after animation
        if (withTransition) {
            setTimeout(() => {
                html.classList.remove('theme-transitioning');
            }, 200);
        }

        this.currentTheme = theme;
        localStorage.setItem('theme', theme);
    }

    /**
     * Toggle between light and dark themes
     */
    toggle() {
        const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.applyTheme(newTheme);
        return newTheme;
    }

    /**
     * Render theme toggle button
     * @param {string} containerId - Container element ID
     */
    static render(containerId) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[ThemeToggle] Container ${containerId} not found`);
            return;
        }

        const button = document.createElement('button');
        button.className = 'theme-toggle-button';
        button.id = 'theme-toggle-btn';
        button.setAttribute('aria-label', 'Toggle theme');
        button.title = 'Toggle dark/light mode';

        // Icon (will be updated based on current theme)
        const icon = document.createElement('i');
        icon.id = 'theme-toggle-icon';
        icon.setAttribute('data-feather', window.themeToggle.currentTheme === 'dark' ? 'sun' : 'moon');

        button.appendChild(icon);
        container.appendChild(button);

        // Initialize Feather icons
        if (typeof feather !== 'undefined') {
            feather.replace();
        }

        // Add click handler
        button.addEventListener('click', () => {
            const newTheme = window.themeToggle.toggle();

            // Update icon with animation
            icon.style.transform = 'rotate(180deg)';
            setTimeout(() => {
                icon.setAttribute('data-feather', newTheme === 'dark' ? 'sun' : 'moon');
                if (typeof feather !== 'undefined') {
                    feather.replace();
                }
                icon.style.transform = 'rotate(0deg)';
            }, 150);
        });

        console.log('[ThemeToggle] Rendered');
    }

    /**
     * Listen for system theme changes
     */
    watchSystemTheme() {
        if (window.matchMedia) {
            const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
            darkModeQuery.addEventListener('change', (e) => {
                // Only auto-switch if user hasn't set a preference
                if (!localStorage.getItem('theme')) {
                    this.applyTheme(e.matches ? 'dark' : 'light');
                }
            });
        }
    }
}

// Initialize theme toggle globally
window.themeToggle = new ThemeToggle();
window.ThemeToggle = ThemeToggle;
