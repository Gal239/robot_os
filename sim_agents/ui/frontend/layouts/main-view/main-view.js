/**
 * Main View Layout
 * =================
 * Assembles the main content area with navigation and canvas
 *
 * Structure:
 * - main-content (container)
 *   - main-view-nav (tab navigation)
 *   - main-view-canvas (content area with tabs OR empty state)
 *
 * Components used:
 * - led-toggle (CSS-only navigation)
 * - script-viewer
 * - camera-gallery
 * - scene-inspector
 * - metalog-viewer
 */

class MainView {
    /**
     * Render the complete main view
     * @param {string} activeTab - Currently active tab ('script', 'cameras', 'data', 'metalog')
     * @param {boolean} hasContent - Whether any content exists (script/cameras/data)
     * @returns {string} HTML string
     */
    static render(activeTab = 'script', hasContent = false) {
        return `
            <main class="main-content">
                <!-- Main View Navigation -->
                <div class="main-view-nav">
                    ${this.renderNavigation(activeTab)}
                </div>

                <!-- Main View Canvas -->
                <div class="main-view-canvas">
                    ${hasContent ? this.renderTabs(activeTab) : this.renderEmptyState()}
                </div>
            </main>
        `;
    }

    /**
     * Render navigation with LED toggles
     * @param {string} activeTab - Currently active tab
     * @returns {string} HTML string
     */
    static renderNavigation(activeTab) {
        const tabs = [
            { id: 'script', label: 'Script' },
            { id: 'cameras', label: 'Cameras' },
            { id: 'data', label: 'Scene Data' },
            { id: 'metalog', label: 'Debug' }
        ];

        return `
            <nav class="led-toggle-nav">
                ${tabs.map(tab => this.renderTabButton(tab, activeTab)).join('')}
            </nav>
        `;
    }

    /**
     * Render tabs content area
     * @param {string} activeTab - Currently active tab
     * @returns {string} HTML string
     */
    static renderTabs(activeTab) {
        const tabs = [
            { id: 'script', containerId: 'script-viewer-container' },
            { id: 'cameras', containerId: 'camera-gallery-container' },
            { id: 'data', containerId: 'scene-inspector-container' },
            { id: 'metalog', containerId: 'metalog-viewer-container' }
        ];

        return `
            <div class="tab-content">
                ${tabs.map(tab => this.renderTabPane(tab, activeTab)).join('')}
            </div>
        `;
    }

    /**
     * Render empty state when no content exists
     * @returns {string} HTML string
     */
    static renderEmptyState() {
        return `
            <div class="main-view-empty">
                <i data-feather="zap" class="main-view-empty-icon"></i>
                <div class="main-view-empty-title">Ready to Build</div>
                <div class="main-view-empty-text">
                    Start a conversation with Echo to create your robot scene
                </div>
            </div>
        `;
    }

    /**
     * Render a single tab button
     * @param {Object} tab - Tab configuration
     * @param {string} activeTab - Currently active tab
     * @returns {string} HTML string
     */
    static renderTabButton(tab, activeTab) {
        const isActive = tab.id === activeTab;
        return `
            <button class="led-toggle ${isActive ? 'active' : ''}" data-tab="${tab.id}">
                <span class="led-indicator"></span>
                <span class="led-label">${tab.label}</span>
            </button>
        `;
    }

    /**
     * Render a single tab pane
     * @param {Object} tab - Tab configuration
     * @param {string} activeTab - Currently active tab
     * @returns {string} HTML string
     */
    static renderTabPane(tab, activeTab) {
        const isActive = tab.id === activeTab;
        return `
            <div id="tab-${tab.id}" class="tab-pane ${isActive ? 'active' : ''}">
                <div id="${tab.containerId}"></div>
            </div>
        `;
    }

    /**
     * Update the main view with new state
     * @param {string} containerId - Container element ID
     * @param {string} activeTab - Currently active tab
     * @param {boolean} hasContent - Whether content exists
     */
    static update(containerId, activeTab, hasContent = false) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[MainView] Container '${containerId}' not found`);
            return;
        }

        container.innerHTML = this.render(activeTab, hasContent);

        // Add lava lamp to empty state
        if (!hasContent) {
            const emptyState = container.querySelector('.main-view-empty');
            if (emptyState && typeof LavaLamp !== 'undefined') {
                LavaLamp.create(emptyState, 50);
            }
        }

        // Replace feather icons
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    }

    /**
     * Switch to a different tab (updates UI only, doesn't re-render)
     * @param {string} tabName - Tab identifier
     */
    static switchTab(tabName) {
        console.log('[MainView] Switching to tab:', tabName);

        // Update tab buttons
        document.querySelectorAll('.led-toggle').forEach(button => {
            if (button.dataset.tab === tabName) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });

        // Update tab panes
        document.querySelectorAll('.tab-pane').forEach(pane => {
            if (pane.id === `tab-${tabName}`) {
                pane.classList.add('active');
            } else {
                pane.classList.remove('active');
            }
        });
    }
}

// Export for window access
window.MainView = MainView;
