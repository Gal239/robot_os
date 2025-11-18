/**
 * Metalog Viewer Component
 * =========================
 * Displays debug/metalog information with syntax highlighting
 */

class MetalogViewer {
    /**
     * Render metalog viewer
     * @param {string} containerId - Container element ID
     * @param {string} content - Metalog content
     */
    static render(containerId, content = '') {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[MetalogViewer] Container '${containerId}' not found`);
            return;
        }

        if (!content) {
            container.innerHTML = this.createEmptyState();
            // Add lava lamp particles to empty state
            const emptyState = container.querySelector('.metalog-viewer-empty');
            if (emptyState && typeof LavaLamp !== 'undefined') {
                LavaLamp.create(emptyState, 50);
            }
            return;
        }

        container.innerHTML = `
            <div class="metalog-viewer">
                <div class="metalog-viewer-content metalog-viewer-content-fullscreen">
                    ${this.createContentBlock(content)}
                </div>
            </div>
        `;
    }

    /**
     * Create content block
     * @param {string} content - Metalog content
     * @returns {string} HTML string
     */
    static createContentBlock(content) {
        return `
            <div class="metalog-viewer-code">
                <pre><code>${this.escapeHtml(content)}</code></pre>
            </div>
        `;
    }

    /**
     * Create empty state
     * @returns {string} HTML string
     */
    static createEmptyState() {
        return `
            <div class="metalog-viewer-empty">
                <i data-feather="list" class="metalog-viewer-empty-icon"></i>
                <div class="metalog-viewer-empty-title">No debug data</div>
                <div class="metalog-viewer-empty-text">
                    Debug information will appear here during execution.
                </div>
            </div>
        `;
    }

    /**
     * Update metalog viewer content
     * @param {string} containerId - Container element ID
     * @param {string} content - New metalog content
     */
    static update(containerId, content = '') {
        this.render(containerId, content);

        // Re-initialize feather icons
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    }

    /**
     * Escape HTML to prevent XSS
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Export for window access
window.MetalogViewer = MetalogViewer;
