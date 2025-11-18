/**
 * Echo Robotics Lab - Script Viewer Component
 * Professional code viewer with copy functionality
 */

class ScriptViewer {
    /**
     * Render script viewer in container
     * @param {string} containerId - Container element ID
     * @param {string} script - Python script content
     */
    static render(containerId, script = '') {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[ScriptViewer] Container ${containerId} not found`);
            return;
        }

        container.innerHTML = '';

        const viewer = document.createElement('div');
        viewer.className = 'script-viewer';

        // Content only (no header for immersive view)
        const content = document.createElement('div');
        content.className = 'script-viewer-content script-viewer-content-fullscreen';

        if (script) {
            const codeBlock = this.createCodeBlock(script);
            content.appendChild(codeBlock);
        } else {
            const emptyState = this.createEmptyState();
            content.appendChild(emptyState);
        }

        viewer.appendChild(content);
        container.appendChild(viewer);
    }

    /**
     * Create header with title and actions
     */
    static createHeader() {
        const header = document.createElement('div');
        header.className = 'script-viewer-header';

        header.innerHTML = `
            <div class="script-viewer-title">
                <i data-feather="file-text" class="script-viewer-title-icon"></i>
                <span class="script-viewer-title-text">Scene Script</span>
            </div>
            <div class="script-viewer-actions">
                <button class="script-viewer-button" id="script-copy-btn">
                    <i data-feather="copy"></i>
                    <span>Copy</span>
                </button>
            </div>
        `;

        // Initialize Feather icons
        if (typeof feather !== 'undefined') {
            setTimeout(() => feather.replace(), 0);
        }

        return header;
    }

    /**
     * Create code block with syntax highlighting and line numbers
     */
    static createCodeBlock(script) {
        const codeBlock = document.createElement('div');
        codeBlock.className = 'script-viewer-code';

        // Create pre and code elements for Prism.js
        const pre = document.createElement('pre');
        const code = document.createElement('code');
        code.className = 'language-python';

        // Set text content (Prism will handle highlighting)
        code.textContent = script;

        pre.appendChild(code);
        codeBlock.appendChild(pre);

        // Apply Prism highlighting
        if (typeof Prism !== 'undefined') {
            Prism.highlightElement(code);
        }

        return codeBlock;
    }

    /**
     * Create empty state
     */
    static createEmptyState() {
        const empty = document.createElement('div');
        empty.className = 'script-viewer-empty';

        empty.innerHTML = `
            <i data-feather="file-text" class="script-viewer-empty-icon"></i>
            <div class="script-viewer-empty-title">No script yet</div>
            <div class="script-viewer-empty-text">
                Start building your scene and the script will appear here.
            </div>
        `;

        // Add lava lamp particles
        LavaLamp.create(empty, 50);

        // Initialize Feather icons
        if (typeof feather !== 'undefined') {
            setTimeout(() => feather.replace(), 0);
        }

        return empty;
    }

    /**
     * Copy script to clipboard
     */
    static async copyToClipboard(script, viewerEl) {
        if (!script) return;

        try {
            await navigator.clipboard.writeText(script);

            // Show success notification
            const notification = document.createElement('div');
            notification.className = 'script-viewer-copy-success';
            notification.textContent = 'Copied to clipboard';

            viewerEl.style.position = 'relative';
            viewerEl.appendChild(notification);

            // Remove notification after animation
            setTimeout(() => {
                notification.remove();
            }, 2000);

            console.log('[ScriptViewer] Copied to clipboard');
        } catch (error) {
            console.error('[ScriptViewer] Copy failed:', error);
        }
    }

    /**
     * Update script content
     * @param {string} containerId - Container element ID
     * @param {string} script - New script content
     */
    static update(containerId, script) {
        this.render(containerId, script);
    }

    /**
     * Render animated script viewer (for live editing)
     * @param {string} containerId - Container element ID
     * @param {string} initialScript - Initial script to display before animating edits
     */
    static renderAnimated(containerId, initialScript = '') {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[ScriptViewer] Container ${containerId} not found`);
            return;
        }

        container.innerHTML = '';

        const viewer = document.createElement('div');
        viewer.className = 'script-viewer';

        // Header
        const header = this.createHeader();
        viewer.appendChild(header);

        // Content for animation
        const content = document.createElement('div');
        content.className = 'script-viewer-content';

        const animatedViewer = document.createElement('div');
        animatedViewer.className = 'script-viewer-animated';
        animatedViewer.id = 'script-animated-content';

        // Pre-populate with current script lines (no animation)
        if (initialScript) {
            const lines = initialScript.split('\n');
            lines.forEach((line) => {
                const lineEl = document.createElement('div');
                lineEl.className = 'code-line';
                lineEl.textContent = line;
                animatedViewer.appendChild(lineEl);
            });
        }

        content.appendChild(animatedViewer);
        viewer.appendChild(content);
        container.appendChild(viewer);
    }
}

// Export for use in app.js
window.ScriptViewer = ScriptViewer;
