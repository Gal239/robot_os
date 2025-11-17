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
        codeBlock.className = 'script-viewer-code script-viewer-code-highlighted';

        const pre = document.createElement('pre');

        // Strip any existing HTML tags (in case backend sends pre-highlighted)
        const cleanScript = script.replace(/<[^>]*>/g, '');

        // Split into lines and add line numbers
        const lines = cleanScript.split('\n');
        lines.forEach((line, index) => {
            const lineDiv = document.createElement('div');
            lineDiv.className = 'code-line';

            // Line number
            const lineNumber = document.createElement('span');
            lineNumber.className = 'line-number';
            lineNumber.textContent = (index + 1).toString().padStart(3, ' ');

            // Line content with syntax highlighting
            const lineContent = document.createElement('span');
            lineContent.className = 'line-content';
            lineContent.innerHTML = this.highlightPython(line);

            lineDiv.appendChild(lineNumber);
            lineDiv.appendChild(lineContent);
            pre.appendChild(lineDiv);
        });

        codeBlock.appendChild(pre);
        return codeBlock;
    }

    /**
     * Simple Python syntax highlighter
     */
    static highlightPython(code) {
        if (!code) return '';

        // Escape HTML
        let highlighted = code
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Python keywords
        const keywords = /\b(def|class|import|from|if|elif|else|for|while|return|yield|try|except|finally|with|as|pass|break|continue|raise|assert|lambda|None|True|False|and|or|not|in|is|async|await)\b/g;
        highlighted = highlighted.replace(keywords, '<span class="keyword">$1</span>');

        // Strings (single and double quotes)
        highlighted = highlighted.replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, '<span class="string">$&</span>');

        // Comments
        highlighted = highlighted.replace(/(#.*$)/g, '<span class="comment">$1</span>');

        // Numbers
        highlighted = highlighted.replace(/\b(\d+\.?\d*)\b/g, '<span class="number">$1</span>');

        // Function definitions
        highlighted = highlighted.replace(/\b(def)\s+(\w+)/g, '<span class="keyword">$1</span> <span class="function">$2</span>');

        // Function calls
        highlighted = highlighted.replace(/\b(\w+)(?=\()/g, '<span class="function">$1</span>');

        return highlighted;
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
     */
    static renderAnimated(containerId) {
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

        content.appendChild(animatedViewer);
        viewer.appendChild(content);
        container.appendChild(viewer);
    }
}

// Export for use in app.js
window.ScriptViewer = ScriptViewer;
