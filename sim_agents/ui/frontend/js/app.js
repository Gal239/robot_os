/**
 * Echo Scene Builder - Main Application
 * Professional UI controller
 */

class EchoApp {
    constructor() {
        this.messages = [];
        this.currentTab = 'script';  // Script tab is now first
        this.isEchoThinking = false;

        // Component instances
        this.appHeader = null;

        // DOM Elements
        this.elements = {
            chatMessages: document.getElementById('chat-messages'),
            userInput: document.getElementById('user-input'),
            sendButton: document.getElementById('send-button'),
            tabButtons: document.querySelectorAll('.led-toggle')
        };

        this.init();
    }

    /**
     * Initialize the application
     */
    async init() {
        console.log('[App] Initializing Echo Robotics Lab');

        // Initialize premium components
        this.initializeComponents();

        // Set up event listeners
        this.setupEventListeners();

        // Start conversation
        try {
            await API.start();
            this.updateStatus('ready');
            console.log('[App] Ready');
        } catch (error) {
            console.error('[App] Initialization error:', error);
            this.updateStatus('error');
        }
    }

    /**
     * Initialize all premium components
     */
    initializeComponents() {
        console.log('[App] Initializing components');

        // App Header
        this.appHeader = new AppHeader();

        // Script Viewer (empty initially)
        ScriptViewer.render('script-viewer-container', '');

        // Camera Gallery (empty initially)
        CameraGallery.render('camera-gallery-container', {});

        // Scene Inspector (empty initially)
        SceneInspector.render('scene-inspector-container', {});

        // Metalog Viewer (will create similar to script viewer)
        this.renderMetalogViewer('metalog-viewer-container', '');
    }

    /**
     * Render metalog viewer (simplified version of script viewer)
     */
    renderMetalogViewer(containerId, content) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="script-viewer">
                <div class="script-viewer-content script-viewer-content-fullscreen">
                    ${content ? `
                        <div class="script-viewer-code">
                            <pre><code>${content}</code></pre>
                        </div>
                    ` : `
                        <div class="script-viewer-empty">
                            <i data-feather="list" class="script-viewer-empty-icon"></i>
                            <div class="script-viewer-empty-title">No debug data</div>
                            <div class="script-viewer-empty-text">
                                Debug information will appear here during execution.
                            </div>
                        </div>
                    `}
                </div>
            </div>
        `;
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Send button
        this.elements.sendButton.addEventListener('click', () => this.sendMessage());

        // Enter key in textarea (Shift+Enter for new line)
        this.elements.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Tab switching
        this.elements.tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tab = button.dataset.tab;
                this.switchTab(tab);
            });
        });

        // Copy buttons are now handled by the components themselves
    }

    /**
     * Send a message to Echo
     */
    async sendMessage() {
        const message = this.elements.userInput.value.trim();

        if (!message || this.isEchoThinking) {
            return;
        }

        console.log('[App] Sending message');

        // Add user message to chat
        this.addMessage('user', message);
        this.elements.userInput.value = '';

        // Update status
        this.updateStatus('thinking');
        this.isEchoThinking = true;
        this.elements.sendButton.disabled = true;

        try {
            // Send to API
            const response = await API.sendMessage(message);

            // Add Echo's response
            this.addMessage('echo', response.echo_response);

            // Update content based on response type
            if (response.response_type === 'handoff') {
                // Check if there are edits to animate
                const editsResponse = await API.getEdits();
                if (editsResponse.edits && editsResponse.edits.length > 0) {
                    // Animate edits then refresh
                    await this.animateEdits(editsResponse.edits);
                }
                // Refresh content (gets final script + screenshots)
                await this.refreshContent();
            }

            this.updateStatus('ready');
        } catch (error) {
            console.error('[App] Send message error:', error);
            this.addMessage('system', `Error: ${error.message}`);
            this.updateStatus('ready');
        } finally {
            this.isEchoThinking = false;
            this.elements.sendButton.disabled = false;
            this.elements.userInput.focus();
        }
    }

    /**
     * Add a message to the chat
     * @param {string} role - 'user', 'echo', or 'system'
     * @param {string} text - Message text (supports markdown for echo messages)
     */
    addMessage(role, text) {
        this.messages.push({ role, text, timestamp: new Date() });

        // Remove empty state if present
        const emptyState = this.elements.chatMessages.querySelector('.chat-empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        // Create message bubble
        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${role}`;

        const textDiv = document.createElement('div');
        textDiv.className = 'message-content';

        // Render markdown for echo messages, plain text for user
        if (role === 'echo' && typeof marked !== 'undefined') {
            // Configure marked for safe rendering
            marked.setOptions({
                breaks: true,  // Convert \n to <br>
                gfm: true,     // GitHub Flavored Markdown
                sanitize: false
            });
            textDiv.innerHTML = marked.parse(text);
        } else {
            // For user messages, escape HTML and convert newlines
            textDiv.innerHTML = text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/\n/g, '<br>');
        }
        bubble.appendChild(textDiv);

        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = this.formatTime(new Date());
        bubble.appendChild(timestamp);

        this.elements.chatMessages.appendChild(bubble);

        // Scroll to bottom
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }

    /**
     * Update Echo's status
     * @param {string} status - 'ready', 'thinking', 'building', 'error', 'not_started'
     */
    updateStatus(status) {
        if (this.appHeader) {
            this.appHeader.setStatus(status);
        }
        console.log(`[App] Status: ${status}`);
    }

    /**
     * Switch to a different tab
     * @param {string} tabName - Tab identifier
     */
    switchTab(tabName) {
        console.log('[App] Switching to tab:', tabName);

        this.currentTab = tabName;

        // Update tab buttons
        this.elements.tabButtons.forEach(button => {
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

    /**
     * Refresh all content from backend
     */
    async refreshContent() {
        console.log('[App] Refreshing content');

        try {
            // Fetch all data in parallel
            const [screenshots, script, sceneData, metalog] = await Promise.all([
                API.getScreenshots(),
                API.getSceneScript(),
                API.getSceneData(),
                API.getMetalog()
            ]);

            // Update cameras using new component
            CameraGallery.update('camera-gallery-container', screenshots);

            // Update script using new component
            ScriptViewer.update('script-viewer-container', script);

            // Update scene data using new component
            SceneInspector.update('scene-inspector-container', sceneData);

            // Update metalog
            this.renderMetalogViewer('metalog-viewer-container', metalog);

            console.log('[App] Content refreshed');
        } catch (error) {
            console.error('[App] Refresh content error:', error);
        }
    }

    /**
     * Render camera screenshots
     * @param {Object} screenshots - Camera screenshots (base64)
     */
    renderCameras(screenshots) {
        const grid = this.elements.cameraGrid;
        grid.innerHTML = '';

        const cameraNames = Object.keys(screenshots);

        if (cameraNames.length === 0) {
            grid.innerHTML = `
                <div class="data-table-empty">
                    <div class="data-table-empty-text">
                        No cameras available. Create a scene to see camera views.
                    </div>
                </div>
            `;
            return;
        }

        cameraNames.forEach(name => {
            const card = document.createElement('div');
            card.className = 'camera-card';

            card.innerHTML = `
                <div class="camera-image-container">
                    <img class="camera-image" src="data:image/png;base64,${screenshots[name]}" alt="${name}">
                    <div class="camera-badge">${name}</div>
                </div>
                <div class="camera-info">
                    <div class="camera-name">${name}</div>
                </div>
            `;

            grid.appendChild(card);
        });
    }

    /**
     * Render scene script
     * @param {string} script - Python script
     */
    renderScript(script) {
        const container = this.elements.scriptContent;

        if (!script) {
            container.innerHTML = `
                <div class="code-viewer-empty">
                    <div class="code-viewer-empty-text">
                        No script available yet
                    </div>
                </div>
            `;
            return;
        }

        const codeBlock = document.createElement('pre');
        codeBlock.className = 'code-block';
        codeBlock.textContent = script;

        container.innerHTML = '';
        container.appendChild(codeBlock);
    }

    /**
     * Animate code edits in real-time
     * @param {Array} edits - Array of edit operations
     */
    async animateEdits(edits) {
        console.log(`[App] Animating ${edits.length} edits`);
        this.updateStatus('building');

        // Switch to script tab to show animation
        this.switchTab('script');

        // Render animated script viewer
        ScriptViewer.renderAnimated('script-viewer-container');

        const viewer = document.getElementById('script-animated-content');
        if (!viewer) {
            console.error('[App] Animated viewer not found');
            return;
        }

        const lines = []; // Track current lines

        // Apply each edit with animation
        for (const edit of edits) {
            if (edit.event !== 'edit') continue; // Skip non-edit events

            await this.applyEditAnimation(viewer, lines, edit);
            await this.wait(200); // 200ms between edits
        }

        console.log('[App] Edit animation complete');
    }

    /**
     * Apply single edit with animation
     * @param {HTMLElement} viewer - Container element
     * @param {Array} lines - Current lines array
     * @param {Object} edit - Edit operation
     */
    async applyEditAnimation(viewer, lines, edit) {
        const { op, code } = edit;

        // Handle both block-based (after_block, block) and line-based (after_line, line) formats
        // Block IDs are strings like "0", "1", "2" where block "0" = line 1
        const afterBlock = edit.after_block !== undefined ? parseInt(edit.after_block) : edit.after_line;
        const blockId = edit.block !== undefined ? parseInt(edit.block) : edit.line;

        if (op === 'insert') {
            // Insert line with fade-in animation
            const lineEl = document.createElement('div');
            lineEl.className = 'code-line code-line-insert';
            lineEl.textContent = code;

            // Insert at correct position
            // after_block = null means insert at start (index 0)
            // after_block = "0" means insert after first line (index 1)
            const insertIndex = afterBlock === null || isNaN(afterBlock) ? 0 : afterBlock + 1;

            if (insertIndex >= lines.length) {
                viewer.appendChild(lineEl);
                lines.push(lineEl);
            } else {
                viewer.insertBefore(lineEl, lines[insertIndex]);
                lines.splice(insertIndex, 0, lineEl);
            }

            // Wait for animation
            await this.wait(400);

        } else if (op === 'delete') {
            // Delete line with fade-out animation
            // Block "0" = first line (index 0), Block "1" = second line (index 1)
            const deleteIndex = blockId;
            if (deleteIndex >= 0 && deleteIndex < lines.length) {
                const lineEl = lines[deleteIndex];
                lineEl.classList.add('code-line-delete');

                // Wait for animation
                await this.wait(400);

                // Remove from DOM and array
                lineEl.remove();
                lines.splice(deleteIndex, 1);
            }

        } else if (op === 'replace') {
            // Replace line with flash animation
            // Block "0" = first line (index 0)
            const replaceIndex = blockId;
            if (replaceIndex >= 0 && replaceIndex < lines.length) {
                const lineEl = lines[replaceIndex];
                lineEl.textContent = code;
                lineEl.classList.add('code-line-replace');

                // Wait for animation
                await this.wait(400);

                // Remove animation class
                lineEl.classList.remove('code-line-replace');
            }
        }
    }

    /**
     * Wait for specified milliseconds
     * @param {number} ms - Milliseconds to wait
     */
    wait(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Render scene data
     * @param {Object} data - Scene metadata
     */
    renderSceneData(data) {
        const list = this.elements.sceneDataList;

        if (!data || Object.keys(data).length === 0) {
            list.innerHTML = `
                <div class="data-table-empty">
                    <div class="data-table-empty-text">
                        No scene data available
                    </div>
                </div>
            `;
            return;
        }

        list.innerHTML = '';

        Object.entries(data).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.className = 'data-list-item';

            item.innerHTML = `
                <div class="data-list-key">${key}</div>
                <div class="data-list-value">${JSON.stringify(value)}</div>
            `;

            list.appendChild(item);
        });
    }

    /**
     * Render metalog
     * @param {string} metalog - Conversation context
     */
    renderMetalog(metalog) {
        const container = this.elements.metalogContent;

        if (!metalog) {
            container.innerHTML = `
                <div class="code-viewer-empty">
                    <div class="code-viewer-empty-text">
                        No metalog available yet
                    </div>
                </div>
            `;
            return;
        }

        const codeBlock = document.createElement('pre');
        codeBlock.className = 'code-block';
        codeBlock.textContent = metalog;

        container.innerHTML = '';
        container.appendChild(codeBlock);
    }

    /**
     * Copy script to clipboard
     */
    async copyScript() {
        try {
            const script = await API.getSceneScript();
            await navigator.clipboard.writeText(script);
            console.log('[App] Script copied to clipboard');
        } catch (error) {
            console.error('[App] Copy script error:', error);
        }
    }

    /**
     * Copy metalog to clipboard
     */
    async copyMetalog() {
        try {
            const metalog = await API.getMetalog();
            await navigator.clipboard.writeText(metalog);
            console.log('[App] Metalog copied to clipboard');
        } catch (error) {
            console.error('[App] Copy metalog error:', error);
        }
    }

    /**
     * Format time for display
     * @param {Date} date
     */
    formatTime(date) {
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new EchoApp();
});
