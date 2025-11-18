/**
 * Scene Maker Page
 * ==================
 * Assembles the complete scene maker page using layouts
 *
 * Uses layouts:
 * - app-header (top navigation bar)
 * - chat-sidebar (left sidebar with chat)
 * - main-view (main content area with tabs)
 */

/**
 * SceneMaker - Page structure renderer
 * Renders the initial HTML structure only
 */
class SceneMaker {
    /**
     * Render the scene maker page structure
     * @param {string} containerId - Container element ID
     */
    static render(containerId) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[SceneMaker] Container '${containerId}' not found`);
            return;
        }

        // Render page structure with 3 layout containers
        container.innerHTML = `
            <div id="app">
                <!-- App Header Container -->
                <div id="app-header-container"></div>

                <!-- Main Layout (contains chat-sidebar and main-view as siblings) -->
                <div class="main-layout">
                    <!-- Chat Sidebar Container (rendered by ChatSidebar layout) -->
                    <div id="chat-sidebar-container"></div>

                    <!-- Main View Container (rendered by MainView layout) -->
                    <div id="main-view-container"></div>
                </div>
            </div>
        `;

        console.log('[SceneMaker] Page structure rendered');
    }
}

/**
 * SceneMakerPage - Page controller
 * Manages state and coordinates layouts
 */
class SceneMakerPage {
    constructor() {
        this.messages = [];
        this.currentTab = 'script';
        this.isEchoThinking = false;
        this.currentStatus = 'not_started';

        // Component instances
        this.appHeader = null;

        // DOM Elements (set after render)
        this.elements = {
            userInput: null,
            sendButton: null
        };

        this.init();
    }

    /**
     * Initialize the application
     */
    async init() {
        console.log('[SceneMakerPage] Initializing Echo Robotics Lab');

        // Initialize app header
        this.initializeAppHeader();

        // Render layouts
        this.renderLayouts();

        // Set up event listeners
        this.setupEventListeners();

        // Start conversation
        try {
            await API.start();

            // Load conversation history
            const conversation = await API.getConversation();
            if (conversation && conversation.length > 0) {
                console.log(`[SceneMakerPage] Restoring ${conversation.length} messages`);

                conversation.forEach(msg => {
                    this.messages.push({
                        role: msg.role,
                        message: msg.message,
                        timestamp: new Date().toISOString()
                    });
                });

                // Update chat sidebar with messages
                ChatSidebar.update('chat-sidebar-container', this.messages, this.currentStatus);
                this.setupEventListeners();
            }

            this.updateStatus('ready');
            console.log('[SceneMakerPage] Ready');
        } catch (error) {
            console.error('[SceneMakerPage] Initialization error:', error);
            this.updateStatus('error');
        }
    }

    /**
     * Initialize app header
     */
    initializeAppHeader() {
        this.appHeader = new AppHeader();
    }

    /**
     * Render all layouts
     */
    renderLayouts() {
        // Render main view
        MainView.update('main-view-container', this.currentTab, false);

        // Initialize all tab components (empty)
        ScriptViewer.render('script-viewer-container', '');
        CameraGallery.render('camera-gallery-container', {});
        SceneInspector.render('scene-inspector-container', {});
        MetalogViewer.render('metalog-viewer-container', '');

        // Render chat sidebar
        ChatSidebar.update('chat-sidebar-container', this.messages, this.currentStatus);

        // Align LED nav with chat header
        this.alignNavHeights();
    }

    /**
     * Dynamically align LED toggle nav height with chat header
     */
    alignNavHeights() {
        const chatHeader = document.querySelector('.chat-header');
        const ledNav = document.querySelector('.led-toggle-nav');

        if (chatHeader && ledNav) {
            const chatHeight = chatHeader.offsetHeight;
            ledNav.style.height = `${chatHeight}px`;
            console.log(`[SceneMakerPage] Aligned LED nav to chat header: ${chatHeight}px`);
        }
    }

    /**
     * Sync send button dimensions with input height
     */
    syncSendButtonSize() {
        const chatInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');

        if (chatInput && sendButton) {
            const inputHeight = chatInput.offsetHeight;
            sendButton.style.width = `${inputHeight}px`;
            sendButton.style.height = `${inputHeight}px`;

            // Scale icon proportionally (50% of button size)
            const icon = sendButton.querySelector('svg');
            if (icon) {
                const iconSize = Math.round(inputHeight * 0.5);
                icon.style.width = `${iconSize}px`;
                icon.style.height = `${iconSize}px`;
            }
        }
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Get DOM elements
        this.elements.userInput = document.getElementById('user-input');
        this.elements.sendButton = document.getElementById('send-button');

        if (!this.elements.userInput || !this.elements.sendButton) {
            console.error('[SceneMakerPage] Input elements not found');
            return;
        }

        // Sync button size initially
        this.syncSendButtonSize();

        // Send button
        this.elements.sendButton.addEventListener('click', () => this.sendMessage());

        // Enter key in textarea
        this.elements.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Sync button size when input resizes
        this.elements.userInput.addEventListener('input', () => {
            this.syncSendButtonSize();
        });

        // Tab switching
        document.querySelectorAll('.led-toggle').forEach(button => {
            button.addEventListener('click', () => {
                const tab = button.dataset.tab;
                this.switchTab(tab);
            });
        });

        // New Dream button (via event)
        window.addEventListener('new-dream-requested', () => {
            this.startNewDream();
        });
    }

    /**
     * Send a message to Echo
     */
    async sendMessage() {
        const message = this.elements.userInput.value.trim();

        if (!message || this.isEchoThinking) {
            return;
        }

        console.log('[SceneMakerPage] Sending message');

        // Capture BEFORE script for animation
        const beforeScript = await API.getSceneScript();

        // Add user message
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

            // Update content if handoff
            if (response.response_type === 'handoff') {
                const editsResponse = await API.getEdits();
                if (editsResponse.edits && editsResponse.edits.length > 0) {
                    await this.animateEdits(editsResponse.edits, beforeScript);
                }
                await this.refreshContent();
            }

            this.updateStatus('ready');
        } catch (error) {
            console.error('[SceneMakerPage] Send message error:', error);
            this.addMessage('system', `Error: ${error.message}`);
            this.updateStatus('ready');
        } finally {
            this.isEchoThinking = false;
            this.elements.sendButton.disabled = false;
            this.elements.userInput.focus();
        }
    }

    /**
     * Start a new dream
     */
    async startNewDream() {
        const confirmed = confirm(
            'Start a new dream?\n\n' +
            'Your current session is already saved to the database.\n' +
            'This will create a fresh session with a new conversation.'
        );

        if (!confirmed) return;

        try {
            const response = await fetch('/api/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const result = await response.json();

            if (result.success) {
                window.location.reload();
            } else {
                alert('Failed to start new dream: ' + result.error);
            }
        } catch (error) {
            console.error('[SceneMakerPage] Error starting new dream:', error);
            alert('Error starting new dream: ' + error.message);
        }
    }

    /**
     * Add a message to chat
     */
    addMessage(role, text) {
        this.messages.push({
            role: role,
            message: text,
            timestamp: new Date().toISOString()
        });

        // Update chat sidebar
        ChatSidebar.update('chat-sidebar-container', this.messages, this.currentStatus);

        // Re-attach event listeners
        this.setupEventListeners();
    }

    /**
     * Update Echo's status
     */
    updateStatus(status) {
        this.currentStatus = status;

        // Update app header
        if (this.appHeader) {
            this.appHeader.setStatus(status);
        }

        // Update chat sidebar (for typing indicator)
        ChatSidebar.update('chat-sidebar-container', this.messages, this.currentStatus);
        this.setupEventListeners();
    }

    /**
     * Switch to a different tab
     */
    switchTab(tabName) {
        this.currentTab = tabName;
        MainView.switchTab(tabName);
    }

    /**
     * Refresh all content from backend
     */
    async refreshContent() {
        try {
            const [screenshots, script, sceneData, metalog] = await Promise.all([
                API.getScreenshots(),
                API.getSceneScript(),
                API.getSceneData(),
                API.getMetalog()
            ]);

            CameraGallery.update('camera-gallery-container', screenshots);
            ScriptViewer.update('script-viewer-container', script);
            SceneInspector.update('scene-inspector-container', sceneData);
            MetalogViewer.update('metalog-viewer-container', metalog);
        } catch (error) {
            console.error('[SceneMakerPage] Refresh content error:', error);
        }
    }

    /**
     * Animate code edits
     */
    async animateEdits(edits, beforeScript = '') {
        this.updateStatus('building');
        this.switchTab('script');

        ScriptViewer.renderAnimated('script-viewer-container', beforeScript);

        const viewer = document.getElementById('script-animated-content');
        if (!viewer) return;

        const lines = Array.from(viewer.querySelectorAll('.code-line'));

        for (const edit of edits) {
            if (edit.event !== 'edit') continue;
            await this.applyEditAnimation(viewer, lines, edit);
            await AppUtils.wait(200);
        }
    }

    /**
     * Apply single edit animation
     */
    async applyEditAnimation(viewer, lines, edit) {
        const { op, code } = edit;
        const afterBlock = edit.after_block !== undefined ? parseInt(edit.after_block) : edit.after_line;
        const blockId = edit.block !== undefined ? parseInt(edit.block) : edit.line;

        if (op === 'insert') {
            const lineEl = document.createElement('div');
            lineEl.className = 'code-line code-line-insert';
            lineEl.textContent = code;

            const insertIndex = afterBlock === null || isNaN(afterBlock) ? 0 : afterBlock + 1;

            if (insertIndex >= lines.length) {
                viewer.appendChild(lineEl);
                lines.push(lineEl);
            } else {
                viewer.insertBefore(lineEl, lines[insertIndex]);
                lines.splice(insertIndex, 0, lineEl);
            }

            await AppUtils.wait(400);
        } else if (op === 'delete') {
            const deleteIndex = blockId;
            if (deleteIndex >= 0 && deleteIndex < lines.length) {
                const lineEl = lines[deleteIndex];
                lineEl.classList.add('code-line-delete');
                await AppUtils.wait(400);
                lineEl.remove();
                lines.splice(deleteIndex, 1);
            }
        } else if (op === 'replace') {
            const replaceIndex = blockId;
            if (replaceIndex >= 0 && replaceIndex < lines.length) {
                const lineEl = lines[replaceIndex];
                lineEl.textContent = code;
                lineEl.classList.add('code-line-replace');
                await AppUtils.wait(400);
                lineEl.classList.remove('code-line-replace');
            }
        }
    }
}

// Export for window access
window.SceneMaker = SceneMaker;
window.SceneMakerPage = SceneMakerPage;

// Initialize page when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.sceneMakerPage = new SceneMakerPage();
});
