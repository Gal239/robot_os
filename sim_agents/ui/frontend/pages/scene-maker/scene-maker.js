/**
 * Scene Maker Page
 * =================
 *
 * Main application logic for Echo Scene Maker
 */

import * as API from '../../shared_scripts/api.js';
import { scrollToBottom, showToast } from '../../shared_scripts/utils.js';
import { AppBar } from '../../layouts/app-bar/app-bar.js';
import { ChatSidebar } from '../../layouts/chat-sidebar/chat-sidebar.js';
import { TabbedArea } from '../../layouts/tabbed-area/tabbed-area.js';

export class SceneMakerPage {
    constructor() {
        this.messages = [];
        this.echoStatus = 'ready';
        this.sceneData = {};
        this.currentScript = '';
        this.metalog = '';

        this.init();
    }

    async init() {
        console.log('🤖 Initializing Echo Scene Maker...');

        // Start conversation with backend
        try {
            const result = await API.startConversation();
            console.log('✅ Conversation started:', result.session_id);
        } catch (error) {
            console.error('❌ Failed to start conversation:', error);
            showToast('Failed to connect to Echo backend', 'error');
        }

        // Listen for send-message event
        window.addEventListener('send-message', (e) => {
            this.handleSendMessage(e.detail.message);
        });

        // Initial render
        this.render();
    }

    async handleSendMessage(message) {
        console.log('💬 Sending message:', message);

        // Add user message to conversation
        this.messages.push({
            role: 'user',
            message: message,
            timestamp: new Date().toISOString()
        });

        // Update status and render
        this.echoStatus = 'thinking';
        this.render();
        this.scrollChat();

        try {
            // Send to backend
            const result = await API.sendMessage(message);

            if (result.success) {
                // Add Echo's response
                this.messages.push({
                    role: 'echo',
                    message: result.echo_response,
                    response_type: result.response_type,
                    timestamp: new Date().toISOString()
                });

                // Update scene data if handoff
                if (result.response_type === 'handoff') {
                    this.sceneData = result.scene_data || {};
                    this.echoStatus = 'ready';

                    // Fetch additional data
                    await this.refreshData();

                    showToast('Scene created successfully!', 'success');
                } else {
                    this.echoStatus = 'ready';
                }

            } else {
                throw new Error(result.error || 'Unknown error');
            }

        } catch (error) {
            console.error('❌ Error sending message:', error);
            showToast('Failed to send message: ' + error.message, 'error');
            this.echoStatus = 'ready';
        }

        // Render and scroll
        this.render();
        this.scrollChat();
    }

    async refreshData() {
        try {
            // Get script
            const scriptResult = await API.getSceneScript();
            if (scriptResult.success) {
                this.currentScript = scriptResult.script;
            }

            // Get metalog
            const metalogResult = await API.getMetalog();
            if (metalogResult.success) {
                this.metalog = metalogResult.metalog;
            }

        } catch (error) {
            console.error('❌ Error refreshing data:', error);
        }
    }

    render() {
        const app = document.getElementById('app');

        if (!app) {
            console.error('❌ #app element not found');
            return;
        }

        app.innerHTML = `
            ${AppBar.render()}

            <div class="page-container">
                <div class="page-sidebar">
                    ${ChatSidebar.render(this.messages, this.echoStatus)}
                </div>

                <div class="page-main">
                    ${TabbedArea.render({
                        scene: { screenshots: this.sceneData.screenshots || {} },
                        script: this.currentScript,
                        sceneData: this.sceneData,
                        metalog: this.metalog
                    })}
                </div>
            </div>
        `;

        // Re-initialize Alpine.js components after render
        if (window.Alpine) {
            window.Alpine.initTree(app);
        }
    }

    scrollChat() {
        // Small delay to ensure DOM is updated
        setTimeout(() => {
            const chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                scrollToBottom(chatMessages);
            }
        }, 100);
    }
}

// Global function for copy script button
window.copyScript = async function() {
    const scriptCode = document.querySelector('.script-code code');
    if (scriptCode) {
        try {
            await navigator.clipboard.writeText(scriptCode.textContent);
            showToast('Script copied to clipboard!', 'success');
        } catch (error) {
            showToast('Failed to copy script', 'error');
        }
    }
};

// Initialize page when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new SceneMakerPage();
    });
} else {
    new SceneMakerPage();
}
