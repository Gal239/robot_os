/**
 * Echo Scene Builder - API Client
 * Professional API wrapper for backend communication
 */

const API_BASE = '';

/**
 * API Client
 */
const API = {
    /**
     * Start a new conversation with Echo
     */
    async start() {
        try {
            const response = await fetch(`${API_BASE}/api/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('[API] Conversation started:', data.session_id);
            return data;
        } catch (error) {
            console.error('[API] Start error:', error);
            throw error;
        }
    },

    /**
     * Send a message to Echo
     * @param {string} message - User message
     */
    async sendMessage(message) {
        try {
            const response = await fetch(`${API_BASE}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'Unknown error');
            }

            console.log('[API] Message sent successfully');
            return data;
        } catch (error) {
            console.error('[API] Chat error:', error);
            throw error;
        }
    },

    /**
     * Get conversation history
     */
    async getConversation() {
        try {
            const response = await fetch(`${API_BASE}/api/conversation`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data.conversation || [];
        } catch (error) {
            console.error('[API] Get conversation error:', error);
            return [];
        }
    },

    /**
     * Get screenshots from latest scene
     */
    async getScreenshots() {
        try {
            const response = await fetch(`${API_BASE}/api/screenshots`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data.screenshots || {};
        } catch (error) {
            console.error('[API] Get screenshots error:', error);
            return {};
        }
    },

    /**
     * Get scene script from workspace (with imports)
     */
    async getSceneScript() {
        try {
            const response = await fetch(`${API_BASE}/api/workspace/script`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data.script || '';
        } catch (error) {
            console.error('[API] Get script error:', error);
            return '';
        }
    },

    /**
     * Get scene data (metadata, cameras, assets, sensors)
     */
    async getSceneData() {
        try {
            const response = await fetch(`${API_BASE}/api/scene-data`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data.scene_data || {};
        } catch (error) {
            console.error('[API] Get scene data error:', error);
            return {};
        }
    },

    /**
     * Get metalog (conversation context)
     */
    async getMetalog() {
        try {
            const response = await fetch(`${API_BASE}/api/metalog`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data.metalog || '';
        } catch (error) {
            console.error('[API] Get metalog error:', error);
            return '';
        }
    },

    /**
     * Get Echo's current status
     */
    async getStatus() {
        try {
            const response = await fetch(`${API_BASE}/api/status`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data.status || 'not_started';
        } catch (error) {
            console.error('[API] Get status error:', error);
            return 'error';
        }
    },

    /**
     * Get document edit stream (for animation)
     */
    async getEdits() {
        try {
            const response = await fetch(`${API_BASE}/api/edits`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('[API] Get edits error:', error);
            return { edits: [] };
        }
    }
};
