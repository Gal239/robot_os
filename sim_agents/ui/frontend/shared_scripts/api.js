/**
 * Echo Scene Maker UI - API Client
 * ==================================
 *
 * Wrapper for API calls to backend
 */

const API_BASE = 'http://localhost:5050/api';

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'API request failed');
        }

        return data;

    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

/**
 * Start a new conversation with Echo
 */
export async function startConversation() {
    return await apiFetch('/start', {
        method: 'POST'
    });
}

/**
 * Send a message to Echo
 *
 * @param {string} message - User's message
 * @returns {Promise<Object>} Response from Echo
 */
export async function sendMessage(message) {
    return await apiFetch('/chat', {
        method: 'POST',
        body: JSON.stringify({ message })
    });
}

/**
 * Get full conversation history
 */
export async function getConversation() {
    return await apiFetch('/conversation');
}

/**
 * Get all screenshots from latest scene
 */
export async function getScreenshots() {
    return await apiFetch('/screenshots');
}

/**
 * Get current scene script
 */
export async function getSceneScript() {
    return await apiFetch('/scene-script');
}

/**
 * Get scene metadata (cameras, assets, sensors)
 */
export async function getSceneData() {
    return await apiFetch('/scene-data');
}

/**
 * Get metalog (conversation context)
 */
export async function getMetalog() {
    return await apiFetch('/metalog');
}

/**
 * Get Echo's current status
 */
export async function getStatus() {
    return await apiFetch('/status');
}
