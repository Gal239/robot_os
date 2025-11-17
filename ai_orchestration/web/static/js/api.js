/*
 * API Communication Layer
 */

const API_BASE_URL = window.location.origin;

class API {
    static async request(endpoint, options = {}) {
        try {
            // Add cache-busting timestamp to ALL requests
            const separator = endpoint.includes('?') ? '&' : '?';
            const cacheBustingUrl = `${API_BASE_URL}${endpoint}${separator}_t=${Date.now()}`;

            const response = await fetch(cacheBustingUrl, {
                headers: {
                    'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    // ========== Sessions ==========

    static async listSessions() {
        return this.request('/api/sessions');
    }

    static async getSession(sessionId) {
        return this.request(`/api/sessions/${sessionId}`);
    }

    static async getSessionGraph(sessionId) {
        return this.request(`/api/sessions/${sessionId}/graph`);
    }

    static async getSessionLogs(sessionId, logType = 'master', taskId = null) {
        let url = `/api/sessions/${sessionId}/logs?log_type=${logType}`;
        if (taskId) url += `&task_id=${taskId}`;
        return this.request(url);
    }

    // ========== Tasks ==========

    static async getTask(taskId, sessionId) {
        return this.request(`/api/tasks/${taskId}?session_id=${sessionId}`);
    }

    // ========== Workspace ==========

    static async getWorkspaceFiles(sessionId) {
        return this.request(`/api/sessions/${sessionId}/workspace`);
    }

    static async getWorkspaceFile(sessionId, filePath) {
        return this.request(`/api/sessions/${sessionId}/workspace/file?file_path=${encodeURIComponent(filePath)}`);
    }

    static async getSnapshots(sessionId) {
        return this.request(`/api/sessions/${sessionId}/snapshots`);
    }

    // ========== Agents ==========

    static async listAgents() {
        return this.request('/api/agents');
    }

    // ========== Health ==========

    static async healthCheck() {
        return this.request('/api/health');
    }
}

// Cache manager
class CacheManager {
    constructor() {
        this.cache = new Map();
        this.ttl = 60000; // 1 minute
    }

    set(key, value) {
        this.cache.set(key, {
            value,
            timestamp: Date.now()
        });
    }

    get(key) {
        const cached = this.cache.get(key);
        if (!cached) return null;

        // Check if expired
        if (Date.now() - cached.timestamp > this.ttl) {
            this.cache.delete(key);
            return null;
        }

        return cached.value;
    }

    clear() {
        this.cache.clear();
    }

    clearKey(key) {
        this.cache.delete(key);
    }
}

// Global cache instance
const cache = new CacheManager();
