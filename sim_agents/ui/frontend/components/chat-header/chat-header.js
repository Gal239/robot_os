/**
 * Chat Header Component
 * ======================
 *
 * Header for chat sidebar showing Echo status
 */

export class ChatHeader {
    /**
     * Render chat header
     *
     * @param {string} status - 'ready', 'thinking', 'building'
     * @returns {string} HTML string
     */
    static render(status = 'ready') {
        const statusText = {
            ready: 'Ready',
            thinking: 'Thinking...',
            building: 'Building scene...'
        }[status] || 'Ready';

        const statusDotClass = {
            ready: 'ready',
            thinking: 'thinking',
            building: 'building'
        }[status] || 'ready';

        return `
            <div class="chat-header">
                <div class="chat-header-title">
                    <span class="chat-header-icon">🤖</span>
                    <div class="chat-header-text">
                        <h3>Echo v0.1</h3>
                        <p>Scene Builder</p>
                    </div>
                </div>
                <div class="chat-header-status">
                    <span class="status-dot ${statusDotClass}"></span>
                    <span>${statusText}</span>
                </div>
            </div>
        `;
    }
}
