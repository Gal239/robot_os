/**
 * Message Bubble Component
 * =========================
 *
 * Chat message display for Echo and user messages
 */

import { formatTime, parseMarkdown } from '../../shared_scripts/utils.js';

export class MessageBubble {
    /**
     * Render a message bubble
     *
     * @param {Object} message - Message data
     * @param {string} message.role - 'echo' or 'user'
     * @param {string} message.message - Message text
     * @param {string} message.timestamp - ISO timestamp
     * @param {string} message.response_type - 'ask_master' or 'handoff' (for Echo)
     * @returns {string} HTML string
     */
    static render(message) {
        const isEcho = message.role === 'echo';
        const isUser = message.role === 'user';

        const bubbleClass = isEcho ? 'message-bubble-echo' : 'message-bubble-user';
        const icon = isEcho ? '🤖' : '👤';
        const name = isEcho ? 'Echo' : 'You';
        const time = formatTime(message.timestamp);

        // Parse markdown for Echo messages
        const content = isEcho ? parseMarkdown(message.message) : message.message;

        // Response type badge (for Echo)
        let badge = '';
        if (isEcho && message.response_type) {
            const badgeText = message.response_type === 'handoff' ? '🎬 Scene Built' : '💬 Chat';
            const badgeColor = message.response_type === 'handoff' ? 'var(--mint-green)' : 'var(--echo-blue)';

            badge = `
                <span class="message-badge" style="background-color: ${badgeColor};">
                    ${badgeText}
                </span>
            `;
        }

        return `
            <div class="${bubbleClass} animate__animated animate__fadeInUp"
                 style="animation-duration: 400ms;">
                <div class="message-header">
                    <span class="message-icon">${icon}</span>
                    <span class="message-name">${name}</span>
                    ${badge}
                    <span class="message-time">${time}</span>
                </div>
                <div class="message-content">
                    ${content}
                </div>
            </div>
        `;
    }
}
