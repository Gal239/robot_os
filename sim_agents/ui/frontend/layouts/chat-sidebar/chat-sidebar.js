/**
 * Chat Sidebar Layout
 * ====================
 *
 * Complete chat interface with header, messages, and input
 */

import { ChatHeader } from '../../components/chat-header/chat-header.js';
import { MessageBubble } from '../../components/message-bubble/message-bubble.js';
import { ChatInput } from '../../components/chat-input/chat-input.js';

export class ChatSidebar {
    /**
     * Render chat sidebar
     *
     * @param {Array} messages - Array of message objects
     * @param {string} status - Echo's current status
     * @returns {string} HTML string
     */
    static render(messages = [], status = 'ready') {
        const messagesHTML = messages.map(msg => MessageBubble.render(msg)).join('');

        // Show "Echo is typing..." if thinking/building
        const typingIndicator = (status !== 'ready') ? `
            <div class="typing-indicator-wrapper animate__animated animate__fadeIn">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span class="typing-text">Echo is ${status}...</span>
            </div>
        ` : '';

        return `
            <div class="chat-sidebar">
                ${ChatHeader.render(status)}

                <div class="chat-messages" id="chat-messages">
                    ${messagesHTML}
                    ${typingIndicator}
                </div>

                ${ChatInput.render(null, status !== 'ready')}
            </div>
        `;
    }
}
