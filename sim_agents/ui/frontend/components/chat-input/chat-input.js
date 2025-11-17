/**
 * Chat Input Component
 * =====================
 *
 * Message input field with send button
 */

export class ChatInput {
    /**
     * Render chat input
     *
     * @param {Function} onSend - Callback when message is sent
     * @param {boolean} disabled - Is input disabled
     * @returns {string} HTML string
     */
    static render(onSend, disabled = false) {
        return `
            <div class="chat-input-container" x-data="chatInput()">
                <div class="chat-input-wrapper">
                    <textarea
                        id="chat-input-field"
                        class="chat-input-field"
                        placeholder="💬 Type your message..."
                        rows="1"
                        :disabled="${disabled}"
                        x-model="message"
                        @keydown.enter.prevent="handleEnter($event)"
                        @input="autoResize($event)"
                    ></textarea>
                    <button
                        class="chat-send-button button-primary"
                        @click="sendMessage()"
                        :disabled="${disabled} || message.trim() === ''"
                    >
                        Send 📤
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Initialize Alpine.js component logic
     */
    static initAlpine() {
        return {
            message: '',

            handleEnter(event) {
                // Shift+Enter = new line, Enter = send
                if (!event.shiftKey) {
                    this.sendMessage();
                }
            },

            sendMessage() {
                if (this.message.trim() === '') return;

                // Dispatch custom event
                window.dispatchEvent(new CustomEvent('send-message', {
                    detail: { message: this.message }
                }));

                this.message = '';
                this.autoResize({ target: document.getElementById('chat-input-field') });
            },

            autoResize(event) {
                const textarea = event.target;
                textarea.style.height = 'auto';
                textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
            }
        };
    }
}

// Register Alpine.js component
window.chatInput = ChatInput.initAlpine;
