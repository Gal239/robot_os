/**
 * Echo Scene Maker UI - Utility Functions
 * =========================================
 *
 * Helper functions used across the app
 */

/**
 * Format timestamp to readable time
 * @param {string} isoString - ISO timestamp
 * @returns {string} Formatted time (e.g., "2:34 PM")
 */
export function formatTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit'
    });
}

/**
 * Format date to readable format
 * @param {string} isoString - ISO timestamp
 * @returns {string} Formatted date (e.g., "Nov 14, 2025")
 */
export function formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    });
}

/**
 * Debounce function calls
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in ms
 * @returns {Function} Debounced function
 */
export function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Scroll element to bottom (for chat)
 * @param {HTMLElement} element - Element to scroll
 * @param {boolean} smooth - Use smooth scrolling
 */
export function scrollToBottom(element, smooth = true) {
    if (!element) return;

    element.scrollTo({
        top: element.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto'
    });
}

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 * @returns {Promise<boolean>} Success status
 */
export async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (error) {
        console.error('Failed to copy:', error);
        return false;
    }
}

/**
 * Show toast notification
 * @param {string} message - Message to show
 * @param {string} type - Type: 'success', 'error', 'info', 'warning'
 * @param {number} duration - Duration in ms
 */
export function showToast(message, type = 'info', duration = 3000) {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type} animate__animated animate__fadeInUp`;
    toast.textContent = message;

    // Style
    Object.assign(toast.style, {
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        padding: '16px 24px',
        borderRadius: '12px',
        backgroundColor: getToastColor(type),
        color: type === 'warning' || type === 'success' ? '#4A4A4A' : '#FFFFFF',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        zIndex: '9999',
        fontWeight: '500',
        fontSize: '14px'
    });

    document.body.appendChild(toast);

    // Remove after duration
    setTimeout(() => {
        toast.classList.remove('animate__fadeInUp');
        toast.classList.add('animate__fadeOutDown');
        setTimeout(() => toast.remove(), 400);
    }, duration);
}

function getToastColor(type) {
    const colors = {
        success: '#A8E6CF',
        error: '#FFB3B3',
        info: '#A8D5E2',
        warning: '#FFE5B4'
    };
    return colors[type] || colors.info;
}

/**
 * Format number with commas
 * @param {number} num - Number to format
 * @returns {string} Formatted number
 */
export function formatNumber(num) {
    return num.toLocaleString('en-US');
}

/**
 * Truncate text with ellipsis
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} Truncated text
 */
export function truncate(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength) + '...';
}

/**
 * Escape HTML to prevent XSS
 * @param {string} text - Text to escape
 * @returns {string} Escaped HTML
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

/**
 * Parse markdown-like text (basic)
 * @param {string} text - Text to parse
 * @returns {string} HTML string
 */
export function parseMarkdown(text) {
    // Code blocks (process first to avoid interfering with other patterns)
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        const language = lang || 'text';
        return `<pre class="code-block"><code class="language-${language}">${escapeHtml(code.trim())}</code></pre>`;
    });

    // Bold
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Italic
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Inline code
    text = text.replace(/`(.*?)`/g, '<code>$1</code>');

    // Line breaks
    text = text.replace(/\n/g, '<br>');

    return text;
}

/**
 * Generate unique ID
 * @returns {string} Unique ID
 */
export function generateId() {
    return `id_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Check if element is in viewport
 * @param {HTMLElement} element - Element to check
 * @returns {boolean} Is in viewport
 */
export function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

/**
 * Wait for specified time
 * @param {number} ms - Milliseconds to wait
 * @returns {Promise} Promise that resolves after wait
 */
export function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
