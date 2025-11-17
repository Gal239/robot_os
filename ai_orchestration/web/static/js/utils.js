/*
 * Utility Functions
 */

// Format timestamp to relative time (e.g., "2 minutes ago")
function formatRelativeTime(timestamp) {
    if (!timestamp) return '-';

    const now = new Date();
    const date = new Date(timestamp);
    const diff = now - date;

    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (seconds < 60) return `${seconds}s ago`;
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
}

// Format duration (ms) to human-readable
function formatDuration(ms) {
    if (!ms || ms < 0) return '-';

    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
}

// Format timestamp to absolute time
function formatAbsoluteTime(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleString();
}

// Truncate text with ellipsis
function truncate(text, maxLength = 50) {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// Copy text to clipboard
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard!', 'success');
        return true;
    } catch (err) {
        console.error('Failed to copy:', err);
        showToast('Failed to copy', 'error');
        return false;
    }
}

// Show toast notification
function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type} animate-slide-in-right`;
    toast.innerHTML = `
        <i data-lucide="${getToastIcon(type)}"></i>
        <span>${message}</span>
    `;

    container.appendChild(toast);
    lucide.createIcons();

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

function getToastIcon(type) {
    const icons = {
        success: 'check-circle',
        error: 'x-circle',
        warning: 'alert-circle',
        info: 'info'
    };
    return icons[type] || 'info';
}

// Show/hide loading overlay
function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.querySelector('p').textContent = message;
        overlay.classList.add('active');
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

// Debounce function
function debounce(func, wait) {
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

// Format JSON for display
function formatJSON(obj) {
    try {
        // Handle null/undefined
        if (obj === null || obj === undefined) {
            return String(obj);
        }

        // If already a string, try to parse it first
        if (typeof obj === 'string') {
            try {
                const parsed = JSON.parse(obj);
                return JSON.stringify(parsed, null, 2);
            } catch {
                // Not valid JSON, return as-is
                return obj;
            }
        }

        // If it's an object, stringify it
        if (typeof obj === 'object') {
            return JSON.stringify(obj, null, 2);
        }

        // For primitives, return as string
        return String(obj);
    } catch (err) {
        console.error('formatJSON error:', err);
        return String(obj);
    }
}

// Get status color
function getStatusColor(status) {
    const colors = {
        completed: 'var(--color-success)',
        running: 'var(--color-info)',
        waiting: 'var(--color-warning)',
        ready: 'var(--color-gray-light)',
        error: 'var(--color-danger)'
    };
    return colors[status] || 'var(--color-gray)';
}

// Get tool type icon
function getToolIcon(toolType) {
    const icons = {
        agent_as_tool: 'git-branch',
        handoff: 'arrow-right',
        ask_master: 'message-circle',
        function_tool: 'zap',
        non_function_tool: 'file-text'
    };
    return icons[toolType] || 'tool';
}

// Create SVG icon element
function createIcon(iconName, className = '') {
    const icon = document.createElement('i');
    icon.setAttribute('data-lucide', iconName);
    if (className) icon.className = className;
    return icon;
}

// Parse session ID to get date
function parseSessionDate(sessionId) {
    // session_20251022_104741 -> Oct 22, 2025 10:47
    const match = sessionId.match(/session_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
    if (!match) return sessionId;

    const [, year, month, day, hour, min, sec] = match;
    const date = new Date(year, month - 1, day, hour, min, sec);
    return date.toLocaleString();
}

// Calculate statistics
function calculateStats(nodes) {
    const stats = {
        total: 0,
        completed: 0,
        running: 0,
        waiting: 0,
        ready: 0,
        totalToolCalls: 0
    };

    if (!nodes) return stats;

    for (const node of Object.values(nodes)) {
        stats.total++;
        if (node.status === 'completed') stats.completed++;
        else if (node.status === 'running') stats.running++;
        else if (node.status === 'waiting') stats.waiting++;
        else if (node.status === 'ready') stats.ready++;

        if (node.tool_timeline) {
            // Filter out "result from" entries
            const filteredTimeline = node.tool_timeline.filter(event => {
                const toolName = (event.tool || '').toLowerCase();
                return !toolName.includes('result from') && !toolName.includes('← result from');
            });
            stats.totalToolCalls += filteredTimeline.length;
        }
    }

    return stats;
}

// Export data as JSON
function exportJSON(data, filename) {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// Export data as CSV
function exportCSV(data, filename) {
    if (!data || data.length === 0) return;

    const headers = Object.keys(data[0]);
    const csv = [
        headers.join(','),
        ...data.map(row =>
            headers.map(header => {
                const value = row[header];
                // Escape quotes and wrap in quotes if contains comma
                const escaped = String(value).replace(/"/g, '""');
                return escaped.includes(',') ? `"${escaped}"` : escaped;
            }).join(',')
        )
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}
