/*
 * Full Log Modal Component
 */

class LogModalClass {
    constructor() {
        this.modal = document.getElementById('logModal');
        this.currentTaskId = null;
        this.currentSessionId = null;

        // Bind ESC key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !this.modal.classList.contains('hidden')) {
                this.close();
            }
        });
    }

    async open(taskId, sessionId) {
        this.currentTaskId = taskId;
        this.currentSessionId = sessionId;

        // Show modal
        this.modal.classList.remove('hidden');

        // Update title
        document.getElementById('logModalTaskId').textContent = taskId;

        // Show loading state
        const content = document.getElementById('logModalContent');
        content.innerHTML = `
            <div class="loading-state">
                <div class="spinner"></div>
                <p>Loading log for ${taskId}...</p>
            </div>
        `;

        try {
            // Fetch task data
            const taskData = await API.getTask(taskId, sessionId);

            // Render metadata
            this.renderMetadata(taskData);

            // Render log timeline
            this.renderLog(taskData);

            // Reinitialize Lucide icons
            lucide.createIcons();

        } catch (error) {
            console.error('Error loading log:', error);
            content.innerHTML = `
                <div class="log-empty-state">
                    <i data-lucide="alert-circle"></i>
                    <p>Failed to load log</p>
                    <p style="font-size: var(--font-size-sm); margin-top: var(--spacing-sm);">${error.message}</p>
                </div>
            `;
            lucide.createIcons();
        }
    }

    renderMetadata(taskData) {
        const metadata = document.getElementById('logModalMetadata');

        const agent = taskData.agent_id || 'N/A';
        const status = taskData.status || 'unknown';
        const created = taskData.created_at ? formatAbsoluteTime(taskData.created_at) : 'N/A';
        const completed = taskData.completed_at ? formatAbsoluteTime(taskData.completed_at) : 'Running...';

        let duration = '-';
        if (taskData.created_at && taskData.completed_at) {
            const start = new Date(taskData.created_at);
            const end = new Date(taskData.completed_at);
            duration = formatDuration(end - start);
        }

        // Filter out "result from" entries
        const filteredTimeline = taskData.tool_timeline ?
            taskData.tool_timeline.filter(event => {
                const toolName = (event.tool || '').toLowerCase();
                return !toolName.includes('result from') && !toolName.includes('← result from');
            }) : [];
        const toolCalls = filteredTimeline.length;
        const parent = taskData.parent_task_id || 'Root';

        metadata.innerHTML = `
            <div class="metadata-item">
                <span class="metadata-label">Agent:</span>
                <span class="metadata-value">${agent}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Status:</span>
                <span class="badge badge-${this.getStatusBadgeClass(status)}">${status}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Parent:</span>
                <span class="metadata-value">${parent}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Tool Calls:</span>
                <span class="metadata-value">${toolCalls}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Created:</span>
                <span class="metadata-value">${created}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Completed:</span>
                <span class="metadata-value">${completed}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Duration:</span>
                <span class="metadata-value">${duration}</span>
            </div>
        `;
    }

    renderLog(taskData) {
        const content = document.getElementById('logModalContent');
        const rawTimeline = taskData.tool_timeline || [];

        // Filter out "result from" entries
        const timeline = rawTimeline.filter(event => {
            const toolName = (event.tool || '').toLowerCase();
            return !toolName.includes('result from') && !toolName.includes('← result from');
        });

        if (timeline.length === 0) {
            content.innerHTML = `
                <div class="log-empty-state">
                    <i data-lucide="inbox"></i>
                    <p>No tool calls yet</p>
                </div>
            `;
            return;
        }

        const timelineHTML = timeline.map((event, index) => {
            const toolIcon = getToolIcon(event.type);
            const timestamp = event.timestamp ? formatAbsoluteTime(event.timestamp) : 'N/A';
            const relativeTime = event.timestamp ? formatRelativeTime(event.timestamp) : '';

            return `
                <div class="log-event">
                    <div class="log-event-header">
                        <div class="log-event-title">
                            <i data-lucide="${toolIcon}"></i>
                            <span>${event.tool || 'Unknown Tool'}</span>
                            <span class="log-event-type-badge ${event.type}">${(event.type || '').replace(/_/g, ' ')}</span>
                        </div>
                        <div class="log-event-time" title="${timestamp}">
                            ${relativeTime}
                        </div>
                    </div>
                    <div class="log-event-body">
                        <div class="log-event-section">
                            <div class="log-event-section-title">
                                Input
                                <button class="copy-btn" onclick="LogModal.copyToClipboard('${this.escapeHtml(JSON.stringify(event.input))}')">
                                    Copy
                                </button>
                            </div>
                            <div class="log-event-content">${formatJSON(event.input)}</div>
                        </div>
                        <div class="log-event-section">
                            <div class="log-event-section-title">
                                Result
                                <button class="copy-btn" onclick="LogModal.copyToClipboard('${this.escapeHtml(JSON.stringify(event.result))}')">
                                    Copy
                                </button>
                            </div>
                            <div class="log-event-content">${formatJSON(event.result)}</div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        content.innerHTML = `
            <div class="log-timeline">
                ${timelineHTML}
            </div>
        `;
    }

    getStatusBadgeClass(status) {
        const mapping = {
            completed: 'success',
            running: 'info',
            waiting: 'warning',
            ready: 'gray'
        };
        return mapping[status] || 'gray';
    }

    escapeHtml(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    copyToClipboard(text) {
        // Unescape HTML entities
        const textarea = document.createElement('textarea');
        textarea.innerHTML = text;
        const decodedText = textarea.value;

        navigator.clipboard.writeText(decodedText).then(() => {
            showToast('Copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Failed to copy:', err);
            showToast('Failed to copy', 'error');
        });
    }

    close() {
        this.modal.classList.add('hidden');
        this.currentTaskId = null;
        this.currentSessionId = null;
    }

    exportLog() {
        if (!this.currentTaskId || !this.currentSessionId) return;

        // TODO: Implement export functionality
        showToast('Export feature coming soon!', 'info');
    }
}

// Global instance
window.LogModal = new LogModalClass();

// Add export button handler
document.addEventListener('DOMContentLoaded', () => {
    const exportBtn = document.getElementById('exportLogBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => LogModal.exportLog());
    }
});
