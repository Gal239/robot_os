/**
 * Workspace View - Premium Card-Based File Browser
 * Uses existing sidebar and modal components
 */

class WorkspaceView {
    constructor() {
        this.files = [];
        this.selectedFile = null;
        this.currentSessionId = null;
        this.currentFileData = null;
    }

    /**
     * Initialize the workspace view
     */
    async init(sessionId) {
        console.log('[Workspace] Initializing for session:', sessionId);
        this.currentSessionId = sessionId;

        try {
            // Fetch workspace files
            const data = await API.getWorkspaceFiles(sessionId);

            if (!data.has_workspace || data.files.length === 0) {
                this.renderEmptyState();
                return;
            }

            this.files = data.files;
            this.renderFileCards();
        } catch (error) {
            console.error('[Workspace] Error loading workspace:', error);
            this.renderError('Failed to load workspace files');
        }
    }

    /**
     * Render file cards in grid
     */
    renderFileCards() {
        const container = document.getElementById('workspaceFilesGrid');

        if (!container) {
            console.error('[Workspace] Container not found');
            return;
        }

        // Build cards HTML
        const cardsHtml = this.files.map(file => this.renderFileCard(file)).join('');

        container.innerHTML = cardsHtml;

        // Add click handlers
        this.attachCardClickHandlers();

        // Re-render Lucide icons
        if (typeof lucide !== 'undefined' && lucide.createIcons) {
            lucide.createIcons();
        }
    }

    /**
     * Render a single file card
     */
    renderFileCard(file) {
        const iconClass = this.getFileIconClass(file.file_type);
        const icon = this.getFileIcon(file.file_type);
        const size = this.formatFileSize(file.size_bytes);

        return `
            <div class="metric-card file-card" data-file-path="${this.escapeHtml(file.path)}">
                <div class="file-card-icon ${iconClass}">
                    ${icon}
                </div>
                <div class="file-card-info">
                    <h3 class="file-card-title">${this.escapeHtml(file.name)}</h3>
                    <div class="file-meta">
                        <span class="badge badge-${iconClass}">${file.file_type}</span>
                        <span class="file-size">${size}</span>
                    </div>
                    ${file.created_by_agent ? `
                        <div class="file-agent-info">
                            <i data-lucide="user"></i>
                            <span>${this.escapeHtml(file.created_by_agent)}</span>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    /**
     * Attach click handlers to file cards
     */
    attachCardClickHandlers() {
        const cards = document.querySelectorAll('.file-card');

        cards.forEach(card => {
            card.addEventListener('click', () => {
                const filePath = card.dataset.filePath;
                this.selectFile(filePath);
            });
        });
    }

    /**
     * Select and preview a file
     */
    async selectFile(filePath) {
        console.log('[Workspace] Selecting file:', filePath);

        // Update UI selection
        document.querySelectorAll('.file-card').forEach(card => {
            card.classList.remove('selected');
        });

        const selectedCard = document.querySelector(`[data-file-path="${filePath}"]`);
        if (selectedCard) {
            selectedCard.classList.add('selected');
        }

        // Find file in list
        this.selectedFile = this.files.find(f => f.path === filePath);

        if (!this.selectedFile) {
            console.error('[Workspace] File not found:', filePath);
            return;
        }

        // Load file content and show in sidebar
        await this.loadAndShowInSidebar(filePath);
    }

    /**
     * Load file content and display in sidebar
     */
    async loadAndShowInSidebar(filePath) {
        const sidebar = document.getElementById('workspaceSidebar');
        if (!sidebar) return;

        // Show loading state
        sidebar.innerHTML = `
            <div class="sidebar-loading">
                <div class="spinner"></div>
                <p>Loading file...</p>
            </div>
        `;

        try {
            const fileData = await API.getWorkspaceFile(this.currentSessionId, filePath);
            this.currentFileData = fileData;
            this.renderSidebarContent(fileData);
        } catch (error) {
            console.error('[Workspace] Error loading file:', error);
            sidebar.innerHTML = `
                <div class="sidebar-empty">
                    <i data-lucide="alert-circle"></i>
                    <p>Failed to load file</p>
                </div>
            `;
            if (typeof lucide !== 'undefined' && lucide.createIcons) {
                lucide.createIcons();
            }
        }
    }

    /**
     * Render sidebar content
     */
    renderSidebarContent(fileData) {
        const sidebar = document.getElementById('workspaceSidebar');
        if (!sidebar) return;

        const taskInfo = fileData.metadata.task_info;

        // Build sidebar HTML
        const sidebarHtml = `
            <div class="sidebar-header">
                <h2 class="sidebar-title">${this.escapeHtml(this.selectedFile.name)}</h2>
            </div>

            <div class="sidebar-content">
                <!-- File Metadata -->
                <div class="sidebar-section">
                    <h3>File Information</h3>
                    <div class="info-row">
                        <span class="info-label">Type</span>
                        <span class="info-value">${this.escapeHtml(fileData.metadata.file_type || 'unknown')}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Size</span>
                        <span class="info-value">${this.formatFileSize(fileData.metadata.size_bytes)}</span>
                    </div>
                    ${taskInfo ? `
                        <div class="info-row">
                            <span class="info-label">Created By</span>
                            <span class="info-value">${this.escapeHtml(taskInfo.agent_id)}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Task</span>
                            <span class="info-value clickable" data-task-id="${taskInfo.task_id}">
                                ${this.escapeHtml(taskInfo.task_id)}
                            </span>
                        </div>
                    ` : ''}
                </div>

                <!-- Preview Section -->
                <div class="sidebar-section">
                    <h3>Preview</h3>
                    <div class="file-preview-content">
                        ${this.renderPreview(fileData, 10)}
                    </div>
                </div>

                <!-- Actions -->
                <div class="sidebar-section">
                    <div class="sidebar-actions">
                        <button class="btn btn-primary" id="viewFullBtn">
                            <i data-lucide="maximize-2"></i>
                            View Full Document
                        </button>
                        <button class="btn btn-secondary" id="downloadFileBtn">
                            <i data-lucide="download"></i>
                            Download
                        </button>
                        <button class="btn btn-secondary" id="copyContentBtn">
                            <i data-lucide="copy"></i>
                            Copy
                        </button>
                    </div>
                </div>
            </div>
        `;

        sidebar.innerHTML = sidebarHtml;

        // Attach action handlers
        this.attachSidebarActionHandlers(fileData);

        // Re-render icons
        if (typeof lucide !== 'undefined' && lucide.createIcons) {
            lucide.createIcons();
        }

        // Attach task click handler
        const taskElement = sidebar.querySelector('[data-task-id]');
        if (taskElement) {
            taskElement.addEventListener('click', () => {
                const taskId = taskElement.dataset.taskId;
                this.openTaskInGraph(taskId);
            });
        }
    }

    /**
     * Render preview (limited lines for sidebar)
     */
    renderPreview(fileData, maxLines = 10) {
        const fileType = this.selectedFile.file_type;
        const lines = fileData.content.split('\n');
        const previewContent = lines.slice(0, maxLines).join('\n');
        const truncated = lines.length > maxLines;

        if (fileType === 'markdown') {
            const htmlContent = this.renderMarkdown(previewContent);
            return `
                <div class="markdown-content preview-truncated">
                    ${htmlContent}
                    ${truncated ? '<p class="preview-note">... click "View Full Document" to see more</p>' : ''}
                </div>
            `;
        } else if (fileType === 'json') {
            try {
                const jsonObj = JSON.parse(fileData.content);
                const prettyJson = JSON.stringify(jsonObj, null, 2);
                const previewJson = prettyJson.split('\n').slice(0, maxLines).join('\n');
                return `
                    <pre class="code-content preview-truncated">${this.escapeHtml(previewJson)}${truncated ? '\n...' : ''}</pre>
                `;
            } catch (e) {
                return `<pre class="text-content preview-truncated">${this.escapeHtml(previewContent)}${truncated ? '\n...' : ''}</pre>`;
            }
        } else {
            return `<pre class="text-content preview-truncated">${this.escapeHtml(previewContent)}${truncated ? '\n...' : ''}</pre>`;
        }
    }

    /**
     * Attach sidebar action handlers
     */
    attachSidebarActionHandlers(fileData) {
        const viewFullBtn = document.getElementById('viewFullBtn');
        const downloadBtn = document.getElementById('downloadFileBtn');
        const copyBtn = document.getElementById('copyContentBtn');

        if (viewFullBtn) {
            viewFullBtn.addEventListener('click', () => this.openFileModal(fileData));
        }

        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadFile(fileData));
        }

        if (copyBtn) {
            copyBtn.addEventListener('click', () => this.copyContent(fileData.content));
        }
    }

    /**
     * Open file in modal for full view
     */
    openFileModal(fileData) {
        const modal = document.getElementById('workspaceFileModal');
        const title = document.getElementById('workspaceModalTitle');
        const content = document.getElementById('workspaceModalContent');
        const downloadBtn = document.getElementById('downloadFromModalBtn');

        if (!modal || !title || !content) return;

        // Set title
        title.textContent = this.selectedFile.name;

        // Render full content
        const fullHtml = this.renderFullContent(fileData);
        content.innerHTML = fullHtml;

        // Show modal
        modal.classList.remove('hidden');

        // Attach download handler
        if (downloadBtn) {
            downloadBtn.onclick = () => this.downloadFile(fileData);
        }

        // Re-render icons
        if (typeof lucide !== 'undefined' && lucide.createIcons) {
            lucide.createIcons();
        }
    }

    /**
     * Close file modal
     */
    closeFileModal() {
        const modal = document.getElementById('workspaceFileModal');
        if (modal) {
            modal.classList.add('hidden');
        }
    }

    /**
     * Render full content for modal
     */
    renderFullContent(fileData) {
        const fileType = this.selectedFile.file_type;

        if (fileType === 'markdown') {
            const htmlContent = this.renderMarkdown(fileData.content);
            return `<div class="markdown-content">${htmlContent}</div>`;
        } else if (fileType === 'json') {
            try {
                const jsonObj = JSON.parse(fileData.content);
                const prettyJson = JSON.stringify(jsonObj, null, 2);
                return `<pre class="code-content">${this.escapeHtml(prettyJson)}</pre>`;
            } catch (e) {
                return `<pre class="text-content">${this.escapeHtml(fileData.content)}</pre>`;
            }
        } else if (fileType === 'code') {
            return `<pre class="code-content">${this.escapeHtml(fileData.content)}</pre>`;
        } else {
            return `<pre class="text-content">${this.escapeHtml(fileData.content)}</pre>`;
        }
    }

    /**
     * Download file with correct filename
     */
    downloadFile(fileData) {
        const blob = new Blob([fileData.content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = this.selectedFile.name; // Use actual filename from database
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        console.log('[Workspace] Downloaded file:', this.selectedFile.name);
    }

    /**
     * Copy content to clipboard
     */
    async copyContent(content) {
        try {
            await navigator.clipboard.writeText(content);
            console.log('[Workspace] Content copied to clipboard');

            // Show toast if available
            if (typeof showToast === 'function') {
                showToast('Content copied to clipboard', 'success');
            }
        } catch (error) {
            console.error('[Workspace] Failed to copy content:', error);
        }
    }

    /**
     * Open task in graph view
     */
    openTaskInGraph(taskId) {
        console.log('[Workspace] Opening task in graph:', taskId);

        // Switch to graph view
        const graphTab = document.querySelector('[data-view="graph"]');
        if (graphTab) {
            graphTab.click();
        }

        // TODO: Highlight task node in graph
    }

    /**
     * Render markdown using marked.js
     */
    renderMarkdown(markdown) {
        if (typeof marked === 'undefined') {
            console.error('[Workspace] marked.js not loaded');
            throw new Error('marked.js library is required');
        }

        if (typeof marked.setOptions === 'function') {
            marked.setOptions({
                breaks: true,
                gfm: true,
                headerIds: false,
                mangle: false
            });
        }

        return marked.parse(markdown);
    }

    /**
     * Render empty state
     */
    renderEmptyState() {
        const container = document.getElementById('workspaceFilesGrid');

        if (!container) return;

        container.innerHTML = `
            <div class="workspace-empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <h3>No Workspace Files</h3>
                <p>Files created by agents during task execution will appear here</p>
            </div>
        `;
    }

    /**
     * Render error state
     */
    renderError(message) {
        const container = document.getElementById('workspaceFilesGrid');

        if (!container) return;

        container.innerHTML = `
            <div class="workspace-empty-state">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <h3>Error</h3>
                <p>${this.escapeHtml(message)}</p>
            </div>
        `;
    }

    /**
     * Get file icon class
     */
    getFileIconClass(fileType) {
        return fileType;
    }

    /**
     * Get file icon
     */
    getFileIcon(fileType) {
        const icons = {
            'markdown': '<i data-lucide="file-text"></i>',
            'json': '<i data-lucide="braces"></i>',
            'code': '<i data-lucide="code"></i>',
            'text': '<i data-lucide="file"></i>',
            'unknown': '<i data-lucide="file-question"></i>'
        };

        return icons[fileType] || icons['unknown'];
    }

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    /**
     * Format date
     */
    formatDate(dateString) {
        if (!dateString) return 'Unknown';

        try {
            const date = new Date(dateString);
            return date.toLocaleString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch (e) {
            return dateString;
        }
    }

    /**
     * Escape HTML
     */
    escapeHtml(text) {
        if (typeof text !== 'string') return '';

        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Export for use in main.js
window.WorkspaceView = WorkspaceView;
