/*
 * Detail Sidebar Component
 */

class SidebarManager {
    constructor(sidebarElement) {
        this.sidebar = sidebarElement;
    }

    renderTaskDetails(taskData) {
        if (!this.sidebar) return;

        this.sidebar.innerHTML = `
            <div class="sidebar-header">
                <h3 class="sidebar-title">Task Details</h3>
                <div class="sidebar-controls">
                    <button class="sidebar-control-btn" onclick="SidebarManager.toggleMaximize()" title="Maximize/Minimize">
                        <i data-lucide="maximize-2" class="maximize-icon"></i>
                        <i data-lucide="minimize-2" class="minimize-icon" style="display: none;"></i>
                    </button>
                    <button class="sidebar-close" onclick="SidebarManager.close()">
                        <i data-lucide="x"></i>
                    </button>
                </div>
            </div>
            <div class="sidebar-content">
                ${this.renderOverviewSection(taskData)}
                ${this.renderChildrenTimelineSection(taskData)}
                ${this.renderTimelineSection(taskData)}
                ${this.renderRelationshipsSection(taskData)}
                ${this.renderMessagesSection(taskData)}
                ${this.renderDocumentsSection(taskData)}
            </div>
        `;

        lucide.createIcons();
        this.attachEventListeners();
    }

    renderOverviewSection(taskData) {
        return `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="info"></i>
                        Overview
                    </h4>
                </div>
                <div class="section-content">
                    <div class="task-info">
                        <div class="info-row">
                            <span class="info-label">Task ID</span>
                            <span class="info-value copyable" onclick="copyToClipboard('${taskData.task_id}')">${taskData.task_id}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Agent</span>
                            <span class="info-value">${taskData.agent_id}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Status</span>
                            <span class="badge badge-${this.getStatusBadgeClass(taskData.status)}">${taskData.status}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Created</span>
                            <span class="info-value">${formatAbsoluteTime(taskData.created_at)}</span>
                        </div>
                        ${taskData.completed_at ? `
                        <div class="info-row">
                            <span class="info-label">Completed</span>
                            <span class="info-value">${formatAbsoluteTime(taskData.completed_at)}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Duration</span>
                            <span class="info-value">${this.calculateDuration(taskData.created_at, taskData.completed_at)}</span>
                        </div>
                        ` : taskData.status === 'running' ? `
                        <div class="info-row">
                            <span class="info-label">Elapsed Time</span>
                            <span class="info-value">${this.calculateDuration(taskData.created_at, new Date())}</span>
                        </div>
                        ` : ''}
                    </div>
                    <div class="task-payload">
                        ${taskData.task_payload || 'No task description'}
                    </div>
                </div>
            </div>
        `;
    }

    renderChildrenTimelineSection(taskData) {
        // Get children from AppState.sessionData
        if (!AppState.sessionData || !AppState.sessionData.graph) return '';

        const allNodes = AppState.sessionData.graph.nodes || {};
        const children = Object.values(allNodes).filter(
            node => node.parent_task_id === taskData.task_id
        );

        if (children.length === 0) return '';

        // Sort by created_at
        children.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));

        // Calculate time range for scaling
        const timestamps = children.map(c => new Date(c.created_at)).filter(d => !isNaN(d));
        if (timestamps.length === 0) return '';

        const startTime = Math.min(...timestamps);
        const endTimes = children.map(c => {
            const end = c.completed_at ? new Date(c.completed_at) : new Date();
            return isNaN(end) ? new Date() : end;
        });
        const endTime = Math.max(...endTimes, startTime + 1000); // Ensure at least 1 second range

        const timeRange = endTime - startTime || 1000; // Prevent division by zero

        const childrenHTML = children.map(child => {
            const start = new Date(child.created_at);
            const end = child.completed_at ? new Date(child.completed_at) : new Date();
            const duration = end - start;
            const offset = ((start - startTime) / timeRange) * 100;
            const width = (duration / timeRange) * 100;

            return `
                <div class="mini-gantt-row" onclick="loadTaskDetails('${child.task_id}', AppState.currentSessionId)">
                    <span class="mini-gantt-label">${child.task_id}</span>
                    <div class="mini-gantt-track">
                        <div class="mini-gantt-bar status-${child.status}"
                             style="left: ${offset}%; width: ${Math.max(width, 2)}%">
                        </div>
                    </div>
                    <span class="mini-gantt-duration">${formatDuration(duration)}</span>
                </div>
            `;
        }).join('');

        return `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="git-branch"></i>
                        Children Timeline (${children.length})
                    </h4>
                </div>
                <div class="section-content">
                    <div class="mini-gantt">
                        ${childrenHTML}
                    </div>
                </div>
            </div>
        `;
    }

    renderTimelineSection(taskData) {
        const timeline = taskData.tool_timeline || [];

        // Filter out "result from" entries
        const filteredTimeline = timeline.filter(event => {
            const toolName = (event.tool || '').toLowerCase();
            return !toolName.includes('result from') && !toolName.includes('← result from');
        });

        if (filteredTimeline.length === 0) {
            return `
                <div class="sidebar-section">
                    <div class="section-header">
                        <h4 class="section-title">
                            <i data-lucide="clock"></i>
                            Timeline
                        </h4>
                    </div>
                    <div class="section-content">
                        <p class="text-muted">No tool calls yet</p>
                    </div>
                </div>
            `;
        }

        const timelineHTML = filteredTimeline.map((event, index) => `
            <div class="timeline-event">
                <div class="timeline-marker"></div>
                <div class="timeline-header">
                    <div class="timeline-tool">
                        <i data-lucide="${getToolIcon(event.type)}"></i>
                        ${event.tool}
                    </div>
                    <div class="timeline-time">${formatRelativeTime(event.timestamp)}</div>
                </div>
                <div class="timeline-details">
                    <details>
                        <summary>Input</summary>
                        <pre class="json-viewer">${formatJSON(event.input)}</pre>
                    </details>
                    <details>
                        <summary>Result</summary>
                        <pre class="json-viewer">${formatJSON(event.result)}</pre>
                    </details>
                </div>
            </div>
        `).join('');

        return `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="clock"></i>
                        Timeline (${filteredTimeline.length})
                    </h4>
                </div>
                <div class="section-content">
                    <div class="timeline">
                        ${timelineHTML}
                    </div>
                </div>
            </div>
        `;
    }

    renderRelationshipsSection(taskData) {
        const relationships = [];

        if (taskData.parent_task_id) {
            relationships.push({
                label: 'Parent Task',
                value: taskData.parent_task_id,
                icon: 'arrow-up'
            });
        }

        if (taskData.master_agent_id) {
            relationships.push({
                label: 'Master Agent',
                value: taskData.master_agent_id,
                icon: 'user'
            });
        }

        if (taskData.children && taskData.children.length > 0) {
            taskData.children.forEach(childId => {
                relationships.push({
                    label: 'Child Task',
                    value: childId,
                    icon: 'arrow-down'
                });
            });
        }

        if (taskData.blockers && taskData.blockers.length > 0) {
            taskData.blockers.forEach(blockerId => {
                relationships.push({
                    label: 'Blocked By',
                    value: blockerId,
                    icon: 'lock'
                });
            });
        }

        if (relationships.length === 0) {
            return `
                <div class="sidebar-section">
                    <div class="section-header">
                        <h4 class="section-title">
                            <i data-lucide="git-branch"></i>
                            Relationships
                        </h4>
                    </div>
                    <div class="section-content">
                        <p class="text-muted">No relationships</p>
                    </div>
                </div>
            `;
        }

        const relationshipsHTML = relationships.map(rel => `
            <div class="relationship-item" onclick="loadTaskDetails('${rel.value}', AppState.currentSessionId)">
                <i data-lucide="${rel.icon}" class="relationship-icon"></i>
                <span class="relationship-label">${rel.label}:</span>
                <span class="relationship-value">${rel.value}</span>
            </div>
        `).join('');

        return `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="git-branch"></i>
                        Relationships
                    </h4>
                </div>
                <div class="section-content">
                    <div class="relationships">
                        ${relationshipsHTML}
                    </div>
                </div>
            </div>
        `;
    }

    renderMessagesSection(taskData) {
        const messages = taskData.messages || [];

        if (messages.length === 0) {
            return '';
        }

        const messagesHTML = messages.map(msg => `
            <div class="message-item">
                <strong>${msg.role}:</strong>
                <pre>${msg.content}</pre>
            </div>
        `).join('');

        return `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="message-square"></i>
                        Messages
                    </h4>
                </div>
                <div class="section-content">
                    ${messagesHTML}
                </div>
            </div>
        `;
    }

    renderDocumentsSection(taskData) {
        const documents = taskData.documents || [];

        if (documents.length === 0) {
            return '';
        }

        const documentsHTML = documents.map(doc => `
            <div class="document-item">
                <div class="document-name">
                    <i data-lucide="file"></i>
                    ${doc}
                </div>
                <button class="document-download" title="Download">
                    <i data-lucide="download"></i>
                </button>
            </div>
        `).join('');

        return `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="paperclip"></i>
                        Documents (${documents.length})
                    </h4>
                </div>
                <div class="section-content">
                    <div class="documents">
                        ${documentsHTML}
                    </div>
                </div>
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

    calculateDuration(startTime, endTime) {
        if (!startTime) return '-';
        const start = new Date(startTime);
        const end = endTime instanceof Date ? endTime : new Date(endTime);
        const duration = end - start;
        return formatDuration(duration);
    }

    renderAgentDetails(agentData) {
        if (!this.sidebar) return;

        this.sidebar.innerHTML = `
            <div class="sidebar-header">
                <h3 class="sidebar-title">Agent Details</h3>
                <div class="sidebar-controls">
                    <button class="sidebar-control-btn" onclick="SidebarManager.toggleMaximize()" title="Maximize/Minimize">
                        <i data-lucide="maximize-2" class="maximize-icon"></i>
                        <i data-lucide="minimize-2" class="minimize-icon" style="display: none;"></i>
                    </button>
                    <button class="sidebar-close" onclick="SidebarManager.close()">
                        <i data-lucide="x"></i>
                    </button>
                </div>
            </div>
            <div class="sidebar-content">
                ${this.renderAgentOverview(agentData)}
                ${this.renderAgentSessionStats(agentData)}
                ${this.renderAgentInstructions(agentData)}
                ${this.renderAgentTools(agentData)}
                ${this.renderAgentTasks(agentData)}
            </div>
        `;

        lucide.createIcons();
        this.attachEventListeners();
    }

    renderAgentOverview(agentData) {
        const modelClass = getModelClass(agentData.force_model || 'unknown');
        const modelDisplay = getModelDisplay(agentData.force_model || 'unknown');

        return `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="info"></i>
                        Overview
                    </h4>
                </div>
                <div class="section-content">
                    <div class="agent-overview">
                        <div class="info-row">
                            <span class="info-label">Agent ID</span>
                            <span class="info-value copyable" onclick="copyToClipboard('${agentData.agent_id}')">${agentData.agent_id}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Model</span>
                            <span class="agent-model-badge-large model-${modelClass}">${modelDisplay}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Description</span>
                            <span class="info-value">${agentData.description || 'No description'}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderAgentSessionStats(agentData) {
        const duration = agentData.total_duration > 0 ? formatDuration(agentData.total_duration * 1000) : '-';

        return `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="activity"></i>
                        Session Statistics
                    </h4>
                </div>
                <div class="section-content">
                    <div class="agent-stats-grid">
                        <div class="stat-item">
                            <span class="stat-label">Tasks</span>
                            <span class="stat-value">${agentData.task_count || 0}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Tool Calls</span>
                            <span class="stat-value">${agentData.tool_calls || 0}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Duration</span>
                            <span class="stat-value">${duration}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderAgentInstructions(agentData) {
        const instructions = agentData.instructions || 'No instructions provided';

        return `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="file-text"></i>
                        Instructions
                    </h4>
                </div>
                <div class="section-content">
                    <div class="agent-instructions">
                        <pre>${instructions}</pre>
                    </div>
                </div>
            </div>
        `;
    }

    renderAgentTools(agentData) {
        const tools = agentData.tools || [];

        if (tools.length === 0) {
            return `
                <div class="sidebar-section">
                    <div class="section-header">
                        <h4 class="section-title">
                            <i data-lucide="wrench"></i>
                            Tools (0)
                        </h4>
                    </div>
                    <div class="section-content">
                        <p class="text-muted">No tools configured</p>
                    </div>
                </div>
            `;
        }

        const toolsHTML = tools.map(tool => `
            <div class="tool-chip">${tool}</div>
        `).join('');

        return `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="wrench"></i>
                        Tools (${tools.length})
                    </h4>
                </div>
                <div class="section-content">
                    <div class="tools-grid">
                        ${toolsHTML}
                    </div>
                </div>
            </div>
        `;
    }

    renderAgentTasks(agentData) {
        // Get all tasks for this agent from AppState
        const nodes = AppState.sessionData?.graph?.nodes || {};
        const agentTasks = Object.values(nodes).filter(
            node => node.agent_id === agentData.agent_id
        );

        if (agentTasks.length === 0) {
            return `
                <div class="sidebar-section">
                    <div class="section-header">
                        <h4 class="section-title">
                            <i data-lucide="list"></i>
                            Tasks (0)
                        </h4>
                    </div>
                    <div class="section-content">
                        <p class="text-muted">No tasks</p>
                    </div>
                </div>
            `;
        }

        const tasksHTML = agentTasks.map(task => {
            const statusClass = this.getStatusBadgeClass(task.status);
            return `
                <div class="task-list-item" onclick="loadTaskDetails('${task.task_id}', AppState.currentSessionId)">
                    <div class="task-list-header">
                        <span class="task-list-id">${task.task_id}</span>
                        <span class="badge badge-${statusClass}">${task.status}</span>
                    </div>
                    <div class="task-list-payload">${task.task_payload || 'No description'}</div>
                </div>
            `;
        }).join('');

        return `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="list"></i>
                        Tasks (${agentTasks.length})
                    </h4>
                </div>
                <div class="section-content">
                    <div class="task-list">
                        ${tasksHTML}
                    </div>
                </div>
            </div>
        `;
    }

    attachEventListeners() {
        // Collapsible sections
        const sectionHeaders = this.sidebar.querySelectorAll('.section-header');
        sectionHeaders.forEach(header => {
            header.addEventListener('click', () => {
                header.classList.toggle('collapsed');
                const content = header.nextElementSibling;
                if (content) {
                    content.classList.toggle('collapsed');
                }
            });
        });
    }

    static close() {
        const sidebar = document.getElementById('detailSidebar');
        if (sidebar) {
            sidebar.innerHTML = `
                <div class="sidebar-empty">
                    <i data-lucide="info"></i>
                    <p>Click on a node to view details</p>
                </div>
            `;
            lucide.createIcons();
        }
    }

    static toggleMaximize() {
        // Find the content-grid container
        const activeView = document.querySelector('.view-container.active');

        if (activeView) {
            const contentGrid = activeView.querySelector('.content-grid');
            const sidebar = activeView.querySelector('.sidebar');
            const sidebarContent = activeView.querySelector('.sidebar-content');

            if (contentGrid) {
                const isMaximized = contentGrid.classList.toggle('sidebar-maximized');

                // Add animation classes
                if (sidebar) {
                    // Remove any existing animation classes
                    sidebar.classList.remove('sidebar-expanding', 'sidebar-collapsing');

                    // Add appropriate animation class
                    if (isMaximized) {
                        sidebar.classList.add('sidebar-expanding');
                    } else {
                        sidebar.classList.add('sidebar-collapsing');
                    }

                    // Remove animation class after completion
                    setTimeout(() => {
                        sidebar.classList.remove('sidebar-expanding', 'sidebar-collapsing');
                    }, 500);
                }

                // Fade content during transition
                if (sidebarContent) {
                    sidebarContent.classList.add('transitioning');
                    setTimeout(() => {
                        sidebarContent.classList.remove('transitioning');
                    }, 450);
                }

                // Toggle icon visibility with animation
                const maximizeIcon = activeView.querySelector('.maximize-icon');
                const minimizeIcon = activeView.querySelector('.minimize-icon');

                if (maximizeIcon && minimizeIcon) {
                    if (isMaximized) {
                        // Switching to minimize icon
                        maximizeIcon.style.display = 'none';
                        minimizeIcon.style.display = 'block';
                        minimizeIcon.classList.add('icon-rotate-in');
                        setTimeout(() => {
                            minimizeIcon.classList.remove('icon-rotate-in');
                        }, 400);
                    } else {
                        // Switching to maximize icon
                        minimizeIcon.style.display = 'none';
                        maximizeIcon.style.display = 'block';
                        maximizeIcon.classList.add('icon-rotate-in');
                        setTimeout(() => {
                            maximizeIcon.classList.remove('icon-rotate-in');
                        }, 400);
                    }
                }
            }
        }
    }
}

// Initialize global sidebar managers for different views
window.SidebarManager = new SidebarManager(document.getElementById('detailSidebar'));
window.GanttSidebarManager = new SidebarManager(document.getElementById('ganttSidebar'));
