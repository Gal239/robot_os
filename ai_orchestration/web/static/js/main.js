/*
 * Main Application Logic
 */

// Application state
const AppState = {
    currentSessionId: null,
    sessionData: null,
    graphData: null,
    selectedTaskId: null,
    masterLog: [],
    metaLog: [],
    currentView: 'graph'
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 AI Orchestration Dashboard loaded');

    // Initialize components
    initializeEventListeners();
    await loadSessions();
    checkHealth();
});

// Initialize event listeners
function initializeEventListeners() {
    // Session selector
    const sessionSelector = document.getElementById('sessionSelector');
    if (sessionSelector) {
        sessionSelector.addEventListener('change', handleSessionChange);
    }

    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', handleRefresh);
    }

    // Settings button
    const settingsBtn = document.getElementById('settingsBtn');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', handleSettings);
    }

    // Graph controls
    const fitScreenBtn = document.getElementById('fitScreenBtn');
    if (fitScreenBtn) {
        fitScreenBtn.addEventListener('click', () => {
            if (window.GraphViz) window.GraphViz.fitToScreen();
        });
    }

    const zoomInBtn = document.getElementById('zoomInBtn');
    if (zoomInBtn) {
        zoomInBtn.addEventListener('click', () => {
            if (window.GraphViz) window.GraphViz.zoomIn();
        });
    }

    const zoomOutBtn = document.getElementById('zoomOutBtn');
    if (zoomOutBtn) {
        zoomOutBtn.addEventListener('click', () => {
            if (window.GraphViz) window.GraphViz.zoomOut();
        });
    }

    // View tabs
    const viewTabs = document.querySelectorAll('#viewTabs .view-tab-btn');
    viewTabs.forEach(tab => {
        tab.addEventListener('click', () => handleViewTabChange(tab));
    });

    // Master Log search
    const masterLogSearch = document.getElementById('masterLogSearch');
    if (masterLogSearch) {
        masterLogSearch.addEventListener('input', debounce(handleMasterLogSearch, 300));
    }

    // Meta Log search
    const metaLogSearch = document.getElementById('metaLogSearch');
    if (metaLogSearch) {
        metaLogSearch.addEventListener('input', debounce(handleMetaLogSearch, 300));
    }

    // Export buttons
    const exportMasterLogBtn = document.getElementById('exportMasterLogBtn');
    if (exportMasterLogBtn) {
        exportMasterLogBtn.addEventListener('click', () => handleExportLogs('master'));
    }


    const exportMetaLogBtn = document.getElementById('exportMetaLogBtn');
    if (exportMetaLogBtn) {
        exportMetaLogBtn.addEventListener('click', () => handleExportLogs('metalog'));
    }

    // Workspace controls
    const refreshWorkspaceBtn = document.getElementById('refreshWorkspaceBtn');
    if (refreshWorkspaceBtn) {
        refreshWorkspaceBtn.addEventListener('click', () => {
            if (window.workspaceViewInstance && AppState.currentSessionId) {
                window.workspaceViewInstance.init(AppState.currentSessionId);
            }
        });
    }

    const closeSidebarBtn = document.getElementById('closeSidebarBtn');
    if (closeSidebarBtn) {
        closeSidebarBtn.addEventListener('click', () => {
            if (window.workspaceViewInstance) {
                window.workspaceViewInstance.hideSidebar();
            }
        });
    }
}

// Load available sessions
async function loadSessions() {
    try {
        showLoading('Loading sessions...');
        const data = await API.listSessions();
        const sessions = data.sessions || [];

        const sessionSelector = document.getElementById('sessionSelector');
        if (!sessionSelector) return;

        // Clear existing options
        sessionSelector.innerHTML = '';

        if (sessions.length === 0) {
            sessionSelector.innerHTML = '<option value="">No sessions found</option>';
            hideLoading();
            return;
        }

        // Add sessions to selector
        sessions.forEach(session => {
            const option = document.createElement('option');
            option.value = session.session_id;
            option.textContent = `${session.session_id} (${session.task_count} tasks, ${session.status})`;
            sessionSelector.appendChild(option);
        });

        // Load first session by default
        if (sessions.length > 0) {
            sessionSelector.value = sessions[0].session_id;
            await loadSession(sessions[0].session_id);
        }

        hideLoading();
        showToast('Sessions loaded successfully', 'success');
    } catch (error) {
        console.error('Error loading sessions:', error);
        hideLoading();
        showToast('Failed to load sessions', 'error');
    }
}

// Handle session change
async function handleSessionChange(event) {
    const sessionId = event.target.value;
    if (!sessionId) return;
    await loadSession(sessionId);
}

// Load session data
async function loadSession(sessionId) {
    try {
        showLoading(`Loading session ${sessionId}...`);

        // CLEAR ALL CACHES to ensure fresh data
        cache.clear();
        console.log(`Loading session ${sessionId} with fresh data (cache cleared)`);

        // Destroy existing graph to force complete rebuild
        if (window.GraphViz) {
            window.GraphViz.destroy();
        }

        // Load full session data
        const sessionData = await API.getSession(sessionId);
        AppState.currentSessionId = sessionId;
        AppState.sessionData = sessionData;

        // Update status indicator
        updateStatusIndicator(sessionData.metadata);

        // Load and render graph
        const graphData = await API.getSessionGraph(sessionId);
        AppState.graphData = graphData;
        console.log(`Loaded graph data:`, graphData);

        // Update metric cards (pass graphData for tool usage) - AFTER graph loads
        updateMetricCards(sessionData, graphData);

        if (window.GraphViz) {
            window.GraphViz.render(graphData);
        }

        // Render Gantt chart
        if (window.GanttViz) {
            window.GanttViz.render(graphData);
        }

        // Load data for all views
        await loadAllViewData(sessionId);

        hideLoading();
        showToast('Session loaded successfully', 'success');
    } catch (error) {
        console.error('Error loading session:', error);
        hideLoading();
        showToast(`Failed to load session: ${error.message}`, 'error');
    }
}

// Load data for all views
async function loadAllViewData(sessionId) {
    // Load master log for both Timeline and Master Log views
    try {
        const masterLogData = await API.getSessionLogs(sessionId, 'master');
        AppState.masterLog = masterLogData.logs || [];

        // Render if Master Log view is active
        if (document.getElementById('masterLogView').classList.contains('active')) {
            renderMasterLog(AppState.masterLog);
        }

        // Render if Timeline view is active
        if (document.getElementById('timelineView').classList.contains('active')) {
            renderTimeline(AppState.masterLog);
        }
    } catch (error) {
        console.error('Error loading master log:', error);
        AppState.masterLog = [];
    }

    // Load meta log
    try {
        const metaLogData = await API.getSessionLogs(sessionId, 'metalog_detailed');
        AppState.metaLog = metaLogData.logs || [];

        // Render if Meta Log view is active
        if (document.getElementById('metaLogView').classList.contains('active')) {
            renderMetaLog(AppState.metaLog);
        }
    } catch (error) {
        console.error('Error loading meta log:', error);
        AppState.metaLog = [];
    }
}

// Update status indicator
function updateStatusIndicator(metadata) {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');

    if (!statusIndicator || !statusText) return;

    const hasRunning = metadata.running > 0;
    const status = hasRunning ? 'running' : 'completed';

    statusIndicator.className = `status-indicator ${status}`;
    statusText.textContent = hasRunning ? 'Running' : 'Completed';
}

// Handle refresh
async function handleRefresh() {
    if (!AppState.currentSessionId) {
        showToast('No session selected', 'warning');
        return;
    }

    cache.clear();
    await loadSession(AppState.currentSessionId);
}

// Handle settings
function handleSettings() {
    showToast('Settings coming soon!', 'info');
}

// Handle view tab change
function handleViewTabChange(tab) {
    // Update active tab
    document.querySelectorAll('#viewTabs .view-tab-btn').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');

    const viewName = tab.dataset.view;
    AppState.currentView = viewName;

    // Hide all views
    document.querySelectorAll('.view-container').forEach(v => v.classList.remove('active'));

    // Show selected view
    const viewMap = {
        'graph': 'graphView',
        'timeline': 'timelineView',
        'gantt': 'ganttView',
        'master-log': 'masterLogView',
        'meta-log': 'metaLogView',
        'workspace': 'workspaceView'
    };

    const viewId = viewMap[viewName];
    const viewContainer = document.getElementById(viewId);
    if (viewContainer) {
        viewContainer.classList.add('active');

        // Render view content if data is available
        if (viewName === 'timeline' && AppState.masterLog.length > 0) {
            renderTimeline(AppState.masterLog);
        } else if (viewName === 'gantt' && AppState.graphData) {
            if (window.GanttViz) {
                window.GanttViz.render(AppState.graphData);
            }
        } else if (viewName === 'master-log' && AppState.masterLog.length > 0) {
            renderMasterLog(AppState.masterLog);
        } else if (viewName === 'meta-log' && AppState.metaLog.length > 0) {
            renderMetaLog(AppState.metaLog);
        } else if (viewName === 'workspace' && AppState.currentSessionId) {
            // Initialize workspace view
            if (window.WorkspaceView) {
                if (!window.workspaceViewInstance) {
                    window.workspaceViewInstance = new WorkspaceView();
                }
                window.workspaceViewInstance.init(AppState.currentSessionId);
            }
        }
    }

    // Re-render icons
    lucide.createIcons();
}

// Get tool type information (icon, color, display name)
function getToolTypeInfo(toolName) {
    const toolLower = (toolName || '').toLowerCase();

    if (toolLower.includes('route_to_') || toolLower.includes('ask_claude') || toolLower.includes('ask_gpt')) {
        return {
            type: 'agent_as_tool',
            icon: 'users',
            badgeClass: 'agent-delegation',
            displayName: 'Agent Delegation',
            description: 'Delegating work to another agent',
            color: '#f59e0b'
        };
    } else if (toolLower.includes('handoff')) {
        return {
            type: 'handoff',
            icon: 'check-circle',
            badgeClass: 'handoff',
            displayName: 'Task Handoff',
            description: 'Returning result to parent agent',
            color: '#10b981'
        };
    } else if (toolLower.includes('ask_master') || toolLower.includes('ask_data')) {
        return {
            type: 'ask_master',
            icon: 'help-circle',
            badgeClass: 'ask-master',
            displayName: 'Question to Master',
            description: 'Asking master agent for clarification',
            color: '#ef4444'
        };
    } else {
        return {
            type: 'function_call',
            icon: 'zap',
            badgeClass: 'function-call',
            displayName: 'Function Call',
            description: 'Regular function execution',
            color: '#3b82f6'
        };
    }
}

// Show tool details in sidebar with professional UI (works for all views)
function showToolDetailsInSidebar(logEntry, sidebarId) {
    const sidebar = document.getElementById(sidebarId);
    if (!sidebar) return;

    const toolInfo = getToolTypeInfo(logEntry.tool);
    const hasInput = logEntry.input && Object.keys(logEntry.input).length > 0;
    const hasResult = logEntry.result && logEntry.result !== '' && logEntry.result !== 'undefined';

    const inputJson = hasInput ? formatJSON(logEntry.input) : '{}';
    const resultJson = hasResult ? formatJSON(logEntry.result) : 'No result';
    const inputSize = hasInput ? JSON.stringify(logEntry.input).length : 0;
    const resultSize = hasResult ? JSON.stringify(logEntry.result).length : 0;

    sidebar.innerHTML = `
        <div class="sidebar-header">
            <div class="tool-type-header">
                <div class="tool-icon" style="background-color: ${toolInfo.color}20; color: ${toolInfo.color};">
                    <i data-lucide="${toolInfo.icon}"></i>
                </div>
                <div class="tool-info">
                    <h3 class="sidebar-title">${logEntry.tool || 'Tool Call'}</h3>
                    <span class="tool-badge ${toolInfo.badgeClass}">${toolInfo.displayName}</span>
                </div>
            </div>
            <div class="sidebar-controls">
                <button class="sidebar-control-btn" onclick="SidebarManager.toggleMaximize()" title="Maximize/Minimize">
                    <i data-lucide="maximize-2" class="maximize-icon"></i>
                    <i data-lucide="minimize-2" class="minimize-icon" style="display: none;"></i>
                </button>
            </div>
        </div>
        <div class="sidebar-content">
            <!-- Overview Section -->
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
                            <span class="info-label">Tool Type</span>
                            <span class="badge badge-${toolInfo.badgeClass}">${toolInfo.displayName}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Task ID</span>
                            <span class="info-value copyable" onclick="copyToClipboard('${logEntry.task_id}')" title="Click to copy">${logEntry.task_id}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Agent</span>
                            <span class="info-value">${logEntry.agent_id || 'N/A'}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Timestamp</span>
                            <span class="info-value">${formatAbsoluteTime(logEntry.timestamp)}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Input Section -->
            ${hasInput ? `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="download"></i>
                        Input
                    </h4>
                </div>
                <div class="section-content">
                    <details open>
                        <summary class="details-summary">
                            <span>Show Input</span>
                            <span class="text-muted">(${inputSize} chars)</span>
                        </summary>
                        <pre class="json-viewer">${inputJson}</pre>
                    </details>
                </div>
            </div>
            ` : ''}

            <!-- Result Section -->
            ${hasResult ? `
            <div class="sidebar-section">
                <div class="section-header">
                    <h4 class="section-title">
                        <i data-lucide="upload"></i>
                        Result
                    </h4>
                </div>
                <div class="section-content">
                    <details>
                        <summary class="details-summary">
                            <span>Show Result</span>
                            <span class="text-muted">(${resultSize} chars)</span>
                        </summary>
                        <pre class="json-viewer">${resultJson}</pre>
                    </details>
                </div>
            </div>
            ` : ''}
        </div>
    `;

    lucide.createIcons();
}

// Render Timeline View
function renderTimeline(logs) {
    const timelineContent = document.getElementById('timelineContent');
    if (!timelineContent) return;

    if (!logs || logs.length === 0) {
        timelineContent.innerHTML = `
            <div class="view-empty-state">
                <i data-lucide="clock"></i>
                <p>No timeline events available</p>
            </div>
        `;
        lucide.createIcons();
        return;
    }

    // Helper to get tool type class and icon
    const getToolInfo = (tool) => {
        const toolLower = (tool || '').toLowerCase();
        if (toolLower.includes('ask_data') || toolLower.includes('ask_master')) {
            return { class: 'ask-master', icon: 'fa-question-circle', label: tool };
        } else if (toolLower.includes('handoff')) {
            return { class: 'handoff', icon: 'fa-check-circle', label: tool };
        } else if (toolLower.includes('ask_claude') || toolLower.includes('ask_gpt')) {
            return { class: 'agent-delegation', icon: 'fa-robot', label: tool };
        } else {
            return { class: 'function-call', icon: 'fa-cog', label: tool };
        }
    };

    // Create vertical timeline
    timelineContent.innerHTML = `
        <div class="timeline-view">
            ${logs.map((log, index) => {
                const toolInfo = getToolInfo(log.tool);
                return `
                    <div class="timeline-item ${toolInfo.class}" data-log-index="${index}" style="cursor: pointer;" title="Click to view tool details">
                        <div class="timeline-marker"></div>
                        <div class="timeline-content-card">
                            <div class="timeline-time">${formatAbsoluteTime(log.timestamp)}</div>
                            <div class="timeline-title">
                                <div class="timeline-icon">
                                    <i class="fas ${toolInfo.icon}"></i>
                                </div>
                                <span class="tool-badge ${toolInfo.class}">${toolInfo.label}</span>
                                <span class="text-mono text-muted">${truncate(log.task_id, 12)}</span>
                            </div>
                            <div class="timeline-details">
                                <strong>Agent:</strong> ${log.agent_id || '-'}<br>
                                <strong>Input:</strong> ${truncate(JSON.stringify(log.input), 80)}<br>
                                <strong>Result:</strong> ${truncate(String(log.result), 80)}
                            </div>
                        </div>
                    </div>
                `;
            }).join('')}
        </div>
    `;

    // Add click handlers to timeline items
    timelineContent.querySelectorAll('.timeline-item').forEach((item, index) => {
        item.addEventListener('click', () => {
            showToolDetailsInSidebar(logs[index], 'timelineSidebar');
        });
    });

    lucide.createIcons();
}

// Render Master Log View
function renderMasterLog(logs) {
    const masterLogContent = document.getElementById('masterLogContent');
    if (!masterLogContent) return;

    if (!logs || logs.length === 0) {
        masterLogContent.innerHTML = `
            <div class="view-empty-state">
                <i data-lucide="file-text"></i>
                <p>No master log entries available</p>
            </div>
        `;
        lucide.createIcons();
        return;
    }

    renderLogsTable(logs, masterLogContent, 'masterLogSidebar');
}

// Render Meta Log View
function renderMetaLog(logs) {
    const metaLogContent = document.getElementById('metaLogContent');
    if (!metaLogContent) return;

    if (!logs || logs.length === 0) {
        metaLogContent.innerHTML = `
            <div class="view-empty-state">
                <i data-lucide="layers"></i>
                <p>No meta log entries available</p>
            </div>
        `;
        lucide.createIcons();
        return;
    }

    renderLogsTable(logs, metaLogContent, 'metaLogSidebar');
}

// Render logs table (reusable)
function renderLogsTable(logs, container, sidebarId) {
    // Helper to get tool badge class
    const getToolBadgeClass = (tool) => {
        const toolLower = (tool || '').toLowerCase();
        if (toolLower.includes('ask_data') || toolLower.includes('ask_master')) {
            return 'ask-master';
        } else if (toolLower.includes('handoff')) {
            return 'handoff';
        } else if (toolLower.includes('ask_claude') || toolLower.includes('ask_gpt')) {
            return 'agent-delegation';
        } else {
            return 'function-call';
        }
    };

    // Create table
    const table = document.createElement('table');
    table.className = 'logs-table';

    // Table header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Timestamp</th>
            <th>Task ID</th>
            <th>Agent</th>
            <th>Tool</th>
            <th>Input</th>
            <th>Result</th>
        </tr>
    `;
    table.appendChild(thead);

    // Table body
    const tbody = document.createElement('tbody');
    logs.forEach(log => {
        const row = document.createElement('tr');
        const badgeClass = getToolBadgeClass(log.tool);
        row.innerHTML = `
            <td class="text-mono">${formatAbsoluteTime(log.timestamp)}</td>
            <td class="text-mono">${truncate(log.task_id, 15)}</td>
            <td>${log.agent_id || '-'}</td>
            <td><span class="tool-badge ${badgeClass}">${log.tool}</span></td>
            <td class="text-mono">${truncate(JSON.stringify(log.input), 50)}</td>
            <td class="text-mono">${truncate(String(log.result), 50)}</td>
        `;
        row.style.cursor = 'pointer';
        row.addEventListener('click', () => {
            showToolDetailsInSidebar(log, sidebarId);
        });
        tbody.appendChild(row);
    });
    table.appendChild(tbody);

    container.innerHTML = '';
    container.appendChild(table);
}

// Handle master log search
function handleMasterLogSearch(event) {
    const query = event.target.value.toLowerCase();
    const rows = document.querySelectorAll('#masterLogContent .logs-table tbody tr');

    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
    });
}

// Handle meta log search
function handleMetaLogSearch(event) {
    const query = event.target.value.toLowerCase();
    const rows = document.querySelectorAll('#metaLogContent .logs-table tbody tr');

    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
    });
}

// Handle export logs
function handleExportLogs(logType) {
    let logs, filename;

    if (logType === 'master') {
        logs = AppState.masterLog;
        filename = `${AppState.currentSessionId}_master_log.json`;
    } else if (logType === 'metalog') {
        logs = AppState.metaLog;
        filename = `${AppState.currentSessionId}_meta_log.json`;
    }

    if (!logs || logs.length === 0) {
        showToast('No logs to export', 'warning');
        return;
    }

    exportJSON(logs, filename);
    showToast('Logs exported successfully', 'success');
}

// Load task details
async function loadTaskDetails(taskId, sessionId) {
    try {
        const taskData = await API.getTask(taskId, sessionId);
        AppState.selectedTaskId = taskId;

        // Render in the appropriate sidebar based on current view
        const currentView = AppState.currentView;

        if (currentView === 'gantt' && window.GanttSidebarManager) {
            window.GanttSidebarManager.renderTaskDetails(taskData);
        } else if (window.SidebarManager) {
            window.SidebarManager.renderTaskDetails(taskData);
        }

        // Highlight node in graph if on graph view
        if (currentView === 'graph' && window.GraphViz) {
            window.GraphViz.selectNode(taskId);
        }
    } catch (error) {
        console.error('Error loading task details:', error);
        showToast('Failed to load task details', 'error');
    }
}

// Load agent details
async function loadAgentDetails(agentId) {
    try {
        // Get agent data from current session
        const agentData = AppState.sessionData?.agents?.[agentId];

        if (!agentData) {
            console.error('Agent not found:', agentId);
            showToast('Agent details not available', 'error');
            return;
        }

        AppState.selectedAgentId = agentId;

        // Render in the appropriate sidebar based on current view
        const currentView = AppState.currentView;

        if (currentView === 'gantt' && window.GanttSidebarManager) {
            window.GanttSidebarManager.renderAgentDetails(agentData);
        } else if (window.SidebarManager) {
            window.SidebarManager.renderAgentDetails(agentData);
        }

        // Highlight agents in the agent activity card
        const agentItems = document.querySelectorAll('.agent-item');
        agentItems.forEach(item => {
            const itemAgentName = item.querySelector('.agent-name')?.textContent;
            if (itemAgentName === agentId) {
                item.style.backgroundColor = 'var(--color-light-dark)';
                item.style.borderLeft = '3px solid var(--color-primary)';
            } else {
                item.style.backgroundColor = 'var(--color-light)';
                item.style.borderLeft = 'none';
            }
        });

    } catch (error) {
        console.error('Error loading agent details:', error);
        showToast('Failed to load agent details', 'error');
    }
}

// Make loadAgentDetails globally available
window.loadAgentDetails = loadAgentDetails;

// Health check
async function checkHealth() {
    try {
        await API.healthCheck();
        console.log('✅ API health check passed');
    } catch (error) {
        console.error('❌ API health check failed:', error);
        showToast('API connection failed', 'error');
    }
}

// Auto-refresh (optional)
let autoRefreshInterval = null;

function startAutoRefresh(intervalMs = 5000) {
    stopAutoRefresh();
    autoRefreshInterval = setInterval(() => {
        if (AppState.currentSessionId) {
            loadSession(AppState.currentSessionId);
        }
    }, intervalMs);
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
}
