/*
 * Metric Cards Component
 */

function updateMetricCards(sessionData, graphData) {
    if (!sessionData || !sessionData.metadata) return;

    const metadata = sessionData.metadata;
    const nodes = sessionData.graph?.nodes || {};

    // Update Task Statistics Card (includes session metrics)
    updateTaskStatsCard(metadata, sessionData);

    // Update Agent Activity Card
    updateAgentActivityCard(metadata, nodes);

    // Update Tool Usage Card (pass sessionData for nodes and graphData for edges)
    updateToolUsageCard(sessionData, graphData);

    // Update Recent Events Card
    updateRecentEventsCard(sessionData);
}

function updateTaskStatsCard(metadata, sessionData) {
    // Total tasks
    const totalTasksEl = document.getElementById('totalTasks');
    if (totalTasksEl) {
        totalTasksEl.textContent = metadata.total_tasks || 0;
    }

    // Completed
    const completedTasksEl = document.getElementById('completedTasks');
    if (completedTasksEl) {
        completedTasksEl.textContent = metadata.completed || 0;
    }

    // Running
    const runningTasksEl = document.getElementById('runningTasks');
    if (runningTasksEl) {
        runningTasksEl.textContent = metadata.running || 0;
    }

    // Waiting
    const waitingTasksEl = document.getElementById('waitingTasks');
    if (waitingTasksEl) {
        waitingTasksEl.textContent = metadata.waiting || 0;
    }

    // Ready
    const readyTasksEl = document.getElementById('readyTasks');
    if (readyTasksEl) {
        readyTasksEl.textContent = metadata.ready || 0;
    }

    // Progress bar
    const taskProgressEl = document.getElementById('taskProgress');
    if (taskProgressEl && metadata.total_tasks > 0) {
        const progress = (metadata.completed / metadata.total_tasks) * 100;
        taskProgressEl.style.width = `${progress}%`;
    }

    // Session metrics (merged from timeline stats)
    const nodes = sessionData.graph?.nodes || {};

    // Session duration (total work time = sum of all task durations)
    const sessionDurationEl = document.getElementById('sessionDurationCompact');
    if (sessionDurationEl) {
        let totalDuration = 0;

        for (const node of Object.values(nodes)) {
            if (node.completed_at && node.created_at) {
                const start = new Date(node.created_at);
                const end = new Date(node.completed_at);
                if (!isNaN(start) && !isNaN(end)) {
                    totalDuration += (end - start);
                }
            }
        }

        if (totalDuration > 0) {
            sessionDurationEl.textContent = formatDuration(totalDuration);
        } else {
            sessionDurationEl.textContent = '-';
        }
    }

    // Total tool calls
    const totalToolCallsEl = document.getElementById('totalToolCallsCompact');
    if (totalToolCallsEl) {
        let totalCalls = 0;
        for (const node of Object.values(nodes)) {
            // Filter out "result from" entries
            const timeline = node.tool_timeline || [];
            const filteredTimeline = timeline.filter(event => {
                const toolName = (event.tool || '').toLowerCase();
                return !toolName.includes('result from') && !toolName.includes('← result from');
            });
            totalCalls += filteredTimeline.length;
        }
        totalToolCallsEl.textContent = totalCalls;
    }

    // Average task time
    const avgTaskTimeEl = document.getElementById('avgTaskTimeCompact');
    if (avgTaskTimeEl) {
        const completedNodes = Object.values(nodes).filter(
            n => n.status === 'completed' && n.completed_at && n.created_at
        );

        if (completedNodes.length > 0) {
            const totalTime = completedNodes.reduce((sum, node) => {
                const start = new Date(node.created_at);
                const end = new Date(node.completed_at);
                return sum + (end - start);
            }, 0);
            const avgTime = totalTime / completedNodes.length;
            avgTaskTimeEl.textContent = formatDuration(avgTime);
        } else {
            avgTaskTimeEl.textContent = '-';
        }
    }
}

function updateAgentActivityCard(metadata, nodes) {
    const agentListEl = document.getElementById('agentList');
    if (!agentListEl) return;

    // Count tasks per agent
    const agentCounts = {};
    for (const node of Object.values(nodes)) {
        const agentId = node.agent_id;
        agentCounts[agentId] = (agentCounts[agentId] || 0) + 1;
    }

    // Update total agents
    const totalAgentsEl = document.getElementById('totalAgents');
    if (totalAgentsEl) {
        totalAgentsEl.textContent = metadata.agents?.length || 0;
    }

    // Render agent list
    agentListEl.innerHTML = '';
    if (metadata.agents && metadata.agents.length > 0) {
        metadata.agents.forEach(agentId => {
            // Get agent details from AppState if available
            const agentData = AppState.sessionData?.agents?.[agentId];
            const model = agentData?.force_model || 'unknown';
            const modelClass = getModelClass(model);
            const modelDisplay = getModelDisplay(model);

            const item = document.createElement('div');
            item.className = 'agent-item';
            item.innerHTML = `
                <i data-lucide="${agentId === 'human' ? 'user' : 'cpu'}" class="agent-icon"></i>
                <div class="agent-info">
                    <span class="agent-name">${agentId}</span>
                    <span class="agent-model-badge model-${modelClass}">${modelDisplay}</span>
                </div>
                <span class="agent-count">${agentCounts[agentId] || 0} tasks</span>
            `;

            // Make clickable to show agent details in sidebar
            item.style.cursor = 'pointer';
            item.onclick = () => {
                if (window.loadAgentDetails) {
                    window.loadAgentDetails(agentId);
                }
            };

            agentListEl.appendChild(item);
        });
        lucide.createIcons();
    } else {
        agentListEl.innerHTML = '<p class="text-muted">No agents</p>';
    }
}

// Helper function to get model CSS class
function getModelClass(model) {
    if (!model || model === 'unknown') return 'unknown';
    if (model.includes('gpt-5-nano')) return 'gpt-nano';
    if (model.includes('gpt-5-mini')) return 'gpt-mini';
    if (model.includes('gpt-5')) return 'gpt5';
    if (model.includes('sonnet')) return 'sonnet';
    return 'unknown';
}

// Helper function to get model display name
function getModelDisplay(model) {
    if (!model || model === 'unknown') return '?';
    if (model.includes('gpt-5-nano')) return 'GPT-5 Nano';
    if (model.includes('gpt-5-mini')) return 'GPT-5 Mini';
    if (model.includes('gpt-5')) return 'GPT-5';
    if (model.includes('sonnet-4-5')) return 'Sonnet 4.5';
    if (model.includes('sonnet')) return 'Sonnet';
    return model;
}

function updateToolUsageCard(sessionData, graphData) {
    const toolUsageListEl = document.getElementById('toolUsageList');
    if (!toolUsageListEl) return;

    const edges = graphData?.edges || [];
    const nodes = sessionData?.graph?.nodes || {};

    // Count edge types (relationships between nodes)
    const toolCounts = {
        agent_as_tool: edges.filter(e => e.type === 'agent_as_tool').length,
        handoff: edges.filter(e => e.type === 'handoff').length,
        ask_master: edges.filter(e => e.type === 'ask_master').length,
        function_tool: 0,
        non_function_tool: 0
    };

    // Count tool_timeline events (internal node tool calls)
    for (const node of Object.values(nodes)) {
        const timeline = node.tool_timeline || [];
        // Filter out "result from" entries
        const filteredTimeline = timeline.filter(event => {
            const toolName = (event.tool || '').toLowerCase();
            return !toolName.includes('result from') && !toolName.includes('← result from');
        });
        for (const event of filteredTimeline) {
            const eventType = event.type || '';
            if (eventType === 'function_tool') {
                toolCounts.function_tool++;
            } else if (eventType === 'non_function_tool') {
                toolCounts.non_function_tool++;
            }
        }
    }

    const total = Object.values(toolCounts).reduce((a, b) => a + b, 0);

    if (total === 0) {
        toolUsageListEl.innerHTML = '<p class="text-muted">No tool usage data</p>';
        return;
    }

    const toolConfig = [
        { key: 'agent_as_tool', label: 'Agent Delegation', color: '#f59e0b', gradient: 'linear-gradient(90deg, #f59e0b, #fbbf24)' },
        { key: 'handoff', label: 'Handoff', color: '#10b981', gradient: 'linear-gradient(90deg, #10b981, #34d399)' },
        { key: 'ask_master', label: 'Ask Master', color: '#ef4444', gradient: 'linear-gradient(90deg, #ef4444, #f87171)' },
        { key: 'function_tool', label: 'Function Calls', color: '#3b82f6', gradient: 'linear-gradient(90deg, #3b82f6, #60a5fa)' },
        { key: 'non_function_tool', label: 'Non-Function Tools', color: '#8b5cf6', gradient: 'linear-gradient(90deg, #8b5cf6, #a78bfa)' }
    ];

    toolUsageListEl.innerHTML = toolConfig.map(tool => {
        const count = toolCounts[tool.key];
        const percentage = total > 0 ? Math.round((count / total) * 100) : 0;

        return `
            <div class="tool-usage-item">
                <div class="tool-usage-header">
                    <div class="tool-usage-label">
                        <span class="tool-usage-dot" style="background-color: ${tool.color};"></span>
                        ${tool.label}
                    </div>
                    <div class="tool-usage-count">
                        <span class="tool-usage-number">${count}</span>
                        <span class="tool-usage-percentage">${percentage}%</span>
                    </div>
                </div>
                <div class="tool-usage-bar">
                    <div class="tool-usage-bar-fill" style="width: ${percentage}%; background: ${tool.gradient};"></div>
                </div>
            </div>
        `;
    }).join('');
}

function updateRecentEventsCard(sessionData) {
    const recentEventsListEl = document.getElementById('recentEventsList');
    if (!recentEventsListEl) return;

    // Get recent events from master log (includes ALL events: orchestration + function calls)
    const masterLog = sessionData.logs?.master_log || [];
    const recentEvents = masterLog.slice(-5).reverse(); // Last 5 events

    if (recentEvents.length === 0) {
        recentEventsListEl.innerHTML = '<p class="text-muted">No events yet</p>';
        return;
    }

    // Icon mapping for different tool types
    const getToolIcon = (toolName) => {
        const tool = (toolName || '').toLowerCase();
        if (tool.includes('route_to_') || tool.includes('ask_claude') || tool.includes('ask_gpt')) return 'users';
        if (tool.includes('handoff')) return 'git-branch';
        if (tool.includes('ask_master')) return 'help-circle';
        return 'zap';
    };

    // Color mapping
    const getToolColor = (toolName) => {
        const tool = (toolName || '').toLowerCase();
        if (tool.includes('route_to_') || tool.includes('ask_claude') || tool.includes('ask_gpt')) return '#f59e0b';
        if (tool.includes('handoff')) return '#10b981';
        if (tool.includes('ask_master')) return '#ef4444';
        return '#3b82f6';
    };

    recentEventsListEl.innerHTML = '';
    recentEvents.forEach((event, index) => {
        const item = document.createElement('div');
        item.className = 'event-item';

        const icon = getToolIcon(event.tool);
        const color = getToolColor(event.tool);
        const relativeTime = formatRelativeTime(event.timestamp);

        item.innerHTML = `
            <div class="event-item-header">
                <i data-lucide="${icon}" class="event-icon" style="color: ${color};"></i>
                <span class="event-tool">${event.tool || 'Unknown Tool'}</span>
            </div>
            <div class="event-item-meta">
                <span class="event-task-id">${event.task_id}</span>
                <span class="event-separator">•</span>
                <span class="event-time">${relativeTime}</span>
            </div>
        `;

        // Add click handler to show event details in sidebar
        item.addEventListener('click', () => {
            // Remove active class from all items
            recentEventsListEl.querySelectorAll('.event-item').forEach(el => el.classList.remove('active'));
            // Add active class to clicked item
            item.classList.add('active');
            // Show event details in sidebar
            showToolDetailsInSidebar(event, 'detailSidebar');
        });

        recentEventsListEl.appendChild(item);
    });

    lucide.createIcons();
}
