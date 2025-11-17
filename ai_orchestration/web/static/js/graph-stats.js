/*
 * Graph Statistics Modal
 */

function openGraphStatsModal() {
    const modal = document.getElementById('graphStatsModal');
    if (!modal) return;

    modal.classList.remove('hidden');
    renderGraphStats();
}

function closeGraphStatsModal() {
    const modal = document.getElementById('graphStatsModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

function renderGraphStats() {
    const content = document.getElementById('graphStatsContent');
    if (!content || !window.currentGraphData) {
        content.innerHTML = '<p class="text-muted">No graph data available</p>';
        return;
    }

    const stats = calculateGraphStats(window.currentGraphData);

    content.innerHTML = `
        <div class="stats-grid">
            <!-- Overview Section -->
            <div class="stats-section stats-section-full">
                <h3 class="stats-section-title">
                    <i data-lucide="grid"></i>
                    Overview
                </h3>
                <div class="stats-cards">
                    <div class="stat-card">
                        <div class="stat-icon" style="background-color: rgba(37, 99, 235, 0.1);">
                            <i data-lucide="layers" style="color: #2563eb;"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value">${stats.totalNodes}</div>
                            <div class="stat-label">Total Tasks</div>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon" style="background-color: rgba(16, 185, 129, 0.1);">
                            <i data-lucide="git-branch" style="color: #10b981;"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value">${stats.totalEdges}</div>
                            <div class="stat-label">Total Connections</div>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon" style="background-color: rgba(245, 158, 11, 0.1);">
                            <i data-lucide="users" style="color: #f59e0b;"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value">${stats.totalAgents}</div>
                            <div class="stat-label">Active Agents</div>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon" style="background-color: rgba(168, 85, 247, 0.1);">
                            <i data-lucide="trending-up" style="color: #a855f7;"></i>
                        </div>
                        <div class="stat-content">
                            <div class="stat-value">${stats.maxDepth}</div>
                            <div class="stat-label">Max Chain Depth</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Task Status Breakdown -->
            <div class="stats-section">
                <h3 class="stats-section-title">
                    <i data-lucide="activity"></i>
                    Task Status Distribution
                </h3>
                <div class="stats-list">
                    <div class="stats-list-item">
                        <div class="stats-item-header">
                            <span class="badge badge-success">Completed</span>
                            <span class="stats-item-value">${stats.statusBreakdown.completed} (${stats.statusPercentages.completed}%)</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${stats.statusPercentages.completed}%; background: linear-gradient(90deg, #10b981, #34d399);"></div>
                        </div>
                    </div>
                    <div class="stats-list-item">
                        <div class="stats-item-header">
                            <span class="badge badge-info">Running</span>
                            <span class="stats-item-value">${stats.statusBreakdown.running} (${stats.statusPercentages.running}%)</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${stats.statusPercentages.running}%; background: linear-gradient(90deg, #3b82f6, #60a5fa);"></div>
                        </div>
                    </div>
                    <div class="stats-list-item">
                        <div class="stats-item-header">
                            <span class="badge badge-warning">Waiting</span>
                            <span class="stats-item-value">${stats.statusBreakdown.waiting} (${stats.statusPercentages.waiting}%)</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${stats.statusPercentages.waiting}%; background: linear-gradient(90deg, #f59e0b, #fbbf24);"></div>
                        </div>
                    </div>
                    <div class="stats-list-item">
                        <div class="stats-item-header">
                            <span class="badge badge-gray">Ready</span>
                            <span class="stats-item-value">${stats.statusBreakdown.ready} (${stats.statusPercentages.ready}%)</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${stats.statusPercentages.ready}%; background: linear-gradient(90deg, #6b7280, #9ca3af);"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Edge Type Breakdown -->
            <div class="stats-section">
                <h3 class="stats-section-title">
                    <i data-lucide="share-2"></i>
                    Connection Types
                </h3>
                <div class="stats-list">
                    <div class="stats-list-item">
                        <div class="stats-item-header">
                            <span class="edge-type-label">
                                <span class="edge-color" style="background-color: #f59e0b;"></span>
                                Agent Delegation
                            </span>
                            <span class="stats-item-value">${stats.edgeBreakdown.agent_as_tool}</span>
                        </div>
                    </div>
                    <div class="stats-list-item">
                        <div class="stats-item-header">
                            <span class="edge-type-label">
                                <span class="edge-color" style="background-color: #10b981;"></span>
                                Handoff
                            </span>
                            <span class="stats-item-value">${stats.edgeBreakdown.handoff}</span>
                        </div>
                    </div>
                    <div class="stats-list-item">
                        <div class="stats-item-header">
                            <span class="edge-type-label">
                                <span class="edge-color" style="background-color: #ef4444;"></span>
                                Ask Master
                            </span>
                            <span class="stats-item-value">${stats.edgeBreakdown.ask_master}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Performance Metrics -->
            <div class="stats-section stats-section-full">
                <h3 class="stats-section-title">
                    <i data-lucide="zap"></i>
                    Performance Metrics
                </h3>
                <div class="stats-metrics">
                    <div class="metric-box">
                        <div class="metric-icon">
                            <i data-lucide="timer"></i>
                        </div>
                        <div class="metric-info">
                            <div class="metric-label">Average Task Duration</div>
                            <div class="metric-value">${stats.avgTaskDuration}</div>
                        </div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-icon">
                            <i data-lucide="user"></i>
                        </div>
                        <div class="metric-info">
                            <div class="metric-label">Most Active Agent</div>
                            <div class="metric-value">${stats.mostActiveAgent}</div>
                        </div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-icon">
                            <i data-lucide="percent"></i>
                        </div>
                        <div class="metric-info">
                            <div class="metric-label">Completion Rate</div>
                            <div class="metric-value">${stats.completionRate}%</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    lucide.createIcons();
}

function calculateGraphStats(graphData) {
    const nodes = graphData.nodes || {};
    const edges = graphData.edges || [];
    const nodeArray = Object.values(nodes);

    // Total counts
    const totalNodes = nodeArray.length;
    const totalEdges = edges.length;

    // Status breakdown
    const statusBreakdown = {
        completed: nodeArray.filter(n => n.status === 'completed').length,
        running: nodeArray.filter(n => n.status === 'running').length,
        waiting: nodeArray.filter(n => n.status === 'waiting').length,
        ready: nodeArray.filter(n => n.status === 'ready').length
    };

    // Status percentages
    const statusPercentages = {
        completed: totalNodes > 0 ? Math.round((statusBreakdown.completed / totalNodes) * 100) : 0,
        running: totalNodes > 0 ? Math.round((statusBreakdown.running / totalNodes) * 100) : 0,
        waiting: totalNodes > 0 ? Math.round((statusBreakdown.waiting / totalNodes) * 100) : 0,
        ready: totalNodes > 0 ? Math.round((statusBreakdown.ready / totalNodes) * 100) : 0
    };

    // Edge type breakdown
    const edgeBreakdown = {
        agent_as_tool: edges.filter(e => e.type === 'agent_as_tool').length,
        handoff: edges.filter(e => e.type === 'handoff').length,
        ask_master: edges.filter(e => e.type === 'ask_master').length
    };

    // Agent count
    const agents = new Set(nodeArray.map(n => n.agent_id));
    const totalAgents = agents.size;

    // Most active agent
    const agentCounts = {};
    nodeArray.forEach(n => {
        agentCounts[n.agent_id] = (agentCounts[n.agent_id] || 0) + 1;
    });
    const mostActiveAgent = Object.keys(agentCounts).reduce((a, b) =>
        agentCounts[a] > agentCounts[b] ? a : b, 'N/A');

    // Max chain depth (calculate from edges)
    const maxDepth = calculateMaxDepth(nodes, edges);

    // Average task duration
    const completedNodes = nodeArray.filter(n => n.status === 'completed' && n.created_at && n.completed_at);
    let avgTaskDuration = '-';
    if (completedNodes.length > 0) {
        const totalDuration = completedNodes.reduce((sum, node) => {
            const start = new Date(node.created_at);
            const end = new Date(node.completed_at);
            return sum + (end - start);
        }, 0);
        avgTaskDuration = formatDuration(totalDuration / completedNodes.length);
    }

    // Completion rate
    const completionRate = totalNodes > 0 ? Math.round((statusBreakdown.completed / totalNodes) * 100) : 0;

    return {
        totalNodes,
        totalEdges,
        totalAgents,
        statusBreakdown,
        statusPercentages,
        edgeBreakdown,
        mostActiveAgent,
        maxDepth,
        avgTaskDuration,
        completionRate
    };
}

function calculateMaxDepth(nodes, edges) {
    // Find root nodes (nodes with no incoming edges)
    const nodeIds = Object.keys(nodes);
    const hasIncoming = new Set(edges.map(e => e.target));
    const roots = nodeIds.filter(id => !hasIncoming.has(id));

    if (roots.length === 0) return 0;

    // Build adjacency list
    const adjacency = {};
    edges.forEach(e => {
        if (!adjacency[e.source]) adjacency[e.source] = [];
        adjacency[e.source].push(e.target);
    });

    // BFS to find max depth
    let maxDepth = 0;
    roots.forEach(root => {
        const queue = [{node: root, depth: 1}];
        const visited = new Set();

        while (queue.length > 0) {
            const {node, depth} = queue.shift();
            if (visited.has(node)) continue;
            visited.add(node);

            maxDepth = Math.max(maxDepth, depth);

            const children = adjacency[node] || [];
            children.forEach(child => {
                queue.push({node: child, depth: depth + 1});
            });
        }
    });

    return maxDepth;
}

// Initialize event listeners
document.addEventListener('DOMContentLoaded', () => {
    const graphStatsBtn = document.getElementById('graphStatsBtn');
    if (graphStatsBtn) {
        graphStatsBtn.addEventListener('click', openGraphStatsModal);
    }
});
