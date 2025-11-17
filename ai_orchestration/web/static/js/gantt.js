/*
 * Gantt Chart Component
 */

class GanttViz {
    constructor(container) {
        this.container = container;
        this.gantt = null;
        this.graphData = null;
    }

    render(graphData) {
        if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
            this.showEmptyState();
            return;
        }

        this.graphData = graphData;
        this.clearEmptyState();

        // Convert nodes to Frappe Gantt format
        const tasks = this.prepareTasksForGantt(graphData.nodes);

        if (tasks.length === 0) {
            this.showEmptyState();
            return;
        }

        // Create or update Gantt chart
        try {
            this.gantt = new Gantt(this.container, tasks, {
                view_mode: 'Hour',
                date_format: 'YYYY-MM-DD HH:mm:ss',
                bar_height: 32,
                bar_corner_radius: 8,
                arrow_curve: 5,
                padding: 18,
                view_modes: ['Quarter Day', 'Half Day', 'Day', 'Week', 'Month'],
                on_click: (task) => {
                    this.handleTaskClick(task);
                },
                on_date_change: (task, start, end) => {
                    return false; // Prevent editing
                },
                on_progress_change: (task, progress) => {
                    return false; // Prevent editing
                },
                on_view_change: (mode) => {
                    console.log('View mode changed to:', mode);
                }
            });

            // Force hide all popups after Gantt is created
            setTimeout(() => {
                const popups = this.container.querySelectorAll('.popup-wrapper, .details-container');
                popups.forEach(popup => {
                    popup.style.display = 'none';
                    popup.style.visibility = 'hidden';
                    popup.style.opacity = '0';
                });
            }, 100);
        } catch (error) {
            console.error('Error creating Gantt chart:', error);
            this.showError(error.message);
        }
    }

    prepareTasksForGantt(nodes) {
        const tasks = [];

        nodes.forEach(node => {
            // Skip nodes without created_at
            if (!node.created_at) return;

            const createdAt = new Date(node.created_at);
            let completedAt = node.completed_at ? new Date(node.completed_at) : new Date();

            // Ensure completed_at is after created_at
            if (completedAt <= createdAt) {
                completedAt = new Date(createdAt.getTime() + 1000); // Add 1 second minimum
            }

            // Calculate progress based on status
            let progress = 0;
            if (node.status === 'completed') {
                progress = 100;
            } else if (node.status === 'running') {
                progress = 50;
            } else if (node.status === 'waiting') {
                progress = 25;
            }

            // Create a shorter, cleaner name for display
            const agentName = node.agent_id ? node.agent_id.replace('agent_', '').substring(0, 15) : 'Unknown';
            const displayName = `${node.id.substring(0, 8)} • ${agentName}`;

            const task = {
                id: node.id,
                name: displayName,
                start: createdAt.toISOString().split('.')[0],
                end: completedAt.toISOString().split('.')[0],
                progress: progress,
                dependencies: node.parent_id || '',
                custom_class: `status-${node.status}`,
                // Store original node data
                _nodeData: node
            };

            tasks.push(task);
        });

        return tasks;
    }

    createPopupHTML(task) {
        const node = task._nodeData;
        const duration = new Date(task.end) - new Date(task.start);
        const durationStr = formatDuration(duration);

        return `
            <div class="details-container">
                <div class="title">${task.id}</div>
                <div class="subtitle">${node.agent_id}</div>
                <p>Status: <strong>${node.status}</strong></p>
                <p>Duration: <strong>${durationStr}</strong></p>
                <p>Tool Calls: <strong>${node.tool_calls || 0}</strong></p>
                <p><em>Click to view full details</em></p>
            </div>
        `;
    }

    handleTaskClick(task) {
        const node = task._nodeData;

        // Load full task details in the Gantt sidebar (same as graph view)
        if (window.loadTaskDetails) {
            window.loadTaskDetails(node.task_id, AppState.currentSessionId);
        }

        console.log('Gantt task clicked:', task.id, node);
    }

    changeViewMode(mode) {
        if (this.gantt) {
            this.gantt.change_view_mode(mode);
        }
    }

    showEmptyState() {
        this.container.innerHTML = `
            <div class="view-empty-state">
                <i data-lucide="bar-chart-3"></i>
                <p>Select a session to view the Gantt chart</p>
            </div>
        `;
        lucide.createIcons();
    }

    clearEmptyState() {
        const emptyState = this.container.querySelector('.view-empty-state');
        if (emptyState) {
            this.container.innerHTML = '';
        }
    }

    showError(message) {
        this.container.innerHTML = `
            <div class="view-empty-state">
                <i data-lucide="alert-circle"></i>
                <p>Error loading Gantt chart: ${message}</p>
            </div>
        `;
        lucide.createIcons();
    }

    destroy() {
        if (this.gantt) {
            this.container.innerHTML = '';
            this.gantt = null;
        }
    }
}

// Initialize global Gantt instance
window.GanttViz = null;

// Helper function to initialize Gantt on page load
function initializeGantt() {
    const ganttContainer = document.getElementById('ganttChart');
    if (ganttContainer) {
        window.GanttViz = new GanttViz(ganttContainer);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeGantt);
} else {
    initializeGantt();
}
