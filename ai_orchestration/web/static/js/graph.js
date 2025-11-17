/*
 * Graph Visualization with D3.js
 */

class GraphVisualization {
    constructor(containerId) {
        this.container = d3.select(`#${containerId}`);
        this.svg = null;
        this.g = null;
        this.simulation = null;
        this.nodes = [];
        this.edges = [];
        this.width = 0;
        this.height = 0;
        this.zoom = null;
        this.selectedNode = null;

        this.initialize();
    }

    initialize() {
        console.log('initialize() called');

        // Clear existing content
        this.container.selectAll('*').remove();

        // Get dimensions
        const bounds = this.container.node().getBoundingClientRect();
        this.width = bounds.width;
        this.height = bounds.height;

        console.log('  Container dimensions:', this.width, 'x', this.height);

        // Create SVG
        this.svg = this.container.append('svg')
            .attr('width', this.width)
            .attr('height', this.height);

        console.log('  SVG created:', this.svg.node());

        // Create zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });

        this.svg.call(this.zoom);

        // Add canvas click to unhighlight (click on empty space)
        this.svg.on('click', (event) => {
            // Only unhighlight if clicking on SVG directly (not on nodes/edges)
            if (event.target.tagName === 'svg' || event.target.classList.contains('edges') || event.target.classList.contains('nodes')) {
                if (this.unhighlight) this.unhighlight();
            }
        });

        // Create main group for graph elements
        this.g = this.svg.append('g');
        console.log('  Main group created:', this.g.node());

        // Create groups for edges and nodes (edges first so they're behind)
        const edgesGroup = this.g.append('g').attr('class', 'edges');
        const nodesGroup = this.g.append('g').attr('class', 'nodes');

        console.log('  Edges group created:', edgesGroup.node());
        console.log('  Nodes group created:', nodesGroup.node());
        console.log('  Can select .nodes?', this.g.select('.nodes').node());

        // Add tooltip
        this.tooltip = this.container.append('div')
            .attr('class', 'graph-tooltip');

        console.log('initialize() complete');
    }

    render(graphData) {
        if (!graphData || !graphData.nodes || !graphData.edges) {
            console.error('Invalid graph data');
            return;
        }

        // Store graph data globally for stats modal
        window.currentGraphData = graphData;

        // Reinitialize if SVG was destroyed or detached from DOM
        const svgNode = this.svg?.node();
        const isAttached = svgNode && document.body.contains(svgNode);

        if (!this.svg || !svgNode || !isAttached) {
            console.log('SVG not found or detached, reinitializing...');
            this.initialize();
        }

        // Keep ALL nodes for D3 simulation (including ask_master for edge connections)
        this.nodes = graphData.nodes.map(n => ({
            ...n,
            // Initialize x/y for ALL nodes to prevent NaN errors
            x: n.x || this.width / 2 + (Math.random() - 0.5) * 100,
            y: n.y || this.height / 2 + (Math.random() - 0.5) * 100
        }));
        this.edges = graphData.edges.map(e => ({ ...e }));

        // Filter out ask_master nodes from display (backend creates edges between parent and grandparent)
        this.visibleNodes = this.nodes.filter(n => n.tool_type !== 'ask_master');

        // Clear empty state
        this.container.select('.graph-empty-state').remove();

        // Create force simulation - ONLY use visible nodes (exclude ask_master)
        this.simulation = d3.forceSimulation(this.visibleNodes)
            .force('link', d3.forceLink(this.edges)
                .id(d => d.id)
                .distance(150))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(50))
            .on('tick', () => this.ticked());

        // Render edges
        this.renderEdges();

        // Render nodes
        this.renderNodes();

        // Render legend
        this.renderLegend();

        // Debug: Log rendering completion
        console.log('='.repeat(80));
        console.log('RENDER COMPLETE');
        console.log('SVG exists:', !!this.svg?.node());
        console.log('Nodes created:', d3.select('#graphCanvas svg g.nodes').selectAll('circle').size());
        console.log('Edges created:', d3.select('#graphCanvas svg g.edges').selectAll('path').size());
        console.log('Simulation running:', this.simulation ? 'YES' : 'NO');
        console.log('='.repeat(80));

        // Fit to screen after layout stabilizes
        setTimeout(() => this.fitToScreen(), 1000);
    }

    renderEdges() {
        const edgesGroup = this.g.select('.edges');

        // NO SVG ARROWS - keep it simple!

        // Group edges by source-target pair to handle multiple edges
        const edgeGroups = {};
        this.edges.forEach(edge => {
            const sourceId = edge.source.id || edge.source;
            const targetId = edge.target.id || edge.target;
            // Create directional key to keep A->B separate from B->A
            // This allows showing delegate (A->B), handoff (B->A), and ask_master (B->A) as separate edges
            const key = `${sourceId}->${targetId}`;
            if (!edgeGroups[key]) edgeGroups[key] = [];
            edgeGroups[key].push(edge);
        });

        console.log('DEBUG: Edge grouping:', {
            totalEdges: this.edges.length,
            numGroups: Object.keys(edgeGroups).length,
            groupSizes: Object.entries(edgeGroups).map(([k, v]) => `${k}:${v.length}`),
            multiEdgeGroups: Object.entries(edgeGroups)
                .filter(([k, v]) => v.length > 1)
                .map(([k, v]) => ({
                    pair: k,
                    count: v.length,
                    types: v.map(e => e.type)
                })),
            sampleGroup: Object.values(edgeGroups)[0]?.map(e => ({
                type: e.type,
                source: e.source.id || e.source,
                target: e.target.id || e.target
            }))
        });

        // SOLUTION: Check if edges are between SAME 2 nodes or MULTIPLE nodes
        console.log('Total edges to process:', this.edges.length);

        // Get unique node pairs
        const nodePairs = new Set();
        this.edges.forEach(edge => {
            const sourceId = edge.source.id || edge.source;
            const targetId = edge.target.id || edge.target;
            const pair = [sourceId, targetId].sort().join('-');
            nodePairs.add(pair);
        });

        console.log(`Edges are between ${nodePairs.size} unique node pair(s)`);

        // If all edges are between the SAME 2 nodes, use special handling
        if (nodePairs.size === 1 && this.edges.length <= 10) {
            // All edges between same 2 nodes - hardcode offsets for 1-10 edges
            const offsets = {
                1: [0],
                2: [-30, 30],
                3: [-40, 0, 40],
                4: [-60, -20, 20, 60],
                5: [-80, -40, 0, 40, 80],
                6: [-100, -60, -20, 20, 60, 100],
                7: [-120, -80, -40, 0, 40, 80, 120],
                8: [-140, -100, -60, -20, 20, 60, 100, 140],
                9: [-160, -120, -80, -40, 0, 40, 80, 120, 160],
                10: [-180, -140, -100, -60, -20, 20, 60, 100, 140, 180]
            };

            const edgeOffsets = offsets[this.edges.length] || [];

            // Group by direction for proper offset assignment
            const forward = [];
            const reverse = [];

            this.edges.forEach(edge => {
                const sourceId = edge.source.id || edge.source;
                const targetId = edge.target.id || edge.target;
                // Use consistent direction detection
                if (sourceId < targetId) {
                    forward.push(edge);
                } else {
                    reverse.push(edge);
                }
            });

            // Assign offsets alternating between directions
            let offsetIndex = 0;
            forward.forEach(edge => {
                edge.offset = edgeOffsets[offsetIndex++] || 0;
                console.log(`Forward edge: ${edge.type}, offset=${edge.offset}px`);
            });
            reverse.forEach(edge => {
                edge.offset = edgeOffsets[offsetIndex++] || 0;
                console.log(`Reverse edge: ${edge.type}, offset=${edge.offset}px`);
            });

        } else {
            // GRAPH with multiple node connections
            console.log("Processing graph with multiple node pairs");
            const processedKeys = new Set();

            Object.entries(edgeGroups).forEach(([key, group]) => {
                if (processedKeys.has(key)) return;

                const [sourceId, targetId] = key.split('->');
                const reverseKey = `${targetId}->${sourceId}`;
                const reverseGroup = edgeGroups[reverseKey] || [];

                // Mark both directions as processed
                processedKeys.add(key);
                if (reverseKey) processedKeys.add(reverseKey);

                // ALWAYS separate bidirectional edges, even if just 1 each way!
                if (reverseGroup.length > 0) {
                    // Bidirectional - ALWAYS separate them
                    console.log(`Bidirectional: ${key} (${group.length}) <-> ${reverseKey} (${reverseGroup.length})`);

                    // Forward edges get negative offsets
                    const spacing = 40;
                    group.forEach((edge, i) => {
                        edge.offset = -30 - (i * spacing);
                        console.log(`  ${key}: offset=${edge.offset}px`);
                    });

                    // Reverse edges get positive offsets
                    reverseGroup.forEach((edge, i) => {
                        edge.offset = 30 + (i * spacing);
                        console.log(`  ${reverseKey}: offset=${edge.offset}px`);
                    });
                } else if (group.length === 1) {
                    // Single edge with no reverse
                    group[0].offset = 0;
                    console.log(`Single edge ${key}: offset=0`);
                } else {
                    // Multiple edges same direction
                    const spacing = 40;
                    const startOffset = -((group.length - 1) * spacing) / 2;
                    group.forEach((edge, i) => {
                        edge.offset = startOffset + i * spacing;
                        console.log(`Multi-edge ${key} [${i}]: offset=${edge.offset}px`);
                    });
                }
            });
        }

        console.log('Rendering edges:', this.edges.map(e => ({
            id: e.id,
            source: e.source.id || e.source,
            target: e.target.id || e.target,
            type: e.type
        })));

        const edgeSelection = edgesGroup.selectAll('.edge')
            .data(this.edges, d => d.id || `${d.source.id || d.source}-${d.target.id || d.target}-${d.type}`);

        console.log('Edge data binding:', {
            existingEdges: edgeSelection.size(),
            newEdges: edgeSelection.enter().size(),
            removedEdges: edgeSelection.exit().size()
        });

        // Remove old edges
        edgeSelection.exit().remove();

        // Add new edges
        const edgeEnter = edgeSelection.enter()
            .append('g')
            .attr('class', 'edge');

        console.log('Edge groups created:', edgeEnter.size());

        // Add single visible interactive path
        edgeEnter.append('path')
            .attr('class', d => `graph-edge edge-${d.type} ${d.status === 'running' ? 'active' : ''}`)
            .each(function(d) {
                console.log(`Path created for edge ${d.id}: ${d.source.id || d.source} -> ${d.target.id || d.target} (${d.type})`);
            });

        // Merge enter + update selections and apply all attributes
        const edgeMerge = edgeEnter.merge(edgeSelection);

        console.log('Total edge paths after merge:', edgeMerge.size());

        edgeMerge.select('path')
            .style('stroke', d => this.getEdgeColorByType(d.type))
            .style('stroke-width', '2px')  // Thinner edges
            .style('stroke-dasharray', d => d.type === 'ask_master' ? '5,3' : 'none')
            .style('fill', 'none')
            .style('cursor', 'pointer')
            .style('pointer-events', 'stroke')
            .attr('opacity', 0.8)
            .on('mouseover', (event, d) => this.showEdgeTooltip(event, d))
            .on('mouseout', () => this.hideTooltip())
            .on('click', (event, d) => this.edgeClicked(event, d));

        this.edgeElements = edgesGroup.selectAll('.edge');
    }

    renderNodes() {
        console.log('renderNodes called');
        console.log('  this.g:', this.g);
        console.log('  Total nodes:', this.nodes.length);
        console.log('  Visible nodes:', this.visibleNodes.length);

        const nodesGroup = this.g.select('.nodes');
        console.log('  nodesGroup:', nodesGroup.node());

        // Only render visible nodes (not ask_master)
        const nodeSelection = nodesGroup.selectAll('.node')
            .data(this.visibleNodes, d => d.id);

        console.log('  nodeSelection size:', nodeSelection.size());
        console.log('  nodeSelection.enter() size:', nodeSelection.enter().size());

        // Remove old nodes
        nodeSelection.exit().remove();

        // Add new nodes
        const nodeEnter = nodeSelection.enter()
            .append('g')
            .attr('class', d => `node graph-node ${d.status}`)
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)))
            .on('click', (event, d) => this.nodeClicked(event, d))
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip());

        // Add circle (colored by node type: ROOT vs DELEGATE)
        nodeEnter.append('circle')
            .attr('r', d => this.getNodeRadius(d))
            .attr('class', d => `node-${d.status}`)
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', '#fff')
            .attr('stroke-width', 3)
            .attr('opacity', d => d.status === 'completed' ? 0.8 : 1);

        // Add task number in center
        nodeEnter.append('text')
            .attr('class', 'node-number')
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .attr('font-size', d => {
                if (d.tool_type === 'initiator') return '16px';  // Initiator: medium
                return !d.parent_id ? '28px' : '20px';  // ROOT: large, DELEGATE: normal
            })
            .attr('font-weight', 'bold')
            .attr('fill', '#fff')
            .style('pointer-events', 'none')
            .text(d => {
                if (d.tool_type === 'initiator') return 'U';  // U for User/Initiator
                return d.id.replace('task_', '');  // Task number for others
            });

        // Add payload label (first line)
        nodeEnter.append('text')
            .attr('class', 'node-label-payload')
            .attr('text-anchor', 'middle')
            .attr('dy', '2.8em')
            .attr('font-size', '10px')
            .attr('font-weight', '500')
            .attr('fill', '#1f2937')
            .text(d => truncate(d.label || d.id, 30));

        // Add agent + node type label (second line)
        nodeEnter.append('text')
            .attr('class', 'node-label-info')
            .attr('text-anchor', 'middle')
            .attr('dy', '4em')
            .attr('font-size', '9px')
            .attr('fill', '#6b7280')
            .text(d => `${d.agent_id || 'unknown'} • ${!d.parent_id ? 'ROOT' : 'DELEGATE'}`);

        this.nodeElements = nodesGroup.selectAll('.node');
    }

    ticked() {
        // CRITICAL: Skip rendering if any visible nodes have NaN coordinates
        // This prevents console spam of 260+ NaN errors during simulation initialization
        const hasInvalidCoords = this.visibleNodes.some(n =>
            isNaN(n.x) || isNaN(n.y) || n.x === undefined || n.y === undefined
        );

        if (hasInvalidCoords) {
            // Silently skip this tick - simulation will call us again once coords are ready
            return;
        }

        // Update edge positions
        if (this.edgeElements) {
            this.edgeElements.select('.graph-edge')
                .attr('d', d => {
                    const source = d.source;
                    const target = d.target;
                    const offset = d.offset || 0;

                    // Check if source/target have valid positions
                    if (!source || !target ||
                        isNaN(source.x) || isNaN(source.y) ||
                        isNaN(target.x) || isNaN(target.y)) {
                        return null;  // Skip this edge
                    }

                    return this.createCurvedPath(source, target, offset);
                });
        }

        // Update node positions
        if (this.nodeElements) {
            this.nodeElements.attr('transform', d => {
                // Guard against NaN - use 0 as fallback
                const x = isNaN(d.x) ? 0 : d.x;
                const y = isNaN(d.y) ? 0 : d.y;
                return `translate(${x},${y})`;
            });
        }
    }

    createCurvedPath(source, target, offset = 0) {
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const dr = Math.sqrt(dx * dx + dy * dy);

        if (offset === 0) {
            // Straight line for single edge
            return `M${source.x},${source.y} L${target.x},${target.y}`;
        } else {
            // FIX: Always calculate perpendicular from a CONSISTENT direction
            // Use the node with smaller ID as the reference to ensure consistency
            const sourceId = source.id || 'task_0';
            const targetId = target.id || 'human';

            // Always calculate angle from "smaller" to "larger" ID
            let consistentDx, consistentDy;
            if (sourceId < targetId) {
                consistentDx = dx;
                consistentDy = dy;
            } else {
                consistentDx = -dx;
                consistentDy = -dy;
            }

            const angle = Math.atan2(consistentDy, consistentDx);
            const perpAngle = angle + Math.PI / 2;
            const offsetX = Math.cos(perpAngle) * offset;
            const offsetY = Math.sin(perpAngle) * offset;

            // Control point for quadratic curve
            const midX = (source.x + target.x) / 2 + offsetX;
            const midY = (source.y + target.y) / 2 + offsetY;

            // Use Q for quadratic bezier curve - this creates an actual curve!
            return `M${source.x},${source.y} Q${midX},${midY} ${target.x},${target.y}`;
        }
    }

    getNodeRadius(node) {
        // Initiator nodes are small
        if (node.tool_type === 'initiator') {
            return 35;
        }

        // Root tasks are bigger
        if (!node.parent_id) {
            return 50;
        }

        const baseRadius = 30;
        const toolCallBonus = Math.min(node.tool_calls * 2, 20);
        return baseRadius + toolCallBonus;
    }

    getNodeColor(node) {
        // INITIATOR: Gray/silver - session master
        if (node.tool_type === 'initiator') {
            return '#6b7280';  // Gray - INITIATOR
        }

        // ROOT: Purple
        if (!node.parent_id) {
            return '#9333ea';  // Purple - ROOT task
        }

        // DELEGATE: Blue
        return '#3b82f6';  // Blue - DELEGATE task
    }

    getEdgeColorByType(edgeType) {
        const colors = {
            'agent_as_tool': '#f59e0b',  // Orange - delegation
            'ask_master': '#ef4444',     // Red - questions
            'handoff': '#10b981'         // Green - completion
        };
        return colors[edgeType?.toLowerCase()] || '#6b7280';
    }

    getEdgeColor(edge) {
        if (edge.status === 'running') return '#3b82f6';
        if (edge.status === 'completed') return '#10b981';
        return '#d1d5db';
    }

    // Drag handlers
    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    // Click handler
    nodeClicked(event, d) {
        event.stopPropagation();

        // Unhighlight everything first
        this.unhighlight();

        this.selectNode(d.id);

        // Load task details
        if (AppState.currentSessionId) {
            loadTaskDetails(d.id, AppState.currentSessionId);
        }
    }

    selectNode(nodeId) {
        // Remove previous selection
        this.nodeElements.classed('selected', false);

        // Add selection to clicked node
        this.nodeElements.filter(d => d.id === nodeId)
            .classed('selected', true);

        this.selectedNode = nodeId;
    }

    // Unhighlight all nodes and edges - reset to normal view
    unhighlight() {
        // Reset all node opacity
        this.nodeElements.attr('opacity', 1);

        // Reset all edge opacity and stroke-width
        this.edgeElements.select('.graph-edge')
            .attr('opacity', 0.8)
            .style('stroke-width', '2px');
    }

    // Tooltip
    showTooltip(event, d) {
        const nodeType = !d.parent_id ? 'ROOT' : 'DELEGATE';
        const nodeTypeColor = this.getNodeColor(d);
        const payload = d.label ? truncate(d.label, 100) : 'N/A';

        // Count children
        const childCount = this.nodes.filter(n => n.parent_id === d.id).length;

        // Format timestamps
        const createdAt = d.created_at ? formatAbsoluteTime(d.created_at) : 'N/A';
        const completedAt = d.completed_at ? formatAbsoluteTime(d.completed_at) : '-';

        // Duration
        let duration = '-';
        if (d.created_at && d.completed_at) {
            const start = new Date(d.created_at);
            const end = new Date(d.completed_at);
            const diffMs = end - start;
            duration = formatDuration(diffMs);
        }

        this.tooltip
            .style('opacity', 0.95)
            .style('left', `${event.pageX + 10}px`)
            .style('top', `${event.pageY - 10}px`)
            .html(`
                <div class="tooltip-header">
                    <strong>${d.id}</strong>
                    <span class="badge badge-${this.getStatusBadgeClass(d.status)}">${d.status}</span>
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Type:</span>
                    <span class="tooltip-value" style="color: ${nodeTypeColor}; font-weight: bold;">${nodeType}</span>
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Agent:</span>
                    <span class="tooltip-value">${d.agent_id || 'N/A'}</span>
                </div>
                ${d.parent_id ? `
                <div class="tooltip-row">
                    <span class="tooltip-label">Parent:</span>
                    <span class="tooltip-value">${d.parent_id}</span>
                </div>
                ` : '<div class="tooltip-row"><span class="tooltip-label">Parent:</span><span class="tooltip-value">Root Task</span></div>'}
                ${d.master_agent_id ? `
                <div class="tooltip-row">
                    <span class="tooltip-label">Master Agent:</span>
                    <span class="tooltip-value">${d.master_agent_id}</span>
                </div>
                ` : ''}
                <div class="tooltip-row">
                    <span class="tooltip-label">Children:</span>
                    <span class="tooltip-value">${childCount} task${childCount !== 1 ? 's' : ''}</span>
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Tool Calls:</span>
                    <span class="tooltip-value">${d.tool_calls}</span>
                </div>
                <div class="tooltip-divider"></div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Created:</span>
                    <span class="tooltip-value">${createdAt}</span>
                </div>
                ${d.completed_at ? `
                <div class="tooltip-row">
                    <span class="tooltip-label">Completed:</span>
                    <span class="tooltip-value">${completedAt}</span>
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Duration:</span>
                    <span class="tooltip-value">${duration}</span>
                </div>
                ` : ''}
                <div class="tooltip-divider"></div>
                <div class="tooltip-payload">
                    <div class="tooltip-label">Payload:</div>
                    <div class="tooltip-payload-text">${payload}</div>
                </div>
            `);
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

    hideTooltip() {
        this.tooltip.style('opacity', 0);
    }

    // Edge tooltip
    showEdgeTooltip(event, d) {
        const edgeType = (d.type || 'unknown').replace(/_/g, ' ').toUpperCase();
        const edgeColor = this.getEdgeColorByType(d.type);
        const timestamp = d.timestamp ? formatAbsoluteTime(d.timestamp) : 'N/A';

        const edgeDescriptions = {
            'agent_as_tool': 'Agent delegated task to another agent',
            'ask_master': 'Agent asked master for guidance',
            'handoff': 'Agent completed and handed off task'
        };

        const description = edgeDescriptions[d.type] || 'Interaction';

        this.tooltip
            .style('opacity', 0.95)
            .style('left', `${event.pageX + 10}px`)
            .style('top', `${event.pageY - 10}px`)
            .html(`
                <div class="tooltip-header">
                    <strong style="color: ${edgeColor}">${edgeType}</strong>
                    <span class="badge badge-${this.getStatusBadgeClass(d.status)}">${d.status}</span>
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">From:</span>
                    <span class="tooltip-value">${d.source.id || d.source}</span>
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">To:</span>
                    <span class="tooltip-value">${d.target.id || d.target}</span>
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">When:</span>
                    <span class="tooltip-value">${timestamp}</span>
                </div>
                <div class="tooltip-divider"></div>
                <div class="tooltip-description">
                    ${description}
                </div>
            `);
    }

    edgeClicked(event, d) {
        event.stopPropagation();
        console.log('='.repeat(80));
        console.log('EDGE CLICKED - FULL DATA:');
        console.log('  Edge object:', d);
        console.log('  Edge type:', d.type);
        console.log('  Tool:', d.tool);
        console.log('  Has input:', !!d.input);
        console.log('  Input type:', typeof d.input);
        console.log('  Input value:', d.input);
        console.log('  Has result:', !!d.result);
        console.log('  Result type:', typeof d.result);
        console.log('  Result value:', d.result);
        console.log('='.repeat(80));

        // Highlight the edge and connected nodes
        this.highlightEdge(d);

        // Show edge details in sidebar
        this.showEdgeDetailsInSidebar(d);
    }

    getToolTypeInfoForEdge(edge) {
        if (edge.type === 'agent_as_tool') {
            return {
                icon: 'users',
                badgeClass: 'agent-delegation',
                displayName: 'Agent Delegation',
                color: '#f59e0b'
            };
        } else if (edge.type === 'handoff') {
            return {
                icon: 'check-circle',
                badgeClass: 'handoff',
                displayName: 'Task Handoff',
                color: '#10b981'
            };
        } else if (edge.type === 'ask_master') {
            return {
                icon: 'help-circle',
                badgeClass: 'ask-master',
                displayName: 'Question to Master',
                color: '#ef4444'
            };
        } else {
            return {
                icon: 'zap',
                badgeClass: 'function-call',
                displayName: 'Function Call',
                color: '#3b82f6'
            };
        }
    }

    showEdgeDetailsInSidebar(edge) {
        const sidebar = document.getElementById('detailSidebar');
        if (!sidebar) return;

        const toolInfo = this.getToolTypeInfoForEdge(edge);
        const timestamp = edge.timestamp ? formatAbsoluteTime(edge.timestamp) : 'N/A';
        const sourceId = edge.source.id || edge.source;
        const targetId = edge.target.id || edge.target;
        const toolName = edge.tool || 'handoff';

        // Check if we have actual data
        const hasInput = edge.input && Object.keys(edge.input).length > 0;
        const hasResult = edge.result && edge.result !== '' && edge.result !== 'undefined';

        const inputJson = hasInput ? formatJSON(edge.input) : '{}';
        const resultJson = hasResult ? formatJSON(edge.result) : 'No result';
        const inputSize = hasInput ? JSON.stringify(edge.input).length : 0;
        const resultSize = hasResult ? JSON.stringify(edge.result).length : 0;

        sidebar.innerHTML = `
            <div class="sidebar-header">
                <div class="tool-type-header">
                    <div class="tool-icon" style="background-color: ${toolInfo.color}20; color: ${toolInfo.color};">
                        <i data-lucide="${toolInfo.icon}"></i>
                    </div>
                    <div class="tool-info">
                        <h3 class="sidebar-title">${toolName}</h3>
                        <span class="tool-badge ${toolInfo.badgeClass}">${toolInfo.displayName}</span>
                    </div>
                </div>
                <button class="sidebar-close" onclick="SidebarManager.close()">
                    <i data-lucide="x"></i>
                </button>
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
                                <span class="info-label">From</span>
                                <span class="info-value copyable" onclick="loadTaskDetails('${sourceId}', AppState.currentSessionId)" title="Click to view task">${sourceId}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">To</span>
                                <span class="info-value copyable" onclick="loadTaskDetails('${targetId}', AppState.currentSessionId)" title="Click to view task">${targetId}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Timestamp</span>
                                <span class="info-value">${timestamp}</span>
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

    highlightEdge(edge) {
        // Dim all edges
        this.edgeElements.select('.graph-edge')
            .attr('opacity', 0.15)
            .style('stroke-width', '2px');

        // Highlight selected edge
        this.edgeElements.filter(e => e.id === edge.id)
            .select('.graph-edge')
            .attr('opacity', 1)
            .style('stroke-width', '4px');

        // Dim unconnected nodes, highlight connected ones
        this.nodeElements
            .attr('opacity', n => {
                const sourceId = edge.source.id || edge.source;
                const targetId = edge.target.id || edge.target;
                return (n.id === sourceId || n.id === targetId) ? 1 : 0.3;
            });
    }

    // Zoom controls
    zoomIn() {
        this.svg.transition().call(this.zoom.scaleBy, 1.3);
    }

    zoomOut() {
        this.svg.transition().call(this.zoom.scaleBy, 0.7);
    }

    fitToScreen() {
        if (!this.nodes || this.nodes.length === 0) return;

        const bounds = this.g.node().getBBox();
        const fullWidth = this.width;
        const fullHeight = this.height;
        const width = bounds.width;
        const height = bounds.height;
        const midX = bounds.x + width / 2;
        const midY = bounds.y + height / 2;

        if (width === 0 || height === 0) return;

        const scale = 0.8 / Math.max(width / fullWidth, height / fullHeight);
        const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];

        this.svg.transition()
            .duration(750)
            .call(
                this.zoom.transform,
                d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
            );
    }

    // Render color legend
    renderLegend() {
        // Remove existing legend
        this.container.select('.graph-legend').remove();

        const legend = this.container.append('div')
            .attr('class', 'graph-legend');

        // Node Types Section
        legend.append('div')
            .attr('class', 'legend-section-title')
            .text('Node Types');

        const nodeTypes = [
            { type: 'initiator', label: 'INITIATOR', description: 'Session master', color: '#6b7280' },
            { type: 'root', label: 'ROOT', description: 'Main entry point', color: '#9333ea' },
            { type: 'delegate', label: 'DELEGATE', description: 'Delegated task', color: '#3b82f6' }
        ];

        nodeTypes.forEach(item => {
            const legendItem = legend.append('div')
                .attr('class', 'legend-item');

            legendItem.append('div')
                .attr('class', 'legend-color')
                .style('background-color', item.color);

            const legendText = legendItem.append('div')
                .attr('class', 'legend-text');

            legendText.append('div')
                .attr('class', 'legend-label')
                .text(item.label);

            legendText.append('div')
                .attr('class', 'legend-description')
                .text(item.description);
        });

        // Edge Types Section
        legend.append('div')
            .attr('class', 'legend-section-title')
            .style('margin-top', '12px')
            .text('Edge Types');

        const edgeTypes = [
            { type: 'agent_as_tool', label: 'Delegation', description: 'Agent delegates' },
            { type: 'ask_master', label: 'Ask Master', description: 'Question (dashed)' },
            { type: 'handoff', label: 'Handoff', description: 'Completion (thick)' }
        ];

        edgeTypes.forEach(item => {
            const legendItem = legend.append('div')
                .attr('class', 'legend-item');

            // Simple colored line element
            legendItem.append('div')
                .attr('class', 'legend-edge-line')
                .style('background-color', this.getEdgeColorByType(item.type))
                .style('border-style', item.type === 'ask_master' ? 'dashed' : 'solid')
                .style('border-top', item.type === 'ask_master'
                    ? `3px dashed ${this.getEdgeColorByType(item.type)}`
                    : `${item.type === 'handoff' ? '5px' : '3px'} solid ${this.getEdgeColorByType(item.type)}`)
                .style('width', '40px')
                .style('height', '0');

            const legendText = legendItem.append('div')
                .attr('class', 'legend-text');

            legendText.append('div')
                .attr('class', 'legend-label')
                .text(item.label);

            legendText.append('div')
                .attr('class', 'legend-description')
                .text(item.description);
        });
    }

    // Cleanup
    destroy() {
        if (this.simulation) {
            this.simulation.stop();
        }
        this.container.selectAll('*').remove();
    }
}

// Initialize global graph visualization
window.GraphViz = new GraphVisualization('graphCanvas');
