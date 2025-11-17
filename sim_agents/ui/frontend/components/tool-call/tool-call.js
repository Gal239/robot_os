/**
 * Echo Robotics Lab - Tool Call Component
 * Professional collapsible tool execution displays
 */

class ToolCall {
    /**
     * Create a tool call element
     * @param {Object} data - Tool call data
     * @param {string} data.name - Tool name (e.g., 'edit_script')
     * @param {string} data.summary - Short summary of what the tool is doing
     * @param {Object} data.input - Tool input parameters
     * @param {Object} data.output - Tool output/result (optional)
     * @param {string} data.status - 'running', 'success', 'error'
     * @param {string} data.icon - Feather icon name (optional)
     * @param {boolean} data.compact - Use compact mode (optional)
     * @returns {HTMLElement} - Tool call element
     */
    static create(data) {
        const {
            name = 'unknown',
            summary = '',
            input = {},
            output = null,
            status = 'running',
            icon = null,
            compact = false
        } = data;

        // Get icon for tool if not provided
        const toolIcon = icon || this.getIconForTool(name);

        // Create container
        const toolCall = document.createElement('div');
        toolCall.className = `tool-call tool-${name}`;
        if (compact) {
            toolCall.classList.add('compact');
        }

        // Create header
        const header = this.createHeader(name, summary, status, toolIcon);
        toolCall.appendChild(header);

        // Create body (collapsible)
        const body = this.createBody(input, output);
        toolCall.appendChild(body);

        // Add click handler to toggle expansion
        header.addEventListener('click', () => {
            toolCall.classList.toggle('expanded');
        });

        return toolCall;
    }

    /**
     * Create tool call header
     */
    static createHeader(name, summary, status, iconName) {
        const header = document.createElement('div');
        header.className = 'tool-call-header';

        // Left side
        const left = document.createElement('div');
        left.className = 'tool-call-header-left';

        const iconEl = document.createElement('i');
        iconEl.className = 'tool-call-icon';
        iconEl.setAttribute('data-feather', iconName);

        const nameEl = document.createElement('span');
        nameEl.className = 'tool-call-name';
        nameEl.textContent = name;

        if (summary) {
            const summaryEl = document.createElement('span');
            summaryEl.className = 'tool-call-summary';
            summaryEl.textContent = summary;
            left.append(iconEl, nameEl, summaryEl);
        } else {
            left.append(iconEl, nameEl);
        }

        // Right side
        const right = document.createElement('div');
        right.className = 'tool-call-header-right';

        const statusBadge = document.createElement('span');
        statusBadge.className = `tool-call-status-badge ${status}`;

        const statusDot = document.createElement('span');
        statusDot.className = `status-dot ${status}`;

        const statusText = document.createElement('span');
        statusText.textContent = status;

        statusBadge.append(statusDot, statusText);

        const expandIcon = document.createElement('i');
        expandIcon.className = 'tool-call-expand-icon';
        expandIcon.setAttribute('data-feather', 'chevron-down');

        right.append(statusBadge, expandIcon);

        header.append(left, right);

        // Initialize Feather icons for this element
        if (typeof feather !== 'undefined') {
            setTimeout(() => feather.replace(), 0);
        }

        return header;
    }

    /**
     * Create tool call body (details)
     */
    static createBody(input, output) {
        const body = document.createElement('div');
        body.className = 'tool-call-body';

        const content = document.createElement('div');
        content.className = 'tool-call-content';

        // Input section
        if (input && Object.keys(input).length > 0) {
            const inputSection = document.createElement('div');
            inputSection.className = 'tool-call-section';

            const inputTitle = document.createElement('div');
            inputTitle.className = 'tool-call-section-title';
            inputTitle.textContent = 'Input';

            const inputContent = document.createElement('div');
            inputContent.className = 'tool-call-input';

            const inputPre = document.createElement('pre');
            inputPre.textContent = JSON.stringify(input, null, 2);
            inputContent.appendChild(inputPre);

            inputSection.append(inputTitle, inputContent);
            content.appendChild(inputSection);
        }

        // Output section (if available)
        if (output) {
            const outputSection = document.createElement('div');
            outputSection.className = 'tool-call-section';

            const outputTitle = document.createElement('div');
            outputTitle.className = 'tool-call-section-title';
            outputTitle.textContent = 'Output';

            const outputContent = document.createElement('div');
            outputContent.className = 'tool-call-output';

            const outputPre = document.createElement('pre');
            outputPre.textContent = typeof output === 'string'
                ? output
                : JSON.stringify(output, null, 2);
            outputContent.appendChild(outputPre);

            outputSection.append(outputTitle, outputContent);
            content.appendChild(outputSection);
        }

        body.appendChild(content);
        return body;
    }

    /**
     * Get Feather icon name for tool
     * @param {string} toolName - Tool name
     * @returns {string} - Feather icon name
     */
    static getIconForTool(toolName) {
        const icons = {
            'edit_script': 'edit-3',
            'handoff': 'send',
            'ask_master': 'help-circle',
            'load_to_context': 'folder-plus',
            'write_file': 'save',
            'read_file': 'file-text',
            'list_files': 'list',
            'search_files': 'search',
            'browse_workspace': 'folder',
            'search_web': 'globe',
            'stop_and_think': 'pause-circle',
            'plan_next_steps': 'check-square'
        };

        return icons[toolName] || 'tool';
    }

    /**
     * Update tool call status
     * @param {HTMLElement} toolCallEl - Tool call element
     * @param {string} status - New status
     * @param {Object} output - Output data (optional)
     */
    static updateStatus(toolCallEl, status, output = null) {
        // Update status badge
        const statusBadge = toolCallEl.querySelector('.tool-call-status-badge');
        if (statusBadge) {
            statusBadge.className = `tool-call-status-badge ${status}`;
            const statusText = statusBadge.querySelector('span:last-child');
            if (statusText) {
                statusText.textContent = status;
            }
        }

        // Update status dot
        const statusDot = toolCallEl.querySelector('.status-dot');
        if (statusDot) {
            statusDot.className = `status-dot ${status}`;
        }

        // Add output if provided
        if (output) {
            const content = toolCallEl.querySelector('.tool-call-content');
            if (content) {
                // Remove existing output section
                const existingOutput = content.querySelector('.tool-call-section:last-child');
                if (existingOutput && existingOutput.querySelector('.tool-call-section-title').textContent === 'Output') {
                    existingOutput.remove();
                }

                // Add new output section
                const outputSection = document.createElement('div');
                outputSection.className = 'tool-call-section';

                const outputTitle = document.createElement('div');
                outputTitle.className = 'tool-call-section-title';
                outputTitle.textContent = 'Output';

                const outputContent = document.createElement('div');
                outputContent.className = 'tool-call-output';

                const outputPre = document.createElement('pre');
                outputPre.textContent = typeof output === 'string'
                    ? output
                    : JSON.stringify(output, null, 2);
                outputContent.appendChild(outputPre);

                outputSection.append(outputTitle, outputContent);
                content.appendChild(outputSection);
            }
        }
    }
}

// Export for use in app.js
window.ToolCall = ToolCall;
