/**
 * Echo Robotics Lab - Scene Inspector Component
 * Displays scene metadata in a professional tree view
 */

class SceneInspector {
    /**
     * Render scene inspector
     * @param {string} containerId - Container element ID
     * @param {Object} sceneData - Scene metadata
     */
    static render(containerId, sceneData = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[SceneInspector] Container ${containerId} not found`);
            return;
        }

        container.innerHTML = '';

        const inspector = document.createElement('div');
        inspector.className = 'scene-inspector scene-inspector-fullscreen';

        // Content only (no header for immersive view)
        const content = document.createElement('div');
        content.className = 'scene-inspector-content scene-inspector-content-fullscreen';

        const keys = Object.keys(sceneData);

        if (keys.length === 0) {
            const emptyState = this.createEmptyState();
            content.appendChild(emptyState);
        } else {
            const list = this.createDataList(sceneData);
            content.appendChild(list);
        }

        inspector.appendChild(content);
        container.appendChild(inspector);
    }

    /**
     * Create header
     */
    static createHeader() {
        const header = document.createElement('div');
        header.className = 'scene-inspector-header';

        header.innerHTML = `
            <div class="scene-inspector-title">
                <i data-feather="search" class="scene-inspector-title-icon"></i>
                <span class="scene-inspector-title-text">Scene Data</span>
            </div>
        `;

        return header;
    }

    /**
     * Create data list
     */
    static createDataList(data) {
        const list = document.createElement('div');
        list.className = 'scene-inspector-list';

        Object.entries(data).forEach(([key, value]) => {
            const item = this.createDataItem(key, value);
            list.appendChild(item);
        });

        return list;
    }

    /**
     * Create data item
     */
    static createDataItem(key, value) {
        const item = document.createElement('div');
        item.className = 'scene-inspector-item';

        const valueType = this.getValueType(value);

        // Header
        const header = document.createElement('div');
        header.className = 'scene-inspector-item-header';

        const keyEl = document.createElement('div');
        keyEl.className = 'scene-inspector-item-key';
        keyEl.textContent = key;

        const typeEl = document.createElement('div');
        typeEl.className = 'scene-inspector-item-type';
        typeEl.textContent = valueType;

        header.append(keyEl, typeEl);
        item.appendChild(header);

        // Value
        const valueEl = document.createElement('div');
        valueEl.className = `scene-inspector-item-value ${valueType}`;

        if (valueType === 'object' && !Array.isArray(value)) {
            // Nested object
            const nested = this.createNestedView(value);
            valueEl.appendChild(nested);
        } else if (valueType === 'array') {
            // Array
            valueEl.textContent = `[${value.length} ${value.length === 1 ? 'item' : 'items'}]`;
            const nested = this.createNestedView(value);
            valueEl.appendChild(nested);
        } else {
            // Primitive value
            valueEl.textContent = this.formatValue(value);
        }

        item.appendChild(valueEl);
        return item;
    }

    /**
     * Create nested view for objects/arrays
     */
    static createNestedView(data) {
        const nested = document.createElement('div');
        nested.className = 'scene-inspector-nested';

        if (Array.isArray(data)) {
            data.forEach((item, index) => {
                const nestedItem = document.createElement('div');
                nestedItem.className = 'scene-inspector-nested-item';

                const nestedKey = document.createElement('div');
                nestedKey.className = 'scene-inspector-nested-key';
                nestedKey.textContent = `[${index}]`;

                const nestedValue = document.createElement('div');
                nestedValue.className = 'scene-inspector-nested-value';
                nestedValue.textContent = this.formatValue(item);

                nestedItem.append(nestedKey, nestedValue);
                nested.appendChild(nestedItem);
            });
        } else {
            Object.entries(data).forEach(([key, value]) => {
                const nestedItem = document.createElement('div');
                nestedItem.className = 'scene-inspector-nested-item';

                const nestedKey = document.createElement('div');
                nestedKey.className = 'scene-inspector-nested-key';
                nestedKey.textContent = key;

                const nestedValue = document.createElement('div');
                nestedValue.className = 'scene-inspector-nested-value';
                nestedValue.textContent = this.formatValue(value);

                nestedItem.append(nestedKey, nestedValue);
                nested.appendChild(nestedItem);
            });
        }

        return nested;
    }

    /**
     * Get value type
     */
    static getValueType(value) {
        if (value === null) return 'null';
        if (Array.isArray(value)) return 'array';
        if (typeof value === 'object') return 'object';
        if (typeof value === 'boolean') return 'boolean';
        if (typeof value === 'number') return 'number';
        if (typeof value === 'string') return 'string';
        return 'unknown';
    }

    /**
     * Format value for display
     */
    static formatValue(value) {
        if (value === null) return 'null';
        if (value === undefined) return 'undefined';
        if (typeof value === 'boolean') return value.toString();
        if (typeof value === 'number') return value.toString();
        if (typeof value === 'string') return `"${value}"`;
        if (Array.isArray(value)) return `[${value.length}]`;
        if (typeof value === 'object') return '{...}';
        return String(value);
    }

    /**
     * Create empty state
     */
    static createEmptyState() {
        const empty = document.createElement('div');
        empty.className = 'scene-inspector-empty';

        empty.innerHTML = `
            <i data-feather="search" class="scene-inspector-empty-icon"></i>
            <div class="scene-inspector-empty-title">No scene data</div>
            <div class="scene-inspector-empty-text">
                Scene metadata will appear here once you build a scene.
            </div>
        `;

        // Add lava lamp particles
        LavaLamp.create(empty, 50);

        return empty;
    }

    /**
     * Update scene inspector
     */
    static update(containerId, sceneData) {
        this.render(containerId, sceneData);
    }
}

// Export for use in app.js
window.SceneInspector = SceneInspector;
