/**
 * Tabbed Area Layout
 * ===================
 *
 * Main content area with tabs (Scene, Script, Data, Metalog)
 */

import { TabButton } from '../../components/tab-button/tab-button.js';

export class TabbedArea {
    /**
     * Render tabbed area
     *
     * @param {Object} data - Content for each tab
     * @param {string} activeTab - Active tab ID
     * @returns {string} HTML string
     */
    static render(data = {}, activeTab = 'scene') {
        return `
            <div class="tabbed-area" x-data="{ activeTab: '${activeTab}' }">
                <!-- Tab Navigation -->
                <div class="tab-nav">
                    ${TabButton.render('scene', 'Scene', '🎬', activeTab === 'scene')}
                    ${TabButton.render('script', 'Script', '📝', activeTab === 'script')}
                    ${TabButton.render('data', 'Data', '📊', activeTab === 'data')}
                    ${TabButton.render('metalog', 'Metalog', '📜', activeTab === 'metalog')}
                </div>

                <!-- Tab Content -->
                <div class="tab-content-wrapper">
                    <!-- Scene Tab -->
                    <div x-show="activeTab === 'scene'"
                         x-transition:enter="tab-transition"
                         class="tab-content">
                        ${this.renderSceneTab(data.scene || {})}
                    </div>

                    <!-- Script Tab -->
                    <div x-show="activeTab === 'script'"
                         x-transition:enter="tab-transition"
                         class="tab-content">
                        ${this.renderScriptTab(data.script || '')}
                    </div>

                    <!-- Data Tab -->
                    <div x-show="activeTab === 'data'"
                         x-transition:enter="tab-transition"
                         class="tab-content">
                        ${this.renderDataTab(data.sceneData || {})}
                    </div>

                    <!-- Metalog Tab -->
                    <div x-show="activeTab === 'metalog'"
                         x-transition:enter="tab-transition"
                         class="tab-content">
                        ${this.renderMetalogTab(data.metalog || '')}
                    </div>
                </div>
            </div>
        `;
    }

    static renderSceneTab(sceneData) {
        const screenshots = sceneData.screenshots || {};
        const cameraIds = Object.keys(screenshots);

        if (cameraIds.length === 0) {
            return `
                <div class="empty-state">
                    <p>🎬 No scene created yet</p>
                    <p class="empty-state-hint">Chat with Echo to build your first scene!</p>
                </div>
            `;
        }

        return `
            <div class="scene-viewer">
                <h3>Scene Screenshots</h3>
                <p class="scene-hint">📸 ${cameraIds.length} camera(s) available</p>

                <div class="camera-grid">
                    ${cameraIds.map(camId => `
                        <div class="camera-card card">
                            <div class="camera-header">
                                <span>📷 ${camId}</span>
                            </div>
                            <img
                                src="${screenshots[camId]}"
                                alt="${camId}"
                                class="camera-image"
                                loading="lazy"
                            />
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    static renderScriptTab(script) {
        if (!script) {
            return `
                <div class="empty-state">
                    <p>📝 No script available</p>
                </div>
            `;
        }

        return `
            <div class="script-viewer">
                <div class="script-header">
                    <h3>Scene Script</h3>
                    <button class="button-primary" onclick="copyScript()">
                        📋 Copy
                    </button>
                </div>
                <pre class="script-code"><code class="language-python">${this.escapeHtml(script)}</code></pre>
            </div>
        `;
    }

    static renderDataTab(sceneData) {
        if (!sceneData || Object.keys(sceneData).length === 0) {
            return `
                <div class="empty-state">
                    <p>📊 No scene data available</p>
                </div>
            `;
        }

        return `
            <div class="data-viewer">
                <h3>Scene Data</h3>
                <div class="data-sections">
                    <div class="card">
                        <h4>Cameras (${(sceneData.cameras || []).length})</h4>
                        <p>Camera information will be displayed here</p>
                    </div>
                    <div class="card">
                        <h4>Assets (${(sceneData.assets || []).length})</h4>
                        <p>Asset information will be displayed here</p>
                    </div>
                    <div class="card">
                        <h4>Sensors (${(sceneData.sensors || []).length})</h4>
                        <p>Sensor information will be displayed here</p>
                    </div>
                </div>
            </div>
        `;
    }

    static renderMetalogTab(metalog) {
        if (!metalog) {
            return `
                <div class="empty-state">
                    <p>📜 No conversation context yet</p>
                </div>
            `;
        }

        return `
            <div class="metalog-viewer">
                <h3>Conversation Context</h3>
                <pre class="metalog-content">${this.escapeHtml(metalog)}</pre>
            </div>
        `;
    }

    static escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
}
