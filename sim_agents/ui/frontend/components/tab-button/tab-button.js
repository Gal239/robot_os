/**
 * Tab Button Component
 * =====================
 *
 * Navigation tab button
 */

export class TabButton {
    /**
     * Render a tab button
     *
     * @param {string} id - Tab ID
     * @param {string} label - Tab label
     * @param {string} icon - Emoji icon
     * @param {boolean} active - Is active tab
     * @returns {string} HTML string
     */
    static render(id, label, icon, active = false) {
        const activeClass = active ? 'tab-button-active' : '';

        return `
            <button
                class="tab-button ${activeClass}"
                @click="activeTab = '${id}'"
                :class="{ 'tab-button-active': activeTab === '${id}' }"
            >
                <span class="tab-icon">${icon}</span>
                <span class="tab-label">${label}</span>
            </button>
        `;
    }
}
