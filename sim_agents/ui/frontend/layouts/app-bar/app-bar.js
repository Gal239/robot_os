/**
 * App Bar Layout
 * ===============
 *
 * Top navigation bar
 */

export class AppBar {
    /**
     * Render app bar
     *
     * @returns {string} HTML string
     */
    static render() {
        return `
            <div class="app-bar">
                <div class="app-bar-left">
                    <span class="app-bar-icon">🤖</span>
                    <h1 class="app-bar-title">Echo v0.1 - Scene Builder</h1>
                </div>
                <div class="app-bar-right">
                    <button class="app-bar-button" title="Help">
                        ❓ Help
                    </button>
                    <button class="app-bar-button" title="Settings">
                        ⚙️
                    </button>
                </div>
            </div>
        `;
    }
}
