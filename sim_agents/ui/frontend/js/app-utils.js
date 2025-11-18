/**
 * App Utilities
 * ==============
 * Shared utility functions used across the application
 */

const AppUtils = {
    /**
     * Wait for specified milliseconds
     * @param {number} ms - Milliseconds to wait
     * @returns {Promise}
     */
    wait(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
};

// Export for window access
window.AppUtils = AppUtils;
