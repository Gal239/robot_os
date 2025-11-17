/* ========================================
   MAIN JS - Simple Documentation Page
   ======================================== */

(function() {
    'use strict';

    // ========================================
    // Initialize Mermaid Diagrams
    // ========================================

    function initMermaid() {
        if (typeof mermaid !== 'undefined') {
            mermaid.initialize({
                startOnLoad: true,
                theme: 'default',
                flowchart: {
                    useMaxWidth: true,
                    htmlLabels: true,
                    curve: 'basis'
                }
            });
        } else {
            console.warn('Mermaid.js not loaded');
        }
    }

    // ========================================
    // Smooth Scroll for Anchor Links
    // ========================================

    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                const href = this.getAttribute('href');

                // Skip if it's just "#"
                if (href === '#') return;

                e.preventDefault();
                const target = document.querySelector(href);

                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    // ========================================
    // Initialize on DOM Ready
    // ========================================

    function init() {
        initMermaid();
        initSmoothScroll();
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
