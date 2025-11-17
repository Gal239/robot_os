/**
 * Echo Robotics Lab - Camera Gallery Component
 * Grid display with lightbox for camera screenshots
 */

class CameraGallery {
    /**
     * Render camera gallery
     * @param {string} containerId - Container element ID
     * @param {Object} screenshots - Camera screenshots {name: base64}
     */
    static render(containerId, screenshots = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[CameraGallery] Container ${containerId} not found`);
            return;
        }

        container.innerHTML = '';

        const gallery = document.createElement('div');
        gallery.className = 'camera-gallery camera-gallery-fullscreen';

        // Grid (no header for immersive view)
        const grid = document.createElement('div');
        grid.className = 'camera-gallery-grid';

        const cameraNames = Object.keys(screenshots);

        if (cameraNames.length === 0) {
            const emptyState = this.createEmptyState();
            gallery.appendChild(emptyState);
        } else {
            cameraNames.forEach(name => {
                const card = this.createCameraCard(name, screenshots[name]);
                grid.appendChild(card);
            });
            gallery.appendChild(grid);
        }

        container.appendChild(gallery);
    }

    /**
     * Create header
     */
    static createHeader(count) {
        const header = document.createElement('div');
        header.className = 'camera-gallery-header';

        header.innerHTML = `
            <div class="camera-gallery-title">
                <i data-feather="camera" class="camera-gallery-title-icon"></i>
                <span class="camera-gallery-title-text">Camera Views</span>
            </div>
            <div class="camera-gallery-count">
                <span>${count}</span>
                <span>${count === 1 ? 'camera' : 'cameras'}</span>
            </div>
        `;

        return header;
    }

    /**
     * Create camera card
     */
    static createCameraCard(name, base64Image) {
        const card = document.createElement('div');
        card.className = 'camera-card';

        const imageUrl = `data:image/png;base64,${base64Image}`;

        card.innerHTML = `
            <div class="camera-card-image-container">
                <img class="camera-card-image" src="${imageUrl}" alt="${name}">
                <div class="camera-card-badge">${name}</div>
            </div>
            <div class="camera-card-info">
                <div class="camera-card-name">${name}</div>
                <div class="camera-card-meta">
                    <div class="camera-card-meta-item">
                        <span>📐</span>
                        <span>16:9</span>
                    </div>
                    <div class="camera-card-meta-item">
                        <span>🎯</span>
                        <span>PNG</span>
                    </div>
                </div>
            </div>
        `;

        // Add click handler for lightbox
        card.addEventListener('click', () => {
            this.openLightbox(name, imageUrl);
        });

        return card;
    }

    /**
     * Create empty state
     */
    static createEmptyState() {
        const empty = document.createElement('div');
        empty.className = 'camera-gallery-empty';

        empty.innerHTML = `
            <i data-feather="camera" class="camera-gallery-empty-icon"></i>
            <div class="camera-gallery-empty-title">No cameras yet</div>
            <div class="camera-gallery-empty-text">
                Build a scene to see camera views from different angles.
            </div>
        `;

        // Add lava lamp particles
        this.createLavaLampParticles(empty);

        return empty;
    }

    /**
     * Create lava lamp rising particles
     */
    static createLavaLampParticles(container) {
        // Use shared LavaLamp utility
        LavaLamp.create(container, 50);
    }

    /**
     * Open lightbox modal
     */
    static openLightbox(name, imageUrl) {
        const lightbox = document.createElement('div');
        lightbox.className = 'camera-lightbox';
        lightbox.id = 'camera-lightbox';

        lightbox.innerHTML = `
            <div class="camera-lightbox-content">
                <img class="camera-lightbox-image" src="${imageUrl}" alt="${name}">
                <button class="camera-lightbox-close">✕</button>
                <div class="camera-lightbox-info">${name}</div>
            </div>
        `;

        document.body.appendChild(lightbox);

        // Close on background click
        lightbox.addEventListener('click', (e) => {
            if (e.target === lightbox) {
                this.closeLightbox();
            }
        });

        // Close on close button click
        const closeBtn = lightbox.querySelector('.camera-lightbox-close');
        closeBtn.addEventListener('click', () => this.closeLightbox());

        // Close on Escape key
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                this.closeLightbox();
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
    }

    /**
     * Close lightbox
     */
    static closeLightbox() {
        const lightbox = document.getElementById('camera-lightbox');
        if (lightbox) {
            lightbox.remove();
        }
    }

    /**
     * Update camera gallery
     */
    static update(containerId, screenshots) {
        this.render(containerId, screenshots);
    }
}

// Export for use in app.js
window.CameraGallery = CameraGallery;
