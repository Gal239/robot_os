#!/usr/bin/env python3
"""
Echo Scene Builder Server - Dream Factory
Minimal Flask server with static file serving
"""
import sys
from pathlib import Path

# Add simulation_center to path
simulation_center = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(simulation_center))

# Add backend to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from flask import send_from_directory, make_response
from api import app


@app.route('/')
def index():
    """Serve frontend index"""
    response = make_response(send_from_directory('../frontend', 'index.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/<path:path>')
def serve_static(path):
    """Serve frontend static files with no-cache headers in dev mode"""
    response = make_response(send_from_directory('../frontend', path))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == '__main__':
    import os

    print("\n" + "="*60)
    print("Echo Scene Builder - Dream Factory")
    print("="*60)
    print("\nServer: http://localhost:5050")
    print("Frontend: Professional design system")
    print("Backend: Flask API with MOP orchestrator")
    print("\n🔥 AUTO-RELOAD ENABLED - Changes will rebuild automatically")
    print("   (database/workspace excluded from watch)")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")

    # Add only frontend files to watch (exclude database/workspace)
    # stat reloader only checks files in extra_files list, not entire directory
    extra_dirs = [
        str(Path(__file__).parent / '../frontend'),
        str(Path(__file__).parent),  # backend code
    ]
    extra_files = []
    for extra_dir in extra_dirs:
        extra_dir_path = Path(extra_dir).resolve()
        if extra_dir_path.exists():
            for filepath in extra_dir_path.rglob('*'):
                # Skip database/workspace directories
                if 'database' in str(filepath) or 'workspace' in str(filepath):
                    continue
                if filepath.is_file() and filepath.suffix in ['.html', '.css', '.js', '.svg', '.py']:
                    extra_files.append(str(filepath))

    app.run(
        host='0.0.0.0',
        port=5050,
        debug=True,
        extra_files=extra_files,
        use_reloader=True,
        reloader_type='stat'  # Use stat reloader instead of watchdog (checks only extra_files)
    )
