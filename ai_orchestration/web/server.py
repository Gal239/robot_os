"""
Web Server Entry Point
Run this file to start the AI Orchestration Dashboard
"""

import os
import sys
import uvicorn
from pathlib import Path

# Force unbuffered output so print statements show immediately
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

def main():
    """Start the agent_orc_web server"""
    # Get the directory of this file
    app_dir = Path(__file__).parent

    print(f"""
    ========================================================
      AI Orchestration Dashboard
    ========================================================

    Starting server...
    Static files: {app_dir / 'static'}
    Templates: {app_dir / 'templates'}

    Dashboard will be available at:
       http://localhost:8002

    Press CTRL+C to stop the server
    ========================================================
    """)

    # Run the server (reload=False to see debug logs)
    uvicorn.run(
        "ai_orchestration.agent_orc_web.api:app",
        host="0.0.0.0",
        port=8002,
        reload=False,  # Disabled to see debug output
        log_level="info"
    )

if __name__ == "__main__":
    main()
