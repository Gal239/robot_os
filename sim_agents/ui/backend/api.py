"""
Flask API Routes
Thin layer connecting HTTP to EchoConversationManager
"""
from flask import Flask, jsonify, request, Response, stream_with_context
import asyncio
import json
import time
from echo_ops import EchoConversationManager

app = Flask(__name__)
echo_manager = None


@app.route('/api/status')
def status():
    """Get Echo's current status"""
    return jsonify({
        "success": True,
        "status": echo_manager.get_status() if echo_manager else "not_started"
    })


@app.route('/api/start', methods=['POST'])
def start():
    """
    Start new session:
    1. Create EchoConversationManager ONCE (creates ops)
    2. Start session (creates agent, returns welcome message)

    IMPORTANT: Only creates manager once! Reuses same instance.
    """
    global echo_manager

    # Only create if doesn't exist - REUSE same manager for whole session!
    if echo_manager is None:
        echo_manager = EchoConversationManager()
        result = echo_manager.start_session()
    else:
        # Already started, return existing session info
        result = {
            "success": True,
            "session_id": echo_manager.ops.graph_ops.modal.session_id,
            "message": "Session already active. Use /api/chat to continue conversation."
        }

    return jsonify(result)


@app.route('/api/chat', methods=['POST'])
def chat():
    """Send message to Echo"""
    if not echo_manager:
        return jsonify({
            "success": False,
            "error": "Session not started"
        }), 400

    data = request.get_json()
    message = data.get('message', '').strip()

    if not message:
        return jsonify({
            "success": False,
            "error": "Empty message"
        }), 400

    try:
        # Run async send_message
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(echo_manager.send_message(message))
        loop.close()

        return jsonify(result)

    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/conversation')
def conversation():
    """Get conversation history"""
    if not echo_manager:
        return jsonify({"success": True, "conversation": []})

    return jsonify({
        "success": True,
        "conversation": echo_manager.get_conversation()
    })


@app.route('/api/screenshots')
def screenshots():
    """Get latest screenshots"""
    if not echo_manager:
        return jsonify({"success": True, "screenshots": {}})

    return jsonify({
        "success": True,
        "screenshots": echo_manager.get_screenshots()
    })


@app.route('/api/scene-script')
def scene_script():
    """Get current scene script"""
    if not echo_manager:
        return jsonify({"success": True, "script": ""})

    return jsonify({
        "success": True,
        "script": echo_manager.get_scene_script()
    })


@app.route('/api/scene-data')
def scene_data():
    """Get scene metadata"""
    if not echo_manager:
        return jsonify({"success": True, "scene_data": {}})

    return jsonify({
        "success": True,
        "scene_data": echo_manager.get_scene_data()
    })


@app.route('/api/metalog')
def metalog():
    """Get metalog (conversation context)"""
    if not echo_manager:
        return jsonify({"success": True, "metalog": ""})

    return jsonify({
        "success": True,
        "metalog": echo_manager.get_metalog()
    })


@app.route('/api/edits/stream')
def edits_stream():
    """
    Server-Sent Events endpoint for streaming document edits

    Returns edit operations as they happen for live UI animation
    Format: Server-Sent Events (SSE)
    """
    if not echo_manager:
        return jsonify({"success": False, "error": "Session not started"}), 400

    def generate():
        """Generator function for SSE"""
        # Get edit stream from document_ops
        document_ops = echo_manager.ops.document_ops
        edit_stream = document_ops.get_edit_stream()

        # Send each edit as SSE
        for edit in edit_stream:
            # Format as SSE (data: prefix required)
            yield f"data: {json.dumps(edit)}\n\n"
            time.sleep(0.2)  # 200ms delay between edits

        # Clear edit stream after sending
        document_ops.clear_edit_stream()

        # Send completion event
        yield f"data: {json.dumps({'event': 'complete'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'  # Disable nginx buffering
        }
    )


@app.route('/api/edits')
def edits():
    """Get current edit stream (polling alternative to SSE)"""
    if not echo_manager:
        return jsonify({"success": True, "edits": []})

    document_ops = echo_manager.ops.document_ops
    edit_stream = document_ops.get_edit_stream()

    return jsonify({
        "success": True,
        "edits": edit_stream
    })


# ========== WORKSPACE ENDPOINTS ==========

@app.route('/api/workspace/script')
def workspace_script():
    """
    Get live script from workspace WITH imports

    Returns complete executable script (imports auto-prepended)
    """
    if not echo_manager:
        return jsonify({"success": True, "script": "", "has_imports": False})

    script = echo_manager.get_scene_script_from_workspace()
    lines = script.split('\n') if script else []

    return jsonify({
        "success": True,
        "script": script,
        "line_count": len(lines),
        "has_imports": "import sys" in script
    })


@app.route('/api/workspace/blocks')
def workspace_blocks():
    """
    Get script as blocks (agent view)

    Shows how agent sees the script:
    Block 0: ops = ExperimentOps(...)
    Block 1: ops.create_scene(...)
    """
    if not echo_manager:
        return jsonify({"success": True, "blocks": "No script yet - create initial scene!", "block_count": 0})

    block_view = echo_manager.get_script_blocks()

    # Count actual blocks
    doc = echo_manager.ops.document_ops.get_document("scene_script.py")
    block_count = len(doc.content_json.get("blocks", {})) if doc else 0

    return jsonify({
        "success": True,
        "blocks": block_view,
        "block_count": block_count
    })


@app.route('/api/workspace/info')
def workspace_info():
    """
    Get workspace metadata

    Returns session info, documents list, etc.
    """
    if not echo_manager:
        return jsonify({"success": True, "workspace": {"session_id": None, "documents": [], "document_count": 0}})

    info = echo_manager.get_workspace_info()

    return jsonify({
        "success": True,
        "workspace": info
    })


@app.route('/api/workspace/edits/clear', methods=['POST'])
def clear_workspace_edits():
    """Clear edit stream after UI has consumed it"""
    if not echo_manager:
        return jsonify({"success": False, "error": "Session not started"}), 400

    echo_manager.clear_edit_stream()

    return jsonify({
        "success": True,
        "message": "Edit stream cleared"
    })
