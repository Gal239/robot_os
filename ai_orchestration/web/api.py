"""
FastAPI Backend for AI Orchestration Visualization Dashboard
Provides REST API endpoints to serve session data, graphs, and logs
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from existing codebase
from ai_orchestration.utils.global_config import agent_engine_db
from ai_orchestration.core.ops.task_graph_ops import TaskGraphOps
from ai_orchestration.core.ops.log_ops import LogOps

app = FastAPI(title="AI Orchestration Dashboard API", version="1.0.0")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files with no-cache headers
static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"

# Custom middleware to disable caching for static files
@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)

    # Disable caching for static files (JS, CSS)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response

if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# ========== API ENDPOINTS ==========

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard HTML with cache-busting timestamps"""
    logger.info(">>> DASHBOARD PAGE REQUESTED <<<")
    index_file = templates_path / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")

    # Read HTML and inject version timestamp
    import time
    import re
    version = int(time.time() * 1000)  # Current timestamp in milliseconds
    html_content = index_file.read_text(encoding='utf-8')

    # Add version to all static file URLs using regex
    # Change: /static/js/file.js -> /static/js/file.js?v=123456
    html_content = re.sub(
        r'(/static/(js|css)/[^"\'?\s]+)',
        rf'\1?v={version}',
        html_content
    )

    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content, headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })


@app.get("/api/sessions")
async def list_sessions() -> Dict[str, Any]:
    """
    List all available sessions
    Returns: {sessions: [{session_id, created_at, task_count, status}]}
    """
    try:
        # Use existing list_runs() method
        session_ids = agent_engine_db.list_runs()

        if not session_ids:
            return {"sessions": [], "total": 0}

        sessions = []
        for session_id in session_ids:
            try:
                # Load graph using existing method
                graph_data = agent_engine_db.load_run_graph(session_id)
                nodes = graph_data.get("nodes", {})

                # Calculate statistics
                task_count = len(nodes)
                completed = sum(1 for n in nodes.values() if n.get("status") == "completed")
                running = sum(1 for n in nodes.values() if n.get("status") == "running")
                waiting = sum(1 for n in nodes.values() if n.get("status") == "waiting")

                sessions.append({
                    "session_id": session_id,
                    "created_at": graph_data.get("updated_at", ""),
                    "task_count": task_count,
                    "completed": completed,
                    "running": running,
                    "waiting": waiting,
                    "status": "running" if running > 0 else "completed"
                })
            except Exception as e:
                print(f"Error loading session {session_id}: {e}")
                continue

        # Sort by created_at descending (most recent first)
        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return {"sessions": sessions, "total": len(sessions)}

    except Exception as e:
        print(f"Error listing sessions: {e}")
        return {"sessions": [], "total": 0}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    """
    Get complete session data including graph and logs
    Returns: {session_id, graph, logs, metadata}
    """
    try:
        # Load graph
        graph_data = agent_engine_db.load_run_graph(session_id)

        # Load logs (try to load each log file)
        logs = {}
        log_files = ["master_log.json", "metalog_detailed.json", "metalog_summary.json", "task_logs.json"]

        for log_file in log_files:
            try:
                log_data = agent_engine_db.load_run_log_file(session_id, log_file)
                log_name = log_file.replace(".json", "")
                logs[log_name] = log_data
            except:
                logs[log_file.replace(".json", "")] = []

        # Calculate metadata
        nodes = graph_data.get("nodes", {})
        metadata = {
            "session_id": session_id,
            "total_tasks": len(nodes),
            "completed": sum(1 for n in nodes.values() if n.get("status") == "completed"),
            "running": sum(1 for n in nodes.values() if n.get("status") == "running"),
            "waiting": sum(1 for n in nodes.values() if n.get("status") == "waiting"),
            "ready": sum(1 for n in nodes.values() if n.get("status") == "ready"),
            "agents": list(set(n.get("agent_id") for n in nodes.values())),
            "updated_at": graph_data.get("updated_at", "")
        }

        # Fetch agent details for all agents in session
        agents_in_session = metadata["agents"]
        agent_details = {}

        # Load agents from session file (agents.json) instead of global database
        try:
            all_agents_data = agent_engine_db.load_run_agents(session_id)
            logger.info(f"Loaded {len(all_agents_data)} agents from session {session_id}")
        except Exception as e:
            logger.warning(f"No agents.json for session {session_id}: {e}")
            all_agents_data = {}

        for agent_id in agents_in_session:
            # Count tasks for this agent
            task_count = sum(1 for n in nodes.values() if n.get("agent_id") == agent_id)

            # Calculate tool calls and duration for this agent
            tool_calls = 0
            total_duration = 0
            for node in nodes.values():
                if node.get("agent_id") == agent_id:
                    tool_calls += len(node.get("tool_timeline", []))
                    if node.get("completed_at") and node.get("created_at"):
                        try:
                            start = datetime.fromisoformat(node.get("created_at"))
                            end = datetime.fromisoformat(node.get("completed_at"))
                            total_duration += (end - start).total_seconds()
                        except:
                            pass

            # Check if agent exists in database
            if agent_id in all_agents_data:
                agent_config = all_agents_data[agent_id]
                agent_details[agent_id] = {
                    "agent_id": agent_id,
                    "description": agent_config.get("description", ""),
                    "force_model": agent_config["force_model"],  # MOP: Show actual agent model, no fallback
                    "tools": agent_config.get("tools", []),
                    "instructions": agent_config.get("instructions", ""),
                    "task_count": task_count,
                    "tool_calls": tool_calls,
                    "total_duration": total_duration
                }
            else:
                # Fallback for agents not in database (e.g., "human")
                agent_details[agent_id] = {
                    "agent_id": agent_id,
                    "description": f"External agent: {agent_id}",
                    "force_model": "unknown",
                    "tools": [],
                    "instructions": "",
                    "task_count": task_count,
                    "tool_calls": tool_calls,
                    "total_duration": total_duration
                }

        return {
            "session_id": session_id,
            "graph": graph_data,
            "logs": logs,
            "metadata": metadata,
            "agents": agent_details
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading session: {str(e)}")


@app.get("/api/sessions/{session_id}/graph")
async def get_session_graph(session_id: str) -> Dict[str, Any]:
    """
    Get graph structure optimized for visualization
    Returns: {nodes: [{id, label, agent, status, parent, ...}], edges: [{source, target, type}]}
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"API CALLED: /api/sessions/{session_id}/graph")
    logger.info(f"{'='*60}\n")
    try:
        graph_data = agent_engine_db.load_run_graph(session_id)
        nodes_data = graph_data.get("nodes", {})

        # Build nodes for visualization
        nodes = []
        edges = []
        edge_id_counter = 0

        # CREATE VIRTUAL INITIATOR NODES (for visualization only)
        # Find all parent_task_id values that don't exist as nodes (e.g., "human")
        initiator_ids = set()
        for task_id, node_data in nodes_data.items():
            parent_id = node_data.get("parent_task_id")
            # If parent exists and is not in nodes_data, it's an initiator
            if parent_id and parent_id not in nodes_data:
                initiator_ids.add(parent_id)

        # Create virtual nodes for initiators
        for initiator_id in initiator_ids:
            nodes_data[initiator_id] = {
                "task_id": initiator_id,
                "agent_id": initiator_id,
                "parent_task_id": None,
                "tool_type": "initiator",
                "status": "completed",
                "task_payload": f"Session Master: {initiator_id}",
                "created_at": "",
                "completed_at": "",
                "tool_timeline": [],
                "blockers": [],
                "result": None
            }
            print(f"[VIEW] Created virtual initiator node: {initiator_id}")

        # REMAP PARENT RELATIONSHIPS: Bypass ask_master nodes
        # If a node's parent is ask_master, remap to grandparent
        print(f"\n{'='*60}")
        print(f"Remapping parent relationships to bypass ask_master nodes")
        print(f"{'='*60}")

        ask_master_nodes = {tid: tdata for tid, tdata in nodes_data.items() if tdata.get("tool_type") == "ask_master"}

        for task_id, node_data in nodes_data.items():
            parent_id = node_data.get("parent_task_id")

            # If parent is ask_master, remap to grandparent
            if parent_id and parent_id in ask_master_nodes:
                grandparent_id = ask_master_nodes[parent_id].get("parent_task_id")
                print(f"  Remapping {task_id}: parent {parent_id} (ask_master) → grandparent {grandparent_id}")
                node_data["parent_task_id"] = grandparent_id

        # Add ALL nodes (including ask_master and initiators) - style them differently in UI
        for task_id, node_data in nodes_data.items():
            nodes.append({
                "id": task_id,
                "label": node_data.get("task_payload", "")[:50],  # Truncate for display
                "agent_id": node_data.get("agent_id", ""),
                "status": node_data.get("status", "ready"),
                "tool_type": node_data.get("tool_type", ""),
                "parent_id": node_data.get("parent_task_id"),
                "master_agent_id": node_data.get("master_agent_id"),
                "created_at": node_data.get("created_at", ""),
                "completed_at": node_data.get("completed_at"),
                "tool_calls": len(node_data.get("tool_timeline", [])),
                "blockers": list(node_data.get("blockers", [])) if isinstance(node_data.get("blockers"), set) else node_data.get("blockers", [])
            })

        # First pass: Build parent-to-children mapping
        children_map = {}
        for tid, tdata in nodes_data.items():
            parent = tdata.get("parent_task_id")
            if parent:
                if parent not in children_map:
                    children_map[parent] = []
                children_map[parent].append(tid)

        # Second pass: Create edges from tool_timeline events
        print(f"\n{'='*60}")
        print(f"Processing timeline events to create edges")
        print(f"{'='*60}")

        for task_id, node_data in nodes_data.items():
            tool_timeline = node_data.get("tool_timeline", [])
            parent_task_id = node_data.get("parent_task_id")
            task_children = children_map.get(task_id, [])

            if tool_timeline:
                print(f"\nTask {task_id}:")
                print(f"  Timeline events: {len(tool_timeline)}")
                print(f"  Parent: {parent_task_id}")
                print(f"  Children: {task_children}")

            for event in tool_timeline:
                event_type = event.get("type", "")
                tool_name = event.get("tool", "").lower()

                # Determine edge type
                edge_type = None

                # AGENT_AS_TOOL (delegation): route_to_*, ask_claude, ask_gpt, etc
                if (event_type == "agent_as_tool" or
                    "agent_as_tool" in tool_name or
                    "route_to_" in tool_name or
                    "ask_claude" in tool_name or
                    "ask_gpt" in tool_name):
                    edge_type = "agent_as_tool"

                # ASK_MASTER (questions to master): ask_master, ask_data, etc
                elif (event_type == "ask_master" or
                      "ask_master" in tool_name or
                      "ask_data" in tool_name):
                    edge_type = "ask_master"

                # HANDOFF (completion): handoff
                elif event_type == "handoff" or "handoff" in tool_name:
                    edge_type = "handoff"

                if not edge_type:
                    print(f"  [SKIP] Event with no edge_type: type={event_type}, tool={tool_name}")
                    continue

                print(f"  [{edge_type}] tool={tool_name}, timestamp={event.get('timestamp', 'N/A')[:19]}")

                # AGENT_AS_TOOL: edge FROM this task TO its children (delegation)
                # The event is in the PARENT's timeline, creating edges to children
                if edge_type == "agent_as_tool":
                    # Skip "start_session" - handled in THIRD PASS (initiator delegation)
                    if tool_name == "start_session":
                        print(f"    → Skipped start_session (handled in THIRD PASS)")
                        continue

                    # Skip "result from" tools - these are return messages, not delegation
                    if "result from" in tool_name or "← result from" in tool_name:
                        print(f"    → Skipped result from tool (return message)")
                        continue

                    # Try to extract child task from result string
                    result_str = str(event.get("result", ""))
                    target_task = None

                    # Parse "→ delegated to X (task task_N)" or similar
                    import re
                    match = re.search(r'task[_\s]+(\d+)', result_str)
                    if match:
                        target_task = f"task_{match.group(1)}"

                    # Fallback: if there's exactly one child, use it
                    if not target_task and len(task_children) == 1:
                        target_task = task_children[0]

                    if target_task and target_task in nodes_data:
                        edges.append({
                            "id": f"edge_{edge_id_counter}",
                            "source": task_id,          # FROM current task
                            "target": target_task,      # TO child task
                            "type": "agent_as_tool",
                            "status": nodes_data[target_task].get("status", "ready"),
                            "timestamp": event.get("timestamp", ""),
                            "tool": event.get("tool", ""),
                            "input": event.get("input", {}),
                            "result": event.get("result", "")
                        })
                        print(f"    ✓ Created edge: {task_id} → {target_task}")
                        edge_id_counter += 1
                    else:
                        print(f"    ✗ Failed: target_task={target_task}, exists={target_task in nodes_data if target_task else False}")

                # ASK_MASTER: Skip! We handle these in FOURTH PASS (convert nodes to edges)
                elif edge_type == "ask_master":
                    print(f"    → Skipped ask_master event (handled in FOURTH PASS)")

                # HANDOFF: edge FROM this task TO parent (returning result)
                elif edge_type == "handoff":
                    # Skip handoff FROM human tasks - handled in FOURTH PASS as answer edge
                    if node_data.get("agent_id") == "human":
                        print(f"    → Skipped handoff from human task (handled in FOURTH PASS)")
                        continue

                    if parent_task_id and parent_task_id in nodes_data:
                        edges.append({
                            "id": f"edge_{edge_id_counter}",
                            "source": task_id,
                            "target": parent_task_id,
                            "type": "handoff",
                            "status": node_data.get("status", "ready"),
                            "timestamp": event.get("timestamp", ""),
                            "tool": event.get("tool", ""),
                            "input": event.get("input", {}),
                            "result": event.get("result", "")
                        })
                        print(f"    ✓ Created edge: {task_id} → {parent_task_id}")
                        edge_id_counter += 1
                    else:
                        print(f"    ✗ Failed: parent_task_id={parent_task_id}, exists={parent_task_id in nodes_data if parent_task_id else False}")

        # THIRD PASS: Create delegation edges FROM initiator TO ROOT tasks
        print(f"\n{'='*60}")
        print(f"Creating delegation edges from initiator to ROOT tasks")
        print(f"{'='*60}")

        for task_id, node_data in nodes_data.items():
            if node_data.get("tool_type") == "root":
                parent_id = node_data.get("parent_task_id")
                if parent_id and parent_id in initiator_ids:
                    # Create delegation edge: initiator → ROOT task
                    edges.append({
                        "id": f"edge_{edge_id_counter}",
                        "source": parent_id,
                        "target": task_id,
                        "type": "agent_as_tool",
                        "status": node_data.get("status", "ready"),
                        "timestamp": node_data.get("created_at", ""),
                        "tool": "start_session",
                        "input": {"task": node_data.get("task_payload", "")[:200]},
                        "result": f"Delegated to {node_data.get('agent_id', 'unknown')}"
                    })
                    print(f"    ✓ Created initiator edge: {parent_id} → {task_id}")
                    edge_id_counter += 1

        # FOURTH PASS: Convert ask_master NODES into EDGES
        # For nodes with tool_type="ask_master", create TWO edges:
        # 1. Ask edge: parent → grandparent (asking)
        # 2. Handoff edge: grandparent → parent (answering)
        print(f"\n{'='*60}")
        print(f"Converting ask_master nodes to edges")
        print(f"{'='*60}")

        for task_id, node_data in nodes_data.items():
            if node_data.get("tool_type") == "ask_master":
                # task_1 has parent_task_id=task_0 (who asked)
                asker_task_id = node_data.get("parent_task_id")
                if asker_task_id and asker_task_id in nodes_data:
                    asker_node = nodes_data[asker_task_id]
                    # task_0 has parent_task_id=human (who will answer)
                    answerer_id = asker_node.get("parent_task_id")

                    if answerer_id and answerer_id in nodes_data:
                        # 1. Create ask_master edge: task_0 → human (ASKING)
                        edges.append({
                            "id": f"edge_{edge_id_counter}",
                            "source": asker_task_id,
                            "target": answerer_id,
                            "type": "ask_master",
                            "status": node_data.get("status", "ready"),
                            "timestamp": node_data.get("created_at", ""),
                            "tool": "ask_master",
                            "input": {"question": node_data.get("task_payload", "")},
                            "result": f"Asked {answerer_id}"
                        })
                        print(f"    ✓ Created ask_master edge: {asker_task_id} → {answerer_id}")
                        edge_id_counter += 1

                        # 2. Create handoff edge: human → task_0 (ANSWERING)
                        # Only if the ask_master task is completed (has answer)
                        if node_data.get("status") == "completed":
                            edges.append({
                                "id": f"edge_{edge_id_counter}",
                                "source": answerer_id,
                                "target": asker_task_id,
                                "type": "handoff",
                                "status": "completed",
                                "timestamp": node_data.get("completed_at", ""),
                                "tool": "handoff",
                                "input": node_data.get("result", {}),
                                "result": node_data.get("result", {})
                            })
                            print(f"    ✓ Created handoff edge: {answerer_id} → {asker_task_id}")
                            edge_id_counter += 1
                    else:
                        print(f"    ✗ Failed: answerer_id={answerer_id}, exists={answerer_id in nodes_data if answerer_id else False}")
                else:
                    print(f"    ✗ Failed: asker_task_id={asker_task_id}, exists={asker_task_id in nodes_data if asker_task_id else False}")

        # FIFTH PASS: Remap edges to bypass ask_master nodes
        print(f"\n{'='*60}")
        print(f"Remapping edges to bypass ask_master nodes")
        print(f"{'='*60}")

        ask_master_node_ids = {tid for tid, tdata in nodes_data.items() if tdata.get("tool_type") == "ask_master"}

        for edge in edges:
            source_id = edge.get("source")
            target_id = edge.get("target")
            changed = False

            # If source is ask_master, replace with its parent (who actually did the work)
            if source_id in ask_master_node_ids:
                parent_of_source = nodes_data[source_id].get("parent_task_id")
                if parent_of_source:
                    print(f"  Remapping edge source: {source_id} (ask_master) → {parent_of_source}")
                    edge["source"] = parent_of_source
                    changed = True

            # If target is ask_master, replace with its first child (where work continues)
            if target_id in ask_master_node_ids:
                # Find children of this ask_master node
                children = children_map.get(target_id, [])
                if children:
                    new_target = children[0]  # Use first child
                    print(f"  Remapping edge target: {target_id} (ask_master) → {new_target}")
                    edge["target"] = new_target
                    changed = True
                else:
                    # No children, remove edge by marking it
                    edge["_remove"] = True
                    print(f"  Marking edge for removal: {source_id} → {target_id} (ask_master with no children)")

            if changed:
                print(f"    Result: {edge['source']} → {edge['target']}")

        # Remove marked edges
        edges = [e for e in edges if not e.get("_remove")]

        # Debug: Print edge statistics with detailed data inspection
        edge_types = {}
        for edge in edges:
            edge_type = edge.get("type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        print(f"\n{'='*80}")
        print(f"DEBUG: Graph for session {session_id}")
        print(f"  Total nodes (raw): {len(nodes_data)}")
        print(f"  Regular nodes (displayed): {len(nodes)}")
        print(f"  Total edges: {len(edges)}")
        print(f"  Edge types: {edge_types}")

        # Show detailed data for first edge of each type
        if edges:
            logger.info(f"\n  DETAILED EDGE DATA INSPECTION:")
            shown_types = set()
            for edge in edges:
                edge_type = edge.get("type")
                if edge_type not in shown_types:
                    shown_types.add(edge_type)
                    logger.info(f"\n  [{edge_type}] Edge Example:")
                    logger.info(f"    - tool: {edge.get('tool')}")
                    logger.info(f"    - timestamp: {edge.get('timestamp')}")
                    logger.info(f"    - has input: {bool(edge.get('input'))}")
                    logger.info(f"    - input type: {type(edge.get('input'))}")
                    logger.info(f"    - input value: {str(edge.get('input'))[:200]}")
                    logger.info(f"    - has result: {bool(edge.get('result'))}")
                    logger.info(f"    - result type: {type(edge.get('result'))}")
                    logger.info(f"    - result value: {str(edge.get('result'))[:200]}")
        logger.info(f"{'='*80}\n")

        return {
            "nodes": nodes,
            "edges": edges,
            "session_id": session_id
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading graph: {str(e)}")


@app.get("/api/sessions/{session_id}/logs")
async def get_session_logs(
    session_id: str,
    log_type: str = "master",  # master, metalog_detailed, metalog_summary, task_logs
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get logs for a session
    Returns: {logs: [...], session_id, log_type}
    """
    try:
        log_file_map = {
            "master": "master_log.json",
            "metalog_detailed": "metalog_detailed.json",
            "metalog_summary": "metalog_summary.json",
            "task_logs": "task_logs.json"
        }

        if log_type not in log_file_map:
            raise HTTPException(status_code=400, detail=f"Invalid log_type: {log_type}")

        log_data = agent_engine_db.load_run_log_file(session_id, log_file_map[log_type])

        # If task_logs requested and task_id provided, filter
        if log_type == "task_logs" and task_id:
            if isinstance(log_data, dict):
                log_data = log_data.get(task_id, [])

        return {
            "logs": log_data,
            "session_id": session_id,
            "log_type": log_type,
            "count": len(log_data) if isinstance(log_data, list) else len(log_data.keys())
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Logs not found for session {session_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading logs: {str(e)}")


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str, session_id: str) -> Dict[str, Any]:
    """
    Get detailed information for a single task
    Returns: {task_id, agent_id, status, timeline, messages, ...}
    """
    try:
        graph_data = agent_engine_db.load_run_graph(session_id)
        nodes = graph_data.get("nodes", {})

        if task_id not in nodes:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        task_data = nodes[task_id]

        # Get task logs
        try:
            task_logs_data = agent_engine_db.load_run_log_file(session_id, "task_logs.json")
            task_logs = task_logs_data.get(task_id, []) if isinstance(task_logs_data, dict) else []
        except:
            task_logs = []

        # Find children tasks
        children = [
            tid for tid, tdata in nodes.items()
            if tdata.get("parent_task_id") == task_id
        ]

        return {
            "task_id": task_id,
            **task_data,
            "task_logs": task_logs,
            "children": children
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading task: {str(e)}")


@app.get("/api/agents")
async def list_agents() -> Dict[str, Any]:
    """
    List all registered agents
    Returns: {agents: [{agent_id, description, tools, ...}]}
    """
    try:
        # Load agents from database
        agents_data = agent_engine_db.agents.list()

        agents = []
        for agent_id, agent_config in agents_data.items():
            agents.append({
                "agent_id": agent_id,
                "description": agent_config.get("description", ""),
                "tools": agent_config.get("tools", []),
                "instructions": agent_config.get("instructions", "")[:100],  # Truncate
                "force_model": agent_config.get("force_model", "")
            })

        return {
            "agents": agents,
            "total": len(agents)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading agents: {str(e)}")


@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/sessions/{session_id}/workspace")
async def get_workspace_files(session_id: str) -> Dict[str, Any]:
    """
    Get all files in the workspace directory for a session
    Returns: {files: [{path, name, size_bytes, created_at, created_by_task, created_by_agent, file_type}], total, has_workspace}
    """
    try:
        # Use agent_engine_db method to list workspace files (supports both local & MongoDB)
        file_paths = agent_engine_db.list_workspace_files(session_id)

        if not file_paths:
            return {
                "files": [],
                "total": 0,
                "has_workspace": False
            }

        # Load graph to match files to tasks
        graph_data = agent_engine_db.load_run_graph(session_id)
        nodes = graph_data.get("nodes", {})

        # Build a mapping of file paths to task info
        file_task_map = {}
        for task_id, node_data in nodes.items():
            tool_timeline = node_data.get("tool_timeline", [])
            for event in tool_timeline:
                if event.get("tool") == "write_file":
                    event_file_path = event.get("input", {}).get("path", "")
                    if event_file_path:
                        file_task_map[event_file_path] = {
                            "task_id": task_id,
                            "agent_id": node_data.get("agent_id", "unknown"),
                            "timestamp": event.get("timestamp", "")
                        }

        # Build file info list
        files = []
        for file_path in file_paths:
            # Normalize path separators
            file_path_normalized = file_path.replace("\\", "/")

            # Get file name
            file_name = Path(file_path).name

            # Determine file type
            suffix = Path(file_path).suffix.lower()
            if suffix == ".md":
                file_type = "markdown"
            elif suffix == ".json":
                file_type = "json"
            elif suffix in [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"]:
                file_type = "code"
            elif suffix in [".txt", ".log"]:
                file_type = "text"
            else:
                file_type = "unknown"

            # Get task info if available
            task_info = file_task_map.get(file_path_normalized, {})

            # Try to get file size (for local files)
            size_bytes = 0
            created_at = task_info.get("timestamp", "")
            try:
                # Only works for local files
                if hasattr(agent_engine_db.backend, 'root'):
                    full_path = agent_engine_db.backend.root / "runs" / session_id / "workspace" / file_path
                    if full_path.exists():
                        stat = full_path.stat()
                        size_bytes = stat.st_size
                        if not created_at:
                            created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
            except:
                pass

            files.append({
                "path": file_path_normalized,
                "name": file_name,
                "size_bytes": size_bytes,
                "created_at": created_at or datetime.now().isoformat(),
                "created_by_task": task_info.get("task_id"),
                "created_by_agent": task_info.get("agent_id"),
                "file_type": file_type,
                "is_directory": False
            })

        # Sort files by created_at descending
        files.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return {
            "files": files,
            "total": len(files),
            "has_workspace": True
        }

    except FileNotFoundError:
        return {
            "files": [],
            "total": 0,
            "has_workspace": False
        }
    except Exception as e:
        logger.error(f"Error loading workspace: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading workspace: {str(e)}")


@app.get("/api/sessions/{session_id}/workspace/file")
async def get_workspace_file(session_id: str, file_path: str) -> Dict[str, Any]:
    """
    Get the content of a specific workspace file
    Returns: {content, path, metadata}
    """
    try:
        # Validate file path (prevent directory traversal)
        if ".." in file_path or file_path.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid file path")

        # Load file content using agent_engine_db method (supports both local & MongoDB)
        content = agent_engine_db.load_workspace_file(session_id, file_path)

        # Try to find task info from graph
        graph_data = agent_engine_db.load_run_graph(session_id)
        nodes = graph_data.get("nodes", {})

        task_info = None
        for task_id, node_data in nodes.items():
            tool_timeline = node_data.get("tool_timeline", [])
            for event in tool_timeline:
                if event.get("tool") == "write_file":
                    event_path = event.get("input", {}).get("path", "")
                    # Normalize paths for comparison
                    normalized_event_path = event_path.replace("\\", "/")
                    normalized_file_path = file_path.replace("\\", "/")
                    if normalized_event_path == normalized_file_path:
                        task_info = {
                            "task_id": task_id,
                            "agent_id": node_data.get("agent_id", "unknown"),
                            "timestamp": event.get("timestamp", ""),
                            "why_created": event.get("input", {}).get("why_created", "")
                        }
                        break
            if task_info:
                break

        # Get file size (for local files only)
        size_bytes = 0
        created_at = task_info.get("timestamp", "") if task_info else ""
        try:
            if hasattr(agent_engine_db.backend, 'root'):
                full_path = agent_engine_db.backend.root / "runs" / session_id / "workspace" / file_path
                if full_path.exists():
                    stat = full_path.stat()
                    size_bytes = stat.st_size
                    if not created_at:
                        created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
        except:
            pass

        return {
            "content": content,
            "path": file_path,
            "metadata": {
                "size_bytes": size_bytes,
                "created_at": created_at or datetime.now().isoformat(),
                "file_type": Path(file_path).suffix[1:],  # Remove leading dot
                "task_info": task_info
            }
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File {file_path} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading file: {str(e)}")


@app.get("/api/sessions/{session_id}/snapshots")
async def get_snapshots(session_id: str) -> Dict[str, Any]:
    """
    Get list of graph snapshots over time
    Returns: {snapshots: [{timestamp, file_name, size_bytes}], total}
    """
    try:
        # Check if snapshots directory exists (local files only)
        if not hasattr(agent_engine_db.backend, 'root'):
            # MongoDB or other backend - snapshots not supported yet
            return {
                "snapshots": [],
                "total": 0
            }

        snapshots_path = agent_engine_db.backend.root / "runs" / session_id / "snapshots"

        if not snapshots_path.exists():
            return {
                "snapshots": [],
                "total": 0
            }

        # List all snapshot files
        snapshots = []
        for file in snapshots_path.iterdir():
            if file.is_file() and file.suffix == ".json":
                stat = file.stat()
                # Extract timestamp from filename (e.g., graph_132514.json -> 132514)
                timestamp_str = file.stem.replace("graph_", "")

                snapshots.append({
                    "timestamp": timestamp_str,
                    "file_name": file.name,
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        # Sort by timestamp ascending
        snapshots.sort(key=lambda x: x.get("timestamp", ""))

        return {
            "snapshots": snapshots,
            "total": len(snapshots)
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as e:
        logger.error(f"Error loading snapshots: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading snapshots: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
