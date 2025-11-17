"""
Monitor test progress in real-time by watching database
"""
import time
import asyncio
from pathlib import Path
from ai_orchestration.utils.global_config import agent_engine_db

async def monitor_latest_session():
    """Monitor the latest session and print progress"""
    runs_dir = Path("ai_orchestration/databases/runs")

    # Get latest session
    sessions = sorted(runs_dir.glob("session_*"))
    if not sessions:
        print("No sessions found")
        return

    latest = sessions[-1]
    session_id = latest.name.replace("session_", "")
    print(f"Monitoring session: {session_id}")
    print("=" * 80)

    seen_events = set()

    while True:
        try:
            # Load graph from DB
            graph_data = agent_engine_db.graphs.get(session_id)
            if not graph_data:
                time.sleep(2)
                continue

            # Extract timeline events
            nodes = graph_data.get("nodes", {})

            for task_id, node in nodes.items():
                timeline = node.get("timeline", [])

                for idx, event in enumerate(timeline):
                    event_key = f"{task_id}_{idx}"

                    if event_key not in seen_events:
                        seen_events.add(event_key)

                        # Print event
                        event_type = event.get("event_type")
                        tool_name = event.get("tool_name", "")
                        tool_input = event.get("tool_input", {})
                        result = event.get("result", {})

                        if event_type == "ROOT":
                            print(f"\n[ROOT] Task: {event.get('tool_input')}")
                        elif event_type == "function_tool":
                            print(f"  [{tool_name}] Input: {str(tool_input)[:80]}")
                            if result:
                                print(f"    → Result: {str(result)[:80]}")
                        elif event_type == "agent_as_tool":
                            print(f"  [DELEGATE → {tool_name}] {str(tool_input)[:80]}")
                        elif event_type == "handoff":
                            print(f"  [HANDOFF] {str(tool_input)[:80]}")

            # Check if complete
            all_complete = all(
                node.get("status") in ["completed", "failed"]
                for node in nodes.values()
            )

            if all_complete:
                print("\n" + "=" * 80)
                print("All tasks complete!")
                break

            time.sleep(2)

        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    asyncio.run(monitor_latest_session())
