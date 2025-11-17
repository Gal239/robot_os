#!/usr/bin/env python3
"""
AutoDB - Purely Local File Storage
Simple JSON-based database for agent orchestration
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


class LocalBackend:
    """Local file backend - saves to directories as JSON"""

    def __init__(self, root_path: str = "data/local_db"):
        self.root = Path(root_path)
        self.root.mkdir(parents=True, exist_ok=True)
        print(f"[DB] Local backend initialized: {self.root}")

    def set(self, collection: str, doc_id: str, data: Dict) -> bool:
        """Save document to local JSON file"""
        col_dir = self.root / collection
        col_dir.mkdir(parents=True, exist_ok=True)

        # Add timestamps
        data['_updated'] = datetime.now().isoformat()
        if '_created' not in data:
            data['_created'] = datetime.now().isoformat()

        doc_path = col_dir / f"{doc_id}.json"
        doc_path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
        return True

    def get(self, collection: str, doc_id: str) -> Optional[Dict]:
        """Load document from local JSON file"""
        doc_path = self.root / collection / f"{doc_id}.json"
        if doc_path.exists():
            return json.loads(doc_path.read_text(encoding='utf-8'))
        return None

    def delete(self, collection: str, doc_id: str) -> bool:
        """Delete local JSON file"""
        doc_path = self.root / collection / f"{doc_id}.json"
        if doc_path.exists():
            doc_path.unlink()
            return True
        return False

    def list(self, collection: str) -> List[str]:
        """List all document IDs in collection"""
        col_dir = self.root / collection
        if col_dir.exists():
            return [f.stem for f in col_dir.glob("*.json")]
        return []

    def exists(self, collection: str, doc_id: str) -> bool:
        """Check if document exists"""
        doc_path = self.root / collection / f"{doc_id}.json"
        return doc_path.exists()

    def query(self, collection: str, field: str, value: Any) -> List[Dict]:
        """Query documents by field value"""
        results = []
        col_dir = self.root / collection
        if col_dir.exists():
            for doc_path in col_dir.glob("*.json"):
                data = json.loads(doc_path.read_text(encoding='utf-8'))
                if data.get(field) == value:
                    data['_id'] = doc_path.stem
                    results.append(data)
        return results

    def clear(self, collection: str) -> bool:
        """Clear all documents in collection"""
        col_dir = self.root / collection
        if col_dir.exists():
            for doc_path in col_dir.glob("*.json"):
                doc_path.unlink()
        return True


class Collection:
    """Collection interface - dict-like access to documents"""

    def __init__(self, name: str, backend):
        self.name = name
        self.backend = backend
        self._cache = {}

    def __setitem__(self, key: str, value: Any):
        """collection["key"] = value"""
        assert key, "Document ID cannot be empty"
        assert isinstance(value, dict), f"Value must be dict, got {type(value)}"

        self.backend.set(self.name, key, value)
        self._cache[key] = value

    def __getitem__(self, key: str) -> Any:
        """value = collection["key"]"""
        assert key, "Document ID cannot be empty"

        if key in self._cache:
            return self._cache[key]

        value = self.backend.get(self.name, key)
        if value is None:
            raise KeyError(f"Document not found: {key}")

        self._cache[key] = value
        return value

    def __delitem__(self, key: str):
        """del collection["key"]"""
        assert key, "Document ID cannot be empty"

        self.backend.delete(self.name, key)
        if key in self._cache:
            del self._cache[key]

    def __contains__(self, key: str) -> bool:
        """if "key" in collection:"""
        return self.backend.exists(self.name, key)

    def __len__(self) -> int:
        """len(collection)"""
        return len(self.backend.list(self.name))

    def __iter__(self):
        """for key in collection:"""
        return iter(self.backend.list(self.name))

    def get(self, key: str, default: Any = None) -> Any:
        """collection.get("key", default)"""
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, key: str, updates: Dict):
        """collection.update("key", {"field": "value"})"""
        assert key, "Document ID cannot be empty"
        assert isinstance(updates, dict), "Updates must be dict"

        if key in self:
            current = self[key]
            current.update(updates)
            self[key] = current

    def find(self, **kwargs) -> List[Dict]:
        """collection.find(field=value)"""
        if not kwargs:
            return []

        # Use first field for query
        field, value = next(iter(kwargs.items()))
        return self.backend.query(self.name, field, value)

    def where(self, field: str, value: Any) -> List[Dict]:
        """collection.where("field", value)"""
        assert field, "Field name cannot be empty"
        return self.backend.query(self.name, field, value)

    def all(self) -> List[Dict]:
        """collection.all()"""
        doc_ids = self.backend.list(self.name)
        docs = []
        for doc_id in doc_ids:
            doc = self.backend.get(self.name, doc_id)
            if doc:
                doc['_id'] = doc_id
                docs.append(doc)
        return docs

    def clear(self):
        """collection.clear()"""
        self.backend.clear(self.name)
        self._cache.clear()

    def add(self, data: Dict) -> str:
        """Add with auto-generated ID"""
        import uuid
        doc_id = str(uuid.uuid4())[:8]
        self[doc_id] = data
        return doc_id


class AutoDB:
    """
    Purely Local Database
    Simple JSON file storage with dict-like interface
    """

    def __init__(self, local_path: str = "data/local_db"):
        """Initialize local-only AutoDB"""
        self.backend = LocalBackend(local_path)
        self._collections = {}

    def __getattr__(self, name: str) -> Collection:
        """db.collection_name creates/gets collection"""
        if name.startswith('_'):
            raise AttributeError(f"No attribute: {name}")

        if name not in self._collections:
            self._collections[name] = Collection(name, self.backend)

        return self._collections[name]

    def __getitem__(self, name: str) -> Collection:
        """db["collection_name"] same as db.collection_name"""
        return getattr(self, name)

    # ========== WORKSPACE FILE OPERATIONS ==========

    def save_workspace(self, session_id: str, workspace_data: Dict):
        """
        Save workspace metadata to runs/{session}/workspace.json

        Workspace data contains:
        - session_id
        - documents: {path: {content_json, metadata, ...}}
        """
        import json
        workspace_path = self.backend.root / "runs" / session_id / "workspace.json"
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        workspace_path.write_text(json.dumps(workspace_data, indent=2, default=str), encoding='utf-8')

    def load_workspace(self, session_id: str) -> Optional[Dict]:
        """Load workspace metadata from runs/{session}/workspace.json"""
        import json
        workspace_path = self.backend.root / "runs" / session_id / "workspace.json"
        if workspace_path.exists():
            return json.loads(workspace_path.read_text(encoding='utf-8'))
        return None

    def load_workspace_file(self, session_id: str, path: str) -> str:
        """Load file from runs/{session}/workspace/{path}"""
        file_path = self.backend.root / "runs" / session_id / "workspace" / path
        if file_path.exists():
            return file_path.read_text(encoding='utf-8')
        raise FileNotFoundError(f"File not found: {path}")

    def save_workspace_file(self, session_id: str, path: str, content: str):
        """Save file to runs/{session}/workspace/{path}"""
        file_path = self.backend.root / "runs" / session_id / "workspace" / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')

    def delete_workspace_file(self, session_id: str, path: str) -> bool:
        """Delete file from workspace"""
        file_path = self.backend.root / "runs" / session_id / "workspace" / path
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_workspace_files(self, session_id: str) -> List[str]:
        """List all files in workspace"""
        workspace_dir = self.backend.root / "runs" / session_id / "workspace"
        if workspace_dir.exists():
            return [str(f.relative_to(workspace_dir)) for f in workspace_dir.rglob("*") if f.is_file()]
        return []

    def workspace_file_exists(self, session_id: str, path: str) -> bool:
        """Check if workspace file exists"""
        file_path = self.backend.root / "runs" / session_id / "workspace" / path
        return file_path.exists()

    # ========== END WORKSPACE FILE OPERATIONS ==========

    # ========== RUN MANAGEMENT (Graph + Logs in one place) ==========

    def save_run_graph(self, session_id: str, graph_data: Dict):
        """Save task graph to runs/{session}/graph.json"""
        import json
        graph_path = self.backend.root / "runs" / session_id / "graph.json"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(json.dumps(graph_data, indent=2, default=str), encoding='utf-8')

    def save_run_logs(self, session_id: str, logs_data: Dict):
        """Save logs to runs/{session}/logs.json"""
        import json
        logs_path = self.backend.root / "runs" / session_id / "logs.json"
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        logs_path.write_text(json.dumps(logs_data, indent=2, default=str), encoding='utf-8')

    def save_run_log_file(self, session_id: str, filename: str, log_data):
        """Save individual log file to runs/{session}/{filename}"""
        import json
        log_path = self.backend.root / "runs" / session_id / filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(log_data, indent=2, default=str), encoding='utf-8')

    def save_run_snapshot(self, session_id: str, snapshot_data: Dict):
        """Save snapshot to runs/{session}/snapshots/graph_{timestamp}.json"""
        import json
        from datetime import datetime
        timestamp = datetime.now().strftime('%H%M%S')
        snapshot_dir = self.backend.root / "runs" / session_id / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"graph_{timestamp}.json"
        snapshot_path.write_text(json.dumps(snapshot_data, indent=2, default=str), encoding='utf-8')

    def save_run_agents(self, session_id: str, agents_data: Dict):
        """Save agents to runs/{session}/agents.json"""
        import json
        agents_path = self.backend.root / "runs" / session_id / "agents.json"
        agents_path.parent.mkdir(parents=True, exist_ok=True)
        agents_path.write_text(json.dumps(agents_data, indent=2, default=str), encoding='utf-8')

    def load_run_graph(self, session_id: str) -> Dict:
        """Load task graph from runs/{session}/graph.json"""
        import json
        graph_path = self.backend.root / "runs" / session_id / "graph.json"
        if graph_path.exists():
            return json.loads(graph_path.read_text(encoding='utf-8'))
        raise FileNotFoundError(f"Graph not found for session: {session_id}")

    def load_run_logs(self, session_id: str) -> Dict:
        """Load logs from runs/{session}/logs.json"""
        import json
        logs_path = self.backend.root / "runs" / session_id / "logs.json"
        if logs_path.exists():
            return json.loads(logs_path.read_text(encoding='utf-8'))
        raise FileNotFoundError(f"Logs not found for session: {session_id}")

    def load_run_log_file(self, session_id: str, filename: str):
        """Load individual log file from runs/{session}/{filename}"""
        import json
        log_path = self.backend.root / "runs" / session_id / filename
        if log_path.exists():
            return json.loads(log_path.read_text(encoding='utf-8'))
        raise FileNotFoundError(f"Log file '{filename}' not found for session: {session_id}")

    def load_run_agents(self, session_id: str) -> Dict:
        """Load agents from runs/{session}/agents.json"""
        import json
        agents_path = self.backend.root / "runs" / session_id / "agents.json"
        if agents_path.exists():
            return json.loads(agents_path.read_text(encoding='utf-8'))
        return {}  # Return empty dict if no agents.json

    def list_runs(self) -> List[str]:
        """List all run session IDs"""
        runs_dir = self.backend.root / "runs"
        if runs_dir.exists():
            return [d.name for d in runs_dir.iterdir() if d.is_dir()]
        return []

    # ========== END RUN MANAGEMENT ==========

    def info(self):
        """Show database info"""
        print(f"""
{'='*60}
AutoDB - Local Mode
{'='*60}
Path: {self.backend.root}
{'='*60}
        """)

        # Show collections
        root = Path(self.backend.root)
        if root.exists():
            collections = [d for d in root.iterdir() if d.is_dir()]
            if collections:
                print("Collections:")
                for col_dir in collections:
                    count = len(list(col_dir.glob("*.json")))
                    print(f"  {col_dir.name}: {count} documents")
            else:
                print("No collections yet")


# Convenience function
def create_db(local_path: str = "data/local_db") -> AutoDB:
    """Create local AutoDB instance"""
    return AutoDB(local_path=local_path)


# Example usage
if __name__ == "__main__":
    print("AUTODB - LOCAL MODE DEMO")
    print("="*60)

    # Create DB
    db = AutoDB(local_path="test_db")
    db.info()

    # Test CRUD operations
    print("\n📝 Testing CRUD operations:")

    # Create
    db.users["alice"] = {"name": "Alice", "age": 30}
    print(f"✅ Created: {db.users['alice']}")

    # Read
    user = db.users["alice"]
    print(f"✅ Read: {user}")

    # Update
    db.users.update("alice", {"age": 31})
    print(f"✅ Updated: {db.users['alice']}")

    # List
    print(f"✅ All users: {list(db.users)}")

    # Delete
    del db.users["alice"]
    print("✅ Deleted alice")

    print("\n✅ All operations work!")
