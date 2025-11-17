"""
DB MODEL - Base class for database-backed models
Eliminates CRUD duplication across Agent, Org, etc.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from ai_orchestration.utils.global_config import agent_engine_db


class DBModel:
    """
    Base class for models backed by agent_engine_db

    Subclasses must define:
    - collection_name: str (e.g., "agents", "orgs")

    Provides:
    - CRUD operations (load, save, list_all, get_all, delete)
    - Timestamp management
    """

    collection_name: str = None  # Override in subclass

    def __init__(self, config: Dict[str, Any], id: Optional[str] = None):
        """
        Initialize model with config

        Args:
            config: Configuration dictionary
            id: Optional ID (will be extracted from config if not provided)
        """
        if self.collection_name is None:
            raise NotImplementedError("Subclass must define collection_name")

        self.config = config
        self.id = id or config.get('id', config.get('name', 'unnamed'))

    @classmethod
    def load(cls, id: str) -> 'DBModel':
        """Load model from database by ID"""
        collection = getattr(agent_engine_db, cls.collection_name)
        config = collection[id]
        return cls(config, id)

    def save(self) -> str:
        """Save model to database, returns ID"""
        # Update timestamps
        self.config['_updated'] = datetime.now().isoformat()
        if '_created' not in self.config:
            self.config['_created'] = datetime.now().isoformat()

        # Save to collection
        collection = getattr(agent_engine_db, self.collection_name)
        collection[self.id] = self.config

        return self.id

    @classmethod
    def list_all(cls) -> List[str]:
        """List all IDs in collection"""
        collection = getattr(agent_engine_db, cls.collection_name)
        return list(collection)

    @classmethod
    def get_all(cls) -> List['DBModel']:
        """Get all models as objects"""
        return [cls.load(id) for id in cls.list_all()]

    def delete(self):
        """Delete model from database"""
        collection = getattr(agent_engine_db, self.collection_name)
        del collection[self.id]
