#!/usr/bin/env python3
"""
Script to clear all contents from the simulation_center/database directory.
Use with caution - this permanently deletes all database files and folders.
"""

import shutil
import os
from pathlib import Path


def clear_database():
    """Delete all contents in the database directory."""
    # Get the database directory path
    script_dir = Path(__file__).parent

    database_dir = script_dir.parent / "database"

    if not database_dir.exists():
        print(f"Database directory does not exist: {database_dir}")
        return


    print(f"Clearing database directory: {database_dir}")
    print("=" * 60)

    # Count items before deletion
    items = list(database_dir.iterdir())
    total_items = len(items)

    if total_items == 0:
        print("Database directory is already empty.")
        return

    # Ask for confirmation
    print(f"Found {total_items} items to delete:")
    for item in items[:10]:  # Show first 10 items
        print(f"  - {item.name}")
    if total_items > 10:
        print(f"  ... and {total_items - 10} more items")

    response = input("\nAre you sure you want to delete all these items? (yes/no): ")

    if response.lower() != 'yes':
        print("Operation cancelled.")
        return

    # Delete all items
    deleted_count = 0
    failed_count = 0

    for item in items:
        try:
            if item.is_dir():
                shutil.rmtree(item)
                print(f"Deleted directory: {item.name}")
            else:
                item.unlink()
                print(f"Deleted file: {item.name}")
            deleted_count += 1
        except Exception as e:
            print(f"Failed to delete {item.name}: {e}")
            failed_count += 1

    print("=" * 60)
    print(f"Deletion complete!")
    print(f"  Successfully deleted: {deleted_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Database directory: {database_dir}")


if __name__ == "__main__":
    clear_database()
