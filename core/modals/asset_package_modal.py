"""
AssetPackageModal - Self-contained MuJoCo asset package with portable paths

PURE MOP:
- AUTO-DISCOVERY: Discovers all assets from compiled XML
- SELF-GENERATION: Creates manifest.json and package_info.json
- SELF-RENDERING: Rewrites XML with relative paths
- SELF-SAVING: Saves complete package to disk

This modal creates a portable MuJoCo package that can be:
1. Loaded by Python MuJoCo (mujoco.MjModel.from_xml_path)
2. Loaded by mujoco_wasm (browser simulation)
3. Used by mjc_viewer (trajectory visualization)
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass
import shutil
import hashlib
import json
from datetime import datetime


@dataclass
class AssetFile:
    """Single asset file metadata"""
    name: str  # Asset name from XML (e.g., "apple_mesh")
    source_path: Path  # Absolute source path
    package_path: str  # Relative path in package (e.g., "assets/meshes/apple.obj")
    size: int  # File size in bytes
    checksum: str  # SHA256 checksum


class AssetPackageModal:
    """PURE MOP: Self-contained asset package that saves itself

    Usage:
        # Lazy creation on first frame save
        package = AssetPackageModal(compiled_xml, experiment_dir)
        package.save(experiment_dir / "mujoco_package")

        # Now mujoco_package/ contains:
        # - scene.xml (relative paths)
        # - assets/meshes/*.obj
        # - assets/textures/*.png
        # - manifest.json
        # - package_info.json
    """

    def __init__(self, compiled_xml: str, experiment_id: str):
        """Initialize with compiled XML

        Args:
            compiled_xml: Complete MuJoCo XML with absolute asset paths
            experiment_id: Experiment identifier
        """
        self.compiled_xml = compiled_xml
        self.experiment_id = experiment_id

        # Parse XML
        self.xml_root = ET.fromstring(compiled_xml)

        # Auto-discovered assets (PURE MOP!)
        self.meshes: List[AssetFile] = []
        self.textures: List[AssetFile] = []
        self.models: List[AssetFile] = []  # Robot XMLs

        # Discover all assets
        self._discover_assets()

    def _discover_assets(self):
        """AUTO-DISCOVERY: Extract all asset paths from XML - OFFENSIVE!"""
        # Find <asset> section
        asset_section = self.xml_root.find('asset')
        if asset_section is None:
            # No assets - empty lists (legitimate for minimal scenes)
            return

        # Discover meshes
        for mesh in asset_section.findall('mesh'):
            file_path = mesh.get('file')
            if file_path:
                name = mesh.get('name', 'unnamed_mesh')
                self._add_mesh(name, file_path)

        # Discover textures
        for texture in asset_section.findall('texture'):
            file_path = texture.get('file')
            if file_path:
                name = texture.get('name', 'unnamed_texture')
                self._add_texture(name, file_path)

        # Discover materials (may reference textures)
        for material in asset_section.findall('material'):
            texture_ref = material.get('texture')
            if texture_ref:
                # Texture already discovered - skip
                pass

    def _add_mesh(self, name: str, source_path: str):
        """Add mesh to package - OFFENSIVE!"""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(
                f"Mesh file not found: {source}\n"
                f"Asset name: {name}\n"
                f"Check XMLResolver asset path resolution!"
            )

        # Determine extension and package path
        ext = source.suffix  # .obj, .stl, etc.
        package_path = f"assets/meshes/{source.name}"

        # Compute checksum
        checksum = self._compute_checksum(source)

        asset = AssetFile(
            name=name,
            source_path=source,
            package_path=package_path,
            size=source.stat().st_size,
            checksum=checksum
        )

        self.meshes.append(asset)

    def _add_texture(self, name: str, source_path: str):
        """Add texture to package - OFFENSIVE!"""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(
                f"Texture file not found: {source}\n"
                f"Asset name: {name}\n"
                f"Check XMLResolver asset path resolution!"
            )

        # Package path
        package_path = f"assets/textures/{source.name}"

        # Compute checksum
        checksum = self._compute_checksum(source)

        asset = AssetFile(
            name=name,
            source_path=source,
            package_path=package_path,
            size=source.stat().st_size,
            checksum=checksum
        )

        self.textures.append(asset)

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    def copy_assets(self, package_dir: Path):
        """Copy all assets to package directory - PURE MOP!

        Creates:
            package_dir/assets/meshes/*.obj
            package_dir/assets/textures/*.png
        """
        # Create directories
        meshes_dir = package_dir / "assets" / "meshes"
        textures_dir = package_dir / "assets" / "textures"

        meshes_dir.mkdir(parents=True, exist_ok=True)
        textures_dir.mkdir(parents=True, exist_ok=True)

        # Copy meshes
        copied_meshes = set()
        for asset in self.meshes:
            dest = package_dir / asset.package_path
            if asset.source_path not in copied_meshes:
                shutil.copy2(asset.source_path, dest)
                copied_meshes.add(asset.source_path)

        # Copy textures
        copied_textures = set()
        for asset in self.textures:
            dest = package_dir / asset.package_path
            if asset.source_path not in copied_textures:
                shutil.copy2(asset.source_path, dest)
                copied_textures.add(asset.source_path)

        # Print summary
        print(f"  📦 Copied {len(copied_meshes)} mesh file(s)")
        print(f"  📦 Copied {len(copied_textures)} texture file(s)")

    def rewrite_xml_paths(self) -> str:
        """SELF-RENDERING: Rewrite XML with package-relative paths

        Converts:
            <mesh file="/abs/path/to/mesh.obj"/>
        To:
            <mesh file="assets/meshes/mesh.obj"/>

        Returns:
            XML string with relative paths
        """
        # Create mapping: absolute path → package relative path
        path_map = {}
        for asset in self.meshes:
            path_map[str(asset.source_path)] = asset.package_path
        for asset in self.textures:
            path_map[str(asset.source_path)] = asset.package_path

        # Rewrite asset section
        asset_section = self.xml_root.find('asset')
        if asset_section is not None:
            # Rewrite meshes
            for mesh in asset_section.findall('mesh'):
                file_path = mesh.get('file')
                if file_path and file_path in path_map:
                    mesh.set('file', path_map[file_path])

            # Rewrite textures
            for texture in asset_section.findall('texture'):
                file_path = texture.get('file')
                if file_path and file_path in path_map:
                    texture.set('file', path_map[file_path])

        # Convert to string with pretty formatting
        ET.indent(self.xml_root, space='  ')
        xml_str = ET.tostring(self.xml_root, encoding='unicode')

        # Add XML declaration
        return f'<?xml version="1.0" encoding="utf-8"?>\n{xml_str}'

    def create_manifest(self) -> Dict:
        """SELF-GENERATION: Create manifest.json - PURE MOP!

        Returns:
            Dict with asset metadata
        """
        return {
            "assets": {
                "meshes": [
                    {
                        "name": asset.name,
                        "source": str(asset.source_path),
                        "package_path": asset.package_path,
                        "size": asset.size,
                        "checksum": asset.checksum
                    }
                    for asset in self.meshes
                ],
                "textures": [
                    {
                        "name": asset.name,
                        "source": str(asset.source_path),
                        "package_path": asset.package_path,
                        "size": asset.size,
                        "checksum": asset.checksum
                    }
                    for asset in self.textures
                ]
            },
            "created_at": datetime.now().isoformat(),
            "source_experiment": self.experiment_id,
            "total_files": len(self.meshes) + len(self.textures),
            "total_size_bytes": sum(a.size for a in self.meshes) + sum(a.size for a in self.textures)
        }

    def create_package_info(self) -> Dict:
        """SELF-GENERATION: Create package_info.json - PURE MOP!"""
        return {
            "package_format": "mujoco_portable_v1",
            "experiment_id": self.experiment_id,
            "created_at": datetime.now().isoformat(),
            "scene_xml": "scene.xml",
            "assets_dir": "assets/",
            "compatible_viewers": [
                "mujoco_wasm",
                "mjc_viewer",
                "mujoco_python"
            ],
            "usage": {
                "python": "model = mujoco.MjModel.from_xml_path('mujoco_package/scene.xml')",
                "javascript": "mujoco.FS.writeFile('/working/scene.xml', scene_xml_content); model = new mujoco.Model('/working/scene.xml')"
            }
        }

    def save(self, package_dir: Path):
        """SELF-SAVING: Save complete package to disk - PURE MOP!

        Creates:
            package_dir/
                scene.xml              (relative paths)
                assets/
                    meshes/*.obj
                    textures/*.png
                manifest.json          (asset metadata)
                package_info.json      (usage info)

        Args:
            package_dir: Directory to save package (e.g., database/{exp_id}/mujoco_package/)
        """
        # Create package directory
        package_dir.mkdir(parents=True, exist_ok=True)

        print(f"  📦 Creating MuJoCo package: {package_dir.name}/")

        # 1. Copy all assets
        self.copy_assets(package_dir)

        # 2. Save scene.xml with relative paths
        scene_xml = self.rewrite_xml_paths()
        scene_xml_path = package_dir / "scene.xml"
        with open(scene_xml_path, 'w') as f:
            f.write(scene_xml)
        print(f"  📦 Saved scene.xml ({len(scene_xml)} bytes)")

        # 3. Save manifest.json
        manifest = self.create_manifest()
        manifest_path = package_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"  📦 Saved manifest.json")

        # 4. Save package_info.json
        package_info = self.create_package_info()
        info_path = package_dir / "package_info.json"
        with open(info_path, 'w') as f:
            json.dump(package_info, f, indent=2)
        print(f"  📦 Saved package_info.json")

        # Summary
        total_size_mb = manifest["total_size_bytes"] / (1024 * 1024)
        print(f"  ✓ Package complete: {manifest['total_files']} files, {total_size_mb:.2f} MB")

    @staticmethod
    def load(package_dir: Path) -> 'AssetPackageModal':
        """Load package from disk (future use)

        Args:
            package_dir: Path to mujoco_package/

        Returns:
            AssetPackageModal instance
        """
        # Load scene.xml
        scene_xml_path = package_dir / "scene.xml"
        if not scene_xml_path.exists():
            raise FileNotFoundError(f"scene.xml not found in {package_dir}")

        with open(scene_xml_path, 'r') as f:
            xml_content = f.read()

        # Load package_info.json
        info_path = package_dir / "package_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                package_info = json.load(f)
                experiment_id = package_info.get('experiment_id', 'unknown')
        else:
            experiment_id = 'unknown'

        # Create instance (will re-discover assets from XML)
        package = AssetPackageModal(xml_content, experiment_id)

        return package