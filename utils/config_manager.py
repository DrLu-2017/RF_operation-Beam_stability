"""
Configuration Manager for ALBuMS
Handles saving, loading, and managing parameter configurations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class ConfigManager:
    """Manages configuration files for different accelerators."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory to store configuration files. 
                       If None, uses ~/.albums/configs/
        """
        if config_dir is None:
            config_dir = Path.home() / ".albums" / "configs"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Session state file to track last used config
        self.session_file = self.config_dir.parent / "session.json"
    
    def save_config(self, 
                    config_name: str, 
                    accelerator_name: str,
                    config_data: Dict[str, Any],
                    source_config: Optional[str] = None) -> str:
        """
        Save a configuration with metadata.
        
        Args:
            config_name: Name of the configuration file (without extension)
            accelerator_name: Name of the accelerator (e.g., "Aladdin", "SOLEIL II")
            config_data: Configuration data dictionary
            source_config: Name of the source config if this is a modification
        
        Returns:
            Path to saved configuration file
        """
        # Create a structured config with metadata
        full_config = {
            "metadata": {
                "name": config_name,
                "accelerator": accelerator_name,
                "created_at": datetime.now().isoformat(),
                "source_config": source_config,  # Track modifications
            },
            "config": config_data
        }
        
        config_file = self.config_dir / f"{accelerator_name}_{config_name}.json"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False)
        
        return str(config_file)
    
    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a configuration by name.
        
        Args:
            config_name: Configuration name (e.g., "Aladdin_passive" or full filename)
        
        Returns:
            Configuration data or None if not found
        """
        # Try exact filename first (with .json extension)
        config_file = self.config_dir / f"{config_name}.json"
        if not config_file.exists():
            # Try pattern matching without extension
            for f in self.config_dir.glob(f"*{config_name}.json"):
                config_file = f
                break
        
        if not config_file.exists():
            # Try exact match
            config_file = self.config_dir / config_name
        
        if not config_file.exists():
            return None
        
        with open(config_file, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
        
        return full_config.get("config", full_config)
    
    def load_config_with_metadata(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a configuration with its metadata.
        
        Args:
            config_name: Configuration name
        
        Returns:
            Full configuration data including metadata or None if not found
        """
        # Try exact filename first (with .json extension)
        config_file = self.config_dir / f"{config_name}.json"
        if not config_file.exists():
            # Try pattern matching without extension
            for f in self.config_dir.glob(f"*{config_name}.json"):
                config_file = f
                break
        
        if not config_file.exists():
            # Try exact match
            config_file = self.config_dir / config_name
        
        if not config_file.exists():
            return None
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_accelerator_configs(self, accelerator_name: str) -> List[str]:
        """
        Get all saved configurations for an accelerator.
        
        Args:
            accelerator_name: Name of the accelerator
        
        Returns:
            List of configuration names for this accelerator
        """
        configs = []
        pattern = f"{accelerator_name}_*.json"
        for config_file in self.config_dir.glob(pattern):
            # Extract config name (without accelerator prefix and extension)
            filename = config_file.name
            config_name = filename[len(accelerator_name)+1:-5]  # Remove prefix and .json
            configs.append(config_name)
        
        return sorted(configs)
    
    def get_all_accelerators(self) -> List[str]:
        """
        Get list of all accelerators that have saved configs.
        
        Returns:
            List of unique accelerator names
        """
        accelerators = set()
        for config_file in self.config_dir.glob("*.json"):
            filename = config_file.name
            # Parse accelerator name from filename
            parts = filename[:-5].split("_", 1)  # Remove .json and split on first _
            if len(parts) >= 1:
                accelerators.add(parts[0])
        
        return sorted(list(accelerators))
    
    def delete_config(self, config_name: str) -> bool:
        """
        Delete a configuration file.
        
        Args:
            config_name: Configuration name
        
        Returns:
            True if deleted successfully
        """
        config_file = self.config_dir / f"{config_name}.json"
        if not config_file.exists():
            for f in self.config_dir.glob(f"*{config_name}.json"):
                config_file = f
                break
        
        if config_file.exists():
            config_file.unlink()
            return True
        
        return False
    
    def rename_config(self, old_name: str, new_name: str) -> bool:
        """
        Rename a configuration file.
        
        Args:
            old_name: Current configuration name
            new_name: New configuration name
        
        Returns:
            True if renamed successfully
        """
        old_file = self.config_dir / f"{old_name}.json"
        if not old_file.exists():
            for f in self.config_dir.glob(f"*{old_name}.json"):
                old_file = f
                break
        
        if old_file.exists():
            new_file = self.config_dir / f"{new_name}.json"
            old_file.rename(new_file)
            return True
        
        return False
    
    def save_session_config(self, config_name: str, accelerator_name: str) -> None:
        """
        Save the currently used configuration to session state.
        
        Args:
            config_name: Name of the current configuration
            accelerator_name: Name of the current accelerator
        """
        session_data = {
            "last_config": config_name,
            "last_accelerator": accelerator_name,
            "last_used": datetime.now().isoformat()
        }
        
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
    
    def load_session_config(self) -> Optional[Dict[str, str]]:
        """
        Load the last used configuration from session state.
        
        Returns:
            Dictionary with last_config and last_accelerator, or None
        """
        if not self.session_file.exists():
            return None
        
        try:
            with open(self.session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def export_config(self, config_name: str, export_path: str) -> bool:
        """
        Export a configuration to an external file.
        
        Args:
            config_name: Configuration name
            export_path: Path to export to
        
        Returns:
            True if exported successfully
        """
        config_data = self.load_config_with_metadata(config_name)
        if config_data is None:
            return False
        
        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        return True
    
    def import_config(self, import_path: str) -> Optional[str]:
        """
        Import a configuration from an external file.
        
        Args:
            import_path: Path to import from
        
        Returns:
            Name of the imported configuration or None if failed
        """
        import_file = Path(import_path)
        if not import_file.exists():
            return None
        
        try:
            with open(import_file, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
            
            metadata = full_config.get("metadata", {})
            config_name = metadata.get("name", import_file.stem)
            accelerator_name = metadata.get("accelerator", "Custom")
            config_data = full_config.get("config", full_config)
            
            self.save_config(config_name, accelerator_name, config_data, 
                           metadata.get("source_config"))
            
            return f"{accelerator_name}_{config_name}"
        except Exception:
            return None

