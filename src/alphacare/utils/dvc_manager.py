"""
DVC Manager Module

This module provides the DVCManager class for managing Data Version Control.
"""

import subprocess
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DVCManager:
    """
    A class for managing DVC operations.
    
    This class provides methods to initialize DVC, add data files,
    and manage data versions.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the DVCManager.
        
        Args:
            project_root: Root directory of the project. Defaults to current directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        logger.info(f"DVCManager initialized with project root: {self.project_root}")
    
    def init_dvc(self) -> bool:
        """
        Initialize DVC in the project.
        
        Returns:
            bool: True if successful
        """
        try:
            result = subprocess.run(
                ["dvc", "init"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("DVC initialized successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize DVC: {e.stderr}")
            return False
    
    def add_remote(self, name: str, url: str, default: bool = True) -> bool:
        """
        Add a DVC remote.
        
        Args:
            name: Name of the remote
            url: URL or path to the remote storage
            default: Whether to set as default remote
            
        Returns:
            bool: True if successful
        """
        try:
            cmd = ["dvc", "remote", "add", name, url]
            if default:
                cmd.extend(["-d"])
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Remote '{name}' added successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add remote: {e.stderr}")
            return False
    
    def add_data(self, data_path: str) -> bool:
        """
        Add a data file to DVC tracking.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            bool: True if successful
        """
        try:
            result = subprocess.run(
                ["dvc", "add", data_path],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Data file '{data_path}' added to DVC")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add data: {e.stderr}")
            return False
    
    def push(self) -> bool:
        """
        Push data to remote storage.
        
        Returns:
            bool: True if successful
        """
        try:
            result = subprocess.run(
                ["dvc", "push"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Data pushed to remote successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push data: {e.stderr}")
            return False

