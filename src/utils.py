"""
Utilities Module
Helper functions and scripts
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
from loguru import logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    """Save configuration to YAML file"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise


def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    
    logger.remove()  # Remove default handler
    
    # Console handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_config.get('level', 'INFO')
    )
    
    # File handler
    log_file = log_config.get('file', 'logs/app.log')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_config.get('level', 'INFO'),
        rotation=f"{log_config.get('max_size_mb', 50)} MB",
        retention=log_config.get('backup_count', 5),
        compression="zip"
    )
    
    logger.info("Logging configured")


def save_violations_json(violations: dict, output_path: str):
    """Save violations to JSON file"""
    try:
        violations_list = [v.to_dict() for v in violations.values()]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(violations_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Violations saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save violations: {e}")
        raise


def create_directory_structure(base_path: Path):
    """Create necessary directory structure"""
    directories = [
        'models',
        'data/videos',
        'data/frames',
        'data/violations',
        'data/sessions',
        'logs'
    ]
    
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep for empty directories
        gitkeep = full_path / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()
    
    logger.info("Directory structure created")
