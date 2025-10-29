"""
Test that project setup is working correctly.
"""

import yaml
import sys
from pathlib import Path

def test_config():
    """Test configuration file loads correctly."""
    config_path = Path("config/config.yaml")
    
    if not config_path.exists():
        print("❌ Config file not found!")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Config file loads successfully!")
        print(f"   Project: {config['project']['name']}")
        print(f"   Version: {config['project']['version']}")
        return True
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_imports():
    """Test that main packages import correctly."""
    packages = ['pandas', 'numpy', 'sklearn', 'yaml', 'xgboost']
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError:
            print(f"❌ Failed to import {package}")
            return False
    return True

def test_directory_structure():
    """Test that all directories exist."""
    dirs = ['data/raw', 'data/processed', 'src/data', 'notebooks', 'models']
    
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            return False
    return True

if __name__ == "__main__":
    print("🔍 Testing Project Setup...\n")
    
    all_good = True
    all_good &= test_directory_structure()
    print()
    all_good &= test_config()
    print()
    all_good &= test_imports()
    
    print("\n" + "="*50)
    if all_good:
        print("✅ All tests passed! Your setup is complete.")
    else:
        print("❌ Some tests failed. Please check the errors above.")