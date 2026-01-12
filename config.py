import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent

class Config:
    """Base configuration"""
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False') == 'True'
    
    # File Upload
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_FILE_SIZE_MB', 10)) * 1024 * 1024
    ALLOWED_EXTENSIONS = set(os.environ.get('ALLOWED_EXTENSIONS', 'png,jpg,jpeg,bmp,tiff,webp').split(','))
    
    # Paths
    UPLOAD_FOLDER = BASE_DIR / 'static/uploads'
    RESULTS_FOLDER = BASE_DIR / 'static/results'
    MODEL_CACHE_DIR = BASE_DIR / os.environ.get('MODEL_CACHE_DIR', 'models/cache')
    
    # Model Settings
    ZEROSCRATCHES_MAX_WORKERS = int(os.environ.get('ZEROSCRATCHES_MAX_WORKERS', 4))
    
    # Create directories
    @staticmethod
    def init_app(app):
        # Create necessary directories
        Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        Config.RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
        Config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Ensure secret key is set in production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    @staticmethod
    def init_app(app):
        Config.init_app(app)
        # Additional production initialization if needed


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': ProductionConfig  # Default to production for safety
}