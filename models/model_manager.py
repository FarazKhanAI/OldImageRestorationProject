"""
Model Manager - Singleton for managing all restoration models.
Provides unified interface to both ZeroScratches and BPBTL models.
"""
import logging
from typing import Dict, Any, Optional

from .zeroscratches_wrapper import ZeroScratchesWrapper
# Remove or comment out BPBTL import since we don't have it
# from .bptbl_wrapper import BPBTLWrapper

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages initialization and access to all restoration models.
    Implements singleton pattern to prevent multiple instances.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.zeroscratches = None
            # self.bptbl = None  # Comment out for now
            self._initialized = True
            logger.info("ModelManager initialized")
    
    def initialize_zeroscratches(self, max_workers: int = 4) -> bool:
        """Initialize ZeroScratches model."""
        try:
            logger.info("Initializing ZeroScratches model...")
            self.zeroscratches = ZeroScratchesWrapper(max_workers=max_workers)
            success = self.zeroscratches.initialize()
            
            if success:
                logger.info("ZeroScratches model ready")
            else:
                logger.error("Failed to initialize ZeroScratches")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing ZeroScratches: {str(e)}")
            return False
    
    # Comment out BPBTL initialization for now
    def initialize_bptbl(self, bptbl_script_path: Optional[str] = None) -> bool:
        """Initialize BPBTL model."""
        logger.warning("BPBTL model not available in current setup")
        return False
    
    def initialize_all(self) -> Dict[str, bool]:
        """Initialize all available models."""
        results = {}
        
        # Initialize ZeroScratches
        results['zeroscratches'] = self.initialize_zeroscratches()
        
        # BPBTL is disabled for now
        results['bptbl'] = False
        
        logger.info(f"Model initialization results: {results}")
        return results
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about all available models."""
        models_info = {}
        
        if self.zeroscratches and self.zeroscratches.is_initialized():
            models_info['zeroscratches'] = self.zeroscratches.get_model_info()
        
        # BPBTL is disabled for now
        # if self.bptbl:
        #     models_info['bptbl'] = self.bptbl.get_model_info()
        
        return models_info
    
    def recommend_model(self, image_path: str = None) -> str:
        """
        Recommend the best model based on scenario.
        This is a simple heuristic - you can make it more sophisticated.
        """
        available = self.get_available_models()
        
        if not available:
            return "none"
        
        # Always use zeroscratches if available
        if 'zeroscratches' in available:
            return 'zeroscratches'
        
        return "none"
    
    def cleanup(self) -> None:
        """Clean up all model resources."""
        if self.zeroscratches:
            self.zeroscratches.cleanup()
        
        # if self.bptbl:
        #     self.bptbl.cleanup()
        
        logger.info("All model resources cleaned up")

# Add this convenience function that was being imported
def restore_image(input_path, output_path, model_type='quick'):
    """Convenience function to restore a single image."""
    try:
        manager = ModelManager()
        if model_type == 'quick':
            if not manager.zeroscratches:
                manager.initialize_zeroscratches()
            
            from PIL import Image
            img = Image.open(input_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            result = manager.zeroscratches.process_single(img)
            result.save(output_path)
            return True
        else:
            logger.error("Only 'quick' model is currently available")
            return False
    except Exception as e:
        logger.error(f"Error in restore_image: {e}")
        return False