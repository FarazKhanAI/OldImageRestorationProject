"""
Model Manager - Singleton for managing all restoration models.
Provides unified interface to both ZeroScratches and BPBTL models.
"""
import logging
from typing import Dict, Any, Optional

from .zeroscratches_wrapper import ZeroScratchesWrapper
from .bptbl_wrapper import BPBTLWrapper

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
            self.bptbl = None
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
    



    # In your model_manager.py's initialize_bptbl method:
    def initialize_bptbl(self, bptbl_script_path: Optional[str] = None) -> bool:
        """Initialize BPBTL model."""
        try:
            logger.info("Initializing BPBTL model...")
            
            # Use the new wrapper
            self.bptbl = BPBTLWrapper(
                bptbl_root="checkpoints/bptbl",  # Path to your BPBTL
                gpu_id=-1,  # Use CPU (-1) or GPU (0, 1, etc.)
                with_scratch=True
            )
            
            success = self.bptbl.initialize()
            
            if success:
                logger.info("✅ BPBTL wrapper ready for production")
            else:
                logger.warning("⚠️ BPBTL initialization issues - check installation")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing BPBTL: {str(e)}")
            return False




    def initialize_all(self) -> Dict[str, bool]:
        """Initialize all available models."""
        results = {}
        
        # Initialize ZeroScratches
        results['zeroscratches'] = self.initialize_zeroscratches()
        
        # Initialize BPBTL
        results['bptbl'] = self.initialize_bptbl()
        
        logger.info(f"Model initialization results: {results}")
        return results
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about all available models."""
        models_info = {}
        
        if self.zeroscratches and self.zeroscratches.is_initialized():
            models_info['zeroscratches'] = self.zeroscratches.get_model_info()
        
        if self.bptbl:
            models_info['bptbl'] = self.bptbl.get_model_info()
        
        return models_info
    
    def recommend_model(self, image_path: str = None) -> str:
        """
        Recommend the best model based on scenario.
        This is a simple heuristic - you can make it more sophisticated.
        """
        available = self.get_available_models()
        
        if not available:
            return "none"
        
        # Simple recommendation logic
        if image_path:
            # You could add image analysis here
            # For now, default to zeroscratches if available
            if 'zeroscratches' in available:
                return 'zeroscratches'
            elif 'bptbl' in available:
                return 'bptbl'
        else:
            # Generic recommendation
            if 'zeroscratches' in available:
                return 'zeroscratches'
            elif 'bptbl' in available:
                return 'bptbl'
        
        return "none"
    
    def cleanup(self) -> None:
        """Clean up all model resources."""
        if self.zeroscratches:
            self.zeroscratches.cleanup()
        
        if self.bptbl:
            self.bptbl.cleanup()
        
        logger.info("All model resources cleaned up")