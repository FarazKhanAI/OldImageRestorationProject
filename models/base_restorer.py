"""
Base abstract class for all image restoration models.
Ensures consistent interface across different models.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseRestorer(ABC):
    """Abstract base class for image restoration models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._is_initialized = False
        logger.info(f"Initialized {model_name} restorer")
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the model (load weights, setup, etc.)."""
        pass
    
    @abstractmethod
    def process_single(
        self, 
        input_data: Union[str, Path, Image.Image, np.ndarray],
        **kwargs
    ) -> Union[Image.Image, np.ndarray, None]:
        """Process a single image."""
        pass
    
    @abstractmethod
    def process_batch(
        self,
        input_list: List[Union[str, Path, Image.Image, np.ndarray]],
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> List[Union[Image.Image, np.ndarray, None]]:
        """Process multiple images."""
        pass
    
    @abstractmethod
    def process_folder(
        self,
        input_folder: Union[str, Path],
        output_folder: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process all images in a folder."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and capabilities."""
        pass
    
    def is_initialized(self) -> bool:
        """Check if model is ready for processing."""
        return self._is_initialized
    
    def cleanup(self) -> None:
        """Release model resources."""
        self._is_initialized = False
        logger.info(f"{self.model_name} resources cleaned up")