"""
ZeroScratches Model Wrapper
Adapter for the ZeroScratches model to handle both single images and batch processing.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from zeroscratches import EraseScratches

from .base_restorer import BaseRestorer

# Configure logging
logger = logging.getLogger(__name__)


class ZeroScratchesWrapper(BaseRestorer):
    """
    Wrapper for ZeroScratches model that adapts single-image model to batch processing.
    
    This adapter handles:
    1. Single image processing (native to ZeroScratches)
    2. Batch processing via parallel execution
    3. Format conversions between PIL, numpy, and file paths
    4. Progress tracking for batch operations
    """
    
    def __init__(self, model_cache_dir: Optional[str] = None, max_workers: int = 4):
        """
        Initialize the ZeroScratches model wrapper.
        
        Args:
            model_cache_dir: Directory to cache model weights. If None, uses default.
            max_workers: Maximum number of parallel workers for batch processing.
        """
        super().__init__(model_name="zeroscratches")
        
        self.model_cache_dir = model_cache_dir
        self.max_workers = max_workers
        self._model = None
        self._is_initialized = False
        
        logger.info(f"ZeroScratchesWrapper initialized with max_workers={max_workers}")
    
    def _initialize_model(self) -> None:
        """Lazy initialization of the ZeroScratches model."""
        if not self._is_initialized:
            try:
                logger.info("Initializing ZeroScratches model...")
                
                # The model will auto-download weights on first use
                self._model = EraseScratches()
                
                # Test with a tiny dummy image to trigger download and verify functionality
                dummy_img = Image.new('RGB', (10, 10), color='white')
                test_result = self._model.erase(dummy_img)
                
                if isinstance(test_result, np.ndarray):
                    logger.info(f"ZeroScratches model initialized successfully. Test output shape: {test_result.shape}")
                    self._is_initialized = True
                else:
                    logger.error("ZeroScratches model test failed - unexpected output type")
                    raise RuntimeError("ZeroScratches model initialization test failed")
                    
            except Exception as e:
                logger.error(f"Failed to initialize ZeroScratches model: {str(e)}")
                raise
    
    def process_single(self, input_data: Union[str, Path, Image.Image, np.ndarray], 
                      **kwargs) -> Union[Image.Image, np.ndarray]:
        """
        Process a single image using ZeroScratches model.
        
        Args:
            input_data: Can be:
                - str/Path: Path to image file
                - PIL.Image: PIL Image object
                - np.ndarray: Numpy array of image
            **kwargs: Additional parameters (not used by ZeroScratches but kept for interface)
            
        Returns:
            Restored image as PIL Image or numpy array (same type as input)
            
        Raises:
            ValueError: If input type is not supported
            RuntimeError: If model processing fails
        """
        # Ensure model is initialized
        if not self._is_initialized:
            self._initialize_model()
        
        # Track input type to return same type
        input_type = type(input_data)
        is_numpy_return = isinstance(input_data, np.ndarray)
        
        try:
            # Convert input to PIL Image for processing
            if isinstance(input_data, (str, Path)):
                # Load from file path
                img = Image.open(input_data)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                original_path = input_data
                
            elif isinstance(input_data, Image.Image):
                # Use PIL Image directly
                img = input_data.copy()
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                original_path = None
                
            elif isinstance(input_data, np.ndarray):
                # Convert numpy array to PIL
                if input_data.dtype != np.uint8:
                    input_data = (input_data * 255).astype(np.uint8)
                img = Image.fromarray(input_data)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                original_path = None
                
            else:
                raise ValueError(f"Unsupported input type: {input_type}. "
                               f"Expected str, Path, PIL.Image, or np.ndarray.")
            
            logger.debug(f"Processing single image: {original_path or 'in-memory'}, "
                        f"size: {img.size}, mode: {img.mode}")
            
            # Process with ZeroScratches
            result_array = self._model.erase(img)
            
            # Convert back to desired output format
            if is_numpy_return:
                return result_array
            else:
                # Convert numpy array back to PIL Image
                if result_array.dtype != np.uint8:
                    result_array = (result_array * 255).astype(np.uint8)
                return Image.fromarray(result_array)
                
        except Exception as e:
            logger.error(f"Error processing single image: {str(e)}")
            raise RuntimeError(f"ZeroScratches processing failed: {str(e)}")
    
    def process_batch(self, input_list: List[Union[str, Path, Image.Image, np.ndarray]], 
                     progress_callback: Optional[callable] = None,
                     **kwargs) -> List[Union[Image.Image, np.ndarray]]:
        """
        Process multiple images in parallel using ThreadPoolExecutor.
        
        Args:
            input_list: List of inputs (same types as process_single)
            progress_callback: Optional callback function for progress updates.
                               Called with (current, total, filename)
            **kwargs: Additional parameters passed to process_single
            
        Returns:
            List of restored images in same order and format as input
            
        Raises:
            ValueError: If input_list is empty
        """
        if not input_list:
            raise ValueError("Input list cannot be empty")
        
        total_items = len(input_list)
        logger.info(f"Starting batch processing of {total_items} items with {self.max_workers} workers")
        
        results = [None] * total_items
        completed_count = 0
        
        # Prepare items for processing (add index for ordering)
        indexed_items = [(idx, item) for idx, item in enumerate(input_list)]
        
        # Determine input type for consistent return format
        # Use first item as reference
        first_item = input_list[0]
        return_numpy = isinstance(first_item, np.ndarray)
        
        def process_item(index, item):
            """Wrapper function for processing a single item with error handling."""
            try:
                result = self.process_single(item, **kwargs)
                
                # Convert to consistent format if needed
                if return_numpy and isinstance(result, Image.Image):
                    result = np.array(result)
                elif not return_numpy and isinstance(result, np.ndarray):
                    result = Image.fromarray(result.astype(np.uint8))
                    
                return index, result, None
                
            except Exception as e:
                logger.error(f"Error processing item {index}: {str(e)}")
                return index, None, str(e)
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_item, idx, item): idx 
                for idx, item in indexed_items
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                
                try:
                    index, result, error = future.result()
                    
                    if error:
                        # Create error placeholder
                        if return_numpy:
                            # Create black image array as error placeholder
                            if isinstance(input_list[index], np.ndarray):
                                error_result = np.zeros_like(input_list[index])
                            else:
                                # Default to 256x256 black image
                                error_result = np.zeros((256, 256, 3), dtype=np.uint8)
                        else:
                            # Create black PIL image as error placeholder
                            error_result = Image.new('RGB', (256, 256), color='black')
                        
                        results[index] = error_result
                        logger.warning(f"Item {index} failed: {error}. Using placeholder.")
                    else:
                        results[index] = result
                    
                except Exception as e:
                    logger.error(f"Unexpected error collecting result for item {idx}: {str(e)}")
                    # Create placeholder for this item too
                    if return_numpy:
                        results[idx] = np.zeros((256, 256, 3), dtype=np.uint8)
                    else:
                        results[idx] = Image.new('RGB', (256, 256), color='black')
                
                # Update progress
                completed_count += 1
                if progress_callback:
                    # Try to get filename for progress reporting
                    filename = None
                    item = input_list[idx]
                    if isinstance(item, (str, Path)):
                        filename = Path(item).name
                    
                    progress_callback(completed_count, total_items, filename)
        
        logger.info(f"Batch processing completed: {completed_count}/{total_items} successful")
        return results
    
    def process_folder(self, input_folder: Union[str, Path], 
                      output_folder: Optional[Union[str, Path]] = None,
                      extensions: List[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Process all images in a folder and save to output folder.
        
        Args:
            input_folder: Path to folder containing images
            output_folder: Path to save processed images. If None, creates a subfolder.
            extensions: List of file extensions to process (default: ['.jpg', '.jpeg', '.png'])
            **kwargs: Additional parameters for processing
            
        Returns:
            Dictionary with processing statistics and output paths
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        input_path = Path(input_folder)
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError(f"Input folder does not exist or is not a directory: {input_folder}")
        
        # Create output folder
        if output_folder is None:
            output_folder = input_path.parent / f"{input_path.name}_restored"
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No image files found in {input_folder} with extensions {extensions}")
            return {
                'success': False,
                'message': f"No image files found in {input_folder}",
                'processed': 0,
                'total': 0,
                'output_folder': str(output_path)
            }
        
        logger.info(f"Found {len(image_files)} images in {input_folder}")
        
        # Process all images
        results = self.process_batch(
            [str(f) for f in image_files],
            progress_callback=kwargs.get('progress_callback'),
            **kwargs
        )
        
        # Save results
        saved_files = []
        for img_file, result in zip(image_files, results):
            if result is not None:
                # Generate output filename
                output_filename = f"{img_file.stem}_restored{img_file.suffix}"
                output_file = output_path / output_filename
                
                # Save based on result type
                if isinstance(result, Image.Image):
                    result.save(output_file)
                elif isinstance(result, np.ndarray):
                    # Convert numpy to PIL for saving
                    if result.dtype != np.uint8:
                        result = (result * 255).astype(np.uint8)
                    Image.fromarray(result).save(output_file)
                
                saved_files.append(str(output_file))
                logger.debug(f"Saved result to {output_file}")
        
        # Return statistics
        return {
            'success': True,
            'message': f"Processed {len(saved_files)} of {len(image_files)} images",
            'processed': len(saved_files),
            'total': len(image_files),
            'output_folder': str(output_path),
            'output_files': saved_files
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'name': 'ZeroScratches',
            'description': 'Lightweight model for removing scratches and minor damages',
            'version': '0.1.0',  # This should match your zeroscratches version
            'supports_batch': True,
            'max_batch_size': self.max_workers,
            'is_initialized': self._is_initialized,
            'best_for': ['Minor scratches', 'Dust spots', 'Small tears', 'Single images'],
            'estimated_time_per_image': '2-10 seconds (depends on size)'
        }
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        self._model = None
        self._is_initialized = False
        logger.info("ZeroScratches model resources cleaned up")


# Convenience function for quick testing
def test_zeroscratches():
    """Quick test function to verify ZeroScratches is working."""
    try:
        wrapper = ZeroScratchesWrapper(max_workers=2)
        
        # Create a test image
        test_img = Image.new('RGB', (256, 256), color='gray')
        
        # Test single image processing
        print("Testing single image processing...")
        result = wrapper.process_single(test_img)
        print(f"✓ Single image processing successful. Result type: {type(result)}")
        
        # Test batch processing
        print("\nTesting batch processing...")
        batch_results = wrapper.process_batch([test_img, test_img, test_img])
        print(f"✓ Batch processing successful. Processed {len(batch_results)} images")
        
        # Get model info
        info = wrapper.get_model_info()
        print(f"\nModel Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        wrapper.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    print("Testing ZeroScratches wrapper...")
    success = test_zeroscratches()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")