"""
BPBTL (Bringing-Old-Photos-Back-to-Life) Model Wrapper - FIXED VERSION
Fixed path issue and Windows Unicode encoding.
"""
import os
import sys
import subprocess
import tempfile
import shutil
import logging
import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import threading

import numpy as np
from PIL import Image

from .base_restorer import BaseRestorer

logger = logging.getLogger(__name__)

class BPBTLWrapper(BaseRestorer):
    """
    BPBTL wrapper that calls the working run.py script via subprocess.
    """
    
    def __init__(
        self,
        bptbl_root: Optional[str] = None,
        gpu_id: int = -1,
        with_scratch: bool = True,
        input_size: str = "full_size"
    ):
        """
        Initialize BPBTL wrapper.
        """
        super().__init__(model_name="bptbl")
        
        # Set paths - FIXED: Use absolute paths
        if bptbl_root:
            self.bptbl_root = Path(bptbl_root).absolute()
        else:
            # Default: assume BPBTL is in checkpoints/bptbl/
            self.bptbl_root = (Path.cwd() / "checkpoints" / "bptbl").absolute()
        
        # Configuration
        self.gpu_id = gpu_id
        self.with_scratch = with_scratch
        self.input_size = input_size
        
        logger.info(f"BPBTLWrapper initialized with root: {self.bptbl_root}")
    
    def initialize(self) -> bool:
        """
        Verify BPBTL is properly installed and accessible.
        """
        try:
            logger.info("Verifying BPBTL installation...")
            
            # Check if BPBTL root exists
            if not self.bptbl_root.exists():
                logger.error(f"BPBTL root not found: {self.bptbl_root}")
                return False
            
            # Check if run.py exists
            run_script = self.bptbl_root / "run.py"
            if not run_script.exists():
                logger.error(f"run.py not found at: {run_script}")
                return False
            
            self._is_initialized = True
            logger.info("✅ BPBTL wrapper initialized successfully")
            logger.info(f"  - Main script: run.py ✓")
            logger.info(f"  - GPU: {'CPU' if self.gpu_id == -1 else f'GPU:{self.gpu_id}'}")
            logger.info(f"  - Scratch detection: {'ENABLED' if self.with_scratch else 'DISABLED'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BPBTL: {str(e)}")
            return False
    
    def _run_bptbl_safely(self, input_dir: Path, output_dir: Path) -> bool:
        """
        Run BPBTL run.py with proper Windows encoding handling.
        """
        try:
            # Build command
            cmd = [
                sys.executable,
                "run.py",  # Just use filename since we'll set cwd to BPBTL root
                '--input_folder', str(input_dir),
                '--output_folder', str(output_dir),
                '--GPU', str(self.gpu_id)
            ]
            
            if self.with_scratch:
                cmd.append('--with_scratch')
            
            logger.info(f"Running BPBTL from: {self.bptbl_root}")
            
            # Set environment to handle Unicode
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            # Run subprocess
            result = subprocess.run(
                cmd,
                cwd=str(self.bptbl_root),  # CRITICAL: Run from BPBTL directory
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600,  # 10 minutes
                env=env
            )
            
            # Check if successful
            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                logger.error(f"BPBTL failed with code {result.returncode}: {error_msg}")
                return False
            
            # Check for output
            output_locations = [
                output_dir / "restored_image",
                output_dir
            ]
            
            for location in output_locations:
                if location.exists():
                    output_files = list(location.glob("*.*"))
                    if output_files:
                        logger.info(f"Found {len(output_files)} output file(s) in {location}")
                        return True
            
            logger.error("BPBTL ran but produced no output files")
            return False
            
        except subprocess.TimeoutExpired:
            logger.error("BPBTL processing timed out after 10 minutes")
            return False
        except Exception as e:
            logger.error(f"Error running BPBTL: {str(e)}")
            return False
    
    def _process_single_image(self, input_path: Path, output_path: Path) -> bool:
        """
        Process a single image using BPBTL's run.py.
        """
        try:
            logger.info(f"Processing image: {input_path.name}")
            
            # Create temporary workspace
            with tempfile.TemporaryDirectory(prefix="bptbl_") as tmpdir:
                tmpdir = Path(tmpdir)
                input_dir = tmpdir / "input"
                output_dir = tmpdir / "output"
                input_dir.mkdir(parents=True, exist_ok=True)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy input image
                shutil.copy2(input_path, input_dir / input_path.name)
                
                # Run BPBTL
                success = self._run_bptbl_safely(input_dir, output_dir)
                
                if not success:
                    logger.error(f"BPBTL processing failed for {input_path.name}")
                    return False
                
                # Find and copy output
                output_locations = [
                    output_dir / "restored_image",
                    output_dir
                ]
                
                output_found = False
                for location in output_locations:
                    if location.exists():
                        # Look for any image file
                        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                            for file in location.glob(f"*{ext}"):
                                # Copy to final output
                                output_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(file, output_path)
                                logger.info(f"✅ Saved output to: {output_path}")
                                output_found = True
                                break
                        if output_found:
                            break
                
                if not output_found:
                    logger.error(f"No output image found for {input_path.name}")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return False
    



    def process_single(
    self,
    input_data: Union[str, Path, Image.Image, np.ndarray],
    **kwargs) -> Union[Image.Image, None]:
        """
        Process a single image with BPBTL - FIXED VERSION.
        """
        if not self._is_initialized and not self.initialize():
            logger.error("BPBTL not initialized")
            return None
        
        try:
            # Convert input to file path
            input_path = None
            
            if isinstance(input_data, (str, Path)):
                input_path = Path(input_data).absolute()
            elif isinstance(input_data, Image.Image):
                # Save PIL Image to a temporary file
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                input_data.save(temp_file.name)
                input_path = Path(temp_file.name)
            elif isinstance(input_data, np.ndarray):
                # Save numpy array to a temporary file
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                Image.fromarray(input_data).save(temp_file.name)
                input_path = Path(temp_file.name)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            if not input_path or not input_path.exists():
                logger.error("Input file not created")
                return None
            
            # Create a permanent output path (not in temp directory)
            if 'output_path' in kwargs:
                output_path = Path(kwargs['output_path']).absolute()
            else:
                # Create output in current directory
                timestamp = int(time.time())
                output_path = Path.cwd() / f"bptbl_output_{timestamp}.jpg"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process the image
            logger.info(f"Processing: {input_path.name}")
            success = self._process_single_image(input_path, output_path)
            
            if success and output_path.exists():
                # Load and return the result
                result_img = Image.open(output_path)
                logger.info(f"✅ Successfully processed: {input_path.name}")
                
                # Clean up temporary file if we created one
                if isinstance(input_data, (Image.Image, np.ndarray)):
                    try:
                        input_path.unlink()
                    except:
                        pass
                
                return result_img
            else:
                logger.error(f"Failed to process: {input_path.name}")
                return None
                
        except Exception as e:
            logger.error(f"Error in process_single: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return None



    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the BPBTL model."""
        return {
            'name': 'BPBTL (Bringing-Old-Photos-Back-to-Life)',
            'description': 'Advanced pipeline for restoring severely damaged old photos with scratch detection and quality enhancement',
            'version': 'Official Implementation',
            'supports_batch': True,
            'is_initialized': self._is_initialized,
            'gpu_id': self.gpu_id,
            'with_scratch': self.with_scratch,
            'input_size': self.input_size,
            'best_for': [
                'Severe scratches and tears',
                'Faded and discolored photos',
                'Major structural damage',
                'Very old and degraded photos'
            ],
            'estimated_time': '30-120 seconds per image',
            'paths': {
                'bptbl_root': str(self.bptbl_root),
                'run_script': str(self.bptbl_root / "run.py")
            }
        }
    


    def process_batch(
    self,
    input_list: List[Union[str, Path, Image.Image, np.ndarray]],
    progress_callback: Optional[callable] = None,
    **kwargs
    ) -> List[Union[Image.Image, None]]:
        """
        Process multiple images in batch. (Required Abstract Method)
        This implementation processes images sequentially.
        """
        if not self._is_initialized and not self.initialize():
            logger.error("BPBTL not initialized")
            return [None] * len(input_list)

        results = []
        for i, input_item in enumerate(input_list):
            if progress_callback:
                progress_callback(i, len(input_list), f"Processing image {i+1}")

            try:
                # Process each image individually using the single method
                result = self.process_single(input_item)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process item {i}: {str(e)}")
                results.append(None)

        if progress_callback:
            progress_callback(len(input_list), len(input_list), "Batch complete")

        logger.info(f"Batch processed: {len([r for r in results if r])}/{len(results)} successful")
        return results
    

    def process_folder(
    self,
    input_folder: Union[str, Path],
    output_folder: Optional[Union[str, Path]] = None,
    **kwargs) -> Dict[str, Any]:
        """
        Process all images in a folder. (Required Abstract Method)
        """
        input_path = Path(input_folder)

        if not input_path.exists():
            return {
                'success': False,
                'message': f"Input folder not found: {input_folder}",
                'processed': 0,
                'total': 0
            }

        # Collect all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        if not image_files:
            return {
                'success': False,
                'message': f"No images found in {input_folder}",
                'processed': 0,
                'total': 0
            }

        # Create output folder
        if output_folder is None:
            output_folder = input_path.parent / f"{input_path.name}_restored"
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process images using the batch method
        results = self.process_batch(image_files)

        # Count successes
        success_count = len([r for r in results if r is not None])

        return {
            'success': success_count > 0,
            'message': f"Processed {success_count} of {len(image_files)} images",
            'processed': success_count,
            'total': len(image_files),
            'input_folder': str(input_path),
            'output_folder': str(output_path)
        }



    def set_gpu(self, gpu_id: int) -> None:
        """Set GPU ID for processing."""
        self.gpu_id = gpu_id
        logger.info(f"GPU set to: {gpu_id} {'(CPU)' if gpu_id == -1 else f'(GPU:{gpu_id})'}")
    
    def set_scratch_detection(self, enabled: bool) -> None:
        """Enable or disable scratch detection."""
        self.with_scratch = enabled
        logger.info(f"Scratch detection: {'ENABLED' if enabled else 'DISABLED'}")

# Quick test function
def test_bptbl_wrapper():
    """Test the BPBTL wrapper."""
    wrapper = BPBTLWrapper(
        bptbl_root='checkpoints/bptbl',
        gpu_id=-1,
        with_scratch=False
    )
    
    if wrapper.initialize():
        print("✅ BPBTL initialized successfully")
        
        # Test with an image
        test_image = Path("tests/test_Old_images/a.png")
        if test_image.exists():
            print(f"Testing with image: {test_image.name}")
            
            try:
                result = wrapper.process_single(test_image, "test_bptbl_final.jpg")
                if result is not None:
                    print(f"✅ BPBTL worked! Output size: {result.size}")
                    print(f"   Output saved to: test_bptbl_final.jpg")
                    return True
                else:
                    print("❌ BPBTL failed to process image")
                    return False
            except Exception as e:
                print(f"❌ Error: {e}")
                return False
        else:
            print(f"❌ Test image not found: {test_image}")
            return False
    else:
        print("❌ Failed to initialize BPBTL")
        return False

if __name__ == "__main__":
    test_bptbl_wrapper()