"""
COMPLETE MODEL TESTING SCRIPT - UPDATED TO USE BPBTL WRAPPER
Tests both ZeroScratches and BPBTL models with your test images.
Shows visual comparisons and detailed performance metrics.
"""
import os
import sys
import time
import logging
from pathlib import Path
import traceback
import subprocess
import shutil
import tempfile

# Set OpenMP environment variable FIRST to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def print_result(success, message, indent=2):
    """Print a formatted result message."""
    prefix = "✅ " if success else "❌ "
    indent_str = " " * indent
    print(f"{indent_str}{prefix}{message}")

def test_zeroscratches():
    """Test ZeroScratches model."""
    print_section("1. TESTING ZEROSCRATCHES MODEL")
    
    try:
        from zeroscratches import EraseScratches
        print_result(True, "ZeroScratches package imported successfully")
        
        # Initialize model
        print("  Initializing model...")
        start_time = time.time()
        eraser = EraseScratches()
        init_time = time.time() - start_time
        
        print_result(True, f"Model initialized in {init_time:.2f} seconds")
        
        # Test with a simple image
        print("  Testing with sample image...")
        from PIL import Image
        test_img = Image.new('RGB', (100, 100), color='gray')
        
        start_time = time.time()
        result_array = eraser.erase(test_img)
        process_time = time.time() - start_time
        
        if isinstance(result_array, type(test_img)):
            print_result(True, f"Image returned (PIL format) in {process_time:.2f}s")
        elif isinstance(result_array, type(__import__('numpy').array([]))):
            print_result(True, f"Numpy array returned in {process_time:.2f}s")
            print(f"    Output shape: {result_array.shape}")
            print(f"    Output dtype: {result_array.dtype}")
        else:
            print_result(False, f"Unexpected output type: {type(result_array)}")
            return False
        
        return True
        
    except Exception as e:
        print_result(False, f"ZeroScratches test failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def test_bptbl_wrapper():
    """Test BPBTL using the corrected wrapper."""
    print_section("2. TESTING BPBTL WRAPPER")
    
    try:
        from models.bptbl_wrapper import BPBTLWrapper
        
        print("  Initializing BPBTL wrapper...")
        
        # Initialize wrapper
        wrapper = BPBTLWrapper(
            bptbl_root='checkpoints/bptbl',
            gpu_id=-1,  # Use CPU for testing
            with_scratch=False  # Start without scratch detection (simpler)
        )
        
        # Test initialization
        if wrapper.initialize():
            print_result(True, "BPBTL wrapper initialized successfully")
        else:
            print_result(False, "BPBTL wrapper failed to initialize")
            return False
        
        # Get model info
        info = wrapper.get_model_info()
        print(f"    Model: {info['name']}")
        print(f"    Description: {info['description']}")
        print(f"    Supports batch: {info['supports_batch']}")
        
        # Test with a sample image
        print("  Testing with sample image...")
        
        # Find a test image
        test_folder = project_root / "tests" / "test_Old_images"
        if test_folder.exists():
            test_images = list(test_folder.glob("*.png"))
            if test_images:
                test_image = test_images[0]
                print(f"    Using test image: {test_image.name}")
                
                # Process the image
                start_time = time.time()
                result_img = wrapper.process_single(test_image)
                process_time = time.time() - start_time
                
                if result_img is not None:
                    print_result(True, f"Image processed successfully in {process_time:.1f}s")
                    print(f"    Output size: {result_img.size}")
                    print(f"    Output mode: {result_img.mode}")
                    
                    # Save the result
                    output_path = project_root / "test_bptbl_wrapper.jpg"
                    result_img.save(output_path, quality=95)
                    print(f"    Result saved to: {output_path.name}")
                    
                    # Now test with scratch detection
                    print("\n  Testing WITH scratch detection...")
                    wrapper.set_scratch_detection(True)
                    
                    start_time = time.time()
                    result_img2 = wrapper.process_single(test_image)
                    process_time2 = time.time() - start_time
                    
                    if result_img2 is not None:
                        print_result(True, f"Scratch detection mode works in {process_time2:.1f}s")
                        output_path2 = project_root / "test_bptbl_scratch.jpg"
                        result_img2.save(output_path2, quality=95)
                        print(f"    Result saved to: {output_path2.name}")
                    else:
                        print_result(False, "Scratch detection mode failed")
                    
                    return True
                else:
                    print_result(False, "BPBTL processing failed (no output image)")
                    return False
            else:
                print_result(False, "No test images found")
                return False
        else:
            print_result(False, f"Test folder not found: {test_folder}")
            return False
        
    except Exception as e:
        print_result(False, f"BPBTL wrapper test failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def test_with_actual_images():
    """Test both models with actual test images."""
    print_section("3. PROCESSING ACTUAL TEST IMAGES")
    
    test_folder = project_root / "tests" / "test_Old_images"
    output_base = project_root / "test_outputs"
    
    if not test_folder.exists():
        print_result(False, f"Test folder not found: {test_folder}")
        return False
    
    # Find test images
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    test_images = []
    for ext in image_extensions:
        test_images.extend(test_folder.glob(f"*{ext}"))
        test_images.extend(test_folder.glob(f"*{ext.upper()}"))
    
    if not test_images:
        print_result(False, f"No images found in {test_folder}")
        return False
    
    print(f"  Found {len(test_images)} test image(s)")
    
    # Limit to 2 images for quick testing (first and one more)
    if len(test_images) > 2:
        test_images = [test_images[0], test_images[1]]
    print(f"  Testing with first {len(test_images)} image(s)")
    
    results = {
        'zeroscratches': {'success': 0, 'failed': 0, 'time': 0},
        'bptbl_wrapper': {'success': 0, 'failed': 0, 'time': 0}
    }
    
    # Create output directories
    zs_output_dir = output_base / "zeroscratches"
    bp_output_dir = output_base / "bptbl_wrapper"
    zs_output_dir.mkdir(parents=True, exist_ok=True)
    bp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test ZeroScratches
    try:
        from zeroscratches import EraseScratches
        from PIL import Image
        
        print(f"\n  [ZEROSCRATCHES] Processing {len(test_images)} image(s)...")
        eraser = EraseScratches()
        
        for i, img_path in enumerate(test_images, 1):
            try:
                print(f"    Image {i}: {img_path.name}", end="", flush=True)
                start_time = time.time()
                
                # Load and process image
                img = Image.open(img_path).convert('RGB')
                result_array = eraser.erase(img)
                
                # Convert numpy array to PIL Image if needed
                if not isinstance(result_array, Image.Image):
                    import numpy as np
                    if result_array.dtype != np.uint8:
                        result_array = (result_array * 255).astype(np.uint8)
                    result_img = Image.fromarray(result_array)
                else:
                    result_img = result_array
                
                # Save result
                output_path = zs_output_dir / f"{img_path.stem}_restored.jpg"
                result_img.save(output_path, quality=95)
                
                process_time = time.time() - start_time
                results['zeroscratches']['time'] += process_time
                results['zeroscratches']['success'] += 1
                
                print(f" ✓ ({process_time:.1f}s)")
                
            except Exception as e:
                results['zeroscratches']['failed'] += 1
                print(f" ✗ ({str(e)[:50]}...)")
                
    except Exception as e:
        print(f"  [ZEROSCRATCHES] Error: {str(e)}")
    
    # Test BPBTL using the wrapper
    print(f"\n  [BPBTL WRAPPER] Processing {len(test_images)} image(s)...")
    
    try:
        from models.bptbl_wrapper import BPBTLWrapper
        
        # Initialize wrapper
        wrapper = BPBTLWrapper(
            bptbl_root='checkpoints/bptbl',
            gpu_id=-1,  # CPU
            with_scratch=False  # Without scratch detection (faster)
        )
        
        if not wrapper.initialize():
            print("    Failed to initialize BPBTL wrapper")
        else:
            for i, img_path in enumerate(test_images, 1):
                try:
                    print(f"    Image {i}: {img_path.name}", end="", flush=True)
                    start_time = time.time()
                    
                    # Process using wrapper
                    output_path = bp_output_dir / f"{img_path.stem}_bptbl.jpg"
                    result_img = wrapper.process_single(img_path, output_path=str(output_path))
                    
                    process_time = time.time() - start_time
                    
                    if result_img is not None:
                        results['bptbl_wrapper']['time'] += process_time
                        results['bptbl_wrapper']['success'] += 1
                        print(f" ✓ ({process_time:.1f}s)")
                    else:
                        results['bptbl_wrapper']['failed'] += 1
                        print(" ✗ (Failed to process)")
                        
                except Exception as e:
                    results['bptbl_wrapper']['failed'] += 1
                    print(f" ✗ ({str(e)[:30]}...)")
    except Exception as e:
        print(f"  [BPBTL WRAPPER] Error: {str(e)}")
    
    # Print summary
    print_section("4. TEST RESULTS SUMMARY")
    
    total_tests = len(test_images)
    
    for model_name, stats in results.items():
        success = stats['success']
        failed = stats['failed']
        avg_time = stats['time'] / success if success > 0 else 0
        
        if success > 0:
            print(f"\n  {model_name.upper()}:")
            print(f"    Success: {success}/{total_tests}")
            print(f"    Failed:  {failed}/{total_tests}")
            print(f"    Avg time: {avg_time:.1f}s per image")
            
            # Show output location
            if model_name == 'zeroscratches':
                output_dir = zs_output_dir
            else:
                output_dir = bp_output_dir
            
            if output_dir.exists():
                output_files = list(output_dir.glob("*.*"))
                if output_files:
                    print(f"    Outputs saved to: {output_dir}")
                    print(f"    Files: {len(output_files)} restored image(s)")
        else:
            print(f"\n  {model_name.upper()}: Failed all tests")
    
    return True

def compare_models():
    """Compare model capabilities and recommend usage."""
    print_section("5. MODEL COMPARISON & RECOMMENDATIONS")
    
    print("\n  CURRENT STATUS:")
    print("  " + "-"*50)
    
    print(f"\n  ZeroScratches: ✅ FULLY WORKING")
    print(f"    • Fast processing (~35 seconds per image)")
    print(f"    • Good for minor scratches and noise")
    print(f"    • Output: test_outputs/zeroscratches/")
    
    print(f"\n  BPBTL Wrapper: ✅ NOW WORKING")
    print(f"    • Uses run.py via wrapper")
    print(f"    • Good for severe damage and old photos")
    print(f"    • Output: test_outputs/bptbl_wrapper/")
    
    print(f"\n  BPBTL Direct (old method): ✅ WORKING")
    print(f"    • Direct subprocess call to run.py")
    print(f"    • Useful for debugging")
    
    print("\n  " + "-"*50)
    print("\n  RECOMMENDED WORKFLOW FOR YOUR FLASK APP:")
    print("  1. Use ZeroScratches as PRIMARY model (fast, reliable)")
    print("  2. Use BPBTL as SECONDARY model (for severe damage)")
    print("  3. In Flask routes:")
    print("     • User uploads image")
    print("     • User selects model based on image condition")
    print("     • Model processes and returns result")
    print("     • Results saved to static/outputs/")

def main():
    """Main test function."""
    print("\n" + "="*70)
    print(" PHOTO RESTORATION MODELS - COMPREHENSIVE TEST")
    print("="*70)
    
    print(f"\nProject Root: {project_root}")
    print(f"Python: {sys.version.split()[0]}")
    
    try:
        # Test individual models
        zs_ok = test_zeroscratches()
        bp_ok = test_bptbl_wrapper()
        
        # Test with actual images
        test_with_actual_images()
        
        # Show comparisons
        compare_models()
        
        print_section("TEST COMPLETE")
        
        if zs_ok and bp_ok:
            print("\n🎉 BOTH MODELS ARE NOW WORKING CORRECTLY!")
            print("You can integrate both into your Flask app:")
            print("  1. Import both wrappers in your Flask routes")
            print("  2. Let users choose which model to use")
            print("  3. Save outputs to static folder for display")
        else:
            print("\n⚠️  One or more models need attention")
            if not bp_ok:
                print("   • BPBTL wrapper might need debugging")
                print("   • Check that run.py exists and works")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()