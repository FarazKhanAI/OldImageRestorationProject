import os
import uuid
import time
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import logging
import queue

# Import your model manager - FIXED IMPORTS
try:
    from models.zeroscratches_wrapper import ZeroScratchesWrapper
    MODELS_AVAILABLE = True
    print("✓ ZeroScratchesWrapper imported successfully")
except ImportError as e:
    logging.error(f"Failed to import ZeroScratchesWrapper: {e}")
    MODELS_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'static/uploads'
RESULTS_FOLDER = BASE_DIR / 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Global model instance (loaded once)
_model_wrapper = None
_model_lock = threading.Lock()
_processing_queue = queue.Queue()
_results_cache = {}  # job_id -> result data

def get_model_wrapper():
    """Get or create the model wrapper singleton with thread safety"""
    global _model_wrapper
    
    if _model_wrapper is None and MODELS_AVAILABLE:
        with _model_lock:
            if _model_wrapper is None:  # Double-check locking
                try:
                    # Lazy load the model only when needed
                    logging.info("Loading ZeroScratches model...")
                    _model_wrapper = ZeroScratchesWrapper(max_workers=2)
                    
                    # Initialize the model
                    if _model_wrapper.initialize():
                        logging.info("ZeroScratches model loaded successfully")
                    else:
                        logging.error("Failed to initialize ZeroScratches model")
                        _model_wrapper = None
                        
                except Exception as e:
                    logging.error(f"Failed to load model: {e}")
                    _model_wrapper = None
    
    return _model_wrapper

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename):
    """Generate unique filename with timestamp"""
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    name, ext = os.path.splitext(original_filename)
    return f"{name}_{timestamp}_{unique_id}{ext}"



def process_image_with_model(input_path, output_path, job_id, model_type='quick'):
    """Process image in background thread"""
    try:
        # Record start time
        _results_cache[job_id] = {
            'status': 'pending',
            'start_time': time.time()
        }
        
        model_wrapper = get_model_wrapper()
        
        if not model_wrapper:
            error_msg = "Model not available"
            _results_cache[job_id] = {'status': 'error', 'error': error_msg}
            return
        
        logging.info(f"Starting image processing for job {job_id}")
        
        # Update status to processing
        _results_cache[job_id]['status'] = 'processing'
        
        # Process the image
        start_time = time.time()
        
        # Load and process the image
        from PIL import Image
        img = Image.open(input_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Process with ZeroScratches model
        restored_img = model_wrapper.process_single(img)
        
        # Save the result
        if isinstance(restored_img, Image.Image):
            restored_img.save(output_path)
        else:
            # Convert numpy array to PIL Image
            import numpy as np
            if restored_img.dtype != np.uint8:
                restored_img = (restored_img * 255).astype(np.uint8)
            Image.fromarray(restored_img).save(output_path)
        
        processing_time = time.time() - start_time
        
        # Store result
        _results_cache[job_id] = {
            'status': 'completed',
            'output_path': str(output_path),
            'processing_time': processing_time
        }
        
        logging.info(f"Image processing completed for job {job_id} in {processing_time:.2f}s")
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error processing image for job {job_id}: {error_msg}")
        _results_cache[job_id] = {'status': 'error', 'error': error_msg}






def start_background_processing(input_path, output_path, job_id, model_type='quick'):
    """Start image processing in background thread"""
    thread = threading.Thread(
        target=process_image_with_model,
        args=(input_path, output_path, job_id, model_type),
        daemon=True  # Thread will be killed when main thread exits
    )
    thread.start()
    return thread

@app.route('/')
def home():
    """Render the home page."""
    return render_template('home.html')

@app.route('/process', methods=['POST'])
@limiter.limit("10 per minute")
def process_image():
    """Process the uploaded image with selected model."""
    try:
        # Check if file is present
        if 'image' not in request.files:
            flash('No image file provided.', 'error')
            return redirect(url_for('home'))
        
        file = request.files['image']
        model_type = request.form.get('model', 'quick')
        
        # Check if file is empty
        if file.filename == '':
            flash('No image selected.', 'error')
            return redirect(url_for('home'))
        
        # Check if file is allowed
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = generate_unique_filename(filename)
            
            # Save the uploaded file
            upload_path = UPLOAD_FOLDER / unique_filename
            file.save(upload_path)
            
            # Validate model type
            if model_type not in ['quick', 'deep']:
                flash('Invalid model selected.', 'error')
                return redirect(url_for('home'))
            
            # Only 'quick' model is available
            if model_type != 'quick':
                flash('Deep Restoration is coming soon. Using Quick Clean instead.', 'info')
                model_type = 'quick'
            
            # Generate result filename
            result_filename = f"restored_{unique_filename}"
            result_path = RESULTS_FOLDER / result_filename
            
            # Generate unique job ID
            job_id = str(uuid.uuid4())
            
            # Store in session for next pages
            session['job_id'] = job_id
            session['original_image'] = str(unique_filename)
            session['restored_image'] = str(result_filename)
            session['model_used'] = model_type
            session['processing_started'] = False
            
            # Initialize result cache for this job
            _results_cache[job_id] = {'status': 'pending'}
            
            # Start background processing
            start_background_processing(upload_path, result_path, job_id, model_type)
            
            # Redirect to processing page
            return redirect(url_for('processing'))
        
        else:
            flash('Invalid file type. Please upload an image file.', 'error')
            return redirect(url_for('home'))
            
    except RequestEntityTooLarge:
        flash('File size exceeds 10MB limit.', 'error')
        return redirect(url_for('home'))
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        flash('An error occurred while processing the image.', 'error')
        return redirect(url_for('home'))



@app.route('/processing')
def processing():
    """Show the processing page with progress updates."""
    job_id = session.get('job_id')
    
    if not job_id:
        flash('No processing job found.', 'error')
        return redirect(url_for('home'))
    
    # Mark processing as started
    session['processing_started'] = True
    
    # Get model type for display
    model_used = session.get('model_used', 'quick')
    
    return render_template('processing.html', 
                         job_id=job_id,
                         model_used=model_used)





@app.route('/api/processing-status/<job_id>')
def processing_status(job_id):
    """API endpoint to check processing status."""
    if job_id not in _results_cache:
        return jsonify({'status': 'not_found'})
    
    result_data = _results_cache.get(job_id, {})
    
    # Clean up old completed jobs (older than 1 hour)
    cleanup_old_jobs()
    
    return jsonify(result_data)

def cleanup_old_jobs():
    """Remove old job data from cache"""
    # Simple implementation: keep last 20 jobs
    if len(_results_cache) > 20:
        # Remove oldest jobs (based on insertion order)
        keys_to_remove = list(_results_cache.keys())[:-20]
        for key in keys_to_remove:
            if _results_cache[key].get('status') == 'completed':
                del _results_cache[key]

@app.route('/results')
def results():
    """Show the results page."""
    # Get data from session
    job_id = session.get('job_id')
    original_image = session.get('original_image')
    restored_image = session.get('restored_image')
    model_used = session.get('model_used')
    
    if not all([job_id, original_image, restored_image, model_used]):
        flash('No image data found. Please upload an image first.', 'error')
        return redirect(url_for('home'))
    
    # Check if processing is complete
    result_data = _results_cache.get(job_id, {})
    
    if result_data.get('status') == 'error':
        error_msg = result_data.get('error', 'Unknown error')
        flash(f'Image processing failed: {error_msg}', 'error')
        return redirect(url_for('home'))
    
    # If still processing, redirect to processing page
    if result_data.get('status') in ['pending', 'processing']:
        return redirect(url_for('processing'))
    
    # Get processing time from result data
    processing_time = result_data.get('processing_time', 0)
    
    return render_template('results.html', 
                         original_image=original_image,
                         restored_image=restored_image,
                         model_used=model_used,
                         processing_time=f"{processing_time:.2f}")

@app.route('/history')
def history():
    """Show the history page."""
    return render_template('history.html')

@app.route('/download/<filename>')
def download_file(filename):
    """Download a processed image."""
    try:
        file_path = RESULTS_FOLDER / filename
        if file_path.exists():
            return send_file(file_path, as_attachment=True)
        else:
            flash('File not found.', 'error')
            return redirect(url_for('home'))
    except Exception as e:
        app.logger.error(f"Error downloading file: {e}")
        flash('Error downloading file.', 'error')
        return redirect(url_for('home'))

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    model_status = "available" if get_model_wrapper() else "unavailable"
    
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'cache_size': len(_results_cache),
        'models_available': MODELS_AVAILABLE,
        'timestamp': time.time()
    })

# Initialize model on startup (but in background)
def initialize_model_in_background():
    """Initialize model in background thread to avoid blocking startup"""
    def init_model():
        try:
            logging.info("Starting background model initialization...")
            wrapper = get_model_wrapper()
            if wrapper:
                logging.info("Background model initialization complete")
            else:
                logging.warning("Background model initialization failed")
        except Exception as e:
            logging.error(f"Background model initialization error: {e}")
    
    # Start initialization in background
    init_thread = threading.Thread(target=init_model, daemon=True)
    init_thread.start()

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("BringMe - Starting Server")
    print("=" * 60)
    print(f"Models available: {MODELS_AVAILABLE}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    print(f"Max file size: {MAX_FILE_SIZE // (1024*1024)}MB")
    print("=" * 60)
    
    # Initialize model in background
    initialize_model_in_background()
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)