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

# Import configuration
from config import config

# Import your model manager - FIXED IMPORTS
try:
    from models.zeroscratches_wrapper import ZeroScratchesWrapper
    MODELS_AVAILABLE = True
    print("✓ ZeroScratchesWrapper imported successfully")
except ImportError as e:
    logging.error(f"Failed to import ZeroScratchesWrapper: {e}")
    MODELS_AVAILABLE = False

# Create Flask app
app = Flask(__name__)

# Load configuration based on environment
env = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[env])

# Initialize app (creates directories)
config[env].init_app(app)

# Set secret key from config
app.secret_key = app.config['SECRET_KEY']

# Rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

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
                    logging.info("Creating ZeroScratches wrapper...")
                    _model_wrapper = ZeroScratchesWrapper(
                        max_workers=app.config['ZEROSCRATCHES_MAX_WORKERS']
                    )
                    logging.info("ZeroScratches wrapper created (will initialize on first use)")
                except Exception as e:
                    logging.error(f"Failed to create model wrapper: {e}")
                    _model_wrapper = None
    
    return _model_wrapper

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_unique_filename(original_filename):
    """Generate unique filename with timestamp"""
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    name, ext = os.path.splitext(original_filename)
    return f"{name}_{timestamp}_{unique_id}{ext}"

def process_image_with_model(input_path, output_path, job_id, model_type='quick'):
    """Process image in background thread"""
    try:
        # Check if another thread is already processing this job
        if job_id in _results_cache and _results_cache[job_id].get('status') == 'processing':
            logging.warning(f"Job {job_id} is already being processed, skipping duplicate")
            return
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

# Routes
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
            upload_path = app.config['UPLOAD_FOLDER'] / unique_filename
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
            result_path = app.config['RESULTS_FOLDER'] / result_filename
            
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
        flash(f'File size exceeds {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB limit.', 'error')
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
    
    # Clean up old completed jobs
    cleanup_old_jobs()
    
    return jsonify(result_data)

def cleanup_old_jobs():
    """Remove old job data from cache"""
    # Simple implementation: keep last 20 jobs
    if len(_results_cache) > 20:
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
        file_path = app.config['RESULTS_FOLDER'] / filename
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
        'environment': env,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024*1024),
        'timestamp': time.time()
    })

# Initialize model on startup (but in background)
def initialize_model_in_background():
    """Initialize model in background thread to avoid blocking startup"""
    def init_model():
        try:
            logging.info("Starting background model initialization...")
            # Only create the wrapper, don't force initialization
            wrapper = ZeroScratchesWrapper(
                max_workers=app.config['ZEROSCRATCHES_MAX_WORKERS']
            )
            # Don't call initialize() here - let it happen on first use
            # Just store it so get_model_wrapper() returns it
            global _model_wrapper
            with _model_lock:
                _model_wrapper = wrapper
            logging.info("Model wrapper created (will initialize on first use)")
        except Exception as e:
            logging.error(f"Background model initialization error: {e}")
    
    # Only initialize if we're not already in a reloader process
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true' and MODELS_AVAILABLE:
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
    print(f"Environment: {env}")
    print(f"Models available: {MODELS_AVAILABLE}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Results folder: {app.config['RESULTS_FOLDER']}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    print(f"Debug mode: {app.config['DEBUG']}")
    print("=" * 60)
    
    # Initialize model in background
    if not app.config['DEBUG'] or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        initialize_model_in_background()
    
    # Get port from environment (Render sets PORT env var)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0',  # Important: Listen on all interfaces
        port=port,
        debug=app.config['DEBUG'],
        threaded=True,
        use_reloader=app.config['DEBUG'] and os.environ.get('WERKZEUG_RUN_MAIN') != 'true'
    )