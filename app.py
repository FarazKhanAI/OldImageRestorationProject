import os
import uuid
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import logging

# Import your model manager
try:
    from models.model_manager import ModelManager, restore_image
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import model manager: {e}")
    MODELS_AVAILABLE = False
    # Create mock functions for testing
    def restore_image(*args, **kwargs):
        return True

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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename):
    """Generate unique filename with timestamp"""
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    name, ext = os.path.splitext(original_filename)
    return f"{name}_{timestamp}_{unique_id}{ext}"


# Routes
# Add these routes to your existing app.py

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
        model_type = request.form.get('model')
        
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
            
            # Generate result filename
            result_filename = f"restored_{unique_filename}"
            result_path = RESULTS_FOLDER / result_filename
            
            # Store in session for next page
            session['original_image'] = str(unique_filename)
            session['restored_image'] = str(result_filename)
            session['model_used'] = model_type
            
            # For now, create a dummy result (we'll add actual model processing later)
            # This is temporary until we connect the actual model
            import shutil
            shutil.copy2(upload_path, result_path)
            
            flash(f'Image processing started with {model_type} model!', 'success')
            return redirect(url_for('results'))
        
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

@app.route('/results')
def results():
    """Show the results page."""
    # Get data from session
    original_image = session.get('original_image')
    restored_image = session.get('restored_image')
    model_used = session.get('model_used')
    
    if not all([original_image, restored_image, model_used]):
        flash('No image data found. Please upload an image first.', 'error')
        return redirect(url_for('home'))
    
    return render_template('results.html', 
                         original_image=original_image,
                         restored_image=restored_image,
                         model_used=model_used)

@app.route('/history')
def history():
    """Show the history page."""
    return render_template('history.html')










if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("BringMe - Starting Server")
    print("=" * 60)
    print(f"Models available: {MODELS_AVAILABLE}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    print(f"Max file size: {MAX_FILE_SIZE // (1024*1024)}MB")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)