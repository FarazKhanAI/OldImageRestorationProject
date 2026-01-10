

## 🚀 Old Image Restoration App

### New Features:
1. **BPBTL Wrapper** - Complete wrapper for Bringing-Old-Photos-Back-to-Life model
2. **ZeroScratches Integration** - Fast scratch removal model
3. **Unified Model Manager** - Single interface for both models
4. **Comprehensive Testing** - Complete test suite with sample images
5. **Production-Ready Flask App** - Full web interface with upload/processing/download

### Model Status:
- ✅ **ZeroScratches**: Fully functional (~7-10 seconds per image)
- ✅ **BPBTL**: Fully functional (~10-15 seconds per image)
- ✅ **Dual-Model Support**: Users can choose between fast cleaning or deep restoration

### Project Structure Updates:
- \`models/\` - Complete model wrappers and test suite
- \`tests/\` - Test images and validation scripts
- \`utils/\` - File handling, image processing, parallel execution
- \`templates/\` - Enhanced Flask templates
- \`static/\` - CSS/JavaScript for modern UI

### Getting Started:
```bash
# Clone the repository
git clone https://github.com/FarazKhanAI/OldImageRestorationProject.git
cd OldImageRestorationProject

# Install dependencies
pip install -r requirements.txt
pip install zeroscratches  # Additional model package

# Download BPBTL checkpoints (see setup_models.py)
python utils/download_models.py

# Run the application
python app.py


### API Usage:
python
from models.model_manager import ModelManager

# Quick scratches removal
result = ModelManager.restore_image('zeroscratches', 'input.jpg', 'output.jpg')

# Deep restoration for damaged photos
result = ModelManager.restore_image('bptbl', 'old_photo.jpg', 'restored.jpg')


---

**Note**: BPBTL model checkpoints need to be downloaded separately due to size.
Run \`python utils/download_models.py\` or place checkpoints in \`checkpoints/bptbl/\`.


# Commit the updated README
git add README.md
git commit -m "Update README with new model integrations and setup instructions"
git push origin main