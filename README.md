# 📸 BringMe — AI Photo Restoration Web App

<p align="center">
  <img src="https://img.shields.io/badge/BringMe-Photo%20Restoration-blue" />
  <img src="https://img.shields.io/badge/Flask-2.3.3-green" />
  <img src="https://img.shields.io/badge/Python-3.13%2B-yellow" />
  
</p>

<p align="center"><b>Revive your old memories with AI-Powered Photo Restoration</b></p>

## ✨ Features

- 🖼️ **AI-Powered Restoration** — Removes scratches, dust, and minor damage using ZeroScratches model
- ⚡ **Quick Processing** — Restores photos in seconds with lightweight AI inference
- 🌓 **Modern UI** — Responsive interface with both Light & Dark themes



## 🛠️ Tech Stack

- **Backend**: Flask (Python 3.13.0)
- **AI Models**: ZeroScratches, BPBTL (Coming Soon)
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with modern theme system
- **File Handling**: Werkzeug, PIL/Pillow
- **Performance**: Multi-threading, rate limiting






### **Clone Repository**
```bash
    git clone https://github.com/FarazKhanAI/OldImageRestorationProject.git
    cd OldImageRestorationProject
```

### **Create virtual environment**
```bash
    python -m venv venv
    source venv/bin/activate 
    # On Windows:venv\Scripts\activate
```


### **Install dependencies**
```bash
    pip install -r requirements.txt
```

### **Run the application**
```bash
    python app.py
```

### **Open in browser**
```bash
    http://localhost:5000
```




## Project Structure

```
BringMe/
├── app.py                    # Main Flask application
├── models/                   # AI model implementations
│   ├── base_restorer.py     # Base abstract class for restorers
│   ├── model_manager.py     # Model management and orchestration
│   └── zeroscratches_wrapper.py  # ZeroScratches model wrapper
├── static/                  # Static assets
│   ├── css/                # Stylesheets for all pages
│   ├── js/                 # JavaScript files
│   ├── uploads/            # User uploaded images
│   └── results/            # AI processed results
├── templates/              # HTML templates
│   ├── base.html          # Base template with header/footer
│   ├── home.html          # Upload page with model selection
│   ├── processing.html    # Loading page with progress
│   ├── results.html       # Results display page
│   └── history.html       # History dashboard
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```



## 👥 Development Team

- Faraz Khan  
- Jawad Khan  
- Gul-e-Rana  

**Repository:** https://github.com/FarazKhanAI/OldImageRestorationProject.git



## 🔮 Future Enhancements
### Batch processing for multiple images

- Additional restoration models

- Cloud storage integration

- Advanced editing tools


##

<div 
    align="center"> Made with ❤️ by the BringMe Team 
    </div>