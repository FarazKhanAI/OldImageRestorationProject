// DOM Elements
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.querySelector('.theme-icon');
const themeText = document.querySelector('.theme-text');
const htmlElement = document.documentElement;
const currentYearElement = document.getElementById('currentYear');

// Initialize current year in footer
document.addEventListener('DOMContentLoaded', function() {
    // Set current year in copyright
    currentYearElement.textContent = new Date().getFullYear();
    
    // Initialize theme from localStorage or system preference
    initializeTheme();
    
    // Add event listener to theme toggle
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
    
    // Initialize flash messages if any
    initializeFlashMessages();
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Theme Management Functions
function initializeTheme() {
    const savedTheme = localStorage.getItem('bringme-theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme) {
        setTheme(savedTheme);
    } else {
        // Default to system preference or dark theme
        const defaultTheme = systemPrefersDark ? 'dark' : 'light';
        setTheme(defaultTheme);
    }
}

function toggleTheme() {
    const currentTheme = htmlElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    saveThemePreference(newTheme);
}

function setTheme(theme) {
    htmlElement.setAttribute('data-theme', theme);
    updateThemeButton(theme);
}

function updateThemeButton(theme) {
    if (!themeIcon || !themeText) return;
    
    if (theme === 'dark') {
        themeIcon.className = 'fas fa-moon theme-icon';
        themeText.textContent = 'Dark Mode';
    } else {
        themeIcon.className = 'fas fa-sun theme-icon';
        themeText.textContent = 'Light Mode';
    }
}

function saveThemePreference(theme) {
    localStorage.setItem('bringme-theme', theme);
}

// Flash Messages Functions
function initializeFlashMessages() {
    // Check if there are any Flask flash messages
    const flashContainer = document.getElementById('flashMessages');
    if (!flashContainer) return;
    
    // Example: You can dynamically add flash messages here
    // In practice, this will be populated by Flask's flash() function
    // For now, we'll create a function to show messages
}

// Function to show flash messages (can be called from other scripts)
function showFlashMessage(message, type = 'info', duration = 5000) {
    const flashContainer = document.getElementById('flashMessages');
    if (!flashContainer) return;
    
    const messageElement = document.createElement('div');
    messageElement.className = `flash-message ${type}`;
    messageElement.innerHTML = `
        <span>${message}</span>
        <button class="flash-close" aria-label="Close message">&times;</button>
    `;
    
    flashContainer.appendChild(messageElement);
    
    // Add click event to close button
    const closeButton = messageElement.querySelector('.flash-close');
    closeButton.addEventListener('click', () => {
        messageElement.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            flashContainer.removeChild(messageElement);
        }, 300);
    });
    
    // Auto-remove after duration
    if (duration > 0) {
        setTimeout(() => {
            if (messageElement.parentNode === flashContainer) {
                messageElement.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => {
                    if (messageElement.parentNode === flashContainer) {
                        flashContainer.removeChild(messageElement);
                    }
                }, 300);
            }
        }, duration);
    }
    
    // Add CSS animation for slide out
    if (!document.querySelector('#flash-animations')) {
        const style = document.createElement('style');
        style.id = 'flash-animations';
        style.textContent = `
            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
}

// Utility Functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Make functions available globally
window.BringMe = {
    showFlashMessage,
    formatFileSize,
    debounce,
    toggleTheme
};

// Listen for system theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
    // Only change theme if user hasn't explicitly set a preference
    if (!localStorage.getItem('bringme-theme')) {
        const newTheme = e.matches ? 'dark' : 'light';
        setTheme(newTheme);
    }
});