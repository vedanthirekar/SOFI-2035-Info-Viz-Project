"""
WSGI configuration for PythonAnywhere deployment
"""
import sys
import os

# Add your project directory to the sys.path
# IMPORTANT: Replace 'vedanthirekarSOFI' with your actual folder name
project_home = '/home/vedanthirekarSOFI/SOFI-2035_info-Viz-code'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Change to project directory
os.chdir(project_home)

# Import the Flask app from viz.py
try:
    from viz import server as application
except Exception as e:
    # If there's an error, create a simple app that shows the error
    from flask import Flask
    application = Flask(__name__)
    
    @application.route('/')
    def show_error():
        return f"""
        <h1>Error loading application</h1>
        <pre>{str(e)}</pre>
        <p>Check the error log for more details.</p>
        <p>Project path: {project_home}</p>
        <p>Python path: {sys.path}</p>
        """
