import os
import subprocess
import time
import webbrowser

project_path = r"C:\Users\USER\Desktop\project\new\Rice-Diseases-predict"
venv_path = os.path.join(project_path, "venv", "Scripts", "python.exe")

os.chdir(project_path)

subprocess.Popen([venv_path, "-m", "streamlit", "run", "app.py"])

time.sleep(2)

webbrowser.open("http://localhost:8501")
