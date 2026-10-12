import os
import sys
import streamlit.web.cli as stcli

# IMPORTANT: Import all external libraries used in your app.py here 
# so PyInstaller knows to include them (e.g., import pandas, import plotly)

if __name__ == "__main__":
    # Resolve the absolute path to your app.py file
    script_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    # Emulate the 'streamlit run app.py' terminal command
    sys.argv = ["streamlit", "run", script_path, "--global.developmentMode=false"]
    sys.exit(stcli.main())
#pyinstaller --onedir --name="MyStreamlitApp" --copy-metadata streamlit --collect-all streamlit --add-data "app.py;." --exclude-module PyQt6 run_app.py
