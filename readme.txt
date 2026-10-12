Please  extract MyStreamlit app folder 
you will find folder called internal 
and

Mysreamlitapp. exe 

Run exe the chat window will appear

for creating the exe file
pyinstaller --onedir --name="MyStreamlitApp" --copy-metadata streamlit --collect-all streamlit --add-data "app.py;." --exclude-module PyQt6 run_app.py

