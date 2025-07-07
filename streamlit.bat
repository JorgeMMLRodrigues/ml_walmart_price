@echo off

cd /d "C:\Users\estif\Desktop\DataAnalyst\_DataCourse_IronHack\Quests\ml_walmart_price"

call venv\Scripts\activate.bat

cd /d "C:\Users\estif\Desktop\DataAnalyst\_DataCourse_IronHack\Quests\ml_walmart_price\streamlitapp"


streamlit run app.py
