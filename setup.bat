@echo off
REM Create and activate virtual environment
python -m venv venv
call venv\Scripts\activate

REM Install Python dependencies
pip install -r requirements.txt

echo.
echo Setup complete! Run 'uvicorn app.main:app --reload' to start the server.
