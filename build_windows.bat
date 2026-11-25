@echo off
REM Сборка PhotoEditor.exe для Windows (рекомендуется Python 3.8 для Windows 7)
REM Запускать из корня проекта в активированном venv.

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

pyinstaller --noconfirm --clean --onefile --windowed ^
  --name PhotoEditor ^
  --hidden-import=PIL._tkinter_finder ^
  --collect-submodules methods ^
  main.py

echo.
echo Готово. EXE: dist\PhotoEditor.exe
pause
