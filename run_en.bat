@echo off
set CLI_LANGUAGE=en
setlocal enabledelayedexpansion

REM Get the current batch file directory
set SCRIPT_DIR=%~dp0

REM Set Python interpreter path and project directory
set PYTHON_EXE=%SCRIPT_DIR%_myPython\python.exe
set SCRIPT_PATH=%SCRIPT_DIR%src\main.py

REM Check if files or directories were dragged and dropped
if "%~1"=="" (
    echo Please drag and drop image files or directories onto this script
    pause
    exit /b
)

REM Get the dropped path
set INPUT_PATH=%~1

echo                  LiYing
echo Github: https://github.com/aoguai/LiYing
echo LICENSE AGPL-3.0 license
echo ----------------------------------------
REM Prompt user for input parameters
REM Ask the user whether to resize images
set /p "resize=Resize images (yes/no, default is yes):"
if /i "!resize!"=="no" (
    set resize=--no-resize
    REM If the user chooses not to resize, do not save the resized image by default
    set save_resized=--no-save-resized
) else (
    set resize=--resize
    REM If the user chooses to resize or enters nothing (default yes), ask whether to save the resized image
    set /p "save_resized=Save resized images (yes/no, default is no):"
    if /i "!save_resized!"=="yes" (
        set save_resized=--save-resized
    ) else (
        set save_resized=--no-save-resized
    )
    REM Ask for photo type
    set /p "photo_type=Enter photo type (default is one_inch_photo):"
    if "!photo_type!"=="" set photo_type="one_inch_photo"
)

set /p "photo-sheet-size=Enter photo sheet size (default is five_inch_photo):"
if "!photo-sheet-size!"=="" set photo-sheet-size="five_inch_photo"

set /p "compress=Compress images (yes/no, default is yes):"
if /i "!compress!"=="no" set compress=--no-compress
if /i "!compress!"=="yes" set compress=--compress
if /i "!compress!"=="" set compress=--compress

set /p "save_corrected=Save corrected images (yes/no, default is no):"
if /i "!save_corrected!"=="yes" set save_corrected=--save-corrected
if /i "!save_corrected!"=="no" set save_corrected=--no-save-corrected

set /p "change_background=Change background (yes/no, default is yes):"
if /i "!change_background!"=="no" (
    set change_background=--no-change-background
    REM If the user chooses not to change the background, do not save the changed background image by default
    set save_background=--no-save-background
) else (
    set change_background=--change-background
    REM Ask for background color
    set /p "rgb_list=Enter RGB channel values (comma separated, default is 255,255,255):"
    if "!rgb_list!"=="red" set rgb_list=255,0,0
    if "!rgb_list!"=="blue" set rgb_list=12,92,165
    if "!rgb_list!"=="white" set rgb_list=255,255,255
    if "!rgb_list!"=="" set rgb_list=255,255,255
    REM Ask whether to save the changed background image
    set /p "save_background=Save images with changed background (yes/no, default is no):"
    if /i "!save_background!"=="yes" (
        set save_background=--save-background
    ) else (
        set save_background=--no-save-background
    )
)

set /p "sheet_rows=Enter the number of rows in the photo sheet (default is 3):"
if "!sheet_rows!"=="" set sheet_rows=3

set /p "sheet_cols=Enter the number of columns in the photo sheet (default is 3):"
if "!sheet_cols!"=="" set sheet_cols=3

set /p "rotate=Rotate photos 90 degrees (yes/no, default is no):"
if /i "!rotate!"=="yes" set rotate=--rotate
if /i "!rotate!"=="no" set rotate=

REM Check if the dropped item is a file or a directory
if exist "%INPUT_PATH%\" (
    REM If it's a directory, iterate through all jpg and png files in it
    for %%f in ("%INPUT_PATH%\*.jpg" "%INPUT_PATH%\*.png") do (
        REM Extract folder path and file name
        set "INPUT_FILE=%%~ff"
        set "OUTPUT_PATH=%%~dpnf_output%%~xf"
        
        REM Execute Python script to process the image
        start "" cmd /k "%PYTHON_EXE% %SCRIPT_PATH% %%~ff -b %rgb_list% -s %%~dpnf_output%%~xf -p %photo_type% --photo-sheet-size %photo-sheet-size% %compress% %save_corrected% %change_background% %save_background% -sr %sheet_rows% -sc %sheet_cols% %rotate% %resize% %save_resized% & pause"
    )
) else (
    REM If it's a file, process the file directly
    set INPUT_DIR=%~dp1
    set INPUT_FILE=%~nx1
    set OUTPUT_PATH=%INPUT_DIR%%~n1_output%~x1
    
    REM Due to setlocal enabledelayedexpansion, use !variable_name! to reference variables
    start "" cmd /k "%PYTHON_EXE% %SCRIPT_PATH% !INPUT_PATH! -b %rgb_list% -s !OUTPUT_PATH! -p %photo_type% --photo-sheet-size %photo-sheet-size% %compress% %save_corrected% %change_background% %save_background% -sr %sheet_rows% -sc %sheet_cols% %rotate% %resize% %save_resized% & pause"
)

pause
