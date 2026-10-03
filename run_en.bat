@echo off
set CLI_LANGUAGE=en
setlocal enabledelayedexpansion

REM Get the current batch file directory
set SCRIPT_DIR=%~dp0

REM Set Python interpreter path and project directory
set PYTHON_EXE=%SCRIPT_DIR%python-embed\python.exe
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

REM Prompt user for model paths
set /p "change_models=Do you want to modify model paths? (yes/no, default is no): "
if /i "!change_models!"=="y" set "change_models=yes"
if /i "!change_models!"=="yes" (
    set /p "yolov8_path=Enter YOLOv8 model path (press Enter for default): "
    if "!yolov8_path!"=="" (
        set yolov8_param=
    ) else (
        set yolov8_param=--yolov8-model-path "!yolov8_path!"
    )

    set /p "yunet_path=Enter YuNet model path (press Enter for default): "
    if "!yunet_path!"=="" (
        set yunet_param=
    ) else (
        set yunet_param=--yunet-model-path "!yunet_path!"
    )

    set /p "rmbg_path=Enter RMBG model path (press Enter for default): "
    if "!rmbg_path!"=="" (
        set rmbg_param=
    ) else (
        set rmbg_param=--rmbg-model-path "!rmbg_path!"
    )
) else (
    set yolov8_param=
    set yunet_param=
    set rmbg_param=
)

REM Prompt user for input parameters
set /p "layout_only=Layout only without changing background (yes/no, default is no): "
if /i "!layout_only!"=="y" set "layout_only=yes"
if /i "!layout_only!"=="yes" (
    set layout_only=--layout-only
    set change_background=--no-change-background
    set save_background=--no-save-background
    set rgb_list=255,255,255
) else (
    set layout_only=
    set /p "change_background=Change background (yes/no, default is no): "
    if /i "!change_background!"=="y" set "change_background=yes"
if /i "!change_background!"=="yes" (
        set change_background=--change-background
        set /p "rgb_list=Enter RGB channel values (comma separated, default is 255,255,255): "
        if "!rgb_list!"=="red" set rgb_list=255,0,0
        if "!rgb_list!"=="blue" set rgb_list=12,92,165
        if "!rgb_list!"=="white" set rgb_list=255,255,255
        if "!rgb_list!"=="" set rgb_list=255,255,255
        set /p "save_background=Save images with changed background (yes/no, default is no): "
        if /i "!save_background!"=="y" set "save_background=yes"
if /i "!save_background!"=="yes" (
            set save_background=--save-background
        ) else (
            set save_background=--no-save-background
        )
    ) else (
        set change_background=--no-change-background
        set save_background=--no-save-background
        set rgb_list=255,255,255
    )
)

set /p "resize=Resize images (yes/no, default is yes): "
if /i "!resize!"=="n" set "resize=no"
if /i "!resize!"=="no" (
    set resize=--no-resize
    set save_resized=--no-save-resized
) else (
    set resize=--resize
    set /p "save_resized=Save resized images (yes/no, default is no): "
    if /i "!save_resized!"=="y" set "save_resized=yes"
if /i "!save_resized!"=="yes" (
        set save_resized=--save-resized
    ) else (
        set save_resized=--no-save-resized
    )
    set /p "photo_type=Enter photo type (default is One Inch): "
    if "!photo_type!"=="" set "photo_type=One Inch"
)

set /p "photo_sheet_size=Enter photo sheet size (default is Five Inch): "
if "!photo_sheet_size!"=="" set "photo_sheet_size=Five Inch"

set /p "face_height_ratio=Enter face height ratio of the final photo (default 0.30; larger values make the face larger): "
if "!face_height_ratio!"=="" set face_height_ratio=0.30
set face_height_ratio=--face-height-ratio !face_height_ratio!

set /p "top_margin_ratio=Enter top-of-head margin ratio (default 0.175; larger values move the face down): "
if "!top_margin_ratio!"=="" set top_margin_ratio=0.175
set top_margin_ratio=--top-margin-ratio !top_margin_ratio!

set /p "compress=Compress images (yes/no, default is no): "
if /i "!compress!"=="y" set "compress=yes"
if /i "!compress!"=="yes" (
    set compress=--compress
    set /p "use_csv_size=Use size limits from CSV file (yes/no, default is yes): "
    if /i "!use_csv_size!"=="n" set "use_csv_size=no"
if /i "!use_csv_size!"=="no" (
        set use_csv_size=--no-use-csv-size
        set /p "target_size=Enter target file size in KB (press Enter to skip): "
        if "!target_size!"=="" (
            set target_size=
            set /p "use_size_range=Do you want to set a file size range? (yes/no, default is no): "
            if /i "!use_size_range!"=="y" set "use_size_range=yes"
if /i "!use_size_range!"=="yes" (
                set /p "size_range=Enter file size range in KB (format: min,max, e.g., 10,20): "
                if "!size_range!"=="" (
                    set size_range=
                ) else (
                    set size_range=--size-range !size_range!
                )
            )
        ) else (
            set target_size=--target-size !target_size!
        )
    ) else (
        set use_csv_size=--use-csv-size
        set target_size=
        set size_range=
    )
) else (
    set compress=--no-compress
    set use_csv_size=--use-csv-size
    set target_size=
    set size_range=
)

set /p "save_corrected=Save corrected images (yes/no, default is no): "
if /i "!save_corrected!"=="y" set "save_corrected=yes"
if /i "!save_corrected!"=="yes" (
    set save_corrected=--save-corrected
) else (
    set save_corrected=--no-save-corrected
)

set /p "skin_retouch=Enable skin retouching (yes/no, default is no): "
if /i "!skin_retouch!"=="y" set "skin_retouch=yes"
if /i "!skin_retouch!"=="yes" (
    set skin_retouch=--skin-retouch
    set /p "retouch_degree=Enter retouch degree 0-1 (default is 0.7): "
    if "!retouch_degree!"=="" set retouch_degree=0.7
    set retouch_degree=--retouch-degree !retouch_degree!
    set /p "whitening_degree=Enter whitening degree 0-1 (default is 0.8): "
    if "!whitening_degree!"=="" set whitening_degree=0.8
    set whitening_degree=--whitening-degree !whitening_degree!
    set /p "skin_retouch_model_dir=Enter skin retouch models directory (press Enter for default): "
    if "!skin_retouch_model_dir!"=="" (
        set skin_retouch_model_dir=
    ) else (
        set skin_retouch_model_dir=--skin-retouch-model-dir "!skin_retouch_model_dir!"
    )
) else (
    set skin_retouch=
    set retouch_degree=
    set whitening_degree=
    set skin_retouch_model_dir=
)

set /p "sheet_rows=Enter the number of rows in the photo sheet (default is 3): "
if "!sheet_rows!"=="" set sheet_rows=3

set /p "sheet_cols=Enter the number of columns in the photo sheet (default is 3): "
if "!sheet_cols!"=="" set sheet_cols=3

set /p "rotate=Rotate photos 90 degrees (yes/no, default is no): "
if /i "!rotate!"=="y" set "rotate=yes"
if /i "!rotate!"=="yes" (
    set rotate=--rotate
) else (
    set rotate=--no-rotate
)

set /p "add_crop_lines=Add crop lines to the photo sheet (yes/no, default is yes): "
if /i "!add_crop_lines!"=="n" set "add_crop_lines=no"
if /i "!add_crop_lines!"=="no" (
    set add_crop_lines=--no-add-crop-lines
) else (
    set add_crop_lines=--add-crop-lines
)

set /p "layout_position=Enter layout position (0-8, default is 4,  0=top-left, 1=top, 2=top-right, 3=middle-left, 4=center, 5=middle-right, 6=bottom-left, 7=bottom, 8=bottom-right):  "
if "!layout_position!"=="" set layout_position=4
set layout_position=--layout-position !layout_position!

set /p "photos_spacing=Enter spacing between photos in pixels (default is 0): "
if "!photos_spacing!"=="" set photos_spacing=0
set photos_spacing=--photos-spacing !photos_spacing!

REM Check if the dropped item is a file or a directory
if exist "%INPUT_PATH%\" (
    REM If it's a directory, iterate through all jpg and png files in it
    for %%f in ("%INPUT_PATH%\*.jpg" "%INPUT_PATH%\*.png") do (
        REM Extract folder path and file name
        set "INPUT_FILE=%%~ff"
        set "OUTPUT_PATH=%%~dpnf_output%%~xf"

        REM Execute Python script to process the image
        start "" cmd /k "%PYTHON_EXE% %SCRIPT_PATH% "%%~ff" -b !rgb_list! -s "%%~dpnf_output%%~xf" -p "!photo_type!" --photo-sheet-size "!photo_sheet_size!" !compress! !save_corrected! !change_background! !save_background! -sr !sheet_rows! -sc !sheet_cols! !rotate! !resize! !save_resized! !layout_only! !add_crop_lines! !target_size! !size_range! !use_csv_size! !yolov8_param! !yunet_param! !rmbg_param! !layout_position! !photos_spacing! !face_height_ratio! !top_margin_ratio! !skin_retouch! !retouch_degree! !whitening_degree! !skin_retouch_model_dir! & pause"
    )
) else (
    REM If it's a file, process the file directly
    set INPUT_DIR=%~dp1
    set INPUT_FILE=%~nx1
    set OUTPUT_PATH=%INPUT_DIR%%~n1_output%~x1

    REM Due to setlocal enabledelayedexpansion, use !variable_name! to reference variables
    start "" cmd /k "%PYTHON_EXE% %SCRIPT_PATH% "!INPUT_PATH!" -b !rgb_list! -s "!OUTPUT_PATH!" -p "!photo_type!" --photo-sheet-size "!photo_sheet_size!" !compress! !save_corrected! !change_background! !save_background! -sr !sheet_rows! -sc !sheet_cols! !rotate! !resize! !save_resized! !layout_only! !add_crop_lines! !target_size! !size_range! !use_csv_size! !yolov8_param! !yunet_param! !rmbg_param! !layout_position! !photos_spacing! !face_height_ratio! !top_margin_ratio! !skin_retouch! !retouch_degree! !whitening_degree! !skin_retouch_model_dir! & pause"
)

pause
