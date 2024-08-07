@echo off
set CLI_LANGUAGE=zh
setlocal enabledelayedexpansion

REM 获取当前批处理文件所在目录
set SCRIPT_DIR=%~dp0

REM 设置Python解释器路径和项目目录
set PYTHON_EXE=%SCRIPT_DIR%_myPython\python.exe
set SCRIPT_PATH=%SCRIPT_DIR%src\main.py

REM 检查是否拖放了文件或目录
if "%~1"=="" (
    echo 请拖放图片文件或目录到此脚本上
    pause
    exit /b
)

REM 获取拖放的路径
set INPUT_PATH=%~1

echo                  LiYing
echo Github: https://github.com/aoguai/LiYing
echo LICENSE AGPL-3 license
echo ----------------------------------------
REM 询问用户输入参数
REM 询问用户是否调整图像尺寸
set /p "resize=是否调整图像尺寸（yes/no，默认为 yes）："
if /i "!resize!"=="no" (
    set resize=--no-resize
    REM 如果用户选择不调整尺寸，则默认不保存调整尺寸后的图像
    set save_resized=--no-save-resized
) else (
    set resize=--resize
    REM 如果用户选择调整尺寸或者没有输入（默认yes），则询问是否保存调整后的图像
    set /p "save_resized=是否保存调整尺寸后的图像（yes/no，默认为 no）："
    if /i "!save_resized!"=="yes" (
        set save_resized=--save-resized
    ) else (
        set save_resized=--no-save-resized
    )
    REM 询问照片类型
    set /p "photo_type=请输入 照片类型（默认为 一寸照片）："
    if "!photo_type!"=="" set photo_type="一寸照片"
)

set /p "photo-sheet-size=请输入 相片纸尺寸（默认为 五寸照片）："
if "!photo-sheet-size!"=="" set photo-sheet-size="五寸照片"

set /p "compress=是否压缩图像（yes/no，默认为 yes）："
if /i "!compress!"=="no" set compress=--no-compress
if /i "!compress!"=="yes" set compress=--compress
if /i "!compress!"=="" set compress=--compress

set /p "save_corrected=是否保存修正图像后的图片（yes/no，默认为 no）："
if /i "!save_corrected!"=="yes" set save_corrected=--save-corrected
if /i "!save_corrected!"=="no" set save_corrected=--no-save-corrected

set /p "change_background=是否替换背景（yes/no，默认为 yes）："
if /i "!change_background!"=="no" (
    set change_background=--no-change-background
    REM 如果用户选择不替换背景，则默认不保存替换背景后的图像
    set save_background=--no-save-background
) else (
    set change_background=--change-background
    REM 询问背景颜色
    set /p "rgb_list=请输入 RGB 通道值列表（逗号分隔，默认为 255,255,255）："
    if "!rgb_list!"=="红" set rgb_list=255,0,0
    if "!rgb_list!"=="蓝" set rgb_list=12,92,165
    if "!rgb_list!"=="白" set rgb_list=255,255,255
    if "!rgb_list!"=="" set rgb_list=255,255,255
    REM 询问是否保存调整后的图像
    set /p "save_background=是否保存替换背景后的图像（yes/no，默认为 no）："
    if /i "!save_background!"=="yes" (
        set save_background=--save-background
    ) else (
        set save_background=--no-save-background
    )
)

set /p "sheet_rows=请输入 照片表格的行数（默认为 3）："
if "!sheet_rows!"=="" set sheet_rows=3

set /p "sheet_cols=请输入 照片表格的列数（默认为 3）："
if "!sheet_cols!"=="" set sheet_cols=3

set /p "rotate=是否旋转照片90度（yes/no，默认为 no）："
if /i "!rotate!"=="yes" set rotate=--rotate
if /i "!rotate!"=="no" set rotate=

REM 判断拖放的是文件还是目录
if exist "%INPUT_PATH%\" (
    REM 如果是目录，则遍历其中的所有 jpg 和 png 文件
    for %%f in ("%INPUT_PATH%\*.jpg" "%INPUT_PATH%\*.png") do (
        REM 提取文件夹路径和文件名
        set "INPUT_FILE=%%~ff"
        set "OUTPUT_PATH=%%~dpnf_output%%~xf"
        
        REM 执行Python脚本处理图像
        start "" cmd /k "%PYTHON_EXE% %SCRIPT_PATH% %%~ff -b %rgb_list% -s %%~dpnf_output%%~xf -p %photo_type% --photo-sheet-size %photo-sheet-size% %compress% %save_corrected% %change_background% %save_background% -sr %sheet_rows% -sc %sheet_cols% %rotate% %resize% %save_resized% & pause"
        
    )
) else (
    REM 如果是文件，则直接处理该文件
    set INPUT_DIR=%~dp1
    set INPUT_FILE=%~nx1
    set OUTPUT_PATH=%INPUT_DIR%%~n1_output%~x1

    REM 由于使用了 setlocal enabledelayedexpansion 需要使用 !变量名! 来引用变量
    start "" cmd /k "%PYTHON_EXE% %SCRIPT_PATH% !INPUT_PATH! -b %rgb_list% -s !OUTPUT_PATH! -p %photo_type% --photo-sheet-size %photo-sheet-size% %compress% %save_corrected% %change_background% %save_background% -sr %sheet_rows% -sc %sheet_cols% %rotate% %resize% %save_resized% & pause"
)

pause

