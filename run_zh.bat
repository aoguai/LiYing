@echo off
set CLI_LANGUAGE=zh
setlocal enabledelayedexpansion

REM ��ȡ��ǰ�������ļ���Ŀ¼
set SCRIPT_DIR=%~dp0

REM ����Python������·������ĿĿ¼
set PYTHON_EXE=%SCRIPT_DIR%python-embed\python.exe
set SCRIPT_PATH=%SCRIPT_DIR%src\main.py

REM ����Ƿ����ļ���Ŀ¼���Ϸ�
if "%~1"=="" (
    echo �뽫ͼ���ļ���Ŀ¼�Ϸŵ��˽ű���
    pause
    exit /b
)

REM ��ȡ�Ϸŵ�·��
set INPUT_PATH=%~1

echo                  LiYing
echo Github: https://github.com/aoguai/LiYing
echo ����֤: AGPL-3.0
echo ----------------------------------------

REM ��ʾ�û�����ģ��·��
set /p "change_models=�Ƿ��޸�ģ��·������/��Ĭ��Ϊ�񣩣�"
if /i "!change_models!"=="��" (
    set /p "yolov8_path=���� YOLOv8 ģ��·����Ϊ��ʹ��Ĭ��·������"
    if "!yolov8_path!"=="" (
        set yolov8_param=
    ) else (
        set yolov8_param=--yolov8-model-path "!yolov8_path!"
    )

    set /p "yunet_path=���� YuNet ģ��·����Ϊ��ʹ��Ĭ��·������"
    if "!yunet_path!"=="" (
        set yunet_param=
    ) else (
        set yunet_param=--yunet-model-path "!yunet_path!"
    )

    set /p "rmbg_path=���� RMBG ģ��·����Ϊ��ʹ��Ĭ��·������"
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

REM ��ʾ�û��������
set /p "layout_only=�Ƿ�����в��ֶ������ı�������/��Ĭ��Ϊ��: "
if /i "!layout_only!"=="��" (
    set layout_only=--layout-only
    set change_background=--no-change-background
    set save_background=--no-save-background
    set rgb_list=255,255,255
) else (
    set layout_only=
    set /p "change_background=�Ƿ���ı�������/��Ĭ��Ϊ��: "
    if /i "!change_background!"=="��" (
        set change_background=--change-background
        set /p "rgb_list=����RGBͨ��ֵ���ö��ŷָ���Ĭ��Ϊ255,255,255��: "
        if "!rgb_list!"=="��" set rgb_list=255,0,0
        if "!rgb_list!"=="��" set rgb_list=12,92,165
        if "!rgb_list!"=="��" set rgb_list=255,255,255
        if "!rgb_list!"=="" set rgb_list=255,255,255
        set /p "save_background=�Ƿ񱣴���ı������ͼ����/��Ĭ��Ϊ��: "
        if /i "!save_background!"=="��" (
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

set /p "resize=�Ƿ����ͼ���С����/��Ĭ��Ϊ�ǣ�: "
if /i "!resize!"=="��" (
    set resize=--no-resize
    set save_resized=--no-save-resized
) else (
    set resize=--resize
    set /p "save_resized=�Ƿ񱣴������С���ͼ����/��Ĭ��Ϊ��: "
    if /i "!save_resized!"=="��" (
        set save_resized=--save-resized
    ) else (
        set save_resized=--no-save-resized
    )
    set /p "photo_type=������Ƭ���ͣ�Ĭ��Ϊһ�磩: "
    if "!photo_type!"=="" set photo_type=һ��
)

set /p "photo_sheet_size=������Ƭ�����С��Ĭ��Ϊ��磩: "
if "!photo_sheet_size!"=="" set photo_sheet_size=���

set /p "face_height_ratio=�����������߶0.30Ƭ�߶ȱ�����Ĭ�� 0.45����ֵԽ��ͷ��Խ�󣩣�"
if "!face_height_ratio!"=="" set face_height_ratio=0.30
set face_height_ratio=--face-height-ratio !face_height_ratio!

set /p "top_margin_ratio=������ͷ�����Ƭ�������ױ�����Ĭ�� 0.175����ֵԽ��ͷ��Խ���ƣ���"
if "!top_margin_ratio!"=="" set top_margin_ratio=0.175
set top_margin_ratio=--top-margin-ratio !top_margin_ratio!

set /p "compress=�Ƿ�ѹ��ͼ����/��Ĭ��Ϊ��: "
if /i "!compress!"=="��" (
    set compress=--compress
    set compress=--compress
    set /p "use_csv_size=ʹ��CSV�ļ��ĳߴ����ƣ���/��Ĭ��Ϊ�ǣ�: "
    if /i "!use_csv_size!"=="��" (
        set use_csv_size=--no-use-csv-size
        set /p "target_size=����Ŀ���ļ���С��KB����ֱ�ӻس�������: "
        if "!target_size!"=="" (
            set target_size=
            set /p "use_size_range=�Ƿ������ļ���С��Χ������/��Ĭ��Ϊ��: "
            if /i "!use_size_range!"=="��" (
                set /p "size_range=�����ļ���С��Χ��KB������ʽ����Сֵ,���ֵ�����磺10,20��: "
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

set /p "save_corrected=�Ƿ񱣴�У�����ͼ����/��Ĭ��Ϊ��: "
if /i "!save_corrected!"=="��" (
    set save_corrected=--save-corrected
) else (
    set save_corrected=--no-save-corrected
)

set /p "sheet_rows=������Ƭ�����������Ĭ��Ϊ3��: "
if "!sheet_rows!"=="" set sheet_rows=3

set /p "sheet_cols=������Ƭ�����������Ĭ��Ϊ3��: "
if "!sheet_cols!"=="" set sheet_cols=3

set /p "rotate=�Ƿ���Ƭ��ת90�ȣ���/��Ĭ��Ϊ��: "
if /i "!rotate!"=="��" (
    set rotate=--rotate
) else (
    set rotate=--no-rotate
)

set /p "add_crop_lines=����ֽ�����Ӳü��ߣ���/��Ĭ��Ϊ�ǣ��� "
if /i "!add_crop_lines!"=="��" (
    set add_crop_lines=--no-add-crop-lines
) else (
    set add_crop_lines=--add-crop-lines
)

set /p "layout_position=���벼��λ�ã�0-8��Ĭ��Ϊ 4��0=���Ͻǣ�1=������2=���Ͻǣ�3=���в���4=�м䣬5=���в���6=���½ǣ�7=�ײ���8=���½ǣ���"
if "!layout_position!"=="" set layout_position=4
set layout_position=--layout-position !layout_position!

set /p "photos_spacing=��������Ƭ֮��ļ�� (Ĭ��Ϊ 0����λ������): "
if "!photos_spacing!"=="" set photos_spacing=0
set photos_spacing=--photos-spacing !photos_spacing!

REM ����Ϸŵ���Ŀ���ļ�����Ŀ¼
if exist "%INPUT_PATH%\" (
    REM �����Ŀ¼�������������е�jpg��png�ļ�
    for %%f in ("%INPUT_PATH%\*.jpg" "%INPUT_PATH%\*.png") do (
        REM ��ȡ�ļ���·�����ļ���
        set "INPUT_FILE=%%~ff"
        set "OUTPUT_PATH=%%~dpnf_output%%~xf"
        
        REM ִ��Python�ű�����ͼ��
        start "" cmd /k "%PYTHON_EXE% %SCRIPT_PATH% "%%~ff" -b !rgb_list! -s "%%~dpnf_output%%~xf" -p !photo_type! --photo-sheet-size !photo_sheet_size! !compress! !save_corrected! !change_background! !save_background! -sr !sheet_rows! -sc !sheet_cols! !rotate! !resize! !save_resized! !layout_only! !add_crop_lines! !target_size! !size_range! !use_csv_size! !yolov8_param! !yunet_param! !rmbg_param! !layout_position! !photos_spacing! !face_height_ratio! !top_margin_ratio! & pause"
    )
) else (
    REM ������ļ���ֱ�Ӵ������ļ�
    set INPUT_DIR=%~dp1
    set INPUT_FILE=%~nx1
    set OUTPUT_PATH=%INPUT_DIR%%~n1_output%~x1
    
    REM ����ʹ����setlocal enabledelayedexpansion��ʹ��!variable_name!�����ñ���
    start "" cmd /k "%PYTHON_EXE% %SCRIPT_PATH% "!INPUT_PATH!" -b !rgb_list! -s "!OUTPUT_PATH!" -p !photo_type! --photo-sheet-size !photo_sheet_size! !compress! !save_corrected! !change_background! !save_background! -sr !sheet_rows! -sc !sheet_cols! !rotate! !resize! !save_resized! !layout_only! !add_crop_lines! !target_size! !size_range! !use_csv_size! !yolov8_param! !yunet_param! !rmbg_param! !layout_position! !photos_spacing! !face_height_ratio! !top_margin_ratio! & pause"
)

pause