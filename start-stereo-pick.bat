@echo off
chcp 65001 >nul
echo ==========================================
echo  双目视觉闭环抓取仿真
echo ==========================================
echo.
echo 启动方式:
echo   [1] 带可视化窗口 (默认)
echo   [2] 无窗口模式
echo   [3] 保存相机帧到 tmp/camera_frames
echo.
set /p choice="选择启动方式 (1/2/3): "

if "%choice%"=="2" (
    echo 正在启动无窗口模式...
    .venv\Scripts\python.exe src/stereo_pick_controller.py --no-viewer
) else if "%choice%"=="3" (
    echo 正在启动并保存相机帧...
    .venv\Scripts\python.exe src/stereo_pick_controller.py --save-frames
) else (
    echo 正在启动可视化仿真...
    .venv\Scripts\python.exe src/stereo_pick_controller.py
)

echo.
pause
