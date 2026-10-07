@echo off
chcp 65001 >nul
echo ==========================================
echo  MuJoCo抓取仿真演示
echo ==========================================
.venv\Scripts\python.exe run.py simulate %*
pause
