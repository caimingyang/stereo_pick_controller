@echo off
chcp 65001 >nul
echo ==========================================
echo  逆运动学（IK）求解演示
echo ==========================================
.venv\Scripts\python.exe run.py ik-demo %*
pause
