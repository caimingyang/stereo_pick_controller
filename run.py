#!/usr/bin/env python3
"""
CAD Robot Project — Unified Launcher
====================================
One entry point for all project modules.

Usage:
    python run.py <command> [options]

Commands:
    vla-mock            VLA闭环: Mock模式（无需API， fastest）
    vla-qwen            VLA闭环: Qwen大模型+双目视觉感知
    vla-qwen-gui        VLA闭环: Qwen模式 + MuJoCo可视化窗口
    vla-openai          VLA闭环: OpenAI GPT-4V视觉模式（需API key）
    simulate            MuJoCo抓取仿真演示（带物理+可视化）
    urdf-test           URDF加载与基础仿真测试
    ik-demo             逆运动学求解演示
    cad-robot           生成6轴机械臂CAD模型（STEP）
    cad-plate           生成安装板CAD模型（STEP）
    cad-block           生成矩形块CAD模型（STEP）
    cad-arm2            生成2-DOF机械臂CAD模型（STEP）
    list                列出所有可用命令
"""

import argparse
import subprocess
import sys

PYTHON = sys.executable

COMMANDS = {
    "vla-mock": {
        "cmd": [PYTHON, "src/vlm_robot_controller.py", "--vlm", "mock", "--no-viewer"],
        "desc": "VLA闭环: Mock VLM模式（无需API）",
        "env": {},
    },
    "vla-qwen": {
        "cmd": [PYTHON, "src/vlm_robot_controller.py", "--vlm", "qwen", "--no-viewer"],
        "desc": "VLA闭环: Qwen大模型+双目视觉感知",
        "env": {},
    },
    "vla-qwen-gui": {
        "cmd": [PYTHON, "src/vlm_robot_controller.py", "--vlm", "qwen"],
        "desc": "VLA闭环: Qwen模式 + MuJoCo可视化窗口",
        "env": {},
    },
    "vla-openai": {
        "cmd": [PYTHON, "src/vlm_robot_controller.py", "--vlm", "openai", "--no-viewer"],
        "desc": "VLA闭环: OpenAI GPT-4V视觉模式（需OPENAI_API_KEY）",
        "env": {},
    },
    "simulate": {
        "cmd": [PYTHON, "src/simulate_pick.py"],
        "desc": "MuJoCo抓取仿真演示（带物理引擎+可视化）",
        "env": {},
    },
    "urdf-test": {
        "cmd": [PYTHON, "src/simulate_mujoco.py"],
        "desc": "URDF加载与基础仿真测试",
        "env": {},
    },
    "ik-demo": {
        "cmd": [PYTHON, "src/robot_controller.py"],
        "desc": "逆运动学（IK）求解演示：home -> approach -> grasp -> lift",
        "env": {},
    },
    "cad-robot": {
        "cmd": [PYTHON, "src/robot_arm_6dof.py"],
        "desc": "生成6轴机械臂CAD模型（输出 STEP/robot_arm_6dof.step）",
        "env": {},
    },
    "cad-plate": {
        "cmd": [PYTHON, "src/mounting_plate.py"],
        "desc": "生成安装板CAD模型（带M6孔和倒角）",
        "env": {},
    },
    "cad-block": {
        "cmd": [PYTHON, "src/block.py"],
        "desc": "生成简单矩形块CAD模型",
        "env": {},
    },
    "cad-arm2": {
        "cmd": [PYTHON, "src/robot_arm.py"],
        "desc": "生成2-DOF机械臂CAD模型",
        "env": {},
    },
}


def list_commands():
    print("=" * 60)
    print("可用命令列表")
    print("=" * 60)
    for name, info in COMMANDS.items():
        print(f"  {name:15s}  {info['desc']}")
    print("=" * 60)
    print(f"\n示例:  python run.py vla-qwen")
    print(f"       python run.py simulate")
    print(f"       python run.py ik-demo")


def run_command(name, extra_args):
    if name not in COMMANDS:
        print(f"未知命令: {name}")
        print(f"请使用 'python run.py list' 查看所有可用命令。\n")
        sys.exit(1)

    info = COMMANDS[name]
    cmd = info["cmd"] + extra_args

    print("=" * 60)
    print(f"启动: {info['desc']}")
    print("=" * 60)
    print(f"命令: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n用户中断。")


def main():
    parser = argparse.ArgumentParser(
        description="CAD Robot Project 统一启动器",
        usage="python run.py <command> [options]\n\n使用 'python run.py list' 查看所有命令。",
    )
    parser.add_argument("command", nargs="?", default="list", help="要执行的命令")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="传递给子命令的额外参数")
    args = parser.parse_args()

    if args.command == "list":
        list_commands()
    else:
        run_command(args.command, args.extra)


if __name__ == "__main__":
    main()
