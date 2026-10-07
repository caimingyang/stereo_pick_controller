"""
MuJoCo-based Numerical Inverse Kinematics
==========================================
Compute joint angles directly inside the MuJoCo simulation,
using body positions and numerical Jacobians.
This avoids mismatches between external DH models and the scene.

Usage:
    from mujoco_ik import compute_ik_mujoco
    q_target = compute_ik_mujoco(env.model, env.data, gripper_body_id, target_pos, q0)
"""

import numpy as np
import mujoco

from robot_controller import JOINT_LIMITS


def get_body_jacobian_position(model, data, body_id, qpos_adr):
    """
    Compute numerical position Jacobian (3x6) for a body w.r.t. the 6 arm joints.
    Uses finite differences inside MuJoCo.
    """
    n_joints = len(qpos_adr)
    J = np.zeros((3, n_joints))

    # Current position
    mujoco.mj_forward(model, data)
    pos0 = data.xpos[body_id].copy()

    delta = 1e-5
    for i in range(n_joints):
        # Perturb joint i
        data.qpos[qpos_adr[i]] += delta
        mujoco.mj_forward(model, data)
        pos1 = data.xpos[body_id].copy()

        J[:, i] = (pos1 - pos0) / delta

        # Restore
        data.qpos[qpos_adr[i]] -= delta

    # Restore forward kinematics
    mujoco.mj_forward(model, data)
    return J, pos0


def compute_ik_mujoco(model, data, body_id, target_pos, qpos_adr,
                      q0=None, max_iter=100, tol=1e-4, damping=0.1):
    """
    Numerical IK using MuJoCo body positions.

    Args:
        model, data: MuJoCo model and data
        body_id: body to control (e.g. gripper)
        target_pos: [x, y, z] target in world coordinates
        qpos_adr: list of qpos addresses for the controlled joints
        q0: initial joint angles (defaults to current data.qpos)
        max_iter: max iterations
        tol: position error tolerance (m)
        damping: damping factor for least squares

    Returns:
        q: joint angles that reach (or best approximate) target_pos
    """
    n_joints = len(qpos_adr)

    if q0 is not None:
        for i in range(n_joints):
            data.qpos[qpos_adr[i]] = q0[i]
        mujoco.mj_forward(model, data)

    q = np.array([data.qpos[adr] for adr in qpos_adr])

    for iteration in range(max_iter):
        J, pos = get_body_jacobian_position(model, data, body_id, qpos_adr)
        err = target_pos - pos

        if np.linalg.norm(err) < tol:
            # print(f"  MuJoCo IK converged in {iteration} iterations")
            break

        # Damped least squares
        JtJ = J.T @ J
        damped = JtJ + damping**2 * np.eye(n_joints)
        dq = np.linalg.solve(damped, J.T @ err)

        # Update with step scaling for stability
        q += dq * 0.5

        # Clamp to limits
        for i in range(n_joints):
            lo, hi = JOINT_LIMITS[i]
            q[i] = float(np.clip(q[i], lo, hi))

        # Apply to MuJoCo
        for i in range(n_joints):
            data.qpos[qpos_adr[i]] = q[i]
        mujoco.mj_forward(model, data)

    # print(f"  MuJoCo IK: final error = {np.linalg.norm(err):.6f}")
    return q
