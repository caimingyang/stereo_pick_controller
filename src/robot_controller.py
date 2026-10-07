"""
6-DOF Robot Arm Controller
==========================
Controls the robot arm to pick up an apple using inverse kinematics.

Usage:
    python robot_controller.py
"""

import math
import numpy as np


# ===== Robot DH Parameters =====
# [a, alpha, d, theta_offset]
# Standard DH convention
DH_TABLE = [
    [0.0,     math.pi/2,  0.160,  0.0],      # J1: base to link1
    [0.275,   0.0,        0.0,    math.pi/2], # J2: shoulder
    [0.228,   0.0,        0.0,    0.0],       # J3: elbow
    [0.0,     math.pi/2,  0.105,  0.0],       # J4: wrist roll
    [0.0,    -math.pi/2,  0.075,  0.0],       # J5: wrist pitch
    [0.0,     0.0,        0.045,  0.0],       # J6: end roll
]

# Joint limits (radians)
JOINT_LIMITS = [
    (-math.pi, math.pi),
    (-math.pi/2, math.pi/2),
    (-2.0, 2.0),
    (-math.pi, math.pi),
    (-math.pi/2, math.pi/2),
    (-math.pi, math.pi),
]

# Gripper
GRIPPER_MAX_OPEN = 0.080  # 80mm
GRIPPER_CLOSED = 0.005    # 5mm (apple hold)
GRIPPER_APPLE_DIAMETER = 0.075  # ~75mm apple


def dh_transform(a, alpha, d, theta):
    """Compute DH transformation matrix."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d    ],
        [0,   0,      0,     1    ]
    ])


def forward_kinematics(joint_angles):
    """
    Compute forward kinematics.
    Returns: 4x4 end-effector transformation matrix
    """
    T = np.eye(4)
    for i, params in enumerate(DH_TABLE):
        a, alpha, d, theta_offset = params
        theta = theta_offset + joint_angles[i]
        T_i = dh_transform(a, alpha, d, theta)
        T = T @ T_i
    return T


def compute_jacobian(joint_angles, delta=1e-6):
    """
    Compute numerical Jacobian (6xN) at current configuration.
    Returns linear and angular velocity Jacobians stacked.
    """
    n = len(joint_angles)
    T0 = forward_kinematics(joint_angles)
    pos0 = T0[:3, 3]
    rot0 = T0[:3, :3]

    J = np.zeros((6, n))
    for i in range(n):
        dq = np.copy(joint_angles)
        dq[i] += delta
        T1 = forward_kinematics(dq)
        pos1 = T1[:3, 3]
        rot1 = T1[:3, :3]

        # Linear velocity part
        J[:3, i] = (pos1 - pos0) / delta

        # Angular velocity part (approximation)
        dR = (rot1 - rot0) / delta
        J[3:, i] = np.array([dR[2,1] - dR[1,2],
                             dR[0,2] - dR[2,0],
                             dR[1,0] - dR[0,1]]) * 0.5

    return J


def inverse_kinematics(target_pos, target_rot=None, q0=None, max_iter=200, tol=1e-4):
    """
    Numerical inverse kinematics using damped least squares.

    target_pos: [x, y, z] target position
    target_rot: optional 3x3 rotation matrix
    q0: initial joint angles
    """
    if q0 is None:
        q0 = np.array([0.0, math.pi/4, -math.pi/2, 0.0, math.pi/4, 0.0])

    q = np.copy(q0)
    damping = 0.1

    for iteration in range(max_iter):
        T = forward_kinematics(q)
        pos = T[:3, 3]
        rot = T[:3, :3]

        # Position error
        err_pos = target_pos - pos

        # Rotation error (if target rotation specified)
        if target_rot is not None:
            err_rot_mat = target_rot @ rot.T - np.eye(3)
            err_rot = 0.5 * np.array([
                err_rot_mat[2,1] - err_rot_mat[1,2],
                err_rot_mat[0,2] - err_rot_mat[2,0],
                err_rot_mat[1,0] - err_rot_mat[0,1]
            ])
            error = np.concatenate([err_pos, err_rot])
        else:
            error = err_pos

        # Check convergence
        if np.linalg.norm(err_pos) < tol:
            print(f"  IK converged in {iteration} iterations")
            return q

        # Compute Jacobian
        J_full = compute_jacobian(q)
        if target_rot is None:
            J = J_full[:3, :]
        else:
            J = J_full

        # Damped least squares update
        JtJ = J.T @ J
        damped = JtJ + damping**2 * np.eye(JtJ.shape[0])
        dq = np.linalg.solve(damped, J.T @ error)

        # Update joints
        q += dq * 0.5  # scale step for stability

        # Clamp to limits
        for i in range(len(q)):
            lo, hi = JOINT_LIMITS[i]
            q[i] = np.clip(q[i], lo, hi)

    print(f"  IK warning: max iterations reached, error = {np.linalg.norm(err_pos):.6f}")
    return q


def plan_trajectory(waypoints, steps_per_segment=20):
    """
    Simple linear interpolation between waypoints.
    waypoints: list of joint angle arrays
    """
    trajectory = []
    for i in range(len(waypoints) - 1):
        q_start = waypoints[i]
        q_end = waypoints[i + 1]
        for t in np.linspace(0, 1, steps_per_segment):
            q = q_start + t * (q_end - q_start)
            trajectory.append(q)
    trajectory.append(waypoints[-1])
    return trajectory


class RobotController:
    """High-level controller for the 6-DOF arm."""

    def __init__(self):
        self.joint_angles = np.array([0.0, math.pi/4, -math.pi/2, 0.0, math.pi/4, 0.0])
        self.gripper_opening = GRIPPER_MAX_OPEN
        self.gripper_positions = {"left": GRIPPER_MAX_OPEN/2, "right": GRIPPER_MAX_OPEN/2}

    def get_end_effector_pose(self):
        T = forward_kinematics(self.joint_angles)
        pos = T[:3, 3]
        rot = T[:3, :3]
        return pos, rot

    def move_joints(self, target_angles, duration=2.0):
        """Move joints to target angles."""
        print(f"  Moving joints...")
        self.joint_angles = np.copy(target_angles)
        pos, _ = self.get_end_effector_pose()
        print(f"  End effector at: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    def open_gripper(self):
        """Open gripper fully."""
        print("  Opening gripper...")
        self.gripper_opening = GRIPPER_MAX_OPEN
        self.gripper_positions = {"left": GRIPPER_MAX_OPEN/2, "right": GRIPPER_MAX_OPEN/2}

    def close_gripper(self, diameter=GRIPPER_APPLE_DIAMETER):
        """Close gripper to grip object of given diameter."""
        print(f"  Closing gripper for object diameter {diameter*1000:.1f}mm...")
        self.gripper_opening = diameter + 0.005  # 5mm clearance
        self.gripper_positions = {
            "left": -self.gripper_opening/2,
            "right": self.gripper_opening/2
        }

    def pick_apple(self, apple_position, approach_offset=0.10):
        """
        Complete pick-and-place sequence for an apple.

        apple_position: [x, y, z] in metres
        approach_offset: approach height above apple (m)
        """
        print("=" * 50)
        print("PICK APPLE SEQUENCE")
        print("=" * 50)
        print(f"\nApple position: ({apple_position[0]:.3f}, {apple_position[1]:.3f}, {apple_position[2]:.3f})")

        # Current pose
        pos, rot = self.get_end_effector_pose()
        print(f"\n1. CURRENT POSE")
        print(f"   End effector: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        # Approach position (above apple)
        approach_pos = np.array(apple_position) + np.array([0, 0, approach_offset])
        print(f"\n2. COMPUTE IK FOR APPROACH")
        q_approach = inverse_kinematics(approach_pos, q0=self.joint_angles)

        # Grasp position (at apple)
        print(f"\n3. COMPUTE IK FOR GRASP")
        q_grasp = inverse_kinematics(np.array(apple_position), q0=q_approach)

        # Lift position (apple lifted)
        lift_pos = np.array(apple_position) + np.array([0, 0, approach_offset])
        print(f"\n4. COMPUTE IK FOR LIFT")
        q_lift = inverse_kinematics(lift_pos, q0=q_grasp)

        # Execute trajectory
        print(f"\n5. EXECUTE TRAJECTORY")

        # Open gripper
        self.open_gripper()

        # Move to approach
        print(f"\n   -> Move to approach position")
        self.move_joints(q_approach)

        # Move to grasp
        print(f"\n   -> Move down to grasp")
        self.move_joints(q_grasp)

        # Close gripper
        self.close_gripper()

        # Lift
        print(f"\n   -> Lift apple")
        self.move_joints(q_lift)

        print(f"\n6. PICK COMPLETE")
        pos, _ = self.get_end_effector_pose()
        print(f"   Apple held at: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"   Joint angles (deg): {np.degrees(self.joint_angles).round(1)}")
        print(f"   Gripper opening: {self.gripper_opening*1000:.1f}mm")
        print("=" * 50)

        return True


def main():
    print("6-DOF Robot Arm Controller")
    print("-" * 50)

    # Initialize robot
    robot = RobotController()

    # Show home position
    pos, rot = robot.get_end_effector_pose()
    print(f"\nHome position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # Apple position (in front of robot, on table)
    apple_pos = [0.35, 0.0, 0.080]  # 35cm forward, 8cm high (on table)

    # Execute pick
    robot.pick_apple(apple_pos)

    print("\nDone!")


if __name__ == "__main__":
    main()
