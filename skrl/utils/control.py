import isaacgym.torch_utils as torch_utils

import torch


def ik(jacobian_end_effector,
       current_position, current_orientation,
       goal_position, goal_orientation,
       damping_factor=0.05):
    """
    Damped Least Squares method: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
    """

    # compute position and orientation error
    position_error = goal_position - current_position
    q_r = torch_utils.quat_mul(goal_orientation, torch_utils.quat_conjugate(current_orientation))
    orientation_error = q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    dpose = torch.cat([position_error, orientation_error], -1).unsqueeze(-1)

    # solve damped least squares (dO = J.T * V)
    transpose = torch.transpose(jacobian_end_effector, 1, 2)
    lmbda = torch.eye(6).to(jacobian_end_effector.device) * (damping_factor ** 2)
    return (transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ dpose)

def osc(jacobian_end_effector, mass_matrix,
        current_position, current_orientation,
        goal_position, goal_orientation,
        current_dof_velocities,
        kp=5, kv=2):
    """
    https://studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
    """

    mass_matrix_end_effector = torch.inverse(jacobian_end_effector @ torch.inverse(mass_matrix) @ torch.transpose(jacobian_end_effector, 1, 2))

    # compute position and orientation error
    position_error = kp * (goal_position - current_position)
    q_r = torch_utils.quat_mul(goal_orientation, torch_utils.quat_conjugate(current_orientation))
    orientation_error = q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    dpose = torch.cat([position_error, orientation_error], -1)

    return torch.transpose(jacobian_end_effector, 1, 2) @ mass_matrix_end_effector @ (kp * dpose).unsqueeze(-1) - kv * mass_matrix @ current_dof_velocities
