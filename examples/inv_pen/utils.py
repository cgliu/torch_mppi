import torch
import numpy as np
import matplotlib.pyplot as plt

def inv_pen_apply_ctrl(x, u, dt):
    m = 1
    l = 1
    g = 9.8
    b = 0

    theta = x[0, :]
    theta_dot = x[1, :]

    dx_div_dt = x.clone().detach()

    dx_div_dt[0, :] = theta_dot
    dx_div_dt[1, :] = -g / l * torch.sin(theta) - (b * theta_dot + u[0, :]) / (m * l ** 2.)

    return x + dx_div_dt * dt


def inv_pen_comp_weights(traj_cost):
    """
    :return: the same size as traj_cost, by default, it is [1, #samples]
    """
    lambda_ = 0.01
    min_cost = torch.min(traj_cost)
    w = torch.exp(-1/lambda_ * (traj_cost - min_cost))

    return w/torch.sum(w)


def inv_pen_control_transform(sample_x, sample_u, dt):
    u = sample_u
    return u


def inv_pen_control_update_converged(du, iteration):
    tol = 0.01
    max_iteration = 5
    retval = False
    if iteration > max_iteration:
        retval = True
    return retval


def inv_pen_filter_du(du):
    return du


def inv_pen_F(x, u, dt):
    """
    :param x: a [#states, #samples] tensor
    :param u: a [#controls, $samples] tensor
    """
    m = 1
    l = 1
    g = 9.8
    b = 0
    I = m * l**2

    theta = x[0, :]
    theta_dot = x[1, :]

    dx_div_dt = x.clone().detach()

    dx_div_dt[0, :] = theta_dot
    dx_div_dt[1, :] = -(b/I) * theta_dot - (m*g*l/I) * torch.sin(theta) - u[0, :] / I

    return x + dx_div_dt * dt


def inv_pen_gen_next_ctrl(u):
    return torch.randn(1)


def inv_pen_g(u):
    clamped_u = u
    return clamped_u


def inv_pen_is_task_complete(x, t):
    is_task_complete = False
    if t > 5:
        is_task_complete = True
    return is_task_complete


def inv_pen_run_cost(x):
    """
    :param x: a [#states, #samples] tensor
    :return: a [1, #samples] tensor
    """
    Q = torch.tensor([[1., 0.],
                      [0., 1.]])
    goal_state = torch.tensor([[np.pi],
                               [0]])
    total_cost = 0.5 * torch.sum((x - goal_state) * (Q @ (x - goal_state)), dim=0, keepdim=True)

    assert total_cost.size() == torch.Size([1, x.size()[1]])
    return total_cost


def inv_pen_state_est(true_x):
    xdim = true_x.size()[0]
    H = torch.eye(xdim)
    return H @ true_x


def inv_pen_state_transform(x):
    sample_x = x
    return sample_x


def inv_pen_term_cost(x):
    """
    :param x: a [#states, #samples] tensor
    :return: a [1, #samples] tensor
    """
    Qf = torch.tensor([[100., 0.],
                       [0., 100.]])
    goal_state = torch.tensor([[np.pi],
                               [0.]])

    return 0.5 * torch.sum((x - goal_state) * (Qf @ (x - goal_state)), dim=0, keepdim=True)

def show_me(theta, ax=None):
    """
    Returns:
    A list of Artists.
    """
    pendulum = plt.Rectangle((0., 0.),
                             width=1.0,
                             height=0.02,
                             angle = 270 + theta * 180 / np.pi,
                             fc='y',
                             ec='r',
                             lw=1,
                             alpha=1.0)
    return [pendulum]
