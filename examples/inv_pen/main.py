import mppi
import torch
import matplotlib.pyplot as plt

import utils


class Params():
    num_samples = 5
    time_horizon = 1  # in seconds
    num_timesteps = 10
    ctrl_dim = 1
    init_ctrl_seq = torch.normal(torch.zeros(ctrl_dim, num_timesteps))
    init_state = torch.zeros(2, 1)
    ctrl_noise_covar = torch.tensor([[5e-1]])
    learning_rate = 0.01
    per_ctrl_based_ctrl_noise = 0.999
    real_traj_cost = True
    plot_traj = True
    print_sim = True
    print_mppi = True
    save_sampling = False
    sampling_filename = "inv_pen"


def main():
    (x_hist, u_hist, sample_x_hist, sample_u_hist, rep_traj_cost_hist,
     time_hist) = mppi.mppisim(utils.inv_pen_is_task_complete,
                               utils.inv_pen_control_update_converged,
                               utils.inv_pen_comp_weights,
                               utils.inv_pen_term_cost,
                               utils.inv_pen_run_cost,
                               utils.inv_pen_gen_next_ctrl,
                               utils.inv_pen_state_est,
                               utils.inv_pen_apply_ctrl,
                               utils.inv_pen_g, utils.inv_pen_F,
                               utils.inv_pen_state_transform,
                               utils.inv_pen_control_transform,
                               utils.inv_pen_filter_du,
                               Params.num_samples, Params.learning_rate,
                               Params.init_state,
                               Params.init_ctrl_seq, Params.ctrl_noise_covar,
                               Params.time_horizon,
                               Params.per_ctrl_based_ctrl_noise,
                               Params.real_traj_cost,
                               Params.plot_traj,
                               Params.print_sim,
                               Params.print_mppi,
                               Params.save_sampling,
                               Params.sampling_filename)
    plt.plot(rep_traj_cost_hist)
    plt.show()


if __name__ == '__main__':
    main()
