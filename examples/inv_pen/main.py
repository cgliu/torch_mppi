import mppi
import matplotlib.pyplot as plt
import torch
import utils


class Params():
    num_samples = 50
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
    print_sim = False
    print_mppi = False


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
                               Params.print_mppi)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(rep_traj_cost_hist)
    ax.set_title('Cost history')
    plt.show(block=False)
    input("Press Enter to exit.")


if __name__ == '__main__':
    main()
