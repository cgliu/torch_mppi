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
    mppi_args = {
        "func_control_update_converged" : utils.inv_pen_control_update_converged,
        "func_comp_weights" : utils.inv_pen_comp_weights,
        "func_term_cost" : utils.inv_pen_term_cost,
        "func_run_cost" : utils.inv_pen_run_cost,
        "func_g" : utils.inv_pen_g,
        "func_F" : utils.inv_pen_F,
        "func_state_transform" : utils.inv_pen_state_transform,
        "func_filter_du" : utils.inv_pen_filter_du,
        "print_mppi" : Params.print_mppi,
    }
    params = {
        "num_samples" : Params.num_samples,
        "learning_rate" : Params.learning_rate,
        "init_state" : Params.init_state,
        "sample_u_traj" : Params.init_ctrl_seq,
        "ctrl_noise_covar" : Params.ctrl_noise_covar,
        "time_horizon" : Params.time_horizon,
        "per_ctrl_based_ctrl_noise" : Params.per_ctrl_based_ctrl_noise,
        "real_traj_cost" : Params.real_traj_cost,
    }
    # (x_hist, u_hist, sample_x_hist, sample_u_hist, rep_traj_cost_hist,
    #  time_hist) = mppi.mppisim(func_is_task_complete = utils.inv_pen_is_task_complete,
    #                                        func_gen_next_ctrl = utils.inv_pen_gen_next_ctrl,
    #                                        func_state_est = utils.inv_pen_state_est,
    #                                        func_apply_ctrl = utils.inv_pen_apply_ctrl,
    #                                        func_control_transform = utils.inv_pen_control_transform,
    #                                        plot_traj = Params.plot_traj,
    #                                        print_sim = Params.print_sim,
    #                                        params = params,
    #                                        mppi_args = mppi_args)
    mppi.interactive_mppisim(func_is_task_complete = utils.inv_pen_is_task_complete,
                             func_gen_next_ctrl = utils.inv_pen_gen_next_ctrl,
                             func_state_est = utils.inv_pen_state_est,
                             func_apply_ctrl = utils.inv_pen_apply_ctrl,
                             func_control_transform = utils.inv_pen_control_transform,
                             plot_traj = Params.plot_traj,
                             print_sim = Params.print_sim,
                             params = params,
                             mppi_args = mppi_args)


if __name__ == '__main__':
    main()
