import mppi
import matplotlib.pyplot as plt
import torch
import utils
from sim import Simulator
from collections import namedtuple
from collections import deque

CallbackRet = namedtuple("CallbackReturn", "patches plots")

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
    print_mppi = False

class PendulumProblem():
    def __init__(self, params, args):
        self.args = args
        self.params = params
        self.state_dim = params["init_state"].size()[0]
        self.curr_x = params["init_state"].clone().detach()
        self.sample_x = args["func_state_transform"](self.curr_x)
        control_dim, num_timesteps = params["sample_u_traj"].size()
        self.dt = params["time_horizon"] / num_timesteps
        self.sample_u_traj = params["sample_u_traj"]
        self.initial_state = params["init_state"]
        self.cost_hist = deque([0] * 10)

    def init_func(self):
        line = plt.Line2D(torch.arange(10), self.cost_hist, c='g', linewidth = 1)
        return CallbackRet(patches = utils.show_me(0),
                           plots = [line])

    def callback_func(self):
        self.sample_u_traj, rep_traj_cost, real_x_traj = mppi.mppi(
            init_state = self.curr_x,
            init_ctrl_seq = self.sample_u_traj,
            num_samples = self.params["num_samples"],
            learning_rate = self.params["learning_rate"],
            ctrl_noise_covar = self.params["ctrl_noise_covar"],
            time_horizon = self.params["time_horizon"],
            per_ctrl_based_ctrl_noise = self.params["per_ctrl_based_ctrl_noise"],
            real_traj_cost = self.params["real_traj_cost"],
            **self.args)

        self.cost_hist.appendleft(rep_traj_cost)
        self.cost_hist.pop()

        # Transform from sample_u to u
        u = self.params["func_control_transform"](self.sample_x,
                                                  self.sample_u_traj[:, None, 0],
                                                  self.dt)

        # Apply control and log data
        true_x = self.params["func_apply_ctrl"](self.curr_x, u, self.dt)

        # state estimation after applying control
        self.curr_x = self.params["func_state_est"](true_x)

        # Transform from state used in dynamics vs state used in control sampling
        self.sample_x = self.args["func_state_transform"](self.curr_x)
        theta = self.sample_x[0]

        # Warmstart next control trajectory using past generated control trajectory
        new_sample_u_traj = self.sample_u_traj.clone().detach()
        new_sample_u_traj[:, :-1] = self.sample_u_traj[:, 1:]
        new_sample_u_traj[:, -1] = self.params["func_gen_next_ctrl"](self.sample_u_traj[:, -1])
        self.sample_u_traj = new_sample_u_traj

        line = plt.Line2D(torch.arange(10), self.cost_hist, c='g', linewidth=1)
        return CallbackRet(patches = utils.show_me(theta),
                           plots = [line])

    def onclick(self, x, y):
        theta = torch.acos((- y) / torch.tensor([x, y]).norm())
        theta = theta if x > 0 else - theta
        self.initial_state[0] = theta
        self.initial_state[1] = 0
        self.curr_x = self.initial_state.clone().detach()

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
        "func_control_transform" : utils.inv_pen_control_transform,
        "func_gen_next_ctrl" : utils.inv_pen_gen_next_ctrl,
        "func_state_est" : utils.inv_pen_state_est,
        "func_apply_ctrl" : utils.inv_pen_apply_ctrl,
    }
    problem = PendulumProblem(params, mppi_args)
    sim = Simulator(problem)
    sim.run()

if __name__ == '__main__':
    main()
