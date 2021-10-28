import torch
import numpy as np
import matplotlib.pyplot as plt


def mppi(func_control_update_converged,
         func_comp_weights, func_term_cost, func_run_cost, func_g, func_F,
         func_state_transform, func_filter_du, num_samples, learning_rate,
         init_state, init_ctrl_seq, ctrl_noise_covar, time_horizon,
         per_ctrl_based_ctrl_noise, real_traj_cost, print_mppi, save_sampling,
         sampling_filename):

    # time stuff
    control_dim, num_timesteps = init_ctrl_seq.size()
    dt = time_horizon / num_timesteps

    # sample state stuff
    sample_init_state = func_state_transform(init_state)
    sample_state_dim, _ = sample_init_state.size()

    # state trajectories
    real_x_traj = torch.zeros(sample_state_dim, num_timesteps + 1)
    real_x_traj[:, 0:1] = sample_init_state

    x_traj = torch.zeros(sample_state_dim, num_samples, num_timesteps + 1)
    # todo
    # x_traj[:, : , 1) = repmat(sample_init_state, [1, num_samples])
    x_traj[:, :, 0] = sample_init_state.repeat(1, num_samples)
    # control stuff
    du = torch.ones(control_dim, num_timesteps) * 1e6

    # control sequence
    sample_u_traj = init_ctrl_seq

    # sampled control trajectories
    v_traj = torch.zeros(control_dim, num_samples, num_timesteps)

    # Begin mppi
    iteration = 1
    while not func_control_update_converged(du, iteration):
        last_sample_u_traj = sample_u_traj.clone().detach()

        # Noise generation
        flat_distribution = torch.normal(torch.zeros(control_dim, num_samples * num_timesteps))
        ctrl_noise_flat = ctrl_noise_covar @ flat_distribution
        ctrl_noise = torch.reshape(ctrl_noise_flat, (control_dim, num_samples, num_timesteps))

        # Compute sampled control trajectories
        # The number of trajectories that have both control and noise
        ctrl_based_ctrl_noise_samples = int(np.round(per_ctrl_based_ctrl_noise * num_samples))

        if ctrl_based_ctrl_noise_samples == 0:
            v_traj = ctrl_noise
        elif ctrl_based_ctrl_noise_samples == num_samples:
            v_traj = sample_u_traj.view(control_dim, 1, num_timesteps) + ctrl_noise
        else:
            v_traj[:, :ctrl_based_ctrl_noise_samples, :] = sample_u_traj.view(
                control_dim, 1, num_timesteps) + ctrl_noise[:, :ctrl_based_ctrl_noise_samples, :]
            v_traj[:, ctrl_based_ctrl_noise_samples:, :] = ctrl_noise[:, ctrl_based_ctrl_noise_samples:, :]

        for timestep_num in range(num_timesteps):
            # Forward propagation #sample trajectories
            x_traj[:, :, timestep_num+1] = func_F(x_traj[:, :, timestep_num], func_g(v_traj[:, :, timestep_num]), dt)
            if print_mppi:
                print("TN: {}, IN: {}, DU: {}".format(timestep_num, iteration, torch.mean(torch.sum(torch.abs(du), dim=0))))

        traj_cost = torch.zeros(1, num_samples)
        for timestep_num in range(num_timesteps):
            traj_cost = (traj_cost + func_run_cost(x_traj[:, :, timestep_num]) +
                         learning_rate * sample_u_traj[:, timestep_num].T  @  ctrl_noise_covar.inverse() @ (sample_u_traj[:, timestep_num] - v_traj[:, :, timestep_num]))

        traj_cost = traj_cost + func_term_cost(x_traj[:, :, timestep_num])

        # Weight and du calculation
        w = func_comp_weights(traj_cost)
        # todo()
        # du = reshape(sum(repmat(w, [control_dim, 1, num_timesteps]) .* ctrl_noise, 2), [control_dim, num_timesteps])
        du = torch.sum(w.view(control_dim, num_samples, 1) * ctrl_noise, dim=1)  # [control_dim, num_timesteps]

        # Filter the output from forward propagation
        du = func_filter_du(du)

        sample_u_traj = sample_u_traj + du
        iteration += 1
        if save_sampling:
            ...

    # why do we need to recalcuate these??
    # Weight and du calculation
    w = func_comp_weights(traj_cost)
    # du = reshape(sum(repmat(w, [control_dim, 1, num_timesteps]) .* ctrl_noise, 2), [control_dim, num_timesteps])
    du = torch.sum(w.view(control_dim, num_samples, 1) * ctrl_noise, dim=1)  # [control_dim, num_timesteps]

    # Filter the output from forward propagation
    du = func_filter_du(du)

    sample_u_traj = sample_u_traj + du
    iteration = iteration + 1

    if real_traj_cost:
        # Loop through the dynamics again to recalcuate traj_cost
        rep_traj_cost = 0
        for timestep_num in range(num_timesteps):
            # Forward propagation
            real_x_traj[:, timestep_num+1:timestep_num+2] = func_F(real_x_traj[:, timestep_num:timestep_num+1],
                                                                   func_g(sample_u_traj[:, timestep_num:timestep_num+1]), dt)

            rep_traj_cost = (rep_traj_cost + func_run_cost(real_x_traj[:, timestep_num:timestep_num+1]) +
                             learning_rate * sample_u_traj[:, timestep_num:timestep_num+1].T @ ctrl_noise_covar.inverse() @ (last_sample_u_traj[:, timestep_num:timestep_num+1] - sample_u_traj[:, timestep_num:timestep_num+1]))

        rep_traj_cost = rep_traj_cost + func_term_cost(real_x_traj[:, timestep_num:timestep_num+1])
    else:
        # normalize weights, in case they are not normalized
        normalized_w = w / torch.sum(w)  # todo() necessary??

        # Compute the representative trajectory cost of what actually happens
        # another way to think about this is weighted average of sample trajectory costs
        rep_traj_cost = torch.sum(normalized_w * traj_cost)
    # my_plot(real_x_traj)

    return sample_u_traj, rep_traj_cost.item()


def mppisim(func_is_task_complete, func_control_update_converged,
            func_comp_weights, func_term_cost, func_run_cost,
            func_gen_next_ctrl,
            func_state_est, func_apply_ctrl, func_g, func_F,
            func_state_transform,
            func_control_transform, func_filter_du, num_samples, learning_rate,
            init_state, init_ctrl_seq, ctrl_noise_covar, time_horizon,
            per_ctrl_based_ctrl_noise, real_traj_cost, plot_traj, print_sim,
            print_mppi,
            save_sampling, sampling_filename):

    # time stuff
    control_dim, num_timesteps = init_ctrl_seq.size()
    dt = time_horizon / num_timesteps
    time = 0
    time_hist = [time]

    # state stuff
    state_dim = init_state.size()[0]
    x_hist = [init_state]
    curr_x = init_state.clone().detach()

    # sample state stuff
    sample_init_state = func_state_transform(init_state)
    sample_x_hist = [sample_init_state]

    # control history
    sample_u_hist = []
    u_hist = []

    # control sequence
    sample_u_traj = init_ctrl_seq

    # trajectory cost history
    rep_traj_cost_hist = []

    total_timestep_num = 0
    while not func_is_task_complete(curr_x, time):
        # Use mppi
        sample_u_traj, rep_traj_cost = mppi(func_control_update_converged,
                                            func_comp_weights, func_term_cost, func_run_cost, func_g, func_F,
                                            func_state_transform, func_filter_du, num_samples, learning_rate,
                                            curr_x, sample_u_traj, ctrl_noise_covar, time_horizon,
                                            per_ctrl_based_ctrl_noise, real_traj_cost, print_mppi, save_sampling,
                                            sampling_filename)

        # Transform from sample_u to u
        u = func_control_transform(sample_x_hist[-1], sample_u_traj[:, None, 0], dt)

        # Apply control and log data
        true_x = func_apply_ctrl(x_hist[-1], u, dt)

        # state estimation after applying control
        curr_x = func_state_est(true_x)

        # Transform from state used in dynamics vs state used in control sampling
        sample_x = func_state_transform(curr_x)

        # Log state data
        x_hist.append(curr_x)

        sample_x_hist.append(sample_x)

        # Log control data
        u_hist.append(u)
        sample_u_hist.append(sample_u_traj[:, 0])

        # Log trajectory cost data
        rep_traj_cost_hist.append(rep_traj_cost)

        # if(print_sim)
        #   fprintf("Simtime: %d\n", time)
        # end

        # Move time forward
        time = time + dt
        time_hist.append(time)

        # Warmstart next control trajectory using past generated control trajectory
        new_sample_u_traj = sample_u_traj.clone().detach()
        new_sample_u_traj[:, :-1] = sample_u_traj[:, 1:]
        new_sample_u_traj[:, -1] = func_gen_next_ctrl(sample_u_traj[:, -1])
        sample_u_traj = new_sample_u_traj
        total_timestep_num = total_timestep_num + 1

    return (x_hist, u_hist, sample_x_hist, sample_u_hist, rep_traj_cost_hist,
            time_hist)
