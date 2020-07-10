"""This script investigates how the drop prob curve changes
"""
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # initalize the state for 50 neurons upon a uniform prior
    mimic_flag = False
    folder = "dropcurves/"
    if mimic_flag:
        folder = folder + "zero_mimic/"
    else:
        folder = folder + "zero_normal/"
    n_units = 500
    p_init = 0.5
    x_axis = np.arange(n_units)
    init_state = p_init * np.ones(n_units)
    mu, sigma = 0.0, 0.1
    updates = 600 * 30 + 1
    p_high = 0.9
    p_low = 1 - p_high
    #betta = [0.9999, 0.9996, 0.9993, 0.999, 0.99, 0.97, 0.95, 0.93, 0.9, 0.87, 0.85]
    betta = [0.9, 0.87, 0.85, .83, .8, .7]
    plot_at = [0, 5, 10, 15, 50, 100, 200, 300]
    per_epoch = [600*i for i in range(1)]
    plot_at.extend(per_epoch)
    for k in range(updates):
        rankings = np.random.normal(mu, sigma, n_units)
        mask = rankings > mu
        rankings[mask] = p_high 
        rankings[~mask] = p_low
        if mimic_flag and k > 600*10:
            rankings[:25] = p_low
            rankings[-25:] = p_high
        if k == 0:
            mv_avg = init_state
            exp_avg = []
            for _ in betta:
                exp_avg.append(init_state)
        else:
            mv_avg = ((k-1) * mv_avg + rankings) / k
            for i, b in enumerate(betta):
                old = exp_avg[i]
                exp_avg[i] = b * old + (1-b) * rankings
        
        if k in plot_at:
            plt_mv_avg = np.sort(mv_avg)
            plt_exp_avg = [np.sort(avg) for avg in exp_avg]
            plt.plot(x_axis, plt_mv_avg, label="Moving Avg")
            for i, b in enumerate(betta):
                plt.plot(x_axis, plt_exp_avg[i], label=f"ExponentialAvg_{b}")
            plt.legend(loc='upper left')

            plt.xlabel('Ordered Units')
            plt.ylabel('Drop Distribution')
            if mimic_flag:
                f_name = f"units_{n_units}_dropcurves_{k}_mimic.png"
            else:
                f_name = f"units_{n_units}_dropcurves_{k}.png"
            plt.savefig(folder+f_name, bbox_inches='tight')
            plt.close()

    # sort by prob
    

    # add moving average
    # add exponential moving average
    # add noise levels (potam curves)