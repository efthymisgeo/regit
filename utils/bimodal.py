import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    folder_name = "mean_cond_distribution"
    experiment_name = "moving_mu"
    use_noise = True
    n_units = 1000
    dx = 10
    mu, sigma = .0, 0.005
    mu2, sigma2 = .06, 0.02
    perc_low = 0.15
    perc_high = 1 - perc_low
    noise_level = 0.01
    time = np.linspace(0, mu2, dx)
    for i, mu_t in enumerate(time):
        X1 = np.random.normal(mu, sigma, int(n_units*perc_low))
        X2 = np.random.normal(mu_t, sigma2, int(n_units*perc_high))
        X = np.concatenate([X1, X2])
        if use_noise:
            X = X + np.random.normal(0.0, noise_level, len(X))
            experiment_name = "noise"
        plt.hist(X, bins=100)
        plt.savefig(os.path.join(folder_name,
                                 experiment_name,
                                 f"bimodal_hist_{i}.png"),
                    bbox_inches="tight")
        plt.close()
        
        sorted_X = np.sort(X)
        plt.plot(np.arange(len(X)), sorted_X)
        plt.savefig(os.path.join(folder_name,
                                 experiment_name,
                                 f"bimodal_ranked_{i}.png"),
                    bbox_inches="tight")
        plt.close()