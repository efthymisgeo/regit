import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from utils.model_utils import Scheduler, LinearScheduler, \
    MultiplicativeScheduler, ExponentialScheduler, StepScheduler, \
    PowerScheduler


if __name__ == "__main__":

    start_prob = 0.0
    end_prob = 0.5
    n_points = 1000
    gamma_1 = 0.004
    gamma_2 = 0.0007
    probs = [start_prob, end_prob] 
    mul = MultiplicativeScheduler(probs, n_points)
    exp = ExponentialScheduler(probs, n_points, gamma_1)
    lin = LinearScheduler(probs, n_points)
    step = StepScheduler(probs, n_points)
    power = PowerScheduler(probs, n_points, gamma_2)

    # m_time = mul.time
    # e_time = exp.get_function()
    # l_time = lin.time
    # s_time = step.time
    # p_time = power.get_function()
    # t = np.arange(n_points)

    m_time = []
    e_time = []
    l_time = []
    s_time = []
    p_rime = []
    t = []
    for k in range(2*n_points):
        t.append(k)
        m_time.append(mul.step())
        e_time.append(exp.step())
        l_time.append(lin.step())
        s_time.append(step.step())

    plt.plot(t, m_time)
    plt.plot(t, e_time)
    plt.plot(t, l_time)
    plt.savefig("schedulers",
                bbox_inches='tight')
    import pdb; pdb.set_trace()
    # for k in range(n_points):
    #     mul.step()
    #     exp.step()
    #     lin.step()
    #     step.step()
