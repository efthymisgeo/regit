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
    n_points = [10, 50, 100, 250, 500, 750, 1000]
    max_pt = max(n_points)
    probs = [start_prob, end_prob] 
    mul = []
    m_time = []
    for i, n in enumerate(n_points):
        mul.append(MultiplicativeScheduler(probs, n))
        m_time.append([])
    
    t = []
    for k in range(max_pt):
        t.append(k)
        for i, m_s in enumerate(mul):
            m_time[i].append(m_s.step())

    for i, m_t in enumerate(m_time):
        plt.plot(t, m_t)
    
    plt.savefig("schedulers",
                bbox_inches='tight')
    import pdb; pdb.set_trace()
    # for k in range(n_points):
    #     mul.step()
    #     exp.step()
    #     lin.step()
    #     step.step()
