import numpy as np

def hellinger_fidelity(p: dict, q: dict) -> float:
    """
       Compute Hellinger fidelity, with p as the perfect distribution and q as the quantum one
       Return a number between 0 and 1
    """
    keys = set(p) | set(q)
    p_arr = np.array([p.get(k, 0.0) for k in keys])
    q_arr = np.array([q.get(k, 0.0) for k in keys])

    p_arr /= p_arr.sum()
    q_arr /= q_arr.sum()

    return (np.sqrt(p_arr) @ np.sqrt(q_arr)) ** 2
