import numpy as np
from scipy.special import rel_entr # 计算离散分布
from scipy.stats import entropy    # 计算连续分布

def k1_divergence(p, q):
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    return np.sum(rel_entr(p, q))

p = [0.1, 0.4, 0.3]
q = [0.3, 0.6, 0.5]

k1_div_discrete = k1_divergence(p, q)
k1_div_continuous = entropy(p, q)
print("KL divergence: ", k1_div_discrete, k1_div_continuous)