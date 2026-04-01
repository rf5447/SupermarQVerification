import sys
import cudaq

from assertions_helpers import assert_classical, assert_uniform, assert_product

# Helper for logdepth

def _fanout(indices):
    if len(indices) < 2:
        return
    mid = len(indices) // 2
    yield (indices[0], indices[mid])
    yield from _fanout(indices[:mid])
    yield from _fanout(indices[mid:])


# Subkernel: H gate only (used across all methods)

@cudaq.kernel
def apply_h(q: cudaq.qview):
    h(q[0])

@cudaq.kernel
def assert_apply_h(n: int):
    q = cudaq.qvector(n)
    apply_h(q)


# Main GHZ kernels (types: ladder, star, log-depth)

@cudaq.kernel
def ghz_ladder(n: int):
    q = cudaq.qvector(n)
    apply_h(q)
    for i in range(1, n):
        x.ctrl(q[i - 1], q[i])
    mz(q)


@cudaq.kernel
def ghz_star(n: int):
    q = cudaq.qvector(n)
    apply_h(q)
    for i in range(1, n):
        x.ctrl(q[0], q[i])
    mz(q)


_indices = list(range(5))
_pairs = list(_fanout(_indices))
_pair_c = [c for c, _ in _pairs]
_pair_t = [t for _, t in _pairs]
_num_pairs = len(_pair_c)

@cudaq.kernel
def ghz_logdepth(n: int):
    q = cudaq.qvector(n)
    apply_h(q)
    for i in range(_num_pairs):
        x.ctrl(q[_pair_c[i]], q[_pair_t[i]])
    mz(q)


# Assertions

n = 5

# assertions for ghz_after_h
# q[0] should be uniform and q[1] still |0>
print("q[0] uniform (after H):    ", assert_uniform(assert_apply_h, n, target_qubits=[0]))
print("q[1] classical (after H):  ", assert_classical(assert_apply_h, n, target_qubits=[1], expval=0))

# assertions for ghz main kernels
# each method should produce only all 0's or all 1's, and be entangled
for name, kernel in {"ladder": ghz_ladder, "star": ghz_star, "logdepth": ghz_logdepth}.items():
    print(f"\n{name}:")
    print("entangled: ", assert_product(kernel, n, group_a=[0], group_b=list(range(1, n)), negate=True))
