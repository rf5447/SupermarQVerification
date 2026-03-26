import collections
import pytest
import cudaq

from cudaqbenchmark import CudaQBenchmark
from cudaqfidelity import hellinger_fidelity


@pytest.fixture
def benchmark() -> CudaQBenchmark:
        """Simple one-qubit benchmark that creates an equal superposition state"""

    class _TestBenchmark(CudaQBenchmark):
        def kernel(self):
            @cudaq.kernel
            def test_kernel():
                q = cudaq.qvector(1) # single qubit
                h(q[0]) # H gate on q
                mz(q[0]) # measure
            return test_kernel

        def score(self, counts: collections.Counter) -> float:
            total = sum(counts.values())
            dist = {b: c / total for b, c in counts.items()}
            ideal = {"0": 0.5, "1": 0.5}
            return hellinger_fidelity(ideal, dist)

    return _TestBenchmark()


def test_benchmark_kernel(benchmark: CudaQBenchmark) -> None:
    kernel = benchmark.kernel()
    counts = cudaq.sample(kernel, shots=1000)

    p0 = counts.get("0", 0) / 1000
    p1 = counts.get("1", 0) / 1000

    assert abs(p0 - 0.5) < 0.1
    assert abs(p1 - 0.5) < 0.1
    # cannot check whether a circuit = another circuit because CUDA-Q cannot compare kernels directly (no circuit objects), so we validate kernel correctness through its behavior


def test_benchmark_score(benchmark: CudaQBenchmark) -> None:
    assert benchmark.score(collections.Counter({"0": 50, "1": 50})) == 1.0

# ideal_counts = cudaq.sample(kernel, qubit_count, shots_count=1000)
# ideal_counts.dump()