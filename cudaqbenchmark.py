import abc
from typing import Any, Callable, Optional
import cudaq

class CudaQBenchmark(abc.ABC):
    """
    Abstract base class for CUDA-Q quantum benchmarks.

    Subclasses must implement:
        - kernel() -> CUDA-Q kernel function
        - score(counts) -> float in [0, 1]

    A benchmark instance should encapsulate all parameters needed
    to build and evaluate the kernel.

    Each instantiation of a Benchmark object represents a single, fully defined
    benchmark application. All the relevant parameters for a benchmark should
    be passed in upon creation, and will be used to generate the correct circuit
    and compute the final score.
    """

    @abc.abstractmethod
    def kernel(self) -> Callable[..., None]:
        """
        Returns a CUDA-Q kernel implementing the benchmark circuit.

        Example return value:
            @cudaq.kernel
            def ghz(n: int): ...
        """

    @abc.abstractmethod
    def score(self, counts: Any) -> float:
        """
        Computes a normalized [0,1] score from execution results.
        Typically uses bitstring counts from cudaq.sample().
        """

    # def run(self, shots: int = 1000, **kwargs):
    #     """
    #     Helper to sample the benchmark kernel using CUDA-Q.

    #     Example:
    #         counts = benchmark.run(shots=1000)
    #     """
    #     k = self.kernel()
    #     return cudaq.sample(k, shots=shots, **kwargs)
