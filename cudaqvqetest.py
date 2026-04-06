from __future__ import annotations

from cudaqvqe import VQEProxy_CUDAQ


def test_vqe_circuit() -> None:
    vqe = VQEProxy_CUDAQ(3, 1)
    kernel = vqe.kernel()
    assert len(kernels) == 2
    assert vqe.num_qubits == 3


def test_vqe_score() -> None:
    vqe = VQEProxy_CUDAQ(3, 1)
    kernel = vqe.kernel()
    probs = [vqe._ideal_probs(kernel)]
    assert vqe.score(probs) > 0.99