from __future__ import annotations

import cudaq

from cudaqfswap import QAOAFermionicSwapProxy_CUDAQ


def test_qaoa_circuit() -> None:
    """Test the circuit generation function."""
    qaoa = QAOAFermionicSwapProxy_CUDAQ(4)
    assert qaoa.num_qubits == 4

    gamma = float(qaoa.params[0])
    beta = float(qaoa.params[1])

    drawing = cudaq.draw(qaoa.kernel(), qaoa.num_qubits, gamma, beta)
    assert drawing is not None


def test_qaoa_score() -> None:
    """Test the score evaluation function."""
    qaoa = QAOAFermionicSwapProxy_CUDAQ(4)

    gamma = float(qaoa.params[0])
    beta = float(qaoa.params[1])

    ideal_counts = cudaq.sample(qaoa.kernel(), qaoa.num_qubits, gamma, beta, shots_count=100000)

    assert qaoa.score(ideal_counts) > 0.99