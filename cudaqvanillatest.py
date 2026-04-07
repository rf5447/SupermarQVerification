from __future__ import annotations
import cudaq
from cudaqvanilla import QAOAVanillaProxy_CUDAQ


def test_qaoa_circuit() -> None:
    qaoa = QAOAVanillaProxy_CUDAQ(4)

    assert qaoa.num_qubits == 4
    assert len(qaoa.hamiltonian) == 6

    # Each Hamiltonian term contributes 2 CNOTs in the p=1 ansatz
    assert 2 * len(qaoa.hamiltonian) == 12


def test_qaoa_score() -> None:
    qaoa = QAOAVanillaProxy_CUDAQ(4)

    gamma, beta = qaoa.params
    kernel = qaoa.kernel()

    ideal_result = cudaq.sample(kernel, qaoa.num_qubits, float(gamma), float(beta), shots_count=10000)

    counts = ideal_result.get_register_counts("__global__")
    assert qaoa.score(counts) > 0.99