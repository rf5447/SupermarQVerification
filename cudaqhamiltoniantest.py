import cudaq
from cudaqhamiltonian import HamiltonianSimulation_CUDAQ


def test_hamiltonian_simulation_kernel() -> None:
    hs = HamiltonianSimulation_CUDAQ(4, 1, 1)
    assert hs.num_qubits == 4
    assert hs.kernel() is not None


def test_hamiltonian_simulation_score() -> None:
    cudaq.set_target("density-matrix-cpu")

    hs = HamiltonianSimulation_CUDAQ(4, 1, 1)

    assert hs._average_magnetization({"1111": 1}, 1) == -1.0
    assert hs._average_magnetization({"0000": 1}, 1) == 1.0

    ideal_counts = hs._ideal_counts()
    assert hs.score(ideal_counts) > 0.99