import pytest
import numpy as np
import cudaq
from cudaqghz import GHZ_CUDAQ

# --------------------------------------------------------
# Helper: expected GHZ statevector
# --------------------------------------------------------
def expected_ghz_statevector(n: int) -> np.ndarray:
    """
    Returns |GHZ_n> = (|0...0> + |1...1>) / sqrt(2)
    """
    vec = np.zeros(2**n, dtype=complex)
    vec[0] = 1 / np.sqrt(2)
    vec[-1] = 1 / np.sqrt(2)
    return vec

# --------------------------------------------------------
# Test GHZ circuit and statevector correctness
# --------------------------------------------------------
@pytest.mark.parametrize("method", ["ladder", "star", "logdepth"])
@pytest.mark.parametrize("num_qubits", [3, 4, 7])
def test_ghz_circuit_cudaq(method, num_qubits):
    ghz = GHZ_CUDAQ(num_qubits, method=method)

    kernel = ghz.kernel()

    # Draw circuit ensures kernel is well-formed
    diagram = cudaq.draw(kernel, num_qubits)
    assert diagram is not None

    # Execute as a statevector
    result = cudaq.get_state(kernel, num_qubits)
    psi = np.array(result)  # convert to numpy array

    expected = expected_ghz_statevector(num_qubits)

    # Cirq's comparison: equal "up to global phase"
    # Normalize both
    psi = psi / np.linalg.norm(psi)
    expected = expected / np.linalg.norm(expected)

    # Inner product magnitude = 1 if equal up to global phase
    overlap = abs(np.vdot(psi, expected))
    assert np.isclose(overlap, 1.0, atol=1e-6)


# --------------------------------------------------------
# Test that logdepth is shallower than ladder and star
# --------------------------------------------------------
def test_ghz_methods_depth():
    n = 8

    
    ghz_star = GHZ_CUDAQ(n, method="star")
    ghz_ladder = GHZ_CUDAQ(n, method="ladder")
    ghz_log = GHZ_CUDAQ(n, method="logdepth")

    # Count operations via drawing
    star_ops = cudaq.draw(ghz_star.kernel(), n).count("\n")
    ladder_ops = cudaq.draw(ghz_ladder.kernel(), n).count("\n")
    log_ops = cudaq.draw(ghz_log.kernel(), n).count("\n")

    # SupermarQ test: logdepth < ladder == star
    assert log_ops < ladder_ops == star_ops


# --------------------------------------------------------
# Test invalid method
# --------------------------------------------------------
def test_ghz_invalid_method():
    with pytest.raises(ValueError, match="Invalid GHZ method"):
        GHZ_CUDAQ(3, method="foo")


# --------------------------------------------------------
# Test score function
# --------------------------------------------------------
def test_ghz_score():
    ghz = GHZ_CUDAQ(3)
    # Perfect GHZ distribution
    counts = {"000": 500, "111": 500}
    assert ghz.score(counts) == 1.0
