from __future__ import annotations

import numpy as np
import cudaq

from cudaqmb import MerminBell_CUDAQ


def test_mermin_bell_circuit() -> None:
    mb = MerminBell_CUDAQ(3)
    assert mb.num_qubits == 3

    mb = MerminBell_CUDAQ(4)
    assert mb.num_qubits == 4

    mb = MerminBell_CUDAQ(5)
    assert mb.num_qubits == 5

def get_ideal_counts(kernel) -> dict[str, float]:
    state = cudaq.get_state(kernel)
    amps = np.array(state)
    num_qubits = int(np.log2(len(amps)))

    counts = {}
    for i, amp in enumerate(amps):
        prob = float(np.abs(amp) ** 2)
        if prob > 1e-15:
            counts[format(i, f"0{num_qubits}b")] = prob

    return counts

def test_mermin_bell_score() -> None:
    mb = MerminBell_CUDAQ(3)
    assert mb.score(get_ideal_counts(mb.kernel())) == 1

    mb = MerminBell_CUDAQ(4)
    assert mb.score(get_ideal_counts(mb.kernel())) == 1

    mb = MerminBell_CUDAQ(5)
    assert mb.score(get_ideal_counts(mb.kernel())) == 1