from __future__ import annotations

from typing import cast

import pytest
import cudaq

from cudaqphasecode import PhaseCode_CUDAQ


def test_phase_code_circuit() -> None:
    bc = PhaseCode_CUDAQ(3, 1, [1, 1, 1])
    drawn = cudaq.draw(bc.kernel(), bc.num_data_qubits)
    assert sum(1 for line in drawn.splitlines() if "q" in line and ":" in line) == 5


def test_phase_code_score() -> None:
    bc = PhaseCode_CUDAQ(4, 2, [0, 1, 1, 0])
    assert bc.score({"1011010010100": 100}) == 1


def test_invalid_inputs() -> None:
    with pytest.raises(
        ValueError, match=r"The length of `phase_state` must match the number of data qubits."
    ):
        PhaseCode_CUDAQ(3, 1, [0])

    with pytest.raises(TypeError, match=r"`phase_state` must be a `list\[int\]`."):
        PhaseCode_CUDAQ(3, 1, cast("list[int]", "010"))

    with pytest.raises(ValueError, match=r"Entries of `phase_state` must be 0, 1 integers."):
        PhaseCode_CUDAQ(3, 1, cast("list[int]", ["0", "1", "0"]))