from __future__ import annotations

from typing import cast

import pytest
import cudaq

from cudaqbitcode import BitCode_CUDAQ


def test_bit_code_circuit() -> None:
    bc = BitCode_CUDAQ(3, 1, [1, 1, 1])
    drawn = cudaq.draw(bc.kernel(), bc.num_data_qubits)
    assert sum(1 for line in drawn.splitlines() if "q" in line and ":" in line) == 5


def test_bit_code_score() -> None:
    bc = BitCode_CUDAQ(4, 2, [0, 1, 1, 0])
    assert bc.score({"1011010010100": 100}) == 1


def test_invalid_inputs() -> None:
    with pytest.raises(
        ValueError, match=r"The length of `bit_state` must match the number of data qubits."
    ):
        BitCode_CUDAQ(3, 1, [0])

    with pytest.raises(TypeError, match=r"`bit_state` must be a `list\[int\]`."):
        BitCode_CUDAQ(3, 1, cast("list[int]", "010"))

    with pytest.raises(ValueError, match=r"Entries of `bit_state` must be 0, 1 integers."):
        BitCode_CUDAQ(3, 1, cast("list[int]", ["0", "1", "0"]))