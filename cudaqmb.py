from __future__ import annotations

import itertools
from typing import cast

import cirq
import cudaq
import numpy as np

import stabilizers
from cudaqbenchmark import CudaQBenchmark

class MerminBell_CUDAQ(CudaQBenchmark):
    """The Mermin-Bell benchmark is a test of a quantum computer's ability to exploit purely quantum
    phenomemna such as superposition and entanglement. It is based on the famous Bell-inequality
    tests of locality. Performance is based on a QPU's ability to prepare a GHZ state and measure
    the Mermin operator.
    """

    def __init__(self, num_qubits: int) -> None:
        """Initializes a `MerminBell_CUDAQ`.

        Args:
            num_qubits: The number of qubits.
        """
        self.num_qubits = num_qubits
        self.qubits = cirq.LineQubit.range(self.num_qubits)

        self.mermin_operator = self._mermin_operator(self.num_qubits)
        self.stabilizer, self.pauli_basis = stabilizers.construct_stabilizer(
            self.num_qubits, self.mermin_operator
        )

        # Ordered measurement circuit encoding:
        # opcode: 0 = H, 1 = S, 2 = CNOT, 3 = CZ, 4 = SWAP
        self.opcodes: list[int] = []
        self.op_q0: list[int] = []
        self.op_q1: list[int] = []

    # -------------------------------
    # Helper function: extract the Cirq measurement circuit into an
    # ordered gate stream for faithful replay inside a CUDA-Q kernel.
    # -------------------------------
    def _measurement_circuit_as_gate_stream(
        self,
    ) -> tuple[list[int], list[int], list[int]]:
        measurement_circuit = self._get_measurement_circuit().get_circuit()

        opcodes: list[int] = []
        op_q0: list[int] = []
        op_q1: list[int] = []

        for op in measurement_circuit.all_operations():
            if isinstance(op.gate, cirq.ops.MeasurementGate):
                break

            if op.gate == cirq.ops.H:
                opcodes.append(0)
                op_q0.append(cast("cirq.LineQubit", op.qubits[0]).x)
                op_q1.append(-1)

            elif op.gate == cirq.ops.S:
                opcodes.append(1)
                op_q0.append(cast("cirq.LineQubit", op.qubits[0]).x)
                op_q1.append(-1)

            elif op.gate == cirq.ops.CNOT:
                opcodes.append(2)
                op_q0.append(cast("cirq.LineQubit", op.qubits[0]).x)
                op_q1.append(cast("cirq.LineQubit", op.qubits[1]).x)

            elif op.gate == cirq.ops.CZ:
                opcodes.append(3)
                op_q0.append(cast("cirq.LineQubit", op.qubits[0]).x)
                op_q1.append(cast("cirq.LineQubit", op.qubits[1]).x)

            elif op.gate == cirq.ops.SWAP:
                opcodes.append(4)
                op_q0.append(cast("cirq.LineQubit", op.qubits[0]).x)
                op_q1.append(cast("cirq.LineQubit", op.qubits[1]).x)

            else:
                raise ValueError(f"Unsupported gate in measurement circuit: {op.gate}")

        return opcodes, op_q0, op_q1

    def kernel(self):
        """The Mermin-Bell CUDA-Q kernel, simultaneously measuring Mermin terms in a GHZ circuit.

        Returns:
            The Mermin-Bell CUDA-Q kernel.
        """
        self.opcodes, self.op_q0, self.op_q1 = self._measurement_circuit_as_gate_stream()

        num_ops = len(self.opcodes)

        @cudaq.kernel
        def k(
            num_qubits: int,
            opcodes: list[int],
            op_q0: list[int],
            op_q1: list[int],
            num_ops: int,
        ):
            q = cudaq.qvector(num_qubits)

            # Create a GHZ state
            rx(-np.pi / 2, q[0])
            for i in range(num_qubits - 1):
                x.ctrl(q[i], q[i + 1])

            # Replay the measurement circuit in the ORIGINAL order
            for i in range(num_ops):
                opcode = opcodes[i]
                q0 = op_q0[i]
                q1 = op_q1[i]

                if opcode == 0:
                    h(q[q0])
                elif opcode == 1:
                    s(q[q0])
                elif opcode == 2:
                    x.ctrl(q[q0], q[q1])
                elif opcode == 3:
                    z.ctrl(q[q0], q[q1])
                elif opcode == 4:
                    swap(q[q0], q[q1])

            mz(q)

        return k

    def kernel_args(self):
        """Return the arguments needed to run the CUDA-Q kernel."""
        return (
            self.num_qubits,
            self.opcodes,
            self.op_q0,
            self.op_q1,
            len(self.opcodes),
        )

    def score(self, counts: dict[str, float]) -> float:
        """Compute the score for the N-qubit Mermin-Bell benchmark.

        This function assumes the regular big endian ordering of bitstring results.

        Args:
            counts: A dictionary containing the measurement counts from circuit execution.

        Returns:
            The score for the Mermin-Bell benchmark score.
        """
        conjugation_rules: dict[cirq.Gate | None, dict[str, str]] = {
            cirq.ops.H: {"I": "I", "X": "Z", "Y": "-Y", "Z": "X"},
            cirq.ops.S: {"I": "I", "X": "Y", "Y": "-X", "Z": "Z"},
            cirq.ops.CNOT: {
                "II": "II",
                "IX": "IX",
                "XI": "XX",
                "XX": "XI",
                "IY": "ZY",
                "YI": "YX",
                "YY": "-XZ",
                "IZ": "ZZ",
                "ZI": "ZI",
                "ZZ": "IZ",
                "XY": "YZ",
                "YX": "YI",
                "XZ": "-YY",
                "ZX": "ZX",
                "YZ": "XY",
                "ZY": "IY",
            },
            cirq.ops.CZ: {
                "II": "II",
                "IX": "ZX",
                "XI": "XZ",
                "XX": "YY",
                "IY": "ZY",
                "YI": "YZ",
                "YY": "XX",
                "IZ": "IZ",
                "ZI": "ZI",
                "ZZ": "ZZ",
                "XY": "-YX",
                "YX": "-XY",
                "XZ": "XI",
                "ZX": "IX",
                "YZ": "YI",
                "ZY": "IY",
            },
            cirq.ops.SWAP: {
                "II": "II",
                "IX": "XI",
                "XI": "IX",
                "XX": "XX",
                "IY": "YI",
                "YI": "IY",
                "YY": "YY",
                "IZ": "ZI",
                "ZI": "IZ",
                "ZZ": "ZZ",
                "XY": "YX",
                "YX": "XY",
                "XZ": "ZX",
                "ZX": "XZ",
                "YZ": "ZY",
                "ZY": "YZ",
            },
        }

        measurement_circuit = self._get_measurement_circuit().get_circuit()

        expect_val = 0.0
        for mermin_coef, mermin_pauli in self.mermin_operator:
            measure_pauli = [p for p in mermin_pauli]
            parity = 1
            for op in measurement_circuit.all_operations():
                if isinstance(op.gate, cirq.ops.MeasurementGate):
                    break

                substr = [measure_pauli[cast("cirq.LineQubit", qubit).x] for qubit in op.qubits]
                conjugated_substr = conjugation_rules[op.gate]["".join(substr)]

                if conjugated_substr[0] == "-":
                    parity = -1 * parity
                    conjugated_substr = conjugated_substr[1:]

                for qubit, pauli in zip(op.qubits, conjugated_substr):
                    measure_pauli[cast("cirq.LineQubit", qubit).x] = pauli

            measurement_qubits = [i for i, pauli in enumerate(measure_pauli) if pauli == "Z"]
            measurement_coef = parity

            numerator = 0.0
            for bitstr, count in counts.items():
                parity = 1
                for qb in measurement_qubits:
                    if bitstr[qb] == "1":
                        parity = -1 * parity

                numerator += mermin_coef * measurement_coef * parity * count

            expect_val += numerator / sum(list(counts.values()))

        return (expect_val + 2 ** (self.num_qubits - 1)) / 2**self.num_qubits

    def _mermin_operator(self, num_qubits: int) -> list[tuple[float, str]]:
        """Generate the Mermin operator
        (https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.65.1838), or M_n
        (Eq. 2.8) in https://arxiv.org/pdf/2005.11271.pdf.
        """
        mermin_op = []
        for num_y in range(1, num_qubits + 1, 2):
            coef = (-1.0) ** (num_y // 2)

            for x_indices in itertools.combinations(range(num_qubits), num_qubits - num_y):
                pauli = np.array(["Y"] * num_qubits)
                pauli.put(x_indices, "X")
                mermin_op.append((coef, "".join(pauli)))

        return mermin_op

    def _get_measurement_circuit(self) -> stabilizers.MeasurementCircuit:
        """Return a MeasurementCircuit for simultaneous measurement of N operators."""
        assert self.stabilizer.shape == (
            2 * self.num_qubits,
            self.num_qubits,
        ), f"{self.num_qubits} qubits, but matrix shape: {self.stabilizer.shape}"

        for i in range(2 * self.num_qubits):
            for j in range(self.num_qubits):
                value = self.stabilizer[i, j]
                assert value in [0, 1], f"[{i}, {j}] index is {value}"

        measurement_circuit = stabilizers.MeasurementCircuit(
            cirq.Circuit(), self.stabilizer, self.num_qubits, self.qubits
        )

        stabilizers.prepare_X_matrix(measurement_circuit)
        stabilizers.row_reduce_X_matrix(measurement_circuit)
        stabilizers.patch_Z_matrix(measurement_circuit)
        stabilizers.change_X_to_Z_basis(measurement_circuit)
        measurement_circuit.get_circuit().append(cirq.measure(*self.qubits))

        return measurement_circuit