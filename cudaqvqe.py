from __future__ import annotations

import random

import cudaq
import numpy as np
import scipy.optimize as opt

from cudaqbenchmark import CudaQBenchmark


class VQEProxy_CUDAQ(CudaQBenchmark):

    def __init__(self, num_qubits: int, num_layers: int = 1):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.hamiltonian = self._gen_tfim_hamiltonian()
        self._params = self._gen_angles()

    # ------------------------------------------------------------
    # TFIM Hamiltonian description
    # ------------------------------------------------------------
    def _gen_tfim_hamiltonian(self):
        hamiltonian = []
        for i in range(self.num_qubits):
            hamiltonian.append(("X", i, 1))
        for i in range(self.num_qubits - 1):
            hamiltonian.append(("ZZ", (i, i + 1), 1))
        hamiltonian.append(("ZZ", (self.num_qubits - 1, 0), 1))
        return hamiltonian

    # ------------------------------------------------------------
    # Measured CUDA-Q kernel (matches Cirq circuit() output)
    # measure_in_x = 0  -> Z-basis circuit
    # measure_in_x = 1  -> X-basis circuit
    # ------------------------------------------------------------
    def kernel(self):
        @cudaq.kernel
        def k(
            num_qubits: int,
            num_layers: int,
            params: list[float],
            measure_in_x: int,
        ):
            q = cudaq.qvector(num_qubits)

            p = 0
            for _ in range(num_layers):
                # Ry rotation block
                for i in range(num_qubits):
                    ry(2.0 * params[p], q[i])
                    p += 1

                # Rz rotation block
                for i in range(num_qubits):
                    rz(2.0 * params[p], q[i])
                    p += 1

                # Entanglement block
                for i in range(num_qubits - 1):
                    x.ctrl(q[i], q[i + 1])

                # Ry rotation block
                for i in range(num_qubits):
                    ry(2.0 * params[p], q[i])
                    p += 1

                # Rz rotation block
                for i in range(num_qubits):
                    rz(2.0 * params[p], q[i])
                    p += 1

            if measure_in_x == 1:
                for i in range(num_qubits):
                    h(q[i])

            mz(q)

        return k

    # ------------------------------------------------------------
    # Unmeasured kernel for ideal probability extraction
    # ------------------------------------------------------------
    def _state_kernel(self):
        @cudaq.kernel
        def k(
            num_qubits: int,
            num_layers: int,
            params: list[float],
            measure_in_x: int,
        ):
            q = cudaq.qvector(num_qubits)

            p = 0
            for _ in range(num_layers):
                # Ry rotation block
                for i in range(num_qubits):
                    ry(2.0 * params[p], q[i])
                    p += 1

                # Rz rotation block
                for i in range(num_qubits):
                    rz(2.0 * params[p], q[i])
                    p += 1

                # Entanglement block
                for i in range(num_qubits - 1):
                    x.ctrl(q[i], q[i + 1])

                # Ry rotation block
                for i in range(num_qubits):
                    ry(2.0 * params[p], q[i])
                    p += 1

                # Rz rotation block
                for i in range(num_qubits):
                    rz(2.0 * params[p], q[i])
                    p += 1

            if measure_in_x == 1:
                for i in range(num_qubits):
                    h(q[i])

        return k

    # ------------------------------------------------------------
    # Runtime kernel arguments
    # ------------------------------------------------------------
    def kernel_args(self, measure_in_x: int):
        return (
            self.num_qubits,
            self.num_layers,
            [float(x) for x in self._params],
            measure_in_x,
        )

    # ------------------------------------------------------------
    # Ideal probability helper
    # ------------------------------------------------------------
    def _ideal_probs(self, measure_in_x: int):
        kernel = self._state_kernel()
        args = self.kernel_args(measure_in_x)

        state = cudaq.get_state(kernel, *args)
        arr = np.array(state)

        if arr.ndim == 2:
            probs = np.real(np.diag(arr))
        else:
            probs = np.abs(arr) ** 2

        out = {}
        for idx, prob in enumerate(probs):
            prob = float(np.real(prob))
            if prob > 0.0:
                bitstr = format(idx, f"0{self.num_qubits}b")[::-1]
                out[bitstr] = prob

        return out

    # ------------------------------------------------------------
    # Parity / energy helpers
    # ------------------------------------------------------------
    def _parity_ones(self, bitstr):
        one_count = 0
        for i in bitstr:
            if i == "1":
                one_count += 1
        return one_count % 2

    def _calc(self, bit_list, bitstr, probs):
        energy = 0.0
        for item in bit_list:
            if self._parity_ones(item) == 0:
                energy += probs.get(bitstr, 0.0)
            else:
                energy -= probs.get(bitstr, 0.0)
        return energy

    def _get_expectation_value_from_probs(self, probs_z, probs_x):
        avg_energy = 0.0

        # X-term contribution: sum_i X_i
        for bitstr in probs_x.keys():
            bit_list_x = [bitstr[i] for i in range(len(bitstr))]
            avg_energy += self._calc(bit_list_x, bitstr, probs_x)

        # ZZ-term contribution: sum_i Z_i Z_{i+1}, with wraparound
        for bitstr in probs_z.keys():
            bit_list_z = [bitstr[(i - 1):(i + 1)] for i in range(1, len(bitstr))]
            bit_list_z.append(bitstr[0] + bitstr[-1])
            avg_energy += self._calc(bit_list_z, bitstr, probs_z)

        return avg_energy

    # ------------------------------------------------------------
    # Classical optimization of parameters
    # ------------------------------------------------------------
    def _get_opt_angles(self):
        def f(params):
            old_params = self._params if hasattr(self, "_params") else None
            self._params = np.array(params, dtype=float)

            z_probs = self._ideal_probs(0)
            x_probs = self._ideal_probs(1)
            energy = self._get_expectation_value_from_probs(z_probs, x_probs)

            if old_params is not None:
                self._params = old_params

            return -energy  # minimize negative energy, matching Cirq style

        rng = np.random.default_rng(random.getrandbits(128))
        init_params = [
            rng.uniform() * 2.0 * np.pi
            for _ in range(self.num_layers * 4 * self.num_qubits)
        ]
        out = opt.minimize(f, init_params, method="COBYLA")

        return out["x"], out["fun"]

    def _gen_angles(self):
        params, _ = self._get_opt_angles()
        return np.array(params, dtype=float)

    # ------------------------------------------------------------
    # Score function
    # ------------------------------------------------------------
    def score(self, counts_list):
        counts_z, counts_x = counts_list

        shots_z = sum(counts_z.values())
        probs_z = {bitstr: count / shots_z for bitstr, count in counts_z.items()}

        shots_x = sum(counts_x.values())
        probs_x = {bitstr: count / shots_x for bitstr, count in counts_x.items()}

        experimental_expectation = self._get_expectation_value_from_probs(probs_z, probs_x)

        ideal_expectation = self._get_expectation_value_from_probs(
            self._ideal_probs(0),
            self._ideal_probs(1),
        )

        return float(
            1.0
            - abs(ideal_expectation - experimental_expectation) / abs(2.0 * ideal_expectation)
        )