import numpy as np
import cudaq

from statistical_assertions import (
    classical_assertion as _classical_stat,
    uniform_assertion as _uniform_stat,
    product_assertion as _product_stat,
)
from state_assertions import (
    classical_assertion as _classical_sv,
    uniform_assertion as _uniform_sv,
    product_assertion as _product_sv,
)


# --- Statistical helpers (cudaq.sample) ---

def assert_classical(kernel, *kernel_args, shots=1000, target_qubits=None, pcrit=None, expval=None, negate=False):
    counts = dict(cudaq.sample(kernel, *kernel_args, shots_count=shots).items())
    return _classical_stat(counts, target_qubits=target_qubits, pcrit=pcrit, expval=expval, negate=negate)


def assert_uniform(kernel, *kernel_args, shots=1000, target_qubits=None, pcrit=None, negate=False):
    counts = dict(cudaq.sample(kernel, *kernel_args, shots_count=shots).items())
    return _uniform_stat(counts, target_qubits=target_qubits, pcrit=pcrit, negate=negate)


def assert_product(kernel, *kernel_args, group_a, group_b, shots=1000, pcrit=None, negate=False):
    counts = dict(cudaq.sample(kernel, *kernel_args, shots_count=shots).items())
    return _product_stat(counts, group_a=group_a, group_b=group_b, pcrit=pcrit, negate=negate)


# --- State vector helpers (cudaq.get_state) ---

def assert_classical_sv(kernel, *kernel_args, target_qubits=None, tolerance=1e-5, expval=None, negate=False):
    sv = np.array(cudaq.get_state(kernel, *kernel_args))
    return _classical_sv(sv, target_qubits=target_qubits, tolerance=tolerance, expval=expval, negate=negate)


def assert_uniform_sv(kernel, *kernel_args, target_qubits=None, tolerance=1e-5, negate=False):
    sv = np.array(cudaq.get_state(kernel, *kernel_args))
    return _uniform_sv(sv, target_qubits=target_qubits, tolerance=tolerance, negate=negate)


def assert_product_sv(kernel, *kernel_args, group_a, group_b, tolerance=1e-5, negate=False):
    sv = np.array(cudaq.get_state(kernel, *kernel_args))
    return _product_sv(sv, group_a=group_a, group_b=group_b, tolerance=tolerance, negate=negate)


# --- Expectation value helpers (cudaq.observe) ---

def assert_expectation(kernel, spin_op, expected, *kernel_args, tolerance=1e-5):
    result = cudaq.observe(kernel, spin_op, *kernel_args)
    val = result.expectation()
    passed = bool(abs(val - expected) <= tolerance)
    return float(val), passed
