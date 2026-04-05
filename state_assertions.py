import numpy as np


def _marginalize_sv(state_vector, target_qubits):
    """
    Compute the marginal probability distribution over target_qubits
    from a full state vector.
    
    Args:
        state_vector(np.ndarray): 1D array of complex amplitudes, length 2^n
        target_qubits(list[int]): qubit indices to keep
        
    Returns:
        np.ndarray: marginal probability distribution, length 2^len(target_qubits)
    """
    num_qubits = int(np.log2(len(state_vector)))
    probs = np.abs(state_vector) ** 2

    prob_tensor = probs.reshape([2] * num_qubits)

    trace_axes = sorted(
        [i for i in range(num_qubits) if i not in target_qubits],
        reverse=True
    )
    for axis in trace_axes:
        prob_tensor = prob_tensor.sum(axis=axis)

    return prob_tensor.flatten()

def _reduced_density_matrix(state_vector, target_qubits):
    state_vector = np.asarray(state_vector, dtype=complex)
    num_qubits = int(np.log2(len(state_vector)))

    tensor = state_vector.reshape([2] * num_qubits)
    remaining = [i for i in range(num_qubits) if i not in target_qubits]
    permutation = list(target_qubits) + remaining
    tensor = tensor.transpose(permutation)

    dim_target = 2 ** len(target_qubits)
    dim_rest = 2 ** (num_qubits - len(target_qubits))
    matrix = tensor.reshape(dim_target, dim_rest)

    return matrix @ matrix.conj().T

def classical_assertion(state_vector, target_qubits=None, tolerance=1e-5, expval=None, negate=False):
    """
    Checks if the state vector represents a purely classical state (a single 
    computational basis state).
    
    Args:
        state_vector(np.ndarray): 1D array of complex amplitudes
        target_qubits(list[int] or None): qubit indices to marginalize over; if None,
            use all 
        tolerance(float): tolerance for probability comparison
        expval(int or str or None): the expected state; if target_qubits is set,
            this indexes into the reduced space.
        negate(bool): inverts the final result
        
    Returns:
        tuple: tuple containing:
            max_prob(float): the highest probability among basis states
            passed(bool): if the test passed
    """
    state_vector = np.asarray(state_vector, dtype=complex)
    num_qubits = int(np.log2(len(state_vector)))

    if target_qubits is None:
        probs = np.abs(state_vector) ** 2
    else:
        if not target_qubits:
            raise ValueError("target_qubits is empty. Provide valid indices, or pass None to evaluate full state.")
        
        if any(i < 0 or i >= num_qubits for i in target_qubits):
            raise IndexError(f"Indices out of range for {num_qubits}-qubit register")
        
        probs = _marginalize_sv(state_vector, target_qubits)

    max_idx = int(np.argmax(probs))
    max_prob = probs[max_idx]

    is_classical = np.isclose(max_prob, 1.0, atol=tolerance)

    if expval is None:
        exp_idx = int(np.argmax(probs))
    else:
        if isinstance(expval, str):
            exp_idx = int(expval, 2)
        else:
            exp_idx = expval

        if exp_idx < 0 or exp_idx >= len(probs):
            raise IndexError(f"Expected state {exp_idx} out of range for {len(probs)} basis states")
        
    prob_to_check = probs[exp_idx]
    is_classical = np.isclose(prob_to_check, 1.0, atol=tolerance)

    if negate:
        passed = not is_classical
    else:
        passed = bool(is_classical)

    return float(max_prob), passed


def uniform_assertion(state_vector, target_qubits=None, tolerance=1e-5, negate=False):
    """
    Checks if the state vector is a perfectly uniform superposition.
    
    Args:
        state_vector(np.ndarray): 1D array of complex amplitudes
        target_qubits(list[int] or None): qubit indices to check. If None,
            checks all qubits.
        tolerance(float): tolerance for probability comparison
        negate(bool): inverts the final result
        
    Returns:
        tuple: tuple containing:
            max_deviation(float): the max deviation of any probability from the 
                expected uniform probability
            passed(bool): if the test passed
    """
    state_vector = np.asarray(state_vector, dtype=complex)
    num_qubits = int(np.log2(len(state_vector)))

    if target_qubits is None:
        probs = np.abs(state_vector) ** 2
    else:
        if not target_qubits:
            raise ValueError("target_qubits is empty. Provide valid indices, or pass None to evaluate full state.")
        
        if any(i < 0 or i >= num_qubits for i in target_qubits):
            raise IndexError(f"Indices out of range for {num_qubits}-qubit register")
        
        probs = _marginalize_sv(state_vector, target_qubits)

    expected_prob = 1.0 / len(probs)

    max_deviation = float(np.max(np.abs(probs - expected_prob)))
    is_uniform = max_deviation <= tolerance

    passed = not is_uniform if negate else bool(is_uniform)
    return max_deviation, passed


def product_assertion(state_vector, group_a, group_b, tolerance=1e-5, negate=False):

    if not group_a or not group_b:
        raise ValueError("Both group_a and group_b must be non-empty")
    if set(group_a) & set(group_b):
        raise ValueError(f"Groups must not overlap, but got indices {sorted(set(group_a) & set(group_b))}")

    state_vector = np.asarray(state_vector, dtype=complex)
    num_qubits = int(np.log2(len(state_vector)))

    if any(i < 0 or i >= num_qubits for i in group_a + group_b):
        raise IndexError(f"Indices out of range for {num_qubits}-qubit register")

    rho = _reduced_density_matrix(state_vector, group_a)
    purity = float(np.real(np.trace(rho @ rho)))

    passed = bool(np.isclose(purity, 1.0, atol=tolerance))
    if negate:
        passed = not passed

    return (purity, passed)
