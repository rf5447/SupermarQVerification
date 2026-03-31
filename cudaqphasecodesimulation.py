import cudaq

def noisy_cudaq_simulation(kernel, num_data_qubits: int, num_rounds: int, phase_state: list[int], p: float, shots=1000):
    """
    Apply depolarizing noise with probability p after every gate.
    Matches Cirq's cirq.depolarize(p=p).
    """

    # CUDA-Q built-in depolarization channel (probability p)
    depol = cudaq.DepolarizationChannel(p)

    # Create a noise model and attach depolarization to every gate type you use
    noise = cudaq.NoiseModel()

    # Attach noise to all gates you expect in your benchmarks
    for gate in ["x", "y", "z", "h", "rz"]:
        noise.add_all_qubit_channel(gate, depol)
    #print("got here")
    # Run the kernel with noise using supplied parameters
    return cudaq.sample(kernel, num_data_qubits, num_rounds, phase_state, noise_model=noise, shots_count=shots)


def cudaq_simulation(kernel, num_data_qubits: int, num_rounds: int, phase_state: list[int], shots=1000):
    return cudaq.sample(kernel, num_data_qubits, num_rounds, phase_state, shots_count=shots)
