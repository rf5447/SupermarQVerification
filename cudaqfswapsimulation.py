import cudaq

def noisy_cudaq_simulation(kernel, *kernel_args, p: float, shots=1000):
    """
    Apply depolarizing noise with probability p after every gate.
    Matches Cirq's cirq.depolarize(p=p).
    """

    # CUDA-Q built-in depolarization channel (probability p)
    depol = cudaq.DepolarizationChannel(p)

    # Create a noise model and attach depolarization to every gate type you use
    noise = cudaq.NoiseModel()

    # Attach noise to all gates used in QAOA Vanilla Proxy
    for gate in ["x", "y", "z", "h", "rx", "rz"]:
        noise.add_all_qubit_channel(gate, depol)

    return cudaq.sample(kernel, *kernel_args, noise_model=noise, shots_count=shots)


def cudaq_simulation(kernel, *kernel_args, shots=1000):
    return cudaq.sample(kernel, *kernel_args, shots_count=shots)