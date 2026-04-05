import cudaq
from assertions_helpers import (
    assert_classical, assert_product,
    assert_classical_sv, assert_product_sv,
)

# Subkernels

@cudaq.kernel
def bitcode_init(q: cudaq.qview, num_data_qubits: int, bit_state: list[int]):
    for i in range(num_data_qubits):
        if bit_state[i] == 1:
            x(q[2 * i])

@cudaq.kernel
def syndrome_round(q: cudaq.qview, num_data_qubits: int):
    num_ancillas = num_data_qubits - 1
    for i in range(num_ancillas):
        idx = 2 * i + 1
        x.ctrl(q[idx - 1], q[idx])
        x.ctrl(q[idx + 1], q[idx])
        result = mz(q[idx])
        if result:
            x(q[idx])

# Wrappers for subkernels

@cudaq.kernel
def assert_init(num_data_qubits: int, bit_state: list[int]) -> list[bool]:
    q = cudaq.qvector(2 * num_data_qubits - 1)
    bitcode_init(q, num_data_qubits, bit_state)
    return [mz(qi) for qi in q]

@cudaq.kernel
def assert_one_round(num_data_qubits: int, bit_state: list[int]) -> list[bool]:
    q = cudaq.qvector(2 * num_data_qubits - 1)
    bitcode_init(q, num_data_qubits, bit_state)
    syndrome_round(q, num_data_qubits)
    return [mz(qi) for qi in q]

@cudaq.kernel
def assert_all_rounds(num_data_qubits: int, num_rounds: int, bit_state: list[int]) -> list[bool]:
    q = cudaq.qvector(2 * num_data_qubits - 1)
    bitcode_init(q, num_data_qubits, bit_state)
    for _ in range(num_rounds):
        syndrome_round(q, num_data_qubits)
    return [mz(qi) for qi in q]

# Full kernel

@cudaq.kernel
def bitcode(num_data_qubits: int, num_rounds: int, bit_state: list[int]):
    q = cudaq.qvector(2 * num_data_qubits - 1)
    bitcode_init(q, num_data_qubits, bit_state)
    for _ in range(num_rounds):
        syndrome_round(q, num_data_qubits)
    for i in range(2 * num_data_qubits - 1):
        mz(q[i])

# Assertions

num_data_qubits = 3
num_rounds = 3
bit_state = [1, 0, 1]
num_qubits = 2 * num_data_qubits - 1

data_indices = list(range(0, num_qubits, 2))     # [0, 2, 4]
ancilla_indices = list(range(1, num_qubits, 2))   # [1, 3]

print("After init")
for i, d in enumerate(data_indices):
    print(f"data q[{d}] = {bit_state[i]} (stat):",
          assert_classical(assert_init, num_data_qubits, bit_state,
                           target_qubits=[d], expval=bit_state[i]))
    print(f"data q[{d}] = {bit_state[i]} (sv):  ",
          assert_classical_sv(assert_init, num_data_qubits, bit_state,
                              target_qubits=[d], expval=bit_state[i]))
for a in ancilla_indices:
    print(f"ancilla q[{a}] = 0 (stat):",
          assert_classical(assert_init, num_data_qubits, bit_state,
                           target_qubits=[a], expval=0))
    print(f"ancilla q[{a}] = 0 (sv):  ",
          assert_classical_sv(assert_init, num_data_qubits, bit_state,
                              target_qubits=[a], expval=0))
print("product (data|ancilla) (stat):",
      assert_product(assert_init, num_data_qubits, bit_state,
                     group_a=data_indices, group_b=ancilla_indices))
print("product (data|ancilla) (sv):  ",
      assert_product_sv(assert_init, num_data_qubits, bit_state,
                        group_a=data_indices, group_b=ancilla_indices))

print("\nAfter one round")
for i, d in enumerate(data_indices):
    print(f"data q[{d}] = {bit_state[i]} (stat):",
          assert_classical(assert_one_round, num_data_qubits, bit_state,
                           target_qubits=[d], expval=bit_state[i]))
    print(f"data q[{d}] = {bit_state[i]} (sv):  ",
          assert_classical_sv(assert_one_round, num_data_qubits, bit_state,
                              target_qubits=[d], expval=bit_state[i]))
for a in ancilla_indices:
    print(f"ancilla q[{a}] reset to 0 (stat):",
          assert_classical(assert_one_round, num_data_qubits, bit_state,
                           target_qubits=[a], expval=0))
    print(f"ancilla q[{a}] reset to 0 (sv):  ",
          assert_classical_sv(assert_one_round, num_data_qubits, bit_state,
                              target_qubits=[a], expval=0))

print("\nAfter all rounds")
for i, d in enumerate(data_indices):
    print(f"data q[{d}] = {bit_state[i]} (stat):",
          assert_classical(assert_all_rounds, num_data_qubits, num_rounds, bit_state,
                           target_qubits=[d], expval=bit_state[i]))
    print(f"data q[{d}] = {bit_state[i]} (sv):  ",
          assert_classical_sv(assert_all_rounds, num_data_qubits, num_rounds, bit_state,
                              target_qubits=[d], expval=bit_state[i]))
print("product (data|ancilla) (stat):",
      assert_product(assert_all_rounds, num_data_qubits, num_rounds, bit_state,
                     group_a=data_indices, group_b=ancilla_indices))
print("product (data|ancilla) (sv):  ",
      assert_product_sv(assert_all_rounds, num_data_qubits, num_rounds, bit_state,
                        group_a=data_indices, group_b=ancilla_indices))
