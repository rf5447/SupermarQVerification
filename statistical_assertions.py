import numpy as np
from scipy.stats import chisquare
from scipy.stats import fisher_exact

def _marginalize(counts_dict, keep_indices):

    if not keep_indices:
        raise ValueError("keep_indices must be non-empty")

    num_qubits = len(next(iter(counts_dict)))

    if any(i < 0 or i >= num_qubits for i in keep_indices):
        raise ValueError(f"keep_indices {keep_indices}, out of range for {num_qubits}-qubit register")

    marginal = {}

    for bitstring, count in counts_dict.items():
        sub_key = ''.join(bitstring[i] for i in keep_indices)
        marginal[sub_key] = marginal.get(sub_key, 0) + count
    
    return marginal

def classical_assertion(counts_dict, target_qubits=None, pcrit=None, expval=None, negate=False):
    """
    Performs a chi-squared statistical test on the observed measurement distribution, which
    is obtained by calling cudaq.sample. Internally, it builds an expected distribution that 
    is a table of all 1's exccept for the expected value, which is 2^16. Then, it normalizes 
    the expected and observed distributions, and compares them through the chi-square test.

    Args:
        counts_dict(dict): dictionary mapping bitstrings to counts
        target_qubits(list[int] or None): qubit indices to marginalize over; if None, use
            all qubits
        pcrit(float): critical p-value. If None passed in, defaults to 0.05
        expval(int or string or None): the expected value
            If no expected value specified, then this assertion just checks
            that the measurement outcomes are in any classical state.
        negate(bool): True if assertion passed is negation of statistical test passed

    Returns:
        tuple: tuple containing:

            chisq(float): the chi-squared value

            pval(float): the p-value

            passed(bool): if the test passed
    """
    if pcrit is None:
        pcrit = 0.05
    
    dict_result = dict(counts_dict)

    if target_qubits is None:
        target_qubits = list(range(len(next(iter(dict_result)))))

    dict_result = _marginalize(dict_result, target_qubits)

    vals_list = list(dict_result.values()) 

    if expval is None:
        index = np.argmax(vals_list) 
    else:
        try:
            index = list(map(lambda x: int (x,2), dict_result.keys())).index(expval)
        except ValueError:
            index = -1

    num_qubits = len(str(list(dict_result)[0])) 

    num_zeros = 2 ** num_qubits - len(dict_result) 
    vals_list.extend([0] * num_zeros) 

    exp_list = [1] * len(vals_list) 
    exp_list[index] = 2 ** 16 

    vals_list = vals_list / np.sum(vals_list)
    exp_list = exp_list / np.sum(exp_list)

    chisq, pval = chisquare(vals_list, f_exp=exp_list, ddof=1) 

    if len(str(list(dict_result.keys())[0])) == 1: 
        pval = vals_list[index] 
        passed = bool(pval >= 1 - pcrit) 
    else:
        passed = bool(pval >= pcrit) 
    
    if negate:
        passed = not passed
    else:
        passed = passed

    return (chisq, pval, passed)
    
def uniform_assertion(counts_dict, target_qubits=None, pcrit=None, negate=False):
    """
    Performs a chi-squared statistical test on the observed measurement 
    distribution, which is obtained by calling cudaq.sample. Internally, compares
    a normalized table of experimental counts to the scipy.stats.chisquare default, for which
    all outcomes are equally likely.

    Args:
        counts_dict(dict): dictionary mapping bitstrings to counts
        target_qubits(list[int] or None): qubit indices to marginalize over; if None, use
            all qubits
        pcrit(float): critical p-value. If None passed in, defaults to 0.05
        negate(bool): True if assertion passed is negation of statistical test passed

    Returns:
        tuple: tuple containing:

            chisq(float): the chi-square value

            pval(float): the p-value

            passed(Boolean): if the test passed
    """
    if pcrit is None:
        pcrit = 0.05

    dict_result = dict(counts_dict)
    
    if target_qubits is None:
        target_qubits = list(range(len(next(iter(dict_result)))))

    dict_result = _marginalize(dict_result, target_qubits)

    vals_list = list(dict_result.values()) 

    num_qubits = len(str(list(dict_result)[0])) 

    num_zeros = 2 ** num_qubits - len(dict_result) 

    vals_list.extend([0] * num_zeros) 

    chisq, pval = chisquare(vals_list)
    if negate:
        passed = not bool(pval >= pcrit)
    else:
        passed = bool(pval >= pcrit)

    return (chisq, pval, passed)

def product_assertion(counts_dict, group_a, group_b, pcrit=None, negate=False):
    """
    Performs a statistical independence test on the observed measurement 
    distribution, which is obtained by calling cudaq.sample.
    Internally, constructs a contingency table from the observed measurement
    distribution counts, then feeds it into scipy.stats.fisher_exact (which
    uses Fisher's exact test for 2x2 tables, and Monte Carlo for larger ones).

    Args:
        counts_dict(dict): dictionary mapping bitstrings to counts
        group_a(list[int]): qubit indices for group A
        group_b(list[int]): qubit indices for group B
        shots_count(int): number of shots taken for measurement distribution. If None, defaults to 1000.
        negate(bool): True if assertion passed is negation of statistical test passed

    Returns:
        tuple: tuple containing:

            odds_ratio(float): the odds ratio, returned from scipy.stats.fisher_exact

            pval(float): the p-value

            passed(bool): if the test passed
    """
    if not group_a or not group_b:
        raise Exception("Both group_a and group_b must be non-empty")
    
    if set(group_a) & set(group_b):
        raise Exception("Groups must not overlap, but got shared indices")
    
    if pcrit is None:
        pcrit = 0.05

    dict_result = dict(counts_dict)

    if (len(list(dict_result.keys())[0]) == 1):
        raise Exception("Only 1 qubit -- product state assertion requires at least 2 qubits")
    
    num_qubits = len(next(iter(dict_result)))

    if any(i < 0 or i >= num_qubits for i in group_a + group_b):
        raise Exception(f"Indices out of range for {num_qubits}-qubit register")
    
    all_indices = group_a + group_b

    if len(all_indices) != num_qubits or set(all_indices) != set(range(num_qubits)):
        dict_result = _marginalize(dict_result, all_indices)

    q0len = len(group_a)
    q1len = len(group_b)
    
    cont_table = np.zeros((2 ** q0len, 2 ** q1len))

    for (key, value) in dict_result.items():
        q0index = int(key[:q0len], 2) 
        q1index = int(key[q0len:], 2)
        cont_table[q0index][q1index] = value 

    odds_ratio, pval = fisher_exact(cont_table) 

    if negate:
        passed = not bool(pval >= pcrit) 
    else:
        passed = bool(pval >= pcrit)

    return (odds_ratio, pval, passed)
