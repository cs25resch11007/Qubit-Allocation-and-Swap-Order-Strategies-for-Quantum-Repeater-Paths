import math
import random
import time
import functools
import bisect
from collections import deque
from contextlib import contextmanager
from math import comb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom
import concurrent.futures
from datetime import datetime

# ===========================================================================
# GLOBAL SIMULATION SWITCHES
# ===========================================================================

# Set to True for O(Q^2) Vectorized Engine | False for O(Q^3) Traditional Loop Engine
USE_VECTORIZED_CONV = True

# ===========================================================================
# TIME EXECUTION MONITOR (CONTEXT MANAGER)
# ===========================================================================
name_ourAlgo = 'Our Algo'
name_FullDP = 'DP'
name_BaselineFixed = 'Baseline(CBT, EA)'

name_GredL_CRIA = '(GSO, L-CRIA)'
name_GreedySwapEqualAlloc = '(GSO, EA)'
name_GreedyJointOrderAlloc = 'GJOA'
name_ProxOrdL_CRIA = '(PSO, L-CRIA)'
name_ProxOrdEA = '(PSO, EA)'
name_ProxGreedyAlloc = 'POGA'




def clear_memory_caches():

    _cached_comb.cache_clear()
    fast_binomial_pmf.cache_clear()
    binomial_dist.cache_clear()
    _bsm_term.cache_clear()

    get_bsm_transition_matrix.cache_clear()

    bsm_conv_traditional.cache_clear()
    bsm_conv_vectorized.cache_clear()

    ev.cache_clear()

    import gc
    gc.collect()



@contextmanager
def monitor_time(section_name):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        elapsed = end - start
        
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        
        print(f"[{section_name}] Completed in {elapsed:.4f} seconds "
              f"({hours} hrs, {minutes} mins, {seconds:.2f} secs)")

# ===========================================================================
# CORE QUANTUM SIMULATION ENGINE (OPTIMIZED WITH MEMOIZATION)
# ===========================================================================

@functools.lru_cache(maxsize=None)
def _cached_comb(n, k):
    return math.comb(n, k)


@functools.lru_cache(maxsize=None)
def fast_binomial_pmf(n, k, p):
    """
    Unified dispatcher for binomial probability calculations.
    Adapts computation strategy based on parameter scale.
    """
    if n < 30:
        return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    elif n < 100:
        return _cached_comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    else:
        return float(binom.pmf(k, n, p))

@functools.lru_cache(maxsize=None)
def binomial_dist(n, p):
    if n >= 100:
        return tuple(binom.pmf(np.arange(n + 1), n, p))
    else:
        return tuple(fast_binomial_pmf(n, k, p) for k in range(n + 1))

@functools.lru_cache(maxsize=None)
def _bsm_term(m, k, p_bsm):
    return fast_binomial_pmf(m, k, p_bsm)


# ===========================================================================
# CACHED BSM TRANSITION MATRIX GENERATOR
# ===========================================================================

@functools.lru_cache(maxsize=None)
def get_bsm_transition_matrix(size, p_bsm):
    B = np.zeros((size, size), dtype=np.float64)
    for m in range(size):
        B[m, :m + 1] = binom.pmf(np.arange(m + 1), m, p_bsm)
    return B


# ===========================================================================
# BSM CONVOLUTION ENGINES & SWITCH
# ===========================================================================

@functools.lru_cache(maxsize=None)
def bsm_conv_traditional(d_L, d_R, p_bsm):
    """
    Traditional O(Q^3) explicit triple-loop BSM convolution engine.
    """
    size = min(len(d_L), len(d_R))
    out = np.zeros(size)

    for c_L, p_L in enumerate(d_L):
        if p_L == 0.0: continue
        for c_R, p_R in enumerate(d_R):
            if p_R == 0.0: continue
            m = min(c_L, c_R)
            for k in range(m + 1):
                out[k] += p_L * p_R * _bsm_term(m, k, p_bsm)

    return tuple(out)


@functools.lru_cache(maxsize=None)
def bsm_conv_vectorized(d_L, d_R, p_bsm):
    """
    Optimized O(Q^2) vectorized cumulative survival BSM convolution engine.
    """
    dL = np.asarray(d_L, dtype=np.float64)
    dR = np.asarray(d_R, dtype=np.float64)
    
    # 1. Compute cumulative survival functions on FULL distributions FIRST
    S_L = np.cumsum(dL[::-1])[::-1]
    S_R = np.cumsum(dR[::-1])[::-1]
    
    # 2. Determine bottleneck size
    size = min(len(dL), len(dR))
    
    # 3. Slice AFTER computing survival probabilities
    dL_sub = dL[:size]
    dR_sub = dR[:size]
    S_L_sub = S_L[:size]
    S_R_sub = S_R[:size]
    
    # 4. Compute P_m with full probability conservation
    P_m = dL_sub * S_R_sub + dR_sub * S_L_sub - dL_sub * dR_sub
    
    B = get_bsm_transition_matrix(size, float(p_bsm))
    return tuple(P_m @ B)


def bsm_conv(d_L, d_R, p_bsm):
    """
    Dispatcher switch for BSM convolution engine.
    Controlled globally via USE_VECTORIZED_CONV flag.
    """
    if USE_VECTORIZED_CONV:
        return bsm_conv_vectorized(d_L, d_R, p_bsm)
    else:
        return bsm_conv_traditional(d_L, d_R, p_bsm)


@functools.lru_cache(maxsize=None)
def ev(dist):
    return float(np.dot(np.arange(len(dist)), np.array(dist)))

# ===========================================================================
# HELPER FOR ORDER STRINGS
# ===========================================================================

def get_cbt_order_fast(i, j, result_list):
    if j - i == 1:
        return
    mid = (i + j) // 2
    get_cbt_order_fast(i, mid, result_list)
    get_cbt_order_fast(mid, j, result_list)
    result_list.append(mid)


def build_proximity_splits(n, p_gens, p_swaps, alpha=0.0):
    """
    Constructs the tree split dictionary using joint proximity balancing.
    - alpha = 0.0 : Links-only proximity rule
    - alpha > 0.0 : Joint proximity rule weighted by alpha swap costs
    """
    gen_distances = [-math.log(p + 1e-9) for p in p_gens]
    prefix_gen = [0.0] * (n + 1)
    for idx in range(1, n):
        prefix_gen[idx] = prefix_gen[idx - 1] + gen_distances[idx - 1]

    prefix_swap = [0.0] * (n + 1)
    if alpha > 0.0:
        swap_distances = [0.0] * (n + 1)
        for node in range(2, n):
            swap_distances[node] = -alpha * math.log(p_swaps[node] + 1e-9)
        for idx in range(1, n + 1):
            prefix_swap[idx] = prefix_swap[idx - 1] + swap_distances[idx]

    fixed_splits = {}

    def find_splits(i, j):
        if j - i <= 1:
            return
        best_k = i + 1
        min_imbalance = float('inf')

        for k in range(i + 1, j):
            left_dist = (prefix_gen[k - 1] - prefix_gen[i - 1]) + (prefix_swap[k - 1] - prefix_swap[i])
            right_dist = (prefix_gen[j - 1] - prefix_gen[k - 1]) + (prefix_swap[j - 1] - prefix_swap[k])
            imbalance = abs(left_dist - right_dist)
            if imbalance < min_imbalance:
                min_imbalance = imbalance
                best_k = k

        fixed_splits[(i, j)] = best_k
        find_splits(i, best_k)
        find_splits(best_k, j)

    find_splits(1, n)
    return fixed_splits

# ===========================================================================
# SYSTEM PATH ROUTING ENGINES
# ===========================================================================

def run_PSO_EA(n, Q, p_gens, p_swaps, alpha_list=(0.0, 1.0, 2.0)):
    left_alloc = {node: (0 if node==1 else Q if node==n else Q//2) for node in range(1, n+1)}
    right_alloc = {node: (Q if node==1 else 0 if node==n else Q - Q//2) for node in range(1, n+1)}
    leaf_cap = {i: min(right_alloc[i], left_alloc[i + 1]) for i in range(1, n)}
    q_alloc_dict = {i: (left_alloc[i], right_alloc[i]) for i in range(1, n + 1)}

    best_val = -1.0
    best_alpha = 0.0
    best_order = []

    for alpha in alpha_list:
        fixed_splits = build_proximity_splits(n, p_gens, p_swaps, alpha=alpha)
        order_sequence = []

        def seg_dist(i, j):
            if j - i == 1:
                return binomial_dist(leaf_cap[i], p_gens[i - 1])
            y = fixed_splits[(i, j)]
            d_combined = bsm_conv(seg_dist(i, y), seg_dist(y, j), p_swaps[y])
            order_sequence.append(y)
            return d_combined

        val = ev(seg_dist(1, n))
        if val > best_val:
            best_val = val
            best_alpha = alpha
            best_order = order_sequence

    return best_val, f"PSO_EA (alpha={best_alpha}): {tuple(best_order)}", str(q_alloc_dict)


def run_GSO_EA(n, Q, p_gens, p_swaps):
    left_alloc = {node: (0 if node==1 else Q if node==n else Q//2) for node in range(1, n+1)}
    right_alloc = {node: (Q if node==1 else 0 if node==n else Q - Q//2) for node in range(1, n+1)}
    leaf_cap = {i: min(right_alloc[i], left_alloc[i + 1]) for i in range(1, n)}
    
    q_alloc_dict = {i: (left_alloc[i], right_alloc[i]) for i in range(1, n + 1)}

    segments = []
    for i in range(1, n):
        segments.append({
            'nodes': (i, i + 1),
            'dist': binomial_dist(leaf_cap[i], p_gens[i - 1])
        })
        
    order_sequence = []
    for _ in range(n - 2):
        best_score = -1.0
        best_idx = -1
        best_dist = None
        chosen_mid = -1
        
        for si in range(len(segments) - 1):
            mid = segments[si]['nodes'][1]
            p_swap_mid = p_swaps[mid]
            dl = segments[si]['dist']
            dr = segments[si + 1]['dist']
            
            d_combined = bsm_conv(dl, dr, p_swap_mid)
            score = ev(d_combined)
            
            if score > best_score:
                best_score = score
                best_idx = si
                best_dist = d_combined
                chosen_mid = mid
                
        order_sequence.append(chosen_mid)
        merged_nodes = (segments[best_idx]['nodes'][0], segments[best_idx + 1]['nodes'][1])
        segments[best_idx] = {'nodes': merged_nodes, 'dist': best_dist}
        segments.pop(best_idx + 1)
        
    return ev(segments[0]['dist']), f"GSO_EA: {tuple(order_sequence)}", str(q_alloc_dict)




def run_full_dp_dynamic(n, Q, p_gens, p_swaps):
    EV_TOL = 1e-12
    dp = [[None] * (n + 1) for _ in range(n + 1)]

    for i in range(1, n):
        j = i + 1
        p_gen = p_gens[i - 1]
        base_dists = [binomial_dist(cap, p_gen) for cap in range(Q + 1)]
        table = [[(base_dists[min(qL, qR)], []) for qR in range(Q + 1)] for qL in range(Q + 1)]
        dp[i][j] = table

    for length in range(2, n):
        for i in range(1, n - length + 1):
            j = i + length
            table = [[None] * (Q + 1) for _ in range(Q + 1)]

            for qL in range(Q + 1):
                for qR in range(Q + 1):
                    best_ev = -1.0
                    best_dist = None
                    ties = []

                    for y in range(i + 1, j):
                        p_swap_y = p_swaps[y]
                        
                        for q_yL in range(Q + 1):
                            q_yR = Q - q_yL

                            d_L = dp[i][y][qL][q_yL][0]
                            d_R = dp[y][j][q_yR][qR][0]

                            d_combined = bsm_conv(d_L, d_R, p_swap_y)
                            e_val = ev(d_combined)

                            if e_val > best_ev + EV_TOL:
                                best_ev = e_val
                                best_dist = d_combined
                                ties = [(y, q_yL)]
                                
                            elif abs(e_val - best_ev) <= EV_TOL:
                                if e_val > best_ev:
                                    best_ev = e_val
                                    best_dist = d_combined
                                ties.append((y, q_yL))

                    table[qL][qR] = (best_dist, ties)

            dp[i][j] = table

    q_alloc_dict = {i: (0, 0) for i in range(1, n + 1)}
    q_alloc_dict[1] = (0, Q)
    q_alloc_dict[n] = (Q, 0)
    order_sequence = []

    def traceback(i, j, qL, qR):
        if j - i == 1:
            return
        y, q_yL = dp[i][j][qL][qR][1][0]
        q_yR = Q - q_yL
        q_alloc_dict[y] = (q_yL, q_yR)
        traceback(i, y, qL, q_yL)
        traceback(y, j, q_yR, qR)
        order_sequence.append(y)

    traceback(1, n, Q, Q)
    return ev(dp[1][n][Q][Q][0]), f"DP: {tuple(order_sequence)}", str(q_alloc_dict)





def run_PSO_LCRIA(n, Q, p_gens, p_swaps, alpha_list=(0.0, 1.0, 2.0)):
    q_alloc = {i: {'L': 0, 'R': 0} for i in range(1, n + 1)}
    q_alloc[1] = {'L': 0, 'R': Q}
    q_alloc[n] = {'L': Q, 'R': 0}

    for j in range(2, n):
        p_left = p_gens[j - 2]
        p_right = p_gens[j - 1]
        denom = p_left + p_right
        qL = round(Q * p_right / denom) if denom > 0 else Q // 2
        qR = Q - qL
        q_alloc[j] = {'L': qL, 'R': qR}

    for j in range(2, n):
        max_left_cap = q_alloc[j - 1]['R']
        if q_alloc[j]['L'] > max_left_cap:
            wasted_qubits = q_alloc[j]['L'] - max_left_cap
            q_alloc[j]['L'] = max_left_cap
            q_alloc[j]['R'] += wasted_qubits

    for j in range(n - 1, 1, -1):
        max_right_cap = q_alloc[j + 1]['L']
        if q_alloc[j]['R'] > max_right_cap:
            wasted_qubits = q_alloc[j]['R'] - max_right_cap
            q_alloc[j]['R'] = max_right_cap
            q_alloc[j]['L'] += wasted_qubits

    q_alloc_dict = {i: (q_alloc[i]['L'], q_alloc[i]['R']) for i in range(1, n + 1)}
    leaf_cap = {i: min(q_alloc[i]['R'], q_alloc[i + 1]['L']) for i in range(1, n)}

    best_val = -1.0
    best_alpha = 0.0
    best_order = []

    for alpha in alpha_list:
        fixed_splits = build_proximity_splits(n, p_gens, p_swaps, alpha=alpha)
        order_sequence = []
        dist_memo = {}

        def seg_dist(i, j):
            if (i, j) in dist_memo: return dist_memo[(i, j)]
            if j - i == 1:
                return binomial_dist(leaf_cap[i], p_gens[i - 1])
            y = fixed_splits[(i, j)]
            d_combined = bsm_conv(seg_dist(i, y), seg_dist(y, j), p_swaps[y])
            order_sequence.append(y)
            return d_combined

        val = ev(seg_dist(1, n))
        if val > best_val:
            best_val = val
            best_alpha = alpha
            best_order = order_sequence

    return best_val, f"PSO_LCRIA (alpha={best_alpha}): {tuple(best_order)}", str(q_alloc_dict)


# ===========================================================================
# ONE-SHOT DISTANCE-2 BOUNDARY PROPAGATION HELPER FUNCTIONS
# ===========================================================================

def _oneshot_set_U(bounds, node, side, value, n):
    if node is None or node < 1 or node > n: return
    if node == 1 and side == 'l': return
    if node == n and side == 'r': return
    L, oldU = bounds[node][side]
    if L == oldU: return
    newU = min(oldU, value)
    if newU != oldU: bounds[node][side][1] = newU

def _oneshot_pin(bounds, node, side, value, n):
    if node is None or node < 1 or node > n: return
    if node == 1 and side == 'l': return
    if node == n and side == 'r': return
    oldL, oldU = bounds[node][side]
    if oldL == oldU: return
    newL, newU = max(oldL, value), min(oldU, value)
    if newL != oldL or newU != oldU: bounds[node][side] = [newL, newU]

def _oneshot_propagate_left(bounds, frozen, active_before, i, qL, Q, n):
    left_actives = sorted([a for a in active_before if a < i], reverse=True)
    im1 = left_actives[0] if len(left_actives) >= 1 else None
    im2 = left_actives[1] if len(left_actives) >= 2 else None
    im3 = left_actives[2] if len(left_actives) >= 3 else None
    
    if im1 is not None and im1 == i - 1:
        if im2 is not None and im2 == i - 2:
            if im3 is not None and im3 == i - 3:
                _oneshot_set_U(bounds, im2, 'r', Q - qL, n)
                _oneshot_set_U(bounds, im1, 'l', Q - qL, n)
                _oneshot_pin(bounds, im1, 'r', qL, n)
            elif im3 is not None and im3 < i - 3:
                q_star_im3_r = frozen[i - 3][1]
                _oneshot_set_U(bounds, im2, 'r', min(Q - q_star_im3_r, Q - qL), n)
                _oneshot_set_U(bounds, im1, 'l', min(Q - q_star_im3_r, Q - qL), n)
                _oneshot_pin(bounds, im1, 'r', qL, n)
            else:
                _oneshot_set_U(bounds, im1, 'l', Q - qL, n)
                _oneshot_pin(bounds, im1, 'r', qL, n)
        elif im2 is not None and im2 < i - 2:
            _oneshot_pin(bounds, im1, 'r', qL, n)

def _oneshot_propagate_right(bounds, frozen, active_before, i, qR, Q, n):
    right_actives = sorted([a for a in active_before if a > i])
    ip1 = right_actives[0] if len(right_actives) >= 1 else None
    ip2 = right_actives[1] if len(right_actives) >= 2 else None
    ip3 = right_actives[2] if len(right_actives) >= 3 else None
    
    if ip1 is not None and ip1 == i + 1:
        if ip2 is not None and ip2 == i + 2:
            if ip3 is not None and ip3 == i + 3:
                _oneshot_set_U(bounds, ip2, 'l', Q - qR, n)
                _oneshot_set_U(bounds, ip1, 'r', Q - qR, n)
                _oneshot_pin(bounds, ip1, 'l', qR, n)
            elif ip3 is not None and ip3 > i + 3:
                q_star_ip3_l = frozen[i + 3][0]
                _oneshot_set_U(bounds, ip2, 'l', min(Q - q_star_ip3_l, Q - qR), n)
                _oneshot_set_U(bounds, ip1, 'r', min(Q - q_star_ip3_l, Q - qR), n)
                _oneshot_pin(bounds, ip1, 'l', qR, n)
            else:
                _oneshot_set_U(bounds, ip1, 'r', Q - qR, n)
                _oneshot_pin(bounds, ip1, 'l', qR, n)
        elif ip2 is not None and ip2 > i + 2:
            _oneshot_pin(bounds, ip1, 'l', qR, n)


def run_GSO_LCRIA(n, Q, p_gens, p_swaps, alpha_list=(0.0, 1.0, 2.0)):
    q_alloc = {i: {'L': 0, 'R': 0} for i in range(1, n + 1)}
    q_alloc[1] = {'L': 0, 'R': Q}
    q_alloc[n] = {'L': Q, 'R': 0}

    for j in range(2, n):
        p_left = p_gens[j - 2]
        p_right = p_gens[j - 1]
        denom = p_left + p_right
        qL = round(Q * p_right / denom) if denom > 0 else Q // 2
        qR = Q - qL
        q_alloc[j] = {'L': qL, 'R': qR}

    for j in range(2, n):
        max_left_cap = q_alloc[j - 1]['R']
        if q_alloc[j]['L'] > max_left_cap:
            wasted_qubits = q_alloc[j]['L'] - max_left_cap
            q_alloc[j]['L'] = max_left_cap
            q_alloc[j]['R'] += wasted_qubits

    for j in range(n - 1, 1, -1):
        max_right_cap = q_alloc[j + 1]['L']
        if q_alloc[j]['R'] > max_right_cap:
            wasted_qubits = q_alloc[j]['R'] - max_right_cap
            q_alloc[j]['R'] = max_right_cap
            q_alloc[j]['L'] += wasted_qubits

    q_alloc_dict = {i: (q_alloc[i]['L'], q_alloc[i]['R']) for i in range(1, n + 1)}
    leaf_cap = {i: min(q_alloc[i]['R'], q_alloc[i + 1]['L']) for i in range(1, n)}

    segments = []
    for i in range(1, n):
        segments.append({
            'nodes': (i, i + 1),
            'dist': binomial_dist(leaf_cap[i], p_gens[i - 1])
        })
        
    order_sequence = []
    for _ in range(n - 2):
        best_score = -1.0
        best_idx = -1
        best_dist = None
        chosen_mid = -1
        
        for si in range(len(segments) - 1):
            mid = segments[si]['nodes'][1]
            p_swap_mid = p_swaps[mid]
            dl = segments[si]['dist']
            dr = segments[si + 1]['dist']
            
            d_combined = bsm_conv(dl, dr, p_swap_mid)
            score = ev(d_combined)
            
            if score > best_score:
                best_score = score
                best_idx = si
                best_dist = d_combined
                chosen_mid = mid
                
        order_sequence.append(chosen_mid)
        merged_nodes = (segments[best_idx]['nodes'][0], segments[best_idx + 1]['nodes'][1])
        segments[best_idx] = {'nodes': merged_nodes, 'dist': best_dist}
        segments.pop(best_idx + 1)
        
    return ev(segments[0]['dist']), f"GSO_LCRIA: {tuple(order_sequence)}", str(q_alloc_dict)

def run_PSO_GA(n, Q, p_gens, p_swaps, alpha_list=(0.0, 1.0, 2.0)):
    best_val = -1.0
    best_alpha = 0.0
    best_order = []
    best_q_alloc_dict = {}

    for alpha in alpha_list:
        fixed_splits = build_proximity_splits(n, p_gens, p_swaps, alpha=alpha)
        order_sequence = []

        def traverse(i, j):
            if j - i <= 1: return
            k = fixed_splits[(i, j)]
            traverse(i, k)
            traverse(k, j)
            order_sequence.append(k)
        traverse(1, n)

        bounds = {i: {'l': [0, Q], 'r': [0, Q]} for i in range(1, n + 1)}
        frozen = {}
        active = list(range(1, n + 1))
        
        segments = []
        for i in range(1, n):
            segments.append({
                'nodes': (i, i + 1),
                'dist': None,
                'p_gen': p_gens[i - 1]
            })

        for sel in order_sequence:
            si = next(idx for idx, seg in enumerate(segments) if seg['nodes'][1] == sel)
            seg_left = segments[si]
            seg_right = segments[si + 1]
            p_bsm_mid = p_swaps[sel]
            
            Ll, Ul = bounds[sel]['l']
            Lr, Ur = bounds[sel]['r']
            
            left_is_leaf = seg_left['p_gen'] is not None
            right_is_leaf = seg_right['p_gen'] is not None
            
            best_e, best_qL, best_qR, best_dl, best_dr = -1.0, 0, 0, None, None
            
            if left_is_leaf and right_is_leaf:
                for qL in range(Ll, Ul + 1):
                    dl = binomial_dist(qL, seg_left['p_gen'])
                    for qR in range(Lr, min(Ur, Q - qL) + 1):
                        dr = binomial_dist(qR, seg_right['p_gen'])
                        e = ev(bsm_conv(dl, dr, p_bsm_mid))
                        if e > best_e:
                            best_e, best_qL, best_qR, best_dl, best_dr = e, qL, qR, dl, dr
                            
            elif left_is_leaf and not right_is_leaf:
                qR = Lr
                dr = seg_right['dist']
                for qL in range(Ll, min(Ul, Q - qR) + 1):
                    dl = binomial_dist(qL, seg_left['p_gen'])
                    e = ev(bsm_conv(dl, dr, p_bsm_mid))
                    if e > best_e:
                        best_e, best_qL, best_qR, best_dl, best_dr = e, qL, qR, dl, dr
                        
            elif not left_is_leaf and right_is_leaf:
                qL = Ll
                dl = seg_left['dist']
                for qR in range(Lr, min(Ur, Q - qL) + 1):
                    dr = binomial_dist(qR, seg_right['p_gen'])
                    e = ev(bsm_conv(dl, dr, p_bsm_mid))
                    if e > best_e:
                        best_e, best_qL, best_qR, best_dl, best_dr = e, qL, qR, dl, dr
                        
            else:
                qL = Ll
                qR = Lr
                if qL + qR <= Q:
                    dl = seg_left['dist']
                    dr = seg_right['dist']
                    best_e = ev(bsm_conv(dl, dr, p_bsm_mid))
                    best_qL, best_qR, best_dl, best_dr = qL, qR, dl, dr

            frozen[sel] = (best_qL, best_qR)
            _oneshot_pin(bounds, sel, 'l', best_qL, n)
            _oneshot_pin(bounds, sel, 'r', best_qR, n)
            _oneshot_propagate_left(bounds, frozen, active, sel, best_qL, Q, n)
            _oneshot_propagate_right(bounds, frozen, active, sel, best_qR, Q, n)
            active.remove(sel)
            
            merged_dist = bsm_conv(best_dl, best_dr, p_bsm_mid)
            merged_nodes = (seg_left['nodes'][0], seg_right['nodes'][1])
            segments[si] = {'nodes': merged_nodes, 'dist': merged_dist, 'p_gen': None}
            segments.pop(si + 1)
            
        q_alloc_dict = {}
        for node_idx in range(1, n + 1):
            if node_idx == 1:
                q_alloc_dict[1] = (0, Q)
            elif node_idx == n:
                q_alloc_dict[n] = (Q, 0)
            else:
                q_alloc_dict[node_idx] = frozen.get(node_idx, (bounds[node_idx]['l'][0], bounds[node_idx]['r'][0]))

        val = ev(segments[0]['dist'])
        if val > best_val:
            best_val = val
            best_alpha = alpha
            best_order = order_sequence
            best_q_alloc_dict = q_alloc_dict

    return best_val, f"PSO_GA (alpha={best_alpha}): {tuple(best_order)}", str(best_q_alloc_dict)



# ===========================================================================
# BUNDLED & FIXED BASELINE ALGORITHMS
# ===========================================================================

def run_our_algo(n, Q, p_gens, p_swaps):
    res_pso_ea = run_PSO_EA(n, Q, p_gens, p_swaps)
    res_gso_ea = run_GSO_EA(n, Q, p_gens, p_swaps)
    res_gso_lcria = run_GSO_LCRIA(n, Q, p_gens, p_swaps)
    res_pso_ga = run_PSO_GA(n, Q, p_gens, p_swaps)
    res_pso_lcria = run_PSO_LCRIA(n, Q, p_gens, p_swaps)
    
    candidates = [
        res_pso_ea, 
        res_gso_ea, 
        res_gso_lcria, 
        res_pso_ga,
        res_pso_lcria,
    ]
    
    best_outcome = max(candidates, key=lambda x: x[0])
    return best_outcome[0], f"Selected Heuristic -> {best_outcome[1]}", best_outcome[2]


def run_baseline_both_fixed(n, Q, p_gens, p_swaps):
    left_alloc = {node: (0 if node==1 else Q if node==n else Q//2) for node in range(1, n+1)}
    right_alloc = {node: (Q if node==1 else 0 if node==n else Q - Q//2) for node in range(1, n+1)}
    leaf_cap = {i: min(right_alloc[i], left_alloc[i + 1]) for i in range(1, n)}
    
    q_alloc_dict = {i: (left_alloc[i], right_alloc[i]) for i in range(1, n + 1)}

    def seg_dist(i, j):
        if j - i == 1:
            d = binomial_dist(leaf_cap[i], p_gens[i - 1])
            return d
        
        y = (i + j) // 2
        d_combined = bsm_conv(seg_dist(i, y), seg_dist(y, j), p_swaps[y])
        return d_combined

    cbt_list = []
    get_cbt_order_fast(1, n, cbt_list)
    cbt_order = tuple(cbt_list)

    return ev(seg_dist(1, n)), f"Fixed CBT Order: {cbt_order}", str(q_alloc_dict)



# ===========================================================================
# MULTI-CORE DATA BATCH GENERATION PIPELINE
# ===========================================================================

def generate_data(n, Q, num_samples, p_gen_range, p_swap_range, enabled_algos, regime_name, max_workers=30):
    base_seed = random.randint(0, 1_000_000)
    
    tasks = [
        (i, n, Q, p_gen_range, p_swap_range, enabled_algos, regime_name, base_seed)
        for i in range(1, num_samples + 1)
    ]
    
    rows = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_process_single_sample, tasks)
        for sample_rows in results:
            rows.extend(sample_rows)
    # print(rows)
    return pd.DataFrame(rows)

# ==========================================
# 1. SAMPLE PROCESSOR WORKER FUNCTION
# ==========================================
def generate_random_bottleneck_instance(
    n,
    bottleneck_fraction=0.30,
    bottleneck_p_gen_range=(0.05, 0.25),
    bottleneck_p_bsm_range=(0.79, 0.95),
    normal_p_gen_range=(0.90, 0.99),
    normal_p_bsm_range=(0.90, 0.99)
):
    """
    Generate a Random Bottleneck network.

    Approximately bottleneck_fraction of the internal nodes
    are selected as bottleneck nodes.

    Constraint:
        No two consecutive p_gens can be bottleneck links.

    Bottleneck nodes:
        p_bsm -> low range

    Links adjacent to bottleneck nodes:
        ONLY the LEFT link gets low p_gen.

    Everything else:
        p_gen, p_bsm -> normal range
    """

    # ------------------------------------------------------------
    # Internal nodes only have BSM probabilities.
    # Nodes 1 and n are endpoints.
    # ------------------------------------------------------------

    internal_nodes = list(range(2, n))
    num_internal = len(internal_nodes)

    num_bottlenecks = max(
        1,
        round(num_internal * bottleneck_fraction)
    )

    # ------------------------------------------------------------
    # Maximum number of non-consecutive bottleneck nodes is:
    #
    # ceil(num_internal / 2)
    #
    # For example:
    #   5 internal nodes -> max 3 bottlenecks
    #   6 internal nodes -> max 3 bottlenecks
    # ------------------------------------------------------------

    max_bottlenecks = (num_internal + 1) // 2

    if num_bottlenecks > max_bottlenecks:
        raise ValueError(
            f"Cannot select {num_bottlenecks} non-consecutive "
            f"bottleneck nodes from {num_internal} internal nodes. "
            f"Maximum possible is {max_bottlenecks}."
        )

    # ------------------------------------------------------------
    # Select bottleneck nodes with NO consecutive nodes.
    # ------------------------------------------------------------

    shuffled_nodes = internal_nodes.copy()
    random.shuffle(shuffled_nodes)

    bottleneck_nodes = []

    for node in shuffled_nodes:

        # Reject this node if it is consecutive to an
        # already selected bottleneck node.
        if (
            node - 1 not in bottleneck_nodes
            and node + 1 not in bottleneck_nodes
        ):
            bottleneck_nodes.append(node)

        if len(bottleneck_nodes) == num_bottlenecks:
            break

    bottleneck_nodes = set(bottleneck_nodes)

    # ------------------------------------------------------------
    # Initially generate everything in the normal
    # mid-to-high regime.
    # ------------------------------------------------------------

    p_gens = [
        round(
            random.uniform(*normal_p_gen_range),
            4
        )
        for _ in range(n - 1)
    ]

    p_swaps = {
        node: round(
            random.uniform(*normal_p_bsm_range),
            4
        )
        for node in internal_nodes
    }

    # ------------------------------------------------------------
    # Make selected nodes bottlenecks.
    # ------------------------------------------------------------

    for node in bottleneck_nodes:

        # Low BSM probability at bottleneck node
        p_swaps[node] = round(
            random.uniform(*bottleneck_p_bsm_range),
            4
        )

        # --------------------------------------------------------
        # ONLY the LEFT link becomes a bottleneck.
        #
        # Node i's left link is L(i-1), which has zero-based
        # Python index i-2.
        # --------------------------------------------------------

        left_link = node - 2

        if 0 <= left_link < n - 1:
            p_gens[left_link] = round(
                random.uniform(*bottleneck_p_gen_range),
                4
            )

    return p_gens, p_swaps, sorted(bottleneck_nodes)


def _process_single_sample(args):
    i, n, Q, p_gen_range, p_swap_range, enabled_algos, regime_name, sample_seed = args

    # if i % CACHE_CLEAR_INTERVAL == 0:
    clear_memory_caches()

    if sample_seed is not None:
        random.seed(sample_seed + i)

    bottleneck_nodes = []
    
    if regime_name == "Random Bottleneck 30%":
    

        (
            p_gens,
            p_swaps,
            bottleneck_nodes
        ) = generate_random_bottleneck_instance(
            n=n,
            bottleneck_fraction=0.30,

            # Bottleneck values
            bottleneck_p_gen_range=(0.05, 0.25),
            bottleneck_p_bsm_range=(0.6, 0.95),

            # Normal nodes/links
            normal_p_gen_range=(0.90, 0.99),
            normal_p_bsm_range=(0.90, 0.99)
        )

    else:

        p_gens = [
            round(
                random.uniform(*p_gen_range),
                4
            )
            for _ in range(n - 1)
        ]

        p_swaps = {
            node: round(
                random.uniform(*p_swap_range),
                4
            )
            for node in range(2, n)
        }
    
    p_gens_str = ", ".join([f"L{idx}->{v}" for idx, v in enumerate(p_gens, start=1)])
    p_swaps_str = ", ".join([f"Node{k}->{v}" for k, v in p_swaps.items()])
    
    base_properties = {
    "Regime": regime_name,
    "Test Case": i,
    "Bottleneck Nodes": (
        ", ".join(map(str, bottleneck_nodes))
        if bottleneck_nodes
        else ""
    ),
    "p_gen Chain Configuration": p_gens_str,
    "p_swap Node Configuration": p_swaps_str
    }
    
    sample_rows = []

    # 1. Core Dynamic Programming & Heuristics
    if enabled_algos.get(name_FullDP, True):
        val, order, alloc = run_full_dp_dynamic(n, Q, p_gens, p_swaps)
        row = base_properties.copy()
        row.update({"Algorithm": name_FullDP, "Qubit Allocation": alloc, "Order": order, "Expected Entanglement": val})
        sample_rows.append(row)
        
        
    if enabled_algos.get(name_ourAlgo, True):
        val, order, alloc = run_our_algo(n, Q, p_gens, p_swaps)
        row = base_properties.copy()
        row.update({"Algorithm": name_ourAlgo, "Qubit Allocation": alloc, "Order": order, "Expected Entanglement": val})
        sample_rows.append(row)
        
    if enabled_algos.get(name_BaselineFixed , True):
        val, order, alloc = run_baseline_both_fixed(n, Q, p_gens, p_swaps)
        row = base_properties.copy()
        row.update({"Algorithm": "Baseline (CBT, Equal Alloc)", "Qubit Allocation": alloc, "Order": order, "Expected Entanglement": val})
        sample_rows.append(row)

    # 2. Heuristics & Variants
    if enabled_algos.get(name_ProxOrdEA, False):
        val, order, alloc = run_PSO_EA(n, Q, p_gens, p_swaps)
        row = base_properties.copy()
        row.update({"Algorithm": name_ProxOrdEA, "Qubit Allocation": alloc, "Order": order, "Expected Entanglement": val})
        sample_rows.append(row)

    if enabled_algos.get(name_GreedySwapEqualAlloc, False):
        val, order, alloc = run_GSO_EA(n, Q, p_gens, p_swaps)
        row = base_properties.copy()
        row.update({"Algorithm": name_GreedySwapEqualAlloc, "Qubit Allocation": alloc, "Order": order, "Expected Entanglement": val})
        sample_rows.append(row)


    if enabled_algos.get(name_GredL_CRIA, False):
        
        val, order, alloc = run_GSO_LCRIA(n, Q, p_gens, p_swaps)
        row = base_properties.copy()
        row.update({"Algorithm": name_GredL_CRIA, "Qubit Allocation": alloc, "Order": order, "Expected Entanglement": val})
        sample_rows.append(row)


    if enabled_algos.get(name_ProxGreedyAlloc, False):
        val, order, alloc = run_PSO_GA(n, Q, p_gens, p_swaps)
        row = base_properties.copy()
        row.update({"Algorithm": name_ProxGreedyAlloc, "Qubit Allocation": alloc, "Order": order, "Expected Entanglement": val})
        sample_rows.append(row)


    if enabled_algos.get(name_ProxOrdL_CRIA, False):
        val, order, alloc = run_PSO_LCRIA(n, Q, p_gens, p_swaps)
        row = base_properties.copy()
        row.update({"Algorithm": name_ProxOrdL_CRIA, "Qubit Allocation": alloc, "Order": order, "Expected Entanglement": val})
        sample_rows.append(row)


    return sample_rows


# ===========================================================================
# MAIN EXECUTION AND EXPORT
# ===========================================================================

if __name__ == "__main__":
    # random.seed(42)

    # -----------------------------------------------------------------------
    # CONVOLUTION ENGINE SELECTION
    # True  -> Uses O(Q^2) Vectorized Engine (Fast)
    # False -> Uses O(Q^3) Traditional Engine (Original Loop)
    # -----------------------------------------------------------------------
    USE_VECTORIZED_CONV = True
    
    n_nodes = 5
    Q_qubits = 25
    samples = 100
    ENABLED_ALGOS = {
            name_FullDP: True,
            name_ourAlgo: True,
            name_BaselineFixed : True,

            name_GredL_CRIA:False,
            name_GreedySwapEqualAlloc: False,
            name_ProxOrdL_CRIA: False,
            name_ProxOrdEA: False,  
            name_ProxGreedyAlloc: False,
              
        }

    

    print(f"Running simulation with USE_VECTORIZED_CONV = {USE_VECTORIZED_CONV}")
    
    with monitor_time("Whole Simulation Pipeline"):
        
        with monitor_time("Quantum Simulation Data Generation"):

            print("Generating raw simulation data...")

            # ============================================================
            # 1. WIDE RANDOM
            # ============================================================

            df_wide = generate_data(
                n_nodes,
                Q_qubits,
                samples,
                (0.05, 0.95),
                (0.05, 0.95),
                ENABLED_ALGOS,
                "Wide Random"
            )
            clear_memory_caches()
            # ============================================================
            # 2. HIGH FIDELITY
            # ============================================================

            df_high = generate_data(
                n_nodes,
                Q_qubits,
                samples,
                (0.79, 0.99),
                (0.79, 0.99),
                ENABLED_ALGOS,
                "High Fidelity",
            )
            clear_memory_caches()
            # ============================================================
            # 3. MID-NOISE
            # ============================================================

            df_mid = generate_data(
                n_nodes,
                Q_qubits,
                samples,
                (0.45, 0.65),
                (0.45, 0.65),
                ENABLED_ALGOS,
                "Mid-Noise",
            )
            clear_memory_caches()
            # ============================================================
            # 4. RANDOM BOTTLENECK - 30%
            # ============================================================

            df_bottleneck = generate_data(
                n_nodes,
                Q_qubits,
                samples,
                (0.60, 0.95),   # ignored for bottleneck regime
                (0.60, 0.95),   # ignored for bottleneck regime
                ENABLED_ALGOS,
                "Random Bottleneck 30%",
            )
            clear_memory_caches()
        with monitor_time("De-duplicated Excel Report Generation"):
            excel_file =(
                                f"comprehensive_quantum_results_"
                                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
                        )
            regime_sheets = [
                ("Wide Random", df_wide),
                ("High Fidelity", df_high),
                ("Mid-Noise", df_mid),
                ("Random Bottleneck 30%", df_bottleneck)
            ]
            
            with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                for sheet_title, df_orig in regime_sheets:
                    df_tab = df_orig.copy()
                    
                    duplicate_mask = df_tab.duplicated(subset=["Test Case"], keep="first")
                    df_tab["Test Case"] = df_tab["Test Case"].astype(object)
                    df_tab.loc[duplicate_mask, ["Test Case", "p_gen Chain Configuration", "p_swap Node Configuration"]] = ""
                    
                    if "Regime" in df_tab.columns:
                        df_tab = df_tab.drop(columns=["Regime"])
                    if "Bottleneck Nodes" not in df_tab.columns:
                        df_tab["Bottleneck Nodes"] = ""
                    column_order = [
                                    "Test Case",
                                    "Bottleneck Nodes",
                                    "p_gen Chain Configuration",
                                    "p_swap Node Configuration",
                                    "Algorithm",
                                    "Qubit Allocation",
                                    "Order",
                                    "Expected Entanglement"
                                ]
                    df_tab = df_tab[column_order]
                    df_tab.to_excel(writer, sheet_name=sheet_title, index=False)
                    
            print(f"Excel file parsed and multi-tab structure saved at: '{excel_file}'.")
        
                    
        
         