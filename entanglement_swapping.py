"""Entanglement swapping with a static swapping order, following Section V of
Mai, Amlou, Abane, Battou, "Towards Optimal Orders for Entanglement Swapping in Path Graphs:
A Greedy Approach" (arXiv:2504.14040).

Default topology is the 6-node path of Fig. 8 (nodes 0..5, link lengths in km):

    0 --32-- 1 --18-- 2 --35-- 3 --16-- 4 --24-- 5

Each link has a BSM node in its middle (Barrett-Kok entanglement generation). Node indices and
swapping orders use the paper's convention: nodes 0..n, interior nodes 1..n-1.

Simulation setup (Section V-A):
  * memory capacity Q_i = 6 per node, memory efficiency 0.95, memory frequency 1 kHz
  * coherence time T_cohere (default 20 ms); memories are cut off after T_cohere
  * fibers: peak frequency 50 MHz, attenuation 0.2 dB/km, speed of light 2e5 km/s
  * detectors: count rate 60 MHz, 100 ps time resolution, efficiency 0.95
  * swapping success probability 0.5 at every node
  * each run lasts 9 s or until 1000 end-to-end (E2E) entanglements are delivered
  * by default two settings are calibrated so link rates match Fig. 4/5 (see CALIBRATED_*);
    pass --literal to use the values above exactly as written

Operation (asynchronous, as in Section V-B):
  * entanglement generation runs continuously on every link whenever memories are free
  * swapping follows the static order: if node k is the i-th node in the order, it joins the
    segments (L <-> k) and (k <-> R), where L and R are the nearest nodes on each side that
    come later in the order (or are the end nodes). A swap happens as soon as k holds one
    memory entangled with L and one with R.
  * E2E entanglements between node 0 and node n are counted and immediately consumed
    (memories freed), so generation restarts on them.

Usage:
    python local/entanglement_swapping.py                         # all named orders (vora, baln, bal2, L2R, R2L)
    python local/entanglement_swapping.py --order vora            # just one named order from Section V-C
    python local/entanglement_swapping.py --order 1 3 2 4         # explicit order
    python local/entanglement_swapping.py --runs 5                # all named orders, averaged over 5 runs
    python local/entanglement_swapping.py --alloc 3 3 3 3 3 --coherence 10
    python local/entanglement_swapping.py --distances 50 50 50 --alloc 3 3 3 --order 1 2
"""

import argparse
import os
import statistics
import sys
import time
import itertools

# make the local `sequence` package importable when running from the local/ folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sequence.kernel.timeline import Timeline
from sequence.topology.node import QuantumRouter, BSMNode
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB
from sequence.constants import BARRET_KOK, SINGLE_HERALDED, BELL_DIAGONAL_STATE_FORMALISM
from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.resource_management.rule_manager import Rule

# ----------------------------------------------------------------------------------------
# Section V-A parameters
# ----------------------------------------------------------------------------------------
DISTANCES_KM = [32, 18, 35, 16, 24]   # Fig. 8: link lengths of the 6-node path
MEMORY_CAPACITY = 25                    # Q_i, memories per node
MEMORY_ALLOCATION = [4, 2, 4, 2, 4]    # m_i, memory pairs per link (optimal allocation, Sec. V-C)
COHERENCE_TIME_MS = 20                 # T_cohere

MEMORY_EFFICIENCY = 0.95
MEMORY_FREQUENCY = 1e3                 # Hz
MEMORY_RAW_FIDELITY = 0.85             # not used by the paper (fidelity is not considered)

FIBER_FREQUENCY = 50e6                 # Hz, peak frequency of quantum channels
FIBER_ATTENUATION = 0.2e-3             # dB/m  (0.2 dB/km)
LIGHT_SPEED = 2e-4                     # m/ps  (2e5 km/s)

DETECTOR_EFFICIENCY = 0.95
DETECTOR_COUNT_RATE = 60e6             # Hz
DETECTOR_RESOLUTION = 100              # ps

# swapping success probability q_x of each interior node: one value for every node, or a list
# [q_1, ..., q_{n-1}] with one value per interior node (e.g. [0.5, 0.4, 0.5, 0.6] for the 6-node path)
SWAP_SUCCESS_PROB = [ 0.5, 0.5, 0.5, 0.5]
SWAP_DEGRADATION = 1.0                 # fidelity is not considered in the paper

# Calibration (see --literal). The paper's link rates (Fig. 4/5: ~115 ent/s at 32 km, ~360 ent/s at
# 18 km with one memory pair) are only reproduced in this SeQUeNCe version when
#   * the memory frequency does not throttle Barrett-Kok excitations (at 1 kHz Memory.excite drops
#     every excitation that comes < 1 ms after the previous one, halving/thirding link rates), and
#   * classical messages only take the propagation delay (this version adds 10 us per message).
CALIBRATED_MEMORY_FREQUENCY = 1e6      # Hz, effectively no throttling
CALIBRATED_CC_DELAY = True             # classical delay = distance / light speed

SIM_DURATION_S = 9                     # each run lasts 9 s ...
TARGET_E2E = 200                      # ... or until 1000 E2E entanglements

# swapping orders compared in Section V-C (for the 6-node path)
NAMED_ORDERS = {
    "vora": [4, 1, 3, 2],   # order found by VoraSwap (also the optimal order)
    "baln": [1, 3, 2, 4],   # balanced tree
    "bal2": [4, 2, 3, 1],   # balanced tree
    "L2R":  [1, 2, 3, 4],   # sequential, left to right
    "R2L":  [4, 3, 2, 1],   # sequential, right to left
}
DEFAULT_ORDER = "all"      # run every named order; set to e.g. "vora" to run just one


# ----------------------------------------------------------------------------------------
# Entanglement generation rules
# ----------------------------------------------------------------------------------------
def eg_rule_condition(memory_info, manager, args):
    if memory_info.state == "RAW" and memory_info.index in args["memory_indices"]:
        return [memory_info]
    return []


def eg_match_func(protocols, args):
    for protocol in protocols:
        if not isinstance(protocol, EntanglementGenerationA):
            continue
        memory_array = protocol.owner.get_components_by_type("MemoryArray")[0]
        if (protocol.remote_node_name == args["remote_node"]
                and memory_array.memories.index(protocol.memory) in args["memory_indices"]):
            return protocol
    return None


def eg_rule_action_request(memories_info, args):
    memory = memories_info[0].memory
    protocol = EntanglementGenerationA.create(None, "EGA." + memory.name, args["mid_name"], args["other_name"], memory)
    req_args = {"remote_node": args["node_name"], "memory_indices": args["remote_indices"]}
    return [protocol, [args["other_name"]], [eg_match_func], [req_args]]


def eg_rule_action_await(memories_info, args):
    memory = memories_info[0].memory
    protocol = EntanglementGenerationA.create(None, "EGA." + memory.name, args["mid_name"], args["other_name"], memory)
    return [protocol, [None], [None], [None]]


# ----------------------------------------------------------------------------------------
# Entanglement swapping rules
# ----------------------------------------------------------------------------------------
# Swapping messages are delivered by protocol name. Naming protocols only after their memories
# lets a late swap result reach a newer protocol on the same memory (after the memory expired and
# was re-entangled), which fails SeQUeNCe's sender check. A unique suffix makes late messages
# match nothing, so they are dropped like messages for any finished protocol.
_protocol_ids = itertools.count()


def es_rule_condition_A(memory_info, manager, args):
    """On the swapping node: one memory entangled with `left`, another with `right`."""
    left, right = args["left"], args["right"]
    if memory_info.state != "ENTANGLED" or memory_info.remote_node not in (left, right):
        return []
    want = right if memory_info.remote_node == left else left
    for info in manager:
        if info.state == "ENTANGLED" and info.remote_node == want:
            return [memory_info, info]
    return []


def es_match_func(protocols, args):
    for protocol in protocols:
        if isinstance(protocol, EntanglementSwappingB) and protocol.memory.name == args["target_memo"]:
            return protocol
    return None


def es_rule_action_A(memories_info, args):
    memories = [info.memory for info in memories_info]
    protocol = EntanglementSwappingA(None, f"ESA.{memories[0].name}.{memories[1].name}.{next(_protocol_ids)}",
                                     memories[0], memories[1],
                                     success_prob=args["succ_prob"], degradation=args["degradation"])
    dsts = [info.remote_node for info in memories_info]
    req_args = [{"target_memo": info.remote_memo} for info in memories_info]
    return [protocol, dsts, [es_match_func, es_match_func], req_args]


def es_rule_condition_B(memory_info, manager, args):
    """On a segment end: a memory entangled with a node that will swap it."""
    if memory_info.state == "ENTANGLED" and memory_info.remote_node in args["swap_nodes"]:
        return [memory_info]
    return []


def es_rule_action_B(memories_info, args):
    memory = memories_info[0].memory
    protocol = EntanglementSwappingB(None, f"ESB.{memory.name}.{next(_protocol_ids)}", memory)
    return [protocol, [None], [None], [None]]


# ----------------------------------------------------------------------------------------
# E2E consumer: counts entanglements between the two end nodes and frees the memories
# ----------------------------------------------------------------------------------------
class E2EConsumer:
    def __init__(self, node, other_end, counter, tl):
        self.node = node
        self.other_end = other_end
        self.counter = counter   # shared dict; only the node-0 side counts
        self.tl = tl

    def get_memory(self, info):
        if info.state == "ENTANGLED" and info.remote_node == self.other_end:
            if self.node.name == "0":
                self.counter["e2e"] += 1
                if self.counter["e2e"] >= TARGET_E2E:
                    self.counter["stop_time"] = self.tl.now()
                    self.tl.stop()
            self.node.resource_manager.update(None, info.memory, "RAW")

    def get_reservation_result(self, reservation, result):
        pass


# ----------------------------------------------------------------------------------------
# Network construction
# ----------------------------------------------------------------------------------------
def link_indices(alloc, i):
    """(left, right) memory indices of node i: left faces link i-1, right faces link i."""
    n = len(alloc)
    left_size = alloc[i - 1] if i > 0 else 0
    right_size = alloc[i] if i < n else 0
    return list(range(0, left_size)), list(range(left_size, left_size + right_size))


def plan_swaps(order, n):
    """Return [(k, L, R), ...] following Algorithm 1: L/R are the predecessor/successor of k
    among the nodes that have not been swapped yet."""
    interior = list(range(1, n))
    if sorted(order) != interior:
        raise ValueError(f"swap order must be a permutation of the interior nodes {interior}, got {order}")
    alive = list(range(0, n + 1))
    plan = []
    for k in order:
        idx = alive.index(k)
        plan.append((k, alive[idx - 1], alive[idx + 1]))
        alive.remove(k)
    return plan


def build_network(tl, distances_km, coherence_s, seed, literal, memory_capacity=None, generation=BARRET_KOK,
                  memory_efficiency=None, detector_efficiency=None, bsm_success_rate=None, attenuations=None):
    """attenuations: optional fiber loss per link in dB/m (default FIBER_ATTENUATION for every link)."""
    n = len(distances_km)
    positions = [0.0]
    for d in distances_km:
        positions.append(positions[-1] + d * 1e3)   # m

    memory_efficiency = MEMORY_EFFICIENCY if memory_efficiency is None else memory_efficiency
    detector_efficiency = DETECTOR_EFFICIENCY if detector_efficiency is None else detector_efficiency
    memory_frequency = MEMORY_FREQUENCY if literal else CALIBRATED_MEMORY_FREQUENCY
    memo_args = {"efficiency": memory_efficiency, "frequency": memory_frequency,
                 "coherence_time": coherence_s, "fidelity": MEMORY_RAW_FIDELITY}
    memory_capacity = MEMORY_CAPACITY if memory_capacity is None else memory_capacity
    routers = [QuantumRouter(str(i), tl, memo_size=memory_capacity,
                             component_templates={"MemoryArray": memo_args})
               for i in range(n + 1)]

    detector = {"efficiency": detector_efficiency, "count_rate": DETECTOR_COUNT_RATE,
                "time_resolution": DETECTOR_RESOLUTION}
    if generation == SINGLE_HERALDED:
        sh_args = {"detectors": [dict(detector), dict(detector)]}
        if bsm_success_rate is not None:
            sh_args["success_rate"] = bsm_success_rate
        bsm_templates = {"encoding_type": "single_heralded", "SingleHeraldedBSM": sh_args}
    else:
        bsm_templates = {"SingleAtomBSM": {"detectors": [dict(detector), dict(detector)]}}
    bsms = []
    bsm_positions = []
    for i in range(n):
        bsm = BSMNode(f"BSM.{i}.{i + 1}", tl, [str(i), str(i + 1)], component_templates=bsm_templates)
        bsms.append(bsm)
        bsm_positions.append((positions[i] + positions[i + 1]) / 2)
        routers[i].add_bsm_node(bsm.name, str(i + 1))
        routers[i + 1].add_bsm_node(bsm.name, str(i))

    all_nodes = routers + bsms
    all_positions = positions + bsm_positions
    # for k, node in enumerate(all_nodes):
    #     node.set_seed(seed * 1000 + k)

    # classical channels between every pair of nodes, length = distance along the path
    for a, pa in zip(all_nodes, all_positions):
        for b, pb in zip(all_nodes, all_positions):
            if a is not b:
                distance = max(abs(pa - pb), 1.0)
                delay = distance / LIGHT_SPEED if (CALIBRATED_CC_DELAY and not literal) else -1
                cc = ClassicalChannel(f"cc_{a.name}_{b.name}", tl, distance, delay=delay)
                cc.set_ends(a, b.name)

    # quantum channels router -> BSM in the middle of each link
    for i, bsm in enumerate(bsms):
        half = distances_km[i] * 1e3 / 2
        attenuation = FIBER_ATTENUATION if attenuations is None else attenuations[i]
        for k in (i, i + 1):
            qc = QuantumChannel(f"qc_{k}_{bsm.name}", tl, attenuation, half,
                                light_speed=LIGHT_SPEED, frequency=FIBER_FREQUENCY)
            qc.set_ends(routers[k], bsm.name)

    return routers, bsms


def swap_probs_by_node(swap_prob, n):
    """Map each interior node 1..n-1 to its swapping success probability q_x."""
    values = list(swap_prob) if isinstance(swap_prob, (list, tuple)) else [swap_prob]
    if len(values) == 1:
        values = values * (n - 1)
    if len(values) != n - 1:
        raise ValueError(f"need 1 or {n - 1} swap success probabilities (nodes 1..{n - 1}), got {values}")
    for q in values:
        if not 0 <= q <= 1:
            raise ValueError(f"swap success probability must be in [0, 1], got {q}")
    return {x: q for x, q in zip(range(1, n), values)}


def load_rules(routers, alloc, swap_plan, swap_probs):
    n = len(alloc)
    # entanglement generation on every link
    for i in range(n):
        j = i + 1
        mid = f"BSM.{i}.{j}"
        _, right_i = link_indices(alloc, i)
        left_j, _ = link_indices(alloc, j)
        action_args = {"mid_name": mid, "other_name": str(j), "node_name": str(i), "remote_indices": left_j}
        routers[i].resource_manager.load(Rule(10, eg_rule_action_request, eg_rule_condition, action_args,
                                              {"memory_indices": right_i}))
        action_args = {"mid_name": mid, "other_name": str(i)}
        routers[j].resource_manager.load(Rule(10, eg_rule_action_await, eg_rule_condition, action_args,
                                              {"memory_indices": left_j}))

    # swapping: node k swaps (L, R); L and R hold the B side for memories entangled with k
    b_side = {i: [] for i in range(n + 1)}
    for k, left, right in swap_plan:
        action_args = {"succ_prob": swap_probs[k], "degradation": SWAP_DEGRADATION}
        routers[k].resource_manager.load(Rule(10, es_rule_action_A, es_rule_condition_A, action_args,
                                              {"left": str(left), "right": str(right)}))
        b_side[left].append(str(k))
        b_side[right].append(str(k))
    for i, swap_nodes in b_side.items():
        if swap_nodes:
            routers[i].resource_manager.load(Rule(10, es_rule_action_B, es_rule_condition_B, {},
                                                  {"swap_nodes": swap_nodes}))


def count_expirations(routers, counter):
    """Wrap Memory.expire to count memories that decohere before being used (Fig. 11 metric)."""
    for router in routers:
        for memory in router.get_components_by_type("MemoryArray")[0].memories:
            original = memory.expire

            def wrapped(original=original, memory=memory):
                if not memory.is_in_application:
                    counter["expired"] += 1
                original()

            memory.expire = wrapped


# ----------------------------------------------------------------------------------------
# One simulation run
# ----------------------------------------------------------------------------------------
def make_timeline(generation):
    """Timeline for the chosen generation protocol.

    Single-heralded generation needs SeQUeNCe's Bell-diagonal formalism, whose run_circuit is a
    no-op returning None; swapping reads measurement results from it, so it is made to return
    all-zero outcomes. Fidelity is not modeled here, so swaps then only relabel which memories
    are entangled, which is all the throughput count needs.
    """
    stop = int(SIM_DURATION_S * 1e12)
    if generation == SINGLE_HERALDED:
        tl = Timeline(stop, formalism=BELL_DIAGONAL_STATE_FORMALISM)
        tl.quantum_manager.run_circuit = lambda circuit, keys, meas_samp=None: {k: 0 for k in keys}
    elif generation == BARRET_KOK:
        tl = Timeline(stop)
    else:
        raise ValueError(f"unknown generation protocol {generation!r}; use {BARRET_KOK!r} or {SINGLE_HERALDED!r}")
    EntanglementGenerationA.set_global_type(generation)
    EntanglementGenerationB.set_global_type(generation)
    return tl


def run_once(order, distances_km, alloc, coherence_ms, seed, literal=False, swap_prob=SWAP_SUCCESS_PROB,
             memory_capacity=None, generation=BARRET_KOK, memory_efficiency=None, detector_efficiency=None,
             bsm_success_rate=None, attenuations=None):
    n = len(distances_km)
    swap_plan = plan_swaps(order, n)
    swap_probs = swap_probs_by_node(swap_prob, n)

    tl = make_timeline(generation)
    routers, _ = build_network(tl, distances_km, coherence_ms / 1e3, seed, literal, memory_capacity, generation,
                               memory_efficiency, detector_efficiency, bsm_success_rate, attenuations)

    counter = {"e2e": 0, "expired": 0, "stop_time": None}
    routers[0].set_app(E2EConsumer(routers[0], str(n), counter, tl))
    routers[n].set_app(E2EConsumer(routers[n], "0", counter, tl))
    count_expirations(routers, counter)

    tl.init()
    load_rules(routers, alloc, swap_plan, swap_probs)
    tl.run()

    elapsed_s = (counter["stop_time"] if counter["stop_time"] is not None else tl.stop_time) / 1e12
    return {"e2e": counter["e2e"], "elapsed_s": elapsed_s,
            "throughput": counter["e2e"] / elapsed_s, "expired": counter["expired"]}


def validate_alloc(alloc, n):
    if len(alloc) != n:
        raise ValueError(f"need one memory allocation per link ({n}), got {alloc}")
    for i in range(n + 1):
        used = (alloc[i - 1] if i > 0 else 0) + (alloc[i] if i < n else 0)
        if used > MEMORY_CAPACITY:
            raise ValueError(f"node {i} needs {used} memories but has Q = {MEMORY_CAPACITY}")


def main():
    parser = argparse.ArgumentParser(description="Static-order entanglement swapping (arXiv:2504.14040, Sec. V)")
    parser.add_argument("--order", nargs="+", default=[DEFAULT_ORDER],
                        help=f"node order (e.g. 4 1 3 2), a name in {list(NAMED_ORDERS)}, or 'all'")
    parser.add_argument("--distances", type=float, nargs="+", default=DISTANCES_KM, help="link lengths in km")
    parser.add_argument("--alloc", type=int, nargs="+", default=None, help="memory pairs per link")
    parser.add_argument("--coherence", type=float, default=COHERENCE_TIME_MS, help="coherence time in ms")
    parser.add_argument("--runs", type=int, default=1, help="number of runs (different seeds) per order")
    parser.add_argument("--swap-prob", type=float, nargs="+", default=None,
                        help="swap success probability: one value for all interior nodes, or one per "
                             "node 1..n-1 (default SWAP_SUCCESS_PROB)")
    parser.add_argument("--literal", action="store_true",
                        help="use 1 kHz memory frequency and SeQUeNCe's default classical delay as-is "
                             "(gives ~3x lower rates than the paper; see CALIBRATED_* constants)")
    args = parser.parse_args()

    n = len(args.distances)
    alloc = args.alloc if args.alloc is not None else (
        MEMORY_ALLOCATION if n == len(MEMORY_ALLOCATION) else [MEMORY_CAPACITY // 2] * n)
    validate_alloc(alloc, n)
    swap_prob = args.swap_prob if args.swap_prob is not None else SWAP_SUCCESS_PROB
    swap_probs = swap_probs_by_node(swap_prob, n)

    if args.order == ["all"]:
        orders = list(NAMED_ORDERS.items())
    elif len(args.order) == 1 and args.order[0] in NAMED_ORDERS:
        orders = [(args.order[0], NAMED_ORDERS[args.order[0]])]
    else:
        order = [int(x) for x in args.order]
        orders = [(str(order), order)]
    if n != len(DISTANCES_KM) and any(name in NAMED_ORDERS for name, _ in orders):
        raise ValueError("named orders are defined for the 6-node path of Fig. 8; pass an explicit order")

    print("=" * 78)
    print("Path: " + "".join(f"{i} --{d:g}km-- " for i, d in enumerate(args.distances)) + str(n))
    print(f"Memory allocation per link: {alloc}   (Q = {MEMORY_CAPACITY} per node)")
    print("Swap success probability per node: "
          + ", ".join(f"q{x} = {q:g}" for x, q in swap_probs.items()))
    print(f"T_cohere = {args.coherence:g} ms, run length = {SIM_DURATION_S} s or {TARGET_E2E} E2E, runs per order = {args.runs}")
    print("Mode: " + ("literal (1 kHz memory frequency, default classical delay)" if args.literal
                      else "calibrated to the paper's link rates (use --literal for raw Sec. V-A values)"))
    print("=" * 78)

    results = []
    for name, order in orders:
        plan = plan_swaps(order, n)
        print(f"\nOrder {name} = {order}")
        for k, left, right in plan:
            print(f"    node {k} swaps ({left} <-> {k}) + ({k} <-> {right})  ->  {left} <-> {right}"
                  f"   (q{k} = {swap_probs[k]:g})")
        throughputs, expired_per_e2e = [], []
        for run in range(args.runs):
            tick = time.time()
            r = run_once(order, args.distances, alloc, args.coherence, seed=run, literal=args.literal,
                         swap_prob=swap_prob)
            throughputs.append(r["throughput"])
            expired_per_e2e.append(r["expired"] / r["e2e"] if r["e2e"] else float("inf"))
            print(f"  run {run}: {r['e2e']} E2E in {r['elapsed_s']:.3f} s  ->  {r['throughput']:.2f} ent/s, "
                  f"{r['expired']} expired memories   (wall {time.time() - tick:.1f} s)")
        mean = statistics.mean(throughputs)
        std = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0
        results.append((name, order, mean, std, statistics.mean(expired_per_e2e)))

    print("\n" + "=" * 78)
    print(f"{'order':<16}{'throughput (ent/s)':>22}{'expired mem / E2E':>22}")
    for name, order, mean, std, exp in sorted(results, key=lambda r: -r[2]):
        label = name if name.startswith("[") else f"{name} {order}"
        print(f"{label:<16}{mean:>14.2f} +/- {std:<6.2f}{exp:>20.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
