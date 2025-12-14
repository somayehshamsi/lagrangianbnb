
import networkx as nx
import numpy as np
from time import time
from collections import defaultdict, OrderedDict
from scipy.optimize import linprog  
import math
import heapq
import hashlib



class UnionFind:
    __slots__ = ['parent', 'rank', 'size']
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.size[pu] < self.size[pv]:
            pu, pv = pv, pu
        self.parent[pv] = pu
        self.size[pu] += self.size[pv]
        self.rank[pu] = max(self.rank[pu], self.rank[pv] + 1)
        return True
    
    def connected(self, u, v):
        return self.find(u) == self.find(v)
    
    def count_components(self):
        return len(set(self.find(i) for i in range(len(self.parent))))

class LRUCache:
    __slots__ = ['cache', 'capacity']
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class LagrangianMST:
    total_compute_time = 0


    def __init__(self, edges, num_nodes, budget, fixed_edges=None, excluded_edges=None,
                 initial_lambda=0.9, step_size=0.001, max_iter=10, 
                 use_cover_cuts=False, cut_frequency=5, use_bisection=False,
                 verbose=False, shared_graph=None):
        start_time = time()
        self.edges = edges
        self.num_nodes = num_nodes
        self.budget = budget
        self.fixed_edges = {tuple(sorted((u, v))) for u, v in (fixed_edges or set())}
        self.excluded_edges = {tuple(sorted((u, v))) for u, v in (excluded_edges or set())}

        edge_key = id(edges)
        if getattr(LagrangianMST, "_edge_key", None) != edge_key:
            LagrangianMST._edge_key = edge_key
            edge_list = [tuple(sorted((u, v))) for u, v, _, _ in edges]
            LagrangianMST._edge_list = edge_list
            LagrangianMST._edge_indices = {edge: idx for idx, edge in enumerate(edge_list)}
            LagrangianMST._edge_weights = np.array([w for _, _, w, _ in edges], dtype=np.float32)
            LagrangianMST._edge_lengths = np.array([l for _, _, _, l in edges], dtype=np.float32)
            LagrangianMST._edge_attributes = {
                edge: (w, l) for (edge, (_, _, w, l)) in zip(edge_list, edges)
            }

        self.edge_list = LagrangianMST._edge_list
        self.edge_indices = LagrangianMST._edge_indices
        self.edge_weights = LagrangianMST._edge_weights
        self.edge_lengths = LagrangianMST._edge_lengths
        self.edge_attributes = LagrangianMST._edge_attributes

        self.lmbda = initial_lambda
        self.step_size = step_size
        # self.p = p
        self.max_iter = max_iter
        self.use_bisection = use_bisection
        self.verbose = verbose

        self.best_lower_bound = float('-inf')
        self.best_upper_bound = float('inf')
        self.last_mst_edges = []
        self.primal_solutions = []
        self.fractional_solutions = []
        self.step_sizes = []
        self.subgradients = []
        self._MAX_HISTORY = 100
        self._primal_history_cap = 30
        self._fractional_history_cap = 50
        self._subgradient_history_cap = 20

        self.best_lambda = self.lmbda
        self.best_mst_edges = None
        self.best_cost = 0

        self.use_cover_cuts = use_cover_cuts
        self.cut_frequency = cut_frequency
        self.best_cuts = []
        self.best_cut_multipliers = {}

        self.multipliers = []

        self.fixed_edge_indices = {
            self.edge_indices.get((u, v)) for u, v in self.fixed_edges
            if (u, v) in self.edge_indices
        }
        self.excluded_edge_indices = {
            self.edge_indices.get((u, v)) for u, v in self.excluded_edges
            if (u, v) in self.edge_indices
        }

        self.cache_tolerance = 1e-6 if num_nodes > 100 else 1e-8
        # self.mst_cache = LRUCache(capacity=max(20, num_nodes * 2))
        self.mst_cache = LRUCache(capacity=64)


        self.last_mst_edges = None

        if shared_graph is not None:
            self.graph = shared_graph
        else:
            self.graph = nx.Graph()
            self.graph.add_edges_from(self.edge_list)

        self._free_mask_cache = None
        self._free_mask_key = None
        self._mw_cached = None
        self._mw_lambda = None
        self._mw_mu = None

        end_time = time()
        LagrangianMST.total_compute_time += end_time - start_time


    

    def reset(self, *, fixed_edges=None, excluded_edges=None, initial_lambda=None,
              step_size=None, max_iter=None, use_cover_cuts=None, cut_frequency=None,
              use_bisection=None, verbose=None):
        if fixed_edges is not None:
            self.fixed_edges = {tuple(sorted((u, v))) for u, v in fixed_edges}
        else:
            self.fixed_edges = set()

        if excluded_edges is not None:
            self.excluded_edges = {tuple(sorted((u, v))) for u, v in excluded_edges}
        else:
            self.excluded_edges = set()

        self.fixed_edge_indices = {
            self.edge_indices.get((u, v)) for u, v in self.fixed_edges
            if (u, v) in self.edge_indices
        }
        self.excluded_edge_indices = {
            self.edge_indices.get((u, v)) for u, v in self.excluded_edges
            if (u, v) in self.edge_indices
        }

        if initial_lambda is not None:
            self.lmbda = float(initial_lambda)
        else:
            self.lmbda = getattr(self, "lmbda", 0.9)

        if step_size is not None:
            self.step_size = float(step_size)
        if max_iter is not None:
            self.max_iter = int(max_iter)
        if use_cover_cuts is not None:
            self.use_cover_cuts = bool(use_cover_cuts)
        if cut_frequency is not None:
            self.cut_frequency = int(cut_frequency)
        if use_bisection is not None:
            self.use_bisection = bool(use_bisection)
        if verbose is not None:
            self.verbose = bool(verbose)

        self.best_lower_bound = float("-inf")
        self.best_upper_bound = float("inf")
        self.best_mst_edges = []
        self.best_cuts = []
        self.best_cut_multipliers = {}
        self.multipliers = []

        self.last_mst_edges = None

        try:
            cap = self.mst_cache.capacity
        except Exception:
            cap = max(20, self.num_nodes * 2)
        self.mst_cache = LRUCache(capacity=cap)

        self._invalidate_weight_cache()


    def clear_iteration_state(self):
        """Clear per-iteration buffers"""
        self.primal_solutions = []
        self.fractional_solutions = []
        self.subgradients = []
        self.step_sizes = []
        self.multipliers = []
        # self.last_modified_weights = None
        # self.last_mst_edges = None
        self._invalidate_weight_cache()
        if hasattr(self, 'mst_cache'):
            self.mst_cache = LRUCache(capacity=5)
   

    
    


    def generate_cover_cuts(self, mst_edges):
        """
        Fast + strong cover cuts (tightened):
        - Residualization: A, B', r' (clamped)
        - Seed minimal cover from T^λ ∩ A; certificate shrinking (optimistic U), lazy DSU confirm
        - Micro-seed from top-L heaviest admissible edges (tiny diversity)
        - τ-lifting (remaining-based, safe, one pass)
        - Strict effective-RHS pruning and current-violation checks
        - Dedup with dominance & subset-dominance
        """
        if not mst_edges:
            return []

        EPS = 1e-12
        L_MICRO = 3
        MAX_RETURN = 10

        # --- normalize edges ---
        def norm(e):
            u, v = e
            return (u, v) if u <= v else (v, u)
        mst_norm = [norm(e) for e in mst_edges]
        mst_set = set(mst_norm)  # for fast lhs checks

        # --- accessors / data ---
        edge_attr = self.edge_attributes           # edge -> (w, ℓ)
        def get_len(e): return edge_attr[e][1]

        fixed = set(getattr(self, "fixed_edges", set()))
        excluded = set(getattr(self, "excluded_edges", set()))
        budget = self.budget

        # Residual data
        L_fix = sum(get_len(e) for e in fixed if e in edge_attr)
        Bp = budget - L_fix
        r_all = self.num_nodes - 1
        r_prime = max(0, r_all - len(fixed))       # clamp

        # If residual is degenerate, early out
        if r_prime == 0:
            return []

        # Admissible edges
        A = {e for e in getattr(self, "edge_list", []) if e not in fixed and e not in excluded and e in edge_attr}
        if not A:
            return []

        # T^λ ∩ A (use provided mst_edges)
        TcapA = [e for e in mst_norm if e in A]

        # If residual MST is feasible, nothing to cut
        mst_len = sum(get_len(e) for e in TcapA)
        if mst_len <= Bp + EPS:
            return []

        # If the residual budget is negative (fixes already exceed B'), cap behavior:
        # still try to emit a strong (minimal) cover; otherwise return empty to avoid INF artefacts.
        cuts = []

        # Pre-sort A for optimistic completion and lifting
        A_sorted = sorted(A, key=lambda e: get_len(e))

        def U_of(Sprime):
            """Optimistic completion: sum of k cheapest in A \\ S', where k=r'-|S'|."""
            k = r_prime - len(Sprime)
            if k <= 0:
                return 0.0
            Sprime_set = Sprime if isinstance(Sprime, set) else set(Sprime)
            total = 0.0; taken = 0
            for e in A_sorted:
                if e in Sprime_set:
                    continue
                total += get_len(e); taken += 1
                if taken == k:
                    break
            return total if taken == k else float('inf')

        def completion_mst_cost(Ssub):
            """Exact completion via Kruskal on contracted graph: DSU scan."""
            parent, rank = {}, {}
            def find(x):
                px = parent.get(x, x)
                if px != x:
                    parent[x] = find(px)
                else:
                    parent.setdefault(x, x)
                return parent[x]
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry: return False
                rxr, ryr = rank.get(rx,0), rank.get(ry,0)
                if rxr < ryr: parent[rx] = ry
                elif rxr > ryr: parent[ry] = rx
                else: parent[ry] = rx; rank[rx] = rxr + 1
                return True

            # nodes
            if hasattr(self, "graph") and hasattr(self.graph, "nodes"):
                nodes = list(self.graph.nodes)
            else:
                nodes = list({n for e in A for n in e})
            for n in nodes:
                parent[n] = n; rank[n] = 0

            # contract fixed ∪ Ssub
            contracted = set(fixed) | set(Ssub)
            for (u, v) in contracted:
                union(u, v)

            k_needed = r_prime - len(Ssub)
            if k_needed <= 0:
                return 0.0

            total = 0.0; taken = 0; Sset = set(Ssub)
            for e in A_sorted:
                if e in Sset: continue
                u, v = e
                if union(u, v):
                    total += get_len(e); taken += 1
                    if taken == k_needed: break
            return total if taken == k_needed else float('inf')

        def build_residual_minimal_cover(desc_edges):
            """Minimal cover on B': add in desc ℓ, then prune shortest while violation remains."""
            S, sL = [], 0.0
            for e in desc_edges:
                if e not in edge_attr:
                    continue
                S.append(e); sL += get_len(e)
                if sL > Bp + EPS:
                    S.sort(key=lambda x: get_len(x))  # increasing
                    k = 0
                    while k < len(S) and (sL - get_len(S[k]) > Bp + EPS):
                        sL -= get_len(S[k]); k += 1
                    if k > 0:
                        S = S[k:]
                    return S, sL
            return None, None

        def rhs_eff(cset):
            """Effective RHS after accounting fixed-in edges."""
            return len(cset) - 1 - sum(1 for e in cset if e in fixed)

        def is_violated_now(cset):
            """Check current MST violation: lhs > rhs_eff."""
            lhs = sum(1 for e in cset if e in mst_set)
            return lhs > rhs_eff(cset)

        def try_shrink_and_add(seed_S, seed_sumL):
            """Shrink from a seed using optimistic certificate; lazily confirm with DSU if margin is thin."""
            if not seed_S or len(seed_S) <= 1:
                return
            S_work = sorted(seed_S, key=lambda e: get_len(e), reverse=True)
            sumL = seed_sumL
            idx = 0
            while idx < len(S_work) and sumL > Bp + EPS:
                sumL -= get_len(S_work[idx]); idx += 1
            Sprime = S_work[idx:]  # first with sumℓ ≤ B'

            def margin(Slist):
                return (sum(get_len(e) for e in Slist) + U_of(Slist)) - Bp

            if Sprime:
                # strict effective-RHS and current-violation screening
                if rhs_eff(Sprime) <= 0:
                    return
                if margin(Sprime) > 0:
                    # If optimistic margin is big enough, still require current violation
                    if is_violated_now(Sprime):
                        cuts.append((set(Sprime), len(Sprime) - 1))
                else:
                    # thin margin: do exact completion
                    exact_total = sum(get_len(e) for e in Sprime) + completion_mst_cost(Sprime)
                    if exact_total > Bp + EPS and is_violated_now(Sprime):
                        cuts.append((set(Sprime), len(Sprime) - 1))

        # --- (1) primary seed from T^λ ∩ A ---
        T_desc = sorted(TcapA, key=lambda e: get_len(e), reverse=True)
        S_seed, sumL_seed = build_residual_minimal_cover(T_desc)
        if not S_seed:
            return []
        S_seed = list(S_seed)

        # Add primary seed only if effective and violated now
        if rhs_eff(S_seed) > 0 and is_violated_now(S_seed):
            cuts.append((set(S_seed), len(S_seed) - 1))
        # and try to shrink/confirm
        try_shrink_and_add(S_seed, sumL_seed)

            # --- (1c) Lifting on S_seed (sequence-independent, safe) ---
        if S_seed and rhs_eff(S_seed) > 0 and is_violated_now(S_seed):
            S_base = set(S_seed)
            # L_max = max length in S_seed
            L_max = max(get_len(e) for e in S_base)

            # Add all admissible edges with ℓ(f) ≥ L_max
            lift_add = {
                f for f in A
                if f not in S_base and get_len(f) + EPS >= L_max
            }

            if lift_add:
                S_lift = S_base | lift_add
                # Keep RHS = |S_seed| - 1 (original minimal cover size)
                if rhs_eff(S_lift) > 0 and is_violated_now(S_lift):
                    cuts.append((S_lift, len(S_seed) - 1))

        # --- (1b) micro-seed: top-L heaviest admissible edges ---
        if L_MICRO > 0 and len(A) > 0:
            heavyA = sorted(A, key=lambda e: get_len(e), reverse=True)[:L_MICRO]
            S2, sumL2 = build_residual_minimal_cover(heavyA)
            if S2:
                S2set = set(S2)
                # require effectiveness and current violation; avoid duplicate of S_seed
                if rhs_eff(S2set) > 0 and S2set != set(S_seed) and is_violated_now(S2set):
                    cuts.append((S2set, len(S2) - 1))
                    try_shrink_and_add(S2, sumL2)
                    # --- (1b-lift) Lifting on S2 (same rule as S_seed) ---
                    L_max2 = max(get_len(e) for e in S2set)

                    lift_add2 = {
                        f for f in A
                        if f not in S2set and get_len(f) + EPS >= L_max2
                    }

                    if lift_add2:
                        S2_lift = S2set | lift_add2
                        # RHS stays |S2| - 1 (based on the original minimal cover)
                        if rhs_eff(S2_lift) > 0 and is_violated_now(S2_lift):
                            cuts.append((S2_lift, len(S2) - 1))

       
        # --- dedup & dominance-aware selection ---
        uniq = {}
        for cset, rhs in cuts:
            key = tuple(sorted(cset))  # stable order for hashing
            best = uniq.get(key)
            # if same support appears with different rhs, keep the *stronger* (smaller rhs),
            # and if rhs ties, keep the one with smaller support just for consistency
            if best is None or rhs < best[1] or (rhs == best[1] and len(cset) < len(best[0])):
                uniq[key] = (cset, rhs)

        final = list(uniq.values())
        # process stronger cuts first: smaller rhs, then smaller |S|
        final.sort(key=lambda t: (t[1], len(t[0])))

        kept = []
        for cset, rhs in final:
            if rhs_eff(cset) <= 0:
                continue
            # drop cset if there is ALREADY a stronger cut (dset,drhs) with
            # dset ⊆ cset and drhs ≤ rhs
            dominated = any(dset <= cset and drhs <= rhs for dset, drhs in kept)
            if not dominated:
                kept.append((cset, rhs))
        return kept[:MAX_RETURN]
    # def generate_cover_cuts(self, mst_edges):
    #     """
    #     Cover-cut generation consistent with the LaTeX logic, with corner-case safety:

    #     Setup:
    #     F_+ fixed, F_- excluded.
    #     A := E \ (F_+ ∪ F_-)
    #     B' := B - sum_{e in F_+} ℓ(e)

    #     (1) Seed residual-minimal cover on B':
    #     From T^λ ∩ A, sort by nonincreasing ℓ and add until sumℓ > B'.
    #     Then delete the currently shortest edge while preserving violation to obtain residual-minimal cover S:
    #         sumℓ(S) > B' and sumℓ(S \ {e}) <= B' for all e in S
    #     Add classical cover cut: sum_{e in S} x_e <= |S|-1  (only if it separates current mst_edges).

    #     (2) Connectivity-aware shrinking:
    #     From S, delete the longest edges until sumℓ <= B' to obtain first S'.
    #     Define r'(S') := κ(F_+ ∪ S') - 1  (components after contracting F_+ ∪ S')
    #     Define optimistic completion bound:
    #         U(S') = sum of r'(S') cheapest ℓ in A \ S'
    #     Residual certificate:
    #         if sumℓ(S') + U(S') > B' then any spanning tree containing S' (and F_+) exceeds B
    #     From S', keep deleting the longest edge while the certificate holds; stop when inclusion-minimal => S*
    #     Add cut on S*.

    #     Safety additions:
    #         - Never certify from INF (insufficient edges/connectivity)
    #         - If optimistic cert fails, try exact completion (Kruskal on contracted graph):
    #             if sumℓ(S') + completion_cost(S') > B'  then S' is infeasible -> allow shrinking with exact-cert.

    #     Lifting cover cuts (LaTeX §Lifting):
    #     Only for residual-minimal covers from (1):
    #         Let L_max = max_{e in S} ℓ(e)
    #         Add all f∉S with ℓ(f) >= L_max with unit coefficient; RHS remains |S|-1.

    #     Notes:
    #     - All certificate logic uses components-based r'(S) = κ(F_+ ∪ S) - 1
    #     - We skip generating cuts from sets where (F_+ ∪ S) contains a cycle (cannot be in any tree; usually useless).
    #     - We only *add* cuts that separate the current mst_edges (separation filter).
    #     """
    #     if not mst_edges:
    #         return []

    #     EPS = 1e-12
    #     L_MICRO = 3
    #     MAX_RETURN = 10

    #     # ---- normalize edges ----
    #     def norm(e):
    #         u, v = e
    #         return (u, v) if u <= v else (v, u)

    #     mst_norm = [norm(e) for e in mst_edges]
    #     mst_set = set(mst_norm)

    #     edge_attr = self.edge_attributes  # edge -> (w, ℓ)
    #     def get_len(e): return edge_attr[e][1]

    #     fixed = set(getattr(self, "fixed_edges", set()))
    #     excluded = set(getattr(self, "excluded_edges", set()))
    #     budget = self.budget

    #     # Nodes
    #     if hasattr(self, "graph") and hasattr(self.graph, "nodes"):
    #         nodes = list(self.graph.nodes)
    #     else:
    #         all_edges = getattr(self, "edge_list", [])
    #         nodes = list({n for (u, v) in all_edges for n in (u, v)})

    #     # Residual budget B'
    #     L_fix = sum(get_len(e) for e in fixed if e in edge_attr)
    #     Bp = budget - L_fix

    #     # r' = |V|-1-|F_+| (degeneracy check only)
    #     r_all = self.num_nodes - 1
    #     r_prime = max(0, r_all - len(fixed))
    #     if r_prime == 0:
    #         return []

    #     # Admissible edges A = E \ (F_+ ∪ F_-)
    #     A = {
    #         e for e in getattr(self, "edge_list", [])
    #         if e in edge_attr and e not in fixed and e not in excluded
    #     }
    #     if not A:
    #         return []

    #     # T^λ ∩ A
    #     TcapA = [e for e in mst_norm if e in A]

    #     # If current residual MST is feasible, no need to cut
    #     if sum(get_len(e) for e in TcapA) <= Bp + EPS:
    #         return []

    #     # Sort admissible edges by nondecreasing length (for U and exact completion)
    #     A_sorted = sorted(A, key=lambda e: get_len(e))

    #     # ---- DSU helpers ----
    #     def dsu_init():
    #         parent = {n: n for n in nodes}
    #         rank = {n: 0 for n in nodes}

    #         def find(x):
    #             while parent[x] != x:
    #                 parent[x] = parent[parent[x]]
    #                 x = parent[x]
    #             return x

    #         def union(a, b):
    #             ra, rb = find(a), find(b)
    #             if ra == rb:
    #                 return False
    #             if rank[ra] < rank[rb]:
    #                 ra, rb = rb, ra
    #             parent[rb] = ra
    #             if rank[ra] == rank[rb]:
    #                 rank[ra] += 1
    #             return True

    #         return find, union

    #     def contract_info(contract_edges):
    #         """
    #         Returns (components, has_cycle) after contracting contract_edges.
    #         """
    #         find, union = dsu_init()
    #         has_cycle = False
    #         for (u, v) in contract_edges:
    #             if (u, v) not in edge_attr:
    #                 continue
    #             if not union(u, v):
    #                 has_cycle = True
    #         comps = len({find(n) for n in nodes})
    #         return comps, has_cycle

    #     def k_needed_after_contract(contract_edges):
    #         comps, _ = contract_info(contract_edges)
    #         return comps - 1

    #     def U_of(S):
    #         """
    #         Optimistic completion bound:
    #         k = κ(F_+ ∪ S) - 1
    #         U(S) = sum of k cheapest lengths in A \ S
    #         """
    #         Sset = S if isinstance(S, set) else set(S)
    #         k = k_needed_after_contract(fixed | Sset)
    #         if k <= 0:
    #             return 0.0
    #         total = 0.0
    #         taken = 0
    #         for e in A_sorted:
    #             if e in Sset:
    #                 continue
    #             total += get_len(e)
    #             taken += 1
    #             if taken == k:
    #                 break
    #         return total if taken == k else float("inf")

    #     def completion_mst_cost(S):
    #         """
    #         Exact completion via Kruskal:
    #         - contract fixed ∪ S
    #         - add cheapest edges from A \ S that connect remaining components
    #         Returns inf if cannot complete.
    #         """
    #         Sset = S if isinstance(S, set) else set(S)

    #         find, union = dsu_init()
    #         has_cycle = False
    #         for (u, v) in (fixed | Sset):
    #             if (u, v) not in edge_attr:
    #                 continue
    #             if not union(u, v):
    #                 has_cycle = True  # cycle in contracted set; means "cannot be in any tree"
    #         if has_cycle:
    #             return float("inf")

    #         comps = len({find(n) for n in nodes})
    #         k = comps - 1
    #         if k <= 0:
    #             return 0.0

    #         total = 0.0
    #         taken = 0
    #         for e in A_sorted:
    #             if e in Sset:
    #                 continue
    #             u, v = e
    #             if union(u, v):
    #                 total += get_len(e)
    #                 taken += 1
    #                 if taken == k:
    #                     break
    #         return total if taken == k else float("inf")

    #     # ---- Residual-minimal cover construction (Step 1) ----
    #     def build_residual_minimal_cover(desc_edges):
    #         """
    #         Build residual-minimal cover S:
    #         - add edges in nonincreasing ℓ until sumℓ > B'
    #         - then delete currently-shortest edges while violation remains
    #         Returns (S_list, sumℓ(S)).
    #         """
    #         S = []
    #         sL = 0.0
    #         for e in desc_edges:
    #             if e not in edge_attr:
    #                 continue
    #             S.append(e)
    #             sL += get_len(e)
    #             if sL > Bp + EPS:
    #                 S.sort(key=lambda x: get_len(x))  # increasing
    #                 i = 0
    #                 while i < len(S) and (sL - get_len(S[i]) > Bp + EPS):
    #                     sL -= get_len(S[i])
    #                     i += 1
    #                 if i > 0:
    #                     S = S[i:]
    #                 return S, sL
    #         return None, None

    #     # ---- Separation check (uses stored RHS; important for lifting) ----
    #     def is_violated_now(cset, rhs):
    #         lhs = sum(1 for e in cset if e in mst_set)
    #         return lhs > rhs

    #     # ---- Certificate that matches LaTeX + safe exact fallback ----
    #     def cert_type(Slist):
    #         """
    #         Returns:
    #         "opt"   if optimistic cert holds (sumℓ + U > B')
    #         "exact" if optimistic fails but exact cert holds (sumℓ + completion > B')
    #         None    otherwise (cannot certify infeasibility)
    #         Also rejects cyclic (fixed ∪ S) sets.
    #         """
    #         Sset = set(Slist)
    #         _, has_cycle = contract_info(fixed | Sset)
    #         if has_cycle:
    #             return None  # skip cyclic sets (cannot appear in any tree; usually useless)

    #         sumS = sum(get_len(e) for e in Sset)

    #         U = U_of(Sset)
    #         if U != float("inf") and (sumS + U) > (Bp + EPS):
    #             return "opt"

    #         comp = completion_mst_cost(Sset)
    #         if comp != float("inf") and (sumS + comp) > (Bp + EPS):
    #             return "exact"

    #         return None

    #     # ---- Step (2) shrinking to inclusion-minimal S* under cert ----
    #     def shrink_to_inclusion_minimal(S_seed, sum_seed):
    #         """
    #         Implements LaTeX Step (2), with safe exact fallback:
    #         - remove longest until sum <= B' -> first S'
    #         - if cert holds for S', keep deleting edges (starting from longest) while cert holds
    #             until inclusion-minimal (no single-edge deletion preserves cert)
    #         """
    #         if not S_seed or len(S_seed) <= 1:
    #             return None

    #         S_work = sorted(S_seed, key=lambda e: get_len(e), reverse=True)
    #         sumL = sum_seed

    #         # first S' with sumℓ <= B'
    #         idx = 0
    #         while idx < len(S_work) and sumL > Bp + EPS:
    #             sumL -= get_len(S_work[idx])
    #             idx += 1
    #         Sprime = S_work[idx:]
    #         if not Sprime or len(Sprime) <= 1:
    #             return None

    #         # must be certifiable at Sprime to proceed (LaTeX logic)
    #         if cert_type(Sprime) is None:
    #             return None

    #         Sstar = sorted(Sprime, key=lambda e: get_len(e), reverse=True)

    #         # inclusion-minimal: keep removing an edge if cert still holds, restart scan
    #         changed = True
    #         while changed and len(Sstar) > 1:
    #             changed = False
    #             for j in range(len(Sstar)):  # longest->shortest due to sorting
    #                 trial = Sstar[:j] + Sstar[j+1:]
    #                 if len(trial) <= 1:
    #                     continue
    #                 if cert_type(trial) is not None:
    #                     Sstar = sorted(trial, key=lambda e: get_len(e), reverse=True)
    #                     changed = True
    #                     break

    #         return Sstar if len(Sstar) > 1 else None

    #     # ---- Lifting only for residual-minimal covers (LaTeX §Lifting) ----
    #     def lifted_cover_from_minimal(S_minimal):
    #         if not S_minimal:
    #             return None
    #         Sbase = set(S_minimal)

    #         # also skip if fixed ∪ S has cycle (means S can't be in a tree anyway)
    #         _, has_cycle = contract_info(fixed | Sbase)
    #         if has_cycle:
    #             return None

    #         Lmax = max(get_len(e) for e in Sbase)
    #         lift_add = {f for f in A if f not in Sbase and get_len(f) + EPS >= Lmax}
    #         if not lift_add:
    #             return None
    #         Slift = Sbase | lift_add
    #         rhs = len(S_minimal) - 1  # RHS stays based on original minimal cover
    #         return Slift, rhs

    #     cuts = []

    #     # -------------------------
    #     # (1) Primary seed from T^λ ∩ A
    #     # -------------------------
    #     T_desc = sorted(TcapA, key=lambda e: get_len(e), reverse=True)
    #     S_seed, sumL_seed = build_residual_minimal_cover(T_desc)
    #     if not S_seed:
    #         return []

    #     # Add seed cut if it separates and is not cyclic
    #     seed_set = set(S_seed)
    #     if cert_type(seed_set) is not None:
    #         rhs_seed = len(S_seed) - 1
    #         if rhs_seed > 0 and is_violated_now(seed_set, rhs_seed):
    #             cuts.append((seed_set, rhs_seed))

    #     # (2) Shrink to inclusion-minimal S* under cert, add if separates
    #     S_star = shrink_to_inclusion_minimal(S_seed, sumL_seed)
    #     if S_star:
    #         S_star_set = set(S_star)
    #         rhs_star = len(S_star) - 1
    #         if rhs_star > 0 and is_violated_now(S_star_set, rhs_star):
    #             cuts.append((S_star_set, rhs_star))

    #     # Lifting on residual-minimal seed cover (LaTeX-consistent)
    #     lifted = lifted_cover_from_minimal(S_seed)
    #     if lifted:
    #         Slift, rhs_lift = lifted
    #         if rhs_lift > 0 and is_violated_now(Slift, rhs_lift):
    #             cuts.append((Slift, rhs_lift))

    #     # -------------------------
    #     # (1b) Micro-seed: top-L heaviest admissible edges
    #     # -------------------------
    #     if L_MICRO > 0:
    #         heavyA = sorted(A, key=lambda e: get_len(e), reverse=True)[:L_MICRO]
    #         S2, sumL2 = build_residual_minimal_cover(heavyA)
    #         if S2:
    #             S2set = set(S2)
    #             if S2set != seed_set and cert_type(S2set) is not None:
    #                 rhs2 = len(S2) - 1
    #                 if rhs2 > 0 and is_violated_now(S2set, rhs2):
    #                     cuts.append((S2set, rhs2))

    #             S2_star = shrink_to_inclusion_minimal(S2, sumL2)
    #             if S2_star:
    #                 S2_star_set = set(S2_star)
    #                 rhs2_star = len(S2_star) - 1
    #                 if rhs2_star > 0 and is_violated_now(S2_star_set, rhs2_star):
    #                     cuts.append((S2_star_set, rhs2_star))

    #             lifted2 = lifted_cover_from_minimal(S2)
    #             if lifted2:
    #                 S2_lift, rhs2_lift = lifted2
    #                 if rhs2_lift > 0 and is_violated_now(S2_lift, rhs2_lift):
    #                     cuts.append((S2_lift, rhs2_lift))

    #     # -------------------------
    #     # Dedup + dominance filtering
    #     # -------------------------
    #     uniq = {}
    #     for cset, rhs in cuts:
    #         key = tuple(sorted(cset))
    #         best = uniq.get(key)
    #         if best is None or rhs < best[1]:
    #             uniq[key] = (cset, rhs)

    #     final = list(uniq.values())
    #     final.sort(key=lambda t: (t[1], len(t[0])))

    #     kept = []
    #     for cset, rhs in final:
    #         if rhs <= 0:
    #             continue
    #         if any(dset <= cset and drhs <= rhs for dset, drhs in kept):
    #             continue
    #         kept.append((cset, rhs))

    #     return kept[:MAX_RETURN]



    
    
    def compute_modified_weights(self):

        base = self.edge_weights.copy()
        lam = max(0.0, min(getattr(self, "lmbda", 0.0), 1e4))
        if lam:
            base = base + lam * self.edge_lengths

        # No cuts? return λ-priced base
        if not (self.use_cover_cuts and self.best_cuts):
            self._mw_cached = None
            self._mw_lambda = lam
            self._mw_mu = None
            self._mw_free_mask_key = None
            return base

        # μ vector aligned to current best_cuts
        cut_idxs_all = getattr(self, "_cut_edge_idx_all", None)  # indices for ALL cut edges
        mu_len = len(cut_idxs_all) if cut_idxs_all is not None else len(self.best_cuts)
        mu = np.array(
            [max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4)) for i in range(mu_len)],
            dtype=float,
        )

        # Cache key: (λ, μ, free-mask signature)
        _ = self._get_free_mask()
        free_mask_key = self._free_mask_key

        if (self._mw_cached is not None and
            self._mw_lambda == lam and
            self._mw_mu is not None and
            self._mw_mu.shape == mu.shape and
            np.allclose(self._mw_mu, mu, rtol=0, atol=0) and
            self._mw_free_mask_key == free_mask_key):
            return self._mw_cached

        # Add μ to ALL edges that belong to each cut (fixed edges included)
        weights = base.copy()
        if cut_idxs_all is not None:
            for i, idxs in enumerate(cut_idxs_all):
                m = mu[i]
                if m > 0.0 and idxs.size:
                    weights[idxs] += m

        else:
            # Fallback
            for i, (cut, _) in enumerate(self.best_cuts):
                m = mu[i]
                if m <= 0.0:
                    continue
                for e in cut:
                    j = self.edge_indices.get(e)
                    if j is not None:
                        weights[j] += m

        self._mw_cached = weights
        self._mw_lambda = lam
        self._mw_mu = mu.copy()
        self._mw_free_mask_key = free_mask_key
        return weights
    # def compute_modified_weights(self):

    #         base = self.edge_weights.copy()
    #         lam = max(0.0, min(getattr(self, "lmbda", 0.0), 1e4))
    #         if lam:
    #             base = base + self.edge_lengths * lam

    #         # No cuts? return λ-priced base
    #         if not (self.use_cover_cuts and self.best_cuts):
    #             self._mw_cached = None
    #             self._mw_lambda = lam
    #             self._mw_mu = None
    #             self._mw_free_mask_key = None
    #             self._last_mw = base
    #             return base

    #         # μ vector aligned to current best_cuts, with per-cut warm-up
    #         cut_idxs_all = getattr(self, "_cut_edge_idx_all", None)  # indices for ALL cut edges
    #         mu_len = len(cut_idxs_all) if cut_idxs_all is not None else len(self.best_cuts)

    #         depth = getattr(self, "depth", 0)
    #         mu_warmup_generations = getattr(self, "mu_warmup_generations", 2)
    #         birth_map = getattr(self, "cut_birth_depth", {})

    #         raw_mu = [
    #             max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
    #             for i in range(mu_len)
    #         ]

    #         def _mu_active(i):
    #             birth = birth_map.get(i, depth)
    #             return (depth - birth) >= mu_warmup_generations

    #         mu = np.array(
    #             [raw_mu[i] if _mu_active(i) else 0.0 for i in range(mu_len)],
    #             dtype=float,
    #         )

    #         # Cache key: (λ, μ, free-mask signature)
    #         _ = self._get_free_mask()
    #         free_mask_key = self._free_mask_key

    #         if (
    #             self._mw_cached is not None
    #             and self._mw_lambda == lam
    #             and self._mw_mu is not None
    #             and self._mw_mu.shape == mu.shape
    #             and np.allclose(self._mw_mu, mu, rtol=0, atol=0)
    #             and self._mw_free_mask_key == free_mask_key
    #         ):
    #             return self._mw_cached

    #         # Add μ to ALL edges that belong to each cut (fixed edges included)
    #         weights = base.copy()
    #         if cut_idxs_all is not None:
    #             for i, idxs in enumerate(cut_idxs_all):
    #                 m = mu[i]
    #                 if m > 0.0 and idxs.size:
    #                     weights[idxs] += m
    #         else:
    #             # Fallback
    #             for i, (cut, _) in enumerate(self.best_cuts):
    #                 m = mu[i]
    #                 if m <= 0.0:
    #                     continue
    #                 for e in cut:
    #                     j = self.edge_indices.get(e)
    #                     if j is not None:
    #                         weights[j] += m

    #         self._mw_cached = weights
    #         self._mw_lambda = lam
    #         self._mw_mu = mu.copy()
    #         self._mw_free_mask_key = free_mask_key
    #         self._last_mw = weights
    #         return weights



    def _invalidate_weight_cache(self):
        self._free_mask_cache = None
        self._free_mask_key = None
        self._mw_cached = None
        self._mw_lambda = None
        self._mw_mu = None
        self._mw_free_mask_key = None

   
    def _get_free_mask(self):

        fixed = frozenset(self.fixed_edges)
        forbidden = frozenset(getattr(self, "excluded_edges", set()))
        key = (fixed, forbidden)
        if self._free_mask_cache is not None and self._free_mask_key == key:
            return self._free_mask_cache

        if not fixed and not forbidden:
            self._free_mask_cache = None
            self._free_mask_key = key
            return None

        mask = np.ones(len(self.edge_list), dtype=bool)
        for e in fixed | forbidden:
            idx = self.edge_indices.get(e)
            if idx is not None:
                mask[idx] = False

        self._free_mask_cache = mask
        self._free_mask_key = key
        return mask


    def _append_with_cap(self, bucket, item, cap):
        bucket.append(item)
        overflow = len(bucket) - cap
        if overflow > 0:
            del bucket[:overflow]

    def _record_primal_solution(self, mst_edges, feasible):
        # snapshot = tuple(sorted(mst_edges)) if mst_edges else ()
        snapshot = tuple(mst_edges) if mst_edges else ()

        self._append_with_cap(
            self.primal_solutions,
            (snapshot, bool(feasible)),
            self._primal_history_cap,
        )

    
    def _record_fractional_solution(self, fractional_solution):
        if not fractional_solution:
            lightweight = ()
        else:
            lightweight = tuple(
                heapq.nlargest(20, fractional_solution.items(), key=lambda kv: abs(kv[1]))
            )
        self._append_with_cap(self.fractional_solutions, lightweight, self._fractional_history_cap)


    def _record_subgradient(self, value):
        self._append_with_cap(
            self.subgradients,
            float(value),
            self._subgradient_history_cap,
        )



    def custom_kruskal(self, modified_weights):
        uf = UnionFind(self.num_nodes)
        mst_edges = []
        mst_cost = 0.0

        for edge_idx in self.fixed_edge_indices:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += modified_weights[edge_idx]
            else:
                return float('inf'), float('inf'), []

        edge_indices = [i for i in range(len(self.edges)) 
                        if i not in self.fixed_edge_indices and i not in self.excluded_edge_indices]
        sorted_edges = sorted(edge_indices, key=lambda i: modified_weights[i])

        for edge_idx in sorted_edges:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += modified_weights[edge_idx]

        if uf.count_components() > 1 or len(set(u for u, _ in mst_edges) | set(v for _, v in mst_edges)) < self.num_nodes:
            return float('inf'), float('inf'), []

        mst_length = sum(self.edge_lengths[self.edge_indices[(u, v)]] 
                         for u, v in mst_edges)

        return mst_cost, mst_length, mst_edges
    
  
    def incremental_kruskal(self, prev_weights, prev_mst_edges, current_weights):
        uf = UnionFind(self.num_nodes)
        mst_edges = []
        mst_cost = 0.0
        mst_length = 0.0

        for edge_idx in self.fixed_edge_indices:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost   += current_weights[edge_idx]
                mst_length += self.edge_lengths[edge_idx]
            else:
                # Fixed edges already create a cycle -> infeasible
                return float('inf'), float('inf'), []

        weight_changes = current_weights - prev_weights
        changed_indices = np.where(np.abs(weight_changes) > self.cache_tolerance)[0]
        changed_edges   = set(changed_indices)

        prev_mst_indices = {
            self.edge_indices[(u, v)] for u, v in prev_mst_edges
            if self.edge_indices[(u, v)] not in self.fixed_edge_indices
        }
        candidate_indices = (
            prev_mst_indices | changed_edges
        ) - self.excluded_edge_indices - self.fixed_edge_indices

        sorted_edges = sorted(candidate_indices, key=lambda i: current_weights[i])

        for edge_idx in sorted_edges:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost   += current_weights[edge_idx]
                mst_length += self.edge_lengths[edge_idx]

        # NEW: cheap validity check – tree must have exactly n-1 edges
        if len(mst_edges) != self.num_nodes - 1:
            return float('inf'), float('inf'), []

        return mst_cost, mst_length, mst_edges

    
    def compute_mst(self, modified_edges=None):
        start_time = time()
        
        if modified_edges is not None:
            weights = np.array([w for _, _, w in modified_edges], dtype=float)
        else:
            weights = self.compute_modified_weights()

        weights = np.nan_to_num(
            weights,
            nan=0.0,
            posinf=1e9,
            neginf=-1e9,
            copy=False,
        )

        # ---- SIMPLE VERSION: NO HASHING, NO CACHE ----
        mst_cost, mst_length, mst_edges = self.custom_kruskal(weights)
        if self.verbose:
            print(f"MST computed (no cache): length={mst_length:.2f}")

        # Optionally remember last MST if you use it elsewhere
        self.last_mst_edges = mst_edges

        end_time = time()
        LagrangianMST.total_compute_time += end_time - start_time
        return mst_cost, mst_length, mst_edges



    def compute_mst_incremental(self, prev_weights, prev_mst_edges):
        # Compute current modified weights ONCE
        current_weights = self.compute_modified_weights()
        # Cache them so the caller (solve) can reuse without recomputing
        self._last_mw = current_weights

        # First call or no previous MST: just run full Kruskal
        if prev_weights is None or prev_mst_edges is None:
            if self.verbose:
                print("Incremental MST: no previous MST, using full custom_kruskal")
            return self.custom_kruskal(current_weights)

        weight_changes = current_weights - prev_weights

        if np.all(np.abs(weight_changes) < 1e-6):
            mst_cost = sum(current_weights[self.edge_indices[(u, v)]]
                           for u, v in prev_mst_edges)
            mst_length = sum(self.edge_lengths[self.edge_indices[(u, v)]]
                             for u, v in prev_mst_edges)
            if self.verbose:
                print(f"Incremental MST: Reusing previous MST with length={mst_length:.2f}")
            return mst_cost, mst_length, prev_mst_edges

        if self.verbose:
            print(f"Incremental MST: Computing new MST due to weight changes")
        return self.incremental_kruskal(prev_weights, prev_mst_edges, current_weights)



   

    def solve(self, inherited_cuts=None, inherited_multipliers=None, depth=0, node=None):
        start_time = time()
        self.depth = depth
        
        # --- robust normalization of inherited_cuts (accept pairs or indices) ---
        edge_indices = self.edge_indices
        idx_to_edge = {j: e for e, j in edge_indices.items()}

        def _norm_edge(e):
            if not (isinstance(e, tuple) and len(e) == 2):
                return None
            u, v = e
            t = (u, v) if u <= v else (v, u)
            return t if t in edge_indices else None

        def _iter_edges_any(cut_like):
            if isinstance(cut_like, tuple) and len(cut_like) == 2:
                e = _norm_edge(cut_like); 
                if e is not None: yield e
                return
            if isinstance(cut_like, int):
                e = _norm_edge(idx_to_edge.get(int(cut_like)))
                if e is not None: yield e
                return
            try:
                for item in cut_like:
                    if isinstance(item, int):
                        e = _norm_edge(idx_to_edge.get(int(item)))
                    elif isinstance(item, tuple) and len(item) == 2:
                        e = _norm_edge(item)
                    elif isinstance(item, (list, set, frozenset)) and len(item) == 2:
                        a, b = tuple(item); e = _norm_edge((a, b))
                    else:
                        e = None
                    if e is not None: 
                        yield e
            except TypeError:
                return

        def _norm_pair(pair):
            cut_like, rhs_like = pair
            return (set(_iter_edges_any(cut_like)), int(rhs_like))

        if inherited_cuts:
            self.best_cuts = [_norm_pair(p) for p in inherited_cuts]
            self.best_cut_multipliers = (inherited_multipliers or {}).copy()
        else:
            self.best_cuts = []
            self.best_cut_multipliers = {}
        self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()


        # --- robust normalization of inherited_cuts (accept pairs or indices) ---
    
            
        prev_weights = None
        prev_mst_edges = None

       
        if self.use_bisection:
        # Validate graph and edges
            if not self.edges or not nx.is_connected(self.graph):
                if self.verbose:
                    print(f"Error at depth {depth}: Empty edge list or disconnected graph in bisection path")
                return self.best_lower_bound, self.best_upper_bound, []
            


        else:  # Subgradient method with Polyak hybrid + cover cuts (λ, μ), depth-based freezing, and saying that is consitent with modified weight
       # increasing the number of iterations in the root, asliiiiiiiiiii tariiiiiiin asliiiiiiiiiitarin
            # --- Tunables / safety limits ---
            import numpy as np

            MAX_SOLUTIONS    = getattr(self, "max_primal_solutions", 50)
            CLEANUP_INTERVAL = getattr(self, "cleanup_interval", 100)
            max_iter         = min(self.max_iter, 200)

            # Polyak / momentum for λ
            self.momentum_beta = getattr(self, "momentum_beta", 0.9)
            gamma_base         = getattr(self, "gamma_base", 0.1)

            # μ update parameters (you can tune these later if needed)
            gamma_mu         = getattr(self, "gamma_mu", 0.30)
            mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
            eps              = 1e-12

            # Depth-based behaviour
            max_cut_depth = getattr(self, "max_cut_depth", 30)                 # where we ADD cuts
            max_mu_depth  = getattr(self, "max_mu_depth", 50)      # where we UPDATE μ / use cuts in dual
            is_root       = (depth == 0)

            # Node-level separation parameters
            max_active_cuts           = getattr(self, "max_active_cuts", 5)
            max_new_cuts_per_node     = getattr(self, "max_new_cuts_per_node", 5)
            min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
            dead_mu_threshold         = getattr(self, "dead_mu_threshold", 1e-6)

            # Extra iterations allowed at root while we haven't seen a violating MST
            root_max_iter = int(getattr(self, "root_max_iter", max_iter * 2))

            # Ensure cut structures exist
            if not hasattr(self, "best_cuts"):
                self.best_cuts = []   # list of (set(edges), rhs)
            if not hasattr(self, "best_cut_multipliers"):
                self.best_cut_multipliers = {}  # μ_i for each cut
            if not hasattr(self, "best_cut_multipliers_for_best_bound"):
                self.best_cut_multipliers_for_best_bound = {}  # μ at best LB

            # Which behaviour at this node?
            cutting_active_here = self.use_cover_cuts and ((depth <= max_cut_depth) ) # can ADD cuts
            mu_dynamic_here     = self.use_cover_cuts and ((depth <= max_mu_depth))   # can UPDATE μ / use in dual
            cuts_present_here   = self.use_cover_cuts and bool(self.best_cuts)

            # IMPORTANT: if cuts are present but this node is *not* allowed to update μ,
            # we ignore μ in the dual here by zeroing them. Cuts still used for pruning.
            # if cuts_present_here and not mu_dynamic_here:
            #     for i in list(self.best_cut_multipliers.keys()):
            #         self.best_cut_multipliers[i] = 0.0

            # Ensure λ starts in a reasonable range (consistent with compute_modified_weights)
            self.lmbda = max(0.0, min(getattr(self, "lmbda", 0.9), 1e4))

            # no_improvement_count = 0
            polyak_enabled       = True

            # Collect newly generated cuts at this node
            node_new_cuts = []

            # --- Quick guards ---
            if not self.edge_list or self.num_nodes <= 1:
                if self.verbose:
                    print(f"Error at depth {depth}: Empty edge list or invalid graph")
                end_time = time()
                LagrangianMST.total_compute_time += end_time - start_time
                return self.best_lower_bound, self.best_upper_bound, node_new_cuts

            # Fixed / forbidden edges
            F_in  = getattr(self, "fixed_edges", set())
            F_out = getattr(self, "excluded_edges", set())
            edge_idx = self.edge_indices

            # Root: flag to delay separation until we see an MST that violates the budget
            root_pending_sep = False

            # ------------------------------------------------------------------
            # 1) PRE-SEPARATION AT NODE START (dynamic cut-generation nodes only)
            #    - Generate cuts only if REAL-weight MST violates the budget.
            #    - At root, if real MST is feasible, delay separation to first
            #      violating MST (under λ+μ) inside the subgradient iterations.
            # ------------------------------------------------------------------
            if cutting_active_here:
                try:

                    if depth ==0:
                                            # Build MST with REAL weights only for cut generation
                        modified_edges_real = [(0, 0, float(w)) for w in self.edge_weights]
                        try:
                           _, mst_len0, mst_edges0 = self.compute_mst(modified_edges_real)
                        except Exception:
                            # Fallback: use default (λ,μ)-priced MST if something goes wrong
                            _, mst_len0, mst_edges0 = self.compute_mst()
                    else:
                        try:
                            # This uses compute_modified_weights() internally, i.e. w + λ·ℓ + μ
                            _, mst_len0, mst_edges0 = self.compute_mst()
                        except Exception:
                           _, mst_len0, mst_edges0 = float('inf'), float('inf'), []

                    # NOTE: we do NOT set pre_mst_available here, because this MST
                    # is not associated with the current λ, μ. It is used purely
                    # for cut generation.
                    if mst_len0 > self.budget:
                        # Only now it makes sense to separate: MST violates knapsack
                        cand_cuts = self.generate_cover_cuts(mst_edges0) or []

                        T0 = set(mst_edges0)
                        scored = []
                        for cut, rhs in cand_cuts:
                            S   = set(cut)
                            lhs = len(T0 & S)
                            violation = lhs - rhs
                            if violation >= min_cut_violation_for_add:
                                scored.append((violation, S, rhs))

                        scored.sort(reverse=True, key=lambda t: t[0])


                        available_slots = max(0, max_active_cuts - len(self.best_cuts))
                        if available_slots > 0:
                            scored = scored[:min(max_new_cuts_per_node, available_slots)]
                        else:
                            scored = []

                        existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

                        for violation, S, rhs in scored:
                            fz = frozenset(S)
                            if fz in existing:
                                old_rhs = existing[fz]
                                if rhs < old_rhs:
                                    idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
                                    self.best_cuts[idx] = (set(S), rhs)
                                    existing[fz] = rhs
                                continue

                            # new cut
                            self.best_cuts.append((set(S), rhs))
                            idx_new = len(self.best_cuts) - 1
                            self.best_cut_multipliers[idx_new] = 0.00001
                            self.best_cut_multipliers_for_best_bound[idx_new] = 0.00001
                            node_new_cuts.append((set(S), rhs))


                        cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

                    else:
                        # Real-weight pre-MST is feasible: at root we delay separation and wait
                        # for a violating MST (under λ+μ) inside the subgradient iterations.
                        if is_root:
                            root_pending_sep = True

                except Exception as e:
                    if self.verbose:
                        print(f"Error in pre-separation at depth {depth}: {e}")

            # ------------------------------------------------------------------
            # 2) Compute rhs_eff and detect infeasibility (fixed edges + cuts)
            # ------------------------------------------------------------------
            self._rhs_eff = {}
            if self.use_cover_cuts and self.best_cuts:
                for idx_c, (cut, rhs) in enumerate(self.best_cuts):
                    rhs_eff = int(rhs) - len(cut & F_in)
                    self._rhs_eff[idx_c] = rhs_eff
                    if rhs_eff < 0:
                        # fixed edges already violate this cut -> node infeasible
                        end_time = time()
                        LagrangianMST.total_compute_time += end_time - start_time
                        return float('inf'), self.best_upper_bound, node_new_cuts

            # ------------------------------------------------------------------
            # 3) Trim number of cuts (keep at most max_active_cuts)
            # ------------------------------------------------------------------
            if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
                parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
                if not parent_mu_map:
                    parent_mu_map = self.best_cut_multipliers

                idx_and_cut = list(enumerate(self.best_cuts))
                idx_and_cut.sort(
                    key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
                    reverse=True
                )
                idx_and_cut = idx_and_cut[:max_active_cuts]

                new_cuts_list = []
                new_mu       = {}
                new_mu_best  = {}
                new_rhs_eff  = {}

                for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
                    new_cuts_list.append(cut_rhs)
                    new_mu[new_i]      = parent_mu_map.get(old_i, 0.0)
                    new_mu_best[new_i] = parent_mu_map.get(old_i, 0.0)
                    new_rhs_eff[new_i] = self._rhs_eff[old_i]

                self.best_cuts                       = new_cuts_list
                self.best_cut_multipliers            = new_mu
                self.best_cut_multipliers_for_best_bound = new_mu_best
                self._rhs_eff                        = new_rhs_eff

            cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

            # ------------------------------------------------------------------
            # 4) Build cut -> edge index arrays (for pricing/subgradients)
            # ------------------------------------------------------------------
            def _rebuild_cut_structures():
                nonlocal cut_edge_idx_free, cut_edge_idx_all, rhs_vec

                cut_edge_idx_free = []
                cut_edge_idx_all  = []

                for cut, rhs in self.best_cuts:
                    # Free indices (not fixed in/out)
                    idxs_free = [
                        edge_idx[e] for e in cut
                        if (e not in F_in and e not in F_out) and (e in edge_idx)
                    ]
                    arr_free = (
                        np.fromiter(idxs_free, dtype=np.int32)
                        if idxs_free else np.empty(0, dtype=np.int32)
                    )
                    cut_edge_idx_free.append(arr_free)

                    # All indices (for μ-subgradient)
                    idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
                    arr_all  = (
                        np.fromiter(idxs_all, dtype=np.int32)
                        if idxs_all else np.empty(0, dtype=np.int32)
                    )
                    cut_edge_idx_all.append(arr_all)

                # store for compute_modified_weights
                self._cut_edge_idx      = cut_edge_idx_free
                self._cut_edge_idx_all  = cut_edge_idx_all

                if self.best_cuts:
                    # rhs_eff_vec = np.array(
                    #     [self._rhs_eff[i] for i in range(len(cut_edge_idx_free))],
                    #     dtype=float
                    # )
                    rhs_vec = np.array(
                        [rhs for (_, rhs) in self.best_cuts],
                        dtype=float
                    )

                else:
                    # rhs_eff_vec = np.zeros(0, dtype=float)
                    rhs_vec     = np.zeros(0, dtype=float)


            cut_edge_idx_free = []
            cut_edge_idx_all  = []
            # rhs_eff_vec       = np.zeros(0, dtype=float)
            rhs_vec           = np.zeros(0, dtype=float)
            if self.use_cover_cuts and self.best_cuts and mu_dynamic_here:
                _rebuild_cut_structures()

            # Track usefulness of cuts at this node
            max_cut_violation = [0.0 for _ in self.best_cuts]

            # Histories / caches
            self._mw_cached = None
            self._mw_lambda = None
            self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)

            if not hasattr(self, "subgradients"):
                self.subgradients = []
            if not hasattr(self, "step_sizes"):
                self.step_sizes = []
            if not hasattr(self, "multipliers"):
                self.multipliers = []

            prev_weights   = None
            prev_mst_edges = None
            # last_g_lambda  = None

            if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
                self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
            mst_mask = self._mst_mask

            # Decide iteration limit for this node:
            if is_root:
                iter_limit = root_max_iter
            else:
                iter_limit = max_iter

            # ------------------------------------------------------------------
            # 5) Subgradient iterations
            # ------------------------------------------------------------------
            for iter_num in range(int(iter_limit)):
                # 1) MST with current λ, μ
                try:
                    mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
                except Exception:
                    mst_cost, mst_length, mst_edges = self.compute_mst()

                self.last_mst_edges = mst_edges
                prev_mst_edges      = mst_edges

                # 1a) ONE-SHOT delayed separation at root:
                #     If pre-separation didn't create cuts because REAL MST was feasible,
                #     wait until we see a violating MST at the root, then separate once
                #     (now using λ+μ-priced MST).
                if (
                    is_root
                    and cutting_active_here
                    and root_pending_sep
                    and len(self.best_cuts) < max_active_cuts
                    and mst_length > self.budget
                ):
                    try:
                        cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []

                        T_loop = set(mst_edges)
                        scored_loop = []
                        for cut, rhs in cand_cuts_loop:
                            S   = set(cut)
                            lhs = len(T_loop & S)
                            violation = lhs - rhs
                            if violation >= min_cut_violation_for_add:
                                scored_loop.append((violation, S, rhs))

                        scored_loop.sort(reverse=True, key=lambda t: t[0])

                        remaining_slots = max(0, max_active_cuts - len(self.best_cuts))
                        if remaining_slots > 0:
                            scored_loop = scored_loop[:min(max_new_cuts_per_node, remaining_slots)]
                        else:
                            scored_loop = []

                        existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}
                        added_any = False

                        for violation, S, rhs in scored_loop:
                            fz = frozenset(S)
                            if fz in existing:
                                continue
                            self.best_cuts.append((set(S), rhs))
                            new_idx = len(self.best_cuts) - 1
                            self.best_cut_multipliers[new_idx] = 0.00001
                            self.best_cut_multipliers_for_best_bound[new_idx] = 0.00001
                            self._rhs_eff[new_idx] = int(rhs) - len(set(S) & F_in)
                            max_cut_violation.append(0.0)
                            node_new_cuts.append((set(S), rhs))
                            added_any = True

                        if added_any:
                            _rebuild_cut_structures()
                            self._mw_cached = None
                            self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)

                    except Exception as e:
                        if self.verbose:
                            print(f"Error in root delayed separation at depth {depth}, iter {iter_num}: {e}")
                    finally:
                        # Do this at most once at the root
                        root_pending_sep = False

                # Prepare weights for next iteration (cache)
                # prev_weights = self.compute_modified_weights()
                prev_weights = getattr(self, "_last_mw", prev_weights)


                # 2) Primal & UB
                is_feasible = (mst_length <= self.budget)
                self._record_primal_solution(self.last_mst_edges, is_feasible)
                
                if is_feasible:
                    try:
                        real_weight, real_length = self.compute_real_weight_length()
                        if (
                            not math.isnan(real_weight)
                            and not math.isinf(real_weight)
                            and real_weight < self.best_upper_bound
                        ):
                            self.best_upper_bound = real_weight
                    except Exception as e:
                        if self.verbose:
                            print(f"Error updating primal solution: {e}")

                if len(self.primal_solutions) > MAX_SOLUTIONS:
                    # self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
                    self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
                    # self.step_sizes       = self.step_sizes[-MAX_SOLUTIONS:]

                # 3) Dual value: L(λ, μ) = MST_cost - λ B - Σ μ_i rhs_i
                # IMPORTANT: use λ and μ in the SAME clamped form as compute_modified_weights
                lam_for_dual = max(0.0, min(self.lmbda, 1e4))

                if self.use_cover_cuts and len(rhs_vec) > 0:
                    mu_vec = np.fromiter(
                        (
                            max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
                            for i in range(len(rhs_vec))
                        ),
                        dtype=float,
                        count=len(rhs_vec),
                    )
                    cover_cut_penalty = float(mu_vec @ rhs_vec)
                else:
                    mu_vec = np.zeros(len(rhs_vec), dtype=float)
                    cover_cut_penalty = 0.0

                lagrangian_bound = mst_cost - lam_for_dual * self.budget - cover_cut_penalty
                
                
                if (
                    not math.isnan(lagrangian_bound)
                    and not math.isinf(lagrangian_bound)
                    and abs(lagrangian_bound) < 1e10
                ):
                    if lagrangian_bound > self.best_lower_bound + 1e-6:
                        self.best_lower_bound = lagrangian_bound
                        self.best_lambda      = lam_for_dual
                        self.best_mst_edges   = self.last_mst_edges
                        self.best_cost        = mst_cost
                        self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
                        # no_improvement_count  = 0
                    # else:
                    #     # no_improvement_count += 1
                    #     if no_improvement_count % CLEANUP_INTERVAL == 0:
                    #         self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
                # else:
                #     no_improvement_count += 1

                # 4) Subgradients
                knapsack_subgradient = float(mst_length - self.budget)

                if cuts_present_here and mu_dynamic_here and len(cut_edge_idx_all) > 0:
                    mst_mask[:] = False
                    for e in mst_edges:
                        j = self.edge_indices.get(e)
                        if j is not None:
                            mst_mask[j] = True

                    violations = []
                    for i, idxs_all in enumerate(cut_edge_idx_all):
                        lhs_i = int(mst_mask[idxs_all].sum()) if idxs_all.size else 0
                        g_i   = lhs_i - rhs_vec[i]
                        violations.append(g_i)
                        if g_i > max_cut_violation[i]:
                            max_cut_violation[i] = g_i
                    violations       = np.array(violations, dtype=float)
                    cut_subgradients = violations.tolist()
                else:
                    cut_subgradients = []

                # self.subgradients.append(knapsack_subgradient)
                norm_sq = knapsack_subgradient ** 2
                for g in cut_subgradients:
                    norm_sq += g ** 2

                # Polyak step size
                if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
                    gap   = max(0.0, self.best_upper_bound - lagrangian_bound)
                    theta = gamma_base

                    alpha = theta * gap / (norm_sq + eps)

                else:
                    alpha = getattr(self, "step_size", 1e-3)


                # λ update with momentum, then clamp it
                v_prev = getattr(self, "_v_lambda", 0.0)
                v_new  = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
                self._v_lambda = v_new
                self.lmbda     = self.lmbda + alpha * v_new
                
                if self.lmbda < 0.0:
                    self.lmbda = 0.0
                if self.lmbda > 1e4:
                    self.lmbda = 1e4

                # μ updates only at μ-dynamic nodes, then clamp them
                if mu_dynamic_here and len(cut_subgradients) > 0:
                    for i, g in enumerate(cut_subgradients):
                        delta = gamma_mu * alpha * g
                        if mu_increment_cap is not None:
                            if delta > mu_increment_cap:
                                delta = mu_increment_cap
                            elif delta < -mu_increment_cap:
                                delta = -mu_increment_cap

                        mu_old = self.best_cut_multipliers.get(i, 0.0)
                        mu_new = mu_old + delta
                        if mu_new < 0.0:
                            mu_new = 0.0
                        if mu_new > 1e4:
                            mu_new = 1e4
                        self.best_cut_multipliers[i] = mu_new
                        # print("self.gg", self.best_cut_multipliers)

                self.step_sizes.append(alpha)
                self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))


                
            # ------------------------------------------------------------------
            # 6) Drop "dead" cuts globally (never violated & μ tiny historically)
            #     - now more conservative: drop only if
            #       * never violated here
            #       * AND both current and historical μ are tiny
            # ------------------------------------------------------------------
            if self.use_cover_cuts and self.best_cuts and mu_dynamic_here:
                keep_indices = []

                parent_mu_map = getattr(
                    self,
                    "best_cut_multipliers_for_best_bound",
                    self.best_cut_multipliers,
                )

                for i, (cut, rhs) in enumerate(self.best_cuts):
                    mu_i    = float(self.best_cut_multipliers.get(i, 0.0))
                    mu_hist = float(parent_mu_map.get(i, 0.0))

                    ever_useful = (i < len(max_cut_violation) and max_cut_violation[i] > 0.0) \
                                or (abs(mu_hist) >= dead_mu_threshold)

                    # Only drop if never useful AND both current and historical μ are tiny
                    if (not ever_useful) and abs(mu_i) < dead_mu_threshold and abs(mu_hist) < dead_mu_threshold:
                        continue
                    keep_indices.append(i)

                if len(keep_indices) < len(self.best_cuts):
                    new_best_cuts = []
                    new_mu        = {}
                    new_mu_best   = {}
                    new_rhs_eff   = {}

                    for new_idx, old_idx in enumerate(keep_indices):
                        new_best_cuts.append(self.best_cuts[old_idx])
                        new_mu[new_idx]      = float(self.best_cut_multipliers.get(old_idx, 0.0))
                        new_mu_best[new_idx] = float(
                            self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0)
                        )
                        new_rhs_eff[new_idx] = self._rhs_eff[old_idx]

                    self.best_cuts                       = new_best_cuts
                    self.best_cut_multipliers            = new_mu
                    self.best_cut_multipliers_for_best_bound = new_mu_best
                    self._rhs_eff                        = new_rhs_eff

            # ------------------------------------------------------------------
            # 7) Restore best (λ, μ) to pass to children
            # ------------------------------------------------------------------
            if hasattr(self, "best_lambda"):
                self.lmbda = self.best_lambda

            if mu_dynamic_here and hasattr(self, "best_cut_multipliers_for_best_bound"):
                self.best_cut_multipliers = self.best_cut_multipliers_for_best_bound.copy()
            # elif not mu_dynamic_here:
            #     # keep μ = 0 for deep nodes
            #     for i in list(self.best_cut_multipliers.keys()):
            #         self.best_cut_multipliers[i] = 0.0

            end_time = time()
            LagrangianMST.total_compute_time += end_time - start_time
            return self.best_lower_bound, self.best_upper_bound, node_new_cuts
               

    #     else:  # Subgradient method with Polyak hybrid + cover cuts (λ, μ), depth-based freezing, and saying that is consitent with modified weight
    #    # increasing the number of iterations in the root, asliiiiiiiiiii tariiiiiiiiiiiin asliiiiiiiiiitarin with early stop
    #         # --- Tunables / safety limits ---
    #         import numpy as np

    #         MAX_SOLUTIONS    = getattr(self, "max_primal_solutions", 50)
    #         CLEANUP_INTERVAL = getattr(self, "cleanup_interval", 100)
    #         max_iter         = min(self.max_iter, 200)

    #         # NEW: how many iterations we tolerate with no LB improvement
    #         max_no_improvement = getattr(self, "max_no_improvement", 50)
    #         min_iters_before_early_stop = getattr(self, "min_iters_before_early_stop", 5)

    #         # Polyak / momentum for λ
    #         self.momentum_beta = getattr(self, "momentum_beta", 0.9)
    #         gamma_base         = getattr(self, "gamma_base", 0.1)

    #         # μ update parameters (you can tune these later if needed)
    #         gamma_mu         = getattr(self, "gamma_mu", 0.30)
    #         mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
    #         eps              = 1e-12

    #         # Depth-based behaviour
    #         max_cut_depth = getattr(self, "max_cut_depth", 3)                 # where we ADD cuts
    #         max_mu_depth  = getattr(self, "max_mu_depth", 5)      # where we UPDATE μ / use cuts in dual
    #         is_root       = (depth == 0)

    #         # Node-level separation parameters
    #         max_active_cuts           = getattr(self, "max_active_cuts", 5)
    #         max_new_cuts_per_node     = getattr(self, "max_new_cuts_per_node", 5)
    #         min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
    #         dead_mu_threshold         = getattr(self, "dead_mu_threshold", 1e-6)

    #         # Extra iterations allowed at root while we haven't seen a violating MST
    #         root_max_iter = int(getattr(self, "root_max_iter", max_iter * 2))

    #         # Ensure cut structures exist
    #         if not hasattr(self, "best_cuts"):
    #             self.best_cuts = []   # list of (set(edges), rhs)
    #         if not hasattr(self, "best_cut_multipliers"):
    #             self.best_cut_multipliers = {}  # μ_i for each cut
    #         if not hasattr(self, "best_cut_multipliers_for_best_bound"):
    #             self.best_cut_multipliers_for_best_bound = {}  # μ at best LB

    #         # Which behaviour at this node?
    #         cutting_active_here = self.use_cover_cuts and (depth <= max_cut_depth)   # can ADD cuts
    #         mu_dynamic_here     = self.use_cover_cuts and (depth <= max_mu_depth)    # can UPDATE μ / use in dual
    #         cuts_present_here   = self.use_cover_cuts and bool(self.best_cuts)

    #         # Ensure λ starts in a reasonable range (consistent with compute_modified_weights)
    #         self.lmbda = max(0.0, min(getattr(self, "lmbda", 0.05), 1e4))

    #         no_improvement_count = 0
    #         polyak_enabled       = True

    #         # Collect newly generated cuts at this node
    #         node_new_cuts = []

    #         # --- Quick guards ---
    #         if not self.edge_list or self.num_nodes <= 1:
    #             if self.verbose:
    #                 print(f"Error at depth {depth}: Empty edge list or invalid graph")
    #             end_time = time()
    #             LagrangianMST.total_compute_time += end_time - start_time
    #             return self.best_lower_bound, self.best_upper_bound, node_new_cuts

    #         # Fixed / forbidden edges
    #         F_in  = getattr(self, "fixed_edges", set())
    #         F_out = getattr(self, "excluded_edges", set())
    #         edge_idx = self.edge_indices

    #         # Root: flag to delay separation until we see an MST that violates the budget
    #         root_pending_sep = False

    #         # ------------------------------------------------------------------
    #         # 1) PRE-SEPARATION AT NODE START (dynamic cut-generation nodes only)
    #         # ------------------------------------------------------------------
    #         if cutting_active_here:
    #             try:

    #                 if depth ==0:
    #                                         # Build MST with REAL weights only for cut generation
    #                     modified_edges_real = [(0, 0, float(w)) for w in self.edge_weights]
    #                     try:
    #                        _, mst_len0, mst_edges0 = self.compute_mst(modified_edges_real)
    #                     except Exception:
    #                         # Fallback: use default (λ,μ)-priced MST if something goes wrong
    #                         _, mst_len0, mst_edges0 = self.compute_mst()
    #                 else:
    #                     try:
    #                         # This uses compute_modified_weights() internally, i.e. w + λ·ℓ + μ
    #                         _, mst_len0, mst_edges0 = self.compute_mst()
    #                     except Exception:
    #                        _, mst_len0, mst_edges0 = float('inf'), float('inf'), []

    #                 if mst_len0 > self.budget:
    #                     # Only now it makes sense to separate: MST violates knapsack
    #                     cand_cuts = self.generate_cover_cuts(mst_edges0) or []

    #                     T0 = set(mst_edges0)
    #                     scored = []
    #                     for cut, rhs in cand_cuts:
    #                         S   = set(cut)
    #                         lhs = len(T0 & S)
    #                         violation = lhs - rhs
    #                         if violation >= min_cut_violation_for_add:
    #                             scored.append((violation, S, rhs))

    #                     scored.sort(reverse=True, key=lambda t: t[0])

    #                     available_slots = max(0, max_active_cuts - len(self.best_cuts))
    #                     if available_slots > 0:
    #                         scored = scored[:min(max_new_cuts_per_node, available_slots)]
    #                     else:
    #                         scored = []

    #                     existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

    #                     for violation, S, rhs in scored:
    #                         fz = frozenset(S)
    #                         if fz in existing:
    #                             old_rhs = existing[fz]
    #                             if rhs < old_rhs:
    #                                 idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
    #                                 self.best_cuts[idx] = (set(S), rhs)
    #                                 existing[fz] = rhs
    #                             continue

    #                         # new cut
    #                         self.best_cuts.append((set(S), rhs))
    #                         idx_new = len(self.best_cuts) - 1
    #                         self.best_cut_multipliers[idx_new] = 0.00001
    #                         self.best_cut_multipliers_for_best_bound[idx_new] = 0.00001
    #                         node_new_cuts.append((set(S), rhs))

    #                     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

    #                 else:
    #                     # Real-weight pre-MST is feasible: at root we delay separation and wait
    #                     # for a violating MST (under λ+μ) inside the subgradient iterations.
    #                     if is_root:
    #                         root_pending_sep = True

    #             except Exception as e:
    #                 if self.verbose:
    #                     print(f"Error in pre-separation at depth {depth}: {e}")

    #         # ------------------------------------------------------------------
    #         # 2) Compute rhs_eff and detect infeasibility (fixed edges + cuts)
    #         # ------------------------------------------------------------------
    #         self._rhs_eff = {}
    #         if self.use_cover_cuts and self.best_cuts:
    #             for idx_c, (cut, rhs) in enumerate(self.best_cuts):
    #                 rhs_eff = int(rhs) - len(cut & F_in)
    #                 self._rhs_eff[idx_c] = rhs_eff
    #                 if rhs_eff < 0:
    #                     # fixed edges already violate this cut -> node infeasible
    #                     end_time = time()
    #                     LagrangianMST.total_compute_time += end_time - start_time
    #                     return float('inf'), self.best_upper_bound, node_new_cuts

    #         # ------------------------------------------------------------------
    #         # 3) Trim number of cuts (keep at most max_active_cuts)
    #         # ------------------------------------------------------------------
    #         if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
    #             parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
    #             if not parent_mu_map:
    #                 parent_mu_map = self.best_cut_multipliers

    #             idx_and_cut = list(enumerate(self.best_cuts))
    #             idx_and_cut.sort(
    #                 key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
    #                 reverse=True
    #             )
    #             idx_and_cut = idx_and_cut[:max_active_cuts]

    #             new_cuts_list = []
    #             new_mu       = {}
    #             new_mu_best  = {}
    #             new_rhs_eff  = {}

    #             for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
    #                 new_cuts_list.append(cut_rhs)
    #                 new_mu[new_i]      = parent_mu_map.get(old_i, 0.0)
    #                 new_mu_best[new_i] = parent_mu_map.get(old_i, 0.0)
    #                 new_rhs_eff[new_i] = self._rhs_eff[old_i]

    #             self.best_cuts                       = new_cuts_list
    #             self.best_cut_multipliers            = new_mu
    #             self.best_cut_multipliers_for_best_bound = new_mu_best
    #             self._rhs_eff                        = new_rhs_eff

    #         cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

    #         # ------------------------------------------------------------------
    #         # 4) Build cut -> edge index arrays (for pricing/subgradients)
    #         # ------------------------------------------------------------------
    #         def _rebuild_cut_structures():
    #             nonlocal cut_edge_idx_free, cut_edge_idx_all, rhs_vec

    #             cut_edge_idx_free = []
    #             cut_edge_idx_all  = []

    #             for cut, rhs in self.best_cuts:
    #                 # Free indices (not fixed in/out)
    #                 idxs_free = [
    #                     edge_idx[e] for e in cut
    #                     if (e not in F_in and e not in F_out) and (e in edge_idx)
    #                 ]
    #                 arr_free = (
    #                     np.fromiter(idxs_free, dtype=np.int32)
    #                     if idxs_free else np.empty(0, dtype=np.int32)
    #                 )
    #                 cut_edge_idx_free.append(arr_free)

    #                 # All indices (for μ-subgradient)
    #                 idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
    #                 arr_all  = (
    #                     np.fromiter(idxs_all, dtype=np.int32)
    #                     if idxs_all else np.empty(0, dtype=np.int32)
    #                 )
    #                 cut_edge_idx_all.append(arr_all)

    #             # store for compute_modified_weights
    #             self._cut_edge_idx      = cut_edge_idx_free
    #             self._cut_edge_idx_all  = cut_edge_idx_all

    #             if self.best_cuts:
    #                 rhs_vec = np.array(
    #                     [rhs for (_, rhs) in self.best_cuts],
    #                     dtype=float
    #                 )
    #             else:
    #                 rhs_vec = np.zeros(0, dtype=float)


    #         cut_edge_idx_free = []
    #         cut_edge_idx_all  = []
    #         rhs_vec           = np.zeros(0, dtype=float)
    #         if self.use_cover_cuts:
    #             _rebuild_cut_structures()

    #         # Track usefulness of cuts at this node
    #         max_cut_violation = [0.0 for _ in self.best_cuts]

    #         # Histories / caches
    #         self._mw_cached = None
    #         self._mw_lambda = None
    #         self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)

    #         if not hasattr(self, "subgradients"):
    #             self.subgradients = []
    #         if not hasattr(self, "step_sizes"):
    #             self.step_sizes = []
    #         if not hasattr(self, "multipliers"):
    #             self.multipliers = []

    #         prev_weights   = None
    #         prev_mst_edges = None
    #         last_g_lambda  = None

    #         if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
    #             self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
    #         mst_mask = self._mst_mask

    #         # Decide iteration limit for this node:
    #         if is_root and cutting_active_here:
    #             iter_limit = root_max_iter
    #         else:
    #             iter_limit = max_iter

    #         # ------------------------------------------------------------------
    #         # 5) Subgradient iterations
    #         # ------------------------------------------------------------------
    #         for iter_num in range(int(iter_limit)):

    #             # 1) MST with current λ, μ
    #             try:
    #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
    #             except Exception:
    #                 mst_cost, mst_length, mst_edges = self.compute_mst()

    #             self.last_mst_edges = mst_edges
    #             prev_mst_edges      = mst_edges

    #             # delayed separation at root (unchanged) ...
    #             if (
    #                 is_root
    #                 and cutting_active_here
    #                 and root_pending_sep
    #                 and len(self.best_cuts) < max_active_cuts
    #                 and mst_length > self.budget
    #             ):
    #                 try:
    #                     cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []

    #                     T_loop = set(mst_edges)
    #                     scored_loop = []
    #                     for cut, rhs in cand_cuts_loop:
    #                         S   = set(cut)
    #                         lhs = len(T_loop & S)
    #                         violation = lhs - rhs
    #                         if violation >= min_cut_violation_for_add:
    #                             scored_loop.append((violation, S, rhs))

    #                     scored_loop.sort(reverse=True, key=lambda t: t[0])

    #                     remaining_slots = max(0, max_active_cuts - len(self.best_cuts))
    #                     if remaining_slots > 0:
    #                         scored_loop = scored_loop[:min(max_new_cuts_per_node, remaining_slots)]
    #                     else:
    #                         scored_loop = []

    #                     existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}
    #                     added_any = False

    #                     for violation, S, rhs in scored_loop:
    #                         fz = frozenset(S)
    #                         if fz in existing:
    #                             continue
    #                         self.best_cuts.append((set(S), rhs))
    #                         new_idx = len(self.best_cuts) - 1
    #                         self.best_cut_multipliers[new_idx] = 0.00001
    #                         self.best_cut_multipliers_for_best_bound[new_idx] = 0.00001
    #                         self._rhs_eff[new_idx] = int(rhs) - len(set(S) & F_in)
    #                         max_cut_violation.append(0.0)
    #                         node_new_cuts.append((set(S), rhs))
    #                         added_any = True

    #                     if added_any:
    #                         _rebuild_cut_structures()
    #                         self._mw_cached = None
    #                         self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)

    #                 except Exception as e:
    #                     if self.verbose:
    #                         print(f"Error in root delayed separation at depth {depth}, iter {iter_num}: {e}")
    #                 finally:
    #                     root_pending_sep = False

    #             # Prepare weights for next iteration (cache)
    #             prev_weights = getattr(self, "_last_mw", prev_weights)

    #             # 2) Primal & UB
    #             is_feasible = (mst_length <= self.budget)
    #             self._record_primal_solution(self.last_mst_edges, is_feasible)
                
    #             if is_feasible:
    #                 try:
    #                     real_weight, real_length = self.compute_real_weight_length()
    #                     if (
    #                         not math.isnan(real_weight)
    #                         and not math.isinf(real_weight)
    #                         and real_weight < self.best_upper_bound
    #                     ):
    #                         self.best_upper_bound = real_weight
    #                 except Exception as e:
    #                     if self.verbose:
    #                         print(f"Error updating primal solution: {e}")

    #             if len(self.primal_solutions) > MAX_SOLUTIONS:
    #                 self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

    #             # 3) Dual value: L(λ, μ) = MST_cost - λ B - Σ μ_i rhs_i
    #             lam_for_dual = max(0.0, min(self.lmbda, 1e4))

    #             if self.use_cover_cuts and len(rhs_vec) > 0:
    #                 mu_vec = np.fromiter(
    #                     (
    #                         max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
    #                         for i in range(len(rhs_vec))
    #                     ),
    #                     dtype=float,
    #                     count=len(rhs_vec),
    #                 )
    #                 cover_cut_penalty = float(mu_vec @ rhs_vec)
    #             else:
    #                 mu_vec = np.zeros(len(rhs_vec), dtype=float)
    #                 cover_cut_penalty = 0.0

    #             lagrangian_bound = mst_cost - lam_for_dual * self.budget - cover_cut_penalty
                
    #             # -------- LB update + no-improvement counter ----------
    #             if (
    #                 not math.isnan(lagrangian_bound)
    #                 and not math.isinf(lagrangian_bound)
    #                 and abs(lagrangian_bound) < 1e10
    #             ):
    #                 if lagrangian_bound > self.best_lower_bound + 1e-3:
    #                     self.best_lower_bound = lagrangian_bound
    #                     self.best_lambda      = lam_for_dual
    #                     self.best_mst_edges   = self.last_mst_edges
    #                     self.best_cost        = mst_cost
    #                     self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
    #                     no_improvement_count  = 0
    #                 else:
    #                     no_improvement_count += 1
    #                     if no_improvement_count % CLEANUP_INTERVAL == 0:
    #                         self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
    #             else:
    #                 no_improvement_count += 1

    #             # NEW: early stopping based purely on "no LB improvement"
    #             if (
    #                 no_improvement_count >= max_no_improvement
    #                 and iter_num >= min_iters_before_early_stop
    #             ):
    #                 if 1==1:
    #                     print(
    #                         f"Terminating early at depth {depth}, iter {iter_num}: "
    #                         f"no LB improvement for {no_improvement_count} iterations"
    #                     )
    #                 break
    #             # ------------------------------------------------------

    #             # 4) Subgradients
    #             knapsack_subgradient = float(mst_length - self.budget)

    #             if cuts_present_here and mu_dynamic_here and len(cut_edge_idx_all) > 0:
    #                 mst_mask[:] = False
    #                 for e in mst_edges:
    #                     j = self.edge_indices.get(e)
    #                     if j is not None:
    #                         mst_mask[j] = True

    #                 violations = []
    #                 for i, idxs_all in enumerate(cut_edge_idx_all):
    #                     lhs_i = int(mst_mask[idxs_all].sum()) if idxs_all.size else 0
    #                     g_i   = lhs_i - rhs_vec[i]
    #                     violations.append(g_i)
    #                     if g_i > max_cut_violation[i]:
    #                         max_cut_violation[i] = g_i
    #                 violations       = np.array(violations, dtype=float)
    #                 cut_subgradients = violations.tolist()
    #             else:
    #                 cut_subgradients = []

    #             self.subgradients.append(knapsack_subgradient)
    #             norm_sq = knapsack_subgradient ** 2
    #             for g in cut_subgradients:
    #                 norm_sq += g ** 2

    #             # Polyak step size
    #             if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
    #                 gap   = max(0.0, self.best_upper_bound - lagrangian_bound)
    #                 theta = gamma_base
    #                 alpha = theta * gap / (norm_sq + eps)
    #             else:
    #                 alpha = getattr(self, "step_size", 1e-3)

    #             # λ update with momentum, then clamp it
    #             v_prev = getattr(self, "_v_lambda", 0.0)
    #             v_new  = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
    #             self._v_lambda = v_new
    #             self.lmbda     = self.lmbda + alpha * v_new
                
    #             if self.lmbda < 0.0:
    #                 self.lmbda = 0.0
    #             if self.lmbda > 1e4:
    #                 self.lmbda = 1e4

    #             # μ updates only at μ-dynamic nodes, then clamp them
    #             if mu_dynamic_here and len(cut_subgradients) > 0:
    #                 for i, g in enumerate(cut_subgradients):
    #                     delta = gamma_mu * alpha * g
    #                     if mu_increment_cap is not None:
    #                         if delta > mu_increment_cap:
    #                             delta = mu_increment_cap
    #                         elif delta < -mu_increment_cap:
    #                             delta = -mu_increment_cap

    #                     mu_old = self.best_cut_multipliers.get(i, 0.0)
    #                     mu_new = mu_old + delta
    #                     if mu_new < 0.0:
    #                         mu_new = 0.0
    #                     if mu_new > 1e4:
    #                         mu_new = 1e4
    #                     self.best_cut_multipliers[i] = mu_new

    #             self.step_sizes.append(alpha)
    #             self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))


    #             if self.verbose and iter_num % 10 == 0:
    #                 print(
    #                     f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
    #                     f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}, "
    #                     f"cuts={len(self.best_cuts)}, "
    #                     f"no_improv={no_improvement_count}"
    #                 )
            

    #         # ------------------------------------------------------------------
    #         # 6) Drop "dead" cuts globally (never violated & μ tiny historically)
    #         # ------------------------------------------------------------------
    #         if self.use_cover_cuts and self.best_cuts:
    #             keep_indices = []

    #             parent_mu_map = getattr(
    #                 self,
    #                 "best_cut_multipliers_for_best_bound",
    #                 self.best_cut_multipliers,
    #             )

    #             for i, (cut, rhs) in enumerate(self.best_cuts):
    #                 mu_i    = float(self.best_cut_multipliers.get(i, 0.0))
    #                 mu_hist = float(parent_mu_map.get(i, 0.0))

    #                 ever_useful = (i < len(max_cut_violation) and max_cut_violation[i] > 0.0) \
    #                             or (abs(mu_hist) >= dead_mu_threshold)

    #                 if (not ever_useful) and abs(mu_i) < dead_mu_threshold and abs(mu_hist) < dead_mu_threshold:
    #                     continue
    #                 keep_indices.append(i)

    #             if len(keep_indices) < len(self.best_cuts):
    #                 new_best_cuts = []
    #                 new_mu        = {}
    #                 new_mu_best   = {}
    #                 new_rhs_eff   = {}

    #                 for new_idx, old_idx in enumerate(keep_indices):
    #                     new_best_cuts.append(self.best_cuts[old_idx])
    #                     new_mu[new_idx]      = float(self.best_cut_multipliers.get(old_idx, 0.0))
    #                     new_mu_best[new_idx] = float(
    #                         self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0)
    #                     )
    #                     new_rhs_eff[new_idx] = self._rhs_eff[old_idx]

    #                 self.best_cuts                       = new_best_cuts
    #                 self.best_cut_multipliers            = new_mu
    #                 self.best_cut_multipliers_for_best_bound = new_mu_best
    #                 self._rhs_eff                        = new_rhs_eff

    #         # ------------------------------------------------------------------
    #         # 7) Restore best (λ, μ) to pass to children
    #         # ------------------------------------------------------------------
    #         if hasattr(self, "best_lambda"):
    #             self.lmbda = self.best_lambda

    #         if mu_dynamic_here and hasattr(self, "best_cut_multipliers_for_best_bound"):
    #             self.best_cut_multipliers = self.best_cut_multipliers_for_best_bound.copy()

    #         end_time = time()
    #         LagrangianMST.total_compute_time += end_time - start_time
    #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts
        # else:  # Subgradient method with Polyak hybrid + cover cuts (λ, μ), depth-based freezing, and consistent with modified weights
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS    = getattr(self, "max_primal_solutions", 50)
        #     CLEANUP_INTERVAL = getattr(self, "cleanup_interval", 100)
        #     max_iter         = min(self.max_iter, 200)

        #     # NEW: how many iterations we tolerate with no LB improvement (early stop)
        #     max_no_improvement           = getattr(self, "max_no_improvement", 30)
        #     min_iters_before_early_stop  = getattr(self, "min_iters_before_early_stop", 5)

        #     # NEW: λ-shock parameters (escape when stuck but gap is big)
        #     shock_gap_threshold   = getattr(self, "shock_gap_threshold", 0.20)   # relative gap
        #     shock_no_improvement  = getattr(self, "shock_no_improvement", 15)    # flat LB this many iters
        #     shock_max_uses        = getattr(self, "shock_max_uses", 1)           # shocks per node
        #     shock_step_boost      = getattr(self, "shock_step_boost", 10.0)      # multiply step_size by this

        #     # Polyak / momentum for λ
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base         = getattr(self, "gamma_base", 0.1)

        #     # μ update parameters
        #     gamma_mu         = getattr(self, "gamma_mu", 0.30)
        #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
        #     eps              = 1e-12

        #     # Depth-based behaviour
        #     max_cut_depth = getattr(self, "max_cut_depth", 3)   # where we ADD cuts
        #     max_mu_depth  = getattr(self, "max_mu_depth", 5)    # where we UPDATE μ / use cuts in dual
        #     is_root       = (depth == 0)

        #     # Node-level separation parameters
        #     max_active_cuts           = getattr(self, "max_active_cuts", 5)
        #     max_new_cuts_per_node     = getattr(self, "max_new_cuts_per_node", 5)
        #     min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
        #     dead_mu_threshold         = getattr(self, "dead_mu_threshold", 1e-6)

        #     # Extra iterations allowed at root while we haven't seen a violating MST
        #     root_max_iter = int(getattr(self, "root_max_iter", max_iter * 2))

        #     # Ensure cut structures exist
        #     if not hasattr(self, "best_cuts"):
        #         self.best_cuts = []   # list of (set(edges), rhs)
        #     if not hasattr(self, "best_cut_multipliers"):
        #         self.best_cut_multipliers = {}  # μ_i for each cut
        #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers_for_best_bound = {}  # μ at best LB

        #     # Behaviour at this node
        #     cutting_active_here = self.use_cover_cuts and (depth <= max_cut_depth)   # can ADD cuts
        #     mu_dynamic_here     = self.use_cover_cuts and (depth <= max_mu_depth)    # can UPDATE μ / use in dual
        #     cuts_present_here   = self.use_cover_cuts and bool(self.best_cuts)

        #     # Ensure λ starts in a reasonable range
        #     self.lmbda = max(0.0, min(getattr(self, "lmbda", 0.05), 1e4))

        #     no_improvement_count = 0
        #     polyak_enabled       = True
        #     local_shocks_used    = 0  # how many shocks we applied at this node

        #     # Collect newly generated cuts at this node
        #     node_new_cuts = []

        #     # --- Quick guards ---
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        #     # Fixed / forbidden edges
        #     F_in  = getattr(self, "fixed_edges", set())
        #     F_out = getattr(self, "excluded_edges", set())
        #     edge_idx = self.edge_indices

        #     # Root: flag to delay separation until we see an MST that violates the budget
        #     root_pending_sep = False

        #     # ------------------------------------------------------------------
        #     # 1) PRE-SEPARATION AT NODE START (dynamic cut-generation nodes only)
        #     # ------------------------------------------------------------------
        #     if cutting_active_here:
        #         try:
        #             if depth == 0:
        #                 # Build MST with REAL weights only for cut generation
        #                 modified_edges_real = [(0, 0, float(w)) for w in self.edge_weights]
        #                 try:
        #                     _, mst_len0, mst_edges0 = self.compute_mst(modified_edges_real)
        #                 except Exception:
        #                     # Fallback: use default (λ,μ)-priced MST if something goes wrong
        #                     _, mst_len0, mst_edges0 = self.compute_mst()
        #             else:
        #                 try:
        #                     # This uses compute_modified_weights() internally, i.e. w + λ·ℓ + μ
        #                     _, mst_len0, mst_edges0 = self.compute_mst()
        #                 except Exception:
        #                     _, mst_len0, mst_edges0 = float('inf'), float('inf'), []

        #             if mst_len0 > self.budget:
        #                 # Only now it makes sense to separate: MST violates knapsack
        #                 cand_cuts = self.generate_cover_cuts(mst_edges0) or []

        #                 T0 = set(mst_edges0)
        #                 scored = []
        #                 for cut, rhs in cand_cuts:
        #                     S   = set(cut)
        #                     lhs = len(T0 & S)
        #                     violation = lhs - rhs
        #                     if violation >= min_cut_violation_for_add:
        #                         scored.append((violation, S, rhs))

        #                 scored.sort(reverse=True, key=lambda t: t[0])

        #                 available_slots = max(0, max_active_cuts - len(self.best_cuts))
        #                 if available_slots > 0:
        #                     scored = scored[:min(max_new_cuts_per_node, available_slots)]
        #                 else:
        #                     scored = []

        #                 existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #                 for violation, S, rhs in scored:
        #                     fz = frozenset(S)
        #                     if fz in existing:
        #                         old_rhs = existing[fz]
        #                         if rhs < old_rhs:
        #                             idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                             self.best_cuts[idx] = (set(S), rhs)
        #                             existing[fz] = rhs
        #                         continue

        #                     # new cut
        #                     self.best_cuts.append((set(S), rhs))
        #                     idx_new = len(self.best_cuts) - 1
        #                     self.best_cut_multipliers[idx_new] = 0.00001
        #                     self.best_cut_multipliers_for_best_bound[idx_new] = 0.00001
        #                     node_new_cuts.append((set(S), rhs))

        #                 cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #             else:
        #                 # Real-weight pre-MST is feasible: at root we delay separation
        #                 if is_root:
        #                     root_pending_sep = True

        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error in pre-separation at depth {depth}: {e}")

        #     # ------------------------------------------------------------------
        #     # 2) Compute rhs_eff and detect infeasibility (fixed edges + cuts)
        #     # ------------------------------------------------------------------
        #     self._rhs_eff = {}
        #     if self.use_cover_cuts and self.best_cuts:
        #         for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #             rhs_eff = int(rhs) - len(cut & F_in)
        #             self._rhs_eff[idx_c] = rhs_eff
        #             if rhs_eff < 0:
        #                 # fixed edges already violate this cut -> node infeasible
        #                 end_time = time()
        #                 LagrangianMST.total_compute_time += end_time - start_time
        #                 return float('inf'), self.best_upper_bound, node_new_cuts

        #     # ------------------------------------------------------------------
        #     # 3) Trim number of cuts (keep at most max_active_cuts)
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
        #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
        #         if not parent_mu_map:
        #             parent_mu_map = self.best_cut_multipliers

        #         idx_and_cut = list(enumerate(self.best_cuts))
        #         idx_and_cut.sort(
        #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
        #             reverse=True
        #         )
        #         idx_and_cut = idx_and_cut[:max_active_cuts]

        #         new_cuts_list = []
        #         new_mu       = {}
        #         new_mu_best  = {}
        #         new_rhs_eff  = {}

        #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
        #             new_cuts_list.append(cut_rhs)
        #             new_mu[new_i]      = parent_mu_map.get(old_i, 0.0)
        #             new_mu_best[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_rhs_eff[new_i] = self._rhs_eff[old_i]

        #         self.best_cuts                       = new_cuts_list
        #         self.best_cut_multipliers            = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu_best
        #         self._rhs_eff                        = new_rhs_eff

        #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #     # ------------------------------------------------------------------
        #     # 4) Build cut -> edge index arrays (for pricing/subgradients)
        #     # ------------------------------------------------------------------
        #     def _rebuild_cut_structures():
        #         nonlocal cut_edge_idx_free, cut_edge_idx_all, rhs_vec

        #         cut_edge_idx_free = []
        #         cut_edge_idx_all  = []

        #         for cut, rhs in self.best_cuts:
        #             # Free indices (not fixed in/out)
        #             idxs_free = [
        #                 edge_idx[e] for e in cut
        #                 if (e not in F_in and e not in F_out) and (e in edge_idx)
        #             ]
        #             arr_free = (
        #                 np.fromiter(idxs_free, dtype=np.int32)
        #                 if idxs_free else np.empty(0, dtype=np.int32)
        #             )
        #             cut_edge_idx_free.append(arr_free)

        #             # All indices (for μ-subgradient)
        #             idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #             arr_all  = (
        #                 np.fromiter(idxs_all, dtype=np.int32)
        #                 if idxs_all else np.empty(0, dtype=np.int32)
        #             )
        #             cut_edge_idx_all.append(arr_all)

        #         # store for compute_modified_weights
        #         self._cut_edge_idx      = cut_edge_idx_free
        #         self._cut_edge_idx_all  = cut_edge_idx_all

        #         if self.best_cuts:
        #             rhs_vec = np.array(
        #                 [rhs for (_, rhs) in self.best_cuts],
        #                 dtype=float
        #             )
        #         else:
        #             rhs_vec = np.zeros(0, dtype=float)

        #     cut_edge_idx_free = []
        #     cut_edge_idx_all  = []
        #     rhs_vec           = np.zeros(0, dtype=float)
        #     if self.use_cover_cuts:
        #         _rebuild_cut_structures()

        #     # Track usefulness of cuts at this node
        #     max_cut_violation = [0.0 for _ in self.best_cuts]

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     prev_weights   = None
        #     prev_mst_edges = None
        #     last_g_lambda  = None

        #     if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
        #         self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #     mst_mask = self._mst_mask

        #     # Decide iteration limit for this node:
        #     if is_root and cutting_active_here:
        #         iter_limit = root_max_iter
        #     else:
        #         iter_limit = max_iter

        #     # ------------------------------------------------------------------
        #     # 5) Subgradient iterations
        #     # ------------------------------------------------------------------
        #     for iter_num in range(int(iter_limit)):
        #         # 1) MST with current λ, μ
        #         try:
        #             mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #         except Exception:
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges      = mst_edges

        #         # 1a) ONE-SHOT delayed separation at root
        #         if (
        #             is_root
        #             and cutting_active_here
        #             and root_pending_sep
        #             and len(self.best_cuts) < max_active_cuts
        #             and mst_length > self.budget
        #         ):
        #             try:
        #                 cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []

        #                 T_loop = set(mst_edges)
        #                 scored_loop = []
        #                 for cut, rhs in cand_cuts_loop:
        #                     S   = set(cut)
        #                     lhs = len(T_loop & S)
        #                     violation = lhs - rhs
        #                     if violation >= min_cut_violation_for_add:
        #                         scored_loop.append((violation, S, rhs))

        #                 scored_loop.sort(reverse=True, key=lambda t: t[0])

        #                 remaining_slots = max(0, max_active_cuts - len(self.best_cuts))
        #                 if remaining_slots > 0:
        #                     scored_loop = scored_loop[:min(max_new_cuts_per_node, remaining_slots)]
        #                 else:
        #                     scored_loop = []

        #                 existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}
        #                 added_any = False

        #                 for violation, S, rhs in scored_loop:
        #                     fz = frozenset(S)
        #                     if fz in existing:
        #                         continue
        #                     self.best_cuts.append((set(S), rhs))
        #                     new_idx = len(self.best_cuts) - 1
        #                     self.best_cut_multipliers[new_idx] = 0.00001
        #                     self.best_cut_multipliers_for_best_bound[new_idx] = 0.00001
        #                     self._rhs_eff[new_idx] = int(rhs) - len(set(S) & F_in)
        #                     max_cut_violation.append(0.0)
        #                     node_new_cuts.append((set(S), rhs))
        #                     added_any = True

        #                 if added_any:
        #                     _rebuild_cut_structures()
        #                     self._mw_cached = None
        #                     self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)

        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error in root delayed separation at depth {depth}, iter {iter_num}: {e}")
        #             finally:
        #                 root_pending_sep = False

        #         # Prepare weights for next iteration (cache)
        #         prev_weights = getattr(self, "_last_mw", prev_weights)

        #         # 2) Primal & UB
        #         is_feasible = (mst_length <= self.budget)
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
                
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (
        #                     not math.isnan(real_weight)
        #                     and not math.isinf(real_weight)
        #                     and real_weight < self.best_upper_bound
        #                 ):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

        #         # 3) Dual value: L(λ, μ) = MST_cost - λ B - Σ μ_i rhs_i
        #         lam_for_dual = max(0.0, min(self.lmbda, 1e4))

        #         if self.use_cover_cuts and len(rhs_vec) > 0:
        #             mu_vec = np.fromiter(
        #                 (
        #                     max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
        #                     for i in range(len(rhs_vec))
        #                 ),
        #                 dtype=float,
        #                 count=len(rhs_vec),
        #             )
        #             cover_cut_penalty = float(mu_vec @ rhs_vec)
        #         else:
        #             mu_vec = np.zeros(len(rhs_vec), dtype=float)
        #             cover_cut_penalty = 0.0

        #         lagrangian_bound = mst_cost - lam_for_dual * self.budget - cover_cut_penalty
                
        #         # -------- LB update + no-improvement counter ----------
        #         if (
        #             not math.isnan(lagrangian_bound)
        #             and not math.isinf(lagrangian_bound)
        #             and abs(lagrangian_bound) < 1e10
        #         ):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda      = lam_for_dual
        #                 self.best_mst_edges   = self.last_mst_edges
        #                 self.best_cost        = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count  = 0
        #             else:
        #                 no_improvement_count += 1
        #                 if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #         else:
        #             no_improvement_count += 1

        #         # --------- λ-SHOCK: try to escape if stuck & gap large ----------
        #         if (
        #             self.best_upper_bound < float('inf')
        #             and no_improvement_count >= shock_no_improvement
        #             and iter_num >= min_iters_before_early_stop
        #             and local_shocks_used < shock_max_uses
        #         ):
        #             gap = self.best_upper_bound - self.best_lower_bound
        #             denom = max(1.0, abs(self.best_upper_bound))
        #             gap_ratio = max(0.0, gap / denom) if denom > 0 else 0.0

        #             if gap_ratio > shock_gap_threshold:
        #                 if self.verbose:
        #                     print(
        #                         f"Applying λ-shock at depth {depth}, iter {iter_num}: "
        #                         f"gap_ratio={gap_ratio:.3f}, no_improv={no_improvement_count}"
        #                     )

        #                 # Reset λ around the best λ we have so far, if any
        #                 if hasattr(self, "best_lambda"):
        #                     self.lmbda = max(0.0, min(self.best_lambda, 1e4))

        #                 # Reset momentum
        #                 self._v_lambda = 0.0

        #                 # Boost step size to allow a bigger move after the shock
        #                 current_step = getattr(self, "step_size", 1e-3)
        #                 self.step_size = max(current_step * shock_step_boost, 1e-4)

        #                 # Reset counters and caches
        #                 no_improvement_count = 0
        #                 local_shocks_used   += 1
        #                 prev_weights         = None  # force a fresh MST pricing next time

        #         # NEW: early stopping based purely on "no LB improvement"
        #         if (
        #             no_improvement_count >= max_no_improvement
        #             and iter_num >= min_iters_before_early_stop
        #         ):
        #             if self.verbose:
        #                 print(
        #                     f"Terminating early at depth {depth}, iter {iter_num}: "
        #                     f"no LB improvement for {no_improvement_count} iterations"
        #                 )
        #             break
        #         # ------------------------------------------------------

        #         # 4) Subgradients
        #         knapsack_subgradient = float(mst_length - self.budget)

        #         if cuts_present_here and mu_dynamic_here and len(cut_edge_idx_all) > 0:
        #             mst_mask[:] = False
        #             for e in mst_edges:
        #                 j = self.edge_indices.get(e)
        #                 if j is not None:
        #                     mst_mask[j] = True

        #             violations = []
        #             for i, idxs_all in enumerate(cut_edge_idx_all):
        #                 lhs_i = int(mst_mask[idxs_all].sum()) if idxs_all.size else 0
        #                 g_i   = lhs_i - rhs_vec[i]
        #                 violations.append(g_i)
        #                 if g_i > max_cut_violation[i]:
        #                     max_cut_violation[i] = g_i
        #             violations       = np.array(violations, dtype=float)
        #             cut_subgradients = violations.tolist()
        #         else:
        #             cut_subgradients = []

        #         self.subgradients.append(knapsack_subgradient)
        #         norm_sq = knapsack_subgradient ** 2
        #         for g in cut_subgradients:
        #             norm_sq += g ** 2

        #         # Polyak step size
        #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #             gap   = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             theta = gamma_base
        #             alpha = theta * gap / (norm_sq + eps)
        #         else:
        #             alpha = getattr(self, "step_size", 1e-3)

        #         # λ update with momentum, then clamp it
        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new  = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda     = self.lmbda + alpha * v_new
                
        #         if self.lmbda < 0.0:
        #             self.lmbda = 0.0
        #         if self.lmbda > 1e4:
        #             self.lmbda = 1e4

        #         # μ updates only at μ-dynamic nodes, then clamp them
        #         if mu_dynamic_here and len(cut_subgradients) > 0:
        #             for i, g in enumerate(cut_subgradients):
        #                 delta = gamma_mu * alpha * g
        #                 if mu_increment_cap is not None:
        #                     if delta > mu_increment_cap:
        #                         delta = mu_increment_cap
        #                     elif delta < -mu_increment_cap:
        #                         delta = -mu_increment_cap

        #                 mu_old = self.best_cut_multipliers.get(i, 0.0)
        #                 mu_new = mu_old + delta
        #                 if mu_new < 0.0:
        #                     mu_new = 0.0
        #                 if mu_new > 1e4:
        #                     mu_new = 1e4
        #                 self.best_cut_multipliers[i] = mu_new

        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #         # stagnation check for λ (your original logic)
        #         if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if not (is_root and root_pending_sep and cutting_active_here and self.use_cover_cuts):
        #                     if self.verbose:
        #                         print("Terminating early: subgradient stagnation")
        #                     break
        #             current_step = getattr(self, "step_size", 1e-3)
        #             self.step_size = max(1e-8, current_step * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0
        #         last_g_lambda = knapsack_subgradient

        #         if self.verbose and iter_num % 10 == 0:
        #             print(
        #                 f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                 f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}, "
        #                 f"cuts={len(self.best_cuts)}, "
        #                 f"no_improv={no_improvement_count}, shocks={local_shocks_used}"
        #             )

        #     # ------------------------------------------------------------------
        #     # 6) Drop "dead" cuts globally (never violated & μ tiny historically)
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts:
        #         keep_indices = []

        #         parent_mu_map = getattr(
        #             self,
        #             "best_cut_multipliers_for_best_bound",
        #             self.best_cut_multipliers,
        #         )

        #         for i, (cut, rhs) in enumerate(self.best_cuts):
        #             mu_i    = float(self.best_cut_multipliers.get(i, 0.0))
        #             mu_hist = float(parent_mu_map.get(i, 0.0))

        #             ever_useful = (i < len(max_cut_violation) and max_cut_violation[i] > 0.0) \
        #                         or (abs(mu_hist) >= dead_mu_threshold)

        #             if (not ever_useful) and abs(mu_i) < dead_mu_threshold and abs(mu_hist) < dead_mu_threshold:
        #                 continue
        #             keep_indices.append(i)

        #         if len(keep_indices) < len(self.best_cuts):
        #             new_best_cuts = []
        #             new_mu        = {}
        #             new_mu_best   = {}
        #             new_rhs_eff   = {}

        #             for new_idx, old_idx in enumerate(keep_indices):
        #                 new_best_cuts.append(self.best_cuts[old_idx])
        #                 new_mu[new_idx]      = float(self.best_cut_multipliers.get(old_idx, 0.0))
        #                 new_mu_best[new_idx] = float(
        #                     self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0)
        #                 )
        #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]

        #             self.best_cuts                       = new_best_cuts
        #             self.best_cut_multipliers            = new_mu
        #             self.best_cut_multipliers_for_best_bound = new_mu_best
        #             self._rhs_eff                        = new_rhs_eff

        #     # ------------------------------------------------------------------
        #     # 7) Restore best (λ, μ) to pass to children
        #     # ------------------------------------------------------------------
        #     if hasattr(self, "best_lambda"):
        #         self.lmbda = self.best_lambda

        #     if mu_dynamic_here and hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers = self.best_cut_multipliers_for_best_bound.copy()

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts



    def compute_mst_for_lambda(self, lambda_val):
        modified_edges = []
        for i, (u, v) in enumerate(self.edge_list):
            modified_w = self.edge_weights[i] + lambda_val * self.edge_lengths[i]
            for cut_idx, (cut, _) in enumerate(self.best_cuts):
                if (u, v) in cut:
                    modified_w += self.best_cut_multipliers.get(cut_idx, 0)
            modified_edges.append((u, v, modified_w))
        return self.compute_mst(modified_edges)

    def _log_fractional_solution(self, method, edge_weights, msts, elapsed_time):
        if self.verbose:
            total_weight = sum(self.edge_weights[self.edge_indices[e]] * w for e, w in edge_weights.items())
            total_length = sum(self.edge_lengths[self.edge_indices[e]] * w for e, w in edge_weights.items())
            print(f"{method} solution: {len(edge_weights)} edges, "
                  f"weight={total_weight:.2f}, length={total_length:.2f}, time={elapsed_time:.2f}s")
            print(f"MSTs used: {len(msts)}")

    
    
    def compute_dantzig_wolfe_solution(self, node):
        start_time = time()
        
        # Need at least 1 MST
        if len(self.primal_solutions) < 1:
            if self.verbose:
                print("Insufficient primal solutions for Dantzig-Wolfe")
            return None

        # More lenient filtering - just check basic validity
        valid_msts = []
        for mst_edges, is_feasible in self.primal_solutions:
            if not mst_edges:
                continue
            mst_edges_normalized = {tuple(sorted((u, v))) for u, v in mst_edges}
            
            # Basic validity: correct number of edges
            if len(mst_edges_normalized) == self.num_nodes - 1:
                valid_msts.append(mst_edges_normalized)
        
        if len(valid_msts) < 1:
            if self.verbose:
                print(f"No valid MSTs after filtering")
            return None

        # Handle single MST case
        if len(valid_msts) == 1:
            edge_weights = {e: 1.0 for e in valid_msts[0]}
            if self.verbose:
                print(f"Dantzig-Wolfe: Single MST, returning as integral solution")
            return edge_weights

        if self.verbose:
            print(f"Using {len(valid_msts)} valid MSTs for Dantzig-Wolfe")

        # Select diverse MSTs
        max_msts = min(10, len(valid_msts))
        selected_msts = []
        covered_edges = set()
        remaining_msts = valid_msts.copy()
        
        while remaining_msts and len(selected_msts) < max_msts:
            best_mst = None
            best_score = -1
            for mst in remaining_msts:
                new_edges = mst - covered_edges
                score = len(new_edges)
                if score > best_score:
                    best_score = score
                    best_mst = mst
            if best_mst:
                selected_msts.append(best_mst)
                covered_edges.update(best_mst)
                remaining_msts.remove(best_mst)
            else:
                break

        if len(selected_msts) < 2:
            if self.verbose:
                print(f"Only {len(selected_msts)} diverse MSTs selected")
            return None

        num_msts = len(selected_msts)
        edge_indices = self.edge_indices
        
        # Objective: minimize total weight
        c = []
        for mst_edges in selected_msts:
            weight = sum(self.edge_weights[edge_indices[e]] for e in mst_edges)
            c.append(weight + 0.1 * (1.0 / num_msts))

        # Convex combination constraint
        A_eq = [np.ones(num_msts)]
        b_eq = [1.0]

        # Budget as inequality constraint
        A_ub = []
        b_ub = []
        lengths = [sum(self.edge_lengths[edge_indices[e]] for e in mst_edges)
                for mst_edges in selected_msts]
        A_ub.append(lengths)
        b_ub.append(self.budget)

        # Cover cuts (limit to avoid infeasibility)
        if self.best_cuts and len(self.best_cuts) <= 20:
            for cut, rhs in self.best_cuts:
                cut_indices = [edge_indices[e] for e in cut if e in edge_indices]
                if cut_indices:
                    row = np.zeros(num_msts)
                    for k, mst_edges in enumerate(selected_msts):
                        cut_count = sum(1 for e in mst_edges if e in cut)
                        row[k] = cut_count
                    A_ub.append(row)
                    b_ub.append(rhs)

        bounds = [(0, None) for _ in range(num_msts)]

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if not res.success:
                if self.verbose:
                    print(f"LP with cuts failed: {res.message}, trying without cuts")
                # Retry without cover cuts
                res = linprog(c, A_ub=[lengths], b_ub=[self.budget], 
                            A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                if not res.success:
                    if self.verbose:
                        print(f"LP without cuts also failed: {res.message}")
                    return None
            
            lambda_k = res.x
        except Exception as e:
            if self.verbose:
                print(f"LP solver error: {e}")
            return None

        # Build fractional edge solution
        edge_weights = {}
        for u, v in self.edge_list:
            e = (u, v)
            weight = sum(lambda_k[k] for k, mst_edges in enumerate(selected_msts) if e in mst_edges)
            if weight > 1e-6:
                edge_weights[e] = weight

        if self.verbose:
            print(f"Dantzig-Wolfe solution: {len(edge_weights)} edges")
            truly_fractional = sum(1 for w in edge_weights.values() if 0.1 < w < 0.9)
            print(f"  Truly fractional (0.1-0.9): {truly_fractional}/{len(edge_weights)} edges")

        return edge_weights if edge_weights else None   
    def compute_weighted_average_solution(self):
        """Compute a fractional primal solution as a weighted average of MSTs."""
        if not self.primal_solutions or not self.step_sizes:
            if self.verbose:
                print("No primal solutions or step sizes available for weighted average")
            return None

        # Ensure lengths match (subgradient iterations should align)
        if len(self.primal_solutions) != len(self.step_sizes):
            if self.verbose:
                print(f"Mismatch: {len(self.primal_solutions)} primal solutions, {len(self.step_sizes)} step sizes")
            return None

        total_step_sum = sum(self.step_sizes)
        if total_step_sum <= 0:
            if self.verbose:
                print("Total step size sum is zero or negative")
            return None

        edge_weights = defaultdict(float)
        for i, (mst_edges, _) in enumerate(self.primal_solutions):
            lambda_i = self.step_sizes[i]

            weight = lambda_i / total_step_sum
            for e in mst_edges:
                # print("dd", e)
                edge_weights[e] += weight

        # Ensure weights are in [0, 1] (should be automatic but added for robustness)
        for e in edge_weights:
            edge_weights[e] = min(1.0, max(0.0, edge_weights[e]))

        if self.verbose:
            total_weight = sum(self.edge_weights[self.edge_indices[e]] * w for e, w in edge_weights.items())
            total_length = sum(self.edge_lengths[self.edge_indices[e]] * w for e, w in edge_weights.items())
            print(f"Weighted Average Solution: {len(edge_weights)} edges, "
                f"weight={total_weight:.2f}, length={total_length:.2f}")

        return dict(edge_weights) if edge_weights else None

    def recover_primal_solution(self, node):
        start_time = time()

        for mst_edges, is_feasible in self.primal_solutions:
            mst_edges_normalized = {tuple(sorted((u, v))) for u, v in mst_edges}
            if not all(e in mst_edges_normalized for e in node.fixed_edges):
                continue
            if any(e in mst_edges_normalized for e in node.excluded_edges):
                continue

            real_length = sum(self.edge_lengths[self.edge_indices[e]] 
                              for e in mst_edges_normalized)
            if real_length > self.budget:
                continue

            valid_cuts = True
            for cut, rhs in node.active_cuts:
                cut_count = sum(1 for e in mst_edges_normalized if e in cut)
                if cut_count > rhs:
                    valid_cuts = False
                    break
            if not valid_cuts:
                continue

            uf = UnionFind(self.num_nodes)
            for u, v in mst_edges_normalized:
                uf.union(u, v)
            if uf.count_components() != 1 or len(set(u for u, _ in mst_edges_normalized) | set(v for _, v in mst_edges_normalized)) < self.num_nodes:
                continue

            real_weight = sum(self.edge_weights[self.edge_indices[e]] 
                              for e in mst_edges_normalized)
            end_time = time()
            if self.verbose:
                print(f"Feasible primal solution found from primal_solutions: weight={real_weight:.2f}, length={real_length:.2f}")
            return list(mst_edges_normalized), real_weight, real_length

        uf = UnionFind(self.num_nodes)
        mst_edges = []
        total_length = 0.0
        total_weight = 0.0

        for edge_idx in self.fixed_edge_indices:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                total_length += self.edge_lengths[edge_idx]
                total_weight += self.edge_weights[edge_idx]
            else:
                if self.verbose:
                    print(f"Fixed edge ({u}, {v}) creates cycle in greedy heuristic")
                return None, float('inf'), float('inf')

        edge_indices = [i for i in range(len(self.edges)) 
                        if i not in self.fixed_edge_indices and i not in self.excluded_edge_indices]
        sorted_edges = sorted(edge_indices, key=lambda i: self.edge_weights[i])

        for edge_idx in sorted_edges:
            u, v = self.edge_list[edge_idx]
            new_length = total_length + self.edge_lengths[edge_idx]
            if new_length > self.budget:
                continue

            temp_edges = mst_edges + [(u, v)]
            valid_cuts = True
            for cut, rhs in node.active_cuts:
                cut_count = sum(1 for e in temp_edges if e in cut)
                if cut_count > rhs:
                    valid_cuts = False
                    break
            if not valid_cuts:
                continue

            if uf.union(u, v):
                mst_edges.append((u, v))
                total_length = new_length
                total_weight += self.edge_weights[edge_idx]

        if uf.count_components() != 1 or len(set(u for u, _ in mst_edges) | set(v for _, v in mst_edges)) < self.num_nodes:
            if self.verbose:
                print("Greedy heuristic failed to produce a valid spanning tree")
            return None, float('inf'), float('inf')

        end_time = time()
        if self.verbose:
            print(f"Feasible primal solution found via greedy heuristic: weight={total_weight:.2f}, length={total_length:.2f}")
        return mst_edges, total_weight, total_length

    def compute_real_weight_length(self):
        real_weight = sum(self.edge_weights[self.edge_indices[e]] 
                          for e in self.last_mst_edges)
        real_length = sum(self.edge_lengths[self.edge_indices[e]] 
                          for e in self.last_mst_edges)
        return real_weight, real_length
