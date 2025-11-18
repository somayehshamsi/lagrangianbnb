


##################################################
# deep thinking
import networkx as nx
import numpy as np
from time import time
from collections import defaultdict, OrderedDict
from scipy.optimize import linprog  
import math

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
                 initial_lambda=0.05, step_size=0.00001, max_iter=10, p=0.95,
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
        self.p = p
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
        self._fractional_history_cap = 5
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


    # def reset(
    #     self,
    #     *,
    #     fixed_edges=None,
    #     excluded_edges=None,
    #     initial_lambda=None,
    #     step_size=None,
    #     max_iter=None,
    #     use_cover_cuts=None,
    #     cut_frequency=None,
    #     use_bisection=None,
    #     verbose=None,
    # ):
    #     """
    #     Reinitialize this solver to the same state you'd get by constructing a new
    #     LagrangianMST instance with the same graph/arrays but different fixed/excluded sets
    #     and (optionally) different run parameters. No heavy data structures are rebuilt.
    #     """
    #     # --- Problem modifiers ---
    #     if fixed_edges is not None:
    #         self.fixed_edges = {tuple(sorted((u, v))) for u, v in fixed_edges}
    #     else:
    #         self.fixed_edges = set()

    #     if excluded_edges is not None:
    #         self.excluded_edges = {tuple(sorted((u, v))) for u, v in excluded_edges}
    #     else:
    #         self.excluded_edges = set()

    #     self.fixed_edge_indices = {
    #         self.edge_indices.get((u, v)) for u, v in self.fixed_edges
    #         if (u, v) in self.edge_indices
    #     }
    #     self.excluded_edge_indices = {
    #         self.edge_indices.get((u, v)) for u, v in self.excluded_edges
    #         if (u, v) in self.edge_indices
    #     }

    #     if initial_lambda is not None:
    #         self.lmbda = float(initial_lambda)
    #     else:
    #         self.lmbda = getattr(self, "lmbda", 0.05)

    #     if step_size is not None:
    #         self.step_size = float(step_size)
    #     if max_iter is not None:
    #         self.max_iter = int(max_iter)
    #     if use_cover_cuts is not None:
    #         self.use_cover_cuts = bool(use_cover_cuts)
    #     if cut_frequency is not None:
    #         self.cut_frequency = int(cut_frequency)
    #     if use_bisection is not None:
    #         self.use_bisection = bool(use_bisection)
    #     if verbose is not None:
    #         self.verbose = bool(verbose)

    #     self.best_lower_bound = float("-inf")
    #     self.best_upper_bound = float("inf")
    #     self.best_mst_edges = []
    #     self.best_cuts = []
    #     self.best_cut_multipliers = {}
    #     self.multipliers = []

    #     if hasattr(self, "edge_weights") and hasattr(self, "modified_weights"):
    #         self.modified_weights[:] = self.edge_weights

    #     self.last_modified_weights = None
    #     self.last_mst_edges = None

    #     try:
    #         cap = self.mst_cache.capacity
    #     except Exception:
    #         cap = max(20, self.num_nodes * 2)
    #     self.mst_cache = LRUCache(capacity=cap)

    #     self._invalidate_weight_cache()

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
            self.lmbda = getattr(self, "lmbda", 0.05)

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
   

    
    # def generate_cover_cuts(self, mst_edges):
    #     """
    #     Generate cover cuts, keeping only the strongest (highest violation) per unique edge set.
    #     Returns at most MAX_CUTS_PER_TYPE cuts.
    #     """
    #     MAX_CUTS_PER_TYPE = 2  # Limit cuts per category
    #     MAX_TOTAL_CUTS = 3     # Absolute maximum cuts to return
        
    #     if not mst_edges:
    #         if self.verbose:
    #             print("No MST edges provided for cover cut generation.")
    #         return []

    #     edge_data = self.edge_attributes
    #     budget = self.budget
    #     excluded = getattr(self, "excluded_edges", set())
    #     fixed = getattr(self, "fixed_edges", set())

    #     mst_edges_norm = [tuple(sorted((u, v))) for (u, v) in mst_edges]
    #     mst_set = {e for e in mst_edges_norm if e in edge_data}

    #     total_len = sum(edge_data[e][1] for e in mst_set)
    #     if total_len <= budget:
    #         if self.verbose:
    #             print(f"MST length {total_len:.2f} <= budget {budget:.2f}; no cover cuts generated.")
    #         return []

    #     # Helper function (same as your original)
    #     def build_min_cover(sorted_desc_edges):
    #         S, s = [], 0.0
    #         for e in sorted_desc_edges:
    #             if e not in edge_data:
    #                 continue
    #             S.append(e)
    #             s += edge_data[e][1]
    #             if s > budget:
    #                 S.sort(key=lambda x: edge_data[x][1])
    #                 k = 0
    #                 while k < len(S) and (s - edge_data[S[k]][1] > budget):
    #                     s -= edge_data[S[k]][1]
    #                     k += 1
    #                 if k > 0:
    #                     S = S[k:]
    #                 violation = s - budget
    #                 rhs = len(S) - 1
    #                 return set(S), rhs, violation
    #         return None

    #     # Collect ALL potential cuts with their violations
    #     all_cuts = []  # List of (cut_set, rhs, violation, type_name)

    #     # Base order: MST edges by decreasing length
    #     mst_desc = sorted(mst_set, key=lambda e: edge_data[e][1], reverse=True)

    #     # 1. Traditional cover
    #     base_cover = build_min_cover(mst_desc)
    #     if base_cover:
    #         S_trad, rhs_trad, viol_trad = base_cover
    #         all_cuts.append((S_trad, rhs_trad, viol_trad, "traditional"))

    #         # 2. Extended (residual-based)
    #         rho = sum(edge_data[e][1] for e in S_trad) - budget
    #         additional = {
    #             e for e in (tuple(sorted((u, v))) for (u, v) in self.graph.edges)
    #             if e not in S_trad and e not in excluded and e in edge_data 
    #             and edge_data[e][1] >= rho - 1e-12
    #         }
    #         if additional:
    #             S_ext = S_trad | additional
    #             all_cuts.append((S_ext, rhs_trad, viol_trad, "extended"))

    #     # 3. Alternative traditional (skip longest edge)
    #     if len(mst_desc) >= 2:
    #         alt_cover = build_min_cover(mst_desc[1:])
    #         if alt_cover:
    #             S_alt, rhs_alt, viol_alt = alt_cover
    #             all_cuts.append((S_alt, rhs_alt, viol_alt, "traditional_alt"))

    #     # 4. Tighter cut (only if we have a seed)
    #     if base_cover:
    #         S_seed, rhs_seed, viol_seed = base_cover
    #         complement = [
    #             (e[0], e[1], edge_data[e][1])
    #             for e in (tuple(sorted((u, v))) for (u, v) in self.graph.edges)
    #             if e not in S_seed and e not in excluded and e in edge_data
    #         ]
    #         seed_desc = sorted(S_seed, key=lambda e: edge_data[e][1], reverse=True)
            
    #         fixed_use = [e for e in fixed if e not in excluded and e in edge_data]
    #         prev_selected = None
    #         prev_rhs_size = None
    #         prev_sum_trad = None
    #         prev_sum_needed_total = None

    #         for k in range(len(seed_desc) - 1, 0, -1):
    #             selected_edges = seed_desc[:k]
    #             selected_set = set(selected_edges)
    #             sum_trad = sum(edge_data[e][1] for e in selected_edges)

    #             removed_edges = seed_desc[k:]
    #             removed_triplets = [(e[0], e[1], edge_data[e][1]) for e in removed_edges]
    #             current_complement = complement + removed_triplets
    #             current_complement.sort(key=lambda x: x[2])

    #             fixed_extra = [e for e in fixed_use if e not in selected_set]
    #             fixed_extra_len = sum(edge_data[e][1] for e in fixed_extra)
    #             needed = max(0, (self.num_nodes - 1) - k - len(fixed_extra))

    #             if needed > 0:
    #                 sum_needed = (fixed_extra_len + sum(length for _, _, length in current_complement[:needed])
    #                             if len(current_complement) >= needed else float('inf'))
    #             else:
    #                 sum_needed = fixed_extra_len

    #             total_est = sum_trad + sum_needed

    #             if total_est <= budget and prev_selected is not None:
    #                 rhs_tight = prev_rhs_size - 1
    #                 violation = (prev_sum_trad + prev_sum_needed_total) - budget
    #                 if violation > 1e-12:
    #                     all_cuts.append((set(prev_selected), rhs_tight, violation, "tighter"))
    #                 break

    #             prev_selected = list(selected_edges)
    #             prev_rhs_size = k
    #             prev_sum_trad = sum_trad
    #             prev_sum_needed_total = sum_needed
    #         else:
    #             # Never found feasible completion
    #             if prev_selected is not None and prev_sum_needed_total is not None:
    #                 rhs_tight = prev_rhs_size - 1
    #                 violation = (prev_sum_trad + prev_sum_needed_total) - budget
    #                 if violation > 1e-12:
    #                     all_cuts.append((set(prev_selected), rhs_tight, violation, "tighter"))

    #         # 5. Extended tighter
    #         tighter_cuts = [c for c in all_cuts if c[3].startswith("tighter")]
    #         if tighter_cuts:
    #             # Take the best tighter cut
    #             best_tighter = max(tighter_cuts, key=lambda x: x[2])
    #             cset, rhs, viol, _ = best_tighter
    #             rho_t = sum(edge_data[e][1] for e in cset) - budget
    #             add_tight = {
    #                 e for e in (tuple(sorted((u, v))) for (u, v) in self.graph.edges)
    #                 if e not in cset and e not in excluded and e in edge_data 
    #                 and edge_data[e][1] >= rho_t - 1e-12
    #             }
    #             if add_tight:
    #                 S_ext_tight = cset | add_tight
    #                 all_cuts.append((S_ext_tight, rhs, viol, "extended_tighter"))

    #     # ========== CRITICAL FILTERING STEP ==========
    #     # Group cuts by their edge set (frozen)
    #     cut_groups = {}
    #     for cut_set, rhs, violation, cut_type in all_cuts:
    #         fz = frozenset(cut_set)
    #         if fz not in cut_groups:
    #             cut_groups[fz] = []
    #         cut_groups[fz].append((cut_set, rhs, violation, cut_type))

    #     # For each unique edge set, keep only the cut with:
    #     # 1. Highest RHS (tightest constraint)
    #     # 2. If tied, highest violation
    #     best_per_set = []
    #     for fz, cuts in cut_groups.items():
    #         best = max(cuts, key=lambda x: (x[1], x[2]))  # (rhs, violation)
    #         best_per_set.append(best)

    #     # Sort by violation (strongest first) and limit total
    #     best_per_set.sort(key=lambda x: x[2], reverse=True)
    #     selected_cuts = best_per_set[:MAX_TOTAL_CUTS]

    #     if self.verbose:
    #         print(f"Generated {len(all_cuts)} candidate cuts, selected {len(selected_cuts)} strongest:")
    #         for cut_set, rhs, viol, cut_type in selected_cuts:
    #             print(f"  {cut_type}: |cut|={len(cut_set)}, rhs={rhs}, violation={viol:.2f}")

    #     return [(cut, rhs) for cut, rhs, _, _ in selected_cuts]


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
        CONFIRM_MARGIN = 0.05
        MAX_RETURN = 10
        LIFT_PREF_Q = 0.5

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

        # --- (3) τ-lifting (remaining-based, safe) on S_seed ---
        # Only try lifting if base seed is effective and currently violated
        # if S_seed and rhs_eff(S_seed) > 0 and is_violated_now(S_seed):
        #     sum_seed = sum(get_len(e) for e in S_seed)
        #     S_seed_sorted = sorted(S_seed, key=lambda e: get_len(e))
        #     q_index = max(0, min(len(S_seed_sorted) - 1, int(len(S_seed_sorted) * LIFT_PREF_Q)))
        #     len_median = get_len(S_seed_sorted[q_index])

        #     ell_min_A = get_len(A_sorted[0]) if A_sorted else float('inf')

        #     def U_of_subset(subset_list):
        #         k = r_prime - len(subset_list)
        #         if k <= 0:
        #             return 0.0
        #         subset = set(subset_list)
        #         total = 0.0; taken = 0
        #         for e in A_sorted:
        #             if e in subset: continue
        #             total += get_len(e); taken += 1
        #             if taken == k: break
        #         return total if taken == k else float('inf')

        #     tau = -float('inf')
        #     S_list = list(S_seed)
        #     for e_drop in S_list:
        #         S_minus = [e for e in S_list if e is not e_drop]
        #         # guard inf in U_of_subset
        #         u_val = U_of_subset(S_minus)
        #         if u_val == float('inf'):
        #             continue
        #         tau_e = Bp - (sum_seed - get_len(e_drop)) - u_val + ell_min_A
        #         if tau_e > tau:
        #             tau = tau_e

        #     if tau > -float('inf') and A_sorted:
        #         lift_add = {e for e in A
        #                     if e not in S_seed
        #                     and get_len(e) >= len_median - EPS
        #                     and get_len(e) > tau + EPS}
        #         if lift_add:
        #             S_lift = set(S_seed) | lift_add
        #             if rhs_eff(S_lift) > 0 and is_violated_now(S_lift):
        #                 cuts.append((S_lift, len(S_seed) - 1))  # keep rhs = |S_seed|-1 (safe)

        # --- dedup & dominance-aware selection ---
        uniq = {}
        for cset, rhs in cuts:
            key = tuple(sorted(cset))  # stable order for hashing
            best = uniq.get(key)
            if best is None or rhs > best[1] or (rhs == best[1] and len(cset) > len(best[0])):
                uniq[key] = (cset, rhs)

        final = list(uniq.values())
        final.sort(key=lambda t: (t[1], len(t[0])), reverse=True)

        # Subset-dominance pruning on a tiny pool
        kept = []
        for cset, rhs in final:
            # Drop if ineffective (double safety)
            if rhs_eff(cset) <= 0:
                continue
            dominated = any(cset <= dset and rhs <= drhs for dset, drhs in kept)
            if not dominated:
                kept.append((cset, rhs))

        return kept[:MAX_RETURN]


    
    # def compute_modified_weights(self):
    #     import numpy as np

    #     base = self.edge_weights.copy()
    #     lam = max(0.0, min(getattr(self, "lmbda", 0.0), 1e4))
    #     if lam:
    #         base = base + lam * self.edge_lengths

    #     if not (self.use_cover_cuts and self.best_cuts):
    #         self._mw_cached = None
    #         self._mw_lambda = lam
    #         self._mw_mu = None
    #         self._mw_free_mask_key = None
    #         return base

    #     free_mask = self._get_free_mask()
    #     cut_idxs = getattr(self, "_cut_edge_idx", None)
    #     mu_len = len(cut_idxs) if cut_idxs is not None else len(self.best_cuts)
    #     mu = np.array(
    #         [max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4)) for i in range(mu_len)],
    #         dtype=float,
    #     )

    #     cache_hit = (
    #         self._mw_cached is not None
    #         and self._mw_lambda == lam
    #         and self._mw_free_mask_key == self._free_mask_key
    #         and self._mw_mu is not None
    #         and mu.shape == self._mw_mu.shape
    #         and np.array_equal(mu, self._mw_mu)
    #     )
    #     if cache_hit:
    #         return self._mw_cached

    #     weights = base.copy()
    #     if cut_idxs is not None:
    #         for idx, edge_indices in enumerate(cut_idxs):
    #             m = mu[idx]
    #             if m <= 0.0 or edge_indices.size == 0:
    #                 continue
    #             if free_mask is None:
    #                 weights[edge_indices] += m
    #             else:
    #                 sel = edge_indices[free_mask[edge_indices]]
    #                 if sel.size:
    #                     weights[sel] += m
    #     else:
    #         for idx, (cut, _) in enumerate(self.best_cuts):
    #             m = mu[idx]
    #             if m <= 0.0:
    #                 continue
    #             for u, v in cut:
    #                 e = (u, v) if u <= v else (v, u)
    #                 j = self.edge_indices.get(e)
    #                 if j is None:
    #                     continue
    #                 if free_mask is not None and not free_mask[j]:
    #                     continue
    #                 weights[j] += m

    #     self._mw_cached = weights
    #     self._mw_lambda = lam
    #     self._mw_mu = mu.copy()
    #     self._mw_free_mask_key = self._free_mask_key
    #     return weights
    # def compute_modified_weights(self):
    #     import numpy as np

    #     base = self.edge_weights.copy()
    #     lam = max(0.0, min(getattr(self, "lmbda", 0.0), 1e4))
    #     if lam:
    #         base = base + lam * self.edge_lengths

    #     # No cuts? return λ-priced base
    #     if not (self.use_cover_cuts and self.best_cuts):
    #         self._mw_cached = None
    #         self._mw_lambda = lam
    #         self._mw_mu = None
    #         self._mw_free_mask_key = None
    #         return base

    #     # μ vector aligned to current best_cuts
    #     cut_idxs_all = getattr(self, "_cut_edge_idx_all", None)  # NEW: indices for ALL cut edges
    #     mu_len = len(cut_idxs_all) if cut_idxs_all is not None else len(self.best_cuts)
    #     mu = np.array(
    #         [max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4)) for i in range(mu_len)],
    #         dtype=float,
    #     )

    #     # Cache key: (λ, μ, free-mask signature). Keep existing structure.
    #     free_mask = self._get_free_mask()
    #     free_mask_key = self._free_mask_key

    #     if (self._mw_cached is not None and
    #         self._mw_lambda == lam and
    #         self._mw_mu is not None and
    #         self._mw_mu.shape == mu.shape and
    #         np.allclose(self._mw_mu, mu, rtol=0, atol=0) and
    #         self._mw_free_mask_key == free_mask_key):
    #         return self._mw_cached

    #     # Add μ to ALL edges that belong to each cut (fixed edges included!)
    #     weights = base.copy()
    #     if cut_idxs_all is not None:
    #         for i, idxs in enumerate(cut_idxs_all):
    #             m = mu[i]
    #             if m > 0.0 and idxs.size:
    #                 weights[idxs] += m
    #     else:
    #         # Fallback (no precompute): touch every edge named in each cut
    #         for i, (cut, _) in enumerate(self.best_cuts):
    #             m = mu[i]
    #             if m <= 0.0:
    #                 continue
    #             for e in cut:
    #                 j = self.edge_indices.get(e)
    #                 if j is not None:
    #                     weights[j] += m

    #     self._mw_cached = weights
    #     self._mw_lambda = lam
    #     self._mw_mu = mu.copy()
    #     self._mw_free_mask_key = free_mask_key
    #     return weights
    def compute_modified_weights(self):
        import numpy as np

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



    def _invalidate_weight_cache(self):
        self._free_mask_cache = None
        self._free_mask_key = None
        self._mw_cached = None
        self._mw_lambda = None
        self._mw_mu = None
        self._mw_free_mask_key = None

    # def _get_free_mask(self):
    #     import numpy as np

    #     fixed = frozenset(getattr(self, "fixed_edges", set()))
    #     excluded = frozenset(getattr(self, "excluded_edges", set()))
    #     key = (fixed, excluded)

    #     # cache hit?
    #     if getattr(self, "_free_mask_cache", None) is not None and getattr(self, "_free_mask_key", None) == key:
    #         return self._free_mask_cache

    #     # all edges free
    #     if not fixed and not excluded:
    #         self._free_mask_cache = None
    #         self._free_mask_key = key
    #         return None

    #     # mask out fixed and excluded
    #     mask = np.ones(len(self.edge_list), dtype=bool)
    #     for e in fixed | excluded:
    #         idx = self.edge_indices.get(e)
    #         if idx is not None:
    #             mask[idx] = False

    #     self._free_mask_cache = mask
    #     self._free_mask_key = key
    #     return mask
    def _get_free_mask(self):
        import numpy as np

        fixed = frozenset(self.fixed_edges)
        forbidden = frozenset(getattr(self, "forbidden_edges", set()))
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

    # def _record_fractional_solution(self, fractional_solution):
    #     if not fractional_solution:
    #         lightweight = ()
    #     else:
    #         top_items = sorted(fractional_solution.items(), key=lambda x: -abs(x[1]))[:20]
    #         lightweight = tuple(top_items)
    #     self._append_with_cap(
    #         self.fractional_solutions,
    #         lightweight,
    #         self._fractional_history_cap,
    #     )
    def _record_fractional_solution(self, fractional_solution):
        if not fractional_solution:
            lightweight = ()
        else:
            import heapq
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

        for edge_idx in self.fixed_edge_indices:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += current_weights[edge_idx]
            else:
                return float('inf'), float('inf'), []

        weight_changes = current_weights - prev_weights
        changed_indices = np.where(np.abs(weight_changes) > self.cache_tolerance)[0]
        changed_edges = set(changed_indices)

        prev_mst_indices = {self.edge_indices[(u, v)] for u, v in prev_mst_edges
                            if self.edge_indices[(u, v)] not in self.fixed_edge_indices}
        candidate_indices = (prev_mst_indices | changed_edges) - self.excluded_edge_indices - self.fixed_edge_indices
        sorted_edges = sorted(candidate_indices, key=lambda i: current_weights[i])

        for edge_idx in sorted_edges:
            u, v = self.edge_list[edge_idx]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += current_weights[edge_idx]

        if uf.count_components() > 1 or len(set(u for u, _ in mst_edges) | set(v for _, v in mst_edges)) < self.num_nodes:
            return float('inf'), float('inf'), []

        mst_length = sum(self.edge_lengths[self.edge_indices[(u, v)]] 
                         for u, v in mst_edges)

        return mst_cost, mst_length, mst_edges
    

    def compute_mst(self, modified_edges=None):
        import hashlib
        start_time = time()
        
        if modified_edges is not None:
            weights = np.array([w for _, _, w in modified_edges], dtype=float)
        else:
            weights = self.compute_modified_weights()

        # Ensure numeric stability before quantization
        # weights = np.nan_to_num(
        #     weights,
        #     nan=0.0,
        #     posinf=1e9,
        #     neginf=-1e9,
        #     copy=False,
        # )
        # # weights_key = tuple(np.round(weights / self.cache_tolerance).astype(int))
        # quantized = np.round(weights / self.cache_tolerance).astype(np.int32, copy=False)
        # digest = hashlib.blake2b(quantized.view(np.uint8), digest_size=16).digest()
        # weights_key = (digest, len(quantized))
        # Sanitize weights early so the ratio doesn’t blow up
        weights = np.nan_to_num(
            weights,
            nan=0.0,
            posinf=1e9,
            neginf=-1e9,
            copy=False,
        )


        ratio = weights / self.cache_tolerance
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        ratio = np.clip(ratio, -1e9, 1e9)

        quantized = np.ascontiguousarray(ratio, dtype=np.float32)
        digest = hashlib.blake2b(quantized.view(np.uint8), digest_size=16).digest()
        weights_key = (digest, quantized.shape[0])
        self.consecutive_cache_hits = getattr(self, 'consecutive_cache_hits', 0)
        cached_result = self.mst_cache.get(weights_key)
        if cached_result is not None and self.consecutive_cache_hits < 5:
            self.consecutive_cache_hits += 1
            if self.verbose:
                print(f"Cache hit: Retrieved MST with length={cached_result[1]:.2f}, consecutive_hits={self.consecutive_cache_hits}")
            end_time = time()
            LagrangianMST.total_compute_time += end_time - start_time
            return cached_result
        
        self.consecutive_cache_hits = 0
        if self.verbose:
            print(f"Cache miss: Computing new MST for weights_key")
        
        mst_cost, mst_length, mst_edges = self.custom_kruskal(weights)
        if self.verbose:
            print(f"New MST computed: length={mst_length:.2f}")
        
        self.mst_cache.put(weights_key, (mst_cost, mst_length, mst_edges))

        end_time = time()
        LagrangianMST.total_compute_time += end_time - start_time
        return mst_cost, mst_length, mst_edges
    
    def compute_mst_incremental(self, prev_weights, prev_mst_edges):
        current_weights = self.compute_modified_weights()
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
        # self.best_cuts = inherited_cuts or []
        # self.best_cut_multipliers = inherited_multipliers or {}
        # self.best_cut_multipliers_for_best_bound = {}
        
        # # Initialize cuts and multipliers
        # if inherited_cuts is not None:
        #     self.best_cuts = [(set(tuple(sorted((u, v))) for u, v in cut), rhs) for cut, rhs in inherited_cuts]
        #     self.best_cut_multipliers = inherited_multipliers.copy() if inherited_multipliers else {}
        # else:
        #     self.best_cuts = []
        #     self.best_cut_multipliers = {}
       
        # self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
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

        lambda_min = 0.0
        lambda_max = 10.0
        bisection_tolerance = 1e-5

       
        if self.use_bisection:
        # Validate graph and edges
            if not self.edges or not nx.is_connected(self.graph):
                if self.verbose:
                    print(f"Error at depth {depth}: Empty edge list or disconnected graph in bisection path")
                return self.best_lower_bound, self.best_upper_bound, []
            
            # Check for invalid edge weights/lengths
            for u, v, w, l in self.edges:
                if math.isnan(w) or math.isinf(w) or math.isnan(l) or math.isinf(l) or l <= 0:
                    if self.verbose:
                        print(f"Error at depth {depth}: Invalid edge ({u},{v}) with weight={w}, length={l}")
                    return self.best_lower_bound, self.best_upper_bound, []

            # Initialize bounds
            lambda_min = 0.0
            lambda_max = 10.0
            try:
                lambda_max = max(10.0, max(w / l for _, _, w, l in self.edges if l > 0))
            except (ValueError, ZeroDivisionError) as e:
                if self.verbose:
                    print(f"Warning at depth {depth}: Failed to compute lambda_max: {e}. Using default lambda_max=10.0")
                lambda_max = 10.0
            
            bisection_tolerance = 1e-5
            subgradient_tolerance = 1e-5
            max_iter = 10
            iter_num = 0
            no_improvement_count = 0
            polyak_enabled = False
            self._moving_upper = self.best_upper_bound if self.best_upper_bound < float('inf') else 1000.0
            
            # Compute initial subgradient at lambda_min
            self.lmbda = lambda_min
            try:
                mst_cost_min, mst_length_min, mst_edges_min = self.compute_mst()
                if math.isnan(mst_cost_min) or math.isinf(mst_cost_min) or math.isnan(mst_length_min) or math.isinf(mst_length_min):
                    raise ValueError("Initial MST at lambda_min produced invalid cost or length")
                S_min = mst_length_min - self.budget
                self.last_mst_edges = [tuple(sorted((u, v))) for u, v in mst_edges_min]
            except Exception as e:
                if self.verbose:
                    print(f"Bisection initialization failed at lambda_min (depth {depth}): {e}")
                return self.best_lower_bound, self.best_upper_bound, []
            
            # Initialize S_max with a safe default
            S_max = float('inf')
            if S_min > 0:
                self.lmbda = lambda_max
                expansion_iter = 0
                while expansion_iter < 10:
                    try:
                        mst_cost_max, mst_length_max, mst_edges_max = self.compute_mst()
                        if math.isnan(mst_cost_max) or math.isinf(mst_cost_max) or math.isnan(mst_length_max) or math.isinf(mst_length_max):
                            raise ValueError("Initial MST at lambda_max produced invalid cost or length")
                        S_max = mst_length_max - self.budget
                        self.last_mst_edges = [tuple(sorted((u, v))) for u, v in mst_edges_max]
                        if S_max <= 0:
                            break
                        lambda_max *= 2
                        self.lmbda = lambda_max
                        expansion_iter += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"Error expanding lambda_max (iter {expansion_iter}, depth {depth}): {e}")
                        S_max = -1.0  # Assume under-budget to continue
                        break

            while (lambda_max - lambda_min >= bisection_tolerance and 
                abs(S_min) > subgradient_tolerance and abs(S_max) > subgradient_tolerance and 
                iter_num < max_iter):
                
                # Linear interpolation (secant method)
                if abs(S_max - S_min) > 1e-8:
                    lambda_new = lambda_min - S_min * (lambda_max - lambda_min) / (S_max - S_min)
                    lambda_new = max(lambda_min, min(lambda_new, lambda_max))
                else:
                    lambda_new = (lambda_min + lambda_max) / 2
                
                self.lmbda = max(1e-4, min(lambda_new, 1e4))
                if self.lmbda < 1e-6:
                    self.lmbda = 0.1
                
                # Compute MST
                try:
                    if prev_weights is not None and prev_mst_edges is not None:
                        mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
                    else:
                        mst_cost, mst_length, mst_edges = self.compute_mst()
                    if math.isnan(mst_cost) or math.isinf(mst_cost) or math.isnan(mst_length) or math.isinf(mst_length):
                        raise ValueError(f"MST computation failed: cost={mst_cost}, length={mst_length}")
                    self.last_mst_edges = [tuple(sorted((u, v))) for u, v in mst_edges]
                except Exception as e:
                    if self.verbose:
                        print(f"Bisection Iter {iter_num} (depth {depth}): MST computation failed: {e}")
                    break
                
                prev_mst_edges = self.last_mst_edges
                prev_weights = self.compute_modified_weights()

                # Store best dual solution
                is_feasible = mst_length <= self.budget
                # self.primal_solutions.append((self.last_mst_edges, is_feasible))
                self._record_primal_solution(self.last_mst_edges, is_feasible)
                
                cover_cut_penalty = sum(
                    multiplier * (sum(1 for e in self.last_mst_edges if e in cut) - rhs)
                    for cut_idx, (cut, rhs) in enumerate(self.best_cuts)
                    for multiplier in [self.best_cut_multipliers.get(cut_idx, 0)]
                )
                lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

                                # Compute subgradients
                knapsack_subgradient = mst_length - self.budget
                knapsack_subgradient = max(-1e4, min(knapsack_subgradient, 1e4))
                cut_subgradients = [
                    sum(1 for e in self.last_mst_edges if e in cut) - rhs
                    for cut, rhs in self.best_cuts
                ]

                # Validate lagrangian_bound
                if not math.isnan(lagrangian_bound) and not math.isinf(lagrangian_bound):
                    if lagrangian_bound > self.best_lower_bound:
                        self.best_lower_bound = lagrangian_bound
                        self.best_lambda = self.lmbda
                        self.best_mst_edges = self.last_mst_edges
                        self.best_cost = mst_cost
                        self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    if self.verbose:
                        print(f"Bisection Iter {iter_num} (depth {depth}): Invalid lagrangian_bound={lagrangian_bound}")
                    no_improvement_count += 1

                # Update primal solution
                if is_feasible:
                    uf = UnionFind(self.num_nodes)
                    for u, v in self.last_mst_edges:
                        uf.union(u, v)
                    if uf.count_components() == 1:
                        real_weight, real_length = self.compute_real_weight_length()
                        if not math.isnan(real_weight) and not math.isinf(real_weight) and real_weight < self.best_upper_bound:
                            self.best_upper_bound = real_weight
                            self._moving_upper = 0.95 * self._moving_upper + 0.05 * self.best_upper_bound

                # Compute fractional LP solution
                try:
                    fractional_solution = self.compute_dantzig_wolfe_solution(None)
                    if fractional_solution:
                        # self.fractional_solutions.append(fractional_solution)
                        self._record_fractional_solution(fractional_solution)
                except Exception as e:
                    if self.verbose:
                        print(f"Bisection Iter {iter_num} (depth {depth}): Dantzig-Wolfe failed: {e}")

                # Check stagnation
                if no_improvement_count > 15:
                    if self.verbose:
                        print(f"Bisection terminated early due to no improvement for {no_improvement_count} iterations (depth {depth})")
                    break

            



                # Update interval
                if knapsack_subgradient > 0:
                    lambda_min = self.lmbda
                    S_min = knapsack_subgradient
                else:
                    lambda_max = self.lmbda
                    S_max = knapsack_subgradient

                # Update cut multipliers with hybrid Polyak/Decay
                self.multipliers.append(self.lmbda)
                self.step_sizes.append(self.step_size)
                
                if self.verbose:
                    print(f"Bisection Iter {iter_num} (depth {depth}): lambda={self.lmbda:.6f}, "
                        f"interval={lambda_max - lambda_min:.6f}, subgradient={knapsack_subgradient:.6f}")

                gamma = 0.1
                current_L = lagrangian_bound
                gap = max(1e-6, self._moving_upper - current_L) if self._moving_upper < float('inf') else 1.0
                knapsack_norm_sq = max(1e-10, knapsack_subgradient ** 2)
                cut_norm_sq = max(1e-10, sum(g ** 2 for g in cut_subgradients))
                
                if polyak_enabled and gap > 1e-6 and cut_norm_sq > 1e-10:
                    polyak_cut_step = gamma * gap / cut_norm_sq
                    polyak_cut_step = max(1e-8, min(polyak_cut_step, self.step_size * 2))
                    
                    for cut_idx, violation in enumerate(cut_subgradients):
                        current_mult = self.best_cut_multipliers.get(cut_idx, 0)
                        violation = max(-1e8, min(violation, 1e4))
                        new_mult = max(1e-10, current_mult + polyak_cut_step * violation)
                        if new_mult < 1e-6 and abs(violation) > 1.0:
                            new_mult = 0.001
                        self.best_cut_multipliers[cut_idx] = min(new_mult, 1e4)
                    
                    if self.verbose:
                        print(f"Polyak cut step: {polyak_cut_step:.6f} (depth {depth})")
                else:
                    polyak_enabled = False
                    self.step_size *= self.p
                    for cut_idx, violation in enumerate(cut_subgradients):
                        current_mult = self.best_cut_multipliers.get(cut_idx, 0)
                        violation = max(-1e4, min(violation, 1e4))
                        new_mult = max(1e-4, current_mult + self.step_size * violation)
                        if new_mult < 1e-6 and abs(violation) > 1.0:
                            new_mult = 0.001
                        self.best_cut_multipliers[cut_idx] = min(new_mult, 1e4)
                    
                    if self.verbose:
                        print(f"Switched to decay for cuts (depth {depth})")

                # Check convergence
                converged = (abs(knapsack_subgradient) < subgradient_tolerance and 
                            all(abs(g) < subgradient_tolerance for g in cut_subgradients))
                duality_gap = self.best_upper_bound - self.best_lower_bound if self.best_upper_bound < float('inf') else float('inf')
                subgrad_norm = math.sqrt(knapsack_subgradient ** 2 + sum(g ** 2 for g in cut_subgradients))
                
                if converged or duality_gap < 1e-5 or subgrad_norm < 1e-5:
                    if self.verbose:
                        print(f"Converged! (Reason: {'Converged' if converged else 'Small duality gap' if duality_gap < 1e-5 else 'Small subgrad norm'}, depth {depth})")
                    break

                iter_num += 1

            # Final subgradient polish
            try:
                knapsack_subgradient = mst_length - self.budget
                knapsack_subgradient = max(-1e4, min(knapsack_subgradient, 1e4))
                cut_subgradients = [
                    sum(1 for e in self.last_mst_edges if e in cut) - rhs
                    for cut, rhs in self.best_cuts
                ]
                self.lmbda = max(1e-4, min(self.lmbda + self.step_size * knapsack_subgradient, 1e4))
                if self.lmbda < 1e-6 and abs(knapsack_subgradient) > 1.0:
                    self.lmbda = 0.1
                for cut_idx, violation in enumerate(cut_subgradients):
                    current_mult = self.best_cut_multipliers.get(cut_idx, 0)
                    violation = max(-1e4, min(violation, 1e4))
                    new_mult = max(1e-4, current_mult + self.step_size * violation)
                    if new_mult < 1e-6 and abs(violation) > 1.0:
                        new_mult = 0.001
                    self.best_cut_multipliers[cut_idx] = min(new_mult, 1e4)
            except NameError:
                if self.verbose:
                    print(f"Warning: Final polish skipped due to undefined mst_length (depth {depth})")
                pass
         
    
        # else:  # Subgradient method with Polyak hybrid (refined, faster & integer-safe)
        #     import numpy as np
        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     MAX_HISTORY = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10
        #     gamma_mu = 1.0
        #     mu_increment_cap = 5.0
        #     eps = 1e-12

        #     no_improvement_count = 0
        #     polyak_enabled = True
        #     stagnation_threshold = 15

        #     # NEW: guard against premature exit at the root
        #     MIN_ITERS_BEFORE_CONV = 3

        #     # Initialize moving UB for Polyak gap control
        #     if not hasattr(self, "_moving_upper"):
        #         self._moving_upper = self.best_upper_bound if self.best_upper_bound < float("inf") else 1000.0

        #     # -------- Build rhs_eff and early-prune impossible nodes (once per node) --------
        #     F_in = getattr(self, "fixed_edges", set())   # normalized tuples
        #     F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()

        #     self._rhs_eff = {}
        #     for idx, (cut, rhs) in enumerate(self.best_cuts):
        #         rhs_eff = rhs - len(cut & F_in)
        #         self._rhs_eff[idx] = rhs_eff
        #         if rhs_eff < 0:
        #             # Node infeasible due to fixed-in edges saturating the cut
        #             end_time = time()
        #             LagrangianMST.total_compute_time += end_time - start_time
        #             return float('inf'), self.best_upper_bound, []

        #     # -------- Precompute cut→edge-index arrays for FREE edges (once per node) --------
        #     # FREE = all edges except F_in and (optionally) F_out
        #     edge_idx = self.edge_indices  # normalized edge -> index
        #     cut_edge_idx = []
        #     cut_free_sizes = []
        #     for cut, _ in self.best_cuts:
        #         # indices for free edges only
        #         idxs = [edge_idx[e] for e in cut if (e not in F_in and e not in F_out) and (e in edge_idx)]
        #         arr = np.fromiter(idxs, dtype=np.int32) if idxs else np.empty(0, dtype=np.int32)
        #         cut_edge_idx.append(arr)
        #         cut_free_sizes.append(max(1, len(idxs)))   # avoid div-by-zero in normalization

        #     self._cut_edge_idx = cut_edge_idx  # used by compute_modified_weights fast path
        #     rhs_eff_vec = np.array([self._rhs_eff[i] for i in range(len(cut_edge_idx))], dtype=float)

        #     # Reset modified-weights cache for this node
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx), dtype=float)

        #     # Ensure histories exist
        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     # --- CRITICAL: seed priced weights so iteration 0 is correct ---
        #     prev_weights = self.compute_modified_weights()   # priced by current (λ, μ) — initially zero is OK
        #     prev_mst_edges = None
        #     plateau_counter = 0
        #     # ensure this exists
        #     self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0)

        #     for iter_num in range(max_iter):
        #         if iter_num % 10 == 0:
        #             if len(self.primal_solutions) > MAX_SOLUTIONS:
        #                 self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #             if len(self.subgradients) > MAX_HISTORY:
        #                 self.subgradients = self.subgradients[-MAX_HISTORY:]
        #             if len(self.step_sizes) > MAX_HISTORY:
        #                 self.step_sizes = self.step_sizes[-MAX_HISTORY:]
        #             if len(self.multipliers) > MAX_HISTORY:
        #                 self.multipliers = self.multipliers[-MAX_HISTORY:]
        #             if len(self.fractional_solutions) > 10:
        #                 self.fractional_solutions = self.fractional_solutions[-5:]
        #         # -------- Step 1: MST (incremental w/ fallback) --------
        #         try:
        #             if prev_weights is not None and prev_mst_edges is not None:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #             else:
        #                 # ensure full MST sees the priced weights for iter 0 (and any non-incremental call)
        #                 # self._weights_for_mst = prev_weights
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()

        #             if (math.isnan(mst_cost) or math.isinf(mst_cost) or
        #                 math.isnan(mst_length) or math.isinf(mst_length)):
        #                 if self.verbose:
        #                     print(f"Subgradient Iter {iter_num}: Invalid MST, fallback to full")
        #                 # fallback must also use the priced weights
        #                 # self._weights_for_mst = prev_weights
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Error in MST: {e}, fallback")
        #             # exception fallback must also use the priced weights
        #             # self._weights_for_mst = prev_weights
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         # IMPORTANT: mst_edges must be normalized tuples matching edge_indices keys
        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges = self.last_mst_edges

        #         # Build/Update modified weights (fast incremental) for NEXT iter
        #         prev_weights = self.compute_modified_weights()

        #         # -------- Step 2: Dual bound & book-keeping --------
        #         is_feasible = (mst_length <= self.budget)
        #         # self.primal_solutions.append((self.last_mst_edges, is_feasible))
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         # Penalty: Σ μ_i * rhs_eff(i)   (vectorized)
        #         mu_vec = np.fromiter((self.best_cut_multipliers.get(i, 0.0) for i in range(len(cut_edge_idx))),
        #                             dtype=float, count=len(cut_edge_idx))
        #         cover_cut_penalty = float(mu_vec @ rhs_eff_vec)
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound) and not math.isinf(lagrangian_bound)
        #             and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #         else:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Invalid L={lagrangian_bound}")
        #             no_improvement_count += 1

        #         # -------- Step 3: Update best primal if feasible (skip UF connectivity) --------
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight) and not math.isinf(real_weight)
        #                     and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # -------- Step 4: Subgradients (λ and μ) --------
        #         knapsack_subgradient = mst_length - self.budget

        #         # k_free via bit mask on edges (vectorized)
        #         nE = len(self.edge_weights)
        #         mst_mask = np.zeros(nE, dtype=bool)
        #         # mark chosen free edges True
        #         for e in self.last_mst_edges:
        #             if e in F_in or e in F_out:
        #                 continue
        #             j = edge_idx.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         cut_subgradients = []
        #         # for i, idxs in enumerate(cut_edge_idx):
        #         #     # number of chosen FREE edges from cut i
        #         #     k_free = int(mst_mask[idxs].sum()) if idxs.size else 0
        #         #     violation = k_free - rhs_eff_vec[i]
        #         #     cut_subgradients.append(violation)
        #         violations = np.array([
        #             (int(mst_mask[idxs].sum()) if idxs.size else 0) - rhs_eff_vec[i]
        #             for i, idxs in enumerate(cut_edge_idx)
        #         ])

        #         # Convert to list for cut_subgradients
        #         cut_subgradients = violations.tolist()

        #         # Precomputed sizes (avoid recompute)
        #         cut_sizes = cut_free_sizes

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                 f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #         # stagnation in knapsack subgradient
        #         if iter_num > 0 and abs(knapsack_subgradient - self.subgradients[-1]) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0

        #         # self.subgradients.append(knapsack_subgradient)
        #         self._record_subgradient(knapsack_subgradient)
        #         if len(self.subgradients) > MAX_HISTORY:
        #             self.subgradients = self.subgradients[-MAX_HISTORY//2:]

        #         # -------- Step 5: Convergence criteria --------
        #         converged = (abs(knapsack_subgradient) < 1e-5 and
        #                     all(abs(g) < 1e-5 for g in cut_subgradients))

        #         duality_gap = float('inf')
        #         if (self.best_upper_bound < float('inf') and self.best_lower_bound > -float('inf')):
        #             duality_gap = self.best_upper_bound - self.best_lower_bound

        #         subgrad_norm = math.sqrt(
        #             knapsack_subgradient ** 2 + sum(g ** 2 for g in cut_subgradients)
        #         )

        #         # Don't allow early exit before a few iterations
        #         if (iter_num >= MIN_ITERS_BEFORE_CONV) and (
        #             converged or
        #             (0 <= duality_gap < 1e-4) or
        #             subgrad_norm < 1e-4 or
        #             no_improvement_count > stagnation_threshold or
        #             self.step_size < 1e-8
        #         ):
        #             if self.verbose:
        #                 reason = ("converged" if converged else
        #                         "small duality gap" if 0 <= duality_gap < 1e-4 else
        #                         "small subgrad norm" if subgrad_norm < 1e-4 else
        #                         "no improvement" if no_improvement_count > stagnation_threshold else
        #                         "step too small")
        #                 print(f"Converged at iter {iter_num}! (Reason: {reason})")
        #             break

        #         # -------- Step 6: Hybrid Polyak + Decay updates (unchanged logic) --------
        #         self.step_sizes.append(self.step_size)
        #         if len(self.step_sizes) > MAX_HISTORY:
        #             self.step_sizes = self.step_sizes[-MAX_HISTORY//2:]
        #         self.multipliers.append(self.lmbda)
        #         if len(self.multipliers) > MAX_HISTORY:
        #             self.multipliers = self.multipliers[-MAX_HISTORY//2:]

        #         # Update moving UB
        #         if self.best_upper_bound < float('inf'):
        #             self._moving_upper = 0.95 * self._moving_upper + 0.05 * self.best_upper_bound

        #         # Adaptive Polyak gamma
        #         gamma_iter = max(0.05, min(0.2, gamma_base * (1 - iter_num / max_iter)))

        #         if polyak_enabled and self._moving_upper < float('inf'):
        #             current_L = (lagrangian_bound if not math.isnan(lagrangian_bound) else self.best_lower_bound)
        #             gap = max(1e-6, self._moving_upper - current_L)

        #             knap_norm2 = max(1e-10, knapsack_subgradient ** 2)
        #             mu_norm2 = 0.0
        #             for vi, size_i in zip(cut_subgradients, cut_sizes):
        #                 g = vi / (1.0 + size_i)
        #                 mu_norm2 += g * g
        #             mu_norm2 = max(1e-10, mu_norm2)

        #             polyak_lambda_step = gamma_iter * gap / knap_norm2
        #             polyak_lambda_step = max(1e-8, min(polyak_lambda_step, self.step_size * 2))

        #             polyak_cut_step = gamma_mu * gap / mu_norm2
        #             polyak_cut_step = max(1e-8, min(polyak_cut_step, self.step_size * 2))

        #             # --- Update λ (momentum & projection) ---
        #             proposed_lambda = self.lmbda + polyak_lambda_step * knapsack_subgradient
        #             beta = self.momentum_beta
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))

        #             # --- Update μ_i (momentum, normalized, projection, increment cap) ---
        #             # for cut_idx, (vi, size_i) in enumerate(zip(cut_subgradients, cut_sizes)):
        #             #     g = vi / (1.0 + size_i)
        #             #     current_mult = self.best_cut_multipliers.get(cut_idx, 0.0)
        #             #     proposed = current_mult + polyak_cut_step * g
        #             #     new_mult = (1 - beta) * current_mult + beta * proposed
        #             #     if mu_increment_cap is not None:
        #             #         new_mult = min(new_mult, current_mult + mu_increment_cap)
        #             #     self.best_cut_multipliers[cut_idx] = max(0.0, min(new_mult, 1e4))

        #             for cut_idx, (vi, size_i) in enumerate(zip(cut_subgradients, cut_sizes)):
        #                 g = vi / (1.0 + size_i)
        #                 current_mult = self.best_cut_multipliers.get(cut_idx, 0.0)
                        
        #                 # Calculate proposed step
        #                 raw_step = polyak_cut_step * g  # or self.step_size * g in decay mode
                        
        #                 # Cap the step magnitude (not the final value)
        #                 # capped_step = max(-mu_increment_cap, min(raw_step, mu_increment_cap))
                        
        #                 # # Apply momentum
        #                 # proposed = current_mult + capped_step
        #                 # new_mult = (1 - beta) * current_mult + beta * proposed
        #                 increment = polyak_cut_step * g
        #                 capped_increment = max(-mu_increment_cap, min(increment, mu_increment_cap))
        #                 proposed = current_mult + capped_increment
        #                 new_mult = max(0.0, min(proposed, 1e4))
                                                
        #                 # Ensure non-negative and reasonable upper bound
        #                 self.best_cut_multipliers[cut_idx] = max(0.0, min(new_mult, 1e4))
        #         else:
        #             # --- DECAY MODE ---
        #             self.step_size *= self.p
        #             beta = self.momentum_beta
        #             proposed_lambda = self.lmbda + self.step_size * knapsack_subgradient
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))

        #             # for cut_idx, (vi, size_i) in enumerate(zip(cut_subgradients, cut_sizes)):
        #             #     g = vi / (1.0 + size_i)
        #             #     current_mult = self.best_cut_multipliers.get(cut_idx, 0.0)
        #             #     proposed = current_mult + self.step_size * g
        #             #     new_mult = (1 - beta) * current_mult + beta * proposed
        #             #     if mu_increment_cap is not None:
        #             #         new_mult = min(new_mult, current_mult + mu_increment_cap)
        #             #     self.best_cut_multipliers[cut_idx] = max(0.0, min(new_mult, 1e4))

        #             for cut_idx, (vi, size_i) in enumerate(zip(cut_subgradients, cut_sizes)):
        #                 g = vi / (1.0 + size_i)
        #                 current_mult = self.best_cut_multipliers.get(cut_idx, 0.0)
                        
        #                 # Calculate proposed step
        #                 raw_step = polyak_cut_step * g  # or self.step_size * g in decay mode
                        
        #                 # Cap the step magnitude (not the final value)
        #                 # capped_step = max(-mu_increment_cap, min(raw_step, mu_increment_cap))
                        
        #                 # # Apply momentum
        #                 # proposed = current_mult + capped_step
        #                 # new_mult = (1 - beta) * current_mult + beta * proposed
        #                 increment = polyak_cut_step * g
        #                 capped_increment = max(-mu_increment_cap, min(increment, mu_increment_cap))
        #                 proposed = current_mult + capped_increment
        #                 new_mult = max(0.0, min(proposed, 1e4))
        #                 # Ensure non-negative and reasonable upper bound
        #                 self.best_cut_multipliers[cut_idx] = max(0.0, min(new_mult, 1e4))

        #         # Optional cache/cleanup
        #         if iter_num % CLEANUP_INTERVAL == 0:
        #             if len(self.fractional_solutions) > 10:
        #                 self.fractional_solutions = self.fractional_solutions[-5:]
        #             # removed undefined `depth`-based cache tuning

        #     # -------- Final cover-cut generation with dedup --------
        #     new_cuts = []
        #     if self.use_cover_cuts and self.best_mst_edges:
        #         try:
        #             new_cuts = self.generate_cover_cuts(self.best_mst_edges)
        #             existing = {frozenset(c): r for c, r in self.best_cuts}
        #             for cut, rhs in new_cuts:
        #                 fz = frozenset(cut)
        #                 if fz in existing:
        #                     if rhs > existing[fz]:
        #                         idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                         self.best_cuts[idx] = (cut, rhs)
        #                         existing[fz] = rhs
        #                 else:
        #                     new_idx = len(self.best_cuts)
        #                     self.best_cuts.append((cut, rhs))
        #                     self.best_cut_multipliers[new_idx] = 0.0
        #                     self.best_cut_multipliers_for_best_bound[new_idx] = 0.0
        #                     existing[fz] = rhs
        #                     if self.verbose:
        #                         print(f"Added final cut: size={len(cut)}, rhs={rhs}")
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error generating final cuts: {e}")

        #     if self.verbose:
        #         print(f"Solve completed: iterations={iter_num+1}, "
        #             f"lower={self.best_lower_bound:.2f}, upper={self.best_upper_bound:.2f}, "
        #             f"λ={self.lmbda:.6f}, cuts={len(self.best_cuts)}")
        #         if self.best_upper_bound < float('inf') and self.best_lower_bound > -float('inf'):
        #             print(f"Duality gap: {self.best_upper_bound - self.best_lower_bound:.4f}")
        #         print(f"Final step size: {self.step_size:.8f}")

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, new_cuts

        # else:  # Subgradient method with Polyak hybrid (correct dual pricing + cover cuts)
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     MAX_HISTORY = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10
        #     gamma_mu = 1.0
        #     mu_increment_cap = 5.0
        #     eps = 1e-12

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # For returning new cuts to the node
        #     new_cuts = []

        #     # --- Quick guards ---
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, new_cuts

        #     # --- Prepare fixed/excluded ---
        #     F_in = getattr(self, "fixed_edges", set())    # normalized tuples
        #     F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()

        #     # --- Setup internal state / indices once per node ---
        #     # Normalize cut sets and check rhs_eff for infeasibility
        #     self._rhs_eff = {}
        #     for idx, (cut, rhs) in enumerate(self.best_cuts):
        #         rhs_eff = int(rhs) - len(cut & F_in)
        #         self._rhs_eff[idx] = rhs_eff
        #         if rhs_eff < 0:
        #             # child infeasible due to fixed edges saturating the cut
        #             end_time = time()
        #             LagrangianMST.total_compute_time += end_time - start_time
        #             return float('inf'), self.best_upper_bound, new_cuts

        #     # -------- Precompute cut→edge-index arrays --------
        #     edge_idx = self.edge_indices  # normalized edge -> index

        #     # FREE = all edges except F_in and F_out (for subgradients)
        #     cut_edge_idx_free = []
        #     cut_free_sizes = []
        #     for cut, _ in self.best_cuts:
        #         idxs = [edge_idx[e] for e in cut
        #                 if (e not in F_in and e not in F_out) and (e in edge_idx)]
        #         arr = np.fromiter(idxs, dtype=np.int32) if idxs else np.empty(0, dtype=np.int32)
        #         cut_edge_idx_free.append(arr)
        #         cut_free_sizes.append(max(1, len(idxs)))  # avoid /0

        #     # ALL = include fixed edges too (for correct dual pricing in MST)
        #     cut_edge_idx_all = []
        #     for cut, _ in self.best_cuts:
        #         idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #         arr_all = np.fromiter(idxs_all, dtype=np.int32) if idxs_all else np.empty(0, dtype=np.int32)
        #         cut_edge_idx_all.append(arr_all)

        #     # Stash for compute_modified_weights()
        #     # NOTE: compute_modified_weights will *only* use μ if use_cover_cuts and best_cuts
        #     self._cut_edge_idx = cut_edge_idx_free          # kept for compatibility
        #     self._cut_edge_idx_all = cut_edge_idx_all       # used by compute_modified_weights()

        #     # Vector views
        #     rhs_eff_vec = np.array([self._rhs_eff[i] for i in range(len(cut_edge_idx_free))], dtype=float)
        #     rhs_vec = np.array([rhs for (_, rhs) in self.best_cuts], dtype=float)  # ORIGINAL RHS for dual penalty

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     # Seed priced weights so iteration 0 is consistent
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None

        #     last_g_lambda = None  # for simple stagnation check

        #     # --- Iterate subgradient steps ---
        #     for iter_num in range(int(max_iter)):
        #         # === 1) Solve MST on current priced weights ===
        #         try:
        #             mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #         except Exception:
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         # Keep normalized
        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges = self.last_mst_edges

        #         # Prepare weights for NEXT iteration
        #         prev_weights = self.compute_modified_weights()

        #         # === 2) Dual & primal bookkeeping ===
        #         is_feasible = (mst_length <= self.budget)

        #         # (a) record primal & update best_upper_bound when feasible
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight)
        #                         and not math.isinf(real_weight)
        #                         and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # prune primal_solutions history a bit
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         # (b) Lagrangian dual value
        #         # If cuts are off or none exist, these vectors are length 0 → penalty = 0
        #         mu_vec = np.fromiter(
        #             (self.best_cut_multipliers.get(i, 0.0) for i in range(len(cut_edge_idx_free))),
        #             dtype=float,
        #             count=len(cut_edge_idx_free),
        #         )
        #         cover_cut_penalty = float(mu_vec @ rhs_vec) if (self.use_cover_cuts and len(rhs_vec) > 0) else 0.0
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound)
        #                 and not math.isinf(lagrangian_bound)
        #                 and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #                 if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #         else:
        #             no_improvement_count += 1

        #         # === 3) Subgradients ===
        #         # g_λ = mst_length − B
        #         knapsack_subgradient = float(mst_length - self.budget)

        #         # g_μi = lhs_total − rhs_i = (lhs_free) − rhs_eff_i
        #         mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #         for e in mst_edges:
        #             j = self.edge_indices.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         violations = np.array([
        #             (int(mst_mask[idxs].sum()) if idxs.size else 0) - rhs_eff_vec[i]
        #             for i, idxs in enumerate(cut_edge_idx_free)
        #         ]) if (self.use_cover_cuts and len(cut_edge_idx_free) > 0) else np.zeros(0, dtype=float)

        #         cut_subgradients = violations.tolist()
        #         cut_sizes = cut_free_sizes

        #         # === 4) Step sizes & updates ===
        #         self.subgradients.append(knapsack_subgradient)

        #         # (a) λ update (Polyak / fallback)
        #         if polyak_enabled and self.best_upper_bound < float('inf'):
        #             gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             alpha = gamma_base * gap / (knapsack_subgradient ** 2 + eps)
        #         else:
        #             alpha = getattr(self, "step_size", 1e-5)

        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # (b) μ updates (only if cuts are active)
        #         if self.use_cover_cuts:
        #             for i, g in enumerate(cut_subgradients):
        #                 step_mu = gamma_mu * g / (cut_sizes[i] + eps)
        #                 step_mu = max(-mu_increment_cap, min(mu_increment_cap, step_mu))
        #                 self.best_cut_multipliers[i] = max(
        #                     0.0,
        #                     self.best_cut_multipliers.get(i, 0.0) + step_mu,
        #                 )

        #         # history bookkeeping (lightweight)
        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #         # Simple stagnation check on g_λ
        #         if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0
        #         last_g_lambda = knapsack_subgradient

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                 f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #     # --- Final cover-cut generation with dedup (only if cuts are enabled) ---
        #     if self.use_cover_cuts and self.best_mst_edges:
        #         try:
        #             new_cuts = self.generate_cover_cuts(self.best_mst_edges) or []
        #             existing = {frozenset(c): r for (c, r) in self.best_cuts}

        #             for cut, rhs in new_cuts:
        #                 fz = frozenset(cut)
        #                 if fz in existing:
        #                     # if same support but stronger rhs, replace
        #                     if rhs > existing[fz]:
        #                         idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                         self.best_cuts[idx] = (cut, rhs)
        #                         existing[fz] = rhs
        #                 else:
        #                     new_idx = len(self.best_cuts)
        #                     self.best_cuts.append((cut, rhs))
        #                     # initialise μ for new cuts to 0 in both maps
        #                     self.best_cut_multipliers[new_idx] = 0.0
        #                     self.best_cut_multipliers_for_best_bound[new_idx] = 0.0
        #                     existing[fz] = rhs
        #                     if self.verbose:
        #                         print(f"Added final cut: size={len(cut)}, rhs={rhs}")
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error generating final cuts: {e}")

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, new_cuts


        # else:  # Subgradient method with Polyak hybrid + in-node cover cut separation
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10

        #     # μ updates: be conservative
        #     gamma_mu = 0.30          # smoother than 1.0
        #     mu_increment_cap = 1.0   # limit per-iter μ change

        #     eps = 1e-12

        #     # Cut management in this node
        #     max_active_cuts = getattr(self, "max_active_cuts", 0)   # total cuts used in dual
        #     max_new_cuts_per_call = getattr(self, "max_new_cuts_per_call", 3)
        #     max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 10)
        #     warmup_iters = getattr(self, "cut_warmup_iters", 5)
        #     sep_frequency = getattr(self, "cut_frequency", 5)  # you already pass cut_frequency

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Collect all cuts generated in THIS node (for children)
        #     node_new_cuts = []

        #     # --- Quick guards ---
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        #     # --- Prepare fixed/excluded ---
        #     F_in = getattr(self, "fixed_edges", set())    # normalized tuples
        #     F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()

        #     edge_idx = self.edge_indices  # normalized edge -> index

        #     # --- Compute rhs_eff per cut; detect infeasible child ---
        #     self._rhs_eff = {}
        #     for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #         rhs_eff = int(rhs) - len(cut & F_in)
        #         self._rhs_eff[idx_c] = rhs_eff
        #         if rhs_eff < 0:
        #             # child infeasible due to fixed edges saturating the cut
        #             end_time = time()
        #             LagrangianMST.total_compute_time += end_time - start_time
        #             return float('inf'), self.best_upper_bound, node_new_cuts

        #     # --- Optionally trim number of cuts at node start (keep largest |μ|) ---
        #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
        #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
        #         if not parent_mu_map:
        #             parent_mu_map = self.best_cut_multipliers

        #         idx_and_cut = list(enumerate(self.best_cuts))
        #         idx_and_cut.sort(
        #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
        #             reverse=True,
        #         )
        #         idx_and_cut = idx_and_cut[:max_active_cuts]

        #         new_cuts_list = []
        #         new_mu = {}
        #         remap = {}
        #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
        #             new_cuts_list.append(cut_rhs)
        #             new_mu[new_i] = parent_mu_map.get(old_i, 0.0)
        #             remap[old_i] = new_i

        #         self.best_cuts = new_cuts_list
        #         self.best_cut_multipliers = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu.copy()
        #         self._rhs_eff = {remap[i]: self._rhs_eff[i] for i in remap}

        #     # --- helper to rebuild index arrays whenever cuts change ---
        #     def _rebuild_cut_structures():
        #         nonlocal cut_edge_idx_free, cut_edge_idx_all, cut_free_sizes, rhs_eff_vec, rhs_vec

        #         cut_edge_idx_free = []
        #         cut_free_sizes = []
        #         cut_edge_idx_all = []

        #         for cut, _ in self.best_cuts:
        #             # FREE indices (for subgradients)
        #             idxs_free = [
        #                 edge_idx[e] for e in cut
        #                 if (e not in F_in and e not in F_out) and (e in edge_idx)
        #             ]
        #             arr_free = np.fromiter(idxs_free, dtype=np.int32) if idxs_free else np.empty(0, dtype=np.int32)
        #             cut_edge_idx_free.append(arr_free)
        #             cut_free_sizes.append(max(1, len(idxs_free)))  # avoid /0

        #             # ALL indices (for dual pricing)
        #             idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #             arr_all = np.fromiter(idxs_all, dtype=np.int32) if idxs_all else np.empty(0, dtype=np.int32)
        #             cut_edge_idx_all.append(arr_all)

        #         # stash for compute_modified_weights
        #         self._cut_edge_idx = cut_edge_idx_free
        #         self._cut_edge_idx_all = cut_edge_idx_all

        #         rhs_eff_vec = np.array(
        #             [self._rhs_eff[i] for i in range(len(cut_edge_idx_free))],
        #             dtype=float
        #         ) if self.best_cuts else np.zeros(0, dtype=float)

        #         rhs_vec = np.array(
        #             [rhs for (_, rhs) in self.best_cuts],
        #             dtype=float
        #         ) if self.best_cuts else np.zeros(0, dtype=float)

        #     # initial structures
        #     cut_edge_idx_free = []
        #     cut_free_sizes = []
        #     cut_edge_idx_all = []
        #     rhs_eff_vec = np.zeros(0, dtype=float)
        #     rhs_vec = np.zeros(0, dtype=float)
        #     _rebuild_cut_structures()

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     # Seed priced weights so iteration 0 is consistent
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None

        #     last_g_lambda = None  # for simple stagnation check
        #     new_cuts_added_here = 0

        #     # --- Iterate subgradient steps ---
        #     for iter_num in range(int(max_iter)):
        #         # === 1) Solve MST on current priced weights ===
        #         try:
        #             mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #         except Exception:
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         # Keep normalized
        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges = self.last_mst_edges

        #         # Prepare weights for NEXT iteration
        #         prev_weights = self.compute_modified_weights()

        #         # === 2) Dual & primal bookkeeping ===
        #         is_feasible = (mst_length <= self.budget)

        #         # (a) record primal & update best_upper_bound when feasible
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight)
        #                         and not math.isinf(real_weight)
        #                         and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # prune primal_solutions history a bit
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         # (b) Lagrangian dual value
        #         mu_vec = np.fromiter(
        #             (self.best_cut_multipliers.get(i, 0.0) for i in range(len(cut_edge_idx_free))),
        #             dtype=float,
        #             count=len(cut_edge_idx_free),
        #         )
        #         cover_cut_penalty = float(mu_vec @ rhs_vec) if (self.use_cover_cuts and len(rhs_vec) > 0) else 0.0
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound)
        #                 and not math.isinf(lagrangian_bound)
        #                 and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #                 if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #         else:
        #             no_improvement_count += 1

        #         # === 3) Subgradients ===
        #         # g_λ = mst_length − B
        #         knapsack_subgradient = float(mst_length - self.budget)

        #         # g_μi = lhs_total − rhs_i = (lhs_free) − rhs_eff_i
        #         mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #         for e in mst_edges:
        #             j = self.edge_indices.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         if self.use_cover_cuts and len(cut_edge_idx_free) > 0:
        #             violations = np.array([
        #                 (int(mst_mask[idxs].sum()) if idxs.size else 0) - rhs_eff_vec[i]
        #                 for i, idxs in enumerate(cut_edge_idx_free)
        #             ], dtype=float)
        #         else:
        #             violations = np.zeros(0, dtype=float)

        #         cut_subgradients = violations.tolist()
        #         cut_sizes = cut_free_sizes

        #         # === 4) Step sizes & updates ===
        #         self.subgradients.append(knapsack_subgradient)

        #         # (a) λ update (Polyak / fallback)
        #         if polyak_enabled and self.best_upper_bound < float('inf'):
        #             gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             alpha = gamma_base * gap / (knapsack_subgradient ** 2 + eps)
        #         else:
        #             alpha = getattr(self, "step_size", 1e-5)

        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # (b) μ updates (only if cuts are active)
        #         if self.use_cover_cuts:
        #             for i, g in enumerate(cut_subgradients):
        #                 step_mu = gamma_mu * g / (cut_sizes[i] + eps)
        #                 step_mu = max(-mu_increment_cap, min(mu_increment_cap, step_mu))
        #                 self.best_cut_multipliers[i] = max(
        #                     0.0,
        #                     self.best_cut_multipliers.get(i, 0.0) + step_mu,
        #                 )

        #         # history bookkeeping (lightweight)
        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #         # Simple stagnation check on g_λ
        #         if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0
        #         last_g_lambda = knapsack_subgradient

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                   f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #         # === 5) In-node cover cut separation ===
        #         if (
        #             self.use_cover_cuts
        #             and len(node_new_cuts) < max_new_cuts_per_node
        #             and iter_num >= warmup_iters
        #             and sep_frequency > 0
        #             and (iter_num % sep_frequency == 0)
        #         ):
        #             try:
        #                 cand_cuts = self.generate_cover_cuts(mst_edges) or []
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error generating in-node cuts at iter {iter_num}: {e}")
        #                 cand_cuts = []

        #             # existing supports
        #             existing = {frozenset(c): (i, rhs) for i, (c, rhs) in enumerate(self.best_cuts)}

        #             added_here = 0
        #             for cut, rhs in cand_cuts:
        #                 if added_here >= max_new_cuts_per_call:
        #                     break
        #                 fz = frozenset(cut)
        #                 if fz in existing:
        #                     # same support: keep stronger rhs if needed
        #                     i_old, rhs_old = existing[fz]
        #                     if rhs > rhs_old:
        #                         self.best_cuts[i_old] = (cut, rhs)
        #                         self._rhs_eff[i_old] = int(rhs) - len(cut & F_in)
        #                         existing[fz] = (i_old, rhs)
        #                     continue

        #                 # totally new cut
        #                 idx_new = len(self.best_cuts)
        #                 self.best_cuts.append((cut, rhs))
        #                 self.best_cut_multipliers[idx_new] = 0.0
        #                 self.best_cut_multipliers_for_best_bound[idx_new] = 0.0
        #                 self._rhs_eff[idx_new] = int(rhs) - len(cut & F_in)
        #                 existing[fz] = (idx_new, rhs)

        #                 node_new_cuts.append((cut, rhs))
        #                 new_cuts_added_here += 1
        #                 added_here += 1
        #                 if len(node_new_cuts) >= max_new_cuts_per_node:
        #                     break

        #             if added_here > 0:
        #                 # rebuild arrays to include new cuts
        #                 _rebuild_cut_structures()
        #                 # refresh cached μ-array length
        #                 self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     # --- Final cover-cut generation with dedup (best MST) ---
        #     if self.use_cover_cuts and self.best_mst_edges:
        #         try:
        #             final_cands = self.generate_cover_cuts(self.best_mst_edges) or []
        #             existing = {frozenset(c): (i, rhs) for i, (c, rhs) in enumerate(self.best_cuts)}

        #             for cut, rhs in final_cands:
        #                 fz = frozenset(cut)
        #                 if fz in existing:
        #                     i_old, rhs_old = existing[fz]
        #                     if rhs > rhs_old:
        #                         self.best_cuts[i_old] = (cut, rhs)
        #                         self._rhs_eff[i_old] = int(rhs) - len(cut & F_in)
        #                         existing[fz] = (i_old, rhs)
        #                 else:
        #                     idx_new = len(self.best_cuts)
        #                     self.best_cuts.append((cut, rhs))
        #                     self.best_cut_multipliers[idx_new] = 0.0
        #                     self.best_cut_multipliers_for_best_bound[idx_new] = 0.0
        #                     self._rhs_eff[idx_new] = int(rhs) - len(cut & F_in)
        #                     existing[fz] = (idx_new, rhs)
        #                     node_new_cuts.append((cut, rhs))

        #             # rebuild index arrays one last time
        #             if final_cands:
        #                 _rebuild_cut_structures()
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error generating final cuts: {e}")

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts


        # else:  # Subgradient method with Polyak hybrid (fixed dual per node + end-of-node cuts)
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10

        #     # μ updates: conservative
        #     gamma_mu = 0.30          # smoother than 1.0
        #     mu_increment_cap = 1.0   # limit per-iter μ change
        #     eps = 1e-12

        #     # Depth-based cutting: only generate new cuts up to this depth
        #     max_cut_depth = getattr(self, "max_cut_depth", 2)
        #     cutting_active_here = self.use_cover_cuts and (depth <= max_cut_depth)

        #     # Cut management in this node
        #     max_active_cuts = getattr(self, "max_active_cuts", 5)      # cuts used in dual at this node
        #     max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)
        #     min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Collect new cuts generated at this node (for children)
        #     node_new_cuts = []

        #     # --- Quick guards ---
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        #     # --- Prepare fixed/excluded ---
        #     F_in = getattr(self, "fixed_edges", set())    # normalized tuples
        #     F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()

        #     edge_idx = self.edge_indices  # normalized edge -> index

        #     # --- Compute rhs_eff per cut; detect infeasible node ---
        #     self._rhs_eff = {}
        #     for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #         rhs_eff = int(rhs) - len(cut & F_in)
        #         self._rhs_eff[idx_c] = rhs_eff
        #         if rhs_eff < 0:
        #             # node infeasible due to fixed edges saturating the cut
        #             end_time = time()
        #             LagrangianMST.total_compute_time += end_time - start_time
        #             return float('inf'), self.best_upper_bound, node_new_cuts

        #     # --- Trim number of cuts at node start (keep most "important") ---
        #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
        #         # use multipliers from best bound as importance, fall back to current
        #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
        #         if not parent_mu_map:
        #             parent_mu_map = self.best_cut_multipliers

        #         idx_and_cut = list(enumerate(self.best_cuts))
        #         # priority: large |μ|
        #         idx_and_cut.sort(
        #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
        #             reverse=True,
        #         )
        #         idx_and_cut = idx_and_cut[:max_active_cuts]

        #         new_cuts_list = []
        #         new_mu = {}
        #         remap = {}
        #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
        #             new_cuts_list.append(cut_rhs)
        #             new_mu[new_i] = parent_mu_map.get(old_i, 0.0)
        #             remap[old_i] = new_i

        #         self.best_cuts = new_cuts_list
        #         self.best_cut_multipliers = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu.copy()
        #         self._rhs_eff = {remap[i]: self._rhs_eff[i] for i in remap}

        #     # --- Precompute cut -> edge index arrays (fixed for this node) ---
        #     cut_edge_idx_free = []
        #     cut_free_sizes = []
        #     cut_edge_idx_all = []

        #     for cut, _ in self.best_cuts:
        #         # FREE indices (for subgradients)
        #         idxs_free = [
        #             edge_idx[e] for e in cut
        #             if (e not in F_in and e not in F_out) and (e in edge_idx)
        #         ]
        #         arr_free = np.fromiter(idxs_free, dtype=np.int32) if idxs_free else np.empty(0, dtype=np.int32)
        #         cut_edge_idx_free.append(arr_free)
        #         cut_free_sizes.append(max(1, len(idxs_free)))  # avoid /0

        #         # ALL indices (for dual pricing; includes fixed edges)
        #         idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #         arr_all = np.fromiter(idxs_all, dtype=np.int32) if idxs_all else np.empty(0, dtype=np.int32)
        #         cut_edge_idx_all.append(arr_all)

        #     # stash for compute_modified_weights()
        #     self._cut_edge_idx = cut_edge_idx_free
        #     self._cut_edge_idx_all = cut_edge_idx_all

        #     rhs_eff_vec = np.array(
        #         [self._rhs_eff[i] for i in range(len(cut_edge_idx_free))],
        #         dtype=float
        #     ) if self.best_cuts else np.zeros(0, dtype=float)

        #     rhs_vec = np.array(
        #         [rhs for (_, rhs) in self.best_cuts],
        #         dtype=float
        #     ) if self.best_cuts else np.zeros(0, dtype=float)

        #     # For tracking how "useful" each cut was at this node
        #     max_cut_violation = [0.0 for _ in self.best_cuts]

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     # Seed priced weights so iteration 0 is consistent
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None

        #     last_g_lambda = None  # for stagnation check

        #     # --- Subgradient iterations (fixed dual structure) ---
        #     for iter_num in range(int(max_iter)):
        #         # 1) Solve MST on current priced weights
        #         try:
        #             mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #         except Exception:
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges = self.last_mst_edges

        #         # Prepare weights for next iteration
        #         prev_weights = self.compute_modified_weights()

        #         # 2) Dual & primal bookkeeping
        #         is_feasible = (mst_length <= self.budget)

        #         # (a) primal & UB
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight)
        #                         and not math.isinf(real_weight)
        #                         and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # prune primal_solutions history
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         # (b) Lagrangian dual value
        #         mu_vec = np.fromiter(
        #             (self.best_cut_multipliers.get(i, 0.0) for i in range(len(cut_edge_idx_free))),
        #             dtype=float,
        #             count=len(cut_edge_idx_free),
        #         )
        #         cover_cut_penalty = float(mu_vec @ rhs_vec) if (self.use_cover_cuts and len(rhs_vec) > 0) else 0.0
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound)
        #                 and not math.isinf(lagrangian_bound)
        #                 and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #                 if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #         else:
        #             no_improvement_count += 1

        #         # 3) Subgradients
        #         # knapsack
        #         knapsack_subgradient = float(mst_length - self.budget)

        #         # cuts
        #         mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #         for e in mst_edges:
        #             j = self.edge_indices.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         if self.use_cover_cuts and len(cut_edge_idx_free) > 0:
        #             violations = []
        #             for i, idxs in enumerate(cut_edge_idx_free):
        #                 lhs_i = int(mst_mask[idxs].sum()) if idxs.size else 0
        #                 g_i = lhs_i - rhs_eff_vec[i]
        #                 violations.append(g_i)
        #                 # track max absolute violation for this cut at this node
        #                 if abs(g_i) > max_cut_violation[i]:
        #                     max_cut_violation[i] = abs(g_i)
        #             violations = np.array(violations, dtype=float)
        #         else:
        #             violations = np.zeros(0, dtype=float)

        #         cut_subgradients = violations.tolist()
        #         cut_sizes = cut_free_sizes

        #         # 4) Step sizes & updates
        #         self.subgradients.append(knapsack_subgradient)

        #         # λ update (Polyak / fallback)
        #         if polyak_enabled and self.best_upper_bound < float('inf'):
        #             gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             alpha = gamma_base * gap / (knapsack_subgradient ** 2 + eps)
        #         else:
        #             alpha = getattr(self, "step_size", 1e-5)

        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # μ updates (if cuts are active)
        #         if self.use_cover_cuts:
        #             for i, g in enumerate(cut_subgradients):
        #                 step_mu = gamma_mu * g / (cut_sizes[i] + eps)
        #                 step_mu = max(-mu_increment_cap, min(mu_increment_cap, step_mu))
        #                 self.best_cut_multipliers[i] = max(
        #                     0.0,
        #                     self.best_cut_multipliers.get(i, 0.0) + step_mu,
        #                 )

        #         # history bookkeeping
        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #         # λ stagnation check
        #         if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0
        #         last_g_lambda = knapsack_subgradient

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                   f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #     # --- After loop: optional cut "cleanup" (drop totally useless cuts) ---
        #     # (optional, you can comment this block out if you prefer)
        #     dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-4)
        #     if self.use_cover_cuts and self.best_cuts:
        #         keep_indices = []
        #         for i, (cut, rhs) in enumerate(self.best_cuts):
        #             mu_i = float(self.best_cut_multipliers.get(i, 0.0))
        #             # if a cut was never violated and μ stayed tiny, drop it for future nodes
        #             if max_cut_violation[i] == 0.0 and abs(mu_i) < dead_mu_threshold:
        #                 continue
        #             keep_indices.append(i)

        #         if len(keep_indices) < len(self.best_cuts):
        #             new_best_cuts = []
        #             new_mu = {}
        #             new_mu_best = {}
        #             new_rhs_eff = {}
        #             for new_idx, old_idx in enumerate(keep_indices):
        #                 new_best_cuts.append(self.best_cuts[old_idx])
        #                 new_mu[new_idx] = float(self.best_cut_multipliers.get(old_idx, 0.0))
        #                 new_mu_best[new_idx] = float(self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0))
        #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]
        #             self.best_cuts = new_best_cuts
        #             self.best_cut_multipliers = new_mu
        #             self.best_cut_multipliers_for_best_bound = new_mu_best
        #             self._rhs_eff = new_rhs_eff

        #     # --- Final cut generation at node boundary (shallow depths only) ---
        #     if cutting_active_here and self.use_cover_cuts and self.best_mst_edges:
        #         try:
        #             final_cands = self.generate_cover_cuts(self.best_mst_edges) or []
        #             existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #             # we rank candidate cuts by their violation on best_mst_edges
        #             best_T = set(self.best_mst_edges)
        #             scored = []
        #             for cut, rhs in final_cands:
        #                 lhs = len(best_T & set(cut))
        #                 violation = lhs - rhs
        #                 if violation >= min_cut_violation_for_add:
        #                     scored.append((violation, cut, rhs))

        #             # keep only strongest few
        #             scored.sort(reverse=True, key=lambda t: t[0])
        #             scored = scored[:max_new_cuts_per_node]

        #             for violation, cut, rhs in scored:
        #                 fz = frozenset(cut)
        #                 if fz in existing:
        #                     # if same support but stronger rhs, replace existing
        #                     old_rhs = existing[fz]
        #                     if rhs > old_rhs:
        #                         idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                         self.best_cuts[idx] = (cut, rhs)
        #                         self._rhs_eff[idx] = int(rhs) - len(set(cut) & F_in)
        #                         existing[fz] = rhs
        #                     continue

        #                 # truly new cut
        #                 self.best_cuts.append((cut, rhs))
        #                 idx_new = len(self.best_cuts) - 1
        #                 self.best_cut_multipliers[idx_new] = 0.0
        #                 self.best_cut_multipliers_for_best_bound[idx_new] = 0.0
        #                 self._rhs_eff[idx_new] = int(rhs) - len(set(cut) & F_in)
        #                 existing[fz] = rhs
        #                 node_new_cuts.append((cut, rhs))

        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error generating final cuts: {e}")

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts


        # else:  # Subgradient method with Polyak hybrid + pre-separation (fixed dual per node)
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams (for λ)
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10

        #     # μ updates: conservative
        #     gamma_mu = getattr(self, "gamma_mu", 0.30)          # smoother than 1.0
        #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
        #     eps = 1e-12

        #     # Depth-based cutting: only generate new cuts at shallow depths
        #     max_cut_depth = getattr(self, "max_cut_depth", 1)
        #     cutting_active_here = self.use_cover_cuts and (depth <= max_cut_depth)

        #     # Cut management in this node
        #     max_active_cuts = getattr(self, "max_active_cuts", 20)      # cuts used in dual at this node
        #     max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)
        #     min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
        #     dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-4)

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Collect new cuts generated at THIS node (to return to the node and children)
        #     node_new_cuts = []

        #     # --- Quick guards ---
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        #     # --- Prepare fixed/excluded ---
        #     F_in = getattr(self, "fixed_edges", set())    # normalized tuples
        #     F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()
        #     edge_idx = self.edge_indices  # normalized edge -> index

        #     # ------------------------------------------------------------------
        #     # 1) PRE-SEPARATION AT NODE START (cuts usable in this node)
        #     # ------------------------------------------------------------------
        #     if cutting_active_here and self.use_cover_cuts:
        #         try:
        #             # priced weights with current λ, μ (for inherited cuts only)
        #             w0 = self.compute_modified_weights()
        #             try:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst_incremental(w0, None)
        #             except Exception:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst()

        #             cand_cuts = self.generate_cover_cuts(mst_edges0) or []

        #             # evaluate violation on this MST
        #             T0 = set(mst_edges0)
        #             scored = []
        #             for cut, rhs in cand_cuts:
        #                 S = set(cut)
        #                 lhs = len(T0 & S)
        #                 violation = lhs - rhs
        #                 if violation >= min_cut_violation_for_add:
        #                     scored.append((violation, S, rhs))

        #             # strongest first, keep only a few
        #             scored.sort(reverse=True, key=lambda t: t[0])
        #             scored = scored[:max_new_cuts_per_node]

        #             existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #             for violation, S, rhs in scored:
        #                 fz = frozenset(S)
        #                 if fz in existing:
        #                     # same support: if stronger rhs, replace
        #                     old_rhs = existing[fz]
        #                     if rhs > old_rhs:
        #                         idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                         self.best_cuts[idx] = (set(S), rhs)
        #                         existing[fz] = rhs
        #                     continue

        #                 # truly new cut
        #                 self.best_cuts.append((set(S), rhs))
        #                 idx_new = len(self.best_cuts) - 1
        #                 self.best_cut_multipliers[idx_new] = 0.0
        #                 self.best_cut_multipliers_for_best_bound[idx_new] = 0.0
        #                 node_new_cuts.append((set(S), rhs))
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error in pre-separation at depth {depth}: {e}")

        #     # ------------------------------------------------------------------
        #     # 2) Compute rhs_eff and detect infeasibility
        #     # ------------------------------------------------------------------
        #     self._rhs_eff = {}
        #     for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #         rhs_eff = int(rhs) - len(cut & F_in)
        #         self._rhs_eff[idx_c] = rhs_eff
        #         if rhs_eff < 0:
        #             # node infeasible due to fixed edges saturating the cut
        #             end_time = time()
        #             LagrangianMST.total_compute_time += end_time - start_time
        #             return float('inf'), self.best_upper_bound, node_new_cuts

        #     # ------------------------------------------------------------------
        #     # 3) Trim number of cuts at node start (keep important ones)
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
        #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
        #         if not parent_mu_map:
        #             parent_mu_map = self.best_cut_multipliers

        #         idx_and_cut = list(enumerate(self.best_cuts))
        #         # priority: large |μ|
        #         idx_and_cut.sort(
        #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
        #             reverse=True,
        #         )
        #         idx_and_cut = idx_and_cut[:max_active_cuts]

        #         new_cuts_list = []
        #         new_mu = {}
        #         new_mu_best = {}
        #         new_rhs_eff = {}
        #         remap = {}
        #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
        #             new_cuts_list.append(cut_rhs)
        #             new_mu[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_mu_best[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_rhs_eff[new_i] = self._rhs_eff[old_i]
        #             remap[old_i] = new_i

        #         self.best_cuts = new_cuts_list
        #         self.best_cut_multipliers = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu_best
        #         self._rhs_eff = new_rhs_eff

        #     # ------------------------------------------------------------------
        #     # 4) Precompute cut -> edge index arrays (FIXED for this node)
        #     # ------------------------------------------------------------------
        #     cut_edge_idx_free = []
        #     cut_free_sizes = []
        #     cut_edge_idx_all = []

        #     for cut, _ in self.best_cuts:
        #         # FREE indices (for subgradients)
        #         idxs_free = [
        #             edge_idx[e] for e in cut
        #             if (e not in F_in and e not in F_out) and (e in edge_idx)
        #         ]
        #         arr_free = np.fromiter(idxs_free, dtype=np.int32) if idxs_free else np.empty(0, dtype=np.int32)
        #         cut_edge_idx_free.append(arr_free)
        #         cut_free_sizes.append(max(1, len(idxs_free)))  # avoid /0

        #         # ALL indices (for dual pricing; includes fixed edges)
        #         idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #         arr_all = np.fromiter(idxs_all, dtype=np.int32) if idxs_all else np.empty(0, dtype=np.int32)
        #         cut_edge_idx_all.append(arr_all)

        #     # stash for compute_modified_weights (if you later want to use them there)
        #     self._cut_edge_idx = cut_edge_idx_free
        #     self._cut_edge_idx_all = cut_edge_idx_all

        #     rhs_eff_vec = np.array(
        #         [self._rhs_eff[i] for i in range(len(cut_edge_idx_free))],
        #         dtype=float
        #     ) if self.best_cuts else np.zeros(0, dtype=float)

        #     rhs_vec = np.array(
        #         [rhs for (_, rhs) in self.best_cuts],
        #         dtype=float
        #     ) if self.best_cuts else np.zeros(0, dtype=float)

        #     # track how "useful" each cut was at this node
        #     max_cut_violation = [0.0 for _ in self.best_cuts]

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     # Seed priced weights so iteration 0 is consistent
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None

        #     last_g_lambda = None  # for stagnation check

        #     # ------------------------------------------------------------------
        #     # 5) Subgradient iterations (dual structure fixed)
        #     # ------------------------------------------------------------------
        #     for iter_num in range(int(max_iter)):
        #         # 1) Solve MST on current priced weights
        #         try:
        #             mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #         except Exception:
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges = self.last_mst_edges

        #         # Prepare weights for next iteration
        #         prev_weights = self.compute_modified_weights()

        #         # 2) Dual & primal bookkeeping
        #         is_feasible = (mst_length <= self.budget)

        #         # (a) primal & UB
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight)
        #                         and not math.isinf(real_weight)
        #                         and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # prune primal_solutions history
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         # (b) Lagrangian dual value
        #         mu_vec = np.fromiter(
        #             (self.best_cut_multipliers.get(i, 0.0) for i in range(len(cut_edge_idx_free))),
        #             dtype=float,
        #             count=len(cut_edge_idx_free),
        #         )
        #         cover_cut_penalty = float(mu_vec @ rhs_vec) if (self.use_cover_cuts and len(rhs_vec) > 0) else 0.0
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound)
        #                 and not math.isinf(lagrangian_bound)
        #                 and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #                 if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #         else:
        #             no_improvement_count += 1

        #         # 3) Subgradients
        #         # knapsack
        #         knapsack_subgradient = float(mst_length - self.budget)

        #         # cuts
        #         mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #         for e in mst_edges:
        #             j = self.edge_indices.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         if self.use_cover_cuts and len(cut_edge_idx_free) > 0:
        #             violations = []
        #             for i, idxs in enumerate(cut_edge_idx_free):
        #                 lhs_i = int(mst_mask[idxs].sum()) if idxs.size else 0
        #                 g_i = lhs_i - rhs_eff_vec[i]
        #                 violations.append(g_i)
        #                 if abs(g_i) > max_cut_violation[i]:
        #                     max_cut_violation[i] = abs(g_i)
        #             violations = np.array(violations, dtype=float)
        #         else:
        #             violations = np.zeros(0, dtype=float)

        #         cut_subgradients = violations.tolist()
        #         cut_sizes = cut_free_sizes

        #         # 4) Step sizes & updates
        #         self.subgradients.append(knapsack_subgradient)

        #         # λ update (Polyak / fallback)
        #         if polyak_enabled and self.best_upper_bound < float('inf'):
        #             gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             alpha = gamma_base * gap / (knapsack_subgradient ** 2 + eps)
        #         else:
        #             alpha = getattr(self, "step_size", 1e-5)

        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # μ updates (if cuts are active)
        #         if self.use_cover_cuts:
        #             for i, g in enumerate(cut_subgradients):
        #                 step_mu = gamma_mu * g / (cut_sizes[i] + eps)
        #                 step_mu = max(-mu_increment_cap, min(mu_increment_cap, step_mu))
        #                 self.best_cut_multipliers[i] = max(
        #                     0.0,
        #                     self.best_cut_multipliers.get(i, 0.0) + step_mu,
        #                 )

        #         # history bookkeeping
        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #         # λ stagnation check
        #         if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0
        #         last_g_lambda = knapsack_subgradient

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                   f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #     # ------------------------------------------------------------------
        #     # 6) Optional: drop "dead" cuts for future nodes
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts:
        #         keep_indices = []
        #         for i, (cut, rhs) in enumerate(self.best_cuts):
        #             mu_i = float(self.best_cut_multipliers.get(i, 0.0))
        #             if max_cut_violation[i] == 0.0 and abs(mu_i) < dead_mu_threshold:
        #                 continue
        #             keep_indices.append(i)

        #         if len(keep_indices) < len(self.best_cuts):
        #             new_best_cuts = []
        #             new_mu = {}
        #             new_mu_best = {}
        #             new_rhs_eff = {}
        #             for new_idx, old_idx in enumerate(keep_indices):
        #                 new_best_cuts.append(self.best_cuts[old_idx])
        #                 new_mu[new_idx] = float(self.best_cut_multipliers.get(old_idx, 0.0))
        #                 new_mu_best[new_idx] = float(self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0))
        #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]
        #             self.best_cuts = new_best_cuts
        #             self.best_cut_multipliers = new_mu
        #             self.best_cut_multipliers_for_best_bound = new_mu_best
        #             self._rhs_eff = new_rhs_eff

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        # else:  # Subgradient method with Polyak hybrid + pre-separation (fixed dual per node)
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams (for λ)
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10

        #     # μ updates: conservative
        #     gamma_mu = getattr(self, "gamma_mu", 0.30)          # smoother than 1.0
        #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
        #     eps = 1e-12

        #     # Depth-based cutting: only generate new cuts at shallow depths
        #     max_cut_depth = getattr(self, "max_cut_depth", 3)
        #     cutting_active_here = self.use_cover_cuts and (depth <= max_cut_depth)

        #     # Cut management in this node
        #     max_active_cuts = getattr(self, "max_active_cuts", 3)      # cuts used in dual at this node
        #     max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)
        #     min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
        #     dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-4)

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Collect new cuts generated at THIS node (to return to the node and children)
        #     node_new_cuts = []

        #     # --- Quick guards ---
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        #     # --- Prepare fixed/excluded ---
        #     F_in = getattr(self, "fixed_edges", set())    # normalized tuples
        #     F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()
        #     edge_idx = self.edge_indices  # normalized edge -> index

        #     # ------------------------------------------------------------------
        #     # 1) PRE-SEPARATION AT NODE START (cuts usable in this node)
        #     # ------------------------------------------------------------------
        #     if cutting_active_here and self.use_cover_cuts:
        #         try:
        #             # priced weights with current λ, μ (for inherited cuts only)
        #             w0 = self.compute_modified_weights()
        #             try:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst_incremental(w0, None)
        #             except Exception:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst()

        #             cand_cuts = self.generate_cover_cuts(mst_edges0) or []

        #             # evaluate violation on this MST
        #             T0 = set(mst_edges0)
        #             scored = []
        #             for cut, rhs in cand_cuts:
        #                 S = set(cut)
        #                 lhs = len(T0 & S)
        #                 violation = lhs - rhs
        #                 if violation >= min_cut_violation_for_add:
        #                     scored.append((violation, S, rhs))

        #             # strongest first, keep only a few
        #             scored.sort(reverse=True, key=lambda t: t[0])
        #             scored = scored[:max_new_cuts_per_node]

        #             existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #             for violation, S, rhs in scored:
        #                 fz = frozenset(S)
        #                 if fz in existing:
        #                     # same support: if stronger rhs, replace
        #                     old_rhs = existing[fz]
        #                     if rhs > old_rhs:
        #                         idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                         self.best_cuts[idx] = (set(S), rhs)
        #                         existing[fz] = rhs
        #                     continue

        #                 # truly new cut
        #                 self.best_cuts.append((set(S), rhs))
        #                 idx_new = len(self.best_cuts) - 1
        #                 self.best_cut_multipliers[idx_new] = 0.0
        #                 self.best_cut_multipliers_for_best_bound[idx_new] = 0.0
        #                 node_new_cuts.append((set(S), rhs))
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error in pre-separation at depth {depth}: {e}")

        #     # ------------------------------------------------------------------
        #     # 2) Compute rhs_eff and detect infeasibility
        #     # ------------------------------------------------------------------
        #     self._rhs_eff = {}
        #     for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #         rhs_eff = int(rhs) - len(cut & F_in)
        #         self._rhs_eff[idx_c] = rhs_eff
        #         if rhs_eff < 0:
        #             # node infeasible due to fixed edges saturating the cut
        #             end_time = time()
        #             LagrangianMST.total_compute_time += end_time - start_time
        #             return float('inf'), self.best_upper_bound, node_new_cuts

        #     # ------------------------------------------------------------------
        #     # 3) Trim number of cuts at node start (keep important ones)
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
        #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
        #         if not parent_mu_map:
        #             parent_mu_map = self.best_cut_multipliers

        #         idx_and_cut = list(enumerate(self.best_cuts))
        #         # priority: large |μ|
        #         idx_and_cut.sort(
        #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
        #             reverse=True,
        #         )
        #         idx_and_cut = idx_and_cut[:max_active_cuts]

        #         new_cuts_list = []
        #         new_mu = {}
        #         new_mu_best = {}
        #         new_rhs_eff = {}
        #         remap = {}
        #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
        #             new_cuts_list.append(cut_rhs)
        #             new_mu[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_mu_best[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_rhs_eff[new_i] = self._rhs_eff[old_i]
        #             remap[old_i] = new_i

        #         self.best_cuts = new_cuts_list
        #         self.best_cut_multipliers = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu_best
        #         self._rhs_eff = new_rhs_eff

        #     # ------------------------------------------------------------------
        #     # 4) Precompute cut -> edge index arrays (FIXED for this node)
        #     # ------------------------------------------------------------------
        #     cut_edge_idx_free = []
        #     cut_free_sizes = []
        #     cut_edge_idx_all = []

        #     for cut, _ in self.best_cuts:
        #         # FREE indices (for subgradients)
        #         idxs_free = [
        #             edge_idx[e] for e in cut
        #             if (e not in F_in and e not in F_out) and (e in edge_idx)
        #         ]
        #         arr_free = np.fromiter(idxs_free, dtype=np.int32) if idxs_free else np.empty(0, dtype=np.int32)
        #         cut_edge_idx_free.append(arr_free)
        #         cut_free_sizes.append(max(1, len(idxs_free)))  # avoid /0

        #         # ALL indices (for dual pricing; includes fixed edges)
        #         idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #         arr_all = np.fromiter(idxs_all, dtype=np.int32) if idxs_all else np.empty(0, dtype=np.int32)
        #         cut_edge_idx_all.append(arr_all)

        #     # stash for compute_modified_weights (if you later want to use them there)
        #     self._cut_edge_idx = cut_edge_idx_free
        #     self._cut_edge_idx_all = cut_edge_idx_all

        #     rhs_eff_vec = np.array(
        #         [self._rhs_eff[i] for i in range(len(cut_edge_idx_free))],
        #         dtype=float
        #     ) if self.best_cuts else np.zeros(0, dtype=float)

        #     rhs_vec = np.array(
        #         [rhs for (_, rhs) in self.best_cuts],
        #         dtype=float
        #     ) if self.best_cuts else np.zeros(0, dtype=float)

        #     # track how "useful" each cut was at this node
        #     max_cut_violation = [0.0 for _ in self.best_cuts]

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     # Seed priced weights so iteration 0 is consistent
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None

        #     last_g_lambda = None  # for stagnation check

        #     # ------------------------------------------------------------------
        #     # 5) Subgradient iterations (dual structure fixed)
        #     # ------------------------------------------------------------------
        #     for iter_num in range(int(max_iter)):
        #         # 1) Solve MST on current priced weights
        #         try:
        #             mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #         except Exception:
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges = self.last_mst_edges

        #         # Prepare weights for next iteration
        #         prev_weights = self.compute_modified_weights()

        #         # 2) Dual & primal bookkeeping
        #         is_feasible = (mst_length <= self.budget)

        #         # (a) primal & UB
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight)
        #                         and not math.isinf(real_weight)
        #                         and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # prune primal_solutions history
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         # (b) Lagrangian dual value
        #         mu_vec = np.fromiter(
        #             (self.best_cut_multipliers.get(i, 0.0) for i in range(len(cut_edge_idx_free))),
        #             dtype=float,
        #             count=len(cut_edge_idx_free),
        #         )
        #         cover_cut_penalty = float(mu_vec @ rhs_vec) if (self.use_cover_cuts and len(rhs_vec) > 0) else 0.0
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound)
        #                 and not math.isinf(lagrangian_bound)
        #                 and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #                 if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #         else:
        #             no_improvement_count += 1

        #         # 3) Subgradients
        #         # knapsack
        #         knapsack_subgradient = float(mst_length - self.budget)

        #         # cuts
        #         mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #         for e in mst_edges:
        #             j = self.edge_indices.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         if self.use_cover_cuts and len(cut_edge_idx_free) > 0:
        #             violations = []
        #             for i, idxs in enumerate(cut_edge_idx_free):
        #                 lhs_i = int(mst_mask[idxs].sum()) if idxs.size else 0
        #                 g_i = lhs_i - rhs_eff_vec[i]
        #                 violations.append(g_i)
        #                 if abs(g_i) > max_cut_violation[i]:
        #                     max_cut_violation[i] = abs(g_i)
        #             violations = np.array(violations, dtype=float)
        #         else:
        #             violations = np.zeros(0, dtype=float)

        #         # print("vilte", violations)
        #         cut_subgradients = violations.tolist()
        #         cut_sizes = cut_free_sizes

        #         # 4) Step sizes & updates
        #         # self.subgradients.append(knapsack_subgradient)

        #         # # λ update (Polyak / fallback)
        #         # if polyak_enabled and self.best_upper_bound < float('inf'):
        #         #     gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #         #     alpha = gamma_base * gap / (knapsack_subgradient ** 2 + eps)
        #         # else:
        #         #     alpha = getattr(self, "step_size", 1e-5)

        #         # v_prev = getattr(self, "_v_lambda", 0.0)
        #         # v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         # self._v_lambda = v_new
        #         # self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # # μ updates (if cuts are active)
        #         # if self.use_cover_cuts:
        #         #     for i, g in enumerate(cut_subgradients):
        #         #         step_mu = gamma_mu * g / (cut_sizes[i] + eps)
        #         #         step_mu = max(-mu_increment_cap, min(mu_increment_cap, step_mu))
        #         #         self.best_cut_multipliers[i] = max(
        #         #             0.0,
        #         #             self.best_cut_multipliers.get(i, 0.0) + step_mu,
        #         #         )
        #         # 4) Step sizes & updates (joint Polyak for λ and μ)
        #         self.subgradients.append(knapsack_subgradient)

        #         # --- joint gradient norm ---
        #         # g_lambda is scalar, cut_subgradients is a list of g_mu_i
        #         norm_sq = knapsack_subgradient ** 2
        #         for g in cut_subgradients:
        #             norm_sq += g ** 2

        #         # --- Polyak-like step size for the joint dual (λ, μ) ---
        #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #             gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             # theta plays the same role as gamma_base, but for the joint gradient
        #             theta = gamma_base  # you can tune this (e.g. 0.1)
        #             alpha = theta * gap / (norm_sq + eps)
        #         else:
        #             # fallback small constant if UB is infinite or gradient is zero
        #             alpha = getattr(self, "step_size", 1e-5)

        #         # --- λ update with momentum, using the joint α ---
        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # --- μ updates (only if cuts are active), using the SAME α ---
        #         if self.use_cover_cuts:
        #             for i, g in enumerate(cut_subgradients):
        #                 # joint Polyak step
        #                 delta = alpha * g

        #                 # optional safety cap (keeps steps moderate)
        #                 if mu_increment_cap is not None:
        #                     if delta > mu_increment_cap:
        #                         delta = mu_increment_cap
        #                     elif delta < -mu_increment_cap:
        #                         delta = -mu_increment_cap

        #                 mu_old = self.best_cut_multipliers.get(i, 0.0)
        #                 mu_new = mu_old + delta

        #                 # project to μ_i >= 0
        #                 if mu_new < 0.0:
        #                     mu_new = 0.0

        #                 self.best_cut_multipliers[i] = mu_new
 


        #         # history bookkeeping
        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #         # λ stagnation check
        #         if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0
        #         last_g_lambda = knapsack_subgradient

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                   f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #     # ------------------------------------------------------------------
        #     # 6) Optional: drop "dead" cuts for future nodes
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts:
        #         keep_indices = []
        #         for i, (cut, rhs) in enumerate(self.best_cuts):
        #             mu_i = float(self.best_cut_multipliers.get(i, 0.0))
        #             if max_cut_violation[i] == 0.0 and abs(mu_i) < dead_mu_threshold:
        #                 continue
        #             keep_indices.append(i)

        #         if len(keep_indices) < len(self.best_cuts):
        #             new_best_cuts = []
        #             new_mu = {}
        #             new_mu_best = {}
        #             new_rhs_eff = {}
        #             for new_idx, old_idx in enumerate(keep_indices):
        #                 new_best_cuts.append(self.best_cuts[old_idx])
        #                 new_mu[new_idx] = float(self.best_cut_multipliers.get(old_idx, 0.0))
        #                 new_mu_best[new_idx] = float(self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0))
        #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]
        #             self.best_cuts = new_best_cuts
        #             self.best_cut_multipliers = new_mu
        #             self.best_cut_multipliers_for_best_bound = new_mu_best
        #             self._rhs_eff = new_rhs_eff

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        
        


        # else:  # Subgradient method with Polyak hybrid + pre-separation (fixed dual per node, global flag)
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams (for λ)
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10

        #     # μ updates: conservative
        #     gamma_mu = getattr(self, "gamma_mu", 0.30)          # smoother than 1.0 (optional)
        #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
        #     eps = 1e-12

        #     # ------------------------------------------------------------------
        #     # Depth-based cutting + global fallback:
        #     # - Normally: cut only if depth <= max_cut_depth
        #     # - But: until ANY node sees a violated cut, allow cutting at all depths.
        #     # ------------------------------------------------------------------
        #     max_cut_depth = getattr(self, "max_cut_depth", 3)

        #     # has *any* node so far in this solve seen a violated cut?
        #     global_violation_seen = getattr(LagrangianMST, "global_cut_violation_seen", False)

        #     # cutting is active here if:
        #     #   - cover cuts are on, and
        #     #   - (depth is shallow OR we haven't seen any violated cuts yet)
        #     cutting_active_here = self.use_cover_cuts and (
        #         depth <= max_cut_depth or not global_violation_seen
        #     )

        #     # Cut management in this node
        #     max_active_cuts = getattr(self, "max_active_cuts", 3)      # cuts used in dual at this node
        #     max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)
        #     min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
        #     dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-4)

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Collect new cuts generated at THIS node (to return to the node and children)
        #     node_new_cuts = []

        #     # --- Quick guards ---
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        #     # --- Prepare fixed/excluded ---
        #     F_in = getattr(self, "fixed_edges", set())    # normalized tuples
        #     F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()
        #     edge_idx = self.edge_indices  # normalized edge -> index

        #     # ------------------------------------------------------------------
        #     # 1) PRE-SEPARATION AT NODE START (cuts usable in this node)
        #     # ------------------------------------------------------------------
        #     if cutting_active_here and self.use_cover_cuts:
        #         try:
        #             # priced weights with current λ, μ (for inherited cuts only)
        #             w0 = self.compute_modified_weights()
        #             try:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst_incremental(w0, None)
        #             except Exception:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst()

        #             cand_cuts = self.generate_cover_cuts(mst_edges0) or []

        #             # evaluate violation on this MST (using ORIGINAL rhs)
        #             T0 = set(mst_edges0)
        #             scored = []
        #             for cut, rhs in cand_cuts:
        #                 S = set(cut)
        #                 lhs = len(T0 & S)
        #                 violation = lhs - rhs
        #                 if violation >= min_cut_violation_for_add:
        #                     scored.append((violation, S, rhs))

        #             # strongest first, keep only a few
        #             scored.sort(reverse=True, key=lambda t: t[0])
        #             scored = scored[:max_new_cuts_per_node]

        #             existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #             for violation, S, rhs in scored:
        #                 fz = frozenset(S)
        #                 if fz in existing:
        #                     # same support: if stronger rhs, replace
        #                     old_rhs = existing[fz]
        #                     if rhs > old_rhs:
        #                         idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                         self.best_cuts[idx] = (set(S), rhs)
        #                         existing[fz] = rhs
        #                     continue

        #                 # truly new cut
        #                 self.best_cuts.append((set(S), rhs))
        #                 idx_new = len(self.best_cuts) - 1
        #                 self.best_cut_multipliers[idx_new] = 0.0
        #                 self.best_cut_multipliers_for_best_bound[idx_new] = 0.0
        #                 node_new_cuts.append((set(S), rhs))
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error in pre-separation at depth {depth}: {e}")

        #     # ------------------------------------------------------------------
        #     # 2) Compute rhs_eff and detect infeasibility (node-level)
        #     # ------------------------------------------------------------------
        #     self._rhs_eff = {}
        #     for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #         rhs_eff = int(rhs) - len(cut & F_in)
        #         self._rhs_eff[idx_c] = rhs_eff
        #         if rhs_eff < 0:
        #             # node infeasible due to fixed edges saturating the cut
        #             end_time = time()
        #             LagrangianMST.total_compute_time += end_time - start_time
        #             return float('inf'), self.best_upper_bound, node_new_cuts

        #     # ------------------------------------------------------------------
        #     # 3) Trim number of cuts at node start (keep important ones)
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
        #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
        #         if not parent_mu_map:
        #             parent_mu_map = self.best_cut_multipliers

        #         idx_and_cut = list(enumerate(self.best_cuts))
        #         # priority: large |μ|
        #         idx_and_cut.sort(
        #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
        #             reverse=True,
        #         )
        #         idx_and_cut = idx_and_cut[:max_active_cuts]

        #         new_cuts_list = []
        #         new_mu = {}
        #         new_mu_best = {}
        #         new_rhs_eff = {}
        #         remap = {}
        #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
        #             new_cuts_list.append(cut_rhs)
        #             new_mu[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_mu_best[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_rhs_eff[new_i] = self._rhs_eff[old_i]
        #             remap[old_i] = new_i

        #         self.best_cuts = new_cuts_list
        #         self.best_cut_multipliers = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu_best
        #         self._rhs_eff = new_rhs_eff

        #     # ------------------------------------------------------------------
        #     # 4) Precompute cut -> edge index arrays (FIXED for this node)
        #     # ------------------------------------------------------------------
        #     cut_edge_idx_free = []
        #     cut_free_sizes = []
        #     cut_edge_idx_all = []

        #     for cut, _ in self.best_cuts:
        #         # FREE indices (for node-level heuristics; not used in μ-gradient)
        #         idxs_free = [
        #             edge_idx[e] for e in cut
        #             if (e not in F_in and e not in F_out) and (e in edge_idx)
        #         ]
        #         arr_free = np.fromiter(idxs_free, dtype=np.int32) if idxs_free else np.empty(0, dtype=np.int32)
        #         cut_edge_idx_free.append(arr_free)
        #         cut_free_sizes.append(max(1, len(idxs_free)))  # avoid /0

        #         # ALL indices (for dual pricing & true μ-subgradient)
        #         idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #         arr_all = np.fromiter(idxs_all, dtype=np.int32) if idxs_all else np.empty(0, dtype=np.int32)
        #         cut_edge_idx_all.append(arr_all)

        #     # stash for compute_modified_weights (used for pricing)
        #     self._cut_edge_idx = cut_edge_idx_free
        #     self._cut_edge_idx_all = cut_edge_idx_all

        #     rhs_eff_vec = np.array(
        #         [self._rhs_eff[i] for i in range(len(cut_edge_idx_free))],
        #         dtype=float
        #     ) if self.best_cuts else np.zeros(0, dtype=float)

        #     rhs_vec = np.array(
        #         [rhs for (_, rhs) in self.best_cuts],
        #         dtype=float
        #     ) if self.best_cuts else np.zeros(0, dtype=float)

        #     # track how "useful" each cut was at this node (only positive violation)
        #     max_cut_violation = [0.0 for _ in self.best_cuts]

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     # Seed priced weights so iteration 0 is consistent
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None

        #     last_g_lambda = None  # for stagnation check

        #     # ------------------------------------------------------------------
        #     # 5) Subgradient iterations (dual structure fixed)
        #     # ------------------------------------------------------------------
        #     for iter_num in range(int(max_iter)):
        #         # 1) Solve MST on current priced weights
        #         try:
        #             mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #         except Exception:
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges = self.last_mst_edges

        #         # Prepare weights for next iteration
        #         prev_weights = self.compute_modified_weights()

        #         # 2) Dual & primal bookkeeping
        #         is_feasible = (mst_length <= self.budget)

        #         # (a) primal & UB
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight)
        #                         and not math.isinf(real_weight)
        #                         and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # prune primal_solutions history
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         # (b) Lagrangian dual value (dualized cuts use ORIGINAL rhs)
        #         mu_vec = np.fromiter(
        #             (self.best_cut_multipliers.get(i, 0.0) for i in range(len(cut_edge_idx_free))),
        #             dtype=float,
        #             count=len(cut_edge_idx_free),
        #         )
        #         cover_cut_penalty = float(mu_vec @ rhs_vec) if (self.use_cover_cuts and len(rhs_vec) > 0) else 0.0
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound)
        #                 and not math.isinf(lagrangian_bound)
        #                 and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #                 if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #         else:
        #             no_improvement_count += 1

        #         # 3) Subgradients
        #         # knapsack
        #         knapsack_subgradient = float(mst_length - self.budget)

        #         # cuts: TRUE dual subgradient for the encoded Lagrangian:
        #         # g_i = lhs_i - rhs_i, lhs_i counts ALL edges in S_i selected by MST
        #         mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #         for e in mst_edges:
        #             j = self.edge_indices.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         if self.use_cover_cuts and len(cut_edge_idx_all) > 0:
        #             violations = []
        #             for i, idxs_all in enumerate(cut_edge_idx_all):
        #                 lhs_i = int(mst_mask[idxs_all].sum()) if idxs_all.size else 0
        #                 g_i = lhs_i - rhs_vec[i]  # <-- consistent with lagrangian_bound
        #                 violations.append(g_i)
        #                 # only *positive* violation counts as "usefulness"
        #                 if g_i > max_cut_violation[i]:
        #                     max_cut_violation[i] = g_i
        #             violations = np.array(violations, dtype=float)
        #         else:
        #             violations = np.zeros(0, dtype=float)

        #         cut_subgradients = violations.tolist()

        #         # 4) Step sizes & updates (joint Polyak for λ and μ)
        #         self.subgradients.append(knapsack_subgradient)

        #         # --- joint gradient norm ---
        #         norm_sq = knapsack_subgradient ** 2
        #         for g in cut_subgradients:
        #             norm_sq += g ** 2

        #         # --- Polyak-like step size for the joint dual (λ, μ) ---
        #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #             gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             theta = gamma_base  # role similar to gamma_base, but for the joint gradient
        #             alpha = theta * gap / (norm_sq + eps)
        #         else:
        #             # fallback small constant if UB is infinite or gradient is zero
        #             alpha = getattr(self, "step_size", 1e-5)

        #         # --- λ update with momentum, using the joint α ---
        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # --- μ updates (only if cuts are active), using the SAME α ---
        #         if self.use_cover_cuts:
        #             for i, g in enumerate(cut_subgradients):
        #                 # joint Polyak step; you can scale with gamma_mu if you like:
        #                 # delta = gamma_mu * alpha * g
        #                 delta = alpha * g

        #                 # optional safety cap (keeps steps moderate)
        #                 if mu_increment_cap is not None:
        #                     if delta > mu_increment_cap:
        #                         delta = mu_increment_cap
        #                     elif delta < -mu_increment_cap:
        #                         delta = -mu_increment_cap

        #                 mu_old = self.best_cut_multipliers.get(i, 0.0)
        #                 mu_new = mu_old + delta

        #                 # project to μ_i >= 0
        #                 if mu_new < 0.0:
        #                     mu_new = 0.0

        #                 self.best_cut_multipliers[i] = mu_new

        #         # history bookkeeping
        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #         # λ stagnation check
        #         if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0
        #         last_g_lambda = knapsack_subgradient

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                 f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #     # ------------------------------------------------------------------
        #     # 6) Optional: drop "dead" cuts for future nodes
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts:
        #         keep_indices = []
        #         for i, (cut, rhs) in enumerate(self.best_cuts):
        #             mu_i = float(self.best_cut_multipliers.get(i, 0.0))
        #             # drop if never strictly violated HERE and μ stayed tiny
        #             if max_cut_violation[i] == 0.0 and abs(mu_i) < dead_mu_threshold:
        #                 continue
        #             keep_indices.append(i)

        #         if len(keep_indices) < len(self.best_cuts):
        #             new_best_cuts = []
        #             new_mu = {}
        #             new_mu_best = {}
        #             new_rhs_eff = {}
        #             for new_idx, old_idx in enumerate(keep_indices):
        #                 new_best_cuts.append(self.best_cuts[old_idx])
        #                 new_mu[new_idx] = float(self.best_cut_multipliers.get(old_idx, 0.0))
        #                 new_mu_best[new_idx] = float(self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0))
        #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]
        #             self.best_cuts = new_best_cuts
        #             self.best_cut_multipliers = new_mu
        #             self.best_cut_multipliers_for_best_bound = new_mu_best
        #             self._rhs_eff = new_rhs_eff

        #     # ------------------------------------------------------------------
        #     # 7) Update global flag if this node saw any violated cut
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and max_cut_violation:
        #         if any(v > 0.0 for v in max_cut_violation):
        #             # first time this is called, attribute is created automatically
        #             LagrangianMST.global_cut_violation_seen = True

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        # else:  # Subgradient method with Polyak hybrid + pre- & in-loop separation
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams (for λ)
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10

        #     # μ updates: conservative
        #     gamma_mu = getattr(self, "gamma_mu", 0.30)            # smoother than 1.0 (optional)
        #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
        #     eps = 1e-12

        #     # ------------------------------------------------------------------
        #     # Depth-based cutting + global fallback:
        #     # - Normally: cut only if depth <= max_cut_depth
        #     # - But: until ANY node sees a violated cut, allow cutting at all depths.
        #     # ------------------------------------------------------------------
        #     max_cut_depth = getattr(self, "max_cut_depth", 4)

        #     # Node-level separation controls
        #     node_cut_frequency = getattr(self, "node_cut_frequency", 10)  # in-loop separation period
        #     max_active_cuts = getattr(self, "max_active_cuts", 5)         # max cuts used in dual at this node
        #     max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)    # pre-separation cap
        #     max_inloop_new_cuts = getattr(self, "max_inloop_new_cuts", 2)        # in-loop cap
        #     min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
        #     dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-4)

        #     # Has *any* node so far in this solve seen a violated cut?
        #     global_violation_seen = getattr(LagrangianMST, "global_cut_violation_seen", False)

        #     # Cutting is active here if:
        #     #   - cover cuts are on, and
        #     #   - (depth is shallow OR we haven't seen any violated cuts yet)
        #     cutting_active_here = self.use_cover_cuts and (
        #         depth <= max_cut_depth or not global_violation_seen
        #     )

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Collect new cuts generated at THIS node (pre + in-loop) to return
        #     node_new_cuts = []

        #     # --- Quick guards ---
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        #     # --- Prepare fixed/excluded ---
        #     F_in = getattr(self, "fixed_edges", set())    # normalized tuples
        #     F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()
        #     edge_idx = self.edge_indices  # normalized edge -> index

        #     # ------------------------------------------------------------------
        #     # 1) PRE-SEPARATION AT NODE START (cuts usable in this node)
        #     # ------------------------------------------------------------------
        #     if cutting_active_here and self.use_cover_cuts:
        #         try:
        #             # priced weights with current λ, μ (for inherited cuts only)
        #             w0 = self.compute_modified_weights()
        #             try:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst_incremental(w0, None)
        #             except Exception:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst()

        #             cand_cuts = self.generate_cover_cuts(mst_edges0) or []

        #             # evaluate violation on this MST (using ORIGINAL rhs)
        #             T0 = set(mst_edges0)
        #             scored = []
        #             for cut, rhs in cand_cuts:
        #                 S = set(cut)
        #                 lhs = len(T0 & S)
        #                 violation = lhs - rhs
        #                 if violation >= min_cut_violation_for_add:
        #                     scored.append((violation, S, rhs))

        #             # strongest first
        #             scored.sort(reverse=True, key=lambda t: t[0])

        #             # respect caps: max_new_cuts_per_node AND max_active_cuts
        #             available_slots = max(0, max_active_cuts - len(self.best_cuts))
        #             if available_slots > 0:
        #                 scored = scored[:min(max_new_cuts_per_node, available_slots)]
        #             else:
        #                 scored = []

        #             existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #             for violation, S, rhs in scored:
        #                 fz = frozenset(S)
        #                 if fz in existing:
        #                     # same support: if stronger rhs, replace
        #                     old_rhs = existing[fz]
        #                     if rhs > old_rhs:
        #                         idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                         self.best_cuts[idx] = (set(S), rhs)
        #                         existing[fz] = rhs
        #                     continue

        #                 # truly new cut
        #                 self.best_cuts.append((set(S), rhs))
        #                 idx_new = len(self.best_cuts) - 1
        #                 self.best_cut_multipliers[idx_new] = 0.0
        #                 self.best_cut_multipliers_for_best_bound[idx_new] = 0.0
        #                 node_new_cuts.append((set(S), rhs))
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error in pre-separation at depth {depth}: {e}")

        #     # ------------------------------------------------------------------
        #     # 2) Compute rhs_eff and detect infeasibility (node-level)
        #     # ------------------------------------------------------------------
        #     self._rhs_eff = {}
        #     for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #         rhs_eff = int(rhs) - len(cut & F_in)
        #         self._rhs_eff[idx_c] = rhs_eff
        #         if rhs_eff < 0:
        #             # node infeasible due to fixed edges saturating the cut
        #             end_time = time()
        #             LagrangianMST.total_compute_time += end_time - start_time
        #             return float('inf'), self.best_upper_bound, node_new_cuts

        #     # ------------------------------------------------------------------
        #     # 3) Trim number of cuts at node start (keep important ones)
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
        #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
        #         if not parent_mu_map:
        #             parent_mu_map = self.best_cut_multipliers

        #         idx_and_cut = list(enumerate(self.best_cuts))
        #         # priority: large |μ|
        #         idx_and_cut.sort(
        #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
        #             reverse=True,
        #         )
        #         idx_and_cut = idx_and_cut[:max_active_cuts]

        #         new_cuts_list = []
        #         new_mu = {}
        #         new_mu_best = {}
        #         new_rhs_eff = {}
        #         remap = {}
        #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
        #             new_cuts_list.append(cut_rhs)
        #             new_mu[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_mu_best[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_rhs_eff[new_i] = self._rhs_eff[old_i]
        #             remap[old_i] = new_i

        #         self.best_cuts = new_cuts_list
        #         self.best_cut_multipliers = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu_best
        #         self._rhs_eff = new_rhs_eff

        #     # ------------------------------------------------------------------
        #     # 4) Precompute cut -> edge index arrays (FIXED for this node, but
        #     #    we will rebuild them if we add cuts in-loop)
        #     # ------------------------------------------------------------------
        #     def _rebuild_cut_structures():
        #         """Rebuild index arrays and rhs vectors from self.best_cuts & self._rhs_eff."""
        #         nonlocal cut_edge_idx_free, cut_free_sizes, cut_edge_idx_all, rhs_eff_vec, rhs_vec

        #         cut_edge_idx_free = []
        #         cut_free_sizes = []
        #         cut_edge_idx_all = []

        #         for cut, rhs in self.best_cuts:
        #             # FREE indices (for possible future refinements – not used in μ-gradient)
        #             idxs_free = [
        #                 edge_idx[e] for e in cut
        #                 if (e not in F_in and e not in F_out) and (e in edge_idx)
        #             ]
        #             arr_free = np.fromiter(idxs_free, dtype=np.int32) if idxs_free else np.empty(0, dtype=np.int32)
        #             cut_edge_idx_free.append(arr_free)
        #             cut_free_sizes.append(max(1, len(idxs_free)))  # avoid /0

        #             # ALL indices (for dual pricing & true μ-subgradient)
        #             idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #             arr_all = np.fromiter(idxs_all, dtype=np.int32) if idxs_all else np.empty(0, dtype=np.int32)
        #             cut_edge_idx_all.append(arr_all)

        #         # stash for compute_modified_weights (used for pricing)
        #         self._cut_edge_idx = cut_edge_idx_free
        #         self._cut_edge_idx_all = cut_edge_idx_all

        #         if self.best_cuts:
        #             rhs_eff_vec = np.array(
        #                 [self._rhs_eff[i] for i in range(len(cut_edge_idx_free))],
        #                 dtype=float
        #             )
        #             rhs_vec = np.array(
        #                 [rhs for (_, rhs) in self.best_cuts],
        #                 dtype=float
        #             )
        #         else:
        #             rhs_eff_vec = np.zeros(0, dtype=float)
        #             rhs_vec = np.zeros(0, dtype=float)

        #     # initialize structures
        #     cut_edge_idx_free = []
        #     cut_free_sizes = []
        #     cut_edge_idx_all = []
        #     rhs_eff_vec = np.zeros(0, dtype=float)
        #     rhs_vec = np.zeros(0, dtype=float)
        #     _rebuild_cut_structures()

        #     # track how "useful" each cut was at this node (only positive violation)
        #     max_cut_violation = [0.0 for _ in self.best_cuts]

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     # Seed priced weights so iteration 0 is consistent
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None

        #     last_g_lambda = None  # for stagnation check

        #     # ------------------------------------------------------------------
        #     # 5) Subgradient iterations (dual structure mostly fixed, but we may
        #     #    add a few extra cuts in-loop with full rebuild).
        #     # ------------------------------------------------------------------
        #     for iter_num in range(int(max_iter)):
        #         # 1) Solve MST on current priced weights
        #         try:
        #             mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #         except Exception:
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges = self.last_mst_edges

        #         # ------------------------------------------------------------------
        #         # 5a) OCCASIONAL IN-LOOP SEPARATION (lightweight, optional)
        #         # ------------------------------------------------------------------
        #         if (
        #             cutting_active_here
        #             and self.use_cover_cuts
        #             and (iter_num % node_cut_frequency == 0)
        #             and len(self.best_cuts) < max_active_cuts
        #         ):
        #             try:
        #                 cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []

        #                 T_loop = set(mst_edges)
        #                 scored_loop = []
        #                 for cut, rhs in cand_cuts_loop:
        #                     S = set(cut)
        #                     lhs = len(T_loop & S)
        #                     violation = lhs - rhs
        #                     if violation >= min_cut_violation_for_add:
        #                         scored_loop.append((violation, S, rhs))

        #                 scored_loop.sort(reverse=True, key=lambda t: t[0])

        #                 remaining_slots = max(0, max_active_cuts - len(self.best_cuts))
        #                 if remaining_slots > 0:
        #                     scored_loop = scored_loop[:min(max_inloop_new_cuts, remaining_slots)]
        #                 else:
        #                     scored_loop = []

        #                 existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #                 added_any = False
        #                 for violation, S, rhs in scored_loop:
        #                     fz = frozenset(S)
        #                     if fz in existing:
        #                         # if we ever wanted to strengthen rhs here we could, but usually in-loop we skip
        #                         continue

        #                     # truly new cut
        #                     self.best_cuts.append((set(S), rhs))
        #                     new_idx = len(self.best_cuts) - 1
        #                     self.best_cut_multipliers[new_idx] = 0.0
        #                     self.best_cut_multipliers_for_best_bound[new_idx] = 0.0
        #                     self._rhs_eff[new_idx] = int(rhs) - len(set(S) & F_in)
        #                     max_cut_violation.append(0.0)
        #                     node_new_cuts.append((set(S), rhs))
        #                     added_any = True

        #                 if added_any:
        #                     # Rebuild index structures & rhs vectors to incorporate new cuts
        #                     _rebuild_cut_structures()
        #                     # Reset modified weights cache with new μ dimension
        #                     self._mw_cached = None
        #                     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error in in-loop separation at depth {depth}, iter {iter_num}: {e}")

        #         # Prepare weights for next iteration (using current λ, μ and possibly updated cuts)
        #         prev_weights = self.compute_modified_weights()

        #         # 2) Dual & primal bookkeeping
        #         is_feasible = (mst_length <= self.budget)

        #         # (a) primal & UB
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight)
        #                         and not math.isinf(real_weight)
        #                         and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # prune primal_solutions history
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         # (b) Lagrangian dual value (dualized cuts use ORIGINAL rhs)
        #         if self.use_cover_cuts and len(rhs_vec) > 0:
        #             mu_vec = np.fromiter(
        #                 (self.best_cut_multipliers.get(i, 0.0) for i in range(len(rhs_vec))),
        #                 dtype=float,
        #                 count=len(rhs_vec),
        #             )
        #             cover_cut_penalty = float(mu_vec @ rhs_vec)
        #         else:
        #             mu_vec = np.zeros(len(rhs_vec), dtype=float)
        #             cover_cut_penalty = 0.0

        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound)
        #                 and not math.isinf(lagrangian_bound)
        #                 and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #                 if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #         else:
        #             no_improvement_count += 1

        #         # 3) Subgradients
        #         # knapsack
        #         knapsack_subgradient = float(mst_length - self.budget)

        #         # cuts: TRUE dual subgradient for the encoded Lagrangian:
        #         # g_i = lhs_i - rhs_i, lhs_i counts ALL edges in S_i selected by MST
        #         mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #         for e in mst_edges:
        #             j = self.edge_indices.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         if self.use_cover_cuts and len(cut_edge_idx_all) > 0:
        #             violations = []
        #             for i, idxs_all in enumerate(cut_edge_idx_all):
        #                 lhs_i = int(mst_mask[idxs_all].sum()) if idxs_all.size else 0
        #                 g_i = lhs_i - rhs_vec[i]  # <-- consistent with lagrangian_bound
        #                 violations.append(g_i)
        #                 # only *positive* violation counts as "usefulness"
        #                 if g_i > max_cut_violation[i]:
        #                     max_cut_violation[i] = g_i
        #             violations = np.array(violations, dtype=float)
        #         else:
        #             violations = np.zeros(0, dtype=float)

        #         cut_subgradients = violations.tolist()

        #         # 4) Step sizes & updates (joint Polyak for λ and μ)
        #         self.subgradients.append(knapsack_subgradient)

        #         # --- joint gradient norm ---
        #         norm_sq = knapsack_subgradient ** 2
        #         for g in cut_subgradients:
        #             norm_sq += g ** 2

        #         # --- Polyak-like step size for the joint dual (λ, μ) ---
        #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #             gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             theta = gamma_base
        #             alpha = theta * gap / (norm_sq + eps)
        #         else:
        #             # fallback small constant if UB is infinite or gradient is zero
        #             alpha = getattr(self, "step_size", 1e-5)

        #         # --- λ update with momentum, using the joint α ---
        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # --- μ updates (only if cuts are active), using the SAME α (scaled) ---
        #         if self.use_cover_cuts and len(cut_subgradients) > 0:
        #             for i, g in enumerate(cut_subgradients):
        #                 # scaled joint Polyak step for μ
        #                 delta = gamma_mu * alpha * g

        #                 # optional safety cap (keeps steps moderate)
        #                 if mu_increment_cap is not None:
        #                     if delta > mu_increment_cap:
        #                         delta = mu_increment_cap
        #                     elif delta < -mu_increment_cap:
        #                         delta = -mu_increment_cap

        #                 mu_old = self.best_cut_multipliers.get(i, 0.0)
        #                 mu_new = mu_old + delta

        #                 # project to μ_i >= 0
        #                 if mu_new < 0.0:
        #                     mu_new = 0.0

        #                 self.best_cut_multipliers[i] = mu_new

        #         # history bookkeeping
        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #         # λ stagnation check
        #         if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0
        #         last_g_lambda = knapsack_subgradient

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                 f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}, "
        #                 f"cuts={len(self.best_cuts)}")

        #     # ------------------------------------------------------------------
        #     # 6) Optional: drop "dead" cuts for future nodes
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts:
        #         keep_indices = []
        #         for i, (cut, rhs) in enumerate(self.best_cuts):
        #             mu_i = float(self.best_cut_multipliers.get(i, 0.0))
        #             # drop if never strictly violated HERE and μ stayed tiny
        #             if max_cut_violation[i] == 0.0 and abs(mu_i) < dead_mu_threshold:
        #                 continue
        #             keep_indices.append(i)

        #         if len(keep_indices) < len(self.best_cuts):
        #             new_best_cuts = []
        #             new_mu = {}
        #             new_mu_best = {}
        #             new_rhs_eff = {}
        #             for new_idx, old_idx in enumerate(keep_indices):
        #                 new_best_cuts.append(self.best_cuts[old_idx])
        #                 new_mu[new_idx] = float(self.best_cut_multipliers.get(old_idx, 0.0))
        #                 new_mu_best[new_idx] = float(self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0))
        #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]
        #             self.best_cuts = new_best_cuts
        #             self.best_cut_multipliers = new_mu
        #             self.best_cut_multipliers_for_best_bound = new_mu_best
        #             self._rhs_eff = new_rhs_eff

        #     # ------------------------------------------------------------------
        #     # 7) Update global flag if this node saw any violated cut
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and max_cut_violation:
        #         if any(v > 0.0 for v in max_cut_violation):
        #             LagrangianMST.global_cut_violation_seen = True

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts


        # else:  # Subgradient method with Polyak hybrid + pre- & in-loop separation saritar hal kard
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams (for λ)
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10

        #     # μ updates: conservative
        #     gamma_mu = getattr(self, "gamma_mu", 0.30)            # smoother than 1.0 (optional)
        #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
        #     eps = 1e-12

        #     # ------------------------------------------------------------------
        #     # Depth-based cutting + global fallback:
        #     # - Normally: cut only if depth <= max_cut_depth
        #     # - But: until ANY node sees a violated cut, allow cutting at all depths.
        #     # ------------------------------------------------------------------
        #     max_cut_depth = getattr(self, "max_cut_depth", 4)

        #     # Node-level separation controls
        #     node_cut_frequency = getattr(self, "node_cut_frequency", 10)  # in-loop separation period
        #     max_active_cuts = getattr(self, "max_active_cuts", 5)         # max cuts used in dual at this node
        #     max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)    # pre-separation cap
        #     max_inloop_new_cuts = getattr(self, "max_inloop_new_cuts", 2)        # in-loop cap
        #     min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
        #     dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-4)

        #     # Has *any* node so far in this solve seen a violated cut?
        #     global_violation_seen = getattr(LagrangianMST, "global_cut_violation_seen", False)

        #     # Cutting is active here if:
        #     #   - cover cuts are on, and
        #     #   - (depth is shallow OR we haven't seen any violated cuts yet)
        #     cutting_active_here = self.use_cover_cuts and (
        #         depth <= max_cut_depth or not global_violation_seen
        #     )

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Collect new cuts generated at THIS node (pre + in-loop) to return
        #     node_new_cuts = []

        #     # --- Quick guards ---
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        #     # --- Prepare fixed/excluded ---
        #     F_in = getattr(self, "fixed_edges", set())    # normalized tuples
        #     F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()
        #     edge_idx = self.edge_indices  # normalized edge -> index

        #     # Will store MST from pre-separation to reuse in iter 0
        #     pre_mst_available = False
        #     pre_mst_cost = None
        #     pre_mst_length = None
        #     pre_mst_edges = None

        #     # ------------------------------------------------------------------
        #     # 1) PRE-SEPARATION AT NODE START (cuts usable in this node)
        #     # ------------------------------------------------------------------
        #     if cutting_active_here and self.use_cover_cuts:
        #         try:
        #             # priced weights with current λ, μ (for inherited cuts only)
        #             w0 = self.compute_modified_weights()
        #             try:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst_incremental(w0, None)
        #             except Exception:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst()

        #             # store this MST to reuse as the first iteration
        #             pre_mst_available = True
        #             pre_mst_cost = mst_cost0
        #             pre_mst_length = mst_len0
        #             pre_mst_edges = mst_edges0

        #             cand_cuts = self.generate_cover_cuts(mst_edges0) or []

        #             # evaluate violation on this MST (using ORIGINAL rhs)
        #             T0 = set(mst_edges0)
        #             scored = []
        #             for cut, rhs in cand_cuts:
        #                 S = set(cut)
        #                 lhs = len(T0 & S)
        #                 violation = lhs - rhs
        #                 if violation >= min_cut_violation_for_add:
        #                     scored.append((violation, S, rhs))

        #             # strongest first
        #             scored.sort(reverse=True, key=lambda t: t[0])

        #             # respect caps: max_new_cuts_per_node AND max_active_cuts
        #             available_slots = max(0, max_active_cuts - len(self.best_cuts))
        #             if available_slots > 0:
        #                 scored = scored[:min(max_new_cuts_per_node, available_slots)]
        #             else:
        #                 scored = []

        #             existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #             for violation, S, rhs in scored:
        #                 fz = frozenset(S)
        #                 if fz in existing:
        #                     # same support: if stronger rhs, replace
        #                     old_rhs = existing[fz]
        #                     if rhs > old_rhs:
        #                         idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                         self.best_cuts[idx] = (set(S), rhs)
        #                         existing[fz] = rhs
        #                     continue

        #                 # truly new cut
        #                 self.best_cuts.append((set(S), rhs))
        #                 idx_new = len(self.best_cuts) - 1
        #                 self.best_cut_multipliers[idx_new] = 0.0
        #                 self.best_cut_multipliers_for_best_bound[idx_new] = 0.0
        #                 node_new_cuts.append((set(S), rhs))
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error in pre-separation at depth {depth}: {e}")

        #     # ------------------------------------------------------------------
        #     # 2) Compute rhs_eff and detect infeasibility (node-level)
        #     # ------------------------------------------------------------------
        #     self._rhs_eff = {}
        #     for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #         rhs_eff = int(rhs) - len(cut & F_in)
        #         self._rhs_eff[idx_c] = rhs_eff
        #         if rhs_eff < 0:
        #             # node infeasible due to fixed edges saturating the cut
        #             end_time = time()
        #             LagrangianMST.total_compute_time += end_time - start_time
        #             return float('inf'), self.best_upper_bound, node_new_cuts

        #     # ------------------------------------------------------------------
        #     # 3) Trim number of cuts at node start (keep important ones)
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
        #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
        #         if not parent_mu_map:
        #             parent_mu_map = self.best_cut_multipliers

        #         idx_and_cut = list(enumerate(self.best_cuts))
        #         # priority: large |μ|
        #         idx_and_cut.sort(
        #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
        #             reverse=True,
        #         )
        #         idx_and_cut = idx_and_cut[:max_active_cuts]

        #         new_cuts_list = []
        #         new_mu = {}
        #         new_mu_best = {}
        #         new_rhs_eff = {}
        #         remap = {}
        #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
        #             new_cuts_list.append(cut_rhs)
        #             new_mu[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_mu_best[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_rhs_eff[new_i] = self._rhs_eff[old_i]
        #             remap[old_i] = new_i

        #         self.best_cuts = new_cuts_list
        #         self.best_cut_multipliers = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu_best
        #         self._rhs_eff = new_rhs_eff

        #     # ------------------------------------------------------------------
        #     # 4) Precompute cut -> edge index arrays (FIXED for this node, but
        #     #    we will rebuild them if we add cuts in-loop)
        #     # ------------------------------------------------------------------
        #     def _rebuild_cut_structures():
        #         """Rebuild index arrays and rhs vectors from self.best_cuts & self._rhs_eff."""
        #         nonlocal cut_edge_idx_free, cut_free_sizes, cut_edge_idx_all, rhs_eff_vec, rhs_vec

        #         cut_edge_idx_free = []
        #         cut_free_sizes = []
        #         cut_edge_idx_all = []

        #         for cut, rhs in self.best_cuts:
        #             # FREE indices (for possible future refinements – not used in μ-gradient)
        #             idxs_free = [
        #                 edge_idx[e] for e in cut
        #                 if (e not in F_in and e not in F_out) and (e in edge_idx)
        #             ]
        #             arr_free = np.fromiter(idxs_free, dtype=np.int32) if idxs_free else np.empty(0, dtype=np.int32)
        #             cut_edge_idx_free.append(arr_free)
        #             cut_free_sizes.append(max(1, len(idxs_free)))  # avoid /0

        #             # ALL indices (for dual pricing & true μ-subgradient)
        #             idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #             arr_all = np.fromiter(idxs_all, dtype=np.int32) if idxs_all else np.empty(0, dtype=np.int32)
        #             cut_edge_idx_all.append(arr_all)

        #         # stash for compute_modified_weights (used for pricing)
        #         self._cut_edge_idx = cut_edge_idx_free
        #         self._cut_edge_idx_all = cut_edge_idx_all

        #         if self.best_cuts:
        #             rhs_eff_vec = np.array(
        #                 [self._rhs_eff[i] for i in range(len(cut_edge_idx_free))],
        #                 dtype=float
        #             )
        #             rhs_vec = np.array(
        #                 [rhs for (_, rhs) in self.best_cuts],
        #                 dtype=float
        #             )
        #         else:
        #             rhs_eff_vec = np.zeros(0, dtype=float)
        #             rhs_vec = np.zeros(0, dtype=float)

        #     # initialize structures
        #     cut_edge_idx_free = []
        #     cut_free_sizes = []
        #     cut_edge_idx_all = []
        #     rhs_eff_vec = np.zeros(0, dtype=float)
        #     rhs_vec = np.zeros(0, dtype=float)
        #     _rebuild_cut_structures()

        #     # track how "useful" each cut was at this node (only positive violation)
        #     max_cut_violation = [0.0 for _ in self.best_cuts]

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     # Seed priced weights so iteration 0 is consistent
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None

        #     last_g_lambda = None  # for stagnation check

        #     # ------------------------------------------------------------------
        #     # 5) Subgradient iterations (dual structure mostly fixed, but we may
        #     #    add a few extra cuts in-loop with full rebuild).
        #     # ------------------------------------------------------------------
        #     for iter_num in range(int(max_iter)):
        #         # 1) Solve MST on current priced weights
        #         if pre_mst_available:
        #             # Reuse the MST computed during pre-separation (same λ, μ)
        #             mst_cost = pre_mst_cost
        #             mst_length = pre_mst_length
        #             mst_edges = pre_mst_edges
        #             pre_mst_available = False  # use only once
        #         else:
        #             try:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #             except Exception:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()

        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges = self.last_mst_edges

        #         # ------------------------------------------------------------------
        #         # 5a) OCCASIONAL IN-LOOP SEPARATION (lightweight, optional)
        #         # ------------------------------------------------------------------
        #         if (
        #             cutting_active_here
        #             and self.use_cover_cuts
        #             and (iter_num % node_cut_frequency == 0)
        #             and len(self.best_cuts) < max_active_cuts
        #         ):
        #             try:
        #                 cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []

        #                 T_loop = set(mst_edges)
        #                 scored_loop = []
        #                 for cut, rhs in cand_cuts_loop:
        #                     S = set(cut)
        #                     lhs = len(T_loop & S)
        #                     violation = lhs - rhs
        #                     if violation >= min_cut_violation_for_add:
        #                         scored_loop.append((violation, S, rhs))

        #                 scored_loop.sort(reverse=True, key=lambda t: t[0])

        #                 remaining_slots = max(0, max_active_cuts - len(self.best_cuts))
        #                 if remaining_slots > 0:
        #                     scored_loop = scored_loop[:min(max_inloop_new_cuts, remaining_slots)]
        #                 else:
        #                     scored_loop = []

        #                 existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #                 added_any = False
        #                 for violation, S, rhs in scored_loop:
        #                     fz = frozenset(S)
        #                     if fz in existing:
        #                         # if we ever wanted to strengthen rhs here we could, but usually in-loop we skip
        #                         continue

        #                     # truly new cut
        #                     self.best_cuts.append((set(S), rhs))
        #                     new_idx = len(self.best_cuts) - 1
        #                     self.best_cut_multipliers[new_idx] = 0.0
        #                     self.best_cut_multipliers_for_best_bound[new_idx] = 0.0
        #                     self._rhs_eff[new_idx] = int(rhs) - len(set(S) & F_in)
        #                     max_cut_violation.append(0.0)
        #                     node_new_cuts.append((set(S), rhs))
        #                     added_any = True

        #                 if added_any:
        #                     # Rebuild index structures & rhs vectors to incorporate new cuts
        #                     _rebuild_cut_structures()
        #                     # Reset modified weights cache with new μ dimension
        #                     self._mw_cached = None
        #                     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error in in-loop separation at depth {depth}, iter {iter_num}: {e}")

        #         # Prepare weights for next iteration (using current λ, μ and possibly updated cuts)
        #         prev_weights = self.compute_modified_weights()

        #         # 2) Dual & primal bookkeeping
        #         is_feasible = (mst_length <= self.budget)

        #         # (a) primal & UB
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight)
        #                         and not math.isinf(real_weight)
        #                         and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # prune primal_solutions history (cheaper: just keep last MAX_SOLUTIONS)
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

        #         # (b) Lagrangian dual value (dualized cuts use ORIGINAL rhs)
        #         if self.use_cover_cuts and len(rhs_vec) > 0:
        #             mu_vec = np.fromiter(
        #                 (self.best_cut_multipliers.get(i, 0.0) for i in range(len(rhs_vec))),
        #                 dtype=float,
        #                 count=len(rhs_vec),
        #             )
        #             cover_cut_penalty = float(mu_vec @ rhs_vec)
        #         else:
        #             mu_vec = np.zeros(len(rhs_vec), dtype=float)
        #             cover_cut_penalty = 0.0

        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound)
        #                 and not math.isinf(lagrangian_bound)
        #                 and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #                 if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #         else:
        #             no_improvement_count += 1

        #         # 3) Subgradients
        #         # knapsack
        #         knapsack_subgradient = float(mst_length - self.budget)

        #         # cuts: TRUE dual subgradient for the encoded Lagrangian:
        #         # g_i = lhs_i - rhs_i, lhs_i counts ALL edges in S_i selected by MST

        #         # Reuse a single mask array instead of allocating every iteration
        #         if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
        #             self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #         mst_mask = self._mst_mask
        #         mst_mask[:] = False
        #         for e in mst_edges:
        #             j = self.edge_indices.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         if self.use_cover_cuts and len(cut_edge_idx_all) > 0:
        #             violations = []
        #             for i, idxs_all in enumerate(cut_edge_idx_all):
        #                 lhs_i = int(mst_mask[idxs_all].sum()) if idxs_all.size else 0
        #                 g_i = lhs_i - rhs_vec[i]  # <-- consistent with lagrangian_bound
        #                 violations.append(g_i)
        #                 # only *positive* violation counts as "usefulness"
        #                 if g_i > max_cut_violation[i]:
        #                     max_cut_violation[i] = g_i
        #             violations = np.array(violations, dtype=float)
        #         else:
        #             violations = np.zeros(0, dtype=float)

        #         cut_subgradients = violations.tolist()

        #         # 4) Step sizes & updates (joint Polyak for λ and μ)
        #         self.subgradients.append(knapsack_subgradient)

        #         # --- joint gradient norm ---
        #         norm_sq = knapsack_subgradient ** 2
        #         for g in cut_subgradients:
        #             norm_sq += g ** 2

        #         # --- Polyak-like step size for the joint dual (λ, μ) ---
        #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #             gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             theta = gamma_base
        #             alpha = theta * gap / (norm_sq + eps)
        #         else:
        #             # fallback small constant if UB is infinite or gradient is zero
        #             alpha = getattr(self, "step_size", 1e-5)

        #         # --- λ update with momentum, using the joint α ---
        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # --- μ updates (only if cuts are active), using the SAME α (scaled) ---
        #         if self.use_cover_cuts and len(cut_subgradients) > 0:
        #             for i, g in enumerate(cut_subgradients):
        #                 # scaled joint Polyak step for μ
        #                 delta = gamma_mu * alpha * g

        #                 # optional safety cap (keeps steps moderate)
        #                 if mu_increment_cap is not None:
        #                     if delta > mu_increment_cap:
        #                         delta = mu_increment_cap
        #                     elif delta < -mu_increment_cap:
        #                         delta = -mu_increment_cap

        #                 mu_old = self.best_cut_multipliers.get(i, 0.0)
        #                 mu_new = mu_old + delta

        #                 # project to μ_i >= 0
        #                 if mu_new < 0.0:
        #                     mu_new = 0.0

        #                 self.best_cut_multipliers[i] = mu_new

        #         # history bookkeeping
        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #         # λ stagnation check
        #         if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0
        #         last_g_lambda = knapsack_subgradient

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                 f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}, "
        #                 f"cuts={len(self.best_cuts)}")

        #     # ------------------------------------------------------------------
        #     # 6) Optional: drop "dead" cuts for future nodes
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts:
        #         keep_indices = []
        #         for i, (cut, rhs) in enumerate(self.best_cuts):
        #             mu_i = float(self.best_cut_multipliers.get(i, 0.0))
        #             # drop if never strictly violated HERE and μ stayed tiny
        #             if max_cut_violation[i] == 0.0 and abs(mu_i) < dead_mu_threshold:
        #                 continue
        #             keep_indices.append(i)

        #         if len(keep_indices) < len(self.best_cuts):
        #             new_best_cuts = []
        #             new_mu = {}
        #             new_mu_best = {}
        #             new_rhs_eff = {}
        #             for new_idx, old_idx in enumerate(keep_indices):
        #                 new_best_cuts.append(self.best_cuts[old_idx])
        #                 new_mu[new_idx] = float(self.best_cut_multipliers.get(old_idx, 0.0))
        #                 new_mu_best[new_idx] = float(self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0))
        #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]
        #             self.best_cuts = new_best_cuts
        #             self.best_cut_multipliers = new_mu
        #             self.best_cut_multipliers_for_best_bound = new_mu_best
        #             self._rhs_eff = new_rhs_eff

        #     # ------------------------------------------------------------------
        #     # 7) Update global flag if this node saw any violated cut
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and max_cut_violation:
        #         if any(v > 0.0 for v in max_cut_violation):
        #             LagrangianMST.global_cut_violation_seen = True

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts



       
        # else:  # Subgradient method with Polyak hybrid + pre- & in-loop separation (with depth-based freezing, no new constants) hamoon k run gereftam rid
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 50
        #     CLEANUP_INTERVAL = 100
        #     max_iter = min(self.max_iter, 200)  # unchanged

        #     # Polyak / momentum hyperparams (for λ)
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.1

        #     # μ updates: conservative
        #     gamma_mu = getattr(self, "gamma_mu", 0.30)            # unchanged
        #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
        #     eps = 1e-12

        #     # ------------------------------------------------------------------
        #     # Depth-based cutting + global fallback:
        #     # - Normally: cut only if depth <= max_cut_depth
        #     # - But: until ANY node sees a violated cut, allow cutting at all depths.
        #     # ------------------------------------------------------------------
        #     max_cut_depth = getattr(self, "max_cut_depth", 4)

        #     # Has *any* node so far in this solve seen a violated cut?
        #     global_violation_seen = getattr(LagrangianMST, "global_cut_violation_seen", False)

        #     # Cutting is active here if:
        #     #   - cover cuts are on, and
        #     #   - (depth is shallow OR we haven't seen any violated cuts yet)
        #     cutting_active_here = self.use_cover_cuts and (
        #         depth <= max_cut_depth or not global_violation_seen
        #     )

        #     # Ensure cut data structures exist
        #     if not hasattr(self, "best_cuts"):
        #         self.best_cuts = []
        #     if not hasattr(self, "best_cut_multipliers"):
        #         self.best_cut_multipliers = {}
        #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers_for_best_bound = {}

        #     # Within this node:
        #     # - dynamic: we may generate cuts and update μ
        #     # - present: we have cuts in the dual (for pricing), even if frozen
        #     cuts_dynamic_here = self.use_cover_cuts and cutting_active_here
        #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Collect new cuts generated at THIS node (pre + in-loop) to return
        #     node_new_cuts = []

        #     # --- Quick guards ---
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        #     # --- Prepare fixed/excluded ---
        #     F_in = getattr(self, "fixed_edges", set())    # normalized tuples
        #     F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()
        #     edge_idx = self.edge_indices  # normalized edge -> index

        #     # Will store MST from pre-separation to reuse in iter 0
        #     pre_mst_available = False
        #     pre_mst_cost = None
        #     pre_mst_length = None
        #     pre_mst_edges = None

        #     # ------------------------------------------------------------------
        #     # 1) PRE-SEPARATION AT NODE START (cuts usable in this node)
        #     # ------------------------------------------------------------------
        #     if cuts_dynamic_here and self.use_cover_cuts:
        #         try:
        #             # priced weights with current λ, μ (for inherited cuts only)
        #             w0 = self.compute_modified_weights()
        #             try:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst_incremental(w0, None)
        #             except Exception:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst()

        #             # store this MST to reuse as the first iteration
        #             pre_mst_available = True
        #             pre_mst_cost = mst_cost0
        #             pre_mst_length = mst_len0
        #             pre_mst_edges = mst_edges0

        #             cand_cuts = self.generate_cover_cuts(mst_edges0) or []

        #             # evaluate violation on this MST (using ORIGINAL rhs)
        #             T0 = set(mst_edges0)
        #             scored = []
        #             min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
        #             for cut, rhs in cand_cuts:
        #                 S = set(cut)
        #                 lhs = len(T0 & S)
        #                 violation = lhs - rhs
        #                 if violation >= min_cut_violation_for_add:
        #                     scored.append((violation, S, rhs))

        #             # strongest first
        #             scored.sort(reverse=True, key=lambda t: t[0])

        #             # respect caps: max_new_cuts_per_node AND max_active_cuts
        #             max_active_cuts = getattr(self, "max_active_cuts", 20)
        #             max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 3)
        #             available_slots = max(0, max_active_cuts - len(self.best_cuts))
        #             if available_slots > 0:
        #                 scored = scored[:min(max_new_cuts_per_node, available_slots)]
        #             else:
        #                 scored = []

        #             existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #             for violation, S, rhs in scored:
        #                 fz = frozenset(S)
        #                 if fz in existing:
        #                     # same support: if stronger rhs, replace
        #                     old_rhs = existing[fz]
        #                     if rhs > old_rhs:
        #                         idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                         self.best_cuts[idx] = (set(S), rhs)
        #                         existing[fz] = rhs
        #                     continue

        #                 # truly new cut
        #                 self.best_cuts.append((set(S), rhs))
        #                 idx_new = len(self.best_cuts) - 1
        #                 self.best_cut_multipliers[idx_new] = 0.0
        #                 self.best_cut_multipliers_for_best_bound[idx_new] = 0.0
        #                 node_new_cuts.append((set(S), rhs))

        #             # after pre-separation, we now definitely have cuts at this node
        #             cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error in pre-separation at depth {depth}: {e}")

        #     # ------------------------------------------------------------------
        #     # 2) Compute rhs_eff and detect infeasibility (node-level)
        #     # ------------------------------------------------------------------
        #     self._rhs_eff = {}
        #     if self.use_cover_cuts and self.best_cuts:
        #         for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #             rhs_eff = int(rhs) - len(cut & F_in)
        #             self._rhs_eff[idx_c] = rhs_eff
        #             if rhs_eff < 0:
        #                 # node infeasible due to fixed edges saturating the cut
        #                 end_time = time()
        #                 LagrangianMST.total_compute_time += end_time - start_time
        #                 return float('inf'), self.best_upper_bound, node_new_cuts

        #     # ------------------------------------------------------------------
        #     # 3) Trim number of cuts at node start (keep important ones)
        #     # ------------------------------------------------------------------
        #     max_active_cuts = getattr(self, "max_active_cuts", 3)
        #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
        #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
        #         if not parent_mu_map:
        #             parent_mu_map = self.best_cut_multipliers

        #         idx_and_cut = list(enumerate(self.best_cuts))
        #         # priority: large |μ|
        #         idx_and_cut.sort(
        #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
        #             reverse=True,
        #         )
        #         idx_and_cut = idx_and_cut[:max_active_cuts]

        #         new_cuts_list = []
        #         new_mu = {}
        #         new_mu_best = {}
        #         new_rhs_eff = {}
        #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
        #             new_cuts_list.append(cut_rhs)
        #             new_mu[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_mu_best[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_rhs_eff[new_i] = self._rhs_eff[old_i]

        #         self.best_cuts = new_cuts_list
        #         self.best_cut_multipliers = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu_best
        #         self._rhs_eff = new_rhs_eff

        #     # Re-evaluate presence after trimming
        #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #     # ------------------------------------------------------------------
        #     # 4) Precompute cut -> edge index arrays (FIXED for this node, but
        #     #    we will rebuild them if we add cuts in-loop)
        #     # ------------------------------------------------------------------
        #     def _rebuild_cut_structures():
        #         """Rebuild index arrays and rhs vectors from self.best_cuts & self._rhs_eff."""
        #         nonlocal cut_edge_idx_free, cut_free_sizes, cut_edge_idx_all, rhs_eff_vec, rhs_vec

        #         cut_edge_idx_free = []
        #         cut_free_sizes = []
        #         cut_edge_idx_all = []

        #         for cut, rhs in self.best_cuts:
        #             # FREE indices (for possible future refinements – not used in μ-gradient)
        #             idxs_free = [
        #                 edge_idx[e] for e in cut
        #                 if (e not in F_in and e not in F_out) and (e in edge_idx)
        #             ]
        #             arr_free = np.fromiter(idxs_free, dtype=np.int32) if idxs_free else np.empty(0, dtype=np.int32)
        #             cut_edge_idx_free.append(arr_free)
        #             cut_free_sizes.append(max(1, len(idxs_free)))  # avoid /0

        #             # ALL indices (for dual pricing & true μ-subgradient)
        #             idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #             arr_all = np.fromiter(idxs_all, dtype=np.int32) if idxs_all else np.empty(0, dtype=np.int32)
        #             cut_edge_idx_all.append(arr_all)

        #         # stash for compute_modified_weights (used for pricing)
        #         self._cut_edge_idx = cut_edge_idx_free
        #         self._cut_edge_idx_all = cut_edge_idx_all

        #         if self.best_cuts:
        #             rhs_eff_vec = np.array(
        #                 [self._rhs_eff[i] for i in range(len(cut_edge_idx_free))],
        #                 dtype=float
        #             )
        #             rhs_vec = np.array(
        #                 [rhs for (_, rhs) in self.best_cuts],
        #                 dtype=float
        #             )
        #         else:
        #             rhs_eff_vec = np.zeros(0, dtype=float)
        #             rhs_vec = np.zeros(0, dtype=float)

        #     # initialize structures
        #     cut_edge_idx_free = []
        #     cut_free_sizes = []
        #     cut_edge_idx_all = []
        #     rhs_eff_vec = np.zeros(0, dtype=float)
        #     rhs_vec = np.zeros(0, dtype=float)
        #     _rebuild_cut_structures()

        #     # track how "useful" each cut was at this node (only positive violation)
        #     max_cut_violation = [0.0 for _ in self.best_cuts]

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     # Seed priced weights so iteration 0 is consistent
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None

        #     last_g_lambda = None  # for stagnation check

        #     node_cut_frequency = getattr(self, "node_cut_frequency", 100)
        #     max_inloop_new_cuts = getattr(self, "max_inloop_new_cuts", 2)
        #     dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-4)

        #     # ------------------------------------------------------------------
        #     # 5) Subgradient iterations (dual structure mostly fixed, but we may
        #     #    add a few extra cuts in-loop with full rebuild).
        #     # ------------------------------------------------------------------
        #     for iter_num in range(int(max_iter)):
        #         # 1) Solve MST on current priced weights
        #         if pre_mst_available:
        #             # Reuse the MST computed during pre-separation (same λ, μ)
        #             mst_cost = pre_mst_cost
        #             mst_length = pre_mst_length
        #             mst_edges = pre_mst_edges
        #             pre_mst_available = False  # use only once
        #         else:
        #             try:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #             except Exception:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()

        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges = self.last_mst_edges

        #         # ------------------------------------------------------------------
        #         # 5a) OCCASIONAL IN-LOOP SEPARATION (lightweight, optional)
        #         #     Only on dynamic nodes (no separation when frozen).
        #         # ------------------------------------------------------------------
        #         if (
        #             cuts_dynamic_here
        #             and self.use_cover_cuts
        #             and (iter_num % node_cut_frequency == 0)
        #             and len(self.best_cuts) < max_active_cuts
        #         ):
        #             try:
        #                 cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []

        #                 T_loop = set(mst_edges)
        #                 scored_loop = []
        #                 for cut, rhs in cand_cuts_loop:
        #                     S = set(cut)
        #                     lhs = len(T_loop & S)
        #                     violation = lhs - rhs
        #                     if violation >= min_cut_violation_for_add:
        #                         scored_loop.append((violation, S, rhs))

        #                 scored_loop.sort(reverse=True, key=lambda t: t[0])

        #                 remaining_slots = max(0, max_active_cuts - len(self.best_cuts))
        #                 if remaining_slots > 0:
        #                     scored_loop = scored_loop[:min(max_inloop_new_cuts, remaining_slots)]
        #                 else:
        #                     scored_loop = []

        #                 existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #                 added_any = False
        #                 for violation, S, rhs in scored_loop:
        #                     fz = frozenset(S)
        #                     if fz in existing:
        #                         # if we ever wanted to strengthen rhs here we could, but usually in-loop we skip
        #                         continue

        #                     # truly new cut
        #                     self.best_cuts.append((set(S), rhs))
        #                     new_idx = len(self.best_cuts) - 1
        #                     self.best_cut_multipliers[new_idx] = 0.0
        #                     self.best_cut_multipliers_for_best_bound[new_idx] = 0.0
        #                     self._rhs_eff[new_idx] = int(rhs) - len(set(S) & F_in)
        #                     max_cut_violation.append(0.0)
        #                     node_new_cuts.append((set(S), rhs))
        #                     added_any = True

        #                 if added_any:
        #                     # Rebuild index structures & rhs vectors to incorporate new cuts
        #                     _rebuild_cut_structures()
        #                     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)
        #                     # Reset modified weights cache with new μ dimension
        #                     self._mw_cached = None
        #                     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error in in-loop separation at depth {depth}, iter {iter_num}: {e}")

        #         # Prepare weights for next iteration (using current λ, μ and possibly updated cuts)
        #         prev_weights = self.compute_modified_weights()

        #         # 2) Dual & primal bookkeeping
        #         is_feasible = (mst_length <= self.budget)

        #         # (a) primal & UB
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight)
        #                         and not math.isinf(real_weight)
        #                         and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # prune primal_solutions history (cheaper: just keep last MAX_SOLUTIONS)
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

        #         # (b) Lagrangian dual value (dualized cuts use ORIGINAL rhs)
        #         if self.use_cover_cuts and len(rhs_vec) > 0:
        #             mu_vec = np.fromiter(
        #                 (self.best_cut_multipliers.get(i, 0.0) for i in range(len(rhs_vec))),
        #                 dtype=float,
        #                 count=len(rhs_vec),
        #             )
        #             cover_cut_penalty = float(mu_vec @ rhs_vec)
        #         else:
        #             mu_vec = np.zeros(len(rhs_vec), dtype=float)
        #             cover_cut_penalty = 0.0

        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound)
        #                 and not math.isinf(lagrangian_bound)
        #                 and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #                 if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #         else:
        #             no_improvement_count += 1

        #         # 3) Subgradients
        #         # knapsack
        #         knapsack_subgradient = float(mst_length - self.budget)

        #         # cuts: TRUE dual subgradient for the encoded Lagrangian:
        #         # g_i = lhs_i - rhs_i, lhs_i counts ALL edges in S_i selected by MST
        #         if cuts_present_here and cuts_dynamic_here and len(cut_edge_idx_all) > 0:
        #             # Reuse a single mask array instead of allocating every iteration
        #             if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
        #                 self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #             mst_mask = self._mst_mask
        #             mst_mask[:] = False
        #             for e in mst_edges:
        #                 j = self.edge_indices.get(e)
        #                 if j is not None:
        #                     mst_mask[j] = True

        #             violations = []
        #             for i, idxs_all in enumerate(cut_edge_idx_all):
        #                 lhs_i = int(mst_mask[idxs_all].sum()) if idxs_all.size else 0
        #                 g_i = lhs_i - rhs_vec[i]  # <-- consistent with lagrangian_bound
        #                 violations.append(g_i)
        #                 # only *positive* violation counts as "usefulness"
        #                 if g_i > max_cut_violation[i]:
        #                     max_cut_violation[i] = g_i
        #             violations = np.array(violations, dtype=float)
        #             cut_subgradients = violations.tolist()
        #         else:
        #             # either no cuts, or we are in "frozen cuts" mode on deep node
        #             violations = np.zeros(0, dtype=float)
        #             cut_subgradients = []

        #         # 4) Step sizes & updates (joint Polyak for λ and μ)
        #         self.subgradients.append(knapsack_subgradient)

        #         # --- joint gradient norm ---
        #         norm_sq = knapsack_subgradient ** 2
        #         for g in cut_subgradients:
        #             norm_sq += g ** 2

        #         # --- Polyak-like step size for the joint dual (λ, μ) ---
        #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #             gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             theta = gamma_base
        #             alpha = theta * gap / (norm_sq + eps)
        #         else:
        #             # fallback small constant if UB is infinite or gradient is zero
        #             alpha = getattr(self, "step_size", 1e-5)

        #         # --- λ update with momentum, using the joint α ---
        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # --- μ updates (only if cuts are dynamic here), using the SAME α (scaled) ---
        #         if cuts_dynamic_here and len(cut_subgradients) > 0:
        #             for i, g in enumerate(cut_subgradients):
        #                 # scaled joint Polyak step for μ
        #                 delta = gamma_mu * alpha * g

        #                 # optional safety cap (keeps steps moderate)
        #                 if mu_increment_cap is not None:
        #                     if delta > mu_increment_cap:
        #                         delta = mu_increment_cap
        #                     elif delta < -mu_increment_cap:
        #                         delta = -mu_increment_cap

        #                 mu_old = self.best_cut_multipliers.get(i, 0.0)
        #                 mu_new = mu_old + delta

        #                 # project to μ_i >= 0
        #                 if mu_new < 0.0:
        #                     mu_new = 0.0

        #                 self.best_cut_multipliers[i] = mu_new

        #         # history bookkeeping
        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #         # λ stagnation check
        #         if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             current_step = getattr(self, "step_size", 1e-5)
        #             self.step_size = max(1e-8, current_step * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0
        #         last_g_lambda = knapsack_subgradient

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                 f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}, "
        #                 f"cuts={len(self.best_cuts)}")

        #     # ------------------------------------------------------------------
        #     # 6) Optional: drop "dead" cuts for future nodes
        #     #     Only where we actually tracked violations (dynamic nodes).
        #     # ------------------------------------------------------------------
        #     if cuts_dynamic_here and self.use_cover_cuts and self.best_cuts:
        #         keep_indices = []
        #         for i, (cut, rhs) in enumerate(self.best_cuts):
        #             mu_i = float(self.best_cut_multipliers.get(i, 0.0))
        #             # drop if never strictly violated HERE and μ stayed tiny
        #             if max_cut_violation[i] == 0.0 and abs(mu_i) < dead_mu_threshold:
        #                 continue
        #             keep_indices.append(i)

        #         if len(keep_indices) < len(self.best_cuts):
        #             new_best_cuts = []
        #             new_mu = {}
        #             new_mu_best = {}
        #             new_rhs_eff = {}
        #             for new_idx, old_idx in enumerate(keep_indices):
        #                 new_best_cuts.append(self.best_cuts[old_idx])
        #                 new_mu[new_idx] = float(self.best_cut_multipliers.get(old_idx, 0.0))
        #                 new_mu_best[new_idx] = float(self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0))
        #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]
        #             self.best_cuts = new_best_cuts
        #             self.best_cut_multipliers = new_mu
        #             self.best_cut_multipliers_for_best_bound = new_mu_best
        #             self._rhs_eff = new_rhs_eff

        #     # ------------------------------------------------------------------
        #     # 7) Update global flag if this node saw any violated cut
        #     #     (only meaningful on dynamic nodes)
        #     # ------------------------------------------------------------------
        #     if cuts_dynamic_here and self.use_cover_cuts and max_cut_violation:
        #         if any(v > 0.0 for v in max_cut_violation):
        #             LagrangianMST.global_cut_violation_seen = True

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts
        # else:  # Subgradient method with Polyak hybrid + pre- & in-loop separation (with depth-based freezing, no new constants) 
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 50
        #     CLEANUP_INTERVAL = 100
        #     max_iter = min(self.max_iter, 200)  # unchanged

        #     # Polyak / momentum hyperparams (for λ)
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.1

        #     # μ updates: conservative
        #     gamma_mu = getattr(self, "gamma_mu", 0.30)            # unchanged
        #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
        #     eps = 1e-12

        #     # ------------------------------------------------------------------
        #     # Depth-based cutting + global fallback:
        #     # - Normally: cut only if depth <= max_cut_depth
        #     # - But: until ANY node sees a violated cut, allow cutting at all depths.
        #     # ------------------------------------------------------------------
        #     max_cut_depth = getattr(self, "max_cut_depth", 1)
            

        #     # Has *any* node so far in this solve seen a violated cut?
        #     # global_violation_seen = getattr(LagrangianMST, "global_cut_violation_seen", False)

        #     # Cutting is active here if:
        #     #   - cover cuts are on, and
        #     #   - (depth is shallow OR we haven't seen any violated cuts yet)
        #     cutting_active_here = self.use_cover_cuts and (
        #         depth <= max_cut_depth
        #         #   or not global_violation_seen
        #     )

        #     # Ensure cut data structures exist
        #     if not hasattr(self, "best_cuts"):
        #         self.best_cuts = []
        #     if not hasattr(self, "best_cut_multipliers"):
        #         self.best_cut_multipliers = {}
        #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers_for_best_bound = {}

        #     # Within this node:
        #     # - dynamic: we may generate cuts and update μ
        #     # - present: we have cuts in the dual (for pricing), even if frozen
        #     cuts_dynamic_here = self.use_cover_cuts and cutting_active_here
        #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Collect new cuts generated at THIS node (pre + in-loop) to return
        #     node_new_cuts = []

        #     # --- Quick guards ---
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        #     # --- Prepare fixed/excluded ---
        #     F_in = getattr(self, "fixed_edges", set())    # normalized tuples
        #     F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()
        #     edge_idx = self.edge_indices  # normalized edge -> index

        #     # Will store MST from pre-separation to reuse in iter 0
        #     pre_mst_available = False
        #     pre_mst_cost = None
        #     pre_mst_length = None
        #     pre_mst_edges = None

        #     # ------------------------------------------------------------------
        #     # 1) PRE-SEPARATION AT NODE START (cuts usable in this node)
        #     # ------------------------------------------------------------------
        #     if cuts_dynamic_here and self.use_cover_cuts:
        #         try:
        #             # priced weights with current λ, μ (for inherited cuts only)
        #             w0 = self.compute_modified_weights()
        #             try:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst_incremental(w0, None)
                        
        #             except Exception:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst()

        #             # store this MST to reuse as the first iteration
        #             pre_mst_available = True
        #             pre_mst_cost = mst_cost0
        #             pre_mst_length = mst_len0
        #             pre_mst_edges = mst_edges0

        #             cand_cuts = self.generate_cover_cuts(mst_edges0) or []

        #             # evaluate violation on this MST (using ORIGINAL rhs)
        #             T0 = set(mst_edges0)
        #             scored = []
        #             min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
        #             for cut, rhs in cand_cuts:
        #                 S = set(cut)
        #                 lhs = len(T0 & S)
        #                 violation = lhs - rhs
        #                 if violation >= min_cut_violation_for_add:
        #                     scored.append((violation, S, rhs))

        #             # strongest first
        #             scored.sort(reverse=True, key=lambda t: t[0])

        #             # respect caps: max_new_cuts_per_node AND max_active_cuts
        #             max_active_cuts = getattr(self, "max_active_cuts", 5)
        #             max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)
        #             available_slots = max(0, max_active_cuts - len(self.best_cuts))
        #             if available_slots > 0:
        #                 scored = scored[:min(max_new_cuts_per_node, available_slots)]
        #             else:
        #                 scored = []

        #             existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #             for violation, S, rhs in scored:
        #                 fz = frozenset(S)
        #                 if fz in existing:
        #                     # same support: if stronger rhs, replace
        #                     old_rhs = existing[fz]
        #                     if rhs > old_rhs:
        #                         idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                         self.best_cuts[idx] = (set(S), rhs)
        #                         existing[fz] = rhs
        #                     continue

        #                 # truly new cut
        #                 self.best_cuts.append((set(S), rhs))
        #                 idx_new = len(self.best_cuts) - 1
        #                 self.best_cut_multipliers[idx_new] = 0.0
        #                 self.best_cut_multipliers_for_best_bound[idx_new] = 0.0
        #                 node_new_cuts.append((set(S), rhs))

        #             # after pre-separation, we now definitely have cuts at this node
        #             cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error in pre-separation at depth {depth}: {e}")

        #     # ------------------------------------------------------------------
        #     # 2) Compute rhs_eff and detect infeasibility (node-level)
        #     # ------------------------------------------------------------------
        #     self._rhs_eff = {}
        #     if self.use_cover_cuts and self.best_cuts:
        #         for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #             rhs_eff = int(rhs) - len(cut & F_in)
        #             self._rhs_eff[idx_c] = rhs_eff
        #             if rhs_eff < 0:
        #                 # node infeasible due to fixed edges saturating the cut
        #                 end_time = time()
        #                 LagrangianMST.total_compute_time += end_time - start_time
        #                 return float('inf'), self.best_upper_bound, node_new_cuts

        #     # ------------------------------------------------------------------
        #     # 3) Trim number of cuts at node start (keep important ones)
        #     # ------------------------------------------------------------------
        #     max_active_cuts = getattr(self, "max_active_cuts", 5)
        #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
        #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
        #         if not parent_mu_map:
        #             parent_mu_map = self.best_cut_multipliers

        #         idx_and_cut = list(enumerate(self.best_cuts))
        #         # priority: large |μ|
        #         idx_and_cut.sort(
        #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
        #             reverse=True,
        #         )
        #         idx_and_cut = idx_and_cut[:max_active_cuts]

        #         new_cuts_list = []
        #         new_mu = {}
        #         new_mu_best = {}
        #         new_rhs_eff = {}
        #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
        #             new_cuts_list.append(cut_rhs)
        #             new_mu[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_mu_best[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_rhs_eff[new_i] = self._rhs_eff[old_i]

        #         self.best_cuts = new_cuts_list
        #         self.best_cut_multipliers = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu_best
        #         self._rhs_eff = new_rhs_eff

        #     # Re-evaluate presence after trimming
        #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #     # ------------------------------------------------------------------
        #     # 4) Precompute cut -> edge index arrays (FIXED for this node, but
        #     #    we will rebuild them if we add cuts in-loop)
        #     # ------------------------------------------------------------------
        #     def _rebuild_cut_structures():
        #         """Rebuild index arrays and rhs vectors from self.best_cuts & self._rhs_eff."""
        #         nonlocal cut_edge_idx_free, cut_free_sizes, cut_edge_idx_all, rhs_eff_vec, rhs_vec

        #         cut_edge_idx_free = []
        #         cut_free_sizes = []
        #         cut_edge_idx_all = []

        #         for cut, rhs in self.best_cuts:
        #             # FREE indices (for possible future refinements – not used in μ-gradient)
        #             idxs_free = [
        #                 edge_idx[e] for e in cut
        #                 if (e not in F_in and e not in F_out) and (e in edge_idx)
        #             ]
        #             arr_free = np.fromiter(idxs_free, dtype=np.int32) if idxs_free else np.empty(0, dtype=np.int32)
        #             cut_edge_idx_free.append(arr_free)
        #             cut_free_sizes.append(max(1, len(idxs_free)))  # avoid /0

        #             # ALL indices (for dual pricing & true μ-subgradient)
        #             idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #             arr_all = np.fromiter(idxs_all, dtype=np.int32) if idxs_all else np.empty(0, dtype=np.int32)
        #             cut_edge_idx_all.append(arr_all)

        #         # stash for compute_modified_weights (used for pricing)
        #         self._cut_edge_idx = cut_edge_idx_free
        #         self._cut_edge_idx_all = cut_edge_idx_all

        #         if self.best_cuts:
        #             rhs_eff_vec = np.array(
        #                 [self._rhs_eff[i] for i in range(len(cut_edge_idx_free))],
        #                 dtype=float
        #             )
        #             rhs_vec = np.array(
        #                 [rhs for (_, rhs) in self.best_cuts],
        #                 dtype=float
        #             )
        #         else:
        #             rhs_eff_vec = np.zeros(0, dtype=float)
        #             rhs_vec = np.zeros(0, dtype=float)

        #     # initialize structures
        #     cut_edge_idx_free = []
        #     cut_free_sizes = []
        #     cut_edge_idx_all = []
        #     rhs_eff_vec = np.zeros(0, dtype=float)
        #     rhs_vec = np.zeros(0, dtype=float)
        #     _rebuild_cut_structures()

        #     # track how "useful" each cut was at this node (only positive violation)
        #     max_cut_violation = [0.0 for _ in self.best_cuts]

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     # Seed priced weights so iteration 0 is consistent
        #     prev_weights = None
        #     prev_mst_edges = None

        #     last_g_lambda = None  # for stagnation check

        #     dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-6)

        #     # ------------------------------------------------------------------
        #     # 5) Subgradient iterations (dual structure mostly fixed, but we may
        #     #    add a few extra cuts in-loop with full rebuild).
        #     # ------------------------------------------------------------------
        #     for iter_num in range(int(max_iter)):
        #         # 1) Solve MST on current priced weights
        #         if pre_mst_available:
        #             # Reuse the MST computed during pre-separation (same λ, μ)
        #             mst_cost = pre_mst_cost
        #             mst_length = pre_mst_length
        #             mst_edges = pre_mst_edges
        #             pre_mst_available = False  # use only once
        #         else:
        #             if prev_weights is None:
        #                 prev_weights = self.compute_modified_weights()
        #             try:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #             except Exception:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()

        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges = self.last_mst_edges

        #         # ------------------------------------------------------------------
        #         # 5a) OCCASIONAL IN-LOOP SEPARATION (lightweight, optional)
        #         #     Only on dynamic nodes (no separation when frozen).
        #         # ------------------------------------------------------------------
        #         # if (
        #         #     cuts_dynamic_here
        #         #     and self.use_cover_cuts
        #         #     and (iter_num % node_cut_frequency == 0)
        #         #     and len(self.best_cuts) < max_active_cuts
        #         # ):
        #         #     try:
        #         #         cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []

        #         #         T_loop = set(mst_edges)
        #         #         scored_loop = []
        #         #         for cut, rhs in cand_cuts_loop:
        #         #             S = set(cut)
        #         #             lhs = len(T_loop & S)
        #         #             violation = lhs - rhs
        #         #             if violation >= min_cut_violation_for_add:
        #         #                 scored_loop.append((violation, S, rhs))

        #         #         scored_loop.sort(reverse=True, key=lambda t: t[0])

        #         #         remaining_slots = max(0, max_active_cuts - len(self.best_cuts))
        #         #         if remaining_slots > 0:
        #         #             scored_loop = scored_loop[:min(max_inloop_new_cuts, remaining_slots)]
        #         #         else:
        #         #             scored_loop = []

        #         #         existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}

        #         #         added_any = False
        #         #         for violation, S, rhs in scored_loop:
        #         #             fz = frozenset(S)
        #         #             if fz in existing:
        #         #                 # if we ever wanted to strengthen rhs here we could, but usually in-loop we skip
        #         #                 continue

        #         #             # truly new cut
        #         #             self.best_cuts.append((set(S), rhs))
        #         #             new_idx = len(self.best_cuts) - 1
        #         #             self.best_cut_multipliers[new_idx] = 0.0
        #         #             self.best_cut_multipliers_for_best_bound[new_idx] = 0.0
        #         #             self._rhs_eff[new_idx] = int(rhs) - len(set(S) & F_in)
        #         #             max_cut_violation.append(0.0)
        #         #             node_new_cuts.append((set(S), rhs))
        #         #             added_any = True

        #         #         if added_any:
        #         #             # Rebuild index structures & rhs vectors to incorporate new cuts
        #         #             _rebuild_cut_structures()
        #         #             cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)
        #         #             # Reset modified weights cache with new μ dimension
        #         #             self._mw_cached = None
        #         #             self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)
        #         #     except Exception as e:
        #         #         if self.verbose:
        #         #             print(f"Error in in-loop separation at depth {depth}, iter {iter_num}: {e}")

        #         # Prepare weights for next iteration (using current λ, μ and possibly updated cuts)
        #         prev_weights = self.compute_modified_weights()

        #         # 2) Dual & primal bookkeeping
        #         is_feasible = (mst_length <= self.budget)

        #         # (a) primal & UB
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight)
        #                         and not math.isinf(real_weight)
        #                         and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # prune primal_solutions history (cheaper: just keep last MAX_SOLUTIONS)
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

        #         # (b) Lagrangian dual value (dualized cuts use ORIGINAL rhs)
        #         if self.use_cover_cuts and len(rhs_vec) > 0:
        #             mu_vec = np.fromiter(
        #                 (self.best_cut_multipliers.get(i, 0.0) for i in range(len(rhs_vec))),
        #                 dtype=float,
        #                 count=len(rhs_vec),
        #             )
        #             cover_cut_penalty = float(mu_vec @ rhs_vec)
        #         else:
        #             mu_vec = np.zeros(len(rhs_vec), dtype=float)
        #             cover_cut_penalty = 0.0

        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound)
        #                 and not math.isinf(lagrangian_bound)
        #                 and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #                 if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #         else:
        #             no_improvement_count += 1

        #         # 3) Subgradients
        #         # knapsack
        #         knapsack_subgradient = float(mst_length - self.budget)

        #         # cuts: TRUE dual subgradient for the encoded Lagrangian:
        #         # g_i = lhs_i - rhs_i, lhs_i counts ALL edges in S_i selected by MST
        #         if cuts_present_here and cuts_dynamic_here and len(cut_edge_idx_all) > 0:
        #             # Reuse a single mask array instead of allocating every iteration
        #             if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
        #                 self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #             mst_mask = self._mst_mask
        #             mst_mask[:] = False
        #             for e in mst_edges:
        #                 j = self.edge_indices.get(e)
        #                 if j is not None:
        #                     mst_mask[j] = True

        #             violations = []
        #             for i, idxs_all in enumerate(cut_edge_idx_all):
        #                 lhs_i = int(mst_mask[idxs_all].sum()) if idxs_all.size else 0
        #                 g_i = lhs_i - rhs_vec[i]  # <-- consistent with lagrangian_bound
        #                 violations.append(g_i)
        #                 # only *positive* violation counts as "usefulness"
        #                 if g_i > max_cut_violation[i]:
        #                     max_cut_violation[i] = g_i
        #             violations = np.array(violations, dtype=float)
        #             cut_subgradients = violations.tolist()
        #         else:
        #             # either no cuts, or we are in "frozen cuts" mode on deep node
        #             violations = np.zeros(0, dtype=float)
        #             cut_subgradients = []

        #         # 4) Step sizes & updates (joint Polyak for λ and μ)
        #         self.subgradients.append(knapsack_subgradient)

        #         # --- joint gradient norm ---
        #         norm_sq = knapsack_subgradient ** 2
        #         for g in cut_subgradients:
        #             norm_sq += g ** 2

        #         # --- Polyak-like step size for the joint dual (λ, μ) ---
        #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #             gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             theta = gamma_base
        #             alpha = theta * gap / (norm_sq + eps)
        #         else:
        #             # fallback small constant if UB is infinite or gradient is zero
        #             alpha = getattr(self, "step_size", 1e-5)

        #         # --- λ update with momentum, using the joint α ---
        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # --- μ updates (only if cuts are dynamic here), using the SAME α (scaled) ---
        #         if cuts_dynamic_here and len(cut_subgradients) > 0:
        #             for i, g in enumerate(cut_subgradients):
        #                 # scaled joint Polyak step for μ
        #                 delta = gamma_mu * alpha * g

        #                 # optional safety cap (keeps steps moderate)
        #                 if mu_increment_cap is not None:
        #                     if delta > mu_increment_cap:
        #                         delta = mu_increment_cap
        #                     elif delta < -mu_increment_cap:
        #                         delta = -mu_increment_cap

        #                 mu_old = self.best_cut_multipliers.get(i, 0.0)
        #                 mu_new = mu_old + delta

        #                 # project to μ_i >= 0
        #                 if mu_new < 0.0:
        #                     mu_new = 0.0

        #                 self.best_cut_multipliers[i] = mu_new

        #         # history bookkeeping
        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #         # λ stagnation check
        #         if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             current_step = getattr(self, "step_size", 1e-5)
        #             self.step_size = max(1e-8, current_step * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0
        #         last_g_lambda = knapsack_subgradient

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                 f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}, "
        #                 f"cuts={len(self.best_cuts)}")

        #     # ------------------------------------------------------------------
        #     # 6) Optional: drop "dead" cuts for future nodes
        #     #     Only where we actually tracked violations (dynamic nodes).
        #     # ------------------------------------------------------------------
        #     # if cuts_dynamic_here and self.use_cover_cuts and self.best_cuts:
        #     #     keep_indices = []
        #     #     for i, (cut, rhs) in enumerate(self.best_cuts):
        #     #         mu_i = float(self.best_cut_multipliers.get(i, 0.0))
        #     #         # drop if never strictly violated HERE and μ stayed tiny
        #     #         if max_cut_violation[i] == 0.0 and abs(mu_i) < dead_mu_threshold:
        #     #             continue
        #     #         keep_indices.append(i)

        #     #     if len(keep_indices) < len(self.best_cuts):
        #     #         new_best_cuts = []
        #     #         new_mu = {}
        #     #         new_mu_best = {}
        #     #         new_rhs_eff = {}
        #     #         for new_idx, old_idx in enumerate(keep_indices):
        #     #             new_best_cuts.append(self.best_cuts[old_idx])
        #     #             new_mu[new_idx] = float(self.best_cut_multipliers.get(old_idx, 0.0))
        #     #             new_mu_best[new_idx] = float(self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0))
        #     #             new_rhs_eff[new_idx] = self._rhs_eff[old_idx]
        #     #         self.best_cuts = new_best_cuts
        #     #         self.best_cut_multipliers = new_mu
        #     #         self.best_cut_multipliers_for_best_bound = new_mu_best
        #     #         self._rhs_eff = new_rhs_eff
        #     # if cuts_dynamic_here and self.use_cover_cuts and self.best_cuts:
        #     if self.use_cover_cuts and self.best_cuts:

        #         keep_indices = []

        #         # Use historical multipliers as a “memory” of usefulness
        #         parent_mu_map = getattr(
        #             self,
        #             "best_cut_multipliers_for_best_bound",
        #             self.best_cut_multipliers,  # fallback if the first map is missing/empty
        #         )

        #         for i, (cut, rhs) in enumerate(self.best_cuts):
        #             mu_i = float(self.best_cut_multipliers.get(i, 0.0))
        #             mu_hist = float(parent_mu_map.get(i, 0.0))

        #             # Has this cut ever been useful? (locally or when best bound was found)
        #             ever_useful = (max_cut_violation[i] > 0.0) or (abs(mu_hist) >= dead_mu_threshold)

        #             # Only drop if:
        #             #  - never violated here AND
        #             #  - both current μ and historical μ are tiny
        #             if (not ever_useful) and abs(mu_i) < dead_mu_threshold:
        #                 continue

        #             keep_indices.append(i)

        #         if len(keep_indices) < len(self.best_cuts):
        #             new_best_cuts = []
        #             new_mu = {}
        #             new_mu_best = {}
        #             new_rhs_eff = {}
        #             for new_idx, old_idx in enumerate(keep_indices):
        #                 new_best_cuts.append(self.best_cuts[old_idx])
        #                 new_mu[new_idx] = float(self.best_cut_multipliers.get(old_idx, 0.0))
        #                 new_mu_best[new_idx] = float(
        #                     self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0)
        #                 )
        #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]

        #             self.best_cuts = new_best_cuts
        #             self.best_cut_multipliers = new_mu
        #             self.best_cut_multipliers_for_best_bound = new_mu_best
        #             self._rhs_eff = new_rhs_eff

        #     # ------------------------------------------------------------------
        #     # 7) Update global flag if this node saw any violated cut
        #     #     (only meaningful on dynamic nodes)
        #     # ------------------------------------------------------------------
        #     # if cuts_dynamic_here and self.use_cover_cuts and max_cut_violation:
        #     #     if any(v > 0.0 for v in max_cut_violation):
        #     #         LagrangianMST.global_cut_violation_seen = True

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts

    

        # else:  # Subgradient method with Polyak hybrid + pre- & in-loop separation (with depth-based freezing, no new constants) badak nabod
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 50
        #     CLEANUP_INTERVAL = 100
        #     max_iter = min(self.max_iter, 200)  # unchanged

        #     # Polyak / momentum hyperparams (for λ)
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10

        #     # μ updates: conservative
        #     gamma_mu = getattr(self, "gamma_mu", 0.30)            # unchanged
        #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
        #     dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-6)
        #     eps = 1e-12

        #     # ------------------------------------------------------------------
        #     # Depth-based cutting policy
        #     # ------------------------------------------------------------------
        #     max_cut_depth = getattr(self, "max_cut_depth", 1)

        #     # Ensure cut data structures exist
        #     if not hasattr(self, "best_cuts"):
        #         self.best_cuts = []
        #     if not hasattr(self, "best_cut_multipliers"):
        #         self.best_cut_multipliers = {}
        #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers_for_best_bound = {}

        #     existing_cuts = self.best_cuts
        #     cutting_active_here = self.use_cover_cuts and (depth <= max_cut_depth)
        #     cuts_present_here = self.use_cover_cuts and bool(existing_cuts)

        #     # FAST-PATH: no cuts globally, or no cuts here (deep node with no inherited cuts)
        #     no_cuts_here = (not self.use_cover_cuts) or (not cutting_active_here and not cuts_present_here)

        #     # ------------------------------------------------------------------
        #     # Graph sanity guard
        #     # ------------------------------------------------------------------
        #     if not self.edge_list or self.num_nodes <= 1:
        #         if self.verbose:
        #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, []

        #     # Prepare fixed/excluded sets & edge index
        #     F_in = getattr(self, "fixed_edges", set())    # normalized tuples
        #     F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()
        #     edge_idx = self.edge_indices

        #     # ------------------------------------------------------------------
        #     # 0) PURE λ-ONLY SUBGRADIENT FAST PATH (no cuts relevant)
        #     # ------------------------------------------------------------------
        #     if no_cuts_here:
        #         # No cuts in the dual: run a compact λ-only subgradient method.
        #         no_improvement_count = 0
        #         polyak_enabled = True

        #         if not hasattr(self, "subgradients"):
        #             self.subgradients = []
        #         if not hasattr(self, "step_sizes"):
        #             self.step_sizes = []
        #         if not hasattr(self, "multipliers"):
        #             self.multipliers = []

        #         prev_weights = None
        #         prev_mst_edges = None
        #         last_g_lambda = None
        #         iter_num = 0

        #         while iter_num < max_iter:
        #             # 1) Solve MST on current priced weights
        #             if prev_weights is None:
        #                 prev_weights = self.compute_modified_weights()
        #             try:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #             except Exception:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()

        #             self.last_mst_edges = mst_edges
        #             prev_mst_edges = mst_edges

        #             # Prepare weights for next iteration
        #             prev_weights = self.compute_modified_weights()

        #             # 2) Primal & UB
        #             is_feasible = (mst_length <= self.budget)
        #             self._record_primal_solution(self.last_mst_edges, is_feasible)

        #             if is_feasible:
        #                 try:
        #                     real_weight, real_length = self.compute_real_weight_length()
        #                     if (not math.isnan(real_weight)
        #                             and not math.isinf(real_weight)
        #                             and real_weight < self.best_upper_bound):
        #                         self.best_upper_bound = real_weight
        #                 except Exception as e:
        #                     if self.verbose:
        #                         print(f"Error updating primal solution: {e}")

        #             # prune primal_solutions history
        #             if len(self.primal_solutions) > MAX_SOLUTIONS:
        #                 self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

        #             # 3) Dual value (no cover cuts)
        #             lagrangian_bound = mst_cost - self.lmbda * self.budget

        #             if (not math.isnan(lagrangian_bound)
        #                     and not math.isinf(lagrangian_bound)
        #                     and abs(lagrangian_bound) < 1e10):
        #                 if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                     self.best_lower_bound = lagrangian_bound
        #                     self.best_lambda = self.lmbda
        #                     self.best_mst_edges = self.last_mst_edges
        #                     self.best_cost = mst_cost
        #                     no_improvement_count = 0
        #                 else:
        #                     no_improvement_count += 1
        #                     if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                         self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #             else:
        #                 no_improvement_count += 1

        #             # 4) Subgradient & λ-update (Polyak + momentum)
        #             knapsack_subgradient = float(mst_length - self.budget)
        #             self.subgradients.append(knapsack_subgradient)

        #             norm_sq = knapsack_subgradient ** 2

        #             if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #                 gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #                 theta = gamma_base
        #                 alpha = theta * gap / (norm_sq + eps)
        #             else:
        #                 alpha = getattr(self, "step_size", 1e-5)

        #             v_prev = getattr(self, "_v_lambda", 0.0)
        #             v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #             self._v_lambda = v_new
        #             self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #             self.step_sizes.append(alpha)
        #             self.multipliers.append((self.lmbda, {}))

        #             # λ stagnation check
        #             if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #                 self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #                 if self.consecutive_same_subgradient > 10:
        #                     if self.verbose:
        #                         print("Terminating early (no-cuts mode): subgradient stagnation")
        #                     break
        #                 current_step = getattr(self, "step_size", 1e-5)
        #                 self.step_size = max(1e-8, current_step * 0.7)
        #             else:
        #                 self.consecutive_same_subgradient = 0
        #             last_g_lambda = knapsack_subgradient

        #             if self.verbose and iter_num % 10 == 0:
        #                 print(f"[No-cuts] Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                       f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #             iter_num += 1

        #         end_time = time()
        #         LagrangianMST.total_compute_time += end_time - start_time
        #         return self.best_lower_bound, self.best_upper_bound, []

        #     # ------------------------------------------------------------------
        #     # 1) CUT-AWARE PATH (dynamic or frozen cuts)
        #     # ------------------------------------------------------------------

        #     # Within this node:
        #     # - dynamic: we may generate cuts and update μ
        #     # - present: we have cuts in the dual (for pricing), even if frozen
        #     cuts_dynamic_here = self.use_cover_cuts and cutting_active_here
        #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Collect new cuts generated at THIS node (pre + in-loop) to return
        #     node_new_cuts = []

        #     # Will store MST from pre-separation to reuse in iter 0
        #     pre_mst_available = False
        #     pre_mst_cost = None
        #     pre_mst_length = None
        #     pre_mst_edges = None

        #     # ------------------------------------------------------------------
        #     # 2) PRE-SEPARATION AT NODE START (only warm-start MST now)
        #     # ------------------------------------------------------------------
        #     if cuts_dynamic_here:
        #         try:
        #             w0 = self.compute_modified_weights()
        #             try:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst_incremental(w0, None)
        #             except Exception:
        #                 mst_cost0, mst_len0, mst_edges0 = self.compute_mst()

        #             pre_mst_available = True
        #             pre_mst_cost = mst_cost0
        #             pre_mst_length = mst_len0
        #             pre_mst_edges = mst_edges0

        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Error in pre-separation at depth {depth}: {e}")

        #     # ------------------------------------------------------------------
        #     # 3) Compute rhs_eff and detect infeasibility (node-level)
        #     # ------------------------------------------------------------------
        #     self._rhs_eff = {}
        #     if cuts_present_here:
        #         for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #             rhs_eff = int(rhs) - len(cut & F_in)
        #             self._rhs_eff[idx_c] = rhs_eff
        #             if rhs_eff < 0:
        #                 # node infeasible due to fixed edges saturating the cut
        #                 end_time = time()
        #                 LagrangianMST.total_compute_time += end_time - start_time
        #                 return float('inf'), self.best_upper_bound, node_new_cuts

        #     # ------------------------------------------------------------------
        #     # 4) Trim number of cuts at node start (keep important ones)
        #     # ------------------------------------------------------------------
        #     max_active_cuts = getattr(self, "max_active_cuts", 5)
        #     if cuts_present_here and len(self.best_cuts) > max_active_cuts:
        #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
        #         if not parent_mu_map:
        #             parent_mu_map = self.best_cut_multipliers

        #         idx_and_cut = list(enumerate(self.best_cuts))
        #         # priority: large |μ|
        #         idx_and_cut.sort(
        #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
        #             reverse=True,
        #         )
        #         idx_and_cut = idx_and_cut[:max_active_cuts]

        #         new_cuts_list = []
        #         new_mu = {}
        #         new_mu_best = {}
        #         new_rhs_eff = {}
        #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
        #             new_cuts_list.append(cut_rhs)
        #             new_mu[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_mu_best[new_i] = parent_mu_map.get(old_i, 0.0)
        #             new_rhs_eff[new_i] = self._rhs_eff[old_i]

        #         self.best_cuts = new_cuts_list
        #         self.best_cut_multipliers = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu_best
        #         self._rhs_eff = new_rhs_eff

        #     # Re-evaluate presence after trimming
        #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #     # ------------------------------------------------------------------
        #     # 5) Precompute cut -> edge index arrays (FIXED for this node, but
        #     #    we will rebuild them if we add cuts in-loop)
        #     # ------------------------------------------------------------------
        #     cut_edge_idx_free = []
        #     cut_free_sizes = []
        #     cut_edge_idx_all = []
        #     rhs_eff_vec = np.zeros(0, dtype=float)
        #     rhs_vec = np.zeros(0, dtype=float)

        #     def _rebuild_cut_structures():
        #         """
        #         Rebuild index arrays and rhs vectors from self.best_cuts & self._rhs_eff.

        #         OPT:
        #         - If cover cuts are off, we just keep everything empty and return.
        #         - If cuts are present but not dynamic (frozen deep node), we skip building
        #           the 'free' arrays that are only used by dynamic separation.
        #         """
        #         nonlocal cut_edge_idx_free, cut_free_sizes, cut_edge_idx_all, rhs_eff_vec, rhs_vec

        #         cut_edge_idx_free = []
        #         cut_free_sizes = []
        #         cut_edge_idx_all = []

        #         if not (self.use_cover_cuts and self.best_cuts):
        #             self._cut_edge_idx = []
        #             self._cut_edge_idx_all = []
        #             rhs_eff_vec = np.zeros(0, dtype=float)
        #             rhs_vec = np.zeros(0, dtype=float)
        #             return

        #         build_free = cuts_dynamic_here

        #         for cut, rhs in self.best_cuts:
        #             # FREE indices (for dynamic refinements) – only if needed
        #             if build_free:
        #                 idxs_free = [
        #                     edge_idx[e] for e in cut
        #                     if (e not in F_in and e not in F_out) and (e in edge_idx)
        #                 ]
        #                 arr_free = (np.fromiter(idxs_free, dtype=np.int32)
        #                             if idxs_free else np.empty(0, dtype=np.int32))
        #                 cut_edge_idx_free.append(arr_free)
        #                 cut_free_sizes.append(max(1, len(idxs_free)))  # avoid /0
        #             else:
        #                 cut_edge_idx_free.append(np.empty(0, dtype=np.int32))
        #                 cut_free_sizes.append(1)

        #             # ALL indices (for dual pricing & true μ-subgradient)
        #             idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #             arr_all = (np.fromiter(idxs_all, dtype=np.int32)
        #                        if idxs_all else np.empty(0, dtype=np.int32))
        #             cut_edge_idx_all.append(arr_all)

        #         # stash for compute_modified_weights (used for pricing)
        #         self._cut_edge_idx = cut_edge_idx_free
        #         self._cut_edge_idx_all = cut_edge_idx_all

        #         if self.best_cuts:
        #             rhs_eff_vec = np.array(
        #                 [self._rhs_eff[i] for i in range(len(self.best_cuts))],
        #                 dtype=float
        #             )
        #             rhs_vec = np.array(
        #                 [rhs for (_, rhs) in self.best_cuts],
        #                 dtype=float
        #             )
        #         else:
        #             rhs_eff_vec = np.zeros(0, dtype=float)
        #             rhs_vec = np.zeros(0, dtype=float)

        #     if self.use_cover_cuts and (cuts_present_here or cuts_dynamic_here):
        #         _rebuild_cut_structures()
        #     else:
        #         self._cut_edge_idx = []
        #         self._cut_edge_idx_all = []
        #         rhs_eff_vec = np.zeros(0, dtype=float)
        #         rhs_vec = np.zeros(0, dtype=float)

        #     # track how "useful" each cut was at this node (only positive violation)
        #     max_cut_violation = [0.0 for _ in self.best_cuts] if (cuts_present_here or cuts_dynamic_here) else []

        #     # Histories / caches
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []

        #     # Seed priced weights so iteration 0 is consistent
        #     prev_weights = None
        #     prev_mst_edges = None

        #     last_g_lambda = None  # for stagnation check

        #     # ------------------------------------------------------------------
        #     # 6) Subgradient iterations (dual structure mostly fixed, but we may
        #     #    add a few extra cuts in-loop with full rebuild).
        #     # ------------------------------------------------------------------
        #     base_max_iter = int(max_iter)
        #     extra_iter_for_cuts = getattr(self, "extra_iter_for_cuts", base_max_iter)
        #     hard_cap_iter = base_max_iter + extra_iter_for_cuts

        #     dynamic_max_iter = base_max_iter
        #     iter_num = 0

        #     # Flags for violation-based cut generation
        #     violation_seen_for_cuts = False   # did we ever see len(T) > B in this node?
        #     did_separate_here = False         # did we already call generate_cover_cuts at this node?

        #     while iter_num < dynamic_max_iter:
        #         # 1) Solve MST on current priced weights
        #         if pre_mst_available:
        #             mst_cost = pre_mst_cost
        #             mst_length = pre_mst_length
        #             mst_edges = pre_mst_edges
        #             pre_mst_available = False
        #         else:
        #             if prev_weights is None:
        #                 prev_weights = self.compute_modified_weights()
        #             try:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #             except Exception:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()

        #         self.last_mst_edges = mst_edges
        #         prev_mst_edges = self.last_mst_edges

        #         # --------------------------------------------------------------
        #         # Detect violating MST and trigger ONE-TIME cut generation
        #         # --------------------------------------------------------------
        #         if mst_length > self.budget + 1e-12:
        #             violation_seen_for_cuts = True

        #             if (
        #                 self.use_cover_cuts
        #                 and cuts_dynamic_here
        #                 and not did_separate_here
        #                 and depth <= max_cut_depth
        #             ):
        #                 try:
        #                     cand_cuts = self.generate_cover_cuts(mst_edges) or []

        #                     T_loop = set(mst_edges)
        #                     scored = []
        #                     min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
        #                     max_active_cuts = getattr(self, "max_active_cuts", 5)
        #                     max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)

        #                     for cut, rhs in cand_cuts:
        #                         S = set(cut)
        #                         lhs = len(T_loop & S)
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

        #                     added_any = False
        #                     for violation, S, rhs in scored:
        #                         fz = frozenset(S)
        #                         if fz in existing:
        #                             old_rhs = existing[fz]
        #                             if rhs > old_rhs:
        #                                 idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                                 self.best_cuts[idx] = (set(S), rhs)
        #                                 existing[fz] = rhs
        #                             continue

        #                         self.best_cuts.append((set(S), rhs))
        #                         idx_new = len(self.best_cuts) - 1
        #                         self.best_cut_multipliers[idx_new] = 0.0
        #                         self.best_cut_multipliers_for_best_bound[idx_new] = 0.0
        #                         self._rhs_eff[idx_new] = int(rhs) - len(set(S) & F_in)
        #                         max_cut_violation.append(0.0)
        #                         node_new_cuts.append((set(S), rhs))
        #                         added_any = True

        #                     if added_any:
        #                         _rebuild_cut_structures()
        #                         cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)
        #                         self._mw_cached = None
        #                         self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

        #                 except Exception as e:
        #                     if self.verbose:
        #                         print(f"Error generating cuts from violating MST at depth {depth}, iter {iter_num}: {e}")

        #                 did_separate_here = True

        #         # Prepare weights for next iteration (using current λ, μ and possibly updated cuts)
        #         prev_weights = self.compute_modified_weights()

        #         # 2) Dual & primal bookkeeping
        #         is_feasible = (mst_length <= self.budget)

        #         # (a) primal & UB
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight)
        #                         and not math.isinf(real_weight)
        #                         and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

        #         # (b) Lagrangian dual value (dualized cuts use ORIGINAL rhs)
        #         if self.use_cover_cuts and len(rhs_vec) > 0:
        #             mu_vec = np.fromiter(
        #                 (self.best_cut_multipliers.get(i, 0.0) for i in range(len(rhs_vec))),
        #                 dtype=float,
        #                 count=len(rhs_vec),
        #             )
        #             cover_cut_penalty = float(mu_vec @ rhs_vec)
        #         else:
        #             mu_vec = np.zeros(len(rhs_vec), dtype=float)
        #             cover_cut_penalty = 0.0

        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound)
        #                 and not math.isinf(lagrangian_bound)
        #                 and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #                 if no_improvement_count % CLEANUP_INTERVAL == 0:
        #                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #         else:
        #             no_improvement_count += 1

        #         # 3) Subgradients
        #         knapsack_subgradient = float(mst_length - self.budget)

        #         if cuts_present_here and cuts_dynamic_here and len(cut_edge_idx_all) > 0:
        #             if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
        #                 self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #             mst_mask = self._mst_mask
        #             mst_mask[:] = False
        #             for e in mst_edges:
        #                 j = self.edge_indices.get(e)
        #                 if j is not None:
        #                     mst_mask[j] = True

        #             violations = []
        #             for i, idxs_all in enumerate(cut_edge_idx_all):
        #                 lhs_i = int(mst_mask[idxs_all].sum()) if idxs_all.size else 0
        #                 g_i = lhs_i - rhs_vec[i]
        #                 violations.append(g_i)
        #                 if g_i > max_cut_violation[i]:
        #                     max_cut_violation[i] = g_i
        #             violations = np.array(violations, dtype=float)
        #             cut_subgradients = violations.tolist()
        #         else:
        #             violations = np.zeros(0, dtype=float)
        #             cut_subgradients = []

        #         # 4) Step sizes & updates (joint Polyak for λ and μ)
        #         self.subgradients.append(knapsack_subgradient)

        #         norm_sq = knapsack_subgradient ** 2
        #         for g in cut_subgradients:
        #             norm_sq += g ** 2

        #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #             gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             theta = gamma_base
        #             alpha = theta * gap / (norm_sq + eps)
        #         else:
        #             alpha = getattr(self, "step_size", 1e-5)

        #         # λ update with momentum
        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # μ updates (only if cuts are dynamic here)
        #         if cuts_dynamic_here and len(cut_subgradients) > 0:
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

        #                 self.best_cut_multipliers[i] = mu_new

        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #         # λ stagnation check
        #         if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             current_step = getattr(self, "step_size", 1e-5)
        #             self.step_size = max(1e-8, current_step * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0
        #         last_g_lambda = knapsack_subgradient

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                   f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}, "
        #                   f"cuts={len(self.best_cuts)}")

        #         iter_num += 1

        #         # Optional extension of iterations ONLY for shallow cut nodes
        #         if (
        #             self.use_cover_cuts
        #             and cuts_dynamic_here
        #             and depth <= max_cut_depth
        #             and not violation_seen_for_cuts
        #             and iter_num == base_max_iter
        #         ):
        #             dynamic_max_iter = min(hard_cap_iter, base_max_iter + extra_iter_for_cuts)

        #     # ------------------------------------------------------------------
        #     # 7) Optional: drop "dead" cuts for future nodes
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts:
        #         keep_indices = []
        #         parent_mu_map = getattr(
        #             self,
        #             "best_cut_multipliers_for_best_bound",
        #             self.best_cut_multipliers,
        #         )

        #         for i, (cut, rhs) in enumerate(self.best_cuts):
        #             mu_i = float(self.best_cut_multipliers.get(i, 0.0))
        #             mu_hist = float(parent_mu_map.get(i, 0.0))

        #             ever_useful = (max_cut_violation[i] > 0.0) or (abs(mu_hist) >= dead_mu_threshold)

        #             if (not ever_useful) and abs(mu_i) < dead_mu_threshold:
        #                 continue

        #             keep_indices.append(i)

        #         if len(keep_indices) < len(self.best_cuts):
        #             new_best_cuts = []
        #             new_mu = {}
        #             new_mu_best = {}
        #             new_rhs_eff = {}
        #             for new_idx, old_idx in enumerate(keep_indices):
        #                 new_best_cuts.append(self.best_cuts[old_idx])
        #                 new_mu[new_idx] = float(self.best_cut_multipliers.get(old_idx, 0.0))
        #                 new_mu_best[new_idx] = float(
        #                     self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0)
        #                 )
        #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]

        #             self.best_cuts = new_best_cuts
        #             self.best_cut_multipliers = new_mu
        #             self.best_cut_multipliers_for_best_bound = new_mu_best
        #             self._rhs_eff = new_rhs_eff

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts



        else:  # Subgradient method with serious-step Polyak + depth-based cuts
            import numpy as np
            import math

            # --- Tunables / safety limits -----------------------------------
            MAX_SOLUTIONS = getattr(self, "max_solutions_hist", 50)
            CLEANUP_INTERVAL = getattr(self, "cleanup_interval", 100)
            max_iter_base = min(self.max_iter, getattr(self, "max_subgrad_iter", 200))

            # Polyak / momentum hyperparams (for λ)
            self.momentum_beta = getattr(self, "momentum_beta", 0.9)
            gamma_base = getattr(self, "gamma_base", 0.10)      # base Polyak scaling

            # μ updates: conservative
            gamma_mu = getattr(self, "gamma_mu", 0.30)
            mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
            dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-6)
            eps = 1e-12

            # serious-step control
            serious_improve_tol = getattr(self, "serious_improve_tol", 1e-3)
            serious_patience = getattr(self, "serious_patience", 8)
            alpha_max = getattr(self, "alpha_max", 1.0)         # cap on Polyak step
            grad_tol = getattr(self, "grad_tol", 1e-5)          # joint grad norm tol

            # ------------------------------------------------------------------
            # Depth-based cutting policy
            # ------------------------------------------------------------------
            max_cut_depth = getattr(self, "max_cut_depth", 1)

            # Ensure cut data structures exist
            if not hasattr(self, "best_cuts"):
                self.best_cuts = []
            if not hasattr(self, "best_cut_multipliers"):
                self.best_cut_multipliers = {}
            if not hasattr(self, "best_cut_multipliers_for_best_bound"):
                self.best_cut_multipliers_for_best_bound = {}

            existing_cuts = self.best_cuts
            cutting_active_here = self.use_cover_cuts and (depth <= max_cut_depth)
            cuts_present_here = self.use_cover_cuts and bool(existing_cuts)

            # FAST-PATH: no cuts globally, or no cuts here (deep node with no inherited cuts)
            no_cuts_here = (not self.use_cover_cuts) or (not cutting_active_here and not cuts_present_here)

            # ------------------------------------------------------------------
            # Graph sanity guard
            # ------------------------------------------------------------------
            if not self.edge_list or self.num_nodes <= 1:
                if self.verbose:
                    print(f"Error at depth {depth}: Empty edge list or invalid graph")
                end_time = time()
                LagrangianMST.total_compute_time += end_time - start_time
                return self.best_lower_bound, self.best_upper_bound, []

            # Prepare fixed/excluded sets & edge index
            F_in = getattr(self, "fixed_edges", set())    # normalized tuples
            F_out = getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set()
            edge_idx = self.edge_indices

            # ------------------------------------------------------------------
            # 0) PURE λ-ONLY SUBGRADIENT FAST PATH (no cuts relevant)
            # ------------------------------------------------------------------
            if no_cuts_here:
                no_improvement_count = 0
                polyak_enabled = True

                if not hasattr(self, "subgradients"):
                    self.subgradients = []
                if not hasattr(self, "step_sizes"):
                    self.step_sizes = []
                if not hasattr(self, "multipliers"):
                    self.multipliers = []

                prev_weights = None
                prev_mst_edges = None
                last_g_lambda = None
                iter_num = 0

                # serious-step tracking for λ-only case (uses global best_upper_bound)
                best_serious_bound = -float("inf")
                non_serious_iters = 0
                theta = gamma_base

                while iter_num < max_iter_base:
                    # 1) Solve MST on current priced weights
                    if prev_weights is None:
                        prev_weights = self.compute_modified_weights()
                    try:
                        mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
                    except Exception:
                        mst_cost, mst_length, mst_edges = self.compute_mst()

                    self.last_mst_edges = mst_edges
                    prev_mst_edges = mst_edges

                    # Prepare weights for next iteration
                    prev_weights = self.compute_modified_weights()

                    # 2) Primal & UB
                    is_feasible = (mst_length <= self.budget)
                    self._record_primal_solution(self.last_mst_edges, is_feasible)

                    if is_feasible:
                        try:
                            real_weight, real_length = self.compute_real_weight_length()
                            if (not math.isnan(real_weight)
                                    and not math.isinf(real_weight)
                                    and real_weight < self.best_upper_bound):
                                self.best_upper_bound = real_weight
                        except Exception as e:
                            if self.verbose:
                                print(f"Error updating primal solution: {e}")

                    if len(self.primal_solutions) > MAX_SOLUTIONS:
                        self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

                    # 3) Dual value (no cover cuts)
                    lagrangian_bound = mst_cost - self.lmbda * self.budget

                    if (not math.isnan(lagrangian_bound)
                            and not math.isinf(lagrangian_bound)
                            and abs(lagrangian_bound) < 1e10):
                        if lagrangian_bound > self.best_lower_bound + 1e-6:
                            self.best_lower_bound = lagrangian_bound
                            self.best_lambda = self.lmbda
                            self.best_mst_edges = self.last_mst_edges
                            self.best_cost = mst_cost
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                            if no_improvement_count % CLEANUP_INTERVAL == 0:
                                self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
                    else:
                        no_improvement_count += 1

                    # 4) Subgradient & λ-update (serious-step Polyak + momentum)
                    knapsack_subgradient = float(mst_length - self.budget)
                    self.subgradients.append(knapsack_subgradient)

                    norm_sq = knapsack_subgradient ** 2

                    if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
                        gap = max(0.0, self.best_upper_bound - lagrangian_bound)
                        alpha = theta * gap / (norm_sq + eps)
                        if alpha_max is not None:
                            alpha = min(alpha, alpha_max)
                    else:
                        alpha = getattr(self, "step_size", 1e-5)

                    # momentum update
                    v_prev = getattr(self, "_v_lambda", 0.0)
                    v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
                    self._v_lambda = v_new
                    self.lmbda = max(0.0, self.lmbda + alpha * v_new)

                    self.step_sizes.append(alpha)
                    self.multipliers.append((self.lmbda, {}))

                    # serious-step logic on dual bound
                    if lagrangian_bound > best_serious_bound + serious_improve_tol:
                        best_serious_bound = lagrangian_bound
                        non_serious_iters = 0
                    else:
                        non_serious_iters += 1
                        if non_serious_iters >= serious_patience:
                            theta *= 0.5
                            non_serious_iters = 0

                    # λ stagnation check
                    if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
                        self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
                        if self.consecutive_same_subgradient > 10:
                            if self.verbose:
                                print("Terminating early (no-cuts mode): subgradient stagnation")
                            break
                        current_step = getattr(self, "step_size", 1e-5)
                        self.step_size = max(1e-8, current_step * 0.7)
                    else:
                        self.consecutive_same_subgradient = 0
                    last_g_lambda = knapsack_subgradient

                    # stopping by gradient norm
                    if norm_sq < grad_tol ** 2:
                        if self.verbose:
                            print("Terminating early (no-cuts mode): small gradient norm")
                        break

                    if self.verbose and iter_num % 10 == 0:
                        print(f"[No-cuts] Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
                              f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

                    iter_num += 1

                end_time = time()
                LagrangianMST.total_compute_time += end_time - start_time
                return self.best_lower_bound, self.best_upper_bound, []

            # ------------------------------------------------------------------
            # 1) CUT-AWARE PATH (dynamic or frozen cuts)
            # ------------------------------------------------------------------
            cuts_dynamic_here = self.use_cover_cuts and cutting_active_here
            cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

            no_improvement_count = 0
            polyak_enabled = True
            node_new_cuts = []

            # --------------------------------------------------------------
            # Pre-separation warm-start (only for dynamic cut nodes)
            # --------------------------------------------------------------
            pre_mst_available = False
            pre_mst_cost = None
            pre_mst_length = None
            pre_mst_edges = None

            if cuts_dynamic_here:
                try:
                    w0 = self.compute_modified_weights()
                    try:
                        mst_cost0, mst_len0, mst_edges0 = self.compute_mst_incremental(w0, None)
                    except Exception:
                        mst_cost0, mst_len0, mst_edges0 = self.compute_mst()

                    pre_mst_available = True
                    pre_mst_cost = mst_cost0
                    pre_mst_length = mst_len0
                    pre_mst_edges = mst_edges0

                except Exception as e:
                    if self.verbose:
                        print(f"Error in pre-separation at depth {depth}: {e}")

            # ------------------------------------------------------------------
            # 2) Compute rhs_eff and detect infeasibility (fixed edges)
            # ------------------------------------------------------------------
            self._rhs_eff = {}
            if cuts_present_here:
                for idx_c, (cut, rhs) in enumerate(self.best_cuts):
                    rhs_eff = int(rhs) - len(cut & F_in)
                    self._rhs_eff[idx_c] = rhs_eff
                    if rhs_eff < 0:
                        end_time = time()
                        LagrangianMST.total_compute_time += end_time - start_time
                        return float('inf'), self.best_upper_bound, node_new_cuts

            # ------------------------------------------------------------------
            # 3) Trim number of cuts at node start (keep important ones)
            # ------------------------------------------------------------------
            max_active_cuts = getattr(self, "max_active_cuts", 5)
            if cuts_present_here and len(self.best_cuts) > max_active_cuts:
                parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
                if not parent_mu_map:
                    parent_mu_map = self.best_cut_multipliers

                idx_and_cut = list(enumerate(self.best_cuts))
                idx_and_cut.sort(
                    key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
                    reverse=True,
                )
                idx_and_cut = idx_and_cut[:max_active_cuts]

                new_cuts_list = []
                new_mu = {}
                new_mu_best = {}
                new_rhs_eff = {}
                for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
                    new_cuts_list.append(cut_rhs)
                    new_mu[new_i] = parent_mu_map.get(old_i, 0.0)
                    new_mu_best[new_i] = parent_mu_map.get(old_i, 0.0)
                    new_rhs_eff[new_i] = self._rhs_eff[old_i]

                self.best_cuts = new_cuts_list
                self.best_cut_multipliers = new_mu
                self.best_cut_multipliers_for_best_bound = new_mu_best
                self._rhs_eff = new_rhs_eff

            cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

            # ------------------------------------------------------------------
            # 4) Precompute cut -> edge index arrays (rebuilt when adding cuts)
            # ------------------------------------------------------------------
            cut_edge_idx_free = []
            cut_free_sizes = []
            cut_edge_idx_all = []
            rhs_eff_vec = np.zeros(0, dtype=float)
            rhs_vec = np.zeros(0, dtype=float)

            def _rebuild_cut_structures():
                nonlocal cut_edge_idx_free, cut_free_sizes, cut_edge_idx_all, rhs_eff_vec, rhs_vec

                cut_edge_idx_free = []
                cut_free_sizes = []
                cut_edge_idx_all = []

                if not (self.use_cover_cuts and self.best_cuts):
                    self._cut_edge_idx = []
                    self._cut_edge_idx_all = []
                    rhs_eff_vec = np.zeros(0, dtype=float)
                    rhs_vec = np.zeros(0, dtype=float)
                    return

                build_free = cuts_dynamic_here

                for cut, rhs in self.best_cuts:
                    if build_free:
                        idxs_free = [
                            edge_idx[e] for e in cut
                            if (e not in F_in and e not in F_out) and (e in edge_idx)
                        ]
                        arr_free = (np.fromiter(idxs_free, dtype=np.int32)
                                    if idxs_free else np.empty(0, dtype=np.int32))
                        cut_edge_idx_free.append(arr_free)
                        cut_free_sizes.append(max(1, len(idxs_free)))
                    else:
                        cut_edge_idx_free.append(np.empty(0, dtype=np.int32))
                        cut_free_sizes.append(1)

                    idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
                    arr_all = (np.fromiter(idxs_all, dtype=np.int32)
                               if idxs_all else np.empty(0, dtype=np.int32))
                    cut_edge_idx_all.append(arr_all)

                self._cut_edge_idx = cut_edge_idx_free
                self._cut_edge_idx_all = cut_edge_idx_all

                if self.best_cuts:
                    rhs_eff_vec = np.array(
                        [self._rhs_eff[i] for i in range(len(self.best_cuts))],
                        dtype=float
                    )
                    rhs_vec = np.array(
                        [rhs for (_, rhs) in self.best_cuts],
                        dtype=float
                    )
                else:
                    rhs_eff_vec = np.zeros(0, dtype=float)
                    rhs_vec = np.zeros(0, dtype=float)

            if self.use_cover_cuts and (cuts_present_here or cuts_dynamic_here):
                _rebuild_cut_structures()
            else:
                self._cut_edge_idx = []
                self._cut_edge_idx_all = []
                rhs_eff_vec = np.zeros(0, dtype=float)
                rhs_vec = np.zeros(0, dtype=float)

            max_cut_violation = [0.0 for _ in self.best_cuts] if (cuts_present_here or cuts_dynamic_here) else []

            # Histories / caches
            self._mw_cached = None
            self._mw_lambda = None
            self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

            if not hasattr(self, "subgradients"):
                self.subgradients = []
            if not hasattr(self, "step_sizes"):
                self.step_sizes = []
            if not hasattr(self, "multipliers"):
                self.multipliers = []

            prev_weights = None
            prev_mst_edges = None
            last_g_lambda = None

            # serious-step tracking (joint λ, μ)
            best_serious_bound = -float("inf")
            non_serious_iters = 0
            theta_joint = gamma_base

            # dynamic iteration control
            base_max_iter = int(max_iter_base)
            extra_iter_for_cuts = getattr(self, "extra_iter_for_cuts", base_max_iter)
            hard_cap_iter = base_max_iter + extra_iter_for_cuts

            dynamic_max_iter = base_max_iter
            iter_num = 0

            violation_seen_for_cuts = False
            did_separate_here = False

            while iter_num < dynamic_max_iter:
                # 1) Solve MST on current priced weights
                if pre_mst_available:
                    mst_cost = pre_mst_cost
                    mst_length = pre_mst_length
                    mst_edges = pre_mst_edges
                    pre_mst_available = False
                else:
                    if prev_weights is None:
                        prev_weights = self.compute_modified_weights()
                    try:
                        mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
                    except Exception:
                        mst_cost, mst_length, mst_edges = self.compute_mst()

                self.last_mst_edges = mst_edges
                prev_mst_edges = self.last_mst_edges

                # ----------------------------------------------------------
                # Cut separation: one-shot, only on clear knapsack violation
                # ----------------------------------------------------------
                if mst_length > self.budget + 1e-12:
                    violation_seen_for_cuts = True

                    if (
                        self.use_cover_cuts
                        and cuts_dynamic_here
                        and not did_separate_here
                        and depth <= max_cut_depth
                    ):
                        try:
                            cand_cuts = self.generate_cover_cuts(mst_edges) or []

                            T_loop = set(mst_edges)
                            scored = []
                            min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
                            max_active_cuts = getattr(self, "max_active_cuts", 5)
                            max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)

                            for cut, rhs in cand_cuts:
                                S = set(cut)
                                lhs = len(T_loop & S)
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

                            added_any = False
                            for violation, S, rhs in scored:
                                fz = frozenset(S)
                                if fz in existing:
                                    old_rhs = existing[fz]
                                    if rhs > old_rhs:
                                        idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
                                        self.best_cuts[idx] = (set(S), rhs)
                                        existing[fz] = rhs
                                    continue

                                self.best_cuts.append((set(S), rhs))
                                idx_new = len(self.best_cuts) - 1
                                self.best_cut_multipliers[idx_new] = 0.0
                                self.best_cut_multipliers_for_best_bound[idx_new] = 0.0
                                self._rhs_eff[idx_new] = int(rhs) - len(set(S) & F_in)
                                max_cut_violation.append(0.0)
                                node_new_cuts.append((set(S), rhs))
                                added_any = True

                            if added_any:
                                _rebuild_cut_structures()
                                cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)
                                self._mw_cached = None
                                self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

                        except Exception as e:
                            if self.verbose:
                                print(f"Error generating cuts at depth {depth}, iter {iter_num}: {e}")

                        did_separate_here = True

                prev_weights = self.compute_modified_weights()

                # 2) Dual & primal bookkeeping
                is_feasible = (mst_length <= self.budget)

                self._record_primal_solution(self.last_mst_edges, is_feasible)
                if is_feasible:
                    try:
                        real_weight, real_length = self.compute_real_weight_length()
                        if (not math.isnan(real_weight)
                                and not math.isinf(real_weight)
                                and real_weight < self.best_upper_bound):
                            self.best_upper_bound = real_weight
                    except Exception as e:
                        if self.verbose:
                            print(f"Error updating primal solution: {e}")

                if len(self.primal_solutions) > MAX_SOLUTIONS:
                    self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

                if self.use_cover_cuts and len(rhs_vec) > 0:
                    mu_vec = np.fromiter(
                        (self.best_cut_multipliers.get(i, 0.0) for i in range(len(rhs_vec))),
                        dtype=float,
                        count=len(rhs_vec),
                    )
                    cover_cut_penalty = float(mu_vec @ rhs_vec)
                else:
                    mu_vec = np.zeros(len(rhs_vec), dtype=float)
                    cover_cut_penalty = 0.0

                lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

                if (not math.isnan(lagrangian_bound)
                        and not math.isinf(lagrangian_bound)
                        and abs(lagrangian_bound) < 1e10):
                    if lagrangian_bound > self.best_lower_bound + 1e-6:
                        self.best_lower_bound = lagrangian_bound
                        self.best_lambda = self.lmbda
                        self.best_mst_edges = self.last_mst_edges
                        self.best_cost = mst_cost
                        self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        if no_improvement_count % CLEANUP_INTERVAL == 0:
                            self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
                else:
                    no_improvement_count += 1

                # 3) Subgradients
                knapsack_subgradient = float(mst_length - self.budget)

                if cuts_present_here and cuts_dynamic_here and len(cut_edge_idx_all) > 0:
                    if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
                        self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
                    mst_mask = self._mst_mask
                    mst_mask[:] = False
                    for e in mst_edges:
                        j = self.edge_indices.get(e)
                        if j is not None:
                            mst_mask[j] = True

                    violations = []
                    for i, idxs_all in enumerate(cut_edge_idx_all):
                        lhs_i = int(mst_mask[idxs_all].sum()) if idxs_all.size else 0
                        g_i = lhs_i - rhs_vec[i]
                        violations.append(g_i)
                        if g_i > max_cut_violation[i]:
                            max_cut_violation[i] = g_i
                    violations = np.array(violations, dtype=float)
                    cut_subgradients = violations.tolist()
                else:
                    violations = np.zeros(0, dtype=float)
                    cut_subgradients = []

                # 4) Joint Polyak step for (λ, μ)
                self.subgradients.append(knapsack_subgradient)

                norm_sq = knapsack_subgradient ** 2
                for g in cut_subgradients:
                    norm_sq += g ** 2

                if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
                    gap = max(0.0, self.best_upper_bound - lagrangian_bound)
                    alpha = theta_joint * gap / (norm_sq + eps)
                    if alpha_max is not None:
                        alpha = min(alpha, alpha_max)
                else:
                    alpha = getattr(self, "step_size", 1e-5)

                # λ update with momentum
                v_prev = getattr(self, "_v_lambda", 0.0)
                v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
                self._v_lambda = v_new
                self.lmbda = max(0.0, self.lmbda + alpha * v_new)

                # μ updates (only dynamic cut nodes)
                if cuts_dynamic_here and len(cut_subgradients) > 0:
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
                        self.best_cut_multipliers[i] = mu_new

                self.step_sizes.append(alpha)
                self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

                # serious-step logic on dual bound
                if lagrangian_bound > best_serious_bound + serious_improve_tol:
                    best_serious_bound = lagrangian_bound
                    non_serious_iters = 0
                else:
                    non_serious_iters += 1
                    if non_serious_iters >= serious_patience:
                        theta_joint *= 0.5
                        non_serious_iters = 0

                # λ stagnation check
                if last_g_lambda is not None and abs(knapsack_subgradient - last_g_lambda) < 1e-6:
                    self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
                    if self.consecutive_same_subgradient > 10:
                        if self.verbose:
                            print("Terminating early: subgradient stagnation")
                        break
                    current_step = getattr(self, "step_size", 1e-5)
                    self.step_size = max(1e-8, current_step * 0.7)
                else:
                    self.consecutive_same_subgradient = 0
                last_g_lambda = knapsack_subgradient

                # stopping: gradient norm and dual gap
                if norm_sq < grad_tol ** 2:
                    if self.verbose:
                        print("Terminating early: small joint gradient norm")
                    break
                if (self.best_upper_bound < float("inf")
                        and self.best_upper_bound - self.best_lower_bound <= getattr(self, "dual_gap_tol", 1e-4)):
                    if self.verbose:
                        print("Terminating early: small dual gap")
                    break

                if self.verbose and iter_num % 10 == 0:
                    print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
                          f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}, "
                          f"cuts={len(self.best_cuts)}")

                iter_num += 1

                # Optional extension of iterations ONLY for shallow cut nodes
                if (
                    self.use_cover_cuts
                    and cuts_dynamic_here
                    and depth <= max_cut_depth
                    and not violation_seen_for_cuts
                    and iter_num == base_max_iter
                ):
                    dynamic_max_iter = min(hard_cap_iter, base_max_iter + extra_iter_for_cuts)

            # ------------------------------------------------------------------
            # 5) Drop "dead" cuts for future nodes (never violated & tiny μ)
            # ------------------------------------------------------------------
            if self.use_cover_cuts and self.best_cuts:
                keep_indices = []
                parent_mu_map = getattr(
                    self,
                    "best_cut_multipliers_for_best_bound",
                    self.best_cut_multipliers,
                )

                for i, (cut, rhs) in enumerate(self.best_cuts):
                    mu_i = float(self.best_cut_multipliers.get(i, 0.0))
                    mu_hist = float(parent_mu_map.get(i, 0.0))

                    ever_useful = (max_cut_violation[i] > 0.0) or (abs(mu_hist) >= dead_mu_threshold)
                    if (not ever_useful) and abs(mu_i) < dead_mu_threshold:
                        continue
                    keep_indices.append(i)

                if len(keep_indices) < len(self.best_cuts):
                    new_best_cuts = []
                    new_mu = {}
                    new_mu_best = {}
                    new_rhs_eff = {}
                    for new_idx, old_idx in enumerate(keep_indices):
                        new_best_cuts.append(self.best_cuts[old_idx])
                        new_mu[new_idx] = float(self.best_cut_multipliers.get(old_idx, 0.0))
                        new_mu_best[new_idx] = float(
                            self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0)
                        )
                        new_rhs_eff[new_idx] = self._rhs_eff[old_idx]

                    self.best_cuts = new_best_cuts
                    self.best_cut_multipliers = new_mu
                    self.best_cut_multipliers_for_best_bound = new_mu_best
                    self._rhs_eff = new_rhs_eff

            end_time = time()
            LagrangianMST.total_compute_time += end_time - start_time
            return self.best_lower_bound, self.best_upper_bound, node_new_cuts




    








            
        
        # else:  # Subgradient method with Polyak hybrid (refined, faster & integer-safe)
        #     import numpy as np

        #     # ---------- helpers to normalize edges (accept (u,v) or edge-index int) ----------
        #     # build reverse map idx -> (u,v) once if missing
        #     if not hasattr(self, "idx_to_edge") or not self.idx_to_edge:
        #         try:
        #             self.idx_to_edge = {idx: e for e, idx in self.edge_indices.items()}
        #         except Exception:
        #             self.idx_to_edge = {}

        #     def _norm_edge(e):
        #         """Return normalized (u,v) with u<=v from either (u,v) or int; else None."""
        #         if isinstance(e, tuple) and len(e) == 2:
        #             u, v = e
        #         elif isinstance(e, int):
        #             uv = self.idx_to_edge.get(e)
        #             if uv is None:
        #                 return None
        #             u, v = uv
        #         else:
        #             return None
        #         return (u, v) if u <= v else (v, u)

        #     def _norm_edge_set(S):
        #         """Normalize an iterable of edges into a set of (u,v) tuples."""
        #         out = set()
        #         if not S:
        #             return out
        #         for e in S:
        #             uv = _norm_edge(e)
        #             if uv is not None:
        #                 out.add(uv)
        #         return out

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     MAX_HISTORY = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10
        #     gamma_mu = 1.0
        #     mu_increment_cap = 5.0
        #     eps = 1e-12

        #     no_improvement_count = 0
        #     polyak_enabled = True
        #     stagnation_threshold = 15

        #     # guard against premature exit at the root
        #     MIN_ITERS_BEFORE_CONV = 3

        #     # Initialize moving UB for Polyak gap control
        #     if not hasattr(self, "_moving_upper"):
        #         self._moving_upper = self.best_upper_bound if self.best_upper_bound < float("inf") else 1000.0

        #     # -------- Build rhs_eff and early-prune impossible nodes (once per node) --------
        #     # Normalize fixed/forbidden sets defensively in case ints sneak in.
        #     F_in = _norm_edge_set(getattr(self, "fixed_edges", set()))   # normalized tuples
        #     F_out = _norm_edge_set(getattr(self, "forbidden_edges", set()) if hasattr(self, "forbidden_edges") else set())

        #     # Normalize existing best_cuts once to avoid int elements inside
        #     if self.best_cuts:
        #         self.best_cuts = [(_norm_edge_set(cut), int(rhs)) for cut, rhs in self.best_cuts]

        #     self._rhs_eff = {}
        #     for idx, (cut, rhs) in enumerate(self.best_cuts):
        #         rhs_eff = rhs - len(cut & F_in)
        #         self._rhs_eff[idx] = rhs_eff
        #         if rhs_eff < 0:
        #             # Node infeasible due to fixed-in edges saturating the cut
        #             end_time = time()
        #             LagrangianMST.total_compute_time += end_time - start_time
        #             return float('inf'), self.best_upper_bound, []

        #     # -------- Precompute cut→edge-index arrays for FREE edges (once per node) --------
        #     # FREE = all edges except F_in and (optionally) F_out
        #     edge_idx = self.edge_indices  # normalized edge -> index
        #     cut_edge_idx = []
        #     cut_free_sizes = []
        #     for cut, _ in self.best_cuts:
        #         # indices for free edges only
        #         idxs = [edge_idx[e] for e in cut if (e not in F_in and e not in F_out) and (e in edge_idx)]
        #         arr = np.fromiter(idxs, dtype=np.int32) if idxs else np.empty(0, dtype=np.int32)
        #         cut_edge_idx.append(arr)
        #         cut_free_sizes.append(max(1, len(idxs)))   # avoid div-by-zero in normalization

        #     self._cut_edge_idx = cut_edge_idx  # used by compute_modified_weights fast path
        #     rhs_eff_vec = np.array([self._rhs_eff[i] for i in range(len(cut_edge_idx))], dtype=float)

        #     # Reset modified-weights cache for this node
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx), dtype=float)

        #     # Ensure histories exist
        #     if not hasattr(self, "subgradients"):
        #         self.subgradients = []
        #     if not hasattr(self, "step_sizes"):
        #         self.step_sizes = []
        #     if not hasattr(self, "multipliers"):
        #         self.multipliers = []
        #     if not hasattr(self, "best_cut_multipliers"):
        #         self.best_cut_multipliers = {}
        #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers_for_best_bound = {}

        #     # seed μ entries for all existing cuts (avoid KeyErrors later)
        #     for i in range(len(self.best_cuts)):
        #         if i not in self.best_cut_multipliers:
        #             self.best_cut_multipliers[i] = 0.0
        #         if i not in self.best_cut_multipliers_for_best_bound:
        #             self.best_cut_multipliers_for_best_bound[i] = 0.0

        #     # --- CRITICAL: seed priced weights so iteration 0 is correct ---
        #     prev_weights = self.compute_modified_weights()   # priced by current (λ, μ)
        #     prev_mst_edges = None
        #     plateau_counter = 0
        #     # ensure this exists
        #     self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0)

        #     # initialize new_cuts to ensure safe return
        #     new_cuts = []

        #     for iter_num in range(max_iter):
        #         if iter_num % 10 == 0:
        #             if len(self.primal_solutions) > MAX_SOLUTIONS:
        #                 self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #             if len(self.subgradients) > MAX_HISTORY:
        #                 self.subgradients = self.subgradients[-MAX_HISTORY:]
        #             if len(self.step_sizes) > MAX_HISTORY:
        #                 self.step_sizes = self.step_sizes[-MAX_HISTORY:]
        #             if len(self.multipliers) > MAX_HISTORY:
        #                 self.multipliers = self.multipliers[-MAX_HISTORY:]
        #             if len(self.fractional_solutions) > 10:
        #                 self.fractional_solutions = self.fractional_solutions[-5:]

        #         # -------- Step 1: MST (incremental w/ fallback) --------
        #         try:
        #             if prev_weights is not None and prev_mst_edges is not None:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #             else:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()

        #             if (math.isnan(mst_cost) or math.isinf(mst_cost) or
        #                 math.isnan(mst_length) or math.isinf(mst_length)):
        #                 if self.verbose:
        #                     print(f"Subgradient Iter {iter_num}: Invalid MST, fallback to full")
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Error in MST: {e}, fallback")
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         # Normalize and store current MST edges (ensure (u,v) tuples)
        #         self.last_mst_edges = [_norm_edge(e) for e in mst_edges if _norm_edge(e) is not None]
        #         prev_mst_edges = self.last_mst_edges

        #         # Build/Update modified weights (fast incremental) for NEXT iter
        #         prev_weights = self.compute_modified_weights()

        #         # -------- Step 2: Dual bound & book-keeping --------
        #         is_feasible = (mst_length <= self.budget)
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         # Penalty: Σ μ_i * rhs_eff(i)   (vectorized)
        #         mu_vec = np.fromiter((self.best_cut_multipliers.get(i, 0.0) for i in range(len(cut_edge_idx))),
        #                             dtype=float, count=len(cut_edge_idx))
        #         cover_cut_penalty = float(mu_vec @ rhs_eff_vec)
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound) and not math.isinf(lagrangian_bound)
        #             and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #         else:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Invalid L={lagrangian_bound}")
        #             no_improvement_count += 1

        #         # -------- Step 3: Update best primal if feasible (skip UF connectivity) --------
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight) and not math.isinf(real_weight)
        #                     and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # -------- Step 4: Subgradients (λ and μ) --------
        #         knapsack_subgradient = mst_length - self.budget

        #         # k_free via bit mask on edges (vectorized)
        #         nE = len(self.edge_weights)
        #         mst_mask = np.zeros(nE, dtype=bool)
        #         # mark chosen FREE edges True
        #         for e in self.last_mst_edges:
        #             if e in F_in or e in F_out:
        #                 continue
        #             j = edge_idx.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         violations = np.array([
        #             (int(mst_mask[idxs].sum()) if idxs.size else 0) - rhs_eff_vec[i]
        #             for i, idxs in enumerate(cut_edge_idx)
        #         ])
        #         cut_subgradients = violations.tolist()
        #         cut_sizes = cut_free_sizes

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, "
        #                 f"len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #         # stagnation in knapsack subgradient
        #         if iter_num > 0 and abs(knapsack_subgradient - self.subgradients[-1]) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0

        #         self._record_subgradient(knapsack_subgradient)
        #         if len(self.subgradients) > MAX_HISTORY:
        #             self.subgradients = self.subgradients[-MAX_HISTORY//2:]

        #         # -------- Step 5: Convergence criteria --------
        #         converged = (abs(knapsack_subgradient) < 1e-5 and
        #                     all(abs(g) < 1e-5 for g in cut_subgradients))

        #         duality_gap = float('inf')
        #         if (self.best_upper_bound < float('inf') and self.best_lower_bound > -float('inf')):
        #             duality_gap = self.best_upper_bound - self.best_lower_bound

        #         subgrad_norm = math.sqrt(
        #             knapsack_subgradient ** 2 + sum(g ** 2 for g in cut_subgradients)
        #         )

        #         # Don't allow early exit before a few iterations
        #         if (iter_num >= MIN_ITERS_BEFORE_CONV) and (
        #             converged or
        #             (0 <= duality_gap < 1e-4) or
        #             subgrad_norm < 1e-4 or
        #             no_improvement_count > stagnation_threshold or
        #             self.step_size < 1e-8
        #         ):
        #             if self.verbose:
        #                 reason = ("converged" if converged else
        #                         "small duality gap" if 0 <= duality_gap < 1e-4 else
        #                         "small subgrad norm" if subgrad_norm < 1e-4 else
        #                         "no improvement" if no_improvement_count > stagnation_threshold else
        #                         "step too small")
        #                 print(f"Converged at iter {iter_num}! (Reason: {reason})")
        #             break

        #         # -------- Step 6: Hybrid Polyak + Decay updates (unchanged logic) --------
        #         self.step_sizes.append(self.step_size)
        #         if len(self.step_sizes) > MAX_HISTORY:
        #             self.step_sizes = self.step_sizes[-MAX_HISTORY//2:]
        #         self.multipliers.append(self.lmbda)
        #         if len(self.multipliers) > MAX_HISTORY:
        #             self.multipliers = self.multipliers[-MAX_HISTORY//2:]

        #         # Update moving UB
        #         if self.best_upper_bound < float('inf'):
        #             self._moving_upper = 0.95 * self._moving_upper + 0.05 * self.best_upper_bound

        #         # Adaptive Polyak gamma
        #         gamma_iter = max(0.05, min(0.2, gamma_base * (1 - iter_num / max_iter)))

        #         if polyak_enabled and self._moving_upper < float('inf'):
        #             current_L = (lagrangian_bound if not math.isnan(lagrangian_bound) else self.best_lower_bound)
        #             gap = max(1e-6, self._moving_upper - current_L)

        #             knap_norm2 = max(1e-10, knapsack_subgradient ** 2)
        #             mu_norm2 = 0.0
        #             for vi, size_i in zip(cut_subgradients, cut_sizes):
        #                 g = vi / (1.0 + size_i)
        #                 mu_norm2 += g * g
        #             mu_norm2 = max(1e-10, mu_norm2)

        #             polyak_lambda_step = gamma_iter * gap / knap_norm2
        #             polyak_lambda_step = max(1e-8, min(polyak_lambda_step, self.step_size * 2))

        #             polyak_cut_step = gamma_mu * gap / mu_norm2
        #             polyak_cut_step = max(1e-8, min(polyak_cut_step, self.step_size * 2))

        #             # --- Update λ (momentum & projection) ---
        #             proposed_lambda = self.lmbda + polyak_lambda_step * knapsack_subgradient
        #             beta = self.momentum_beta
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))

        #             # --- Update μ_i (projected; increment cap per step) ---
        #             for cut_idx, (vi, size_i) in enumerate(zip(cut_subgradients, cut_sizes)):
        #                 g = vi / (1.0 + size_i)
        #                 current_mult = self.best_cut_multipliers.get(cut_idx, 0.0)
        #                 increment = polyak_cut_step * g
        #                 capped_increment = max(-mu_increment_cap, min(increment, mu_increment_cap))
        #                 proposed = current_mult + capped_increment
        #                 new_mult = max(0.0, min(proposed, 1e4))
        #                 self.best_cut_multipliers[cut_idx] = new_mult
        #         else:
        #             # --- DECAY MODE ---
        #             self.step_size *= self.p
        #             beta = self.momentum_beta
        #             proposed_lambda = self.lmbda + self.step_size * knapsack_subgradient
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))

        #             for cut_idx, (vi, size_i) in enumerate(zip(cut_subgradients, cut_sizes)):
        #                 g = vi / (1.0 + size_i)
        #                 current_mult = self.best_cut_multipliers.get(cut_idx, 0.0)
        #                 increment = self.step_size * g
        #                 capped_increment = max(-mu_increment_cap, min(increment, mu_increment_cap))
        #                 proposed = current_mult + capped_increment
        #                 new_mult = max(0.0, min(proposed, 1e4))
        #                 self.best_cut_multipliers[cut_idx] = new_mult

        #         # Optional cache/cleanup
        #         if iter_num % CLEANUP_INTERVAL == 0:
        #             if len(self.fractional_solutions) > 10:
        #                 self.fractional_solutions = self.fractional_solutions[-5:]
        #             # no depth-based cache tuning

        #     # -------- Final cover-cut generation with dedup (SAFE + NORMALIZED) --------
        #         if iter_num ==1:
        #             new_cuts = []
        #             if self.use_cover_cuts and self.last_mst_edges:
        #                 try:
        #                     # generate and normalize cuts
        #                     raw_cuts = self.generate_cover_cuts(self.last_mst_edges) or []
        #                     normalized_existing = {frozenset(c): r for c, r in self.best_cuts}
        #                     for cut, rhs in raw_cuts:
        #                         cut_norm = _norm_edge_set(cut)
        #                         if not cut_norm:
        #                             continue
        #                         rhs_int = int(rhs)
        #                         fz = frozenset(cut_norm)
        #                         if fz in normalized_existing:
        #                             # strengthen RHS if larger
        #                             if rhs_int > normalized_existing[fz]:
        #                                 # update best_cuts entry (keep μ as-is or initialize if missing)
        #                                 idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                                 self.best_cuts[idx] = (set(cut_norm), rhs_int)
        #                                 self.best_cut_multipliers.setdefault(idx, 0.0)
        #                                 self.best_cut_multipliers_for_best_bound.setdefault(idx, 0.0)
        #                                 normalized_existing[fz] = rhs_int
        #                         else:
        #                             # append brand-new cut
        #                             new_idx = len(self.best_cuts)
        #                             self.best_cuts.append((set(cut_norm), rhs_int))
        #                             self.best_cut_multipliers[new_idx] = 0.0
        #                             self.best_cut_multipliers_for_best_bound[new_idx] = 0.0
        #                             normalized_existing[fz] = rhs_int
        #                             # also return it for children
        #                             new_cuts.append((set(cut_norm), rhs_int))
        #                             if self.verbose:
        #                                 print(f"Added final cut: size={len(cut_norm)}, rhs={rhs_int}")
        #                 except Exception as e:
        #                     if self.verbose:
        #                         print(f"Error generating final cuts: {e}")

        #     if self.verbose:
        #         print(f"Solve completed: iterations={iter_num+1}, "
        #             f"lower={self.best_lower_bound:.2f}, upper={self.best_upper_bound:.2f}, "
        #             f"λ={self.lmbda:.6f}, cuts={len(self.best_cuts)}")
        #         if self.best_upper_bound < float('inf') and self.best_lower_bound > -float('inf'):
        #             print(f"Duality gap: {self.best_upper_bound - self.best_lower_bound:.4f}")
        #         print(f"Final step size: {self.step_size:.8f}")

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, new_cuts

       
        # else:  # Subgradient method with Polyak hybrid (robust: no early finish at root)
        #     import numpy as np

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     MAX_HISTORY = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10
        #     gamma_mu = 1.0
        #     mu_increment_cap = 5.0

        #     no_improvement_count = 0
        #     polyak_enabled = True
        #     stagnation_threshold = 15

        #     # Stronger guard at the root: never allow quick exits
        #     MIN_ITERS_BEFORE_CONV = 16 if getattr(self, "depth", 0) == 0 else 3

        #     # Initialize moving UB for Polyak gap control
        #     if not hasattr(self, "_moving_upper"):
        #         self._moving_upper = self.best_upper_bound if self.best_upper_bound < float("inf") else 1000.0

        #     # ---------- helpers ----------
        #     def _norm_edge(e):
        #         u, v = e
        #         return (u, v) if u <= v else (v, u)

        #     # Normalize fixed/forbidden once (keys must match self.edge_indices)
        #     F_in = {_norm_edge(e) for e in getattr(self, "fixed_edges", set())}
        #     F_out = {_norm_edge(e) for e in getattr(self, "excluded_edges", set())}

        #     # Ensure cut containers exist
        #     if not hasattr(self, "best_cut_multipliers"):
        #         self.best_cut_multipliers = {}
        #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers_for_best_bound = {}

        #     # -------- Build rhs_eff and early-prune impossible nodes (once per node) --------
        #     self._rhs_eff = {}
        #     for idx, (cut, rhs) in enumerate(self.best_cuts):
        #         rhs_eff = int(rhs) - len(cut & F_in)
        #         self._rhs_eff[idx] = rhs_eff
        #         if rhs_eff < 0:
        #             end_time = time()
        #             LagrangianMST.total_compute_time += end_time - start_time
        #             return float('inf'), self.best_upper_bound, []

        #     # -------- Precompute cut→edge-index arrays for FREE edges (once per node) --------
        #     edge_idx = self.edge_indices  # normalized (u,v) -> index
        #     cut_edge_idx = []
        #     cut_free_sizes = []
        #     for cut, _ in self.best_cuts:
        #         ids = [edge_idx[e] for e in cut if (e not in F_in and e not in F_out) and (e in edge_idx)]
        #         arr = np.fromiter(ids, dtype=np.int32) if ids else np.empty(0, dtype=np.int32)
        #         cut_edge_idx.append(arr)
        #         cut_free_sizes.append(max(1, arr.size))   # avoid div-by-zero

        #     self._cut_edge_idx = cut_edge_idx
        #     rhs_eff_vec = np.array([self._rhs_eff[i] for i in range(len(cut_edge_idx))], dtype=float)

        #     # Reset modified-weights cache for this node
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx), dtype=float)

        #     # Ensure histories exist
        #     if not hasattr(self, "subgradients"): self.subgradients = []
        #     if not hasattr(self, "step_sizes"):   self.step_sizes = []
        #     if not hasattr(self, "multipliers"):  self.multipliers = []
        #     if not hasattr(self, "primal_solutions"): self.primal_solutions = []
        #     if not hasattr(self, "fractional_solutions"): self.fractional_solutions = []

        #     # --- cuts to return to children (strongest versions only if we add any at the end) ---
        #     new_cuts = []

        #     # --- seed priced weights so iteration 0 is correct ---
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None
        #     self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0)

        #     for iter_num in range(max_iter):
        #         if iter_num % 10 == 0:
        #             if len(self.primal_solutions) > MAX_SOLUTIONS: self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #             if len(self.subgradients) > MAX_HISTORY:       self.subgradients = self.subgradients[-MAX_HISTORY:]
        #             if len(self.step_sizes) > MAX_HISTORY:         self.step_sizes = self.step_sizes[-MAX_HISTORY:]
        #             if len(self.multipliers) > MAX_HISTORY:        self.multipliers = self.multipliers[-MAX_HISTORY:]
        #             if len(self.fractional_solutions) > 10:        self.fractional_solutions = self.fractional_solutions[-5:]

        #         # -------- Step 1: MST (incremental w/ fallback) --------
        #         try:
        #             if prev_weights is not None and prev_mst_edges is not None:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #             else:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()

        #             if (math.isnan(mst_cost) or math.isinf(mst_cost) or
        #                 math.isnan(mst_length) or math.isinf(mst_length)):
        #                 if self.verbose:
        #                     print(f"Subgradient Iter {iter_num}: Invalid MST, fallback to full")
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Error in MST: {e}, fallback")
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         # *** normalize MST edges so keys match edge_indices ***
        #         self.last_mst_edges = [tuple(sorted((u, v))) for u, v in (mst_edges or [])]
        #         prev_mst_edges = self.last_mst_edges

        #         # price for next iter
        #         prev_weights = self.compute_modified_weights()

        #         # -------- Step 2: Dual bound & book-keeping --------
        #         is_feasible = (mst_length <= self.budget)
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         mu_vec = np.fromiter((self.best_cut_multipliers.get(i, 0.0) for i in range(len(cut_edge_idx))),
        #                             dtype=float, count=len(cut_edge_idx))
        #         cover_cut_penalty = float(mu_vec @ rhs_eff_vec)
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound) and not math.isinf(lagrangian_bound) and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #         else:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Invalid L={lagrangian_bound}")
        #             no_improvement_count += 1

        #         # -------- Step 3: Update best primal if feasible --------
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight) and not math.isinf(real_weight)
        #                     and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # -------- Step 4: Subgradients (λ and μ) --------
        #         knapsack_subgradient = mst_length - self.budget

        #         # build mask over FREE edges
        #         nE = len(self.edge_weights)
        #         mst_mask = np.zeros(nE, dtype=bool)
        #         for e in self.last_mst_edges:
        #             if e in F_in or e in F_out:
        #                 continue
        #             j = edge_idx.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         violations = np.array([
        #             (int(mst_mask[idxs].sum()) if idxs.size else 0) - self._rhs_eff[i]
        #             for i, idxs in enumerate(cut_edge_idx)
        #         ], dtype=float)
        #         cut_subgradients = violations.tolist()

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #         # stagnation on λ-subgradient
        #         if iter_num > 0 and abs(knapsack_subgradient - self.subgradients[-1]) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0

        #         self._record_subgradient(knapsack_subgradient)
        #         if len(self.subgradients) > MAX_HISTORY:
        #             self.subgradients = self.subgradients[-MAX_HISTORY//2:]

        #         # -------- Step 5: Conservative stopping (no small-gap / no small-norm) --------
        #         # Only after warmup allow stopping — and only on stagnation or tiny step-size
        #         if (iter_num >= MIN_ITERS_BEFORE_CONV) and (
        #             no_improvement_count > stagnation_threshold or
        #             self.step_size < 1e-8
        #         ):
        #             if self.verbose:
        #                 reason = ("no improvement" if no_improvement_count > stagnation_threshold else "step too small")
        #                 print(f"Stop at iter {iter_num} (Reason: {reason})")
        #             break

        #         # -------- Step 6: Hybrid Polyak + Decay updates (μ skips dead cuts) --------
        #         self.step_sizes.append(self.step_size)
        #         if len(self.step_sizes) > MAX_HISTORY:
        #             self.step_sizes = self.step_sizes[-MAX_HISTORY//2:]
        #         self.multipliers.append(self.lmbda)
        #         if len(self.multipliers) > MAX_HISTORY:
        #             self.multipliers = self.multipliers[-MAX_HISTORY//2:]

        #         # Update moving UB
        #         if self.best_upper_bound < float('inf'):
        #             self._moving_upper = 0.95 * self._moving_upper + 0.05 * self.best_upper_bound

        #         gamma_iter = max(0.05, min(0.2, gamma_base * (1 - iter_num / max_iter)))

        #         if polyak_enabled and self._moving_upper < float('inf'):
        #             current_L = (lagrangian_bound if not math.isnan(lagrangian_bound) else self.best_lower_bound)
        #             gap = max(1e-6, self._moving_upper - current_L)

        #             knap_norm2 = max(1e-10, knapsack_subgradient ** 2)
        #             mu_norm2 = 0.0
        #             for vi, size_i, idxs in zip(cut_subgradients, cut_free_sizes, cut_edge_idx):
        #                 if idxs.size == 0:  # skip dead cuts
        #                     continue
        #                 g = vi / (1.0 + size_i)
        #                 mu_norm2 += g * g
        #             mu_norm2 = max(1e-10, mu_norm2)

        #             polyak_lambda_step = gamma_iter * gap / knap_norm2
        #             polyak_lambda_step = max(1e-8, min(polyak_lambda_step, self.step_size * 2))

        #             polyak_cut_step = gamma_mu * gap / mu_norm2
        #             polyak_cut_step = max(1e-8, min(polyak_cut_step, self.step_size * 2))

        #             # λ update (momentum + projection)
        #             proposed_lambda = self.lmbda + polyak_lambda_step * knapsack_subgradient
        #             beta = self.momentum_beta
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))

        #             # μ updates (projected, capped, skip dead cuts)
        #             for cut_idx, (vi, size_i, idxs) in enumerate(zip(cut_subgradients, cut_free_sizes, cut_edge_idx)):
        #                 if idxs.size == 0:
        #                     continue
        #                 g = vi / (1.0 + size_i)
        #                 cur = self.best_cut_multipliers.get(cut_idx, 0.0)
        #                 inc = polyak_cut_step * g
        #                 inc = max(-mu_increment_cap, min(inc, mu_increment_cap))
        #                 self.best_cut_multipliers[cut_idx] = max(0.0, min(cur + inc, 1e4))
        #         else:
        #             # decay mode
        #             self.step_size *= self.p
        #             beta = self.momentum_beta
        #             proposed_lambda = self.lmbda + self.step_size * knapsack_subgradient
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))

        #             for cut_idx, (vi, size_i, idxs) in enumerate(zip(cut_subgradients, cut_free_sizes, cut_edge_idx)):
        #                 if idxs.size == 0:
        #                     continue
        #                 g = vi / (1.0 + size_i)
        #                 cur = self.best_cut_multipliers.get(cut_idx, 0.0)
        #                 inc = self.step_size * g
        #                 inc = max(-mu_increment_cap, min(inc, mu_increment_cap))
        #                 self.best_cut_multipliers[cut_idx] = max(0.0, min(cur + inc, 1e4))

        #         # Optional cleanup
        #         if iter_num % CLEANUP_INTERVAL == 0:
        #             if len(self.fractional_solutions) > 10:
        #                 self.fractional_solutions = self.fractional_solutions[-5:]

        #     # -------- Final one-shot separation if current MST violates --------
        #     # Keep this cheap: one pass, only if we have a violating current tree.
        #     # It strengthens the node for children without risking early exit here.
        #     if self.use_cover_cuts and self.last_mst_edges:
        #         try:
        #             # recompute the last MST length under original lengths
        #             # (self.compute_real_weight_length already does this; reuse if feasible flag was set)
        #             # We only try cuts when the current tree violates the budget.
        #             _, last_len = self.compute_real_weight_length()
        #             if last_len > self.budget + 1e-9:
        #                 new_cuts = self.generate_cover_cuts(self.last_mst_edges) or []
        #                 if new_cuts:
        #                     # Merge strongest versions into best_cuts and for children
        #                     existing = {frozenset(c): r for c, r in self.best_cuts}
        #                     child_map = {}
        #                     for cut, rhs in new_cuts:
        #                         fz = frozenset(cut)
        #                         rhs_int = int(rhs)
        #                         if fz in existing:
        #                             if rhs_int > existing[fz]:
        #                                 # strengthen in-place
        #                                 idx = next(i for i, (c, r) in enumerate(self.best_cuts) if frozenset(c) == fz)
        #                                 self.best_cuts[idx] = (cut, rhs_int)
        #                                 existing[fz] = rhs_int
        #                         else:
        #                             new_idx = len(self.best_cuts)
        #                             self.best_cuts.append((cut, rhs_int))
        #                             self.best_cut_multipliers[new_idx] = 0.0
        #                             self.best_cut_multipliers_for_best_bound[new_idx] = 0.0
        #                             existing[fz] = rhs_int
        #                         child_map[fz] = max(rhs_int, child_map.get(fz, 0))
        #                     # return strongest versions for children
        #                     new_cuts = [(set(k), int(v)) for k, v in ((set(fz), r) for fz, r in child_map.items())]
        #                 else:
        #                     new_cuts = []
        #             else:
        #                 new_cuts = []
        #         except Exception:
        #             new_cuts = []

        #     if self.verbose:
        #         print(f"Solve completed: iterations={iter_num+1}, "
        #             f"lower={self.best_lower_bound:.2f}, upper={self.best_upper_bound:.2f}, "
        #             f"λ={self.lmbda:.6f}, cuts={len(self.best_cuts)}")
        #         if self.best_upper_bound < float('inf') and self.best_lower_bound > -float('inf'):
        #             print(f"Duality gap: {self.best_upper_bound - self.best_lower_bound:.4f}")
        #         print(f"Final step size: {self.step_size:.8f}")

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, new_cuts

        # else:  # Subgradient method with Polyak hybrid (in-node separation; robust & index-safe)
        #     import math
        #     import numpy as np

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     MAX_HISTORY = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10
        #     gamma_mu = 0.30          # gentler μ updates
        #     mu_increment_cap = 5.0
        #     mu_value_cap = 1e3       # cap μ values

        #     # In-node separation policy
        #     node_cut_frequency = getattr(self, "node_cut_frequency", 10)  # try every N iters after warmup
        #     stagnation_threshold = 15
        #     big_violation_len = getattr(self, "big_violation_len", 0.0)   # optional (0 = off)

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Root guard
        #     MIN_ITERS_BEFORE_CONV = 16 if getattr(self, "depth", 0) == 0 else 3

        #     # Moving UB for Polyak gap control
        #     if not hasattr(self, "_moving_upper"):
        #         self._moving_upper = self.best_upper_bound if self.best_upper_bound < float("inf") else 1000.0

        #     # ---------- helpers ----------
        #     def _norm_edge(e):
        #         u, v = e
        #         return (u, v) if u <= v else (v, u)

        #     # Normalize fixed / excluded
        #     F_in  = {_norm_edge(e) for e in getattr(self, "fixed_edges", set())}
        #     F_out = {_norm_edge(e) for e in getattr(self, "excluded_edges", set())}

        #     # Ensure maps exist
        #     if not hasattr(self, "best_cut_multipliers"):
        #         self.best_cut_multipliers = {}
        #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers_for_best_bound = {}

        #     edge_idx = self.edge_indices  # normalized (u,v) -> idx

        #     def _rebuild_cut_structures(cuts, mu_map):
        #         """
        #         Compact cuts against (F_in, F_out), rebuild indices & rhs_eff, and remap μ to dense indices.
        #         Returns: compact_cuts, compact_mu, cut_edge_idx, cut_free_sizes, rhs_eff_vec
        #         """
        #         compact_cuts, cut_edge_idx, cut_free_sizes, rhs_eff = [], [], [], []
        #         # original lookup (for μ remap)
        #         original = {frozenset(c): (i, r) for i, (c, r) in enumerate(cuts)}
        #         for (cut, rhs_i) in cuts:
        #             cut_n = {tuple(sorted(e)) for e in cut if tuple(sorted(e)) in edge_idx}
        #             rhs_eff_i = int(rhs_i) - len(cut_n & F_in)
        #             if rhs_eff_i <= 0:
        #                 continue
        #             ids = [edge_idx[e] for e in cut_n if (e not in F_in) and (e not in F_out)]
        #             if not ids:
        #                 continue
        #             arr = np.fromiter(ids, dtype=np.int32)
        #             compact_cuts.append((cut_n, int(rhs_i)))
        #             cut_edge_idx.append(arr)
        #             cut_free_sizes.append(max(1, arr.size))
        #             rhs_eff.append(float(rhs_eff_i))

        #         # remap μ to dense indices
        #         compact_mu = {}
        #         if compact_cuts:
        #             for new_i, (cut_n, _) in enumerate(compact_cuts):
        #                 key = frozenset(cut_n)
        #                 if key in original:
        #                     old_i, _ = original[key]
        #                     compact_mu[new_i] = min(float(mu_map.get(old_i, 0.0)), mu_value_cap)
        #                 else:
        #                     compact_mu[new_i] = 0.0
        #         return (compact_cuts,
        #                 compact_mu,
        #                 cut_edge_idx,
        #                 cut_free_sizes,
        #                 np.array(rhs_eff, dtype=float))

        #     # Normalize any inherited cuts then compact
        #     self.best_cuts = [(set(_norm_edge(e) for e in cut), int(rhs)) for (cut, rhs) in (self.best_cuts or [])]
        #     (self.best_cuts,
        #     self.best_cut_multipliers,
        #     cut_edge_idx,
        #     cut_free_sizes,
        #     rhs_eff_vec) = _rebuild_cut_structures(self.best_cuts, self.best_cut_multipliers)


        #     self._cut_edge_idx = cut_edge_idx
        #     self._rhs_eff_vec  = rhs_eff_vec
        #     # invalidate any cached modified-weights that depend on μ / cuts
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx), dtype=float)
            
        #     # Track inactivity for μ drop-to-zero
        #     inactive_counts = getattr(self, "_inactive_counts", {})

        #     # For children: collect only brand-new cuts produced in this node
        #     new_cuts_for_children = []

        #     # seed weights
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None
        #     self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0)

        #     for iter_num in range(max_iter):
        #         if iter_num % 10 == 0:
        #             if len(self.primal_solutions) > MAX_SOLUTIONS: self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #             if len(self.subgradients) > MAX_HISTORY:       self.subgradients = self.subgradients[-MAX_HISTORY:]
        #             if len(self.step_sizes) > MAX_HISTORY:         self.step_sizes = self.step_sizes[-MAX_HISTORY:]
        #             if len(self.multipliers) > MAX_HISTORY:        self.multipliers = self.multipliers[-MAX_HISTORY:]
        #             if len(self.fractional_solutions) > 10:        self.fractional_solutions = self.fractional_solutions[-5:]

        #         # -------- MST (incremental + fallback) --------
        #         try:
        #             if prev_weights is not None and prev_mst_edges is not None:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #             else:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()

        #             if (math.isnan(mst_cost) or math.isinf(mst_cost) or
        #                 math.isnan(mst_length) or math.isinf(mst_length)):
        #                 if self.verbose:
        #                     print(f"Subgradient Iter {iter_num}: Invalid MST, fallback to full")
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Error in MST: {e}, fallback")
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         # normalize MST edges
        #         self.last_mst_edges = [tuple(sorted((u, v))) for u, v in (mst_edges or [])]
        #         prev_mst_edges = self.last_mst_edges
        #         prev_weights = self.compute_modified_weights()  # for next iter

        #         # -------- Dual bound & bookkeeping --------
        #         is_feasible = (mst_length <= self.budget)
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         mu_vec = (np.fromiter((self.best_cut_multipliers.get(i, 0.0) for i in range(len(self.best_cuts))),
        #                             dtype=float, count=len(self.best_cuts))
        #                 if self.best_cuts else np.array([], dtype=float))
        #         cover_cut_penalty = float(mu_vec @ rhs_eff_vec) if mu_vec.size else 0.0
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound) and not math.isinf(lagrangian_bound) and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #         else:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Invalid L={lagrangian_bound}")
        #             no_improvement_count += 1

        #         # -------- Primal UB update --------
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight) and not math.isinf(real_weight)
        #                     and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # -------- Subgradients (λ and μ) --------
        #         knapsack_subgradient = mst_length - self.budget
        #         if knapsack_subgradient > 1e4: knapsack_subgradient = 1e4
        #         elif knapsack_subgradient < -1e4: knapsack_subgradient = -1e4

        #         # FREE-edge mask
        #         nE = len(self.edge_weights)
        #         mst_mask = np.zeros(nE, dtype=bool)
        #         for e in self.last_mst_edges:
        #             if e in F_in or e in F_out:
        #                 continue
        #             j = edge_idx.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         # violations (index-safe)
        #         if self.best_cuts:
        #             violations = np.fromiter(
        #                 ((int(mst_mask[idxs].sum()) if idxs.size else 0) - rhs_eff
        #                 for idxs, rhs_eff in zip(cut_edge_idx, rhs_eff_vec)),
        #                 dtype=float,
        #                 count=len(cut_edge_idx)
        #             )
        #         else:
        #             violations = np.array([], dtype=float)

        #         # μ hygiene: drop dead/satisfied & track inactivity (index-safe)
        #         for (vi, idxs, rhs_eff_i), i in zip(zip(violations, cut_edge_idx, rhs_eff_vec),
        #                                             range(len(cut_edge_idx))):
        #             if (idxs.size == 0) or (rhs_eff_i <= 0.0):
        #                 self.best_cut_multipliers[i] = 0.0
        #                 inactive_counts[i] = 0
        #             else:
        #                 if vi <= 0.0:
        #                     inactive_counts[i] = inactive_counts.get(i, 0) + 1
        #                     if inactive_counts[i] >= 3:
        #                         self.best_cut_multipliers[i] = 0.0
        #                 else:
        #                     inactive_counts[i] = 0
        #         self._inactive_counts = inactive_counts

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #         # stagnation detection
        #         if iter_num > 0 and abs(knapsack_subgradient - self.subgradients[-1]) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0

        #         self._record_subgradient(knapsack_subgradient)
        #         if len(self.subgradients) > MAX_HISTORY:
        #             self.subgradients = self.subgradients[-MAX_HISTORY//2:]

        #         # -------- In-node separation (apply immediately) --------
        #         do_separate = (
        #             self.use_cover_cuts and
        #             (iter_num >= MIN_ITERS_BEFORE_CONV) and (
        #                 (node_cut_frequency > 0 and (iter_num % node_cut_frequency == 0)) or
        #                 (no_improvement_count >= max(3, stagnation_threshold // 2)) or
        #                 (big_violation_len > 0.0 and (mst_length - self.budget) >= big_violation_len)
        #             )
        #         )
        #         if do_separate:
        #             try:
        #                 cand_cuts = self.generate_cover_cuts(self.last_mst_edges) or []
        #             except Exception:
        #                 cand_cuts = []
        #             if cand_cuts:
        #                 # Score and keep a few strongest
        #                 scored = []
        #                 for cut, rhs in cand_cuts:
        #                     cut_n = {tuple(sorted(e)) for e in cut if tuple(sorted(e)) in edge_idx}
        #                     rhs_int = int(rhs)
        #                     rhs_eff_i = rhs_int - len(cut_n & F_in)
        #                     if rhs_eff_i <= 0:
        #                         continue
        #                     ids = [edge_idx[e] for e in cut_n if (e not in F_in and e not in F_out)]
        #                     if not ids:
        #                         continue
        #                     idxs = np.fromiter(ids, dtype=np.int32)
        #                     lhs = int(mst_mask[idxs].sum()) if idxs.size else 0
        #                     viol = max(0, lhs - rhs_eff_i)
        #                     score = viol / max(1, len(cut_n))
        #                     if score > 0:
        #                         scored.append((score, cut_n, rhs_int))
        #                 scored.sort(reverse=True, key=lambda t: t[0])
        #                 picked = [(cut_n, rhs_int) for s, cut_n, rhs_int in scored[:3]]

        #                 if picked:
        #                     # Merge, keep μ for existing, μ=0.0 for new
        #                     existing = {frozenset(c): (i, r) for i, (c, r) in enumerate(self.best_cuts)}
        #                     for cut_n, rhs_int in picked:
        #                         key = frozenset(cut_n)
        #                         if key in existing:
        #                             i_old, old_rhs = existing[key]
        #                             if rhs_int > old_rhs:
        #                                 self.best_cuts[i_old] = (cut_n, rhs_int)
        #                         else:
        #                             self.best_cuts.append((cut_n, rhs_int))
        #                             self.best_cut_multipliers[len(self.best_cuts)-1] = 0.0
        #                             new_cuts_for_children.append((set(cut_n), int(rhs_int)))

        #                     # Rebuild (index-safe)
        #                     (self.best_cuts,
        #                     self.best_cut_multipliers,
        #                     cut_edge_idx,
        #                     cut_free_sizes,
        #                     rhs_eff_vec) = _rebuild_cut_structures(self.best_cuts, self.best_cut_multipliers)

        #                     # Make sure pricing sees the fresh cut structure too
        #                     self._cut_edge_idx = cut_edge_idx
        #                     self._rhs_eff_vec = rhs_eff_vec
        #                     self._mw_cached = None
        #                     self._mw_lambda = None
        #                     self._mw_mu = np.zeros(len(cut_edge_idx), dtype=float)

        #                     if len(self.best_cuts) == 0:
        #                         cut_edge_idx = []
        #                         rhs_eff_vec = np.array([], dtype=float)

        #         # -------- Stopping --------
        #         if (iter_num >= MIN_ITERS_BEFORE_CONV) and (
        #             no_improvement_count > stagnation_threshold or
        #             self.step_size < 1e-8
        #         ):
        #             if self.verbose:
        #                 reason = ("no improvement" if no_improvement_count > stagnation_threshold else "step too small")
        #                 print(f"Stop at iter {iter_num} (Reason: {reason})")
        #             break

        #         # -------- Updates (Polyak / Decay), all index-safe via zip --------
        #         self.step_sizes.append(self.step_size)
        #         if len(self.step_sizes) > MAX_HISTORY:
        #             self.step_sizes = self.step_sizes[-MAX_HISTORY//2:]
        #         self.multipliers.append(self.lmbda)
        #         if len(self.multipliers) > MAX_HISTORY:
        #             self.multipliers = self.multipliers[-MAX_HISTORY//2:]

        #         if self.best_upper_bound < float('inf'):
        #             self._moving_upper = 0.95 * self._moving_upper + 0.05 * self.best_upper_bound
        #         gamma_iter = max(0.05, min(0.2, gamma_base * (1 - iter_num / max_iter)))

        #         # norms
        #         mu_norm2 = 0.0
        #         for vi, size_i, idxs in zip(violations, cut_free_sizes, cut_edge_idx):
        #             if idxs.size == 0:
        #                 continue
        #             g = vi / (1.0 + size_i)
        #             mu_norm2 += g * g
        #         mu_norm2 = max(1e-10, mu_norm2)
        #         knap_norm2 = max(1e-10, knapsack_subgradient ** 2)

        #         if polyak_enabled and self._moving_upper < float('inf'):
        #             current_L = (lagrangian_bound if not math.isnan(lagrangian_bound) else self.best_lower_bound)
        #             gap = max(1e-6, self._moving_upper - current_L)

        #             polyak_lambda_step = gamma_iter * gap / knap_norm2
        #             polyak_lambda_step = max(1e-8, min(polyak_lambda_step, self.step_size * 2))

        #             polyak_cut_step = gamma_mu * gap / mu_norm2
        #             polyak_cut_step = max(1e-8, min(polyak_cut_step, self.step_size * 2))

        #             # λ update
        #             proposed_lambda = self.lmbda + polyak_lambda_step * knapsack_subgradient
        #             beta = self.momentum_beta
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))

        #             # μ update
        #             for i, (vi, size_i, idxs) in enumerate(zip(violations, cut_free_sizes, cut_edge_idx)):
        #                 if idxs.size == 0:
        #                     continue
        #                 g = vi / (1.0 + size_i)
        #                 cur = self.best_cut_multipliers.get(i, 0.0)
        #                 inc = polyak_cut_step * g
        #                 if inc > mu_increment_cap: inc = mu_increment_cap
        #                 elif inc < -mu_increment_cap: inc = -mu_increment_cap
        #                 new_mu = cur + inc
        #                 if new_mu < 0.0: new_mu = 0.0
        #                 if new_mu > mu_value_cap: new_mu = mu_value_cap
        #                 self.best_cut_multipliers[i] = new_mu
        #         else:
        #             # decay mode
        #             self.step_size *= self.p
        #             beta = self.momentum_beta
        #             proposed_lambda = self.lmbda + self.step_size * knapsack_subgradient
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))
        #             for i, (vi, size_i, idxs) in enumerate(zip(violations, cut_free_sizes, cut_edge_idx)):
        #                 if idxs.size == 0:
        #                     continue
        #                 g = vi / (1.0 + size_i)
        #                 cur = self.best_cut_multipliers.get(i, 0.0)
        #                 inc = self.step_size * g
        #                 if inc > mu_increment_cap: inc = mu_increment_cap
        #                 elif inc < -mu_increment_cap: inc = -mu_increment_cap
        #                 new_mu = cur + inc
        #                 if new_mu < 0.0: new_mu = 0.0
        #                 if new_mu > mu_value_cap: new_mu = mu_value_cap
        #                 self.best_cut_multipliers[i] = new_mu

        #         # Optional cleanup
        #         if iter_num % CLEANUP_INTERVAL == 0:
        #             if len(self.fractional_solutions) > 10:
        #                 self.fractional_solutions = self.fractional_solutions[-5:]

        #     # No final separation; children get only cuts added during this node
        #     if self.verbose:
        #         print(f"Solve completed: iterations={iter_num+1}, "
        #             f"lower={self.best_lower_bound:.2f}, upper={self.best_upper_bound:.2f}, "
        #             f"λ={self.lmbda:.6f}, cuts={len(self.best_cuts)}")
        #         if self.best_upper_bound < float('inf') and self.best_lower_bound > -float('inf'):
        #             print(f"Duality gap: {self.best_upper_bound - self.best_lower_bound:.4f}")
        #         print(f"Final step size: {self.step_size:.8f}")

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, new_cuts_for_children


        # else:  # Subgradient method with Polyak hybrid (in-node separation; robust & index-safe)
        #     import math
        #     import numpy as np

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     MAX_HISTORY = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10
        #     gamma_mu = 0.30          # gentler μ updates
        #     mu_increment_cap = 5.0
        #     mu_value_cap = 1e3       # cap μ values

        #     # In-node separation policy
        #     node_cut_frequency = getattr(self, "node_cut_frequency", 10)  # try every N iters after warmup
        #     stagnation_threshold = 15
        #     big_violation_len = getattr(self, "big_violation_len", 0.0)   # optional (0 = off)

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Root guard
        #     MIN_ITERS_BEFORE_CONV = 16 if getattr(self, "depth", 0) == 0 else 3

        #     # Moving UB for Polyak gap control  (kept exactly as you had it)
        #     if not hasattr(self, "_moving_upper"):
        #         self._moving_upper = self.best_upper_bound if self.best_upper_bound < float("inf") else 1000.0

        #     # ---------- helpers ----------
        #     def _norm_edge(e):
        #         u, v = e
        #         return (u, v) if u <= v else (v, u)

        #     # Normalize fixed / excluded
        #     F_in  = {_norm_edge(e) for e in getattr(self, "fixed_edges", set())}
        #     F_out = {_norm_edge(e) for e in getattr(self, "excluded_edges", set())}

        #     # Ensure maps exist
        #     if not hasattr(self, "best_cut_multipliers"):
        #         self.best_cut_multipliers = {}
        #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers_for_best_bound = {}

        #     edge_idx = self.edge_indices  # normalized (u,v) -> idx

        #     def _rebuild_cut_structures(cuts, mu_map):
        #         """
        #         Compact cuts against (F_in, F_out), rebuild indices & rhs_eff, and remap μ to dense indices.
        #         Returns: compact_cuts, compact_mu, cut_edge_idx, cut_free_sizes, rhs_eff_vec
        #         """
        #         compact_cuts, cut_edge_idx, cut_free_sizes, rhs_eff = [], [], [], []
        #         # original lookup (for μ remap)
        #         original = {frozenset(c): (i, r) for i, (c, r) in enumerate(cuts)}
        #         for (cut, rhs_i) in cuts:
        #             cut_n = {tuple(sorted(e)) for e in cut if tuple(sorted(e)) in edge_idx}
        #             rhs_eff_i = int(rhs_i) - len(cut_n & F_in)
        #             if rhs_eff_i <= 0:
        #                 continue
        #             ids = [edge_idx[e] for e in cut_n if (e not in F_in) and (e not in F_out)]
        #             if not ids:
        #                 continue
        #             arr = np.fromiter(ids, dtype=np.int32)
        #             compact_cuts.append((cut_n, int(rhs_i)))
        #             cut_edge_idx.append(arr)
        #             cut_free_sizes.append(max(1, arr.size))
        #             rhs_eff.append(float(rhs_eff_i))

        #         # remap μ to dense indices
        #         compact_mu = {}
        #         if compact_cuts:
        #             for new_i, (cut_n, _) in enumerate(compact_cuts):
        #                 key = frozenset(cut_n)
        #                 if key in original:
        #                     old_i, _ = original[key]
        #                     compact_mu[new_i] = min(float(mu_map.get(old_i, 0.0)), mu_value_cap)
        #                 else:
        #                     compact_mu[new_i] = 0.0
        #         return (compact_cuts,
        #                 compact_mu,
        #                 cut_edge_idx,
        #                 cut_free_sizes,
        #                 np.array(rhs_eff, dtype=float))

        #     # Normalize any inherited cuts then compact
        #     self.best_cuts = [(set(_norm_edge(e) for e in cut), int(rhs)) for (cut, rhs) in (self.best_cuts or [])]
        #     (self.best_cuts,
        #     self.best_cut_multipliers,
        #     cut_edge_idx,
        #     cut_free_sizes,
        #     rhs_eff_vec) = _rebuild_cut_structures(self.best_cuts, self.best_cut_multipliers)

        #     self._cut_edge_idx = cut_edge_idx
        #     self._rhs_eff_vec  = rhs_eff_vec
        #     # invalidate any cached modified-weights that depend on μ / cuts
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx), dtype=float)

        #     # Track inactivity for μ drop-to-zero
        #     inactive_counts = getattr(self, "_inactive_counts", {})

        #     # NEW: separation stabilization flags (no effect if cuts are OFF)
        #     _just_separated = False
        #     _just_separated_cooldown = 0

        #     # For children: collect only brand-new cuts produced in this node
        #     new_cuts_for_children = []

        #     # seed weights
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None
        #     self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0)

        #     for iter_num in range(max_iter):
        #         if iter_num % 10 == 0:
        #             if len(self.primal_solutions) > MAX_SOLUTIONS: self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #             if len(self.subgradients) > MAX_HISTORY:       self.subgradients = self.subgradients[-MAX_HISTORY:]
        #             if len(self.step_sizes) > MAX_HISTORY:         self.step_sizes = self.step_sizes[-MAX_HISTORY:]
        #             if len(self.multipliers) > MAX_HISTORY:        self.multipliers = self.multipliers[-MAX_HISTORY:]
        #             if len(self.fractional_solutions) > 10:        self.fractional_solutions = self.fractional_solutions[-5:]

        #         # -------- MST (incremental + fallback) --------
        #         try:
        #             if prev_weights is not None and prev_mst_edges is not None:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #             else:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()

        #             if (math.isnan(mst_cost) or math.isinf(mst_cost) or
        #                 math.isnan(mst_length) or math.isinf(mst_length)):
        #                 if self.verbose:
        #                     print(f"Subgradient Iter {iter_num}: Invalid MST, fallback to full")
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Error in MST: {e}, fallback")
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         # normalize MST edges
        #         self.last_mst_edges = [tuple(sorted((u, v))) for u, v in (mst_edges or [])]
        #         prev_mst_edges = self.last_mst_edges
        #         prev_weights = self.compute_modified_weights()  # for next iter

        #         # -------- Dual bound & bookkeeping --------
        #         is_feasible = (mst_length <= self.budget)
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         mu_vec = (np.fromiter((self.best_cut_multipliers.get(i, 0.0) for i in range(len(self.best_cuts))),
        #                             dtype=float, count=len(self.best_cuts))
        #                 if self.best_cuts else np.array([], dtype=float))
        #         cover_cut_penalty = float(mu_vec @ rhs_eff_vec) if mu_vec.size else 0.0
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound) and not math.isinf(lagrangian_bound) and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #         else:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Invalid L={lagrangian_bound}")
        #             no_improvement_count += 1

        #         # -------- Primal UB update --------
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight) and not math.isinf(real_weight)
        #                     and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # -------- Subgradients (λ and μ) --------
        #         knapsack_subgradient = mst_length - self.budget
        #         if knapsack_subgradient > 1e4: knapsack_subgradient = 1e4
        #         elif knapsack_subgradient < -1e4: knapsack_subgradient = -1e4

        #         # FREE-edge mask
        #         nE = len(self.edge_weights)
        #         mst_mask = np.zeros(nE, dtype=bool)
        #         for e in self.last_mst_edges:
        #             if e in F_in or e in F_out:
        #                 continue
        #             j = edge_idx.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         # violations (index-safe)
        #         if self.best_cuts:
        #             violations = np.fromiter(
        #                 ((int(mst_mask[idxs].sum()) if idxs.size else 0) - rhs_eff
        #                 for idxs, rhs_eff in zip(cut_edge_idx, rhs_eff_vec)),
        #                 dtype=float,
        #                 count=len(cut_edge_idx)
        #             )
        #         else:
        #             violations = np.array([], dtype=float)

        #         # μ hygiene: drop dead/satisfied & track inactivity (index-safe)
        #         for (vi, idxs, rhs_eff_i), i in zip(zip(violations, cut_edge_idx, rhs_eff_vec),
        #                                             range(len(cut_edge_idx))):
        #             if (idxs.size == 0) or (rhs_eff_i <= 0.0):
        #                 self.best_cut_multipliers[i] = 0.0
        #                 inactive_counts[i] = 0
        #             else:
        #                 if vi <= 0.0:
        #                     inactive_counts[i] = inactive_counts.get(i, 0) + 1
        #                     if inactive_counts[i] >= 3:
        #                         self.best_cut_multipliers[i] = 0.0
        #                 else:
        #                     inactive_counts[i] = 0
        #         self._inactive_counts = inactive_counts

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #         # stagnation detection
        #         if iter_num > 0 and abs(knapsack_subgradient - self.subgradients[-1]) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0

        #         self._record_subgradient(knapsack_subgradient)
        #         if len(self.subgradients) > MAX_HISTORY:
        #             self.subgradients = self.subgradients[-MAX_HISTORY//2:]

        #         # -------- In-node separation (apply immediately) --------
        #         do_separate = (
        #             self.use_cover_cuts and
        #             (iter_num >= MIN_ITERS_BEFORE_CONV) and (
        #                 (node_cut_frequency > 0 and (iter_num % node_cut_frequency == 0)) or
        #                 (no_improvement_count >= max(3, stagnation_threshold // 2)) or
        #                 (big_violation_len > 0.0 and (mst_length - self.budget) >= big_violation_len)
        #             )
        #         )
        #         if do_separate:
        #             try:
        #                 cand_cuts = self.generate_cover_cuts(self.last_mst_edges) or []
        #             except Exception:
        #                 cand_cuts = []
        #             if cand_cuts:
        #                 # Score and keep a few strongest (kept exactly as you had it)
        #                 scored = []
        #                 for cut, rhs in cand_cuts:
        #                     cut_n = {tuple(sorted(e)) for e in cut if tuple(sorted(e)) in edge_idx}
        #                     rhs_int = int(rhs)
        #                     rhs_eff_i = rhs_int - len(cut_n & F_in)
        #                     if rhs_eff_i <= 0:
        #                         continue
        #                     ids = [edge_idx[e] for e in cut_n if (e not in F_in and e not in F_out)]
        #                     if not ids:
        #                         continue
        #                     idxs = np.fromiter(ids, dtype=np.int32)
        #                     lhs = int(mst_mask[idxs].sum()) if idxs.size else 0
        #                     viol = max(0, lhs - rhs_eff_i)
        #                     score = viol / max(1, len(cut_n))
        #                     if score > 0:
        #                         scored.append((score, cut_n, rhs_int))
        #                 scored.sort(reverse=True, key=lambda t: t[0])
        #                 picked = [(cut_n, rhs_int) for s, cut_n, rhs_int in scored[:3]]

        #                 if picked:
        #                     # Merge, keep μ for existing, μ=0.0 for new
        #                     existing = {frozenset(c): (i, r) for i, (c, r) in enumerate(self.best_cuts)}
        #                     any_new = False
        #                     for cut_n, rhs_int in picked:
        #                         key = frozenset(cut_n)
        #                         if key in existing:
        #                             i_old, old_rhs = existing[key]
        #                             if rhs_int > old_rhs:
        #                                 self.best_cuts[i_old] = (cut_n, rhs_int)
        #                         else:
        #                             self.best_cuts.append((cut_n, rhs_int))
        #                             self.best_cut_multipliers[len(self.best_cuts)-1] = 0.0
        #                             new_cuts_for_children.append((set(cut_n), int(rhs_int)))
        #                             any_new = True

        #                     # Rebuild (index-safe)
        #                     (self.best_cuts,
        #                     self.best_cut_multipliers,
        #                     cut_edge_idx,
        #                     cut_free_sizes,
        #                     rhs_eff_vec) = _rebuild_cut_structures(self.best_cuts, self.best_cut_multipliers)

        #                     # Keep pricing caches aligned
        #                     self._cut_edge_idx = cut_edge_idx
        #                     self._rhs_eff_vec  = rhs_eff_vec
        #                     self._mw_cached = None
        #                     self._mw_lambda = None
        #                     self._mw_mu = np.zeros(len(cut_edge_idx), dtype=float)

        #                     if len(self.best_cuts) == 0:
        #                         cut_edge_idx = []
        #                         rhs_eff_vec = np.array([], dtype=float)

        #                     # >>> NEW: immediate reprice & μ cooldown (only when cuts were added) <<<
        #                     if any_new:
        #                         try:
        #                             mst_cost, mst_length, mst_edges = self.compute_mst()
        #                         except Exception:
        #                             mst_cost, mst_length, mst_edges = self.compute_mst()
        #                         self.last_mst_edges = [tuple(sorted((u, v))) for u, v in (mst_edges or [])]
        #                         prev_mst_edges = self.last_mst_edges
        #                         prev_weights = self.compute_modified_weights()
        #                         _just_separated = True
        #                         _just_separated_cooldown = 2

        #         # -------- Stopping --------
        #         if (iter_num >= MIN_ITERS_BEFORE_CONV) and (
        #             no_improvement_count > stagnation_threshold or
        #             self.step_size < 1e-8
        #         ):
        #             if self.verbose:
        #                 reason = ("no improvement" if no_improvement_count > stagnation_threshold else "step too small")
        #                 print(f"Stop at iter {iter_num} (Reason: {reason})")
        #             break

        #         # -------- Updates (Polyak / Decay), all index-safe via zip --------
        #         self.step_sizes.append(self.step_size)
        #         if len(self.step_sizes) > MAX_HISTORY:
        #             self.step_sizes = self.step_sizes[-MAX_HISTORY//2:]
        #         self.multipliers.append(self.lmbda)
        #         if len(self.multipliers) > MAX_HISTORY:
        #             self.multipliers = self.multipliers[-MAX_HISTORY//2:]

        #         if self.best_upper_bound < float('inf'):
        #             self._moving_upper = 0.95 * self._moving_upper + 0.05 * self.best_upper_bound
        #         gamma_iter = max(0.05, min(0.2, gamma_base * (1 - iter_num / max_iter)))

        #         # norms
        #         mu_norm2 = 0.0
        #         for vi, size_i, idxs in zip(violations, cut_free_sizes, cut_edge_idx):
        #             if idxs.size == 0:
        #                 continue
        #             g = vi / (1.0 + size_i)
        #             mu_norm2 += g * g
        #         mu_norm2 = max(1e-10, mu_norm2)
        #         knap_norm2 = max(1e-10, knapsack_subgradient ** 2)

        #         if polyak_enabled and self._moving_upper < float('inf'):
        #             current_L = (lagrangian_bound if not math.isnan(lagrangian_bound) else self.best_lower_bound)
        #             gap = max(1e-6, self._moving_upper - current_L)

        #             polyak_lambda_step = gamma_iter * gap / knap_norm2
        #             polyak_lambda_step = max(1e-8, min(polyak_lambda_step, self.step_size * 2))

        #             polyak_cut_step = gamma_mu * gap / mu_norm2
        #             polyak_cut_step = max(1e-8, min(polyak_cut_step, self.step_size * 2))

        #             # λ update (unchanged)
        #             proposed_lambda = self.lmbda + polyak_lambda_step * knapsack_subgradient
        #             beta = self.momentum_beta
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))

        #             # μ update (NEW: cooldown active only after a separation)
        #             if _just_separated_cooldown <= 0:
        #                 for i, (vi, size_i, idxs) in enumerate(zip(violations, cut_free_sizes, cut_edge_idx)):
        #                     if idxs.size == 0:
        #                         continue
        #                     g = vi / (1.0 + size_i)   # keep your normalization
        #                     cur = self.best_cut_multipliers.get(i, 0.0)
        #                     inc = polyak_cut_step * g
        #                     if inc > mu_increment_cap: inc = mu_increment_cap
        #                     elif inc < -mu_increment_cap: inc = -mu_increment_cap
        #                     new_mu = cur + inc
        #                     if new_mu < 0.0: new_mu = 0.0
        #                     if new_mu > mu_value_cap: new_mu = mu_value_cap
        #                     self.best_cut_multipliers[i] = new_mu
        #             else:
        #                 _just_separated_cooldown -= 1
        #         else:
        #             # decay mode
        #             self.step_size *= self.p
        #             beta = self.momentum_beta
        #             proposed_lambda = self.lmbda + self.step_size * knapsack_subgradient
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))

        #             # μ update (NEW: cooldown active only after a separation)
        #             if _just_separated_cooldown <= 0:
        #                 for i, (vi, size_i, idxs) in enumerate(zip(violations, cut_free_sizes, cut_edge_idx)):
        #                     if idxs.size == 0:
        #                         continue
        #                     g = vi / (1.0 + size_i)
        #                     cur = self.best_cut_multipliers.get(i, 0.0)
        #                     inc = self.step_size * g
        #                     if inc > mu_increment_cap: inc = mu_increment_cap
        #                     elif inc < -mu_increment_cap: inc = -mu_increment_cap
        #                     new_mu = cur + inc
        #                     if new_mu < 0.0: new_mu = 0.0
        #                     if new_mu > mu_value_cap: new_mu = mu_value_cap
        #                     self.best_cut_multipliers[i] = new_mu
        #             else:
        #                 _just_separated_cooldown -= 1

        #         # Optional cleanup
        #         if iter_num % CLEANUP_INTERVAL == 0:
        #             if len(self.fractional_solutions) > 10:
        #                 self.fractional_solutions = self.fractional_solutions[-5:]

        #     # No final separation; children get only cuts added during this node
        #     if self.verbose:
        #         print(f"Solve completed: iterations={iter_num+1}, "
        #             f"lower={self.best_lower_bound:.2f}, upper={self.best_upper_bound:.2f}, "
        #             f"λ={self.lmbda:.6f}, cuts={len(self.best_cuts)}")
        #         if self.best_upper_bound < float('inf') and self.best_lower_bound > -float('inf'):
        #             print(f"Duality gap: {self.best_upper_bound - self.best_lower_bound:.4f}")
        #         print(f"Final step size: {self.step_size:.8f}")

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, new_cuts_for_children
    
        
        # else:  # Subgradient method with Polyak hybrid (in-node separation; robust & index-safe)
        #     import math
        #     import numpy as np

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     MAX_HISTORY = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10
        #     gamma_mu = 0.30          # gentler μ updates
        #     mu_increment_cap = 5.0
        #     mu_value_cap = 1e3       # cap μ values

        #     # In-node separation policy
        #     node_cut_frequency = getattr(self, "node_cut_frequency", 10)  # try every N iters after warmup
        #     stagnation_threshold = 15
        #     big_violation_len = getattr(self, "big_violation_len", 0.0)   # optional (0 = off)

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Root guard
        #     MIN_ITERS_BEFORE_CONV = 16 if getattr(self, "depth", 0) == 0 else 3

        #     # Moving UB for Polyak gap control (keep your original init)
        #     if not hasattr(self, "_moving_upper"):
        #         self._moving_upper = self.best_upper_bound if self.best_upper_bound < float("inf") else 1000.0

        #     # ---------- helpers ----------
        #     def _norm_edge(e):
        #         u, v = e
        #         return (u, v) if u <= v else (v, u)

        #     # Normalize fixed / excluded
        #     F_in  = {_norm_edge(e) for e in getattr(self, "fixed_edges", set())}
        #     F_out = {_norm_edge(e) for e in getattr(self, "excluded_edges", set())}

        #     # Ensure maps exist
        #     if not hasattr(self, "best_cut_multipliers"):
        #         self.best_cut_multipliers = {}
        #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers_for_best_bound = {}

        #     edge_idx = self.edge_indices  # normalized (u,v) -> idx

        #     def _rebuild_cut_structures(cuts, mu_map):
        #         """
        #         Compact cuts against (F_in, F_out), rebuild indices & rhs_eff, and remap μ to dense indices.
        #         Returns: compact_cuts, compact_mu, cut_edge_idx, cut_free_sizes, rhs_eff_vec
        #         """
        #         compact_cuts, cut_edge_idx, cut_free_sizes, rhs_eff = [], [], [], []
        #         # original lookup (for μ remap)
        #         original = {frozenset(c): (i, r) for i, (c, r) in enumerate(cuts)}
        #         for (cut, rhs_i) in cuts:
        #             cut_n = {tuple(sorted(e)) for e in cut if tuple(sorted(e)) in edge_idx}
        #             rhs_eff_i = int(rhs_i) - len(cut_n & F_in)
        #             if rhs_eff_i <= 0:
        #                 continue
        #             ids = [edge_idx[e] for e in cut_n if (e not in F_in) and (e not in F_out)]
        #             if not ids:
        #                 continue
        #             arr = np.fromiter(ids, dtype=np.int32)
        #             compact_cuts.append((cut_n, int(rhs_i)))
        #             cut_edge_idx.append(arr)
        #             cut_free_sizes.append(max(1, arr.size))
        #             rhs_eff.append(float(rhs_eff_i))

        #         # remap μ to dense indices
        #         compact_mu = {}
        #         if compact_cuts:
        #             for new_i, (cut_n, _) in enumerate(compact_cuts):
        #                 key = frozenset(cut_n)
        #                 if key in original:
        #                     old_i, _ = original[key]
        #                     compact_mu[new_i] = min(float(mu_map.get(old_i, 0.0)), mu_value_cap)
        #                 else:
        #                     compact_mu[new_i] = 0.0
        #         return (compact_cuts,
        #                 compact_mu,
        #                 cut_edge_idx,
        #                 cut_free_sizes,
        #                 np.array(rhs_eff, dtype=float))

        #     # Normalize any inherited cuts then compact
        #     self.best_cuts = [(set(_norm_edge(e) for e in cut), int(rhs)) for (cut, rhs) in (self.best_cuts or [])]
        #     (self.best_cuts,
        #     self.best_cut_multipliers,
        #     cut_edge_idx,
        #     cut_free_sizes,
        #     rhs_eff_vec) = _rebuild_cut_structures(self.best_cuts, self.best_cut_multipliers)

        #     self._cut_edge_idx = cut_edge_idx
        #     self._rhs_eff_vec  = rhs_eff_vec
        #     # invalidate any cached modified-weights that depend on μ / cuts
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx), dtype=float)

        #     # Track inactivity for μ drop-to-zero
        #     inactive_counts = getattr(self, "_inactive_counts", {})

        #     # NEW: separation stabilization flags (no effect if cuts are OFF)
        #     _just_separated = False
        #     _just_separated_cooldown = 0
        #     _separation_freeze = 0   # how many iterations to skip separation after we add cuts

        #     # For children: collect only brand-new cuts produced in this node
        #     new_cuts_for_children = []

        #     # seed weights
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None
        #     self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0)

        #     for iter_num in range(max_iter):
        #         if iter_num % 10 == 0:
        #             if len(self.primal_solutions) > MAX_SOLUTIONS: self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #             if len(self.subgradients) > MAX_HISTORY:       self.subgradients = self.subgradients[-MAX_HISTORY:]
        #             if len(self.step_sizes) > MAX_HISTORY:         self.step_sizes = self.step_sizes[-MAX_HISTORY:]
        #             if len(self.multipliers) > MAX_HISTORY:        self.multipliers = self.multipliers[-MAX_HISTORY:]
        #             if len(self.fractional_solutions) > 10:        self.fractional_solutions = self.fractional_solutions[-5:]

        #         # -------- MST (incremental + fallback) --------
        #         try:
        #             if prev_weights is not None and prev_mst_edges is not None:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #             else:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()

        #             if (math.isnan(mst_cost) or math.isinf(mst_cost) or
        #                 math.isnan(mst_length) or math.isinf(mst_length)):
        #                 if self.verbose:
        #                     print(f"Subgradient Iter {iter_num}: Invalid MST, fallback to full")
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Error in MST: {e}, fallback")
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         # normalize MST edges
        #         self.last_mst_edges = [tuple(sorted((u, v))) for u, v in (mst_edges or [])]
        #         prev_mst_edges = self.last_mst_edges
        #         prev_weights = self.compute_modified_weights()  # for next iter

        #         # -------- Dual bound & bookkeeping --------
        #         is_feasible = (mst_length <= self.budget)
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         mu_vec = (np.fromiter((self.best_cut_multipliers.get(i, 0.0) for i in range(len(self.best_cuts))),
        #                             dtype=float, count=len(self.best_cuts))
        #                 if self.best_cuts else np.array([], dtype=float))
        #         cover_cut_penalty = float(mu_vec @ rhs_eff_vec) if mu_vec.size else 0.0
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound) and not math.isinf(lagrangian_bound) and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #         else:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Invalid L={lagrangian_bound}")
        #             no_improvement_count += 1

        #         # -------- Primal UB update --------
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight) and not math.isinf(real_weight)
        #                     and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # -------- Subgradients (λ and μ) --------
        #         knapsack_subgradient = mst_length - self.budget
        #         if knapsack_subgradient > 1e4: knapsack_subgradient = 1e4
        #         elif knapsack_subgradient < -1e4: knapsack_subgradient = -1e4

        #         # FREE-edge mask
        #         nE = len(self.edge_weights)
        #         mst_mask = np.zeros(nE, dtype=bool)
        #         for e in self.last_mst_edges:
        #             if e in F_in or e in F_out:
        #                 continue
        #             j = edge_idx.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         # violations (index-safe)
        #         if self.best_cuts:
        #             violations = np.fromiter(
        #                 ((int(mst_mask[idxs].sum()) if idxs.size else 0) - rhs_eff
        #                 for idxs, rhs_eff in zip(cut_edge_idx, rhs_eff_vec)),
        #                 dtype=float,
        #                 count=len(cut_edge_idx)
        #             )
        #         else:
        #             violations = np.array([], dtype=float)

        #         # μ hygiene: drop dead/satisfied & track inactivity (index-safe)
        #         for (vi, idxs, rhs_eff_i), i in zip(zip(violations, cut_edge_idx, rhs_eff_vec),
        #                                             range(len(cut_edge_idx))):
        #             if (idxs.size == 0) or (rhs_eff_i <= 0.0):
        #                 self.best_cut_multipliers[i] = 0.0
        #                 inactive_counts[i] = 0
        #             else:
        #                 if vi <= 0.0:
        #                     inactive_counts[i] = inactive_counts.get(i, 0) + 1
        #                     if inactive_counts[i] >= 3:
        #                         self.best_cut_multipliers[i] = 0.0
        #                 else:
        #                     inactive_counts[i] = 0
        #         self._inactive_counts = inactive_counts

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #         # stagnation detection
        #         if iter_num > 0 and abs(knapsack_subgradient - self.subgradients[-1]) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0

        #         self._record_subgradient(knapsack_subgradient)
        #         if len(self.subgradients) > MAX_HISTORY:
        #             self.subgradients = self.subgradients[-MAX_HISTORY//2:]

        #         # -------- In-node separation (apply immediately) --------
        #         do_separate = (
        #             self.use_cover_cuts
        #             and (iter_num >= MIN_ITERS_BEFORE_CONV)
        #             and (_separation_freeze <= 0)  # freeze window after adding cuts
        #             and (
        #                 (mst_length > self.budget)  # only if current MST violates budget
        #                 or (no_improvement_count >= max(3, stagnation_threshold // 2))
        #                 or (big_violation_len > 0.0 and (mst_length - self.budget) >= big_violation_len)
        #             )
        #             and (
        #                 (node_cut_frequency > 0 and (iter_num % node_cut_frequency == 0))
        #                 or (no_improvement_count >= max(3, stagnation_threshold // 2))
        #                 or (big_violation_len > 0.0 and (mst_length - self.budget) >= big_violation_len)
        #             )
        #         )
        #         if do_separate:
        #             try:
        #                 cand_cuts = self.generate_cover_cuts(self.last_mst_edges) or []
        #             except Exception:
        #                 cand_cuts = []
        #             if cand_cuts:
        #                 # Score and keep a few strongest (require viol>=1; score by viol/sqrt(|S|))
        #                 scored = []
        #                 for cut, rhs in cand_cuts:
        #                     cut_n = {tuple(sorted(e)) for e in cut if tuple(sorted(e)) in edge_idx}
        #                     rhs_int = int(rhs)
        #                     rhs_eff_i = rhs_int - len(cut_n & F_in)
        #                     if rhs_eff_i <= 0:
        #                         continue
        #                     ids = [edge_idx[e] for e in cut_n if (e not in F_in and e not in F_out)]
        #                     if not ids:
        #                         continue
        #                     idxs = np.fromiter(ids, dtype=np.int32)
        #                     lhs = int(mst_mask[idxs].sum()) if idxs.size else 0
        #                     viol = max(0, lhs - rhs_eff_i)
        #                     if viol < 1:
        #                         continue
        #                     score = viol / max(1, int(math.sqrt(len(idxs))))
        #                     scored.append((score, cut_n, rhs_int))
        #                 scored.sort(reverse=True, key=lambda t: t[0])
        #                 picked = [(cut_n, rhs_int) for s, cut_n, rhs_int in scored[:3]]

        #                 if picked:
        #                     # Merge, keep μ for existing, μ=0.0 for new
        #                     existing = {frozenset(c): (i, r) for i, (c, r) in enumerate(self.best_cuts)}
        #                     any_new = False
        #                     for cut_n, rhs_int in picked:
        #                         key = frozenset(cut_n)
        #                         if key in existing:
        #                             i_old, old_rhs = existing[key]
        #                             if rhs_int > old_rhs:
        #                                 self.best_cuts[i_old] = (cut_n, rhs_int)
        #                         else:
        #                             self.best_cuts.append((cut_n, rhs_int))
        #                             self.best_cut_multipliers[len(self.best_cuts)-1] = 0.0
        #                             new_cuts_for_children.append((set(cut_n), int(rhs_int)))
        #                             any_new = True

        #                     # Rebuild (index-safe)
        #                     (self.best_cuts,
        #                     self.best_cut_multipliers,
        #                     cut_edge_idx,
        #                     cut_free_sizes,
        #                     rhs_eff_vec) = _rebuild_cut_structures(self.best_cuts, self.best_cut_multipliers)

        #                     # Make sure pricing sees the fresh cut structure too
        #                     self._cut_edge_idx = cut_edge_idx
        #                     self._rhs_eff_vec  = rhs_eff_vec
        #                     self._mw_cached = None
        #                     self._mw_lambda = None
        #                     self._mw_mu = np.zeros(len(cut_edge_idx), dtype=float)

        #                     if len(self.best_cuts) == 0:
        #                         cut_edge_idx = []
        #                         rhs_eff_vec = np.array([], dtype=float)

        #                     # Immediate reprice after separation; start μ cooldown and freeze separation
        #                     if any_new:
        #                         try:
        #                             mst_cost, mst_length, mst_edges = self.compute_mst()
        #                         except Exception:
        #                             mst_cost, mst_length, mst_edges = self.compute_mst()
        #                         self.last_mst_edges = [tuple(sorted((u, v))) for u, v in (mst_edges or [])]
        #                         prev_mst_edges = self.last_mst_edges
        #                         prev_weights = self.compute_modified_weights()
        #                         _just_separated = True
        #                         _just_separated_cooldown = 2
        #                         _separation_freeze = 3

        #         # -------- Stopping --------
        #         if (iter_num >= MIN_ITERS_BEFORE_CONV) and (
        #             no_improvement_count > stagnation_threshold or
        #             self.step_size < 1e-8
        #         ):
        #             if self.verbose:
        #                 reason = ("no improvement" if no_improvement_count > stagnation_threshold else "step too small")
        #                 print(f"Stop at iter {iter_num} (Reason: {reason})")
        #             break

        #         # -------- Updates (Polyak / Decay), all index-safe via zip --------
        #         self.step_sizes.append(self.step_size)
        #         if len(self.step_sizes) > MAX_HISTORY:
        #             self.step_sizes = self.step_sizes[-MAX_HISTORY//2:]
        #         self.multipliers.append(self.lmbda)
        #         if len(self.multipliers) > MAX_HISTORY:
        #             self.multipliers = self.multipliers[-MAX_HISTORY//2:]

        #         if self.best_upper_bound < float('inf'):
        #             self._moving_upper = 0.95 * self._moving_upper + 0.05 * self.best_upper_bound
        #         gamma_iter = max(0.05, min(0.2, gamma_base * (1 - iter_num / max_iter)))

        #         # norms
        #         mu_norm2 = 0.0
        #         for vi, size_i, idxs in zip(violations, cut_free_sizes, cut_edge_idx):
        #             if idxs.size == 0:
        #                 continue
        #             g = vi / (1.0 + size_i)
        #             mu_norm2 += g * g
        #         mu_norm2 = max(1e-10, mu_norm2)
        #         knap_norm2 = max(1e-10, knapsack_subgradient ** 2)

        #         if polyak_enabled and self._moving_upper < float('inf'):
        #             current_L = (lagrangian_bound if not math.isnan(lagrangian_bound) else self.best_lower_bound)
        #             gap = max(1e-6, self._moving_upper - current_L)

        #             polyak_lambda_step = gamma_iter * gap / knap_norm2
        #             polyak_lambda_step = max(1e-8, min(polyak_lambda_step, self.step_size * 2))

        #             polyak_cut_step = gamma_mu * gap / mu_norm2
        #             polyak_cut_step = max(1e-8, min(polyak_cut_step, self.step_size * 2))

        #             # λ update
        #             proposed_lambda = self.lmbda + polyak_lambda_step * knapsack_subgradient
        #             beta = self.momentum_beta
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))

        #             # μ update (cooldown after separation)
        #             if _just_separated_cooldown <= 0:
        #                 for i, (vi, size_i, idxs) in enumerate(zip(violations, cut_free_sizes, cut_edge_idx)):
        #                     if idxs.size == 0:
        #                         continue
        #                     g = vi / (1.0 + size_i)
        #                     cur = self.best_cut_multipliers.get(i, 0.0)
        #                     inc = polyak_cut_step * g
        #                     if inc > mu_increment_cap: inc = mu_increment_cap
        #                     elif inc < -mu_increment_cap: inc = -mu_increment_cap
        #                     new_mu = cur + inc
        #                     if new_mu < 0.0: new_mu = 0.0
        #                     if new_mu > mu_value_cap: new_mu = mu_value_cap
        #                     self.best_cut_multipliers[i] = new_mu
        #             else:
        #                 _just_separated_cooldown -= 1
        #         else:
        #             # decay mode
        #             self.step_size *= self.p
        #             beta = self.momentum_beta
        #             proposed_lambda = self.lmbda + self.step_size * knapsack_subgradient
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))

        #             # μ update (cooldown after separation)
        #             if _just_separated_cooldown <= 0:
        #                 for i, (vi, size_i, idxs) in enumerate(zip(violations, cut_free_sizes, cut_edge_idx)):
        #                     if idxs.size == 0:
        #                         continue
        #                     g = vi / (1.0 + size_i)
        #                     cur = self.best_cut_multipliers.get(i, 0.0)
        #                     inc = self.step_size * g
        #                     if inc > mu_increment_cap: inc = mu_increment_cap
        #                     elif inc < -mu_increment_cap: inc = -mu_increment_cap
        #                     new_mu = cur + inc
        #                     if new_mu < 0.0: new_mu = 0.0
        #                     if new_mu > mu_value_cap: new_mu = mu_value_cap
        #                     self.best_cut_multipliers[i] = new_mu
        #             else:
        #                 _just_separated_cooldown -= 1

        #         # Soft decay for inactive μ to reduce weight thrashing
        #         for i, (vi, idxs) in enumerate(zip(violations, cut_edge_idx)):
        #             if len(idxs) == 0:
        #                 continue
        #             if vi <= 0.0:
        #                 self.best_cut_multipliers[i] *= 0.9
        #                 if self.best_cut_multipliers[i] < 1e-12:
        #                     self.best_cut_multipliers[i] = 0.0

        #         # Ensure next iterate sees fresh priced weights after λ/μ changed
        #         self._mw_cached = None
        #         self._mw_lambda = None

        #         # decrement separation freeze counter
        #         if _separation_freeze > 0:
        #             _separation_freeze -= 1

        #         # Optional cleanup
        #         if iter_num % CLEANUP_INTERVAL == 0:
        #             if len(self.fractional_solutions) > 10:
        #                 self.fractional_solutions = self.fractional_solutions[-5:]

        #     # No final separation; children get only cuts added during this node
        #     if self.verbose:
        #         print(f"Solve completed: iterations={iter_num+1}, "
        #             f"lower={self.best_lower_bound:.2f}, upper={self.best_upper_bound:.2f}, "
        #             f"λ={self.lmbda:.6f}, cuts={len(self.best_cuts)}")
        #         if self.best_upper_bound < float('inf') and self.best_lower_bound > -float('inf'):
        #             print(f"Duality gap: {self.best_upper_bound - self.best_lower_bound:.4f}")
        #         print(f"Final step size: {self.step_size:.8f}")

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, new_cuts_for_children


        
        # else:  # Subgradient method with Polyak hybrid (in-node separation; minimal & consistent)
        #     import math
        #     import numpy as np

        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS = 30
        #     MAX_HISTORY = 30
        #     CLEANUP_INTERVAL = 10
        #     max_iter = min(self.max_iter, 200)

        #     # Polyak / momentum hyperparams (unchanged)
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = 0.10
        #     gamma_mu = 0.30
        #     mu_increment_cap = 5.0
        #     mu_value_cap = 1e3

        #     # In-node separation policy (cheap triggers)
        #     node_cut_frequency = getattr(self, "node_cut_frequency", 10)  # try every N iters after warmup
        #     stagnation_threshold = 15
        #     big_violation_len = getattr(self, "big_violation_len", 0.0)   # optional (0 = off)

        #     no_improvement_count = 0
        #     polyak_enabled = True

        #     # Root guard: a bit more patience at the root
        #     MIN_ITERS_BEFORE_CONV = 16 if getattr(self, "depth", 0) == 0 else 3

        #     # Moving UB for Polyak gap control (unchanged)
        #     if not hasattr(self, "_moving_upper"):
        #         self._moving_upper = self.best_upper_bound if self.best_upper_bound < float("inf") else 1000.0

        #     # ---------- helpers ----------
        #     def _norm_edge(e):
        #         u, v = e
        #         return (u, v) if u <= v else (v, u)

        #     # Normalize fixed / excluded
        #     F_in  = {_norm_edge(e) for e in getattr(self, "fixed_edges", set())}
        #     F_out = {_norm_edge(e) for e in getattr(self, "excluded_edges", set())}

        #     # Ensure maps exist
        #     if not hasattr(self, "best_cut_multipliers"):
        #         self.best_cut_multipliers = {}
        #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers_for_best_bound = {}

        #     edge_idx = self.edge_indices  # normalized (u,v) -> idx

        #     def _rebuild_cut_structures(cuts, mu_map):
        #         """
        #         Compact cuts against (F_in, F_out), rebuild indices & rhs_eff, and remap μ to dense indices.
        #         Returns: compact_cuts, compact_mu, cut_edge_idx, cut_free_sizes, rhs_eff_vec
        #         """
        #         compact_cuts, cut_edge_idx, cut_free_sizes, rhs_eff = [], [], [], []
        #         original = {frozenset(c): (i, r) for i, (c, r) in enumerate(cuts)}
        #         for (cut, rhs_i) in cuts:
        #             cut_n = {tuple(sorted(e)) for e in cut if tuple(sorted(e)) in edge_idx}
        #             rhs_eff_i = int(rhs_i) - len(cut_n & F_in)
        #             if rhs_eff_i <= 0:
        #                 continue
        #             ids = [edge_idx[e] for e in cut_n if (e not in F_in) and (e not in F_out)]
        #             if not ids:
        #                 continue
        #             arr = np.fromiter(ids, dtype=np.int32)
        #             compact_cuts.append((cut_n, int(rhs_i)))
        #             cut_edge_idx.append(arr)
        #             cut_free_sizes.append(max(1, arr.size))
        #             rhs_eff.append(float(rhs_eff_i))

        #         # remap μ to dense indices
        #         compact_mu = {}
        #         if compact_cuts:
        #             for new_i, (cut_n, _) in enumerate(compact_cuts):
        #                 key = frozenset(cut_n)
        #                 if key in original:
        #                     old_i, _ = original[key]
        #                     compact_mu[new_i] = min(float(mu_map.get(old_i, 0.0)), mu_value_cap)
        #                 else:
        #                     compact_mu[new_i] = 0.0
        #         return (compact_cuts,
        #                 compact_mu,
        #                 cut_edge_idx,
        #                 cut_free_sizes,
        #                 np.array(rhs_eff, dtype=float))

        #     # Normalize any inherited cuts then compact
        #     self.best_cuts = [(set(_norm_edge(e) for e in cut), int(rhs)) for (cut, rhs) in (self.best_cuts or [])]
        #     (self.best_cuts,
        #     self.best_cut_multipliers,
        #     cut_edge_idx,
        #     cut_free_sizes,
        #     rhs_eff_vec) = _rebuild_cut_structures(self.best_cuts, self.best_cut_multipliers)

        #     # expose for pricing
        #     self._cut_edge_idx = cut_edge_idx
        #     self._rhs_eff_vec  = rhs_eff_vec
        #     # invalidate any cached modified-weights that depend on μ / cuts
        #     self._mw_cached = None
        #     self._mw_lambda = None
        #     self._mw_mu = np.zeros(len(cut_edge_idx), dtype=float)

        #     # Track inactivity for μ drop-to-zero
        #     inactive_counts = getattr(self, "_inactive_counts", {})

        #     # Separation stabilization flags (only matter when cuts are ON)
        #     _separation_freeze = 0   # skip separation for a few iters after adding cuts
        #     _just_separated_cooldown = 0  # skip μ updates for a couple of iters after adding cuts

        #     # For children: collect only brand-new cuts produced in this node
        #     new_cuts_for_children = []

        #     # seed weights
        #     prev_weights = self.compute_modified_weights()
        #     prev_mst_edges = None
        #     self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0)

        #     for iter_num in range(max_iter):
        #         if iter_num % 10 == 0:
        #             if len(self.primal_solutions) > MAX_SOLUTIONS: self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]
        #             if len(self.subgradients) > MAX_HISTORY:       self.subgradients = self.subgradients[-MAX_HISTORY:]
        #             if len(self.step_sizes) > MAX_HISTORY:         self.step_sizes = self.step_sizes[-MAX_HISTORY:]
        #             if len(self.multipliers) > MAX_HISTORY:        self.multipliers = self.multipliers[-MAX_HISTORY:]
        #             if len(self.fractional_solutions) > 10:        self.fractional_solutions = self.fractional_solutions[-5:]

        #         # -------- MST (incremental + fallback) --------
        #         try:
        #             if prev_weights is not None and prev_mst_edges is not None:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
        #             else:
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()

        #             if (math.isnan(mst_cost) or math.isinf(mst_cost) or
        #                 math.isnan(mst_length) or math.isinf(mst_length)):
        #                 if self.verbose:
        #                     print(f"Subgradient Iter {iter_num}: Invalid MST, fallback to full")
        #                 mst_cost, mst_length, mst_edges = self.compute_mst()
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Error in MST: {e}, fallback")
        #             mst_cost, mst_length, mst_edges = self.compute_mst()

        #         # normalize MST edges
        #         self.last_mst_edges = [tuple(sorted((u, v))) for u, v in (mst_edges or [])]
        #         prev_mst_edges = self.last_mst_edges
        #         prev_weights = self.compute_modified_weights()  # for next iter

        #         # -------- Dual bound & bookkeeping --------
        #         is_feasible = (mst_length <= self.budget)
        #         self._record_primal_solution(self.last_mst_edges, is_feasible)
        #         if len(self.primal_solutions) > MAX_SOLUTIONS:
        #             recent = self.primal_solutions[-15:]
        #             older = self.primal_solutions[:-15:3]
        #             self.primal_solutions = older + recent

        #         mu_vec = (np.fromiter((self.best_cut_multipliers.get(i, 0.0) for i in range(len(self.best_cuts))),
        #                             dtype=float, count=len(self.best_cuts))
        #                 if self.best_cuts else np.array([], dtype=float))
        #         cover_cut_penalty = float(mu_vec @ rhs_eff_vec) if mu_vec.size else 0.0
        #         lagrangian_bound = mst_cost - self.lmbda * self.budget - cover_cut_penalty

        #         if (not math.isnan(lagrangian_bound) and not math.isinf(lagrangian_bound) and abs(lagrangian_bound) < 1e10):
        #             if lagrangian_bound > self.best_lower_bound + 1e-6:
        #                 self.best_lower_bound = lagrangian_bound
        #                 self.best_lambda = self.lmbda
        #                 self.best_mst_edges = self.last_mst_edges
        #                 self.best_cost = mst_cost
        #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()
        #                 no_improvement_count = 0
        #             else:
        #                 no_improvement_count += 1
        #         else:
        #             if self.verbose:
        #                 print(f"Subgradient Iter {iter_num}: Invalid L={lagrangian_bound}")
        #             no_improvement_count += 1

        #         # -------- Primal UB update --------
        #         if is_feasible:
        #             try:
        #                 real_weight, real_length = self.compute_real_weight_length()
        #                 if (not math.isnan(real_weight) and not math.isinf(real_weight)
        #                     and real_weight < self.best_upper_bound):
        #                     self.best_upper_bound = real_weight
        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error updating primal solution: {e}")

        #         # -------- Subgradients (λ and μ) --------
        #         knapsack_subgradient = mst_length - self.budget
        #         if knapsack_subgradient > 1e4: knapsack_subgradient = 1e4
        #         elif knapsack_subgradient < -1e4: knapsack_subgradient = -1e4

        #         # FREE-edge mask
        #         nE = len(self.edge_weights)
        #         mst_mask = np.zeros(nE, dtype=bool)
        #         for e in self.last_mst_edges:
        #             if e in F_in or e in F_out:
        #                 continue
        #             j = edge_idx.get(e)
        #             if j is not None:
        #                 mst_mask[j] = True

        #         # violations (index-safe)
        #         if self.best_cuts:
        #             violations = np.fromiter(
        #                 ((int(mst_mask[idxs].sum()) if idxs.size else 0) - rhs_eff
        #                 for idxs, rhs_eff in zip(cut_edge_idx, rhs_eff_vec)),
        #                 dtype=float,
        #                 count=len(cut_edge_idx)
        #             )
        #         else:
        #             violations = np.array([], dtype=float)

        #         # μ hygiene: drop/suspend inactive
        #         for (vi, idxs, rhs_eff_i), i in zip(zip(violations, cut_edge_idx, rhs_eff_vec),
        #                                             range(len(cut_edge_idx))):
        #             if (idxs.size == 0) or (rhs_eff_i <= 0.0):
        #                 self.best_cut_multipliers[i] = 0.0
        #                 inactive_counts[i] = 0
        #             else:
        #                 if vi <= 0.0:
        #                     inactive_counts[i] = inactive_counts.get(i, 0) + 1
        #                     if inactive_counts[i] >= 3:
        #                         self.best_cut_multipliers[i] = 0.0
        #                 else:
        #                     inactive_counts[i] = 0
        #         self._inactive_counts = inactive_counts

        #         if self.verbose and iter_num % 10 == 0:
        #             print(f"Subgrad Iter {iter_num}: λ={self.lmbda:.6f}, len={mst_length:.2f}, gλ={knapsack_subgradient:.2f}")

        #         # stagnation detection
        #         if iter_num > 0 and abs(knapsack_subgradient - self.subgradients[-1]) < 1e-6:
        #             self.consecutive_same_subgradient = getattr(self, 'consecutive_same_subgradient', 0) + 1
        #             if self.consecutive_same_subgradient > 10:
        #                 if self.verbose:
        #                     print("Terminating early: subgradient stagnation")
        #                 break
        #             self.step_size = max(1e-8, self.step_size * 0.7)
        #         else:
        #             self.consecutive_same_subgradient = 0

        #         self._record_subgradient(knapsack_subgradient)
        #         if len(self.subgradients) > MAX_HISTORY:
        #             self.subgradients = self.subgradients[-MAX_HISTORY//2:]

        #         # -------- In-node separation (generate -> use now) --------
        #         do_separate = (
        #             self.use_cover_cuts
        #             and (iter_num >= MIN_ITERS_BEFORE_CONV)
        #             and (_separation_freeze <= 0)
        #             and (
        #                 (mst_length > self.budget)  # only if current MST violates budget
        #                 or (no_improvement_count >= max(3, stagnation_threshold // 2))
        #                 or (big_violation_len > 0.0 and (mst_length - self.budget) >= big_violation_len)
        #             )
        #             and (
        #                 (node_cut_frequency > 0 and (iter_num % node_cut_frequency == 0))
        #                 or (no_improvement_count >= max(3, stagnation_threshold // 2))
        #                 or (big_violation_len > 0.0 and (mst_length - self.budget) >= big_violation_len)
        #             )
        #         )
        #         if do_separate:
        #             try:
        #                 cand_cuts = self.generate_cover_cuts(self.last_mst_edges) or []
        #             except Exception:
        #                 cand_cuts = []

        #             if cand_cuts:
        #                 # Score by violation strength (lhs - rhs_eff) and keep a few strongest
        #                 scored = []
        #                 for cut, rhs in cand_cuts:
        #                     cut_n = {tuple(sorted(e)) for e in cut if tuple(sorted(e)) in edge_idx}
        #                     rhs_int = int(rhs)
        #                     rhs_eff_i = rhs_int - len(cut_n & F_in)
        #                     if rhs_eff_i <= 0:
        #                         continue
        #                     ids = [edge_idx[e] for e in cut_n if (e not in F_in and e not in F_out)]
        #                     if not ids:
        #                         continue
        #                     idxs = np.fromiter(ids, dtype=np.int32)
        #                     lhs = int(mst_mask[idxs].sum()) if idxs.size else 0
        #                     viol = max(0, lhs - rhs_eff_i)
        #                     if viol < 1:
        #                         continue
        #                     # normalize a bit by size to avoid very large noisy sets
        #                     score = viol / max(1, int(math.sqrt(len(idxs))))
        #                     scored.append((score, cut_n, rhs_int))

        #                 scored.sort(reverse=True, key=lambda t: t[0])
        #                 picked = [(cut_n, rhs_int) for s, cut_n, rhs_int in scored[:3]]

        #                 if picked:
        #                     # Merge into best_cuts; existing keep μ, brand-new μ=0.0
        #                     existing = {frozenset(c): (i, r) for i, (c, r) in enumerate(self.best_cuts)}
        #                     any_new = False
        #                     for cut_n, rhs_int in picked:
        #                         key = frozenset(cut_n)
        #                         if key in existing:
        #                             i_old, old_rhs = existing[key]
        #                             if rhs_int > old_rhs:
        #                                 self.best_cuts[i_old] = (cut_n, rhs_int)
        #                         else:
        #                             self.best_cuts.append((cut_n, rhs_int))
        #                             self.best_cut_multipliers[len(self.best_cuts)-1] = 0.0
        #                             new_cuts_for_children.append((set(cut_n), int(rhs_int)))
        #                             any_new = True

        #                     # Rebuild structures and reprice once so cuts *take effect now*
        #                     (self.best_cuts,
        #                     self.best_cut_multipliers,
        #                     cut_edge_idx,
        #                     cut_free_sizes,
        #                     rhs_eff_vec) = _rebuild_cut_structures(self.best_cuts, self.best_cut_multipliers)

        #                     self._cut_edge_idx = cut_edge_idx
        #                     self._rhs_eff_vec  = rhs_eff_vec
        #                     self._mw_cached = None
        #                     self._mw_lambda = None
        #                     self._mw_mu = np.zeros(len(cut_edge_idx), dtype=float)

        #                     if any_new:
        #                         try:
        #                             mst_cost, mst_length, mst_edges = self.compute_mst()
        #                         except Exception:
        #                             mst_cost, mst_length, mst_edges = self.compute_mst()
        #                         self.last_mst_edges = [tuple(sorted((u, v))) for u, v in (mst_edges or [])]
        #                         prev_mst_edges = self.last_mst_edges
        #                         prev_weights = self.compute_modified_weights()
        #                         # cool-down: avoid thrashing right after separation
        #                         _separation_freeze = 3
        #                         _just_separated_cooldown = 2

        #         # -------- Stopping --------
        #         if (iter_num >= MIN_ITERS_BEFORE_CONV) and (
        #             no_improvement_count > stagnation_threshold or
        #             self.step_size < 1e-8
        #         ):
        #             if self.verbose:
        #                 reason = ("no improvement" if no_improvement_count > stagnation_threshold else "step too small")
        #                 print(f"Stop at iter {iter_num} (Reason: {reason})")
        #             break

        #         # -------- Updates (Polyak / Decay), as in your code --------
        #         self.step_sizes.append(self.step_size)
        #         if len(self.step_sizes) > MAX_HISTORY:
        #             self.step_sizes = self.step_sizes[-MAX_HISTORY//2:]
        #         self.multipliers.append(self.lmbda)
        #         if len(self.multipliers) > MAX_HISTORY:
        #             self.multipliers = self.multipliers[-MAX_HISTORY//2:]

        #         if self.best_upper_bound < float('inf'):
        #             self._moving_upper = 0.95 * self._moving_upper + 0.05 * self.best_upper_bound
        #         gamma_iter = max(0.05, min(0.2, gamma_base * (1 - iter_num / max_iter)))

        #         # norms (unchanged)
        #         mu_norm2 = 0.0
        #         for vi, size_i, idxs in zip(violations, cut_free_sizes, cut_edge_idx):
        #             if idxs.size == 0:
        #                 continue
        #             g = vi / (1.0 + size_i)
        #             mu_norm2 += g * g
        #         mu_norm2 = max(1e-10, mu_norm2)
        #         knap_norm2 = max(1e-10, knapsack_subgradient ** 2)

        #         if polyak_enabled and self._moving_upper < float('inf'):
        #             current_L = (lagrangian_bound if not math.isnan(lagrangian_bound) else self.best_lower_bound)
        #             gap = max(1e-6, self._moving_upper - current_L)

        #             polyak_lambda_step = gamma_iter * gap / knap_norm2
        #             polyak_lambda_step = max(1e-8, min(polyak_lambda_step, self.step_size * 2))

        #             polyak_cut_step = gamma_mu * gap / mu_norm2
        #             polyak_cut_step = max(1e-8, min(polyak_cut_step, self.step_size * 2))

        #             # λ update (unchanged)
        #             proposed_lambda = self.lmbda + polyak_lambda_step * knapsack_subgradient
        #             beta = self.momentum_beta
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))

        #             # μ update (unchanged rule; very light cooldown after separation)
        #             if self.use_cover_cuts and _just_separated_cooldown > 0:
        #                 _just_separated_cooldown -= 1
        #             else:
        #                 for i, (vi, size_i, idxs) in enumerate(zip(violations, cut_free_sizes, cut_edge_idx)):
        #                     if idxs.size == 0:
        #                         continue
        #                     g = vi / (1.0 + size_i)
        #                     cur = self.best_cut_multipliers.get(i, 0.0)
        #                     inc = polyak_cut_step * g
        #                     if inc > mu_increment_cap: inc = mu_increment_cap
        #                     elif inc < -mu_increment_cap: inc = -mu_increment_cap
        #                     new_mu = cur + inc
        #                     if new_mu < 0.0: new_mu = 0.0
        #                     if new_mu > mu_value_cap: new_mu = mu_value_cap
        #                     self.best_cut_multipliers[i] = new_mu
        #         else:
        #             # decay mode (unchanged)
        #             self.step_size *= self.p
        #             beta = self.momentum_beta
        #             proposed_lambda = self.lmbda + self.step_size * knapsack_subgradient
        #             new_lambda = (1 - beta) * self.lmbda + beta * proposed_lambda
        #             self.lmbda = max(0.0, min(new_lambda, 1e4))
        #             if self.use_cover_cuts and _just_separated_cooldown > 0:
        #                 _just_separated_cooldown -= 1
        #             else:
        #                 for i, (vi, size_i, idxs) in enumerate(zip(violations, cut_free_sizes, cut_edge_idx)):
        #                     if idxs.size == 0:
        #                         continue
        #                     g = vi / (1.0 + size_i)
        #                     cur = self.best_cut_multipliers.get(i, 0.0)
        #                     inc = self.step_size * g
        #                     if inc > mu_increment_cap: inc = mu_increment_cap
        #                     elif inc < -mu_increment_cap: inc = -mu_increment_cap
        #                     new_mu = cur + inc
        #                     if new_mu < 0.0: new_mu = 0.0
        #                     if new_mu > mu_value_cap: new_mu = mu_value_cap
        #                     self.best_cut_multipliers[i] = new_mu

        #         # Ensure next iterate sees fresh priced weights after λ/μ changed
        #         self._mw_cached = None
        #         self._mw_lambda = None

        #         # decrement separation freeze counter
        #         if _separation_freeze > 0:
        #             _separation_freeze -= 1

        #         # Optional cleanup
        #         if iter_num % CLEANUP_INTERVAL == 0:
        #             if len(self.fractional_solutions) > 10:
        #                 self.fractional_solutions = self.fractional_solutions[-5:]

        #     # Done: children get only the brand-new cuts from this node
        #     if self.verbose:
        #         print(f"Solve completed: iterations={iter_num+1}, "
        #             f"lower={self.best_lower_bound:.2f}, upper={self.best_upper_bound:.2f}, "
        #             f"λ={self.lmbda:.6f}, cuts={len(self.best_cuts)}")
        #         if self.best_upper_bound < float('inf') and self.best_lower_bound > -float('inf'):
        #             print(f"Duality gap: {self.best_upper_bound - self.best_lower_bound:.4f}")
        #         print(f"Final step size: {self.step_size:.8f}")

        #     end_time = time()
        #     LagrangianMST.total_compute_time += end_time - start_time
        #     return self.best_lower_bound, self.best_upper_bound, new_cuts_for_children




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

    
    # def compute_dantzig_wolfe_solution(self, node):
    #     start_time = time()
        
    #     if len(self.primal_solutions) < 2 or len(self.multipliers) != len(self.primal_solutions):
    #         if self.verbose:
    #             print("Insufficient primal solutions for Dantzig-Wolfe")
    #         return None

    #     # Filter valid MSTs
    #     valid_msts = []
    #     for mst_edges, _ in self.primal_solutions:
    #         mst_edges_normalized = {tuple(sorted((u, v))) for u, v in mst_edges}
    #         if (all(e in mst_edges_normalized for e in self.fixed_edges) and
    #             not any(e in mst_edges_normalized for e in self.excluded_edges)):
    #             valid_msts.append(mst_edges_normalized)
        
    #     if len(valid_msts) < 2:
    #         if self.verbose:
    #             print(f"Only {len(valid_msts)} valid MSTs after filtering")
    #         return None

    #     # Select diverse MSTs
    #     max_msts = min(10, len(valid_msts))
    #     selected_msts = []
    #     covered_edges = set()
    #     remaining_msts = valid_msts.copy()
        
    #     while remaining_msts and len(selected_msts) < max_msts:
    #         best_mst = None
    #         best_score = -1
    #         for mst in remaining_msts:
    #             new_edges = mst - covered_edges
    #             score = len(new_edges)
    #             if score > best_score:
    #                 best_score = score
    #                 best_mst = mst
    #         if best_mst:
    #             selected_msts.append(best_mst)
    #             covered_edges.update(best_mst)
    #             remaining_msts.remove(best_mst)
    #         else:
    #             break

    #     if len(selected_msts) < 2:
    #         if self.verbose:
    #             print(f"Only {len(selected_msts)} diverse MSTs selected")
    #         return None

    #     if self.verbose:
    #         print(f"Using {len(selected_msts)} diverse MSTs for Dantzig-Wolfe")

    #     num_msts = len(selected_msts)
    #     edge_indices = self.edge_indices
    #     c = []
        
    #     for mst_edges in selected_msts:
    #         weight = sum(self.edge_weights[edge_indices[e]] for e in mst_edges)
    #         c.append(weight + 0.1 * (1.0 / num_msts))  # Small tie-breaker

    #     # Convex combination constraint
    #     A_eq = [np.ones(num_msts)]
    #     b_eq = [1.0]

    #     # Budget as inequality constraint
    #     A_ub = []
    #     b_ub = []

    #     lengths = [sum(self.edge_lengths[edge_indices[e]] for e in mst_edges)
    #             for mst_edges in selected_msts]
    #     A_ub.append(lengths)
    #     b_ub.append(self.budget)

    #     # Cover cuts as inequality constraints
    #     for cut, rhs in self.best_cuts:
    #         cut_indices = [edge_indices[e] for e in cut if e in edge_indices]
    #         if cut_indices:
    #             row = np.zeros(num_msts)
    #             for k, mst_edges in enumerate(selected_msts):
    #                 cut_count = sum(1 for e in mst_edges if e in cut)
    #                 row[k] = cut_count
    #             A_ub.append(row)
    #             b_ub.append(rhs)

    #     bounds = [(0, None) for _ in range(num_msts)]

    #     try:
    #         res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    #         if not res.success:
    #             if self.verbose:
    #                 print(f"LP failed: {res.message}")
    #             return None
    #         lambda_k = res.x
    #     except Exception as e:
    #         if self.verbose:
    #             print(f"LP solver error: {e}")
    #         return None

    #     # Build fractional edge solution
    #     edge_weights = {}
    #     edge_counts = defaultdict(int)
    #     for u, v in self.edge_list:
    #         e = (u, v)
    #         weight = sum(lambda_k[k] for k, mst_edges in enumerate(selected_msts) if e in mst_edges)
    #         if weight > 1e-6:
    #             edge_weights[e] = weight
    #             edge_counts[weight] += 1

    #     self._log_fractional_solution("Dantzig-Wolfe", edge_weights, selected_msts, time() - start_time)

    #     if self.verbose:
    #         print(f"Dantzig-Wolfe solution: {len(edge_weights)} edges")
    #         print(f"LP weights (lambda_k): {lambda_k}")
    #         print(f"Edge weight distribution: {dict(sorted(edge_counts.items()))}")
    #         for cut, rhs in self.best_cuts:
    #             cut_count = sum(edge_weights.get(e, 0) for e in cut)
    #             print(f"Cut {cut}: count={cut_count:.2f}, rhs={rhs}")

    #     # Fallback if weights are too uniform
    #     unique_weights = len(set(edge_weights.values()))
    #     if unique_weights < 0.5 * len(edge_weights) and len(valid_msts) > len(selected_msts):
    #         if self.verbose:
    #             print("Too many identical weights; falling back to frequency-based weights")
    #         edge_freq = defaultdict(float)
    #         for mst in valid_msts:
    #             for e in mst:
    #                 edge_freq[e] += 1.0 / len(valid_msts)
    #         edge_weights = {e: w for e, w in edge_freq.items() if w > 1e-6}
    #         self._log_fractional_solution("Dantzig-Wolfe-Fallback", edge_weights, valid_msts, time() - start_time)

    #     return edge_weights if edge_weights else None
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
##################################  
