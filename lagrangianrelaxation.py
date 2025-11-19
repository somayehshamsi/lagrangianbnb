
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
         

    

        else:  # Subgradient method with Polyak hybrid + pre- & in-loop separation (with depth-based freezing, no new constants) badak nabod
            import numpy as np
            import math

            # --- Tunables / safety limits ---
            MAX_SOLUTIONS = 50
            CLEANUP_INTERVAL = 100
            max_iter = min(self.max_iter, 200)  # unchanged

            # Polyak / momentum hyperparams (for λ)
            self.momentum_beta = getattr(self, "momentum_beta", 0.9)
            gamma_base = 0.10

            # μ updates: conservative
            gamma_mu = getattr(self, "gamma_mu", 0.30)            # unchanged
            mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
            dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-6)
            eps = 1e-12

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
                # No cuts in the dual: run a compact λ-only subgradient method.
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

                while iter_num < max_iter:
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

                    # prune primal_solutions history
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

                    # 4) Subgradient & λ-update (Polyak + momentum)
                    knapsack_subgradient = float(mst_length - self.budget)
                    self.subgradients.append(knapsack_subgradient)

                    norm_sq = knapsack_subgradient ** 2

                    if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
                        gap = max(0.0, self.best_upper_bound - lagrangian_bound)
                        theta = gamma_base
                        alpha = theta * gap / (norm_sq + eps)
                    else:
                        alpha = getattr(self, "step_size", 1e-5)

                    v_prev = getattr(self, "_v_lambda", 0.0)
                    v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
                    self._v_lambda = v_new
                    self.lmbda = max(0.0, self.lmbda + alpha * v_new)

                    self.step_sizes.append(alpha)
                    self.multipliers.append((self.lmbda, {}))

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

            # Within this node:
            # - dynamic: we may generate cuts and update μ
            # - present: we have cuts in the dual (for pricing), even if frozen
            cuts_dynamic_here = self.use_cover_cuts and cutting_active_here
            cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

            no_improvement_count = 0
            polyak_enabled = True

            # Collect new cuts generated at THIS node (pre + in-loop) to return
            node_new_cuts = []

            # Will store MST from pre-separation to reuse in iter 0
            pre_mst_available = False
            pre_mst_cost = None
            pre_mst_length = None
            pre_mst_edges = None

            # ------------------------------------------------------------------
            # 2) PRE-SEPARATION AT NODE START (only warm-start MST now)
            # ------------------------------------------------------------------
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
            # 3) Compute rhs_eff and detect infeasibility (node-level)
            # ------------------------------------------------------------------
            self._rhs_eff = {}
            if cuts_present_here:
                for idx_c, (cut, rhs) in enumerate(self.best_cuts):
                    rhs_eff = int(rhs) - len(cut & F_in)
                    self._rhs_eff[idx_c] = rhs_eff
                    if rhs_eff < 0:
                        # node infeasible due to fixed edges saturating the cut
                        end_time = time()
                        LagrangianMST.total_compute_time += end_time - start_time
                        return float('inf'), self.best_upper_bound, node_new_cuts

            # ------------------------------------------------------------------
            # 4) Trim number of cuts at node start (keep important ones)
            # ------------------------------------------------------------------
            max_active_cuts = getattr(self, "max_active_cuts", 5)
            if cuts_present_here and len(self.best_cuts) > max_active_cuts:
                parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
                if not parent_mu_map:
                    parent_mu_map = self.best_cut_multipliers

                idx_and_cut = list(enumerate(self.best_cuts))
                # priority: large |μ|
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

            # Re-evaluate presence after trimming
            cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

            # ------------------------------------------------------------------
            # 5) Precompute cut -> edge index arrays (FIXED for this node, but
            #    we will rebuild them if we add cuts in-loop)
            # ------------------------------------------------------------------
            cut_edge_idx_free = []
            cut_free_sizes = []
            cut_edge_idx_all = []
            rhs_eff_vec = np.zeros(0, dtype=float)
            rhs_vec = np.zeros(0, dtype=float)

            def _rebuild_cut_structures():
                """
                Rebuild index arrays and rhs vectors from self.best_cuts & self._rhs_eff.

                OPT:
                - If cover cuts are off, we just keep everything empty and return.
                - If cuts are present but not dynamic (frozen deep node), we skip building
                  the 'free' arrays that are only used by dynamic separation.
                """
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
                    # FREE indices (for dynamic refinements) – only if needed
                    if build_free:
                        idxs_free = [
                            edge_idx[e] for e in cut
                            if (e not in F_in and e not in F_out) and (e in edge_idx)
                        ]
                        arr_free = (np.fromiter(idxs_free, dtype=np.int32)
                                    if idxs_free else np.empty(0, dtype=np.int32))
                        cut_edge_idx_free.append(arr_free)
                        cut_free_sizes.append(max(1, len(idxs_free)))  # avoid /0
                    else:
                        cut_edge_idx_free.append(np.empty(0, dtype=np.int32))
                        cut_free_sizes.append(1)

                    # ALL indices (for dual pricing & true μ-subgradient)
                    idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
                    arr_all = (np.fromiter(idxs_all, dtype=np.int32)
                               if idxs_all else np.empty(0, dtype=np.int32))
                    cut_edge_idx_all.append(arr_all)

                # stash for compute_modified_weights (used for pricing)
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

            # track how "useful" each cut was at this node (only positive violation)
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

            # Seed priced weights so iteration 0 is consistent
            prev_weights = None
            prev_mst_edges = None

            last_g_lambda = None  # for stagnation check

            # ------------------------------------------------------------------
            # 6) Subgradient iterations (dual structure mostly fixed, but we may
            #    add a few extra cuts in-loop with full rebuild).
            # ------------------------------------------------------------------
            base_max_iter = int(max_iter)
            extra_iter_for_cuts = getattr(self, "extra_iter_for_cuts", base_max_iter)
            hard_cap_iter = base_max_iter + extra_iter_for_cuts

            dynamic_max_iter = base_max_iter
            iter_num = 0

            # Flags for violation-based cut generation
            violation_seen_for_cuts = False   # did we ever see len(T) > B in this node?
            did_separate_here = False         # did we already call generate_cover_cuts at this node?

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

                # --------------------------------------------------------------
                # Detect violating MST and trigger ONE-TIME cut generation
                # --------------------------------------------------------------
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
                                print(f"Error generating cuts from violating MST at depth {depth}, iter {iter_num}: {e}")

                        did_separate_here = True

                # Prepare weights for next iteration (using current λ, μ and possibly updated cuts)
                prev_weights = self.compute_modified_weights()

                # 2) Dual & primal bookkeeping
                is_feasible = (mst_length <= self.budget)

                # (a) primal & UB
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

                # (b) Lagrangian dual value (dualized cuts use ORIGINAL rhs)
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

                # 4) Step sizes & updates (joint Polyak for λ and μ)
                self.subgradients.append(knapsack_subgradient)

                norm_sq = knapsack_subgradient ** 2
                for g in cut_subgradients:
                    norm_sq += g ** 2

                if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
                    gap = max(0.0, self.best_upper_bound - lagrangian_bound)
                    theta = gamma_base
                    alpha = theta * gap / (norm_sq + eps)
                else:
                    alpha = getattr(self, "step_size", 1e-5)

                # λ update with momentum
                v_prev = getattr(self, "_v_lambda", 0.0)
                v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
                self._v_lambda = v_new
                self.lmbda = max(0.0, self.lmbda + alpha * v_new)

                # μ updates (only if cuts are dynamic here)
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
            # 7) Optional: drop "dead" cuts for future nodes
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



        # else:  # Subgradient method with serious-step Polyak + depth-based cuts
        #     import numpy as np
        #     import math

        #     # --- Tunables / safety limits -----------------------------------
        #     MAX_SOLUTIONS = getattr(self, "max_solutions_hist", 50)
        #     CLEANUP_INTERVAL = getattr(self, "cleanup_interval", 100)
        #     max_iter_base = min(self.max_iter, getattr(self, "max_subgrad_iter", 200))

        #     # Polyak / momentum hyperparams (for λ)
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base = getattr(self, "gamma_base", 0.10)      # base Polyak scaling

        #     # μ updates: conservative
        #     gamma_mu = getattr(self, "gamma_mu", 0.30)
        #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
        #     dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-6)
        #     eps = 1e-12

        #     # serious-step control
        #     serious_improve_tol = getattr(self, "serious_improve_tol", 1e-3)
        #     serious_patience = getattr(self, "serious_patience", 8)
        #     alpha_max = getattr(self, "alpha_max", 1.0)         # cap on Polyak step
        #     grad_tol = getattr(self, "grad_tol", 1e-5)          # joint grad norm tol

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

        #         # serious-step tracking for λ-only case (uses global best_upper_bound)
        #         best_serious_bound = -float("inf")
        #         non_serious_iters = 0
        #         theta = gamma_base

        #         while iter_num < max_iter_base:
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

        #             # 4) Subgradient & λ-update (serious-step Polyak + momentum)
        #             knapsack_subgradient = float(mst_length - self.budget)
        #             self.subgradients.append(knapsack_subgradient)

        #             norm_sq = knapsack_subgradient ** 2

        #             if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #                 gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #                 alpha = theta * gap / (norm_sq + eps)
        #                 if alpha_max is not None:
        #                     alpha = min(alpha, alpha_max)
        #             else:
        #                 alpha = getattr(self, "step_size", 1e-5)

        #             # momentum update
        #             v_prev = getattr(self, "_v_lambda", 0.0)
        #             v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #             self._v_lambda = v_new
        #             self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #             self.step_sizes.append(alpha)
        #             self.multipliers.append((self.lmbda, {}))

        #             # serious-step logic on dual bound
        #             if lagrangian_bound > best_serious_bound + serious_improve_tol:
        #                 best_serious_bound = lagrangian_bound
        #                 non_serious_iters = 0
        #             else:
        #                 non_serious_iters += 1
        #                 if non_serious_iters >= serious_patience:
        #                     theta *= 0.5
        #                     non_serious_iters = 0

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

        #             # stopping by gradient norm
        #             if norm_sq < grad_tol ** 2:
        #                 if self.verbose:
        #                     print("Terminating early (no-cuts mode): small gradient norm")
        #                 break

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
        #     cuts_dynamic_here = self.use_cover_cuts and cutting_active_here
        #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #     no_improvement_count = 0
        #     polyak_enabled = True
        #     node_new_cuts = []

        #     # --------------------------------------------------------------
        #     # Pre-separation warm-start (only for dynamic cut nodes)
        #     # --------------------------------------------------------------
        #     pre_mst_available = False
        #     pre_mst_cost = None
        #     pre_mst_length = None
        #     pre_mst_edges = None

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
        #     # 2) Compute rhs_eff and detect infeasibility (fixed edges)
        #     # ------------------------------------------------------------------
        #     self._rhs_eff = {}
        #     if cuts_present_here:
        #         for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #             rhs_eff = int(rhs) - len(cut & F_in)
        #             self._rhs_eff[idx_c] = rhs_eff
        #             if rhs_eff < 0:
        #                 end_time = time()
        #                 LagrangianMST.total_compute_time += end_time - start_time
        #                 return float('inf'), self.best_upper_bound, node_new_cuts

        #     # ------------------------------------------------------------------
        #     # 3) Trim number of cuts at node start (keep important ones)
        #     # ------------------------------------------------------------------
        #     max_active_cuts = getattr(self, "max_active_cuts", 5)
        #     if cuts_present_here and len(self.best_cuts) > max_active_cuts:
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

        #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #     # ------------------------------------------------------------------
        #     # 4) Precompute cut -> edge index arrays (rebuilt when adding cuts)
        #     # ------------------------------------------------------------------
        #     cut_edge_idx_free = []
        #     cut_free_sizes = []
        #     cut_edge_idx_all = []
        #     rhs_eff_vec = np.zeros(0, dtype=float)
        #     rhs_vec = np.zeros(0, dtype=float)

        #     def _rebuild_cut_structures():
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
        #             if build_free:
        #                 idxs_free = [
        #                     edge_idx[e] for e in cut
        #                     if (e not in F_in and e not in F_out) and (e in edge_idx)
        #                 ]
        #                 arr_free = (np.fromiter(idxs_free, dtype=np.int32)
        #                             if idxs_free else np.empty(0, dtype=np.int32))
        #                 cut_edge_idx_free.append(arr_free)
        #                 cut_free_sizes.append(max(1, len(idxs_free)))
        #             else:
        #                 cut_edge_idx_free.append(np.empty(0, dtype=np.int32))
        #                 cut_free_sizes.append(1)

        #             idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #             arr_all = (np.fromiter(idxs_all, dtype=np.int32)
        #                        if idxs_all else np.empty(0, dtype=np.int32))
        #             cut_edge_idx_all.append(arr_all)

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

        #     prev_weights = None
        #     prev_mst_edges = None
        #     last_g_lambda = None

        #     # serious-step tracking (joint λ, μ)
        #     best_serious_bound = -float("inf")
        #     non_serious_iters = 0
        #     theta_joint = gamma_base

        #     # dynamic iteration control
        #     base_max_iter = int(max_iter_base)
        #     extra_iter_for_cuts = getattr(self, "extra_iter_for_cuts", base_max_iter)
        #     hard_cap_iter = base_max_iter + extra_iter_for_cuts

        #     dynamic_max_iter = base_max_iter
        #     iter_num = 0

        #     violation_seen_for_cuts = False
        #     did_separate_here = False

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

        #         # ----------------------------------------------------------
        #         # Cut separation: one-shot, only on clear knapsack violation
        #         # ----------------------------------------------------------
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
        #                         print(f"Error generating cuts at depth {depth}, iter {iter_num}: {e}")

        #                 did_separate_here = True

        #         prev_weights = self.compute_modified_weights()

        #         # 2) Dual & primal bookkeeping
        #         is_feasible = (mst_length <= self.budget)

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

        #         # 4) Joint Polyak step for (λ, μ)
        #         self.subgradients.append(knapsack_subgradient)

        #         norm_sq = knapsack_subgradient ** 2
        #         for g in cut_subgradients:
        #             norm_sq += g ** 2

        #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #             gap = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             alpha = theta_joint * gap / (norm_sq + eps)
        #             if alpha_max is not None:
        #                 alpha = min(alpha, alpha_max)
        #         else:
        #             alpha = getattr(self, "step_size", 1e-5)

        #         # λ update with momentum
        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda = max(0.0, self.lmbda + alpha * v_new)

        #         # μ updates (only dynamic cut nodes)
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

        #         # serious-step logic on dual bound
        #         if lagrangian_bound > best_serious_bound + serious_improve_tol:
        #             best_serious_bound = lagrangian_bound
        #             non_serious_iters = 0
        #         else:
        #             non_serious_iters += 1
        #             if non_serious_iters >= serious_patience:
        #                 theta_joint *= 0.5
        #                 non_serious_iters = 0

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

        #         # stopping: gradient norm and dual gap
        #         if norm_sq < grad_tol ** 2:
        #             if self.verbose:
        #                 print("Terminating early: small joint gradient norm")
        #             break
        #         if (self.best_upper_bound < float("inf")
        #                 and self.best_upper_bound - self.best_lower_bound <= getattr(self, "dual_gap_tol", 1e-4)):
        #             if self.verbose:
        #                 print("Terminating early: small dual gap")
        #             break

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
        #     # 5) Drop "dead" cuts for future nodes (never violated & tiny μ)
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
