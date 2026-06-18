
# import networkx as nx
# import numpy as np
# from time import time
# from collections import defaultdict, OrderedDict
# from scipy.optimize import linprog  
# import math
# import heapq
# import hashlib



# class UnionFind:
#     __slots__ = ['parent', 'rank', 'size']
#     def __init__(self, n):
#         self.parent = list(range(n))
#         self.rank = [0] * n
#         self.size = [1] * n
    
#     def find(self, u):
#         if self.parent[u] != u:
#             self.parent[u] = self.find(self.parent[u])
#         return self.parent[u]
    
#     def union(self, u, v):
#         pu, pv = self.find(u), self.find(v)
#         if pu == pv:
#             return False
#         if self.size[pu] < self.size[pv]:
#             pu, pv = pv, pu
#         self.parent[pv] = pu
#         self.size[pu] += self.size[pv]
#         self.rank[pu] = max(self.rank[pu], self.rank[pv] + 1)
#         return True
    
#     def connected(self, u, v):
#         return self.find(u) == self.find(v)
    
#     def count_components(self):
#         return len(set(self.find(i) for i in range(len(self.parent))))

# class LRUCache:
#     __slots__ = ['cache', 'capacity']
#     def __init__(self, capacity):
#         self.cache = OrderedDict()
#         self.capacity = capacity
    
#     def get(self, key):
#         if key not in self.cache:
#             return None
#         self.cache.move_to_end(key)
#         return self.cache[key]
    
#     def put(self, key, value):
#         if key in self.cache:
#             self.cache.move_to_end(key)

#         self.cache[key] = value
#         if len(self.cache) > self.capacity:
#             self.cache.popitem(last=False)

# class LagrangianMST:
#     total_compute_time = 0


#     def __init__(self, edges, num_nodes, budget, fixed_edges=None, excluded_edges=None,
#                  initial_lambda=0.05, step_size=0.001, max_iter=10, 
#                  use_cover_cuts=False, cut_frequency=5, use_bisection=False,
#                  verbose=False, shared_graph=None):
#         start_time = time()
#         self.edges = edges
#         self.num_nodes = num_nodes
#         self.budget = budget
#         self.fixed_edges = {tuple(sorted((u, v))) for u, v in (fixed_edges or set())}
#         self.excluded_edges = {tuple(sorted((u, v))) for u, v in (excluded_edges or set())}

#         edge_key = id(edges)
#         if getattr(LagrangianMST, "_edge_key", None) != edge_key:
#             LagrangianMST._edge_key = edge_key
#             edge_list = [tuple(sorted((u, v))) for u, v, _, _ in edges]
#             LagrangianMST._edge_list = edge_list
#             LagrangianMST._edge_indices = {edge: idx for idx, edge in enumerate(edge_list)}
#             LagrangianMST._edge_weights = np.array([w for _, _, w, _ in edges], dtype=np.float32)
#             LagrangianMST._edge_lengths = np.array([l for _, _, _, l in edges], dtype=np.float32)
#             LagrangianMST._edge_attributes = {
#                 edge: (w, l) for (edge, (_, _, w, l)) in zip(edge_list, edges)
#             }

#         self.edge_list = LagrangianMST._edge_list
#         self.edge_indices = LagrangianMST._edge_indices
#         self.edge_weights = LagrangianMST._edge_weights
#         self.edge_lengths = LagrangianMST._edge_lengths
#         self.edge_attributes = LagrangianMST._edge_attributes

#         self.lmbda = initial_lambda
#         self.step_size = step_size
#         # self.p = p
#         self.max_iter = max_iter
#         self.use_bisection = use_bisection
#         self.verbose = verbose

#         self.best_lower_bound = float('-inf')
#         self.best_upper_bound = float('inf')
#         self.last_mst_edges = []
#         self.primal_solutions = []
#         self.fractional_solutions = []
#         self.step_sizes = []
#         self.subgradients = []
#         self._MAX_HISTORY = 100
#         self._primal_history_cap = 30
#         self._fractional_history_cap = 50
#         self._subgradient_history_cap = 20

#         self.best_lambda = self.lmbda
#         self.best_mst_edges = None
#         self.best_cost = 0

#         self.use_cover_cuts = use_cover_cuts
#         self.cut_frequency = cut_frequency
#         self.best_cuts = []
#         self.best_cut_multipliers = {}

#         self.multipliers = []

#         self.fixed_edge_indices = {
#             self.edge_indices.get((u, v)) for u, v in self.fixed_edges
#             if (u, v) in self.edge_indices
#         }
#         self.excluded_edge_indices = {
#             self.edge_indices.get((u, v)) for u, v in self.excluded_edges
#             if (u, v) in self.edge_indices
#         }

#         self.cache_tolerance = 1e-6 if num_nodes > 100 else 1e-8
#         self.mst_cache = LRUCache(capacity=64)


#         self.last_mst_edges = None

#         if shared_graph is not None:
#             self.graph = shared_graph
#         else:
#             self.graph = nx.Graph()
#             self.graph.add_edges_from(self.edge_list)

#         self._free_mask_cache = None
#         self._free_mask_key = None
#         self._mw_cached = None
#         self._mw_lambda = None
#         self._mw_mu = None

#         end_time = time()
#         LagrangianMST.total_compute_time += end_time - start_time


    

#     def reset(self, *, fixed_edges=None, excluded_edges=None, initial_lambda=None,
#               step_size=None, max_iter=None, use_cover_cuts=None, cut_frequency=None,
#               use_bisection=None, verbose=None):
#         if fixed_edges is not None:
#             self.fixed_edges = {tuple(sorted((u, v))) for u, v in fixed_edges}
#         else:
#             self.fixed_edges = set()

#         if excluded_edges is not None:
#             self.excluded_edges = {tuple(sorted((u, v))) for u, v in excluded_edges}
#         else:
#             self.excluded_edges = set()

#         self.fixed_edge_indices = {
#             self.edge_indices.get((u, v)) for u, v in self.fixed_edges
#             if (u, v) in self.edge_indices
#         }
#         self.excluded_edge_indices = {
#             self.edge_indices.get((u, v)) for u, v in self.excluded_edges
#             if (u, v) in self.edge_indices
#         }

#         if initial_lambda is not None:
#             self.lmbda = float(initial_lambda)
#         else:
#             self.lmbda = getattr(self, "lmbda", 0.05)

#         if step_size is not None:
#             self.step_size = float(step_size)
#         if max_iter is not None:
#             self.max_iter = int(max_iter)
#         if use_cover_cuts is not None:
#             self.use_cover_cuts = bool(use_cover_cuts)
#         if cut_frequency is not None:
#             self.cut_frequency = int(cut_frequency)
#         if use_bisection is not None:
#             self.use_bisection = bool(use_bisection)
#         if verbose is not None:
#             self.verbose = bool(verbose)

#         self.best_lower_bound = float("-inf")
#         self.best_upper_bound = float("inf")

#         self.best_lambda = float(self.lmbda)
#         self.best_mst_edges = []
#         self.best_cost = 0

#         self.best_cuts = []
#         self.best_cut_multipliers = {}
#         self.best_cut_multipliers_for_best_bound = {}

#         self.multipliers = []

#         # Important when reusing solver objects in strong branching
#         self._v_lambda = 0.0

#         self.last_mst_edges = None

#         try:
#             cap = self.mst_cache.capacity
#         except Exception:
#             cap = max(20, self.num_nodes * 2)
#         self.mst_cache = LRUCache(capacity=cap)

#         self._invalidate_weight_cache()


#     def clear_iteration_state(self):
#         """Clear per-iteration buffers"""
#         self.primal_solutions = []
#         self.fractional_solutions = []
#         self.subgradients = []
#         self.step_sizes = []
#         self.multipliers = []
#         self._v_lambda = 0.0
#         # self.last_modified_weights = None
#         # self.last_mst_edges = None
#         self._invalidate_weight_cache()
#         if hasattr(self, 'mst_cache'):
#             self.mst_cache = LRUCache(capacity=5)
#     # def generate_cover_cuts(self, mst_edges):  
#     #     """
#     #     Stronger cover cuts (tightened):
#     #     - Residualization: A, B' (clamped), fixed/excluded respected
#     #     - Seed residual-minimal cover from T^λ ∩ A
#     #     - Certificate shrinking using optimistic U(S) with component-based k, and exact Kruskal fallback
#     #     - Inclusion-minimal S* shrinking under the certificate (THIS was missing)
#     #     - Micro-seed from top-L heaviest admissible edges
#     #     - Stronger safe lifting for residual-minimal covers
#     #     - Strict effective-RHS pruning + current-violation checks
#     #     - Dedup with dominance & subset-dominance
#     #     """
#     #     if not mst_edges:
#     #         return []

#     #     EPS = 1e-12
#     #     L_MICRO = 3
#     #     MAX_RETURN = 10

#     #     # --- normalize edges ---
#     #     def norm(e):
#     #         u, v = e
#     #         return (u, v) if u <= v else (v, u)

#     #     mst_norm = [norm(e) for e in mst_edges]
#     #     mst_set = set(mst_norm)

#     #     # --- accessors / data ---
#     #     edge_attr = self.edge_attributes  # edge -> (w, ℓ)
#     #     def get_len(e): return edge_attr[e][1]

#     #     fixed = set(getattr(self, "fixed_edges", set()))
#     #     excluded = set(getattr(self, "excluded_edges", set()))
#     #     budget = self.budget

#     #     # Residual budget
#     #     L_fix = sum(get_len(e) for e in fixed if e in edge_attr)
#     #     Bp = budget - L_fix

#     #     # If fixes already exceed the budget, cuts may still be useful, but be careful with rhs_eff.
#     #     # We will still attempt separation.

#     #     # Admissible edges A
#     #     A = {e for e in getattr(self, "edge_list", []) if e not in fixed and e not in excluded and e in edge_attr}
#     #     if not A:
#     #         return []

#     #     # T^λ ∩ A (use provided mst_edges)
#     #     TcapA = [e for e in mst_norm if e in A]

#     #     # If residual MST is feasible, nothing to cut
#     #     mst_len = sum(get_len(e) for e in TcapA)
#     #     if mst_len <= Bp + EPS:
#     #         return []

#     #     cuts = []

#     #     # Pre-sort A by length for U(S) and Kruskal completion
#     #     A_sorted = sorted(A, key=lambda e: get_len(e))

#     #     # --- DSU helpers (for component count & exact completion) ---
#     #     def get_nodes():
#     #         # best-effort: use graph nodes if present, otherwise infer from edge keys
#     #         if hasattr(self, "graph") and hasattr(self.graph, "nodes"):
#     #             try:
#     #                 return list(self.graph.nodes)
#     #             except Exception:
#     #                 pass
#     #         nodes = set()
#     #         for (u, v) in edge_attr.keys():
#     #             nodes.add(u); nodes.add(v)
#     #         for (u, v) in fixed:
#     #             nodes.add(u); nodes.add(v)
#     #         return list(nodes)

#     #     NODES = get_nodes()

#     #     def component_k_needed(contracted_edges):
#     #         """Number of edges needed to connect after contracting 'contracted_edges': k = #components - 1."""
#     #         parent = {n: n for n in NODES}
#     #         rank = {n: 0 for n in NODES}

#     #         def find(x):
#     #             while parent[x] != x:
#     #                 parent[x] = parent[parent[x]]
#     #                 x = parent[x]
#     #             return x

#     #         def union(x, y):
#     #             rx, ry = find(x), find(y)
#     #             if rx == ry:
#     #                 return
#     #             if rank[rx] < rank[ry]:
#     #                 parent[rx] = ry
#     #             elif rank[rx] > rank[ry]:
#     #                 parent[ry] = rx
#     #             else:
#     #                 parent[ry] = rx
#     #                 rank[rx] += 1

#     #         for (u, v) in contracted_edges:
#     #             if u in parent and v in parent:
#     #                 union(u, v)

#     #         reps = {find(n) for n in NODES}
#     #         comps = len(reps)
#     #         return max(0, comps - 1)

#     #     def U_of(Sprime):
#     #         """
#     #         Optimistic completion:
#     #         sum of k cheapest edges in A \\ S', where k = (#components after contracting fixed ∪ S') - 1.
#     #         This is stronger/more accurate than r' - |S'|.
#     #         """
#     #         Sprime_set = Sprime if isinstance(Sprime, set) else set(Sprime)
#     #         contracted = set(fixed) | Sprime_set
#     #         k = component_k_needed(contracted)
#     #         if k <= 0:
#     #             return 0.0

#     #         total = 0.0
#     #         taken = 0
#     #         for e in A_sorted:
#     #             if e in Sprime_set:
#     #                 continue
#     #             total += get_len(e)
#     #             taken += 1
#     #             if taken == k:
#     #                 break
#     #         return total if taken == k else float("inf")

#     #     def completion_mst_cost(Ssub):
#     #         """
#     #         Exact completion via Kruskal after contracting fixed ∪ Ssub.
#     #         Returns minimum additional length needed to connect components using edges in A \\ Ssub.
#     #         """
#     #         parent = {n: n for n in NODES}
#     #         rank = {n: 0 for n in NODES}

#     #         def find(x):
#     #             while parent[x] != x:
#     #                 parent[x] = parent[parent[x]]
#     #                 x = parent[x]
#     #             return x

#     #         def union(x, y):
#     #             rx, ry = find(x), find(y)
#     #             if rx == ry:
#     #                 return False
#     #             if rank[rx] < rank[ry]:
#     #                 parent[rx] = ry
#     #             elif rank[rx] > rank[ry]:
#     #                 parent[ry] = rx
#     #             else:
#     #                 parent[ry] = rx
#     #                 rank[rx] += 1
#     #             return True

#     #         contracted = set(fixed) | set(Ssub)
#     #         for (u, v) in contracted:
#     #             if u in parent and v in parent:
#     #                 union(u, v)

#     #         reps = {find(n) for n in NODES}
#     #         k_needed = max(0, len(reps) - 1)
#     #         if k_needed <= 0:
#     #             return 0.0

#     #         Sset = set(Ssub)
#     #         total = 0.0
#     #         taken = 0
#     #         for e in A_sorted:
#     #             if e in Sset:
#     #                 continue
#     #             u, v = e
#     #             if u not in parent or v not in parent:
#     #                 continue
#     #             if union(u, v):
#     #                 total += get_len(e)
#     #                 taken += 1
#     #                 if taken == k_needed:
#     #                     break
#     #         return total if taken == k_needed else float("inf")

#     #     def build_residual_minimal_cover(desc_edges):
#     #         """Minimal cover on B': add in desc ℓ, then prune shortest while violation remains."""
#     #         S, sL = [], 0.0
#     #         for e in desc_edges:
#     #             if e not in edge_attr:
#     #                 continue
#     #             S.append(e)
#     #             sL += get_len(e)
#     #             if sL > Bp + EPS:
#     #                 # prune shortest while still violating
#     #                 S.sort(key=lambda x: get_len(x))  # increasing
#     #                 k = 0
#     #                 while k < len(S) and (sL - get_len(S[k]) > Bp + EPS):
#     #                     sL -= get_len(S[k])
#     #                     k += 1
#     #                 if k > 0:
#     #                     S = S[k:]
#     #                 return S, sL
#     #         return None, None

#     #     def rhs_eff(cset):
#     #         """Effective RHS after accounting fixed-in edges."""
#     #         return len(cset) - 1 - sum(1 for e in cset if e in fixed)

#     #     def is_violated_now(cset):
#     #         """Check current MST violation: lhs > rhs_eff."""
#     #         lhs = sum(1 for e in cset if e in mst_set)
#     #         return lhs > rhs_eff(cset)

#     #     def cert_holds(Slist):
#     #         """
#     #         Certificate: sumℓ(S) + U(S) > B' (optimistic), else fallback to exact completion.
#     #         """
#     #         if not Slist or len(Slist) <= 1:
#     #             return False
#     #         if rhs_eff(Slist) <= 0:
#     #             return False
#     #         sumS = sum(get_len(e) for e in Slist)
#     #         U = U_of(Slist)
#     #         if U != float("inf") and (sumS + U) > (Bp + EPS):
#     #             return True
#     #         exact = completion_mst_cost(Slist)
#     #         return exact != float("inf") and (sumS + exact) > (Bp + EPS)

#     #     def inclusion_minimal_shrink(Sstart):
#     #         """
#     #         Make S inclusion-minimal under cert_holds by removing one edge at a time.
#     #         We try removals from longest to shortest for a small S.
#     #         """
#     #         Sstar = sorted(Sstart, key=lambda e: get_len(e), reverse=True)
#     #         changed = True
#     #         while changed and len(Sstar) > 1:
#     #             changed = False
#     #             for j in range(len(Sstar)):  # longest -> shortest
#     #                 trial = Sstar[:j] + Sstar[j+1:]
#     #                 if len(trial) <= 1:
#     #                     continue
#     #                 if cert_holds(trial):
#     #                     Sstar = sorted(trial, key=lambda e: get_len(e), reverse=True)
#     #                     changed = True
#     #                     break
#     #         return Sstar

#     #     def try_shrink_and_add(seed_S, seed_sumL):
#     #         """
#     #         Full LaTeX Step (2):
#     #         - remove longest edges until sumℓ <= B' => first S'
#     #         - require cert_holds(S')
#     #         - shrink to inclusion-minimal S* while certificate holds
#     #         - add the cut if it separates current MST
#     #         """
#     #         if not seed_S or len(seed_S) <= 1:
#     #             return

#     #         S_work = sorted(seed_S, key=lambda e: get_len(e), reverse=True)
#     #         sumL = float(seed_sumL)

#     #         # First S' with sumℓ <= B'
#     #         idx = 0
#     #         while idx < len(S_work) and sumL > Bp + EPS:
#     #             sumL -= get_len(S_work[idx])
#     #             idx += 1
#     #         Sprime = S_work[idx:]
#     #         if not Sprime or len(Sprime) <= 1:
#     #             return

#     #         if not cert_holds(Sprime):
#     #             return

#     #         Sstar = inclusion_minimal_shrink(Sprime)
#     #         if len(Sstar) <= 1:
#     #             return

#     #         if is_violated_now(Sstar):
#     #             cuts.append((set(Sstar), len(Sstar) - 1))

#     #     def lift_minimal_cover(S_min, rhs_base):
#     #         """
#     #         Stronger safe lifting for residual-minimal cover S:
#     #         Lift any f with ℓ(f) > B' - sumℓ(S) + Lmax.
#     #         (This is typically much stronger than ℓ(f) >= Lmax.)
#     #         """
#     #         S_base = set(S_min)
#     #         if not S_base:
#     #             return None
#     #         sumS = sum(get_len(e) for e in S_base)
#     #         Lmax = max(get_len(e) for e in S_base)
#     #         threshold = (Bp - sumS + Lmax)  # lift if len(f) > threshold

#     #         lift_add = {f for f in A if f not in S_base and get_len(f) > threshold + EPS}
#     #         if not lift_add:
#     #             return None

#     #         S_lift = S_base | lift_add
#     #         # RHS remains rhs_base (|S|-1 of original minimal cover)
#     #         if rhs_eff(S_lift) > 0 and is_violated_now(S_lift):
#     #             return (S_lift, rhs_base)
#     #         return None

#     #     # --- (1) primary seed from T^λ ∩ A ---
#     #     T_desc = sorted(TcapA, key=lambda e: get_len(e), reverse=True)
#     #     S_seed, sumL_seed = build_residual_minimal_cover(T_desc)
#     #     if not S_seed:
#     #         return []

#     #     S_seed = list(S_seed)
#     #     if rhs_eff(S_seed) > 0 and is_violated_now(S_seed):
#     #         cuts.append((set(S_seed), len(S_seed) - 1))

#     #     # Step (2): certificate shrink to inclusion-minimal S*
#     #     try_shrink_and_add(S_seed, sumL_seed)

#     #     # --- stronger lifting on the residual-minimal seed cover ---
#     #     lifted = lift_minimal_cover(S_seed, rhs_base=(len(S_seed) - 1))
#     #     if lifted is not None:
#     #         cuts.append(lifted)

#     #     # --- (1b) micro-seed: top-L heaviest admissible edges ---
#     #     if L_MICRO > 0 and len(A) > 0:
#     #         heavyA = sorted(A, key=lambda e: get_len(e), reverse=True)[:L_MICRO]
#     #         S2, sumL2 = build_residual_minimal_cover(heavyA)
#     #         if S2:
#     #             S2set = set(S2)
#     #             if rhs_eff(S2set) > 0 and S2set != set(S_seed) and is_violated_now(S2set):
#     #                 cuts.append((S2set, len(S2) - 1))

#     #             try_shrink_and_add(S2, sumL2)

#     #             lifted2 = lift_minimal_cover(S2, rhs_base=(len(S2) - 1))
#     #             if lifted2 is not None:
#     #                 cuts.append(lifted2)

#     #     # --- dedup & dominance-aware selection ---
#     #     uniq = {}
#     #     for cset, rhs in cuts:
#     #         key = tuple(sorted(cset))
#     #         best = uniq.get(key)
#     #         if best is None or rhs < best[1] or (rhs == best[1] and len(cset) < len(best[0])):
#     #             uniq[key] = (cset, rhs)

#     #     final = list(uniq.values())
#     #     final.sort(key=lambda t: (t[1], len(t[0])))

#     #     kept = []
#     #     for cset, rhs in final:
#     #         if rhs_eff(cset) <= 0:
#     #             continue
#     #         dominated = any(dset <= cset and drhs <= rhs for dset, drhs in kept)
#     #         if not dominated:
#     #             kept.append((cset, rhs))
#     #     return kept[:MAX_RETURN]
#     def generate_cover_cuts(self, mst_edges):
#         """
#         Cover-cut generation with exact tree-completion certificate.

#         Main logic:
#         - Work at the current B&B node with fixed edges F+ and excluded edges F-.
#         - Define residual budget B' = B - length(F+).
#         - Define admissible edges A = E \ (F+ union F-).
#         - Generate a residual-minimal seed cover from the current violating
#         Lagrangian MST T^lambda.
#         - Refine the seed using an exact minimum-length MST completion certificate:
#             contract F+ union S'
#             complete using admissible edges A \ S'
#             compute the minimum additional length by Kruskal using edge lengths
#         - Add a cut only if it is violated by the current Lagrangian MST.
#         - Optionally add lifted cuts using the residual-aware lifting rule.
#         """
#         if not mst_edges:
#             return []

#         EPS = 1e-12
#         L_MICRO = 3
#         MAX_RETURN = 10

#         # ------------------------------------------------------------
#         # Normalize edges
#         # ------------------------------------------------------------
#         def norm(e):
#             u, v = e
#             return (u, v) if u <= v else (v, u)

#         mst_norm = [norm(e) for e in mst_edges]
#         mst_set = set(mst_norm)

#         # ------------------------------------------------------------
#         # Accessors and node data
#         # ------------------------------------------------------------
#         edge_attr = self.edge_attributes  # edge -> (weight, length)

#         def get_len(e):
#             return edge_attr[e][1]

#         fixed = set(getattr(self, "fixed_edges", set()))
#         excluded = set(getattr(self, "excluded_edges", set()))
#         budget = self.budget

#         # Residual budget B' = B - length(F+)
#         L_fix = sum(get_len(e) for e in fixed if e in edge_attr)
#         Bp = budget - L_fix

#         # Admissible residual edges A = E \ (F+ union F-)
#         A = {
#             e for e in getattr(self, "edge_list", [])
#             if e not in fixed and e not in excluded and e in edge_attr
#         }

#         if not A:
#             return []

#         # Current Lagrangian tree restricted to admissible residual edges
#         TcapA = [e for e in mst_norm if e in A]

#         # If the current residual tree already respects the residual budget,
#         # there is no budget-violating tree to separate.
#         mst_len = sum(get_len(e) for e in TcapA)
#         if mst_len <= Bp + EPS:
#             return []

#         cuts = []

#         # Admissible edges sorted by LENGTH for exact completion.
#         # This is the length-MST completion part.
#         A_sorted = sorted(A, key=lambda e: get_len(e))

#         # ------------------------------------------------------------
#         # Node list for local DSU
#         # ------------------------------------------------------------
#         def get_nodes():
#             if hasattr(self, "graph") and hasattr(self.graph, "nodes"):
#                 try:
#                     return list(self.graph.nodes)
#                 except Exception:
#                     pass

#             nodes = set()
#             for (u, v) in edge_attr.keys():
#                 nodes.add(u)
#                 nodes.add(v)
#             for (u, v) in fixed:
#                 nodes.add(u)
#                 nodes.add(v)
#             return list(nodes)

#         NODES = get_nodes()

#         # ------------------------------------------------------------
#         # Exact minimum-length completion
#         # ------------------------------------------------------------
#         def completion_mst_cost(Ssub):
#             """
#             Minimum additional LENGTH needed to complete F+ union Ssub
#             to a spanning tree.

#             Steps:
#             1. Contract all fixed edges F+.
#             2. Contract all edges in Ssub.
#             3. Complete the remaining components using Kruskal on A \ Ssub,
#             sorted by edge length.
#             4. Return +inf if no spanning tree completion exists.

#             Important:
#             - If F+ union Ssub already creates a cycle, then no spanning tree can
#             contain all of those forced edges, so completion is impossible.
#             - The returned value is only the additional length beyond Ssub.
#             The fixed-edge length has already been removed through B'.
#             """
#             Sset = set(Ssub)

#             parent = {n: n for n in NODES}
#             rank = {n: 0 for n in NODES}
#             components = len(NODES)

#             def find(x):
#                 while parent[x] != x:
#                     parent[x] = parent[parent[x]]
#                     x = parent[x]
#                 return x

#             def union(x, y):
#                 nonlocal components

#                 rx, ry = find(x), find(y)
#                 if rx == ry:
#                     return False

#                 if rank[rx] < rank[ry]:
#                     parent[rx] = ry
#                 elif rank[rx] > rank[ry]:
#                     parent[ry] = rx
#                 else:
#                     parent[ry] = rx
#                     rank[rx] += 1

#                 components -= 1
#                 return True

#             # Force/contract fixed edges.
#             for e in fixed:
#                 if e not in edge_attr:
#                     return float("inf")

#                 u, v = e
#                 if u not in parent or v not in parent:
#                     return float("inf")

#                 # Fixed edges creating a cycle means no tree can contain all of them.
#                 if not union(u, v):
#                     return float("inf")

#             # Force/contract Ssub.
#             for e in Sset:
#                 if e not in edge_attr:
#                     return float("inf")

#                 u, v = e
#                 if u not in parent or v not in parent:
#                     return float("inf")

#                 # If F+ union Ssub creates a cycle, no spanning tree can contain Ssub.
#                 if not union(u, v):
#                     return float("inf")

#             # Already connected after contracting fixed union Ssub.
#             if components == 1:
#                 return 0.0

#             total_completion_length = 0.0

#             # Complete with cheapest admissible edges by LENGTH.
#             # A already excludes fixed and excluded edges.
#             # We also exclude Ssub because those edges are already forced.
#             for e in A_sorted:
#                 if e in Sset:
#                     continue

#                 u, v = e
#                 if u not in parent or v not in parent:
#                     continue

#                 if union(u, v):
#                     total_completion_length += get_len(e)

#                     if components == 1:
#                         return total_completion_length

#             # Could not connect all components.
#             return float("inf")

#         # ------------------------------------------------------------
#         # Build residual-minimal seed cover
#         # ------------------------------------------------------------
#         def build_residual_minimal_cover(desc_edges):
#             """
#             Build a residual-minimal cover with respect to B'.

#             We add edges in nonincreasing length order until the residual budget
#             is exceeded, then prune shortest edges while the violation remains.
#             """
#             S = []
#             sL = 0.0

#             for e in desc_edges:
#                 if e not in edge_attr:
#                     continue

#                 S.append(e)
#                 sL += get_len(e)

#                 if sL > Bp + EPS:
#                     # Prune shortest edges while still violating B'.
#                     S.sort(key=lambda x: get_len(x))  # increasing length

#                     k = 0
#                     while k < len(S) and (sL - get_len(S[k]) > Bp + EPS):
#                         sL -= get_len(S[k])
#                         k += 1

#                     if k > 0:
#                         S = S[k:]

#                     return S, sL

#             return None, None

#         # ------------------------------------------------------------
#         # RHS and violation helpers
#         # ------------------------------------------------------------
#         def rhs_eff(cset):
#             """
#             Effective RHS after fixed-in edges.

#             For generated cuts at the current node, cset is usually a subset of A,
#             so this is normally |S|-1. This form is kept for safety.
#             """
#             return len(cset) - 1 - sum(1 for e in cset if e in fixed)

#         def is_violated_now(cset):
#             """
#             Check whether the current Lagrangian MST violates the cut.
#             """
#             lhs = sum(1 for e in cset if e in mst_set)
#             return lhs > rhs_eff(cset)

#         # ------------------------------------------------------------
#         # Exact completion certificate
#         # ------------------------------------------------------------
#         def cert_holds(Slist):
#             """
#             Exact tree-completion certificate.

#             The cut sum_{e in Slist} x_e <= |Slist|-1 is valid at this node if
#             every spanning-tree completion containing F+ union Slist violates
#             the residual budget.

#             We test this by computing the minimum additional LENGTH needed to
#             complete F+ union Slist to a spanning tree.
#             """
#             if not Slist or len(Slist) <= 1:
#                 return False

#             if rhs_eff(Slist) <= 0:
#                 return False

#             sumS = sum(get_len(e) for e in Slist)
#             completion = completion_mst_cost(Slist)

#             # If completion is impossible, then no feasible spanning tree can
#             # contain all edges in Slist, so the cut is valid.
#             if completion == float("inf"):
#                 return True

#             return (sumS + completion) > (Bp + EPS)

#         # ------------------------------------------------------------
#         # Inclusion-minimal shrinking under exact certificate
#         # ------------------------------------------------------------
#         def inclusion_minimal_shrink(Sstart):
#             """
#             Make S inclusion-minimal under cert_holds by removing one edge at a time.

#             This version scans removals from shortest to longest, matching the
#             current LaTeX description.
#             """
#             Sstar = sorted(Sstart, key=lambda e: get_len(e))  # shortest -> longest

#             changed = True
#             while changed and len(Sstar) > 1:
#                 changed = False

#                 for j in range(len(Sstar)):
#                     trial = Sstar[:j] + Sstar[j + 1:]

#                     if len(trial) <= 1:
#                         continue

#                     if cert_holds(trial):
#                         Sstar = sorted(trial, key=lambda e: get_len(e))
#                         changed = True
#                         break

#             return Sstar

#         # ------------------------------------------------------------
#         # Try completion-aware shrinking and add resulting cut
#         # ------------------------------------------------------------
#         def try_shrink_and_add(seed_S, seed_sumL):
#             """
#             Starting from a residual-minimal seed cover S, find a smaller set S'
#             that is still invalid under exact MST completion.

#             Procedure:
#             - Remove long edges until the set is no longer a simple residual cover.
#             - Test the exact completion certificate.
#             - Shrink to an inclusion-minimal certified set.
#             - Add the cut only if it separates the current Lagrangian MST.
#             """
#             if not seed_S or len(seed_S) <= 1:
#                 return

#             S_work = sorted(seed_S, key=lambda e: get_len(e), reverse=True)
#             sumL = float(seed_sumL)

#             # First candidate S': remove longest edges until sum length <= B'.
#             idx = 0
#             while idx < len(S_work) and sumL > Bp + EPS:
#                 sumL -= get_len(S_work[idx])
#                 idx += 1

#             Sprime = S_work[idx:]

#             if not Sprime or len(Sprime) <= 1:
#                 return

#             if not cert_holds(Sprime):
#                 return

#             Sstar = inclusion_minimal_shrink(Sprime)

#             if len(Sstar) <= 1:
#                 return

#             if is_violated_now(Sstar):
#                 cuts.append((set(Sstar), len(Sstar) - 1))

#         # ------------------------------------------------------------
#         # Safe residual-aware lifting
#         # ------------------------------------------------------------
#         def lift_minimal_cover(S_min, rhs_base):
#             """
#             Safe unit lifting for a residual-minimal cover S.

#             If length(f) > B' - length(S) + Lmax,
#             then f can be lifted with coefficient 1 while keeping the same RHS.
#             """
#             S_base = set(S_min)

#             if not S_base:
#                 return None

#             sumS = sum(get_len(e) for e in S_base)
#             Lmax = max(get_len(e) for e in S_base)
#             threshold = Bp - sumS + Lmax

#             lift_add = {
#                 f for f in A
#                 if f not in S_base and get_len(f) > threshold + EPS
#             }

#             if not lift_add:
#                 return None

#             S_lift = S_base | lift_add

#             # RHS remains rhs_base = |S_min|-1.
#             if rhs_eff(S_lift) > 0 and is_violated_now(S_lift):
#                 return (S_lift, rhs_base)

#             return None

#         # ============================================================
#         # Main separation logic
#         # ============================================================

#         # Primary seed from T^lambda intersect A.
#         T_desc = sorted(TcapA, key=lambda e: get_len(e), reverse=True)
#         S_seed, sumL_seed = build_residual_minimal_cover(T_desc)

#         if not S_seed:
#             return []

#         S_seed = list(S_seed)

#         # Add simple residual cover cut.
#         if rhs_eff(S_seed) > 0 and is_violated_now(S_seed):
#             cuts.append((set(S_seed), len(S_seed) - 1))

#         # Add exact completion-aware refined cut.
#         try_shrink_and_add(S_seed, sumL_seed)

#         # Add lifted residual-minimal seed cover.
#         lifted = lift_minimal_cover(S_seed, rhs_base=(len(S_seed) - 1))
#         if lifted is not None:
#             cuts.append(lifted)

#         # Optional micro-seed from globally heaviest admissible edges.
#         # Keep this only if you also mention it in the paper/implementation section.
#         if L_MICRO > 0 and len(A) > 0:
#             heavyA = sorted(A, key=lambda e: get_len(e), reverse=True)[:L_MICRO]
#             S2, sumL2 = build_residual_minimal_cover(heavyA)

#             if S2:
#                 S2set = set(S2)

#                 if (
#                     rhs_eff(S2set) > 0
#                     and S2set != set(S_seed)
#                     and is_violated_now(S2set)
#                 ):
#                     cuts.append((S2set, len(S2) - 1))

#                 try_shrink_and_add(S2, sumL2)

#                 lifted2 = lift_minimal_cover(S2, rhs_base=(len(S2) - 1))
#                 if lifted2 is not None:
#                     cuts.append(lifted2)

#         # ------------------------------------------------------------
#         # Deduplication and dominance filtering
#         # ------------------------------------------------------------
#         uniq = {}

#         for cset, rhs in cuts:
#             key = tuple(sorted(cset))
#             best = uniq.get(key)

#             if best is None or rhs < best[1] or (
#                 rhs == best[1] and len(cset) < len(best[0])
#             ):
#                 uniq[key] = (cset, rhs)

#         final = list(uniq.values())
#         final.sort(key=lambda t: (t[1], len(t[0])))

#         kept = []

#         for cset, rhs in final:
#             if rhs_eff(cset) <= 0:
#                 continue

#             dominated = any(
#                 dset <= cset and drhs <= rhs
#                 for dset, drhs in kept
#             )

#             if not dominated:
#                 kept.append((cset, rhs))

#         return kept[:MAX_RETURN]

    
    
#     def compute_modified_weights(self):

#         base = self.edge_weights.copy()
#         lam = max(0.0, min(getattr(self, "lmbda", 0.0), 1e4))
#         if lam:
#             base = base + lam * self.edge_lengths

#         # No cuts? return λ-priced base
#         if not (self.use_cover_cuts and self.best_cuts):
#             self._mw_cached = None
#             self._mw_lambda = lam
#             self._mw_mu = None
#             self._mw_free_mask_key = None
#             return base
#         cut_idxs_free = getattr(self, "_cut_edge_idx", None)  # FREE indices only

#         mu_len = len(self.best_cuts)

#         mu = np.array([max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4)) for i in range(mu_len)], dtype=float)

#         # Cache key: (λ, μ, free-mask signature)
#         _ = self._get_free_mask()
#         free_mask_key = self._free_mask_key

#         if (self._mw_cached is not None and
#             self._mw_lambda == lam and
#             self._mw_mu is not None and
#             self._mw_mu.shape == mu.shape and
#             np.allclose(self._mw_mu, mu, rtol=0, atol=0) and
#             self._mw_free_mask_key == free_mask_key):
#             return self._mw_cached

#         # Add μ to ALL edges that belong to each cut (fixed edges included)
#         weights = base.copy()

#         if cut_idxs_free is not None:
#             for i, idxs in enumerate(cut_idxs_free):
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
#         return weights


#     def _invalidate_weight_cache(self):
#         self._free_mask_cache = None
#         self._free_mask_key = None
#         self._mw_cached = None
#         self._mw_lambda = None
#         self._mw_mu = None
#         self._mw_free_mask_key = None

   
#     def _get_free_mask(self):

#         fixed = frozenset(self.fixed_edges)
#         forbidden = frozenset(getattr(self, "excluded_edges", set()))
#         key = (fixed, forbidden)
#         if self._free_mask_cache is not None and self._free_mask_key == key:
#             return self._free_mask_cache

#         if not fixed and not forbidden:
#             self._free_mask_cache = None
#             self._free_mask_key = key
#             return None

#         mask = np.ones(len(self.edge_list), dtype=bool)
#         for e in fixed | forbidden:
#             idx = self.edge_indices.get(e)
#             if idx is not None:
#                 mask[idx] = False

#         self._free_mask_cache = mask
#         self._free_mask_key = key
#         return mask


#     def _append_with_cap(self, bucket, item, cap):
#         bucket.append(item)
#         overflow = len(bucket) - cap
#         if overflow > 0:
#             del bucket[:overflow]

#     def _record_primal_solution(self, mst_edges, feasible):
#         # snapshot = tuple(sorted(mst_edges)) if mst_edges else ()
#         snapshot = tuple(mst_edges) if mst_edges else ()

#         self._append_with_cap(
#             self.primal_solutions,
#             (snapshot, bool(feasible)),
#             self._primal_history_cap,
#         )

    
#     def _record_fractional_solution(self, fractional_solution):
#         if not fractional_solution:
#             lightweight = ()
#         else:
#             lightweight = tuple(
#                 heapq.nlargest(20, fractional_solution.items(), key=lambda kv: abs(kv[1]))
#             )
#         self._append_with_cap(self.fractional_solutions, lightweight, self._fractional_history_cap)


#     def _record_subgradient(self, value):
#         self._append_with_cap(
#             self.subgradients,
#             float(value),
#             self._subgradient_history_cap,
#         )




#     def custom_kruskal(self, modified_weights):
#         uf = UnionFind(self.num_nodes)

#         mst_edges = []
#         mst_edge_indices = []   # <--- NEW: track indices
#         mst_cost = 0.0

#         # Add fixed edges first
#         for i in self.fixed_edge_indices:
#             u, v = self.edge_list[i]
#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 mst_edge_indices.append(i)          # <--- NEW
#                 mst_cost += modified_weights[i]
#             else:
#                 return float('inf'), float('inf'), []

#         # Remaining candidate edges (canonical size!)
#         m = len(self.edge_list)
#         candidates = [i for i in range(m)
#                     if i not in self.fixed_edge_indices and i not in self.excluded_edge_indices]

#         candidates.sort(key=lambda i: modified_weights[i])

#         for i in candidates:
#             u, v = self.edge_list[i]
#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 mst_edge_indices.append(i)          # <--- NEW
#                 mst_cost += modified_weights[i]
#                 if len(mst_edges) == self.num_nodes - 1:
#                     break

#         # Connectivity / size check
#         if len(mst_edges) != self.num_nodes - 1 or uf.count_components() > 1:
#             return float('inf'), float('inf'), []

#         # Length computed by indices (consistent with everything)
#         mst_length = float(np.sum(self.edge_lengths[mst_edge_indices]))

#         return mst_cost, mst_length, mst_edges

    
  
#     def incremental_kruskal(self, prev_weights, prev_mst_edges, current_weights):
#         uf = UnionFind(self.num_nodes)
#         mst_edges = []
#         mst_cost = 0.0
#         mst_length = 0.0

#         for edge_idx in self.fixed_edge_indices:
#             u, v = self.edge_list[edge_idx]
#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 mst_cost   += current_weights[edge_idx]
#                 mst_length += self.edge_lengths[edge_idx]
#             else:
#                 # Fixed edges already create a cycle -> infeasible
#                 return float('inf'), float('inf'), []

#         weight_changes = current_weights - prev_weights
#         changed_indices = np.where(np.abs(weight_changes) > self.cache_tolerance)[0]
#         changed_edges   = set(changed_indices)

#         prev_mst_indices = {
#             self.edge_indices[(u, v)] for u, v in prev_mst_edges
#             if self.edge_indices[(u, v)] not in self.fixed_edge_indices
#         }
#         candidate_indices = (
#             prev_mst_indices | changed_edges
#         ) - self.excluded_edge_indices - self.fixed_edge_indices

#         sorted_edges = sorted(candidate_indices, key=lambda i: current_weights[i])

#         for edge_idx in sorted_edges:
#             u, v = self.edge_list[edge_idx]
#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 mst_cost   += current_weights[edge_idx]
#                 mst_length += self.edge_lengths[edge_idx]

#         # NEW: cheap validity check – tree must have exactly n-1 edges
#         if len(mst_edges) != self.num_nodes - 1:
#             return float('inf'), float('inf'), []

#         return mst_cost, mst_length, mst_edges

    
#     def compute_mst(self, modified_edges=None):
#         start_time = time()
        
#         if modified_edges is not None:
#             weights = np.array([w for _, _, w in modified_edges], dtype=float)
#         else:
#             weights = self.compute_modified_weights()

#         weights = np.nan_to_num(
#             weights,
#             nan=0.0,
#             posinf=1e9,
#             neginf=-1e9,
#             copy=False,
#         )

#         # ---- SIMPLE VERSION: NO HASHING, NO CACHE ----
#         mst_cost, mst_length, mst_edges = self.custom_kruskal(weights)
#         if self.verbose:
#             print(f"MST computed (no cache): length={mst_length:.2f}")

#         # Optionally remember last MST if you use it elsewhere
#         self.last_mst_edges = mst_edges

#         end_time = time()
#         LagrangianMST.total_compute_time += end_time - start_time
#         return mst_cost, mst_length, mst_edges



#     def compute_mst_incremental(self, prev_weights, prev_mst_edges):
#         # Compute current modified weights ONCE
#         current_weights = self.compute_modified_weights()
#         # Cache them so the caller (solve) can reuse without recomputing
#         self._last_mw = current_weights

#         # First call or no previous MST: just run full Kruskal
#         if prev_weights is None or prev_mst_edges is None:
#             # if self.verbose:
#             #     print("Incremental MST: no previous MST, using full custom_kruskal")
#             return self.custom_kruskal(current_weights)

#         weight_changes = current_weights - prev_weights

#         if np.all(np.abs(weight_changes) < 1e-6):
#             mst_cost = sum(current_weights[self.edge_indices[(u, v)]]
#                            for u, v in prev_mst_edges)
#             mst_length = sum(self.edge_lengths[self.edge_indices[(u, v)]]
#                              for u, v in prev_mst_edges)
#             # if self.verbose:
#             #     print(f"Incremental MST: Reusing previous MST with length={mst_length:.2f}")
#             return mst_cost, mst_length, prev_mst_edges

#         # if self.verbose:
#         #     print(f"Incremental MST: Computing new MST due to weight changes")
#         return self.incremental_kruskal(prev_weights, prev_mst_edges, current_weights)



   

#     def solve(self, inherited_cuts=None, inherited_multipliers=None, depth=0, node=None):
#         start_time = time()
#         self.depth = depth
        
#         # --- robust normalization of inherited_cuts (accept pairs or indices) ---
#         edge_indices = self.edge_indices
#         idx_to_edge = {j: e for e, j in edge_indices.items()}

#         def _norm_edge(e):
#             if not (isinstance(e, tuple) and len(e) == 2):
#                 return None
#             u, v = e
#             t = (u, v) if u <= v else (v, u)
#             return t if t in edge_indices else None

#         def _iter_edges_any(cut_like):
#             if isinstance(cut_like, tuple) and len(cut_like) == 2:
#                 e = _norm_edge(cut_like); 
#                 if e is not None: yield e
#                 return
#             if isinstance(cut_like, int):
#                 e = _norm_edge(idx_to_edge.get(int(cut_like)))
#                 if e is not None: yield e
#                 return
#             try:
#                 for item in cut_like:
#                     if isinstance(item, int):
#                         e = _norm_edge(idx_to_edge.get(int(item)))
#                     elif isinstance(item, tuple) and len(item) == 2:
#                         e = _norm_edge(item)
#                     elif isinstance(item, (list, set, frozenset)) and len(item) == 2:
#                         a, b = tuple(item); e = _norm_edge((a, b))
#                     else:
#                         e = None
#                     if e is not None: 
#                         yield e
#             except TypeError:
#                 return

#         def _norm_pair(pair):
#             cut_like, rhs_like = pair
#             return (set(_iter_edges_any(cut_like)), int(rhs_like))

#         if inherited_cuts:
#             self.best_cuts = [_norm_pair(p) for p in inherited_cuts]
#             self.best_cut_multipliers = (inherited_multipliers or {}).copy()
#         else:
#             self.best_cuts = []
#             self.best_cut_multipliers = {}
#         self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()


#         # --- robust normalization of inherited_cuts (accept pairs or indices) ---
    
            
#         prev_weights = None
#         prev_mst_edges = None

       
#         if self.use_bisection:
#         # Validate graph and edges
#             if not self.edges or not nx.is_connected(self.graph):
#                 if self.verbose:
#                     print(f"Error at depth {depth}: Empty edge list or disconnected graph in bisection path")
#                 return self.best_lower_bound, self.best_upper_bound, []
            

#         # else:  # Subgradient method with Polyak hybrid + cover cuts (λ, μ), depth-based freezing
#         #     # --- Tunables / safety limits ---
#         #     MAX_SOLUTIONS    = getattr(self, "max_primal_solutions", 50)
#         #     max_iter         = min(self.max_iter, 200)

#         #     # Polyak / momentum for λ
#         #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
#         #     gamma_base         = getattr(self, "gamma_base", 0.1)

#         #     # μ update parameters
#         #     gamma_mu         = getattr(self, "gamma_mu", 0.30)
#         #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
#         #     eps              = 1e-12

#         #     # Depth-based behaviour
#         #     max_cut_depth = getattr(self, "max_cut_depth", 30)   # where we ADD cuts
#         #     max_mu_depth  = getattr(self, "max_mu_depth", 50)    # where we UPDATE μ / use cuts in dual
#         #     is_root       = (depth == 0)

#         #     # Node-level separation parameters
#         #     max_active_cuts           = getattr(self, "max_active_cuts", 5)
#         #     max_new_cuts_per_node     = getattr(self, "max_new_cuts_per_node", 5)
#         #     min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
#         #     dead_mu_threshold         = getattr(self, "dead_mu_threshold", 1e-6)

#         #     # Extra iterations allowed at root
#         #     root_max_iter = int(getattr(self, "root_max_iter", max_iter * 2))

#         #     # Ensure cut structures exist
#         #     if not hasattr(self, "best_cuts"):
#         #         self.best_cuts = []   # list of (set(edges), rhs)
#         #     if not hasattr(self, "best_cut_multipliers"):
#         #         self.best_cut_multipliers = {}  # μ_i for each cut
#         #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
#         #         self.best_cut_multipliers_for_best_bound = {}  # μ at best LB

#         #     # Which behaviour at this node?
#         #     cutting_active_here = self.use_cover_cuts and (depth <= max_cut_depth)   # can ADD cuts
#         #     mu_dynamic_here     = self.use_cover_cuts and (depth <= max_mu_depth)    # can UPDATE μ / use in dual
#         #     cuts_present_here   = self.use_cover_cuts and bool(self.best_cuts)

#         #     # Ensure λ starts in a reasonable range (consistent with compute_modified_weights)
#         #     self.lmbda = max(0.0, min(getattr(self, "lmbda", 0.05), 1e4))

#         #     polyak_enabled = True

#         #     # Collect newly generated cuts at this node
#         #     node_new_cuts = []

#         #     # --- Quick guards ---
#         #     if not self.edge_list or self.num_nodes <= 1:
#         #         if self.verbose:
#         #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
#         #         end_time = time()
#         #         LagrangianMST.total_compute_time += end_time - start_time
#         #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

#         #     # Fixed / forbidden edges
#         #     F_in  = getattr(self, "fixed_edges", set())
#         #     F_out = getattr(self, "excluded_edges", set())
#         #     edge_idx = self.edge_indices
#         #     if not hasattr(self, "_rhs_eff"):
#         #         self._rhs_eff = {}

#         #     # ------------------------------------------------------------------
#         #     # Separation policy (FIXED):
#         #     #   - DO NOT do objective-only pre-separation at root.
#         #     #   - Always delay separation to the first violating MST inside the loop.
#         #     #   - Still obey depth limits: only add cuts when cutting_active_here AND μ is dynamic.
#         #     # ------------------------------------------------------------------
#         #     pending_sep = bool(cutting_active_here and mu_dynamic_here)

#         #     # ------------------------------------------------------------------
#         #     # 2) Compute rhs_eff and detect infeasibility (fixed edges + cuts)
#         #     #    rhs_eff = rhs - |cut ∩ F_in|
#         #     # ------------------------------------------------------------------
#         #     if self.use_cover_cuts and self.best_cuts:
#         #         for idx_c, (cut, rhs) in enumerate(self.best_cuts):
#         #             rhs_eff = int(rhs) - len(cut & F_in)
#         #             self._rhs_eff[idx_c] = rhs_eff
#         #             if rhs_eff < 0:
#         #                 end_time = time()
#         #                 LagrangianMST.total_compute_time += end_time - start_time
#         #                 return float('inf'), self.best_upper_bound, node_new_cuts

#         #     # ------------------------------------------------------------------
#         #     # 3) Trim number of cuts (keep at most max_active_cuts)
#         #     # ------------------------------------------------------------------
#         #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
#         #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
#         #         if not parent_mu_map:
#         #             parent_mu_map = self.best_cut_multipliers

#         #         idx_and_cut = list(enumerate(self.best_cuts))
#         #         idx_and_cut.sort(
#         #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
#         #             reverse=True
#         #         )
#         #         idx_and_cut = idx_and_cut[:max_active_cuts]

#         #         new_cuts_list = []
#         #         new_mu       = {}
#         #         new_mu_best  = {}
#         #         new_rhs_eff  = {}

#         #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
#         #             new_cuts_list.append(cut_rhs)
#         #             new_mu[new_i]      = float(parent_mu_map.get(old_i, 0.0))
#         #             new_mu_best[new_i] = float(parent_mu_map.get(old_i, 0.0))
#         #             new_rhs_eff[new_i] = self._rhs_eff[old_i]

#         #         self.best_cuts = new_cuts_list
#         #         self.best_cut_multipliers = new_mu
#         #         self.best_cut_multipliers_for_best_bound = new_mu_best
#         #         self._rhs_eff = new_rhs_eff

#         #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

#         #     # ------------------------------------------------------------------
#         #     # 4) Build cut -> edge index arrays (for pricing/subgradients)
#         #     # ------------------------------------------------------------------
#         #     def _rebuild_cut_structures():
#         #         nonlocal cut_edge_idx_free, cut_edge_idx_all, rhs_eff_vec

#         #         cut_edge_idx_free = []
#         #         cut_edge_idx_all  = []

#         #         for cut, rhs in self.best_cuts:
#         #             idxs_free = [
#         #                 edge_idx[e] for e in cut
#         #                 if (e not in F_in and e not in F_out) and (e in edge_idx)
#         #             ]
#         #             arr_free = (
#         #                 np.fromiter(idxs_free, dtype=np.int32)
#         #                 if idxs_free else np.empty(0, dtype=np.int32)
#         #             )
#         #             cut_edge_idx_free.append(arr_free)

#         #             idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
#         #             arr_all  = (
#         #                 np.fromiter(idxs_all, dtype=np.int32)
#         #                 if idxs_all else np.empty(0, dtype=np.int32)
#         #             )
#         #             cut_edge_idx_all.append(arr_all)

#         #         self._cut_edge_idx     = cut_edge_idx_free
#         #         self._cut_edge_idx_all = cut_edge_idx_all

#         #         rhs_eff_vec = (
#         #             np.array([self._rhs_eff[i] for i in range(len(self.best_cuts))], dtype=float)
#         #             if self.best_cuts else np.zeros(0, dtype=float)
#         #         )

#         #     cut_edge_idx_free = []
#         #     cut_edge_idx_all  = []
#         #     rhs_eff_vec       = np.zeros(0, dtype=float)

#         #     if self.use_cover_cuts and self.best_cuts:
#         #         _rebuild_cut_structures()

#         #     # Track usefulness of cuts at this node
#         #     max_cut_violation = [0.0 for _ in self.best_cuts]

#         #     # Histories / caches
#         #     self._mw_cached = None
#         #     self._mw_lambda = None
#         #     self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)

#         #     if not hasattr(self, "subgradients"):
#         #         self.subgradients = []
#         #     if not hasattr(self, "step_sizes"):
#         #         self.step_sizes = []
#         #     if not hasattr(self, "multipliers"):
#         #         self.multipliers = []

#         #     prev_weights   = None
#         #     prev_mst_edges = None

#         #     if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
#         #         self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
#         #     mst_mask = self._mst_mask

#         #     # Decide iteration limit for this node:
#         #     if is_root:
#         #         iter_limit = root_max_iter * 1.1 if self.use_cover_cuts else root_max_iter
#         #     else:
#         #         iter_limit = max_iter
#         #     # ------------------------------------------------------------------
#         #     # 5) Subgradient iterations
#         #     # ------------------------------------------------------------------
#         #     for iter_num in range(int(iter_limit)):
#         #         # 1) MST with current λ, μ              
#         #         try:
#         #             mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
#         #         except Exception:
#         #             mst_cost, mst_length, mst_edges = self.compute_mst()

#         #         self.last_mst_edges = mst_edges
#         #         prev_mst_edges      = mst_edges
#         #         cut_g_signed = []

#         #         # 1a) ONE-SHOT delayed separation (root AND non-root)
#         #         if (
#         #             cutting_active_here
#         #             and mu_dynamic_here
#         #             and pending_sep
#         #             and len(self.best_cuts) < max_active_cuts
#         #             and mst_length > self.budget
#         #         ):
#         #             try:
#         #                 cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []
#         #                 print("sss")

#         #                 T_loop = set(mst_edges)
#         #                 scored_loop = []
#         #                 F_in_set = set(F_in)  # (already defined above)

#         #                 for cut, rhs in cand_cuts_loop:
#         #                     S_set   = set(cut)
#         #                     S_free  = S_set - F_in_set                 # remove fixed edges from LHS set
#         #                     lhs_free = len(T_loop & S_free)            # only MST edges that are NOT fixed
#         #                     rhs_eff  = int(rhs) - len(S_set & F_in_set)
#         #                     violation = lhs_free - rhs_eff

#         #                     if violation >= min_cut_violation_for_add:
#         #                         scored_loop.append((violation, S_set, rhs))

#         #                 scored_loop.sort(reverse=True, key=lambda t: t[0])

#         #                 remaining_slots = max(0, max_active_cuts - len(self.best_cuts))
#         #                 if remaining_slots > 0:
#         #                     scored_loop = scored_loop[:min(max_new_cuts_per_node, remaining_slots)]
#         #                 else:
#         #                     scored_loop = []

#         #                 existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}
#         #                 added_any = False

#         #                 for violation, S, rhs in scored_loop:
#         #                     fz = frozenset(S)
#         #                     if fz in existing:
#         #                         continue

#         #                     self.best_cuts.append((set(S), rhs))
#         #                     new_idx = len(self.best_cuts) - 1
#         #                     MU0 = getattr(self, "mu_init", 0.0)  # safe default: 0 (avoid immediate decay overhead)
#         #                     self.best_cut_multipliers[new_idx] = MU0
#         #                     self.best_cut_multipliers_for_best_bound[new_idx] = MU0


#         #                     # keep rhs_eff consistent
#         #                     self._rhs_eff[new_idx] = int(rhs) - len(set(S) & F_in)
#         #                     if self._rhs_eff[new_idx] < 0:
#         #                         end_time = time()
#         #                         LagrangianMST.total_compute_time += end_time - start_time
#         #                         return float('inf'), self.best_upper_bound, node_new_cuts

#         #                     max_cut_violation.append(0.0)
#         #                     node_new_cuts.append((set(S), rhs))
#         #                     added_any = True

#         #                 if added_any:
#         #                     _rebuild_cut_structures()
#         #                     self._mw_cached = None
#         #                     self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)
#         #                     cuts_present_here = True

#         #             except Exception as e:
#         #                 if self.verbose:
#         #                     print(f"Error in delayed separation at depth {depth}, iter {iter_num}: {e}")
#         #             finally:
#         #                 pending_sep = False  # do at most once per node

#         #         # Prepare weights for next iteration (cache)
#         #         prev_weights = getattr(self, "_last_mw", prev_weights)

#         #         # 2) Primal & UB
#         #         is_feasible = (mst_length <= self.budget)
#         #         self._record_primal_solution(self.last_mst_edges, is_feasible)

#         #         if is_feasible:
#         #             try:
#         #                 real_weight, real_length = self.compute_real_weight_length()
#         #                 if (
#         #                     not math.isnan(real_weight)
#         #                     and not math.isinf(real_weight)
#         #                     and real_weight < self.best_upper_bound
#         #                 ):
#         #                     self.best_upper_bound = real_weight
#         #             except Exception as e:
#         #                 if self.verbose:
#         #                     print(f"Error updating primal solution: {e}")

#         #         if len(self.primal_solutions) > MAX_SOLUTIONS:
#         #             self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

#         #         # 3) Dual value: L(λ, μ) = MST_cost - λ B - Σ μ_i rhs_eff_i
#         #         lam_for_dual = max(0.0, min(self.lmbda, 1e4))

#         #         if self.use_cover_cuts and len(rhs_eff_vec) > 0:
#         #             mu_vec = np.fromiter(
#         #                 (
#         #                     max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
#         #                     for i in range(len(rhs_eff_vec))
#         #                 ),
#         #                 dtype=float,
#         #                 count=len(rhs_eff_vec),
#         #             )
#         #             cover_cut_penalty = float(mu_vec @ rhs_eff_vec)
#         #         else:
#         #             cover_cut_penalty = 0.0

#         #         lagrangian_bound = mst_cost - lam_for_dual * self.budget - cover_cut_penalty
#         #         # if cover_cut_penalty != 0.0:
#         #             # print("ggg", cover_cut_penalty)
#         #         # print("lagrangian bound:", lagrangian_bound)

#         #         if (
#         #             not math.isnan(lagrangian_bound)
#         #             and not math.isinf(lagrangian_bound)
#         #             and abs(lagrangian_bound) < 1e10
#         #         ):
#         #             if lagrangian_bound > self.best_lower_bound + 1e-6:
#         #                 self.best_lower_bound = lagrangian_bound
#         #                 self.best_lambda      = lam_for_dual
#         #                 self.best_mst_edges   = self.last_mst_edges
#         #                 self.best_cost        = mst_cost
#         #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()

#         #         # 4) Subgradients
#         #         knapsack_subgradient = float(mst_length - self.budget)
#         #         # print("fff", mst_length)
#         #         # print("lala",self.lmbda)
#         #         # print("wer", knapsack_subgradient)

#         #         # Fast skip: if MST feasible and all μ are ~0, don't pay cut gradient cost
#         #         all_mu_small = (not self.best_cut_multipliers) or \
#         #                     (max(self.best_cut_multipliers.values()) <= dead_mu_threshold)

#         #         if cuts_present_here and mu_dynamic_here and len(cut_edge_idx_all) > 0 and not (is_feasible and all_mu_small):
#         #             mst_mask[:] = False
#         #             for e in mst_edges:
#         #                 j = self.edge_indices.get(e)
#         #                 if j is not None:
#         #                     mst_mask[j] = True

#         #             cut_g_signed = []
#         #             cut_g_pos    = []

#         #             for i, idxs_free in enumerate(cut_edge_idx_free):
#         #                 lhs_free = int(mst_mask[idxs_free].sum()) if idxs_free.size else 0
#         #                 g_i = float(lhs_free) - float(rhs_eff_vec[i])
#         #                 cut_g_signed.append(g_i)
#         #                 cut_g_pos.append(g_i if g_i > 0.0 else 0.0)

#         #                 if g_i > max_cut_violation[i]:
#         #                     max_cut_violation[i] = g_i

#         #             cut_subgradients = cut_g_pos
#         #         else:
#         #             cut_subgradients = []
#         #             cut_g_signed = []
#         #             cut_g_pos = []


#         #         norm_sq = knapsack_subgradient ** 2
#         #         for g in cut_subgradients:
#         #             norm_sq += float(g) ** 2

#         #         # Polyak step size
#         #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
#         #             gap   = max(0.0, self.best_upper_bound - lagrangian_bound)
#         #             alpha = gamma_base * gap / (norm_sq + eps)
#         #         else:
#         #             alpha = getattr(self, "step_size", 0.001)

#         #         # λ update with momentum, then clamp
#         #         v_prev = getattr(self, "_v_lambda", 0.0)
#         #         v_new  = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
#         #         self._v_lambda = v_new
#         #         self.lmbda     = self.lmbda + alpha * v_new
#         #         # print("ooo", alpha)

#         #         if self.lmbda < 0.0:
#         #             self.lmbda = 0.0
#         #         if self.lmbda > 1e4:
#         #             self.lmbda = 1e4

#         #         # μ updates: projected subgradient for constraints sum_{e in S} x_e <= rhs_eff
#         #         if mu_dynamic_here and len(cut_g_pos) > 0:
#         #             for i, g in enumerate(cut_g_pos):
#         #                 g = float(g)
#         #                 if g <= 0.0:
#         #                     continue

#         #                 delta = gamma_mu * alpha * g

#         #                 # cap only positive increment
#         #                 if mu_increment_cap is not None:
#         #                     delta = min(mu_increment_cap, delta)

#         #                 mu_old = float(self.best_cut_multipliers.get(i, 0.0))
#         #                 mu_new = mu_old + delta

#         #                 # projection + clamp
#         #                 if mu_new > 1e4:
#         #                     mu_new = 1e4

#         #                 self.best_cut_multipliers[i] = mu_new


#         #         self.step_sizes.append(alpha)
#         #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

#         #     # ------------------------------------------------------------------
#         #     # 6) Drop "dead" cuts globally
#         #     # ------------------------------------------------------------------
#         #     if self.use_cover_cuts and self.best_cuts and mu_dynamic_here:
#         #         keep_indices = []

#         #         parent_mu_map = getattr(
#         #             self,
#         #             "best_cut_multipliers_for_best_bound",
#         #             self.best_cut_multipliers,
#         #         )

#         #         for i, (cut, rhs) in enumerate(self.best_cuts):
#         #             mu_i    = float(self.best_cut_multipliers.get(i, 0.0))
#         #             mu_hist = float(parent_mu_map.get(i, 0.0))

#         #             ever_useful = (i < len(max_cut_violation) and max_cut_violation[i] > 0.0) \
#         #                         or (abs(mu_hist) >= dead_mu_threshold)

#         #             if (not ever_useful) and abs(mu_i) < dead_mu_threshold and abs(mu_hist) < dead_mu_threshold:
#         #                 continue
#         #             keep_indices.append(i)

#         #         if len(keep_indices) < len(self.best_cuts):
#         #             new_best_cuts = []
#         #             new_mu        = {}
#         #             new_mu_best   = {}
#         #             new_rhs_eff   = {}

#         #             for new_idx, old_idx in enumerate(keep_indices):
#         #                 new_best_cuts.append(self.best_cuts[old_idx])
#         #                 new_mu[new_idx]      = float(self.best_cut_multipliers.get(old_idx, 0.0))
#         #                 new_mu_best[new_idx] = float(self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0))
#         #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]

#         #             self.best_cuts = new_best_cuts
#         #             self.best_cut_multipliers = new_mu
#         #             self.best_cut_multipliers_for_best_bound = new_mu_best
#         #             self._rhs_eff = new_rhs_eff

#         #     # ------------------------------------------------------------------
#         #     # 7) Restore best (λ, μ) to pass to children
#         #     # ------------------------------------------------------------------
#         #     if hasattr(self, "best_lambda"):
#         #         self.lmbda = self.best_lambda

#         #     if mu_dynamic_here and hasattr(self, "best_cut_multipliers_for_best_bound"):
#         #         self.best_cut_multipliers = self.best_cut_multipliers_for_best_bound.copy()

#         #     end_time = time()
#         #     LagrangianMST.total_compute_time += end_time - start_time
#         #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        
#         else:  # Subgradient method with Polyak hybrid + cover cuts (λ, μ), depth-based freezing
#             import os

#             # --- Tunables / safety limits ---
#             MAX_SOLUTIONS = getattr(self, "max_primal_solutions", 50)
#             max_iter = min(self.max_iter, 200)

#             # Polyak / momentum for λ
#             self.momentum_beta = getattr(self, "momentum_beta", 0.7)
#             gamma_base = getattr(self, "gamma_base", 0.05)

#             # Safety controls for λ update
#             fallback_alpha = getattr(self, "fallback_alpha", 1e-5)
#             max_lambda_delta = getattr(self, "max_lambda_delta", 0.02)

#             # μ update parameters
#             gamma_mu = getattr(self, "gamma_mu", 0.25)
#             mu_increment_cap = getattr(self, "mu_increment_cap", 0.002)

#             eps = 1e-12

#             # Depth-based behaviour
#             max_cut_depth = getattr(self, "max_cut_depth", 30)
#             max_mu_depth = getattr(self, "max_mu_depth", 50)
#             is_root = depth == 0

#             # Node-level separation parameters
#             max_active_cuts = getattr(self, "max_active_cuts", 5)
#             max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)
#             min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
#             dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-6)

#             # Extra iterations allowed at root
#             root_max_iter = int(getattr(self, "root_max_iter", max_iter * 2))

#             # ------------------------------------------------------------------
#             # DEBUG SETTINGS
#             # ------------------------------------------------------------------
#             debug_cuts = False
#             debug_iter_every = 1       # change to 5 or 10 if the log becomes too large
#             debug_cut_max_rows = 10

#             debug_log_path = getattr(
#                 self,
#                 "debug_cut_log_path",
#                 os.path.join(os.path.expanduser("~/Desktop"), "cut_debug_log.txt"),
#             )

#             # Clear the log only once at the root node
#             if depth == 0:
#                 with open(debug_log_path, "w") as f:
#                     f.write("CUT DEBUG LOG\n")
#                     f.write("=" * 100 + "\n")

#             def _dbg(msg, iter_num=None, force=False):
#                 if not debug_cuts:
#                     return

#                 if iter_num is not None and not force:
#                     if iter_num % debug_iter_every != 0:
#                         return

#                 if iter_num is None:
#                     line = f"[CUTDBG depth={depth}] {msg}"
#                 else:
#                     line = f"[CUTDBG depth={depth} iter={iter_num}] {msg}"

#                 with open(debug_log_path, "a") as f:
#                     f.write(line + "\n")

#             def _edge_len(e):
#                 try:
#                     return float(self.edge_lengths[self.edge_indices[e]])
#                 except Exception:
#                     return float("nan")

#             def _cut_len(cut):
#                 return sum(_edge_len(e) for e in cut if e in self.edge_indices)

#             def _cut_repr(cut, max_edges=6):
#                 cut_list = sorted(list(cut))
#                 shown = cut_list[:max_edges]
#                 suffix = "" if len(cut_list) <= max_edges else f", ... +{len(cut_list) - max_edges}"
#                 return f"{shown}{suffix}"

#             def _print_cut_table(stage, iter_num=None, force=False):
#                 if not debug_cuts:
#                     return

#                 if iter_num is not None and not force:
#                     if iter_num % debug_iter_every != 0:
#                         return

#                 _dbg(
#                     f"{stage}: active cuts = {len(getattr(self, 'best_cuts', []))}",
#                     iter_num,
#                     force,
#                 )

#                 if not getattr(self, "best_cuts", []):
#                     return

#                 for i, (cut, rhs) in enumerate(self.best_cuts[:debug_cut_max_rows]):
#                     mu = float(getattr(self, "best_cut_multipliers", {}).get(i, 0.0))
#                     mu_best = float(getattr(self, "best_cut_multipliers_for_best_bound", {}).get(i, 0.0))
#                     rhs_eff = getattr(self, "_rhs_eff", {}).get(i, rhs)

#                     _dbg(
#                         f"  cut[{i}] size={len(cut)} rhs={rhs} rhs_eff={rhs_eff} "
#                         f"mu={mu:.6g} mu_best={mu_best:.6g} "
#                         f"len_sum={_cut_len(cut):.3f} edges={_cut_repr(cut)}",
#                         iter_num,
#                         force,
#                     )

#                 if len(self.best_cuts) > debug_cut_max_rows:
#                     _dbg(
#                         f"  ... {len(self.best_cuts) - debug_cut_max_rows} more cuts not shown",
#                         iter_num,
#                         force,
#                     )

#             # ------------------------------------------------------------------
#             # Ensure cut structures exist
#             # ------------------------------------------------------------------
#             if not hasattr(self, "best_cuts"):
#                 self.best_cuts = []

#             if not hasattr(self, "best_cut_multipliers"):
#                 self.best_cut_multipliers = {}

#             if not hasattr(self, "best_cut_multipliers_for_best_bound"):
#                 self.best_cut_multipliers_for_best_bound = {}

#             # Which behaviour at this node?
#             cutting_active_here = self.use_cover_cuts and depth <= max_cut_depth
#             mu_dynamic_here = self.use_cover_cuts and depth <= max_mu_depth
#             use_cuts_in_dual_here = self.use_cover_cuts and bool(self.best_cuts)

#             # Ensure λ starts in a reasonable range
#             self.lmbda = max(0.0, min(getattr(self, "lmbda", 0.05), 1e4))

#             polyak_enabled = True
#             node_new_cuts = []

#             _dbg(
#                 f"START NODE | use_cover_cuts={self.use_cover_cuts}, "
#                 f"cutting_active_here={cutting_active_here}, "
#                 f"mu_dynamic_here={mu_dynamic_here}, "
#                 f"use_cuts_in_dual_here={use_cuts_in_dual_here}, "
#                 f"lambda_start={self.lmbda:.6g}, "
#                 f"inherited_cuts={len(self.best_cuts)}, "
#                 f"log_file={debug_log_path}",
#                 force=True,
#             )

#             _print_cut_table("Inherited cuts before reduction", force=True)

#             # ------------------------------------------------------------------
#             # Quick guards
#             # ------------------------------------------------------------------
#             if not self.edge_list or self.num_nodes <= 1:
#                 _dbg("STOP: empty edge list or invalid graph", force=True)

#                 end_time = time()
#                 LagrangianMST.total_compute_time += end_time - start_time
#                 return self.best_lower_bound, self.best_upper_bound, node_new_cuts

#             # Fixed / forbidden edges
#             F_in = set(getattr(self, "fixed_edges", set()))
#             F_out = set(getattr(self, "excluded_edges", set()))
#             edge_idx = self.edge_indices

#             self._rhs_eff = {}

#             _dbg(
#                 f"Node fixings: |F_in|={len(F_in)}, |F_out|={len(F_out)}, "
#                 f"fixed_length={sum(_edge_len(e) for e in F_in if e in edge_idx):.3f}, "
#                 f"budget={self.budget:.3f}",
#                 force=True,
#             )

#             # ------------------------------------------------------------------
#             # 2) Reduce inherited cuts and remove redundant cuts
#             # ------------------------------------------------------------------
#             if self.use_cover_cuts and self.best_cuts:
#                 old_mu = dict(getattr(self, "best_cut_multipliers", {}) or {})
#                 old_mu_best = dict(getattr(self, "best_cut_multipliers_for_best_bound", {}) or {})

#                 reduced_cuts = []
#                 reduced_mu = {}
#                 reduced_mu_best = {}
#                 reduced_rhs_eff = {}

#                 kept_count = 0
#                 redundant_count = 0

#                 for old_i, (cut, rhs) in enumerate(self.best_cuts):
#                     S = set(cut)

#                     S_fixed = S & F_in
#                     S_excluded = S & F_out
#                     S_free = S - F_in - F_out
#                     rhs_eff = int(rhs) - len(S_fixed)

#                     _dbg(
#                         f"Reduce old_cut[{old_i}]: old_size={len(S)}, old_rhs={rhs}, "
#                         f"|S_fixed|={len(S_fixed)}, |S_excluded|={len(S_excluded)}, "
#                         f"|S_free|={len(S_free)}, rhs_eff={rhs_eff}, "
#                         f"mu_old={float(old_mu.get(old_i, 0.0)):.6g}",
#                         force=True,
#                     )

#                     if rhs_eff < 0:
#                         _dbg(
#                             f"STOP: inherited cut[{old_i}] makes node infeasible "
#                             f"because rhs_eff={rhs_eff}<0",
#                             force=True,
#                         )

#                         end_time = time()
#                         LagrangianMST.total_compute_time += end_time - start_time
#                         return float("inf"), self.best_upper_bound, node_new_cuts

#                     # Redundant at this node
#                     if len(S_free) <= rhs_eff:
#                         redundant_count += 1
#                         _dbg(
#                             f"Drop old_cut[{old_i}] as redundant: "
#                             f"|S_free|={len(S_free)} <= rhs_eff={rhs_eff}",
#                             force=True,
#                         )
#                         continue

#                     new_i = len(reduced_cuts)
#                     reduced_cuts.append((set(S_free), int(rhs_eff)))

#                     mu_val = float(old_mu.get(old_i, 0.0))
#                     mu_best_val = float(old_mu_best.get(old_i, mu_val))

#                     reduced_mu[new_i] = mu_val
#                     reduced_mu_best[new_i] = mu_best_val
#                     reduced_rhs_eff[new_i] = int(rhs_eff)
#                     kept_count += 1

#                 self.best_cuts = reduced_cuts
#                 self.best_cut_multipliers = reduced_mu
#                 self.best_cut_multipliers_for_best_bound = reduced_mu_best
#                 self._rhs_eff = reduced_rhs_eff

#                 _dbg(
#                     f"Cut reduction summary: kept={kept_count}, "
#                     f"redundant_dropped={redundant_count}",
#                     force=True,
#                 )

#             _print_cut_table("Cuts after reduction", force=True)

#             # ------------------------------------------------------------------
#             # 3) Trim number of cuts
#             # ------------------------------------------------------------------
#             if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
#                 parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)

#                 if not parent_mu_map:
#                     parent_mu_map = self.best_cut_multipliers

#                 idx_and_cut = list(enumerate(self.best_cuts))
#                 idx_and_cut.sort(
#                     key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
#                     reverse=True,
#                 )

#                 kept_old_indices = [old_i for old_i, _ in idx_and_cut[:max_active_cuts]]
#                 dropped_old_indices = [old_i for old_i, _ in idx_and_cut[max_active_cuts:]]

#                 _dbg(
#                     f"Trim cuts: max_active_cuts={max_active_cuts}, "
#                     f"kept_old_indices={kept_old_indices}, "
#                     f"dropped_old_indices={dropped_old_indices}",
#                     force=True,
#                 )

#                 idx_and_cut = idx_and_cut[:max_active_cuts]

#                 new_cuts_list = []
#                 new_mu = {}
#                 new_mu_best = {}
#                 new_rhs_eff = {}

#                 for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
#                     new_cuts_list.append(cut_rhs)
#                     new_mu[new_i] = float(self.best_cut_multipliers.get(old_i, 0.0))
#                     new_mu_best[new_i] = float(parent_mu_map.get(old_i, new_mu[new_i]))
#                     new_rhs_eff[new_i] = int(self._rhs_eff.get(old_i, cut_rhs[1]))

#                 self.best_cuts = new_cuts_list
#                 self.best_cut_multipliers = new_mu
#                 self.best_cut_multipliers_for_best_bound = new_mu_best
#                 self._rhs_eff = new_rhs_eff

#             cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)
#             use_cuts_in_dual_here = self.use_cover_cuts and bool(self.best_cuts)

#             _print_cut_table("Cuts after trimming", force=True)

#             # ------------------------------------------------------------------
#             # 4) Build cut -> edge index arrays
#             # ------------------------------------------------------------------
#             def _rebuild_cut_structures():
#                 nonlocal cut_edge_idx_free, cut_edge_idx_all, rhs_eff_vec

#                 cut_edge_idx_free = []
#                 cut_edge_idx_all = []

#                 for i, (cut, rhs) in enumerate(self.best_cuts):
#                     S = set(cut)

#                     idxs_free = [
#                         edge_idx[e]
#                         for e in S
#                         if e in edge_idx and e not in F_in and e not in F_out
#                     ]

#                     arr_free = (
#                         np.fromiter(idxs_free, dtype=np.int32)
#                         if idxs_free
#                         else np.empty(0, dtype=np.int32)
#                     )

#                     cut_edge_idx_free.append(arr_free)

#                     idxs_all = [edge_idx[e] for e in S if e in edge_idx]

#                     arr_all = (
#                         np.fromiter(idxs_all, dtype=np.int32)
#                         if idxs_all
#                         else np.empty(0, dtype=np.int32)
#                     )

#                     cut_edge_idx_all.append(arr_all)

#                     if i not in self._rhs_eff:
#                         self._rhs_eff[i] = int(rhs)

#                 self._cut_edge_idx = cut_edge_idx_free
#                 self._cut_edge_idx_all = cut_edge_idx_all

#                 rhs_eff_vec = (
#                     np.array(
#                         [self._rhs_eff[i] for i in range(len(self.best_cuts))],
#                         dtype=float,
#                     )
#                     if self.best_cuts
#                     else np.zeros(0, dtype=float)
#                 )

#                 _dbg(
#                     f"Rebuilt cut structures: num_cuts={len(self.best_cuts)}, "
#                     f"rhs_eff_vec={rhs_eff_vec.tolist()}, "
#                     f"free_edge_counts={[len(a) for a in cut_edge_idx_free]}",
#                     force=True,
#                 )

#             cut_edge_idx_free = []
#             cut_edge_idx_all = []
#             rhs_eff_vec = np.zeros(0, dtype=float)

#             if self.use_cover_cuts and self.best_cuts:
#                 _rebuild_cut_structures()

#             max_cut_violation = [0.0 for _ in self.best_cuts]

#             # Histories / caches
#             self._mw_cached = None
#             self._mw_lambda = None
#             self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

#             if not hasattr(self, "subgradients"):
#                 self.subgradients = []

#             if not hasattr(self, "step_sizes"):
#                 self.step_sizes = []

#             if not hasattr(self, "multipliers"):
#                 self.multipliers = []

#             prev_weights = None
#             prev_mst_edges = None

#             if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
#                 self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)

#             mst_mask = self._mst_mask

#             # Decide iteration limit for this node
#             if is_root:
#                 iter_limit = root_max_iter * 1.1 if self.use_cover_cuts else root_max_iter
#             else:
#                 iter_limit = max_iter

#             sep_rounds = 0
#             max_sep_rounds = 1
#             separate_every = 5

#             _dbg(
#                 f"Iteration setup: iter_limit={int(iter_limit)}, "
#                 f"max_sep_rounds={max_sep_rounds}, "
#                 f"separate_every={separate_every}",
#                 force=True,
#             )

#             # ------------------------------------------------------------------
#             # 5) Subgradient iterations
#             # ------------------------------------------------------------------
#             for iter_num in range(int(iter_limit)):
#                 # --------------------------------------------------------------
#                 # 5.1) MST with current λ and μ
#                 # --------------------------------------------------------------
#                 try:
#                     mst_cost, mst_length, mst_edges = self.compute_mst_incremental(
#                         prev_weights,
#                         prev_mst_edges,
#                     )
#                     mst_method = "incremental"

#                 except Exception as e:
#                     _dbg(
#                         f"Incremental MST failed: {e}. Falling back to full MST.",
#                         iter_num,
#                         force=True,
#                     )

#                     mst_cost, mst_length, mst_edges = self.compute_mst()
#                     mst_method = "full"

#                 if (
#                     not mst_edges
#                     or math.isinf(mst_cost)
#                     or math.isinf(mst_length)
#                     or math.isnan(mst_cost)
#                     or math.isnan(mst_length)
#                 ):
#                     _dbg(
#                         f"STOP: invalid MST. method={mst_method}, "
#                         f"mst_cost={mst_cost}, mst_length={mst_length}, "
#                         f"num_edges={len(mst_edges) if mst_edges else 0}",
#                         iter_num,
#                         force=True,
#                     )

#                     end_time = time()
#                     LagrangianMST.total_compute_time += end_time - start_time
#                     return float("inf"), self.best_upper_bound, node_new_cuts

#                 self.last_mst_edges = mst_edges
#                 prev_mst_edges = mst_edges

#                 _dbg(
#                     f"MST: method={mst_method}, cost={mst_cost:.6g}, "
#                     f"length={mst_length:.6g}, budget={self.budget:.6g}, "
#                     f"budget_violation={mst_length - self.budget:.6g}, "
#                     f"num_edges={len(mst_edges)}",
#                     iter_num,
#                 )

#                 # --------------------------------------------------------------
#                 # 5.2) Delayed separation
#                 #
#                 # Modified:
#                 # We now separate at the first budget-violating MST, not only at
#                 # iteration 0 or multiples of separate_every.
#                 # max_sep_rounds still limits this to one separation round per node.
#                 # --------------------------------------------------------------
#                 should_separate = (
#                     cutting_active_here
#                     and mu_dynamic_here
#                     and sep_rounds < max_sep_rounds
#                     and len(self.best_cuts) < max_active_cuts
#                     and mst_length > self.budget
#                 )

#                 _dbg(
#                     f"Separation check: should_separate={should_separate}, "
#                     f"sep_rounds={sep_rounds}/{max_sep_rounds}, "
#                     f"active_cuts={len(self.best_cuts)}/{max_active_cuts}, "
#                     f"budget_violated={mst_length > self.budget}",
#                     iter_num,
#                 )

#                 if should_separate:
#                     try:
#                         cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []

#                         _dbg(
#                             f"Generated candidate cuts: count={len(cand_cuts_loop)}",
#                             iter_num,
#                             force=True,
#                         )

#                         T_loop = set(mst_edges)
#                         scored_loop = []

#                         for cand_i, (cut, rhs) in enumerate(cand_cuts_loop):
#                             S_set = set(cut)

#                             S_fixed = S_set & F_in
#                             S_excluded = S_set & F_out
#                             S_free = S_set - F_in - F_out
#                             rhs_eff_new = int(rhs) - len(S_fixed)

#                             if rhs_eff_new < 0:
#                                 _dbg(
#                                     f"STOP: candidate cut[{cand_i}] gives "
#                                     f"rhs_eff_new={rhs_eff_new}<0",
#                                     iter_num,
#                                     force=True,
#                                 )

#                                 end_time = time()
#                                 LagrangianMST.total_compute_time += end_time - start_time
#                                 return float("inf"), self.best_upper_bound, node_new_cuts

#                             if len(S_free) <= rhs_eff_new:
#                                 _dbg(
#                                     f"Candidate cut[{cand_i}] dropped as redundant: "
#                                     f"|S_free|={len(S_free)} <= rhs_eff={rhs_eff_new}",
#                                     iter_num,
#                                     force=True,
#                                 )
#                                 continue

#                             lhs_free = len(T_loop & S_free)
#                             violation = lhs_free - rhs_eff_new

#                             _dbg(
#                                 f"Candidate cut[{cand_i}]: orig_size={len(S_set)}, "
#                                 f"|fixed|={len(S_fixed)}, "
#                                 f"|excluded|={len(S_excluded)}, "
#                                 f"|free|={len(S_free)}, "
#                                 f"rhs={rhs}, rhs_eff={rhs_eff_new}, "
#                                 f"lhs_on_current_MST={lhs_free}, "
#                                 f"violation={violation}, "
#                                 f"len_sum={_cut_len(S_free):.3f}",
#                                 iter_num,
#                                 force=True,
#                             )

#                             if violation >= min_cut_violation_for_add:
#                                 scored_loop.append(
#                                     (float(violation), set(S_free), int(rhs_eff_new))
#                                 )

#                         scored_loop.sort(
#                             reverse=True,
#                             key=lambda t: (t[0], len(t[1])),
#                         )

#                         remaining_slots = max(0, max_active_cuts - len(self.best_cuts))

#                         if remaining_slots > 0:
#                             scored_loop = scored_loop[
#                                 : min(max_new_cuts_per_node, remaining_slots)
#                             ]
#                         else:
#                             scored_loop = []

#                         _dbg(
#                             f"Candidate cuts after filtering: "
#                             f"kept_for_addition={len(scored_loop)}, "
#                             f"remaining_slots={remaining_slots}",
#                             iter_num,
#                             force=True,
#                         )

#                         existing = {
#                             frozenset(c): (i, int(rhs))
#                             for i, (c, rhs) in enumerate(self.best_cuts)
#                         }

#                         changed_any = False

#                         for violation, S, rhs in scored_loop:
#                             fz = frozenset(S)

#                             if fz in existing:
#                                 old_i, old_rhs = existing[fz]

#                                 if rhs < old_rhs:
#                                     _dbg(
#                                         f"Replace duplicate cut at index {old_i}: "
#                                         f"old_rhs={old_rhs}, new_rhs={rhs}, "
#                                         f"violation={violation}",
#                                         iter_num,
#                                         force=True,
#                                     )

#                                     self.best_cuts[old_i] = (set(S), int(rhs))
#                                     self._rhs_eff[old_i] = int(rhs)
#                                     max_cut_violation[old_i] = max(
#                                         max_cut_violation[old_i],
#                                         violation,
#                                     )
#                                     changed_any = True

#                                 else:
#                                     _dbg(
#                                         f"Skip duplicate cut: existing_rhs={old_rhs}, "
#                                         f"new_rhs={rhs}, violation={violation}",
#                                         iter_num,
#                                         force=True,
#                                     )

#                                 continue

#                             self.best_cuts.append((set(S), int(rhs)))
#                             new_idx = len(self.best_cuts) - 1

#                             # Positive initial μ makes a newly added cut affect the next MST.
#                             MU0 = getattr(self, "mu_init", 0.001)

#                             self.best_cut_multipliers[new_idx] = float(MU0)
#                             self.best_cut_multipliers_for_best_bound[new_idx] = float(MU0)
#                             self._rhs_eff[new_idx] = int(rhs)

#                             max_cut_violation.append(max(0.0, violation))
#                             node_new_cuts.append((set(S), int(rhs)))

#                             existing[fz] = (new_idx, int(rhs))
#                             changed_any = True

#                             _dbg(
#                                 f"ADD cut[{new_idx}]: size={len(S)}, rhs={rhs}, "
#                                 f"initial_mu={MU0}, violation={violation}, "
#                                 f"len_sum={_cut_len(S):.3f}, edges={_cut_repr(S)}",
#                                 iter_num,
#                                 force=True,
#                             )

#                         if changed_any:
#                             _rebuild_cut_structures()

#                             self._mw_cached = None
#                             self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

#                             cuts_present_here = True
#                             use_cuts_in_dual_here = self.use_cover_cuts and bool(self.best_cuts)

#                             _print_cut_table(
#                                 "Cuts after separation/addition",
#                                 iter_num,
#                                 force=True,
#                             )

#                     except Exception as e:
#                         _dbg(
#                             f"ERROR in delayed separation at depth={depth}, "
#                             f"iter={iter_num}: {e}",
#                             iter_num,
#                             force=True,
#                         )

#                     finally:
#                         sep_rounds += 1

#                 # Prepare weights for next iteration
#                 prev_weights = getattr(self, "_last_mw", prev_weights)

#                 # --------------------------------------------------------------
#                 # 5.3) Primal and upper bound
#                 # --------------------------------------------------------------
#                 is_feasible = mst_length <= self.budget

#                 self._record_primal_solution(self.last_mst_edges, is_feasible)

#                 if is_feasible:
#                     try:
#                         real_weight, real_length = self.compute_real_weight_length()

#                         if (
#                             not math.isnan(real_weight)
#                             and not math.isinf(real_weight)
#                             and real_weight < self.best_upper_bound
#                         ):
#                             old_ub = self.best_upper_bound
#                             self.best_upper_bound = real_weight

#                             _dbg(
#                                 f"UB improved: old_UB={old_ub}, "
#                                 f"new_UB={self.best_upper_bound:.6g}, "
#                                 f"real_length={real_length:.6g}",
#                                 iter_num,
#                                 force=True,
#                             )

#                     except Exception as e:
#                         _dbg(
#                             f"ERROR updating primal solution: {e}",
#                             iter_num,
#                             force=True,
#                         )

#                 if len(self.primal_solutions) > MAX_SOLUTIONS:
#                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

#                 # --------------------------------------------------------------
#                 # 5.4) Dual value
#                 # --------------------------------------------------------------
#                 lam_for_dual = max(0.0, min(self.lmbda, 1e4))

#                 if use_cuts_in_dual_here and len(rhs_eff_vec) > 0:
#                     mu_vec = np.fromiter(
#                         (
#                             max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
#                             for i in range(len(rhs_eff_vec))
#                         ),
#                         dtype=float,
#                         count=len(rhs_eff_vec),
#                     )

#                     cover_cut_penalty = float(mu_vec @ rhs_eff_vec)

#                 else:
#                     mu_vec = np.zeros(0, dtype=float)
#                     cover_cut_penalty = 0.0

#                 lagrangian_bound = (
#                     mst_cost
#                     - lam_for_dual * self.budget
#                     - cover_cut_penalty
#                 )

#                 _dbg(
#                     f"Dual: lambda={lam_for_dual:.6g}, "
#                     f"mst_cost={mst_cost:.6g}, "
#                     f"lambdaB={lam_for_dual * self.budget:.6g}, "
#                     f"cover_penalty={cover_cut_penalty:.6g}, "
#                     f"LB_candidate={lagrangian_bound:.6g}, "
#                     f"best_LB_before={self.best_lower_bound:.6g}, "
#                     f"UB={self.best_upper_bound}",
#                     iter_num,
#                 )

#                 if len(mu_vec) > 0:
#                     _dbg(
#                         f"mu_vec={mu_vec.tolist()}, "
#                         f"rhs_eff_vec={rhs_eff_vec.tolist()}",
#                         iter_num,
#                     )

#                 if (
#                     not math.isnan(lagrangian_bound)
#                     and not math.isinf(lagrangian_bound)
#                     and abs(lagrangian_bound) < 1e10
#                 ):
#                     if lagrangian_bound > self.best_lower_bound + 1e-6:
#                         old_lb = self.best_lower_bound

#                         self.best_lower_bound = lagrangian_bound
#                         self.best_lambda = lam_for_dual
#                         self.best_mst_edges = self.last_mst_edges
#                         self.best_cost = mst_cost
#                         self.best_cut_multipliers_for_best_bound = (
#                             self.best_cut_multipliers.copy()
#                         )

#                         _dbg(
#                             f"LB improved: old_LB={old_lb:.6g}, "
#                             f"new_LB={self.best_lower_bound:.6g}, "
#                             f"best_lambda={self.best_lambda:.6g}, "
#                             f"saved_mu={self.best_cut_multipliers_for_best_bound}",
#                             iter_num,
#                             force=True,
#                         )

#                 # --------------------------------------------------------------
#                 # 5.5) Subgradients
#                 # --------------------------------------------------------------
#                 knapsack_subgradient = float(mst_length - self.budget)

#                 all_mu_small = (
#                     not self.best_cut_multipliers
#                     or max(self.best_cut_multipliers.values()) <= dead_mu_threshold
#                 )

#                 if (
#                     cuts_present_here
#                     and mu_dynamic_here
#                     and len(cut_edge_idx_free) > 0
#                     and not (is_feasible and all_mu_small)
#                 ):
#                     mst_mask[:] = False

#                     for e in mst_edges:
#                         j = self.edge_indices.get(e)
#                         if j is not None:
#                             mst_mask[j] = True

#                     cut_g_signed = []
#                     cut_g_pos = []

#                     for i, idxs_free in enumerate(cut_edge_idx_free):
#                         lhs_free = int(mst_mask[idxs_free].sum()) if idxs_free.size else 0
#                         g_i = float(lhs_free) - float(rhs_eff_vec[i])

#                         cut_g_signed.append(g_i)
#                         cut_g_pos.append(g_i if g_i > 0.0 else 0.0)

#                         if i < len(max_cut_violation) and g_i > max_cut_violation[i]:
#                             max_cut_violation[i] = g_i

#                         _dbg(
#                             f"Cut subgradient cut[{i}]: lhs_free={lhs_free}, "
#                             f"rhs_eff={rhs_eff_vec[i]}, "
#                             f"g_signed={g_i}, "
#                             f"g_pos={cut_g_pos[-1]}, "
#                             f"mu_before={self.best_cut_multipliers.get(i, 0.0):.6g}",
#                             iter_num,
#                         )

#                     # Modified:
#                     # Use the signed cut subgradient in the norm and μ update.
#                     # This allows μ to decrease when the cut becomes slack.
#                     cut_subgradients = cut_g_signed

#                 else:
#                     cut_g_signed = []
#                     cut_g_pos = []
#                     cut_subgradients = []

#                     _dbg(
#                         f"Skip cut subgradients: cuts_present={cuts_present_here}, "
#                         f"mu_dynamic={mu_dynamic_here}, "
#                         f"num_cut_arrays={len(cut_edge_idx_free)}, "
#                         f"is_feasible={is_feasible}, "
#                         f"all_mu_small={all_mu_small}",
#                         iter_num,
#                     )

#                 norm_sq = knapsack_subgradient ** 2

#                 for g in cut_subgradients:
#                     norm_sq += float(g) ** 2

#                 # --------------------------------------------------------------
#                 # 5.6) Polyak step size
#                 # --------------------------------------------------------------
#                 if (
#                     polyak_enabled
#                     and self.best_upper_bound < float("inf")
#                     and norm_sq > 0.0
#                 ):
#                     gap = max(0.0, self.best_upper_bound - lagrangian_bound)
#                     alpha = gamma_base * gap / (norm_sq + eps)
#                 else:
#                     gap = None
#                     # Before we have a finite UB, avoid the huge first lambda jump.
#                     alpha = fallback_alpha

#                 _dbg(
#                     f"Step: knapsack_g={knapsack_subgradient:.6g}, "
#                     f"cut_g_signed={cut_g_signed}, "
#                     f"cut_g_pos={cut_g_pos}, "
#                     f"norm_sq={norm_sq:.6g}, "
#                     f"gap={gap}, "
#                     f"alpha={alpha:.6g}",
#                     iter_num,
#                 )

#                 # --------------------------------------------------------------
#                 # 5.7) λ update
#                 # --------------------------------------------------------------
#                 lambda_before = self.lmbda

#                 v_prev = getattr(self, "_v_lambda", 0.0)
#                 v_new = (
#                     self.momentum_beta * v_prev
#                     + (1.0 - self.momentum_beta) * knapsack_subgradient
#                 )

#                 self._v_lambda = v_new

#                 delta_lambda = alpha * v_new
#                 delta_lambda = max(-max_lambda_delta, min(max_lambda_delta, delta_lambda))

#                 self.lmbda = self.lmbda + delta_lambda
#                 self.lmbda = max(0.0, min(self.lmbda, 1e4))

#                 _dbg(
#                     f"Lambda update: before={lambda_before:.6g}, "
#                     f"v_prev={v_prev:.6g}, "
#                     f"v_new={v_new:.6g}, "
#                     f"after={self.lmbda:.6g}",
#                     iter_num,
#                 )

#                 # --------------------------------------------------------------
#                 # 5.8) μ updates
#                 #
#                 # Modified:
#                 # Signed projected update:
#                 #     μ_i <- max(0, μ_i + gamma_mu * alpha * g_i)
#                 #
#                 # If g_i > 0, the cut is violated and μ_i increases.
#                 # If g_i < 0, the cut is slack and μ_i decreases.
#                 # --------------------------------------------------------------
#                 if mu_dynamic_here and len(cut_g_signed) > 0:
#                     for i, g in enumerate(cut_g_signed):
#                         g = float(g)

#                         delta = gamma_mu * alpha * g

#                         # Symmetric cap because delta can now be positive or negative.
#                         if mu_increment_cap is not None:
#                             delta = max(-mu_increment_cap, min(mu_increment_cap, delta))

#                         mu_old = float(self.best_cut_multipliers.get(i, 0.0))
#                         mu_new = mu_old + delta
#                         mu_new = max(0.0, min(mu_new, 1e4))

#                         self.best_cut_multipliers[i] = mu_new

#                         _dbg(
#                             f"Mu signed update cut[{i}]: g={g:.6g}, "
#                             f"delta={delta:.6g}, "
#                             f"mu_old={mu_old:.6g}, "
#                             f"mu_new={mu_new:.6g}",
#                             iter_num,
#                             force=True,
#                         )

#                 self.step_sizes.append(alpha)
#                 self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

#             # ------------------------------------------------------------------
#             # 6) Drop dead cuts
#             # ------------------------------------------------------------------
#             if self.use_cover_cuts and self.best_cuts and mu_dynamic_here:
#                 keep_indices = []

#                 best_mu_map = getattr(
#                     self,
#                     "best_cut_multipliers_for_best_bound",
#                     self.best_cut_multipliers,
#                 )

#                 _dbg(
#                     f"Dead-cut check starts: active_cuts={len(self.best_cuts)}, "
#                     f"max_cut_violation={max_cut_violation}",
#                     force=True,
#                 )

#                 for i, (cut, rhs) in enumerate(self.best_cuts):
#                     mu_i = float(self.best_cut_multipliers.get(i, 0.0))
#                     mu_best_i = float(best_mu_map.get(i, 0.0))

#                     ever_useful = (
#                         i < len(max_cut_violation)
#                         and max_cut_violation[i] > 0.0
#                     ) or abs(mu_best_i) >= dead_mu_threshold

#                     keep = not (
#                         not ever_useful
#                         and abs(mu_i) < dead_mu_threshold
#                         and abs(mu_best_i) < dead_mu_threshold
#                     )

#                     _dbg(
#                         f"Dead-cut decision cut[{i}]: "
#                         f"max_violation={max_cut_violation[i] if i < len(max_cut_violation) else None}, "
#                         f"mu_current={mu_i:.6g}, "
#                         f"mu_best={mu_best_i:.6g}, "
#                         f"ever_useful={ever_useful}, "
#                         f"keep={keep}",
#                         force=True,
#                     )

#                     if keep:
#                         keep_indices.append(i)

#                 if len(keep_indices) < len(self.best_cuts):
#                     _dbg(
#                         f"Dropping dead cuts: keep_indices={keep_indices}, "
#                         f"drop_count={len(self.best_cuts) - len(keep_indices)}",
#                         force=True,
#                     )

#                     new_best_cuts = []
#                     new_mu = {}
#                     new_mu_best = {}
#                     new_rhs_eff = {}

#                     for new_idx, old_idx in enumerate(keep_indices):
#                         new_best_cuts.append(self.best_cuts[old_idx])
#                         new_mu[new_idx] = float(
#                             self.best_cut_multipliers.get(old_idx, 0.0)
#                         )
#                         new_mu_best[new_idx] = float(
#                             self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0)
#                         )
#                         new_rhs_eff[new_idx] = int(
#                             self._rhs_eff.get(old_idx, self.best_cuts[old_idx][1])
#                         )

#                     self.best_cuts = new_best_cuts
#                     self.best_cut_multipliers = new_mu
#                     self.best_cut_multipliers_for_best_bound = new_mu_best
#                     self._rhs_eff = new_rhs_eff

#                     if self.best_cuts:
#                         _rebuild_cut_structures()
#                     else:
#                         self._cut_edge_idx = []
#                         self._cut_edge_idx_all = []
#                         rhs_eff_vec = np.zeros(0, dtype=float)

#             _print_cut_table("Final cuts before returning from node", force=True)

#             # ------------------------------------------------------------------
#             # 7) Restore best λ and μ to pass to children
#             #
#             # Unchanged strategy:
#             # λ and μ are both restored to the values that gave the best lower bound.
#             # ------------------------------------------------------------------
#             if hasattr(self, "best_lambda"):
#                 _dbg(
#                     f"Restore lambda: current={self.lmbda:.6g}, "
#                     f"best_lambda={self.best_lambda:.6g}",
#                     force=True,
#                 )
#                 self.lmbda = self.best_lambda

#             if hasattr(self, "best_cut_multipliers_for_best_bound"):
#                 _dbg(
#                     f"Restore best μ for children: "
#                     f"{self.best_cut_multipliers_for_best_bound}",
#                     force=True,
#                 )
#                 self.best_cut_multipliers = (
#                     self.best_cut_multipliers_for_best_bound.copy()
#                 )

#             _dbg(
#                 f"END NODE: best_LB={self.best_lower_bound:.6g}, "
#                 f"best_UB={self.best_upper_bound}, "
#                 f"return_new_cuts={len(node_new_cuts)}, "
#                 f"final_active_cuts={len(self.best_cuts)}",
#                 force=True,
#             )

#             end_time = time()
#             LagrangianMST.total_compute_time += end_time - start_time
#             return self.best_lower_bound, self.best_upper_bound, node_new_cuts


#     def compute_mst_for_lambda(self, lambda_val):
#         modified_edges = []
#         for i, (u, v) in enumerate(self.edge_list):
#             modified_w = self.edge_weights[i] + lambda_val * self.edge_lengths[i]
#             for cut_idx, (cut, _) in enumerate(self.best_cuts):
#                 if (u, v) in cut:
#                     modified_w += self.best_cut_multipliers.get(cut_idx, 0)
#             modified_edges.append((u, v, modified_w))
#         return self.compute_mst(modified_edges)

#     def _log_fractional_solution(self, method, edge_weights, msts, elapsed_time):
#         if self.verbose:
#             total_weight = sum(self.edge_weights[self.edge_indices[e]] * w for e, w in edge_weights.items())
#             total_length = sum(self.edge_lengths[self.edge_indices[e]] * w for e, w in edge_weights.items())
#             print(f"{method} solution: {len(edge_weights)} edges, "
#                   f"weight={total_weight:.2f}, length={total_length:.2f}, time={elapsed_time:.2f}s")
#             print(f"MSTs used: {len(msts)}")

    
    
#     def compute_dantzig_wolfe_solution(self, node):
#         start_time = time()
        
#         # Need at least 1 MST
#         if len(self.primal_solutions) < 1:
#             if self.verbose:
#                 print("Insufficient primal solutions for Dantzig-Wolfe")
#             return None

#         # More lenient filtering - just check basic validity
#         valid_msts = []
#         for mst_edges, is_feasible in self.primal_solutions:
#             if not mst_edges:
#                 continue
#             mst_edges_normalized = {tuple(sorted((u, v))) for u, v in mst_edges}
            
#             # Basic validity: correct number of edges
#             if len(mst_edges_normalized) == self.num_nodes - 1:
#                 valid_msts.append(mst_edges_normalized)
        
#         if len(valid_msts) < 1:
#             if self.verbose:
#                 print(f"No valid MSTs after filtering")
#             return None

#         # Handle single MST case
#         if len(valid_msts) == 1:
#             edge_weights = {e: 1.0 for e in valid_msts[0]}
#             if self.verbose:
#                 print(f"Dantzig-Wolfe: Single MST, returning as integral solution")
#             return edge_weights

#         if self.verbose:
#             print(f"Using {len(valid_msts)} valid MSTs for Dantzig-Wolfe")

#         # Select diverse MSTs
#         max_msts = min(10, len(valid_msts))
#         selected_msts = []
#         covered_edges = set()
#         remaining_msts = valid_msts.copy()
        
#         while remaining_msts and len(selected_msts) < max_msts:
#             best_mst = None
#             best_score = -1
#             for mst in remaining_msts:
#                 new_edges = mst - covered_edges
#                 score = len(new_edges)
#                 if score > best_score:
#                     best_score = score
#                     best_mst = mst
#             if best_mst:
#                 selected_msts.append(best_mst)
#                 covered_edges.update(best_mst)
#                 remaining_msts.remove(best_mst)
#             else:
#                 break

#         if len(selected_msts) < 2:
#             if self.verbose:
#                 print(f"Only {len(selected_msts)} diverse MSTs selected")
#             return None

#         num_msts = len(selected_msts)
#         edge_indices = self.edge_indices
        
#         # Objective: minimize total weight
#         c = []
#         for mst_edges in selected_msts:
#             weight = sum(self.edge_weights[edge_indices[e]] for e in mst_edges)
#             c.append(weight + 0.1 * (1.0 / num_msts))

#         # Convex combination constraint
#         A_eq = [np.ones(num_msts)]
#         b_eq = [1.0]

#         # Budget as inequality constraint
#         A_ub = []
#         b_ub = []
#         lengths = [sum(self.edge_lengths[edge_indices[e]] for e in mst_edges)
#                 for mst_edges in selected_msts]
#         A_ub.append(lengths)
#         b_ub.append(self.budget)

#         # Cover cuts (limit to avoid infeasibility)
#         if self.best_cuts and len(self.best_cuts) <= 20:
#             for cut, rhs in self.best_cuts:
#                 cut_indices = [edge_indices[e] for e in cut if e in edge_indices]
#                 if cut_indices:
#                     row = np.zeros(num_msts)
#                     for k, mst_edges in enumerate(selected_msts):
#                         cut_count = sum(1 for e in mst_edges if e in cut)
#                         row[k] = cut_count
#                     A_ub.append(row)
#                     b_ub.append(rhs)

#         bounds = [(0, None) for _ in range(num_msts)]

#         try:
#             res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
#             if not res.success:
#                 if self.verbose:
#                     print(f"LP with cuts failed: {res.message}, trying without cuts")
#                 # Retry without cover cuts
#                 res = linprog(c, A_ub=[lengths], b_ub=[self.budget], 
#                             A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
#                 if not res.success:
#                     if self.verbose:
#                         print(f"LP without cuts also failed: {res.message}")
#                     return None
            
#             lambda_k = res.x
#         except Exception as e:
#             if self.verbose:
#                 print(f"LP solver error: {e}")
#             return None

#         # Build fractional edge solution
#         edge_weights = {}
#         for u, v in self.edge_list:
#             e = (u, v)
#             weight = sum(lambda_k[k] for k, mst_edges in enumerate(selected_msts) if e in mst_edges)
#             if weight > 1e-6:
#                 edge_weights[e] = weight

#         if self.verbose:
#             print(f"Dantzig-Wolfe solution: {len(edge_weights)} edges")
#             truly_fractional = sum(1 for w in edge_weights.values() if 0.1 < w < 0.9)
#             print(f"  Truly fractional (0.1-0.9): {truly_fractional}/{len(edge_weights)} edges")

#         return edge_weights if edge_weights else None   
#     def compute_weighted_average_solution(self):
#         """Compute a fractional primal solution as a weighted average of MSTs."""
#         if not self.primal_solutions or not self.step_sizes:
#             if self.verbose:
#                 print("No primal solutions or step sizes available for weighted average")
#             return None

#         # Ensure lengths match (subgradient iterations should align)
#         if len(self.primal_solutions) != len(self.step_sizes):
#             if self.verbose:
#                 print(f"Mismatch: {len(self.primal_solutions)} primal solutions, {len(self.step_sizes)} step sizes")
#             return None

#         total_step_sum = sum(self.step_sizes)
#         if total_step_sum <= 0:
#             if self.verbose:
#                 print("Total step size sum is zero or negative")
#             return None

#         edge_weights = defaultdict(float)
#         for i, (mst_edges, _) in enumerate(self.primal_solutions):
#             lambda_i = self.step_sizes[i]

#             weight = lambda_i / total_step_sum
#             for e in mst_edges:
#                 # print("dd", e)
#                 edge_weights[e] += weight

#         # Ensure weights are in [0, 1] (should be automatic but added for robustness)
#         for e in edge_weights:
#             edge_weights[e] = min(1.0, max(0.0, edge_weights[e]))

#         if self.verbose:
#             total_weight = sum(self.edge_weights[self.edge_indices[e]] * w for e, w in edge_weights.items())
#             total_length = sum(self.edge_lengths[self.edge_indices[e]] * w for e, w in edge_weights.items())
#             print(f"Weighted Average Solution: {len(edge_weights)} edges, "
#                 f"weight={total_weight:.2f}, length={total_length:.2f}")

#         return dict(edge_weights) if edge_weights else None

#     def recover_primal_solution(self, node):
#         start_time = time()

#         for mst_edges, is_feasible in self.primal_solutions:
#             mst_edges_normalized = {tuple(sorted((u, v))) for u, v in mst_edges}
#             if not all(e in mst_edges_normalized for e in node.fixed_edges):
#                 continue
#             if any(e in mst_edges_normalized for e in node.excluded_edges):
#                 continue

#             real_length = sum(self.edge_lengths[self.edge_indices[e]] 
#                               for e in mst_edges_normalized)
#             if real_length > self.budget:
#                 continue

#             valid_cuts = True
#             for cut, rhs in node.active_cuts:
#                 cut_count = sum(1 for e in mst_edges_normalized if e in cut)
#                 if cut_count > rhs:
#                     valid_cuts = False
#                     break
#             if not valid_cuts:
#                 continue

#             uf = UnionFind(self.num_nodes)
#             for u, v in mst_edges_normalized:
#                 uf.union(u, v)
#             if uf.count_components() != 1 or len(set(u for u, _ in mst_edges_normalized) | set(v for _, v in mst_edges_normalized)) < self.num_nodes:
#                 continue

#             real_weight = sum(self.edge_weights[self.edge_indices[e]] 
#                               for e in mst_edges_normalized)
#             end_time = time()
#             if self.verbose:
#                 print(f"Feasible primal solution found from primal_solutions: weight={real_weight:.2f}, length={real_length:.2f}")
#             return list(mst_edges_normalized), real_weight, real_length

#         uf = UnionFind(self.num_nodes)
#         mst_edges = []
#         total_length = 0.0
#         total_weight = 0.0

#         for edge_idx in self.fixed_edge_indices:
#             u, v = self.edge_list[edge_idx]
#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 total_length += self.edge_lengths[edge_idx]
#                 total_weight += self.edge_weights[edge_idx]
#             else:
#                 if self.verbose:
#                     print(f"Fixed edge ({u}, {v}) creates cycle in greedy heuristic")
#                 return None, float('inf'), float('inf')

#         edge_indices = [i for i in range(len(self.edges)) 
#                         if i not in self.fixed_edge_indices and i not in self.excluded_edge_indices]
#         sorted_edges = sorted(edge_indices, key=lambda i: self.edge_weights[i])

#         for edge_idx in sorted_edges:
#             u, v = self.edge_list[edge_idx]
#             new_length = total_length + self.edge_lengths[edge_idx]
#             if new_length > self.budget:
#                 continue

#             temp_edges = mst_edges + [(u, v)]
#             valid_cuts = True
#             for cut, rhs in node.active_cuts:
#                 cut_count = sum(1 for e in temp_edges if e in cut)
#                 if cut_count > rhs:
#                     valid_cuts = False
#                     break
#             if not valid_cuts:
#                 continue

#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 total_length = new_length
#                 total_weight += self.edge_weights[edge_idx]

#         if uf.count_components() != 1 or len(set(u for u, _ in mst_edges) | set(v for _, v in mst_edges)) < self.num_nodes:
#             if self.verbose:
#                 print("Greedy heuristic failed to produce a valid spanning tree")
#             return None, float('inf'), float('inf')

#         end_time = time()
#         if self.verbose:
#             print(f"Feasible primal solution found via greedy heuristic: weight={total_weight:.2f}, length={total_length:.2f}")
#         return mst_edges, total_weight, total_length

#     def compute_real_weight_length(self):
#         real_weight = sum(self.edge_weights[self.edge_indices[e]] 
#                           for e in self.last_mst_edges)
#         real_length = sum(self.edge_lengths[self.edge_indices[e]] 
#                           for e in self.last_mst_edges)
#         return real_weight, real_length


# import networkx as nx
# import numpy as np
# from time import time
# from collections import defaultdict, OrderedDict
# from scipy.optimize import linprog  
# import math
# import heapq
# import hashlib



# class UnionFind:
#     __slots__ = ['parent', 'rank', 'size']
#     def __init__(self, n):
#         self.parent = list(range(n))
#         self.rank = [0] * n
#         self.size = [1] * n
    
#     def find(self, u):
#         if self.parent[u] != u:
#             self.parent[u] = self.find(self.parent[u])
#         return self.parent[u]
    
#     def union(self, u, v):
#         pu, pv = self.find(u), self.find(v)
#         if pu == pv:
#             return False
#         if self.size[pu] < self.size[pv]:
#             pu, pv = pv, pu
#         self.parent[pv] = pu
#         self.size[pu] += self.size[pv]
#         self.rank[pu] = max(self.rank[pu], self.rank[pv] + 1)
#         return True
    
#     def connected(self, u, v):
#         return self.find(u) == self.find(v)
    
#     def count_components(self):
#         return len(set(self.find(i) for i in range(len(self.parent))))

# class LRUCache:
#     __slots__ = ['cache', 'capacity']
#     def __init__(self, capacity):
#         self.cache = OrderedDict()
#         self.capacity = capacity
    
#     def get(self, key):
#         if key not in self.cache:
#             return None
#         self.cache.move_to_end(key)
#         return self.cache[key]
    
#     def put(self, key, value):
#         if key in self.cache:
#             self.cache.move_to_end(key)

#         self.cache[key] = value
#         if len(self.cache) > self.capacity:
#             self.cache.popitem(last=False)

# class LagrangianMST:
#     total_compute_time = 0


#     def __init__(self, edges, num_nodes, budget, fixed_edges=None, excluded_edges=None,
#                  initial_lambda=0.05, step_size=0.001, max_iter=10, 
#                  use_cover_cuts=False, cut_frequency=5, use_bisection=False,
#                  verbose=False, shared_graph=None):
#         start_time = time()
#         self.edges = edges
#         self.num_nodes = num_nodes
#         self.budget = budget
#         self.fixed_edges = {tuple(sorted((u, v))) for u, v in (fixed_edges or set())}
#         self.excluded_edges = {tuple(sorted((u, v))) for u, v in (excluded_edges or set())}

#         edge_key = id(edges)
#         if getattr(LagrangianMST, "_edge_key", None) != edge_key:
#             LagrangianMST._edge_key = edge_key
#             edge_list = [tuple(sorted((u, v))) for u, v, _, _ in edges]
#             LagrangianMST._edge_list = edge_list
#             LagrangianMST._edge_indices = {edge: idx for idx, edge in enumerate(edge_list)}
#             LagrangianMST._edge_weights = np.array([w for _, _, w, _ in edges], dtype=np.float32)
#             LagrangianMST._edge_lengths = np.array([l for _, _, _, l in edges], dtype=np.float32)
#             LagrangianMST._edge_attributes = {
#                 edge: (w, l) for (edge, (_, _, w, l)) in zip(edge_list, edges)
#             }

#         self.edge_list = LagrangianMST._edge_list
#         self.edge_indices = LagrangianMST._edge_indices
#         self.edge_weights = LagrangianMST._edge_weights
#         self.edge_lengths = LagrangianMST._edge_lengths
#         self.edge_attributes = LagrangianMST._edge_attributes

#         self.lmbda = initial_lambda
#         self.step_size = step_size
#         # self.p = p
#         self.max_iter = max_iter
#         self.use_bisection = use_bisection
#         self.verbose = verbose

#         self.best_lower_bound = float('-inf')
#         self.best_upper_bound = float('inf')
#         self.last_mst_edges = []
#         self.primal_solutions = []
#         self.fractional_solutions = []
#         self.step_sizes = []
#         self.subgradients = []
#         self._MAX_HISTORY = 100
#         self._primal_history_cap = 30
#         self._fractional_history_cap = 50
#         self._subgradient_history_cap = 20

#         self.best_lambda = self.lmbda
#         self.best_mst_edges = None
#         self.best_cost = 0

#         self.use_cover_cuts = use_cover_cuts
#         self.cut_frequency = cut_frequency
#         self.best_cuts = []
#         self.best_cut_multipliers = {}

#         self.multipliers = []

#         self.fixed_edge_indices = {
#             self.edge_indices.get((u, v)) for u, v in self.fixed_edges
#             if (u, v) in self.edge_indices
#         }
#         self.excluded_edge_indices = {
#             self.edge_indices.get((u, v)) for u, v in self.excluded_edges
#             if (u, v) in self.edge_indices
#         }

#         self.cache_tolerance = 1e-6 if num_nodes > 100 else 1e-8
#         self.mst_cache = LRUCache(capacity=64)


#         self.last_mst_edges = None

#         if shared_graph is not None:
#             self.graph = shared_graph
#         else:
#             self.graph = nx.Graph()
#             self.graph.add_edges_from(self.edge_list)

#         self._free_mask_cache = None
#         self._free_mask_key = None
#         self._mw_cached = None
#         self._mw_lambda = None
#         self._mw_mu = None

#         end_time = time()
#         LagrangianMST.total_compute_time += end_time - start_time


    

#     def reset(self, *, fixed_edges=None, excluded_edges=None, initial_lambda=None,
#               step_size=None, max_iter=None, use_cover_cuts=None, cut_frequency=None,
#               use_bisection=None, verbose=None):
#         if fixed_edges is not None:
#             self.fixed_edges = {tuple(sorted((u, v))) for u, v in fixed_edges}
#         else:
#             self.fixed_edges = set()

#         if excluded_edges is not None:
#             self.excluded_edges = {tuple(sorted((u, v))) for u, v in excluded_edges}
#         else:
#             self.excluded_edges = set()

#         self.fixed_edge_indices = {
#             self.edge_indices.get((u, v)) for u, v in self.fixed_edges
#             if (u, v) in self.edge_indices
#         }
#         self.excluded_edge_indices = {
#             self.edge_indices.get((u, v)) for u, v in self.excluded_edges
#             if (u, v) in self.edge_indices
#         }

#         if initial_lambda is not None:
#             self.lmbda = float(initial_lambda)
#         else:
#             self.lmbda = getattr(self, "lmbda", 0.05)

#         if step_size is not None:
#             self.step_size = float(step_size)
#         if max_iter is not None:
#             self.max_iter = int(max_iter)
#         if use_cover_cuts is not None:
#             self.use_cover_cuts = bool(use_cover_cuts)
#         if cut_frequency is not None:
#             self.cut_frequency = int(cut_frequency)
#         if use_bisection is not None:
#             self.use_bisection = bool(use_bisection)
#         if verbose is not None:
#             self.verbose = bool(verbose)

#         self.best_lower_bound = float("-inf")
#         self.best_upper_bound = float("inf")

#         self.best_lambda = float(self.lmbda)
#         self.best_mst_edges = []
#         self.best_cost = 0

#         self.best_cuts = []
#         self.best_cut_multipliers = {}
#         self.best_cut_multipliers_for_best_bound = {}

#         self.multipliers = []

#         # Important when reusing solver objects in strong branching
#         self._v_lambda = 0.0

#         self.last_mst_edges = None

#         try:
#             cap = self.mst_cache.capacity
#         except Exception:
#             cap = max(20, self.num_nodes * 2)
#         self.mst_cache = LRUCache(capacity=cap)

#         self._invalidate_weight_cache()


#     def clear_iteration_state(self):
#         """Clear per-iteration buffers"""
#         self.primal_solutions = []
#         self.fractional_solutions = []
#         self.subgradients = []
#         self.step_sizes = []
#         self.multipliers = []
#         self._v_lambda = 0.0
#         # self.last_modified_weights = None
#         # self.last_mst_edges = None
#         self._invalidate_weight_cache()
#         if hasattr(self, 'mst_cache'):
#             self.mst_cache = LRUCache(capacity=5)
#     # def generate_cover_cuts(self, mst_edges):  
#     #     """
#     #     Stronger cover cuts (tightened):
#     #     - Residualization: A, B' (clamped), fixed/excluded respected
#     #     - Seed residual-minimal cover from T^λ ∩ A
#     #     - Certificate shrinking using optimistic U(S) with component-based k, and exact Kruskal fallback
#     #     - Inclusion-minimal S* shrinking under the certificate (THIS was missing)
#     #     - Micro-seed from top-L heaviest admissible edges
#     #     - Stronger safe lifting for residual-minimal covers
#     #     - Strict effective-RHS pruning + current-violation checks
#     #     - Dedup with dominance & subset-dominance
#     #     """
#     #     if not mst_edges:
#     #         return []

#     #     EPS = 1e-12
#     #     L_MICRO = 3
#     #     MAX_RETURN = 10

#     #     # --- normalize edges ---
#     #     def norm(e):
#     #         u, v = e
#     #         return (u, v) if u <= v else (v, u)

#     #     mst_norm = [norm(e) for e in mst_edges]
#     #     mst_set = set(mst_norm)

#     #     # --- accessors / data ---
#     #     edge_attr = self.edge_attributes  # edge -> (w, ℓ)
#     #     def get_len(e): return edge_attr[e][1]

#     #     fixed = set(getattr(self, "fixed_edges", set()))
#     #     excluded = set(getattr(self, "excluded_edges", set()))
#     #     budget = self.budget

#     #     # Residual budget
#     #     L_fix = sum(get_len(e) for e in fixed if e in edge_attr)
#     #     Bp = budget - L_fix

#     #     # If fixes already exceed the budget, cuts may still be useful, but be careful with rhs_eff.
#     #     # We will still attempt separation.

#     #     # Admissible edges A
#     #     A = {e for e in getattr(self, "edge_list", []) if e not in fixed and e not in excluded and e in edge_attr}
#     #     if not A:
#     #         return []

#     #     # T^λ ∩ A (use provided mst_edges)
#     #     TcapA = [e for e in mst_norm if e in A]

#     #     # If residual MST is feasible, nothing to cut
#     #     mst_len = sum(get_len(e) for e in TcapA)
#     #     if mst_len <= Bp + EPS:
#     #         return []

#     #     cuts = []

#     #     # Pre-sort A by length for U(S) and Kruskal completion
#     #     A_sorted = sorted(A, key=lambda e: get_len(e))

#     #     # --- DSU helpers (for component count & exact completion) ---
#     #     def get_nodes():
#     #         # best-effort: use graph nodes if present, otherwise infer from edge keys
#     #         if hasattr(self, "graph") and hasattr(self.graph, "nodes"):
#     #             try:
#     #                 return list(self.graph.nodes)
#     #             except Exception:
#     #                 pass
#     #         nodes = set()
#     #         for (u, v) in edge_attr.keys():
#     #             nodes.add(u); nodes.add(v)
#     #         for (u, v) in fixed:
#     #             nodes.add(u); nodes.add(v)
#     #         return list(nodes)

#     #     NODES = get_nodes()

#     #     def component_k_needed(contracted_edges):
#     #         """Number of edges needed to connect after contracting 'contracted_edges': k = #components - 1."""
#     #         parent = {n: n for n in NODES}
#     #         rank = {n: 0 for n in NODES}

#     #         def find(x):
#     #             while parent[x] != x:
#     #                 parent[x] = parent[parent[x]]
#     #                 x = parent[x]
#     #             return x

#     #         def union(x, y):
#     #             rx, ry = find(x), find(y)
#     #             if rx == ry:
#     #                 return
#     #             if rank[rx] < rank[ry]:
#     #                 parent[rx] = ry
#     #             elif rank[rx] > rank[ry]:
#     #                 parent[ry] = rx
#     #             else:
#     #                 parent[ry] = rx
#     #                 rank[rx] += 1

#     #         for (u, v) in contracted_edges:
#     #             if u in parent and v in parent:
#     #                 union(u, v)

#     #         reps = {find(n) for n in NODES}
#     #         comps = len(reps)
#     #         return max(0, comps - 1)

#     #     def U_of(Sprime):
#     #         """
#     #         Optimistic completion:
#     #         sum of k cheapest edges in A \\ S', where k = (#components after contracting fixed ∪ S') - 1.
#     #         This is stronger/more accurate than r' - |S'|.
#     #         """
#     #         Sprime_set = Sprime if isinstance(Sprime, set) else set(Sprime)
#     #         contracted = set(fixed) | Sprime_set
#     #         k = component_k_needed(contracted)
#     #         if k <= 0:
#     #             return 0.0

#     #         total = 0.0
#     #         taken = 0
#     #         for e in A_sorted:
#     #             if e in Sprime_set:
#     #                 continue
#     #             total += get_len(e)
#     #             taken += 1
#     #             if taken == k:
#     #                 break
#     #         return total if taken == k else float("inf")

#     #     def completion_mst_cost(Ssub):
#     #         """
#     #         Exact completion via Kruskal after contracting fixed ∪ Ssub.
#     #         Returns minimum additional length needed to connect components using edges in A \\ Ssub.
#     #         """
#     #         parent = {n: n for n in NODES}
#     #         rank = {n: 0 for n in NODES}

#     #         def find(x):
#     #             while parent[x] != x:
#     #                 parent[x] = parent[parent[x]]
#     #                 x = parent[x]
#     #             return x

#     #         def union(x, y):
#     #             rx, ry = find(x), find(y)
#     #             if rx == ry:
#     #                 return False
#     #             if rank[rx] < rank[ry]:
#     #                 parent[rx] = ry
#     #             elif rank[rx] > rank[ry]:
#     #                 parent[ry] = rx
#     #             else:
#     #                 parent[ry] = rx
#     #                 rank[rx] += 1
#     #             return True

#     #         contracted = set(fixed) | set(Ssub)
#     #         for (u, v) in contracted:
#     #             if u in parent and v in parent:
#     #                 union(u, v)

#     #         reps = {find(n) for n in NODES}
#     #         k_needed = max(0, len(reps) - 1)
#     #         if k_needed <= 0:
#     #             return 0.0

#     #         Sset = set(Ssub)
#     #         total = 0.0
#     #         taken = 0
#     #         for e in A_sorted:
#     #             if e in Sset:
#     #                 continue
#     #             u, v = e
#     #             if u not in parent or v not in parent:
#     #                 continue
#     #             if union(u, v):
#     #                 total += get_len(e)
#     #                 taken += 1
#     #                 if taken == k_needed:
#     #                     break
#     #         return total if taken == k_needed else float("inf")

#     #     def build_residual_minimal_cover(desc_edges):
#     #         """Minimal cover on B': add in desc ℓ, then prune shortest while violation remains."""
#     #         S, sL = [], 0.0
#     #         for e in desc_edges:
#     #             if e not in edge_attr:
#     #                 continue
#     #             S.append(e)
#     #             sL += get_len(e)
#     #             if sL > Bp + EPS:
#     #                 # prune shortest while still violating
#     #                 S.sort(key=lambda x: get_len(x))  # increasing
#     #                 k = 0
#     #                 while k < len(S) and (sL - get_len(S[k]) > Bp + EPS):
#     #                     sL -= get_len(S[k])
#     #                     k += 1
#     #                 if k > 0:
#     #                     S = S[k:]
#     #                 return S, sL
#     #         return None, None

#     #     def rhs_eff(cset):
#     #         """Effective RHS after accounting fixed-in edges."""
#     #         return len(cset) - 1 - sum(1 for e in cset if e in fixed)

#     #     def is_violated_now(cset):
#     #         """Check current MST violation: lhs > rhs_eff."""
#     #         lhs = sum(1 for e in cset if e in mst_set)
#     #         return lhs > rhs_eff(cset)

#     #     def cert_holds(Slist):
#     #         """
#     #         Certificate: sumℓ(S) + U(S) > B' (optimistic), else fallback to exact completion.
#     #         """
#     #         if not Slist or len(Slist) <= 1:
#     #             return False
#     #         if rhs_eff(Slist) <= 0:
#     #             return False
#     #         sumS = sum(get_len(e) for e in Slist)
#     #         U = U_of(Slist)
#     #         if U != float("inf") and (sumS + U) > (Bp + EPS):
#     #             return True
#     #         exact = completion_mst_cost(Slist)
#     #         return exact != float("inf") and (sumS + exact) > (Bp + EPS)

#     #     def inclusion_minimal_shrink(Sstart):
#     #         """
#     #         Make S inclusion-minimal under cert_holds by removing one edge at a time.
#     #         We try removals from longest to shortest for a small S.
#     #         """
#     #         Sstar = sorted(Sstart, key=lambda e: get_len(e), reverse=True)
#     #         changed = True
#     #         while changed and len(Sstar) > 1:
#     #             changed = False
#     #             for j in range(len(Sstar)):  # longest -> shortest
#     #                 trial = Sstar[:j] + Sstar[j+1:]
#     #                 if len(trial) <= 1:
#     #                     continue
#     #                 if cert_holds(trial):
#     #                     Sstar = sorted(trial, key=lambda e: get_len(e), reverse=True)
#     #                     changed = True
#     #                     break
#     #         return Sstar

#     #     def try_shrink_and_add(seed_S, seed_sumL):
#     #         """
#     #         Full LaTeX Step (2):
#     #         - remove longest edges until sumℓ <= B' => first S'
#     #         - require cert_holds(S')
#     #         - shrink to inclusion-minimal S* while certificate holds
#     #         - add the cut if it separates current MST
#     #         """
#     #         if not seed_S or len(seed_S) <= 1:
#     #             return

#     #         S_work = sorted(seed_S, key=lambda e: get_len(e), reverse=True)
#     #         sumL = float(seed_sumL)

#     #         # First S' with sumℓ <= B'
#     #         idx = 0
#     #         while idx < len(S_work) and sumL > Bp + EPS:
#     #             sumL -= get_len(S_work[idx])
#     #             idx += 1
#     #         Sprime = S_work[idx:]
#     #         if not Sprime or len(Sprime) <= 1:
#     #             return

#     #         if not cert_holds(Sprime):
#     #             return

#     #         Sstar = inclusion_minimal_shrink(Sprime)
#     #         if len(Sstar) <= 1:
#     #             return

#     #         if is_violated_now(Sstar):
#     #             cuts.append((set(Sstar), len(Sstar) - 1))

#     #     def lift_minimal_cover(S_min, rhs_base):
#     #         """
#     #         Stronger safe lifting for residual-minimal cover S:
#     #         Lift any f with ℓ(f) > B' - sumℓ(S) + Lmax.
#     #         (This is typically much stronger than ℓ(f) >= Lmax.)
#     #         """
#     #         S_base = set(S_min)
#     #         if not S_base:
#     #             return None
#     #         sumS = sum(get_len(e) for e in S_base)
#     #         Lmax = max(get_len(e) for e in S_base)
#     #         threshold = (Bp - sumS + Lmax)  # lift if len(f) > threshold

#     #         lift_add = {f for f in A if f not in S_base and get_len(f) > threshold + EPS}
#     #         if not lift_add:
#     #             return None

#     #         S_lift = S_base | lift_add
#     #         # RHS remains rhs_base (|S|-1 of original minimal cover)
#     #         if rhs_eff(S_lift) > 0 and is_violated_now(S_lift):
#     #             return (S_lift, rhs_base)
#     #         return None

#     #     # --- (1) primary seed from T^λ ∩ A ---
#     #     T_desc = sorted(TcapA, key=lambda e: get_len(e), reverse=True)
#     #     S_seed, sumL_seed = build_residual_minimal_cover(T_desc)
#     #     if not S_seed:
#     #         return []

#     #     S_seed = list(S_seed)
#     #     if rhs_eff(S_seed) > 0 and is_violated_now(S_seed):
#     #         cuts.append((set(S_seed), len(S_seed) - 1))

#     #     # Step (2): certificate shrink to inclusion-minimal S*
#     #     try_shrink_and_add(S_seed, sumL_seed)

#     #     # --- stronger lifting on the residual-minimal seed cover ---
#     #     lifted = lift_minimal_cover(S_seed, rhs_base=(len(S_seed) - 1))
#     #     if lifted is not None:
#     #         cuts.append(lifted)

#     #     # --- (1b) micro-seed: top-L heaviest admissible edges ---
#     #     if L_MICRO > 0 and len(A) > 0:
#     #         heavyA = sorted(A, key=lambda e: get_len(e), reverse=True)[:L_MICRO]
#     #         S2, sumL2 = build_residual_minimal_cover(heavyA)
#     #         if S2:
#     #             S2set = set(S2)
#     #             if rhs_eff(S2set) > 0 and S2set != set(S_seed) and is_violated_now(S2set):
#     #                 cuts.append((S2set, len(S2) - 1))

#     #             try_shrink_and_add(S2, sumL2)

#     #             lifted2 = lift_minimal_cover(S2, rhs_base=(len(S2) - 1))
#     #             if lifted2 is not None:
#     #                 cuts.append(lifted2)

#     #     # --- dedup & dominance-aware selection ---
#     #     uniq = {}
#     #     for cset, rhs in cuts:
#     #         key = tuple(sorted(cset))
#     #         best = uniq.get(key)
#     #         if best is None or rhs < best[1] or (rhs == best[1] and len(cset) < len(best[0])):
#     #             uniq[key] = (cset, rhs)

#     #     final = list(uniq.values())
#     #     final.sort(key=lambda t: (t[1], len(t[0])))

#     #     kept = []
#     #     for cset, rhs in final:
#     #         if rhs_eff(cset) <= 0:
#     #             continue
#     #         dominated = any(dset <= cset and drhs <= rhs for dset, drhs in kept)
#     #         if not dominated:
#     #             kept.append((cset, rhs))
#     #     return kept[:MAX_RETURN]
#     def generate_cover_cuts(self, mst_edges):
#         """
#         Cover-cut generation with exact tree-completion certificate.

#         Main logic:
#         - Work at the current B&B node with fixed edges F+ and excluded edges F-.
#         - Define residual budget B' = B - length(F+).
#         - Define admissible edges A = E \ (F+ union F-).
#         - Generate a residual-minimal seed cover from the current violating
#         Lagrangian MST T^lambda.
#         - Refine the seed using an exact minimum-length MST completion certificate:
#             contract F+ union S'
#             complete using admissible edges A \ S'
#             compute the minimum additional length by Kruskal using edge lengths
#         - Add a cut only if it is violated by the current Lagrangian MST.
#         - Optionally add lifted cuts using the residual-aware lifting rule.
#         """
#         if not mst_edges:
#             return []

#         EPS = 1e-12
#         L_MICRO = 3
#         MAX_RETURN = 10

#         # ------------------------------------------------------------
#         # Normalize edges
#         # ------------------------------------------------------------
#         def norm(e):
#             u, v = e
#             return (u, v) if u <= v else (v, u)

#         mst_norm = [norm(e) for e in mst_edges]
#         mst_set = set(mst_norm)

#         # ------------------------------------------------------------
#         # Accessors and node data
#         # ------------------------------------------------------------
#         edge_attr = self.edge_attributes  # edge -> (weight, length)

#         def get_len(e):
#             return edge_attr[e][1]

#         fixed = set(getattr(self, "fixed_edges", set()))
#         excluded = set(getattr(self, "excluded_edges", set()))
#         budget = self.budget

#         # Residual budget B' = B - length(F+)
#         L_fix = sum(get_len(e) for e in fixed if e in edge_attr)
#         Bp = budget - L_fix

#         # Admissible residual edges A = E \ (F+ union F-)
#         A = {
#             e for e in getattr(self, "edge_list", [])
#             if e not in fixed and e not in excluded and e in edge_attr
#         }

#         if not A:
#             return []

#         # Current Lagrangian tree restricted to admissible residual edges
#         TcapA = [e for e in mst_norm if e in A]

#         # If the current residual tree already respects the residual budget,
#         # there is no budget-violating tree to separate.
#         mst_len = sum(get_len(e) for e in TcapA)
#         if mst_len <= Bp + EPS:
#             return []

#         cuts = []

#         # Admissible edges sorted by LENGTH for exact completion.
#         # This is the length-MST completion part.
#         A_sorted = sorted(A, key=lambda e: get_len(e))

#         # ------------------------------------------------------------
#         # Node list for local DSU
#         # ------------------------------------------------------------
#         def get_nodes():
#             if hasattr(self, "graph") and hasattr(self.graph, "nodes"):
#                 try:
#                     return list(self.graph.nodes)
#                 except Exception:
#                     pass

#             nodes = set()
#             for (u, v) in edge_attr.keys():
#                 nodes.add(u)
#                 nodes.add(v)
#             for (u, v) in fixed:
#                 nodes.add(u)
#                 nodes.add(v)
#             return list(nodes)

#         NODES = get_nodes()

#         # ------------------------------------------------------------
#         # Exact minimum-length completion
#         # ------------------------------------------------------------
#         def completion_mst_cost(Ssub):
#             """
#             Minimum additional LENGTH needed to complete F+ union Ssub
#             to a spanning tree.

#             Steps:
#             1. Contract all fixed edges F+.
#             2. Contract all edges in Ssub.
#             3. Complete the remaining components using Kruskal on A \ Ssub,
#             sorted by edge length.
#             4. Return +inf if no spanning tree completion exists.

#             Important:
#             - If F+ union Ssub already creates a cycle, then no spanning tree can
#             contain all of those forced edges, so completion is impossible.
#             - The returned value is only the additional length beyond Ssub.
#             The fixed-edge length has already been removed through B'.
#             """
#             Sset = set(Ssub)

#             parent = {n: n for n in NODES}
#             rank = {n: 0 for n in NODES}
#             components = len(NODES)

#             def find(x):
#                 while parent[x] != x:
#                     parent[x] = parent[parent[x]]
#                     x = parent[x]
#                 return x

#             def union(x, y):
#                 nonlocal components

#                 rx, ry = find(x), find(y)
#                 if rx == ry:
#                     return False

#                 if rank[rx] < rank[ry]:
#                     parent[rx] = ry
#                 elif rank[rx] > rank[ry]:
#                     parent[ry] = rx
#                 else:
#                     parent[ry] = rx
#                     rank[rx] += 1

#                 components -= 1
#                 return True

#             # Force/contract fixed edges.
#             for e in fixed:
#                 if e not in edge_attr:
#                     return float("inf")

#                 u, v = e
#                 if u not in parent or v not in parent:
#                     return float("inf")

#                 # Fixed edges creating a cycle means no tree can contain all of them.
#                 if not union(u, v):
#                     return float("inf")

#             # Force/contract Ssub.
#             for e in Sset:
#                 if e not in edge_attr:
#                     return float("inf")

#                 u, v = e
#                 if u not in parent or v not in parent:
#                     return float("inf")

#                 # If F+ union Ssub creates a cycle, no spanning tree can contain Ssub.
#                 if not union(u, v):
#                     return float("inf")

#             # Already connected after contracting fixed union Ssub.
#             if components == 1:
#                 return 0.0

#             total_completion_length = 0.0

#             # Complete with cheapest admissible edges by LENGTH.
#             # A already excludes fixed and excluded edges.
#             # We also exclude Ssub because those edges are already forced.
#             for e in A_sorted:
#                 if e in Sset:
#                     continue

#                 u, v = e
#                 if u not in parent or v not in parent:
#                     continue

#                 if union(u, v):
#                     total_completion_length += get_len(e)

#                     if components == 1:
#                         return total_completion_length

#             # Could not connect all components.
#             return float("inf")

#         # ------------------------------------------------------------
#         # Build residual-minimal seed cover
#         # ------------------------------------------------------------
#         def build_residual_minimal_cover(desc_edges):
#             """
#             Build a residual-minimal cover with respect to B'.

#             We add edges in nonincreasing length order until the residual budget
#             is exceeded, then prune shortest edges while the violation remains.
#             """
#             S = []
#             sL = 0.0

#             for e in desc_edges:
#                 if e not in edge_attr:
#                     continue

#                 S.append(e)
#                 sL += get_len(e)

#                 if sL > Bp + EPS:
#                     # Prune shortest edges while still violating B'.
#                     S.sort(key=lambda x: get_len(x))  # increasing length

#                     k = 0
#                     while k < len(S) and (sL - get_len(S[k]) > Bp + EPS):
#                         sL -= get_len(S[k])
#                         k += 1

#                     if k > 0:
#                         S = S[k:]

#                     return S, sL

#             return None, None

#         # ------------------------------------------------------------
#         # RHS and violation helpers
#         # ------------------------------------------------------------
#         def rhs_eff(cset):
#             """
#             Effective RHS after fixed-in edges.

#             For generated cuts at the current node, cset is usually a subset of A,
#             so this is normally |S|-1. This form is kept for safety.
#             """
#             return len(cset) - 1 - sum(1 for e in cset if e in fixed)

#         def is_violated_now(cset):
#             """
#             Check whether the current Lagrangian MST violates the cut.
#             """
#             lhs = sum(1 for e in cset if e in mst_set)
#             return lhs > rhs_eff(cset)

#         # ------------------------------------------------------------
#         # Exact completion certificate
#         # ------------------------------------------------------------
#         def cert_holds(Slist):
#             """
#             Exact tree-completion certificate.

#             The cut sum_{e in Slist} x_e <= |Slist|-1 is valid at this node if
#             every spanning-tree completion containing F+ union Slist violates
#             the residual budget.

#             We test this by computing the minimum additional LENGTH needed to
#             complete F+ union Slist to a spanning tree.
#             """
#             if not Slist or len(Slist) <= 1:
#                 return False

#             if rhs_eff(Slist) <= 0:
#                 return False

#             sumS = sum(get_len(e) for e in Slist)
#             completion = completion_mst_cost(Slist)

#             # If completion is impossible, then no feasible spanning tree can
#             # contain all edges in Slist, so the cut is valid.
#             if completion == float("inf"):
#                 return True

#             return (sumS + completion) > (Bp + EPS)

#         # ------------------------------------------------------------
#         # Inclusion-minimal shrinking under exact certificate
#         # ------------------------------------------------------------
#         def inclusion_minimal_shrink(Sstart):
#             """
#             Make S inclusion-minimal under cert_holds by removing one edge at a time.

#             This version scans removals from shortest to longest, matching the
#             current LaTeX description.
#             """
#             Sstar = sorted(Sstart, key=lambda e: get_len(e))  # shortest -> longest

#             changed = True
#             while changed and len(Sstar) > 1:
#                 changed = False

#                 for j in range(len(Sstar)):
#                     trial = Sstar[:j] + Sstar[j + 1:]

#                     if len(trial) <= 1:
#                         continue

#                     if cert_holds(trial):
#                         Sstar = sorted(trial, key=lambda e: get_len(e))
#                         changed = True
#                         break

#             return Sstar

#         # ------------------------------------------------------------
#         # Try completion-aware shrinking and add resulting cut
#         # ------------------------------------------------------------
#         def try_shrink_and_add(seed_S, seed_sumL):
#             """
#             Starting from a residual-minimal seed cover S, find a smaller set S'
#             that is still invalid under exact MST completion.

#             Procedure:
#             - Remove long edges until the set is no longer a simple residual cover.
#             - Test the exact completion certificate.
#             - Shrink to an inclusion-minimal certified set.
#             - Add the cut only if it separates the current Lagrangian MST.
#             """
#             if not seed_S or len(seed_S) <= 1:
#                 return

#             S_work = sorted(seed_S, key=lambda e: get_len(e), reverse=True)
#             sumL = float(seed_sumL)

#             # First candidate S': remove longest edges until sum length <= B'.
#             idx = 0
#             while idx < len(S_work) and sumL > Bp + EPS:
#                 sumL -= get_len(S_work[idx])
#                 idx += 1

#             Sprime = S_work[idx:]

#             if not Sprime or len(Sprime) <= 1:
#                 return

#             if not cert_holds(Sprime):
#                 return

#             Sstar = inclusion_minimal_shrink(Sprime)

#             if len(Sstar) <= 1:
#                 return

#             if is_violated_now(Sstar):
#                 cuts.append((set(Sstar), len(Sstar) - 1))

#         # ------------------------------------------------------------
#         # Safe residual-aware lifting
#         # ------------------------------------------------------------
#         def lift_minimal_cover(S_min, rhs_base):
#             """
#             Safe unit lifting for a residual-minimal cover S.

#             If length(f) > B' - length(S) + Lmax,
#             then f can be lifted with coefficient 1 while keeping the same RHS.
#             """
#             S_base = set(S_min)

#             if not S_base:
#                 return None

#             sumS = sum(get_len(e) for e in S_base)
#             Lmax = max(get_len(e) for e in S_base)
#             threshold = Bp - sumS + Lmax

#             lift_add = {
#                 f for f in A
#                 if f not in S_base and get_len(f) > threshold + EPS
#             }

#             if not lift_add:
#                 return None

#             S_lift = S_base | lift_add

#             # RHS remains rhs_base = |S_min|-1.
#             if rhs_eff(S_lift) > 0 and is_violated_now(S_lift):
#                 return (S_lift, rhs_base)

#             return None

#         # ============================================================
#         # Main separation logic
#         # ============================================================

#         # Primary seed from T^lambda intersect A.
#         T_desc = sorted(TcapA, key=lambda e: get_len(e), reverse=True)
#         S_seed, sumL_seed = build_residual_minimal_cover(T_desc)

#         if not S_seed:
#             return []

#         S_seed = list(S_seed)

#         # Add simple residual cover cut.
#         if rhs_eff(S_seed) > 0 and is_violated_now(S_seed):
#             cuts.append((set(S_seed), len(S_seed) - 1))

#         # Add exact completion-aware refined cut.
#         try_shrink_and_add(S_seed, sumL_seed)

#         # Add lifted residual-minimal seed cover.
#         lifted = lift_minimal_cover(S_seed, rhs_base=(len(S_seed) - 1))
#         if lifted is not None:
#             cuts.append(lifted)

#         # Optional micro-seed from globally heaviest admissible edges.
#         # Keep this only if you also mention it in the paper/implementation section.
#         if L_MICRO > 0 and len(A) > 0:
#             heavyA = sorted(A, key=lambda e: get_len(e), reverse=True)[:L_MICRO]
#             S2, sumL2 = build_residual_minimal_cover(heavyA)

#             if S2:
#                 S2set = set(S2)

#                 if (
#                     rhs_eff(S2set) > 0
#                     and S2set != set(S_seed)
#                     and is_violated_now(S2set)
#                 ):
#                     cuts.append((S2set, len(S2) - 1))

#                 try_shrink_and_add(S2, sumL2)

#                 lifted2 = lift_minimal_cover(S2, rhs_base=(len(S2) - 1))
#                 if lifted2 is not None:
#                     cuts.append(lifted2)

#         # ------------------------------------------------------------
#         # Deduplication and dominance filtering
#         # ------------------------------------------------------------
#         uniq = {}

#         for cset, rhs in cuts:
#             key = tuple(sorted(cset))
#             best = uniq.get(key)

#             if best is None or rhs < best[1] or (
#                 rhs == best[1] and len(cset) < len(best[0])
#             ):
#                 uniq[key] = (cset, rhs)

#         final = list(uniq.values())
#         final.sort(key=lambda t: (t[1], len(t[0])))

#         kept = []

#         for cset, rhs in final:
#             if rhs_eff(cset) <= 0:
#                 continue

#             dominated = any(
#                 dset <= cset and drhs <= rhs
#                 for dset, drhs in kept
#             )

#             if not dominated:
#                 kept.append((cset, rhs))

#         return kept[:MAX_RETURN]

    
    
#     def compute_modified_weights(self):

#         base = self.edge_weights.copy()
#         lam = max(0.0, min(getattr(self, "lmbda", 0.0), 1e4))
#         if lam:
#             base = base + lam * self.edge_lengths

#         # No cuts? return λ-priced base
#         if not (self.use_cover_cuts and self.best_cuts):
#             self._mw_cached = None
#             self._mw_lambda = lam
#             self._mw_mu = None
#             self._mw_free_mask_key = None
#             return base
#         cut_idxs_free = getattr(self, "_cut_edge_idx", None)  # FREE indices only

#         mu_len = len(self.best_cuts)

#         mu = np.array([max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4)) for i in range(mu_len)], dtype=float)

#         # Cache key: (λ, μ, free-mask signature)
#         _ = self._get_free_mask()
#         free_mask_key = self._free_mask_key

#         if (self._mw_cached is not None and
#             self._mw_lambda == lam and
#             self._mw_mu is not None and
#             self._mw_mu.shape == mu.shape and
#             np.allclose(self._mw_mu, mu, rtol=0, atol=0) and
#             self._mw_free_mask_key == free_mask_key):
#             return self._mw_cached

#         # Add μ to ALL edges that belong to each cut (fixed edges included)
#         weights = base.copy()

#         if cut_idxs_free is not None:
#             for i, idxs in enumerate(cut_idxs_free):
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
#         return weights


#     def _invalidate_weight_cache(self):
#         self._free_mask_cache = None
#         self._free_mask_key = None
#         self._mw_cached = None
#         self._mw_lambda = None
#         self._mw_mu = None
#         self._mw_free_mask_key = None

   
#     def _get_free_mask(self):

#         fixed = frozenset(self.fixed_edges)
#         forbidden = frozenset(getattr(self, "excluded_edges", set()))
#         key = (fixed, forbidden)
#         if self._free_mask_cache is not None and self._free_mask_key == key:
#             return self._free_mask_cache

#         if not fixed and not forbidden:
#             self._free_mask_cache = None
#             self._free_mask_key = key
#             return None

#         mask = np.ones(len(self.edge_list), dtype=bool)
#         for e in fixed | forbidden:
#             idx = self.edge_indices.get(e)
#             if idx is not None:
#                 mask[idx] = False

#         self._free_mask_cache = mask
#         self._free_mask_key = key
#         return mask


#     def _append_with_cap(self, bucket, item, cap):
#         bucket.append(item)
#         overflow = len(bucket) - cap
#         if overflow > 0:
#             del bucket[:overflow]

#     def _record_primal_solution(self, mst_edges, feasible):
#         # snapshot = tuple(sorted(mst_edges)) if mst_edges else ()
#         snapshot = tuple(mst_edges) if mst_edges else ()

#         self._append_with_cap(
#             self.primal_solutions,
#             (snapshot, bool(feasible)),
#             self._primal_history_cap,
#         )

    
#     def _record_fractional_solution(self, fractional_solution):
#         if not fractional_solution:
#             lightweight = ()
#         else:
#             lightweight = tuple(
#                 heapq.nlargest(20, fractional_solution.items(), key=lambda kv: abs(kv[1]))
#             )
#         self._append_with_cap(self.fractional_solutions, lightweight, self._fractional_history_cap)


#     def _record_subgradient(self, value):
#         self._append_with_cap(
#             self.subgradients,
#             float(value),
#             self._subgradient_history_cap,
#         )




#     def custom_kruskal(self, modified_weights):
#         uf = UnionFind(self.num_nodes)

#         mst_edges = []
#         mst_edge_indices = []   # <--- NEW: track indices
#         mst_cost = 0.0

#         # Add fixed edges first
#         for i in self.fixed_edge_indices:
#             u, v = self.edge_list[i]
#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 mst_edge_indices.append(i)          # <--- NEW
#                 mst_cost += modified_weights[i]
#             else:
#                 return float('inf'), float('inf'), []

#         # Remaining candidate edges (canonical size!)
#         m = len(self.edge_list)
#         candidates = [i for i in range(m)
#                     if i not in self.fixed_edge_indices and i not in self.excluded_edge_indices]

#         candidates.sort(key=lambda i: modified_weights[i])

#         for i in candidates:
#             u, v = self.edge_list[i]
#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 mst_edge_indices.append(i)          # <--- NEW
#                 mst_cost += modified_weights[i]
#                 if len(mst_edges) == self.num_nodes - 1:
#                     break

#         # Connectivity / size check
#         if len(mst_edges) != self.num_nodes - 1 or uf.count_components() > 1:
#             return float('inf'), float('inf'), []

#         # Length computed by indices (consistent with everything)
#         mst_length = float(np.sum(self.edge_lengths[mst_edge_indices]))

#         return mst_cost, mst_length, mst_edges

    
  
#     def incremental_kruskal(self, prev_weights, prev_mst_edges, current_weights):
#         # Fast path (opt-in via flag, set only for negative-correlation runs):
#         # when lambda moves globally nearly every edge "changes", so the
#         # incremental candidate set degenerates to the full edge list and the
#         # Python `sorted` below dominates runtime. A single vectorized
#         # numpy.argsort over the whole array is far faster and yields an
#         # identical MST. Non-negative runs never set this flag, so they take
#         # the original code path below unchanged.
#         if getattr(self, "use_fast_kruskal", False):
#             return self._argsort_kruskal(current_weights)

#         uf = UnionFind(self.num_nodes)
#         mst_edges = []
#         mst_cost = 0.0
#         mst_length = 0.0

#         for edge_idx in self.fixed_edge_indices:
#             u, v = self.edge_list[edge_idx]
#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 mst_cost   += current_weights[edge_idx]
#                 mst_length += self.edge_lengths[edge_idx]
#             else:
#                 # Fixed edges already create a cycle -> infeasible
#                 return float('inf'), float('inf'), []

#         weight_changes = current_weights - prev_weights
#         changed_indices = np.where(np.abs(weight_changes) > self.cache_tolerance)[0]
#         changed_edges   = set(changed_indices)

#         prev_mst_indices = {
#             self.edge_indices[(u, v)] for u, v in prev_mst_edges
#             if self.edge_indices[(u, v)] not in self.fixed_edge_indices
#         }
#         candidate_indices = (
#             prev_mst_indices | changed_edges
#         ) - self.excluded_edge_indices - self.fixed_edge_indices

#         sorted_edges = sorted(candidate_indices, key=lambda i: current_weights[i])

#         for edge_idx in sorted_edges:
#             u, v = self.edge_list[edge_idx]
#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 mst_cost   += current_weights[edge_idx]
#                 mst_length += self.edge_lengths[edge_idx]

#         # NEW: cheap validity check – tree must have exactly n-1 edges
#         if len(mst_edges) != self.num_nodes - 1:
#             return float('inf'), float('inf'), []

#         return mst_cost, mst_length, mst_edges

#     def _argsort_kruskal(self, weights):
#         """Full Kruskal using numpy.argsort for the edge ordering. Honors
#         fixed/excluded edges identically to custom_kruskal. Used only on the
#         opt-in fast path; produces the same MST as the Python-sort version."""
#         uf = UnionFind(self.num_nodes)
#         mst_edges = []
#         mst_cost = 0.0
#         mst_length = 0.0

#         # Fixed edges first.
#         for i in self.fixed_edge_indices:
#             u, v = self.edge_list[i]
#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 mst_cost += float(weights[i])
#                 mst_length += float(self.edge_lengths[i])
#             else:
#                 return float('inf'), float('inf'), []

#         # Vectorized global ordering by weight.
#         order = np.argsort(weights, kind="stable")
#         fixed = self.fixed_edge_indices
#         excluded = self.excluded_edge_indices
#         need = self.num_nodes - 1

#         for i in order:
#             ii = int(i)
#             if ii in fixed or ii in excluded:
#                 continue
#             u, v = self.edge_list[ii]
#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 mst_cost += float(weights[ii])
#                 mst_length += float(self.edge_lengths[ii])
#                 if len(mst_edges) == need:
#                     break

#         if len(mst_edges) != need or uf.count_components() > 1:
#             return float('inf'), float('inf'), []

#         return mst_cost, mst_length, mst_edges
#     def compute_mst(self, modified_edges=None):
#         start_time = time()
        
#         if modified_edges is not None:
#             weights = np.array([w for _, _, w in modified_edges], dtype=float)
#         else:
#             weights = self.compute_modified_weights()

#         weights = np.nan_to_num(
#             weights,
#             nan=0.0,
#             posinf=1e9,
#             neginf=-1e9,
#             copy=False,
#         )

#         # ---- SIMPLE VERSION: NO HASHING, NO CACHE ----
#         mst_cost, mst_length, mst_edges = self.custom_kruskal(weights)
#         if self.verbose:
#             print(f"MST computed (no cache): length={mst_length:.2f}")

#         # Optionally remember last MST if you use it elsewhere
#         self.last_mst_edges = mst_edges

#         end_time = time()
#         LagrangianMST.total_compute_time += end_time - start_time
#         return mst_cost, mst_length, mst_edges



#     def compute_mst_incremental(self, prev_weights, prev_mst_edges):
#         # Compute current modified weights ONCE
#         current_weights = self.compute_modified_weights()
#         # Cache them so the caller (solve) can reuse without recomputing
#         self._last_mw = current_weights

#         # First call or no previous MST: just run full Kruskal
#         if prev_weights is None or prev_mst_edges is None:
#             if self.verbose:
#                 print("Incremental MST: no previous MST, using full custom_kruskal")
#             return self.custom_kruskal(current_weights)

#         weight_changes = current_weights - prev_weights

#         if np.all(np.abs(weight_changes) < 1e-6):
#             mst_cost = sum(current_weights[self.edge_indices[(u, v)]]
#                            for u, v in prev_mst_edges)
#             mst_length = sum(self.edge_lengths[self.edge_indices[(u, v)]]
#                              for u, v in prev_mst_edges)
#             if self.verbose:
#                 print(f"Incremental MST: Reusing previous MST with length={mst_length:.2f}")
#             return mst_cost, mst_length, prev_mst_edges

#         if self.verbose:
#             print(f"Incremental MST: Computing new MST due to weight changes")
#         return self.incremental_kruskal(prev_weights, prev_mst_edges, current_weights)



   

#     def solve(self, inherited_cuts=None, inherited_multipliers=None, depth=0, node=None):
#         start_time = time()
#         self.depth = depth
        
#         # --- robust normalization of inherited_cuts (accept pairs or indices) ---
#         edge_indices = self.edge_indices
#         idx_to_edge = {j: e for e, j in edge_indices.items()}

#         def _norm_edge(e):
#             if not (isinstance(e, tuple) and len(e) == 2):
#                 return None
#             u, v = e
#             t = (u, v) if u <= v else (v, u)
#             return t if t in edge_indices else None

#         def _iter_edges_any(cut_like):
#             if isinstance(cut_like, tuple) and len(cut_like) == 2:
#                 e = _norm_edge(cut_like); 
#                 if e is not None: yield e
#                 return
#             if isinstance(cut_like, int):
#                 e = _norm_edge(idx_to_edge.get(int(cut_like)))
#                 if e is not None: yield e
#                 return
#             try:
#                 for item in cut_like:
#                     if isinstance(item, int):
#                         e = _norm_edge(idx_to_edge.get(int(item)))
#                     elif isinstance(item, tuple) and len(item) == 2:
#                         e = _norm_edge(item)
#                     elif isinstance(item, (list, set, frozenset)) and len(item) == 2:
#                         a, b = tuple(item); e = _norm_edge((a, b))
#                     else:
#                         e = None
#                     if e is not None: 
#                         yield e
#             except TypeError:
#                 return

#         def _norm_pair(pair):
#             cut_like, rhs_like = pair
#             return (set(_iter_edges_any(cut_like)), int(rhs_like))

#         if inherited_cuts:
#             self.best_cuts = [_norm_pair(p) for p in inherited_cuts]
#             self.best_cut_multipliers = (inherited_multipliers or {}).copy()
#         else:
#             self.best_cuts = []
#             self.best_cut_multipliers = {}
#         self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()


#         # --- robust normalization of inherited_cuts (accept pairs or indices) ---
    
            
#         prev_weights = None
#         prev_mst_edges = None

       
#         if self.use_bisection:
#         # Validate graph and edges
#             if not self.edges or not nx.is_connected(self.graph):
#                 if self.verbose:
#                     print(f"Error at depth {depth}: Empty edge list or disconnected graph in bisection path")
#                 return self.best_lower_bound, self.best_upper_bound, []
            

#         # else:  # Subgradient method with Polyak hybrid + cover cuts (λ, μ), depth-based freezing
#         #     # --- Tunables / safety limits ---
#         #     MAX_SOLUTIONS    = getattr(self, "max_primal_solutions", 50)
#         #     max_iter         = min(self.max_iter, 200)

#         #     # Polyak / momentum for λ
#         #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
#         #     gamma_base         = getattr(self, "gamma_base", 0.1)

#         #     # μ update parameters
#         #     gamma_mu         = getattr(self, "gamma_mu", 0.30)
#         #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
#         #     eps              = 1e-12

#         #     # Depth-based behaviour
#         #     max_cut_depth = getattr(self, "max_cut_depth", 30)   # where we ADD cuts
#         #     max_mu_depth  = getattr(self, "max_mu_depth", 50)    # where we UPDATE μ / use cuts in dual
#         #     is_root       = (depth == 0)

#         #     # Node-level separation parameters
#         #     max_active_cuts           = getattr(self, "max_active_cuts", 5)
#         #     max_new_cuts_per_node     = getattr(self, "max_new_cuts_per_node", 5)
#         #     min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
#         #     dead_mu_threshold         = getattr(self, "dead_mu_threshold", 1e-6)

#         #     # Extra iterations allowed at root
#         #     root_max_iter = int(getattr(self, "root_max_iter", max_iter * 2))

#         #     # Ensure cut structures exist
#         #     if not hasattr(self, "best_cuts"):
#         #         self.best_cuts = []   # list of (set(edges), rhs)
#         #     if not hasattr(self, "best_cut_multipliers"):
#         #         self.best_cut_multipliers = {}  # μ_i for each cut
#         #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
#         #         self.best_cut_multipliers_for_best_bound = {}  # μ at best LB

#         #     # Which behaviour at this node?
#         #     cutting_active_here = self.use_cover_cuts and (depth <= max_cut_depth)   # can ADD cuts
#         #     mu_dynamic_here     = self.use_cover_cuts and (depth <= max_mu_depth)    # can UPDATE μ / use in dual
#         #     cuts_present_here   = self.use_cover_cuts and bool(self.best_cuts)

#         #     # Ensure λ starts in a reasonable range (consistent with compute_modified_weights)
#         #     self.lmbda = max(0.0, min(getattr(self, "lmbda", 0.05), 1e4))

#         #     polyak_enabled = True

#         #     # Collect newly generated cuts at this node
#         #     node_new_cuts = []

#         #     # --- Quick guards ---
#         #     if not self.edge_list or self.num_nodes <= 1:
#         #         if self.verbose:
#         #             print(f"Error at depth {depth}: Empty edge list or invalid graph")
#         #         end_time = time()
#         #         LagrangianMST.total_compute_time += end_time - start_time
#         #         return self.best_lower_bound, self.best_upper_bound, node_new_cuts

#         #     # Fixed / forbidden edges
#         #     F_in  = getattr(self, "fixed_edges", set())
#         #     F_out = getattr(self, "excluded_edges", set())
#         #     edge_idx = self.edge_indices
#         #     if not hasattr(self, "_rhs_eff"):
#         #         self._rhs_eff = {}

#         #     # ------------------------------------------------------------------
#         #     # Separation policy (FIXED):
#         #     #   - DO NOT do objective-only pre-separation at root.
#         #     #   - Always delay separation to the first violating MST inside the loop.
#         #     #   - Still obey depth limits: only add cuts when cutting_active_here AND μ is dynamic.
#         #     # ------------------------------------------------------------------
#         #     pending_sep = bool(cutting_active_here and mu_dynamic_here)

#         #     # ------------------------------------------------------------------
#         #     # 2) Compute rhs_eff and detect infeasibility (fixed edges + cuts)
#         #     #    rhs_eff = rhs - |cut ∩ F_in|
#         #     # ------------------------------------------------------------------
#         #     if self.use_cover_cuts and self.best_cuts:
#         #         for idx_c, (cut, rhs) in enumerate(self.best_cuts):
#         #             rhs_eff = int(rhs) - len(cut & F_in)
#         #             self._rhs_eff[idx_c] = rhs_eff
#         #             if rhs_eff < 0:
#         #                 end_time = time()
#         #                 LagrangianMST.total_compute_time += end_time - start_time
#         #                 return float('inf'), self.best_upper_bound, node_new_cuts

#         #     # ------------------------------------------------------------------
#         #     # 3) Trim number of cuts (keep at most max_active_cuts)
#         #     # ------------------------------------------------------------------
#         #     if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
#         #         parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)
#         #         if not parent_mu_map:
#         #             parent_mu_map = self.best_cut_multipliers

#         #         idx_and_cut = list(enumerate(self.best_cuts))
#         #         idx_and_cut.sort(
#         #             key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
#         #             reverse=True
#         #         )
#         #         idx_and_cut = idx_and_cut[:max_active_cuts]

#         #         new_cuts_list = []
#         #         new_mu       = {}
#         #         new_mu_best  = {}
#         #         new_rhs_eff  = {}

#         #         for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
#         #             new_cuts_list.append(cut_rhs)
#         #             new_mu[new_i]      = float(parent_mu_map.get(old_i, 0.0))
#         #             new_mu_best[new_i] = float(parent_mu_map.get(old_i, 0.0))
#         #             new_rhs_eff[new_i] = self._rhs_eff[old_i]

#         #         self.best_cuts = new_cuts_list
#         #         self.best_cut_multipliers = new_mu
#         #         self.best_cut_multipliers_for_best_bound = new_mu_best
#         #         self._rhs_eff = new_rhs_eff

#         #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

#         #     # ------------------------------------------------------------------
#         #     # 4) Build cut -> edge index arrays (for pricing/subgradients)
#         #     # ------------------------------------------------------------------
#         #     def _rebuild_cut_structures():
#         #         nonlocal cut_edge_idx_free, cut_edge_idx_all, rhs_eff_vec

#         #         cut_edge_idx_free = []
#         #         cut_edge_idx_all  = []

#         #         for cut, rhs in self.best_cuts:
#         #             idxs_free = [
#         #                 edge_idx[e] for e in cut
#         #                 if (e not in F_in and e not in F_out) and (e in edge_idx)
#         #             ]
#         #             arr_free = (
#         #                 np.fromiter(idxs_free, dtype=np.int32)
#         #                 if idxs_free else np.empty(0, dtype=np.int32)
#         #             )
#         #             cut_edge_idx_free.append(arr_free)

#         #             idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
#         #             arr_all  = (
#         #                 np.fromiter(idxs_all, dtype=np.int32)
#         #                 if idxs_all else np.empty(0, dtype=np.int32)
#         #             )
#         #             cut_edge_idx_all.append(arr_all)

#         #         self._cut_edge_idx     = cut_edge_idx_free
#         #         self._cut_edge_idx_all = cut_edge_idx_all

#         #         rhs_eff_vec = (
#         #             np.array([self._rhs_eff[i] for i in range(len(self.best_cuts))], dtype=float)
#         #             if self.best_cuts else np.zeros(0, dtype=float)
#         #         )

#         #     cut_edge_idx_free = []
#         #     cut_edge_idx_all  = []
#         #     rhs_eff_vec       = np.zeros(0, dtype=float)

#         #     if self.use_cover_cuts and self.best_cuts:
#         #         _rebuild_cut_structures()

#         #     # Track usefulness of cuts at this node
#         #     max_cut_violation = [0.0 for _ in self.best_cuts]

#         #     # Histories / caches
#         #     self._mw_cached = None
#         #     self._mw_lambda = None
#         #     self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)

#         #     if not hasattr(self, "subgradients"):
#         #         self.subgradients = []
#         #     if not hasattr(self, "step_sizes"):
#         #         self.step_sizes = []
#         #     if not hasattr(self, "multipliers"):
#         #         self.multipliers = []

#         #     prev_weights   = None
#         #     prev_mst_edges = None

#         #     if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
#         #         self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
#         #     mst_mask = self._mst_mask

#         #     # Decide iteration limit for this node:
#         #     if is_root:
#         #         iter_limit = root_max_iter * 1.1 if self.use_cover_cuts else root_max_iter
#         #     else:
#         #         iter_limit = max_iter
#         #     # ------------------------------------------------------------------
#         #     # 5) Subgradient iterations
#         #     # ------------------------------------------------------------------
#         #     for iter_num in range(int(iter_limit)):
#         #         # 1) MST with current λ, μ              
#         #         try:
#         #             mst_cost, mst_length, mst_edges = self.compute_mst_incremental(prev_weights, prev_mst_edges)
#         #         except Exception:
#         #             mst_cost, mst_length, mst_edges = self.compute_mst()

#         #         self.last_mst_edges = mst_edges
#         #         prev_mst_edges      = mst_edges
#         #         cut_g_signed = []

#         #         # 1a) ONE-SHOT delayed separation (root AND non-root)
#         #         if (
#         #             cutting_active_here
#         #             and mu_dynamic_here
#         #             and pending_sep
#         #             and len(self.best_cuts) < max_active_cuts
#         #             and mst_length > self.budget
#         #         ):
#         #             try:
#         #                 cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []
#         #                 print("sss")

#         #                 T_loop = set(mst_edges)
#         #                 scored_loop = []
#         #                 F_in_set = set(F_in)  # (already defined above)

#         #                 for cut, rhs in cand_cuts_loop:
#         #                     S_set   = set(cut)
#         #                     S_free  = S_set - F_in_set                 # remove fixed edges from LHS set
#         #                     lhs_free = len(T_loop & S_free)            # only MST edges that are NOT fixed
#         #                     rhs_eff  = int(rhs) - len(S_set & F_in_set)
#         #                     violation = lhs_free - rhs_eff

#         #                     if violation >= min_cut_violation_for_add:
#         #                         scored_loop.append((violation, S_set, rhs))

#         #                 scored_loop.sort(reverse=True, key=lambda t: t[0])

#         #                 remaining_slots = max(0, max_active_cuts - len(self.best_cuts))
#         #                 if remaining_slots > 0:
#         #                     scored_loop = scored_loop[:min(max_new_cuts_per_node, remaining_slots)]
#         #                 else:
#         #                     scored_loop = []

#         #                 existing = {frozenset(c): rhs for (c, rhs) in self.best_cuts}
#         #                 added_any = False

#         #                 for violation, S, rhs in scored_loop:
#         #                     fz = frozenset(S)
#         #                     if fz in existing:
#         #                         continue

#         #                     self.best_cuts.append((set(S), rhs))
#         #                     new_idx = len(self.best_cuts) - 1
#         #                     MU0 = getattr(self, "mu_init", 0.0)  # safe default: 0 (avoid immediate decay overhead)
#         #                     self.best_cut_multipliers[new_idx] = MU0
#         #                     self.best_cut_multipliers_for_best_bound[new_idx] = MU0


#         #                     # keep rhs_eff consistent
#         #                     self._rhs_eff[new_idx] = int(rhs) - len(set(S) & F_in)
#         #                     if self._rhs_eff[new_idx] < 0:
#         #                         end_time = time()
#         #                         LagrangianMST.total_compute_time += end_time - start_time
#         #                         return float('inf'), self.best_upper_bound, node_new_cuts

#         #                     max_cut_violation.append(0.0)
#         #                     node_new_cuts.append((set(S), rhs))
#         #                     added_any = True

#         #                 if added_any:
#         #                     _rebuild_cut_structures()
#         #                     self._mw_cached = None
#         #                     self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)
#         #                     cuts_present_here = True

#         #             except Exception as e:
#         #                 if self.verbose:
#         #                     print(f"Error in delayed separation at depth {depth}, iter {iter_num}: {e}")
#         #             finally:
#         #                 pending_sep = False  # do at most once per node

#         #         # Prepare weights for next iteration (cache)
#         #         prev_weights = getattr(self, "_last_mw", prev_weights)

#         #         # 2) Primal & UB
#         #         is_feasible = (mst_length <= self.budget)
#         #         self._record_primal_solution(self.last_mst_edges, is_feasible)

#         #         if is_feasible:
#         #             try:
#         #                 real_weight, real_length = self.compute_real_weight_length()
#         #                 if (
#         #                     not math.isnan(real_weight)
#         #                     and not math.isinf(real_weight)
#         #                     and real_weight < self.best_upper_bound
#         #                 ):
#         #                     self.best_upper_bound = real_weight
#         #             except Exception as e:
#         #                 if self.verbose:
#         #                     print(f"Error updating primal solution: {e}")

#         #         if len(self.primal_solutions) > MAX_SOLUTIONS:
#         #             self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

#         #         # 3) Dual value: L(λ, μ) = MST_cost - λ B - Σ μ_i rhs_eff_i
#         #         lam_for_dual = max(0.0, min(self.lmbda, 1e4))

#         #         if self.use_cover_cuts and len(rhs_eff_vec) > 0:
#         #             mu_vec = np.fromiter(
#         #                 (
#         #                     max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
#         #                     for i in range(len(rhs_eff_vec))
#         #                 ),
#         #                 dtype=float,
#         #                 count=len(rhs_eff_vec),
#         #             )
#         #             cover_cut_penalty = float(mu_vec @ rhs_eff_vec)
#         #         else:
#         #             cover_cut_penalty = 0.0

#         #         lagrangian_bound = mst_cost - lam_for_dual * self.budget - cover_cut_penalty
#         #         # if cover_cut_penalty != 0.0:
#         #             # print("ggg", cover_cut_penalty)
#         #         # print("lagrangian bound:", lagrangian_bound)

#         #         if (
#         #             not math.isnan(lagrangian_bound)
#         #             and not math.isinf(lagrangian_bound)
#         #             and abs(lagrangian_bound) < 1e10
#         #         ):
#         #             if lagrangian_bound > self.best_lower_bound + 1e-6:
#         #                 self.best_lower_bound = lagrangian_bound
#         #                 self.best_lambda      = lam_for_dual
#         #                 self.best_mst_edges   = self.last_mst_edges
#         #                 self.best_cost        = mst_cost
#         #                 self.best_cut_multipliers_for_best_bound = self.best_cut_multipliers.copy()

#         #         # 4) Subgradients
#         #         knapsack_subgradient = float(mst_length - self.budget)
#         #         # print("fff", mst_length)
#         #         # print("lala",self.lmbda)
#         #         # print("wer", knapsack_subgradient)

#         #         # Fast skip: if MST feasible and all μ are ~0, don't pay cut gradient cost
#         #         all_mu_small = (not self.best_cut_multipliers) or \
#         #                     (max(self.best_cut_multipliers.values()) <= dead_mu_threshold)

#         #         if cuts_present_here and mu_dynamic_here and len(cut_edge_idx_all) > 0 and not (is_feasible and all_mu_small):
#         #             mst_mask[:] = False
#         #             for e in mst_edges:
#         #                 j = self.edge_indices.get(e)
#         #                 if j is not None:
#         #                     mst_mask[j] = True

#         #             cut_g_signed = []
#         #             cut_g_pos    = []

#         #             for i, idxs_free in enumerate(cut_edge_idx_free):
#         #                 lhs_free = int(mst_mask[idxs_free].sum()) if idxs_free.size else 0
#         #                 g_i = float(lhs_free) - float(rhs_eff_vec[i])
#         #                 cut_g_signed.append(g_i)
#         #                 cut_g_pos.append(g_i if g_i > 0.0 else 0.0)

#         #                 if g_i > max_cut_violation[i]:
#         #                     max_cut_violation[i] = g_i

#         #             cut_subgradients = cut_g_pos
#         #         else:
#         #             cut_subgradients = []
#         #             cut_g_signed = []
#         #             cut_g_pos = []


#         #         norm_sq = knapsack_subgradient ** 2
#         #         for g in cut_subgradients:
#         #             norm_sq += float(g) ** 2

#         #         # Polyak step size
#         #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
#         #             gap   = max(0.0, self.best_upper_bound - lagrangian_bound)
#         #             alpha = gamma_base * gap / (norm_sq + eps)
#         #         else:
#         #             alpha = getattr(self, "step_size", 0.001)

#         #         # λ update with momentum, then clamp
#         #         v_prev = getattr(self, "_v_lambda", 0.0)
#         #         v_new  = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
#         #         self._v_lambda = v_new
#         #         self.lmbda     = self.lmbda + alpha * v_new
#         #         # print("ooo", alpha)

#         #         if self.lmbda < 0.0:
#         #             self.lmbda = 0.0
#         #         if self.lmbda > 1e4:
#         #             self.lmbda = 1e4

#         #         # μ updates: projected subgradient for constraints sum_{e in S} x_e <= rhs_eff
#         #         if mu_dynamic_here and len(cut_g_pos) > 0:
#         #             for i, g in enumerate(cut_g_pos):
#         #                 g = float(g)
#         #                 if g <= 0.0:
#         #                     continue

#         #                 delta = gamma_mu * alpha * g

#         #                 # cap only positive increment
#         #                 if mu_increment_cap is not None:
#         #                     delta = min(mu_increment_cap, delta)

#         #                 mu_old = float(self.best_cut_multipliers.get(i, 0.0))
#         #                 mu_new = mu_old + delta

#         #                 # projection + clamp
#         #                 if mu_new > 1e4:
#         #                     mu_new = 1e4

#         #                 self.best_cut_multipliers[i] = mu_new


#         #         self.step_sizes.append(alpha)
#         #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

#         #     # ------------------------------------------------------------------
#         #     # 6) Drop "dead" cuts globally
#         #     # ------------------------------------------------------------------
#         #     if self.use_cover_cuts and self.best_cuts and mu_dynamic_here:
#         #         keep_indices = []

#         #         parent_mu_map = getattr(
#         #             self,
#         #             "best_cut_multipliers_for_best_bound",
#         #             self.best_cut_multipliers,
#         #         )

#         #         for i, (cut, rhs) in enumerate(self.best_cuts):
#         #             mu_i    = float(self.best_cut_multipliers.get(i, 0.0))
#         #             mu_hist = float(parent_mu_map.get(i, 0.0))

#         #             ever_useful = (i < len(max_cut_violation) and max_cut_violation[i] > 0.0) \
#         #                         or (abs(mu_hist) >= dead_mu_threshold)

#         #             if (not ever_useful) and abs(mu_i) < dead_mu_threshold and abs(mu_hist) < dead_mu_threshold:
#         #                 continue
#         #             keep_indices.append(i)

#         #         if len(keep_indices) < len(self.best_cuts):
#         #             new_best_cuts = []
#         #             new_mu        = {}
#         #             new_mu_best   = {}
#         #             new_rhs_eff   = {}

#         #             for new_idx, old_idx in enumerate(keep_indices):
#         #                 new_best_cuts.append(self.best_cuts[old_idx])
#         #                 new_mu[new_idx]      = float(self.best_cut_multipliers.get(old_idx, 0.0))
#         #                 new_mu_best[new_idx] = float(self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0))
#         #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]

#         #             self.best_cuts = new_best_cuts
#         #             self.best_cut_multipliers = new_mu
#         #             self.best_cut_multipliers_for_best_bound = new_mu_best
#         #             self._rhs_eff = new_rhs_eff

#         #     # ------------------------------------------------------------------
#         #     # 7) Restore best (λ, μ) to pass to children
#         #     # ------------------------------------------------------------------
#         #     if hasattr(self, "best_lambda"):
#         #         self.lmbda = self.best_lambda

#         #     if mu_dynamic_here and hasattr(self, "best_cut_multipliers_for_best_bound"):
#         #         self.best_cut_multipliers = self.best_cut_multipliers_for_best_bound.copy()

#         #     end_time = time()
#         #     LagrangianMST.total_compute_time += end_time - start_time
#         #     return self.best_lower_bound, self.best_upper_bound, node_new_cuts

        
#         else:  # Subgradient method with Polyak hybrid + cover cuts (λ, μ), depth-based freezing
#             import os

#             # --- Tunables / safety limits ---
#             MAX_SOLUTIONS = getattr(self, "max_primal_solutions", 50)
#             max_iter = min(self.max_iter, 200)

#             # Polyak / momentum for λ
#             self.momentum_beta = getattr(self, "momentum_beta", 0.7)
#             gamma_base = getattr(self, "gamma_base", 0.05)

#             # Safety controls for λ update
#             fallback_alpha = getattr(self, "fallback_alpha", 1e-5)
#             max_lambda_delta = getattr(self, "max_lambda_delta", 0.02)

#             # μ update parameters
#             gamma_mu = getattr(self, "gamma_mu", 0.25)
#             mu_increment_cap = getattr(self, "mu_increment_cap", 0.002)

#             eps = 1e-12

#             # Depth-based behaviour
#             max_cut_depth = getattr(self, "max_cut_depth", 30)
#             max_mu_depth = getattr(self, "max_mu_depth", 50)
#             is_root = depth == 0

#             # Node-level separation parameters
#             max_active_cuts = getattr(self, "max_active_cuts", 5)
#             max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)
#             min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
#             dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-6)

#             # Extra iterations allowed at root
#             root_max_iter = int(getattr(self, "root_max_iter", max_iter * 2))

#             # ------------------------------------------------------------------
#             # DEBUG SETTINGS
#             # ------------------------------------------------------------------
#             debug_cuts = False
#             debug_iter_every = 1       # change to 5 or 10 if the log becomes too large
#             debug_cut_max_rows = 10

#             debug_log_path = getattr(
#                 self,
#                 "debug_cut_log_path",
#                 os.path.join(os.path.expanduser("~/Desktop"), "cut_debug_log.txt"),
#             )

#             # Clear the log only once at the root node
#             if depth == 0:
#                 with open(debug_log_path, "w") as f:
#                     f.write("CUT DEBUG LOG\n")
#                     f.write("=" * 100 + "\n")

#             def _dbg(msg, iter_num=None, force=False):
#                 if not debug_cuts:
#                     return

#                 if iter_num is not None and not force:
#                     if iter_num % debug_iter_every != 0:
#                         return

#                 if iter_num is None:
#                     line = f"[CUTDBG depth={depth}] {msg}"
#                 else:
#                     line = f"[CUTDBG depth={depth} iter={iter_num}] {msg}"

#                 with open(debug_log_path, "a") as f:
#                     f.write(line + "\n")

#             def _edge_len(e):
#                 try:
#                     return float(self.edge_lengths[self.edge_indices[e]])
#                 except Exception:
#                     return float("nan")

#             def _cut_len(cut):
#                 return sum(_edge_len(e) for e in cut if e in self.edge_indices)

#             def _cut_repr(cut, max_edges=6):
#                 cut_list = sorted(list(cut))
#                 shown = cut_list[:max_edges]
#                 suffix = "" if len(cut_list) <= max_edges else f", ... +{len(cut_list) - max_edges}"
#                 return f"{shown}{suffix}"

#             def _print_cut_table(stage, iter_num=None, force=False):
#                 if not debug_cuts:
#                     return

#                 if iter_num is not None and not force:
#                     if iter_num % debug_iter_every != 0:
#                         return

#                 _dbg(
#                     f"{stage}: active cuts = {len(getattr(self, 'best_cuts', []))}",
#                     iter_num,
#                     force,
#                 )

#                 if not getattr(self, "best_cuts", []):
#                     return

#                 for i, (cut, rhs) in enumerate(self.best_cuts[:debug_cut_max_rows]):
#                     mu = float(getattr(self, "best_cut_multipliers", {}).get(i, 0.0))
#                     mu_best = float(getattr(self, "best_cut_multipliers_for_best_bound", {}).get(i, 0.0))
#                     rhs_eff = getattr(self, "_rhs_eff", {}).get(i, rhs)

#                     _dbg(
#                         f"  cut[{i}] size={len(cut)} rhs={rhs} rhs_eff={rhs_eff} "
#                         f"mu={mu:.6g} mu_best={mu_best:.6g} "
#                         f"len_sum={_cut_len(cut):.3f} edges={_cut_repr(cut)}",
#                         iter_num,
#                         force,
#                     )

#                 if len(self.best_cuts) > debug_cut_max_rows:
#                     _dbg(
#                         f"  ... {len(self.best_cuts) - debug_cut_max_rows} more cuts not shown",
#                         iter_num,
#                         force,
#                     )

#             # ------------------------------------------------------------------
#             # Ensure cut structures exist
#             # ------------------------------------------------------------------
#             if not hasattr(self, "best_cuts"):
#                 self.best_cuts = []

#             if not hasattr(self, "best_cut_multipliers"):
#                 self.best_cut_multipliers = {}

#             if not hasattr(self, "best_cut_multipliers_for_best_bound"):
#                 self.best_cut_multipliers_for_best_bound = {}

#             # Which behaviour at this node?
#             cutting_active_here = self.use_cover_cuts and depth <= max_cut_depth
#             mu_dynamic_here = self.use_cover_cuts and depth <= max_mu_depth
#             use_cuts_in_dual_here = self.use_cover_cuts and bool(self.best_cuts)

#             # Ensure λ starts in a reasonable range
#             self.lmbda = max(0.0, min(getattr(self, "lmbda", 0.05), 1e4))

#             polyak_enabled = True
#             node_new_cuts = []

#             _dbg(
#                 f"START NODE | use_cover_cuts={self.use_cover_cuts}, "
#                 f"cutting_active_here={cutting_active_here}, "
#                 f"mu_dynamic_here={mu_dynamic_here}, "
#                 f"use_cuts_in_dual_here={use_cuts_in_dual_here}, "
#                 f"lambda_start={self.lmbda:.6g}, "
#                 f"inherited_cuts={len(self.best_cuts)}, "
#                 f"log_file={debug_log_path}",
#                 force=True,
#             )

#             _print_cut_table("Inherited cuts before reduction", force=True)

#             # ------------------------------------------------------------------
#             # Quick guards
#             # ------------------------------------------------------------------
#             if not self.edge_list or self.num_nodes <= 1:
#                 _dbg("STOP: empty edge list or invalid graph", force=True)

#                 end_time = time()
#                 LagrangianMST.total_compute_time += end_time - start_time
#                 return self.best_lower_bound, self.best_upper_bound, node_new_cuts

#             # Fixed / forbidden edges
#             F_in = set(getattr(self, "fixed_edges", set()))
#             F_out = set(getattr(self, "excluded_edges", set()))
#             edge_idx = self.edge_indices

#             self._rhs_eff = {}

#             _dbg(
#                 f"Node fixings: |F_in|={len(F_in)}, |F_out|={len(F_out)}, "
#                 f"fixed_length={sum(_edge_len(e) for e in F_in if e in edge_idx):.3f}, "
#                 f"budget={self.budget:.3f}",
#                 force=True,
#             )

#             # ------------------------------------------------------------------
#             # 2) Reduce inherited cuts and remove redundant cuts
#             # ------------------------------------------------------------------
#             if self.use_cover_cuts and self.best_cuts:
#                 old_mu = dict(getattr(self, "best_cut_multipliers", {}) or {})
#                 old_mu_best = dict(getattr(self, "best_cut_multipliers_for_best_bound", {}) or {})

#                 reduced_cuts = []
#                 reduced_mu = {}
#                 reduced_mu_best = {}
#                 reduced_rhs_eff = {}

#                 kept_count = 0
#                 redundant_count = 0

#                 for old_i, (cut, rhs) in enumerate(self.best_cuts):
#                     S = set(cut)

#                     S_fixed = S & F_in
#                     S_excluded = S & F_out
#                     S_free = S - F_in - F_out
#                     rhs_eff = int(rhs) - len(S_fixed)

#                     _dbg(
#                         f"Reduce old_cut[{old_i}]: old_size={len(S)}, old_rhs={rhs}, "
#                         f"|S_fixed|={len(S_fixed)}, |S_excluded|={len(S_excluded)}, "
#                         f"|S_free|={len(S_free)}, rhs_eff={rhs_eff}, "
#                         f"mu_old={float(old_mu.get(old_i, 0.0)):.6g}",
#                         force=True,
#                     )

#                     if rhs_eff < 0:
#                         _dbg(
#                             f"STOP: inherited cut[{old_i}] makes node infeasible "
#                             f"because rhs_eff={rhs_eff}<0",
#                             force=True,
#                         )

#                         end_time = time()
#                         LagrangianMST.total_compute_time += end_time - start_time
#                         return float("inf"), self.best_upper_bound, node_new_cuts

#                     # Redundant at this node
#                     if len(S_free) <= rhs_eff:
#                         redundant_count += 1
#                         _dbg(
#                             f"Drop old_cut[{old_i}] as redundant: "
#                             f"|S_free|={len(S_free)} <= rhs_eff={rhs_eff}",
#                             force=True,
#                         )
#                         continue

#                     new_i = len(reduced_cuts)
#                     reduced_cuts.append((set(S_free), int(rhs_eff)))

#                     mu_val = float(old_mu.get(old_i, 0.0))
#                     mu_best_val = float(old_mu_best.get(old_i, mu_val))

#                     reduced_mu[new_i] = mu_val
#                     reduced_mu_best[new_i] = mu_best_val
#                     reduced_rhs_eff[new_i] = int(rhs_eff)
#                     kept_count += 1

#                 self.best_cuts = reduced_cuts
#                 self.best_cut_multipliers = reduced_mu
#                 self.best_cut_multipliers_for_best_bound = reduced_mu_best
#                 self._rhs_eff = reduced_rhs_eff

#                 _dbg(
#                     f"Cut reduction summary: kept={kept_count}, "
#                     f"redundant_dropped={redundant_count}",
#                     force=True,
#                 )

#             _print_cut_table("Cuts after reduction", force=True)

#             # ------------------------------------------------------------------
#             # 3) Trim number of cuts
#             # ------------------------------------------------------------------
#             if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
#                 parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)

#                 if not parent_mu_map:
#                     parent_mu_map = self.best_cut_multipliers

#                 idx_and_cut = list(enumerate(self.best_cuts))
#                 idx_and_cut.sort(
#                     key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
#                     reverse=True,
#                 )

#                 kept_old_indices = [old_i for old_i, _ in idx_and_cut[:max_active_cuts]]
#                 dropped_old_indices = [old_i for old_i, _ in idx_and_cut[max_active_cuts:]]

#                 _dbg(
#                     f"Trim cuts: max_active_cuts={max_active_cuts}, "
#                     f"kept_old_indices={kept_old_indices}, "
#                     f"dropped_old_indices={dropped_old_indices}",
#                     force=True,
#                 )

#                 idx_and_cut = idx_and_cut[:max_active_cuts]

#                 new_cuts_list = []
#                 new_mu = {}
#                 new_mu_best = {}
#                 new_rhs_eff = {}

#                 for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
#                     new_cuts_list.append(cut_rhs)
#                     new_mu[new_i] = float(self.best_cut_multipliers.get(old_i, 0.0))
#                     new_mu_best[new_i] = float(parent_mu_map.get(old_i, new_mu[new_i]))
#                     new_rhs_eff[new_i] = int(self._rhs_eff.get(old_i, cut_rhs[1]))

#                 self.best_cuts = new_cuts_list
#                 self.best_cut_multipliers = new_mu
#                 self.best_cut_multipliers_for_best_bound = new_mu_best
#                 self._rhs_eff = new_rhs_eff

#             cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)
#             use_cuts_in_dual_here = self.use_cover_cuts and bool(self.best_cuts)

#             _print_cut_table("Cuts after trimming", force=True)

#             # ------------------------------------------------------------------
#             # 4) Build cut -> edge index arrays
#             # ------------------------------------------------------------------
#             def _rebuild_cut_structures():
#                 nonlocal cut_edge_idx_free, cut_edge_idx_all, rhs_eff_vec

#                 cut_edge_idx_free = []
#                 cut_edge_idx_all = []

#                 for i, (cut, rhs) in enumerate(self.best_cuts):
#                     S = set(cut)

#                     idxs_free = [
#                         edge_idx[e]
#                         for e in S
#                         if e in edge_idx and e not in F_in and e not in F_out
#                     ]

#                     arr_free = (
#                         np.fromiter(idxs_free, dtype=np.int32)
#                         if idxs_free
#                         else np.empty(0, dtype=np.int32)
#                     )

#                     cut_edge_idx_free.append(arr_free)

#                     idxs_all = [edge_idx[e] for e in S if e in edge_idx]

#                     arr_all = (
#                         np.fromiter(idxs_all, dtype=np.int32)
#                         if idxs_all
#                         else np.empty(0, dtype=np.int32)
#                     )

#                     cut_edge_idx_all.append(arr_all)

#                     if i not in self._rhs_eff:
#                         self._rhs_eff[i] = int(rhs)

#                 self._cut_edge_idx = cut_edge_idx_free
#                 self._cut_edge_idx_all = cut_edge_idx_all

#                 rhs_eff_vec = (
#                     np.array(
#                         [self._rhs_eff[i] for i in range(len(self.best_cuts))],
#                         dtype=float,
#                     )
#                     if self.best_cuts
#                     else np.zeros(0, dtype=float)
#                 )

#                 _dbg(
#                     f"Rebuilt cut structures: num_cuts={len(self.best_cuts)}, "
#                     f"rhs_eff_vec={rhs_eff_vec.tolist()}, "
#                     f"free_edge_counts={[len(a) for a in cut_edge_idx_free]}",
#                     force=True,
#                 )

#             cut_edge_idx_free = []
#             cut_edge_idx_all = []
#             rhs_eff_vec = np.zeros(0, dtype=float)

#             if self.use_cover_cuts and self.best_cuts:
#                 _rebuild_cut_structures()

#             max_cut_violation = [0.0 for _ in self.best_cuts]

#             # Histories / caches
#             self._mw_cached = None
#             self._mw_lambda = None
#             self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

#             if not hasattr(self, "subgradients"):
#                 self.subgradients = []

#             if not hasattr(self, "step_sizes"):
#                 self.step_sizes = []

#             if not hasattr(self, "multipliers"):
#                 self.multipliers = []

#             prev_weights = None
#             prev_mst_edges = None

#             if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
#                 self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)

#             mst_mask = self._mst_mask

#             # Decide iteration limit for this node
#             if is_root:
#                 iter_limit = root_max_iter * 1.1 if self.use_cover_cuts else root_max_iter
#             else:
#                 # Optional depth decay: with lambda inheritance, deep children
#                 # only need to REFINE the parent's near-optimal lambda, not
#                 # rediscover it, so fewer iterations suffice. Controlled by
#                 # `child_iter_decay` (per-level multiplier) and `child_min_iter`
#                 # (floor). Both default to no-op values, so when unset the cap
#                 # is exactly the old flat `max_iter` -> non-negative runs, which
#                 # set neither, are unaffected.
#                 decay = getattr(self, "child_iter_decay", 1.0)
#                 min_iter = getattr(self, "child_min_iter", max_iter)
#                 if decay < 1.0 and depth > 0:
#                     decayed = int(round(max_iter * (decay ** depth)))
#                     iter_limit = max(min_iter, decayed)
#                 else:
#                     iter_limit = max_iter

#             sep_rounds = 0
#             max_sep_rounds = 1
#             separate_every = 5

#             _dbg(
#                 f"Iteration setup: iter_limit={int(iter_limit)}, "
#                 f"max_sep_rounds={max_sep_rounds}, "
#                 f"separate_every={separate_every}",
#                 force=True,
#             )

#             # ------------------------------------------------------------------
#             # 5) Subgradient iterations
#             # ------------------------------------------------------------------
#             for iter_num in range(int(iter_limit)):
#                 # --------------------------------------------------------------
#                 # 5.1) MST with current λ and μ
#                 # --------------------------------------------------------------
#                 try:
#                     mst_cost, mst_length, mst_edges = self.compute_mst_incremental(
#                         prev_weights,
#                         prev_mst_edges,
#                     )
#                     mst_method = "incremental"

#                 except Exception as e:
#                     _dbg(
#                         f"Incremental MST failed: {e}. Falling back to full MST.",
#                         iter_num,
#                         force=True,
#                     )

#                     mst_cost, mst_length, mst_edges = self.compute_mst()
#                     mst_method = "full"

#                 if (
#                     not mst_edges
#                     or math.isinf(mst_cost)
#                     or math.isinf(mst_length)
#                     or math.isnan(mst_cost)
#                     or math.isnan(mst_length)
#                 ):
#                     _dbg(
#                         f"STOP: invalid MST. method={mst_method}, "
#                         f"mst_cost={mst_cost}, mst_length={mst_length}, "
#                         f"num_edges={len(mst_edges) if mst_edges else 0}",
#                         iter_num,
#                         force=True,
#                     )

#                     end_time = time()
#                     LagrangianMST.total_compute_time += end_time - start_time
#                     return float("inf"), self.best_upper_bound, node_new_cuts

#                 self.last_mst_edges = mst_edges
#                 prev_mst_edges = mst_edges

#                 _dbg(
#                     f"MST: method={mst_method}, cost={mst_cost:.6g}, "
#                     f"length={mst_length:.6g}, budget={self.budget:.6g}, "
#                     f"budget_violation={mst_length - self.budget:.6g}, "
#                     f"num_edges={len(mst_edges)}",
#                     iter_num,
#                 )

#                 # --------------------------------------------------------------
#                 # 5.2) Delayed separation
#                 #
#                 # Modified:
#                 # We now separate at the first budget-violating MST, not only at
#                 # iteration 0 or multiples of separate_every.
#                 # max_sep_rounds still limits this to one separation round per node.
#                 # --------------------------------------------------------------
#                 should_separate = (
#                     cutting_active_here
#                     and mu_dynamic_here
#                     and sep_rounds < max_sep_rounds
#                     and len(self.best_cuts) < max_active_cuts
#                     and mst_length > self.budget
#                 )

#                 _dbg(
#                     f"Separation check: should_separate={should_separate}, "
#                     f"sep_rounds={sep_rounds}/{max_sep_rounds}, "
#                     f"active_cuts={len(self.best_cuts)}/{max_active_cuts}, "
#                     f"budget_violated={mst_length > self.budget}",
#                     iter_num,
#                 )

#                 if should_separate:
#                     try:
#                         cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []

#                         _dbg(
#                             f"Generated candidate cuts: count={len(cand_cuts_loop)}",
#                             iter_num,
#                             force=True,
#                         )

#                         T_loop = set(mst_edges)
#                         scored_loop = []

#                         for cand_i, (cut, rhs) in enumerate(cand_cuts_loop):
#                             S_set = set(cut)

#                             S_fixed = S_set & F_in
#                             S_excluded = S_set & F_out
#                             S_free = S_set - F_in - F_out
#                             rhs_eff_new = int(rhs) - len(S_fixed)

#                             if rhs_eff_new < 0:
#                                 _dbg(
#                                     f"STOP: candidate cut[{cand_i}] gives "
#                                     f"rhs_eff_new={rhs_eff_new}<0",
#                                     iter_num,
#                                     force=True,
#                                 )

#                                 end_time = time()
#                                 LagrangianMST.total_compute_time += end_time - start_time
#                                 return float("inf"), self.best_upper_bound, node_new_cuts

#                             if len(S_free) <= rhs_eff_new:
#                                 _dbg(
#                                     f"Candidate cut[{cand_i}] dropped as redundant: "
#                                     f"|S_free|={len(S_free)} <= rhs_eff={rhs_eff_new}",
#                                     iter_num,
#                                     force=True,
#                                 )
#                                 continue

#                             lhs_free = len(T_loop & S_free)
#                             violation = lhs_free - rhs_eff_new

#                             _dbg(
#                                 f"Candidate cut[{cand_i}]: orig_size={len(S_set)}, "
#                                 f"|fixed|={len(S_fixed)}, "
#                                 f"|excluded|={len(S_excluded)}, "
#                                 f"|free|={len(S_free)}, "
#                                 f"rhs={rhs}, rhs_eff={rhs_eff_new}, "
#                                 f"lhs_on_current_MST={lhs_free}, "
#                                 f"violation={violation}, "
#                                 f"len_sum={_cut_len(S_free):.3f}",
#                                 iter_num,
#                                 force=True,
#                             )

#                             if violation >= min_cut_violation_for_add:
#                                 scored_loop.append(
#                                     (float(violation), set(S_free), int(rhs_eff_new))
#                                 )

#                         scored_loop.sort(
#                             reverse=True,
#                             key=lambda t: (t[0], len(t[1])),
#                         )

#                         remaining_slots = max(0, max_active_cuts - len(self.best_cuts))

#                         if remaining_slots > 0:
#                             scored_loop = scored_loop[
#                                 : min(max_new_cuts_per_node, remaining_slots)
#                             ]
#                         else:
#                             scored_loop = []

#                         _dbg(
#                             f"Candidate cuts after filtering: "
#                             f"kept_for_addition={len(scored_loop)}, "
#                             f"remaining_slots={remaining_slots}",
#                             iter_num,
#                             force=True,
#                         )

#                         existing = {
#                             frozenset(c): (i, int(rhs))
#                             for i, (c, rhs) in enumerate(self.best_cuts)
#                         }

#                         changed_any = False

#                         for violation, S, rhs in scored_loop:
#                             fz = frozenset(S)

#                             if fz in existing:
#                                 old_i, old_rhs = existing[fz]

#                                 if rhs < old_rhs:
#                                     _dbg(
#                                         f"Replace duplicate cut at index {old_i}: "
#                                         f"old_rhs={old_rhs}, new_rhs={rhs}, "
#                                         f"violation={violation}",
#                                         iter_num,
#                                         force=True,
#                                     )

#                                     self.best_cuts[old_i] = (set(S), int(rhs))
#                                     self._rhs_eff[old_i] = int(rhs)
#                                     max_cut_violation[old_i] = max(
#                                         max_cut_violation[old_i],
#                                         violation,
#                                     )
#                                     changed_any = True

#                                 else:
#                                     _dbg(
#                                         f"Skip duplicate cut: existing_rhs={old_rhs}, "
#                                         f"new_rhs={rhs}, violation={violation}",
#                                         iter_num,
#                                         force=True,
#                                     )

#                                 continue

#                             self.best_cuts.append((set(S), int(rhs)))
#                             new_idx = len(self.best_cuts) - 1

#                             # Positive initial μ makes a newly added cut affect the next MST.
#                             MU0 = getattr(self, "mu_init", 0.001)

#                             self.best_cut_multipliers[new_idx] = float(MU0)
#                             self.best_cut_multipliers_for_best_bound[new_idx] = float(MU0)
#                             self._rhs_eff[new_idx] = int(rhs)

#                             max_cut_violation.append(max(0.0, violation))
#                             node_new_cuts.append((set(S), int(rhs)))

#                             existing[fz] = (new_idx, int(rhs))
#                             changed_any = True

#                             _dbg(
#                                 f"ADD cut[{new_idx}]: size={len(S)}, rhs={rhs}, "
#                                 f"initial_mu={MU0}, violation={violation}, "
#                                 f"len_sum={_cut_len(S):.3f}, edges={_cut_repr(S)}",
#                                 iter_num,
#                                 force=True,
#                             )

#                         if changed_any:
#                             _rebuild_cut_structures()

#                             self._mw_cached = None
#                             self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

#                             cuts_present_here = True
#                             use_cuts_in_dual_here = self.use_cover_cuts and bool(self.best_cuts)

#                             _print_cut_table(
#                                 "Cuts after separation/addition",
#                                 iter_num,
#                                 force=True,
#                             )

#                     except Exception as e:
#                         _dbg(
#                             f"ERROR in delayed separation at depth={depth}, "
#                             f"iter={iter_num}: {e}",
#                             iter_num,
#                             force=True,
#                         )

#                     finally:
#                         sep_rounds += 1

#                 # Prepare weights for next iteration
#                 prev_weights = getattr(self, "_last_mw", prev_weights)

#                 # --------------------------------------------------------------
#                 # 5.3) Primal and upper bound
#                 # --------------------------------------------------------------
#                 is_feasible = mst_length <= self.budget

#                 self._record_primal_solution(self.last_mst_edges, is_feasible)

#                 if is_feasible:
#                     try:
#                         real_weight, real_length = self.compute_real_weight_length()

#                         if (
#                             not math.isnan(real_weight)
#                             and not math.isinf(real_weight)
#                             and real_weight < self.best_upper_bound
#                         ):
#                             old_ub = self.best_upper_bound
#                             self.best_upper_bound = real_weight

#                             _dbg(
#                                 f"UB improved: old_UB={old_ub}, "
#                                 f"new_UB={self.best_upper_bound:.6g}, "
#                                 f"real_length={real_length:.6g}",
#                                 iter_num,
#                                 force=True,
#                             )

#                     except Exception as e:
#                         _dbg(
#                             f"ERROR updating primal solution: {e}",
#                             iter_num,
#                             force=True,
#                         )

#                 # Repair fallback: when the natural Lagrangian MST is over
#                 # budget we still try to construct a feasible incumbent, so
#                 # B&B gets a finite UB to prune against. Gated by a flag that
#                 # defaults to OFF, so non-negative-correlation runs (which set
#                 # no overrides) are completely unaffected.
#                 elif getattr(self, "enable_primal_repair", False):
#                     try:
#                         rw, rl, rep_edges = self.primal_repair()
#                         if (
#                             rep_edges is not None
#                             and not math.isnan(rw)
#                             and not math.isinf(rw)
#                             and rw < self.best_upper_bound
#                         ):
#                             old_ub = self.best_upper_bound
#                             self.best_upper_bound = rw
#                             self._record_primal_solution(rep_edges, True)
#                             _dbg(
#                                 f"UB improved via repair: old_UB={old_ub}, "
#                                 f"new_UB={self.best_upper_bound:.6g}, "
#                                 f"length={rl:.6g}",
#                                 iter_num,
#                                 force=True,
#                             )
#                     except Exception as e:
#                         _dbg(f"ERROR in primal_repair: {e}", iter_num, force=True)

#                 if len(self.primal_solutions) > MAX_SOLUTIONS:
#                     self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

#                 # --------------------------------------------------------------
#                 # 5.4) Dual value
#                 # --------------------------------------------------------------
#                 lam_for_dual = max(0.0, min(self.lmbda, 1e4))

#                 if use_cuts_in_dual_here and len(rhs_eff_vec) > 0:
#                     mu_vec = np.fromiter(
#                         (
#                             max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
#                             for i in range(len(rhs_eff_vec))
#                         ),
#                         dtype=float,
#                         count=len(rhs_eff_vec),
#                     )

#                     cover_cut_penalty = float(mu_vec @ rhs_eff_vec)

#                 else:
#                     mu_vec = np.zeros(0, dtype=float)
#                     cover_cut_penalty = 0.0

#                 lagrangian_bound = (
#                     mst_cost
#                     - lam_for_dual * self.budget
#                     - cover_cut_penalty
#                 )

#                 _dbg(
#                     f"Dual: lambda={lam_for_dual:.6g}, "
#                     f"mst_cost={mst_cost:.6g}, "
#                     f"lambdaB={lam_for_dual * self.budget:.6g}, "
#                     f"cover_penalty={cover_cut_penalty:.6g}, "
#                     f"LB_candidate={lagrangian_bound:.6g}, "
#                     f"best_LB_before={self.best_lower_bound:.6g}, "
#                     f"UB={self.best_upper_bound}",
#                     iter_num,
#                 )

#                 if len(mu_vec) > 0:
#                     _dbg(
#                         f"mu_vec={mu_vec.tolist()}, "
#                         f"rhs_eff_vec={rhs_eff_vec.tolist()}",
#                         iter_num,
#                     )

#                 if (
#                     not math.isnan(lagrangian_bound)
#                     and not math.isinf(lagrangian_bound)
#                     and abs(lagrangian_bound) < 1e10
#                 ):
#                     if lagrangian_bound > self.best_lower_bound + 1e-6:
#                         old_lb = self.best_lower_bound

#                         self.best_lower_bound = lagrangian_bound
#                         self.best_lambda = lam_for_dual
#                         self.best_mst_edges = self.last_mst_edges
#                         self.best_cost = mst_cost
#                         self.best_cut_multipliers_for_best_bound = (
#                             self.best_cut_multipliers.copy()
#                         )

#                         _dbg(
#                             f"LB improved: old_LB={old_lb:.6g}, "
#                             f"new_LB={self.best_lower_bound:.6g}, "
#                             f"best_lambda={self.best_lambda:.6g}, "
#                             f"saved_mu={self.best_cut_multipliers_for_best_bound}",
#                             iter_num,
#                             force=True,
#                         )

#                 # --------------------------------------------------------------
#                 # 5.5) Subgradients
#                 # --------------------------------------------------------------
#                 knapsack_subgradient = float(mst_length - self.budget)

#                 all_mu_small = (
#                     not self.best_cut_multipliers
#                     or max(self.best_cut_multipliers.values()) <= dead_mu_threshold
#                 )

#                 if (
#                     cuts_present_here
#                     and mu_dynamic_here
#                     and len(cut_edge_idx_free) > 0
#                     and not (is_feasible and all_mu_small)
#                 ):
#                     mst_mask[:] = False

#                     for e in mst_edges:
#                         j = self.edge_indices.get(e)
#                         if j is not None:
#                             mst_mask[j] = True

#                     cut_g_signed = []
#                     cut_g_pos = []

#                     for i, idxs_free in enumerate(cut_edge_idx_free):
#                         lhs_free = int(mst_mask[idxs_free].sum()) if idxs_free.size else 0
#                         g_i = float(lhs_free) - float(rhs_eff_vec[i])

#                         cut_g_signed.append(g_i)
#                         cut_g_pos.append(g_i if g_i > 0.0 else 0.0)

#                         if i < len(max_cut_violation) and g_i > max_cut_violation[i]:
#                             max_cut_violation[i] = g_i

#                         _dbg(
#                             f"Cut subgradient cut[{i}]: lhs_free={lhs_free}, "
#                             f"rhs_eff={rhs_eff_vec[i]}, "
#                             f"g_signed={g_i}, "
#                             f"g_pos={cut_g_pos[-1]}, "
#                             f"mu_before={self.best_cut_multipliers.get(i, 0.0):.6g}",
#                             iter_num,
#                         )

#                     # Modified:
#                     # Use the signed cut subgradient in the norm and μ update.
#                     # This allows μ to decrease when the cut becomes slack.
#                     cut_subgradients = cut_g_signed

#                 else:
#                     cut_g_signed = []
#                     cut_g_pos = []
#                     cut_subgradients = []

#                     _dbg(
#                         f"Skip cut subgradients: cuts_present={cuts_present_here}, "
#                         f"mu_dynamic={mu_dynamic_here}, "
#                         f"num_cut_arrays={len(cut_edge_idx_free)}, "
#                         f"is_feasible={is_feasible}, "
#                         f"all_mu_small={all_mu_small}",
#                         iter_num,
#                     )

#                 norm_sq = knapsack_subgradient ** 2

#                 for g in cut_subgradients:
#                     norm_sq += float(g) ** 2

#                 # --------------------------------------------------------------
#                 # 5.6) Polyak step size
#                 # --------------------------------------------------------------
#                 if (
#                     polyak_enabled
#                     and self.best_upper_bound < float("inf")
#                     and norm_sq > 0.0
#                 ):
#                     gap = max(0.0, self.best_upper_bound - lagrangian_bound)
#                     alpha = gamma_base * gap / (norm_sq + eps)
#                 else:
#                     gap = None
#                     # Before we have a finite UB, avoid the huge first lambda jump.
#                     alpha = fallback_alpha

#                 _dbg(
#                     f"Step: knapsack_g={knapsack_subgradient:.6g}, "
#                     f"cut_g_signed={cut_g_signed}, "
#                     f"cut_g_pos={cut_g_pos}, "
#                     f"norm_sq={norm_sq:.6g}, "
#                     f"gap={gap}, "
#                     f"alpha={alpha:.6g}",
#                     iter_num,
#                 )

#                 # --------------------------------------------------------------
#                 # 5.7) λ update
#                 # --------------------------------------------------------------
#                 lambda_before = self.lmbda

#                 v_prev = getattr(self, "_v_lambda", 0.0)
#                 v_new = (
#                     self.momentum_beta * v_prev
#                     + (1.0 - self.momentum_beta) * knapsack_subgradient
#                 )

#                 self._v_lambda = v_new

#                 delta_lambda = alpha * v_new
#                 delta_lambda = max(-max_lambda_delta, min(max_lambda_delta, delta_lambda))

#                 self.lmbda = self.lmbda + delta_lambda
#                 self.lmbda = max(0.0, min(self.lmbda, 1e4))

#                 _dbg(
#                     f"Lambda update: before={lambda_before:.6g}, "
#                     f"v_prev={v_prev:.6g}, "
#                     f"v_new={v_new:.6g}, "
#                     f"after={self.lmbda:.6g}",
#                     iter_num,
#                 )

#                 # --------------------------------------------------------------
#                 # 5.8) μ updates
#                 #
#                 # Modified:
#                 # Signed projected update:
#                 #     μ_i <- max(0, μ_i + gamma_mu * alpha * g_i)
#                 #
#                 # If g_i > 0, the cut is violated and μ_i increases.
#                 # If g_i < 0, the cut is slack and μ_i decreases.
#                 # --------------------------------------------------------------
#                 if mu_dynamic_here and len(cut_g_signed) > 0:
#                     for i, g in enumerate(cut_g_signed):
#                         g = float(g)

#                         delta = gamma_mu * alpha * g

#                         # Symmetric cap because delta can now be positive or negative.
#                         if mu_increment_cap is not None:
#                             delta = max(-mu_increment_cap, min(mu_increment_cap, delta))

#                         mu_old = float(self.best_cut_multipliers.get(i, 0.0))
#                         mu_new = mu_old + delta
#                         mu_new = max(0.0, min(mu_new, 1e4))

#                         self.best_cut_multipliers[i] = mu_new

#                         _dbg(
#                             f"Mu signed update cut[{i}]: g={g:.6g}, "
#                             f"delta={delta:.6g}, "
#                             f"mu_old={mu_old:.6g}, "
#                             f"mu_new={mu_new:.6g}",
#                             iter_num,
#                             force=True,
#                         )

#                 self.step_sizes.append(alpha)
#                 self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

#             # ------------------------------------------------------------------
#             # 6) Drop dead cuts
#             # ------------------------------------------------------------------
#             if self.use_cover_cuts and self.best_cuts and mu_dynamic_here:
#                 keep_indices = []

#                 best_mu_map = getattr(
#                     self,
#                     "best_cut_multipliers_for_best_bound",
#                     self.best_cut_multipliers,
#                 )

#                 _dbg(
#                     f"Dead-cut check starts: active_cuts={len(self.best_cuts)}, "
#                     f"max_cut_violation={max_cut_violation}",
#                     force=True,
#                 )

#                 for i, (cut, rhs) in enumerate(self.best_cuts):
#                     mu_i = float(self.best_cut_multipliers.get(i, 0.0))
#                     mu_best_i = float(best_mu_map.get(i, 0.0))

#                     ever_useful = (
#                         i < len(max_cut_violation)
#                         and max_cut_violation[i] > 0.0
#                     ) or abs(mu_best_i) >= dead_mu_threshold

#                     keep = not (
#                         not ever_useful
#                         and abs(mu_i) < dead_mu_threshold
#                         and abs(mu_best_i) < dead_mu_threshold
#                     )

#                     _dbg(
#                         f"Dead-cut decision cut[{i}]: "
#                         f"max_violation={max_cut_violation[i] if i < len(max_cut_violation) else None}, "
#                         f"mu_current={mu_i:.6g}, "
#                         f"mu_best={mu_best_i:.6g}, "
#                         f"ever_useful={ever_useful}, "
#                         f"keep={keep}",
#                         force=True,
#                     )

#                     if keep:
#                         keep_indices.append(i)

#                 if len(keep_indices) < len(self.best_cuts):
#                     _dbg(
#                         f"Dropping dead cuts: keep_indices={keep_indices}, "
#                         f"drop_count={len(self.best_cuts) - len(keep_indices)}",
#                         force=True,
#                     )

#                     new_best_cuts = []
#                     new_mu = {}
#                     new_mu_best = {}
#                     new_rhs_eff = {}

#                     for new_idx, old_idx in enumerate(keep_indices):
#                         new_best_cuts.append(self.best_cuts[old_idx])
#                         new_mu[new_idx] = float(
#                             self.best_cut_multipliers.get(old_idx, 0.0)
#                         )
#                         new_mu_best[new_idx] = float(
#                             self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0)
#                         )
#                         new_rhs_eff[new_idx] = int(
#                             self._rhs_eff.get(old_idx, self.best_cuts[old_idx][1])
#                         )

#                     self.best_cuts = new_best_cuts
#                     self.best_cut_multipliers = new_mu
#                     self.best_cut_multipliers_for_best_bound = new_mu_best
#                     self._rhs_eff = new_rhs_eff

#                     if self.best_cuts:
#                         _rebuild_cut_structures()
#                     else:
#                         self._cut_edge_idx = []
#                         self._cut_edge_idx_all = []
#                         rhs_eff_vec = np.zeros(0, dtype=float)

#             _print_cut_table("Final cuts before returning from node", force=True)

#             # ------------------------------------------------------------------
#             # 7) Restore best λ and μ to pass to children
#             #
#             # Unchanged strategy:
#             # λ and μ are both restored to the values that gave the best lower bound.
#             # ------------------------------------------------------------------
#             if hasattr(self, "best_lambda"):
#                 _dbg(
#                     f"Restore lambda: current={self.lmbda:.6g}, "
#                     f"best_lambda={self.best_lambda:.6g}",
#                     force=True,
#                 )
#                 self.lmbda = self.best_lambda

#             if hasattr(self, "best_cut_multipliers_for_best_bound"):
#                 _dbg(
#                     f"Restore best μ for children: "
#                     f"{self.best_cut_multipliers_for_best_bound}",
#                     force=True,
#                 )
#                 self.best_cut_multipliers = (
#                     self.best_cut_multipliers_for_best_bound.copy()
#                 )

#             _dbg(
#                 f"END NODE: best_LB={self.best_lower_bound:.6g}, "
#                 f"best_UB={self.best_upper_bound}, "
#                 f"return_new_cuts={len(node_new_cuts)}, "
#                 f"final_active_cuts={len(self.best_cuts)}",
#                 force=True,
#             )

#             end_time = time()
#             LagrangianMST.total_compute_time += end_time - start_time
#             return self.best_lower_bound, self.best_upper_bound, node_new_cuts


#     def compute_mst_for_lambda(self, lambda_val):
#         modified_edges = []
#         for i, (u, v) in enumerate(self.edge_list):
#             modified_w = self.edge_weights[i] + lambda_val * self.edge_lengths[i]
#             for cut_idx, (cut, _) in enumerate(self.best_cuts):
#                 if (u, v) in cut:
#                     modified_w += self.best_cut_multipliers.get(cut_idx, 0)
#             modified_edges.append((u, v, modified_w))
#         return self.compute_mst(modified_edges)

#     def _log_fractional_solution(self, method, edge_weights, msts, elapsed_time):
#         if self.verbose:
#             total_weight = sum(self.edge_weights[self.edge_indices[e]] * w for e, w in edge_weights.items())
#             total_length = sum(self.edge_lengths[self.edge_indices[e]] * w for e, w in edge_weights.items())
#             print(f"{method} solution: {len(edge_weights)} edges, "
#                   f"weight={total_weight:.2f}, length={total_length:.2f}, time={elapsed_time:.2f}s")
#             print(f"MSTs used: {len(msts)}")

    
    
#     def compute_dantzig_wolfe_solution(self, node):
#         start_time = time()
        
#         # Need at least 1 MST
#         if len(self.primal_solutions) < 1:
#             if self.verbose:
#                 print("Insufficient primal solutions for Dantzig-Wolfe")
#             return None

#         # More lenient filtering - just check basic validity
#         valid_msts = []
#         for mst_edges, is_feasible in self.primal_solutions:
#             if not mst_edges:
#                 continue
#             mst_edges_normalized = {tuple(sorted((u, v))) for u, v in mst_edges}
            
#             # Basic validity: correct number of edges
#             if len(mst_edges_normalized) == self.num_nodes - 1:
#                 valid_msts.append(mst_edges_normalized)
        
#         if len(valid_msts) < 1:
#             if self.verbose:
#                 print(f"No valid MSTs after filtering")
#             return None

#         # Handle single MST case
#         if len(valid_msts) == 1:
#             edge_weights = {e: 1.0 for e in valid_msts[0]}
#             if self.verbose:
#                 print(f"Dantzig-Wolfe: Single MST, returning as integral solution")
#             return edge_weights

#         if self.verbose:
#             print(f"Using {len(valid_msts)} valid MSTs for Dantzig-Wolfe")

#         # Select diverse MSTs
#         max_msts = min(10, len(valid_msts))
#         selected_msts = []
#         covered_edges = set()
#         remaining_msts = valid_msts.copy()
        
#         while remaining_msts and len(selected_msts) < max_msts:
#             best_mst = None
#             best_score = -1
#             for mst in remaining_msts:
#                 new_edges = mst - covered_edges
#                 score = len(new_edges)
#                 if score > best_score:
#                     best_score = score
#                     best_mst = mst
#             if best_mst:
#                 selected_msts.append(best_mst)
#                 covered_edges.update(best_mst)
#                 remaining_msts.remove(best_mst)
#             else:
#                 break

#         if len(selected_msts) < 2:
#             if self.verbose:
#                 print(f"Only {len(selected_msts)} diverse MSTs selected")
#             return None

#         num_msts = len(selected_msts)
#         edge_indices = self.edge_indices
        
#         # Objective: minimize total weight
#         c = []
#         for mst_edges in selected_msts:
#             weight = sum(self.edge_weights[edge_indices[e]] for e in mst_edges)
#             c.append(weight + 0.1 * (1.0 / num_msts))

#         # Convex combination constraint
#         A_eq = [np.ones(num_msts)]
#         b_eq = [1.0]

#         # Budget as inequality constraint
#         A_ub = []
#         b_ub = []
#         lengths = [sum(self.edge_lengths[edge_indices[e]] for e in mst_edges)
#                 for mst_edges in selected_msts]
#         A_ub.append(lengths)
#         b_ub.append(self.budget)

#         # Cover cuts (limit to avoid infeasibility)
#         if self.best_cuts and len(self.best_cuts) <= 20:
#             for cut, rhs in self.best_cuts:
#                 cut_indices = [edge_indices[e] for e in cut if e in edge_indices]
#                 if cut_indices:
#                     row = np.zeros(num_msts)
#                     for k, mst_edges in enumerate(selected_msts):
#                         cut_count = sum(1 for e in mst_edges if e in cut)
#                         row[k] = cut_count
#                     A_ub.append(row)
#                     b_ub.append(rhs)

#         bounds = [(0, None) for _ in range(num_msts)]

#         try:
#             res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
#             if not res.success:
#                 if self.verbose:
#                     print(f"LP with cuts failed: {res.message}, trying without cuts")
#                 # Retry without cover cuts
#                 res = linprog(c, A_ub=[lengths], b_ub=[self.budget], 
#                             A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
#                 if not res.success:
#                     if self.verbose:
#                         print(f"LP without cuts also failed: {res.message}")
#                     return None
            
#             lambda_k = res.x
#         except Exception as e:
#             if self.verbose:
#                 print(f"LP solver error: {e}")
#             return None

#         # Build fractional edge solution
#         edge_weights = {}
#         for u, v in self.edge_list:
#             e = (u, v)
#             weight = sum(lambda_k[k] for k, mst_edges in enumerate(selected_msts) if e in mst_edges)
#             if weight > 1e-6:
#                 edge_weights[e] = weight

#         if self.verbose:
#             print(f"Dantzig-Wolfe solution: {len(edge_weights)} edges")
#             truly_fractional = sum(1 for w in edge_weights.values() if 0.1 < w < 0.9)
#             print(f"  Truly fractional (0.1-0.9): {truly_fractional}/{len(edge_weights)} edges")

#         return edge_weights if edge_weights else None   
#     def compute_weighted_average_solution(self):
#         """Compute a fractional primal solution as a weighted average of MSTs."""
#         if not self.primal_solutions or not self.step_sizes:
#             if self.verbose:
#                 print("No primal solutions or step sizes available for weighted average")
#             return None

#         # Ensure lengths match (subgradient iterations should align)
#         if len(self.primal_solutions) != len(self.step_sizes):
#             if self.verbose:
#                 print(f"Mismatch: {len(self.primal_solutions)} primal solutions, {len(self.step_sizes)} step sizes")
#             return None

#         total_step_sum = sum(self.step_sizes)
#         if total_step_sum <= 0:
#             if self.verbose:
#                 print("Total step size sum is zero or negative")
#             return None

#         edge_weights = defaultdict(float)
#         for i, (mst_edges, _) in enumerate(self.primal_solutions):
#             lambda_i = self.step_sizes[i]

#             weight = lambda_i / total_step_sum
#             for e in mst_edges:
#                 # print("dd", e)
#                 edge_weights[e] += weight

#         # Ensure weights are in [0, 1] (should be automatic but added for robustness)
#         for e in edge_weights:
#             edge_weights[e] = min(1.0, max(0.0, edge_weights[e]))

#         if self.verbose:
#             total_weight = sum(self.edge_weights[self.edge_indices[e]] * w for e, w in edge_weights.items())
#             total_length = sum(self.edge_lengths[self.edge_indices[e]] * w for e, w in edge_weights.items())
#             print(f"Weighted Average Solution: {len(edge_weights)} edges, "
#                 f"weight={total_weight:.2f}, length={total_length:.2f}")

#         return dict(edge_weights) if edge_weights else None

#     def recover_primal_solution(self, node):
#         start_time = time()

#         for mst_edges, is_feasible in self.primal_solutions:
#             mst_edges_normalized = {tuple(sorted((u, v))) for u, v in mst_edges}
#             if not all(e in mst_edges_normalized for e in node.fixed_edges):
#                 continue
#             if any(e in mst_edges_normalized for e in node.excluded_edges):
#                 continue

#             real_length = sum(self.edge_lengths[self.edge_indices[e]] 
#                               for e in mst_edges_normalized)
#             if real_length > self.budget:
#                 continue

#             valid_cuts = True
#             for cut, rhs in node.active_cuts:
#                 cut_count = sum(1 for e in mst_edges_normalized if e in cut)
#                 if cut_count > rhs:
#                     valid_cuts = False
#                     break
#             if not valid_cuts:
#                 continue

#             uf = UnionFind(self.num_nodes)
#             for u, v in mst_edges_normalized:
#                 uf.union(u, v)
#             if uf.count_components() != 1 or len(set(u for u, _ in mst_edges_normalized) | set(v for _, v in mst_edges_normalized)) < self.num_nodes:
#                 continue

#             real_weight = sum(self.edge_weights[self.edge_indices[e]] 
#                               for e in mst_edges_normalized)
#             end_time = time()
#             if self.verbose:
#                 print(f"Feasible primal solution found from primal_solutions: weight={real_weight:.2f}, length={real_length:.2f}")
#             return list(mst_edges_normalized), real_weight, real_length

#         uf = UnionFind(self.num_nodes)
#         mst_edges = []
#         total_length = 0.0
#         total_weight = 0.0

#         for edge_idx in self.fixed_edge_indices:
#             u, v = self.edge_list[edge_idx]
#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 total_length += self.edge_lengths[edge_idx]
#                 total_weight += self.edge_weights[edge_idx]
#             else:
#                 if self.verbose:
#                     print(f"Fixed edge ({u}, {v}) creates cycle in greedy heuristic")
#                 return None, float('inf'), float('inf')

#         edge_indices = [i for i in range(len(self.edges)) 
#                         if i not in self.fixed_edge_indices and i not in self.excluded_edge_indices]
#         sorted_edges = sorted(edge_indices, key=lambda i: self.edge_weights[i])

#         for edge_idx in sorted_edges:
#             u, v = self.edge_list[edge_idx]
#             new_length = total_length + self.edge_lengths[edge_idx]
#             if new_length > self.budget:
#                 continue

#             temp_edges = mst_edges + [(u, v)]
#             valid_cuts = True
#             for cut, rhs in node.active_cuts:
#                 cut_count = sum(1 for e in temp_edges if e in cut)
#                 if cut_count > rhs:
#                     valid_cuts = False
#                     break
#             if not valid_cuts:
#                 continue

#             if uf.union(u, v):
#                 mst_edges.append((u, v))
#                 total_length = new_length
#                 total_weight += self.edge_weights[edge_idx]

#         if uf.count_components() != 1 or len(set(u for u, _ in mst_edges) | set(v for _, v in mst_edges)) < self.num_nodes:
#             if self.verbose:
#                 print("Greedy heuristic failed to produce a valid spanning tree")
#             return None, float('inf'), float('inf')

#         end_time = time()
#         if self.verbose:
#             print(f"Feasible primal solution found via greedy heuristic: weight={total_weight:.2f}, length={total_length:.2f}")
#         return mst_edges, total_weight, total_length

#     def compute_real_weight_length(self):
#         real_weight = sum(self.edge_weights[self.edge_indices[e]] 
#                           for e in self.last_mst_edges)
#         real_length = sum(self.edge_lengths[self.edge_indices[e]] 
#                           for e in self.last_mst_edges)
#         return real_weight, real_length

#     def primal_repair(self):
#         """
#         Produce a budget-FEASIBLE spanning tree (an incumbent) regardless of
#         whether the current Lagrangian MST is over budget.

#         Strategy:
#           1. Build the minimum-LENGTH spanning tree over the allowed edges
#              (respecting fixed/excluded via custom_kruskal on edge_lengths).
#              This is the shortest possible tree for this node; if its length
#              still exceeds the budget, the node is genuinely infeasible.
#           2. If feasible, try to lower its real weight with budget-preserving
#              swaps: for each non-tree edge, if adding it and dropping the
#              heaviest-weight edge on the induced cycle keeps length <= budget
#              and reduces weight, do it. Cheap local improvement, optional.

#         Returns (weight, length, edges) for a feasible tree, or
#         (inf, inf, None) if no feasible tree exists at this node.
#         """
#         # Step 1: minimum-length tree using the existing Kruskal machinery.
#         _, min_len, len_tree = self.custom_kruskal(self.edge_lengths)
#         if not len_tree or min_len == float("inf") or min_len > self.budget:
#             return float("inf"), float("inf"), None

#         tree = [tuple(sorted(e)) for e in len_tree]
#         tree_set = set(tree)
#         ei = self.edge_indices
#         W = self.edge_weights
#         L = self.edge_lengths

#         def tree_weight(edges):
#             return float(sum(W[ei[e]] for e in edges))

#         def tree_length(edges):
#             return float(sum(L[ei[e]] for e in edges))

#         cur_len = tree_length(tree)
#         cur_w = tree_weight(tree)

#         # Step 2: budget-preserving weight-reducing swaps (bounded effort).
#         # The min-length tree is ALREADY a valid feasible incumbent, so the
#         # swaps are pure optional improvement. They are O(candidates * n) with a
#         # Python BFS per candidate, which is far too slow on large/dense graphs,
#         # so we skip improvement entirely beyond a size threshold. B&B still
#         # gets a finite UB from the min-length tree itself.
#         repair_improve_cap = getattr(self, "repair_improve_max_nodes", 120)
#         if self.num_nodes > repair_improve_cap:
#             return cur_w, cur_len, tree

#         # Non-tree candidate edges, cheapest weight first.
#         non_tree = [
#             e for e in self.edge_list
#             if e not in tree_set
#             and ei[e] not in self.excluded_edge_indices
#         ]
#         non_tree.sort(key=lambda e: W[ei[e]])

#         max_swaps = min(len(non_tree), 2 * self.num_nodes)
#         swaps_done = 0

#         for add_e in non_tree:
#             if swaps_done >= max_swaps:
#                 break
#             # Find the cycle created by adding add_e: path between its endpoints
#             # in the current tree.
#             adj = {}
#             for (u, v) in tree:
#                 adj.setdefault(u, []).append((v, (u, v)))
#                 adj.setdefault(v, []).append((u, (u, v)))
#             su, sv = add_e
#             # BFS for the path su -> sv
#             prev = {su: None}
#             stack = [su]
#             found = False
#             while stack:
#                 x = stack.pop()
#                 if x == sv:
#                     found = True
#                     break
#                 for (y, edge) in adj.get(x, []):
#                     if y not in prev:
#                         prev[y] = (x, edge)
#                         stack.append(y)
#             if not found:
#                 continue
#             # Reconstruct cycle edges.
#             cycle_edges = []
#             node = sv
#             while prev[node] is not None:
#                 px, edge = prev[node]
#                 cycle_edges.append(tuple(sorted(edge)))
#                 node = px
#             if not cycle_edges:
#                 continue
#             # Candidate to drop: the heaviest-weight tree edge on the cycle
#             # whose removal keeps us within budget after adding add_e.
#             add_w = float(W[ei[add_e]])
#             add_l = float(L[ei[add_e]])
#             best_drop = None
#             best_gain = 0.0
#             for drop_e in cycle_edges:
#                 new_len = cur_len - float(L[ei[drop_e]]) + add_l
#                 if new_len > self.budget:
#                     continue
#                 gain = float(W[ei[drop_e]]) - add_w  # weight reduction
#                 if gain > best_gain:
#                     best_gain = gain
#                     best_drop = drop_e
#             if best_drop is not None and best_gain > 0:
#                 tree_set.discard(best_drop)
#                 tree_set.add(add_e)
#                 tree = list(tree_set)
#                 cur_len = cur_len - float(L[ei[best_drop]]) + add_l
#                 cur_w = cur_w - float(W[ei[best_drop]]) + add_w
#                 swaps_done += 1

#         return cur_w, cur_len, list(tree_set)
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
                 initial_lambda=0.05, step_size=0.001, max_iter=10, 
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

        self.best_lambda = float(self.lmbda)
        self.best_mst_edges = []
        self.best_cost = 0

        self.best_cuts = []
        self.best_cut_multipliers = {}
        self.best_cut_multipliers_for_best_bound = {}

        self.multipliers = []

        # Important when reusing solver objects in strong branching
        self._v_lambda = 0.0

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
        self._v_lambda = 0.0
        # self.last_modified_weights = None
        # self.last_mst_edges = None
        self._invalidate_weight_cache()
        if hasattr(self, 'mst_cache'):
            self.mst_cache = LRUCache(capacity=5)
    # def generate_cover_cuts(self, mst_edges):  
    #     """
    #     Stronger cover cuts (tightened):
    #     - Residualization: A, B' (clamped), fixed/excluded respected
    #     - Seed residual-minimal cover from T^λ ∩ A
    #     - Certificate shrinking using optimistic U(S) with component-based k, and exact Kruskal fallback
    #     - Inclusion-minimal S* shrinking under the certificate (THIS was missing)
    #     - Micro-seed from top-L heaviest admissible edges
    #     - Stronger safe lifting for residual-minimal covers
    #     - Strict effective-RHS pruning + current-violation checks
    #     - Dedup with dominance & subset-dominance
    #     """
    #     if not mst_edges:
    #         return []

    #     EPS = 1e-12
    #     L_MICRO = 3
    #     MAX_RETURN = 10

    #     # --- normalize edges ---
    #     def norm(e):
    #         u, v = e
    #         return (u, v) if u <= v else (v, u)

    #     mst_norm = [norm(e) for e in mst_edges]
    #     mst_set = set(mst_norm)

    #     # --- accessors / data ---
    #     edge_attr = self.edge_attributes  # edge -> (w, ℓ)
    #     def get_len(e): return edge_attr[e][1]

    #     fixed = set(getattr(self, "fixed_edges", set()))
    #     excluded = set(getattr(self, "excluded_edges", set()))
    #     budget = self.budget

    #     # Residual budget
    #     L_fix = sum(get_len(e) for e in fixed if e in edge_attr)
    #     Bp = budget - L_fix

    #     # If fixes already exceed the budget, cuts may still be useful, but be careful with rhs_eff.
    #     # We will still attempt separation.

    #     # Admissible edges A
    #     A = {e for e in getattr(self, "edge_list", []) if e not in fixed and e not in excluded and e in edge_attr}
    #     if not A:
    #         return []

    #     # T^λ ∩ A (use provided mst_edges)
    #     TcapA = [e for e in mst_norm if e in A]

    #     # If residual MST is feasible, nothing to cut
    #     mst_len = sum(get_len(e) for e in TcapA)
    #     if mst_len <= Bp + EPS:
    #         return []

    #     cuts = []

    #     # Pre-sort A by length for U(S) and Kruskal completion
    #     A_sorted = sorted(A, key=lambda e: get_len(e))

    #     # --- DSU helpers (for component count & exact completion) ---
    #     def get_nodes():
    #         # best-effort: use graph nodes if present, otherwise infer from edge keys
    #         if hasattr(self, "graph") and hasattr(self.graph, "nodes"):
    #             try:
    #                 return list(self.graph.nodes)
    #             except Exception:
    #                 pass
    #         nodes = set()
    #         for (u, v) in edge_attr.keys():
    #             nodes.add(u); nodes.add(v)
    #         for (u, v) in fixed:
    #             nodes.add(u); nodes.add(v)
    #         return list(nodes)

    #     NODES = get_nodes()

    #     def component_k_needed(contracted_edges):
    #         """Number of edges needed to connect after contracting 'contracted_edges': k = #components - 1."""
    #         parent = {n: n for n in NODES}
    #         rank = {n: 0 for n in NODES}

    #         def find(x):
    #             while parent[x] != x:
    #                 parent[x] = parent[parent[x]]
    #                 x = parent[x]
    #             return x

    #         def union(x, y):
    #             rx, ry = find(x), find(y)
    #             if rx == ry:
    #                 return
    #             if rank[rx] < rank[ry]:
    #                 parent[rx] = ry
    #             elif rank[rx] > rank[ry]:
    #                 parent[ry] = rx
    #             else:
    #                 parent[ry] = rx
    #                 rank[rx] += 1

    #         for (u, v) in contracted_edges:
    #             if u in parent and v in parent:
    #                 union(u, v)

    #         reps = {find(n) for n in NODES}
    #         comps = len(reps)
    #         return max(0, comps - 1)

    #     def U_of(Sprime):
    #         """
    #         Optimistic completion:
    #         sum of k cheapest edges in A \\ S', where k = (#components after contracting fixed ∪ S') - 1.
    #         This is stronger/more accurate than r' - |S'|.
    #         """
    #         Sprime_set = Sprime if isinstance(Sprime, set) else set(Sprime)
    #         contracted = set(fixed) | Sprime_set
    #         k = component_k_needed(contracted)
    #         if k <= 0:
    #             return 0.0

    #         total = 0.0
    #         taken = 0
    #         for e in A_sorted:
    #             if e in Sprime_set:
    #                 continue
    #             total += get_len(e)
    #             taken += 1
    #             if taken == k:
    #                 break
    #         return total if taken == k else float("inf")

    #     def completion_mst_cost(Ssub):
    #         """
    #         Exact completion via Kruskal after contracting fixed ∪ Ssub.
    #         Returns minimum additional length needed to connect components using edges in A \\ Ssub.
    #         """
    #         parent = {n: n for n in NODES}
    #         rank = {n: 0 for n in NODES}

    #         def find(x):
    #             while parent[x] != x:
    #                 parent[x] = parent[parent[x]]
    #                 x = parent[x]
    #             return x

    #         def union(x, y):
    #             rx, ry = find(x), find(y)
    #             if rx == ry:
    #                 return False
    #             if rank[rx] < rank[ry]:
    #                 parent[rx] = ry
    #             elif rank[rx] > rank[ry]:
    #                 parent[ry] = rx
    #             else:
    #                 parent[ry] = rx
    #                 rank[rx] += 1
    #             return True

    #         contracted = set(fixed) | set(Ssub)
    #         for (u, v) in contracted:
    #             if u in parent and v in parent:
    #                 union(u, v)

    #         reps = {find(n) for n in NODES}
    #         k_needed = max(0, len(reps) - 1)
    #         if k_needed <= 0:
    #             return 0.0

    #         Sset = set(Ssub)
    #         total = 0.0
    #         taken = 0
    #         for e in A_sorted:
    #             if e in Sset:
    #                 continue
    #             u, v = e
    #             if u not in parent or v not in parent:
    #                 continue
    #             if union(u, v):
    #                 total += get_len(e)
    #                 taken += 1
    #                 if taken == k_needed:
    #                     break
    #         return total if taken == k_needed else float("inf")

    #     def build_residual_minimal_cover(desc_edges):
    #         """Minimal cover on B': add in desc ℓ, then prune shortest while violation remains."""
    #         S, sL = [], 0.0
    #         for e in desc_edges:
    #             if e not in edge_attr:
    #                 continue
    #             S.append(e)
    #             sL += get_len(e)
    #             if sL > Bp + EPS:
    #                 # prune shortest while still violating
    #                 S.sort(key=lambda x: get_len(x))  # increasing
    #                 k = 0
    #                 while k < len(S) and (sL - get_len(S[k]) > Bp + EPS):
    #                     sL -= get_len(S[k])
    #                     k += 1
    #                 if k > 0:
    #                     S = S[k:]
    #                 return S, sL
    #         return None, None

    #     def rhs_eff(cset):
    #         """Effective RHS after accounting fixed-in edges."""
    #         return len(cset) - 1 - sum(1 for e in cset if e in fixed)

    #     def is_violated_now(cset):
    #         """Check current MST violation: lhs > rhs_eff."""
    #         lhs = sum(1 for e in cset if e in mst_set)
    #         return lhs > rhs_eff(cset)

    #     def cert_holds(Slist):
    #         """
    #         Certificate: sumℓ(S) + U(S) > B' (optimistic), else fallback to exact completion.
    #         """
    #         if not Slist or len(Slist) <= 1:
    #             return False
    #         if rhs_eff(Slist) <= 0:
    #             return False
    #         sumS = sum(get_len(e) for e in Slist)
    #         U = U_of(Slist)
    #         if U != float("inf") and (sumS + U) > (Bp + EPS):
    #             return True
    #         exact = completion_mst_cost(Slist)
    #         return exact != float("inf") and (sumS + exact) > (Bp + EPS)

    #     def inclusion_minimal_shrink(Sstart):
    #         """
    #         Make S inclusion-minimal under cert_holds by removing one edge at a time.
    #         We try removals from longest to shortest for a small S.
    #         """
    #         Sstar = sorted(Sstart, key=lambda e: get_len(e), reverse=True)
    #         changed = True
    #         while changed and len(Sstar) > 1:
    #             changed = False
    #             for j in range(len(Sstar)):  # longest -> shortest
    #                 trial = Sstar[:j] + Sstar[j+1:]
    #                 if len(trial) <= 1:
    #                     continue
    #                 if cert_holds(trial):
    #                     Sstar = sorted(trial, key=lambda e: get_len(e), reverse=True)
    #                     changed = True
    #                     break
    #         return Sstar

    #     def try_shrink_and_add(seed_S, seed_sumL):
    #         """
    #         Full LaTeX Step (2):
    #         - remove longest edges until sumℓ <= B' => first S'
    #         - require cert_holds(S')
    #         - shrink to inclusion-minimal S* while certificate holds
    #         - add the cut if it separates current MST
    #         """
    #         if not seed_S or len(seed_S) <= 1:
    #             return

    #         S_work = sorted(seed_S, key=lambda e: get_len(e), reverse=True)
    #         sumL = float(seed_sumL)

    #         # First S' with sumℓ <= B'
    #         idx = 0
    #         while idx < len(S_work) and sumL > Bp + EPS:
    #             sumL -= get_len(S_work[idx])
    #             idx += 1
    #         Sprime = S_work[idx:]
    #         if not Sprime or len(Sprime) <= 1:
    #             return

    #         if not cert_holds(Sprime):
    #             return

    #         Sstar = inclusion_minimal_shrink(Sprime)
    #         if len(Sstar) <= 1:
    #             return

    #         if is_violated_now(Sstar):
    #             cuts.append((set(Sstar), len(Sstar) - 1))

    #     def lift_minimal_cover(S_min, rhs_base):
    #         """
    #         Stronger safe lifting for residual-minimal cover S:
    #         Lift any f with ℓ(f) > B' - sumℓ(S) + Lmax.
    #         (This is typically much stronger than ℓ(f) >= Lmax.)
    #         """
    #         S_base = set(S_min)
    #         if not S_base:
    #             return None
    #         sumS = sum(get_len(e) for e in S_base)
    #         Lmax = max(get_len(e) for e in S_base)
    #         threshold = (Bp - sumS + Lmax)  # lift if len(f) > threshold

    #         lift_add = {f for f in A if f not in S_base and get_len(f) > threshold + EPS}
    #         if not lift_add:
    #             return None

    #         S_lift = S_base | lift_add
    #         # RHS remains rhs_base (|S|-1 of original minimal cover)
    #         if rhs_eff(S_lift) > 0 and is_violated_now(S_lift):
    #             return (S_lift, rhs_base)
    #         return None

    #     # --- (1) primary seed from T^λ ∩ A ---
    #     T_desc = sorted(TcapA, key=lambda e: get_len(e), reverse=True)
    #     S_seed, sumL_seed = build_residual_minimal_cover(T_desc)
    #     if not S_seed:
    #         return []

    #     S_seed = list(S_seed)
    #     if rhs_eff(S_seed) > 0 and is_violated_now(S_seed):
    #         cuts.append((set(S_seed), len(S_seed) - 1))

    #     # Step (2): certificate shrink to inclusion-minimal S*
    #     try_shrink_and_add(S_seed, sumL_seed)

    #     # --- stronger lifting on the residual-minimal seed cover ---
    #     lifted = lift_minimal_cover(S_seed, rhs_base=(len(S_seed) - 1))
    #     if lifted is not None:
    #         cuts.append(lifted)

    #     # --- (1b) micro-seed: top-L heaviest admissible edges ---
    #     if L_MICRO > 0 and len(A) > 0:
    #         heavyA = sorted(A, key=lambda e: get_len(e), reverse=True)[:L_MICRO]
    #         S2, sumL2 = build_residual_minimal_cover(heavyA)
    #         if S2:
    #             S2set = set(S2)
    #             if rhs_eff(S2set) > 0 and S2set != set(S_seed) and is_violated_now(S2set):
    #                 cuts.append((S2set, len(S2) - 1))

    #             try_shrink_and_add(S2, sumL2)

    #             lifted2 = lift_minimal_cover(S2, rhs_base=(len(S2) - 1))
    #             if lifted2 is not None:
    #                 cuts.append(lifted2)

    #     # --- dedup & dominance-aware selection ---
    #     uniq = {}
    #     for cset, rhs in cuts:
    #         key = tuple(sorted(cset))
    #         best = uniq.get(key)
    #         if best is None or rhs < best[1] or (rhs == best[1] and len(cset) < len(best[0])):
    #             uniq[key] = (cset, rhs)

    #     final = list(uniq.values())
    #     final.sort(key=lambda t: (t[1], len(t[0])))

    #     kept = []
    #     for cset, rhs in final:
    #         if rhs_eff(cset) <= 0:
    #             continue
    #         dominated = any(dset <= cset and drhs <= rhs for dset, drhs in kept)
    #         if not dominated:
    #             kept.append((cset, rhs))
    #     return kept[:MAX_RETURN]
    def generate_cover_cuts(self, mst_edges):
        """
        Cover-cut generation with exact tree-completion certificate.

        Main logic:
        - Work at the current B&B node with fixed edges F+ and excluded edges F-.
        - Define residual budget B' = B - length(F+).
        - Define admissible edges A = E \ (F+ union F-).
        - Generate a residual-minimal seed cover from the current violating
        Lagrangian MST T^lambda.
        - Refine the seed using an exact minimum-length MST completion certificate:
            contract F+ union S'
            complete using admissible edges A \ S'
            compute the minimum additional length by Kruskal using edge lengths
        - Add a cut only if it is violated by the current Lagrangian MST.
        - Optionally add lifted cuts using the residual-aware lifting rule.
        """
        if not mst_edges:
            return []

        EPS = 1e-12
        L_MICRO = 3
        MAX_RETURN = 10

        # ------------------------------------------------------------
        # Normalize edges
        # ------------------------------------------------------------
        def norm(e):
            u, v = e
            return (u, v) if u <= v else (v, u)

        mst_norm = [norm(e) for e in mst_edges]
        mst_set = set(mst_norm)

        # ------------------------------------------------------------
        # Accessors and node data
        # ------------------------------------------------------------
        edge_attr = self.edge_attributes  # edge -> (weight, length)

        def get_len(e):
            return edge_attr[e][1]

        fixed = set(getattr(self, "fixed_edges", set()))
        excluded = set(getattr(self, "excluded_edges", set()))
        budget = self.budget

        # Residual budget B' = B - length(F+)
        L_fix = sum(get_len(e) for e in fixed if e in edge_attr)
        Bp = budget - L_fix

        # Admissible residual edges A = E \ (F+ union F-)
        A = {
            e for e in getattr(self, "edge_list", [])
            if e not in fixed and e not in excluded and e in edge_attr
        }

        if not A:
            return []

        # Current Lagrangian tree restricted to admissible residual edges
        TcapA = [e for e in mst_norm if e in A]

        # If the current residual tree already respects the residual budget,
        # there is no budget-violating tree to separate.
        mst_len = sum(get_len(e) for e in TcapA)
        if mst_len <= Bp + EPS:
            return []

        cuts = []

        # Admissible edges sorted by LENGTH for exact completion.
        # This is the length-MST completion part.
        A_sorted = sorted(A, key=lambda e: get_len(e))

        # ------------------------------------------------------------
        # Node list for local DSU
        # ------------------------------------------------------------
        def get_nodes():
            if hasattr(self, "graph") and hasattr(self.graph, "nodes"):
                try:
                    return list(self.graph.nodes)
                except Exception:
                    pass

            nodes = set()
            for (u, v) in edge_attr.keys():
                nodes.add(u)
                nodes.add(v)
            for (u, v) in fixed:
                nodes.add(u)
                nodes.add(v)
            return list(nodes)

        NODES = get_nodes()

        # ------------------------------------------------------------
        # Exact minimum-length completion
        # ------------------------------------------------------------
        def completion_mst_cost(Ssub):
            """
            Minimum additional LENGTH needed to complete F+ union Ssub
            to a spanning tree.

            Steps:
            1. Contract all fixed edges F+.
            2. Contract all edges in Ssub.
            3. Complete the remaining components using Kruskal on A \ Ssub,
            sorted by edge length.
            4. Return +inf if no spanning tree completion exists.

            Important:
            - If F+ union Ssub already creates a cycle, then no spanning tree can
            contain all of those forced edges, so completion is impossible.
            - The returned value is only the additional length beyond Ssub.
            The fixed-edge length has already been removed through B'.
            """
            Sset = set(Ssub)

            parent = {n: n for n in NODES}
            rank = {n: 0 for n in NODES}
            components = len(NODES)

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                nonlocal components

                rx, ry = find(x), find(y)
                if rx == ry:
                    return False

                if rank[rx] < rank[ry]:
                    parent[rx] = ry
                elif rank[rx] > rank[ry]:
                    parent[ry] = rx
                else:
                    parent[ry] = rx
                    rank[rx] += 1

                components -= 1
                return True

            # Force/contract fixed edges.
            for e in fixed:
                if e not in edge_attr:
                    return float("inf")

                u, v = e
                if u not in parent or v not in parent:
                    return float("inf")

                # Fixed edges creating a cycle means no tree can contain all of them.
                if not union(u, v):
                    return float("inf")

            # Force/contract Ssub.
            for e in Sset:
                if e not in edge_attr:
                    return float("inf")

                u, v = e
                if u not in parent or v not in parent:
                    return float("inf")

                # If F+ union Ssub creates a cycle, no spanning tree can contain Ssub.
                if not union(u, v):
                    return float("inf")

            # Already connected after contracting fixed union Ssub.
            if components == 1:
                return 0.0

            total_completion_length = 0.0

            # Complete with cheapest admissible edges by LENGTH.
            # A already excludes fixed and excluded edges.
            # We also exclude Ssub because those edges are already forced.
            for e in A_sorted:
                if e in Sset:
                    continue

                u, v = e
                if u not in parent or v not in parent:
                    continue

                if union(u, v):
                    total_completion_length += get_len(e)

                    if components == 1:
                        return total_completion_length

            # Could not connect all components.
            return float("inf")

        # ------------------------------------------------------------
        # Build residual-minimal seed cover
        # ------------------------------------------------------------
        def build_residual_minimal_cover(desc_edges):
            """
            Build a residual-minimal cover with respect to B'.

            We add edges in nonincreasing length order until the residual budget
            is exceeded, then prune shortest edges while the violation remains.
            """
            S = []
            sL = 0.0

            for e in desc_edges:
                if e not in edge_attr:
                    continue

                S.append(e)
                sL += get_len(e)

                if sL > Bp + EPS:
                    # Prune shortest edges while still violating B'.
                    S.sort(key=lambda x: get_len(x))  # increasing length

                    k = 0
                    while k < len(S) and (sL - get_len(S[k]) > Bp + EPS):
                        sL -= get_len(S[k])
                        k += 1

                    if k > 0:
                        S = S[k:]

                    return S, sL

            return None, None

        # ------------------------------------------------------------
        # RHS and violation helpers
        # ------------------------------------------------------------
        def rhs_eff(cset):
            """
            Effective RHS after fixed-in edges.

            For generated cuts at the current node, cset is usually a subset of A,
            so this is normally |S|-1. This form is kept for safety.
            """
            return len(cset) - 1 - sum(1 for e in cset if e in fixed)

        def is_violated_now(cset):
            """
            Check whether the current Lagrangian MST violates the cut.
            """
            lhs = sum(1 for e in cset if e in mst_set)
            return lhs > rhs_eff(cset)

        # ------------------------------------------------------------
        # Exact completion certificate
        # ------------------------------------------------------------
        def cert_holds(Slist):
            """
            Exact tree-completion certificate.

            The cut sum_{e in Slist} x_e <= |Slist|-1 is valid at this node if
            every spanning-tree completion containing F+ union Slist violates
            the residual budget.

            We test this by computing the minimum additional LENGTH needed to
            complete F+ union Slist to a spanning tree.
            """
            if not Slist or len(Slist) <= 1:
                return False

            if rhs_eff(Slist) <= 0:
                return False

            sumS = sum(get_len(e) for e in Slist)
            completion = completion_mst_cost(Slist)

            # If completion is impossible, then no feasible spanning tree can
            # contain all edges in Slist, so the cut is valid.
            if completion == float("inf"):
                return True

            return (sumS + completion) > (Bp + EPS)

        # ------------------------------------------------------------
        # Inclusion-minimal shrinking under exact certificate
        # ------------------------------------------------------------
        def inclusion_minimal_shrink(Sstart):
            """
            Make S inclusion-minimal under cert_holds by removing one edge at a time.

            This version scans removals from shortest to longest, matching the
            current LaTeX description.
            """
            Sstar = sorted(Sstart, key=lambda e: get_len(e))  # shortest -> longest

            changed = True
            while changed and len(Sstar) > 1:
                changed = False

                for j in range(len(Sstar)):
                    trial = Sstar[:j] + Sstar[j + 1:]

                    if len(trial) <= 1:
                        continue

                    if cert_holds(trial):
                        Sstar = sorted(trial, key=lambda e: get_len(e))
                        changed = True
                        break

            return Sstar

        # ------------------------------------------------------------
        # Try completion-aware shrinking and add resulting cut
        # ------------------------------------------------------------
        def try_shrink_and_add(seed_S, seed_sumL):
            """
            Starting from a residual-minimal seed cover S, find a smaller set S'
            that is still invalid under exact MST completion.

            Procedure:
            - Remove long edges until the set is no longer a simple residual cover.
            - Test the exact completion certificate.
            - Shrink to an inclusion-minimal certified set.
            - Add the cut only if it separates the current Lagrangian MST.
            """
            if not seed_S or len(seed_S) <= 1:
                return

            S_work = sorted(seed_S, key=lambda e: get_len(e), reverse=True)
            sumL = float(seed_sumL)

            # First candidate S': remove longest edges until sum length <= B'.
            idx = 0
            while idx < len(S_work) and sumL > Bp + EPS:
                sumL -= get_len(S_work[idx])
                idx += 1

            Sprime = S_work[idx:]

            if not Sprime or len(Sprime) <= 1:
                return

            if not cert_holds(Sprime):
                return

            Sstar = inclusion_minimal_shrink(Sprime)

            if len(Sstar) <= 1:
                return

            if is_violated_now(Sstar):
                cuts.append((set(Sstar), len(Sstar) - 1))

        # ------------------------------------------------------------
        # Safe residual-aware lifting
        # ------------------------------------------------------------
        def lift_minimal_cover(S_min, rhs_base):
            """
            Safe unit lifting for a residual-minimal cover S.

            If length(f) > B' - length(S) + Lmax,
            then f can be lifted with coefficient 1 while keeping the same RHS.
            """
            S_base = set(S_min)

            if not S_base:
                return None

            sumS = sum(get_len(e) for e in S_base)
            Lmax = max(get_len(e) for e in S_base)
            threshold = Bp - sumS + Lmax

            lift_add = {
                f for f in A
                if f not in S_base and get_len(f) > threshold + EPS
            }

            if not lift_add:
                return None

            S_lift = S_base | lift_add

            # RHS remains rhs_base = |S_min|-1.
            if rhs_eff(S_lift) > 0 and is_violated_now(S_lift):
                return (S_lift, rhs_base)

            return None

        # ============================================================
        # Main separation logic
        # ============================================================

        # Primary seed from T^lambda intersect A.
        T_desc = sorted(TcapA, key=lambda e: get_len(e), reverse=True)
        S_seed, sumL_seed = build_residual_minimal_cover(T_desc)

        if not S_seed:
            return []

        S_seed = list(S_seed)

        # Add simple residual cover cut.
        if rhs_eff(S_seed) > 0 and is_violated_now(S_seed):
            cuts.append((set(S_seed), len(S_seed) - 1))

        # Add exact completion-aware refined cut.
        try_shrink_and_add(S_seed, sumL_seed)

        # Add lifted residual-minimal seed cover.
        lifted = lift_minimal_cover(S_seed, rhs_base=(len(S_seed) - 1))
        if lifted is not None:
            cuts.append(lifted)

        # Optional micro-seed from globally heaviest admissible edges.
        # Keep this only if you also mention it in the paper/implementation section.
        if L_MICRO > 0 and len(A) > 0:
            heavyA = sorted(A, key=lambda e: get_len(e), reverse=True)[:L_MICRO]
            S2, sumL2 = build_residual_minimal_cover(heavyA)

            if S2:
                S2set = set(S2)

                if (
                    rhs_eff(S2set) > 0
                    and S2set != set(S_seed)
                    and is_violated_now(S2set)
                ):
                    cuts.append((S2set, len(S2) - 1))

                try_shrink_and_add(S2, sumL2)

                lifted2 = lift_minimal_cover(S2, rhs_base=(len(S2) - 1))
                if lifted2 is not None:
                    cuts.append(lifted2)

        # ------------------------------------------------------------
        # Deduplication and dominance filtering
        # ------------------------------------------------------------
        uniq = {}

        for cset, rhs in cuts:
            key = tuple(sorted(cset))
            best = uniq.get(key)

            if best is None or rhs < best[1] or (
                rhs == best[1] and len(cset) < len(best[0])
            ):
                uniq[key] = (cset, rhs)

        final = list(uniq.values())
        final.sort(key=lambda t: (t[1], len(t[0])))

        kept = []

        for cset, rhs in final:
            if rhs_eff(cset) <= 0:
                continue

            dominated = any(
                dset <= cset and drhs <= rhs
                for dset, drhs in kept
            )

            if not dominated:
                kept.append((cset, rhs))

        return kept[:MAX_RETURN]

    
    
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
        cut_idxs_free = getattr(self, "_cut_edge_idx", None)  # FREE indices only

        mu_len = len(self.best_cuts)

        mu = np.array([max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4)) for i in range(mu_len)], dtype=float)

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

        if cut_idxs_free is not None:
            for i, idxs in enumerate(cut_idxs_free):
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
        mst_edge_indices = []   # <--- NEW: track indices
        mst_cost = 0.0

        # Add fixed edges first
        for i in self.fixed_edge_indices:
            u, v = self.edge_list[i]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_edge_indices.append(i)          # <--- NEW
                mst_cost += modified_weights[i]
            else:
                return float('inf'), float('inf'), []

        # Remaining candidate edges (canonical size!)
        m = len(self.edge_list)
        candidates = [i for i in range(m)
                    if i not in self.fixed_edge_indices and i not in self.excluded_edge_indices]

        candidates.sort(key=lambda i: modified_weights[i])

        for i in candidates:
            u, v = self.edge_list[i]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_edge_indices.append(i)          # <--- NEW
                mst_cost += modified_weights[i]
                if len(mst_edges) == self.num_nodes - 1:
                    break

        # Connectivity / size check
        if len(mst_edges) != self.num_nodes - 1 or uf.count_components() > 1:
            return float('inf'), float('inf'), []

        # Length computed by indices (consistent with everything)
        mst_length = float(np.sum(self.edge_lengths[mst_edge_indices]))

        return mst_cost, mst_length, mst_edges

    
  
    def incremental_kruskal(self, prev_weights, prev_mst_edges, current_weights):
        # Fast path (opt-in via flag, set only for negative-correlation runs):
        # when lambda moves globally nearly every edge "changes", so the
        # incremental candidate set degenerates to the full edge list and the
        # Python `sorted` below dominates runtime. A single vectorized
        # numpy.argsort over the whole array is far faster and yields an
        # identical MST. Non-negative runs never set this flag, so they take
        # the original code path below unchanged.
        if getattr(self, "use_fast_kruskal", False):
            return self._argsort_kruskal(current_weights)

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

    def _argsort_kruskal(self, weights):
        """Full Kruskal using numpy.argsort for the edge ordering. Honors
        fixed/excluded edges identically to custom_kruskal. Used only on the
        opt-in fast path; produces the same MST as the Python-sort version."""
        uf = UnionFind(self.num_nodes)
        mst_edges = []
        mst_cost = 0.0
        mst_length = 0.0

        # Fixed edges first.
        for i in self.fixed_edge_indices:
            u, v = self.edge_list[i]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += float(weights[i])
                mst_length += float(self.edge_lengths[i])
            else:
                return float('inf'), float('inf'), []

        # Vectorized global ordering by weight.
        order = np.argsort(weights, kind="stable")
        fixed = self.fixed_edge_indices
        excluded = self.excluded_edge_indices
        need = self.num_nodes - 1

        for i in order:
            ii = int(i)
            if ii in fixed or ii in excluded:
                continue
            u, v = self.edge_list[ii]
            if uf.union(u, v):
                mst_edges.append((u, v))
                mst_cost += float(weights[ii])
                mst_length += float(self.edge_lengths[ii])
                if len(mst_edges) == need:
                    break

        if len(mst_edges) != need or uf.count_components() > 1:
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
            

        # else:  # Subgradient method with Polyak hybrid + cover cuts (λ, μ), depth-based freezing
        #     # --- Tunables / safety limits ---
        #     MAX_SOLUTIONS    = getattr(self, "max_primal_solutions", 50)
        #     max_iter         = min(self.max_iter, 200)

        #     # Polyak / momentum for λ
        #     self.momentum_beta = getattr(self, "momentum_beta", 0.9)
        #     gamma_base         = getattr(self, "gamma_base", 0.1)

        #     # μ update parameters
        #     gamma_mu         = getattr(self, "gamma_mu", 0.30)
        #     mu_increment_cap = getattr(self, "mu_increment_cap", 1.0)
        #     eps              = 1e-12

        #     # Depth-based behaviour
        #     max_cut_depth = getattr(self, "max_cut_depth", 30)   # where we ADD cuts
        #     max_mu_depth  = getattr(self, "max_mu_depth", 50)    # where we UPDATE μ / use cuts in dual
        #     is_root       = (depth == 0)

        #     # Node-level separation parameters
        #     max_active_cuts           = getattr(self, "max_active_cuts", 5)
        #     max_new_cuts_per_node     = getattr(self, "max_new_cuts_per_node", 5)
        #     min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
        #     dead_mu_threshold         = getattr(self, "dead_mu_threshold", 1e-6)

        #     # Extra iterations allowed at root
        #     root_max_iter = int(getattr(self, "root_max_iter", max_iter * 2))

        #     # Ensure cut structures exist
        #     if not hasattr(self, "best_cuts"):
        #         self.best_cuts = []   # list of (set(edges), rhs)
        #     if not hasattr(self, "best_cut_multipliers"):
        #         self.best_cut_multipliers = {}  # μ_i for each cut
        #     if not hasattr(self, "best_cut_multipliers_for_best_bound"):
        #         self.best_cut_multipliers_for_best_bound = {}  # μ at best LB

        #     # Which behaviour at this node?
        #     cutting_active_here = self.use_cover_cuts and (depth <= max_cut_depth)   # can ADD cuts
        #     mu_dynamic_here     = self.use_cover_cuts and (depth <= max_mu_depth)    # can UPDATE μ / use in dual
        #     cuts_present_here   = self.use_cover_cuts and bool(self.best_cuts)

        #     # Ensure λ starts in a reasonable range (consistent with compute_modified_weights)
        #     self.lmbda = max(0.0, min(getattr(self, "lmbda", 0.05), 1e4))

        #     polyak_enabled = True

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
        #     if not hasattr(self, "_rhs_eff"):
        #         self._rhs_eff = {}

        #     # ------------------------------------------------------------------
        #     # Separation policy (FIXED):
        #     #   - DO NOT do objective-only pre-separation at root.
        #     #   - Always delay separation to the first violating MST inside the loop.
        #     #   - Still obey depth limits: only add cuts when cutting_active_here AND μ is dynamic.
        #     # ------------------------------------------------------------------
        #     pending_sep = bool(cutting_active_here and mu_dynamic_here)

        #     # ------------------------------------------------------------------
        #     # 2) Compute rhs_eff and detect infeasibility (fixed edges + cuts)
        #     #    rhs_eff = rhs - |cut ∩ F_in|
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts:
        #         for idx_c, (cut, rhs) in enumerate(self.best_cuts):
        #             rhs_eff = int(rhs) - len(cut & F_in)
        #             self._rhs_eff[idx_c] = rhs_eff
        #             if rhs_eff < 0:
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
        #             new_mu[new_i]      = float(parent_mu_map.get(old_i, 0.0))
        #             new_mu_best[new_i] = float(parent_mu_map.get(old_i, 0.0))
        #             new_rhs_eff[new_i] = self._rhs_eff[old_i]

        #         self.best_cuts = new_cuts_list
        #         self.best_cut_multipliers = new_mu
        #         self.best_cut_multipliers_for_best_bound = new_mu_best
        #         self._rhs_eff = new_rhs_eff

        #     cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)

        #     # ------------------------------------------------------------------
        #     # 4) Build cut -> edge index arrays (for pricing/subgradients)
        #     # ------------------------------------------------------------------
        #     def _rebuild_cut_structures():
        #         nonlocal cut_edge_idx_free, cut_edge_idx_all, rhs_eff_vec

        #         cut_edge_idx_free = []
        #         cut_edge_idx_all  = []

        #         for cut, rhs in self.best_cuts:
        #             idxs_free = [
        #                 edge_idx[e] for e in cut
        #                 if (e not in F_in and e not in F_out) and (e in edge_idx)
        #             ]
        #             arr_free = (
        #                 np.fromiter(idxs_free, dtype=np.int32)
        #                 if idxs_free else np.empty(0, dtype=np.int32)
        #             )
        #             cut_edge_idx_free.append(arr_free)

        #             idxs_all = [edge_idx[e] for e in cut if e in edge_idx]
        #             arr_all  = (
        #                 np.fromiter(idxs_all, dtype=np.int32)
        #                 if idxs_all else np.empty(0, dtype=np.int32)
        #             )
        #             cut_edge_idx_all.append(arr_all)

        #         self._cut_edge_idx     = cut_edge_idx_free
        #         self._cut_edge_idx_all = cut_edge_idx_all

        #         rhs_eff_vec = (
        #             np.array([self._rhs_eff[i] for i in range(len(self.best_cuts))], dtype=float)
        #             if self.best_cuts else np.zeros(0, dtype=float)
        #         )

        #     cut_edge_idx_free = []
        #     cut_edge_idx_all  = []
        #     rhs_eff_vec       = np.zeros(0, dtype=float)

        #     if self.use_cover_cuts and self.best_cuts:
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

        #     if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
        #         self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)
        #     mst_mask = self._mst_mask

        #     # Decide iteration limit for this node:
        #     if is_root:
        #         iter_limit = root_max_iter * 1.1 if self.use_cover_cuts else root_max_iter
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
        #         cut_g_signed = []

        #         # 1a) ONE-SHOT delayed separation (root AND non-root)
        #         if (
        #             cutting_active_here
        #             and mu_dynamic_here
        #             and pending_sep
        #             and len(self.best_cuts) < max_active_cuts
        #             and mst_length > self.budget
        #         ):
        #             try:
        #                 cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []
        #                 print("sss")

        #                 T_loop = set(mst_edges)
        #                 scored_loop = []
        #                 F_in_set = set(F_in)  # (already defined above)

        #                 for cut, rhs in cand_cuts_loop:
        #                     S_set   = set(cut)
        #                     S_free  = S_set - F_in_set                 # remove fixed edges from LHS set
        #                     lhs_free = len(T_loop & S_free)            # only MST edges that are NOT fixed
        #                     rhs_eff  = int(rhs) - len(S_set & F_in_set)
        #                     violation = lhs_free - rhs_eff

        #                     if violation >= min_cut_violation_for_add:
        #                         scored_loop.append((violation, S_set, rhs))

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
        #                     MU0 = getattr(self, "mu_init", 0.0)  # safe default: 0 (avoid immediate decay overhead)
        #                     self.best_cut_multipliers[new_idx] = MU0
        #                     self.best_cut_multipliers_for_best_bound[new_idx] = MU0


        #                     # keep rhs_eff consistent
        #                     self._rhs_eff[new_idx] = int(rhs) - len(set(S) & F_in)
        #                     if self._rhs_eff[new_idx] < 0:
        #                         end_time = time()
        #                         LagrangianMST.total_compute_time += end_time - start_time
        #                         return float('inf'), self.best_upper_bound, node_new_cuts

        #                     max_cut_violation.append(0.0)
        #                     node_new_cuts.append((set(S), rhs))
        #                     added_any = True

        #                 if added_any:
        #                     _rebuild_cut_structures()
        #                     self._mw_cached = None
        #                     self._mw_mu     = np.zeros(len(cut_edge_idx_free), dtype=float)
        #                     cuts_present_here = True

        #             except Exception as e:
        #                 if self.verbose:
        #                     print(f"Error in delayed separation at depth {depth}, iter {iter_num}: {e}")
        #             finally:
        #                 pending_sep = False  # do at most once per node

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

        #         # 3) Dual value: L(λ, μ) = MST_cost - λ B - Σ μ_i rhs_eff_i
        #         lam_for_dual = max(0.0, min(self.lmbda, 1e4))

        #         if self.use_cover_cuts and len(rhs_eff_vec) > 0:
        #             mu_vec = np.fromiter(
        #                 (
        #                     max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
        #                     for i in range(len(rhs_eff_vec))
        #                 ),
        #                 dtype=float,
        #                 count=len(rhs_eff_vec),
        #             )
        #             cover_cut_penalty = float(mu_vec @ rhs_eff_vec)
        #         else:
        #             cover_cut_penalty = 0.0

        #         lagrangian_bound = mst_cost - lam_for_dual * self.budget - cover_cut_penalty
        #         # if cover_cut_penalty != 0.0:
        #             # print("ggg", cover_cut_penalty)
        #         # print("lagrangian bound:", lagrangian_bound)

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

        #         # 4) Subgradients
        #         knapsack_subgradient = float(mst_length - self.budget)
        #         # print("fff", mst_length)
        #         # print("lala",self.lmbda)
        #         # print("wer", knapsack_subgradient)

        #         # Fast skip: if MST feasible and all μ are ~0, don't pay cut gradient cost
        #         all_mu_small = (not self.best_cut_multipliers) or \
        #                     (max(self.best_cut_multipliers.values()) <= dead_mu_threshold)

        #         if cuts_present_here and mu_dynamic_here and len(cut_edge_idx_all) > 0 and not (is_feasible and all_mu_small):
        #             mst_mask[:] = False
        #             for e in mst_edges:
        #                 j = self.edge_indices.get(e)
        #                 if j is not None:
        #                     mst_mask[j] = True

        #             cut_g_signed = []
        #             cut_g_pos    = []

        #             for i, idxs_free in enumerate(cut_edge_idx_free):
        #                 lhs_free = int(mst_mask[idxs_free].sum()) if idxs_free.size else 0
        #                 g_i = float(lhs_free) - float(rhs_eff_vec[i])
        #                 cut_g_signed.append(g_i)
        #                 cut_g_pos.append(g_i if g_i > 0.0 else 0.0)

        #                 if g_i > max_cut_violation[i]:
        #                     max_cut_violation[i] = g_i

        #             cut_subgradients = cut_g_pos
        #         else:
        #             cut_subgradients = []
        #             cut_g_signed = []
        #             cut_g_pos = []


        #         norm_sq = knapsack_subgradient ** 2
        #         for g in cut_subgradients:
        #             norm_sq += float(g) ** 2

        #         # Polyak step size
        #         if polyak_enabled and self.best_upper_bound < float('inf') and norm_sq > 0.0:
        #             gap   = max(0.0, self.best_upper_bound - lagrangian_bound)
        #             alpha = gamma_base * gap / (norm_sq + eps)
        #         else:
        #             alpha = getattr(self, "step_size", 0.001)

        #         # λ update with momentum, then clamp
        #         v_prev = getattr(self, "_v_lambda", 0.0)
        #         v_new  = self.momentum_beta * v_prev + (1.0 - self.momentum_beta) * knapsack_subgradient
        #         self._v_lambda = v_new
        #         self.lmbda     = self.lmbda + alpha * v_new
        #         # print("ooo", alpha)

        #         if self.lmbda < 0.0:
        #             self.lmbda = 0.0
        #         if self.lmbda > 1e4:
        #             self.lmbda = 1e4

        #         # μ updates: projected subgradient for constraints sum_{e in S} x_e <= rhs_eff
        #         if mu_dynamic_here and len(cut_g_pos) > 0:
        #             for i, g in enumerate(cut_g_pos):
        #                 g = float(g)
        #                 if g <= 0.0:
        #                     continue

        #                 delta = gamma_mu * alpha * g

        #                 # cap only positive increment
        #                 if mu_increment_cap is not None:
        #                     delta = min(mu_increment_cap, delta)

        #                 mu_old = float(self.best_cut_multipliers.get(i, 0.0))
        #                 mu_new = mu_old + delta

        #                 # projection + clamp
        #                 if mu_new > 1e4:
        #                     mu_new = 1e4

        #                 self.best_cut_multipliers[i] = mu_new


        #         self.step_sizes.append(alpha)
        #         self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

        #     # ------------------------------------------------------------------
        #     # 6) Drop "dead" cuts globally
        #     # ------------------------------------------------------------------
        #     if self.use_cover_cuts and self.best_cuts and mu_dynamic_here:
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
        #                 new_mu_best[new_idx] = float(self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0))
        #                 new_rhs_eff[new_idx] = self._rhs_eff[old_idx]

        #             self.best_cuts = new_best_cuts
        #             self.best_cut_multipliers = new_mu
        #             self.best_cut_multipliers_for_best_bound = new_mu_best
        #             self._rhs_eff = new_rhs_eff

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

        
        else:  # Subgradient method with Polyak hybrid + cover cuts (λ, μ), depth-based freezing
            import os

            # --- Tunables / safety limits ---
            MAX_SOLUTIONS = getattr(self, "max_primal_solutions", 50)
            max_iter = min(self.max_iter, 200)

            # Polyak / momentum for λ
            self.momentum_beta = getattr(self, "momentum_beta", 0.7)
            gamma_base = getattr(self, "gamma_base", 0.05)

            # Safety controls for λ update
            fallback_alpha = getattr(self, "fallback_alpha", 1e-5)
            max_lambda_delta = getattr(self, "max_lambda_delta", 0.02)

            # μ update parameters
            gamma_mu = getattr(self, "gamma_mu", 0.25)
            mu_increment_cap = getattr(self, "mu_increment_cap", 0.002)

            eps = 1e-12

            # Depth-based behaviour
            max_cut_depth = getattr(self, "max_cut_depth", 30)
            max_mu_depth = getattr(self, "max_mu_depth", 50)
            is_root = depth == 0

            # Node-level separation parameters
            max_active_cuts = getattr(self, "max_active_cuts", 5)
            max_new_cuts_per_node = getattr(self, "max_new_cuts_per_node", 5)
            min_cut_violation_for_add = getattr(self, "min_cut_violation_for_add", 1.0)
            dead_mu_threshold = getattr(self, "dead_mu_threshold", 1e-6)

            # Extra iterations allowed at root
            root_max_iter = int(getattr(self, "root_max_iter", max_iter * 2))

            # ------------------------------------------------------------------
            # DEBUG SETTINGS
            # ------------------------------------------------------------------
            debug_cuts = False
            debug_iter_every = 1       # change to 5 or 10 if the log becomes too large
            debug_cut_max_rows = 10

            debug_log_path = getattr(
                self,
                "debug_cut_log_path",
                os.path.join(os.path.expanduser("~/Desktop"), "cut_debug_log.txt"),
            )

            # Clear the log only once at the root node
            if depth == 0:
                with open(debug_log_path, "w") as f:
                    f.write("CUT DEBUG LOG\n")
                    f.write("=" * 100 + "\n")

            def _dbg(msg, iter_num=None, force=False):
                if not debug_cuts:
                    return

                if iter_num is not None and not force:
                    if iter_num % debug_iter_every != 0:
                        return

                if iter_num is None:
                    line = f"[CUTDBG depth={depth}] {msg}"
                else:
                    line = f"[CUTDBG depth={depth} iter={iter_num}] {msg}"

                with open(debug_log_path, "a") as f:
                    f.write(line + "\n")

            def _edge_len(e):
                try:
                    return float(self.edge_lengths[self.edge_indices[e]])
                except Exception:
                    return float("nan")

            def _cut_len(cut):
                return sum(_edge_len(e) for e in cut if e in self.edge_indices)

            def _cut_repr(cut, max_edges=6):
                cut_list = sorted(list(cut))
                shown = cut_list[:max_edges]
                suffix = "" if len(cut_list) <= max_edges else f", ... +{len(cut_list) - max_edges}"
                return f"{shown}{suffix}"

            def _print_cut_table(stage, iter_num=None, force=False):
                if not debug_cuts:
                    return

                if iter_num is not None and not force:
                    if iter_num % debug_iter_every != 0:
                        return

                _dbg(
                    f"{stage}: active cuts = {len(getattr(self, 'best_cuts', []))}",
                    iter_num,
                    force,
                )

                if not getattr(self, "best_cuts", []):
                    return

                for i, (cut, rhs) in enumerate(self.best_cuts[:debug_cut_max_rows]):
                    mu = float(getattr(self, "best_cut_multipliers", {}).get(i, 0.0))
                    mu_best = float(getattr(self, "best_cut_multipliers_for_best_bound", {}).get(i, 0.0))
                    rhs_eff = getattr(self, "_rhs_eff", {}).get(i, rhs)

                    _dbg(
                        f"  cut[{i}] size={len(cut)} rhs={rhs} rhs_eff={rhs_eff} "
                        f"mu={mu:.6g} mu_best={mu_best:.6g} "
                        f"len_sum={_cut_len(cut):.3f} edges={_cut_repr(cut)}",
                        iter_num,
                        force,
                    )

                if len(self.best_cuts) > debug_cut_max_rows:
                    _dbg(
                        f"  ... {len(self.best_cuts) - debug_cut_max_rows} more cuts not shown",
                        iter_num,
                        force,
                    )

            # ------------------------------------------------------------------
            # Ensure cut structures exist
            # ------------------------------------------------------------------
            if not hasattr(self, "best_cuts"):
                self.best_cuts = []

            if not hasattr(self, "best_cut_multipliers"):
                self.best_cut_multipliers = {}

            if not hasattr(self, "best_cut_multipliers_for_best_bound"):
                self.best_cut_multipliers_for_best_bound = {}

            # Which behaviour at this node?
            cutting_active_here = self.use_cover_cuts and depth <= max_cut_depth
            mu_dynamic_here = self.use_cover_cuts and depth <= max_mu_depth
            use_cuts_in_dual_here = self.use_cover_cuts and bool(self.best_cuts)

            # Ensure λ starts in a reasonable range
            self.lmbda = max(0.0, min(getattr(self, "lmbda", 0.05), 1e4))

            polyak_enabled = True
            node_new_cuts = []

            _dbg(
                f"START NODE | use_cover_cuts={self.use_cover_cuts}, "
                f"cutting_active_here={cutting_active_here}, "
                f"mu_dynamic_here={mu_dynamic_here}, "
                f"use_cuts_in_dual_here={use_cuts_in_dual_here}, "
                f"lambda_start={self.lmbda:.6g}, "
                f"inherited_cuts={len(self.best_cuts)}, "
                f"log_file={debug_log_path}",
                force=True,
            )

            _print_cut_table("Inherited cuts before reduction", force=True)

            # ------------------------------------------------------------------
            # Quick guards
            # ------------------------------------------------------------------
            if not self.edge_list or self.num_nodes <= 1:
                _dbg("STOP: empty edge list or invalid graph", force=True)

                end_time = time()
                LagrangianMST.total_compute_time += end_time - start_time
                return self.best_lower_bound, self.best_upper_bound, node_new_cuts

            # Fixed / forbidden edges
            F_in = set(getattr(self, "fixed_edges", set()))
            F_out = set(getattr(self, "excluded_edges", set()))
            edge_idx = self.edge_indices

            self._rhs_eff = {}

            _dbg(
                f"Node fixings: |F_in|={len(F_in)}, |F_out|={len(F_out)}, "
                f"fixed_length={sum(_edge_len(e) for e in F_in if e in edge_idx):.3f}, "
                f"budget={self.budget:.3f}",
                force=True,
            )

            # ------------------------------------------------------------------
            # 2) Reduce inherited cuts and remove redundant cuts
            # ------------------------------------------------------------------
            if self.use_cover_cuts and self.best_cuts:
                old_mu = dict(getattr(self, "best_cut_multipliers", {}) or {})
                old_mu_best = dict(getattr(self, "best_cut_multipliers_for_best_bound", {}) or {})

                reduced_cuts = []
                reduced_mu = {}
                reduced_mu_best = {}
                reduced_rhs_eff = {}

                kept_count = 0
                redundant_count = 0

                for old_i, (cut, rhs) in enumerate(self.best_cuts):
                    S = set(cut)

                    S_fixed = S & F_in
                    S_excluded = S & F_out
                    S_free = S - F_in - F_out
                    rhs_eff = int(rhs) - len(S_fixed)

                    _dbg(
                        f"Reduce old_cut[{old_i}]: old_size={len(S)}, old_rhs={rhs}, "
                        f"|S_fixed|={len(S_fixed)}, |S_excluded|={len(S_excluded)}, "
                        f"|S_free|={len(S_free)}, rhs_eff={rhs_eff}, "
                        f"mu_old={float(old_mu.get(old_i, 0.0)):.6g}",
                        force=True,
                    )

                    if rhs_eff < 0:
                        _dbg(
                            f"STOP: inherited cut[{old_i}] makes node infeasible "
                            f"because rhs_eff={rhs_eff}<0",
                            force=True,
                        )

                        end_time = time()
                        LagrangianMST.total_compute_time += end_time - start_time
                        return float("inf"), self.best_upper_bound, node_new_cuts

                    # Redundant at this node
                    if len(S_free) <= rhs_eff:
                        redundant_count += 1
                        _dbg(
                            f"Drop old_cut[{old_i}] as redundant: "
                            f"|S_free|={len(S_free)} <= rhs_eff={rhs_eff}",
                            force=True,
                        )
                        continue

                    new_i = len(reduced_cuts)
                    reduced_cuts.append((set(S_free), int(rhs_eff)))

                    mu_val = float(old_mu.get(old_i, 0.0))
                    mu_best_val = float(old_mu_best.get(old_i, mu_val))

                    reduced_mu[new_i] = mu_val
                    reduced_mu_best[new_i] = mu_best_val
                    reduced_rhs_eff[new_i] = int(rhs_eff)
                    kept_count += 1

                self.best_cuts = reduced_cuts
                self.best_cut_multipliers = reduced_mu
                self.best_cut_multipliers_for_best_bound = reduced_mu_best
                self._rhs_eff = reduced_rhs_eff

                _dbg(
                    f"Cut reduction summary: kept={kept_count}, "
                    f"redundant_dropped={redundant_count}",
                    force=True,
                )

            _print_cut_table("Cuts after reduction", force=True)

            # ------------------------------------------------------------------
            # 3) Trim number of cuts
            # ------------------------------------------------------------------
            if self.use_cover_cuts and self.best_cuts and len(self.best_cuts) > max_active_cuts:
                parent_mu_map = getattr(self, "best_cut_multipliers_for_best_bound", None)

                if not parent_mu_map:
                    parent_mu_map = self.best_cut_multipliers

                idx_and_cut = list(enumerate(self.best_cuts))
                idx_and_cut.sort(
                    key=lambda ic: abs(parent_mu_map.get(ic[0], 0.0)),
                    reverse=True,
                )

                kept_old_indices = [old_i for old_i, _ in idx_and_cut[:max_active_cuts]]
                dropped_old_indices = [old_i for old_i, _ in idx_and_cut[max_active_cuts:]]

                _dbg(
                    f"Trim cuts: max_active_cuts={max_active_cuts}, "
                    f"kept_old_indices={kept_old_indices}, "
                    f"dropped_old_indices={dropped_old_indices}",
                    force=True,
                )

                idx_and_cut = idx_and_cut[:max_active_cuts]

                new_cuts_list = []
                new_mu = {}
                new_mu_best = {}
                new_rhs_eff = {}

                for new_i, (old_i, cut_rhs) in enumerate(idx_and_cut):
                    new_cuts_list.append(cut_rhs)
                    new_mu[new_i] = float(self.best_cut_multipliers.get(old_i, 0.0))
                    new_mu_best[new_i] = float(parent_mu_map.get(old_i, new_mu[new_i]))
                    new_rhs_eff[new_i] = int(self._rhs_eff.get(old_i, cut_rhs[1]))

                self.best_cuts = new_cuts_list
                self.best_cut_multipliers = new_mu
                self.best_cut_multipliers_for_best_bound = new_mu_best
                self._rhs_eff = new_rhs_eff

            cuts_present_here = self.use_cover_cuts and bool(self.best_cuts)
            use_cuts_in_dual_here = self.use_cover_cuts and bool(self.best_cuts)

            _print_cut_table("Cuts after trimming", force=True)

            # ------------------------------------------------------------------
            # 4) Build cut -> edge index arrays
            # ------------------------------------------------------------------
            def _rebuild_cut_structures():
                nonlocal cut_edge_idx_free, cut_edge_idx_all, rhs_eff_vec

                cut_edge_idx_free = []
                cut_edge_idx_all = []

                for i, (cut, rhs) in enumerate(self.best_cuts):
                    S = set(cut)

                    idxs_free = [
                        edge_idx[e]
                        for e in S
                        if e in edge_idx and e not in F_in and e not in F_out
                    ]

                    arr_free = (
                        np.fromiter(idxs_free, dtype=np.int32)
                        if idxs_free
                        else np.empty(0, dtype=np.int32)
                    )

                    cut_edge_idx_free.append(arr_free)

                    idxs_all = [edge_idx[e] for e in S if e in edge_idx]

                    arr_all = (
                        np.fromiter(idxs_all, dtype=np.int32)
                        if idxs_all
                        else np.empty(0, dtype=np.int32)
                    )

                    cut_edge_idx_all.append(arr_all)

                    if i not in self._rhs_eff:
                        self._rhs_eff[i] = int(rhs)

                self._cut_edge_idx = cut_edge_idx_free
                self._cut_edge_idx_all = cut_edge_idx_all

                rhs_eff_vec = (
                    np.array(
                        [self._rhs_eff[i] for i in range(len(self.best_cuts))],
                        dtype=float,
                    )
                    if self.best_cuts
                    else np.zeros(0, dtype=float)
                )

                _dbg(
                    f"Rebuilt cut structures: num_cuts={len(self.best_cuts)}, "
                    f"rhs_eff_vec={rhs_eff_vec.tolist()}, "
                    f"free_edge_counts={[len(a) for a in cut_edge_idx_free]}",
                    force=True,
                )

            cut_edge_idx_free = []
            cut_edge_idx_all = []
            rhs_eff_vec = np.zeros(0, dtype=float)

            if self.use_cover_cuts and self.best_cuts:
                _rebuild_cut_structures()

            max_cut_violation = [0.0 for _ in self.best_cuts]

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

            if not hasattr(self, "_mst_mask") or self._mst_mask.size != len(self.edge_weights):
                self._mst_mask = np.zeros(len(self.edge_weights), dtype=bool)

            mst_mask = self._mst_mask

            # Decide iteration limit for this node
            if is_root:
                iter_limit = root_max_iter * 1.1 if self.use_cover_cuts else root_max_iter
            else:
                # Optional depth decay: with lambda inheritance, deep children
                # only need to REFINE the parent's near-optimal lambda, not
                # rediscover it, so fewer iterations suffice. Controlled by
                # `child_iter_decay` (per-level multiplier) and `child_min_iter`
                # (floor). Both default to no-op values, so when unset the cap
                # is exactly the old flat `max_iter` -> non-negative runs, which
                # set neither, are unaffected.
                decay = getattr(self, "child_iter_decay", 1.0)
                min_iter = getattr(self, "child_min_iter", max_iter)
                if decay < 1.0 and depth > 0:
                    decayed = int(round(max_iter * (decay ** depth)))
                    iter_limit = max(min_iter, decayed)
                else:
                    iter_limit = max_iter

            sep_rounds = 0
            max_sep_rounds = 1
            separate_every = 5

            _dbg(
                f"Iteration setup: iter_limit={int(iter_limit)}, "
                f"max_sep_rounds={max_sep_rounds}, "
                f"separate_every={separate_every}",
                force=True,
            )

            # ------------------------------------------------------------------
            # 5) Subgradient iterations
            # ------------------------------------------------------------------
            for iter_num in range(int(iter_limit)):
                # --------------------------------------------------------------
                # 5.1) MST with current λ and μ
                # --------------------------------------------------------------
                try:
                    mst_cost, mst_length, mst_edges = self.compute_mst_incremental(
                        prev_weights,
                        prev_mst_edges,
                    )
                    mst_method = "incremental"

                except Exception as e:
                    _dbg(
                        f"Incremental MST failed: {e}. Falling back to full MST.",
                        iter_num,
                        force=True,
                    )

                    mst_cost, mst_length, mst_edges = self.compute_mst()
                    mst_method = "full"

                if (
                    not mst_edges
                    or math.isinf(mst_cost)
                    or math.isinf(mst_length)
                    or math.isnan(mst_cost)
                    or math.isnan(mst_length)
                ):
                    _dbg(
                        f"STOP: invalid MST. method={mst_method}, "
                        f"mst_cost={mst_cost}, mst_length={mst_length}, "
                        f"num_edges={len(mst_edges) if mst_edges else 0}",
                        iter_num,
                        force=True,
                    )

                    end_time = time()
                    LagrangianMST.total_compute_time += end_time - start_time
                    return float("inf"), self.best_upper_bound, node_new_cuts

                self.last_mst_edges = mst_edges
                prev_mst_edges = mst_edges

                _dbg(
                    f"MST: method={mst_method}, cost={mst_cost:.6g}, "
                    f"length={mst_length:.6g}, budget={self.budget:.6g}, "
                    f"budget_violation={mst_length - self.budget:.6g}, "
                    f"num_edges={len(mst_edges)}",
                    iter_num,
                )

                # --------------------------------------------------------------
                # 5.2) Delayed separation
                #
                # Modified:
                # We now separate at the first budget-violating MST, not only at
                # iteration 0 or multiples of separate_every.
                # max_sep_rounds still limits this to one separation round per node.
                # --------------------------------------------------------------
                should_separate = (
                    cutting_active_here
                    and mu_dynamic_here
                    and sep_rounds < max_sep_rounds
                    and len(self.best_cuts) < max_active_cuts
                    and mst_length > self.budget
                )

                _dbg(
                    f"Separation check: should_separate={should_separate}, "
                    f"sep_rounds={sep_rounds}/{max_sep_rounds}, "
                    f"active_cuts={len(self.best_cuts)}/{max_active_cuts}, "
                    f"budget_violated={mst_length > self.budget}",
                    iter_num,
                )

                if should_separate:
                    try:
                        cand_cuts_loop = self.generate_cover_cuts(mst_edges) or []

                        _dbg(
                            f"Generated candidate cuts: count={len(cand_cuts_loop)}",
                            iter_num,
                            force=True,
                        )

                        T_loop = set(mst_edges)
                        scored_loop = []

                        for cand_i, (cut, rhs) in enumerate(cand_cuts_loop):
                            S_set = set(cut)

                            S_fixed = S_set & F_in
                            S_excluded = S_set & F_out
                            S_free = S_set - F_in - F_out
                            rhs_eff_new = int(rhs) - len(S_fixed)

                            if rhs_eff_new < 0:
                                _dbg(
                                    f"STOP: candidate cut[{cand_i}] gives "
                                    f"rhs_eff_new={rhs_eff_new}<0",
                                    iter_num,
                                    force=True,
                                )

                                end_time = time()
                                LagrangianMST.total_compute_time += end_time - start_time
                                return float("inf"), self.best_upper_bound, node_new_cuts

                            if len(S_free) <= rhs_eff_new:
                                _dbg(
                                    f"Candidate cut[{cand_i}] dropped as redundant: "
                                    f"|S_free|={len(S_free)} <= rhs_eff={rhs_eff_new}",
                                    iter_num,
                                    force=True,
                                )
                                continue

                            lhs_free = len(T_loop & S_free)
                            violation = lhs_free - rhs_eff_new

                            _dbg(
                                f"Candidate cut[{cand_i}]: orig_size={len(S_set)}, "
                                f"|fixed|={len(S_fixed)}, "
                                f"|excluded|={len(S_excluded)}, "
                                f"|free|={len(S_free)}, "
                                f"rhs={rhs}, rhs_eff={rhs_eff_new}, "
                                f"lhs_on_current_MST={lhs_free}, "
                                f"violation={violation}, "
                                f"len_sum={_cut_len(S_free):.3f}",
                                iter_num,
                                force=True,
                            )

                            if violation >= min_cut_violation_for_add:
                                scored_loop.append(
                                    (float(violation), set(S_free), int(rhs_eff_new))
                                )

                        scored_loop.sort(
                            reverse=True,
                            key=lambda t: (t[0], len(t[1])),
                        )

                        remaining_slots = max(0, max_active_cuts - len(self.best_cuts))

                        if remaining_slots > 0:
                            scored_loop = scored_loop[
                                : min(max_new_cuts_per_node, remaining_slots)
                            ]
                        else:
                            scored_loop = []

                        _dbg(
                            f"Candidate cuts after filtering: "
                            f"kept_for_addition={len(scored_loop)}, "
                            f"remaining_slots={remaining_slots}",
                            iter_num,
                            force=True,
                        )

                        existing = {
                            frozenset(c): (i, int(rhs))
                            for i, (c, rhs) in enumerate(self.best_cuts)
                        }

                        changed_any = False

                        for violation, S, rhs in scored_loop:
                            fz = frozenset(S)

                            if fz in existing:
                                old_i, old_rhs = existing[fz]

                                if rhs < old_rhs:
                                    _dbg(
                                        f"Replace duplicate cut at index {old_i}: "
                                        f"old_rhs={old_rhs}, new_rhs={rhs}, "
                                        f"violation={violation}",
                                        iter_num,
                                        force=True,
                                    )

                                    self.best_cuts[old_i] = (set(S), int(rhs))
                                    self._rhs_eff[old_i] = int(rhs)
                                    max_cut_violation[old_i] = max(
                                        max_cut_violation[old_i],
                                        violation,
                                    )
                                    changed_any = True

                                else:
                                    _dbg(
                                        f"Skip duplicate cut: existing_rhs={old_rhs}, "
                                        f"new_rhs={rhs}, violation={violation}",
                                        iter_num,
                                        force=True,
                                    )

                                continue

                            self.best_cuts.append((set(S), int(rhs)))
                            new_idx = len(self.best_cuts) - 1

                            # Positive initial μ makes a newly added cut affect the next MST.
                            MU0 = getattr(self, "mu_init", 0.001)

                            self.best_cut_multipliers[new_idx] = float(MU0)
                            self.best_cut_multipliers_for_best_bound[new_idx] = float(MU0)
                            self._rhs_eff[new_idx] = int(rhs)

                            max_cut_violation.append(max(0.0, violation))
                            node_new_cuts.append((set(S), int(rhs)))

                            existing[fz] = (new_idx, int(rhs))
                            changed_any = True

                            _dbg(
                                f"ADD cut[{new_idx}]: size={len(S)}, rhs={rhs}, "
                                f"initial_mu={MU0}, violation={violation}, "
                                f"len_sum={_cut_len(S):.3f}, edges={_cut_repr(S)}",
                                iter_num,
                                force=True,
                            )

                        if changed_any:
                            _rebuild_cut_structures()

                            self._mw_cached = None
                            self._mw_mu = np.zeros(len(cut_edge_idx_free), dtype=float)

                            cuts_present_here = True
                            use_cuts_in_dual_here = self.use_cover_cuts and bool(self.best_cuts)

                            _print_cut_table(
                                "Cuts after separation/addition",
                                iter_num,
                                force=True,
                            )

                    except Exception as e:
                        _dbg(
                            f"ERROR in delayed separation at depth={depth}, "
                            f"iter={iter_num}: {e}",
                            iter_num,
                            force=True,
                        )

                    finally:
                        sep_rounds += 1

                # Prepare weights for next iteration
                prev_weights = getattr(self, "_last_mw", prev_weights)

                # --------------------------------------------------------------
                # 5.3) Primal and upper bound
                # --------------------------------------------------------------
                is_feasible = mst_length <= self.budget

                self._record_primal_solution(self.last_mst_edges, is_feasible)

                if is_feasible:
                    try:
                        real_weight, real_length = self.compute_real_weight_length()

                        if (
                            not math.isnan(real_weight)
                            and not math.isinf(real_weight)
                            and real_weight < self.best_upper_bound
                        ):
                            old_ub = self.best_upper_bound
                            self.best_upper_bound = real_weight

                            _dbg(
                                f"UB improved: old_UB={old_ub}, "
                                f"new_UB={self.best_upper_bound:.6g}, "
                                f"real_length={real_length:.6g}",
                                iter_num,
                                force=True,
                            )

                    except Exception as e:
                        _dbg(
                            f"ERROR updating primal solution: {e}",
                            iter_num,
                            force=True,
                        )

                # Repair fallback: when the natural Lagrangian MST is over
                # budget we still try to construct a feasible incumbent, so
                # B&B gets a finite UB to prune against. Gated by a flag that
                # defaults to OFF, so non-negative-correlation runs (which set
                # no overrides) are completely unaffected.
                elif getattr(self, "enable_primal_repair", False):
                    try:
                        # The budget-aware repair is strong but costly (a
                        # binary search of Kruskals). Running it every
                        # subgradient iteration is wasteful since the incumbent
                        # barely moves. Run the cheap min-length repair each
                        # iteration to guarantee a UB exists, but only run the
                        # expensive budget repair periodically and on the last
                        # iteration. Controlled by `budget_repair_every`.
                        every = getattr(self, "budget_repair_every", 25)
                        is_last = (iter_num >= iter_limit - 1)
                        # The expensive budget repair (mu-grid of Kruskals) only
                        # needs to run where it can actually improve the GLOBAL
                        # incumbent: at shallow depth. Deep nodes almost never
                        # beat the root's incumbent, so run only the cheap
                        # min-length repair there. This keeps per-node cost low
                        # so far more nodes are explored. Controlled by
                        # `budget_repair_max_depth` (default: root + a few).
                        max_depth = getattr(self, "budget_repair_max_depth", 3)
                        shallow = (self.depth <= max_depth)
                        want_budget = (
                            getattr(self, "use_budget_repair", False)
                            and shallow
                            and (iter_num % every == 0 or is_last)
                        )
                        # Temporarily toggle budget path per-iteration.
                        saved = getattr(self, "use_budget_repair", False)
                        self.use_budget_repair = want_budget
                        rw, rl, rep_edges = self.primal_repair()
                        self.use_budget_repair = saved
                        if (
                            rep_edges is not None
                            and not math.isnan(rw)
                            and not math.isinf(rw)
                            and rw < self.best_upper_bound
                        ):
                            old_ub = self.best_upper_bound
                            self.best_upper_bound = rw
                            self._record_primal_solution(rep_edges, True)
                            _dbg(
                                f"UB improved via repair: old_UB={old_ub}, "
                                f"new_UB={self.best_upper_bound:.6g}, "
                                f"length={rl:.6g}",
                                iter_num,
                                force=True,
                            )
                    except Exception as e:
                        _dbg(f"ERROR in primal_repair: {e}", iter_num, force=True)

                if len(self.primal_solutions) > MAX_SOLUTIONS:
                    self.primal_solutions = self.primal_solutions[-MAX_SOLUTIONS:]

                # --------------------------------------------------------------
                # 5.4) Dual value
                # --------------------------------------------------------------
                lam_for_dual = max(0.0, min(self.lmbda, 1e4))

                if use_cuts_in_dual_here and len(rhs_eff_vec) > 0:
                    mu_vec = np.fromiter(
                        (
                            max(0.0, min(self.best_cut_multipliers.get(i, 0.0), 1e4))
                            for i in range(len(rhs_eff_vec))
                        ),
                        dtype=float,
                        count=len(rhs_eff_vec),
                    )

                    cover_cut_penalty = float(mu_vec @ rhs_eff_vec)

                else:
                    mu_vec = np.zeros(0, dtype=float)
                    cover_cut_penalty = 0.0

                lagrangian_bound = (
                    mst_cost
                    - lam_for_dual * self.budget
                    - cover_cut_penalty
                )

                _dbg(
                    f"Dual: lambda={lam_for_dual:.6g}, "
                    f"mst_cost={mst_cost:.6g}, "
                    f"lambdaB={lam_for_dual * self.budget:.6g}, "
                    f"cover_penalty={cover_cut_penalty:.6g}, "
                    f"LB_candidate={lagrangian_bound:.6g}, "
                    f"best_LB_before={self.best_lower_bound:.6g}, "
                    f"UB={self.best_upper_bound}",
                    iter_num,
                )

                if len(mu_vec) > 0:
                    _dbg(
                        f"mu_vec={mu_vec.tolist()}, "
                        f"rhs_eff_vec={rhs_eff_vec.tolist()}",
                        iter_num,
                    )

                if (
                    not math.isnan(lagrangian_bound)
                    and not math.isinf(lagrangian_bound)
                    and abs(lagrangian_bound) < 1e10
                ):
                    if lagrangian_bound > self.best_lower_bound + 1e-6:
                        old_lb = self.best_lower_bound

                        self.best_lower_bound = lagrangian_bound
                        self.best_lambda = lam_for_dual
                        self.best_mst_edges = self.last_mst_edges
                        self.best_cost = mst_cost
                        self.best_cut_multipliers_for_best_bound = (
                            self.best_cut_multipliers.copy()
                        )

                        _dbg(
                            f"LB improved: old_LB={old_lb:.6g}, "
                            f"new_LB={self.best_lower_bound:.6g}, "
                            f"best_lambda={self.best_lambda:.6g}, "
                            f"saved_mu={self.best_cut_multipliers_for_best_bound}",
                            iter_num,
                            force=True,
                        )

                # --------------------------------------------------------------
                # 5.5) Subgradients
                # --------------------------------------------------------------
                knapsack_subgradient = float(mst_length - self.budget)

                all_mu_small = (
                    not self.best_cut_multipliers
                    or max(self.best_cut_multipliers.values()) <= dead_mu_threshold
                )

                if (
                    cuts_present_here
                    and mu_dynamic_here
                    and len(cut_edge_idx_free) > 0
                    and not (is_feasible and all_mu_small)
                ):
                    mst_mask[:] = False

                    for e in mst_edges:
                        j = self.edge_indices.get(e)
                        if j is not None:
                            mst_mask[j] = True

                    cut_g_signed = []
                    cut_g_pos = []

                    for i, idxs_free in enumerate(cut_edge_idx_free):
                        lhs_free = int(mst_mask[idxs_free].sum()) if idxs_free.size else 0
                        g_i = float(lhs_free) - float(rhs_eff_vec[i])

                        cut_g_signed.append(g_i)
                        cut_g_pos.append(g_i if g_i > 0.0 else 0.0)

                        if i < len(max_cut_violation) and g_i > max_cut_violation[i]:
                            max_cut_violation[i] = g_i

                        _dbg(
                            f"Cut subgradient cut[{i}]: lhs_free={lhs_free}, "
                            f"rhs_eff={rhs_eff_vec[i]}, "
                            f"g_signed={g_i}, "
                            f"g_pos={cut_g_pos[-1]}, "
                            f"mu_before={self.best_cut_multipliers.get(i, 0.0):.6g}",
                            iter_num,
                        )

                    # Modified:
                    # Use the signed cut subgradient in the norm and μ update.
                    # This allows μ to decrease when the cut becomes slack.
                    cut_subgradients = cut_g_signed

                else:
                    cut_g_signed = []
                    cut_g_pos = []
                    cut_subgradients = []

                    _dbg(
                        f"Skip cut subgradients: cuts_present={cuts_present_here}, "
                        f"mu_dynamic={mu_dynamic_here}, "
                        f"num_cut_arrays={len(cut_edge_idx_free)}, "
                        f"is_feasible={is_feasible}, "
                        f"all_mu_small={all_mu_small}",
                        iter_num,
                    )

                norm_sq = knapsack_subgradient ** 2

                for g in cut_subgradients:
                    norm_sq += float(g) ** 2

                # --------------------------------------------------------------
                # 5.6) Polyak step size
                # --------------------------------------------------------------
                if (
                    polyak_enabled
                    and self.best_upper_bound < float("inf")
                    and norm_sq > 0.0
                ):
                    gap = max(0.0, self.best_upper_bound - lagrangian_bound)
                    alpha = gamma_base * gap / (norm_sq + eps)
                else:
                    gap = None
                    # Before we have a finite UB, avoid the huge first lambda jump.
                    alpha = fallback_alpha

                _dbg(
                    f"Step: knapsack_g={knapsack_subgradient:.6g}, "
                    f"cut_g_signed={cut_g_signed}, "
                    f"cut_g_pos={cut_g_pos}, "
                    f"norm_sq={norm_sq:.6g}, "
                    f"gap={gap}, "
                    f"alpha={alpha:.6g}",
                    iter_num,
                )

                # --------------------------------------------------------------
                # 5.7) λ update
                # --------------------------------------------------------------
                lambda_before = self.lmbda

                v_prev = getattr(self, "_v_lambda", 0.0)
                v_new = (
                    self.momentum_beta * v_prev
                    + (1.0 - self.momentum_beta) * knapsack_subgradient
                )

                self._v_lambda = v_new

                delta_lambda = alpha * v_new
                delta_lambda = max(-max_lambda_delta, min(max_lambda_delta, delta_lambda))

                self.lmbda = self.lmbda + delta_lambda
                self.lmbda = max(0.0, min(self.lmbda, 1e4))

                _dbg(
                    f"Lambda update: before={lambda_before:.6g}, "
                    f"v_prev={v_prev:.6g}, "
                    f"v_new={v_new:.6g}, "
                    f"after={self.lmbda:.6g}",
                    iter_num,
                )

                # --------------------------------------------------------------
                # 5.8) μ updates
                #
                # Modified:
                # Signed projected update:
                #     μ_i <- max(0, μ_i + gamma_mu * alpha * g_i)
                #
                # If g_i > 0, the cut is violated and μ_i increases.
                # If g_i < 0, the cut is slack and μ_i decreases.
                # --------------------------------------------------------------
                if mu_dynamic_here and len(cut_g_signed) > 0:
                    for i, g in enumerate(cut_g_signed):
                        g = float(g)

                        delta = gamma_mu * alpha * g

                        # Symmetric cap because delta can now be positive or negative.
                        if mu_increment_cap is not None:
                            delta = max(-mu_increment_cap, min(mu_increment_cap, delta))

                        mu_old = float(self.best_cut_multipliers.get(i, 0.0))
                        mu_new = mu_old + delta
                        mu_new = max(0.0, min(mu_new, 1e4))

                        self.best_cut_multipliers[i] = mu_new

                        _dbg(
                            f"Mu signed update cut[{i}]: g={g:.6g}, "
                            f"delta={delta:.6g}, "
                            f"mu_old={mu_old:.6g}, "
                            f"mu_new={mu_new:.6g}",
                            iter_num,
                            force=True,
                        )

                self.step_sizes.append(alpha)
                self.multipliers.append((self.lmbda, self.best_cut_multipliers.copy()))

            # ------------------------------------------------------------------
            # 6) Drop dead cuts
            # ------------------------------------------------------------------
            if self.use_cover_cuts and self.best_cuts and mu_dynamic_here:
                keep_indices = []

                best_mu_map = getattr(
                    self,
                    "best_cut_multipliers_for_best_bound",
                    self.best_cut_multipliers,
                )

                _dbg(
                    f"Dead-cut check starts: active_cuts={len(self.best_cuts)}, "
                    f"max_cut_violation={max_cut_violation}",
                    force=True,
                )

                for i, (cut, rhs) in enumerate(self.best_cuts):
                    mu_i = float(self.best_cut_multipliers.get(i, 0.0))
                    mu_best_i = float(best_mu_map.get(i, 0.0))

                    ever_useful = (
                        i < len(max_cut_violation)
                        and max_cut_violation[i] > 0.0
                    ) or abs(mu_best_i) >= dead_mu_threshold

                    keep = not (
                        not ever_useful
                        and abs(mu_i) < dead_mu_threshold
                        and abs(mu_best_i) < dead_mu_threshold
                    )

                    _dbg(
                        f"Dead-cut decision cut[{i}]: "
                        f"max_violation={max_cut_violation[i] if i < len(max_cut_violation) else None}, "
                        f"mu_current={mu_i:.6g}, "
                        f"mu_best={mu_best_i:.6g}, "
                        f"ever_useful={ever_useful}, "
                        f"keep={keep}",
                        force=True,
                    )

                    if keep:
                        keep_indices.append(i)

                if len(keep_indices) < len(self.best_cuts):
                    _dbg(
                        f"Dropping dead cuts: keep_indices={keep_indices}, "
                        f"drop_count={len(self.best_cuts) - len(keep_indices)}",
                        force=True,
                    )

                    new_best_cuts = []
                    new_mu = {}
                    new_mu_best = {}
                    new_rhs_eff = {}

                    for new_idx, old_idx in enumerate(keep_indices):
                        new_best_cuts.append(self.best_cuts[old_idx])
                        new_mu[new_idx] = float(
                            self.best_cut_multipliers.get(old_idx, 0.0)
                        )
                        new_mu_best[new_idx] = float(
                            self.best_cut_multipliers_for_best_bound.get(old_idx, 0.0)
                        )
                        new_rhs_eff[new_idx] = int(
                            self._rhs_eff.get(old_idx, self.best_cuts[old_idx][1])
                        )

                    self.best_cuts = new_best_cuts
                    self.best_cut_multipliers = new_mu
                    self.best_cut_multipliers_for_best_bound = new_mu_best
                    self._rhs_eff = new_rhs_eff

                    if self.best_cuts:
                        _rebuild_cut_structures()
                    else:
                        self._cut_edge_idx = []
                        self._cut_edge_idx_all = []
                        rhs_eff_vec = np.zeros(0, dtype=float)

            _print_cut_table("Final cuts before returning from node", force=True)

            # ------------------------------------------------------------------
            # 7) Restore best λ and μ to pass to children
            #
            # Unchanged strategy:
            # λ and μ are both restored to the values that gave the best lower bound.
            # ------------------------------------------------------------------
            if hasattr(self, "best_lambda"):
                _dbg(
                    f"Restore lambda: current={self.lmbda:.6g}, "
                    f"best_lambda={self.best_lambda:.6g}",
                    force=True,
                )
                self.lmbda = self.best_lambda

            if hasattr(self, "best_cut_multipliers_for_best_bound"):
                _dbg(
                    f"Restore best μ for children: "
                    f"{self.best_cut_multipliers_for_best_bound}",
                    force=True,
                )
                self.best_cut_multipliers = (
                    self.best_cut_multipliers_for_best_bound.copy()
                )

            _dbg(
                f"END NODE: best_LB={self.best_lower_bound:.6g}, "
                f"best_UB={self.best_upper_bound}, "
                f"return_new_cuts={len(node_new_cuts)}, "
                f"final_active_cuts={len(self.best_cuts)}",
                force=True,
            )

            end_time = time()
            LagrangianMST.total_compute_time += end_time - start_time
            return self.best_lower_bound, self.best_upper_bound, node_new_cuts


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

    def primal_repair_budget(self):
        """
        Strong budget-feasible primal heuristic via a parametric MST.

        The min-length tree (used by primal_repair) wastes the budget: it picks
        the shortest tree even when far more length is allowed, paying huge
        weight. Instead we compute the MST under a blended cost
            cost_mu(e) = weight(e) + mu * length(e)
        and binary-search mu >= 0 so the resulting tree's LENGTH lands just
        under the budget. Small mu -> min-weight tree (low weight, long); large
        mu -> min-length tree. The crossover tree spends the budget on length
        to buy low weight, which is exactly what the optimum does. Uses the
        fast argsort Kruskal, so it is cheap even at 500+ nodes.

        Returns (weight, length, edges) or (inf, inf, None) if infeasible.
        """
        ei = self.edge_indices
        W = self.edge_weights
        L = self.edge_lengths
        budget = self.budget

        def tree_at(mu):
            blended = W + mu * L
            # honor fixed/excluded via the existing argsort kruskal
            _, _, edges = self._argsort_kruskal(blended)
            if not edges:
                return None, float("inf"), float("inf")
            edges = [tuple(sorted(e)) for e in edges]
            w = float(sum(W[ei[e]] for e in edges))
            l = float(sum(L[ei[e]] for e in edges))
            return edges, w, l

        # Feasibility floor: even the min-length tree must fit, else infeasible.
        e_hi, w_hi, l_hi = tree_at(1e9)  # ~ min-length tree
        if e_hi is None or l_hi > budget:
            return float("inf"), float("inf"), None

        # If the min-weight tree already fits, it is optimal for this relaxation.
        e_lo, w_lo, l_lo = tree_at(0.0)
        if e_lo is not None and l_lo <= budget:
            return w_lo, l_lo, e_lo

        # The parametric MST length is NON-MONOTONIC in a way that defeats
        # binary search: feasible trees can appear in isolated mu-bands
        # separated by large breakpoints, so a bisection that pushes toward the
        # feasibility boundary often locks onto the wasteful min-length tree and
        # misses a far better feasible tree at moderate mu. Instead we SCAN a
        # geometric grid of mu values, keep the lowest-WEIGHT feasible tree, and
        # also remember the lowest-weight infeasible tree just over budget to
        # repair as a backup.
        # Find the feasibility BREAKPOINT by bisection on mu: the smallest mu
        # at which the parametric tree first fits the budget. The best low-
        # weight feasible incumbent lives right at this transition. This is much
        # cheaper than a dense grid (≈50 Kruskals via bisection vs grid*refine)
        # and finds an equal-or-better tree, because it targets the exact
        # breakpoint instead of sampling near it.
        best = (w_hi, l_hi, e_hi)              # feasible fallback (min-length)

        lo, hi = 0.0, 1e9                       # lo: over budget, hi: feasible
        for _ in range(getattr(self, "budget_repair_bisect_iters", 50)):
            mid = (lo + hi) / 2.0
            edges, w, l = tree_at(mid)
            if edges is None:
                lo = mid
                continue
            if l <= budget:
                if w < best[0]:
                    best = (w, l, edges)
                hi = mid
            else:
                lo = mid

        # The tree just BELOW the breakpoint (mu=lo) is over budget but has the
        # lowest weight near here; shorten it to budget for the best incumbent.
        e_over, w_over, l_over = tree_at(lo)
        if e_over is not None and l_over > budget:
            rep = self._shorten_to_budget(e_over)
            if rep is not None:
                rw, rl, redges = rep
                if rl <= budget and rw < best[0]:
                    best = (rw, rl, redges)

        # Backup: shorten the absolute min-weight tree (mu=0) too.
        if e_lo is not None and w_lo < best[0]:
            rep = self._shorten_to_budget(e_lo)
            if rep is not None:
                rw, rl, redges = rep
                if rl <= budget and rw < best[0]:
                    best = (rw, rl, redges)

        return best[0], best[1], best[2]

    def _shorten_to_budget(self, tree_edges):
        """
        Given a spanning tree slightly OVER budget on length, repeatedly swap
        its longest-length edges for shorter non-tree edges that reconnect the
        two components, choosing swaps that cut the most length per unit weight
        gained, until the tree's length <= budget. Returns (weight, length,
        edges) or None if it cannot be made feasible within the effort cap.

        This fixes the parametric-MST 'gap' case: the input tree has near-
        optimal (low) weight but is a few percent too long; a handful of swaps
        recover feasibility while keeping weight low.
        """
        ei = self.edge_indices
        W = self.edge_weights
        L = self.edge_lengths
        budget = self.budget

        tree = set(tuple(sorted(e)) for e in tree_edges)
        cur_len = float(sum(L[ei[e]] for e in tree))
        cur_w = float(sum(W[ei[e]] for e in tree))

        # Candidate replacement edges (non-tree, not excluded), shortest first.
        non_tree = [
            e for e in self.edge_list
            if e not in tree and ei[e] not in self.excluded_edge_indices
        ]
        non_tree.sort(key=lambda e: L[ei[e]])

        max_swaps = getattr(self, "shorten_max_swaps", 20 * self.num_nodes)
        swaps = 0

        # Build adjacency once; maintain it incrementally across swaps.
        adj = {}
        for (u, v) in tree:
            adj.setdefault(u, []).append((v, (u, v)))
            adj.setdefault(v, []).append((u, (u, v)))

        def path_edges(su, sv):
            # BFS path su->sv over current tree adjacency
            prev = {su: None}
            stack = [su]
            while stack:
                x = stack.pop()
                if x == sv:
                    break
                for (y, edge) in adj.get(x, []):
                    if y not in prev:
                        prev[y] = (x, edge)
                        stack.append(y)
            if sv not in prev:
                return []
            out = []
            node = sv
            while prev[node] is not None:
                px, edge = prev[node]
                out.append(tuple(sorted(edge)))
                node = px
            return out

        for add_e in non_tree:
            if cur_len <= budget or swaps >= max_swaps:
                break
            add_l = float(L[ei[add_e]])
            add_w = float(W[ei[add_e]])
            su, sv = add_e
            cyc = path_edges(su, sv)
            if not cyc:
                continue
            # Drop the LONGEST edge on the cycle that is longer than add_e,
            # to reduce length; among those pick the one giving best length cut.
            best_drop = None
            best_dl = 0.0
            for de in cyc:
                dl = float(L[ei[de]]) - add_l   # length reduction if we swap
                if dl > best_dl:
                    best_dl = dl
                    best_drop = de
            if best_drop is None or best_dl <= 0:
                continue
            # Apply swap: remove best_drop, add add_e
            tree.discard(best_drop); tree.add(add_e)
            cur_len += add_l - float(L[ei[best_drop]])
            cur_w += add_w - float(W[ei[best_drop]])
            # update adjacency
            du, dv = best_drop
            adj[du] = [(y, e) for (y, e) in adj.get(du, []) if tuple(sorted(e)) != best_drop]
            adj[dv] = [(y, e) for (y, e) in adj.get(dv, []) if tuple(sorted(e)) != best_drop]
            adj.setdefault(su, []).append((sv, add_e))
            adj.setdefault(sv, []).append((su, add_e))
            swaps += 1

        if cur_len <= budget:
            # Feasible now. Spend remaining slack to REDUCE weight: swap in
            # low-weight non-tree edges, dropping a heavier tree edge on the
            # induced cycle, as long as length stays within budget. This pulls
            # the incumbent down toward the optimum instead of stopping at the
            # first feasible tree.
            improve_cap = getattr(self, "shorten_improve_swaps", 0)
            cand = [
                e for e in self.edge_list
                if e not in tree and ei[e] not in self.excluded_edge_indices
            ]
            cand.sort(key=lambda e: W[ei[e]])  # cheapest weight first
            imp = 0
            for add_e in cand:
                if imp >= improve_cap:
                    break
                add_w = float(W[ei[add_e]]); add_l = float(L[ei[add_e]])
                su, sv = add_e
                cyc = path_edges(su, sv)
                if not cyc:
                    continue
                # drop the heaviest-weight cycle edge whose swap keeps budget
                best_drop = None; best_gain = 0.0
                for de in cyc:
                    new_len = cur_len - float(L[ei[de]]) + add_l
                    if new_len > budget:
                        continue
                    gain = float(W[ei[de]]) - add_w
                    if gain > best_gain:
                        best_gain = gain; best_drop = de
                if best_drop is None or best_gain <= 0:
                    continue
                tree.discard(best_drop); tree.add(add_e)
                cur_len += add_l - float(L[ei[best_drop]])
                cur_w += add_w - float(W[ei[best_drop]])
                du, dv = best_drop
                adj[du] = [(y, e) for (y, e) in adj.get(du, []) if tuple(sorted(e)) != best_drop]
                adj[dv] = [(y, e) for (y, e) in adj.get(dv, []) if tuple(sorted(e)) != best_drop]
                adj.setdefault(su, []).append((sv, add_e))
                adj.setdefault(sv, []).append((su, add_e))
                imp += 1
            return cur_w, cur_len, list(tree)
        return None

    def primal_repair(self):
        """
        Produce a budget-FEASIBLE spanning tree (an incumbent) regardless of
        whether the current Lagrangian MST is over budget.

        Strategy:
          1. Build the minimum-LENGTH spanning tree over the allowed edges
             (respecting fixed/excluded via custom_kruskal on edge_lengths).
             This is the shortest possible tree for this node; if its length
             still exceeds the budget, the node is genuinely infeasible.
          2. If feasible, try to lower its real weight with budget-preserving
             swaps: for each non-tree edge, if adding it and dropping the
             heaviest-weight edge on the induced cycle keeps length <= budget
             and reduces weight, do it. Cheap local improvement, optional.

        Returns (weight, length, edges) for a feasible tree, or
        (inf, inf, None) if no feasible tree exists at this node.
        """
        # Step 1: minimum-length tree using the existing Kruskal machinery.
        _, min_len, len_tree = self.custom_kruskal(self.edge_lengths)
        if not len_tree or min_len == float("inf") or min_len > self.budget:
            return float("inf"), float("inf"), None

        # Strong budget-aware path (opt-in, negative-correlation only): the
        # min-length tree wastes budget and pays huge weight; the parametric
        # MST spends the budget to buy low weight. Falls back to the min-length
        # tree below if it somehow fails.
        if getattr(self, "use_budget_repair", False):
            bw, bl, be = self.primal_repair_budget()
            if be is not None and bl <= self.budget:
                return bw, bl, be

        tree = [tuple(sorted(e)) for e in len_tree]
        tree_set = set(tree)
        ei = self.edge_indices
        W = self.edge_weights
        L = self.edge_lengths

        def tree_weight(edges):
            return float(sum(W[ei[e]] for e in edges))

        def tree_length(edges):
            return float(sum(L[ei[e]] for e in edges))

        cur_len = tree_length(tree)
        cur_w = tree_weight(tree)

        # Step 2: budget-preserving weight-reducing swaps (bounded effort).
        # The min-length tree is ALREADY a valid feasible incumbent, so the
        # swaps are pure optional improvement. They are O(candidates * n) with a
        # Python BFS per candidate, which is far too slow on large/dense graphs,
        # so we skip improvement entirely beyond a size threshold. B&B still
        # gets a finite UB from the min-length tree itself.
        repair_improve_cap = getattr(self, "repair_improve_max_nodes", 120)
        if self.num_nodes > repair_improve_cap:
            return cur_w, cur_len, tree

        # Non-tree candidate edges, cheapest weight first.
        non_tree = [
            e for e in self.edge_list
            if e not in tree_set
            and ei[e] not in self.excluded_edge_indices
        ]
        non_tree.sort(key=lambda e: W[ei[e]])

        max_swaps = min(len(non_tree), 2 * self.num_nodes)
        swaps_done = 0

        for add_e in non_tree:
            if swaps_done >= max_swaps:
                break
            # Find the cycle created by adding add_e: path between its endpoints
            # in the current tree.
            adj = {}
            for (u, v) in tree:
                adj.setdefault(u, []).append((v, (u, v)))
                adj.setdefault(v, []).append((u, (u, v)))
            su, sv = add_e
            # BFS for the path su -> sv
            prev = {su: None}
            stack = [su]
            found = False
            while stack:
                x = stack.pop()
                if x == sv:
                    found = True
                    break
                for (y, edge) in adj.get(x, []):
                    if y not in prev:
                        prev[y] = (x, edge)
                        stack.append(y)
            if not found:
                continue
            # Reconstruct cycle edges.
            cycle_edges = []
            node = sv
            while prev[node] is not None:
                px, edge = prev[node]
                cycle_edges.append(tuple(sorted(edge)))
                node = px
            if not cycle_edges:
                continue
            # Candidate to drop: the heaviest-weight tree edge on the cycle
            # whose removal keeps us within budget after adding add_e.
            add_w = float(W[ei[add_e]])
            add_l = float(L[ei[add_e]])
            best_drop = None
            best_gain = 0.0
            for drop_e in cycle_edges:
                new_len = cur_len - float(L[ei[drop_e]]) + add_l
                if new_len > self.budget:
                    continue
                gain = float(W[ei[drop_e]]) - add_w  # weight reduction
                if gain > best_gain:
                    best_gain = gain
                    best_drop = drop_e
            if best_drop is not None and best_gain > 0:
                tree_set.discard(best_drop)
                tree_set.add(add_e)
                tree = list(tree_set)
                cur_len = cur_len - float(L[ei[best_drop]]) + add_l
                cur_w = cur_w - float(W[ei[best_drop]]) + add_w
                swaps_done += 1

        return cur_w, cur_len, list(tree_set)