# import heapq
# import random
import networkx as nx
from lagrangianrelaxation import LagrangianMST
from branchandbound import Node
# , BranchAndBound, RandomBranchingRule
import math
from collections import defaultdict  # Add at top if not imported
import random

from contextlib import contextmanager

class SolverPool:
    """
    Keeps a small pool of LagrangianMST objects that share the same static
    graph/arrays. Each probe 'borrows' a solver, resets it, runs a few iters,
    then returns it to the pool.
    """
    def __init__(self, factory, size=3):
        self._objs = [factory() for _ in range(size)]
        self._free = self._objs.copy()

    @contextmanager
    def borrow(self):
        # If all are busy, reuse the first one; strong-branching is sequential here
        obj = self._free.pop() if self._free else self._objs[0]
        try:
            yield obj
        finally:
            # Make sure per-iteration buffers are cleared between probes
            cleanup = getattr(obj, "clear_iteration_state", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass
            if obj not in self._free:
                self._free.append(obj)


# def _child_filter_and_remap(cuts, mu_dict, edge_indices, fixed_child_edges, excluded_child_edges):
#     """
#     Filter cuts for a specific child and remap multipliers so indices stay dense.
#     - cuts: list[(set[(u,v)], rhs)]   (edges normalized)
#     - mu_dict: {old_idx -> μ} aligned with 'cuts'
#     - edge_indices: dict[(u,v) -> idx] (normalized; from solver)
#     - fixed_child_edges / excluded_child_edges: sets of normalized edges for that child
#     Returns: (kept_cuts, kept_mu_dict) with new contiguous indices.
#     """
#     kept_cuts, kept_mu = [], {}
#     for old_i, (cut, rhs) in enumerate(cuts):
#         # normalize & drop unknown edges
#         cut_n = {tuple(sorted(e)) for e in cut if tuple(sorted(e)) in edge_indices}
#         # effective RHS given child's fixed edges
#         rhs_eff = int(rhs) - len(cut_n & fixed_child_edges)
#         if rhs_eff <= 0:
#             continue
#         # free edges available in child
#         free_ids = [edge_indices[e] for e in cut_n
#                     if (e not in fixed_child_edges) and (e not in excluded_child_edges) and (e in edge_indices)]
#         if not free_ids:
#             continue
#         new_i = len(kept_cuts)
#         kept_cuts.append((cut_n, int(rhs)))
#         kept_mu[new_i] = float(mu_dict.get(old_i, 0.0))  # inherit μ for survivors; new defaults upstream to 0.0
#     return kept_cuts, kept_mu

# --- REPLACE the whole helper with this version ---
# def _child_filter_and_remap(
#     cuts,                      # list[(set[(u,v)], rhs)]  (normalized edges)
#     mu_dict,                   # {old_idx -> μ}, aligned with `cuts`
#     edge_indices,              # dict[(u,v)->idx]
#     fixed_child_edges,         # set[(u,v)] fixed in child
#     excluded_child_edges,      # set[(u,v)] excluded in child
#     last_parent_tree_mask,     # np.array(bool) over all edges (parent's last MST)
#     K_keep=8,
#     mu_min=1e-3,
#     mu_cap=1e3,
#     damp=0.5
# ):
#     """
#     Filter cuts for a specific child and remap μ densely.
#     Keep only viable & effective cuts:
#       - effective RHS > 0 after child fixing
#       - at least two free edges remain
#       - active (violation ≥ 1 on parent's last tree) or had μ >= mu_min
#     Score: violation / sqrt(|free|)
#     Dampen μ on inheritance, and cap to mu_cap.
#     """
#     candidates = []
#     for old_i, (cut, rhs) in enumerate(cuts):
#         # normalize + known edges only
#         cut_n = {tuple(sorted(e)) for e in cut if tuple(sorted(e)) in edge_indices}

#         # effective RHS in child
#         rhs_eff = int(rhs) - len(cut_n & fixed_child_edges)
#         if rhs_eff <= 0:
#             continue

#         # free edges remaining in child
#         free_ids = [edge_indices[e] for e in cut_n
#                     if (e not in fixed_child_edges)
#                     and (e not in excluded_child_edges)
#                     and (e in edge_indices)]
#         if len(free_ids) < 2:
#             continue

#         # activity on parent's last tree
#         lhs = int(last_parent_tree_mask[free_ids].sum()) if len(free_ids) else 0
#         viol = max(0, lhs - rhs_eff)
#         mu_i = float(mu_dict.get(old_i, 0.0))

#         if viol < 1 and mu_i < mu_min:
#             # neither active nor had meaningful μ — drop it
#             continue

#         score = viol / max(1, int(len(free_ids) ** 0.5))
#         candidates.append((score, tuple(sorted(cut_n)), int(rhs), mu_i, rhs_eff, tuple(free_ids)))

#     if not candidates:
#         return [], {}

#     # deduplicate by edge-set key, keep best (higher score, then stricter rhs)
#     best = {}
#     for score, cut_n, rhs, mu_i, rhs_eff, free_ids in candidates:
#         key = frozenset(cut_n)
#         prev = best.get(key)
#         if (prev is None) or (score > prev[0]) or (rhs > prev[1]):
#             best[key] = (score, rhs, mu_i, rhs_eff, cut_n, free_ids)

#     # top-K
#     picked = sorted(best.values(), key=lambda t: t[0], reverse=True)[:K_keep]

#     # remap densely + damp μ
#     kept_cuts, kept_mu = [], {}
#     for new_i, (score, rhs, mu_i, rhs_eff, cut_n, free_ids) in enumerate(picked):
#         kept_cuts.append((set(cut_n), int(rhs)))
#         mu_new = min(mu_cap, damp * mu_i)
#         # if big chunk of rhs eaten by fixing, damp again
#         if rhs_eff <= max(1, rhs // 2):
#             mu_new *= 0.5
#         kept_mu[new_i] = mu_new

#     return kept_cuts, kept_mu




class MSTNode(Node):
    _solver_pool = None  # shared across all nodes in one solve

    def __init__(self, edges, num_nodes, budget, fixed_edges=set(), excluded_edges=set(), branched_edges=set(),
                 initial_lambda=0.05, inherit_lambda=False, branching_rule="random_mst",
                 step_size=0.00001, inherit_step_size=False, use_cover_cuts=False, cut_frequency=5,
                 node_cut_frequency=10, parent_cover_cuts=None, parent_cover_multipliers=None,
                 use_bisection=False, max_iter=10, verbose=False, depth=0,
                 pseudocosts_up=None, pseudocosts_down=None, counts_up=None, counts_down=None,
                 reliability_eta=5, lookahead_lambda=4):
        if depth == 0:
            MSTNode.global_edges = [(min(u, v), max(u, v), w, l) for u, v, w, l in edges]
            MSTNode.global_graph = nx.Graph()
            MSTNode.global_graph.add_edges_from(
                [(u, v, {"w": w, "l": l}) for u, v, w, l in MSTNode.global_edges]
            )
            MSTNode._solver_pool = None  # reset pool for a fresh instance

        self.pseudocosts_up = pseudocosts_up or defaultdict(float)
        self.pseudocosts_down = pseudocosts_down or defaultdict(float)
        self.counts_up = counts_up or defaultdict(int)
        self.counts_down = counts_down or defaultdict(int)
        self.reliability_eta = reliability_eta


        self.lookahead_lambda = lookahead_lambda

        self.depth = depth
        self.fixed_edges = {tuple(sorted((u, v))) for u, v in fixed_edges}
        self.excluded_edges = {tuple(sorted((u, v))) for u, v in excluded_edges}
        self.branched_edges = {tuple(sorted((u, v))) for u, v in branched_edges}
        if not hasattr(MSTNode, "global_edges"):
            MSTNode.global_edges = edges
        self.edges = MSTNode.global_edges

        self.num_nodes = num_nodes
        self.budget = budget

        self.inherit_lambda = inherit_lambda
        self.initial_lambda = initial_lambda if initial_lambda is not None else 0.05
        self.branching_rule = branching_rule
        self.step_size = step_size
        self.inherit_step_size = inherit_step_size

        self.use_cover_cuts = use_cover_cuts
        self.cut_frequency = cut_frequency
        self.node_cut_frequency = node_cut_frequency
        self.use_bisection = use_bisection
        self.verbose = verbose

        self.lagrangian_solver = LagrangianMST(
            MSTNode.global_edges, num_nodes, budget, self.fixed_edges, self.excluded_edges,
            initial_lambda=self.initial_lambda if not inherit_lambda else initial_lambda,
            step_size=self.step_size, max_iter=max_iter, p=0.95,
            use_cover_cuts=self.use_cover_cuts, cut_frequency=self.cut_frequency,
            use_bisection=self.use_bisection, verbose=self.verbose,
            shared_graph=MSTNode.global_graph
        )
        self.lagrangian_solver.graph = MSTNode.global_graph

        if MSTNode._solver_pool is None:
            def _factory():
                return LagrangianMST(
                    MSTNode.global_edges, self.num_nodes, self.budget,
                    fixed_edges=set(), excluded_edges=set(),
                    initial_lambda=self.initial_lambda,
                    step_size=self.step_size, max_iter=5, p=0.95,
                    use_cover_cuts=self.use_cover_cuts, cut_frequency=self.cut_frequency,
                    use_bisection=False, verbose=False, shared_graph=MSTNode.global_graph
                )
            MSTNode._solver_pool = SolverPool(_factory, size=1)
        self._sb_pool = MSTNode._solver_pool

        self.active_cuts = []
        self.cut_multipliers = {}
        if parent_cover_cuts:
            for cut_idx, (cut, rhs) in enumerate(parent_cover_cuts):
                normalized_cut = {tuple(sorted((u, v))) for u, v in cut}
                new_idx = len(self.active_cuts)
                self.active_cuts.append((normalized_cut, rhs))
                self.cut_multipliers[new_idx] = (
                    parent_cover_multipliers.get(cut_idx, 0.001)
                    if parent_cover_multipliers else 0.001
                )

        self.local_lower_bound, self.best_upper_bound, self.new_cuts = self.lagrangian_solver.solve(
            inherited_cuts=[(set(tuple(sorted((u, v))) for u, v in cut), rhs) for cut, rhs in self.active_cuts],
            inherited_multipliers=self.cut_multipliers,
            depth=self.depth
        )

        # self.mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.last_mst_edges]
        raw_edges = self.lagrangian_solver.last_mst_edges
        if not raw_edges:
            if self.verbose:
                print("Warning: solver returned no MST; treating as empty node.")
            raw_edges = []
            self.lagrangian_solver.last_mst_edges = []

        self.mst_edges = [tuple(sorted((u, v))) for u, v in raw_edges]

        self.actual_cost, _ = self.lagrangian_solver.compute_real_weight_length()
        # self.lagrangian_solver.clear_iteration_state()   # safe here


        super().__init__(self.local_lower_bound)

        if self.verbose:
            print(
                f"Node initialized: lower_bound={self.local_lower_bound}, "
                f"upper_bound={self.best_upper_bound}, lambda={self.lagrangian_solver.best_lambda}, "
                f"fixed={self.fixed_edges}, excluded={self.excluded_edges}"
            )


    def __lt__(self, other):
        return self.local_lower_bound < other.local_lower_bound

    def is_child_likely_feasible(self):
        fixed_length = sum(
            self.lagrangian_solver.edge_attributes[(u, v)][1]
            for u, v in self.fixed_edges
        )
        num_fixed_edges = len(self.fixed_edges)
        num_nodes_covered = len(set(u for u, v in self.fixed_edges) | set(v for u, v in self.fixed_edges))
        components = max(1, num_nodes_covered - num_fixed_edges + 1)
        edges_needed = self.num_nodes - num_nodes_covered + components - 1

        if edges_needed < 0:
            return False

        min_edge_length = float('inf')
        for u, v, _, l in self.edges:
            if (u, v) not in self.excluded_edges:
                min_edge_length = min(min_edge_length, l)

        estimated_length = fixed_length + edges_needed * min_edge_length
        return estimated_length <= self.budget

    # def create_children(self, branched_edge):
    #     u, v = branched_edge

    #     normalized_edge = tuple(sorted((u, v)))
    #     new_branched_edges = self.branched_edges | {normalized_edge}

    #     # Combine active cuts and new cuts from the parent
    #     all_cuts = self.active_cuts + [(set(tuple(sorted((x, y))) for x, y in cut), rhs) for cut, rhs in self.new_cuts]
    #     current_multipliers = self.lagrangian_solver.best_cut_multipliers_for_best_bound.copy()
    #     # Assign default multipliers for new cuts
    #     for cut_idx in range(len(self.active_cuts), len(all_cuts)):
    #         current_multipliers[cut_idx] = 0.001  # Default multiplier for new cuts

    #     # Fixed child: inherit all cuts (active + new) and multipliers
    #     fixed_child = MSTNode(
    #         self.edges,
    #         self.num_nodes,
    #         self.budget,
    #         self.fixed_edges | {normalized_edge},
    #         self.excluded_edges,
    #         new_branched_edges,
    #         initial_lambda=self.lagrangian_solver.best_lambda if self.inherit_lambda else 0.05,
    #         inherit_lambda=self.inherit_lambda,
    #         branching_rule=self.branching_rule,
    #         step_size=self.lagrangian_solver.step_size if self.inherit_step_size else 0.00001,
    #         inherit_step_size=self.inherit_step_size,
    #         use_cover_cuts=self.use_cover_cuts,
    #         cut_frequency=self.cut_frequency,
    #         node_cut_frequency=self.node_cut_frequency,
    #         parent_cover_cuts=all_cuts,
    #         parent_cover_multipliers=current_multipliers,
    #         use_bisection=self.use_bisection,
    #         max_iter=10,
    #         verbose=self.verbose,
    #         depth=self.depth + 1,
    #         pseudocosts_up=self.pseudocosts_up,
    #         pseudocosts_down=self.pseudocosts_down,
    #         counts_up=self.counts_up,
    #         counts_down=self.counts_down,
    #         reliability_eta=self.reliability_eta,
    #         lookahead_lambda=self.lookahead_lambda,
    #         # partial_iters=self.partial_iters
            
    #     )

    #     # Excluded child: filter out trivially satisfied cuts
    #     excluded_cuts = []
    #     for cut, rhs in all_cuts:
    #         remaining_cover = cut - {normalized_edge}
    #         if len(remaining_cover) <= rhs:
    #             excluded_cuts.append((cut, rhs))

    #     # kept_cuts = [c for c in all_cuts if c not in excluded_cuts]
    #     # kept_indices = [i for i, c in enumerate(all_cuts) if c in kept_cuts]
    #     # kept_multipliers = {i: current_multipliers[i] for i in kept_indices if i in current_multipliers}
    #     kept_cuts = [c for c in all_cuts if c not in excluded_cuts]
    #     kept_indices = [i for i, c in enumerate(all_cuts) if c in kept_cuts]
    #     kept_multipliers = {}
    #     for new_i, old_i in enumerate(kept_indices):
    #         if old_i in current_multipliers:
    #             kept_multipliers[new_i] = current_multipliers[old_i]

    #     excluded_child = MSTNode(
    #         self.edges,
    #         self.num_nodes,
    #         self.budget,
    #         self.fixed_edges,
    #         self.excluded_edges | {normalized_edge},
    #         new_branched_edges,
    #         initial_lambda=self.lagrangian_solver.best_lambda if self.inherit_lambda else 0.05,
    #         inherit_lambda=self.inherit_lambda,
    #         branching_rule=self.branching_rule,
    #         step_size=self.lagrangian_solver.step_size if self.inherit_step_size else 0.00001,
    #         inherit_step_size=self.inherit_step_size,
    #         use_cover_cuts=self.use_cover_cuts,
    #         cut_frequency=self.cut_frequency,
    #         node_cut_frequency=self.node_cut_frequency,
    #         parent_cover_cuts=kept_cuts,
    #         parent_cover_multipliers=kept_multipliers,
    #         use_bisection=self.use_bisection,
    #         max_iter=10,
    #         verbose=self.verbose,
    #         depth=self.depth + 1,
    #         pseudocosts_up=self.pseudocosts_up,
    #         pseudocosts_down=self.pseudocosts_down,
    #         counts_up=self.counts_up,
    #         counts_down=self.counts_down,
    #         reliability_eta=self.reliability_eta,
    #         lookahead_lambda=self.lookahead_lambda,
    #         # partial_iters=self.partial_iters
    #     )

    #     if self.verbose:
    #         print(f"Created fixed child: fixed_edges={len(fixed_child.fixed_edges)}, cuts={len(fixed_child.active_cuts)}")
    #         print(f"Created excluded child: excluded_edges={len(excluded_child.excluded_edges)}, cuts={len(excluded_child.active_cuts)}")

    #     return [fixed_child, excluded_child]
    # def create_children(self, branched_edge):
    #     u, v = branched_edge
    #     normalized_edge = tuple(sorted((u, v)))
    #     new_branched_edges = self.branched_edges | {normalized_edge}

    #     # --- NEW: quick infeasibility guard for the fixed child ---
    #     F = self.fixed_edges | {normalized_edge}
    #     must_prune_fixed = False
    #     # All cuts we plan to pass to the child (same as your original 'all_cuts')
    #     all_cuts = self.active_cuts + [
    #         (set(tuple(sorted((x, y))) for x, y in cut), rhs) for cut, rhs in self.new_cuts
    #     ]
    #     for (cut_set, rhs) in all_cuts:
    #         # If too many fixed edges already inside this cover -> impossible
    #         if len(cut_set & F) > rhs:
    #             must_prune_fixed = True
    #             break
    #     # ----------------------------------------------------------

    #     current_multipliers = self.lagrangian_solver.best_cut_multipliers_for_best_bound.copy()
    #     for cut_idx in range(len(self.active_cuts), len(all_cuts)):
    #         current_multipliers[cut_idx] = 0.001  # keep your default

    #     # Fixed child
    #     fixed_child = None
    #     if not must_prune_fixed:
    #         fixed_child = MSTNode(
    #             self.edges, self.num_nodes, self.budget,
    #             self.fixed_edges | {normalized_edge}, self.excluded_edges, new_branched_edges,
    #             initial_lambda=self.lagrangian_solver.best_lambda if self.inherit_lambda else 0.05,
    #             inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
    #             step_size=self.lagrangian_solver.step_size if self.inherit_step_size else 0.00001,
    #             inherit_step_size=self.inherit_step_size, use_cover_cuts=self.use_cover_cuts,
    #             cut_frequency=self.cut_frequency, node_cut_frequency=self.node_cut_frequency,
    #             parent_cover_cuts=all_cuts, parent_cover_multipliers=current_multipliers,
    #             use_bisection=self.use_bisection, max_iter=10, verbose=self.verbose, depth=self.depth + 1,
    #             pseudocosts_up=self.pseudocosts_up, pseudocosts_down=self.pseudocosts_down,
    #             counts_up=self.counts_up, counts_down=self.counts_down,
    #             reliability_eta=self.reliability_eta, lookahead_lambda=self.lookahead_lambda,
    #         )

    #     # Excluded child (your original trivial-cut filter)
    #     excluded_cuts = []
    #     for cut, rhs in all_cuts:
    #         remaining_cover = cut - {normalized_edge}
    #         if len(remaining_cover) <= rhs:
    #             excluded_cuts.append((cut, rhs))

    #     kept_cuts = [c for c in all_cuts if c not in excluded_cuts]
    #     kept_indices = [i for i, c in enumerate(all_cuts) if c in kept_cuts]
    #     kept_multipliers = {}
    #     for new_i, old_i in enumerate(kept_indices):
    #         if old_i in current_multipliers:
    #             kept_multipliers[new_i] = current_multipliers[old_i]

    #     excluded_child = MSTNode(
    #         self.edges, self.num_nodes, self.budget,
    #         self.fixed_edges, self.excluded_edges | {normalized_edge}, new_branched_edges,
    #         initial_lambda=self.lagrangian_solver.best_lambda if self.inherit_lambda else 0.05,
    #         inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
    #         step_size=self.lagrangian_solver.step_size if self.inherit_step_size else 0.00001,
    #         inherit_step_size=self.inherit_step_size, use_cover_cuts=self.use_cover_cuts,
    #         cut_frequency=self.cut_frequency, node_cut_frequency=self.node_cut_frequency,
    #         parent_cover_cuts=kept_cuts, parent_cover_multipliers=kept_multipliers,
    #         use_bisection=self.use_bisection, max_iter=10, verbose=self.verbose, depth=self.depth + 1,
    #         pseudocosts_up=self.pseudocosts_up, pseudocosts_down=self.pseudocosts_down,
    #         counts_up=self.counts_up, counts_down=self.counts_down,
    #         reliability_eta=self.reliability_eta, lookahead_lambda=self.lookahead_lambda,
    #     )

    #     if self.verbose:
    #         if fixed_child is not None:
    #             print(f"Created fixed child: fixed_edges={len(fixed_child.fixed_edges)}, cuts={len(fixed_child.active_cuts)}")
    #         print(f"Created excluded child: excluded_edges={len(excluded_child.excluded_edges)}, cuts={len(excluded_child.active_cuts)}")

    #     try:
    #         # Clear parent's heavy data structures
    #         self.lagrangian_solver.primal_solutions = []
    #         self.lagrangian_solver.fractional_solutions = []
    #         self.lagrangian_solver.subgradients = []
    #         self.lagrangian_solver.step_sizes = []
    #         self.lagrangian_solver.multipliers = []
    #         # Reset cache to minimal size
    #         from lagrangianrelaxation import LRUCache
    #         self.lagrangian_solver.mst_cache = LRUCache(capacity=5)
    #     except:
    #         pass

    #     return [fixed_child, excluded_child]
    


    # def create_children(self, branched_edge):
    #     import numpy as np
    #     u, v = branched_edge
    #     normalized_edge = tuple(sorted((u, v)))
    #     new_branched_edges = self.branched_edges | {normalized_edge}

    #     # Merge inherited + this node's new cuts
    #     all_cuts = list(self.active_cuts) + [
    #         ({tuple(sorted((x, y))) for x, y in cut}, int(rhs))
    #         for (cut, rhs) in getattr(self, "new_cuts", [])
    #     ]

    #     # μ for inherited cuts from parent's best snapshot; new cuts start at 0
    #     parent_mu = getattr(self.lagrangian_solver, "best_cut_multipliers_for_best_bound", {}) or {}
    #     current_multipliers = dict(parent_mu)
    #     first_new = len(self.active_cuts)
    #     for cut_idx in range(first_new, len(all_cuts)):
    #         current_multipliers[cut_idx] = 0.0

    #     # quick prune for fixed child
    #     F_fixed = set(self.fixed_edges) | {normalized_edge}
    #     must_prune_fixed = any((len(cut_set & F_fixed) > rhs) for (cut_set, rhs) in all_cuts)

    #     edge_indices = self.lagrangian_solver.edge_indices

    #     # --- build parent's MST mask once (for activity scoring) ---
    #     nE = len(self.lagrangian_solver.edge_weights)
    #     last_parent_tree_mask = np.zeros(nE, dtype=bool)
    #     for e in self.mst_edges:
    #         j = edge_indices.get(e)
    #         if j is not None:
    #             last_parent_tree_mask[j] = True

    #     # ---- FIXED child ----
    #     fixed_child = None
    #     if not must_prune_fixed:
    #         kept_cuts_fixed, kept_mu_fixed = _child_filter_and_remap(
    #             all_cuts,
    #             current_multipliers,
    #             edge_indices,
    #             F_fixed,
    #             set(self.excluded_edges),
    #             last_parent_tree_mask,      # NEW
    #             K_keep=8, mu_min=1e-3, mu_cap=1e3, damp=0.5
    #         )
    #         fixed_child = MSTNode(
    #             self.edges, self.num_nodes, self.budget,
    #             F_fixed, set(self.excluded_edges), new_branched_edges,
    #             initial_lambda=self.lagrangian_solver.best_lambda if self.inherit_lambda else 0.05,
    #             inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
    #             step_size=self.lagrangian_solver.step_size if self.inherit_step_size else 0.00001,
    #             inherit_step_size=self.inherit_step_size, use_cover_cuts=self.use_cover_cuts,
    #             cut_frequency=self.cut_frequency, node_cut_frequency=self.node_cut_frequency,
    #             parent_cover_cuts=kept_cuts_fixed,
    #             parent_cover_multipliers=kept_mu_fixed,
    #             use_bisection=self.use_bisection, max_iter=self.lagrangian_solver.max_iter,
    #             verbose=self.verbose, depth=self.depth + 1,
    #             pseudocosts_up=self.pseudocosts_up, pseudocosts_down=self.pseudocosts_down,
    #             counts_up=self.counts_up, counts_down=self.counts_down,
    #             reliability_eta=self.reliability_eta, lookahead_lambda=self.lookahead_lambda
    #         )

    #     # ---- EXCLUDED child ----
    #     F_excluded = set(self.excluded_edges) | {normalized_edge}
    #     kept_cuts_excl, kept_mu_excl = _child_filter_and_remap(
    #         all_cuts,
    #         current_multipliers,
    #         edge_indices,
    #         set(self.fixed_edges),
    #         F_excluded,
    #         last_parent_tree_mask,          # NEW
    #         K_keep=8, mu_min=1e-3, mu_cap=1e3, damp=0.5
    #     )
    #     excluded_child = MSTNode(
    #         self.edges, self.num_nodes, self.budget,
    #         set(self.fixed_edges), F_excluded, new_branched_edges,
    #         initial_lambda=self.lagrangian_solver.best_lambda if self.inherit_lambda else 0.05,
    #         inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
    #         step_size=self.lagrangian_solver.step_size if self.inherit_step_size else 0.00001,
    #         inherit_step_size=self.inherit_step_size, use_cover_cuts=self.use_cover_cuts,
    #         cut_frequency=self.cut_frequency, node_cut_frequency=self.node_cut_frequency,
    #         parent_cover_cuts=kept_cuts_excl,
    #         parent_cover_multipliers=kept_mu_excl,
    #         use_bisection=self.use_bisection, max_iter=self.lagrangian_solver.max_iter,
    #         verbose=self.verbose, depth=self.depth + 1,
    #         pseudocosts_up=self.pseudocosts_up, pseudocosts_down=self.pseudocosts_down,
    #         counts_up=self.counts_up, counts_down=self.counts_down,
    #         reliability_eta=self.reliability_eta, lookahead_lambda=self.lookahead_lambda
    #     )

    #     return fixed_child, excluded_child

    # def create_children(self, branched_edge):
    #     """
    #     Branch on `branched_edge` and build children with cover cuts inherited correctly.

    #     Correctness rules per cover cut (S, rhs):
    #     - S_fixed = S ∩ Fixed_child
    #     - S_excl  = S ∩ Excluded_child
    #     - S_free  = S \ (Fixed_child ∪ Excluded_child)
    #     - rhs'    = rhs - |S_fixed|
    #     - If rhs' < 0          -> child infeasible (prune)
    #     - If |S_free| <= rhs'  -> cut redundant (drop)
    #     - Else pass (S_free, rhs') to child
    #     Multipliers are remapped 1:1 to the new dense indices without damping/capping.
    #     """
    #     import numpy as np

    #     # --- Normalize the branched edge ---
    #     u, v = branched_edge
    #     normalized_edge = (u, v) if u <= v else (v, u)
    #     new_branched_edges = self.branched_edges | {normalized_edge}

    #     # --- Merge inherited cuts + this node's new cuts; normalize edges inside each cut ---
    #     def _norm_edge(e):
    #         a, b = e
    #         return (a, b) if a <= b else (b, a)

    #     all_cuts = list(self.active_cuts)
    #     for (cut, rhs) in getattr(self, "new_cuts", []):
    #         all_cuts.append(({_norm_edge((x, y)) for (x, y) in cut}, int(rhs)))

    #     # Multipliers: start from parent's best snapshot; new cuts get μ=0.0
    #     parent_mu = (getattr(self.lagrangian_solver, "best_cut_multipliers_for_best_bound", {}) or {}).copy()
    #     first_new = len(self.active_cuts)
    #     for cut_idx in range(first_new, len(all_cuts)):
    #         parent_mu[cut_idx] = 0.0

    #     edge_indices = self.lagrangian_solver.edge_indices  # map (u,v)->idx

    #     # --- helper: project & remap cuts for a particular child; prune if infeasible ---
    #     def _project_and_remap_for_child(fixed_child_edges, excluded_child_edges):
    #         infeasible = False
    #         # key: frozenset(S_free) -> (rhs_prime, mu_agg)
    #         # For duplicates we keep the *strongest* (smallest rhs') and carry max |μ|
    #         projected = {}

    #         for old_i, (cut_set, rhs) in enumerate(all_cuts):
    #             # restrict to known edges
    #             S = {e for e in cut_set if e in edge_indices}

    #             # compute partitions
    #             S_fixed = S & fixed_child_edges
    #             # excluded edges are simply removed from S_free; they don't affect rhs'
    #             S_free  = S - fixed_child_edges - excluded_child_edges

    #             rhs_prime = int(rhs) - len(S_fixed)
    #             if rhs_prime < 0:
    #                 infeasible = True
    #                 break  # this child is infeasible due to this cut

    #             # redundancy check: if |S_free| <= rhs', inequality is non-binding
    #             if len(S_free) <= rhs_prime:
    #                 continue

    #             key = frozenset(S_free)
    #             mu_old = float(parent_mu.get(old_i, 0.0))

    #             prev = projected.get(key)
    #             if prev is None:
    #                 projected[key] = (rhs_prime, mu_old)
    #             else:
    #                 prev_rhs, prev_mu = prev
    #                 # keep the strongest inequality: minimal rhs'
    #                 if rhs_prime < prev_rhs:
    #                     projected[key] = (rhs_prime, max(abs(mu_old), abs(prev_mu)) * (1.0 if mu_old >= 0 else -1.0))
    #                 elif rhs_prime == prev_rhs:
    #                     # tie on rhs': keep the larger |μ| (sign from the one with larger magnitude)
    #                     chosen_mu = mu_old if abs(mu_old) >= abs(prev_mu) else prev_mu
    #                     projected[key] = (rhs_prime, chosen_mu)

    #         if infeasible:
    #             return None, None, True

    #         # build dense lists with deterministic order (optional but nice for reproducibility)
    #         # sort by (len(S_free) descending, rhs' ascending, then lexicographic edge list)
    #         def _edge_tuple_sort_key(sfree):
    #             return tuple(sorted(sfree))

    #         ordered = sorted(
    #             projected.items(),
    #             key=lambda kv: (-len(kv[0]), kv[1][0], _edge_tuple_sort_key(kv[0]))
    #         )

    #         kept_cuts = []
    #         kept_mu = {}
    #         for new_idx, (sfree_key, (rhs_prime, mu_val)) in enumerate(ordered):
    #             kept_cuts.append((set(sfree_key), int(rhs_prime)))
    #             kept_mu[new_idx] = float(mu_val)

    #         return kept_cuts, kept_mu, False

    #     # ---- FIXED child ----
    #     F_fixed = set(self.fixed_edges) | {normalized_edge}
    #     kept_cuts_fixed, kept_mu_fixed, prune_fixed = _project_and_remap_for_child(
    #         fixed_child_edges=F_fixed,
    #         excluded_child_edges=set(self.excluded_edges),
    #     )
    #     fixed_child = None
    #     if not prune_fixed:
    #         fixed_child = MSTNode(
    #             self.edges, self.num_nodes, self.budget,
    #             F_fixed, set(self.excluded_edges), new_branched_edges,
    #             initial_lambda=self.lagrangian_solver.best_lambda if self.inherit_lambda else 0.05,
    #             inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
    #             step_size=self.lagrangian_solver.step_size if self.inherit_step_size else 0.00001,
    #             inherit_step_size=self.inherit_step_size, use_cover_cuts=self.use_cover_cuts,
    #             cut_frequency=self.cut_frequency, node_cut_frequency=self.node_cut_frequency,
    #             parent_cover_cuts=kept_cuts_fixed,
    #             parent_cover_multipliers=kept_mu_fixed,
    #             use_bisection=self.use_bisection, max_iter=self.lagrangian_solver.max_iter,
    #             verbose=self.verbose, depth=self.depth + 1,
    #             pseudocosts_up=self.pseudocosts_up, pseudocosts_down=self.pseudocosts_down,
    #             counts_up=self.counts_up, counts_down=self.counts_down,
    #             reliability_eta=self.reliability_eta, lookahead_lambda=self.lookahead_lambda
    #         )

    #     # ---- EXCLUDED child ----
    #     F_excluded = set(self.excluded_edges) | {normalized_edge}
    #     kept_cuts_excl, kept_mu_excl, _ = _project_and_remap_for_child(
    #         fixed_child_edges=set(self.fixed_edges),
    #         excluded_child_edges=F_excluded,
    #     )
    #     excluded_child = MSTNode(
    #         self.edges, self.num_nodes, self.budget,
    #         set(self.fixed_edges), F_excluded, new_branched_edges,
    #         initial_lambda=self.lagrangian_solver.best_lambda if self.inherit_lambda else 0.05,
    #         inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
    #         step_size=self.lagrangian_solver.step_size if self.inherit_step_size else 0.00001,
    #         inherit_step_size=self.inherit_step_size, use_cover_cuts=self.use_cover_cuts,
    #         cut_frequency=self.cut_frequency, node_cut_frequency=self.node_cut_frequency,
    #         parent_cover_cuts=kept_cuts_excl,
    #         parent_cover_multipliers=kept_mu_excl,
    #         use_bisection=self.use_bisection, max_iter=self.lagrangian_solver.max_iter,
    #         verbose=self.verbose, depth=self.depth + 1,
    #         pseudocosts_up=self.pseudocosts_up, pseudocosts_down=self.pseudocosts_down,
    #         counts_up=self.counts_up, counts_down=self.counts_down,
    #         reliability_eta=self.reliability_eta, lookahead_lambda=self.lookahead_lambda
    #     )

    #     return fixed_child, excluded_child
    # def create_children(self, branched_edge):
    #     """
    #     Branch on `branched_edge` and build children with cover cuts inherited correctly.

    #     Correctness per cover cut (S, rhs):
    #     - S_fixed = S ∩ Fixed_child
    #     - S_excl  = S ∩ Excluded_child
    #     - S_free  = S \ (Fixed_child ∪ Excluded_child)
    #     - rhs'    = rhs - |S_fixed|
    #     - If rhs' < 0          -> child infeasible (prune)
    #     - If |S_free| <= rhs'  -> cut redundant (drop)
    #     - Else pass (S_free, rhs') to child
    #     Multipliers are remapped 1:1 (no damping/caps).
    #     """
    #     import numpy as np

    #     # --- normalize branched edge ---
    #     u, v = branched_edge
    #     normalized_edge = (u, v) if u <= v else (v, u)
    #     new_branched_edges = self.branched_edges | {normalized_edge}

    #     # --- robustly merge & normalize active_cuts + new_cuts (accept pairs or edge indices) ---
    #     solver = self.lagrangian_solver
    #     edge_indices = solver.edge_indices                    # {(u,v): idx}
    #     known_edges = set(edge_indices.keys())
    #     idx_to_edge = getattr(solver, "idx_to_edge", None)
    #     if idx_to_edge is None:
    #         idx_to_edge = {j: e for e, j in edge_indices.items()}
    #         solver.idx_to_edge = idx_to_edge

    #     def _norm_edge(e):
    #         if not (isinstance(e, tuple) and len(e) == 2):
    #             return None
    #         a, b = e
    #         t = (a, b) if a <= b else (b, a)
    #         return t if t in known_edges else None

    #     def _iter_edges_any(cut_like):
    #         # single (u,v)
    #         if isinstance(cut_like, tuple) and len(cut_like) == 2:
    #             e = _norm_edge(cut_like)
    #             if e is not None:
    #                 yield e
    #             return
    #         # single index
    #         if isinstance(cut_like, int):
    #             e_raw = idx_to_edge.get(int(cut_like))
    #             e = _norm_edge(e_raw)
    #             if e is not None:
    #                 yield e
    #             return
    #         # iterable
    #         try:
    #             for item in cut_like:
    #                 if isinstance(item, int):
    #                     e_raw = idx_to_edge.get(int(item))
    #                     e = _norm_edge(e_raw)
    #                 elif isinstance(item, tuple) and len(item) == 2:
    #                     e = _norm_edge(item)
    #                 elif isinstance(item, (list, set, frozenset)) and len(item) == 2:
    #                     a, b = tuple(item)
    #                     e = _norm_edge((a, b))
    #                 else:
    #                     e = None
    #                 if e is not None:
    #                     yield e
    #         except TypeError:
    #             return

    #     def _norm_pair(pair):
    #         cut_like, rhs_like = pair
    #         return (set(_iter_edges_any(cut_like)), int(rhs_like))

    #     all_cuts = []
    #     for p in (self.active_cuts or []):
    #         all_cuts.append(_norm_pair(p))
    #     for p in (getattr(self, "new_cuts", []) or []):
    #         all_cuts.append(_norm_pair(p))

    #     # Multipliers: parent snapshot + 0 for newly added cuts
    #     parent_mu = getattr(solver, "best_cut_multipliers_for_best_bound", {}) or {}
    #     current_multipliers = dict(parent_mu)
    #     first_new = len(self.active_cuts or [])
    #     for cut_idx in range(first_new, len(all_cuts)):
    #         current_multipliers[cut_idx] = 0.0

    #     # Quick prune for fixed child
    #     F_fixed = set(self.fixed_edges) | {normalized_edge}
    #     must_prune_fixed = any((len(cut_set & F_fixed) > rhs) for (cut_set, rhs) in all_cuts)

    #     # --- helper: project & remap to a child ---
    #     def _project_and_remap_for_child(fixed_child_edges, excluded_child_edges):
    #         infeasible = False
    #         # key: frozenset(S_free) -> (rhs', μ)  (if duplicates after projection, keep strongest = smallest rhs')
    #         projected = {}

    #         for old_i, (S, rhs) in enumerate(all_cuts):
    #             S_known = {e for e in S if e in known_edges}
    #             S_fixed = S_known & fixed_child_edges
    #             S_free  = S_known - fixed_child_edges - excluded_child_edges

    #             rhs_prime = int(rhs) - len(S_fixed)
    #             if rhs_prime < 0:
    #                 infeasible = True
    #                 break
    #             if len(S_free) <= rhs_prime:
    #                 continue

    #             key = frozenset(S_free)
    #             mu_old = float(current_multipliers.get(old_i, 0.0))
    #             prev = projected.get(key)
    #             if prev is None or rhs_prime < prev[0] or (rhs_prime == prev[0] and abs(mu_old) > abs(prev[1])):
    #                 projected[key] = (rhs_prime, mu_old)

    #         if infeasible:
    #             return None, None, True

    #         # deterministic ordering
    #         def _edge_tuple_sort_key(sfree):
    #             return tuple(sorted(sfree))

    #         ordered = sorted(projected.items(),
    #                         key=lambda kv: (-len(kv[0]), kv[1][0], _edge_tuple_sort_key(kv[0])))

    #         kept_cuts, kept_mu = [], {}
    #         for new_idx, (sfree_key, (rhs_prime, mu_val)) in enumerate(ordered):
    #             kept_cuts.append((set(sfree_key), int(rhs_prime)))
    #             kept_mu[new_idx] = float(mu_val)

    #         return kept_cuts, kept_mu, False

    #     # ---- children ----
    #     fixed_child = None
    #     if not must_prune_fixed:
    #         kept_cuts_fixed, kept_mu_fixed, prune_fixed = _project_and_remap_for_child(
    #             fixed_child_edges=F_fixed,
    #             excluded_child_edges=set(self.excluded_edges),
    #         )
    #         if not prune_fixed:
    #             fixed_child = MSTNode(
    #                 self.edges, self.num_nodes, self.budget,
    #                 F_fixed, set(self.excluded_edges), new_branched_edges,
    #                 initial_lambda=solver.best_lambda if self.inherit_lambda else 0.05,
    #                 inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
    #                 step_size=solver.step_size if self.inherit_step_size else 0.00001,
    #                 inherit_step_size=self.inherit_step_size, use_cover_cuts=self.use_cover_cuts,
    #                 cut_frequency=self.cut_frequency, node_cut_frequency=self.node_cut_frequency,
    #                 parent_cover_cuts=kept_cuts_fixed,
    #                 parent_cover_multipliers=kept_mu_fixed,
    #                 use_bisection=self.use_bisection, max_iter=solver.max_iter,
    #                 verbose=self.verbose, depth=self.depth + 1,
    #                 pseudocosts_up=self.pseudocosts_up, pseudocosts_down=self.pseudocosts_down,
    #                 counts_up=self.counts_up, counts_down=self.counts_down,
    #                 reliability_eta=self.reliability_eta, lookahead_lambda=self.lookahead_lambda
    #             )

    #     F_excluded = set(self.excluded_edges) | {normalized_edge}
    #     kept_cuts_excl, kept_mu_excl, _ = _project_and_remap_for_child(
    #         fixed_child_edges=set(self.fixed_edges),
    #         excluded_child_edges=F_excluded,
    #     )
    #     excluded_child = MSTNode(
    #         self.edges, self.num_nodes, self.budget,
    #         set(self.fixed_edges), F_excluded, new_branched_edges,
    #         initial_lambda=solver.best_lambda if self.inherit_lambda else 0.05,
    #         inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
    #         step_size=solver.step_size if self.inherit_step_size else 0.00001,
    #         inherit_step_size=self.inherit_step_size, use_cover_cuts=self.use_cover_cuts,
    #         cut_frequency=self.cut_frequency, node_cut_frequency=self.node_cut_frequency,
    #         parent_cover_cuts=kept_cuts_excl,
    #         parent_cover_multipliers=kept_mu_excl,
    #         use_bisection=self.use_bisection, max_iter=solver.max_iter,
    #         verbose=self.verbose, depth=self.depth + 1,
    #         pseudocosts_up=self.pseudocosts_up, pseudocosts_down=self.pseudocosts_down,
    #         counts_up=self.counts_up, counts_down=self.counts_down,
    #         reliability_eta=self.reliability_eta, lookahead_lambda=self.lookahead_lambda
    #     )

    #     return fixed_child, excluded_child

    def create_children(self, branched_edge):
        """
        Branch on `branched_edge` and build children with cover cuts inherited correctly.

        Correctness per cover cut (S, rhs):
        - S_fixed = S ∩ Fixed_child
        - S_excl  = S ∩ Excluded_child
        - S_free  = S \ (Fixed_child ∪ Excluded_child)
        - rhs'    = rhs - |S_fixed|
        - If rhs' < 0          -> child infeasible (prune)
        - If |S_free| <= rhs'  -> cut redundant (drop)
        - Else pass (S_free, rhs') to child

        Multipliers are remapped 1:1 (no damping/caps).
        We additionally LIMIT the number of cuts passed to each child
        to at most `max_child_cuts`, keeping the strongest ones.
        """
        import numpy as np  # (note: currently unused, safe to remove if you like)

        # how many cuts to keep per child (you can tune this)
        max_child_cuts = getattr(self, "max_child_cuts", 25)

        # --- normalize branched edge ---
        u, v = branched_edge
        normalized_edge = (u, v) if u <= v else (v, u)
        new_branched_edges = self.branched_edges | {normalized_edge}

        # --- robustly merge & normalize active_cuts + new_cuts (accept pairs or edge indices) ---
        solver = self.lagrangian_solver
        edge_indices = solver.edge_indices                    # {(u,v): idx}
        known_edges = set(edge_indices.keys())
        idx_to_edge = getattr(solver, "idx_to_edge", None)
        if idx_to_edge is None:
            idx_to_edge = {j: e for e, j in edge_indices.items()}
            solver.idx_to_edge = idx_to_edge

        def _norm_edge(e):
            if not (isinstance(e, tuple) and len(e) == 2):
                return None
            a, b = e
            t = (a, b) if a <= b else (b, a)
            return t if t in known_edges else None

        def _iter_edges_any(cut_like):
            # single (u,v)
            if isinstance(cut_like, tuple) and len(cut_like) == 2:
                e = _norm_edge(cut_like)
                if e is not None:
                    yield e
                return
            # single index
            if isinstance(cut_like, int):
                e_raw = idx_to_edge.get(int(cut_like))
                e = _norm_edge(e_raw)
                if e is not None:
                    yield e
                return
            # iterable
            try:
                for item in cut_like:
                    if isinstance(item, int):
                        e_raw = idx_to_edge.get(int(item))
                        e = _norm_edge(e_raw)
                    elif isinstance(item, tuple) and len(item) == 2:
                        e = _norm_edge(item)
                    elif isinstance(item, (list, set, frozenset)) and len(item) == 2:
                        a, b = tuple(item)
                        e = _norm_edge((a, b))
                    else:
                        e = None
                    if e is not None:
                        yield e
            except TypeError:
                return

        def _norm_pair(pair):
            cut_like, rhs_like = pair
            return (set(_iter_edges_any(cut_like)), int(rhs_like))

        all_cuts = []
        for p in (self.active_cuts or []):
            all_cuts.append(_norm_pair(p))
        for p in (getattr(self, "new_cuts", []) or []):
            all_cuts.append(_norm_pair(p))

        # Multipliers: parent snapshot + 0 for newly added cuts
        parent_mu = getattr(solver, "best_cut_multipliers_for_best_bound", {}) or {}
        current_multipliers = dict(parent_mu)
        first_new = len(self.active_cuts or [])
        for cut_idx in range(first_new, len(all_cuts)):
            current_multipliers[cut_idx] = 0.0

        # Quick prune for fixed child
        F_fixed = set(self.fixed_edges) | {normalized_edge}
        must_prune_fixed = any((len(cut_set & F_fixed) > rhs) for (cut_set, rhs) in all_cuts)

        # --- helper: project & remap to a child ---
        def _project_and_remap_for_child(fixed_child_edges, excluded_child_edges):
            infeasible = False
            # key: frozenset(S_free) -> (rhs', μ)  (if duplicates after projection, keep strongest = smallest rhs')
            projected = {}

            for old_i, (S, rhs) in enumerate(all_cuts):
                S_known = {e for e in S if e in known_edges}
                S_fixed = S_known & fixed_child_edges
                S_free  = S_known - fixed_child_edges - excluded_child_edges

                rhs_prime = int(rhs) - len(S_fixed)
                if rhs_prime < 0:
                    infeasible = True
                    break
                if len(S_free) <= rhs_prime:
                    continue

                key = frozenset(S_free)
                mu_old = float(current_multipliers.get(old_i, 0.0))
                prev = projected.get(key)
                if prev is None or rhs_prime < prev[0] or (rhs_prime == prev[0] and abs(mu_old) > abs(prev[1])):
                    projected[key] = (rhs_prime, mu_old)

            if infeasible:
                return None, None, True

            # deterministic ordering: you already sort by (-|S_free|, rhs', edges)
            def _edge_tuple_sort_key(sfree):
                return tuple(sorted(sfree))

            ordered = sorted(
                projected.items(),
                key=lambda kv: (-len(kv[0]), kv[1][0], _edge_tuple_sort_key(kv[0]))
            )

            # *** limit to at most max_child_cuts strongest cuts ***
            if len(ordered) > max_child_cuts:
                ordered = ordered[:max_child_cuts]

            kept_cuts, kept_mu = [], {}
            for new_idx, (sfree_key, (rhs_prime, mu_val)) in enumerate(ordered):
                kept_cuts.append((set(sfree_key), int(rhs_prime)))
                kept_mu[new_idx] = float(mu_val)

            return kept_cuts, kept_mu, False

        # ---- children ----
        fixed_child = None
        if not must_prune_fixed:
            kept_cuts_fixed, kept_mu_fixed, prune_fixed = _project_and_remap_for_child(
                fixed_child_edges=F_fixed,
                excluded_child_edges=set(self.excluded_edges),
            )
            if not prune_fixed:
                fixed_child = MSTNode(
                    self.edges, self.num_nodes, self.budget,
                    F_fixed, set(self.excluded_edges), new_branched_edges,
                    initial_lambda=solver.best_lambda if self.inherit_lambda else 0.05,
                    inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
                    step_size=solver.step_size if self.inherit_step_size else 0.00001,
                    inherit_step_size=self.inherit_step_size, use_cover_cuts=self.use_cover_cuts,
                    cut_frequency=self.cut_frequency, node_cut_frequency=self.node_cut_frequency,
                    parent_cover_cuts=kept_cuts_fixed,
                    parent_cover_multipliers=kept_mu_fixed,
                    use_bisection=self.use_bisection, max_iter=solver.max_iter,
                    verbose=self.verbose, depth=self.depth + 1,
                    pseudocosts_up=self.pseudocosts_up, pseudocosts_down=self.pseudocosts_down,
                    counts_up=self.counts_up, counts_down=self.counts_down,
                    reliability_eta=self.reliability_eta, lookahead_lambda=self.lookahead_lambda
                )

        F_excluded = set(self.excluded_edges) | {normalized_edge}
        kept_cuts_excl, kept_mu_excl, _ = _project_and_remap_for_child(
            fixed_child_edges=set(self.fixed_edges),
            excluded_child_edges=F_excluded,
        )
        excluded_child = MSTNode(
            self.edges, self.num_nodes, self.budget,
            set(self.fixed_edges), F_excluded, new_branched_edges,
            initial_lambda=solver.best_lambda if self.inherit_lambda else 0.05,
            inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
            step_size=solver.step_size if self.inherit_step_size else 0.00001,
            inherit_step_size=self.inherit_step_size, use_cover_cuts=self.use_cover_cuts,
            cut_frequency=self.cut_frequency, node_cut_frequency=self.node_cut_frequency,
            parent_cover_cuts=kept_cuts_excl,
            parent_cover_multipliers=kept_mu_excl,
            use_bisection=self.use_bisection, max_iter=solver.max_iter,
            verbose=self.verbose, depth=self.depth + 1,
            pseudocosts_up=self.pseudocosts_up, pseudocosts_down=self.pseudocosts_down,
            counts_up=self.counts_up, counts_down=self.counts_down,
            reliability_eta=self.reliability_eta, lookahead_lambda=self.lookahead_lambda
        )

        return fixed_child, excluded_child




    

    def is_feasible(self):
        real_weight, real_length = self.lagrangian_solver.compute_real_weight_length()
        if real_length > self.budget:
            return False, "MST length exceeds budget"

        mst_nodes = set()
        for u, v in self.mst_edges:
            mst_nodes.add(u)
            mst_nodes.add(v)

        if len(mst_nodes) < self.num_nodes:
            return False, "MST does not include all nodes"

        mst_graph = nx.Graph(self.mst_edges)
        if not nx.is_connected(mst_graph):
            return False, "MST is not connected"

        return True, "MST is feasible"

    def compute_upper_bound(self):
        real_weight, _ = self.lagrangian_solver.compute_real_weight_length()
        return real_weight

    def get_branching_candidates(self):


        if self.branching_rule in ["strong_branching", "strong_branching_all", "sb_fractional"]:
            # Select candidate edges based on branching rule
            if self.branching_rule == "strong_branching_all":
                candidate_edges = [
                    (u, v) for u, v, _, _ in self.edges
                    if (u, v) not in self.fixed_edges and
                    (u, v) not in self.excluded_edges and
                    (u, v) not in self.branched_edges
                ]
            elif self.branching_rule == "strong_branching":
                mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                candidate_edges = [
                    e for e in mst_edges
                    if e not in self.fixed_edges and
                    e not in self.excluded_edges and
                    e not in self.branched_edges

                ]
            else:  # sb_fractional
                shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
                # shor_primal_solution = self.lagrangian_solver.compute_dantzig_wolfe_solution(self)
                normalized_edge_weights = shor_primal_solution
                tolerance = 1e-6
                candidate_edges = [
                    e for e in normalized_edge_weights
                    if e not in self.fixed_edges and
                    e not in self.excluded_edges 
                    and
                    e not in self.branched_edges and
                    normalized_edge_weights[e] > tolerance and
                    normalized_edge_weights[e] < 1.0 - tolerance
                ]
                if not candidate_edges:
                    mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                    candidate_edges = [
                        e for e in mst_edges
                        if e not in self.fixed_edges and
                        e not in self.excluded_edges and
                        e not in self.branched_edges
                    ]
                # else:
                #     normalized_edge_weights = shor_primal_solution
                #     tolerance = 1e-6
                #     candidate_edges = [
                #         e for e in normalized_edge_weights
                #         if e not in self.fixed_edges and
                #         e not in self.excluded_edges 
                #         and
                #         e not in self.branched_edges and
                #         normalized_edge_weights[e] > tolerance and
                #         normalized_edge_weights[e] < 1.0 - tolerance
                #     ]

            if not candidate_edges:
                print("bgj")
                if self.verbose:
                    print(f"No {self.branching_rule} candidates available")
                return None

            if self.verbose:
                print(f"Node {id(self)}: {self.branching_rule} evaluating {len(candidate_edges)} edges: {candidate_edges}")

            # Collect edges that lead to pruning
            edges_to_fix = set()
            edges_to_exclude = set()
            best_edge = None
            best_score = -float('inf')
            scores = []
            for edge in candidate_edges:
                
                score,_,_, fix_infeasible, exclude_infeasible = self.calculate_strong_branching_score(edge)
                scores.append((edge, score))
                if fix_infeasible:
                    edges_to_exclude.add(edge)
                if exclude_infeasible:
                    edges_to_fix.add(edge)
                if not (fix_infeasible or exclude_infeasible):
                    if score > best_score:
                        best_score = score
                        best_edge = edge

            if edges_to_fix or edges_to_exclude:
                # Create a single child with all pruning decisions
                if self.verbose:
                    print(f"Creating single child with fixed edges: {edges_to_fix}, excluded edges: {edges_to_exclude}")
                child = self.create_single_child(edges_to_fix, edges_to_exclude)
                # Return the single child to continue the search
                return ([list(edges_to_fix)[0] if edges_to_fix else list(edges_to_exclude)[0]], child)

            # No pruning edges found, proceed with standard strong branching
            if not best_edge:
                if self.verbose:
                    print(f"No viable branching edge found after scoring")
                return None

            scores.sort(key=lambda x: x[1], reverse=True)
            if self.verbose:
                print(f"Selected best edge {best_edge} with score {best_score}")
            return [best_edge]

        elif self.branching_rule == "strong_branching_sim":
            mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
            candidate_edges = [
                e for e in mst_edges
                if e not in self.fixed_edges and
                e not in self.excluded_edges and
                e not in self.branched_edges
            ]
            if not candidate_edges:
                if self.verbose:
                    print("No strong branching sim candidates available")
                return None

            if self.verbose:
                print(f"Node {id(self)}: Strong branching sim evaluating {len(candidate_edges)} MST edges: {candidate_edges}")

            best_edge = None
            best_score = -float('inf')
            for edge in candidate_edges:
                u, v = edge
                fixed_lower_bound = self.simulate_fix_edge(u, v)
                fix_score = (fixed_lower_bound - self.local_lower_bound) if fixed_lower_bound != float('inf') else float('inf')
                excluded_lower_bound = self.simulate_exclude_edge(u, v)
                exc_score = (excluded_lower_bound - self.local_lower_bound) if excluded_lower_bound != float('inf') else float('inf')

                # score = 0.5 * min(fix_score, exc_score) + 0.5 * max(fix_score, exc_score) if fix_score != float('inf') and exc_score != float('inf') else float('inf')
                score = max(fix_score, 1e-6) * max(exc_score, 1e-6) if fix_score != float('inf') and exc_score != float('inf') else float('inf')
                if self.verbose:
                    print(f"Edge {edge}: Score {score}")
                if score > best_score:
                    best_score = score
                    best_edge = edge

            return [best_edge] if best_edge else None

        elif self.branching_rule == "most_fractional":
            shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
            # shor_primal_solution = self.lagrangian_solver.compute_dantzig_wolfe_solution(self)


            if shor_primal_solution is None:
                candidates = [
                    tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges
                    if tuple(sorted((u, v))) not in self.fixed_edges and
                    tuple(sorted((u, v))) not in self.excluded_edges and
                    tuple(sorted((u, v))) not in self.branched_edges
                ]

                if not candidates:
                    return None
                return [candidates[0]]

            # normalized_edge_weights = shor_primal_solution

            candidates = [
                e for e in shor_primal_solution
                if e not in self.fixed_edges and
                e not in self.excluded_edges and
                e not in self.branched_edges 
                # and
                # abs(normalized_edge_weights[e]) > 1e-6 
                # and
                # abs(normalized_edge_weights[e] - 1.0) > 1e-6
            ]

            branching_scores = []
            for e in candidates:
                w = shor_primal_solution.get(e, 0)
                distance_score = -abs(w - 0.5)
                branching_scores.append((e, distance_score))

            branching_scores.sort(key=lambda x: x[1], reverse=True)

            return [branching_scores[0][0]] if branching_scores else None
        # elif self.branching_rule == "most_fractional":
        #     # Try Dantzig-Wolfe first for better fractional solution quality
        #     shor_primal_solution = self.lagrangian_solver.compute_dantzig_wolfe_solution(self)
            
        #     # Fallback to weighted average if DW fails
        #     if shor_primal_solution is None:
        #         shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()

        #     if shor_primal_solution is None:
        #         # Smart fallback: pick critical MST edge
        #         mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
        #         candidates = [
        #             e for e in mst_edges
        #             if e not in self.fixed_edges
        #             and e not in self.excluded_edges
        #             and e not in self.branched_edges
        #         ]
        #         if not candidates:
        #             return None
                
        #         # Score by criticality: weight/length ratio + cut tightness
        #         scores = []
        #         for e in candidates:
        #             w, l = self.lagrangian_solver.edge_attributes[e]
        #             criticality = w / max(l, 1e-6)
                    
        #             # Check if edge is in tight cuts
        #             in_tight_cut = 0
        #             for cut_set, rhs in self.active_cuts:
        #                 if e in cut_set:
        #                     edges_in_cut = len([c for c in cut_set if c in mst_edges])
        #                     if edges_in_cut >= rhs:  # Tight or violated
        #                         in_tight_cut += 1
                    
        #             score = (criticality, in_tight_cut)
        #             scores.append((e, score))
                
        #         scores.sort(key=lambda x: x[1], reverse=True)
                
        #         if self.verbose:
        #             print(f"Most fractional fallback: selected {scores[0][0]} with score={scores[0][1]}")
                
        #         return [scores[0][0]]

        #     # Adaptive fractionality threshold based on depth
        #     if self.depth < 5:
        #         min_frac = 0.01
        #     elif self.depth < 15:
        #         min_frac = 0.1
        #     else:
        #         min_frac = 0.2

        #     # Filter candidates with fractionality threshold
        #     candidates = [
        #         e for e in shor_primal_solution
        #         if e not in self.fixed_edges
        #         and e not in self.excluded_edges
        #         and e not in self.branched_edges
        #         and min_frac < shor_primal_solution[e] < (1.0 - min_frac)
        #     ]

        #     # If too restrictive, relax the threshold
        #     if not candidates:
        #         candidates = [
        #             e for e in shor_primal_solution
        #             if e not in self.fixed_edges
        #             and e not in self.excluded_edges
        #             and e not in self.branched_edges
        #         ]
            
        #     if not candidates:
        #         return None

        #     # Score edges with multiple criteria
        #     branching_scores = []
        #     for e in candidates:
        #         w = shor_primal_solution.get(e, 0)
                
        #         # Primary: fractionality (distance from 0.5)
        #         fractionality = -abs(w - 0.5)
                
        #         # Secondary: edge impact (weight)
        #         edge_weight = self.lagrangian_solver.edge_attributes[e][0]
                
        #         # Tertiary: stability (how often edge appears in recent primal solutions)
        #         appearance_count = 0
        #         if hasattr(self.lagrangian_solver, 'primal_solutions') and self.lagrangian_solver.primal_solutions:
        #             recent_solutions = self.lagrangian_solver.primal_solutions[-10:]  # Last 10 only
        #             for sol, _ in recent_solutions:
        #                 if e in sol:
        #                     appearance_count += 1
                
        #         # Combined score: (primary, secondary, tertiary)
        #         score = (fractionality, edge_weight / 100.0, appearance_count)
        #         branching_scores.append((e, score))

        #     branching_scores.sort(key=lambda x: x[1], reverse=True)
            
        #     # 10% randomization from top-3 for exploration
        #     if random.random() < 0.1 and len(branching_scores) >= 3:
        #         selected = random.choice(branching_scores[:3])[0]
        #         if self.verbose:
        #             w = shor_primal_solution[selected]
        #             print(f"Most fractional (random top-3): edge {selected}, value={w:.3f}")
        #     else:
        #         selected = branching_scores[0][0]
        #         if self.verbose:
        #             w = shor_primal_solution[selected]
        #             print(f"Most fractional: edge {selected}, value={w:.3f}, distance from 0.5={abs(w-0.5):.3f}")
            
        #     return [selected]

        
        elif self.branching_rule == "random_fractional":
            shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
            # shor_primal_solution = self.lagrangian_solver.compute_dantzig_wolfe_solution(self)
            if shor_primal_solution is None:
                candidates = [
                    tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges
                    if tuple(sorted((u, v))) not in self.fixed_edges and
                    tuple(sorted((u, v))) not in self.excluded_edges and
                    tuple(sorted((u, v))) not in self.branched_edges
                ]
                if not candidates:
                    return None
                return [candidates[0]]

            # normalized_edge_weights = shor_primal_solution
            candidates = [
                e for e in shor_primal_solution
                if e not in self.fixed_edges and
                e not in self.excluded_edges and
                e not in self.branched_edges 
                # and
                # abs(normalized_edge_weights[e]) > 1e-6 and
                # abs(normalized_edge_weights[e] - 1.0) > 1e-6
            ]

            return candidates if candidates else None
            # return [random.choice(candidates)] if candidates else None
            

        
        elif self.branching_rule == "most_violated":
            candidate_edges = sorted(
                [(u, v, w, l) for u, v, w, l in self.edges if (u, v) not in self.fixed_edges and (u, v) not in self.excluded_edges],
                key=lambda x: x[2] / x[3],
                reverse=True,
            )
            return [(u, v) for u, v, _, _ in candidate_edges] if candidate_edges else None

        elif self.branching_rule == "random_mst":
            candidate_edges = [e for e in self.mst_edges if e not in self.fixed_edges and
                            e not in self.excluded_edges and
                            e not in self.branched_edges]
            return candidate_edges if candidate_edges else None
            # return [random.choice(candidate_edges)] if candidate_edges else None


        elif self.branching_rule == "random_all":
            candidate_edges = [(u, v) for u, v, _, _ in self.edges if (u, v) not in self.fixed_edges and
                            (u, v) not in self.excluded_edges and
                            (u, v) not in self.branched_edges]
            return candidate_edges if candidate_edges else None
        # elif self.branching_rule == "reliability":
        #     edges_to_fix = set()
        #     edges_to_exclude = set()
        #     best_score = float('-inf')
        #     best_edge = None
        #     scores = []

        #     # Get candidate edges - prioritize fractional solution if available
        #     shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
        #     if shor_primal_solution is not None:
        #         # normalized_edge_weights = shor_primal_solution
        #         tolerance = 1e-6
        #         candidate_edges = [
        #             e for e in shor_primal_solution
        #             if e not in self.fixed_edges and
        #             e not in self.excluded_edges and
        #             e not in self.branched_edges and
        #             shor_primal_solution[e] > tolerance and
        #             shor_primal_solution[e] < 1.0 - tolerance
        #         ]
        #         # Sort by most fractional (closest to 0.5)
        #         candidate_edges.sort(key=lambda e: abs(shor_primal_solution[e] - 0.5))
        #     else:
        #         # Fallback to MST edges
        #         mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
        #         candidate_edges = [
        #             e for e in mst_edges
        #             if e not in self.fixed_edges and
        #             e not in self.excluded_edges and
        #             e not in self.branched_edges
        #         ]
            
        #     if not candidate_edges:
        #         return None

        #     # Separate unreliable and reliable candidates
        #     unhistoried = [e for e in candidate_edges if self.counts_up.get(e, 0) < self.reliability_eta or self.counts_down.get(e, 0) < self.reliability_eta]
        #     reliable_candidates = [e for e in candidate_edges if self.counts_up.get(e, 0) >= self.reliability_eta and self.counts_down.get(e, 0) >= self.reliability_eta]
            
        #     # Limit SB evaluations
        #     unhistoried = unhistoried[:self.lookahead_lambda]

        #     for edge in unhistoried:
        #         f = self.get_fractional_value(edge)
        #         count_up, count_down = self.counts_up.get(edge, 0), self.counts_down.get(edge, 0)

        #         sb_score, fix_delta, exc_delta, fix_inf, exc_inf = self.calculate_strong_branching_score(edge)

        #         if fix_inf and exc_inf:
        #             continue  # Skip if both infeasible

        #         # Update pseudocosts with better numerical stability
        #         if not fix_inf and (1 - f) > 1e-6:
        #             new_pc_up = max(0, fix_delta) / (1 - f)  # Ensure non-negative
        #             if count_up == 0:
        #                 self.pseudocosts_up[edge] = new_pc_up
        #             else:
        #                 # Exponentially weighted moving average
        #                 alpha = 0.1
        #                 old_pc = self.pseudocosts_up.get(edge, 0)
        #                 if not math.isnan(old_pc) and not math.isinf(old_pc):
        #                     self.pseudocosts_up[edge] = (1 - alpha) * old_pc + alpha * new_pc_up
        #                 else:
        #                     self.pseudocosts_up[edge] = new_pc_up
        #             self.counts_up[edge] = count_up + 1

        #         if not exc_inf and f > 1e-6:
        #             new_pc_down = max(0, exc_delta) / f  # Ensure non-negative
        #             if count_down == 0:
        #                 self.pseudocosts_down[edge] = new_pc_down
        #             else:
        #                 # Exponentially weighted moving average
        #                 alpha = 0.1
        #                 old_pc = self.pseudocosts_down.get(edge, 0)
        #                 if not math.isnan(old_pc) and not math.isinf(old_pc):
        #                     self.pseudocosts_down[edge] = (1 - alpha) * old_pc + alpha * new_pc_down
        #                 else:
        #                     self.pseudocosts_down[edge] = new_pc_down
        #             self.counts_down[edge] = count_down + 1

        #         if not fix_inf and not exc_inf:
        #             scores.append((sb_score, edge, fix_inf, exc_inf))
        #             if sb_score > best_score:
        #                 best_score = sb_score
        #                 best_edge = edge
        #         else:
        #             if fix_inf:
        #                 edges_to_exclude.add(edge)  # Fixing infeasible: must exclude
        #             if exc_inf:
        #                 edges_to_fix.add(edge)  # Excluding infeasible: must fix

        #     # Evaluate reliable candidates using pseudocosts
        #     for edge in reliable_candidates:
        #         f = self.get_fractional_value(edge)
        #         pc_up = self.pseudocosts_up.get(edge, 1.0)    # Default pseudocost
        #         pc_down = self.pseudocosts_down.get(edge, 1.0) # Default pseudocost
                
        #         delta_up = pc_up * (1 - f)
        #         delta_down = pc_down * f
        #         score = (delta_up * delta_down) ** 0.5  # Geometric mean for balanced exploration
        #         scores.append((score, edge, False, False))

        #     if not scores and not (edges_to_fix or edges_to_exclude):
        #         return None  # No viable branching candidates

        #     if edges_to_fix or edges_to_exclude:
        #         if self.verbose:
        #             print(f"Creating single child with fixed edges: {edges_to_fix}, excluded edges: {edges_to_exclude}")
        #         child = self.create_single_child(edges_to_fix, edges_to_exclude)
        #         return ([list(edges_to_fix)[0] if edges_to_fix else list(edges_to_exclude)[0]], child)
        #     else:
        #         scores.sort(key=lambda x: x[0], reverse=True)
        #         best_score, best_edge, fix_inf, exc_inf = scores[0]

        #         if self.verbose:
        #             print(f"Selected best edge {best_edge} with score {best_score}")

        #     return [best_edge]     
        
        elif self.branching_rule == "reliability":
            # Get fractional solution for prioritization
            shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
            # shor_primal_solution = self.lagrangian_solver.compute_dantzig_wolfe_solution(self)

            
            if shor_primal_solution is not None:
                tolerance = 1e-6
                candidate_edges = [
                    e for e in shor_primal_solution
                    if e not in self.fixed_edges
                    and e not in self.excluded_edges
                    and e not in self.branched_edges
                    and shor_primal_solution[e] > tolerance
                    and shor_primal_solution[e] < 1.0 - tolerance
                ]
                candidate_edges.sort(key=lambda e: abs(shor_primal_solution.get(e, 0.5) - 0.5))
            else:
                mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                candidate_edges = [
                    e for e in mst_edges
                    if e not in self.fixed_edges
                    and e not in self.excluded_edges
                    and e not in self.branched_edges
                ]
            
            if not candidate_edges:
                return None

            # Separate by reliability
            unhistoried = []
            reliable_candidates = []
            for e in candidate_edges:
                if self.counts_up.get(e, 0) < self.reliability_eta or self.counts_down.get(e, 0) < self.reliability_eta:
                    unhistoried.append(e)
                else:
                    reliable_candidates.append(e)
            
            # Adaptive lookahead
            duality_gap = (self.best_upper_bound - self.local_lower_bound 
                        if self.best_upper_bound < float('inf') else float('inf'))
            if self.depth < 5:
                max_sb_evals = self.lookahead_lambda
            elif self.depth < 10 and duality_gap > 0.1 * self.best_upper_bound:
                max_sb_evals = max(2, self.lookahead_lambda - 1)
            else:
                max_sb_evals = max(1, self.lookahead_lambda // 2)
            
            unhistoried = unhistoried[:max_sb_evals]

            edges_to_fix = set()
            edges_to_exclude = set()
            best_score = float('-inf')
            best_edge = None
            scores = []

            # Evaluate unreliable edges with strong branching
            for edge in unhistoried:
                # Better fractional estimation
                if shor_primal_solution is not None and edge in shor_primal_solution:
                    f = shor_primal_solution[edge]
                    f = max(0.01, min(0.99, f))
                else:
                    f = self.get_fractional_value(edge)
                
                count_up, count_down = self.counts_up.get(edge, 0), self.counts_down.get(edge, 0)

                sb_score, fix_delta, exc_delta, fix_inf, exc_inf = self.calculate_strong_branching_score(edge)

                if fix_inf and exc_inf:
                    continue

                # Adaptive learning rate
                if count_up == 0 and count_down == 0:
                    alpha = 0.5
                elif count_up < 3 or count_down < 3:
                    alpha = 0.3
                else:
                    alpha = 0.1

                # Update pseudocosts
                if not fix_inf and (1 - f) > 1e-6:
                    new_pc_up = max(0, fix_delta) / max(1e-9, (1 - f))
                    if count_up == 0:
                        self.pseudocosts_up[edge] = new_pc_up
                    else:
                        old_pc = self.pseudocosts_up.get(edge, 0)
                        if not math.isnan(old_pc) and not math.isinf(old_pc):
                            self.pseudocosts_up[edge] = (1 - alpha) * old_pc + alpha * new_pc_up
                        else:
                            self.pseudocosts_up[edge] = new_pc_up
                    self.counts_up[edge] = count_up + 1

                if not exc_inf and f > 1e-6:
                    new_pc_down = max(0, exc_delta) / max(1e-9, f)
                    if count_down == 0:
                        self.pseudocosts_down[edge] = new_pc_down
                    else:
                        old_pc = self.pseudocosts_down.get(edge, 0)
                        if not math.isnan(old_pc) and not math.isinf(old_pc):
                            self.pseudocosts_down[edge] = (1 - alpha) * old_pc + alpha * new_pc_down
                        else:
                            self.pseudocosts_down[edge] = new_pc_down
                    self.counts_down[edge] = count_down + 1

                if not fix_inf and not exc_inf:
                    scores.append((sb_score, edge, fix_inf, exc_inf))
                    if sb_score > best_score:
                        best_score = sb_score
                        best_edge = edge
                else:
                    if fix_inf:
                        edges_to_exclude.add(edge)
                    if exc_inf:
                        edges_to_fix.add(edge)

            # Evaluate reliable candidates using pseudocosts
            for edge in reliable_candidates:
                if shor_primal_solution is not None and edge in shor_primal_solution:
                    f = shor_primal_solution[edge]
                    f = max(0.01, min(0.99, f))
                else:
                    f = self.get_fractional_value(edge)
                
                pc_up = self.pseudocosts_up.get(edge, 1.0)
                pc_down = self.pseudocosts_down.get(edge, 1.0)
                
                # Confidence-weighted scoring
                count_up = self.counts_up.get(edge, 0)
                count_down = self.counts_down.get(edge, 0)
                confidence_up = min(1.0, count_up / (2 * self.reliability_eta))
                confidence_down = min(1.0, count_down / (2 * self.reliability_eta))
                confidence = (confidence_up + confidence_down) / 2
                
                delta_up = pc_up * (1 - f)
                delta_down = pc_down * f
                geometric_mean = (delta_up * delta_down) ** 0.5
                score = geometric_mean * (0.9 + 0.1 * confidence)
                
                scores.append((score, edge, False, False))

            if not scores and not (edges_to_fix or edges_to_exclude):
                return None

            # Handle forced decisions
            if edges_to_fix or edges_to_exclude:
                if self.verbose:
                    print(f"Creating single child with fixed edges: {edges_to_fix}, excluded edges: {edges_to_exclude}")
                child = self.create_single_child(edges_to_fix, edges_to_exclude)
                return ([list(edges_to_fix)[0] if edges_to_fix else list(edges_to_exclude)[0]], child)
            else:
                scores.sort(key=lambda x: x[0], reverse=True)
                best_score, best_edge, fix_inf, exc_inf = scores[0]

                if self.verbose:
                    print(f"Selected best edge {best_edge} with score {best_score}")

            return [best_edge] 
        


        # elif self.branching_rule == "hybrid_strong_fractional":
        #     # Adaptive criteria for choosing strong vs fractional branching

        #     bu = self.best_upper_bound
        #     lb = self.local_lower_bound
        #     if math.isfinite(bu) and math.isfinite(lb):
        #         denom = max(abs(bu), abs(lb), 1.0)
        #         gap_ratio = (bu - lb) / denom
        #     else:
        #         # treat as large gap early on when bounds aren't finite yet
        #         gap_ratio = 1.0

        #     use_strong_branching = (
        #         self.depth < 5
        #         or (self.depth < 10 and (len(self.fixed_edges) + len(self.excluded_edges)) < 0.1 * len(self.edges))
        #         or (gap_ratio > 0.99)  # large gap remaining
        #     )
        #     # use_strong_branching = (
        #     #     self.depth < 5 or  # Early in tree
        #     #     (self.depth < 10 and len(self.fixed_edges) + len(self.excluded_edges) < 0.1 * len(self.edges)) or  # Not much branching yet
        #     #     (self.best_upper_bound - self.local_lower_bound) / max(abs(self.best_upper_bound), 1) > 0.1  # Large gap remaining
        #     # )
            
        #     if use_strong_branching:
        #         # Strong branching phase with computational limits
        #         shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
                
        #         if shor_primal_solution is not None:
        #             normalized_edge_weights = shor_primal_solution
        #             tolerance = 1e-6
        #             candidate_edges = [
        #                 e for e in normalized_edge_weights
        #                 if e not in self.fixed_edges and
        #                 e not in self.excluded_edges and
        #                 e not in self.branched_edges and
        #                 normalized_edge_weights[e] > tolerance and
        #                 normalized_edge_weights[e] < 1.0 - tolerance
        #             ]
        #             # Prioritize most fractional edges for strong branching
        #             candidate_edges.sort(key=lambda e: abs(normalized_edge_weights[e] - 0.5))
        #         else:
        #             # Fallback to MST edges
        #             mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
        #             candidate_edges = [
        #                 e for e in mst_edges
        #                 if e not in self.fixed_edges and
        #                 e not in self.excluded_edges and
        #                 e not in self.branched_edges
        #             ]
                
        #         if not candidate_edges:
        #             if self.verbose:
        #                 print("No hybrid strong candidates available")
        #             return None
                
        #         # Limit strong branching evaluations based on depth
        #         max_sb_evals = max(1, min(5, 8 - self.depth)) if self.depth < 8 else 1
        #         candidate_edges = candidate_edges[:max_sb_evals]
                
        #         if self.verbose:
        #             print(f"Hybrid (strong): evaluating {len(candidate_edges)} edges at depth {self.depth}")
                
        #         edges_to_fix = set()
        #         edges_to_exclude = set()
        #         best_edge = None
        #         best_score = -float('inf')
        #         scores = []
                
        #         for edge in candidate_edges:
        #             score, _, _, fix_infeasible, exclude_infeasible = self.calculate_strong_branching_score(edge)
        #             scores.append((edge, score))
                    
        #             if fix_infeasible:
        #                 edges_to_exclude.add(edge)
        #             if exclude_infeasible:
        #                 edges_to_fix.add(edge)
        #             if not (fix_infeasible or exclude_infeasible):
        #                 if score > best_score:
        #                     best_score = score
        #                     best_edge = edge
                
        #         # Handle forced decisions
        #         if edges_to_fix or edges_to_exclude:
        #             if self.verbose:
        #                 print(f"Hybrid: forced decisions - fix: {edges_to_fix}, exclude: {edges_to_exclude}")
        #             child = self.create_single_child(edges_to_fix, edges_to_exclude)
        #             return ([list(edges_to_fix)[0] if edges_to_fix else list(edges_to_exclude)[0]], child)
                
        #         if not best_edge:
        #             if self.verbose:
        #                 print("No viable edge found in strong branching phase")
        #             # Fall through to fractional branching
        #         else:
        #             if self.verbose:
        #                 print(f"Hybrid (strong) selected edge {best_edge} with score {best_score}")
        #             return [best_edge]
            
        #     # Fractional branching phase (either chosen initially or fallback)
        #     if self.verbose and use_strong_branching:
        #         print("Falling back to fractional branching")
        #     elif self.verbose:
        #         print(f"Hybrid (fractional): using fractional at depth {self.depth}")
            
        #     shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
            
        #     if shor_primal_solution is None:
        #         # Final fallback to MST edges
        #         candidates = [
        #             tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges
        #             if tuple(sorted((u, v))) not in self.fixed_edges and
        #             tuple(sorted((u, v))) not in self.excluded_edges and
        #             tuple(sorted((u, v))) not in self.branched_edges
        #         ]
        #         return [candidates[0]] if candidates else None
            
        #     # Use fractional solution
        #     normalized_edge_weights = shor_primal_solution
        #     candidates = [
        #         e for e in normalized_edge_weights
        #         if e not in self.fixed_edges and
        #         e not in self.excluded_edges and
        #         e not in self.branched_edges
        #     ]
            
        #     if not candidates:
        #         return None
            
        #     # Score by distance from 0.5 (most fractional first)
        #     branching_scores = []
        #     for e in candidates:
        #         w = normalized_edge_weights.get(e, 0)
        #         distance_score = -abs(w - 0.5)  # Negative so sorting descending gives smallest distance first
        #         branching_scores.append((e, distance_score))
            
        #     branching_scores.sort(key=lambda x: x[1], reverse=True)
            
        #     if self.verbose:
        #         selected_edge = branching_scores[0][0]
        #         selected_weight = normalized_edge_weights[selected_edge]
        #         print(f"Hybrid (fractional) selected edge {selected_edge} with weight {selected_weight:.3f}")
            
        #     return [branching_scores[0][0]]
       
        elif self.branching_rule == "hybrid_strong_fractional":

            # --- Adaptive criteria for choosing strong vs fractional branching ---
            bu = self.best_upper_bound
            lb = self.local_lower_bound
            if math.isfinite(bu) and math.isfinite(lb):
                denom = max(abs(bu), abs(lb), 1.0)
                gap_ratio = (bu - lb) / denom
            else:
                # treat as large gap early on when bounds aren't finite yet
                gap_ratio = 1.0

            use_strong_branching = (
                self.depth < 5
                or (self.depth < 10 and (len(self.fixed_edges) + len(self.excluded_edges)) < 0.1 * len(self.edges))
                or (gap_ratio > 0.99)  # large gap remaining (keep your threshold)
            )

            def _norm(e):
                u, v = e
                return (u, v) if u <= v else (v, u)

            if use_strong_branching:
                # Strong branching phase with computational limits
                shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
                # shor_primal_solution = self.lagrangian_solver.compute_dantzig_wolfe_solution(self)


                if shor_primal_solution is not None:
                    # Normalize keys and drop non-finite weights
                    normalized_edge_weights = {}
                    for e, w in shor_primal_solution.items():
                        if w is None or not math.isfinite(w):
                            continue
                        try:
                            ne = _norm(e)
                        except Exception:
                            continue
                        normalized_edge_weights[ne] = w

                    tolerance = 1e-6
                    candidate_edges = [
                        e for e in normalized_edge_weights
                        if e not in self.fixed_edges
                        and e not in self.excluded_edges
                        and e not in self.branched_edges
                        and normalized_edge_weights[e] > tolerance
                        and normalized_edge_weights[e] < 1.0 - tolerance
                    ]
                    # Prioritize most fractional edges for strong branching
                    candidate_edges.sort(key=lambda e: abs(normalized_edge_weights[e] - 0.5))
                else:
                    # Fallback to MST edges
                    mst_edges = [_norm((u, v)) for u, v in (self.lagrangian_solver.best_mst_edges or [])]
                    candidate_edges = [
                        e for e in mst_edges
                        if e not in self.fixed_edges
                        and e not in self.excluded_edges
                        and e not in self.branched_edges
                    ]

                if not candidate_edges:
                    if self.verbose:
                        print("No hybrid strong candidates available")
                    return None

                # Limit strong branching evaluations based on depth (kept as-is)
                max_sb_evals = max(1, min(5, 8 - self.depth)) if self.depth < 8 else 1
                candidate_edges = candidate_edges[:max_sb_evals]

                if self.verbose:
                    print(f"Hybrid (strong): evaluating {len(candidate_edges)} edges at depth {self.depth}")

                edges_to_fix = set()
                edges_to_exclude = set()
                best_edge = None
                best_score = -float('inf')
                scores = []

                for edge in candidate_edges:
                    score, _, _, fix_infeasible, exclude_infeasible = self.calculate_strong_branching_score(edge)
                    scores.append((edge, score))

                    if fix_infeasible:
                        edges_to_exclude.add(edge)
                    if exclude_infeasible:
                        edges_to_fix.add(edge)
                    if not (fix_infeasible or exclude_infeasible):
                        if score > best_score:
                            best_score = score
                            best_edge = edge

                # Handle forced decisions
                if edges_to_fix or edges_to_exclude:
                    if self.verbose:
                        print(f"Hybrid: forced decisions - fix: {edges_to_fix}, exclude: {edges_to_exclude}")
                    child = self.create_single_child(edges_to_fix, edges_to_exclude)
                    return ([list(edges_to_fix)[0] if edges_to_fix else list(edges_to_exclude)[0]], child)

                if not best_edge:
                    if self.verbose:
                        print("No viable edge found in strong branching phase")
                    # Fall through to fractional branching
                else:
                    if self.verbose:
                        print(f"Hybrid (strong) selected edge {best_edge} with score {best_score}")
                    return [best_edge]

            # Fractional branching phase (either chosen initially or fallback)
            if self.verbose and use_strong_branching:
                print("Falling back to fractional branching")
            elif self.verbose:
                print(f"Hybrid (fractional): using fractional at depth {self.depth}")

            shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
            # shor_primal_solution = self.lagrangian_solver.compute_dantzig_wolfe_solution(self)


            if shor_primal_solution is None:
                # Final fallback to MST edges
                candidates = [
                    _norm((u, v)) for u, v in (self.lagrangian_solver.best_mst_edges or [])
                    if _norm((u, v)) not in self.fixed_edges
                    and _norm((u, v)) not in self.excluded_edges
                    and _norm((u, v)) not in self.branched_edges
                ]
                return [candidates[0]] if candidates else None

            # Use fractional solution — normalize keys and drop non-finite weights
            normalized_edge_weights = {}
            for e, w in shor_primal_solution.items():
                if w is None or not math.isfinite(w):
                    continue
                try:
                    ne = _norm(e)
                except Exception:
                    continue
                normalized_edge_weights[ne] = w

            candidates = [
                e for e in normalized_edge_weights
                if e not in self.fixed_edges
                and e not in self.excluded_edges
                and e not in self.branched_edges
            ]

            if not candidates:
                return None

            # Score by distance from 0.5 (most fractional first)
            branching_scores = []
            for e in candidates:
                w = normalized_edge_weights.get(e, 0.0)
                distance_score = -abs(w - 0.5)  # Negative so sorting descending gives smallest distance first
                branching_scores.append((e, distance_score))

            branching_scores.sort(key=lambda x: x[1], reverse=True)

            if self.verbose:
                selected_edge = branching_scores[0][0]
                selected_weight = normalized_edge_weights[selected_edge]
                print(f"Hybrid (fractional) selected edge {selected_edge} with weight {selected_weight:.3f}")

            return [branching_scores[0][0]]

            
       
        else:
            raise ValueError(f"Unknown branching rule: {self.branching_rule}")

    def create_single_child(self, edges_to_fix, edges_to_exclude):
        """Create a single child node with multiple edges fixed or excluded."""
        print("hhhhhhhhhhhhhh")
        
        new_branched_edges = self.branched_edges | edges_to_fix | edges_to_exclude

        # Combine active cuts and new cuts from the parent
        all_cuts = self.active_cuts + [(set(tuple(sorted((x, y))) for x, y in cut), rhs) for cut, rhs in self.new_cuts]
        current_multipliers = self.lagrangian_solver.best_cut_multipliers_for_best_bound.copy()
        for cut_idx in range(len(self.active_cuts), len(all_cuts)):
            current_multipliers[cut_idx] = 0.001  # Default multiplier for new cuts

        # Filter cuts for excluded edges
        excluded_cuts = []
        for cut, rhs in all_cuts:
            remaining_cover = cut - edges_to_exclude
            if len(remaining_cover) <= rhs:
                excluded_cuts.append((cut, rhs))
        # kept_cuts = [c for c in all_cuts if c not in excluded_cuts]
        # kept_indices = [i for i, c in enumerate(all_cuts) if c in kept_cuts]
        # kept_multipliers = {i: current_multipliers[i] for i in kept_indices if i in current_multipliers}
        kept_cuts = [c for c in all_cuts if c not in excluded_cuts]
        kept_indices = [i for i, c in enumerate(all_cuts) if c in kept_cuts]
        kept_multipliers = {}
        for new_i, old_i in enumerate(kept_indices):
            if old_i in current_multipliers:
                kept_multipliers[new_i] = current_multipliers[old_i]

        child = MSTNode(
            self.edges,
            self.num_nodes,
            self.budget,
            fixed_edges=self.fixed_edges | edges_to_fix,
            excluded_edges=self.excluded_edges | edges_to_exclude,
            branched_edges=new_branched_edges,
            initial_lambda=self.lagrangian_solver.best_lambda if self.inherit_lambda else 0.05,
            inherit_lambda=self.inherit_lambda,
            branching_rule=self.branching_rule,
            step_size=self.lagrangian_solver.step_size if self.inherit_step_size else 0.00001,
            inherit_step_size=self.inherit_step_size,
            use_cover_cuts=self.use_cover_cuts,
            cut_frequency=self.cut_frequency,
            node_cut_frequency=self.node_cut_frequency,
            parent_cover_cuts=kept_cuts,
            parent_cover_multipliers=kept_multipliers,
            use_bisection=self.use_bisection,
            max_iter=10,
            verbose=self.verbose
        )

        if self.verbose:
            print(f"Created single child: fixed_edges={child.fixed_edges}, excluded_edges={child.excluded_edges}, cuts={len(child.active_cuts)}")

        return child

   

    def get_modified_weight(self, edge):
        u, v = tuple(sorted(edge))
        w, l = self.lagrangian_solver.edge_attributes[(u, v)]

        modified = w + self.lagrangian_solver.best_lambda * l
        
        for cut_idx, (cut, _) in enumerate(self.active_cuts):
            if (u, v) in cut:
                modified += self.cut_multipliers.get(cut_idx, 0)
        
        return modified

   
    # def calculate_strong_branching_score(self, edge):

    #     u, v = tuple(sorted(edge))
    #     all_cuts = self.active_cuts + [(set(tuple(sorted((x, y))) for x, y in cut), rhs) for cut, rhs in self.new_cuts]
    #     capped_multipliers = self.lagrangian_solver.best_cut_multipliers_for_best_bound.copy()
    #     for cut_idx in range(len(self.active_cuts), len(all_cuts)):
    #         capped_multipliers[cut_idx] = 0.001

    #     current_lambda = self.lagrangian_solver.best_lambda if self.inherit_lambda else 0.05
    #     current_step_size = self.lagrangian_solver.step_size if self.inherit_step_size else 0.00001


    #     # Evaluate fixed child
    #     fixed_child = MSTNode(
    #         self.edges,
    #         self.num_nodes,
    #         self.budget,
    #         fixed_edges=self.fixed_edges | {(u, v)},
    #         excluded_edges=self.excluded_edges,
    #         branched_edges=self.branched_edges | {(u, v)},
    #         initial_lambda=current_lambda,
    #         inherit_lambda=self.inherit_lambda,
    #         branching_rule=self.branching_rule,
    #         step_size=current_step_size,
    #         inherit_step_size=self.inherit_step_size,
    #         use_cover_cuts=self.use_cover_cuts,
    #         cut_frequency=self.cut_frequency,
    #         node_cut_frequency=self.node_cut_frequency,
    #         parent_cover_cuts=all_cuts,
    #         parent_cover_multipliers=capped_multipliers,
    #         use_bisection=self.use_bisection,
    #         max_iter=10,
    #         verbose=self.verbose
    #     )
    #     fixed_lower_bound = fixed_child.local_lower_bound
    #     fix_infeasible = fixed_lower_bound > self.best_upper_bound or math.isnan(fixed_lower_bound) or math.isinf(fixed_lower_bound)
    #     if fix_infeasible and self.verbose:
    #         print(f"Fixed child for edge {edge} has invalid bound: {fixed_lower_bound}")

    #     # Evaluate excluded child
    #     excluded_cuts = [(cut, rhs) for cut, rhs in all_cuts if len(cut - {(u, v)}) <= rhs]
    #     kept_cuts = [c for c in all_cuts if c not in excluded_cuts]
    #     kept_indices = [i for i, c in enumerate(all_cuts) if c in kept_cuts]
    #     kept_multipliers = {i: capped_multipliers[i] for i in kept_indices if i in capped_multipliers}

    #     excluded_child = MSTNode(
    #         self.edges,
    #         self.num_nodes,
    #         self.budget,
    #         fixed_edges=self.fixed_edges,
    #         excluded_edges=self.excluded_edges | {(u, v)},
    #         branched_edges=self.branched_edges | {(u, v)},
    #         initial_lambda=current_lambda,
    #         inherit_lambda=self.inherit_lambda,
    #         branching_rule=self.branching_rule,
    #         step_size=current_step_size,
    #         inherit_step_size=self.inherit_step_size,
    #         use_cover_cuts=self.use_cover_cuts,
    #         cut_frequency=self.cut_frequency,
    #         node_cut_frequency=self.node_cut_frequency,
    #         parent_cover_cuts=kept_cuts,
    #         parent_cover_multipliers=kept_multipliers,
    #         use_bisection=self.use_bisection,
    #         max_iter=10,
    #         verbose=self.verbose
    #     )
    #     excluded_lower_bound = excluded_child.local_lower_bound
    #     exclude_infeasible = excluded_lower_bound > self.best_upper_bound or math.isnan(excluded_lower_bound) or math.isinf(excluded_lower_bound)
    #     if exclude_infeasible and self.verbose:
    #         print(f"Excluded child for edge {edge} has invalid bound: {excluded_lower_bound}")

    #     # Compute score
    #     fix_lower_score = (fixed_lower_bound - self.local_lower_bound) if not math.isnan(fixed_lower_bound) and not math.isinf(fixed_lower_bound) else float('inf')
    #     exc_lower_score = (excluded_lower_bound - self.local_lower_bound) if not math.isnan(excluded_lower_bound) and not math.isinf(excluded_lower_bound) else float('inf')
    #     score = max(fix_lower_score, 1e-6) * max(exc_lower_score, 1e-6) if fix_lower_score != float('inf') and exc_lower_score != float('inf') else float('inf')

    #     if self.verbose:
    #         print(f"Edge {edge}: Score {score}, Fix infeasible: {fix_infeasible}, Exclude infeasible: {exclude_infeasible}")

    #     # return score, fix_infeasible, exclude_infeasible
    #     return score, fix_lower_score, exc_lower_score, fix_infeasible, exclude_infeasible  # Change return to include deltas (fix_lower_score, exc_lower_score)
   # Add this method to MSTNode class in mstkpbranchandbound.py

   

    def simulate_branching_bound(self, edge, fix_edge: bool = True, max_iters: int = 10):
        """
        Behavior-preserving strong-branching probe using a pooled LagrangianMST.
        No unsupported kwargs are passed to solve().
        """
        # --- Normalize edge ---
        try:
            u, v = edge
        except Exception:
            raise ValueError(f"simulate_branching_bound: invalid edge {edge!r}")
        branched_edge = tuple(sorted((u, v)))

        # --- Build hypothetical fixed/excluded sets (avoid deep copies) ---
        if fix_edge:
            new_fixed = self.fixed_edges | {branched_edge}
            new_excluded = self.excluded_edges
        else:
            new_fixed = self.fixed_edges
            new_excluded = self.excluded_edges | {branched_edge}

        # --- Inherit cuts (keep stable order) ---
        if getattr(self, "new_cuts", None):
            all_cuts = list(self.active_cuts) + list(self.new_cuts)
        else:
            all_cuts = list(self.active_cuts)

        # Multipliers aligned to all_cuts indices
        cut_multipliers = {}
        if getattr(self, "cut_multipliers", None) is not None and self.active_cuts:
            parent_index = {}
            for idx, (cset, rhs) in enumerate(self.active_cuts):
                parent_index[(frozenset(cset), rhs)] = idx
            for idx, (cset, rhs) in enumerate(all_cuts):
                pidx = parent_index.get((frozenset(cset), rhs))
                cut_multipliers[idx] = self.cut_multipliers.get(pidx, 0.001) if pidx is not None else 0.001
        else:
            for idx in range(len(all_cuts)):
                cut_multipliers[idx] = 0.001

        # --- Borrow, reset, (optionally) set warm-start state via attributes, then solve ---
        with self._sb_pool.borrow() as sim_solver:
            sim_solver.reset(
                fixed_edges=new_fixed,
                excluded_edges=new_excluded,
                initial_lambda=getattr(self.lagrangian_solver, "best_lambda", getattr(self, "initial_lambda", 0.05)),
                step_size=getattr(self.lagrangian_solver, "step_size", getattr(self, "step_size", 0.1)),
                max_iter=int(max_iters),
                use_cover_cuts=bool(getattr(self, "use_cover_cuts", False)),
                cut_frequency=int(getattr(self, "cut_frequency", 10)),
                use_bisection=False,
                verbose=False,
            )

            # (Optional) Warm-start by assigning fields directly if your solver honors them.
            # These two lines are safe; they won't break if attrs don't exist.
            if getattr(self, "mst_edges", None) is not None and hasattr(sim_solver, "best_mst_edges"):
                sim_solver.best_mst_edges = list(self.mst_edges)
            if hasattr(self.lagrangian_solver, "best_lambda") and hasattr(sim_solver, "lmbda"):
                sim_solver.lmbda = float(self.lagrangian_solver.best_lambda)

            # Call solve WITHOUT unsupported kwargs
            lower_bound, upper_bound, _info = sim_solver.solve(
                inherited_cuts=all_cuts,
                inherited_multipliers=cut_multipliers
            )

        return float(lower_bound)


    def calculate_strong_branching_score(self, edge):
        """
        Fast strong branching using simulation instead of full child creation.
        
        Returns:
            (score, fix_delta, exc_delta, fix_infeasible, exclude_infeasible)
        """
        u, v = tuple(sorted(edge))
            # === EARLY PRUNING: Check if fixing violates any cut ===
    
        # Simulate fixing edge
        fixed_lower_bound = self.simulate_branching_bound(edge, fix_edge=True, max_iters=5)
        fix_infeasible = (fixed_lower_bound > self.best_upper_bound or 
                        math.isnan(fixed_lower_bound) or 
                        math.isinf(fixed_lower_bound))
        
        if self.verbose and fix_infeasible:
            print(f"Fixed simulation for edge {edge} is infeasible: {fixed_lower_bound}")
        
        # Simulate excluding edge
        excluded_lower_bound = self.simulate_branching_bound(edge, fix_edge=False, max_iters=5)
        exclude_infeasible = (excluded_lower_bound > self.best_upper_bound or 
                            math.isnan(excluded_lower_bound) or 
                            math.isinf(excluded_lower_bound))
        
        if self.verbose and exclude_infeasible:
            print(f"Excluded simulation for edge {edge} is infeasible: {excluded_lower_bound}")
        
        # Compute score
        fix_delta = (fixed_lower_bound - self.local_lower_bound) if not fix_infeasible else float('inf')
        exc_delta = (excluded_lower_bound - self.local_lower_bound) if not exclude_infeasible else float('inf')
        
        # Product score (as in your original)
        score = (max(fix_delta, 1e-6) * max(exc_delta, 1e-6) 
                if fix_delta != float('inf') and exc_delta != float('inf') 
                else float('inf'))
        
        if self.verbose:
            print(f"Edge {edge}: Score={score:.4f}, Fix Δ={fix_delta:.4f}, Exc Δ={exc_delta:.4f}")
        
        return score, fix_delta, exc_delta, fix_infeasible, exclude_infeasible
    
    def simulate_fix_edge(self, u, v):
        normalized_edge = tuple(sorted((u, v)))
        mst_edges = [tuple(sorted((x, y))) for x, y in self.lagrangian_solver.best_mst_edges]
        if normalized_edge in mst_edges:
            return self.local_lower_bound

        mst_graph = nx.Graph(mst_edges)
        mst_graph.add_edge(u, v)

        try:
            cycle = nx.find_cycle(mst_graph, source=u)
        except nx.NetworkXNoCycle:
            return self.local_lower_bound

        cycle_without_fixed = [edge for edge in cycle if edge not in self.fixed_edges]
        heaviest_edge = None
        max_weight = float('-inf')
        for edge in cycle_without_fixed:
            if edge not in self.fixed_edges:
                edge_weight = self.get_modified_weight(edge)
                if edge_weight > max_weight:
                    max_weight = edge_weight
                    heaviest_edge = edge

        if not heaviest_edge:
            return float('inf')

        fixed_edge_weight = self.get_modified_weight(normalized_edge)
        heaviest_edge_weight = self.get_modified_weight(heaviest_edge)
        new_lower_bound = self.local_lower_bound + fixed_edge_weight - heaviest_edge_weight
        return new_lower_bound

    def simulate_exclude_edge(self, u, v):
        normalized_edge = tuple(sorted((u, v)))
        mst_edges = [tuple(sorted((x, y))) for x, y in self.lagrangian_solver.best_mst_edges]
        if normalized_edge not in mst_edges:
            return self.local_lower_bound

        mst_graph = nx.Graph(mst_edges)
        mst_graph.remove_edge(u, v)

        components = list(nx.connected_components(mst_graph))
        if len(components) != 2:
            return float('inf')

        cheapest_edge = None
        min_weight = float('inf')
        for x, y, w, l in self.edges:
            normalized = tuple(sorted((x, y)))
            if normalized == normalized_edge:
                continue
            if (x in components[0] and y in components[1]) or (x in components[1] and y in components[0]):
                if normalized not in self.excluded_edges:
                    edge_weight = self.get_modified_weight(normalized)
                    if edge_weight < min_weight:
                        min_weight = edge_weight
                        cheapest_edge = normalized

        if not cheapest_edge:
            return float('inf')

        excluded_edge_weight = self.get_modified_weight(normalized_edge)
        replacement_edge_weight = self.get_modified_weight(cheapest_edge)
        new_lower_bound = self.local_lower_bound - excluded_edge_weight + replacement_edge_weight
        return new_lower_bound

    def print_cut_info(self):
        if self.verbose:
            print(f"\nNode Cut Status (Fixed: {self.fixed_edges}, Excluded: {self.excluded_edges})")
            print("Active Cuts:")
            for i, (cut, rhs) in enumerate(self.active_cuts):
                mult = self.cut_multipliers.get(i, 0)
                print(f"Cut {i}: Cut {cut} (RHS: {rhs}, Multiplier: {mult:.3f})")
            
            print("\nInherited Cuts Breakdown:")
            inherited_from_parent = 0
            new_generated = 0
            for cut, rhs in self.lagrangian_solver.best_cuts:
                if (cut, rhs) in self.active_cuts:
                    inherited_from_parent += 1
                else:
                    new_generated += 1
            print(f"Total cuts: {len(self.lagrangian_solver.best_cuts)}")
            print(f" - Inherited: {inherited_from_parent}")
            print(f" - New: {new_generated}")

    def get_fractional_value(self, edge):
        normalized = tuple(sorted(edge))
        violation = sum(self.lagrangian_solver.edge_attributes[e][1] for e in self.mst_edges) - self.budget  # Positive if over budget
        if violation <= 0:
            return 0.5  # Feasible or under; neutral uncertainty
        if normalized in self.mst_edges:
            edge_contrib = self.lagrangian_solver.edge_attributes[normalized][1] / violation if violation > 0 else 0.5
            f = 1 - min(1.0, max(0.0, edge_contrib))  # High contrib = more fractional (likely to flip out)
        else:
            sim_lb = self.simulate_fix_edge(*edge)
            delta = max(0, sim_lb - self.local_lower_bound)
            f = min(1.0, max(0.0, delta / (self.lagrangian_solver.best_lambda or 1.0)))  # Delta normalized by lambda
        if self.verbose:
            print(f"Slackness-based f={f:.2f} for edge {edge}")
        return max(0.01, min(0.99, f))  # Clamp away from 0/1 to avoid div-by-zero
    

