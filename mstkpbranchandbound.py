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






class MSTNode(Node):
    _solver_pool = None  # shared across all nodes in one solve

    def __init__(self, edges, num_nodes, budget, fixed_edges=set(), excluded_edges=set(), branched_edges=set(),
                 initial_lambda=0.4, inherit_lambda=False, branching_rule="random_mst",
                 step_size=0.001, inherit_step_size=False, use_cover_cuts=False, cut_frequency=5,
                 node_cut_frequency=10, parent_cover_cuts=None, parent_cover_multipliers=None,
                 use_bisection=False, max_iter=12, verbose=False, depth=0,
                 pseudocosts_up=None, pseudocosts_down=None, counts_up=None, counts_down=None,
                 reliability_eta=5, lookahead_lambda=4):
        if depth == 0:
            MSTNode.global_edges = [(min(u, v), max(u, v), w, l) for u, v, w, l in edges]
            MSTNode.global_graph = nx.Graph()
            MSTNode.global_graph.add_edges_from(
                [(u, v, {"w": w, "l": l}) for u, v, w, l in MSTNode.global_edges]
            )
            MSTNode._solver_pool = None  # reset pool for a fresh instance

        # self.pseudocosts_up = pseudocosts_up or defaultdict(float)
        # self.pseudocosts_down = pseudocosts_down or defaultdict(float)
        # self.counts_up = counts_up or defaultdict(int)
        # self.counts_down = counts_down or defaultdict(int)
        self.pseudocosts_up = (
        pseudocosts_up if pseudocosts_up is not None else defaultdict(float)
        )
        self.pseudocosts_down = (
            pseudocosts_down if pseudocosts_down is not None else defaultdict(float)
        )
        self.counts_up = (
            counts_up if counts_up is not None else defaultdict(int)
        )
        self.counts_down = (
            counts_down if counts_down is not None else defaultdict(int)
        )
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
        self.initial_lambda = initial_lambda if initial_lambda is not None else 0.4
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
            step_size=self.step_size, max_iter=max_iter, 
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
                    step_size=self.step_size, max_iter=5,
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
                    parent_cover_multipliers.get(cut_idx, 0.00001)
                    if parent_cover_multipliers else 0.00001
                )

        self.local_lower_bound, self.best_upper_bound, self.new_cuts = self.lagrangian_solver.solve(
            inherited_cuts=[(set(tuple(sorted((u, v))) for u, v in cut), rhs) for cut, rhs in self.active_cuts],
            inherited_multipliers=self.cut_multipliers,
            depth=self.depth
        )
        # --- SYNC cuts and multipliers with solver's final state ---
        # self.active_cuts = [
        #     (set(cut), rhs) for (cut, rhs) in self.lagrangian_solver.best_cuts
        # ]
        # self.cut_multipliers = self.lagrangian_solver.best_cut_multipliers_for_best_bound.copy()
        # self.new_cuts = []  # best_cuts already includes surviving cuts
        # -----------------------------------------------------------


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

        # all_cuts = []
        # for p in (self.active_cuts or []):
        #     all_cuts.append(_norm_pair(p))
        # for p in (getattr(self, "new_cuts", []) or []):
        #     all_cuts.append(_norm_pair(p))

        # # Multipliers: parent snapshot + 0 for newly added cuts
        # parent_mu = getattr(solver, "best_cut_multipliers_for_best_bound", {}) or {}
        # current_multipliers = dict(parent_mu)
        # first_new = len(self.active_cuts or [])
        # for cut_idx in range(first_new, len(all_cuts)):
        #     current_multipliers[cut_idx] = 0.00001
        # 1) Build merged cuts
        all_cuts = []
        for p in (self.active_cuts or []):
            all_cuts.append(_norm_pair(p))
        for p in (getattr(self, "new_cuts", []) or []):
            all_cuts.append(_norm_pair(p))

        # 2) Build a map from support -> μ using solver.best_cuts
        best_cuts = getattr(solver, "best_cuts", []) or []
        best_mu   = getattr(solver, "best_cut_multipliers_for_best_bound", {}) or {}

        support_to_mu = {}
        for i, (cut_i, rhs_i) in enumerate(best_cuts):
            key = (frozenset(cut_i), rhs_i)
            support_to_mu[key] = float(best_mu.get(i, 0.0))

        # 3) Assign multipliers to all_cuts by support, default small μ for unseen cuts
        current_multipliers = {}
        for idx, (cut, rhs) in enumerate(all_cuts):
            key = (frozenset(cut), rhs)
            current_multipliers[idx] = support_to_mu.get(key, 0.00001)

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
                    initial_lambda=solver.best_lambda if self.inherit_lambda else 0.4,
                    inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
                    step_size=solver.step_size if self.inherit_step_size else 0.001,
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
            initial_lambda=solver.best_lambda if self.inherit_lambda else 0.4,
            inherit_lambda=self.inherit_lambda, branching_rule=self.branching_rule,
            step_size=solver.step_size if self.inherit_step_size else 0.001,
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
                normalized_edge_weights = None  # only used in sb_fractional

                if shor_primal_solution is None:
                    # No fractional info available -> fall back to MST edges
                    mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                    candidate_edges = [
                        e for e in mst_edges
                        if e not in self.fixed_edges
                        and e not in self.excluded_edges
                        and e not in self.branched_edges
                    ]
                else:
                    normalized_edge_weights = shor_primal_solution
                    tolerance = 1e-6
                    candidate_edges = [
                        e for e in normalized_edge_weights
                        if e not in self.fixed_edges and
                        e not in self.excluded_edges and
                        e not in self.branched_edges and
                        normalized_edge_weights[e] > tolerance and
                        normalized_edge_weights[e] < 1.0 - tolerance
                    ]
                    if not candidate_edges:
                        candidate_edges = [
                        e for e in normalized_edge_weights
                        if e not in self.fixed_edges and
                        e not in self.excluded_edges and
                        e not in self.branched_edges
                        ]

            if not candidate_edges:
                if self.verbose:
                    print(f"No {self.branching_rule} candidates available")
                return None

            if self.verbose:
                print(f"Node {id(self)}: {self.branching_rule} evaluating {len(candidate_edges)} edges: {candidate_edges}")

            MAX_SB_CANDIDATES = 10
            if self.branching_rule == "sb_fractional" and normalized_edge_weights is not None:
                candidate_edges.sort(
                    key=lambda e: -abs(normalized_edge_weights[e] - 0.5)
                )
            candidate_edges = candidate_edges[:MAX_SB_CANDIDATES]
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

            normalized_edge_weights = shor_primal_solution
            candidates = [
                e for e in shor_primal_solution
                if e not in self.fixed_edges and
                e not in self.excluded_edges and
                e not in self.branched_edges and
                abs(normalized_edge_weights[e]) > 1e-6 and
                abs(normalized_edge_weights[e] - 1.0) > 1e-6
            ]
                            
            if not candidates:
                candidates = [
                e for e in shor_primal_solution
                if e not in self.fixed_edges and
                e not in self.excluded_edges and
                e not in self.branched_edges
                ]

            return candidates if candidates else None            

        
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
        #     # Get fractional solution for prioritization
        #     shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
        #     # shor_primal_solution = self.lagrangian_solver.compute_dantzig_wolfe_solution(self)

        #     candidate_edges = []
        #     if shor_primal_solution is not None:
        #         tolerance = 1e-6
        #         candidate_edges = [
        #             e for e in shor_primal_solution
        #             if e not in self.fixed_edges
        #             and e not in self.excluded_edges
        #             and e not in self.branched_edges
        #             and shor_primal_solution[e] > 0.0
        #             and shor_primal_solution[e] < 1.0
        #         ]
        #         candidate_edges.sort(key=lambda e: abs(shor_primal_solution.get(e, 0.5) - 0.5))

        #     if shor_primal_solution is None or not candidate_edges:
        #         mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
        #         candidate_edges = [
        #             e for e in mst_edges
        #             if e not in self.fixed_edges
        #             and e not in self.excluded_edges
        #             and e not in self.branched_edges
        #         ]

        #     if not candidate_edges:
        #         return None

        #     # Separate by reliability
        #     unhistoried = []
        #     reliable_candidates = []
        #     for e in candidate_edges:
        #         cu = self.counts_up.get(e, 0)
        #         cd = self.counts_down.get(e, 0)
        #         if cu < self.reliability_eta or cd < self.reliability_eta:
        #             unhistoried.append(e)
        #         else:
        #             reliable_candidates.append(e)

        #     # Adaptive lookahead
        #     duality_gap = (
        #         self.best_upper_bound - self.local_lower_bound
        #         if self.best_upper_bound < float("inf") else float("inf")
        #     )
        #     if self.depth < 5:
        #         max_sb_evals = self.lookahead_lambda
        #     else:
        #         max_sb_evals = max(2, self.lookahead_lambda - 1)

        #     # if shor_primal_solution is not None:

        #     #     unhistoried.sort(key=lambda e: abs(shor_primal_solution.get(e, 0.5) - 0.5))

        #     unhistoried = unhistoried[:max_sb_evals]

        #     edges_to_fix = set()
        #     edges_to_exclude = set()
        #     best_score = float("-inf")
        #     best_edge = None
        #     scores = []

        #     # Evaluate unreliable edges with strong branching
        #     for edge in unhistoried:
        #         # Better fractional estimation
        #         if shor_primal_solution is not None and edge in shor_primal_solution:
        #             f = shor_primal_solution[edge]
        #             f = max(0.01, min(0.99, f))
        #         else:
        #             f = self.get_fractional_value(edge)

        #         count_up = self.counts_up.get(edge, 0)
        #         count_down = self.counts_down.get(edge, 0)


        #         sb_score, fix_delta, exc_delta, fix_inf, exc_inf = self.calculate_strong_branching_score(edge)

        #         # Adaptive learning rate
        #         if count_up == 0 and count_down == 0:
        #             alpha = 0.5
        #         elif count_up < 3 or count_down < 3:
        #             alpha = 0.3
        #         else:
        #             alpha = 0.1

        #         # Update pseudocosts (UP / fix = x_e → 1)
        #         if not fix_inf and (1 - f) > 1e-6:
        #             new_pc_up = max(0, fix_delta) / max(1e-9, (1 - f))
        #             if count_up == 0:
        #                 self.pseudocosts_up[edge] = new_pc_up
        #             else:
        #                 old_pc = self.pseudocosts_up.get(edge, 0)
        #                 if not math.isnan(old_pc) and not math.isinf(old_pc):
        #                     self.pseudocosts_up[edge] = (1 - alpha) * old_pc + alpha * new_pc_up
        #                 else:
        #                     self.pseudocosts_up[edge] = new_pc_up
        #             self.counts_up[edge] = count_up + 1

        #         # Update pseudocosts (DOWN / exclude = x_e → 0)
        #         if not exc_inf and f > 1e-6:
        #             new_pc_down = max(0, exc_delta) / max(1e-9, f)
        #             if count_down == 0:
        #                 self.pseudocosts_down[edge] = new_pc_down
        #             else:
        #                 old_pc = self.pseudocosts_down.get(edge, 0)
        #                 if not math.isnan(old_pc) and not math.isinf(old_pc):
        #                     self.pseudocosts_down[edge] = (1 - alpha) * old_pc + alpha * new_pc_down
        #                 else:
        #                     self.pseudocosts_down[edge] = new_pc_down
        #             self.counts_down[edge] = count_down + 1

        #         # DEBUG: state after strong branching update for this edge
        #         # print(
        #         #     f"[RLB] NodeDepth={self.depth} EDGE={edge} AFTER_SB "
        #         #     f"pc_up={self.pseudocosts_up.get(edge, None)} "
        #         #     f"pc_down={self.pseudocosts_down.get(edge, None)} "
        #         #     f"counts=({self.counts_up.get(edge, 0)}, {self.counts_down.get(edge, 0)}) "
        #         #     f"fix_inf={fix_inf} exc_inf={exc_inf} sb_score={sb_score:.4f}"
        #         # )

        #         # Process SB outcome
        #         if not fix_inf and not exc_inf:
        #             scores.append((sb_score, edge, fix_inf, exc_inf))
        #             if sb_score > best_score:
        #                 best_score = sb_score
        #                 best_edge = edge
        #         else:
        #             # If one side is infeasible, we can force the other decision
        #             if fix_inf:
        #                 edges_to_exclude.add(edge)
        #             if exc_inf:
        #                 edges_to_fix.add(edge)

        #     # Evaluate reliable candidates using pseudocosts
        #     for edge in reliable_candidates:
        #         if shor_primal_solution is not None and edge in shor_primal_solution:
        #             f = shor_primal_solution[edge]
        #             f = max(0.01, min(0.99, f))
        #         else:
        #             f = self.get_fractional_value(edge)

        #         pc_up = self.pseudocosts_up.get(edge, 1.0)
        #         pc_down = self.pseudocosts_down.get(edge, 1.0)

        #         count_up = self.counts_up.get(edge, 0)
        #         count_down = self.counts_down.get(edge, 0)

        #         # Confidence-weighted scoring
        #         confidence_up = min(1.0, count_up / (2 * self.reliability_eta))
        #         confidence_down = min(1.0, count_down / (2 * self.reliability_eta))
        #         confidence = (confidence_up + confidence_down) / 2

        #         delta_up = pc_up * (1 - f)
        #         delta_down = pc_down * f
        #         geometric_mean = (delta_up * delta_down) ** 0.5
        #         score = geometric_mean * (0.9 + 0.1 * confidence)

        #         scores.append((score, edge, False, False))

        #     if not scores and not (edges_to_fix or edges_to_exclude):
        #         return None

        #     # Handle forced decisions
        #     if edges_to_fix or edges_to_exclude:
        #         if self.verbose:
        #             print(f"[RLB] FORCED CHILD: fix={edges_to_fix}, exclude={edges_to_exclude}")
        #         child = self.create_single_child(edges_to_fix, edges_to_exclude)
        #         return ([list(edges_to_fix)[0] if edges_to_fix else list(edges_to_exclude)[0]], child)
        #     else:
        #         scores.sort(key=lambda x: x[0], reverse=True)
        #         best_score, best_edge, fix_inf, exc_inf = scores[0]

        #         if self.verbose:
        #             print(f"[RLB] SELECTED best_edge={best_edge} score={best_score:.4f}")

        #     return [best_edge]
        # elif self.branching_rule == "reliability":
        #     # 1) Fractional solution for prioritization
        #     shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()

        #     candidate_edges = []
        #     if shor_primal_solution is not None:
        #         tolerance = 1e-6
        #         candidate_edges = [
        #             e for e in shor_primal_solution
        #             if e not in self.fixed_edges
        #             and e not in self.excluded_edges
        #             and e not in self.branched_edges
        #             and shor_primal_solution[e] > tolerance
        #             and shor_primal_solution[e] < 1.0 - tolerance
        #         ]
        #         # most fractional first
        #         candidate_edges.sort(key=lambda e: abs(shor_primal_solution.get(e, 0.5) - 0.5))

        #     if shor_primal_solution is None or not candidate_edges:
        #         mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
        #         candidate_edges = [
        #             e for e in mst_edges
        #             if e not in self.fixed_edges
        #             and e not in self.excluded_edges
        #             and e not in self.branched_edges
        #         ]

        #     if not candidate_edges:
        #         return None

        #     # 2) Split into unhistoried vs reliable, based on TOTAL observations
        #     unhistoried = []
        #     reliable_candidates = []
        #     for e in candidate_edges:
        #         cu = self.counts_up.get(e, 0)
        #         cd = self.counts_down.get(e, 0)

        #         # print("counts",  cu, cd)
        #         total = cu + cd
        #         if total >= self.reliability_eta:
        #             reliable_candidates.append(e)
        #         else:
        #             unhistoried.append(e)

        #     # 3) Adaptive lookahead: how many edges to strong-branch
        #     if self.depth < 5:
        #         max_sb_evals = self.lookahead_lambda
        #     else:
        #         max_sb_evals = max(2, self.lookahead_lambda - 1)

        #     # Strong branching only on the most fractional unhistoried edges
        #     if shor_primal_solution is not None:
        #         unhistoried.sort(key=lambda e: abs(shor_primal_solution.get(e, 0.5) - 0.5))
        #         reliable_candidates.sort(key=lambda e: abs(shor_primal_solution.get(e, 0.5) - 0.5))

        #     else:
        #         unhistoried.sort(key=lambda e: abs(self.get_fractional_value(e) - 0.5))
        #         reliable_candidates.sort(key=lambda e: abs(self.get_fractional_value(e) - 0.5))


        #     unhistoried = unhistoried[:max_sb_evals]

        #     edges_to_fix = set()
        #     edges_to_exclude = set()
        #     scores = []

        #     # 4) Strong branching on unhistoried edges (also updates pseudocosts)
        #     for edge in unhistoried:
        #         if shor_primal_solution is not None and edge in shor_primal_solution:
        #             f = shor_primal_solution[edge]
        #         else:
        #             f = self.get_fractional_value(edge)
        #         f = max(0.01, min(0.99, f))

        #         count_up = self.counts_up.get(edge, 0)
        #         count_down = self.counts_down.get(edge, 0)

        #         sb_score, fix_delta, exc_delta, fix_inf, exc_inf = self.calculate_strong_branching_score(edge)
        #         # print("unhistoriedscore", sb_score)

        #         # adaptive learning rate
        #         if count_up == 0 and count_down == 0:
        #             alpha = 0.5
        #         elif count_up < 3 or count_down < 3:
        #             alpha = 0.3
        #         else:
        #             alpha = 0.1

        #         # update pseudocosts up
        #         if not fix_inf and (1 - f) > 1e-6:
        #             new_pc_up = max(0, fix_delta) / max(1e-9, (1 - f))
        #             old = self.pseudocosts_up.get(edge, None)
        #             if old is None or math.isnan(old) or math.isinf(old):
        #                 self.pseudocosts_up[edge] = new_pc_up
        #             else:
        #                 self.pseudocosts_up[edge] = (1 - alpha) * old + alpha * new_pc_up
        #             self.counts_up[edge] = count_up + 1

        #         # update pseudocosts down
        #         if not exc_inf and f > 1e-6:
        #             new_pc_down = max(0, exc_delta) / max(1e-9, f)
        #             old = self.pseudocosts_down.get(edge, None)
        #             if old is None or math.isnan(old) or math.isinf(old):
        #                 self.pseudocosts_down[edge] = new_pc_down
        #             else:
        #                 self.pseudocosts_down[edge] = (1 - alpha) * old + alpha * new_pc_down
        #             self.counts_down[edge] = count_down + 1

        #         if not fix_inf and not exc_inf:
        #             scores.append((sb_score, edge, False, False))
        #         else:
        #             if fix_inf:
        #                 edges_to_exclude.add(edge)
        #             if exc_inf:
        #                 edges_to_fix.add(edge)

        #     # 5) Pseudocost scoring for reliable edges
        #     for edge in reliable_candidates:
        #         if shor_primal_solution is not None and edge in shor_primal_solution:
        #             f = shor_primal_solution[edge]
        #         else:
        #             f = self.get_fractional_value(edge)
        #         f = max(0.01, min(0.99, f))

        #         pc_up = self.pseudocosts_up.get(edge, 1.0)
        #         pc_down = self.pseudocosts_down.get(edge, 1.0)

        #         cu = self.counts_up.get(edge, 0)
        #         cd = self.counts_down.get(edge, 0)
        #         confidence_up = min(1.0, cu / (2 * self.reliability_eta))
        #         confidence_down = min(1.0, cd / (2 * self.reliability_eta))
        #         confidence = 0.5 * (confidence_up + confidence_down)

        #         # delta_up = pc_up * (1 - f)
        #         # delta_down = pc_down * f
        #         # geometric_mean = max(1e-9, delta_up * delta_down) ** 0.5
        #         # score = geometric_mean * (0.9 + 0.1 * confidence)
        #         # print("realiablescore", score)

        #         # scores.append((score, edge, False, False))
        #         delta_up = pc_up * (1 - f)
        #         delta_down = pc_down * f

        #         # Use product, like in strong branching, to match scale
        #         gain_up = max(delta_up, 0.0)
        #         gain_down = max(delta_down, 0.0)

        #         score = max(gain_up, 1e-6) * max(gain_down, 1e-6)

        #         # Optional: keep a small confidence modulation, but don't shrink the scale too much
        #         score *= (0.9 + 0.1 * confidence)

        #         # print("reliablescore", score)

        #         scores.append((score, edge, False, False))


        #     if not scores and not (edges_to_fix or edges_to_exclude):
        #         return None

        #     # 6) Forced decisions from infeasible SB sides
        #     if edges_to_fix or edges_to_exclude:
        #         if self.verbose:
        #             print(f"[RLB] FORCED CHILD: fix={edges_to_fix}, exclude={edges_to_exclude}")
        #         child = self.create_single_child(edges_to_fix, edges_to_exclude)
        #         # pick any representative edge for logging
        #         rep = list(edges_to_fix)[0] if edges_to_fix else list(edges_to_exclude)[0]
        #         return ([rep], child)

        #     # 7) Normal case: choose best score
        #     scores.sort(key=lambda x: x[0], reverse=True)
        #     best_score, best_edge, _, _ = scores[0]

        #     if self.verbose:
        #         print(f"[RLB] SELECTED best_edge={best_edge} score={best_score:.4f}")

        #     return [best_edge]
        elif self.branching_rule == "reliability":
            # 1) Candidate edges: current MST edges (no fractional needed)
            # mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
            # candidate_edges = [
            #     e for e in mst_edges
            #     if e not in self.fixed_edges
            #     and e not in self.excluded_edges
            #     and e not in self.branched_edges
            # ]

            # if not candidate_edges:
            #     return None

            # # 2) Split into unhistoried vs reliable, based on TOTAL observations
            # unhistoried = []
            # reliable_candidates = []
            # for e in candidate_edges:
            #     cu = self.counts_up.get(e, 0)
            #     cd = self.counts_down.get(e, 0)
            #     total = cu + cd
            #     if total >= self.reliability_eta:
            #         reliable_candidates.append(e)
            #     else:
            #         unhistoried.append(e)

            # # 3) Adaptive lookahead: how many edges to strong-branch
            # if self.depth < 5:
            #     max_sb_evals = self.lookahead_lambda
            # else:
            #     max_sb_evals = max(2, self.lookahead_lambda - 1)

            # # You can optionally order unhistoried by some heuristic, e.g. by edge weight/length.
            # # For now, we just take them as they come from MST.
            # unhistoried = unhistoried[:max_sb_evals]
            # 1) Fractional solution for prioritization
            shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()

            candidate_edges = []
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
                # most fractional first
                candidate_edges.sort(key=lambda e: abs(shor_primal_solution.get(e, 0.5) - 0.5))

            if shor_primal_solution is None or not candidate_edges:
                mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                candidate_edges = [
                    e for e in mst_edges
                    if e not in self.fixed_edges
                    and e not in self.excluded_edges
                    and e not in self.branched_edges
                ]

            if not candidate_edges:
                return None

            # 2) Split into unhistoried vs reliable, based on TOTAL observations
            unhistoried = []
            reliable_candidates = []
            for e in candidate_edges:
                cu = self.counts_up.get(e, 0)
                cd = self.counts_down.get(e, 0)

                # print("counts",  cu, cd)
                total = cu + cd
                if total >= self.reliability_eta:
                    reliable_candidates.append(e)
                else:
                    unhistoried.append(e)

            # 3) Adaptive lookahead: how many edges to strong-branch
            if self.depth < 5:
                max_sb_evals = self.lookahead_lambda
                # max_sb_evals = 1

            else:
                # max_sb_evals = max(2, self.lookahead_lambda - 1)
                max_sb_evals = 1


            # Strong branching only on the most fractional unhistoried edges
            if shor_primal_solution is not None:
                unhistoried.sort(key=lambda e: abs(shor_primal_solution.get(e, 0.5) - 0.5))
                reliable_candidates.sort(key=lambda e: abs(shor_primal_solution.get(e, 0.5) - 0.5))

            else:
                unhistoried.sort(key=lambda e: abs(self.get_fractional_value(e) - 0.5))
                reliable_candidates.sort(key=lambda e: abs(self.get_fractional_value(e) - 0.5))


            unhistoried = unhistoried[:max_sb_evals]
            edges_to_fix = set()
            edges_to_exclude = set()
            scores = []

            # 4) Strong branching on unhistoried edges (also updates pseudocosts)
            for edge in unhistoried:
                # Strong branching returns LB improvements for fix/exclude
                sb_score, fix_delta, exc_delta, fix_inf, exc_inf = self.calculate_strong_branching_score(edge)
                if self.verbose:
                    print("unhistoriedscore", edge, sb_score)

                # Get current counts
                count_up = self.counts_up.get(edge, 0)
                count_down = self.counts_down.get(edge, 0)

                # Learning rate (EMA over LB gains)
                if count_up == 0 and count_down == 0:
                    alpha = 0.5
                elif count_up < 3 or count_down < 3:
                    alpha = 0.3
                else:
                    alpha = 0.1

                # Update pseudocosts UP: use raw LB improvement (no division by f)
                if not fix_inf:
                    new_pc_up = max(0.0, fix_delta)
                    old = self.pseudocosts_up.get(edge, None)
                    if old is None or math.isnan(old) or math.isinf(old):
                        self.pseudocosts_up[edge] = new_pc_up
                    else:
                        self.pseudocosts_up[edge] = (1 - alpha) * old + alpha * new_pc_up
                    self.counts_up[edge] = count_up + 1

                # Update pseudocosts DOWN: same idea
                if not exc_inf:
                    new_pc_down = max(0.0, exc_delta)
                    old = self.pseudocosts_down.get(edge, None)
                    if old is None or math.isnan(old) or math.isinf(old):
                        self.pseudocosts_down[edge] = new_pc_down
                    else:
                        self.pseudocosts_down[edge] = (1 - alpha) * old + alpha * new_pc_down
                    self.counts_down[edge] = count_down + 1

                # Store SB score if both sides feasible
                if not fix_inf and not exc_inf:
                    scores.append((sb_score, edge, False, False))
                else:
                    if fix_inf:
                        edges_to_exclude.add(edge)
                    if exc_inf:
                        edges_to_fix.add(edge)

            # 5) Pseudocost scoring for reliable edges (no fractional x)
            for edge in reliable_candidates:
                pc_up = self.pseudocosts_up.get(edge, 0.0)
                pc_down = self.pseudocosts_down.get(edge, 0.0)

                cu = self.counts_up.get(edge, 0)
                cd = self.counts_down.get(edge, 0)
                confidence_up = min(1.0, cu / (2 * self.reliability_eta))
                confidence_down = min(1.0, cd / (2 * self.reliability_eta))
                confidence = 0.5 * (confidence_up + confidence_down)

                gain_up = max(pc_up, 0.0)
                gain_down = max(pc_down, 0.0)

                score = max(gain_up, 1e-6) * max(gain_down, 1e-6)
                score *= (0.9 + 0.1 * confidence)

                if self.verbose:
                    print("reliablescore", edge, score)

                scores.append((score, edge, False, False))

            if not scores and not (edges_to_fix or edges_to_exclude):
                return None

            # 6) Forced decisions from infeasible SB sides
            if edges_to_fix or edges_to_exclude:
                if self.verbose:
                    print(f"[RLB] FORCED CHILD: fix={edges_to_fix}, exclude={edges_to_exclude}")
                child = self.create_single_child(edges_to_fix, edges_to_exclude)
                rep = list(edges_to_fix)[0] if edges_to_fix else list(edges_to_exclude)[0]
                return ([rep], child)

            # 7) Normal case: choose best score
            scores.sort(key=lambda x: x[0], reverse=True)
            best_score, best_edge, _, _ = scores[0]

            if self.verbose:
                origin = "reliable" if best_edge in reliable_candidates else "unhistoried"
                print(f"[RLB] SELECTED best_edge={best_edge} from {origin} score={best_score:.4f}")

            return [best_edge]



              
        elif self.branching_rule == "hybrid_strong_fractional":

            # --- Adaptive criteria for choosing strong vs fractional branching ---
            # bu = self.best_upper_bound
            # lb = self.local_lower_bound
            # if math.isfinite(bu) and math.isfinite(lb):
            #     denom = max(abs(bu), abs(lb), 1.0)
            #     gap_ratio = (bu - lb) / denom
            # else:
            #     # treat as large gap early on when bounds aren't finite yet
            #     gap_ratio = 1.0

            use_strong_branching = (
                self.depth < 5
                # or (self.depth < 10 and (len(self.fixed_edges) + len(self.excluded_edges)) < 0.1 * len(self.edges))
                # or (gap_ratio > 0.99)  # large gap remaining (keep your threshold)
            )
            if use_strong_branching:
                # Strong branching phase with computational limits
                shor_primal_solution = self.lagrangian_solver.compute_weighted_average_solution()
                # shor_primal_solution = self.lagrangian_solver.compute_dantzig_wolfe_solution(self)


                if shor_primal_solution is not None:
                    # Normalize keys and drop non-finite weights
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
                if shor_primal_solution is None or not candidate_edges:
                    mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
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
                # max_sb_evals = max(2, min(5, 8 - self.depth)) if self.depth < 8 else 1
                max_sb_evals = max(2, 8 - self.depth)

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
                mst_edges = [tuple(sorted((u, v))) for u, v in self.lagrangian_solver.best_mst_edges]
                candidate_edges = [
                    e for e in mst_edges
                    if e not in self.fixed_edges
                    and e not in self.excluded_edges
                    and e not in self.branched_edges
                ]
                return [candidate_edges[0]] if candidate_edges else None

            candidates = [
                e for e in shor_primal_solution
                if e not in self.fixed_edges
                and e not in self.excluded_edges
                and e not in self.branched_edges
            ]

            if not candidates:
                return None

            # Score by distance from 0.5 (most fractional first)
            branching_scores = []
            for e in candidates:
                distance_score = -abs(shor_primal_solution.get(e, 0.5) - 0.5)  # Negative so sorting descending gives smallest distance first
                branching_scores.append((e, distance_score))

            branching_scores.sort(key=lambda x: x[1], reverse=True)

            return [branching_scores[0][0]]

            
       
        else:
            raise ValueError(f"Unknown branching rule: {self.branching_rule}")

   
    def create_single_child(self, edges_to_fix, edges_to_exclude):
        """
        Fully consistent with create_children:
        - Correct (S, rhs) projection: S_free, rhs'
        - Redundant cut removal
        - Infeasibility detection
        - Support-based μ remapping
        - max_child_cuts limiting
        - Proper state propagation (depth, pseudocosts, reliability, etc.)
        """

        print("Creating single child with cuts projection")
        solver = self.lagrangian_solver
        edge_indices = solver.edge_indices
        known_edges = set(edge_indices.keys())

        idx_to_edge = getattr(solver, "idx_to_edge", None)
        if idx_to_edge is None:
            idx_to_edge = {j: e for e, j in edge_indices.items()}
            solver.idx_to_edge = idx_to_edge

        max_child_cuts = getattr(self, "max_child_cuts", 25)

        # --- Normalize edges ---
        def _norm_edge(e):
            if not (isinstance(e, tuple) and len(e) == 2):
                return None
            a, b = e
            t = (a, b) if a <= b else (b, a)
            return t if t in known_edges else None

        # --- Normalize any cut representation into edge set ---
        def _iter_edges_any(x):
            if isinstance(x, tuple) and len(x) == 2:
                e = _norm_edge(x)
                if e: yield e
                return
            if isinstance(x, int):
                e_raw = idx_to_edge.get(int(x))
                e = _norm_edge(e_raw)
                if e: yield e
                return
            try:
                for item in x:
                    if isinstance(item, int):
                        e_raw = idx_to_edge.get(item)
                        e = _norm_edge(e_raw)
                    elif isinstance(item, tuple) and len(item) == 2:
                        e = _norm_edge(item)
                    elif isinstance(item, (list, set, frozenset)) and len(item) == 2:
                        e = _norm_edge(tuple(item))
                    else:
                        e = None
                    if e: yield e
            except TypeError:
                return

        def _norm_pair(pair):
            cut_like, rhs_like = pair
            return (set(_iter_edges_any(cut_like)), int(rhs_like))

        # ---- Merge active cuts + new cuts ----
        all_cuts = []
        for p in (self.active_cuts or []):
            all_cuts.append(_norm_pair(p))
        for p in (getattr(self, "new_cuts", []) or []):
            all_cuts.append(_norm_pair(p))

        # ---- μ mapping from solver.best_cuts ----
        best_cuts = getattr(solver, "best_cuts", []) or []
        best_mu   = getattr(solver, "best_cut_multipliers_for_best_bound", {}) or {}

        support_to_mu = {}
        for i, (cut_i, rhs_i) in enumerate(best_cuts):
            support_to_mu[(frozenset(cut_i), rhs_i)] = float(best_mu.get(i, 0.0))

        # each cut in all_cuts receives μ by support
        current_multipliers = {}
        for idx, (cut, rhs) in enumerate(all_cuts):
            key = (frozenset(cut), rhs)
            current_multipliers[idx] = support_to_mu.get(key, 0.00001)

        # --- Final fixed / excluded sets ---
        child_fixed = set(self.fixed_edges)
        for e in edges_to_fix:
            ne = _norm_edge(e)
            if ne: child_fixed.add(ne)

        child_excl = set(self.excluded_edges)
        for e in edges_to_exclude:
            ne = _norm_edge(e)
            if ne: child_excl.add(ne)

        new_branched_edges = self.branched_edges | child_fixed | child_excl

        # --- Project cuts exactly like create_children ---
        def _project_for_child(fixed_child, excluded_child):
            infeasible = False
            proj = {}

            for old_i, (S, rhs) in enumerate(all_cuts):
                S_known = {e for e in S if e in known_edges}
                S_fixed = S_known & fixed_child
                S_free  = S_known - fixed_child - excluded_child

                rhs_prime = rhs - len(S_fixed)
                if rhs_prime < 0:
                    infeasible = True
                    break
                if len(S_free) <= rhs_prime:
                    continue

                key = frozenset(S_free)
                mu_old = float(current_multipliers.get(old_i, 0.0))

                prev = proj.get(key)
                if (prev is None or
                    rhs_prime < prev[0] or
                    (rhs_prime == prev[0] and abs(mu_old) > abs(prev[1]))):
                    proj[key] = (rhs_prime, mu_old)

            if infeasible:
                return None, None, True

            # sort like create_children
            def _key(sfree, rhs_, mu_):
                return (-len(sfree), rhs_, tuple(sorted(sfree)))

            ordered = sorted(
                ((sfree, rhs_mu[0], rhs_mu[1]) for sfree, rhs_mu in proj.items()),
                key=lambda x: _key(x[0], x[1], x[2])
            )

            if len(ordered) > max_child_cuts:
                ordered = ordered[:max_child_cuts]

            kept_cuts = [(set(sfree), rhs) for (sfree, rhs, mu) in ordered]
            kept_mu   = {i: float(mu) for i, (_, _, mu) in enumerate(ordered)}

            return kept_cuts, kept_mu, False

        kept_cuts, kept_mu, prune = _project_for_child(child_fixed, child_excl)
        if prune:
            return None

        # ---- Create the child (identical arguments as create_children) ----
        child = MSTNode(
            self.edges, self.num_nodes, self.budget,
            fixed_edges=child_fixed,
            excluded_edges=child_excl,
            branched_edges=new_branched_edges,
            initial_lambda=solver.best_lambda if self.inherit_lambda else 0.4,
            inherit_lambda=self.inherit_lambda,
            branching_rule=self.branching_rule,
            step_size=solver.step_size if self.inherit_step_size else 0.001,
            inherit_step_size=self.inherit_step_size,
            use_cover_cuts=self.use_cover_cuts,
            cut_frequency=self.cut_frequency,
            node_cut_frequency=self.node_cut_frequency,
            parent_cover_cuts=kept_cuts,
            parent_cover_multipliers=kept_mu,
            use_bisection=self.use_bisection,
            max_iter=solver.max_iter,
            verbose=self.verbose,
            depth=self.depth + 1,
            pseudocosts_up=self.pseudocosts_up,
            pseudocosts_down=self.pseudocosts_down,
            counts_up=self.counts_up,
            counts_down=self.counts_down,
            reliability_eta=self.reliability_eta,
            lookahead_lambda=self.lookahead_lambda,
        )

        return child



   

    def get_modified_weight(self, edge):
        u, v = tuple(sorted(edge))
        w, l = self.lagrangian_solver.edge_attributes[(u, v)]

        modified = w + self.lagrangian_solver.best_lambda * l
        
        for cut_idx, (cut, _) in enumerate(self.active_cuts):
            if (u, v) in cut:
                modified += self.cut_multipliers.get(cut_idx, 0)
        
        return modified

   
    

   

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
                cut_multipliers[idx] = self.cut_multipliers.get(pidx, 0.00001) if pidx is not None else 0.00001
        else:
            for idx in range(len(all_cuts)):
                cut_multipliers[idx] = 0.00001

        # --- Borrow, reset, (optionally) set warm-start state via attributes, then solve ---
        with self._sb_pool.borrow() as sim_solver:
                # --- Sync cut-related parameters from the main solver to the SB solver ---
            if hasattr(self.lagrangian_solver, "max_cut_depth"):
                sim_solver.max_cut_depth = getattr(self.lagrangian_solver, "max_cut_depth")
            if hasattr(self.lagrangian_solver, "extra_iter_for_cuts"):
                sim_solver.extra_iter_for_cuts = getattr(self.lagrangian_solver, "extra_iter_for_cuts")
            if hasattr(self.lagrangian_solver, "min_cut_violation_for_add"):
                sim_solver.min_cut_violation_for_add = getattr(
                    self.lagrangian_solver, "min_cut_violation_for_add"
                )

            sim_solver.reset(
                fixed_edges=new_fixed,
                excluded_edges=new_excluded,
                initial_lambda=getattr(self.lagrangian_solver, "best_lambda", getattr(self, "initial_lambda", 0.4)),
                step_size=getattr(self.lagrangian_solver, "step_size", getattr(self, "step_size", 0.001)),
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
            # lower_bound, upper_bound, _info = sim_solver.solve(
            #     inherited_cuts=all_cuts,
            #     inherited_multipliers=cut_multipliers
            # )
            # Depth for the probe: children of this node → depth + 1
            probe_depth = getattr(self, "depth", 0) + 1

            lower_bound, upper_bound, _info = sim_solver.solve(
                inherited_cuts=all_cuts,
                inherited_multipliers=cut_multipliers,
                depth=probe_depth,          # <<< IMPORTANT
            )


        return float(lower_bound)


    def calculate_strong_branching_score(self, edge):
        """
        Fast strong branching using simulation instead of full child creation.

        Returns:
            (score, fix_delta, exc_delta, fix_infeasible, exclude_infeasible)
        """
        u, v = tuple(sorted(edge))

        # --- Strong-branching probe: FIX edge ---
        fixed_lower_bound = self.simulate_branching_bound(edge, fix_edge=True, max_iters=5)
        fix_infeasible = math.isnan(fixed_lower_bound) or math.isinf(fixed_lower_bound)

        if self.verbose and fix_infeasible:
            print(f"Fixed simulation for edge {edge} is infeasible (LB={fixed_lower_bound})")

        # --- Strong-branching probe: EXCLUDE edge ---
        excluded_lower_bound = self.simulate_branching_bound(edge, fix_edge=False, max_iters=5)
        exclude_infeasible = math.isnan(excluded_lower_bound) or math.isinf(excluded_lower_bound)

        if self.verbose and exclude_infeasible:
            print(f"Excluded simulation for edge {edge} is infeasible (LB={excluded_lower_bound})")

        # If both directions are infeasible, this edge is useless as a branching candidate
        if fix_infeasible and exclude_infeasible:
            if self.verbose:
                print(f"Edge {edge}: both branches infeasible in strong branching probe")
            # Worst possible score, deltas 0 (won't affect pseudocosts either)
            return -float("inf"), 0.0, 0.0, True, True

        # --- Compute LB improvements (relative to current node LB) ---
        # For infeasible side, treat delta as 0 for logging/pseudocosts (we don't use it when *_infeasible is True).
        fix_delta = (fixed_lower_bound - self.local_lower_bound) if not fix_infeasible else 0.0
        exc_delta = (excluded_lower_bound - self.local_lower_bound) if not exclude_infeasible else 0.0

        # Only positive improvements should contribute to the score
        fix_gain = max(fix_delta, 0.0)
        exc_gain = max(exc_delta, 0.0)

        # --- Score ---
        # If one side is infeasible and the other is feasible, we want a "forced" decision.
        # The hybrid / reliability code already treats `fix_infeasible` / `exclude_infeasible`
        # as forcing edges_to_exclude / edges_to_fix, so we can just give any finite score here.
        if not fix_infeasible and not exclude_infeasible:
            # Your original product-based score, but using gains and a small epsilon
            score = max(fix_gain, 1e-6) * max(exc_gain, 1e-6)
        else:
            # One side infeasible → the calling code will handle the forcing;
            # we don't rely on the numeric score for ranking in that case.
            score = float("inf")

        if self.verbose:
            print(
                f"Edge {edge}: "
                f"score={score:.6g}, "
                f"fix_LB={fixed_lower_bound:.6g}, exc_LB={excluded_lower_bound:.6g}, "
                f"Δfix={fix_delta:.6g}, Δexc={exc_delta:.6g}, "
                f"fix_inf={fix_infeasible}, exc_inf={exclude_infeasible}"
            )

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
    
