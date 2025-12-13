

import argparse
import random
import networkx as nx
from typing import List, Tuple, Optional
from mstkpbranchandbound import MSTNode
from branchandbound import RandomBranchingRule, BranchAndBound
from lagrangianrelaxation import LagrangianMST
from mstkpinstance import MSTKPInstance


def parse_arguments():
    parser = argparse.ArgumentParser(prog='MST Lagrangean B&B', usage='%(prog)s [options]')
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=100,
        help="The number of nodes in the graph (default: 100)"
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1,
        help="The density of the graph (default: 0.3)"
    )
    parser.add_argument(
        "rule",
        # choices=["random_mst", "random_all","random_fractional", "most_violated", "critical_edge", "most_fractional", "strong_branching", "strong_branching_sim","sb_fractional", "strong_branching_all"],
        choices=["random_mst", "random_all","random_fractional", "most_violated", "critical_edge", "most_fractional", "strong_branching", "strong_branching_sim","sb_fractional", "strong_branching_all", "reliability", "hybrid_strong_fractional"],
        help="The branching rule to use (random_mst: pick from MST edges, random_all: pick from all variables, most_fractional: pick the most fractional edge, strong_branching: use strong branching)"
    )
    parser.add_argument(
        "--inherit-lambda",
        action="store_true",
        help="Inherit lambda from the parent node (default: False)"
    )
    parser.add_argument(
        "--inherit-step-size",
        action="store_true",
        help="Inherit step size from the parent node (default: False)"
    )
    parser.add_argument(
        "--cover-cuts",
        action="store_true",
        help="Enable cover cuts generation (default: False)"
    )
    parser.add_argument(
        "--cut-frequency",
        type=int,
        default=5,
        help="Frequency of cut generation in Lagrangian iterations (default: 5)"
    )
    parser.add_argument(
        "--node-cut-frequency",
        type=int,
        default=10,
        help="Frequency of cut generation in B&B nodes (default: 10)"
    )
    parser.add_argument(
        "--use-bisection",
        action="store_true",
        help="Use bisection algorithm for updating the knapsack multiplier (default: False)"
    )
    parser.add_argument(
        "--use-2opt",
        action="store_true",
        help="Use 2-opt local-search heuristic to improve initial solution (default: False)"
    )
    parser.add_argument(
        "--use-shooting",
        action="store_true",
        help="Use shooting method to enhance branch-and-bound (default: False)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output (default: False)"
    )

    args = parser.parse_args()
    print(f"Using branching rule: {args.rule}")
    print(f"Inherit lambda: {args.inherit_lambda}")
    print(f"Use 2-opt local-search: {args.use_2opt}")
    print(f"Use shooting method: {args.use_shooting}")
    print(f"Verbose: {args.verbose}")
    return args

def two_opt_local_search(edges: List[Tuple[int, int, int, int]], num_nodes: int, budget: int, initial_edges: List[Tuple[int, int]], verbose: bool = False) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """
    Perform 2-opt local-search to improve the initial spanning tree, as described in Yamada et al.
    
    Args:
        edges: List of edges [(u, v, weight, length), ...]
        num_nodes: Number of nodes in the graph
        budget: Knapsack constraint budget (total length must not exceed this)
        initial_edges: Initial feasible spanning tree edges [(u, v), ...]
        verbose: Enable verbose debug output (default: False)
    
    Returns:
        Tuple[Optional[List[Tuple[int, int]]], float]: Improved edges and their total weight
    """
    # Initialize graph and edge attributes
    graph = nx.Graph()
    edge_attributes = {}
    for u, v, w, l in edges:
        graph.add_edge(u, v, weight=w, length=l)
        edge_attributes[(min(u, v), max(u, v))] = (w, l)

    def is_spanning_tree(edges: List[Tuple[int, int]]) -> bool:
        if len(edges) != num_nodes - 1:
            return False
        mst_graph = nx.Graph(edges)
        return nx.is_connected(mst_graph) and len(mst_graph.nodes) == num_nodes

    def compute_weight(edges: List[Tuple[int, int]]) -> int:
        return sum(edge_attributes[(min(u, v), max(u, v))][0] for u, v in edges)

    def compute_length(edges: List[Tuple[int, int]]) -> int:
        return sum(edge_attributes[(min(u, v), max(u, v))][1] for u, v in edges)

    def get_cycle(tree_edges: List[Tuple[int, int]], new_edge: Tuple[int, int]) -> List[Tuple[int, int]]:
        mst_graph = nx.Graph(tree_edges)
        mst_graph.add_edge(*new_edge)
        try:
            cycle = nx.find_cycle(mst_graph, source=new_edge[0])
            return [(u, v) for u, v in cycle]
        except nx.NetworkXNoCycle:
            return []

    # Validate initial solution
    if not is_spanning_tree(initial_edges) or compute_length(initial_edges) > budget:
        print("Initial solution is not feasible for 2-opt.")
        return None, float("inf")

    current_edges = initial_edges.copy()
    current_weight = compute_weight(current_edges)
    improved = True

    while improved:
        improved = False
        co_tree_edges = [(u, v) for u, v in graph.edges if (u, v) not in set(current_edges) and (v, u) not in set(current_edges)]
        
        for co_edge in co_tree_edges:
            cycle = get_cycle(current_edges, co_edge)
            if not cycle:
                continue
            
            for cycle_edge in cycle:
                if cycle_edge == co_edge or cycle_edge == (co_edge[1], co_edge[0]):
                    continue
                
                new_edges = [e for e in current_edges if e != cycle_edge and e != (cycle_edge[1], cycle_edge[0])]
                new_edges.append(co_edge)
                
                if not is_spanning_tree(new_edges):
                    continue
                
                new_length = compute_length(new_edges)
                if new_length > budget:
                    continue
                
                new_weight = compute_weight(new_edges)
                if new_weight < current_weight:
                    current_edges = new_edges
                    current_weight = new_weight
                    improved = True
                    if verbose:
                        print(f"2-opt improved solution: Weight = {current_weight}, Length = {new_length}")
                    break
            
            if improved:
                break

    return current_edges, current_weight

if __name__ == "__main__":
    args = parse_arguments()
    random.seed(args.seed)

    mstkp_instance = MSTKPInstance(args.num_nodes, args.density)
    root_node = MSTNode(
        mstkp_instance.edges,
        mstkp_instance.num_nodes,
        mstkp_instance.budget,
        initial_lambda=0.4,
        inherit_lambda=args.inherit_lambda,
        branching_rule=args.rule,
        step_size=0.001,
        inherit_step_size=args.inherit_step_size,
        use_cover_cuts=args.cover_cuts,
        cut_frequency=args.cut_frequency,
        node_cut_frequency=10,
        parent_cover_cuts=None,
        parent_cover_multipliers=None,
        use_bisection=args.use_bisection,
        reliability_eta=3,  # Or argparse it
        lookahead_lambda=4,
        # partial_iters=5
    )
   
    config = {
        "branching_rule": args.rule,
        "use_2opt": args.use_2opt,
        "use_shooting": args.use_shooting,
        "use_bisection": args.use_bisection
    }
    branching_rule = RandomBranchingRule()

    bnb_solver = BranchAndBound(
        branching_rule,
        verbose=args.verbose,
        config=config,
        instance_seed=args.seed
    )

    # Apply 2-opt local-search if enabled
    initial_solution = None
    initial_upper_bound = float("inf")
    if args.use_2opt:
        initial_solution, initial_upper_bound = two_opt_local_search(
            mstkp_instance.edges,
            mstkp_instance.num_nodes,
            mstkp_instance.budget,
            root_node.mst_edges,
            verbose=args.verbose  # Pass verbose argument
        )
        if initial_solution and args.verbose:
            print(f"2-opt local-search completed. Initial upper bound: {initial_upper_bound}")
            print("Edges in 2-opt solution:")
            for edge in initial_solution:
                print(edge)

    # Use shooting method if enabled, otherwise use standard B&B
    if args.use_shooting:
        lower_bound = root_node.local_lower_bound
        upper_bound = initial_upper_bound if args.use_2opt and initial_upper_bound < float("inf") else root_node.best_upper_bound
        best_solution, best_upper_bound = bnb_solver.solve_with_shooting(
            root_node,
            initial_lower_bound=lower_bound,
            initial_upper_bound=upper_bound,
            initial_solution=initial_solution if args.use_2opt else None
        )
    else:
        if args.use_2opt and initial_solution and initial_upper_bound < float("inf"):
            # bnb_solver.best_solution = MSTNode(
            #     mstkp_instance.edges,
            #     mstkp_instance.num_nodes,
            #     mstkp_instance.budget,
            #     fixed_edges=set(initial_solution)
            # )
            bnb_solver.best_solution_edges = list(initial_solution)
            bnb_solver.best_upper_bound = initial_upper_bound
            bnb_solver.best_upper_bound = initial_upper_bound
        best_solution, best_upper_bound = bnb_solver.solve(root_node)

    print(f"Optimal MST Cost within Budget: {best_upper_bound}")
    if best_solution:
        print("Edges in the Optimal MST:")
        # for edge in best_solution.mst_edges:
        #     print(edge)
        for edge in best_solution if isinstance(best_solution, list) else best_solution.mst_edges:
            print(edge)
    else:
        print("No feasible solution found.")

    print(f"Lagrangian MST time: {LagrangianMST.total_compute_time:.2f}s")

    ##############################################
#