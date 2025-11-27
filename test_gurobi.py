

# import random
# from mstkpinstance import MSTKPInstance  # Import the class from mstkpinstance.py
# import pickle
# import gurobipy as gp
# from gurobipy import GRB
# import time
# import pandas as pd
# import os
# from datetime import datetime
# import argparse
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Status mapping for readability (optional)
# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
# }

# def parse_arguments():
#     parser = argparse.ArgumentParser(prog='Gurobi MST Knapsack Benchmark', usage='%(prog)s [options]')
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed (default: 42)"
#     )
#     parser.add_argument(
#         "--num-nodes",
#         type=int,
#         default=50,
#         help="The number of nodes in the graph (default: 50)"
#     )
#     parser.add_argument(
#         "--density",
#         type=float,
#         default=0.3,
#         help="The density of the graph (default: 0.3)"
#     )
#     parser.add_argument(
#         "--num-instances",
#         type=int,
#         default=5,
#         help="Number of instances to generate and solve (default: 5)"
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="/Users/ssha0224/Desktop",
#         help="Directory to save results and instances (default: /Users/ssha0224/Desktop)"
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Enable verbose output (default: False)"
#     )
#     return parser.parse_args()

# # Generate instances (adapted from benchmark_mstkp.py)
# def generate_instances(num_instances, num_nodes, density, seed, output_dir):
#     random.seed(seed)
#     instances = []
#     for i in range(num_instances):
#         start_gen = time.time()
#         instance_seed = random.randint(0, 1000000)  # Random seed per instance for variety
#         random.seed(instance_seed)
#         instance = MSTKPInstance(num_nodes, density)  # Will print edges and budget here
#         gen_time = time.time() - start_gen
#         instances.append((instance, instance_seed, gen_time))
#         logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")
    
#     # Save to pickle
#     os.makedirs(output_dir, exist_ok=True)
#     instances_path = os.path.join(output_dir, "instances.pkl")
#     with open(instances_path, 'wb') as f:
#         pickle.dump(instances, f)
#     logger.info(f"Saved instances to {instances_path}")
#     return instances

# # Always generate new instances
# def get_instances(args):
#     logger.info("Generating new instances based on provided seed.")
#     return generate_instances(args.num_instances, args.num_nodes, args.density, args.seed, args.output_dir)

# # Solve MST + Knapsack using Multi Commodity Flow formulation with Gurobi
# def solve_with_gurobi(instance, seed, verbose=False):
#     start_solve = time.time()
#     V = list(range(instance.num_nodes))  # Nodes 0 to n-1
#     E = [(min(u,v), max(u,v)) for u, v, _, _ in instance.edges]  # Undirected edges
#     c = {(min(u,v), max(u,v)): w for u, v, w, _ in instance.edges}  # Costs
#     w_knap = {(min(u,v), max(u,v)): l for u, v, _, l in instance.edges}  # Knapsack weights (lengths)
#     B = instance.budget
#     root = 0  # Choose node 0 as root

#     # Commodities: one for each non-root node
#     K = list(range(1, instance.num_nodes))  # Commodities 1 to n-1

#     # Directed arcs for flow (both directions)
#     A = [(i,j) for i in V for j in V if i != j and (min(i,j), max(i,j)) in E]

#     # Gurobi model
#     model = gp.Model("MST_Knapsack_MultiCommodityFlow")
#     model.setParam("OutputFlag", 1 if verbose else 0)  # Verbose logging
#     model.setParam("TimeLimit", 3600)  # 1 hour limit
#     model.setParam("MIPGap", 0.001)  # Small gap for convergence

#     # Variables
#     x = model.addVars(E, vtype=GRB.BINARY, name="x")  # Edge selection
#     f = model.addVars([(i,j,k) for (i,j) in A for k in K], vtype=GRB.CONTINUOUS, lb=0, name="f")  # Flow on arcs for each commodity

#     # Objective: Minimize sum c_e * x_e
#     model.setObjective(gp.quicksum(c[e] * x[e] for e in E), GRB.MINIMIZE)

#     # Constraints
#     # 1. Exactly |V|-1 edges in the tree
#     model.addConstr(gp.quicksum(x[e] for e in E) == len(V) - 1, name="tree_size")

#     # 2. Knapsack constraint: sum w_e * x_e <= B
#     model.addConstr(gp.quicksum(w_knap[e] * x[e] for e in E) <= B, name="knapsack")

#     # 3. Flow conservation for each commodity k
#     for k in K:
#         for i in V:
#             inflow = gp.quicksum(f[(j,i,k)] for j in V if (j,i) in A)
#             outflow = gp.quicksum(f[(i,j,k)] for j in V if (i,j) in A)
#             if i == root:
#                 # Root supplies 1 unit for commodity k
#                 model.addConstr(outflow - inflow == 1, name=f"flow_supply_{k}")
#             elif i == k:
#                 # Sink k demands 1 unit
#                 model.addConstr(outflow - inflow == -1, name=f"flow_demand_{k}_{i}")
#             else:
#                 # Intermediate nodes: conservation
#                 model.addConstr(outflow - inflow == 0, name=f"flow_balance_{k}_{i}")

#     # 4. Flow capacity: f_{ij}^k + f_{ji}^k <= x_e for each commodity k and each undirected edge e={i,j}
#     for e in E:
#         u, v = e
#         for k in K:
#             model.addConstr(f[(u,v,k)] + f[(v,u,k)] <= x[e] if (u,v) in A and (v,u) in A else 0, name=f"capacity_{u}_{v}_{k}")

#     # Optimize
#     start_opt = time.time()
#     model.optimize()
#     opt_time = time.time() - start_opt

#     # Collect results with error handling
#     obj_val = model.objVal if model.status == GRB.OPTIMAL else float('inf')
#     selected_edges = [e for e in E if x[e].x > 0.99] if model.status == GRB.OPTIMAL else []
#     nodes_explored = model.NodeCount if hasattr(model, 'NodeCount') else 0
#     gap = model.MIPGap if hasattr(model, 'MIPGap') else float('inf')
#     status_str = STATUS_MAP.get(model.status, f"Unknown ({model.status})")  # Use mapping for string

#     solve_time = time.time() - start_solve

#     logger.info(f"Optimal objective: {obj_val}")
#     logger.info(f"Selected edges: {selected_edges}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Optimization status: {status_str}")

#     result = {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,
#         "solve_time": solve_time,
#         "opt_time": opt_time,
#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": model.status,  # Save as integer (for CSV)
#         "status_str": status_str,  # Human-readable
#         "selected_edges": selected_edges
#     }
#     return result

# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg({
#         "solve_time": ["mean", "std"],
#         "opt_time": ["mean", "std"],
#         "nodes_explored": ["mean", "std"],
#         "best_objective": ["mean", "std"],
#         "mip_gap": ["mean", "std"]
#     }).round(2)
#     return summary

# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
#     os.makedirs(output_dir, exist_ok=True)

#     # Get instances (load if exist, else generate)
#     instances = get_instances(args)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
#         result = solve_with_gurobi(instance, instance_seed, verbose=args.verbose)
#         result["gen_time"] = gen_time
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     # Save results
#     final_path = os.path.join(output_dir, "gurobi_results.csv")
#     df = pd.DataFrame(results)
#     df.to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     # Analyze results
#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(output_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved Gurobi summary statistics to {summary_path}")
#     print("\nGurobi Summary Statistics:")
#     print(summary)

# if __name__ == "__main__":
#     main()

#########################################################singlr
import random
from mstkpinstance import MSTKPInstance  # Import the class from mstkpinstance.py
import pickle
import gurobipy as gp
from gurobipy import GRB
import time
import pandas as pd
import os
from datetime import datetime
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Status mapping for readability (optional)
STATUS_MAP = {
    GRB.OPTIMAL: "Optimal",
    GRB.INFEASIBLE: "Infeasible",
    GRB.UNBOUNDED: "Unbounded",
    GRB.TIME_LIMIT: "Time Limit",
    GRB.INF_OR_UNBD: "Infeasible or Unbounded",
}

def parse_arguments():
    parser = argparse.ArgumentParser(prog='Gurobi MST Knapsack Benchmark', usage='%(prog)s [options]')
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=50,
        help="The number of nodes in the graph (default: 50)"
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1,
        help="The density of the graph (default: 0.3)"
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=5,
        help="Number of instances to generate and solve (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/ssha0224/Desktop",
        help="Directory to save results and instances (default: /Users/ssha0224/Desktop)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (default: False)"
    )
    return parser.parse_args()

# Generate instances (adapted from benchmark_mstkp.py)
def generate_instances(num_instances, num_nodes, density, seed, output_dir):
    random.seed(seed)
    instances = []
    for i in range(num_instances):
        start_gen = time.time()
        instance_seed = random.randint(0, 1000000)  # Random seed per instance for variety
        random.seed(instance_seed)
        instance = MSTKPInstance(num_nodes, density)  # Will print edges and budget here
        gen_time = time.time() - start_gen
        instances.append((instance, instance_seed, gen_time))
        logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")
    
    # Save to pickle
    os.makedirs(output_dir, exist_ok=True)
    instances_path = os.path.join(output_dir, "instances.pkl")
    with open(instances_path, 'wb') as f:
        pickle.dump(instances, f)
    logger.info(f"Saved instances to {instances_path}")
    return instances

# Always generate new instances
def get_instances(args):
    logger.info("Generating new instances based on provided seed.")
    return generate_instances(args.num_instances, args.num_nodes, args.density, args.seed, args.output_dir)

# Solve MST + Knapsack using Single Commodity Flow formulation with Gurobi
def solve_with_gurobi(instance, seed, verbose=False):
    start_solve = time.time()
    V = list(range(instance.num_nodes))  # Nodes 0 to n-1
    E = [(min(u,v), max(u,v)) for u, v, _, _ in instance.edges]  # Undirected edges
    c = {(min(u,v), max(u,v)): w for u, v, w, _ in instance.edges}  # Costs
    w_knap = {(min(u,v), max(u,v)): l for u, v, _, l in instance.edges}  # Knapsack weights (lengths)
    B = instance.budget
    root = 0  # Choose node 0 as root

    # Directed arcs for flow (both directions)
    A = [(i,j) for i in V for j in V if i != j and (min(i,j), max(i,j)) in E]

    # Gurobi model
    model = gp.Model("MST_Knapsack_SingleCommodityFlow")
    model.setParam("OutputFlag", 1 if verbose else 0)  # Verbose logging
    model.setParam("TimeLimit", 2700)  # 1 hour limit
    model.setParam("MIPGap", 0.003)  # Small gap for convergence

    # Variables
    x = model.addVars(E, vtype=GRB.BINARY, name="x")  # Edge selection
    f = model.addVars(A, vtype=GRB.CONTINUOUS, lb=0, name="f")  # Flow on directed arcs (single commodity)

    # Objective: Minimize sum c_e * x_e
    model.setObjective(gp.quicksum(c[e] * x[e] for e in E), GRB.MINIMIZE)

    # Constraints
    # 1. Exactly |V|-1 edges in the tree
    model.addConstr(gp.quicksum(x[e] for e in E) == len(V) - 1, name="tree_size")

    # 2. Knapsack constraint: sum w_e * x_e <= B
    model.addConstr(gp.quicksum(w_knap[e] * x[e] for e in E) <= B, name="knapsack")

    # 3. Flow conservation (single commodity)
    # Root supplies n-1 units
    model.addConstr(
        gp.quicksum(f[(root, j)] for j in V if (root, j) in A) -
        gp.quicksum(f[(j, root)] for j in V if (j, root) in A) == len(V) - 1,
        name="flow_supply_root"
    )
    # Each non-root node demands 1 unit
    for i in V:
        if i == root:
            continue
        inflow = gp.quicksum(f[(j, i)] for j in V if (j, i) in A)
        outflow = gp.quicksum(f[(i, j)] for j in V if (i, j) in A)
        model.addConstr(inflow - outflow == 1, name=f"flow_demand_{i}")

    # 4. Flow capacity: f_{ij} + f_{ji} <= (n-1) * x_e for each undirected edge e={i,j}
    for e in E:
        u, v = e
        model.addConstr(
            f[(u, v)] + f[(v, u)] <= (len(V) - 1) * x[e] if (u, v) in A and (v, u) in A else 0,
            name=f"capacity_{u}_{v}"
        )

    # Optimize
    start_opt = time.time()
    model.optimize()
    opt_time = time.time() - start_opt

    # Collect results with error handling
    obj_val = model.objVal if model.status == GRB.OPTIMAL else float('inf')
    selected_edges = [e for e in E if x[e].x > 0.99] if model.status == GRB.OPTIMAL else []
    nodes_explored = model.NodeCount if hasattr(model, 'NodeCount') else 0
    gap = model.MIPGap if hasattr(model, 'MIPGap') else float('inf')
    status_str = STATUS_MAP.get(model.status, f"Unknown ({model.status})")  # Use mapping for string

    solve_time = time.time() - start_solve

    logger.info(f"Optimal objective: {obj_val}")
    logger.info(f"Selected edges: {selected_edges}")
    logger.info(f"Nodes explored: {nodes_explored}")
    logger.info(f"MIP gap: {gap}")
    logger.info(f"Optimization status: {status_str}")

    result = {
        "instance_seed": seed,
        "num_nodes": instance.num_nodes,
        "density": instance.density,
        "budget": instance.budget,
        "solve_time": solve_time,
        "opt_time": opt_time,
        "nodes_explored": nodes_explored,
        "best_objective": obj_val,
        "mip_gap": gap,
        "status": model.status,  # Save as integer (for CSV)
        "status_str": status_str,  # Human-readable
        "selected_edges": selected_edges
    }
    return result

def analyze_gurobi_results(results):
    df = pd.DataFrame(results)
    summary = df.agg({
        "solve_time": ["mean", "std"],
        "opt_time": ["mean", "std"],
        "nodes_explored": ["mean", "std"],
        "best_objective": ["mean", "std"],
        "mip_gap": ["mean", "std"]
    }).round(2)
    return summary

def main():
    args = parse_arguments()
    random.seed(args.seed)

    output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    # Get instances (load if exist, else generate)
    instances = get_instances(args)

    results = []
    for idx, (instance, instance_seed, gen_time) in enumerate(instances):
        logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
        result = solve_with_gurobi(instance, instance_seed, verbose=args.verbose)
        result["gen_time"] = gen_time
        result["total_time"] = result["gen_time"] + result["solve_time"]
        results.append(result)

    # Save results
    final_path = os.path.join(output_dir, "gurobi_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(final_path, index=False)
    logger.info(f"Saved Gurobi results to {final_path}")

    # Analyze results
    summary = analyze_gurobi_results(results)
    summary_path = os.path.join(output_dir, "gurobi_summary.csv")
    summary.to_csv(summary_path)
    logger.info(f"Saved Gurobi summary statistics to {summary_path}")
    print("\nGurobi Summary Statistics:")
    print(summary)

if __name__ == "__main__":
    main()



#######################################cycle elimination
# import random
# from mstkpinstance import MSTKPInstance  # Import the class from mstkpinstance.py
# import pickle
# import gurobipy as gp
# from gurobipy import GRB
# import time
# import pandas as pd
# import os
# from datetime import datetime
# import argparse
# import logging
# import networkx as nx

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Status mapping for readability (optional)
# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
# }

# def parse_arguments():
#     parser = argparse.ArgumentParser(prog='Gurobi MST Knapsack Benchmark', usage='%(prog)s [options]')
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed (default: 42)"
#     )
#     parser.add_argument(
#         "--num-nodes",
#         type=int,
#         default=50,
#         help="The number of nodes in the graph (default: 50)"
#     )
#     parser.add_argument(
#         "--density",
#         type=float,
#         default=0.3,
#         help="The density of the graph (default: 0.3)"
#     )
#     parser.add_argument(
#         "--num-instances",
#         type=int,
#         default=5,
#         help="Number of instances to generate and solve (default: 5)"
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="/Users/ssha0224/Desktop",
#         help="Directory to save results and instances (default: /Users/ssha0224/Desktop)"
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Enable verbose output (default: False)"
#     )
#     return parser.parse_args()

# # Generate instances (adapted from benchmark_mstkp.py)
# def generate_instances(num_instances, num_nodes, density, seed, output_dir):
#     random.seed(seed)
#     instances = []
#     for i in range(num_instances):
#         start_gen = time.time()
#         instance_seed = random.randint(0, 1000000)  # Random seed per instance for variety
#         random.seed(instance_seed)
#         instance = MSTKPInstance(num_nodes, density)  # Will print edges and budget here
#         gen_time = time.time() - start_gen
#         instances.append((instance, instance_seed, gen_time))
#         logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")
    
#     # Save to pickle
#     os.makedirs(output_dir, exist_ok=True)
#     instances_path = os.path.join(output_dir, "instances.pkl")
#     with open(instances_path, 'wb') as f:
#         pickle.dump(instances, f)
#     logger.info(f"Saved instances to {instances_path}")
#     return instances

# # Always generate new instances
# def get_instances(args):
#     logger.info("Generating new instances based on provided seed.")
#     return generate_instances(args.num_instances, args.num_nodes, args.density, args.seed, args.output_dir)

# def subtour_elim(model, where):
#     if where == GRB.Callback.MIPSOL:
#         x_vals = model.cbGetSolution(model._x)
#         G = nx.Graph()
#         for e in model._E:
#             if x_vals[e] > 0.5:
#                 G.add_edge(e[0], e[1], weight=x_vals[e])
#         components = list(nx.connected_components(G))
#         for comp in components:
#             if len(comp) < model._n:
#                 edges_in_comp = [(u,v) for u,v in model._E if u in comp and v in comp]
#                 if sum(x_vals[e] for e in edges_in_comp) > len(comp) - 1:
#                     model.cbLazy(gp.quicksum(model._x[e] for e in edges_in_comp) <= len(comp) - 1)

# # Solve MST + Knapsack using Cycle Elimination formulation with Gurobi
# def solve_with_gurobi(instance, seed, verbose=False):
#     start_solve = time.time()
#     V = list(range(instance.num_nodes))  # Nodes 0 to n-1
#     E = [(min(u,v), max(u,v)) for u, v, _, _ in instance.edges]  # Undirected edges
#     c = {(min(u,v), max(u,v)): w for u, v, w, _ in instance.edges}  # Costs
#     w_knap = {(min(u,v), max(u,v)): l for u, v, _, l in instance.edges}  # Knapsack weights (lengths)
#     B = instance.budget

#     # Gurobi model
#     model = gp.Model("MST_Knapsack_CycleElimination")
#     model.setParam("OutputFlag", 1 if verbose else 0)  # Verbose logging
#     model.setParam("TimeLimit", 3600)  # 1 hour limit
#     model.setParam("MIPGap", 0.001)  # Small gap for convergence
#     model.setParam("LazyConstraints", 1)  # Enable lazy constraints

#     # Variables
#     model._x = model.addVars(E, vtype=GRB.BINARY, name="x")  # Edge selection
#     model._E = E
#     model._n = len(V)

#     # Objective: Minimize sum c_e * x_e
#     model.setObjective(gp.quicksum(c[e] * model._x[e] for e in E), GRB.MINIMIZE)

#     # Constraints
#     # 1. Exactly |V|-1 edges in the tree
#     model.addConstr(gp.quicksum(model._x[e] for e in E) == len(V) - 1, name="tree_size")

#     # 2. Knapsack constraint: sum w_e * x_e <= B
#     model.addConstr(gp.quicksum(w_knap[e] * model._x[e] for e in E) <= B, name="knapsack")

#     # Optimize with callback
#     model.optimize(subtour_elim)

#     # Collect results with error handling
#     obj_val = model.objVal if model.status == GRB.OPTIMAL else float('inf')
#     selected_edges = [e for e in E if model._x[e].x > 0.99] if model.status == GRB.OPTIMAL else []
#     nodes_explored = model.NodeCount if hasattr(model, 'NodeCount') else 0
#     gap = model.MIPGap if hasattr(model, 'MIPGap') else float('inf')
#     status_str = STATUS_MAP.get(model.status, f"Unknown ({model.status})")  # Use mapping for string

#     solve_time = time.time() - start_solve

#     logger.info(f"Optimal objective: {obj_val}")
#     logger.info(f"Selected edges: {selected_edges}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Optimization status: {status_str}")

#     result = {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,
#         "solve_time": solve_time,
#         "opt_time": solve_time,  # Since no separate opt_time measured
#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": model.status,  # Save as integer (for CSV)
#         "status_str": status_str,  # Human-readable
#         "selected_edges": selected_edges
#     }
#     return result

# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg({
#         "solve_time": ["mean", "std"],
#         "opt_time": ["mean", "std"],
#         "nodes_explored": ["mean", "std"],
#         "best_objective": ["mean", "std"],
#         "mip_gap": ["mean", "std"]
#     }).round(2)
#     return summary

# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
#     os.makedirs(output_dir, exist_ok=True)

#     # Get instances (load if exist, else generate)
#     instances = get_instances(args)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
#         result = solve_with_gurobi(instance, instance_seed, verbose=args.verbose)
#         result["gen_time"] = gen_time
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     # Save results
#     final_path = os.path.join(output_dir, "gurobi_results.csv")
#     df = pd.DataFrame(results)
#     df.to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     # Analyze results
#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(output_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved Gurobi summary statistics to {summary_path}")
#     print("\nGurobi Summary Statistics:")
#     print(summary)

# if __name__ == "__main__":
#     main()
###########################################tree cut set formulation
# import random
# import pickle
# import gurobipy as gp
# from gurobipy import GRB
# import time
# import pandas as pd
# import os
# from datetime import datetime
# import argparse
# import logging
# import networkx as nx

# # Note: Assuming MSTKPInstance is available from user's code
# from mstkpinstance import MSTKPInstance

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Status mapping for readability (optional)
# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
# }

# def parse_arguments():
#     parser = argparse.ArgumentParser(prog='Gurobi MST Knapsack Benchmark', usage='%(prog)s [options]')
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed (default: 42)"
#     )
#     parser.add_argument(
#         "--num-nodes",
#         type=int,
#         default=50,
#         help="The number of nodes in the graph (default: 50)"
#     )
#     parser.add_argument(
#         "--density",
#         type=float,
#         default=0.3,
#         help="The density of the graph (default: 0.3)"
#     )
#     parser.add_argument(
#         "--num-instances",
#         type=int,
#         default=5,
#         help="Number of instances to generate and solve (default: 5)"
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="/Users/ssha0224/Desktop",
#         help="Directory to save results and instances (default: /Users/ssha0224/Desktop)"
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Enable verbose output (default: False)"
#     )
#     return parser.parse_args()

# # Generate instances (adapted from benchmark_mstkp.py)
# def generate_instances(num_instances, num_nodes, density, seed, output_dir):
#     random.seed(seed)
#     instances = []
#     for i in range(num_instances):
#         start_gen = time.time()
#         instance_seed = random.randint(0, 1000000)  # Random seed per instance for variety
#         random.seed(instance_seed)
#         instance = MSTKPInstance(num_nodes, density)  # Will print edges and budget here
#         gen_time = time.time() - start_gen
#         instances.append((instance, instance_seed, gen_time))
#         logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")
    
#     # Save to pickle
#     os.makedirs(output_dir, exist_ok=True)
#     instances_path = os.path.join(output_dir, "instances.pkl")
#     with open(instances_path, 'wb') as f:
#         pickle.dump(instances, f)
#     logger.info(f"Saved instances to {instances_path}")
#     return instances

# # Always generate new instances
# def get_instances(args):
#     logger.info("Generating new instances based on provided seed.")
#     return generate_instances(args.num_instances, args.num_nodes, args.density, args.seed, args.output_dir)

# # Callback for cut-set constraints
# def cutset_callback(model, where):
#     if where == GRB.Callback.MIPSOL:
#         x_vals = model.cbGetSolution(model._x)
#         G = nx.Graph()
#         for e in model._E:
#             if x_vals[e] > 0.5:
#                 G.add_edge(e[0], e[1])
#         components = list(nx.connected_components(G))
#         for comp in components:
#             if 0 < len(comp) < model._n:
#                 cut_edges = [(min(u,v), max(u,v)) for u in comp for v in (model._V - set(comp)) if (min(u,v), max(u,v)) in model._E]
#                 if sum(x_vals[e] for e in cut_edges) < 1:
#                     model.cbLazy(gp.quicksum(model._x[e] for e in cut_edges) >= 1)

# # Solve MST + Knapsack using Tree Cut-Set formulation with Gurobi (10.2.3)
# def solve_with_gurobi(instance, seed, verbose=False):
#     start_solve = time.time()
#     V = set(range(instance.num_nodes))  # Nodes 0 to n-1 as set for difference
#     E = [(min(u,v), max(u,v)) for u, v, _, _ in instance.edges]  # Undirected edges
#     c = {(min(u,v), max(u,v)): w for u, v, w, _ in instance.edges}  # Costs
#     w_knap = {(min(u,v), max(u,v)): l for u, v, _, l in instance.edges}  # Knapsack weights (lengths)
#     B = instance.budget

#     # Gurobi model
#     model = gp.Model("MST_Knapsack_TreeCutSet")
#     model.setParam("OutputFlag", 1 if verbose else 0)  # Verbose logging
#     model.setParam("TimeLimit", 3600)  # 1 hour limit
#     model.setParam("MIPGap", 0.001)  # Small gap for convergence
#     model.setParam("LazyConstraints", 1)  # Enable lazy constraints

#     # Variables
#     model._x = model.addVars(E, vtype=GRB.BINARY, name="x")  # Edge selection
#     model._E = E
#     model._V = V
#     model._n = len(V)

#     # Objective: Minimize sum c_e * x_e
#     model.setObjective(gp.quicksum(c[e] * model._x[e] for e in E), GRB.MINIMIZE)

#     # Constraints
#     # 1. Exactly |V|-1 edges in the tree
#     model.addConstr(gp.quicksum(model._x[e] for e in E) == len(V) - 1, name="tree_size")

#     # 2. Knapsack constraint: sum w_e * x_e <= B
#     model.addConstr(gp.quicksum(w_knap[e] * model._x[e] for e in E) <= B, name="knapsack")

#     # Optimize with callback
#     model.optimize(cutset_callback)

#     # Collect results with error handling
#     obj_val = model.objVal if model.status == GRB.OPTIMAL else float('inf')
#     selected_edges = [e for e in E if model._x[e].x > 0.99] if model.status == GRB.OPTIMAL else []
#     nodes_explored = model.NodeCount if hasattr(model, 'NodeCount') else 0
#     gap = model.MIPGap if hasattr(model, 'MIPGap') else float('inf')
#     status_str = STATUS_MAP.get(model.status, f"Unknown ({model.status})")  # Use mapping for string

#     solve_time = time.time() - start_solve

#     logger.info(f"Optimal objective: {obj_val}")
#     logger.info(f"Selected edges: {selected_edges}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Optimization status: {status_str}")

#     result = {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,
#         "solve_time": solve_time,
#         "opt_time": solve_time,  # Since no separate opt_time measured
#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": model.status,  # Save as integer (for CSV)
#         "status_str": status_str,  # Human-readable
#         "selected_edges": selected_edges
#     }
#     return result

# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg({
#         "solve_time": ["mean", "std"],
#         "opt_time": ["mean", "std"],
#         "nodes_explored": ["mean", "std"],
#         "best_objective": ["mean", "std"],
#         "mip_gap": ["mean", "std"]
#     }).round(2)
#     return summary

# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
#     os.makedirs(output_dir, exist_ok=True)

#     # Get instances (load if exist, else generate)
#     instances = get_instances(args)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
#         result = solve_with_gurobi(instance, instance_seed, verbose=args.verbose)
#         result["gen_time"] = gen_time
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     # Save results
#     final_path = os.path.join(output_dir, "gurobi_results.csv")
#     df = pd.DataFrame(results)
#     df.to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     # Analyze results
#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(output_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved Gurobi summary statistics to {summary_path}")
#     print("\nGurobi Summary Statistics:")
#     print(summary)

# if __name__ == "__main__":
#     main()



####################################METHod 4
# import random
# from mstkpinstance import MSTKPInstance  # Import the class from mstkpinstance.py
# import pickle
# import gurobipy as gp
# from gurobipy import GRB
# import time
# import pandas as pd
# import os
# from datetime import datetime
# import argparse
# import logging
# import networkx as nx

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Status mapping for readability (optional)
# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
# }

# def parse_arguments():
#     parser = argparse.ArgumentParser(prog='Gurobi MST Knapsack Benchmark', usage='%(prog)s [options]')
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed (default: 42)"
#     )
#     parser.add_argument(
#         "--num-nodes",
#         type=int,
#         default=50,
#         help="The number of nodes in the graph (default: 50)"
#     )
#     parser.add_argument(
#         "--density",
#         type=float,
#         default=0.3,
#         help="The density of the graph (default: 0.3)"
#     )
#     parser.add_argument(
#         "--num-instances",
#         type=int,
#         default=5,
#         help="Number of instances to generate and solve (default: 5)"
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="/Users/ssha0224/Desktop",
#         help="Directory to save results and instances (default: /Users/ssha0224/Desktop)"
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Enable verbose output (default: False)"
#     )
#     return parser.parse_args()

# # Generate instances (adapted from benchmark_mstkp.py)
# def generate_instances(num_instances, num_nodes, density, seed, output_dir):
#     random.seed(seed)
#     instances = []
#     for i in range(num_instances):
#         start_gen = time.time()
#         instance_seed = random.randint(0, 1000000)  # Random seed per instance for variety
#         random.seed(instance_seed)
#         instance = MSTKPInstance(num_nodes, density)  # Will print edges and budget here
#         gen_time = time.time() - start_gen
#         instances.append((instance, instance_seed, gen_time))
#         logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")
    
#     # Save to pickle
#     os.makedirs(output_dir, exist_ok=True)
#     instances_path = os.path.join(output_dir, "instances.pkl")
#     with open(instances_path, 'wb') as f:
#         pickle.dump(instances, f)
#     logger.info(f"Saved instances to {instances_path}")
#     return instances

# # Always generate new instances
# def get_instances(args):
#     logger.info("Generating new instances based on provided seed.")
#     return generate_instances(args.num_instances, args.num_nodes, args.density, args.seed, args.output_dir)

# # Callback for arborescence cut-set constraints
# def arborescence_cutset_callback(model, where):
#     if where == GRB.Callback.MIPSOL:
#         f_vals = model.cbGetSolution(model._f)
#         G = nx.DiGraph()  # Directed graph for arborescence
#         for (i,j) in model._A:
#             if f_vals[(i,j)] > 0.5:
#                 G.add_edge(i, j)
#         # Find nodes reachable from root
#         reachable = set(nx.descendants(G, model._root)) | {model._root}
#         if len(reachable) < model._n:
#             unreachable = set(model._V) - reachable
#             # Cut: edges from reachable to unreachable
#             cut_edges = [(min(i,j), max(i,j)) for i in reachable for j in unreachable if (min(i,j), max(i,j)) in model._E]
#             if sum(model.cbGetSolution(model._x[e]) for e in cut_edges) < 1:
#                 model.cbLazy(gp.quicksum(model._x[e] for e in cut_edges) >= 1)

# # Solve MST + Knapsack using Arborescence Cut-Set formulation with Gurobi
# def solve_with_gurobi(instance, seed, verbose=False):
#     start_solve = time.time()
#     V = list(range(instance.num_nodes))  # Nodes 0 to n-1
#     E = [(min(u,v), max(u,v)) for u, v, _, _ in instance.edges]  # Undirected edges
#     c = {(min(u,v), max(u,v)): w for u, v, w, _ in instance.edges}  # Costs
#     w_knap = {(min(u,v), max(u,v)): l for u, v, _, l in instance.edges}  # Knapsack weights (lengths)
#     B = instance.budget
#     root = 0  # Choose node 0 as root

#     # Directed arcs for flow (both directions)
#     A = [(i,j) for i in V for j in V if i != j and (min(i,j), max(i,j)) in E]

#     # Gurobi model
#     model = gp.Model("MST_Knapsack_ArborescenceCutSet")
#     model.setParam("OutputFlag", 1 if verbose else 0)  # Verbose logging
#     model.setParam("TimeLimit", 3600)  # 1 hour limit
#     model.setParam("MIPGap", 0.001)  # Small gap for convergence
#     model.setParam("LazyConstraints", 1)  # Enable lazy constraints

#     # Variables
#     model._x = model.addVars(E, vtype=GRB.BINARY, name="x")  # Edge selection
#     model._f = model.addVars(A, vtype=GRB.CONTINUOUS, lb=0, name="f")  # Flow on directed arcs
#     model._E = E
#     model._A = A
#     model._V = V
#     model._n = len(V)
#     model._root = root

#     # Objective: Minimize sum c_e * x_e
#     model.setObjective(gp.quicksum(c[e] * model._x[e] for e in E), GRB.MINIMIZE)

#     # Constraints
#     # 1. Exactly |V|-1 edges in the tree
#     model.addConstr(gp.quicksum(model._x[e] for e in E) == len(V) - 1, name="tree_size")

#     # 2. Knapsack constraint: sum w_e * x_e <= B
#     model.addConstr(gp.quicksum(w_knap[e] * model._x[e] for e in E) <= B, name="knapsack")

#     # 3. Flow conservation for single commodity
#     # Root supplies n-1 units
#     model.addConstr(
#         gp.quicksum(model._f[(root, j)] for j in V if (root, j) in A) -
#         gp.quicksum(model._f[(j, root)] for j in V if (j, root) in A) == len(V) - 1,
#         name="flow_supply_root"
#     )
#     # Each non-root node demands 1 unit
#     for i in V:
#         if i == root:
#             continue
#         inflow = gp.quicksum(model._f[(j, i)] for j in V if (j, i) in A)
#         outflow = gp.quicksum(model._f[(i, j)] for j in V if (i, j) in A)
#         model.addConstr(inflow - outflow == 1, name=f"flow_demand_{i}")

#     # 4. Flow capacity: f_{ij} <= (n-1) * x_e for each directed arc (i,j)
#     for (i,j) in A:
#         e = (min(i,j), max(i,j))
#         model.addConstr(model._f[(i,j)] <= (len(V) - 1) * model._x[e], name=f"capacity_{i}_{j}")

#     # Optimize with callback
#     model.optimize(arborescence_cutset_callback)

#     # Collect results with error handling
#     obj_val = model.objVal if model.status == GRB.OPTIMAL else float('inf')
#     selected_edges = [e for e in E if model._x[e].x > 0.99] if model.status == GRB.OPTIMAL else []
#     nodes_explored = model.NodeCount if hasattr(model, 'NodeCount') else 0
#     gap = model.MIPGap if hasattr(model, 'MIPGap') else float('inf')
#     status_str = STATUS_MAP.get(model.status, f"Unknown ({model.status})")  # Use mapping for string

#     solve_time = time.time() - start_solve

#     logger.info(f"Optimal objective: {obj_val}")
#     logger.info(f"Selected edges: {selected_edges}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Optimization status: {status_str}")

#     result = {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,
#         "solve_time": solve_time,
#         "opt_time": solve_time,  # Since no separate opt_time measured
#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": model.status,  # Save as integer (for CSV)
#         "status_str": status_str,  # Human-readable
#         "selected_edges": selected_edges
#     }
#     return result

# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg({
#         "solve_time": ["mean", "std"],
#         "opt_time": ["mean", "std"],
#         "nodes_explored": ["mean", "std"],
#         "best_objective": ["mean", "std"],
#         "mip_gap": ["mean", "std"]
#     }).round(2)
#     return summary

# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
#     os.makedirs(output_dir, exist_ok=True)

#     # Get instances (load if exist, else generate)
#     instances = get_instances(args)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
#         result = solve_with_gurobi(instance, instance_seed, verbose=args.verbose)
#         result["gen_time"] = gen_time
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     # Save results
#     final_path = os.path.join(output_dir, "gurobi_results.csv")
#     df = pd.DataFrame(results)
#     df.to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     # Analyze results
#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(output_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved Gurobi summary statistics to {summary_path}")
#     print("\nGurobi Summary Statistics:")
#     print(summary)

# if __name__ == "__main__":
#     main()
#######################################method 6
# import random
# import pickle
# import gurobipy as gp
# from gurobipy import GRB
# import time
# import pandas as pd
# import os
# from datetime import datetime
# import argparse
# import logging
# import networkx as nx

# # Note: Assuming MSTKPInstance is available from user's code
# from mstkpinstance import MSTKPInstance

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Status mapping for readability (optional)
# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
# }

# def parse_arguments():
#     parser = argparse.ArgumentParser(prog='Gurobi MST Knapsack Benchmark', usage='%(prog)s [options]')
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed (default: 42)"
#     )
#     parser.add_argument(
#         "--num-nodes",
#         type=int,
#         default=50,
#         help="The number of nodes in the graph (default: 50)"
#     )
#     parser.add_argument(
#         "--density",
#         type=float,
#         default=0.3,
#         help="The density of the graph (default: 0.3)"
#     )
#     parser.add_argument(
#         "--num-instances",
#         type=int,
#         default=5,
#         help="Number of instances to generate and solve (default: 5)"
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="/Users/ssha0224/Desktop",
#         help="Directory to save results and instances (default: /Users/ssha0224/Desktop)"
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Enable verbose output (default: False)"
#     )
#     return parser.parse_args()

# # Generate instances (adapted from benchmark_mstkp.py)
# def generate_instances(num_instances, num_nodes, density, seed, output_dir):
#     random.seed(seed)
#     instances = []
#     for i in range(num_instances):
#         start_gen = time.time()
#         instance_seed = random.randint(0, 1000000)  # Random seed per instance for variety
#         random.seed(instance_seed)
#         instance = MSTKPInstance(num_nodes, density)  # Will print edges and budget here
#         gen_time = time.time() - start_gen
#         instances.append((instance, instance_seed, gen_time))
#         logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")
    
#     # Save to pickle
#     os.makedirs(output_dir, exist_ok=True)
#     instances_path = os.path.join(output_dir, "instances.pkl")
#     with open(instances_path, 'wb') as f:
#         pickle.dump(instances, f)
#     logger.info(f"Saved instances to {instances_path}")
#     return instances

# # Always generate new instances
# def get_instances(args):
#     logger.info("Generating new instances based on provided seed.")
#     return generate_instances(args.num_instances, args.num_nodes, args.density, args.seed, args.output_dir)

# # Solve MST + Knapsack using k-Arborescence formulation with Gurobi (10.2.6 - approximated as Multi-Commodity Flow with k commodities)
# def solve_with_gurobi(instance, seed, verbose=False):
#     start_solve = time.time()
#     V = list(range(instance.num_nodes))  # Nodes 0 to n-1
#     E = [(min(u,v), max(u,v)) for u, v, _, _ in instance.edges]  # Undirected edges
#     c = {(min(u,v), max(u,v)): w for u, v, w, _ in instance.edges}  # Costs
#     w_knap = {(min(u,v), max(u,v)): l for u, v, _, l in instance.edges}  # Knapsack weights (lengths)
#     B = instance.budget
#     root = 0  # Choose node 0 as root

#     # Commodities: one for each non-root node
#     K = list(range(1, instance.num_nodes))  # Commodities 1 to n-1

#     # Directed arcs for flow (both directions)
#     A = [(i,j) for i in V for j in V if i != j and (min(i,j), max(i,j)) in E]

#     # Gurobi model
#     model = gp.Model("MST_Knapsack_kArborescence")
#     model.setParam("OutputFlag", 1 if verbose else 0)  # Verbose logging
#     model.setParam("TimeLimit", 3600)  # 1 hour limit
#     model.setParam("MIPGap", 0.001)  # Small gap for convergence

#     # Variables
#     x = model.addVars(E, vtype=GRB.BINARY, name="x")  # Edge selection
#     f = model.addVars([(i,j,k) for (i,j) in A for k in K], vtype=GRB.CONTINUOUS, lb=0, name="f")  # Flow on arcs for each commodity

#     # Objective: Minimize sum c_e * x_e
#     model.setObjective(gp.quicksum(c[e] * x[e] for e in E), GRB.MINIMIZE)

#     # Constraints
#     # 1. Exactly |V|-1 edges in the tree
#     model.addConstr(gp.quicksum(x[e] for e in E) == len(V) - 1, name="tree_size")

#     # 2. Knapsack constraint: sum w_e * x_e <= B
#     model.addConstr(gp.quicksum(w_knap[e] * x[e] for e in E) <= B, name="knapsack")

#     # 3. Flow conservation for each commodity k
#     for k in K:
#         for i in V:
#             inflow = gp.quicksum(f[(j,i,k)] for j in V if (j,i) in A)
#             outflow = gp.quicksum(f[(i,j,k)] for j in V if (i,j) in A)
#             if i == root:
#                 # Root supplies 1 unit for commodity k
#                 model.addConstr(outflow - inflow == 1, name=f"flow_supply_{k}")
#             elif i == k:
#                 # Sink k demands 1 unit
#                 model.addConstr(outflow - inflow == -1, name=f"flow_demand_{k}_{i}")
#             else:
#                 # Intermediate nodes: conservation
#                 model.addConstr(outflow - inflow == 0, name=f"flow_balance_{k}_{i}")

#     # 4. Flow capacity: f_{ij}^k + f_{ji}^k <= x_e for each commodity k and each undirected edge e={i,j}
#     for e in E:
#         u, v = e
#         for k in K:
#             model.addConstr(
#                 f[(u, v, k)] + f[(v, u, k)] <= x[e] if (u, v) in A and (v, u) in A else 0,
#                 name=f"capacity_{u}_{v}_{k}"
#             )

#     # Optimize
#     start_opt = time.time()
#     model.optimize()
#     opt_time = time.time() - start_opt

#     # Collect results with error handling
#     obj_val = model.objVal if model.status == GRB.OPTIMAL else float('inf')
#     selected_edges = [e for e in E if x[e].x > 0.99] if model.status == GRB.OPTIMAL else []
#     nodes_explored = model.NodeCount if hasattr(model, 'NodeCount') else 0
#     gap = model.MIPGap if hasattr(model, 'MIPGap') else float('inf')
#     status_str = STATUS_MAP.get(model.status, f"Unknown ({model.status})")  # Use mapping for string

#     solve_time = time.time() - start_solve

#     logger.info(f"Optimal objective: {obj_val}")
#     logger.info(f"Selected edges: {selected_edges}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Optimization status: {status_str}")

#     result = {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,
#         "solve_time": solve_time,
#         "opt_time": opt_time,
#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": model.status,  # Save as integer (for CSV)
#         "status_str": status_str,  # Human-readable
#         "selected_edges": selected_edges
#     }
#     return result

# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg({
#         "solve_time": ["mean", "std"],
#         "opt_time": ["mean", "std"],
#         "nodes_explored": ["mean", "std"],
#         "best_objective": ["mean", "std"],
#         "mip_gap": ["mean", "std"]
#     }).round(2)
#     return summary

# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
#     os.makedirs(output_dir, exist_ok=True)

#     # Get instances (load if exist, else generate)
#     instances = get_instances(args)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
#         result = solve_with_gurobi(instance, instance_seed, verbose=args.verbose)
#         result["gen_time"] = gen_time
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     # Save results
#     final_path = os.path.join(output_dir, "gurobi_results.csv")
#     df = pd.DataFrame(results)
#     df.to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     # Analyze results
#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(output_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved Gurobi summary statistics to {summary_path}")
#     print("\nGurobi Summary Statistics:")
#     print(summary)

# if __name__ == "__main__":
#     main()