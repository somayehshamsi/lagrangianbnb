
######10.2
# import random
# from mstkpinstance import MSTKPInstance  # Import the class from mstkpinstance.py
# import pickle
# import gurobipy as gp
# from gurobipy import GRB
# import time
# import pandas as pd
# import os
# import argparse
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Status mapping for readability (optional)
# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
#     GRB.INTERRUPTED: "Interrupted (user time limit)",  # NEW
# }


# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         prog='Gurobi MST Knapsack Benchmark',
#         usage='%(prog)s [options]'
#     )
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
#         default=1.0,  # CHANGED to match previous code
#         help="The density of the graph (default: 1.0)"
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
#         "--time-limit",
#         type=float,
#         default=1800.0,
#         help="Wall-clock time limit per instance in seconds (default: 1800)"
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
#         instance = MSTKPInstance(num_nodes, density)
#         gen_time = time.time() - start_gen
#         instances.append((instance, instance_seed, gen_time))
#         logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")

#     # Save to pickle (same behavior as your previous script)
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


# # === Callback to enforce global wall-clock limit on the whole solve ===
# def global_time_callback(model, where):
#     """
#     Gurobi callback: terminate when wall-clock from solver start
#     exceeds model._time_limit_s (set before optimize()).
#     """
#     if where in (GRB.Callback.MIP, GRB.Callback.MIPNODE, GRB.Callback.SIMPLEX):
#         now = time.time()
#         elapsed = now - model._start_total
#         if elapsed >= model._time_limit_s:
#             model.terminate()  # status will be GRB.INTERRUPTED


# # Solve MST + Knapsack using Multi Commodity Flow formulation with Gurobi
# def solve_with_gurobi(instance, seed, time_limit_s, verbose=False):
#     """
#     Multi-commodity flow formulation, but with the SAME experiment semantics as the previous script:
#     - global wall-clock limit (build + optimize) via callback
#     - single-thread (Threads=1)
#     - MIPGap aligned
#     - solve_time measures total wall-clock (build + optimize)
#     """
#     start_total = time.time()

#     V = list(range(instance.num_nodes))
#     E = [(min(u, v), max(u, v)) for u, v, _, _ in instance.edges]
#     c = {(min(u, v), max(u, v)): w for u, v, w, _ in instance.edges}
#     w_knap = {(min(u, v), max(u, v)): l for u, v, _, l in instance.edges}
#     B = instance.budget
#     root = 0

#     # Commodities: one for each non-root node
#     K = list(range(1, instance.num_nodes))

#     # Directed arcs for flow (both directions)
#     A = [(i, j) for i in V for j in V if i != j and (min(i, j), max(i, j)) in E]

#     # If we've already exceeded the budget before building the model
#     if time.time() - start_total >= time_limit_s:
#         total_time = time.time() - start_total
#         logger.warning("Time limit reached before model construction.")
#         return {
#             "instance_seed": seed,
#             "num_nodes": instance.num_nodes,
#             "density": instance.density,
#             "budget": instance.budget,
#             "solve_time": total_time,
#             "opt_time": 0.0,
#             "nodes_explored": 0,
#             "best_objective": float('inf'),
#             "mip_gap": float('inf'),
#             "status": GRB.TIME_LIMIT,
#             "status_str": "Time Limit (before optimize)",
#             "selected_edges": [],
#         }

#     # Gurobi model
#     model = gp.Model("MST_Knapsack_MultiCommodityFlow")
#     model.setParam("OutputFlag", 1 if verbose else 0)
#     model.setParam("Threads", 1)       # NEW: fairness/consistency
#     model.setParam("MIPGap", 0.003)    # CHANGED to match previous code
#     # IMPORTANT: no model.setParam("TimeLimit", ...) because we use callback global wall-clock

#     # Variables
#     x = model.addVars(E, vtype=GRB.BINARY, name="x")
#     f = model.addVars([(i, j, k) for (i, j) in A for k in K],
#                       vtype=GRB.CONTINUOUS, lb=0.0, name="f")

#     # Objective
#     model.setObjective(gp.quicksum(c[e] * x[e] for e in E), GRB.MINIMIZE)

#     # Constraints
#     # 1) Exactly |V|-1 edges
#     model.addConstr(gp.quicksum(x[e] for e in E) == len(V) - 1, name="tree_size")

#     # 2) Knapsack
#     model.addConstr(gp.quicksum(w_knap[e] * x[e] for e in E) <= B, name="knapsack")

#     # 3) Flow conservation for each commodity k
#     for k in K:
#         for i in V:
#             inflow = gp.quicksum(f[(j, i, k)] for j in V if (j, i) in A)
#             outflow = gp.quicksum(f[(i, j, k)] for j in V if (i, j) in A)
#             if i == root:
#                 model.addConstr(outflow - inflow == 1, name=f"flow_supply_{k}")
#             elif i == k:
#                 model.addConstr(outflow - inflow == -1, name=f"flow_demand_{k}_{i}")
#             else:
#                 model.addConstr(outflow - inflow == 0, name=f"flow_balance_{k}_{i}")

#     # 4) Capacity constraints (FIXED: no "else 0" constraint)
#     for (u, v) in E:
#         if (u, v) in A and (v, u) in A:
#             for k in K:
#                 model.addConstr(
#                     f[(u, v, k)] + f[(v, u, k)] <= x[(u, v)],
#                     name=f"capacity_{u}_{v}_{k}"
#                 )

#     # === GLOBAL WALL-CLOCK LIMIT via callback ===
#     model._start_total = start_total
#     model._time_limit_s = time_limit_s

#     # Optimize with callback
#     start_opt = time.time()
#     model.optimize(global_time_callback)
#     opt_time = time.time() - start_opt
#     solve_time = time.time() - start_total

#     # Collect results
#     if model.status == GRB.OPTIMAL:
#         obj_val = model.objVal
#         selected_edges = [e for e in E if x[e].X > 0.99]
#     else:
#         obj_val = float('inf')
#         selected_edges = []

#     nodes_explored = model.NodeCount if hasattr(model, "NodeCount") else 0
#     gap = model.MIPGap if hasattr(model, "MIPGap") else float('inf')
#     status_str = STATUS_MAP.get(model.status, f"Unknown ({model.status})")

#     logger.info(f"Optimal objective: {obj_val}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Optimization status: {status_str}")

#     return {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,
#         "solve_time": solve_time,
#         "opt_time": opt_time,
#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": model.status,
#         "status_str": status_str,
#         "selected_edges": selected_edges,
#     }


# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg({
#         "solve_time": ["mean", "std"],
#         "opt_time": ["mean", "std"],
#         "nodes_explored": ["mean", "std"],
#         "best_objective": ["mean", "std"],
#         "mip_gap": ["mean", "std"],
#     }).round(2)
#     return summary


# def summarize_for_paper(results, time_limit):
#     """
#     Build a one-row summary with:
#     Solved (%), PAR10, Time (all), Time (solved), Nodes (all), Nodes (solved)
#     consistent with your previous script.
#     """
#     df = pd.DataFrame(results)

#     # solved flag: only exact optimal solutions
#     df["solved"] = df["status"].isin([GRB.OPTIMAL])

#     # capped_time and PAR10
#     df["capped_time"] = df["solve_time"].clip(upper=time_limit)
#     df["par10"] = df["solve_time"].where(df["solved"], 10.0 * time_limit)

#     # nodes only for solved
#     df["nodes_solved"] = df["nodes_explored"].where(df["solved"])

#     solved_pct = 100.0 * df["solved"].mean()
#     par10_mean = df["par10"].mean()
#     time_all_mean = df["solve_time"].mean()
#     nodes_all_mean = df["nodes_explored"].mean()

#     time_solved_mean = df.loc[df["solved"], "solve_time"].mean()
#     nodes_solved_mean = df["nodes_solved"].mean()

#     summary_row = pd.DataFrame([{
#         "solved_pct": solved_pct,
#         "par10": par10_mean,
#         "time_all": time_all_mean,
#         "nodes_all": nodes_all_mean,
#         "time_solved": time_solved_mean,
#         "nodes_solved": nodes_solved_mean,
#     }])

#     return summary_row.round(2)


# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     # one folder per script seed (consistent with previous code)
#     run_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
#     os.makedirs(run_dir, exist_ok=True)

#     # Generate instances
#     instances = get_instances(args)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
#         result = solve_with_gurobi(
#             instance,
#             instance_seed,
#             time_limit_s=args.time_limit,
#             verbose=args.verbose
#         )
#         result["gen_time"] = gen_time
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     # Save per-instance results
#     final_path = os.path.join(run_dir, "gurobi_results.csv")
#     df = pd.DataFrame(results)
#     df.to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     # Basic debug summary
#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(run_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved Gurobi summary statistics to {summary_path}")

#     # Paper-style one-row summary (for aggregation across seeds)
#     paper_summary = summarize_for_paper(results, args.time_limit)
#     paper_summary_path = os.path.join(run_dir, "gurobi_summary_for_paper.csv")
#     paper_summary.to_csv(paper_summary_path, index=False)
#     logger.info(f"Saved paper-style Gurobi summary to {paper_summary_path}")

#     print("\nGurobi Summary Statistics:")
#     print(summary)
#     print("\nPaper-style summary:")
#     print(paper_summary)


# if __name__ == "__main__":
#     main()


#########################################################singlr, 2.1
# import random
# from mstkpinstance import MSTKPInstance  # Import the class from mstkpinstance.py
# import pickle
# import gurobipy as gp
# from gurobipy import GRB
# import time
# import pandas as pd
# import os
# import argparse
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Status mapping for readability (optional)
# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
#     GRB.INTERRUPTED: "Interrupted (user time limit)",  # NEW (for callback terminate)
# }


# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         prog='Gurobi MST Knapsack Benchmark',
#         usage='%(prog)s [options]'
#     )
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
#         default=1.0,  # make consistent
#         help="The density of the graph (default: 1.0)"
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
#         "--time-limit",
#         type=float,
#         default=1800.0,
#         help="Wall-clock time limit per instance in seconds (default: 1800)"
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Enable verbose Gurobi output (default: False)"
#     )
#     return parser.parse_args()


# # Generate instances (adapted from benchmark_mstkp.py)
# def generate_instances(num_instances, num_nodes, density, seed, output_dir):
#     random.seed(seed)
#     instances = []
#     for i in range(num_instances):
#         start_gen = time.time()
#         instance_seed = random.randint(0, 1000000)  # Random seed per instance
#         random.seed(instance_seed)
#         instance = MSTKPInstance(num_nodes, density)
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


# # Always generate new instances for this script
# def get_instances(args):
#     logger.info("Generating new instances based on provided seed.")
#     return generate_instances(args.num_instances, args.num_nodes,
#                               args.density, args.seed, args.output_dir)


# # === Callback to enforce global wall-clock limit on the whole solve ===
# def global_time_callback(model, where):
#     """
#     Gurobi callback: terminate when wall-clock from solver start
#     exceeds model._time_limit_s (set before optimize()).
#     """
#     if where in (GRB.Callback.MIP, GRB.Callback.MIPNODE, GRB.Callback.SIMPLEX):
#         now = time.time()
#         elapsed = now - model._start_total
#         if elapsed >= model._time_limit_s:
#             model.terminate()  # status will be GRB.INTERRUPTED


# # Solve MST + Knapsack using Single Commodity Flow formulation with Gurobi
# # (FORMULATION kept as-is; experiment semantics aligned)
# def solve_with_gurobi(instance, seed, time_limit_s, verbose=False):
#     """
#     Enforce a global wall-clock time limit (build + optimize) via callback,
#     single-thread fairness, and consistent outputs (same as previous scripts).
#     """
#     start_total = time.time()

#     V = list(range(instance.num_nodes))  # Nodes 0..n-1
#     E = [(min(u, v), max(u, v)) for u, v, _, _ in instance.edges]
#     c = {(min(u, v), max(u, v)): w for u, v, w, _ in instance.edges}
#     w_knap = {(min(u, v), max(u, v)): l for u, v, _, l in instance.edges}
#     B = instance.budget
#     root = 0

#     # Directed arcs for flow (both directions)
#     A = [(i, j) for i in V for j in V if i != j and (min(i, j), max(i, j)) in E]

#     # If we've already exceeded the limit before building the model
#     if time.time() - start_total >= time_limit_s:
#         total_time = time.time() - start_total
#         logger.warning("Time limit reached before model construction.")
#         return {
#             "instance_seed": seed,
#             "num_nodes": instance.num_nodes,
#             "density": instance.density,
#             "budget": instance.budget,
#             "solve_time": total_time,
#             "opt_time": 0.0,
#             "nodes_explored": 0,
#             "best_objective": float('inf'),
#             "mip_gap": float('inf'),
#             "status": GRB.TIME_LIMIT,
#             "status_str": "Time Limit (before optimize)",
#             "selected_edges": [],
#         }

#     # Gurobi model
#     model = gp.Model("MST_Knapsack_SingleCommodityFlow")
#     model.setParam("OutputFlag", 1 if verbose else 0)
#     model.setParam("Threads", 1)       # consistency/fairness
#     model.setParam("MIPGap", 0.003)    # match previous scripts
#     # IMPORTANT: do NOT set model.setParam("TimeLimit", ...) because we use callback global wall-clock

#     # Variables
#     x = model.addVars(E, vtype=GRB.BINARY, name="x")                 # Edge selection
#     f = model.addVars(A, vtype=GRB.CONTINUOUS, lb=0.0, name="f")     # Flow on arcs (single commodity)

#     # Objective: Minimize sum c_e * x_e
#     model.setObjective(gp.quicksum(c[e] * x[e] for e in E), GRB.MINIMIZE)

#     # Constraints (FORMULATION AS YOU WROTE IT)
#     # 1) Exactly |V|-1 edges
#     model.addConstr(gp.quicksum(x[e] for e in E) == len(V) - 1, name="tree_size")

#     # 2) Knapsack
#     model.addConstr(gp.quicksum(w_knap[e] * x[e] for e in E) <= B, name="knapsack")

#     # 3) Flow conservation
#     # Root supplies n-1 units
#     model.addConstr(
#         gp.quicksum(f[(root, j)] for j in V if (root, j) in A) -
#         gp.quicksum(f[(j, root)] for j in V if (j, root) in A) == len(V) - 1,
#         name="flow_supply_root"
#     )

#     # Each non-root node demands 1 unit
#     for i in V:
#         if i == root:
#             continue
#         inflow = gp.quicksum(f[(j, i)] for j in V if (j, i) in A)
#         outflow = gp.quicksum(f[(i, j)] for j in V if (i, j) in A)
#         model.addConstr(inflow - outflow == 1, name=f"flow_demand_{i}")

#     # 4) Capacity constraints (FIXED: avoid "else 0" constraint bug)
#     for (u, v) in E:
#         if (u, v) in A and (v, u) in A:
#             model.addConstr(
#                 f[(u, v)] + f[(v, u)] <= (len(V) - 1) * x[(u, v)],
#                 name=f"capacity_{u}_{v}"
#             )

#     # === GLOBAL WALL-CLOCK LIMIT via callback ===
#     model._start_total = start_total
#     model._time_limit_s = time_limit_s

#     # Optimize (with callback)
#     start_opt = time.time()
#     model.optimize(global_time_callback)
#     opt_time = time.time() - start_opt
#     solve_time = time.time() - start_total

#     # Collect results
#     if model.status == GRB.OPTIMAL:
#         obj_val = model.objVal
#         selected_edges = [e for e in E if x[e].X > 0.99]
#     else:
#         obj_val = float('inf')
#         selected_edges = []

#     nodes_explored = model.NodeCount if hasattr(model, "NodeCount") else 0
#     gap = model.MIPGap if hasattr(model, "MIPGap") else float('inf')
#     status_str = STATUS_MAP.get(model.status, f"Unknown ({model.status})")

#     logger.info(f"Optimal objective: {obj_val}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Optimization status: {status_str}")

#     return {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,
#         "solve_time": solve_time,
#         "opt_time": opt_time,
#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": model.status,
#         "status_str": status_str,
#         "selected_edges": selected_edges,
#     }


# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg({
#         "solve_time": ["mean", "std"],
#         "opt_time": ["mean", "std"],
#         "nodes_explored": ["mean", "std"],
#         "best_objective": ["mean", "std"],
#         "mip_gap": ["mean", "std"],
#     }).round(2)
#     return summary


# def summarize_for_paper(results, time_limit):
#     """
#     Build a one-row summary with:
#     Solved (%), PAR10, Time (all), Time (solved), Nodes (all), Nodes (solved)
#     consistent with your other scripts.
#     """
#     df = pd.DataFrame(results)

#     df["solved"] = df["status"].isin([GRB.OPTIMAL])

#     df["capped_time"] = df["solve_time"].clip(upper=time_limit)
#     df["par10"] = df["solve_time"].where(df["solved"], 10.0 * time_limit)

#     df["nodes_solved"] = df["nodes_explored"].where(df["solved"])

#     solved_pct = 100.0 * df["solved"].mean()
#     par10_mean = df["par10"].mean()
#     time_all_mean = df["solve_time"].mean()
#     nodes_all_mean = df["nodes_explored"].mean()

#     time_solved_mean = df.loc[df["solved"], "solve_time"].mean()
#     nodes_solved_mean = df["nodes_solved"].mean()

#     summary_row = pd.DataFrame([{
#         "solved_pct": solved_pct,
#         "par10": par10_mean,
#         "time_all": time_all_mean,
#         "nodes_all": nodes_all_mean,
#         "time_solved": time_solved_mean,
#         "nodes_solved": nodes_solved_mean,
#     }])

#     return summary_row.round(2)


# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     # one folder per script seed (consistent with previous code)
#     run_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
#     os.makedirs(run_dir, exist_ok=True)

#     # Generate instances
#     instances = get_instances(args)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
#         result = solve_with_gurobi(
#             instance,
#             instance_seed,
#             time_limit_s=args.time_limit,
#             verbose=args.verbose
#         )
#         result["gen_time"] = gen_time
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     # Save per-instance results
#     final_path = os.path.join(run_dir, "gurobi_results.csv")
#     df = pd.DataFrame(results)
#     df.to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     # Debug summary
#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(run_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved Gurobi summary statistics to {summary_path}")

#     # Paper-style one-row summary (for aggregation across seeds)
#     paper_summary = summarize_for_paper(results, args.time_limit)
#     paper_summary_path = os.path.join(run_dir, "gurobi_summary_for_paper.csv")
#     paper_summary.to_csv(paper_summary_path, index=False)
#     logger.info(f"Saved paper-style Gurobi summary to {paper_summary_path}")

#     print("\nGurobi Summary Statistics:")
#     print(summary)
#     print("\nPaper-style summary:")
#     print(paper_summary)


# if __name__ == "__main__":
#     main()


#######################################################################
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
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Status mapping for readability (optional)
# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
#     GRB.INTERRUPTED: "Interrupted (user time limit)",  # for callback terminate()
# }


# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         prog='Gurobi MST Knapsack Benchmark',
#         usage='%(prog)s [options]'
#     )
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
#         default=1.0,
#         help="The density of the graph (default: 1.0)"
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
#         "--time-limit",
#         type=float,
#         default=1800.0,
#         help="Wall-clock time limit per instance in seconds (default: 60)"
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Enable verbose Gurobi output (default: False)"
#     )
#     return parser.parse_args()


# # Generate instances (adapted from benchmark_mstkp.py)
# def generate_instances(num_instances, num_nodes, density, seed, output_dir):
#     random.seed(seed)
#     instances = []
#     for i in range(num_instances):
#         start_gen = time.time()
#         instance_seed = random.randint(0, 1000000)  # Random seed per instance
#         random.seed(instance_seed)
#         instance = MSTKPInstance(num_nodes, density)
#         gen_time = time.time() - start_gen
#         instances.append((instance, instance_seed, gen_time))
#         logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")

#     # Save to pickle (optional, but useful if you want to reuse instances)
#     os.makedirs(output_dir, exist_ok=True)
#     instances_path = os.path.join(output_dir, "instances.pkl")
#     with open(instances_path, 'wb') as f:
#         pickle.dump(instances, f)
#     logger.info(f"Saved instances to {instances_path}")
#     return instances


# # Always generate new instances for this script
# def get_instances(args):
#     logger.info("Generating new instances based on provided seed.")
#     return generate_instances(args.num_instances, args.num_nodes,
#                               args.density, args.seed, args.output_dir)


# # === Callback to enforce global wall-clock limit on the whole solve ===
# def global_time_callback(model, where):
#     """
#     Gurobi callback: terminate when wall-clock from solver start
#     exceeds model._time_limit_s (set before optimize()).
#     """
#     if where in (GRB.Callback.MIP, GRB.Callback.MIPNODE, GRB.Callback.SIMPLEX):
#         now = time.time()
#         elapsed = now - model._start_total
#         if elapsed >= model._time_limit_s:
#             model.terminate()  # status will be GRB.INTERRUPTED


# def solve_with_gurobi(instance, seed, time_limit_s, verbose=False):
#     """
#     Solve one MSTKP instance with Gurobi, enforcing a *global* wall-clock
#     time limit `time_limit_s` on the whole process (model build + optimize),
#     similar to how your own solver enforces its time limit.
#     """
#     start_total = time.time()

#     # Build basic data
#     V = list(range(instance.num_nodes))  # 0..n-1
#     E = [(min(u, v), max(u, v)) for u, v, _, _ in instance.edges]
#     c = {(min(u, v), max(u, v)): w for u, v, w, _ in instance.edges}
#     w_knap = {(min(u, v), max(u, v)): l for u, v, _, l in instance.edges}
#     B = instance.budget
#     root = 0

#     # Directed arcs for flow
#     A = [(i, j) for i in V for j in V if i != j and (min(i, j), max(i, j)) in E]

#     # If we've already exceeded the budget before building the model
#     if time.time() - start_total >= time_limit_s:
#         total_time = time.time() - start_total
#         logger.warning("Time limit reached before model construction.")
#         return {
#             "instance_seed": seed,
#             "num_nodes": instance.num_nodes,
#             "density": instance.density,
#             "budget": instance.budget,
#             "solve_time": total_time,
#             "opt_time": 0.0,
#             "nodes_explored": 0,
#             "best_objective": float('inf'),
#             "mip_gap": float('inf'),
#             "status": GRB.TIME_LIMIT,
#             "status_str": "Time Limit (before optimize)",
#             "selected_edges": [],
#         }

#     # Create model
#     model = gp.Model("MST_Knapsack_SingleCommodityFlow")
#     model.setParam("OutputFlag", 1 if verbose else 0)
#     model.setParam("Threads", 1)           # single-thread for fairness
#     model.setParam("MIPGap", 0.003)

#     # Variables
#     x = model.addVars(E, vtype=GRB.BINARY, name="x")          # edge selection
#     f = model.addVars(A, vtype=GRB.CONTINUOUS, lb=0.0, name="f")  # flow

#     # Objective: minimize sum c_e x_e
#     model.setObjective(gp.quicksum(c[e] * x[e] for e in E), GRB.MINIMIZE)

#     # 1. Tree size: |E| = |V| - 1
#     model.addConstr(gp.quicksum(x[e] for e in E) == len(V) - 1, name="tree_size")

#     # 2. Knapsack constraint
#     model.addConstr(gp.quicksum(w_knap[e] * x[e] for e in E) <= B, name="knapsack")

#     # 3. Flow conservation
#     # Root supplies n-1 units
#     model.addConstr(
#         gp.quicksum(f[(root, j)] for j in V if (root, j) in A) -
#         gp.quicksum(f[(j, root)] for j in V if (j, root) in A) == len(V) - 1,
#         name="flow_supply_root"
#     )
#     # Each non-root node demands 1 unit
#     for i in V:
#         if i == root:
#             continue
#         inflow = gp.quicksum(f[(j, i)] for j in V if (j, i) in A)
#         outflow = gp.quicksum(f[(i, j)] for j in V if (i, j) in A)
#         model.addConstr(inflow - outflow == 1, name=f"flow_demand_{i}")

#     # 4. Capacity constraints: f_uv + f_vu <= (n-1) x_e
#     for e in E:
#         u, v = e
#         if (u, v) in A and (v, u) in A:
#             model.addConstr(
#                 f[(u, v)] + f[(v, u)] <= (len(V) - 1) * x[e],
#                 name=f"capacity_{u}_{v}"
#             )

#     # === GLOBAL WALL-CLOCK LIMIT via callback ===
#     model._start_total = start_total
#     model._time_limit_s = time_limit_s

#     start_opt = time.time()
#     model.optimize(global_time_callback)
#     opt_time = time.time() - start_opt
#     total_time = time.time() - start_total

#     # Collect results
#     if model.status == GRB.OPTIMAL:
#         obj_val = model.objVal
#         selected_edges = [e for e in E if x[e].x > 0.99]
#     else:
#         obj_val = float('inf')
#         selected_edges = []

#     nodes_explored = model.NodeCount if hasattr(model, "NodeCount") else 0
#     gap = model.MIPGap if hasattr(model, "MIPGap") else float('inf')
#     status_str = STATUS_MAP.get(model.status, f"Unknown ({model.status})")

#     logger.info(f"Optimal objective: {obj_val}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Optimization status: {status_str}")

#     result = {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,

#         # Full process time (what you compare to your solver's total_time)
#         "solve_time": total_time,
#         # Time inside model.optimize() only (diagnostic)
#         "opt_time": opt_time,

#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": model.status,
#         "status_str": status_str,
#         "selected_edges": selected_edges,
#     }
#     return result


# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg({
#         "solve_time": ["mean", "std"],
#         "opt_time": ["mean", "std"],
#         "nodes_explored": ["mean", "std"],
#         "best_objective": ["mean", "std"],
#         "mip_gap": ["mean", "std"],
#     }).round(2)
#     return summary


# def summarize_for_paper(results, time_limit):
#     """
#     Build a one-row summary with:
#     Solved (%), PAR10, Time (all), Time (solved), Nodes (solved)
#     analogous to your MSTKP tables.
#     """
#     df = pd.DataFrame(results)

#     # 1) solved flag: only exact optimal solutions
#     df["solved"] = df["status"].isin([GRB.OPTIMAL])

#     # 2) capped_time and PAR10, same logic as your solver:
#     #    capped_time = min(total_time, limit)
#     #    par10 = total_time if solved else 10*limit
#     df["capped_time"] = df["solve_time"].clip(upper=time_limit)
#     df["par10"] = df["solve_time"].where(df["solved"],
#                                          10.0 * time_limit)

#     # 3) nodes only for solved instances
#     df["nodes_solved"] = df["nodes_explored"].where(df["solved"])

#     # 4) Aggregates
#     solved_pct = 100.0 * df["solved"].mean()
#     par10_mean = df["par10"].mean()
#     time_all_mean = df["solve_time"].mean()
#     nodes_all_mean = df["nodes_explored"].mean()   

#     time_solved_mean = df.loc[df["solved"], "solve_time"].mean()
#     nodes_solved_mean = df["nodes_solved"].mean()

#     summary_row = pd.DataFrame([{
#         "solved_pct": solved_pct,
#         "par10": par10_mean,
#         "time_all": time_all_mean,
#         "nodes_all": nodes_all_mean,         
#         "time_solved": time_solved_mean,
#         "nodes_solved": nodes_solved_mean,
#     }])

#     return summary_row.round(2)


# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     # one folder per script seed (the one you pass in --seed)
#     run_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
#     os.makedirs(run_dir, exist_ok=True)


#     # Generate instances
#     instances = get_instances(args)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
#         result = solve_with_gurobi(
#             instance,
#             instance_seed,
#             time_limit_s=args.time_limit,
#             verbose=args.verbose
#         )
#         result["gen_time"] = gen_time
#         # If you ever want “generation + solve”:
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     # Save per-instance results
#     final_path = os.path.join(run_dir, "gurobi_results.csv")
#     df = pd.DataFrame(results)
#     df.to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     # Basic debug summary
#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(run_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved Gurobi summary statistics to {summary_path}")

#     # Paper-style single-row summary (for comparison with your solver)
#     paper_summary = summarize_for_paper(results, args.time_limit)
#     paper_summary_path = os.path.join(run_dir, "gurobi_summary_for_paper.csv")
#     paper_summary.to_csv(paper_summary_path, index=False)
#     logger.info(f"Saved paper-style Gurobi summary to {paper_summary_path}")

#     print("\nGurobi Summary Statistics:")
#     print(summary)
#     print("\nPaper-style summary:")
#     print(paper_summary)


# if __name__ == "__main__":
#     main()


#######################################cycle elimination, 10-2-5
# import random
# import pickle
# import gurobipy as gp
# from gurobipy import GRB
# import time
# import pandas as pd
# import os
# import argparse
# import logging
# import itertools
# import networkx as nx

# from mstkpinstance import MSTKPInstance  # your class

# # ---------------- Logging ----------------
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
#     GRB.INTERRUPTED: "Interrupted (callback terminate)",
# }

# # ---------------- CLI ----------------
# def parse_arguments():
#     parser = argparse.ArgumentParser(prog="Gurobi MST Knapsack Benchmark", usage="%(prog)s [options]")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
#     parser.add_argument("--num-nodes", type=int, default=50, help="Number of nodes (default: 50)")
#     parser.add_argument("--density", type=float, default=1.0, help="Graph density (default: 1.0)")
#     parser.add_argument("--num-instances", type=int, default=5, help="Number of instances (default: 5)")
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="/Users/ssha0224/Desktop",
#         help="Base output directory (default: /Users/ssha0224/Desktop)",
#     )
#     parser.add_argument("--time-limit", type=float, default=1800.0, help="Time limit per instance (seconds)")
#     parser.add_argument("--verbose", action="store_true", help="Enable Gurobi output")
#     return parser.parse_args()

# # ---------------- Instance generation ----------------
# def generate_instances(num_instances, num_nodes, density, seed, outdir_for_this_run):
#     """
#     Generate instances and save them under outdir_for_this_run (seed-specific folder),
#     so parallel runs don't overwrite each other.
#     """
#     random.seed(seed)
#     instances = []
#     for i in range(num_instances):
#         start_gen = time.time()
#         instance_seed = random.randint(0, 1_000_000)
#         random.seed(instance_seed)
#         instance = MSTKPInstance(num_nodes, density)
#         gen_time = time.time() - start_gen
#         instances.append((instance, instance_seed, gen_time))
#         logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")

#     os.makedirs(outdir_for_this_run, exist_ok=True)
#     instances_path = os.path.join(outdir_for_this_run, "instances.pkl")
#     with open(instances_path, "wb") as f:
#         pickle.dump(instances, f)
#     logger.info(f"Saved instances to {instances_path}")
#     return instances

# def get_instances(args, run_dir):
#     logger.info("Generating new instances based on provided seed.")
#     return generate_instances(
#         args.num_instances, args.num_nodes, args.density, args.seed, run_dir
#     )

# # ---------------- Callbacks ----------------
# def global_time_callback(model, where):
#     """
#     Terminate when global wall-clock (build + optimize) exceeds model._time_limit_s.
#     Include MIPSOL too, because that's where cbLazy is called and time can also be exceeded there.
#     """
#     if where in (GRB.Callback.MIP, GRB.Callback.MIPNODE, GRB.Callback.SIMPLEX, GRB.Callback.MIPSOL):
#         elapsed = time.time() - model._start_total
#         if elapsed >= model._time_limit_s:
#             model.terminate()

# def cycle_elim_lazy_callback(model, where):
#     """
#     Lazy constraints to eliminate cycles:
#       For any proper subset S that forms a connected component in the selected edges,
#       enforce sum_{e in E(S)} x_e <= |S| - 1
#     """
#     global_time_callback(model, where)

#     if where != GRB.Callback.MIPSOL:
#         return

#     x_vals = model.cbGetSolution(model._x)

#     # Build graph from selected edges in incumbent
#     G = nx.Graph()
#     G.add_nodes_from(range(model._n))
#     for (u, v) in model._E:
#         if x_vals[(u, v)] > 0.5:
#             G.add_edge(u, v)

#     components = list(nx.connected_components(G))

#     for comp in components:
#         if len(comp) == model._n:
#             continue  # whole graph

#         # In a component S, if selected edges >= |S|, there is a cycle.
#         nodes = sorted(comp)
#         lhs = G.subgraph(nodes).number_of_edges()
#         rhs = len(nodes) - 1

#         if lhs > rhs + 1e-6:
#             # Build all edges inside S that exist in the graph/model (need all E(S), not only selected edges)
#             edges_in_comp = []
#             for i, j in itertools.combinations(nodes, 2):
#                 e = (i, j)  # model edges are stored with i < j
#                 if e in model._Eset:
#                     edges_in_comp.append(e)

#             if edges_in_comp:
#                 model.cbLazy(gp.quicksum(model._x[e] for e in edges_in_comp) <= rhs)

# # ---------------- Solve ----------------
# def _extract_unique_edges(instance):
#     """
#     Build a unique undirected edge set E with canonical orientation (u < v),
#     plus cost c[e] and knapsack weight w_knap[e].
#     This avoids duplicate keys crashing model.addVars().
#     """
#     edge_data = {}
#     for rec in instance.edges:
#         # Expect (u, v, cost, knap_weight)
#         if len(rec) != 4:
#             raise ValueError(f"Expected instance.edges records of length 4, got: {rec}")

#         u, v, cost, wlen = rec
#         a, b = (u, v) if u < v else (v, u)
#         if a == b:
#             continue

#         # Keep first occurrence; warn if duplicates disagree
#         if (a, b) in edge_data:
#             old_cost, old_wlen = edge_data[(a, b)]
#             if abs(old_cost - float(cost)) > 1e-9 or abs(old_wlen - float(wlen)) > 1e-9:
#                 logger.warning(f"Duplicate edge ({a},{b}) with inconsistent data; keeping first.")
#             continue

#         edge_data[(a, b)] = (float(cost), float(wlen))

#     E = list(edge_data.keys())
#     c = {e: edge_data[e][0] for e in E}
#     w_knap = {e: edge_data[e][1] for e in E}
#     return E, c, w_knap

# def solve_with_gurobi(instance, seed, time_limit_s, verbose=False):
#     """
#     MSTKP via cycle-elimination lazy constraints.
#     - Global wall-clock limit via callback (build + optimize)
#     - Threads=1, MIPGap=0.003
#     """
#     start_total = time.time()

#     # If already over limit (unlikely, but consistent with your pattern)
#     if time.time() - start_total >= time_limit_s:
#         total_time = time.time() - start_total
#         logger.warning("Time limit reached before model construction.")
#         return {
#             "instance_seed": seed,
#             "num_nodes": instance.num_nodes,
#             "density": instance.density,
#             "budget": instance.budget,
#             "solve_time": total_time,
#             "opt_time": 0.0,
#             "nodes_explored": 0,
#             "best_objective": float("inf"),
#             "mip_gap": float("inf"),
#             "status": GRB.TIME_LIMIT,
#             "status_str": "Time Limit (before optimize)",
#             "selected_edges": [],
#         }

#     n = int(instance.num_nodes)
#     B = float(instance.budget)

#     # Build unique E, c, w_knap (prevents duplicate-key crashes)
#     E, c, w_knap = _extract_unique_edges(instance)

#     model = gp.Model("MST_Knapsack_CycleElimination")
#     model.setParam("OutputFlag", 1 if verbose else 0)
#     model.setParam("Threads", 1)
#     model.setParam("MIPGap", 0.003)
#     model.setParam("LazyConstraints", 1)
#     # Do NOT set model TimeLimit; we enforce global wall-clock ourselves.

#     # Variables
#     x = model.addVars(E, vtype=GRB.BINARY, name="x")

#     # Objective
#     model.setObjective(gp.quicksum(c[e] * x[e] for e in E), GRB.MINIMIZE)

#     # Constraints: size + knapsack
#     model.addConstr(gp.quicksum(x[e] for e in E) == n - 1, name="tree_size")
#     model.addConstr(gp.quicksum(w_knap[e] * x[e] for e in E) <= B, name="knapsack")

#     # Attach for callback
#     model._x = x
#     model._E = E
#     model._Eset = set(E)
#     model._n = n
#     model._start_total = start_total
#     model._time_limit_s = time_limit_s

#     # Optimize
#     start_opt = time.time()
#     model.optimize(cycle_elim_lazy_callback)
#     opt_time = time.time() - start_opt
#     solve_time = time.time() - start_total

#     # Collect results robustly (even if time-limited but feasible incumbent exists)
#     status = model.status
#     status_str = STATUS_MAP.get(status, f"Unknown ({status})")

#     nodes_explored = float(model.NodeCount) if hasattr(model, "NodeCount") else 0.0
#     gap = float(model.MIPGap) if hasattr(model, "MIPGap") else float("inf")

#     if getattr(model, "SolCount", 0) > 0:
#         obj_val = float(model.objVal)
#         selected_edges = [e for e in E if x[e].X > 0.99]
#     else:
#         obj_val = float("inf")
#         selected_edges = []

#     logger.info(f"Objective: {obj_val}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Status: {status_str}")

#     return {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,
#         "solve_time": solve_time,
#         "opt_time": opt_time,
#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": status,
#         "status_str": status_str,
#         "selected_edges": selected_edges,
#     }

# # ---------------- Analysis helpers ----------------
# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg(
#         {
#             "solve_time": ["mean", "std"],
#             "opt_time": ["mean", "std"],
#             "nodes_explored": ["mean", "std"],
#             "best_objective": ["mean", "std"],
#             "mip_gap": ["mean", "std"],
#         }
#     ).round(2)
#     return summary

# def summarize_for_paper(results, time_limit):
#     df = pd.DataFrame(results)

#     df["solved"] = df["status"].isin([GRB.OPTIMAL])
#     df["capped_time"] = df["solve_time"].clip(upper=time_limit)
#     df["par10"] = df["solve_time"].where(df["solved"], 10.0 * time_limit)
#     df["nodes_solved"] = df["nodes_explored"].where(df["solved"])

#     solved_pct = 100.0 * df["solved"].mean()
#     par10_mean = df["par10"].mean()
#     time_all_mean = df["solve_time"].mean()
#     nodes_all_mean = df["nodes_explored"].mean()

#     time_solved_mean = df.loc[df["solved"], "solve_time"].mean()
#     nodes_solved_mean = df["nodes_solved"].mean()

#     summary_row = pd.DataFrame(
#         [
#             {
#                 "solved_pct": solved_pct,
#                 "par10": par10_mean,
#                 "time_all": time_all_mean,
#                 "nodes_all": nodes_all_mean,
#                 "time_solved": time_solved_mean,
#                 "nodes_solved": nodes_solved_mean,
#             }
#         ]
#     )
#     return summary_row.round(2)

# # ---------------- Main ----------------
# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     # Seed-specific run directory to avoid collisions in parallel runs
#     run_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
#     os.makedirs(run_dir, exist_ok=True)

#     # Generate instances (saved under run_dir)
#     instances = get_instances(args, run_dir)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")

#         try:
#             result = solve_with_gurobi(
#                 instance,
#                 instance_seed,
#                 time_limit_s=args.time_limit,
#                 verbose=args.verbose,
#             )
#         except Exception as e:
#             # Prevent whole job from crashing; record failure cleanly
#             logger.exception(f"Exception while solving instance seed {instance_seed}: {e}")
#             result = {
#                 "instance_seed": instance_seed,
#                 "num_nodes": instance.num_nodes,
#                 "density": instance.density,
#                 "budget": instance.budget,
#                 "solve_time": float("nan"),
#                 "opt_time": float("nan"),
#                 "nodes_explored": 0,
#                 "best_objective": float("inf"),
#                 "mip_gap": float("inf"),
#                 "status": -1,
#                 "status_str": f"Exception: {type(e).__name__}",
#                 "selected_edges": [],
#             }

#         result["gen_time"] = gen_time
#         # If solve_time is nan due to exception, total_time will be nan too (fine)
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     # Save per-instance results
#     final_path = os.path.join(run_dir, "gurobi_results.csv")
#     df = pd.DataFrame(results)
#     df.to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     # Debug summary
#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(run_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved Gurobi summary statistics to {summary_path}")

#     # Paper-style one-row summary
#     paper_summary = summarize_for_paper(results, args.time_limit)
#     paper_summary_path = os.path.join(run_dir, "gurobi_summary_for_paper.csv")
#     paper_summary.to_csv(paper_summary_path, index=False)
#     logger.info(f"Saved paper-style Gurobi summary to {paper_summary_path}")

#     print("\nGurobi Summary Statistics:")
#     print(summary)
#     print("\nPaper-style summary:")
#     print(paper_summary)

# if __name__ == "__main__":
#     main()



###########################################tree cut set formulation, 10-2-3
# import random
# import pickle
# import gurobipy as gp
# from gurobipy import GRB
# import time
# import pandas as pd
# import os
# import argparse
# import logging
# import networkx as nx

# from mstkpinstance import MSTKPInstance

# # Configure logging (aligned with your other benchmark scripts)
# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# # Status mapping (aligned)
# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
#     GRB.INTERRUPTED: "Interrupted (user time limit)",  # for callback terminate()
# }


# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         prog="Gurobi MST Knapsack Benchmark",
#         usage="%(prog)s [options]"
#     )
#     parser.add_argument("--seed", type=int, default=42,
#                         help="Random seed (default: 42)")
#     parser.add_argument("--num-nodes", type=int, default=50,
#                         help="The number of nodes in the graph (default: 50)")
#     parser.add_argument("--density", type=float, default=1.0,
#                         help="The density of the graph (default: 1.0)")
#     parser.add_argument("--num-instances", type=int, default=5,
#                         help="Number of instances to generate and solve (default: 5)")
#     parser.add_argument("--output-dir", type=str, default="/Users/ssha0224/Desktop",
#                         help="Directory to save results and instances (default: /Users/ssha0224/Desktop)")
#     parser.add_argument("--time-limit", type=float, default=1800.0,
#                         help="Wall-clock time limit per instance in seconds (default: 1800)")
#     parser.add_argument("--verbose", action="store_true",
#                         help="Enable verbose Gurobi output (default: False)")
#     return parser.parse_args()


# # Generate instances (same behavior as your other scripts)
# def generate_instances(num_instances, num_nodes, density, seed, output_dir):
#     random.seed(seed)
#     instances = []
#     for i in range(num_instances):
#         start_gen = time.time()
#         instance_seed = random.randint(0, 1000000)
#         random.seed(instance_seed)
#         instance = MSTKPInstance(num_nodes, density)
#         gen_time = time.time() - start_gen
#         instances.append((instance, instance_seed, gen_time))
#         logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")

#     # Save instances.pkl (same behavior as your other scripts)
#     os.makedirs(output_dir, exist_ok=True)
#     instances_path = os.path.join(output_dir, "instances.pkl")
#     with open(instances_path, "wb") as f:
#         pickle.dump(instances, f)
#     logger.info(f"Saved instances to {instances_path}")
#     return instances


# def get_instances(args):
#     logger.info("Generating new instances based on provided seed.")
#     return generate_instances(args.num_instances, args.num_nodes, args.density, args.seed, args.output_dir)


# # === Global wall-clock limit callback (same semantics as your other scripts) ===
# def global_time_callback(model, where):
#     """
#     Terminate when wall-clock from solver start exceeds model._time_limit_s.
#     This enforces a *global* limit (build + optimize).
#     """
#     if where in (GRB.Callback.MIP, GRB.Callback.MIPNODE, GRB.Callback.SIMPLEX):
#         elapsed = time.time() - model._start_total
#         if elapsed >= model._time_limit_s:
#             model.terminate()  # status will be GRB.INTERRUPTED


# # === Tree cut-set (connectivity) lazy constraint callback (FORMULATION piece) ===
# def cutset_lazy_callback(model, where):
#     """
#     Your Tree Cut-Set lazy constraints.
#     Also enforces the SAME global wall-clock limit by calling global_time_callback first.
#     """
#     # Enforce global time at all callback points
#     global_time_callback(model, where)

#     if where == GRB.Callback.MIPSOL:
#         x_vals = model.cbGetSolution(model._x)

#         # Build graph of selected edges
#         G = nx.Graph()
#         for e in model._E:
#             if x_vals[e] > 0.5:
#                 G.add_edge(e[0], e[1])

#         components = list(nx.connected_components(G))
#         for comp in components:
#             if 0 < len(comp) < model._n:
#                 comp_set = set(comp)
#                 rest = model._V - comp_set

#                 # Cut edges crossing (comp, rest)
#                 cut_edges = []
#                 for (u, v) in model._E:
#                     if (u in comp_set and v in rest) or (v in comp_set and u in rest):
#                         cut_edges.append((u, v))

#                 # If violated: sum_{e in delta(S)} x_e >= 1
#                 lhs = sum(x_vals[e] for e in cut_edges) if cut_edges else 0.0
#                 if lhs < 1.0 - 1e-6:
#                     model.cbLazy(gp.quicksum(model._x[e] for e in cut_edges) >= 1)


# # Solve MST + Knapsack using Tree Cut-Set formulation with Gurobi
# # (Formulation kept; experiment semantics aligned with your other benchmark scripts.)
# def solve_with_gurobi(instance, seed, time_limit_s, verbose=False):
#     """
#     Same experiment semantics as your other scripts:
#     - global wall-clock limit (build + optimize) via callback
#     - Threads=1
#     - MIPGap=0.003
#     - solve_time measures total wall-clock (build + optimize)
#     """
#     start_total = time.time()

#     V_set = set(range(instance.num_nodes))
#     V_list = list(V_set)

#     E = [(min(u, v), max(u, v)) for u, v, _, _ in instance.edges]
#     c = {(min(u, v), max(u, v)): w for u, v, w, _ in instance.edges}
#     w_knap = {(min(u, v), max(u, v)): l for u, v, _, l in instance.edges}
#     B = instance.budget

#     # If we've already exceeded the budget before building the model
#     if time.time() - start_total >= time_limit_s:
#         total_time = time.time() - start_total
#         logger.warning("Time limit reached before model construction.")
#         return {
#             "instance_seed": seed,
#             "num_nodes": instance.num_nodes,
#             "density": instance.density,
#             "budget": instance.budget,
#             "solve_time": total_time,
#             "opt_time": 0.0,
#             "nodes_explored": 0,
#             "best_objective": float("inf"),
#             "mip_gap": float("inf"),
#             "status": GRB.TIME_LIMIT,
#             "status_str": "Time Limit (before optimize)",
#             "selected_edges": [],
#         }

#     # Model
#     model = gp.Model("MST_Knapsack_TreeCutSet")
#     model.setParam("OutputFlag", 1 if verbose else 0)
#     model.setParam("Threads", 1)         # fairness/consistency
#     model.setParam("MIPGap", 0.003)      # aligned
#     model.setParam("LazyConstraints", 1) # required for cbLazy
#     # IMPORTANT: do NOT set model.setParam("TimeLimit", ...) because we enforce global wall-clock

#     # Variables (formulation)
#     x = model.addVars(E, vtype=GRB.BINARY, name="x")

#     # Attach for callback
#     model._x = x
#     model._E = E
#     model._V = V_set
#     model._n = len(V_set)

#     # Objective
#     model.setObjective(gp.quicksum(c[e] * x[e] for e in E), GRB.MINIMIZE)

#     # Constraints (formulation)
#     model.addConstr(gp.quicksum(x[e] for e in E) == len(V_list) - 1, name="tree_size")
#     model.addConstr(gp.quicksum(w_knap[e] * x[e] for e in E) <= B, name="knapsack")

#     # Global wall-clock limit parameters
#     model._start_total = start_total
#     model._time_limit_s = time_limit_s

#     # Optimize with callback (lazy cut-set + global time)
#     start_opt = time.time()
#     model.optimize(cutset_lazy_callback)
#     opt_time = time.time() - start_opt
#     solve_time = time.time() - start_total

#     # Collect results
#     if model.status == GRB.OPTIMAL:
#         obj_val = model.objVal
#         selected_edges = [e for e in E if x[e].X > 0.99]
#     else:
#         obj_val = float("inf")
#         selected_edges = []

#     nodes_explored = model.NodeCount if hasattr(model, "NodeCount") else 0
#     gap = model.MIPGap if hasattr(model, "MIPGap") else float("inf")
#     status_str = STATUS_MAP.get(model.status, f"Unknown ({model.status})")

#     logger.info(f"Optimal objective: {obj_val}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Optimization status: {status_str}")

#     return {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,
#         "solve_time": solve_time,   # build + optimize
#         "opt_time": opt_time,       # optimize only
#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": model.status,
#         "status_str": status_str,
#         "selected_edges": selected_edges,
#     }


# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg({
#         "solve_time": ["mean", "std"],
#         "opt_time": ["mean", "std"],
#         "nodes_explored": ["mean", "std"],
#         "best_objective": ["mean", "std"],
#         "mip_gap": ["mean", "std"],
#     }).round(2)
#     return summary


# def summarize_for_paper(results, time_limit):
#     """
#     One-row paper-style summary consistent with your other scripts:
#     solved_pct, par10, time_all, nodes_all, time_solved, nodes_solved
#     """
#     df = pd.DataFrame(results)

#     df["solved"] = df["status"].isin([GRB.OPTIMAL])
#     df["capped_time"] = df["solve_time"].clip(upper=time_limit)
#     df["par10"] = df["solve_time"].where(df["solved"], 10.0 * time_limit)
#     df["nodes_solved"] = df["nodes_explored"].where(df["solved"])

#     solved_pct = 100.0 * df["solved"].mean()
#     par10_mean = df["par10"].mean()
#     time_all_mean = df["solve_time"].mean()
#     nodes_all_mean = df["nodes_explored"].mean()

#     time_solved_mean = df.loc[df["solved"], "solve_time"].mean()
#     nodes_solved_mean = df["nodes_solved"].mean()

#     summary_row = pd.DataFrame([{
#         "solved_pct": solved_pct,
#         "par10": par10_mean,
#         "time_all": time_all_mean,
#         "nodes_all": nodes_all_mean,
#         "time_solved": time_solved_mean,
#         "nodes_solved": nodes_solved_mean,
#     }])

#     return summary_row.round(2)


# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     # one folder per script seed (consistent with your other scripts)
#     run_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
#     os.makedirs(run_dir, exist_ok=True)

#     # Generate instances
#     instances = get_instances(args)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
#         result = solve_with_gurobi(
#             instance,
#             instance_seed,
#             time_limit_s=args.time_limit,
#             verbose=args.verbose
#         )
#         result["gen_time"] = gen_time
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     # Save per-instance results
#     final_path = os.path.join(run_dir, "gurobi_results.csv")
#     df = pd.DataFrame(results)
#     df.to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     # Debug summary
#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(run_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved Gurobi summary statistics to {summary_path}")

#     # Paper-style one-row summary
#     paper_summary = summarize_for_paper(results, args.time_limit)
#     paper_summary_path = os.path.join(run_dir, "gurobi_summary_for_paper.csv")
#     paper_summary.to_csv(paper_summary_path, index=False)
#     logger.info(f"Saved paper-style Gurobi summary to {paper_summary_path}")

#     print("\nGurobi Summary Statistics:")
#     print(summary)
#     print("\nPaper-style summary:")
#     print(paper_summary)


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
# import argparse
# import logging
# import networkx as nx

# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Status mapping for readability
# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
#     GRB.INTERRUPTED: "Interrupted (user time limit)",  # for callback terminate()
# }


# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         prog='Gurobi MST Knapsack Benchmark',
#         usage='%(prog)s [options]'
#     )
#     parser.add_argument("--seed", type=int, default=42,
#                         help="Random seed (default: 42)")
#     parser.add_argument("--num-nodes", type=int, default=50,
#                         help="The number of nodes in the graph (default: 50)")
#     parser.add_argument("--density", type=float, default=1.0,
#                         help="The density of the graph (default: 1.0)")
#     parser.add_argument("--num-instances", type=int, default=5,
#                         help="Number of instances to generate and solve (default: 5)")
#     parser.add_argument("--output-dir", type=str, default="/Users/ssha0224/Desktop",
#                         help="Directory to save results and instances (default: /Users/ssha0224/Desktop)")
#     parser.add_argument("--time-limit", type=float, default=1800.0,
#                         help="Global wall-clock time limit per instance in seconds (default: 1800)")
#     parser.add_argument("--verbose", action="store_true",
#                         help="Enable verbose Gurobi output (default: False)")
#     return parser.parse_args()


# # Generate instances
# def generate_instances(num_instances, num_nodes, density, seed, output_dir):
#     random.seed(seed)
#     instances = []
#     for i in range(num_instances):
#         start_gen = time.time()
#         instance_seed = random.randint(0, 1000000)
#         random.seed(instance_seed)
#         instance = MSTKPInstance(num_nodes, density)
#         gen_time = time.time() - start_gen
#         instances.append((instance, instance_seed, gen_time))
#         logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")

#     # Save to pickle
#     os.makedirs(output_dir, exist_ok=True)
#     instances_path = os.path.join(output_dir, "instances.pkl")
#     with open(instances_path, "wb") as f:
#         pickle.dump(instances, f)
#     logger.info(f"Saved instances to {instances_path}")
#     return instances


# def get_instances(args):
#     logger.info("Generating new instances based on provided seed.")
#     return generate_instances(args.num_instances, args.num_nodes, args.density, args.seed, args.output_dir)


# # === Global wall-clock callback (same semantics as your other scripts) ===
# def global_time_callback(model, where):
#     """
#     Terminate when wall-clock time since model._start_total exceeds model._time_limit_s.
#     """
#     if where in (GRB.Callback.MIP, GRB.Callback.MIPNODE, GRB.Callback.SIMPLEX):
#         elapsed = time.time() - model._start_total
#         if elapsed >= model._time_limit_s:
#             model.terminate()  # status becomes GRB.INTERRUPTED


# # === Arborescence cut-set callback (FORMULATION-RELATED; kept) ===
# def arborescence_cutset_callback(model, where):
#     if where == GRB.Callback.MIPSOL:
#         # NOTE: Using directed flow _f to detect reachability from root
#         f_vals = model.cbGetSolution(model._f)

#         G = nx.DiGraph()
#         for (i, j) in model._A:
#             if f_vals[(i, j)] > 0.5:
#                 G.add_edge(i, j)

#         # Find nodes reachable from root
#         reachable = set(nx.descendants(G, model._root)) | {model._root}
#         if len(reachable) < model._n:
#             unreachable = set(model._V) - reachable

#             # Cut: at least one selected edge must cross from reachable to unreachable
#             cut_edges = [
#                 (min(i, j), max(i, j))
#                 for i in reachable
#                 for j in unreachable
#                 if (min(i, j), max(i, j)) in model._E
#             ]

#             # Only add a cut if it is violated
#             x_vals = model.cbGetSolution(model._x)
#             if sum(x_vals[e] for e in cut_edges) < 1.0:
#                 model.cbLazy(gp.quicksum(model._x[e] for e in cut_edges) >= 1)


# def solve_with_gurobi(instance, seed, time_limit_s, verbose=False):
#     """
#     Keep the ARBORESCENCE CUT-SET formulation, but match the benchmark semantics:
#     - global wall-clock limit (build + optimize) via callback
#     - Threads=1
#     - MIPGap=0.003
#     - run_dir = output_dir/seed_{seed}
#     - solve_time is total wall-clock; opt_time is optimize() only
#     - consistent result dict + paper summary
#     """
#     start_total = time.time()

#     V = list(range(instance.num_nodes))
#     E = [(min(u, v), max(u, v)) for u, v, _, _ in instance.edges]
#     c = {(min(u, v), max(u, v)): w for u, v, w, _ in instance.edges}
#     w_knap = {(min(u, v), max(u, v)): l for u, v, _, l in instance.edges}
#     B = instance.budget
#     root = 0

#     # Directed arcs for flow (both directions)
#     A = [(i, j) for i in V for j in V if i != j and (min(i, j), max(i, j)) in E]

#     # If we already exceeded time limit before model construction
#     if time.time() - start_total >= time_limit_s:
#         total_time = time.time() - start_total
#         logger.warning("Time limit reached before model construction.")
#         return {
#             "instance_seed": seed,
#             "num_nodes": instance.num_nodes,
#             "density": instance.density,
#             "budget": instance.budget,
#             "solve_time": total_time,
#             "opt_time": 0.0,
#             "nodes_explored": 0,
#             "best_objective": float("inf"),
#             "mip_gap": float("inf"),
#             "status": GRB.TIME_LIMIT,
#             "status_str": "Time Limit (before optimize)",
#             "selected_edges": [],
#         }

#     # Gurobi model
#     model = gp.Model("MST_Knapsack_ArborescenceCutSet")
#     model.setParam("OutputFlag", 1 if verbose else 0)
#     model.setParam("Threads", 1)          # consistency/fairness
#     model.setParam("MIPGap", 0.003)       # match other scripts
#     model.setParam("LazyConstraints", 1)  # required for cbLazy cuts
#     # IMPORTANT: do NOT set model.setParam("TimeLimit", ...) because we use global wall-clock callback

#     # Variables (FORMULATION)
#     model._x = model.addVars(E, vtype=GRB.BINARY, name="x")
#     model._f = model.addVars(A, vtype=GRB.CONTINUOUS, lb=0.0, name="f")

#     # Store for callbacks
#     model._E = E
#     model._A = A
#     model._V = V
#     model._n = len(V)
#     model._root = root

#     # Objective
#     model.setObjective(gp.quicksum(c[e] * model._x[e] for e in E), GRB.MINIMIZE)

#     # Constraints (FORMULATION AS PROVIDED)
#     model.addConstr(gp.quicksum(model._x[e] for e in E) == len(V) - 1, name="tree_size")
#     model.addConstr(gp.quicksum(w_knap[e] * model._x[e] for e in E) <= B, name="knapsack")

#     # Flow conservation (single commodity)
#     model.addConstr(
#         gp.quicksum(model._f[(root, j)] for j in V if (root, j) in A) -
#         gp.quicksum(model._f[(j, root)] for j in V if (j, root) in A) == len(V) - 1,
#         name="flow_supply_root"
#     )
#     for i in V:
#         if i == root:
#             continue
#         inflow = gp.quicksum(model._f[(j, i)] for j in V if (j, i) in A)
#         outflow = gp.quicksum(model._f[(i, j)] for j in V if (i, j) in A)
#         model.addConstr(inflow - outflow == 1, name=f"flow_demand_{i}")

#     # Capacity: f_ij <= (n-1) * x_e
#     for (i, j) in A:
#         e = (min(i, j), max(i, j))
#         model.addConstr(
#             model._f[(i, j)] <= (len(V) - 1) * model._x[e],
#             name=f"capacity_{i}_{j}"
#         )

#     # === GLOBAL WALL-CLOCK LIMIT ===
#     model._start_total = start_total
#     model._time_limit_s = time_limit_s

#     # Optimize with BOTH callbacks:
#     # - global_time_callback enforces total wall clock
#     # - arborescence_cutset_callback adds lazy cuts
#     def combined_callback(cb_model, where):
#         global_time_callback(cb_model, where)
#         arborescence_cutset_callback(cb_model, where)

#     start_opt = time.time()
#     model.optimize(combined_callback)
#     opt_time = time.time() - start_opt
#     solve_time = time.time() - start_total

#     # Collect results
#     if model.status == GRB.OPTIMAL:
#         obj_val = model.objVal
#         selected_edges = [e for e in E if model._x[e].X > 0.99]
#     else:
#         obj_val = float("inf")
#         selected_edges = []

#     nodes_explored = model.NodeCount if hasattr(model, "NodeCount") else 0
#     gap = model.MIPGap if hasattr(model, "MIPGap") else float("inf")
#     status_str = STATUS_MAP.get(model.status, f"Unknown ({model.status})")

#     logger.info(f"Optimal objective: {obj_val}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Optimization status: {status_str}")

#     return {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,
#         "solve_time": solve_time,
#         "opt_time": opt_time,
#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": model.status,
#         "status_str": status_str,
#         "selected_edges": selected_edges,
#     }


# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg({
#         "solve_time": ["mean", "std"],
#         "opt_time": ["mean", "std"],
#         "nodes_explored": ["mean", "std"],
#         "best_objective": ["mean", "std"],
#         "mip_gap": ["mean", "std"],
#     }).round(2)
#     return summary


# def summarize_for_paper(results, time_limit):
#     """
#     One-row summary consistent with your other scripts:
#     solved_pct, par10, time_all, nodes_all, time_solved, nodes_solved
#     """
#     df = pd.DataFrame(results)

#     df["solved"] = df["status"].isin([GRB.OPTIMAL])
#     df["capped_time"] = df["solve_time"].clip(upper=time_limit)
#     df["par10"] = df["solve_time"].where(df["solved"], 10.0 * time_limit)
#     df["nodes_solved"] = df["nodes_explored"].where(df["solved"])

#     solved_pct = 100.0 * df["solved"].mean()
#     par10_mean = df["par10"].mean()
#     time_all_mean = df["solve_time"].mean()
#     nodes_all_mean = df["nodes_explored"].mean()

#     time_solved_mean = df.loc[df["solved"], "solve_time"].mean()
#     nodes_solved_mean = df["nodes_solved"].mean()

#     summary_row = pd.DataFrame([{
#         "solved_pct": solved_pct,
#         "par10": par10_mean,
#         "time_all": time_all_mean,
#         "nodes_all": nodes_all_mean,
#         "time_solved": time_solved_mean,
#         "nodes_solved": nodes_solved_mean,
#     }])

#     return summary_row.round(2)


# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     # one folder per script seed (consistent with your other scripts)
#     run_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
#     os.makedirs(run_dir, exist_ok=True)

#     instances = get_instances(args)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
#         result = solve_with_gurobi(
#             instance,
#             instance_seed,
#             time_limit_s=args.time_limit,
#             verbose=args.verbose
#         )
#         result["gen_time"] = gen_time
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     # Save per-instance results
#     final_path = os.path.join(run_dir, "gurobi_results.csv")
#     df = pd.DataFrame(results)
#     df.to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     # Debug summary
#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(run_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved Gurobi summary statistics to {summary_path}")

#     # Paper-style one-row summary
#     paper_summary = summarize_for_paper(results, args.time_limit)
#     paper_summary_path = os.path.join(run_dir, "gurobi_summary_for_paper.csv")
#     paper_summary.to_csv(paper_summary_path, index=False)
#     logger.info(f"Saved paper-style Gurobi summary to {paper_summary_path}")

#     print("\nGurobi Summary Statistics:")
#     print(summary)
#     print("\nPaper-style summary:")
#     print(paper_summary)


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
# import argparse
# import logging

# from mstkpinstance import MSTKPInstance  # your instance generator

# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# # Status mapping for readability
# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
#     GRB.INTERRUPTED: "Interrupted (user time limit)",  # for callback terminate()
# }


# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         prog="Gurobi MST Knapsack Benchmark",
#         usage="%(prog)s [options]"
#     )
#     parser.add_argument("--seed", type=int, default=42,
#                         help="Random seed (default: 42)")
#     parser.add_argument("--num-nodes", type=int, default=50,
#                         help="The number of nodes in the graph (default: 50)")
#     parser.add_argument("--density", type=float, default=1.0,
#                         help="The density of the graph (default: 1.0)")
#     parser.add_argument("--num-instances", type=int, default=5,
#                         help="Number of instances to generate and solve (default: 5)")
#     parser.add_argument("--output-dir", type=str, default="/Users/ssha0224/Desktop",
#                         help="Directory to save results and instances (default: /Users/ssha0224/Desktop)")
#     parser.add_argument("--time-limit", type=float, default=1800.0,
#                         help="Global wall-clock time limit per instance in seconds (default: 1800)")
#     parser.add_argument("--verbose", action="store_true",
#                         help="Enable verbose output (default: False)")
#     return parser.parse_args()


# # Generate instances
# def generate_instances(num_instances, num_nodes, density, seed, output_dir):
#     random.seed(seed)
#     instances = []
#     for i in range(num_instances):
#         start_gen = time.time()
#         instance_seed = random.randint(0, 1000000)
#         random.seed(instance_seed)
#         instance = MSTKPInstance(num_nodes, density)
#         gen_time = time.time() - start_gen
#         instances.append((instance, instance_seed, gen_time))
#         logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")

#     os.makedirs(output_dir, exist_ok=True)
#     instances_path = os.path.join(output_dir, "instances.pkl")
#     with open(instances_path, "wb") as f:
#         pickle.dump(instances, f)
#     logger.info(f"Saved instances to {instances_path}")
#     return instances


# def get_instances(args):
#     logger.info("Generating new instances based on provided seed.")
#     return generate_instances(args.num_instances, args.num_nodes, args.density, args.seed, args.output_dir)


# # === Global wall-clock callback (same as your other aligned scripts) ===
# def global_time_callback(model, where):
#     if where in (GRB.Callback.MIP, GRB.Callback.MIPNODE, GRB.Callback.SIMPLEX):
#         elapsed = time.time() - model._start_total
#         if elapsed >= model._time_limit_s:
#             model.terminate()  # status becomes GRB.INTERRUPTED


# # Solve MST + Knapsack using k-Arborescence formulation (approximated as multi-commodity flow)
# # (FORMULATION KEPT; only benchmarking semantics aligned)
# def solve_with_gurobi(instance, seed, time_limit_s, verbose=False):
#     """
#     Align to standard benchmark semantics:
#     - global wall-clock limit (build + optimize) via callback
#     - Threads=1
#     - MIPGap=0.003
#     - solve_time = total wall-clock; opt_time = optimize only
#     """
#     start_total = time.time()

#     V = list(range(instance.num_nodes))
#     E = [(min(u, v), max(u, v)) for u, v, _, _ in instance.edges]
#     c = {(min(u, v), max(u, v)): w for u, v, w, _ in instance.edges}
#     w_knap = {(min(u, v), max(u, v)): l for u, v, _, l in instance.edges}
#     B = instance.budget
#     root = 0

#     # Commodities: one for each non-root node
#     K = list(range(1, instance.num_nodes))

#     # Directed arcs for flow (both directions)
#     A = [(i, j) for i in V for j in V if i != j and (min(i, j), max(i, j)) in E]

#     # If already out of time before building the model
#     if time.time() - start_total >= time_limit_s:
#         total_time = time.time() - start_total
#         logger.warning("Time limit reached before model construction.")
#         return {
#             "instance_seed": seed,
#             "num_nodes": instance.num_nodes,
#             "density": instance.density,
#             "budget": instance.budget,
#             "solve_time": total_time,
#             "opt_time": 0.0,
#             "nodes_explored": 0,
#             "best_objective": float("inf"),
#             "mip_gap": float("inf"),
#             "status": GRB.TIME_LIMIT,
#             "status_str": "Time Limit (before optimize)",
#             "selected_edges": [],
#         }

#     # Gurobi model
#     model = gp.Model("MST_Knapsack_kArborescence")
#     model.setParam("OutputFlag", 1 if verbose else 0)
#     model.setParam("Threads", 1)       # fairness/consistency
#     model.setParam("MIPGap", 0.003)    # match your other scripts
#     # IMPORTANT: do NOT set model.setParam("TimeLimit", ...) because we use callback global wall-clock

#     # Variables (FORMULATION)
#     x = model.addVars(E, vtype=GRB.BINARY, name="x")
#     f = model.addVars([(i, j, k) for (i, j) in A for k in K],
#                       vtype=GRB.CONTINUOUS, lb=0.0, name="f")

#     # Objective
#     model.setObjective(gp.quicksum(c[e] * x[e] for e in E), GRB.MINIMIZE)

#     # Constraints (FORMULATION AS PROVIDED)
#     # 1) Exactly |V|-1 edges
#     model.addConstr(gp.quicksum(x[e] for e in E) == len(V) - 1, name="tree_size")

#     # 2) Knapsack
#     model.addConstr(gp.quicksum(w_knap[e] * x[e] for e in E) <= B, name="knapsack")

#     # 3) Flow conservation for each commodity k
#     for k in K:
#         for i in V:
#             inflow = gp.quicksum(f[(j, i, k)] for j in V if (j, i) in A)
#             outflow = gp.quicksum(f[(i, j, k)] for j in V if (i, j) in A)
#             if i == root:
#                 model.addConstr(outflow - inflow == 1, name=f"flow_supply_{k}")
#             elif i == k:
#                 model.addConstr(outflow - inflow == -1, name=f"flow_demand_{k}_{i}")
#             else:
#                 model.addConstr(outflow - inflow == 0, name=f"flow_balance_{k}_{i}")

#     # 4) Capacity constraints
#     # NOTE: keep your structure; just avoid the "else 0" bug (do nothing if arc missing)
#     for (u, v) in E:
#         if (u, v) in A and (v, u) in A:
#             for k in K:
#                 model.addConstr(
#                     f[(u, v, k)] + f[(v, u, k)] <= x[(u, v)],
#                     name=f"capacity_{u}_{v}_{k}"
#                 )

#     # === GLOBAL WALL-CLOCK LIMIT ===
#     model._start_total = start_total
#     model._time_limit_s = time_limit_s

#     # Optimize with callback
#     start_opt = time.time()
#     model.optimize(global_time_callback)
#     opt_time = time.time() - start_opt
#     solve_time = time.time() - start_total

#     # Collect results
#     if model.status == GRB.OPTIMAL:
#         obj_val = model.objVal
#         selected_edges = [e for e in E if x[e].X > 0.99]
#     else:
#         obj_val = float("inf")
#         selected_edges = []

#     nodes_explored = model.NodeCount if hasattr(model, "NodeCount") else 0
#     gap = model.MIPGap if hasattr(model, "MIPGap") else float("inf")
#     status_str = STATUS_MAP.get(model.status, f"Unknown ({model.status})")

#     logger.info(f"Optimal objective: {obj_val}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Optimization status: {status_str}")

#     return {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,
#         "solve_time": solve_time,
#         "opt_time": opt_time,
#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": model.status,
#         "status_str": status_str,
#         "selected_edges": selected_edges,
#     }


# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg({
#         "solve_time": ["mean", "std"],
#         "opt_time": ["mean", "std"],
#         "nodes_explored": ["mean", "std"],
#         "best_objective": ["mean", "std"],
#         "mip_gap": ["mean", "std"],
#     }).round(2)
#     return summary


# def summarize_for_paper(results, time_limit):
#     """
#     One-row summary consistent with your other scripts:
#     solved_pct, par10, time_all, nodes_all, time_solved, nodes_solved
#     """
#     df = pd.DataFrame(results)

#     df["solved"] = df["status"].isin([GRB.OPTIMAL])
#     df["capped_time"] = df["solve_time"].clip(upper=time_limit)
#     df["par10"] = df["solve_time"].where(df["solved"], 10.0 * time_limit)
#     df["nodes_solved"] = df["nodes_explored"].where(df["solved"])

#     solved_pct = 100.0 * df["solved"].mean()
#     par10_mean = df["par10"].mean()
#     time_all_mean = df["solve_time"].mean()
#     nodes_all_mean = df["nodes_explored"].mean()

#     time_solved_mean = df.loc[df["solved"], "solve_time"].mean()
#     nodes_solved_mean = df["nodes_solved"].mean()

#     summary_row = pd.DataFrame([{
#         "solved_pct": solved_pct,
#         "par10": par10_mean,
#         "time_all": time_all_mean,
#         "nodes_all": nodes_all_mean,
#         "time_solved": time_solved_mean,
#         "nodes_solved": nodes_solved_mean,
#     }])

#     return summary_row.round(2)


# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     # one folder per script seed (consistent with your aligned scripts)
#     run_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
#     os.makedirs(run_dir, exist_ok=True)

#     instances = get_instances(args)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
#         result = solve_with_gurobi(
#             instance,
#             instance_seed,
#             time_limit_s=args.time_limit,
#             verbose=args.verbose
#         )
#         result["gen_time"] = gen_time
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     # Save per-instance results
#     final_path = os.path.join(run_dir, "gurobi_results.csv")
#     df = pd.DataFrame(results)
#     df.to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     # Debug summary
#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(run_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved Gurobi summary statistics to {summary_path}")

#     # Paper-style one-row summary
#     paper_summary = summarize_for_paper(results, args.time_limit)
#     paper_summary_path = os.path.join(run_dir, "gurobi_summary_for_paper.csv")
#     paper_summary.to_csv(paper_summary_path, index=False)
#     logger.info(f"Saved paper-style Gurobi summary to {paper_summary_path}")

#     print("\nGurobi Summary Statistics:")
#     print(summary)
#     print("\nPaper-style summary:")
#     print(paper_summary)


# if __name__ == "__main__":
#     main()

####################10--2-6
# import random
# import pickle
# import gurobipy as gp
# from gurobipy import GRB
# import time
# import pandas as pd
# import os
# import argparse
# import logging

# from mstkpinstance import MSTKPInstance

# # ---------------- Logging ----------------
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# STATUS_MAP = {
#     GRB.OPTIMAL: "Optimal",
#     GRB.INFEASIBLE: "Infeasible",
#     GRB.UNBOUNDED: "Unbounded",
#     GRB.TIME_LIMIT: "Time Limit",
#     GRB.INF_OR_UNBD: "Infeasible or Unbounded",
#     GRB.INTERRUPTED: "Interrupted (callback terminate)",
# }

# # ---------------- CLI ----------------
# def parse_arguments():
#     parser = argparse.ArgumentParser(prog="Gurobi MST Knapsack Benchmark", usage="%(prog)s [options]")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--num-nodes", type=int, default=50)
#     parser.add_argument("--density", type=float, default=1.0)
#     parser.add_argument("--num-instances", type=int, default=5)
#     parser.add_argument("--output-dir", type=str, default="/Users/ssha0224/Desktop")
#     parser.add_argument("--time-limit", type=float, default=1800.0)
#     parser.add_argument("--verbose", action="store_true")
#     return parser.parse_args()

# # ---------------- Instance generation ----------------
# def generate_instances(num_instances, num_nodes, density, seed, output_dir):
#     random.seed(seed)
#     instances = []
#     for i in range(num_instances):
#         start_gen = time.time()
#         instance_seed = random.randint(0, 1_000_000)
#         random.seed(instance_seed)
#         instance = MSTKPInstance(num_nodes, density)
#         gen_time = time.time() - start_gen
#         instances.append((instance, instance_seed, gen_time))
#         logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")

#     os.makedirs(output_dir, exist_ok=True)
#     instances_path = os.path.join(output_dir, "instances.pkl")
#     with open(instances_path, "wb") as f:
#         pickle.dump(instances, f)
#     logger.info(f"Saved instances to {instances_path}")
#     return instances

# def get_instances(args):
#     logger.info("Generating new instances based on provided seed.")
#     return generate_instances(args.num_instances, args.num_nodes, args.density, args.seed, args.output_dir)

# # ---------------- Global wall-clock callback ----------------
# def global_time_callback(model, where):
#     # include MIPSOL too; safe if you later add callbacks
#     if where in (GRB.Callback.MIP, GRB.Callback.MIPNODE, GRB.Callback.SIMPLEX, GRB.Callback.MIPSOL):
#         elapsed = time.time() - model._start_total
#         if elapsed >= model._time_limit_s:
#             model.terminate()

# # ---------------- Utility: unique undirected edges ----------------
# def _extract_unique_edges(instance):
#     """
#     Build a unique undirected edge list E with canonical orientation (u < v),
#     plus cost c[e] and knapsack weight w_knap[e].
#     This avoids duplicate keys in addVars().
#     """
#     edge_data = {}
#     for rec in instance.edges:
#         if len(rec) != 4:
#             raise ValueError(f"Expected instance.edges records of length 4, got: {rec}")
#         u, v, cost, wlen = rec
#         a, b = (u, v) if u < v else (v, u)
#         if a == b:
#             continue
#         if (a, b) in edge_data:
#             old_cost, old_wlen = edge_data[(a, b)]
#             if abs(old_cost - float(cost)) > 1e-9 or abs(old_wlen - float(wlen)) > 1e-9:
#                 logger.warning(f"Duplicate edge ({a},{b}) with inconsistent data; keeping first.")
#             continue
#         edge_data[(a, b)] = (float(cost), float(wlen))

#     E = list(edge_data.keys())
#     c = {e: edge_data[e][0] for e in E}
#     w_knap = {e: edge_data[e][1] for e in E}
#     return E, c, w_knap

# # ---------------- 10.2.6 k-arborescence formulation ----------------
# def solve_with_gurobi(instance, seed, time_limit_s, verbose=False):
#     """
#     Implements Section 10.2.6 (k-arborescence formulation) + knapsack:
#       Variables:
#         x_e in {0,1} for undirected edges e={i,j}
#         z^k_{ij} in {0,1} for directed arcs (i,j) and each root k

#       Constraints (10.2.6 core):
#         (1) x(V) = sum_{e in E} x_e = n-1
#         (2) For each k (root), out-degree constraints:
#               sum_{j} z^k_{ij} <= 1   for all i != k
#               sum_{j} z^k_{kj} = 0    (root has out-degree 0)
#         (3) Linking for each undirected edge e={i,j} and each k:
#               z^k_{ij} + z^k_{ji} = x_{ij}

#       Plus MSTKP knapsack:
#         (4) sum_{e} w_e x_e <= B

#     Benchmark semantics preserved:
#       - global wall-clock time limit (build + optimize) via callback
#       - Threads=1, MIPGap=0.003
#       - solve_time includes build + optimize, opt_time is optimize() only
#     """
#     start_total = time.time()

#     n = int(instance.num_nodes)
#     V = list(range(n))
#     B = float(instance.budget)

#     # Unique edges + data
#     E, c, w_knap = _extract_unique_edges(instance)

#     # Early time check (consistent with your style)
#     if time.time() - start_total >= time_limit_s:
#         total_time = time.time() - start_total
#         logger.warning("Time limit reached before model construction.")
#         return {
#             "instance_seed": seed,
#             "num_nodes": instance.num_nodes,
#             "density": instance.density,
#             "budget": instance.budget,
#             "solve_time": total_time,
#             "opt_time": 0.0,
#             "nodes_explored": 0,
#             "best_objective": float("inf"),
#             "mip_gap": float("inf"),
#             "status": GRB.TIME_LIMIT,
#             "status_str": "Time Limit (before optimize)",
#             "selected_edges": [],
#         }

#     # Build directed arc set A from E (both directions)
#     A = []
#     for (i, j) in E:
#         A.append((i, j))
#         A.append((j, i))

#     model = gp.Model("MSTKP_kArborescence_10_2_6")
#     model.setParam("OutputFlag", 1 if verbose else 0)
#     model.setParam("Threads", 1)
#     model.setParam("MIPGap", 0.003)
#     # Do NOT set model TimeLimit; we use global wall-clock callback.

#     # Variables
#     x = model.addVars(E, vtype=GRB.BINARY, name="x")

#     # z[k,i,j] for k in V, (i,j) in A
#     # IMPORTANT: this is big: O(n * |A|) binaries.
#     z = model.addVars(((k, i, j) for k in V for (i, j) in A),
#                       vtype=GRB.BINARY, name="z")

#     # Objective
#     model.setObjective(gp.quicksum(c[e] * x[e] for e in E), GRB.MINIMIZE)

#     # (1) Tree size
#     model.addConstr(gp.quicksum(x[e] for e in E) == n - 1, name="tree_size")

#     # (4) Knapsack
#     model.addConstr(gp.quicksum(w_knap[e] * x[e] for e in E) <= B, name="knapsack")

#     # (2) Out-degree constraints for each k:
#     #   for i != k: sum_j z^k_{ij} <= 1
#     #   for i == k: sum_j z^k_{kj} == 0
#     # We can build outgoing lists via A
#     outgoing_by_i = {i: [] for i in V}
#     for (i, j) in A:
#         outgoing_by_i[i].append(j)

#     for k in V:
#         for i in V:
#             out_expr = gp.quicksum(z[(k, i, j)] for j in outgoing_by_i[i])
#             if i == k:
#                 model.addConstr(out_expr == 0, name=f"outdeg_root_k{k}_i{i}")
#             else:
#                 model.addConstr(out_expr <= 1, name=f"outdeg_k{k}_i{i}")

#     # (3) Linking: for each undirected edge {i,j} and each k:
#     #     z^k_{ij} + z^k_{ji} = x_{ij}
#     for (i, j) in E:
#         for k in V:
#             model.addConstr(z[(k, i, j)] + z[(k, j, i)] == x[(i, j)],
#                             name=f"link_k{k}_{i}_{j}")

#     # Attach wall-clock data
#     model._start_total = start_total
#     model._time_limit_s = time_limit_s

#     # Optimize with callback
#     start_opt = time.time()
#     model.optimize(global_time_callback)
#     opt_time = time.time() - start_opt
#     solve_time = time.time() - start_total

#     status = model.status
#     status_str = STATUS_MAP.get(status, f"Unknown ({status})")
#     nodes_explored = float(model.NodeCount) if hasattr(model, "NodeCount") else 0.0
#     gap = float(model.MIPGap) if hasattr(model, "MIPGap") else float("inf")

#     if status == GRB.OPTIMAL:
#         obj_val = float(model.objVal)
#         selected_edges = [e for e in E if x[e].X > 0.99]
#     else:
#         # If time-limited but feasible incumbent exists, you may want objVal too:
#         if getattr(model, "SolCount", 0) > 0:
#             obj_val = float(model.objVal)
#             selected_edges = [e for e in E if x[e].X > 0.99]
#         else:
#             obj_val = float("inf")
#             selected_edges = []

#     logger.info(f"Objective: {obj_val}")
#     logger.info(f"Nodes explored: {nodes_explored}")
#     logger.info(f"MIP gap: {gap}")
#     logger.info(f"Status: {status_str}")

#     return {
#         "instance_seed": seed,
#         "num_nodes": instance.num_nodes,
#         "density": instance.density,
#         "budget": instance.budget,
#         "solve_time": solve_time,
#         "opt_time": opt_time,
#         "nodes_explored": nodes_explored,
#         "best_objective": obj_val,
#         "mip_gap": gap,
#         "status": status,
#         "status_str": status_str,
#         "selected_edges": selected_edges,
#     }

# # ---------------- Analysis helpers ----------------
# def analyze_gurobi_results(results):
#     df = pd.DataFrame(results)
#     summary = df.agg({
#         "solve_time": ["mean", "std"],
#         "opt_time": ["mean", "std"],
#         "nodes_explored": ["mean", "std"],
#         "best_objective": ["mean", "std"],
#         "mip_gap": ["mean", "std"],
#     }).round(2)
#     return summary

# def summarize_for_paper(results, time_limit):
#     df = pd.DataFrame(results)
#     df["solved"] = df["status"].isin([GRB.OPTIMAL])
#     df["capped_time"] = df["solve_time"].clip(upper=time_limit)
#     df["par10"] = df["solve_time"].where(df["solved"], 10.0 * time_limit)
#     df["nodes_solved"] = df["nodes_explored"].where(df["solved"])

#     solved_pct = 100.0 * df["solved"].mean()
#     par10_mean = df["par10"].mean()
#     time_all_mean = df["solve_time"].mean()
#     nodes_all_mean = df["nodes_explored"].mean()

#     time_solved_mean = df.loc[df["solved"], "solve_time"].mean()
#     nodes_solved_mean = df["nodes_solved"].mean()

#     summary_row = pd.DataFrame([{
#         "solved_pct": solved_pct,
#         "par10": par10_mean,
#         "time_all": time_all_mean,
#         "nodes_all": nodes_all_mean,
#         "time_solved": time_solved_mean,
#         "nodes_solved": nodes_solved_mean,
#     }]).round(2)
#     return summary_row

# # ---------------- Main ----------------
# def main():
#     args = parse_arguments()
#     random.seed(args.seed)

#     run_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
#     os.makedirs(run_dir, exist_ok=True)

#     # Generate instances (NOTE: this writes instances.pkl to args.output_dir, same as your baseline.
#     # If you want no collisions across seeds, change generate_instances() to save to run_dir.)
#     instances = get_instances(args)

#     results = []
#     for idx, (instance, instance_seed, gen_time) in enumerate(instances):
#         logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
#         result = solve_with_gurobi(instance, instance_seed, args.time_limit, args.verbose)
#         result["gen_time"] = gen_time
#         result["total_time"] = result["gen_time"] + result["solve_time"]
#         results.append(result)

#     final_path = os.path.join(run_dir, "gurobi_results.csv")
#     pd.DataFrame(results).to_csv(final_path, index=False)
#     logger.info(f"Saved Gurobi results to {final_path}")

#     summary = analyze_gurobi_results(results)
#     summary_path = os.path.join(run_dir, "gurobi_summary.csv")
#     summary.to_csv(summary_path)
#     logger.info(f"Saved summary statistics to {summary_path}")

#     paper_summary = summarize_for_paper(results, args.time_limit)
#     paper_summary_path = os.path.join(run_dir, "gurobi_summary_for_paper.csv")
#     paper_summary.to_csv(paper_summary_path, index=False)
#     logger.info(f"Saved paper-style summary to {paper_summary_path}")

#     print("\nGurobi Summary Statistics:")
#     print(summary)
#     print("\nPaper-style summary:")
#     print(paper_summary)

# if __name__ == "__main__":
#     main()
######################10-2-4
import random
from mstkpinstance import MSTKPInstance
import pickle
import gurobipy as gp
from gurobipy import GRB
import time
import pandas as pd
import os
import argparse
import logging

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATUS_MAP = {
    GRB.OPTIMAL: "Optimal",
    GRB.INFEASIBLE: "Infeasible",
    GRB.UNBOUNDED: "Unbounded",
    GRB.TIME_LIMIT: "Time Limit",
    GRB.INF_OR_UNBD: "Infeasible or Unbounded",
    GRB.INTERRUPTED: "Interrupted (user time limit)",
}

# ---------------- CLI ----------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='Gurobi MST Knapsack Benchmark',
        usage='%(prog)s [options]'
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--num-instances", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="/Users/ssha0224/Desktop")
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

# ---------------- Instance generation ----------------
def generate_instances(num_instances, num_nodes, density, seed, output_dir):
    random.seed(seed)
    instances = []
    for i in range(num_instances):
        start_gen = time.time()
        instance_seed = random.randint(0, 1_000_000)
        random.seed(instance_seed)
        instance = MSTKPInstance(num_nodes, density)
        gen_time = time.time() - start_gen
        instances.append((instance, instance_seed, gen_time))
        logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")

    os.makedirs(output_dir, exist_ok=True)
    instances_path = os.path.join(output_dir, "instances.pkl")
    with open(instances_path, "wb") as f:
        pickle.dump(instances, f)
    logger.info(f"Saved instances to {instances_path}")
    return instances

def get_instances(args):
    logger.info("Generating new instances based on provided seed.")
    return generate_instances(args.num_instances, args.num_nodes, args.density, args.seed, args.output_dir)

# ---------------- Global wall-clock callback ----------------
def global_time_callback(model, where):
    if where in (GRB.Callback.MIP, GRB.Callback.MIPNODE, GRB.Callback.SIMPLEX, GRB.Callback.MIPSOL):
        elapsed = time.time() - model._start_total
        if elapsed >= model._time_limit_s:
            model.terminate()

# ---------------- 10.2.4 Arborescence cut-set separation ----------------
def arborescence_cutset_callback_10_2_4(model, where):
    # (not part of 10.2.4, but keeps your experiment semantics)
    global_time_callback(model, where)

    if where != GRB.Callback.MIPSOL:
        return

    xbar_vals = model.cbGetSolution(model._xbar)

    # Build adjacency of selected arcs
    adj = {i: [] for i in model._V}
    for (i, j) in model._A:
        if xbar_vals[(i, j)] > 0.5:
            adj[i].append(j)

    # Reachability from root
    root = model._root
    reachable = set([root])
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in reachable:
                reachable.add(v)
                stack.append(v)

    if len(reachable) == model._n:
        return

    unreachable = set(model._V) - reachable

    # For S = unreachable: add cut  xbar(delta^-(S)) >= 1
    # delta^-(S) = arcs (i,j) with i in V \ S and j in S
    cut_arcs = [(i, j) for (i, j) in model._A if (i in reachable and j in unreachable)]
    if not cut_arcs:
        return

    lhs = sum(xbar_vals[a] for a in cut_arcs)
    if lhs < 1.0 - 1e-6:
        model.cbLazy(gp.quicksum(model._xbar[a] for a in cut_arcs) >= 1)

# ---------------- Solve with 10.2.4 (Model AC) + knapsack ----------------
def solve_with_gurobi(instance, seed, time_limit_s, verbose=False):
    """
    EXACT 10.2.4 core (Model AC):
      - variables: xbar_a in {0,1} for arcs a in A
      - constraints (lazy separated): xbar(delta^-(S)) >= 1 for all S ⊂ V, root ∉ S
      - objective: min sum_{a in A} c_a xbar_a
    Notes:
      - We add: sum_a xbar_a = n-1 (recommended when you can't rely on all costs positive).
      - We keep: knapsack sum_a w_a xbar_a <= B (because you're solving MSTKP).
    """
    start_total = time.time()

    n = int(instance.num_nodes)
    V = list(range(n))
    root = 0
    B = float(instance.budget)

    # Build unique undirected edges E (u < v) and keep first cost/weight if duplicates appear
    E = []
    c_e = {}
    w_e = {}
    seen = set()
    for (u, v, cost, wlen) in instance.edges:
        a, b = (u, v) if u < v else (v, u)
        if a == b:
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        E.append((a, b))
        c_e[(a, b)] = float(cost)
        w_e[(a, b)] = float(wlen)

    # Directed arcs A: both directions per undirected edge
    A = []
    c_a = {}
    w_a = {}
    for (u, v) in E:
        A.append((u, v))
        A.append((v, u))
        c_a[(u, v)] = c_e[(u, v)]
        c_a[(v, u)] = c_e[(u, v)]
        w_a[(u, v)] = w_e[(u, v)]
        w_a[(v, u)] = w_e[(u, v)]

    # early time check
    if time.time() - start_total >= time_limit_s:
        total_time = time.time() - start_total
        logger.warning("Time limit reached before model construction.")
        return {
            "instance_seed": seed,
            "num_nodes": instance.num_nodes,
            "density": instance.density,
            "budget": instance.budget,
            "solve_time": total_time,
            "opt_time": 0.0,
            "nodes_explored": 0,
            "best_objective": float("inf"),
            "mip_gap": float("inf"),
            "status": GRB.TIME_LIMIT,
            "status_str": "Time Limit (before optimize)",
            "selected_edges": [],
        }

    model = gp.Model("MSTKP_ArborescenceCutSet_10_2_4_exact")
    model.setParam("OutputFlag", 1 if verbose else 0)
    model.setParam("Threads", 1)
    model.setParam("MIPGap", 0.003)
    model.setParam("LazyConstraints", 1)
    # (no TimeLimit param; we use global wall-clock callback)

    # 10.2.4 variable: directed arc selection
    xbar = model.addVars(A, vtype=GRB.BINARY, name="xbar")

    # 10.2.4 objective: min sum_{a in A} c_a xbar_a
    model.setObjective(gp.quicksum(c_a[a] * xbar[a] for a in A), GRB.MINIMIZE)

    # Recommended extra (per note in the text when you can't rely on c>0): sum_a xbar_a = n-1
    model.addConstr(gp.quicksum(xbar[a] for a in A) == n - 1, name="cardinality_n_minus_1")

    # MSTKP knapsack (kept)
    model.addConstr(gp.quicksum(w_a[a] * xbar[a] for a in A) <= B, name="knapsack")

    # Attach callback data
    model._xbar = xbar
    model._A = A
    model._V = V
    model._n = n
    model._root = root
    model._start_total = start_total
    model._time_limit_s = time_limit_s

    # Optimize (cut separation happens in callback)
    start_opt = time.time()
    model.optimize(arborescence_cutset_callback_10_2_4)
    opt_time = time.time() - start_opt
    solve_time = time.time() - start_total

    status = model.status
    status_str = STATUS_MAP.get(status, f"Unknown ({status})")
    nodes_explored = model.NodeCount if hasattr(model, "NodeCount") else 0
    gap = model.MIPGap if hasattr(model, "MIPGap") else float("inf")

    # Extract a feasible incumbent if available
    selected_edges = []
    if getattr(model, "SolCount", 0) > 0:
        # Convert selected arcs to undirected edges
        chosen_arcs = [a for a in A if xbar[a].X > 0.99]
        undirected = []
        seen_e = set()
        for (i, j) in chosen_arcs:
            e = (i, j) if i < j else (j, i)
            if e not in seen_e:
                seen_e.add(e)
                undirected.append(e)
        selected_edges = undirected
        obj_val = float(model.objVal)
    else:
        obj_val = float("inf")

    logger.info(f"Objective: {obj_val}")
    logger.info(f"Nodes explored: {nodes_explored}")
    logger.info(f"MIP gap: {gap}")
    logger.info(f"Status: {status_str}")

    return {
        "instance_seed": seed,
        "num_nodes": instance.num_nodes,
        "density": instance.density,
        "budget": instance.budget,
        "solve_time": solve_time,
        "opt_time": opt_time,
        "nodes_explored": nodes_explored,
        "best_objective": obj_val,
        "mip_gap": gap,
        "status": status,
        "status_str": status_str,
        "selected_edges": selected_edges,
    }

# ---------------- Analysis helpers ----------------
def analyze_gurobi_results(results):
    df = pd.DataFrame(results)
    summary = df.agg({
        "solve_time": ["mean", "std"],
        "opt_time": ["mean", "std"],
        "nodes_explored": ["mean", "std"],
        "best_objective": ["mean", "std"],
        "mip_gap": ["mean", "std"],
    }).round(2)
    return summary

def summarize_for_paper(results, time_limit):
    df = pd.DataFrame(results)
    df["solved"] = df["status"].isin([GRB.OPTIMAL])
    df["capped_time"] = df["solve_time"].clip(upper=time_limit)
    df["par10"] = df["solve_time"].where(df["solved"], 10.0 * time_limit)
    df["nodes_solved"] = df["nodes_explored"].where(df["solved"])

    solved_pct = 100.0 * df["solved"].mean()
    par10_mean = df["par10"].mean()
    time_all_mean = df["solve_time"].mean()
    nodes_all_mean = df["nodes_explored"].mean()

    time_solved_mean = df.loc[df["solved"], "solve_time"].mean()
    nodes_solved_mean = df["nodes_solved"].mean()

    return pd.DataFrame([{
        "solved_pct": solved_pct,
        "par10": par10_mean,
        "time_all": time_all_mean,
        "nodes_all": nodes_all_mean,
        "time_solved": time_solved_mean,
        "nodes_solved": nodes_solved_mean,
    }]).round(2)

# ---------------- Main ----------------
def main():
    args = parse_arguments()
    random.seed(args.seed)

    run_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    instances = get_instances(args)

    results = []
    for idx, (instance, instance_seed, gen_time) in enumerate(instances):
        logger.info(f"\nSolving instance {idx+1}/{len(instances)} with seed {instance_seed}")
        result = solve_with_gurobi(instance, instance_seed, args.time_limit, args.verbose)
        result["gen_time"] = gen_time
        result["total_time"] = result["gen_time"] + result["solve_time"]
        results.append(result)

    final_path = os.path.join(run_dir, "gurobi_results.csv")
    pd.DataFrame(results).to_csv(final_path, index=False)
    logger.info(f"Saved Gurobi results to {final_path}")

    summary = analyze_gurobi_results(results)
    summary_path = os.path.join(run_dir, "gurobi_summary.csv")
    summary.to_csv(summary_path)
    logger.info(f"Saved summary statistics to {summary_path}")

    paper_summary = summarize_for_paper(results, args.time_limit)
    paper_summary_path = os.path.join(run_dir, "gurobi_summary_for_paper.csv")
    paper_summary.to_csv(paper_summary_path, index=False)
    logger.info(f"Saved paper-style summary to {paper_summary_path}")

    print("\nGurobi Summary Statistics:")
    print(summary)
    print("\nPaper-style summary:")
    print(paper_summary)

if __name__ == "__main__":
    main()
