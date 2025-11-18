
#######################################
import os
import random
import argparse
import pandas as pd
# import numpy as np
from time import time
from mstkpinstance import MSTKPInstance
from mstkpbranchandbound import MSTNode
from branchandbound import RandomBranchingRule, BranchAndBound
# from branchandbound import RandomBranchingRule, BranchAndBound, StrongBranchingRule, PseudoCostBranchingRule, ReliabilityBranchingRule, MostFractionalRule
import pickle
import logging
from datetime import datetime
from collections import defaultdict
from lagrangianrelaxation import LagrangianMST
import gc
from pathlib import Path




# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(prog='MSTKP Benchmark Comparison', usage='%(prog)s [options]')
    parser.add_argument(
        "--num-instances",
        type=int,
        default=5,
        help="Number of instances to generate (default: 5)"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=50,
        help="Number of nodes in each graph (default: 50)"
    )
    parser.add_argument(
        "--density",
        type=float,
        default= 0.3,
        help="Edge density of the graph (default: 0.3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
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
    parser.add_argument(
    "--time-limit",
    type=float,
    default=1800.0,  # screening default; override to 3600 for finals
    help="Time limit per instance in seconds (default: 1800)"
)
    return parser.parse_args()

def generate_instances(num_instances, num_nodes, density, seed):
    random.seed(seed)
    instances = []
    for i in range(num_instances):
        instance_seed = random.randint(0, 1000000)
        random.seed(instance_seed)
        instance = MSTKPInstance(num_nodes, density)
        instances.append((instance, instance_seed))
        logger.info(f"Generated instance {i+1}/{num_instances} with seed {instance_seed}")
    return instances

def save_instances(instances, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    instances_path = os.path.join(output_dir, "instances.pkl")
    with open(instances_path, 'wb') as f:
        pickle.dump(instances, f)
    logger.info(f"Saved instances to {instances_path}")

def load_instances(output_dir):
    instances_path = os.path.join(output_dir, "instances.pkl")
    if not os.path.exists(instances_path):
        raise FileNotFoundError(f"No instances found at {instances_path}")
    with open(instances_path, 'rb') as f:
        instances = pickle.load(f)
    logger.info(f"Loaded {len(instances)} instances from {instances_path}")
    return instances

def run_experiment(instance, seed, config, args):
    start_total = time()
    if hasattr(MSTNode, 'global_edges'):
        del MSTNode.global_edges  # Better than =None: forces 'not hasattr' to trigger
    if hasattr(MSTNode, 'global_graph'):
        del MSTNode.global_graph

    gc.collect()

    random.seed(seed)
    LagrangianMST._edge_key = None
    LagrangianMST._edge_list = None
    LagrangianMST._edge_indices = None
    LagrangianMST._edge_weights = None
    LagrangianMST._edge_lengths = None
    LagrangianMST._edge_attributes = None
    mstkp_instance = instance
    branching_rule = config["branching_rule"]
    use_bisection = config["use_bisection"]
    use_2opt = config["use_2opt"]
    use_shooting = config["use_shooting"]
    cover_cuts = config["cover_cuts"]
    inherit_lambda = config["inherit_lambda"]
    inherit_step_size=config["inherit-step-size"]

    pseudocosts_up = defaultdict(float)
    pseudocosts_down = defaultdict(float)
    counts_up = defaultdict(int)
    counts_down = defaultdict(int)
    reliability_eta = 5
    lookahead_lambda = 4


    root_node = MSTNode(
        mstkp_instance.edges,
        mstkp_instance.num_nodes,
        mstkp_instance.budget,
        initial_lambda=0.05,
        inherit_lambda=inherit_lambda,
        inherit_step_size = inherit_step_size,
        branching_rule=branching_rule,
        step_size=0.00001,
        # inherit_step_size=False,
        use_cover_cuts=cover_cuts,
        cut_frequency=5,
        node_cut_frequency=10,
        parent_cover_cuts=None,
        parent_cover_multipliers=None,
        use_bisection=use_bisection,
        verbose=args.verbose,
        pseudocosts_up=pseudocosts_up,
        pseudocosts_down=pseudocosts_down,
        counts_up=counts_up,
        counts_down=counts_down,
        reliability_eta=reliability_eta,
        lookahead_lambda=lookahead_lambda,
        # partial_iters=partial_iters
    )

    branching_rule_obj = RandomBranchingRule()


    bnb_solver = BranchAndBound(
        branching_rule_obj,
        verbose=args.verbose,
        config=config,
        instance_seed=seed
    )

    two_opt_time = 0.0
    initial_solution = None
    initial_upper_bound = float("inf")

    if use_2opt:
        from mstkpsolver import two_opt_local_search
        start_2opt = time()
        initial_solution, initial_upper_bound = two_opt_local_search(
            mstkp_instance.edges,
            mstkp_instance.num_nodes,
            mstkp_instance.budget,
            root_node.mst_edges,
            verbose=args.verbose
        )
        two_opt_time = time() - start_2opt

    start_bnb = time()
    if use_shooting:
        best_solution, best_upper_bound = bnb_solver.solve_with_shooting(
            root_node,
            initial_lower_bound=root_node.local_lower_bound,
            initial_upper_bound=initial_upper_bound,
            initial_solution=initial_solution if initial_solution else None
        )
    else:
        # best_solution, best_upper_bound = bnb_solver.solve(root_node)
        best_solution, best_upper_bound = bnb_solver.solve(root_node, time_limit_s=args.time_limit)

    bnb_time = time() - start_bnb

    lagrangian_time = LagrangianMST.total_compute_time
    total_time = time() - start_total

    result = {
        "instance_seed": seed,
        "num_nodes": mstkp_instance.num_nodes,
        "density": mstkp_instance.density,
        "budget": mstkp_instance.budget,
        "branching_rule": branching_rule,
        "use_2opt": use_2opt,
        "use_shooting": use_shooting,
        "use_bisection": use_bisection,
        "cover_cuts": cover_cuts,
        "inherit_lambda": inherit_lambda,
        "total_time": total_time,
        "bnb_time": bnb_time,
        "lagrangian_time": lagrangian_time,
        "two_opt_time": two_opt_time,
        "total_nodes_solved": bnb_solver.total_nodes_solved,
        # "best_solution": best_solution,
        "best_solution": list(best_solution.mst_edges) if best_solution else None,
        "best_upper_bound": best_upper_bound,
        "timed_out": bnb_solver.timed_out
    }

    status = ("optimal" if (best_solution is not None and not bnb_solver.timed_out)
              else "timeout" if bnb_solver.timed_out else "incomplete")

    final_lower_bound = (bnb_solver.best_upper_bound - bnb_solver.final_duality_gap
                         if getattr(bnb_solver, "final_duality_gap", float("inf")) < float("inf")
                         else float("-inf"))

    # capped_time counts unsolved at the cutoff; par10 penalises timeouts
    capped_time = min(total_time, args.time_limit)
    par10 = total_time if (best_solution is not None and not bnb_solver.timed_out) else 10.0 * args.time_limit

    result.update({
        "status": status,
        "final_lower_bound": final_lower_bound,
        "final_duality_gap": getattr(bnb_solver, "final_duality_gap", float("inf")),
        "capped_time": capped_time,
        "par10": par10,
        "time_limit": args.time_limit
    })

    try:
        del root_node
        del bnb_solver
        if 'initial_solution' in locals():
            del initial_solution
    except:
        pass
    
    # Force cleanup of globals
    MSTNode.global_edges = None
    MSTNode.global_graph = None

    LagrangianMST._edge_key = None
    LagrangianMST._edge_list = None
    LagrangianMST._edge_indices = None
    LagrangianMST._edge_weights = None
    LagrangianMST._edge_lengths = None
    LagrangianMST._edge_attributes = None

    gc.collect()
    return result


def analyze_results(results):
    import pandas as pd
    df_all = pd.DataFrame(results).copy()

    # solved flag
    df_all["solved"] = (df_all["best_solution"].notna()) & (~df_all["timed_out"])

    # group-by keys (keep only those that exist in your results)
    keys = ["branching_rule", "use_bisection", "use_2opt", "use_shooting", "cover_cuts", "inherit_lambda"]
    keys = [k for k in keys if k in df_all.columns]

    # Solved-only summary (speed on successful runs)
    df_solved = df_all[df_all["solved"]].copy()
    solved_summary = (df_solved
        .groupby(keys, dropna=False)
        .agg(
            total_time_mean=("total_time","mean"),
            total_time_std =("total_time","std"),
            bnb_time_mean  =("bnb_time","mean") if "bnb_time" in df_solved else ("total_time","mean"),
            lagr_time_mean =("lagrangian_time","mean") if "lagrangian_time" in df_solved else ("total_time","mean"),
            nodes_mean     =("total_nodes_solved","mean") if "total_nodes_solved" in df_solved else ("total_time","mean"),
            solved_count   =("solved","size"),
        ).round(2)
        .reset_index()
    )

    # All-runs summary (includes timeouts) with PAR-10 & capped time
    all_summary = (df_all
        .groupby(keys, dropna=False)
        .agg(
            solved_rate      = ("solved", "mean"),
            par10_mean       = ("par10", "mean"),
            par10_std        = ("par10", "std"),
            capped_time_mean = ("capped_time","mean"),
            nodes_mean       = ("total_nodes_solved","mean") if "total_nodes_solved" in df_all else ("total_time","mean"),
        ).round(2)
        .reset_index()
    )
    all_summary["solved_rate"] = (100*all_summary["solved_rate"]).round(1)

    return {"solved_only": solved_summary, "all_runs": all_summary}

def rank_hard_instances(results):
    import pandas as pd
    df = pd.DataFrame(results).copy()
    df["solved"] = (df["best_solution"].notna()) & (~df["timed_out"])
    key = "instance_seed" if "instance_seed" in df.columns else "instance_id"
    agg = (df
        .groupby(key, dropna=False)
        .agg(
            unsolved_rate=("solved", lambda s: 1.0 - s.mean()),
            median_par10=("par10","median"),
            max_gap=("final_duality_gap","max"),
            runs=("solved","size"),
        )
        .reset_index()
    )
    # Sort: hardest first
    agg = agg.sort_values(["unsolved_rate","median_par10","max_gap"], ascending=[False, False, False])
    return agg



from pathlib import Path

def _write_latex_tables(summary_all_csv: str, summary_solved_csv: str, outdir: str) -> None:
    """
    Read the two summary CSVs and write LaTeX tables next to them.
    No external imports needed beyond pandas.
    """
    try:
        import pandas as pd
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)

        def _to_tex(df: pd.DataFrame, caption: str, label: str) -> str:
            prefer = ["branching_rule","use_bisection","use_2opt","use_shooting",
                      "cover_cuts","inherit_lambda","solved_rate","par10_mean",
                      "par10_std","capped_time_mean","nodes_mean","total_time_mean",
                      "bnb_time_mean","lagr_time_mean","nodes_std","solved_count"]
            cols = [c for c in prefer if c in df.columns] + [c for c in df.columns if c not in prefer]
            df = df[cols]
            # Format floats nicely
            latex = df.to_latex(index=False, escape=True, float_format="%.2f")
            return (
                "\\begin{table}[t]\n\\centering\n" +
                latex +
                f"\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{table}\n"
            )

        df_all = pd.read_csv(summary_all_csv)
        df_solved = pd.read_csv(summary_solved_csv)

        tex_all = _to_tex(df_all, "All runs with capped-time and PAR-10.", "tab:all-runs")
        tex_solved = _to_tex(df_solved, "Solved-only summary (means over successful runs).", "tab:solved-only")

        (out / "summary_all_runs.tex").write_text(tex_all, encoding="utf-8")
        (out / "summary_solved_only.tex").write_text(tex_solved, encoding="utf-8")
        print("LaTeX tables written:",
              str(out / "summary_all_runs.tex"),
              str(out / "summary_solved_only.tex"))
    except Exception as e:
        print(f"[WARN] LaTeX table generation skipped: {e}")
        
def main():
    args = parse_arguments()
    random.seed(args.seed)

    output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    # Delete existing instances.pkl to ensure new instances are generated
    instances_path = os.path.join(args.output_dir, "instances.pkl")
    if os.path.exists(instances_path):
        os.remove(instances_path)
        logger.info(f"Deleted existing {instances_path} to generate new instances")

    # Generate instances
    instances = generate_instances(args.num_instances, args.num_nodes, args.density, args.seed)
    save_instances(instances, args.output_dir)

    # Define configurations with varied cover_cuts and inherit_lambda (kept as you provided)
    configs = [
        {
            "branching_rule": "most_fractional",
            "use_bisection": False,
            "use_2opt": False,
            "use_shooting": False,
            "cover_cuts": True,
            "inherit-step-size": False,
            "inherit_lambda": True
        }
        ,
        {
            "branching_rule": "reliability",
            "use_bisection": False,
            "use_2opt": False,
            "use_shooting": False,
            "cover_cuts": True,
            "inherit-step-size": False,
            "inherit_lambda": True
        }
                ,

        {
            "branching_rule": "hybrid_strong_fractional",
            "use_bisection": False,
            "use_2opt": False,
            "use_shooting": False,
            "cover_cuts": True,
            "inherit-step-size": False,
            "inherit_lambda": True
        }

        # ,
        # {
        #     "branching_rule": "random_mst",
        #     "use_bisection": False,
        #     "use_2opt": False,
        #     "use_shooting": False,
        #     "cover_cuts": True,
        #     "inherit-step-size": False,
        #     "inherit_lambda": True
        # }
        # , 
        # {
        #     "branching_rule": "random_fractional",
        #     "use_bisection": False,
        #     "use_2opt": False,
        #     "use_shooting": False,
        #     "cover_cuts": True,
        #     "inherit-step-size": False,
        #     "inherit_lambda": True
        # }
        # ,
        # {
        #     "branching_rule": "sb_fractional",
        #     "use_bisection": False,
        #     "use_2opt": False,
        #     "use_shooting": False,
        #     "cover_cuts": True,
        #     "inherit-step-size": False,
        #     "inherit_lambda": True
        # }
        # ,

        # {
        #     "branching_rule": "strong_branching",
        #     "use_bisection": False,
        #     "use_2opt": False,
        #     "use_shooting": False,
        #     "cover_cuts": False,
        #     "inherit-step-size": False,
        #     "inherit_lambda": True
        # }
        # ,

        # {
        #     "branching_rule": "random_all",
        #     "use_bisection": False,
        #     "use_2opt": False,
        #     "use_shooting": False,
        #     "cover_cuts": True,
        #     "inherit-step-size": False,
        #     "inherit_lambda": True
        # }
        # ,
        # {
        #     "branching_rule": "strong_branching_all",
        #     "use_bisection": False,
        #     "use_2opt": False,
        #     "use_shooting": False,
        #     "cover_cuts": True,
        #     "inherit-step-size": False,
        #     "inherit_lambda": True
        # }
    ]

    results = []
    for config_idx, config in enumerate(configs):
        branching_rule = config["branching_rule"]
        logger.info(
            f"Running configuration {config_idx+1}/{len(configs)}: "
            f"Branching={branching_rule}, Bisection={config['use_bisection']}, "
            f"2-opt={config['use_2opt']}, Shooting={config['use_shooting']}, "
            f"Cover Cuts={config['cover_cuts']}, Inherit Lambda={config['inherit_lambda']}"
        )

        # Intermediate file for this config (append one row per instance)
        intermediate_path = os.path.join(output_dir, f"results_intermediate_{branching_rule}.csv")

        for instance_idx, (instance, instance_seed) in enumerate(instances):
            logger.info(f"Processing instance {instance_idx+1}/{len(instances)} with seed {instance_seed}")
            
            LagrangianMST.total_compute_time = 0.0  # Reset compute time

            try:
                result = run_experiment(instance, instance_seed, config, args)
                results.append(result)

                # --- Intermediate row append (so you don't lose progress) ---
                try:
                    df_row = pd.DataFrame([result])
                    write_header = not os.path.exists(intermediate_path)
                    df_row.to_csv(intermediate_path, mode='a', header=write_header, index=False)
                except Exception as w:
                    logger.warning(f"Could not append intermediate row for {branching_rule}: {w}")

                logger.info(
                    f"Completed instance {instance_idx+1}: "
                    f"Total={result['total_time']:.2f}s, "
                    f"BNB={result['bnb_time']:.2f}s, "
                    f"Lagrangian={result['lagrangian_time']:.2f}s, "
                    f"Nodes={result['total_nodes_solved']}"
                )
            except Exception as e:
                logger.error(f"Error processing instance {instance_idx+1}: {str(e)}")
                continue

        # --- Per-config snapshot (overwrite with all rows of this config) ---
        try:
            df_cfg = pd.DataFrame([r for r in results if r.get('branching_rule') == branching_rule])
            df_cfg.to_csv(intermediate_path, index=False)
            logger.info(f"Saved intermediate snapshot to {intermediate_path}")
        except Exception as e:
            logger.warning(f"Could not write per-config intermediate snapshot: {e}")

    # --- Save final results for the whole batch ---
    final_path = os.path.join(output_dir, "results.csv")
    df_all = pd.DataFrame(results)
    df_all.to_csv(final_path, index=False)
    logger.info(f"Saved final results to {final_path}")

    # Also keep a row-identical copy named results_detailed.csv
    detailed_path = os.path.join(output_dir, "results_detailed.csv")
    df_all.to_csv(detailed_path, index=False)

    # --- Build summaries and save them ---
    try:
        summary = analyze_results(results)  # <- updated analyze_results returns {"solved_only": DF, "all_runs": DF}
        if isinstance(summary, dict):
            solved_only_path = os.path.join(output_dir, "summary_solved_only.csv")
            all_runs_path = os.path.join(output_dir, "summary_all_runs.csv")
            summary["solved_only"].to_csv(solved_only_path, index=False)
            summary["all_runs"].to_csv(all_runs_path, index=False)
            logger.info(f"Saved summaries to {solved_only_path} and {all_runs_path}")
        else:
            # Fallback if your analyze_results returns a single DF
            summary_path = os.path.join(output_dir, "summary.csv")
            summary.to_csv(summary_path)
            logger.info(f"Saved summary statistics to {summary_path}")
            print("\nSummary Statistics:\n", summary)
    except Exception as e:
        logger.error(f"Failed to compute/save summaries: {e}")

    # --- Rank & persist hard instances for finals ---
    try:
        hard_df = rank_hard_instances(results)  # ensure this helper exists in this file
        hard_path = os.path.join(output_dir, "hard_instances.csv")
        hard_df.to_csv(hard_path, index=False)
        logger.info(f"Saved hard instance ranking to {hard_path}")
    except Exception as e:
        logger.warning(f"Failed to build/save hard instances: {e}")

    # --- Write LaTeX tables inline (no external imports) ---
    try:
        _write_latex_tables(
            os.path.join(output_dir, "summary_all_runs.csv"),
            os.path.join(output_dir, "summary_solved_only.csv"),
            output_dir
        )
    except Exception as e:
        logger.info(f"LaTeX table generation skipped ({e}). Run later if needed.")


if __name__ == "__main__":
    main()
####################################
