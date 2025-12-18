import heapq
import numpy as np
import matplotlib.pyplot as plt
from pyscipopt import Model, quicksum, SCIP_STATUS
import time
import gc
import os, psutil
import cProfile
from create_model import create_model_N_valves

# ==========================
# System parameters
# ==========================
# Np : Prediction horizon (number of time steps considered)
# d : Delay between irrigation command and its effect
# ymax : Maximum acceptable soil moisture level
# ymin : Minimum acceptable soil moisture level
# y0 : Initial moisture levels for each soil patch
# a : System parameter representing natural moisture loss
# b : System parameter representing water absorption capacity
# N : Number of irrigation valves / soil patches


def memory_mb():
    """Return current memory usage of the process in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # Mo


# ==========================
# Node class
# ==========================
class Node:
    """
    Represents a node in the branch-and-bound tree.

    Attributes:
        fixed_variable_ids (list): List of indices of fixed decision variables.
        fixed_values (list): Corresponding values for fixed variables.
        objective (float): Objective value of this node.
        is_pruned (bool): Whether the node has been pruned.
        solution (dict): Dictionary containing variable values after LP solve.
        is_feasible (bool): Whether the LP problem is feasible.
    """

    def __init__(self, fixed_variable_ids, fixed_values, objective=None):
        self.objective = objective
        self.is_pruned = False
        self.fixed_variable_ids = fixed_variable_ids
        self.fixed_values = fixed_values
        self.solution = None
        self.is_feasible = None

    def branch(self, uk_id, a, b, d, N, y0, Np, ymin, ymax, max_valves):
        """
        Create two child nodes by fixing a fractional variable to 0 and 1.

        Args:
            uk_id (tuple): Indices of the variable to branch on.
            Other args: System parameters for model creation.

        Returns:
            tuple: Two new Node instances (child nodes).
        """
        fixed_var_1 = self.fixed_variable_ids.copy()
        fixed_var_2 = self.fixed_variable_ids.copy()
        fixed_var_1.append(uk_id)
        fixed_var_2.append(uk_id)

        fixed_val_1 = self.fixed_values.copy()
        fixed_val_2 = self.fixed_values.copy()
        fixed_val_1.append(0)
        fixed_val_2.append(1)

        m1 = create_model_N_valves(
            a, b, d, N, y0, Np, ymin, ymax, fixed_var_1, fixed_val_1, max_valves
        )
        m2 = create_model_N_valves(
            a, b, d, N, y0, Np, ymin, ymax, fixed_var_2, fixed_val_2, max_valves
        )

        n1 = Node(fixed_var_1, fixed_val_1)
        n2 = Node(fixed_var_2, fixed_val_2)

        n1.solve_lp(m1)
        n2.solve_lp(m2)

        del m1
        del m2

        return n1, n2

    def get_non_integer_var(self, N, Np):
        """
        Find the first variable u that is fractional in the LP solution.

        Args:
            N (int): Number of valves/patches.
            Np (int): Prediction horizon.

        Returns:
            tuple: Indices of a fractional variable (i,j), or None if all integer.
        """
        for i in range(N):
            for j in range(Np):
                var_name = f"u_{i}_{j}"
                val = self.solution[var_name]
                if (
                    abs(val - round(val)) > 1e-6
                ):  # Tolérance pour les variables fractionnaires
                    return (i, j)
        return None

    def solve_lp(self, model):
        """
        Solve the LP relaxation for this node.

        Args:
            model (pyscipopt.Model): SCIP model to solve.

        Returns:
            dict: Dictionary of variable values if feasible, None otherwise.
        """

        model.optimize()
        if model.getStatus() != "optimal":
            self.is_feasible = False
            self.objective = float("inf")
            # print("here ?", self.model.getStatus())
            return
        else:
            self.is_feasible = True
            res = {var.name: model.getVal(var) for var in model.getVars()}
            self.objective = model.getObjVal()
            # print("objective", self.objective)
            self.solution = res
            return res

    def __lt__(self, other):
        """Allows Nodes to be compared for heapq"""
        return self.objective < other.objective


# ==========================
# Branch-and-Bound class
# ==========================
class Branch_and_Bound:
    """
    Branch-and-bound solver for irrigation optimization.

    Attributes:
        queue (list): Priority queue of nodes to explore.
        incumbent (Node): Current best feasible solution.
        best_bound (Node): Node with lowest LP bound.
        tolerance (float): Relative tolerance for early stopping.
        Other attributes store system parameters and optimal solution.
    """

    def __init__(
        self,
        a,
        b,
        d,
        N,
        max_valves=None,
        y0=[0.7, 0.6, 0.67, 0.68, 0.65, 0.64, 0.68, 0.66, 0.59, 0.58],
        Np=20,
        ymin=0.4,
        ymax=0.7,
        tolerance=0.0001,
    ):

        model = create_model_N_valves(
            a, b, d, N, y0, Np, ymin, ymax, [], [], max_valves
        )

        self.default_node = Node(
            fixed_variable_ids=[], fixed_values=[], objective=float("inf")
        )
        self.root = Node(fixed_variable_ids=[], fixed_values=[])
        self.incumbent = self.default_node
        self.tolerance = tolerance
        self.queue = []
        self.queue.append(self.root)
        self.a = a
        self.b = b
        self.d = d
        self.N = N
        self.y0 = y0
        self.Np = Np
        self.ymin = ymin
        self.ymax = ymax
        self.max_valves = max_valves
        self.optimal_solution = None
        self.optimal_obj = None
        self.runtime = None
        self.model = model

    def run(self):
        """
        Main branch-and-bound loop to find optimal irrigation schedule.

        Returns:
            dict: Optimal solution if found, None if infeasible.
        """
        start = time.perf_counter()

        self.root.solve_lp(self.model)
        while self.queue:
            node = self.queue.pop()
            if not node.is_feasible:
                node.is_pruned = True
            else:
                if node.objective >= self.incumbent.objective:
                    node.is_pruned = True

            if not node.is_pruned:
                uk_id = node.get_non_integer_var(self.N, self.Np)
                if uk_id is None:  # node satisfies the MILP
                    if node.objective <= self.incumbent.objective:
                        self.incumbent = node

                    node.is_pruned = True
                else:
                    n1, n2 = node.branch(
                        uk_id=uk_id,
                        a=self.a,
                        b=self.b,
                        d=self.d,
                        N=self.N,
                        y0=self.y0,
                        Np=self.Np,
                        ymin=self.ymin,
                        ymax=self.ymax,
                        max_valves=self.max_valves,
                    )

                    del node

                    self.queue.append(n1)
                    self.queue.append(n2)

        if self.incumbent == self.default_node:
            print("Problem Infeasible")
            end = time.perf_counter()
            self.runtime = end - start
            return None

        self.optimal_solution = self.incumbent.solution
        self.optimal_obj = self.incumbent.objective

        end = time.perf_counter()
        self.runtime = end - start
        return self.optimal_solution

    def formatted_solution(self):
        """
        Format the optimal solution into a structured dictionary.

        Returns:
            dict or None: A dictionary with keys 0..N-1 for each valve/soil patch.
                Each entry contains sub-dictionaries:
                    - 'u': list of irrigation commands over Np time steps
                    - 'y': list of soil moisture levels over Np time steps
                    - 'w': list of McCormick variables over Np time steps
                    - 'delta': list of delayed irrigation indicators over Np time steps
                Returns None if no optimal solution has been computed.
        """
        if self.optimal_solution is None:
            return None

        formatted_solution = {}
        for i in range(self.N):
            u_vals = [self.optimal_solution[f"u_{i}_{j}"] for j in range(self.Np)]
            y_vals = [self.optimal_solution[f"y_{i}_{j}"] for j in range(self.Np)]
            w_vals = [self.optimal_solution[f"w_{i}_{j}"] for j in range(self.Np)]
            delta_vals = [
                self.optimal_solution[f"delta_{i}_{j}"] for j in range(self.Np)
            ]
            formatted_solution[i] = {
                "u": u_vals,
                "y": y_vals,
                "w": w_vals,
                "delta": delta_vals,
            }
        return formatted_solution

    def plot_result(self):
        """
        Plot the results of the optimal irrigation schedule.

        Generates two plots:
            1. Soil moisture evolution over time for each patch.
            2. Valve activation schedule as a scatter plot over time steps.

        Uses matplotlib for visualization. Does nothing if no solution is available.
        """
        sol = self.formatted_solution()
        if sol is None:
            return

        plt.figure()
        for i in range(self.N):
            plt.plot(sol[i]["y"], label=f"plot {i}")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.xticks(list(range(self.Np)))
        plt.xlabel("Time steps")
        plt.ylabel("Soil moisture level")
        plt.title(f"Moisture evolution in soil of parameter a={self.a}, b={self.b}")

        plt.figure()
        for i in range(self.N):
            t_active = [k for k in range(self.Np) if sol[i]["u"][k] > 0.5]
            plt.scatter(t_active, [i] * len(t_active), label=f"Valve {i}", marker="s")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.xticks(list(range(self.Np)))
        plt.yticks(range(self.N), [f"Valve {i+1}" for i in range(self.N)])

        plt.title("Valve activation schedule")
        plt.xlabel("Time steps")
        plt.ylabel("Valves")

        plt.show()


bnb = Branch_and_Bound(0.95, 0.8, 2, 1, Np=20, max_valves=1)
# bnb.run()
# print(bnb.formatted_solution())
# print("Optimal Objective : ", bnb.optimal_obj)
# print(bnb.runtime, " secondes")
# bnb.plot_result()


cProfile.run("bnb.run()")

# from line_profiler import LineProfiler

# lp = LineProfiler()
# lp.add_function(create_model_N_valves)

# # Exemple d'appel avec vos arguments
# lp(create_model_N_valves)(
#     a=0.95,
#     b=0.8,
#     d=2,
#     N=2,
#     y0=[0.7, 0.6],
#     Np=20,
#     ymin=0.4,
#     ymax=0.7,
#     fixed_var_ids=[],
#     fixed_val=[],
# )

# lp.print_stats()
