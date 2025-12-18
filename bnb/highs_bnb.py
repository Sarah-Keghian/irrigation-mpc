import heapq
import matplotlib.pyplot as plt
import time
import os, psutil
import highspy as highs
from highspy import HighsModelStatus
import numpy as np


import numpy as np
from highspy import Highs, HighsSparseMatrix


import numpy as np
import highspy as highs
from highspy import Highs, HighsSparseMatrix


def create_model_N_valves_highs(
    init: bool, a, b, d, N, y0, Np, ymin, ymax, fixed_var_ids, fixed_val, max_valve
):
    """
    Create a MILP model for N valves over Np time steps using HiGHS.
    Args:
        a, b (float): System dynamics parameters.
        d (int): Delay.
        N (int): Number of valves.
        y0 (list): Initial moisture levels.
        Np (int): Prediction horizon.
        ymin, ymax (float): Min/max moisture levels.
        fixed_var_ids (list): List of (i,j) tuples of variables to fix.
        fixed_val (list): Corresponding values.
        max_valve (int): Constraint on the number of opened valves.
    Returns:
        highspy.Highs: Optimizable MILP model.
    """
    # Initialize the HiGHS model
    h = highs.Highs()

    h.setOptionValue("output_flag", False)  # No text output

    # Create variables
    u = np.empty((N, Np), dtype=object)
    y = np.empty((N, Np), dtype=object)
    w = np.empty((N, Np), dtype=object)
    delta = np.empty((N, Np), dtype=object)
    if init:

        u_id = np.empty((N, Np), dtype=int)
        y_id = np.empty((N, Np), dtype=int)
        w_id = np.empty((N, Np), dtype=int)
        delta_id = np.empty((N, Np), dtype=int)
        compte = -1
        # Add variables to the model
        for i in range(N):
            for j in range(Np):
                u[i, j] = h.addVariable(lb=0, ub=1)
                compte += 1
                u_id[i, j] = compte

                y[i, j] = h.addVariable(lb=ymin, ub=ymax)
                compte += 1
                y_id[i, j] = compte

                w[i, j] = h.addVariable(lb=0, ub=1)
                compte += 1
                w_id[i, j] = compte

                delta[i, j] = h.addVariable(lb=0, ub=1)
                compte += 1
                delta_id[i, j] = compte
    else:
        # Add variables to the model
        for i in range(N):
            for j in range(Np):
                u[i, j] = h.addVariable(lb=0, ub=1)
                y[i, j] = h.addVariable(lb=ymin, ub=ymax)
                w[i, j] = h.addVariable(lb=0, ub=1)
                delta[i, j] = h.addVariable(lb=0, ub=1)

    # Define initial moisture
    for i in range(N):
        h.addConstr(y[i, 0] == y0[i])

    # Constraints for step 0 to d-1
    for k in range(d):
        for i in range(N):
            h.addConstr(delta[i, k] == 0)
            h.addConstr(y[i, k + 1] == a * y[i, k])
            h.addConstr(w[i, k] == 0)

    # Constraints for steps d to Np-1
    for k in range(d, Np):
        for i in range(N):
            h.addConstr(delta[i, k] == u[i, k - d])
            h.addConstr(w[i, k] >= 0)
            h.addConstr(w[i, k] >= delta[i, k] - 1 + y[i, k])
            h.addConstr(w[i, k] <= y[i, k])
            h.addConstr(w[i, k] <= delta[i, k])
        if max_valve is not None:
            h.addConstr(sum(u[i, k] for i in range(N)) <= max_valve)

    # Constraints for steps d to Np-2
    for k in range(d, Np - 1):
        for i in range(N):
            h.addConstr(
                y[i, k + 1] == a * y[i, k] + (b - a) * w[i, k] + (1 - b) * delta[i, k]
            )

    # Objective: minimize the total number of opened valves
    h.minimize(sum(u[i, k] for i in range(N) for k in range(Np)))

    # Fix variables
    for (i, j), value in zip(fixed_var_ids, fixed_val):
        h.addConstr(u[i, j] == value)

    if init:
        return h, u_id, w_id, y_id, delta_id
    else:
        return h


class Node:
    def __init__(self, fixed_variable_ids, fixed_values, objective=None):
        self.objective = objective
        self.is_pruned = False
        self.fixed_variable_ids = fixed_variable_ids
        self.fixed_values = fixed_values
        self.solution = None
        self.is_feasible = None

    def get_non_integer_var(self, N, Np):
        for i in range(N):
            for j, val in enumerate(self.solution[i]["u"]):
                if abs(val - round(val)) > 1e-6:
                    return (i, j)
        return None

    def solve_lp(self, model, u_id, w_id, y_id, delta_id, Np, N):
        model.run()
        model_status = model.getModelStatus()
        if model_status != HighsModelStatus.kOptimal:
            self.is_feasible = False
            self.objective = float("inf")
        else:
            self.is_feasible = True
            solution = model.getSolution()
            self.solution = {}
            for i in range(N):
                u_vals = []
                for j in range(Np):
                    # récupération des valeurs
                    val_u = solution.col_value[u_id[i, j]]

                    # arrondir uniquement si très proche de 0 ou 1
                    val_u = (
                        1
                        if abs(val_u - 1) < 1e-6
                        else 0 if abs(val_u) < 1e-6 else val_u
                    )

                    u_vals.append(val_u)

                self.solution[i] = {
                    "u": u_vals,
                    "y": [solution.col_value[y_id[i, j]] for j in range(Np)],
                    "w": [solution.col_value[w_id[i, j]] for j in range(Np)],
                    "delta": [solution.col_value[delta_id[i, j]] for j in range(Np)],
                }

            self.objective = sum(
                self.solution[i]["u"][k] for i in range(N) for k in range(Np)
            )

        return self.solution

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
        cons=None,
        y0=[0.7, 0.6, 0.67, 0.68, 0.65, 0.64, 0.68, 0.66, 0.59, 0.58],
        Np=20,
        ymin=0.4,
        ymax=0.7,
        tolerance=0.0001,
    ):

        model, u_id, w_id, y_id, delta_id = create_model_N_valves_highs(
            True, a, b, d, N, y0, Np, ymin, ymax, [], [], cons
        )

        self.default_node = Node(
            fixed_variable_ids=[], fixed_values=[], objective=float("inf")
        )
        self.root = Node(fixed_variable_ids=[], fixed_values=[])
        self.incumbent = self.default_node
        self.best_bound = self.root
        self.tolerance = tolerance
        self.queue = []
        heapq.heappush(self.queue, self.root)
        self.a = a
        self.b = b
        self.d = d
        self.N = N
        self.y0 = y0
        self.Np = Np
        self.ymin = ymin
        self.ymax = ymax
        self.cons = cons
        self.optimal_solution = None
        self.optimal_obj = None
        self.runtime = None
        self.milp_leaves = []
        self.model = model
        self.u_indices = u_id
        self.w_indices = w_id
        self.y_indices = y_id
        self.delta_indices = delta_id

    def run(self):
        """
        Main branch-and-bound loop to find optimal irrigation schedule.

        Returns:
            dict: Optimal solution if found, None if infeasible.
        """
        start = time.perf_counter()

        self.root.solve_lp(
            self.model,
            self.u_indices,
            self.w_indices,
            self.y_indices,
            self.delta_indices,
            self.Np,
            self.N,
        )
        while self.queue:
            node = heapq.heappop(self.queue)
            if not node.is_feasible:
                node.is_pruned = True
            else:
                if node.objective >= self.incumbent.objective:
                    node.is_pruned = True
                if (
                    abs(self.best_bound.objective - self.incumbent.objective)
                    <= self.tolerance
                ):
                    print("early stop")
                    end = time.perf_counter()
                    self.runtime = end - start
                    self.optimal_solution = self.incumbent.solution
                    self.optimal_obj = self.incumbent.objective
                    return self.optimal_solution

            if not node.is_pruned:
                uk_id = node.get_non_integer_var(self.N, self.Np)
                if uk_id is None:  # node satisfies the MILP
                    if node.objective <= self.incumbent.objective:
                        self.incumbent = node

                    node.is_pruned = True
                    heapq.heappush(self.milp_leaves, node)
                else:
                    n1, n2 = self.branch(node, uk_id)
                    del node

                    # print("RAM après suppression du modèle :", memory_mb(), "MB")

                    heapq.heappush(self.queue, n1)
                    heapq.heappush(self.queue, n2)

                self.best_bound = min(
                    self.queue + self.milp_leaves,
                )

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

    def branch(self, node, uk_id):
        fixed_var_1 = node.fixed_variable_ids.copy()
        fixed_var_2 = node.fixed_variable_ids.copy()
        fixed_var_1.append(uk_id)
        fixed_var_2.append(uk_id)
        fixed_val_1 = node.fixed_values.copy()
        fixed_val_2 = node.fixed_values.copy()
        fixed_val_1.append(0)
        fixed_val_2.append(1)

        m1 = create_model_N_valves_highs(
            False,
            self.a,
            self.b,
            self.d,
            self.N,
            self.y0,
            self.Np,
            self.ymin,
            self.ymax,
            fixed_var_1,
            fixed_val_1,
            self.cons,
        )
        m2 = create_model_N_valves_highs(
            False,
            self.a,
            self.b,
            self.d,
            self.N,
            self.y0,
            self.Np,
            self.ymin,
            self.ymax,
            fixed_var_2,
            fixed_val_2,
            self.cons,
        )

        n1 = Node(fixed_var_1, fixed_val_1)
        n2 = Node(fixed_var_2, fixed_val_2)

        n1.solve_lp(
            m1,
            self.u_indices,
            self.w_indices,
            self.y_indices,
            self.delta_indices,
            self.Np,
            self.N,
        )

        n2.solve_lp(
            m2,
            self.u_indices,
            self.w_indices,
            self.y_indices,
            self.delta_indices,
            self.Np,
            self.N,
        )
        del m1, m2

        return n1, n2

    def plot_result(self):
        """
        Plot the results of the optimal irrigation schedule.

        Generates two plots:
            1. Soil moisture evolution over time for each patch.
            2. Valve activation schedule as a scatter plot over time steps.

        Uses matplotlib for visualization. Does nothing if no solution is available.
        """
        sol = self.optimal_solution
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


bnb = Branch_and_Bound(0.95, 0.8, 2, 2, Np=20, cons=1)
# bnb.run()
# print(bnb.optimal_solution)
# print("Optimal Objective : ", bnb.optimal_obj)
# print(bnb.runtime, " secondes")
# bnb.plot_result()

import cProfile

cProfile.run("bnb.run()")
