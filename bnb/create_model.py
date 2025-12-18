from pyscipopt import Model, quicksum
import numpy as np

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


def create_model_N_valves(
    a, b, d, N, y0, Np, ymin, ymax, fixed_var_ids, fixed_val, cons
):
    """
    Create a MILP model for N valves over Np time steps.

    Args:
        a, b (float): System dynamics parameters.
        d (int): Delay.
        N (int): Number of valves.
        y0 (list): Initial moisture levels.
        Np (int): Prediction horizon.
        ymin, ymax (float): Min/max moisture levels.
        fixed_var_ids (list): List of (i,j) tuples of variables to fix.
        fixed_val (list): Corresponding values.

    Returns:
        pyscipopt.Model: Optimizable MILP model.
    """
    model = Model()
    u = np.empty((N, Np), dtype=object)
    y = np.empty((N, Np), dtype=object)
    w = np.empty((N, Np), dtype=object)
    delta = np.empty((N, Np), dtype=object)
    for i in range(N):
        for j in range(Np):
            u[i, j] = model.addVar(vtype="C", lb=0, ub=1, name=f"u_{i}_{j}")
            y[i, j] = model.addVar(vtype="C", lb=ymin, ub=ymax, name=f"y_{i}_{j}")
            w[i, j] = model.addVar(vtype="C", lb=0, ub=1, name=f"w_{i}_{j}")
            delta[i, j] = model.addVar(vtype="C", lb=0, ub=1, name=f"delta_{i}_{j}")
    # Define initial moisture
    for i in range(N):
        model.addCons(y[i, 0] == y0[i])
    # Constraints for step 0 to d-1
    for k in range(d):
        for i in range(N):
            # No valve opened before step 0
            model.addCons(delta[i, k] == 0)
            # Model constraint
            model.addCons(y[i, k + 1] == a * y[i, k])
            # McCormick's contraints
            model.addCons(w[i, k] == 0)
    # Constraints for steps d to Np-1
    for k in range(d, Np):
        for i in range(N):
            # Delta definition
            model.addCons(delta[i, k] == u[i, k - d])
            # McCormick's contraints
            model.addCons(w[i, k] >= 0)
            model.addCons(w[i, k] >= (delta[i, k] - 1) + y[i, k])
            model.addCons(w[i, k] <= y[i, k])
            model.addCons(w[i, k] <= delta[i, k])

        if cons is not None:
            # Constraint on the number of opened valves
            model.addCons(quicksum(u[i, k] for i in range(N)) <= cons)
    # Constraints for steps d to Np-2
    for k in range(d, Np - 1):
        for i in range(N):
            # Model constraint
            model.addCons(
                y[i, k + 1] == a * y[i, k] + (b - a) * w[i, k] + (1 - b) * delta[i, k]
            )
    # Define the objective : minimize the total number of opened valves
    model.setObjective(
        quicksum(u[i, k] for i in range(N) for k in range(Np)), sense="minimize"
    )
    for (i, j), value in zip(fixed_var_ids, fixed_val):
        model.addCons(u[i, j] == value)
    model.hideOutput()  # Hide text
    return model
