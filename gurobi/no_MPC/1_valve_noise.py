import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import os
import json

dossier = "mes_plots"
os.makedirs(dossier, exist_ok=True)
# Np = 20  # Prediction horizon (nb of time steps considered)
# d = 2  # Delay between irrigation command and its effect
# ymax = 0.65  # Maximum acceptanble soil moisture level
# ymin = 0.4  # Minimum acceptanble soil moisture level
# y0 = 0.7  # Initial moisture level
# a = 0.9  # Sytem specific parameters representing natural moisture loss
# b = 0.8  # Sytem specific parameters representing water absorption capacity


import gurobipy as gp
from gurobipy import GRB
import numpy as np
import json


def get_prct_feasible(n_trys, Np=20, a=0.9, b=0.8, d=2, ymin=0.4, ymax=0.7, y0=0.7):
    res = {
        "0.03": 0,
        "0.05": 0,
        "0.07": 0,
        "0.08": 0,
        "0.09": 0,
        "0.10": 0,
        "0.12": 0,
        "0.14": 0,
        "0.16": 0,
    }
    possible_stds = [float(std) for std in list(res.keys())]

    for std in possible_stds:
        for _ in range(n_trys):
            noise_a = np.random.normal(loc=0.0, scale=std, size=Np)
            noise_b = np.random.normal(loc=0.0, scale=std, size=Np)

            m = gp.Model("1_valve")
            y = m.addVars(Np, vtype=GRB.CONTINUOUS, name="yk", lb=ymin, ub=ymax)
            u = m.addVars(Np, vtype=GRB.BINARY, name="uk")
            w = m.addVars(Np, vtype=GRB.CONTINUOUS, name="wk")
            delta = m.addVars(Np, vtype=GRB.BINARY, name="deltak")

            # Define initial moisture
            m.addConstr(y[0] == y0)
            for k in range(d):
                # We consider no valve opened from 0 to d-1 steps
                m.addConstr(u[k] == 0)
                # We consider no valve opened before step 0
                m.addConstr(delta[k] == 0)
                # Model constraint & we consider no valve opening from 0 to d-1 steps
                m.addConstr(y[k + 1] == (a + noise_a[k]) * y[k])
                # Mc Cormick's constraints (same as saying m.addConstr(w[k] == 0))
                m.addConstr(w[k] >= ymin * delta[k])
                m.addConstr(w[k] >= ymax * (delta[k] - 1) + y[k])
                m.addConstr(w[k] <= ymin * (delta[k] - 1) + y[k])
                m.addConstr(w[k] <= ymax * delta[k])
            for k in range(d, Np):
                # # Soil moisture constraints
                # m.addConstr(y[k] <= ymax)
                # m.addConstr(y[k] >= ymin)

                # Delta definition constraint
                m.addConstr(delta[k] == u[k - d])
                # Mc Cormick's constraints
                m.addConstr(w[k] >= ymin * delta[k])
                m.addConstr(w[k] >= ymax * (delta[k] - 1) + y[k])
                m.addConstr(w[k] <= ymin * (delta[k] - 1) + y[k])
                m.addConstr(w[k] <= ymax * delta[k])
            for k in range(d, Np - 1):
                # Model constraint
                m.addConstr(
                    y[k + 1]
                    == (a + noise_a[k]) * y[k]
                    + (b + noise_b[k] - a - noise_a[k]) * w[k]
                    + (1 - b - noise_b[k]) * delta[k]
                )
            # We want to minimize the total number of valve that openened during the prediction horizon
            m.setObjective(sum(u[k] for k in range(d, Np)))
            m.optimize()

            if m.Status == GRB.OPTIMAL:
                res[f"{std:.2f}"] += 1
        res[f"{std:.2f}"] /= n_trys
    print(res)
    with open("std_results.json", "w") as f:
        json.dump(res, f)


def get_plot(std, a=0.9, b=0.8, Np=20, d=2, ymin=0.4, ymax=0.7, y0=0.7):

    m = gp.Model("1_valve_noise")
    y = m.addVars(Np, vtype=GRB.CONTINUOUS, name="yk", lb=ymin, ub=ymax)
    u = m.addVars(Np, vtype=GRB.BINARY, name="uk")
    w = m.addVars(Np, vtype=GRB.CONTINUOUS, name="wk")
    delta = m.addVars(Np, vtype=GRB.BINARY, name="deltak")

    noise_a = np.random.normal(loc=0.0, scale=std, size=Np)
    noise_b = np.random.normal(loc=0.0, scale=std, size=Np)

    # Define initial moisture
    m.addConstr(y[0] == y0)
    for k in range(d):
        # We consider no valve opened from 0 to d-1 steps
        m.addConstr(u[k] == 0)
        # We consider no valve opened before step 0
        m.addConstr(delta[k] == 0)
        # Model constraint & we consider no valve opening from 0 to d-1 steps
        m.addConstr(y[k + 1] == (a + noise_a[k]) * y[k])
        # Mc Cormick's constraints (same as saying m.addConstr(w[k] == 0))
        m.addConstr(w[k] >= ymin * delta[k])
        m.addConstr(w[k] >= ymax * (delta[k] - 1) + y[k])
        m.addConstr(w[k] <= ymin * (delta[k] - 1) + y[k])
        m.addConstr(w[k] <= ymax * delta[k])
    for k in range(d, Np):
        # # Soil moisture constraints
        # m.addConstr(y[k] <= ymax)
        # m.addConstr(y[k] >= ymin)

        # Delta definition constraint
        m.addConstr(delta[k] == u[k - d])
        # Mc Cormick's constraints
        m.addConstr(w[k] >= ymin * delta[k])
        m.addConstr(w[k] >= ymax * (delta[k] - 1) + y[k])
        m.addConstr(w[k] <= ymin * (delta[k] - 1) + y[k])
        m.addConstr(w[k] <= ymax * delta[k])
    for k in range(d, Np - 1):
        # Model constraint
        m.addConstr(
            y[k + 1]
            == (a + noise_a[k]) * y[k]
            + (b + noise_b[k] - a - noise_a[k]) * w[k]
            + (1 - b - noise_b[k]) * delta[k]
        )
    # We want to minimize the total number of valve that openened during the prediction horizon
    m.setObjective(sum(u[k] for k in range(d, Np)))
    m.optimize()

    u_best = []
    y_best = []
    for i in range(0, Np):
        print(f"u[{i}] = {u[i].X}")
        u_best.append(u[i].X)
        print(f"y[{i}] = {y[i].X}")
        y_best.append(y[i].X)
        print("*" * 80)

    print(f"Optimal value of the objective function : {m.ObjVal}")

    plt.figure()
    plt.scatter(list(range(Np)), u_best)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(list(range(Np)))
    plt.title("Valve activation schedule")
    plt.xlabel("Time steps")
    plt.ylabel("Valve activation")
    chemin = os.path.join(dossier, "valve_activation_noise.png")
    plt.savefig(chemin)

    plt.figure()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("Time steps")
    plt.ylabel("Soil moisture level")
    plt.xticks(list(range(Np)))
    plt.plot(list(range(Np)), y_best, marker="x")
    plt.title(f"Moisture evolution in soil of parameter a={a}, b={b}")
    chemin = os.path.join(dossier, "moisture_evolution_noise.png")
    plt.savefig(chemin)

    plt.show()


def relax(std, a=0.9, b=0.8, Np=20, d=2, ymin=0.4, ymax=0.7, y0=0.7):
    m = gp.Model("1_valve")
    y = m.addVars(Np, vtype=GRB.CONTINUOUS, name="yk", lb=ymin, ub=ymax)
    u = m.addVars(Np, vtype=GRB.BINARY, name="uk")
    w = m.addVars(Np, vtype=GRB.CONTINUOUS, name="wk")
    delta = m.addVars(Np, vtype=GRB.BINARY, name="deltak")
    mc_constrs = []

    noise_a = np.random.normal(loc=0.0, scale=std, size=Np)
    noise_b = np.random.normal(loc=0.0, scale=std, size=Np)

    # Define initial moisture
    m.addConstr(y[0] == y0)

    for k in range(d):
        # We consider no valve opened from 0 to d-1 steps
        m.addConstr(u[k] == 0)

        # We consider no valve opened before step 0
        m.addConstr(delta[k] == 0)

        # Model constraint & we consider no valve opening from 0 to d-1 steps
        m.addConstr(y[k + 1] == (a + noise_a[k]) * y[k])

        # Mc Cormick's constraints (same as saying m.addConstr(w[k] == 0))
        mc_constrs.append(m.addConstr(w[k] >= ymin * delta[k], name=f"mc_low_1_{k}"))
        mc_constrs.append(
            m.addConstr(w[k] >= ymax * (delta[k] - 1) + y[k], name=f"mc_low_2_{k}")
        )
        mc_constrs.append(
            m.addConstr(w[k] <= ymin * (delta[k] - 1) + y[k], name=f"mc_high_2_{k}")
        )
        mc_constrs.append(m.addConstr(w[k] <= ymax * delta[k], name=f"mc_high_1_{k}"))

    for k in range(d, Np):
        # # Soil moisture constraints
        # m.addConstr(y[k] <= ymax)
        # m.addConstr(y[k] >= ymin)

        # Delta definition constraint
        m.addConstr(delta[k] == u[k - d])

        # Mc Cormick's constraints
        mc_constrs.append(m.addConstr(w[k] >= ymin * delta[k], name=f"mc_low_1_{k}"))
        mc_constrs.append(
            m.addConstr(w[k] >= ymax * (delta[k] - 1) + y[k], name=f"mc_low_2_{k}")
        )
        mc_constrs.append(
            m.addConstr(w[k] <= ymin * (delta[k] - 1) + y[k], name=f"mc_high_2_{k}")
        )
        mc_constrs.append(m.addConstr(w[k] <= ymax * delta[k], name=f"mc_high_1_{k}"))
    for k in range(d, Np - 1):
        # Model constraint
        m.addConstr(
            y[k + 1]
            == (a + noise_a[k]) * y[k]
            + (b + noise_b[k] - a - noise_a[k]) * w[k]
            + (1 - b - noise_b[k]) * delta[k]
        )
    # We want to minimize the total number of valve that openened during the prediction horizon
    m.setObjective(sum(u[k] for k in range(d, Np)))

    vars_relax = [
        y[i] for i in range(Np)  # variables dont les bornes peuvent être dépassées
    ]
    # m.feasRelaxS(relaxobjtype=0, minrelax=True, vrelax=True, crelax=False)

    lb_pen = [1.0 for _ in vars_relax]  # poids pour lb
    ub_pen = [1.0 for _ in vars_relax]
    mc_pen = [1000.0] * len(mc_constrs)

    m.feasRelax(
        relaxobjtype=0,  # linéaire
        minrelax=True,
        vars=vars_relax,
        lbpen=lb_pen,
        ubpen=ub_pen,
        constrs=mc_constrs,
        rhspen=mc_pen,
    )
    m.optimize()

    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("modele.ilp")
    else:
        for v in m.getVars():
            print(v.VarName, v.X)

        for v in m.getVars():
            lb_slack = max(0, v.LB - v.X)
            ub_slack = max(0, v.X - v.UB)
            if lb_slack > 1e-6 or ub_slack > 1e-6:
                print(
                    f"{v.VarName} a été relâchée: LB slack={lb_slack:.4f}, UB slack={ub_slack:.4f}, X={v.X:.4f}"
                )


def plot_relax(std, a=0.9, b=0.8, Np=20, d=2, ymin=0.4, ymax=0.7, y0=0.7):
    m = gp.Model("1_valve_relax")
    y = m.addVars(Np, vtype=GRB.CONTINUOUS, name="yk", lb=ymin, ub=ymax)
    u = m.addVars(Np, vtype=GRB.BINARY, name="uk")
    w = m.addVars(Np, vtype=GRB.CONTINUOUS, name="wk")
    delta = m.addVars(Np, vtype=GRB.BINARY, name="deltak")
    mc_constrs = []

    noise_a = np.random.normal(0, std, Np)
    noise_b = np.random.normal(0, std, Np)

    # Initial moisture
    m.addConstr(y[0] == y0)

    for k in range(d):
        m.addConstr(u[k] == 0)
        m.addConstr(delta[k] == 0)
        m.addConstr(y[k + 1] == (a + noise_a[k]) * y[k])

        # mc_constrs.append(m.addConstr(w[k] >= ymin * delta[k]))
        # mc_constrs.append(m.addConstr(w[k] >= ymax * (delta[k] - 1) + y[k]))
        # mc_constrs.append(m.addConstr(w[k] <= ymin * (delta[k] - 1) + y[k]))
        # mc_constrs.append(m.addConstr(w[k] <= ymax * delta[k]))
        m.addConstr(w[k] >= 0)
        m.addConstr(w[k] >= (delta[k] - 1) + y[k])
        m.addConstr(w[k] <= y[k])
        m.addConstr(w[k] <= delta[k])

    for k in range(d, Np):
        m.addConstr(delta[k] == u[k - d])

        m.addConstr(w[k] >= 0)
        m.addConstr(w[k] >= (delta[k] - 1) + y[k])
        m.addConstr(w[k] <= y[k])
        m.addConstr(w[k] <= delta[k])

        # mc_constrs.append(m.addConstr(w[k] >= ymin * delta[k]))
        # mc_constrs.append(m.addConstr(w[k] >= ymax * (delta[k] - 1) + y[k]))
        # mc_constrs.append(m.addConstr(w[k] <= ymin * (delta[k] - 1) + y[k]))
        # mc_constrs.append(m.addConstr(w[k] <= ymax * delta[k]))

    for k in range(d, Np - 1):
        m.addConstr(
            y[k + 1]
            == (a + noise_a[k]) * y[k]
            + (b + noise_b[k] - a - noise_a[k]) * w[k]
            + (1 - b - noise_b[k]) * delta[k]
        )

    m.setObjective(sum(u[k] for k in range(d, Np)))

    # Relaxation setup
    vars_relax = [y[i] for i in range(Np)]
    lb_pen = [1.0] * Np
    ub_pen = [1.0] * Np
    mc_pen = [10000.0] * len(mc_constrs)

    m.feasRelax(
        relaxobjtype=0,
        minrelax=True,
        vars=vars_relax,
        lbpen=lb_pen,
        ubpen=ub_pen,
        constrs=None,
        rhspen=None,
    )
    # m.feasRelax(
    #     relaxobjtype=0,
    #     minrelax=True,
    #     vars=vars_relax,
    #     lbpen=lb_pen,
    #     ubpen=ub_pen,
    #     constrs=mc_constrs,
    #     rhspen=mc_pen,
    # )
    m.optimize()

    # Récupérer les valeurs
    y_vals = [y[i].X for i in range(Np)]
    u_vals = [u[i].X for i in range(Np)]

    # Afficher la planification des vannes
    plt.figure()
    plt.scatter(range(Np), u_vals)
    plt.title("Valve activation schedule (relaxed)")
    plt.xlabel("Time step")
    plt.ylabel("Valve activation")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(range(Np))
    plt.savefig(os.path.join(dossier, "valve_activation_relax.png"))

    # Afficher l'évolution de l'humidité
    plt.figure()
    plt.plot(range(Np), y_vals, marker="x")
    plt.title("Soil moisture evolution (relaxed)")
    plt.xlabel("Time step")
    plt.ylabel("Soil moisture")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(range(Np))
    plt.savefig(os.path.join(dossier, "moisture_evolution_relax.png"))

    plt.show()

    # Afficher les variables relâchées
    print("\nVariables potentiellement relâchées:")
    for v in m.getVars():
        lb_slack = max(0, v.LB - v.X)
        ub_slack = max(0, v.X - v.UB)
        if lb_slack > 1e-6 or ub_slack > 1e-6:
            print(
                f"{v.VarName}: LB slack={lb_slack:.4f}, UB slack={ub_slack:.4f}, X={v.X:.4f}"
            )


def simple_noise(std, a=0.9, b=0.8, Np=20, d=2, ymin=0.4, ymax=0.7, y0=0.7):
    m = gp.Model("1_valve_simple_noise")
    y = m.addVars(Np, vtype=GRB.CONTINUOUS, name="yk", lb=ymin, ub=ymax)
    u = m.addVars(Np, vtype=GRB.BINARY, name="uk")
    w = m.addVars(Np, vtype=GRB.CONTINUOUS, name="wk")
    delta = m.addVars(Np, vtype=GRB.BINARY, name="deltak")

    # Define initial moisture
    m.addConstr(y[0] == y0)

    for k in range(d):
        # We consider no valve opened from 0 to d-1 steps
        m.addConstr(u[k] == 0)
        # We consider no valve opened before step 0
        m.addConstr(delta[k] == 0)

        # Model constraint & we consider no valve opening from 0 to d-1 steps
        m.addConstr(y[k + 1] == a * y[k] + np.random.normal(0.0, std))

        # Mc Cormick's constraints (same as saying m.addConstr(w[k] == 0))
        m.addConstr(w[k] >= 0)
        m.addConstr(w[k] >= (delta[k] - 1) + y[k])
        m.addConstr(w[k] <= y[k])
        m.addConstr(w[k] <= delta[k])

    for k in range(d, Np):
        # # Soil moisture constraints
        # m.addConstr(y[k] <= ymax)
        # m.addConstr(y[k] >= ymin)

        # Delta definition constraint
        m.addConstr(delta[k] == u[k - d])

        # Mc Cormick's constraints
        m.addConstr(w[k] >= 0)
        m.addConstr(w[k] >= (delta[k] - 1) + y[k])
        m.addConstr(w[k] <= y[k])
        m.addConstr(w[k] <= delta[k])

    for k in range(d, Np - 1):
        # Model constraint
        m.addConstr(
            y[k + 1]
            == a * y[k]
            + (b - a) * w[k]
            + (1 - b) * delta[k]
            + np.random.normal(0.0, std)
        )

    # We want to minimize the total number of valve that openened during the prediction horizon
    m.setObjective(sum(u[k] for k in range(d, Np)))

    m.optimize()

    u_best = []
    y_best = []
    for i in range(0, Np):
        print(f"u[{i}] = {u[i].X}")
        u_best.append(u[i].X)
        print(f"y[{i}] = {y[i].X}")
        y_best.append(y[i].X)
        print("*" * 80)

    print(f"Optimal value of the objective function : {m.ObjVal}")

    plt.figure()
    plt.scatter(list(range(Np)), u_best)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(list(range(Np)))
    plt.title("Valve activation schedule")
    plt.xlabel("Time")
    plt.ylabel("Valve activation")
    chemin = os.path.join(dossier, "valve_activation_simple_noise.png")
    plt.savefig(chemin)

    plt.figure()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("Time")
    plt.ylabel("Soil moisture level")
    plt.xticks(list(range(Np)))
    plt.plot(list(range(Np)), y_best, marker="x")
    plt.title(f"Moisture evolution in soil of parameter a={a}, b={b} with simple noise")
    chemin = os.path.join(dossier, "moisture_evolution_simple_noise.png")
    plt.savefig(chemin)

    plt.show()


plot_relax(0.17)
