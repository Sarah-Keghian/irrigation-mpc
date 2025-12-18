import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

dossier = "mes_plots"
os.makedirs(dossier, exist_ok=True)

Np = 21  # Prediction horizon (nb of time steps considered)
d = [6, 5, 5, 4, 3, 3, 2, 3, 3, 2]  # Delay between irrigation command and its effect
ymax = 0.7  # Maximum acceptanble soil moisture level
ymin = 0.4  # Minimum acceptanble soil moisture level
y0 = [0.7, 0.7, 0.7, 0.7]  # Initial moisture level
a = [
    0.99,
    0.98,
    0.96,
    0.95,
    0.9,
    0.9,
    0.88,
    0.88,
    0.87,
    0.85,
]  # Sytem specific parameters representing natural moisture loss
b = [
    0.89,
    0.85,
    0.87,
    0.8,
    0.8,
    0.76,
    0.82,
    0.75,
    0.8,
    0.76,
]  # Sytem specific parameters representing water absorption capacity
N = 10
y0 = [0.7, 0.6, 0.67, 0.68, 0.65, 0.64, 0.58, 0.68, 0.66, 0.59]
# y0_5 = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]


def N_valve(max_valves, Np=Np, d=d, ymax=ymax, ymin=ymin, a=a, b=b, N=N, y0=y0):
    m = gp.Model("N_valve")
    y = m.addMVar((N, Np + 1), vtype=GRB.CONTINUOUS, lb=ymin, ub=ymax)
    u = m.addMVar((N, Np + 1), vtype=GRB.BINARY)
    w = m.addMVar((N, Np + 1), vtype=GRB.CONTINUOUS)
    delta = m.addMVar((N, Np + 1), vtype=GRB.BINARY)

    # Define initial moisture
    for i in range(N):
        m.addConstr(y[i, 0] == y0[i])

    for i in range(N):
        # We consider no valve opened before step 0
        m.addConstrs(delta[i, k] == 0 for k in range(d[i]))

        # Model constraint & we consider no valve opening from 0 to d-1 steps
        m.addConstrs(y[i, k + 1] == a[i] * y[i, k] for k in range(d[i]))

        # Mc Cormick's constraints (same as saying m.addConstr(w[k] == 0))
        m.addConstrs((w[i, k] >= 0) for k in range(d[i]))
        m.addConstrs((w[i, k] >= (delta[i, k] - 1) + y[i, k]) for k in range(d[i]))
        m.addConstrs((w[i, k] <= y[i, k]) for k in range(d[i]))
        m.addConstrs((w[i, k] <= delta[i, k]) for k in range(d[i]))

        # Delta definition constraint
        m.addConstrs(delta[i, k] == u[i, k - d[i]] for k in range(d[i], Np + 1))

        # Mc Cormick's constraints
        m.addConstrs((w[i, k] >= 0) for k in range(d[i], Np + 1))
        m.addConstrs(
            (w[i, k] >= (delta[i, k] - 1) + y[i, k]) for k in range(d[i], Np + 1)
        )
        m.addConstrs((w[i, k] <= y[i, k]) for k in range(d[i], Np + 1))
        m.addConstrs((w[i, k] <= delta[i, k]) for k in range(d[i], Np + 1))

        # Model constraint
        m.addConstrs(
            (
                y[i, k + 1]
                == a[i] * y[i, k] + (b[i] - a[i]) * w[i, k] + (1 - b[i]) * delta[i, k]
            )
            for k in range(d[i], Np)
        )

    for k in range(1, Np + 1):
        # Max nb of activated valves constraint
        m.addConstr(u[:, k].sum() <= max_valves)

    # We want to minimize the total number of valve that openened during the prediction horizon
    m.setObjective(u.sum())

    m.optimize()

    return m, y, u, w, delta


def N_valve_noise(
    std, max_valves, Np=Np, d=d, ymax=ymax, ymin=ymin, a=a, b=b, N=N, y0=y0
):
    m = gp.Model("N_valve")
    y = m.addMVar((N, Np + 1), vtype=GRB.CONTINUOUS, lb=ymin, ub=ymax)
    u = m.addMVar((N, Np + 1), vtype=GRB.BINARY)
    w = m.addMVar((N, Np + 1), vtype=GRB.CONTINUOUS)
    delta = m.addMVar((N, Np + 1), vtype=GRB.BINARY)

    noise_a = {}
    noise_b = {}

    for i in range(N):
        noise_a[i] = np.random.normal(loc=0.0, scale=std[i], size=Np)
        noise_b[i] = np.random.normal(loc=0.0, scale=std[i], size=Np)

    # Define initial moisture
    for i in range(N):
        m.addConstr(y[i, 0] == y0[i])

    for i in range(N):
        # We consider no valve opened before step 0
        m.addConstrs(delta[i, k] == 0 for k in range(d[i]))

        # Model constraint & we consider no valve opening from 0 to d-1 steps
        m.addConstrs(
            y[i, k + 1] == (a[i] + noise_a[i][k]) * y[i, k] for k in range(d[i])
        )

        # Mc Cormick's constraints (same as saying m.addConstr(w[k] == 0))
        m.addConstrs((w[i, k] >= 0) for k in range(d[i]))
        m.addConstrs((w[i, k] >= (delta[i, k] - 1) + y[i, k]) for k in range(d[i]))
        m.addConstrs((w[i, k] <= y[i, k]) for k in range(d[i]))
        m.addConstrs((w[i, k] <= delta[i, k]) for k in range(d[i]))

        # Delta definition constraint
        m.addConstrs(delta[i, k] == u[i, k - d[i]] for k in range(d[i], Np + 1))

        # Mc Cormick's constraints
        m.addConstrs((w[i, k] >= 0) for k in range(d[i], Np + 1))
        m.addConstrs(
            (w[i, k] >= (delta[i, k] - 1) + y[i, k]) for k in range(d[i], Np + 1)
        )
        m.addConstrs((w[i, k] <= y[i, k]) for k in range(d[i], Np + 1))
        m.addConstrs((w[i, k] <= delta[i, k]) for k in range(d[i], Np + 1))

        # Model constraint
        m.addConstrs(
            (
                y[i, k + 1]
                == (a[i] + noise_a[i][k]) * y[i, k]
                + (b[i] + noise_b[i][k] - a[i] - noise_a[i][k]) * w[i, k]
                + (1 - b[i] - noise_b[i][k]) * delta[i, k]
            )
            for k in range(d[i], Np)
        )

    for k in range(1, Np + 1):
        # Max nb of activated valves constraint
        m.addConstr(u[:, k].sum() <= max_valves)

    # We want to minimize the total number of valve that openened during the prediction horizon
    m.setObjective(u.sum())

    # Relaxation
    vars_relax = [v for row in y.tolist() for v in row]
    lb_pen = [1e6] * (Np + 1) * N
    ub_pen = [1e6] * (Np + 1) * N
    m.feasRelax(
        relaxobjtype=0,
        minrelax=True,
        vars=vars_relax,
        lbpen=lb_pen,
        ubpen=ub_pen,
        constrs=None,
        rhspen=None,
    )

    m.optimize()

    return m, y.X, u.X


def plot(ynext, unext, N, Np):
    # Ensure we have enough colors for all valves/soils
    base_colors = plt.cm.tab10.colors  # 10 base colors
    colors = base_colors * ((N // 10) + 1)  # repeat if more than 10
    colors = colors[:N]  # slice exactly N colors

    # Plot soil moisture evolution
    plt.figure()
    for i in range(N):
        plt.plot(ynext[i, :], label=f"Soil {i+1}", color=colors[i])
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(list(range(Np + 1)))
    plt.xlabel("Time steps")
    plt.ylabel("Soil moisture level")
    plt.title("Moisture evolution in soil")

    # Plot valve activations
    plt.figure()
    for i in range(N):
        for t in range(Np + 1):
            if unext[i, t] > 0.5:
                plt.fill_between([t, t + 1], i, i + 1, color=colors[i], alpha=0.8)
    plt.xticks(list(range(Np + 1)))
    plt.yticks(range(N), [f"Valve {i+1}" for i in range(N)])
    plt.xlabel("Time steps")
    plt.ylabel("Valve")
    plt.title("Valve activation schedule")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()


m, y, u = N_valve_noise(
    std=np.array([0.03, 0.02, 0.02, 0.03, 0.01]),
    y0=np.array([0.6, 0.7, 0.66, 0.56, 0.69]),
    a=[0.95, 0.9, 0.9, 0.98, 0.85],
    b=[0.85, 0.72, 0.8, 0.75, 0.7],
    N=5,
    Np=20,
    d=[2, 4, 2, 3, 2],
    max_valves=3,
)
# m, y, u, w, delta = N_valve(N=3, max_valves=2)
plot(y, u, N=5, Np=20)
