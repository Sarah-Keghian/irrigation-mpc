import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

dossier = "mes_plots"
os.makedirs(dossier, exist_ok=True)


def plot(ynext, unext, N, Np):
    """
    Plot the simulation results:
    - soil moisture evolution for each plot
    - valve activation schedule over time

    Parameters
    ----------
    ynext : np.ndarray (N, Np+1)
        Soil moisture trajectories for each plot and time step
    unext : np.ndarray (N, Np+1)
        Binary valve control signals (1 = open, 0 = closed)
    N : int
        Number of plots / valves
    Np : int
        Prediction horizon (number of time steps)
    """
    colors = plt.cm.tab10.colors
    if N > 10:
        colors = colors * ((N // 10) + 1)

    plt.figure()
    for i in range(N):
        plt.plot(ynext[i, :], label=f"Soil {i+1}", color=colors[i])
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(list(range(Np + 1)))
    plt.xlabel("Time steps")
    plt.ylabel("Soil moisture level")
    plt.title("Moisture evolution in soil of parameter a, b, d")

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


def N_valve_initial_noise(
    y0_noise: np.ndarray,
    u0: np.ndarray,
    uinf0: np.ndarray,
    max_valves: int,
    N: int,
    d: list,
    Np: int,
    ymax: float,
    ymin: float,
    a: list,
    b: list,
):
    """
    Solve a Gurobi optimization problem to compute the optimal valve
    activation strategy under noisy initial conditions.

    The model accounts for:
    - soil moisture dynamics
    - valve actuation delays
    - a constraint on the maximum number of simultaneously open valves

    Parameters
    ----------
    y0_noise : np.ndarray (N,)
        Noisy initial soil moisture values
    u0 : np.ndarray (N,)
        Control applied at the previous time step
    uinf0 : np.ndarray (N, max(d))
        Past control history used to handle valve delays
    max_valves : int
        Maximum number of valves that can be open simultaneously
    N : int
        Number of plots / valves
    d : list[int]
        Valve actuation delays
    Np : int
        Prediction horizon
    ymax : float
        Upper bound on soil moisture
    ymin : float
        Lower bound on soil moisture
    a : list[float]
        Soil moisture decay coefficients
    b : list[float]
        Irrigation efficiency coefficients

    Returns
    -------
    y : np.ndarray (N, Np+1)
        Optimal soil moisture trajectories
    u : np.ndarray (N, Np+1)
        Optimal valve control signals
    """
    max_d = max(d)

    m = gp.Model("N_valve_noise")
    m.Params.OutputFlag = 0
    y = m.addMVar((N, Np + 1), vtype=GRB.CONTINUOUS, lb=ymin, ub=ymax)
    u = m.addMVar((N, Np + 1), vtype=GRB.BINARY)
    w = m.addMVar((N, Np + 1), vtype=GRB.CONTINUOUS)
    delta = m.addMVar((N, Np + 1), vtype=GRB.BINARY)

    # Define initial moisture
    for i in range(N):
        m.addConstr(y[i, 0] == y0_noise[i])

    # print(d)
    for i in range(N):
        m.addConstrs(delta[i, k] == uinf0[i, max_d - d[i] + k] for k in range(d[i]))

        # Apply previous control choice
        m.addConstr(u[i, 0] == u0[i])

        # Delta definition constraint
        m.addConstrs(delta[i, k] == u[i, k - d[i]] for k in range(d[i], Np + 1))

        # Mc Cormick's constraints
        m.addConstrs((w[i, k] >= 0) for k in range(Np + 1))
        m.addConstrs((w[i, k] >= (delta[i, k] - 1) + y[i, k]) for k in range(Np + 1))
        m.addConstrs((w[i, k] <= y[i, k]) for k in range(Np + 1))
        m.addConstrs((w[i, k] <= delta[i, k]) for k in range(Np + 1))

        # Model constraint
        m.addConstrs(
            (
                y[i, k + 1]
                == a[i] * y[i, k] + (b[i] - a[i]) * w[i, k] + (1 - b[i]) * delta[i, k]
            )
            for k in range(Np)
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
    # plot(ynext=y.X, unext=u.X, N=N, Np=Np)
    return y.X, u.X


def solve(
    a: list,
    b: list,
    d: list,
    y0: np.ndarray,
    std: np.ndarray,
    Np: int,
    N: int,
    max_valves: int,
    ymax=0.7,
    ymin=0.4,
):
    """
    Run a closed-loop simulation with noisy state measurements.

    At each time step:
    - noise is added to the measured soil moisture
    - an optimization problem is solved
    - only the first optimal control action is applied (MPC principle)

    Parameters
    ----------
    a : list[float]
        Soil dynamics coefficients
    b : list[float]
        Irrigation coefficients
    d : list[int]
        Valve actuation delays
    y0 : np.ndarray (N,)
        Initial soil moisture values
    std : np.ndarray (N,)
        Standard deviation of the measurement noise for each soil
    Np : int
        Prediction horizon
    N : int
        Number of plots / valves
    max_valves : int
        Maximum number of valves open simultaneously
    ymax : float, optional
        Maximum allowed soil moisture
    ymin : float, optional
        Minimum allowed soil moisture

    Returns
    -------
    None
        Results are displayed using the plot function
    """
    ynext = np.zeros((N, Np + 1))
    unext = np.zeros((N, Np + 1))
    u0 = np.array([0] * N)
    max_d = max(d)
    ynext[:, 0] = y0
    unext[:, 0] = u0
    for k in range(1, Np + 1):
        noise = np.random.normal(loc=0.0, scale=1.0, size=N) * std
        y0_noise = y0 + noise

        u_window = np.zeros((N, max_d))

        for i in range(N):
            delay = d[i]
            n_prev = min(delay, k)
            if n_prev > 0:
                u_window[i, -n_prev:] = unext[i, k - n_prev : k]

        # print(k)
        # print(u_window)

        y, u = N_valve_initial_noise(
            y0_noise=y0_noise,
            u0=u0,
            max_valves=max_valves,
            d=d,
            Np=Np,
            ymax=ymax,
            ymin=ymin,
            a=a,
            b=b,
            N=N,
            uinf0=u_window,
        )

        y0 = [y[i, 1] for i in range(N)]
        u0 = [u[i, 1] for i in range(N)]
        ynext[:, k] = y0
        unext[:, k] = u0

    return ynext, unext, N, Np


import cProfile
import pstats

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    ynext, unext, N, Np = solve(
        std=np.array([0.03, 0.02, 0.02, 0.03, 0.01, 0.01, 0.04, 0.03, 0.04, 0.02]),
        y0=np.array([0.6, 0.7, 0.66, 0.56, 0.69, 0.62, 0.58, 0.59, 0.65, 0.67]),
        a=[0.95, 0.9, 0.9, 0.98, 0.85, 0.92, 0.88, 0.93, 0.9, 0.94],
        b=[0.85, 0.72, 0.8, 0.75, 0.7, 0.78, 0.74, 0.8, 0.76, 0.79],
        d=[2, 4, 2, 3, 2, 3, 1, 4, 2, 3],
        N=10,
        Np=20,
        max_valves=5,
    )

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(30)
