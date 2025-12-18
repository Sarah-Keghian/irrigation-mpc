import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import os

dossier = "mes_plots"
os.makedirs(dossier, exist_ok=True)

Np = 20  # Prediction horizon (nb of time steps considered)
d = 2  # Delay between irrigation command and its effect
ymax = 0.7  # Maximum acceptanble soil moisture level
ymin = 0.4  # Minimum acceptanble soil moisture level
y0 = 0.7  # Initial moisture level
a = 0.9  # Sytem specific parameters representing natural moisture loss
b = 0.8  # Sytem specific parameters representing water absorption capacity

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
    m.addConstr(y[k + 1] == a * y[k])

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
    m.addConstr(y[k + 1] == a * y[k] + (b - a) * w[k] + (1 - b) * delta[k])

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
chemin = os.path.join(dossier, "valve_activation2.png")
plt.savefig(chemin)

plt.figure()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.xlabel("Time")
plt.ylabel("Soil moisture level")
plt.xticks(list(range(Np)))
plt.plot(list(range(Np)), y_best, marker="x")
plt.title(f"Moisture evolution in soil of parameter a={a}, b={b}")
chemin = os.path.join(dossier, "moisture_evolution2.png")
plt.savefig(chemin)

plt.show()
