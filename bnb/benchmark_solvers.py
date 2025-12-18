import time
from pyomo.environ import *
import numpy as np


def create_model_N_valves(a, b, d, N, y0, Np, ymin, ymax, fixed_var, fixed_val):
    # Création du modèle Pyomo
    model = ConcreteModel()

    # Définition des ensembles
    model.I = RangeSet(0, N - 1)  # Ensemble des valves : {0, 1, ..., N-1}
    model.K = RangeSet(0, Np - 1)  # Ensemble des pas de temps : {0, 1, ..., Np-1}

    # Définition des variables
    model.u = Var(
        model.I,
        model.K,
        bounds=(0, 1),
        within=Reals,
        doc="Control variables of valves",
    )
    model.y = Var(
        model.I, model.K, bounds=(ymin, ymax), within=Reals, doc="Humidity variables"
    )
    model.w = Var(
        model.I, model.K, bounds=(0, 1), within=Reals, doc="McCormick variables"
    )
    model.delta = Var(
        model.I, model.K, bounds=(0, 1), within=Reals, doc="Delta variables"
    )

    # Contraintes initiales pour y[i, 0]
    def initial_moisture_rule(model, i):
        return model.y[i, 0] == y0[i]

    model.initial_moisture = Constraint(model.I, rule=initial_moisture_rule)

    # Contraintes pour les pas de temps 0 à d-1
    def early_steps_u_rule(model, i, k):
        return model.u[i, k] == 0

    model.early_steps_u = Constraint(
        model.I, RangeSet(0, d - 1), rule=early_steps_u_rule
    )

    def early_steps_delta_rule(model, i, k):
        return model.delta[i, k] == 0

    model.early_steps_delta = Constraint(
        model.I, RangeSet(0, d - 1), rule=early_steps_delta_rule
    )

    def early_steps_y_rule(model, i, k):
        return model.y[i, k + 1] == a * model.y[i, k]

    model.early_steps_y = Constraint(
        model.I, RangeSet(0, d - 1), rule=early_steps_y_rule
    )

    def early_steps_w_rule(model, i, k):
        return model.w[i, k] == 0

    model.early_steps_w = Constraint(
        model.I, RangeSet(0, d - 1), rule=early_steps_w_rule
    )

    # Contraintes pour les pas de temps d à Np-1
    def delta_definition_rule(model, i, k):
        return model.delta[i, k] == model.u[i, k - d]

    model.delta_definition = Constraint(
        model.I, RangeSet(d, Np - 1), rule=delta_definition_rule
    )

    def mccormick_rule_1(model, i, k):
        return model.w[i, k] >= 0

    model.mccormick_1 = Constraint(model.I, RangeSet(d, Np - 1), rule=mccormick_rule_1)

    def mccormick_rule_2(model, i, k):
        return model.w[i, k] >= (model.delta[i, k] - 1) + model.y[i, k]

    model.mccormick_2 = Constraint(model.I, RangeSet(d, Np - 1), rule=mccormick_rule_2)

    def mccormick_rule_3(model, i, k):
        return model.w[i, k] <= model.y[i, k]

    model.mccormick_3 = Constraint(model.I, RangeSet(d, Np - 1), rule=mccormick_rule_3)

    def mccormick_rule_4(model, i, k):
        return model.w[i, k] <= model.delta[i, k]

    model.mccormick_4 = Constraint(model.I, RangeSet(d, Np - 1), rule=mccormick_rule_4)

    def valve_limit_rule(model, k):
        return sum(model.u[i, k] for i in model.I) <= 1

    model.valve_limit = Constraint(RangeSet(d, Np - 1), rule=valve_limit_rule)

    # Contraintes pour les pas de temps d à Np-2
    def model_constraint_rule(model, i, k):
        return (
            model.y[i, k + 1]
            == a * model.y[i, k] + (b - a) * model.w[i, k] + (1 - b) * model.delta[i, k]
        )

    model.model_constraint = Constraint(
        model.I, RangeSet(d, Np - 2), rule=model_constraint_rule
    )

    # Fonction objectif : minimiser le nombre total de valves ouvertes
    def objective_rule(model):
        return sum(model.u[i, k] for i in model.I for k in model.K)

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # # Fixation des variables si nécessaire
    # for var_name, value in zip(fixed_var, fixed_val):
    #     # Supposons que fixed_var contient des tuples comme ("u", (i, k))
    #     var_type, (i, k) = var_name
    #     if var_type == "u":
    #         model.u[i, k].fix(value)
    #     elif var_type == "y":
    #         model.y[i, k].fix(value)
    #     elif var_type == "w":
    #         model.w[i, k].fix(value)
    #     elif var_type == "delta":
    #         model.delta[i, k].fix(value)

    return model


def benchmark_solver(model, solver_name):
    solver = SolverFactory(solver_name)
    start_time = time.time()
    solver.solve(model, tee=False)
    return time.time() - start_time


if __name__ == "__main__":
    model = create_model_N_valves(
        a=0.95,
        b=0.8,
        d=2,
        N=10,
        y0=[0.7, 0.6, 0.67, 0.68, 0.65, 0.64, 0.68, 0.66, 0.59, 0.58],
        Np=20,
        ymin=0.4,
        ymax=0.7,
        fixed_var=[],
        fixed_val=[],
    )

    solvers = ["cbc", "scip"]
    results = {}
    for solver_name in solvers:
        try:
            time_taken = benchmark_solver(model, solver_name)
            results[solver_name] = time_taken
            print(f"{solver_name}: {time_taken:.4f} secondes")
        except Exception as e:
            print(f"{solver_name}: Non disponible ou erreur")
            print(f"Erreur: {e}")

    if results:
        fastest_solver = min(results, key=results.get)
        print(
            f"\nLe solveur le plus rapide est : {fastest_solver} ({results[fastest_solver]:.4f} secondes)"
        )
    else:
        print("\nAucun solveur n'a pu résoudre le modèle.")

for obj in model.component_objects(Objective, active=True):
    print("Nom de l'objectif :", obj.name)
