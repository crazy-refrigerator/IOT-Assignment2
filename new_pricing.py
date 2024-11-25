import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd

# Define task information
tasks = {
    "User1": [(20, 23, 1, 1), (18, 23, 1, 2), (19, 21, 1, 1), (12, 20, 1, 3), (6, 12, 1, 3),
              (18, 20, 1, 2), (4, 10, 1, 2), (12, 18, 1, 2), (7, 14, 1, 3), (8, 14, 1, 3)],
    "User2": [(11, 22, 1, 2), (5, 11, 1, 2), (5, 23, 1, 1), (6, 20, 1, 3), (19, 19, 1, 1),
              (18, 21, 1, 2), (3, 23, 1, 3), (21, 23, 1, 2), (13, 17, 1, 1), (6, 11, 1, 2)],
    "User3": [(20, 23, 1, 2), (15, 21, 1, 3), (11, 15, 1, 2), (2, 17, 1, 3), (13, 16, 1, 2),
              (10, 18, 1, 2), (21, 23, 1, 2), (20, 23, 1, 1), (7, 21, 1, 2), (0, 7, 1, 3)],
    "User4": [(1, 8, 1, 1), (11, 20, 1, 2), (12, 19, 1, 3), (11, 16, 1, 3), (16, 18, 1, 1),
              (19, 23, 1, 3), (22, 23, 1, 1), (12, 19, 1, 2), (8, 20, 1, 2), (4, 12, 1, 2)],
    "User5": [(4, 20, 1, 1), (18, 22, 1, 3), (4, 16, 1, 1), (2, 16, 1, 3), (16, 23, 1, 2),
              (6, 18, 1, 2), (2, 6, 1, 1), (13, 17, 1, 3), (15, 23, 1, 1), (17, 23, 1, 1)]
}

# Define hourly range
hours = range(24)

# Decision variables: energy allocation per task per hour
energy_usage = {
    (user, task_idx, hour): cp.Variable(nonneg=True)
    for user, task_list in tasks.items()
    for task_idx, (start, end, max_energy, demand) in enumerate(task_list)
    for hour in hours
}

# Decision variables: total energy usage per hour
total_energy = {hour: cp.Variable(nonneg=True) for hour in hours}

# Auxiliary variables: linearization of total cost
linear_cost_vars = {hour: cp.Variable() for hour in hours}

# Objective function: minimize total cost
objective = cp.Minimize(cp.sum(list(linear_cost_vars.values())))

# Constraints
constraints = []

# Ensure total energy usage per hour matches task allocations
for hour in hours:
    constraints.append(
        total_energy[hour] == cp.sum(
            [
                energy_usage[(user, task_idx, hour)]
                for user, task_list in tasks.items()
                for task_idx, (start, end, max_energy, demand) in enumerate(task_list)
                if start <= hour <= end
            ]
        )
    )

# Linearize the quadratic cost function
for hour in hours:
    constraints.append(linear_cost_vars[hour] >= 0.5 * total_energy[hour] ** 2)

# Task energy allocation constraints
for user, task_list in tasks.items():
    for task_idx, (start, end, max_energy, demand) in enumerate(task_list):
        # Total energy demand constraint
        constraints.append(
            cp.sum(
                [energy_usage[(user, task_idx, hour)] for hour in range(start, end + 1)]
            ) == demand
        )
        # Per-hour maximum energy constraint
        for hour in range(start, end + 1):
            constraints.append(energy_usage[(user, task_idx, hour)] <= max_energy)

# Solve the optimization problem
problem = cp.Problem(objective, constraints)
problem.solve()

if problem.status == cp.OPTIMAL:
    # Extract hourly total energy usage
    hourly_total_energy = [total_energy[hour].value for hour in hours]

    # Compute unit cost per hour (0.5 * E_h)
    hourly_unit_prices = [0.5 * energy for energy in hourly_total_energy]

    # Compute total cost per hour (0.5 * E_h * E_h)
    hourly_prices = [unit_price * energy for unit_price, energy in zip(hourly_unit_prices, hourly_total_energy)]

    # Compute total cost
    total_cost = sum(hourly_prices)
    print(f"\nHourly Prices (0.5 * E_h * E_h): {hourly_prices}")
    print(f"\nTotal Cost (sum of 0.5 * E_h * E_h): {total_cost:.2f} currency units")

    # Compute total energy demand
    total_energy_demand = sum(demand for user_tasks in tasks.values() for _, _, _, demand in user_tasks)
    print(f"\nTotal Energy Demand: {total_energy_demand} units")

    # Display hourly total energy usage
    print("\nHourly Energy Usage:")
    for hour, energy in enumerate(hourly_total_energy):
        print(f"Hour {hour}: {energy:.2f} units")

    # Calculate energy contributions per user
    user_contributions = {user: [0] * len(hours) for user in tasks.keys()}
    for user, task_list in tasks.items():
        for task_idx, (start, end, max_energy, demand) in enumerate(task_list):
            for hour in range(start, end + 1):
                user_contributions[user][hour] += energy_usage[(user, task_idx, hour)].value

    # Plot hourly energy usage by user
    plt.figure(figsize=(12, 6))
    bottom = [0] * len(hours)
    for user, contributions in user_contributions.items():
        plt.bar(hours, contributions, bottom=bottom, label=user)
        bottom = [bottom[i] + contributions[i] for i in range(len(bottom))]

    plt.xlabel("Hour")
    plt.ylabel("Total Energy Usage")
    plt.title("Hourly Energy Usage by User for Minimum Cost")
    plt.xticks(hours)
    plt.legend(title="Users")
    plt.show()

    # Display hourly total energy usage in a dataframe
    results_df = pd.DataFrame({
        "Hour": hours,
        "Total Energy Usage": hourly_total_energy,
        "Unit Cost (0.5 * E_h)": hourly_unit_prices,
        "Total Cost per Hour (0.5 * E_h * E_h)": hourly_prices,
        **{user: contributions for user, contributions in user_contributions.items()}
    })
    print("\nHourly Total Energy Usage:")
    print(results_df)
else:
    print("No optimal solution found.")