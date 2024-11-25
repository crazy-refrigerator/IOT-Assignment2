import pulp
import matplotlib.pyplot as plt
import pandas as pd

# Unit costs per hour
unit_costs = [
    4.246522377, 3.640027796, 3.480502639, 3.245460995, 3.162915992, 3.597667495,
    3.905355954, 4.078340246, 5.374797426, 4.944699124, 5.438100083, 3.909231366,
    6.200666726, 4.482141894, 5.410801558, 6.149170969, 5.8837687, 6.329263208,
    6.469511152, 5.58349762, 5.558922379, 5.255354797, 5.568480613, 5.441475567
]

# Tasks for each user
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

# Linear programming problem
lp_prob = pulp.LpProblem("Minimize_Energy_Cost_With_Stability", pulp.LpMinimize)

# Decision variables for each task at each hour
energy_usage = {}
for user, task_list in tasks.items():
    for task_idx, (ready, deadline, max_energy, demand) in enumerate(task_list):
        for hour in range(24):
            energy_usage[(user, task_idx, hour)] = pulp.LpVariable(f"E_{user}_{task_idx}_{hour}", 0, max_energy)

# Hourly total cost variables
hourly_cost = {hour: pulp.LpVariable(f"HourlyCost_{hour}", 0) for hour in range(24)}
# Hourly difference variables for stability
hourly_diff = {hour: pulp.LpVariable(f"HourlyDiff_{hour}", 0) for hour in range(23)}

# Calculate hourly costs
for hour in range(24):
    lp_prob += (
        hourly_cost[hour] == pulp.lpSum(
            energy_usage[(user, task_idx, hour)] * unit_costs[hour]
            for user in tasks
            for task_idx, (ready, deadline, max_energy, demand) in enumerate(tasks[user])
            if ready <= hour <= deadline
        ),
        f"Cost_Calculation_{hour}"
    )

# Add stability constraints for hourly differences
for hour in range(23):
    lp_prob += (
        hourly_diff[hour] >= hourly_cost[hour + 1] - hourly_cost[hour],
        f"Positive_Diff_{hour}"
    )
    lp_prob += (
        hourly_diff[hour] >= hourly_cost[hour] - hourly_cost[hour + 1],
        f"Negative_Diff_{hour}"
    )

# Constraints for each task's total energy demand and time restrictions
for user, task_list in tasks.items():
    for task_idx, (ready, deadline, max_energy, demand) in enumerate(task_list):
        lp_prob += (
            pulp.lpSum(energy_usage[(user, task_idx, hour)] for hour in range(ready, deadline + 1)) == demand,
            f"Demand_{user}_{task_idx}"
        )

# Objective function: Minimize total cost and fluctuation
alpha = 0  # Weight for total cost
beta = 1  # Weight for stability
lp_prob += (
    alpha * pulp.lpSum(hourly_cost.values()) + beta * pulp.lpSum(hourly_diff.values()),
    "Total_Cost_and_Stability"
)

# Solve the problem
lp_prob.solve()

# Check if the solution is optimal
if pulp.LpStatus[lp_prob.status] == "Optimal":
    # Gather hourly costs
    hourly_cost_values = [pulp.value(hourly_cost[hour]) for hour in range(24)]
    # Calculate total cost
    total_cost = sum(hourly_cost_values)
    print(f"Total Cost: {total_cost:.2f} currency units")

    # Gather results for each user's hourly energy usage
    user_hourly_usage = {user: [0] * 24 for user in tasks}
    for user in tasks:
        for task_idx in range(10):
            for hour in range(24):
                energy = pulp.value(energy_usage[(user, task_idx, hour)]) or 0
                user_hourly_usage[user][hour] += energy

    # Calculate total hourly demand
    total_hourly_demand = [sum(user_hourly_usage[user][hour] for user in tasks) for hour in range(24)]

    # Plot hourly costs
    plt.figure(figsize=(12, 6))
    plt.plot(range(24), hourly_cost_values, marker='o', label="Hourly Costs")
    plt.xlabel("Hour")
    plt.ylabel("Cost")
    plt.title("Hourly Costs with Stability Constraint")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot stacked bar chart for energy usage
    bottom = [0] * 24
    plt.figure(figsize=(12, 6))
    for user, usage in user_hourly_usage.items():
        plt.bar(range(24), usage, bottom=bottom, label=user)
        bottom = [bottom[i] + usage[i] for i in range(24)]

    plt.xlabel("Hour")
    plt.ylabel("Total Energy Usage")
    plt.title("Hourly Energy Usage by User")
    plt.legend()
    plt.show()

    # Create a DataFrame to display results
    results_df = pd.DataFrame({"Hour": range(24), "Total Energy Demand": total_hourly_demand})
    print(results_df)
else:
    print("No optimal solution found.")
