import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Problem parameters
num_facilities = 3  # Number of facilities to select
num_customers = 10  # Number of customers
num_candidates = 20  # Number of candidate facility locations
search_space = np.random.rand(num_candidates, 2)  # Random coordinates for candidate locations
customer_locations = np.random.rand(num_customers, 2)  # Random coordinates for customers
customer_demand = np.random.randint(1, 10, size=num_customers)  # Random demand for customers
setup_cost = np.random.randint(100, 500, size=num_candidates)  # Random setup costs for facilities

# Algorithm parameters
num_employed_bees = 10  # Number of employed bees
num_onlooker_bees = 10  # Number of onlooker bees
num_scout_bees = 10  # Number of scout bees
max_iterations = 100  # Maximum number of iterations
limit = 10  # Scout bee threshold

# Visualization setup
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fitness_history = []

def update_visualization(iteration, best_solution, best_fitness):
    fitness_history.append(best_fitness)

    # Fitness plot
    ax[0].clear()
    ax[0].plot(fitness_history, label="Best Fitness")
    ax[0].set_title("Fitness Improvement Over Iterations")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Fitness Value")
    ax[0].legend()

    # Facility selection plot
    ax[1].clear()
    ax[1].scatter(search_space[:, 0], search_space[:, 1], label="Candidate Facilities", c="blue")
    ax[1].scatter(customer_locations[:, 0], customer_locations[:, 1], label="Customers", c="green")
    selected_locations = search_space[best_solution]
    ax[1].scatter(selected_locations[:, 0], selected_locations[:, 1], label="Selected Facilities", c="red")
    ax[1].set_title(f"Facility Selection (Iteration {iteration+1})")
    ax[1].legend()
    plt.pause(0.1)
    # time.sleep(2)  # Add delay for better visualization

# Distance function
def calculate_distance(facilities, customer):
    return np.linalg.norm(facilities - customer, axis=1)

# Fitness function
def fitness_function(solution):
    selected_facilities = search_space[solution]
    total_cost = np.sum(setup_cost[solution])

    transportation_cost = 0
    for customer_idx, customer in enumerate(customer_locations):
        distances = calculate_distance(selected_facilities, customer)
        nearest_distance = np.min(distances)
        transportation_cost += nearest_distance * customer_demand[customer_idx]

    return total_cost + transportation_cost

# Initialize population
def initialize_population():
    total_bees = num_employed_bees + num_onlooker_bees + num_scout_bees
    population = [np.random.choice(range(num_candidates), num_facilities, replace=False) for _ in range(total_bees)]
    return population

# Neighborhood search
def neighborhood_search(solution):
    new_solution = solution.copy()
    index_to_replace = np.random.randint(0, len(solution))
    while True:
        new_candidate = np.random.randint(0, num_candidates)
        if new_candidate not in new_solution:
            break
    new_solution[index_to_replace] = new_candidate
    return new_solution

# ABC algorithm
def abc_algorithm():
    population = initialize_population()
    fitness = [fitness_function(sol) for sol in population]

    trial = [0] * len(population)
    best_solution = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    for iteration in range(max_iterations):
        # Employed Bees Phase
        for i in range(num_employed_bees):
            new_solution = neighborhood_search(population[i])
            new_fitness = fitness_function(new_solution)

            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trial[i] = 0
            else:
                trial[i] += 1

        # Onlooker Bees Phase
        prob = np.array([1 / (1 + f) for f in fitness[:num_employed_bees]])
        prob /= np.sum(prob)

        for i in range(num_onlooker_bees):
            selected = np.random.choice(range(num_employed_bees), p=prob)
            new_solution = neighborhood_search(population[selected])
            new_fitness = fitness_function(new_solution)

            if new_fitness < fitness[selected]:
                population[selected] = new_solution
                fitness[selected] = new_fitness
                trial[selected] = 0

        # Scout Bees Phase
        for i in range(len(population)):
            if trial[i] > limit:
                population[i] = np.random.choice(range(num_candidates), num_facilities, replace=False)
                fitness[i] = fitness_function(population[i])
                trial[i] = 0

        # Update best solution
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_solution = population[current_best_idx]
            best_fitness = fitness[current_best_idx]

        # Update visualizations
        update_visualization(iteration, best_solution, best_fitness)

        print(f"Iteration {iteration + 1}/{max_iterations}, Best Fitness: {best_fitness}")

    return best_solution, best_fitness

# Run ABC algorithm
best_solution, best_fitness = abc_algorithm()
plt.show()
print("Best Facility Locations:", best_solution)
print("Best Fitness Value:", best_fitness)
