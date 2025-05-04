# practical1---------------------------------------------------------

# client.py
import xmlrpc.client

# Connect to the server
proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

# Get user input
try:
    num = int(input("Enter a number to compute factorial: "))
    result = proxy.factorial(num)
    print(f"Factorial of {num} is: {result}")
except ValueError:
    print("Please enter a valid integer.")

# server.py
from xmlrpc.server import SimpleXMLRPCServer
import math

def factorial(n):
    if n < 0:
        return "Error: Negative numbers not allowed"
    return math.factorial(n)

# Create server
server = SimpleXMLRPCServer(("localhost", 8000))
print("Server started on port 8000...")

# Register function
server.register_function(factorial, "factorial")

# Run the server's main loop
server.serve_forever()

# practical2--------------------------------------------------------------

# client.py

import Pyro4

# Use the URI printed by the server, e.g., PYRO:obj_xxxxxx@localhost:port
uri = input("Enter the URI of the remote object (e.g., PYRO:obj_xxx@localhost:port): ")

string_concatenator = Pyro4.Proxy(uri)

str1 = input("Enter first string: ")
str2 = input("Enter second string: ")

result = string_concatenator.concatenate(str1, str2)
print("Result from server:", result)


# server.py

import Pyro4

@Pyro4.expose
class StringConcatenator:
    def concatenate(self, str1, str2):
        return str1 + str2

# Start the Pyro daemon and register the object
daemon = Pyro4.Daemon()
uri = daemon.register(StringConcatenator)

# Optionally register with name server
print("Server is ready.")
print("Object URI:", uri)

daemon.requestLoop()


# practical3----------------------------------------------------------------

# char_count.py

from mrjob.job import MRJob

class MRCharCount(MRJob):
    def mapper(self, _, line):
        for char in line.strip():
            yield char, 1
    
    def reducer(self, char, counts):
        yield char, sum(counts)

if __name__ == '__main__':
    MRCharCount.run()

# word_count.py

from mrjob.job import MRJob
import re

WORD_REGEXP = re.compile(r"[\w']+")
class MRWordCount(MRJob):
    def mapper(self, _, line):
        for word in WORD_REGEXP.findall(line):
            yield word.lower(), 1

    def reducer(self, word, counts):
        yield word, sum(counts)

if __name__ == '__main__':
    MRWordCount.run()

# practical4-------------------------------------------------------------------------


# load_balnacer.py

# both the codes are right

# import random

# class LoadBalancer:
#     def __init__(self, servers):
#         self.servers = servers
#         self.round_robin_index = 0  # Track the index for Round Robin

#     def round_robin(self):
#         """Round Robin load balancing algorithm"""
#         while True:
#             yield self.servers[self.round_robin_index]
#             self.round_robin_index = (self.round_robin_index + 1) % len(self.servers)

#     def random_selection(self):
#         """Random load balancing algorithm"""
#         while True:
#             yield random.choice(self.servers)

#     def least_connection(self):
#         """Least Connections load balancing algorithm"""
#         while True:
#             min_connections = min(self.servers, key=lambda x: x.connections)
#             min_connections.connections += 1  # Simulate a new connection
#             yield min_connections


# class Server:
#     def __init__(self, name):
#         self.name = name
#         self.connections = 0

#     def handle_request(self):
#         """Simulate handling a request by the server"""
#         print(f"Server {self.name} is handling the request. Total connections: {self.connections}")
#         self.connections -= 1  # Simulate that the server finishes handling the request


# def simulate_requests(load_balancer, num_requests):
#     """Simulate handling multiple requests"""
#     print(f"Simulating {num_requests} requests...\n")
#     for i in range(num_requests):
#         server = next(load_balancer)
#         print(f"Request {i + 1} handled by {server.name}")
#         server.connections += 1  # Simulate a new request coming in
#         server.handle_request()
#         print(f"Request {i + 1} complete.\n")
#     print("Simulation complete.")


# if __name__ == "__main__":
#     # Create servers
#     server1 = Server("Server1")
#     server2 = Server("Server2")
#     server3 = Server("Server3")
#     servers = [server1, server2, server3]

#     # Create load balancer
#     lb = LoadBalancer(servers)

#     # Choose which load balancing algorithm to use
#     # Uncomment one of the following load balancer algorithms
#     load_balancer = lb.round_robin()        # Round Robin
#     # load_balancer = lb.random_selection()  # Random Selection
#     # load_balancer = lb.least_connection()  # Least Connections

#     # Simulate 10 requests
#     simulate_requests(load_balancer, 10)


# chatgpt code

import random
import time

class Server:
    def __init__(self, name):
        self.name = name
        self.connections = 0  # Track the number of connections for least connections

    def handle_request(self):
        """Simulate handling a request"""
        self.connections += 1
        print(f"Server {self.name} is handling the request. Total connections: {self.connections}")
        time.sleep(1)  # Simulate the time it takes to handle a request
        self.connections -= 1


class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.round_robin_index = 0

    def round_robin(self):
        """Distribute requests using Round Robin"""
        server = self.servers[self.round_robin_index]
        self.round_robin_index = (self.round_robin_index + 1) % len(self.servers)
        return server

    def least_connections(self):
        """Distribute requests based on the server with the least connections"""
        server = min(self.servers, key=lambda s: s.connections)
        return server

    def random_selection(self):
        """Distribute requests randomly"""
        return random.choice(self.servers)

    def distribute_request(self, algorithm="round_robin"):
        """Distribute a request using the selected algorithm"""
        if algorithm == "round_robin":
            server = self.round_robin()
        elif algorithm == "least_connections":
            server = self.least_connections()
        elif algorithm == "random":
            server = self.random_selection()
        else:
            raise ValueError("Unknown algorithm")
        
        server.handle_request()


# Create servers
servers = [Server("A"), Server("B"), Server("C")]

# Create load balancer
lb = LoadBalancer(servers)

# Simulate requests being distributed
for i in range(10):  # Simulating 10 client requests
    print(f"\nRequest {i+1}:")
    lb.distribute_request(algorithm="least_connections")  # You can change the algorithm here


# practical5-----------------------------------------------------------------

# clonal.py
import random
import numpy as np

# Define a simple fitness function: f(x) = sum(x^2)
# Here x is a 3D vector, so the fitness function sums the squares of the components.
def fitness(x):
    return np.sum(np.square(x))

# Parameters
population_size = 10        # Number of antibodies (solutions)
max_generations = 100       # Number of generations
mutation_rate = 0.1        # Mutation rate
cloning_rate = 0.5         # Fraction of the best population to clone
tournament_size = 3        # Tournament selection size

# Initialize population (random 3D vectors)
def initialize_population(population_size):
    return np.random.uniform(-10, 10, (population_size, 3))  # 3D vectors with values between -10 and 10

# Tournament selection for the best antibody
def tournament_selection(population, fitness_values, tournament_size):
    selected = random.sample(list(zip(population, fitness_values)), tournament_size)
    selected.sort(key=lambda x: x[1])  # Sort by fitness (lower is better)
    return selected[0][0]  # Return the best (with lowest fitness)

# Clonal selection algorithm
def clonal_selection_algorithm():
    population = initialize_population(population_size)
    best_solution = None
    best_fitness = float('inf')
    
    for generation in range(max_generations):
        # Evaluate fitness of the population
        fitness_values = [fitness(x) for x in population]

        # Find the best antibody (solution)
        min_fitness_index = np.argmin(fitness_values)
        best_antibody = population[min_fitness_index]
        current_best_fitness = fitness_values[min_fitness_index]

        if current_best_fitness < best_fitness:
            best_solution = best_antibody
            best_fitness = current_best_fitness
        
        print(f"Generation {generation}: Best Solution = {best_solution}, Fitness = {best_fitness}")

        # Clone the best antibodies
        clones = []
        num_clones = int(population_size * cloning_rate)
        for _ in range(num_clones):
            clone = best_antibody
            mutated_clone = clone + np.random.normal(0, mutation_rate, 3)  # Mutation in 3D space
            clones.append(mutated_clone)

        # Replace worst antibodies with clones
        population_sorted = sorted(zip(population, fitness_values), key=lambda x: x[1])
        for i in range(num_clones):
            population_sorted[i] = (clones[i], fitness(clones[i]))

        population = [x[0] for x in population_sorted]

    # Return the best solution found
    return best_solution

# Run the algorithm
best_solution = clonal_selection_algorithm()
print(f"Best solution found: {best_solution}")

# practial6----------------------------------------------------
# None: The code for practical6 is not provided in the original snippet. Please provide the code for practical6 if you want me to include it here.

# practical7----------------------------------------------------------

# damage.py

#install numpy, scikit-learn

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Define the Artificial Immune System (AIS) model for pattern recognition
class AIPR:
    def __init__(self, population_size=50, generations=10, mutation_rate=0.1, clone_factor=5):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.clone_factor = clone_factor

    def initialize_population(self, n_features, n_classes):
        # Randomly generate population
        population = np.random.rand(self.population_size, n_features)
        labels = np.random.choice(n_classes, self.population_size)
        return population, labels


    def fitness(self, population, X_train, y_train):
    # Simple fitness function: using Euclidean distance to target class centers
        fitness_scores = []
        for ind, individual in enumerate(population):
            distance = np.linalg.norm(X_train - individual, axis=1)  # Calculate distances
            predicted_labels = np.argmin(distance)  # Find the index of the minimum distance
            accuracy = np.sum(predicted_labels == y_train) / len(y_train)
            fitness_scores.append(accuracy)
        return np.array(fitness_scores)


    def selection(self, population, fitness_scores):
        # Select individuals with the best fitness scores
        selected_indices = np.argsort(fitness_scores)[-self.clone_factor:]
        return population[selected_indices]

    def mutation(self, population):
        # Mutate individuals with a certain probability
        for ind in range(len(population)):
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.rand(population.shape[1])
                population[ind] += mutation
        return population

    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))

        # Initialize population
        population, labels = self.initialize_population(n_features, n_classes)

        # Iterate through generations
        for generation in range(self.generations):
            print(f"Generation {generation+1}/{self.generations}")
            fitness_scores = self.fitness(population, X_train, y_train)
            
            # Selection
            selected_population = self.selection(population, fitness_scores)

            # Cloning & Mutation
            cloned_population = np.tile(selected_population, (self.clone_factor, 1))
            mutated_population = self.mutation(cloned_population)

            # Reassign new population
            population = mutated_population

        # Return final population and fitness for testing
        return population

    def predict(self, X_test, population):
        # Predict labels based on closest individual in the population
        distance = np.linalg.norm(X_test[:, np.newaxis] - population, axis=2)
        predicted_labels = np.argmin(distance, axis=1)
        return predicted_labels

# Generating synthetic data for structure damage classification
# X, y = make_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)
X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=3, n_clusters_per_class=1, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize and train the AIPR classifier
aipr = AIPR(population_size=50, generations=10, mutation_rate=0.1, clone_factor=5)
population = aipr.train(X_train, y_train)

# Predict and evaluate
y_pred = aipr.predict(X_test, population)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

# practical8--------------------------------------------------------------------------

# DEAP.py
# pip install deap

import random
import numpy as np
from deap import base, creator, tools, algorithms

# Step 1: Define the Fitness and Individual Classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Step 2: Initialize the Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Step 3: Define the Evaluation Function

def evalOneMax(individual):
    return sum(individual),

# Step 4: Register the Evaluation Function with the Toolbox
toolbox.register("evaluate", evalOneMax)
# Step 5: Define the Operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
# Step 6: Define the Main Loop of the Evolutionary Algorithm
def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,

    stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof

# Step 7: Run the Algorithm
if __name__ == "__main__":
    pop, log, hof = main()
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness.values))


# practical9----------------------------------------------------------------------

# temp.py

from mrjob.job import MRJob
import csv

class MRCoolestHottestYear(MRJob):
    
    def mapper(self, _, line):
        # Read the CSV data (skip the header if present)
        reader = csv.reader([line])
        for row in reader:
            if row[0] != 'Year':  # Skip the header if present
                year = row[0]
                temperature = float(row[1])
                # Emit each year with its corresponding temperature
                yield year, temperature
    
    def reducer(self, year, temperatures):
        # Initialize max and min temperatures
        max_temp = float('-inf')
        min_temp = float('inf')
        
        # Iterate over all temperatures for a specific year
        for temp in temperatures:
            max_temp = max(max_temp, temp)
            min_temp = min(min_temp, temp)
        
        # Emit the result: the hottest and coolest temperatures for the year
        yield f"Year {year}: Hottest = {max_temp}, Coolest = {min_temp}", None


if __name__ == '__main__':
    MRCoolestHottestYear.run()

# practical10------------------------------------------------------------------------------------

# ant.py

import random
import numpy as np

# Parameters
num_ants = 10           # Number of ants
num_iterations = 100     # Number of iterations
alpha = 1               # Influence of pheromone
beta = 5                # Influence of distance
evaporation_rate = 0.5   # Rate at which pheromone evaporates
pheromone_init = 1      # Initial pheromone value
Q = 100                 # Total pheromone deposited by each ant

# Distance matrix (example)
distances = np.array([[0, 10, 12, 11, 14],
                      [10, 0, 13, 15, 10],
                      [12, 13, 0, 9, 8],
                      [11, 15, 9, 0, 7],
                      [14, 10, 8, 7, 0]])

# Number of cities
num_cities = len(distances)

# Initialize pheromone matrix
pheromone = np.ones((num_cities, num_cities)) * pheromone_init

def calculate_route_length(route, distances):
    """Calculate the total length of the route."""
    length = 0
    for i in range(len(route) - 1):
        length += distances[route[i], route[i + 1]]
    length += distances[route[-1], route[0]]  # Returning to the starting city
    return length

def select_next_city(ant, current_city, pheromone, distances, alpha, beta):
    """Select the next city based on pheromone levels and distances."""
    probabilities = []
    total = 0
    for i in range(num_cities):
        if i not in ant['visited']:
            pheromone_strength = pheromone[current_city][i] ** alpha
            distance_strength = (1 / distances[current_city][i]) ** beta
            probability = pheromone_strength * distance_strength
            total += probability
            probabilities.append(probability)
        else:
            probabilities.append(0)

    # Normalize probabilities
    probabilities = [p / total for p in probabilities]

    # Select next city based on probabilities
    next_city = random.choices(range(num_cities), weights=probabilities, k=1)[0]
    return next_city

def update_pheromone(pheromone, ants, evaporation_rate, Q, distances):
    """Update the pheromone levels based on the ants' tours."""
    # Evaporate pheromone
    pheromone *= (1 - evaporation_rate)

    # Update pheromone with new deposits
    for ant in ants:
        length = calculate_route_length(ant['route'], distances)
        pheromone_deposit = Q / length
        for i in range(len(ant['route']) - 1):
            pheromone[ant['route'][i], ant['route'][i + 1]] += pheromone_deposit
        pheromone[ant['route'][-1], ant['route'][0]] += pheromone_deposit  # Return to start city

def ant_colony_optimization(distances, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_init, Q):
    """Solve the TSP using Ant Colony Optimization."""
    pheromone = np.ones((num_cities, num_cities)) * pheromone_init
    best_route = None
    best_length = float('inf')

    for iteration in range(num_iterations):
        ants = []
        for _ in range(num_ants):
            ant = {'visited': [], 'route': []}
            start_city = random.randint(0, num_cities - 1)
            ant['visited'].append(start_city)
            ant['route'].append(start_city)

            # Construct a tour
            for _ in range(num_cities - 1):
                current_city = ant['route'][-1]
                next_city = select_next_city(ant, current_city, pheromone, distances, alpha, beta)
                ant['visited'].append(next_city)
                ant['route'].append(next_city)

            # Update the best solution
            length = calculate_route_length(ant['route'], distances)
            if length < best_length:
                best_route = ant['route']
                best_length = length

        # Update pheromone
        update_pheromone(pheromone, ants, evaporation_rate, Q, distances)

    return best_route, best_length

# Run ACO
best_route, best_length = ant_colony_optimization(distances, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_init, Q)

print("Best route:", best_route)
print("Best route length:", best_length)

