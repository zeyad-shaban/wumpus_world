#the third idea is to maximise continuous functions using genetic algorithm
#f(x,y) = sin(x) * cos(y) + x + y 
import random
import math


def random_chromosome():
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    return (x, y)

def fitness(chromosome):
    x, y = chromosome
    return math.sin(x) * math.cos(y) + x + y

def numeric_corssover(parent1,parent2,alpha=0.5):
    x1, y1 = parent1
    x2, y2 = parent2

    child_x = alpha * x1 + (1 - alpha) * x2
    child_y = alpha * y1 + (1 - alpha) * y2
    return (child_x, child_y)

def numric_mutate(chromosome, mutation_rate=0.1,mutation_strength=0.5):
    x,y = chromosome
    if random.random() < mutation_rate:
        x += random.uniform(-mutation_strength, mutation_strength)
    if random.random() < mutation_rate:
        y += random.uniform(-mutation_strength, mutation_strength)
    
    #clamp values to range
    x = max(-10, min(10, x))
    y = max(-10, min(10, y))
    return (x,y)

def create_initial_population(size):
    population = []
    for _ in range(size):
        chromosome = random_chromosome()
        population.append(chromosome)
    return population

def select_parent(population):
    a = random.choice(population)
    b = random.choice(population)

    if fitness(a) > fitness(b):
        return a
    else:
        return b
    
def get_best(population):
    best = population[0]
    for chromo in population:
        if fitness(chromo) > fitness(best):
            best = chromo
    return best

def main():
    print("Genetic Algorithm to maximize f(x,y) = sin(x) * cos(y) + x + y")
    print("Running genetic algorithm...")
    population_size = 100
    generations = 200
    population = create_initial_population(population_size)

    print("Initial population:")
    for chromo in population:
        print(f"x = {chromo[0]:.3f}, y = {chromo[1]:.3f}, f(x,y) = {fitness(chromo):.3f}")
    print("-"*40)

    overall_best = get_best(population)
    overall_best_fitness = fitness(overall_best)

    for gen in range(generations):
        new_population = []
        for _ in range(population_size):
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            child = numeric_corssover(parent1, parent2)
            child = numric_mutate(child)
            new_population.append(child)
        population = new_population

        best = get_best(population)
        best_fitness = fitness(best)

        if best_fitness > overall_best_fitness:
            overall_best = best
            overall_best_fitness = best_fitness

        print(f"Generation {gen+1}: Best x = {best[0]:.3f}, y = {best[1]:.3f}, f(x,y) = {best_fitness:.3f}")
              
    print("=" * 40)
    print("Overall best solution:")
    print("x =", overall_best[0],
          "y =", overall_best[1],
          "f(x,y) =", overall_best_fitness)

if __name__ == "__main__":
    main()