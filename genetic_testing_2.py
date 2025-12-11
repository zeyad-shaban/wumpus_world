#next step genetic algorithm to evolve text towards a target phrase
import random

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ !?"
TARGET = "HELLO WUMPUS!"

def random_text_chromosome():
    chromosome = ""
    for i in range(len(TARGET)):
        chromosome += random.choice(ALPHABET)
    return chromosome

def fitness_text(chromosome):
    score = 0
    for i in range(min(len(TARGET), len(chromosome))):
        if chromosome[i] == TARGET[i]:
            score += 1
    return score

def corssover_text(parent1, parent2):
    point = random.randint(1, len(TARGET)-1)
    left = parent1[:point]
    right = parent2[point:]
    child = left + right
    return child

def mutate_text(chromosome, mutation_rate=0.1):
    new_chromosome = ""
    for char in chromosome:
        if random.random() < mutation_rate:
            new_char = random.choice(ALPHABET)
            new_chromosome += new_char
        else:
            new_chromosome += char
    return new_chromosome

def create_initial_population_text(size):
    population = []
    for i in range(size):
        chromosome = random_text_chromosome()
        population.append(chromosome)
    return population

def select_parent(population):
    a = random.choice(population)
    b = random.choice(population)

    if fitness_text(a) > fitness_text(b):
        return a
    else:
        return b

def get_best(population):
    best = population[0]
    for chromo in population:
        if fitness_text(chromo) > fitness_text(best):
            best = chromo
    return best

def main():
    print("Genetic Algorithm to evolve text towards target:", TARGET)
    print("Running genetic algorithm...")
    population_size = 100
    generations = 100
    population = create_initial_population_text(population_size)

    print("Initial population:")
    for chromo in population:
        print(chromo,":","->", fitness_text(chromo))
    print("-"*40)

    #keep tracking best solution
    overall_best = get_best(population)
    overall_best_fitness = fitness_text(overall_best)


    #evolve
    for gen in range(generations):
        new_population = []

        #add elitism - carry over the best solution
        best = get_best(population)
        new_population.append(best)

        if fitness_text(best) > overall_best_fitness:
            overall_best = best
            overall_best_fitness = fitness_text(best)

        #create new childern
        for i in range(population_size-1):
            #select parents:
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            #corss over between these parents:
            child = corssover_text(parent1,parent2)
            #mutate child 
            child = mutate_text(child,mutation_rate=0.1)
            #add child to new population
            new_population.append(child)

        #replace old population:
        population=new_population

        #find best solution in this population:

        best = get_best(population)
        print("Generation", gen + 1, "best:",
            best,
            "| x =", best,
            "| fitness =", fitness_text(best))
        
        if fitness_text(best) == len(TARGET):
            print("Target reached!")
            break
        
    final_best = get_best(population)
    print("=" * 40)
    print("Final best solution:")
    print("Chromosome:", final_best)
    print("x =", final_best)
    print("fitness =", fitness_text(final_best))

        
if __name__ == "__main__":
   main()
    

