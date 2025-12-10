#this will be a simple genetic testing to try out genetic algotrithm

#lets start simple
#lets maximize a function f(x) = x^2 where x is an integer between 0 and 31
import random

development_mode = False


def random_chromosome():
    chromosome = ""
    for i in range(5):
        bt = random.choice(["0","1"])
        chromosome += bt
    return chromosome

def create_initial_population(size):
    population = []
    for _ in range(size):
        chromosome = random_chromosome()
        population.append(chromosome)
    return population

def decode_chromosome(chromosome):
    value = 0
    power = len(chromosome) - 1
    for bit in chromosome:
        if bit == "1":
            value += 2**power
        power -= 1
    
    return value

def fitness(chromosome):
    x = decode_chromosome(chromosome)
    return x*x

def get_best(population):
    best = population[0]
    for chromo in population:
        if fitness(chromo) > fitness(best):
            best = chromo
    return best

def select_parent(population):
    a = random.choice(population)
    b = random.choice(population)

    if fitness(a) > fitness(b):
        return a
    else:
        return b

def crossover(parent1, parent2):
    point = random.randint(1,4) #crossover point
    left = parent1[:point]
    right = parent2[point:]
    child = left + right
    return child

def mutate(chromosome, mutation_rate=0.01):
    new_chromosome = ""
    for bit in chromosome:
        if random.random() < mutation_rate:
            new_bit = "1" if bit == "0" else "0"
            new_chromosome += new_bit
        else:
            new_chromosome += bit
    return new_chromosome


def main():
    print("Genetic Algorithm to maximize f(x) = x^2 where x is in [0,31]")
    
    if development_mode:
        print("Development mode is ON")
        population = []
        population_size = 0
        generations = 20
        #allow user to input a chromosome and see its fitness
        while True:
            user_input = input("Enter a chromosome (5 bits) or 'q' to quit: ")
            if user_input == 'q':
                break
            if len(user_input) != 5 or any(c not in "01" for c in user_input):
                print("Invalid chromosome. Please enter a 5-bit binary string.")
                continue
            else:
             population.append(user_input)
             population_size += 1
    else:
        print("Running genetic algorithm...")
        population_size = 10
        generations = 20
        population = create_initial_population(population_size)

    print("Initial population:")
    for chromo in population:
        print(chromo,": ",decode_chromosome(chromo),"->", fitness(chromo))
    print("-"*40)

    #keep tracking best solution
    #overall_best = get_best(population)
    #overall_best_fitness = fitness(overall_best)


    #evolve
    for gen in range(generations):
        new_population = []

        #add elitism - carry over the best solution
        #best = get_best(population)
        #new_population.append(best)

        #if fitness(best) > overall_best_fitness:
            #overall_best = best
            #overall_best_fitness = fitness(best)

        #create new childern
        for i in range(population_size-1):
            #select parents:
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            #corss over between these parents:
            child = crossover(parent1,parent2)
            #mutate child 
            child = mutate(child,mutation_rate=0.1)
            #add child to new population
            new_population.append(child)

        #replace old population:
        population=new_population

        #find best solution in this population:

        best = get_best(population)
        print("Generation", gen + 1, "best:",
            best,
            "| x =", decode_chromosome(best),
            "| fitness =", fitness(best))
        
    final_best = get_best(population)
    print("=" * 40)
    print("Final best solution:")
    print("Chromosome:", final_best)
    print("x =", decode_chromosome(final_best))
    print("fitness =", fitness(final_best))

        
if __name__ == "__main__":
    main()

