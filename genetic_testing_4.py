#a mini wumpus trying to find goal i will think of some tweaks for this one 

import random

GRID_SIZE = 10
START = (0,0)
GOAL = (9,9)
MOVES = "UDLR"
MAX_STEPS = 30
WALL_PENALTY = 2

def random_chromosome():
    path = ""
    for _ in range(MAX_STEPS):
        move = random.choice(MOVES)
        path += move
    return path

def simulate_path(chromosome):
    x, y = START
    steps = 0
    wall_hits = 0
    reached_step = None

    for move in chromosome:
        new_x, new_y = x, y   # start from current

        # propose a move
        if move == 'U':
            new_y -= 1
        elif move == 'D':
            new_y += 1
        elif move == 'L':
            new_x -= 1
        elif move == 'R':
            new_x += 1

        # check if this would hit a wall
        if new_x < 0 or new_x >= GRID_SIZE or new_y < 0 or new_y >= GRID_SIZE:
            # wall hit: stay in place, count a penalty
            wall_hits += 1
        else:
            # valid move: update position
            x, y = new_x, new_y

        steps += 1

        # check if goal reached
        if (x, y) == GOAL and reached_step is None:
            reached_step = steps

    return x, y, steps, wall_hits, reached_step


# manhattan distance and invered 
def fitness(chromosome):
    x, y, steps, wall_hits, reached_step = simulate_path(chromosome) #ex: x=3, y=4
    gx,gy = GOAL #ex: gx=9, gy=9
    dist = abs(gx - x) + abs(gy - y) #ex: dist= (9-3)+(9-4)=11
    max_dist = 2 * (GRID_SIZE - 1) #ex: max_dist=18
    base_fit = max_dist - dist #ex: fit=18-11=7
    penalty = wall_hits * WALL_PENALTY #ex : 3 *2=6
    fit = base_fit - penalty 
    if reached_step is not None:
        fit += (MAX_STEPS - reached_step)  #bonus for reaching goal earlier
    return fit

def crossover(parent1, parent2):
    point = random.randint(1, MAX_STEPS - 1)
    left = parent1[:point]
    right = parent2[point:]
    child = left + right
    return child

def mutate(chromosome, mutation_rate=0.05):
    new_chromosome = ""
    for move in chromosome:
        if random.random() < mutation_rate:
            new_move = random.choice(MOVES)
            new_chromosome += new_move
        else:
            new_chromosome += move
    return new_chromosome

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

def apply_move(x, y, move):
    if move == 'U':
        y -= 1
    elif move == 'D':
        y += 1
    elif move == 'L':
        x -= 1
    elif move == 'R':
        x += 1

    # clamp inside grid
    if x < 0: x = 0
    if x >= GRID_SIZE: x = GRID_SIZE - 1
    if y < 0: y = 0
    if y >= GRID_SIZE: y = GRID_SIZE - 1

    return x, y

def visualize_path_simple(chromosome):
    """Simple ASCII visualization of the path"""
    print("\nPath visualization:")
    
    # Create empty grid
    grid = [['.' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    
    # Mark start and goal
    sx, sy = START
    gx, gy = GOAL
    grid[sy][sx] = 'S'
    grid[gy][gx] = 'G'
    
    # Trace the path
    x, y = START
    step_count = 0
    
    for move in chromosome:
        old_x, old_y = x, y  # Remember where we were
        
        # Apply the move
        x, y = apply_move(x, y, move)
        
        # Mark the path
        if grid[y][x] == '.' and (x, y) != GOAL:
            grid[y][x] = 'o'  # o for path
        elif (x, y) != GOAL:
            grid[y][x] = 'x'  # x for overlap
        
        step_count += 1
    
    # Print the grid
    print("  0 1 2 3 4 5 6 7 8 9")
    for y in range(GRID_SIZE):
        row = f"{y} "
        for x in range(GRID_SIZE):
            row += grid[y][x] + " "
        print(row)
    
    # Show final position
    print(f"\nPath length: {len(chromosome)} moves")
    print(f"Final position: ({x},{y})")
    print(f"Distance from goal: {abs(GOAL[0]-x) + abs(GOAL[1]-y)}")

def main():
    print("Genetic Algorithm to evolve path to goal in Wumpus Grid")
    print("Running genetic algorithm...")
    population_size = 100
    generations = 200
    population = create_initial_population(population_size)

    print("Initial population:")
    for chromo in population[:10]:  #print only first 10 for brevity
        print("Path:", chromo, "Fitness:", fitness(chromo))
    print("-"*40)

    overall_best = get_best(population)
    overall_best_fitness = fitness(overall_best)

    #evolve
    for gen in range(generations):
        new_population = []

        #elitism: carry over best solution
        best = get_best(population)
        new_population.append(best)

        #for tracking overall best
        if fitness(best) > overall_best_fitness:
            overall_best = best
            overall_best_fitness = fitness(best)

        for _ in range(population_size - 1):
            parent1 = select_parent(population)
            parent2 = select_parent(population)

            child = crossover(parent1, parent2)

            child = mutate(child)

            new_population.append(child)

        population = new_population

        best = get_best(population)
        print("Generation", gen + 1,
            "| Path =", best,
            "| Fitness =", fitness(best))
        #stop if goal reached
        #if fitness(best) == 2 * (GRID_SIZE - 1):
          #  print("Goal reached!")
          #  break

    final_best = get_best(population)
    print("=" * 40)
    print("Final best solution:")
    print("Path:", final_best, "Fitness:", fitness(final_best))
    visualize_path_simple(final_best)


if __name__ == "__main__":
    main()