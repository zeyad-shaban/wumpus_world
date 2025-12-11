import random

class GeneticcAlgorithm:
    
    def __init__(self, env, max_moves=50, population_size=50, generations=100, mutation_rate=0.05, elitism=2):
        self.max_moves = max_moves
        self.moves = "UDLR"  # possible moves
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism = elitism  # number of best chromosomes to keep each generation
        
        # For tracking evolution progress
        self.current_generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_chromosome = None
        self.best_fitness = float('-inf')
        self.population = []
        self.is_evolving = False
        self.evolution_complete = False
        
    def start(self):
        """Initialize population and start evolution"""
        self.population = self.create_initial_population(self.population_size)
        self.current_generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_chromosome = None
        self.best_fitness = float('-inf')
        self.is_evolving = True
        self.evolution_complete = False
        
        # Evaluate initial population to find starting best
        self._update_best()
        
    def _update_best(self):
        """Update the best chromosome from current population"""
        total_fitness = 0
        for chromo in self.population:
            fit, reached_goal, steps = self.evaluate_chromosome(chromo, self.env)
            total_fitness += fit
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_chromosome = chromo
        
        avg_fitness = total_fitness / len(self.population) if self.population else 0
        self.avg_fitness_history.append(avg_fitness)
        self.best_fitness_history.append(self.best_fitness)
        
    def evolve_one_generation(self):
        """
        Run one generation of evolution.
        Returns: (generation_num, best_fitness, avg_fitness, is_complete)
        """
        if not self.is_evolving or self.evolution_complete:
            return self.current_generation, self.best_fitness, 0, True
            
        # Get fitness scores for all chromosomes
        fitness_scores = []
        for chromo in self.population:
            fit, _, _ = self.evaluate_chromosome(chromo, self.env)
            fitness_scores.append((chromo, fit))
        
        # Sort by fitness (descending)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Elitism: keep the best chromosomes
        new_population = [chromo for chromo, _ in fitness_scores[:self.elitism]]
        
        # Generate rest of new population through selection, crossover, mutation
        while len(new_population) < self.population_size:
            parent1 = self.select_parent(self.population)
            parent2 = self.select_parent(self.population)
            
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, self.mutation_rate)
            
            new_population.append(child)
        
        self.population = new_population
        self.current_generation += 1
        
        # Update best and history
        self._update_best()
        
        # Check if evolution is complete
        if self.current_generation >= self.generations:
            self.evolution_complete = True
            self.is_evolving = False
        
        avg_fitness = self.avg_fitness_history[-1] if self.avg_fitness_history else 0
        return self.current_generation, self.best_fitness, avg_fitness, self.evolution_complete
    
    def run_full_evolution(self):
        """
        Run the complete evolution process (non-visual mode).
        Returns the best chromosome found.
        """
        self.start()
        
        while not self.evolution_complete:
            gen, best_fit, avg_fit, complete = self.evolve_one_generation()
            print(f"Generation {gen}/{self.generations} | Best: {best_fit} | Avg: {avg_fit:.1f}")
        
        return self.best_chromosome
    
    def get_best_actions(self):
        """Get the actions for the best chromosome found"""
        if self.best_chromosome:
            return self.chromosome_to_actions(self.best_chromosome)
        return []
    
    def get_evolution_stats(self):
        """Get current evolution statistics"""
        return {
            'generation': self.current_generation,
            'total_generations': self.generations,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness_history[-1] if self.avg_fitness_history else 0,
            'is_complete': self.evolution_complete,
            'best_chromosome': self.best_chromosome
        }
    #Generate a random chromosome
    def random_chromosome(self):
        chromosome = ""
        for _ in range(self.max_moves):
            move = random.choice(self.moves)
            chromosome += move
        return chromosome
    
    #decode chromosome to actions ex: U=0, D=1, L=2, R=3
    def chromosome_to_actions(self, chromosome):
        """Convert chromosome string to list of action numbers"""
        # U=0, D=1, L=2, R=3 (matching env.step() expectations)
        move_map = {'U': 0, 'D': 1, 'L': 2, 'R': 3}
        actions = []
        for move in chromosome:
            action = move_map.get(move)
            if action is not None:
                actions.append(action)
        return actions
    
    #fitness function
    def evaluate_chromosome(self, chromosome, env):
        """
        Run a chromosome in the environment and return its fitness
        Returns: (final_score, reached_goal, steps_taken)
        """
        # Reset environment to starting position
        env.reset()
        
        # Convert chromosome to actions
        actions = self.chromosome_to_actions(chromosome)
        
        steps_taken = 0
        reached_goal = False
        
        # Execute each action
        for action in actions:
            reward, done = env.step(action)
            steps_taken += 1
            
            if done:
                # Check if we won (positive final reward means we got gold and exited)
                if reward > 0:
                    reached_goal = True
                break
        
        final_score = env.total
        return final_score, reached_goal, steps_taken
    
    #corssover between two chromosomes
    def crossover(self, parent1, parent2):
        point = random.randint(1, self.max_moves - 1)
        left = parent1[:point]
        right = parent2[point:]
        child = left + right
        return child
    #mutate a chromosome
    def mutate(self, chromosome, mutation_rate=0.05):
        new_chromosome = ""
        for move in chromosome:
            if random.random() < mutation_rate:
                new_move = random.choice(self.moves)
                new_chromosome += new_move
            else:
                new_chromosome += move
        return new_chromosome
    
    #def intialize population
    def create_initial_population(self, size):
        population = []
        for _ in range(size):
            chromosome = self.random_chromosome()
            population.append(chromosome)
        return population
    
    def select_parent(self, population):
        a = random.choice(population)
        b = random.choice(population)

        fit_a, _, _ = self.evaluate_chromosome(a, self.env)
        fit_b, _, _ = self.evaluate_chromosome(b, self.env)

        if fit_a > fit_b:
            return a
        else:
            return b
        
    def get_best(self, population):
        best = population[0]
        best_fit, _, _ = self.evaluate_chromosome(best, self.env)
        for chromo in population:
            fit, _, _ = self.evaluate_chromosome(chromo, self.env)
            if fit > best_fit:
                best = chromo
                best_fit = fit
        return best
    
