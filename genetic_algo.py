import random

class GeneticcAlgorithm:
    
    def __init__(self,env, max_moves=50):
        self.max_moves = max_moves
        self.moves = "UDLR"  # possible moves
        self.env = env
        
    def start(self):
        pass
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