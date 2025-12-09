class WumpusBFS:
    """BFS-based agent for Wumpus World (Mock Template)"""
    
    def __init__(self, env):
        """
        Initialize the BFS agent
        
        Args:
            env: WumpusEnv instance
        """
        self.env = env
        pass
    
    def reset(self, env):
        """
        Reset the agent state for a new episode
        
        Args:
            env: WumpusEnv instance (possibly new environment)
        """
        self.env = env
        pass
    
    def decide_action(self):
        """
        Decide the next action to take
        
        Returns:
            int: Action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT) or None if no action available
        """
        # TODO: Implement BFS logic
        return None
