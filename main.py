import pygame
import sys
from enviornment import WumpusEnv, load_img, pick_player, wumpus_img, pit_img, gold_img, exit_img
from a_star_agent import WumpusAStarAgent


class WumpusGame:
    """Main game controller supporting multiple modes"""
    
    def __init__(self, mode="manual", w=8, h=8, n_pits=6, n_wumpus=3, seed=123, wumpus_orientation="down"):
        """
        Initialize the game
        
        Args:
            mode: "manual", "astar", or other agent modes
            w, h: Grid dimensions
            n_pits: Number of pits
            n_wumpus: Number of wumpuses
            seed: Random seed for reproducibility
            wumpus_orientation: Direction wumpuses face
        """
        pygame.init()
        
        self.mode = mode
        self.CELL = 64
        self.ASSET_SIZE = (self.CELL - 8, self.CELL - 8)
        
        # Load assets
        self.assets = {
            "wumpus": load_img(wumpus_img, self.ASSET_SIZE),
            "player": load_img(pick_player(), self.ASSET_SIZE),
            "pit": load_img(pit_img, self.ASSET_SIZE),
            "gold": load_img(gold_img, self.ASSET_SIZE),
            "exit": load_img(exit_img, self.ASSET_SIZE),
        }
        
        # Create environment
        self.env = WumpusEnv(w=w, h=h, n_pits=n_pits, n_wumpus=n_wumpus, seed=seed, wumpus_orientation=wumpus_orientation)
        
        # Initialize agent if not manual mode
        self.agent = None
        if self.mode == "astar":
            self.agent = WumpusAStarAgent(self.env)
        # Add other agent types here as needed
        # elif self.mode == "qlearning":
        #     self.agent = QLearningAgent(self.env)
        
        # Setup display
        self.W = self.env.w * self.CELL
        self.H = self.env.h * self.CELL + 40
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption(f"Wumpus World - Mode: {self.mode.upper()}")
        self.clock = pygame.time.Clock()
        
        # Auto-play settings (for AI agents)
        self.auto_play = False
        self.step_delay = 300  # milliseconds between auto steps
        self.last_step_time = 0
        
        # Statistics
        self.episode_count = 0
        self.total_rewards = []
        
    def reset_game(self):
        """Reset the game environment and agent"""
        self.env.reset()
        self.assets["player"] = load_img(pick_player(), self.ASSET_SIZE)
        
        if self.agent:
            self.agent.reset(self.env)
        
        self.episode_count += 1
        print(f"\n{'='*50}")
        print(f"Episode {self.episode_count} - Environment Reset")
        print(f"{'='*50}\n")
    
    def handle_manual_input(self, event):
        """Handle manual player input"""
        if self.env.done:
            return
        
        keymap = {
            pygame.K_UP: 0, pygame.K_w: 0,
            pygame.K_DOWN: 1, pygame.K_s: 1,
            pygame.K_LEFT: 2, pygame.K_a: 2,
            pygame.K_RIGHT: 3, pygame.K_d: 3
        }
        
        if event.key in keymap:
            action = keymap[event.key]
            reward, done = self.env.step(action)
            
            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            print(f"Action: {action_names[action]}, Reward: {reward}, "
                  f"Position: {self.env.agent}, Total: {self.env.total}")
            
            self.render()
            
            if done:
                self.show_game_over(reward)
                if reward > 0:
                    self.total_rewards.append(self.env.total)
    
    def handle_agent_step(self):
        """Execute one step for AI agent"""
        if not self.agent or self.env.done:
            return False
        
        action = self.agent.decide_action()
        
        if action is not None:
            reward, done = self.env.step(action)
            
            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            print(f"Action: {action_names[action]}, Reward: {reward}, "
                  f"Position: {self.env.agent}, Total: {self.env.total}")
            
            self.render()
            
            if done:
                self.show_game_over(reward)
                self.total_rewards.append(self.env.total)
                return False
            return True
        else:
            print("Agent has no more actions available.")
            return False
    
    def render(self):
        """Render the current game state"""
        self.env.render(self.screen, self.CELL, self.assets)
        
        # Display mode indicator
        font = pygame.font.SysFont(None, 20)
        mode_text = font.render(f"Mode: {self.mode.upper()}", True, (200, 200, 200))
        self.screen.blit(mode_text, (self.W - 150, 8))
        
        pygame.display.flip()
    
    def show_game_over(self, reward):
        """Display game over overlay"""
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))
        
        font = pygame.font.SysFont(None, 36)
        msg = "You won!" if reward > 0 else "You died!"
        text = font.render(msg + "  (R to restart)", True, (255, 255, 255))
        self.screen.blit(text, text.get_rect(center=(self.W // 2, self.H // 2)))
        
        # Show statistics
        if self.total_rewards:
            stats_font = pygame.font.SysFont(None, 24)
            avg_reward = sum(self.total_rewards) / len(self.total_rewards)
            stats_text = stats_font.render(
                f"Episodes: {len(self.total_rewards)} | Avg Score: {avg_reward:.1f}",
                True, (200, 200, 200)
            )
            self.screen.blit(stats_text, stats_text.get_rect(center=(self.W // 2, self.H // 2 + 40)))
        
        pygame.display.flip()
        
        print(f"\n{'='*50}")
        print(f"Episode finished! Final score: {self.env.total}")
        if self.total_rewards:
            avg = sum(self.total_rewards) / len(self.total_rewards)
            print(f"Episodes completed: {len(self.total_rewards)}")
            print(f"Average score: {avg:.2f}")
        print(f"{'='*50}\n")
    
    def print_controls(self):
        """Print control instructions"""
        print("\n" + "="*60)
        print(f"WUMPUS WORLD - {self.mode.upper()} MODE")
        print("="*60)
        print("\nControls:")
        
        if self.mode == "manual":
            print("  Arrow Keys / WASD: Move agent")
        else:
            print("  SPACE: Toggle auto-play")
            print("  ENTER: Execute single step")
        
        print("  R: Reset environment")
        print("  M: Switch to Manual mode")
        print("  A: Switch to A* Agent mode")
        print("  Q / ESC: Quit")
        print("\n" + "="*60 + "\n")
    
    def switch_mode(self, new_mode):
        """Switch between game modes"""
        if new_mode == self.mode:
            return
        
        self.mode = new_mode
        pygame.display.set_caption(f"Wumpus World - Mode: {self.mode.upper()}")
        
        # Reinitialize agent for new mode
        if self.mode == "manual":
            self.agent = None
            self.auto_play = False
        elif self.mode == "astar":
            self.agent = WumpusAStarAgent(self.env)
        # Add other agent modes here
        
        self.reset_game()
        self.print_controls()
    
    def run(self):
        """Main game loop"""
        self.print_controls()
        self.render()
        
        running = True
        while running:
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    # Global controls
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    
                    elif event.key == pygame.K_r:
                        self.reset_game()
                        self.render()
                    
                    # Mode switching
                    elif event.key == pygame.K_m:
                        self.switch_mode("manual")
                    
                    elif event.key == pygame.K_a:
                        self.switch_mode("astar")
                    
                    # Mode-specific controls
                    elif self.mode == "manual":
                        self.handle_manual_input(event)
                    
                    else:  # AI agent mode
                        if event.key == pygame.K_SPACE:
                            self.auto_play = not self.auto_play
                            print(f"Auto-play: {'ON' if self.auto_play else 'OFF'}")
                        
                        elif event.key == pygame.K_RETURN:
                            self.handle_agent_step()
            
            # Auto-play for AI agents
            if self.mode != "manual" and self.auto_play and not self.env.done:
                if (current_time - self.last_step_time) > self.step_delay:
                    if not self.handle_agent_step():
                        self.auto_play = False
                    self.last_step_time = current_time
            
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()


def main():
    """Entry point - parse command line arguments and start game"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wumpus World AI Environment")
    parser.add_argument("--mode", type=str, default="manual", 
                        choices=["manual", "astar"],
                        help="Game mode: manual, astar")
    parser.add_argument("--width", type=int, default=8, help="Grid width")
    parser.add_argument("--height", type=int, default=8, help="Grid height")
    parser.add_argument("--pits", type=int, default=6, help="Number of pits")
    parser.add_argument("--wumpus", type=int, default=3, help="Number of wumpuses")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--orientation", type=str, default="down",
                        choices=["up", "down", "left", "right", "random"],
                        help="Wumpus orientation")
    
    args = parser.parse_args()
    
    game = WumpusGame(
        mode=args.mode,
        w=args.width,
        h=args.height,
        n_pits=args.pits,
        n_wumpus=args.wumpus,
        seed=args.seed,
        wumpus_orientation=args.orientation
    )
    
    game.run()


if __name__ == "__main__":
    main()
