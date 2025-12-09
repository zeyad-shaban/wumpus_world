import pygame
import sys
from enviornment import WumpusEnv, load_img, pick_player, wumpus_img, pit_img, gold_img, exit_img, ARROW_IMG_PATH
from bfs_agent import WumpusBFS
from agent_visualizer import AgentInspectorWindow


class WumpusGame:
    """Main game controller supporting multiple modes"""

    def __init__(self, mode="manual", w=8, h=8, n_pits=6, n_wumpus=3, n_arrows=1, seed=123, wumpus_orientation="down"):
        """
        Initialize the game
        """
        pygame.init()

        self.mode = mode
        self.CELL = 80
        self.ASSET_SIZE = (self.CELL - 8, self.CELL - 8)

        # Fullscreen state
        self.is_fullscreen = False
        self.windowed_size = None

        # Load assets
        self.assets = {
            "wumpus": load_img(wumpus_img, self.ASSET_SIZE),
            "player": load_img(pick_player(), self.ASSET_SIZE),
            "pit": load_img(pit_img, self.ASSET_SIZE),
            "gold": load_img(gold_img, self.ASSET_SIZE),
            "exit": load_img(exit_img, self.ASSET_SIZE),
            "arrow": load_img(ARROW_IMG_PATH, self.ASSET_SIZE),
        }

        # Create environment
        self.env = WumpusEnv(w=w, h=h, n_pits=n_pits, n_wumpus=n_wumpus, n_arrows=n_arrows, seed=seed,
                             wumpus_orientation=wumpus_orientation)

        # Initialize agent if not manual mode
        self.agent = None
        if self.mode == "bfs":
            self.agent = WumpusBFS(self.env)

        # Setup display dimensions
        self.game_W = self.env.w * self.CELL
        self.game_H = self.env.h * self.CELL + 40
        
        # Inspector panel width (only shown when agent is active)
        self.inspector_W = 320
        
        # Total window size depends on mode
        if self.agent:
            self.W = self.game_W + self.inspector_W
        else:
            self.W = self.game_W
        self.H = self.game_H
        
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption(f"Wumpus World - Mode: {self.mode.upper()}")
        self.clock = pygame.time.Clock()

        # Auto-play settings (for AI agents)
        self.auto_play = False
        self.step_delay = 50
        self.last_step_time = 0

        # Statistics
        self.episode_count = 0
        self.total_rewards = []

        # Initialize Agent Inspector
        if self.agent:
            self.agent_inspector = AgentInspectorWindow(
                agent=self.agent,
                grid_width=self.env.w,
                grid_height=self.env.h
            )
        else:
            self.agent_inspector = None

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        if not self.is_fullscreen:
            # Save current windowed size
            self.windowed_size = (self.W, self.H)
            # Switch to fullscreen
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.is_fullscreen = True
            
            # Get actual fullscreen size
            info = pygame.display.Info()
            self.screen_w = info.current_w
            self.screen_h = info.current_h
            
            print(f"Switched to fullscreen: {self.screen_w}x{self.screen_h}")
        else:
            # Restore windowed mode
            if self.windowed_size:
                self.W, self.H = self.windowed_size
            self.screen = pygame.display.set_mode((self.W, self.H))
            self.is_fullscreen = False
            print("Switched to windowed mode")

    def _resize_window(self):
        """Resize window based on whether agent inspector is needed"""
        if self.agent:
            self.W = self.game_W + self.inspector_W
        else:
            self.W = self.game_W
        
        if not self.is_fullscreen:
            self.screen = pygame.display.set_mode((self.W, self.H))

    def reset_game(self):
        """Reset the game environment and agent"""
        self.env.reset()
        self.assets["player"] = load_img(pick_player(), self.ASSET_SIZE)

        if self.agent:
            self.agent.reset(self.env)

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

        shoot_keymap = {
            pygame.K_i: 4,
            pygame.K_k: 5,
            pygame.K_j: 6,
            pygame.K_l: 7
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

        elif event.key in shoot_keymap:
            action = shoot_keymap[event.key]
            reward, done = self.env.step(action)

            if getattr(self.env, "last_shot_path", None):
                self.animate_arrow(self.env.last_shot_path, action - 4)
                self.env.last_shot_path = []

            print(f"Action: SHOOT {['UP','DOWN','LEFT','RIGHT'][action-4]}, Reward: {reward}")

            self.render()

            if done:
                self.show_game_over(reward)

    def handle_agent_step(self):
        """Execute one step for AI agent"""
        if not self.agent or self.env.done:
            return False

        # Get action with decision information
        action = self.agent.decide_action()

        if action is not None:
            # Log decision to inspector
            if self.agent_inspector:
                action_probs = {}
                action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SHOOT_UP', 'SHOOT_DOWN', 'SHOOT_LEFT', 'SHOOT_RIGHT']
                chosen_reason = f"Chose {action_names[action] if 0 <= action < len(action_names) else 'UNKNOWN'}"
                self.agent_inspector.add_decision(action, action_probs, chosen_reason)

            reward, done = self.env.step(action)

            if getattr(self.env, "last_shot_path", None):
                self.animate_arrow(self.env.last_shot_path, action - 4)
                self.env.last_shot_path = []

            self.render()

            if done:
                self.show_game_over(reward)
                return False

            return True
        else:
            print("Agent has no more actions available.")
            return False

    def animate_arrow(self, path, dir_index):
        """Animate arrow sprite over tile-by-tile path."""
        if not path:
            return
        arrow_img = self.assets.get("arrow")
        if not arrow_img:
            return

        angle_map = {0: 270, 1: 90, 2: 180, 3: 0}
        rot = angle_map.get(dir_index, 0)
        aimg = pygame.transform.rotate(arrow_img, rot)

        delay_ms = 100
        for pos in path:
            self.render()
            
            # Calculate position considering fullscreen scaling
            if self.is_fullscreen:
                scale = min(self.screen_w / self.W, self.screen_h / self.H)
                offset_x = (self.screen_w - self.W * scale) // 2
                offset_y = (self.screen_h - self.H * scale) // 2
                x = offset_x + int((pos[1] * self.CELL + 2) * scale)
                y = offset_y + int((pos[0] * self.CELL + 42) * scale)
                scaled_arrow = pygame.transform.scale(aimg, 
                                                      (int(self.CELL * scale), int(self.CELL * scale)))
                self.screen.blit(scaled_arrow, (x, y))
            else:
                x = pos[1] * self.CELL + 2
                y = pos[0] * self.CELL + 42
                self.screen.blit(aimg, (x, y))
            
            pygame.display.flip()
            pygame.time.delay(delay_ms)

    def render(self):
        """Render the Wumpus World window AND the Agent Inspector panel"""
        if self.is_fullscreen:
            self._render_fullscreen()
        else:
            self._render_windowed()

    def _render_windowed(self):
        """Render in windowed mode"""
        self.screen.fill((20, 20, 30))
        
        game_surface = self.screen.subsurface((0, 0, self.game_W, self.game_H))
        self.env.render(game_surface, self.CELL, self.assets)

        font = pygame.font.SysFont(None, 20)
        mode_text = font.render(f"Mode: {self.mode.upper()}", True, (200, 200, 200))
        self.screen.blit(mode_text, (self.game_W - 220, 8))

        arrow_text = font.render(f"Arrows: {self.env.arrows_remaining}", True, (200, 200, 200))
        self.screen.blit(arrow_text, (self.game_W - 100, 8))

        if self.agent_inspector and self.agent:
            inspector_rect = pygame.Rect(self.game_W, 0, self.inspector_W, self.H)
            inspector_surface = self.screen.subsurface(inspector_rect)
            self.agent_inspector.draw(inspector_surface)

        pygame.display.flip()

    def _render_fullscreen(self):
        """Render in fullscreen mode with scaling"""
        self.screen.fill((0, 0, 0))
        
        # Calculate scaling to fit screen while maintaining aspect ratio
        scale = min(self.screen_w / self.W, self.screen_h / self.H)
        scaled_w = int(self.W * scale)
        scaled_h = int(self.H * scale)
        
        # Center the scaled content
        offset_x = (self.screen_w - scaled_w) // 2
        offset_y = (self.screen_h - scaled_h) // 2
        
        # Create a temporary surface at original size
        temp_surface = pygame.Surface((self.W, self.H))
        temp_surface.fill((20, 20, 30))
        
        # Render game to temp surface
        game_surface = temp_surface.subsurface((0, 0, self.game_W, self.game_H))
        self.env.render(game_surface, self.CELL, self.assets)
        
        # Render inspector if available
        if self.agent_inspector and self.agent:
            inspector_rect = pygame.Rect(self.game_W, 0, self.inspector_W, self.H)
            inspector_surface = temp_surface.subsurface(inspector_rect)
            self.agent_inspector.draw(inspector_surface)
        
        # Add UI text
        font = pygame.font.SysFont(None, 20)
        mode_text = font.render(f"Mode: {self.mode.upper()}", True, (200, 200, 200))
        temp_surface.blit(mode_text, (self.game_W - 220, 8))
        
        arrow_text = font.render(f"Arrows: {self.env.arrows_remaining}", True, (200, 200, 200))
        temp_surface.blit(arrow_text, (self.game_W - 100, 8))
        
        # Scale using smoothscale for better quality and blit to screen
        scaled_surface = pygame.transform.smoothscale(temp_surface, (scaled_w, scaled_h))
        self.screen.blit(scaled_surface, (offset_x, offset_y))
        
        # Add fullscreen indicator
        fs_font = pygame.font.SysFont(None, 18)
        fs_text = fs_font.render("F11: Exit Fullscreen", True, (150, 150, 150))
        self.screen.blit(fs_text, (10, self.screen_h - 25))
        
        pygame.display.flip()

    def show_game_over(self, reward):
        """Show game over overlay"""
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        
        if self.is_fullscreen:
            # For fullscreen, we need to scale the overlay
            scale = min(self.screen_w / self.W, self.screen_h / self.H)
            scaled_w = int(self.W * scale)
            scaled_h = int(self.H * scale)
            offset_x = (self.screen_w - scaled_w) // 2
            offset_y = (self.screen_h - scaled_h) // 2
            
            scaled_overlay = pygame.transform.smoothscale(overlay, (scaled_w, scaled_h))
            self.screen.blit(scaled_overlay, (offset_x, offset_y))
            
            font = pygame.font.SysFont(None, int(36 * scale))
            msg = "You won!" if reward > 0 else "You died!"
            text = font.render(msg + "  (R to restart)", True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.screen_w // 2, self.screen_h // 2))
            self.screen.blit(text, text_rect)
        else:
            self.screen.blit(overlay, (0, 0))
            font = pygame.font.SysFont(None, 36)
            msg = "You won!" if reward > 0 else "You died!"
            text = font.render(msg + "  (R to restart)", True, (255, 255, 255))
            self.screen.blit(text, text.get_rect(center=(self.W // 2, self.H // 2)))
        
        pygame.display.flip()

    def print_controls(self):
        print("\n" + "="*60)
        print(f"WUMPUS WORLD - {self.mode.upper()} MODE")
        print("="*60)
        print("\nControls:")
        if self.mode == "manual":
            print("  Arrow Keys / WASD: Move agent")
            print("  I/J/K/L: Shoot directions")
        else:
            print("  SPACE: Auto-play toggle")
            print("  ENTER: Single AI step")
        print("  R: Reset environment")
        print("  F11: Toggle fullscreen")
        print("  M: Switch to manual mode")
        print("  B: Switch to BFS agent mode")
        print("  Q/ESC: Quit\n")

    def switch_mode(self, new_mode):
        if new_mode == self.mode:
            return
        self.mode = new_mode

        if self.mode == "manual":
            self.agent = None
            self.agent_inspector = None
        else:
            self.agent = WumpusBFS(self.env)
            self.agent_inspector = AgentInspectorWindow(
                agent=self.agent,
                grid_width=self.env.w,
                grid_height=self.env.h
            )

        self._resize_window()
        pygame.display.set_caption(f"Wumpus World - Mode: {self.mode.upper()}")
        
        self.reset_game()
        self.print_controls()

    def handle_mouse_click(self, pos):
        """Handle mouse clicks for UI interactions"""
        if self.agent_inspector and self.agent:
            # Adjust for fullscreen
            if self.is_fullscreen:
                scale = min(self.screen_w / self.W, self.screen_h / self.H)
                offset_x = (self.screen_w - self.W * scale) // 2
                offset_y = (self.screen_h - self.H * scale) // 2
                
                # Convert screen coordinates to virtual coordinates
                virt_x = int((pos[0] - offset_x) / scale)
                virt_y = int((pos[1] - offset_y) / scale)
                pos = (virt_x, virt_y)
            
            # Check if click is in inspector area
            if self.game_W <= pos[0] <= self.W:
                return self.agent_inspector.handle_click(pos, self.game_W)
        
        return False

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

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        if self.is_fullscreen:
                            self.toggle_fullscreen()
                        else:
                            running = False
                    elif event.key == pygame.K_F11:
                        self.toggle_fullscreen()
                        self.render()
                    elif event.key == pygame.K_r:
                        self.reset_game()
                        self.render()
                    elif event.key == pygame.K_m:
                        self.switch_mode("manual")
                    elif event.key == pygame.K_b:
                        self.switch_mode("bfs")
                    elif self.mode == "manual":
                        self.handle_manual_input(event)
                    else:
                        if event.key == pygame.K_SPACE:
                            self.auto_play = not self.auto_play
                            print("Auto-play:", self.auto_play)
                        elif event.key == pygame.K_RETURN:
                            self.handle_agent_step()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        if self.handle_mouse_click(event.pos):
                            self.render()

            # Auto-play AI mode
            if self.mode != "manual" and self.auto_play and not self.env.done:
                if (current_time - self.last_step_time) > self.step_delay:
                    if not self.handle_agent_step():
                        self.auto_play = False
                    self.last_step_time = current_time

            self.clock.tick(60)

        pygame.quit()
        sys.exit()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="manual")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--pits", type=int, default=6)
    parser.add_argument("--wumpus", type=int, default=3)
    parser.add_argument("--arrows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--orientation", type=str, default="down")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    args = parser.parse_args()

    game = WumpusGame(
        mode=args.mode,
        w=args.width,
        h=args.height,
        n_pits=args.pits,
        n_wumpus=args.wumpus,
        n_arrows=args.arrows,
        seed=args.seed,
        wumpus_orientation=args.orientation
    )
    
    if args.fullscreen:
        game.toggle_fullscreen()

    game.run()


if __name__ == "__main__":
    main()