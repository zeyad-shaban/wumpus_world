import random
import pygame, random, sys
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import time


def pick_player():
    return random.choice(glob("images/player_images/*"))


wumpus_img = "images/wumpus.png"
pit_img = "images/pit.png"
gold_img = "images/gold.png"
exit_img = "images/exit.png"


class WumpusEnv:
    def __init__(self, w=8, h=8, n_pits=6, n_wumpus=1, seed=None, wumpus_orientation="down"):
        self.w, self.h, self.n_pits, self.n_wumpus = w, h, n_pits, n_wumpus
        self.rng = random.Random(seed)
        self.wumpus_orientation = wumpus_orientation
        
        # Store initial map configuration
        self.initial_agent = None
        self.initial_gold = None
        self.initial_wumpuses = []
        self.initial_pits = set()
        self.initial_exit = None
        
        self.generate_new_map()

    def generate_new_map(self):
        """Generate a completely new map layout"""
        cells = [(r, c) for r in range(self.h) for c in range(self.w)]
        self.rng.shuffle(cells)
        it = iter(cells)
        
        # Generate new positions
        self.initial_agent = next(it)
        self.initial_gold = next(it)
        
        # place multiple wumpuses (each is (pos, facing))
        self.initial_wumpuses = []
        for _ in range(self.n_wumpus):
            pos = next(it)
            if self.wumpus_orientation == "random":
                facing = self.rng.choice(["up", "down", "left", "right"])
            else:
                facing = self.wumpus_orientation
            self.initial_wumpuses.append((pos, facing))
        
        self.initial_pits = {next(it) for _ in range(self.n_pits)}
        self.initial_exit = next(it)
        
        # Now reset to use this map
        self.reset()
        return self.get_state()

    def reset(self):
        """Reset agent to starting position on the SAME map"""
        self.agent = self.initial_agent
        self.gold = self.initial_gold
        self.wumpuses = list(self.initial_wumpuses)  # copy the list
        self.pits = set(self.initial_pits)  # copy the set
        self.exit = self.initial_exit
        self.have_gold = False
        self.done = False
        self.total = 0
        return self.get_state()

    def inb(self, p):
        return 0 <= p[0] < self.h and 0 <= p[1] < self.w

    def step(self, a):
        if self.done:
            return 0, True
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[a]
        old = self.agent
        nr, nc = old[0] + dr, old[1] + dc
        if not self.inb((nr, nc)):
            self.total += -1
            return -1, False
        self.agent = (nr, nc)
        if self.agent == self.gold:
            self.have_gold = True
            self.gold = None
            self.total += 100
            return 100, False
        if self.agent in self.pits:
            self.total += -100
            self.done = True
            return -100, True

        # check against each wumpus
        for i, (wpos, wfacing) in enumerate(self.wumpuses):
            if self.agent == wpos:
                mv = (self.agent[0] - old[0], self.agent[1] - old[1])
                vulnerable_move = (1, 0) if wfacing == "down" else ((-1, 0) if wfacing == "up" else ((0, 1) if wfacing == "right" else (0, -1)))
                if mv == vulnerable_move:
                    # kill this wumpus
                    self.wumpuses.pop(i)
                    self.total += 150
                    return 150, False
                else:
                    self.total += -100
                    self.done = True
                    return -100, True

        if self.agent == self.exit and self.have_gold:
            self.total += 200
            self.done = True
            return 200, True
        self.total += -1
        return -1, False

    def get_state(self):
        # return positions list and a dict of facings for easy inspection
        w_positions = [pos for pos, face in self.wumpuses]
        w_facings = {pos: face for pos, face in self.wumpuses}
        return {
            "agent": self.agent,
            "gold": self.gold,
            "wumpus": w_positions,
            "wumpus_facing": w_facings,
            "pits": set(self.pits),
            "exit": self.exit,
            "have_gold": self.have_gold,
            "done": self.done,
            "total": self.total,
        }

    def render(self, surf, cell=64, assets=None):
        surf.fill((20, 20, 30))
        font = pygame.font.SysFont(None, 22)
        for r in range(self.h):
            for c in range(self.w):
                rect = pygame.Rect(c * cell, r * cell + 40, cell - 1, cell - 1)
                pygame.draw.rect(surf, (40, 40, 60), rect)

        def blit_img(pos, img):
            if img:
                surf.blit(img, (pos[1] * cell + 2, pos[0] * cell + 42))
            else:
                x = pos[1] * cell
                y = pos[0] * cell + 40
                pygame.draw.rect(surf, (180, 180, 180), (x + 4, y + 4, cell - 8, cell - 8))

        if assets is None:
            assets = {}
        for p in self.pits:
            blit_img(p, assets.get("pit"))
        if self.gold:
            blit_img(self.gold, assets.get("gold"))
        # draw each wumpus with rotation according to its facing
        w_img = assets.get("wumpus")
        angle_map = {"down": 0, "left": 90, "up": 180, "right": 270}
        for wpos, wfacing in list(self.wumpuses):
            if w_img:
                ang = angle_map.get(wfacing, 0)
                wrot = pygame.transform.rotate(w_img, -ang)
                blit_img(wpos, wrot)
            else:
                blit_img(wpos, None)
        blit_img(self.exit, assets.get("exit"))
        blit_img(self.agent, assets.get("player"))
        s = f"Total:{self.total}  Have gold:{self.have_gold}  Wumpuses:{len(self.wumpuses)}"
        surf.blit(font.render(s, True, (220, 220, 220)), (8, 8))


# ============================================================================
# GENETIC ALGORITHM CLASS - Following Mazen's coding style
# ============================================================================

class GeneticAlgorithm:
    
    def __init__(self, max_moves=50):
        self.max_moves = max_moves
        self.moves = "UDLR"  # possible moves
        
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


def load_img(path, size):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or BGRA
    if im is None:
        raise ValueError("cv2.imread returned None")
    h0, w0 = im.shape[:2]
    target_w, target_h = size
    scale = min(target_w / w0, target_h / h0)
    new_w, new_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
    im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

    im_rgba = cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGBA)

    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)  # h x w x 4

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = im_rgba

    surf = pygame.image.frombuffer(canvas.tobytes(), (target_w, target_h), "RGBA")
    return surf


def main():
    pygame.init()
    CELL = 64
    ASSET_SIZE = (CELL - 8, CELL - 8)

    assets = {
        "wumpus": load_img(wumpus_img, ASSET_SIZE),
        "player": load_img(pick_player(), ASSET_SIZE),
        "pit": load_img(pit_img, ASSET_SIZE),
        "gold": load_img(gold_img, ASSET_SIZE),
        "exit": load_img(exit_img, ASSET_SIZE),
    }

    # Ask user for mode
    print("=" * 50)
    print("WUMPUS WORLD")
    print("=" * 50)
    print("Choose mode:")
    print("  1. Manual control (arrow keys)")
    print("  2. Algorithm control (watch AI play)")
    mode = input("Enter 1 or 2: ").strip()
    
    if mode == "2":
        manual_mode = False
        print("\nAlgorithm mode selected!")
        print("Generating random chromosome...")
    else:
        manual_mode = True
        print("\nManual mode selected!")

    env = WumpusEnv(w=8, h=8, n_pits=6, n_wumpus=3, seed=123, wumpus_orientation="down")
    W, H = env.w * CELL, env.h * CELL + 40
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    # Create genetic algorithm instance
    ga = GeneticAlgorithm(max_moves=50)
    
    # Generate one chromosome for testing
    current_chromosome = ga.random_chromosome()
    current_actions = []
    action_index = 0
    
    if not manual_mode:
        current_actions = ga.chromosome_to_actions(current_chromosome)
        print(f"Chromosome: {current_chromosome[:30]}...")  # print first 30 moves
        print(f"Total moves in chromosome: {len(current_chromosome)}")
        print("\nControls:")
        print("  SPACE - Execute next move")
        print("  A - Auto-play (watch full sequence)")
        print("  R - Reset")
        print("  N - New map + new chromosome")

    env.render(screen, CELL, assets)
    pygame.display.flip()

    if manual_mode:
        print("\nManual Controls:")
        print("  Arrow keys / WASD - Move")
        print("  R - Reset (same map)")
        print("  N - New map")
    
    auto_play = False  # for algorithm mode - auto execute moves
    
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if e.type == pygame.KEYDOWN:
                # R key - reset to starting position on SAME map
                if e.key == pygame.K_r:
                    env.reset()
                    action_index = 0  # reset action counter
                    auto_play = False
                    env.render(screen, CELL, assets)
                    pygame.display.flip()
                    print("Map reset to starting position")
                    continue
                
                # N key - generate completely NEW map
                if e.key == pygame.K_n:
                    env.generate_new_map()
                    assets["player"] = load_img(pick_player(), ASSET_SIZE)
                    action_index = 0
                    auto_play = False
                    
                    # Generate new chromosome for algorithm mode
                    if not manual_mode:
                        current_chromosome = ga.random_chromosome()
                        current_actions = ga.chromosome_to_actions(current_chromosome)
                        print(f"\nNew chromosome: {current_chromosome[:30]}...")
                        print(f"Total moves: {len(current_chromosome)}")
                    
                    env.render(screen, CELL, assets)
                    pygame.display.flip()
                    print("New map generated!")
                    continue
                
                # ALGORITHM MODE CONTROLS
                if not manual_mode:
                    # SPACE - execute next move from chromosome
                    if e.key == pygame.K_SPACE:
                        if not env.done and action_index < len(current_actions):
                            action = current_actions[action_index]
                            reward, done = env.step(action)
                            action_index += 1
                            
                            move_name = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action]
                            print(f"Move {action_index}: {move_name} | Reward: {reward} | Total: {env.total}")
                            
                            env.render(screen, CELL, assets)
                            pygame.display.flip()
                            
                            if done:
                                result = "WON!" if reward > 0 else "DIED"
                                print(f"\n{result} Final score: {env.total}")
                        else:
                            print("Chromosome finished or game over!")
                    
                    # A - auto-play mode
                    if e.key == pygame.K_a:
                        auto_play = not auto_play
                        if auto_play:
                            print("Auto-play enabled! Watching AI...")
                        else:
                            print("Auto-play disabled")
                
                # MANUAL MODE CONTROLS
                else:
                    if env.done:
                        continue

                    # Movement controls
                    keymap = {
                        pygame.K_UP: 0, pygame.K_w: 0, 
                        pygame.K_DOWN: 1, pygame.K_s: 1, 
                        pygame.K_LEFT: 2, pygame.K_a: 2, 
                        pygame.K_RIGHT: 3, pygame.K_d: 3
                    }
                    if e.key in keymap:
                        a = keymap[e.key]
                        reward, done = env.step(a)

                        env.render(screen, CELL, assets)
                        pygame.display.flip()

                        if done:
                            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
                            overlay.fill((0, 0, 0, 160))
                            screen.blit(overlay, (0, 0))
                            font = pygame.font.SysFont(None, 36)
                            msg = "You won!" if reward > 0 else "You died!"
                            text = font.render(msg + "  (R to restart / N for new map)", True, (255, 255, 255))
                            screen.blit(text, text.get_rect(center=(W // 2, H // 2)))
                            pygame.display.flip()
        
        # Auto-play logic for algorithm mode
        if not manual_mode and auto_play and not env.done and action_index < len(current_actions):
            action = current_actions[action_index]
            reward, done = env.step(action)
            action_index += 1
            
            move_name = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action]
            print(f"Move {action_index}: {move_name} | Reward: {reward} | Total: {env.total}")
            
            env.render(screen, CELL, assets)
            pygame.display.flip()
            
            if done:
                result = "WON!" if reward > 0 else "DIED"
                print(f"\n{result} Final score: {env.total}")
                auto_play = False
            
            # Small delay so you can see the moves
            pygame.time.delay(200)  # 200ms between moves
        
        clock.tick(30)


if __name__ == "__main__":
    main()