import random
import pygame, random, sys
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
from genetic_algo import GeneticcAlgorithm


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
    print("  3. Watch Evolution (see GA evolve & play best path)")
    mode = input("Enter 1, 2, or 3: ").strip()
    
    if mode == "2":
        manual_mode = False
        evolution_mode = False
        print("\nAlgorithm mode selected!")
        print("Generating random chromosome...")
    elif mode == "3":
        manual_mode = False
        evolution_mode = True
        print("\nEvolution mode selected!")
        print("Watch the genetic algorithm evolve and find the best path!")
    else:
        manual_mode = True
        evolution_mode = False
        print("\nManual mode selected!")

    env = WumpusEnv(w=8, h=8, n_pits=6, n_wumpus=3, seed=123, wumpus_orientation="down")
    W, H = env.w * CELL, env.h * CELL + 40
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    # Variables for evolution mode
    evolution_phase = "idle"  # "idle", "evolving", "showing_best"
    ga = None
    current_chromosome = None
    current_actions = []
    action_index = 0
    evolution_speed = 10  # generations per frame during fast evolution
    
    if not manual_mode:
        # Create genetic algorithm instance
        ga = GeneticcAlgorithm(env, max_moves=50, population_size=50, generations=100, mutation_rate=0.1)
        
        if evolution_mode:
            # Start evolution
            ga.start()
            evolution_phase = "evolving"
            print("\nEvolution started!")
            print("Controls:")
            print("  SPACE - Pause/Resume evolution")
            print("  F - Toggle fast mode (skip visualization)")
            print("  R - Restart evolution")
            print("  N - New map + restart evolution")
            print("\nWatch as the algorithm finds the best path...")
        else:
            # Generate one chromosome for testing (mode 2)
            current_chromosome = ga.random_chromosome()
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
    fast_mode = False  # for evolution mode - skip visualization during evolution
    paused = False  # for evolution mode - pause evolution
    
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if e.type == pygame.KEYDOWN:
                # R key - reset/restart
                if e.key == pygame.K_r:
                    env.reset()
                    action_index = 0
                    auto_play = False
                    
                    if evolution_mode:
                        # Restart evolution
                        ga.start()
                        evolution_phase = "evolving"
                        paused = False
                        print("\nEvolution restarted!")
                    
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
                    
                    if evolution_mode:
                        # Restart evolution with new map
                        ga.start()
                        evolution_phase = "evolving"
                        paused = False
                        print("\nNew map generated! Evolution restarted!")
                    elif not manual_mode:
                        # Generate new chromosome for algorithm mode (mode 2)
                        current_chromosome = ga.random_chromosome()
                        current_actions = ga.chromosome_to_actions(current_chromosome)
                        print(f"\nNew chromosome: {current_chromosome[:30]}...")
                        print(f"Total moves: {len(current_chromosome)}")
                    
                    env.render(screen, CELL, assets)
                    pygame.display.flip()
                    if not evolution_mode:
                        print("New map generated!")
                    continue
                
                # EVOLUTION MODE CONTROLS
                if evolution_mode:
                    # SPACE - Pause/Resume evolution OR step through best path
                    if e.key == pygame.K_SPACE:
                        if evolution_phase == "evolving":
                            paused = not paused
                            print("Evolution PAUSED" if paused else "Evolution RESUMED")
                        elif evolution_phase == "showing_best":
                            # Step through best path manually
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
                    
                    # F - Toggle fast mode
                    if e.key == pygame.K_f:
                        fast_mode = not fast_mode
                        print(f"Fast mode: {'ON' if fast_mode else 'OFF'}")
                    
                    # A - Auto-play best path (when evolution complete)
                    if e.key == pygame.K_a and evolution_phase == "showing_best":
                        auto_play = not auto_play
                        if auto_play:
                            print("Auto-play enabled! Watching best solution...")
                        else:
                            print("Auto-play disabled")
                
                # ALGORITHM MODE CONTROLS (mode 2)
                elif not manual_mode:
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
        
        # EVOLUTION MODE - Run evolution
        if evolution_mode and evolution_phase == "evolving" and not paused:
            # Run one or more generations
            gens_to_run = evolution_speed if fast_mode else 1
            
            for _ in range(gens_to_run):
                gen, best_fit, avg_fit, is_complete = ga.evolve_one_generation()
                
                if is_complete:
                    break
            
            # Print progress
            print(f"\rGeneration {gen}/{ga.generations} | Best Fitness: {best_fit} | Avg: {avg_fit:.1f}", end="")
            
            # Update display with evolution info
            env.render(screen, CELL, assets)
            
            # Draw evolution overlay
            font = pygame.font.SysFont(None, 28)
            overlay_text = f"EVOLVING... Gen: {gen}/{ga.generations} | Best: {best_fit}"
            text_surf = font.render(overlay_text, True, (255, 255, 0))
            screen.blit(text_surf, (10, H - 30))
            
            pygame.display.flip()
            
            # Check if evolution is complete
            if is_complete:
                evolution_phase = "showing_best"
                print(f"\n\n{'='*50}")
                print("EVOLUTION COMPLETE!")
                print(f"Best fitness found: {best_fit}")
                print(f"Best chromosome: {ga.best_chromosome[:30]}...")
                print(f"{'='*50}")
                print("\nNow showing the best path found!")
                print("Controls:")
                print("  SPACE - Step through moves")
                print("  A - Auto-play best path")
                print("  R - Restart evolution")
                print("  N - New map + restart")
                
                # Setup for showing best path
                env.reset()
                current_actions = ga.get_best_actions()
                action_index = 0
                auto_play = False
                
                env.render(screen, CELL, assets)
                pygame.display.flip()
            
            # Small delay to see evolution (not in fast mode)
            if not fast_mode:
                pygame.time.delay(50)
        
        # EVOLUTION MODE - Show best path (auto-play)
        if evolution_mode and evolution_phase == "showing_best" and auto_play:
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
                    auto_play = False
                    
                    # Show final overlay
                    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
                    overlay.fill((0, 0, 0, 160))
                    screen.blit(overlay, (0, 0))
                    font = pygame.font.SysFont(None, 36)
                    msg = f"Best Solution: {result} Score: {env.total}"
                    text = font.render(msg, True, (255, 255, 255))
                    screen.blit(text, text.get_rect(center=(W // 2, H // 2 - 20)))
                    hint = font.render("(R to restart / N for new map)", True, (200, 200, 200))
                    screen.blit(hint, hint.get_rect(center=(W // 2, H // 2 + 20)))
                    pygame.display.flip()
                
                pygame.time.delay(300)  # Slower for best path visualization
        
        # ALGORITHM MODE (mode 2) - Auto-play logic
        if not manual_mode and not evolution_mode and auto_play and not env.done and action_index < len(current_actions):
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
            
            pygame.time.delay(200)
        
        clock.tick(30)

if __name__ == "__main__":
    main()