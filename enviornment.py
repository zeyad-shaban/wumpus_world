import random
import pygame, random, sys
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2

# Path to arrow image (must exist)
ARROW_IMG_PATH = "images/arrow.png"


def pick_player():
    return random.choice(glob("images/player_images/*"))


wumpus_img = "images/wumpus.png"
pit_img = "images/pit.png"
gold_img = "images/gold.png"
exit_img = "images/exit.png"


class WumpusEnv:
    def __init__(self, w=8, h=8, n_pits=6, n_wumpus=1, n_arrows=1, seed=None, wumpus_orientation="down"):
        self.w, self.h, self.n_pits, self.n_wumpus, self.n_arrows = w, h, n_pits, n_wumpus, n_arrows
        self.rng = random.Random(seed)
        self.wumpus_orientation = wumpus_orientation
        self.reset()

    def reset(self):
        cells = [(r, c) for r in range(self.h) for c in range(self.w)]
        self.rng.shuffle(cells)
        it = iter(cells)
        self.agent = next(it)
        self.gold = next(it)
        # place multiple wumpuses (each is (pos, facing))
        self.wumpuses = []
        for _ in range(self.n_wumpus):
            pos = next(it)
            if self.wumpus_orientation == "random":
                facing = self.rng.choice(["up", "down", "left", "right"])
            else:
                facing = self.wumpus_orientation
            self.wumpuses.append((pos, facing))
        self.pits = {next(it) for _ in range(self.n_pits)}
        self.exit = next(it)
        self.have_gold = False
        self.done = False
        self.total = 0

        # Arrow state: track remaining arrows
        self.arrows_remaining = self.n_arrows
        # path of most recent shot for main to animate (list of positions)
        self.last_shot_path = []

        return self.get_state()

    def inb(self, p):
        return 0 <= p[0] < self.h and 0 <= p[1] < self.w

    def step(self, a):
        """
        Actions:
          0: UP
          1: DOWN
          2: LEFT
          3: RIGHT
          4: SHOOT UP
          5: SHOOT DOWN
          6: SHOOT LEFT
          7: SHOOT RIGHT
        """
        if self.done:
            return 0, True

        # movement
        if 0 <= a <= 3:
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

            # check if agent stepped on any wumpus - always fatal
            for wpos, wfacing in self.wumpuses:
                if self.agent == wpos:
                    self.total += -100
                    self.done = True
                    return -100, True

            if self.agent == self.exit and self.have_gold:
                self.total += 200
                self.done = True
                return 200, True
            self.total += -1
            return -1, False

        # shooting actions
        elif 4 <= a <= 7:
            dir_index = a - 4  # 0=up,1=down,2=left,3=right
            reward, done, path = self._shoot(dir_index)
            # set last_shot_path so main can animate
            self.last_shot_path = path
            return reward, done

        else:
            # invalid action
            return 0, False

    def _shoot(self, dir_index):
        """
        Shoot arrow along line from the agent in direction dir_index.
        dir_index: 0=up,1=down,2=left,3=right
        Returns: (reward, done, path)
        Decrements arrows_remaining.
        """
        if self.arrows_remaining <= 0:
            # no arrow left — small penalty
            self.total += -1
            return -1, False, []

        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[dir_index]
        r, c = self.agent
        path = []
        hit_index = None
        hit_pos = None

        # consume arrow immediately
        self.arrows_remaining -= 1

        # walk tile-by-tile outwards
        nr, nc = r + dr, c + dc
        while self.inb((nr, nc)):
            path.append((nr, nc))
            # if a wumpus exists in this tile, kill it
            for i, (wpos, wfacing) in enumerate(self.wumpuses):
                if wpos == (nr, nc):
                    hit_index = i
                    hit_pos = (nr, nc)
                    break
            if hit_index is not None:
                break
            nr += dr
            nc += dc

        if hit_index is not None:
            # kill wumpus, reward
            self.wumpuses.pop(hit_index)
            self.total += 150
            # not finishing the episode necessarily, so done=False
            return 150, False, path
        else:
            # missed — small penalty
            self.total += -10
            return -10, False, path

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
            "arrows_remaining": self.arrows_remaining,
        }

    def get_percepts(self):
        p = self.agent
        percepts = {"breeze": False, "stench": False, "glitter": False}

        # Check for breeze (pit adjacent) and stench (wumpus adjacent)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (p[0] + dr, p[1] + dc)
            if neighbor in self.pits:
                percepts["breeze"] = True
            if neighbor in [wp for wp, _ in self.wumpuses]:
                percepts["stench"] = True
        if self.gold and self.agent == self.gold:
            percepts["glitter"] = True
        return percepts

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
        s = f"Total:{self.total}  Have gold:{self.have_gold}  Wumpuses:{len(self.wumpuses)}  Arrows:{self.arrows_remaining}"
        surf.blit(font.render(s, True, (220, 220, 220)), (8, 8))


def load_img(path, size):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or BGRA
    if im is None:
        raise ValueError(f"cv2.imread returned None for {path}")
    h0, w0 = im.shape[:2]
    target_w, target_h = size
    scale = min(target_w / w0, target_h / h0)
    new_w, new_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
    im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

    im_rgba = cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGBA)

    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)  # h x w x 4

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y: y + new_h, x: x + new_w] = im_rgba

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
        "arrow": load_img(ARROW_IMG_PATH, ASSET_SIZE),
    }

    env = WumpusEnv(w=8, h=8, n_pits=6, n_wumpus=3, n_arrows=1, seed=123, wumpus_orientation="down")
    W, H = env.w * CELL, env.h * CELL + 40
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    env.render(screen, CELL, assets)
    pygame.display.flip()

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    env.reset()
                    assets["player"] = load_img(pick_player(), ASSET_SIZE)
                    env.render(screen, CELL, assets)
                    pygame.display.flip()
                    continue
                if env.done:
                    continue

                keymap = {pygame.K_UP: 0, pygame.K_w: 0, pygame.K_DOWN: 1, pygame.K_s: 1, pygame.K_LEFT: 2, pygame.K_a: 2, pygame.K_RIGHT: 3, pygame.K_d: 3}
                shoot_map = {pygame.K_i: 4, pygame.K_k: 5, pygame.K_j: 6, pygame.K_l: 7}

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
                        text = font.render(msg + "  (R to restart)", True, (255, 255, 255))
                        screen.blit(text, text.get_rect(center=(W // 2, H // 2)))
                        pygame.display.flip()

                elif e.key in shoot_map:
                    a = shoot_map[e.key]
                    reward, done = env.step(a)

                    # animate arrow if path available
                    path = getattr(env, "last_shot_path", [])
                    if path:
                        # Rotate arrow based on direction (arrow image assumed pointing RIGHT)
                        dir_idx = a - 4  # 0=up,1=down,2=left,3=right
                        angle_map = {0: 270, 1: 90, 2: 180, 3: 0}
                        aimg = pygame.transform.rotate(assets.get("arrow"), angle_map.get(dir_idx, 0))
                        delay_ms = 100
                        for pos in path:
                            env.render(screen, CELL, assets)
                            x = pos[1] * CELL + 2
                            y = pos[0] * CELL + 42
                            screen.blit(aimg, (x, y))
                            pygame.display.flip()
                            pygame.time.delay(delay_ms)
                        env.last_shot_path = []

                    env.render(screen, CELL, assets)
                    pygame.display.flip()

                    print(f"Shot {['UP','DOWN','LEFT','RIGHT'][a-4]} Reward: {reward} Arrows left: {env.arrows_remaining} Total: {env.total}")

                    if done:
                        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
                        overlay.fill((0, 0, 0, 160))
                        screen.blit(overlay, (0, 0))
                        font = pygame.font.SysFont(None, 36)
                        msg = "You won!" if reward > 0 else "You died!"
                        text = font.render(msg + "  (R to restart)", True, (255, 255, 255))
                        screen.blit(text, text.get_rect(center=(W // 2, H // 2)))
                        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
