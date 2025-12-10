import random
import pygame, random, sys
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2


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

    env = WumpusEnv(w=8, h=8, n_pits=6, n_wumpus=3, seed=123, wumpus_orientation="down")
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


                #this is the action mapping for player can be replaced with an AI agent in mazens case it will be genetic algorithm
                keymap = {pygame.K_UP: 0, pygame.K_w: 0, pygame.K_DOWN: 1, pygame.K_s: 1, pygame.K_LEFT: 2, pygame.K_a: 2, pygame.K_RIGHT: 3, pygame.K_d: 3}
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
        clock.tick(30)


if __name__ == "__main__":
    main()
