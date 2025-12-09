"""
A simplified, more readable, less human-like BFS agent
with risk filtering for Wumpus World.

Behavioral properties:
- Deterministic decision structure
- Minimal heuristic reasoning
- Consistent rule-based risk evaluation
- No psychologically human-like "intuition"
"""

from collections import deque

class WumpusBFS:
    def __init__(self, env, risk_threshold=0.10):
        self.env = env
        self.risk_threshold = risk_threshold
        self.reset(env)

    # -----------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------
    def reset(self, env):
        self.env = env
        s = env.get_state()

        self.w = env.w
        self.h = env.h

        # Knowledge sets
        self.visited = set()
        self.safe = set()
        self.possible_pit_count = {}      # {tile: int}
        self.possible_wumpus_count = {}   # {tile: int}
        self.confirmed_wumpus = set()
        self.confirmed_no_wumpus = set()

        # Update KB at initial tile
        self._update_knowledge()

    # -----------------------------------------------------------
    # Basic utilities
    # -----------------------------------------------------------
    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.h and 0 <= c < self.w

    def neighbors(self, pos):
        r, c = pos
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        for dr, dc in dirs:
            np = (r+dr, c+dc)
            if self.in_bounds(np):
                yield np

    # -----------------------------------------------------------
    # Percept-driven KB update
    # -----------------------------------------------------------
    def _update_knowledge(self):
        state = self.env.get_state()
        agent = state["agent"]
        percepts = self.env.get_percepts()

        # Mark visited / safe
        self.visited.add(agent)
        self.safe.add(agent)

        breeze = percepts.get("breeze", False)
        stench = percepts.get("stench", False)

        # Pit reasoning
        for nbr in self.neighbors(agent):
            if not breeze:
                # Breeze absent → neighbor not a pit
                self.safe.add(nbr)
                self.possible_pit_count.pop(nbr, None)
            else:
                # Breeze present → neighbor may be pit
                if nbr not in self.visited:
                    self.possible_pit_count[nbr] = self.possible_pit_count.get(nbr, 0) + 1

        # Wumpus reasoning
        for nbr in self.neighbors(agent):
            if not stench:
                # No stench → likely no Wumpus
                self.confirmed_no_wumpus.add(nbr)
                self.possible_wumpus_count.pop(nbr, None)
            else:
                if nbr not in self.visited:
                    self.possible_wumpus_count[nbr] = self.possible_wumpus_count.get(nbr, 0) + 1

        # Deduce high-confidence wumpus
        for tile, cnt in list(self.possible_wumpus_count.items()):
            if tile not in self.confirmed_no_wumpus and cnt >= 2:
                self.confirmed_wumpus.add(tile)

    # -----------------------------------------------------------
    # Risk estimation (simple procedural model, not human-like)
    # -----------------------------------------------------------
    def pit_probability(self, tile):
        if tile in self.safe or tile in self.visited:
            return 0.0
        cnt = self.possible_pit_count.get(tile, 0)
        # Linearized risk model for readability
        return min(0.25 * cnt, 0.90)

    def wumpus_probability(self, tile):
        if tile in self.confirmed_no_wumpus:
            return 0.0
        if tile in self.confirmed_wumpus:
            return 0.95
        cnt = self.possible_wumpus_count.get(tile, 0)
        return min(0.20 * cnt, 0.80)

    def death_probability(self, tile):
        return max(self.pit_probability(tile), self.wumpus_probability(tile))

    # -----------------------------------------------------------
    # BFS path computation with risk filtering
    # -----------------------------------------------------------
    def bfs_path(self, start, goal, max_risk):
        queue = deque([start])
        parent = {start: None}

        while queue:
            cur = queue.popleft()
            if cur == goal:
                return self.reconstruct_path(parent, cur)

            for nbr in self.neighbors(cur):
                if nbr in parent:
                    continue
                if self.death_probability(nbr) > max_risk:
                    continue
                parent[nbr] = cur
                queue.append(nbr)

        return None

    def reconstruct_path(self, parent, end):
        path = []
        tile = end
        while tile is not None:
            path.append(tile)
            tile = parent[tile]
        return list(reversed(path))

    # -----------------------------------------------------------
    # Convert movement to action IDs
    # -----------------------------------------------------------
    def movement_action(self, a, b):
        (r1, c1), (r2, c2) = a, b
        if r2 == r1 - 1: return 0  # UP
        if r2 == r1 + 1: return 1  # DOWN
        if c2 == c1 - 1: return 2  # LEFT
        if c2 == c1 + 1: return 3  # RIGHT
        return None

    # -----------------------------------------------------------
    # Shooting logic
    # -----------------------------------------------------------
    def shooting_action(self, wumpus_pos, agent):
        ar = agent
        wr = wumpus_pos

        # same column
        if ar[1] == wr[1]:
            if wr[0] < ar[0]: return 4  # UP
            if wr[0] > ar[0]: return 5  # DOWN

        # same row
        if ar[0] == wr[0]:
            if wr[1] < ar[1]: return 6  # LEFT
            if wr[1] > ar[1]: return 7  # RIGHT

        return None

    # -----------------------------------------------------------
    # Main decision function
    # -----------------------------------------------------------
    def decide_action(self):
        self._update_knowledge()
        state = self.env.get_state()

        agent = state["agent"]
        gold_pos = state["gold"]
        have_gold = state["have_gold"]
        arrows = state["arrows_remaining"]

        # -----------------------------
        # 1. Shooting option (rule-based)
        # -----------------------------
        if arrows > 0 and self.confirmed_wumpus:
            for wpos in self.confirmed_wumpus:
                act = self.shooting_action(wpos, agent)
                if act is not None:
                    return act

        # -----------------------------
        # 2. Goal determination
        # -----------------------------
        if not have_gold and gold_pos is not None:
            goal = gold_pos
        elif have_gold:
            goal = state["exit"]
        else:
            goal = None

        # -----------------------------
        # 3. Try safe BFS plan
        # -----------------------------
        if goal is not None:
            path = self.bfs_path(agent, goal, self.risk_threshold)
            if path:
                return self.movement_action(path[0], path[1])

        # -----------------------------
        # 4. Relax safety requirements (machine-like fallback)
        # -----------------------------
        relax_levels = [self.risk_threshold, 0.30, 0.60, 1.0]
        for thr in relax_levels:
            if goal is not None:
                path = self.bfs_path(agent, goal, thr)
                if path:
                    return self.movement_action(path[0], path[1])

        # -----------------------------
        # 5. Exploration: choose neighbor with minimal risk
        # -----------------------------
        best_tile = None
        best_risk = 999.0

        for nbr in self.neighbors(agent):
            risk = self.death_probability(nbr)
            if risk < best_risk:
                best_risk = risk
                best_tile = nbr

        if best_tile is not None:
            return self.movement_action(agent, best_tile)

        return None
