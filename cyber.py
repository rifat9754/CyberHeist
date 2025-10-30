"""
Cyber Heist — AI vs AI (Autonomous Thief & Guards)
------------------------------------------------------------------------------
This file is the SAME GAME you shared — only rewritten with simple wording,
clean structure, and many small comments so it's easy to read and follow.
No game logic was changed.

What the game does (quick):
- Generates a grid map of rooms with walls, doors, cameras, terminals, and exits.
- Spawns an autonomous Thief at a boundary EXIT (green).
  Thief plan: go to Vault → pick data → run to an Exit.
- Spawns several Guards. They patrol, listen to noise/hacks, share Last Known
  Position (LKP), and chase or blockade using A* pathfinding.
- Movement is step-delayed for clearer visuals. SPACE toggles slow motion.
- Results are appended to runs_log.csv (winner, time, ticks, loot value).

Debug keys (for observing behavior):
[1] Toggle guard LOS lines   [2] Toggle paths
[4] Toggle noise pings       [5] Toggle blockade hints
[SPACE] Slow-mo toggle       [R] Restart
[ESC] Quit
"""

# ------------------------------ Imports --------------------------------------
import math
import random
import sys
import time
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
import pygame


# =============================================================================
#                               LOGGING UTILS
# =============================================================================
def log_result_csv(seed: int, result: str, elapsed_s: float, ticks: int, thief_value: int,
                   path: str = "runs_log.csv"):
    """
    Append one row to a CSV so you can analyze results later.

    seed         -> RNG seed used for reproducible layouts
    result       -> "THIEF" or "GUARDS"
    elapsed_s    -> real time since run started (seconds)
    ticks        -> number of frames advanced
    thief_value  -> how much loot thief carried when the game ended
    path         -> CSV filename (created if missing)
    """
    header = ["timestamp", "seed", "result", "elapsed_s", "ticks", "thief_value"]
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), seed, result, round(elapsed_s, 2), ticks, thief_value]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


# =============================================================================
#                              GLOBAL SETTINGS
# =============================================================================
# Window/Grid
WIDTH, HEIGHT = 1180, 780
TILE = 20
COLS, ROWS = WIDTH // TILE, HEIGHT // TILE
FPS = 60

# Colors
BLACK  = (15, 18, 22)
DARK   = (26, 30, 34)
MID    = (45, 50, 57)
LIGHT  = (200, 220, 240)
RED    = (240, 84, 84)
YELLOW = (250, 224, 120)
ORANGE = (255, 170, 70)
BLUE   = (120, 180, 255)
GREEN  = (140, 220, 140)
PURPLE = (200, 140, 255)
ACCENT = (82, 220, 200)  # thief body

# Thief timings / decisions
THIEF_REPLAN_SECONDS = 0.6
THIEF_DECOY_COOLDOWN = 2.5
THIEF_HACK_COOLDOWN  = 1.2

# Step pacing (visual only)
THIEF_STEP_DELAY        = 0.12
GUARD_STEP_DELAY_PATROL = 0.18
GUARD_STEP_DELAY_CHASE  = 0.12

# Slow-mo
TIME_SCALE_SLOW = 0.5
TIME_SCALE_NORM = 1.0

# Noise model
NOISE_BASE             = 0.2
NOISE_SPRINT           = 1.0
NOISE_HACK             = 1.2
NOISE_DECOY            = 1.5
NOISE_DECAY_PER_SECOND = 0.85

# Guards
GUARD_VIEW_DIST      = 14  # tiles
GUARD_FOV_DEG        = 80
GUARD_MEMORY_SECONDS = 6.0
GUARD_COUNT          = 4

# Suspicion thresholds & gains
SUSP_DECAY_PER_SECOND       = 0.92
SUSP_INCREASE_SIGHT         = 0.25
SUSP_INCREASE_NOISE         = 0.08
SUSP_THRESHOLD_INVESTIGATE  = 0.25
SUSP_THRESHOLD_PATROL_PLUS  = 0.45
SUSP_THRESHOLD_CHASE        = 0.65

# Room scoring weights (for search)
W_DIST  = 0.6
W_VALUE = -1.2
W_EXIT  = -0.4
W_NOISE = -1.0
W_BLIND = -0.3

# Randomness (change this to get new layouts)
SEED = 3
RNG  = random.Random(SEED)


# =============================================================================
#                           MAP / TILE / ROOM TYPES
# =============================================================================
class TileType:
    FLOOR    = 0
    WALL     = 1
    DOOR     = 2
    CAMERA   = 3
    TERMINAL = 4
    EXIT     = 5


@dataclass
class Room:
    id: int
    rect: pygame.Rect
    type: str   # 'regular' | 'security' | 'vault' | 'utility'
    value: int  # helps scoring/search


@dataclass
class NoisePing:
    pos: tuple
    strength: float


# =============================================================================
#                              BASIC HELPERS
# =============================================================================
def heuristic(a, b):
    """Manhattan distance on the grid (no diagonals)."""
    (x1, y1), (x2, y2) = a, b
    return abs(x1 - x2) + abs(y1 - y2)


def neighbors(x, y):
    """Return 4-neighborhood within bounds."""
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < COLS and 0 <= ny < ROWS:
            yield nx, ny


# -----------------------------------------------------------------------------
#                                 A* PATHFINDING
# -----------------------------------------------------------------------------
def astar(grid, start, goal, passable=lambda t: t != TileType.WALL, cost_fn=None, limit=4000):
    """
    Grid A* with optional per-tile cost function and node expansion limit.
    Returns path [start...goal] or [] if unreachable.
    """
    if start == goal:
        return [start]

    from heapq import heappush, heappop
    frontier = []
    heappush(frontier, (0, start))

    came_from = {start: None}
    g_cost    = {start: 0}

    expanded = 0
    while frontier and expanded < limit:
        _, current = heappop(frontier)
        expanded += 1

        if current == goal:
            break

        cx, cy = current
        for nx, ny in neighbors(cx, cy):
            t = grid[ny][nx]
            if not passable(t):
                continue

            step = 1 if cost_fn is None else cost_fn((nx, ny), t)
            new_cost = g_cost[current] + step
            if (nx, ny) not in g_cost or new_cost < g_cost[(nx, ny)]:
                g_cost[(nx, ny)] = new_cost
                priority = new_cost + heuristic((nx, ny), goal)
                heappush(frontier, (priority, (nx, ny)))
                came_from[(nx, ny)] = current

    if goal not in came_from:
        return []

    # Reconstruct path backward
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


# -----------------------------------------------------------------------------
#                          LINE OF SIGHT (grid ray)
# -----------------------------------------------------------------------------
def los(grid, a, b):
    """
    Simple Bresenham-style ray cast on the grid.
    Returns True if no WALL blocks the line between a and b.
    """
    (x0, y0) = a
    (x1, y1) = b
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if grid[y][x] == TileType.WALL:
            return False
        if (x, y) == (x1, y1):
            return True
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


# =============================================================================
#                          WORLD / LEVEL GENERATION
# =============================================================================
def empty_grid():
    """Start with all FLOOR."""
    return [[TileType.FLOOR for _ in range(COLS)] for _ in range(ROWS)]


def carve_rect(grid, rect, wall=True):
    """Draw perimeter walls for a room rectangle (interior left as-is)."""
    for x in range(rect.left, rect.right):
        for y in range(rect.top, rect.bottom):
            if x in (rect.left, rect.right - 1) or y in (rect.top, rect.bottom - 1):
                grid[y][x] = TileType.WALL if wall else grid[y][x]


def make_doors(grid, rooms):
    """
    Place doors along boundaries where two rooms touch.
    """
    doors = []
    for i, r in enumerate(rooms):
        for j, s in enumerate(rooms):
            if j <= i:
                continue
            inter = r.rect.inflate(2, 2).clip(s.rect)

            # Vertical boundary
            if inter.width == 0 and inter.height > 2:
                y = RNG.randrange(max(r.rect.top + 2, s.rect.top + 2), min(r.rect.bottom - 2, s.rect.bottom - 2))
                x = r.rect.right - 1
                if 0 <= x < COLS and 0 <= y < ROWS:
                    grid[y][x] = TileType.DOOR
                    doors.append((x, y))

            # Horizontal boundary
            elif inter.height == 0 and inter.width > 2:
                x = RNG.randrange(max(r.rect.left + 2, s.rect.left + 2), min(r.rect.right - 2, s.rect.right - 2))
                y = r.rect.bottom - 1
                if 0 <= x < COLS and 0 <= y < ROWS:
                    grid[y][x] = TileType.DOOR
                    doors.append((x, y))
    return doors


def ensure_room_entrance(grid, room, doors_list):
    """
    Guarantee each room has at least one DOOR on its perimeter.
    Avoids sealed rooms.
    """
    sides = [
        ("top",   [(x, room.rect.top)      for x in range(room.rect.left+1, room.rect.right-1)]),
        ("bottom",[(x, room.rect.bottom-1) for x in range(room.rect.left+1, room.rect.right-1)]),
        ("left",  [(room.rect.left,  y)    for y in range(room.rect.top+1, room.rect.bottom-1)]),
        ("right", [(room.rect.right-1, y)  for y in range(room.rect.top+1, room.rect.bottom-1)]),
    ]

    # Already has a door?
    for _, cells in sides:
        if any(grid[y][x] == TileType.DOOR for (x, y) in cells):
            return

    # Prefer a side that opens to FLOOR outside the wall
    candidates = []
    for name, cells in sides:
        for (x, y) in cells:
            if name == "top":      nb = (x, y-1)
            elif name == "bottom": nb = (x, y+1)
            elif name == "left":   nb = (x-1, y)
            else:                  nb = (x+1, y)
            nx, ny = nb
            if 0 <= nx < COLS and 0 <= ny < ROWS and grid[ny][nx] != TileType.WALL:
                candidates.append((x, y))
                break

    if not candidates:
        for _, cells in sides:
            candidates += cells

    dx, dy = RNG.choice(candidates)
    grid[dy][dx] = TileType.DOOR
    doors_list.append((dx, dy))


def generate_world():
    """
    Build a full level and return:
    grid, rooms, doors, hackables, cameras, exits
    """
    grid = empty_grid()
    rooms = []

    # Border walls
    for x in range(COLS):
        grid[0][x] = TileType.WALL
        grid[ROWS - 1][x] = TileType.WALL
    for y in range(ROWS):
        grid[y][0] = TileType.WALL
        grid[y][COLS - 1] = TileType.WALL

    # Place rooms (avoid overlaps)
    room_count = 9
    attempts = 0
    while len(rooms) < room_count and attempts < 200:
        attempts += 1
        w = RNG.randrange(14, 24)
        h = RNG.randrange(10, 18)
        x = RNG.randrange(2, COLS - w - 2)
        y = RNG.randrange(2, ROWS - h - 2)
        rect = pygame.Rect(x, y, w, h)
        if any(rect.inflate(4, 4).colliderect(r.rect) for r in rooms):
            continue
        rooms.append(Room(id=len(rooms), rect=rect, type='regular', value=RNG.randrange(1, 10)))
        carve_rect(grid, rect, wall=True)

    # Mark special rooms
    if rooms:
        vault = max(rooms, key=lambda r: r.value)
        vault.type = 'vault'
        vault.value = 12

        sec_rooms = RNG.sample([r for r in rooms if r is not vault], k=min(2, max(1, len(rooms)//5)))
        for s in sec_rooms:
            s.type = 'security'
            s.value = max(6, s.value)

        util_candidates = [r for r in rooms if r not in sec_rooms and r is not vault]
        if util_candidates:
            util = RNG.choice(util_candidates)
            util.type = 'utility'
            util.value = max(4, util.value)

    # Doors + entrance guarantee
    doors = make_doors(grid, rooms)
    for r in rooms:
        ensure_room_entrance(grid, r, doors)

    # Cameras & terminals inside rooms
    hackables = []
    cameras = []
    for r in rooms:
        for _ in range(RNG.randrange(1, 3)):
            x = RNG.randrange(r.rect.left + 2, r.rect.right - 2)
            y = RNG.randrange(r.rect.top + 2, r.rect.bottom - 2)
            if grid[y][x] == TileType.FLOOR:
                if r.type in ('security', 'utility') and RNG.random() < 0.5:
                    grid[y][x] = TileType.CAMERA
                    cameras.append((x, y))
                else:
                    grid[y][x] = TileType.TERMINAL
                    hackables.append((x, y))

    # A few exits on the outer walls
    exits = []
    for _ in range(3):
        side = RNG.choice(['top', 'bottom', 'left', 'right'])
        if side in ('top', 'bottom'):
            x = RNG.randrange(3, COLS - 3)
            y = 1 if side == 'top' else ROWS - 2
        else:
            y = RNG.randrange(3, ROWS - 3)
            x = 1 if side == 'left' else COLS - 2
        if grid[y][x] != TileType.WALL:
            grid[y][x] = TileType.EXIT
            exits.append((x, y))

    return grid, rooms, doors, hackables, cameras, exits


# =============================================================================
#                           ENTITIES & SHARED STATE
# =============================================================================
@dataclass
class Blackboard:
    """
    Shared memory for guards:
    - lkp / lkp_time: last seen position & when it was updated
    - noises: list of NoisePing (pos, strength), decays over time
    - hacked: set of hacked tiles (for small suspicion bonus)
    """
    lkp: tuple | None = None
    lkp_time: float = 0.0
    noises: list = field(default_factory=list)
    hacked: set = field(default_factory=set)

    def add_noise(self, pos, strength):
        self.noises.append(NoisePing(pos, strength))

    def decay_noises(self, dt):
        for n in self.noises:
            n.strength *= (NOISE_DECAY_PER_SECOND ** dt)
        self.noises = [n for n in self.noises if n.strength > 0.05]


@dataclass
class ThiefAI:
    x: int
    y: int
    carrying_value: int = 0
    sprinting: bool = False
    path: list = field(default_factory=list)
    last_plan: float = 0.0
    decoy_cd: float = 0.0
    hack_cd: float = 0.0
    def tile(self): return (self.x, self.y)


class GuardState:
    PATROL      = 'patrol'
    INVESTIGATE = 'investigate'
    SEARCH      = 'search'
    CHASE       = 'chase'
    BLOCKADE    = 'blockade'
    RETURN      = 'return'


@dataclass
class Guard:
    x: int
    y: int
    facing: tuple = (1, 0)
    state: str = GuardState.PATROL
    suspicion: float = 0.0
    path: list = field(default_factory=list)
    patrol_route: list = field(default_factory=list)
    last_plan: float = 0.0
    color: tuple = field(default_factory=lambda: (
        RNG.randrange(140, 220), RNG.randrange(140, 220), RNG.randrange(140, 220)
    ))
    step_acc: float = 0.0
    def tile(self): return (self.x, self.y)


# =============================================================================
#                         GUARD PERCEPTION & CHOICES
# =============================================================================
def guard_can_see_thief(grid, guard: Guard, thief: ThiefAI):
    """
    LOS + distance + FOV angle check.
    Returns (saw: bool, confidence: 0..1)
    """
    gx, gy = guard.tile()
    px, py = thief.tile()

    if heuristic((gx, gy), (px, py)) > GUARD_VIEW_DIST:
        return False, 0.0
    if not los(grid, (gx, gy), (px, py)):
        return False, 0.0

    fx, fy = guard.facing
    vx, vy = px - gx, py - gy
    vlen = math.hypot(vx, vy) + 1e-5
    vx /= vlen; vy /= vlen
    dot = fx * vx + fy * vy
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    if angle <= GUARD_FOV_DEG / 2 + 10:
        dist = max(1, heuristic((gx, gy), (px, py)))
        conf = max(0.15, min(1.0, 1.6 / dist))
        return True, conf
    return False, 0.0


def fuzzy_suspicion(guard: Guard, saw: bool, sight_conf: float, local_noise: float,
                    hacked_events: float, dt: float):
    """
    Suspicion rises from sight/noise/hacks, decays over time.
    Returns the desired state based on thresholds.
    """
    guard.suspicion *= (SUSP_DECAY_PER_SECOND ** dt)
    if saw:
        guard.suspicion += SUSP_INCREASE_SIGHT * sight_conf
    guard.suspicion += SUSP_INCREASE_NOISE * min(2.0, local_noise)
    guard.suspicion += 0.03 * min(1.0, hacked_events)
    guard.suspicion = max(0.0, min(1.0, guard.suspicion))

    if guard.suspicion >= SUSP_THRESHOLD_CHASE:        return GuardState.CHASE
    if guard.suspicion >= SUSP_THRESHOLD_PATROL_PLUS:  return GuardState.INVESTIGATE
    if guard.suspicion >= SUSP_THRESHOLD_INVESTIGATE:  return GuardState.INVESTIGATE
    return GuardState.PATROL


def heuristic_room_score(room: Room, lkp: tuple | None, exits, noise_map, camera_map):
    """
    Lower score = more likely place to search next.
    Mixes distance to LKP, room value, exit closeness, noise, and camera density.
    """
    if lkp is None:
        dist_term = 8.0
    else:
        cx = (room.rect.left + room.rect.right) // 2
        cy = (room.rect.top + room.rect.bottom) // 2
        dist_term = min(8.0, heuristic(lkp, (cx, cy)) / 10)

    value_term = (10 - room.value) / 10

    exit_term = 1.0
    if exits:
        center = ((room.rect.left + room.rect.right)//2, (room.rect.top + room.rect.bottom)//2)
        exdist = min(heuristic(center, e) for e in exits)
        exit_term = min(1.5, exdist / 20)

    noise_term = 1.0
    if noise_map:
        best = 0
        for (px, py), strength in noise_map.items():
            if room.rect.collidepoint(px, py) and strength > best:
                best = strength
        noise_term = 1.2 - min(1.0, best)

    blind_term = 1.0
    if camera_map:
        n_cam = sum(1 for (px, py) in camera_map if room.rect.collidepoint(px, py))
        blind_term = 1.2 - min(1.0, n_cam / 4)

    score = (
        W_DIST  * dist_term +
        W_VALUE * value_term +
        W_EXIT  * exit_term +
        W_NOISE * (1.2 - noise_term) +
        W_BLIND * (1.2 - blind_term)
    )
    return score


def pick_search_rooms(rooms, lkp, exits, noises, cameras, k=2):
    """Pick k rooms with the best (lowest) scores."""
    noise_map = defaultdict(float)
    for n in noises:
        noise_map[n.pos] = max(noise_map[n.pos], n.strength)
    camera_set = set(cameras)
    scored = [(heuristic_room_score(r, lkp, exits, noise_map, camera_set), r) for r in rooms]
    scored.sort(key=lambda t: t[0])
    return [r for _, r in scored[:k]]


def plan_blockade(grid, lkp, exits):
    """
    Find up to 2 chokepoints along shortest paths from LKP to each Exit.
    """
    points = []
    if lkp is None or not exits:
        return points

    def passable(t): return t != TileType.WALL

    for ex in exits:
        path = astar(grid, lkp, ex, passable=passable)
        if not path:
            continue
        best_tile = None
        best_deg = 5
        for (x, y) in path[2:-2]:
            deg = sum(1 for nx, ny in neighbors(x, y) if passable(grid[ny][nx]))
            if deg <= best_deg:
                best_deg = deg
                best_tile = (x, y)
        if best_tile:
            points.append(best_tile)

    uniq = []
    for p in points:
        if p not in uniq:
            uniq.append(p)
    return uniq[:2]


# =============================================================================
#                                 GAME CLASS
# =============================================================================
class Game:
    def __init__(self):
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Cyber Heist — AI vs AI (Prototype)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.big  = pygame.font.SysFont("consolas", 22)

        # Build world
        self.grid, self.rooms, self.doors, self.hackables, self.cameras, self.exits = generate_world()

        # Thief spawns at any boundary EXIT (create one if list is empty)
        if self.exits:
            spawn = RNG.choice(self.exits)
        else:
            x, y = 1, ROWS // 2
            self.grid[y][x] = TileType.EXIT
            self.exits.append((x, y))
            spawn = (x, y)
        self.thief = ThiefAI(*spawn)

        # Shared blackboard
        self.blackboard = Blackboard()

        # Guards at centers of some rooms with simple perimeter patrols
        self.guards: list[Guard] = []
        bases = RNG.sample(self.rooms, k=min(GUARD_COUNT, len(self.rooms)))
        for b in bases:
            gx = (b.rect.left + b.rect.right) // 2
            gy = (b.rect.top + b.rect.bottom) // 2
            g = Guard(gx, gy)
            g.patrol_route = self.build_patrol(b)
            self.guards.append(g)

        # Debug toggles
        self.show_los       = True
        self.show_paths     = True
        self.show_scores    = False
        self.show_noises    = True
        self.show_blockades = True

        # Game state
        self.time = 0.0
        self.win  = False
        self.lose = False

        # Logging / pacing helpers
        self._start_wall     = time.time()
        self._ticks          = 0
        self._thief_step_acc = 0.0
        for g in self.guards:
            g.step_acc = 0.0

        # Slow-mo
        self.time_scale = TIME_SCALE_NORM

    # ----------------------- Patrol route for one room ------------------------
    def build_patrol(self, room: Room):
        route = []
        for x in range(room.rect.left + 2, room.rect.right - 2, 4):
            route.append((x, room.rect.top + 2))
        for y in range(room.rect.top + 2, room.rect.bottom - 2, 4):
            route.append((room.rect.right - 3, y))
        for x in range(room.rect.right - 3, room.rect.left + 1, -4):
            route.append((x, room.rect.bottom - 3))
        for y in range(room.rect.bottom - 3, room.rect.top + 1, -4):
            route.append((room.rect.left + 2, y))
        if not route:
            route = [(room.rect.centerx, room.rect.centery)]
        return route

    # =============================== THIEF ===================================
    def passable_for_thief(self, tile_type):
        return tile_type != TileType.WALL

    def thief_cost(self, pos, tile_type):
        if tile_type == TileType.CAMERA: return 1.6  # avoid cameras if possible
        if tile_type == TileType.DOOR:   return 1.3  # small extra cost
        return 1.0

    def tile_inside_room(self, room: Room):
        """Pick a walkable tile roughly at the room center."""
        cx, cy = room.rect.centerx, room.rect.centery
        if self.grid[cy][cx] != TileType.WALL:
            return (cx, cy)
        for y in range(room.rect.top + 1, room.rect.bottom - 1):
            for x in range(room.rect.left + 1, room.rect.right - 1):
                if self.grid[y][x] != TileType.WALL:
                    return (x, y)
        return (cx, cy)

    def plan_thief(self):
        """Path to Vault (if empty-handed) or to nearest Exit (if carrying)."""
        if self.thief.carrying_value == 0:
            vaults = [r for r in self.rooms if r.type == 'vault']
            if not vaults:
                return
            goal = self.tile_inside_room(vaults[0])
            self.thief.path = astar(self.grid, self.thief.tile(), goal,
                                    passable=self.passable_for_thief, cost_fn=self.thief_cost)
        else:
            best_path = []
            for ex in self.exits:
                p = astar(self.grid, self.thief.tile(), ex,
                          passable=self.passable_for_thief, cost_fn=self.thief_cost)
                if p and (not best_path or len(p) < len(best_path)):
                    best_path = p
            self.thief.path = best_path
        self.thief.last_plan = self.time

    def thief_hack_if_beneficial(self):
        """
        If a DOOR/CAMERA/TERMINAL is adjacent and helpful (or a camera),
        turn it to FLOOR (hack), make hack noise, and start cooldown.
        """
        if self.thief.hack_cd > 0:
            return
        for tx, ty in neighbors(*self.thief.tile()):
            tile = self.grid[ty][tx]
            if tile in (TileType.DOOR, TileType.CAMERA, TileType.TERMINAL):
                near_on_path = self.thief.path and (tx, ty) in self.thief.path[:6]
                if tile == TileType.CAMERA or near_on_path:
                    self.grid[ty][tx] = TileType.FLOOR
                    self.blackboard.hacked.add((tx, ty))
                    self.blackboard.add_noise((tx, ty), NOISE_HACK)
                    self.thief.hack_cd = THIEF_HACK_COOLDOWN
                    return

    def thief_decoy_if_chased(self):
        """Drop a decoy noise if chased or if a guard is very close."""
        if self.thief.decoy_cd > 0 or not self.guards:
            return
        nearest = min(heuristic(g.tile(), self.thief.tile()) for g in self.guards)
        chased = any(g.state == GuardState.CHASE or g.suspicion >= SUSP_THRESHOLD_CHASE for g in self.guards) or nearest <= 5
        if not chased:
            return
        g = min(self.guards, key=lambda gg: heuristic(gg.tile(), self.thief.tile()))
        gx, gy = g.tile(); px, py = self.thief.tile()
        dx = max(-1, min(1, px - gx)); dy = max(-1, min(1, py - gy))
        tx = max(1, min(COLS - 2, px + dx * 4)); ty = max(1, min(ROWS - 2, py + dy * 4))
        if self.grid[ty][tx] != TileType.WALL:
            self.blackboard.add_noise((tx, ty), NOISE_DECOY)
            self.thief.decoy_cd = THIEF_DECOY_COOLDOWN

    def update_thief(self, dt):
        """One frame of thief logic: cooldowns, planning, actions, movement, win-check."""
        self.thief.decoy_cd = max(0.0, self.thief.decoy_cd - dt)
        self.thief.hack_cd  = max(0.0, self.thief.hack_cd  - dt)

        if (not self.thief.path) or (self.time - self.thief.last_plan) > THIEF_REPLAN_SECONDS:
            self.plan_thief()

        self.thief_hack_if_beneficial()
        self.thief_decoy_if_chased()

        nearest = min(heuristic(g.tile(), self.thief.tile()) for g in self.guards) if self.guards else 999
        self.thief.sprinting = nearest <= 5 or any(g.state == GuardState.CHASE for g in self.guards)

        if self.thief.path:
            if self.thief.tile() == self.thief.path[0]:
                self.thief.path.pop(0)
            if self.thief.path:
                self._thief_step_acc += dt
                if self._thief_step_acc >= THIEF_STEP_DELAY:
                    self._thief_step_acc -= THIEF_STEP_DELAY
                    nx, ny = self.thief.path[0]
                    if self.grid[ny][nx] != TileType.WALL:
                        self.thief.x, self.thief.y = nx, ny
                        self.blackboard.add_noise(self.thief.tile(), NOISE_SPRINT if self.thief.sprinting else NOISE_BASE)

        # Pick up loot when standing in the vault area
        for r in self.rooms:
            if r.type == 'vault' and r.rect.collidepoint(self.thief.x, self.thief.y):
                self.thief.carrying_value = r.value

        # If carrying loot and stepping on an Exit → win
        if self.thief.carrying_value > 0:
            for ex in self.exits:
                if heuristic(self.thief.tile(), ex) == 0:
                    self.win = True
                    log_result_csv(SEED, "THIEF", time.time() - self._start_wall, self._ticks, self.thief.carrying_value)
                    return

    # =============================== GUARDS ==================================
    def passable_for_guard(self, tile_type):
        return tile_type != TileType.WALL

    def guard_cost(self, pos, tile_type):
        return 2.0 if tile_type == TileType.DOOR else 1.0

    def build_patrol_targets(self, g: Guard):
        if not g.patrol_route:
            g.patrol_route = [(g.x, g.y)]
        return RNG.choice(g.patrol_route)

    def update_guards(self, dt):
        """One frame of guard logic for every guard."""
        self.blackboard.decay_noises(dt)

        # Small suspicion bonus if hacks occurred near thief recently
        hacked_recent = 0.0
        for (hx, hy) in list(self.blackboard.hacked)[-6:]:
            if heuristic((hx, hy), self.thief.tile()) < 12:
                hacked_recent += 0.15
        hacked_recent = min(1.0, hacked_recent)

        # If there is a recent LKP, consider chokepoints to block
        blockades = plan_blockade(self.grid, self.blackboard.lkp, self.exits)

        for gi, g in enumerate(self.guards):
            # Vision check updates LKP if thief is seen
            saw, conf = guard_can_see_thief(self.grid, g, self.thief)
            if saw:
                self.blackboard.lkp = self.thief.tile()
                self.blackboard.lkp_time = self.time
                g.facing = (max(-1, min(1, self.thief.x - g.x)), max(-1, min(1, self.thief.y - g.y)))

            # Sum nearby noise (within ~10 tiles) with linear falloff
            local_noise = 0.0
            for n in self.blackboard.noises:
                d = heuristic(g.tile(), n.pos)
                if d <= 10:
                    local_noise += n.strength * (1.0 - d / 10)

            # Suspicion → desired state
            desired = fuzzy_suspicion(g, saw, conf, local_noise, hacked_recent, dt)

            # Choose state
            if desired == GuardState.CHASE:
                g.state = GuardState.CHASE
            else:
                if self.blackboard.lkp and (self.time - self.blackboard.lkp_time) < GUARD_MEMORY_SECONDS:
                    if gi % 3 == 0 and blockades:
                        g.state = GuardState.BLOCKADE
                    else:
                        g.state = GuardState.SEARCH
                else:
                    g.state = GuardState.PATROL

            # Plan path for current state
            if g.state == GuardState.PATROL:
                if not g.path:
                    wp = self.build_patrol_targets(g)
                    g.path = astar(self.grid, g.tile(), wp, passable=self.passable_for_guard, cost_fn=self.guard_cost)
            elif g.state == GuardState.SEARCH:
                if (not g.path) or self.time - g.last_plan > 1.5:
                    rooms = pick_search_rooms(self.rooms, self.blackboard.lkp, self.exits, self.blackboard.noises, self.cameras, k=2)
                    dest = RNG.choice(rooms) if rooms else RNG.choice(self.rooms)
                    tx = RNG.randrange(dest.rect.left + 1, dest.rect.right - 1)
                    ty = RNG.randrange(dest.rect.top + 1,  dest.rect.bottom - 1)
                    g.path = astar(self.grid, g.tile(), (tx, ty), passable=self.passable_for_guard, cost_fn=self.guard_cost)
                    g.last_plan = self.time
            elif g.state == GuardState.BLOCKADE:
                if blockades:
                    bp = blockades[min(gi, len(blockades) - 1)]
                    g.path = astar(self.grid, g.tile(), bp, passable=self.passable_for_guard, cost_fn=self.guard_cost)
            elif g.state == GuardState.CHASE:
                dest = self.thief.tile() if saw else self.blackboard.lkp
                if dest:
                    g.path = astar(self.grid, g.tile(), dest, passable=self.passable_for_guard, cost_fn=self.guard_cost, limit=8000)

            # Step along the path using per-state delay
            if g.path:
                if g.tile() == g.path[0]:
                    g.path.pop(0)
                if g.path:
                    delay = GUARD_STEP_DELAY_CHASE if g.state == GuardState.CHASE else GUARD_STEP_DELAY_PATROL
                    g.step_acc += dt
                    if g.step_acc >= delay:
                        g.step_acc -= delay
                        nx, ny = g.path[0]
                        g.facing = (max(-1, min(1, nx - g.x)), max(-1, min(1, ny - g.y)))
                        g.x, g.y = nx, ny

            # Capture condition
            if heuristic(g.tile(), self.thief.tile()) == 0:
                self.lose = True
                log_result_csv(SEED, "GUARDS", time.time() - self._start_wall, self._ticks, self.thief.carrying_value)

    # =============================== DRAWING =================================
    def draw_grid(self):
        """Draw floor/walls/doors/cameras/terminals/exits and room frames."""
        s = self.screen
        for y in range(ROWS):
            for x in range(COLS):
                t = self.grid[y][x]
                color = DARK
                if t == TileType.WALL:     color = MID
                elif t == TileType.DOOR:   color = (100, 120, 140)
                elif t == TileType.CAMERA: color = (120, 140, 180)
                elif t == TileType.TERMINAL: color = (90, 140, 90)
                elif t == TileType.EXIT:   color = (70, 140, 90)
                pygame.draw.rect(s, color, (x * TILE, y * TILE, TILE - 1, TILE - 1))

        for r in self.rooms:
            pygame.draw.rect(s, (60, 65, 72),
                             (r.rect.left * TILE, r.rect.top * TILE, r.rect.width * TILE, r.rect.height * TILE), 2)
            label = f"{r.type[:1].upper()}:{r.value}"
            s.blit(self.font.render(label, True, (160, 170, 185)),
                   ((r.rect.left + 1) * TILE, (r.rect.top + 1) * TILE))

        for ex in self.exits:
            pygame.draw.rect(s, GREEN, (ex[0] * TILE, ex[1] * TILE, TILE - 1, TILE - 1))

    def draw_entities(self):
        """Draw noises, the thief, and guards (with LOS and paths if enabled)."""
        if self.show_noises:
            for n in self.blackboard.noises:
                pygame.draw.circle(self.screen, ORANGE,
                                   (n.pos[0] * TILE + TILE // 2, n.pos[1] * TILE + TILE // 2),
                                   max(4, int(n.strength * 12)), 1)

        # Thief (white outline + teal body + 'T')
        tx, ty = self.thief.x * TILE, self.thief.y * TILE
        body = pygame.Rect(tx + 2, ty + 2, TILE - 4, TILE - 4)
        pygame.draw.rect(self.screen, (255, 255, 255), body.inflate(4, 4), 2)
        pygame.draw.rect(self.screen, ACCENT, body)
        self.screen.blit(self.font.render("T", True, (10, 10, 10)), (tx + TILE // 2 - 5, ty + TILE // 2 - 8))

        # Thief future path (cyan)
        if self.thief.path and self.show_paths:
            for (px, py) in self.thief.path[:20]:
                pygame.draw.rect(self.screen, (180, 240, 240), (px * TILE + 6, py * TILE + 6, TILE - 12, TILE - 12), 1)

        # Guards
        for g in self.guards:
            color = RED if g.state == GuardState.CHASE else YELLOW if g.state in \
                (GuardState.SEARCH, GuardState.BLOCKADE, GuardState.INVESTIGATE) else BLUE
            pygame.draw.rect(self.screen, color, (g.x * TILE + 2, g.y * TILE + 2, TILE - 4, TILE - 4))

            # suspicion bar (tiny red strip at bottom)
            w = int((TILE - 4) * g.suspicion)
            pygame.draw.rect(self.screen, (255, 120, 120), (g.x * TILE + 2, g.y * TILE + TILE - 6, w, 4))

            # LOS ray
            if self.show_los:
                end = (int((g.x + g.facing[0] * GUARD_VIEW_DIST) * TILE + TILE / 2),
                       int((g.y + g.facing[1] * GUARD_VIEW_DIST) * TILE + TILE / 2))
                pygame.draw.line(self.screen, (120, 120, 160),
                                 (g.x * TILE + TILE // 2, g.y * TILE + TILE // 2), end, 1)

            # Guard path (purple)
            if self.show_paths and g.path:
                for (px, py) in g.path[:12]:
                    pygame.draw.rect(self.screen, PURPLE, (px * TILE + 6, py * TILE + 6, TILE - 12, TILE - 12), 1)

    def draw_hud(self):
        """Draw LKP marker, blockade hints, info text, and WIN/LOSE banner."""
        if self.blackboard.lkp and (self.time - self.blackboard.lkp_time) < GUARD_MEMORY_SECONDS:
            x, y = self.blackboard.lkp
            pygame.draw.rect(self.screen, (255, 160, 160), (x * TILE + 4, y * TILE + 4, TILE - 8, TILE - 8), 2)

        if self.show_blockades:
            for bp in plan_blockade(self.grid, self.blackboard.lkp, self.exits):
                pygame.draw.rect(self.screen, (180, 100, 255), (bp[0] * TILE + 4, bp[1] * TILE + 4, TILE - 8, TILE - 8), 2)

        info = f"ThiefValue:{self.thief.carrying_value}  Noises:{len(self.blackboard.noises)}  LKP:{self.blackboard.lkp}  Thief:{self.thief.tile()}"
        self.screen.blit(self.font.render(info, True, LIGHT), (10, 8))
        self.screen.blit(self.font.render("[SPACE] Slow-mo  [1-5] Debug  [R] Restart  [ESC] Quit — AI vs AI", True, LIGHT),
                         (10, HEIGHT - 28))

        if self.win:
            txt = self.big.render("WIN: Thief exfiltrated the data!", True, GREEN)
            self.screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, 20))
        if self.lose:
            txt = self.big.render("LOSE: Guards captured the thief!", True, RED)
            self.screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, 20))

    # =============================== MAIN LOOP ================================
    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            dt *= self.time_scale
            self._ticks += 1
            self.time += dt

            # Only debug/utility keys; no manual movement
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit(0)
                    if event.key == pygame.K_r:      self.__init__()  # restart
                    if event.key == pygame.K_1:      self.show_los = not self.show_los
                    if event.key == pygame.K_2:      self.show_paths = not self.show_paths
                    if event.key == pygame.K_4:      self.show_noises = not self.show_noises
                    if event.key == pygame.K_5:      self.show_blockades = not self.show_blockades
                    if event.key == pygame.K_SPACE:  self.time_scale = TIME_SCALE_SLOW if self.time_scale == TIME_SCALE_NORM else TIME_SCALE_NORM

            # Update AIs if game not finished
            if not (self.win or self.lose):
                self.update_thief(dt)
                self.update_guards(dt)

            # Draw the frame
            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_entities()
            self.draw_hud()
            pygame.display.flip()


# =============================================================================
#                             PROGRAM ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    Game().run()
