# Cyber Heist — AI vs AI (Autonomous Thief & Guards)

A small Pygame simulation where an **autonomous Thief AI** steals data from a **Vault** then tries to escape via an **Exit**, while multiple **Guard AIs** patrol, investigate noise/hacks, share last known positions, and coordinate (including simple chokepoint blockades). Movement is slowed intentionally so you can *see* the step-by-step decisions.



---

##  Features

- **Procedural Level** — Random rooms with perimeter walls, guaranteed doors (no sealed rooms), cameras, terminals, and exits on outer walls.  
- **Autonomous Thief AI** — Vault → pick data → A* to Exit. Can **hack** adjacent doors/cameras/terminals (turns to floor), **sprint** when chased (louder noise), and **drop decoy noise**.  
- **Guard AI** — Patrolling, **fuzzy suspicion** (sight + noise + hacks with decay), **shared LKP**, **A*** chase, heuristic **room search**, and **blockade** of chokepoints along LKP→Exit paths.  
- **Readable Visuals** — Per-tile step delays; optional slow-mo.  
- **Run Logging** — Appends `runs_log.csv` (timestamp, seed, winner, elapsed, ticks, loot).

---

## Controls / Debug Keys

- `SPACE` — Slow-mo toggle  
- `R` — Restart  
- `ESC` — Quit

Overlays:
- `1` — Guard LOS (line of sight) on/off  
- `2` — Paths (thief + guards) on/off  
- `4` — Noise pings on/off  
- `5` — Blockade hints on/off

**Legend:** Thief = teal with white outline + **“T”** • Exits = **green** • Noise = **orange** rings • LKP = **red** square • Guard paths = **purple** • Thief path = light cyan

---

##  Requirements

- Python **3.9+**  
- `pygame` **2.x**

Install:

`pip install pygame`

Run:

`python cyber.py`

OR

`python3 cyber.py`

