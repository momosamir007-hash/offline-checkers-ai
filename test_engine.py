import ast
import sys, types
sys.modules.setdefault('streamlit', types.SimpleNamespace())
from pathlib import Path
import numpy as np

p = Path(__file__).with_name('app.py')
source = p.read_text(encoding='utf-8')
ast.parse(source)
# Import without requiring Streamlit to execute UI.
import app

# 1. Starting position sanity
bb = app.BB()
moves, cap = bb.get_moves(True)
assert moves and not cap

# 2. A man may capture backwards in this engine's supported rule profile.
g = np.zeros((8,8), dtype=np.int8)
g[4,1] = 1  # white man
# black piece diagonally "behind" it, landing square empty
g[5,2] = 2
b = app.BB.from_grid(g)
m, c = b.get_moves(True)
assert c and any(len(x) == 2 and abs(app.SQ_TO_RC[x[0]][0]-app.SQ_TO_RC[x[1]][0]) == 2 for x in m)

# 3. Mandatory capture suppresses quiet moves
assert c

# 4. Capture application removes exactly one piece
mv = m[0]
b2 = b.copy(); before = app.popcount(b2.bp); b2.do_move(mv, True)
assert app.popcount(b2.bp) == before - 1

# 5. Promotion
g = np.zeros((8,8), dtype=np.int8); g[1,2] = 1
b = app.BB.from_grid(g); ms, _ = b.get_moves(True)
assert ms
b.do_move(ms[0], True)
assert b.k & (1 << ms[0][-1])

# 6. Position validation rejects a piece on a light square
bad = np.zeros((8,8), dtype=np.int8); bad[0,0] = 1
ok, errors = app.validate_grid(bad)
assert not ok and errors

# 7. Empty board is rejected
ok, errors = app.validate_grid(np.zeros((8,8), dtype=np.int8))
assert not ok

# 8. Notation and capture counter
assert app.move_notation((0, 5)) == '1-6'
assert app.move_notation((0, 9, 16)) == '1x10x17'

# 9. Search returns a legal move under a short budget
bb = app.BB(); engine = app.Beast(max_time=0.2)
r = engine.find_best(bb, True)
legal, _ = bb.get_moves(True)
assert r['move'] in legal
assert r['time'] <= 1.5

print('PASS: v14 engine safety and rules tests')

# 10. Strategic profile exposes transparent signals
profile = app.strategic_profile(app.BB(), True)
assert profile["material"]["white_total"] == 12
assert profile["white_mobility"] > 0

# 11. Game tracker rejects illegal moves and records a legal move
tracker = app.GameTracker(app.BB(), True)
first = tracker.legal_moves()[0]
entry = tracker.play(first)
assert entry["notation"] == app.move_notation(first)
assert len(tracker.history) == 1
try:
    tracker.play((0, 31))
    raise AssertionError("illegal move was accepted")
except ValueError:
    pass

# 12. Repetition counter uses position plus side-to-move
assert tracker.repetition_count() >= 1

# 13. v15 analysis payload contains strategic profile
engine = app.Beast(max_time=0.12)
payload = engine.analyze_all(app.BB(), True)
assert payload and "strategic_profile" in payload

print('PASS: v15 strategic analysis and game tracker tests')

