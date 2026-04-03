#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║       ♟️ مساعد الداما الذكي — النسخة النهائية v11    ║
║  Bitboard Engine + XGBoost Model + Groq + OpenCV     ║
║  مدمج مع نموذج التدريب العميق Brutal AI (مصحح)       ║
╚══════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time, random, json, urllib.request, urllib.error, os
from enum import IntEnum

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ══════════════════════════════════════════════════════
# تحميل نموذج التدريب العميق (XGBoost)
# ══════════════════════════════════════════════════════
AI_MODEL = None
MODEL_PATH = "brutal_ai_model (1).json"

if HAS_XGB and os.path.exists(MODEL_PATH):
    try:
        AI_MODEL = xgb.Booster()
        AI_MODEL.load_model(MODEL_PATH)
    except Exception as e:
        print(f"⚠️ خطأ أثناء تحميل نموذج الذكاء الاصطناعي: {e}")
        AI_MODEL = None

# ══════════════════════════════════════════════════════
# 1. جداول محسوبة مسبقاً
# ══════════════════════════════════════════════════════

# تحويل المربعات 0-31 إلى (صف, عمود) والعكس
SQ_TO_RC = []
RC_TO_SQ = {}
_sq = 0
for _r in range(8):
    for _c in range(8):
        if (_r + _c) % 2 != 0:
            SQ_TO_RC.append((_r, _c))
            RC_TO_SQ[(_r, _c)] = _sq
            _sq += 1

# جيران كل مربع: [NW, NE, SW, SE] أو -1
NBR = []
for _sq in range(32):
    _r, _c = SQ_TO_RC[_sq]
    _nb = []
    for _dr, _dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        _nr, _nc = _r + _dr, _c + _dc
        if 0 <= _nr < 8 and 0 <= _nc < 8 and (_nr, _nc) in RC_TO_SQ:
            _nb.append(RC_TO_SQ[(_nr, _nc)])
        else:
            _nb.append(-1)
    NBR.append(tuple(_nb))

# جدول القفز: (وسط, هبوط) لكل اتجاه
JUMP_TABLE = []
for _sq in range(32):
    _jt = []
    for _d in range(4):
        _mid = NBR[_sq][_d]
        if _mid == -1:
            _jt.append((-1, -1))
        else:
            _land = NBR[_mid][_d]
            _jt.append((_mid, _land) if _land != -1 else (-1, -1))
    JUMP_TABLE.append(tuple(_jt))

# أقنعة الصفوف
ROW_MASK = [0] * 8
for _sq in range(32):
    _r = SQ_TO_RC[_sq][0]
    ROW_MASK[_r] |= (1 << _sq)

W_PROMO = ROW_MASK[0]
B_PROMO = ROW_MASK[7]

# أقنعة المركز
CENTER_MASK = 0
INNER_CENTER = 0
for _sq in range(32):
    _r, _c = SQ_TO_RC[_sq]
    if 2 <= _r <= 5 and 2 <= _c <= 5:
        CENTER_MASK |= (1 << _sq)
    if 3 <= _r <= 4 and 3 <= _c <= 4:
        INNER_CENTER |= (1 << _sq)

# قيم موضعية لكل مربع
W_POS_VAL = []
B_POS_VAL = []
K_POS_VAL = []
for _sq in range(32):
    _r, _c = SQ_TO_RC[_sq]
    w_val = (7 - _r) * 3
    if _r == 7: w_val = 5
    if _r <= 1: w_val += 15
    if 2 <= _r <= 5 and 2 <= _c <= 5: w_val += 4
    W_POS_VAL.append(w_val)

    b_val = _r * 3
    if _r == 0: b_val = 5
    if _r >= 6: b_val += 15
    if 2 <= _r <= 5 and 2 <= _c <= 5: b_val += 4
    B_POS_VAL.append(b_val)

    dist = abs(_r - 3.5) + abs(_c - 3.5)
    K_POS_VAL.append(max(0, int(8 - dist * 1.5)))

# Zobrist
_RNG = random.Random(42)
Z_KEYS = {}
for _p in range(4):
    for _sq in range(32):
        Z_KEYS[(_p, _sq)] = _RNG.getrandbits(64)
Z_SIDE = _RNG.getrandbits(64)

# الوضعية الابتدائية
INIT_BP = sum(1 << i for i in range(12))
INIT_WP = sum(1 << i for i in range(20, 32))

CELL = 80
BPX = CELL * 8
W_DIRS = (0, 1)
B_DIRS = (2, 3)
K_DIRS = (0, 1, 2, 3)
EXACT, LOWER, UPPER = 0, 1, 2


# ══════════════════════════════════════════════════════
# 2. Bitboard Engine
# ══════════════════════════════════════════════════════

def popcount(x):
    return bin(x).count('1')

def iter_bits(x):
    while x:
        lsb = x & (-x)
        yield lsb.bit_length() - 1
        x ^= lsb

class BB:
    __slots__ = ['wp', 'bp', 'k']

    def __init__(self, wp=INIT_WP, bp=INIT_BP, k=0):
        self.wp = wp
        self.bp = bp
        self.k = k

    def copy(self):
        b = BB.__new__(BB)
        b.wp = self.wp; b.bp = self.bp; b.k = self.k
        return b

    def zobrist(self, white_turn):
        h = Z_SIDE if white_turn else 0
        for sq in iter_bits(self.wp & ~self.k): h ^= Z_KEYS[(0, sq)]
        for sq in iter_bits(self.bp & ~self.k): h ^= Z_KEYS[(1, sq)]
        for sq in iter_bits(self.wp & self.k):  h ^= Z_KEYS[(2, sq)]
        for sq in iter_bits(self.bp & self.k):  h ^= Z_KEYS[(3, sq)]
        return h

    def get_moves(self, white):
        my = self.wp if white else self.bp
        opp = self.bp if white else self.wp
        occ = self.wp | self.bp
        empty = ~occ & 0xFFFFFFFF

        all_jumps = []
        all_simple = []

        for sq in iter_bits(my):
            is_king = bool((1 << sq) & self.k)
            dirs = K_DIRS if is_king else (W_DIRS if white else B_DIRS)

            can_jump = False
            for d in dirs:
                mid, land = JUMP_TABLE[sq][d]
                if mid != -1 and land != -1:
                    if (1 << mid) & opp and (1 << land) & empty:
                        can_jump = True
                        break

            if can_jump:
                chains = []
                self._find_jumps(sq, dirs, my, opp, empty,
                                 is_king, white, chains, frozenset(), [sq])
                all_jumps.extend(chains)
            else:
                for d in dirs:
                    nb = NBR[sq][d]
                    if nb != -1 and (1 << nb) & empty:
                        all_simple.append((sq, nb))

        if all_jumps:
            mx = max(len(j) for j in all_jumps)
            return [tuple(j) for j in all_jumps if len(j) == mx], True
        return all_simple, False

    def _find_jumps(self, sq, dirs, my, opp, empty, is_king, white, out, eaten, path):
        found = False
        for d in dirs:
            mid, land = JUMP_TABLE[sq][d]
            if mid == -1 or land == -1:
                continue
            if not ((1 << mid) & opp):
                continue
            if mid in eaten:
                continue
            if not ((1 << land) & empty):
                if land != path[0]:
                    continue

            found = True
            promo = False
            if not is_king:
                if white and (1 << land) & W_PROMO: promo = True
                elif not white and (1 << land) & B_PROMO: promo = True

            new_path = path + [land]
            new_eaten = eaten | {mid}
            new_empty = (empty | (1 << sq)) & ~(1 << land)
            new_opp = opp & ~(1 << mid)

            if promo:
                out.append(new_path)
            else:
                new_dirs = dirs
                self._find_jumps(land, new_dirs, my, new_opp,
                                 new_empty, is_king, white,
                                 out, new_eaten, new_path)

        if not found and len(path) > 1:
            out.append(path)

    def do_move(self, move, white):
        start = move[0]
        end = move[-1]
        start_bit = 1 << start
        is_king = bool(start_bit & self.k)

        if white: self.wp &= ~start_bit
        else: self.bp &= ~start_bit
        self.k &= ~start_bit

        for i in range(len(move) - 1):
            s, e = move[i], move[i + 1]
            sr, sc = SQ_TO_RC[s]
            er, ec = SQ_TO_RC[e]
            if abs(sr - er) == 2:
                mr, mc = (sr + er) // 2, (sc + ec) // 2
                if (mr, mc) in RC_TO_SQ:
                    mid_sq = RC_TO_SQ[(mr, mc)]
                    mid_bit = 1 << mid_sq
                    self.wp &= ~mid_bit
                    self.bp &= ~mid_bit
                    self.k &= ~mid_bit

        end_bit = 1 << end
        if white: self.wp |= end_bit
        else: self.bp |= end_bit

        if is_king:
            self.k |= end_bit
        else:
            if white and (end_bit & W_PROMO): self.k |= end_bit
            elif not white and (end_bit & B_PROMO): self.k |= end_bit

    def winner(self):
        if self.wp == 0: return 2
        if self.bp == 0: return 1
        return None

    def full_winner(self):
        w = self.winner()
        if w: return w
        wm, _ = self.get_moves(True)
        bm, _ = self.get_moves(False)
        if not wm and not bm: return 0
        if not wm: return 2
        if not bm: return 1
        return None

    def to_grid(self):
        g = np.zeros((8, 8), dtype=np.int8)
        for sq in iter_bits(self.wp & ~self.k):
            r, c = SQ_TO_RC[sq]; g[r][c] = 1
        for sq in iter_bits(self.bp & ~self.k):
            r, c = SQ_TO_RC[sq]; g[r][c] = 2
        for sq in iter_bits(self.wp & self.k):
            r, c = SQ_TO_RC[sq]; g[r][c] = 3
        for sq in iter_bits(self.bp & self.k):
            r, c = SQ_TO_RC[sq]; g[r][c] = 4
        return g

    @staticmethod
    def from_grid(g):
        wp = bp = k = 0
        for r in range(8):
            for c in range(8):
                p = int(g[r][c])
                if p == 0 or (r, c) not in RC_TO_SQ: continue
                sq = RC_TO_SQ[(r, c)]
                bit = 1 << sq
                if p == 1: wp |= bit
                elif p == 2: bp |= bit
                elif p == 3: wp |= bit; k |= bit
                elif p == 4: bp |= bit; k |= bit
        return BB(wp, bp, k)


# ══════════════════════════════════════════════════════
# 3. Transposition Table
# ══════════════════════════════════════════════════════

class TT:
    __slots__ = ['t', 'hits', 'sz']

    def __init__(self, sz=600000):
        self.t = {}; self.sz = sz; self.hits = 0

    def probe(self, key, depth, alpha, beta):
        e = self.t.get(key)
        if e and e[0] >= depth:
            self.hits += 1
            sc, fl, mv = e[1], e[2], e[3]
            if fl == EXACT: return sc, mv
            if fl == LOWER and sc >= beta: return sc, mv
            if fl == UPPER and sc <= alpha: return sc, mv
        return None

    def store(self, key, depth, sc, fl, mv):
        old = self.t.get(key)
        if not old or old[0] <= depth:
            self.t[key] = (depth, sc, fl, mv)
            if len(self.t) > self.sz:
                ks = list(self.t.keys())
                for k_ in ks[:len(ks) // 2]:
                    del self.t[k_]

    def best(self, key):
        e = self.t.get(key)
        return e[3] if e else None

    def clear(self):
        self.t.clear(); self.hits = 0


# ══════════════════════════════════════════════════════
# 4. المحرك الوحشي (Beast AI) المصحح لـ XGBoost
# ══════════════════════════════════════════════════════

class Beast:

    def __init__(self, max_time=3.0):
        self.max_time = max_time
        self.tt = TT()
        self.nodes = 0
        self.t0 = 0
        self.stop = False
        self.depth_r = 0
        self.killers = [[None, None] for _ in range(64)]
        self.hist = {}
        self._eval_cache = {}  

    def evaluate(self, bb, white_turn):
        # 1. التحقق من الفوز/الخسارة الحتمية
        w = bb.winner()
        if w is not None:
            if w == 0: return 0
            my_win = (w == 1 and white_turn) or (w == 2 and not white_turn)
            return 99999 if my_win else -99999

        # 2. فحص الذاكرة المؤقتة لتسريع الحساب
        cache_key = (bb.wp, bb.bp, bb.k, white_turn)
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        # 3. استخدام نموذج الذكاء الاصطناعي (XGBoost) إن وجد
        if AI_MODEL is not None:
            grid = bb.to_grid().flatten()
            
            # الدقة هنا: نمرر دور اللاعب الحالي الفعلي في الشجرة
            turn = 1 if white_turn else 0
            
            features = np.append(grid, turn).reshape(1, -1)
            feature_names = [f"sq_{i}" for i in range(64)] + ["player_turn"]
            
            dmatrix = xgb.DMatrix(features, feature_names=feature_names)
            
            # النموذج مدرب ليعطي التقييم المطلق (عادة من منظور الأبيض)
            score = float(AI_MODEL.predict(dmatrix)[0])
            
            # خوارزمية NegaMax تتطلب التقييم من منظور اللاعب الذي عليه الدور!
            res = score if white_turn else -score
            self._eval_cache[cache_key] = res
            return res

        # 4. التقييم اليدوي الكلاسيكي (بديل)
        wm = bb.wp & ~bb.k
        bm = bb.bp & ~bb.k
        wk = bb.wp & bb.k
        bk = bb.bp & bb.k

        wn = popcount(wm); wkn = popcount(wk)
        bn = popcount(bm); bkn = popcount(bk)
        wt = wn + wkn; bt = bn + bkn

        sc = 0.0
        endgame = (wt + bt) <= 8
        occ = bb.wp | bb.bp

        sc += wn * 100 + wkn * (170 if endgame else 150)
        sc -= bn * 100 + bkn * (170 if endgame else 150)

        for sq in iter_bits(wm): sc += W_POS_VAL[sq]
        for sq in iter_bits(bm): sc -= B_POS_VAL[sq]
        for sq in iter_bits(wk): sc += K_POS_VAL[sq]
        for sq in iter_bits(bk): sc -= K_POS_VAL[sq]

        sc += popcount(bb.wp & CENTER_MASK) * 4
        sc -= popcount(bb.bp & CENTER_MASK) * 4

        if not endgame:
            sc += popcount(wm & ROW_MASK[7]) * 6
            sc -= popcount(bm & ROW_MASK[0]) * 6

        sc += popcount(wm & ROW_MASK[1]) * 12
        sc -= popcount(bm & ROW_MASK[6]) * 12

        for sq in iter_bits(bb.wp):
            is_k = bool((1 << sq) & bb.k)
            dirs = K_DIRS if is_k else W_DIRS
            allies = 0
            for d in range(4):
                nb = NBR[sq][d]
                if nb != -1 and (1 << nb) & bb.wp: allies += 1
            sc += allies * 3

            for d in dirs:
                mid, land = JUMP_TABLE[sq][d]
                if mid != -1 and land != -1:
                    if (1 << mid) & bb.bp and not ((1 << land) & occ):
                        sc += 8
                        break

            for d in range(4):
                mid, land = JUMP_TABLE[sq][d]
                if mid != -1 and land != -1:
                    opp_d = d ^ 2
                    attacker_sq = NBR[sq][opp_d] if opp_d < 4 else -1
                    if attacker_sq == -1: continue
                    for ad in range(4):
                        a_mid, a_land = JUMP_TABLE[attacker_sq][ad]
                        if a_mid == sq and a_land != -1:
                            if (1 << attacker_sq) & bb.bp and not ((1 << a_land) & occ):
                                sc -= 12
                                break

            if not is_k:
                r = SQ_TO_RC[sq][0]
                clear = True
                for rr in range(r):
                    if (bm | bk) & ROW_MASK[rr]:
                        clear = False; break
                if clear and r <= 3: sc += (4 - r) * 10

            if is_k:
                r, c = SQ_TO_RC[sq]
                if r in (0, 7) or c in (0, 7):
                    exits = 0
                    for d in range(4):
                        nb = NBR[sq][d]
                        if nb != -1 and not ((1 << nb) & occ): exits += 1
                    if exits <= 1: sc -= 8

        for sq in iter_bits(bb.bp):
            is_k = bool((1 << sq) & bb.k)
            dirs = K_DIRS if is_k else B_DIRS
            allies = 0
            for d in range(4):
                nb = NBR[sq][d]
                if nb != -1 and (1 << nb) & bb.bp: allies += 1
            sc -= allies * 3

            for d in dirs:
                mid, land = JUMP_TABLE[sq][d]
                if mid != -1 and land != -1:
                    if (1 << mid) & bb.wp and not ((1 << land) & occ):
                        sc -= 8
                        break

            if not is_k:
                r = SQ_TO_RC[sq][0]
                clear = True
                for rr in range(r + 1, 8):
                    if (wm | wk) & ROW_MASK[rr]:
                        clear = False; break
                if clear and r >= 4: sc -= (r - 3) * 10

            if is_k:
                r, c = SQ_TO_RC[sq]
                if r in (0, 7) or c in (0, 7):
                    exits = 0
                    for d in range(4):
                        nb = NBR[sq][d]
                        if nb != -1 and not ((1 << nb) & occ): exits += 1
                    if exits <= 1: sc += 8

        if endgame:
            diff = (wn + wkn * 2) - (bn + bkn * 2)
            sc += diff * 15

        res = sc if white_turn else -sc
        self._eval_cache[cache_key] = res
        return res

    def _order(self, moves, is_cap, depth, tt_mv, white):
        scored = []
        for m in moves:
            s = 0
            if tt_mv and m == tt_mv: s += 200000
            elif is_cap: s += 100000 + len(m) * 500
            elif depth < 64 and m in self.killers[depth]: s += 80000
            else:
                key = (m[0], m[-1])
                s += self.hist.get(key, 0)

            end = m[-1]
            if white and (1 << end) & W_PROMO: s += 60000
            if not white and (1 << end) & B_PROMO: s += 60000
            if (1 << end) & CENTER_MASK: s += 100
            scored.append((s, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _quiesce(self, bb, alpha, beta, white, qd=0):
        self.nodes += 1
        if self.nodes & 4095 == 0:
            if time.time() - self.t0 >= self.max_time:
                self.stop = True; return 0

        stand = self.evaluate(bb, white)
        if qd >= 6: return stand
        if stand >= beta: return beta
        if stand > alpha: alpha = stand

        moves, is_cap = bb.get_moves(white)
        if not is_cap: return stand

        for mv in moves:
            child = bb.copy()
            child.do_move(mv, white)
            sc = -self._quiesce(child, -beta, -alpha, not white, qd + 1)
            if self.stop: return 0
            if sc >= beta: return beta
            if sc > alpha: alpha = sc
        return alpha

    def _search(self, bb, depth, alpha, beta, white, pv=True):
        self.nodes += 1
        if self.nodes & 4095 == 0:
            if time.time() - self.t0 >= self.max_time:
                self.stop = True; return 0, None

        w = bb.winner()
        if w is not None:
            if w == 0: return 0, None
            my_win = (w == 1 and white) or (w == 2 and not white)
            return (99999 + depth if my_win else -99999 - depth), None

        if depth <= 0:
            return self._quiesce(bb, alpha, beta, white), None

        key = bb.zobrist(white)
        tr = self.tt.probe(key, depth, alpha, beta)
        if tr and not pv: return tr

        moves, is_cap = bb.get_moves(white)
        if not moves: return -99999 - depth, None

        tt_mv = self.tt.best(key)
        moves = self._order(moves, is_cap, depth, tt_mv, white)

        best = moves[0]
        best_sc = float("-inf")
        oa = alpha
        i = 0

        for mv in moves:
            child = bb.copy()
            child.do_move(mv, white)

            red = 0
            if i >= 4 and depth >= 3 and not is_cap and not pv:
                red = 1 if i < 8 else 2

            if i == 0:
                sc = -self._search(child, depth - 1, -beta, -alpha, not white, True)[0]
            else:
                sc = -self._search(child, depth - 1 - red, -alpha - 1, -alpha, not white, False)[0]
                if alpha < sc < beta and not self.stop:
                    if red > 0:
                        sc = -self._search(child, depth - 1, -alpha - 1, -alpha, not white, False)[0]
                    if alpha < sc < beta and not self.stop:
                        sc = -self._search(child, depth - 1, -beta, -sc, not white, True)[0]

            if self.stop: break
            i += 1

            if sc > best_sc:
                best_sc = sc; best = mv
            if sc > alpha:
                alpha = sc
            if alpha >= beta:
                if not is_cap and depth < 64:
                    k = self.killers[depth]
                    if mv != k[0]: k[1] = k[0]; k[0] = mv
                    hk = (mv[0], mv[-1])
                    self.hist[hk] = self.hist.get(hk, 0) + depth * depth
                break

        if not self.stop:
            fl = UPPER if best_sc <= oa else (LOWER if best_sc >= beta else EXACT)
            self.tt.store(key, depth, best_sc, fl, best)

        return best_sc, best

    def find_best(self, bb, white):
        self.nodes = 0; self.t0 = time.time()
        self.stop = False; self.depth_r = 0
        self.killers = [[None, None] for _ in range(64)]
        self.hist = {}; self._eval_cache = {}

        best_mv = None; best_sc = 0; log = []

        for d in range(1, 50):
            sc, mv = self._search(bb, d, float("-inf"), float("inf"), white, True)
            if self.stop: break
            if mv is not None:
                best_mv = mv; best_sc = sc; self.depth_r = d
                el = time.time() - self.t0
                log.append({"d": d, "sc": round(sc, 1), "n": self.nodes, "t": round(el, 2)})
            if abs(best_sc) > 90000: break
            if time.time() - self.t0 >= self.max_time * 0.75: break

        el = time.time() - self.t0
        return {"move": best_mv, "score": round(best_sc, 1),
                "nodes": self.nodes, "time": round(el, 2),
                "depth": self.depth_r,
                "nps": int(self.nodes / max(el, 0.001)),
                "tt_hits": self.tt.hits, "log": log}

    def analyze_all(self, bb, white):
        self.tt.clear()
        self.nodes = 0; self.t0 = time.time()
        moves, is_cap = bb.get_moves(white)
        if not moves: return None

        each = max(0.3, self.max_time * 0.6 / len(moves))
        results = []

        for mv in moves:
            child = bb.copy()
            child.do_move(mv, white)
            sub = Beast(max_time=each)
            sub.tt = self.tt
            sub._eval_cache = self._eval_cache
            
            # هنا يتبادل الأدوار (الخصم يبحث عن أفضل رد)
            res = sub.find_best(child, not white)
            # النتيجة نعكسها لنحصل على التقييم بالنسبة لنا
            sc = -res["score"]

            is_capt = len(mv) > 2
            if not is_capt and len(mv) == 2:
                sr = SQ_TO_RC[mv[0]][0]
                er = SQ_TO_RC[mv[1]][0]
                is_capt = abs(sr - er) == 2

            cap_n = max(0, len(mv) - 2) if is_capt else 0
            end = mv[-1]
            piece_bit = 1 << mv[0]
            promo = ((white and (1 << end) & W_PROMO) or
                     (not white and (1 << end) & B_PROMO)) and \
                    not (piece_bit & bb.k)

            if is_capt and cap_n >= 2: v = "🔥 أكل متعدد!"
            elif promo: v = "👑 ترقية!"
            elif sc > 300: v = "💪 ساحقة"
            elif sc > 100: v = "✅ ممتازة"
            elif sc > 30: v = "✅ جيدة"
            elif sc > -30: v = "⚖️ متكافئة"
            elif sc > -100: v = "⚠️ حذر"
            else: v = "❌ تجنبها"

            wp = max(1, min(99, int(50 + sc / 10)))
            rc_path = tuple(SQ_TO_RC[s] for s in mv)

            results.append({"move": mv, "rc": rc_path,
                "score": round(sc, 1), "depth": res["depth"],
                "is_capture": is_capt, "captured": cap_n,
                "promotes": promo, "verdict": v, "win_pct": wp})

        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results): r["rank"] = i + 1

        pe = self.evaluate(bb, white)
        return {"moves": results,
                "time": round(time.time() - self.t0, 2),
                "forced": is_cap, "eval": round(pe, 1),
                "tt_hits": self.tt.hits}


# ══════════════════════════════════════════════════════
# 5. مدرب Groq الذكي
# ══════════════════════════════════════════════════════

class DamaCoach:
    URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.ok = bool(api_key and len(api_key) > 10)

    def explain(self, grid, best, all_moves, ev, white):
        if not self.ok:
            return self._basic(best, all_moves, ev, white)
        try:
            return self._llm(grid, best, all_moves, ev, white)
        except Exception as e:
            return self._basic(best, all_moves, ev, white) + \
                   f"\n\n⚠️ Groq: {str(e)[:80]}"

    def _llm(self, grid, best, moves, ev, white):
        color = "الأبيض" if white else "الأسود"
        board_txt = self._grid_text(grid)
        rc = best["rc"]
        path = " → ".join(f"({p[0]},{p[1]})" for p in rc)

        top3 = ""
        for mv in moves[:3]:
            p = " → ".join(f"({x[0]},{x[1]})" for x in mv["rc"])
            top3 += f"  {mv['rank']}. {p} (تقييم: {mv['score']}, {mv['verdict']})\n"

        worst = moves[-1] if len(moves) > 1 else None
        w_txt = ""
        if worst:
            wp = " → ".join(f"({x[0]},{x[1]})" for x in worst["rc"])
            w_txt = f"أسوأ حركة: {wp} ({worst['verdict']})"

        prompt = f"""أنت مدرب داما محترف عربي.

الرقعة (0=فارغ 1=أبيض 2=أسود 3=ملك_أبيض 4=ملك_أسود):
{board_txt}

اللاعب: {color} | التقييم: {ev}

أفضل 3 حركات:
{top3}{w_txt}

اشرح باختصار شديد (4 أسطر فقط):
1. لماذا الحركة الأولى هي الأفضل تكتيكياً؟
2. ما الخطة بعدها؟
3. ما الحركة التي يجب تجنبها ولماذا؟
4. نصيحة ذهبية

استخدم إيموجي وكن حماسياً."""

        data = json.dumps({
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": "مدرب داما عربي محترف. أجب باختصار."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300, "temperature": 0.7
        }).encode("utf-8")

        req = urllib.request.Request(self.URL, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }, method="POST")

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]

    def _basic(self, best, moves, ev, white):
        color = "الأبيض" if white else "الأسود"
        rc = best["rc"]
        path = " → ".join(f"({p[0]},{p[1]})" for p in rc)
        lines = [f"🎯 **أفضل حركة لـ {color}:** `{path}`\n"]

        if best["is_capture"]: lines.append("💥 **السبب:** تأكل قطعة للخصم!")
        elif best["promotes"]: lines.append("👑 **السبب:** ترقية للملك!")
        elif best["score"] > 100: lines.append("💪 **السبب:** تعزز تفوقك")
        elif best["score"] > 0: lines.append("✅ **السبب:** تحسّن وضعيتك")
        else: lines.append("🛡️ **السبب:** أفضل دفاع متاح")

        if ev > 200: lines.append(f"\n📊 أنت متفوق بـ {ev} نقطة!")
        elif ev > 0: lines.append(f"\n📊 أفضل قليلاً ({ev})")
        elif ev > -100: lines.append("\n📊 متكافئة تقريباً")
        else: lines.append("\n⚠️ الخصم أفضل — ركّز!")

        if len(moves) > 1:
            w = moves[-1]
            wp = " → ".join(f"({x[0]},{x[1]})" for x in w["rc"])
            lines.append(f"\n❌ **تجنب:** `{wp}` — {w['verdict']}")

        lines.append("\n💡 " + ("حافظ على الضغط!" if ev > 0 else "ابحث عن فرصة أكل!"))
        return "\n".join(lines)

    def ask(self, grid, question, white):
        if not self.ok: return "⚠️ أضف مفتاح Groq لتفعيل المدرب!"
        color = "الأبيض" if white else "الأسود"
        board_txt = self._grid_text(grid)
        prompt = f"""مدرب داما. الرقعة:\n{board_txt}\nاللاعب: {color}\nسؤال: {question}\nأجب بالعربي باختصار (3 أسطر). استخدم إيموجي."""

        data = json.dumps({
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": "مدرب داما عربي. أجب باختصار."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200, "temperature": 0.7
        }).encode("utf-8")

        try:
            req = urllib.request.Request(self.URL, data=data, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"⚠️ خطأ: {str(e)[:100]}"

    @staticmethod
    def _grid_text(grid):
        syms = {0: "·", 1: "W", 2: "B", 3: "WK", 4: "BK"}
        lines = []
        for r in range(8):
            row = f"{r}|"
            for c in range(8):
                if (r + c) % 2 != 0: row += f" {syms.get(int(grid[r][c]), '·'):>2}"
                else: row += "   "
            lines.append(row)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════
# 6. الرؤية الحاسوبية
# ══════════════════════════════════════════════════════

class Vision:
    @staticmethod
    def process(img_bgr, lt=160, dt=100):
        fixed, was_fixed = Vision._fix_perspective(img_bgr)
        fixed_pil = Image.fromarray(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))
        board_hsv, hsv_info = Vision._detect_hsv(fixed, lt, dt)
        board_circles, circles_vis = Vision._detect_circles(fixed, lt, dt)

        merged = np.zeros((8, 8), dtype=np.int8)
        for r in range(8):
            for c in range(8):
                h, ci = int(board_hsv[r][c]), int(board_circles[r][c])
                if h != 0 and ci != 0 and h == ci: merged[r][c] = h
                elif ci != 0: merged[r][c] = ci
                elif h != 0: merged[r][c] = h

        return {"merged": merged, "hsv_board": board_hsv, "circle_board": board_circles,
                "fixed_pil": fixed_pil, "circles_vis": circles_vis, "was_fixed": was_fixed, "hsv_info": hsv_info}

    @staticmethod
    def _fix_perspective(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, k, iterations=2)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts: return cv2.resize(img, (400, 400)), False
        big = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(big) < img.shape[0] * img.shape[1] * 0.15:
            return cv2.resize(img, (400, 400)), False

        peri = cv2.arcLength(big, True)
        approx = cv2.approxPolyDP(big, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
            o = np.zeros((4, 2), dtype=np.float32)
            o[0] = pts[np.argmin(s)]; o[2] = pts[np.argmax(s)]
            o[1] = pts[np.argmin(d)]; o[3] = pts[np.argmax(d)]
            dst = np.float32([[0, 0], [399, 0], [399, 399], [0, 399]])
            M = cv2.getPerspectiveTransform(o, dst)
            return cv2.warpPerspective(img, M, (400, 400)), True
        x, y, w, h = cv2.boundingRect(big)
        return cv2.resize(img[y:y + h, x:x + w], (400, 400)), False

    @staticmethod
    def _detect_hsv(img, lt, dt):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = 50; board = np.zeros((8, 8), dtype=np.int8); info = []
        for r in range(8):
            for col in range(8):
                if (r + col) % 2 == 0: continue
                m = c // 4
                roi_g = gray[r * c + m:(r + 1) * c - m, col * c + m:(col + 1) * c - m]
                roi_h = hsv[r * c + m:(r + 1) * c - m, col * c + m:(col + 1) * c - m]
                if roi_g.size == 0: continue
                br = float(np.mean(roi_g)); sa = float(np.mean(roi_h[:, :, 1])); va = float(np.var(roi_g))
                det = 0
                if va > 100 or sa > 25:
                    if br > lt: det = 1
                    elif br < dt: det = 2
                board[r][col] = det
                info.append({"r": r, "c": col, "br": round(br), "sa": round(sa), "va": round(va), "d": det})
        return board, info

    @staticmethod
    def _detect_circles(img, lt, dt):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        c = 50
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 35, param1=60, param2=35, minRadius=12, maxRadius=24)
        board = np.zeros((8, 8), dtype=np.int8); vis = img.copy()

        if circles is not None:
            for cx, cy, rad in np.uint16(np.around(circles))[0]:
                co, ro = int(cx / c), int(cy / c)
                if not (0 <= ro < 8 and 0 <= co < 8): continue
                if (ro + co) % 2 == 0: continue
                s = gray[max(0, int(cy) - 5):min(400, int(cy) + 5), max(0, int(cx) - 5):min(400, int(cx) + 5)]
                if s.size == 0: continue
                a = float(np.mean(s))
                if a > lt: board[ro][co] = 1; clr = (0, 255, 0)
                elif a < dt: board[ro][co] = 2; clr = (0, 0, 255)
                else: continue
                cv2.circle(vis, (int(cx), int(cy)), int(rad), clr, 2)
        vis_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        return board, vis_pil


# ══════════════════════════════════════════════════════
# 7. رسم الرقعة
# ══════════════════════════════════════════════════════

class Ren:
    @staticmethod
    def draw(grid, arrows=None, hl=None):
        img = Image.new("RGB", (BPX, BPX)); dr = ImageDraw.Draw(img)
        for r in range(8):
            for c in range(8):
                x1, y1 = c * CELL, r * CELL; x2, y2 = x1 + CELL, y1 + CELL
                sq = (235, 215, 180) if (r + c) % 2 == 0 else (175, 130, 95)
                if hl and (r, c) in hl: sq = (100, 200, 100)
                dr.rectangle([x1, y1, x2, y2], fill=sq)
                p = int(grid[r][c])
                if p == 0: continue
                cx, cy = x1 + CELL // 2, y1 + CELL // 2; pr = CELL // 2 - 10
                dr.ellipse([cx - pr + 3, cy - pr + 3, cx + pr + 3, cy + pr + 3], fill=(70, 50, 30))
                fl = (250, 248, 240) if p in (1, 3) else (45, 45, 45)
                ed = (195, 185, 170) if p in (1, 3) else (25, 25, 25)
                dr.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=fl, outline=ed, width=2)
                dr.ellipse([cx - pr + 5, cy - pr + 5, cx + pr - 5, cy + pr - 5], outline=ed, width=1)
                if p in (3, 4):
                    dr.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(255, 215, 0), outline=(200, 170, 0), width=2)
        for i in range(8):
            try:
                dr.text((3, i * CELL + 3), str(i), fill=(130, 110, 90))
                dr.text((i * CELL + CELL // 2 - 4, BPX - 14), chr(65 + i), fill=(130, 110, 90))
            except Exception: pass
        if arrows:
            for a in arrows: Ren._arrow(dr, a["m"], a.get("c", (255, 50, 50)), a.get("w", 5))
        return img

    @staticmethod
    def _arrow(d, mv, color, w):
        if not mv or len(mv) < 2: return
        for i in range(len(mv) - 1):
            s, e = mv[i], mv[i + 1]
            sx, sy = s[1] * CELL + CELL // 2, s[0] * CELL + CELL // 2
            ex, ey = e[1] * CELL + CELL // 2, e[0] * CELL + CELL // 2
            d.line([(sx, sy), (ex, ey)], fill=color, width=w)
            d.ellipse([ex - 8, ey - 8, ex + 8, ey + 8], fill=color)
        s = mv[0]; sx, sy = s[1] * CELL + CELL // 2, s[0] * CELL + CELL // 2
        d.ellipse([sx - 12, sy - 12, sx + 12, sy + 12], outline=(0, 220, 0), width=4)


# ══════════════════════════════════════════════════════
# 8. واجهة Streamlit
# ══════════════════════════════════════════════════════

def app():
    st.set_page_config("♟️ مساعد الداما الذكي", "♟️", layout="wide")
    st.markdown("""<style>
    .block-container{max-width:1100px}
    .best{background:linear-gradient(135deg,#28a745,#20c997);color:#fff;
        padding:18px;border-radius:12px;text-align:center;margin:10px 0}
    .best h2{margin:0;font-size:1.5em}.best p{margin:5px 0}
    .warn{background:#f8d7da;border:2px solid #dc3545;color:#721c24;
        padding:12px;border-radius:10px;text-align:center;margin:8px 0;font-weight:bold}
    .ebar{background:#e9ecef;border-radius:8px;overflow:hidden;height:30px;margin:8px 0}
    .efill{height:100%;text-align:center;color:#fff;font-weight:bold;
        line-height:30px;border-radius:8px}
    .coach{background:#f8f9fa;padding:16px;border-radius:12px;
        border-left:4px solid #667eea;margin:10px 0;line-height:1.8}
    .perf{background:#f0f2f6;padding:10px;border-radius:10px;
        border-left:4px solid #28a745;margin:8px 0;font-size:0.9em}
    .qbox{background:#e8f5e9;padding:14px;border-radius:10px;margin:8px 0;line-height:1.8}
    </style>""", unsafe_allow_html=True)

    if "board" not in st.session_state: st.session_state.board = BB().to_grid().tolist()
    if "coach" not in st.session_state: st.session_state.coach = DamaCoach()

    st.title("♟️ مساعد الداما الذكي (Brutal AI - مصحح)")
    st.caption("Bitboard Engine + XGBoost Model + Groq AI Coach")

    if AI_MODEL:
        st.success("🧠 نموذج الذكاء الاصطناعي (XGBoost) محمل وفعال (تم إصلاح تقييم NegaMax)!")
    else:
        st.warning("⚠️ نموذج الذكاء الاصطناعي مفقود! سيتم استخدام التقييم اليدوي.")

    with st.sidebar:
        st.header("⚙️ الإعدادات")
        mc = st.radio("♟️ لونك:", ["⚪ أبيض", "⚫ أسود"])
        fw = "أبيض" in mc
        tt = st.select_slider("⏱ وقت التحليل:", [1, 2, 3, 5, 8, 10, 15], value=3)

        st.divider()
        st.markdown("### 🤖 المدرب الذكي (Groq)")
        api_key = st.text_input("🔑 API Key:", type="password", placeholder="gsk_...")
        if api_key: st.session_state.coach = DamaCoach(api_key); st.success("✅ المدرب مفعّل!")
        else: st.session_state.coach = DamaCoach()

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 جديدة", use_container_width=True):
                st.session_state.board = BB().to_grid().tolist(); st.rerun()
        with c2:
            if st.button("🗑️ مسح", use_container_width=True):
                st.session_state.board = np.zeros((8, 8), dtype=int).tolist(); st.rerun()

        st.divider()
        g_ = np.array(st.session_state.board)
        wn = int(np.sum((g_ == 1) | (g_ == 3)))
        bn = int(np.sum((g_ == 2) | (g_ == 4)))
        mc1, mc2 = st.columns(2)
        with mc1: st.metric("⚪", wn)
        with mc2: st.metric("⚫", bn)

    tab_names = ["✏️ إدخال يدوي"]
    if HAS_CV2: tab_names.append("📷 تحليل صورة")
    tab_names.append("🧠 التحليل")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        opts = {"⬜ فارغ": 0, "⚪ أبيض": 1, "⚫ أسود": 2, "👑W ملك أبيض": 3, "♛B ملك أسود": 4}
        sel = st.radio("_", list(opts.keys()), horizontal=True, label_visibility="collapsed")
        sv = opts[sel]
        syms = {0: "·", 1: "⚪", 2: "⚫", 3: "👑", 4: "♛"}
        ba = np.array(st.session_state.board)
        for r in range(8):
            cols = st.columns(8)
            for c in range(8):
                with cols[c]:
                    ok = (r + c) % 2 != 0
                    s = syms.get(int(ba[r][c]), "·") if ok else ""
                    if st.button(s, key=f"m{r}{c}", use_container_width=True, disabled=not ok):
                        st.session_state.board[r][c] = sv; st.rerun()
        st.image(Ren.draw(ba), caption="🎨 الرقعة الحالية", use_container_width=True)

    if HAS_CV2:
        with tabs[1]:
            up = st.file_uploader("📸 ارفع صورة الرقعة", type=["jpg", "png", "jpeg"])
            if up:
                pil = Image.open(up).convert("RGB")
                img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                c1, c2 = st.columns(2)
                with c1: st.image(pil, caption="📸 الأصلية", use_container_width=True)
                tc1, tc2 = st.columns(2)
                with tc1: lt = st.slider("عتبة الفاتح (سطوع)", 100, 230, 160)
                with tc2: dt = st.slider("عتبة الداكن (سطوع)", 30, 150, 100)
                if st.button("🔍 تحليل الصورة", type="primary"):
                    with st.spinner("🧠 جاري التحليل..."):
                        result = Vision.process(img_cv, lt, dt)
                    with c2: st.image(result["fixed_pil"], use_container_width=True)
                    st.success("✅ اكتمل التحليل!")
                    if st.button("📥 استيراد للرقعة", type="primary"):
                        st.session_state.board = result["merged"].tolist()
                        st.rerun()

    ai_idx = 2 if HAS_CV2 else 1
    with tabs[ai_idx]:
        ba2 = np.array(st.session_state.board, dtype=np.int8)
        bb = BB.from_grid(ba2)
        st.image(Ren.draw(ba2), caption="🎨 الرقعة الحالية", use_container_width=True)

        w = bb.full_winner()
        if w is not None:
            if w == 0: st.info("🤝 تعادل!")
            elif w == 1: st.success("🏆 الأبيض فائز!")
            else: st.success("🏆 الأسود فائز!")
            return

        emoji = "⚪" if fw else "⚫"
        name = "الأبيض" if fw else "الأسود"
        st.markdown(f"### {emoji} تحليل حركات **{name}**")

        if st.button("🧠 حلّل الآن!", type="primary", use_container_width=True):
            prg = st.empty(); prg.info(f"🧠 المحرك يحلل...")
            ai = Beast(max_time=tt)
            analysis = ai.analyze_all(bb, fw)
            prg.empty()

            if not analysis or not analysis["moves"]:
                st.error("❌ لا توجد حركات متاحة!"); return

            mvs = analysis["moves"]; best = mvs[0]

            pe = analysis["eval"]
            if pe > 300: em, ec = "🟢 متفوق جداً!", "#28a745"
            elif pe > 100: em, ec = "🟢 أفضل", "#20c997"
            elif pe > 30: em, ec = "🟢 أفضل قليلاً", "#17a2b8"
            elif pe > -30: em, ec = "🟡 متكافئة", "#ffc107"
            elif pe > -100: em, ec = "🟠 الخصم أفضل", "#fd7e14"
            else: em, ec = "🔴 خطر!", "#dc3545"
            pct = max(5, min(95, int(50 + pe / 10)))
            st.markdown(f'<div class="ebar"><div class="efill" style="width:{pct}%;background:{ec}">{em} ({pe})</div></div>', unsafe_allow_html=True)

            if analysis["forced"]:
                st.markdown('<div class="warn">⚡ أكل إجباري! يجب عليك الأكل</div>', unsafe_allow_html=True)

            rc = best["rc"]; path = " → ".join(f"({p[0]},{p[1]})" for p in rc)
            ex = []
            if best["is_capture"]: ex.append(f"💥 أكل ×{best['captured']}")
            if best["promotes"]: ex.append("👑 ترقية!")
            ex.append(best["verdict"])
            ext = " • ".join(ex)

            st.markdown(f"""<div class="best"><h2>🏆 أفضل حركة</h2><p style="font-size:1.4em">{path}</p><p>{ext}</p>
                <p>تقييم: {best['score']} • فوز: {best['win_pct']}% • عمق: {best['depth']}</p></div>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                bi = Ren.draw(ba2, arrows=[{"m": list(rc), "c": (50, 205, 50), "w": 6}], hl=set(rc))
                st.image(bi, caption="🏆 أفضل حركة", use_container_width=True)
            with col2:
                af = bb.copy(); af.do_move(best["move"], fw)
                st.image(Ren.draw(af.to_grid()), caption="📋 الرقعة بعد التنفيذ", use_container_width=True)

            if st.button("✅ طبّق هذه الحركة على الرقعة", use_container_width=True):
                st.session_state.board = af.to_grid().tolist(); st.rerun()

            st.markdown("### 🗣️ تحليل المدرب")
            coach = st.session_state.coach
            with st.spinner("🧠 المدرب يحلل..."):
                explanation = coach.explain(ba2, best, mvs, analysis["eval"], fw)
            st.markdown(f'<div class="coach">{explanation}</div>', unsafe_allow_html=True)

            if coach.ok:
                st.markdown("### 💬 اسأل المدرب")
                question = st.text_input("اكتب سؤالك عن الوضعية:")
                if question:
                    with st.spinner("🗣️ المدرب يجيب..."):
                        answer = coach.ask(ba2, question, fw)
                    st.markdown(f'<div class="qbox">🎓 {answer}</div>', unsafe_allow_html=True)

            st.markdown(f"""<div class="perf">⚡ <b>Bitboard + XGBoost Engine:</b> عمق {best['depth']} • {analysis.get('tt_hits', 0):,} TT hits • الوقت: {analysis['time']}s</div>""", unsafe_allow_html=True)

            st.markdown(f"### 📊 تصنيف كل الحركات ({len(mvs)})")
            rc_colors = {1: ("🥇", (50, 205, 50)), 2: ("🥈", (65, 105, 225)), 3: ("🥉", (255, 165, 0))}
            top_arrows = []
            for i, mv in enumerate(mvs[:5]):
                _, cl = rc_colors.get(mv["rank"], (f"#{mv['rank']}", (180, 180, 180)))
                top_arrows.append({"m": list(mv["rc"]), "c": cl, "w": 6 if i == 0 else 3})
            st.image(Ren.draw(ba2, arrows=top_arrows), caption="أفضل 5 حركات", use_container_width=True)

            for mv in mvs:
                icon = rc_colors.get(mv["rank"], (f"#{mv['rank']}", None))[0]
                p = " → ".join(f"({x[0]},{x[1]})" for x in mv["rc"])
                with st.expander(f"{icon} {p} • {mv['score']} • {mv['verdict']}"):
                    mi = Ren.draw(ba2, arrows=[{"m": list(mv["rc"]), "c": (255, 100, 50), "w": 5}], hl=set(mv["rc"]))
                    st.image(mi, use_container_width=True)

if __name__ == "__main__":
    app()
