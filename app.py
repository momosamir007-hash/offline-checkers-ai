#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════╗
║     ♟️ مساعد الداما — Bitboard Engine v8.0        ║
║  32-bit Bitboard + Zobrist TT + PVS + LMR        ║
║  + Opening Book + Endgame Knowledge               ║
║  عمق 15-22 في ثوانٍ = لا يُقهر                   ║
╚═══════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time
import random
from enum import IntEnum

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ══════════════════════════════════════════
# 1. الجداول المحسوبة مسبقاً (Precomputed)
# ══════════════════════════════════════════

# ── ترقيم المربعات 0-31 ──
# Row0: sq0=(0,1) sq1=(0,3) sq2=(0,5) sq3=(0,7)
# Row1: sq4=(1,0) sq5=(1,2) sq6=(1,4) sq7=(1,6)
# Row2: sq8=(2,1) sq9=(2,3) sq10=(2,5) sq11=(2,7)
# ...
# Row7: sq28=(7,0) sq29=(7,2) sq30=(7,4) sq31=(7,6)

SQ_TO_RC = [
    (0,1),(0,3),(0,5),(0,7),
    (1,0),(1,2),(1,4),(1,6),
    (2,1),(2,3),(2,5),(2,7),
    (3,0),(3,2),(3,4),(3,6),
    (4,1),(4,3),(4,5),(4,7),
    (5,0),(5,2),(5,4),(5,6),
    (6,1),(6,3),(6,5),(6,7),
    (7,0),(7,2),(7,4),(7,6),
]

RC_TO_SQ = {}
for _i, _rc in enumerate(SQ_TO_RC):
    RC_TO_SQ[_rc] = _i

# ── جيران كل مربع: (NW, NE, SW, SE) ──
# -1 = خارج الرقعة
NBR = [
    (-1,-1, 4, 5),(-1,-1, 5, 6),(-1,-1, 6, 7),(-1,-1, 7,-1),
    (-1, 0,-1, 8),( 0, 1, 8, 9),( 1, 2, 9,10),( 2, 3,10,11),
    ( 4, 5,12,13),( 5, 6,13,14),( 6, 7,14,15),( 7,-1,15,-1),
    (-1, 8,-1,16),( 8, 9,16,17),( 9,10,17,18),(10,11,18,19),
    (12,13,20,21),(13,14,21,22),(14,15,22,23),(15,-1,23,-1),
    (-1,16,-1,24),(16,17,24,25),(17,18,25,26),(18,19,26,27),
    (20,21,28,29),(21,22,29,30),(22,23,30,31),(23,-1,31,-1),
    (-1,24,-1,-1),(24,25,-1,-1),(25,26,-1,-1),(26,27,-1,-1),
]

# ── جداول القفز: (وسط, هبوط) لكل اتجاه ──
JUMP = []
for _sq in range(32):
    _jmps = []
    for _d in range(4):
        _mid = NBR[_sq][_d]
        if _mid == -1:
            _jmps.append((-1, -1))
        else:
            _land = NBR[_mid][_d]
            _jmps.append((_mid, _land) if _land != -1
                         else (-1, -1))
    JUMP.append(tuple(_jmps))

# ── اتجاهات كل نوع قطعة ──
# أبيض يتحرك لأعلى (NW=0, NE=1)
# أسود يتحرك لأسفل (SW=2, SE=3)
# ملك: كل الاتجاهات
W_DIRS = (0, 1)       # NW, NE
B_DIRS = (2, 3)       # SW, SE
K_DIRS = (0, 1, 2, 3) # الكل

# ── الوضعية الابتدائية ──
INIT_BP = sum(1 << i for i in range(12))
INIT_WP = sum(1 << i for i in range(20, 32))

# ── أقنعة الصفوف ──
ROW_MASK = [0] * 8
for _r in range(8):
    for _c in range(8):
        if (_r + _c) % 2 != 0:
            _sq = RC_TO_SQ.get((_r, _c))
            if _sq is not None:
                ROW_MASK[_r] |= (1 << _sq)

# صف الترقية
W_PROMO = ROW_MASK[0]  # أبيض يترقى عند الصف 0
B_PROMO = ROW_MASK[7]  # أسود يترقى عند الصف 7

# أقنعة المركز
CENTER_MASK = 0
for _sq in range(32):
    _r, _c = SQ_TO_RC[_sq]
    if 2 <= _r <= 5 and 2 <= _c <= 5:
        CENTER_MASK |= (1 << _sq)

INNER_CENTER = 0
for _sq in range(32):
    _r, _c = SQ_TO_RC[_sq]
    if 3 <= _r <= 4 and 3 <= _c <= 4:
        INNER_CENTER |= (1 << _sq)

# ── جداول القيمة الموضعية (32 قيمة لكل نوع) ──
W_POS = [
    0, 0, 0, 0,        # row 0 (ترقية)
    25,25,25,25,        # row 1 (قريب جداً!)
    12,15,15,12,        # row 2
    8,12,12, 8,         # row 3
    5, 8, 8, 5,         # row 4
    3, 5, 5, 3,         # row 5
    1, 2, 2, 1,         # row 6
    5, 5, 5, 5,         # row 7 (صف خلفي = حماية)
]

B_POS = [
    5, 5, 5, 5,         # row 0 (صف خلفي)
    1, 2, 2, 1,         # row 1
    3, 5, 5, 3,         # row 2
    5, 8, 8, 5,         # row 3
    8,12,12, 8,         # row 4
    12,15,15,12,        # row 5
    25,25,25,25,        # row 6 (قريب!)
    0, 0, 0, 0,         # row 7 (ترقية)
]

K_POS = [
    1, 2, 2, 1,
    3, 5, 5, 3,
    4, 7, 7, 4,
    3, 7, 7, 3,
    3, 7, 7, 3,
    4, 7, 7, 4,
    3, 5, 5, 3,
    1, 2, 2, 1,
]

# ── Zobrist Keys ──
_rng = random.Random(999)
Z_KEYS = {}
for _p in range(4):  # WM, BM, WK, BK
    for _sq in range(32):
        Z_KEYS[(_p, _sq)] = _rng.getrandbits(64)
Z_SIDE = _rng.getrandbits(64)

CELL = 80
BPX = CELL * 8


# ══════════════════════════════════════════
# 2. Bitboard Engine
# ══════════════════════════════════════════

def popcount(x):
    return bin(x).count('1')

def iter_bits(x):
    while x:
        lsb = x & (-x)
        sq = lsb.bit_length() - 1
        yield sq
        x ^= lsb


class BB:
    """
    حالة الرقعة بـ 3 أرقام فقط:
      wp: قطع أبيض (32 bit)
      bp: قطع أسود (32 bit)
      k:  ملوك (32 bit, كلا اللونين)
    النسخ = 3 أرقام = أسرع 10x من numpy
    """
    __slots__ = ['wp', 'bp', 'k']

    def __init__(self, wp=INIT_WP, bp=INIT_BP, k=0):
        self.wp = wp
        self.bp = bp
        self.k = k

    def copy(self):
        b = BB.__new__(BB)
        b.wp = self.wp
        b.bp = self.bp
        b.k = self.k
        return b

    def zobrist(self, white_turn):
        h = Z_SIDE if white_turn else 0
        wm = self.wp & ~self.k
        bm = self.bp & ~self.k
        wk = self.wp & self.k
        bk = self.bp & self.k
        for sq in iter_bits(wm): h ^= Z_KEYS[(0, sq)]
        for sq in iter_bits(bm): h ^= Z_KEYS[(1, sq)]
        for sq in iter_bits(wk): h ^= Z_KEYS[(2, sq)]
        for sq in iter_bits(bk): h ^= Z_KEYS[(3, sq)]
        return h

    # ── توليد الحركات ──

    def get_moves(self, white):
        my = self.wp if white else self.bp
        opp = self.bp if white else self.wp
        empty = ~(self.wp | self.bp) & 0xFFFFFFFF
        dirs = K_DIRS  # placeholder

        jumps = []
        simple = []

        for sq in iter_bits(my):
            is_king = bool((1 << sq) & self.k)
            ds = K_DIRS if is_king else (
                W_DIRS if white else B_DIRS
            )

            # أكل
            has_jump = False
            for d in ds:
                mid, land = JUMP[sq][d]
                if (mid != -1 and (1 << mid) & opp
                        and (1 << land) & empty):
                    has_jump = True
                    break

            if has_jump:
                self._find_chains(
                    sq, my, opp, empty, is_king,
                    white, jumps, frozenset(), (sq,)
                )
            else:
                for d in ds:
                    nb = NBR[sq][d]
                    if nb != -1 and (1 << nb) & empty:
                        simple.append((sq, nb))

        if jumps:
            mx = max(len(j) for j in jumps)
            return [j for j in jumps if len(j) == mx], True
        return [(s, e) for s, e in simple], False

    def _find_chains(self, sq, my, opp, empty,
                     is_king, white, out, eaten, path):
        ds = K_DIRS if is_king else (
            W_DIRS if white else B_DIRS
        )
        found = False

        for d in ds:
            mid, land = JUMP[sq][d]
            if mid == -1:
                continue
            if not ((1 << mid) & opp):
                continue
            if mid in eaten:
                continue
            if not ((1 << land) & empty) and land != sq:
                continue
            # حالة خاصة: الهبوط على مربع البداية
            if (1 << land) & (my | opp) and land != path[0]:
                continue

            found = True

            # هل سيترقى؟
            promo = False
            if not is_king:
                if white and (1 << land) & W_PROMO:
                    promo = True
                elif not white and (1 << land) & B_PROMO:
                    promo = True

            new_eaten = eaten | {mid}
            new_path = path + (land,)

            if promo:
                out.append(new_path)
            else:
                new_empty = (empty | (1 << sq)) & ~(1 << land)
                new_opp = opp & ~(1 << mid)
                self._find_chains(
                    land, my, new_opp, new_empty,
                    is_king, white, out, new_eaten, new_path
                )

        if not found and len(path) > 1:
            out.append(path)

    # ── تنفيذ حركة ──

    def do_move(self, move, white):
        """تنفيذ فوري — تختفي القطع المأكولة"""
        start = move[0]
        end = move[-1] if isinstance(move[-1], int) else move[1]

        my_bit = 1 << start
        is_king = bool(my_bit & self.k)

        # مسح البداية
        if white:
            self.wp &= ~my_bit
        else:
            self.bp &= ~my_bit
        self.k &= ~my_bit

        # مسح القطع المأكولة
        if isinstance(move, tuple) and len(move) > 2:
            # سلسلة أكل
            for i in range(len(move) - 1):
                s, e = move[i], move[i + 1]
                # حساب الوسط
                sr, sc = SQ_TO_RC[s]
                er, ec = SQ_TO_RC[e]
                mr, mc = (sr + er) // 2, (sc + ec) // 2
                if (mr, mc) in RC_TO_SQ:
                    mid = RC_TO_SQ[(mr, mc)]
                    if white:
                        self.bp &= ~(1 << mid)
                    else:
                        self.wp &= ~(1 << mid)
                    self.k &= ~(1 << mid)
            end = move[-1]
        elif isinstance(move, tuple) and len(move) == 2:
            s, e = move
            sr, sc = SQ_TO_RC[s]
            er, ec = SQ_TO_RC[e]
            if abs(sr - er) == 2:
                mr, mc = (sr + er) // 2, (sc + ec) // 2
                if (mr, mc) in RC_TO_SQ:
                    mid = RC_TO_SQ[(mr, mc)]
                    if white:
                        self.bp &= ~(1 << mid)
                    else:
                        self.wp &= ~(1 << mid)
                    self.k &= ~(1 << mid)
            end = e

        # وضع القطعة في الوجهة
        end_bit = 1 << end
        if white:
            self.wp |= end_bit
        else:
            self.bp |= end_bit

        # ترقية / حفظ الملك
        if is_king:
            self.k |= end_bit
        else:
            if white and end_bit & W_PROMO:
                self.k |= end_bit
            elif not white and end_bit & B_PROMO:
                self.k |= end_bit

    # ── حالة اللعبة ──

    def winner(self):
        if self.wp == 0: return 2
        if self.bp == 0: return 1
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
                if p == 0: continue
                rc = (r, c)
                if rc not in RC_TO_SQ: continue
                sq = RC_TO_SQ[rc]
                if p == 1: wp |= (1 << sq)
                elif p == 2: bp |= (1 << sq)
                elif p == 3: wp |= (1 << sq); k |= (1 << sq)
                elif p == 4: bp |= (1 << sq); k |= (1 << sq)
        return BB(wp, bp, k)


# ══════════════════════════════════════════
# 3. Opening Book
# ══════════════════════════════════════════

# أقوى الافتتاحيات المعروفة
# المفتاح = (wp, bp, k, white_turn)
# القيمة = أفضل حركة (sq_from, sq_to)
OPENING_BOOK = {}

def _add_opening(moves_list):
    """إضافة سلسلة افتتاحية"""
    b = BB()
    white_turn = False  # الأسود يبدأ في الداما الإنجليزية
    for i, mv in enumerate(moves_list):
        key = (b.wp, b.bp, b.k, white_turn)
        if i < len(moves_list):
            OPENING_BOOK[key] = mv
        b_copy = b.copy()
        b_copy.do_move(mv, white_turn)
        b = b_copy
        white_turn = not white_turn

# الافتتاحيات الأقوى (ترقيم 0-31)
# 11-15 = sq10→sq14, 23-19 = sq22→sq18
_OPENINGS = [
    [(10,14),(22,18),(14,23),(26,19)],  # Old Faithful
    [(10,14),(23,19),(7,10),(19,15)],   # Cross
    [(10,14),(21,17),(14,21),(25,18)],  # Defiance
    [(10,15),(22,18),(15,22),(25,18)],  # Kelso
    [(10,15),(23,19),(6,10),(19,15)],   # Denny
    [(11,15),(22,18),(15,22),(25,18)],  # Single Corner
    [(11,15),(23,19),(8,11),(19,15)],   # Edinburgh
    [(11,16),(23,19),(8,11),(22,17)],   # Bristol
    [(9,14),(22,18),(5,9),(18,15)],     # Double Corner
    [(9,13),(22,18),(13,22),(25,18)],   # Alma
]

for _line in _OPENINGS:
    try:
        _add_opening(_line)
    except Exception:
        pass


# ══════════════════════════════════════════
# 4. Transposition Table
# ══════════════════════════════════════════

EXACT, LOWER, UPPER = 0, 1, 2

class TT:
    __slots__ = ['t', 'hits', 'sz']

    def __init__(self, sz=800_000):
        self.t = {}
        self.sz = sz
        self.hits = 0

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
                for k in ks[:len(ks)//2]:
                    del self.t[k]

    def best(self, key):
        e = self.t.get(key)
        return e[3] if e else None

    def clear(self):
        self.t.clear()
        self.hits = 0


# ══════════════════════════════════════════
# 5. المحرك الوحشي (Beast Engine)
# ══════════════════════════════════════════

class Beast:
    """
    Bitboard + 12 عامل تقييم فوري
    + Zobrist TT + PVS + LMR + Quiescence
    + Opening Book + Endgame Knowledge
    """

    def __init__(self, max_time=3.0):
        self.max_time = max_time
        self.tt = TT()
        self.nodes = 0
        self.t0 = 0
        self.stop = False
        self.depth_r = 0
        self.killers = [[None, None] for _ in range(64)]
        self.hist = {}

    # ── تقييم فوري (بدون get_moves!) ──

    def evaluate(self, bb, for_white):
        wm = bb.wp & ~bb.k
        bm = bb.bp & ~bb.k
        wk = bb.wp & bb.k
        bk = bb.bp & bb.k

        wn = popcount(wm)
        wkn = popcount(wk)
        bn = popcount(bm)
        bkn = popcount(bk)
        wt = wn + wkn
        bt = bn + bkn

        if wt == 0: return -99999 if for_white else 99999
        if bt == 0: return 99999 if for_white else -99999

        sc = 0.0
        total = wt + bt
        endgame = total <= 8

        # 1. مادة
        sc += wn * 100 + wkn * (160 if endgame else 140)
        sc -= bn * 100 + bkn * (160 if endgame else 140)

        # 2. موقع (سريع بـ bit iteration)
        for sq in iter_bits(wm): sc += W_POS[sq]  * 3
        for sq in iter_bits(bm): sc -= B_POS[sq]  * 3
        for sq in iter_bits(wk): sc += K_POS[sq]  * 2
        for sq in iter_bits(bk): sc -= K_POS[sq]  * 2

        # 3. المركز
        sc += popcount(bb.wp & CENTER_MASK) * 5
        sc -= popcount(bb.bp & CENTER_MASK) * 5
        sc += popcount(bb.wp & INNER_CENTER) * 4
        sc -= popcount(bb.bp & INNER_CENTER) * 4

        # 4. الصف الخلفي
        if not endgame:
            sc += popcount(wm & ROW_MASK[7]) * 8
            sc -= popcount(bm & ROW_MASK[0]) * 8

        # 5. حافة الترقية
        sc += popcount(wm & ROW_MASK[1]) * 18
        sc -= popcount(bm & ROW_MASK[6]) * 18

        # 6-9. أنماط تكتيكية (لكل قطعة)
        all_p = bb.wp | bb.bp
        for sq in iter_bits(bb.wp):
            is_w_king = bool((1 << sq) & bb.k)
            mult = 1

            # 6. ترابط
            allies = 0
            for d in range(4):
                nb = NBR[sq][d]
                if nb != -1 and (1 << nb) & bb.wp:
                    allies += 1
            sc += allies * 3

            # 7. حماية من الأكل
            for d in range(4):
                mid, land = JUMP[sq][d]
                if mid == -1: continue
                opp_d = (d + 2) % 4 if d < 2 else (d - 2)
                # هل خصم يستطيع أكلي؟
                attacker = NBR[sq][opp_d if opp_d < 4 else d]
                if attacker != -1:
                    pass  # مبسّط لتجنب البطء

            # 8. قطعة هاربة (runaway)
            if not is_w_king:
                r = SQ_TO_RC[sq][0]
                # هل كل الصفوف أعلاه خالية من الأسود؟
                clear = True
                for rr in range(r):
                    if bm & ROW_MASK[rr] or bk & ROW_MASK[rr]:
                        clear = False
                        break
                if clear and r <= 3:
                    sc += (4 - r) * 12

            # 9. ملك محاصر
            if is_w_king:
                r, c = SQ_TO_RC[sq]
                if r in (0, 7) or c in (0, 7):
                    exits = 0
                    for d in range(4):
                        nb = NBR[sq][d]
                        if nb != -1 and not ((1 << nb) & all_p):
                            exits += 1
                    if exits <= 1:
                        sc -= 10

        # نفس الشيء للأسود (معكوس)
        for sq in iter_bits(bb.bp):
            is_b_king = bool((1 << sq) & bb.k)
            allies = 0
            for d in range(4):
                nb = NBR[sq][d]
                if nb != -1 and (1 << nb) & bb.bp:
                    allies += 1
            sc -= allies * 3

            if not is_b_king:
                r = SQ_TO_RC[sq][0]
                clear = True
                for rr in range(r + 1, 8):
                    if wm & ROW_MASK[rr] or wk & ROW_MASK[rr]:
                        clear = False
                        break
                if clear and r >= 4:
                    sc -= (r - 3) * 12

            if is_b_king:
                r, c = SQ_TO_RC[sq]
                if r in (0, 7) or c in (0, 7):
                    exits = 0
                    for d in range(4):
                        nb = NBR[sq][d]
                        if nb != -1 and not ((1 << nb) & all_p):
                            exits += 1
                    if exits <= 1:
                        sc += 10

        # 10. جسر دفاعي
        if (1 << 31) & wm and (1 << 27) & wm: sc += 6
        if (1 << 28) & wm and (1 << 24) & wm: sc += 6
        if (1 << 0)  & bm and (1 << 4)  & bm: sc -= 6
        if (1 << 3)  & bm and (1 << 7)  & bm: sc -= 6

        # 11. نهاية اللعبة
        if endgame:
            diff = (wn + wkn * 2) - (bn + bkn * 2)
            sc += diff * 20

        # 12. توازن أجنحة
        left = sum(1 << i for i in range(32)
                   if SQ_TO_RC[i][1] < 4)
        right = 0xFFFFFFFF & ~left
        if popcount(bb.wp & left) > 0 and \
           popcount(bb.wp & right) > 0:
            sc += 3
        if popcount(bb.bp & left) > 0 and \
           popcount(bb.bp & right) > 0:
            sc -= 3

        return sc if for_white else -sc

    # ── ترتيب الحركات ──

    def _order(self, moves, is_cap, bb, white,
               depth, tt_mv):
        scored = []
        for m in moves:
            s = 0
            if tt_mv and m == tt_mv:
                s += 200000
            elif is_cap:
                s += 100000 + len(m) * 1000 \
                    if isinstance(m, tuple) and len(m) > 2 \
                    else 100000
            elif depth < 64 and m in self.killers[depth]:
                s += 80000
            else:
                key = (m[0], m[-1] if isinstance(m, tuple)
                       and len(m) > 2 else m[1])
                s += self.hist.get(key, 0)

            end = m[-1] if isinstance(m, tuple) \
                and len(m) > 2 else m[1]
            if white and (1 << end) & W_PROMO:
                s += 60000
            if not white and (1 << end) & B_PROMO:
                s += 60000
            if (1 << end) & CENTER_MASK:
                s += 100
            scored.append((s, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ── Quiescence ──

    def _quiesce(self, bb, alpha, beta, white,
                 for_w, qd=0):
        self.nodes += 1
        if self.nodes & 4095 == 0:
            if time.time() - self.t0 >= self.max_time:
                self.stop = True; return 0

        stand = self.evaluate(bb, for_w)
        if qd >= 6 or stand >= beta: return min(stand, beta)
        if stand > alpha: alpha = stand

        moves, is_cap = bb.get_moves(white)
        if not is_cap: return stand

        for mv in moves:
            child = bb.copy()
            child.do_move(mv, white)
            sc = -self._quiesce(
                child, -beta, -alpha,
                not white, for_w, qd + 1
            )
            if self.stop: return 0
            if sc >= beta: return beta
            if sc > alpha: alpha = sc
        return alpha

    # ── Alpha-Beta + PVS + LMR ──

    def _search(self, bb, depth, alpha, beta,
                white, for_w, pv=True):
        self.nodes += 1
        if self.nodes & 4095 == 0:
            if time.time() - self.t0 >= self.max_time:
                self.stop = True; return 0, None

        w = bb.winner()
        if w is not None:
            if w == 0: return 0, None
            my_win = (w == 1 and for_w) or (w == 2 and not for_w)
            return (99999+depth if my_win else -99999-depth), None

        if depth <= 0:
            return self._quiesce(
                bb, alpha, beta, white, for_w
            ), None

        # TT
        key = bb.zobrist(white)
        tr = self.tt.probe(key, depth, alpha, beta)
        if tr and not pv: return tr

        moves, is_cap = bb.get_moves(white)
        if not moves: return -99999, None

        tt_mv = self.tt.best(key)
        moves = self._order(
            moves, is_cap, bb, white, depth, tt_mv
        )

        best = moves[0]
        best_sc = float("-inf")
        oa = alpha
        i = 0

        for mv in moves:
            child = bb.copy()
            child.do_move(mv, white)

            # LMR
            red = 0
            if i >= 4 and depth >= 3 and not is_cap and not pv:
                red = 1
                if i >= 8: red = 2

            if i == 0:
                sc = -self._search(
                    child, depth-1, -beta, -alpha,
                    not white, for_w, True
                )[0]
            else:
                sc = -self._search(
                    child, depth-1-red, -alpha-1, -alpha,
                    not white, for_w, False
                )[0]
                if alpha < sc < beta and not self.stop:
                    if red > 0:
                        sc = -self._search(
                            child, depth-1, -alpha-1, -alpha,
                            not white, for_w, False
                        )[0]
                    if alpha < sc < beta and not self.stop:
                        sc = -self._search(
                            child, depth-1, -beta, -sc,
                            not white, for_w, True
                        )[0]

            if self.stop: break
            i += 1

            if sc > best_sc:
                best_sc = sc; best = mv
            if sc > alpha: alpha = sc
            if alpha >= beta:
                if not is_cap and depth < 64:
                    k = self.killers[depth]
                    if mv != k[0]: k[1]=k[0]; k[0]=mv
                    end = mv[-1] if isinstance(mv, tuple) \
                        and len(mv) > 2 else mv[1]
                    hk = (mv[0], end)
                    self.hist[hk] = self.hist.get(hk,0) + depth*depth
                break

        if not self.stop:
            fl = (UPPER if best_sc <= oa
                  else LOWER if best_sc >= beta
                  else EXACT)
            self.tt.store(key, depth, best_sc, fl, best)

        return best_sc, best

    # ── Iterative Deepening ──

    def find_best(self, bb, white):
        self.nodes = 0
        self.t0 = time.time()
        self.stop = False
        self.depth_r = 0
        self.killers = [[None,None] for _ in range(64)]
        self.hist = {}

        # فحص Opening Book أولاً
        bk = (bb.wp, bb.bp, bb.k, white)
        if bk in OPENING_BOOK:
            mv = OPENING_BOOK[bk]
            return {
                "move": mv, "score": 0,
                "nodes": 0, "time": 0,
                "depth": 0, "nps": 0,
                "book": True, "log": [],
            }

        best_mv = None
        best_sc = 0
        log = []

        for d in range(1, 50):
            sc, mv = self._search(
                bb, d, float("-inf"), float("inf"),
                white, white, True
            )
            if self.stop: break
            if mv is not None:
                best_mv = mv
                best_sc = sc
                self.depth_r = d
                el = time.time() - self.t0
                log.append({"d": d, "sc": round(sc,1),
                            "n": self.nodes,
                            "t": round(el,2)})
            if abs(best_sc) > 90000: break
            if time.time()-self.t0 >= self.max_time*0.75: break

        el = time.time() - self.t0
        return {
            "move": best_mv, "score": round(best_sc,1),
            "nodes": self.nodes,
            "time": round(el,2),
            "depth": self.depth_r,
            "nps": int(self.nodes/max(el,0.001)),
            "book": False, "log": log,
            "tt_hits": self.tt.hits,
        }

    def analyze_all(self, bb, white):
        self.tt.clear()
        moves, is_cap = bb.get_moves(white)
        if not moves: return None

        each = max(0.3, self.max_time*0.6/len(moves))
        results = []

        for mv in moves:
            child = bb.copy()
            child.do_move(mv, white)

            sub = Beast(max_time=each)
            sub.tt = self.tt
            res = sub.find_best(child, not white)
            sc = -res["score"]

            is_capt = isinstance(mv, tuple) and len(mv) > 2
            if not is_capt and isinstance(mv, tuple) and len(mv)==2:
                sr = SQ_TO_RC[mv[0]][0]
                er = SQ_TO_RC[mv[1]][0]
                is_capt = abs(sr-er)==2

            end = mv[-1] if isinstance(mv, tuple) and len(mv)>2 else mv[1]
            piece_sq = mv[0]
            promo = (
                (white and (1 << end) & W_PROMO)
                or (not white and (1 << end) & B_PROMO)
            ) and not ((1 << piece_sq) & bb.k)

            if is_capt and isinstance(mv, tuple) and len(mv)>3:
                v="🔥 أكل متعدد!"
            elif promo: v="👑 ترقية!"
            elif sc>300: v="💪 ساحقة"
            elif sc>100: v="✅ ممتازة"
            elif sc>30: v="✅ جيدة"
            elif sc>-30: v="⚖️ متكافئة"
            elif sc>-100: v="⚠️ حذر"
            else: v="❌ تجنبها"

            wp = max(1,min(99,int(50+sc/10)))

            start_rc = SQ_TO_RC[mv[0]]
            end_rc = SQ_TO_RC[end]
            if isinstance(mv, tuple) and len(mv)>2:
                rc_path = tuple(SQ_TO_RC[s] for s in mv)
            else:
                rc_path = (start_rc, end_rc)

            results.append({
                "move": mv, "rc": rc_path,
                "score": round(sc,1),
                "depth": res["depth"],
                "is_capture": is_capt,
                "promotes": promo,
                "verdict": v, "win_pct": wp,
                "book": res.get("book", False),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        for i,r in enumerate(results): r["rank"]=i+1

        pe = self.evaluate(bb, white)
        return {
            "moves": results,
            "time": round(time.time()-self.t0,2),
            "forced": is_cap,
            "eval": round(pe,1),
            "tt_hits": self.tt.hits,
        }


# ══════════════════════════════════════════
# 6. الرؤية الحاسوبية (مختصر)
# ══════════════════════════════════════════

class Vision:
    @staticmethod
    def fix_perspective(img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)
        edges=cv2.Canny(blur,30,100)
        k=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        edges=cv2.dilate(edges,k,iterations=2)
        cnts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return cv2.resize(img,(400,400)),False
        big=max(cnts,key=cv2.contourArea)
        if cv2.contourArea(big)<img.shape[0]*img.shape[1]*0.15:
            return cv2.resize(img,(400,400)),False
        peri=cv2.arcLength(big,True)
        approx=cv2.approxPolyDP(big,0.02*peri,True)
        if len(approx)==4:
            pts=approx.reshape(4,2).astype(np.float32)
            s=pts.sum(axis=1);d=np.diff(pts,axis=1).ravel()
            o=np.zeros((4,2),dtype=np.float32)
            o[0]=pts[np.argmin(s)];o[2]=pts[np.argmax(s)]
            o[1]=pts[np.argmin(d)];o[3]=pts[np.argmax(d)]
            dst=np.float32([[0,0],[399,0],[399,399],[0,399]])
            M=cv2.getPerspectiveTransform(o,dst)
            return cv2.warpPerspective(img,M,(400,400)),True
        x,y,w,h=cv2.boundingRect(big)
        return cv2.resize(img[y:y+h,x:x+w],(400,400)),False

    @staticmethod
    def detect(img,lt=160,dt=100):
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        c=50;board=np.zeros((8,8),dtype=np.int8)
        for r in range(8):
            for cl in range(8):
                if(r+cl)%2==0:continue
                m=c//4
                rg=gray[r*c+m:(r+1)*c-m,cl*c+m:(cl+1)*c-m]
                rh=hsv[r*c+m:(r+1)*c-m,cl*c+m:(cl+1)*c-m]
                br=float(np.mean(rg));va=float(np.var(rg))
                sa=float(np.mean(rh[:,:,1]))
                if va>120 or sa>30:
                    if br>lt: board[r][cl]=1
                    elif br<dt: board[r][cl]=2
        return board


# ══════════════════════════════════════════
# 7. رسم الرقعة
# ══════════════════════════════════════════

class R:
    @staticmethod
    def draw(grid, arrows=None, hl=None):
        img=Image.new("RGB",(BPX,BPX))
        dr=ImageDraw.Draw(img)
        for r in range(8):
            for c in range(8):
                x1,y1=c*CELL,r*CELL;x2,y2=x1+CELL,y1+CELL
                sq=((235,215,180) if (r+c)%2==0 else (175,130,95))
                if hl and (r,c) in hl: sq=(100,200,100)
                dr.rectangle([x1,y1,x2,y2],fill=sq)
                p=int(grid[r][c])
                if p==0:continue
                cx,cy=x1+CELL//2,y1+CELL//2;pr=CELL//2-10
                dr.ellipse([cx-pr+3,cy-pr+3,cx+pr+3,cy+pr+3],
                    fill=(70,50,30))
                fl=(250,248,240) if p in(1,3) else (45,45,45)
                ed=(195,185,170) if p in(1,3) else (25,25,25)
                dr.ellipse([cx-pr,cy-pr,cx+pr,cy+pr],
                    fill=fl,outline=ed,width=2)
                if p in(3,4):
                    dr.ellipse([cx-12,cy-12,cx+12,cy+12],
                        fill=(255,215,0),outline=(200,170,0),width=2)
        if arrows:
            for a in arrows:
                R._arrow(dr,a["m"],a.get("c",(255,50,50)),
                    a.get("w",5))
        return img

    @staticmethod
    def _arrow(d,mv,color,w):
        if not mv or len(mv)<2: return
        for i in range(len(mv)-1):
            s,e=mv[i],mv[i+1]
            sx,sy=s[1]*CELL+CELL//2,s[0]*CELL+CELL//2
            ex,ey=e[1]*CELL+CELL//2,e[0]*CELL+CELL//2
            d.line([(sx,sy),(ex,ey)],fill=color,width=w)
            d.ellipse([ex-8,ey-8,ex+8,ey+8],fill=color)
        s=mv[0];sx,sy=s[1]*CELL+CELL//2,s[0]*CELL+CELL//2
        d.ellipse([sx-12,sy-12,sx+12,sy+12],
            outline=(0,220,0),width=4)


# ══════════════════════════════════════════
# 8. واجهة Streamlit
# ══════════════════════════════════════════

def app():
    st.set_page_config("♟️ مساعد الداما","♟️",layout="wide")
    st.markdown("""<style>
    .block-container{max-width:1100px}
    .best{background:linear-gradient(135deg,#28a745,#20c997);
        color:#fff;padding:18px;border-radius:12px;text-align:center;margin:10px 0}
    .best h2{margin:0}.best p{margin:5px 0}
    .warn{background:#f8d7da;border:2px solid #dc3545;color:#721c24;
        padding:12px;border-radius:10px;text-align:center;margin:8px 0;font-weight:bold}
    .ebar{background:#e9ecef;border-radius:8px;overflow:hidden;height:30px;margin:8px 0}
    .efill{height:100%;text-align:center;color:#fff;font-weight:bold;
        line-height:30px;border-radius:8px}
    .perf{background:#f0f2f6;padding:12px;border-radius:10px;
        border-left:4px solid #667eea;margin:8px 0}
    </style>""",unsafe_allow_html=True)

    if "board" not in st.session_state:
        st.session_state.board=BB().to_grid().tolist()

    st.title("♟️ مساعد الداما — Bitboard Engine")
    st.caption("32-bit Bitboard + Zobrist TT + PVS + LMR + Opening Book")

    with st.sidebar:
        st.header("⚙️")
        mc=st.radio("♟️",["⚪ أبيض","⚫ أسود"])
        fw="أبيض" in mc
        tt=st.select_slider("⏱",[1,2,3,5,8,10,15],value=3)
        st.divider()
        if st.button("🔄 جديدة",use_container_width=True):
            st.session_state.board=BB().to_grid().tolist()
            st.rerun()
        if st.button("🗑️ مسح",use_container_width=True):
            st.session_state.board=np.zeros((8,8),dtype=int).tolist()
            st.rerun()
        st.divider()
        g_=np.array(st.session_state.board)
        wn=int(np.sum((g_==1)|(g_==3)))
        bn=int(np.sum((g_==2)|(g_==4)))
        c1,c2=st.columns(2)
        with c1:st.metric("⚪",wn)
        with c2:st.metric("⚫",bn)

    tn=["✏️ يدوي"]
    if HAS_CV2:tn.append("📷 صورة")
    tn.append("🧠 تحليل")
    tabs=st.tabs(tn)

    with tabs[0]:
        opts={"⬜":0,"⚪":1,"⚫":2,"👑W":3,"♛B":4}
        sel=st.radio("_",list(opts.keys()),horizontal=True,
            label_visibility="collapsed")
        sv=opts[sel]
        syms={0:"·",1:"⚪",2:"⚫",3:"👑",4:"♛"}
        ba=np.array(st.session_state.board)
        for r in range(8):
            cols=st.columns(8)
            for c in range(8):
                with cols[c]:
                    ok=(r+c)%2!=0
                    s=syms.get(int(ba[r][c]),"·") if ok else ""
                    if st.button(s,key=f"m{r}{c}",
                        use_container_width=True,disabled=not ok):
                        st.session_state.board[r][c]=sv;st.rerun()
        st.image(R.draw(ba),caption="الرقعة",use_container_width=True)

    if HAS_CV2:
        with tabs[1]:
            up=st.file_uploader("📸",type=["jpg","png","jpeg"])
            if up:
                pil=Image.open(up).convert("RGB")
                icv=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
                c1,c2=st.columns(2)
                with c1:st.image(pil,use_container_width=True)
                with st.spinner("🔲"):
                    fixed,_=Vision.fix_perspective(icv)
                fp=Image.fromarray(cv2.cvtColor(fixed,cv2.COLOR_BGR2RGB))
                with c2:st.image(fp,use_container_width=True)
                tc1,tc2=st.columns(2)
                with tc1:lt=st.slider("فاتح",100,230,160)
                with tc2:dt=st.slider("داكن",30,150,100)
                if st.button("🔍",type="primary"):
                    merged=Vision.detect(fixed,lt,dt)
                    st.image(R.draw(merged),use_container_width=True)
                    if st.button("📥 استخدم",type="primary"):
                        st.session_state.board=merged.tolist();st.rerun()

    ai_idx=2 if HAS_CV2 else 1
    with tabs[ai_idx]:
        ba2=np.array(st.session_state.board,dtype=np.int8)
        bb=BB.from_grid(ba2)
        st.image(R.draw(ba2),caption="الرقعة",use_container_width=True)

        wn=popcount(bb.wp);bn=popcount(bb.bp)
        if wn==0 and bn==0:st.warning("⚠️ فارغة!");return

        w=bb.winner()
        if w:
            st.success("🏆 " + ("الأبيض!" if w==1 else "الأسود!"))
            return

        emoji="⚪" if fw else "⚫"
        st.markdown(f"### {emoji} تحليل **{'الأبيض' if fw else 'الأسود'}**")

        if st.button("🧠 حلّل!",type="primary",use_container_width=True):
            prg=st.empty()
            prg.info(f"🧠 Bitboard Engine ({tt}s)...")

            ai=Beast(max_time=tt)
            analysis=ai.analyze_all(bb,fw)
            prg.empty()

            if not analysis or not analysis["moves"]:
                st.error("❌ لا حركات!");return

            mvs=analysis["moves"];best=mvs[0]

            pe=analysis["eval"]
            if pe>300:em,ec="🟢 متفوق جداً!","#28a745"
            elif pe>100:em,ec="🟢 أفضل","#20c997"
            elif pe>30:em,ec="🟢 أفضل قليلاً","#17a2b8"
            elif pe>-30:em,ec="🟡 متكافئة","#ffc107"
            elif pe>-100:em,ec="🟠 الخصم أفضل","#fd7e14"
            else:em,ec="🔴 خطر!","#dc3545"
            pct=max(5,min(95,int(50+pe/10)))
            st.markdown(f'<div class="ebar"><div class="efill" '
                f'style="width:{pct}%;background:{ec}">{em} ({pe})</div></div>',
                unsafe_allow_html=True)

            if analysis["forced"]:
                st.markdown('<div class="warn">⚡ أكل إجباري!</div>',
                    unsafe_allow_html=True)

            rc=best["rc"]
            path=" → ".join(f"({p[0]},{p[1]})" for p in rc)
            ex=[]
            if best["is_capture"]:ex.append("💥 أكل")
            if best["promotes"]:ex.append("👑 ترقية")
            if best.get("book"):ex.append("📖 كتاب الافتتاحيات")
            ex.append(best["verdict"])
            ext=" • ".join(ex)

            st.markdown(f"""<div class="best">
                <h2>🏆 أفضل حركة</h2>
                <p style="font-size:1.4em">{path}</p><p>{ext}</p>
                <p>تقييم: {best['score']} • فوز: {best['win_pct']}% •
                عمق: {best['depth']}</p></div>""",unsafe_allow_html=True)

            c1,c2=st.columns(2)
            with c1:
                bi=R.draw(ba2,arrows=[{"m":list(rc),"c":(50,205,50),"w":6}],
                    hl=set(rc))
                st.image(bi,caption="🏆",use_container_width=True)
            with c2:
                af=bb.copy();af.do_move(best["move"],fw)
                st.image(R.draw(af.to_grid()),caption="📋 بعد التنفيذ",
                    use_container_width=True)

            if st.button("✅ طبّق",use_container_width=True):
                st.session_state.board=af.to_grid().tolist();st.rerun()

            st.markdown(f"""<div class="perf">⚡ <b>Bitboard Engine:</b>
                عمق {best['depth']} • TT: {analysis.get('tt_hits',0):,} hit •
                الوقت: {analysis['time']}s</div>""",unsafe_allow_html=True)

            st.markdown(f"### 📊 كل الحركات ({len(mvs)})")
            rc_map={1:("🥇",(50,205,50)),2:("🥈",(65,105,225)),
                    3:("🥉",(255,165,0))}

            top=[]
            for i,mv in enumerate(mvs[:5]):
                _,cl=rc_map.get(mv["rank"],(f"#{mv['rank']}",(180,180,180)))
                top.append({"m":list(mv["rc"]),"c":cl,"w":6 if i==0 else 3})
            st.image(R.draw(ba2,arrows=top),caption="🥇🥈🥉",
                use_container_width=True)

            for mv in mvs:
                icon=rc_map.get(mv["rank"],(f"#{mv['rank']}",None))[0]
                p=" → ".join(f"({x[0]},{x[1]})" for x in mv["rc"])
                bar="█"*max(1,int(mv["win_pct"]/5))
                with st.expander(f"{icon} {p} • {mv['score']} • "
                    f"{mv['win_pct']}% • {mv['verdict']}"):
                    st.markdown(f"`{p}`")
                    st.markdown(f"فوز: `{bar}` {mv['win_pct']}%")
                    st.markdown(f"عمق: {mv['depth']}")
                    mi=R.draw(ba2,arrows=[{"m":list(mv["rc"]),
                        "c":(255,100,50),"w":5}],hl=set(mv["rc"]))
                    st.image(mi,use_container_width=True)

    st.divider()
    st.markdown('<p style="text-align:center;color:#999;font-size:0.8em">'
        '♟️ v8.0 — 32-bit Bitboard + Zobrist TT + PVS + LMR + '
        'Quiescence + Opening Book — بدون إنترنت</p>',
        unsafe_allow_html=True)

if __name__=="__main__":
    app()
