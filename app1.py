#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════╗
║     ♟️ مساعد الداما الذكي — محرك وحشي v7     ║
║  تقييم 12 عامل + Zobrist TT + PVS + LMR      ║
║  عمق 15-20 في 3 ثوانٍ = لا يُقهر              ║
╚═══════════════════════════════════════════════╝
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
# 1. ثوابت
# ══════════════════════════════════════════

class P(IntEnum):
    E = 0; W = 1; B = 2; WK = 3; BK = 4

CELL = 80
BPX = CELL * 8

# جداول التقييم الموضعي
ADV_W = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,2,0,2,0,2,0],
    [0,3,0,4,0,4,0,3],
    [0,0,5,0,6,0,5,0],
    [0,6,0,8,0,8,0,6],
    [0,0,9,0,10,0,9,0],
    [0,12,0,14,0,14,0,12],
    [0,0,0,0,0,0,0,0],
], dtype=np.float32)
ADV_B = ADV_W[::-1].copy()

KING_TABLE = np.array([
    [0,2,0,2,0,2,0,2],
    [2,0,4,0,4,0,4,0],
    [0,4,0,6,0,6,0,4],
    [2,0,6,0,8,0,6,0],
    [0,6,0,8,0,6,0,2],
    [4,0,6,0,6,0,4,0],
    [0,4,0,4,0,4,0,2],
    [2,0,2,0,2,0,2,0],
], dtype=np.float32)


# ══════════════════════════════════════════
# 2. محرك القواعد (مُحسّن)
# ══════════════════════════════════════════

class Engine:

    __slots__ = ['board']

    def __init__(self, board=None):
        self.board = (
            np.array(board, dtype=np.int8)
            if board is not None
            else self._init()
        )

    @staticmethod
    def _init():
        b = np.zeros((8, 8), dtype=np.int8)
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 != 0:
                    if r < 3: b[r][c] = P.B
                    elif r > 4: b[r][c] = P.W
        return b

    def copy(self):
        e = Engine.__new__(Engine)
        e.board = self.board.copy()
        return e

    @staticmethod
    def _own(piece, white):
        if white: return piece in (P.W, P.WK)
        return piece in (P.B, P.BK)

    @staticmethod
    def _foe(piece, white):
        if piece == P.E: return False
        if white: return piece in (P.B, P.BK)
        return piece in (P.W, P.WK)

    @staticmethod
    def _dirs(piece):
        if piece == P.W: return ((-1,-1),(-1,1))
        if piece == P.B: return ((1,-1),(1,1))
        if piece in (P.WK,P.BK):
            return ((-1,-1),(-1,1),(1,-1),(1,1))
        return ()

    def get_moves(self, white):
        b = self.board
        jumps = []
        simple = []
        for r in range(8):
            for c in range(8):
                p = b[r][c]
                if not self._own(p, white):
                    continue
                # أكل
                for dr, dc in self._dirs(p):
                    mr, mc = r+dr, c+dc
                    nr, nc = r+2*dr, c+2*dc
                    if not (0<=nr<8 and 0<=nc<8):
                        continue
                    if (b[nr][nc] == P.E
                            and self._foe(b[mr][mc], white)):
                        self._chain(
                            r, c, b, white, jumps, frozenset()
                        )
                        break
                else:
                    # بسيطة
                    for dr, dc in self._dirs(p):
                        nr, nc = r+dr, c+dc
                        if (0<=nr<8 and 0<=nc<8
                                and b[nr][nc] == P.E):
                            simple.append(((r,c),(nr,nc)))

        if jumps:
            mx = max(len(j) for j in jumps)
            return [j for j in jumps if len(j)==mx], True
        return simple, False

    def _chain(self, r, c, bd, white, out, eaten):
        p = bd[r][c]
        found = False
        for dr, dc in self._dirs(p):
            mr, mc = r+dr, c+dc
            nr, nc = r+2*dr, c+2*dc
            if not (0<=nr<8 and 0<=nc<8):
                continue
            if bd[nr][nc] != P.E:
                continue
            if not self._foe(bd[mr][mc], white):
                continue
            if (mr,mc) in eaten:
                continue
            found = True
            nb = bd.copy()
            nb[nr][nc] = p; nb[r][c] = P.E; nb[mr][mc] = P.E
            promo = False
            if nr == 0 and p == P.W:
                nb[nr][nc] = P.WK; promo = True
            elif nr == 7 and p == P.B:
                nb[nr][nc] = P.BK; promo = True
            ne = eaten | {(mr,mc)}
            if not promo:
                sub = []
                self._chain(nr, nc, nb, white, sub, ne)
                if sub:
                    for s in sub:
                        out.append(((r,c),) + s)
                else:
                    out.append(((r,c),(nr,nc)))
            else:
                out.append(((r,c),(nr,nc)))

        if not found and eaten:
            pass  # نهاية السلسلة

    def do_move(self, move):
        b = self.board
        piece = b[move[0][0]][move[0][1]]
        b[move[0][0]][move[0][1]] = P.E
        for i in range(len(move)-1):
            sr,sc = move[i]; er,ec = move[i+1]
            dr,dc = er-sr, ec-sc
            if abs(dr) == 2:
                b[sr+dr//2][sc+dc//2] = P.E
        fr, fc = move[-1]
        b[fr][fc] = piece
        if fr == 0 and piece == P.W: b[fr][fc] = P.WK
        if fr == 7 and piece == P.B: b[fr][fc] = P.BK

    def winner(self):
        """None=مستمرة, 1=أبيض, 2=أسود, 0=تعادل"""
        wn = int(np.sum((self.board==P.W)|(self.board==P.WK)))
        bn = int(np.sum((self.board==P.B)|(self.board==P.BK)))
        if wn == 0: return 2
        if bn == 0: return 1
        wm, _ = self.get_moves(True)
        if not wm:
            bm, _ = self.get_moves(False)
            if not bm: return 0
            return 2
        return None


# ══════════════════════════════════════════
# 3. Zobrist Transposition Table
# ══════════════════════════════════════════

EXACT, LOWER, UPPER = 0, 1, 2

class ZobristTT:
    def __init__(self, size=500_000):
        rng = random.Random(12345)
        self.z = {}
        for p in range(1, 5):
            for r in range(8):
                for c in range(8):
                    self.z[(p,r,c)] = rng.getrandbits(64)
        self.side = rng.getrandbits(64)
        self.table = {}
        self.size = size
        self.hits = 0

    def key(self, board, white_turn):
        h = 0
        for r in range(8):
            for c in range(8):
                p = int(board[r][c])
                if p: h ^= self.z[(p,r,c)]
        if white_turn: h ^= self.side
        return h

    def probe(self, k, depth, alpha, beta):
        e = self.table.get(k)
        if e and e[0] >= depth:
            self.hits += 1
            sc, fl, mv = e[1], e[2], e[3]
            if fl == EXACT: return sc, mv
            if fl == LOWER and sc >= beta: return sc, mv
            if fl == UPPER and sc <= alpha: return sc, mv
        return None

    def store(self, k, depth, score, flag, move):
        old = self.table.get(k)
        if not old or old[0] <= depth:
            self.table[k] = (depth, score, flag, move)
            if len(self.table) > self.size:
                # حذف نصف المدخلات
                keys = list(self.table.keys())
                for kk in keys[:len(keys)//2]:
                    del self.table[kk]

    def best_move(self, k):
        e = self.table.get(k)
        return e[3] if e else None

    def clear(self):
        self.table.clear(); self.hits = 0


# ══════════════════════════════════════════
# 4. المحرك الوحشي
# ══════════════════════════════════════════

class BeastAI:
    """
    تقييم بـ 12 عامل بدون get_moves() = فوري
    + Zobrist TT + PVS + Killer + History + LMR
    + Quiescence + Iterative Deepening
    = عمق 15-20 في ثوانٍ
    """

    def __init__(self, max_time=3.0):
        self.max_time = max_time
        self.tt = ZobristTT()
        self.nodes = 0
        self.t0 = 0
        self.stopped = False
        self.depth_reached = 0
        self.killers = [[None,None] for _ in range(64)]
        self.history = {}

    # ── تقييم فوري (12 عامل، بدون get_moves) ──

    def evaluate(self, eng, for_white):
        b = eng.board

        # مصفوفات سريعة
        wm = (b == P.W)
        wk = (b == P.WK)
        bm = (b == P.B)
        bk = (b == P.BK)

        w_men = int(np.sum(wm))
        w_kings = int(np.sum(wk))
        b_men = int(np.sum(bm))
        b_kings = int(np.sum(bk))

        w_total = w_men + w_kings
        b_total = b_men + b_kings
        all_total = w_total + b_total

        if w_total == 0: return -99999 if for_white else 99999
        if b_total == 0: return 99999 if for_white else -99999

        score = 0.0

        # ═══ 1. مادة ═══
        score += (w_men * 100 + w_kings * 150)
        score -= (b_men * 100 + b_kings * 150)

        # ═══ 2. موقع (تقدم + مركز) ═══
        score += float(np.sum(wm * ADV_W)) * 3
        score -= float(np.sum(bm * ADV_B)) * 3
        score += float(np.sum(wk * KING_TABLE)) * 2
        score -= float(np.sum(bk * KING_TABLE)) * 2

        # ═══ 3. الصف الخلفي (مهم في الافتتاح) ═══
        if all_total > 16:
            score += int(np.sum(b[7,:]==P.W)) * 8
            score -= int(np.sum(b[0,:]==P.B)) * 8

        # ═══ 4. المركز الحقيقي ═══
        center = b[2:6, 2:6]
        for v in (P.W, P.WK):
            score += int(np.sum(center==v)) * 5
        for v in (P.B, P.BK):
            score -= int(np.sum(center==v)) * 5

        # ═══ 5-8. أنماط تكتيكية (لكل قطعة) ═══
        for r in range(8):
            for c in range(8):
                p = b[r][c]
                if p == P.E:
                    continue

                is_w = (p in (P.W, P.WK))
                mult = 1 if is_w else -1

                # 5. الترابط (حلفاء قطريين)
                allies = 0
                for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
                    ar, ac = r+dr, c+dc
                    if 0<=ar<8 and 0<=ac<8:
                        ap = b[ar][ac]
                        if is_w and ap in (P.W,P.WK):
                            allies += 1
                        elif not is_w and ap in (P.B,P.BK):
                            allies += 1
                score += allies * 3 * mult

                # 6. الحماية من الأكل
                for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
                    ar, ac = r+dr, c+dc
                    br_, bc = r-dr, c-dc
                    if (0<=ar<8 and 0<=ac<8
                            and 0<=br_<8 and 0<=bc<8):
                        ap = b[ar][ac]
                        is_enemy = (
                            (is_w and ap in (P.B,P.BK))
                            or (not is_w and ap in (P.W,P.WK))
                        )
                        if is_enemy and b[br_][bc] == P.E:
                            score -= 10 * mult
                            break

                # 7. القطعة الهاربة (runaway)
                if p == P.W and not (b[0:r, :] == P.B).any() \
                        and not (b[0:r, :] == P.BK).any():
                    if r <= 3:
                        score += (4 - r) * 15
                elif p == P.B and not (b[r+1:, :] == P.W).any() \
                        and not (b[r+1:, :] == P.WK).any():
                    if r >= 4:
                        score -= (r - 3) * 15

                # 8. الملك المحاصر (على الحافة)
                if p in (P.WK, P.BK):
                    if r == 0 or r == 7 or c == 0 or c == 7:
                        exits = 0
                        for dr, dc in ((-1,-1),(-1,1),
                                       (1,-1),(1,1)):
                            nr, nc = r+dr, c+dc
                            if 0<=nr<8 and 0<=nc<8:
                                if b[nr][nc] == P.E:
                                    exits += 1
                        if exits <= 1:
                            score -= 12 * mult

        # ═══ 9. جسر دفاعي ═══
        # الأبيض: قطع على (7,0)+(6,1) أو (7,6)+(6,7)
        if b[7][0] == P.W and b[6][1] == P.W:
            score += 8
        if b[7][6] == P.W and b[6][7] == P.W:
            score += 8
        # الأسود: قطع على (0,1)+(1,0) أو (0,7)+(1,6)
        if b[0][1] == P.B and b[1][0] == P.B:
            score -= 8
        if b[0][7] == P.B and b[1][6] == P.B:
            score -= 8

        # ═══ 10. نهاية اللعبة ═══
        if all_total <= 8:
            w_mat = w_men + w_kings * 2
            b_mat = b_men + b_kings * 2
            diff = w_mat - b_mat
            score += diff * 20
            if diff > 0:
                score += (16 - all_total) * 5

        # ═══ 11. حافة الترقية ═══
        for c in range(8):
            if b[1][c] == P.W: score += 15
            if b[6][c] == P.B: score -= 15

        # ═══ 12. توازن الأجنحة ═══
        w_left = int(np.sum(wm[:,:4]) + np.sum(wk[:,:4]))
        w_right = int(np.sum(wm[:,4:]) + np.sum(wk[:,4:]))
        b_left = int(np.sum(bm[:,:4]) + np.sum(bk[:,:4]))
        b_right = int(np.sum(bm[:,4:]) + np.sum(bk[:,4:]))
        if w_left > 0 and w_right > 0: score += 3
        if b_left > 0 and b_right > 0: score -= 3

        return score if for_white else -score

    # ── ترتيب الحركات ──

    def _order(self, moves, is_cap, eng, white,
               depth, tt_move):
        scored = []
        for m in moves:
            s = 0
            if tt_move and m == tt_move:
                s += 100000
            elif is_cap:
                s += 50000 + len(m) * 500
                # أكل ملك أفضل من أكل قطعة
                for i in range(len(m)-1):
                    sr,sc = m[i]; er,ec = m[i+1]
                    dr,dc = er-sr, ec-sc
                    if abs(dr)==2:
                        mid = eng.board[sr+dr//2][sc+dc//2]
                        if mid in (P.WK,P.BK):
                            s += 2000
            elif depth < 64 and m in self.killers[depth]:
                s += 40000
            else:
                key = (m[0], m[-1])
                s += self.history.get(key, 0)

            dest = m[-1]
            if white and dest[0] == 0: s += 30000
            if not white and dest[0] == 7: s += 30000
            if 2<=dest[0]<=5 and 2<=dest[1]<=5: s += 50
            scored.append((s, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _,m in scored]

    # ── Quiescence ──

    def _quiesce(self, eng, alpha, beta, white, depth=0):
        self.nodes += 1
        if self.nodes % 4096 == 0:
            if time.time() - self.t0 >= self.max_time:
                self.stopped = True
                return 0

        stand = self.evaluate(eng, white)
        if depth >= 8 or stand >= beta:
            return stand if stand < beta else beta
        if stand > alpha:
            alpha = stand

        moves, is_cap = eng.get_moves(white)
        if not is_cap:
            return stand

        for mv in moves:
            child = eng.copy()
            child.do_move(mv)
            sc = -self._quiesce(child, -beta, -alpha,
                                not white, depth+1)
            if self.stopped: return 0
            if sc >= beta: return beta
            if sc > alpha: alpha = sc
        return alpha

    # ── Alpha-Beta + PVS + LMR ──

    def _search(self, eng, depth, alpha, beta,
                white, pv=True):
        self.nodes += 1
        if self.nodes % 4096 == 0:
            if time.time() - self.t0 >= self.max_time:
                self.stopped = True
                return 0, None

        # نهاية اللعبة
        w = eng.winner()
        if w is not None:
            if w == 0: return 0, None
            is_my_win = (w==1 and white) or (w==2 and not white)
            return (99999+depth if is_my_win
                    else -99999-depth), None

        if depth <= 0:
            return self._quiesce(eng, alpha, beta, white), None

        # ── TT Probe ──
        bk = self.tt.key(eng.board, white)
        tt_res = self.tt.probe(bk, depth, alpha, beta)
        if tt_res and not pv:
            return tt_res

        moves, is_cap = eng.get_moves(white)
        if not moves:
            return -99999, None

        tt_mv = self.tt.best_move(bk)
        moves = self._order(
            moves, is_cap, eng, white, depth, tt_mv
        )

        best = moves[0]
        best_sc = float("-inf")
        orig_alpha = alpha
        i = 0

        for mv in moves:
            child = eng.copy()
            child.do_move(mv)

            # ── LMR: Late Move Reduction ──
            reduction = 0
            if (i >= 4 and depth >= 3
                    and not is_cap and not pv):
                reduction = 1
                if i >= 8:
                    reduction = 2

            if i == 0:
                sc = -self._search(
                    child, depth-1, -beta, -alpha,
                    not white, True
                )[0]
            else:
                # نافذة ضيقة + تقليل
                sc = -self._search(
                    child, depth-1-reduction,
                    -alpha-1, -alpha,
                    not white, False
                )[0]

                # إعادة بحث إذا لزم
                if (alpha < sc < beta
                        and not self.stopped):
                    if reduction > 0:
                        sc = -self._search(
                            child, depth-1,
                            -alpha-1, -alpha,
                            not white, False
                        )[0]
                    if (alpha < sc < beta
                            and not self.stopped):
                        sc = -self._search(
                            child, depth-1,
                            -beta, -sc,
                            not white, True
                        )[0]

            if self.stopped:
                break
            i += 1

            if sc > best_sc:
                best_sc = sc; best = mv
            if sc > alpha:
                alpha = sc
            if alpha >= beta:
                if not is_cap and depth < 64:
                    k = self.killers[depth]
                    if mv != k[0]:
                        k[1] = k[0]; k[0] = mv
                    key = (mv[0], mv[-1])
                    self.history[key] = \
                        self.history.get(key, 0) + depth*depth
                break

        if not self.stopped:
            if best_sc <= orig_alpha: fl = UPPER
            elif best_sc >= beta: fl = LOWER
            else: fl = EXACT
            self.tt.store(bk, depth, best_sc, fl, best)

        return best_sc, best

    # ── Iterative Deepening ──

    def find_best(self, eng, white):
        self.nodes = 0
        self.t0 = time.time()
        self.stopped = False
        self.depth_reached = 0
        self.killers = [[None,None] for _ in range(64)]
        self.history = {}

        best_mv = None
        best_sc = 0
        log = []

        for d in range(1, 50):
            sc, mv = self._search(
                eng, d, float("-inf"), float("inf"),
                white, True
            )
            if self.stopped:
                break
            if mv is not None:
                best_mv = mv
                best_sc = sc
                self.depth_reached = d
                el = time.time() - self.t0
                log.append({
                    "d": d, "sc": round(sc,1),
                    "n": self.nodes,
                    "t": round(el,2),
                    "nps": int(self.nodes/max(el,0.001)),
                })
            if abs(best_sc) > 90000: break
            if time.time()-self.t0 >= self.max_time*0.75: break

        el = time.time() - self.t0
        return {
            "move": best_mv,
            "score": round(best_sc, 1),
            "nodes": self.nodes,
            "time": round(el, 2),
            "depth": self.depth_reached,
            "nps": int(self.nodes/max(el,0.001)),
            "tt_hits": self.tt.hits,
            "log": log,
        }

    def analyze_all(self, eng, white):
        """تحليل كل الحركات مع مشاركة TT"""
        self.tt.clear()
        self.nodes = 0
        self.t0 = time.time()
        self.stopped = False

        moves, is_cap = eng.get_moves(white)
        if not moves: return None

        each_t = max(0.3, self.max_time*0.6 / len(moves))
        results = []

        for mv in moves:
            child = eng.copy()
            child.do_move(mv)

            sub = BeastAI(max_time=each_t)
            sub.tt = self.tt  # مشاركة الكاش!
            res = sub.find_best(child, not white)
            sc = -res["score"]

            is_capt = (
                len(mv) > 2
                or (len(mv)==2
                    and abs(mv[0][0]-mv[1][0])==2)
            )
            cap_n = 0
            if is_capt:
                for j in range(len(mv)-1):
                    if abs(mv[j][0]-mv[j+1][0])==2:
                        cap_n += 1

            dest = mv[-1]
            piece = eng.board[mv[0][0]][mv[0][1]]
            promo = (
                (dest[0]==0 and piece==P.W)
                or (dest[0]==7 and piece==P.B)
            )

            if is_capt and cap_n>=2: v="🔥 أكل متعدد!"
            elif promo: v="👑 ترقية!"
            elif sc>300: v="💪 ساحقة"
            elif sc>100: v="✅ ممتازة"
            elif sc>30: v="✅ جيدة"
            elif sc>-30: v="⚖️ متكافئة"
            elif sc>-100: v="⚠️ حذر"
            else: v="❌ تجنبها"

            wp = max(1, min(99, int(50+sc/10)))
            results.append({
                "move": mv, "score": round(sc,1),
                "depth": res["depth"],
                "is_capture": is_capt,
                "captured": cap_n,
                "promotes": promo,
                "verdict": v, "win_pct": wp,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i+1

        pe = self.evaluate(eng, white)
        total_n = sum(
            r.get("depth", 0) for r in results
        )
        return {
            "moves": results,
            "time": round(time.time()-self.t0, 2),
            "forced": is_cap,
            "eval": round(pe, 1),
            "tt_hits": self.tt.hits,
        }


# ══════════════════════════════════════════
# 5. الرؤية الحاسوبية
# ══════════════════════════════════════════

class Vision:
    @staticmethod
    def fix_perspective(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 30, 100)
        k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        edges = cv2.dilate(edges, k, iterations=2)
        cnts,_ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            return cv2.resize(img,(400,400)), False
        big = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(big)<img.shape[0]*img.shape[1]*0.15:
            return cv2.resize(img,(400,400)), False
        peri = cv2.arcLength(big,True)
        approx = cv2.approxPolyDP(big,0.02*peri,True)
        if len(approx)==4:
            pts = approx.reshape(4,2).astype(np.float32)
            s=pts.sum(axis=1); d=np.diff(pts,axis=1).ravel()
            o=np.zeros((4,2),dtype=np.float32)
            o[0]=pts[np.argmin(s)]; o[2]=pts[np.argmax(s)]
            o[1]=pts[np.argmin(d)]; o[3]=pts[np.argmax(d)]
            dst=np.float32([[0,0],[399,0],[399,399],[0,399]])
            M=cv2.getPerspectiveTransform(o,dst)
            return cv2.warpPerspective(img,M,(400,400)), True
        x,y,w,h = cv2.boundingRect(big)
        return cv2.resize(img[y:y+h,x:x+w],(400,400)), False

    @staticmethod
    def detect(img, lt=160, dt=100):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c=50; b1=np.zeros((8,8),dtype=np.int8); info=[]
        for r in range(8):
            for cl in range(8):
                if (r+cl)%2==0: continue
                m=c//4
                rg=gray[r*c+m:(r+1)*c-m, cl*c+m:(cl+1)*c-m]
                rh=hsv[r*c+m:(r+1)*c-m, cl*c+m:(cl+1)*c-m]
                br=float(np.mean(rg))
                sa=float(np.mean(rh[:,:,1]))
                va=float(np.var(rg))
                det=int(P.E)
                if va>120 or sa>30:
                    if br>lt: det=int(P.W)
                    elif br<dt: det=int(P.B)
                b1[r][cl]=det
                info.append({"r":r,"c":cl,
                    "br":round(br),"va":round(va),"d":det})

        gm=cv2.medianBlur(gray,5)
        circles=cv2.HoughCircles(gm,cv2.HOUGH_GRADIENT,
            1.2,35,param1=60,param2=35,minRadius=12,maxRadius=24)
        b2=np.zeros((8,8),dtype=np.int8)
        vis=img.copy()
        if circles is not None:
            for cx,cy,rad in np.uint16(np.around(circles))[0]:
                co,ro=int(cx/c),int(cy/c)
                if not(0<=ro<8 and 0<=co<8) or (ro+co)%2==0:
                    continue
                s=gray[max(0,int(cy)-5):min(400,int(cy)+5),
                       max(0,int(cx)-5):min(400,int(cx)+5)]
                if s.size==0: continue
                a=float(np.mean(s))
                if a>lt: b2[ro][co]=int(P.W); clr=(0,255,0)
                elif a<dt: b2[ro][co]=int(P.B); clr=(0,0,255)
                else: continue
                cv2.circle(vis,(int(cx),int(cy)),int(rad),clr,2)

        vis_pil=Image.fromarray(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB))
        merged=np.zeros((8,8),dtype=np.int8)
        for r in range(8):
            for c_ in range(8):
                h,ci=b1[r][c_],b2[r][c_]
                if h!=P.E and ci!=P.E: merged[r][c_]=h
                elif ci!=P.E: merged[r][c_]=ci
                elif h!=P.E: merged[r][c_]=h
        return merged, b1, b2, vis_pil, info


# ══════════════════════════════════════════
# 6. رسم الرقعة
# ══════════════════════════════════════════

class R:
    @staticmethod
    def draw(board, arrows=None, hl=None):
        img=Image.new("RGB",(BPX,BPX))
        dr=ImageDraw.Draw(img)
        for r in range(8):
            for c in range(8):
                x1,y1=c*CELL,r*CELL; x2,y2=x1+CELL,y1+CELL
                sq=((235,215,180) if (r+c)%2==0
                    else (175,130,95))
                if hl and (r,c) in hl: sq=(100,200,100)
                dr.rectangle([x1,y1,x2,y2],fill=sq)
                p=board[r][c]
                if p==P.E: continue
                cx,cy=x1+CELL//2,y1+CELL//2; pr=CELL//2-10
                dr.ellipse([cx-pr+3,cy-pr+3,cx+pr+3,cy+pr+3],
                    fill=(70,50,30))
                fl=((250,248,240) if p in(P.W,P.WK)
                    else (45,45,45))
                ed=((195,185,170) if p in(P.W,P.WK)
                    else (25,25,25))
                dr.ellipse([cx-pr,cy-pr,cx+pr,cy+pr],
                    fill=fl,outline=ed,width=2)
                dr.ellipse([cx-pr+5,cy-pr+5,cx+pr-5,cy+pr-5],
                    outline=ed,width=1)
                if p in(P.WK,P.BK):
                    dr.ellipse([cx-12,cy-12,cx+12,cy+12],
                        fill=(255,215,0),
                        outline=(200,170,0),width=2)
        for i in range(8):
            try:
                dr.text((3,i*CELL+3),str(i),fill=(130,110,90))
                dr.text((i*CELL+CELL//2-4,BPX-14),
                    chr(65+i),fill=(130,110,90))
            except: pass
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
        s=mv[0]; sx,sy=s[1]*CELL+CELL//2,s[0]*CELL+CELL//2
        d.ellipse([sx-12,sy-12,sx+12,sy+12],
            outline=(0,220,0),width=4)


# ══════════════════════════════════════════
# 7. واجهة Streamlit
# ══════════════════════════════════════════

def app():
    st.set_page_config("♟️ مساعد الداما","♟️",layout="wide")

    st.markdown("""<style>
    .block-container{max-width:1100px}
    .best{background:linear-gradient(135deg,#28a745,#20c997);
        color:#fff;padding:18px;border-radius:12px;
        text-align:center;margin:10px 0}
    .best h2{margin:0;font-size:1.6em}
    .best p{margin:5px 0}
    .warn{background:#f8d7da;border:2px solid #dc3545;
        color:#721c24;padding:12px;border-radius:10px;
        text-align:center;margin:8px 0;font-weight:bold}
    .ebar{background:#e9ecef;border-radius:8px;
        overflow:hidden;height:30px;margin:8px 0}
    .efill{height:100%;text-align:center;color:#fff;
        font-weight:bold;line-height:30px;border-radius:8px}
    .perf{background:#f0f2f6;padding:12px;border-radius:10px;
        border-left:4px solid #667eea;margin:8px 0}
    </style>""", unsafe_allow_html=True)

    if "board" not in st.session_state:
        st.session_state.board = Engine._init().tolist()

    st.title("♟️ مساعد الداما الوحشي")
    st.caption(
        "12 عامل تقييم + Zobrist TT + PVS + LMR = "
        "عمق 15-20 في ثوانٍ"
    )

    # الشريط الجانبي
    with st.sidebar:
        st.header("⚙️")
        mc = st.radio("♟️ لونك:",["⚪ أبيض","⚫ أسود"])
        fw = "أبيض" in mc
        tt = st.select_slider("⏱ وقت:",
            [1,2,3,5,8,10,15], value=3)
        st.divider()
        if st.button("🔄 جديدة",use_container_width=True):
            st.session_state.board=Engine._init().tolist()
            st.rerun()
        if st.button("🗑️ مسح",use_container_width=True):
            st.session_state.board = \
                np.zeros((8,8),dtype=int).tolist()
            st.rerun()
        st.divider()
        g_=np.array(st.session_state.board)
        wn=int(np.sum((g_==P.W)|(g_==P.WK)))
        bn=int(np.sum((g_==P.B)|(g_==P.BK)))
        c1,c2=st.columns(2)
        with c1: st.metric("⚪",wn)
        with c2: st.metric("⚫",bn)

    # التبويبات
    tn=["✏️ يدوي"]
    if HAS_CV2: tn.append("📷 صورة")
    tn.append("🧠 تحليل")
    tabs=st.tabs(tn)

    # يدوي
    with tabs[0]:
        opts={"⬜":int(P.E),"⚪":int(P.W),
              "⚫":int(P.B),"👑W":int(P.WK),"♛B":int(P.BK)}
        sel=st.radio("_",list(opts.keys()),horizontal=True,
            label_visibility="collapsed")
        sv=opts[sel]
        syms={int(P.E):"·",int(P.W):"⚪",int(P.B):"⚫",
              int(P.WK):"👑",int(P.BK):"♛"}
        ba=np.array(st.session_state.board)
        for r in range(8):
            cols=st.columns(8)
            for c in range(8):
                with cols[c]:
                    ok=(r+c)%2!=0
                    s=syms.get(int(ba[r][c]),"·") if ok else ""
                    if st.button(s,key=f"m{r}{c}",
                        use_container_width=True,disabled=not ok):
                        st.session_state.board[r][c]=sv
                        st.rerun()
        st.image(R.draw(ba),caption="الرقعة",
            use_container_width=True)

    # صورة
    if HAS_CV2:
        with tabs[1]:
            st.subheader("📷 تحليل صورة")
            up=st.file_uploader("📸",type=["jpg","png","jpeg"])
            if up:
                pil=Image.open(up).convert("RGB")
                icv=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
                c1,c2=st.columns(2)
                with c1: st.image(pil,use_container_width=True)
                with st.spinner("🔲..."):
                    fixed,was=Vision.fix_perspective(icv)
                fp=Image.fromarray(
                    cv2.cvtColor(fixed,cv2.COLOR_BGR2RGB))
                with c2: st.image(fp,use_container_width=True)
                tc1,tc2=st.columns(2)
                with tc1: lt=st.slider("فاتح",100,230,160)
                with tc2: dt=st.slider("داكن",30,150,100)
                if st.button("🔍 تحليل",type="primary"):
                    with st.spinner("🧠..."):
                        merged,_,_,cvis,_=Vision.detect(
                            fixed,lt,dt)
                    st.success("✅")
                    c1,c2=st.columns(2)
                    with c1: st.image(cvis,
                        use_container_width=True)
                    with c2: st.image(R.draw(merged),
                        use_container_width=True)
                    if st.button("📥 استخدم",type="primary"):
                        st.session_state.board=merged.tolist()
                        st.rerun()

    # تحليل
    ai_idx = 2 if HAS_CV2 else 1
    with tabs[ai_idx]:
        ba2=np.array(st.session_state.board,dtype=np.int8)
        eng=Engine(ba2)
        st.image(R.draw(ba2),caption="الرقعة",
            use_container_width=True)

        wn=int(np.sum((ba2==P.W)|(ba2==P.WK)))
        bn=int(np.sum((ba2==P.B)|(ba2==P.BK)))
        if wn==0 and bn==0:
            st.warning("⚠️ فارغة!"); return

        w=eng.winner()
        if w is not None:
            if w==0: st.info("🤝 تعادل")
            elif w==1: st.success("🏆 الأبيض!")
            else: st.success("🏆 الأسود!")
            return

        emoji="⚪" if fw else "⚫"
        name="الأبيض" if fw else "الأسود"
        st.markdown(f"### {emoji} تحليل **{name}**")

        if st.button("🧠 حلّل!",type="primary",
            use_container_width=True):

            prg=st.empty()
            prg.info(f"🧠 تحليل ({tt}s)...")

            ai=BeastAI(max_time=tt)
            analysis=ai.analyze_all(eng, fw)
            prg.empty()

            if not analysis or not analysis["moves"]:
                st.error("❌ لا حركات!"); return

            mvs=analysis["moves"]
            best=mvs[0]

            # تقييم الوضعية
            pe=analysis["eval"]
            if pe>300: em,ec="🟢 متفوق جداً!","#28a745"
            elif pe>100: em,ec="🟢 أفضل","#20c997"
            elif pe>30: em,ec="🟢 أفضل قليلاً","#17a2b8"
            elif pe>-30: em,ec="🟡 متكافئة","#ffc107"
            elif pe>-100: em,ec="🟠 الخصم أفضل","#fd7e14"
            else: em,ec="🔴 خطر!","#dc3545"
            pct=max(5,min(95,int(50+pe/10)))
            st.markdown(
                f'<div class="ebar"><div class="efill" '
                f'style="width:{pct}%;background:{ec}">'
                f'{em} ({pe})</div></div>',
                unsafe_allow_html=True)

            if analysis["forced"]:
                st.markdown('<div class="warn">'
                    '⚡ أكل إجباري!</div>',
                    unsafe_allow_html=True)

            # أفضل حركة
            path=" → ".join(
                f"({p[0]},{p[1]})" for p in best["move"])
            ex=[]
            if best["is_capture"]:
                ex.append(f"💥×{best['captured']}")
            if best["promotes"]: ex.append("👑")
            ex.append(best["verdict"])
            ext=" • ".join(ex)

            st.markdown(f"""
            <div class="best">
                <h2>🏆 أفضل حركة</h2>
                <p style="font-size:1.4em">{path}</p>
                <p>{ext}</p>
                <p>تقييم: {best['score']} •
                   فوز: {best['win_pct']}% •
                   عمق: {best['depth']}</p>
            </div>""", unsafe_allow_html=True)

            c1,c2=st.columns(2)
            with c1:
                bi=R.draw(ba2,
                    arrows=[{"m":best["move"],
                        "c":(50,205,50),"w":6}],
                    hl=set(best["move"]))
                st.image(bi,caption="🏆",
                    use_container_width=True)
            with c2:
                af=eng.copy()
                af.do_move(best["move"])
                st.image(R.draw(af.board),
                    caption="📋 بعد التنفيذ",
                    use_container_width=True)

            if st.button("✅ طبّق",
                use_container_width=True):
                st.session_state.board=af.board.tolist()
                st.rerun()

            # أداء
            st.markdown(f"""
            <div class="perf">
            ⚡ <b>أداء:</b>
            عمق {best['depth']} •
            TT: {analysis['tt_hits']:,} hit •
            الوقت: {analysis['time']}s
            </div>""", unsafe_allow_html=True)

            # كل الحركات
            st.markdown(f"### 📊 كل الحركات ({len(mvs)})")
            rc_={1:("🥇",(50,205,50)),
                 2:("🥈",(65,105,225)),
                 3:("🥉",(255,165,0))}

            top=[]
            for i,mv in enumerate(mvs[:5]):
                _,cl=rc_.get(mv["rank"],
                    (f"#{mv['rank']}",(180,180,180)))
                top.append({"m":mv["move"],"c":cl,
                    "w":6 if i==0 else 3})
            st.image(R.draw(ba2,arrows=top),
                caption="🥇🥈🥉",
                use_container_width=True)

            for mv in mvs:
                icon=rc_.get(mv["rank"],
                    (f"#{mv['rank']}",None))[0]
                p=" → ".join(
                    f"({x[0]},{x[1]})" for x in mv["move"])
                bar="█"*max(1,int(mv["win_pct"]/5))
                with st.expander(
                    f"{icon} {p} • {mv['score']} • "
                    f"{mv['win_pct']}% • {mv['verdict']}"):
                    st.markdown(f"`{p}`")
                    st.markdown(
                        f"فوز: `{bar}` {mv['win_pct']}%")
                    st.markdown(f"عمق: {mv['depth']}")
                    mi=R.draw(ba2,
                        arrows=[{"m":mv["move"],
                            "c":(255,100,50),"w":5}],
                        hl=set(mv["move"]))
                    st.image(mi,use_container_width=True)

    st.divider()
    st.markdown(
        '<p style="text-align:center;color:#999;'
        'font-size:0.8em">'
        '♟️ v7.0 — 12 عامل + Zobrist TT + PVS + LMR + '
        'Quiescence — بدون إنترنت</p>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    app()
