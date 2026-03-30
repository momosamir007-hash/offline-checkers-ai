#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════╗
║          ♟️ مساعد الداما الشرس v6.0                   ║
║  Zobrist TT + Iterative Deepening + Quiescence        ║
║  + Killer Moves + History + PVS + Vectorized Eval     ║
╚═══════════════════════════════════════════════════════╝
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
# ثوابت
# ══════════════════════════════════════════

class P(IntEnum):
    E = 0
    L = 1
    D = 2
    LK = 3
    DK = 4

CELL = 80
BOARD_PX = CELL * 8

# جداول التقييم الموضعي
# القيم مصممة لتشجيع السيطرة على المركز والتقدم
LIGHT_POS = np.array([
    [0,  0, 0,  0, 0,  0, 0,  0],
    [0,  0, 1,  0, 1,  0, 1,  0],
    [0,  3, 0,  3, 0,  3, 0,  0],
    [0,  0, 5,  0, 5,  0, 4,  0],
    [0,  5, 0,  7, 0,  5, 0,  0],
    [0,  0, 7,  0, 7,  0, 6,  0],
    [0,  8, 0,  9, 0,  9, 0,  0],
    [0,  0, 10, 0, 10, 0, 10, 0],
], dtype=np.float32)

DARK_POS = LIGHT_POS[::-1].copy()

KING_POS = np.array([
    [0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 3, 0, 3, 0, 3, 0],
    [0, 3, 0, 5, 0, 5, 0, 3],
    [1, 0, 5, 0, 7, 0, 5, 0],
    [0, 5, 0, 7, 0, 5, 0, 1],
    [3, 0, 5, 0, 5, 0, 3, 0],
    [0, 3, 0, 3, 0, 3, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0],
], dtype=np.float32)


# ══════════════════════════════════════════
# محرك القواعد
# ══════════════════════════════════════════

class Engine:

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
                    if r < 3:
                        b[r][c] = P.D
                    elif r > 4:
                        b[r][c] = P.L
        return b

    def copy(self):
        e = Engine.__new__(Engine)
        e.board = self.board.copy()
        return e

    @staticmethod
    def is_light(p): return p in (P.L, P.LK)
    @staticmethod
    def is_dark(p): return p in (P.D, P.DK)
    @staticmethod
    def is_king(p): return p in (P.LK, P.DK)

    @staticmethod
    def owns(piece, player):
        if player in (P.L, P.LK):
            return piece in (P.L, P.LK)
        return piece in (P.D, P.DK)

    @staticmethod
    def enemy(piece, player):
        if piece == P.E: return False
        if player in (P.L, P.LK):
            return piece in (P.D, P.DK)
        return piece in (P.L, P.LK)

    @staticmethod
    def opp(player):
        return P.D if player in (P.L, P.LK) else P.L

    @staticmethod
    def dirs(piece):
        if piece == P.L:  return ((-1,-1),(-1,1))
        if piece == P.D:  return ((1,-1),(1,1))
        if piece in (P.LK, P.DK):
            return ((-1,-1),(-1,1),(1,-1),(1,1))
        return ()

    def _simple(self, r, c):
        p = self.board[r][c]
        out = []
        for dr, dc in self.dirs(p):
            nr, nc = r+dr, c+dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if self.board[nr][nc] == P.E:
                    out.append(((r,c),(nr,nc)))
        return out

    def _jumps(self, r, c, bd=None, eaten=None):
        if bd is None: bd = self.board
        if eaten is None: eaten = frozenset()
        p = bd[r][c]
        chains = []
        for dr, dc in self.dirs(p):
            mr, mc = r+dr, c+dc
            nr, nc = r+2*dr, c+2*dc
            if not (0<=nr<8 and 0<=nc<8): continue
            if bd[nr][nc] != P.E: continue
            if not self.enemy(bd[mr][mc], p): continue
            if (mr,mc) in eaten: continue
            nb = bd.copy()
            nb[nr][nc] = p; nb[r][c] = P.E; nb[mr][mc] = P.E
            promo = False
            if nr==0 and p==P.L: nb[nr][nc]=P.LK; promo=True
            elif nr==7 and p==P.D: nb[nr][nc]=P.DK; promo=True
            ne = eaten | {(mr,mc)}
            fur = [] if promo else self._jumps(nr,nc,nb,ne)
            if fur:
                for ch in fur: chains.append(((r,c),)+ch)
            else:
                chains.append(((r,c),(nr,nc)))
        return chains

    def get_moves(self, player):
        jumps, simple = [], []
        for r in range(8):
            for c in range(8):
                if self.owns(self.board[r][c], player):
                    jumps.extend(self._jumps(r,c))
                    simple.extend(self._simple(r,c))
        if jumps:
            mx = max(len(j) for j in jumps)
            return [j for j in jumps if len(j)==mx], True
        return simple, False

    def do_move(self, move):
        piece = self.board[move[0][0]][move[0][1]]
        self.board[move[0][0]][move[0][1]] = P.E
        for i in range(len(move)-1):
            sr,sc = move[i]; er,ec = move[i+1]
            dr,dc = er-sr, ec-sc
            if abs(dr)==2 and abs(dc)==2:
                self.board[sr+dr//2][sc+dc//2] = P.E
        fr,fc = move[-1]
        self.board[fr][fc] = piece
        if fr==0 and piece==P.L: self.board[fr][fc]=P.LK
        if fr==7 and piece==P.D: self.board[fr][fc]=P.DK

    def count(self, player):
        n=k=0
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if self.owns(p, player):
                    if self.is_king(p): k+=1
                    else: n+=1
        return n, k

    def game_over(self):
        ln,lk = self.count(P.L); dn,dk = self.count(P.D)
        if ln+lk==0: return P.D
        if dn+dk==0: return P.L
        lm,_ = self.get_moves(P.L); dm,_ = self.get_moves(P.D)
        if not lm and not dm: return -1
        if not lm: return P.D
        if not dm: return P.L
        return None


# ══════════════════════════════════════════
# 🔑 Zobrist Hashing
# ══════════════════════════════════════════

class ZobristHash:
    """مفاتيح عشوائية ثابتة لكل (قطعة, موقع)
    تحويل الرقعة لرقم فريد بـ O(64) XOR"""

    def __init__(self):
        rng = random.Random(777)
        self.keys = {}
        for p in range(1, 5):
            for r in range(8):
                for c in range(8):
                    self.keys[(p, r, c)] = rng.getrandbits(64)
        self.side_key = rng.getrandbits(64)

    def compute(self, board, is_max_side=True):
        h = 0
        for r in range(8):
            for c in range(8):
                p = int(board[r][c])
                if p != 0:
                    h ^= self.keys[(p, r, c)]
        if is_max_side:
            h ^= self.side_key
        return h


# ══════════════════════════════════════════
# 📦 Transposition Table
# ══════════════════════════════════════════

EXACT, LOWER, UPPER = 0, 1, 2

class TransTable:
    """جدول تحويل بسعة 1M مدخلة
    يخزن: العمق, التقييم, النوع, أفضل حركة"""

    __slots__ = ['table', 'hits', 'writes', 'max_size']

    def __init__(self, max_size=1_000_000):
        self.table = {}
        self.max_size = max_size
        self.hits = 0
        self.writes = 0

    def probe(self, key, depth, alpha, beta):
        """البحث في الجدول. يُرجع (score, move) أو None"""
        e = self.table.get(key)
        if e is None:
            return None

        if e[0] >= depth:     # e = (depth, score, flag, move)
            self.hits += 1
            s, f, m = e[1], e[2], e[3]
            if f == EXACT:
                return s, m
            if f == LOWER and s >= beta:
                return s, m
            if f == UPPER and s <= alpha:
                return s, m
        return None

    def store(self, key, depth, score, flag, move):
        # Always-replace scheme (بسيط وفعال)
        old = self.table.get(key)
        if old is None or old[0] <= depth:
            self.table[key] = (depth, score, flag, move)
            self.writes += 1
            # تنظيف عند الامتلاء
            if len(self.table) > self.max_size:
                # حذف نصف المدخلات الأقدم
                keys = list(self.table.keys())
                for k in keys[:len(keys)//2]:
                    del self.table[k]

    def get_best_move(self, key):
        """جلب أفضل حركة مخزنة (لترتيب الحركات)"""
        e = self.table.get(key)
        return e[3] if e else None

    def clear(self):
        self.table.clear()
        self.hits = 0
        self.writes = 0


# ══════════════════════════════════════════
# 🧠 المحرك الشرس
# ══════════════════════════════════════════

class BrutalAI:
    """
    محرك تحليل احترافي بـ 7 تقنيات:
    1. Zobrist Hashing + Transposition Table
    2. Iterative Deepening (بحث بالوقت)
    3. Quiescence Search (لا يتوقف أثناء الأكل)
    4. Killer Moves (حركتين قاتلتين لكل عمق)
    5. History Heuristic (أولوية للحركات الناجحة)
    6. Principal Variation Search (نافذة ضيقة)
    7. Vectorized Evaluation (تقييم سريع بـ NumPy)
    """

    def __init__(self, max_time=5.0):
        self.max_time = max_time
        self.zobrist = ZobristHash()
        self.tt = TransTable()
        self.nodes = 0
        self.q_nodes = 0
        self.start_time = 0
        self.time_up = False
        self.max_depth_reached = 0

        # Killer moves: حركتين لكل عمق
        self.killers = [[None, None] for _ in range(50)]
        # History heuristic: (from, to) → score
        self.history = {}

    # ── تقييم سريع بـ NumPy ──

    def evaluate(self, eng, player):
        """تقييم متجه (vectorized) بـ 10 عوامل"""
        b = eng.board
        opp = eng.opp(player)

        # مصفوفات القطع
        if player in (P.L, P.LK):
            my_men  = (b == P.L)
            my_king = (b == P.LK)
            op_men  = (b == P.D)
            op_king = (b == P.DK)
        else:
            my_men  = (b == P.D)
            my_king = (b == P.DK)
            op_men  = (b == P.L)
            op_king = (b == P.LK)

        mn  = int(np.sum(my_men))
        mk  = int(np.sum(my_king))
        on  = int(np.sum(op_men))
        ok_ = int(np.sum(op_king))

        my_total = mn + mk
        op_total = on + ok_
        all_total = my_total + op_total

        # فوز/خسارة
        if my_total == 0: return -99999
        if op_total == 0: return  99999

        score = 0.0

        # 1. المادة
        score += mn * 100 + mk * 330
        score -= on * 100 + ok_ * 330

        # 2. الموقع (vectorized)
        if player in (P.L, P.LK):
            score += float(np.sum(my_men * LIGHT_POS)) * 4
            score -= float(np.sum(op_men * DARK_POS)) * 4
        else:
            score += float(np.sum(my_men * DARK_POS)) * 4
            score -= float(np.sum(op_men * LIGHT_POS)) * 4

        score += float(np.sum(my_king * KING_POS)) * 3
        score -= float(np.sum(op_king * KING_POS)) * 3

        # 3. حماية الصف الخلفي
        if player in (P.L, P.LK):
            score += int(np.sum(b[7, :] == P.L)) * 12
            score -= int(np.sum(b[0, :] == P.D)) * 12
        else:
            score += int(np.sum(b[0, :] == P.D)) * 12
            score -= int(np.sum(b[7, :] == P.L)) * 12

        # 4. حرية الحركة
        my_moves, my_cap = eng.get_moves(player)
        op_moves, op_cap = eng.get_moves(opp)
        score += len(my_moves) * 5
        score -= len(op_moves) * 5

        # 5. تهديد الأكل
        if my_cap: score += 25
        if op_cap: score -= 25

        # 6. سيطرة المركز
        center = b[2:6, 2:6]
        my_center = 0
        op_center = 0
        for p_type in ([P.L, P.LK] if player in (P.L, P.LK)
                       else [P.D, P.DK]):
            my_center += int(np.sum(center == p_type))
        for p_type in ([P.D, P.DK] if player in (P.L, P.LK)
                       else [P.L, P.LK]):
            op_center += int(np.sum(center == p_type))
        score += (my_center - op_center) * 8

        # 7. ترابط القطع
        for r in range(8):
            for c in range(8):
                p = b[r][c]
                if p == P.E:
                    continue
                is_mine = eng.owns(p, player)
                allies = 0
                for dr2, dc2 in ((-1,-1),(-1,1),(1,-1),(1,1)):
                    ar, ac = r+dr2, c+dc2
                    if 0<=ar<8 and 0<=ac<8:
                        if eng.owns(b[ar][ac], p):
                            allies += 1
                if is_mine:
                    score += allies * 3
                else:
                    score -= allies * 3

        # 8. حماية من الأكل
        for r in range(8):
            for c in range(8):
                p = b[r][c]
                if p == P.E:
                    continue
                is_mine = eng.owns(p, player)
                for dr2, dc2 in ((-1,-1),(-1,1),(1,-1),(1,1)):
                    ar, ac = r+dr2, c+dc2
                    br_, bc = r-dr2, c-dc2
                    if (0<=ar<8 and 0<=ac<8
                            and 0<=br_<8 and 0<=bc<8):
                        if (eng.enemy(b[ar][ac], p)
                                and b[br_][bc] == P.E):
                            if is_mine:
                                score -= 15
                            else:
                                score += 15
                            break

        # 9. قرب الترقية
        if player in (P.L, P.LK):
            for c in range(8):
                if b[1][c] == P.L: score += 20
                if b[6][c] == P.D: score -= 20
        else:
            for c in range(8):
                if b[6][c] == P.D: score += 20
                if b[1][c] == P.L: score -= 20

        # 10. استراتيجية نهاية اللعبة
        if all_total <= 8:
            mat_diff = (mn + mk*3) - (on + ok_*3)
            if mat_diff > 0:
                score += mat_diff * 20
                score += (16 - all_total) * 8

        return score

    # ── ترتيب الحركات ──

    def order_moves(self, moves, is_cap, eng, player,
                    depth, tt_move):
        """ترتيب ذكي: TT > أكل > killer > history"""
        scored = []
        for m in moves:
            s = 0

            # أولوية 1: حركة الجدول
            if tt_move and m == tt_move:
                s += 100000
            # أولوية 2: الأكل
            elif is_cap:
                s += 50000 + len(m) * 1000
            # أولوية 3: Killer moves
            elif (depth < len(self.killers)
                  and m in self.killers[depth]):
                s += 40000
            # أولوية 4: History
            else:
                key = (m[0], m[-1])
                s += self.history.get(key, 0)

            # ترقية
            dest = m[-1]
            piece = eng.board[m[0][0]][m[0][1]]
            if dest[0] == 0 and piece == P.L:
                s += 30000
            if dest[0] == 7 and piece == P.D:
                s += 30000

            # مركز
            if 2 <= dest[0] <= 5 and 2 <= dest[1] <= 5:
                s += 100

            scored.append((s, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ── تحديث Killer + History ──

    def update_killers(self, depth, move):
        if depth >= len(self.killers):
            return
        if move != self.killers[depth][0]:
            self.killers[depth][1] = self.killers[depth][0]
            self.killers[depth][0] = move

    def update_history(self, move, depth):
        key = (move[0], move[-1])
        self.history[key] = self.history.get(key, 0) + depth*depth

    # ── فحص الوقت ──

    def check_time(self):
        if self.nodes % 2000 == 0:
            if time.time() - self.start_time >= self.max_time:
                self.time_up = True

    # ── Quiescence Search ──

    def quiescence(self, eng, alpha, beta, player,
                   original, qdepth=0):
        """بحث الهدوء: لا يتوقف أثناء سلاسل الأكل"""
        self.q_nodes += 1
        self.check_time()
        if self.time_up:
            return 0

        stand_pat = self.evaluate(eng, original)

        if qdepth >= 8:
            return stand_pat

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        moves, is_cap = eng.get_moves(player)

        # فقط أكل في Quiescence
        if not is_cap:
            return stand_pat

        for move in moves:
            child = eng.copy()
            child.do_move(move)

            score = -self.quiescence(
                child, -beta, -alpha,
                eng.opp(player), original, qdepth+1
            )

            if self.time_up:
                return 0
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # ── Alpha-Beta + PVS ──

    def alpha_beta(self, eng, depth, alpha, beta,
                   player, original, is_pv=True):
        """Alpha-Beta مع PVS + TT + Killers"""
        self.nodes += 1
        self.check_time()
        if self.time_up:
            return 0, None

        # فحص نهاية اللعبة
        result = eng.game_over()
        if result is not None:
            if result == original:
                return 99999 + depth, None
            elif result == -1:
                return 0, None
            else:
                return -99999 - depth, None

        # Quiescence في العمق 0
        if depth <= 0:
            return self.quiescence(
                eng, alpha, beta, player, original
            ), None

        # فحص TT
        is_max = (player == original)
        board_key = self.zobrist.compute(eng.board, is_max)
        tt_result = self.tt.probe(board_key, depth, alpha, beta)
        if tt_result is not None and not is_pv:
            return tt_result

        # جلب الحركات
        moves, is_cap = eng.get_moves(player)
        if not moves:
            return self.evaluate(eng, original), None

        # ترتيب الحركات
        tt_move = self.tt.get_best_move(board_key)
        moves = self.order_moves(
            moves, is_cap, eng, player, depth, tt_move
        )

        best_move = moves[0]
        best_score = float("-inf")
        orig_alpha = alpha
        searched = 0

        for i, move in enumerate(moves):
            child = eng.copy()
            child.do_move(move)
            nxt = eng.opp(player)

            # ── PVS: Principal Variation Search ──
            if searched == 0:
                # أول حركة: بحث كامل
                score = -self.alpha_beta(
                    child, depth-1, -beta, -alpha,
                    nxt, original, True
                )[0]
            else:
                # باقي الحركات: نافذة ضيقة أولاً
                score = -self.alpha_beta(
                    child, depth-1, -alpha-1, -alpha,
                    nxt, original, False
                )[0]

                # إذا فشلت النافذة الضيقة: إعادة بحث كامل
                if alpha < score < beta and not self.time_up:
                    score = -self.alpha_beta(
                        child, depth-1, -beta, -score,
                        nxt, original, True
                    )[0]

            if self.time_up:
                break

            searched += 1

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                # Beta cutoff
                if not is_cap:
                    self.update_killers(depth, move)
                    self.update_history(move, depth)
                break

        if not self.time_up:
            # تحديد نوع المدخلة
            if best_score <= orig_alpha:
                flag = UPPER
            elif best_score >= beta:
                flag = LOWER
            else:
                flag = EXACT

            self.tt.store(
                board_key, depth, best_score,
                flag, best_move
            )

        return best_score, best_move

    # ── Iterative Deepening ──

    def find_best(self, eng, player):
        """بحث متكرر بالتعميق مع تحكم بالوقت"""
        self.nodes = 0
        self.q_nodes = 0
        self.start_time = time.time()
        self.time_up = False
        self.max_depth_reached = 0
        self.killers = [[None, None] for _ in range(50)]
        self.history = {}

        best_move = None
        best_score = 0
        depth_scores = []

        # بحث تدريجي من العمق 1 حتى نفاد الوقت
        for depth in range(1, 40):
            score, move = self.alpha_beta(
                eng, depth,
                float("-inf"), float("inf"),
                player, player, True
            )

            if self.time_up:
                break

            if move is not None:
                best_move = move
                best_score = score
                self.max_depth_reached = depth

                elapsed = time.time() - self.start_time
                nps = int(self.nodes / max(elapsed, 0.001))
                depth_scores.append({
                    "depth": depth,
                    "score": round(score, 1),
                    "time": round(elapsed, 2),
                    "nodes": self.nodes,
                    "nps": nps,
                })

            # إذا وجدنا فوز أكيد أو استهلكنا 80% من الوقت
            if (abs(best_score) > 90000
                    or time.time() - self.start_time
                    >= self.max_time * 0.8):
                break

        elapsed = time.time() - self.start_time
        return {
            "move": best_move,
            "score": round(best_score, 1),
            "nodes": self.nodes,
            "q_nodes": self.q_nodes,
            "time": round(elapsed, 2),
            "depth": self.max_depth_reached,
            "nps": int(self.nodes / max(elapsed, 0.001)),
            "tt_hits": self.tt.hits,
            "tt_writes": self.tt.writes,
            "depth_log": depth_scores,
        }

    # ── تحليل كل الحركات ──

    def analyze_all(self, eng, player):
        """تحليل وتصنيف كل الحركات"""
        self.tt.clear()
        self.nodes = 0
        self.q_nodes = 0
        self.start_time = time.time()
        self.time_up = False

        moves, is_cap = eng.get_moves(player)
        if not moves:
            return None

        # تقسيم الوقت بالتساوي على الحركات
        time_per = max(1.0, self.max_time * 0.7 / len(moves))

        results = []
        for move in moves:
            child = eng.copy()
            child.do_move(move)

            # تحليل عميق لكل حركة
            sub_ai = BrutalAI(max_time=time_per)
            sub_ai.tt = self.tt  # مشاركة الجدول
            opp_result = sub_ai.find_best(child, eng.opp(player))

            # النتيجة = سالب أفضل رد للخصم
            score = -opp_result["score"]
            depth_reached = opp_result["depth"]

            is_capture = (
                len(move) > 2
                or (len(move)==2
                    and abs(move[0][0]-move[1][0])==2)
            )

            cap_count = 0
            if is_capture:
                for i in range(len(move)-1):
                    if abs(move[i][0]-move[i+1][0])==2:
                        cap_count += 1

            dest = move[-1]
            piece = eng.board[move[0][0]][move[0][1]]
            promotes = (
                (dest[0]==0 and piece==P.L)
                or (dest[0]==7 and piece==P.D)
            )

            # تحليل الخطورة
            danger = ""
            if is_capture and cap_count >= 2:
                danger = "🔥 أكل متعدد ممتاز!"
            elif promotes:
                danger = "👑 ترقية للملك!"
            elif score > 200:
                danger = "💪 تفوق واضح"
            elif score > 50:
                danger = "✅ جيدة"
            elif score > -50:
                danger = "⚖️ متكافئة"
            elif score > -200:
                danger = "⚠️ محفوفة بالمخاطر"
            else:
                danger = "❌ خطيرة"

            results.append({
                "move": move,
                "score": round(score, 1),
                "depth": depth_reached,
                "is_capture": is_capture,
                "captured": cap_count,
                "promotes": promotes,
                "piece": int(piece),
                "verdict": danger,
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        best_s = results[0]["score"] if results else 0
        for i, r in enumerate(results):
            r["rank"] = i + 1
            s = r["score"]
            if s > 5000:     r["win_pct"] = 99
            elif s < -5000:  r["win_pct"] = 1
            else:
                r["win_pct"] = max(1, min(99,
                    int(50 + s / 15)))

        elapsed = time.time() - self.start_time
        pos_eval = self.evaluate(eng, player)

        return {
            "moves": results,
            "time": round(elapsed, 2),
            "is_forced_capture": is_cap,
            "position_eval": round(pos_eval, 1),
            "tt_hits": self.tt.hits,
        }


# ══════════════════════════════════════════
# تحليل الصور
# ══════════════════════════════════════════

class Vision:

    @staticmethod
    def fix_perspective(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 30, 100)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        edges = cv2.dilate(edges, k, iterations=2)
        cnts, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            return cv2.resize(img_bgr, (400,400)), False

        largest = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < img_bgr.shape[0]*img_bgr.shape[1]*0.15:
            return cv2.resize(img_bgr, (400,400)), False

        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02*peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4,2).astype(np.float32)
            s = pts.sum(axis=1); d = np.diff(pts,axis=1).ravel()
            o = np.zeros((4,2), dtype=np.float32)
            o[0]=pts[np.argmin(s)]; o[2]=pts[np.argmax(s)]
            o[1]=pts[np.argmin(d)]; o[3]=pts[np.argmax(d)]
            dst = np.float32([[0,0],[399,0],[399,399],[0,399]])
            M = cv2.getPerspectiveTransform(o, dst)
            return cv2.warpPerspective(img_bgr, M, (400,400)), True

        x,y,w,h = cv2.boundingRect(largest)
        return cv2.resize(img_bgr[y:y+h,x:x+w], (400,400)), False

    @staticmethod
    def detect_hsv(img, lt=160, dt=100):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = 50; board = np.zeros((8,8), dtype=np.int8); info=[]
        for r in range(8):
            for col in range(8):
                if (r+col)%2==0: continue
                m=c//4
                roi_g = gray[r*c+m:(r+1)*c-m, col*c+m:(col+1)*c-m]
                roi_h = hsv[r*c+m:(r+1)*c-m, col*c+m:(col+1)*c-m]
                br=float(np.mean(roi_g))
                sa=float(np.mean(roi_h[:,:,1]))
                va=float(np.var(roi_g))
                det=int(P.E)
                if va>120 or sa>30:
                    if br>lt: det=int(P.L)
                    elif br<dt: det=int(P.D)
                board[r][col]=det
                info.append({"r":r,"c":col,"br":round(br),
                             "sa":round(sa),"va":round(va),"d":det})
        return board, info

    @staticmethod
    def detect_circles(img, lt=160, dt=100):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5); c=50
        circles = cv2.HoughCircles(
            gray,cv2.HOUGH_GRADIENT,1.2,35,
            param1=60,param2=35,minRadius=12,maxRadius=24
        )
        board = np.zeros((8,8),dtype=np.int8)
        vis = img.copy()
        if circles is not None:
            for cx,cy,rad in np.uint16(np.around(circles))[0]:
                col_=int(cx/c); row_=int(cy/c)
                if not(0<=row_<8 and 0<=col_<8): continue
                if (row_+col_)%2==0: continue
                s = gray[max(0,int(cy)-5):min(400,int(cy)+5),
                         max(0,int(cx)-5):min(400,int(cx)+5)]
                if s.size==0: continue
                a=float(np.mean(s))
                if a>lt: board[row_][col_]=int(P.L); clr=(0,255,0)
                elif a<dt: board[row_][col_]=int(P.D); clr=(0,0,255)
                else: continue
                cv2.circle(vis,(int(cx),int(cy)),int(rad),clr,2)
        return board, Image.fromarray(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB))

    @staticmethod
    def merge(a, b):
        m = np.zeros((8,8),dtype=np.int8)
        for r in range(8):
            for c in range(8):
                if a[r][c]!=P.E and b[r][c]!=P.E and a[r][c]==b[r][c]:
                    m[r][c]=a[r][c]
                elif b[r][c]!=P.E: m[r][c]=b[r][c]
                elif a[r][c]!=P.E: m[r][c]=a[r][c]
        return m


# ══════════════════════════════════════════
# رسم الرقعة
# ══════════════════════════════════════════

class Render:
    @staticmethod
    def draw(board, arrows=None, highlight=None, label=None):
        img = Image.new("RGB", (BOARD_PX, BOARD_PX))
        dr = ImageDraw.Draw(img)
        for r in range(8):
            for c in range(8):
                x1,y1 = c*CELL, r*CELL; x2,y2 = x1+CELL, y1+CELL
                sq = (235,215,180) if (r+c)%2==0 else (175,130,95)
                if highlight and (r,c) in highlight:
                    sq = (100,200,100)
                dr.rectangle([x1,y1,x2,y2], fill=sq)
                p = board[r][c]
                if p == P.E: continue
                cx,cy = x1+CELL//2, y1+CELL//2; pr=CELL//2-10
                dr.ellipse([cx-pr+3,cy-pr+3,cx+pr+3,cy+pr+3],
                           fill=(70,50,30))
                fl = (250,248,240) if Engine.is_light(p) else (45,45,45)
                ed = (195,185,170) if Engine.is_light(p) else (25,25,25)
                dr.ellipse([cx-pr,cy-pr,cx+pr,cy+pr],
                           fill=fl, outline=ed, width=2)
                dr.ellipse([cx-pr+5,cy-pr+5,cx+pr-5,cy+pr-5],
                           outline=ed, width=1)
                if Engine.is_king(p):
                    kr=12
                    dr.ellipse([cx-kr,cy-kr,cx+kr,cy+kr],
                               fill=(255,215,0),
                               outline=(200,170,0), width=2)
        for i in range(8):
            try:
                dr.text((3,i*CELL+3),str(i),fill=(130,110,90))
                dr.text((i*CELL+CELL//2-4,BOARD_PX-14),
                        chr(65+i),fill=(130,110,90))
            except: pass
        if arrows:
            for a in arrows:
                Render._arrow(dr, a["move"],
                              a.get("color",(255,50,50)),
                              a.get("width",5))
        if label:
            try:
                dr.rectangle([0,0,BOARD_PX,22],fill=(0,0,0))
                dr.text((5,3),label,fill=(255,255,255))
            except: pass
        return img

    @staticmethod
    def _arrow(draw, move, color, width):
        if not move or len(move)<2: return
        for i in range(len(move)-1):
            sr,sc=move[i]; er,ec=move[i+1]
            sx,sy = sc*CELL+CELL//2, sr*CELL+CELL//2
            ex,ey = ec*CELL+CELL//2, er*CELL+CELL//2
            draw.line([(sx,sy),(ex,ey)], fill=color, width=width)
            draw.ellipse([ex-8,ey-8,ex+8,ey+8], fill=color)
        sr,sc=move[0]; sx,sy=sc*CELL+CELL//2,sr*CELL+CELL//2
        draw.ellipse([sx-12,sy-12,sx+12,sy+12],
                     outline=(0,220,0), width=4)


# ══════════════════════════════════════════
# واجهة Streamlit
# ══════════════════════════════════════════

def main():
    st.set_page_config("♟️ مساعد الداما الشرس","♟️",layout="wide")

    st.markdown("""<style>
    .block-container{max-width:1100px}
    .best-box{background:linear-gradient(135deg,#28a745,#20c997);
        color:#fff;padding:18px;border-radius:12px;
        text-align:center;margin:10px 0}
    .best-box h2{margin:0;font-size:1.6em}
    .best-box p{margin:5px 0;font-size:1.1em}
    .warn-box{background:#f8d7da;border:2px solid #dc3545;
        color:#721c24;padding:12px;border-radius:10px;
        text-align:center;margin:8px 0;font-weight:bold}
    .eval-bar{background:#e9ecef;border-radius:8px;
        overflow:hidden;height:30px;margin:8px 0}
    .eval-fill{height:100%;text-align:center;color:#fff;
        font-weight:bold;line-height:30px;border-radius:8px}
    .perf-box{background:#f0f2f6;padding:12px;border-radius:10px;
        border-left:4px solid #667eea;margin:8px 0}
    </style>""", unsafe_allow_html=True)

    st.title("♟️ مساعد الداما الشرس")
    st.caption("حمّل صورة أو أدخل يدوياً → AI يحلل بـ 7 تقنيات متقدمة")

    if "board" not in st.session_state:
        st.session_state.board = Engine._init().tolist()

    # ═══ الشريط الجانبي ═══
    with st.sidebar:
        st.header("⚙️ الإعدادات")

        my_color = st.radio("♟️ لون قطعك:",
                             ["⚪ الفاتح","⚫ الداكن"])
        player = P.L if "الفاتح" in my_color else P.D

        st.markdown("**⏱ وقت التحليل:**")
        think_time = st.select_slider("ثوانٍ:",
            [2, 3, 5, 8, 10, 15, 20, 30],
            value=5)

        st.info(f"⏱ {think_time}s — AI يبحث أعمق ما يمكن")

        st.divider()
        if st.button("🔄 رقعة ابتدائية", use_container_width=True):
            st.session_state.board = Engine._init().tolist()
            st.rerun()
        if st.button("🗑️ مسح", use_container_width=True):
            st.session_state.board = np.zeros((8,8),dtype=int).tolist()
            st.rerun()

        st.divider()
        eng_ = Engine(st.session_state.board)
        ln,lk = eng_.count(P.L); dn,dk = eng_.count(P.D)
        st.markdown("### 📊 القطع")
        c1,c2 = st.columns(2)
        with c1:
            st.metric("⚪",ln+lk,delta=f"👑{lk}" if lk else None)
        with c2:
            st.metric("⚫",dn+dk,delta=f"👑{dk}" if dk else None)

    # ═══ التبويبات ═══
    tabs = ["📥 إدخال الرقعة", "🧠 التحليل"]
    if HAS_CV2: tabs.insert(1, "📷 تحليل صورة")
    active = st.tabs(tabs)

    # ─── إدخال يدوي ───
    with active[0]:
        st.markdown("**اختر قطعة واضغط المربع:**")
        opts = {"⬜ فارغ":int(P.E),"⚪ فاتح":int(P.L),
                "⚫ داكن":int(P.D),"👑W":int(P.LK),"♛B":int(P.DK)}
        sel = st.radio("_",list(opts.keys()),horizontal=True,
                        label_visibility="collapsed")
        sv = opts[sel]
        syms = {int(P.E):"·",int(P.L):"⚪",int(P.D):"⚫",
                int(P.LK):"👑",int(P.DK):"♛"}

        ba = np.array(st.session_state.board)
        for r in range(8):
            cols = st.columns(8)
            for c in range(8):
                with cols[c]:
                    ok = (r+c)%2!=0
                    s = syms.get(int(ba[r][c]),"·") if ok else ""
                    if st.button(s,key=f"m{r}{c}",
                                 use_container_width=True,
                                 disabled=not ok):
                        st.session_state.board[r][c]=sv
                        st.rerun()

        vis = Render.draw(ba)
        st.image(vis,caption="الرقعة الحالية",
                 use_container_width=True)

    # ─── تحليل صورة ───
    if HAS_CV2:
        with active[1]:
            st.subheader("📷 تحليل صورة الرقعة")
            uploaded = st.file_uploader("📸",type=["jpg","png","jpeg"])
            if uploaded:
                pil = Image.open(uploaded).convert("RGB")
                img_cv = cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)
                c1,c2 = st.columns(2)
                with c1: st.image(pil,caption="الأصلية",
                                   use_container_width=True)
                with st.spinner("🔲 تصحيح المنظور..."):
                    fixed, was = Vision.fix_perspective(img_cv)
                fp = Image.fromarray(cv2.cvtColor(fixed,cv2.COLOR_BGR2RGB))
                with c2:
                    st.image(fp,caption="✅ مصحح" if was else "📐 مقتصة",
                             use_container_width=True)
                tc1,tc2 = st.columns(2)
                with tc1: lt=st.slider("عتبة الفاتح",100,230,160)
                with tc2: dt=st.slider("عتبة الداكن",30,150,100)

                if st.button("🔍 تحليل",type="primary"):
                    with st.spinner("🧠 تحليل متقدم..."):
                        hb,hi = Vision.detect_hsv(fixed,lt,dt)
                        cb,cv_ = Vision.detect_circles(fixed,lt,dt)
                        merged = Vision.merge(hb,cb)
                    st.success("✅ تم!")
                    t1,t2,t3 = st.tabs(["HSV","دوائر","مدمج"])
                    with t1:
                        st.image(Render.draw(hb),use_container_width=True)
                        with st.expander("تفاصيل"):
                            for d in hi:
                                if d["d"]!=0:
                                    st.text(f"({d['r']},{d['c']}) "
                                            f"{'⚪' if d['d']==1 else '⚫'} "
                                            f"br={d['br']} var={d['va']}")
                    with t2:
                        st.image(cv_,use_container_width=True)
                        st.image(Render.draw(cb),use_container_width=True)
                    with t3:
                        st.image(Render.draw(merged),use_container_width=True)
                        e=Engine(merged); a,b2=e.count(P.L); c3,d2=e.count(P.D)
                        st.info(f"⚪ {a}+{b2}👑 • ⚫ {c3}+{d2}👑")
                        if st.button("📥 استخدم للتحليل",type="primary"):
                            st.session_state.board=merged.tolist()
                            st.rerun()

    # ─── التحليل ───
    analyze_idx = 2 if HAS_CV2 else 1
    with active[analyze_idx]:
        ba2 = np.array(st.session_state.board, dtype=np.int8)
        eng = Engine(ba2)
        st.image(Render.draw(ba2),caption="الرقعة",
                 use_container_width=True)

        ln,lk = eng.count(P.L); dn,dk = eng.count(P.D)
        if (ln+lk)==0 and (dn+dk)==0:
            st.warning("⚠️ الرقعة فارغة!")
            return

        go = eng.game_over()
        if go is not None:
            if go==-1: st.info("🤝 تعادل")
            elif go==P.L: st.success("🏆 فاز الفاتح!")
            else: st.success("🏆 فاز الداكن!")
            return

        emoji = "⚪" if player==P.L else "⚫"
        st.markdown(f"### {emoji} تحليل حركات "
                    f"**{'الفاتح' if player==P.L else 'الداكن'}**")

        if st.button("🧠 حلّل الآن!",type="primary",
                     use_container_width=True):

            progress = st.empty()
            progress.info(f"🧠 جاري التحليل لمدة {think_time} ثانية...")

            t0 = time.time()
            ai = BrutalAI(max_time=think_time)
            analysis = ai.analyze_all(eng, player)
            total_time = round(time.time() - t0, 2)

            progress.empty()

            if not analysis or not analysis["moves"]:
                st.error("❌ لا حركات!")
                return

            moves = analysis["moves"]
            best = moves[0]

            # ── شريط التقييم ──
            pe = analysis["position_eval"]
            if pe > 200:
                em,ec = "🟢 متفوق بوضوح!","#28a745"
            elif pe > 50:
                em,ec = "🟢 أفضل قليلاً","#20c997"
            elif pe > -50:
                em,ec = "🟡 متكافئة","#ffc107"
            elif pe > -200:
                em,ec = "🟠 الخصم أفضل","#fd7e14"
            else:
                em,ec = "🔴 خطر!","#dc3545"
            pct = max(5,min(95,int(50+pe/15)))
            st.markdown(f'<div class="eval-bar">'
                        f'<div class="eval-fill" '
                        f'style="width:{pct}%;background:{ec}">'
                        f'{em} ({pe})</div></div>',
                        unsafe_allow_html=True)

            # ── الأكل الإجباري ──
            if analysis["is_forced_capture"]:
                st.markdown('<div class="warn-box">'
                            '⚡ أكل إجباري!</div>',
                            unsafe_allow_html=True)

            # ── أفضل حركة ──
            path = " → ".join(f"({p[0]},{p[1]})" for p in best["move"])
            extras = []
            if best["is_capture"]:
                extras.append(f"💥 أكل ×{best['captured']}")
            if best["promotes"]:
                extras.append("👑 ترقية!")
            extras.append(best["verdict"])
            ext = " • ".join(extras)

            st.markdown(f"""
            <div class="best-box">
                <h2>🏆 أفضل حركة</h2>
                <p style="font-size:1.5em">{path}</p>
                <p>{ext}</p>
                <p>تقييم: {best['score']} •
                   فوز: {best['win_pct']}% •
                   عمق: {best['depth']}</p>
            </div>""", unsafe_allow_html=True)

            # ── صور ──
            c1,c2 = st.columns(2)
            with c1:
                bi = Render.draw(ba2,
                    arrows=[{"move":best["move"],
                             "color":(50,205,50),"width":6}],
                    highlight=set(best["move"]),
                    label="BEST MOVE")
                st.image(bi,caption="🏆 أفضل حركة",
                         use_container_width=True)
            with c2:
                af = eng.copy(); af.do_move(best["move"])
                st.image(Render.draw(af.board),
                         caption="📋 بعد التنفيذ",
                         use_container_width=True)

            if st.button("✅ طبّق الحركة",use_container_width=True):
                st.session_state.board = af.board.tolist()
                st.rerun()

            # ── إحصائيات الأداء ──
            st.markdown(f"""
            <div class="perf-box">
            ⚡ <b>أداء المحرك:</b>
            الوقت: {total_time}s •
            TT hits: {analysis['tt_hits']:,} •
            أفضل عمق: {best['depth']}
            </div>""", unsafe_allow_html=True)

            # ── كل الحركات ──
            st.markdown(f"### 📊 تصنيف الحركات ({len(moves)})")

            rank_colors = {
                1:("🥇",(50,205,50)),
                2:("🥈",(65,105,225)),
                3:("🥉",(255,165,0)),
            }

            # صورة أعلى 5
            top_arrows = []
            for i,mv in enumerate(moves[:5]):
                _,clr = rank_colors.get(mv["rank"],
                    (f"#{mv['rank']}",(180,180,180)))
                top_arrows.append({"move":mv["move"],
                    "color":clr,"width":6 if i==0 else 3})
            all_img = Render.draw(ba2, arrows=top_arrows)
            st.image(all_img,
                     caption="🥇أخضر 🥈أزرق 🥉برتقالي",
                     use_container_width=True)

            for mv in moves:
                icon = rank_colors.get(mv["rank"],
                    (f"#{mv['rank']}",None))[0]
                path = " → ".join(f"({p[0]},{p[1]})"
                                  for p in mv["move"])
                bar = "█" * max(1, int(mv["win_pct"]/5))

                with st.expander(
                    f"{icon} {path} • {mv['score']} • "
                    f"{mv['win_pct']}% • {mv['verdict']}"
                ):
                    st.markdown(f"**المسار:** `{path}`")
                    st.markdown(f"**الحكم:** {mv['verdict']}")
                    st.markdown(f"**فرصة الفوز:** `{bar}` "
                                f"{mv['win_pct']}%")
                    st.markdown(f"**عمق التحليل:** {mv['depth']}")

                    mi = Render.draw(ba2,
                        arrows=[{"move":mv["move"],
                                 "color":(255,100,50),"width":5}],
                        highlight=set(mv["move"]))
                    st.image(mi, use_container_width=True)

    st.divider()
    st.markdown('<p style="text-align:center;color:#999;'
                'font-size:0.8em">'
                '♟️ v6.0 — Zobrist TT + Iterative Deepening + '
                'Quiescence + PVS + Killer Moves — '
                'يعمل بدون إنترنت</p>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
