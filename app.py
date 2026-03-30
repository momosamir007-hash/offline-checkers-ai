#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════╗
║ ♟️ مساعد الداما الذكي — النسخة النهائية           ║
║ pydraughts (قواعد مثالية) + Minimax+αβ (تحليل)   ║
║ OpenCV (رؤية) + Streamlit (واجهة)                ║
╚═══════════════════════════════════════════════════╝
"""
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time
from enum import IntEnum

try:
    from draughts import Board as DrBoard
    HAS_DRAUGHTS = True
except ImportError:
    HAS_DRAUGHTS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ══════════════════════════════════════════
# 1. ثوابت
# ══════════════════════════════════════════
class P(IntEnum):
    E = 0   # فارغ
    W = 1   # أبيض (فاتح)
    B = 2   # أسود (داكن)
    WK = 3  # ملك أبيض
    BK = 4  # ملك أسود

CELL = 80
BOARD_PX = CELL * 8

# ══════════════════════════════════════════
# 2. تحويل الإحداثيات
# pydraughts يرقم المربعات 1-32
# نحن نستخدم شبكة (صف, عمود) 8×8
# ══════════════════════════════════════════
def sq_to_rc(sq):
    """مربع (1-32) → (صف, عمود)"""
    s = sq - 1
    row = s // 4
    pos = s % 4
    col = pos * 2 + (1 if row % 2 == 0 else 0)
    return row, col

def rc_to_sq(row, col):
    """(صف, عمود) → مربع (1-32)"""
    if row % 2 == 0:
        pos = (col - 1) // 2
    else:
        pos = col // 2
    return row * 4 + pos + 1

def grid_to_fen(grid, turn_white=True):
    """مصفوفة 8×8 → FEN نص لـ pydraughts"""
    wp, bp = [], []
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 0:
                continue
            p = int(grid[r][c])
            sq = rc_to_sq(r, c)
            if p == P.W:
                wp.append(str(sq))
            elif p == P.WK:
                wp.append(f"K{sq}")
            elif p == P.B:
                bp.append(str(sq))
            elif p == P.BK:
                bp.append(f"K{sq}")
    t = "W" if turn_white else "B"
    w_str = ",".join(wp) if wp else ""
    b_str = ",".join(bp) if bp else ""
    return f"{t}:W{w_str}:B{b_str}"

def fen_to_grid(fen):
    """FEN نص → مصفوفة 8×8"""
    grid = np.zeros((8, 8), dtype=np.int8)
    parts = fen.split(":")
    for part in parts[1:]:
        if not part:
            continue
        color = part[0]  # W or B
        squares_str = part[1:]
        if not squares_str:
            continue
        for sq_str in squares_str.split(","):
            sq_str = sq_str.strip()
            if not sq_str:
                continue
            is_king = sq_str.startswith("K")
            sq_num = int(sq_str[1:] if is_king else sq_str)
            r, c = sq_to_rc(sq_num)
            if color == "W":
                grid[r][c] = P.WK if is_king else P.W
            else:
                grid[r][c] = P.BK if is_king else P.B
    turn_white = fen.startswith("W")
    return grid, turn_white

# ══════════════════════════════════════════
# 3. واجهة اللعبة الموحدة
# تستخدم pydraughts إذا متوفر
# وإلا تستخدم محرك احتياطي
# ══════════════════════════════════════════
class GameInterface:
    """واجهة موحدة للعبة الداما.
       تستخدم pydraughts تلقائياً (قواعد مثالية) مع محرك احتياطي إذا لم تكن المكتبة مثبتة.
    """
    def __init__(self, grid=None, turn_white=True):
        self._grid = (
            grid.copy() if grid is not None else self._initial_grid()
        )
        self._turn_white = turn_white
        self._use_dr = HAS_DRAUGHTS
        self._dr_board = None
        self._move_stack = []
        if self._use_dr:
            self._sync_to_draughts()

    @staticmethod
    def _initial_grid():
        b = np.zeros((8, 8), dtype=np.int8)
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 != 0:
                    if r < 3:
                        b[r][c] = P.B
                    elif r > 4:
                        b[r][c] = P.W
        return b

    def copy(self):
        g = GameInterface.__new__(GameInterface)
        g._grid = self._grid.copy()
        g._turn_white = self._turn_white
        g._use_dr = self._use_dr
        g._move_stack = []
        if g._use_dr:
            g._sync_to_draughts()
        return g

    # ── pydraughts sync ──
    def _sync_to_draughts(self):
        """مزامنة مصفوفتنا مع pydraughts"""
        if not self._use_dr:
            return
        try:
            fen = grid_to_fen(self._grid, self._turn_white)
            self._dr_board = DrBoard(variant="english", fen=fen)
        except Exception:
            self._use_dr = False
            self._dr_board = None

    def _sync_from_draughts(self):
        """قراءة حالة pydraughts إلى مصفوفتنا"""
        if not self._dr_board:
            return
        try:
            fen = self._dr_board.fen
            self._grid, self._turn_white = fen_to_grid(fen)
        except Exception:
            pass

    # ── الحركات ──
    def get_moves(self):
        """كل الحركات القانونية.
           يُرجع: قائمة من tuples: ((r1,c1), (r2,c2), ...) + هل هي أكل إجباري
        """
        if self._use_dr and self._dr_board:
            return self._get_moves_draughts()
        return self._get_moves_fallback()

    def _get_moves_draughts(self):
        """حركات من pydraughts (مثالية)"""
        try:
            legal = self._dr_board.legal_moves()
            if not legal:
                return [], False
            moves = []
            is_capture = False
            for mv in legal:
                # تحويل خطوات الحركة لإحداثيات (r,c)
                steps = mv.steps_move
                rc_steps = tuple(sq_to_rc(s) for s in steps)
                moves.append(rc_steps)
                # كشف الأكل: المسافة > 1 بين خطوتين متتاليتين
                if len(steps) >= 2:
                    r1, c1 = sq_to_rc(steps[0])
                    r2, c2 = sq_to_rc(steps[1])
                    if abs(r1 - r2) == 2:
                        is_capture = True
            return moves, is_capture
        except Exception:
            return self._get_moves_fallback()

    def do_move(self, move_rc):
        """تنفيذ حركة بإحداثيات (r,c)"""
        # حفظ للتراجع
        self._move_stack.append((self._grid.copy(), self._turn_white))
        if self._use_dr and self._dr_board:
            try:
                # تحويل لأرقام مربعات
                sqs = [rc_to_sq(r, c) for r, c in move_rc]
                # البحث عن الحركة المطابقة في pydraughts
                for mv in self._dr_board.legal_moves():
                    if mv.steps_move == sqs:
                        self._dr_board.push(mv)
                        self._sync_from_draughts()
                        return True
                # fallback: محاولة بأول وآخر مربع
                for mv in self._dr_board.legal_moves():
                    if (mv.steps_move[0] == sqs[0] and
                        mv.steps_move[-1] == sqs[-1]):
                        self._dr_board.push(mv)
                        self._sync_from_draughts()
                        return True
            except Exception:
                pass
        # Fallback: تنفيذ يدوي
        self._do_move_manual(move_rc)
        return True

    def undo_move(self):
        """التراجع عن آخر حركة"""
        if self._move_stack:
            self._grid, self._turn_white = self._move_stack.pop()
            if self._use_dr:
                self._sync_to_draughts()
            return True
        return False

    def is_over(self):
        if self._use_dr and self._dr_board:
            try:
                return self._dr_board.is_over()
            except Exception:
                pass
        # Fallback
        moves, _ = self.get_moves()
        return len(moves) == 0

    def winner(self):
        """1=أبيض فاز, 2=أسود فاز, 0=تعادل, None=مستمرة"""
        if self._use_dr and self._dr_board:
            try:
                if not self._dr_board.is_over():
                    return None
                w = self._dr_board.winner()
                if w == 2:
                    return 1   # White
                elif w == 1:
                    return 2   # Black
                return 0       # Draw
            except Exception:
                pass
        # Fallback
        wn, wk = self.count(P.W)
        bn, bk = self.count(P.B)
        if wn + wk == 0:
            return 2
        if bn + bk == 0:
            return 1
        moves, _ = self.get_moves()
        if not moves:
            return 2 if self._turn_white else 1
        return None

    @property
    def grid(self):
        return self._grid

    @property
    def turn_white(self):
        return self._turn_white

    def count(self, color):
        """عدد القطع العادية والملوك"""
        n = k = 0
        g = self._grid
        if color == P.W:
            n = int(np.sum(g == P.W))
            k = int(np.sum(g == P.WK))
        else:
            n = int(np.sum(g == P.B))
            k = int(np.sum(g == P.BK))
        return n, k

    # ── Fallback الاحتياطي ──
    @staticmethod
    def _is_own(piece, is_white):
        if is_white:
            return piece in (P.W, P.WK)
        return piece in (P.B, P.BK)

    @staticmethod
    def _is_enemy(piece, is_white):
        if piece == P.E:
            return False
        if is_white:
            return piece in (P.B, P.BK)
        return piece in (P.W, P.WK)

    @staticmethod
    def _dirs(piece):
        if piece == P.W:
            return ((-1, -1), (-1, 1))
        if piece == P.B:
            return ((1, -1), (1, 1))
        if piece in (P.WK, P.BK):
            return ((-1, -1), (-1, 1), (1, -1), (1, 1))
        return ()

    def _get_moves_fallback(self):
        g = self._grid
        tw = self._turn_white
        jumps, simple = [], []
        for r in range(8):
            for c in range(8):
                p = g[r][c]
                if not self._is_own(p, tw):
                    continue
                jumps.extend(self._find_jumps(r, c, g))
                for dr, dc in self._dirs(p):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        if g[nr][nc] == P.E:
                            simple.append(((r, c), (nr, nc)))
        if jumps:
            mx = max(len(j) for j in jumps)
            return ([j for j in jumps if len(j) == mx], True)
        return simple, False

    def _find_jumps(self, r, c, bd=None, eaten=None):
        if bd is None:
            bd = self._grid
        if eaten is None:
            eaten = frozenset()
        p = bd[r][c]
        chains = []
        for dr, dc in self._dirs(p):
            mr, mc = r + dr, c + dc
            nr, nc = r + 2 * dr, c + 2 * dc
            if not (0 <= nr < 8 and 0 <= nc < 8):
                continue
            if bd[nr][nc] != P.E:
                continue
            if not self._is_enemy(bd[mr][mc], self._turn_white):
                continue
            if (mr, mc) in eaten:
                continue
            nb = bd.copy()
            nb[nr][nc] = p
            nb[r][c] = P.E
            nb[mr][mc] = P.E
            promo = False
            if nr == 0 and p == P.W:
                nb[nr][nc] = P.WK
                promo = True
            elif nr == 7 and p == P.B:
                nb[nr][nc] = P.BK
                promo = True
            ne = eaten | {(mr, mc)}
            fur = [] if promo else self._find_jumps(nr, nc, nb, ne)
            if fur:
                for ch in fur:
                    chains.append(((r, c),) + ch)
            else:
                chains.append(((r, c), (nr, nc)))
        return chains

    def _do_move_manual(self, move):
        g = self._grid
        piece = g[move[0][0]][move[0][1]]
        g[move[0][0]][move[0][1]] = P.E
        for i in range(len(move) - 1):
            sr, sc = move[i]
            er, ec = move[i + 1]
            dr, dc = er - sr, ec - sc
            if abs(dr) == 2 and abs(dc) == 2:
                g[sr + dr // 2][sc + dc // 2] = P.E
        fr, fc = move[-1]
        g[fr][fc] = piece
        if fr == 0 and piece == P.W:
            g[fr][fc] = P.WK
        if fr == 7 and piece == P.B:
            g[fr][fc] = P.BK
        self._turn_white = not self._turn_white

# ══════════════════════════════════════════
# 4. الذكاء الاصطناعي
# ══════════════════════════════════════════
# جداول التقييم الموضعي
W_POS = np.array([
    [0,0,0,0,0,0,0,0],[0,0,1,0,1,0,1,0],
    [0,3,0,3,0,3,0,0],[0,0,5,0,5,0,4,0],
    [0,5,0,7,0,5,0,0],[0,0,7,0,7,0,6,0],
    [0,8,0,9,0,9,0,0],[0,0,10,0,10,0,10,0],
], dtype=np.float32)

B_POS = W_POS[::-1].copy()

K_POS = np.array([
    [0,1,0,1,0,1,0,1],[1,0,3,0,3,0,3,0],
    [0,3,0,5,0,5,0,3],[1,0,5,0,7,0,5,0],
    [0,5,0,7,0,5,0,1],[3,0,5,0,5,0,3,0],
    [0,3,0,3,0,3,0,1],[1,0,1,0,1,0,1,0],
], dtype=np.float32)

class AI:
    """Minimax + Alpha-Beta + PVS + Quiescence"""
    def __init__(self, max_time=3.0):
        self.max_time = max_time
        self.nodes = 0
        self.t0 = 0
        self.stopped = False
        self.best_depth = 0

    def _check(self):
        if self.nodes % 500 == 0:
            if time.time() - self.t0 >= self.max_time:
                self.stopped = True

    def evaluate(self, game, for_white):
        """تقييم سريع بـ NumPy"""
        g = game.grid
        wm = (g == P.W).astype(np.float32)
        wk = (g == P.WK).astype(np.float32)
        bm = (g == P.B).astype(np.float32)
        bk = (g == P.BK).astype(np.float32)

        wn = int(np.sum(wm))
        wkn = int(np.sum(wk))
        bn = int(np.sum(bm))
        bkn = int(np.sum(bk))

        if for_white:
            if wn + wkn == 0:
                return -99999
            if bn + bkn == 0:
                return 99999
        else:
            if bn + bkn == 0:
                return -99999
            if wn + wkn == 0:
                return 99999

        # مادة
        w_score = wn * 100 + wkn * 330
        b_score = bn * 100 + bkn * 330

        # موقع
        w_score += float(np.sum(wm * W_POS)) * 4
        w_score += float(np.sum(wk * K_POS)) * 3
        b_score += float(np.sum(bm * B_POS)) * 4
        b_score += float(np.sum(bk * K_POS)) * 3

        # صف خلفي
        w_score += int(np.sum(g[7, :] == P.W)) * 12
        b_score += int(np.sum(g[0, :] == P.B)) * 12

        # مركز
        center = g[2:6, 2:6]
        for v in (P.W, P.WK):
            w_score += int(np.sum(center == v)) * 8
        for v in (P.B, P.BK):
            b_score += int(np.sum(center == v)) * 8

        # حرية الحركة
        saved = game._turn_white
        game._turn_white = True
        wm_list, wc = game._get_moves_fallback()
        game._turn_white = False
        bm_list, bc = game._get_moves_fallback()
        game._turn_white = saved
        w_score += len(wm_list) * 5
        b_score += len(bm_list) * 5
        if wc:
            w_score += 20
        if bc:
            b_score += 20

        if for_white:
            return w_score - b_score
        return b_score - w_score

    def quiescence(self, game, alpha, beta, for_white, qdepth=0):
        self.nodes += 1
        self._check()
        if self.stopped:
            return 0

        stand = self.evaluate(game, for_white)
        if qdepth >= 6 or stand >= beta:
            return stand if stand < beta else beta
        if stand > alpha:
            alpha = stand

        moves, is_cap = game.get_moves()
        if not is_cap:
            return stand

        for mv in moves:
            game.do_move(mv)
            sc = -self.quiescence(
                game, -beta, -alpha, not for_white, qdepth + 1
            )
            game.undo_move()
            if self.stopped:
                return 0
            if sc >= beta:
                return beta
            if sc > alpha:
                alpha = sc
        return alpha

    def search(self, game, depth, alpha, beta, for_white, pv=True):
        self.nodes += 1
        self._check()
        if self.stopped:
            return 0, None

        w = game.winner()
        if w is not None:
            if w == 0:
                return 0, None
            is_my_win = (w == 1 and for_white) or (w == 2 and not for_white)
            return (99999 + depth if is_my_win else -99999 - depth), None

        if depth <= 0:
            return self.quiescence(game, alpha, beta, for_white), None

        moves, is_cap = game.get_moves()
        if not moves:
            return -99999, None

        # ترتيب: أكل أولاً، ثم مركز
        def sort_key(m):
            s = 0
            if is_cap:
                s += 10000 + len(m) * 100
            d = m[-1]
            if 2 <= d[0] <= 5 and 2 <= d[1] <= 5:
                s += 50
            if for_white and d[0] == 0:
                s += 5000
            if not for_white and d[0] == 7:
                s += 5000
            return s
        moves.sort(key=sort_key, reverse=True)

        best = moves[0]
        best_sc = float("-inf")
        searched = 0
        for mv in moves:
            game.do_move(mv)
            if searched == 0:
                sc = -self.search(
                    game, depth - 1, -beta, -alpha, not for_white, True
                )[0]
            else:
                sc = -self.search(
                    game, depth - 1, -alpha - 1, -alpha, not for_white, False
                )[0]
                if alpha < sc < beta and not self.stopped:
                    sc = -self.search(
                        game, depth - 1, -beta, -sc, not for_white, True
                    )[0]
            game.undo_move()
            if self.stopped:
                break
            searched += 1
            if sc > best_sc:
                best_sc = sc
                best = mv
            if sc > alpha:
                alpha = sc
            if alpha >= beta:
                break
        return best_sc, best

    def find_best(self, game, for_white):
        """Iterative Deepening — يبحث أعمق ما يمكن"""
        self.nodes = 0
        self.t0 = time.time()
        self.stopped = False
        self.best_depth = 0
        best_move = None
        best_score = 0
        log = []
        for d in range(1, 40):
            sc, mv = self.search(
                game, d, float("-inf"), float("inf"), for_white, True
            )
            if self.stopped:
                break
            if mv is not None:
                best_move = mv
                best_score = sc
                self.best_depth = d
            el = time.time() - self.t0
            log.append({
                "depth": d,
                "score": round(sc, 1),
                "nodes": self.nodes,
                "time": round(el, 2),
            })
            if abs(best_score) > 90000:
                break
            if time.time() - self.t0 >= self.max_time * 0.8:
                break
        el = time.time() - self.t0
        return {
            "move": best_move,
            "score": round(best_score, 1),
            "nodes": self.nodes,
            "time": round(el, 2),
            "depth": self.best_depth,
            "nps": int(self.nodes / max(el, 0.001)),
            "log": log,
        }

    def analyze_all(self, game, for_white):
        """تحليل كل الحركات"""
        moves, is_cap = game.get_moves()
        if not moves:
            return None
        each_t = max(0.3, self.max_time * 0.6 / len(moves))
        results = []
        for mv in moves:
            game.do_move(mv)
            sub = AI(max_time=each_t)
            res = sub.find_best(game, not for_white)
            sc = -res["score"]
            dep = res["depth"]
            game.undo_move()

            # تصنيف
            is_capt = (len(mv) > 2 or
                       (len(mv) == 2 and abs(mv[0][0] - mv[1][0]) == 2))
            cap_n = 0
            if is_capt:
                for i in range(len(mv) - 1):
                    if abs(mv[i][0] - mv[i + 1][0]) == 2:
                        cap_n += 1
            dest = mv[-1]
            piece = game.grid[mv[0][0]][mv[0][1]]
            promo = ((dest[0] == 0 and piece == P.W) or
                     (dest[0] == 7 and piece == P.B))

            if is_capt and cap_n >= 2:
                v = "🔥 أكل متعدد!"
            elif promo:
                v = "👑 ترقية!"
            elif sc > 200:
                v = "💪 ممتازة"
            elif sc > 50:
                v = "✅ جيدة"
            elif sc > -50:
                v = "⚖️ متكافئة"
            elif sc > -200:
                v = "⚠️ حذر"
            else:
                v = "❌ تجنبها"
            wp = max(1, min(99, int(50 + sc / 15)))

            results.append({
                "move": mv,
                "score": round(sc, 1),
                "depth": dep,
                "is_capture": is_capt,
                "captured": cap_n,
                "promotes": promo,
                "verdict": v,
                "win_pct": wp,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        pe = self.evaluate(game, for_white)
        return {
            "moves": results,
            "time": round(time.time() - self.t0, 2),
            "forced": is_cap,
            "eval": round(pe, 1),
        }

# ══════════════════════════════════════════
# 5. الرؤية الحاسوبية
# ══════════════════════════════════════════
class Vision:
    @staticmethod
    def fix_perspective(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, k, iterations=2)
        cnts, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            return cv2.resize(img, (400, 400)), False
        big = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(big) < img.shape[0] * img.shape[1] * 0.15:
            return cv2.resize(img, (400, 400)), False
        peri = cv2.arcLength(big, True)
        approx = cv2.approxPolyDP(big, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            s = pts.sum(axis=1)
            d = np.diff(pts, axis=1).ravel()
            o = np.zeros((4, 2), dtype=np.float32)
            o[0] = pts[np.argmin(s)]
            o[2] = pts[np.argmax(s)]
            o[1] = pts[np.argmin(d)]
            o[3] = pts[np.argmax(d)]
            dst = np.float32([[0, 0], [399, 0], [399, 399], [0, 399]])
            M = cv2.getPerspectiveTransform(o, dst)
            return cv2.warpPerspective(img, M, (400, 400)), True
        x, y, w, h = cv2.boundingRect(big)
        return cv2.resize(img[y:y+h, x:x+w], (400, 400)), False

    @staticmethod
    def detect(img, lt=160, dt=100):
        """كشف HSV + دوائر + دمج"""
        # HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = 50
        b1 = np.zeros((8, 8), dtype=np.int8)
        info = []
        for r in range(8):
            for col in range(8):
                if (r + col) % 2 == 0:
                    continue
                m = c // 4
                rg = gray[r*c+m:(r+1)*c-m, col*c+m:(col+1)*c-m]
                rh = hsv[r*c+m:(r+1)*c-m, col*c+m:(col+1)*c-m]
                br = float(np.mean(rg))
                sa = float(np.mean(rh[:, :, 1]))
                va = float(np.var(rg))
                det = int(P.E)
                if va > 120 or sa > 30:
                    if br > lt:
                        det = int(P.W)
                    elif br < dt:
                        det = int(P.B)
                b1[r][col] = det
                info.append({
                    "r": r,
                    "c": col,
                    "br": round(br),
                    "va": round(va),
                    "d": det,
                })

        # دوائر
        gm = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gm, cv2.HOUGH_GRADIENT, 1.2, 35,
            param1=60, param2=35, minRadius=12, maxRadius=24
        )
        b2 = np.zeros((8, 8), dtype=np.int8)
        vis = img.copy()
        if circles is not None:
            for cx, cy, rad in np.uint16(np.around(circles))[0]:
                co, ro = int(cx/c), int(cy/c)
                if not (0 <= ro < 8 and 0 <= co < 8):
                    continue
                if (ro + co) % 2 == 0:
                    continue
                s = gray[
                    max(0, int(cy)-5):min(400, int(cy)+5),
                    max(0, int(cx)-5):min(400, int(cx)+5)
                ]
                if s.size == 0:
                    continue
                a = float(np.mean(s))
                if a > lt:
                    b2[ro][co] = int(P.W)
                    clr = (0, 255, 0)
                elif a < dt:
                    b2[ro][co] = int(P.B)
                    clr = (0, 0, 255)
                else:
                    continue
                cv2.circle(vis, (int(cx), int(cy)), int(rad), clr, 2)
        vis_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        # دمج
        merged = np.zeros((8, 8), dtype=np.int8)
        for r in range(8):
            for c_ in range(8):
                h, ci = b1[r][c_], b2[r][c_]
                if h != P.E and ci != P.E:
                    merged[r][c_] = h
                elif ci != P.E:
                    merged[r][c_] = ci
                elif h != P.E:
                    merged[r][c_] = h
        return merged, b1, b2, vis_pil, info

# ══════════════════════════════════════════
# 6. رسم الرقعة
# ══════════════════════════════════════════
class R:
    @staticmethod
    def draw(board, arrows=None, hl=None):
        img = Image.new("RGB", (BOARD_PX, BOARD_PX))
        dr = ImageDraw.Draw(img)
        for r in range(8):
            for c in range(8):
                x1, y1 = c * CELL, r * CELL
                x2, y2 = x1 + CELL, y1 + CELL
                sq = ((235, 215, 180) if (r + c) % 2 == 0 else (175, 130, 95))
                if hl and (r, c) in hl:
                    sq = (100, 200, 100)
                dr.rectangle([x1, y1, x2, y2], fill=sq)

                p = board[r][c]
                if p == P.E:
                    continue
                cx, cy = x1 + CELL // 2, y1 + CELL // 2
                pr = CELL // 2 - 10
                dr.ellipse(
                    [cx - pr + 3, cy - pr + 3, cx + pr + 3, cy + pr + 3],
                    fill=(70, 50, 30)
                )
                fl = ((250, 248, 240) if p in (P.W, P.WK) else (45, 45, 45))
                ed = ((195, 185, 170) if p in (P.W, P.WK) else (25, 25, 25))
                dr.ellipse(
                    [cx - pr, cy - pr, cx + pr, cy + pr],
                    fill=fl, outline=ed, width=2
                )
                dr.ellipse(
                    [cx - pr + 5, cy - pr + 5, cx + pr - 5, cy + pr - 5],
                    outline=ed, width=1
                )
                if p in (P.WK, P.BK):
                    dr.ellipse(
                        [cx - 12, cy - 12, cx + 12, cy + 12],
                        fill=(255, 215, 0), outline=(200, 170, 0), width=2
                    )
        for i in range(8):
            try:
                dr.text((3, i * CELL + 3), str(i), fill=(130, 110, 90))
                dr.text(
                    (i * CELL + CELL // 2 - 4, BOARD_PX - 14),
                    chr(65 + i), fill=(130, 110, 90)
                )
            except Exception:
                pass
        if arrows:
            for a in arrows:
                R._arrow(dr, a["m"], a.get("c", (255, 50, 50)), a.get("w", 5))
        return img

    @staticmethod
    def _arrow(d, mv, color, w):
        if not mv or len(mv) < 2:
            return
        for i in range(len(mv) - 1):
            s = mv[i]
            e = mv[i + 1]
            sx, sy = s[1] * CELL + CELL // 2, s[0] * CELL + CELL // 2
            ex, ey = e[1] * CELL + CELL // 2, e[0] * CELL + CELL // 2
            d.line([(sx, sy), (ex, ey)], fill=color, width=w)
            d.ellipse([ex - 8, ey - 8, ex + 8, ey + 8], fill=color)
        s = mv[0]
        sx, sy = s[1] * CELL + CELL // 2, s[0] * CELL + CELL // 2
        d.ellipse([sx - 12, sy - 12, sx + 12, sy + 12], outline=(0, 220, 0), width=4)

# ══════════════════════════════════════════
# 7. واجهة Streamlit
# ══════════════════════════════════════════
def app():
    st.set_page_config("♟️ مساعد الداما", "♟️", layout="wide")
    st.markdown("""
<style>
.block-container{max-width:1100px}
.best-box{background:linear-gradient(135deg,#28a745,#20c997); color:#fff;padding:18px;border-radius:12px; text-align:center;margin:10px 0}
.best-box h2{margin:0;font-size:1.6em}
.best-box p{margin:5px 0}
.warn{background:#f8d7da;border:2px solid #dc3545; color:#721c24;padding:12px;border-radius:10px; text-align:center;margin:8px 0;font-weight:bold}
.ebar{background:#e9ecef;border-radius:8px; overflow:hidden;height:30px;margin:8px 0}
.efill{height:100%;text-align:center;color:#fff; font-weight:bold;line-height:30px;border-radius:8px}
.badge{display:inline-block;padding:4px 12px; border-radius:20px;font-size:0.85em;font-weight:bold}
.g{background:#28a745;color:#fff}
.y{background:#ffc107;color:#333}
</style>
""", unsafe_allow_html=True)

    if "board" not in st.session_state:
        st.session_state.board = GameInterface._initial_grid().tolist()

    st.title("♟️ مساعد الداما الذكي")
    # شارة المحرك
    if HAS_DRAUGHTS:
        st.markdown(
            '<span class="badge g">✅ pydraughts — قواعد مثالية</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="badge y">🔧 محرك احتياطي — ثبّت pydraughts لقواعد مثالية</span>',
            unsafe_allow_html=True
        )
    st.caption("حمّل صورة أو أدخل يدوياً → AI يريك كيف تفوز")

    # ═══ الشريط الجانبي ═══
    with st.sidebar:
        st.header("⚙️")
        my_c = st.radio("♟️ لونك:", ["⚪ أبيض", "⚫ أسود"])
        for_white = "أبيض" in my_c
        tt = st.select_slider("⏱ وقت:", [1, 2, 3, 5, 8, 10], value=3)

        st.divider()
        if st.button("🔄 جديدة", use_container_width=True):
            st.session_state.board = GameInterface._initial_grid().tolist()
            st.rerun()
        if st.button("🗑️ مسح", use_container_width=True):
            st.session_state.board = np.zeros((8, 8), dtype=int).tolist()
            st.rerun()

        st.divider()
        g_ = np.array(st.session_state.board)
        wn = int(np.sum((g_ == P.W) | (g_ == P.WK)))
        bn = int(np.sum((g_ == P.B) | (g_ == P.BK)))
        c1, c2 = st.columns(2)
        with c1:
            st.metric("⚪", wn)
        with c2:
            st.metric("⚫", bn)

    # ═══ التبويبات ═══
    t_names = ["✏️ يدوي"]
    if HAS_CV2:
        t_names.append("📷 صورة")
    t_names.append("🧠 تحليل")
    tabs = st.tabs(t_names)

    # ─── يدوي ───
    with tabs[0]:
        opts = {
            "⬜ فارغ": int(P.E),
            "⚪ أبيض": int(P.W),
            "⚫ أسود": int(P.B),
            "👑W": int(P.WK),
            "♛B": int(P.BK),
        }
        sel = st.radio(
            "_", list(opts.keys()), horizontal=True, label_visibility="collapsed"
        )
        sv = opts[sel]
        syms = {
            int(P.E): "·",
            int(P.W): "⚪",
            int(P.B): "⚫",
            int(P.WK): "👑",
            int(P.BK): "♛",
        }
        ba = np.array(st.session_state.board)
        for r in range(8):
            cols = st.columns(8)
            for c in range(8):
                with cols[c]:
                    ok = (r + c) % 2 != 0
                    s = (syms.get(int(ba[r][c]), "·") if ok else "")
                    if st.button(
                        s, key=f"m{r}{c}", use_container_width=True, disabled=not ok
                    ):
                        st.session_state.board[r][c] = sv
                        st.rerun()
        st.image(R.draw(ba), caption="الرقعة", use_container_width=True)

    # ─── صورة ───
    if HAS_CV2:
        with tabs[1]:
            st.subheader("📷 تحليل صورة")
            up = st.file_uploader("📸", type=["jpg", "png", "jpeg"])
            if up:
                pil = Image.open(up).convert("RGB")
                icv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                c1, c2 = st.columns(2)
                with c1:
                    st.image(pil, caption="الأصلية", use_container_width=True)
                with st.spinner("🔲 ..."):
                    fixed, was = Vision.fix_perspective(icv)
                    fp = Image.fromarray(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))
                with c2:
                    st.image(
                        fp, caption=("✅ مصحح" if was else "📐 مقتصة"),
                        use_container_width=True
                    )
                tc1, tc2 = st.columns(2)
                with tc1:
                    lt = st.slider("فاتح", 100, 230, 160)
                with tc2:
                    dt = st.slider("داكن", 30, 150, 100)

                if st.button("🔍 تحليل", type="primary"):
                    with st.spinner("🧠 ..."):
                        merged, hb, cb, cvis, inf = Vision.detect(fixed, lt, dt)
                    st.success("✅")
                    t1, t2, t3 = st.tabs(["HSV", "دوائر", "مدمج"])
                    with t1:
                        st.image(R.draw(hb), use_container_width=True)
                    with t2:
                        st.image(cvis, use_container_width=True)
                    with t3:
                        st.image(R.draw(merged), use_container_width=True)
                    if st.button("📥 استخدم", type="primary"):
                        st.session_state.board = merged.tolist()
                        st.rerun()

    # ─── تحليل ───
    ai_idx = 2 if HAS_CV2 else 1
    with tabs[ai_idx]:
        ba2 = np.array(st.session_state.board, dtype=np.int8)
        game = GameInterface(ba2, turn_white=for_white)
        st.image(R.draw(ba2), caption="الرقعة", use_container_width=True)

        wn = int(np.sum((ba2 == P.W) | (ba2 == P.WK)))
        bn = int(np.sum((ba2 == P.B) | (ba2 == P.BK)))
        if wn == 0 and bn == 0:
            st.warning("⚠️ الرقعة فارغة!")
            return
        w = game.winner()
        if w is not None:
            if w == 0:
                st.info("🤝 تعادل")
            elif w == 1:
                st.success("🏆 الأبيض فاز!")
            else:
                st.success("🏆 الأسود فاز!")
            return

        emoji = "⚪" if for_white else "⚫"
        st.markdown(
            f"### {emoji} تحليل **{'الأبيض' if for_white else 'الأسود'}**"
        )

        if st.button("🧠 حلّل!", type="primary", use_container_width=True):
            prg = st.empty()
            prg.info(f"🧠 تحليل ({tt}s)...")
            ai = AI(max_time=tt)
            analysis = ai.analyze_all(game, for_white)
            prg.empty()

            if not analysis or not analysis["moves"]:
                st.error("❌ لا حركات!")
                return

            mvs = analysis["moves"]
            best = mvs[0]

            # تقييم
            pe = analysis["eval"]
            if pe > 200:
                em, ec = "🟢 متفوق!", "#28a745"
            elif pe > 50:
                em, ec = "🟢 أفضل", "#20c997"
            elif pe > -50:
                em, ec = "🟡 متكافئة", "#ffc107"
            elif pe > -200:
                em, ec = "🟠 الخصم أفضل", "#fd7e14"
            else:
                em, ec = "🔴 خطر!", "#dc3545"
            pct = max(5, min(95, int(50 + pe / 15)))
            st.markdown(
                f'<div class="ebar">'
                f'<div class="efill" style="width:{pct}%;background:{ec}">'
                f'{em} ({pe})</div></div>',
                unsafe_allow_html=True
            )

            if analysis["forced"]:
                st.markdown(
                    '<div class="warn">⚡ أكل إجباري!</div>',
                    unsafe_allow_html=True
                )

            # أفضل حركة
            path = " → ".join(f"({p[0]},{p[1]})" for p in best["move"])
            ex = []
            if best["is_capture"]:
                ex.append(f"💥×{best['captured']}")
            if best["promotes"]:
                ex.append("👑")
            ex.append(best["verdict"])
            ext = " • ".join(ex)
            st.markdown(f"""
<div class="best-box">
<h2>🏆 أفضل حركة</h2>
<p style="font-size:1.4em">{path}</p>
<p>{ext}</p>
<p>تقييم: {best['score']} • فوز: {best['win_pct']}% • عمق: {best['depth']}</p>
</div>
""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                bi = R.draw(
                    ba2,
                    arrows=[{"m": best["move"], "c": (50, 205, 50), "w": 6}],
                    hl=set(best["move"])
                )
                st.image(bi, caption="🏆 الحركة", use_container_width=True)
            with c2:
                af = game.copy()
                af.do_move(best["move"])
                st.image(R.draw(af.grid), caption="📋 بعد التنفيذ", use_container_width=True)

            if st.button("✅ طبّق", use_container_width=True):
                st.session_state.board = af.grid.tolist()
                st.rerun()

            # كل الحركات
            st.markdown(f"### 📊 كل الحركات ({len(mvs)})")
            rc = {
                1: ("🥇", (50, 205, 50)),
                2: ("🥈", (65, 105, 225)),
                3: ("🥉", (255, 165, 0)),
            }
            top = []
            for i, mv in enumerate(mvs[:5]):
                _, cl = rc.get(
                    mv["rank"], (f"#{mv['rank']}", (180, 180, 180))
                )
                top.append({"m": mv["move"], "c": cl, "w": 6 if i == 0 else 3})
            st.image(
                R.draw(ba2, arrows=top),
                caption="🥇🥈🥉",
                use_container_width=True
            )
            for mv in mvs:
                icon = rc.get(mv["rank"], (f"#{mv['rank']}", None))[0]
                p = " → ".join(f"({x[0]},{x[1]})" for x in mv["move"])
                bar = "█" * max(1, int(mv["win_pct"] / 5))
                with st.expander(
                    f"{icon} {p} • {mv['score']} • {mv['win_pct']}% • {mv['verdict']}"
                ):
                    st.markdown(f"`{p}`")
                    st.markdown(f"فوز: `{bar}` {mv['win_pct']}%")
                    mi = R.draw(
                        ba2,
                        arrows=[{"m": mv["move"], "c": (255, 100, 50), "w": 5}],
                        hl=set(mv["move"])
                    )
                    st.image(mi, use_container_width=True)
            st.divider()

    eng_name = "pydraughts" if HAS_DRAUGHTS else "fallback"
    st.markdown(
        f'<p style="text-align:center;color:#999;font-size:0.8em">'
        f'♟️ مساعد الداما — {eng_name} + Minimax+αβ+PVS + OpenCV — بدون إنترنت</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    app()
