#!/usr/bin/env python3
"""
╔═════════════════════════════════════════════════════════╗
║ ♟️ مساعد الداما الذكي — النسخة النهائية                ║
║ XGBoost (تقييم فوري) + Minimax (بحث عميق) + OpenCV    ║
║ ليس لعبة — مساعد يريك كيف تفوز كل مرة                 ║
╚═════════════════════════════════════════════════════════╝
"""
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time
import random
import os
from enum import IntEnum

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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
MODEL_FILE = "brutal_ai_model.json"

# جداول التقييم الاحتياطية (تُستخدم فقط إذا لم يوجد نموذج)
LIGHT_POS = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0],
    [0, 3, 0, 3, 0, 3, 0, 0],
    [0, 0, 5, 0, 5, 0, 4, 0],
    [0, 5, 0, 7, 0, 5, 0, 0],
    [0, 0, 7, 0, 7, 0, 6, 0],
    [0, 8, 0, 9, 0, 9, 0, 0],
    [0, 0, 10, 0, 10, 0, 10, 0],
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
            np.array(board, dtype=np.int8) if board is not None else self._init()
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
    def is_light(p):
        return p in (P.L, P.LK)

    @staticmethod
    def is_dark(p):
        return p in (P.D, P.DK)

    @staticmethod
    def is_king(p):
        return p in (P.LK, P.DK)

    @staticmethod
    def owns(piece, player):
        if player in (P.L, P.LK):
            return piece in (P.L, P.LK)
        return piece in (P.D, P.DK)

    @staticmethod
    def enemy(piece, player):
        if piece == P.E:
            return False
        if player in (P.L, P.LK):
            return piece in (P.D, P.DK)
        return piece in (P.L, P.LK)

    @staticmethod
    def opp(player):
        return P.D if player in (P.L, P.LK) else P.L

    @staticmethod
    def dirs(piece):
        if piece == P.L:
            return ((-1, -1), (-1, 1))
        if piece == P.D:
            return ((1, -1), (1, 1))
        if piece in (P.LK, P.DK):
            return ((-1, -1), (-1, 1), (1, -1), (1, 1))
        return ()

    def _simple(self, r, c):
        p = self.board[r][c]
        out = []
        for dr, dc in self.dirs(p):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if self.board[nr][nc] == P.E:
                    out.append(((r, c), (nr, nc)))
        return out

    def _jumps(self, r, c, bd=None, eaten=None):
        if bd is None:
            bd = self.board
        if eaten is None:
            eaten = frozenset()
        p = bd[r][c]
        chains = []
        for dr, dc in self.dirs(p):
            mr, mc = r + dr, c + dc
            nr, nc = r + 2 * dr, c + 2 * dc
            if not (0 <= nr < 8 and 0 <= nc < 8):
                continue
            if bd[nr][nc] != P.E:
                continue
            if not self.enemy(bd[mr][mc], p):
                continue
            if (mr, mc) in eaten:
                continue
            nb = bd.copy()
            nb[nr][nc] = p
            nb[r][c] = P.E
            nb[mr][mc] = P.E
            promo = False
            if nr == 0 and p == P.L:
                nb[nr][nc] = P.LK
                promo = True
            elif nr == 7 and p == P.D:
                nb[nr][nc] = P.DK
                promo = True
            ne = eaten | {(mr, mc)}
            fur = [] if promo else self._jumps(nr, nc, nb, ne)
            if fur:
                for ch in fur:
                    chains.append(((r, c),) + ch)
            else:
                chains.append(((r, c), (nr, nc)))
        return chains

    def get_moves(self, player):
        jumps, simple = [], []
        for r in range(8):
            for c in range(8):
                if self.owns(self.board[r][c], player):
                    jumps.extend(self._jumps(r, c))
                    simple.extend(self._simple(r, c))
        if jumps:
            mx = max(len(j) for j in jumps)
            return [j for j in jumps if len(j) == mx], True
        return simple, False

    def do_move(self, move):
        piece = self.board[move[0][0]][move[0][1]]
        self.board[move[0][0]][move[0][1]] = P.E
        for i in range(len(move) - 1):
            sr, sc = move[i]
            er, ec = move[i + 1]
            dr, dc = er - sr, ec - sc
            if abs(dr) == 2 and abs(dc) == 2:
                self.board[sr + dr // 2][sc + dc // 2] = P.E
        fr, fc = move[-1]
        self.board[fr][fc] = piece
        if fr == 0 and piece == P.L:
            self.board[fr][fc] = P.LK
        if fr == 7 and piece == P.D:
            self.board[fr][fc] = P.DK

    def count(self, player):
        n = k = 0
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if self.owns(p, player):
                    if self.is_king(p):
                        k += 1
                    else:
                        n += 1
        return n, k

    def game_over(self):
        ln, lk = self.count(P.L)
        dn, dk = self.count(P.D)
        if ln + lk == 0:
            return P.D
        if dn + dk == 0:
            return P.L
        lm, _ = self.get_moves(P.L)
        dm, _ = self.get_moves(P.D)
        if not lm and not dm:
            return -1
        if not lm:
            return P.D
        if not dm:
            return P.L
        return None

# ══════════════════════════════════════════
# 🧠 المقيّم الهجين: XGBoost + Fallback
# ══════════════════════════════════════════
class HybridEvaluator:
    """يحاول استخدام XGBoost أولاً (سريع جداً) إذا لم يوجد النموذج → يستخدم تقييم رياضي احتياطي"""
    def __init__(self):
        self.model = None
        self.mode = "fallback"
        self._load_model()

    def _load_model(self):
        if not HAS_XGB:
            return
        if os.path.exists(MODEL_FILE):
            try:
                self.model = xgb.XGBRegressor()
                self.model.load_model(MODEL_FILE)
                self.mode = "xgboost"
            except Exception as e:
                self.model = None
                self.mode = "fallback"

    def predict(self, eng, player):
        """تقييم الرقعة بأسرع طريقة متاحة"""
        if self.mode == "xgboost" and self.model is not None:
            return self._xgb_predict(eng, player)
        return self._fallback_predict(eng, player)

    def _xgb_predict(self, eng, player):
        """تقييم فوري بـ XGBoost (< 0.001 ثانية)"""
        flat = eng.board.flatten().astype(np.float32)
        features = np.append(flat, int(player)).reshape(1, -1)
        score = float(self.model.predict(features)[0])
        return score

    def _fallback_predict(self, eng, player):
        """تقييم رياضي احتياطي (سريع أيضاً)"""
        b = eng.board
        opp = eng.opp(player)
        if player in (P.L, P.LK):
            my_men = (b == P.L)
            my_king = (b == P.LK)
            op_men = (b == P.D)
            op_king = (b == P.DK)
        else:
            my_men = (b == P.D)
            my_king = (b == P.DK)
            op_men = (b == P.L)
            op_king = (b == P.LK)

        mn = int(np.sum(my_men))
        mk = int(np.sum(my_king))
        on = int(np.sum(op_men))
        ok_ = int(np.sum(op_king))

        if mn + mk == 0:
            return -99999
        if on + ok_ == 0:
            return 99999

        score = (mn * 100 + mk * 330) - (on * 100 + ok_ * 330)

        # موقع
        if player in (P.L, P.LK):
            score += float(np.sum(my_men * LIGHT_POS)) * 4
            score -= float(np.sum(op_men * DARK_POS)) * 4
        else:
            score += float(np.sum(my_men * DARK_POS)) * 4
            score -= float(np.sum(op_men * LIGHT_POS)) * 4

        score += float(np.sum(my_king * KING_POS)) * 3
        score -= float(np.sum(op_king * KING_POS)) * 3

        # حرية الحركة
        my_m, my_cap = eng.get_moves(player)
        op_m, op_cap = eng.get_moves(opp)
        score += len(my_m) * 5 - len(op_m) * 5
        if my_cap:
            score += 25
        if op_cap:
            score -= 25

        # ترابط
        for r in range(8):
            for c in range(8):
                p = b[r][c]
                if p == P.E:
                    continue
                is_mine = eng.owns(p, player)
                allies = 0
                for dr2, dc2 in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    ar, ac = r + dr2, c + dc2
                    if 0 <= ar < 8 and 0 <= ac < 8:
                        if eng.owns(b[ar][ac], p):
                            allies += 1
                if is_mine:
                    score += allies * 3
                else:
                    score -= allies * 3
        return score

# ══════════════════════════════════════════
# ⚡ المحرك السريع
# ══════════════════════════════════════════
class FastAI:
    """Minimax + Alpha-Beta + XGBoost تقييم فوري = بحث عميق في وقت قصير جداً"""
    def __init__(self, evaluator, max_time=3.0):
        self.evaluator = evaluator
        self.max_time = max_time
        self.nodes = 0
        self.start_time = 0
        self.time_up = False
        self.best_depth = 0
        self.killers = [[None, None] for _ in range(50)]
        self.history = {}

    def check_time(self):
        if self.nodes % 500 == 0:
            if time.time() - self.start_time >= self.max_time:
                self.time_up = True

    def order_moves(self, moves, is_cap, eng, player, depth):
        """ترتيب ذكي: أكل > killer > history > موقع"""
        scored = []
        for m in moves:
            s = 0
            if is_cap:
                s += 50000 + len(m) * 1000
            elif (depth < len(self.killers) and m in self.killers[depth]):
                s += 40000
            else:
                key = (m[0], m[-1])
                s += self.history.get(key, 0)

            dest = m[-1]
            piece = eng.board[m[0][0]][m[0][1]]
            if dest[0] == 0 and piece == P.L:
                s += 30000
            if dest[0] == 7 and piece == P.D:
                s += 30000
            if 2 <= dest[0] <= 5 and 2 <= dest[1] <= 5:
                s += 100
            scored.append((s, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def quiescence(self, eng, alpha, beta, player, original, qdepth=0):
        """بحث الهدوء: لا يتوقف أثناء الأكل"""
        self.nodes += 1
        self.check_time()
        if self.time_up:
            return 0
        stand = self.evaluator.predict(eng, original)
        if qdepth >= 6:
            return stand
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand
        moves, is_cap = eng.get_moves(player)
        if not is_cap:
            return stand
        for move in moves:
            child = eng.copy()
            child.do_move(move)
            score = -self.quiescence(
                child, -beta, -alpha, eng.opp(player), original, qdepth + 1
            )
            if self.time_up:
                return 0
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def alpha_beta(self, eng, depth, alpha, beta, player, original, is_pv=True):
        self.nodes += 1
        self.check_time()
        if self.time_up:
            return 0, None

        result = eng.game_over()
        if result is not None:
            if result == original:
                return 99999 + depth, None
            elif result == -1:
                return 0, None
            else:
                return -99999 - depth, None

        if depth <= 0:
            return self.quiescence(eng, alpha, beta, player, original), None

        moves, is_cap = eng.get_moves(player)
        if not moves:
            return self.evaluator.predict(eng, original), None

        moves = self.order_moves(moves, is_cap, eng, player, depth)
        best_move = moves[0]
        best_score = float("-inf")
        searched = 0

        for move in moves:
            child = eng.copy()
            child.do_move(move)
            nxt = eng.opp(player)

            # PVS
            if searched == 0:
                score = -self.alpha_beta(
                    child, depth - 1, -beta, -alpha, nxt, original, True
                )[0]
            else:
                score = -self.alpha_beta(
                    child, depth - 1, -alpha - 1, -alpha, nxt, original, False
                )[0]
                if alpha < score < beta and not self.time_up:
                    score = -self.alpha_beta(
                        child, depth - 1, -beta, -score, nxt, original, True
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
                if not is_cap and depth < len(self.killers):
                    k = self.killers[depth]
                    if move != k[0]:
                        k[1] = k[0]
                        k[0] = move
                key = (move[0], move[-1])
                self.history[key] = self.history.get(key, 0) + depth * depth
                break

        return best_score, best_move

    def find_best(self, eng, player):
        """Iterative Deepening مع تحكم بالوقت"""
        self.nodes = 0
        self.start_time = time.time()
        self.time_up = False
        self.best_depth = 0
        self.killers = [[None, None] for _ in range(50)]
        self.history = {}
        best_move = None
        best_score = 0
        depth_log = []

        for depth in range(1, 40):
            score, move = self.alpha_beta(
                eng, depth, float("-inf"), float("inf"), player, player, True
            )
            if self.time_up:
                break
            if move is not None:
                best_move = move
                best_score = score
                self.best_depth = depth

            elapsed = time.time() - self.start_time
            depth_log.append({
                "depth": depth,
                "score": round(score, 1),
                "nodes": self.nodes,
                "time": round(elapsed, 2),
            })
            if (abs(best_score) > 90000 or
                time.time() - self.start_time >= self.max_time * 0.8):
                break

        elapsed = time.time() - self.start_time
        return {
            "move": best_move,
            "score": round(best_score, 1),
            "nodes": self.nodes,
            "time": round(elapsed, 2),
            "depth": self.best_depth,
            "nps": int(self.nodes / max(elapsed, 0.001)),
            "depth_log": depth_log,
        }

    def analyze_all(self, eng, player):
        """تحليل وتصنيف كل الحركات"""
        self.nodes = 0
        self.start_time = time.time()
        self.time_up = False

        moves, is_cap = eng.get_moves(player)
        if not moves:
            return None

        time_each = max(0.5, self.max_time * 0.7 / len(moves))
        results = []
        for move in moves:
            child = eng.copy()
            child.do_move(move)
            sub = FastAI(self.evaluator, max_time=time_each)
            opp_res = sub.find_best(child, eng.opp(player))
            score = -opp_res["score"]
            depth_r = opp_res["depth"]

            is_capture = (len(move) > 2 or
                          (len(move) == 2 and abs(move[0][0] - move[1][0]) == 2))
            cap_count = 0
            if is_capture:
                for i in range(len(move) - 1):
                    if abs(move[i][0] - move[i + 1][0]) == 2:
                        cap_count += 1

            dest = move[-1]
            piece = eng.board[move[0][0]][move[0][1]]
            promotes = ((dest[0] == 0 and piece == P.L) or
                        (dest[0] == 7 and piece == P.D))

            if is_capture and cap_count >= 2:
                verdict = "🔥 أكل متعدد ممتاز!"
            elif promotes:
                verdict = "👑 ترقية للملك!"
            elif score > 200:
                verdict = "💪 تفوق واضح"
            elif score > 50:
                verdict = "✅ جيدة"
            elif score > -50:
                verdict = "⚖️ متكافئة"
            elif score > -200:
                verdict = "⚠️ محفوفة بالمخاطر"
            else:
                verdict = "❌ خطيرة — تجنبها"

            results.append({
                "move": move,
                "score": round(score, 1),
                "depth": depth_r,
                "is_capture": is_capture,
                "captured": cap_count,
                "promotes": promotes,
                "verdict": verdict,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
            s = r["score"]
            if s > 5000:
                r["win_pct"] = 99
            elif s < -5000:
                r["win_pct"] = 1
            else:
                r["win_pct"] = max(1, min(99, int(50 + s / 15)))

        elapsed = time.time() - self.start_time
        pos_eval = self.evaluator.predict(eng, player)
        return {
            "moves": results,
            "time": round(elapsed, 2),
            "is_forced": is_cap,
            "pos_eval": round(pos_eval, 1),
        }

# ══════════════════════════════════════════
# تحليل الصور (OpenCV)
# ══════════════════════════════════════════
class Vision:
    @staticmethod
    def fix_perspective(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, k, iterations=2)
        cnts, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            return cv2.resize(img_bgr, (400, 400)), False
        largest = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < img_bgr.shape[0] * img_bgr.shape[1] * 0.15:
            return cv2.resize(img_bgr, (400, 400)), False
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
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
            return cv2.warpPerspective(img_bgr, M, (400, 400)), True
        x, y, w, h = cv2.boundingRect(largest)
        return cv2.resize(img_bgr[y:y + h, x:x + w], (400, 400)), False

    @staticmethod
    def detect_hsv(img, lt=160, dt=100):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = 50
        board = np.zeros((8, 8), dtype=np.int8)
        info = []
        for r in range(8):
            for col in range(8):
                if (r + col) % 2 == 0:
                    continue
                m = c // 4
                roi_g = gray[
                    r * c + m:(r + 1) * c - m,
                    col * c + m:(col + 1) * c - m
                ]
                roi_h = hsv[
                    r * c + m:(r + 1) * c - m,
                    col * c + m:(col + 1) * c - m
                ]
                br = float(np.mean(roi_g))
                sa = float(np.mean(roi_h[:, :, 1]))
                va = float(np.var(roi_g))
                det = int(P.E)
                if va > 120 or sa > 30:
                    if br > lt:
                        det = int(P.L)
                    elif br < dt:
                        det = int(P.D)
                board[r][col] = det
                info.append({
                    "r": r,
                    "c": col,
                    "br": round(br),
                    "sa": round(sa),
                    "va": round(va),
                    "d": det,
                })
        return board, info

    @staticmethod
    def detect_circles(img, lt=160, dt=100):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        c = 50
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1.2, 35,
            param1=60, param2=35, minRadius=12, maxRadius=24
        )
        board = np.zeros((8, 8), dtype=np.int8)
        vis = img.copy()
        if circles is not None:
            for cx, cy, rad in np.uint16(np.around(circles))[0]:
                co = int(cx / c)
                ro = int(cy / c)
                if not (0 <= ro < 8 and 0 <= co < 8):
                    continue
                if (ro + co) % 2 == 0:
                    continue
                s = gray[
                    max(0, int(cy) - 5):min(400, int(cy) + 5),
                    max(0, int(cx) - 5):min(400, int(cx) + 5)
                ]
                if s.size == 0:
                    continue
                a = float(np.mean(s))
                if a > lt:
                    board[ro][co] = int(P.L)
                    clr = (0, 255, 0)
                elif a < dt:
                    board[ro][co] = int(P.D)
                    clr = (0, 0, 255)
                else:
                    continue
                cv2.circle(vis, (int(cx), int(cy)), int(rad), clr, 2)
        return board, Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    @staticmethod
    def merge(a, b):
        m = np.zeros((8, 8), dtype=np.int8)
        for r in range(8):
            for c in range(8):
                if a[r][c] != P.E and b[r][c] != P.E:
                    m[r][c] = a[r][c]
                elif b[r][c] != P.E:
                    m[r][c] = b[r][c]
                elif a[r][c] != P.E:
                    m[r][c] = a[r][c]
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
                x1, y1 = c * CELL, r * CELL
                x2, y2 = x1 + CELL, y1 + CELL
                sq = ((235, 215, 180) if (r + c) % 2 == 0 else (175, 130, 95))
                if highlight and (r, c) in highlight:
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
                fl = ((250, 248, 240) if Engine.is_light(p) else (45, 45, 45))
                ed = ((195, 185, 170) if Engine.is_light(p) else (25, 25, 25))
                dr.ellipse(
                    [cx - pr, cy - pr, cx + pr, cy + pr],
                    fill=fl, outline=ed, width=2
                )
                dr.ellipse(
                    [cx - pr + 5, cy - pr + 5, cx + pr - 5, cy + pr - 5],
                    outline=ed, width=1
                )
                if Engine.is_king(p):
                    kr = 12
                    dr.ellipse(
                        [cx - kr, cy - kr, cx + kr, cy + kr],
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
                Render._arrow(
                    dr, a["move"], a.get("color", (255, 50, 50)), a.get("width", 5)
                )
        if label:
            try:
                dr.rectangle([0, 0, BOARD_PX, 22], fill=(0, 0, 0))
                dr.text((5, 3), label, fill=(255, 255, 255))
            except Exception:
                pass
        return img

    @staticmethod
    def _arrow(draw, move, color, width):
        if not move or len(move) < 2:
            return
        for i in range(len(move) - 1):
            sr, sc = move[i]
            er, ec = move[i + 1]
            sx = sc * CELL + CELL // 2
            sy = sr * CELL + CELL // 2
            ex = ec * CELL + CELL // 2
            ey = er * CELL + CELL // 2
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
            draw.ellipse([ex - 8, ey - 8, ex + 8, ey + 8], fill=color)
        sr, sc = move[0]
        sx = sc * CELL + CELL // 2
        sy = sr * CELL + CELL // 2
        draw.ellipse([sx - 12, sy - 12, sx + 12, sy + 12], outline=(0, 220, 0), width=4)

# ══════════════════════════════════════════
# واجهة Streamlit
# ══════════════════════════════════════════
def main():
    st.set_page_config(
        "♟️ مساعد الداما الذكي", "♟️", layout="wide"
    )
    st.markdown("""
<style>
.block-container{max-width:1100px}
.best-box{background:linear-gradient(135deg,#28a745,#20c997); color:#fff;padding:18px;border-radius:12px; text-align:center;margin:10px 0}
.best-box h2{margin:0;font-size:1.6em}
.best-box p{margin:5px 0;font-size:1.1em}
.warn-box{background:#f8d7da;border:2px solid #dc3545; color:#721c24;padding:12px;border-radius:10px; text-align:center;margin:8px 0;font-weight:bold}
.eval-bar{background:#e9ecef;border-radius:8px; overflow:hidden;height:30px;margin:8px 0}
.eval-fill{height:100%;text-align:center;color:#fff; font-weight:bold;line-height:30px;border-radius:8px}
.perf-box{background:#f0f2f6;padding:12px; border-radius:10px;border-left:4px solid #667eea; margin:8px 0}
.mode-badge{display:inline-block;padding:4px 12px; border-radius:20px;font-size:0.85em;font-weight:bold}
.xgb-badge{background:#28a745;color:#fff}
.fb-badge{background:#ffc107;color:#333}
</style>
""", unsafe_allow_html=True)

    # ── تحميل المقيّم ──
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = HybridEvaluator()
    if "board" not in st.session_state:
        st.session_state.board = Engine._init().tolist()
    evaluator = st.session_state.evaluator

    st.title("♟️ مساعد الداما الذكي")
    # شارة وضع التشغيل
    if evaluator.mode == "xgboost":
        st.markdown(
            '<span class="mode-badge xgb-badge">'
            '⚡ XGBoost — تقييم فوري</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="mode-badge fb-badge">'
            '🔧 وضع احتياطي — ارفع brutal_ai_model.json لتفعيل XGBoost</span>',
            unsafe_allow_html=True
        )
    st.caption("حمّل صورة أو أدخل يدوياً → AI يحلل ويريك كيف تفوز كل مرة")

    # ═══ الشريط الجانبي ═══
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        my_color = st.radio(
            "♟️ لون قطعك:", ["⚪ الفاتح", "⚫ الداكن"]
        )
        player = P.L if "الفاتح" in my_color else P.D
        think_time = st.select_slider(
            "⏱ وقت التحليل:", [1, 2, 3, 5, 8, 10], value=3
        )
        speed_label = "⚡ سريع" if evaluator.mode == "xgboost" else "🔧 عادي"
        st.info(f"⏱ {think_time}s • {speed_label}")

        # رفع النموذج
        if evaluator.mode != "xgboost":
            st.divider()
            st.markdown("### 📤 رفع النموذج")
            uploaded_model = st.file_uploader(
                "ارفع brutal_ai_model.json", type=["json"]
            )
            if uploaded_model:
                with open(MODEL_FILE, "wb") as f:
                    f.write(uploaded_model.read())
                st.session_state.evaluator = HybridEvaluator()
                st.success("✅ تم تفعيل XGBoost!")
                st.rerun()

        st.divider()
        if st.button("🔄 رقعة ابتدائية", use_container_width=True):
            st.session_state.board = Engine._init().tolist()
            st.rerun()
        if st.button("🗑️ مسح", use_container_width=True):
            st.session_state.board = np.zeros((8, 8), dtype=int).tolist()
            st.rerun()

        st.divider()
        eng_ = Engine(st.session_state.board)
        ln, lk = eng_.count(P.L)
        dn, dk = eng_.count(P.D)
        st.markdown("### 📊 القطع")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("⚪", ln + lk, delta=f"👑{lk}" if lk else None)
        with c2:
            st.metric("⚫", dn + dk, delta=f"👑{dk}" if dk else None)

    # ═══ التبويبات ═══
    tab_names = ["✏️ إدخال يدوي"]
    if HAS_CV2:
        tab_names.append("📷 تحليل صورة")
    tab_names.append("🧠 التحليل")
    tabs = st.tabs(tab_names)

    # ─── إدخال يدوي ───
    with tabs[0]:
        st.markdown("**اختر قطعة واضغط المربع:**")
        opts = {
            "⬜ فارغ": int(P.E),
            "⚪ فاتح": int(P.L),
            "⚫ داكن": int(P.D),
            "👑W ملك فاتح": int(P.LK),
            "♛B ملك داكن": int(P.DK),
        }
        sel = st.radio(
            "_", list(opts.keys()), horizontal=True, label_visibility="collapsed"
        )
        sv = opts[sel]
        syms = {
            int(P.E): "·",
            int(P.L): "⚪",
            int(P.D): "⚫",
            int(P.LK): "👑",
            int(P.DK): "♛",
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
        st.image(Render.draw(ba), caption="الرقعة الحالية", use_container_width=True)

    # ─── تحليل صورة ───
    if HAS_CV2:
        with tabs[1]:
            st.subheader("📷 تحليل صورة الرقعة")
            uploaded = st.file_uploader("📸 ارفع صورة", type=["jpg", "png", "jpeg"])
            if uploaded:
                pil = Image.open(uploaded).convert("RGB")
                img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                c1, c2 = st.columns(2)
                with c1:
                    st.image(pil, caption="الأصلية", use_container_width=True)
                with st.spinner("🔲 تصحيح المنظور..."):
                    fixed, was = Vision.fix_perspective(img_cv)
                    fp = Image.fromarray(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))
                with c2:
                    st.image(
                        fp, caption=("✅ مصحح" if was else "📐 مقتصة"),
                        use_container_width=True
                    )
                tc1, tc2 = st.columns(2)
                with tc1:
                    lt = st.slider("عتبة الفاتح", 100, 230, 160)
                with tc2:
                    dt = st.slider("عتبة الداكن", 30, 150, 100)

                if st.button("🔍 تحليل", type="primary"):
                    with st.spinner("🧠 جاري التحليل..."):
                        hb, hi = Vision.detect_hsv(fixed, lt, dt)
                        cb, cv_ = Vision.detect_circles(fixed, lt, dt)
                        merged = Vision.merge(hb, cb)
                    st.success("✅ تم!")
                    t1, t2, t3 = st.tabs(["🎨 HSV", "⭕ دوائر", "🔀 مدمج"])
                    with t1:
                        st.image(Render.draw(hb), use_container_width=True)
                        with st.expander("تفاصيل"):
                            for d in hi:
                                if d["d"] != 0:
                                    st.text(
                                        f"({d['r']},{d['c']}) "
                                        f"{'⚪' if d['d']==1 else '⚫'} "
                                        f"br={d['br']} var={d['va']}"
                                    )
                    with t2:
                        st.image(cv_, use_container_width=True)
                        st.image(Render.draw(cb), use_container_width=True)
                    with t3:
                        st.image(Render.draw(merged), use_container_width=True)
                        e = Engine(merged)
                        a, b2 = e.count(P.L)
                        c3, d2 = e.count(P.D)
                        st.info(f"⚪ {a}+{b2}👑 • ⚫ {c3}+{d2}👑")
                        if st.button("📥 استخدم للتحليل", type="primary"):
                            st.session_state.board = merged.tolist()
                            st.rerun()

    # ─── التحليل ───
    analyze_idx = 2 if HAS_CV2 else 1
    with tabs[analyze_idx]:
        ba2 = np.array(st.session_state.board, dtype=np.int8)
        eng = Engine(ba2)
        st.image(Render.draw(ba2), caption="الرقعة", use_container_width=True)

        ln, lk = eng.count(P.L)
        dn, dk = eng.count(P.D)
        if (ln + lk) == 0 and (dn + dk) == 0:
            st.warning("⚠️ الرقعة فارغة!")
            return
        go = eng.game_over()
        if go is not None:
            if go == -1:
                st.info("🤝 تعادل")
            elif go == P.L:
                st.success("🏆 فاز الفاتح!")
            else:
                st.success("🏆 فاز الداكن!")
            return

        emoji = "⚪" if player == P.L else "⚫"
        st.markdown(
            f"### {emoji} تحليل حركات **{'الفاتح' if player == P.L else 'الداكن'}**"
        )

        if st.button("🧠 حلّل الآن!", type="primary", use_container_width=True):
            progress = st.empty()
            progress.info(f"🧠 جاري التحليل ({think_time}s)...")
            ai = FastAI(evaluator, max_time=think_time)
            analysis = ai.analyze_all(eng, player)
            progress.empty()

            if not analysis or not analysis["moves"]:
                st.error("❌ لا حركات!")
                return

            moves = analysis["moves"]
            best = moves[0]

            # ── شريط التقييم ──
            pe = analysis["pos_eval"]
            if pe > 200:
                em, ec = "🟢 متفوق بوضوح!", "#28a745"
            elif pe > 50:
                em, ec = "🟢 أفضل قليلاً", "#20c997"
            elif pe > -50:
                em, ec = "🟡 متكافئة", "#ffc107"
            elif pe > -200:
                em, ec = "🟠 الخصم أفضل", "#fd7e14"
            else:
                em, ec = "🔴 خطر!", "#dc3545"
            pct = max(5, min(95, int(50 + pe / 15)))
            st.markdown(
                f'<div class="eval-bar">'
                f'<div class="eval-fill" style="width:{pct}%;background:{ec}">'
                f'{em} ({pe})</div></div>',
                unsafe_allow_html=True
            )

            if analysis["is_forced"]:
                st.markdown(
                    '<div class="warn-box">⚡ أكل إجباري!</div>',
                    unsafe_allow_html=True
                )

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
<p>تقييم: {best['score']} • فوز: {best['win_pct']}% • عمق: {best['depth']}</p>
</div>
""", unsafe_allow_html=True)

            # ── صور ──
            c1, c2 = st.columns(2)
            with c1:
                bi = Render.draw(
                    ba2,
                    arrows=[{
                        "move": best["move"],
                        "color": (50, 205, 50),
                        "width": 6
                    }],
                    highlight=set(best["move"]),
                    label="BEST MOVE"
                )
                st.image(bi, caption="🏆 أفضل حركة", use_container_width=True)
            with c2:
                af = eng.copy()
                af.do_move(best["move"])
                st.image(Render.draw(af.board), caption="📋 بعد التنفيذ", use_container_width=True)

            if st.button("✅ طبّق الحركة", use_container_width=True):
                st.session_state.board = af.board.tolist()
                st.rerun()

            # ── أداء ──
            st.markdown(f"""
<div class="perf-box">
⚡ <b>محرك:</b> {evaluator.mode.upper()} • الوقت: {analysis['time']}s • أفضل عمق: {best['depth']}
</div>
""", unsafe_allow_html=True)

            # ── كل الحركات ──
            st.markdown(f"### 📊 تصنيف الحركات ({len(moves)})")
            rank_colors = {
                1: ("🥇", (50, 205, 50)),
                2: ("🥈", (65, 105, 225)),
                3: ("🥉", (255, 165, 0)),
            }
            top = []
            for i, mv in enumerate(moves[:5]):
                _, clr = rank_colors.get(
                    mv["rank"], (f"#{mv['rank']}", (180, 180, 180))
                )
                top.append({
                    "move": mv["move"],
                    "color": clr,
                    "width": 6 if i == 0 else 3
                })
            st.image(
                Render.draw(ba2, arrows=top),
                caption="🥇أخضر 🥈أزرق 🥉برتقالي",
                use_container_width=True
            )
            for mv in moves:
                icon = rank_colors.get(mv["rank"], (f"#{mv['rank']}", None))[0]
                path = " → ".join(f"({p[0]},{p[1]})" for p in mv["move"])
                bar = "█" * max(1, int(mv["win_pct"] / 5))
                with st.expander(
                    f"{icon} {path} • {mv['score']} • {mv['win_pct']}% • {mv['verdict']}"
                ):
                    st.markdown(f"**المسار:** `{path}`")
                    st.markdown(f"**الحكم:** {mv['verdict']}")
                    st.markdown(f"**فوز:** `{bar}` {mv['win_pct']}%")
                    st.markdown(f"**عمق:** {mv['depth']}")
                    mi = Render.draw(
                        ba2,
                        arrows=[{
                            "move": mv["move"],
                            "color": (255, 100, 50),
                            "width": 5
                        }],
                        highlight=set(mv["move"])
                    )
                    st.image(mi, use_container_width=True)
            st.divider()

    st.markdown(
        '<p style="text-align:center;color:#999;font-size:0.8em">'
        '♟️ مساعد الداما الذكي — XGBoost + Minimax + Alpha-Beta + PVS + Quiescence + OpenCV — بدون إنترنت</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
