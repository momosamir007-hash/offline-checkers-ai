#!/usr/bin/env python3
"""
╔═════════════════════════════════════════════════════════╗
║ ♟️ مساعد الداما الذكي — النسخة السحابية النهائية        ║
║ Super Engine (Depth 8+) + Memory (TT) + OpenCV          ║
║ لا يحتاج إلى تدريب أو ملفات خارجية! يعمل مباشرة         ║
╚═════════════════════════════════════════════════════════╝
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
# 1. الثوابت وجداول التقييم الموضعي
# ══════════════════════════════════════════
class P(IntEnum):
    E = 0; L = 1; D = 2; LK = 3; DK = 4

CELL = 80
BOARD_PX = CELL * 8

EXACT, LOWER, UPPER = 0, 1, 2

# جداول تشجع القطع على التقدم وحماية الحواف
LIGHT_POS = np.array([
    [0,  0,  0,  0,  0,  0,  0,  0], 
    [0,  0, 10,  0, 10,  0, 10,  0],
    [0,  8,  0,  9,  0,  9,  0,  0],
    [0,  0,  7,  0,  7,  0,  6,  0],
    [0,  5,  0,  7,  0,  5,  0,  0],
    [0,  0,  5,  0,  5,  0,  4,  0],
    [0,  3,  0,  3,  0,  3,  0,  0],
    [0,  0,  1,  0,  1,  0,  1,  0], 
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
# 2. محرك القواعد الأساسي
# ══════════════════════════════════════════
class Engine:
    def __init__(self, board=None):
        self.board = np.array(board, dtype=np.int8) if board is not None else self._init()

    @staticmethod
    def _init():
        b = np.zeros((8, 8), dtype=np.int8)
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 != 0:
                    if r < 3: b[r][c] = P.D
                    elif r > 4: b[r][c] = P.L
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
        if player in (P.L, P.LK): return piece in (P.L, P.LK)
        return piece in (P.D, P.DK)

    @staticmethod
    def enemy(piece, player):
        if piece == P.E: return False
        if player in (P.L, P.LK): return piece in (P.D, P.DK)
        return piece in (P.L, P.LK)

    @staticmethod
    def opp(player): return P.D if player in (P.L, P.LK) else P.L

    @staticmethod
    def dirs(piece):
        if piece == P.L: return ((-1, -1), (-1, 1))
        if piece == P.D: return ((1, -1), (1, 1))
        if piece in (P.LK, P.DK): return ((-1, -1), (-1, 1), (1, -1), (1, 1))
        return ()

    def _simple(self, r, c):
        p = self.board[r][c]; out = []
        for dr, dc in self.dirs(p):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == P.E:
                out.append(((r, c), (nr, nc)))
        return out

    def _jumps(self, r, c, bd=None, eaten=None):
        if bd is None: bd = self.board
        if eaten is None: eaten = frozenset()
        p = bd[r][c]; chains = []
        for dr, dc in self.dirs(p):
            mr, mc = r + dr, c + dc; nr, nc = r + 2 * dr, c + 2 * dc
            if not (0 <= nr < 8 and 0 <= nc < 8): continue
            if bd[nr][nc] != P.E or not self.enemy(bd[mr][mc], p) or (mr, mc) in eaten: continue
            
            nb = bd.copy()
            nb[nr][nc] = p; nb[r][c] = P.E; nb[mr][mc] = P.E
            promo = False
            if nr == 0 and p == P.L: nb[nr][nc] = P.LK; promo = True
            elif nr == 7 and p == P.D: nb[nr][nc] = P.DK; promo = True
            
            ne = eaten | {(mr, mc)}
            fur = [] if promo else self._jumps(nr, nc, nb, ne)
            if fur:
                for ch in fur: chains.append(((r, c),) + ch)
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
            sr, sc = move[i]; er, ec = move[i + 1]
            dr, dc = er - sr, ec - sc
            if abs(dr) == 2 and abs(dc) == 2:
                self.board[sr + dr // 2][sc + dc // 2] = P.E
        fr, fc = move[-1]
        self.board[fr][fc] = piece
        if fr == 0 and piece == P.L: self.board[fr][fc] = P.LK
        if fr == 7 and piece == P.D: self.board[fr][fc] = P.DK

    def count(self, player):
        n = k = 0
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if self.owns(p, player):
                    if self.is_king(p): k += 1
                    else: n += 1
        return n, k

    def game_over(self):
        ln, lk = self.count(P.L); dn, dk = self.count(P.D)
        if ln + lk == 0: return P.D
        if dn + dk == 0: return P.L
        lm, _ = self.get_moves(P.L); dm, _ = self.get_moves(P.D)
        if not lm and not dm: return -1
        if not lm: return P.D
        if not dm: return P.L
        return None

# ══════════════════════════════════════════
# 3. الذاكرة السحابية للمحرك (TT & Zobrist)
# ══════════════════════════════════════════
class ZobristHash:
    def __init__(self):
        rng = random.Random(42)
        self.keys = {(p, r, c): rng.getrandbits(64) for p in range(1, 5) for r in range(8) for c in range(8)}
        self.side_key = rng.getrandbits(64)

    def compute(self, board, is_max_side):
        h = 0
        for r in range(8):
            for c in range(8):
                p = int(board[r][c])
                if p != 0: h ^= self.keys[(p, r, c)]
        if is_max_side: h ^= self.side_key
        return h

class TransTable:
    def __init__(self, max_size=200000):
        self.table = {}
        self.max_size = max_size
        self.hits = 0

    def probe(self, key, depth, alpha, beta):
        e = self.table.get(key)
        if e and e[0] >= depth:
            self.hits += 1
            s, f, m = e[1], e[2], e[3]
            if f == EXACT: return s, m
            if f == LOWER and s >= beta: return s, m
            if f == UPPER and s <= alpha: return s, m
        return None

    def store(self, key, depth, score, flag, move):
        if len(self.table) > self.max_size: self.table.clear() # تفريغ لتجنب استهلاك الرام السحابي
        self.table[key] = (depth, score, flag, move)

# ══════════════════════════════════════════
# 4. الذكاء الاصطناعي (Super AI)
# ══════════════════════════════════════════
class SuperAI:
    def __init__(self, max_time, zobrist, tt):
        self.max_time = max_time
        self.zobrist = zobrist
        self.tt = tt
        self.nodes = 0
        self.start_time = 0
        self.time_up = False
        self.killers = [[None, None] for _ in range(50)]
        self.history = {}

    def evaluate(self, eng, player):
        b = eng.board
        opp = eng.opp(player)
        my_men = (b == P.L) if player in (P.L, P.LK) else (b == P.D)
        my_king = (b == P.LK) if player in (P.L, P.LK) else (b == P.DK)
        op_men = (b == P.D) if player in (P.L, P.LK) else (b == P.L)
        op_king = (b == P.DK) if player in (P.L, P.LK) else (b == P.LK)

        mn, mk = int(np.sum(my_men)), int(np.sum(my_king))
        on, ok_ = int(np.sum(op_men)), int(np.sum(op_king))

        if mn + mk == 0: return -99999
        if on + ok_ == 0: return 99999

        score = (mn * 100 + mk * 300) - (on * 100 + ok_ * 300)
        
        if player in (P.L, P.LK):
            score += float(np.sum(my_men * LIGHT_POS)) * 3
            score -= float(np.sum(op_men * DARK_POS)) * 3
        else:
            score += float(np.sum(my_men * DARK_POS)) * 3
            score -= float(np.sum(op_men * LIGHT_POS)) * 3

        score += float(np.sum(my_king * KING_POS)) * 2 - float(np.sum(op_king * KING_POS)) * 2
        return score

    def check_time(self):
        if self.nodes % 1000 == 0 and time.time() - self.start_time >= self.max_time:
            self.time_up = True

    def order_moves(self, moves, is_cap, depth, tt_move):
        scored = []
        for m in moves:
            s = 0
            if tt_move and m == tt_move: s += 100000
            elif is_cap: s += 50000 + len(m) * 1000
            elif depth < len(self.killers) and m in self.killers[depth]: s += 40000
            else: s += self.history.get((m[0], m[-1]), 0)
            scored.append((s, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def alpha_beta(self, eng, depth, alpha, beta, player, original):
        self.nodes += 1
        self.check_time()
        if self.time_up: return 0, None

        res = eng.game_over()
        if res is not None:
            if res == original: return 99999 + depth, None
            elif res == -1: return 0, None
            else: return -99999 - depth, None

        if depth <= 0: return self.evaluate(eng, original), None

        board_key = self.zobrist.compute(eng.board, player == original)
        tt_res = self.tt.probe(board_key, depth, alpha, beta)
        if tt_res: return tt_res

        moves, is_cap = eng.get_moves(player)
        if not moves: return self.evaluate(eng, original), None

        tt_move = self.tt.table.get(board_key)[3] if board_key in self.tt.table else None
        moves = self.order_moves(moves, is_cap, depth, tt_move)

        best_move = moves[0]; best_score = float("-inf"); orig_alpha = alpha
        
        for move in moves:
            child = eng.copy(); child.do_move(move)
            score = -self.alpha_beta(child, depth - 1, -beta, -alpha, eng.opp(player), original)[0]
            if self.time_up: break
            
            if score > best_score: best_score = score; best_move = move
            if score > alpha: alpha = score
            if alpha >= beta:
                if not is_cap and depth < len(self.killers):
                    k = self.killers[depth]
                    if move != k[0]: k[1] = k[0]; k[0] = move
                self.history[(move[0], move[-1])] = self.history.get((move[0], move[-1]), 0) + depth * depth
                break

        if not self.time_up:
            flag = UPPER if best_score <= orig_alpha else LOWER if best_score >= beta else EXACT
            self.tt.store(board_key, depth, best_score, flag, best_move)

        return best_score, best_move

    def analyze_all(self, eng, player):
        self.nodes = 0; self.start_time = time.time(); self.time_up = False
        moves, is_cap = eng.get_moves(player)
        if not moves: return None

        time_each = max(0.5, self.max_time * 0.8 / len(moves))
        results = []

        for move in moves:
            child = eng.copy(); child.do_move(move)
            
            best_s = 0; best_d = 0
            # Iterative Deepening for each move
            for d in range(1, 15):
                s, _ = self.alpha_beta(child, d, float("-inf"), float("inf"), eng.opp(player), eng.opp(player))
                if self.time_up or time.time() - self.start_time > time_each * (len(results)+1): break
                best_s = -s; best_d = d

            is_capture = (len(move) > 2 or (len(move) == 2 and abs(move[0][0] - move[1][0]) == 2))
            cap_count = sum(1 for i in range(len(move)-1) if abs(move[i][0] - move[i+1][0]) == 2) if is_capture else 0
            dest, piece = move[-1], eng.board[move[0][0]][move[0][1]]
            promotes = (dest[0] == 0 and piece == P.L) or (dest[0] == 7 and piece == P.D)

            verdict = "🔥 ممتاز!" if is_capture and cap_count >= 2 else "👑 ترقية!" if promotes else "💪 قوي" if best_s > 100 else "✅ جيد" if best_s > 20 else "⚖️ متكافئ" if best_s > -50 else "⚠️ مخاطرة" if best_s > -150 else "❌ خطأ فادح"
            
            results.append({
                "move": move, "score": round(best_s, 1), "depth": best_d,
                "is_capture": is_capture, "captured": cap_count, "promotes": promotes, "verdict": verdict
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
            s = r["score"]
            r["win_pct"] = 99 if s > 5000 else 1 if s < -5000 else max(1, min(99, int(50 + s / 15)))

        return {
            "moves": results, "time": round(time.time() - self.start_time, 2),
            "is_forced": is_cap, "pos_eval": self.evaluate(eng, player)
        }

# ══════════════════════════════════════════
# 5. تحليل الصور (OpenCV)
# ══════════════════════════════════════════
class Vision:
    @staticmethod
    def fix_perspective(img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cnts, _ = cv2.findContours(cv2.dilate(edges, k, iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return cv2.resize(img_bgr, (400, 400)), False
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) < img_bgr.shape[0] * img_bgr.shape[1] * 0.15: return cv2.resize(img_bgr, (400, 400)), False
        approx = cv2.approxPolyDP(largest, 0.02 * cv2.arcLength(largest, True), True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
            o = np.zeros((4, 2), dtype=np.float32)
            o[0], o[2], o[1], o[3] = pts[np.argmin(s)], pts[np.argmax(s)], pts[np.argmin(d)], pts[np.argmax(d)]
            dst = np.float32([[0, 0], [399, 0], [399, 399], [0, 399]])
            return cv2.warpPerspective(img_bgr, cv2.getPerspectiveTransform(o, dst), (400, 400)), True
        x, y, w, h = cv2.boundingRect(largest)
        return cv2.resize(img_bgr[y:y + h, x:x + w], (400, 400)), False

    @staticmethod
    def detect_hsv(img, lt=160, dt=100):
        hsv, gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = 50; board = np.zeros((8, 8), dtype=np.int8); info = []
        for r in range(8):
            for col in range(8):
                if (r + col) % 2 == 0: continue
                m = c // 4
                roi_g, roi_h = gray[r*c+m:(r+1)*c-m, col*c+m:(col+1)*c-m], hsv[r*c+m:(r+1)*c-m, col*c+m:(col+1)*c-m]
                br, sa, va = float(np.mean(roi_g)), float(np.mean(roi_h[:, :, 1])), float(np.var(roi_g))
                det = int(P.L) if (va > 120 or sa > 30) and br > lt else int(P.D) if (va > 120 or sa > 30) and br < dt else int(P.E)
                board[r][col] = det
                info.append({"r": r, "c": col, "br": round(br), "sa": round(sa), "va": round(va), "d": det})
        return board, info

    @staticmethod
    def detect_circles(img, lt=160, dt=100):
        gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5); c = 50
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 35, param1=60, param2=35, minRadius=12, maxRadius=24)
        board = np.zeros((8, 8), dtype=np.int8); vis = img.copy()
        if circles is not None:
            for cx, cy, rad in np.uint16(np.around(circles))[0]:
                co, ro = int(cx / c), int(cy / c)
                if not (0 <= ro < 8 and 0 <= co < 8) or (ro + co) % 2 == 0: continue
                s = gray[max(0, int(cy) - 5):min(400, int(cy) + 5), max(0, int(cx) - 5):min(400, int(cx) + 5)]
                if s.size == 0: continue
                a = float(np.mean(s))
                if a > lt: board[ro][co] = int(P.L); cv2.circle(vis, (int(cx), int(cy)), int(rad), (0, 255, 0), 2)
                elif a < dt: board[ro][co] = int(P.D); cv2.circle(vis, (int(cx), int(cy)), int(rad), (0, 0, 255), 2)
        return board, Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    @staticmethod
    def merge(a, b):
        m = np.zeros((8, 8), dtype=np.int8)
        for r in range(8):
            for c in range(8):
                m[r][c] = a[r][c] if a[r][c] != P.E and b[r][c] != P.E else b[r][c] if b[r][c] != P.E else a[r][c]
        return m

# ══════════════════════════════════════════
# 6. رسم الرقعة
# ══════════════════════════════════════════
class Render:
    @staticmethod
    def draw(board, arrows=None, highlight=None):
        img = Image.new("RGB", (BOARD_PX, BOARD_PX))
        dr = ImageDraw.Draw(img)
        for r in range(8):
            for c in range(8):
                x1, y1, x2, y2 = c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL
                sq = (235, 215, 180) if (r + c) % 2 == 0 else (175, 130, 95)
                if highlight and (r, c) in highlight: sq = (100, 200, 100)
                dr.rectangle([x1, y1, x2, y2], fill=sq)
                p = board[r][c]
                if p == P.E: continue
                cx, cy, pr = x1 + CELL // 2, y1 + CELL // 2, CELL // 2 - 10
                dr.ellipse([cx - pr + 3, cy - pr + 3, cx + pr + 3, cy + pr + 3], fill=(70, 50, 30))
                fl, ed = ((250, 248, 240), (195, 185, 170)) if Engine.is_light(p) else ((45, 45, 45), (25, 25, 25))
                dr.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=fl, outline=ed, width=2)
                dr.ellipse([cx - pr + 5, cy - pr + 5, cx + pr - 5, cy + pr - 5], outline=ed, width=1)
                if Engine.is_king(p): dr.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(255, 215, 0), outline=(200, 170, 0), width=2)
        
        for i in range(8):
            dr.text((3, i * CELL + 3), str(i), fill=(130, 110, 90))
            dr.text((i * CELL + CELL // 2 - 4, BOARD_PX - 14), chr(65 + i), fill=(130, 110, 90))
        
        if arrows:
            for a in arrows:
                m, clr, w = a["move"], a.get("color", (255, 50, 50)), a.get("width", 5)
                for i in range(len(m) - 1):
                    sx, sy, ex, ey = m[i][1]*CELL+CELL//2, m[i][0]*CELL+CELL//2, m[i+1][1]*CELL+CELL//2, m[i+1][0]*CELL+CELL//2
                    dr.line([(sx, sy), (ex, ey)], fill=clr, width=w)
                    dr.ellipse([ex - 8, ey - 8, ex + 8, ey + 8], fill=clr)
                sx, sy = m[0][1]*CELL+CELL//2, m[0][0]*CELL+CELL//2
                dr.ellipse([sx - 12, sy - 12, sx + 12, sy + 12], outline=(0, 220, 0), width=4)
        return img

# ══════════════════════════════════════════
# 7. واجهة التطبيق
# ══════════════════════════════════════════
def main():
    st.set_page_config("♟️ مساعد الداما الاحترافي", "♟️", layout="wide")
    st.markdown("""
    <style>
    .block-container{max-width:1100px}
    .best-box{background:linear-gradient(135deg,#28a745,#20c997); color:#fff;padding:18px;border-radius:12px; text-align:center;margin:10px 0}
    .warn-box{background:#f8d7da;border:2px solid #dc3545; color:#721c24;padding:12px;border-radius:10px; text-align:center;margin:8px 0;font-weight:bold}
    .eval-bar{background:#e9ecef;border-radius:8px; overflow:hidden;height:30px;margin:8px 0}
    .eval-fill{height:100%;text-align:center;color:#fff; font-weight:bold;line-height:30px;border-radius:8px}
    .perf-box{background:#f0f2f6;padding:12px; border-radius:10px;border-left:4px solid #667eea; margin:8px 0}
    </style>
    """, unsafe_allow_html=True)

    # ── تهيئة ذاكرة الذكاء الاصطناعي السحابية ──
    if "zobrist" not in st.session_state: st.session_state.zobrist = ZobristHash()
    if "tt" not in st.session_state: st.session_state.tt = TransTable()
    if "board" not in st.session_state: st.session_state.board = Engine._init().tolist()

    st.title("♟️ مساعد الداما الاحترافي (Cloud Engine)")
    st.caption("مدمج بمحرك رياضي خارق لا يحتاج لملفات خارجية. أداة تحليل نهائية!")

    with st.sidebar:
        st.header("⚙️ الإعدادات")
        player = P.L if "الفاتح" in st.radio("♟️ لون قطعك:", ["⚪ الفاتح", "⚫ الداكن"]) else P.D
        think_time = st.select_slider("⏱ وقت التفكير (كلما زاد أصبح أذكى):", [1, 2, 3, 5, 8], value=3)
        
        st.divider()
        if st.button("🔄 رقعة ابتدائية", use_container_width=True):
            st.session_state.board = Engine._init().tolist(); st.rerun()
        if st.button("🗑️ مسح الرقعة", use_container_width=True):
            st.session_state.board = np.zeros((8, 8), dtype=int).tolist(); st.rerun()

        st.divider()
        eng_ = Engine(st.session_state.board)
        ln, lk = eng_.count(P.L); dn, dk = eng_.count(P.D)
        st.markdown("### 📊 القطع المتبقية")
        c1, c2 = st.columns(2)
        c1.metric("⚪", ln + lk, delta=f"👑{lk}" if lk else None)
        c2.metric("⚫", dn + dk, delta=f"👑{dk}" if dk else None)

    tabs = st.tabs(["✏️ إدخال يدوي", "📷 تحليل صورة", "🧠 التحليل الذكي"])

    with tabs[0]:
        opts = {"⬜ فارغ": int(P.E), "⚪ فاتح": int(P.L), "⚫ داكن": int(P.D), "👑 ملك أبيض": int(P.LK), "♛ ملك أسود": int(P.DK)}
        sv = opts[st.radio("_", list(opts.keys()), horizontal=True, label_visibility="collapsed")]
        ba = np.array(st.session_state.board)
        for r in range(8):
            cols = st.columns(8)
            for c in range(8):
                with cols[c]:
                    if (r + c) % 2 != 0:
                        if st.button({0:"·", 1:"⚪", 2:"⚫", 3:"👑", 4:"♛"}[int(ba[r][c])], key=f"m{r}{c}", use_container_width=True):
                            st.session_state.board[r][c] = sv; st.rerun()
        st.image(Render.draw(ba), caption="الرقعة", use_container_width=True)

    with tabs[1]:
        if HAS_CV2:
            uploaded = st.file_uploader("📸 ارفع صورة", type=["jpg", "png", "jpeg"])
            if uploaded:
                img_cv = cv2.cvtColor(np.array(Image.open(uploaded).convert("RGB")), cv2.COLOR_RGB2BGR)
                fixed, was = Vision.fix_perspective(img_cv)
                st.image(Image.fromarray(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)), caption="✅ مصحح" if was else "مقتصة", use_container_width=True)
                tc1, tc2 = st.columns(2)
                lt = tc1.slider("عتبة الفاتح", 100, 230, 160)
                dt = tc2.slider("عتبة الداكن", 30, 150, 100)
                
                # الزر الأول: معالجة الصورة وحفظ النتيجة في الذاكرة المؤقتة
                if st.button("🔍 استخراج الرقعة", type="primary"):
                    with st.spinner("جاري الرؤية..."):
                        merged = Vision.merge(Vision.detect_hsv(fixed, lt, dt)[0], Vision.detect_circles(fixed, lt, dt)[0])
                        st.session_state.temp_merged_board = merged.tolist()
                
                # عرض الرقعة المؤقتة إذا كانت موجودة وإظهار زر الاعتماد المستقل
                if "temp_merged_board" in st.session_state:
                    st.image(Render.draw(st.session_state.temp_merged_board), use_container_width=True)
                    if st.button("📥 اعتماد الرقعة", type="primary"):
                        st.session_state.board = st.session_state.temp_merged_board
                        del st.session_state.temp_merged_board
                        st.rerun()
        else:
            st.error("مكتبة OpenCV غير متوفرة في بيئة الاستضافة.")

    with tabs[2]:
        ba2 = np.array(st.session_state.board, dtype=np.int8)
        eng = Engine(ba2)
        st.image(Render.draw(ba2), use_container_width=True)

        if eng.game_over() is not None:
            st.success("🏆 اللعبة منتهية!")
            return

        if st.button("🧠 استخرج أفضل حركة!", type="primary", use_container_width=True):
            with st.spinner(f"يُفكر كالمحترفين... ({think_time} ثوانٍ)"):
                ai = SuperAI(max_time=think_time, zobrist=st.session_state.zobrist, tt=st.session_state.tt)
                analysis = ai.analyze_all(eng, player)

            if not analysis or not analysis["moves"]:
                st.error("❌ لا يوجد حركات متاحة!")
                return

            best = analysis["moves"][0]
            pe = analysis["pos_eval"]
            pct = max(5, min(95, int(50 + pe / 6)))
            st.markdown(f'<div class="eval-bar"><div class="eval-fill" style="width:{pct}%;background:{"#28a745" if pe>50 else "#ffc107" if pe>-50 else "#dc3545"}">{"🟢 متفوق" if pe>50 else "🟡 متكافئ" if pe>-50 else "🔴 خطر"} ({pe})</div></div>', unsafe_allow_html=True)

            if analysis["is_forced"]: st.markdown('<div class="warn-box">⚡ أكل إجباري!</div>', unsafe_allow_html=True)

            path = " → ".join(f"({p[0]},{p[1]})" for p in best["move"])
            st.markdown(f'<div class="best-box"><h2>🏆 {path}</h2><p>{best["verdict"]} • تقييم: {best["score"]}</p></div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.image(Render.draw(ba2, arrows=[{"move": best["move"], "color": (50, 205, 50), "width": 6}], highlight=set(best["move"])), caption="🏆 الحركة", use_container_width=True)
            af = eng.copy(); af.do_move(best["move"])
            c2.image(Render.draw(af.board), caption="📋 النتيجة", use_container_width=True)

            st.markdown(f'<div class="perf-box">⚡ <b>الأداء:</b> الوقت: {analysis["time"]}s • ذاكرة AI الحالية: {len(st.session_state.tt.table):,} وضعية محفوظة!</div>', unsafe_allow_html=True)

            if st.button("✅ العب هذه الحركة", use_container_width=True):
                st.session_state.board = af.board.tolist(); st.rerun()

            for mv in analysis["moves"][1:4]:
                with st.expander(f"بديل: {' → '.join(f'({p[0]},{p[1]})' for p in mv['move'])} • {mv['verdict']}"):
                    st.image(Render.draw(ba2, arrows=[{"move": mv["move"], "color": (255, 100, 50), "width": 5}], highlight=set(mv["move"])), use_container_width=True)

if __name__ == "__main__":
    main()
