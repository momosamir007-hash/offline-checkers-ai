#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════╗
║        ♟️ مساعد الداما الذكي v5.0                 ║
║   حمّل صورة أو أدخل الرقعة → يريك كيف تفوز      ║
║   ليس لعبة — بل محلل شرس يضمن لك الفوز          ║
╚═══════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
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
    E = 0       # فارغ
    L = 1       # فاتح
    D = 2       # داكن
    LK = 3      # ملك فاتح
    DK = 4      # ملك داكن

CELL = 80
BOARD_PX = CELL * 8

# جدول القيمة الموضعية لكل خلية
POS_VALUE = np.array([
    [0, 4, 0, 4, 0, 4, 0, 4],
    [4, 0, 3, 0, 3, 0, 3, 0],
    [0, 3, 0, 5, 0, 5, 0, 3],
    [3, 0, 5, 0, 6, 0, 5, 0],
    [0, 5, 0, 6, 0, 5, 0, 3],
    [3, 0, 5, 0, 5, 0, 3, 0],
    [0, 3, 0, 3, 0, 3, 0, 4],
    [4, 0, 4, 0, 4, 0, 4, 0],
], dtype=np.float32)


# ══════════════════════════════════════════
# محرك القواعد الكامل
# ══════════════════════════════════════════

class Engine:

    def __init__(self, board=None):
        if board is not None:
            self.board = np.array(board, dtype=np.int8)
        else:
            self.board = self._init()

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
            return [(-1, -1), (-1, 1)]
        if piece == P.D:
            return [(1, -1), (1, 1)]
        if piece in (P.LK, P.DK):
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return []

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
# الذكاء الاصطناعي الشرس
# ══════════════════════════════════════════

class BrutalAI:
    """
    محلل شرس بـ 10 عوامل تقييم:
    1. المادة (قطع + ملوك بأوزان عالية)
    2. الموقع (جدول POS_VALUE)
    3. التقدم نحو الترقية
    4. السيطرة على المركز
    5. حماية الصف الخلفي
    6. حماية الأجناب
    7. ترابط القطع
    8. حرية الحركة
    9. تهديد الأكل
    10. استراتيجية نهاية اللعبة
    """

    def __init__(self, depth=6):
        self.depth = depth
        self.nodes = 0

    def evaluate(self, eng, player):
        score = 0.0
        opp = eng.opp(player)
        total = 0

        my_n, my_k = eng.count(player)
        op_n, op_k = eng.count(opp)
        my_total = my_n + my_k
        op_total = op_n + op_k
        total = my_total + op_total

        # هل نحن في نهاية اللعبة؟
        endgame = total <= 10

        for r in range(8):
            for c in range(8):
                p = eng.board[r][c]
                if p == P.E:
                    continue

                val = 0.0

                # 1. المادة
                if eng.is_king(p):
                    val = 350.0 if endgame else 300.0
                else:
                    val = 100.0

                # 2. الموقع
                val += POS_VALUE[r][c] * 4

                # 3. التقدم نحو الترقية
                if not eng.is_king(p):
                    if p == P.L:
                        progress = 7 - r
                        val += progress * 8
                        if r <= 1:
                            val += 25  # قريب جداً من الترقية
                    elif p == P.D:
                        progress = r
                        val += progress * 8
                        if r >= 6:
                            val += 25

                # 4. المركز
                if 2 <= r <= 5 and 2 <= c <= 5:
                    val += 10
                    if 3 <= r <= 4 and 3 <= c <= 4:
                        val += 8

                # 5. الصف الخلفي
                if not eng.is_king(p):
                    if p == P.L and r == 7:
                        val += 15
                    elif p == P.D and r == 0:
                        val += 15

                # 6. الأجناب
                if c == 0 or c == 7:
                    val += 4

                # 7. الترابط
                allies = 0
                for dr2, dc2 in [(-1, -1), (-1, 1),
                                 (1, -1), (1, 1)]:
                    ar, ac = r + dr2, c + dc2
                    if 0 <= ar < 8 and 0 <= ac < 8:
                        if eng.owns(eng.board[ar][ac], p):
                            allies += 1
                val += allies * 5

                # 8. هل محمية من الأكل؟
                safe = True
                for dr2, dc2 in [(-1, -1), (-1, 1),
                                 (1, -1), (1, 1)]:
                    ar, ac = r + dr2, c + dc2
                    br, bc = r - dr2, c - dc2
                    if (0 <= ar < 8 and 0 <= ac < 8
                            and 0 <= br < 8 and 0 <= bc < 8):
                        if (eng.enemy(eng.board[ar][ac], p)
                                and eng.board[br][bc] == P.E):
                            safe = False
                            break
                if safe:
                    val += 6

                if eng.owns(p, player):
                    score += val
                else:
                    score -= val

        # 9. حرية الحركة + تهديد
        my_m, my_cap = eng.get_moves(player)
        op_m, op_cap = eng.get_moves(opp)
        score += len(my_m) * 6
        score -= len(op_m) * 6
        if my_cap:
            score += 30
        if op_cap:
            score -= 30

        # 10. استراتيجية نهاية اللعبة
        if endgame:
            my_mat = my_n + my_k * 3
            op_mat = op_n + op_k * 3
            if my_mat > op_mat:
                score += (my_mat - op_mat) * 15
                # ادفع نحو التبادل
                score += (20 - total) * 5

        return score

    def order_moves(self, moves, is_cap, eng, player):
        def key(m):
            s = 0
            if is_cap:
                s += len(m) * 300
            dest = m[-1]
            s += POS_VALUE[dest[0]][dest[1]] * 5

            # أولوية الترقية
            if player in (P.L, P.LK) and dest[0] == 0:
                s += 200
            if player in (P.D, P.DK) and dest[0] == 7:
                s += 200

            # المركز
            if 2 <= dest[0] <= 5 and 2 <= dest[1] <= 5:
                s += 20
            return s

        return sorted(moves, key=key, reverse=True)

    def minimax(self, eng, depth, alpha, beta,
                maximizing, current, original):
        self.nodes += 1

        result = eng.game_over()
        if result is not None:
            if result == original:
                return 99999 + depth, None
            elif result == -1:
                return 0, None
            else:
                return -99999 - depth, None

        if depth == 0:
            return self.evaluate(eng, original), None

        moves, is_cap = eng.get_moves(current)
        if not moves:
            return self.evaluate(eng, original), None

        moves = self.order_moves(moves, is_cap, eng, current)
        best = moves[0]
        nxt = eng.opp(current)

        if maximizing:
            mx = float("-inf")
            for m in moves:
                child = eng.copy()
                child.do_move(m)
                v, _ = self.minimax(
                    child, depth - 1, alpha, beta,
                    False, nxt, original
                )
                if v > mx:
                    mx = v
                    best = m
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return mx, best
        else:
            mn = float("inf")
            for m in moves:
                child = eng.copy()
                child.do_move(m)
                v, _ = self.minimax(
                    child, depth - 1, alpha, beta,
                    True, nxt, original
                )
                if v < mn:
                    mn = v
                    best = m
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return mn, best

    def analyze_all_moves(self, eng, player):
        """تحليل وتصنيف كل الحركات المتاحة"""
        self.nodes = 0
        t0 = time.time()

        moves, is_cap = eng.get_moves(player)
        if not moves:
            return []

        results = []
        for move in moves:
            child = eng.copy()
            child.do_move(move)

            score, _ = self.minimax(
                child, self.depth - 1,
                float("-inf"), float("inf"),
                False, eng.opp(player), player
            )

            # تصنيف الحركة
            is_capture = (
                len(move) > 2
                or (len(move) == 2
                    and abs(move[0][0] - move[1][0]) == 2)
            )

            captured_count = 0
            if is_capture:
                for i in range(len(move) - 1):
                    if abs(move[i][0] - move[i+1][0]) == 2:
                        captured_count += 1

            dest = move[-1]
            promotes = False
            piece = eng.board[move[0][0]][move[0][1]]
            if dest[0] == 0 and piece == P.L:
                promotes = True
            if dest[0] == 7 and piece == P.D:
                promotes = True

            results.append({
                "move": move,
                "score": round(score, 1),
                "is_capture": is_capture,
                "captured": captured_count,
                "promotes": promotes,
                "piece": int(piece),
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        elapsed = time.time() - t0

        # إضافة ترتيب ونسبة فوز تقديرية
        best_score = results[0]["score"] if results else 0
        for i, r in enumerate(results):
            r["rank"] = i + 1

            s = r["score"]
            if s > 5000:
                r["win_pct"] = 99
            elif s < -5000:
                r["win_pct"] = 1
            else:
                r["win_pct"] = max(1, min(99,
                    int(50 + s / 20)))

        return {
            "moves": results,
            "total_nodes": self.nodes,
            "time": round(elapsed, 2),
            "is_forced_capture": is_cap,
            "position_eval": round(
                self.evaluate(eng, player), 1
            ),
        }


# ══════════════════════════════════════════
# تحليل الصور المتقدم
# ══════════════════════════════════════════

class Vision:

    @staticmethod
    def fix_perspective(img_bgr):
        """تصحيح المنظور التلقائي"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)
        )
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return cv2.resize(img_bgr, (400, 400)), False

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        img_area = img_bgr.shape[0] * img_bgr.shape[1]

        if area < img_area * 0.15:
            return cv2.resize(img_bgr, (400, 400)), False

        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            s = pts.sum(axis=1)
            d = np.diff(pts, axis=1).ravel()

            ordered = np.zeros((4, 2), dtype=np.float32)
            ordered[0] = pts[np.argmin(s)]
            ordered[2] = pts[np.argmax(s)]
            ordered[1] = pts[np.argmin(d)]
            ordered[3] = pts[np.argmax(d)]

            dst = np.array(
                [[0, 0], [399, 0], [399, 399], [0, 399]],
                dtype=np.float32
            )
            M = cv2.getPerspectiveTransform(ordered, dst)
            warped = cv2.warpPerspective(
                img_bgr, M, (400, 400)
            )
            return warped, True

        x, y, w, h = cv2.boundingRect(largest)
        cropped = img_bgr[y:y + h, x:x + w]
        return cv2.resize(cropped, (400, 400)), False

    @staticmethod
    def detect_by_hsv(img_400, lt=160, dt=100):
        """كشف بتحليل HSV + Variance"""
        hsv = cv2.cvtColor(img_400, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_400, cv2.COLOR_BGR2GRAY)
        cell = 50

        board = np.zeros((8, 8), dtype=np.int8)
        details = []

        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 0:
                    continue

                m = cell // 4
                y1, y2 = r * cell + m, (r + 1) * cell - m
                x1, x2 = c * cell + m, (c + 1) * cell - m

                roi_g = gray[y1:y2, x1:x2]
                roi_h = hsv[y1:y2, x1:x2]

                bright = float(np.mean(roi_g))
                sat = float(np.mean(roi_h[:, :, 1]))
                var = float(np.var(roi_g))

                det = int(P.E)
                if var > 120 or sat > 30:
                    if bright > lt:
                        det = int(P.L)
                    elif bright < dt:
                        det = int(P.D)

                board[r][c] = det
                details.append({
                    "r": r, "c": c,
                    "bright": round(bright),
                    "sat": round(sat),
                    "var": round(var),
                    "det": det,
                })
        return board, details

    @staticmethod
    def detect_by_circles(img_400, lt=160, dt=100):
        """كشف بـ HoughCircles"""
        gray = cv2.cvtColor(img_400, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        cell = 50

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=35,
            param1=60, param2=35,
            minRadius=12, maxRadius=24
        )

        board = np.zeros((8, 8), dtype=np.int8)
        vis = img_400.copy()

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for cx, cy, radius in circles[0]:
                col = int(cx / cell)
                row = int(cy / cell)
                if not (0 <= row < 8 and 0 <= col < 8):
                    continue
                if (row + col) % 2 == 0:
                    continue

                sample = gray[
                    max(0, int(cy) - 5):min(400, int(cy) + 5),
                    max(0, int(cx) - 5):min(400, int(cx) + 5)
                ]
                if sample.size == 0:
                    continue
                avg = float(np.mean(sample))

                if avg > lt:
                    board[row][col] = int(P.L)
                    color = (0, 255, 0)
                elif avg < dt:
                    board[row][col] = int(P.D)
                    color = (0, 0, 255)
                else:
                    continue

                cv2.circle(
                    vis, (int(cx), int(cy)),
                    int(radius), color, 2
                )

        vis_pil = Image.fromarray(
            cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        )
        return board, vis_pil

    @staticmethod
    def merge(hsv_board, circle_board):
        """دمج ذكي"""
        merged = np.zeros((8, 8), dtype=np.int8)
        for r in range(8):
            for c in range(8):
                h, ci = hsv_board[r][c], circle_board[r][c]
                if h != P.E and ci != P.E and h == ci:
                    merged[r][c] = h
                elif ci != P.E:
                    merged[r][c] = ci
                elif h != P.E:
                    merged[r][c] = h
        return merged


# ══════════════════════════════════════════
# رسم الرقعة والأسهم
# ══════════════════════════════════════════

class Render:

    @staticmethod
    def draw_board(board, arrows=None, highlight=None,
                   label=None):
        """رسم الرقعة مع أسهم الحركات"""
        img = Image.new("RGB", (BOARD_PX, BOARD_PX))
        dr = ImageDraw.Draw(img)

        for r in range(8):
            for c in range(8):
                x1, y1 = c * CELL, r * CELL
                x2, y2 = x1 + CELL, y1 + CELL

                if (r + c) % 2 == 0:
                    sq = (235, 215, 180)
                else:
                    sq = (175, 130, 95)

                if highlight and (r, c) in highlight:
                    sq = (100, 200, 100)

                dr.rectangle([x1, y1, x2, y2], fill=sq)

                p = board[r][c]
                if p == P.E:
                    continue

                cx = x1 + CELL // 2
                cy = y1 + CELL // 2
                pr = CELL // 2 - 10

                # ظل
                dr.ellipse(
                    [cx - pr + 3, cy - pr + 3,
                     cx + pr + 3, cy + pr + 3],
                    fill=(70, 50, 30)
                )

                fl = ((250, 248, 240)
                      if Engine.is_light(p) else (45, 45, 45))
                ed = ((195, 185, 170)
                      if Engine.is_light(p) else (25, 25, 25))

                dr.ellipse(
                    [cx - pr, cy - pr, cx + pr, cy + pr],
                    fill=fl, outline=ed, width=2
                )
                dr.ellipse(
                    [cx - pr + 5, cy - pr + 5,
                     cx + pr - 5, cy + pr - 5],
                    outline=ed, width=1
                )

                if Engine.is_king(p):
                    kr = 12
                    dr.ellipse(
                        [cx - kr, cy - kr,
                         cx + kr, cy + kr],
                        fill=(255, 215, 0),
                        outline=(200, 170, 0), width=2
                    )

        # إحداثيات
        for i in range(8):
            try:
                dr.text(
                    (3, i * CELL + 3), str(i),
                    fill=(130, 110, 90)
                )
                dr.text(
                    (i * CELL + CELL // 2 - 4,
                     BOARD_PX - 14),
                    chr(65 + i), fill=(130, 110, 90)
                )
            except Exception:
                pass

        # أسهم
        if arrows:
            for arrow in arrows:
                move = arrow["move"]
                color = arrow.get("color", (255, 50, 50))
                width = arrow.get("width", 5)
                Render._draw_arrow(dr, move, color, width)

        # عنوان
        if label:
            try:
                dr.rectangle([0, 0, BOARD_PX, 22],
                             fill=(0, 0, 0, 180))
                dr.text((5, 3), label, fill=(255, 255, 255))
            except Exception:
                pass

        return img

    @staticmethod
    def _draw_arrow(draw, move, color, width):
        if not move or len(move) < 2:
            return

        for i in range(len(move) - 1):
            sr, sc = move[i]
            er, ec = move[i + 1]
            sx = sc * CELL + CELL // 2
            sy = sr * CELL + CELL // 2
            ex = ec * CELL + CELL // 2
            ey = er * CELL + CELL // 2

            draw.line(
                [(sx, sy), (ex, ey)],
                fill=color, width=width
            )
            draw.ellipse(
                [ex - 8, ey - 8, ex + 8, ey + 8],
                fill=color
            )

        sr, sc = move[0]
        sx = sc * CELL + CELL // 2
        sy = sr * CELL + CELL // 2
        draw.ellipse(
            [sx - 12, sy - 12, sx + 12, sy + 12],
            outline=(0, 220, 0), width=4
        )


# ══════════════════════════════════════════
# واجهة المساعد الذكي
# ══════════════════════════════════════════

def main():
    st.set_page_config(
        "♟️ مساعد الداما الشرس", "♟️", layout="wide"
    )

    st.markdown("""<style>
    .block-container {max-width:1100px}
    .best-box {
        background:linear-gradient(135deg,#28a745,#20c997);
        color:#fff; padding:18px; border-radius:12px;
        text-align:center; margin:10px 0;
    }
    .best-box h2 {margin:0; font-size:1.6em}
    .best-box p {margin:5px 0; font-size:1.1em}
    .warn-box {
        background:#f8d7da; border:2px solid #dc3545;
        color:#721c24; padding:12px; border-radius:10px;
        text-align:center; margin:8px 0; font-weight:bold;
    }
    .eval-bar {
        background:#e9ecef; border-radius:8px;
        overflow:hidden; height:30px; margin:8px 0;
    }
    .eval-fill {
        height:100%; text-align:center; color:#fff;
        font-weight:bold; line-height:30px;
        border-radius:8px;
    }
    </style>""", unsafe_allow_html=True)

    st.title("♟️ مساعد الداما الشرس")
    st.caption(
        "حمّل صورة أو أدخل الرقعة يدوياً → "
        "المساعد يحلل ويريك كيف تفوز"
    )

    # حالة
    if "board" not in st.session_state:
        st.session_state.board = Engine._init().tolist()

    # ═══ الشريط الجانبي ═══
    with st.sidebar:
        st.header("⚙️ الإعدادات")

        my_color = st.radio(
            "♟️ لون قطعك:",
            ["⚪ الفاتح", "⚫ الداكن"]
        )
        player = P.L if "الفاتح" in my_color else P.D

        depth = st.select_slider(
            "🧠 عمق التحليل:",
            ["سريع (3)", "متوسط (5)", "عميق (7)",
             "شرس (9)", "وحشي (11)"],
            value="متوسط (5)"
        )
        depth_val = int(depth.split("(")[1].split(")")[0])

        st.divider()

        if st.button(
            "🔄 رقعة ابتدائية",
            use_container_width=True
        ):
            st.session_state.board = Engine._init().tolist()
            st.rerun()

        if st.button(
            "🗑️ مسح الرقعة",
            use_container_width=True
        ):
            st.session_state.board = np.zeros(
                (8, 8), dtype=int
            ).tolist()
            st.rerun()

        st.divider()
        board_arr = np.array(st.session_state.board)
        eng_info = Engine(board_arr)
        ln, lk = eng_info.count(P.L)
        dn, dk = eng_info.count(P.D)

        st.markdown("### 📊 القطع")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("⚪ فاتح", ln + lk,
                       delta=f"👑{lk}" if lk else None)
        with c2:
            st.metric("⚫ داكن", dn + dk,
                       delta=f"👑{dk}" if dk else None)

    # ═══ المحتوى ═══
    input_tab, analyze_tab = st.tabs(
        ["📥 إدخال الرقعة", "🧠 التحليل"]
    )

    # ─── تبويب الإدخال ───
    with input_tab:
        method = st.radio(
            "طريقة الإدخال:",
            ["✏️ يدوي", "📷 صورة"],
            horizontal=True
        )

        if method == "✏️ يدوي":
            st.markdown(
                "**اختر نوع القطعة ثم اضغط على المربع:**"
            )

            piece_opts = {
                "⬜ فارغ": int(P.E),
                "⚪ فاتح": int(P.L),
                "⚫ داكن": int(P.D),
                "👑 ملك فاتح": int(P.LK),
                "♛ ملك داكن": int(P.DK),
            }
            selected_piece = st.radio(
                "القطعة:", list(piece_opts.keys()),
                horizontal=True,
                label_visibility="collapsed"
            )
            sel_val = piece_opts[selected_piece]

            symbols = {
                int(P.E): "·",
                int(P.L): "⚪",
                int(P.D): "⚫",
                int(P.LK): "👑",
                int(P.DK): "♛",
            }

            board_arr = np.array(st.session_state.board)

            for r in range(8):
                cols = st.columns(8)
                for c in range(8):
                    with cols[c]:
                        playable = (r + c) % 2 != 0
                        p = int(board_arr[r][c])
                        sym = symbols.get(p, "·")
                        if not playable:
                            sym = ""

                        if st.button(
                            sym, key=f"m{r}{c}",
                            use_container_width=True,
                            disabled=not playable
                        ):
                            st.session_state.board[r][c] = \
                                sel_val
                            st.rerun()

            # عرض بصري
            vis = Render.draw_board(board_arr)
            st.image(vis, caption="الرقعة الحالية",
                     use_container_width=True)

        elif method == "📷 صورة" and HAS_CV2:
            st.markdown("""
            **📸 ارفع صورة رقعة الداما:**
            - يُفضّل صورة من الأعلى مباشرة
            - يدعم صور مائلة (تصحيح منظور تلقائي)
            - يكتشف القطع بـ HSV + HoughCircles
            """)

            uploaded = st.file_uploader(
                "ارفع الصورة",
                type=["jpg", "png", "jpeg"]
            )

            if uploaded:
                pil = Image.open(uploaded).convert("RGB")
                img_cv = cv2.cvtColor(
                    np.array(pil), cv2.COLOR_RGB2BGR
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.image(pil, caption="📸 الأصلية",
                             use_container_width=True)

                # تصحيح المنظور
                with st.spinner("🔲 تصحيح المنظور..."):
                    fixed, was_fixed = \
                        Vision.fix_perspective(img_cv)

                fixed_pil = Image.fromarray(
                    cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)
                )
                with c2:
                    cap = ("✅ تم التصحيح" if was_fixed
                           else "📐 اقتصاص")
                    st.image(fixed_pil, caption=cap,
                             use_container_width=True)

                st.markdown("**⚙️ ضبط العتبات:**")
                tc1, tc2 = st.columns(2)
                with tc1:
                    lt = st.slider(
                        "عتبة الفاتح (سطوع)",
                        100, 230, 160
                    )
                with tc2:
                    dt = st.slider(
                        "عتبة الداكن (سطوع)",
                        30, 150, 100
                    )

                if st.button(
                    "🔍 تحليل الصورة", type="primary"
                ):
                    with st.spinner("🧠 تحليل متقدم..."):
                        # HSV
                        hsv_b, hsv_info = \
                            Vision.detect_by_hsv(
                                fixed, lt, dt
                            )
                        # دوائر
                        circle_b, circle_vis = \
                            Vision.detect_by_circles(
                                fixed, lt, dt
                            )
                        # دمج
                        merged = Vision.merge(
                            hsv_b, circle_b
                        )

                    st.success("✅ اكتمل التحليل!")

                    r1, r2, r3 = st.tabs(
                        ["🎨 HSV", "⭕ دوائر", "🔀 مدمج"]
                    )
                    with r1:
                        v1 = Render.draw_board(hsv_b)
                        st.image(v1,
                                 caption="نتيجة HSV",
                                 use_container_width=True)
                        with st.expander("🔬 تفاصيل"):
                            for d in hsv_info:
                                if d["det"] != 0:
                                    nm = ("⚪" if d["det"] == 1
                                          else "⚫")
                                    st.text(
                                        f"({d['r']},{d['c']})"
                                        f" {nm}"
                                        f" سطوع={d['bright']}"
                                        f" تشبع={d['sat']}"
                                        f" تباين={d['var']}"
                                    )

                    with r2:
                        st.image(
                            circle_vis,
                            caption="الدوائر المكتشفة",
                            use_container_width=True
                        )
                        v2 = Render.draw_board(circle_b)
                        st.image(
                            v2, caption="نتيجة الدوائر",
                            use_container_width=True
                        )

                    with r3:
                        v3 = Render.draw_board(merged)
                        st.image(
                            v3, caption="النتيجة المدمجة",
                            use_container_width=True
                        )

                        eng_m = Engine(merged)
                        mln, mlk = eng_m.count(P.L)
                        mdn, mdk = eng_m.count(P.D)
                        st.info(
                            f"⚪ {mln} قطع + {mlk} ملوك  •  "
                            f"⚫ {mdn} قطع + {mdk} ملوك"
                        )

                        if st.button(
                            "📥 استخدم هذه الرقعة للتحليل",
                            type="primary"
                        ):
                            st.session_state.board = \
                                merged.tolist()
                            st.success("✅ تم!")
                            st.rerun()

        elif method == "📷 صورة" and not HAS_CV2:
            st.error(
                "❌ مكتبة OpenCV غير مثبتة. "
                "أضف opencv-python-headless "
                "في requirements.txt"
            )

    # ─── تبويب التحليل ───
    with analyze_tab:
        board_arr = np.array(
            st.session_state.board, dtype=np.int8
        )
        eng = Engine(board_arr)

        # عرض الرقعة الحالية
        st.image(
            Render.draw_board(board_arr),
            caption="الرقعة الحالية",
            use_container_width=True
        )

        ln, lk = eng.count(P.L)
        dn, dk = eng.count(P.D)

        if (ln + lk) == 0 and (dn + dk) == 0:
            st.warning("⚠️ الرقعة فارغة! أدخل القطع أولاً")
            return

        go = eng.game_over()
        if go is not None:
            if go == -1:
                st.info("🤝 الوضعية منتهية بتعادل")
            elif go == P.L:
                st.success("🏆 الفاتح فائز!")
            else:
                st.success("🏆 الداكن فائز!")
            return

        emoji = "⚪" if player == P.L else "⚫"
        st.markdown(f"### {emoji} تحليل حركات: **{'الفاتح' if player == P.L else 'الداكن'}**")

        if st.button(
            "🧠 حلّل الآن!", type="primary",
            use_container_width=True
        ):
            with st.spinner(
                f"🧠 جاري التحليل بعمق {depth_val}..."
            ):
                ai = BrutalAI(depth=depth_val)
                analysis = ai.analyze_all_moves(eng, player)

            if not analysis or not analysis["moves"]:
                st.error("❌ لا توجد حركات متاحة!")
                return

            moves = analysis["moves"]
            best = moves[0]

            # ── تقييم الوضعية ──
            pos_eval = analysis["position_eval"]
            if pos_eval > 200:
                eval_msg = "🟢 أنت متفوق بوضوح!"
                eval_color = "#28a745"
            elif pos_eval > 50:
                eval_msg = "🟢 أنت أفضل قليلاً"
                eval_color = "#20c997"
            elif pos_eval > -50:
                eval_msg = "🟡 الوضعية متكافئة"
                eval_color = "#ffc107"
            elif pos_eval > -200:
                eval_msg = "🟠 الخصم أفضل قليلاً"
                eval_color = "#fd7e14"
            else:
                eval_msg = "🔴 أنت في خطر!"
                eval_color = "#dc3545"

            pct = max(5, min(95,
                int(50 + pos_eval / 20)))
            st.markdown(f"""
            <div class="eval-bar">
                <div class="eval-fill"
                     style="width:{pct}%;
                     background:{eval_color}">
                    {eval_msg} ({pos_eval})
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── الأكل الإجباري ──
            if analysis["is_forced_capture"]:
                st.markdown(
                    '<div class="warn-box">'
                    '⚡ أكل إجباري! يجب عليك الأكل'
                    '</div>',
                    unsafe_allow_html=True
                )

            # ── أفضل حركة ──
            path = " → ".join(
                f"({p[0]},{p[1]})" for p in best["move"]
            )
            extras = []
            if best["is_capture"]:
                extras.append(
                    f"💥 تأكل {best['captured']} قطعة"
                )
            if best["promotes"]:
                extras.append("👑 ترقية للملك!")
            extra_text = " • ".join(extras) if extras else ""

            st.markdown(f"""
            <div class="best-box">
                <h2>🏆 أفضل حركة</h2>
                <p style="font-size:1.5em">{path}</p>
                <p>{extra_text}</p>
                <p>تقييم: {best['score']} •
                   فرصة الفوز: {best['win_pct']}%</p>
            </div>
            """, unsafe_allow_html=True)

            # ── صورة أفضل حركة ──
            best_img = Render.draw_board(
                board_arr,
                arrows=[{
                    "move": best["move"],
                    "color": (50, 205, 50),
                    "width": 6
                }],
                highlight=set(best["move"]),
                label="BEST MOVE"
            )
            st.image(
                best_img,
                caption="🏆 أفضل حركة",
                use_container_width=True
            )

            # ── نتيجة بعد الحركة ──
            after = eng.copy()
            after.do_move(best["move"])
            after_img = Render.draw_board(after.board)
            st.image(
                after_img,
                caption="📋 الرقعة بعد تنفيذ الحركة",
                use_container_width=True
            )

            if st.button(
                "✅ طبّق هذه الحركة على الرقعة",
                use_container_width=True
            ):
                st.session_state.board = after.board.tolist()
                st.rerun()

            # ── تصنيف كل الحركات ──
            st.markdown("---")
            st.markdown(
                f"### 📊 تصنيف كل الحركات "
                f"({len(moves)} حركة)"
            )

            # ألوان التصنيف
            rank_colors = {
                1: ("🥇", (50, 205, 50)),
                2: ("🥈", (65, 105, 225)),
                3: ("🥉", (255, 165, 0)),
            }

            all_arrows = []
            for i, mv in enumerate(moves):
                icon, color = rank_colors.get(
                    mv["rank"],
                    (f"#{mv['rank']}", (180, 180, 180))
                )
                all_arrows.append({
                    "move": mv["move"],
                    "color": color,
                    "width": 6 if i == 0 else 3
                })

            all_img = Render.draw_board(
                board_arr, arrows=all_arrows[:5]
            )
            st.image(
                all_img,
                caption=(
                    "🥇 أخضر = الأفضل  "
                    "🥈 أزرق = ثاني  "
                    "🥉 برتقالي = ثالث"
                ),
                use_container_width=True
            )

            for mv in moves:
                icon = rank_colors.get(
                    mv["rank"], (f"#{mv['rank']}", None)
                )[0]
                path = " → ".join(
                    f"({p[0]},{p[1]})" for p in mv["move"]
                )

                tags = []
                if mv["is_capture"]:
                    tags.append(
                        f"💥 أكل ×{mv['captured']}"
                    )
                if mv["promotes"]:
                    tags.append("👑 ترقية")

                score_bar = "█" * max(
                    1, int(mv["win_pct"] / 5)
                )

                with st.expander(
                    f"{icon} {path}  •  "
                    f"تقييم: {mv['score']}  •  "
                    f"فوز: {mv['win_pct']}%"
                ):
                    st.markdown(
                        f"**المسار:** `{path}`"
                    )
                    if tags:
                        st.markdown(
                            f"**ملاحظات:** {' • '.join(tags)}"
                        )
                    st.markdown(
                        f"**فرصة الفوز:** "
                        f"`{score_bar}` {mv['win_pct']}%"
                    )

                    mv_img = Render.draw_board(
                        board_arr,
                        arrows=[{
                            "move": mv["move"],
                            "color": (255, 100, 50),
                            "width": 5
                        }],
                        highlight=set(mv["move"])
                    )
                    st.image(
                        mv_img,
                        use_container_width=True
                    )

            # ── إحصائيات التحليل ──
            st.markdown("---")
            st.markdown("### ⚙️ إحصائيات التحليل")
            s1, s2, s3 = st.columns(3)
            with s1:
                st.metric(
                    "🔢 العقد المحسوبة",
                    f"{analysis['total_nodes']:,}"
                )
            with s2:
                st.metric(
                    "⏱ الوقت",
                    f"{analysis['time']}s"
                )
            with s3:
                st.metric("📏 العمق", depth_val)

    # Footer
    st.divider()
    st.markdown(
        '<p style="text-align:center;color:#999;'
        'font-size:0.8em">'
        '♟️ مساعد الداما الشرس v5.0 — '
        'Minimax + Alpha-Beta + OpenCV — '
        'يعمل بدون إنترنت</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
