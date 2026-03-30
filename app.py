#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════╗
║      محرك الداما المستقل - النسخة النهائية v2.0       ║
║      Offline Checkers AI Engine - Final Edition       ║
║                                                       ║
║  ✅ محرك كامل: أكل، ملوك، أكل متعدد، أكل إجباري     ║
║  ✅ Minimax + Alpha-Beta Pruning                      ║
║  ✅ رؤية حاسوبية بـ HSV + تباين                       ║
║  ✅ رسم بصري للرقعة + أسهم الحركات                    ║
║  ✅ محرر يدوي تفاعلي                                  ║
║  ✅ 4 مستويات صعوبة                                   ║
╚═══════════════════════════════════════════════════════╝
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import time
from enum import IntEnum
from typing import List, Tuple, Optional, Dict

# ══════════════════════════════════════════════════════
# 1. الثوابت والتعريفات
# ══════════════════════════════════════════════════════

class Piece(IntEnum):
    """أنواع القطع على الرقعة"""
    EMPTY      = 0
    LIGHT      = 1   # قطعة فاتحة عادية
    DARK       = 2   # قطعة داكنة عادية
    LIGHT_KING = 3   # ملك فاتح (ضامة)
    DARK_KING  = 4   # ملك داكن (ضامة)


BOARD_SIZE = 8

# ── ألوان الرسم ──
COLORS = {
    "light_sq":    (240, 217, 181),
    "dark_sq":     (181, 136, 99),
    "light_piece": (255, 253, 245),
    "dark_piece":  (60,  60,  60),
    "light_edge":  (210, 200, 185),
    "dark_edge":   (30,  30,  30),
    "king_mark":   (255, 215,   0),
    "king_edge":   (200, 170,   0),
    "highlight":   (80,  220,  80),
    "arrow":       (255, 255,   0),
    "arrow_start": (0,   200, 100),
    "coord":       (140, 120, 100),
    "shadow":      (80,  60,  40),
}

# ── ثوابت الرسم ──
CELL_SIZE    = 72
PIECE_RADIUS = 28
KING_RADIUS  = 11


# ══════════════════════════════════════════════════════
# 2. محرك اللعبة الكامل (Full Checkers Engine)
# ══════════════════════════════════════════════════════

class CheckersEngine:
    """
    محرك داما كامل القواعد:
      • حركات بسيطة قُطرية
      • أكل (قفز) فوق قطع الخصم
      • أكل متعدد متسلسل
      • أكل إجباري (إذا يمكنك الأكل يجب عليك)
      • ترقية للملك عند الوصول للحافة
      • الملك يتحرك في الاتجاهات الأربعة
    """

    def __init__(self, board=None):
        if board is not None:
            self.board = np.array(board, dtype=np.int8)
        else:
            self.board = self._create_initial_board()

    # ──────────────────────────────────
    # إنشاء الرقعة الابتدائية
    # ──────────────────────────────────
    @staticmethod
    def _create_initial_board() -> np.ndarray:
        """رقعة داما كلاسيكية 8×8"""
        board = np.zeros((8, 8), dtype=np.int8)
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 != 0:       # المربعات الداكنة فقط
                    if r < 3:
                        board[r][c] = Piece.DARK
                    elif r > 4:
                        board[r][c] = Piece.LIGHT
        return board

    # ──────────────────────────────────
    # نسخة سريعة (بدون deepcopy)
    # ──────────────────────────────────
    def copy(self) -> "CheckersEngine":
        new = CheckersEngine.__new__(CheckersEngine)
        new.board = self.board.copy()   # numpy copy أسرع 50x من deepcopy
        return new

    # ──────────────────────────────────
    # دوال مساعدة للقطع
    # ──────────────────────────────────
    @staticmethod
    def is_light(piece: int) -> bool:
        return piece in (Piece.LIGHT, Piece.LIGHT_KING)

    @staticmethod
    def is_dark(piece: int) -> bool:
        return piece in (Piece.DARK, Piece.DARK_KING)

    @staticmethod
    def is_king(piece: int) -> bool:
        return piece in (Piece.LIGHT_KING, Piece.DARK_KING)

    @staticmethod
    def same_team(piece: int, player: int) -> bool:
        """هل القطعة من نفس فريق اللاعب؟"""
        if player in (Piece.LIGHT, Piece.LIGHT_KING):
            return piece in (Piece.LIGHT, Piece.LIGHT_KING)
        return piece in (Piece.DARK, Piece.DARK_KING)

    @staticmethod
    def enemy_of(piece: int, player: int) -> bool:
        """هل القطعة من فريق الخصم؟"""
        if piece == Piece.EMPTY:
            return False
        if player in (Piece.LIGHT, Piece.LIGHT_KING):
            return piece in (Piece.DARK, Piece.DARK_KING)
        return piece in (Piece.LIGHT, Piece.LIGHT_KING)

    @staticmethod
    def opponent(player: int) -> int:
        """إرجاع الخصم"""
        if player in (Piece.LIGHT, Piece.LIGHT_KING):
            return Piece.DARK
        return Piece.LIGHT

    # ──────────────────────────────────
    # اتجاهات الحركة
    # ──────────────────────────────────
    @staticmethod
    def get_directions(piece: int) -> List[Tuple[int, int]]:
        """الاتجاهات القُطرية المسموحة لكل نوع قطعة"""
        if piece == Piece.LIGHT:
            return [(-1, -1), (-1, 1)]         # لأعلى فقط
        elif piece == Piece.DARK:
            return [(1, -1), (1, 1)]           # لأسفل فقط
        elif piece in (Piece.LIGHT_KING, Piece.DARK_KING):
            return [(-1, -1), (-1, 1),
                    (1, -1),  (1, 1)]          # كل الاتجاهات
        return []

    # ──────────────────────────────────
    # الحركات البسيطة (بدون أكل)
    # ──────────────────────────────────
    def _get_simple_moves(self, r: int, c: int) -> list:
        piece = self.board[r][c]
        if piece == Piece.EMPTY:
            return []

        moves = []
        for dr, dc in self.get_directions(piece):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if self.board[nr][nc] == Piece.EMPTY:
                    moves.append(((r, c), (nr, nc)))
        return moves

    # ──────────────────────────────────
    # حركات الأكل (مع أكل متعدد)
    # ──────────────────────────────────
    def _get_jump_chains(
        self, r: int, c: int,
        board: np.ndarray = None,
        eaten: frozenset = None
    ) -> list:
        """
        إيجاد كل سلاسل الأكل الممكنة من موقع (r, c).
        يدعم الأكل المتعدد المتسلسل.
        يُرجع قائمة من المسارات، كل مسار = tuple من المواقع.
        """
        if board is None:
            board = self.board
        if eaten is None:
            eaten = frozenset()

        piece = board[r][c]
        if piece == Piece.EMPTY:
            return []

        chains = []

        for dr, dc in self.get_directions(piece):
            # موقع القطعة المُراد أكلها
            mr, mc = r + dr, c + dc
            # موقع الهبوط بعد القفز
            nr, nc = r + 2 * dr, c + 2 * dc

            if not (0 <= nr < 8 and 0 <= nc < 8):
                continue
            if board[nr][nc] != Piece.EMPTY:
                continue
            if not self.enemy_of(board[mr][mc], piece):
                continue
            if (mr, mc) in eaten:
                continue

            # ─ تنفيذ الأكل مؤقتاً ─
            new_board = board.copy()
            new_board[nr][nc] = piece
            new_board[r][c]   = Piece.EMPTY
            new_board[mr][mc] = Piece.EMPTY

            # ترقية إذا وصل الحافة
            promoted = False
            if nr == 0 and piece == Piece.LIGHT:
                new_board[nr][nc] = Piece.LIGHT_KING
                promoted = True
            elif nr == 7 and piece == Piece.DARK:
                new_board[nr][nc] = Piece.DARK_KING
                promoted = True

            new_eaten = eaten | {(mr, mc)}

            # ─ البحث عن أكل إضافي ─
            # (في بعض القوانين: الترقية توقف الأكل المتعدد)
            if not promoted:
                further = self._get_jump_chains(
                    nr, nc, new_board, new_eaten
                )
            else:
                further = []

            if further:
                for chain in further:
                    # ربط المسار: (البداية, القفزة الأولى, ...باقي السلسلة)
                    chains.append(
                        ((r, c),) + chain
                    )
            else:
                chains.append(((r, c), (nr, nc)))

        return chains

    # ──────────────────────────────────
    # جميع الحركات المتاحة
    # ──────────────────────────────────
    def get_all_moves(self, player: int) -> Tuple[list, bool]:
        """
        كل الحركات القانونية للاعب.
        قاعدة الأكل الإجباري: إذا يمكنك الأكل → يجب عليك.
        يُرجع: (قائمة_الحركات, هل_هي_أكل)
        """
        all_jumps  = []
        all_simple = []

        for r in range(8):
            for c in range(8):
                if self.same_team(self.board[r][c], player):
                    # حركات الأكل
                    jumps = self._get_jump_chains(r, c)
                    all_jumps.extend(jumps)
                    # حركات بسيطة
                    simple = self._get_simple_moves(r, c)
                    all_simple.extend(simple)

        # ── الأكل الإجباري ──
        if all_jumps:
            # إرجاع أطول سلاسل الأكل فقط (اختياري)
            max_len = max(len(j) for j in all_jumps)
            longest = [j for j in all_jumps if len(j) == max_len]
            return longest, True

        return all_simple, False

    # ──────────────────────────────────
    # تنفيذ حركة
    # ──────────────────────────────────
    def execute_move(self, move: tuple):
        """تنفيذ حركة (بسيطة أو سلسلة أكل) على الرقعة"""
        if len(move) < 2:
            return

        piece = self.board[move[0][0]][move[0][1]]
        self.board[move[0][0]][move[0][1]] = Piece.EMPTY

        for i in range(len(move) - 1):
            sr, sc = move[i]
            er, ec = move[i + 1]

            dr = er - sr
            dc = ec - sc

            # إذا كانت قفزة (المسافة = 2)
            if abs(dr) == 2 and abs(dc) == 2:
                mr = sr + dr // 2
                mc = sc + dc // 2
                self.board[mr][mc] = Piece.EMPTY

        # وضع القطعة في الموقع النهائي
        fr, fc = move[-1]
        self.board[fr][fc] = piece

        # ── ترقية للملك ──
        if fr == 0 and piece == Piece.LIGHT:
            self.board[fr][fc] = Piece.LIGHT_KING
        elif fr == 7 and piece == Piece.DARK:
            self.board[fr][fc] = Piece.DARK_KING

    # ──────────────────────────────────
    # إحصائيات
    # ──────────────────────────────────
    def count_pieces(self, player: int) -> dict:
        normal = 0
        kings  = 0
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if self.same_team(p, player):
                    if self.is_king(p):
                        kings += 1
                    else:
                        normal += 1
        return {"normal": normal, "kings": kings, "total": normal + kings}

    def is_game_over(self) -> Optional[int]:
        """
        None = اللعبة مستمرة
        Piece.LIGHT = فاز الفاتح
        Piece.DARK  = فاز الداكن
        -1 = تعادل (لا حركات لأحد)
        """
        light = self.count_pieces(Piece.LIGHT)["total"]
        dark  = self.count_pieces(Piece.DARK)["total"]

        if light == 0:
            return Piece.DARK
        if dark == 0:
            return Piece.LIGHT

        light_moves, _ = self.get_all_moves(Piece.LIGHT)
        dark_moves,  _ = self.get_all_moves(Piece.DARK)

        if not light_moves and not dark_moves:
            return -1
        if not light_moves:
            return Piece.DARK
        if not dark_moves:
            return Piece.LIGHT

        return None


# ══════════════════════════════════════════════════════
# 3. الذكاء الاصطناعي (Minimax + Alpha-Beta)
# ══════════════════════════════════════════════════════

class CheckersAI:
    """
    محرك ذكاء اصطناعي:
      • Minimax مع Alpha-Beta Pruning
      • دالة تقييم متعددة العوامل
      • إحصائيات الأداء
    """

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.nodes_evaluated = 0

    # ──────────────────────────────────
    # دالة التقييم المتقدمة
    # ──────────────────────────────────
    def evaluate(self, engine: CheckersEngine, player: int) -> float:
        """
        تقييم متعدد العوامل:
          1. عدد القطع (×10) والملوك (×25)
          2. مكافأة التقدم نحو الترقية
          3. مكافأة السيطرة على المركز
          4. مكافأة حماية الحواف
          5. مكافأة حرية الحركة
          6. عقوبة القطع المعزولة
        """
        score = 0.0
        opp = engine.opponent(player)

        for r in range(8):
            for c in range(8):
                p = engine.board[r][c]
                if p == Piece.EMPTY:
                    continue

                val = 0.0

                # ── (1) قيمة القطعة ──
                if engine.is_king(p):
                    val = 25.0
                else:
                    val = 10.0

                # ── (2) مكافأة التقدم ──
                if p == Piece.LIGHT:
                    val += (7 - r) * 0.7    # كلما اقتربت من السطر 0
                elif p == Piece.DARK:
                    val += r * 0.7          # كلما اقتربت من السطر 7

                # ── (3) مكافأة المركز ──
                center_dist = abs(r - 3.5) + abs(c - 3.5)
                val += (5.0 - center_dist) * 0.3

                # ── (4) مكافأة الحواف ──
                if c == 0 or c == 7:
                    val += 0.8
                if r == 0 or r == 7:
                    val += 0.5

                # ── (5) عقوبة العزلة ──
                has_ally = False
                for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    ar, ac = r + dr, c + dc
                    if 0 <= ar < 8 and 0 <= ac < 8:
                        if engine.same_team(engine.board[ar][ac], p):
                            has_ally = True
                            break
                if not has_ally:
                    val -= 1.0

                # ── جمع النتيجة ──
                if engine.same_team(p, player):
                    score += val
                else:
                    score -= val

        # ── (6) مكافأة حرية الحركة ──
        my_moves,  _ = engine.get_all_moves(player)
        opp_moves, _ = engine.get_all_moves(opp)
        score += len(my_moves)  * 0.4
        score -= len(opp_moves) * 0.4

        return score

    # ──────────────────────────────────
    # خوارزمية Minimax + Alpha-Beta
    # ──────────────────────────────────
    def _minimax(
        self,
        engine: CheckersEngine,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        current_player: int,
        original_player: int
    ) -> Tuple[float, Optional[tuple]]:

        self.nodes_evaluated += 1

        # ── حالة الإنهاء ──
        game_result = engine.is_game_over()
        if game_result is not None:
            if game_result == original_player:
                return 9999.0, None     # فوز
            elif game_result == -1:
                return 0.0, None        # تعادل
            else:
                return -9999.0, None    # خسارة

        if depth == 0:
            return self.evaluate(engine, original_player), None

        moves, _ = engine.get_all_moves(current_player)
        if not moves:
            return self.evaluate(engine, original_player), None

        best_move = moves[0]
        next_player = engine.opponent(current_player)

        if maximizing:
            max_eval = float("-inf")
            for move in moves:
                child = engine.copy()
                child.execute_move(move)

                val, _ = self._minimax(
                    child, depth - 1, alpha, beta,
                    False, next_player, original_player
                )

                if val > max_eval:
                    max_eval = val
                    best_move = move

                alpha = max(alpha, val)
                if beta <= alpha:
                    break               # ← قطع Beta ✂️

            return max_eval, best_move

        else:
            min_eval = float("inf")
            for move in moves:
                child = engine.copy()
                child.execute_move(move)

                val, _ = self._minimax(
                    child, depth - 1, alpha, beta,
                    True, next_player, original_player
                )

                if val < min_eval:
                    min_eval = val
                    best_move = move

                beta = min(beta, val)
                if beta <= alpha:
                    break               # ← قطع Alpha ✂️

            return min_eval, best_move

    # ──────────────────────────────────
    # الواجهة العامة
    # ──────────────────────────────────
    def find_best_move(
        self, engine: CheckersEngine, player: int
    ) -> Dict:
        """حساب أفضل حركة مع إحصائيات الأداء"""
        self.nodes_evaluated = 0
        t0 = time.time()

        score, move = self._minimax(
            engine, self.max_depth,
            float("-inf"), float("inf"),
            True, player, player
        )

        elapsed = time.time() - t0

        return {
            "move":  move,
            "score": round(score, 2),
            "nodes": self.nodes_evaluated,
            "time":  round(elapsed, 3),
            "depth": self.max_depth,
        }


# ══════════════════════════════════════════════════════
# 4. الرؤية الحاسوبية المحسّنة (OpenCV)
# ══════════════════════════════════════════════════════

class BoardVision:
    """تحليل صورة الرقعة بـ HSV + Variance"""

    @staticmethod
    def auto_crop(image_pil: Image.Image) -> Image.Image:
        """محاولة اكتشاف حدود الرقعة تلقائياً"""
        img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            img_area = img.shape[0] * img.shape[1]

            if area > img_area * 0.2:
                x, y, w, h = cv2.boundingRect(largest)
                ratio = w / h if h > 0 else 0
                if 0.6 < ratio < 1.4:
                    cropped = img[y:y+h, x:x+w]
                    return Image.fromarray(
                        cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    )

        return image_pil

    @staticmethod
    def analyze(
        image_pil: Image.Image,
        light_thresh: int = 160,
        dark_thresh: int = 100
    ) -> Tuple[np.ndarray, list]:
        """
        تحليل الصورة → مصفوفة 8×8
        يستخدم HSV color space + variance للدقة.
        """
        img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (400, 400))

        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cell_h = 50   # 400 / 8
        cell_w = 50

        board = np.zeros((8, 8), dtype=np.int8)
        debug = []

        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 0:
                    continue  # المربعات الفاتحة → تخطي

                # عينة من وسط المربع (50% المركزية)
                margin = cell_h // 4
                y1 = r * cell_h + margin
                y2 = (r + 1) * cell_h - margin
                x1 = c * cell_w + margin
                x2 = (c + 1) * cell_w - margin

                roi_gray = gray[y1:y2, x1:x2]
                roi_hsv  = hsv[y1:y2, x1:x2]

                brightness = float(np.mean(roi_gray))
                saturation = float(np.mean(roi_hsv[:, :, 1]))
                variance   = float(np.var(roi_gray))

                # ── منطق التمييز ──
                detected = Piece.EMPTY

                # وجود قطعة = تباين عالي أو تشبّع ملحوظ
                has_piece = (variance > 150) or (saturation > 35)

                if has_piece:
                    if brightness > light_thresh:
                        detected = Piece.LIGHT
                    elif brightness < dark_thresh:
                        detected = Piece.DARK

                board[r][c] = detected
                debug.append({
                    "row": r, "col": c,
                    "brightness": round(brightness, 1),
                    "saturation": round(saturation, 1),
                    "variance":   round(variance, 1),
                    "detected":   int(detected),
                })

        return board, debug


# ══════════════════════════════════════════════════════
# 5. رسم الرقعة البصري (Board Renderer)
# ══════════════════════════════════════════════════════

class BoardRenderer:
    """رسم رقعة احترافية بـ Pillow"""

    @staticmethod
    def render(
        board: np.ndarray,
        highlight: list = None
    ) -> Image.Image:
        """رسم الرقعة كصورة"""
        size = CELL_SIZE * 8
        img  = Image.new("RGB", (size, size))
        draw = ImageDraw.Draw(img)

        # ── المربعات ──
        for r in range(8):
            for c in range(8):
                x1 = c * CELL_SIZE
                y1 = r * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE

                if (r + c) % 2 == 0:
                    sq_color = COLORS["light_sq"]
                else:
                    sq_color = COLORS["dark_sq"]

                draw.rectangle([x1, y1, x2, y2], fill=sq_color)

        # ── تمييز مربعات الحركة ──
        if highlight:
            for pos in highlight:
                r, c = pos
                x1 = c * CELL_SIZE + 2
                y1 = r * CELL_SIZE + 2
                x2 = x1 + CELL_SIZE - 4
                y2 = y1 + CELL_SIZE - 4
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline=COLORS["highlight"], width=3
                )

        # ── القطع ──
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece == Piece.EMPTY:
                    continue

                cx = c * CELL_SIZE + CELL_SIZE // 2
                cy = r * CELL_SIZE + CELL_SIZE // 2
                pr = PIECE_RADIUS

                # ظل
                draw.ellipse(
                    [cx - pr + 3, cy - pr + 3,
                     cx + pr + 3, cy + pr + 3],
                    fill=COLORS["shadow"]
                )

                # لون القطعة
                if CheckersEngine.is_light(piece):
                    fill = COLORS["light_piece"]
                    edge = COLORS["light_edge"]
                else:
                    fill = COLORS["dark_piece"]
                    edge = COLORS["dark_edge"]

                draw.ellipse(
                    [cx - pr, cy - pr, cx + pr, cy + pr],
                    fill=fill, outline=edge, width=2
                )

                # حلقة داخلية للجمال
                draw.ellipse(
                    [cx - pr + 6, cy - pr + 6,
                     cx + pr - 6, cy + pr - 6],
                    outline=edge, width=1
                )

                # علامة الملك 👑
                if CheckersEngine.is_king(piece):
                    kr = KING_RADIUS
                    draw.ellipse(
                        [cx - kr, cy - kr, cx + kr, cy + kr],
                        fill=COLORS["king_mark"],
                        outline=COLORS["king_edge"], width=2
                    )
                    try:
                        draw.text(
                            (cx - 4, cy - 6), "K",
                            fill=(0, 0, 0)
                        )
                    except Exception:
                        pass

        # ── إحداثيات ──
        for i in range(8):
            try:
                draw.text(
                    (3, i * CELL_SIZE + 3),
                    str(i), fill=COLORS["coord"]
                )
                draw.text(
                    (i * CELL_SIZE + CELL_SIZE // 2 - 3,
                     size - 13),
                    chr(65 + i), fill=COLORS["coord"]
                )
            except Exception:
                pass

        return img

    @staticmethod
    def draw_arrow(
        img: Image.Image, move: tuple
    ) -> Image.Image:
        """رسم سهم الحركة المقترحة"""
        if not move or len(move) < 2:
            return img

        draw = ImageDraw.Draw(img)

        for i in range(len(move) - 1):
            sr, sc = move[i]
            er, ec = move[i + 1]

            sx = sc * CELL_SIZE + CELL_SIZE // 2
            sy = sr * CELL_SIZE + CELL_SIZE // 2
            ex = ec * CELL_SIZE + CELL_SIZE // 2
            ey = er * CELL_SIZE + CELL_SIZE // 2

            # خط السهم
            draw.line(
                [(sx, sy), (ex, ey)],
                fill=COLORS["arrow"], width=5
            )
            # رأس السهم
            draw.ellipse(
                [ex - 9, ey - 9, ex + 9, ey + 9],
                fill=COLORS["arrow"]
            )

        # دائرة البداية
        sr, sc = move[0]
        sx = sc * CELL_SIZE + CELL_SIZE // 2
        sy = sr * CELL_SIZE + CELL_SIZE // 2
        draw.ellipse(
            [sx - 12, sy - 12, sx + 12, sy + 12],
            outline=COLORS["arrow_start"], width=4
        )

        return img


# ══════════════════════════════════════════════════════
# 6. واجهة Streamlit الكاملة
# ══════════════════════════════════════════════════════

def run_app():

    st.set_page_config(
        page_title="♟️ محرك الداما الذكي",
        page_icon="♟️",
        layout="wide",
    )

    # ── CSS ──
    st.markdown("""
    <style>
    .block-container { max-width: 1100px; }
    h1 { text-align: center; }
    .result-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #fff; padding: 18px; border-radius: 14px;
        text-align: center; margin: 12px 0;
    }
    .result-box h3 { margin: 0 0 6px; }
    .info-box {
        background: #f0f2f6; padding: 14px;
        border-radius: 10px; margin: 8px 0;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("♟️ محرك الداما الذكي")
    st.caption(
        "ذكاء اصطناعي محلي بالكامل — "
        "Minimax + Alpha-Beta Pruning + OpenCV"
    )

    # ── Session State ──
    if "board" not in st.session_state:
        st.session_state.board = (
            CheckersEngine._create_initial_board().tolist()
        )
    if "history" not in st.session_state:
        st.session_state.history = []

    # ═════════ الشريط الجانبي ═════════
    with st.sidebar:
        st.header("⚙️ الإعدادات")

        input_mode = st.radio(
            "🎮 وضع الإدخال:",
            ["✏️ إعداد يدوي", "📷 تحليل صورة"],
        )

        st.divider()

        player_label = st.radio(
            "♟️ أنت تلعب بـ:",
            ["⚪ الفاتح (لأعلى)", "⚫ الداكن (لأسفل)"],
        )
        my_player = (
            Piece.LIGHT if "الفاتح" in player_label
            else Piece.DARK
        )

        st.divider()

        difficulty = st.select_slider(
            "🧠 مستوى الذكاء:",
            ["مبتدئ", "متوسط", "خبير", "بطل"],
            value="متوسط",
        )
        depth_map = {
            "مبتدئ": 2, "متوسط": 4,
            "خبير": 6,  "بطل": 8
        }
        ai_depth = depth_map[difficulty]

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 رقعة جديدة", use_container_width=True):
                st.session_state.board = (
                    CheckersEngine._create_initial_board().tolist()
                )
                st.session_state.history = []
                st.rerun()
        with c2:
            if st.button("🗑️ مسح الرقعة", use_container_width=True):
                st.session_state.board = (
                    np.zeros((8, 8), dtype=int).tolist()
                )
                st.session_state.history = []
                st.rerun()

        st.divider()

        # إحصائيات
        board_arr = np.array(st.session_state.board, dtype=np.int8)
        engine_tmp = CheckersEngine(board_arr)
        light_info = engine_tmp.count_pieces(Piece.LIGHT)
        dark_info  = engine_tmp.count_pieces(Piece.DARK)

        st.markdown("### 📊 القطع على الرقعة")
        lc, dc = st.columns(2)
        with lc:
            st.metric(
                "⚪ فاتحة",
                light_info["total"],
                delta=(
                    f"+{light_info['kings']} ملك"
                    if light_info["kings"] else None
                ),
            )
        with dc:
            st.metric(
                "⚫ داكنة",
                dark_info["total"],
                delta=(
                    f"+{dark_info['kings']} ملك"
                    if dark_info["kings"] else None
                ),
            )

    # ═════════ المحتوى الرئيسي ═════════
    col_board, col_ai = st.columns([1.2, 1])

    # ─────── العمود الأيسر: الرقعة ───────
    with col_board:
        st.subheader("📋 الرقعة")

        if "صورة" in input_mode:
            # ──── وضع الصورة ────
            uploaded = st.file_uploader(
                "📸 ارفع صورة الرقعة",
                type=["jpg", "png", "jpeg"],
            )

            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                cropped = BoardVision.auto_crop(pil_img)

                ca, cb = st.columns(2)
                with ca:
                    st.image(
                        pil_img, caption="الأصلية",
                        use_container_width=True
                    )
                with cb:
                    st.image(
                        cropped, caption="بعد الاقتصاص",
                        use_container_width=True
                    )

                st.markdown("**ضبط حساسية الألوان:**")
                tc1, tc2 = st.columns(2)
                with tc1:
                    lt = st.slider(
                        "عتبة الفاتح", 120, 220, 160
                    )
                with tc2:
                    dt = st.slider(
                        "عتبة الداكن", 40, 140, 100
                    )

                if st.button("🔍 تحليل الصورة", type="primary"):
                    with st.spinner("جاري التحليل..."):
                        board, dbg = BoardVision.analyze(
                            cropped, lt, dt
                        )
                        st.session_state.board = board.tolist()
                        st.success("✅ تم تحليل الصورة!")

                        with st.expander("🔬 تفاصيل التحليل"):
                            for d in dbg:
                                if d["detected"] != 0:
                                    name = (
                                        "⚪ فاتحة"
                                        if d["detected"] == 1
                                        else "⚫ داكنة"
                                    )
                                    st.write(
                                        f"({d['row']},{d['col']}): "
                                        f"{name} — سطوع "
                                        f"{d['brightness']} — "
                                        f"تباين {d['variance']}"
                                    )
                        st.rerun()

        else:
            # ──── وضع يدوي ────
            st.markdown("**اختر قطعة ثم اضغط مربعاً:**")

            piece_options = {
                "⬜ فارغ":       int(Piece.EMPTY),
                "⚪ فاتح":       int(Piece.LIGHT),
                "⚫ داكن":       int(Piece.DARK),
                "👑⚪ ملك فاتح": int(Piece.LIGHT_KING),
                "👑⚫ ملك داكن": int(Piece.DARK_KING),
            }

            selected = st.radio(
                "القطعة:", list(piece_options.keys()),
                horizontal=True, label_visibility="collapsed",
            )
            sel_val = piece_options[selected]

            symbols = {
                int(Piece.EMPTY):      "·",
                int(Piece.LIGHT):      "⚪",
                int(Piece.DARK):       "⚫",
                int(Piece.LIGHT_KING): "👑W",
                int(Piece.DARK_KING):  "👑B",
            }

            board_arr = np.array(st.session_state.board)

            for r in range(8):
                cols = st.columns(8)
                for c in range(8):
                    with cols[c]:
                        is_playable = (r + c) % 2 != 0
                        p = int(board_arr[r][c])
                        sym = symbols.get(p, "·")
                        if not is_playable:
                            sym = ""

                        if st.button(
                            sym,
                            key=f"c_{r}_{c}",
                            use_container_width=True,
                            disabled=not is_playable,
                        ):
                            st.session_state.board[r][c] = sel_val
                            st.rerun()

        # ── رسم الرقعة البصرية ──
        rendered = BoardRenderer.render(
            np.array(st.session_state.board, dtype=np.int8)
        )
        st.image(
            rendered,
            caption="🎨 عرض الرقعة",
            use_container_width=True,
        )

    # ─────── العمود الأيمن: الذكاء ───────
    with col_ai:
        st.subheader("🤖 تحليل AI")

        if st.button(
            "🚀 احسب أفضل حركة",
            use_container_width=True,
            type="primary",
        ):
            engine = CheckersEngine(st.session_state.board)
            my_count = engine.count_pieces(my_player)["total"]

            if my_count == 0:
                st.error("❌ لا توجد قطع لك على الرقعة!")
            else:
                moves, is_jump = engine.get_all_moves(my_player)

                if not moves:
                    st.warning("⚠️ لا توجد حركات متاحة لك!")
                else:
                    with st.spinner(
                        f"🧠 يفكر بعمق {ai_depth}..."
                    ):
                        ai = CheckersAI(max_depth=ai_depth)
                        result = ai.find_best_move(engine, my_player)

                    move = result["move"]

                    if move:
                        # ── عرض النتيجة ──
                        start = move[0]
                        end   = move[-1]
                        steps = len(move) - 1

                        move_type = (
                            "أكل متعدد! 🔥"
                            if steps > 1 and is_jump
                            else "أكل ⚡"
                            if is_jump
                            else "حركة عادية"
                        )

                        st.markdown(f"""
                        <div class="result-box">
                            <h3>✅ أفضل حركة — {move_type}</h3>
                            <p style="font-size:1.4em;">
                                ({start[0]},{start[1]}) ➜
                                ({end[0]},{end[1]})
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # مقاييس
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("📊 التقييم", result["score"])
                        with m2:
                            st.metric(
                                "🔢 العقد",
                                f"{result['nodes']:,}"
                            )
                        with m3:
                            st.metric(
                                "⏱ الوقت",
                                f"{result['time']}s"
                            )

                        # تفاصيل الأكل المتعدد
                        if steps > 1:
                            st.markdown(
                                f'<div class="info-box">'
                                f"🔥 سلسلة أكل من {steps} "
                                f"خطوات!</div>",
                                unsafe_allow_html=True,
                            )
                            path = " ➜ ".join(
                                f"({p[0]},{p[1]})" for p in move
                            )
                            st.code(path)

                        if is_jump:
                            st.info("⚠️ الأكل إجباري بقوانين الداما!")

                        # ── رسم الحركة ──
                        result_img = BoardRenderer.render(
                            np.array(st.session_state.board,
                                     dtype=np.int8),
                            highlight=list(move),
                        )
                        result_img = BoardRenderer.draw_arrow(
                            result_img, move
                        )
                        st.image(
                            result_img,
                            caption="🎯 الحركة المقترحة",
                            use_container_width=True,
                        )

                        # ── زر التنفيذ ──
                        if st.button(
                            "✅ نفّذ هذه الحركة",
                            use_container_width=True,
                        ):
                            eng = CheckersEngine(
                                st.session_state.board
                            )
                            eng.execute_move(move)
                            st.session_state.board = (
                                eng.board.tolist()
                            )
                            st.session_state.history.append(
                                " ➜ ".join(
                                    f"({p[0]},{p[1]})"
                                    for p in move
                                )
                            )
                            st.rerun()

                        # ── جميع الحركات ──
                        with st.expander(
                            f"📋 كل الحركات ({len(moves)})"
                        ):
                            for i, m in enumerate(moves, 1):
                                is_best = m == move
                                icon = "⭐" if is_best else f"{i}."
                                path = " ➜ ".join(
                                    f"({p[0]},{p[1]})"
                                    for p in m
                                )
                                st.write(f"{icon} {path}")

                    else:
                        st.error("❌ لم يُعثر على حركة!")

        # ── حالة اللعبة ──
        engine_check = CheckersEngine(st.session_state.board)
        game_state = engine_check.is_game_over()

        if game_state is not None:
            if game_state == Piece.LIGHT:
                st.balloons()
                st.success("🏆 فاز الفاتح!")
            elif game_state == Piece.DARK:
                st.balloons()
                st.success("🏆 فاز الداكن!")
            else:
                st.warning("🤝 تعادل!")

        # ── سجل الحركات ──
        if st.session_state.history:
            with st.expander(
                f"📜 السجل ({len(st.session_state.history)})"
            ):
                for i, h in enumerate(
                    st.session_state.history, 1
                ):
                    player_icon = (
                        "⚪" if i % 2 != 0 else "⚫"
                    )
                    st.write(f"{i}. {player_icon} {h}")

    # ── Footer ──
    st.divider()
    st.markdown(
        '<p style="text-align:center;color:#999;'
        'font-size:0.85em;">'
        "♟️ محرك الداما المستقل v2.0 — "
        "Python + OpenCV + Minimax — "
        "يعمل بالكامل بدون إنترنت أو APIs"
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════
# 7. التشغيل
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    run_app()
