#!/usr/bin/env python3
"""
╔════════════════════════════════════════════╗
║   ♟️ لعبة الداما التفاعلية v3.0           ║
║   اضغط على القطعة → اضغط على الوجهة      ║
║   Minimax + Alpha-Beta AI                  ║
╚════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import time
from enum import IntEnum


# ══════════════════════════════════════════
# 1. ثوابت اللعبة
# ══════════════════════════════════════════

class Piece(IntEnum):
    EMPTY = 0
    LIGHT = 1
    DARK = 2
    LIGHT_KING = 3
    DARK_KING = 4


# ══════════════════════════════════════════
# 2. محرك اللعبة الكامل
# ══════════════════════════════════════════

class CheckersEngine:
    """
    محرك داما كامل القواعد:
    - حركات عادية + أكل + أكل متعدد
    - أكل إجباري + ملوك (ضامة)
    - ترقية تلقائية عند الحافة
    """

    def __init__(self, board=None):
        if board is not None:
            self.board = np.array(board, dtype=np.int8)
        else:
            self.board = self._initial_board()

    @staticmethod
    def _initial_board():
        b = np.zeros((8, 8), dtype=np.int8)
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 != 0:
                    if r < 3:
                        b[r][c] = Piece.DARK
                    elif r > 4:
                        b[r][c] = Piece.LIGHT
        return b

    def copy(self):
        e = CheckersEngine.__new__(CheckersEngine)
        e.board = self.board.copy()
        return e

    # ── تصنيف القطع ──

    @staticmethod
    def is_light(p):
        return p in (Piece.LIGHT, Piece.LIGHT_KING)

    @staticmethod
    def is_dark(p):
        return p in (Piece.DARK, Piece.DARK_KING)

    @staticmethod
    def is_king(p):
        return p in (Piece.LIGHT_KING, Piece.DARK_KING)

    @staticmethod
    def belongs_to(piece, player):
        if player in (Piece.LIGHT, Piece.LIGHT_KING):
            return piece in (Piece.LIGHT, Piece.LIGHT_KING)
        return piece in (Piece.DARK, Piece.DARK_KING)

    @staticmethod
    def is_enemy(piece, player):
        if piece == Piece.EMPTY:
            return False
        if player in (Piece.LIGHT, Piece.LIGHT_KING):
            return piece in (Piece.DARK, Piece.DARK_KING)
        return piece in (Piece.LIGHT, Piece.LIGHT_KING)

    @staticmethod
    def opponent(player):
        if player in (Piece.LIGHT, Piece.LIGHT_KING):
            return Piece.DARK
        return Piece.LIGHT

    @staticmethod
    def directions(piece):
        if piece == Piece.LIGHT:
            return [(-1, -1), (-1, 1)]
        if piece == Piece.DARK:
            return [(1, -1), (1, 1)]
        if piece in (Piece.LIGHT_KING, Piece.DARK_KING):
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return []

    # ── الحركات البسيطة ──

    def _simple_moves(self, r, c):
        p = self.board[r][c]
        moves = []
        for dr, dc in self.directions(p):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                if self.board[nr][nc] == Piece.EMPTY:
                    moves.append(((r, c), (nr, nc)))
        return moves

    # ── سلاسل الأكل (مع أكل متعدد) ──

    def _jump_chains(self, r, c, board=None, eaten=None):
        if board is None:
            board = self.board
        if eaten is None:
            eaten = frozenset()

        p = board[r][c]
        chains = []

        for dr, dc in self.directions(p):
            mr, mc = r + dr, c + dc
            nr, nc = r + 2 * dr, c + 2 * dc

            if not (0 <= nr < 8 and 0 <= nc < 8):
                continue
            if board[nr][nc] != Piece.EMPTY:
                continue
            if not self.is_enemy(board[mr][mc], p):
                continue
            if (mr, mc) in eaten:
                continue

            nb = board.copy()
            nb[nr][nc] = p
            nb[r][c] = Piece.EMPTY
            nb[mr][mc] = Piece.EMPTY

            promoted = False
            if nr == 0 and p == Piece.LIGHT:
                nb[nr][nc] = Piece.LIGHT_KING
                promoted = True
            elif nr == 7 and p == Piece.DARK:
                nb[nr][nc] = Piece.DARK_KING
                promoted = True

            ne = eaten | {(mr, mc)}

            further = []
            if not promoted:
                further = self._jump_chains(nr, nc, nb, ne)

            if further:
                for ch in further:
                    chains.append(((r, c),) + ch)
            else:
                chains.append(((r, c), (nr, nc)))

        return chains

    # ── جميع الحركات (مع الأكل الإجباري) ──

    def get_all_moves(self, player):
        """يُرجع (قائمة_الحركات, هل_هي_أكل)"""
        jumps = []
        simple = []

        for r in range(8):
            for c in range(8):
                if self.belongs_to(self.board[r][c], player):
                    jumps.extend(self._jump_chains(r, c))
                    simple.extend(self._simple_moves(r, c))

        # الأكل الإجباري
        if jumps:
            max_len = max(len(j) for j in jumps)
            longest = [j for j in jumps if len(j) == max_len]
            return longest, True

        return simple, False

    # ── تنفيذ حركة ──

    def execute_move(self, move):
        """
        تنفيذ الحركة:
        - القطعة تختفي من مكانها الأصلي
        - تظهر في المكان الجديد
        - القطع المأكولة تختفي
        """
        if len(move) < 2:
            return

        piece = self.board[move[0][0]][move[0][1]]

        # ── مسح الموقع الأصلي (القطعة تختفي) ──
        self.board[move[0][0]][move[0][1]] = Piece.EMPTY

        # ── معالجة كل خطوة في السلسلة ──
        for i in range(len(move) - 1):
            sr, sc = move[i]
            er, ec = move[i + 1]
            dr, dc = er - sr, ec - sc

            # إذا كانت قفزة (أكل)
            if abs(dr) == 2 and abs(dc) == 2:
                mr = sr + dr // 2
                mc = sc + dc // 2
                # ── القطعة المأكولة تختفي ──
                self.board[mr][mc] = Piece.EMPTY

        # ── القطعة تظهر في الموقع الجديد ──
        fr, fc = move[-1]
        self.board[fr][fc] = piece

        # ── ترقية للملك ──
        if fr == 0 and piece == Piece.LIGHT:
            self.board[fr][fc] = Piece.LIGHT_KING
        elif fr == 7 and piece == Piece.DARK:
            self.board[fr][fc] = Piece.DARK_KING

    # ── إحصائيات ──

    def count_pieces(self, player):
        n = k = 0
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if self.belongs_to(p, player):
                    if self.is_king(p):
                        k += 1
                    else:
                        n += 1
        return {"normal": n, "kings": k, "total": n + k}

    def game_over(self):
        """None = مستمرة, LIGHT/DARK = فائز, -1 = تعادل"""
        lt = self.count_pieces(Piece.LIGHT)["total"]
        dt = self.count_pieces(Piece.DARK)["total"]

        if lt == 0:
            return Piece.DARK
        if dt == 0:
            return Piece.LIGHT

        lm, _ = self.get_all_moves(Piece.LIGHT)
        dm, _ = self.get_all_moves(Piece.DARK)

        if not lm and not dm:
            return -1
        if not lm:
            return Piece.DARK
        if not dm:
            return Piece.LIGHT
        return None


# ══════════════════════════════════════════
# 3. محرك الذكاء الاصطناعي
# ══════════════════════════════════════════

class CheckersAI:
    """Minimax + Alpha-Beta Pruning"""

    def __init__(self, depth=4):
        self.depth = depth
        self.nodes = 0

    def evaluate(self, engine, player):
        score = 0.0
        opp = engine.opponent(player)

        for r in range(8):
            for c in range(8):
                p = engine.board[r][c]
                if p == Piece.EMPTY:
                    continue

                val = 25.0 if engine.is_king(p) else 10.0

                if p == Piece.LIGHT:
                    val += (7 - r) * 0.7
                elif p == Piece.DARK:
                    val += r * 0.7

                cd = abs(r - 3.5) + abs(c - 3.5)
                val += (5.0 - cd) * 0.3

                if c == 0 or c == 7:
                    val += 0.5

                if engine.belongs_to(p, player):
                    score += val
                else:
                    score -= val

        my_m, _ = engine.get_all_moves(player)
        op_m, _ = engine.get_all_moves(opp)
        score += len(my_m) * 0.3 - len(op_m) * 0.3

        return score

    def minimax(self, engine, depth, alpha, beta,
                maximizing, current, original):
        self.nodes += 1

        result = engine.game_over()
        if result is not None:
            if result == original:
                return 9999.0, None
            elif result == -1:
                return 0.0, None
            else:
                return -9999.0, None

        if depth == 0:
            return self.evaluate(engine, original), None

        moves, _ = engine.get_all_moves(current)
        if not moves:
            return self.evaluate(engine, original), None

        best_move = moves[0]
        nxt = engine.opponent(current)

        if maximizing:
            max_e = float("-inf")
            for m in moves:
                child = engine.copy()
                child.execute_move(m)
                v, _ = self.minimax(
                    child, depth - 1, alpha, beta,
                    False, nxt, original
                )
                if v > max_e:
                    max_e = v
                    best_move = m
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return max_e, best_move
        else:
            min_e = float("inf")
            for m in moves:
                child = engine.copy()
                child.execute_move(m)
                v, _ = self.minimax(
                    child, depth - 1, alpha, beta,
                    True, nxt, original
                )
                if v < min_e:
                    min_e = v
                    best_move = m
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return min_e, best_move

    def find_best(self, engine, player):
        self.nodes = 0
        t0 = time.time()
        score, move = self.minimax(
            engine, self.depth,
            float("-inf"), float("inf"),
            True, player, player
        )
        return {
            "move": move,
            "score": round(score, 2),
            "nodes": self.nodes,
            "time": round(time.time() - t0, 3),
        }


# ══════════════════════════════════════════
# 4. رسم الرقعة البصرية
# ══════════════════════════════════════════

class Renderer:
    CELL = 70

    @staticmethod
    def render(board, selected=None, valid_dests=None,
               last_move=None):
        C = Renderer.CELL
        sz = C * 8
        img = Image.new("RGB", (sz, sz))
        draw = ImageDraw.Draw(img)

        for r in range(8):
            for c in range(8):
                x1, y1 = c * C, r * C
                x2, y2 = x1 + C, y1 + C

                if (r + c) % 2 == 0:
                    sq = (240, 217, 181)
                else:
                    sq = (181, 136, 99)

                if selected and (r, c) == selected:
                    sq = (100, 220, 100)
                elif valid_dests and (r, c) in valid_dests:
                    sq = (255, 220, 60)
                elif last_move and (r, c) in last_move:
                    if (r + c) % 2 != 0:
                        sq = (150, 180, 220)

                draw.rectangle([x1, y1, x2, y2], fill=sq)

                piece = board[r][c]
                if piece != Piece.EMPTY:
                    cx = x1 + C // 2
                    cy = y1 + C // 2
                    pr = 26

                    draw.ellipse(
                        [cx - pr + 3, cy - pr + 3,
                         cx + pr + 3, cy + pr + 3],
                        fill=(80, 60, 40)
                    )

                    if CheckersEngine.is_light(piece):
                        fl = (255, 253, 245)
                        ed = (200, 190, 175)
                    else:
                        fl = (50, 50, 50)
                        ed = (30, 30, 30)

                    draw.ellipse(
                        [cx - pr, cy - pr, cx + pr, cy + pr],
                        fill=fl, outline=ed, width=2
                    )
                    draw.ellipse(
                        [cx - pr + 6, cy - pr + 6,
                         cx + pr - 6, cy + pr - 6],
                        outline=ed, width=1
                    )

                    if CheckersEngine.is_king(piece):
                        kr = 10
                        draw.ellipse(
                            [cx - kr, cy - kr,
                             cx + kr, cy + kr],
                            fill=(255, 215, 0),
                            outline=(200, 170, 0),
                            width=2
                        )

        return img

    @staticmethod
    def draw_arrow(img, move):
        if not move or len(move) < 2:
            return img
        draw = ImageDraw.Draw(img)
        C = Renderer.CELL

        for i in range(len(move) - 1):
            sr, sc = move[i]
            er, ec = move[i + 1]
            sx = sc * C + C // 2
            sy = sr * C + C // 2
            ex = ec * C + C // 2
            ey = er * C + C // 2

            draw.line(
                [(sx, sy), (ex, ey)],
                fill=(255, 50, 50), width=4
            )
            draw.ellipse(
                [ex - 8, ey - 8, ex + 8, ey + 8],
                fill=(255, 50, 50)
            )

        sr, sc = move[0]
        sx = sc * C + C // 2
        sy = sr * C + C // 2
        draw.ellipse(
            [sx - 10, sy - 10, sx + 10, sy + 10],
            outline=(0, 220, 0), width=4
        )

        return img


# ══════════════════════════════════════════
# 5. نظام التحكم (اضغط → حرّك)
# ══════════════════════════════════════════

def init_state():
    """تهيئة جميع متغيرات الحالة"""
    defaults = {
        "engine": None,
        "selected": None,
        "valid_moves_map": {},
        "movable_pieces": set(),
        "current_player": Piece.LIGHT,
        "human_color": Piece.LIGHT,
        "ai_color": Piece.DARK,
        "history": [],
        "last_move": None,
        "message": "",
        "ai_depth": 4,
        "started": False,
        "move_count": 0,
        "undo_stack": [],
        "mode": "pvai",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state.engine is None:
        st.session_state.engine = CheckersEngine()


def compute_movable():
    """حساب القطع القابلة للتحريك"""
    engine = st.session_state.engine
    player = st.session_state.current_player
    all_moves, _ = engine.get_all_moves(player)
    st.session_state.movable_pieces = set(
        m[0] for m in all_moves
    )


def select_piece(r, c):
    """
    اختيار قطعة: تتحول لـ 🟢
    وتظهر الوجهات المتاحة 🟡
    """
    engine = st.session_state.engine
    player = st.session_state.current_player

    all_moves, is_capture = engine.get_all_moves(player)
    piece_moves = [m for m in all_moves if m[0] == (r, c)]

    if not piece_moves:
        st.session_state.selected = None
        st.session_state.valid_moves_map = {}
        st.session_state.message = "⚠️ هذه القطعة لا تملك حركات"
        return

    st.session_state.selected = (r, c)

    # ربط كل وجهة نهائية بالحركة الكاملة
    dest_map = {}
    for move in piece_moves:
        dest = move[-1]
        if (dest not in dest_map
                or len(move) > len(dest_map[dest])):
            dest_map[dest] = move

    st.session_state.valid_moves_map = dest_map

    if is_capture:
        st.session_state.message = "⚡ أكل إجباري! اختر وجهة"
    else:
        st.session_state.message = "🟡 اضغط على مربع أصفر للتحرك"


def execute_human_move(dest_r, dest_c):
    """
    تنفيذ حركة اللاعب:
    - القطعة تختفي من مكانها ✅
    - تظهر في المكان الجديد ✅
    - القطع المأكولة تختفي ✅
    """
    move = st.session_state.valid_moves_map.get(
        (dest_r, dest_c)
    )
    if not move:
        return

    engine = st.session_state.engine

    # حفظ نسخة للتراجع
    st.session_state.undo_stack.append(
        engine.board.copy().tolist()
    )

    is_cap = (
        len(move) > 2
        or (len(move) == 2
            and abs(move[0][0] - move[1][0]) == 2)
    )

    # ── تنفيذ الحركة ──
    engine.execute_move(move)

    # ── تسجيل الحركة ──
    path = " → ".join(f"({p[0]},{p[1]})" for p in move)
    icon = (
        "⚪" if CheckersEngine.is_light(
            st.session_state.current_player
        ) else "⚫"
    )
    cap_txt = " 💥أكل!" if is_cap else ""
    st.session_state.history.append(
        f"{icon} {path}{cap_txt}"
    )
    st.session_state.move_count += 1

    st.session_state.last_move = move
    st.session_state.selected = None
    st.session_state.valid_moves_map = {}

    # تبديل الدور
    st.session_state.current_player = engine.opponent(
        st.session_state.current_player
    )
    st.session_state.message = ""


def handle_click(r, c):
    """معالجة ضغطة على خلية"""
    engine = st.session_state.engine
    player = st.session_state.current_player

    if (st.session_state.mode == "pvai"
            and player != st.session_state.human_color):
        return

    selected = st.session_state.selected
    valid_map = st.session_state.valid_moves_map

    if selected is None:
        # لا شيء مختار → حاول اختيار قطعة
        if engine.belongs_to(engine.board[r][c], player):
            select_piece(r, c)
    else:
        if (r, c) in valid_map:
            # ضغط على وجهة صالحة → نفّذ الحركة
            execute_human_move(r, c)
        elif (r, c) == selected:
            # ضغط على نفس القطعة → إلغاء الاختيار
            st.session_state.selected = None
            st.session_state.valid_moves_map = {}
            st.session_state.message = ""
        elif engine.belongs_to(engine.board[r][c], player):
            # ضغط على قطعة أخرى → تغيير الاختيار
            select_piece(r, c)
        else:
            # ضغط على مكان غير صالح → إلغاء
            st.session_state.selected = None
            st.session_state.valid_moves_map = {}
            st.session_state.message = ""


def play_ai():
    """دور الذكاء الاصطناعي"""
    engine = st.session_state.engine

    if engine.game_over() is not None:
        return

    ai = CheckersAI(depth=st.session_state.ai_depth)
    result = ai.find_best(engine, st.session_state.ai_color)

    move = result["move"]
    if not move:
        return

    st.session_state.undo_stack.append(
        engine.board.copy().tolist()
    )

    is_cap = (
        len(move) > 2
        or (len(move) == 2
            and abs(move[0][0] - move[1][0]) == 2)
    )

    engine.execute_move(move)

    path = " → ".join(f"({p[0]},{p[1]})" for p in move)
    icon = (
        "⚪" if CheckersEngine.is_light(
            st.session_state.ai_color
        ) else "⚫"
    )
    cap_txt = " 💥أكل!" if is_cap else ""

    st.session_state.history.append(
        f"🤖{icon} {path}{cap_txt}"
    )
    st.session_state.move_count += 1

    st.session_state.last_move = move
    st.session_state.current_player = engine.opponent(
        st.session_state.ai_color
    )
    st.session_state.message = (
        f"🤖 AI لعب: {path} "
        f"({result['time']}s, {result['nodes']} عقدة)"
    )


def undo_move():
    """التراجع عن آخر حركة"""
    if st.session_state.undo_stack:
        old = st.session_state.undo_stack.pop()
        st.session_state.engine = CheckersEngine(old)
        st.session_state.selected = None
        st.session_state.valid_moves_map = {}
        st.session_state.last_move = None

        if st.session_state.history:
            st.session_state.history.pop()

        st.session_state.current_player = (
            CheckersEngine.opponent(
                st.session_state.current_player
            )
        )
        st.session_state.message = "↩️ تم التراجع"


def new_game():
    """بدء لعبة جديدة"""
    st.session_state.engine = CheckersEngine()
    st.session_state.selected = None
    st.session_state.valid_moves_map = {}
    st.session_state.movable_pieces = set()
    st.session_state.current_player = Piece.LIGHT
    st.session_state.history = []
    st.session_state.last_move = None
    st.session_state.message = "🎮 لعبة جديدة! الفاتح يبدأ"
    st.session_state.started = True
    st.session_state.move_count = 0
    st.session_state.undo_stack = []


# ══════════════════════════════════════════
# 6. واجهة Streamlit التفاعلية
# ══════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="♟️ داما تفاعلية",
        page_icon="♟️",
        layout="wide",
    )

    # ── CSS مخصص ──
    st.markdown("""
    <style>
    .block-container {max-width:1100px}

    div[data-testid="stHorizontalBlock"]
        > div[data-testid="column"] {
        padding: 0 1px !important;
    }

    .stButton > button {
        height: 62px !important;
        min-height: 62px !important;
        font-size: 28px !important;
        padding: 0 !important;
        border-radius: 4px !important;
        line-height: 1 !important;
    }

    .turn-box {
        text-align:center; font-size:1.2em;
        padding:12px; border-radius:10px;
        margin:8px 0; font-weight:bold;
    }
    .your-turn {
        background:#d4edda; border:2px solid #28a745;
        color:#155724;
    }
    .ai-turn {
        background:#fff3cd; border:2px solid #ffc107;
        color:#856404;
    }
    .capture-msg {
        background:#f8d7da; border:2px solid #dc3545;
        color:#721c24; text-align:center;
        padding:8px; border-radius:8px; margin:5px 0;
    }
    .result-box {
        background:linear-gradient(135deg,#667eea,#764ba2);
        color:#fff; padding:20px; border-radius:12px;
        text-align:center; font-size:1.5em; margin:10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    st.title("♟️ لعبة الداما التفاعلية")
    st.caption("اضغط على قطعتك 🟢 ثم اضغط على الوجهة 🟡")

    # ═══ الشريط الجانبي ═══
    with st.sidebar:
        st.header("⚙️ الإعدادات")

        mode = st.radio(
            "🎮 نوع اللعبة:",
            ["🤖 ضد الذكاء الاصطناعي", "👥 لاعبَين"],
        )
        st.session_state.mode = (
            "pvai" if "ذكاء" in mode else "pvp"
        )

        if st.session_state.mode == "pvai":
            color = st.radio(
                "♟️ لونك:",
                ["⚪ الفاتح (أعلى)", "⚫ الداكن (أسفل)"],
            )
            st.session_state.human_color = (
                Piece.LIGHT if "الفاتح" in color
                else Piece.DARK
            )
            st.session_state.ai_color = (
                CheckersEngine.opponent(
                    st.session_state.human_color
                )
            )

            diff = st.select_slider(
                "🧠 الصعوبة:",
                ["سهل", "متوسط", "صعب", "خبير"],
                value="متوسط",
            )
            depth_map = {
                "سهل": 2, "متوسط": 4,
                "صعب": 6, "خبير": 8,
            }
            st.session_state.ai_depth = depth_map[diff]

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "🆕 جديدة",
                use_container_width=True,
                type="primary",
            ):
                new_game()
                st.rerun()
        with c2:
            if st.button(
                "↩️ تراجع",
                use_container_width=True,
            ):
                undo_move()
                st.rerun()

        st.divider()

        # إحصائيات
        engine = st.session_state.engine
        li = engine.count_pieces(Piece.LIGHT)
        di = engine.count_pieces(Piece.DARK)

        st.markdown("### 📊 القطع")
        mc1, mc2 = st.columns(2)
        with mc1:
            lbl = "⚪"
            if st.session_state.mode == "pvai":
                lbl += (
                    " أنت"
                    if st.session_state.human_color
                       == Piece.LIGHT
                    else " AI"
                )
            kd = (
                f"👑×{li['kings']}"
                if li["kings"] else None
            )
            st.metric(lbl, li["total"], delta=kd)
        with mc2:
            lbl = "⚫"
            if st.session_state.mode == "pvai":
                lbl += (
                    " أنت"
                    if st.session_state.human_color
                       == Piece.DARK
                    else " AI"
                )
            kd = (
                f"👑×{di['kings']}"
                if di["kings"] else None
            )
            st.metric(lbl, di["total"], delta=kd)

        st.metric("🔢 الحركات", st.session_state.move_count)

        # السجل
        if st.session_state.history:
            st.divider()
            with st.expander(
                f"📜 السجل ({len(st.session_state.history)})"
            ):
                for i, h in enumerate(
                    st.session_state.history, 1
                ):
                    st.text(f"{i}. {h}")

        st.divider()
        st.markdown("""
        **🎮 طريقة اللعب:**
        1. اضغط على قطعتك → 🟢
        2. اضغط على 🟡 للتحرك
        3. القطعة تنتقل تلقائياً
        4. المأكولة تختفي فوراً
        """)

    # ═══ المحتوى الرئيسي ═══

    # ── فحص نهاية اللعبة ──
    game_result = engine.game_over()
    if game_result is not None:
        if game_result == -1:
            st.markdown(
                '<div class="result-box">🤝 تعادل!</div>',
                unsafe_allow_html=True,
            )
        elif (st.session_state.mode == "pvp"
              or game_result == st.session_state.human_color):
            st.balloons()
            winner = (
                "⚪ الفاتح" if game_result == Piece.LIGHT
                else "⚫ الداكن"
            )
            st.markdown(
                f'<div class="result-box">'
                f"🏆 فاز {winner}!</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-box">'
                "😞 خسرت! حاول مرة أخرى</div>",
                unsafe_allow_html=True,
            )

        if st.button("🔄 العب مرة أخرى", type="primary"):
            new_game()
            st.rerun()
        return

    # ── دور الذكاء الاصطناعي ──
    is_ai_turn = (
        st.session_state.mode == "pvai"
        and st.session_state.current_player
            == st.session_state.ai_color
        and st.session_state.started
    )

    if is_ai_turn:
        with st.spinner("🤖 الذكاء الاصطناعي يفكر..."):
            play_ai()
        st.rerun()

    # ── حساب القطع القابلة للتحريك ──
    compute_movable()

    # ── مؤشر الدور ──
    current_emoji = (
        "⚪" if st.session_state.current_player
              == Piece.LIGHT
        else "⚫"
    )

    is_human = (
        st.session_state.mode == "pvp"
        or st.session_state.current_player
           == st.session_state.human_color
    )

    if is_human:
        if st.session_state.selected:
            txt = (
                f"{current_emoji} اختر المربع الأصفر"
                f" 🟡 للتحرك — أو اضغط قطعة أخرى"
            )
        else:
            txt = (
                f"{current_emoji} دورك!"
                f" اضغط على إحدى قطعك لاختيارها"
            )
        st.markdown(
            f'<div class="turn-box your-turn">{txt}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.message:
        st.info(st.session_state.message)

    # ── فحص الأكل الإجباري ──
    _, is_forced_capture = engine.get_all_moves(
        st.session_state.current_player
    )
    if is_forced_capture and st.session_state.selected is None:
        st.markdown(
            '<div class="capture-msg">'
            "⚡ أكل إجباري! يجب عليك الأكل</div>",
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════
    #  🎮 الرقعة التفاعلية
    # ═══════════════════════════════

    col_grid, col_visual = st.columns([1.4, 1])

    with col_grid:
        board = engine.board
        selected = st.session_state.selected
        valid_dests = set(
            st.session_state.valid_moves_map.keys()
        )
        player = st.session_state.current_player
        movable = st.session_state.movable_pieces

        for r in range(8):
            cols = st.columns(8)
            for c in range(8):
                with cols[c]:
                    piece = int(board[r][c])
                    playable = (r + c) % 2 != 0
                    is_sel = selected == (r, c)
                    is_dest = (r, c) in valid_dests

                    # ── تحديد الرمز ──
                    if is_sel:
                        label = "🟢"
                    elif is_dest:
                        label = "🟡"
                    elif piece == Piece.LIGHT:
                        label = "⚪"
                    elif piece == Piece.DARK:
                        label = "⚫"
                    elif piece == Piece.LIGHT_KING:
                        label = "👑"
                    elif piece == Piece.DARK_KING:
                        label = "♛"
                    elif playable:
                        label = "▪"
                    else:
                        label = "▫"

                    # ── هل يمكن الضغط؟ ──
                    can_click = False

                    if is_human and playable:
                        if selected is None:
                            can_click = (
                                (r, c) in movable
                            )
                        else:
                            can_click = (
                                is_dest
                                or is_sel
                                or (r, c) in movable
                            )

                    # ── الزر ──
                    st.button(
                        label,
                        key=f"b{r}{c}",
                        use_container_width=True,
                        disabled=not can_click,
                        on_click=handle_click,
                        args=(r, c),
                    )

    # ── العرض البصري ──
    with col_visual:
        st.markdown("##### 🎨 العرض البصري")

        vd = (
            set(st.session_state.valid_moves_map.keys())
            if st.session_state.valid_moves_map
            else None
        )
        visual = Renderer.render(
            board,
            selected=st.session_state.selected,
            valid_dests=vd,
            last_move=st.session_state.last_move,
        )

        if st.session_state.last_move:
            visual = Renderer.draw_arrow(
                visual, st.session_state.last_move
            )

        st.image(visual, use_container_width=True)

        # دليل الرموز
        st.markdown("""
        | الرمز | المعنى |
        |:-----:|:------:|
        | 🟢 | قطعتك المختارة |
        | 🟡 | حركة متاحة (اضغط!) |
        | ⚪ | قطعة فاتحة |
        | ⚫ | قطعة داكنة |
        | 👑 | ملك فاتح |
        | ♛ | ملك داكن |
        """)

        # زر اقتراح
        if (is_human
            and st.session_state.mode == "pvai"
            and st.session_state.started):
            if st.button(
                "💡 ساعدني!",
                use_container_width=True,
            ):
                with st.spinner("🧠 جاري التحليل..."):
                    ai = CheckersAI(
                        depth=st.session_state.ai_depth
                    )
                    res = ai.find_best(
                        engine,
                        st.session_state.human_color,
                    )
                    if res["move"]:
                        hint = Renderer.render(board)
                        hint = Renderer.draw_arrow(
                            hint, res["move"]
                        )
                        st.image(
                            hint,
                            caption=(
                                f"💡 اقتراح "
                                f"(تقييم: {res['score']})"
                            ),
                            use_container_width=True,
                        )

    # ── Footer ──
    st.divider()
    st.markdown(
        '<p style="text-align:center;color:#999;'
        'font-size:0.8em;">'
        "♟️ داما تفاعلية v3.0 — "
        "Minimax + Alpha-Beta — "
        "بدون إنترنت أو APIs"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
