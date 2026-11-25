#!/usr/bin/env python3
"""Dual-backend builder for the WSDM 2027 overview figure (fig0_overview).

Generates by default (anchored to repository root):
  1. paper/figs/fig0_overview.pdf  (vector PDF for LaTeX \textwidth = 7.01 in)
  2. assets/overview.png           (hi-res PNG preview used in README.md)

Pass --pptx to additionally export assets/overview.pptx (pure ECMA-376 OpenXML
native shapes/paths/textboxes). All coordinates live in one shared
1600 x 520 px scene graph (1 px = 9525 EMU = 0.75 pt).
"""
from __future__ import annotations

import argparse
import math
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, Circle, PathPatch

# ---------------------------------------------------------------------------
# Canvas & Typography Constants (1600 x 520 px, 1 px = 0.75 pt = 9525 EMU)
# ---------------------------------------------------------------------------
W_PX = 1600
H_PX = 520
EMU_PER_PX = 9525
PT_PER_PX = 0.75
FONT_NAME = "Times New Roman"

# Palette (Technical + Mathematical Hybrid)
C_BG = "#FFFFFF"
C_INK = "#1E2933"
C_MUTED = "#5A6570"
C_DIVIDER = "#D5DCE0"

C_GOLD_FILL = "#FFF3CD"
C_GOLD_STROKE = "#B78313"
C_GOLD_TEXT = "#7A5506"

C_BLUE_FILL = "#DCEAF7"
C_BLUE_FILL_SOFT = "#EAF3FB"
C_BLUE_STROKE = "#2B5C8F"
C_BLUE_TEXT = "#1A3A5C"

C_AMBER_FILL = "#FDEBD0"
C_AMBER_STROKE = "#C87A19"
C_AMBER_TEXT = "#874C08"

C_CORAL_FILL = "#FADBD8"
C_CORAL_FILL_LIGHT = "#FDF2F0"
C_CORAL_FILL_DARK = "#F5B7B1"
C_CORAL_STROKE = "#B03A2E"
C_CORAL_TEXT = "#78281F"

C_GRAY_FILL = "#EAECEF"
C_GRAY_FILL_LIGHT = "#F5F7F9"
C_GRAY_STROKE = "#6C757D"
C_GRAY_TEXT = "#3D454D"

C_CAN_FILL = "#EAF5EE"
C_CAN_STROKE = "#2D7A4D"
C_CAN_TEXT = "#1F5C38"

C_CANNOT_FILL = "#FBECEB"
C_CANNOT_STROKE = "#A93226"
C_CANNOT_TEXT = "#7D231B"


# ---------------------------------------------------------------------------
# Shared Scene Graph
# ---------------------------------------------------------------------------
@dataclass
class BoxObj:
    group: str
    name: str
    x: float
    y: float
    w: float
    h: float
    fill: Optional[str] = "#FFFFFF"
    stroke: Optional[str] = C_INK
    lw: float = 1.4
    radius: float = 0.0
    dash: str = "solid"
    alpha: float = 1.0


@dataclass
class EllipseObj:
    group: str
    name: str
    cx: float
    cy: float
    rx: float
    ry: float
    fill: Optional[str] = C_BLUE_STROKE
    stroke: Optional[str] = None
    lw: float = 0.0
    alpha: float = 1.0


@dataclass
class PathObj:
    group: str
    name: str
    cmds: List[Tuple]
    fill: Optional[str] = None
    stroke: Optional[str] = C_INK
    lw: float = 1.5
    dash: str = "solid"
    alpha: float = 1.0


@dataclass
class TextObj:
    group: str
    name: str
    x: float
    y: float
    w: float
    h: float
    mpl_text: str
    pptx_text: Optional[str] = None
    size_px: float = 22.0
    bold: bool = False
    italic: bool = False
    color: str = C_INK
    align: str = "left"
    valign: str = "center"
    line_spacing: float = 1.12


class Scene:
    def __init__(self) -> None:
        self.items: List[object] = []

    def box(self, group: str, name: str, x: float, y: float, w: float, h: float, **kw) -> None:
        self.items.append(BoxObj(group, name, x, y, w, h, **kw))

    def circle(self, group: str, name: str, cx: float, cy: float, r: float, **kw) -> None:
        self.items.append(EllipseObj(group, name, cx, cy, r, r, **kw))

    def line(self, group: str, name: str, x1: float, y1: float, x2: float, y2: float,
             stroke: str = C_INK, lw: float = 1.5, dash: str = "solid", alpha: float = 1.0) -> None:
        self.items.append(PathObj(group, name, [("M", x1, y1), ("L", x2, y2)],
                                  fill=None, stroke=stroke, lw=lw, dash=dash, alpha=alpha))

    def poly(self, group: str, name: str, pts: Sequence[Tuple[float, float]],
             fill: Optional[str] = None, stroke: Optional[str] = C_INK,
             lw: float = 1.2, dash: str = "solid", alpha: float = 1.0) -> None:
        cmds: List[Tuple] = [("M", pts[0][0], pts[0][1])]
        for px, py in pts[1:]:
            cmds.append(("L", px, py))
        cmds.append(("Z",))
        self.items.append(PathObj(group, name, cmds, fill=fill, stroke=stroke, lw=lw, dash=dash, alpha=alpha))

    def polyline(self, group: str, name: str, pts: Sequence[Tuple[float, float]],
                 stroke: str = C_INK, lw: float = 2.0, dash: str = "solid", alpha: float = 1.0) -> None:
        cmds: List[Tuple] = [("M", pts[0][0], pts[0][1])]
        for px, py in pts[1:]:
            cmds.append(("L", px, py))
        self.items.append(PathObj(group, name, cmds, fill=None, stroke=stroke, lw=lw, dash=dash, alpha=alpha))

    def arrow(self, group: str, name: str, x1: float, y1: float, x2: float, y2: float,
              color: str = C_INK, lw: float = 1.8, head_len: float = 8.5, head_w: float = 6.5,
              dash: str = "solid", double: bool = False) -> None:
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return
        ux, uy = dx / dist, dy / dist
        nx, ny = -uy, ux
        sx = x1 + ux * (head_len * 0.78 if double else 0.0)
        sy = y1 + uy * (head_len * 0.78 if double else 0.0)
        ex = x2 - ux * (head_len * 0.78)
        ey = y2 - uy * (head_len * 0.78)
        self.line(group, name + " shaft", sx, sy, ex, ey, stroke=color, lw=lw, dash=dash)
        tip = (x2, y2)
        b1 = (x2 - ux * head_len + nx * (head_w / 2.0), y2 - uy * head_len + ny * (head_w / 2.0))
        b2 = (x2 - ux * head_len - nx * (head_w / 2.0), y2 - uy * head_len - ny * (head_w / 2.0))
        self.poly(group, name + " head", [tip, b1, b2], fill=color, stroke=None, lw=0.0)
        if double:
            tip0 = (x1, y1)
            a1 = (x1 + ux * head_len + nx * (head_w / 2.0), y1 + uy * head_len + ny * (head_w / 2.0))
            a2 = (x1 + ux * head_len - nx * (head_w / 2.0), y1 + uy * head_len - ny * (head_w / 2.0))
            self.poly(group, name + " head0", [tip0, a1, a2], fill=color, stroke=None, lw=0.0)

    def curve_arrow(self, group: str, name: str,
                    p0: Tuple[float, float], p1: Tuple[float, float],
                    p2: Tuple[float, float], p3: Tuple[float, float],
                    color: str = C_INK, lw: float = 1.8, head_len: float = 8.0, head_w: float = 6.0,
                    dash: str = "solid") -> None:
        tx, ty = p3[0] - p2[0], p3[1] - p2[1]
        tlen = math.hypot(tx, ty)
        if tlen < 1e-3:
            tx, ty, tlen = p3[0] - p0[0], p3[1] - p0[1], math.hypot(p3[0] - p0[0], p3[1] - p0[1])
        ux, uy = tx / max(tlen, 1e-3), ty / max(tlen, 1e-3)
        nx, ny = -uy, ux
        end_shaft = (p3[0] - ux * (head_len * 0.72), p3[1] - uy * (head_len * 0.72))
        cmds = [("M", p0[0], p0[1]), ("C", p1[0], p1[1], p2[0], p2[1], end_shaft[0], end_shaft[1])]
        self.items.append(PathObj(group, name + " curve", cmds, fill=None, stroke=color, lw=lw, dash=dash))
        b1 = (p3[0] - ux * head_len + nx * (head_w / 2.0), p3[1] - uy * head_len + ny * (head_w / 2.0))
        b2 = (p3[0] - ux * head_len - nx * (head_w / 2.0), p3[1] - uy * head_len - ny * (head_w / 2.0))
        self.poly(group, name + " head", [p3, b1, b2], fill=color, stroke=None, lw=0.0)

    def text(self, group: str, name: str, x: float, y: float, w: float, h: float,
             mpl_text: str, pptx_text: Optional[str] = None, **kw) -> None:
        self.items.append(TextObj(group, name, x, y, w, h, mpl_text, pptx_text, **kw))


# ---------------------------------------------------------------------------
# Scene Definition (Panels a, b, c, d + Dividers)
# ---------------------------------------------------------------------------
def outcome_strip(sc: Scene, group: str, x: float, y: float, w: float,
                  kind: str, body_mpl: str, body_pptx: Optional[str] = None) -> None:
    """Render subtle bottom boundary pill (CAN / CANNOT / LIMIT) without cartoons."""
    if kind == "CAN":
        fill, stroke, pref_col, tag = C_CAN_FILL, C_CAN_STROKE, C_CAN_TEXT, "CAN:"
        tag_w = 52.0
    elif kind == "CANNOT":
        fill, stroke, pref_col, tag = C_CANNOT_FILL, C_CANNOT_STROKE, C_CANNOT_TEXT, "CANNOT:"
        tag_w = 94.0
    else:
        fill, stroke, pref_col, tag = C_CANNOT_FILL, C_CANNOT_STROKE, C_CANNOT_TEXT, "LIMIT:"
        tag_w = 72.0
    h = 36.0
    sc.box(group, f"{kind} pill", x, y, w, h, fill=fill, stroke=stroke, lw=1.35, radius=5.0)
    sc.text(group, f"{kind} tag", x + 8.0, y, tag_w, h, tag, tag,
            size_px=20.0, bold=True, color=pref_col, align="left", valign="center")
    body_x = x + 8.0 + tag_w + 5.0
    sc.text(group, f"{kind} text", body_x, y, x + w - 6.0 - body_x, h,
            body_mpl, body_pptx, size_px=20.0, bold=False, color=C_INK, align="left", valign="center")


def build_scene() -> Scene:
    sc = Scene()

    # Thin vertical section dividers (no heavy card frames!)
    for idx, dx in enumerate([415.0, 835.0, 1218.0], start=1):
        sc.line("dividers", f"Divider {idx}", dx, 12.0, dx, 506.0, stroke=C_DIVIDER, lw=1.2)

    # =========================================================================
    # PANEL (a): Volume-matched slicing  [x = 12 .. 404]
    # =========================================================================
    sc.text("panel-a-header", "Panel a title", 12, 8, 392, 34,
            "(a) Volume-matched slicing", "(a) Volume-matched slicing",
            size_px=26.5, bold=True, color=C_INK, align="left")

    # Top timeline & interaction log
    g_log = "a-timeline"
    sc.text(g_log, "Log label", 14, 48, 170, 22, "Interaction log", "Interaction log",
            size_px=20.5, color=C_MUTED, align="left")
    sc.text(g_log, "t0 label", 304, 48, 40, 22, r"$t_0$", r"$t_0$",
            size_px=20.5, bold=True, color=C_GOLD_TEXT, align="center")
    sc.box(g_log, "Log bar", 14, 72, 310, 27, fill=C_GRAY_FILL_LIGHT, stroke=C_GRAY_STROKE, lw=1.2, radius=2.0)

    # Event ticks whose density visibly increases left -> right
    n_ticks = 30
    tick_xs = [18.0 + 301.0 * (1.0 - (1.0 - i / (n_ticks - 1)) ** 2.15) for i in range(n_ticks)]
    clean_ticks: List[float] = []
    for tx in tick_xs:
        if not clean_ticks or tx - clean_ticks[-1] >= 4.2:
            clean_ticks.append(tx)
    for i, tx in enumerate(clean_ticks):
        sc.line(g_log, f"Log tick {i}", tx, 76.0, tx, 95.0, stroke="#4A5568", lw=1.25)

    # Gold Test window [t_0, t_test]
    sc.box(g_log, "Test window", 324, 70, 58, 31, fill=C_GOLD_FILL, stroke=C_GOLD_STROKE, lw=1.6, radius=3.0)
    sc.text(g_log, "Test label", 324, 70, 58, 31, "Test", "Test",
            size_px=20.5, bold=True, color=C_GOLD_TEXT, align="center")
    sc.arrow(g_log, "Time axis", 14, 106, 402, 106, color=C_GRAY_STROKE, lw=1.2, head_len=7.0, head_w=5.5)
    sc.text(g_log, "t label", 386, 75, 18, 22, r"$t$", r"$t$", size_px=21.0, italic=False, color=C_INK, align="center")

    # Dashed vertical boundary line at t_0
    sc.line("a-t0-guide", "t0 dashed guide", 324, 70, 324, 274, stroke=C_GOLD_STROKE, lw=1.2, dash="dash")

    # Arm A (Translated)
    g_arma = "a-arm-a"
    sc.text(g_arma, "Arm A title", 14, 111, 210, 22, "Arm A (translated)", "Arm A (translated)",
            size_px=21.0, bold=True, color=C_BLUE_TEXT, align="left")
    arm_a_specs = [
        (r"$\tau_3$", 18.0, 68.0),
        (r"$\tau_2$", 140.0, 68.0),
        (r"$\tau_1$", 244.0, 68.0),
    ]
    for label, wx, ww in arm_a_specs:
        sc.box(g_arma, f"ArmA box {label}", wx, 135, ww, 33,
               fill=C_BLUE_FILL, stroke=C_BLUE_STROKE, lw=1.45, radius=3.0)
        sc.text(g_arma, f"ArmA label {label}", wx, 135, ww, 20, label, label,
                size_px=20.0, bold=True, color=C_BLUE_TEXT, align="center")
        sub_ticks = [tx for tx in clean_ticks if wx + 4.0 <= tx <= wx + ww - 4.0]
        for j, tx in enumerate(sub_ticks):
            sc.line(g_arma, f"ArmA tick {label} {j}", tx, 155.0, tx, 165.0, stroke="#3B6FA6", lw=1.15)

    # Arm B (Widened)
    g_armb = "a-arm-b"
    sc.text(g_armb, "Arm B title", 14, 173, 195, 22, "Arm B (widened)", "Arm B (widened)",
            size_px=21.0, bold=True, color=C_CORAL_TEXT, align="left")
    arm_b_tiers = [
        (r"$\tau_3$", 18.0, 306.0, 196.0, 60.0, C_CORAL_FILL_LIGHT, C_CORAL_STROKE),
        (r"$\tau_2$", 144.0, 180.0, 206.0, 47.0, C_CORAL_FILL, C_CORAL_STROKE),
        (r"$\tau_1$", 244.0, 80.0, 217.0, 33.0, C_CORAL_FILL_DARK, "#922B21"),
    ]
    for label, wx, ww, wy, wh, fill_c, str_c in arm_b_tiers:
        sc.box(g_armb, f"ArmB box {label}", wx, wy, ww, wh,
               fill=fill_c, stroke=str_c, lw=1.4, radius=3.0)
        sc.text(g_armb, f"ArmB label {label}", wx + 6.0, wy + 1.0, 32.0, 21.0,
                label, label, size_px=20.0, bold=True, color=C_CORAL_TEXT, align="left")
    for j, tx in enumerate(clean_ticks):
        if 22.0 <= tx <= 320.0:
            sc.line(g_armb, f"ArmB tick {j}", tx, 241.0, tx, 253.0, stroke="#9E4840", lw=1.05, alpha=0.72)

    # Event-weighted age != midpoint offset callout below Arm B
    mid_x = 171.0
    ewt_x = 246.0
    sc.line(g_armb, "Mid tick", mid_x, 257, mid_x, 270, stroke=C_MUTED, lw=1.5)
    sc.line(g_armb, "Ewt tick", ewt_x, 257, ewt_x, 270, stroke=C_CORAL_STROKE, lw=1.8)
    sc.arrow(g_armb, "Shift arrow", mid_x + 2, 264, ewt_x - 2, 264,
             color=C_CORAL_STROKE, lw=1.5, head_len=6.5, head_w=5.0)
    sc.text(g_armb, "Event-wt label", 82, 273, 230, 21,
            r"event-weighted $\neq$ midpoint", "event-weighted ≠ midpoint",
            size_px=20.0, italic=True, color=C_CORAL_TEXT, align="center")

    # Clean non-overlapping funnel arrows -> Uniform subsample = D events -> dots -> Probe -> L_X(tau)
    g_sub = "a-subsample"
    sc.arrow(g_sub, "Funnel L", 52, 260, 75, 298, color=C_MUTED, lw=1.35, head_len=6.5, head_w=5.0)
    sc.arrow(g_sub, "Funnel R", 320, 260, 317, 298, color=C_MUTED, lw=1.35, head_len=6.5, head_w=5.0)
    sc.text(g_sub, "Subsample label", 24, 300, 344, 23,
            r"Uniform subsample = $D$ events", r"Uniform subsample = $D$ events",
            size_px=20.5, bold=True, color=C_INK, align="center")

    # Two rows of 6 equal solid dots (visually identical sample count D)
    for k in range(6):
        cx = 26.0 + k * 21.0
        sc.circle(g_sub, f"DotA {k}", cx, 345.0, 5.2, fill=C_BLUE_STROKE)
        sc.circle(g_sub, f"DotB {k}", cx, 373.0, 5.2, fill=C_CORAL_STROKE)
    sc.polyline(g_sub, "D bracket", [(144, 339), (150, 339), (150, 379), (144, 379)], stroke=C_MUTED, lw=1.3)
    sc.text(g_sub, "D label", 152, 347, 30, 24, r"$D$", r"$D$", size_px=20.5, bold=True, color=C_INK, align="center")

    sc.arrow(g_sub, "To Probe", 183, 359, 204, 359, color=C_INK, lw=1.6, head_len=7.0, head_w=5.5)
    sc.box(g_sub, "Probe pill", 206, 335, 86, 48, fill=C_GRAY_FILL_LIGHT, stroke=C_INK, lw=1.4, radius=6.0)
    sc.text(g_sub, "Probe text", 206, 336, 86, 46,
            "Probe\n" + r"$d{=}32$", "Probe\n" + r"$d=32$",
            size_px=20.0, bold=True, color=C_INK, align="center")
    sc.arrow(g_sub, "To Curve", 294, 359, 316, 359, color=C_INK, lw=1.6, head_len=7.0, head_w=5.5)

    # Mini L_X(tau) Box-Cox plot
    g_curve = "a-boxcox"
    sc.text(g_curve, "LX title", 320, 326, 78, 22, r"$L_X(\tau)$", r"$L_X(\tau)$",
            size_px=20.5, bold=True, color=C_INK, align="center")
    sc.polyline(g_curve, "Mini axes", [(325, 348), (325, 396), (395, 396)], stroke=C_INK, lw=1.3)
    pts_a = [(330 + i * 10.0, 389 - 35.0 * (i / 6.0) ** 1.45) for i in range(7)]
    pts_b = [(330 + i * 10.0, 382 - 30.0 * (i / 6.0) ** 1.15) for i in range(7)]
    sc.polyline(g_curve, "Mini curve A", pts_a, stroke=C_BLUE_STROKE, lw=2.2)
    sc.polyline(g_curve, "Mini curve B", pts_b, stroke=C_CORAL_STROKE, lw=2.2, dash="dash")

    # Bottom boundary strips (Panel a)
    outcome_strip(sc, "a-can", 12, 425, 392, "CAN",
                  r"decouple age $\tau$ from volume $D$", r"decouple age $\tau$ from volume $D$")
    outcome_strip(sc, "a-cannot", 12, 467, 392, "CANNOT",
                  r"fix $\hat{\delta}$ across period or axis", "fix δ̂ across period or axis")

    # =========================================================================
    # PANEL (b): Four-way visibility & cold mix  [x = 425 .. 824]
    # =========================================================================
    sc.text("panel-b-header", "Panel b title", 425, 8, 399, 34,
            "(b) Four-way visibility & cold mix", "(b) Four-way visibility & cold mix",
            size_px=26.5, bold=True, color=C_INK, align="left")

    def draw_matrix(group: str, prefix: str, mx: float, my: float, title_mpl: str, title_pptx: str,
                    title_col: str, dot_counts: Dict[str, int], aged: bool = False) -> None:
        cw = 63.0
        sc.text(group, f"{prefix} title", mx, my - 51, cw * 2, 23, title_mpl, title_pptx,
                size_px=21.5, bold=True, color=title_col, align="center")
        sc.text(group, f"{prefix} item hdr", mx, my - 26, cw * 2, 21, "Item: 1     0", "Item: 1     0",
                size_px=20.0, color=C_MUTED, align="center")
        if prefix == "Fresh":
            sc.text(group, "User hdr", mx - 46, my + 2, 42, 22, "User", "User",
                    size_px=20.0, color=C_MUTED, align="right")
            sc.text(group, "Row 1", mx - 24, my + 26, 18, 22, "1", "1", size_px=20.0, color=C_MUTED, align="right")
            sc.text(group, "Row 0", mx - 24, my + cw + 20, 18, 22, "0", "0", size_px=20.0, color=C_MUTED, align="right")

        cell_specs = [
            ("11", 0, 0, C_BLUE_FILL, C_BLUE_STROKE, C_BLUE_TEXT),
            ("10", 1, 0, C_AMBER_FILL, C_AMBER_STROKE, C_AMBER_TEXT),
            ("01", 0, 1, C_CORAL_FILL, C_CORAL_STROKE, C_CORAL_TEXT),
            ("00", 1, 1, C_GRAY_FILL, C_GRAY_STROKE if not aged else "#495057", C_GRAY_TEXT),
        ]
        offsets = [
            (0.42, 0.56), (0.64, 0.44), (0.76, 0.68), (0.52, 0.78),
            (0.80, 0.34), (0.34, 0.78), (0.66, 0.64), (0.48, 0.40),
            (0.82, 0.80),
        ]
        for code, col_i, row_i, fc, sc_c, tc in cell_specs:
            cx0 = mx + col_i * cw
            cy0 = my + row_i * cw
            lw_c = 1.8 if (aged and code == "00") else 1.4
            sc.box(group, f"{prefix} cell {code}", cx0, cy0, cw, cw, fill=fc, stroke=sc_c, lw=lw_c)
            sc.text(group, f"{prefix} code {code}", cx0 + 4, cy0 + 2, 28, 21,
                    rf"$\mathbf{{{code}}}$", rf"$\mathbf{{{code}}}$",
                    size_px=20.0, bold=True, color=tc, align="left")
            nd = dot_counts[code]
            dot_col = sc_c
            if aged and code in ("10", "01"):
                pts = [(0.64, 0.36), (0.36, 0.54)] if code == "10" else [(0.36, 0.68), (0.54, 0.78)]
                for k, (ox, oy) in enumerate(pts[:nd]):
                    sc.circle(group, f"{prefix} {code} dot {k}", cx0 + ox * cw, cy0 + oy * cw, 3.6, fill=dot_col, alpha=0.75)
            elif aged and code == "00":
                pts00 = [(0.52, 0.42), (0.70, 0.40), (0.42, 0.64), (0.62, 0.60),
                         (0.80, 0.64), (0.36, 0.84), (0.56, 0.82), (0.76, 0.84), (0.62, 0.24)]
                for k, (ox, oy) in enumerate(pts00[:nd]):
                    sc.circle(group, f"{prefix} {code} dot {k}", cx0 + ox * cw, cy0 + oy * cw, 3.6, fill="#3D454D")
            else:
                for k, (ox, oy) in enumerate(offsets[:nd]):
                    sc.circle(group, f"{prefix} {code} dot {k}", cx0 + ox * cw, cy0 + oy * cw, 3.6, fill=dot_col)

        if aged:
            sc.text(group, "omega drop", mx + 20, my + 38, 40, 21,
                    r"$\omega{\downarrow}$", "ω↓", size_px=20.0, bold=True, color=C_BLUE_TEXT, align="right")

    draw_matrix("b-fresh", "Fresh", 476, 108, r"Fresh ($\tau_1$)", r"Fresh ($\tau_1$)",
                C_BLUE_TEXT, {"11": 9, "10": 4, "01": 4, "00": 1}, aged=False)

    sc.arrow("b-age-arrow", "Age arrow", 610, 171, 654, 171, color=C_INK, lw=2.0, head_len=8.5, head_w=6.5)
    sc.text("b-age-arrow", "Age label", 606, 144, 52, 22, "age", "age",
            size_px=20.5, bold=True, italic=True, color=C_INK, align="center")

    draw_matrix("b-aged", "Aged", 666, 108, r"Aged ($\tau_3$)", r"Aged ($\tau_3$)",
                C_CORAL_TEXT, {"11": 3, "10": 2, "01": 2, "00": 9}, aged=True)

    # Explicit curved migration arrows from aged 10 & 01 into 00
    g_mig = "b-migration"
    sc.curve_arrow(g_mig, "Arrow 10->00", (778, 150), (806, 160), (806, 184), (784, 192),
                   color=C_AMBER_STROKE, lw=2.1, head_len=8.0, head_w=6.0)
    sc.curve_arrow(g_mig, "Arrow 01->00", (712, 214), (724, 232), (736, 232), (746, 216),
                   color=C_CORAL_STROKE, lw=2.1, head_len=8.0, head_w=6.0)
    sc.text(g_mig, "Cold-mix shift", 630, 244, 194, 24,
            "cold-mix shift (24%)", "cold-mix shift (24%)",
            size_px=20.5, bold=True, color=C_CORAL_TEXT, align="center")

    # Compact legend / mechanism breakdown rows
    g_leg = "b-mechanism"
    sc.box(g_leg, "Swatch 10", 430, 286, 14, 14, fill=C_AMBER_FILL, stroke=C_AMBER_STROKE, lw=1.3, radius=2.0)
    sc.box(g_leg, "Swatch 01", 447, 286, 14, 14, fill=C_CORAL_FILL, stroke=C_CORAL_STROKE, lw=1.3, radius=2.0)
    sc.text(g_leg, "Row 10/01", 467, 280, 356, 26,
            r"$\mathbf{10}/\mathbf{01}$ ($\mu{+}b_u,\mu{+}b_i$): stale bias; shift $\to$ $\mathbf{00}$",
            r"$\mathbf{10}/\mathbf{01}$ ($\mu+b_u,\mu+b_i$): stale bias; shift → $\mathbf{00}$",
            size_px=20.0, color=C_INK, align="left")

    sc.box(g_leg, "Swatch 00", 430, 322, 14, 14, fill=C_GRAY_FILL, stroke=C_GRAY_STROKE, lw=1.3, radius=2.0)
    sc.text(g_leg, "Row 00", 451, 316, 370, 26,
            r"$\mathbf{00}$ ($\mu$): global base rate, $+0.5\%$ aging",
            r"$\mathbf{00}$ ($\mu$): global base rate, +0.5% aging",
            size_px=20.0, color=C_INK, align="left")

    sc.line(g_leg, "SideFeat bar", 432, 357, 432, 405, stroke=C_BLUE_STROKE, lw=3.0)
    sc.text(g_leg, "SideFeat text", 442, 354, 380, 54,
            "Static metadata (SideFeatMF) lifts " + r"$\mathbf{00}$" + " AUC\n& flattens " + r"$\hat{\delta}$" + "; cold aging remains substantial",
            "Static metadata (SideFeatMF) lifts " + r"$\mathbf{00}$" + " AUC\n& flattens δ̂; cold aging remains substantial",
            size_px=20.0, color=C_INK, align="left", line_spacing=1.18)

    # Bottom boundary strips (Panel b)
    outcome_strip(sc, "b-can", 425, 425, 399, "CAN",
                  "isolate cold sources & platform regime", "isolate cold sources & platform regime")
    outcome_strip(sc, "b-cannot", 425, 467, 399, "CANNOT",
                  r"transfer $\hat{\delta}$ blindly across platforms", "transfer δ̂ blindly across platforms")

    # =========================================================================
    # PANEL (c): Fixed cohort & horizon cap  [x = 844 .. 1208]
    # =========================================================================
    sc.text("panel-c-header", "Panel c title", 844, 8, 364, 34,
            "(c) Fixed cohort & horizon cap", "(c) Fixed cohort & horizon cap",
            size_px=26.5, bold=True, color=C_INK, align="left")

    # Top strip: labels ABOVE boxes -> Intersection -> same 4 colored dots
    g_coh = "c-cohort"
    cohort_cols = ["#2B5C8F", "#C87A19", "#2D7A4D", "#7D3C98"]
    win_xs = [848.0, 944.0, 1040.0]
    win_labs = [r"$\tau_3$", r"$\tau_2$", r"$\tau_1$"]
    noise_pos = [
        [(56, 12), (58, 28)],
        [(14, 28), (58, 13)],
        [(56, 13), (14, 12)],
    ]
    core_pos = [(18, 20), (34, 12), (32, 29), (48, 24)]
    for wi, (wx, wlab) in enumerate(zip(win_xs, win_labs)):
        sc.text(g_coh, f"Win lab {wi}", wx, 42, 72, 20, wlab, wlab,
                size_px=20.0, bold=True, color=C_MUTED, align="center")
        sc.box(g_coh, f"Win {wi}", wx, 62, 72, 38, fill=C_GRAY_FILL_LIGHT, stroke=C_GRAY_STROKE, lw=1.2, radius=4.0)
        for ni, (nx, ny) in enumerate(noise_pos[wi]):
            sc.circle(g_coh, f"Noise {wi} {ni}", wx + nx, 62 + ny, 2.9, fill="#FFFFFF", stroke="#9AA5B1", lw=1.1)
        for ci, ((cx, cy), cc) in enumerate(zip(core_pos, cohort_cols)):
            sc.circle(g_coh, f"Core {wi} {ci}", wx + cx, 62 + cy, 3.6, fill=cc)
        if wi < 2:
            sc.text(g_coh, f"Cap {wi}", wx + 72, 66, 24, 28, r"$\cap$", "∩",
                    size_px=22.0, bold=True, color=C_INK, align="center")
    sc.arrow(g_coh, "To Cohort", 1115, 81, 1136, 81, color=C_INK, lw=1.5, head_len=6.5, head_w=5.0)
    sc.box(g_coh, "Cohort box", 1138, 62, 66, 38, fill=C_BLUE_FILL_SOFT, stroke=C_BLUE_STROKE, lw=1.5, radius=4.0)
    for ci, cc in enumerate(cohort_cols):
        cx = 1155.0 + (ci % 2) * 26.0
        cy = 72.0 + (ci // 2) * 16.0
        sc.circle(g_coh, f"Cohort dot {ci}", cx, cy, 4.0, fill=cc)
    sc.text(g_coh, "Fixed comp", 848, 104, 356, 22,
            "fixed composition (same entities)", "fixed composition (same entities)",
            size_px=20.0, italic=True, color=C_MUTED, align="center")

    # Schematic non-saturating plot
    g_plt = "c-plot"
    ax_x0, ax_x1 = 888.0, 1196.0
    ax_y0, ax_y1 = 368.0, 146.0
    sc.arrow(g_plt, "Y axis", ax_x0, ax_y0, ax_x0, ax_y1, color=C_INK, lw=1.5, head_len=7.5, head_w=5.5)
    sc.arrow(g_plt, "X axis", ax_x0, ax_y0, ax_x1, ax_y0, color=C_INK, lw=1.5, head_len=7.5, head_w=5.5)
    sc.text(g_plt, "Y label", 844, 140, 40, 24, r"$\Delta L$", r"$\Delta L$",
            size_px=21.5, bold=True, color=C_INK, align="right")
    sc.text(g_plt, "X label", 1140, 392, 64, 22, r"$\ln \tau$", "ln " + r"$\tau$",
            size_px=21.0, bold=True, color=C_INK, align="right")
    sc.text(g_plt, "Schematic tag", 896, 142, 92, 20, "schematic", "schematic",
            size_px=20.0, italic=True, color=C_MUTED, align="left")

    # Smooth concave-up curve y(u) = 356 - 190 * u**1.8
    x_start, x_end = 898.0, 1172.0
    y_base = 356.0
    rise_max = 190.0
    pwr = 1.8

    def c_xy(u: float) -> Tuple[float, float]:
        return x_start + u * (x_end - x_start), y_base - rise_max * (u ** pwr)

    curve_pts = [c_xy(i / 40.0) for i in range(41)]
    sc.polyline(g_plt, "Drift curve", curve_pts, stroke=C_INK, lw=2.3)

    u1, u2 = 0.44, 1.0
    x_h1, _ = c_xy(u1)
    x_h2, _ = c_xy(u2)
    u_half1 = u1 * (0.5 ** (1.0 / pwr))
    u_half2 = u2 * (0.5 ** (1.0 / pwr))
    xh1_half, yh1_half = c_xy(u_half1)
    xh2_half, yh2_half = c_xy(u_half2)

    sc.line(g_plt, "Cap H1 line", x_h1, 162, x_h1, ax_y0, stroke=C_AMBER_STROKE, lw=1.4, dash="dash")
    sc.line(g_plt, "Cap H2 line", x_h2, 162, x_h2, ax_y0, stroke=C_BLUE_STROKE, lw=1.4, dash="dash")
    sc.text(g_plt, "H1 top", x_h1 - 18, 136, 36, 22, r"$H_1$", r"$H_1$",
            size_px=20.5, bold=True, color=C_AMBER_TEXT, align="center")
    sc.text(g_plt, "H2 top", x_h2 - 18, 136, 36, 22, r"$H_2$", r"$H_2$",
            size_px=20.5, bold=True, color=C_BLUE_TEXT, align="center")

    sc.line(g_plt, "Half1 hline", ax_x0, yh1_half, xh1_half, yh1_half, stroke=C_AMBER_STROKE, lw=1.35, dash="dash")
    sc.arrow(g_plt, "Half1 drop", xh1_half, yh1_half, xh1_half, ax_y0, color=C_AMBER_STROKE, lw=1.6, head_len=7.0, head_w=5.0, dash="dash")
    sc.circle(g_plt, "Half1 dot", xh1_half, yh1_half, 4.0, fill=C_AMBER_STROKE)

    sc.line(g_plt, "Half2 hline", ax_x0, yh2_half, xh2_half, yh2_half, stroke=C_BLUE_STROKE, lw=1.35, dash="dash")
    sc.arrow(g_plt, "Half2 drop", xh2_half, yh2_half, xh2_half, ax_y0, color=C_BLUE_STROKE, lw=1.6, head_len=7.0, head_w=5.0, dash="dash")
    sc.circle(g_plt, "Half2 dot", xh2_half, yh2_half, 4.0, fill=C_BLUE_STROKE)

    sc.text(g_plt, "Half2 label", 894, yh2_half - 26, 92, 21,
            r"$\frac{1}{2}\Delta L(H_2)$", r"½$\Delta L$($H_2$)",
            size_px=20.0, color=C_BLUE_TEXT, align="left")
    sc.text(g_plt, "Half1 label", 894, yh1_half - 26, 92, 21,
            r"$\frac{1}{2}\Delta L(H_1)$", r"½$\Delta L$($H_1$)",
            size_px=20.0, color=C_AMBER_TEXT, align="left")

    sc.text(g_plt, "tau12 H1", xh1_half - 44, 372, 86, 21,
            r"$\tau_{1/2}(H_1)$", r"$\tau_{1/2}$($H_1$)",
            size_px=20.0, bold=True, color=C_AMBER_TEXT, align="center")
    sc.text(g_plt, "tau12 H2", xh2_half - 44, 372, 86, 21,
            r"$\tau_{1/2}(H_2)$", r"$\tau_{1/2}$($H_2$)",
            size_px=20.0, bold=True, color=C_BLUE_TEXT, align="center")

    # Horizontal shift bracket placed cleanly below the curve
    sc.arrow(g_plt, "Shift tau12", xh1_half + 4, 353, xh2_half - 4, 353,
             color=C_INK, lw=1.7, head_len=7.0, head_w=5.0)
    sc.text(g_plt, "Propto H", (xh1_half + xh2_half) / 2 - 42, 329, 84, 21,
            r"$\tau_{1/2} \propto H$", r"$\tau_{1/2}$ ∝ $H$",
            size_px=20.0, bold=True, color=C_INK, align="center")

    # Highlighted matched-span slope segment + completely unobstructed callout in open lower-right area
    xa, ya = c_xy(0.68)
    xb, yb = c_xy(0.90)
    sc.line(g_plt, "Matched secant", xa, ya, xb, yb, stroke=C_CAN_STROKE, lw=3.2)
    sc.circle(g_plt, "Secant pt a", xa, ya, 3.6, fill=C_CAN_STROKE)
    sc.circle(g_plt, "Secant pt b", xb, yb, 3.6, fill=C_CAN_STROKE)
    sc.text(g_plt, "Matched rate", 1096, 286, 114, 44,
            "matched rate\n" + r"($\mathrm{d}L / \mathrm{d}\ln \tau$)",
            "matched rate\n(dL / d ln " + r"$\tau$)",
            size_px=20.0, bold=True, color=C_CAN_TEXT, align="left")
    sc.arrow(g_plt, "Rate ptr", 1130, 282, 1118, 240, color=C_CAN_STROKE, lw=1.4, head_len=6.5, head_w=4.5)

    # Bottom boundary strips (Panel c)
    outcome_strip(sc, "c-can", 844, 425, 364, "CAN",
                  "compare matched-horizon rates", "compare matched-horizon rates")
    outcome_strip(sc, "c-cannot", 844, 467, 364, "CANNOT",
                  r"find horizon-invariant $\tau_{1/2}$", r"find horizon-invariant $\tau_{1/2}$")

    # =========================================================================
    # PANEL (d): Rolling cadence & base age  [x = 1228 .. 1590]
    # =========================================================================
    sc.text("panel-d-header", "Panel d title", 1228, 8, 362, 34,
            "(d) Rolling cadence & base age", "(d) Rolling cadence & base age",
            size_px=26.5, bold=True, color=C_INK, align="left")

    # Top training slice diagram explaining a = W/2
    g_wslice = "d-base-age"
    sc.box(g_wslice, "Train slice W", 1246, 51, 56, 24, fill=C_BLUE_FILL, stroke=C_BLUE_STROKE, lw=1.4, radius=3.0)
    sc.text(g_wslice, "W label", 1246, 51, 56, 24, r"$W$", r"$W$", size_px=20.0, bold=True, color=C_BLUE_TEXT, align="center")
    sc.line(g_wslice, "Mid tick", 1274, 75, 1274, 87, stroke=C_BLUE_STROKE, lw=1.4)
    sc.line(g_wslice, "End tick", 1302, 75, 1302, 87, stroke=C_INK, lw=1.4)
    sc.arrow(g_wslice, "W/2 brace", 1275, 82, 1301, 82, color=C_CORAL_STROKE, lw=1.4, head_len=5.0, head_w=4.0, double=True)
    sc.text(g_wslice, "Base age text", 1310, 51, 174, 24,
            r"base age $a = W/2$", r"base age $a=W/2$",
            size_px=20.0, bold=True, color=C_CORAL_TEXT, align="left")
    sc.text(g_wslice, "Schematic d", 1502, 51, 86, 22, "schematic", "schematic",
            size_px=20.0, italic=True, color=C_MUTED, align="right")

    # Center Serving Age Sawtooth
    g_saw = "d-sawtooth"
    dx0, dx1 = 1302.0, 1578.0
    dy0, dy1 = 324.0, 114.0
    y_base_a = 278.0
    y_top_p = 144.0
    x_p1 = 1425.0
    x_p2 = 1548.0

    sc.box(g_saw, "Naive band", dx0, y_base_a, dx1 - dx0 - 8, dy0 - y_base_a,
           fill="#F8EDEC", stroke=None, lw=0.0)
    sc.line(g_saw, "W/2 dash", dx0, y_base_a, x_p2 + 12, y_base_a, stroke=C_CORAL_STROKE, lw=1.4, dash="dash")
    sc.line(g_saw, "Top dash", dx0, y_top_p, x_p2 + 8, y_top_p, stroke=C_GRAY_STROKE, lw=1.1, dash="dot")

    sc.poly(g_saw, "Sawtooth fill 1", [(dx0, y_base_a), (x_p1, y_top_p), (x_p1, y_base_a)],
            fill=C_BLUE_FILL, stroke=None, lw=0.0, alpha=0.80)
    sc.poly(g_saw, "Sawtooth fill 2", [(x_p1, y_base_a), (x_p2, y_top_p), (x_p2, y_base_a)],
            fill=C_BLUE_FILL, stroke=None, lw=0.0, alpha=0.80)
    sc.polyline(g_saw, "Sawtooth curve",
                [(dx0, y_base_a), (x_p1, y_top_p), (x_p1, y_base_a), (x_p2, y_top_p), (x_p2, y_base_a)],
                stroke=C_BLUE_STROKE, lw=2.5)
    sc.circle(g_saw, "Retrain pt1", x_p1, y_base_a, 3.8, fill=C_BLUE_STROKE)
    sc.circle(g_saw, "Retrain pt2", x_p2, y_base_a, 3.8, fill=C_BLUE_STROKE)
    sc.text(g_saw, "Retrain lab", x_p1 - 74, 120, 68, 20, "retrain", "retrain",
            size_px=20.0, bold=True, color=C_BLUE_TEXT, align="right")
    sc.arrow(g_saw, "Retrain ptr", x_p1 - 5, 131, x_p1 - 1, 142, color=C_BLUE_STROKE, lw=1.3, head_len=5.5, head_w=4.0)

    sc.arrow(g_saw, "Y axis", dx0, dy0, dx0, dy1, color=C_INK, lw=1.5, head_len=7.5, head_w=5.5)
    sc.arrow(g_saw, "X axis", dx0, dy0, dx1, dy0, color=C_INK, lw=1.5, head_len=7.5, head_w=5.5)

    sc.text(g_saw, "Y label", 1228, 95, 108, 22, r"data age $\tau$", r"data age $\tau$",
            size_px=20.0, bold=True, color=C_INK, align="left")
    sc.text(g_saw, "Tick P+W/2", 1226, y_top_p - 11, 72, 22, r"$P{+}W/2$", r"$P+W/2$",
            size_px=20.0, color=C_BLUE_TEXT, align="right")
    sc.text(g_saw, "Tick W/2", 1226, y_base_a - 11, 72, 22, r"$a{=}W/2$", r"$a=W/2$",
            size_px=20.0, bold=True, color=C_CORAL_TEXT, align="right")
    sc.text(g_saw, "Tick 0y", 1270, dy0 - 11, 28, 22, "0", "0", size_px=20.0, color=C_MUTED, align="right")

    sc.line(g_saw, "Tick P line", x_p1, dy0 - 3, x_p1, dy0 + 4, stroke=C_INK, lw=1.3)
    sc.line(g_saw, "Tick 2P line", x_p2, dy0 - 3, x_p2, dy0 + 4, stroke=C_INK, lw=1.3)
    sc.text(g_saw, "Tick 0x", dx0 - 10, dy0 + 5, 22, 20, "0", "0", size_px=20.0, color=C_INK, align="center")
    sc.text(g_saw, "Tick Px", x_p1 - 16, dy0 + 5, 32, 20, r"$P$", r"$P$", size_px=20.0, bold=True, color=C_INK, align="center")
    sc.text(g_saw, "Tick 2Px", x_p2 - 18, dy0 + 5, 36, 20, r"$2P$", r"$2P$", size_px=20.0, bold=True, color=C_INK, align="center")
    sc.text(g_saw, "X label", 1368, 344, 150, 22, r"serving time $t$", r"serving time $t$",
            size_px=20.0, bold=True, color=C_INK, align="center")

    sc.arrow(g_saw, "Offset bracket", 1316, y_base_a + 2, 1316, dy0 - 2,
             color=C_CORAL_STROKE, lw=1.5, head_len=5.5, head_w=4.5, double=True)
    sc.text(g_saw, "Offset tag", 1322, 289, 102, 22,
            r"$a{=}W/2 > 0$", r"$a=W/2 > 0$",
            size_px=20.0, bold=True, color=C_CORAL_TEXT, align="left")
    sc.text(g_saw, "Naive tag", 1438, 290, 110, 22,
            r"naive $a{=}0$", r"naive $a=0$",
            size_px=20.0, italic=True, color=C_MUTED, align="center")

    # Equation pill below plot
    g_eq = "d-equation"
    sc.box(g_eq, "Eq pill", 1230, 369, 360, 48, fill=C_GRAY_FILL_LIGHT, stroke="#8798A5", lw=1.3, radius=5.0)
    sc.text(g_eq, "Eq line 1", 1234, 371, 352, 23,
            r"$\bar{L}(P) + \lambda C / P \to P^\ast$", r"L̄(P) + $\lambda C / P$ → $P^*$",
            size_px=20.5, bold=True, color=C_INK, align="center")
    sc.text(g_eq, "Eq line 2", 1234, 394, 352, 21,
            r"avoids $a{=}0$ collapse for $\delta \leq 0$", r"avoids $a=0$ collapse for $\delta$ ≤ 0",
            size_px=20.0, italic=True, color=C_BLUE_TEXT, align="center")

    # Bottom boundary strips (Panel d)
    outcome_strip(sc, "d-can", 1228, 425, 362, "CAN",
                  r"transfer slopes & recover $P^\ast$",
                  r"transfer slopes & recover $P^*$")
    outcome_strip(sc, "d-cannot", 1228, 467, 362, "LIMIT",
                  r"steep $\delta \gg 1$ overpredicts $P^\ast$",
                  r"steep $\delta$ ≫ 1 overpredicts $P^*$")

    return sc


# ---------------------------------------------------------------------------
# Backend 1: Matplotlib Vector PDF & PNG Preview
# ---------------------------------------------------------------------------
def _ls(dash: str) -> str:
    return {"solid": "-", "dash": "--", "dot": ":"}[dash]


def render_matplotlib(scene: Scene, out_pdf: Path, out_png: Path, check_overflow: bool = True) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [FONT_NAME],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
    })

    fig = plt.figure(figsize=(W_PX / 96.0, H_PX / 96.0), dpi=192, facecolor=C_BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=C_BG)
    ax.set_xlim(0.0, float(W_PX))
    ax.set_ylim(float(H_PX), 0.0)
    ax.axis("off")

    text_artists = []

    for obj in scene.items:
        if isinstance(obj, BoxObj):
            fc = "none" if obj.fill is None else obj.fill
            ec = "none" if (obj.stroke is None or obj.lw <= 0) else obj.stroke
            lw = obj.lw * PT_PER_PX
            if obj.radius > 0:
                p = FancyBboxPatch(
                    (obj.x, obj.y), obj.w, obj.h,
                    boxstyle=f"round,pad=0,rounding_size={obj.radius}",
                    facecolor=fc, edgecolor=ec, linewidth=lw,
                    linestyle=_ls(obj.dash), alpha=obj.alpha,
                    mutation_aspect=1.0,
                )
            else:
                p = FancyBboxPatch(
                    (obj.x, obj.y), obj.w, obj.h,
                    boxstyle="square,pad=0",
                    facecolor=fc, edgecolor=ec, linewidth=lw,
                    linestyle=_ls(obj.dash), alpha=obj.alpha,
                )
            ax.add_patch(p)

        elif isinstance(obj, EllipseObj):
            fc = "none" if obj.fill is None else obj.fill
            ec = "none" if (obj.stroke is None or obj.lw <= 0) else obj.stroke
            p = Circle((obj.cx, obj.cy), obj.rx, facecolor=fc, edgecolor=ec,
                       linewidth=obj.lw * PT_PER_PX, alpha=obj.alpha)
            ax.add_patch(p)

        elif isinstance(obj, PathObj):
            verts: List[Tuple[float, float]] = []
            codes: List[int] = []
            for cmd in obj.cmds:
                if cmd[0] == "M":
                    verts.append((cmd[1], cmd[2]))
                    codes.append(MplPath.MOVETO)
                elif cmd[0] == "L":
                    verts.append((cmd[1], cmd[2]))
                    codes.append(MplPath.LINETO)
                elif cmd[0] == "C":
                    verts.extend([(cmd[1], cmd[2]), (cmd[3], cmd[4]), (cmd[5], cmd[6])])
                    codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
                elif cmd[0] == "Z":
                    verts.append(verts[0])
                    codes.append(MplPath.CLOSEPOLY)
            path = MplPath(verts, codes)
            fc = "none" if obj.fill is None else obj.fill
            ec = "none" if (obj.stroke is None or obj.lw <= 0) else obj.stroke
            patch = PathPatch(
                path, facecolor=fc, edgecolor=ec,
                linewidth=obj.lw * PT_PER_PX, linestyle=_ls(obj.dash),
                alpha=obj.alpha, capstyle="round", joinstyle="round",
            )
            ax.add_patch(patch)

        elif isinstance(obj, TextObj):
            tx = obj.x if obj.align == "left" else (obj.x + obj.w / 2.0 if obj.align == "center" else obj.x + obj.w)
            ty = obj.y if obj.valign == "top" else (obj.y + obj.h / 2.0 if obj.valign == "center" else obj.y + obj.h)
            t = ax.text(
                tx, ty, obj.mpl_text,
                fontsize=obj.size_px * PT_PER_PX,
                fontweight="bold" if obj.bold else "normal",
                fontstyle="italic" if obj.italic else "normal",
                color=obj.color,
                ha=obj.align,
                va=obj.valign,
                linespacing=obj.line_spacing,
                fontfamily=FONT_NAME,
            )
            text_artists.append((obj, t))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_pdf), format="pdf", facecolor=C_BG)
    fig.savefig(str(out_png), format="png", dpi=192, facecolor=C_BG)

    if check_overflow:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        print("--- Text Box Overflow Audit (data px) ---")
        over = 0
        for obj, t in text_artists:
            bb = t.get_window_extent(renderer=renderer)
            p0 = inv.transform((bb.x0, bb.y0))
            p1 = inv.transform((bb.x1, bb.y1))
            tw = abs(p1[0] - p0[0])
            if tw > obj.w + 1.5:
                over += 1
                print(f"  [OVERFLOW W] {obj.name:22s}: text_w={tw:5.1f} > box_w={obj.w:5.1f}  text={obj.mpl_text!r}")
        if over == 0:
            print("  ALL text boxes fit cleanly inside bounding boxes (0 overflows).")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Backend 2: Pure-Python ECMA-376 OpenXML (.pptx) Emitter
# ---------------------------------------------------------------------------
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

for _pfx, _uri in [("p", NS_P), ("a", NS_A), ("r", NS_R)]:
    ET.register_namespace(_pfx, _uri)


def _hex(c: str) -> str:
    return c.lstrip("#").upper()


def _emu(px: float) -> int:
    return int(round(px * EMU_PER_PX))


def _add_fill(parent: ET.Element, color: Optional[str], alpha: float = 1.0) -> None:
    if color is None:
        ET.SubElement(parent, f"{{{NS_A}}}noFill")
    else:
        sf = ET.SubElement(parent, f"{{{NS_A}}}solidFill")
        clr = ET.SubElement(sf, f"{{{NS_A}}}srgbClr", {"val": _hex(color)})
        if alpha < 0.999:
            ET.SubElement(clr, f"{{{NS_A}}}alpha", {"val": str(int(round(alpha * 100000)))})


def _add_line(parent: ET.Element, color: Optional[str], lw_px: float, dash: str = "solid", alpha: float = 1.0) -> None:
    if color is None or lw_px <= 0:
        ln = ET.SubElement(parent, f"{{{NS_A}}}ln")
        ET.SubElement(ln, f"{{{NS_A}}}noFill")
        return
    ln = ET.SubElement(parent, f"{{{NS_A}}}ln", {
        "w": str(max(1270, int(round(lw_px * EMU_PER_PX)))),
        "cap": "rnd",
        "cmpd": "sng",
    })
    _add_fill(ln, color, alpha)
    dash_map = {"solid": "solid", "dash": "dash", "dot": "dot"}
    ET.SubElement(ln, f"{{{NS_A}}}prstDash", {"val": dash_map.get(dash, "solid")})
    ET.SubElement(ln, f"{{{NS_A}}}round")


def _build_slide_xml(scene: Scene) -> bytes:
    sld = ET.Element(f"{{{NS_P}}}sld")
    cSld = ET.SubElement(sld, f"{{{NS_P}}}cSld")

    bg = ET.SubElement(cSld, f"{{{NS_P}}}bg")
    bgPr = ET.SubElement(bg, f"{{{NS_P}}}bgPr")
    _add_fill(bgPr, C_BG)
    ET.SubElement(bgPr, f"{{{NS_A}}}effectLst")

    spTree = ET.SubElement(cSld, f"{{{NS_P}}}spTree")
    nvGrp = ET.SubElement(spTree, f"{{{NS_P}}}nvGrpSpPr")
    ET.SubElement(nvGrp, f"{{{NS_P}}}cNvPr", {"id": "1", "name": ""})
    ET.SubElement(nvGrp, f"{{{NS_P}}}cNvGrpSpPr")
    ET.SubElement(nvGrp, f"{{{NS_P}}}nvPr")
    grpPr = ET.SubElement(spTree, f"{{{NS_P}}}grpSpPr")
    xf = ET.SubElement(grpPr, f"{{{NS_A}}}xfrm")
    for tag in ["off", "ext", "chOff", "chExt"]:
        ET.SubElement(xf, f"{{{NS_A}}}{tag}", {"x": "0", "y": "0"} if "off" in tag.lower() else {"cx": "0", "cy": "0"})

    next_id = 2

    def new_sp(group: str, name: str, is_textbox: bool = False) -> Tuple[ET.Element, ET.Element]:
        nonlocal next_id
        sp = ET.SubElement(spTree, f"{{{NS_P}}}sp")
        nv = ET.SubElement(sp, f"{{{NS_P}}}nvSpPr")
        label = f"[g:{group}] {name}" if group else name
        ET.SubElement(nv, f"{{{NS_P}}}cNvPr", {"id": str(next_id), "name": label})
        next_id += 1
        cNvSp = ET.SubElement(nv, f"{{{NS_P}}}cNvSpPr")
        if is_textbox:
            cNvSp.set("txBox", "1")
        ET.SubElement(nv, f"{{{NS_P}}}nvPr")
        spPr = ET.SubElement(sp, f"{{{NS_P}}}spPr")
        return sp, spPr

    for obj in scene.items:
        if isinstance(obj, BoxObj):
            _, spPr = new_sp(obj.group, obj.name, is_textbox=False)
            xfrm = ET.SubElement(spPr, f"{{{NS_A}}}xfrm")
            ET.SubElement(xfrm, f"{{{NS_A}}}off", {"x": str(_emu(obj.x)), "y": str(_emu(obj.y))})
            ET.SubElement(xfrm, f"{{{NS_A}}}ext", {"cx": str(max(1, _emu(obj.w))), "cy": str(max(1, _emu(obj.h)))})
            if obj.radius > 0:
                prst = ET.SubElement(spPr, f"{{{NS_A}}}prstGeom", {"prst": "roundRect"})
                av = ET.SubElement(prst, f"{{{NS_A}}}avLst")
                adj = min(50000, max(0, int(round(obj.radius / max(1e-3, min(obj.w, obj.h)) * 100000))))
                ET.SubElement(av, f"{{{NS_A}}}gd", {"name": "adj", "fmla": f"val {adj}"})
            else:
                prst = ET.SubElement(spPr, f"{{{NS_A}}}prstGeom", {"prst": "rect"})
                ET.SubElement(prst, f"{{{NS_A}}}avLst")
            _add_fill(spPr, obj.fill, obj.alpha)
            _add_line(spPr, obj.stroke, obj.lw, obj.dash, obj.alpha)

        elif isinstance(obj, EllipseObj):
            _, spPr = new_sp(obj.group, obj.name, is_textbox=False)
            xfrm = ET.SubElement(spPr, f"{{{NS_A}}}xfrm")
            ET.SubElement(xfrm, f"{{{NS_A}}}off", {"x": str(_emu(obj.cx - obj.rx)), "y": str(_emu(obj.cy - obj.ry))})
            ET.SubElement(xfrm, f"{{{NS_A}}}ext", {"cx": str(max(1, _emu(2 * obj.rx))), "cy": str(max(1, _emu(2 * obj.ry)))})
            prst = ET.SubElement(spPr, f"{{{NS_A}}}prstGeom", {"prst": "ellipse"})
            ET.SubElement(prst, f"{{{NS_A}}}avLst")
            _add_fill(spPr, obj.fill, obj.alpha)
            _add_line(spPr, obj.stroke, obj.lw, "solid", obj.alpha)

        elif isinstance(obj, PathObj):
            # Use native preset 'line' for exact 2-point straight segments (cross-viewer compatible)
            if len(obj.cmds) == 2 and obj.cmds[0][0] == "M" and obj.cmds[1][0] == "L":
                x1, y1 = obj.cmds[0][1], obj.cmds[0][2]
                x2, y2 = obj.cmds[1][1], obj.cmds[1][2]
                x0, y0 = min(x1, x2), min(y1, y2)
                w, h = abs(x2 - x1), abs(y2 - y1)
                _, spPr = new_sp(obj.group, obj.name, is_textbox=False)
                xfrm_attrs = {}
                if (x2 - x1) * (y2 - y1) < 0:
                    xfrm_attrs["flipV"] = "1"
                xfrm = ET.SubElement(spPr, f"{{{NS_A}}}xfrm", xfrm_attrs)
                ET.SubElement(xfrm, f"{{{NS_A}}}off", {"x": str(_emu(x0)), "y": str(_emu(y0))})
                ET.SubElement(xfrm, f"{{{NS_A}}}ext", {"cx": str(_emu(w)), "cy": str(_emu(h))})
                prst = ET.SubElement(spPr, f"{{{NS_A}}}prstGeom", {"prst": "line"})
                ET.SubElement(prst, f"{{{NS_A}}}avLst")
                _add_fill(spPr, None)
                _add_line(spPr, obj.stroke, obj.lw, obj.dash, obj.alpha)
                continue

            xs: List[float] = []
            ys: List[float] = []
            for cmd in obj.cmds:
                if cmd[0] in ("M", "L"):
                    xs.append(cmd[1]); ys.append(cmd[2])
                elif cmd[0] == "C":
                    xs.extend([cmd[1], cmd[3], cmd[5]])
                    ys.extend([cmd[2], cmd[4], cmd[6]])
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            if x1 - x0 < 0.5:
                cx_mid = 0.5 * (x0 + x1)
                x0, x1 = cx_mid - 0.5, cx_mid + 0.5
            if y1 - y0 < 0.5:
                cy_mid = 0.5 * (y0 + y1)
                y0, y1 = cy_mid - 0.5, cy_mid + 0.5
            w = x1 - x0
            h = y1 - y0

            _, spPr = new_sp(obj.group, obj.name, is_textbox=False)
            cx_emu = max(1, _emu(w))
            cy_emu = max(1, _emu(h))
            xfrm = ET.SubElement(spPr, f"{{{NS_A}}}xfrm")
            ET.SubElement(xfrm, f"{{{NS_A}}}off", {"x": str(_emu(x0)), "y": str(_emu(y0))})
            ET.SubElement(xfrm, f"{{{NS_A}}}ext", {"cx": str(cx_emu), "cy": str(cy_emu)})

            cg = ET.SubElement(spPr, f"{{{NS_A}}}custGeom")
            ET.SubElement(cg, f"{{{NS_A}}}avLst")
            ET.SubElement(cg, f"{{{NS_A}}}gdLst")
            ET.SubElement(cg, f"{{{NS_A}}}ahLst")
            ET.SubElement(cg, f"{{{NS_A}}}cxnLst")
            pl = ET.SubElement(cg, f"{{{NS_A}}}pathLst")
            ap = ET.SubElement(pl, f"{{{NS_A}}}path", {"w": str(cx_emu), "h": str(cy_emu)})

            def gx(vx: float) -> str:
                return str(int(round((vx - x0) / w * cx_emu)))

            def gy(vy: float) -> str:
                return str(int(round((vy - y0) / h * cy_emu)))

            for cmd in obj.cmds:
                if cmd[0] == "M":
                    el = ET.SubElement(ap, f"{{{NS_A}}}moveTo")
                    ET.SubElement(el, f"{{{NS_A}}}pt", {"x": gx(cmd[1]), "y": gy(cmd[2])})
                elif cmd[0] == "L":
                    el = ET.SubElement(ap, f"{{{NS_A}}}lnTo")
                    ET.SubElement(el, f"{{{NS_A}}}pt", {"x": gx(cmd[1]), "y": gy(cmd[2])})
                elif cmd[0] == "C":
                    el = ET.SubElement(ap, f"{{{NS_A}}}cubicBezTo")
                    ET.SubElement(el, f"{{{NS_A}}}pt", {"x": gx(cmd[1]), "y": gy(cmd[2])})
                    ET.SubElement(el, f"{{{NS_A}}}pt", {"x": gx(cmd[3]), "y": gy(cmd[4])})
                    ET.SubElement(el, f"{{{NS_A}}}pt", {"x": gx(cmd[5]), "y": gy(cmd[6])})
                elif cmd[0] == "Z":
                    ET.SubElement(ap, f"{{{NS_A}}}close")

            _add_fill(spPr, obj.fill, obj.alpha)
            _add_line(spPr, obj.stroke, obj.lw, obj.dash, obj.alpha)

        elif isinstance(obj, TextObj):
            sp, spPr = new_sp(obj.group, obj.name, is_textbox=True)
            xfrm = ET.SubElement(spPr, f"{{{NS_A}}}xfrm")
            ET.SubElement(xfrm, f"{{{NS_A}}}off", {"x": str(_emu(obj.x)), "y": str(_emu(obj.y))})
            ET.SubElement(xfrm, f"{{{NS_A}}}ext", {"cx": str(max(1, _emu(obj.w))), "cy": str(max(1, _emu(obj.h)))})
            prst = ET.SubElement(spPr, f"{{{NS_A}}}prstGeom", {"prst": "rect"})
            ET.SubElement(prst, f"{{{NS_A}}}avLst")
            ET.SubElement(spPr, f"{{{NS_A}}}noFill")
            ln = ET.SubElement(spPr, f"{{{NS_A}}}ln")
            ET.SubElement(ln, f"{{{NS_A}}}noFill")

            txBody = ET.SubElement(sp, f"{{{NS_P}}}txBody")
            valign_map = {"top": "t", "center": "ctr", "bottom": "b"}
            bodyPr = ET.SubElement(txBody, f"{{{NS_A}}}bodyPr", {
                "wrap": "none",
                "lIns": "0", "tIns": "0", "rIns": "0", "bIns": "0",
                "anchor": valign_map.get(obj.valign, "ctr"),
            })
            ET.SubElement(bodyPr, f"{{{NS_A}}}noAutofit")
            ET.SubElement(txBody, f"{{{NS_A}}}lstStyle")

            txt = obj.pptx_text if obj.pptx_text is not None else obj.mpl_text
            align_map = {"left": "l", "center": "ctr", "right": "r"}
            sz_cp = str(int(round(obj.size_px * PT_PER_PX * 100)))
            for line in txt.split("\n"):
                p = ET.SubElement(txBody, f"{{{NS_A}}}p")
                pPr = ET.SubElement(p, f"{{{NS_A}}}pPr", {"algn": align_map.get(obj.align, "l")})
                lnSpc = ET.SubElement(pPr, f"{{{NS_A}}}lnSpc")
                ET.SubElement(lnSpc, f"{{{NS_A}}}spcPct", {"val": str(int(round(obj.line_spacing * 100000)))})
                r = ET.SubElement(p, f"{{{NS_A}}}r")
                rPr = ET.SubElement(r, f"{{{NS_A}}}rPr", {
                    "lang": "en-US",
                    "sz": sz_cp,
                    "b": "1" if obj.bold else "0",
                    "i": "1" if obj.italic else "0",
                    "dirty": "0",
                })
                _add_fill(rPr, obj.color)
                for tag in ["latin", "ea", "cs"]:
                    ET.SubElement(rPr, f"{{{NS_A}}}{tag}", {"typeface": FONT_NAME})
                t_el = ET.SubElement(r, f"{{{NS_A}}}t")
                t_el.text = line
                if line[:1].isspace() or line[-1:].isspace():
                    t_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    clrMapOvr = ET.SubElement(sld, f"{{{NS_P}}}clrMapOvr")
    ET.SubElement(clrMapOvr, f"{{{NS_A}}}masterClrMapping")
    return ET.tostring(sld, encoding="utf-8", xml_declaration=True)


def write_pptx(scene: Scene, out_pptx: Path) -> None:
    cx = _emu(W_PX)
    cy = _emu(H_PX)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>
  <Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>
  <Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>
</Types>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"""

    pres_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                saveSubsetFonts="1">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
  <p:sldSz cx="{cx}" cy="{cy}" type="custom"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle>
    <a:defPPr><a:defRPr lang="en-US"/></a:defPPr>
  </p:defaultTextStyle>
</p:presentation>"""

    pres_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>
  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>
  <Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>
  <Relationship Id="rId6" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
</Relationships>"""

    pres_props = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presProps xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>"""

    view_props = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:viewProps xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>"""

    table_styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>"""

    theme_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="PaperTheme">
  <a:themeElements>
    <a:clrScheme name="Paper">
      <a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>
      <a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1E2933"/></a:dk2>
      <a:lt2><a:srgbClr val="F5F7F9"/></a:lt2>
      <a:accent1><a:srgbClr val="2B5C8F"/></a:accent1>
      <a:accent2><a:srgbClr val="B03A2E"/></a:accent2>
      <a:accent3><a:srgbClr val="C87A19"/></a:accent3>
      <a:accent4><a:srgbClr val="2D7A4D"/></a:accent4>
      <a:accent5><a:srgbClr val="6C757D"/></a:accent5>
      <a:accent6><a:srgbClr val="B78313"/></a:accent6>
      <a:hlink><a:srgbClr val="2B5C8F"/></a:hlink>
      <a:folHlink><a:srgbClr val="6C757D"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="PaperFonts">
      <a:majorFont><a:latin typeface="{FONT_NAME}"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>
      <a:minorFont><a:latin typeface="{FONT_NAME}"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="PaperFmt">
      <a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
      <a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="28575"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
</a:theme>"""

    master_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>"""

    master_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""

    layout_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""

    layout_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""

    slide_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""

    slide_xml = _build_slide_xml(scene)

    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_pptx, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types.encode("utf-8"))
        z.writestr("_rels/.rels", root_rels.encode("utf-8"))
        z.writestr("ppt/presentation.xml", pres_xml.encode("utf-8"))
        z.writestr("ppt/_rels/presentation.xml.rels", pres_rels.encode("utf-8"))
        z.writestr("ppt/presProps.xml", pres_props.encode("utf-8"))
        z.writestr("ppt/viewProps.xml", view_props.encode("utf-8"))
        z.writestr("ppt/tableStyles.xml", table_styles.encode("utf-8"))
        z.writestr("ppt/theme/theme1.xml", theme_xml.encode("utf-8"))
        z.writestr("ppt/slideMasters/slideMaster1.xml", master_xml.encode("utf-8"))
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", master_rels.encode("utf-8"))
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout_xml.encode("utf-8"))
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels.encode("utf-8"))
        z.writestr("ppt/slides/slide1.xml", slide_xml)
        z.writestr("ppt/slides/_rels/slide1.xml.rels", slide_rels.encode("utf-8"))


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Figure 1 (overview diagram).")
    parser.add_argument(
        "--pptx",
        action="store_true",
        help="Also write editable ECMA-376 OpenXML presentation to assets/overview.pptx",
    )
    args = parser.parse_args()

    out_pdf = ROOT / "paper" / "figs" / "fig0_overview.pdf"
    out_png = ROOT / "assets" / "overview.png"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    scene = build_scene()
    render_matplotlib(scene, out_pdf, out_png, check_overflow=True)
    written = [str(out_pdf), str(out_png)]
    if args.pptx:
        out_pptx = ROOT / "assets" / "overview.pptx"
        write_pptx(scene, out_pptx)
        written.append(str(out_pptx))
    print("Wrote:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
