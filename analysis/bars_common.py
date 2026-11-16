"""Shared two-row bar layout of the attention-layer and end-to-end figures. Each length or resolution group is its own
axis with its own scale, so the tallest bar of every group reaches the top; titles and speedup labels sit inside the bar
area; per row, one legend strip above the panels: systems at the left, bar segments in the middle.
"""
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

HATCH = {"Ring": "..", "All-gather": "xx", "Sparse all-gather": "--", "DCP": "oo", "UA": "++", "Magi": "||"}   # baselines only
HEAD = 1.22                       # headroom: the tallest bar of a group ends below the title band


def draw(fig, rows, data, order, title, col, ours, segments, unit, missing_tag, row_legends, gap_in=0.06, speedups=None, label_shift=None, inside=None):
    """rows: [[W, ...], ...]; data[W] = [(label, {system: values})]; row_legends[r] = (system list, segment handles).
    Returns {f"{W} {label}": (speedup, best baseline)}."""
    summary = {}
    nr = len(rows)
    gs = fig.add_gridspec(nr, 1, hspace=0.42, top=0.90, bottom=0.07, left=0.005, right=0.995)
    LE = dict(edgecolor="black", linewidth=0.5)
    for r, row in enumerate(rows):
        # one axis per (workload, group); width = number of bars; a small extra gap between workloads
        cols, widths = [], []
        for wi, W in enumerate(row):
            for gi, (lab, d) in enumerate(data[W]):
                if wi > 0 and gi == 0:
                    cols.append(None); widths.append(0.7)          # extra spacer between workloads
                cols.append((W, gi)); widths.append(len(order[W]))
        if not cols:                                          # no group of this row has a result yet
            continue
        sg = gs[r].subgridspec(1, len(cols), width_ratios=widths, wspace=0.28)
        row_axes = []; titles = []
        for ci, c in enumerate(cols):
            if c is None:
                continue
            W, gi = c; lab, d = data[W][gi]; sysl = order[W]
            ax = fig.add_subplot(sg[0, ci]); row_axes.append(ax)
            tot = {s: sum(d[s]) for s in d}
            ymax = max(tot.values())
            best = min(((tot[s], s) for s in d if s not in ours), default=(float("nan"), ""))
            sp = best[0] / tot["Dex"]
            if not best[1]:                                   # no baseline in this group yet: bars only, no speedup
                pass
            elif speedups is not None:                        # per-case geometric means supplied by the caller
                sp, bname = speedups[(W, lab)]["Dex"]
                summary[f"{W} {lab}"] = (round(sp, 2), bname) if "Dex w/o assigner" not in speedups[(W, lab)] else \
                    (round(sp, 2), bname, round(speedups[(W, lab)]["Dex w/o assigner"][0], 2))
            else:
                summary[f"{W} {lab}"] = (round(sp, 2), best[1])
            lab_y = {}; bar_top = {}; edges = []
            for i, s in enumerate(sysl):
                if s not in d:
                    ax.text(i, ymax * 0.02, missing_tag(W, s, lab), rotation=90, ha="center", va="bottom", fontsize=8.5,
                            color="#444444")
                    continue
                y = 0.0
                for val, sty in segments(s, d[s]):
                    if val > 0:
                        st2 = dict(sty)
                        if s in HATCH and "hatch" not in st2 and st2.get("color", "").upper() not in ("WHITE", "#FFFFFF"):
                            st2["hatch"] = HATCH[s]
                        ax.bar(i, val, bottom=y, width=1.0, **st2, **LE); y += val
                        edges.append((i - 0.5, i + 0.5, y))
                if s in ours and best[1]:                     # speedup labels above Dex's (and Dex w/o assigner's) bar
                    sps = speedups[(W, lab)][s][0] if speedups is not None else best[0] / tot[s]
                    lab_y[s] = (i, min(y + ymax * 0.02, ymax * 0.72), sps)
                    bar_top[s] = y
            ax.set_xlim(-0.5, len(sysl) - 0.5); ax.set_ylim(0, ymax * HEAD)
            # place the labels: the larger speedup sits lower (its bar is shorter); no label may cover a bar's
            # horizontal edge (segment boundary or bar top) -> lift it just above the highest edge it would cross
            ins = {s_ for s_ in lab_y if (W, lab, s_) in (inside or set())}
            dy = (label_shift or {}).get((W, lab), 0.0) * ymax   # optional manual nudge (fraction of the group's tallest bar)
            rend = fig.canvas.get_renderer(); inv = ax.transData.inverted(); prev = None
            for s, (i, yl, sps) in sorted(lab_y.items(), key=lambda kv: -kv[1][2]):
                t = ax.text(i - 0.45, 0, f"{sps:.2f}×", ha="left", fontsize=9 if s == "Dex" else 8.5,
                            fontweight="bold", color=col[s], clip_on=False, zorder=7,
                            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
                if s in ins:                                  # inside the bar, just under its top edge
                    t.set_va("top"); t.set_y(bar_top[s] - ymax * 0.03); continue
                t.set_va("bottom"); yl += dy
                if prev is not None and yl < prev + ymax * 0.12:
                    yl = prev + ymax * 0.12                   # stagger above the lower (larger-speedup) label
                for _ in range(12):
                    t.set_y(yl)
                    (x0, y0), (x1, y1) = inv.transform(t.get_window_extent(rend).get_points())
                    pad = ymax * 0.012
                    hit = [e for e in edges if e[0] < x1 and e[1] > x0 and y0 - pad < e[2] < y1 + pad]
                    if not hit: break
                    yl = max(e[2] for e in hit) + pad + (yl - y0)
                prev = yl
            ax.set_xticks([(len(sysl) - 1) / 2])
            tk = [t for t in MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(0, ymax) if 0 <= t <= ymax]
            ax.set_yticks(tk); ax.set_yticklabels([f"{t:g}" for t in tk], fontsize=7)   # below the title band; unit in title
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.5); ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=2, pad=1, labelsize=7); ax.tick_params(axis="x", length=0, pad=2)
            ax.spines[["top", "right"]].set_visible(False)
            # x label: the group name
            ax.set_xticklabels([lab])
            if gi == 0:
                titles.append((ax, f"{title[W]} ({unit})"))
        # legend strip above the row: systems at the left, segments in the middle
        fig.canvas.draw()
        for ax, t in titles:                                  # workload title, figure level so later axes cannot cover it
            pp = ax.get_position()
            fig.text(pp.x0 + 0.004, pp.y1 - 0.004, t, ha="left", va="top", fontsize=9.5, fontweight="bold", zorder=20,
                     bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.9))
        y1 = max(a.get_position().y1 for a in row_axes); x0 = min(a.get_position().x0 for a in row_axes)
        sysl, seg_h = row_legends[r]
        LK = dict(frameon=False, fontsize=9, handlelength=1.1, handleheight=0.8, handletextpad=0.35, columnspacing=0.8,
                  borderaxespad=0.0, borderpad=0.0)
        fig.legend(handles=[Patch(facecolor=col[s], hatch=HATCH.get(s), label=s, **LE) for s in sysl], loc="lower left",
                   bbox_to_anchor=(x0, y1 + 0.012), bbox_transform=fig.transFigure, ncol=len(sysl), **LK)
        fig.legend(handles=seg_h, loc="lower left", bbox_to_anchor=(0.56, y1 + 0.012), bbox_transform=fig.transFigure,
                   ncol=len(seg_h), **LK)
    return summary
