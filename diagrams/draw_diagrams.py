"""Draw the QUBO selection diagram and the multi-task network diagram at the journal's size:
16 cm wide, Times New Roman 10 pt, black lines, no background colour, no title inside the figure.

    python draw_diagrams.py OUTPUT_FOLDER

Every label is checked against the box it sits in; a label that runs over an edge is reported.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

CM = 1 / 2.54
FILL = "#e6e6e6"  # light grey for boxes that hold data or settings, white for the steps
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


class Diagram:
    def __init__(self, width_cm, height_cm):
        self.fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width_cm)
        self.ax.set_ylim(0, height_cm)
        self.ax.axis("off")
        self.width, self.height = width_cm, height_cm
        self.fits = []  # (text, x0, y0, x1, y1) the text must stay inside

    def box(self, x, y, w, h, text, fill="white", heading=None):
        """Rounded box with its lower-left corner at (x, y) cm. An optional bold heading sits at the top."""
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                         facecolor=fill, edgecolor="black", linewidth=1.0))
        top = y + h
        if heading:
            t = self.ax.text(x + w / 2, top - 0.12, heading, ha="center", va="top", fontweight="bold")
            self.fits.append((t, x, y, x + w, top))
            top -= 0.5
        if text:
            t = self.ax.text(x + w / 2, (y + top) / 2, text, ha="center", va="center", linespacing=1.2)
            self.fits.append((t, x, y, x + w, top))
        return {"left": (x, y + h / 2), "right": (x + w, y + h / 2), "top": (x + w / 2, y + h),
                "bottom": (x + w / 2, y), "x": x, "y": y, "w": w, "h": h}

    def label(self, x, y, text, **kw):
        t = self.ax.text(x, y, text, ha="center", va="center", linespacing=1.2, **kw)
        self.fits.append((t, 0, 0, self.width, self.height))
        return t

    def arrow(self, start, end, dashed=False):
        self.ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color="black",
                                          linewidth=1.0, linestyle=(0, (3, 2)) if dashed else "-",
                                          shrinkA=0, shrinkB=0))

    def elbow(self, points):
        xs, ys = zip(*points)
        self.ax.plot(xs[:-1], ys[:-1], color="black", linewidth=1.0)
        self.arrow(points[-2], points[-1])

    def save(self, stem):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        to_cm = self.ax.transData.inverted()
        for text, x0, y0, x1, y1 in self.fits:
            (a, b), (c, d) = to_cm.transform(text.get_window_extent(renderer).get_points())
            margin = 0.06
            if a < x0 + margin or c > x1 - margin or b < y0 + margin or d > y1 - margin:
                print(f"  does not fit in {stem.name}: {text.get_text()[:50]!r}")
        self.fig.savefig(stem.with_suffix(".pdf"))
        self.fig.savefig(stem.with_suffix(".png"))
        plt.close(self.fig)


def qubo_figure(out):
    g = Diagram(16, 7.4)
    data = g.box(0.1, 2.95, 1.75, 1.5, "Training\nsplit of\nLAT or IC", fill=FILL)
    attrs = g.box(2.35, 4.25, 3.25, 1.9, "Encoded attributes\n37 in LAT\n11 in IC", fill=FILL)
    joint = g.box(2.35, 1.25, 3.25, 1.9, "Joint label\n$g \\times N_{\\mathrm{age}} + a$\n(IC: age group only)",
                  fill=FILL)
    mid = data["right"][1]
    g.ax.plot([data["right"][0], 2.1], [mid, mid], color="black", linewidth=1.0)
    g.elbow([(2.1, mid), (2.1, attrs["left"][1]), attrs["left"]])
    g.elbow([(2.1, mid), (2.1, joint["left"][1]), joint["left"]])

    nmi = g.box(6.0, 1.25, 2.85, 4.9,
                "Relevance $R_i$\nof attribute $i$\nto the label\n\nRedundancy $Q_{ij}$\nbetween\nattributes $i$ and $j$",
                heading="Normalized MI")
    g.arrow(attrs["right"], (nmi["x"], attrs["right"][1]))
    g.arrow(joint["right"], (nmi["x"], joint["right"][1]))

    weights = g.box(9.6, 6.05, 6.3, 1.3,
                    "$\\alpha = 1$, $\\beta = 0.05$, $\\lambda = 1.5$\n"
                    "$k = 25$ (LAT); $k \\in \\{4, 6, 8, 10, 11\\}$ (IC)", fill=FILL)
    energy = g.box(9.6, 4.1, 6.3, 1.5,
                   "$E(\\boldsymbol{x}) = -\\alpha\\sum_i R_i x_i + \\beta\\sum_{i<j} Q_{ij} x_i x_j"
                   " + \\lambda(\\sum_i x_i - k)^2$", heading="QUBO energy")
    g.arrow(weights["bottom"], energy["top"])
    g.arrow((nmi["x"] + nmi["w"], 5.15), (energy["x"], 5.15))
    g.label(9.2, 5.4, "$R_i$")
    g.arrow((nmi["x"] + nmi["w"], 4.55), (energy["x"], 4.55))
    g.label(9.2, 4.25, "$Q_{ij}$")

    anneal = g.box(9.6, 2.1, 6.3, 1.5,
                   "6 restarts, $2 \\times 10^4$ bit flips each,\ncooling from $T = 5$ to $T = 0.01$",
                   heading="Simulated annealing on a CPU")
    g.arrow(energy["bottom"], anneal["top"])
    chosen = g.box(9.6, 0.1, 6.3, 1.5, "LAT: $k = 25$\nIC: $k$ chosen by validation macro-F1",
                   fill=FILL, heading="Selected attributes")
    g.arrow(anneal["bottom"], chosen["top"])
    g.save(out / "fig_qubo_selection")


def network_figure(out):
    g = Diagram(16, 10.2)
    # settings, top left
    g.box(0.1, 7.0, 8.0, 3.1,
          "$\\mathcal{L} = 2\\,\\mathcal{L}_{\\mathrm{gender}} + \\mathcal{L}_{\\mathrm{age}}$;"
          " in IC $\\mathcal{L} = \\mathcal{L}_{\\mathrm{age}}$ (no gender label)\n"
          "class weights $(1/n_c)^{0.5}$ (softened inverse frequency)\n"
          "AdamW, learning rate $10^{-3} \\rightarrow 10^{-5}$ (cosine)\n"
          "weight decay $5 \\times 10^{-4}$, batch 256, 150 epochs\n"
          "best validation weights kept\n"
          "$C \\in \\{3, 6, 10, 48\\}$ (LAT), $C = 5$ (IC)", fill=FILL, heading="Joint objective and training")

    # shared trunk
    g.ax.add_patch(FancyBboxPatch((2.45, 1.2), 5.65, 4.85, boxstyle="round,pad=0,rounding_size=0.15",
                                  facecolor="none", edgecolor="black", linewidth=1.0, linestyle=(0, (4, 2))))
    g.label(5.275, 5.7, "Shared trunk (both tasks)", fontstyle="italic")
    inp = g.box(0.1, 2.15, 2.15, 2.9, "Input $\\boldsymbol{x}$\n$d$ selected\nattributes\n\n$d = 25$ (LAT)\n"
                                      "$d = 8$–$11$ (IC)", fill=FILL)
    fc1 = g.box(2.75, 1.55, 2.4, 3.8, "$\\boldsymbol{W}_1$, $\\boldsymbol{b}_1$\n$d \\times 64$\n\nReLU\n"
                                      "dropout 0.3\n\n64 units", heading="Layer 1")
    fc2 = g.box(5.45, 1.55, 2.4, 3.8, "$\\boldsymbol{W}_2$, $\\boldsymbol{b}_2$\n$64 \\times 32$\n\nReLU\n"
                                      "dropout 0.3\n\n$\\boldsymbol{h}$: 32 units", heading="Layer 2")
    g.arrow(inp["right"], (fc1["x"], inp["right"][1]))
    g.arrow((fc1["x"] + fc1["w"], inp["right"][1]), (fc2["x"], inp["right"][1]))

    # gender head, top right
    head = g.box(8.95, 7.4, 2.2, 2.3, "$\\boldsymbol{W}_g$, $b_g$\n$32 \\times 1$\nsigmoid", heading="Gender head")
    prob = g.box(11.55, 7.85, 1.6, 1.4, "$\\hat{p}_{\\mathrm{male}}$\n$\\in [0, 1]$", fill=FILL)
    lg = g.box(13.55, 7.4, 2.35, 2.3, "$\\mathcal{L}_{\\mathrm{gender}}$\nweighted\nbinary\ncross-entropy")
    g.arrow(head["right"], prob["left"])
    g.arrow(prob["right"], lg["left"])

    # age head, bottom right
    cat = g.box(8.95, 0.1, 2.35, 2.6, "$\\boldsymbol{h}$ and\ndetach($\\hat{p}_{\\mathrm{male}}$)\n33 units",
                heading="Concatenate")
    a1 = g.box(11.55, 0.1, 2.0, 2.6, "$\\boldsymbol{W}_{a1}$, $\\boldsymbol{b}_{a1}$\n$33 \\times 16$\nReLU\n16 units",
               heading="Age layer")
    a2 = g.box(13.8, 0.1, 2.1, 2.6, "$\\boldsymbol{W}_{a2}$, $\\boldsymbol{b}_{a2}$\n$16 \\times C$\nsoftmax\n$C$ groups",
               heading="Age output")
    la = g.box(13.8, 3.35, 2.1, 1.5, "$\\mathcal{L}_{\\mathrm{age}}$\nweighted\ncross-entropy")
    g.arrow(cat["right"], a1["left"])
    g.arrow(a1["right"], a2["left"])
    g.arrow(a2["top"], la["bottom"])

    # trunk output to both heads
    bus = 8.5
    y = inp["right"][1]
    g.ax.plot([fc2["x"] + fc2["w"], bus], [y, y], color="black", linewidth=1.0)
    g.elbow([(bus, y), (bus, head["left"][1]), head["left"]])
    g.elbow([(bus, y), (bus, cat["left"][1]), cat["left"]])
    # detached gender probability into the age head
    g.arrow(prob["bottom"], (cat["x"] + cat["w"] - 0.3, cat["y"] + cat["h"]), dashed=True)
    g.label(9.85, 4.75, "detach:\ngender informs\nage, but no\ngradient flows\nback to it",
            fontstyle="italic")
    g.save(out / "fig_mtl_network")


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    qubo_figure(folder)
    network_figure(folder)
