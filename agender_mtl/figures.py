"""
Step F: every figure and table of the article, drawn from the files Steps 1-3 wrote.

Nothing is typed in. The only training here is one repeat of the network (seed 42) for the 6- and
10-class schemes, because confusion matrices, ROC curves and permutation importance need the test
predictions; the split, attributes and training are those of Step 1, so the model is the same one.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from agender_mtl import benchmark, metrics, qubo, step2
from agender_mtl.mtl import MultiTaskNet

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
GREY = "#52514e"
MODELS = ["Logistic Regression", "Random Forest", "Gradient Boosting", "MTL"]
SHOWN = {"MTL": "MTL (proposed)"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": "#0b0b0b", "xtick.color": GREY,
                     "ytick.color": GREY, "axes.grid": False, "savefig.dpi": 300, "figure.dpi": 100})


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def escape(text):
    return (str(text).replace("\\", r"\textbackslash{}").replace("%", r"\%").replace("&", r"\&")
            .replace("_", r"\_").replace("#", r"\#").replace("<", "$<$").replace(">", "$>$").replace("+-", r"$\pm$"))


def latex_table(frame, caption, label, bold_best=None, lower_better=(), wide=False, note=None):
    """frame: first column = row names, other columns = numbers (float) or ready strings."""
    env = "table*" if wide else "table"
    columns = list(frame.columns)
    lines = [rf"\begin{{{env}}}[h]", r"\centering", rf"\caption{{{caption}}}\label{{{label}}}",
             r"\small\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l" + "c" * (len(columns) - 1) + "}", r"\toprule",
             " & ".join(escape(c) for c in columns) + r"\\", r"\midrule"]
    best = {}
    for c in columns[1:]:
        values = pd.to_numeric(frame[c].map(lambda v: v[0] if isinstance(v, tuple) else v), errors="coerce")
        if bold_best and values.notna().any():
            best[c] = values.min() if c in lower_better else values.max()
    for _, row in frame.iterrows():
        cells = [escape(row[columns[0]])]
        for c in columns[1:]:
            v = row[c]
            if isinstance(v, tuple):                    # (mean, standard deviation)
                text, v = rf"{v[0]:.3f}$\pm${v[1]:.3f}", v[0]
            elif isinstance(v, (float, np.floating)) and np.isfinite(v):
                text = f"{v:.3f}"
            else:
                text = escape(v)
            if c in best and isinstance(v, (float, np.floating)) and np.isclose(round(v, 3), round(best[c], 3)):
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\\[2pt]\parbox{{0.95\linewidth}}{{\footnotesize {note}}}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines) + "\n"


def write_table(folder, name, frame, **kwargs):
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / f"{name}.csv", index=False)
    (folder / f"{name}.tex").write_text(latex_table(frame, **kwargs), encoding="utf-8")


# ---- data description ----

def data_figures(raw, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    counts = raw["GENDER"].map({"F": "Female", "M": "Male"}).value_counts().reindex(["Female", "Male"])
    axes[0].bar(counts.index, counts.values, color=COLOURS[:2], width=0.6)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Subscribers")
    axes[0].set_title("(a) Gender", loc="left")
    axes[0].set_ylim(0, counts.max() * 1.25)
    groups = pd.Series(benchmark.age_classes(raw["AGE"], "6")).map(dict(enumerate(benchmark.AGE_NAMES)))
    groups = groups.value_counts().reindex(benchmark.AGE_NAMES)
    axes[1].bar(groups.index, groups.values, color=COLOURS[0], width=0.6)
    for i, v in enumerate(groups.values):
        axes[1].text(i, v, f"{v / groups.sum():.1%}", ha="center", va="bottom", fontsize=8)
    axes[1].set_title("(b) Six age groups", loc="left")
    axes[1].set_ylim(0, groups.max() * 1.18)
    save(fig, out, "fig_data_distribution")

    numeric = raw.select_dtypes("number").drop(columns=["AGE"])
    corr = numeric.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.04, label="Pearson correlation")
    save(fig, out, "fig_correlation")
    corr.to_csv(out / "fig_correlation.csv")
    return {"rows": len(raw), "gender": counts.to_dict(), "six_groups": groups.to_dict()}


# ---- one repeat with test predictions (seed 42) ----

def seed_model(raw, scheme, seed=42, epochs=150):
    data = benchmark.prepare(raw, seed)
    n_age = benchmark.N_CLASSES[scheme]
    age = benchmark.age_classes(data["age"], scheme)
    male = data["male"]
    joint = male * n_age + age
    train, validation, test = data["train"], data["validation"], data["test"]
    chosen, terms = qubo.select(data["X"][train], joint[train], **benchmark.DEFAULTS, seed=seed)
    X = data["X"][:, chosen]
    net = MultiTaskNet(X.shape[1], n_age, seed=seed)
    net.fit(X[train], male[train], age[train], X[validation], male[validation], age[validation], n_age, epochs=epochs)
    p_male, p_age = net.predict(X[test])
    baselines = {}
    for name, model in benchmark.baselines(scheme, seed).items():
        fitted = model.fit(X[train], age[train])
        baselines[name] = benchmark.full_probabilities(fitted, X[test], n_age)
    return {"net": net, "X_test": X[test], "male": male[test], "age": age[test], "p_male": p_male, "p_age": p_age,
            "names": [data["names"][i] for i in chosen], "relevance": terms["relevance"], "all_names": data["names"],
            "chosen": chosen, "baselines": baselines, "n_age": n_age}


def permutation_importance(model, repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    net, X, male, age = model["net"], model["X_test"], model["male"], model["age"]
    p_male, p_age = net.predict(X)
    base_age = metrics.age_scores(p_age, age)["macro_f1"]
    base_gender = metrics.gender_scores(p_male, male)["auc"]
    rows = []
    for j, name in enumerate(model["names"]):
        drops_age, drops_gender = [], []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            pm, pa = net.predict(shuffled)
            drops_age.append(base_age - metrics.age_scores(pa, age)["macro_f1"])
            drops_gender.append(base_gender - metrics.gender_scores(pm, male)["auc"])
        rows.append({"attribute": name, "age_macro_f1_drop": np.mean(drops_age), "gender_auc_drop": np.mean(drops_gender)})
    return pd.DataFrame(rows)


def model_figures(model, scheme, out):
    names = step2.CLASS_NAMES[scheme]
    tag = f"{scheme}class"
    # confusion matrices (row-normalised: each row is one true group and sums to 100 %)
    panels = [("age", model["age"], model["p_age"].argmax(1), names)]
    if scheme == "6":
        panels.append(("gender", model["male"], (model["p_male"] >= 0.5).astype(int), ["Female", "Male"]))
    for task, truth, predicted, labels in panels:
        matrix = confusion_matrix(truth, predicted, labels=range(len(labels)))
        share = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        size = 3.2 if task == "gender" else (4.2 if len(labels) <= 6 else 5.6)
        fig, ax = plt.subplots(figsize=(size, size * 0.9))
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{matrix[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=6.5,
                        color="white" if share[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(len(labels)), labels, rotation=45 if len(labels) > 6 else 0)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.spines[:].set_visible(False)
        save(fig, out, f"fig_confusion_{task}_{tag}")
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(out / f"fig_confusion_{task}_{tag}.csv")

    # one-vs-rest ROC of the network, one curve per age group
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    rows = []
    for c, label in enumerate(names):
        positive = model["age"] == c
        if positive.any() and not positive.all():
            fpr, tpr, _ = roc_curve(positive, model["p_age"][:, c])
            auc = np.trapezoid(tpr, fpr)
            ax.plot(fpr, tpr, lw=1.5, color=COLOURS[c % len(COLOURS)], label=f"{label} ({auc:.3f})",
                    ls="-" if c < len(COLOURS) else "--")
            rows.append({"group": label, "auc": auc})
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=6.5, frameon=False, title="Age group (AUC)", title_fontsize=7)
    save(fig, out, f"fig_roc_age_{tag}")
    pd.DataFrame(rows).to_csv(out / f"fig_roc_age_{tag}.csv", index=False)
    if scheme == "6":
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        fpr, tpr, _ = roc_curve(model["male"], model["p_male"])
        ax.plot(fpr, tpr, lw=1.8, color=COLOURS[0], label=f"MTL (AUC {np.trapezoid(tpr, fpr):.3f})")
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls=":")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(frameon=False, fontsize=7)
        save(fig, out, "fig_roc_gender_6class")


def feature_figures(model, out):
    relevance = pd.DataFrame({"attribute": model["all_names"], "relevance": model["relevance"]})
    relevance["chosen"] = relevance.index.isin(model["chosen"])
    relevance = relevance.sort_values("relevance", ascending=True)
    fig, ax = plt.subplots(figsize=(4.6, 7.0))
    ax.barh(relevance["attribute"], relevance["relevance"],
            color=[COLOURS[0] if c else "#c3c2b7" for c in relevance["chosen"]], height=0.7)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlabel("Relevance to gender x age group (NMI)")
    ax.scatter([], [], color=COLOURS[0], marker="s", label="chosen by QUBO")
    ax.scatter([], [], color="#c3c2b7", marker="s", label="not chosen")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, out, "fig_qubo_relevance")
    relevance.to_csv(out / "fig_qubo_relevance.csv", index=False)

    importance = permutation_importance(model)
    importance.to_csv(out / "fig_importance.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=False)
    for ax, column, title, colour in ((axes[0], "age_macro_f1_drop", "Age: drop in macro-F1", COLOURS[0]),
                                      (axes[1], "gender_auc_drop", "Gender: drop in AUC", COLOURS[1])):
        top = importance.sort_values(column).tail(12)
        ax.barh(top["attribute"], top[column], color=colour, height=0.7)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(title, loc="left")
    fig.tight_layout()
    save(fig, out, "fig_importance")


# ---- Step 1 tables and figures ----

def scheme_means(step1_dir, scheme):
    scores = pd.read_csv(Path(step1_dir) / f"scores_{scheme}class.csv")
    test = scores[scores["part"] == "test"]
    return test.drop(columns=["seed", "part", "scheme"]).groupby(["task", "model"]).agg(["mean", "std"])


def mean_sd(means, task, model, measure):
    return means.loc[(task, model), (measure, "mean")], means.loc[(task, model), (measure, "std")]


def step1_outputs(step1_dir, fig_dir, table_dir):
    step1_dir = Path(step1_dir)
    numbers = {}
    m6 = scheme_means(step1_dir, "6")
    rows = []
    for model in MODELS:
        row = {"Model": SHOWN.get(model, model)}
        for measure, title in (("accuracy", "Gender Acc."), ("f1", "Gender F1"), ("auc", "Gender AUC")):
            row[title] = mean_sd(m6, "gender", model, measure)
        for measure, title in (("accuracy", "Age Acc."), ("within_1", "+-1-bin"), ("macro_f1", "macro-F1"),
                               ("balanced", "Balanced"), ("auc_ovr", "AUC OvR")):
            row[title] = mean_sd(m6, "age", model, measure)
        rows.append(row)
    frame = pd.DataFrame(rows)
    write_table(table_dir, "table_lat_6class", frame, wide=True, bold_best=True, label="tab:LAT",
                caption="LAT dataset: test performance (mean$\\pm$std over five seeds). Best mean per column in \\textbf{bold}.")
    sd = {}
    for model in MODELS:
        for task, measure in (("gender", "accuracy"), ("gender", "auc"), ("age", "macro_f1"), ("age", "auc_ovr")):
            mean, s = mean_sd(m6, task, model, measure)
            sd[f"{model} / {task} {measure}"] = (round(mean, 3), round(s, 3))
    numbers["lat_6class_mean_sd"] = sd

    ece = pd.DataFrame([{"Model": SHOWN.get(m, m), "Gender ECE": m6.loc[("gender", m), ("ece", "mean")],
                         "Age ECE": m6.loc[("age", m), ("ece", "mean")]} for m in MODELS])
    write_table(table_dir, "table_ece", ece, bold_best=True, lower_better=("Gender ECE", "Age ECE"), label="tab:ece",
                caption="Expected Calibration Error on the LAT test rows, six age groups (mean over five seeds; lower is better).")

    # every age scheme in one table
    rows = []
    for scheme, title in (("3", "3 classes"), ("6", "6 classes"), ("10", "10 classes"), ("48", "48 classes")):
        path = step1_dir / f"scores_{scheme}class.csv"
        if not path.exists():
            continue
        means = scheme_means(step1_dir, scheme)
        for model in MODELS:
            row = {"Scheme": title, "Model": SHOWN.get(model, model)}
            for measure, name in (("accuracy", "Acc"), ("within_1", "+-1"), ("balanced", "Bal."), ("macro_f1", "mF1"),
                                  ("weighted_f1", "wF1"), ("auc_ovr", "AUC OvR"), ("ece", "ECE")):
                row[name] = means.loc[("age", model), (measure, "mean")]
            if scheme == "48":
                row["+-5 years"] = means.loc[("age", model), ("within_5", "mean")]
            rows.append(row)
    full = pd.DataFrame(rows)
    numbers["schemes"] = full.round(3).to_dict("records")
    blocks = []
    for title, block in full.groupby("Scheme", sort=False):
        block = block.drop(columns=["Scheme", "+-5 years"], errors="ignore")
        blocks.append(latex_table(block, caption="", label="", bold_best=True, lower_better=("ECE",)))
    # one table with a block per scheme: the rows of each block come from latex_table, best marked within the block
    body = []
    for (title, _), text in zip(full.groupby("Scheme", sort=False), blocks):
        lines = text.splitlines()
        start = lines.index(r"\midrule") + 1
        end = lines.index(r"\bottomrule")
        body.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{title}}}}}\\")
        body += lines[start:end] + [r"\midrule"]
    header = " & ".join(escape(c) for c in ["Model", "Acc", "+-1", "Bal.", "mF1", "wF1", "AUC OvR", "ECE"]) + r"\\"
    tex = "\n".join([r"\begin{table*}[h]", r"\centering",
                     r"\caption{Age classification over different class settings on the LAT public feature set (mean over five "
                     r"seeds). Acc: exact accuracy; $\pm1$: adjacent-class tolerance; Bal.: balanced accuracy; mF1/wF1: "
                     r"macro/weighted $F_1$; AUC OvR: one-vs-rest AUC; ECE: Expected Calibration Error (lower is better). "
                     r"Best per column within each class setting in \textbf{bold}.}\label{tab:granfull}",
                     r"\small\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{lccccccc}", r"\toprule", header, r"\midrule"]
                    + body[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]) + "\n"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "table_schemes.tex").write_text(tex, encoding="utf-8")
    full.to_csv(table_dir / "table_schemes.csv", index=False)

    # the comparison with published work: the best of our models per scheme
    def best(scheme, measure):
        means = scheme_means(step1_dir, scheme)
        values = {m: means.loc[("age", m), (measure, "mean")] for m in MODELS}
        winner = max(values, key=values.get)
        return values[winner], SHOWN.get(winner, winner).replace(" (proposed)", "")
    comparison = {}
    for scheme, measure in (("3", "accuracy"), ("6", "accuracy"), ("6", "macro_f1"), ("10", "accuracy"), ("48", "within_5")):
        if (step1_dir / f"scores_{scheme}class.csv").exists():
            comparison[f"{scheme} / {measure}"] = best(scheme, measure)
    numbers["comparison_best"] = comparison
    prior = {"3 / accuracy": ("3-class (Al-Zuabi style)", "accuracy", "Al-Zuabi 0.655"),
             "6 / accuracy": ("6-class", "accuracy", "--"),
             "6 / macro_f1": ("6-class", "macro-F1", "--"),
             "10 / accuracy": ("10-class (Kuang exact)", "exact accuracy", "Kuang 0.184-0.259"),
             "48 / within_5": ("48-bin (Kuang +-5 yr)", "+-5-year tolerance", "Kuang 0.78")}
    rows = [{"Approach (classes)": prior[key][0], "Metric": prior[key][1], "Our best": f"{value:.3f} ({who})",
             "Prior Work": prior[key][2]} for key, (value, who) in comparison.items()]
    write_table(table_dir, "table_comparison", pd.DataFrame(rows), label="tab:gran",
                caption="Age prediction on the publicly available atomic features (five seeds average). \"Our best\" shows the "
                        "best results achieved either by MTL or the baselines (Logistic Regression, Random Forest, Gradient "
                        "Boosting). Note: previous works use the full feature spaces which are not publicly available and are "
                        "shown here only for context.")

    # loss curves (mean over seeds, band = range)
    for scheme in ("6", "10"):
        path = step1_dir / f"mtl_loss_{scheme}class.csv"
        if not path.exists():
            continue
        loss = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(4.2, 2.9))
        for column, colour, name in (("train_loss", COLOURS[0], "Training"), ("validation_loss", COLOURS[1], "Validation")):
            pivot = loss.pivot(index="epoch", columns="seed", values=column)
            ax.plot(pivot.index + 1, pivot.mean(axis=1), color=colour, lw=1.8, label=name)
            ax.fill_between(pivot.index + 1, pivot.min(axis=1), pivot.max(axis=1), color=colour, alpha=0.15, lw=0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted loss")
        ax.legend(frameon=False)
        save(fig, fig_dir, f"fig_loss_{scheme}class")

    # QUBO sweep
    sweep_path = step1_dir / "qubo_sweep_means.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        defaults = benchmark.DEFAULTS
        fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.2), sharey=True)
        rows = []
        for ax, name in zip(axes, ["k", "beta", "lam", "alpha"]):
            others = [o for o in defaults if o != name]
            part = sweep[np.all([np.isclose(sweep[o], defaults[o]) for o in others], axis=0)].sort_values(name)
            ax.errorbar(part[name], part["mean"], yerr=part["std"], color=COLOURS[0], marker="o", ms=4, lw=1.5, capsize=2)
            ax.axvline(defaults[name], color=GREY, lw=0.8, ls=":")
            ax.set_xlabel({"k": "k", "beta": "beta", "lam": "lambda", "alpha": "alpha"}[name])
            for _, r in part.iterrows():
                rows.append({"Weight": name, "Value": r[name], "Validation macro-F1": r["mean"], "SD": r["std"]})
        axes[0].set_ylabel("Validation macro-F1")
        fig.tight_layout()
        save(fig, fig_dir, "fig_qubo_sweep")
        frame = pd.DataFrame(rows)
        frame["Weight"] = frame["Weight"].map({"k": "k", "beta": "beta", "lam": "lambda", "alpha": "alpha"}) + " = " + frame["Value"].map(lambda v: f"{v:g}")
        write_table(table_dir, "table_sweep", frame.drop(columns="Value"), label="tab:sweep", bold_best=False,
                    caption="QUBO sensitivity: validation macro-F1 of the six-group age head when one weight changes and the "
                            "others keep their default values (k 25, beta 0.05, lambda 1.5, alpha 1; mean and standard deviation over five seeds).")
    return numbers


# ---- Step 2: oversampling ----

def step2_outputs(step2_dir, fig_dir, table_dir):
    step2_dir = Path(step2_dir)
    scores_path = step2_dir / "scores.csv"
    if not scores_path.exists():
        return {"note": "no Step 2 scores"}
    scores = pd.read_csv(scores_path, dtype={"scheme": str})
    paired = pd.read_csv(step2_dir / "paired.csv", dtype={"scheme": str})
    records = pd.read_csv(step2_dir / "records.csv", dtype={"scheme": str})
    order = list(dict.fromkeys(m for m in step2_oversample_order() if m in set(scores["method"])))
    numbers = {}

    rows = []
    for scheme in ("6", "10"):
        for model in ("MTL", "Gradient Boosting"):
            age = scores[(scores["scheme"] == scheme) & (scores["model"] == model) & (scores["task"] == "age")]
            gender = scores[(scores["scheme"] == scheme) & (scores["model"] == model) & (scores["task"] == "gender")]
            for method in order:
                a = age[age["method"] == method]
                g = gender[gender["method"] == method]
                if a.empty:
                    continue
                rows.append({"Model": f"{scheme} classes, {SHOWN.get(model, model)}: {method}",
                             "macro-F1": a["macro_f1"].mean(), "Balanced": a["balanced"].mean(),
                             "Rare-group recall": a["rare_recall"].mean(), "+-1-bin": a["within_1"].mean(),
                             "Age ECE": a["ece"].mean(), "Gender AUC": g["auc"].mean()})
    write_table(table_dir, "table_oversampling", pd.DataFrame(rows), wide=True, label="tab:oversampling",
                caption="Oversampling the training rows: test results on LAT (mean over five seeds). Rare-group recall is the "
                        "mean recall of the two smallest age groups in the training rows. \"none\" uses class weights only.")
    numbers["oversampling"] = pd.DataFrame(rows).round(3).to_dict("records")

    for scheme in ("6", "10"):
        part = paired[(paired["scheme"] == scheme) & (paired["task"] == "age")]
        if part.empty:
            continue
        table = []
        for model in ("MTL", "Gradient Boosting"):
            for method in order[1:]:
                row = {"Model": f"{SHOWN.get(model, model)}: {method}"}
                for measure, title in (("macro_f1", "macro-F1"), ("rare_recall", "Rare-group recall"), ("ece", "ECE")):
                    r = part[(part["model"] == model) & (part["method"] == method) & (part["measure"] == measure)]
                    if r.empty:
                        row[title] = "-"
                        continue
                    r = r.iloc[0]
                    row[title] = f"{r['difference']:+.3f} ({r['p']:.3f})" if np.isfinite(r["p"]) else f"{r['difference']:+.3f}"
                table.append(row)
        write_table(table_dir, f"table_oversampling_paired_{scheme}class", pd.DataFrame(table), label=f"tab:paired{scheme}",
                    caption=f"Each oversampling method minus no oversampling, {scheme} age groups, on the same five seeds: mean "
                            "difference (paired t-test p-value). A positive ECE difference means worse calibration.")

    # per-group recall, 10 classes, the network
    for scheme in ("10", "6"):
        part = scores[(scores["scheme"] == scheme) & (scores["model"] == "MTL") & (scores["task"] == "age")]
        if part.empty:
            continue
        names = step2.CLASS_NAMES[scheme]
        shown = [m for m in ("none", "SMOTE, earlier setting", "SMOTE-NC", "Borderline-SMOTE", "SMOTE-NC, full balance", "CTGAN")
                 if m in set(part["method"])]
        fig, ax = plt.subplots(figsize=(7.2, 2.8))
        width = 0.8 / len(shown)
        for i, method in enumerate(shown):
            values = part[part["method"] == method][[f"recall {n}" for n in names]].mean()
            ax.bar(np.arange(len(names)) + (i - (len(shown) - 1) / 2) * width, values.values, width * 0.9,
                   color=COLOURS[i % len(COLOURS)], label=method)
        ax.set_xticks(range(len(names)), names)
        ax.set_ylabel("Test recall")
        ax.set_xlabel("Age group")
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
        save(fig, fig_dir, f"fig_oversampling_recall_{scheme}class")

    # paired differences in rare-group recall and macro-F1 for every method
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    methods = order[1:]
    for ax, measure, title in ((axes[0], "rare_recall", "(a) Rare-group recall"), (axes[1], "macro_f1", "(b) macro-F1")):
        for k, (scheme, model) in enumerate((("6", "MTL"), ("10", "MTL"), ("6", "Gradient Boosting"), ("10", "Gradient Boosting"))):
            part = paired[(paired["scheme"] == scheme) & (paired["model"] == model) & (paired["measure"] == measure)
                          & (paired["task"] == "age")].set_index("method")
            y = np.arange(len(methods)) + (k - 1.5) * 0.18
            values = [part.loc[m] if m in part.index else None for m in methods]
            for yy, v in zip(y, values):
                if v is None:
                    continue
                ax.plot([v["low"], v["high"]], [yy, yy], color=COLOURS[k], lw=1.2)
                ax.plot(v["difference"], yy, "o", color=COLOURS[k], ms=4,
                        label=f"{scheme} classes, {SHOWN.get(model, model)}" if yy == y[0] else None)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Difference from no oversampling")
    axes[0].set_yticks(range(len(methods)), methods, fontsize=7)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, fig_dir, "fig_oversampling_paired")

    quality = records.groupby(["scheme", "method"])[["rows_added", "seconds", "male_share_before", "male_share_after",
                                                     "distance_ratio", "discriminator_auc"]].mean().reset_index()
    quality = quality[quality["method"] != "none"]
    frame = pd.DataFrame({"Model": quality["scheme"] + " classes: " + quality["method"],
                          "Rows added": quality["rows_added"].round(0).astype(int).astype(str),
                          "Male share before": quality["male_share_before"], "Male share after": quality["male_share_after"],
                          "Distance ratio": quality["distance_ratio"], "Discriminator AUC": quality["discriminator_auc"],
                          "Seconds": quality["seconds"].round(1).astype(str)})
    write_table(table_dir, "table_synthetic_rows", frame, label="tab:synthetic", wide=True,
                caption="What each method adds to the training rows (mean over five seeds). Distance ratio: distance from a new row "
                        "to the closest real row of its class, divided by the same distance between real rows (0 = copies). "
                        "Discriminator AUC: how well a classifier tells new rows from real ones (0.5 = it cannot).")
    numbers["synthetic_rows"] = frame.to_dict("records")
    return numbers


def step2_oversample_order():
    from agender_mtl import oversample
    return list(oversample.METHODS)


# ---- Step 3: Iranian Churn ----

def step3_outputs(step3_dir, table_dir):
    step3_dir = Path(step3_dir)
    if not (step3_dir / "means.csv").exists():
        return {"note": "no Step 3 output"}
    means = pd.read_csv(step3_dir / "means.csv")
    order = ["Logistic Regression", "Random Forest", "Gradient Boosting", "MTL (age head)"]
    base = means[means["method"] == "none"].set_index("model").reindex(order).reset_index()
    frame = pd.DataFrame({"Model": base["model"].replace({"MTL (age head)": "MTL, age head (proposed)"}),
                          "Accuracy": base["accuracy mean"], "+-1-bin": base["within_1 mean"], "macro-F1": base["macro_f1 mean"],
                          "Balanced": base["balanced mean"], "AUC OvR": base["auc_ovr mean"], "ECE": base["ece mean"]})
    write_table(table_dir, "table_iranian", frame, bold_best=True, lower_better=("ECE",), label="tab:iranian", wide=True,
                caption="External check on Iranian Churn: five age groups, no gender (test rows, mean over five seeds).")
    over = means[means["method"] != "none"].copy()
    frame2 = pd.DataFrame({"Model": over["model"].str.replace("MTL (age head)", "MTL", regex=False) + ": " + over["method"],
                           "macro-F1": over["macro_f1 mean"], "Balanced": over["balanced mean"],
                           "Rare-group recall": over["rare_recall mean"], "ECE": over["ece mean"]})
    write_table(table_dir, "table_iranian_oversampling", frame2, label="tab:iranian_os",
                caption="Oversampling on Iranian Churn (test rows, mean over five seeds).")
    choice = json.loads((step3_dir / "qubo_choice.json").read_text())
    return {"iranian": frame.round(3).to_dict("records"), "k_chosen": [c["k"] for c in choice]}


def scheme_tables(step1_dir, step2_dir, table_dir):
    """The 10- and 48-class tables in the article's layout, with the network after the earlier SMOTE setting."""
    smote = None
    if (Path(step2_dir) / "scores.csv").exists():
        s2 = pd.read_csv(Path(step2_dir) / "scores.csv", dtype={"scheme": str})
        smote = s2[(s2["model"] == "MTL") & (s2["task"] == "age") & (s2["method"] == "SMOTE, earlier setting")]
    out = {}
    layouts = {"10": [("accuracy", "Accuracy"), ("within_1", "+-1-bin"), ("macro_f1", "macro-F1"), ("balanced", "Balanced"),
                      ("weighted_f1", "weighted-F1"), ("auc_ovr", "AUC OvR"), ("ece", "ECE")],
               "48": [("accuracy", "Exact Accuracy"), ("macro_f1", "macro-F1"), ("balanced", "Balance"),
                      ("within_5", "+-5-year tolerance")]}
    captions = {"10": "10-class Age Formulation on the LAT data (mean over 5 seeds) where best per column is shown in "
                      "\\textbf{bold}. $\\pm1$-bin: adjacent-class tolerance; AUC OvR: One-v-Rest ROC curve; ECE: Expected "
                      "Calibration Error.",
                "48": "48-class (one year) Age Formulation on the LAT dataset (mean over 5 seeds). The $\\pm5$-year tolerance "
                      "is the metric comparable with Kuang et al.'s $78\\%$ that employed high-dimensional temporal features."}
    for scheme, layout in layouts.items():
        if not (Path(step1_dir) / f"scores_{scheme}class.csv").exists():
            continue
        means = scheme_means(step1_dir, scheme)
        rows = []
        for model in MODELS:
            rows.append({"Model": SHOWN.get(model, model), **{title: means.loc[("age", model), (m, "mean")] for m, title in layout}})
        if smote is not None and not smote[smote["scheme"] == scheme].empty:
            part = smote[smote["scheme"] == scheme]
            rows.append({"Model": "MTL + SMOTE", **{title: part[m].mean() for m, title in layout}})
        frame = pd.DataFrame(rows)
        write_table(table_dir, f"table_{scheme}class", frame, bold_best=True, lower_better=("ECE",), label=f"tab:{'ten' if scheme == '10' else '48'}",
                    caption=captions[scheme])
        out[scheme] = frame.round(3).to_dict("records")
    return out


def ece_table(step1_dir, step3_dir, table_dir):
    """ECE of every model: LAT gender and age (six groups), and IC age when Step 3 exists."""
    m6 = scheme_means(step1_dir, "6")
    ic = None
    if (Path(step3_dir) / "means.csv").exists():
        ic = pd.read_csv(Path(step3_dir) / "means.csv")
        ic = ic[ic["method"] == "none"].set_index("model")["ece mean"].rename({"MTL (age head)": "MTL"})
    rows = []
    for model in MODELS:
        row = {"Model": SHOWN.get(model, model), "LAT gender": m6.loc[("gender", model), ("ece", "mean")],
               "LAT age": m6.loc[("age", model), ("ece", "mean")]}
        if ic is not None:
            row["IC age"] = ic.get(model, np.nan)
        rows.append(row)
    frame = pd.DataFrame(rows)
    write_table(table_dir, "table_ece", frame, bold_best=True, lower_better=tuple(frame.columns[1:]), label="tab:ece",
                caption="Expected Calibration Error (mean over seeds, lower is better). Best per column in \\textbf{bold}.")
    return frame.round(3).to_dict("records")


def regression_check(raw, seeds=benchmark.SEEDS):
    """Age as a number instead of a class: gradient boosting regression on the 25 QUBO attributes (chosen for
    the six-group joint label, as in Step 1). Reports the share within +-5 years and the mean absolute error."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = []
    for seed in seeds:
        data = benchmark.prepare(raw, seed)
        joint = data["male"] * 6 + benchmark.age_classes(data["age"], "6")
        train, test = data["train"], data["test"]
        chosen, _ = qubo.select(data["X"][train], joint[train], **benchmark.DEFAULTS, seed=seed)
        model = HistGradientBoostingRegressor(random_state=seed).fit(data["X"][train][:, chosen], data["age"][train])
        error = np.abs(model.predict(data["X"][test][:, chosen]) - data["age"][test])
        rows.append({"seed": seed, "within_5_years": float(np.mean(error <= 5)), "mae_years": float(error.mean())})
    frame = pd.DataFrame(rows)
    return {"per_seed": frame.round(3).to_dict("records"), "within_5_years_mean": round(frame["within_5_years"].mean(), 3),
            "within_5_years_range": [round(frame["within_5_years"].min(), 3), round(frame["within_5_years"].max(), 3)],
            "mae_years_mean": round(frame["mae_years"].mean(), 2)}


def run(raw, step1_dir, step2_dir, step3_dir, out_dir, epochs=150):
    out_dir = Path(out_dir)
    fig_dir, table_dir = out_dir / "figures", out_dir / "tables"
    numbers = {"data": data_figures(raw, fig_dir)}
    numbers.update(step1_outputs(step1_dir, fig_dir, table_dir))
    for scheme in ("6", "10"):
        model = seed_model(raw, scheme, epochs=epochs)
        model_figures(model, scheme, fig_dir)
        if scheme == "6":
            feature_figures(model, fig_dir)
    numbers["step2"] = step2_outputs(step2_dir, fig_dir, table_dir)
    numbers["step3"] = step3_outputs(step3_dir, table_dir)
    numbers["ece"] = ece_table(step1_dir, step3_dir, table_dir)
    numbers["scheme_tables"] = scheme_tables(step1_dir, step2_dir, table_dir)
    numbers["regression"] = regression_check(raw)
    with open(out_dir / "numbers.json", "w") as f:
        json.dump(numbers, f, indent=1, default=str)
    return numbers
