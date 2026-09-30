#!/usr/bin/env python3
"""LaTeX table of trial-protocol relocalization success on the public datasets (outputs/public/<run>/<cond>)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "outputs/public"
VK = ["clone", "morning", "overcast", "sunset", "fog", "rain", "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right"]
TA = ["P000", "P005", "P002", "P004"]


def load(run, cond):
    f = PUB / run / cond / "reloc_summary.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text())


def cell(d, k):
    if d is None or d.get(k) is None:
        return "--"
    return f"{d[k]:.2f}"


def rows(prefix, conds, names):
    out = []
    for c, n in zip(conds, names):
        p, f = load(f"{prefix}_pnp", c), load(f"{prefix}_ff", c)
        vals = [cell(p, "RS"), cell(f, "RS"), cell(p, "RS_1m_5deg"), cell(f, "RS_1m_5deg"),
                cell((p or {}).get("map_relative"), "recall_1m_5deg"), cell((f or {}).get("map_relative"), "recall_1m_5deg")]
        # bold the better RS
        for i in (0, 2, 4):
            a, b = vals[i], vals[i + 1]
            if a != "--" and b != "--" and a != b:
                j = i if float(a) > float(b) else i + 1
                vals[j] = r"\textbf{" + vals[j] + "}"
        out.append(f"    {n} & " + " & ".join(vals) + r" \\")
    return out


def main():
    lines = [r"\begin{table}[t]", r"  \centering\small", r"  \setlength{\tabcolsep}{3.5pt}",
             r"  \begin{tabular}{@{}l|cc|cc|cc@{}}", r"    \toprule",
             r"    & \multicolumn{2}{c|}{RS ($r_D$)} & \multicolumn{2}{c|}{RS (1\,m/$5^\circ$)} & \multicolumn{2}{c}{recall (1\,m/$5^\circ$)}\\",
             r"    Query & PnP & FF & PnP & FF & PnP & FF\\", r"    \midrule",
             r"    \multicolumn{7}{@{}l}{\emph{vKITTI2 Scene01, map = clone, trials 100/50, $r_D=5$\,m}}\\"]
    lines += rows("vk01", VK, [c.replace("-deg-", "$^\\circ$ ") for c in VK])
    lines += [r"    \midrule", r"    \multicolumn{7}{@{}l}{\emph{TartanAir V2 TinyHouseNight, map = P000, trials 40/20, $r_D=2$\,m}}\\"]
    lines += rows("ta", TA, TA)
    lines += [r"    \bottomrule", r"  \end{tabular}",
              r"  \caption{System-level relocalization on the public datasets: \cross{} with the original PnP module (SGBM depth) versus \method{} (FF). RS: fraction of successful trials; recall: fraction of all trial frames within 1\,m/$5^\circ$ (map-relative).}",
              r"  \label{tab:public-system}", r"\end{table}"]
    (ROOT / "report/sections/results_system_table.tex").write_text("\n".join(lines) + "\n")
    # summary numbers for the text
    for prefix, conds in (("vk01", VK), ("ta", TA)):
        for est in ("pnp", "ff"):
            rs = [load(f"{prefix}_{est}", c) for c in conds]
            rs = [d["RS"] for d in rs if d]
            print(prefix, est, "mean RS %.3f over %d" % (np.mean(rs), len(rs)) if rs else "no runs")


if __name__ == "__main__":
    main()
