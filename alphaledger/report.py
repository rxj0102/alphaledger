"""Self-contained HTML report for a ledger, and the data behind the dashboard.

    alphaledger-report research.db -o report.html

The report is one HTML file with no external assets: summary tiles, a
scatter of naive Sharpe against deflated Sharpe ratio, the leaderboard,
the trial status breakdown and the ledger verification result.
"""

from __future__ import annotations

import argparse
import html
import math

import pandas as pd

from .ledger import Ledger
from .stats import evaluate


def ledger_view(ledger: Ledger) -> dict:
    """Everything the report and the dashboard show, as plain Python data."""
    trials = ledger.trials()
    status_counts = trials["status"].fillna("untested").value_counts().to_dict()
    ver = ledger.verify()
    out = {
        "n_hypotheses": int(len(trials)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "chain_ok": ver.ok,
        "head_hash": ledger.head_hash(),
        "n_entries": ver.n_entries,
        "evaluation": None,
    }
    try:
        ev = evaluate(ledger)
    except ValueError:
        return out
    specs = ledger.hypotheses().set_index("hypothesis_id")
    tab = ev.table.copy()
    tab["label"] = [
        str(specs.loc[h, "spec"].get("expression") or specs.loc[h, "spec"]) for h in tab.index
    ]
    tab["parent_id"] = [specs.loc[h, "parent_id"] for h in tab.index]
    s = ev.summary()
    out["evaluation"] = {"summary": s, "table": tab}
    return out


# ---------------------------------------------------------------- html
CSS = """
.viz-root { color-scheme: light; --surface-1:#ffffff; --surface-2:#f6f6f4; --text-primary:#191919;
  --text-secondary:#666666; --grid:#e4e4e0; --series-1:#257de0; --series-2:#f47230; --good:#008a17;
  --critical:#ec4647; font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:var(--surface-1);
  color:var(--text-primary); margin:0; padding:32px; }
@media (prefers-color-scheme: dark) { .viz-root { color-scheme: dark; --surface-1:#191919; --surface-2:#232323;
  --text-primary:#efefef; --text-secondary:#9c9c9c; --grid:#333333; --series-1:#338ced; --series-2:#e76624;
  --good:#0aac23; --critical:#eb6263; } }
.wrap { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 32px 0 12px; }
.sub { color: var(--text-secondary); font-size: 13px; margin-bottom: 24px; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.tile { background: var(--surface-2); border-radius: 10px; padding: 14px 16px; }
.tile .k { color: var(--text-secondary); font-size: 12px; } .tile .v { font-size: 24px; font-weight: 600; margin-top: 4px; }
.tile .n { color: var(--text-secondary); font-size: 11px; margin-top: 2px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--text-secondary); font-weight: 500; } td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
code { font-size: 12px; } .badge { display:inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; }
.ok { color: var(--good); } .bad { color: var(--critical); }
.legend { font-size: 12px; color: var(--text-secondary); display:flex; gap:16px; margin: 4px 0 8px; }
.sw { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }
svg text { fill: var(--text-secondary); font-size: 11px; }
.pt { stroke: var(--surface-1); stroke-width: 2; } .pt:hover { stroke: var(--text-primary); }
"""


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _scatter(tab: pd.DataFrame, hurdle: float, w: int = 960, h: int = 340, ycol: str = "dsr_effective") -> str:
    pad_l, pad_r, pad_t, pad_b = 56, 16, 16, 40
    xs = tab["sr_annual"].to_numpy()
    lo = min(xs.min(), 0.0, hurdle) - 0.1
    hi = max(xs.max(), hurdle) + 0.1

    def X(v):
        return pad_l + (v - lo) / (hi - lo) * (w - pad_l - pad_r)

    def Y(v):
        return pad_t + (1 - v) * (h - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
             f'aria-label="Naive Sharpe against effective deflated Sharpe ratio for each trial">']
    for yv in (0, 0.5, 0.95):
        dash = ' stroke-dasharray="4 4"' if yv == 0.95 else ""
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" '
                     f'stroke="var(--grid)"{dash}/>')
        parts.append(f'<text x="{pad_l - 6}" y="{Y(yv) + 4:.1f}" text-anchor="end">{yv:g}</text>')
    parts.append(f'<text transform="translate(12 {(h - pad_b + pad_t) / 2:.0f}) rotate(-90)" '
                 f'text-anchor="middle">DSR, effective trials</text>')
    step = 0.5 if hi - lo < 4 else 1.0
    v = math.ceil(lo / step) * step
    while v <= hi:
        parts.append(f'<text x="{X(v):.1f}" y="{h - pad_b + 16}" text-anchor="middle">{v:g}</text>')
        v += step
    parts.append(f'<line x1="{X(hurdle):.1f}" x2="{X(hurdle):.1f}" y1="{pad_t}" y2="{h - pad_b}" '
                 f'stroke="var(--text-secondary)" stroke-dasharray="2 3"/>')
    right = X(hurdle) > w - 140
    parts.append(f'<text x="{X(hurdle) + (-4 if right else 4):.1f}" y="{pad_t + 10}" '
                 f'text-anchor="{"end" if right else "start"}">hurdle Sharpe {hurdle:.2f}</text>')
    parts.append(f'<text x="{(w + pad_l) / 2:.0f}" y="{h - 4}" text-anchor="middle">naive annual Sharpe</text>')
    for hid, r in tab.iterrows():
        color = "var(--series-2)" if r["sig_bh"] else "var(--series-1)"
        tip = html.escape(f"{r['label']}\nnaive Sharpe {r['sr_annual']:.2f}\nDSR {r['dsr']:.3f}\n"
                          f"DSR effective {r['dsr_effective']:.3f}\nBH significant: {bool(r['sig_bh'])}\n{hid}")
        parts.append(f'<circle class="pt" cx="{X(r["sr_annual"]):.1f}" cy="{Y(r[ycol]):.1f}" r="5" '
                     f'fill="{color}"><title>{tip}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def build_report(ledger: Ledger, title: str = "alphaledger report", top: int = 25) -> str:
    view = ledger_view(ledger)
    e = view["evaluation"]
    chain = ('<span class="ok">verified</span>' if view["chain_ok"] else '<span class="bad">BROKEN</span>')
    tiles = [("Hypotheses logged", str(view["n_hypotheses"]),
              ", ".join(f"{k} {v}" for k, v in sorted(view["status_counts"].items()))),
             ("Ledger chain", chain, f"{view['n_entries']} entries, head {view['head_hash'][:12]}")]
    body = []
    if e is not None:
        s, tab = e["summary"], e["table"]
        tiles = [
            ("Trials counted", str(s["n_trials"]), "includes failed and leaky runs"),
            ("Best naive Sharpe", _fmt(s["best_naive_sharpe"]), "annualized, in sample"),
            ("Effective trials", _fmt(s["n_effective_trials"], 1), "independent trials after correlation"),
            ("Hurdle Sharpe", _fmt(s["hurdle_sharpe_annual_effective"]),
             f"effective; analytic {_fmt(s['hurdle_sharpe_annual_analytic'])}, "
             f"trial spread {_fmt(s['hurdle_sharpe_annual'])}"),
            ("Best deflated Sharpe ratio", _fmt(s["best_dsr_effective"], 3),
             f"effective; analytic {_fmt(s['best_dsr_analytic'], 3)}, trial spread {_fmt(s['best_dsr'], 3)}; "
             "0.95 or more is convincing"),
            ("Significant after BH", f"{s['n_significant_bh']}",
             f"{s['n_significant_raw']} before correction, one-sided"),
            ("PBO", _fmt(s["pbo"]), "probability of backtest overfitting"),
        ] + tiles
        body.append("<h2>Naive Sharpe against deflated Sharpe ratio (effective trials)</h2>")
        body.append('<div class="legend"><span><span class="sw" style="background:var(--series-1)"></span>'
                    'not significant after BH</span><span><span class="sw" style="background:var(--series-2)">'
                    '</span>significant after BH (one-sided)</span><span>dashed line: DSR 0.95</span></div>')
        body.append(_scatter(tab, s["hurdle_sharpe_annual_effective"]))
        rows = []
        for hid, r in tab.head(top).iterrows():
            rows.append(
                f"<tr><td><code>{html.escape(r['label'])}</code><br><span class='sub'>{hid}</span></td>"
                f"<td class='num'>{_fmt(r['sr_annual'])}</td><td class='num'>{_fmt(r['dsr'], 3)}</td>"
                f"<td class='num'>{_fmt(r['dsr_analytic'], 3)}</td><td class='num'>{_fmt(r['dsr_effective'], 3)}</td>"
                f"<td class='num'>{_fmt(r['p_bh'], 3)}</td><td class='num'>{_fmt(r['sr_haircut_holm'])}</td>"
                f"<td class='num'>{int(r['t_obs'])}</td></tr>")
        body.append(f"<h2>Leaderboard (top {min(top, len(tab))} of {len(tab)} by naive Sharpe)</h2>")
        body.append("<table><tr><th>Hypothesis</th><th class='num'>Naive Sharpe</th><th class='num'>DSR trial spread</th>"
                    "<th class='num'>DSR analytic</th><th class='num'>DSR effective</th>"
                    "<th class='num'>BH p</th><th class='num'>Haircut Sharpe</th><th class='num'>Days</th></tr>"
                    + "".join(rows) + "</table>")
    tiles_html = "".join(f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v}</div>"
                         f"<div class='n'>{n}</div></div>" for k, v, n in tiles)
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><style>{CSS}</style></head>"
            f"<body class='viz-root'><div class='wrap'><h1>{html.escape(title)}</h1>"
            f"<div class='sub'>Every number is computed from the ledger. Sharpe ratios are annualized. "
            f"DSR, BH and haircut account for every trial, including failures.</div>"
            f"<div class='tiles'>{tiles_html}</div>{''.join(body)}</div></body></html>")


def main(argv=None) -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(prog="alphaledger-report")
    ap.add_argument("ledger")
    ap.add_argument("-o", "--output", default="report.html")
    ap.add_argument("--title", default="alphaledger report")
    args = ap.parse_args(argv)
    with Ledger(args.ledger) as lg:
        html_text = build_report(lg, args.title)
    with open(args.output, "w") as f:
        f.write(html_text)
    print(f"wrote {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
