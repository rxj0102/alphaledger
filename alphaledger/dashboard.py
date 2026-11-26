"""Streamlit dashboard for a ledger.

    pip install -e .[dashboard]
    alphaledger-dashboard research.db
"""

from __future__ import annotations

import os
import sys


def app() -> None:
    import streamlit as st

    from alphaledger.ledger import Ledger
    from alphaledger.report import ledger_view

    path = os.environ.get("ALPHALEDGER_DB") or (sys.argv[1] if len(sys.argv) > 1 else "")
    st.set_page_config(page_title="alphaledger", layout="wide")
    st.title("alphaledger")
    path = st.text_input("Ledger file", value=path)
    if not path or not os.path.exists(path):
        st.info("Enter the path of a ledger file.")
        return
    with Ledger(path) as lg:
        view = ledger_view(lg)
        hyp = lg.hypotheses().set_index("hypothesis_id")

    c = st.columns(4)
    c[0].metric("Hypotheses logged", view["n_hypotheses"])
    c[1].metric("Ledger entries", view["n_entries"])
    c[2].metric("Chain", "verified" if view["chain_ok"] else "BROKEN")
    c[3].metric("Head hash", view["head_hash"][:12])
    st.caption("Trial status: " + ", ".join(f"{k} {v}" for k, v in sorted(view["status_counts"].items())))

    e = view["evaluation"]
    if e is None:
        st.warning("No successful trials with returns yet.")
        return
    s, tab = e["summary"], e["table"]
    c = st.columns(5)
    c[0].metric("Trials counted", s["n_trials"])
    c[1].metric("Best naive Sharpe", f"{s['best_naive_sharpe']:.2f}")
    c[2].metric("Hurdle Sharpe", f"{s['hurdle_sharpe_annual']:.2f}")
    c[3].metric("Best DSR", f"{s['best_dsr']:.3f}", help=f"analytic null: {s['best_dsr_analytic']:.3f}")
    c[4].metric("PBO", "n/a" if s["pbo"] is None else f"{s['pbo']:.2f}")

    st.subheader("Naive Sharpe against deflated Sharpe ratio (analytic null)")
    plot = tab.reset_index()[["hypothesis_id", "label", "sr_annual", "dsr_analytic", "sig_bh"]]
    plot["group"] = plot["sig_bh"].map({True: "significant after BH", False: "not significant"})
    st.scatter_chart(plot, x="sr_annual", y="dsr_analytic", color="group",
                     x_label="naive annual Sharpe", y_label="DSR, analytic null")

    st.subheader("Leaderboard")
    show = tab.reset_index()[["hypothesis_id", "label", "sr_annual", "dsr", "dsr_analytic", "p_bh", "sig_bh",
                              "sr_haircut_holm", "t_obs"]]
    show.columns = ["id", "hypothesis", "naive Sharpe", "DSR", "DSR analytic", "BH p", "BH sig",
                    "haircut Sharpe", "days"]
    st.dataframe(show, hide_index=True, use_container_width=True)

    st.subheader("Lineage")
    pick = st.selectbox("Hypothesis", list(tab.index), format_func=lambda h: f"{h}  {tab.loc[h, 'label']}")
    if pick:
        chain, cur = [], pick
        while cur is not None and cur in hyp.index:
            chain.append(cur)
            p = hyp.loc[cur, "parent_id"]
            cur = p if isinstance(p, str) else None
        for depth, h in enumerate(reversed(chain)):
            spec = hyp.loc[h, "spec"]
            st.markdown(f"{'&nbsp;' * 4 * depth}{'↳ ' if depth else ''}`{spec.get('expression', spec)}` ({h})")


def main() -> None:  # pragma: no cover
    from streamlit.web import cli

    if len(sys.argv) < 2:
        print("usage: alphaledger-dashboard LEDGER_FILE")
        sys.exit(2)
    os.environ["ALPHALEDGER_DB"] = os.path.abspath(sys.argv[1])
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(cli.main())


if __name__ == "__main__":
    app()
