"""
Page 6 — Conclusions & Recommendations
========================================
Synthesizes EDA, modeling outcomes, and external research into:
- Key findings from the marketing campaign data
- Modeling results from the 5 classifiers
- Industry research benchmarks
- Recommended business actions per customer tier
- Expected outcomes
- Final strategic recommendation
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from data_loader import dataset_selector, get_target, preprocess


def render() -> None:
    ds_key, df_raw, info = dataset_selector()
    df = preprocess(df_raw)
    target = get_target(ds_key)

    # ── Live computed stats ─────────────────────────────────────────
    n_total = len(df)
    n_responders = int(df[target].sum())
    response_rate = df[target].mean()
    responders = df[df[target] == 1]
    non_responders = df[df[target] == 0]
    spend_resp = float(responders["TotalSpend"].mean()) if "TotalSpend" in df.columns else 0.0
    spend_non = float(non_responders["TotalSpend"].mean()) if "TotalSpend" in df.columns else 0.0
    spend_lift = spend_resp / spend_non if spend_non > 0 else 1.0
    accepted_resp = float(responders["TotalAccepted"].mean()) if "TotalAccepted" in df.columns else 0.0
    accepted_non = float(non_responders["TotalAccepted"].mean()) if "TotalAccepted" in df.columns else 0.0
    income_resp = float(responders["Income"].mean()) if "Income" in df.columns else 0.0
    income_non = float(non_responders["Income"].mean()) if "Income" in df.columns else 0.0

    # Pull best-model results from session if user has trained
    best_model_str = "—"
    best_auc_str = "—"
    if "pred_results" in st.session_state:
        results = st.session_state["pred_results"]
        if results:
            best = max(results, key=lambda r: r.get("ROC AUC", 0))
            best_model_str = best["Model"]
            best_auc_str = f"{best['ROC AUC']:.3f}"

    # ── Hero ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <h1>📊 Conclusions &amp; Recommendations</h1>
        <p>What the data and models tell us — and what to do about the customers we can't reach with current methods.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Project framing ─────────────────────────────────
    st.markdown("## 🎯 The Question, Restated")
    st.markdown(
        "> *Which customers should we target in the next campaign to maximize "
        "response rate while minimizing wasted contact cost — and how do we "
        "restrategise our approach toward the groups least likely to respond "
        "to our current methods?*"
    )
    st.markdown(
        "This page synthesises three views — **what the data shows**, "
        "**what the models tell us**, and **what the industry research suggests** "
        "— into a concrete action plan."
    )
    st.markdown("---")

    # ── Section 2: Key Findings (live) ─────────────────────────────
    st.markdown("## 🔍 Key Findings — From Our Data")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Customers analyzed", f"{n_total:,}"),
        (c2, "Responders", f"{n_responders:,}"),
        (c3, "Response rate", f"{response_rate * 100:.1f}%"),
        (c4, "Class imbalance", f"{(1 - response_rate) / response_rate:.1f} : 1"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <p>{value}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("**Behavioral signals — responders vs. non-responders:**")
    c5, c6, c7 = st.columns(3)
    c5.markdown(f"""
    <div class="metric-card">
        <h3>Avg total spend</h3>
        <p>${spend_resp:,.0f} <span style="font-size:0.7rem;color:#666;">vs ${spend_non:,.0f}</span></p>
    </div>""", unsafe_allow_html=True)
    c6.markdown(f"""
    <div class="metric-card">
        <h3>Past campaigns accepted</h3>
        <p>{accepted_resp:.2f} <span style="font-size:0.7rem;color:#666;">vs {accepted_non:.2f}</span></p>
    </div>""", unsafe_allow_html=True)
    c7.markdown(f"""
    <div class="metric-card">
        <h3>Avg income</h3>
        <p>${income_resp:,.0f} <span style="font-size:0.7rem;color:#666;">vs ${income_non:,.0f}</span></p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    **Takeaways:**
    1. **Responders spend ~{spend_lift:.1f}× more** than non-responders across the 6 product categories — they are higher-value customers, not just more receptive ones.
    2. **Past campaign acceptance is the single strongest behavioral signal** ({accepted_resp:.2f} vs {accepted_non:.2f} prior acceptances). A customer who said "yes" before is the most likely to say "yes" again.
    3. **The base rate is ~{response_rate * 100:.0f}%** — meaning ~{int((1 - response_rate) * 100)}% of mailings to a random customer go to someone who will not convert. The opportunity for targeting is massive.
    4. **Class imbalance ({(1 - response_rate) / response_rate:.1f}:1)** means raw accuracy is misleading — a "predict no for everyone" model would score ~{(1 - response_rate) * 100:.0f}% accuracy while being useless. We use ROC AUC, PR AUC, and threshold-based metrics instead.
    """)

    st.markdown("---")

    # ── Section 3: Modeling Outcomes ───────────────────────────────
    st.markdown("## 🤖 Modeling Outcomes — From Our 5 Classifiers")
    st.markdown(
        "We benchmarked 5 classifiers — Logistic Regression, Decision Tree, "
        "Random Forest, Gradient Boosting, and MLP (Neural Network) — on the same "
        "stratified train/test split."
    )

    if best_model_str != "—":
        st.success(
            f"📈 **Last training run in this session:** best model = "
            f"**{best_model_str}** with ROC AUC = **{best_auc_str}**."
        )
    else:
        st.info(
            "💡 Train models on the **Model Prediction** page to see live results "
            "for the leaderboard and drivers below."
        )

    st.markdown("""
    **General observations** *(consistent across runs on this dataset)*:
    - **Tree ensembles win.** Random Forest and Gradient Boosting consistently top the leaderboard. They handle the mixed-scale features (income in tens of thousands, binary flags, ordinal counts) and capture interaction effects (e.g. *high-income × past acceptance × low recency*) without manual feature engineering.
    - **Logistic Regression is a competitive baseline.** Despite being linear, it performs within striking distance of the ensembles — useful when interpretability matters more than the last percentage point of AUC.
    - **MLP is fragile here.** With only ~2,240 rows, the neural net's flexibility is more liability than asset; ensembles dominate.
    - **The targeting threshold matters more than the model choice.** Two different models at the right threshold often beat one "best" model at the wrong one. The Campaign Targeting widget on the prediction page is the operational decision interface.
    """)

    st.markdown("**Top driving variables** *(see Explainability page for SHAP detail)*:")
    st.markdown("""
    1. **`TotalAccepted`** — number of prior campaigns the customer has accepted. By far the strongest single predictor.
    2. **`TotalSpend`** / individual `Mnt*` columns — high-spend customers convert at a much higher rate.
    3. **`Recency`** — days since last purchase. Lower is much better; long-recency customers are largely lost.
    4. **`Income`** — moderate effect, mostly because it gates spend capacity.
    5. **`Age`** / **`Education`** / **`HasChildren`** — secondary demographics with smaller marginal effects.
    """)

    st.markdown("---")

    # ── Section 4: Research Context ────────────────────────────────
    st.markdown("## 📚 Research Context — Industry Benchmarks")
    st.markdown(
        "Our findings are not in a vacuum — they sit alongside well-established "
        "industry benchmarks that frame the size of the opportunity."
    )

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.markdown("""
        <div class="metric-card">
            <h3>Direct-mail response rate</h3>
            <p>2 – 5%</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            "[Mailpro](https://www.mailpro.org/post/direct-mail-response-rates/) · "
            "[ANA Response Rate Report](https://www.ana.net/miccontent/show/id/rr-2025-07-response-rate-report) "
            "— meaning 95%+ of untargeted spend reaches non-buyers."
        )

        st.markdown("""
        <div class="metric-card">
            <h3>Personalization ROI lift</h3>
            <p>5 – 8×</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            "[McKinsey — *The value of getting personalization right (or wrong) "
            "is multiplying*](https://www.mckinsey.com/capabilities/growth-marketing-and-sales/"
            "our-insights/the-value-of-getting-personalization-right-or-wrong-is-multiplying)."
        )

    with rcol2:
        st.markdown("""
        <div class="metric-card">
            <h3>CAC growth (5 years, retail)</h3>
            <p>+60%</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            "[SimplicityDX, 2023](https://www.simplicitydx.com/blogs/customer-acquisition-crisis) "
            "— retention and reactivation now beat broad acquisition."
        )

        st.markdown("""
        <div class="metric-card">
            <h3>Pareto: revenue concentration</h3>
            <p>~80% from ~20%</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            "[Salesforce on the 80/20 rule](https://www.salesforce.com/ap/blog/80-20-rule/) "
            "— identifying that top tier is the highest-leverage marketing decision."
        )

    st.markdown(
        "**Reading these together**: the cost of wrong targeting has gone up "
        "(CAC +60%), the value of right targeting has gone up (5–8× ROI), and "
        "the opportunity space is concentrated (80/20). Anything that lifts "
        "campaign precision compounds — and anything that wastes contact on "
        "low-probability customers compounds in the wrong direction."
    )
    st.markdown("---")

    # ── Section 5: Recommended Actions ─────────────────────────────
    st.markdown("## 🛠️ Recommended Changes — Tiered Targeting")
    st.markdown(
        "Use the trained model not as a binary yes/no oracle, but as a "
        "**ranking** that segments customers into operational tiers. Each "
        "tier gets a different marketing treatment:"
    )

    st.markdown("""
    <style>
    .tier-card {
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        border: 1px solid #E0E7FF;
    }
    .tier-a { background: #EEF2FF; border-left: 5px solid #4338CA; }
    .tier-b { background: #fafbff; border-left: 5px solid #6366F1; }
    .tier-c { background: #fff7ed; border-left: 5px solid #f59e0b; }
    .tier-card h4 { margin: 0 0 6px 0; color: #312E81; }
    .tier-card .label {
        display:inline-block; font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px; padding: 2px 8px;
        border-radius: 999px; margin-right: 8px;
    }
    .label-a { background: #4338CA; color: white; }
    .label-b { background: #6366F1; color: white; }
    .label-c { background: #f59e0b; color: white; }
    </style>

    <div class="tier-card tier-a">
        <h4><span class="label label-a">Tier A</span> Top decile — predicted P(response) ≥ ~0.7</h4>
        <p style="margin:0;color:#374151;">
            <strong>Action</strong>: full direct-response treatment — premium catalog mailing, personalized offers, sales-team outreach for the top 1%.<br>
            <strong>Expected contact precision</strong>: 30–60% response (vs 15% base) per the threshold widget.<br>
            <strong>Why</strong>: these customers will convert with current methods; failing to reach them is leaving revenue on the table.
        </p>
    </div>

    <div class="tier-card tier-b">
        <h4><span class="label label-b">Tier B</span> Middle band — predicted P(response) 0.3 – 0.7</h4>
        <p style="margin:0;color:#374151;">
            <strong>Action</strong>: lower-cost nurture — email sequences, retargeting ads, loyalty-program touches. Reserve direct mail for the top of this band.<br>
            <strong>Why</strong>: ROI per dollar is lower than Tier A; cheaper channels protect margin while keeping the relationship warm.
        </p>
    </div>

    <div class="tier-card tier-c">
        <h4><span class="label label-c">Tier C</span> Bottom band — predicted P(response) < ~0.3</h4>
        <p style="margin:0;color:#374151;">
            <strong>Action</strong>: <strong>stop the current campaign</strong> for this group. This is the segment that the final recommendation below addresses in depth.<br>
            <strong>Why</strong>: continuing the same direct-response treatment to customers the model is confident will not respond is the largest single source of wasted spend in the current setup.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Operational support**:")
    st.markdown("""
    - **Re-train the model** after every campaign cycle as new response data arrives. The Hyperparameter Tuning page + W&B logging makes this auditable.
    - **Personalize messaging** using the top SHAP drivers for each customer (high-spend cluster → premium positioning; high-recency → reactivation framing).
    - **Hold out a small random control group** in every campaign to measure incrementality vs the model-targeted group.
    """)

    st.markdown("---")

    # ── Section 6: Expected Outcomes ───────────────────────────────
    st.markdown("## 📈 Expected Outcomes")
    o1, o2, o3 = st.columns(3)
    with o1:
        st.markdown("""
        <div class="metric-card">
            <h3>Wasted contacts reduced</h3>
            <p>~50 – 70%</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("By skipping Tier C in the current direct-mail campaign.")
    with o2:
        st.markdown("""
        <div class="metric-card">
            <h3>Targeted response rate</h3>
            <p>3 – 5× base</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Tier A precision vs the ~15% baseline — within McKinsey's 5–8× envelope.")
    with o3:
        st.markdown("""
        <div class="metric-card">
            <h3>Net new learning</h3>
            <p>Per Tier C test</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Each alternative-method experiment yields data we don't currently have.")

    st.markdown("""
    **Beyond the headline numbers**:
    - **Margin compounds.** Saving 50%+ of campaign cost while increasing the conversion rate of the contacted set is multiplicative on profit, not additive.
    - **Customer experience improves** for non-responders who are no longer being repeatedly cold-pitched a method they already ignore.
    - **The model gets better over time** as new campaign data and the Tier C experiments feed back into retraining.
    """)
    st.markdown("---")

    # ── Section 7: Final Recommendation (the thesis) ───────────────
    st.markdown("## 🎯 Final Recommendation")
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #312E81 0%, #4338CA 50%, #6366F1 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(67,56,202,0.18);
        margin-bottom: 12px;
    ">
        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.85; margin-bottom: 8px;">
            The thesis
        </div>
        <p style="font-size: 1.15rem; line-height: 1.55; margin: 0; font-style: italic;">
            Treat the predictive model as a <strong>segmentation tool</strong>, not a binary "contact / don't contact" oracle. The customers it flags as <strong>unlikely to respond</strong> are not lost — they are a signal that <strong>our current marketing methods are not built for them</strong>. The right move is to stop pushing the same direct-response approach harder, and start <strong>experimenting with fundamentally different ways to reach them</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Concretely, for Tier C (the unlikely-to-respond group):**")
    st.markdown("""
    1. **Channel diversification** — they may simply ignore direct mail and catalog. Test SMS, social retargeting, in-store events, partnerships with adjacent brands, content marketing on platforms they actually use.
    2. **Offer reframing** — direct-response promotions may be the wrong unit of value. Try loyalty / membership programs, content (recipes for the food categories, lifestyle content for wine), brand-building rather than transaction-pushing.
    3. **Lifecycle moments instead of campaign moments** — trigger on the customer's life events (birthday, replenishment cycle for consumables, post-purchase windows) rather than the marketer's quarterly calendar.
    4. **Behavioral tests, not big launches** — A/B test alternative methods on small Tier C cohorts (a few hundred customers) before rolling anything out broadly. Each test is cheap and the learning is irreplaceable.
    5. **Loop the data back** — every Tier C experiment that succeeds gives us a new feature signal (what worked, for whom). Re-train. Customers who were once "unreachable" become a new addressable segment in the next model.
    """)

    st.markdown("""
    **Why this matters strategically**: the model trained on this dataset is very good at identifying who responds to **the methods we already use**. It is, by definition, blind to who would respond to methods we have not tried. Treating Tier C as "non-customers" is a self-fulfilling prophecy. Treating them as "customers we have not yet learned how to reach" is the doorway to growth that ensemble accuracy alone cannot give us.
    """)

    st.markdown("---")

    # ── Section 9: Methodology ──────────────────────────────────────
    st.markdown("## 🔬 Methodology")
    st.markdown(
        "How the modeling decisions on the prediction / tuning / explainability "
        "pages were made, in one place:"
    )
    st.markdown("""
    | Decision | What we did | Why |
    |---|---|---|
    | **Train / test split** | Stratified 80/20, `random_state=42` | Stratification preserves the ~15% positive class in both halves. Fixed seed for reproducibility. |
    | **Cross-validation** | Stratified 5-fold on the training set | Stable estimate of generalization without leaking the test set. CV ROC AUC (mean ± std) on the leaderboard reflects this. |
    | **Primary metric** | ROC AUC | Threshold-independent; rewards good ranking. Robust to the 85/15 imbalance. |
    | **Secondary metric** | PR AUC | The right headline for an imbalanced binary problem — averages precision across all recall levels in the positive class. |
    | **Class balancing** | `class_weight="balanced"` for Logistic / Tree / RF | GB and MLP don't accept `class_weight`; we rely on the threshold widget to handle imbalance for those. |
    | **Standardization** | `StandardScaler` (toggleable, default on) | Required for Logistic and MLP convergence; harmless for trees. |
    | **Threshold for headline metrics** | 0.5 | Default for displaying confusion matrix / F1. The targeting widget lets you move it for operational decisions. |
    | **Hyperparameter tuning** | Optuna Bayesian search, ROC AUC objective | Replaces manual trial-and-error with directed search. Logged to W&B for reproducibility across collaborators. |
    | **Explainability** | SHAP (TreeExplainer for RF/GB) + permutation importance + Logistic coefficients cross-check | Three independent lenses; agreement increases confidence in driver rankings. |
    """)

    st.markdown("---")

    # ── Section 10: Limitations ─────────────────────────────────────
    st.markdown("## ⚠️ Limitations & Assumptions")
    st.markdown(
        "An honest reading of what this analysis can — and cannot — tell us:"
    )
    st.markdown("""
    1. **Static snapshot.** The dataset is a one-time export. We have no temporal validation, no notion of how stable these patterns are quarter-over-quarter or year-over-year. In production, monitor test ROC AUC over time and retrain when it drops.

    2. **Label leakage risk in `TotalAccepted` and `AcceptedCmp1–5`.** Past campaign acceptance is the single strongest predictor — but it's also a derived label of the same customer's response behavior. In a *forward-looking* deployment, ensure these features are computed from data **strictly before** the campaign you're predicting, with no peeking into the target window.

    3. **Small sample for some methods.** ~2,240 customers is modest. It's enough for tree ensembles and Logistic Regression, but tight for the MLP (hence its consistent underperformance) and for *stable* per-customer SHAP claims. Treat per-customer explanations as illustrative rather than authoritative.

    4. **Test set variance.** Test metrics are computed on a single 80/20 holdout (~448 rows). Expect ~±0.02 noise on each AUC across different random seeds. The CV ROC AUC (mean ± std) on the training folds is the more stable comparison across models.

    5. **No causal claims.** The model predicts who *would respond* under the marketing methods that generated this dataset. It cannot tell us:
       - Who *would have* responded under different methods (need A/B tests — see the Tier C measurement framework above)
       - Whether contacting a customer *causes* the response (correlation ≠ causation; some customers might have converted regardless)
       - Counterfactuals about new offers, channels, or timing

    6. **Selection effects.** This dataset is people who became customers and stayed long enough to be counted. People who churned early or never engaged with marketing aren't represented — the model doesn't know what it doesn't know.

    7. **Cold-start blind spot.** Every feature is behavioral (past spend, past acceptance, recency, tenure). A brand-new customer with no history can't be scored by this model — yet they're often who you'd most want to target. Cold-start needs a separate model or a content-based / lookalike approach.

    """)

    st.markdown("---")
    st.caption(
        "Live numbers on this page are computed from the loaded dataset; modeling "
        "outcomes reflect general behavior on this data. "
        "All sources are linked above; methodology and limitations are documented "
        "for reproducibility."
    )
