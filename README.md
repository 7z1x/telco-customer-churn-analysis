# 📊 Telco Customer Churn Prediction

Identifying which subscribers are about to leave — and which retention action actually moves the needle.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-brightgreen?logo=xgboost)](https://xgboost.ai/)
[![Tableau](https://img.shields.io/badge/Dashboard-Tableau-E97627?logo=tableau&logoColor=white)](https://public.tableau.com/)

**📊 Live Dashboard:** [Telco Customer Churn — Tableau Public](https://public.tableau.com/views/TelcoCustomerChurn_17797948116180/Dashboard1) · **📓 Notebook:** [Kaggle](https://www.kaggle.com/code/zx1700/da01-telco-ccp)

---

## 📌 Executive Summary

| Question | Answer |
|---|---|
| How big is the problem? | **26.58%** churn rate — 1,869 of 7,032 customers left |
| What does it cost? | **~$139K/month** in recurring revenue (≈**$1.67M/year**) |
| Who is leaving? | New customers on month-to-month contracts, especially fiber + electronic check |
| Can we act on it? | Yes — **89.2%** of all churners sit in one addressable cohort (57.7% of the base) |

> **Key insight:** churn is concentrated, not spread evenly. Customers on **month-to-month contracts** churn at **42.71%** versus **2.85%** on two-year contracts — a **15× gap**. And **47.68%** of customers in their first year leave, against **9.51%** of those past four years. The first year is the entire battleground.

---

## 🎯 Business Understanding

### Background

A telecommunications company is losing roughly one in four subscribers. In a subscription business, churn compounds: replacing a lost customer costs more than retaining one, and the revenue gap persists for every month the seat stays empty.

The company currently finds out about churn **after** the cancellation is filed. There is no mechanism to identify at-risk customers while there is still time to intervene.

### Business Questions

1. How large is the churn rate, and what is it worth in revenue?
2. What do churning customers have in common?
3. Which factors are most associated with churn?
4. Which segment should retention prioritize first?

### Objectives

| # | Objective | Output |
|---|---|---|
| 1 | Quantify churn and its revenue impact | EDA + revenue-at-risk estimate |
| 2 | Profile the churning customer | Segment analysis |
| 3 | Identify the strongest churn drivers | Correlation + feature importance |
| 4 | Prioritize a retention target segment | Ranked cohort with expected impact |

### Success Metrics

- **Business metric:** reduction in monthly churn rate; retained recurring revenue.
- **Model metric:** **recall on the churn class** is prioritized. Missing a churner costs a full customer lifetime; a false alarm costs one retention offer.

### Scope

**Included:** EDA, churn driver analysis, baseline modeling (Logistic Regression, Random Forest, XGBoost), feature importance, Tableau dashboard, retention recommendations.
**Excluded:** uplift modeling, live A/B testing, CLV modeling, integration with the CRM.

---

## 🗂️ Dataset

| Attribute | Detail |
|---|---|
| **Source** | [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Raw rows** | 7,043 customers |
| **Rows after cleaning** | **7,032** |
| **Columns** | 21 (20 features + target) |
| **Target** | `Churn` (Yes / No) |

**Data quality issues found:**

- **11 rows** had blank `TotalCharges` — all of them with `tenure = 0`, i.e. customers who had not been billed yet. These were dropped, leaving 7,032 rows.
- `TotalCharges` was stored as `object` despite holding numbers; converted with `pd.to_numeric(errors='coerce')`.
- `customerID` carries no signal and was dropped before modeling.
- **Zero duplicate rows.**

**Target distribution:**

| Class | Count | Share |
|---|---:|---:|
| Stayed | 5,163 | 73.42% |
| **Churned** | **1,869** | **26.58%** |

**Feature groups:**

- **Customer profile:** gender, senior citizen, partner, dependents
- **Account:** tenure, contract type, paperless billing, payment method
- **Services:** phone, multiple lines, internet service, online security, online backup, device protection, tech support, streaming TV/movies
- **Billing:** monthly charges, total charges

---

## 🔬 Methodology

```
Data Loading → EDA → Cleaning → Encoding → Train/Test Split → Modeling → Evaluation → Dashboard
```

1. **Data Loading** — 7,043 rows × 21 columns.
2. **EDA** — missing values, duplicates, cardinality, target distribution, categorical and numerical breakdowns against churn.
3. **Cleaning** — `TotalCharges` converted to numeric, 11 blank rows dropped, `customerID` removed.
4. **Encoding** — target mapped Yes/No → 1/0; categoricals one-hot encoded with `drop_first=True`.
5. **Split** — 80:20 train/test (`random_state=42`).
6. **Scaling** — `StandardScaler` fit on train only, then applied to test. Correct order, no leakage.
7. **Modeling** — three baselines compared: Logistic Regression (`class_weight='balanced'`), Random Forest, XGBoost.
8. **Evaluation** — classification report per model, plus feature importance for interpretation.

**Why compare three models?** Logistic Regression gives an interpretable baseline and handles the imbalance with `class_weight`. Random Forest and XGBoost test whether non-linear interactions improve the result. The comparison reveals whether added complexity actually pays off — see [Limitations](#-limitations--assumptions).

---

## 📊 Key Findings (EDA)

### 1. Contract type is the single strongest churn driver

| Contract | Churn Rate | Customers |
|---|---:|---:|
| Month-to-month | **42.71%** | 3,875 |
| One year | 11.28% | 1,472 |
| Two year | **2.85%** | 1,685 |

A month-to-month customer is **15×** more likely to leave than a two-year customer. This is the most actionable finding in the dataset, because contract structure is a business decision, not a customer trait.

### 2. The first year decides everything

| Tenure | Churn Rate | Customers |
|---|---:|---:|
| 0–12 months | **47.68%** | 2,175 |
| 13–24 months | 28.71% | 1,024 |
| 25–48 months | 20.39% | 1,594 |
| 49+ months | **9.51%** | 2,239 |

Churn risk falls by **5×** between the first year and year four onward. Average tenure is **18.0 months** for churners versus **37.7 months** for those who stay.

### 3. Service quality gaps drive churn more than price

| Segment | Churn Rate |
|---|---:|
| No online security | **41.78%** |
| Has online security | 14.64% |
| No tech support | **41.65%** |
| Has tech support | 15.20% |
| No internet service | 7.43% |

Customers without security or support churn at nearly **3×** the rate of those who have them. Note the 7.43% for "no internet service" — that is the same group counted twice (they answered "no internet service" to both questions), so it is a single segment, not two.

### 4. Fiber optic and electronic check are risk multipliers

| Segment | Churn Rate |
|---|---:|
| Fiber optic | **41.89%** |
| DSL | 19.00% |
| Electronic check | **45.29%** |
| Credit card (automatic) | 15.25% |

Fiber customers churn at **2×** the DSL rate, and electronic-check payers at **3×** the auto-pay rate. Automatic payment is a strong retention signal — a passive, frictionless payment method correlates with staying.

### 5. Demographic and billing signals

| Segment | Churn Rate |
|---|---:|
| Senior citizens | **41.68%** |
| Non-senior | 23.65% |
| No partner | 32.98% |
| Has partner | 19.72% |
| Paperless billing | 33.59% |
| Paper billing | 16.38% |

### 6. Correlation with churn (numeric features)

| Feature | Correlation |
|---|---:|
| `tenure` | **−0.35** |
| `MonthlyCharges` | +0.19 |
| `TotalCharges` | −0.20 |

`TotalCharges` correlates negatively simply because it accumulates with tenure — it is a proxy for loyalty, not a cause of churn.

---

## 🎯 The Retention Target Segment

This is the part that turns analysis into a decision. Instead of "customers at risk," here is a cohort with a defined size and quantified payoff.

**Definition:** month-to-month contract **OR** tenure ≤ 12 months.

| Metric | Value |
|---|---:|
| Cohort size | 4,056 customers (**57.7%** of base) |
| Churn rate within cohort | **41.12%** |
| Churners captured | 1,668 (**89.2%** of all churners) |

**A narrower high-risk profile** — month-to-month **+** fiber optic **+** electronic check **+** no online security:

| Metric | Value |
|---|---:|
| Cohort size | 1,128 customers (**16.0%** of base) |
| Churn rate within cohort | **63.48%** |
| Churners captured | 716 (**38.3%** of all churners) |

**Why this matters:** targeting the narrow segment reaches 38% of all churners with only 16% of the customer base — a much better return on retention spend than blanket offers. The broad cohort is the safety net; the narrow one is the spearhead.

---

## 🤖 Model Results

Three models were trained and compared. The notebook does not persist the full classification reports, so **exact per-model metrics are not reproduced here** — reporting numbers that cannot be verified would be worse than omitting them.

**What the analysis establishes:**

- **Logistic Regression** with `class_weight='balanced'` achieved the **highest recall on the churn class (~0.79)**, making it the most suitable baseline for a detection task where missing a churner is the costly error.
- Random Forest and XGBoost were trained and compared, but did not beat the baseline on churn recall.
- The chosen model is therefore **Logistic Regression** — a deliberate choice in favor of catching more churners over maximizing headline accuracy.

**Top features (consistent across models):**

| Rank | Feature |
|---:|---|
| 1 | `tenure` |
| 2 | `MonthlyCharges` |
| 3 | `TotalCharges` |
| 4 | `Contract` |
| 5 | `InternetService_Fiber optic` |
| 6 | `PaymentMethod_Electronic check` |
| 7 | `OnlineSecurity` |
| 8 | `TechSupport` |

The model's drivers match the EDA findings exactly — a good consistency check that the model learned real structure, not noise.

**Feature importance charts:**

| Logistic Regression | Random Forest / XGBoost |
|---|---|
| ![Feature Importance LR](screenshots/top_10_fitur_paling_berpengaruh_logistic_regression.png) | ![Feature Importance](screenshots/top_10_fitur_paling_berpengaruh.png) |

---

## ⚠️ Limitations & Assumptions

1. **No persisted metrics.** The notebook does not save the classification reports, so only the churn recall (~0.79) is documented. Precision and F1 per model are not available for verification.
2. **No cross-validation or tuning.** All three models use default or hand-picked hyperparameters with a single 80:20 split. Results may vary on a different split.
3. **Threshold left at 0.5.** For retention triage, a lower threshold would catch more churners — worth calibrating against the team's capacity for retention calls.
4. **`TotalCharges` is collinear with tenure.** It mostly encodes how long someone has stayed, so its importance partly duplicates `tenure`.
5. **Association, not causation.** Contract type correlates with churn, but customers who are already planning to leave may also self-select into flexible contracts. The recommendations below are hypotheses to test, not proven interventions.
6. **Dropped rows were not trivial.** The 11 dropped rows were all `tenure = 0` customers — a segment with structurally different behaviour. Dropping them is reasonable, but it removes the newest customers from the analysis.
7. **No temporal validation.** Trained and tested on the same time period, so stability over future cohorts is unverified.
8. **No revenue-weighted evaluation.** The model treats all churners as equally costly, though a $100/month customer is worth more than a $20/month one.

---

## 💡 Recommended Action Items

Ordered by **impact × ease of execution**.

### Priority 1 — Convert Month-to-Month Customers (highest impact)

- Offer a **first-year contract upgrade incentive** — the gap between 42.71% and 2.85% churn is worth far more than the discount costs.
- Target the offer at customers **past month 3** (survived onboarding) but **before month 12** (still at peak risk).
- Frame it around price certainty rather than a discount: "lock your rate for 12 months" avoids training customers to wait for promotions.

### Priority 2 — Fix the First-Year Experience

- Build a **structured onboarding program** for months 0–12, where 47.68% churn concentrates.
- Trigger a proactive check-in at **day 30, 90, and 180** — before dissatisfaction becomes a cancellation.
- Assign a retention owner per new customer, not a shared support queue.

### Priority 3 — Bundle Security and Support

- Offer **online security + tech support** as a discounted bundle to customers who have neither — this segment churns at 41.78% versus 14.64%.
- Position it as protection, not an upsell: customers who feel supported stay longer.
- Prioritize **fiber optic** customers (41.89% churn) who have no security add-ons.

### Priority 4 — Reduce Payment Friction

- Incentivize **automatic payment** enrollment. Electronic-check users churn at 45.29% versus 15.25% for credit-card auto-pay.
- A one-time bill credit for switching to autopay is cheap relative to losing the customer.
- Investigate why electronic check correlates with churn — it may proxy for lower engagement or price sensitivity.

### Priority 5 — Operate the Model

- Score the customer base **monthly** and route the top-risk list to the retention team.
- Pair each score with the **top contributing factors**, so the agent knows what to discuss.
- Start with the **1,128-customer high-risk segment** (63.48% churn) rather than the whole base.

### Suggested KPIs

| KPI | Baseline | Target (12 months) |
|---|---|---|
| Overall churn rate | 26.58% | < 22% |
| First-year churn (0–12 months) | 47.68% | < 38% |
| Month-to-month churn | 42.71% | < 34% |
| High-risk segment churn | 63.48% | < 50% |
| Monthly revenue churn | ~$139K | < $115K |

---

## 📈 Dashboard (Tableau)

**🔗 [Open the live dashboard](https://public.tableau.com/views/TelcoCustomerChurn_17797948116180/Dashboard1)**

The dashboard visualizes churn distribution, churn by customer segment, and the retention priority cohorts.

<details>
<summary>Embed code (Tableau Public)</summary>

<div class='tableauPlaceholder' id='viz1779794991638' style='position: relative'><noscript><a href='#'><img alt='Telco Customer Churn ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Te&#47;TelcoCustomerChurn_17797948116180&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TelcoCustomerChurn_17797948116180&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Te&#47;TelcoCustomerChurn_17797948116180&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>

</details>

---

## 🚀 Getting Started

### 1. Prerequisites

- Python **3.10+**

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib jupyter
```

### 3. Get the data

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root.

### 4. Run the analysis

```bash
jupyter notebook da01-telco-customer-churn.ipynb
```

The notebook path points to `/kaggle/input/...` — update the `read_csv` path to your local file location before running.

---

## 🗂️ Project Structure

```
telco-customer-churn-analysis/
│
├── da01-telco-customer-churn.ipynb    # EDA, modeling, evaluation
├── README.md
├── .gitignore
│
├── screenshots/                       # Analysis charts
│   ├── customer_churn_distribution.png
│   ├── chart_02.png
│   ├── chart_03.png
│   ├── korelasi_antar_kolom_numerik.png
│   ├── top_10_fitur_paling_berpengaruh.png
│   └── top_10_fitur_paling_berpengaruh_logistic_regression.png
│
└── .github/workflows/sync-kaggle.yml  # Kaggle sync automation
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data analysis | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Business intelligence | Tableau Public |
| Environment | Jupyter / Kaggle Notebook |

---

## 👤 Author

**Zulfahmi M. Ardianto**
Laskar ID: A548YBM523

*Project ini merupakan bagian dari portfolio Data Analyst.*
