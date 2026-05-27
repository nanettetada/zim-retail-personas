<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:8E44AD,100:4A2065&height=200&section=header&text=Zim%20Retail%20Personas&fontSize=48&fontColor=ffffff&fontAlignY=40&animation=fadeIn" />

<a href="https://github.com/nanettetada">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&duration=3500&pause=800&color=8E44AD&center=true&vCenter=true&width=620&lines=Find+the+12%25+that+drives+the+revenue;K-Means+on+RFM+features;Five+actionable+personas" />
</a>

<p>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/K--Means-8E44AD?style=for-the-badge" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</p>

<a href="https://huggingface.co/spaces/NanetteTada/zim-retail-personas"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Open%20Live%20Demo-FFD21E?style=for-the-badge" /></a>

</div>

---

## :dart: Why I built this

K-Means is one of the first algorithms you learn, but using it *well* on data that looks like a real retail dataset is its own skill. I built this around a **Zimbabwean retailer** — think TM Pick n Pay, OK Zimbabwe, or Edgars — where understanding the difference between a loyal Borrowdale shopper and a price-sensitive Kambuzuma shopper is the kind of insight a marketing team actually pays for.

## :sparkles: At a glance

|  |  |
|---|---|
| **Problem** | Cluster Zim retail customers into actionable personas |
| **Features** | Recency · Frequency · Monetary (RFM) |
| **Method** | K-Means with optimal k chosen by elbow + silhouette |
| **Result** | 5 named personas; the top 12% of customers drive most of the revenue |
| **Stack** | scikit-learn · pandas · Streamlit · Plotly |

## :wrench: How I approached it

1. **Generated a synthetic transactions dataset** for 5,000 customers with realistic RFM patterns (with hidden personas underneath so I could check if K-Means recovered them).
2. **Engineered RFM features** — Recency (days since last order), Frequency (orders), Monetary (total spend).
3. **Scaled the features** — K-Means is distance-based.
4. **Picked k** using both the elbow method *and* silhouette score, then chose a business-sensible value.
5. **Fit K-Means** and projected the clusters into 2D with PCA to sanity-check.
6. **Profiled each cluster** and gave it a name a non-technical stakeholder would understand.

## :bar_chart: The personas

| Cluster | Persona | Share | Avg recency | Avg frequency | Avg monetary |
|---|---|---|---|---|---|
| 0 | **Loyal high-value** | 12% | 8 days | 24 orders | $4,800 |
| 1 | Regulars | 31% | 22 days | 9 orders | $1,400 |
| 2 | One-time buyers | 27% | 95 days | 1 order | $180 |
| 3 | At-risk / lapsed | 18% | 180 days | 4 orders | $620 |
| 4 | New customers | 12% | 6 days | 1 order | $90 |

## :computer: Run it yourself

```bash
pip install -r requirements.txt
jupyter notebook customer_segmentation.ipynb
streamlit run dashboard.py
```

## :tv: Interactive dashboard

Three tabs:
- **Overview** — customers per persona, revenue share, PCA scatter of every customer.
- **Persona explorer** — pick a persona, see customer count, lifetime value, and the RFM scatter.
- **Suggested actions** — a concrete play for each persona (VIP perks, win-back, onboarding nudge).

Move the **k slider** in the sidebar and watch every chart redraw live.

## :rocket: What I'd do next

- Add demographics and product-category features for a richer view.
- A/B test win-back offer sizes on the "lapsed" segment.
- Refresh the clusters on a monthly schedule from the data warehouse.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:8E44AD,100:4A2065&height=100&section=footer" />

Built by <b>Tadaishe Maumbe</b> · <a href="https://github.com/nanettetada">@nanettetada</a> · <a href="mailto:maumbetadaishe@gmail.com">email</a>

</div>
