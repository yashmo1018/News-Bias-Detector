# News Bias Detector

A portfolio-grade Python pipeline that scrapes news from 15+ global outlets, uses an LLM to analyze political bias, sentiment, and framing per article, clusters related articles by real-world event, and displays everything on an interactive Streamlit dashboard.

## Live Demo
[**→ View on Streamlit Cloud**](https://your-app.streamlit.app) <!-- replace after deploying -->

![Dashboard Preview](preview.png) <!-- optional: add a screenshot -->

---

## What It Does

- **Scrapes** RSS feeds from 15+ outlets across the political spectrum and globe
- **Analyzes** each article using LLaMA 3.3 70B via Groq API — classifies bias (Left → Right), sentiment (−1 to +1), framing type, and entity portrayal
- **Clusters** articles covering the same real-world event using sentence embeddings + DBSCAN
- **Visualizes** coverage gaps, entity portrayal heatmaps, outlet bias radar, and sentiment distribution

---

## Pipeline

```
scraper.py → articles.json
analyzer.py → analyzed.json
clusterer.py → clustered.json
streamlit run app.py
```

| Layer | Tool | Purpose |
|-------|------|---------|
| Scraper | `feedparser` + `newspaper4k` | RSS → full article text |
| Analyzer | Groq API · LLaMA 3.3 70B | Bias, sentiment, framing per article |
| Clusterer | `sentence-transformers` + DBSCAN | Group articles by event |
| Dashboard | Streamlit + Plotly | Interactive frontend |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — free tier is sufficient.

### 3. Set your API key

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY = "gsk_..."
```

**Mac/Linux:**
```bash
export GROQ_API_KEY="gsk_..."
```

### 4. Run the pipeline
```bash
python scraper.py
python analyzer.py
python clusterer.py
streamlit run app.py
```

---

## Dashboard Tabs

| Tab | What you see |
|-----|-------------|
| Coverage Gaps | Which outlets covered which events, bias per outlet per story, polarization score |
| Entity Heatmap | How each outlet portrays the same people and organizations |
| Bias Radar | Per-outlet bias "personality" across the Left→Right spectrum |
| Article Browser | Searchable, sortable list of all articles with full analysis |

---

## Auto-Refresh

The dashboard checks how old `articles.json` is on every load. If it's older than 6 hours, the full pipeline (scraper → analyzer → clusterer) runs automatically in the background. A manual **Refresh now** button is also available in the sidebar.

Articles older than 7 days are automatically pruned to maintain a rolling window.

---

## Cluster Tuning

```bash
python clusterer.py --eps 0.30      # tighter clusters
python clusterer.py --eps 0.45      # looser clusters
python clusterer.py --min-samples 1 # no noise points
```

---

## Sources Monitored

| Outlet | Region | Lean |
|--------|--------|------|
| BBC News | UK | Center |
| Reuters | Global | Center |
| Associated Press | Global | Center |
| The Guardian | UK | Left-Center |
| Al Jazeera | Qatar | Left-Center |
| Deutsche Welle | Germany | Center |
| France 24 | France | Center |
| ABC Australia | Australia | Center |
| The Hindu | India | Left-Center |
| NDTV | India | Center |
| Indian Express | India | Center |
| Times of India | India | Center |
| Hindustan Times | India | Center |
| South China Morning Post | Hong Kong | Center |
| Arab News | Saudi Arabia | Right-Center |

---

## Tech Stack

`Python` · `feedparser` · `newspaper4k` · `Groq API` · `LLaMA 3.3 70B` · `sentence-transformers` · `scikit-learn` · `Streamlit` · `Plotly` · `pandas`

---

## Roadmap

- [ ] Sensationalism score
- [ ] Bias over time chart
- [ ] Word cloud per outlet
- [ ] Outlet comparison mode
- [ ] Export PDF report

---

*Built by Note · RIT Sangli · 2025–26*
