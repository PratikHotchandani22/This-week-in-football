# This Week in Football 

Transform Reddit match‑week chatter into multilingual summaries, embeddings, and audio highlights.

---

## 📌 Project Overview

This Week in Football is an end‑to‑end **asynchronous Python** pipeline that

1. **Scrapes** selected football sub‑reddits for the previous ISO week.
2. **Translates** non‑English comments & titles with a local NMT model.
3. **Classifies** comments as human / bot noise and filters accordingly.
4. **Tags sentiment & emotion** using a compact Llama 3‑based model.
5. **Generates** per‑submission and week‑level natural‑language summaries.
6. **Creates vector embeddings** for comments, titles, and summaries.
7. **Persists** raw & enriched data to Supabase (PostgreSQL + storage).
8. **Optionally synthesises audio** via **MeloTTS** for social distribution.

All heavy LLM work is handled locally or via model endpoints exposed through **LangChain**, ensuring reproducibility and cost control.

---

## 🏗️ Architecture

```text
┌──────────┐   scrape   ┌────────────┐   translate   ┌────────────┐
│ Reddit 🦑 │──────────▶│ Raw CSV/JSON│──────────────▶│ Translated │
└──────────┘            └────────────┘               └────────────┘
       │                                          │
       │classify & tag                            │summarise & embed
       ▼                                          ▼
┌────────────┐    sentiment    ┌────────────┐   embeddings  ┌────────────┐
│Cleaned Data│───────────────▶│Summaries📝 │──────────────▶│Supabase ☁️│
└────────────┘                └────────────┘               └────────────┘
                                                 │
                                                 ▼
                                      ┌──────────────────┐
                                      │MeloTTS 🎧 (opt.) │
                                      └──────────────────┘
```

---

## 🛠️ Key Technologies

| Domain                   | Stack                                      |              |
| ------------------------ | ------------------------------------------ | ------------ |
| Async scraping           | **asyncpraw**, **asyncio**                 |              |
| Translation              | Local NMT model via **transformers**       |              |
| Classification & tagging | **LangChain** ⟷ Llama 3                    |  Qwen models |
| Vector embeddings        | **sentence‑transformers** (bge‑large)      |              |
| Data storage             | **Supabase** (PostgreSQL, storage buckets) |              |
| Audio                    | **MeloTTS**                                |              |

---

## 📂 Directory Structure

```text
.
├── main.py                # Orchestrates the weekly pipeline
├── configurations.py      # Central config & prompt templates
├── helper_functions.py    # Reusable processing helpers
├── scrapping_reddit.py    # Async Reddit client & scraper
├── generate_embeddings.py # Embedding utilities
├── supabase_backend.py    # Async Supabase wrapper
├── supabase_helper_functions.py
├── audio/                 # MeloTTS synthesis scripts
└── README.md              # (this file)
```

---

## ⚙️ Configuration

Create `.env` at project root with the following keys:

```bash
REDDIT_CLIENT_ID=xxxx
REDDIT_CLIENT_SECRET=xxxx
REDDIT_USER_AGENT=this_week_in_football
SUPABASE_URL=https://xyz.supabase.co
SUPABASE_ANON_KEY=public‑anon‑key
# Optional LLM / Hugging Face / OpenAI keys
HF_HOME=/models
```

Other tunables (sub‑reddit list, prompts, model paths, target language) live in `configurations.py`.

---

## 🚀 Quick Start

```bash
# 1. Install Python 3.10+ & poetry (or pip)
poetry install   # installs from pyproject.toml

# 2. Activate
poetry shell     # or `source venv/bin/activate`

# 3. Create .env (see above)

# 4. Run the weekly pipeline
python main.py   # runs asyncio event‑loop
```

Pipeline prints progress and drops intermediate CSVs for inspection.

### Audio Generation (optional)

```bash
python audio/generate_audio.py  # requires MeloTTS checkpoints
```

Audio files land in `audio/out/` and can be attached to social posts.

---

## 📝 Supabase Schema (simplified)

| Table                   | Purpose                    |
| ----------------------- | -------------------------- |
| `reddit_submissions`    | Metadata & translated text |
| `reddit_embeddings`     | Vectors for search / RAG   |
| `reddit_summary`        | Per‑submission summaries   |
| `reddit_weekly_summary` | Aggregated weekly insights |

---

## ✅ Testing

```bash
pytest -q
```

Unit tests cover helper utilities and prompt validation.

---

## 🤝 Contributing

1. Fork & clone.
2. Create feature branch: `git checkout -b feat/awesome`.
3. Commit with conventional messages.
4. Open PR; ensure CI (lint & tests) pass.

---

## 📄 License

MIT — see `LICENSE` for full text.

---

## 🙏 Acknowledgements

* Reddit API team for asyncpraw.
* Wang et al. for **MeloTTS**.
* The open‑source LLM community for continual model improvements.
