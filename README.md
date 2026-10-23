# TripMate-AI

TripMate is a conversational travel agent for Indian travelers, built as a hands-on AI
engineering project: **RAG for facts, a fine-tuned small model for behavior, and plain code
for arithmetic and hard constraints.** Scope is PRD phases 1 and 2 (`PRD/V1.0.md`).

| Concern | Owner |
|---|---|
| Places, hours, fees, safety notes | Retrieval over Wikivoyage (hybrid dense + BM25, RRF, rerank) |
| Budget totals, time overlaps, pace caps, routes | Deterministic code (`agent/budget.py`, `validator.py`, `route.py`) |
| Trip memory across a long chat | `TripState`, re-rendered into every prompt |
| Persona, planning judgment, proactive warnings | The model (Qwen2.5-1.5B, to be LoRA fine-tuned) |

## Setup

```bash
uv sync
uv run tripmate ingest      # fetch Wikivoyage pages -> data/kb/*.jsonl (cached in data/raw)
                            # also builds destination profiles from "RAG data/" if present
uv run tripmate index       # embed all chunks; vectors are saved under data/index

ollama serve                # in another terminal
ollama pull qwen2.5:1.5b
ollama create tripmate-base -f finetune/Modelfile.base
```

## Use

The local setup runs the fine-tuned model (v2, see the eval below):

```bash
ollama create tripmate-ft -f finetune/Modelfile.v2     # from finetune/tripmate-v2-q4_k_m.gguf
echo "TRIPMATE_LLM_MODEL=tripmate-ft" > .env
```

Deployment to a Hugging Face Space is prepared (`Dockerfile`, `deploy/`) but not published.

```bash
uv run streamlit run app/streamlit_app.py

# or from the terminal
uv run tripmate plan --dest Goa --days 3 --travelers 4 --party family --budget 60000 \
    --pace relaxed --interests "history,beaches" --constraints "vegetarian" \
    --say "Make day 2 lighter" --what-if-budget -20
uv run tripmate plan --dest "Goa,Gokarna" --days 5 --budget 50000     # multi-city
uv run tripmate search Goa "evening boat ride"                        # inspect retrieval
uv run tripmate eval-retrieval                                        # dense vs bm25 vs hybrid
uv run pytest
```

## Fine-tuning data

```bash
ollama create tripmate-teacher -f finetune/Modelfile.teacher
uv run tripmate seeds                 # real traveler questions from the HF dataset (wording only)
uv run tripmate datagen -n 150        # teacher plays TripMate; resumable, about 6 min per scenario
uv run tripmate datagen -n 60 --focus conflict   # extra pushback data: conflicting requests,
                                                 # over-budget asks, infeasible trips
uv run tripmate split                 # finetune/data/{train,valid,test}.jsonl (Udaipur, Munnar held out)
uv sync --extra finetune && uv run mlx_lm.lora --config finetune/lora.yaml
finetune/export.sh                    # fuse -> Ollama tripmate-ft (Q4_K_M) -> GGUF for deploy
```

## Behavior eval

Scripted conversations on the held-out cities (Udaipur, Munnar), scored with the same checks
that filtered the training data. Repair is off, so this measures first-pass behavior.

```bash
uv run tripmate eval-behavior --name base-short                      # base, short prompt
uv run tripmate eval-behavior --name base-long --style long          # base, full rule prompt
uv run tripmate eval-behavior --name base-short-free --unconstrained # no JSON grammar
uv run tripmate eval-behavior --name ft-short --model tripmate-ft
uv run tripmate eval-compare base-short base-long ft-short
```

Results (16 conversations, 82–87 scored turns each; Qwen2.5-1.5B Q4_K_M, temperature 0.3).
Each fine-tune uses its lowest-validation-loss checkpoint.

| Metric | Base, full rule prompt | v2 | **v2 + repair** (shipped) | v3 | v3 + repair |
|---|---|---|---|---|---|
| All turns pass every check | 30% | 67% | **75%** | 62% | 69% |
| First plan passes validator | 36% | 79% | **79%** | 50% | 79% |
| Infeasible trip declined | 80% | 100% | **100%** | 100% | 100% |
| Pushback on constraint clash | 40% | 80% | **80%** | 0% | 0% |
| Pushback on over-budget ask | 73% | 100% | **100%** | 100% | 100% |
| Edit requests change the plan | 13% | 4% | **70%** | 87% | 74% |
| Reply never claims an edit it did not make | 92% | 72% | **100%** | 97% | 94% |
| Late turns (3+) pass | 31% | 55% | **60%** | 55% | 60% |
| Median latency per call | 7.1 s | 6.0 s | **8.5 s** | 5.8 s | 7.6 s |

The base model with the short prompt passes 0% of turns. What each version changed:

- **v1** (plan before reply in the JSON) learned "follow-up turn -> `plan: null`": it never
  made a requested edit, yet said it had. An early version of this eval missed that and
  reported 83%; rescored it is 55%.
- **v2** puts `reply` first and drops samples that claim edits they did not make. Edits still
  rarely happen on the first try, so the agent now treats "I added X" with no plan as a
  failed check, and the repair turn must contain a plan (`agent/orchestrator.py`).
- **v3** doubles the edit samples in training. First-try edits jump to 87%, but pushback on
  constraint clashes collapses: it learned "follow-up -> edit". Kept as `tripmate-v3`.

Samples are small (5 constraint clashes, 5 infeasible trips, 14 first plans), and a rerun of
the same baseline moved from 28% to 50% overall, so read single-digit gaps as noise. Replies
written on a repair turn are weaker (an occasional invented rupee figure).

Settings are `TRIPMATE_*` environment variables (see `src/tripmate/config.py`), for example
`TRIPMATE_LLM_MODEL` or `TRIPMATE_TEACHER_BASE_URL`.

## Layout

```
src/tripmate/
  kb/wikivoyage.py     fetch + parse listings into structured places and guide chunks
  rag/                 bm25.py (from scratch), retriever.py (Qdrant, RRF, cross-encoder rerank)
  agent/               orchestrator, prompts, validator, budget, route (multi-city), whatif
  state.py             TripState, plan merge and diff, experience score
  profile.py           saved travel preferences
  llm/client.py        one OpenAI-compatible client for Ollama / llama-server / APIs
  eval/                retrieval ablation
app/streamlit_app.py   UI
data/destinations.yaml which cities are in the knowledge base
```

## Data and attribution

Place and guide text comes from [Wikivoyage](https://en.wikivoyage.org), licensed
CC BY-SA 4.0. Each place keeps its source URL and last-edit date, shown in the UI. Many
listings have no hours or price; the app shows "unknown" rather than guessing. Intercity fares
are distance-based estimates, not quotes.

Seasons, safety, customs and typical price ranges per destination come from the Kaggle
[Indian Tourism Dataset](https://www.kaggle.com/datasets/sushanthnaidu24/indian-tourism-dataset)
(Apache-2.0). Its sources cannot be traced, so those notes are labeled unverified. Simulated
user turns borrow question wording from the Hugging Face dataset
`soniawmeyer/travel-conversations-finetuning`; its answers are not used.
