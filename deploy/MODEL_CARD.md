---
base_model: Qwen/Qwen2.5-1.5B-Instruct
license: apache-2.0
tags: [gguf, lora, travel, india, qwen2]
---

# TripMate 1.5B (GGUF, Q4_K_M)

Qwen2.5-1.5B-Instruct with a fused LoRA adapter, trained with MLX on 348 synthetic
conversation turns from a local teacher (qwen3.5:9b) playing a travel agent, filtered by a
deterministic validator. It is tuned for **behaviour, not knowledge**: a consistent persona,
pushback when a request breaks a stated constraint or the budget, grounded heads-ups, and
a JSON response envelope. Facts come from retrieval at inference time.

It expects the TripMate prompt format (trip summary, candidate places with ids, guide notes)
and is not a general chat model.

| Held-out eval (Udaipur, Munnar) | Base, full prompt | This model, short prompt |
|---|---|---|
| All turns pass every check | 50% | 83% |
| Pushback on constraint clash | 40% | 100% |
| Pushback on over-budget ask | 73% | 100% |
| First plan passes validator | 36% | 43% |

Training: LoRA rank 16 on the top 16 layers, lr 2e-4, checkpoint 750 of 1050 batches
(lowest validation loss). Chat format: ChatML.
