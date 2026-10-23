---
title: TripMate AI
emoji: 🧭
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
license: cc-by-sa-4.0
short_description: Indian trip planner — RAG plus a fine-tuned 1.5B model
---

# TripMate AI

A conversational travel planner for domestic India. Facts come from retrieval over
Wikivoyage (hybrid dense + BM25, reranked); budgets, routes and checks are plain code; a
LoRA fine-tuned Qwen2.5-1.5B handles persona, pushback and warnings.

Runs on a free CPU: expect 20–60 seconds per answer, one user at a time.

Place data from [Wikivoyage](https://en.wikivoyage.org) (CC BY-SA 4.0); seasonal and safety
notes partly from the Kaggle Indian Tourism Dataset (Apache-2.0, unverified).
