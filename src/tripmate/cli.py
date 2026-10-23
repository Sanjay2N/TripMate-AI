import argparse

from tripmate.config import get_settings


def cmd_ingest(args):
    from tripmate.kb.wikivoyage import ingest

    s = get_settings()
    print(ingest(s.data_dir / "destinations.yaml", s.kb_dir))
    cmd_profiles(args)


def cmd_profiles(args):
    from tripmate.kb.profiles import build_profiles

    s = get_settings()
    if s.profiles_file.exists():
        build_profiles(s.profiles_file, s.data_dir / "destinations.yaml", s.kb_dir)
    else:
        print(f"no destination profiles: {s.profiles_file} not found")


def cmd_index(args):
    from tripmate.rag.retriever import build_index

    build_index()


def cmd_search(args):
    from tripmate.rag.retriever import Retriever

    hits = Retriever().search(
        args.query,
        args.destination,
        doc_type=args.type,
        k=args.k,
        mode=args.mode,
        rerank=not args.no_rerank,
    )
    for h in hits:
        d = h.doc
        label = d.get("name") or d.get("section")
        print(f"{h.score:8.4f}  [{d.get('kind', 'guide')}] {label} ({d['area']})")
        print(f"          {d['text'][:140]}")


def print_state(agent, state, result):
    from tripmate.state import experience_score

    if not result.parsed:
        print("!! a model response did not parse:\n", result.raw[-1][:500])
    print(f"\nTripMate: {result.reply}")
    if result.ask:
        print(f"  (asks: {result.ask})")
    if state.plan:
        for d in state.plan.days:
            print(f"\n  Day {d.day} [{state.day_city(d.day)}] - {d.theme}")
            for a in d.activities:
                name = state.pois[a.poi]["name"] if a.poi in state.pois else f"?{a.poi}"
                print(
                    f"    {a.start}  {name} ({a.duration_min} min, Rs {a.cost_inr}/person) - {a.why}"
                )
            print(
                f"    drop if short: {d.drop_if_short or '-'} | add if extra: {d.add_if_extra or '-'}"
            )
        for city, c in state.plan.cities.items():
            stay = f"{c.stay.area} Rs {c.stay.per_night_inr}/night" if c.stay else "-"
            print(
                f"\n  {city}: stay {stay} | food Rs {c.food_per_day_inr}/person/day | "
                f"local Rs {c.local_transport_per_day_inr}/day"
            )
        print(f"  Budget: {agent.budget(state).model_dump()}")
        print(f"  Experience score: {experience_score(state)}")
    for h in state.heads_up:
        print(f"  Heads-up [{h.source}]: {h.risk} -> {h.mitigation}")
    if result.changes:
        print("  What changed: " + " | ".join(result.changes))
    print(f"  constraints now: {state.request.constraints}")
    print(
        f"  [attempts={result.attempts} applied={result.applied} {result.seconds:.1f}s "
        f"{result.completion_tokens} tok] first-pass violations: {result.first_pass_violations}"
    )
    if result.violations:
        print(f"  REMAINING VIOLATIONS: {result.violations}")


def cmd_plan(args):
    from tripmate.agent.orchestrator import Agent
    from tripmate.agent.whatif import what_if
    from tripmate.llm.client import serving_llm, teacher_llm
    from tripmate.rag.retriever import Retriever
    from tripmate.schemas import Pace, TripRequest

    s = get_settings()
    agent = Agent(
        Retriever(),
        teacher_llm(s) if args.teacher else serving_llm(s),
        style=args.style,
        repair=not args.no_repair,
    )
    request = TripRequest(
        source=args.source,
        destinations=[d.strip() for d in args.dest.split(",") if d.strip()],
        days=args.days,
        travelers=args.travelers,
        party=args.party,
        mode=args.mode,
        budget_inr=args.budget,
        tier=args.tier,
        pace=args.pace,
        interests=[i.strip() for i in args.interests.split(",") if i.strip()],
        constraints=[c.strip() for c in args.constraints.split(";") if c.strip()],
    )
    state = agent.start(request)
    print("Route: " + " -> ".join(f"{seg.city} (days {seg.days})" for seg in state.route))
    print_state(agent, state, agent.turn(state))
    for message in args.say or []:
        print(f"\nYou: {message}")
        print_state(agent, state, agent.turn(state, message))
    if args.what_if_budget or args.what_if_days or args.what_if_pace:
        sim = what_if(
            agent,
            state,
            args.what_if_budget,
            args.what_if_days,
            Pace(args.what_if_pace) if args.what_if_pace else None,
        )
        print(f"\nWhat-if (model used: {sim.used_model}): {sim.summary}")
        print("  changes: " + (" | ".join(sim.changes) or "none"))
        print(f"  budget: {agent.budget(sim.state).model_dump()}")


def cmd_seeds(args):
    from tripmate.datagen.seeds import fetch_questions

    fetch_questions(get_settings().data_dir / "seeds" / "questions.txt", pages=args.pages)


def cmd_datagen(args):
    from tripmate.agent.orchestrator import Agent
    from tripmate.datagen.generate import generate
    from tripmate.llm.client import teacher_llm
    from tripmate.rag.retriever import Retriever

    s = get_settings()
    agent = Agent(Retriever(), teacher_llm(s), style="long")
    total = generate(agent, s.finetune_dir / "data" / "all.jsonl", args.n, args.seed, args.focus)
    print("\n".join(f"  {k}: {v}" for k, v in sorted(total.items())))


def cmd_split(args):
    from tripmate.datagen.generate import split

    data = get_settings().finetune_dir / "data"
    held_out = {d.strip() for d in args.test_destinations.split(",")}
    print(split(data / "all.jsonl", data, held_out, edit_weight=args.edit_weight))


def cmd_eval_retrieval(args):
    from tripmate.eval.retrieval import evaluate, format_table

    gold = get_settings().data_dir / "eval" / "retrieval_gold.jsonl"
    print(format_table(evaluate(gold, k=args.k)))


def cmd_eval_behavior(args):
    from tripmate.agent.orchestrator import Agent
    from tripmate.datagen.seeds import load_questions
    from tripmate.eval.behavior import compare, eval_scenarios, load_rows, run_eval
    from tripmate.llm.client import LLM
    from tripmate.rag.retriever import Retriever

    s = get_settings()
    llm = LLM(args.base_url or s.llm_base_url, args.model or s.llm_model, s.llm_api_key)
    agent = Agent(
        Retriever(), llm, style=args.style, repair=args.repair, constrained=not args.unconstrained
    )
    cities = [c.strip() for c in args.cities.split(",")]
    scenarios = eval_scenarios(args.n, agent.places, cities, follow_ups=args.follow_ups)
    questions = load_questions(s.data_dir / "seeds" / "questions.txt")
    out = s.finetune_dir / "eval" / f"{args.name}.jsonl"
    run_eval(agent, scenarios, questions, out)
    print(compare({args.name: load_rows(out)}))


def cmd_eval_compare(args):
    from tripmate.eval.behavior import compare, load_rows

    folder = get_settings().finetune_dir / "eval"
    print(compare({n: load_rows(folder / f"{n}.jsonl") for n in args.names}))


def main():
    parser = argparse.ArgumentParser(prog="tripmate")
    sub = parser.add_subparsers(required=True)

    sub.add_parser("ingest", help="fetch Wikivoyage pages into data/kb").set_defaults(
        func=cmd_ingest
    )
    sub.add_parser(
        "profiles", help="rebuild destination profiles from the Kaggle dataset only"
    ).set_defaults(func=cmd_profiles)
    sub.add_parser("index", help="embed data/kb and build the vector index").set_defaults(
        func=cmd_index
    )

    p = sub.add_parser("search", help="query the knowledge base")
    p.add_argument("destination")
    p.add_argument("query")
    p.add_argument("--type", default="poi", choices=["poi", "guide"])
    p.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("-k", type=int, default=8)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("plan", help="plan a trip, then optionally send follow-up messages")
    p.add_argument("--source", default="Bengaluru")
    p.add_argument("--dest", default="Goa", help="one city, or several separated by commas")
    p.add_argument("--days", type=int, default=4)
    p.add_argument("--travelers", type=int, default=2)
    p.add_argument("--party", default="couple")
    p.add_argument("--mode", default="train")
    p.add_argument("--budget", type=int, default=40000)
    p.add_argument("--tier", default="balanced")
    p.add_argument("--pace", default="moderate")
    p.add_argument("--interests", default="")
    p.add_argument("--constraints", default="", help="semicolon-separated")
    p.add_argument("--say", action="append", help="follow-up message (repeatable)")
    p.add_argument("--what-if-budget", type=float, default=0, help="percent change, e.g. -20")
    p.add_argument("--what-if-days", type=int, default=0, help="days to add or remove")
    p.add_argument("--what-if-pace", default="", help="relaxed | moderate | packed")
    p.add_argument("--teacher", action="store_true", help="use the teacher model instead")
    p.add_argument("--style", default="long", choices=["long", "short"])
    p.add_argument("--no-repair", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("seeds", help="fetch real traveler questions from the HF dataset")
    p.add_argument("--pages", type=int, default=40, help="pages of 100 rows to sample")
    p.set_defaults(func=cmd_seeds)

    p = sub.add_parser("datagen", help="generate training conversations with the teacher")
    p.add_argument("-n", type=int, default=200, help="number of scenarios (resumable)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--focus",
        choices=["conflict"],
        help="only scenarios that test pushback (conflicting requests, infeasible budgets)",
    )
    p.set_defaults(func=cmd_datagen)

    p = sub.add_parser("split", help="split generated data into train/valid/test")
    p.add_argument("--test-destinations", default="Udaipur,Munnar")
    p.add_argument(
        "--edit-weight", type=int, default=1, help="copies of each plan-edit sample in train"
    )
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("eval-retrieval", help="compare retrieval modes on the gold set")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_eval_retrieval)

    p = sub.add_parser("eval-behavior", help="score a model on scripted held-out conversations")
    p.add_argument("--name", required=True, help="run name, e.g. base-short")
    p.add_argument("--model", help="model name at the endpoint (default TRIPMATE_LLM_MODEL)")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint (default TRIPMATE_LLM_BASE_URL)")
    p.add_argument("--style", choices=["short", "long"], default="short")
    p.add_argument("--unconstrained", action="store_true", help="no JSON grammar")
    p.add_argument("--repair", action="store_true", help="allow one repair turn, as the app does")
    p.add_argument("--cities", default="Udaipur,Munnar", help="held-out destinations")
    p.add_argument("-n", type=int, default=16, help="number of conversations")
    p.add_argument("--follow-ups", type=int, default=5)
    p.set_defaults(func=cmd_eval_behavior)

    p = sub.add_parser("eval-compare", help="side-by-side table of earlier eval-behavior runs")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=cmd_eval_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
